// Optiland Canvas — M2 web shell ("it's optics").
//
// Scene lives in Rust (§2). This file holds UI state only. Two tracers, per the
// spec's tiering (§4.1): the interactive f32 path draws the picture every frame
// (spectral colour, ghost layer, TIR glow); the f64 reference feeds the
// detector dock's *numbers*, on idle.

import init, { Canvas } from "../pkg/oc_wasm.js";
import { initM3 } from "./m3";
import { PLAIN_RAYS, RayLayer, type RayDrawOpts, type RayLayout } from "./rays";
import { initSlice3d, type Slice3d } from "./slice3d";
import { initView3dThree, type FrameReport, type View3dThree } from "./render3d/view3d";
import { createFastPreview } from "./fastpreview";
import { initInset } from "./inset/inset";
import { createComponentView, type ComponentView } from "./inset/component-view";
import { initOpenUc2 } from "./features/openuc2/OpenUc2Page";
import { drawFacePixels, faceBox, facePath, formatFlux, type DetectorFace, type DetectorReadout } from "./detector-face";
import { drawPolarMap } from "./polarmap";
import { initGallery } from "./gallery";
import { createReference, type ReferenceBundle } from "./reference";
import { initSunPanel } from "./sun-panel";

const SCALE = 6.5; // px per mm
type V2 = { x: number; y: number };
type Glass = { name: string; nd: number; vd: number };

async function main() {
  await init();

  // Switching modes rebuilds the page. Each mode wires its own listeners to the
  // one shared canvas, so tearing the other one down cleanly is more trouble
  // than a reload is worth — and without this, typing the route into the address
  // bar changes the hash and does nothing visible.
  window.addEventListener("hashchange", () => location.reload());

  // The openUC2 mode is a route, not a fork: it shares this page's canvas and
  // the same RayLayer. The freeform Canvas below is untouched and stays the
  // default (§15.1).
  if (location.hash.startsWith("#openuc2")) {
    await initOpenUc2();
    return;
  }

  const view = document.getElementById("view") as HTMLCanvasElement;
  const ctx = view.getContext("2d")!;
  const stat = document.getElementById("stat")!;
  const abbe = document.getElementById("abbe") as HTMLCanvasElement;
  const actx = abbe.getContext("2d")!;
  const abbeLabel = document.getElementById("abbeLabel")!;
  const hist = document.getElementById("hist") as HTMLCanvasElement;
  const hctx = hist.getContext("2d")!;
  const dockNums = document.getElementById("dockNums")!;
  const ghostsBox = document.getElementById("ghosts") as HTMLInputElement;
  const ghostOpacity = document.getElementById("ghostOpacity") as HTMLInputElement;
  const tirBox = document.getElementById("tir") as HTMLInputElement;
  const density = document.getElementById("density") as HTMLInputElement;
  const ghostBounces = document.getElementById("ghostBounces") as HTMLInputElement;

  // --- scene -------------------------------------------------------------
  const scene = new Canvas();
  // The interactive 3D picture: WebGPU when the browser has a device, the CPU
  // f32 mirror until then and otherwise. Created before anything can render,
  // because the status line reads it.
  const fast = createFastPreview();
  // Ghost reflection-generation cap G (§4.2): the only bounce knob. The primary
  // path through the elements is auto-sized to the scene, so element count is
  // unlimited; this just bounds the stray-light explosion. 2 catches the classic
  // two-surface ghost image.
  scene.setMaxGhostBounces(Number(ghostBounces.value) || 2);
  // Stand-in pictures the workers have delivered (for `?debug`).
  let standIns = 0;
  // The f64 reference, traced in a worker (reference.ts). When a trace lands,
  // the view showing the scene takes its picture and the panels its numbers;
  // while one runs, the status line says how far it has got.
  const reference = createReference(
    scene,
    () => ({ revision: scene.sceneRevision(), ghosts: scene.ghostsEnabled(), maxGhost: scene.maxGhostBounces() }),
    (why) => {
      if (why === "result" && slice3d?.active()) {
        refreshActive3d();
        renderView();
        refresh3dDetector();
        // A load, an undo or an edit elsewhere may have moved the sun.
        sunPanel.sync();
      } else if (why === "standIn" && slice3d?.active()) {
        standIns++;
        refreshActive3d();
        renderView();
      } else {
        writeReferenceStatus();
      }
    },
    () => fast.costly(scene),
    (ghosts, ms) => fast.noteCpuCost(ghosts, ms),
  );
  // A frame the page cannot afford (the GPU declined a ghost-lit lens, or a
  // drag on the CPU mirror) is traced by the reference's workers instead.
  fast.whenCostly(() => reference.requestStandIn());
  const committed: Glass = { name: "N-BK7", nd: 1.5168, vd: 64.17 };
  const lensId = scene.addSinglet(0, 0, 50, 4, 12, committed.nd, committed.vd);
  // Spectral count only affects dispersion smoothness (in the collimated beam
  // all wavelengths are coincident → they sum to white regardless of count), so
  // keep it modest for a cheap trace; spatial density is what fills the beam.
  scene.addWhiteBeam(-70, 0, 0, 10, 16, 9); // 16 rays × 9 spectral lines
  scene.addDetector(52, -7, 52, 7, 48);
  const lens: V2 = { x: 0, y: 0 };

  // M3 module (draw / inspect / snap / pin). It owns element manipulation on the
  // canvas; assigned near the end once renderView/refreshReference exist.
  let m3: { overlay: (ctx: CanvasRenderingContext2D) => void; sync: () => void; getSelected: () => number } | null = null;
  // The canonical 3D scene, drawn two ways: the meridional cross-section (slice3d)
  // and an orbitable 3D model on the GPU (render3d/). A toggle picks which owns
  // the canvas. `?render3d=webgl` forces the WebGL2 backend and `?render3d=off`
  // simulates a browser with no usable 3D renderer.
  let slice3d: Slice3d | null = null;
  let viewport3d: View3dThree | null = null;
  // The component view of the selection, in an inset window (web/src/inset).
  let componentView: ComponentView | null = null;
  const params = new URLSearchParams(location.search);
  type ViewMode = "2d" | "3d";
  let viewMode: ViewMode = "2d";
  const in3dModel = () => viewMode === "3d";
  // Re-trace the f64 3D path for whichever view is currently showing.
  const refreshActive3d = () => { if (in3dModel()) viewport3d?.refresh(); else slice3d?.refresh(); };
  // The body the material picker targets: the active view's selection.
  const selectedBody3d = () => (in3dModel() ? viewport3d?.selectedBody() : slice3d?.selectedBody()) ?? -1;

  // Layer cache the m3 overlay reads each frame (avoids re-parsing layer JSON per
  // element); refreshed whenever a layer changes. Also feeds onion-skinning.
  type LayerInfo = { id: number; visible: boolean; opacity: number };
  const layerCache: LayerInfo[] = JSON.parse(scene.layersJson());

  // Snapshot the settled v1 scene into the undo tree (§5.2). Deduped in Rust.
  // Only the (hidden) v1 legacy paths reach this in the always-3D app.
  function autoCommit(label: string) {
    scene.commit(label);
  }

  const glasses: Glass[] = JSON.parse(scene.abbeGlassesJson());
  const hull = Array.from(scene.abbeHull()); // [vd, nd, ...]

  // --- world <-> screen (zoom + pan) -------------------------------------
  const w = () => window.innerWidth;
  const h = () => window.innerHeight;
  let scale = SCALE;        // px per mm (zoom)
  let panX = 0, panY = 0;   // screen-px offset of the world origin from centre
  const toScreen = (p: V2): V2 => ({ x: w() / 2 + panX + p.x * scale, y: h() / 2 + panY - p.y * scale });
  const toWorld = (px: number, py: number): V2 => ({ x: (px - w() / 2 - panX) / scale, y: (h() / 2 + panY - py) / scale });
  // `toScreen` for the view as it is now, for rasterizing millions of points:
  // the same arithmetic in the same order, without reading the window's size
  // twice per point.
  const screenMap = (): ((p: V2) => V2) => {
    const cx = w() / 2 + panX, cy = h() / 2 + panY, k = scale;
    return (p) => ({ x: cx + p.x * k, y: cy - p.y * k });
  };

  // Zoom about a screen point, keeping the world point under it fixed.
  function zoomAt(clientX: number, clientY: number, factor: number) {
    const before = toWorld(clientX, clientY);
    scale = Math.min(400, Math.max(0.4, scale * factor));
    panX = clientX - w() / 2 - before.x * scale;
    panY = clientY - h() / 2 + before.y * scale;
  }
  const panBy = (dxScreen: number, dyScreen: number) => { panX += dxScreen; panY += dyScreen; };
  const resetView = () => { scale = SCALE; panX = 0; panY = 0; };

  // Frame the 2D section on the loaded scene, its sources included, in the
  // part of the window the side panels leave clear.
  function fitView() {
    const e = slice3d?.extent();
    if (!e) return;
    const [x0, y0, x1, y1] = e;
    const left = 270, right = 280, top = 64, bottom = 32;
    const aw = Math.max(160, w() - left - right);
    const ah = Math.max(120, h() - top - bottom);
    const dx = Math.max(x1 - x0, 1e-6), dy = Math.max(y1 - y0, 1e-6);
    scale = Math.min(400, Math.max(0.4, 0.9 * Math.min(aw / dx, ah / dy)));
    panX = left + aw / 2 - w() / 2 - ((x0 + x1) / 2) * scale;
    panY = top + ah / 2 - h() / 2 + ((y0 + y1) / 2) * scale;
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    view.width = Math.floor(window.innerWidth * dpr);
    view.height = Math.floor(window.innerHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", () => { resize(); renderView(); });
  resize();

  // --- HDR ray layer (spectral, additive, tonemapped) -------------------
  // The scientifically correct model (spec §4.3): accumulate each spectral ray
  // additively into a *linear* HDR buffer (colours from oc-wasm are linear light
  // weighted by remaining flux), then tonemap on present. White light is the
  // sum of its wavelengths → a full spectrum adds up to white; where wavelengths
  // physically separate, you get coloured fringes; and Reinhard tonemapping maps
  // the huge dynamic range (a focus is enormously brighter than a stray ray)
  // into [0,1] without the blow-out that plagues un-tonemapped additive.
  const rays = new RayLayer();
  const EXPOSURE = 1.0;

  // Grid controls — shared by both views (drag snapping reads the same DOM).
  const gridBox = document.getElementById("gridBox") as HTMLInputElement;
  const gridSpacing = document.getElementById("gridSpacing") as HTMLInputElement;

  // The meridional grid + optical-axis reference for the 2D cross-section view.
  // View-agnostic (main owns it) so it shows regardless of which module draws the
  // elements — the always-3D app boots into 3D, and the grid must still be there.
  function draw2dGrid(W: number, H: number) {
    if (gridBox.checked) {
      const sp = Math.max(0.1, Number(gridSpacing.value) || 5);
      const tl = toWorld(0, 0), br = toWorld(W, H);
      const x0 = Math.floor(Math.min(tl.x, br.x) / sp) * sp, x1 = Math.ceil(Math.max(tl.x, br.x) / sp) * sp;
      const y0 = Math.floor(Math.min(tl.y, br.y) / sp) * sp, y1 = Math.ceil(Math.max(tl.y, br.y) / sp) * sp;
      if ((x1 - x0) / sp <= 600 && (y1 - y0) / sp <= 600) {
        ctx.lineWidth = 1;
        for (let gx = x0; gx <= x1 + 1e-9; gx += sp) {
          const s = toScreen({ x: gx, y: 0 });
          ctx.strokeStyle = Math.abs(gx) < 1e-9 ? "rgba(120,150,190,0.22)" : "rgba(120,150,190,0.07)";
          ctx.beginPath(); ctx.moveTo(s.x, 0); ctx.lineTo(s.x, H); ctx.stroke();
        }
        for (let gy = y0; gy <= y1 + 1e-9; gy += sp) {
          const s = toScreen({ x: 0, y: gy });
          ctx.strokeStyle = Math.abs(gy) < 1e-9 ? "rgba(120,150,190,0.22)" : "rgba(120,150,190,0.07)";
          ctx.beginPath(); ctx.moveTo(0, s.y); ctx.lineTo(W, s.y); ctx.stroke();
        }
      }
    }
    // Zemax-convention labels: the horizontal optical axis reads as Z, the
    // vertical height as Y — the meridional Y–Z plane a lens designer expects.
    // Internally these are canvas +X and +Y (§4.1); the labels are presentation.
    const axisY = toScreen({ x: 0, y: 0 }).y, axisX = toScreen({ x: 0, y: 0 }).x;
    ctx.fillStyle = "rgba(150,175,210,0.55)";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("Z", W - 16, axisY - 6);
    ctx.fillText("Y", axisX + 6, 14);
  }

  // --- interactive view -------------------------------------------------
  // The ray layer is drawn a slice per animation frame and shown when a
  // present lands, so a picture of millions of segments never stops the page.
  // `shown` is what the layer's canvas holds and the picture and view it was
  // drawn for; `work` is the raster being drawn. Until the next raster is
  // shown, the last one stays on screen, moved to the view as it is now: a
  // repaint with the same picture and view (a hover, a card) reuses it, an
  // edit shows the last picture until the next is whole, and a pan or zoom of
  // a picture that was slow to draw moves and scales it, and draws it again
  // once the view has been still for RASTER_SETTLE_MS. The rays are the same
  // either way; this only decides when they are drawn.
  type RasterKey = {
    version: number; scale: number; panX: number; panY: number;
    W: number; H: number; dpr: number; gOpacity: number;
  };
  type Shown = RasterKey & { count: number; ms: number };
  type Work = RasterKey & {
    segs: Float32Array; layout: RayLayout; o: RayDrawOpts; map: (p: V2) => V2;
    next: number; started: number; shownAt: number; hold: boolean;
    /// A present of it is in progress, showing this many segments.
    presenting: boolean; presentCount: number;
  };
  let shown: Shown | null = null;
  let work: Work | null = null;
  let settleTimer = 0;
  let redrawNow = false;
  const SLOW_RASTER_MS = 30;
  const RASTER_SETTLE_MS = 150;
  // Every picture is rasterized RASTER_SLICE_MS at a time: a ghost-lit lens's
  // segments cross the whole lens, and a few tens of thousands of them are
  // already a hundred milliseconds of drawing. A light picture is done, and
  // shown, in the first slice.
  const RASTER_SLICE_MS = 12;
  // Segments between checks of the slice's clock: a ghost segment can cross
  // the whole lens, a thousand pixels of line.
  const RASTER_CHUNK = 500;
  // A raster being drawn shows its progress this often, once the picture it
  // replaces has held for RASTER_HOLD_MS; tonemapping the lit region is work
  // too, so not per slice.
  const RASTER_SHOW_MS = 120;
  const RASTER_HOLD_MS = 250;
  let rasterFrame = 0;

  const samePicture = (a: RasterKey, b: RasterKey) =>
    a.version === b.version && a.W === b.W && a.H === b.H && a.dpr === b.dpr && a.gOpacity === b.gOpacity;
  const sameView = (a: RasterKey, b: RasterKey) => a.scale === b.scale && a.panX === b.panX && a.panY === b.panY;
  const keyOf = (k: RasterKey): RasterKey => ({
    version: k.version, scale: k.scale, panX: k.panX, panY: k.panY,
    W: k.W, H: k.H, dpr: k.dpr, gOpacity: k.gOpacity,
  });

  // Draw `w` for up to a slice: its segments, then its present, a band of
  // rows at a time. Says whether a present landed and whether `w` is whole.
  function rasterSlice(w: Work): { landed: boolean; whole: boolean } {
    const t = performance.now();
    const total = w.segs.length / w.layout.stride;
    if (!w.presenting) {
      while (w.next < total && performance.now() - t < RASTER_SLICE_MS) {
        rays.add(w.segs, w.next, w.next + RASTER_CHUNK, w.map, w.o, w.layout);
        w.next += RASTER_CHUNK;
      }
      const done = w.next >= total;
      const progress = !w.hold || t - w.started >= RASTER_HOLD_MS;
      if (done || (progress && t - w.shownAt >= RASTER_SHOW_MS)) {
        w.presentCount = rays.presentBegin().count;
        w.presenting = true;
      }
    }
    if (!w.presenting || !rays.presentRows(Math.max(1, RASTER_SLICE_MS - (performance.now() - t)))) {
      return { landed: false, whole: false };
    }
    w.presenting = false;
    w.shownAt = t;
    const whole = w.next >= total;
    shown = { ...keyOf(w), count: w.presentCount, ms: whole ? performance.now() - w.started : Infinity };
    return { landed: true, whole };
  }

  function continueRaster() {
    rasterFrame = 0;
    const w = work;
    if (!w) return;
    const { landed, whole } = rasterSlice(w);
    if (whole) work = null;
    else rasterFrame = requestAnimationFrame(continueRaster);
    if (landed) renderView();
  }

  function renderView() {
    // The GPU 3D view paints on its own animation frame; ask it for one.
    if (in3dModel()) {
      viewport3d?.invalidate();
      return;
    }
    const t0 = performance.now();
    const W = w(), H = h();
    const dpr = window.devicePixelRatio || 1;
    const gOpacity = Number(ghostOpacity.value) / 100;
    const showTir = tirBox.checked;
    const o: RayDrawOpts = { ghostOpacity: gOpacity, exposure: EXPOSURE, dpr, width: W, height: H };

    // Ray segments (9 floats: ax ay bx by r g b flux flags) in meridional-plane
    // coordinates, mapped world→screen by `toScreen`.
    const version = slice3d?.active() ? slice3d.picture() : -1;
    const want: RasterKey = { version, scale, panX, panY, W, H, dpr, gOpacity };
    let traceMs = 0;
    if (version < 0) {
      // A v1 scene: light, and drawn at once.
      const t1 = performance.now();
      const segs = scene.traceColored();
      traceMs = performance.now() - t1;
      shown = { ...want, count: rays.draw(segs, screenMap(), o, PLAIN_RAYS).count, ms: 0 };
      work = null;
    } else {
      if (work && !(samePicture(work, want) && sameView(work, want))) {
        // The picture or the view changed under a raster being drawn.
        cancelAnimationFrame(rasterFrame);
        rasterFrame = 0;
        work = null;
      }
      const upToDate = shown !== null && samePicture(shown, want) && sameView(shown, want);
      if (!work && !upToDate) {
        const moving = shown !== null && samePicture(shown, want) && shown.ms > SLOW_RASTER_MS;
        if (moving && !redrawNow) {
          clearTimeout(settleTimer);
          settleTimer = window.setTimeout(() => {
            redrawNow = true;
            renderView();
          }, RASTER_SETTLE_MS);
        } else {
          redrawNow = false;
          const t1 = performance.now();
          const { segs, layout } = slice3d!.rayPicture();
          traceMs = performance.now() - t1;
          const next: Work = {
            ...want, segs, layout, o, map: screenMap(),
            next: 0, started: t1, shownAt: -Infinity,
            hold: shown !== null && shown.dpr === dpr,
            presenting: false, presentCount: 0,
          };
          rays.begin(segs, o, layout);
          work = next;
          if (rasterSlice(next).whole) work = null;
          else rasterFrame = requestAnimationFrame(continueRaster);
        }
      }
    }
    const count = shown?.count ?? 0;

    // Compose: dark background, the grid, then add the ray layer as last
    // shown, at device resolution and moved from the view it was drawn for to
    // this one.
    ctx.fillStyle = "#06080d";
    ctx.fillRect(0, 0, W, H);
    draw2dGrid(W, H);
    if (shown && shown.dpr === dpr) {
      const k = scale / shown.scale;
      const ox = W / 2 + panX - k * (shown.W / 2 + shown.panX);
      const oy = H / 2 + panY - k * (shown.H / 2 + shown.panY);
      ctx.save();
      ctx.setTransform(k, 0, 0, k, ox * dpr, oy * dpr);
      ctx.globalCompositeOperation = "lighter";
      ctx.drawImage(rays.canvas, 0, 0);
      // TIR highlight: a ring where a surface is in total internal reflection.
      // The ray layer draws the rings with the rays, so they move with them.
      if (showTir) {
        ctx.globalCompositeOperation = "source-over";
        ctx.drawImage(rays.rings, 0, 0);
      }
      ctx.restore();
    }

    // Overlays: element cross-sections, selection handles, cards.
    if (!slice3d?.active()) m3?.overlay(ctx);
    slice3d?.overlay(ctx, toScreen);

    writeStatus({ count, ms: performance.now() - t0, traceMs });
  }

  // The status line. In the 3D view it also names the presentation backend,
  // which is a different subsystem from the ray-preview backend before it.
  function writeStatus({ count, ms, traceMs }: FrameReport) {
    // Both views report every drawn frame here, so the component view catches
    // up with any edit or selection change once per frame.
    componentView?.sync();
    const three = in3dModel() ? viewport3d : null;
    const view = in3dModel() ? " · 3D model" : "";
    const render = three ? ` · ${three.backendLabel()}` : "";
    stat.textContent = `${count} segs · ${ms.toFixed(1)} ms (trace ${traceMs.toFixed(1)}) · ${committed.name} nd ${committed.nd.toFixed(4)} V ${committed.vd.toFixed(1)}${view} · ${fast.label()}${render}`;
  }

  // --- Abbe picker (§5.4) -----------------------------------------------
  const VD = [95, 20], ND = [1.43, 1.95]; // axis ranges (Vd decreases →)
  const AW = abbe.width, AH = abbe.height;
  const abX = (vd: number) => ((VD[0] - vd) / (VD[0] - VD[1])) * AW;
  const abY = (nd: number) => AH - ((nd - ND[0]) / (ND[1] - ND[0])) * AH;
  const abVd = (px: number) => VD[0] - (px / AW) * (VD[0] - VD[1]);
  const abNd = (py: number) => ND[0] + ((AH - py) / AH) * (ND[1] - ND[0]);

  function drawAbbe(cursor?: V2, obtainable = true) {
    actx.clearRect(0, 0, AW, AH);
    // hatch the whole panel (the "unobtainable" fill)
    actx.strokeStyle = "rgba(90,70,70,0.35)";
    actx.lineWidth = 1;
    for (let d = -AH; d < AW; d += 7) {
      actx.beginPath();
      actx.moveTo(d, 0);
      actx.lineTo(d + AH, AH);
      actx.stroke();
    }
    // carve out the obtainable region (fill hull with panel bg)
    if (hull.length >= 6) {
      actx.beginPath();
      actx.moveTo(abX(hull[0]), abY(hull[1]));
      for (let i = 2; i < hull.length; i += 2) actx.lineTo(abX(hull[i]), abY(hull[i + 1]));
      actx.closePath();
      actx.fillStyle = "#0a0e16";
      actx.fill();
      actx.strokeStyle = "rgba(127,214,255,0.5)";
      actx.stroke();
    }
    // glass dots
    for (const gl of glasses) {
      actx.fillStyle = "rgba(200,220,245,0.85)";
      actx.beginPath();
      actx.arc(abX(gl.vd), abY(gl.nd), 2.2, 0, Math.PI * 2);
      actx.fill();
    }
    // committed marker (the active target's current glass)
    const cm = curCommitted();
    actx.strokeStyle = "#ffd678";
    actx.lineWidth = 1.5;
    actx.beginPath();
    actx.arc(abX(cm.vd), abY(cm.nd), 4, 0, Math.PI * 2);
    actx.stroke();
    // cursor preview
    if (cursor) {
      actx.strokeStyle = obtainable ? "#7fffa0" : "#ff5a3c";
      actx.beginPath();
      actx.arc(cursor.x, cursor.y, 5, 0, Math.PI * 2);
      actx.stroke();
    }
  }

  function nearestGlass(vd: number, nd: number): Glass | null {
    let best: Glass | null = null;
    let bestD = Infinity;
    for (const gl of glasses) {
      const dx = abX(gl.vd) - abX(vd), dy = abY(gl.nd) - abY(nd);
      const dsq = dx * dx + dy * dy;
      if (dsq < bestD) { bestD = dsq; best = gl; }
    }
    return bestD < 100 ? best : null; // within 10 px → snap
  }

  function applyGlass(g: Glass) {
    committed.name = g.name;
    committed.nd = g.nd;
    committed.vd = g.vd;
    scene.setElementAbbe(lensId, g.nd, g.vd);
  }

  // The Abbe picker retargets the selected 3D body's glass when a 3D scene is
  // active, else the v1 demo lens. `committed3d` is the selected body's glass to
  // revert to when the cursor leaves the map without clicking.
  let committed3d: Glass | null = null;
  const glassTarget = () => (slice3d?.active() ? selectedBody3d() : -1);
  const readGlass3d = (id: number): Glass | null => {
    const g = JSON.parse(scene.bodyGlassJSON(id));
    return g ? { name: "custom", nd: g.nd, vd: g.vd } : null;
  };
  // The glass marker to draw on the map: the selected 3D body's, or the v1 lens'.
  const curCommitted = (): Glass => (slice3d?.active() && committed3d ? committed3d : committed);

  // Live preview as the cursor moves over the map.
  function previewGlass(nd: number, vd: number) {
    const id = glassTarget();
    if (id >= 0) { scene.setBodyGlassAbbe(id, nd, vd); refreshActive3d(); }
    else scene.setElementAbbe(lensId, nd, vd);
    renderView();
  }
  // Commit the clicked glass.
  function commitGlass(g: Glass) {
    const id = glassTarget();
    if (id >= 0) { scene.setBodyGlassAbbe(id, g.nd, g.vd); committed3d = { ...g }; refreshActive3d(); }
    else applyGlass(g);
  }
  // Revert the preview to the committed glass when the cursor leaves.
  function revertGlass() {
    const id = glassTarget();
    if (id >= 0) { if (committed3d) scene.setBodyGlassAbbe(id, committed3d.nd, committed3d.vd); refreshActive3d(); }
    else applyGlass(committed);
  }

  abbe.addEventListener("pointermove", (e) => {
    const rect = abbe.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const snap = nearestGlass(abVd(px), abNd(py));
    const nd = snap ? snap.nd : abNd(py);
    const vd = snap ? snap.vd : abVd(px);
    const ok = scene.isObtainable(nd, vd);
    drawAbbe({ x: snap ? abX(vd) : px, y: snap ? abY(nd) : py }, ok);
    if (ok) {
      abbeLabel.textContent = snap ? `${snap.name} — nd ${nd.toFixed(4)}, V ${vd.toFixed(1)}`
        : `custom — nd ${nd.toFixed(4)}, V ${vd.toFixed(1)}`;
      previewGlass(nd, vd); // live preview: chromatic spread re-converges
    } else {
      abbeLabel.textContent = "unobtainable — no glass exists here";
    }
  });
  abbe.addEventListener("pointerdown", (e) => {
    const rect = abbe.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const snap = nearestGlass(abVd(px), abNd(py));
    const nd = snap ? snap.nd : abNd(py);
    const vd = snap ? snap.vd : abVd(px);
    if (scene.isObtainable(nd, vd)) {
      commitGlass(snap ?? { name: "custom", nd, vd });
      renderView();
      refreshReference();
      if (!slice3d?.active()) autoCommit("glass"); // v1 undo-tree snapshot (§5.2)
    }
  });
  abbe.addEventListener("pointerleave", () => {
    revertGlass(); // revert preview to the committed glass
    drawAbbe();
    renderView();
    refreshReference();
  });

  // --- detector dock (f64 reference numbers, on idle) --------------------
  function refreshReference() {
    if (slice3d?.active()) return refresh3dDetector();
    const info = JSON.parse(scene.detectorJson(0));
    hctx.clearRect(0, 0, hist.width, hist.height);
    if (!info || info.hits === 0) {
      dockNums.textContent = "no rays on detector";
      return;
    }
    const bins: number[] = info.bins;
    const max = Math.max(...bins, 1e-12);
    const bw = hist.width / bins.length;
    hctx.fillStyle = "#7fd6ff";
    for (let i = 0; i < bins.length; i++) {
      const bh = (bins[i] / max) * (hist.height - 2);
      hctx.fillRect(i * bw, hist.height - bh, Math.max(1, bw - 0.5), bh);
    }
    const cen = info.centroid == null ? "—" : (info.centroid * 14 - 7).toFixed(3);
    dockNums.textContent = `hits ${info.hits} · flux ${info.flux.toFixed(3)} · centroid y ${cen} mm · OPL ${Number(info.opl).toFixed(2)} mm`;
  }

  // The 3D area detector's f64 analysis (§6.9), true 3D — not a slice of a 2D
  // analysis. Two modes: the 2D binning heatmap (incident flux per cell) and the
  // spot diagram (the actual ray-detector intersection points). The numbers line
  // is the same reference readout under both.
  type DetMode = "heatmap" | "spot";
  let detMode: DetMode = "heatmap";
  const detHead = document.getElementById("detHead")!;

  // The centred square drawing region of the dock canvas, for the polar map.
  function detSquare() {
    const s = Math.min(hist.width, hist.height);
    return { s, ox: (hist.width - s) / 2, oy: (hist.height - s) / 2 };
  }

  // The angular histogram, when the detector measures intensity. A different
  // histogram from the face one, so a different picture: polar angle outward
  // from the middle rather than millimetres across a face.
  function drawPolar(info: {
    cells: number[];
    coneBins: [number, number];
    coneDeg: number;
  }) {
    const { s, ox, oy } = detSquare();
    drawPolarMap(
      hctx,
      {
        cells: info.cells,
        rings: info.coneBins[0],
        sectors: info.coneBins[1],
        coneDeg: info.coneDeg,
      },
      { x: ox, y: oy, w: s, h: s },
    );
  }

  // The face, pixel by pixel, in its true proportions and clipped to its
  // shape: flux per pixel, or under irradiance that flux over each pixel's
  // area on the face. The component inset draws the same picture from the
  // same reply (detector-face.ts). Irradiance is scaled to its reported peak,
  // which skips rim slivers, so one sliver cannot darken the rest.
  function drawHeatmap(info: DetectorReadout) {
    if (!info.face) return;
    const irradiance = info.measure === "irradiance";
    drawFacePixels(
      hctx,
      info.face,
      irradiance ? info.cells : info.incident,
      { x: 0, y: 0, w: hist.width, h: hist.height },
      irradiance ? info.peak : null,
    );
  }

  // Spot diagram: the actual ray-detector hits on the face, to scale, coloured
  // by wavelength (the linear-light rgb the tracer records, normalized to a dot).
  function drawSpot(a: Float32Array, face: DetectorFace) {
    const b = faceBox(face, { x: 0, y: 0, w: hist.width, h: hist.height });
    hctx.fillStyle = "#04060a";
    facePath(hctx, face, b);
    hctx.fill("evenodd");
    // Crosshair at the face centre.
    hctx.strokeStyle = "rgba(127,214,255,0.12)";
    hctx.beginPath();
    hctx.moveTo(b.x + b.w / 2, b.y); hctx.lineTo(b.x + b.w / 2, b.y + b.h);
    hctx.moveTo(b.x, b.y + b.h / 2); hctx.lineTo(b.x + b.w, b.y + b.h / 2);
    hctx.stroke();
    hctx.strokeStyle = "rgba(127,214,255,0.25)";
    facePath(hctx, face, b);
    hctx.stroke();
    if (a.length < 3) return;
    const n = a[2];
    for (let i = 0; i < n; i++) {
      const k = 3 + i * 6;
      const u = a[k], v = a[k + 1];
      const r = a[k + 2], g = a[k + 3], bl = a[k + 4];
      const m = Math.max(r, g, bl, 1e-6);
      const px = b.x + (u / face.width + 0.5) * b.w;
      const py = b.y + (0.5 - v / face.height) * b.h; // v up, screen down
      hctx.fillStyle = `rgb(${(r / m * 255) | 0},${(g / m * 255) | 0},${(bl / m * 255) | 0})`;
      hctx.fillRect(px - 0.75, py - 0.75, 1.7, 1.7);
    }
  }

  // What the dock's numbers are: the reference of the scene as it is, the
  // previous scene's while the new one is traced, or nothing yet.
  const refStatus = document.getElementById("refStatus")!;
  const dockPanel = document.getElementById("dock")!;
  function writeReferenceStatus() {
    const st = reference.status();
    const secs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
    const pct = (f: number) => `${Math.round(f * 100)}%`;
    let text = "";
    if (st.state === "current") text = `reference current · ${st.bundle.rays} rays in ${secs(st.bundle.elapsedMs)}`;
    else if (st.state === "stale") text = `updating ${pct(st.fraction)} · numbers are the previous scene's`;
    else if (st.state === "updating") text = `tracing the reference ${pct(st.fraction)}`;
    refStatus.textContent = text;
    refStatus.classList.toggle("stale", st.state === "stale" || st.state === "updating");
    dockPanel.classList.toggle("stale", st.state === "stale");
  }

  /** The reference the panels read: this scene's, else the newest there is. */
  const shownReference = (): ReferenceBundle | null => {
    const now = reference.current();
    if (!now) reference.request();
    return now ?? reference.latest();
  };

  function refresh3dDetector() {
    // Whatever made the dock re-read (an edit settling, a trace setting such
    // as the ghost layer that moves no scene revision) makes the component
    // inset re-read too, so the two never quote different traces.
    componentView?.retraced();
    hctx.clearRect(0, 0, hist.width, hist.height);
    const b = shownReference();
    writeReferenceStatus();
    if (!b) {
      dockNums.textContent = "tracing…";
      return;
    }
    // Show the selected detector if one is selected, else the first — so a scene
    // with several detectors reports whichever the user is inspecting.
    const sel = selectedBody3d();
    const selRaw = sel >= 0 ? b.readouts.get(sel) : undefined;
    const selInfo = selRaw ? (JSON.parse(selRaw) as DetectorReadout | null) : null;
    const detId = selInfo ? sel : (b.firstDetector ?? -1);
    const firstRaw = b.firstDetector != null ? b.readouts.get(b.firstDetector) : undefined;
    const info = (selInfo ?? (firstRaw ? JSON.parse(firstRaw) : null)) as DetectorReadout | null;
    if (!info || info.hits === 0) {
      dockNums.textContent = info ? "no rays on detector" : "no detector";
      return;
    }
    if (detMode === "spot") {
      if (info.face) drawSpot(b.hits.get(detId) ?? new Float32Array(0), info.face);
    } else if (info.measure === "intensity") {
      // Not a heatmap of anything on the face, so the label stops saying so.
      detHead.textContent = "detector — intensity, polar (f64)";
      drawPolar({ cells: info.cells, coneBins: info.coneBins, coneDeg: info.coneDeg ?? 0 });
    } else {
      detHead.textContent = info.measure === "irradiance" ? "detector — irradiance heatmap (f64)" : "detector — heatmap (f64)";
      drawHeatmap(info);
    }
    const c = info.centroid ? `(${info.centroid[0].toFixed(2)}, ${info.centroid[1].toFixed(2)})` : "—";
    const opl = info.meanOpl == null ? "—" : Number(info.meanOpl).toFixed(2);
    const tag = selInfo ? `#${detId} · ` : "";
    // Polarization, when there is any. An unpolarized beam says nothing here
    // rather than printing a zero the reader has to interpret. The tilt is the
    // caveat on the numbers: the summed state is written in the detector's own
    // frame, and a wide cone of arrival directions makes that a shared frame in
    // name only.
    let pol = "";
    const dop = Number(info.dop);
    if (Number.isFinite(dop) && dop > 1e-6) {
      const kind = Number(info.dolp) > dop - 1e-6 ? `linear ${Number(info.azimuthDeg).toFixed(0)}°`
        : Number(info.dolp) < 1e-6 ? "circular"
        : `elliptical ${Number(info.azimuthDeg).toFixed(0)}°`;
      const tilt = Number(info.maxTiltDeg);
      const spread = Number.isFinite(tilt) && tilt > 5 ? ` (±${tilt.toFixed(0)}° cone)` : "";
      pol = ` · pol ${(dop * 100).toFixed(1)}% ${kind}${spread}`;
    }
    // What the detector measures, next to the flux it collected. Power says
    // nothing extra, because the peak of a flux histogram in flux is the
    // number already there under another name.
    let reads = "";
    if (info.measure === "irradiance") {
      reads = ` · peak ${formatFlux(info.peak)} ${info.unit}`;
    } else if (info.measure === "intensity") {
      const share = Number(info.totalIncident) > 0 ? Number(info.outsideCone) / Number(info.totalIncident) : 0;
      const past = share > 1e-9 ? `, ${(share * 100).toFixed(0)}% past ${Number(info.coneDeg).toFixed(0)}°` : "";
      reads = ` · peak ${formatFlux(info.peak)} ${info.unit}${past}`;
    }
    dockNums.textContent = `${tag}hits ${info.hits} · flux ${formatFlux(info.totalIncident)}${reads} · centroid ${c} mm · OPL ${opl} mm${pol}`;
  }

  // Detector analysis-mode toggle (heatmap ⇄ spot).
  const detHeatBtn = document.getElementById("detHeatmap") as HTMLButtonElement;
  const detSpotBtn = document.getElementById("detSpot") as HTMLButtonElement;
  function setDetMode(m: DetMode) {
    detMode = m;
    detHeatBtn.classList.toggle("active", m === "heatmap");
    detSpotBtn.classList.toggle("active", m === "spot");
    detHead.textContent = `detector — ${m === "spot" ? "spot diagram" : "heatmap"} (f64)`;
    refreshReference();
  }
  detHeatBtn.onclick = () => setDetMode("heatmap");
  detSpotBtn.onclick = () => setDetMode("spot");

  // --- controls ----------------------------------------------------------
  // These knobs feed both tracers: `setGhosts`/`setMaxGhostBounces` set the
  // Canvas fields that v1 and the 3D `cfg3` both read, so a change re-traces
  // whichever scene is active (the 3D path via `slice3d.refresh()`).
  ghostsBox.addEventListener("change", () => {
    scene.setGhosts(ghostsBox.checked);
    refreshActive3d();
    renderView();
  });
  ghostOpacity.addEventListener("input", renderView); // a RayLayer render param
  tirBox.addEventListener("change", renderView);       // a render overlay toggle
  ghostBounces.addEventListener("change", () => {
    scene.setMaxGhostBounces(Number(ghostBounces.value) || 2);
    refreshActive3d();
    renderView();
    refreshReference();
  });
  // Grid toggle / spacing repaint the view (snapping reads the same DOM live).
  gridBox.addEventListener("change", renderView);
  gridSpacing.addEventListener("input", renderView);
  // --- light source picker (§6.6) ---------------------------------------
  // Collimated (rect/round), point (cone), or diffuse disc. The params and their
  // labels follow the type; `ray density` is the total ray count, mapped to the
  // right sampling dimension by `setSourceConfig`. In 3D all of this drives the
  // Scene3 source; in the (hidden) v1 fallback density still drives the v1 beam.
  const srcType = document.getElementById("srcType") as HTMLSelectElement;
  const srcP0 = document.getElementById("srcP0") as HTMLInputElement;
  const srcP1 = document.getElementById("srcP1") as HTMLInputElement;
  const srcL0 = document.getElementById("srcL0")!;
  const srcU0 = document.getElementById("srcU0")!;
  const srcL1 = document.getElementById("srcL1")!;
  const srcRow1 = document.getElementById("srcRow1")!;

  // Sensible starting params per source type (the param means a different thing
  // for each, so a type switch resets rather than reinterpreting the old number).
  const SRC_DEFAULTS: Record<string, [number, number]> = {
    collimated_rect: [20, 0.5],
    collimated_disc: [20, 0],
    point: [5, 0], // cone half-angle in degrees
    lambertian: [10, 0],
  };

  function updateSrcLabels() {
    const t = srcType.value;
    srcRow1.style.display = t === "collimated_rect" ? "" : "none";
    if (t === "point") { srcL0.textContent = "cone"; srcU0.textContent = "°"; }
    else { srcL0.textContent = t === "collimated_rect" ? "⌀ Y" : "⌀"; srcU0.textContent = "mm"; srcL1.textContent = "⌀ X"; }
  }

  // Push the current picker state to the Scene3 source and re-trace.
  function applySource() {
    if (!slice3d?.active()) { scene.setSourceRays(0, Number(density.value)); return; }
    scene.setSourceConfig(srcType.value, Number(srcP0.value) || 1, Number(srcP1.value) || 1, Number(density.value));
    refreshActive3d();
  }

  srcType.addEventListener("change", () => {
    const d = SRC_DEFAULTS[srcType.value];
    if (d) { srcP0.value = String(d[0]); srcP1.value = String(d[1]); }
    updateSrcLabels(); applySource(); renderView(); refreshReference();
  });
  srcP0.addEventListener("input", () => { applySource(); renderView(); });
  srcP1.addEventListener("input", () => { applySource(); renderView(); });
  srcP0.addEventListener("change", refreshReference);
  srcP1.addEventListener("change", refreshReference);

  // Ray density: the beam's total ray count. Live retrace while sliding.
  density.addEventListener("input", () => { applySource(); renderView(); });
  density.addEventListener("change", () => {
    refreshReference();
    if (!slice3d?.active()) autoCommit("ray density");
  });

  // --- wavelength picker (§5.5) -----------------------------------------
  // A preset (white, or Fraunhofer lines) or a custom list of wavelengths in nm.
  // The swatches show each line's rendered colour (a wavelength outside the
  // visible band is drawn yellow, matching the rays). Wavelengths drive real
  // dispersion, so a change re-traces.
  const wlPreset = document.getElementById("wlPreset") as HTMLSelectElement;
  const wlList = document.getElementById("wlList") as HTMLInputElement;
  const wlSwatches = document.getElementById("wlSwatches")!;
  const WL_PRESETS: Record<string, number[]> = {
    // White: nine equal lines across the visible band (the default beam).
    white: [420, 450, 480, 510, 540, 570, 600, 630, 660],
    d: [587.56],
    fdc: [486.13, 587.56, 656.27],
    gfedc: [435.83, 486.13, 546.07, 587.56, 656.27],
  };

  const parseWlList = () =>
    wlList.value.split(/[\s,]+/).map(Number).filter((x) => Number.isFinite(x) && x > 0);

  function renderSwatches(nm: number[]) {
    wlSwatches.replaceChildren();
    for (const w of nm) {
      const dot = document.createElement("span");
      dot.title = `${w} nm`;
      dot.style.cssText =
        `width:12px; height:12px; border-radius:3px; display:inline-block; background:${scene.wavelengthCssColor(w)}; box-shadow:0 0 0 1px #0006`;
      wlSwatches.appendChild(dot);
    }
  }

  // Push a wavelength list to the source and re-trace (no-op on an empty list).
  function applyWavelengths(nm: number[]) {
    if (nm.length === 0) { renderSwatches([]); return; }
    scene.setSourceWavelengths(new Float64Array(nm), new Float64Array(0));
    renderSwatches(nm);
    refreshActive3d();
  }

  wlPreset.addEventListener("change", () => {
    const custom = wlPreset.value === "custom";
    wlList.disabled = !custom;
    if (custom) { wlList.focus(); applyWavelengths(parseWlList()); }
    else { applyWavelengths(WL_PRESETS[wlPreset.value]); renderView(); refreshReference(); }
  });
  wlList.addEventListener("input", () => { wlPreset.value = "custom"; wlList.disabled = false; applyWavelengths(parseWlList()); renderView(); });
  wlList.addEventListener("change", refreshReference);

  // Sync the picker from the current Scene3 source (called after boot, since the
  // 3D scene is created later in init).
  function syncSourcePicker() {
    const s = JSON.parse(scene.firstSourceJSON());
    if (s) {
      srcType.value = s.kind;
      srcP0.value = String(Number(s.p0.toFixed(2)));
      srcP1.value = String(Number(s.p1.toFixed(2)));
      density.value = String(Math.min(240, Math.max(8, Math.round(s.samples))));
    }
    updateSrcLabels();
    // Wavelengths: match a known preset, else fall back to a custom list.
    const arr = (JSON.parse(scene.firstSourceWavelengthsJSON()) as { nm: number }[] | null) ?? [];
    const nm = arr.map((o) => Number(o.nm.toFixed(2)));
    const preset = Object.keys(WL_PRESETS).find(
      (k) => WL_PRESETS[k].length === nm.length && WL_PRESETS[k].every((v, i) => Math.abs(v - nm[i]) < 0.5),
    );
    wlPreset.value = preset ?? "custom";
    wlList.disabled = wlPreset.value !== "custom";
    wlList.value = preset === "white" ? "" : nm.join(", ");
    renderSwatches(nm);
  }
  updateSrcLabels();

  // --- drag the lens (fallback; M3 module takes over when loaded) --------
  let dragging = false;
  view.addEventListener("pointerdown", (e) => {
    if (m3) return; // the M3 module owns on-canvas manipulation (select/draw/pin)
    const p = toWorld(e.clientX, e.clientY);
    if (Math.abs(p.x - lens.x) < 6 && Math.abs(p.y - lens.y) < 14) {
      dragging = true;
      view.classList.add("dragging");
      view.setPointerCapture(e.pointerId);
    }
  });
  view.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const p = toWorld(e.clientX, e.clientY);
    lens.x = p.x;
    lens.y = p.y;
    scene.moveElement(lensId, lens.x, lens.y);
    renderView();
  });
  const endDrag = (e: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    view.classList.remove("dragging");
    try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    refreshReference(); // numbers settle after release (§4.1)
  };
  view.addEventListener("pointerup", endDrag);
  view.addEventListener("pointercancel", endDrag);

  // --- go ----------------------------------------------------------------
  m3 = initM3({
    scene,
    view,
    toScreen,
    toWorld,
    zoomAt,
    panBy,
    resetView,
    render: renderView,
    refresh: refreshReference,
    demoLensId: lensId,
    threeDActive: () => slice3d?.active() ?? false,
    viewActive: () => !in3dModel(),
    // Layer hooks are inert in the always-3D app (the hidden v1 scene has one
    // layer), but M3 keeps them for standalone use.
    commit: autoCommit,
    layerVisible: (id) => layerCache[id]?.visible ?? true,
    layerOpacity: (id) => layerCache[id]?.opacity ?? 1,
    onionSkin: () => false,
  });

  // The two views over the canonical 3D scene. Both read the same Scene3 in Rust
  // and trace the same authoritative f64 path; they differ only in projection and
  // interaction. `owns` gates each view's pointer/keyboard handlers so only the
  // visible one edits the scene.
  slice3d = initSlice3d({
    scene, view, toScreen, toWorld, render: renderView, fast, reference,
    owns: () => !in3dModel(),
    // When a 3D body is selected, aim the Abbe picker at its glass; the readout
    // dock follows the selected detector when a scene has more than one.
    onSelect: (id) => { committed3d = readGlass3d(id); drawAbbe(); refresh3dDetector(); },
  });
  const onSelect3d = (id: number) => { committed3d = readGlass3d(id); drawAbbe(); refresh3dDetector(); };
  viewport3d = initView3dThree({
    scene,
    view,
    canvas: document.getElementById("view3d") as HTMLCanvasElement,
    render: renderView,
    fast,
    owns: () => in3dModel(),
    onSelect: onSelect3d,
    detectorRefresh: refreshReference,
    reference,
    loads: () => slice3d?.loads() ?? 0,
    rayOptions: () => ({
      ghostOpacity: Number(ghostOpacity.value) / 100,
      showTir: tirBox.checked,
    }),
    grid: () => ({ on: gridBox.checked, spacing: Math.max(0.1, Number(gridSpacing.value) || 5) }),
    status: writeStatus,
    forceWebGL: params.get("render3d") === "webgl",
    disable: params.get("render3d") === "off",
    timing: params.has("debug"),
  });

  // The component view: a tab in the top bar opens it as an inset window, in
  // place or wherever the tab is dropped. Its handles edit the selection; the
  // view that owns the canvas draws the fast picture while a handle is held
  // and settles on the exact trace when it is let go.
  const inset = initInset({
    tab: document.getElementById("insetTab")!,
    bar: document.getElementById("tools")!,
    frame: document.getElementById("componentInset")!,
    ghost: document.getElementById("insetGhost")!,
  });
  componentView = createComponentView({
    scene,
    // The readout the dock shows: this scene's reference, else the newest.
    readout: (id) => shownReference()?.readouts.get(id) ?? null,
    body: inset.body,
    setSubtitle: inset.setSubtitle,
    selected: () => selectedBody3d(),
    edited: (settled) => {
      if (in3dModel()) {
        if (settled) { viewport3d!.refresh(); refreshReference(); }
        else viewport3d!.previewEdit();
        return;
      }
      if (settled) slice3d!.refresh();
      else slice3d!.previewEdit();
      renderView();
      if (settled) refreshReference();
    },
  });
  inset.attach(componentView);

  // Always-3D: the app boots straight into the canonical 3D scene — a singlet
  // traced by the f64 3D engine in the Y–Z plane (§4.1), shown as the meridional
  // cross-section. The v1 freeform layer stays initialized but hidden (its
  // picking is suppressed via `threeDActive`), so pan/zoom still work.
  slice3d.bootDefault();
  syncSourcePicker();
  // The status line names the preview backend; repaint once it is chosen.
  void fast.ready.then(() => renderView());
  // The GPU 3D view's code loads in idle time after startup.
  setTimeout(() => viewport3d?.prefetch(), 2000);

  // `?debug` exposes the scene and both fast paths to the opt-in WebGPU browser
  // test (web/tests/webgpu.spec.ts). The app itself never reads it.
  if (new URLSearchParams(location.search).has("debug")) {
    (window as unknown as { __ocDebug: unknown }).__ocDebug = {
      scene,
      fast,
      ready: fast.ready,
      cpuProjected: () => scene.traceProjected3DFast(),
      // Through the app's own backend choice, as a drag would ask.
      fastProjected: () => new Promise((resolve) => fast.requestProjected(scene, resolve, true)),
      gpuProjected: () => fast.gpu()?.traceProjected3D(scene) ?? Promise.resolve(null),
      exactProjected: () => scene.traceProjected3D(),
      // The GPU 3D view's camera, statistics, and pixel probe (tests).
      render3d: viewport3d?.debug ?? null,
      // The active view, drawn now (for timing the 2D view).
      renderView: () => {
        const t0 = performance.now();
        renderView();
        return performance.now() - t0;
      },
      setViewMode: (m: "2d" | "3d") => setViewMode(m),
      // The selection the active view (and the material picker) targets.
      selectedBody3d: () => selectedBody3d(),
      // Select an object in the cross-section view, as a click on it would.
      select: (id: number) => { slice3d!.setSelected(id); renderView(); },
      // Is the f64 reference of the scene as it is in? (The dock re-reads in
      // the task that accepts it, so its numbers are this scene's once true.)
      referenceCurrent: () => reference.current() !== null,
      // How many workers share the reference, and what its status line says.
      referenceWorkers: () => reference.workers(),
      referenceSegments: () => (reference.current()?.projected.length ?? 0) / 11,
      standIns: () => standIns,
      // The 2D view's zoom and pan, and the section's extent it frames.
      view2d: () => ({ scale, panX, panY, extent: slice3d?.extent() ?? null }),
      // The component view's inset window.
      inset: {
        open: () => inset.open(),
        close: () => inset.close(),
        isOpen: () => inset.isOpen(),
        view: () => componentView?.debug() ?? null,
      },
    };
  }

  // --- 2D ⇄ 3D view toggle ----------------------------------------------
  const view2dBtn = document.getElementById("view-2d") as HTMLButtonElement;
  const view3dBtn = document.getElementById("view-3d") as HTMLButtonElement;
  function setViewMode(m: ViewMode) {
    if (m === viewMode) return;
    viewMode = m;
    view2dBtn.classList.toggle("active", m === "2d");
    view3dBtn.classList.toggle("active", m === "3d");
    // The draw-mirror tool is a 2D cross-section action; disable it in 3D.
    document.getElementById("drawMirror")?.toggleAttribute("disabled", m === "3d");
    // The 3D view-controls bar is shown only in the 3D model view.
    document.getElementById("view3dctl")!.style.display = m === "3d" ? "flex" : "none";
    // The GPU view's appearance options belong to it alone.
    document.getElementById("view3dopts")!.style.display = m === "3d" ? "flex" : "none";
    // Hand interaction to the incoming view and stand the other one down (close
    // its stray cards). Refresh the incoming view's cached geometry/trace from
    // Rust so an edit made in the other view is reflected — both are views of the
    // one Scene3, and each caches its own cross-section/objects.
    if (m === "3d") { slice3d!.standDown(); viewport3d!.activate(slice3d!.selectedBody()); }
    else { viewport3d!.standDown(); slice3d!.refresh(); slice3d!.setSelected(viewport3d!.selectedBody()); }
    drawAbbe();
    renderView();
    refreshReference();
  }
  view2dBtn.onclick = () => setViewMode("2d");
  view3dBtn.onclick = () => setViewMode("3d");

  // 3D view controls (reset / presets / perspective).
  document.getElementById("v3dReset")!.onclick = () => viewport3d!.resetView();
  document.getElementById("v3dIso")!.onclick = () => viewport3d!.setPreset("iso");
  document.getElementById("v3dFront")!.onclick = () => viewport3d!.setPreset("front");
  document.getElementById("v3dTop")!.onclick = () => viewport3d!.setPreset("top");
  document.getElementById("v3dSide")!.onclick = () => viewport3d!.setPreset("side");
  const perspBtn = document.getElementById("v3dPersp") as HTMLButtonElement;
  perspBtn.onclick = () => {
    const on = !viewport3d!.isPerspective();
    viewport3d!.setPerspective(on);
    perspBtn.classList.toggle("active", on);
  };
  // The GPU view restores the last projection used in this browser.
  perspBtn.classList.toggle("active", viewport3d!.isPerspective());

  // Adders: element (round plate, per-face response), aperture (circular stop),
  // prism (dispersing glass triangle). Each spawns at a sensible spot, then opens
  // its card so it can be positioned / configured. `focusObject` re-reads + selects.
  const addAndFocus = (id: number) => {
    if (id < 0) return;
    (in3dModel() ? viewport3d! : slice3d!).focusObject(id);
    renderView();
    refreshReference();
  };

  // A free spot on the axis near `x`, so a second click of the same button does
  // not drop a plate exactly on top of the first.
  //
  // Two coincident zero-thickness surfaces are a degenerate scene, and the
  // tracer resolves only one of them: the epsilon that stops a child ray
  // re-hitting the surface it was born on cannot tell "the same surface" from
  // "a different surface in the same place". Two crossed polarizers stacked at
  // one point would report the light one of them passes, which is a wrong
  // number arrived at quietly. Spreading them is the cheap half of the fix; the
  // tracer learning to skip by identity rather than by distance is the other
  // half, and is not done.
  const GAP = 6;
  const freeX = (x: number, y: number) => {
    const taken = JSON.parse(scene.sceneObjectsJSON())
      .filter((o: { pos: number[] }) => Math.abs(o.pos[1] - y) < GAP && Math.abs(o.pos[2]) < GAP)
      .map((o: { pos: number[] }) => o.pos[0]);
    let at = x;
    // Step downstream until nothing is within a plate's width of the spot.
    for (let i = 0; i < 64 && taken.some((t: number) => Math.abs(t - at) < GAP); i++) at += GAP;
    return at;
  };
  document.getElementById("addElement")!.onclick = () => addAndFocus(scene.addPlanarElement(25, 0, 8, 45));
  document.getElementById("addAperture")!.onclick = () => addAndFocus(scene.addAperture(-20, 0, 5, 0));
  document.getElementById("addPrism")!.onclick = () => addAndFocus(scene.addPrism(-40, 0, 16, 0, 1.62, 36));
  document.getElementById("addDetector")!.onclick = () => addAndFocus(scene.addAreaDetector(40, 0, 8, 0));
  // Splitter and dichroic plates at 135°: the default beam meets their front
  // face and the reflected arm goes up (+Y).
  document.getElementById("addSplitter")!.onclick = () => addAndFocus(scene.addBeamSplitter(-35, 0, 15, 135, 0.5, 0.5));
  // A real coated glass plate: a 3 mm disc of N-BK7 with the stack on the face
  // the light meets first, its edge solved for the 45° it hangs at. The tracer
  // crosses it the way light does, so the transmitted beam comes out displaced
  // about a millimetre and the far face returns its own reflection. The older
  // table-driven dichroic is still in the kernel for documents that carry one,
  // but nothing in the UI creates one any more.
  document.getElementById("addDichroic")!.onclick = () => addAndFocus(scene.addPlateDichroic(-35, 0, 0, 15, 3, 135, 550, true));
  // Polarizing elements face the beam (tilt 0) rather than folding it: their
  // axis is a direction in the plate, so tilting one is a separate decision
  // from choosing its axis. Both are ideal and zero thickness, and the f32
  // preview has no polarization state to act on, so it draws a polarizer as a
  // 50% attenuator and a waveplate as a window. The f64 readout is the one that
  // knows what they do.
  document.getElementById("addPolarizer")!.onclick = () => addAndFocus(scene.addPolarizer(freeX(-25, 0), 0, 12, 0, 0, 0));
  document.getElementById("addRetarder")!.onclick = () => addAndFocus(scene.addRetarder(freeX(-15, 0), 0, 12, 0, 45, 90));
  // The polarizing cube is the odd one out: it is the only polarizing part
  // here whose behaviour is computed rather than declared. The stack is a
  // plain quarter-wave mirror, and what makes it polarizing is that it sits at
  // the Brewster angle between its own layers, which is why the cube's glass
  // is not an argument: the coating pair fixes it at 1.754, dense flint. So it
  // has a real angular acceptance and a real passband, and nobody typed
  // either.
  document.getElementById("addPbs")!.onclick = () =>
    addAndFocus(scene.addPolarizingBeamSplitter(freeX(-35, 0), 0, 0, 12, 135, 550, 12, 2.35, 1.46));

  // Tubes, the sun and heliostats. A tube is black outside with a protected
  // silver bore, the light pipe. The sun is placed from the site and the
  // clock; a heliostat follows it onto its target: the selected object, else
  // the first detector, else a point down the axis.
  const sunPanel = initSunPanel({
    scene,
    edited: (live) => {
      refreshActive3d();
      renderView();
      if (!live) refreshReference();
    },
  });
  document.getElementById("addTube")!.onclick = () =>
    addAndFocus(scene.addTube(freeX(10, 0), 0, 0, 20, 24, 80, "silver-protected", 0.95));
  document.getElementById("addSun")!.onclick = () => sunPanel.ensureSun();
  document.getElementById("addHeliostat")!.onclick = () => {
    const objs = JSON.parse(scene.sceneObjectsJSON()) as { id: number; kind: string }[];
    const picked = selectedBody3d();
    const target = picked >= 0 ? picked : (objs.find((o) => o.kind === "detector")?.id ?? -1);
    sunPanel.ensureSun();
    addAndFocus(scene.addHeliostat(-60, -40, 0, 30, 30, 0.95, target, 60, 0, 0));
    sunPanel.sync();
  };

  // --- save .ocanvas — the Optiland handoff (§6) --------------------------
  // Save downloads the canonical 3D scene (the one the user actually edits) as a
  // frozen, versioned v2 document. This closes the loop for a non-Python user:
  // design here, save, then `optiland_canvas.to_optic(file)`. Opening a scene is
  // in the 3D-scene panel above (it loads straight into Scene3).
  const saveBtn = document.getElementById("saveScene") as HTMLButtonElement;
  const sceneMsg = document.getElementById("sceneMsg")!;
  const flash = (msg: string, ok = true) => {
    sceneMsg.textContent = msg;
    sceneMsg.style.color = ok ? "#7fffa0" : "#ff8a6b";
  };

  saveBtn.addEventListener("click", () => {
    const json = scene.scene3JSON() || scene.toJSON();
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `scene-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.ocanvas`;
    a.click();
    URL.revokeObjectURL(url);
    flash(`saved ${a.download} · ${(json.length / 1024).toFixed(1)} kB`);
  });

  // --- the gallery: ready-made designs (web/src/gallery.ts) ---------------
  // A design opens the way a file does, then the view frames it and the panels
  // follow it: the source picker reads its first field, the readout its image.
  const gallery = initGallery({
    load: (name, text) => slice3d!.openText(name, text),
    frame: () => {
      if (in3dModel()) viewport3d!.resetView();
      else fitView();
      refreshActive3d();
      syncSourcePicker();
      drawAbbe();
      renderView();
      refreshReference();
    },
    opened: (d) => {
      const approx = d.notes?.length ? ` Approximate: ${d.notes.join("; ")}.` : "";
      flash(`${d.name}, from Optiland's samples (${d.source}).${approx}`);
    },
  });
  // `?gallery` opens the gallery; `?gallery=<id>` opens that design.
  if (params.has("gallery")) {
    const id = params.get("gallery");
    if (id) void gallery.open(id);
    else void gallery.show();
  }

  drawAbbe();
  renderView();
  refreshReference();
}

main().catch((e) => {
  document.body.innerHTML = `<pre style="padding:2rem;color:#f88">Failed to start: ${e}\n\nBuild the wasm first:  npm run wasm</pre>`;
  console.error(e);
});
