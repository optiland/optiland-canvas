// Optiland Canvas — M2 web shell ("it's optics").
//
// Scene lives in Rust (§2). This file holds UI state only. Two tracers, per the
// spec's tiering (§4.1): the interactive f32 path draws the picture every frame
// (spectral colour, ghost layer, TIR glow); the f64 reference feeds the
// detector dock's *numbers*, on idle.

import init, { Canvas } from "../pkg/oc_wasm.js";
import { initM3 } from "./m3";
import { RayLayer } from "./rays";
import { initSlice3d, type Slice3d } from "./slice3d";
import { initView3dThree, type FrameReport, type View3dThree } from "./render3d/view3d";
import { createFastPreview } from "./fastpreview";
import { initInset } from "./inset/inset";
import { createComponentView, type ComponentView } from "./inset/component-view";
import { initOpenUc2 } from "./features/openuc2/OpenUc2Page";
import { heatRgb } from "./heat";
import { drawPolarMap } from "./polarmap";

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

  // Zoom about a screen point, keeping the world point under it fixed.
  function zoomAt(clientX: number, clientY: number, factor: number) {
    const before = toWorld(clientX, clientY);
    scale = Math.min(400, Math.max(0.4, scale * factor));
    panX = clientX - w() / 2 - before.x * scale;
    panY = clientY - h() / 2 + before.y * scale;
  }
  const panBy = (dxScreen: number, dyScreen: number) => { panX += dxScreen; panY += dyScreen; };
  const resetView = () => { scale = SCALE; panX = 0; panY = 0; };

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

    // Ray segments (9 floats: ax ay bx by r g b flux flags) in meridional-plane
    // coordinates, mapped world→screen by `toScreen`.
    const segs = slice3d?.active() ? slice3d.raySegments() : scene.traceColored();
    const traceMs = performance.now() - t0;
    const { count, tirDots } = rays.draw(segs, toScreen, {
      ghostOpacity: gOpacity,
      exposure: EXPOSURE,
      dpr,
      width: W,
      height: H,
    });

    // Compose: dark background, the grid, then add the ray layer.
    ctx.fillStyle = "#06080d";
    ctx.fillRect(0, 0, W, H);
    draw2dGrid(W, H);

    // Add the device-resolution ray layer 1:1 (bypass the dpr transform).
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(rays.canvas, 0, 0);
    ctx.restore();

    // TIR highlight: a ring where a surface is in total internal reflection.
    if (showTir) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(255,120,80,0.9)";
      for (const p of tirDots) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.stroke();
      }
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

  // The centred square drawing region of the dock canvas (detectors are square-ish).
  function detSquare() {
    const s = Math.min(hist.width, hist.height);
    return { s, ox: (hist.width - s) / 2, oy: (hist.height - s) / 2 };
  }

  // The flux heatmap ramp (heat.ts), shared with the 3D view's detector faces.
  function heatColor(t: number): string {
    const [r, g, b] = heatRgb(t);
    return `rgb(${r},${g},${b})`;
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

  // 2D binning: paint the nu×nv incident-flux grid as a heatmap. Detector v
  // increases upward, canvas y downward, so the rows are flipped.
  function drawHeatmap(info: { bins: [number, number]; incident: number[] }) {
    const [nu, nv] = info.bins;
    const cells = info.incident;
    const max = Math.max(...cells, 1e-12);
    const { s, ox, oy } = detSquare();
    const cw = s / nu, ch = s / nv;
    for (let v = 0; v < nv; v++) {
      for (let u = 0; u < nu; u++) {
        hctx.fillStyle = heatColor(cells[v * nu + u] / max);
        hctx.fillRect(ox + u * cw, oy + (nv - 1 - v) * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    hctx.strokeStyle = "rgba(127,214,255,0.25)";
    hctx.strokeRect(ox + 0.5, oy + 0.5, s - 1, s - 1);
  }

  // Spot diagram: the actual ray-detector hits, scaled into the face and coloured
  // by wavelength (the linear-light rgb the tracer records, normalized to a dot).
  function drawSpot(detId: number) {
    const a = detId >= 0 ? scene.detectorHitsF32(detId) : scene.firstDetectorHitsF32();
    const { s, ox, oy } = detSquare();
    hctx.fillStyle = "#04060a";
    hctx.fillRect(ox, oy, s, s);
    hctx.strokeStyle = "rgba(127,214,255,0.25)";
    hctx.strokeRect(ox + 0.5, oy + 0.5, s - 1, s - 1);
    // Crosshair at the face centre.
    hctx.strokeStyle = "rgba(127,214,255,0.12)";
    hctx.beginPath();
    hctx.moveTo(ox + s / 2, oy); hctx.lineTo(ox + s / 2, oy + s);
    hctx.moveTo(ox, oy + s / 2); hctx.lineTo(ox + s, oy + s / 2);
    hctx.stroke();
    if (a.length < 3) return;
    const hw = a[0] || 1, hh = a[1] || 1, n = a[2];
    for (let i = 0; i < n; i++) {
      const k = 3 + i * 6;
      const u = a[k], v = a[k + 1];
      const r = a[k + 2], g = a[k + 3], b = a[k + 4];
      const m = Math.max(r, g, b, 1e-6);
      const px = ox + (u / hw * 0.5 + 0.5) * s;
      const py = oy + (1 - (v / hh * 0.5 + 0.5)) * s; // flip v to screen-down
      hctx.fillStyle = `rgb(${(r / m * 255) | 0},${(g / m * 255) | 0},${(b / m * 255) | 0})`;
      hctx.fillRect(px - 0.75, py - 0.75, 1.7, 1.7);
    }
  }

  function refresh3dDetector() {
    hctx.clearRect(0, 0, hist.width, hist.height);
    // Show the selected detector if one is selected, else the first — so a scene
    // with several detectors reports whichever the user is inspecting.
    const sel = selectedBody3d();
    const selInfo = sel >= 0 ? JSON.parse(scene.detectorResultJSON(sel)) : null;
    const detId = selInfo ? sel : -1;
    const info = selInfo ?? JSON.parse(scene.firstDetectorResultJSON());
    if (!info || info.hits === 0) {
      dockNums.textContent = info ? "no rays on detector" : "no detector";
      return;
    }
    if (detMode === "spot") {
      drawSpot(detId);
    } else if (info.measure === "intensity") {
      // Not a heatmap of anything on the face, so the label stops saying so.
      detHead.textContent = "detector — intensity, polar (f64)";
      drawPolar(info);
    } else {
      detHead.textContent = "detector — heatmap (f64)";
      drawHeatmap(info);
    }
    const c = info.centroid ? `(${info.centroid[0].toFixed(2)}, ${info.centroid[1].toFixed(2)})` : "—";
    const opl = info.meanOpl == null ? "—" : Number(info.meanOpl).toFixed(2);
    const tag = detId >= 0 ? `#${detId} · ` : "";
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
      reads = ` · peak ${Number(info.peak).toPrecision(3)} ${info.unit}`;
    } else if (info.measure === "intensity") {
      const share = Number(info.totalIncident) > 0 ? Number(info.outsideCone) / Number(info.totalIncident) : 0;
      const past = share > 1e-9 ? `, ${(share * 100).toFixed(0)}% past ${Number(info.coneDeg).toFixed(0)}°` : "";
      reads = ` · peak ${Number(info.peak).toPrecision(3)} ${info.unit}${past}`;
    }
    dockNums.textContent = `${tag}hits ${info.hits} · flux ${Number(info.totalIncident).toFixed(3)}${reads} · centroid ${c} mm · OPL ${opl} mm${pol}`;
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
    scene, view, toScreen, toWorld, render: renderView, fast,
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
      fastProjected: () => new Promise((resolve) => fast.requestProjected(scene, resolve)),
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

  drawAbbe();
  renderView();
  refreshReference();
}

main().catch((e) => {
  document.body.innerHTML = `<pre style="padding:2rem;color:#f88">Failed to start: ${e}\n\nBuild the wasm first:  npm run wasm</pre>`;
  console.error(e);
});
