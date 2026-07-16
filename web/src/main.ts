// Optiland Canvas — M2 web shell ("it's optics").
//
// Scene lives in Rust (§2). This file holds UI state only. Two tracers, per the
// spec's tiering (§4.1): the interactive f32 path draws the picture every frame
// (spectral colour, ghost layer, TIR glow); the f64 reference feeds the
// detector dock's *numbers*, on idle.

import init, { Canvas } from "../pkg/oc_wasm.js";
import { initM3 } from "./m3";
import { initM4, type M4 } from "./m4";
import { RayLayer } from "./rays";
import { initOpenUc2 } from "./features/openuc2/OpenUc2Page";

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
  // M4 module (layers / undo tree / wavefront scrub / variation gallery).
  let m4: M4 | null = null;

  // Layer cache the m3 overlay reads each frame (avoids re-parsing layer JSON per
  // element); refreshed whenever a layer changes. Also feeds onion-skinning.
  type LayerInfo = { id: number; visible: boolean; opacity: number };
  let layerCache: LayerInfo[] = JSON.parse(scene.layersJson());
  const refreshLayerCache = () => { layerCache = JSON.parse(scene.layersJson()); };
  const onionBox = document.getElementById("onionBox") as HTMLInputElement;

  // Snapshot the settled scene into the undo tree (§5.2). Deduped in Rust, so
  // it is safe to call from every settle point; a real change also invalidates
  // the wavefront cache and repaints the M4 panels.
  function autoCommit(label: string) {
    const id = scene.commit(label);
    if (id >= 0) m4?.onSceneChanged();
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

  // --- interactive view -------------------------------------------------
  function renderView() {
    const t0 = performance.now();
    // 9 floats/segment: ax ay bx by r g b flux flags. In wavefront-scrub mode
    // (§5.5) the segments come from the reference tracer, clipped at the scrub
    // time; otherwise the interactive f32 path draws the live picture.
    const wavefront = m4?.wavefrontActive() ?? false;
    const segs = wavefront ? m4!.wavefrontSegments() : scene.traceColored();
    const traceMs = performance.now() - t0;
    const gOpacity = Number(ghostOpacity.value) / 100;
    const showTir = tirBox.checked;
    const W = w(), H = h();
    const dpr = window.devicePixelRatio || 1;
    const { count, tirDots } = rays.draw(segs, toScreen, {
      ghostOpacity: gOpacity,
      exposure: EXPOSURE,
      dpr,
      width: W,
      height: H,
    });

    // Compose: dark background, faint axis + detector, then add the ray layer.
    ctx.fillStyle = "#06080d";
    ctx.fillRect(0, 0, W, H);
    // Optical axis is world y = 0, so it pans/zooms with the scene (not fixed
    // at screen centre — otherwise a stray line lingers after panning).
    const axisY = toScreen({ x: 0, y: 0 }).y;
    ctx.strokeStyle = "rgba(120,140,170,0.16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, axisY);
    ctx.lineTo(W, axisY);
    ctx.stroke();
    const d0 = toScreen({ x: 52, y: -7 });
    const d1 = toScreen({ x: 52, y: 7 });
    ctx.strokeStyle = "rgba(127,214,255,0.4)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(d0.x, d0.y);
    ctx.lineTo(d1.x, d1.y);
    ctx.stroke();

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

    // M3 overlay: element outlines, selection handles, snap guides, and the
    // freehand fit preview. It owns on-canvas manipulation, so the old fixed
    // lens-handle ellipse is retired in its favour.
    m3?.overlay(ctx);
    // M4 overlay: the sweeping wavefront front (§5.5), drawn on top.
    m4?.overlay(ctx);

    const mode = wavefront ? " · wavefront scrub" : "";
    const ms = performance.now() - t0;
    stat.textContent = `${count} segs · ${ms.toFixed(1)} ms (trace ${traceMs.toFixed(1)}) · ${committed.name} nd ${committed.nd.toFixed(4)} V ${committed.vd.toFixed(1)}${mode}`;
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
    // committed marker
    actx.strokeStyle = "#ffd678";
    actx.lineWidth = 1.5;
    actx.beginPath();
    actx.arc(abX(committed.vd), abY(committed.nd), 4, 0, Math.PI * 2);
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
      scene.setElementAbbe(lensId, nd, vd); // live preview: chromatic spread re-converges
      renderView();
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
      applyGlass(snap ?? { name: "custom", nd, vd });
      renderView();
      refreshReference();
      autoCommit("glass"); // undo-tree snapshot (§5.2)
    }
  });
  abbe.addEventListener("pointerleave", () => {
    applyGlass(committed); // revert preview to the committed glass
    drawAbbe();
    renderView();
    refreshReference();
  });

  // --- detector dock (f64 reference numbers, on idle) --------------------
  function refreshReference() {
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

  // --- controls ----------------------------------------------------------
  ghostsBox.addEventListener("change", () => { scene.setGhosts(ghostsBox.checked); m4?.onSceneChanged(); renderView(); });
  ghostOpacity.addEventListener("input", renderView);
  tirBox.addEventListener("change", renderView);
  // Ghost bounce cap: retrace live; the reference dock + wavefront recompute too.
  ghostBounces.addEventListener("change", () => {
    scene.setMaxGhostBounces(Number(ghostBounces.value) || 2);
    m4?.onSceneChanged(); // invalidate the wavefront (ghost depth changed)
    renderView();
    refreshReference();
  });
  // Ray density: source 0 is the beam. Live retrace while sliding; the reference
  // dock (heavier) recomputes only on release.
  density.addEventListener("input", () => { scene.setSourceRays(0, Number(density.value)); renderView(); });
  density.addEventListener("change", () => { refreshReference(); autoCommit("ray density"); });

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
    // M4 hooks (§5.2/§5.6).
    commit: autoCommit,
    layerVisible: (id) => layerCache[id]?.visible ?? true,
    layerOpacity: (id) => layerCache[id]?.opacity ?? 1,
    onionSkin: () => onionBox.checked,
  });
  m4 = initM4({
    scene,
    view,
    toScreen,
    render: renderView,
    refresh: refreshReference,
    commit: autoCommit,
    getSelected: () => m3!.getSelected(),
    syncM3: () => m3!.sync(),
    refreshLayers: refreshLayerCache,
  });
  // Freeze the demo scene as the undo-tree root so undo doesn't walk back into
  // the empty start document (§5.2).
  scene.historyReset("scene");
  m4.refreshPanels();

  // --- M5: save / open .ocanvas — the Optiland handoff (§6) ---------------
  // Save downloads the frozen .ocanvas document; Open loads one back (Rust
  // version-checks it and re-roots the undo history). This closes the loop for a
  // non-Python user: sketch here, save, then `optiland_canvas.to_optic(file)`.
  const saveBtn = document.getElementById("saveScene") as HTMLButtonElement;
  const openBtn = document.getElementById("openScene") as HTMLButtonElement;
  const openFile = document.getElementById("openSceneFile") as HTMLInputElement;
  const sceneMsg = document.getElementById("sceneMsg")!;
  const flash = (msg: string, ok = true) => {
    sceneMsg.textContent = msg;
    sceneMsg.style.color = ok ? "#7fffa0" : "#ff8a6b";
  };

  saveBtn.addEventListener("click", () => {
    const json = scene.toJSON();
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `scene-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.ocanvas`;
    a.click();
    URL.revokeObjectURL(url);
    flash(`saved ${a.download} · ${(json.length / 1024).toFixed(1)} kB`);
  });

  openBtn.addEventListener("click", () => openFile.click());
  openFile.addEventListener("change", async () => {
    const file = openFile.files?.[0];
    if (!file) return;
    try {
      scene.loadJSON(await file.text()); // version-checked in Rust; throws on bad input
      refreshLayerCache();
      m3?.sync();
      m4?.onSceneChanged(); // invalidate the wavefront cache + repaint M4 panels
      drawAbbe();
      renderView();
      refreshReference();
      flash(`opened ${file.name} · ${scene.elementCount()} element(s)`);
    } catch (err) {
      flash(`could not open ${file.name}: ${err}`, false);
    } finally {
      openFile.value = ""; // let the same file be re-opened
    }
  });

  drawAbbe();
  renderView();
  refreshReference();
}

main().catch((e) => {
  document.body.innerHTML = `<pre style="padding:2rem;color:#f88">Failed to start: ${e}\n\nBuild the wasm first:  npm run wasm</pre>`;
  console.error(e);
});
