// Optiland Canvas — M4 web module ("it's a workflow").
//
// Everything the M4 milestone adds, kept in its own module (like m3.ts) so the
// M2/M3 pipeline stays intact. Scene state still lives in Rust (§2); this file
// is UI only and drives the opaque handle.
//
//   • Layers            — show/hide + opacity groups; assign the selection to a
//                         layer, solo/flip between candidate architectures (§5.6)
//   • Onion-skinning    — hidden layers drawn as faint ghosts for comparison (§5.6)
//   • Undo tree         — non-linear history; every settled edit snapshots the
//                         scene; branches are clickable in a sidebar (§5.2)
//   • Variation gallery — saved history nodes as live, traced thumbnails with a
//                         synced-detector readout; click to load a variant (§5.6)
//   • Wavefront scrub   — a timeline; rays are drawn only up to propagation
//                         time t, clipped by cumulative optical path length (§5.5)

import type { Canvas } from "../pkg/oc_wasm.js";

type V2 = { x: number; y: number };
type LayerInfo = { id: number; name: string; visible: boolean; opacity: number; count: number };
type HistNode = { id: number; seq: number; parent: number; label: string; children: number[] };
type Hist = { current: number; nodes: HistNode[] };

export interface App4 {
  scene: Canvas;
  view: HTMLCanvasElement;
  toScreen: (p: V2) => V2;
  render: () => void; // main.renderView
  refresh: () => void; // main.refreshReference (f64 detector numbers)
  commit: (label: string) => void; // main.autoCommit → scene.commit + onSceneChanged
  getSelected: () => number; // m3's current selection (−1 if none)
  syncM3: () => void; // re-sync m3 cards/selection after a jump
  refreshLayers: () => void; // main re-reads its layer cache (feeds the m3 overlay)
}

export interface M4 {
  overlay: (ctx: CanvasRenderingContext2D) => void;
  wavefrontActive: () => boolean;
  /** Clipped render segments (9 floats each) at the current scrub time. */
  wavefrontSegments: () => Float32Array;
  /** A geometry change happened — invalidate the wavefront cache + repaint panels. */
  onSceneChanged: () => void;
  /** Repaint every M4 panel (layers, history, gallery). */
  refreshPanels: () => void;
}

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function initM4(app: App4): M4 {
  const { scene } = app;

  // ======================= layers (§5.6) ================================
  const layerList = el("layerList");
  const onionBox = el<HTMLInputElement>("onionBox");

  function layers(): LayerInfo[] {
    return JSON.parse(scene.layersJson());
  }

  function renderLayers() {
    const ls = layers();
    const active = scene.activeLayer();
    layerList.textContent = "";
    for (const l of ls) {
      const row = document.createElement("div");
      row.className = "layer-row" + (l.id === active ? " active" : "");

      const vis = document.createElement("button");
      vis.className = "layer-vis";
      vis.title = l.visible ? "hide layer" : "show layer";
      vis.textContent = l.visible ? "👁" : "◌";
      vis.onclick = (e) => {
        e.stopPropagation();
        scene.setLayerVisible(l.id, !l.visible);
        afterLayerChange();
      };

      const name = document.createElement("span");
      name.className = "layer-name";
      name.textContent = `${l.name}`;
      const count = document.createElement("span");
      count.className = "layer-count";
      count.textContent = String(l.count);
      // Double-click the name to rename.
      name.ondblclick = (e) => {
        e.stopPropagation();
        const nn = prompt("Rename layer", l.name);
        if (nn != null) { scene.renameLayer(l.id, nn); afterLayerChange(); }
      };

      const op = document.createElement("input");
      op.type = "range";
      op.className = "layer-op";
      op.min = "0"; op.max = "100"; op.value = String(Math.round(l.opacity * 100));
      op.title = "layer opacity (onion-skin)";
      op.oninput = () => { scene.setLayerOpacity(l.id, Number(op.value) / 100); app.refreshLayers(); app.render(); };
      op.onchange = () => app.commit("layer opacity");
      op.onclick = (e) => e.stopPropagation();

      const del = document.createElement("button");
      del.className = "layer-del";
      del.title = "remove layer (merges its optics into layer 0)";
      del.textContent = "✕";
      del.onclick = (e) => {
        e.stopPropagation();
        if (scene.removeLayer(l.id)) afterLayerChange();
      };

      // Click the row → make this the active layer (new optics land here).
      row.onclick = () => { scene.setActiveLayer(l.id); renderLayers(); };

      row.append(vis, name, count, op, del);
      layerList.appendChild(row);
    }
  }

  function afterLayerChange() {
    app.refreshLayers(); // main's cache feeds the m3 overlay
    renderLayers();
    app.render();
    app.refresh();
    app.commit("layers");
  }

  el<HTMLButtonElement>("addLayer").onclick = () => {
    const id = scene.addLayer("");
    scene.setActiveLayer(id);
    afterLayerChange();
  };
  // Assign the current selection to the active layer.
  el<HTMLButtonElement>("assignLayer").onclick = () => {
    const sel = app.getSelected();
    if (sel < 0) return;
    if (scene.setElementLayer(sel, scene.activeLayer())) afterLayerChange();
  };
  onionBox.onchange = () => app.render();

  // ======================= undo tree (§5.2) =============================
  // Rendered as a small "subway map": every dot is a saved state, the gold dot
  // is where you are, the gold line is how you got here (the lineage from the
  // start), and a fork appears whenever you edit after an undo — so nothing is
  // ever lost. Hover a dot to read its label; click to jump there.
  const histCanvas = el<HTMLCanvasElement>("histCanvas");
  const hctx = histCanvas.getContext("2d")!;
  const histLabel = el("histLabel");
  const undoBtn = el<HTMLButtonElement>("undoBtn");
  const redoBtn = el<HTMLButtonElement>("redoBtn");
  const DX = 24, DY = 26, R = 6, PAD = 16; // tree layout metrics (px)
  let nodePos: Map<number, V2> = new Map(); // grid coords (col, row)
  let nodeScreen: Map<number, V2> = new Map(); // pixel centres, for hit-testing
  let hovered = -1;

  // Tidy pre-order layout: row = depth from root, column = pre-order visit index.
  function layoutTree(h: Hist) {
    nodePos = new Map();
    const root = h.nodes.find((n) => n.parent < 0);
    if (!root) return { w: 0, rows: 0 };
    let col = 0, maxCol = 0, maxRow = 0;
    const visit = (id: number, d: number) => {
      const n = h.nodes[id];
      if (n.children.length === 0) {
        nodePos.set(id, { x: col, y: d });
        maxCol = Math.max(maxCol, col); maxRow = Math.max(maxRow, d);
        col++;
        return;
      }
      const xs: number[] = [];
      for (const c of n.children) { visit(c, d + 1); xs.push(nodePos.get(c)!.x); }
      // Parent sits above the mean of its children — reads as a proper fork.
      const x = xs.reduce((a, b) => a + b, 0) / xs.length;
      nodePos.set(id, { x, y: d });
      maxRow = Math.max(maxRow, d);
    };
    visit(root.id, 0);
    return { w: maxCol, rows: maxRow };
  }

  // The chain of nodes from the root down to `id` — the "you are here" lineage.
  function lineage(h: Hist, id: number): Set<number> {
    const path = new Set<number>();
    let cur = id;
    while (cur >= 0 && h.nodes[cur]) { path.add(cur); cur = h.nodes[cur].parent; }
    return path;
  }

  function renderHistory() {
    const h: Hist = JSON.parse(scene.historyJson());
    const { w, rows } = layoutTree(h);
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(PAD * 2 + w * DX + 2 * R, 190);
    const cssH = Math.max(PAD * 2 + rows * DY + 2 * R, 44);
    histCanvas.width = cssW * dpr; histCanvas.height = cssH * dpr;
    histCanvas.style.width = `${cssW}px`; histCanvas.style.height = `${cssH}px`;
    hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    hctx.clearRect(0, 0, cssW, cssH);
    hctx.font = "9px ui-monospace, monospace";
    const at = (p: V2): V2 => ({ x: PAD + p.x * DX + R, y: PAD + p.y * DY + R });
    nodeScreen = new Map();
    for (const [id, p] of nodePos) nodeScreen.set(id, at(p));
    const path = lineage(h, h.current);

    // edges — the current lineage brighter than the branches you left behind.
    for (const n of h.nodes) {
      if (n.parent < 0) continue;
      const a = nodeScreen.get(n.parent)!, b = nodeScreen.get(n.id)!;
      const onPath = path.has(n.id) && path.has(n.parent);
      hctx.strokeStyle = onPath ? "rgba(255,214,120,0.7)" : "rgba(127,214,255,0.28)";
      hctx.lineWidth = onPath ? 1.7 : 1.1;
      hctx.beginPath(); hctx.moveTo(a.x, a.y); hctx.lineTo(b.x, b.y); hctx.stroke();
    }
    // nodes + sequence numbers
    for (const n of h.nodes) {
      const p = nodeScreen.get(n.id)!;
      const cur = n.id === h.current;
      if (n.id === hovered) {
        hctx.beginPath(); hctx.arc(p.x, p.y, R + 3, 0, Math.PI * 2);
        hctx.strokeStyle = "rgba(232,246,255,0.85)"; hctx.lineWidth = 1; hctx.stroke();
      }
      hctx.beginPath(); hctx.arc(p.x, p.y, cur ? R : R - 1.5, 0, Math.PI * 2);
      hctx.fillStyle = cur ? "#ffd678" : path.has(n.id) ? "#2a6f96" : "#163049";
      hctx.strokeStyle = cur ? "#ffd678" : "rgba(127,214,255,0.6)";
      hctx.fill(); hctx.lineWidth = 1.4; hctx.stroke();
      hctx.fillStyle = "rgba(205,214,228,0.5)";
      hctx.fillText(String(n.seq), p.x + R + 2, p.y + 3);
    }
    const show = hovered >= 0 ? h.nodes[hovered] : h.nodes[h.current];
    histLabel.textContent = show ? `${hovered >= 0 ? "jump to" : "at"} #${show.seq} · ${show.label}` : "";
    undoBtn.disabled = !scene.canUndo();
    redoBtn.disabled = !scene.canRedo();
  }

  function nodeAtPixel(mx: number, my: number): number {
    let best = -1, bestD = 13 * 13;
    for (const [id, p] of nodeScreen) {
      const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
      if (d < bestD) { bestD = d; best = id; }
    }
    return best;
  }
  const histXY = (e: MouseEvent): [number, number] => {
    const r = histCanvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  histCanvas.addEventListener("mousemove", (e) => {
    const id = nodeAtPixel(...histXY(e));
    if (id !== hovered) { hovered = id; histCanvas.style.cursor = id >= 0 ? "pointer" : "default"; renderHistory(); }
  });
  histCanvas.addEventListener("mouseleave", () => { if (hovered !== -1) { hovered = -1; renderHistory(); } });
  histCanvas.addEventListener("click", (e) => {
    const id = nodeAtPixel(...histXY(e));
    if (id >= 0 && id !== scene.currentHistoryNode()) { scene.historyJumpTo(id); afterRestore(); }
  });

  undoBtn.onclick = () => { if (scene.undo()) afterRestore(); };
  redoBtn.onclick = () => { if (scene.redo()) afterRestore(); };

  // Restore path shared by undo / redo / jump: the scene changed under the whole
  // UI, so re-read layers, re-sync m3, invalidate the wavefront, repaint.
  function afterRestore() {
    app.refreshLayers();
    app.syncM3();
    onSceneChanged();
    app.render();
    app.refresh();
  }

  // Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y) — m3 owns Ctrl+C/V/X/D, so these are free.
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return;
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) return;
    const k = e.key.toLowerCase();
    if (k === "z" && !e.shiftKey) { if (scene.undo()) afterRestore(); e.preventDefault(); }
    else if ((k === "z" && e.shiftKey) || k === "y") { if (scene.redo()) afterRestore(); e.preventDefault(); }
  });

  // ==================== variation gallery (§5.6) ========================
  const galleryGrid = el("galleryGrid");
  let variants: number[] = [];

  el<HTMLButtonElement>("saveVariant").onclick = () => {
    app.commit("variant"); // ensure the current state is a committed node
    const id = scene.currentHistoryNode();
    if (!variants.includes(id)) variants.push(id);
    renderGallery();
  };

  function renderGallery() {
    galleryGrid.textContent = "";
    if (variants.length === 0) {
      const hint = document.createElement("div");
      hint.className = "gallery-hint";
      hint.textContent = "save the current design as a variant to compare it live";
      galleryGrid.appendChild(hint);
      return;
    }
    const current = scene.currentHistoryNode();
    for (const id of variants) {
      const summary = JSON.parse(scene.nodeSummaryJson(id));
      if (!summary) continue;
      const cell = document.createElement("div");
      cell.className = "variant" + (id === current ? " current" : "");

      const cv = document.createElement("canvas");
      cv.width = 168; cv.height = 96; cv.className = "variant-thumb";
      drawThumb(cv, Array.from(scene.traceNodeColored(id)));

      const info = document.createElement("div");
      info.className = "variant-info";
      const det = JSON.parse(scene.nodeDetectorJson(id, 0));
      const cen = det && det.centroid != null ? (det.centroid * 14 - 7).toFixed(2) : "—";
      const hits = det ? det.hits : 0;
      info.innerHTML = `<b>#${summary.seq}</b> ${escapeHtml(summary.label)}<br>`
        + `<span class="dim">${summary.elements} el · det y ${cen} mm · ${hits} hits</span>`;

      const close = document.createElement("button");
      close.className = "variant-del";
      close.textContent = "✕";
      close.title = "remove from gallery";
      close.onclick = (e) => { e.stopPropagation(); variants = variants.filter((v) => v !== id); renderGallery(); };

      cell.append(cv, info, close);
      cell.onclick = () => { scene.historyJumpTo(id); afterRestore(); };
      galleryGrid.appendChild(cell);
    }
  }

  // A cheap self-contained thumbnail: fit all ray segments to the canvas and
  // draw them additively (no HDR) — the ray bundle's shape *is* the comparison.
  function drawThumb(cv: HTMLCanvasElement, segs: number[]) {
    const c = cv.getContext("2d")!;
    c.fillStyle = "#04060a";
    c.fillRect(0, 0, cv.width, cv.height);
    if (segs.length < 9) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, maxFlux = 1e-9;
    for (let i = 0; i + 8 < segs.length; i += 9) {
      // Clamp the scene-exit rays (which run to ±1e4) so bounds stay sane.
      for (const [px, py] of [[segs[i], segs[i + 1]], [segs[i + 2], segs[i + 3]]]) {
        if (Math.abs(px) > 400 || Math.abs(py) > 400) continue;
        x0 = Math.min(x0, px); y0 = Math.min(y0, py); x1 = Math.max(x1, px); y1 = Math.max(y1, py);
      }
      if (segs[i + 7] > maxFlux) maxFlux = segs[i + 7];
    }
    if (!isFinite(x0)) return;
    const bw = Math.max(x1 - x0, 1), bh = Math.max(y1 - y0, 1);
    const s = 0.9 * Math.min(cv.width / bw, cv.height / bh);
    const ox = (cv.width - s * (x0 + x1)) / 2, oy = (cv.height + s * (y0 + y1)) / 2;
    const tx = (wx: number) => ox + s * wx;
    const ty = (wy: number) => oy - s * wy; // world +y is up
    const g = (v: number) => Math.round(255 * Math.min(1, Math.pow(Math.max(0, v), 1 / 2.2)));
    c.globalCompositeOperation = "lighter";
    c.lineWidth = 1;
    for (let i = 0; i + 8 < segs.length; i += 9) {
      const frac = Math.min(1, segs[i + 7] / maxFlux);
      if (frac < 1e-3) continue;
      const r = segs[i + 4] * frac, gg = segs[i + 5] * frac, b = segs[i + 6] * frac;
      c.strokeStyle = `rgb(${g(r)},${g(gg)},${g(b)})`;
      c.beginPath(); c.moveTo(tx(segs[i]), ty(segs[i + 1])); c.lineTo(tx(segs[i + 2]), ty(segs[i + 3])); c.stroke();
    }
    c.globalCompositeOperation = "source-over";
  }

  // ==================== wavefront scrub (§5.5) ==========================
  const wfToggle = el<HTMLInputElement>("wfToggle");
  const wfSlider = el<HTMLInputElement>("wfSlider");
  const wfPlay = el<HTMLButtonElement>("wfPlay");
  const wfLabel = el("wfLabel");
  const wfBar = el("wavefront");

  type WfVert = { x: number; y: number; opl: number };
  let wfRaw: Float32Array | null = null; // 11 floats/seg from traceWavefront
  let wfMax = 0; // largest opl1 (mm) — the timeline's end
  let wfDirty = true;
  let wfT = 0; // scrub position (mm of OPL)
  let wfRays: WfVert[][] = []; // per-ray primary polylines w/ cumulative OPL
  let playing = false;
  let lastTs = 0;

  const wavefrontActive = () => wfToggle.checked;

  function ensureWf() {
    if (!wfDirty && wfRaw) return;
    wfRaw = scene.traceWavefront();
    wfRays = parseCurves(scene.wavefrontCurves());
    let m = 0;
    for (let i = 10; i < wfRaw.length; i += 11) if (wfRaw[i] > m && isFinite(wfRaw[i])) m = wfRaw[i];
    // Ignore the huge scene-exit tails so the slider spans the useful range.
    wfMax = m > 0 ? Math.min(m, usefulMax(wfRaw)) : 1;
    wfDirty = false;
    wfSlider.max = wfMax.toFixed(3);
    if (wfT > wfMax || wfT <= 0) wfT = wfMax;
    wfSlider.value = String(wfT);
  }

  // Decode `[nrays, len, (x,y,opl)×len, …]` into per-ray vertex lists.
  function parseCurves(a: Float32Array): WfVert[][] {
    const rays: WfVert[][] = [];
    let i = 0;
    const nrays = a[i++] | 0;
    for (let r = 0; r < nrays && i < a.length; r++) {
      const len = a[i++] | 0;
      const pts: WfVert[] = [];
      for (let k = 0; k < len; k++) { pts.push({ x: a[i], y: a[i + 1], opl: a[i + 2] }); i += 3; }
      rays.push(pts);
    }
    return rays;
  }

  // The point where a ray's primary path reaches OPL = t (null once it is past
  // the ray's end — that light has already gone by).
  function pointAtOpl(ray: WfVert[], t: number): V2 | null {
    if (ray.length === 0) return null;
    if (t <= ray[0].opl) return { x: ray[0].x, y: ray[0].y };
    for (let i = 0; i < ray.length - 1; i++) {
      const a = ray[i], b = ray[i + 1];
      if (t >= a.opl && t <= b.opl) {
        const f = b.opl > a.opl ? (t - a.opl) / (b.opl - a.opl) : 0;
        return { x: a.x + f * (b.x - a.x), y: a.y + f * (b.y - a.y) };
      }
    }
    return null;
  }

  // The 95th-percentile-ish opl among segments whose end actually hits geometry
  // (opl1 finite and within the scene) — keeps a single stray exit ray from
  // stretching the whole timeline.
  function usefulMax(raw: Float32Array): number {
    const ends: number[] = [];
    for (let i = 0; i + 10 < raw.length; i += 11) {
      // A segment whose length is enormous is a scene-exit ray; cap its opl1.
      const dx = raw[i + 2] - raw[i], dy = raw[i + 3] - raw[i + 1];
      if (Math.hypot(dx, dy) < 500) ends.push(raw[i + 10]);
    }
    if (ends.length === 0) return raw.length ? raw[10] : 1;
    ends.sort((a, b) => a - b);
    return ends[Math.min(ends.length - 1, Math.floor(ends.length * 0.98))];
  }

  // Clip every segment at OPL = wfT → 9-float render segments (the rays drawn
  // only up to propagation time t). The wavefront *curve* itself is drawn in
  // overlay() from the per-ray polylines.
  function wavefrontSegments(): Float32Array {
    ensureWf();
    const raw = wfRaw!;
    const out: number[] = [];
    for (let i = 0; i + 10 < raw.length; i += 11) {
      const opl0 = raw[i + 9], opl1 = raw[i + 10];
      if (opl0 > wfT) continue; // the front hasn't reached this segment yet
      const span = opl1 - opl0;
      const frac = span > 1e-9 ? Math.min(1, (wfT - opl0) / span) : 1;
      const ax = raw[i], ay = raw[i + 1];
      const ex = ax + frac * (raw[i + 2] - ax), ey = ay + frac * (raw[i + 3] - ay);
      out.push(ax, ay, ex, ey, raw[i + 4], raw[i + 5], raw[i + 6], raw[i + 7], 0);
    }
    updateWfLabel();
    return new Float32Array(out);
  }

  function updateWfLabel() {
    wfLabel.textContent = `t = ${wfT.toFixed(1)} / ${wfMax.toFixed(1)} mm OPL`;
  }

  wfToggle.onchange = () => {
    const on = wfToggle.checked;
    wfBar.classList.toggle("wf-on", on);
    wfSlider.disabled = !on;
    wfPlay.disabled = !on;
    if (on) { wfDirty = true; ensureWf(); }
    else { stopPlay(); }
    app.render();
  };
  wfSlider.oninput = () => { wfT = Number(wfSlider.value); app.render(); };
  wfPlay.onclick = () => { playing ? stopPlay() : startPlay(); };

  function startPlay() {
    ensureWf();
    playing = true; wfPlay.textContent = "❚❚";
    if (wfT >= wfMax) wfT = 0;
    lastTs = 0;
    requestAnimationFrame(tick);
  }
  function stopPlay() { playing = false; wfPlay.textContent = "▶"; }
  function tick(ts: number) {
    if (!playing) return;
    if (lastTs === 0) lastTs = ts;
    const dt = (ts - lastTs) / 1000; lastTs = ts;
    wfT += (wfMax / 2.4) * dt; // sweep the full front in ~2.4 s
    if (wfT >= wfMax) { wfT = wfMax; app.render(); stopPlay(); return; } // stop at the end
    wfSlider.value = String(wfT);
    app.render();
    requestAnimationFrame(tick);
  }

  // ===================== overlay + lifecycle ============================
  function overlay(ctx: CanvasRenderingContext2D) {
    if (!wavefrontActive() || wfRays.length === 0) return;
    // The wavefront is the surface of constant OPL: join every ray's OPL = wfT
    // point across the fan. It is flat in the collimated beam and curves as the
    // rays converge/aberrate — the picture *is* the physics (§5.5).
    const pts: Array<V2 | null> = wfRays.map((ray) => pointAtOpl(ray, wfT));
    ctx.save();
    // Connect consecutive active rays (break the line across any that have
    // already passed, so no false chord jumps a gap).
    ctx.strokeStyle = "rgba(255,238,180,0.9)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let pen = false;
    for (const p of pts) {
      if (!p) { pen = false; continue; }
      const s = app.toScreen(p);
      if (pen) ctx.lineTo(s.x, s.y);
      else { ctx.moveTo(s.x, s.y); pen = true; }
    }
    ctx.stroke();
    // A dot at each front point for emphasis.
    ctx.fillStyle = "rgba(255,238,180,0.95)";
    for (const p of pts) {
      if (!p) continue;
      const s = app.toScreen(p);
      ctx.beginPath(); ctx.arc(s.x, s.y, 1.8, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  function onSceneChanged() {
    wfDirty = true;
    refreshPanels();
  }
  function refreshPanels() {
    renderLayers();
    renderHistory();
    renderGallery();
  }

  refreshPanels();
  wfSlider.disabled = true; wfPlay.disabled = true;
  return { overlay, wavefrontActive, wavefrontSegments, onSceneChanged, refreshPanels };
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] || c));
}
