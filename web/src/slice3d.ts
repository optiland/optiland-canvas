// PR 3D-09 — the 2D Canvas as a slice/projection of the canonical 3D scene.
//
// The freeform demo above stays a v1 scene. This module is the *view over 3D
// state*: it opens an `.ocanvas` (v1 migrated, or v2) into the canonical
// `Scene3` held in Rust, reads the preview-mode badge (§10.3), and draws the
// authoritative **f64** ray paths projected onto the Z=0 slice. No scene graph
// is mirrored in JS — the projected segments come straight from Rust
// (`traceProjected3D`), and every redraw re-reads them.
//
// Slicing (§10.4): each segment carries the signed distance of its endpoints
// from the plane, so a half-thickness slider hides paths that never come within
// the slab. Thickness 0 means "show everything projected".

import type { Canvas } from "../pkg/oc_wasm.js";

type V2 = { x: number; y: number };

export type Slice3d = {
  /** Is a 3D scene loaded and being previewed? (Suppresses the v1 demo rays.) */
  active: () => boolean;
  /** Draw the projected f64 paths for the current pan/zoom. */
  overlay: (ctx: CanvasRenderingContext2D, toScreen: (p: V2) => V2) => void;
};

// The flat layout of `traceProjected3D`: 9 floats per segment.
const STRIDE = 9;

/** A coarse visible-wavelength (µm) → CSS rgb, good enough to tell colours apart. */
function wavelengthColor(um: number): [number, number, number] {
  const nm = um * 1000;
  let r = 0, g = 0, b = 0;
  if (nm < 440) { r = (440 - nm) / (440 - 380); b = 1; }
  else if (nm < 490) { g = (nm - 440) / (490 - 440); b = 1; }
  else if (nm < 510) { g = 1; b = (510 - nm) / (510 - 490); }
  else if (nm < 580) { r = (nm - 510) / (580 - 510); g = 1; }
  else if (nm < 645) { r = 1; g = (645 - nm) / (645 - 580); }
  else { r = 1; }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export function initSlice3d(opts: {
  scene: Canvas;
  view: HTMLCanvasElement;
  toScreen: (p: V2) => V2;
  toWorld: (px: number, py: number) => V2;
  render: () => void;
}): Slice3d {
  const { scene, view, toScreen, toWorld, render } = opts;

  const badge = document.getElementById("preview3dBadge")!;
  const openBtn = document.getElementById("open3d") as HTMLButtonElement;
  const openFile = document.getElementById("open3dFile") as HTMLInputElement;
  const msg = document.getElementById("preview3dMsg")!;
  const sliceRow = document.getElementById("slice3dRow")!;
  const sliceInput = document.getElementById("slice3d") as HTMLInputElement;
  const sliceLabel = document.getElementById("slice3dLabel")!;
  const objects = document.getElementById("objects3d")!;

  let loaded = false;
  // The projected segments for the current scene + view plane, cached so pan/zoom
  // and slice edits redraw without re-running the f64 tracer. Refreshed on load.
  let segs: Float32Array<ArrayBufferLike> = new Float32Array(0);
  // The scene revision the cached projection was traced at (§3.3). The overlay
  // refuses to draw a projection older than the scene: a mutation bumps the
  // revision, so a cached result that predates it is re-traced before it paints.
  // (Synchronous today; the same guard rejects a stale *async* result once the
  // GPU path lands.)
  let tracedRevision = -1;
  // Slice half-thickness in mm; 0 = show everything projected.
  let slice = 0;

  // The bodies, cached with their world position and a coarse radius, for
  // pointer hit-testing. Refreshed whenever the scene changes.
  type Obj = { id: number; kind: string; construction: string; is3d: boolean; pos: number[]; radius: number };
  let objList: Obj[] = [];
  // The elements' cross-section with the view plane, as polylines to draw. Kept
  // in step with the geometry (load, drag).
  type XLine = { kind: string; closed: boolean; pts: number[] };
  let xsec: XLine[] = [];
  // The body being dragged, and the grab offset (view-plane u,v) so it does not
  // jump to the cursor. `origZ` keeps the out-of-plane coordinate fixed — an
  // in-plane move (§10.5).
  let dragBody: number | null = null;
  let grabU = 0, grabV = 0, origZ = 0;

  const flash = (text: string, ok = true) => {
    msg.textContent = text;
    msg.style.color = ok ? "#7fffa0" : "#ff8a6b";
  };

  const badgeText: Record<string, string> = {
    full_3d: "full 3D (projected)",
    exact_planar: "exact planar reduction",
    legacy_planar: "legacy planar",
    none: "no 3D scene",
  };

  function refreshBadge() {
    const mode = scene.previewMode();
    badge.className = `badge3d ${mode}`;
    badge.textContent = badgeText[mode] ?? mode;
  }

  // `fast` uses the interactive f32 path (during a drag); the default settles on
  // the authoritative f64 reference.
  function retrace(fast = false) {
    segs = fast ? scene.traceProjected3DFast() : scene.traceProjected3D();
    tracedRevision = scene.sceneRevision();
  }

  // The element outlines follow the geometry: refreshed on load and on every drag
  // step so a moved lens's section moves with it.
  function refreshGeometry() {
    xsec = JSON.parse(scene.sceneCrossSectionJSON());
  }

  // The object inspector: each body's construction and world transform, so its
  // 3D shape and any missing depth are visible (§13). Re-read from Rust — never
  // a JS-side mirror of the scene graph. Also refreshes the hit-test cache.
  function refreshInspector() {
    objList = JSON.parse(scene.sceneObjectsJSON());
    objects.replaceChildren();
    for (const o of objList) {
      const row = document.createElement("div");
      row.className = "row";
      const [x, y, z] = o.pos.map((v) => v.toFixed(1));
      const shape = o.kind === "profile" ? `profile · ${o.construction}` : o.kind;
      const planar = o.is3d ? "" : ` <span class="planar" title="a 2D-only body: no third dimension, so it is not traced in full 3D">⚠ 2D-only</span>`;
      row.innerHTML =
        `<span class="oid">#${o.id}</span>` +
        `<span class="okind">${shape}${planar}</span>` +
        `<span class="opos">(${x}, ${y}, ${z})</span>`;
      objects.appendChild(row);
    }
  }

  async function open(file: File) {
    try {
      const report = JSON.parse(scene.loadScene3JSON(await file.text()));
      loaded = true;
      sliceRow.style.display = "";
      refreshBadge();
      refreshInspector();
      refreshGeometry();
      retrace();
      const note = report.migratedFromV1
        ? `migrated from v1 (2D-only bodies are not traced in full 3D)`
        : `v2 · dimension ${report.dimension}`;
      flash(`opened ${file.name} · ${note}`);
      render();
    } catch (err) {
      flash(`could not open ${file.name}: ${err}`, false);
    }
  }

  openBtn.addEventListener("click", () => openFile.click());
  openFile.addEventListener("change", async () => {
    const file = openFile.files?.[0];
    if (file) await open(file);
    openFile.value = "";
  });

  sliceInput.addEventListener("input", () => {
    slice = Number(sliceInput.value);
    sliceLabel.textContent = slice === 0 ? "∞" : slice.toFixed(1);
    // A slice change is a filter, not a retrace: reuse the cached projection.
    render();
  });

  // --- interactive editing: drag a body in the view plane (§10.5) ----------
  //
  // The body's projected centre and a coarse radius decide a grab. The handlers
  // sit on `document` in the capture phase so they run *before* M3's canvas
  // listeners, and only swallow the event when a body is actually grabbed — an
  // empty-space drag still falls through to M3's pan/select, and middle-drag pan
  // is untouched.

  // The body under a client point (its projected disc), or null. In-plane
  // coordinates are the world x,y for the default Z=0 view.
  function pickBody(cx: number, cy: number): Obj | null {
    let best: Obj | null = null;
    let bestD = Infinity;
    for (const o of objList) {
      const c = toScreen({ x: o.pos[0], y: o.pos[1] });
      const rim = toScreen({ x: o.pos[0] + o.radius, y: o.pos[1] });
      const rScreen = Math.hypot(rim.x - c.x, rim.y - c.y);
      const d = Math.hypot(cx - c.x, cy - c.y);
      if (d <= rScreen + 8 && d < bestD) {
        bestD = d;
        best = o;
      }
    }
    return best;
  }

  function onDown(e: PointerEvent) {
    if (!loaded || e.button !== 0 || e.target !== view) return;
    const o = pickBody(e.clientX, e.clientY);
    if (!o) return; // empty space → M3 handles pan/select
    dragBody = o.id;
    const w = toWorld(e.clientX, e.clientY);
    grabU = o.pos[0] - w.x;
    grabV = o.pos[1] - w.y;
    origZ = o.pos[2];
    view.setPointerCapture(e.pointerId);
    view.style.cursor = "grabbing";
    e.stopPropagation();
    e.preventDefault();
  }

  function onMove(e: PointerEvent) {
    if (dragBody === null) return;
    e.stopPropagation();
    const w = toWorld(e.clientX, e.clientY);
    // Translate in the view plane; the out-of-plane coordinate is unchanged.
    scene.setBodyPosition(dragBody, w.x + grabU, w.y + grabV, origZ);
    retrace(true); // f32 interactive picture during the drag
    refreshGeometry();
    refreshInspector();
    render();
  }

  function onUp(e: PointerEvent) {
    if (dragBody === null) return;
    e.stopPropagation();
    dragBody = null;
    view.style.cursor = "";
    retrace(false); // settle on the f64 reference
    render();
  }

  document.addEventListener("pointerdown", onDown, true);
  document.addEventListener("pointermove", onMove, true);
  document.addEventListener("pointerup", onUp, true);
  document.addEventListener("pointercancel", onUp, true);

  // The sub-range [t0, t1] ⊆ [0, 1] of a segment whose signed distance from the
  // plane stays within the slab |dist| <= slice, interpolating linearly between
  // the endpoint distances. `null` when the segment never enters the slab. Slice 0
  // means no filter (the whole segment). This is the §10.4 slice: the *portion*
  // near the plane is drawn, so a path that climbs out of plane is truncated at
  // the slab, not shown whole because one end happens to touch it.
  function slabRange(dA: number, dB: number): [number, number] | null {
    if (slice === 0) return [0, 1];
    const dd = dB - dA;
    if (Math.abs(dd) < 1e-9) return Math.abs(dA) <= slice ? [0, 1] : null;
    const ta = (slice - dA) / dd;
    const tb = (-slice - dA) / dd;
    const lo = Math.max(0, Math.min(ta, tb));
    const hi = Math.min(1, Math.max(ta, tb));
    return lo < hi ? [lo, hi] : null;
  }

  // Stroke (and lightly fill) the element cross-sections. Each kind reads
  // differently: a glass body, a mirror, a plain surface, a detector, an
  // aperture mask.
  const STYLE: Record<string, { stroke: string; width: number; fill?: string }> = {
    body: { stroke: "rgba(150,185,225,0.9)", width: 1.5, fill: "rgba(120,155,205,0.10)" },
    mirror: { stroke: "rgba(190,205,235,0.95)", width: 2.5 },
    surface: { stroke: "rgba(150,160,175,0.7)", width: 1.5 },
    detector: { stroke: "rgba(127,214,255,0.85)", width: 2 },
    aperture: { stroke: "rgba(210,165,120,0.85)", width: 2.5 },
  };

  function drawElements(ctx: CanvasRenderingContext2D, toScreen: (p: V2) => V2) {
    for (const el of xsec) {
      const pts = el.pts;
      if (pts.length < 4) continue;
      const st = STYLE[el.kind] ?? STYLE.surface;
      ctx.beginPath();
      for (let i = 0; i < pts.length; i += 2) {
        const s = toScreen({ x: pts[i], y: pts[i + 1] });
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      }
      if (el.closed) ctx.closePath();
      if (el.closed && st.fill) {
        ctx.fillStyle = st.fill;
        ctx.fill();
      }
      ctx.strokeStyle = st.stroke;
      ctx.lineWidth = st.width;
      ctx.stroke();
    }
  }

  return {
    active: () => loaded,
    overlay(ctx, toScreen) {
      if (!loaded) return;
      // Never paint a projection older than the scene (§3.3): a mutation that
      // bumped the revision without a re-trace is caught here.
      if (scene.sceneRevision() !== tracedRevision) retrace();

      // The optical elements first (a lens as its profile, a mirror/detector as a
      // line), so the rays draw over them.
      drawElements(ctx, toScreen);

      if (segs.length === 0) return;
      ctx.save();
      ctx.lineWidth = 1.2;
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i + STRIDE <= segs.length; i += STRIDE) {
        const au = segs[i], av = segs[i + 1], bu = segs[i + 2], bv = segs[i + 3];
        const flux = segs[i + 4], lambda = segs[i + 5];
        const range = slabRange(segs[i + 7], segs[i + 8]);
        if (!range) continue;
        const [t0, t1] = range;
        const [r, g, b] = wavelengthColor(lambda);
        const a = Math.min(1, 0.25 + flux * 0.75);
        ctx.strokeStyle = `rgba(${r},${g},${b},${a})`;
        const p0 = toScreen({ x: au + (bu - au) * t0, y: av + (bv - av) * t0 });
        const p1 = toScreen({ x: au + (bu - au) * t1, y: av + (bv - av) * t1 });
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
      ctx.restore();
    },
  };
}
