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
  const promoteRow = document.getElementById("promote3dRow")!;
  const promoteRevolve = document.getElementById("promoteRevolve") as HTMLButtonElement;
  const promoteExtrude = document.getElementById("promoteExtrude") as HTMLButtonElement;
  const promoteDepth = document.getElementById("promoteDepth") as HTMLInputElement;
  const objects = document.getElementById("objects3d")!;

  let loaded = false;
  // Ids of bodies whose third dimension is still unresolved (PlanarOnly), from
  // the migration report — the ones the promotion controls act on.
  let planarBodies: number[] = [];
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

  // Promotion is offered only while unresolved bodies remain.
  function updatePromoteRow() {
    promoteRow.style.display = planarBodies.length > 0 ? "" : "none";
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
      const planar = o.is3d ? "" : ` <span class="planar" title="no third dimension yet — promote to trace in full 3D">⚠ planar-only</span>`;
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
      planarBodies = report.planarOnlyBodies ?? [];
      sliceRow.style.display = "";
      updatePromoteRow();
      refreshBadge();
      refreshInspector();
      retrace();
      const note = report.migratedFromV1
        ? `migrated from v1 · ${planarBodies.length} body(ies) still planar-only — promote to trace in full 3D`
        : `v2 · dimension ${report.dimension}`;
      flash(`opened ${file.name} · ${note}`);
      render();
    } catch (err) {
      flash(`could not open ${file.name}: ${err}`, false);
    }
  }

  // Promote every unresolved body to a 3D construction (§11.3). A body the
  // construction refuses (e.g. an asymmetric profile cannot revolve) stays
  // planar-only with its reason surfaced, never silently changed.
  function promoteAll(revolve: boolean) {
    const depth = Number(promoteDepth.value) || 5;
    const remaining: number[] = [];
    let ok = 0;
    let reason = "";
    for (const id of planarBodies) {
      try {
        if (revolve) scene.promoteBodyToRevolved(id);
        else scene.promoteBodyToExtruded(id, -depth, depth);
        ok++;
      } catch (err) {
        remaining.push(id);
        reason = String(err);
      }
    }
    planarBodies = remaining;
    updatePromoteRow();
    refreshBadge();
    refreshInspector();
    retrace();
    if (ok > 0) {
      const refused = remaining.length ? ` · ${remaining.length} refused` : "";
      flash(`promoted ${ok} body(ies) → ${revolve ? "revolved" : "extruded"}${refused}`);
    } else {
      flash(`promotion refused: ${reason}`, false);
    }
    render();
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

  promoteRevolve.addEventListener("click", () => promoteAll(true));
  promoteExtrude.addEventListener("click", () => promoteAll(false));

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

  return {
    active: () => loaded,
    overlay(ctx, toScreen) {
      if (!loaded) return;
      // Never paint a projection older than the scene (§3.3): a mutation that
      // bumped the revision without a re-trace is caught here.
      if (scene.sceneRevision() !== tracedRevision) retrace();
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
