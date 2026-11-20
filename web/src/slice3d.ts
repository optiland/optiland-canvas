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
import { createCardManager, type CardSource } from "./cards";
import type { FastPreview } from "./fastpreview";
import type { Reference, StandIn } from "./reference";
import type { RayLayout } from "./rays";
import { panKeyHeld } from "./pan-key";
import { commitAperture, commitLink, fieldChanged, renderAperture, renderLink } from "./card-extras";

type V2 = { x: number; y: number };

export type Slice3d = {
  /** Is a 3D scene loaded and being previewed? (Suppresses the v1 demo rays.) */
  active: () => boolean;
  /** Draw the element cross-sections and inspector cards for the current view. */
  overlay: (ctx: CanvasRenderingContext2D, toScreen: (p: V2) => V2) => void;
  /** The projected ray paths and how RayLayer reads them in place: each
   *  segment's first nine floats are its `[ax,ay,bx,by,r,g,b,flux,flags]`,
   *  clipped to the slab as it is drawn, so 3D rays render through the same
   *  HDR renderer as v1. */
  rayPicture: () => { segs: Float32Array; layout: RayLayout };
  /** Re-trace (f64) and refresh geometry/inspector after an external scene edit
   *  (a glass change, a density change). Does not repaint — the caller renders. */
  refresh: () => void;
  /** The id of the selected/last-inspected body, or -1. */
  selectedBody: () => number;
  /** The picture's version, after bringing it up to date: it changes
   *  whenever the rays `rayPicture` draws would, so a painter can keep its
   *  raster while it does not. */
  picture: () => number;
  /** Load a scene document (any readable version) into the canonical 3D
   *  scene. False, with the reason shown, when it does not load. */
  openText: (name: string, text: string) => boolean;
  /** Counts the documents loaded: a view whose picture was drawn before the
   *  count moved is showing another design. */
  loads: () => number;
  /** The section's extent in the view plane, `[xmin, ymin, xmax, ymax]` mm,
   *  or null with nothing loaded. */
  extent: () => [number, number, number, number] | null;
  /** Boot the canonical 3D scene into the default authoring scene (a singlet,
   *  a collimated beam, an area detector) so the app opens straight into 3D. */
  bootDefault: () => void;
  /** Stand this view down when another view takes the canvas: close its cards
   *  and cancel any in-progress drag, so nothing lingers behind the 3D model. */
  standDown: () => void;
  /** Adopt a selection from the other view (id, or -1 to clear). */
  setSelected: (id: number) => void;
  /** Select an object and open its inspector card (after adding an element). */
  focusObject: (id: number) => void;
  /** Re-read after an edit made elsewhere while it is still in progress (the
   *  component view's handle drags): the fast picture, no exact trace. The
   *  caller repaints. */
  previewEdit: () => void;
};

// The flat layout of `traceProjected3D`: 11 floats per segment
// (au,av,bu,bv, r,g,b, flux, flags, dist_a, dist_b).
const STRIDE = 11;

export function initSlice3d(opts: {
  scene: Canvas;
  view: HTMLCanvasElement;
  toScreen: (p: V2) => V2;
  toWorld: (px: number, py: number) => V2;
  render: () => void;
  /** True while this view owns the canvas (2D cross-section active). When false,
   *  its pointer/keyboard handlers stand down so the 3D model view has the canvas. */
  owns?: () => boolean;
  /** A body was selected/inspected (for the Abbe picker to target). */
  onSelect?: (id: number) => void;
  /** A file from the open button loaded: frame it, as the gallery does. */
  onFileOpened?: () => void;
  /** The interactive f32 picture, on WebGPU or the CPU mirror. Without it the
   *  view calls the CPU mirror directly. */
  fast?: FastPreview;
  /** The f64 reference, traced off this thread. Without it the view traces
   *  the reference itself, blocking the page while it does. */
  reference?: Reference;
}): Slice3d {
  const { scene, view, toScreen, toWorld, render } = opts;
  const owns = () => opts.owns?.() ?? true;

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
  // Bumped whenever the projected rays or the slab change, so a painter can
  // tell the picture is the one it last rasterized.
  let pictureVersion = 0;

  // The bodies, cached with their world position and a coarse radius, for
  // pointer hit-testing. Refreshed whenever the scene changes.
  type Obj = { id: number; kind: string; construction: string; is3d: boolean; body: boolean; pos: number[]; radius: number; rotZ: number };
  let objList: Obj[] = [];
  // The elements' cross-section with the view plane, as polylines to draw. Kept
  // in step with the geometry (load, drag).
  type XLine = { kind: string; closed: boolean; pts: number[]; tint?: string };
  let xsec: XLine[] = [];
  // The body being dragged, and the grab offset (view-plane u,v) so it does not
  // jump to the cursor. `origZ` keeps the out-of-plane coordinate fixed — an
  // in-plane move (§10.5).
  let dragBody: number | null = null;
  let grabU = 0, grabV = 0, origZ = 0;
  // Dragging the selected body's rotation handle (an in-plane tilt about +Z).
  let rotating = false;
  // The selected / last-inspected body (what the Abbe material picker targets).
  let selected = -1;
  function select(id: number) { selected = id; opts.onSelect?.(id); }
  // Clipboard for copy/paste (a body id), and the diagonal offset counter so
  // repeated pastes stagger, exactly like the v1 canvas.
  let clipboard = -1;
  let pasteN = 0;

  // Grid snapping mirrors the v1 canvas — it reads the same shared grid controls.
  const gridBox = document.getElementById("gridBox") as HTMLInputElement | null;
  const gridSpacing = document.getElementById("gridSpacing") as HTMLInputElement | null;
  const gridOn = () => gridBox?.checked ?? false;
  const gridStep = () => Math.max(0.1, Number(gridSpacing?.value) || 5);
  const snapToGrid = (v: number) => Math.round(v / gridStep()) * gridStep();

  // Draw-mirror tool (§5.3, retargeted to Scene3): arm it, then click-drag in the
  // cross-section to lay a planar fold mirror along the segment.
  let drawMode = false;
  let drawStart: V2 | null = null;
  let drawEnd: V2 | null = null;
  const drawBtn = document.getElementById("drawMirror") as HTMLButtonElement | null;
  function setDrawMode(on: boolean) {
    drawMode = on;
    drawStart = drawEnd = null;
    drawBtn?.classList.toggle("active", on);
    view.style.cursor = on ? "crosshair" : "";
  }
  drawBtn?.addEventListener("click", () => setDrawMode(!drawMode));

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

  // What the f32 preview cannot draw, read once per scene revision: its rays
  // pass straight through those elements, so the badge says so and names them.
  let omittedRev = -1;
  let omitted: { object: number; reason: string }[] = [];
  function refreshBadge() {
    const mode = scene.previewMode();
    const rev = scene.sceneRevision();
    if (rev !== omittedRev) {
      omittedRev = rev;
      omitted = JSON.parse(scene.previewOmittedJSON());
    }
    badge.className = `badge3d ${mode}`;
    badge.textContent = (badgeText[mode] ?? mode) + (omitted.length ? ` · preview omits ${omitted.length}` : "");
    badge.title = omitted.length
      ? `the f32 preview leaves these out (its rays pass through them; the f64 reference traces them):\n${omitted.map((o) => `#${o.object}: ${o.reason}`).join("\n")}`
      : "";
  }

  // `fast` uses the interactive f32 path (during a drag); the default settles on
  // the authoritative f64 reference. A fast frame can arrive after this returns
  // (WebGPU), so it is drawn only when it is newer than the picture on screen,
  // and never over an exact frame of the same scene.
  let shownExact = false;
  // The workers' stand-in on screen, when the picture is one.
  let shownStandIn: StandIn | null = null;
  function retrace(fast = false) {
    if (!fast && opts.reference) {
      // The reference of the scene as it is, when it has arrived; until then
      // the fast picture stands in, and the reference is asked for.
      const b = opts.reference.current();
      if (b) {
        opts.fast?.settled();
        segs = b.projected;
        pictureVersion++;
        tracedRevision = b.key.revision;
        shownExact = true;
        shownStandIn = null;
        return;
      }
      opts.reference.request();
      // What is on screen is not the reference of this scene (it may be of
      // the same revision under other trace settings, as after toggling the
      // ghost layer), so a fast frame may replace it.
      shownExact = false;
      // Where tracing that frame here would stall the page, the workers trace
      // it; the picture on screen holds until one lands. During a drag that
      // may be the frame of an earlier position, newer than the one shown.
      const s = opts.reference.standIn(shownStandIn);
      if (s === "coming") return;
      if (s !== "here") {
        segs = s.segs;
        pictureVersion++;
        tracedRevision = s.revision;
        shownStandIn = s;
        return;
      }
      fast = true;
    }
    if (!fast) {
      opts.fast?.settled();
      segs = scene.traceProjected3D();
      pictureVersion++;
      tracedRevision = scene.sceneRevision();
      shownExact = true;
      shownStandIn = null;
      return;
    }
    if (!opts.fast) {
      segs = scene.traceProjected3DFast();
      pictureVersion++;
      tracedRevision = scene.sceneRevision();
      shownExact = false;
      shownStandIn = null;
      return;
    }
    opts.fast.requestProjected(scene, (f) => {
      if (f.revision < tracedRevision || (f.revision === tracedRevision && shownExact)) return;
      segs = f.segs;
      pictureVersion++;
      tracedRevision = f.revision;
      shownExact = false;
      shownStandIn = null;
      if (f.late) render();
    });
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

  // Load a scene document into the canonical 3D scene: a file the user opens,
  // or a design from the gallery. False, with the reason flashed, when it does
  // not load.
  let loadCount = 0;
  function openText(name: string, text: string): boolean {
    try {
      const report = JSON.parse(scene.loadScene3JSON(text));
      loaded = true;
      // Nothing of the last design stays on screen while this one's first
      // picture is traced.
      loadCount++;
      opts.fast?.forget();
      segs = new Float32Array(0);
      pictureVersion++;
      tracedRevision = -1;
      shownExact = false;
      shownStandIn = null;
      sliceRow.style.display = "";
      refreshBadge();
      refreshInspector();
      refreshGeometry();
      retrace();
      const note = report.migratedFromV1
        ? `migrated from v1 (2D-only bodies are not traced in full 3D)`
        : `v2 · dimension ${report.dimension}`;
      flash(`opened ${name} · ${note}`);
      render();
      return true;
    } catch (err) {
      flash(`could not open ${name}: ${err}`, false);
      return false;
    }
  }

  async function open(file: File) {
    if (openText(file.name, await file.text())) opts.onFileOpened?.();
  }

  // The section's extent in the view plane, `[xmin, ymin, xmax, ymax]` in mm:
  // every element's outline and every source's position, so framing it shows
  // the beams arriving as well as the lens. Null with nothing to frame.
  function extent(): [number, number, number, number] | null {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const take = (x: number, y: number) => {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    };
    for (const l of xsec) for (let i = 0; i + 1 < l.pts.length; i += 2) take(l.pts[i], l.pts[i + 1]);
    for (const s of JSON.parse(scene.sourceObjectsJSON()) as { pos: number[] }[]) take(s.pos[0], s.pos[1]);
    return Number.isFinite(x0) ? [x0, y0, x1, y1] : null;
  }

  openBtn.addEventListener("click", () => openFile.click());
  openFile.addEventListener("change", async () => {
    const file = openFile.files?.[0];
    if (file) await open(file);
    openFile.value = "";
  });

  sliceInput.addEventListener("input", () => {
    slice = Number(sliceInput.value);
    pictureVersion++;
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

  // The selected body's rotation handle: a dot `radius + 6` mm out along the
  // body's current in-plane angle. Grabbing it tilts the body about +Z.
  function rotHandleWorld(o: Obj): V2 {
    const hr = o.radius + 6;
    const a = (o.rotZ * Math.PI) / 180;
    return { x: o.pos[0] + hr * Math.cos(a), y: o.pos[1] + hr * Math.sin(a) };
  }

  function onDown(e: PointerEvent) {
    if (!owns() || !loaded || e.button !== 0 || e.target !== view || panKeyHeld()) return;
    // Draw-mirror tool: a click-drag lays a segment, released into a fold mirror.
    if (drawMode) {
      const w = toWorld(e.clientX, e.clientY);
      const p = gridOn() ? { x: snapToGrid(w.x), y: snapToGrid(w.y) } : { x: w.x, y: w.y };
      drawStart = p; drawEnd = { ...p };
      view.setPointerCapture(e.pointerId);
      e.stopPropagation(); e.preventDefault();
      return;
    }
    // The rotation handle sits outside the object, so grab it before the body.
    const so = objList.find((o) => o.id === selected);
    if (so) {
      const rh = toScreen(rotHandleWorld(so));
      if (Math.hypot(e.clientX - rh.x, e.clientY - rh.y) <= 10) {
        rotating = true;
        view.setPointerCapture(e.pointerId);
        view.style.cursor = "grabbing";
        e.stopPropagation();
        e.preventDefault();
        return;
      }
    }
    const o = pickBody(e.clientX, e.clientY);
    if (!o) return; // empty space → M3 handles pan/select
    dragBody = o.id;
    select(o.id);
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
    if (drawStart) {
      const w = toWorld(e.clientX, e.clientY);
      drawEnd = gridOn() ? { x: snapToGrid(w.x), y: snapToGrid(w.y) } : { x: w.x, y: w.y };
      e.stopPropagation();
      render();
      return;
    }
    if (rotating) {
      e.stopPropagation();
      const so = objList.find((o) => o.id === selected);
      if (so) {
        const w = toWorld(e.clientX, e.clientY);
        const deg = (Math.atan2(w.y - so.pos[1], w.x - so.pos[0]) * 180) / Math.PI;
        // Bodies rotate about +Z; a surface's tilt is its normal angle.
        if (so.body) scene.setObjectRotationZDeg(selected, deg);
        else scene.setSurfaceTiltZDeg(selected, deg);
        retrace(true);
        refreshGeometry();
        refreshInspector();
        cardMgr.renderAll();
        render();
      }
      return;
    }
    if (dragBody === null) return;
    e.stopPropagation();
    const w = toWorld(e.clientX, e.clientY);
    // Translate in the view plane; the out-of-plane coordinate is unchanged.
    // Snap to the grid while it is on, exactly like the v1 canvas.
    let x = w.x + grabU, y = w.y + grabV;
    if (gridOn()) { x = snapToGrid(x); y = snapToGrid(y); }
    scene.setObjectPosition(dragBody, x, y, origZ);
    retrace(true); // f32 interactive picture during the drag
    refreshGeometry();
    refreshInspector();
    cardMgr.renderAll(); // the dragged body's card fields track it
    render();
  }

  function onUp(e: PointerEvent) {
    if (drawStart) {
      e.stopPropagation();
      const a = drawStart, b = drawEnd ?? drawStart;
      drawStart = drawEnd = null;
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      if (len > 1) {
        const id = scene.addPlanarMirror(a.x, a.y, b.x, b.y, 0.95);
        setDrawMode(false); // a placement tool: one mirror per arming
        if (id >= 0) select(id);
        retrace(false); refreshGeometry(); refreshInspector(); render();
      } else {
        view.style.cursor = drawMode ? "crosshair" : "";
        render();
      }
      return;
    }
    if (rotating) {
      e.stopPropagation();
      rotating = false;
      view.style.cursor = "";
      retrace(false); // settle on the f64 reference
      render();
      return;
    }
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
  const STYLE: Record<string, { stroke: string; width: number; fill?: string; dash?: number[] }> = {
    body: { stroke: "rgba(150,185,225,0.9)", width: 1.5, fill: "rgba(120,155,205,0.10)" },
    mirror: { stroke: "rgba(190,205,235,0.95)", width: 2.5 },
    // A splitter half-silvered (dashed); a dichroic in the colour it reflects.
    splitter: { stroke: "rgba(215,225,240,0.95)", width: 2.5, dash: [5, 3] },
    dichroic: { stroke: "rgba(160,200,255,0.95)", width: 3 },
    // A polarizer or a waveplate: hatched, because it does something to the
    // light that a line cannot show and should not look like a plain window.
    polarizing: { stroke: "rgba(186,168,235,0.95)", width: 2.5, dash: [2, 2] },
    surface: { stroke: "rgba(150,160,175,0.7)", width: 1.5 },
    // A DMD or an SLM: a pixelated face, drawn dotted.
    modulator: { stroke: "rgba(175,195,225,0.95)", width: 3, dash: [1.5, 1.5] },
    detector: { stroke: "rgba(127,214,255,0.85)", width: 2 },
    aperture: { stroke: "rgba(210,165,120,0.85)", width: 2.5 },
  };

  // The selected body's centre dot and rotation handle (v1's on-canvas handles).
  function drawSelection(ctx: CanvasRenderingContext2D, toScreen: (p: V2) => V2) {
    if (selected < 0) return;
    const o = objList.find((x) => x.id === selected);
    if (!o) return;
    const c = toScreen({ x: o.pos[0], y: o.pos[1] });
    ctx.fillStyle = "#7fd6ff";
    ctx.beginPath(); ctx.arc(c.x, c.y, 4, 0, Math.PI * 2); ctx.fill();
    const rh = toScreen(rotHandleWorld(o));
    ctx.strokeStyle = "rgba(127,214,255,0.6)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(rh.x, rh.y); ctx.stroke();
    ctx.fillStyle = "#ffd678";
    ctx.beginPath(); ctx.arc(rh.x, rh.y, 4, 0, Math.PI * 2); ctx.fill();
  }

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
      ctx.strokeStyle = el.tint ?? st.stroke;
      ctx.lineWidth = st.width;
      ctx.setLineDash(st.dash ?? []);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // --- inspector card (the same floating-window engine the v1 canvas uses) ---
  //
  // A Scene3 adapter over `./cards`: the card anchors to a body's projected
  // position, shows its position and (for a singlet) its editable geometry and
  // glass, and every field writes straight back to the Rust scene and re-traces.
  const card3dTemplate = document.getElementById("card3dTemplate") as HTMLTemplateElement;

  // Re-trace (f64), refresh the geometry/inspector, and repaint. Called after any
  // scene edit so the picture and numbers settle honestly.
  function afterEdit() {
    retrace(false);
    refreshGeometry();
    refreshInspector();
    cardMgr.renderAll();
    render();
  }

  // Write a card's fields back to the scene. A refused geometry edit leaves the
  // scene unchanged; the following re-render reverts the field to the real value.
  function commitCard(id: number, root: HTMLElement) {
    const num = (n: string) => Number((root.querySelector(`input[data-name="${n}"]`) as HTMLInputElement).value);
    const o = objList.find((o) => o.id === id);
    // Only what was changed: an untouched x, y or tilt written back would
    // flatten a 3D pose, or undo where a link placed the element.
    if (o && (fieldChanged(root, "x", o.pos[0]) || fieldChanged(root, "y", o.pos[1]))) {
      scene.setObjectPosition(id, num("x"), num("y"), o.pos[2]);
    }
    if (o && fieldChanged(root, "rot", o.rotZ)) {
      if (o.body) scene.setObjectRotationZDeg(id, num("rot")); else scene.setSurfaceTiltZDeg(id, num("rot"));
    }
    if (JSON.parse(scene.singletGeomJSON(id))) {
      scene.setSingletGeom(id, num("r1"), num("r2"), num("thickness"), num("semiAperture"), num("conic1"), num("conic2"));
    }
    if (JSON.parse(scene.bodyGlassJSON(id))) {
      scene.setBodyGlassAbbe(id, num("nd"), num("vd"));
    }
    commitAperture(scene, id, root);
    commitLink(scene, id, root);
    const pick = (n: string) => (root.querySelector(`select[data-name="${n}"]`) as HTMLSelectElement)?.value;
    if (JSON.parse(scene.tubeJSON(id))) {
      scene.setTubeDims(id, num("tubeInner"), num("tubeOuter"), num("tubeLength"));
      const bore = pick("tubeBore");
      if (bore && bore !== "custom") scene.setTubeBore(id, bore, num("tubeR"));
    }
    if (heliostatOf(id)) {
      const t = pick("helioTarget");
      if (t) scene.setHeliostatTarget(id, Number(t), 0, 0, 0);
    }
    if (o && !o.body && !heliostatOf(id)) {
      const f = pick("front"), b = pick("back");
      if (f && b) scene.setSurfaceResponse(id, f, b);
    }
    afterEdit();
  }

  // The rig's record of a heliostat, if `id` is one.
  type Helio = { mirror: number; target: { object?: number; point?: number[] } | null; incidence: number; cosine: number; tracking: boolean };
  function heliostatOf(id: number): Helio | null {
    const sun = JSON.parse(scene.solarJSON()) as { heliostats: Helio[] } | null;
    return sun?.heliostats.find((h) => h.mirror === id) ?? null;
  }

  // A tube's dimensions and bore, and a heliostat's target and what it costs.
  function renderSolarCard(id: number, root: HTMLElement) {
    const q = (sel: string) => root.querySelector(sel) as HTMLElement;
    const setV = (name: string, v: number) => {
      const inp = root.querySelector(`input[data-name="${name}"]`) as HTMLInputElement;
      if (inp && document.activeElement !== inp) inp.value = String(Number(v.toFixed(4)));
    };
    const tube = JSON.parse(scene.tubeJSON(id)) as
      | { inner: number; outer: number; length: number; bore: { kind: string; reflectivity?: number } }
      | null;
    q(".c3-tube").style.display = tube ? "block" : "none";
    if (tube) {
      setV("tubeInner", tube.inner); setV("tubeOuter", tube.outer); setV("tubeLength", tube.length);
      const sel = root.querySelector(`select[data-name="tubeBore"]`) as HTMLSelectElement;
      if (document.activeElement !== sel) sel.value = tube.bore.kind;
      q(".c3-tubeR").style.display = tube.bore.kind === "mirror" ? "" : "none";
      setV("tubeR", tube.bore.reflectivity ?? 0.95);
    }
    const h = heliostatOf(id);
    q(".c3-helio").style.display = h ? "block" : "none";
    if (h) {
      const sel = root.querySelector(`select[data-name="helioTarget"]`) as HTMLSelectElement;
      if (document.activeElement !== sel) {
        const want = String(h.target?.object ?? "");
        sel.replaceChildren(
          ...objList
            .filter((x) => x.id !== id)
            .map((x) => new Option(`#${x.id} · ${x.kind === "profile" ? x.construction : x.kind}`, String(x.id))),
        );
        if (!h.target?.object && h.target?.point) {
          sel.prepend(new Option(`point (${h.target.point.map((v) => v.toFixed(0)).join(", ")})`, ""));
        }
        sel.value = want;
      }
      q(".c3-helioNote").textContent = h.tracking
        ? `follows the sun · incidence ${h.incidence.toFixed(1)}° · cosine ${h.cosine.toFixed(3)} · its tilt is set by the sun`
        : "the sun and the target are on opposite sides: it cannot reach it";
    }
    // A heliostat's faces are the rig's, not a free choice.
    const surf = root.querySelector(".c3-surface") as HTMLElement | null;
    if (surf && h) surf.style.display = "none";
  }

  // Show/populate the per-face response block for a surface object (hidden for
  // bodies), reading current responses from Rust.
  function renderSurfaceCard(id: number, root: HTMLElement, o: Obj) {
    const block = root.querySelector(".c3-surface") as HTMLElement | null;
    if (!block) return;
    const r = o.body ? null : JSON.parse(scene.surfaceResponseJSON(id));
    if (!r) { block.style.display = "none"; return; } // bodies + apertures have no per-face response
    block.style.display = "block";
    const set = (n: string, v: string) => {
      const s = root.querySelector(`select[data-name="${n}"]`) as HTMLSelectElement;
      if (s && document.activeElement !== s) s.value = v;
    };
    set("front", r.front); set("back", r.back);
  }

  const cardSource: CardSource = {
    template: card3dTemplate,
    toScreen,
    render2d: render,
    anchorWorld: (id) => {
      const o = objList.find((o) => o.id === id);
      return o ? { x: o.pos[0], y: o.pos[1] } : null;
    },
    render(id, root) {
      const o = objList.find((o) => o.id === id);
      if (!o) return false; // body deleted → close the orphan card
      const q = (sel: string) => root.querySelector(sel) as HTMLElement;
      q(".c-title").textContent = `#${id} · ${o.kind === "profile" ? o.construction : o.kind}`;
      const setV = (name: string, v: number | null) => {
        const inp = root.querySelector(`input[data-name="${name}"]`) as HTMLInputElement;
        if (!inp || document.activeElement === inp) return;
        inp.value = v == null ? "" : String(Number(v.toFixed(4)));
      };
      setV("x", o.pos[0]);
      setV("y", o.pos[1]);
      setV("rot", o.rotZ); // tilt: body Z-rotation or surface normal angle
      const g = JSON.parse(scene.singletGeomJSON(id));
      const glass = JSON.parse(scene.bodyGlassJSON(id));
      q(".c3-singlet").style.display = g ? "block" : "none";
      if (g) {
        setV("r1", g.r1); setV("r2", g.r2); setV("thickness", g.thickness);
        setV("semiAperture", g.semiAperture); setV("conic1", g.conic1); setV("conic2", g.conic2);
      }
      q(".c3-glass").style.display = glass ? "block" : "none"; // any glass body (singlet, prism)
      if (glass) { setV("nd", glass.nd); setV("vd", glass.vd); }
      renderAperture(scene, id, root);
      renderLink(scene, id, root, objList);
      renderSurfaceCard(id, root, o);
      renderSolarCard(id, root);
      return true;
    },
    wire(id, root) {
      root.querySelectorAll<HTMLInputElement>("input[data-name]").forEach((input) => {
        let before = "";
        input.addEventListener("focus", () => { before = input.value; });
        input.addEventListener("keydown", (e) => {
          if (e.key === "Escape") { input.value = before; input.blur(); commitCard(id, root); }
          e.stopPropagation(); // keep field keys out of the global shortcuts
        });
        input.addEventListener("input", () => commitCard(id, root));
        input.addEventListener("wheel", (e) => { // hover-scroll scrubs
          e.preventDefault();
          const step = Number(input.step) || 0.5;
          const dir = e.deltaY < 0 ? 1 : -1;
          input.value = String(Number((Number(input.value) + dir * step * (e.shiftKey ? 10 : 1)).toFixed(6)));
          commitCard(id, root);
        }, { passive: false });
      });
      root.querySelectorAll<HTMLSelectElement>("select[data-name]").forEach((sel) => {
        sel.addEventListener("keydown", (e) => e.stopPropagation());
        sel.addEventListener("change", () => commitCard(id, root));
      });
    },
  };

  const cardMgr = createCardManager(cardSource);

  // Double-click a body to open its inspector card. Runs in the capture phase so
  // it beats M3's canvas dblclick (which would otherwise open a v1 card on the
  // hidden demo). Only acts when a body is actually under the cursor.
  document.addEventListener("dblclick", (e) => {
    if (!owns() || !loaded || e.target !== view) return;
    const o = pickBody(e.clientX, e.clientY);
    if (o) { select(o.id); cardMgr.open(o.id); e.stopPropagation(); }
  }, true);

  // The projected ray paths in RayLayer's `[ax,ay,bx,by,r,g,b,flux,flags]`
  // layout, each segment clipped to the active slab. The main renderer feeds this
  // to the same HDR RayLayer as v1, so 3D rays get the identical spectral colour,
  // tonemap, ghost layer, and TIR handling.
  // A stale picture is retraced exactly, unless a GPU frame for a newer
  // revision is already on its way (then the old picture holds until it lands).
  function freshen() {
    if (scene.sceneRevision() !== tracedRevision && !opts.fast?.pending()) retrace();
  }

  // The picture is read in place: its first nine floats are RayLayer's, and
  // the last two, the ends' distances from the plane, clip it to the slab. A
  // ghost-lit lens has millions of segments, too many to copy per picture.
  function rayPicture(): { segs: Float32Array; layout: RayLayout } {
    freshen();
    return { segs: segs as Float32Array, layout: { stride: STRIDE, clip: slice === 0 ? null : slabRange } };
  }

  // Re-trace (f64) and refresh derived state after an external edit (a glass or
  // density change made outside the drag/card paths). The caller repaints.
  function refresh() {
    retrace(false);
    refreshGeometry();
    refreshInspector();
    cardMgr.renderAll();
  }
  const afterEdit3d = () => { refresh(); render(); };

  // --- copy / paste / delete / nudge on the selected body (§5.1) -----------
  // The v1 canvas' keyboard verbs, retargeted to the Scene3 selection. M3's own
  // keydown defers these to us while a 3D scene is active (`threeDActive`).
  function pasteBody() {
    if (clipboard < 0) return;
    pasteN++;
    const d = gridOn() ? gridStep() : 5;
    const id = scene.cloneBody3D(clipboard, d * pasteN, -d * pasteN, 0);
    if (id >= 0) { select(id); afterEdit3d(); cardMgr.open(id); }
  }
  function deleteSelected() {
    if (selected < 0) return;
    if (scene.deleteObject3D(selected)) {
      cardMgr.close(selected);
      selected = -1;
      afterEdit3d();
    }
  }
  function nudge(key: string) {
    const o = objList.find((x) => x.id === selected);
    if (!o) return;
    const s = gridOn() ? gridStep() : 1;
    const dx = key === "ArrowLeft" ? -s : key === "ArrowRight" ? s : 0;
    const dy = key === "ArrowUp" ? s : key === "ArrowDown" ? -s : 0;
    scene.setObjectPosition(selected, o.pos[0] + dx, o.pos[1] + dy, o.pos[2]);
    afterEdit3d();
  }

  window.addEventListener("keydown", (e) => {
    if (!owns() || !loaded || e.target instanceof HTMLInputElement) return;
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Escape") { cardMgr.closeAll(); selected = -1; render(); return; }
    if (mod && (e.key === "c" || e.key === "C")) { if (selected >= 0) clipboard = selected; e.preventDefault(); return; }
    if (mod && (e.key === "v" || e.key === "V")) { pasteBody(); e.preventDefault(); return; }
    if (mod && (e.key === "x" || e.key === "X")) { if (selected >= 0) { clipboard = selected; deleteSelected(); } e.preventDefault(); return; }
    if (mod && (e.key === "d" || e.key === "D")) { if (selected >= 0) { clipboard = selected; pasteBody(); } e.preventDefault(); return; }
    if (e.key === "Delete" || e.key === "Backspace") { deleteSelected(); e.preventDefault(); return; }
    if (e.key.startsWith("Arrow")) { nudge(e.key); e.preventDefault(); return; }
  });

  function bootDefault() {
    const singletId = scene.newDefaultScene3();
    loaded = true;
    sliceRow.style.display = "";
    refreshBadge();
    refreshInspector();
    refreshGeometry();
    retrace();
    select(singletId); // the Abbe picker targets the lens by default
    flash("default 3D scene · a singlet traced in the Y–Z plane");
    render();
  }

  // Stand down when the 3D model view takes over: drop any in-progress drag and
  // close the floating cards so they don't hover over the other view.
  function standDown() {
    dragBody = null;
    rotating = false;
    setDrawMode(false);
    view.style.cursor = "";
    cardMgr.closeAll();
  }
  // Adopt a selection handed over from the other view (keeps the material picker
  // and the highlighted body consistent across a toggle).
  function setSelected(id: number) {
    select(id);
  }
  // Re-read, select an object, and open its card (used after adding an element).
  function focusObject(id: number) {
    retrace(false); refreshGeometry(); refreshInspector();
    select(id); cardMgr.open(id); render();
  }

  return {
    active: () => loaded,
    picture: () => {
      freshen();
      return pictureVersion;
    },
    openText,
    loads: () => loadCount,
    extent,
    bootDefault,
    standDown,
    setSelected,
    focusObject,
    previewEdit: () => {
      retrace(true);
      refreshGeometry();
      refreshInspector();
      cardMgr.renderAll();
    },
    rayPicture,
    refresh,
    selectedBody: () => selected,
    // The element cross-sections, the selection/rotation handle, and the
    // inspector cards; the rays themselves go through the shared RayLayer (see
    // `rayPicture`), drawn by the main renderer.
    overlay(ctx, toScreen) {
      if (!loaded) return;
      // Cards follow their body under any pan/zoom/drag: reposition every frame.
      cardMgr.positionAll();
      drawElements(ctx, toScreen);
      drawSelection(ctx, toScreen);
      // The in-progress draw-mirror segment.
      if (drawStart && drawEnd) {
        const a = toScreen(drawStart), b = toScreen(drawEnd);
        ctx.strokeStyle = "rgba(190,205,235,0.9)";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.setLineDash([]);
      }
      cardMgr.drawLeaders(ctx, selected);
    },
  };
}
