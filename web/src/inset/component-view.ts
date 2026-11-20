// The component view: the selected element, drawn on its own, with handles.
//
// It follows whichever view owns the canvas (2D cross-section or 3D model) and
// its selection, and re-reads the element whenever the scene revision moves,
// so an edit made anywhere shows here and an edit made here shows everywhere.
// Each kind of element has its own editor:
//   a singlet: its profile (lens-editor.ts);
//   a Fresnel lens: its recipe (fresnel-editor.ts);
//   a flat surface: its coating (coating-editor.ts);
//   an aperture: its opening, face on;
//   a detector: its face to scale, its pixels, and what it measured;
//   a source: its spectral lines;
//   anything else: a summary.
// Edits call `edited(false)` while a handle is held, so the main view draws the
// fast picture, and `edited(true)` when they end, for the exact trace.

import type { Canvas } from "../../pkg/oc_wasm.js";
import type { SceneObject } from "../render3d/cards3d";
import { bringToFront } from "../cards";
import { halfExtent, readAperture, reshape, roomForOpening, type ApertureShape, type Outline } from "../card-extras";
import { createCoatingEditor } from "./coating-editor";
import { createDetectorView } from "./detector-view";
import { createFresnelEditor } from "./fresnel-editor";
import type { InsetContent } from "./inset";
import { createLensEditor } from "./lens-editor";
import { createPolarizationEditor } from "./polarization-editor";
import { createGratingEditor } from "./grating-editor";
import { createModulatorEditor } from "./modulator-editor";
import {
  drawHandle,
  fmt,
  hitHandle,
  label,
  numberField,
  segmented,
  type DragPhase,
  type Editor,
  type EditorContext,
  type Handle,
  type NumberField,
} from "./ui";

export type ComponentView = InsetContent & {
  /** Re-read the selection and the scene if either changed (cheap otherwise). */
  sync: (force?: boolean) => void;
  /** The exact trace was re-read (the dock just did): re-read what is shown,
   *  even if the scene revision did not move, as it does not for a trace
   *  setting such as the ghost layer. A new selection is left to `sync`. */
  retraced: () => void;
  /** What is shown, and the handles in page coordinates (tests). */
  debug: () => { id: number; kind: string; note: string; handles: { key: string; x: number; y: number }[] };
};

type EditorKind = "lens" | "fresnel" | "coating" | "polarizing" | "grating" | "modulator" | "aperture" | "detector" | "source" | "summary" | "none";

export function createComponentView(opts: {
  scene: Canvas;
  body: HTMLElement;
  setSubtitle: (text: string) => void;
  /** The selected object id in the active view, or -1. */
  selected: () => number;
  edited: (settled: boolean) => void;
  /** A detector's f64 readout (`detectorResultJSON`) from the reference the
   *  dock shows, or null while none is traced. Without it the view traces. */
  readout?: (id: number) => string | null;
  /** The window this view fills, so a panel beside the drawing can widen it. */
  frame?: { width: () => number; setWidth: (w: number) => void };
  /** A trace setting changed; trace the reference again. */
  retrace?: () => void;
}): ComponentView {
  const { scene, body } = opts;
  const canvas = document.createElement("canvas");
  canvas.className = "cv-canvas";
  const bar = document.createElement("div");
  bar.className = "cv-bar";
  const fields = document.createElement("div");
  fields.className = "cv-fields";
  const read = document.createElement("div");
  read.className = "cv-read";
  const note = document.createElement("div");
  note.className = "cv-note";
  body.append(canvas, bar, fields, read, note);
  const c2 = canvas.getContext("2d")!;

  let open = false;
  let shownId = -2;
  let shownKind: EditorKind = "none";
  let shownRev = -1;
  let objects: SceneObject[] = [];
  let objectsRev = -1;
  let editor: Editor | null = null;
  let held: Handle | null = null;
  let hot: Handle | null = null;
  // The handles as last drawn: the pointer is tested against what is on
  // screen, not against a layout that may have moved since.
  let drawn: Handle[] = [];
  let message = "";
  let warn = false;
  // The width the window had before a panel beside the drawing widened it,
  // given back when the panel closes; null while nothing is borrowed.
  let widenedFrom: number | null = null;
  const WIDEN_BY = 340;
  // While an editor is being built, what it asks for, applied once it is.
  let building = false;
  let wantWide = false;
  function widen(on: boolean) {
    const f = opts.frame;
    if (!f) return;
    if (on && widenedFrom == null) {
      widenedFrom = f.width();
      f.setWidth(widenedFrom + WIDEN_BY);
    } else if (!on && widenedFrom != null) {
      // Only if the user has not resized it since: their width wins.
      if (Math.abs(f.width() - (widenedFrom + WIDEN_BY)) < 2) f.setWidth(widenedFrom);
      widenedFrom = null;
    }
  }

  function readObjects() {
    const rev = scene.sceneRevision();
    if (rev === objectsRev) return;
    objectsRev = rev;
    objects = [...JSON.parse(scene.sceneObjectsJSON()), ...JSON.parse(scene.sourceObjectsJSON())];
  }
  const objectOf = (id: number) => objects.find((o) => o.id === id);

  function kindFor(id: number): EditorKind {
    const o = objectOf(id);
    if (!o) return "none";
    // A polarizing element answers `surfaceCoatingJSON` too, with nulls where
    // its R and T would be, so it has to be asked about first or it would open
    // the coating editor and show two empty curves.
    if (JSON.parse(scene.surfacePolarizationJSON(id))) return "polarizing";
    // A DMD or SLM answers `surfaceCoatingJSON` with nulls as well; its
    // pattern is what you came to edit.
    if (JSON.parse(scene.modulatorJSON(id))) return "modulator";
    // A grating answers `surfaceCoatingJSON` with nulls too: it sends light
    // new ways rather than reflecting and transmitting it.
    if (JSON.parse(scene.gratingJSON(id))) return "grating";
    // A Fresnel lens is a revolved glass body too, but its facets follow from
    // a recipe, so it has an editor of its own.
    if (JSON.parse(scene.fresnelLensJSON(id))) return "fresnel";
    // A coated plate is a glass body, so it would read as a lens by its shape,
    // and its film is the thing you came to edit. But a coated *lens* is still
    // a lens: what tells them apart is curvature, since a plate is flat on both
    // faces. Asking by shape first is what lets a lens keep its own editor once
    // somebody puts an AR coating on it.
    const lens = JSON.parse(scene.singletShapeJSON(id, 2));
    if (lens && (lens.r1 != null || lens.r2 != null)) return "lens";
    if (JSON.parse(scene.surfaceFilmJSON(id))) return "coating";
    if (lens) return "lens";
    if (o.kind === "aperture") return "aperture";
    if (o.kind === "detector") return "detector";
    if (o.kind === "source") return "source";
    if (JSON.parse(scene.surfaceCoatingJSON(id))) return "coating";
    return "summary";
  }

  function showNote() {
    note.textContent = message || editor?.hint || "";
    note.classList.toggle("warn", warn && !!message);
  }

  function context(id: number): EditorContext {
    return {
      scene,
      id,
      edited: opts.edited,
      note: (text, w = false) => {
        message = text;
        warn = w;
        showNote();
      },
      redraw: draw,
      bar,
      fields,
      read,
      readout: opts.readout ?? ((d) => scene.detectorResultJSON(d)),
      widen: (on) => {
        wantWide = on;
        if (!building) widen(on);
      },
      retrace: () => opts.retrace?.(),
    };
  }

  function build(id: number, kind: EditorKind) {
    bar.replaceChildren();
    fields.replaceChildren();
    read.replaceChildren();
    read.classList.remove("tall");
    message = "";
    building = true;
    wantWide = false;
    held = hot = null;
    const ctx = context(id);
    switch (kind) {
      case "lens": editor = createLensEditor(ctx, () => { readObjects(); return objectOf(id); }); break;
      case "fresnel": editor = createFresnelEditor(ctx); break;
      case "coating": editor = createCoatingEditor(ctx); break;
      case "polarizing": editor = createPolarizationEditor(ctx); break;
      case "grating": editor = createGratingEditor(ctx); break;
      case "modulator": editor = createModulatorEditor(ctx); break;
      case "aperture": editor = apertureView(ctx); break;
      case "detector": editor = createDetectorView(ctx); break;
      case "source": editor = sourceView(ctx); break;
      case "summary": editor = summaryView(ctx, objectOf(id)!); break;
      default: editor = null;
    }
    // A panel the last editor opened beside its drawing goes with it, unless
    // this one opens one too.
    building = false;
    widen(wantWide);
    shownId = id;
    shownKind = kind;
  }

  function subtitle(id: number, kind: EditorKind) {
    const o = objectOf(id);
    if (!o || kind === "none") return "nothing selected";
    const what = kind === "lens"
      ? "singlet"
      : kind === "fresnel"
      ? "Fresnel lens"
      // "surface" says nothing a user wants; name the part.
      : kind === "polarizing"
      ? (JSON.parse(scene.surfacePolarizationJSON(id))?.kind ?? "surface")
      : kind === "grating"
      ? `${JSON.parse(scene.gratingJSON(id))?.transmissive ? "transmission" : "reflection"} grating`
      : kind === "modulator"
      ? (JSON.parse(scene.modulatorJSON(id))?.kind === "dmd" ? "DMD" : "LCoS SLM")
      : o.kind === "profile"
      ? o.construction
      : o.kind;
    return `#${id} · ${what}`;
  }

  function sync(force = false) {
    if (!open) return;
    const id = opts.selected();
    const rev = scene.sceneRevision();
    if (!force && id === shownId && rev === shownRev) return;
    // A held handle's editor keeps itself current.
    if (held && id === shownId) {
      shownRev = rev;
      return;
    }
    readObjects();
    const kind = kindFor(id);
    if (id !== shownId || kind !== shownKind) {
      build(id, kind);
      // Selecting an object opens its inspector card, which joins the same
      // floating stack and would land on top of this window. The window the
      // user is about to edit in comes back to the front; clicking a card
      // still puts that card above it.
      const frame = opts.body.closest(".inset");
      if (frame) bringToFront(frame as HTMLElement);
    }
    editor?.refresh();
    shownRev = rev;
    opts.setSubtitle(subtitle(id, kind));
    if (!editor) {
      read.textContent = "";
      message = "select an element in the canvas (click it) to see it here";
    }
    showNote();
    draw();
  }

  function fitCanvas(): { w: number; h: number } {
    const r = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    c2.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  function draw() {
    if (!open) return;
    const { w, h } = fitCanvas();
    c2.clearRect(0, 0, w, h);
    drawn = [];
    if (!editor) {
      c2.fillStyle = "rgba(205,214,228,0.45)";
      c2.font = "11px ui-monospace, monospace";
      c2.fillText("no element selected", 12, 20);
      return;
    }
    editor.draw(c2, w, h);
    drawn = editor.handles();
    for (const hd of drawn) drawHandle(c2, hd, hot?.key === hd.key || held?.key === hd.key);
  }

  // --- pointer: handles -----------------------------------------------------
  const local = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !editor) return;
    const p = local(e);
    const h = hitHandle(drawn, p.x, p.y) ?? editor.pick?.(p.x, p.y) ?? null;
    if (!h) return;
    held = h;
    canvas.setPointerCapture(e.pointerId);
    editor.drag(h, p.x, p.y, "start");
    e.preventDefault();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!editor) return;
    const p = local(e);
    if (held) {
      editor.drag(held, p.x, p.y, "move");
      return;
    }
    const h = hitHandle(drawn, p.x, p.y) ?? editor.pick?.(p.x, p.y) ?? null;
    if (h?.key !== hot?.key) {
      hot = h;
      canvas.style.cursor = h ? h.cursor : "";
      if (!warn) note.textContent = h ? h.title : message || editor.hint;
      draw();
    }
  });
  const release = (e: PointerEvent, phase: DragPhase) => {
    if (!held || !editor) return;
    const p = local(e);
    const h = held;
    held = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    editor.drag(h, p.x, p.y, phase);
    draw();
  };
  canvas.addEventListener("pointerup", (e) => release(e, "end"));
  canvas.addEventListener("pointercancel", (e) => release(e, "end"));
  canvas.addEventListener("pointerleave", () => {
    if (held || !hot) return;
    hot = null;
    canvas.style.cursor = "";
    showNote();
    draw();
  });

  return {
    setOpen: (o) => {
      open = o;
      if (o) sync(true);
    },
    resized: () => draw(),
    sync,
    retraced: () => {
      // A selection change is the frame's `sync` to make, after the object's
      // inspector card has opened, so the window can come back above it.
      if (!open || held || !editor || opts.selected() !== shownId) return;
      editor.refresh();
      shownRev = scene.sceneRevision();
      showNote();
      draw();
    },
    debug: () => {
      const r = canvas.getBoundingClientRect();
      return {
        id: shownId,
        kind: shownKind,
        note: note.textContent ?? "",
        handles: (open ? drawn : []).map((h) => ({ key: h.key, x: r.left + h.x, y: r.top + h.y })),
      };
    },
  };
}

// ---------------------------------------------------------------------------
// The smaller views
// ---------------------------------------------------------------------------

/** An aperture, face on: the mask and the opening, each a circle or a
 *  rectangle (the opening may also be a ring, a circle with a central stop),
 *  sized on its own. Handles on the drawing size the opening and the mask;
 *  the fields under it take exact sizes. */
function apertureView(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  let shape: ApertureShape = { opening: { kind: "circle", d: 10 }, mask: { kind: "circle", d: 25 } };
  // The scale is held while a handle is dragged, so the drawing does not
  // rescale under the pointer.
  const view = { cx: 0, cy: 0, scale: 1, held: false };

  const openPick = segmented<"circle" | "ring" | "rect">(
    ctx.bar,
    [
      { value: "circle", text: "circle", title: "a round hole" },
      { value: "ring", text: "ring", title: "a round hole with a central stop" },
      { value: "rect", text: "rect", title: "a rectangular hole" },
    ],
    (k) => write(roomForOpening({ opening: reshape(shape.opening, k), mask: shape.mask }), true),
  );
  label(ctx.bar, "mask");
  const maskPick = segmented<"circle" | "rect" | "none">(
    ctx.bar,
    [
      { value: "circle", text: "circle", title: "a round plate" },
      { value: "rect", text: "rect", title: "a rectangular plate" },
      { value: "none", text: "∞", title: "a plate with no edge: blocks everything outside the hole" },
    ],
    (k) => write({ opening: shape.opening, mask: reshape(shape.mask, k, shape.opening) }, true),
  );
  const field = (text: string, name: string, step: number, set: (v: number) => ApertureShape | null) =>
    numberField(ctx.fields, text, "mm", {
      name,
      step,
      commit: (v) => {
        const next = set(v);
        if (next) write(next, true);
      },
    });
  const o = () => shape.opening;
  const m = () => shape.mask;
  const hole = (next: Outline) => roomForOpening({ ...shape, opening: next });
  const fD = field("hole ⌀", "apOpenD", 1, (v) => { const x = o(); return x.kind === "circle" || x.kind === "ring" ? hole({ ...x, d: v }) : null; });
  const fDin = field("stop ⌀", "apOpenDIn", 1, (v) => { const x = o(); return x.kind === "ring" ? { ...shape, opening: { ...x, dIn: v } } : null; });
  const fW = field("hole w", "apOpenW", 1, (v) => { const x = o(); return x.kind === "rect" ? hole({ ...x, w: v }) : null; });
  const fH = field("hole h", "apOpenH", 1, (v) => { const x = o(); return x.kind === "rect" ? hole({ ...x, h: v }) : null; });
  const fMD = field("mask ⌀", "apMaskD", 5, (v) => { const x = m(); return x.kind === "circle" ? { ...shape, mask: { ...x, d: v } } : null; });
  const fMW = field("mask w", "apMaskW", 5, (v) => { const x = m(); return x.kind === "rect" ? { ...shape, mask: { ...x, w: v } } : null; });
  const fMH = field("mask h", "apMaskH", 5, (v) => { const x = m(); return x.kind === "rect" ? { ...shape, mask: { ...x, h: v } } : null; });
  const showRow = (f: NumberField, on: boolean) => {
    (f.input.parentElement as HTMLElement).style.display = on ? "" : "none";
  };

  function reread() {
    const s = readAperture(scene, id);
    if (!s) return;
    shape = s;
    const op = s.opening;
    const mk = s.mask;
    openPick.set(op.kind === "circle" || op.kind === "ring" || op.kind === "rect" ? op.kind : null);
    maskPick.set(mk.kind === "circle" || mk.kind === "rect" || mk.kind === "none" ? mk.kind : null);
    showRow(fD, op.kind === "circle" || op.kind === "ring");
    showRow(fDin, op.kind === "ring");
    showRow(fW, op.kind === "rect");
    showRow(fH, op.kind === "rect");
    showRow(fMD, mk.kind === "circle");
    showRow(fMW, mk.kind === "rect");
    showRow(fMH, mk.kind === "rect");
    if (op.kind === "circle" || op.kind === "ring") fD.set(op.d);
    if (op.kind === "ring") fDin.set(op.dIn);
    if (op.kind === "rect") { fW.set(op.w); fH.set(op.h); }
    if (mk.kind === "circle") fMD.set(mk.d);
    if (mk.kind === "rect") { fMW.set(mk.w); fMH.set(mk.h); }
    const size = (x: Outline) =>
      x.kind === "circle" ? `⌀ <b>${fmt(x.d, 1)}</b> mm`
        : x.kind === "ring" ? `⌀ <b>${fmt(x.d, 1)}</b> mm, stop ⌀ <b>${fmt(x.dIn, 1)}</b> mm`
          : x.kind === "rect" ? `<b>${fmt(x.w, 1)}</b> × <b>${fmt(x.h, 1)}</b> mm`
            : "unbounded";
    ctx.read.innerHTML = `hole ${size(op)} · mask ${size(mk)}`;
  }

  function write(next: ApertureShape, settled: boolean) {
    const why = scene.setApertureShape(id, JSON.stringify(next));
    if (why) {
      ctx.note(`refused: ${why}`, true);
      return;
    }
    ctx.note("");
    ctx.edited(settled);
    reread();
    ctx.redraw();
  }

  /** Trace an outline as a path centred on the drawing. */
  function outline(c: CanvasRenderingContext2D, x: Outline, reverse: boolean) {
    const s = view.scale;
    if (x.kind === "rect") {
      const hw = (x.w / 2) * s;
      const hh = (x.h / 2) * s;
      if (reverse) c.rect(view.cx + hw, view.cy - hh, -2 * hw, 2 * hh);
      else c.rect(view.cx - hw, view.cy - hh, 2 * hw, 2 * hh);
    } else if (x.kind === "circle" || x.kind === "ring") {
      c.moveTo(view.cx + (x.d / 2) * s, view.cy);
      c.arc(view.cx, view.cy, (x.d / 2) * s, 0, Math.PI * 2, reverse);
    }
  }

  /** Half extents of what is drawn: the mask, or around the hole when the
   *  mask has no edge. */
  const drawnExtent = (): [number, number] => {
    const mask = halfExtent(shape.mask);
    if (mask) return mask;
    const hole = halfExtent(shape.opening) ?? [5, 5];
    return [hole[0] * 1.8, hole[1] * 1.8];
  };

  reread();
  return {
    hint: "face on · drag the handles to size the hole and the mask",
    refresh: reread,
    draw(c, w, h) {
      view.cx = w / 2;
      view.cy = h / 2;
      const [ex, ey] = drawnExtent();
      if (!view.held) view.scale = Math.min((0.44 * w) / ex, (0.44 * h) / ey);
      c.fillStyle = "rgba(40,44,52,0.95)";
      c.beginPath();
      if (shape.mask.kind === "none") c.rect(0, 0, w, h);
      else outline(c, shape.mask, false);
      outline(c, shape.opening, true);
      c.fill("evenodd");
      c.strokeStyle = "rgba(210,165,120,0.9)";
      c.lineWidth = 1.5;
      c.beginPath();
      outline(c, shape.opening, false);
      c.stroke();
      if (shape.opening.kind === "ring") {
        c.fillStyle = "rgba(40,44,52,0.95)";
        c.beginPath();
        c.arc(view.cx, view.cy, (shape.opening.dIn / 2) * view.scale, 0, Math.PI * 2);
        c.fill();
        c.stroke();
      }
      if (shape.mask.kind !== "none") {
        c.strokeStyle = "rgba(150,175,210,0.6)";
        c.beginPath();
        outline(c, shape.mask, false);
        c.stroke();
      }
      c.strokeStyle = "rgba(150,175,210,0.35)";
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(view.cx - ex * view.scale, view.cy);
      c.lineTo(view.cx + ex * view.scale, view.cy);
      c.stroke();
      c.setLineDash([]);
    },
    handles() {
      const s = view.scale;
      const hs: Handle[] = [];
      const op = shape.opening;
      const mk = shape.mask;
      const hole = "#d2a578";
      const plate = "#96afd2";
      if (op.kind === "circle" || op.kind === "ring") {
        hs.push({ key: "hole-d", x: view.cx + (op.d / 2) * s, y: view.cy, shape: "ring", color: hole, cursor: "ew-resize", title: `hole ⌀ ${fmt(op.d, 1)} mm: drag` });
      }
      if (op.kind === "ring") {
        hs.push({ key: "stop-d", x: view.cx + Math.max((op.dIn / 2) * s, 7), y: view.cy, shape: "diamond", color: "#cdd6e4", cursor: "ew-resize", title: `central stop ⌀ ${fmt(op.dIn, 1)} mm: drag` });
      }
      if (op.kind === "rect") {
        hs.push({ key: "hole-w", x: view.cx + (op.w / 2) * s, y: view.cy, shape: "square", color: hole, cursor: "ew-resize", title: `hole width ${fmt(op.w, 1)} mm: drag` });
        hs.push({ key: "hole-h", x: view.cx, y: view.cy - (op.h / 2) * s, shape: "square", color: hole, cursor: "ns-resize", title: `hole height ${fmt(op.h, 1)} mm: drag` });
      }
      if (mk.kind === "circle") {
        hs.push({ key: "mask-d", x: view.cx - (mk.d / 2) * s, y: view.cy, shape: "ring", color: plate, cursor: "ew-resize", title: `mask ⌀ ${fmt(mk.d, 1)} mm: drag` });
      }
      if (mk.kind === "rect") {
        hs.push({ key: "mask-w", x: view.cx - (mk.w / 2) * s, y: view.cy, shape: "square", color: plate, cursor: "ew-resize", title: `mask width ${fmt(mk.w, 1)} mm: drag` });
        hs.push({ key: "mask-h", x: view.cx, y: view.cy + (mk.h / 2) * s, shape: "square", color: plate, cursor: "ns-resize", title: `mask height ${fmt(mk.h, 1)} mm: drag` });
      }
      return hs;
    },
    drag(hd, x, y, phase) {
      if (phase === "start") {
        view.held = true;
        return;
      }
      if (phase === "end") {
        view.held = false;
        ctx.edited(true);
        reread();
        ctx.redraw();
        return;
      }
      // Full sizes from the pointer's distance to the centre.
      const across = Math.max(0.2, (2 * Math.abs(x - view.cx)) / view.scale);
      const up = Math.max(0.2, (2 * Math.abs(y - view.cy)) / view.scale);
      const op = shape.opening;
      const mk = shape.mask;
      let next: ApertureShape | null = null;
      if (hd.key === "hole-d" && (op.kind === "circle" || op.kind === "ring")) next = roomForOpening({ ...shape, opening: { ...op, d: across } });
      if (hd.key === "stop-d" && op.kind === "ring") next = { ...shape, opening: { ...op, dIn: Math.min(across, op.d - 0.1) } };
      if (hd.key === "hole-w" && op.kind === "rect") next = roomForOpening({ ...shape, opening: { ...op, w: across } });
      if (hd.key === "hole-h" && op.kind === "rect") next = roomForOpening({ ...shape, opening: { ...op, h: up } });
      if (hd.key === "mask-d" && mk.kind === "circle") next = { ...shape, mask: { ...mk, d: across } };
      if (hd.key === "mask-w" && mk.kind === "rect") next = { ...shape, mask: { ...mk, w: across } };
      if (hd.key === "mask-h" && mk.kind === "rect") next = { ...shape, mask: { ...mk, h: up } };
      if (next) write(next, false);
    },
  };
}

/** A light source: its spectral lines, bar height by weight. The source panel
 *  on the right edits the first source. */
function sourceView(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  const first = () => (JSON.parse(scene.sourceObjectsJSON()) as { id: number }[])[0]?.id === id;
  return {
    hint: "the source panel on the right sets its shape, size and wavelengths",
    refresh() {
      const o = (JSON.parse(scene.sourceObjectsJSON()) as SceneObject[]).find((s) => s.id === id);
      ctx.read.innerHTML = o
        ? `${o.construction} aperture at (${o.pos.map((v) => fmt(v, 1)).join(", ")}) mm, aimed at ${fmt(o.rotZ, 1)}°`
        : "";
    },
    draw(c, w, h) {
      const lines = first() ? ((JSON.parse(scene.firstSourceWavelengthsJSON()) ?? []) as { nm: number; weight: number }[]) : [];
      const l = 20, r = 10, t = 14, b = 22;
      const X = (nm: number) => l + ((nm - 380) / 400) * (w - l - r);
      c.font = "10px ui-monospace, monospace";
      c.fillStyle = "rgba(205,214,228,0.55)";
      for (let nm = 400; nm <= 780; nm += 50) c.fillText(String(nm), X(nm) - 9, h - 6);
      c.strokeStyle = "rgba(120,150,190,0.25)";
      c.beginPath(); c.moveTo(l, h - b); c.lineTo(w - r, h - b); c.stroke();
      const peak = Math.max(1e-12, ...lines.map((x) => x.weight));
      for (const line of lines) {
        const x = Math.min(w - r, Math.max(l, X(line.nm)));
        const top = t + (1 - line.weight / peak) * (h - t - b);
        c.fillStyle = scene.wavelengthCssColor(line.nm);
        c.fillRect(x - 2, top, 4, h - b - top);
      }
      if (!lines.length) c.fillText("spectral lines are shown for the first source", l, t + 10);
    },
    handles: () => [],
    drag: () => {},
  };
}

/** Anything without an editor yet: what it is and where. */
function summaryView(ctx: EditorContext, o: SceneObject): Editor {
  const { scene, id } = ctx;
  return {
    hint: "no shape editor for this kind yet; its card (double-click it) has its fields",
    refresh() {
      const glass = JSON.parse(scene.bodyGlassJSON(id));
      const g = glass ? ` · glass n<sub>d</sub> <b>${fmt(glass.nd, 4)}</b> V<sub>d</sub> <b>${fmt(glass.vd, 1)}</b>` : "";
      ctx.read.innerHTML = `${o.kind === "profile" ? o.construction : o.kind} at (${o.pos.map((v) => fmt(v, 1)).join(", ")}) mm${g}`;
    },
    draw(c) {
      c.fillStyle = "rgba(205,214,228,0.45)";
      c.font = "11px ui-monospace, monospace";
      c.fillText(`${o.kind === "profile" ? o.construction : o.kind} #${id}`, 12, 20);
    },
    handles: () => [],
    drag: () => {},
  };
}
