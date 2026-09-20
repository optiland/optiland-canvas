// The component view: the selected element, drawn on its own, with handles.
//
// It follows whichever view owns the canvas (2D cross-section or 3D model) and
// its selection, and re-reads the element whenever the scene revision moves,
// so an edit made anywhere shows here and an edit made here shows everywhere.
// Each kind of element has its own editor:
//   a singlet: its profile (lens-editor.ts);
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
import {
  drawFacePixels,
  formatFlux,
  raysPerLitPixel,
  type DetectorFace,
  type DetectorReadout,
  type FaceBox,
} from "../detector-face";
import { drawPolarMap } from "../polarmap";
import { createCoatingEditor } from "./coating-editor";
import type { InsetContent } from "./inset";
import { createLensEditor } from "./lens-editor";
import { createPolarizationEditor } from "./polarization-editor";
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

type EditorKind = "lens" | "coating" | "polarizing" | "aperture" | "detector" | "source" | "summary" | "none";

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
    };
  }

  function build(id: number, kind: EditorKind) {
    bar.replaceChildren();
    fields.replaceChildren();
    read.replaceChildren();
    message = "";
    held = hot = null;
    const ctx = context(id);
    switch (kind) {
      case "lens": editor = createLensEditor(ctx, () => { readObjects(); return objectOf(id); }); break;
      case "coating": editor = createCoatingEditor(ctx); break;
      case "polarizing": editor = createPolarizationEditor(ctx); break;
      case "aperture": editor = apertureView(ctx, () => { readObjects(); return objectOf(id); }); break;
      case "detector": editor = detectorView(ctx); break;
      case "source": editor = sourceView(ctx); break;
      case "summary": editor = summaryView(ctx, objectOf(id)!); break;
      default: editor = null;
    }
    shownId = id;
    shownKind = kind;
  }

  function subtitle(id: number, kind: EditorKind) {
    const o = objectOf(id);
    if (!o || kind === "none") return "nothing selected";
    const what = kind === "lens"
      ? "singlet"
      // "surface" says nothing a user wants; name the part.
      : kind === "polarizing"
      ? (JSON.parse(scene.surfacePolarizationJSON(id))?.kind ?? "surface")
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
    const h = hitHandle(drawn, p.x, p.y);
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
    const h = hitHandle(drawn, p.x, p.y);
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

/** An aperture, face on: the mask, the clear opening, and an optional central
 *  stop, with ring handles for the opening's outer and inner radius. */
function apertureView(ctx: EditorContext, objectOf: () => SceneObject | undefined): Editor {
  const { scene, id } = ctx;
  let open = { inner: 0, outer: 1 };
  let mask = 2;
  // The scale is held while a ring is dragged: the mask can grow under the
  // pointer, and a rescale would pull the ring away from it.
  const view = { cx: 0, cy: 0, scale: 1, held: false };
  const fMin = numberField(ctx.fields, "min r", "mm", { name: "apMin", step: 0.5, commit: (v) => write(v, open.outer, true), title: "central stop radius (0: none)" });
  const fMax = numberField(ctx.fields, "max r", "mm", { name: "apMax", step: 0.5, commit: (v) => write(open.inner, v, true), title: "clear opening radius" });

  function reread() {
    const o = JSON.parse(scene.apertureOpeningJSON(id));
    if (!o) return;
    open = o;
    mask = Math.max(objectOf()?.radius ?? open.outer * 2.5, open.outer + 1);
    fMin.set(open.inner);
    fMax.set(open.outer);
    ctx.read.innerHTML = `opening <b>${fmt(open.outer, 2)}</b> mm${open.inner > 0 ? ` · central stop <b>${fmt(open.inner, 2)}</b> mm` : ""} · mask <b>${fmt(mask, 1)}</b> mm`;
  }

  function write(inner: number, outer: number, settled: boolean) {
    if (!scene.setApertureOpening(id, inner, outer)) {
      ctx.note("refused: the stop must stay inside the opening", true);
      return;
    }
    ctx.note("");
    ctx.edited(settled);
    reread();
    ctx.redraw();
  }

  reread();
  return {
    hint: "face on · drag the rings to size the opening and the central stop",
    refresh: reread,
    draw(c, w, h) {
      view.cx = w / 2;
      view.cy = h / 2;
      if (!view.held) view.scale = (0.44 * Math.min(w, h)) / mask;
      const R = (r: number) => r * view.scale;
      c.fillStyle = "rgba(40,44,52,0.95)";
      c.beginPath();
      c.arc(view.cx, view.cy, R(mask), 0, Math.PI * 2);
      c.arc(view.cx, view.cy, R(open.outer), 0, Math.PI * 2, true);
      c.fill("evenodd");
      c.strokeStyle = "rgba(210,165,120,0.9)";
      c.lineWidth = 1.5;
      c.beginPath(); c.arc(view.cx, view.cy, R(open.outer), 0, Math.PI * 2); c.stroke();
      if (open.inner > 0) {
        c.fillStyle = "rgba(40,44,52,0.95)";
        c.beginPath(); c.arc(view.cx, view.cy, R(open.inner), 0, Math.PI * 2); c.fill();
        c.stroke();
      }
      c.strokeStyle = "rgba(150,175,210,0.35)";
      c.setLineDash([4, 4]);
      c.beginPath(); c.moveTo(view.cx - R(mask), view.cy); c.lineTo(view.cx + R(mask), view.cy); c.stroke();
      c.setLineDash([]);
    },
    handles: () => [
      { key: "outer", x: view.cx + open.outer * view.scale, y: view.cy, shape: "ring", color: "#d2a578", cursor: "ew-resize", title: `opening ${fmt(open.outer, 2)} mm: drag` },
      { key: "inner", x: view.cx + Math.max(open.inner * view.scale, 7), y: view.cy - (open.inner > 0 ? 0 : 12), shape: "diamond", color: "#cdd6e4", cursor: "ew-resize", title: open.inner > 0 ? `central stop ${fmt(open.inner, 2)} mm: drag` : "drag outward to add a central stop" },
    ],
    drag(h, x, _y, phase) {
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
      const r = Math.abs(x - view.cx) / view.scale;
      if (h.key === "outer") write(open.inner, Math.max(r, open.inner + 0.05, 0.2), false);
      else write(r < 0.2 ? 0 : Math.min(r, open.outer - 0.05), open.outer, false);
    },
  };
}

/** A detector: its face (shape, physical size, pixels), what it measures,
 *  and the picture that measure makes.
 *
 *  The face is drawn to scale and the pixels are drawn as the grid they are,
 *  so a coarse grid looks coarse. Power and irradiance are the same face
 *  histogram with different divisors, so both draw the face; intensity is a
 *  different histogram, binned by the direction light arrived from, so it
 *  draws the polar map. The numbers and the picture come from one
 *  `detectorResultJSON` reply, the same one the dock reads, and the dock
 *  re-reads this view whenever it re-reads itself, so the flux printed here
 *  is the flux printed there. */
function detectorView(ctx: EditorContext): Editor {
  const { scene, id } = ctx;

  type Measure = DetectorReadout["measure"];
  type Shape = DetectorFace["shape"];
  let info: DetectorReadout | null = null;
  let face: DetectorFace | null = null;
  // Where the face was last drawn, and the scale held while a handle is
  // dragged so the face does not rescale under the pointer.
  let box: FaceBox | null = null;
  let held: number | null = null;

  const pick = segmented<Measure>(
    ctx.bar,
    [
      { value: "power", text: "power", title: "flux landing in each pixel" },
      { value: "irradiance", text: "irradiance", title: "each pixel's flux over its area on the face, per mm²" },
      {
        value: "intensity",
        text: "intensity",
        title: "flux per steradian, binned by the direction light arrived from",
      },
    ],
    (m) => {
      if (!scene.setDetectorMeasure(id, m, info?.coneDeg ?? 30)) {
        ctx.note("this detector cannot measure that", true);
        return;
      }
      ctx.note("");
      ctx.edited(true);
    },
  );
  label(ctx.bar, "face");
  const shapePick = segmented<Shape>(
    ctx.bar,
    [
      { value: "rectangle", text: "rect", title: "a rectangular face, as a camera sensor has" },
      { value: "disc", text: "disc", title: "a round face, as a power meter head or a photodiode has" },
      { value: "annulus", text: "ring", title: "a round face with a round hole in it" },
    ],
    (s) => reshape(s),
  );

  const fWidth = numberField(ctx.fields, "width", "mm", {
    name: "detWidth", step: 0.5, digits: 3,
    title: "the face's full width, along its local u axis",
    commit: (v) => face && writeFace({ ...face, width: v }, true),
  });
  const fHeight = numberField(ctx.fields, "height", "mm", {
    name: "detHeight", step: 0.5, digits: 3,
    title: "the face's full height, along its local v axis",
    commit: (v) => face && writeFace({ ...face, height: v }, true),
  });
  const fDiameter = numberField(ctx.fields, "⌀", "mm", {
    name: "detDiameter", step: 0.5, digits: 3,
    title: "the face's diameter",
    commit: (v) => face && writeFace({ ...face, width: v, height: v }, true),
  });
  const fHole = numberField(ctx.fields, "hole ⌀", "mm", {
    name: "detHole", step: 0.5, digits: 3,
    title: "the diameter of the hole in the ring",
    commit: (v) => face && writeFace({ ...face, inner: v }, true),
  });
  const fAcross = numberField(ctx.fields, "px across", "", {
    name: "detPixelsU", step: 1, digits: 0,
    title: "pixels across the face (u); the pitch is the width over this",
    commit: (v) => face && writePixels(v, face.pixels[1]),
  });
  const fDown = numberField(ctx.fields, "px down", "", {
    name: "detPixelsV", step: 1, digits: 0,
    title: "pixels down the face (v); the pitch is the height over this",
    commit: (v) => face && writePixels(face.pixels[0], v),
  });
  const cone = numberField(ctx.fields, "cone", "°", {
    name: "cone",
    step: 5,
    digits: 1,
    title: "the half angle the histogram spans, from the face normal",
    commit: (v) => {
      if (!scene.setDetectorMeasure(id, "intensity", v)) {
        ctx.note("a cone has to be inside (0, 90] degrees", true);
        return;
      }
      ctx.note("");
      ctx.edited(true);
    },
  });
  const fRings = numberField(ctx.fields, "rings", "", {
    name: "detRings", step: 1, digits: 0,
    title: "polar rings out to the cone: the resolution of I(θ)",
    commit: (v) => writeDirections(v, face?.directionBins?.[1] ?? 1),
  });
  const fSectors = numberField(ctx.fields, "sectors", "", {
    name: "detSectors", step: 1, digits: 0,
    title: "azimuth sectors around the normal; one sector is the I(θ) curve",
    commit: (v) => writeDirections(face?.directionBins?.[0] ?? 24, v),
  });

  const show = (f: { input: HTMLInputElement }, on: boolean) =>
    (f.input.closest(".cv-field") as HTMLElement | null)?.toggleAttribute("hidden", !on);

  /** Send a face to the scene. `settled` false while a handle is held: the
   *  fast picture follows, and the numbers wait for the release. A settled
   *  edit re-reads this view on its way through the dock (main.ts,
   *  `refresh3dDetector`), so nothing here traces a second time. */
  function writeFace(next: DetectorFace, settled: boolean): boolean {
    const ok =
      next.shape === "rectangle" ? scene.setDetectorRectangle(id, next.width, next.height)
      : next.shape === "disc" ? scene.setDetectorDisc(id, next.width)
      : scene.setDetectorAnnulus(id, next.inner, next.width);
    if (!ok) {
      ctx.note(
        next.shape === "annulus"
          ? "a ring needs a positive size and a hole smaller than itself"
          : "a face needs a positive size",
        true,
      );
      return false;
    }
    ctx.note("");
    ctx.edited(settled);
    if (!settled) readFace();
    ctx.redraw();
    return true;
  }

  /** Switch the face's shape, keeping it the size it was: the largest disc
   *  inside a rectangle, the square around a disc, and a ring whose hole is a
   *  third of it. The pixel counts stay. */
  function reshape(to: Shape) {
    if (!face || face.shape === to) return;
    const d = Math.min(face.width, face.height);
    const next: DetectorFace =
      to === "rectangle" ? { ...face, shape: to, width: face.width, height: face.height }
      : to === "disc" ? { ...face, shape: to, width: d, height: d }
      : { ...face, shape: to, width: d, height: d, inner: face.inner > 0 && face.inner < d ? face.inner : d / 3 };
    writeFace(next, true);
  }

  function writePixels(across: number, down: number) {
    if (!Number.isInteger(across) || !Number.isInteger(down)) {
      ctx.note("a pixel count is a whole number", true);
      return;
    }
    if (!scene.setDetectorPixels(id, across, down)) {
      const most = face?.maxCells ?? 0;
      ctx.note(`pixel counts start at 1, and a face holds at most ${most.toLocaleString("en")} pixels`, true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
  }

  function writeDirections(rings: number, sectors: number) {
    if (!Number.isInteger(rings) || !Number.isInteger(sectors)) {
      ctx.note("rings and sectors are whole numbers", true);
      return;
    }
    if (!scene.setDetectorDirectionBins(id, rings, sectors)) {
      ctx.note("rings and sectors start at 1", true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
  }

  /** The face alone: cheap, so a drag can follow it. */
  function readFace() {
    const raw = JSON.parse(scene.detectorFaceJSON(id));
    face = raw && typeof raw === "object" ? (raw as DetectorFace) : null;
    if (!face) return;
    shapePick.set(face.shape);
    const round = face.shape !== "rectangle";
    show(fWidth, !round);
    show(fHeight, !round);
    show(fDiameter, round);
    show(fHole, face.shape === "annulus");
    fWidth.set(face.width);
    fHeight.set(face.height);
    fDiameter.set(face.width);
    fHole.set(face.inner);
    fAcross.set(face.pixels[0]);
    fDown.set(face.pixels[1]);
    const angular = face.directionBins != null;
    show(fRings, angular);
    show(fSectors, angular);
    if (face.directionBins) {
      fRings.set(face.directionBins[0]);
      fSectors.set(face.directionBins[1]);
    }
  }

  /** The face and a fresh f64 readout of it. */
  function reread() {
    readFace();
    const text = ctx.readout(id);
    // No reference traced yet: the last readout stays until one is.
    if (text === null) return;
    const raw = JSON.parse(text);
    info = raw && typeof raw === "object" ? (raw as DetectorReadout) : null;
    pick.set(info?.measure ?? null);
    show(cone, info?.measure === "intensity");
    if (info?.coneDeg != null) cone.set(info.coneDeg);
    writeReadout();
  }

  const length = (mm: number) => (mm < 1 ? `${fmt(mm * 1000, 1)} µm` : `${fmt(mm, 3)} mm`);

  function writeReadout() {
    ctx.read.replaceChildren();
    const line = (html: string, warn = false) => {
      const div = document.createElement("div");
      div.innerHTML = html;
      if (warn) div.className = "warn";
      ctx.read.appendChild(div);
    };
    if (!info || !face) return;

    // The flux on the whole face first: the number the dock quotes, in the
    // same digits, whatever the measure divides it by.
    const rays = `${info.hits} ray${info.hits === 1 ? "" : "s"}`;
    if (info.hits === 0) {
      line("flux <b>0</b> · no rays on this face yet");
    } else if (info.measure === "power") {
      line(`flux <b>${formatFlux(info.totalIncident)}</b> on the face from ${rays} · brightest pixel <b>${formatFlux(info.peak)}</b>`);
    } else if (info.measure === "irradiance") {
      line(
        `flux <b>${formatFlux(info.totalIncident)}</b> from ${rays} · peak <b>${formatFlux(info.peak)}</b> ${info.unit}` +
          ` · face mean <b>${formatFlux(info.meanIrradiance)}</b>`,
      );
    } else {
      const [rings, sectors] = info.coneBins;
      const share = info.totalIncident > 0 ? info.outsideCone / info.totalIncident : 0;
      const past = share > 1e-9 ? `, ${(share * 100).toFixed(1)}% arrived past it` : "";
      line(
        `flux <b>${formatFlux(info.totalIncident)}</b> from ${rays} · peak <b>${formatFlux(info.peak)}</b> ${info.unit}` +
          ` over ${rings} ring${rings === 1 ? "" : "s"} × ${sectors} sector${sectors === 1 ? "" : "s"} to ${fmt(info.coneDeg ?? 0, 0)}°${past}`,
      );
    }

    // The instrument: its size, its pixels, and what a pixel is.
    const [nu, nv] = face.pixels;
    const [pu, pv] = face.pitch;
    const size =
      face.shape === "rectangle" ? `${fmt(face.width, 3)} × ${fmt(face.height, 3)} mm`
      : face.shape === "disc" ? `⌀ ${fmt(face.width, 3)} mm`
      : `⌀ ${fmt(face.width, 3)} mm, hole ⌀ ${fmt(face.inner, 3)} mm`;
    const pitch = Math.abs(pu - pv) <= 1e-9 * Math.max(pu, pv) ? length(pu) : `${length(pu)} × ${length(pv)}`;
    const rim = face.shape === "rectangle" ? "" : " · rim pixels read over their part on the face";
    line(`${size} · ${nu} × ${nv} px of ${pitch}${rim}`);

    // Whether a pixel's reading means much: the rays behind it. With k rays a
    // random source puts one standard deviation at 1/√k, and a grid source
    // can gain or lose a row and a column of rays, which is 2/√k; the larger
    // is the one quoted, since the app's sources sample on a grid.
    const k = raysPerLitPixel(info);
    if (k != null && info.measure !== "intensity") {
      const per = k < 10 ? fmt(k, 1) : String(Math.round(k));
      if (k < 10) {
        line(`${per} rays per lit pixel: single pixels are sampling noise; add rays or use fewer pixels`, true);
      } else {
        line(`${per} rays per lit pixel: sampling alone can move one pixel by about ±${fmt(200 / Math.sqrt(k), 0)}%`);
      }
    }
  }

  reread();

  return {
    hint: "face on, to scale · drag the corner or the rim to size the face",
    refresh: reread,
    draw(c, w, h) {
      if (!face) return;
      const area = { x: 0, y: 0, w, h: h - 14 };
      if (info?.measure === "intensity" && held == null) {
        drawPolarMap(
          c,
          { cells: info.cells, rings: info.coneBins[0], sectors: info.coneBins[1], coneDeg: info.coneDeg ?? 0 },
          { x: 0, y: 0, w, h },
        );
        box = null;
        return;
      }
      // While a handle is held the face is new and its numbers are not in
      // yet, so it is drawn empty rather than with the old face's pixels.
      const values = held != null || !info ? [] : info.measure === "irradiance" ? info.cells : info.incident;
      const scaleTo = info?.measure === "irradiance" ? info.peak : null;
      box = drawFacePixels(c, face, values, area, scaleTo, held ?? undefined);
      c.fillStyle = "rgba(205,214,228,0.55)";
      c.font = "10px ui-monospace, monospace";
      c.textAlign = "center";
      const label =
        face.shape === "rectangle" ? `${fmt(face.width, 2)} × ${fmt(face.height, 2)} mm`
        : `⌀ ${fmt(face.width, 2)} mm`;
      c.fillText(label, w / 2, Math.min(h - 3, box.y + box.h + 12));
      c.textAlign = "start";
    },
    handles: () => {
      if (!face || !box) return [];
      const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
      if (face.shape === "rectangle") {
        return [{ key: "corner", x: box.x + box.w, y: box.y, shape: "square", color: "#7fd6ff", cursor: "nesw-resize",
          title: `${fmt(face.width, 2)} × ${fmt(face.height, 2)} mm: drag the corner` }];
      }
      const out: Handle[] = [{ key: "rim", x: box.x + box.w, y: cy, shape: "ring", color: "#7fd6ff", cursor: "ew-resize",
        title: `⌀ ${fmt(face.width, 2)} mm: drag the rim` }];
      if (face.shape === "annulus") {
        out.push({ key: "hole", x: cx + (face.inner / 2) * box.scale, y: cy, shape: "diamond", color: "#cdd6e4", cursor: "ew-resize",
          title: `hole ⌀ ${fmt(face.inner, 2)} mm: drag` });
      }
      return out;
    },
    drag(hd, x, y, phase) {
      if (!face || !box) return;
      if (phase === "start") {
        held = box.scale;
        ctx.redraw();
        return;
      }
      if (phase === "end") {
        held = null;
        ctx.edited(true);
        ctx.redraw();
        return;
      }
      const s = held ?? box.scale;
      const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
      const du = Math.max(0.05, (2 * Math.abs(x - cx)) / s);
      const dv = Math.max(0.05, (2 * Math.abs(y - cy)) / s);
      if (hd.key === "corner") writeFace({ ...face, width: du, height: dv }, false);
      else if (hd.key === "rim") writeFace({ ...face, width: Math.max(du, face.inner + 0.05), height: Math.max(du, face.inner + 0.05) }, false);
      else writeFace({ ...face, inner: Math.min(du, face.width - 0.05) }, false);
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
