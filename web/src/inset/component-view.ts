// The component view: the selected element, drawn on its own, with handles.
//
// It follows whichever view owns the canvas (2D cross-section or 3D model) and
// its selection, and re-reads the element whenever the scene revision moves,
// so an edit made anywhere shows here and an edit made here shows everywhere.
// Each kind of element has its own editor:
//   a singlet: its profile (lens-editor.ts);
//   a flat surface: its coating (coating-editor.ts);
//   an aperture: its opening, face on;
//   a detector: its last exact hit map, face on;
//   a source: its spectral lines;
//   anything else: a summary.
// Edits call `edited(false)` while a handle is held, so the main view draws the
// fast picture, and `edited(true)` when they end, for the exact trace.

import type { Canvas } from "../../pkg/oc_wasm.js";
import { heatRgb } from "../heat";
import type { SceneObject } from "../render3d/cards3d";
import { parseDetectorMaps } from "../render3d/types";
import { createCoatingEditor } from "./coating-editor";
import type { InsetContent } from "./inset";
import { createLensEditor } from "./lens-editor";
import {
  drawHandle,
  fmt,
  hitHandle,
  numberField,
  type DragPhase,
  type Editor,
  type EditorContext,
  type Handle,
} from "./ui";

export type ComponentView = InsetContent & {
  /** Re-read the selection and the scene if either changed (cheap otherwise). */
  sync: (force?: boolean) => void;
  /** What is shown, and the handles in page coordinates (tests). */
  debug: () => { id: number; kind: string; note: string; handles: { key: string; x: number; y: number }[] };
};

type EditorKind = "lens" | "coating" | "aperture" | "detector" | "source" | "summary" | "none";

export function createComponentView(opts: {
  scene: Canvas;
  body: HTMLElement;
  setSubtitle: (text: string) => void;
  /** The selected object id in the active view, or -1. */
  selected: () => number;
  edited: (settled: boolean) => void;
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
    if (JSON.parse(scene.singletShapeJSON(id, 2))) return "lens";
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
    const what = kind === "lens" ? "singlet" : o.kind === "profile" ? o.construction : o.kind;
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
    if (id !== shownId || kind !== shownKind) build(id, kind);
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

/** A detector, face on, with the incident-flux cells of the last exact trace.
 *  The numbers stay in the detector dock, which follows the selection. */
function detectorView(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  return {
    hint: "face on · the cells of the last exact trace; the numbers are in the detector dock",
    refresh() {
      ctx.read.textContent = "";
    },
    draw(c, w, h) {
      const map = parseDetectorMaps(scene.lastDetectorMapsF32()).find((m) => m.id === id);
      if (!map) {
        c.fillStyle = "rgba(205,214,228,0.45)";
        c.font = "11px ui-monospace, monospace";
        c.fillText("no hit map yet (an exact trace fills it)", 12, 20);
        return;
      }
      const aspect = map.halfW / map.halfH;
      const side = Math.min(w - 20, (h - 20) * aspect);
      const fw = side, fh = side / aspect;
      const x0 = (w - fw) / 2, y0 = (h - fh) / 2;
      let peak = 0;
      for (const v of map.cells) peak = Math.max(peak, v);
      c.fillStyle = "#050709";
      c.fillRect(x0, y0, fw, fh);
      const cw = fw / map.nu, ch = fh / map.nv;
      for (let v = 0; v < map.nv; v++) {
        for (let u = 0; u < map.nu; u++) {
          const val = map.cells[v * map.nu + u];
          if (!(val > 0) || !(peak > 0)) continue;
          const [r, g, b] = heatRgb(val / peak);
          c.fillStyle = `rgb(${r},${g},${b})`;
          // Detector v increases upward; the canvas y grows downward.
          c.fillRect(x0 + u * cw, y0 + (map.nv - 1 - v) * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      c.strokeStyle = "rgba(127,214,255,0.45)";
      c.strokeRect(x0 + 0.5, y0 + 0.5, fw - 1, fh - 1);
      ctx.read.textContent = `${map.nu} × ${map.nv} cells, scaled to this detector's brightest cell`;
    },
    handles: () => [],
    drag: () => {},
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
