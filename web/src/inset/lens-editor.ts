// The lens editor: a singlet's meridional profile, drawn large, with handles.
//
// In the lens's own frame (optical axis to the right, light entering the front
// face):
//   ■ a face's vertex: drag sideways to change the centre thickness; the other
//     face stays where it is in the scene (the body moves by half the change);
//   ● a face's rim (sphere mode): drag sideways to bend the face, which sets
//     its radius and keeps its conic and aspheric terms;
//   ▲ the rim line: drag up or down to change the clear semi-aperture;
//   ○ control points (asphere mode): four per face at 1/4, 1/2, 3/4 and all of
//     the semi-aperture; dragging one reshapes the face through all four,
//     with the least-bending terms A4 to A18 on top of its unchanged conic.
//
// A lens is much taller than it is thick, so the axial direction is drawn
// stretched (up to 6x, labelled) to make sags visible and easy to grab.
//
// Every edit goes through a Rust setter, which refuses a shape whose faces
// cross; a refused step leaves the scene as it was and says so. The shape is
// re-read from Rust after each step, and the numbers shown (focal lengths,
// edge thickness) are Rust's.

import type { SceneObject } from "../render3d/cards3d";
import {
  button,
  fmt,
  label,
  niceStep,
  numberField,
  segmented,
  type DragPhase,
  type Editor,
  type EditorContext,
  type Handle,
  type NumberField,
} from "./ui";

type Face = { vertex: number; pts: number[] };
type Shape = {
  r1: number | null;
  r2: number | null;
  thickness: number;
  semiAperture: number;
  conic1: number;
  conic2: number;
  asph1: number[];
  asph2: number[];
  nd: number | null;
  efl: number | null;
  bfl: number | null;
  edge: number;
  faces: [Face, Face];
};

/** Profile samples per face, axis to rim. A multiple of four, so the control
 *  heights fall on samples. */
const SAMPLES = 64;
/** Control point heights, as fractions of the semi-aperture. */
const CTRL = [0.25, 0.5, 0.75, 1];
const FACE_COLOR = ["#7fd6ff", "#ffb46b"];

/** The wavelength a lens coating here is designed at: the middle of the
 *  visible, which is where a general-purpose AR coating is centred. */
const AR_LAMBDA_NM = 550;
const FACE_NAME = ["front", "back"];

type Mode = "sphere" | "asphere";
const MODE_KEY = "optiland-canvas:lens-editor-mode";

export function createLensEditor(ctx: EditorContext, objectOf: () => SceneObject | undefined): Editor {
  const { scene, id } = ctx;
  let shape: Shape = JSON.parse(scene.singletShapeJSON(id, SAMPLES));
  // The base conic under each face, when the face has aspheric terms.
  let conic: [number[] | null, number[] | null] = [null, null];
  let mode: Mode = "sphere";
  try {
    if (localStorage.getItem(MODE_KEY) === "asphere") mode = "asphere";
  } catch {
    // Storage refused: sphere mode.
  }
  let active: 0 | 1 = 0;

  // Local mm to CSS px, with separate axial (sx) and radial (sy) scales.
  // `shift` is how far the body centre has moved along its axis since a
  // thickness drag began; drawing with it keeps the held face still on
  // screen. The view only changes on a fit, never between a hover and a
  // press, so a handle stays where the pointer found it.
  const view = { ox: 0, oy: 0, sx: 1, sy: 1, w: 0, h: 0, fitted: false, refit: false };
  let shift = 0;
  const X = (x: number) => view.ox + (x + shift) * view.sx;
  const Y = (y: number) => view.oy - y * view.sy;
  const localX = (px: number) => (px - view.ox) / view.sx - shift;

  type Drag = {
    key: string;
    px0: number;
    t0: number;
    pos0: number[];
    axis: [number, number, number];
    sags: number[];
  };
  let drag: Drag | null = null;

  const hasTerms = (f: 0 | 1) => (f === 0 ? shape.asph1 : shape.asph2).some((v) => v !== 0);
  const heights = () => shape.faces[0].pts.filter((_, i) => i % 2 === 1);

  // --- controls -------------------------------------------------------------
  const modeSeg = segmented<Mode>(ctx.bar, [
    { value: "sphere", text: "sphere", title: "drag a face's rim to bend it (its radius)" },
    { value: "asphere", text: "asphere", title: "drag control points to reshape a face with aspheric terms" },
  ], (m) => {
    mode = m;
    try { localStorage.setItem(MODE_KEY, m); } catch { /* not remembered */ }
    modeSeg.set(m);
    ctx.note("");
    ctx.redraw();
  });
  label(ctx.bar, "face");
  const faceSeg = segmented<"0" | "1">(ctx.bar, [
    { value: "0", text: "front", title: "edit the front face's numbers" },
    { value: "1", text: "back", title: "edit the back face's numbers" },
  ], (v) => {
    active = v === "0" ? 0 : 1;
    faceSeg.set(v);
    refreshFields();
    ctx.redraw();
  });
  button(ctx.bar, "clear terms", "remove the selected face's aspheric terms (keeps its radius and conic)", () => {
    const zero = new Array(8).fill(0);
    commitShape(active === 0 ? { asph1: zero } : { asph2: zero }, true);
  });

  const fR = numberField(ctx.fields, "R", "mm", {
    name: "R",
    allowInfinite: true,
    step: 1,
    title: "radius of curvature (∞ or empty: flat); positive bends toward +X",
    commit: (v) => commitShape(active === 0 ? { r1: v } : { r2: v }),
  });
  const fK = numberField(ctx.fields, "k", "", {
    name: "k",
    step: 0.1,
    title: "conic constant (0 sphere, -1 paraboloid)",
    commit: (v) => commitShape(active === 0 ? { conic1: v } : { conic2: v }),
  });
  const fA: NumberField[] = [4, 6, 8, 10, 12, 14, 16, 18].map((order, j) =>
    numberField(ctx.fields, `A${order}`, "", {
      name: `A${order}`,
      exponent: true,
      title: `aspheric term: sag += A${order}·r^${order} (mm^-${order - 1})`,
      commit: (v) => {
        const terms = [...(active === 0 ? shape.asph1 : shape.asph2)];
        terms[j] = v;
        commitShape(active === 0 ? { asph1: terms } : { asph2: terms });
      },
    }),
  );
  const fT = numberField(ctx.fields, "t", "mm", {
    name: "t",
    step: 0.1,
    title: "centre thickness",
    commit: (v) => commitShape({ thickness: v }),
  });
  const fCa = numberField(ctx.fields, "ca", "mm", {
    name: "ca",
    step: 0.5,
    title: "clear semi-aperture",
    commit: (v) => commitShape({ semiAperture: v }),
  });

  // The coating follows whichever face is selected, the same way R and k do.
  // A real lens is coated a face at a time, and the two are often different.
  label(ctx.fields, "coat");
  const coatPick = segmented(
    ctx.fields,
    [
      { value: "none", text: "bare", title: "plain glass: about 4% of the light reflected at this face" },
      {
        value: "ar-1",
        text: "AR ×1",
        title: "a single quarter wave of magnesium fluoride, the classic single-layer AR: about 1.3% here, over a wide band",
      },
      {
        value: "ar-3",
        text: "AR ×3",
        title: "quarter-half-quarter, the broadband three-layer AR: lower again and flatter across the visible",
      },
    ],
    (v) => setCoating(v),
  );

  /** Coat, or strip, the face that is currently selected. */
  function setCoating(kind: string) {
    if (!scene.setLensFaceCoating(id, active === 0 ? "front" : "back", kind, AR_LAMBDA_NM)) {
      ctx.note("this face will not take a coating", true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
    reread();
    ctx.redraw();
  }

  /** What each face carries, as the JSON reports it. */
  function coatings(): { front: string | null; back: string | null } {
    return JSON.parse(scene.lensFaceCoatingsJSON(id)) ?? { front: null, back: null };
  }

  function refreshFields() {
    const f = active;
    fR.set(f === 0 ? shape.r1 ?? Infinity : shape.r2 ?? Infinity);
    fK.set(f === 0 ? shape.conic1 : shape.conic2);
    const terms = f === 0 ? shape.asph1 : shape.asph2;
    fA.forEach((field, j) => field.set(terms[j]));
    fT.set(shape.thickness);
    fCa.set(shape.semiAperture);
    for (const field of [fR, fK, ...fA]) field.input.style.borderColor = FACE_COLOR[f];
    const edgeWarn = shape.edge < 0.3 ? " (thin)" : "";
    const c = coatings();
    // Anything the segments do not offer (a dichroic's stack, say) selects
    // nothing rather than pretending to be "bare".
    const on = f === 0 ? c.front : c.back;
    coatPick.set(on === "ar-1" || on === "ar-3" ? on : on ? null : "none");
    const coatName = (v: string | null) =>
      v === "ar-1" ? "AR ×1" : v === "ar-3" ? "AR ×3" : v ? v : "bare";
    ctx.read.innerHTML =
      `n<sub>d</sub> <b>${fmt(shape.nd, 4)}</b> · EFL <b>${fmt(shape.efl, 2)}</b> mm · ` +
      `BFL <b>${fmt(shape.bfl, 2)}</b> mm · edge <b>${fmt(shape.edge, 2)}</b> mm${edgeWarn}` +
      ` · faces <b>${coatName(c.front)}</b> / <b>${coatName(c.back)}</b>` +
      `<br><span style="opacity:0.6">paraxial, d line, in the scene's ambient medium;` +
      ` a coating is solved against this lens's own glass at ${AR_LAMBDA_NM} nm on axis</span>`;
  }

  /** Write the whole shape with `patch` applied. False if Rust refused it. */
  function writeShape(patch: Partial<Shape>): boolean {
    const s = { ...shape, ...patch };
    return scene.setSingletShape(
      id,
      s.r1 ?? Infinity,
      s.r2 ?? Infinity,
      s.thickness,
      s.semiAperture,
      s.conic1,
      s.conic2,
      new Float64Array(s.asph1),
      new Float64Array(s.asph2),
    );
  }

  function commitShape(patch: Partial<Shape>, fromButton = false) {
    if (writeShape(patch)) {
      ctx.note("");
      ctx.edited(true);
      reread();
      checkFit();
      ctx.redraw();
    } else {
      ctx.note(fromButton ? "refused: the faces would cross" : "refused: that value makes the faces cross, or is not a number the lens accepts", true);
    }
  }

  function reread() {
    const s = JSON.parse(scene.singletShapeJSON(id, SAMPLES));
    if (!s) return;
    shape = s;
    const hs = new Float64Array(heights());
    conic = [0, 1].map((f) =>
      hasTerms(f as 0 | 1)
        ? Array.from(scene.singletFaceSagF64(id, f === 1, hs, true)).map((sag) => shape.faces[f].vertex + sag)
        : null,
    ) as [number[] | null, number[] | null];
    refreshFields();
  }

  // --- view -----------------------------------------------------------------
  function bounds() {
    let lo = Infinity, hi = -Infinity;
    for (const f of shape.faces) {
      for (let i = 0; i < f.pts.length; i += 2) {
        lo = Math.min(lo, f.pts[i]);
        hi = Math.max(hi, f.pts[i]);
      }
    }
    return { lo, hi, ha: shape.semiAperture };
  }

  const STRETCH_MAX = 6;
  function fit(w: number, h: number) {
    const { lo, hi, ha } = bounds();
    view.sy = Math.max(0.05, (h - 28) / (2 * ha * 1.18 + 2));
    // Axial room for the lens plus margins for dragging either face outward.
    const spanX = Math.max(hi - lo, 0.5) * 2.2 + 2;
    view.sx = Math.min(view.sy * STRETCH_MAX, Math.max(view.sy, (w - 40) / spanX));
    view.ox = w / 2 - ((lo + hi) / 2) * view.sx;
    view.oy = h / 2 + 4;
    view.w = w;
    view.h = h;
    view.fitted = true;
    view.refit = false;
  }

  /** After an edit settles: refit if the lens left the drawing, or fills
   *  too little of it. */
  function checkFit() {
    if (!view.fitted) return;
    const { lo, hi, ha } = bounds();
    const x0 = X(lo), x1 = X(hi), y0 = Y(ha), y1 = Y(-ha);
    const outside = x0 < 6 || x1 > view.w - 6 || y0 < 14 || y1 > view.h - 4;
    const small = x1 - x0 < 0.15 * view.w && y1 - y0 < 0.4 * view.h;
    if (outside || small) view.refit = true;
  }

  function draw(c: CanvasRenderingContext2D, w: number, h: number) {
    if (!drag && (!view.fitted || view.refit || view.w !== w || view.h !== h)) fit(w, h);

    // Grid, in mm, per axis.
    const stepX = niceStep(36 / view.sx);
    const stepY = niceStep(36 / view.sy);
    c.lineWidth = 1;
    c.strokeStyle = "rgba(120,150,190,0.08)";
    c.beginPath();
    for (let x = Math.floor(localX(0) / stepX) * stepX; X(x) <= w; x += stepX) {
      c.moveTo(Math.round(X(x)) + 0.5, 0);
      c.lineTo(Math.round(X(x)) + 0.5, h);
    }
    for (let y = Math.floor(((view.oy - h) / view.sy) / stepY) * stepY; Y(y) >= 0; y += stepY) {
      c.moveTo(0, Math.round(Y(y)) + 0.5);
      c.lineTo(w, Math.round(Y(y)) + 0.5);
    }
    c.stroke();

    // The optical axis.
    c.strokeStyle = "rgba(150,175,210,0.45)";
    c.setLineDash([6, 4]);
    c.beginPath(); c.moveTo(0, Y(0)); c.lineTo(w, Y(0)); c.stroke();
    c.setLineDash([]);

    // The body: front face rim to rim, then the back face rim to rim.
    const [fp, bp] = [shape.faces[0].pts, shape.faces[1].pts];
    const n = fp.length / 2 - 1;
    c.beginPath();
    for (let i = n; i >= 0; i--) c.lineTo(X(fp[2 * i]), Y(fp[2 * i + 1]));
    for (let i = 1; i <= n; i++) c.lineTo(X(fp[2 * i]), Y(-fp[2 * i + 1]));
    for (let i = n; i >= 0; i--) c.lineTo(X(bp[2 * i]), Y(-bp[2 * i + 1]));
    for (let i = 1; i <= n; i++) c.lineTo(X(bp[2 * i]), Y(bp[2 * i + 1]));
    c.closePath();
    c.fillStyle = "rgba(120,155,205,0.16)";
    c.fill();
    c.strokeStyle = "rgba(170,195,225,0.55)";
    c.lineWidth = 1;
    c.stroke();

    // Each face, and its base conic where aspheric terms bend it away.
    for (const f of [0, 1] as const) {
      const pts = shape.faces[f].pts;
      const base = conic[f];
      if (base) {
        c.strokeStyle = FACE_COLOR[f];
        c.globalAlpha = 0.45;
        c.setLineDash([3, 3]);
        c.beginPath();
        for (let i = 0; i <= n; i++) c.lineTo(X(base[i]), Y(pts[2 * i + 1]));
        c.moveTo(X(base[0]), Y(0));
        for (let i = 0; i <= n; i++) c.lineTo(X(base[i]), Y(-pts[2 * i + 1]));
        c.stroke();
        c.setLineDash([]);
        c.globalAlpha = 1;
      }
      c.strokeStyle = FACE_COLOR[f];
      c.lineWidth = f === active ? 2.5 : 1.5;
      c.beginPath();
      for (let i = n; i >= 0; i--) c.lineTo(X(pts[2 * i]), Y(pts[2 * i + 1]));
      for (let i = 1; i <= n; i++) c.lineTo(X(pts[2 * i]), Y(-pts[2 * i + 1]));
      c.stroke();
    }

    // The paraxial back focus, when it is on the drawing.
    if (shape.bfl != null && Number.isFinite(shape.bfl)) {
      const fx = X(shape.faces[1].vertex + shape.bfl);
      if (fx > 8 && fx < w - 8) {
        c.strokeStyle = "rgba(255,214,120,0.8)";
        c.lineWidth = 1.2;
        c.beginPath(); c.moveTo(fx - 5, Y(0) - 5); c.lineTo(fx + 5, Y(0) + 5); c.moveTo(fx + 5, Y(0) - 5); c.lineTo(fx - 5, Y(0) + 5); c.stroke();
        c.fillStyle = "rgba(255,214,120,0.85)";
        c.font = "10px ui-monospace, monospace";
        c.fillText("F′", fx + 6, Y(0) - 6);
      }
    }

    // A radial scale bar, and the axial stretch when there is one.
    const bar = niceStep(50 / view.sy);
    c.strokeStyle = "rgba(205,214,228,0.6)";
    c.lineWidth = 1;
    c.beginPath(); c.moveTo(10, h - 8); c.lineTo(10, h - 8 - bar * view.sy); c.stroke();
    c.fillStyle = "rgba(205,214,228,0.6)";
    c.font = "10px ui-monospace, monospace";
    c.fillText(`${fmt(bar)} mm`, 14, h - 10);
    const stretch = view.sx / view.sy;
    const label = stretch > 1.05 ? `axial ×${fmt(stretch, 1)} · ` : "";
    c.fillText(`${label}${mode === "sphere" ? "sphere mode" : "asphere mode"}`, 8, 12);
  }

  function handles(): Handle[] {
    const out: Handle[] = [];
    const ha = shape.semiAperture;
    for (const f of [0, 1] as const) {
      const face = shape.faces[f];
      const color = FACE_COLOR[f];
      const other = FACE_NAME[1 - f];
      out.push({
        key: `v${f}`, x: X(face.vertex), y: Y(0), shape: "square", color, cursor: "ew-resize",
        title: `${FACE_NAME[f]} vertex: drag sideways to change the thickness (the ${other} face stays put)`,
      });
      if (mode === "sphere") {
        const rim = face.pts[2 * SAMPLES];
        for (const [suffix, y] of [["", ha], ["m", -ha]] as const) {
          out.push({
            key: `b${f}${suffix}`, x: X(rim), y: Y(y), shape: "circle", color, cursor: "ew-resize",
            title: `${FACE_NAME[f]} rim: drag sideways to bend the face (R ${fmt(f === 0 ? shape.r1 ?? Infinity : shape.r2 ?? Infinity, 2)} mm)`,
          });
        }
      } else {
        CTRL.forEach((frac, j) => {
          const i = Math.round(frac * SAMPLES);
          for (const [suffix, sign] of [["", 1], ["m", -1]] as const) {
            out.push({
              key: `c${f}:${j}${suffix}`, x: X(face.pts[2 * i]), y: Y(sign * face.pts[2 * i + 1]), shape: "ring", color, cursor: "ew-resize",
              title: `${FACE_NAME[f]} control point at ${fmt(frac * ha, 2)} mm: drag sideways to reshape the face`,
            });
          }
        });
      }
    }
    const mid = (shape.faces[0].pts[2 * SAMPLES] + shape.faces[1].pts[2 * SAMPLES]) / 2;
    out.push({
      key: "ap", x: X(mid), y: Y(ha) - 12, shape: "triangle", color: "#cdd6e4", cursor: "ns-resize",
      title: `clear semi-aperture ${fmt(ha, 2)} mm: drag up or down`,
    });
    return out;
  }

  /** The body's world position and its axis (local +X) in world, now. */
  function placement(): { pos: number[]; axis: [number, number, number] } {
    const o = objectOf();
    const pos = o ? o.pos.slice(0, 3) : [0, 0, 0];
    const q = o?.rot ?? [0, 0, 0, 1];
    const [x, y, z, w] = q;
    // Rotate (1, 0, 0) by the unit quaternion.
    const axis: [number, number, number] = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
    return { pos, axis };
  }

  function dragMove(d: Drag, x: number, y: number): boolean {
    const k = d.key;
    if (k === "ap") {
      return writeShape({ semiAperture: Math.max(0.5, Math.abs((view.oy - y) / view.sy)) });
    }
    const f = Number(k[1]) as 0 | 1;
    if (k[0] === "v") {
      const dx = (x - d.px0) / view.sx;
      const t = f === 0 ? d.t0 - dx : d.t0 + dx;
      if (!writeShape({ thickness: t })) return false;
      // The other face keeps its place in the scene: move the body centre by
      // half the change, along the lens axis.
      const s = f === 0 ? -(t - d.t0) / 2 : (t - d.t0) / 2;
      scene.setObjectPosition(id, d.pos0[0] + d.axis[0] * s, d.pos0[1] + d.axis[1] * s, d.pos0[2] + d.axis[2] * s);
      shift = s;
      return true;
    }
    if (k[0] === "b") {
      return scene.bendSingletFace(id, f === 1, localX(x) - shape.faces[f].vertex);
    }
    // A control point: its new sag, and the other three where they were.
    const j = Number(k.slice(3, 4));
    d.sags[j] = localX(x) - shape.faces[f].vertex;
    const hs = new Float64Array(CTRL.map((frac) => frac * shape.semiAperture));
    return scene.fitSingletFace(id, f === 1, hs, new Float64Array(d.sags));
  }

  const refusal: Record<string, string> = {
    v: "refused: the faces would cross",
    b: "refused: that bend passes a hemisphere or makes the faces cross",
    c: "refused: that shape makes the faces cross",
    a: "refused: at that aperture the faces cross",
  };

  function onDrag(h: Handle, x: number, y: number, phase: DragPhase) {
    if (phase === "start") {
      const { pos, axis } = placement();
      const f = Number(h.key[1]) as 0 | 1;
      drag = { key: h.key, px0: x, t0: shape.thickness, pos0: pos, axis, sags: [] };
      if (h.key[0] === "c") {
        active = f;
        faceSeg.set(String(f) as "0" | "1");
        const hs = new Float64Array(CTRL.map((frac) => frac * shape.semiAperture));
        drag.sags = Array.from(scene.singletFaceSagF64(id, f === 1, hs, false));
      } else if (h.key[0] === "v" || h.key[0] === "b") {
        active = f;
        faceSeg.set(String(f) as "0" | "1");
      }
      return;
    }
    if (!drag) return;
    if (phase === "end") {
      // Fold the drag's shift into the view so nothing jumps.
      view.ox += shift * view.sx;
      shift = 0;
      drag = null;
      ctx.edited(true);
      reread();
      checkFit();
      ctx.redraw();
      return;
    }
    if (dragMove(drag, x, y)) {
      ctx.note("");
      ctx.edited(false);
      reread();
    } else {
      ctx.note(refusal[drag.key[0]] ?? "refused", true);
    }
    ctx.redraw();
  }

  modeSeg.set(mode);
  faceSeg.set("0");
  reread();

  return {
    hint: "■ thickness · ● bend (sphere mode) · ○ control points (asphere mode) · ▲ aperture",
    refresh: () => {
      if (drag) return;
      reread();
      checkFit();
    },
    draw,
    handles,
    drag: onDrag,
  };
}
