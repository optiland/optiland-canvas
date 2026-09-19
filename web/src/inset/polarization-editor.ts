// The polarizing-element editor: an ideal linear polarizer or an ideal linear
// retarder (a waveplate), face on.
//
// These two are unlike every other element the component view edits. A lens has
// a shape and a coating has a curve in wavelength; a polarizer has an **axis**,
// which is a direction in the element, and what it does depends on the state of
// the light reaching it rather than on any property of the surface alone. So
// there is no R or T to plot. What is drawn instead:
//
//   polarizer: the transmission axis across the face, with ◆ to turn it, and
//     Malus' law beside it, the fraction passed against the angle of an
//     incoming fully linear state. The flat line at 0.5 is what unpolarized
//     light gets, whatever the angle.
//   retarder: the fast axis, with ◆ to turn it, and the polarization ellipse
//     that linear light at 45 degrees to that axis comes out as. Drag ● around
//     the ellipse to set the retardance, and watch it open from a line at 0
//     into a circle at a quarter wave and close again at a half.
//
// The f32 preview has no polarization state, so it cannot show any of this: to
// it a polarizer is a 50% attenuator and a waveplate is a window. The note says
// so, because a picture that disagrees with the numbers should say which one is
// the model.

import {
  fmt,
  label,
  numberField,
  segmented,
  type DragPhase,
  type Editor,
  type EditorContext,
  type Handle,
  type NumberField,
} from "./ui";

type Crystal = {
  nO: number;
  nE: number;
  order: number;
  designLambdaUm: number;
  /** What those imply, computed in Rust so nothing here has to know how. */
  thicknessUm: number;
};

type Pol = {
  kind: "polarizer" | "retarder";
  axisDeg: number;
  /** On axis. When a table is present this is its first sample. */
  extinction: number | null;
  retardanceDeg: number | null;
  substrateIndex: number;
  crystal: Crystal | null;
  /** A measured contrast curve, or null for an ideal layer. */
  extinctionTable: { aoiDeg: number[]; extinction: number[] } | null;
};

/** The crystals a waveplate is usually cut from, at the sodium D line.
 *
 *  Quartz and magnesium fluoride are positive, so their optic axis is the slow
 *  one; calcite is negative, and its is the fast one. The element names its
 *  fast axis either way, so which it is only shows in how the retardance moves
 *  when the plate is tilted. */
const CRYSTALS: Record<string, { nO: number; nE: number; text: string; title: string }> = {
  quartz: {
    nO: 1.54424,
    nE: 1.55335,
    text: "quartz",
    title: "crystal quartz: the usual waveplate, birefringence 0.0091",
  },
  mgf2: {
    nO: 1.3777,
    nE: 1.3895,
    text: "MgF₂",
    title: "magnesium fluoride: birefringence 0.0118, so a thinner plate for the same retardance",
  },
  calcite: {
    nO: 1.6584,
    nE: 1.4864,
    text: "calcite",
    title: "calcite: negative and strongly birefringent, 0.172, so a very thin plate",
  },
};

/** Which entry a crystal's indices came from, or null for an ideal plate. */
function crystalKey(c: Crystal | null): string {
  if (!c) return "ideal";
  for (const [k, v] of Object.entries(CRYSTALS)) {
    if (Math.abs(v.nO - c.nO) < 1e-6 && Math.abs(v.nE - c.nE) < 1e-6) return k;
  }
  return "quartz";
}

const AXIS_COLOR = "#7fd6ff";
const CURVE_COLOR = "#ffd678";

export function createPolarizationEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  let pol: Pol = JSON.parse(scene.surfacePolarizationJSON(id));
  let axisField: NumberField | null = null;
  let valueField: NumberField | null = null;
  let substrateField: NumberField | null = null;
  let orderField: NumberField | null = null;
  let designField: NumberField | null = null;
  let crystalPick: { set: (v: string | null) => void } | null = null;

  // The two panels, in CSS px, set by `draw`: the face on the left and the
  // plot on the right.
  const face = { cx: 0, cy: 0, r: 0 };
  const plot = { l: 0, t: 0, w: 0, h: 0 };

  function reread() {
    const next = JSON.parse(scene.surfacePolarizationJSON(id));
    if (next) pol = next;
    axisField?.set(pol.axisDeg);
    if (pol.kind === "polarizer") valueField?.set(pol.extinction ?? 0);
    else valueField?.set(pol.retardanceDeg ?? 0);
    substrateField?.set(pol.substrateIndex ?? 1);
    crystalPick?.set(crystalKey(pol.crystal));
    orderField?.set(pol.crystal?.order ?? 0);
    designField?.set((pol.crystal?.designLambdaUm ?? 0.55) * 1000);
    showCrystalFields();
    writeReadout();
  }

  function setAxis(deg: number, settled: boolean) {
    if (!scene.setPolarizationAxis(id, deg)) {
      ctx.note("this element has no axis to turn", true);
      return;
    }
    pol.axisDeg = deg;
    ctx.edited(settled);
    writeReadout();
    ctx.redraw();
  }

  function setValue(v: number, settled: boolean) {
    const ok = pol.kind === "polarizer"
      ? scene.setPolarizerExtinction(id, v)
      : scene.setRetarderRetardance(id, v);
    if (!ok) {
      ctx.note("refused", true);
      return;
    }
    if (pol.kind === "polarizer") pol.extinction = Math.min(1, Math.max(0, v));
    else pol.retardanceDeg = v;
    ctx.edited(settled);
    writeReadout();
    ctx.redraw();
  }

  function setSubstrate(v: number, settled: boolean) {
    if (!scene.setPolarizingSubstrate(id, v)) {
      ctx.note("a substrate cannot be thinner than air", true);
      return;
    }
    pol.substrateIndex = v;
    ctx.edited(settled);
    writeReadout();
    ctx.redraw();
  }

  /** Rebuild the crystal, or drop it, from whatever the controls now say. */
  function setCrystal(key: string, order: number, designNm: number, settled: boolean) {
    const ok = key === "ideal"
      ? scene.clearRetarderCrystal(id)
      : scene.setRetarderCrystal(id, CRYSTALS[key].nO, CRYSTALS[key].nE, Math.max(0, Math.round(order)), designNm / 1000);
    if (!ok) {
      ctx.note("refused", true);
      return;
    }
    const next = JSON.parse(scene.surfacePolarizationJSON(id));
    if (next) pol = next;
    crystalPick?.set(crystalKey(pol.crystal));
    showCrystalFields();
    ctx.edited(settled);
    writeReadout();
    ctx.redraw();
  }

  /** The order and design wavelength only mean anything for a real plate. */
  function showCrystalFields() {
    const on = !!pol.crystal;
    for (const f of [orderField, designField]) {
      if (f) (f.input.closest(".cv-field") as HTMLElement | null)?.toggleAttribute("hidden", !on);
    }
  }

  // --- the controls -------------------------------------------------------

  axisField = numberField(ctx.fields, pol.kind === "polarizer" ? "axis" : "fast axis", "°", {
    commit: (v) => setAxis(v, true),
    step: 5,
    digits: 1,
    name: "axis",
    title: "degrees from the element's own local +X, measured in the plane of the plate",
  });

  if (pol.kind === "polarizer") {
    valueField = numberField(ctx.fields, "extinction", "", {
      commit: (v) => setValue(v, true),
      step: 1e-4,
      digits: 6,
      exponent: true,
      name: "extinction",
      title:
        "the power fraction of the blocked component that still gets through: 0 is perfect, 1e-4 an ordinary sheet. Typing here makes it one number at every angle; a measured curve is set through setPolarizerExtinctionTable",
    });
  } else {
    valueField = numberField(ctx.fields, "retardance", "°", {
      commit: (v) => setValue(v, true),
      step: 5,
      digits: 1,
      name: "retardance",
      title: "90° is a quarter wave, 180° a half wave",
    });
  }

  if (pol.kind === "retarder") {
    label(ctx.fields, "cut from");
    crystalPick = segmented(
      ctx.fields,
      [
        {
          value: "ideal",
          text: "ideal",
          title: "the retardance is the number above, at every wavelength and every angle",
        },
        ...Object.entries(CRYSTALS).map(([k, v]) => ({ value: k, text: v.text, title: v.title })),
      ],
      (v) => setCrystal(v, pol.crystal?.order ?? 0, (pol.crystal?.designLambdaUm ?? 0.55) * 1000, true),
    );
    orderField = numberField(ctx.fields, "order", "", {
      commit: (v) => setCrystal(crystalKey(pol.crystal), v, (pol.crystal?.designLambdaUm ?? 0.55) * 1000, true),
      step: 1,
      digits: 0,
      name: "order",
      title: "whole waves under the retardance above: 0 is a true zero-order plate, and a cheap quarter wave is often 5 or 10",
    });
    designField = numberField(ctx.fields, "design λ", "nm", {
      commit: (v) => setCrystal(crystalKey(pol.crystal), pol.crystal?.order ?? 0, v, true),
      step: 10,
      digits: 1,
      name: "design",
      title: "the wavelength the retardance above is quoted at",
    });
    crystalPick.set(crystalKey(pol.crystal));
    orderField.set(pol.crystal?.order ?? 0);
    designField.set((pol.crystal?.designLambdaUm ?? 0.55) * 1000);
    showCrystalFields();
  }

  substrateField = numberField(ctx.fields, "substrate", "n", {
    commit: (v) => setSubstrate(v, true),
    step: 0.05,
    digits: 4,
    name: "substrate",
    title:
      "index of the glass the element is laminated in, or 1 for a bare layer with no faces: the glass is what makes a real one care about the angle it is used at",
  });

  /** Whether the element is a bare layer or laminated in something. */
  function bare(): boolean {
    return (pol.substrateIndex ?? 1) <= 1 + 1e-9;
  }

  /** What the element's own faces do to the light, on axis. */
  function facesLine(): string {
    if (bare()) {
      return "bare layer: no faces, so nothing here changes with the angle but the axis you see";
    }
    const n = pol.substrateIndex;
    const r = ((n - 1) / (n + 1)) ** 2;
    return (
      `in n ${fmt(n, 3)} glass: its two faces pass ${fmt(100 * (1 - r) ** 2, 1)}% on axis,` +
      ` and off axis they pass p better than s`
    );
  }

  function writeReadout() {
    ctx.read.replaceChildren();
    const line = document.createElement("div");
    if (pol.kind === "polarizer") {
      const e = pol.extinction ?? 0;
      // What a real sheet is quoted as: the ratio of the two crossed states.
      const ratio = e > 0 ? `1 : ${fmt(1 / e, 0)}` : "perfect";
      const t = pol.extinctionTable;
      const where = t
        ? ` on axis, measured out to ${fmt(t.aoiDeg[t.aoiDeg.length - 1], 0)}°`
        : " at every angle";
      line.textContent =
        `polarizer · axis ${fmt(pol.axisDeg, 1)}° · extinction ${ratio}${where}` +
        ` · unpolarized light loses half its flux here`;
    } else {
      const d = pol.retardanceDeg ?? 0;
      const waves = d / 360;
      const name = near(d, 90) ? "quarter wave" : near(d, 180) ? "half wave" : near(d, 0) ? "no retardance" : `${fmt(waves, 3)} wave`;
      const loss = bare() ? "lossless" : "its faces reflect";
      const c = pol.crystal;
      const where = c
        ? `at ${fmt(c.designLambdaUm * 1000, 0)} nm on axis`
        : "at every wavelength and every angle";
      line.textContent =
        `retarder · fast axis ${fmt(pol.axisDeg, 1)}° · ${fmt(d, 1)}° (${name}) ${where}` +
        ` · ${loss}`;
    }
    ctx.read.appendChild(line);
    const c = pol.crystal;
    if (c) {
      const plate = document.createElement("div");
      const total = (pol.retardanceDeg ?? 0) / 360 + c.order;
      // How far the retardance moves for a given slip in wavelength or angle
      // is set by the whole retardance, not by the quarter wave you can see.
      plate.textContent =
        `${fmt(c.thicknessUm, 3)} µm of ${CRYSTALS[crystalKey(c)].text}, ${fmt(total, 3)} waves in all:` +
        ` it moves as 1/λ and as the square of the angle off axis, ${fmt(total * 4, 0)}× as far as a zero-order quarter wave`;
      ctx.read.appendChild(plate);
    }
    const faces = document.createElement("div");
    faces.textContent = facesLine();
    ctx.read.appendChild(faces);
  }

  writeReadout();

  // --- drawing ------------------------------------------------------------

  function draw(c: CanvasRenderingContext2D, w: number, h: number) {
    // Left panel: the element face on. Right panel: what it does.
    const half = Math.min(w * 0.42, h - 18);
    face.r = Math.max(18, half / 2 - 12);
    face.cx = 10 + face.r + 6;
    face.cy = h / 2;
    plot.l = face.cx + face.r + 26;
    plot.t = 16;
    plot.w = Math.max(30, w - plot.l - 12);
    plot.h = Math.max(24, h - plot.t - 24);

    drawFace(c);
    if (pol.kind === "retarder") drawEllipse(c);
    else if (pol.extinctionTable) drawContrast(c);
    else drawMalus(c);
  }

  /** The plate seen along the beam, with its axis across it. */
  function drawFace(c: CanvasRenderingContext2D) {
    c.save();
    c.beginPath();
    c.arc(face.cx, face.cy, face.r, 0, Math.PI * 2);
    c.fillStyle = "rgba(120,130,160,0.12)";
    c.fill();
    c.strokeStyle = "rgba(180,196,220,0.45)";
    c.lineWidth = 1;
    c.stroke();

    // The element's local +X, which the axis angle is measured from.
    c.beginPath();
    c.moveTo(face.cx - face.r, face.cy);
    c.lineTo(face.cx + face.r, face.cy);
    c.strokeStyle = "rgba(180,196,220,0.22)";
    c.setLineDash([3, 3]);
    c.stroke();
    c.setLineDash([]);

    const a = rad(pol.axisDeg);
    const dx = Math.cos(a) * face.r;
    // Screen y runs down, so a positive angle draws upward.
    const dy = -Math.sin(a) * face.r;
    c.beginPath();
    c.moveTo(face.cx - dx, face.cy - dy);
    c.lineTo(face.cx + dx, face.cy + dy);
    c.strokeStyle = AXIS_COLOR;
    c.lineWidth = 2;
    c.stroke();

    if (pol.kind === "polarizer") {
      // The blocked direction, drawn faintly: it is where the light goes.
      c.beginPath();
      c.moveTo(face.cx + dy, face.cy - dx);
      c.lineTo(face.cx - dy, face.cy + dx);
      c.strokeStyle = "rgba(255,122,92,0.35)";
      c.lineWidth = 1;
      c.stroke();
    }

    c.fillStyle = "rgba(205,214,228,0.75)";
    c.font = "10px ui-monospace, monospace";
    c.textAlign = "center";
    c.fillText(pol.kind === "polarizer" ? "pass" : "fast", face.cx, face.cy + face.r + 13);
    c.restore();
  }

  /** Malus: the fraction a fully linear input passes, against its angle. */
  function drawMalus(c: CanvasRenderingContext2D) {
    const X = (deg: number) => plot.l + (deg / 180) * plot.w;
    const Y = (v: number) => plot.t + (1 - v) * plot.h;
    axes(c, "0°", "180°", "1");

    const e = pol.extinction ?? 0;
    c.beginPath();
    for (let deg = 0; deg <= 180; deg += 2) {
      // Malus with a leak: cos² along the axis, plus the extinction of what is
      // blocked. At 90° the whole of it is the leak.
      const t = rad(deg - pol.axisDeg);
      const v = Math.cos(t) ** 2 + e * Math.sin(t) ** 2;
      const x = X(deg), y = Y(v);
      if (deg === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.strokeStyle = CURVE_COLOR;
    c.lineWidth = 1.5;
    c.stroke();

    // Unpolarized light: half, at every angle. The line the preview draws.
    c.beginPath();
    c.moveTo(X(0), Y(0.5 * (1 + e)));
    c.lineTo(X(180), Y(0.5 * (1 + e)));
    c.strokeStyle = "rgba(180,196,220,0.40)";
    c.setLineDash([4, 3]);
    c.lineWidth = 1;
    c.stroke();
    c.setLineDash([]);

    c.fillStyle = "rgba(180,196,220,0.65)";
    c.font = "9px ui-monospace, monospace";
    c.textAlign = "left";
    c.fillText("unpolarized", X(4), Y(0.5 * (1 + e)) - 4);
    c.textAlign = "center";
    c.fillText("linear in", plot.l + plot.w / 2, plot.t + plot.h + 15);
  }

  /** A measured contrast curve: what leaks past the blocked axis, against the
   *  angle the polarizer is used at.
   *
   *  Malus' law is not what is interesting once there is measured data. Every
   *  polarizer obeys Malus; what tells one apart from another is where it
   *  stops obeying it, and that is this curve. Drawn on a log scale, because
   *  an extinction table spans decades and a linear axis would show a flat
   *  line and then a cliff. */
  function drawContrast(c: CanvasRenderingContext2D) {
    const t = pol.extinctionTable!;
    const maxDeg = Math.max(10, t.aoiDeg[t.aoiDeg.length - 1]);
    // Decades from the best contrast measured down to a tenth, rounded out.
    const lo = Math.log10(Math.max(1e-8, Math.min(...t.extinction)));
    const hi = Math.log10(Math.max(1e-7, Math.max(...t.extinction)));
    const top = Math.ceil(hi);
    const bottom = Math.min(top - 1, Math.floor(lo));
    const X = (deg: number) => plot.l + (deg / maxDeg) * plot.w;
    const Y = (v: number) =>
      plot.t + (1 - (Math.log10(Math.max(1e-12, v)) - bottom) / (top - bottom)) * plot.h;

    axes(c, "0°", `${fmt(maxDeg, 0)}°`, `1e${top}`);

    c.beginPath();
    t.aoiDeg.forEach((deg, i) => {
      const x = X(deg), y = Y(t.extinction[i]);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    });
    c.strokeStyle = CURVE_COLOR;
    c.lineWidth = 1.5;
    c.stroke();

    // The samples themselves, because this is data and not a model: a reader
    // should see where the measurements are and where the line is a guess
    // between them.
    c.fillStyle = CURVE_COLOR;
    t.aoiDeg.forEach((deg, i) => {
      c.beginPath();
      c.arc(X(deg), Y(t.extinction[i]), 2, 0, Math.PI * 2);
      c.fill();
    });

    c.fillStyle = "rgba(180,196,220,0.65)";
    c.font = "9px ui-monospace, monospace";
    c.textAlign = "center";
    c.fillText("angle of incidence", plot.l + plot.w / 2, plot.t + plot.h + 15);
    c.textAlign = "left";
    c.fillText(`1e${bottom}`, plot.l + 2, plot.t + plot.h - 3);
  }

  /** The ellipse linear light at 45° to the fast axis comes out as. */
  function drawEllipse(c: CanvasRenderingContext2D) {
    const cx = plot.l + plot.w / 2;
    const cy = plot.t + plot.h / 2;
    const r = Math.max(10, Math.min(plot.w, plot.h) / 2 - 10);

    c.save();
    c.strokeStyle = "rgba(180,196,220,0.18)";
    c.lineWidth = 1;
    c.beginPath();
    c.arc(cx, cy, r, 0, Math.PI * 2);
    c.stroke();
    c.beginPath();
    c.moveTo(cx - r, cy); c.lineTo(cx + r, cy);
    c.moveTo(cx, cy - r); c.lineTo(cx, cy + r);
    c.stroke();
    // The dial's quarter-wave mark: the one setting anybody reaches for.
    c.beginPath();
    c.moveTo(cx, cy - r - 4);
    c.lineTo(cx, cy - r + 4);
    c.strokeStyle = "rgba(255,214,120,0.55)";
    c.lineWidth = 1.5;
    c.stroke();

    // Ex along the fast axis, Ey retarded: the curve the field tip traces.
    const d = rad(pol.retardanceDeg ?? 0);
    c.beginPath();
    for (let i = 0; i <= 96; i++) {
      const t = (i / 96) * Math.PI * 2;
      const ex = Math.cos(t);
      const ey = Math.cos(t - d);
      // Drawn in the fast-axis frame, then turned by the axis angle.
      const a = rad(pol.axisDeg);
      const px = ex * Math.cos(a) - ey * Math.sin(a);
      const py = ex * Math.sin(a) + ey * Math.cos(a);
      const x = cx + (px / Math.SQRT2) * r;
      const y = cy - (py / Math.SQRT2) * r;
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.closePath();
    c.strokeStyle = CURVE_COLOR;
    c.lineWidth = 1.5;
    c.stroke();

    c.fillStyle = "rgba(180,196,220,0.65)";
    c.font = "9px ui-monospace, monospace";
    c.textAlign = "center";
    c.fillText("linear at 45° in", cx, plot.t + plot.h + 15);
    c.restore();
  }

  function axes(c: CanvasRenderingContext2D, x0: string, x1: string, ymax: string) {
    c.save();
    c.strokeStyle = "rgba(180,196,220,0.25)";
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(plot.l, plot.t);
    c.lineTo(plot.l, plot.t + plot.h);
    c.lineTo(plot.l + plot.w, plot.t + plot.h);
    c.stroke();
    c.fillStyle = "rgba(180,196,220,0.65)";
    c.font = "9px ui-monospace, monospace";
    c.textAlign = "left";
    c.fillText(x0, plot.l, plot.t + plot.h + 15);
    c.fillText(ymax, plot.l - 8, plot.t + 4);
    c.textAlign = "right";
    c.fillText(x1, plot.l + plot.w, plot.t + plot.h + 15);
    c.restore();
  }

  // --- handles ------------------------------------------------------------

  function handles(): Handle[] {
    const out: Handle[] = [];
    const a = rad(pol.axisDeg);
    out.push({
      key: "axis",
      x: face.cx + Math.cos(a) * face.r,
      y: face.cy - Math.sin(a) * face.r,
      shape: "diamond",
      color: AXIS_COLOR,
      title: pol.kind === "polarizer"
        ? "the transmission axis: drag to turn it"
        : "the fast axis: drag to turn it",
      cursor: "grab",
    });
    if (pol.kind === "retarder") {
      // A point on the ellipse, dragged around to set the retardance.
      const cx = plot.l + plot.w / 2;
      const cy = plot.t + plot.h / 2;
      const r = Math.max(10, Math.min(plot.w, plot.h) / 2 - 10);
      const d = rad(pol.retardanceDeg ?? 0);
      // On the guide circle, which is a dial: the angle round it *is* the
      // retardance, so the handle sits where the number says and not on the
      // ellipse, whose own shape is the thing being set.
      out.push({
        key: "retardance",
        x: cx + r * Math.cos(d),
        y: cy - r * Math.sin(d),
        shape: "circle",
        color: CURVE_COLOR,
        title: "the retardance: drag round the dial (90° is a quarter wave, and circular)",
        cursor: "grab",
      });
    }
    return out;
  }

  function drag(h: Handle, x: number, y: number, phase: DragPhase) {
    const settled = phase === "end";
    if (h.key === "axis") {
      const deg = deg360(Math.atan2(face.cy - y, x - face.cx));
      setAxis(round1(deg), settled);
      return;
    }
    if (h.key === "retardance") {
      const cx = plot.l + plot.w / 2;
      const cy = plot.t + plot.h / 2;
      setValue(round1(deg360(Math.atan2(cy - y, x - cx))), settled);
    }
  }

  return {
    hint: pol.kind === "polarizer"
      ? "an ideal polarizer. Unpolarized light loses half its flux here: that is physics, not a modelled loss. The f32 preview has no polarization state, so it draws this as a plain 50% attenuator and two crossed plates stay bright there; the f64 readout is the one that knows."
      : "an ideal waveplate: lossless, zero thickness, and the same retardance at every wavelength. It is the polarization action of a plate and not the crystal, so there is no walk-off and no ordinary and extraordinary ray. The f32 preview draws it as a window.",
    refresh: reread,
    draw,
    handles,
    drag,
  };
}

const rad = (deg: number) => (deg * Math.PI) / 180;
const deg360 = (r: number) => ((r * 180) / Math.PI + 360) % 360;
const round1 = (v: number) => Math.round(v * 10) / 10;
const near = (v: number, target: number) => Math.abs(v - target) < 0.5;
