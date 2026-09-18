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
  numberField,
  type DragPhase,
  type Editor,
  type EditorContext,
  type Handle,
  type NumberField,
} from "./ui";

type Pol = {
  kind: "polarizer" | "retarder";
  axisDeg: number;
  extinction: number | null;
  retardanceDeg: number | null;
  substrateIndex: number;
};

const AXIS_COLOR = "#7fd6ff";
const CURVE_COLOR = "#ffd678";

export function createPolarizationEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  let pol: Pol = JSON.parse(scene.surfacePolarizationJSON(id));
  let axisField: NumberField | null = null;
  let valueField: NumberField | null = null;
  let substrateField: NumberField | null = null;

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
      title: "the power fraction of the blocked component that still gets through: 0 is perfect, 1e-4 an ordinary sheet",
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
      line.textContent =
        `polarizer · axis ${fmt(pol.axisDeg, 1)}° · extinction ${ratio}` +
        ` · unpolarized light loses half its flux here`;
    } else {
      const d = pol.retardanceDeg ?? 0;
      const waves = d / 360;
      const name = near(d, 90) ? "quarter wave" : near(d, 180) ? "half wave" : near(d, 0) ? "no retardance" : `${fmt(waves, 3)} wave`;
      const loss = bare() ? "lossless, and" : "";
      line.textContent =
        `retarder · fast axis ${fmt(pol.axisDeg, 1)}° · ${fmt(d, 1)}° (${name})` +
        ` · ${loss} the same at every wavelength`.replace("·  ", "· ");
    }
    ctx.read.appendChild(line);
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
    if (pol.kind === "polarizer") drawMalus(c);
    else drawEllipse(c);
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
