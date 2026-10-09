// The grating editor: a plane diffraction grating seen edge on, its grooves
// running into the page.
//
// A grating's recipe is its groove density, the order it is used in, whether
// it reflects or transmits, and how much light goes into that order and into
// the zero order. None of that is a curve in wavelength to plot the way a
// coating's is: what a grating does is send each wavelength a different way.
// So the drawing is the fan itself. An incoming ray arrives at an angle you
// drag (◆); every 25 nm from 400 to 700 leaves in its own colour where the
// grating equation puts it, the zero order dashed. A wavelength past grazing
// has no order and is marked. The angle is the drawing's alone: in the scene
// every ray meets the grating at its own angle.
//
// The efficiencies are declared, not computed (a real grating's depend on its
// groove profile and the polarization): a number here replaces a table.

import { fmt, label, numberField, segmented, type DragPhase, type Editor, type EditorContext, type Handle } from "./ui";

type Grating = {
  linesPerMm: number;
  order: number;
  transmissive: boolean;
  efficiency: number | null;
  zeroOrder: number | null;
  efficiencyTable: { lambdaUm: number[]; values: number[] } | null;
  zeroOrderTable: { lambdaUm: number[]; values: number[] } | null;
};

const FAN_NM = Array.from({ length: 13 }, (_, i) => 400 + 25 * i);

/** The angle a wavelength leaves at, from the grating's normal, degrees;
 *  null past grazing. Angles are signed along the grating: `alphaDeg` is the
 *  arriving ray's, positive when it travels the way a positive order turns
 *  light, and the result is the leaving ray's on the same scale. */
export function diffractedDeg(g: { linesPerMm: number; order: number }, alphaDeg: number, nm: number): number | null {
  const s = Math.sin((alphaDeg * Math.PI) / 180) + (g.order * g.linesPerMm * nm) / 1e6;
  return Math.abs(s) < 1 ? (Math.asin(s) * 180) / Math.PI : null;
}

export function createGratingEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  let g: Grating = JSON.parse(scene.gratingJSON(id));
  // The drawing's arriving angle: the Bragg angle at 550 nm to start with,
  // where a transmission grating is most efficient and a reflection grating
  // sends 550 nm straight back (Littrow).
  let alphaDeg = -(Math.asin(Math.min(0.99, (g.linesPerMm * 550) / 2e6)) * 180) / Math.PI;
  let box = { cx: 0, cy: 0, r: 1 };

  const read = () => {
    const next = JSON.parse(scene.gratingJSON(id));
    if (next) g = next;
  };
  const commit = (lines: number, order: number, transmissive: boolean) => {
    if (!scene.setGrating(id, lines, order, transmissive)) {
      ctx.note("refused: 1 to 20000 lines/mm and a nonzero order", true);
      return;
    }
    ctx.note("");
    read();
    ctx.edited(true);
    refresh();
  };
  const commitEta = (eta: number, zero: number) => {
    if (!scene.setGratingEfficiency(id, eta, zero)) {
      ctx.note("refused: each 0 to 1, and together at most 1", true);
      return;
    }
    ctx.note("");
    read();
    ctx.edited(true);
    refresh();
  };

  label(ctx.bar, "type");
  const kindPick = segmented<"reflect" | "transmit">(
    ctx.bar,
    [
      { value: "reflect", text: "reflection", title: "a reflection grating: the orders leave on the side the light arrives from; the back absorbs" },
      { value: "transmit", text: "transmission", title: "a transmission grating (a volume-phase hologram, say): the orders leave through it" },
    ],
    (k) => commit(g.linesPerMm, g.order, k === "transmit"),
  );

  const fLines = numberField(ctx.fields, "grooves", "/mm", {
    name: "gratingLines", step: 50, digits: 0,
    title: "grooves per millimetre, across the grating",
    commit: (v) => commit(v, g.order, g.transmissive),
  });
  const fOrder = numberField(ctx.fields, "order", "", {
    name: "gratingOrder", step: 1, digits: 0,
    title: "the diffraction order traced besides the zero order; its sign says which way it turns the light",
    commit: (v) => commit(g.linesPerMm, Math.round(v), g.transmissive),
  });
  const fEta = numberField(ctx.fields, "in order", "", {
    name: "gratingEfficiency", step: 0.01, digits: 3,
    title: "the power fraction sent into the order (a number here replaces a tabulated curve)",
    commit: (v) => commitEta(v, NaN),
  });
  const fZero = numberField(ctx.fields, "zero order", "", {
    name: "gratingZero", step: 0.01, digits: 3,
    title: "the power fraction left in the zero order: the mirror reflection, or the straight-through beam (a number here replaces a curve)",
    commit: (v) => commitEta(NaN, v),
  });
  const fAlpha = numberField(ctx.fields, "incidence", "°", {
    name: "gratingIncidence", step: 1, digits: 1,
    title: "the drawing's arriving angle, from the normal: the scene's rays each have their own",
    commit: (v) => {
      alphaDeg = Math.max(-89, Math.min(89, v));
      refresh();
    },
  });

  function refresh() {
    read();
    kindPick.set(g.transmissive ? "transmit" : "reflect");
    fLines.set(g.linesPerMm);
    fOrder.set(g.order);
    fEta.set(g.efficiency);
    fZero.set(g.zeroOrder);
    fAlpha.set(alphaDeg);
    writeReadout();
    ctx.redraw();
  }

  function writeReadout() {
    const lines: string[] = [];
    const out = FAN_NM.map((nm) => [nm, diffractedDeg(g, alphaDeg, nm)] as const);
    const live = out.filter(([, b]) => b != null) as (readonly [number, number])[];
    if (live.length) {
      const [n0, b0] = live[0], [n1, b1] = live[live.length - 1];
      lines.push(
        `${n0}–${n1} nm leave between <b>${fmt(b0, 1)}°</b> and <b>${fmt(b1, 1)}°</b>, a fan of ${fmt(Math.abs(b1 - b0), 1)}°`,
      );
    }
    const gone = out.filter(([, b]) => b == null).map(([nm]) => nm);
    if (gone.length) lines.push(`<span class="warn">no order at ${gone[0]}–${gone[gone.length - 1]} nm: past grazing</span>`);
    const b550 = diffractedDeg(g, alphaDeg, 550);
    if (b550 != null) {
      const disp = (Math.abs(g.order) * g.linesPerMm * 1e-6) / Math.cos((b550 * Math.PI) / 180);
      lines.push(`at 550 nm: ${fmt(disp * 1e3, 3)} mrad per nm; sodium's D lines ${fmt(disp * 0.597 * 1e3, 3)} mrad apart`);
    }
    const eta = g.efficiencyTable ? `tabulated (${fmt(g.efficiency ?? 0, 2)} at 550 nm)` : fmt(g.efficiency ?? 0, 3);
    const zero = g.zeroOrderTable ? `tabulated (${fmt(g.zeroOrder ?? 0, 2)} at 550 nm)` : fmt(g.zeroOrder ?? 0, 3);
    lines.push(`order ${g.order}: ${eta}; zero order: ${zero}; the rest is absorbed`);
    ctx.read.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
  }

  function draw(c: CanvasRenderingContext2D, w: number, h: number) {
    const r = Math.min(w * 0.42, h * (g.transmissive ? 0.38 : 0.52));
    const cx = w / 2, cy = g.transmissive ? h * 0.42 : h * 0.6;
    box = { cx, cy, r };
    const ray = (deg: number, side: 1 | -1, color: string, dash: number[] = [], width = 1.5) => {
      // `side` -1: the arriving side (up), +1 the far side (down).
      const t = (deg * Math.PI) / 180;
      c.save();
      c.strokeStyle = color;
      c.lineWidth = width;
      c.setLineDash(dash);
      c.beginPath();
      c.moveTo(cx, cy);
      c.lineTo(cx + r * Math.sin(t), cy + side * r * Math.cos(t));
      c.stroke();
      c.restore();
    };
    // The grating, edge on, and its normal.
    c.fillStyle = "#050709";
    c.fillRect(0, 0, w, h);
    c.strokeStyle = "rgba(127,214,255,0.25)";
    c.setLineDash([3, 4]);
    c.beginPath();
    c.moveTo(cx, cy - r);
    c.lineTo(cx, cy + r);
    c.stroke();
    c.setLineDash([]);
    c.strokeStyle = "#9fb3c8";
    c.lineWidth = 3;
    c.beginPath();
    c.moveTo(cx - r * 0.35, cy);
    c.lineTo(cx + r * 0.35, cy);
    c.stroke();
    // The arriving ray comes down from the upper side, travelling at
    // alpha: it is drawn back from the grating to where it came from.
    ray(-alphaDeg, -1, "#e8f6ff", [], 2);
    // Everything leaving travels at its own angle: down through a
    // transmission grating, back up off a reflection grating. The zero order
    // keeps the arriving angle, dashed.
    const side = g.transmissive ? 1 : -1;
    ray(alphaDeg, side, "rgba(232,246,255,0.55)", [5, 4]);
    for (const nm of FAN_NM) {
      const b = diffractedDeg(g, alphaDeg, nm);
      if (b != null) ray(b, side, scene.wavelengthCssColor(nm), [], 1.5);
    }
    c.fillStyle = "rgba(205,214,228,0.7)";
    c.font = "11px ui-monospace, monospace";
    c.fillText(`${g.linesPerMm} /mm, order ${g.order}`, 8, 16);
    c.fillText(g.transmissive ? "transmission" : "reflection", 8, 30);
  }

  function alphaHandle(): Handle {
    const t = (-alphaDeg * Math.PI) / 180;
    return {
      key: "alpha",
      x: box.cx + box.r * Math.sin(t),
      y: box.cy - box.r * Math.cos(t),
      shape: "diamond",
      color: "#e8f6ff",
      title: "drag to change the arriving angle (the drawing's only)",
      cursor: "grab",
    };
  }

  function drag(_h: Handle, x: number, y: number, _phase: DragPhase) {
    const dx = x - box.cx, dy = box.cy - y;
    if (dy <= 0) return;
    alphaDeg = Math.max(-89, Math.min(89, (-Math.atan2(dx, dy) * 180) / Math.PI));
    fAlpha.set(alphaDeg);
    writeReadout();
    ctx.redraw();
  }

  refresh();
  return {
    hint: "edge on, grooves into the page · drag ◆ to change the arriving angle",
    refresh,
    draw,
    handles: () => [alphaHandle()],
    drag,
  };
}
