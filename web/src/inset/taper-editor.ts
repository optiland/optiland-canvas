// The taper editor: a taper's (or light pipe's) profile, with handles, its
// sizes and its glass.
//
// In the taper's own frame (axis to the right, light entering the input face):
//   ● the input corner: drag up or down for the input radius; the input face
//     stays where it is in the scene;
//   ● the exit corner: drag for the length and the exit radius;
//   ○ a wall point between them: drag anywhere between its neighbours.
// Each corner and point has a mirror below the axis that drags the same way.
//
// The wall is straight between its points, each step a cone. "＋ point" bends
// it at the middle of its longest step, "− point" takes out the point that
// bends it least, "straighten" keeps only the two corners.
//
// Setting the wall angle keeps the input and the length and tilts the whole
// wall about the input corner, so a bent wall keeps its bends. The index is
// the body's glass: a catalog glass shows its name; typing n_d or V_d makes it
// an Abbe point (Cauchy dispersion through n_d and V_d).
//
// Every edit goes through `setTaper` or `setBodyGlassAbbe`, which refuse a wall
// that does not run forward along the axis or dips to it; a refused step
// leaves the scene as it was and says why.

import {
  button,
  fmt,
  niceStep,
  numberField,
  type DragPhase,
  type Editor,
  type EditorContext,
  type Handle,
} from "./ui";

type Pt = [number, number];
type Glass = { nd: number; vd: number; name: string | null };

const WALL = "#7fd6ff";
const FACE = "#ffb46b";
/** The least axial gap kept between neighbouring wall points while dragging. */
const GAP = 0.05;

export function createTaperEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  let wall: Pt[] = read();
  let glass: Glass | null = JSON.parse(scene.bodyIndexJSON(id));

  function read(): Pt[] {
    return (JSON.parse(scene.taperJSON(id))?.wall ?? []) as Pt[];
  }
  const x0 = () => wall[0][0];
  const len = () => wall[wall.length - 1][0] - wall[0][0];
  const rIn = () => wall[0][1];
  const rOut = () => wall[wall.length - 1][1];
  const angleDeg = () => (Math.atan2(rOut() - rIn(), len()) * 180) / Math.PI;

  // --- controls -------------------------------------------------------------
  button(ctx.bar, "＋ point", "bend the wall: a new point in the middle of its longest step", () => {
    let k = 0;
    for (let i = 1; i < wall.length - 1; i++) if (wall[i + 1][0] - wall[i][0] > wall[k + 1][0] - wall[k][0]) k = i;
    const mid: Pt = [(wall[k][0] + wall[k + 1][0]) / 2, (wall[k][1] + wall[k + 1][1]) / 2];
    commit([...wall.slice(0, k + 1), mid, ...wall.slice(k + 1)]);
  });
  button(ctx.bar, "− point", "take out the wall point that bends the wall least", () => {
    if (wall.length <= 2) return;
    let best = 1, bestD = Infinity;
    for (let i = 1; i < wall.length - 1; i++) {
      const [a, p, b] = [wall[i - 1], wall[i], wall[i + 1]];
      const d = Math.abs(p[1] - (a[1] + ((b[1] - a[1]) * (p[0] - a[0])) / (b[0] - a[0])));
      if (d < bestD) { bestD = d; best = i; }
    }
    commit(wall.filter((_, i) => i !== best));
  });
  button(ctx.bar, "straighten", "a straight wall from the input corner to the exit corner", () => {
    commit([wall[0], wall[wall.length - 1]]);
  });

  const fIn = numberField(ctx.fields, "Ø in", "mm", {
    name: "dIn", step: 0.5, title: "input face diameter",
    commit: (v) => commit(wall.map((p, i) => (i === 0 ? [p[0], v / 2] : p)), false),
  });
  const fOut = numberField(ctx.fields, "Ø out", "mm", {
    name: "dOut", step: 0.5, title: "exit face diameter",
    commit: (v) => commit(wall.map((p, i) => (i === wall.length - 1 ? [p[0], v / 2] : p)), false),
  });
  const fLen = numberField(ctx.fields, "L", "mm", {
    name: "length", step: 1, title: "length, input face to exit face (the wall points keep their share of it)",
    commit: (v) => commit(wall.map(([x, r]) => [x0() + ((x - x0()) * v) / len(), r]), false),
  });
  const fAng = numberField(ctx.fields, "angle", "°", {
    name: "angle", step: 0.5,
    title: "wall angle to the axis, input corner to exit corner (negative narrows); keeps the input and the length",
    commit: (v) => {
      const dr = rIn() + len() * Math.tan((v * Math.PI) / 180) - rOut();
      commit(wall.map(([x, r]) => [x, r + (dr * (x - x0())) / len()]), false);
    },
  });
  const fNd = numberField(ctx.fields, "n_d", "", {
    name: "nd", step: 0.01, digits: 5, title: "refractive index at the d line (587.6 nm)",
    commit: (v) => setGlass(v, glass?.vd ?? 57),
  });
  const fVd = numberField(ctx.fields, "V_d", "", {
    name: "vd", step: 1, digits: 2, title: "Abbe number (dispersion; larger is less)",
    commit: (v) => setGlass(glass?.nd ?? 1.49, v),
  });

  function setGlass(nd: number, vd: number) {
    if (!(nd >= 1 && vd > 0)) {
      ctx.note("refused: an index needs n_d ≥ 1 and V_d > 0", true);
      return;
    }
    if (!scene.setBodyGlassAbbe(id, nd, vd)) {
      ctx.note("refused: this body takes no glass", true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
    reread();
    ctx.redraw();
  }

  /** Write `next` as the wall; true if Rust took it. */
  function write(next: Pt[]): boolean {
    const why = scene.setTaper(id, new Float64Array(next.flat()));
    if (why) ctx.note(`refused: ${why}`, true);
    return !why;
  }

  function commit(next: Pt[], refit = true) {
    if (!write(next)) return;
    ctx.note("");
    ctx.edited(true);
    reread();
    if (refit) view.refit = true;
    else checkFit();
    ctx.redraw();
  }

  function reread() {
    const w = read();
    if (w.length >= 2) wall = w;
    glass = JSON.parse(scene.bodyIndexJSON(id));
    fIn.set(2 * rIn());
    fOut.set(2 * rOut());
    fLen.set(len());
    fAng.set(angleDeg());
    fNd.set(glass?.nd ?? null);
    fVd.set(glass?.vd ?? null);
    const area = (rOut() / rIn()) ** 2;
    const limit = rOut() > rIn() ? `${fmt((Math.asin(rIn() / rOut()) * 180) / Math.PI, 2)}°` : "none (it narrows)";
    const crit = glass && glass.nd > 1 ? `${fmt((Math.asin(1 / glass.nd) * 180) / Math.PI, 2)}°` : "—";
    const what = glass ? (glass.name ? `${glass.name} (catalog)` : "Abbe point") : "no glass";
    ctx.read.innerHTML =
      `area out/in <b>×${fmt(area, 3)}</b> · étendue limit in air <b>${limit}</b> · ` +
      `TIR beyond <b>${crit}</b> · ${wall.length - 2} bend${wall.length === 3 ? "" : "s"} · glass <b>${what}</b>` +
      `<br><span style="opacity:0.6">étendue limit: the narrowest half-angle light that fills the input over a` +
      ` hemisphere can leave the exit within, asin(r_in / r_out); TIR from n_d</span>`;
  }

  // --- view -----------------------------------------------------------------
  // Local mm to CSS px. A long, thin taper is stretched across its axis (up to
  // 4x, labelled) so its wall is there to grab. The view changes only on a fit.
  const view = { ox: 0, oy: 0, sx: 1, sy: 1, w: 0, h: 0, fitted: false, refit: false };
  const X = (x: number) => view.ox + x * view.sx;
  const Y = (y: number) => view.oy - y * view.sy;
  const localX = (px: number) => (px - view.ox) / view.sx;
  const localY = (py: number) => (view.oy - py) / view.sy;

  function rMax() {
    return Math.max(...wall.map((p) => p[1]));
  }

  const STRETCH_MAX = 4;
  function fit(w: number, h: number) {
    const fitX = (w - 50) / (Math.max(len(), 1) * 1.25);
    const fitY = (h - 30) / (2 * rMax() * 1.25);
    view.sx = Math.max(0.05, Math.min(fitX, fitY));
    view.sy = Math.max(view.sx, Math.min(fitY, view.sx * STRETCH_MAX));
    view.ox = w / 2 - (x0() + len() / 2) * view.sx;
    view.oy = h / 2 + 4;
    view.w = w;
    view.h = h;
    view.fitted = true;
    view.refit = false;
  }

  /** After an edit settles: refit if the taper left the drawing, or fills too
   *  little of it. */
  function checkFit() {
    if (!view.fitted) return;
    const xa = X(x0()), xb = X(x0() + len()), ya = Y(rMax()), yb = Y(-rMax());
    const outside = xa < 6 || xb > view.w - 6 || ya < 14 || yb > view.h - 4;
    const small = xb - xa < 0.3 * view.w && yb - ya < 0.3 * view.h;
    if (outside || small) view.refit = true;
  }

  let dragging: string | null = null;

  function draw(c: CanvasRenderingContext2D, w: number, h: number) {
    if (wall.length < 2) return;
    if (!dragging && (!view.fitted || view.refit || view.w !== w || view.h !== h)) fit(w, h);

    const stepX = niceStep(36 / view.sx);
    const stepY = niceStep(36 / view.sy);
    c.lineWidth = 1;
    c.strokeStyle = "rgba(120,150,190,0.08)";
    c.beginPath();
    for (let x = Math.floor(localX(0) / stepX) * stepX; X(x) <= w; x += stepX) {
      c.moveTo(Math.round(X(x)) + 0.5, 0);
      c.lineTo(Math.round(X(x)) + 0.5, h);
    }
    for (let y = Math.floor(localY(h) / stepY) * stepY; Y(y) >= 0; y += stepY) {
      c.moveTo(0, Math.round(Y(y)) + 0.5);
      c.lineTo(w, Math.round(Y(y)) + 0.5);
    }
    c.stroke();

    c.strokeStyle = "rgba(150,175,210,0.45)";
    c.setLineDash([6, 4]);
    c.beginPath(); c.moveTo(0, Y(0)); c.lineTo(w, Y(0)); c.stroke();
    c.setLineDash([]);

    // The body, then its wall and its two faces.
    c.beginPath();
    for (const [x, r] of wall) c.lineTo(X(x), Y(r));
    for (let i = wall.length - 1; i >= 0; i--) c.lineTo(X(wall[i][0]), Y(-wall[i][1]));
    c.closePath();
    c.fillStyle = "rgba(120,155,205,0.16)";
    c.fill();
    c.strokeStyle = WALL;
    c.lineWidth = 2;
    for (const s of [1, -1]) {
      c.beginPath();
      for (const [x, r] of wall) c.lineTo(X(x), Y(s * r));
      c.stroke();
    }
    c.strokeStyle = FACE;
    for (const [x, r] of [wall[0], wall[wall.length - 1]]) {
      c.beginPath(); c.moveTo(X(x), Y(r)); c.lineTo(X(x), Y(-r)); c.stroke();
    }

    c.fillStyle = "rgba(205,214,228,0.6)";
    c.font = "10px ui-monospace, monospace";
    c.fillText("in", X(x0()) - 14, Y(0) - 4);
    c.fillText("out", X(x0() + len()) + 4, Y(0) - 4);
    const bar = niceStep(50 / view.sy);
    c.strokeStyle = "rgba(205,214,228,0.6)";
    c.lineWidth = 1;
    c.beginPath(); c.moveTo(10, h - 8); c.lineTo(10, h - 8 - bar * view.sy); c.stroke();
    c.fillText(`${fmt(bar)} mm`, 14, h - 10);
    const stretch = view.sy / view.sx;
    if (stretch > 1.05) c.fillText(`radial ×${fmt(stretch, 1)}`, 8, 12);
  }

  function handles(): Handle[] {
    const out: Handle[] = [];
    wall.forEach(([x, r], i) => {
      const end = i === 0 || i === wall.length - 1;
      const what = i === 0
        ? `input corner, Ø ${fmt(2 * r, 3)} mm: drag up or down`
        : i === wall.length - 1
        ? `exit corner, Ø ${fmt(2 * r, 3)} mm at L ${fmt(len(), 3)} mm: drag`
        : `wall point at ${fmt(x - x0(), 3)} mm, r ${fmt(r, 3)} mm: drag`;
      for (const [suffix, s] of [["", 1], ["m", -1]] as const) {
        out.push({
          key: `p${i}${suffix}`, x: X(x), y: Y(s * r), shape: end ? "circle" : "ring", color: end ? FACE : WALL,
          cursor: i === 0 ? "ns-resize" : "move", title: what,
        });
      }
    });
    return out;
  }

  function onDrag(h: Handle, px: number, py: number, phase: DragPhase) {
    if (phase === "start") { dragging = h.key; return; }
    if (!dragging) return;
    if (phase === "end") {
      dragging = null;
      ctx.edited(true);
      reread();
      checkFit();
      ctx.redraw();
      return;
    }
    const i = Number(h.key.slice(1).replace("m", ""));
    const r = Math.abs(localY(py));
    let x = wall[i][0];
    if (i > 0) {
      const lo = wall[i - 1][0] + GAP;
      const hi = i < wall.length - 1 ? wall[i + 1][0] - GAP : Infinity;
      x = Math.min(hi, Math.max(lo, localX(px)));
    }
    const next = wall.map((p, k) => (k === i ? [x, r] : p)) as Pt[];
    if (write(next)) {
      ctx.note("");
      ctx.edited(false);
      wall = next;
      reread();
    }
    ctx.redraw();
  }

  reread();

  return {
    hint: "● input and exit corners · ○ wall points · fields for sizes, wall angle and glass",
    refresh: () => {
      if (dragging) return;
      reread();
      checkFit();
    },
    draw,
    handles,
    drag: onDrag,
  };
}
