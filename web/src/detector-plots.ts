// What a detector's readout draws besides its face: the far-field grid of an
// intensity measure, cross-sections cut through either picture, and the plot
// axes they are drawn on. Shared by the detector dock and the component inset,
// so the two draw a histogram the same way.

import { exposureWhite, trueColourPixels } from "./true-colour";
import { drawFacePixels, type Box, type DetectorFace, type DetectorReadout, type FaceBox } from "./detector-face";
import { fmt, niceStep } from "./inset/ui";

/** A picture made of a grid of cells: the face's pixels, or the far field's
 *  direction cells. Lengths are in `unit` (mm on the face, degrees in the far
 *  field); the grid is centred on 0 and `v` grows upward. */
export type CellGrid = {
  values: ArrayLike<number>;
  nu: number;
  nv: number;
  width: number;
  height: number;
  unit: "mm" | "°";
  /** The face to draw the cells into: the real one, or a square standing in
   *  for the far field. */
  face: DetectorFace;
  /** The reading's unit, for the plot's axis. */
  reads: string;
  /** Scale the picture so this is the top of the ramp; null scales to the
   *  largest cell. */
  top: number | null;
  /** Each cell's screen colour, three bytes a cell, for a true-colour
   *  picture in place of the ramp. */
  rgb?: Uint8ClampedArray;
};

/** A square face `2·cone` degrees on a side, cut into the far field's cells,
 *  for drawing an angular grid with the face's own painter. */
export function angleFace(coneDeg: number, bins: [number, number]): DetectorFace {
  const side = 2 * coneDeg;
  return {
    shape: "rectangle",
    width: side,
    height: side,
    inner: 0,
    pixels: bins,
    pitch: [side / bins[0], side / bins[1]],
    pixelArea: 0,
    faceArea: 0,
    directionBins: null,
    maxCells: 0,
  };
}

/** The grid a readout's picture is made of, or null when there is none to
 *  cut (a polar intensity histogram, or a readout with no face). */
export function cellGrid(info: DetectorReadout, face: DetectorFace | null, exposureEv = 0): CellGrid | null {
  if (info.measure === "intensity") {
    if (info.directionLayout !== "grid") return null;
    const cone = info.coneDeg ?? 0;
    const bins = info.coneBins;
    return {
      values: info.cells,
      nu: bins[0],
      nv: bins[1],
      width: 2 * cone,
      height: 2 * cone,
      unit: "°",
      face: angleFace(cone, bins),
      reads: info.unit,
      top: null,
    };
  }
  if (!face) return null;
  if (info.measure === "illuminance" && info.colour) {
    return {
      values: info.cells,
      nu: face.pixels[0],
      nv: face.pixels[1],
      width: face.width,
      height: face.height,
      unit: "mm",
      face,
      reads: info.unit,
      top: info.peak,
      rgb: trueColourPixels(info.colour, exposureWhite(info.colour), exposureEv),
    };
  }
  const irradiance = info.measure === "irradiance";
  return {
    values: irradiance ? info.cells : info.incident,
    nu: face.pixels[0],
    nv: face.pixels[1],
    width: face.width,
    height: face.height,
    unit: "mm",
    face,
    reads: info.unit,
    top: irradiance ? info.peak : null,
  };
}

/** Draw a grid's cells into `area`, with its size under it. Returns where the
 *  grid landed. */
export function drawCellGrid(c: CanvasRenderingContext2D, g: CellGrid, area: Box, fixedScale?: number): FaceBox {
  const box = drawFacePixels(c, g.face, g.values, { ...area, h: area.h - 14 }, g.top, fixedScale, g.rgb);
  c.fillStyle = "rgba(205,214,228,0.55)";
  c.font = "10px ui-monospace, monospace";
  c.textAlign = "center";
  const text =
    g.unit === "°" ? `±${fmt(g.width / 2, 1)}° × ±${fmt(g.height / 2, 1)}° far field`
    : g.face.shape === "rectangle" ? `${fmt(g.width, 2)} × ${fmt(g.height, 2)} mm`
    : `⌀ ${fmt(g.width, 2)} mm`;
  c.fillText(text, box.x + box.w / 2, Math.min(area.y + area.h - 3, box.y + box.h + 12));
  if (g.unit === "°") {
    // The normal, where the far field is centred.
    c.strokeStyle = "rgba(127,214,255,0.25)";
    c.beginPath();
    c.moveTo(box.x + box.w / 2, box.y);
    c.lineTo(box.x + box.w / 2, box.y + box.h);
    c.moveTo(box.x, box.y + box.h / 2);
    c.lineTo(box.x + box.w, box.y + box.h / 2);
    c.stroke();
  }
  c.textAlign = "start";
  return box;
}

// ---------------------------------------------------------------------------
// Cross-sections
// ---------------------------------------------------------------------------

/** A cut through a grid: along a row (`u` varies, at height `at`) or down a
 *  column (`v` varies, at `at` across). `at` is in the grid's unit. */
export type Cut = { axis: "row" | "col"; at: number };

/** The row or column index a cut samples. */
export function cutIndex(g: CellGrid, cut: Cut): number {
  const [n, span] = cut.axis === "row" ? [g.nv, g.height] : [g.nu, g.width];
  return Math.min(n - 1, Math.max(0, Math.floor(((cut.at + span / 2) / span) * n)));
}

/** Where the cut's row or column is centred, in the grid's unit: the line the
 *  picture draws, since that is the line the numbers come from. */
export function cutCentre(g: CellGrid, cut: Cut): number {
  const [n, span] = cut.axis === "row" ? [g.nv, g.height] : [g.nu, g.width];
  return -span / 2 + ((cutIndex(g, cut) + 0.5) * span) / n;
}

/** The cells a cut passes through, as (position, reading) at each cell's
 *  centre. */
export function profile(g: CellGrid, cut: Cut): { x: number; y: number }[] {
  const k = cutIndex(g, cut);
  const out: { x: number; y: number }[] = [];
  if (cut.axis === "row") {
    for (let i = 0; i < g.nu; i++) out.push({ x: -g.width / 2 + ((i + 0.5) * g.width) / g.nu, y: g.values[k * g.nu + i] ?? 0 });
  } else {
    for (let j = 0; j < g.nv; j++) out.push({ x: -g.height / 2 + ((j + 0.5) * g.height) / g.nv, y: g.values[j * g.nu + k] ?? 0 });
  }
  return out;
}

/** The full width at half the peak of a profile, found by interpolating
 *  between cell centres on either side of the peak; null for a profile with
 *  no peak or one whose half height runs off an end. */
export function fwhm(p: { x: number; y: number }[]): number | null {
  let top = 0;
  let at = -1;
  p.forEach((q, i) => {
    if (q.y > top) {
      top = q.y;
      at = i;
    }
  });
  if (at < 0) return null;
  const half = top / 2;
  const cross = (step: number): number | null => {
    for (let i = at; i + step >= 0 && i + step < p.length; i += step) {
      const a = p[i], b = p[i + step];
      if (b.y <= half) return a.x + ((a.y - half) / (a.y - b.y)) * (b.x - a.x);
    }
    return null;
  };
  const lo = cross(-1), hi = cross(1);
  return lo == null || hi == null ? null : Math.abs(hi - lo);
}

/** The colours the cuts are drawn in, one per cut, in the order they were
 *  made. Chosen to read against the heat ramp. */
export const CUT_COLOURS = ["#7fd6ff", "#9bff8a", "#ff7fd0", "#ffe07f", "#b8a2ff", "#7fffe0"];

// ---------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------

export type Axes = {
  /** Data to screen. */
  X: (x: number) => number;
  Y: (y: number) => number;
  /** The plotting area inside the axes. */
  inner: Box;
};

/** Draw axes over `x` and `y` ranges into `area`, ticked at tidy steps and
 *  labelled with their units. */
export function drawAxes(
  c: CanvasRenderingContext2D,
  area: Box,
  x: [number, number],
  y: [number, number],
  xLabel: string,
  yLabel: string,
): Axes {
  const l = 44, r = 8, t = 14, b = 26;
  const inner = { x: area.x + l, y: area.y + t, w: Math.max(10, area.w - l - r), h: Math.max(10, area.h - t - b) };
  const [x0, x1] = x[1] > x[0] ? x : [x[0] - 1, x[0] + 1];
  const [y0, y1] = y[1] > y[0] ? y : [y[0] - 1, y[0] + 1];
  const X = (v: number) => inner.x + ((v - x0) / (x1 - x0)) * inner.w;
  const Y = (v: number) => inner.y + inner.h - ((v - y0) / (y1 - y0)) * inner.h;

  c.save();
  c.fillStyle = "#04060a";
  c.fillRect(inner.x, inner.y, inner.w, inner.h);
  c.font = "10px ui-monospace, monospace";
  c.lineWidth = 1;
  const xs = niceStep((x1 - x0) / Math.max(2, inner.w / 70));
  const ys = niceStep((y1 - y0) / Math.max(2, inner.h / 40));
  c.strokeStyle = "rgba(120,150,190,0.18)";
  c.fillStyle = "rgba(205,214,228,0.6)";
  c.textAlign = "center";
  c.beginPath();
  // Ticks by index, so a tick at zero is zero and not the rounding left
  // after adding a step to a negative start.
  const ticks = (lo: number, hi: number, step: number) => {
    const out: number[] = [];
    for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) out.push(Number((k * step).toPrecision(12)));
    return out;
  };
  for (const v of ticks(x0, x1, xs)) {
    const px = Math.round(X(v)) + 0.5;
    c.moveTo(px, inner.y);
    c.lineTo(px, inner.y + inner.h);
    c.fillText(fmt(v, 4), px, inner.y + inner.h + 12);
  }
  c.stroke();
  c.textAlign = "right";
  c.beginPath();
  for (const v of ticks(y0, y1, ys)) {
    const py = Math.round(Y(v)) + 0.5;
    c.moveTo(inner.x, py);
    c.lineTo(inner.x + inner.w, py);
    c.fillText(fmt(v, 4), inner.x - 4, py + 3);
  }
  c.stroke();
  c.strokeStyle = "rgba(127,214,255,0.35)";
  c.strokeRect(inner.x + 0.5, inner.y + 0.5, inner.w - 1, inner.h - 1);
  c.textAlign = "center";
  c.fillText(xLabel, inner.x + inner.w / 2, area.y + area.h - 2);
  c.textAlign = "left";
  c.fillText(yLabel, inner.x, area.y + 10);
  c.restore();
  return { X, Y, inner };
}
