// The angular histogram a detector fills when it measures radiant intensity,
// drawn the way a goniometer plots one: polar angle outward from the middle,
// azimuth around it.
//
// The radius is the polar angle itself, not its sine. A sine-space plot is
// what a conoscope shows and is the right picture for an interference figure;
// this one is for reading degrees off, so degrees are what the radius is.
//
// Shared by the detector dock and the component inset so the same histogram
// cannot be drawn two different ways.

import { heatRgb } from "./heat";

export type PolarMap = {
  /** Cell values, row-major as `sector * rings + ring`. */
  cells: number[];
  rings: number;
  sectors: number;
  /** The cone the outermost ring reaches, degrees. */
  coneDeg: number;
};

/** Draw `map` into the largest circle that fits the box, with a scale ring and
 *  its angle in degrees. Cells are scaled to the brightest one. */
export function drawPolarMap(
  c: CanvasRenderingContext2D,
  map: PolarMap,
  box: { x: number; y: number; w: number; h: number },
) {
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  const radius = Math.max(8, Math.min(box.w, box.h) / 2 - 12);

  c.fillStyle = "#04060a";
  c.beginPath();
  c.arc(cx, cy, radius, 0, Math.PI * 2);
  c.fill();

  let peak = 0;
  for (const v of map.cells) if (v > peak) peak = v;

  if (peak > 0) {
    const dphi = (Math.PI * 2) / map.sectors;
    for (let s = 0; s < map.sectors; s++) {
      for (let r = 0; r < map.rings; r++) {
        const v = map.cells[s * map.rings + r];
        if (!(v > 0)) continue;
        const r0 = (radius * r) / map.rings;
        const r1 = (radius * (r + 1)) / map.rings;
        const [red, green, blue] = heatRgb(v / peak);
        c.fillStyle = `rgb(${red},${green},${blue})`;
        c.beginPath();
        if (map.sectors === 1) {
          // One sector is a full annulus, and an arc pair would leave a seam.
          c.arc(cx, cy, r1, 0, Math.PI * 2);
          c.arc(cx, cy, r0, Math.PI * 2, 0, true);
        } else {
          // Canvas y grows downward, so a positive azimuth has to turn the
          // other way for the picture to match the detector's own frame.
          const a0 = -(s + 1) * dphi;
          const a1 = -s * dphi;
          c.arc(cx, cy, r1, a0, a1);
          c.arc(cx, cy, r0, a1, a0, true);
        }
        c.closePath();
        c.fill();
      }
    }
  }

  // The rim, a half-angle ring to read against, and the cone in degrees.
  c.strokeStyle = "rgba(127,214,255,0.45)";
  c.beginPath();
  c.arc(cx, cy, radius, 0, Math.PI * 2);
  c.stroke();
  c.strokeStyle = "rgba(127,214,255,0.18)";
  c.beginPath();
  c.arc(cx, cy, radius / 2, 0, Math.PI * 2);
  c.moveTo(cx - radius, cy);
  c.lineTo(cx + radius, cy);
  c.moveTo(cx, cy - radius);
  c.lineTo(cx, cy + radius);
  c.stroke();

  c.fillStyle = "rgba(205,214,228,0.6)";
  c.font = "10px ui-monospace, monospace";
  c.textAlign = "center";
  c.fillText(`${map.coneDeg.toFixed(0)}°`, cx, cy + radius + 11);
  c.textAlign = "left";
}
