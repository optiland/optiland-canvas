// 2D annotations drawn over the GPU 3D view (no three.js here).
//
// The overlay is the app's existing `#view` canvas, cleared to transparent while
// the 3D view is active. Text labels, the selection dot, TIR markers, and card
// leader lines stay crisp Canvas 2D; they are projected through the same camera
// math the GPU view uses (`camera.ts`).

import type { CameraRig, V2, V3 } from "./camera";

const LABEL_FONT = "11px ui-monospace, monospace";

/** The axis labels, in the UI's Zemax convention: world +X (the optical axis)
 *  reads "Z", +Y reads "Y", and +Z (out of the meridional plane) reads "X". */
const AXIS_LABELS: { label: string; color: string }[] = [
  { label: "Z", color: "rgba(150,175,210,0.85)" },
  { label: "Y", color: "rgba(150,210,175,0.85)" },
  { label: "X", color: "rgba(210,175,150,0.85)" },
];

export function drawAxisLabels(ctx: CanvasRenderingContext2D, rig: CameraRig, ends: V3[]) {
  const B = rig.basis();
  ctx.font = LABEL_FONT;
  ends.forEach((end, i) => {
    const p = rig.project(end, B);
    ctx.fillStyle = AXIS_LABELS[i].color;
    ctx.fillText(AXIS_LABELS[i].label, p.x + 3, p.y - 3);
  });
}

/** The selected object's centre, a small cyan dot (the drag handle). */
export function drawSelectionDot(ctx: CanvasRenderingContext2D, rig: CameraRig, centre: V3) {
  const c = rig.project(centre);
  ctx.fillStyle = "#7fd6ff";
  ctx.beginPath();
  ctx.arc(c.x, c.y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(6,8,13,0.8)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

/** Rings where a surface is in total internal reflection, as in the 2D view. */
export function drawTirRings(ctx: CanvasRenderingContext2D, points: V2[]) {
  if (points.length === 0) return;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(255,120,80,0.9)";
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.stroke();
  }
}

/** A centred notice (loading, renderer unavailable). */
export function drawNotice(ctx: CanvasRenderingContext2D, W: number, H: number, text: string, tone: "info" | "error") {
  ctx.font = "13px ui-monospace, monospace";
  const w = ctx.measureText(text).width + 28;
  const x = (W - w) / 2, y = H / 2 - 18;
  ctx.fillStyle = "rgba(10,14,22,0.9)";
  ctx.strokeStyle = tone === "error" ? "rgba(255,138,107,0.8)" : "rgba(58,134,176,0.8)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, 36, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = tone === "error" ? "#ff9b8a" : "#cdd6e4";
  ctx.fillText(text, x + 14, y + 23);
}
