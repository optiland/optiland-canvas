// The flux heatmap colour ramp, shared by the detector dock and the 3D view's
// detector faces so both show a cell in the same colour.

// An inferno-like ramp: dark, violet, orange, white.
const STOPS: [number, number, number, number][] = [
  [0.0, 6, 4, 20], [0.25, 70, 20, 110], [0.5, 170, 45, 90],
  [0.75, 240, 120, 40], [1.0, 250, 250, 190],
];

/** The ramp at `t` in [0, 1], as sRGB bytes. */
export function heatRgb(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, r0, g0, b0] = STOPS[i - 1];
      const [t1, r1, g1, b1] = STOPS[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return [(r0 + (r1 - r0) * f) | 0, (g0 + (g1 - g0) * f) | 0, (b0 + (b1 - b0) * f) | 0];
    }
  }
  return [250, 250, 190];
}
