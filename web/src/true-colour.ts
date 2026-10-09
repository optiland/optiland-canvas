// A detector's face as a person would see it: each pixel's CIE 1931 colour
// (X, Y, Z), which the tracer sums from every ray's wavelength, put on an
// sRGB screen.
//
// Two choices are the display's rather than the physics':
//
// - Exposure. The eye adapts, so a picture of 20 lux and one of 2000 look
//   alike. The brightest part of the face (its 99th percentile, so one hot
//   pixel does not set it) is drawn at full scale, times 2^EV. Brightest by
//   its largest display channel, not its luminance: a saturated red needs
//   1.7 times full red at a third of white's luminance, and exposing for
//   luminance would flatten the whole red end into one clipped colour.
// - Gamut. Spectral colours lie outside what sRGB's three primaries mix: the
//   sodium yellow needs negative blue. Such a colour is moved toward the grey
//   of its own luminance until it fits, which keeps how bright it is and its
//   hue and gives up saturation, the least visible of the three.

/** XYZ to linear sRGB, D65 (IEC 61966-2-1); the same matrix as
 *  `oc_core::colorimetry::xyz_to_linear_srgb`. */
function linearSrgb(x: number, y: number, z: number): [number, number, number] {
  return [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
}

const encode = (c: number) => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

/** Linear sRGB of one `(X, Y, Z)`, moved toward the grey of its own
 *  luminance until no channel is negative. Scaling the input scales the
 *  output: the move does not depend on brightness. */
function inGamut(x: number, y: number, z: number): [number, number, number] {
  let [r, g, b] = linearSrgb(x, y, z);
  const lo = Math.min(r, g, b);
  if (lo < 0 && y > 0) {
    const t = -lo / (y - lo);
    r += t * (y - r);
    g += t * (y - g);
    b += t * (y - b);
  }
  return [r, g, b];
}

/** The scale a face is exposed for: the 99th percentile, over its lit pixels,
 *  of each pixel's largest display channel, so a few hot pixels do not
 *  darken the rest. 0 for a dark face. */
export function exposureWhite(xyz: ArrayLike<number>): number {
  const peaks: number[] = [];
  for (let i = 0; i + 2 < xyz.length; i += 3) {
    if (!(xyz[i + 1] > 0)) continue;
    peaks.push(Math.max(...inGamut(xyz[i], xyz[i + 1], xyz[i + 2])));
  }
  if (!peaks.length) return 0;
  peaks.sort((a, b) => a - b);
  return peaks[Math.min(peaks.length - 1, Math.floor(0.99 * peaks.length))];
}

/** Each pixel's screen colour, three bytes per pixel, from its `(X, Y, Z)`
 *  flattened. `white` is the scale drawn at full scale (see
 *  [`exposureWhite`]) before `ev` stops of exposure. */
export function trueColourPixels(xyz: ArrayLike<number>, white: number, ev = 0): Uint8ClampedArray {
  const n = Math.floor(xyz.length / 3);
  const out = new Uint8ClampedArray(3 * n);
  if (!(white > 0)) return out;
  const k = Math.pow(2, ev) / white;
  for (let i = 0; i < n; i++) {
    const y = xyz[3 * i + 1] * k;
    if (!(y > 0)) continue;
    let [r, g, b] = inGamut(xyz[3 * i] * k, y, xyz[3 * i + 2] * k);
    // Past full scale, scale down whole so the hue holds.
    const hi = Math.max(r, g, b);
    if (hi > 1) {
      r /= hi;
      g /= hi;
      b /= hi;
    }
    out[3 * i] = encode(r);
    out[3 * i + 1] = encode(g);
    out[3 * i + 2] = encode(b);
  }
  return out;
}
