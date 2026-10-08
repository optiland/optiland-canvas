// A detector's face as a picture, and the flux it caught as a number.
//
// The detector dock and the component inset both draw a detector with this,
// from the same `detectorResultJSON` reply, so they cannot show two different
// pictures or print two different totals. The face is drawn in its true
// proportions (a 20 x 5 mm sensor is four times wider than tall), clipped to
// the shape light can land on, with the pixels painted through an ImageData so
// a 1024 x 1024 grid costs one drawImage rather than a million fillRects.

import { heatRgb } from "./heat";

/** `detectorFaceJSON`, and the `face` field of `detectorResultJSON`. Lengths
 *  are mm, areas mm². `width` and `height` are the full size; `inner` is an
 *  annulus's hole diameter, 0 otherwise. */
export type DetectorFace = {
  shape: "rectangle" | "disc" | "annulus";
  width: number;
  height: number;
  inner: number;
  /** Pixel counts across (u) and down (v). */
  pixels: [number, number];
  /** Pixel size along u and v. */
  pitch: [number, number];
  pixelArea: number;
  faceArea: number;
  /** `[rings, sectors]` under the intensity measure, else null. */
  directionBins: [number, number] | null;
  maxCells: number;
};

/** The parts of `detectorResultJSON` the pictures and the numbers read. */
export type DetectorReadout = {
  hits: number;
  /** The flux on the whole face: the one number every view quotes. */
  totalIncident: number;
  measure: "power" | "irradiance" | "intensity" | "illuminance";
  unit: string;
  /** Under illuminance, each pixel's CIE 1931 `(X, Y, Z)` scaled like the
   *  lux in `cells`, flattened; null under any other measure. */
  colour: number[] | null;
  bins: [number, number];
  /** Flux per pixel, whatever the measure. */
  incident: number[];
  /** The readings in the measure's unit: per pixel, or per direction cell. */
  cells: number[];
  coneDeg: number | null;
  coneBins: [number, number];
  outsideCone: number;
  peak: number | null;
  meanIrradiance: number | null;
  pixelArea: number | null;
  faceArea: number | null;
  face: DetectorFace | null;
  /** Flux-weighted centroid on the face, mm, local (u, v). */
  centroid: [number, number] | null;
  meanOpl: number | null;
  dop: number | null;
  dolp: number | null;
  azimuthDeg: number | null;
  maxTiltDeg: number | null;
  /** How an intensity histogram is laid out: `grid` (two angles, `cells`
   *  row-major like the face's pixels over `coneBins` = [across, down]) or
   *  `polar` (rings and sectors); null for any other measure. */
  directionLayout: "grid" | "polar" | null;
  /** The hits by wavelength, in order of wavelength. */
  spectral: SpectralLine[];
  /** Flux on lines past the ones broken out (only a source with more lines
   *  than the tracer keeps has any). */
  spectralUntracked: number;
};

/** One wavelength's hits on a detector. Lengths are mm and angles radians:
 *  `angle` is the mean arrival direction's angle to the normal in the `u` and
 *  `v` planes, `spread` the RMS angular spread about it, and `focus` how far
 *  along the normal, the way light travels, this line's spot is smallest
 *  (null when its rays are all parallel). */
export type SpectralLine = {
  nm: number;
  hits: number;
  flux: number;
  centroid: [number, number];
  rms: number | null;
  angle: [number, number];
  spread: number | null;
  focus: number | null;
  /** The raw sums behind the numbers: hits, Σw, Σwu, Σwv, Σwu², Σwv², Σwa,
   *  Σwb, Σwa², Σwb², Σwua, Σwvb, with `a, b` the slopes. Bins merge
   *  exactly by adding these. */
  sums?: number[];
};

/** Bins merged into one: their sums added, and the numbers taken again from
 *  the total, exactly as the tracer takes them from one bin's. */
export function mergeLines(bins: SpectralLine[]): SpectralLine {
  const s = new Array(12).fill(0);
  let nmFlux = 0;
  for (const b of bins) {
    b.sums?.forEach((v, i) => (s[i] += v));
    nmFlux += b.nm * b.flux;
  }
  const [hits, w, u, v, uu, vv, a, b_, aa, bb, ua, vb] = s;
  const m = (x: number) => x / w;
  const [mu, mv, ma, mb] = [m(u), m(v), m(a), m(b_)];
  const varU = Math.max(0, m(uu) - mu * mu), varV = Math.max(0, m(vv) - mv * mv);
  const varA = Math.max(0, m(aa) - ma * ma), varB = Math.max(0, m(bb) - mb * mb);
  const cov = m(ua) - mu * ma + (m(vb) - mv * mb);
  const spread = varA + varB;
  return {
    nm: w > 0 ? nmFlux / w : bins[0]?.nm ?? 0,
    hits,
    flux: w,
    centroid: [mu, mv],
    rms: Math.sqrt(varU + varV),
    angle: [Math.atan(ma), Math.atan(mb)],
    spread: Math.sqrt(spread),
    focus: spread > 1e-18 ? -cov / spread : null,
    sums: s,
  };
}

/** At most `most` groups of neighbouring bins carrying about equal flux, for
 *  a continuum whose hundreds of 5 nm bins hold a handful of rays each. A
 *  spectrum of fewer bins than that, or bins without sums, is left as it is. */
export function groupLines(lines: SpectralLine[], most = 24): SpectralLine[] {
  if (lines.length <= most || lines.some((l) => !l.sums)) return lines;
  const total = lines.reduce((t, l) => t + l.flux, 0);
  const out: SpectralLine[] = [];
  let group: SpectralLine[] = [];
  let acc = 0;
  for (const l of lines) {
    group.push(l);
    acc += l.flux;
    if (acc >= ((out.length + 1) * total) / most) {
      out.push(mergeLines(group));
      group = [];
    }
  }
  if (group.length) out.push(mergeLines(group));
  return out;
}

/** A flux, or any reading, for display: four significant figures, and an
 *  exponent outside the range where that reads well. One formatter, so the
 *  dock and the inset print the same digits for the same number. */
export function formatFlux(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e-3 && a < 1e4 ? v.toPrecision(4) : v.toExponential(3);
}

/** How many rays the lit pixels average. A pixel's reading rests on the rays
 *  that landed in it, so below a handful it is sampling as much as light. */
export function raysPerLitPixel(r: DetectorReadout): number | null {
  let lit = 0;
  for (const v of r.incident) if (v > 0) lit++;
  return lit > 0 ? r.hits / lit : null;
}

export type Box = { x: number; y: number; w: number; h: number };
export type FaceBox = Box & { scale: number };

/** Where the face lands in `area`: its full width and height at one scale
 *  (pixels per mm), centred, `margin` px clear of the edges. `fixedScale`
 *  keeps a scale instead, so a face being dragged larger does not shrink back
 *  under the pointer. */
export function faceBox(
  face: { width: number; height: number },
  area: Box,
  margin = 8,
  fixedScale?: number,
): FaceBox {
  const scale =
    fixedScale ??
    Math.max(1e-9, Math.min((area.w - 2 * margin) / face.width, (area.h - 2 * margin) / face.height));
  const w = face.width * scale;
  const h = face.height * scale;
  return { x: area.x + (area.w - w) / 2, y: area.y + (area.h - h) / 2, w, h, scale };
}

/** The face's outline as the current path: the rectangle, the disc, or the
 *  ring with its hole (fill and clip it "evenodd"). */
export function facePath(c: CanvasRenderingContext2D, face: DetectorFace, b: FaceBox) {
  c.beginPath();
  if (face.shape === "rectangle") {
    c.rect(b.x, b.y, b.w, b.h);
    return;
  }
  const cx = b.x + b.w / 2;
  const cy = b.y + b.h / 2;
  c.arc(cx, cy, b.w / 2, 0, Math.PI * 2);
  if (face.shape === "annulus" && face.inner > 0) {
    const ri = (face.inner / 2) * b.scale;
    c.moveTo(cx + ri, cy);
    c.arc(cx, cy, ri, 0, Math.PI * 2, true);
  }
}

let scratch: HTMLCanvasElement | null = null;

/** Paint `values`, one per pixel (row-major, u fastest, v upward), into the
 *  face in `area`, scaled so `scaleTo` is the top of the ramp; a value above
 *  it saturates. Unlit pixels and everything off the face stay dark. The
 *  pixel grid is drawn when a pixel is big enough on screen to see. Returns
 *  where the face landed; `fixedScale` is passed to [`faceBox`]. `colours`,
 *  three bytes a pixel, paints the pixels in those colours instead of the
 *  ramp (a true-colour picture). */
export function drawFacePixels(
  c: CanvasRenderingContext2D,
  face: DetectorFace,
  values: ArrayLike<number>,
  area: Box,
  scaleTo?: number | null,
  fixedScale?: number,
  colours?: ArrayLike<number>,
): FaceBox {
  const b = faceBox(face, area, 8, fixedScale);
  const [nu, nv] = face.pixels;

  c.save();
  c.fillStyle = "#050709";
  facePath(c, face, b);
  c.fill("evenodd");

  let top = scaleTo ?? 0;
  if (!(top > 0)) for (let i = 0; i < values.length; i++) top = Math.max(top, values[i]);
  if (values.length === nu * nv && top > 0) {
    scratch ??= document.createElement("canvas");
    scratch.width = nu;
    scratch.height = nv;
    const sc = scratch.getContext("2d")!;
    const img = sc.createImageData(nu, nv);
    for (let v = 0; v < nv; v++) {
      // Detector v increases upward; image rows grow downward.
      const row = nv - 1 - v;
      for (let u = 0; u < nu; u++) {
        const val = values[v * nu + u];
        if (!(val > 0)) continue;
        const i = v * nu + u;
        const [r, g, bl] = colours ? [colours[3 * i], colours[3 * i + 1], colours[3 * i + 2]] : heatRgb(val / top);
        const k = (row * nu + u) * 4;
        img.data[k] = r;
        img.data[k + 1] = g;
        img.data[k + 2] = bl;
        img.data[k + 3] = 255;
      }
    }
    sc.putImageData(img, 0, 0);
    facePath(c, face, b);
    c.clip("evenodd");
    c.imageSmoothingEnabled = false;
    c.drawImage(scratch, b.x, b.y, b.w, b.h);
  }
  c.restore();

  if (b.w / nu >= 7 && b.h / nv >= 7) {
    c.save();
    facePath(c, face, b);
    c.clip("evenodd");
    c.strokeStyle = "rgba(127,214,255,0.12)";
    c.lineWidth = 1;
    c.beginPath();
    for (let u = 1; u < nu; u++) {
      const x = Math.round(b.x + (u * b.w) / nu) + 0.5;
      c.moveTo(x, b.y);
      c.lineTo(x, b.y + b.h);
    }
    for (let v = 1; v < nv; v++) {
      const y = Math.round(b.y + (v * b.h) / nv) + 0.5;
      c.moveTo(b.x, y);
      c.lineTo(b.x + b.w, y);
    }
    c.stroke();
    c.restore();
  }

  c.strokeStyle = "rgba(127,214,255,0.45)";
  c.lineWidth = 1;
  facePath(c, face, b);
  c.stroke();
  return b;
}
