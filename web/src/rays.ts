// The ray layer: HDR accumulation, Reinhard tonemap, present.
//
// Extracted verbatim from main.ts so the openUC2 mode draws rays through the
// same code rather than growing a second, subtly different renderer (§15.1:
// "Do not fork the renderer"). Behaviour is unchanged; only the state moved
// from a closure into a class, one instance per canvas.
//
// Why it works this way: rays are additive in *linear* light, so a pixel's
// colour is the sum of its wavelengths — a full spectrum adds up to white, and
// where wavelengths physically separate you get coloured fringes. Reinhard
// tonemapping then maps the huge dynamic range (a focus is enormously brighter
// than a stray ray) into [0,1] without the blow-out that plagues un-tonemapped
// additive.

export type V2 = { x: number; y: number };

/// Per-frame knobs.
export type RayDrawOpts = {
  /// Ghosts are a dim (~4% energy) layer with their own opacity control.
  ghostOpacity: number;
  exposure: number;
  dpr: number;
  /// CSS-space viewport size.
  width: number;
  height: number;
};

/// How a frame's segments are laid out: `stride` floats each, the first nine
/// `[ax, ay, bx, by, r, g, b, flux, flags]`. With `clip`, floats 9 and 10 are
/// the ends' signed distances from the view plane, and `clip` gives the part
/// of the segment to draw, as fractions of its length, or null for none. The
/// clipped ends are rounded to f32, as a clipped copy of the frame would hold
/// them, so reading a picture in place draws what the copy drew.
export type RayLayout = {
  stride: number;
  clip: ((dA: number, dB: number) => [number, number] | null) | null;
};

/// Nine floats a segment, nothing to clip.
export const PLAIN_RAYS: RayLayout = { stride: 9, clip: null };

export type RayDrawResult = {
  /// Segments actually rasterized (i.e. above the visibility floor).
  count: number;
  /// CSS-space points where a surface went into total internal reflection.
  tirDots: V2[];
};

/// One accumulation buffer and its presented canvas.
export class RayLayer {
  readonly canvas = document.createElement("canvas");
  private ctx = this.canvas.getContext("2d")!;
  /// The frame's TIR markers, a ring at each spot, at device resolution, for
  /// compositing over [`canvas`]. Drawn as the frame is rasterized, each ring
  /// once, so a repaint copies them instead of stroking every ring again: a
  /// ghost-lit lens has tens of thousands.
  readonly rings = document.createElement("canvas");
  private ringCtx = this.rings.getContext("2d")!;
  private ringsDrawn = 0;
  // The frame's rings are drawn when its first present lands, over a
  // cleared canvas, so the last frame's rings stay while this one is drawn.
  private ringsFresh = false;
  private ringsUpTo = 0;
  private dpr = 1;
  private hdr = new Float32Array(0);
  private img: ImageData | null = null;
  // The image as 32-bit pixels, one write per pixel.
  private px32 = new Uint32Array(0);
  // A present in progress: the region being tonemapped and the next row.
  private presenting: { x0: number; y0: number; x1: number; y1: number; y: number } | null = null;
  private bufW = 0;
  private bufH = 0;
  // Dirty-rectangle tracking (device px): the lit region this frame (c*) and
  // last frame (lb*). Clearing, tonemapping and uploading only this rectangle —
  // instead of the whole screen every frame — is what keeps it responsive on a
  // hi-DPI panel where the full buffer is millions of pixels.
  private cx0 = 0;
  private cy0 = 0;
  private cx1 = -1;
  private cy1 = -1;
  private lbx0 = 0;
  private lby0 = 0;
  private lbx1 = -1;
  private lby1 = -1;
  private static TONE = RayLayer.buildTone();

  // Reinhard tonemap x/(1+x) ∈ [0,1) followed by linear→sRGB, baked into a LUT
  // so the per-pixel present loop stays cheap.
  private static buildTone(): Uint8ClampedArray {
    const l2s = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
    const t = new Uint8ClampedArray(1025);
    for (let i = 0; i <= 1024; i++) t[i] = Math.round(l2s(i / 1024) * 255);
    return t;
  }

  private ensureBuffers(W: number, H: number) {
    if (W === this.bufW && H === this.bufH) return;
    this.bufW = W;
    this.bufH = H;
    this.canvas.width = W;
    this.canvas.height = H;
    this.rings.width = W;
    this.rings.height = H;
    this.hdr = new Float32Array(W * H * 3);
    this.img = this.ctx.createImageData(W, H);
    this.px32 = new Uint32Array(this.img.data.buffer);
    this.lbx1 = -1;
    this.lby1 = -1; // fresh buffers → no stale lit region
  }

  private plot(x: number, y: number, r: number, g: number, b: number, c: number) {
    if (c <= 0 || x < 0 || x >= this.bufW || y < 0 || y >= this.bufH) return;
    const k = (y * this.bufW + x) * 3;
    this.hdr[k] += r * c;
    this.hdr[k + 1] += g * c;
    this.hdr[k + 2] += b * c;
    if (x < this.cx0) this.cx0 = x;
    if (x > this.cx1) this.cx1 = x;
    if (y < this.cy0) this.cy0 = y;
    if (y > this.cy1) this.cy1 = y;
  }

  // Antialiased additive line (Xiaolin Wu). Coordinates are in device pixels;
  // `r,g,b` are the flux-weighted linear contribution. Each column deposits
  // total coverage 1 split across the two straddling pixels, so brightness per
  // unit length is constant and edges are smooth.
  private addLine(x0: number, y0: number, x1: number, y1: number, r: number, g: number, b: number) {
    // Clip to the buffer first (Liang–Barsky). Exit rays run to the scene
    // bounds — tens of thousands of pixels off-screen — so rasterizing the
    // unclipped span would burn the whole frame on no-op iterations.
    const dx0 = x1 - x0, dy0 = y1 - y0;
    let t0 = 0, t1 = 1;
    const edges = [[-dx0, x0], [dx0, this.bufW - 1 - x0], [-dy0, y0], [dy0, this.bufH - 1 - y0]];
    for (const [p, q] of edges) {
      if (p === 0) {
        if (q < 0) return; // parallel and outside
      } else {
        const t = q / p;
        if (p < 0) { if (t > t1) return; if (t > t0) t0 = t; }
        else { if (t < t0) return; if (t < t1) t1 = t; }
      }
    }
    x1 = x0 + t1 * dx0; y1 = y0 + t1 * dy0;
    x0 = x0 + t0 * dx0; y0 = y0 + t0 * dy0;

    let steep = Math.abs(y1 - y0) > Math.abs(x1 - x0);
    if (steep) { [x0, y0] = [y0, x0]; [x1, y1] = [y1, x1]; }
    if (x0 > x1) { [x0, x1] = [x1, x0]; [y0, y1] = [y1, y0]; }
    const dx = x1 - x0;
    const grad = dx === 0 ? 1 : (y1 - y0) / dx;
    const ix0 = Math.round(x0), ix1 = Math.round(x1);
    let intery = y0 + grad * (ix0 - x0 + 0.5);
    for (let x = ix0; x <= ix1; x++) {
      const y = Math.floor(intery);
      const f = intery - y;
      if (steep) {
        this.plot(y, x, r, g, b, 1 - f);
        this.plot(y + 1, x, r, g, b, f);
      } else {
        this.plot(x, y, r, g, b, 1 - f);
        this.plot(x, y + 1, r, g, b, f);
      }
      intery += grad;
    }
  }

  /// Rasterize `segs` (9 floats each: ax ay bx by r g b flux flags) into
  /// [`canvas`], ready to be composited with `globalCompositeOperation =
  /// "lighter"`.
  draw(
    segs: Float32Array | number[],
    toScreen: (p: V2) => V2,
    o: RayDrawOpts,
    layout: RayLayout = PLAIN_RAYS,
  ): RayDrawResult {
    this.begin(segs, o, layout);
    this.add(segs, 0, segs.length / layout.stride, toScreen, o, layout);
    return this.present();
  }

  // What the frame being rasterized has so far: the brightest segment's flux
  // (the scale every segment's share is taken against), the segments drawn,
  // and the TIR points.
  private maxFlux = 1e-9;
  private count = 0;
  private tirDots: V2[] = [];
  // Where the frame already has a TIR marker, to a quarter of a CSS pixel. A
  // ghost-lit lens puts hundreds of thousands of TIR segment ends on a few
  // hundred spots; one marker per spot draws the same rings.
  private tirSeen = new Set<string>();

  /// Start a frame of `segs`: clear the last frame's lit region and take the
  /// brightness scale from all of them. The frame is then rasterized a range
  /// at a time ([`add`]) and shown with [`present`], so a picture of hundreds
  /// of thousands of segments can be drawn over several frames rather than
  /// in one long task. [`draw`] is the three in one.
  begin(segs: Float32Array | number[], o: RayDrawOpts, layout: RayLayout = PLAIN_RAYS) {
    const DW = Math.round(o.width * o.dpr), DH = Math.round(o.height * o.dpr);
    this.ensureBuffers(DW, DH);

    // Zero only what the last frames lit (not the whole buffer): the region
    // last presented, and what the frame before this one added since, which
    // a frame abandoned between presents never showed. Then open a fresh
    // current box that plot() grows.
    const zx0 = Math.min(this.lbx0, this.cx0), zy0 = Math.min(this.lby0, this.cy0);
    const zx1 = Math.max(this.lbx1, this.cx1), zy1 = Math.max(this.lby1, this.cy1);
    for (let y = Math.max(0, zy0); y <= Math.min(zy1, this.bufH - 1); y++) {
      let k = (y * this.bufW + Math.max(0, zx0)) * 3;
      for (let x = Math.max(0, zx0); x <= Math.min(zx1, this.bufW - 1); x++, k += 3) {
        this.hdr[k] = 0; this.hdr[k + 1] = 0; this.hdr[k + 2] = 0;
      }
    }
    this.cx0 = this.bufW; this.cy0 = this.bufH; this.cx1 = -1; this.cy1 = -1;

    // Flux fraction relative to a fresh launch ray (all launch rays start equal),
    // so brightness is energy, not ray count.
    // Of the segments the frame draws: a clipped-away segment sets no scale.
    let maxFlux = 1e-9;
    const { stride, clip } = layout;
    for (let i = 0; i + stride <= segs.length; i += stride) {
      if (segs[i + 7] > maxFlux && (!clip || clip(segs[i + 9], segs[i + 10]))) maxFlux = segs[i + 7];
    }
    this.maxFlux = maxFlux;
    this.count = 0;
    this.tirDots = [];
    this.tirSeen.clear();
    this.dpr = o.dpr;
    this.ringsFresh = true;
    this.ringsDrawn = 0;
    this.presenting = null;
  }

  /// Rasterize segments `from` (inclusive) to `to` (exclusive) of the frame.
  add(
    segs: Float32Array | number[],
    from: number,
    to: number,
    toScreen: (p: V2) => V2,
    o: RayDrawOpts,
    layout: RayLayout = PLAIN_RAYS,
  ) {
    const maxFlux = this.maxFlux;
    const tirDots = this.tirDots;
    const tirSeen = this.tirSeen;
    const { stride, clip } = layout;
    const f32 = Math.fround;
    let count = 0;
    const end = Math.min(to, Math.floor(segs.length / stride)) * stride;
    for (let i = from * stride; i < end; i += stride) {
      let ax = segs[i], ay = segs[i + 1], bx = segs[i + 2], by = segs[i + 3];
      if (clip) {
        const range = clip(segs[i + 9], segs[i + 10]);
        if (!range) continue;
        const [t0, t1] = range;
        [ax, ay, bx, by] = [
          f32(ax + (bx - ax) * t0), f32(ay + (by - ay) * t0),
          f32(ax + (bx - ax) * t1), f32(ay + (by - ay) * t1),
        ];
      }
      const flags = Math.round(segs[i + 8]);
      const ghost = (flags & 4) !== 0;
      const frac = Math.min(1, segs[i + 7] / maxFlux);
      const wgt = (ghost ? frac * o.ghostOpacity * 8 : frac) * o.exposure;
      if (wgt < 1e-4) continue;
      const a = toScreen({ x: ax, y: ay });
      const b = toScreen({ x: bx, y: by });
      // rasterize in device px; keep CSS-space endpoint for overlay markers
      this.addLine(a.x * o.dpr, a.y * o.dpr, b.x * o.dpr, b.y * o.dpr,
        segs[i + 4] * wgt, segs[i + 5] * wgt, segs[i + 6] * wgt);
      if ((flags & 2) !== 0) {
        const spot = `${Math.round(b.x * 4)},${Math.round(b.y * 4)}`;
        if (!tirSeen.has(spot)) {
          tirSeen.add(spot);
          tirDots.push(b);
        }
      }
      count++;
    }
    this.count += count;
  }

  /// Tonemap what the frame holds so far into [`canvas`], all at once.
  present(): RayDrawResult {
    const shown = this.presentBegin();
    this.presentRows(Infinity);
    return shown;
  }

  /// Begin tonemapping what the frame holds so far. [`presentRows`] does it
  /// a band of rows at a time and puts it on [`canvas`] with the last band,
  /// so a picture that lights millions of pixels (a ghost-lit lens on a
  /// high-DPI screen) is shown without a long task. Returns what it shows.
  presentBegin(): RayDrawResult {
    const count = this.count;
    const tirDots = this.tirDots.slice();
    // Tonemap over the union of this frame's and last frame's lit boxes: lit
    // pixels get written, pixels that went dark get cleared to transparent.
    let ux0 = this.bufW, uy0 = this.bufH, ux1 = -1, uy1 = -1;
    if (this.cx1 >= this.cx0) {
      ux0 = Math.min(ux0, this.cx0); uy0 = Math.min(uy0, this.cy0);
      ux1 = Math.max(ux1, this.cx1); uy1 = Math.max(uy1, this.cy1);
    }
    if (this.lbx1 >= this.lbx0) {
      ux0 = Math.min(ux0, this.lbx0); uy0 = Math.min(uy0, this.lby0);
      ux1 = Math.max(ux1, this.lbx1); uy1 = Math.max(uy1, this.lby1);
    }
    this.presenting = { x0: ux0, y0: uy0, x1: ux1, y1: uy1, y: uy0 };
    // Remember this frame's lit box so next frame can clear exactly it.
    this.lbx0 = this.cx0; this.lby0 = this.cy0; this.lbx1 = this.cx1; this.lby1 = this.cy1;
    this.ringsUpTo = this.tirDots.length;
    return { count, tirDots };
  }

  /// Tonemap rows of the present [`presentBegin`] began until `ms` have
  /// passed. True once [`canvas`] shows it.
  presentRows(ms: number): boolean {
    const p = this.presenting;
    if (!p) return true;
    const t0 = performance.now();
    const hdr = this.hdr, tone = RayLayer.TONE, px = this.px32, w = this.bufW;
    while (p.y <= p.y1) {
      let k = (p.y * w + p.x0) * 3;
      let q = p.y * w + p.x0;
      for (let x = p.x0; x <= p.x1; x++, k += 3, q++) {
        const r = hdr[k], g = hdr[k + 1], b = hdr[k + 2];
        if (r === 0 && g === 0 && b === 0) {
          px[q] = 0;
          continue;
        }
        px[q] = RayLayer.rgba(
          tone[(r / (1 + r) * 1024) | 0],
          tone[(g / (1 + g) * 1024) | 0],
          tone[(b / (1 + b) * 1024) | 0],
        );
      }
      p.y++;
      if ((p.y & 15) === 0 && performance.now() - t0 >= ms) break;
    }
    if (p.y <= p.y1) return false;
    if (p.x1 >= p.x0) this.ctx.putImageData(this.img!, 0, 0, p.x0, p.y0, p.x1 - p.x0 + 1, p.y1 - p.y0 + 1);
    this.presenting = null;
    this.drawRings();
    return true;
  }

  // One opaque pixel as the image's 32-bit word, in the machine's byte order.
  private static LITTLE = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  private static rgba(r: number, g: number, b: number): number {
    return RayLayer.LITTLE ? (255 << 24) | (b << 16) | (g << 8) | r : (r << 24) | (g << 16) | (b << 8) | 255;
  }

  // The rings of the TIR spots shown by the present that just landed.
  private drawRings() {
    const c = this.ringCtx, d = this.dpr;
    if (this.ringsFresh) {
      c.clearRect(0, 0, this.bufW, this.bufH);
      this.ringsFresh = false;
    }
    if (this.ringsDrawn >= this.ringsUpTo) return;
    c.lineWidth = 1.5 * d;
    c.strokeStyle = "rgba(255,120,80,0.9)";
    c.beginPath();
    for (let i = this.ringsDrawn; i < this.ringsUpTo; i++) {
      const q = this.tirDots[i];
      c.moveTo((q.x + 4) * d, q.y * d);
      c.arc(q.x * d, q.y * d, 4 * d, 0, Math.PI * 2);
    }
    c.stroke();
    this.ringsDrawn = this.ringsUpTo;
  }
}
