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
  private hdr = new Float32Array(0);
  private img: ImageData | null = null;
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
    this.hdr = new Float32Array(W * H * 3);
    this.img = this.ctx.createImageData(W, H);
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
  draw(segs: Float32Array | number[], toScreen: (p: V2) => V2, o: RayDrawOpts): RayDrawResult {
    const DW = Math.round(o.width * o.dpr), DH = Math.round(o.height * o.dpr);
    this.ensureBuffers(DW, DH);

    // Zero only last frame's lit region (not the whole buffer), then open a
    // fresh current box that plot() grows.
    for (let y = this.lby0; y <= this.lby1; y++) {
      let k = (y * this.bufW + this.lbx0) * 3;
      for (let x = this.lbx0; x <= this.lbx1; x++, k += 3) {
        this.hdr[k] = 0; this.hdr[k + 1] = 0; this.hdr[k + 2] = 0;
      }
    }
    this.cx0 = this.bufW; this.cy0 = this.bufH; this.cx1 = -1; this.cy1 = -1;

    // Flux fraction relative to a fresh launch ray (all launch rays start equal),
    // so brightness is energy, not ray count.
    let maxFlux = 1e-9;
    for (let i = 7; i < segs.length; i += 9) if (segs[i] > maxFlux) maxFlux = segs[i];

    let count = 0;
    const tirDots: V2[] = [];
    for (let i = 0; i + 8 < segs.length; i += 9) {
      const flags = Math.round(segs[i + 8]);
      const ghost = (flags & 4) !== 0;
      const frac = Math.min(1, segs[i + 7] / maxFlux);
      const wgt = (ghost ? frac * o.ghostOpacity * 8 : frac) * o.exposure;
      if (wgt < 1e-4) continue;
      const a = toScreen({ x: segs[i], y: segs[i + 1] });
      const b = toScreen({ x: segs[i + 2], y: segs[i + 3] });
      // rasterize in device px; keep CSS-space endpoint for overlay markers
      this.addLine(a.x * o.dpr, a.y * o.dpr, b.x * o.dpr, b.y * o.dpr,
        segs[i + 4] * wgt, segs[i + 5] * wgt, segs[i + 6] * wgt);
      if ((flags & 2) !== 0) tirDots.push(b);
      count++;
    }

    // Tonemap over the union of this frame's and last frame's lit boxes: lit
    // pixels get written, pixels that went dark get cleared to transparent.
    const px = this.img!.data;
    let ux0 = this.bufW, uy0 = this.bufH, ux1 = -1, uy1 = -1;
    if (this.cx1 >= this.cx0) {
      ux0 = Math.min(ux0, this.cx0); uy0 = Math.min(uy0, this.cy0);
      ux1 = Math.max(ux1, this.cx1); uy1 = Math.max(uy1, this.cy1);
    }
    if (this.lbx1 >= this.lbx0) {
      ux0 = Math.min(ux0, this.lbx0); uy0 = Math.min(uy0, this.lby0);
      ux1 = Math.max(ux1, this.lbx1); uy1 = Math.max(uy1, this.lby1);
    }
    if (ux1 >= ux0) {
      for (let y = uy0; y <= uy1; y++) {
        let k = (y * this.bufW + ux0) * 3;
        let p = (y * this.bufW + ux0) * 4;
        for (let x = ux0; x <= ux1; x++, k += 3, p += 4) {
          const r = this.hdr[k], g = this.hdr[k + 1], b = this.hdr[k + 2];
          if (r === 0 && g === 0 && b === 0) { px[p + 3] = 0; continue; }
          px[p] = RayLayer.TONE[(r / (1 + r) * 1024) | 0];
          px[p + 1] = RayLayer.TONE[(g / (1 + g) * 1024) | 0];
          px[p + 2] = RayLayer.TONE[(b / (1 + b) * 1024) | 0];
          px[p + 3] = 255;
        }
      }
      this.ctx.putImageData(this.img!, 0, 0, ux0, uy0, ux1 - ux0 + 1, uy1 - uy0 + 1);
    }
    // Remember this frame's lit box so next frame can clear exactly it.
    this.lbx0 = this.cx0; this.lby0 = this.cy0; this.lbx1 = this.cx1; this.lby1 = this.cy1;

    return { count, tirDots };
  }
}
