// The 3D view's camera, as plain state and math (no three.js here).
//
// The orbit camera the Canvas 2D wireframe view used, kept with the same user
// semantics: yaw/pitch about a target, a zoom expressed as pixels per mm at the
// target, orthographic by default with a mild perspective toggle, and presets.
// The renderer copies this state into its three.js cameras every frame
// (`renderer.ts`), and the overlay projects through `project()` here. Both use
// the same formulas, so a card leader line lands where the mesh is drawn.
//
// Axes: world coordinates are the Rust scene's, 1:1 and in millimetres. +X is
// the optical axis, +Y is up, +Z is out of the meridional plane. The camera's up
// vector is +Y. The UI labels these the Zemax way (optical axis "Z", out of
// plane "X"); that relabelling happens only in the overlay text (`overlays.ts`).
//
// One deliberate change from the wireframe view: pan moves the orbit target in
// the image plane instead of offsetting the finished picture, so orbiting after
// a pan turns about what is on screen. For the orthographic camera the two are
// the same picture.

export type V2 = { x: number; y: number };
export type V3 = { x: number; y: number; z: number };

export const v3 = (x: number, y: number, z: number): V3 => ({ x, y, z });
export const add = (a: V3, b: V3): V3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const sub = (a: V3, b: V3): V3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const scl = (a: V3, s: number): V3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const dot = (a: V3, b: V3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: V3, b: V3): V3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const norm = (a: V3): V3 => {
  const l = Math.hypot(a.x, a.y, a.z) || 1;
  return { x: a.x / l, y: a.y / l, z: a.z / l };
};

export type Basis = { right: V3; up: V3; dir: V3 };

/** An axis-aligned box in world mm. */
export type Bounds3 = { min: V3; max: V3 };

export type CameraPreset = "iso" | "front" | "top" | "side";

export const DEFAULT_YAW = 0.6;
export const DEFAULT_PITCH = 0.42;

// Pitch stays shy of the poles so the right vector never degenerates.
const PITCH_LIMIT = 1.45;

const PRESETS: Record<CameraPreset, [number, number]> = {
  iso: [DEFAULT_YAW, DEFAULT_PITCH],
  front: [0, 0.0001], // the meridional Y-Z plane, face-on
  top: [0, PITCH_LIMIT], // looking down the vertical
  side: [Math.PI / 2, 0.0001], // end-on down the optical axis
};

/** A snapshot of the camera, for tests and the debug hooks. */
export type CameraState = {
  yaw: number;
  pitch: number;
  scale: number;
  target: V3;
  perspective: boolean;
  distance: number;
};

export class CameraRig {
  yaw = DEFAULT_YAW;
  pitch = DEFAULT_PITCH;
  /** Pixels per mm at the target. Zoom changes this, in both projections. */
  scale = 5;
  target: V3 = v3(0, 0, 0);
  perspective = false;
  /** Eye distance from the target in mm. The perspective view converges here,
   *  and the orthographic camera sits here so its depth range is well placed. */
  distance = 500;
  /** Radius of the framed assembly, which sets the depth range. */
  sceneRadius = 50;
  /** CSS pixel size of the view. */
  width = 1;
  height = 1;
  /** Bumped on every change, so the renderer knows to redraw. */
  revision = 0;

  private touch() {
    this.revision++;
  }

  setSize(width: number, height: number) {
    if (width === this.width && height === this.height) return;
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.touch();
  }

  basis(): Basis {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const dir = v3(cp * sy, sp, cp * cy); // target -> eye
    const f = scl(dir, -1); // eye -> target
    const right = norm(cross(f, v3(0, 1, 0)));
    const up = cross(right, f); // unit, since f and right are unit and orthogonal
    return { right, up, dir };
  }

  eye(B = this.basis()): V3 {
    return add(this.target, scl(B.dir, this.distance));
  }

  /** Vertical field of view (radians) that gives `scale` px/mm at the target. */
  fovY(): number {
    const half = this.height / 2 / (this.scale * this.distance);
    return 2 * Math.atan(half);
  }

  /** Near and far clip distances from the eye, from the framed assembly size.
   *  Rays can run far past the hardware, so the far plane reaches a fixed span
   *  beyond it rather than hugging the assembly. */
  clipRange(): { near: number; far: number } {
    const reach = Math.max(4 * this.sceneRadius, 5000);
    if (this.perspective) {
      return { near: Math.max(0.5, this.distance * 0.02), far: this.distance + reach };
    }
    return { near: Math.max(0.1, this.distance - reach), far: this.distance + reach };
  }

  /** World point to CSS pixels, exactly as the three.js camera projects it. */
  project(p: V3, B = this.basis()): V2 {
    const rel = sub(p, this.target);
    const xc = dot(rel, B.right), yc = dot(rel, B.up);
    const cx = this.width / 2, cy = this.height / 2;
    if (!this.perspective) return { x: cx + this.scale * xc, y: cy - this.scale * yc };
    const depth = Math.max(1e-3, this.distance - dot(rel, B.dir));
    const f = this.scale * this.distance;
    return { x: cx + (f * xc) / depth, y: cy - (f * yc) / depth };
  }

  /** Screen pixels per mm at a world point (perspective foreshortening). */
  pxPerMm(p: V3, B = this.basis()): number {
    if (!this.perspective) return this.scale;
    const depth = Math.max(1e-3, this.distance - dot(sub(p, this.target), B.dir));
    return (this.scale * this.distance) / depth;
  }

  orbit(dxPx: number, dyPx: number) {
    this.yaw += dxPx * 0.01;
    this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch + dyPx * 0.01));
    this.touch();
  }

  /** Move the target so the picture follows the pointer by (dx, dy) px. */
  pan(dxPx: number, dyPx: number) {
    const B = this.basis();
    const ppm = this.pxPerMm(this.target, B);
    this.target = add(this.target, add(scl(B.right, -dxPx / ppm), scl(B.up, dyPx / ppm)));
    this.touch();
  }

  zoom(factor: number) {
    // Same limits as the wireframe view. In perspective the zoom is a field of
    // view change, capped so the frustum stays sane.
    let s = Math.max(0.3, Math.min(200, this.scale * factor));
    if (this.perspective) {
      const minScale = this.height / 2 / (this.distance * Math.tan((150 * Math.PI) / 360));
      s = Math.max(s, minScale);
    }
    this.scale = s;
    this.touch();
  }

  /** A screen-space drag of (dx, dy) px, as a world displacement in the image
   *  plane at depth `at`. This is how a dragged body follows the pointer. */
  screenDeltaToWorld(dxPx: number, dyPx: number, at: V3): V3 {
    const B = this.basis();
    const ppm = this.pxPerMm(at, B);
    return add(scl(B.right, dxPx / ppm), scl(B.up, -dyPx / ppm));
  }

  /** The image-plane unit vectors (for arrow-key nudges). */
  imagePlaneAxes(): { right: V3; up: V3 } {
    const B = this.basis();
    return { right: B.right, up: B.up };
  }

  setPerspective(on: boolean) {
    this.perspective = on;
    this.touch();
  }

  setPreset(name: CameraPreset) {
    const p = PRESETS[name];
    if (!p) return;
    [this.yaw, this.pitch] = p;
    this.touch();
  }

  resetOrientation() {
    this.yaw = DEFAULT_YAW;
    this.pitch = DEFAULT_PITCH;
    this.touch();
  }

  /** Horizontal pixels on each side covered by the app's side panels, which
   *  framing keeps the assembly clear of. */
  sideInset = 0;

  /** Fit `bounds` into about 84% of the free view, looking at its centre. */
  frame(bounds: Bounds3 | null) {
    if (!bounds || !Number.isFinite(bounds.min.x)) {
      this.target = v3(0, 0, 0);
      this.scale = 5;
      this.sceneRadius = 50;
      this.distance = 500;
      this.touch();
      return;
    }
    const { min: lo, max: hi } = bounds;
    this.target = scl(add(lo, hi), 0.5);
    const B = this.basis();
    let mx = 1, my = 1;
    for (const x of [lo.x, hi.x]) {
      for (const y of [lo.y, hi.y]) {
        for (const z of [lo.z, hi.z]) {
          const rel = sub(v3(x, y, z), this.target);
          mx = Math.max(mx, Math.abs(dot(rel, B.right)));
          my = Math.max(my, Math.abs(dot(rel, B.up)));
        }
      }
    }
    const freeWidth = Math.max(this.width * 0.4, this.width - 2 * this.sideInset);
    this.scale = Math.max(0.5, Math.min((freeWidth * 0.42) / mx, (this.height * 0.42) / my));
    this.sceneRadius = Math.max(10, 0.5 * Math.hypot(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z));
    // Far enough that the whole assembly sits in front of the perspective eye.
    this.distance = Math.max(500, 4 * this.sceneRadius);
    this.touch();
  }

  state(): CameraState {
    return {
      yaw: this.yaw,
      pitch: this.pitch,
      scale: this.scale,
      target: { ...this.target },
      perspective: this.perspective,
      distance: this.distance,
    };
  }
}
