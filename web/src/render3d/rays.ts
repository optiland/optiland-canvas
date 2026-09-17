// The ray bundle on the GPU: every traced segment in one draw call.
//
// The segments are the tracer's, unchanged: the f32 preview during a drag, the
// exact f64 trace otherwise (view3d.ts decides). Nothing is retraced or
// resampled here. One interleaved buffer holds, per vertex, the world position,
// the wavelength colour (linear light, as the tracer reports it), the segment's
// flux relative to the brightest launch ray, and a ghost flag.
//
// Display mapping (a visualization choice, not physics):
//   dark backgrounds   additive light, like the 2D view: overlapping
//                      wavelengths sum toward white, a dispersed bundle shows
//                      its colours, and dim ghosts stay dim. The frame is
//                      tone-mapped once at output.
//   light backgrounds  additive light cannot show on a light ground. The rays
//                      are summed the same additive way in a separate pass, and
//                      the sum is laid over the scene as ink: its opacity
//                      follows the summed brightness, its colour the summed hue.
//                      A white bundle stays neutral and a dispersed one keeps
//                      its colours, independent of drawing order.
// The ray gain and the ghost gain are display parameters. Neither changes a
// traced flux, and neither is a number the app reports.
//
// Width: at 1 px the segments are GPU lines. Wider, each segment is a
// screen-facing strip (one instanced quad per segment, still one draw) with a
// one-pixel soft edge. Width is in CSS pixels, so it holds while the view drops
// its pixel ratio during a drag. A wider line covers more pixels, so its
// brightness per pixel is lowered a little to keep dense bundles readable.

import * as THREE from "three/webgpu";
import {
  Fn,
  If,
  abs,
  attribute,
  cameraProjectionMatrix,
  float,
  max,
  min,
  mix,
  modelViewMatrix,
  positionGeometry,
  screenDPR,
  screenUV,
  select,
  texture,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  viewport,
} from "three/tsl";

// The projected trace layout (traceProjected3D): 11 floats per segment,
// [au,av,bu,bv, r,g,b, flux, flags, dist_a, dist_b]; for the default XY view
// plane, (au, av, dist_a) is the world start point.
const SRC_STRIDE = 11;
// Per line vertex: x y z, r g b, weight, ghost.
const VERTEX_FLOATS = 8;
// Per strip instance: start x y z, end x y z, r g b, weight, ghost.
const INSTANCE_FLOATS = 11;
/** Segments this dim (relative flux) are not uploaded at all. */
const FLUX_FLOOR = 1e-5;
const FLAG_TIR = 2;
const FLAG_GHOST = 4;
/** The widest line offered, CSS px. */
export const MAX_RAY_WIDTH = 6;

export type RayVisibility = "through" | "occluded";

export type RayUpload = {
  /** Segments uploaded. */
  count: number;
  /** World points where a surface is in total internal reflection. */
  tir: { x: number; y: number; z: number }[];
  ms: number;
};

export class RayBuffer {
  /** The 1 px picture: GPU lines. */
  readonly object: THREE.LineSegments;
  /** The wide picture: one screen-facing quad per segment. */
  readonly strips: THREE.Mesh;
  private buffer: THREE.InterleavedBuffer;
  private capacity = 0;
  private stripBuffer: THREE.InstancedInterleavedBuffer;
  private stripCapacity = 0;
  private readonly ghostGain = uniform(3.6);
  private readonly rayGain = uniform(1.4);
  private readonly halfWidth = uniform(0.5);
  private readonly widthGain = uniform(1);
  /** The camera, for cutting wide segments at the near plane. */
  private readonly perspective = uniform(0);
  private readonly nearZ = uniform(-0.1);
  private readonly additive: THREE.LineBasicNodeMaterial;
  private readonly stripMaterial: THREE.MeshBasicNodeMaterial;
  private visibility: RayVisibility = "through";
  private width = 1;
  private last: Float32Array | null = null;
  /** The light-background pass: summed rays, then an ink overlay. */
  readonly inkTarget = new THREE.RenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4, depthBuffer: true });
  readonly inkOverlay: THREE.Mesh;
  /** Opaque parts, drawn depth-only into the ink pass when rays are occluded. */
  readonly depthOnly = new THREE.MeshBasicNodeMaterial({ colorWrite: false });
  inkMode = false;
  count = 0;
  uploads = 0;

  constructor() {
    this.buffer = this.allocate(1024);
    this.stripBuffer = this.allocateStrips(1024);
    this.additive = this.makeAdditive();
    this.stripMaterial = this.makeStrip();
    this.inkOverlay = this.makeInkOverlay();
    this.object = new THREE.LineSegments(this.geometryFor(this.buffer), this.additive);
    this.object.name = "rays";
    this.strips = new THREE.Mesh(this.stripGeometryFor(this.stripBuffer), this.stripMaterial);
    this.strips.name = "rays-wide";
    for (const o of [this.object, this.strips]) {
      o.frustumCulled = false;
      // After the components, the glass, and the cues.
      o.renderOrder = 40;
      o.raycast = () => {};
    }
    this.strips.visible = false;
    this.setVisibility("through");
  }

  /** The objects that draw rays; add both to the scene. */
  get objects(): THREE.Object3D[] {
    return [this.object, this.strips];
  }

  isRayObject(o: THREE.Object3D): boolean {
    return o === this.object || o === this.strips;
  }

  /** Show the active picture (lines or strips), or neither. */
  setDrawVisible(on: boolean) {
    this.object.visible = on && this.width <= 1;
    this.strips.visible = on && this.width > 1;
  }

  private allocate(segments: number): THREE.InterleavedBuffer {
    this.capacity = segments;
    const b = new THREE.InterleavedBuffer(new Float32Array(segments * 2 * VERTEX_FLOATS), VERTEX_FLOATS);
    b.setUsage(THREE.DynamicDrawUsage);
    return b;
  }

  private allocateStrips(segments: number): THREE.InstancedInterleavedBuffer {
    this.stripCapacity = segments;
    const b = new THREE.InstancedInterleavedBuffer(new Float32Array(segments * INSTANCE_FLOATS), INSTANCE_FLOATS, 1);
    b.setUsage(THREE.DynamicDrawUsage);
    return b;
  }

  private geometryFor(b: THREE.InterleavedBuffer): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.InterleavedBufferAttribute(b, 3, 0));
    g.setAttribute("rayColor", new THREE.InterleavedBufferAttribute(b, 3, 3));
    g.setAttribute("rayWeight", new THREE.InterleavedBufferAttribute(b, 1, 6));
    g.setAttribute("rayGhost", new THREE.InterleavedBufferAttribute(b, 1, 7));
    g.setDrawRange(0, 0);
    return g;
  }

  /** A unit quad (x across the line in [-1, 1], y along it in {0, 1}) drawn
   *  once per segment. */
  private stripGeometryFor(b: THREE.InstancedInterleavedBuffer): THREE.InstancedBufferGeometry {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1, 0, 0, 1, 0, 0, -1, 1, 0, 1, 1, 0]), 3));
    g.setIndex([0, 1, 2, 2, 1, 3]);
    g.setAttribute("rayStart", new THREE.InterleavedBufferAttribute(b, 3, 0));
    g.setAttribute("rayEnd", new THREE.InterleavedBufferAttribute(b, 3, 3));
    g.setAttribute("rayColor", new THREE.InterleavedBufferAttribute(b, 3, 6));
    g.setAttribute("rayWeight", new THREE.InterleavedBufferAttribute(b, 1, 9));
    g.setAttribute("rayGhost", new THREE.InterleavedBufferAttribute(b, 1, 10));
    g.instanceCount = 0;
    return g;
  }

  /** The per-segment display weight: relative flux, times the ghost gain on
   *  ghost segments. */
  private weight() {
    return attribute("rayWeight", "float").mul(mix(float(1), this.ghostGain, attribute("rayGhost", "float")));
  }

  private makeAdditive(): THREE.LineBasicNodeMaterial {
    const m = new THREE.LineBasicNodeMaterial({ transparent: true, depthWrite: false });
    m.blending = THREE.AdditiveBlending;
    m.colorNode = attribute("rayColor", "vec3").mul(this.weight()).mul(this.rayGain);
    m.name = "rays-additive";
    return m;
  }

  /** Wide rays: each quad is placed in clip space around its segment, like
   *  three.js's Line2, and faded over its outer pixel. */
  private makeStrip(): THREE.MeshBasicNodeMaterial {
    const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide });
    m.blending = THREE.AdditiveBlending;
    // A quad faces either way depending on the segment's screen direction.
    // One pass draws both sides; additive light needs no back-to-front split.
    m.forceSinglePass = true;
    // Half the drawn width in device pixels, and one more for the soft edge.
    const core = this.halfWidth.mul(screenDPR);
    const reach = core.add(1);
    m.vertexNode = Fn(() => {
      const start = modelViewMatrix.mul(vec4(attribute("rayStart", "vec3"), 1)).toVar();
      const end = modelViewMatrix.mul(vec4(attribute("rayEnd", "vec3"), 1)).toVar();
      // A perspective segment that reaches behind the camera is cut at the near
      // plane first, so its projected direction stays finite.
      const nearZ = this.nearZ;
      const cut = (a: typeof start, b: typeof start) => nearZ.sub(a.z).div(b.z.sub(a.z));
      If(this.perspective.greaterThan(0.5), () => {
        If(start.z.lessThan(0).and(end.z.greaterThan(0)), () => {
          end.assign(vec4(mix(start.xyz, end.xyz, cut(start, end)), end.w));
        }).ElseIf(end.z.lessThan(0).and(start.z.greaterThanEqual(0)), () => {
          start.assign(vec4(mix(end.xyz, start.xyz, cut(end, start)), start.w));
        });
      });
      const clipA = cameraProjectionMatrix.mul(start);
      const clipB = cameraProjectionMatrix.mul(end);
      // Screen direction in device pixels (up to a common factor).
      const size = viewport.zw;
      const d = clipB.xy.div(clipB.w).sub(clipA.xy.div(clipA.w)).mul(size);
      const len = d.length();
      const dir = select(len.greaterThan(1e-6), d.div(len), vec2(1, 0));
      const across = vec2(dir.y.negate(), dir.x);
      const clip = select(positionGeometry.y.greaterThan(0.5), clipB, clipA);
      // Pixels to NDC is ×2/size; NDC to clip is ×w.
      const offset = across.mul(positionGeometry.x).mul(reach).mul(2).div(size).mul(clip.w);
      return vec4(clip.xy.add(offset), clip.zw);
    })();
    // Distance from the centre line in device pixels, and the covered share of
    // the pixel there.
    const fromCentre = abs(varying(positionGeometry.x)).mul(reach);
    const coverage = core.add(0.5).sub(fromCentre).clamp(0, 1);
    m.colorNode = attribute("rayColor", "vec3").mul(this.weight()).mul(this.rayGain).mul(this.widthGain);
    m.opacityNode = coverage;
    m.name = "rays-wide";
    return m;
  }

  /** A fullscreen quad that lays the summed rays over the scene as ink. */
  private makeInkOverlay(): THREE.Mesh {
    const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false });
    // Clip-space quad: the plane's vertices are already at the screen corners.
    m.vertexNode = vec4(positionGeometry.xy, 0, 1);
    const sum = texture(this.inkTarget.texture, screenUV).rgb;
    const peak = max(max(sum.x, sum.y), sum.z);
    const low = min(min(sum.x, sum.y), sum.z);
    const hue = sum.div(max(peak, float(1e-6)));
    // Saturation 0 (a white bundle) inks dark grey; saturated light inks its
    // own hue, darkened so it reads on a light ground.
    const saturation = peak.sub(low).div(max(peak, float(1e-6)));
    m.colorNode = mix(vec3(0.1, 0.11, 0.12), hue.mul(0.55), saturation);
    // Coverage follows the summed brightness, compressed like the 2D view.
    m.opacityNode = peak.div(peak.add(1)).mul(1.9).min(0.92);
    m.name = "rays-ink";
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), m);
    quad.name = "rays-ink";
    quad.frustumCulled = false;
    quad.renderOrder = 40;
    quad.visible = false;
    quad.raycast = () => {};
    return quad;
  }

  /** Match the ink pass to the drawing buffer. */
  setDrawingSize(width: number, height: number) {
    if (this.inkTarget.width !== width || this.inkTarget.height !== height) this.inkTarget.setSize(width, height);
  }

  /** The projection in use: wide segments reaching behind a perspective
   *  camera are cut at its near plane (`near` > 0, mm). */
  setCamera(perspective: boolean, near: number) {
    this.perspective.value = perspective ? 1 : 0;
    this.nearZ.value = -near;
  }

  /** Line width in CSS pixels, 1 to MAX_RAY_WIDTH. */
  setWidth(px: number) {
    const w = Math.min(MAX_RAY_WIDTH, Math.max(1, Number.isFinite(px) ? px : 1));
    if (w === this.width) return;
    const switched = (w > 1) !== (this.width > 1);
    this.width = w;
    this.halfWidth.value = w / 2;
    this.widthGain.value = 1 / Math.sqrt(w);
    this.setDrawVisible(true);
    // The other picture holds an older upload; refill it from the last trace.
    if (switched && this.last) this.update(this.last);
  }

  getWidth(): number {
    return this.width;
  }

  /** Upload a projected trace into the active picture. */
  update(segs: Float32Array): RayUpload {
    const t0 = performance.now();
    this.last = segs;
    let maxFlux = 1e-9;
    for (let i = 7; i < segs.length; i += SRC_STRIDE) if (segs[i] > maxFlux) maxFlux = segs[i];
    const tir: RayUpload["tir"] = [];
    const count = this.width > 1 ? this.fillStrips(segs, maxFlux, tir) : this.fillLines(segs, maxFlux, tir);
    this.count = count;
    this.uploads++;
    return { count, tir, ms: performance.now() - t0 };
  }

  private fillLines(segs: Float32Array, maxFlux: number, tir: RayUpload["tir"]): number {
    const n = Math.floor(segs.length / SRC_STRIDE);
    if (n > this.capacity) {
      const old = this.object.geometry;
      this.buffer = this.allocate(Math.ceil(n * 1.5));
      this.object.geometry = this.geometryFor(this.buffer);
      old.dispose();
    }
    const out = this.buffer.array as Float32Array;
    let k = 0;
    let count = 0;
    for (let i = 0; i + SRC_STRIDE <= segs.length; i += SRC_STRIDE) {
      const frac = Math.min(1, segs[i + 7] / maxFlux);
      if (frac < FLUX_FLOOR) continue;
      const flags = Math.round(segs[i + 8]);
      const ghost = (flags & FLAG_GHOST) !== 0 ? 1 : 0;
      const r = segs[i + 4], g = segs[i + 5], b = segs[i + 6];
      out[k++] = segs[i]; out[k++] = segs[i + 1]; out[k++] = segs[i + 9];
      out[k++] = r; out[k++] = g; out[k++] = b; out[k++] = frac; out[k++] = ghost;
      out[k++] = segs[i + 2]; out[k++] = segs[i + 3]; out[k++] = segs[i + 10];
      out[k++] = r; out[k++] = g; out[k++] = b; out[k++] = frac; out[k++] = ghost;
      if ((flags & FLAG_TIR) !== 0 && tir.length < 4096) tir.push({ x: segs[i + 2], y: segs[i + 3], z: segs[i + 10] });
      count++;
    }
    this.buffer.clearUpdateRanges();
    this.buffer.addUpdateRange(0, k);
    this.buffer.needsUpdate = true;
    this.object.geometry.setDrawRange(0, count * 2);
    return count;
  }

  private fillStrips(segs: Float32Array, maxFlux: number, tir: RayUpload["tir"]): number {
    const n = Math.floor(segs.length / SRC_STRIDE);
    if (n > this.stripCapacity) {
      const old = this.strips.geometry;
      this.stripBuffer = this.allocateStrips(Math.ceil(n * 1.5));
      this.strips.geometry = this.stripGeometryFor(this.stripBuffer);
      old.dispose();
    }
    const out = this.stripBuffer.array as Float32Array;
    let k = 0;
    let count = 0;
    for (let i = 0; i + SRC_STRIDE <= segs.length; i += SRC_STRIDE) {
      const frac = Math.min(1, segs[i + 7] / maxFlux);
      if (frac < FLUX_FLOOR) continue;
      const flags = Math.round(segs[i + 8]);
      out[k++] = segs[i]; out[k++] = segs[i + 1]; out[k++] = segs[i + 9];
      out[k++] = segs[i + 2]; out[k++] = segs[i + 3]; out[k++] = segs[i + 10];
      out[k++] = segs[i + 4]; out[k++] = segs[i + 5]; out[k++] = segs[i + 6];
      out[k++] = frac;
      out[k++] = (flags & FLAG_GHOST) !== 0 ? 1 : 0;
      if ((flags & FLAG_TIR) !== 0 && tir.length < 4096) tir.push({ x: segs[i + 2], y: segs[i + 3], z: segs[i + 10] });
      count++;
    }
    this.stripBuffer.clearUpdateRanges();
    this.stripBuffer.addUpdateRange(0, k);
    this.stripBuffer.needsUpdate = true;
    (this.strips.geometry as THREE.InstancedBufferGeometry).instanceCount = count;
    return count;
  }

  /** The 2D view's ghost-opacity control, mapped the same way (x8). */
  setGhostOpacity(opacity: number) {
    this.ghostGain.value = opacity * 8;
  }

  /** Through components (drawn over everything) or occluded by opaque parts.
   *  Glass never occludes rays in either mode. */
  setVisibility(mode: RayVisibility) {
    this.visibility = mode;
    for (const m of [this.additive, this.stripMaterial]) {
      m.depthTest = mode === "occluded";
      m.needsUpdate = true;
    }
  }

  getVisibility(): RayVisibility {
    return this.visibility;
  }

  /** Additive on dark backgrounds, the ink pass on light ones. */
  setLightBackground(light: boolean) {
    this.inkMode = light;
    this.inkOverlay.visible = light;
  }

  dispose() {
    this.object.geometry.dispose();
    this.strips.geometry.dispose();
    this.additive.dispose();
    this.stripMaterial.dispose();
    this.inkTarget.dispose();
    this.inkOverlay.geometry.dispose();
    (this.inkOverlay.material as THREE.Material).dispose();
    this.depthOnly.dispose();
  }
}
