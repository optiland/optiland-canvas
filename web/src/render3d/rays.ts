// The ray bundle on the GPU: every traced segment, in one draw call per
// CHUNK_SEGS segments. WebGPU caps a buffer at 256 MB unless the device asks
// for more, and a ghost-lit lens traces millions of segments, so the picture
// is held in chunks, one buffer and one object each. A picture of more than
// one chunk is written and uploaded a chunk per frame, so the page does not
// stop while tens of millions of vertices go to the GPU.
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
/** Room a buffer grows by, so a drag that adds a few segments per frame does
 *  not reallocate every frame; a picture past `GROW_UP_TO` segments gets the
 *  room it needs and no more (a ghost-lit lens has millions). */
const GROWTH = 1.5;
const GROW_UP_TO = 1_000_000;

// The ray objects are never culled, and nothing else transparent shares their
// render order, so their bounding sphere decides nothing. Given one, three.js
// does not compute it over every vertex of a new buffer: seconds for a
// picture of millions of segments.
const EVERYWHERE = new THREE.Sphere(new THREE.Vector3(), Infinity);
const room = (n: number) => (n > GROW_UP_TO ? n : Math.ceil(n * GROWTH));
/** Segments per GPU buffer: 32 MB of lines, 22 MB of strips. */
const CHUNK_SEGS = 500_000;
/** Whole chunks written per frame while a large picture is uploaded, at most,
 *  and the arrays on this side they are written into, in turn. */
const CHUNKS_PER_FRAME = 2;
/** Time a frame spends writing chunks, at most, ms. */
const FILL_MS = 12;

/** One buffer of the picture and the object that draws it. */
type Chunk = {
  buffer: THREE.InterleavedBuffer;
  capacity: number;
  object: THREE.LineSegments | THREE.Mesh;
};

export type RayVisibility = "through" | "occluded";

export type RayUpload = {
  /** Segments uploaded. */
  count: number;
  /** World points where a surface is in total internal reflection. */
  tir: { x: number; y: number; z: number }[];
  ms: number;
};

export class RayBuffer {
  /** The 1 px picture: GPU lines, an object per chunk. */
  readonly lines = new THREE.Group();
  /** The wide picture: one screen-facing quad per segment, chunked alike. */
  readonly wide = new THREE.Group();
  private lineChunks: Chunk[] = [];
  private stripChunks: Chunk[] = [];
  /** A picture of more than one chunk being written, a chunk per `step`:
   *  its segments, their brightness scale, the next segment to write, and
   *  the next chunk to write it into. */
  private filling: { segs: Float32Array; maxFlux: number; src: number; at: number; wide: boolean } | null = null;
  private tir: RayUpload["tir"] = [];
  // Whole chunks share CHUNKS_PER_FRAME arrays per kind on this side, in
  // turn: a chunk is written into one, the next render uploads it, and the
  // array is written again only after that (`uploaded`). A picture of
  // millions of segments then holds a few chunks of vertices here instead of
  // all of them, next to its copy on the GPU.
  private shared: { lines: Float32Array[]; strips: Float32Array[] } = { lines: [], strips: [] };
  // Chunks written into a shared array and not uploaded yet.
  private awaiting: Chunk[] = [];
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
    this.additive = this.makeAdditive();
    this.stripMaterial = this.makeStrip();
    this.inkOverlay = this.makeInkOverlay();
    this.lines.name = "rays";
    this.wide.name = "rays-wide";
    this.lineChunks.push(this.lineChunk(1024, 0));
    this.stripChunks.push(this.stripChunk(1024, 0));
    this.wide.visible = false;
    this.setVisibility("through");
  }

  /** The objects that draw rays; add both to the scene. */
  get objects(): THREE.Object3D[] {
    return [this.lines, this.wide];
  }

  isRayObject(o: THREE.Object3D): boolean {
    return o.parent === this.lines || o.parent === this.wide;
  }

  /** Show the active picture (lines or strips), or neither. */
  setDrawVisible(on: boolean) {
    this.lines.visible = on && this.width <= 1;
    this.wide.visible = on && this.width > 1;
  }

  private place(o: THREE.Object3D, group: THREE.Group) {
    o.frustumCulled = false;
    // After the components, the glass, and the cues.
    o.renderOrder = 40;
    o.raycast = () => {};
    group.add(o);
  }

  private lineChunk(capacity: number, position: number): Chunk {
    // Static usage: three.js uploads a buffer when it is marked changed, and
    // then only the range marked. A dynamic buffer it writes whole on every
    // frame, which for a ghost-lit lens is hundreds of megabytes per frame.
    const array =
      capacity === CHUNK_SEGS
        ? this.sharedArray(this.shared.lines, position, CHUNK_SEGS * 2 * VERTEX_FLOATS)
        : new Float32Array(capacity * 2 * VERTEX_FLOATS);
    const buffer = new THREE.InterleavedBuffer(array, VERTEX_FLOATS);
    const object = new THREE.LineSegments(this.geometryFor(buffer), this.additive);
    object.name = "rays";
    this.place(object, this.lines);
    return { buffer, capacity, object };
  }

  private stripChunk(capacity: number, position: number): Chunk {
    // Static usage, as for the lines.
    const array =
      capacity === CHUNK_SEGS
        ? this.sharedArray(this.shared.strips, position, CHUNK_SEGS * INSTANCE_FLOATS)
        : new Float32Array(capacity * INSTANCE_FLOATS);
    const buffer = new THREE.InstancedInterleavedBuffer(array, INSTANCE_FLOATS, 1);
    const object = new THREE.Mesh(this.stripGeometryFor(buffer), this.stripMaterial);
    object.name = "rays-wide";
    this.place(object, this.wide);
    return { buffer, capacity, object };
  }

  /** Enough chunks for `n` segments: the first grows, with room, up to a
   *  whole chunk; the rest are whole chunks; and the chunks a smaller picture
   *  does not need are freed, since a ghost-lit picture can hold gigabytes. */
  private fit(chunks: Chunk[], n: number, make: (capacity: number, position: number) => Chunk) {
    const need = Math.max(1, Math.ceil(n / CHUNK_SEGS));
    const first = need > 1 ? CHUNK_SEGS : n;
    if (first > chunks[0].capacity) {
      const old = chunks[0];
      chunks[0] = make(Math.min(CHUNK_SEGS, room(first)), 0);
      this.drop(old);
    }
    while (chunks.length < need) chunks.push(make(CHUNK_SEGS, chunks.length));
    while (chunks.length > need) this.drop(chunks.pop()!);
  }

  private drop(c: Chunk) {
    c.object.removeFromParent();
    c.object.geometry.dispose();
  }

  /** Mark the first `floats` of a chunk for upload, drawing `segments`. */
  private close(c: Chunk, floats: number, segments: number) {
    c.buffer.clearUpdateRanges();
    c.buffer.addUpdateRange(0, floats);
    c.buffer.needsUpdate = true;
    const g = c.object.geometry;
    if (g instanceof THREE.InstancedBufferGeometry) g.instanceCount = segments;
    else g.setDrawRange(0, segments * 2);
  }

  private geometryFor(b: THREE.InterleavedBuffer): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.InterleavedBufferAttribute(b, 3, 0));
    g.setAttribute("rayColor", new THREE.InterleavedBufferAttribute(b, 3, 3));
    g.setAttribute("rayWeight", new THREE.InterleavedBufferAttribute(b, 1, 6));
    g.setAttribute("rayGhost", new THREE.InterleavedBufferAttribute(b, 1, 7));
    g.setDrawRange(0, 0);
    g.boundingSphere = EVERYWHERE.clone();
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
    g.boundingSphere = EVERYWHERE.clone();
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

  /** Upload a projected trace into the active picture. A picture that fits
   *  one chunk is written now. A larger one is written a chunk now and a
   *  chunk per later `step` while `pending`; the chunks not yet rewritten
   *  keep the last picture's segments until then, which is a frame or two of
   *  a drag, and the view drops the picture outright when a new design loads. */
  update(segs: Float32Array): RayUpload {
    const t0 = performance.now();
    this.last = segs;
    let maxFlux = 1e-9;
    for (let i = 7; i < segs.length; i += SRC_STRIDE) if (segs[i] > maxFlux) maxFlux = segs[i];
    const wide = this.width > 1;
    const chunks = wide ? this.stripChunks : this.lineChunks;
    const n = Math.floor(segs.length / SRC_STRIDE);
    this.count = 0;
    this.tir = [];
    this.uploads++;
    this.filling = null;
    // A chunk of the last picture still waiting for its upload would upload
    // whatever is written next into the array it shares: it is hidden until
    // this picture rewrites or frees it.
    for (const c of this.awaiting) this.hide(c);
    this.awaiting = [];
    if (n <= CHUNK_SEGS) {
      this.fit(chunks, n, (c, at) => this.make(wide, c, at));
      this.fill(chunks[0], segs, 0, maxFlux, wide);
      return { count: this.count, tir: this.tir, ms: performance.now() - t0 };
    }
    if (chunks[0].capacity < CHUNK_SEGS) {
      const old = chunks[0];
      chunks[0] = this.make(wide, CHUNK_SEGS, 0);
      this.drop(old);
    }
    this.filling = { segs, maxFlux, src: 0, at: 0, wide };
    const up = this.step(t0);
    return { ...up, ms: performance.now() - t0 };
  }

  /** A large picture is still being written. */
  get pending(): boolean {
    return this.filling !== null;
  }

  /** Write the next chunks of the picture `update` began, for the next
   *  render to upload: at least one, more while the frame's FILL_MS (from
   *  `since`) last, and none while the shared arrays wait for an upload. */
  step(since = performance.now()): RayUpload {
    const t0 = performance.now();
    const f = this.filling;
    const chunks = f ? (f.wide ? this.stripChunks : this.lineChunks) : [];
    let wrote = 0;
    while (
      f && this.filling && this.awaiting.length < CHUNKS_PER_FRAME &&
      (wrote === 0 || performance.now() - since < FILL_MS)
    ) {
      wrote++;
      if (f.at >= chunks.length) chunks.push(this.make(f.wide, CHUNK_SEGS, f.at));
      f.src = this.fill(chunks[f.at], f.segs, f.src, f.maxFlux, f.wide);
      f.at++;
      if (f.src >= Math.floor(f.segs.length / SRC_STRIDE)) {
        // Written: the chunks the last picture needed beyond these are freed.
        while (chunks.length > f.at) this.drop(chunks.pop()!);
        this.filling = null;
      }
    }
    return { count: this.count, tir: this.tir, ms: performance.now() - t0 };
  }

  /** The render has uploaded what was written: the shared arrays are free. */
  uploaded() {
    this.awaiting = [];
  }

  /** The shared array a whole chunk at position `index` is written into. */
  private sharedArray(pool: Float32Array[], index: number, floats: number): Float32Array {
    const slot = index % CHUNKS_PER_FRAME;
    return (pool[slot] ??= new Float32Array(floats));
  }

  private hide(c: Chunk) {
    const g = c.object.geometry;
    if (g instanceof THREE.InstancedBufferGeometry) g.instanceCount = 0;
    else g.setDrawRange(0, 0);
  }

  private make(wide: boolean, capacity: number, position: number): Chunk {
    return wide ? this.stripChunk(capacity, position) : this.lineChunk(capacity, position);
  }

  /** Write segments into a chunk from segment `from` until it is full or the
   *  picture ends, and mark them for upload. Returns the next segment. */
  private fill(c: Chunk, segs: Float32Array, from: number, maxFlux: number, wide: boolean): number {
    const out = c.buffer.array as Float32Array;
    if (this.shared.lines.includes(out) || this.shared.strips.includes(out)) this.awaiting.push(c);
    const tir = this.tir;
    const n = Math.floor(segs.length / SRC_STRIDE);
    let k = 0;
    let held = 0;
    let s = from;
    for (; s < n; s++) {
      const i = s * SRC_STRIDE;
      const frac = Math.min(1, segs[i + 7] / maxFlux);
      if (frac < FLUX_FLOOR) continue;
      if (held === c.capacity) break;
      held++;
      const flags = Math.round(segs[i + 8]);
      const ghost = (flags & FLAG_GHOST) !== 0 ? 1 : 0;
      if (wide) {
        out[k++] = segs[i]; out[k++] = segs[i + 1]; out[k++] = segs[i + 9];
        out[k++] = segs[i + 2]; out[k++] = segs[i + 3]; out[k++] = segs[i + 10];
        out[k++] = segs[i + 4]; out[k++] = segs[i + 5]; out[k++] = segs[i + 6];
        out[k++] = frac;
        out[k++] = ghost;
      } else {
        const r = segs[i + 4], g = segs[i + 5], b = segs[i + 6];
        out[k++] = segs[i]; out[k++] = segs[i + 1]; out[k++] = segs[i + 9];
        out[k++] = r; out[k++] = g; out[k++] = b; out[k++] = frac; out[k++] = ghost;
        out[k++] = segs[i + 2]; out[k++] = segs[i + 3]; out[k++] = segs[i + 10];
        out[k++] = r; out[k++] = g; out[k++] = b; out[k++] = frac; out[k++] = ghost;
      }
      if ((flags & FLAG_TIR) !== 0 && tir.length < 4096) tir.push({ x: segs[i + 2], y: segs[i + 3], z: segs[i + 10] });
    }
    this.close(c, k, held);
    this.count += held;
    return s;
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
    for (const c of [...this.lineChunks, ...this.stripChunks]) c.object.geometry.dispose();
    this.additive.dispose();
    this.stripMaterial.dispose();
    this.inkTarget.dispose();
    this.inkOverlay.geometry.dispose();
    (this.inkOverlay.material as THREE.Material).dispose();
    this.depthOnly.dispose();
  }
}
