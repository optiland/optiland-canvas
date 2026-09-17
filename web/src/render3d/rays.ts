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

import * as THREE from "three/webgpu";
import { attribute, float, max, min, mix, positionGeometry, screenUV, texture, uniform, vec3, vec4 } from "three/tsl";

// The projected trace layout (traceProjected3D): 11 floats per segment,
// [au,av,bu,bv, r,g,b, flux, flags, dist_a, dist_b]; for the default XY view
// plane, (au, av, dist_a) is the world start point.
const SRC_STRIDE = 11;
// Per vertex: x y z, r g b, weight, ghost.
const VERTEX_FLOATS = 8;
/** Segments this dim (relative flux) are not uploaded at all. */
const FLUX_FLOOR = 1e-5;
const FLAG_TIR = 2;
const FLAG_GHOST = 4;

export type RayVisibility = "through" | "occluded";

export type RayUpload = {
  /** Segments uploaded. */
  count: number;
  /** World points where a surface is in total internal reflection. */
  tir: { x: number; y: number; z: number }[];
  ms: number;
};

export class RayBuffer {
  readonly object: THREE.LineSegments;
  private buffer: THREE.InterleavedBuffer;
  private capacity = 0;
  private readonly ghostGain = uniform(3.6);
  private readonly rayGain = uniform(1.4);
  private readonly additive: THREE.LineBasicNodeMaterial;
  private visibility: RayVisibility = "through";
  /** The light-background pass: summed rays, then an ink overlay. */
  readonly inkTarget = new THREE.RenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4, depthBuffer: true });
  readonly inkOverlay: THREE.Mesh;
  /** Opaque parts, drawn depth-only into the ink pass when rays are occluded. */
  readonly depthOnly = new THREE.MeshBasicNodeMaterial({ colorWrite: false });
  inkMode = false;
  count = 0;

  constructor() {
    this.buffer = this.allocate(1024);
    const geometry = this.geometryFor(this.buffer);
    this.additive = this.makeAdditive();
    this.inkOverlay = this.makeInkOverlay();
    this.object = new THREE.LineSegments(geometry, this.additive);
    this.object.name = "rays";
    this.object.frustumCulled = false;
    // After the components, the glass, and the cues.
    this.object.renderOrder = 40;
    this.object.raycast = () => {};
    this.setVisibility("through");
  }

  private allocate(segments: number): THREE.InterleavedBuffer {
    this.capacity = segments;
    const b = new THREE.InterleavedBuffer(new Float32Array(segments * 2 * VERTEX_FLOATS), VERTEX_FLOATS);
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

  /** The per-vertex display weight: relative flux, times the ghost gain on
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

  /** Upload a projected trace. */
  update(segs: Float32Array): RayUpload {
    const t0 = performance.now();
    const n = Math.floor(segs.length / SRC_STRIDE);
    let maxFlux = 1e-9;
    for (let i = 7; i < segs.length; i += SRC_STRIDE) if (segs[i] > maxFlux) maxFlux = segs[i];

    if (n > this.capacity) {
      const old = this.object.geometry;
      this.buffer = this.allocate(Math.ceil(n * 1.5));
      this.object.geometry = this.geometryFor(this.buffer);
      old.dispose();
    }
    const out = this.buffer.array as Float32Array;
    const tir: RayUpload["tir"] = [];
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
    this.count = count;
    return { count, tir, ms: performance.now() - t0 };
  }

  /** The 2D view's ghost-opacity control, mapped the same way (x8). */
  setGhostOpacity(opacity: number) {
    this.ghostGain.value = opacity * 8;
  }

  /** Through components (drawn over everything) or occluded by opaque parts.
   *  Glass never occludes rays in either mode. */
  setVisibility(mode: RayVisibility) {
    this.visibility = mode;
    this.additive.depthTest = mode === "occluded";
    this.additive.needsUpdate = true;
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
    this.additive.dispose();
    this.inkTarget.dispose();
    this.inkOverlay.geometry.dispose();
    (this.inkOverlay.material as THREE.Material).dispose();
    this.depthOnly.dispose();
  }
}
