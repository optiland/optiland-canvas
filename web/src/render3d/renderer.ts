// The 3D view's GPU renderer: a three.js presentation layer over the Rust scene.
//
// This is the only module (with its helpers in this folder) that imports
// three.js, and main.ts reaches it through a dynamic import, so the library
// loads only when the 3D view is first used.
//
// What this module is not: a scene model, an optics engine, or a ray tracer.
// Every object here is a disposable render proxy for a Rust object at some
// revision. No optical quantity is computed here, and nothing drawn here is a
// measurement. Three.js materials give physically plausible *display* shading.
//
// Units and axes: 1 three.js unit = 1 mm, world axes are the Rust scene's
// (+X optical axis, +Y up, +Z out of plane), and the camera's up is +Y.
//
// Frames are event driven. three.js 0.186 starts its own requestAnimationFrame
// loop in `init()` for bookkeeping, even with no animation callback; that loop
// is stopped here and its per-frame bookkeeping runs inside `render()` instead,
// so an idle view costs nothing.

import * as THREE from "three/webgpu";
import type { Bounds3, CameraRig } from "./camera";
import { Scaffold, type ScaffoldTheme } from "./scaffold";
import type { RenderBackend, RenderStats } from "./types";

export type RendererOptions = {
  /** Use the WebGL2 backend even where WebGPU exists (tests, comparisons). */
  forceWebGL?: boolean;
};

/** A temporary stand-in for a scene object (Phase 1, before the mesh bridge). */
export type Placeholder = { id: number; pos: [number, number, number]; radius: number };

type Internals = {
  _animation?: { stop(): void };
  _nodes?: { nodeFrame: { update(): void; frameId: number } };
};

export class OpticalRenderer3D {
  readonly backend: Exclude<RenderBackend, "loading" | "unavailable">;
  private readonly scene = new THREE.Scene();
  private readonly ortho = new THREE.OrthographicCamera();
  private readonly persp = new THREE.PerspectiveCamera();
  private readonly scaffold = new Scaffold();
  private readonly placeholders = new THREE.Group();
  private readonly placeholderMaterial = new THREE.MeshBasicNodeMaterial({ color: 0x7fb6e0, wireframe: true });
  private readonly sphere = new THREE.IcosahedronGeometry(1, 2);
  private cssWidth = 1;
  private cssHeight = 1;
  private dpr = 1;
  private frames = 0;
  private lastRenderMs = 0;
  private gridSpacing = 5;
  private gridOn = true;
  private groundOn = true;
  private bounds: Bounds3 | null = null;
  private theme: ScaffoldTheme = {
    grid: new THREE.Color(0.47, 0.59, 0.75),
    gridOpacity: 0.35,
    ground: null,
  };

  private constructor(private readonly renderer: THREE.WebGPURenderer) {
    const backend = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend ? "webgpu" : "webgl2";
    this.backend = backend;
    this.scene.name = "optiland-canvas-3d";
    this.scene.background = new THREE.Color(0x06080d);
    this.scene.add(this.scaffold.root);
    this.placeholders.name = "placeholders";
    this.scene.add(this.placeholders);
    this.scaffold.layout(null, this.gridSpacing, this.gridOn, this.groundOn, this.theme);
  }

  /** Create and initialize a renderer on `canvas`. Tries WebGPU, falls back to
   *  WebGL2 inside three.js, and rejects if neither works. */
  static async create(canvas: HTMLCanvasElement, opts: RendererOptions = {}): Promise<OpticalRenderer3D> {
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      forceWebGL: opts.forceWebGL ?? false,
      alpha: false,
    });
    await renderer.init();
    (renderer as unknown as Internals)._animation?.stop();
    renderer.info.autoReset = false;
    // A restrained tone curve that keeps hue, so wavelength colours stay
    // recognisable, and an sRGB output.
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    return new OpticalRenderer3D(renderer);
  }

  resize(cssWidth: number, cssHeight: number, dpr: number) {
    if (cssWidth === this.cssWidth && cssHeight === this.cssHeight && dpr === this.dpr) return;
    this.cssWidth = Math.max(1, cssWidth);
    this.cssHeight = Math.max(1, cssHeight);
    this.dpr = dpr;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.cssWidth, this.cssHeight, false);
  }

  /** Temporary Phase 1 stand-ins: a wire sphere per object. */
  setPlaceholders(items: Placeholder[]) {
    this.placeholders.clear();
    const lo = { x: Infinity, y: Infinity, z: Infinity };
    const hi = { x: -Infinity, y: -Infinity, z: -Infinity };
    for (const it of items) {
      const m = new THREE.Mesh(this.sphere, this.placeholderMaterial);
      m.position.set(...it.pos);
      m.scale.setScalar(Math.max(0.5, it.radius));
      m.userData.objectId = it.id;
      this.placeholders.add(m);
      const r = Math.max(0.5, it.radius);
      lo.x = Math.min(lo.x, it.pos[0] - r); hi.x = Math.max(hi.x, it.pos[0] + r);
      lo.y = Math.min(lo.y, it.pos[1] - r); hi.y = Math.max(hi.y, it.pos[1] + r);
      lo.z = Math.min(lo.z, it.pos[2] - r); hi.z = Math.max(hi.z, it.pos[2] + r);
    }
    this.bounds = Number.isFinite(lo.x) ? { min: lo, max: hi } : null;
    this.relayout();
  }

  setGrid(on: boolean, spacing: number) {
    if (on === this.gridOn && spacing === this.gridSpacing) return;
    this.gridOn = on;
    this.gridSpacing = spacing;
    this.relayout();
  }

  /** The assembly's world bounds, from the render proxies. */
  sceneBounds(): Bounds3 | null {
    return this.bounds;
  }

  axisEnds() {
    return this.scaffold.axisEnds();
  }

  private relayout() {
    this.scaffold.layout(this.bounds, this.gridSpacing, this.gridOn, this.groundOn, this.theme);
  }

  /** Copy the rig into the active three.js camera. */
  private applyCamera(rig: CameraRig): THREE.Camera {
    const B = rig.basis();
    const eye = rig.eye(B);
    const { near, far } = rig.clipRange();
    const cam = rig.perspective ? this.persp : this.ortho;
    cam.position.set(eye.x, eye.y, eye.z);
    cam.up.set(0, 1, 0);
    cam.lookAt(rig.target.x, rig.target.y, rig.target.z);
    if (rig.perspective) {
      this.persp.fov = THREE.MathUtils.radToDeg(rig.fovY());
      this.persp.aspect = rig.width / rig.height;
      this.persp.near = near;
      this.persp.far = far;
    } else {
      const hw = rig.width / 2 / rig.scale, hh = rig.height / 2 / rig.scale;
      this.ortho.left = -hw;
      this.ortho.right = hw;
      this.ortho.top = hh;
      this.ortho.bottom = -hh;
      this.ortho.near = near;
      this.ortho.far = far;
    }
    cam.updateMatrixWorld();
    (cam as THREE.OrthographicCamera | THREE.PerspectiveCamera).updateProjectionMatrix();
    return cam;
  }

  render(rig: CameraRig) {
    const t0 = performance.now();
    const cam = this.applyCamera(rig);
    // The bookkeeping three.js's own loop would have done this frame.
    const internals = this.renderer as unknown as Internals;
    this.renderer.info.reset();
    if (internals._nodes) {
      internals._nodes.nodeFrame.update();
      (this.renderer.info as { frame: number }).frame = internals._nodes.nodeFrame.frameId;
    }
    this.renderer.render(this.scene, cam);
    this.frames++;
    this.lastRenderMs = performance.now() - t0;
  }

  stats(): Partial<RenderStats> {
    const info = this.renderer.info;
    return {
      backend: this.backend,
      dpr: this.dpr,
      frames: this.frames,
      renderMs: this.lastRenderMs,
      gpuMs: null,
      drawCalls: info.render.drawCalls,
      triangles: info.render.triangles,
      lines: info.render.lines,
      geometries: info.memory.geometries,
      textures: info.memory.textures,
    };
  }

  dispose() {
    this.placeholders.clear();
    this.sphere.dispose();
    this.placeholderMaterial.dispose();
    this.scaffold.dispose();
    void this.renderer.dispose();
  }
}
