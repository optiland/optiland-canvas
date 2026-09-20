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
import type { RenderMesh } from "../../pkg/oc_wasm.js";
import type { Bounds3, CameraRig } from "./camera";
import { CueLayer } from "./cues";
import { EnvironmentManager, luminance } from "./environments";
import { HitMapLayer } from "./hitmaps";
import { MaterialLibrary } from "./materials";
import { RayBuffer, type RayUpload, type RayVisibility } from "./rays";
import { Scaffold, type ScaffoldTheme } from "./scaffold";
import { SceneRenderCache, type SyncReport } from "./scene-cache";
import type {
  BackgroundId,
  DetectorMap,
  EnvironmentId,
  RenderBackend,
  RenderObjectDescriptor,
  RenderStats,
} from "./types";

export type RendererOptions = {
  /** Use the WebGL2 backend even where WebGPU exists (tests, comparisons). */
  forceWebGL?: boolean;
  /** Measure GPU time per frame with timestamp queries, where the device has
   *  them (developer diagnostics; costs a read-back per frame). */
  timing?: boolean;
};

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
  private readonly materials = new MaterialLibrary();
  private readonly cache = new SceneRenderCache(this.materials);
  private readonly environments: EnvironmentManager;
  private readonly cues = new CueLayer(this.cache);
  private readonly raycaster = new THREE.Raycaster();
  private readonly rays = new RayBuffer();
  private readonly hitMaps = new HitMapLayer(this.cache);
  private lastRayUploadMs = 0;
  private lastSync: SyncReport | null = null;
  private lastSyncMs = 0;
  private cssWidth = 1;
  private cssHeight = 1;
  private dpr = 1;
  private frames = 0;
  private lastRenderMs = 0;
  private lastGpuMs: number | null = null;
  private timing = false;
  private gridSpacing = 5;
  private gridOn = true;
  private groundOn = true;
  private bounds: Bounds3 | null = null;
  private theme: ScaffoldTheme;

  private constructor(private readonly renderer: THREE.WebGPURenderer, timing: boolean) {
    this.timing = timing;
    const backend = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend ? "webgpu" : "webgl2";
    this.backend = backend;
    this.scene.name = "optiland-canvas-3d";
    this.scene.add(this.scaffold.root);
    this.scene.add(this.cache.root);
    for (const o of this.rays.objects) this.scene.add(o);
    this.scene.add(this.rays.inkOverlay);
    this.cache.onRebuilt = (id) => this.cues.refresh(id);
    this.environments = new EnvironmentManager(renderer, this.scene);
    this.environments.setEnvironment("dark_studio");
    this.theme = this.environments.scaffoldTheme();
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
      trackTimestamp: opts.timing ?? false,
    });
    await renderer.init();
    (renderer as unknown as Internals)._animation?.stop();
    renderer.info.autoReset = false;
    // A restrained tone curve that keeps hue, so wavelength colours stay
    // recognisable, and an sRGB output.
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const timing = (opts.timing ?? false) && renderer.hasFeature("timestamp-query");
    return new OpticalRenderer3D(renderer, timing);
  }

  resize(cssWidth: number, cssHeight: number, dpr: number) {
    if (cssWidth === this.cssWidth && cssHeight === this.cssHeight && dpr === this.dpr) return;
    this.cssWidth = Math.max(1, cssWidth);
    this.cssHeight = Math.max(1, cssHeight);
    this.dpr = dpr;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.cssWidth, this.cssHeight, false);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.rays.setDrawingSize(size.x, size.y);
  }

  /** Bring the render proxies up to the Rust scene's current descriptors.
   *  Only changed revisions do work (scene-cache.ts). `syncMs` covers the
   *  whole sync, including the wasm calls the caller made to get `descriptors`
   *  when it passes `startedAt`. */
  syncScene(
    descriptors: RenderObjectDescriptor[],
    fetchMesh: (id: number) => RenderMesh | undefined,
    startedAt = performance.now(),
  ): SyncReport {
    const report = this.cache.sync(descriptors, fetchMesh);
    this.lastSync = report;
    if (report.meshBuilds || report.added || report.removed || report.transformed) {
      this.bounds = this.cache.bounds();
      this.relayout();
    }
    this.lastSyncMs = performance.now() - startedAt;
    return report;
  }

  /** Lighting preset. Builds its PMREM the first time only. Different lights
   *  mean new shader variants; call `prepare` before the next frame so they
   *  compile off the main thread's frame instead of stalling it. */
  setEnvironment(id: EnvironmentId) {
    this.environments.setEnvironment(id);
    this.retheme();
  }

  /** Compile every pipeline the current scene needs, asynchronously. */
  async prepare(rig: CameraRig): Promise<void> {
    const cam = this.applyCamera(rig);
    await this.renderer.compileAsync(this.scene, cam);
  }

  /** Background, independent of the lighting. */
  setBackground(id: BackgroundId) {
    this.environments.setBackground(id);
    this.retheme();
  }

  /** Is the background shown light enough that bright display lines vanish? */
  lightBackground(): boolean {
    return luminance(this.environments.backgroundHex()) > 0.25;
  }

  private retheme() {
    this.theme = this.environments.scaffoldTheme();
    const light = this.lightBackground();
    this.materials.setCreaseTone(light);
    this.cues.setLightBackground(light);
    this.rays.setLightBackground(light);
    this.relayout();
  }

  /** Upload the traced segments (the 11-float projected layout). A large
   *  picture goes up a chunk per frame: call `stepRays` while `raysPending`. */
  updateRays(segs: Float32Array): RayUpload {
    const up = this.rays.update(segs);
    this.lastRayUploadMs = up.ms;
    return up;
  }

  raysPending(): boolean {
    return this.rays.pending;
  }

  stepRays(): RayUpload {
    const up = this.rays.step();
    this.lastRayUploadMs = up.ms;
    return up;
  }

  setGhostOpacity(opacity: number) {
    this.rays.setGhostOpacity(opacity);
  }

  setRayVisibility(mode: RayVisibility) {
    this.rays.setVisibility(mode);
  }

  /** Ray line width in CSS pixels (1 draws GPU lines, wider draws strips). A
   *  change of kind re-uploads the last trace, so call `prepare` after it. */
  setRayWidth(px: number) {
    this.rays.setWidth(px);
  }

  rayWidth(): number {
    return this.rays.getWidth();
  }

  /** Paint the detectors' hit maps from the latest exact trace. Returns true
   *  when a new material was made (call `prepare` before the next frame). */
  setDetectorMaps(maps: DetectorMap[]): boolean {
    return this.hitMaps.update(maps);
  }

  /** The user's hit-map toggle. */
  setHitMapsEnabled(on: boolean) {
    this.hitMaps.setEnabled(on);
  }

  /** Hide the maps while the picture on screen is not the exact trace. */
  setHitMapsShown(shown: boolean) {
    this.hitMaps.setShown(shown);
  }

  setHovered(id: number) {
    this.cues.setHovered(id);
  }

  setSelected(id: number) {
    this.cues.setSelected(id);
  }

  /** The object under a CSS-pixel point, by a ray through the render proxies.
   *  Only an id comes back; the selection itself is the app's. */
  pick(x: number, y: number, rig: CameraRig): number | null {
    const cam = this.applyCamera(rig);
    const ndc = new THREE.Vector2((x / rig.width) * 2 - 1, 1 - (y / rig.height) * 2);
    this.raycaster.setFromCamera(ndc, cam);
    for (const hit of this.raycaster.intersectObjects(this.cache.pickables(), false)) {
      const id = hit.object.userData.objectId;
      if (typeof id === "number") return id;
    }
    return null;
  }

  /** The background colour in effect (sRGB hex). */
  backgroundHex(): number {
    return this.environments.backgroundHex();
  }

  /** Display exposure (a multiplier before tone mapping). */
  setExposure(value: number) {
    this.renderer.toneMappingExposure = value;
  }

  setGround(on: boolean) {
    if (on === this.groundOn) return;
    this.groundOn = on;
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
    this.rays.setCamera(rig.perspective, near);
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
    if (this.rays.inkMode) this.renderInkPass(cam);
    this.rays.setDrawVisible(!this.rays.inkMode);
    this.renderer.render(this.scene, cam);
    this.rays.uploaded();
    this.rays.setDrawVisible(true);
    this.frames++;
    this.lastRenderMs = performance.now() - t0;
    if (this.timing) {
      void this.renderer.resolveTimestampsAsync("render").then((ms) => {
        if (typeof ms === "number") this.lastGpuMs = ms;
      });
    }
  }

  /** GPU time of the most recent frame the device has reported, or null. */
  async gpuTime(): Promise<number | null> {
    if (!this.timing) return null;
    const ms = await this.renderer.resolveTimestampsAsync("render");
    if (typeof ms === "number") this.lastGpuMs = ms;
    return this.lastGpuMs;
  }

  /** Light backgrounds: sum the rays into the ink target first. Only the rays,
   *  and (when rays are occluded) the opaque parts as depth, are drawn. */
  private renderInkPass(cam: THREE.Camera) {
    const r = this.renderer;
    const rays = this.rays;
    const occluded = rays.getVisibility() === "occluded";
    const background = this.scene.background;
    const clear = r.getClearColor(new THREE.Color());
    const clearAlpha = r.getClearAlpha();
    this.scene.background = null;
    r.setClearColor(0x000000, 0);
    r.setRenderTarget(rays.inkTarget);
    r.setRenderObjectFunction((object, scene, camera, geometry, material, group, lightsNode, clippingContext, passId) => {
      if (rays.isRayObject(object)) {
        r.renderObject(object, scene, camera, geometry, material, group, lightsNode, clippingContext, passId);
      } else if (occluded && object.userData.objectId !== undefined && material.depthWrite && !material.transparent) {
        r.renderObject(object, scene, camera, geometry, rays.depthOnly, group, lightsNode, clippingContext, passId);
      }
    });
    try {
      r.render(this.scene, cam);
    } finally {
      r.setRenderObjectFunction(null);
      r.setRenderTarget(null);
      r.setClearColor(clear, clearAlpha);
      this.scene.background = background;
    }
  }

  stats(): Partial<RenderStats> {
    const info = this.renderer.info;
    return {
      backend: this.backend,
      dpr: this.dpr,
      frames: this.frames,
      renderMs: this.lastRenderMs,
      gpuMs: this.lastGpuMs,
      drawCalls: info.render.drawCalls,
      triangles: info.render.triangles,
      lines: info.render.lines,
      geometries: info.memory.geometries,
      textures: info.memory.textures,
      meshBuilds: this.cache.meshBuilds,
      meshBuildsLastSync: this.lastSync?.meshBuilds ?? 0,
      meshBuildMs: this.lastSync?.meshBuildMs ?? 0,
      syncMs: this.lastSyncMs,
      cachedObjects: this.cache.size,
      environmentBuilds: this.environments.builds,
      raySegments: this.rays.count,
      rayUploads: this.rays.uploads,
      rayUploadMs: this.lastRayUploadMs,
      rayWidth: this.rays.getWidth(),
      hitMaps: this.hitMaps.size,
    };
  }

  dispose() {
    this.hitMaps.dispose();
    this.cues.dispose();
    this.rays.dispose();
    this.cache.dispose();
    this.materials.dispose();
    this.environments.dispose();
    this.scaffold.dispose();
    void this.renderer.dispose();
  }
}
