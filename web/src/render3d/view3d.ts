// The 3D model view on the GPU renderer: interaction, state, and frame pacing.
//
// This module holds only UI state (camera, selection, hover, clipboard, the
// cached ray picture) and never imports three.js. The renderer arrives through a
// dynamic import the first time the 3D view is shown, so the 2D app starts
// without it.
//
// The scene lives in Rust. A drag computes the intended world position, writes
// it through the Rust setter, and the render proxies follow on the next frame
// from what Rust reports; nothing is moved in three.js first.
//
// Rays follow the app's two-tier rule: the interactive f32 picture while a body
// is dragged (fastpreview.ts picks the CPU mirror or WebGPU), and the exact f64
// trace once the drag ends or after any other edit. No number is computed here.

import type { Canvas } from "../../pkg/oc_wasm.js";
import type { FastPreview } from "../fastpreview";
import type { Reference, StandIn } from "../reference";
import { CameraRig, type Bounds3, type CameraPreset, type CameraState, type V2, type V3 } from "./camera";
import { createScene3Cards, type SceneObject } from "./cards3d";
import { drawAxisLabels, drawNotice, drawSelectionDot, drawTirRings } from "./overlays";
import { QualityController } from "./quality";
import type { OpticalRenderer3D } from "./renderer";
import {
  parseDetectorMaps,
  type BackgroundId,
  type DetectorMap,
  type EnvironmentId,
  type FrameCost,
  type RayVisibilityMode,
  type RenderBackend,
  type RenderStats,
} from "./types";

// The projected trace layout (traceProjected3D): 11 floats per segment,
// [au,av,bu,bv, r,g,b, flux, flags, dist_a, dist_b]. With the default XY view
// plane the in-plane (au,av) are world (x,y) and the signed distance is world z.
const STRIDE = 11;

export type FrameReport = {
  /** Ray segments drawn. */
  count: number;
  /** Whole frame, ms (trace + upload + render submission + overlay). */
  ms: number;
  /** Of which the ray trace, ms. */
  traceMs: number;
};

export type View3dThree = {
  kind: "three";
  /** Ask for a frame. Frames are coalesced to one per animation frame. */
  invalidate: () => void;
  refresh: () => void;
  selectedBody: () => number;
  activate: (selectId: number) => void;
  standDown: () => void;
  focusObject: (id: number) => void;
  /** Re-read after an edit still in progress elsewhere (the component view's
   *  handle drags): the fast picture, no exact trace. */
  previewEdit: () => void;
  resetView: () => void;
  setPreset: (name: string) => void;
  setPerspective: (on: boolean) => void;
  isPerspective: () => boolean;
  /** "render WebGPU" etc., for the status line. */
  backendLabel: () => string;
  /** Fetch the renderer code in idle time, so the first switch to 3D does not
   *  wait for the download. No GPU device is created. */
  prefetch: () => void;
  /** Debug hooks (the `?debug` build of the page reads these). */
  debug: View3dDebug;
};

export type View3dDebug = {
  ready: () => Promise<RenderBackend>;
  /** Draw one frame now (outside the animation-frame queue) and report it,
   *  with the GPU time when timing is on. */
  renderNow: () => Promise<RenderStats>;
  /** The current interaction-quality state. */
  quality: () => string;
  /** How many times the view has entered the interactive state. */
  qualityEpisodes: () => number;
  /** The selected object id, or -1. */
  selected: () => number;
  /** Start recording every drawn frame's cost (benchmarks). */
  startLog: () => void;
  /** Stop recording and return the frames recorded. */
  stopLog: () => FrameLogEntry[];
  /** Orbit the camera as a pointer drag of (dx, dy) px would (benchmarks). */
  orbit: (dx: number, dy: number) => void;
  /** Retrace exactly and re-read the scene, as after an external edit. */
  refresh: () => void;
  camera: () => CameraState;
  stats: () => RenderStats;
  /** Render now and count lit pixels of the GPU canvas (tests). */
  litPixels: () => Promise<number>;
  /** Render now and count clearly red or orange pixels of the GPU canvas only
   *  (the hit map's warm cells; rays, glass, and the default finishes are not
   *  warm). */
  warmPixels: () => Promise<number>;
  project: (p: V3) => V2;
  hovered: () => number;
  /** The appearance settings in effect (as saved). */
  prefs: () => View3dPrefs;
};

export type FrameLogEntry = FrameCost & { meshBuilds: number; raySegments: number; quality: string; dpr: number };

/** The 3D view's appearance, remembered in this browser. Display settings
 *  only; none of them is part of the scene or its file. */
export type View3dPrefs = {
  environment: EnvironmentId;
  background: BackgroundId;
  exposureEv: number;
  ground: boolean;
  rays: RayVisibilityMode;
  hitMaps: boolean;
  rayWidth: number;
  perspective: boolean;
};

const PREFS_KEY = "optiland-canvas:view3d";

export const DEFAULT_PREFS: View3dPrefs = {
  environment: "dark_studio",
  background: "match",
  exposureEv: 0,
  ground: true,
  rays: "through",
  hitMaps: true,
  rayWidth: 1,
  perspective: false,
};

/** The saved preferences, each field checked, or the defaults. Storage can be
 *  missing or refused (private windows); that is not an error. */
function loadPrefs(): View3dPrefs {
  const out = { ...DEFAULT_PREFS };
  let saved: Record<string, unknown> = {};
  try {
    saved = JSON.parse(localStorage.getItem(PREFS_KEY) ?? "{}") ?? {};
  } catch {
    return out;
  }
  const oneOf = <T extends string>(v: unknown, allowed: readonly T[]): v is T => allowed.includes(v as T);
  const num = (v: unknown, lo: number, hi: number) => typeof v === "number" && Number.isFinite(v) && v >= lo && v <= hi;
  if (oneOf(saved.environment, ["dark_studio", "light_studio", "optical_bench", "inspection"] as const)) out.environment = saved.environment;
  if (oneOf(saved.background, ["match", "black", "charcoal", "mid_gray", "light_gray"] as const)) out.background = saved.background;
  if (num(saved.exposureEv, -2, 2)) out.exposureEv = saved.exposureEv as number;
  if (typeof saved.ground === "boolean") out.ground = saved.ground;
  if (oneOf(saved.rays, ["through", "occluded"] as const)) out.rays = saved.rays;
  if (typeof saved.hitMaps === "boolean") out.hitMaps = saved.hitMaps;
  if (num(saved.rayWidth, 1, 6)) out.rayWidth = saved.rayWidth as number;
  if (typeof saved.perspective === "boolean") out.perspective = saved.perspective;
  return out;
}

function savePrefs(prefs: View3dPrefs) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // Not saved; the settings still apply to this page.
  }
}

export type View3dOptions = {
  scene: Canvas;
  /** The overlay canvas (#view): 2D annotations, and it receives the pointer. */
  view: HTMLCanvasElement;
  /** The GPU canvas (#view3d), underneath the overlay. */
  canvas: HTMLCanvasElement;
  owns: () => boolean;
  /** Repaint through main (which calls `invalidate` while 3D is active). */
  render: () => void;
  onSelect?: (id: number) => void;
  detectorRefresh?: () => void;
  fast?: FastPreview;
  /** The f64 reference, traced off this thread. Without it the view traces
   *  the reference itself, blocking the page while it does. */
  reference?: Reference;
  /** Counts the documents loaded; when it moves, the picture on screen is of
   *  another design and is dropped. */
  loads?: () => number;
  rayOptions: () => { ghostOpacity: number; showTir: boolean };
  grid: () => { on: boolean; spacing: number };
  /** Write the status line for a finished frame. */
  status: (r: FrameReport) => void;
  forceWebGL?: boolean;
  /** Simulate a platform with no usable 3D renderer (tests). */
  disable?: boolean;
  /** Measure GPU frame time (developer diagnostics). */
  timing?: boolean;
};

export function initView3dThree(opts: View3dOptions): View3dThree {
  const { scene, view, canvas } = opts;
  const ctx = view.getContext("2d")!;
  const rig = new CameraRig();
  let renderer: OpticalRenderer3D | null = null;
  let backend: RenderBackend = "loading";
  let failure = "";
  let loading: Promise<RenderBackend> | null = null;
  let active = false;
  let framed = false;
  let pending = 0;

  const quality = new QualityController(() => invalidate());

  // --- appearance settings (display only; remembered in this browser) ------
  const settings = loadPrefs();
  rig.setPerspective(settings.perspective);
  const remember = () => savePrefs({ ...settings, perspective: rig.perspective });
  function applySettings() {
    if (!renderer) return;
    renderer.setEnvironment(settings.environment);
    renderer.setBackground(settings.background);
    renderer.setExposure(2 ** settings.exposureEv);
    renderer.setGround(settings.ground);
    renderer.setRayVisibility(settings.rays);
    renderer.setRayWidth(settings.rayWidth);
    renderer.setHitMapsEnabled(settings.hitMaps);
  }
  const envSelect = document.getElementById("v3dEnv") as HTMLSelectElement | null;
  const bgSelect = document.getElementById("v3dBg") as HTMLSelectElement | null;
  const exposureInput = document.getElementById("v3dExposure") as HTMLInputElement | null;
  const exposureLabel = document.getElementById("v3dExposureVal");
  const groundBox = document.getElementById("v3dGround") as HTMLInputElement | null;
  const raysSelect = document.getElementById("v3dRays") as HTMLSelectElement | null;
  const hitsBox = document.getElementById("v3dHits") as HTMLInputElement | null;
  const widthInput = document.getElementById("v3dRayWidth") as HTMLInputElement | null;
  const widthLabel = document.getElementById("v3dRayWidthVal");
  const showExposure = () => {
    const v = settings.exposureEv;
    if (exposureLabel) exposureLabel.textContent = `${v === 0 ? "±" : v > 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} EV`;
  };
  const showWidth = () => {
    if (widthLabel) widthLabel.textContent = `${settings.rayWidth.toFixed(1)} px`;
  };
  // The controls start from the saved settings.
  if (envSelect) envSelect.value = settings.environment;
  if (bgSelect) bgSelect.value = settings.background;
  if (exposureInput) exposureInput.value = String(settings.exposureEv);
  if (groundBox) groundBox.checked = settings.ground;
  if (raysSelect) raysSelect.value = settings.rays;
  if (hitsBox) hitsBox.checked = settings.hitMaps;
  if (widthInput) widthInput.value = String(settings.rayWidth);
  showExposure();
  showWidth();
  for (const el of [envSelect, bgSelect, raysSelect]) el?.addEventListener("keydown", (e) => e.stopPropagation());
  raysSelect?.addEventListener("change", () => {
    settings.rays = raysSelect.value as RayVisibilityMode;
    remember();
    renderer?.setRayVisibility(settings.rays);
    invalidate();
  });
  envSelect?.addEventListener("change", () => {
    settings.environment = envSelect.value as EnvironmentId;
    remember();
    if (!renderer) return;
    renderer.setEnvironment(settings.environment);
    holdFramesWhile(renderer.prepare(rig));
  });
  hitsBox?.addEventListener("change", () => {
    settings.hitMaps = hitsBox.checked;
    remember();
    if (!renderer) return;
    renderer.setHitMapsEnabled(settings.hitMaps);
    holdFramesWhile(renderer.prepare(rig));
  });
  widthInput?.addEventListener("input", () => {
    settings.rayWidth = Number(widthInput.value) || 1;
    showWidth();
    remember();
    if (!renderer) return;
    const kindChanges = (settings.rayWidth > 1) !== (renderer.rayWidth() > 1);
    renderer.setRayWidth(settings.rayWidth);
    // Switching between lines and strips needs the other shader.
    if (kindChanges) holdFramesWhile(renderer.prepare(rig));
    else invalidate();
  });

  /** Keep the last picture on screen until `work` is done, then draw. */
  function holdFramesWhile(work: Promise<void>) {
    const p = work.catch(() => {}).finally(() => {
      if (compiling === p) compiling = null;
      invalidate();
    });
    compiling = p;
  }
  bgSelect?.addEventListener("change", () => {
    settings.background = bgSelect.value as BackgroundId;
    remember();
    renderer?.setBackground(settings.background);
    invalidate();
  });
  exposureInput?.addEventListener("input", () => {
    settings.exposureEv = Number(exposureInput.value);
    showExposure();
    remember();
    renderer?.setExposure(2 ** settings.exposureEv);
    quality.pulse();
    invalidate();
  });
  groundBox?.addEventListener("change", () => {
    settings.ground = groundBox.checked;
    remember();
    renderer?.setGround(settings.ground);
    invalidate();
  });

  // --- scene data (re-read from Rust; never mirrored) --------------------
  let segs: Float32Array<ArrayBufferLike> = new Float32Array(0);
  let tracedRevision = -1;
  let shownExact = false;
  // The workers' stand-in on screen, when the picture is one, and the load
  // count the picture belongs to.
  let shownStandIn: StandIn | null = null;
  let seenLoads = 0;
  let objList: SceneObject[] = [];
  let selected = -1;
  let hovered = -1;
  let clipboard = -1;
  let pasteN = 0;
  let lastTraceMs = 0;
  // Trace time since the last drawn frame, including the drag path's fast
  // traces made in the pointer handler.
  let traceSinceFrame = 0;
  // Frames are held while new shader variants compile (a lighting change).
  let compiling: Promise<void> | null = null;
  // The scene revision the render proxies were last synced at.
  let syncedRevision = -1;
  // The ray picture last uploaded to the GPU (by identity), and its TIR points.
  let uploadedSegs: Float32Array<ArrayBufferLike> | null = null;
  let rayCount = 0;
  let tirPoints: V3[] = [];
  // The detectors' hit maps from the latest exact trace, the revision they
  // belong to, and whether the renderer has them yet.
  let maps: DetectorMap[] = [];
  let mapsRevision = -1;
  let mapsUploaded = true;
  const cost: FrameCost = { total: 0, trace: 0, sync: 0, upload: 0, render: 0, overlay: 0 };
  let frameLog: FrameLogEntry[] | null = null;

  /** Bring the GPU proxies up to date with Rust, if the scene moved. */
  function syncRender(force = false) {
    if (!renderer) return;
    const rev = scene.sceneRevision();
    if (!force && rev === syncedRevision) return;
    const t0 = performance.now();
    const descriptors = JSON.parse(scene.renderObjectsJSON());
    renderer.syncScene(descriptors, (id) => scene.renderMesh(id), t0);
    syncedRevision = rev;
  }

  function select(id: number) {
    selected = id;
    renderer?.setSelected(id);
    opts.onSelect?.(id);
  }
  function clearSelection() {
    selected = -1;
    renderer?.setSelected(-1);
  }

  // A fast frame can arrive after `retrace` returns (WebGPU), so it is drawn only
  // when it is newer than the picture on screen, and never over an exact frame of
  // the same scene.
  function retrace(fast = false) {
    const t0 = performance.now();
    const loads = opts.loads?.() ?? 0;
    if (loads !== seenLoads) {
      seenLoads = loads;
      segs = new Float32Array(0);
      tracedRevision = -1;
      shownExact = false;
      shownStandIn = null;
    }
    if (!fast && opts.reference) {
      // The reference of the scene as it is, when it has arrived; until then
      // the fast picture stands in (it may replace whatever is on screen,
      // which is not this scene's reference), and the reference is asked for.
      const b = opts.reference.current();
      if (b) {
        opts.fast?.settled();
        segs = b.projected;
        tracedRevision = b.key.revision;
        shownExact = true;
        shownStandIn = null;
        maps = parseDetectorMaps(b.maps);
        mapsRevision = tracedRevision;
        mapsUploaded = false;
        lastTraceMs = performance.now() - t0;
        traceSinceFrame += lastTraceMs;
        return;
      }
      opts.reference.request();
      shownExact = false;
      // Where tracing that frame here would stall the page, the workers trace
      // it; the picture on screen holds until one lands. During a drag that
      // may be the frame of an earlier position, newer than the one shown.
      const s = opts.reference.standIn(shownStandIn);
      if (s === "coming") return;
      if (s !== "here") {
        segs = s.segs;
        tracedRevision = s.revision;
        shownStandIn = s;
        lastTraceMs = performance.now() - t0;
        traceSinceFrame += lastTraceMs;
        return;
      }
      fast = true;
    }
    if (!fast) {
      opts.fast?.settled();
      segs = scene.traceProjected3D();
      tracedRevision = scene.sceneRevision();
      shownExact = true;
      shownStandIn = null;
      // The same trace left the detector maps behind; no second trace.
      maps = parseDetectorMaps(scene.lastDetectorMapsF32());
      mapsRevision = tracedRevision;
      mapsUploaded = false;
    } else if (!opts.fast) {
      segs = scene.traceProjected3DFast();
      tracedRevision = scene.sceneRevision();
      shownExact = false;
      shownStandIn = null;
    } else {
      opts.fast.requestProjected(scene, (f) => {
        if (f.revision < tracedRevision || (f.revision === tracedRevision && shownExact)) return;
        segs = f.segs;
        tracedRevision = f.revision;
        shownExact = false;
        shownStandIn = null;
        if (f.late) invalidate();
      });
    }
    lastTraceMs = performance.now() - t0;
    traceSinceFrame += lastTraceMs;
  }

  const objectsPanel = document.getElementById("objects3d");
  function refreshObjects() {
    // This view also draws the light sources, so they are selectable here.
    objList = [...JSON.parse(scene.sceneObjectsJSON()), ...JSON.parse(scene.sourceObjectsJSON())];
    if (!objectsPanel) return;
    objectsPanel.replaceChildren();
    for (const o of objList) {
      const row = document.createElement("div");
      row.className = "row";
      const [x, y, z] = o.pos.map((v) => v.toFixed(1));
      const shape = o.kind === "profile" || o.kind === "source" ? `${o.kind} · ${o.construction}` : o.kind;
      const planar = o.is3d ? "" : ` <span class="planar" title="a 2D-only body: not traced in full 3D">⚠ 2D-only</span>`;
      row.innerHTML =
        `<span class="oid">#${o.id}</span>` +
        `<span class="okind">${shape}${planar}</span>` +
        `<span class="opos">(${x}, ${y}, ${z})</span>`;
      objectsPanel.appendChild(row);
    }
  }

  const project = (p: V3) => rig.project(p);
  const cards = createScene3Cards({
    scene,
    objects: () => objList,
    project,
    render: () => invalidate(),
    afterEdit: () => { afterEdit(); opts.detectorRefresh?.(); },
  });

  function afterEdit() {
    retrace(false);
    refreshObjects();
    cards.renderAll();
    invalidate();
  }

  // --- framing -----------------------------------------------------------
  // Frame the hardware, plus the ray points near it, so a steeply dispersed or
  // stray ray running to the scene bounds cannot blow up the box.
  function hardwareBounds(): Bounds3 | null {
    const fromRenderer = renderer?.sceneBounds();
    if (fromRenderer) return fromRenderer;
    if (objList.length === 0) return null;
    const lo = { x: Infinity, y: Infinity, z: Infinity };
    const hi = { x: -Infinity, y: -Infinity, z: -Infinity };
    for (const o of objList) {
      const r = Math.max(o.radius, 1);
      lo.x = Math.min(lo.x, o.pos[0] - r); hi.x = Math.max(hi.x, o.pos[0] + r);
      lo.y = Math.min(lo.y, o.pos[1] - r); hi.y = Math.max(hi.y, o.pos[1] + r);
      lo.z = Math.min(lo.z, o.pos[2] - r); hi.z = Math.max(hi.z, o.pos[2] + r);
    }
    return { min: lo, max: hi };
  }

  // The side panels cover the canvas edges; framing keeps the assembly between them.
  function panelInset(): number {
    let inset = 0;
    for (const id of ["m4dock", "controls", "dock"]) {
      const r = document.getElementById(id)?.getBoundingClientRect();
      if (!r || r.width === 0) continue;
      inset = Math.max(inset, r.left < window.innerWidth / 2 ? r.right : window.innerWidth - r.left);
    }
    return Math.min(inset, window.innerWidth * 0.3);
  }

  function frameScene() {
    // Frame what Rust holds now, not what the proxies held last frame.
    syncRender();
    rig.sideInset = panelInset();
    const hw = hardwareBounds();
    if (!hw) { rig.frame(null); return; }
    const lo = { ...hw.min }, hi = { ...hw.max };
    const c0 = { x: (lo.x + hi.x) / 2, y: (lo.y + hi.y) / 2, z: (lo.z + hi.z) / 2 };
    const span = Math.max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
    const limit = Math.max(80, span * 2);
    const grow = (x: number, y: number, z: number) => {
      if (Math.abs(x - c0.x) > limit || Math.abs(y - c0.y) > limit || Math.abs(z - c0.z) > limit) return;
      lo.x = Math.min(lo.x, x); hi.x = Math.max(hi.x, x);
      lo.y = Math.min(lo.y, y); hi.y = Math.max(hi.y, y);
      lo.z = Math.min(lo.z, z); hi.z = Math.max(hi.z, z);
    };
    for (let i = 0; i + STRIDE <= segs.length; i += STRIDE) {
      grow(segs[i], segs[i + 1], segs[i + 9]);
      grow(segs[i + 2], segs[i + 3], segs[i + 10]);
    }
    rig.frame({ min: lo, max: hi });
  }

  // --- renderer lifecycle --------------------------------------------------
  function ensureRenderer(): Promise<RenderBackend> {
    if (loading) return loading;
    loading = (async () => {
      if (opts.disable) {
        failure = "disabled for this page";
        backend = "unavailable";
        return backend;
      }
      try {
        const mod = await import("./renderer");
        renderer = await mod.OpticalRenderer3D.create(canvas, { forceWebGL: opts.forceWebGL, timing: opts.timing });
        backend = renderer.backend;
        applySettings();
        refreshObjects();
        syncRender(true);
        renderer.setSelected(selected);
        // Compile the first picture's shaders before drawing it.
        rig.setSize(window.innerWidth, window.innerHeight);
        holdFramesWhile(renderer.prepare(rig));
      } catch (e) {
        console.warn("3D renderer unavailable.", e);
        failure = e instanceof Error ? e.message : String(e);
        backend = "unavailable";
      }
      invalidate();
      return backend;
    })();
    return loading;
  }

  function invalidate() {
    if (!active || pending) return;
    pending = requestAnimationFrame(() => {
      pending = 0;
      drawFrame();
    });
  }

  function drawFrame() {
    if (!active) return;
    const t0 = performance.now();
    const W = window.innerWidth, H = window.innerHeight;
    rig.setSize(W, H);
    const dpr = window.devicePixelRatio || 1;

    // A stale picture is retraced exactly, unless a GPU frame for a newer
    // revision is already on its way (then the old picture holds until it lands).
    if (compiling) return; // the previous frame stays until the shaders are ready
    if (scene.sceneRevision() !== tracedRevision && !opts.fast?.pending()) {
      retrace(false);
      refreshObjects();
    }
    const traceMs = traceSinceFrame;
    traceSinceFrame = 0;
    let syncMs = 0, uploadMs = 0, renderMs = 0;

    const grid = opts.grid();
    const ro = opts.rayOptions();
    if (renderer) {
      const ts = performance.now();
      syncRender();
      syncMs = performance.now() - ts;
      if (segs !== uploadedSegs || renderer.raysPending()) {
        // A large picture goes up a chunk per frame.
        const up = segs !== uploadedSegs ? renderer.updateRays(segs) : renderer.stepRays();
        uploadedSegs = segs;
        rayCount = up.count;
        tirPoints = up.tir;
        uploadMs = up.ms;
      }
      if (!mapsUploaded) {
        mapsUploaded = true;
        // A detector's first map needs its shader; draw once it is ready.
        if (renderer.setDetectorMaps(maps)) {
          holdFramesWhile(renderer.prepare(rig));
          return;
        }
      }
      // A map belongs to the exact trace it came from, and only while the
      // scene is still the one that trace saw.
      renderer.setHitMapsShown(shownExact && mapsRevision === tracedRevision && tracedRevision === scene.sceneRevision());
      renderer.setGhostOpacity(ro.ghostOpacity);
      renderer.setGrid(grid.on, grid.spacing);
      renderer.resize(W, H, quality.pixelRatio(dpr));
      const tr = performance.now();
      renderer.render(rig);
      renderMs = performance.now() - tr;
    }
    const to = performance.now();

    // The overlay: transparent, so the GPU canvas shows through.
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (renderer) {
      if (ro.showTir && tirPoints.length) {
        const B = rig.basis();
        drawTirRings(ctx, tirPoints.map((p) => rig.project(p, B)));
      }
      drawAxisLabels(ctx, rig, renderer.axisEnds(), renderer.lightBackground());
      const so = objList.find((o) => o.id === selected);
      if (so) drawSelectionDot(ctx, rig, { x: so.pos[0], y: so.pos[1], z: so.pos[2] });
      cards.positionAll();
      cards.drawLeaders(ctx, selected);
    } else if (backend === "unavailable") {
      drawNotice(ctx, W, H, `3D renderer unavailable: ${failure}`, "error");
    } else {
      drawNotice(ctx, W, H, "loading the 3D renderer…", "info");
    }
    ctx.restore();

    const t1 = performance.now();
    Object.assign(cost, {
      total: t1 - t0,
      trace: traceMs,
      sync: syncMs,
      upload: uploadMs,
      render: renderMs,
      overlay: t1 - to,
    });
    if (frameLog && renderer) {
      const s = renderer.stats();
      frameLog.push({
        ...cost,
        meshBuilds: s.meshBuilds ?? 0,
        raySegments: rayCount,
        quality: quality.state,
        dpr: s.dpr ?? 0,
      });
    }
    opts.status({ count: renderer ? rayCount : 0, ms: t1 - t0, traceMs });
    if (renderer?.raysPending()) invalidate();
  }

  // --- picking -------------------------------------------------------------
  // A ray through the render proxies decides first. A thin plate seen edge-on
  // is hard to hit that way, so the projected centre and a coarse radius (the
  // wireframe view's rule) is the fallback.
  function pick(cx: number, cy: number): SceneObject | null {
    if (renderer) {
      syncRender();
      const id = renderer.pick(cx, cy, rig);
      if (id !== null) {
        const hit = objList.find((o) => o.id === id);
        if (hit) return hit;
      }
    }
    const B = rig.basis();
    let best: SceneObject | null = null, bestD = Infinity;
    for (const o of objList) {
      const P = { x: o.pos[0], y: o.pos[1], z: o.pos[2] };
      const c = rig.project(P, B);
      const d = Math.hypot(cx - c.x, cy - c.y);
      if (d <= o.radius * rig.pxPerMm(P, B) + 10 && d < bestD) { bestD = d; best = o; }
    }
    return best;
  }

  // --- interaction ---------------------------------------------------------
  type Mode = "orbit" | "pan" | "drag";
  let mode: Mode | null = null;
  let last: V2 = { x: 0, y: 0 };
  let dragId = -1;
  let dragStart: V3 = { x: 0, y: 0, z: 0 };
  let grab: V2 = { x: 0, y: 0 };
  const gridSnap = (v: number) => {
    const g = opts.grid();
    return g.on ? Math.round(v / g.spacing) * g.spacing : v;
  };

  view.addEventListener("pointerdown", (e) => {
    if (!owns() || e.button === 2) return;
    last = { x: e.clientX, y: e.clientY };
    const hit = e.button === 0 ? pick(e.clientX, e.clientY) : null;
    if (hit) {
      mode = "drag";
      dragId = hit.id;
      select(hit.id);
      dragStart = { x: hit.pos[0], y: hit.pos[1], z: hit.pos[2] };
      grab = { x: e.clientX, y: e.clientY };
    } else if (e.button === 1 || e.shiftKey) {
      mode = "pan";
    } else {
      mode = "orbit";
    }
    view.setPointerCapture(e.pointerId);
    view.style.cursor = "grabbing";
    quality.begin();
    invalidate();
    e.preventDefault();
  });

  view.addEventListener("pointermove", (e) => {
    if (!owns()) return;
    if (!mode) {
      updateHover(e.clientX, e.clientY);
      return;
    }
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    if (mode === "orbit") {
      rig.orbit(dx, dy);
    } else if (mode === "pan") {
      rig.pan(dx, dy);
    } else {
      const d = rig.screenDeltaToWorld(e.clientX - grab.x, e.clientY - grab.y, dragStart);
      scene.setObjectPosition(dragId, gridSnap(dragStart.x + d.x), gridSnap(dragStart.y + d.y), gridSnap(dragStart.z + d.z));
      retrace(true);
      refreshObjects();
      cards.renderAll();
    }
    invalidate();
  });

  function endPointer(e: PointerEvent) {
    if (!mode) return;
    const settled = mode === "drag";
    mode = null;
    try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    view.style.cursor = "";
    quality.end();
    if (settled) {
      retrace(false);
      refreshObjects();
      opts.detectorRefresh?.();
    }
    invalidate();
  }
  view.addEventListener("pointerup", endPointer);
  view.addEventListener("pointercancel", endPointer);

  // Hover is resolved at most once per animation frame, and never traces.
  let hoverAt: V2 | null = null;
  let hoverFrame = 0;
  function updateHover(cx: number, cy: number) {
    hoverAt = { x: cx, y: cy };
    if (hoverFrame) return;
    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      if (!hoverAt || mode || !active) return;
      const id = pick(hoverAt.x, hoverAt.y)?.id ?? -1;
      if (id === hovered) return;
      hovered = id;
      renderer?.setHovered(id);
      view.style.cursor = id >= 0 ? "pointer" : "";
      invalidate();
    });
  }
  view.addEventListener("pointerleave", () => {
    hoverAt = null;
    if (hovered < 0) return;
    hovered = -1;
    renderer?.setHovered(-1);
    invalidate();
  });

  view.addEventListener("wheel", (e) => {
    if (!owns()) return;
    e.preventDefault();
    rig.zoom(e.deltaY < 0 ? 1.12 : 1 / 1.12);
    quality.pulse();
    invalidate();
  }, { passive: false });

  view.addEventListener("dblclick", (e) => {
    if (!owns()) return;
    const o = pick(e.clientX, e.clientY);
    if (o) { select(o.id); cards.open(o.id); invalidate(); }
  });

  function pasteBody() {
    if (clipboard < 0) return;
    pasteN++;
    const g = opts.grid();
    const d = g.on ? g.spacing : 5;
    const id = scene.cloneBody3D(clipboard, d * pasteN, -d * pasteN, 0);
    if (id >= 0) { select(id); afterEdit(); cards.open(id); opts.detectorRefresh?.(); }
  }
  function deleteSelected() {
    if (selected < 0) return;
    if (scene.deleteObject3D(selected)) {
      cards.close(selected);
      clearSelection();
      afterEdit();
      opts.detectorRefresh?.();
    }
  }
  function nudge(key: string) {
    const o = objList.find((x) => x.id === selected);
    if (!o) return;
    const g = opts.grid();
    const s = g.on ? g.spacing : 1;
    // Arrows nudge along the image plane's horizontal and vertical.
    const { right, up } = rig.imagePlaneAxes();
    const h = key === "ArrowLeft" ? -s : key === "ArrowRight" ? s : 0;
    const v = key === "ArrowUp" ? s : key === "ArrowDown" ? -s : 0;
    scene.setObjectPosition(
      selected,
      o.pos[0] + right.x * h + up.x * v,
      o.pos[1] + right.y * h + up.y * v,
      o.pos[2] + right.z * h + up.z * v,
    );
    afterEdit();
    opts.detectorRefresh?.();
  }

  window.addEventListener("keydown", (e) => {
    if (!owns() || e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Escape") { cards.closeAll(); clearSelection(); invalidate(); return; }
    if (e.key === "0") { frameScene(); invalidate(); return; }
    if (mod && (e.key === "c" || e.key === "C")) { if (selected >= 0) clipboard = selected; e.preventDefault(); return; }
    if (mod && (e.key === "v" || e.key === "V")) { pasteBody(); e.preventDefault(); return; }
    if (mod && (e.key === "x" || e.key === "X")) { if (selected >= 0) { clipboard = selected; deleteSelected(); } e.preventDefault(); return; }
    if (mod && (e.key === "d" || e.key === "D")) { if (selected >= 0) { clipboard = selected; pasteBody(); } e.preventDefault(); return; }
    if (e.key === "Delete" || e.key === "Backspace") { deleteSelected(); e.preventDefault(); return; }
    if (e.key.startsWith("Arrow")) { nudge(e.key); e.preventDefault(); return; }
  });

  window.addEventListener("resize", () => invalidate());

  function owns() {
    return active && opts.owns();
  }

  // --- view lifecycle ------------------------------------------------------
  function activate(selectId: number) {
    active = true;
    canvas.style.display = "block";
    retrace(false);
    refreshObjects();
    if (selectId >= 0) select(selectId);
    void ensureRenderer().then(() => {
      if (!framed && renderer) {
        rig.setSize(window.innerWidth, window.innerHeight);
        frameScene();
        framed = true;
      }
      invalidate();
    });
    invalidate();
  }

  function standDown() {
    active = false;
    mode = null;
    hovered = -1;
    renderer?.setHovered(-1);
    quality.settleNow();
    cards.closeAll();
    canvas.style.display = "none";
    view.style.cursor = "";
    if (pending) { cancelAnimationFrame(pending); pending = 0; }
  }

  function focusObject(id: number) {
    retrace(false);
    refreshObjects();
    select(id);
    cards.open(id);
    invalidate();
  }

  const backendLabel = () => {
    switch (backend) {
      case "webgpu": return "render WebGPU";
      case "webgl2": return "render WebGL2 fallback";
      case "unavailable": return "render unavailable";
      default: return "render loading";
    }
  };

  const stats = (): RenderStats => ({
    backend,
    quality: quality.state,
    dpr: 0,
    frames: 0,
    renderMs: 0,
    gpuMs: null,
    drawCalls: 0,
    triangles: 0,
    lines: 0,
    raySegments: rayCount,
    meshBuilds: 0,
    meshBuildsLastSync: 0,
    meshBuildMs: 0,
    syncMs: 0,
    rayUploadMs: 0,
    rayUploads: 0,
    rayWidth: settings.rayWidth,
    hitMaps: 0,
    environmentBuilds: 0,
    geometries: 0,
    textures: 0,
    cachedObjects: objList.length,
    ...(renderer?.stats() ?? {}),
    frame: { ...cost },
    jsHeap: (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize ?? null,
  });

  /** Draw a frame now. A frame can stop to compile a new shader (a detector's
   *  first hit map); then wait for it and draw again. */
  async function drawSettled() {
    for (let i = 0; i < 4; i++) {
      if (compiling) await compiling;
      drawFrame();
      if (!compiling) return;
    }
  }

  /** Draw a frame and count the pixels `test` accepts, on the GPU canvas and,
   *  with `overlay`, the 2D annotations over it. */
  async function countPixels(overlay: boolean, test: (r: number, g: number, b: number) => boolean) {
    await ensureRenderer();
    if (!renderer) return 0;
    await drawSettled();
    const c = document.createElement("canvas");
    c.width = canvas.width;
    c.height = canvas.height;
    const c2 = c.getContext("2d")!;
    c2.drawImage(canvas, 0, 0);
    if (overlay) c2.drawImage(view, 0, 0, c.width, c.height);
    const { data } = c2.getImageData(0, 0, c.width, c.height);
    let n = 0;
    for (let i = 0; i < data.length; i += 4) if (test(data[i], data[i + 1], data[i + 2])) n++;
    return n;
  }

  const debug: View3dDebug = {
    ready: () => ensureRenderer(),
    renderNow: async () => {
      await ensureRenderer();
      await drawSettled();
      await renderer?.gpuTime();
      return stats();
    },
    quality: () => quality.state,
    qualityEpisodes: () => quality.episodes,
    selected: () => selected,
    startLog: () => { frameLog = []; },
    orbit: (dx, dy) => { rig.orbit(dx, dy); },
    refresh: () => { retrace(false); refreshObjects(); syncRender(true); },
    stopLog: () => {
      const out = frameLog ?? [];
      frameLog = null;
      return out;
    },
    camera: () => rig.state(),
    stats,
    litPixels: () => countPixels(true, (r, g, b) => r + g + b > 60),
    warmPixels: () => countPixels(false, (r, g, b) => r > 120 && r > g + 30 && r > b + 30),
    project,
    hovered: () => hovered,
    prefs: () => ({ ...settings, perspective: rig.perspective }),
  };

  return {
    kind: "three",
    invalidate,
    refresh: () => { retrace(false); refreshObjects(); syncRender(true); cards.renderAll(); invalidate(); },
    selectedBody: () => selected,
    activate,
    standDown,
    focusObject,
    previewEdit: () => {
      retrace(true);
      refreshObjects();
      cards.renderAll();
      quality.pulse();
      invalidate();
    },
    resetView: () => { rig.resetOrientation(); frameScene(); invalidate(); },
    setPreset: (name) => { rig.setPreset(name as CameraPreset); frameScene(); invalidate(); },
    setPerspective: (on) => { rig.setPerspective(on); remember(); invalidate(); },
    isPerspective: () => rig.perspective,
    backendLabel,
    prefetch: () => {
      if (opts.disable) return;
      const idle = (window as { requestIdleCallback?: (cb: () => void) => void }).requestIdleCallback;
      const go = () => void import("./renderer").catch(() => {});
      if (idle) idle(go);
      else setTimeout(go, 1500);
    },
    debug,
  };
}
