// The interactive 3D picture (§4.1: f32 draws, f64 settles), traced on WebGPU,
// on the CPU f32 mirror on this thread, or by the reference's workers,
// whichever is fastest for the scene.
//
// Both views ask for the fast projected trace through here. The CPU mirror
// answers before `requestProjected` returns. The GPU answers later, so GPU
// requests are single-flight: while one runs, a newer request replaces any
// queued one, and the queued one runs next. Every frame carries the scene
// revision it was traced at, and the views decide whether it is still worth
// drawing. A frame this thread cannot afford goes to the workers
// (`whenCostly`), and the views draw it when it lands.
//
// The GPU is not always the fastest. A ghost-lit lens's ray trees are deep
// and uneven, and measured in Chromium (docs/performance.md) an RTX 5070 took
// 244 ms for a frame of the 60x objective that the CPU mirror traced in 286 ms
// on one thread, and a Radeon iGPU took 540 ms; eight workers share the CPU
// frame. A GPU frame also holds the GPU for as long as it runs. So a ghost
// setting whose GPU frames are slow, or whose ray trees the GPU cannot hold,
// is drawn by the workers until another design loads.
//
// No displayed number comes from here, on any backend.

import { createGpuPreview3, type Canvas, type GpuPreview3 } from "../pkg/oc_wasm.js";

export type FastFrame = {
  segs: Float32Array<ArrayBufferLike>;
  /** The scene revision the frame was traced at. */
  revision: number;
  /** The frame arrived after `requestProjected` returned (the caller repaints). */
  late: boolean;
};

type Request = { scene: Canvas; deliver: (f: FastFrame) => void; anyCost: boolean };

/** The ghost layer off or on: the two costs the preview keeps apart. */
type Setting = "plain" | "ghosts";
const settingOf = (scene: Canvas): Setting => (scene.ghostsEnabled() ? "ghosts" : "plain");

export type FastPreview = {
  /** Resolves once the GPU is ready or refused. */
  ready: Promise<void>;
  /** The backend that draws, and why, for the status line. */
  label: () => string;
  /** Ask for the projected fast trace of `scene` at its current revision. A
   *  frame this thread cannot afford is handed to `whenCostly` and not
   *  delivered here, unless `anyCost` (a test or a debug hook) asks for it
   *  on this thread whatever it costs. */
  requestProjected: (scene: Canvas, deliver: (f: FastFrame) => void, anyCost?: boolean) => void;
  /** A GPU trace is running or queued, so a newer frame is on its way. */
  pending: () => boolean;
  /** The exact trace just replaced the picture: drop the queued request. */
  settled: () => void;
  /** A new design: what was learned about the last one's costs is dropped. */
  forget: () => void;
  /** Should the workers trace this scene's frame rather than this thread?
   *  Yes when a CPU frame would stall the page (its last cost with the same
   *  ghost setting, or with ghosts on and no cost known yet, since a
   *  ghost-lit lens can take seconds) and the GPU has not shown it draws the
   *  setting quickly. */
  costly: (scene: Canvas) => boolean;
  /** Who draws a frame this thread cannot afford: a frame asked for while
   *  the CPU mirror is costly, or one the GPU declined, is not traced here
   *  but handed to `f` (the reference's workers trace it). */
  whenCostly: (f: () => void) => void;
  /** What a CPU frame of the scene costs, as the workers measured it. */
  noteCpuCost: (ghosts: boolean, ms: number) => void;
  /** The GPU handle, for the `?debug` hooks. `null` on the CPU mirror. */
  gpu: () => GpuPreview3 | null;
};

// The projected layout: 11 floats per segment.
const STRIDE = 11;

// Measured on an RTX 5070 in Chromium (web/tests/webgpu.spec.ts): a GPU frame
// costs about 3.5 ms even for a tiny scene (upload, dispatch, read-back), and the
// CPU mirror falls behind it at roughly 5,000 segments. The last frame's segment
// count picks the backend, with a gap so a scene near the line does not flap.
const GPU_FROM_SEGS = 5000;
const CPU_BELOW_SEGS = 3000;

// A CPU frame slower than this, on the page's thread, is a stall a user feels
// (a long task, in the browser's terms).
const COSTLY_MS = 50;

// A GPU frame slower than this, twice running, hands the setting to the
// workers. The first frame of a setting also finds its segment budget, so one
// slow frame alone says little.
const GPU_SLOW_MS = 100;
const GPU_SLOW_FRAMES = 2;

// A ghost-lit frame that costs the CPU mirror more than this is not tried on
// the GPU at all. Every such gallery design measured was slower on the GPU or
// did not fit it (UV projection lens: 1.8 s on an RTX 5070 before it gave
// up), and the attempt holds the GPU all that time.
const GPU_TRY_MAX_GHOST_MS = 400;

// Just enough of the WebGPU types for this file, without a types dependency.
type DeviceRequest = { requiredLimits?: Record<string, number> } & Record<string, unknown>;
type Adapter = {
  limits: object;
  info?: { vendor?: string; architecture?: string };
  requestDevice(desc?: DeviceRequest): Promise<unknown>;
};
type AdapterPrototype = Adapter & { ocDropsRetiredLimits?: boolean };
type WebGpuNavigator = Navigator & { gpu?: { requestAdapter(): Promise<Adapter | null> } };

// wgpu 22 asks for every limit it knows, including `maxInterStageShaderComponents`,
// which WebGPU has removed. A browser must reject a device request that names a
// limit its adapter does not report, so current Chrome refuses wgpu's request
// outright. Drop those names before the request reaches the browser; a limit the
// adapter does report passes through unchanged. Retire this with the wgpu upgrade.
function dropRetiredLimits() {
  const scope = globalThis as { GPUAdapter?: { prototype: AdapterPrototype } };
  const proto = scope.GPUAdapter?.prototype;
  if (!proto || proto.ocDropsRetiredLimits) return;
  const requestDevice = proto.requestDevice;
  proto.requestDevice = function (this: Adapter, desc?: DeviceRequest) {
    const wanted = desc?.requiredLimits;
    if (desc && wanted) {
      const kept: Record<string, number> = {};
      for (const key of Object.keys(wanted)) {
        if (key in this.limits) kept[key] = wanted[key];
      }
      desc = { ...desc, requiredLimits: kept };
    }
    return requestDevice.call(this, desc);
  };
  proto.ocDropsRetiredLimits = true;
}

export function createFastPreview(): FastPreview {
  let gpu: GpuPreview3 | null = null;
  let adapterName = "";
  // Why there is no GPU, when there is none.
  let unavailable = "";
  let lastMs = NaN;
  let inFlight = false;
  let queued: Request | null = null;
  // The last frame was big enough that the GPU is the faster backend.
  let heavy = false;
  // Per ghost setting, since the last design loaded: why the GPU no longer
  // draws it ("declined": its ray trees do not fit; "slow": its frames took
  // longer than GPU_SLOW_MS), how many slow frames in a row it has drawn, and
  // whether its last frame was quick.
  const blank = <T>(v: T): Record<Setting, T> => ({ plain: v, ghosts: v });
  let refused = blank<"" | "declined" | "slow">("");
  let slowFrames = blank(0);
  let quick = blank(false);
  // The setting of the last scene asked about, for the status line.
  let lastSetting: Setting = "plain";

  // What a CPU frame last cost, with the ghost layer off and on.
  const cpuMs = blank(NaN);
  const useGpu = (s: Setting) =>
    gpu !== null && heavy && refused[s] === "" && !(s === "ghosts" && cpuMs.ghosts > GPU_TRY_MAX_GHOST_MS);
  let costlyFallback: () => void = () => {};

  function deliver(req: Request, frame: FastFrame) {
    const segments = frame.segs.length / STRIDE;
    if (segments >= GPU_FROM_SEGS) heavy = true;
    else if (segments < CPU_BELOW_SEGS) heavy = false;
    req.deliver(frame);
  }

  const cpuCostly = (scene: Canvas) => {
    const ms = cpuMs[settingOf(scene)];
    return Number.isNaN(ms) ? scene.ghostsEnabled() : ms > COSTLY_MS;
  };
  const cpuFrame = (scene: Canvas, late: boolean): FastFrame => {
    const t0 = performance.now();
    const segs = scene.traceProjected3DFast();
    cpuMs[settingOf(scene)] = performance.now() - t0;
    return { segs, revision: scene.sceneRevision(), late };
  };

  async function run(req: Request, device: GpuPreview3) {
    inFlight = true;
    const revision = req.scene.sceneRevision();
    const setting = settingOf(req.scene);
    const t0 = performance.now();
    try {
      const out = await device.traceProjected3D(req.scene);
      if (out instanceof Float32Array) {
        lastMs = performance.now() - t0;
        slowFrames[setting] = lastMs > GPU_SLOW_MS ? slowFrames[setting] + 1 : 0;
        quick[setting] = lastMs <= GPU_SLOW_MS;
        if (slowFrames[setting] >= GPU_SLOW_FRAMES) refused[setting] = "slow";
        deliver(req, { segs: out, revision, late: true });
      } else if (out === "truncated") {
        // Some ray tree does not fit on the GPU: draw the CPU picture, never a
        // thinner one, on this thread when it can afford it.
        refused[setting] = "declined";
        quick[setting] = false;
        cpuOrElsewhere(req, true);
      }
      // "busy" cannot happen: this module is the only caller, and it waits.
    } catch (e) {
      console.warn("WebGPU preview failed; staying on the CPU mirror.", e);
      gpu = null;
      unavailable = "GPU failed";
      cpuOrElsewhere(req, true);
    } finally {
      inFlight = false;
      const next = queued;
      queued = null;
      if (next) dispatch(next, true);
    }
  }

  function cpuOrElsewhere(req: Request, late: boolean) {
    if (cpuCostly(req.scene) && !req.anyCost) costlyFallback();
    else deliver(req, cpuFrame(req.scene, late));
  }

  function dispatch(req: Request, late: boolean) {
    if (useGpu(settingOf(req.scene))) void run(req, gpu!);
    else cpuOrElsewhere(req, late);
  }

  const ready = (async () => {
    const nav = navigator as WebGpuNavigator;
    if (!nav.gpu) {
      unavailable = "no WebGPU";
      return;
    }
    try {
      // Ask the browser first: calling into wasm without an adapter is wasted
      // work. The browser also names the GPU, which wgpu 22 cannot on the web.
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) {
        unavailable = "no GPU adapter";
        return;
      }
      adapterName = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(" ");
      dropRetiredLimits();
      gpu = await createGpuPreview3();
    } catch (e) {
      console.warn("WebGPU preview unavailable; using the CPU mirror.", e);
      unavailable = "GPU init failed";
    }
  })();

  return {
    ready,
    label: () => {
      if (!gpu) return unavailable ? `preview CPU (${unavailable})` : "preview CPU";
      if (refused[lastSetting] === "declined") return "preview CPU (ray trees too large for the GPU)";
      const deep = lastSetting === "ghosts" && cpuMs.ghosts > GPU_TRY_MAX_GHOST_MS;
      if (refused[lastSetting] === "slow" || deep) return "preview CPU (faster than the GPU for this scene)";
      const name = adapterName ? ` (${adapterName})` : "";
      if (!heavy) return `preview CPU (light scene; GPU${name} ready)`;
      const timing = Number.isNaN(lastMs) ? "" : ` ${lastMs.toFixed(1)} ms`;
      return `preview GPU${name}${timing}`;
    },
    requestProjected: (scene, deliverTo, anyCost = false) => {
      const req = { scene, deliver: deliverTo, anyCost };
      lastSetting = settingOf(scene);
      if (useGpu(lastSetting) && inFlight) {
        queued = req;
        return;
      }
      dispatch(req, false);
    },
    pending: () => inFlight || queued !== null,
    costly: (scene) => {
      const s = settingOf(scene);
      lastSetting = s;
      return cpuCostly(scene) && !(useGpu(s) && quick[s]);
    },
    whenCostly: (f) => {
      costlyFallback = f;
    },
    noteCpuCost: (ghosts, ms) => {
      cpuMs[ghosts ? "ghosts" : "plain"] = ms;
    },
    settled: () => {
      queued = null;
    },
    forget: () => {
      refused = blank("");
      slowFrames = blank(0);
      quick = blank(false);
      cpuMs.plain = NaN;
      cpuMs.ghosts = NaN;
    },
    gpu: () => gpu,
  };
}
