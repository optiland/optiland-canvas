// The interactive 3D picture (§4.1: f32 draws, f64 settles), traced on WebGPU or
// on the CPU f32 mirror, whichever is faster for the scene.
//
// Both views ask for the fast projected trace through here. The CPU mirror
// answers before `requestProjected` returns. The GPU answers later, so GPU
// requests are single-flight: while one runs, a newer request replaces any
// queued one, and the queued one runs next. Every frame carries the scene
// revision it was traced at, and the views decide whether it is still worth
// drawing.
//
// No displayed number comes from here, on either backend.

import { createGpuPreview3, type Canvas, type GpuPreview3 } from "../pkg/oc_wasm.js";

export type FastFrame = {
  segs: Float32Array<ArrayBufferLike>;
  /** The scene revision the frame was traced at. */
  revision: number;
  /** The frame arrived after `requestProjected` returned (the caller repaints). */
  late: boolean;
};

type Request = { scene: Canvas; deliver: (f: FastFrame) => void };

export type FastPreview = {
  /** Resolves once the GPU is ready or refused. */
  ready: Promise<void>;
  /** The backend that draws, and why, for the status line. */
  label: () => string;
  /** Ask for the projected fast trace of `scene` at its current revision. */
  requestProjected: (scene: Canvas, deliver: (f: FastFrame) => void) => void;
  /** A GPU trace is running or queued, so a newer frame is on its way. */
  pending: () => boolean;
  /** The exact trace just replaced the picture: drop the queued request, and
   *  give the GPU another chance if it declined the last scene. */
  settled: () => void;
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
  // The GPU could not hold the last scene's ray trees. Until the next settled
  // trace, go straight to the CPU instead of paying for a GPU attempt per frame.
  let declined = false;

  const useGpu = () => gpu !== null && heavy && !declined;

  function deliver(req: Request, frame: FastFrame) {
    const segments = frame.segs.length / STRIDE;
    if (segments >= GPU_FROM_SEGS) heavy = true;
    else if (segments < CPU_BELOW_SEGS) heavy = false;
    req.deliver(frame);
  }

  const cpuFrame = (scene: Canvas, late: boolean): FastFrame => ({
    segs: scene.traceProjected3DFast(),
    revision: scene.sceneRevision(),
    late,
  });

  async function run(req: Request, device: GpuPreview3) {
    inFlight = true;
    const revision = req.scene.sceneRevision();
    const t0 = performance.now();
    try {
      const out = await device.traceProjected3D(req.scene);
      if (out instanceof Float32Array) {
        lastMs = performance.now() - t0;
        deliver(req, { segs: out, revision, late: true });
      } else if (out === "truncated") {
        // Some ray tree does not fit on the GPU: draw the CPU picture, never a
        // thinner one.
        declined = true;
        deliver(req, cpuFrame(req.scene, true));
      }
      // "busy" cannot happen: this module is the only caller, and it waits.
    } catch (e) {
      console.warn("WebGPU preview failed; staying on the CPU mirror.", e);
      gpu = null;
      unavailable = "GPU failed";
      deliver(req, cpuFrame(req.scene, true));
    } finally {
      inFlight = false;
      const next = queued;
      queued = null;
      if (next) dispatch(next, true);
    }
  }

  function dispatch(req: Request, late: boolean) {
    if (useGpu()) void run(req, gpu!);
    else deliver(req, cpuFrame(req.scene, late));
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
      if (declined) return "preview CPU (ray trees too large for the GPU)";
      const name = adapterName ? ` (${adapterName})` : "";
      if (!heavy) return `preview CPU (light scene; GPU${name} ready)`;
      const timing = Number.isNaN(lastMs) ? "" : ` ${lastMs.toFixed(1)} ms`;
      return `preview GPU${name}${timing}`;
    },
    requestProjected: (scene, deliverTo) => {
      const req = { scene, deliver: deliverTo };
      if (useGpu() && inFlight) {
        queued = req;
        return;
      }
      dispatch(req, false);
    },
    pending: () => inFlight || queued !== null,
    settled: () => {
      queued = null;
      declined = false;
    },
    gpu: () => gpu,
  };
}
