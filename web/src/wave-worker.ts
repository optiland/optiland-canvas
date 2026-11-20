// The wave path's worker: one wasm canvas, one job at a time, an event per
// step. A newer request drops the job in hand at the next step boundary.

import init, { Canvas } from "../pkg/oc_wasm.js";
import type { FromWaveWorker, ToWaveWorker } from "./wave";

const scope = self as unknown as {
  postMessage: (m: FromWaveWorker, transfer?: Transferable[]) => void;
  onmessage: ((e: MessageEvent<ToWaveWorker>) => void) | null;
};

let canvas: Canvas | null = null;
let pending: ToWaveWorker | null = null;
let running = false;

const ready = init().then(() => {
  canvas = new Canvas();
  scope.postMessage({ type: "ready" });
});

// Yield through a task boundary, so a message posted meanwhile is delivered.
const channel = new MessageChannel();
const yieldTask = () =>
  new Promise<void>((resolve) => {
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(null);
  });

scope.onmessage = (e) => {
  pending = e.data;
  if (!running) void run();
};

async function run() {
  running = true;
  await ready;
  while (pending) {
    const job = pending;
    pending = null;
    await doRun(job);
  }
  running = false;
}

async function doRun(job: ToWaveWorker) {
  const c = canvas!;
  const t0 = performance.now();
  const fail = (message: string) => scope.postMessage({ type: "error", key: job.key, message });
  try {
    c.loadScene3JSON(job.scene);
  } catch (e) {
    fail(`the scene did not load: ${e}`);
    return;
  }
  const begin = JSON.parse(c.waveBegin(job.key.source, job.key.detector, job.key.n, job.key.windowMm));
  if (begin.error) {
    fail(begin.error);
    return;
  }
  for (;;) {
    const s = JSON.parse(c.waveStep(1));
    if (s.error) {
      fail(s.error);
      return;
    }
    if (s.done) break;
    scope.postMessage({ type: "progress", key: job.key, fraction: s.fraction });
    await yieldTask();
    if (pending) return; // superseded: the next request takes over
  }
  const power = c.wavePower();
  const intensity = c.waveFieldIntensity();
  const phase = c.wavePhase();
  scope.postMessage(
    { type: "result", key: job.key, result: c.waveResultJSON(), power, intensity, phase, ms: performance.now() - t0 },
    [power.buffer, intensity.buffer, phase.buffer],
  );
}
