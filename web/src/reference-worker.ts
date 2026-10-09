// An f64 reference worker (see reference.ts). A pool of these shares each
// reference trace.
//
// Each worker owns its own Canvas and loads a copy of the scene with each
// request. The trace is cut into blocks of launch rays (`Canvas.poolPrepare`).
// The page hands a worker one block at a time; the worker traces it a batch
// of rays at a time, yielding to its event loop between batches so a newer
// scene is seen within one batch, and posts back the block's share of the
// picture and its numbers. Once every block is in, the page sends the numbers
// to one worker, which adds them up in block order (`Canvas.poolMerge`) and
// posts the readouts. The newest scene wins: work for a scene it supersedes
// is dropped, not finished.
//
// Batches are sized to take about a slice (SLICE_MS) each, from how long the
// last one took, so a light scene is traced in a few slices and a heavy one
// still yields every few milliseconds.
//
// When the page asks for it, each worker first traces its share of the fast
// f32 picture the page would otherwise trace on its own thread, and posts it
// for the stand-in. Asked for one later, a worker traces its share between two
// batches of the block in hand.

import init, { Canvas } from "../pkg/oc_wasm.js";
import {
  concat,
  numbersBuffers,
  readNumbers,
  sameKey,
  type FromWorker,
  type ReferenceKey,
  type ToWorker,
} from "./reference";

const SLICE_MS = 12;

const scope = self as unknown as {
  postMessage: (m: FromWorker, transfer?: Transferable[]) => void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

type Load = Extract<ToWorker, { type: "load" }>;
type Task = Exclude<ToWorker, Load>;

let canvas: Canvas | null = null;
// The scene the canvas holds, and a newer one waiting to be loaded.
let loaded: ReferenceKey | null = null;
let load: Load | null = null;
const tasks: Task[] = [];
let running = false;
// Launch rays per batch, adapted to the scene as it is traced.
let batch = 1;

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
  const m = e.data;
  if (m.type === "load") {
    // Everything queued was for the older scene.
    load = m;
    tasks.length = 0;
  } else {
    tasks.push(m);
  }
  if (!running) void run();
};

async function run() {
  running = true;
  try {
    await ready;
    for (;;) {
      if (load) {
        const l = load;
        load = null;
        doLoad(l);
        continue;
      }
      const t = tasks.shift();
      if (!t) break;
      if (!loaded || !sameKey(t.key, loaded)) continue;
      if (t.type === "block") await doBlock(t.key, t.block, t.records);
      else if (t.type === "standIn") doStandIn(t.key, t.share, t.shares);
      else doMerge(t.key, t.parts, t.pictures);
    }
  } finally {
    running = false;
  }
}

function fail(key: ReferenceKey | null, err: unknown) {
  scope.postMessage({ type: "error", key, message: String(err) });
}

function doLoad(l: Load) {
  const c = canvas!;
  try {
    c.loadScene3JSON(l.scene);
    c.setGhosts(l.key.ghosts);
    c.setMaxGhostBounces(l.key.maxGhost);
    c.setAnalysisRays(l.key.rays);
    loaded = l.key;
    batch = 1;
    if (l.standIn) doStandIn(l.key, l.standIn.share, l.standIn.shares);
    const blocks = c.poolPrepare();
    scope.postMessage({ type: "loaded", key: l.key, blocks, rays: c.poolRays() });
  } catch (err) {
    loaded = null;
    fail(l.key, err);
  }
}

function doStandIn(key: ReferenceKey, share: number, shares: number) {
  try {
    const t0 = performance.now();
    const segs = canvas!.traceProjected3DFastShare(share, shares);
    const ms = performance.now() - t0;
    scope.postMessage({ type: "standIn", key, share, segs, ms }, [segs.buffer as ArrayBuffer]);
  } catch (err) {
    fail(key, err);
  }
}

// A stand-in asked for while a block is traced goes first: the page is
// waiting on it to draw anything.
function standInsFirst() {
  for (let i = 0; i < tasks.length; ) {
    const t = tasks[i];
    if (t.type !== "standIn") {
      i++;
      continue;
    }
    tasks.splice(i, 1);
    if (loaded && sameKey(t.key, loaded)) doStandIn(t.key, t.share, t.shares);
  }
}

async function doBlock(key: ReferenceKey, block: number, records: boolean) {
  const c = canvas!;
  try {
    c.poolBegin(block, records);
    let fraction = 0;
    while (fraction < 1) {
      const s = performance.now();
      const now = c.poolStep(batch);
      if (now <= fraction) throw new Error(`block ${block} made no progress`);
      fraction = now;
      const dt = performance.now() - s;
      batch = Math.max(1, Math.min(4096, Math.round((batch * SLICE_MS) / Math.max(dt, 0.05))));
      scope.postMessage({ type: "progress", key, block, fraction });
      await yieldTask();
      // A newer scene supersedes this one: drop the block.
      if (load) return;
      standInsFirst();
    }
    const picture = c.poolPicture();
    const numbers = c.poolNumbers();
    const report = c.poolReport();
    const recordsFull = c.poolRecordsFull();
    scope.postMessage({ type: "part", key, block, picture, numbers, report, recordsFull }, [
      picture.buffer as ArrayBuffer,
      numbers.buffer as ArrayBuffer,
    ]);
  } catch (err) {
    fail(key, err);
  }
}

function doMerge(key: ReferenceKey, parts: { numbers: Float64Array; report: string }[], pictures: Float32Array[]) {
  const c = canvas!;
  try {
    let done = false;
    for (const p of parts) done = c.poolMerge(p.numbers, p.report);
    if (!done) throw new Error("the merge is short of blocks");
    const numbers = readNumbers(c);
    // The blocks' pictures in block order are the reference's picture.
    const projected = concat(pictures);
    scope.postMessage({ type: "merged", key, numbers, projected }, [
      ...numbersBuffers(numbers),
      projected.buffer as ArrayBuffer,
    ]);
  } catch (err) {
    fail(key, err);
  }
}
