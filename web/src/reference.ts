// The f64 reference, computed off the UI thread (spec §5, M1).
//
// The reference is the exact trace every displayed number comes from. It used
// to run on the page's one thread, several times per edit: once for the 2D
// picture, once for the detector dock, once more for the component view. On a
// 21-element lens that froze the page for a second after every edit, and with
// ghosts on it froze it for minutes.
//
// Now a pool of Web Workers shares it (reference-worker.ts). Each worker loads
// a copy of the scene. The trace is cut into a fixed number of blocks of
// launch rays; this module hands the blocks out to whichever worker is free,
// collects each block's share of the picture and its numbers, and has one
// worker add the numbers up in block order. The blocks do not depend on the
// number of workers, so the reference is the same, to the bit, on a machine
// with one core or sixteen, and the same one tracing it whole gives
// (`Canvas.referenceStep`, `oc_trace::TraceJob3`).
//
// An edit that arrives meanwhile supersedes the trace in progress: the
// workers drop it and start on the new scene. A finished trace is one bundle,
// tagged with the scene revision and trace settings it was traced for, and
// everything that shows a reference number reads that bundle:
//
//   - `current()` is the bundle for the scene as it is now, or null;
//   - `latest()` is the newest bundle, which may be for an older scene and is
//     then shown as stale, never as the value of the edited scene;
//   - `status()` says which: updating (and how far), current, or stale.
//
// While a trace is in progress the picture is the fast preview, and the
// numbers are the last reference's, marked stale. Where the fast preview would
// stall the page (a ghost-lit lens on the CPU mirror), the workers trace that
// stand-in picture too, a share of its rays each before their first block, and
// the page draws it when every share has landed (`standIn`). One stand-in is
// traced at a time: while a lens is dragged, the next position waits for the
// one in flight rather than superseding it, so the rays follow the drag a few
// times a second instead of not at all.

import type { Canvas } from "../pkg/oc_wasm.js";

/** What a reference trace is of: the scene's revision and the trace settings
 *  that move no revision (the ghost layer). */
export type ReferenceKey = { revision: number; ghosts: boolean; maxGhost: number };

/** The reference's numbers, as the page reads them. */
export type ReferenceNumbers = {
  /** `referenceMapsF32`: each area detector's flux map. */
  maps: Float32Array;
  /** `detectorResultJSON`, by detector id. */
  readouts: Map<number, string>;
  /** `detectorHitsF32`, by detector id. */
  hits: Map<number, Float32Array>;
  /** The scene's first detector, the dock's default. */
  firstDetector: number | null;
};

/** Everything the page reads from one reference trace. */
export type ReferenceBundle = ReferenceNumbers & {
  key: ReferenceKey;
  /** `traceProjected3D`: every segment on the default view plane. */
  projected: Float32Array;
  /** Launch rays traced, and the wall time the trace took, in ms. */
  rays: number;
  elapsedMs: number;
};

export type ReferenceStatus =
  | { state: "none" }
  | { state: "updating"; fraction: number; since: number }
  | { state: "current"; bundle: ReferenceBundle }
  | { state: "stale"; bundle: ReferenceBundle; fraction: number };

export type Reference = {
  /** The bundle for the scene as it is now, or null while it is traced. */
  current: () => ReferenceBundle | null;
  /** The newest bundle, possibly of an older scene. */
  latest: () => ReferenceBundle | null;
  /** Ask for the scene as it is now. A no-op when that is current or already
   *  being traced; otherwise it supersedes any trace in progress. */
  request: () => void;
  status: () => ReferenceStatus;
  /** The picture to show until the reference lands, given the stand-in on
   *  screen (null when the picture is not one): the workers' newest frame
   *  when it is not the one on screen (the same f32 trace, in the
   *  `traceProjected3D` layout, of this scene or of an earlier position of a
   *  drag); "coming" while one is on its way, and the picture on screen
   *  holds; otherwise "here", and the page traces its own fast frame. */
  standIn: (shown: StandIn | null) => "here" | "coming" | StandIn;
  /** Have the workers trace the stand-in of the scene as it is, when this
   *  thread found it cannot afford the frame after all (the GPU declined
   *  it, or a drag asked for one). Asks for the reference too, if needed. */
  requestStandIn: () => void;
  /** How many workers trace the reference (0: this thread does). */
  workers: () => number;
};

export const sameKey = (a: ReferenceKey, b: ReferenceKey) =>
  a.revision === b.revision && a.ghosts === b.ghosts && a.maxGhost === b.maxGhost;

/** The same trace settings, at any revision. */
const sameSettings = (a: ReferenceKey, b: ReferenceKey) => a.ghosts === b.ghosts && a.maxGhost === b.maxGhost;

/** A stand-in picture and the scene revision it was traced at. */
export type StandIn = { segs: Float32Array; revision: number };

/** Pictures one after another, as one. */
export function concat(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** Read the numbers of a `Canvas` whose reference is in: traced whole, or
 *  added up from a pool's blocks. */
export function readNumbers(c: Canvas): ReferenceNumbers {
  const maps = c.referenceMapsF32();
  const readouts = new Map<number, string>();
  const hits = new Map<number, Float32Array>();
  const ids = Array.from(c.detectorIds());
  for (const id of ids) {
    readouts.set(id, c.detectorResultJSON(id));
    hits.set(id, c.detectorHitsF32(id));
  }
  return { maps, readouts, hits, firstDetector: ids.length ? ids[0] : null };
}

/** Read a reference traced whole on `c` (the fallback on the page's thread). */
export function readBundle(c: Canvas, key: ReferenceKey, rays: number, elapsedMs: number): ReferenceBundle {
  return { key, projected: c.traceProjected3D(), ...readNumbers(c), rays, elapsedMs };
}

/** The transferable buffers of a reference's numbers, so posting them copies
 *  nothing. */
export function numbersBuffers(n: ReferenceNumbers): ArrayBuffer[] {
  return [n.maps.buffer, ...Array.from(n.hits.values(), (h) => h.buffer)] as ArrayBuffer[];
}

// --- the protocol --------------------------------------------------------

/** A block's numbers, as `Canvas.poolNumbers` and `poolReport` give them. */
export type EncodedPart = { numbers: Float64Array; report: string };

export type ToWorker =
  | { type: "load"; key: ReferenceKey; scene: string; standIn: { share: number; shares: number } | null }
  | { type: "standIn"; key: ReferenceKey; share: number; shares: number }
  | { type: "block"; key: ReferenceKey; block: number; records: boolean }
  | { type: "merge"; key: ReferenceKey; parts: EncodedPart[]; pictures: Float32Array[] };
export type FromWorker =
  | { type: "ready" }
  | { type: "standIn"; key: ReferenceKey; share: number; segs: Float32Array; ms: number }
  | { type: "loaded"; key: ReferenceKey; blocks: number; rays: number }
  | { type: "progress"; key: ReferenceKey; block: number; fraction: number }
  | {
      type: "part";
      key: ReferenceKey;
      block: number;
      picture: Float32Array;
      numbers: Float64Array;
      report: string;
      recordsFull: boolean;
    }
  | { type: "merged"; key: ReferenceKey; numbers: ReferenceNumbers; projected: Float32Array }
  | { type: "error"; key: ReferenceKey | null; message: string };

// Enough workers to use the machine and leave a core to the page, and no more
// than the memory of a heavy scene's blocks in flight justifies: a worker
// holding a block of a ghost-lit twenty-element lens is about 50 MB, so a
// device reporting little memory (`navigator.deviceMemory`, in GB) gets fewer.
const MAX_WORKERS = 8;
const workersFor = (cores: number, memoryGb: number) =>
  Math.max(1, Math.min(MAX_WORKERS, cores - 1, Math.floor(memoryGb) - 1));
// Progress reaches the status line at most this often, in ms: a pool of
// workers reports every few milliseconds each.
const STATUS_MS = 100;

/** One reference being traced by the pool. */
type PoolTrace = {
  key: ReferenceKey;
  t0: number;
  /** The number of blocks and of launch rays, once a worker has said. */
  blocks: number;
  rays: number;
  /** The next block to hand out. */
  next: number;
  /** Each block's picture and numbers, as they come in. */
  parts: ({ picture: Float32Array; encoded: EncodedPart } | null)[];
  /** How far each block has got. */
  fractions: number[];
  /** Blocks that filled every detector's hit records by themselves. */
  full: number[];
  merging: boolean;
};

/** Start the reference. `keyOf` reads the scene's current key; `changed` is
 *  called when a bundle, the stand-in or the status changes; `costly` says
 *  whether the page's own fast frame of the scene would stall it, and so
 *  whether the workers trace the stand-in; `measured` hears what a stand-in
 *  cost the workers in all, which is what the frame would cost this thread.
 *  Falls back to tracing on this thread, as before, when the browser has no
 *  module workers. */
export function createReference(
  scene: Canvas,
  keyOf: () => ReferenceKey,
  changed: (why: "result" | "standIn" | "status") => void,
  costly: () => boolean,
  measured: (ghosts: boolean, ms: number) => void = () => {},
): Reference {
  let latest: ReferenceBundle | null = null;
  let asked: ReferenceKey | null = null;
  // The stand-in the workers are tracing and the shares of it they have sent;
  // the newest one to land; and whether a newer scene waits for the one in
  // flight before it is asked for.
  let flight: { key: ReferenceKey; shares: (Float32Array | null)[]; ms: number } | null = null;
  let standIn: { key: ReferenceKey; frame: StandIn } | null = null;
  let held = false;
  let fraction = 0;
  let since = 0;
  let statusAt = 0;
  let trace: PoolTrace | null = null;

  let workers: Worker[] = [];
  const nav = navigator as Navigator & { deviceMemory?: number };
  const size = workersFor(nav.hardwareConcurrency || 2, nav.deviceMemory ?? 8);
  try {
    for (let i = 0; i < size; i++) {
      workers.push(new Worker(new URL("./reference-worker.ts", import.meta.url), { type: "module" }));
    }
  } catch {
    stopPool();
  }

  const current = () => (latest && sameKey(latest.key, keyOf()) ? latest : null);

  function accept(bundle: ReferenceBundle) {
    // A late result for a scene older than the one already shown is dropped:
    // it must never overwrite a newer reference.
    if (latest && bundle.key.revision < latest.key.revision) return;
    latest = bundle;
    if (asked && sameKey(asked, bundle.key)) asked = null;
    changed("result");
  }

  function stopPool() {
    for (const w of workers) w.terminate();
    workers = [];
    trace = null;
    flight = null;
    held = false;
  }

  // A worker failed: serve this request on the page's thread instead, so a
  // number still arrives. A worker that died takes the pool with it.
  function fallBack(why: string, err: unknown, fatal: boolean) {
    console.error(`reference ${why}; tracing on the page's thread`, err);
    trace = null;
    flight = null;
    held = false;
    if (fatal) stopPool();
    if (asked) inThread(asked);
  }

  const live = (key: ReferenceKey) => trace !== null && sameKey(trace.key, key);

  function handOut(t: PoolTrace, w: Worker) {
    if (t.next >= t.blocks) return;
    const block = t.next++;
    // A block after one that filled every detector's hit records keeps none:
    // the merge keeps only the first records, in block order.
    const records = !t.full.some((f) => f < block);
    const msg: ToWorker = { type: "block", key: t.key, block, records };
    w.postMessage(msg);
  }

  function progress(t: PoolTrace) {
    fraction = t.blocks ? t.fractions.reduce((a, b) => a + b, 0) / t.blocks : 0;
    const now = performance.now();
    if (now - statusAt >= STATUS_MS) {
      statusAt = now;
      changed("status");
    }
  }

  function finishTrace(t: PoolTrace, numbers: ReferenceNumbers, projected: Float32Array) {
    trace = null;
    accept({ key: t.key, projected, ...numbers, rays: t.rays, elapsedMs: performance.now() - t.t0 });
  }

  function onMessage(w: Worker, m: FromWorker) {
    if (m.type === "ready") return;
    if (m.type === "error") {
      if (m.key && !live(m.key)) return;
      fallBack("worker", m.message, false);
      return;
    }
    if (m.type === "standIn") {
      if (!flight || !sameKey(flight.key, m.key)) return;
      flight.shares[m.share] = m.segs;
      flight.ms += m.ms;
      if (flight.shares.every((x) => x !== null)) {
        measured(m.key.ghosts, flight.ms);
        const segs = concat(flight.shares as Float32Array[]);
        standIn = { key: m.key, frame: { segs, revision: m.key.revision } };
        flight = null;
        changed("standIn");
        // The scene that waited for this one is asked for now.
        if (held) request();
      }
      return;
    }
    if (!live(m.key)) return;
    const t = trace!;
    if (m.type === "loaded") {
      if (m.blocks === 0) {
        // No 3D scene: nothing to trace, and nothing to read.
        const none = { maps: new Float32Array(0), readouts: new Map(), hits: new Map(), firstDetector: null };
        finishTrace(t, none, new Float32Array(0));
        return;
      }
      if (!t.blocks) {
        t.blocks = m.blocks;
        t.rays = m.rays;
        t.parts = new Array(m.blocks).fill(null);
        t.fractions = new Array(m.blocks).fill(0);
      }
      handOut(t, w);
    } else if (m.type === "progress") {
      t.fractions[m.block] = m.fraction;
      progress(t);
    } else if (m.type === "part") {
      t.parts[m.block] = { picture: m.picture, encoded: { numbers: m.numbers, report: m.report } };
      t.fractions[m.block] = 1;
      if (m.recordsFull) t.full.push(m.block);
      progress(t);
      handOut(t, w);
      if (!t.merging && t.parts.every((p) => p !== null)) {
        // One worker adds the numbers up and puts the picture together: a
        // ghost-lit lens's picture is hundreds of megabytes to copy.
        t.merging = true;
        const parts = t.parts.map((p) => p!.encoded);
        const pictures = t.parts.map((p) => p!.picture);
        const msg: ToWorker = { type: "merge", key: t.key, parts, pictures };
        const buffers = [...parts.map((p) => p.numbers.buffer), ...pictures.map((p) => p.buffer)];
        workers[0].postMessage(msg, buffers as ArrayBuffer[]);
      }
    } else if (m.type === "merged") {
      finishTrace(t, m.numbers, m.projected);
    }
  }

  for (const w of workers) {
    w.onmessage = (e: MessageEvent<FromWorker>) => onMessage(w, e.data);
    w.onerror = (e) => fallBack("worker failed", e, true);
  }

  // The fallback: the page's own Canvas traces (once per key: it caches), after
  // this frame has painted.
  function inThread(key: ReferenceKey) {
    setTimeout(() => {
      if (!sameKey(key, keyOf())) return;
      const t0 = performance.now();
      const rays = scene.referenceBegin();
      // One step traces every ray; short of 1 there is no scene to trace.
      if (scene.referenceStep(0x7fffffff) < 1) return;
      accept(readBundle(scene, key, rays, performance.now() - t0));
    }, 0);
  }

  function request() {
    const key = keyOf();
    if (current() || (asked && sameKey(asked, key))) return;
    // A stand-in of an earlier position of this scene is on its way: this
    // one waits for it, or no stand-in would land while a drag goes on.
    if (flight && sameSettings(flight.key, key)) {
      held = true;
      return;
    }
    held = false;
    asked = key;
    fraction = 0;
    since = performance.now();
    flight = workers.length > 0 && costly() ? { key, shares: new Array(workers.length).fill(null), ms: 0 } : null;
    if (workers.length) {
      trace = {
        key,
        t0: since,
        blocks: 0,
        rays: 0,
        next: 0,
        parts: [],
        fractions: [],
        full: [],
        merging: false,
      };
      const json = scene.scene3JSON();
      workers.forEach((w, i) => {
        const share = flight ? { share: i, shares: workers.length } : null;
        const msg: ToWorker = { type: "load", key, scene: json, standIn: share };
        w.postMessage(msg);
      });
    } else {
      inThread(key);
    }
    changed("status");
  }

  function status(): ReferenceStatus {
    const now = current();
    if (now) return { state: "current", bundle: now };
    if (latest) return { state: "stale", bundle: latest, fraction };
    if (asked) return { state: "updating", fraction, since };
    return { state: "none" };
  }

  function requestStandIn() {
    if (!workers.length || current()) return;
    const key = keyOf();
    // Asked for with a stand-in now that the page cannot afford one, or held
    // behind the one in flight.
    if (!asked || !sameKey(asked, key)) {
      request();
      return;
    }
    // Asked for without one (the GPU was to draw it, and declined): add it.
    if (flight || (standIn && sameKey(standIn.key, key))) return;
    flight = { key, shares: new Array(workers.length).fill(null), ms: 0 };
    workers.forEach((w, i) => {
      const msg: ToWorker = { type: "standIn", key, share: i, shares: workers.length };
      w.postMessage(msg);
    });
  }

  function standInFor(shown: StandIn | null): "here" | "coming" | StandIn {
    if (!workers.length) return "here";
    const key = keyOf();
    const newest = standIn && sameSettings(standIn.key, key) && standIn.key.revision <= key.revision ? standIn : null;
    if (newest && newest.frame !== shown) return newest.frame;
    // This scene's stand-in is on screen, or one is on its way: hold.
    if (newest && newest.key.revision === key.revision) return "coming";
    if (flight && sameSettings(flight.key, key)) return "coming";
    return "here";
  }

  return {
    current,
    latest: () => latest,
    request,
    status,
    standIn: standInFor,
    requestStandIn,
    workers: () => workers.length,
  };
}
