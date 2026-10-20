// The coherent wave path, run off the page thread (W5).
//
// The page asks for the wave path of the scene as it is: from the first
// source that declares a coherent beam to a rectangular detector, the field
// sampled `n` across a window. A worker loads the scene, begins the job
// (`waveBegin`) and carries it an event at a time (`waveStep`), yielding
// between events so a newer request is seen at once and the stale job
// dropped. A result carries the key it was computed for; one for an older
// revision is kept only as "the previous scene's", never as current. The
// numbers are the f64 wave engine's (oc_wave::path); nothing here computes
// any.

export type WaveKey = { revision: number; source: number; detector: number; n: number; windowMm: number };

export function sameWaveKey(a: WaveKey | null, b: WaveKey | null): boolean {
  return !!a && !!b && a.revision === b.revision && a.source === b.source && a.detector === b.detector
    && a.n === b.n && a.windowMm === b.windowMm;
}

/** `waveResultJSON`. */
export type WaveResult = {
  revision: number;
  detector: number;
  bins: [number, number];
  power: number;
  field: { n: [number, number]; pitchMm: [number, number]; lambdaUm: number };
  report: {
    warnings: unknown[];
    validity: unknown[];
    approximated: string[];
    rejected: string[];
    /** What the ledger booked that a passive model cannot; empty for a sound result. */
    inconsistencies?: string[];
    steps: { method: unknown; z_mm: number; note: string }[];
    ledger: Record<string, number>;
  };
};

export type WaveBundle = {
  key: WaveKey;
  result: WaveResult;
  /** Power per detector pixel, row-major, u fastest. */
  power: Float64Array;
  /** |U|² and the phase of the arriving field, across the beam. */
  intensity: Float64Array;
  phase: Float64Array;
  ms: number;
};

export type ToWaveWorker = { type: "run"; key: WaveKey; scene: string };
export type FromWaveWorker =
  | { type: "ready" }
  | { type: "progress"; key: WaveKey; fraction: number }
  | { type: "result"; key: WaveKey; result: string; power: Float64Array; intensity: Float64Array; phase: Float64Array; ms: number }
  | { type: "error"; key: WaveKey; message: string };

export type WaveStatus =
  | { state: "idle" }
  | { state: "running"; key: WaveKey; fraction: number }
  | { state: "current"; bundle: WaveBundle }
  | { state: "error"; key: WaveKey; message: string };

export function createWave(changed: () => void) {
  const worker = new Worker(new URL("./wave-worker.ts", import.meta.url), { type: "module" });
  let asked: WaveKey | null = null;
  let fraction = 0;
  let latest: WaveBundle | null = null;
  let failure: { key: WaveKey; message: string } | null = null;

  worker.onmessage = (e: MessageEvent<FromWaveWorker>) => {
    const m = e.data;
    if (m.type === "ready") return;
    if (!sameWaveKey(m.key, asked)) return; // superseded
    if (m.type === "progress") {
      fraction = m.fraction;
    } else if (m.type === "result") {
      latest = { key: m.key, result: JSON.parse(m.result), power: m.power, intensity: m.intensity, phase: m.phase, ms: m.ms };
      failure = null;
    } else if (m.type === "error") {
      failure = { key: m.key, message: m.message };
    }
    changed();
  };

  return {
    /** Ask for `key`'s wave path; a no-op when it is already asked for. */
    request(key: WaveKey, scene: () => string) {
      if (sameWaveKey(key, asked)) return;
      asked = key;
      fraction = 0;
      failure = null;
      worker.postMessage({ type: "run", key, scene: scene() } satisfies ToWaveWorker);
    },
    /** Forget what was asked for (no coherent source, no detector). */
    clear() {
      asked = null;
    },
    status(): WaveStatus {
      if (!asked) return { state: "idle" };
      if (failure && sameWaveKey(failure.key, asked)) return { state: "error", key: failure.key, message: failure.message };
      if (latest && sameWaveKey(latest.key, asked)) return { state: "current", bundle: latest };
      return { state: "running", key: asked, fraction };
    },
    /** The newest result there is, current or not. */
    latest: () => latest,
  };
}

export type Wave = ReturnType<typeof createWave>;
