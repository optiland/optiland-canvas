// The WebGPU preview against the CPU mirror, in a browser that has a device.
//
// Opt-in, because headless CI has no GPU:
//   OC_WEBGPU=1 npx playwright test --project=webgpu
// Add OC_WEBGPU_CHANNEL=chrome to use the installed Chrome, and OC_WEBGPU_HEADED=1
// to watch it. The shader's own parity guard is the native gpu3_vs_cpu3 suite;
// this proves the browser wiring returns the same picture and times both paths.

import { test, expect, type Page } from "@playwright/test";

type Comparison = {
  ok: boolean;
  reason: string;
  label: string;
  cpuLen: number;
  gpuLen: number;
  worstStart: number;
  worstEnd: number;
  flagMismatches: number;
  cpuMs: number;
  gpuMs: number;
};

// Compare the GPU projected frame with the CPU one, segment by segment, with the
// §15.5 landing tolerance. Escape endpoints sit far away where f32 spreads, so
// only segments shorter than 100 mm have their end compared (as natively).
async function compare(page: Page, runs: number): Promise<Comparison> {
  return page.evaluate(async (runs) => {
    const d = (window as unknown as { __ocDebug: any }).__ocDebug;
    await d.ready;
    const empty = { cpuLen: 0, gpuLen: 0, worstStart: 0, worstEnd: 0, flagMismatches: 0, cpuMs: 0, gpuMs: 0 };
    if (!d.fast.gpu()) {
      return { ok: false, reason: `no GPU backend: ${d.fast.label()}`, label: d.fast.label(), ...empty };
    }

    // Warm both paths, then time the median of `runs`.
    for (let i = 0; i < 2; i++) { d.cpuProjected(); await d.gpuProjected(); }
    const cpuTimes: number[] = [], gpuTimes: number[] = [];
    let cpu = d.cpuProjected();
    let gpu: unknown = null;
    for (let i = 0; i < runs; i++) {
      let t0 = performance.now();
      cpu = d.cpuProjected();
      cpuTimes.push(performance.now() - t0);
      t0 = performance.now();
      gpu = await d.gpuProjected();
      gpuTimes.push(performance.now() - t0);
    }
    const median = (xs: number[]) => xs.sort((a, b) => a - b)[Math.floor(xs.length / 2)];
    const timing = { cpuMs: median(cpuTimes), gpuMs: median(gpuTimes) };
    if (!(gpu instanceof Float32Array)) {
      return { ok: false, reason: `GPU returned ${String(gpu)}`, label: d.fast.label(), ...empty, ...timing };
    }

    const S = 11;
    let worstStart = 0, worstEnd = 0, flagMismatches = 0;
    const n = Math.min(cpu.length, gpu.length);
    for (let i = 0; i < n; i += S) {
      const ds = Math.hypot(cpu[i] - gpu[i], cpu[i + 1] - gpu[i + 1], cpu[i + 9] - gpu[i + 9]);
      worstStart = Math.max(worstStart, ds);
      const len = Math.hypot(cpu[i + 2] - cpu[i], cpu[i + 3] - cpu[i + 1], cpu[i + 10] - cpu[i + 9]);
      if (len < 100) {
        const de = Math.hypot(cpu[i + 2] - gpu[i + 2], cpu[i + 3] - gpu[i + 3], cpu[i + 10] - gpu[i + 10]);
        worstEnd = Math.max(worstEnd, de);
      }
      if (cpu[i + 8] !== gpu[i + 8]) flagMismatches++;
    }
    return {
      ok: true, reason: "", label: d.fast.label(),
      cpuLen: cpu.length, gpuLen: gpu.length, worstStart, worstEnd, flagMismatches, ...timing,
    };
  }, runs);
}

function expectParity(r: Comparison) {
  expect(r.ok, r.reason).toBe(true);
  expect(r.gpuLen, "segment count").toBe(r.cpuLen);
  expect(r.flagMismatches, "TIR/ghost flags").toBe(0);
  // max(1e-2 mm, 1e-5 x scene scale); these scenes are well under 1 m.
  expect(r.worstStart).toBeLessThan(1e-2);
  expect(r.worstEnd).toBeLessThan(1e-2);
}

test.describe("WebGPU preview", () => {
  // The app logs why it refused a device; show that next to the failure.
  test.beforeEach(async ({ page }) => {
    page.on("console", (m) => {
      if (m.type() === "warning" || m.type() === "error") console.log(`[browser ${m.type()}] ${m.text()}`);
    });
  });

  test("the GPU picture matches the CPU mirror on the default scene", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    const r = await compare(page, 9);
    console.log(`default scene: ${r.cpuLen / 11} segs, CPU ${r.cpuMs.toFixed(2)} ms, GPU ${r.gpuMs.toFixed(2)} ms (${r.label})`);
    expectParity(r);
    // A light scene stays on the CPU mirror, with the GPU standing by.
    await expect(page.locator("#stat")).toContainText(/light scene; GPU.* ready/);
  });

  // A splitter and a dichroic fold parts of the beam up. The fold is a primary
  // path, so it must be in the picture with ghosts off, and the dichroic's
  // tables are sampled per wavelength on both sides of and inside its edge.
  test("the GPU picture matches the CPU mirror through a splitter and a dichroic", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.evaluate(() => {
      const s = (window as unknown as { __ocDebug: any }).__ocDebug.scene;
      s.setGhosts(false);
      s.setSourceWavelengths(new Float64Array([450, 546, 553, 650]), new Float64Array(0));
      s.addBeamSplitter(-50, 0, 15, 135, 0.4, 0.6);
      s.addDichroic(-35, 0, 15, 135, 550, 20, true);
      s.addAreaDetector(-50, 40, 20, 90);
      s.addAreaDetector(-35, 40, 20, 90);
    });
    const r = await compare(page, 5);
    console.log(`splitter + dichroic: ${r.cpuLen / 11} segs (${r.label})`);
    expectParity(r);
    const folds = await page.evaluate(async () => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      const g = (await d.gpuProjected()) as Float32Array;
      const at = new Set<number>();
      for (let i = 0; i + 11 <= g.length; i += 11) {
        // A segment rising straight up from one of the plates.
        if (g[i + 3] - g[i + 1] > 5 && Math.abs(g[i + 2] - g[i]) < 0.01) at.add(Math.round(g[i] + 50 - g[i + 1]) === 0 ? -50 : -35);
      }
      return [...at].sort((a, b) => a - b);
    });
    expect(folds).toEqual([-50, -35]);
  });

  // Lenses one to seven with ghosts on (the case that motivated the GPU path),
  // then seven lenses ghost-free at the densest source. Parity at every step; the
  // timings are logged, not asserted, because they belong to this machine.
  test("scaling sweep: parity at every step, timings logged", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    const lensId = await page.evaluate(() => {
      const s = (window as unknown as { __ocDebug: any }).__ocDebug.scene;
      const objs = JSON.parse(s.sceneObjectsJSON()) as { id: number; kind: string }[];
      const det = objs.find((o) => o.kind === "detector")!;
      s.setObjectPosition(det.id, 200, 0, 0);
      s.setGhosts(true);
      s.setMaxGhostBounces(2);
      return objs.find((o) => o.kind === "profile")!.id;
    });
    const rows: string[] = [];
    const measure = async (name: string) => {
      const r = await compare(page, 5);
      const budget = await page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.fast.gpu()?.segmentBudget());
      rows.push(`${name}: ${r.cpuLen / 11} segs, CPU ${r.cpuMs.toFixed(1)} ms, GPU ${r.gpuMs.toFixed(1)} ms, budget ${budget}${r.reason ? ` (${r.reason})` : ""}`);
      return r;
    };
    for (let k = 1; k <= 7; k++) {
      if (k > 1) {
        await page.evaluate(([id, dx]) => {
          (window as unknown as { __ocDebug: any }).__ocDebug.scene.cloneBody3D(id, dx, 0, 0);
        }, [lensId, 20 * (k - 1)]);
      }
      const r = await measure(`${k} lens${k > 1 ? "es" : ""}, ghosts`);
      // A scene too bushy for the GPU must say so rather than draw less.
      if (!r.ok) expect(r.reason).toContain("truncated");
      else expectParity(r);
    }
    // The app's own choice: a heavy scene moves the picture to the GPU.
    const label = await page.evaluate(async () => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      await d.fastProjected(); // a big frame marks the scene heavy
      await d.fastProjected(); // so this one runs on the GPU
      return d.fast.label() as string;
    });
    expect(label).toContain("preview GPU");

    await page.evaluate(() => {
      const s = (window as unknown as { __ocDebug: any }).__ocDebug.scene;
      s.setGhosts(false);
      s.setSourceSamples(240);
    });
    expectParity(await measure("7 lenses, no ghosts, 240 samples"));
    console.log(rows.join("\n"));
  });
});
