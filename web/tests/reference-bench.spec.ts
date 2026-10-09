// How responsive the page stays while the f64 reference is traced. Opt-in:
//   OC_BENCH=1 npx playwright test --project=bench tests/reference-bench.spec.ts
//
// For each gallery design, with and without ghosts: the longest task the page's
// main thread ran (a task over 50 ms is one a user can feel), and how long the
// reference took to come back from its worker. Written to
// test-results/reference-bench.json and printed.

import { test, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const CASES: { id: string; ghosts: boolean }[] = [
  { id: "cooke-triplet", ghosts: false },
  { id: "uv-projection-lens", ghosts: false },
  { id: "objective-us008879901", ghosts: false },
  { id: "cooke-triplet", ghosts: true },
  { id: "double-gauss", ghosts: true },
  { id: "objective-60x", ghosts: true },
  { id: "uv-projection-lens", ghosts: true },
  { id: "objective-us008879901", ghosts: true },
];

async function longTasks(page: Page) {
  await page.evaluate(() => {
    const w = window as any;
    w.__long = [];
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) w.__long.push(e.duration);
    }).observe({ type: "longtask", buffered: false });
  });
}

test("the page stays responsive while the reference is traced", async ({ page }) => {
  test.setTimeout(900_000);
  const rows: object[] = [];
  for (const c of CASES) {
    await page.goto(`/?debug&gallery=${c.id}`);
    await page.waitForFunction(() => (window as any).__ocDebug.referenceCurrent(), null, { timeout: 600_000 });
    await longTasks(page);
    const t0 = Date.now();
    if (c.ghosts) await page.locator("#ghosts").check();
    else {
      // An edit: nudge the density, which retraces everything.
      await page.locator("#density").fill("24");
      await page.locator("#density").dispatchEvent("change");
    }
    await page.waitForFunction(() => (window as any).__ocDebug.referenceCurrent(), null, { timeout: 600_000 });
    const wall = Date.now() - t0;
    const long: number[] = await page.evaluate(() => (window as any).__long);
    const status = await page.locator("#refStatus").textContent();
    const workers: number = await page.evaluate(() => (window as any).__ocDebug.referenceWorkers());
    // What a repaint costs with the reference's picture on screen, and one
    // frame of the fast preview on the CPU mirror.
    const cost = await page.evaluate(() => {
      const d = (window as any).__ocDebug;
      const draws = [0, 1, 2].map(() => d.renderView());
      const t0 = performance.now();
      const segs = d.cpuProjected() as Float32Array;
      const cpu = performance.now() - t0;
      return {
        draw_ms: Math.round(Math.min(...draws)),
        cpu_preview_ms: Math.round(cpu),
        preview_segments: segs.length / 11,
        exact_segments: d.referenceSegments() as number,
      };
    });
    const row = {
      design: c.id,
      ghosts: c.ghosts,
      until_current_ms: wall,
      longest_main_task_ms: Math.round(Math.max(0, ...long)),
      main_tasks_over_50ms: long.length,
      status,
      workers,
      ...cost,
    };
    rows.push(row);
    console.log(JSON.stringify(row));
  }
  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/reference-bench.json", JSON.stringify(rows, null, 1));
});
