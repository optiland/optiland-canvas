// How long the wave path takes in the browser's worker, and whether the page
// stays responsive meanwhile. Opt-in:
//   OC_BENCH=1 npx playwright test --project=bench tests/wave-bench.spec.ts
//
// The default scene with its singlet taken out and a single line: a Gaussian
// beam carried 118 mm through empty space to the detector (one band-limited
// angular-spectrum step and the readout), at each grid size the app offers,
// and the smallest again once warm.
// Written to test-results/wave-bench.json and printed.

import { test, expect } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

test("the wave path's time in the worker, and the page's longest task", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#wlPreset").selectOption("d");
  await page.evaluate(() => {
    const d = (window as any).__ocDebug;
    for (const o of JSON.parse(d.scene.sceneObjectsJSON())) if (o.body) d.scene.deleteObject3D(o.id);
  });
  await page.locator("#cohSummary").click();
  await page.locator("#cohType").selectOption("gaussian");
  await page.evaluate(() => {
    const w = window as any;
    w.__long = [];
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) w.__long.push(e.duration);
    }).observe({ type: "longtask", buffered: false });
  });
  const rows: { n: number; ms: number; longestTaskMs: number }[] = [];
  for (const n of [128, 256, 512, 1024, 128]) {
    await page.evaluate(() => ((window as any).__long = []));
    await page.locator("#cohN").selectOption(String(n));
    await page.locator("#detWave").click();
    const status = page.locator("#refStatus");
    await expect(status).toContainText(`wave current · ${n}²`, { timeout: 300_000 });
    const text = (await status.textContent()) ?? "";
    const m = /in ([\d.]+) (ms|s)/.exec(text);
    const ms = m ? Number(m[1]) * (m[2] === "s" ? 1000 : 1) : NaN;
    const longest = await page.evaluate(() => Math.max(0, ...((window as any).__long as number[])));
    rows.push({ n, ms, longestTaskMs: longest });
  }
  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/wave-bench.json", JSON.stringify(rows, null, 2));
  console.log(rows.map((r) => `${r.n}²: ${r.ms} ms in the worker; longest page task ${r.longestTaskMs.toFixed(0)} ms`).join("\n"));
});
