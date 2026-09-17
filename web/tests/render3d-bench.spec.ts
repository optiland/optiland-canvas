// The GPU 3D view benchmark (spec §47). Opt-in:
//   OC_BENCH=1 npx playwright test --project=bench
// Writes test-results/render3d-bench.json and .md and prints the tables.
//
// What is measured, per scene:
//   CPU frame   the view's whole frame on the main thread (trace if stale, wasm
//               sync, ray upload, render submission, overlay), median and p95
//               over orbiting frames
//   GPU frame   GPU time of the same frames from timestamp queries, where the
//               device offers them
//   legacy      the Canvas 2D wireframe view's frame for the same scene
//               (`?renderer=legacy`), static camera
//   exact trace one f64 retrace, and the ray upload that follows it
// plus a pointer drag (fast path), curvature edits (mesh rebuilds), lighting
// and projection switches, and a memory stress run.

import { test, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { enter3d, lensId, lenses, openApp, removeSource, setGhosts } from "./render3d-scenes";

type Scene = { name: string; lenses: number; ghosts: boolean; noRays?: boolean };

const SCENES: Scene[] = [
  { name: "default singlet", lenses: 1, ghosts: false },
  { name: "2 lenses", lenses: 2, ghosts: false },
  { name: "3 lenses", lenses: 3, ghosts: false },
  { name: "5 lenses", lenses: 5, ghosts: false },
  { name: "7 lenses", lenses: 7, ghosts: false },
  { name: "10 solids, no rays", lenses: 10, ghosts: false, noRays: true },
  // Ghost trees: the spec's "~30k" and "~100k" segment cases fall between
  // these, and the last is a very heavy picture. The table reports the count
  // actually drawn (segments at or above 1e-5 of the brightest).
  { name: "2 lenses + ghosts", lenses: 2, ghosts: true },
  { name: "3 lenses + ghosts", lenses: 3, ghosts: true },
  { name: "4 lenses + ghosts", lenses: 4, ghosts: true },
  { name: "7 lenses + ghosts (very heavy)", lenses: 7, ghosts: true },
];

const FRAMES = 60;

const stat = (xs: number[]) => {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return { median: NaN, p95: NaN };
  return { median: v[Math.floor(v.length / 2)], p95: v[Math.min(v.length - 1, Math.floor(v.length * 0.95))] };
};
const f = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "n/a");

async function setup(page: Page, s: Scene, query = "") {
  await openApp(page, query);
  if (s.lenses > 1) await lenses(page, s.lenses);
  if (s.noRays) await removeSource(page);
  if (s.ghosts) await setGhosts(page, true);
}

const dbg = (page: Page, expr: string) => page.evaluate(expr);

async function measureThree(page: Page) {
  const backend = await enter3d(page);
  await page.keyboard.press("0");
  // Warm up: shader compilation and first uploads are not steady-state frames.
  for (let i = 0; i < 5; i++) {
    await dbg(page, "window.__ocDebug.render3d.orbit(2, 0)");
    await dbg(page, "window.__ocDebug.render3d.renderNow()");
  }
  const cpu: number[] = [], gpu: number[] = [];
  let last: any = null;
  for (let i = 0; i < FRAMES; i++) {
    await dbg(page, "window.__ocDebug.render3d.orbit(2, 0)");
    last = await dbg(page, "window.__ocDebug.render3d.renderNow()");
    cpu.push(last.frame.total);
    if (typeof last.gpuMs === "number") gpu.push(last.gpuMs);
  }
  // One exact retrace and the upload it causes.
  const retrace: number[] = [], upload: number[] = [];
  for (let i = 0; i < 3; i++) {
    await dbg(page, "window.__ocDebug.render3d.refresh()");
    const t0 = await page.evaluate(() => {
      const t = performance.now();
      (window as any).__ocDebug.exactProjected();
      return performance.now() - t;
    });
    retrace.push(t0);
    const s: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
    upload.push(s.frame.upload);
  }
  return {
    backend,
    segments: last.raySegments,
    triangles: last.triangles,
    drawCalls: last.drawCalls,
    cpu: stat(cpu),
    gpu: stat(gpu),
    exactTrace: stat(retrace).median,
    upload: stat(upload).median,
  };
}

async function measureLegacy(page: Page) {
  await page.evaluate(() => (window as any).__ocDebug.setViewMode("3d"));
  const t: number[] = [];
  for (let i = 0; i < 5; i++) await dbg(page, "window.__ocDebug.renderView()");
  for (let i = 0; i < 30; i++) t.push((await dbg(page, "window.__ocDebug.renderView()")) as number);
  return stat(t);
}

test("GPU 3D view benchmark", async ({ page, browser }) => {
  const report: any = { browser: browser.version(), date: new Date().toISOString(), scenes: [] };

  // --- static scenes, orbiting camera -----------------------------------
  for (const s of SCENES) {
    await setup(page, s);
    const three = await measureThree(page);
    await setup(page, s, "renderer=legacy");
    const legacy = await measureLegacy(page);
    report.scenes.push({ scene: s.name, ...three, legacy });
  }
  report.backend = report.scenes[0].backend;

  // --- a pointer drag of the lens (the interactive f32 path) -------------
  {
    await setup(page, { name: "drag", lenses: 3, ghosts: false });
    await enter3d(page);
    const id = await lensId(page);
    const at: any = await page.evaluate(() => (window as any).__ocDebug.render3d.project({ x: 0, y: 0, z: 0 }));
    const before: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
    await dbg(page, "window.__ocDebug.render3d.startLog()");
    await page.mouse.move(at.x + 2, at.y + 25);
    await page.mouse.down();
    for (let i = 1; i <= 60; i++) {
      await page.mouse.move(at.x + 2 + i * 2, at.y + 25);
      await page.evaluate(() => new Promise(requestAnimationFrame));
    }
    await page.mouse.up();
    await page.waitForTimeout(300);
    const log: any[] = (await dbg(page, "window.__ocDebug.render3d.stopLog()")) as any[];
    const after: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
    const dragging = log.filter((e) => e.quality === "interactive");
    report.drag = {
      lens: id,
      frames: log.length,
      interactiveFrames: dragging.length,
      total: stat(dragging.map((e) => e.total)),
      trace: stat(dragging.map((e) => e.trace)),
      sync: stat(dragging.map((e) => e.sync)),
      upload: stat(dragging.map((e) => e.upload)),
      render: stat(dragging.map((e) => e.render)),
      dpr: dragging[0]?.dpr,
      meshBuildsDuringDrag: after.meshBuilds - before.meshBuilds,
      segments: dragging.length ? dragging[dragging.length - 1].raySegments : 0,
    };
  }

  // --- curvature edits: one mesh rebuild each -----------------------------
  {
    await setup(page, SCENES[0]);
    await enter3d(page);
    const id = await lensId(page);
    const builds: number[] = [], sync: number[] = [];
    const b0: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
    for (let i = 0; i < 30; i++) {
      await page.evaluate(([id, i]) => {
        const s = (window as any).__ocDebug.scene;
        const g = JSON.parse(s.singletGeomJSON(id));
        s.setSingletGeom(id, 40 + (i % 10), g.r2, g.thickness, g.semiAperture, g.conic1, g.conic2);
      }, [id, i]);
      const st: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
      builds.push(st.meshBuildMs);
      sync.push(st.frame.sync);
    }
    const b1: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
    report.curvature = { edits: 30, meshBuilds: b1.meshBuilds - b0.meshBuilds, meshBuildMs: stat(builds), syncMs: stat(sync) };
  }

  // --- lighting and projection switches ------------------------------------
  {
    await setup(page, SCENES[0]);
    await enter3d(page);
    const envs = ["light_studio", "optical_bench", "inspection", "dark_studio"];
    const first: Record<string, number> = {};
    const cached: number[] = [];
    for (let round = 0; round < 3; round++) {
      for (const env of envs) {
        const t0 = Date.now();
        await page.selectOption("#v3dEnv", env);
        const st: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
        const ms = Math.max(Date.now() - t0, st.frame.total);
        if (round === 0) first[env] = ms;
        else cached.push(st.frame.total);
      }
    }
    const persp: number[] = [];
    for (let i = 0; i < 20; i++) {
      await page.locator("#v3dPersp").click();
      const st: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
      persp.push(st.frame.total);
    }
    const st: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
    report.switching = { firstEnvironmentMs: first, cachedEnvironmentFrame: stat(cached), environmentBuilds: st.environmentBuilds, projectionFrame: stat(persp) };
  }

  // --- memory stress (§71) -------------------------------------------------
  {
    await setup(page, SCENES[0]);
    await enter3d(page);
    const snap = async () => {
      const s: any = await dbg(page, "window.__ocDebug.render3d.renderNow()");
      return { geometries: s.geometries, textures: s.textures, jsHeapMB: s.jsHeap ? s.jsHeap / 1048576 : null, cached: s.cachedObjects };
    };
    const id = await lensId(page);
    const start = await snap();
    for (let i = 0; i < 100; i++) {
      const c = await page.evaluate((id) => (window as any).__ocDebug.scene.cloneBody3D(id, 15, 0, 0), id);
      await dbg(page, "window.__ocDebug.render3d.renderNow()");
      await page.evaluate((c) => (window as any).__ocDebug.scene.deleteObject3D(c), c);
      await dbg(page, "window.__ocDebug.render3d.renderNow()");
    }
    const afterAdd = await snap();
    for (let i = 0; i < 100; i++) {
      await page.evaluate(([id, i]) => {
        const s = (window as any).__ocDebug.scene;
        const g = JSON.parse(s.singletGeomJSON(id));
        s.setSingletGeom(id, 35 + (i % 20), g.r2, g.thickness, g.semiAperture, g.conic1, g.conic2);
      }, [id, i]);
      await dbg(page, "window.__ocDebug.render3d.renderNow()");
    }
    const afterCurv = await snap();
    const envs = ["light_studio", "optical_bench", "inspection", "dark_studio", "light_studio"];
    for (let i = 0; i < 50; i++) {
      await page.selectOption("#v3dEnv", envs[i % envs.length]);
      await dbg(page, "window.__ocDebug.render3d.renderNow()");
    }
    const afterEnv = await snap();
    for (let i = 0; i < 20; i++) {
      await page.evaluate(() => (window as any).__ocDebug.setViewMode("2d"));
      await page.evaluate(() => (window as any).__ocDebug.setViewMode("3d"));
      await dbg(page, "window.__ocDebug.render3d.renderNow()");
    }
    const afterViews = await snap();
    report.memory = { start, afterAddDelete100: afterAdd, afterCurvature100: afterCurv, afterEnvironment50: afterEnv, afterViewSwitch20: afterViews };
  }

  // --- report ----------------------------------------------------------------
  const md: string[] = [];
  md.push(`Browser ${report.browser}, 3D backend ${report.backend}, ${new Date(report.date).toUTCString()}`, "");
  md.push("| Scene | Segments | Triangles | Draws | CPU frame median / p95 (ms) | GPU frame median / p95 (ms) | Legacy frame median (ms) | Exact trace (ms) | Ray upload (ms) |");
  md.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const r of report.scenes) {
    md.push(`| ${r.scene} | ${r.segments} | ${r.triangles} | ${r.drawCalls} | ${f(r.cpu.median)} / ${f(r.cpu.p95)} | ${f(r.gpu.median)} / ${f(r.gpu.p95)} | ${f(r.legacy.median)} | ${f(r.exactTrace)} | ${f(r.upload)} |`);
  }
  const d = report.drag;
  md.push("", `Drag (3 lenses, ${d.interactiveFrames} interactive frames of ${d.frames}, DPR ${d.dpr}, ${d.segments} segments): total ${f(d.total.median)} / ${f(d.total.p95)} ms, trace ${f(d.trace.median)}, sync ${f(d.sync.median)}, upload ${f(d.upload.median)}, render ${f(d.render.median)}; mesh rebuilds during the drag: ${d.meshBuildsDuringDrag}.`);
  const c = report.curvature;
  md.push("", `Curvature edits: ${c.edits} edits, ${c.meshBuilds} mesh rebuilds, build ${f(c.meshBuildMs.median)} / ${f(c.meshBuildMs.p95)} ms, sync ${f(c.syncMs.median)} / ${f(c.syncMs.p95)} ms.`);
  const sw = report.switching;
  md.push("", `Lighting: first use ${Object.entries(sw.firstEnvironmentMs).map(([k, v]) => `${k} ${f(v as number, 0)} ms`).join(", ")}; cached switch frame ${f(sw.cachedEnvironmentFrame.median)} / ${f(sw.cachedEnvironmentFrame.p95)} ms; PMREM builds ${sw.environmentBuilds}. Projection switch frame ${f(sw.projectionFrame.median)} / ${f(sw.projectionFrame.p95)} ms.`);
  md.push("", "| Memory | Geometries | Textures | JS heap (MB) | Cached objects |", "|---|---:|---:|---:|---:|");
  for (const [k, v] of Object.entries(report.memory) as [string, any][]) {
    md.push(`| ${k} | ${v.geometries} | ${v.textures} | ${v.jsHeapMB === null ? "n/a" : f(v.jsHeapMB, 1)} | ${v.cached} |`);
  }
  const text = md.join("\n");
  console.log(text);
  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/render3d-bench.json", JSON.stringify(report, null, 2));
  writeFileSync("test-results/render3d-bench.md", text + "\n");
});
