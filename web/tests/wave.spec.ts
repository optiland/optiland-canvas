// The coherent wave path in the app (acceptance cases C54, C58): give the
// source a coherent beam, switch the detector dock to wave, and the worker
// carries the beam to the detector, a lens included; a white source runs as
// its lines, added in power; a Michelson's two arms meet on the detector.

import { test, expect, type Page } from "@playwright/test";

type Debug = {
  scene: {
    sceneObjectsJSON(): string;
    sourceObjectsJSON(): string;
    sourceCoherenceJSON(id: number): string;
    deleteObject3D(id: number): boolean;
  };
};

const debug = (page: Page) => page.evaluate(() => (window as unknown as { __ocDebug: Debug }).__ocDebug !== undefined);

async function giveTheSourceABeam(page: Page) {
  await page.locator("#cohSummary").click();
  await page.locator("#cohType").selectOption("gaussian");
  await expect(page.locator("#cohSummary")).toContainText("Gaussian");
  const coh = await page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    const src = (JSON.parse(d.scene.sourceObjectsJSON()) as { id: number }[])[0].id;
    return JSON.parse(d.scene.sourceCoherenceJSON(src));
  });
  expect(coh.Gaussian.waist_radius_mm).toBe(0.5);
}

test("a coherent beam: a white source as its lines, carried through the singlet, and through empty space", async ({ page }) => {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  expect(await debug(page)).toBe(true);
  await giveTheSourceABeam(page);
  await page.locator("#detWave").click();

  // The default source is white, nine lines: each carried coherently, the
  // lines added in power.
  await expect(page.locator("#refStatus")).toContainText("wave current", { timeout: 180_000 });
  await expect(page.locator("#dockNums")).toHaveAttribute("title", /9 spectral lines added in power/);
  await page.locator("#wlPreset").selectOption("d");
  // The default scene's singlet is traced face by face; the beam reaches the
  // detector near its focus.
  await page.locator("#detWave").click();
  await expect(page.locator("#refStatus")).toContainText("wave current", { timeout: 60_000 });
  await expect(page.locator("#dockNums")).toContainText("detected");

  // Without it, the beam crosses empty space to the detector.
  await page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    for (const o of JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; body: boolean }[]) {
      if (o.body) d.scene.deleteObject3D(o.id);
    }
  });
  await page.locator("#detWave").click();
  await expect(page.locator("#refStatus")).toContainText("wave current", { timeout: 60_000 });
  await expect(page.locator("#dockNums")).toContainText("detected (100.0%)");

  // The phase view of the arriving field.
  await page.locator("#wavePhaseBtn").click();
  await expect(page.locator("#detHead")).toContainText("phase");
});

test("a DMD that turns the beam away from the detector: the dock names why", async ({ page }) => {
  // The DMD's own wave physics is held in Rust (oc-wave/tests/path.rs: its
  // orders focus fλ/p apart); here the app must say why there is no picture.
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#wlPreset").selectOption("d");
  await giveTheSourceABeam(page);
  await page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    for (const o of JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; body: boolean }[]) {
      if (o.body) d.scene.deleteObject3D(o.id);
    }
  });
  // Facing the beam, its ON mirrors send it back at twice their tilt.
  await page.locator("#addDmd").click();
  await page.locator("#detWave").click();
  await expect(page.locator("#refStatus")).toContainText("no chain from the source to the detector", { timeout: 60_000 });
});

test("a Michelson interferometer: its two arms meet on the detector", async ({ page }) => {
  // The fringes' period and phase are held in Rust (oc-wave/tests/path.rs);
  // here the app must open the scene, run both arms and add them.
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#open3dFile").setInputFiles("public/demo-michelson.ocanvas");
  await page.locator("#detWave").click();
  await expect(page.locator("#refStatus")).toContainText("wave current", { timeout: 60_000 });
  await expect(page.locator("#dockNums")).toHaveAttribute("title", /2 arms added as fields/);
});
