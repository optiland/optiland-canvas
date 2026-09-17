// Screenshot regressions for the GPU 3D view. Opt-in (see playwright.config.ts):
//   OC_VISUAL=1 npx playwright test --project=visual [--update-snapshots]
//
// Baselines are per machine and backend (Playwright suffixes them with the
// platform), so they are recorded on the development machine rather than in CI.
// A small pixel tolerance absorbs driver noise; the status line, which carries
// frame timings, is masked.

import { test, expect, type Page } from "@playwright/test";
import { enter3d, environment, lensId, lenses, openApp, refresh3d } from "./render3d-scenes";

const shot = async (page: Page, name: string) => {
  await page.evaluate(() => (window as any).__ocDebug.render3d.renderNow());
  await expect(page).toHaveScreenshot(name, {
    mask: [page.locator("#stat")],
    maxDiffPixelRatio: 0.01,
    animations: "disabled",
  });
};

test.describe("GPU 3D view appearance", () => {
  test.beforeEach(async ({ page }) => {
    await openApp(page);
  });

  test("default singlet, dark studio", async ({ page }) => {
    await enter3d(page);
    await shot(page, "default-dark.png");
  });

  test("default singlet, light studio", async ({ page }) => {
    await enter3d(page);
    await environment(page, "light_studio");
    await shot(page, "default-light.png");
  });

  test("prism", async ({ page }) => {
    await page.locator("#addPrism").click();
    await enter3d(page);
    await refresh3d(page);
    await shot(page, "prism.png");
  });

  test("mirror fold", async ({ page }) => {
    await page.locator("#open3dFile").setInputFiles("public/demo-fold.ocanvas");
    await expect(page.locator("#preview3dMsg")).toContainText("opened");
    await enter3d(page);
    await refresh3d(page);
    await shot(page, "mirror-fold.png");
  });

  test("aperture and second detector", async ({ page }) => {
    await page.locator("#addAperture").click();
    await page.locator("#addDetector").click();
    await enter3d(page);
    await page.keyboard.press("Escape");
    await refresh3d(page);
    await shot(page, "aperture-detector.png");
  });

  test("two overlapping glass elements", async ({ page }) => {
    await page.evaluate(() => {
      const s = (window as any).__ocDebug.scene;
      const lens = JSON.parse(s.sceneObjectsJSON()).find((o: any) => o.kind === "profile");
      s.cloneBody3D(lens.id, 3, 4, 0);
    });
    await enter3d(page);
    await refresh3d(page);
    await shot(page, "overlapping-glass.png");
  });

  test("three lenses with ghosts", async ({ page }) => {
    await lenses(page, 3);
    await page.locator("#ghosts").check();
    await enter3d(page);
    await refresh3d(page);
    await shot(page, "three-lenses-ghosts.png");
  });

  test("selected and hovered lens", async ({ page }) => {
    await enter3d(page);
    const id = await lensId(page);
    const at = await page.evaluate(() => (window as any).__ocDebug.render3d.project({ x: 0, y: 0, z: 0 }));
    await page.mouse.click(at.x + 2, at.y + 25);
    await expect.poll(() => page.evaluate(() => (window as any).__ocDebug.render3d.selected())).toBe(id);
    await shot(page, "selected-lens.png");
    await page.keyboard.press("Escape");
    await page.mouse.move(at.x + 2, at.y + 26);
    await expect.poll(() => page.evaluate(() => (window as any).__ocDebug.render3d.hovered())).toBe(id);
    await shot(page, "hovered-lens.png");
  });

  test("occluded rays, optical bench, perspective", async ({ page }) => {
    await page.locator("#addAperture").click();
    await enter3d(page);
    await page.keyboard.press("Escape");
    await environment(page, "optical_bench");
    await page.selectOption("#v3dRays", "occluded");
    await page.locator("#v3dPersp").click();
    await refresh3d(page);
    await shot(page, "bench-occluded-perspective.png");
  });

  test("inspection lighting on a medium grey background", async ({ page }) => {
    await page.locator("#addPrism").click();
    await enter3d(page);
    await page.keyboard.press("Escape");
    await environment(page, "inspection", "mid_gray");
    await refresh3d(page);
    await shot(page, "inspection-grey.png");
  });

  test("wide rays through three lenses, dark and light", async ({ page }) => {
    await lenses(page, 3);
    await page.locator("#ghosts").check();
    await enter3d(page);
    await page.locator("#v3dRayWidth").fill("3");
    await refresh3d(page);
    await shot(page, "wide-rays-dark.png");
    await environment(page, "light_studio");
    await shot(page, "wide-rays-light.png");
  });

  test("a hit map on a detector seen end-on", async ({ page }) => {
    // A point source 70 mm out: the image lies past the detector, so the
    // detector sees a spread pattern rather than one focused cell.
    await page.selectOption("#srcType", "point");
    await enter3d(page);
    await page.selectOption("#v3dRays", "occluded");
    await page.locator("#v3dSide").click();
    await refresh3d(page);
    await shot(page, "hit-map-end-on.png");
  });
});
