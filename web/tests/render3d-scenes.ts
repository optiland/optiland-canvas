// Scene setups shared by the GPU 3D view's opt-in visual and benchmark specs.
// Everything goes through the page's own `?debug` scene handle, so a setup is
// the same scene the app would build from its buttons.

import { expect, type Page } from "@playwright/test";

export type Obj = { id: number; kind: string; pos: number[] };

export async function openApp(page: Page, query = "") {
  await page.goto(`/?debug${query ? `&${query}` : ""}`);
  await expect(page.locator("#stat")).toContainText("segs");
}

export async function enter3d(page: Page) {
  await page.evaluate(() => (window as any).__ocDebug.setViewMode("3d"));
  const backend = await page.evaluate(() => (window as any).__ocDebug.render3d.ready());
  await page.evaluate(() => (window as any).__ocDebug.render3d.renderNow());
  return backend as string;
}

export const objects = (page: Page): Promise<Obj[]> =>
  page.evaluate(() => JSON.parse((window as any).__ocDebug.scene.sceneObjectsJSON()));

export async function lensId(page: Page): Promise<number> {
  return (await objects(page)).find((o) => o.kind === "profile")!.id;
}

/** `n` singlets along the axis, 20 mm apart, with the detector moved past the
 *  last one. The first is the default lens. */
export async function lenses(page: Page, n: number) {
  await page.evaluate((n) => {
    const s = (window as any).__ocDebug.scene;
    const objs = JSON.parse(s.sceneObjectsJSON());
    const lens = objs.find((o: any) => o.kind === "profile");
    const det = objs.find((o: any) => o.kind === "detector");
    for (let i = 1; i < n; i++) s.cloneBody3D(lens.id, 20 * i, 0, 0);
    s.setObjectPosition(det.id, 20 * (n - 1) + 180, 0, 0);
  }, n);
}

export async function setGhosts(page: Page, on: boolean) {
  const box = page.locator("#ghosts");
  if (on) await box.check();
  else await box.uncheck();
}

/** Remove the light source, for a picture of the solids alone. */
export async function removeSource(page: Page) {
  await page.evaluate(() => {
    const s = (window as any).__ocDebug.scene;
    const src = JSON.parse(s.scene3JSON()).sources?.[0];
    if (src) s.deleteObject3D(src.id);
  });
}

export async function refresh3d(page: Page) {
  // A 3D-view refresh retraces exactly and re-reads the objects.
  await page.evaluate(() => (window as any).__ocDebug.render3d.renderNow());
  await page.keyboard.press("0");
  await page.evaluate(() => (window as any).__ocDebug.render3d.renderNow());
}

export async function environment(page: Page, env: string, bg = "match") {
  await page.selectOption("#v3dEnv", env);
  await page.selectOption("#v3dBg", bg);
}
