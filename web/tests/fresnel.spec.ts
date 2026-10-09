// The Fresnel lens: the palette button, its recipe editor in the component
// view, and the proof that the grooves focus.
//
// The optical check reads the f64 detector readout, so what is asserted is the
// reference tracer's own spot, not a picture of it.

import { test, expect, type Page } from "@playwright/test";

test.describe.configure({ timeout: 180_000 });

type Debug = { scene: any; inset: { view: () => { id: number; kind: string } } };
async function freshApp(page: Page) {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.evaluate(() => {
    localStorage.clear();
    localStorage.setItem("optiland-canvas:tools", "open");
  });
  await page.reload();
  await expect(page.locator("#stat")).toContainText("segs");
}

async function openInset(page: Page) {
  if (!(await page.locator("#componentInset").isVisible())) await page.locator("#insetTab").click();
  await expect(page.locator("#componentInset")).toBeVisible();
}

/** The flux-weighted rms radius of what the detector caught, from the f64
 *  reference's cells, and the flux itself. */
const spot = (page: Page, det: number) =>
  page.evaluate((det) => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    const r = JSON.parse(d.scene.detectorResultJSON(det));
    const [nx, ny] = r.bins as [number, number];
    const cells = r.incident as number[];
    const hw = 12, hh = 12;
    let sum = 0, m2 = 0;
    cells.forEach((f, k) => {
      const u = -hw + ((k % nx) + 0.5) * (2 * hw) / nx;
      const v = -hh + (Math.floor(k / nx) + 0.5) * (2 * hh) / ny;
      sum += f;
      m2 += f * (u * u + v * v);
    });
    return { flux: r.totalIncident as number, rms: Math.sqrt(m2 / Math.max(sum, 1e-300)) };
  }, det);

const recipe = (page: Page, id: number) =>
  page.evaluate((id) => JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.fresnelLensJSON(id)), id);

test("a Fresnel lens from the palette focuses the beam, and its recipe edits rebuild it", async ({ page }) => {
  await freshApp(page);
  // Take the default singlet out, so the Fresnel lens is the only lens, and
  // find the default detector.
  const { det } = await page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    const objs = JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; kind: string }[];
    for (const o of objs.filter((o) => o.kind === "profile")) d.scene.deleteObject3D(o.id);
    return { det: objs.find((o) => o.kind === "detector")!.id };
  });

  await openInset(page);
  await page.locator("#addFresnel").click();
  await expect(page.locator("#componentInset .inset-sub")).toContainText("Fresnel lens");
  const view = await page.evaluate(() => (window as unknown as { __ocDebug: Debug }).__ocDebug.inset.view());
  expect(view.kind).toBe("fresnel");
  const lens = view.id;
  const info = await recipe(page, lens);
  // 25 mm at the default 0.73 mm pitch, which no whole-millimetre beam height
  // is a multiple of: a ray exactly on a zero-draft riser grazes it.
  expect(info.zones).toBe(35);
  await expect(page.locator("#componentInset .cv-read")).toContainText("35");

  // Another pitch off the grid rebuilds the lens.
  const groove = page.locator('#componentInset input[data-name="fresnelGroove"]');
  await groove.fill("0.61");
  await expect.poll(async () => (await recipe(page, lens)).zones).toBe(41);

  // The detector at the focus the recipe promises: R/(n-1) past the central
  // facet, which sits about 1 mm downstream of the lens centre.
  const x = await page.evaluate((id) => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    return (JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; pos: number[] }[]).find((o) => o.id === id)!.pos[0];
  }, lens);
  await page.evaluate(([det, at]) => (window as unknown as { __ocDebug: Debug }).__ocDebug.scene.setObjectPosition(det, at, 0, 0), [det, x + 1 + info.efl]);

  // The 20 mm wide beam converges to a spot well under a millimetre.
  const s = await spot(page, det);
  expect(s.flux).toBeGreaterThan(0.8);
  expect(s.rms).toBeLessThan(0.6);

  // Half the pitch: twice the zones, and it still focuses in the same place.
  await groove.fill("0.305");
  await expect.poll(async () => (await recipe(page, lens)).zones).toBe(82);
  expect((await spot(page, det)).rms).toBeLessThan(0.6);

  // Turned round, grooves toward the light: still a converging lens.
  await page.locator('#componentInset button[data-value="MinusX"]').click();
  await expect.poll(async () => (await recipe(page, lens)).recipe.facing).toBe("MinusX");
  const turned = await recipe(page, lens);
  await page.evaluate(([det, at]) => (window as unknown as { __ocDebug: Debug }).__ocDebug.scene.setObjectPosition(det, at, 0, 0), [det, x + 1 + turned.efl]);
  expect((await spot(page, det)).rms).toBeLessThan(1.5);

  // A recipe Rust refuses leaves the lens as it was and says why.
  await page.locator('#componentInset input[data-name="fresnelClear"]').fill("-3");
  await expect(page.locator("#componentInset .cv-note")).toContainText("refused");
  expect((await recipe(page, lens)).recipe.clear_radius_mm).toBe(25);
});
