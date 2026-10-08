// Coating authoring in the component view (coating-fidelity spec CF-11): a
// stack edited a layer at a time, the curve that follows it, a save and load
// that keeps it, the report that names a material with no data at a launched
// wavelength, the s/p and angle plots, and the CSV dialog for a measured table.

import { test, expect, type Page } from "@playwright/test";

test.describe.configure({ timeout: 180_000 });

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

const viewId = (page: Page) => page.evaluate(() => (window as any).__ocDebug.inset.view().id as number);

const stack = (page: Page, id: number) =>
  page.evaluate((id) => JSON.parse((window as any).__ocDebug.scene.filmStackJSON(id)), id);

const curve = (page: Page, id: number) =>
  page.evaluate((id) => Array.from((window as any).__ocDebug.scene.surfaceRtF64(id, false, new Float64Array([480, 520, 560, 600, 640]), 45) as Float64Array), id);

test.describe("coating authoring", () => {
  test("a layer added in the stack editor changes the curve and survives a save and load", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await page.locator("#addDichroic").click();
    await expect(page.locator("#componentInset .inset-sub")).toContainText("dichroic");
    const id = await viewId(page);
    await expect(page.locator("#cvStack .cv-stack-row").first()).toBeVisible();
    const n0 = (await stack(page, id)).layers.length;
    const before = await curve(page, id);

    await page.locator("#cvAddLayer").selectOption("magnesium_fluoride_bulk");
    await expect.poll(async () => (await stack(page, id)).layers.length).toBe(n0 + 1);
    const after = await curve(page, id);
    expect(after).not.toEqual(before);
    await expect(page.locator(`#cvStack .cv-stack-row[data-layer="${n0}"] .cv-film-pick`)).toHaveValue("magnesium_fluoride_bulk");

    // Save as v3 and load it back: the layer is still there and the curve the same.
    await page.evaluate(() => {
      const s = (window as any).__ocDebug.scene;
      s.loadScene3JSON(s.scene3JSON());
    });
    expect((await stack(page, id)).layers.length).toBe(n0 + 1);
    expect(await curve(page, id)).toEqual(after);
  });

  test("the report names a material with no data at a launched wavelength", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await page.locator("#addDichroic").click();
    await expect(page.locator("#componentInset .inset-sub")).toContainText("dichroic");
    // A 2.4 um line, past hafnia's data.
    await page.evaluate(() => (window as any).__ocDebug.scene.setSourceWavelengths(new Float64Array([550, 2400]), new Float64Array(0)));
    await page.locator('#cvStack .cv-stack-row[data-layer="0"] .cv-film-pick').selectOption("hafnia");
    await expect(page.locator("#cvReport")).toContainText("has no data at 2400 nm");
    await expect(page.locator("#cvReport")).toContainText("refused at 2400 nm");
    await expect(page.locator('#cvStack .cv-stack-row[data-layer="0"]')).toHaveClass(/warn/);
  });

  test("the plot shows s and p apart, and against angle", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await page.locator("#addDichroic").click();
    await expect(page.locator("#componentInset .inset-sub")).toContainText("dichroic");
    const id = await viewId(page);
    await page.locator('#componentInset button[data-value="sp"]').click();
    await expect(page.locator('#componentInset button[data-value="sp"]')).toHaveClass(/active/);
    await page.locator('#componentInset button[data-value="angle"]').click();
    await expect(page.locator('#componentInset button[data-value="angle"]')).toHaveClass(/active/);
    // At 45 degrees near the edge s and p differ, which is what the plot shows.
    const sp = await page.evaluate((id) => Array.from((window as any).__ocDebug.scene.coatingSpF64(id, false, new Float64Array([540]), new Float64Array([45])) as Float64Array), id);
    expect(Math.abs(sp[0] - sp[1])).toBeGreaterThan(0.05);
  });

  test("a vendor table imports through the CSV dialog", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await page.locator("#addSplitter").click();
    await expect(page.locator("#componentInset .inset-sub")).toContainText("splitter");
    const id = await viewId(page);
    await page.locator('#componentInset button:has-text("table…")').click();
    await expect(page.locator("#tableDialog")).toBeVisible();
    // Too few columns for the data choice is said before anything is sent.
    await page.locator("#tdKind").selectOption("sp");
    await page.locator("#tdImport").click();
    await expect(page.locator("#tdError")).toContainText("needs 4 column numbers");
    await page.locator("#tdKind").selectOption("lossless");
    await page.locator("#tdText").fill("Wavelength (nm),T (%)\n400,10\n550,80\n700,85\n");
    await page.locator("#tdSource").fill("FB550 datasheet");
    await page.locator("#tdImport").click();
    await expect(page.locator("#tableDialog")).toHaveCount(0);
    await expect(page.locator("#componentInset .cv-read")).toContainText("measured table");
    await expect(page.locator("#componentInset .cv-read")).toContainText("at 45° only");
    const t = await page.evaluate((id) => JSON.parse((window as any).__ocDebug.scene.surfaceTableJSON(id)), id);
    expect(t.aoi_deg).toEqual([45]);
    expect(t.provenance.reference).toBe("FB550 datasheet");
  });
});
