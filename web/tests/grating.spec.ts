// A diffraction grating from the palette: it arrives facing the beam, the
// component view opens its editor, and the editor's fields reach the scene.

import { test, expect, type Page } from "@playwright/test";

type Debug = {
  scene: { sceneObjectsJSON(): string; gratingJSON(id: number): string };
  inset: { view(): { id: number; kind: string } };
};

const grating = (page: Page, id: number) =>
  page.evaluate((id) => JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.gratingJSON(id)), id);

test("a grating from the palette opens its editor, and the editor edits it", async ({ page }) => {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#addGrating").click();
  const id = await page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    const objs = JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; kind: string }[];
    return objs.filter((o) => o.kind === "grating").pop()!.id;
  });
  expect((await grating(page, id)).linesPerMm).toBe(600);

  if (!(await page.locator("#componentInset").isVisible())) await page.locator("#insetTab").click();
  const inset = page.locator("#componentInset");
  await expect(inset).toContainText("reflection grating");
  await expect(inset.locator(".cv-read")).toContainText("nm leave between");

  const lines = inset.locator('input[data-name="gratingLines"]');
  await lines.fill("1200");
  await lines.press("Enter");
  await expect.poll(async () => (await grating(page, id)).linesPerMm).toBe(1200);

  await inset.getByText("transmission", { exact: true }).click();
  await expect.poll(async () => (await grating(page, id)).transmissive).toBe(true);
  await expect(inset).toContainText("transmission grating");

  // More light than there is is refused, and says so.
  const eta = inset.locator('input[data-name="gratingEfficiency"]');
  await eta.fill("0.95");
  await eta.press("Enter");
  await expect(inset.locator(".cv-note")).toContainText("refused");
  expect((await grating(page, id)).efficiency).toBe(0.65);
});
