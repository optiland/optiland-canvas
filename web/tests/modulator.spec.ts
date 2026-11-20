// A DMD and an SLM from the palette (acceptance case C54): each arrives facing
// the beam, the component view opens its editor, the editor's fields reach the
// scene, a refused edit says why, and the f32 preview draws the device rather
// than leaving it out.

import { test, expect, type Page } from "@playwright/test";

type Debug = {
  scene: {
    sceneObjectsJSON(): string;
    modulatorJSON(id: number): string;
    modulatorPreviewSize(id: number, side: number): Uint32Array;
    modulatorPreviewRGBA(id: number, side: number): Uint8Array;
  };
};

const modulator = (page: Page, id: number) =>
  page.evaluate((id) => JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.modulatorJSON(id)), id);

const lastOfKind = (page: Page, kind: string) =>
  page.evaluate((kind) => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    const objs = JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; kind: string }[];
    return objs.filter((o) => o.kind === kind).pop()!.id;
  }, kind);

test("a DMD from the palette opens its editor, edits its pattern, and refuses an impossible mirror", async ({ page }) => {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#addDmd").click();
  const id = await lastOfKind(page, "dmd");
  const m = await modulator(page, id);
  // TI DLP9000's pitch and tilt (DLPA083B Table 2-1), all mirrors ON.
  expect(m.device.pitch_um).toBe(7.56);
  expect(m.device.kind.Dmd.tilt_deg).toBe(12);
  expect(m.device.pattern).toEqual({ Uniform: { value: 1 } });

  if (!(await page.locator("#componentInset").isVisible())) await page.locator("#insetTab").click();
  const inset = page.locator("#componentInset");
  await expect(inset).toContainText("DMD");
  await expect(inset.locator(".cv-read")).toContainText("orders propagate");

  await inset.getByText("grating", { exact: true }).click();
  await expect.poll(async () => Object.keys((await modulator(page, id)).device.pattern.Generated ?? {})).toEqual(["BinaryGrating"]);
  const period = inset.locator('input[data-name="modulatorPeriod"]');
  await period.fill("4");
  await period.press("Enter");
  await expect.poll(async () => (await modulator(page, id)).device.pattern.Generated.BinaryGrating.period_px).toBe(4);

  // The ray tracer can trace its diffraction orders instead of the geometric limit.
  await inset.getByText("orders", { exact: true }).click();
  await expect.poll(async () => (await modulator(page, id)).rayModel).toEqual({ Orders: { max_orders: 16, min_efficiency: 0.001 } });
  await expect(inset.locator(".cv-read")).toContainText("booked as untraced orders");

  // A mirror wider than its pitch is refused, and nothing changes.
  const fill = inset.locator('input[data-name="modulatorFill"]');
  const before = (await modulator(page, id)).device.kind.Dmd.mirror_fill;
  await fill.fill("1.5");
  await fill.press("Enter");
  await expect(inset.locator(".cv-note")).toContainText("refused");
  expect((await modulator(page, id)).device.kind.Dmd.mirror_fill).toBe(before);

  // The preview traces the device: nothing is left out of its picture.
  await expect(page.locator("#preview3dBadge")).not.toContainText("omits");

  // An image of your own: top half white, bottom half black. The device's
  // top rows turn ON, its bottom rows OFF.
  const png = await page.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 8;
    c.height = 8;
    const x = c.getContext("2d")!;
    x.fillStyle = "#000";
    x.fillRect(0, 0, 8, 8);
    x.fillStyle = "#fff";
    x.fillRect(0, 0, 8, 4);
    return c.toDataURL("image/png").split(",")[1];
  });
  await inset.locator('input[data-name="modulatorImport"]').setInputFiles({
    name: "half.png", mimeType: "image/png", buffer: Buffer.from(png, "base64"),
  });
  const [nx, ny] = (await modulator(page, id)).device.pixels;
  await expect.poll(async () => (await modulator(page, id)).device.pattern.Explicit?.count).toBe(nx * ny);
  await expect(inset.locator(".cv-read")).toContainText("an imported bitmap");
  const [top, bottom] = await page.evaluate((id) => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    const [w, h] = d.scene.modulatorPreviewSize(id, 8);
    const px = d.scene.modulatorPreviewRGBA(id, 8);
    return [px[0], px[4 * w * (h - 1)]];
  }, id);
  expect(top).toBeGreaterThan(200); // ON
  expect(bottom).toBeLessThan(60); // OFF
});

test("an SLM from the palette shows a blazed grating and its orders", async ({ page }) => {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#addSlm").click();
  const id = await lastOfKind(page, "slm");
  expect((await modulator(page, id)).kind).toBe("slm");

  if (!(await page.locator("#componentInset").isVisible())) await page.locator("#insetTab").click();
  const inset = page.locator("#componentInset");
  await expect(inset).toContainText("LCoS SLM");
  await inset.getByText("blazed", { exact: true }).click();
  await expect.poll(async () => Object.keys((await modulator(page, id)).device.pattern.Generated ?? {})).toEqual(["BlazedGrating"]);
  // An 8 µm pixel at 633 nm is 12.6λ: below the 14λ where scalar theory's
  // error passes ±5%, and the editor says so.
  await expect(inset.locator(".cv-read")).toContainText("below 14λ");
  await expect(page.locator("#preview3dBadge")).not.toContainText("omits");
});
