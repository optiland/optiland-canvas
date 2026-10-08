// Locking one element to another, and sizing an aperture's hole and mask.
//
// The case that asked for both: a baffle just in front of a heliostat, its
// hole the size of the mirror, its plate big enough to catch the sunlight the
// mirror misses. Locked "in front of" the heliostat, it has to stay parallel
// to the mirror and on its axis while the sun turns the mirror.

import { test, expect, type Page } from "@playwright/test";

test.describe.configure({ timeout: 180_000 });

type Obj = { id: number; kind: string; pos: number[]; rot: number[] };
type Debug = {
  scene: {
    sceneObjectsJSON(): string;
    solarJSON(): string;
    apertureShapeJSON(id: number): string;
    linkJSON(id: number): string;
    setSunTime(y: number, m: number, d: number, minutes: number, utc: number): boolean;
  };
};

const objects = (page: Page) =>
  page.evaluate(() =>
    JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.sceneObjectsJSON()) as Obj[]);

/** A unit quaternion `[x, y, z, w]` applied to `v`. */
function rotate(q: number[], v: number[]): number[] {
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)];
}

test("an aperture locked in front of a heliostat stays on it through the day", async ({ page }) => {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#addHeliostat").click();
  const helio = await page.evaluate(() =>
    JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.solarJSON()).heliostats[0].mirror as number);

  await page.locator(".card .c-close").first().click().catch(() => {});
  await page.locator("#addAperture").click();
  const card = page.locator(".card", { has: page.locator(".c3-aperture:visible") });
  await expect(card).toHaveCount(1);
  const ap = (await objects(page)).filter((o) => o.kind === "aperture").pop()!.id;

  // The hole and the mask, sized on their own: a rectangular hole in a
  // rectangular plate.
  await card.locator('select[data-name="apOpenKind"]').selectOption("rect");
  await card.locator('input[data-name="apOpenW"]').fill("1200");
  await card.locator('input[data-name="apOpenH"]').fill("1200");
  await card.locator('select[data-name="apMaskKind"]').selectOption("rect");
  await card.locator('input[data-name="apMaskW"]').fill("3000");
  await card.locator('input[data-name="apMaskH"]').fill("3000");
  const shape = await page.evaluate((id) =>
    JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.apertureShapeJSON(id)), ap);
  expect(shape).toEqual({ opening: { kind: "rect", w: 1200, h: 1200 }, mask: { kind: "rect", w: 3000, h: 3000 } });

  // A plate made smaller than its hole is refused, and the card says why; a
  // hole made larger than its plate takes the plate with it.
  await card.locator('input[data-name="apMaskW"]').fill("1000");
  await expect(card.locator(".c3-apNote")).toContainText("refused");
  await card.locator('input[data-name="apMaskW"]').fill("3000");
  await expect(card.locator(".c3-apNote")).toHaveText("");

  // Lock it in front of the heliostat, 30 mm out.
  await card.locator('select[data-name="linkParent"]').selectOption(String(helio));
  await expect(card.locator(".c3-linkGap")).toBeVisible();
  await card.locator('input[data-name="linkGap"]').fill("30");
  const link = await page.evaluate((id) =>
    JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.linkJSON(id)), ap);
  expect(link).toEqual({ parent: helio, mode: "front", gap: 30 });

  // Sweep the day: the aperture is on the mirror's normal, 30 mm out, and
  // parallel to it, at every hour.
  for (const minutes of [8 * 60, 12 * 60, 17 * 60]) {
    await page.evaluate((m) => {
      const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
      const t = JSON.parse(d.scene.solarJSON()).time;
      d.scene.setSunTime(t.year, t.month, t.day, m, t.utcOffset);
    }, minutes);
    const objs = await objects(page);
    const m = objs.find((o) => o.id === helio)!, a = objs.find((o) => o.id === ap)!;
    const n = rotate(m.rot, [0, 0, 1]);
    const na = rotate(a.rot, [0, 0, 1]);
    for (let k = 0; k < 3; k++) {
      expect(a.pos[k]).toBeCloseTo(m.pos[k] + 30 * n[k], 6);
      expect(na[k]).toBeCloseTo(n[k], 9);
    }
  }

  // Free again: it stays where the link left it.
  const before = (await objects(page)).find((o) => o.id === ap)!.pos;
  await card.locator('select[data-name="linkParent"]').selectOption("");
  expect(await page.evaluate((id) =>
    (window as unknown as { __ocDebug: Debug }).__ocDebug.scene.linkJSON(id), ap)).toBe("null");
  expect((await objects(page)).find((o) => o.id === ap)!.pos).toEqual(before);
});
