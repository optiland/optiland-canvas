// Tubes, the sun and heliostats, from the palette.
//
// A tube arrives with its card open and a silvered bore, and the card's bore
// select reaches the scene. A heliostat arrives with a sun: the sun panel
// shows where the sun stands and what its beam carries, and moving the clock
// turns the mirror, because the kernel re-aims it on every edit.

import { test, expect, type Page } from "@playwright/test";

type Obj = { id: number; kind: string; rot: number[] };
type Debug = {
  scene: {
    sceneObjectsJSON(): string;
    tubeJSON(id: number): string;
    solarJSON(): string;
  };
};

const debug = (page: Page) =>
  page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    return {
      objects: JSON.parse(d.scene.sceneObjectsJSON()) as Obj[],
      solar: JSON.parse(d.scene.solarJSON()),
    };
  });

test.describe("solar — tubes, the sun, heliostats", () => {
  test("a tube opens with a silvered bore, and its card refinishes it", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#addTube").click();
    const card = page.locator(".card");
    await expect(card).toHaveCount(1);
    await expect(card.locator(".c3-tube")).toBeVisible();
    await expect(card.locator('input[data-name="tubeInner"]')).toHaveValue("20");
    await expect(card.locator('select[data-name="tubeBore"]')).toHaveValue("silver-protected");

    await card.locator('select[data-name="tubeBore"]').selectOption("black");
    const bore = await page.evaluate(() => {
      const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
      const tube = (JSON.parse(d.scene.sceneObjectsJSON()) as Obj[]).find((o) => o.kind === "tube")!;
      return JSON.parse(d.scene.tubeJSON(tube.id)).bore.kind;
    });
    expect(bore).toBe("black");
  });

  test("a heliostat brings the sun, and the clock turns it", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#sunSec")).toBeHidden();
    await page.locator("#addHeliostat").click();
    await expect(page.locator("#sunSec")).toBeVisible();
    await expect(page.locator("#sunReadout")).toContainText("heliostat #");

    const before = await debug(page);
    expect(before.solar.heliostats).toHaveLength(1);
    const mirror = before.solar.heliostats[0].mirror as number;
    const rot = (o: { objects: Obj[] }) => o.objects.find((x) => x.id === mirror)!.rot;

    // Two hours later on the same day.
    const slider = page.locator("#sunTime");
    const minutes = Number(await slider.inputValue());
    await slider.evaluate((el, v) => {
      (el as HTMLInputElement).value = String(v);
      el.dispatchEvent(new Event("change"));
    }, (minutes + 120) % 1440);

    const after = await debug(page);
    expect(after.solar.time.minutes).toBeCloseTo((minutes + 120) % 1440, 3);
    expect(after.solar.azimuth).not.toBeCloseTo(before.solar.azimuth, 1);
    expect(rot(after)).not.toEqual(rot(before));
  });
});
