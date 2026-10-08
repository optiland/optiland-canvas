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

  test("a new sun is the measured one, and its shape and spectrum are the panel's", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#addHeliostat").click();
    await expect(page.locator("#sunSec")).toBeVisible();
    // Jena, the Deutsches Optisches Museum, until someone says otherwise.
    let s = (await debug(page)).solar;
    expect(s.site.lat).toBeCloseTo(50.92781, 5);
    expect(s.site.lon).toBeCloseTo(11.57928, 5);
    // Every wavelength of G173, a limb-darkened disc, a 5% aureole.
    expect(s.continuous).toBe(true);
    expect(s.wavelengths).toBe(1661);
    expect(s.disc).toBe("limb");
    expect(s.csr).toBeCloseTo(0.05, 12);
    await expect(page.locator("#sunReadout")).toContainText("continuous, 1661 wavelengths");
    await expect(page.locator("#sunReadout")).toContainText("limb-darkened disc, 5% aureole");
    const watts = s.fluxW as number;

    // The shape moves light around and makes none.
    await page.locator("#sunDisc").selectOption("buie");
    await page.locator("#sunCsr").fill("20");
    await page.locator("#sunCsr").press("Enter");
    await page.locator("#sunCsr").blur();
    await expect.poll(async () => (await debug(page)).solar.csr).toBeCloseTo(0.2, 12);
    s = (await debug(page)).solar;
    expect(s.disc).toBe("buie");
    expect(s.fluxW).toBeCloseTo(watts, 6);

    // Lines instead of the continuum.
    await page.locator("#sunSpectral").selectOption("lines");
    await expect.poll(async () => (await debug(page)).solar.continuous).toBe(false);
    await expect(page.locator("#sunBands")).toBeVisible();
    await expect(page.locator("#sunReadout")).toContainText("6 lines");

    // A table for the site: a flat 1 W/m²/nm from 300 to 2500 nm carries
    // 2200 W/m², all of it in the band, so the DNI becomes 2200.
    await page.locator("#sunFile").setInputFiles({
      name: "flat.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(["wavelength_nm,direct_W/m2/nm", "300,1", "2500,1"].join("\n")),
    });
    await expect(page.locator("#sunTable")).toHaveText("flat.csv");
    await expect(page.locator("#sunReadout")).toContainText("imported 2 rows, 300–2500 nm");
    s = (await debug(page)).solar;
    expect(s.dni).toBeCloseTo(2200, 9);
    expect(s.spectrum.name).toBe("flat.csv");
    // Back to the standard.
    await page.locator("#sunG173").click();
    await expect(page.locator("#sunTable")).toHaveText("ASTM G173-03");
    expect((await debug(page)).solar.dni).toBe(900);

    // Nonsense is refused and says why, and nothing changes.
    await page.locator("#sunFile").setInputFiles({ name: "junk.txt", mimeType: "text/plain", buffer: Buffer.from("no numbers here") });
    await expect(page.locator("#sunReadout")).toContainText("junk.txt refused");
    await expect(page.locator("#sunTable")).toHaveText("ASTM G173-03");
  });
});
