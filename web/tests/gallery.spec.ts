// The gallery (web/src/gallery.ts): Optiland's sample designs, opened on the
// canvas.
//
// What only a browser can show: that the gallery lists what public/gallery
// holds, that a card opens its design as the scene, that the view frames it,
// and that its light reaches its image in the app's own trace. That each
// design is Optiland's, ray for ray, is oc-trace/tests/gallery3.rs's to show.

import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

type Design = { id: string; name: string; category: string; source: string; status: string };
const index: { designs: Design[] } = JSON.parse(
  readFileSync(new URL("../public/gallery/index.json", import.meta.url), "utf8"),
);
const ported = index.designs.filter((d) => d.status === "ported");
const refused = index.designs.filter((d) => d.status !== "ported");

async function ready(page: Page, query = "") {
  await page.goto(`/?debug${query}`);
  await expect(page.locator("#stat")).toContainText("segs");
}

/** The exact trace's flux on the image detector, and how many fields light it. */
async function imageFlux(page: Page): Promise<{ total: number; sources: number }> {
  return page.evaluate(() => {
    const scene = (window as unknown as { __ocDebug: { scene: any } }).__ocDebug.scene;
    const res = JSON.parse(scene.firstDetectorResultJSON());
    const doc = JSON.parse(scene.scene3JSON());
    return { total: res?.totalIncident ?? 0, sources: doc.sources.length };
  });
}

test.describe("the gallery", () => {
  test("lists every design, and says why the rest are not there", async ({ page }) => {
    await ready(page);
    await page.locator("#galleryOpen").click();
    await expect(page.locator("#galleryOverlay")).toBeVisible();
    await expect(page.locator(".g-card")).toHaveCount(ported.length);
    // Every card carries its drawing.
    await expect(page.locator(".g-card svg path.glass")).toHaveCount(ported.length);
    for (const d of refused) await expect(page.locator("#galleryOut")).toContainText(d.name);

    // A category narrows the list, and All brings it back.
    const infrared = ported.filter((d) => d.category === "Infrared").length;
    expect(infrared).toBeGreaterThan(0);
    await page.locator("#galleryCats button", { hasText: "Infrared" }).click();
    await expect(page.locator(".g-card")).toHaveCount(infrared);
    await page.locator("#galleryCats button", { hasText: "All" }).click();
    await expect(page.locator(".g-card")).toHaveCount(ported.length);

    // Escape closes it, and a key pressed while it was up reached nothing
    // behind it.
    await page.keyboard.press("Escape");
    await expect(page.locator("#galleryOverlay")).toBeHidden();
  });

  test("a card opens its design, framed, with its light on the image", async ({ page }) => {
    await ready(page);
    await page.locator("#galleryOpen").click();
    await page.locator(".g-card", { hasText: "Cooke triplet" }).click();
    await expect(page.locator("#galleryOverlay")).toBeHidden();
    await expect(page.locator("#preview3dMsg")).toContainText("opened Cooke triplet");
    await expect(page.locator("#sceneMsg")).toContainText("optiland.samples.objectives.CookeTriplet");

    const doc = await page.evaluate(() => {
      const scene = (window as unknown as { __ocDebug: { scene: any } }).__ocDebug.scene;
      return JSON.parse(scene.scene3JSON());
    });
    // Three singlets, the stop, a beam per field, and the image plane.
    expect(doc.bodies.length).toBe(3);
    expect(doc.apertures.length).toBe(1);
    expect(doc.sources.length).toBe(3);
    expect(doc.detectors.length).toBe(1);

    // The view frames the lens: its section spans a good part of the window
    // and lies inside it.
    const frame = await page.evaluate(() => {
      const v = (window as unknown as { __ocDebug: { view2d: () => any } }).__ocDebug.view2d();
      return { ...v, w: window.innerWidth, h: window.innerHeight };
    });
    const [x0, y0, x1, y1] = frame.extent;
    const sx = (x: number) => frame.w / 2 + frame.panX + x * frame.scale;
    const sy = (y: number) => frame.h / 2 + frame.panY - y * frame.scale;
    expect(sx(x0)).toBeGreaterThanOrEqual(0);
    expect(sx(x1)).toBeLessThanOrEqual(frame.w);
    expect(sy(y1)).toBeGreaterThanOrEqual(0);
    expect(sy(y0)).toBeLessThanOrEqual(frame.h);
    expect(sx(x1) - sx(x0)).toBeGreaterThan(0.4 * frame.w);

    // The exact trace lands the light of all three fields on the image.
    const { total } = await imageFlux(page);
    expect(total).toBeGreaterThan(0.5 * 3);
  });

  test("a link opens a design directly", async ({ page }) => {
    await ready(page, "&gallery=double-gauss");
    await expect(page.locator("#preview3dMsg")).toContainText("opened Double Gauss");
    await expect(page.locator("#galleryOverlay")).toBeHidden();
  });

  test("a link to a design the canvas cannot build says why", async ({ page }) => {
    const hubble = refused.find((d) => d.id === "hubble-telescope");
    test.skip(!hubble, "the Hubble telescope is in the canvas now");
    await ready(page, "&gallery=hubble-telescope");
    await expect(page.locator("#galleryOverlay")).toBeVisible();
    await expect(page.locator("#galleryErr")).toContainText("not in the canvas yet");
    await expect(page.locator("#galleryErr")).toContainText("mirror");
  });

  // Every design, through the app: it opens, and the app's own exact trace,
  // at the depth the app sizes to the scene, carries light to the image. A
  // 21-element lithography lens needs 43 interactions a ray, where the app
  // once stopped at 24 and drew its rays dying inside the lens.
  test("every design opens, and its light reaches its image", async ({ page }) => {
    test.setTimeout(180_000);
    await ready(page);
    for (const d of ported) {
      await page.evaluate(() => {
        document.getElementById("preview3dMsg")!.textContent = "";
      });
      await page.locator("#galleryOpen").click();
      await page.locator(`.g-card[data-design="${d.id}"]`).click();
      await expect(page.locator("#preview3dMsg"), d.id).toContainText(`opened ${d.name}`);
      const { total, sources } = await imageFlux(page);
      expect(total, `${d.id}: flux on the image`).toBeGreaterThan(0.02 * sources);
    }
  });
});
