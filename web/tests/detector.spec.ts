// Detector analysis modes (§6.9): the area detector shows a true-3D analysis,
// switchable between the 2D binning heatmap and the spot diagram (the actual
// ray-detector intersection points). Both are computed from the f64 3D trace of
// the default focusing singlet, so the dock canvas gains lit pixels in each mode.

import { test, expect } from "@playwright/test";

test.describe("detector — heatmap ⇄ spot diagram", () => {
  test("boots in heatmap mode and paints the binned flux", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#detHeatmap")).toHaveClass(/active/);
    await expect(page.locator("#detHead")).toContainText("heatmap");
    // The focused beam registers on the detector.
    await expect(page.locator("#dockNums")).toContainText("hits");
    expect(await litHist(page)).toBeGreaterThan(20);
  });

  test("switching to spot renders the ray-detector hits", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#detSpot").click();
    await expect(page.locator("#detSpot")).toHaveClass(/active/);
    await expect(page.locator("#detHead")).toContainText("spot diagram");
    expect(await litHist(page)).toBeGreaterThan(10);
    // And back to heatmap.
    await page.locator("#detHeatmap").click();
    await expect(page.locator("#detHeatmap")).toHaveClass(/active/);
    expect(await litHist(page)).toBeGreaterThan(20);
  });
});

/// Count perceptibly-lit pixels on the detector dock canvas.
async function litHist(page: import("@playwright/test").Page): Promise<number> {
  return page.evaluate(() => {
    const c = document.getElementById("hist") as HTMLCanvasElement;
    const ctx = c.getContext("2d")!;
    const { data } = ctx.getImageData(0, 0, c.width, c.height);
    let n = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] + data[i + 1] + data[i + 2] > 80) n++;
    }
    return n;
  });
}
