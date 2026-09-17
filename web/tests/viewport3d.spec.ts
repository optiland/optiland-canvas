// The 3D model view (§5, always-3D). What only a browser can prove: the 2D⇄3D
// toggle shows the *same* Scene3 through an orbit camera, the real 3D ray
// bundle paints, and orbiting changes the picture. The view draws on the GPU
// canvas under the #view overlay, so the checks wait for its renderer (which
// takes seconds to start on software GL) and read the picture through the
// `?debug` hooks.

import { test, expect, type Page } from "@playwright/test";

const ready = (page: Page) => page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.render3d.ready());
const drawn = (page: Page) => page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.render3d.renderNow());
const litGpu = (page: Page) => page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.render3d.litPixels() as Promise<number>);

test.describe("3D model view — orbit camera over the same Scene3", () => {
  test("toggling to 3D re-projects the scene and paints the ray bundle", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");

    // Boots into the 2D cross-section; the toggle sits in the view bar.
    await expect(page.locator("#view-2d")).toHaveClass(/active/);
    await expect(page.locator("#view-3d")).not.toHaveClass(/active/);

    await page.locator("#view-3d").click();
    await expect(page.locator("#view-3d")).toHaveClass(/active/);
    await expect(page.locator("#stat")).toContainText("3D model");

    // The authoritative f64 3D bundle paints once the renderer is up.
    await ready(page);
    expect(await litGpu(page)).toBeGreaterThan(200);
    // The inspector lists the singlet and the default detector, re-read in 3D,
    // plus the light source, which only the 3D view draws.
    await expect(page.locator("#objects3d .row")).toHaveCount(3);
  });

  test("orbiting the camera changes the picture", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#view-3d").click();
    await expect(page.locator("#stat")).toContainText("3D model");
    await ready(page);
    await drawn(page);

    const before = await frameHash(page);
    // Drag on empty space to orbit — bottom-centre, clear of the side panels and
    // away from the body at the view centre (so it orbits, not drags the lens).
    const box = (await page.locator("#view").boundingBox())!;
    const sx = box.x + box.width / 2, sy = box.y + box.height - 120;
    await page.mouse.move(sx, sy);
    await page.mouse.down();
    await page.mouse.move(sx + 220, sy - 90, { steps: 12 });
    await page.mouse.up();
    await drawn(page);

    expect(await litGpu(page)).toBeGreaterThan(100);
    expect(await frameHash(page)).not.toBe(before); // the view actually moved
  });

  test("toggling back to 2D restores the cross-section view", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#view-3d").click();
    await expect(page.locator("#view-3d")).toHaveClass(/active/);
    await page.locator("#view-2d").click();
    await expect(page.locator("#view-2d")).toHaveClass(/active/);
    await expect(page.locator("#stat")).not.toContainText("3D model");
    expect(await litPixels(page)).toBeGreaterThan(200);
  });
});

/// Count perceptibly-lit pixels on the main view canvas (background is near-black):
/// the whole 2D picture, or the 3D view's overlay (axis labels, markers).
async function litPixels(page: import("@playwright/test").Page): Promise<number> {
  return page.evaluate(() => {
    const c = document.getElementById("view") as HTMLCanvasElement;
    const ctx = c.getContext("2d")!;
    const { data } = ctx.getImageData(0, 0, c.width, c.height);
    let n = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] + data[i + 1] + data[i + 2] > 60) n++;
    }
    return n;
  });
}

/// A cheap hash of the lit-pixel distribution, to tell whether the view changed.
async function frameHash(page: import("@playwright/test").Page): Promise<number> {
  return page.evaluate(() => {
    const c = document.getElementById("view") as HTMLCanvasElement;
    const ctx = c.getContext("2d")!;
    const { data, width } = ctx.getImageData(0, 0, c.width, c.height);
    let h = 0;
    for (let i = 0; i < data.length; i += 4) {
      const lum = data[i] + data[i + 1] + data[i + 2];
      if (lum > 60) { const p = i / 4; h = (h + (p % width) * 3 + ((p / width) | 0) * 7) % 2147483647; }
    }
    return h;
  });
}
