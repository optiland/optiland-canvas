// Scene authoring (§6.6, §5.3): the light-source picker and the draw-mirror tool.
// Both mutate the one Scene3 and re-trace, so the picture and the object list
// respond. Exercised through the page's own controls and canvas.

import { test, expect } from "@playwright/test";

test.describe("light source picker", () => {
  test("boots as a collimated rectangle and switches to a point source", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#srcType")).toHaveValue("collimated_rect");

    // Switch to a point source: it re-traces and still forms an image on the
    // detector (the default 5° cone from the object plane focuses through the lens).
    await page.locator("#srcType").selectOption("point");
    await expect(page.locator("#srcL0")).toHaveText("cone");
    await expect(page.locator("#dockNums")).toContainText("hits");
    expect(await litPixels(page)).toBeGreaterThan(150);
  });

  test("a round collimated beam hides the second diameter field", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#srcRow1")).toBeVisible(); // rect has ⌀X and ⌀Y
    await page.locator("#srcType").selectOption("collimated_disc");
    await expect(page.locator("#srcRow1")).toBeHidden();
    expect(await litPixels(page)).toBeGreaterThan(150);
  });

  test("wavelength presets and a custom list drive the beam colours", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    // Boots white: a swatch per line, and the custom nm box is disabled.
    const swatches = page.locator("#wlSwatches span");
    await expect(page.locator("#wlPreset")).toHaveValue("white");
    await expect(swatches).toHaveCount(9);
    await expect(page.locator("#wlList")).toBeDisabled();

    // The F, d, C preset: three lines, still forming an image.
    await page.locator("#wlPreset").selectOption("fdc");
    await expect(swatches).toHaveCount(3);
    expect(await litPixels(page)).toBeGreaterThan(100);

    // A custom list, including an infrared line that must render yellow.
    await page.locator("#wlPreset").selectOption("custom");
    await expect(page.locator("#wlList")).toBeEnabled();
    await page.locator("#wlList").fill("532, 900");
    await page.locator("#wlList").dispatchEvent("input");
    await expect(swatches).toHaveCount(2);
    // The 900 nm (IR) swatch is the fixed yellow the renderer uses out of band.
    const ir = swatches.nth(1);
    await expect(ir).toHaveAttribute("title", "900 nm");
    await expect(ir).toHaveCSS("background-color", "rgb(255, 230, 0)");
    expect(await litPixels(page)).toBeGreaterThan(100);
  });
});

test.describe("draw-mirror tool", () => {
  test("arming and dragging lays a selectable fold mirror", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#objects3d .row")).toHaveCount(2); // the singlet + the default detector

    await page.locator("#drawMirror").click();
    await expect(page.locator("#drawMirror")).toHaveClass(/active/);

    // Drag a vertical segment on empty canvas to the right of the lens.
    const box = (await page.locator("#view").boundingBox())!;
    const x = box.x + box.width / 2 + 150;
    const cy = box.y + box.height / 2;
    await page.mouse.move(x, cy - 60);
    await page.mouse.down();
    await page.mouse.move(x, cy + 60, { steps: 8 });
    await page.mouse.up();

    // The mirror is added and listed as a (non-body) surface; the tool disarms.
    await expect(page.locator("#objects3d .row")).toHaveCount(3);
    await expect(page.locator("#objects3d")).toContainText("mirror");
    await expect(page.locator("#drawMirror")).not.toHaveClass(/active/);
  });

  test("＋ element adds a configurable surface with per-face responses", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#objects3d .row")).toHaveCount(2);

    await page.locator("#addElement").click();
    await expect(page.locator("#objects3d .row")).toHaveCount(3);

    // Its card opens with the surface response block (front/back), no singlet fields.
    const card = page.locator(".card");
    await expect(card.locator(".c3-surface")).toBeVisible();
    await expect(card.locator(".c3-singlet")).toBeHidden();
    await expect(card.locator('select[data-name="front"]')).toHaveValue("mirror");

    // Change a face to absorbing; the scene re-traces (still has lit pixels).
    await card.locator('select[data-name="back"]').selectOption("absorb");
    await expect(card.locator('select[data-name="back"]')).toHaveValue("absorb");
    expect(await litPixels(page)).toBeGreaterThan(100);
  });

  test("＋ aperture adds a circular stop", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#objects3d .row")).toHaveCount(2);

    await page.locator("#addAperture").click();
    await expect(page.locator("#objects3d .row")).toHaveCount(3);
    await expect(page.locator("#objects3d")).toContainText("aperture");
    // Its card shows position/tilt but no per-face response block (not a surface).
    await expect(page.locator(".card .c3-surface")).toBeHidden();
    expect(await litPixels(page)).toBeGreaterThan(100);
  });

  test("＋ prism adds a dispersing glass body", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#objects3d .row")).toHaveCount(2);

    await page.locator("#addPrism").click();
    await expect(page.locator("#objects3d .row")).toHaveCount(3);
    await expect(page.locator("#objects3d")).toContainText("extruded");
    // A prism is a glass body: its card carries nd/Vd, not the singlet radii.
    const card = page.locator(".card");
    await expect(card.locator('input[data-name="nd"]')).toHaveValue(/1\.6/);
    await expect(card.locator('input[data-name="r1"]')).toBeHidden();
    expect(await litPixels(page)).toBeGreaterThan(100);
  });

  test("＋ detector adds a second, movable/tiltable detector", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    // The default scene already has one detector; add a second.
    await expect(page.locator("#objects3d")).toContainText("detector");
    await expect(page.locator("#objects3d .row")).toHaveCount(2);

    await page.locator("#addDetector").click();
    await expect(page.locator("#objects3d .row")).toHaveCount(3);

    // Its card offers position + tilt, but no per-face response block (not a surface)
    // and no singlet geometry — a detector is neither.
    const card = page.locator(".card");
    await expect(card.locator('input[data-name="x"]')).toHaveValue(/\d/);
    await expect(card.locator('input[data-name="rot"]')).toBeVisible();
    await expect(card.locator(".c3-surface")).toBeHidden();
    await expect(card.locator(".c3-singlet")).toBeHidden();

    // Tilt it via the card: the edit reaches the scene and re-traces.
    const rot = card.locator('input[data-name="rot"]');
    await rot.fill("15");
    await rot.dispatchEvent("input");
    await expect(rot).toHaveValue("15");
    expect(await litPixels(page)).toBeGreaterThan(100);
  });
});

test.describe("3D view controls", () => {
  test("reset, presets, and perspective change the 3D view", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#view-3d").click();
    await expect(page.locator("#view3dctl")).toBeVisible();

    const iso = await frameHash(page);
    await page.locator("#v3dFront").click(); // meridional face-on: a different frame
    expect(await frameHash(page)).not.toBe(iso);

    await page.locator("#v3dReset").click(); // back to the default 3/4
    await page.locator("#v3dPersp").click(); // perspective on
    await expect(page.locator("#v3dPersp")).toHaveClass(/active/);
    expect(await litPixels(page)).toBeGreaterThan(150);
  });
});

async function frameHash(page: import("@playwright/test").Page): Promise<number> {
  return page.evaluate(() => {
    const c = document.getElementById("view") as HTMLCanvasElement;
    const ctx = c.getContext("2d")!;
    const { data, width } = ctx.getImageData(0, 0, c.width, c.height);
    let h = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] + data[i + 1] + data[i + 2] > 60) { const p = i / 4; h = (h + (p % width) * 3 + ((p / width) | 0) * 7) % 2147483647; }
    }
    return h;
  });
}

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
