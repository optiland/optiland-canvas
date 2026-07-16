// The openUC2 mode's browser test (§19.5).
//
// The first-release success scenario (§3) driven the way a user drives it: place
// modules on the grid, rotate one, watch the light, save, reopen, export. It
// asserts through the UI's own panels rather than by reaching into the wasm
// handle, so it fails if the *page* stops working, not merely if a binding does.
//
// The Rust side already proves the physics and the round-trip headlessly
// (`crates/oc-openuc2/tests/folded_singlet.rs`). What only a browser can show is
// that the route wires them together: that a drag reaches `move_module`, that the
// f64 numbers settle after it, and that a downloaded file is the setup.

import { test, expect, type Page } from "@playwright/test";

/// Screen coordinates of a grid cell's centre.
///
/// Mirrors the page's own `toScreen` at its initial viewport: 3 px/mm, panned to
/// (0.25, 0.5). Read from the canvas rather than assumed, so a different window
/// size does not move the target.
async function cell(page: Page, gx: number, gy: number) {
  return page.evaluate(
    ([gx, gy]) => {
      const c = document.getElementById("view") as HTMLCanvasElement;
      const r = c.getBoundingClientRect();
      const scale = 3.0;
      return {
        x: r.left + (gx * 50 * scale + c.clientWidth * 0.25),
        y: r.top + (c.clientHeight * 0.5 - gy * 50 * scale),
      };
    },
    [gx, gy],
  );
}

/// Drag a library card onto a grid cell.
async function placeCard(page: Page, name: string, gx: number, gy: number) {
  const card = page.locator(".uc2-card", { hasText: name }).first();
  const target = await cell(page, gx, gy);

  // A drop onto a side panel is silently ignored, which shows up much later as a
  // wrong module count and says nothing about why. Fail here instead.
  const over = await page.evaluate(
    (p) => document.elementFromPoint(p.x, p.y)?.id ?? "",
    target,
  );
  expect(over, `cell [${gx},${gy}] is covered by another element, not the canvas`).toBe("view");

  await card.hover();
  await page.mouse.down();
  await page.mouse.move(target.x, target.y, { steps: 8 });
  await page.mouse.up();
}

/// How much of the canvas is lit. The rays are the only bright thing on it, so a
/// count well above zero means light is being drawn.
async function litPixels(page: Page): Promise<number> {
  return page.evaluate(() => {
    const c = document.getElementById("view") as HTMLCanvasElement;
    const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    let lit = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] > 40 || d[i + 1] > 40 || d[i + 2] > 40) lit++;
    }
    return lit;
  });
}

const moduleRows = (page: Page) => page.locator("#uc2Layout .uc2-row");

test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#openuc2");
  await expect(page.locator("#uc2Status")).toContainText("modules");
  (page as Page & { _errors?: string[] })._errors = errors;
});

test("the catalogue loads and says it is a fixture", async ({ page }) => {
  // Nobody should mistake the development fixture's invented part numbers and
  // prices for openUC2 products they can order (§5.4).
  await expect(page.locator("#uc2Src")).toContainText("development fixture");
  await expect(page.locator("#uc2List .uc2-card")).toHaveCount(5);
});

test("a layout-only module is visible and says why it adds no light", async ({ page }) => {
  // §15.4: incomplete modules are never hidden by default.
  const cube = page.locator(".uc2-card", { hasText: "Empty structural cube" });
  await expect(cube).toBeVisible();
  await expect(cube).toContainText("layout only");
});

test("drag four modules onto the grid, fold the beam, and read the f64 numbers", async ({
  page,
}) => {
  await placeCard(page, "Three-line collimated source", 0, 0);
  await placeCard(page, "N-BK7 singlet lens", 1, 0);
  await placeCard(page, "45 degree fold mirror", 2, 0);
  await placeCard(page, "Camera cross-section", 2, -1);
  await expect(moduleRows(page)).toHaveCount(4);

  // The camera is dropped unrotated, so its sensor lies along the folded beam
  // rather than across it. Turn it until it faces the mirror.
  const camera = await cell(page, 2, -1);
  await page.mouse.click(camera.x, camera.y);
  await page.keyboard.press("r");
  await page.keyboard.press("r");
  await page.keyboard.press("r");

  // Rays are on screen...
  expect(await litPixels(page)).toBeGreaterThan(1000);
  // ...and the numbers come from the f64 reference path, settled after the edit.
  await expect(page.locator("#uc2Nums")).toContainText("hits");
  await expect(page.locator("#uc2Nums")).not.toContainText("no detector hits");
  // Every port mates: the chain really is source -> lens -> mirror -> camera.
  await expect(page.locator("#uc2Ports")).toContainText("6 of 6 connected");
});

test("rotating the mirror moves every optical object the module owns", async ({ page }) => {
  await page.click("#uc2Demo");
  await expect(moduleRows(page)).toHaveCount(4);
  await expect(page.locator("#uc2Nums")).toContainText("hits");
  const before = await page.locator("#uc2Nums").innerText();

  const mirror = await cell(page, 2, 0);
  await page.mouse.click(mirror.x, mirror.y);
  await expect(page.locator(".uc2-row.uc2-sel")).toContainText("fold mirror");
  await page.keyboard.press("r");

  // The fold now sends the beam somewhere the camera is not.
  await expect(page.locator("#uc2Nums")).toContainText("no detector hits");
  // And one undo puts it back, exactly.
  await page.keyboard.press("Control+z");
  await expect(page.locator("#uc2Nums")).toHaveText(before);
});

test("a drag is one undo step and rays survive it", async ({ page }) => {
  await page.click("#uc2Demo");
  await expect(page.locator("#uc2Ports")).toContainText("6 of 6");

  const from = await cell(page, 1, 0);
  const to = await cell(page, 1, 1);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  for (let i = 1; i <= 6; i++) {
    await page.mouse.move(from.x + ((to.x - from.x) * i) / 6, from.y + ((to.y - from.y) * i) / 6);
  }
  // Mid-drag the rays are still being drawn, from the interactive f32 path.
  expect(await litPixels(page)).toBeGreaterThan(1000);
  await page.mouse.up();

  // The lens left the chain, so its ports no longer mate...
  await expect(page.locator("#uc2Ports")).not.toContainText("6 of 6");
  // ...and a single undo undoes the whole drag, not one pointer move of it.
  await page.keyboard.press("Control+z");
  await expect(page.locator("#uc2Ports")).toContainText("6 of 6");
});

test("a module cannot be dropped onto an occupied cell", async ({ page }) => {
  await page.click("#uc2Demo");
  await expect(moduleRows(page)).toHaveCount(4);

  const lens = await cell(page, 1, 0);
  const mirror = await cell(page, 2, 0);
  await page.mouse.move(lens.x, lens.y);
  await page.mouse.down();
  await page.mouse.move(mirror.x, mirror.y, { steps: 5 });
  await page.mouse.up();

  // Still four modules, and the lens did not land on the mirror: the chain is
  // intact, which it would not be if the move had gone through.
  await expect(moduleRows(page)).toHaveCount(4);
  await expect(page.locator("#uc2Ports")).toContainText("6 of 6 connected");
});

test("only the active layer is traced", async ({ page }) => {
  await page.click("#uc2Demo");
  await expect(page.locator("#uc2Nums")).toContainText("hits");

  await page.click("#uc2LayerUp");
  await expect(page.locator("#uc2LayerNum")).toHaveText("1");
  // Canvas traces one horizontal plane at a time (§5.6), and the demo is on 0.
  await expect(page.locator("#uc2Nums")).toContainText("no detector hits");
  await expect(page.locator("#uc2Warn")).toContainText("layer");

  await page.click("#uc2LayerDown");
  await expect(page.locator("#uc2Nums")).toContainText("hits");
});

test("the module list matches what is placed", async ({ page }) => {
  await page.click("#uc2Demo");
  await expect(moduleRows(page)).toHaveCount(4);

  const bom = page.locator("#uc2Bom");
  await expect(bom).toContainText("N-BK7 singlet lens module");
  await expect(bom).toContainText("Camera cross-section module");
  // 40 + 24 + 18 + 120, from the fixture's invented prices.
  await expect(bom).toContainText("€202.00");

  // Delete one and the list follows.
  const lens = await cell(page, 1, 0);
  await page.mouse.click(lens.x, lens.y);
  await page.keyboard.press("Delete");
  await expect(moduleRows(page)).toHaveCount(3);
  await expect(bom).toContainText("€178.00");
});

test("save a setup, reopen it, and get the same project back", async ({ page }) => {
  await page.click("#uc2Demo");
  await expect(moduleRows(page)).toHaveCount(4);
  const numbers = await page.locator("#uc2Nums").innerText();

  const saved = page.waitForEvent("download");
  await page.click("#uc2Save");
  const file = await saved;
  const stream = await file.createReadStream();
  const body = await new Promise<string>((resolve, reject) => {
    let s = "";
    stream.on("data", (c) => (s += c));
    stream.on("end", () => resolve(s));
    stream.on("error", reject);
  });
  expect(file.suggestedFilename()).toMatch(/\.uc2canvas$/);
  expect(JSON.parse(body).format).toBe("openuc2-canvas-setup");
  expect(JSON.parse(body).instances).toHaveLength(4);

  // Clear the board, then open the file back.
  for (let i = 0; i < 4; i++) {
    await page.locator("#uc2Layout .uc2-row button[title='remove (Del)']").first().click();
  }
  await expect(moduleRows(page)).toHaveCount(0);

  await page.locator("#uc2File").setInputFiles({
    name: "demo.uc2canvas",
    mimeType: "application/json",
    buffer: Buffer.from(body),
  });

  // The reopened setup is not merely similar: the scene is derived from it, so
  // the reference numbers come back identical.
  await expect(moduleRows(page)).toHaveCount(4);
  await expect(page.locator("#uc2Nums")).toHaveText(numbers);
  await expect(page.locator("#uc2Bom")).toContainText("€202.00");
});

test("exporting .ocanvas hands over a scene and says what it cannot carry", async ({ page }) => {
  await page.click("#uc2Demo");
  await expect(moduleRows(page)).toHaveCount(4);

  const started = page.waitForEvent("download");
  await page.click("#uc2ExportOcanvas");
  const file = await started;
  expect(file.suggestedFilename()).toMatch(/\.ocanvas$/);

  const stream = await file.createReadStream();
  const body = await new Promise<string>((resolve, reject) => {
    let s = "";
    stream.on("data", (c) => (s += c));
    stream.on("end", () => resolve(s));
    stream.on("error", reject);
  });
  // A real .ocanvas: the generic scene format, carrying the optics and none of
  // the product identity (§12.4).
  const scene = JSON.parse(body);
  expect(scene.elements).toHaveLength(2);
  expect(scene.detectors).toHaveLength(1);
  expect(JSON.stringify(scene)).not.toContain("EXAMPLE-LENS-01");

  // §15.9: the fidelity report is shown *with* the export, because the file has
  // nowhere to put it. The source and the camera are idealized.
  await expect(page.locator("#uc2Warn")).toContainText("fidelity caveat");
  await expect(page.locator("#uc2Warn")).toContainText("idealized");
});

test("the module list downloads as CSV", async ({ page }) => {
  await page.click("#uc2Demo");
  const started = page.waitForEvent("download");
  await page.click("#uc2BomCsv");
  const file = await started;
  expect(file.suggestedFilename()).toMatch(/-bom\.csv$/);

  const stream = await file.createReadStream();
  const body = await new Promise<string>((resolve, reject) => {
    let s = "";
    stream.on("data", (c) => (s += c));
    stream.on("end", () => resolve(s));
    stream.on("error", reject);
  });
  expect(body.split("\n")[0]).toContain("quantity,name,module_id");
  expect(body).toContain("EXAMPLE-LENS-01");
});

test.afterEach(async ({ page }) => {
  const errors = (page as Page & { _errors?: string[] })._errors ?? [];
  expect(errors, `uncaught page errors: ${errors.join(" | ")}`).toHaveLength(0);
});
