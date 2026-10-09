// The detector's component view beyond its face: cross-sections dragged
// across the picture, the colour panel, and tracing the numbers with more
// rays than the picture draws.
//
// The default scene is the case for all three: a white beam through a
// singlet onto a detector near its focus, so there is a spot to cut through
// and nine wavelengths that do not focus in the same place.

import { test, expect, type Page } from "@playwright/test";

test.describe.configure({ timeout: 180_000 });

type Pt = { key: string; x: number; y: number };
type Debug = {
  scene: {
    sceneObjectsJSON(): string;
    detectorResultJSON(id: number): string;
    analysisRays(): number;
    launchRayCounts(): Float64Array;
  };
  select(id: number): void;
  referenceSegments(): number;
  inset: { view(): { id: number; kind: string; handles: Pt[] } };
};

async function freshApp(page: Page) {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.evaluate(() => {
    localStorage.clear();
    localStorage.setItem("optiland-canvas:tools", "open");
  });
  await page.reload();
  await expect(page.locator("#stat")).toContainText("segs");
}

/** Open the inset on the default scene's detector. */
async function openOnDetector(page: Page): Promise<number> {
  const id = await page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: Debug }).__ocDebug;
    const det = (JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; kind: string }[]).find((o) => o.kind === "detector")!;
    d.select(det.id);
    return det.id;
  });
  if (!(await page.locator("#componentInset").isVisible())) await page.locator("#insetTab").click();
  await expect.poll(() => view(page).then((v) => v.kind)).toBe("detector");
  await expect(page.locator("#refStatus")).toContainText("reference current");
  return id;
}

const view = (page: Page) =>
  page.evaluate(() => (window as unknown as { __ocDebug: Debug }).__ocDebug.inset.view());
const bar = (page: Page) => page.locator("#componentInset .cv-bar");
const read = (page: Page) => page.locator("#componentInset .cv-read");

test("cross-sections open beside the picture and follow a dragged line", async ({ page }) => {
  await freshApp(page);
  await openOnDetector(page);
  const frame = page.locator("#componentInset");
  const narrow = (await frame.boundingBox())!.width;

  // Asking for cross-sections makes room for them.
  await bar(page).getByText("cross-sections", { exact: true }).click();
  await expect.poll(async () => (await frame.boundingBox())!.width).toBeGreaterThan(narrow + 300);
  // Two cuts through the brightest pixel, a row and a column, each with its
  // peak and its width at half the peak.
  await expect(read(page)).toContainText("row at v =");
  await expect(read(page)).toContainText("col at u =");
  await expect(read(page)).toContainText("FWHM");

  // Drag the row's line off the spot: its peak falls.
  const rowLine = () => read(page).locator("div", { hasText: "row at v =" });
  const before = (await rowLine().textContent()) ?? "";
  const grip = (await view(page)).handles.find((h) => h.key === "cut-0")!;
  expect(grip).toBeTruthy();
  // Grab it in the middle of the line, not at its grip: a line is held
  // anywhere along it.
  const mid = { x: grip.x + 60, y: grip.y };
  await page.mouse.move(mid.x, mid.y);
  await page.mouse.down();
  await page.mouse.move(mid.x, mid.y - 60, { steps: 6 });
  await page.mouse.up();
  const after = (await rowLine().textContent()) ?? "";
  expect(after).not.toBe(before);
  const v = (t: string) => Number(/v = (-?[\d.e+-]+)/.exec(t)?.[1]);
  expect(v(after)).toBeGreaterThan(v(before));

  // One more column; and the panel closes back to the width it took.
  await bar(page).getByText("+ col", { exact: true }).click();
  await expect(read(page).locator("div", { hasText: "col at u =" })).toHaveCount(2);
  await bar(page).getByText("nothing", { exact: true }).click();
  await expect.poll(async () => Math.round((await frame.boundingBox())!.width)).toBe(Math.round(narrow));
});

test("the far field is a grid with straight axes, and cuts through it read in degrees", async ({ page }) => {
  await freshApp(page);
  await openOnDetector(page);
  await bar(page).getByText("intensity", { exact: true }).click();
  await expect(bar(page).getByText("grid", { exact: true })).toHaveClass(/active/);
  await bar(page).getByText("cross-sections", { exact: true }).click();
  await expect(read(page)).toContainText("° · peak");
  await expect(read(page)).toContainText("flux/sr");
  // The polar map has no rows, and says what to do about it.
  await bar(page).getByText("polar", { exact: true }).click();
  await expect(read(page)).toContainText("cross-sections cut a grid");
});

test("the colour panel measures the singlet's axial colour", async ({ page }) => {
  await freshApp(page);
  const id = await openOnDetector(page);
  await bar(page).getByText("colour", { exact: true }).click();
  // A beam converging on the face: the panel opens on focus.
  await expect(bar(page).getByText("focus", { exact: true })).toHaveClass(/active/);
  await expect(read(page)).toContainText("axial colour: best focus spans");

  const lines = await page.evaluate(
    (i) => JSON.parse((window as unknown as { __ocDebug: Debug }).__ocDebug.scene.detectorResultJSON(i)).spectral,
    id,
  ) as { nm: number; focus: number }[];
  expect(lines.length).toBeGreaterThan(2);
  // N-BK7 bends blue more, so blue focuses first.
  for (let k = 1; k < lines.length; k++) expect(lines[k].focus).toBeGreaterThan(lines[k - 1].focus);
  const span = lines[lines.length - 1].focus - lines[0].focus;
  const shown = Number(/best focus spans ([\d.]+)/.exec((await read(page).textContent()) ?? "")?.[1]);
  expect(shown).toBeCloseTo(span, 3);

  // Each plot is a click away.
  for (const m of ["direction", "position", "spot"]) {
    await bar(page).getByText(m, { exact: true }).click();
    await expect(bar(page).getByText(m, { exact: true })).toHaveClass(/active/);
  }
});

test("more analysis rays sharpen the numbers and leave the picture alone", async ({ page }) => {
  await freshApp(page);
  await openOnDetector(page);
  const hits = async () => Number(/hits (\d+)/.exec((await page.locator("#dockNums").textContent()) ?? "")?.[1]);
  const segs = () => page.evaluate(() => (window as unknown as { __ocDebug: Debug }).__ocDebug.referenceSegments());
  const h1 = await hits();
  const s1 = await segs();
  expect(h1).toBeGreaterThan(0);

  // Any count, typed as the user thinks of it.
  const own = await page.evaluate(() => (window as unknown as { __ocDebug: Debug }).__ocDebug.scene.launchRayCounts()[0]);
  const field = bar(page).locator('input[data-name="analysisRays"]');
  await field.fill(`${(16 * own) / 1000}k`);
  await field.press("Enter");
  await expect.poll(() => page.evaluate(() => (window as unknown as { __ocDebug: Debug }).__ocDebug.scene.analysisRays())).toBe(16 * own);
  await expect.poll(hits, { timeout: 60_000 }).toBeGreaterThan(8 * h1);
  await expect(page.locator("#refStatus")).toContainText("reference current");
  expect(await segs()).toBe(s1);
  await expect(read(page)).toContainText("the picture keeps the ray density's");

  // Nonsense is refused and the count stays.
  await field.fill("lots");
  await field.press("Enter");
  expect(await page.evaluate(() => (window as unknown as { __ocDebug: Debug }).__ocDebug.scene.analysisRays())).toBe(16 * own);

  // A preset, and back to the ray density's own count.
  await bar(page).getByText("100k", { exact: true }).click();
  await expect(bar(page).getByText("100k", { exact: true })).toHaveClass(/active/);
  await expect(field).toHaveValue("100k");
  await bar(page).getByText("scene", { exact: true }).click();
  await expect(field).toHaveValue("");
  await expect.poll(hits, { timeout: 60_000 }).toBe(h1);
});
