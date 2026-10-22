// The taper editor (web/src/inset/taper-editor.ts) and a body's glass by name.
//
// What only a browser can show: a taper opens its own editor in the component
// view, and its fields, handles and buttons reshape the body through Rust
// (`setTaper`); typing an index replaces a catalog glass with an Abbe point,
// and a preview over the Abbe map puts the catalog glass back exactly.

import { test, expect, type Page } from "@playwright/test";

test.describe.configure({ timeout: 180_000 });

type Pt = { key: string; x: number; y: number };
const view = (page: Page) =>
  page.evaluate(() => (window as any).__ocDebug.inset.view()) as Promise<{ id: number; kind: string; note: string; handles: Pt[] }>;
const wall = (page: Page, id: number) =>
  page.evaluate((i) => JSON.parse((window as any).__ocDebug.scene.taperJSON(i))?.wall as [number, number][], id);
const glass = (page: Page, id: number) =>
  page.evaluate((i) => JSON.parse((window as any).__ocDebug.scene.bodyIndexJSON(i)), id);
const material = (page: Page, id: number) =>
  page.evaluate((i) => JSON.parse((window as any).__ocDebug.scene.bodyMaterialJSON(i)), id);
const field = (page: Page, name: string) => page.locator(`#componentInset input[data-name="${name}"]`);

/** The boot scene with its lens swapped for a PMMA taper (Ø 4 > 12, L 40). */
async function openTaper(page: Page, inset = true): Promise<number> {
  await page.goto("/?debug");
  await page.evaluate(() => { localStorage.clear(); localStorage.setItem("optiland-canvas:tools", "open"); });
  await page.reload();
  await expect(page.locator("#stat")).toContainText("segs");
  const { json, id } = await page.evaluate(() => {
    const doc = JSON.parse((window as any).__ocDebug.scene.scene3JSON());
    let next = doc.next_id;
    const line = (a: number[], b: number[]) => ({ patch: next++, curve: { Line: { a, b } } });
    const id = next++;
    const edges = [line([0, -2], [0, 2]), line([0, 2], [40, 6]), line([40, 6], [40, -6]), line([40, -6], [0, -2])];
    doc.catalog.glasses.push({
      name: "PMMA",
      sell: { b: [1.19195585, 0.0262462388, 0.0], c: [0.0101775815, 18.195663, 0.0] },
      nd: 1.492533, vd: 58.68,
    });
    doc.bodies = [{
      common: { id, transform: { rot_xyzw: [0, 0, 0, 1], pos: [-20, 0, 0] }, layer: 0, locked: false, label: "taper", visible: true },
      geometry: { Profile: { profile: { edges }, construction: "RevolveLocalX" } },
      material: { Catalog: doc.catalog.glasses.length - 1 },
      absorption: "None",
      surface: { default: { from_outside: { DielectricFresnel: { ar_residual: { Constant: 1.0 } } }, from_inside: { DielectricFresnel: { ar_residual: { Constant: 1.0 } } } }, overrides: [] },
      priority: 0,
    }];
    doc.next_id = next;
    return { json: JSON.stringify(doc), id };
  });
  await page.locator("#open3dFile").setInputFiles({ name: "taper.ocanvas", mimeType: "application/json", buffer: Buffer.from(json) });
  await expect.poll(() => wall(page, id)).toEqual([[0, 2], [40, 6]]);
  await page.evaluate((i) => (window as any).__ocDebug.select(i), id);
  if (!inset) return id;
  if (!(await page.locator("#componentInset").isVisible())) await page.locator("#insetTab").click();
  await expect.poll(async () => (await view(page)).kind).toBe("taper");
  return id;
}

test.describe("the taper editor", () => {
  test("fields set the diameters, the length and the wall angle", async ({ page }) => {
    const id = await openTaper(page);
    await expect(field(page, "dIn")).toHaveValue("4");
    await expect(field(page, "dOut")).toHaveValue("12");
    await expect(field(page, "length")).toHaveValue("40");
    await expect(field(page, "angle")).toHaveValue(String(Number(((Math.atan2(4, 40) * 180) / Math.PI).toFixed(4))));

    await field(page, "dOut").fill("16");
    await expect.poll(() => wall(page, id)).toEqual([[0, 2], [40, 8]]);
    await field(page, "length").fill("60");
    await expect.poll(() => wall(page, id)).toEqual([[0, 2], [60, 8]]);
    // The angle keeps the input and the length: 0 is a rod of the input's size.
    await field(page, "angle").fill("0");
    await expect.poll(async () => (await wall(page, id))[1][1]).toBeCloseTo(2, 9);
    await field(page, "dIn").fill("5");
    await expect.poll(async () => (await wall(page, id))[0]).toEqual([0, 2.5]);
    // A wall that dips to the axis is refused, and says so.
    await field(page, "dOut").fill("0");
    await expect.poll(async () => (await view(page)).note).toContain("refused");
    expect((await wall(page, id))[1][1]).toBeCloseTo(2, 9);
  });

  test("points bend the wall, a handle drags one, straighten takes them out", async ({ page }) => {
    const id = await openTaper(page);
    await page.locator("#componentInset button", { hasText: "＋ point" }).click();
    await expect.poll(() => wall(page, id)).toEqual([[0, 2], [20, 4], [40, 6]]);
    const h = (await view(page)).handles.find((x) => x.key === "p1")!;
    await page.mouse.move(h.x, h.y);
    await page.mouse.down();
    await page.mouse.move(h.x, h.y - 30, { steps: 5 });
    await page.mouse.up();
    const bent = await wall(page, id);
    expect(bent.length).toBe(3);
    expect(bent[1][1]).toBeGreaterThan(4.5);
    expect(bent[0]).toEqual([0, 2]);
    expect(bent[2]).toEqual([40, 6]);
    await page.locator("#componentInset button", { hasText: "straighten" }).click();
    await expect.poll(() => wall(page, id)).toEqual([[0, 2], [40, 6]]);
  });

  test("the index shows a catalog glass by name; typing one makes an Abbe point", async ({ page }) => {
    const id = await openTaper(page);
    const pmma = await glass(page, id);
    expect(pmma).toMatchObject({ name: "PMMA" });
    await expect(page.locator("#componentInset")).toContainText("PMMA (catalog)");
    await field(page, "nd").fill("1.585");
    await expect.poll(() => glass(page, id)).toMatchObject({ nd: 1.585, name: null });
    expect(await material(page, id)).toEqual({ AbbePoint: { nd: 1.585, vd: pmma.vd } });
  });

  test("a preview over the Abbe map gives the catalog glass back", async ({ page }) => {
    const id = await openTaper(page, false);
    const before = await material(page, id);
    expect(before).toHaveProperty("Catalog");
    const map = (await page.locator("#abbe").boundingBox())!;
    // Across the map until the pointer is over a glass that exists.
    search: for (const fy of [0.5, 0.35, 0.65, 0.2, 0.8]) {
      for (const fx of [0.5, 0.35, 0.65, 0.2, 0.8]) {
        await page.mouse.move(map.x + fx * map.width, map.y + fy * map.height, { steps: 2 });
        if (!(await page.locator("#abbeLabel").textContent())?.includes("unobtainable")) break search;
      }
    }
    await expect.poll(() => material(page, id)).toHaveProperty("AbbePoint");
    await page.mouse.move(map.x - 40, map.y - 40);
    await expect.poll(() => material(page, id)).toEqual(before);
  });
});
