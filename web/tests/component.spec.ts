// The component view (web/src/inset) and the beam splitter and dichroic plates.
//
// What only a browser can prove: the tab opens the inset in place or where it
// is dropped, the window remembers where it was, and dragging its handles
// edits the scene through Rust. The optical checks read the f64 detector
// readout, so a handle drag is held to the tracer's own numbers.

import { test, expect, type Page } from "@playwright/test";

// The 3D test renders on software GL in CI.
test.describe.configure({ timeout: 180_000 });

type Pt = { key: string; x: number; y: number };
type InsetView = { id: number; kind: string; note: string; handles: Pt[] };

async function openApp(page: Page) {
  await page.goto("/?debug");
  await expect(page.locator("#stat")).toContainText("segs");
}

/** A clean browser profile: no remembered window or editor mode. */
async function freshApp(page: Page) {
  await openApp(page);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.locator("#stat")).toContainText("segs");
}

const insetView = (page: Page) =>
  page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.inset.view()) as Promise<InsetView>;

async function handle(page: Page, key: string): Promise<Pt> {
  const h = (await insetView(page)).handles.find((x) => x.key === key);
  expect(h, `handle ${key}`).toBeTruthy();
  return h!;
}

async function dragBy(page: Page, h: Pt, dx: number, dy: number, steps = 6) {
  await page.mouse.move(h.x, h.y);
  await page.mouse.down();
  await page.mouse.move(h.x + dx, h.y + dy, { steps });
  await page.mouse.up();
}

const shape = (page: Page) =>
  page.evaluate(() => {
    const d = (window as unknown as { __ocDebug: any }).__ocDebug;
    return JSON.parse(d.scene.singletShapeJSON(d.inset.view().id, 4));
  });

/** Open the inset with a click on the tab. */
async function openInset(page: Page) {
  if (!(await page.locator("#componentInset").isVisible())) await page.locator("#insetTab").click();
  await expect(page.locator("#componentInset")).toBeVisible();
}

/** A detector above x = -35 that catches what a 135° plate there folds up. */
const addUpDetector = (page: Page) =>
  page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.scene.addAreaDetector(-35, 40, 20, 90) as number);

const incident = (page: Page, id: number) =>
  page.evaluate((id) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.detectorResultJSON(id)).totalIncident as number, id);

test.describe("component view", () => {
  test("the tab opens the inset in place or where it is dropped, and it is remembered", async ({ page }) => {
    await freshApp(page);
    const frame = page.locator("#componentInset");
    const tab = page.locator("#insetTab");
    await expect(frame).toBeHidden();

    // A click opens it in place, a second click closes it.
    await tab.click();
    await expect(frame).toBeVisible();
    await expect(tab).toHaveClass(/active/);
    await expect(frame.locator(".inset-sub")).toContainText("singlet");
    await tab.click();
    await expect(frame).toBeHidden();
    await expect(tab).not.toHaveClass(/active/);

    // Dragged out and dropped on the canvas, it opens where it was dropped.
    const t = (await tab.boundingBox())!;
    await page.mouse.move(t.x + t.width / 2, t.y + t.height / 2);
    await page.mouse.down();
    await page.mouse.move(500, 300, { steps: 6 });
    await expect(page.locator("#insetGhost")).toBeVisible();
    await page.mouse.move(640, 330, { steps: 4 });
    await page.mouse.up();
    await expect(page.locator("#insetGhost")).toBeHidden();
    await expect(frame).toBeVisible();
    let box = (await frame.boundingBox())!;
    expect(Math.abs(box.x - (640 - 28))).toBeLessThanOrEqual(1);
    expect(Math.abs(box.y - (330 - 12))).toBeLessThanOrEqual(1);

    // A reload brings it back in the same place.
    await page.reload();
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(frame).toBeVisible();
    box = (await frame.boundingBox())!;
    expect(Math.abs(box.x - (640 - 28))).toBeLessThanOrEqual(1);

    // Dragging the header back onto the top bar closes it, and that is
    // remembered too.
    const head = (await frame.locator(".inset-head").boundingBox())!;
    const bar = (await page.locator("#tools").boundingBox())!;
    await page.mouse.move(head.x + 60, head.y + head.height / 2);
    await page.mouse.down();
    await page.mouse.move(bar.x + bar.width / 2, bar.y + bar.height / 2, { steps: 8 });
    await expect(tab).toHaveClass(/dock-target/);
    await page.mouse.up();
    await expect(frame).toBeHidden();
    await expect(tab).not.toHaveClass(/dock-target|active/);
    await page.reload();
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(frame).toBeHidden();
  });

  test("lens handles bend, thicken, and reshape the singlet through Rust", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    expect((await insetView(page)).kind).toBe("lens");
    const s0 = await shape(page);

    // Bend: pull the front rim outward, the face gets steeper.
    await dragBy(page, await handle(page, "b0"), 6, 0);
    const s1 = await shape(page);
    expect(s1.r1).toBeLessThan(s0.r1);
    expect(s1.r1).toBeGreaterThan(0);

    // Thickness: pull the front vertex away; the back face stays in place.
    const pos = () => page.evaluate(() => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      return (JSON.parse(d.scene.sceneObjectsJSON()) as { id: number; pos: number[] }[]).find((o) => o.id === d.inset.view().id)!.pos;
    });
    const p1 = await pos();
    await dragBy(page, await handle(page, "v0"), -20, 0);
    const s2 = await shape(page);
    const p2 = await pos();
    expect(s2.thickness).toBeGreaterThan(s1.thickness + 0.3);
    expect(Math.abs(p2[0] + s2.thickness / 2 - (p1[0] + s1.thickness / 2))).toBeLessThan(1e-9);

    // A thickness the faces cannot take is refused and said so.
    await dragBy(page, await handle(page, "v1"), -400, 0, 10);
    expect((await insetView(page)).note).toContain("refused");
    expect((await shape(page)).thickness).toBeGreaterThan(0);

    // Asphere mode: a control point reshapes the face through all four.
    await page.locator("#componentInset button", { hasText: "asphere" }).click();
    const hs = [0.25, 0.5, 0.75, 1].map((f) => f * s2.semiAperture);
    const sags = () => page.evaluate((hs) => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      return Array.from(d.scene.singletFaceSagF64(d.inset.view().id, false, new Float64Array(hs), false)) as number[];
    }, hs);
    const before = await sags();
    const trace0 = await page.evaluate(() => Array.from((window as unknown as { __ocDebug: any }).__ocDebug.exactProjected() as Float32Array).slice(0, 400));
    await dragBy(page, await handle(page, "c0:1"), 10, 0);
    const after = await sags();
    const s3 = await shape(page);
    expect(s3.asph1.some((a: number) => a !== 0)).toBe(true);
    expect(after[1]).toBeGreaterThan(before[1] + 0.05);
    for (const i of [0, 2, 3]) expect(Math.abs(after[i] - before[i])).toBeLessThan(1e-9);
    const trace1 = await page.evaluate(() => Array.from((window as unknown as { __ocDebug: any }).__ocDebug.exactProjected() as Float32Array).slice(0, 400));
    expect(trace1).not.toEqual(trace0);
    // The mode is remembered for the next lens.
    expect(await page.evaluate(() => localStorage.getItem("optiland-canvas:lens-editor-mode"))).toBe("asphere");

    // Fields write through too, and "clear terms" removes the terms.
    await page.locator('#componentInset input[data-name="k"]').fill("-1");
    await expect.poll(async () => (await shape(page)).conic1).toBe(-1);
    await page.locator("#componentInset button", { hasText: "clear terms" }).click();
    expect((await shape(page)).asph1.every((a: number) => a === 0)).toBe(true);

    // The inset follows the selection: nothing selected, nothing shown.
    // (Keys typed inside the inset stay there, so leave it first.)
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("Escape");
    await expect(page.locator("#componentInset .inset-sub")).toContainText("nothing selected");
  });

  test("a beam splitter divides the beam, and its handles set the split", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await page.locator("#addSplitter").click();
    await expect(page.locator("#componentInset .inset-sub")).toContainText("beamsplitter");
    expect((await insetView(page)).kind).toBe("coating");
    const up = await addUpDetector(page);
    // Every ray meets the plate: half the launched flux folds up.
    expect(await incident(page, up)).toBeCloseTo(0.5, 9);
    await expect(page.locator("#componentInset .cv-read")).toContainText("R 50%");

    const splitter = (await insetView(page)).id;
    const coat = () => page.evaluate((id) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.surfaceCoatingJSON(id)).front, splitter);

    // Hold the R handle lower: while it is held, the fast picture still shows
    // the folded arm (ghosts are off; the fold is not a ghost).
    const r = await handle(page, "r");
    await page.mouse.move(r.x, r.y);
    await page.mouse.down();
    await page.mouse.move(r.x, r.y + 40, { steps: 5 });
    const folded = await page.evaluate(async () => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      const segs = (await d.fastProjected()).segs as Float32Array;
      let n = 0;
      for (let i = 0; i + 11 <= segs.length; i += 11) {
        if (segs[i + 3] - segs[i + 1] > 5 && Math.abs(segs[i + 2] - segs[i]) < 0.01) n++;
      }
      return n;
    });
    expect(folded).toBeGreaterThan(0);
    await page.mouse.up();

    const c = await coat();
    expect(c.r).toBeLessThan(0.45);
    expect(c.r + c.t).toBeCloseTo(1, 12);
    expect(await incident(page, up)).toBeCloseTo(c.r, 9);

    // The fields write the split too.
    await page.locator('#componentInset input[data-name="R"]').fill("30");
    await expect.poll(async () => (await coat()).r).toBeCloseTo(0.3, 12);
    expect(await incident(page, up)).toBeCloseTo(0.3, 9);
  });

  test("a dichroic is a real film: it routes the lines and knows its angle", async ({ page }) => {
    await freshApp(page);
    await page.selectOption("#wlPreset", "fdc");
    await openInset(page);
    await page.locator("#addDichroic").click();
    await expect(page.locator("#componentInset .inset-sub")).toContainText("dichroic");
    const id = (await insetView(page)).id;
    const up = await addUpDetector(page);
    const lines = [486.13, 587.56, 656.27];

    // The plate hangs at 135°, so the light meets it at 45°, and that is the
    // angle its stack was solved for and the angle its curve is plotted at.
    const film = () => page.evaluate(
      (id) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.surfaceFilmJSON(id)),
      id,
    );
    const f0 = await film();
    expect(f0.aoiDeg).toBeCloseTo(45, 6);
    expect(f0.layers.length).toBeGreaterThan(10);
    expect(f0.edgeNm).toBeGreaterThan(545);
    expect(f0.edgeNm).toBeLessThan(555);
    await expect(page.locator("#componentInset .cv-read")).toContainText("used at");

    // What the up arm should get: each line's reflectance at the angle the
    // plate is used at, times its third of the flux. The 550 nm long-pass
    // reflects the blue line and passes the other two.
    const expected = () => page.evaluate(([id, lines, aoi]) => {
      const rt = (window as unknown as { __ocDebug: any }).__ocDebug.scene
        .surfaceRtF64(id, false, new Float64Array(lines as number[]), aoi as number) as Float64Array;
      return (rt[0] + rt[2] + rt[4]) / 3;
    }, [id, lines, f0.aoiDeg] as const);
    expect(await expected()).toBeGreaterThan(0.3);
    expect(await incident(page, up)).toBeCloseTo(await expected(), 9);

    // The same stack read on axis is a different coating: its band sits some
    // 40 nm higher, so the yellow line it passes at 45° it reflects on axis.
    const onAxis = await page.evaluate(([id, nm]) => {
      const rt = (window as unknown as { __ocDebug: any }).__ocDebug.scene
        .surfaceRtF64(id, false, new Float64Array([nm as number]), 0) as Float64Array;
      return rt[0];
    }, [id, 587.56] as const);
    const atUse = await page.evaluate(([id, nm, aoi]) => {
      const rt = (window as unknown as { __ocDebug: any }).__ocDebug.scene
        .surfaceRtF64(id, false, new Float64Array([nm as number]), aoi as number) as Float64Array;
      return rt[0];
    }, [id, 587.56, f0.aoiDeg] as const);
    expect(onAxis).toBeGreaterThan(atUse + 0.5);

    // Dragging a layer boundary changes that layer and nothing to its left.
    const before = (await film()).layers.map((l: { thicknessNm: number }) => l.thicknessNm);
    await dragBy(page, await handle(page, "f0"), 14, 0);
    const after = (await film()).layers.map((l: { thicknessNm: number }) => l.thicknessNm);
    expect(after[0]).toBeGreaterThan(before[0] + 2);
    expect(after[1]).toBeCloseTo(before[1], 9);
    expect(await incident(page, up)).toBeCloseTo(await expected(), 9);

    // Asking for a different edge rebuilds the stack, and it lands there.
    await page.locator('#componentInset input[data-name="edge"]').fill("620");
    await expect.poll(async () => (await film()).edgeNm).toBeGreaterThan(612);
    await expect.poll(async () => (await film()).edgeNm).toBeLessThan(628);

    // Short-pass swaps which side reflects.
    await page.locator("#componentInset button", { hasText: "short-pass" }).click();
    await expect.poll(async () => (await film()).design.kind).toBe("short-pass");

    // The 2D section tints the plate by what it reflects.
    const tint = await page.evaluate(() => {
      const xs = JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.sceneCrossSectionJSON()) as { kind: string; tint?: string }[];
      return xs.find((p) => p.kind === "dichroic")?.tint ?? null;
    });
    expect(tint).toMatch(/^rgb\(/);
  });

  test("the inset follows the 3D view and edits the lens there", async ({ page }) => {
    await freshApp(page);
    await page.locator("#view-3d").click();
    await expect(page.locator("#stat")).toContainText("3D model");
    await page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.render3d.ready());
    await openInset(page);
    expect((await insetView(page)).kind).toBe("lens");
    const builds0 = (await page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.render3d.renderNow())).meshBuilds;
    const r0 = (await shape(page)).r2;
    await dragBy(page, await handle(page, "b1"), -6, 0);
    expect((await shape(page)).r2).not.toBe(r0);
    const stats = await page.evaluate(() => (window as unknown as { __ocDebug: any }).__ocDebug.render3d.renderNow());
    expect(stats.meshBuilds).toBeGreaterThan(builds0);

    // A splitter added in 3D shows as a coated plate there.
    await page.locator("#addSplitter").click();
    await expect(page.locator("#componentInset .inset-sub")).toContainText("beamsplitter");
    const role = await page.evaluate(() => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      const id = d.inset.view().id;
      return (JSON.parse(d.scene.renderObjectsJSON()) as { id: number; front: string }[]).find((o) => o.id === id)!.front;
    });
    expect(role).toBe("beamsplitter");
  });
});
