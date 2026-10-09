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
  await page.evaluate(() => {
    localStorage.clear();
    localStorage.setItem("optiland-canvas:tools", "open");
  });
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

// ---------------------------------------------------------------------------
// The component view on a polarizing element
// ---------------------------------------------------------------------------

test.describe("component view — coating a lens", () => {
  test("the default singlet can be given an AR coating, a face at a time", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    // The default singlet is what the inset opens on.
    const view = await insetView(page);
    expect(view.kind).toBe("lens");
    const id = view.id;

    const faces = () =>
      page.evaluate(
        (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.lensFaceCoatingsJSON(i)),
        id,
      );
    const flux = () =>
      page.evaluate(() => {
        const d = (window as unknown as { __ocDebug: any }).__ocDebug;
        const dets = JSON.parse(d.scene.sceneObjectsJSON()).filter((o: any) => o.kind === "detector");
        return JSON.parse(d.scene.detectorResultJSON(dets[0].id)).totalIncident as number;
      });

    expect(await faces()).toEqual({ front: null, back: null });
    await expect(page.locator("#componentInset .cv-read")).toContainText("bare / bare");
    const bare = await flux();

    // The control follows whichever face is selected, so coating twice needs
    // the face buttons in between, the same way R and k do.
    await page.locator('#componentInset button[data-value="ar-3"]').click();
    expect((await faces()).front).toBe("ar-3");
    expect((await faces()).back).toBeNull();
    await expect(page.locator("#componentInset .cv-read")).toContainText("AR ×3 / bare");

    const oneFace = await flux();
    expect(oneFace).toBeGreaterThan(bare);

    // And the lens keeps its own editor once it is coated, rather than being
    // taken for a coated plate.
    expect((await insetView(page)).kind).toBe("lens");

    // Strip it again and the light goes back where it was.
    await page.locator('#componentInset button[data-value="none"]').click();
    expect((await faces()).front).toBeNull();
    expect(await flux()).toBeCloseTo(bare, 9);
  });

  test("coating a lens draws no extra rays while the ghost layer is off", async ({ page }) => {
    // An AR coat's reflection is a ghost, the same as bare glass's. It used to
    // be traced as a primary ray, so coating the lens drew its reflections at
    // full strength with the ghost layer switched off.
    await freshApp(page);
    await openInset(page);
    await expect(page.locator("#ghosts")).not.toBeChecked();

    const drawn = () =>
      page.evaluate(() => {
        const d = (window as unknown as { __ocDebug: any }).__ocDebug;
        return {
          preview: (d.cpuProjected() as Float32Array).length,
          exact: (d.exactProjected() as Float32Array).length,
        };
      });
    const bare = await drawn();

    for (const kind of ["ar-1", "ar-3"]) {
      await page.locator(`#componentInset button[data-value="${kind}"]`).click();
      await expect(page.locator("#stat")).toContainText("segs");
      // The preview draws the picture and the exact trace the numbers; both
      // have to leave the reflection to the ghost switch.
      expect(await drawn(), kind).toEqual(bare);
    }

    // With the ghost layer on, the reflections are there to see.
    await page.locator("#ghosts").check();
    await expect(page.locator("#stat")).toContainText("segs");
    const lit = await drawn();
    expect(lit.preview).toBeGreaterThan(bare.preview);
    expect(lit.exact).toBeGreaterThan(bare.exact);
  });
});

test.describe("component view — polarizing elements", () => {
  /** Add an element through its own button and select it. */
  async function addAndSelect(page: Page, button: string): Promise<number> {
    const id = await page.evaluate((b) => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      (document.getElementById(b) as HTMLButtonElement).click();
      return d.scene.sceneRevision();
    }, button);
    expect(id).toBeGreaterThan(0);
    await expect(page.locator("#stat")).toContainText("segs");
    return id;
  }

  test("a polarizer opens its own editor, not the coating one", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await addAndSelect(page, "addPolarizer");

    const view = await insetView(page);
    expect(view.kind).toBe("polarizing");
    // The subtitle names the part rather than calling it a surface.
    await expect(page.locator("#componentInset .inset-sub")).toContainText("polarizer");
    // Its axis is a handle, and there is no coating curve to drag.
    expect(view.handles.map((h) => h.key)).toContain("axis");
    // The readout says the thing a user needs to know about a polarizer.
    await expect(page.locator("#componentInset .cv-read")).toContainText("loses half");
  });

  test("turning the axis handle changes the scene and the numbers follow", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await addAndSelect(page, "addPolarizer");
    const id = (await insetView(page)).id;

    const axisOf = (page: Page, id: number) =>
      page.evaluate(
        (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.surfacePolarizationJSON(i)).axisDeg as number,
        id,
      );
    const before = await axisOf(page, id);

    const h = await handle(page, "axis");
    await dragBy(page, h, -40, -40);
    const after = await axisOf(page, id);
    expect(Math.abs(after - before)).toBeGreaterThan(5);

    // And the field shows what the scene holds.
    const field = page.locator('#componentInset input[data-name="axis"]');
    await expect(field).toHaveValue(String(Number(after.toFixed(4))));
  });

  test("a waveplate shows its retardance and a quarter wave makes the beam circular", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    // A polarizer first, so there is a linear state for the plate to act on.
    await addAndSelect(page, "addPolarizer");
    await addAndSelect(page, "addRetarder");

    const view = await insetView(page);
    expect(view.kind).toBe("polarizing");
    await expect(page.locator("#componentInset .inset-sub")).toContainText("retarder");
    await expect(page.locator("#componentInset .cv-read")).toContainText("quarter wave");
    expect(view.handles.map((h) => h.key)).toContain("retardance");

    // Typing a half wave into the field changes what the tracer reports, and
    // the readout renames the element.
    const field = page.locator('#componentInset input[data-name="retardance"]');
    await field.fill("180");
    await expect(page.locator("#componentInset .cv-read")).toContainText("half wave");
  });

  test("the substrate field is on both, and putting one in glass costs it light", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await addAndSelect(page, "addPolarizer");

    const field = page.locator('#componentInset input[data-name="substrate"]');
    await expect(field).toBeVisible();
    // A bare layer by default: the readout says so rather than showing a 1.
    await expect(page.locator("#componentInset .cv-read")).toContainText("bare layer");

    const id = (await insetView(page)).id;
    const detector = (page: Page) =>
      page.evaluate(() => {
        const d = (window as unknown as { __ocDebug: any }).__ocDebug;
        const dets = JSON.parse(d.scene.sceneObjectsJSON()).filter((o: any) => o.kind === "detector");
        return JSON.parse(d.scene.detectorResultJSON(dets[0].id)).totalIncident as number;
      });
    const bare = await detector(page);

    await field.fill("1.5");
    await expect(page.locator("#stat")).toContainText("segs");
    // The readout names the glass and what its faces cost on axis.
    await expect(page.locator("#componentInset .cv-read")).toContainText("92.2%");
    expect(
      await page.evaluate(
        (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.surfacePolarizationJSON(i)).substrateIndex as number,
        id,
      ),
    ).toBeCloseTo(1.5, 6);

    // Two air-to-glass faces, so the detector loses about 8% of what it had.
    const glass = await detector(page);
    expect(glass / bare).toBeGreaterThan(0.90);
    expect(glass / bare).toBeLessThan(0.93);

    // And a waveplate has the field too, because it has faces for the same
    // reason a polarizer does.
    await addAndSelect(page, "addRetarder");
    await expect(page.locator('#componentInset input[data-name="substrate"]')).toBeVisible();
  });

  test("a waveplate can be cut from a real crystal, and then it has a thickness", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await addAndSelect(page, "addPolarizer");
    await addAndSelect(page, "addRetarder");
    const id = (await insetView(page)).id;

    // Ideal to begin with: no crystal, and the readout says the retardance
    // holds everywhere rather than at one wavelength.
    await expect(page.locator("#componentInset .cv-read")).toContainText("every wavelength");
    await expect(page.locator('#componentInset input[data-name="order"]')).toBeHidden();

    await page.locator('#componentInset button[data-value="quartz"]').click();
    await expect(page.locator('#componentInset input[data-name="order"]')).toBeVisible();
    // A zero-order quartz quarter wave is about 15 µm thick.
    const crystal = async () =>
      page.evaluate(
        (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.surfacePolarizationJSON(i)).crystal,
        id,
      );
    const zero = await crystal();
    expect(zero.order).toBe(0);
    expect(zero.thicknessUm).toBeGreaterThan(10);
    expect(zero.thicknessUm).toBeLessThan(20);
    await expect(page.locator("#componentInset .cv-read")).toContainText("µm of quartz");

    // Ten whole waves under it makes the same plate forty-one times thicker
    // and forty-one times as fussy.
    await page.locator('#componentInset input[data-name="order"]').fill("10");
    const tenth = await crystal();
    expect(tenth.order).toBe(10);
    expect(tenth.thicknessUm / zero.thicknessUm).toBeCloseTo(41, 1);

    // And back to ideal, which drops the crystal rather than hiding it.
    await page.locator('#componentInset button[data-value="ideal"]').click();
    expect(await crystal()).toBeNull();
    await expect(page.locator('#componentInset input[data-name="order"]')).toBeHidden();
  });

  test("a polarizer's contrast can come off a datasheet and the plot says so", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await addAndSelect(page, "addPolarizer");
    const id = (await insetView(page)).id;

    // Ideal to begin with: one number, and the readout says it holds anywhere.
    await expect(page.locator("#componentInset .cv-read")).toContainText("at every angle");

    const ok = await page.evaluate(
      (i) =>
        (window as unknown as { __ocDebug: any }).__ocDebug.scene.setPolarizerExtinctionTable(
          i,
          new Float64Array([0, 10, 25, 40]),
          new Float64Array([1e-4, 3e-4, 5e-3, 8e-2]),
        ),
      id,
    );
    expect(ok).toBe(true);
    await expect(page.locator("#stat")).toContainText("segs");
    // A table is set from script, not from the panel, so nudge the inset the
    // way reopening it would.
    await page.evaluate(() => {
      const d = (window as unknown as { __ocDebug: any }).__ocDebug;
      d.inset.close();
      d.inset.open();
    });

    // The readout names the range somebody measured rather than implying the
    // number holds everywhere.
    await expect(page.locator("#componentInset .cv-read")).toContainText("measured out to 40°");
    const table = await page.evaluate(
      (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.surfacePolarizationJSON(i)).extinctionTable,
      id,
    );
    expect(table.aoiDeg).toEqual([0, 10, 25, 40]);
    // The field still shows the on-axis contrast, which is the headline number.
    await expect(page.locator('#componentInset input[data-name="extinction"]')).toHaveValue("1.000e-4");

    // Typing one number puts it back to an ideal layer.
    await page.locator('#componentInset input[data-name="extinction"]').fill("1e-5");
    await expect(page.locator("#componentInset .cv-read")).toContainText("at every angle");
    expect(
      await page.evaluate(
        (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.surfacePolarizationJSON(i)).extinctionTable,
        id,
      ),
    ).toBeNull();
  });

  test("the extinction field is there on a polarizer and not on a waveplate", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await addAndSelect(page, "addPolarizer");
    await expect(page.locator('#componentInset input[data-name="extinction"]')).toBeVisible();
    await expect(page.locator('#componentInset input[data-name="retardance"]')).toHaveCount(0);

    await addAndSelect(page, "addRetarder");
    await expect(page.locator('#componentInset input[data-name="retardance"]')).toBeVisible();
    await expect(page.locator('#componentInset input[data-name="extinction"]')).toHaveCount(0);
  });
});

test.describe("component view — what a detector measures", () => {
  /** Add a detector through its own button, which selects it. */
  async function addDetector(page: Page) {
    await page.locator("#addDetector").click();
    await expect(page.locator("#stat")).toContainText("segs");
    const view = await insetView(page);
    expect(view.kind).toBe("detector");
    return view.id;
  }

  const result = (page: Page, id: number) =>
    page.evaluate(
      (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.detectorResultJSON(i)),
      id,
    );

  test("a detector starts on power and the cone field is not in the way", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    const id = await addDetector(page);

    const bar = page.locator("#componentInset .cv-bar");
    for (const name of ["power", "irradiance", "intensity", "illuminance"]) {
      await expect(bar.getByText(name, { exact: true })).toBeVisible();
    }
    await expect(bar.getByText("power", { exact: true })).toHaveClass(/active/);
    // Only intensity has a cone, so nothing else shows the field.
    await expect(page.locator('#componentInset input[data-name="cone"]')).toBeHidden();
    expect((await result(page, id)).measure).toBe("power");
  });

  test("irradiance is the same light divided by the cell, and says so", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    const id = await addDetector(page);
    const before = await result(page, id);
    expect(before.hits).toBeGreaterThan(0);

    await page.locator("#componentInset .cv-bar").getByText("irradiance", { exact: true }).click();
    await expect(page.locator("#stat")).toContainText("segs");

    const after = await result(page, id);
    expect(after.measure).toBe("irradiance");
    expect(after.unit).toBe("flux/mm²");
    // Same flux, same cells, one division between the two readings.
    expect(after.totalIncident).toBeCloseTo(before.totalIncident, 12);
    expect(after.peak).toBeCloseTo(before.peak / before.pixelArea, 9);
    await expect(page.locator("#componentInset .cv-read")).toContainText("flux/mm²");
    // And the dock agrees, which is the number a user actually reads.
    await expect(page.locator("#dockNums")).toContainText("flux/mm²");
  });

  test("illuminance reads lux and gives each pixel its colour", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    const id = await addDetector(page);
    const exposure = page.locator('#componentInset input[data-name="detExposure"]');
    await expect(exposure).toBeHidden();

    await page.locator("#componentInset .cv-bar").getByText("illuminance", { exact: true }).click();
    await expect(page.locator("#stat")).toContainText("segs");

    const info = await result(page, id);
    expect(info.measure).toBe("illuminance");
    // No sun, so the flux is relative and so are the lux.
    expect(info.unit).toBe("lx (relative)");
    expect(info.colour.length).toBe(3 * info.cells.length);
    // Y is the illuminance, pixel for pixel.
    for (let i = 0; i < info.cells.length; i += 97) {
      expect(info.colour[3 * i + 1]).toBeCloseTo(info.cells[i], 9);
    }
    expect(info.peak).toBeGreaterThan(0);
    await expect(exposure).toBeVisible();
    await expect(page.locator("#componentInset .cv-read")).toContainText("lx (relative)");
    await expect(page.locator("#dockNums")).toContainText("lx (relative)");
  });

  test("intensity bins the arrival directions and reports per steradian", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    const id = await addDetector(page);

    await page.locator("#componentInset .cv-bar").getByText("intensity", { exact: true }).click();
    await expect(page.locator("#stat")).toContainText("segs");

    const info = await result(page, id);
    expect(info.measure).toBe("intensity");
    expect(info.unit).toBe("flux/sr");
    // Its own grid, not the face's 64 by 64: two angles on straight axes,
    // with a cell on the normal.
    expect(info.directionLayout).toBe("grid");
    expect(info.coneBins).toEqual([31, 31]);
    expect(info.bins).toEqual([64, 64]);
    expect(info.cells.length).toBe(31 * 31);
    // The solid angles partition the square of directions exactly, which is
    // the property the whole measure rests on.
    const t = Math.tan((info.coneDeg * Math.PI) / 180);
    const square = 4 * Math.atan((t * t) / Math.sqrt(1 + 2 * t * t));
    const sum = (info.solidAngle as number[]).reduce((a: number, b: number) => a + b, 0);
    expect(sum).toBeCloseTo(square, 12);
    // Every ray is either in the grid or counted as being outside it.
    const binned = (info.cells as number[]).reduce(
      (a: number, v: number, i: number) => a + v * info.solidAngle[i],
      0,
    );
    expect(binned + info.outsideCone).toBeCloseTo(info.totalIncident, 9);

    // The cone field is now the one that matters, and it holds a real cone.
    const cone = page.locator('#componentInset input[data-name="cone"]');
    await expect(cone).toBeVisible();
    await cone.fill("15");
    await expect(page.locator("#stat")).toContainText("segs");
    expect((await result(page, id)).coneDeg).toBeCloseTo(15, 9);

    // And the dock stops calling the picture a heatmap of the face.
    await expect(page.locator("#detHead")).toContainText("far field");
    await expect(page.locator("#componentInset .cv-read")).toContainText("cells over");

    // The polar map is still there for whoever reads a goniometer, with its
    // own resolution: rings out to the cone, sectors round.
    await page.locator("#componentInset .cv-bar").getByText("polar", { exact: true }).click();
    await expect.poll(async () => (await result(page, id)).directionLayout).toBe("polar");
    expect((await result(page, id)).coneBins).toEqual([24, 1]);
    await expect(page.locator("#detHead")).toContainText("polar");
    await expect(page.locator("#componentInset .cv-read")).toContainText("ring");
    const polar = await result(page, id);
    const cap = 2 * Math.PI * (1 - Math.cos((polar.coneDeg * Math.PI) / 180));
    expect((polar.solidAngle as number[]).reduce((a: number, b: number) => a + b, 0)).toBeCloseTo(cap, 12);
    await page.locator('#componentInset input[data-name="detRings"]').fill("12");
    await expect.poll(async () => (await result(page, id)).coneBins).toEqual([12, 1]);
    await page.locator('#componentInset input[data-name="detSectors"]').fill("4");
    await expect.poll(async () => (await result(page, id)).coneBins).toEqual([12, 4]);
    expect((await result(page, id)).bins).toEqual([64, 64]);
  });
});

test.describe("component view — a detector's face", () => {
  const result = (page: Page, id: number) =>
    page.evaluate(
      (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.detectorResultJSON(i)),
      id,
    );
  const face = (page: Page, id: number) =>
    page.evaluate(
      (i) => JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.detectorFaceJSON(i)),
      id,
    );
  /** The flux the inset quotes first, and the flux the dock quotes. */
  const quoted = async (page: Page) => ({
    inset: (await page.locator("#componentInset .cv-read b").first().textContent()) ?? "",
    dock: /flux ([^ ]+)/.exec((await page.locator("#dockNums").textContent()) ?? "")?.[1] ?? "",
  });

  test("the inset and the dock quote the same flux on both arms of a splitter", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    // The default beam, a splitter folding half of it up, a detector on the
    // folded arm, and the default detector behind the lens on the other.
    await page.locator("#addSplitter").click();
    const up = await addUpDetector(page);
    const detectors: number[] = await page.evaluate(() =>
      JSON.parse((window as unknown as { __ocDebug: any }).__ocDebug.scene.sceneObjectsJSON())
        .filter((o: { kind: string }) => o.kind === "detector")
        .map((o: { id: number }) => o.id),
    );
    expect(detectors).toContain(up);
    expect(detectors.length).toBe(2);

    const select = (id: number) =>
      page.evaluate((i) => (window as unknown as { __ocDebug: any }).__ocDebug.select(i), id);
    for (const id of detectors) {
      await select(id);
      await expect.poll(async () => (await insetView(page)).id).toBe(id);
      const total = (await result(page, id)).totalIncident as number;
      expect(total).toBeGreaterThan(0);
      const q = await quoted(page);
      expect(q.inset, `detector ${id}`).toBe(q.dock);
      expect(Number(q.inset)).toBeCloseTo(total, 3);
    }

    // The ghost layer changes the trace without moving the scene revision.
    // The inset used to keep the old numbers while the dock moved on.
    const through = detectors.find((d) => d !== up)!;
    await select(through);
    await expect.poll(async () => (await insetView(page)).id).toBe(through);
    const before = (await result(page, through)).totalIncident as number;
    await page.locator("#ghosts").click();
    await expect.poll(async () => (await result(page, through)).totalIncident).not.toBe(before);
    await expect.poll(async () => {
      const q = await quoted(page);
      return q.inset === q.dock && q.inset !== "";
    }).toBe(true);
    const after = (await result(page, through)).totalIncident as number;
    expect(Number((await quoted(page)).inset)).toBeCloseTo(after, 3);
  });

  test("a detector's size and pixels are set in the inset", async ({ page }) => {
    await freshApp(page);
    await openInset(page);
    await page.locator("#addDetector").click();
    await expect(page.locator("#stat")).toContainText("segs");
    const id = (await insetView(page)).id;
    expect((await face(page, id)).shape).toBe("rectangle");

    const field = (name: string) => page.locator(`#componentInset input[data-name="${name}"]`);
    await field("detWidth").fill("12");
    await field("detHeight").fill("4");
    await expect.poll(async () => [(await face(page, id)).width, (await face(page, id)).height]).toEqual([12, 4]);

    await field("detPixelsU").fill("24");
    await field("detPixelsV").fill("8");
    await expect.poll(async () => (await face(page, id)).pixels).toEqual([24, 8]);
    const f = await face(page, id);
    expect(f.pitch).toEqual([0.5, 0.5]);
    expect(f.pixelArea).toBeCloseTo(0.25, 15);
    expect(f.faceArea).toBeCloseTo(48, 12);
    // The trace bins into the grid the face now has.
    expect((await result(page, id)).bins).toEqual([24, 8]);
    await expect(page.locator("#componentInset .cv-read")).toContainText("24 × 8 px of 500 µm");

    // A count of zero is refused, said so, and changes nothing.
    await field("detPixelsU").fill("0");
    await expect(page.locator("#componentInset .cv-note")).toContainText("pixel counts start at 1");
    expect((await face(page, id)).pixels).toEqual([24, 8]);

    // A round face: the largest disc inside the rectangle, the same counts.
    await page.locator("#componentInset .cv-bar").getByText("disc", { exact: true }).click();
    await expect.poll(async () => (await face(page, id)).shape).toBe("disc");
    expect((await face(page, id)).width).toBe(4);
    await expect(field("detDiameter")).toBeVisible();
    await expect(field("detWidth")).toBeHidden();
    await expect(page.locator("#componentInset .cv-read")).toContainText("rim pixels");

    // And the corner of a rectangle drags its size.
    await page.locator("#componentInset .cv-bar").getByText("rect", { exact: true }).click();
    await expect.poll(async () => (await face(page, id)).shape).toBe("rectangle");
    const w0 = (await face(page, id)).width;
    await dragBy(page, await handle(page, "corner"), 20, -10);
    await expect.poll(async () => (await face(page, id)).width).toBeGreaterThan(w0);
    const q = await quoted(page);
    expect(q.inset).toBe(q.dock);
  });
});
