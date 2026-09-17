// The GPU 3D view (render3d/): interaction, render-cache discipline, rays,
// quality states, fallbacks, and the rule that nothing here touches the optics.
//
// Runs in the default project. Headless CI has no WebGPU device, so there the
// view runs on three.js's WebGL2 backend (SwiftShader); a desktop Chromium with
// a GPU runs WebGPU. Either way the assertions are the same. The page is opened
// with `?debug`, which exposes the scene handle and the view's camera, stats,
// and pixel probe (main.ts, render3d/view3d.ts).

import { test, expect, type Page } from "@playwright/test";

type Stats = {
  backend: string;
  quality: string;
  frames: number;
  drawCalls: number;
  triangles: number;
  raySegments: number;
  rayUploads: number;
  meshBuilds: number;
  environmentBuilds: number;
  geometries: number;
  textures: number;
  cachedObjects: number;
};
type Cam = { yaw: number; pitch: number; scale: number; target: { x: number; y: number; z: number }; perspective: boolean };
type Obj = { id: number; kind: string; pos: number[] };

async function open3d(page: Page, query = "") {
  await page.goto(`/?debug${query ? `&${query}` : ""}`);
  await expect(page.locator("#stat")).toContainText("segs");
  await page.locator("#view-3d").click();
  await expect(page.locator("#view-3d")).toHaveClass(/active/);
  return page.evaluate(() => (window as any).__ocDebug.render3d.ready());
}

const renderNow = (page: Page): Promise<Stats> =>
  page.evaluate(() => (window as any).__ocDebug.render3d.renderNow());
const camera = (page: Page): Promise<Cam> => page.evaluate(() => (window as any).__ocDebug.render3d.camera());
const litPixels = (page: Page): Promise<number> =>
  page.evaluate(() => (window as any).__ocDebug.render3d.litPixels());
const objects = (page: Page): Promise<Obj[]> =>
  page.evaluate(() => JSON.parse((window as any).__ocDebug.scene.sceneObjectsJSON()));
const project = (page: Page, p: { x: number; y: number; z: number }) =>
  page.evaluate((p) => (window as any).__ocDebug.render3d.project(p), p);

/** The optical state a view setting must never change: the saved scene, the
 *  exact trace, and the detector readout. */
const opticalState = (page: Page) =>
  page.evaluate(() => {
    const d = (window as any).__ocDebug;
    return {
      scene: d.scene.scene3JSON(),
      trace: Array.from(d.exactProjected() as Float32Array),
      detector: d.scene.firstDetectorResultJSON(),
    };
  });

/** A point on empty canvas, clear of the panels and the default assembly. */
async function emptySpot(page: Page) {
  const box = (await page.locator("#view").boundingBox())!;
  return { x: box.x + box.width / 2, y: box.y + box.height - 160 };
}

/** Set a range input and fire its input event, as a user drag would. */
const setRange = (page: Page, selector: string, value: string) =>
  page.locator(selector).evaluate((el, v) => {
    (el as HTMLInputElement).value = v;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);

test.describe("GPU 3D view", () => {
  test("opens on a GPU backend, names it, and paints the scene", async ({ page }) => {
    const backend = await open3d(page);
    expect(["webgpu", "webgl2"]).toContain(backend);
    await expect(page.locator("#stat")).toContainText(/render (WebGPU|WebGL2 fallback)/);
    await expect(page.locator("#stat")).toContainText("3D model");
    // The ray-preview backend is still reported, separately.
    await expect(page.locator("#stat")).toContainText(/preview (GPU|CPU)/);
    await expect(page.locator("#view3dopts")).toBeVisible();
    await expect(page.locator("#view3d")).toBeVisible();
    expect(await litPixels(page)).toBeGreaterThan(200);
    const s = await renderNow(page);
    expect(s.cachedObjects).toBe(2); // the singlet and the detector
    expect(s.raySegments).toBeGreaterThan(100);
  });

  test("orbit, pan, and wheel zoom move the camera; reset restores it", async ({ page }) => {
    await open3d(page);
    const start = await camera(page);
    const p = await emptySpot(page);

    await page.mouse.move(p.x, p.y);
    await page.mouse.down();
    await page.mouse.move(p.x + 120, p.y - 40, { steps: 6 });
    await page.mouse.up();
    const orbited = await camera(page);
    expect(orbited.yaw).not.toBeCloseTo(start.yaw);

    await page.keyboard.down("Shift");
    await page.mouse.move(p.x, p.y);
    await page.mouse.down();
    await page.mouse.move(p.x + 80, p.y + 30, { steps: 4 });
    await page.mouse.up();
    await page.keyboard.up("Shift");
    const panned = await camera(page);
    expect(panned.yaw).toBeCloseTo(orbited.yaw);
    expect(Math.hypot(panned.target.x - orbited.target.x, panned.target.y - orbited.target.y, panned.target.z - orbited.target.z)).toBeGreaterThan(1);

    await page.mouse.move(p.x, p.y);
    await page.mouse.wheel(0, -300);
    await expect.poll(async () => (await camera(page)).scale).toBeGreaterThan(panned.scale);

    await page.locator("#v3dReset").click();
    const reset = await camera(page);
    expect(reset.yaw).toBeCloseTo(start.yaw);
    expect(reset.pitch).toBeCloseTo(start.pitch);
    expect(reset.scale).toBeCloseTo(start.scale, 3);
  });

  test("presets, perspective, and the 0 key", async ({ page }) => {
    await open3d(page);
    const iso = await camera(page);
    for (const id of ["#v3dTop", "#v3dSide", "#v3dFront"]) {
      await page.locator(id).click();
      const c = await camera(page);
      expect(Math.abs(c.yaw - iso.yaw) + Math.abs(c.pitch - iso.pitch)).toBeGreaterThan(0.1);
    }
    await page.locator("#v3dIso").click();
    expect((await camera(page)).yaw).toBeCloseTo(iso.yaw);
    await page.locator("#v3dPersp").click();
    expect((await camera(page)).perspective).toBe(true);
    await page.locator("#v3dPersp").click();
    expect((await camera(page)).perspective).toBe(false);
    // Zoom away, then 0 re-frames to the same scale.
    const p = await emptySpot(page);
    await page.mouse.move(p.x, p.y);
    await page.mouse.wheel(0, 400);
    await expect.poll(async () => (await camera(page)).scale).toBeLessThan(iso.scale);
    await page.keyboard.press("0");
    expect((await camera(page)).scale).toBeCloseTo(iso.scale, 3);
  });

  test("clicking picks a body; dragging moves the canonical body without a new mesh", async ({ page }) => {
    await open3d(page);
    const lens = (await objects(page)).find((o) => o.kind === "profile")!;
    const at = await project(page, { x: 0, y: 0, z: 0 });
    // Aim a little off the axis line, inside the lens face.
    const grab = { x: at.x + 2, y: at.y + 25 };
    await page.mouse.click(grab.x, grab.y);
    await expect.poll(() => page.evaluate(() => (window as any).__ocDebug.render3d.selected())).toBe(lens.id);
    const before = await renderNow(page);

    await page.mouse.move(grab.x, grab.y);
    await page.mouse.down();
    await page.mouse.move(grab.x + 60, grab.y, { steps: 6 });
    expect(await page.evaluate(() => (window as any).__ocDebug.render3d.quality())).toBe("interactive");
    await page.mouse.up();

    const moved = (await objects(page)).find((o) => o.id === lens.id)!;
    expect(Math.hypot(moved.pos[0], moved.pos[1], moved.pos[2])).toBeGreaterThan(1);
    const after = await renderNow(page);
    // A move is a transform: no geometry was rebuilt for it.
    expect(after.meshBuilds).toBe(before.meshBuilds);
  });

  test("double-click opens the card, and the selection survives a 2D/3D switch", async ({ page }) => {
    await open3d(page);
    const lens = (await objects(page)).find((o) => o.kind === "profile")!;
    const at = await project(page, { x: 0, y: 0, z: 0 });
    await page.mouse.dblclick(at.x + 2, at.y + 25);
    await expect(page.locator(".card")).toBeVisible();
    await expect(page.locator(".card .c-title")).toContainText(`#${lens.id}`);

    await page.locator("#view-2d").click();
    await expect(page.locator("#view3d")).toBeHidden();
    expect(await page.evaluate(() => (window as any).__ocDebug.selectedBody3d())).toBe(lens.id);
    await page.locator("#view-3d").click();
    expect(await page.evaluate(() => (window as any).__ocDebug.render3d.selected())).toBe(lens.id);
  });

  test("copy, paste, delete, and nudge act on the 3D selection", async ({ page }) => {
    await open3d(page);
    await expect(page.locator("#objects3d .row")).toHaveCount(2);
    const lens = (await objects(page)).find((o) => o.kind === "profile")!;
    const at = await project(page, { x: 0, y: 0, z: 0 });
    await page.mouse.click(at.x + 2, at.y + 25);

    await page.keyboard.press("ArrowRight");
    const nudged = (await objects(page)).find((o) => o.id === lens.id)!;
    expect(Math.hypot(nudged.pos[0], nudged.pos[1], nudged.pos[2])).toBeGreaterThan(0.5);

    await page.keyboard.press("Control+c");
    await page.keyboard.press("Control+v");
    await expect(page.locator("#objects3d .row")).toHaveCount(3);
    await page.keyboard.press("Delete"); // the pasted copy is selected
    await expect(page.locator("#objects3d .row")).toHaveCount(2);
    const s = await renderNow(page);
    expect(s.cachedObjects).toBe(2);
  });

  test("lighting, background, exposure, and ray visibility never touch the optics", async ({ page }) => {
    await open3d(page);
    const before = await opticalState(page);
    for (const env of ["light_studio", "optical_bench", "inspection", "dark_studio", "light_studio"]) {
      await page.selectOption("#v3dEnv", env);
    }
    for (const bg of ["black", "light_gray", "mid_gray", "charcoal", "match"]) {
      await page.selectOption("#v3dBg", bg);
    }
    await setRange(page, "#v3dExposure", "1.5");
    await page.selectOption("#v3dRays", "occluded");
    await page.locator("#v3dGround").uncheck();
    const s = await renderNow(page);
    // Each preset's lighting is built once, however often it is chosen.
    expect(s.environmentBuilds).toBe(4);
    expect(await opticalState(page)).toEqual(before);
    expect(await litPixels(page)).toBeGreaterThan(200);
  });

  test("only geometry edits rebuild meshes", async ({ page }) => {
    await open3d(page);
    const lensId = (await objects(page)).find((o) => o.kind === "profile")!.id;
    const builds = async () => (await renderNow(page)).meshBuilds;
    const base = await builds();
    const edit = (fn: string) => page.evaluate(([fn, id]) => {
      const s = (window as any).__ocDebug.scene;
      switch (fn) {
        case "move": s.setObjectPosition(id, 3, 1, 0); break;
        case "rotate": s.setObjectRotationZDeg(id, 9); break;
        case "glass": s.setBodyGlassAbbe(id, 1.72, 30); break;
        case "radius": {
          const g = JSON.parse(s.singletGeomJSON(id));
          s.setSingletGeom(id, 42, g.r2, g.thickness, g.semiAperture, g.conic1, g.conic2);
          break;
        }
      }
    }, [fn, lensId] as const);

    await edit("move");
    expect(await builds()).toBe(base);
    await edit("rotate");
    expect(await builds()).toBe(base);
    await edit("glass");
    expect(await builds()).toBe(base);
    await edit("radius");
    expect(await builds()).toBe(base + 1);
    // Orbiting reads no geometry.
    const p = await emptySpot(page);
    await page.mouse.move(p.x, p.y);
    await page.mouse.down();
    await page.mouse.move(p.x + 90, p.y - 30, { steps: 5 });
    await page.mouse.up();
    expect(await builds()).toBe(base + 1);
  });

  test("the ray buffer holds every segment in a bounded number of draws", async ({ page }) => {
    await open3d(page);
    const expected = () => page.evaluate(() => {
      const a = (window as any).__ocDebug.exactProjected() as Float32Array;
      let max = 1e-9;
      for (let i = 7; i < a.length; i += 11) max = Math.max(max, a[i]);
      let n = 0;
      for (let i = 7; i < a.length; i += 11) if (Math.min(1, a[i] / max) >= 1e-5) n++;
      return n;
    });
    const s0 = await renderNow(page);
    expect(s0.raySegments).toBe(await expected());

    // A ghost-heavy picture: many more segments, the same meshes, a few draws.
    await page.locator("#ghosts").check();
    await expect.poll(async () => (await renderNow(page)).raySegments).toBeGreaterThan(s0.raySegments);
    const s1 = await renderNow(page);
    expect(s1.raySegments).toBe(await expected());
    expect(s1.meshBuilds).toBe(s0.meshBuilds);
    expect(s1.drawCalls).toBeLessThan(60);

    // Ghost opacity is a shader value: no re-upload.
    const uploads = s1.rayUploads;
    await setRange(page, "#ghostOpacity", "90");
    expect((await renderNow(page)).rayUploads).toBe(uploads);

    // The depth mode changes depth only.
    const cam = await camera(page);
    await page.selectOption("#v3dRays", "occluded");
    const s2 = await renderNow(page);
    expect(s2.raySegments).toBe(s1.raySegments);
    expect(s2.rayUploads).toBe(uploads);
    expect(await camera(page)).toEqual(cam);
  });

  test("interaction lowers quality only while it lasts", async ({ page }) => {
    await open3d(page);
    const quality = () => page.evaluate(() => (window as any).__ocDebug.render3d.quality());
    const before = await opticalState(page);
    expect(await quality()).toBe("settled");
    const p = await emptySpot(page);
    await page.mouse.move(p.x, p.y);
    await page.mouse.down();
    await page.mouse.move(p.x + 40, p.y, { steps: 3 });
    expect(await quality()).toBe("interactive");
    await page.mouse.up();
    await expect.poll(quality).toBe("settled");
    // A wheel tick is a short pulse; count it rather than race its 150 ms.
    const episodes = () => page.evaluate(() => (window as any).__ocDebug.render3d.qualityEpisodes());
    const before2 = await episodes();
    await page.mouse.wheel(0, -100);
    await expect.poll(episodes).toBe(before2 + 1);
    await expect.poll(quality).toBe("settled");
    // The settled frame is drawn, and the optics did not change.
    const frames = (await renderNow(page)).frames;
    expect(frames).toBeGreaterThan(1);
    expect(await opticalState(page)).toEqual(before);
  });

  test("an idle view draws no frames", async ({ page }) => {
    await open3d(page);
    await renderNow(page);
    const frames = async () => (await page.evaluate(() => (window as any).__ocDebug.render3d.stats())).frames;
    await page.waitForTimeout(400); // let any queued frame land
    const first = await frames();
    await page.waitForTimeout(800);
    expect(await frames()).toBe(first);
  });

  test("adding and removing bodies does not grow GPU resources", async ({ page }) => {
    await open3d(page);
    // Each cycle renders the copy, so the cache really builds and frees it.
    const cycle = async () => {
      const id = await page.evaluate(() => {
        const s = (window as any).__ocDebug.scene;
        const lens = JSON.parse(s.sceneObjectsJSON()).find((o: any) => o.kind === "profile");
        return s.cloneBody3D(lens.id, 10, 0, 0);
      });
      await renderNow(page);
      await page.evaluate((id) => (window as any).__ocDebug.scene.deleteObject3D(id), id);
    };
    await cycle();
    const base = await renderNow(page);
    for (let i = 0; i < 20; i++) {
      await cycle();
      await renderNow(page);
    }
    const after = await renderNow(page);
    expect(after.cachedObjects).toBe(base.cachedObjects);
    expect(after.geometries).toBeLessThanOrEqual(base.geometries);
    expect(after.textures).toBeLessThanOrEqual(base.textures);
  });
});

test.describe("GPU 3D view fallbacks", () => {
  test("the forced WebGL2 backend draws the same scene", async ({ page }) => {
    const backend = await open3d(page, "render3d=webgl");
    expect(backend).toBe("webgl2");
    await expect(page.locator("#stat")).toContainText("render WebGL2 fallback");
    expect(await litPixels(page)).toBeGreaterThan(200);
  });

  test("with no usable renderer the app says so and the 2D view still works", async ({ page }) => {
    const backend = await open3d(page, "render3d=off");
    expect(backend).toBe("unavailable");
    await expect(page.locator("#stat")).toContainText("render unavailable");
    await expect(page.locator("#view")).toBeVisible();
    await page.locator("#view-2d").click();
    await expect(page.locator("#stat")).not.toContainText("3D model");
    const lit = await page.evaluate(() => {
      const c = document.getElementById("view") as HTMLCanvasElement;
      const { data } = c.getContext("2d")!.getImageData(0, 0, c.width, c.height);
      let n = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i] + data[i + 1] + data[i + 2] > 60) n++;
      return n;
    });
    expect(lit).toBeGreaterThan(200);
  });

  test("the legacy wireframe view is still reachable for comparison", async ({ page }) => {
    await page.goto("/?renderer=legacy");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#view-3d").click();
    await expect(page.locator("#stat")).toContainText("3D model");
    await expect(page.locator("#stat")).not.toContainText("render ");
    await expect(page.locator("#view3dopts")).toBeHidden();
  });
});
