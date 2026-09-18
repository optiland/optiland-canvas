// Polarizing elements in the app: a polarizer and a waveplate you can place,
// and a detector readout that says what state the light arrives in.
//
// The reference trace carries a Stokes vector; the f32 preview does not. That
// gap is deliberate, and the readout comes from the f64 trace, so it knows
// things the picture cannot show.

import { test, expect } from "@playwright/test";

/** The detector's f64 readout line. */
async function readout(page: import("@playwright/test").Page): Promise<string> {
  return (await page.locator("#dockNums").textContent()) ?? "";
}

function fluxOf(line: string): number {
  return Number(/flux ([\d.]+)/.exec(line)?.[1]);
}

function dopOf(line: string): number {
  const m = /pol ([\d.]+)%/.exec(line);
  return m ? Number(m[1]) : 0;
}

test.describe("polarization", () => {
  test("the default singlet already polarizes its light a little", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    const line = await readout(page);
    // A lens is two glass surfaces, and a ray away from the axis meets them
    // obliquely, so a beam through one comes out slightly polarized. Small, and
    // real: it is the same effect that makes a pile of plates a polarizer.
    const dop = dopOf(line);
    expect(dop).toBeGreaterThan(0);
    expect(dop).toBeLessThan(3);
  });

  test("a polarizer halves the light and says what comes out", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    const fluxBefore = fluxOf(await readout(page));
    expect(fluxBefore).toBeGreaterThan(0);

    await page.locator("#addPolarizer").click();
    await expect(page.locator("#stat")).toContainText("segs");
    const after = await readout(page);

    // Unpolarized light loses half its flux at an ideal polarizer. That is
    // physics rather than a modelled loss.
    const fluxAfter = fluxOf(after);
    expect(fluxAfter).toBeGreaterThan(fluxBefore * 0.4);
    expect(fluxAfter).toBeLessThan(fluxBefore * 0.6);
    // And what arrives is fully polarized, and linear.
    expect(dopOf(after)).toBeGreaterThan(99);
    expect(after).toContain("linear");
  });

  test("a waveplate changes the state and not the flux", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");

    await page.locator("#addPolarizer").click();
    await expect(page.locator("#dockNums")).toContainText("linear");
    const polarized = fluxOf(await readout(page));

    // The waveplate button adds a quarter wave with its fast axis at 45
    // degrees to the polarizer, which on its own turns linear light circular.
    // What the detector sees is elliptical rather than perfectly circular, and
    // that is right: the singlet still stands between the plate and the
    // detector, and its two glass faces treat s and p differently, so they
    // pull the state back off the pole. A crossed-polarizer bench with nothing
    // in between is the test for exact circularity, and it lives in the Rust
    // suite where the geometry can be made bare.
    await page.locator("#addRetarder").click();
    await expect(page.locator("#stat")).toContainText("segs");
    const line = await readout(page);
    expect(line).not.toContain("linear");
    expect(dopOf(line)).toBeGreaterThan(99);

    // A retarder is lossless: it moves the state, not the light.
    expect(fluxOf(line)).toBeGreaterThan(polarized * 0.95);
  });

  test("both elements land in the object list and can be picked", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    const before = await page.locator("#objects3d .row").count();

    await page.locator("#addPolarizer").click();
    await page.locator("#addRetarder").click();
    await expect(page.locator("#stat")).toContainText("segs");

    await expect(page.locator("#objects3d .row")).toHaveCount(before + 2);
  });
});
