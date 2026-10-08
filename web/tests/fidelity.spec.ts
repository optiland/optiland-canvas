// The fidelity profile in the app (coating-fidelity spec §5, CF-01): the user
// picks how much coating physics a trace pays for, every number says which
// profile it was computed under, and the final check compares against
// Reference without changing the profile being edited.

import { test, expect, type Page } from "@playwright/test";

async function readout(page: Page): Promise<string> {
  await page.waitForFunction(() => (window as any).__ocDebug.referenceCurrent());
  return (await page.locator("#dockNums").textContent()) ?? "";
}

function fluxOf(line: string): number {
  return Number(/flux ([\d.]+)/.exec(line)?.[1]);
}

test.describe("fidelity", () => {
  test("a preset changes the stamp and the traced flux, and Standard is the kernel as it was", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    // A dichroic in the beam: a coating whose answer depends on wavelength,
    // which Fast replaces with its design point.
    await page.locator("#addDichroic").click();
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#fidSummary").click();

    await expect(page.locator("#fidPreset")).toHaveValue("Standard");
    const standard = fluxOf(await readout(page));
    expect(standard).toBeGreaterThan(0);
    await expect(page.locator("#dockStamp")).toContainText("computed with Standard");

    await page.locator("#fidPreset").selectOption("Fast");
    await expect(page.locator("#dockStamp")).toContainText("computed with Fast");
    await expect(page.locator("#dockStamp")).toContainText("nominal, not verified");
    // Fast turns the ghost layer off, and the toggle shows it.
    await expect(page.locator("#ghosts")).not.toBeChecked();
    const fast = fluxOf(await readout(page));
    expect(Math.abs(fast - standard)).toBeGreaterThan(1e-3 * standard);

    // One switch away from a preset is custom.
    await page.locator("#fid-numerics").selectOption("Strict");
    await expect(page.locator("#fidPreset")).toHaveValue("custom");
    await expect(page.locator("#dockStamp")).toContainText("custom profile");

    // Back to Standard gives the same number to the digit.
    await page.locator("#fidPreset").selectOption("Standard");
    await expect(page.locator("#dockStamp")).toContainText("computed with Standard");
    expect(fluxOf(await readout(page))).toBe(standard);
  });

  test("each switch shows its measured cost or says it was not measured", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#fidSummary").click();
    const hints = await page.locator(".fid-cost").allTextContents();
    expect(hints.length).toBe(6);
    for (const h of hints) expect(h).toMatch(/^(×[\d.]+ time against \w+|not measured)$/);
    // The preview says what it cannot run under Standard.
    await expect(page.locator("#fidPreview")).toContainText("Preview: unpolarized coatings");
  });

  test("the final check compares with Reference and leaves the profile alone", async ({ page }) => {
    await page.goto("/?debug");
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#addDichroic").click();
    await expect(page.locator("#stat")).toContainText("segs");
    await page.locator("#fidSummary").click();
    await page.locator("#fidPreset").selectOption("Fast");
    await page.locator("#fidCheck").click();
    const table = page.locator("#fidCheckTable");
    await expect(table).toBeVisible({ timeout: 60_000 });
    await expect(table).toContainText("Reference");
    await expect(table).toContainText("flux");
    await expect(page.locator("#fidCheckOut")).toContainText("Fast against Reference");
    await expect(page.locator("#fidPreset")).toHaveValue("Fast");
  });
});
