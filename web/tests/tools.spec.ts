// The scene tools card: folded to its wrench for a new viewer, opened with a
// click, and remembered open. No banner above it.

import { test, expect } from "@playwright/test";

test("the tools card starts folded and opens on a click", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.removeItem("optiland-canvas:tools"));
  await page.reload();
  await expect(page.locator("#stat")).toContainText("segs");
  const card = page.locator("#m4dock");
  const toggle = page.locator("#toolsToggle");
  await expect(card).toHaveClass(/collapsed/);
  await expect(toggle).toBeVisible();
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await expect(page.locator("#addTube")).toBeHidden();
  // The banner is gone, and the status line with it.
  await expect(page.getByText("always 3D")).toHaveCount(0);
  await expect(page.locator("#stat")).toBeHidden();

  await toggle.click();
  await expect(card).not.toHaveClass(/collapsed/);
  await expect(page.locator("#addTube")).toBeVisible();
  await page.reload();
  await expect(page.locator("#addTube")).toBeVisible();

  await toggle.click();
  await expect(page.locator("#addTube")).toBeHidden();
});
