// 3D interaction parity (§5.1): the v1 keyboard verbs, retargeted to the Scene3
// selection. Clicking a body selects it (the object inspector confirms count),
// then Ctrl+D duplicates and Delete removes — proving the keyboard → Rust edit
// API → re-trace path works in the always-3D app.

import { test, expect } from "@playwright/test";

async function centre(page: import("@playwright/test").Page) {
  const box = (await page.locator("#view").boundingBox())!;
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

test.describe("3D scene editing — copy/paste/delete on the selection", () => {
  test("Ctrl+D duplicates the selected body", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    // The default scene lists the singlet plus the default detector.
    await expect(page.locator("#objects3d .row")).toHaveCount(2);

    const c = await centre(page);
    await page.mouse.click(c.x, c.y); // select the singlet
    await page.keyboard.press("Control+d"); // copy + paste

    await expect(page.locator("#objects3d .row")).toHaveCount(3);
  });

  test("Delete removes the selected body", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    await expect(page.locator("#objects3d .row")).toHaveCount(2);

    const c = await centre(page);
    await page.mouse.click(c.x, c.y); // select the singlet
    await page.keyboard.press("Delete");

    // The singlet is gone; the default detector remains.
    await expect(page.locator("#objects3d .row")).toHaveCount(1);
  });
});
