// Inspector-card browser test (§5.2).
//
// The floating-card engine (src/cards.ts) is driven by a per-scene adapter. The
// app boots into the canonical 3D scene, so these exercise the Scene3 adapter
// (src/slice3d.ts): double-click opens a card, a field edit reaches the Rust
// scene (survives a close/reopen round-trip), the header drag pulls the card off
// its element, pin-location parks it, and close removes it.
//
// The default singlet sits at world origin, so it projects to the view centre at
// the default pan/zoom — that is where we double-click to inspect it.

import { test, expect } from "@playwright/test";

async function centre(page: import("@playwright/test").Page) {
  const box = (await page.locator("#view").boundingBox())!;
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

test.describe("inspector cards — the extracted floating-window engine", () => {
  test("double-click opens a singlet card next to the element", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");

    await expect(page.locator(".card")).toHaveCount(0);
    const c = await centre(page);
    await page.mouse.click(c.x, c.y, { clickCount: 2 });

    const card = page.locator(".card");
    await expect(card).toHaveCount(1);
    // A singlet: the title carries its id, and the singlet field block is shown.
    await expect(card.locator(".c-title")).toContainText("#");
    await expect(card.locator(".c3-singlet")).toBeVisible();
    await expect(card.locator('input[data-name="r1"]')).toHaveValue(/\d/);
  });

  test("a field edit reaches the scene and survives a close/reopen", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    const c = await centre(page);
    await page.mouse.click(c.x, c.y, { clickCount: 2 });

    const r1 = page.locator('.card input[data-name="r1"]');
    await r1.fill("40");
    await r1.dispatchEvent("input"); // commitCard → scene.setSingletGeom → re-trace

    // Close and reopen: the reopened card is rendered fresh from the Rust scene,
    // so the new radius is only there if the edit actually landed in the scene.
    await page.locator(".card .c-close").click();
    await expect(page.locator(".card")).toHaveCount(0);
    await page.mouse.click(c.x, c.y, { clickCount: 2 });
    await expect(page.locator('.card input[data-name="r1"]')).toHaveValue("40");
  });

  test("the tilt field rotates the element and survives a close/reopen", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    const c = await centre(page);
    await page.mouse.click(c.x, c.y, { clickCount: 2 });

    const tilt = page.locator('.card input[data-name="rot"]');
    await expect(tilt).toHaveValue(/0/); // the default singlet is untilted
    await tilt.fill("15");
    await tilt.dispatchEvent("input"); // commitCard → setObjectRotationZDeg → re-trace

    await page.locator(".card .c-close").click();
    await page.mouse.click(c.x, c.y, { clickCount: 2 });
    await expect(page.locator('.card input[data-name="rot"]')).toHaveValue("15");
  });

  test("the header drag pulls the card off its element", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    const c = await centre(page);
    await page.mouse.click(c.x, c.y, { clickCount: 2 });

    const card = page.locator(".card");
    const before = (await card.boundingBox())!;
    // Grab the header near its left (the title, clear of the pin/close buttons).
    const h = (await card.locator("h3").boundingBox())!;
    await page.mouse.move(h.x + 12, h.y + h.height / 2);
    await page.mouse.down();
    await page.mouse.move(h.x + 12 + 120, h.y + h.height / 2 + 80, { steps: 8 });
    await page.mouse.up();

    const after = (await card.boundingBox())!;
    expect(Math.abs(after.x - before.x)).toBeGreaterThan(60);
  });

  test("pin-location toggles, and close removes the card", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#stat")).toContainText("segs");
    const c = await centre(page);
    await page.mouse.click(c.x, c.y, { clickCount: 2 });

    const pin = page.locator(".card .c-pinpos");
    await expect(pin).toHaveText("📍");
    await pin.click();
    await expect(pin).toHaveText("📌");
    await expect(pin).toHaveClass(/pinned/);

    await page.locator(".card .c-close").click();
    await expect(page.locator(".card")).toHaveCount(0);
  });
});
