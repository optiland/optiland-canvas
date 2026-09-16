// The interactive picture's backend (§4.1). Headless CI has no WebGPU device, so
// this checks the app says which backend draws, and the other specs check it
// still paints and retraces on the CPU mirror. The GPU run is webgpu.spec.ts,
// which is opt-in.

import { test, expect } from "@playwright/test";

test("the status line names the preview backend", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#stat")).toContainText("segs");
  await expect(page.locator("#stat")).toContainText(/preview (GPU|CPU)/);
});
