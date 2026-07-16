// Browser tests (§19.5).
//
// Against the built app, not the dev server: `vite preview` serves exactly what
// ships, so a test cannot pass on a bundle nobody deploys. The wasm is rebuilt as
// part of `npm run build`, so the binding surface under test is the current one.

import { defineConfig, devices } from "@playwright/test";

const PORT = 4173;

export default defineConfig({
  testDir: "./tests",
  // The openUC2 scenario is a sequence of edits against one shared project; the
  // specs are independent of each other but each is inherently serial.
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? "list" : "line",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // After the device spread, which carries its own 1280x720 and would
        // otherwise win. Wide enough that the module library does not sit over
        // the origin cell: at 1280 the panel's right edge (340) is past where
        // cell [0,0,0] is drawn (320), so a drop there lands on a card and is
        // silently ignored. `placeCard` checks this too rather than trust the
        // arithmetic.
        viewport: { width: 1600, height: 900 },
      },
    },
  ],
  webServer: {
    command: `npm run build && npx vite preview --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    // The wasm build dominates: a cold `cargo build --release` is minutes.
    timeout: 600_000,
    stdout: "pipe",
  },
});
