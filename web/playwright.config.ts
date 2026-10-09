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
      // Need a GPU or a local baseline; they run only in the opt-in projects below.
      testIgnore: /webgpu\.spec\.ts|render3d-(visual|bench)\.spec\.ts|reference-bench\.spec\.ts/,
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
    // A real GPU, opt-in: OC_WEBGPU=1 npx playwright test --project=webgpu.
    // OC_WEBGPU_CHANNEL picks an installed browser (e.g. chrome), and
    // OC_WEBGPU_HEADED=1 shows the window.
    ...(process.env.OC_WEBGPU
      ? [
          {
            name: "webgpu",
            // The GPU 3D view suite runs here too, on a real WebGPU device.
            testMatch: /webgpu\.spec\.ts|render3d\.spec\.ts/,
            use: {
              ...devices["Desktop Chrome"],
              viewport: { width: 1600, height: 900 },
              channel: process.env.OC_WEBGPU_CHANNEL ?? "chromium",
              headless: !process.env.OC_WEBGPU_HEADED,
              launchOptions: { args: ["--enable-unsafe-webgpu"] },
            },
          },
        ]
      : []),
    // Screenshot regressions of the GPU 3D view, opt-in because the baselines
    // are per machine and backend: OC_VISUAL=1 npx playwright test --project=visual
    // (add --update-snapshots to record). OC_VISUAL_CHANNEL picks the browser.
    ...(process.env.OC_VISUAL
      ? [
          {
            name: "visual",
            testMatch: /render3d-visual\.spec\.ts/,
            use: {
              ...devices["Desktop Chrome"],
              viewport: { width: 1280, height: 800 },
              deviceScaleFactor: 1,
              channel: process.env.OC_VISUAL_CHANNEL ?? "chrome",
              launchOptions: { args: ["--enable-unsafe-webgpu"] },
            },
          },
        ]
      : []),
    // The renderer benchmark: OC_BENCH=1 npx playwright test --project=bench.
    // Writes test-results/render3d-bench.{json,md}. OC_BENCH_CHANNEL picks the
    // browser (default: the installed Chrome, for a real WebGPU device).
    ...(process.env.OC_BENCH
      ? [
          {
            name: "bench",
            testMatch: /(render3d|reference)-bench\.spec\.ts/,
            timeout: 600_000,
            use: {
              ...devices["Desktop Chrome"],
              viewport: { width: 1600, height: 900 },
              deviceScaleFactor: 1,
              channel: process.env.OC_BENCH_CHANNEL ?? "chrome",
              launchOptions: { args: ["--enable-unsafe-webgpu"] },
            },
          },
        ]
      : []),
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
