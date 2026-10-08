// What the Save button writes is what the Python bridge has to read.
//
// The bridge's fixtures in crates/oc-py/tests/data are files this app
// saved, not files written by hand to look like them. This spec saves the same
// scenes again and holds them to the fixtures, so the day the document changes
// shape the bridge's tests are told rather than left reading an old one.
// Regenerate them with
//
//   UPDATE_BRIDGE_FIXTURES=1 npx playwright test tests/save.spec.ts
//
// and run the bridge's tests against the new files.

import { test, expect, type Page } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const fixture = (name: string) =>
  fileURLToPath(new URL(`../../crates/oc-py/tests/data/${name}`, import.meta.url));

async function freshApp(page: Page) {
  await page.goto("/");
  await expect(page.locator("#stat")).toContainText("segs");
  await page.evaluate(() => {
    localStorage.clear();
    localStorage.setItem("optiland-canvas:tools", "open");
  });
  await page.reload();
  await expect(page.locator("#stat")).toContainText("segs");
}

/** Click Save and return the file the browser downloaded. */
async function save(page: Page): Promise<string> {
  const started = page.waitForEvent("download");
  await page.locator("#saveScene").click();
  const file = await started;
  expect(file.suggestedFilename()).toMatch(/\.ocanvas$/);
  const stream = await file.createReadStream();
  return new Promise<string>((resolve, reject) => {
    let s = "";
    stream.on("data", (c) => (s += c));
    stream.on("end", () => resolve(s));
    stream.on("error", reject);
  });
}

/** Write the fixture when asked to, and otherwise hold the save to it. */
function matchFixture(name: string, body: string) {
  if (process.env.UPDATE_BRIDGE_FIXTURES) {
    writeFileSync(fixture(name), body.endsWith("\n") ? body : `${body}\n`);
    return;
  }
  const kept = JSON.parse(readFileSync(fixture(name), "utf8"));
  expect(JSON.parse(body), `${name} no longer matches what the app saves`).toEqual(kept);
}

test.describe("save writes the document the bridge reads", () => {
  test("the default scene saves as a v3 document", async ({ page }) => {
    await freshApp(page);
    const body = await save(page);
    const doc = JSON.parse(body);
    expect(doc.format).toBe("ocanvas");
    expect(doc.version).toBe(3);
    expect(doc.bodies.length).toBe(1);
    expect(doc.detectors.length).toBe(1);
    matchFixture("browser-default-v3.ocanvas", body);
  });

  test("a scene with a beam splitter saves as a v3 document", async ({ page }) => {
    await freshApp(page);
    await page.locator("#addSplitter").click();
    await expect(page.locator("#stat")).toContainText("segs");
    const body = await save(page);
    const doc = JSON.parse(body);
    expect(doc.version).toBe(3);
    expect(doc.surfaces.length).toBe(1);
    matchFixture("browser-splitter-v3.ocanvas", body);
  });
});
