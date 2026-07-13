import { defineConfig } from "vite";

// The wasm-pack output lives in ./pkg (see `npm run wasm`). Vite serves the
// .wasm with the right MIME type in dev; `fs.allow` lets it read the sibling
// crate output during the wasm build step.
export default defineConfig({
  server: { fs: { allow: [".."] } },
  build: { target: "esnext" },
});
