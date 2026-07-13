# web/ — Optiland Canvas M1 shell

A minimal Vite + TypeScript shell that loads the `oc-wasm` module, builds a
one-lens scene, and lets you **drag the lens while the light follows** — every
frame is a full retrace (spec §0: no run button).

> **Status.** This shell is scaffolded and reviewed but has **not** been run in
> the environment M1 was built in (no browser/npm there). The tracer it drives
> *is* fully validated: `oc-gpu`'s compute shader and CPU fallback are
> cross-checked against the f64 reference in `cargo test` (see the repo README).
> The shell uses the portable CPU-f32 wasm backend; swapping in the WebGPU
> backend is behind the same `Canvas.trace()` call.

## Run it

Prerequisites: a stable Rust toolchain, [`wasm-pack`](https://rustwasm.github.io/wasm-pack/),
and Node 18+.

```sh
cd web
npm install
npm run dev        # builds the wasm (npm run wasm) then starts Vite
```

Open the printed URL and drag the blue lens outline. `npm run build` produces a
static bundle.

## How it fits together

- `../crates/oc-wasm` exposes an opaque `Canvas` handle (scene lives in Rust).
- `npm run wasm` runs `wasm-pack build --target web`, emitting `web/pkg/`.
- `src/main.ts` holds only UI state (the drag) + the handle, calls
  `scene.moveElement(...)` then `scene.trace()` per pointer-move, and draws the
  returned `[ax,ay,bx,by,flux]` segments with flux-weighted alpha (§5.5).

## Next (later milestones)

- Render via the WebGPU compute + HDR polyline path instead of the 2D canvas
  (§4.3), with progressive accumulation on pointer-up.
- Numeric inspector with bidirectional binding (M3, §5.2).
