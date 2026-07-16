# Optiland Canvas

A browser-based, real-time optical sketchpad. Draw an optical system and watch the light move.

Rust → WebAssembly → WebGPU. Part of the [Optiland](https://github.com/optiland/optiland) family: use the canvas to *ideate*, then hand the design off to Optiland for rigorous analysis.

> **The one rule: there is never a "run" button.** Move a lens and the light moves. Drag a mirror to a grazing angle and total internal reflection lights up while you are still dragging. Every trade-off resolves toward sub-16 ms feedback; exact numbers settle once you let go, computed by an f64 reference tracer.

**Status:** M0–M5 complete and green, under active development (see the [roadmap](#roadmap)).

---

## What it is

Optiland Canvas is a tool for the *ideation* phase of optical design, the part before you commit to a prescription. It is a 2D (meridional-plane), non-sequential geometric ray tracer wrapped in a canvas-style UI: sketch lenses, mirrors, and prisms freehand, drag them around, and see refraction, reflection, TIR, dispersion, and stray-light ghosts update live. When a design is worth pursuing, export it to [Optiland](https://github.com/optiland/optiland), the Python engine for optimization, tolerancing, and full analysis.

**Why 2D.** For a rotationally symmetric system the meridional cross-section *is* the design. For prisms and folded paths, the top-down plane is how designers already sketch. 2D is also where a full retrace fits inside a 16 ms frame.

## Features

**Optics**
- Non-sequential ray tracing: refraction, reflection, total internal reflection, Fresnel splitting, ghost reflections, Beer–Lambert absorption.
- Per-ray dispersion: closed-form Cauchy from an `(n_d, V_d)` point, or full Sellmeier for catalog glasses. Chromatic spread is a first-class output, not a post-process.
- Spectral rendering: rays colored by wavelength. Toggle the ghost layer and TIR highlight live.
- The Abbe glass map is the material picker: hover to reassign glass and watch the focus re-converge. The obtainable region is bounded, so you can't paint with glass that doesn't exist.
- Live detectors: irradiance histogram, flux-weighted centroid, optical path length.

**Interaction**
- Draw a surface freehand and it's fit to a parametric curve (`line → arc → conic → asphere → spline`) with a complexity slider that trades manufacturability against fit residual in real time.
- Every parameter has an on-canvas handle and a numeric twin. Drag the handle or scrub the number, and the light follows. No Apply button, no modal dialogs.
- Pop-up inspector cards (several at once, one per element), an adjustable grid with snapping, copy/paste/delete, per-parameter and whole-element pinning, plus zoom & pan.

**Workflow**
- Layers with show/hide and per-layer opacity; onion-skin hidden layers to compare candidate architectures superimposed, and flip between them.
- A non-linear undo tree: every settled edit snapshots the scene, branches are visible in a sidebar, and you can jump to any of them (not just a flat undo stack).
- A variation gallery: save designs as live, independently-traced thumbnails and compare them on the same detector, then click to load one.
- Wavefront scrub: a timeline that draws rays only up to a propagation time *t* (clipped by cumulative optical path length), so you can watch the front sweep through the system.

**Under the hood**
- Two-tier tracer: a fast f32 path (GPU compute via `wgpu`, with a CPU fallback) draws the picture; an exact f64 reference computes every displayed *number*. Numbers never come from the fast path.
- Scene state lives in Rust; the browser holds only UI state.
- `.ocanvas`: a plain-JSON, human-readable, diffable scene format.

## Try it

Prerequisites: [Rust](https://rustup.rs) with the `wasm32-unknown-unknown` target, [`wasm-pack`](https://rustwasm.github.io/wasm-pack/), and [Node.js](https://nodejs.org) 18+.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

cd web
npm install
npm run dev          # builds the wasm, starts Vite; open the printed URL
```

Then pick the draw tool and sketch a lens surface, slide the complexity slider, double-click an element to inspect it, drag it around, and scroll to zoom. A WebGPU-capable browser uses the GPU tracer; others fall back to CPU.

Prefer the engine without the UI?

```sh
cargo run -p oc-trace --example singlet         # trace an N-BK7 singlet, dump .ocanvas
cargo run -p oc-gpu --release --example drag    # interactive tracer frame budget
```

## Performance

The target is sub-16 ms frames so interaction feels instant. Trace time on a mid-range GPU (release build):

```
2000 rays × 8 bounces:  0.72 ms/frame   (~1380 fps)   ✅ target ≤ 8 ms
8000 rays × 8 bounces:  0.99 ms/frame   (~1000 fps)
```

The fast (f32) and exact (f64) tracers are cross-validated on every `cargo test` against a set of golden scenes, the project's single most important correctness guard:

| Comparison | Worst ray-landing error |
|---|---|
| CPU f32 vs f64 reference | 8 × 10⁻⁵ mm |
| GPU vs CPU f32 | 1.7 × 10⁻⁴ mm |

The reference tracer also agrees with analytic solutions (Snell's law, the TIR critical angle, thin-lens focus, the parabolic mirror, and Fresnel R/T at normal and Brewster incidence) to between 10⁻⁶ and 10⁻¹².

A performance-regression gate (`cargo test -p oc-gpu --release --test perf_regression`) keeps every path (f64 CPU, f32 CPU, and f32 GPU on each `wgpu` adapter) from regressing against a committed baseline. CPU costs are machine-normalized so the baseline is portable; real GPUs must additionally hold the ≤ 8 ms @ 2k×8 budget.

## Validated against Optiland

Beyond the internal f32/f64 cross-check, the f64 reference tracer's forward trace is validated against the main [Optiland](https://github.com/optiland/optiland) engine, the 3D sequential tracer this canvas hands designs off to. Matched rotationally-symmetric systems are traced in both and their final ray position and direction compared meridional-ray by meridional-ray (canvas `+x` ↔ Optiland `+z`):

| Case (constant index unless noted) | Worst Δ direction | Worst Δ position |
|---|---|---|
| spheres & planes: biconvex, plano-convex, meniscus | ≤ 5 × 10⁻¹⁶ | ≤ 2 × 10⁻¹⁵ mm |
| conic (parabola, k = −1) | 6 × 10⁻¹⁶ | 2 × 10⁻¹⁵ mm |
| air-spaced & cemented doublets | 1 × 10⁻¹⁵ | 2 × 10⁻¹⁵ mm |
| mirror (reflection) | 2 × 10⁻¹⁶ | 0 |
| N-BK7 dispersion (F/d/C) | 4 × 10⁻¹⁶ | ~10⁻¹⁵ mm |
| even asphere (r⁴, r⁶) | 4 × 10⁻¹¹ | 6 × 10⁻⁹ mm |

The two engines are numerically identical on the forward trace: vector Snell's-law refraction and reflection agree to machine precision, and the N-BK7 Sellmeier index is bit-identical. Even aspheres agree to ~10⁻⁹ because both sides run an independent Newton solver seeded by the analytic conic: the reference finds the intersection *point* to machine precision but takes its surface normal by finite difference. (Positions are compared after removing the reference tracer's deliberate 10⁻⁷ mm ray-spawn offset.) See [`validation/`](validation/) to reproduce.

The wavefront is validated the same way. A wavefront is a surface of constant optical path, so the M4 wavefront scrub draws the equal-OPL locus, and the wavefront aberration is the optical path difference, each ray's OPL to the image plane relative to the axial ray. The reference tracer's `Σ nᵢ·dᵢ` matches Optiland's `RealRays.opd` in OPD to ≤ 4 × 10⁻¹⁴ mm on spheres and ≤ 8 × 10⁻¹⁰ mm with N-BK7 dispersion (the even-asphere case reaches ~2 × 10⁻⁸ mm, bounded by the same finite-difference normal). Taking OPD relative to the axial ray cancels the ray-spawn-offset piston. `cargo test -p oc-trace --test wavefront` locks two of these Optiland OPD sets in as golden values.

## The handoff to Optiland

The canvas decides *what* to build; [Optiland](https://github.com/optiland/optiland)
makes it right. That seam is M5: the `.ocanvas` format is frozen and versioned
(documented in [`docs/ocanvas-format.md`](docs/ocanvas-format.md)), and the
[`oc-py`](crates/oc-py) PyO3 crate round-trips a scene to an Optiland `Optic` and
back.

Sketch a system in the browser and hit **⤓ save .ocanvas** (the Scene panel, top
left); **⤒ open** loads one back. Then hand the file to Optiland in Python:

```sh
cd crates/oc-py && maturin develop --release   # build the `optiland_canvas` wheel
pip install optiland
```

```python
import optiland_canvas as oc
import matplotlib.pyplot as plt

optic = oc.to_optic("design.ocanvas")          # export: scene → Optiland Optic
optic.draw(); plt.show()                       # Optiland's own layout plot

open("out.ocanvas", "w").write(oc.from_optic(optic))   # import: Optic → scene
```

Export is exact for the rotationally-symmetric refractive subset: a
prescription survives `scene → Optic → scene` with parameter-level equality,
asserted on a corpus of singlets, doublets, aspheres, and cemented pairs in both
Rust (`cargo test -p oc-core`) and Python (`pytest crates/oc-py/tests`, against a
live Optiland). Fidelity is reported, never faked: anything the
sequential model can't hold (splines, GRIN, tilted or off-axis surfaces) comes
back as a `FidelityWarning` (or a hard error under `strict=True`), not a silent
approximation. "A tool that lies about fidelity is worthless to a professional."

## Architecture

A Cargo workspace of small crates plus a TypeScript shell:

```
crates/
  oc-core    scene model, geometry, materials, dispersion, .ocanvas serde
  oc-trace   reference tracer: CPU, f64, exact (the source of truth)
  oc-gpu     interactive tracer: wgpu compute (f32) + CPU-f32 fallback
  oc-fit     freehand stroke → parametric surface fitting
  oc-openuc2 openUC2/OptiKit adapter: catalogue, setup, module placement
  oc-wasm    wasm-bindgen bindings; the browser's handle to the scene
  oc-py      PyO3 bindings; .ocanvas ⇄ Optiland Optic round-trip
web/         Vite + TypeScript UI shell
```

Everything is f64 in the core; wavelengths are in micrometres. The kernel is a library first: `oc-core` and `oc-trace` depend on neither the UI nor wasm, and between them pull only `glam`, `serde`, and an optional `rayon`.

## openUC2 mode

At `#openuc2`: browse a catalogue of [openUC2](https://openuc2.com) modules, drag
them onto the 50 mm grid, and trace the light through them. The freeform Canvas is
untouched and remains the default route.

`oc-core` never learns what a cube is. openUC2 is an adapter on top of the generic
core, and what it needed from the core (scene fragments, rigid batch transforms,
scene-table merging, removal remaps) was added generically. Canvas is still 2D,
one layer is traced at a time, and every number still comes from the f64 tracer.

Start with [`docs/openuc2/architecture.md`](docs/openuc2/architecture.md), or open
[the folded-singlet demo](examples/openuc2/README.md):

| | |
|---|---|
| [architecture.md](docs/openuc2/architecture.md) | the design, and what is true of the engine |
| [coordinates.md](docs/openuc2/coordinates.md) | the canonical frame and the grid |
| [catalogue.md](docs/openuc2/catalogue.md) | how the catalogue is built and pinned |
| [setup-format.md](docs/openuc2/setup-format.md) | `.uc2canvas` and the flat `.ocanvas` export |
| [module-fragments.md](docs/openuc2/module-fragments.md) | a module's optics, and the id hazards |
| [fidelity.md](docs/openuc2/fidelity.md) | what this engine will not claim |

## Contributing

A few principles guide the project, worth knowing before a large change:

- **Feedback beats completeness.** When a trade-off pits physical exactness against sub-16 ms response, response wins; the exact answer is computed on idle by the reference tracer. **Never display a number that came from the fast f32 path.**
- **The two tracers must agree.** The f32/f64 cross-validation (`cargo test -p oc-gpu`) is the guard rail: a fast tracer that quietly disagrees with the exact one is worse than none. Keep it green.
- **Scene state lives in Rust.** Don't mirror the scene graph into JavaScript.

Getting set up:

```sh
cargo test                    # whole workspace
cargo test --test analytic    # the analytic optics suite
cargo clippy --workspace
cd web && npm run build       # wasm + typecheck + production bundle
```

The crate boundaries are clean, so a feature usually lives in one crate plus a little `oc-wasm` / `web` glue. Please open an issue to discuss anything substantial before opening a large PR.

## Roadmap

Each milestone ships something you can open and play with.

- ✅ **M0: Reference tracer.** Exact f64 non-sequential tracer + scene model.
- ✅ **M1: It moves.** GPU/CPU interactive tracer; drag a lens, light follows at 60 fps.
- ✅ **M2: It's optics.** Glass catalog, Abbe picker, dispersion, spectral rendering, ghosts, TIR, live detectors.
- ✅ **M3: It's precise.** Freehand→fit with a complexity slider, numeric inspector, grid, snapping, clipboard, pinning, zoom/pan.
- ✅ **M4: It's a workflow.** Layers with onion-skinning, a non-linear undo tree, a live variation gallery, and wavefront scrub.
- ✅ **M5: It's an Optiland tool.** Frozen, versioned `.ocanvas` format; round-trip import/export against the Optiland engine; PyO3 bindings (`optiland_canvas`). See [the handoff](#the-handoff-to-optiland).
- ⬜ **M6: It's fast.** Performance hardening; WebGL2 / CPU fallback path.
- ⬜ **M7: It's a canvas.** GRIN painting: brush strokes that *are* the optics.

**Out of scope** (that's what Optiland is for): interference, diffraction, polarization ray tracing, coatings beyond scalar R/T, thermal, tolerancing, and optimization.

3D was on that list until recently. It is now being built: the canonical scene model becomes 3D, and this 2D canvas becomes a fast slice through it. Nothing is wired up yet, so everything above still describes the engine as it ships. [ADR 0001](docs/architecture/adr/0001-scene3-is-canonical-2d-is-a-view.md) has the reasoning and the costs; [the contract](docs/architecture/canonical-3d-kernel.md) is what the work has to satisfy.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. This is the standard Rust licensing pair: MIT keeps compatibility with GPLv2, while Apache-2.0 adds an explicit patent grant, so anyone reusing the code can pick whichever fits their needs.
