# Optiland Canvas

A browser-based, real-time optical sketchpad. Draw an optical system and watch the light move.

Rust → WebAssembly, with WebGPU for heavy scenes where the browser offers it. Part of the [Optiland](https://github.com/optiland/optiland) family: use the canvas to *ideate*, then hand the design off to Optiland for rigorous analysis.

> **The one rule: there is never a "run" button.** Move a lens and the light moves. Drag a mirror to a grazing angle and total internal reflection lights up while you are still dragging. Every trade-off resolves toward sub-16 ms feedback; exact numbers settle once you let go, computed by an f64 reference tracer.

**Status:** M0–M5 complete and green, and the canonical 3D kernel (PRs 3D-00 to 3D-09) has landed. Under active development (see the [roadmap](#roadmap)).

---

## What it is

Optiland Canvas is a tool for the *ideation* phase of optical design, the part before you commit to a prescription. The physical model is a 3D non-sequential geometric ray tracer, and the canvas is a view over it: place lenses, mirrors, prisms and stops, drag them around, and see refraction, reflection, TIR, dispersion, and stray-light ghosts update live. You edit in a meridional cross-section by default and can switch to a 3D model view; both drive the same scene. When a design is worth pursuing, export it to [Optiland](https://github.com/optiland/optiland), the Python engine for optimization, tolerancing, and full analysis.

**Why a cross-section is still the default view.** For a rotationally symmetric system the meridional cross-section *is* the design, and for prisms and folded paths the top-down plane is how designers already sketch. What changed with the 3D kernel is underneath: rays carry all three direction components, so a skew ray, a fold out of the plane, and a detector with real `mm²` cells are traced rather than approximated, and the cross-section draws where those rays cross it.

## Features

**Optics**
- Non-sequential ray tracing: refraction, reflection, total internal reflection, Fresnel splitting, ghost reflections, Beer–Lambert absorption.
- Per-ray dispersion: closed-form Cauchy from an `(n_d, V_d)` point, or full Sellmeier for catalog glasses. Chromatic spread is a first-class output, not a post-process.
- Spectral rendering: rays colored by wavelength. Toggle the ghost layer and TIR highlight live.
- The Abbe glass map is the material picker: hover to reassign glass and watch the focus re-converge. The obtainable region is bounded, so you can't paint with glass that doesn't exist.
- Live detectors: a flux histogram over the face, read as power, irradiance or radiant intensity ([what a detector measures](docs/detectors.md)), with a flux-weighted centroid and optical path length.

**Interaction**
- Two views over one scene: a meridional cross-section and an orbitable 3D model. Elements drag, tilt and snap to the grid in either, and the rays retrace while you drag.
- The 3D model draws the parts as solids: clear glass, mirrors, blackened stops, and detectors, under a choice of four studio lightings and five backgrounds, with rays either drawn over the parts or hidden by opaque ones.
- In the 3D model, each detector face shows its flux map from the exact trace, and the light source has a glyph you can pick and move. Ray width is adjustable, and the view settings are remembered in the browser.
- Every parameter has an on-canvas handle and a numeric twin. Drag the handle or scrub the number, and the light follows. No Apply button, no modal dialogs.
- Pop-up inspector cards (several at once, one per element), an adjustable grid with snapping, copy/paste/delete, per-parameter and whole-element pinning, plus zoom & pan.
- The component view: the `⬚ component` tab in the top bar opens the selected element on its own, in a window you can drop anywhere over the canvas. On a singlet, drag handles to bend a face, change the thickness or the clear aperture, or reshape a face through four control points that set its aspheric terms. On a plate, drag the reflectance and transmittance curves: a splitter's ratio, a dichroic's edge and width.
- The `▦ gallery` opens ready-made designs: 26 of the sample lenses that ship with Optiland, from a singlet to a 21-element lithography lens, each traced by the canvas and checked ray for ray against Optiland's own trace. See [the gallery](docs/gallery.md).
- Add a round element, a fold mirror, a beam splitter or a dichroic filter from the scene dock, and choose the source (collimated, point cone, Lambertian disc) and how densely it samples. Splitters and dichroics divide each ray into a reflected and a transmitted branch, weighted per wavelength, in both tracers.
- Light **carries its polarization**. The reference tracer puts a Stokes state on every ray, every interface applies its Mueller matrix, and there are ideal polarizers and waveplates to place in a beam. Two crossed polarizers go dark, sliding a third between them at 45 degrees passes an eighth, and two tilted dichroics in series no longer report the answer for unpolarized light. See [polarization](docs/polarization.md).
- A dichroic is a **real thin-film stack**, not a curve: the transfer matrix runs at each ray's own wavelength and angle of incidence, so the band walks toward the blue as you tilt the plate, exactly as a real filter's does. Its edge is solved for the angle the plate is hung at, and the component view draws the layers with every boundary draggable. Checked against Optiland's own transfer-matrix code to 1e-11, and against the angle-shift law every filter vendor publishes: one effective index, fitted once at 45 degrees, predicts the edge to within 0.06 nm from 10 to 50 degrees. See [coatings](docs/coatings.md) and [validating against published data](docs/validation.md).
- **Scalar wave optics, as a library so far.** The `oc-wave` crate propagates a sampled complex field between planes: band-limited angular spectrum, Fresnel transfer function and impulse response, Fraunhofer, and direct Rayleigh–Sommerfeld, each in the regime its source supports, with every step's method, sampling warnings and a power ledger reported. Validated against closed-form results: Heurtley's exact on-axis field of a circular aperture, the Airy pattern and its encircled energy, Fresnel diffraction of a square, Kogelnik and Li's Gaussian beam, Self's thin-lens focus, and the Talbot distance. It is callable from Rust and from Python (`optiland_canvas.wave`), and in the app the detector dock's wave mode carries a source's coherent beam along the traced path in a worker: through lenses face by face, and through both arms of an interferometer whose arms meet on the detector, where they make fringes. See [wave optics](docs/wave-optics.md).
- **DMD and LCoS SLM models, in the wave kernel.** A digital micromirror device as a blazed 2D grating: its orders in closed form, its near field sampled with each mirror state's tilt carried analytically, and a whole 1920 × 1080 device's far field from one transform. Its projector efficiency follows Texas Instruments' DLPA083B method and lands within 2 percentage points of TI's published values for six of seven devices (one is 4.4 pp off, and the docs say so). A phase-only LCoS with grey-level phase tables, dead space, the director's polarization split and fringing, checked against the classic N-level efficiency and an independent quadrature. Lee holograms and Gerchberg–Saxton design. Both are scene elements: the reference tracer and the preview trace them in the geometric limit, mirror by mirror and pixel by pixel, and a coherent wave path carries a laser beam's diffraction along the chain the tracer finds. In the app, ＋ DMD and ＋ SLM add one, and its editor draws the pattern (generated, or an image of your own) beside the orders it makes; see [DMD](docs/dmd.md), [SLM](docs/slm.md) and [the wave path](docs/wave-optics.md#the-wave-path).

**Workflow**

The M4 workflow panels (layers with onion-skinning, the non-linear undo tree, the variation gallery, the wavefront scrub) were built against the 2D scene and were retired from the UI when the canvas became a view over the 3D one. Their code is still in the tree; re-exposing them over `Scene3` has not been done. The freehand fit (`oc-fit`: stroke to `line → arc → conic → asphere → spline` with a complexity slider) is in the same position, engine present and drawing tool not currently wired.

What the app carries today: the detector dock (a heatmap of the face, its polar map when the detector measures intensity, or a spot diagram, with hits, flux, centroid and OPL) and the Abbe glass map as the material picker.

**Under the hood**
- Two-tier tracer: a fast f32 path draws the picture, an exact f64 reference computes every displayed *number*, and numbers never come from the fast path. The f32 path has a `wgpu` compute backend and a CPU mirror; in the browser the WebGPU backend takes over from about 5,000 segments a frame, where it becomes the faster of the two.
- Scene state lives in Rust; the browser holds only UI state.
- The 3D model view renders through three.js (WebGPU, with a WebGL2 fallback) as a disposable picture of the Rust scene. Its meshes are tessellated in Rust from the analytic geometry the tracer uses, and nothing it draws is a measurement: the glass is display shading, not a second ray trace. See [the renderer report](docs/3d-renderer-implementation-report.md).
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

The app opens on a biconvex singlet lit by a collimated white beam, with a detector downstream. Double-click an element to inspect it, drag it to move, use the handle to tilt it, and scroll to zoom. `◳ 3D model` swaps the cross-section for an orbit view of the same scene, drawn on the GPU; its lighting, background, exposure and ray visibility sit in a bar at the bottom. `⬚ component` opens the selected element in its own window: click it, or drag it onto the canvas. Add a fold mirror or a round element from the scene dock. Every number comes from the f64 reference tracer, which a pool of Web Workers traces off the page's thread, so a ghost-lit twenty-element lens never freezes the page ([performance](docs/performance.md)). The picture comes from the f32 tracer, on the CPU for light scenes and on WebGPU for heavy ones when the browser has a device; the status line says which, and in the 3D view it also names the backend that draws the solids.

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
  oc-wave    scalar wave optics (f64): sampled complex fields, FFT propagation
  oc-wasm    wasm-bindgen bindings; the browser's handle to the scene
  oc-py      PyO3 bindings; .ocanvas ⇄ Optiland Optic round-trip
web/         Vite + TypeScript UI shell
```

Everything is f64 in the core; wavelengths are in micrometres. The kernel is a library first: `oc-core` and `oc-trace` depend on neither the UI nor wasm, and between them pull only `glam`, `serde` (with `serde_json`), and an optional `rayon`. The one FFT in the workspace, `rustfft`, is confined to `oc-wave` ([ADR 0003](docs/architecture/adr/0003-scalar-wave-optics-engine.md)).

## openUC2 mode

At `#openuc2`: browse a catalogue of [openUC2](https://openuc2.com) modules, drag
them onto the 50 mm grid, and trace the light through them. The freeform Canvas is
untouched and remains the default route.

`oc-core` never learns what a cube is. openUC2 is an adapter on top of the generic
core, and what it needed from the core (scene fragments, rigid batch transforms,
scene-table merging, removal remaps) was added generically. This route still runs
on the 2D engine: one layer is traced at a time, in the plane, and every number
still comes from the f64 tracer. Porting it onto the 3D scene has not been done.

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
- ✅ **M4: It's a workflow.** Layers with onion-skinning, a non-linear undo tree, a live variation gallery, and wavefront scrub. Shipped, then retired from the UI when the canvas became a view over the 3D scene. The code is still in the tree, unexposed.
- ✅ **M5: It's an Optiland tool.** Frozen, versioned `.ocanvas` format; round-trip import/export against the Optiland engine; PyO3 bindings (`optiland_canvas`). See [the handoff](#the-handoff-to-optiland).
- ⬜ **M6: It's fast.** Performance hardening; WebGL2 / CPU fallback path.
- ⬜ **M7: It's a canvas.** GRIN painting: brush strokes that *are* the optics.
- ✅ **M8: It diffracts.** Scalar wave optics, and the two devices that need it: the digital micromirror device (DMD) and the phase-only LCoS spatial light modulator (SLM). Every model is validated against published sources, which are named in the tests and the docs. See [ADR 0003](docs/architecture/adr/0003-scalar-wave-optics-engine.md) and the [acceptance cases](docs/architecture/acceptance-manifest.md#wave-cases-c35-onward).
  - ✅ W0: the `oc-wave` crate, its transform and its conventions.
  - ✅ W1: free-space propagation (band-limited angular spectrum, Fresnel, Fraunhofer, direct Rayleigh–Sommerfeld) and thin elements, against closed-form results ([wave optics](docs/wave-optics.md)).
  - ✅ W2: a zoomed far field (chirp-z), Python bindings (`optiland_canvas.wave`), and Optiland's FFT and matrix-DFT PSFs as a cross-check, agreeing to round-off.
  - ✅ W3: DMD and LCoS device models: closed-form orders, sampled near fields, whole-device far fields, and TI's projector efficiency, against Popoff et al. (2026; their 2024 text was retracted for its blazed-number equation) and TI's DLPA083B ([DMD](docs/dmd.md), [SLM](docs/slm.md)). The superpixel method is still to come.
  - ✅ W4: DMDs and SLMs as scene elements, traced in the geometric limit by the reference and the preview alike; a coherent wave path along the traced beam, resumable in steps ([wave path](docs/wave-optics.md#the-wave-path)).
  - ✅ W5: the app: DMD and SLM from the tools dock, an editor with the pattern and its orders, TI presets and image import; a source's coherent beam, saved with the document; the detector dock's wave mode, run in a worker. The 3D view does not draw the pattern yet.
  - ✅ W6: refractive optics in the wave path, face by face (geometric field tracing across each surface, diffraction between), and a diffracted readout that resolves a focus; a singlet's PSF agrees with Optiland's `HuygensPSF` (Strehl 0.862 against 0.864).
  - ✅ W7: ray-traced diffraction orders for projector-style, incoherent illumination: each hit spawns the brightest orders of the local unit cell, the rest booked in the energy ledger as untraced orders; TI's method through the tracer agrees with the closed-form integration to 2e-6. A wave path that follows a chosen order through a Fourier-plane filter is still to come.
  - ✅ W8: tilted planes and off-axis windows: the angular spectrum onto any plane (Matsushima, Schimmel & Wyrowski 2003) and on any window (chirp-z), so a tilted detector is read on its own plane; oblique elements are still handed over by projection, and say so.
  - ✅ Reviewed against a written specification of coherent wave propagation (a scattering-network, power-coordinate model with frozen benchmarks), then fixed and extended in batches. I1: apertures keep their outer extent as the tracer does, an element's outline and a lens's rim clip the beam at the field's own sampling, an LCoS splits unpolarized light, and the field keeps one phase clock. I2: interferometers, where splitter arms that meet on the detector are added as fields, with unitary splitters (a Mach–Zehnder's two ports carry all the power at every phase); a source's spectral lines are added in power, so two lines wash fringes out where their beats cancel. I3: planar layers plane wave by plane wave, a plane-parallel plate in closed form with its round trips (an etalon's fringes) and flat coatings from their transfer matrix, s and p as two incoherent modes. I4: the field's window follows the beam into a focus and out again, sized from its second moments, and the automatic propagator choice uses the paraxial step's exact error on the field. I5: polarization on the wave path, two components per sample turned through each element's s and p, polarizers and retarders as the tracer's Jones matrices, so a half-wave plate in one arm of an interferometer takes its fringes away and an LCoS's two beams add in power. I6: a lens face traced along a reference fixed by the geometry, the beam's paraxial wavefront carried through every element, so the transfer is one linear operator, and a modulator's orders carried apart through the lenses that follow, each about its own direction. Then checked by an outside physics reviewer, given a brief of the models and the code (October 2026), and fixed in batches again. E1: the ledger in two groups, physical (never negative) and numerical (reported term by term), with its own checks on every result, and one junction rule for the arms a splitter, a grating or an LCoS sends apart. E2: a grating's order carries the grating's own phase, so it leaves with its momentum and its phase moves with the grating; a window fitted off the chief ray is carried where it is, by the zoomed and the rotated spectrum, a traced face's ray grid and a modulator's footprint alike; the blazed number from the 2026 paper, held against its figure and against the kernel's own facet kinematics; the readout padded so that no periodic copy of the field lands on the detector; and two gauges, a baked carrier and a rigidly moved scene, that change nothing. E3: one readout for every arm, on the detector's own grid, its points resolving the field's whole band so that the sum is exact, each arm's carrier analytic and the fringes between arms integrated over each cell exactly (two plane waves give the pixel's sinc factors to round-off), the irradiance by a declared detector model, an oblique arm's two components turned onto the face without loss, and the tilted-plane spectrum read between its samples by gridding, accurate to 1e-9 where it was 1e-3; arms that meet at a splitter and leave it as one beam join there, so an aperture behind a recombining splitter clips their sum and the unread port gets their interference once. E4: the propagation choice reads the spectrum as the steps pad it and calls a step it cannot certify by name; the tilted-plane spectrum keeps every wave that crosses the plane and books what it cannot hold by cause; a transfer's departure is read from the field's spectrum as well as its phase, so a field of several directions cannot hide them; phase steps are read between adjacent samples; a window that lost light is tried again twice as wide; and the propagators converge on exact fields as pitch, window, padding and output sampling are refined one at a time. E5: a plate's round trips beyond the window are booked exactly, as a field and its interference with those kept; a thin low-index gap past its critical angle is refused by name, since frustrated total internal reflection is not modelled; a coated plate takes its round trips as it declares, first pass, incoherent or coherent, so a monochromatic beam through a parallel coated plate shows its fringes, its lateral shift and its delay; and the ideal splitter is shown passive for any excitation of both ports.

**Out of scope** (that's what Optiland is for): birefringence, thermal, tolerancing, and optimization. Interference and diffraction left the list in scalar form: `oc-wave` propagates coherent fields, models DMDs and SLMs, and carries a beam along a traced path, the arms of an interferometer added as fields and a source's lines in power. Rigorous vector diffraction (RCWA, FDTD), vector high-NA focusing, a cavity's round trips beyond a plane-parallel plate's, and partial coherence beyond lines and polarizations added in power stay out, and features below about 14 wavelengths, where scalar theory's error passes ±5%, are flagged. Coatings left that list when they became real multilayer stacks, and polarized transport followed: the reference tracer carries a Stokes state, interfaces apply their Mueller matrices, and there are ideal polarizers and waveplates to put in a beam ([`docs/polarization.md`](docs/polarization.md)). A polarizing beam splitter is a part now, and it is the one whose behaviour is derived rather than declared: a quarter-wave stack cemented in the glass the Brewster condition demands, so its contrast and its acceptance angle come out of the transfer matrix. A source that emits polarized light on its own is still missing. The f32 preview stays unpolarized on purpose.

3D was on that list until the canonical-3D migration, which has now landed. The scene model is 3D, the f64 reference tracer traces skew rays through it, and this canvas is a view over that scene rather than the model itself. Two pieces are still open: Python round-trips scene files rather than calling the tracer, and the `Scene3` to `Scene` rename is pending. [ADR 0001](docs/architecture/adr/0001-scene3-is-canonical-2d-is-a-view.md) has the reasoning and the costs; [the contract](docs/architecture/canonical-3d-kernel.md) tracks what remains.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. This is the standard Rust licensing pair: MIT keeps compatibility with GPLv2, while Apache-2.0 adds an explicit patent grant, so anyone reusing the code can pick whichever fits their needs.
