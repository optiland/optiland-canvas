# Optiland Canvas

A browser-based, real-time, non-sequential optical sketchpad. Rust → WASM →
WebGPU. Open source. Exports to [Optiland](../optiland).

> **The one rule (spec §0): there is never a "run" button.** You move a lens, the
> light moves. Every tradeoff resolves toward sub-16 ms feedback; exact numbers
> are computed on idle by the reference tracer.

See [`optiland-canvas-spec.md`](../optiland-canvas-spec.md) for the full
engineering spec. This repository currently implements **M0 + M1 + M2 + M3**.

---

## Status: M3 — it's precise (green ✅)

> *M3 — It's precise. Freehand → fit pipeline, complexity slider, numeric
> inspector with bidirectional binding, snapping, constraints, pinning.*

Sketch and precision become the **same control** (§5.2), and the fit is an
explicit, adjustable step rather than a hidden one (§5.3):

- **Freehand → parametric, on a ladder** (new [`oc-fit`](crates/oc-fit) crate).
  A stroke is fitted against `Line → Arc → Conic → Asphere(4) → Asphere(8) →
  Spline` in one pass, and **every rung is returned with its geometric residual
  in microns**, so the complexity slider moves up and down the ladder with no
  refit. The pipeline is RDP simplification → corner segmentation → least-squares
  fit per rung, auto-selecting the least-complex rung under tolerance (§5.3.4).
  The fits are exact where it counts: a radius-40 cap recovers its radius to
  **< 1×10⁻⁶ mm**, a parabola recovers `k` to **< 0.05**, and four aspheric
  terms drive a 20 µm departure down to **< 0.1 µm** residual. Residuals are
  **monotone** down the ladder, and `Spline` is the true-freeform escape hatch.
- **The complexity slider changes the light.** A committed drawn surface keeps
  its ladder; sliding it re-extrudes the element and **the rays visibly change**
  — you are watching the cost of manufacturability in real time (§5.3).
- **Every parameter has a handle *and* a numeric twin, bidirectionally bound**
  (§5.2). Drag on the canvas → the inspector number updates; scrub the number
  (hover + scroll) or type it → the light moves live; no Apply button, no modal.
  Escape reverts an in-flight scrub. Position, rotation, and a singlet's radius /
  thickness / semi-aperture / conic constant are all live.
- **The inspector is a pop-up card** — double-click an element to open a compact
  card that **follows the element as you drag**; **several can be open at once**,
  one per element. The two faces of a singlet are edited **independently**:
  `r1`/`r2`, clear aperture `ca1`/`ca2`, conic `k1`/`k2`, each with its own pin.
  Elements rotate by their on-canvas handle.
- **A grid** with adjustable spacing; drawn elements **snap to grid
  intersections** while dragging (hold **Shift** instead for the optical-axis /
  vertex snaps). The engaged guide glows.
- **Snapping** (hold **Shift**, §5.2): the optical axis (`y = 0`) and vertex
  alignment with other elements. Tangency / concentric-with-a-focus /
  equal-thickness are stubbed for a later pass.
- **Full element lifecycle**: **Del** deletes, **Ctrl+C/V** copy/paste (a
  serialized-element clipboard that survives re-indexing), **Ctrl+D** duplicate,
  **Ctrl+X** cut, and arrow keys nudge by one grid step.
- **Pinning** (§5.1): lock a single parameter (📌) or a whole element. Locked
  elements are **visibly ghosted** and reject every edit — enforced in Rust, so
  a pinned radius survives a slider drag and a locked element cannot be moved.

The fitter is covered by **15 unit tests** in `oc-fit`; the whole M3 WASM surface
(fit ladder, commit + rung-swap, per-face inspector round-trip, delete +
copy/paste, pin/lock enforcement, snapping) is **runtime-tested end to end
through the real wasm boundary** (a 37-assertion Node harness). The web shell
(new [`m3.ts`](web/src/m3.ts) module + panels) type-checks and
production-builds clean.

```sh
cargo test -p oc-fit                              # the fitter: radius, k, ladder, spline
cargo build -p oc-wasm --target wasm32-unknown-unknown
cd web && npm run dev                             # draw a surface, slide complexity, scrub the inspector
```

---

## Status: M2 — it's optics (green ✅)

> *M2 — It's optics. Glass catalog, Abbe picker, dispersion, spectral rendering,
> TIR highlight, ghost layer, detectors with live docks.*

The interactive tracer went **spectral and non-sequential**:

- **Dispersion in the tracer.** Each element carries its model — closed-form
  Cauchy for an Abbe point, full Sellmeier for a catalog glass — and each ray
  carries its wavelength, so `n(λ)` is evaluated per ray (GPU + CPU). Chromatic
  spread is a first-class output: the F/d/C foci separate (**F=48.97 < d=49.48 <
  C=49.71 mm** in the test) and match the f64 reference to 1×10⁻⁶.
- **Spectral rendering.** [`color.rs`](crates/oc-gpu/src/color.rs) maps λ→sRGB
  (analytic CIE 1931 fit); rays are drawn in true colour.
- **Ghost layer.** The tracer now splits: transmit **and** the Fresnel
  back-reflection (per-thread stack). The strongest front-surface ghost carries
  exactly R₀ (**0.04000** measured vs analytic), toggleable with an opacity.
- **TIR highlight.** Surfaces in total internal reflection are flagged per
  segment and glow in the UI.
- **The Abbe diagram is the palette (§5.4).** A scatter of real glasses with the
  convex hull of the obtainable region drawn; outside it is hatched and
  "unobtainable." Hovering re-applies the glass live and the chromatic spread
  re-converges under the cursor.
- **Detector docks — honest numbers.** Metrics (irradiance histogram, centroid,
  OPL) come from the **f64 reference** tracer, never the f32 path (§4.1). To run
  in the browser, `oc-trace`'s rayon is now optional (serial on wasm).

Everything is cross-validated (`cargo test -p oc-gpu`) and the WASM API is
runtime-tested end to end. Run the shell as in M1 (`cd web && npm run dev`):
drag the lens, hover the Abbe map, toggle ghosts, watch the dock.

---

## Status: M1 — it moves (green ✅)

> *M1 — It moves. GPU tracer, web canvas, drag a lens, light follows at 60 fps.
> One glass, one wavelength.*

An **interactive f32 tracer** with two interchangeable backends over one shared
algorithm, both cross-validated against the M0 reference:

- **wgpu compute** ([`oc-gpu`](crates/oc-gpu/src/shader.wgsl)) — one thread per
  ray follows its primary path (refract / reflect / TIR) with containment-based
  medium resolution. Runs headless on a real GPU or the software fallback.
- **CPU f32** ([`cpu.rs`](crates/oc-gpu/src/cpu.rs)) — a line-for-line mirror of
  the shader; the no-WebGPU fallback (§4.3) and the oracle the GPU is checked
  against.

**Cross-validation (spec §10.1 — the project's #1 correctness risk).**
[`cross_validate.rs`](crates/oc-gpu/tests/cross_validate.rs) launches identical
rays through the f64 reference, the CPU f32 path, and the GPU, and compares
where each ray lands on golden scenes (lens, parabolic mirror, prism):

| Comparison | Worst \|Δy\| |
|---|---|
| CPU f32 vs f64 reference | **8×10⁻⁵ mm** |
| GPU (RTX 5070, Vulkan) vs CPU f32 | **1.7×10⁻⁴ mm** |

**Frame budget** (`cargo run -p oc-gpu --release --example drag`), §7 target is
≤ 8 ms trace at 2k rays × 8 bounces:

```
 1000 rays × 8 bounces:  0.63 ms/frame  (1590 fps)
 2000 rays × 8 bounces:  0.72 ms/frame  (1382 fps)  ✅ within budget
 8000 rays × 8 bounces:  0.99 ms/frame  (1005 fps)
```

**WASM + web** ([`oc-wasm`](crates/oc-wasm), [`web/`](web)) — `oc-wasm` compiles
to `wasm32-unknown-unknown` and exposes an opaque `Canvas` handle (scene stays
in Rust, §2); `web/` is a Vite + TypeScript shell that drags the lens and
retraces live. The web shell is **scaffolded but not run** in the environment M1
was built in (no browser there); the tracer it drives is fully validated by the
tests above. See [`web/README.md`](web/README.md) to run it.

```sh
cargo run -p oc-gpu --release --example drag      # GPU frame budget + "light follows"
cargo test -p oc-gpu --test cross_validate -- --nocapture   # f32/f64 agreement
cargo build -p oc-wasm --target wasm32-unknown-unknown       # wasm compiles
```

---

## Status: M0 — reference tracer (green ✅)

> *M0 — Reference tracer, no UI. CPU, f64, shapes + materials + Fresnel +
> ghosts. Passes the entire §7 analytic suite. **Nothing else starts until this
> is green.***

M0 is the gate the rest of the project builds on: an exact, non-sequential CPU
ray tracer plus the scene model it operates on. It is a plain Rust library with
no UI, no GPU, and no I/O beyond JSON serialization — but it already traces real
optics and agrees with closed-form solutions to the spec's tolerances.

### What works today

- **Scene model** (`oc-core`): elements with `Line`/`Arc`/`Conic`/`Asphere`/
  `Spline` boundary edges, catalog + Abbe-point + mirror + absorber materials,
  point & collimated sources with line spectra, line detectors, and the
  `.ocanvas` JSON format (human-readable, diffable — spec §6).
- **Dispersion**: closed-form two-term Cauchy from `(n_d, V_d)` (the "brush"
  model, §3.1) and full three-term Sellmeier for a seed glass catalog (N-BK7,
  F2, N-SF11, N-LAK22, fused silica — public-domain data).
- **Reference tracer** (`oc-trace`): non-sequential, f64, BVH-accelerated,
  rayon-parallel. Fresnel splitting into transmitted **and** reflected rays,
  TIR, priority-based medium resolution (no medium stack — §4.2), ghosts,
  Beer–Lambert absorption, depth/flux caps, and unbiased Russian roulette.
- **Detectors**: irradiance histogram, flux-weighted centroid, and optical path
  length, accumulated during the trace.

### §7 analytic acceptance suite

| Case | Test | Agreement |
|---|---|---|
| Snell at a plane | `snell_at_a_plane_full_trace` | < 1e-9 |
| TIR critical angle | `tir_critical_angle` | exact reflect/transmit switch |
| Thin/thick-lens paraxial focus | `paraxial_lens_focus` | < 1e-6 (Richardson→paraxial) |
| Parabolic mirror on-axis focus | `parabolic_mirror_perfect_focus` | < 1e-9, aberration-free |
| Fresnel R/T (normal + Brewster) | `fresnel::tests`, `fresnel_energy_split` | < 1e-12; energy conserved to 1e-6 |
| Luneburg GRIN lens | `luneburg_lens` *(ignored)* | deferred to **M7** (§8) |

Run them:

```sh
cargo test                       # everything (52 tests across the workspace)
cargo test --test analytic       # just the §7 suite
cargo run -p oc-trace --example singlet   # trace an N-BK7 singlet, dump .ocanvas
```

---

## Repository layout

```
optiland-canvas/
  crates/
    oc-core/    # scene model, geometry, materials, dispersion, serde. No I/O, no GPU.
    oc-trace/   # reference tracer. CPU, f64, rayon. Exact. The source of truth.
    oc-gpu/     # interactive tracer. wgpu compute (f32) + CPU-f32 fallback.
    oc-fit/     # freehand stroke → parametric surface fitting (M3). Pure geometry.
    oc-wasm/    # wasm-bindgen bindings; opaque Canvas handle for the web shell.
  web/          # Vite + TypeScript shell (draw, inspect, snap, pin; light follows).
  optiland-canvas-spec.md   # (in the parent workspace)
```

Future-milestone crates are added as workspace members when their milestone
begins:

- `oc-py` (M5) — PyO3 bindings; Optiland import/export.

**Scene state lives in Rust** (spec §2). Everything is f64; wavelengths are in
micrometres.

---

## Design notes / deviations from the spec

Deliberate choices, documented so they are honest rather than silent.

M0:
- **IDs are indices**, not a `Slab`. The spec writes `Slab<Element>`; M0 uses
  `Vec<Element>` with newtype index IDs because it keeps the serialized
  `.ocanvas` JSON clean and diffable (a slab with holes does not). Revisit if
  stable IDs across deletes become necessary.
- **Element-level BVH**, not per-edge. A valid coarser acceleration structure;
  ample at M0 scene sizes. The tracer still refines to the exact nearest edge.
- **Conic hit clamping.** The implicit conic is the whole surface of revolution;
  hits are clamped to the physical cap via a sag envelope so a ray cannot strike
  the far nappe of a large sphere. (This bites every implicit-conic tracer; the
  parabola is immune because it has no far nappe.)
- **Beer–Lambert** machinery is wired through the loop but the seeded materials
  are transparent (α = 0). Colored/absorbing glass plugs into
  `region_absorption`.
- **Russian roulette** is off by default: the *reference* tracer must be
  deterministic. It exists (seeded, unbiased) for the interactive preset.

M1:
- **Interactive tracer is primary-path only** (transmit / reflect / TIR, no ray
  splitting): matches M1's "one glass, one wavelength, ugly" scope and maps
  cleanly onto one GPU thread per ray. Ghost splitting is M2.
- **Monochromatic per trace.** Per-element index is evaluated on the CPU at the
  chosen wavelength and uploaded as a float — no dispersion math in the shader.
- **Per-thread ray loop**, not the wavefront-queue-with-compaction architecture
  the spec targets (§4.3). Ample for 1–2k rays; the queue/compaction upgrade is
  a perf task. Buffers are also rebuilt per trace (persistent buffers +
  progressive accumulation are M6).
- **GPU line/conic only.** Arc/Asphere/Spline are approximated (chord / base
  conic / polyline) in the interactive path; the f64 reference remains exact.
- **Web shell not run here.** M1/M2 were built without a browser; the shell is
  reviewed, its wasm build + TypeScript typecheck, and its wasm API is
  runtime-tested in Node. The tracer it drives is validated by the
  cross-validation tests.

M2:
- **Ghost tree is bounded** by a per-ray segment budget (12 without ghosts, 24
  with) and depth/flux caps — interactive branch explosion is capped, not
  exhaustive. The exhaustive picture is the reference's job.
- **Abbe picker floats in (n_d, V_d)** as `AbbePoint`/Cauchy materials; catalog
  dots are the full-Sellmeier snap targets. The reference-glass scatter is a
  curated `(n_d, V_d)` table (published values); expand from Schott/RII (§10.6).
- **Detector numbers come from the f64 reference**, computed on idle (pointer-up
  / glass commit), never the f32 path — this is why `oc-trace` gained an
  optional-rayon serial mode for wasm.

M3:
- **A drawn stroke becomes a thin plano body** — the fitted surface as the front
  face, a flat back extruded `thickness` along the surface axis, closed by two
  rim lines. Single-surface fitting is the demonstrated commit path; corner
  segmentation (`segment_corners`) exists for splitting a closed outline into
  several surfaces, but multi-surface bodies are a later pass.
- **Least-squares fits run on the full stroke, not the RDP-thinned one**, so the
  reported residual is honest; RDP feeds only the `Spline` control points and the
  corner segmentation. Higher rungs are seeded from lower ones, guaranteeing the
  residual is monotone down the ladder.
- **Conic fit is a linear implicit solve** (`A y² + B z² + z + C = 0`) that
  recovers curvature, conic constant, **and** the vertex offset along the axis in
  one shot; it falls back to the sphere when a shallow cap can't distinguish the
  two (§5.3), so the slider never lies about what the data supports.
- **Snap targets: optical axis + vertex x/y alignment.** Tangency,
  concentric-with-a-focus, and equal-thickness are stubs for a later pass.
- **The interactive preview approximates a fitted Asphere by its base conic**
  (the M1 GPU line/conic limitation); the f64 reference traces the full asphere
  exactly, so displayed numbers stay honest.
- **Web shell now runs here.** Unlike M1/M2, the M3 shell is built and validated
  in-environment: `tsc --noEmit` and `vite build` are clean, and the whole WASM
  surface is exercised through the real wasm module by a Node harness (27/27).

## Roadmap

~~M0 *Reference tracer*~~ ✅ · ~~M1 *It moves*~~ ✅ · ~~M2 *It's optics*~~ ✅ ·
~~M3 *It's precise*~~ ✅ (freehand→fit pipeline, complexity slider, numeric
inspector with bidirectional binding, snapping, pinning) · M4 *It's a workflow* ·
M5 *It's an Optiland tool* · M6 *It's fast* · M7 *It's a canvas* (GRIN). See the
spec's §9.

The single highest correctness risk (spec §10.1) is f32/f64 divergence between
the reference and interactive tracers. The golden-scene cross-validation
harness that guards against it landed with `oc-gpu` in M1 and runs in
`cargo test` on every change — exactly as §10.1 demands ("from M1, not M6").

## License

MIT OR Apache-2.0.
