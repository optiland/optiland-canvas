# The canonical 3D non-sequential kernel

**The canonical physical model is 3D. The Canvas you draw on is a view over it.**

That sentence is the whole architecture, and it reverses the project's founding
assumption. Why, and what it costs, is [ADR
0001](adr/0001-scene3-is-canonical-2d-is-a-view.md). This page is the contract
the migration is held to.

**Status: PR 3D-01 of 12.** Nothing here traces yet. The 3D scene model and
tracer are behind off-by-default feature flags, and the shipped engine is exactly
the 2D one the README describes. See [migration state](#migration-state).

## One kernel

```text
                     ┌──────────────────────────┐
                     │  canonical Scene3 in Rust│
                     └─────────────┬────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     f64 reference tracer   validated planar reducer   f32 3D preview
       authoritative          optional fast path       CPU / WebGPU
              │                    │                    │
              └────────────────────┴────────────────────┘
                                   │
                     browser / WASM / PyO3 / Optiland
```

Optiland provides Python scene builders, analyses, reports, optimization, and
plotting **around** this kernel. It does not maintain a second production
implementation of nearest-hit transport, branching, absorption, or surface
interaction.

## The invariants

These are the rules a reviewer should reject a PR for breaking.

### One canonical scene

The persisted scene is 3D; a 2D view is derived from it. No JavaScript copy of
the scene graph, ever: the browser holds UI state and an opaque Rust handle, as
it does today.

### No invented geometry

A drawn outline does not say whether it is a revolved lens or an extruded prism,
and the two trace differently. So loading a legacy 2D body **must not** assume it
is revolved, extruded, infinitely deep, or one millimetre thick.

A body with no reliable 3D interpretation loads as `ProfileConstruction3::PlanarOnly`.
It displays and traces in compatibility mode; a full-3D trace returns a typed
capability error until somebody promotes it on purpose. Authoring tools state
their construction default up front (lens tool revolves, prism tool extrudes with
a visible depth). A freehand closed shape **asks**.

### Numerical authority

- f64 full-3D reference results are authoritative.
- A proof-backed f64 planar reduction is authoritative **only** while it is
  demonstrably equivalent to the 3D scene.
- f32 CPU and GPU results never supply a displayed number. This is the existing
  two-tier rule and it does not relax.
- Every displayed number carries the scene revision and trace configuration that
  produced it. Stale results are discarded, not shown.

### Deterministic reference transport

The reference tracer splits reflected and transmitted branches deterministically.
Russian roulette stays off for authoritative results. Monte Carlo may arrive
later as an explicit analysis mode; it does not replace deterministic reference
transport.

### Fidelity is reported, never faked

Every compiler and backend produces a structured report:

```rust
pub struct CapabilityReport {
    pub supported: Vec<PhysicsFeature>,
    pub exact: Vec<PhysicsFeature>,
    pub approximated: Vec<Approximation>,
    pub rejected: Vec<CapabilityIssue>,
}
```

A backend that cannot trace a scene **rejects** it. It does not turn an arc into
a chord, an asphere into its base conic, or an unsupported source into a
superficially similar one and stay quiet. This is the existing project rule
(`docs/openuc2/fidelity.md`) applied to geometry rather than to modules: *a tool
that lies about fidelity is worthless to a professional.*

### Interaction performance is protected

The current planar preview stays available throughout. **No intermediate PR
removes the current browser experience before its replacement is green.** The
`2,000 rays × 8 interactions ≤ 8 ms` gate stays green, and no 3D PR regresses it
by more than 10% without an accepted benchmark update.

### Right-handed coordinates only

Every external coordinate conversion lives in one named module with round-trip
and handedness tests. Inline axis swaps are forbidden. See
[coordinates.md](coordinates.md).

### Stable identity

Persistent references use stable 64-bit ids, never vector positions. A stop, a
surface response override, a selection, an optimization variable, a detector, or
a path record **survives insertion and removal of unrelated objects**.

Ids are allocated by `Scene3`, never reused within a scene lifetime, and never
inferred from a vector index. Vector order is presentation order only. Removal
does not renumber survivors.

## What the model gains

The things a plane cannot hold, which are the point of the exercise:

| | Baseline | Canonical 3D |
|---|---|---|
| ray | in-plane only | skew, all three direction components |
| detector | line segment, bins of length | area, cells with real `mm²` |
| source | fan with in-plane measure | solid angle, area, deterministic sampling |
| aperture | none, effectively | finite, bidirectional, physical |
| body | closed loop of 2D edges | revolved / extruded / analytic solid |
| fold | in-plane only | out-of-plane, onto a non-coplanar detector |
| fans | meridional | tangential **and** sagittal, distinct |

## Units

Millimetres for geometry and optical path. Micrometres for wavelength. Inverse
millimetres for absorption. Radians internally. Direction vectors and normals
unitless and normalized. Detector area in `mm²`, solid angle in steradians.

Flux is **relative** unless the source is explicitly calibrated. Nothing is
called absolute irradiance without radiometric calibration to justify the unit,
which is the rule the detector histogram already follows today.

## Migration state

The migration splits in two, and the line between them is what ships.

The **math** layer (`oc_core::vec3`, `oc_core::id`, `oc_core::coordinates`) is
always compiled. It is additive and self-contained: no existing type changed, and
the only 2D caller is `oc-openuc2`, which uses `coordinates` because §3.7 allows
exactly one conversion module. Gating it would have forced `oc-openuc2` to keep a
duplicate axis mapping, which is the drift that rule exists to prevent.

The **model** layer (`Scene3`, 3D geometry, the 3D tracer, the 3D preview) is
behind a feature flag that is **off by default**, so the shipped engine cannot
reach a half-built scene model and the migration cannot regress the browser while
it is in progress.

| Flag | Crate | Gates |
|---|---|---|
| `scene3` | `oc-core` | the canonical 3D scene model and geometry |
| `trace3` | `oc-trace` | the f64 full-3D reference tracer (implies `scene3`) |
| `gpu3` | `oc-gpu` | the full-3D preview, f32 CPU and WGSL (implies `scene3`) |

`gpu3` is **not** a newer `gpu`. `gpu` picks a *backend* (wgpu compute vs the CPU
f32 mirror); `gpu3` picks the *dimensionality* of the scene being previewed. All
four combinations mean something, and `gpu3` without `gpu` is the important one:
it is the CPU f32 oracle the shader is validated against.

| PR | Delivers | State |
|---|---|---|
| 3D-00 | this contract, the ADR, coordinate docs, test scaffolding, flags | **done** |
| 3D-01 | f64 3D math, stable ids, coordinate conversion | **done** |
| 3D-02 | `Scene3` schema, `.ocanvas` v2, v1 migration | next |
| 3D-03 | 3D analytic geometry and containment | |
| 3D-04 | the f64 full-3D reference tracer | |
| 3D-05 | the exact planar reducer and legacy compatibility | |
| 3D-06 | 3D sources, detectors, apertures, authoring constructors | |
| 3D-07 | CPU f32 3D preview | |
| 3D-08 | WebGPU full-3D preview | |
| 3D-09 | the 2D slice/projection UI over `Scene3` | |
| 3D-10 | PyO3 direct trace API and the Optiland wrapper | |
| 3D-11 | cleanup, canonical rename, claim updates | |

What each suite owes, and which PR owes it, is
[acceptance-manifest.md](acceptance-manifest.md).

## Definition of done

The project may call itself a 3D non-sequential engine only when **all** of
these are true. Until then the README says 2D, because it is.

- `.ocanvas` v2 stores explicit 3D construction;
- a skew ray with all three direction components traces through a 3D refractive
  system;
- an out-of-plane mirror sends light to an out-of-plane detector;
- tangential and sagittal fans are distinct and validated;
- finite 3D apertures clip correctly;
- detector cells have physical area;
- full-3D f64 results are authoritative;
- full-3D f32 CPU and GPU previews are cross-validated;
- current planar performance is preserved;
- legacy v1 files load without invented geometry;
- Python calls the Rust non-sequential tracer;
- the Optiland sequential cross-validation corpus includes skew rays;
- all approximations are visible through capability reports;
- the energy ledger accounts for launched flux;
- no unsupported geometry is represented by a fake substitute.

## Not in this epic

The data model leaves room for these. The migration does not implement them, and
they do not delay it:

coherent wave propagation, diffraction PSF/MTF, full Jones polarization,
physical thin-film optimization, fluorescence, nonlinear optics, thermal
deformation, CAD-grade Boolean solids, general watertight triangle-mesh editing,
distributed rendering, differentiable tracing.

Once the 3D reference path exists (PR 3D-04), physics features land in this
order, and each targets the 3D path **first**: volume absorption and extinction
import, spectral surface R/T/A, detector QE and area-normalized signal, broad
source spectrum, scale-aware fuzz hardening, true B-spline/NURBS, extended
sources, scatter/roughness, GRIN eikonal transport, polarization and coatings.

The first two of those are the openUC2 spec's P1, the gate on any honest filter
or dichroic claim. They are no longer satisfiable by a 2D implementation.
