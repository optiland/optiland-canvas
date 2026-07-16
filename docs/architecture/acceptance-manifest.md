# Acceptance manifest

The evidence the 3D migration owes, listed before it exists.

A test suite that has not been written yet is still tracked here, with the file
it will live in and the PR that owes it. The point is that "3D works" decomposes
into a fixed list of measurements agreed at the start, so the finish line cannot
drift toward whatever happened to get implemented. Every command below is
already runnable today: the suites exist as gated placeholders and report zero
tests until their PR fills them in.

Tolerances are **initial gates, not marketing claims**. Observed results are
usually far tighter (the 2D engine agrees with Optiland to ~1e-15 on spheres
against a 1e-10 gate). Record the tighter number; do not ratchet the gate to it
without a reason.

## Suites

| File | Feature | PR | Holds |
|---|---|---|---|
| [`oc-core/src/coordinates.rs`](../../crates/oc-core/src/coordinates.rs) | always on | 3D-01 ✅ | frame mappings: round trip, handedness, meridional compatibility |
| [`oc-core/src/vec3.rs`](../../crates/oc-core/src/vec3.rs) | always on | 3D-01 ✅ | `Isometry3` compose/inverse/round-trip, quaternion validation and canonicalization, `Ray3` normalization, conservative `Aabb3` |
| [`oc-core/tests/scene3.rs`](../../crates/oc-core/tests/scene3.rs) | `scene3` | 3D-02 ✅ | v2 schema, validation, v1 migration, id stability |
| [`oc-core/tests/geom3.rs`](../../crates/oc-core/tests/geom3.rs) | `scene3` | 3D-03 | analytic intersection, analytic normals, containment, AABBs, fuzz |
| [`oc-trace/tests/analytic3.rs`](../../crates/oc-trace/tests/analytic3.rs) | `trace3` | 3D-04 | 3D optical fixtures; the proof the engine is not merely meridional |
| [`oc-trace/tests/energy3.rs`](../../crates/oc-trace/tests/energy3.rs) | `trace3` | 3D-04 | the energy ledger closes |
| [`oc-trace/tests/planar_equivalence.rs`](../../crates/oc-trace/tests/planar_equivalence.rs) | `trace3` | 3D-05 | the 2D path is an exact reduction of the 3D one |
| [`oc-gpu/tests/cpu3_vs_f64.rs`](../../crates/oc-gpu/tests/cpu3_vs_f64.rs) | `gpu3` | 3D-07 | 3D f32 CPU against the f64 reference |
| [`oc-gpu/tests/gpu3_vs_cpu3.rs`](../../crates/oc-gpu/tests/gpu3_vs_cpu3.rs) | `gpu3` | 3D-08 | 3D WGSL against the 3D f32 CPU mirror |

Each file's module doc is the authoritative list of what that suite owes. The
table says where; the file says what.

## The release proofs

Five of these are not ordinary tests. They are the specific claims that separate
"3D" from "2D with three coordinates", and the release is defined by them
(§20):

1. **Skew ray through a refractive system.** All three direction components
   non-zero, through a centered spherical singlet, against Optiland's sequential
   real-ray tracer.
2. **Tangential and sagittal fans are distinct.** For an off-axis field, the two
   fans produce different ray coordinates where theory says they must. A
   meridional engine cannot produce this, so it is the single sharpest test that
   the migration actually happened.
3. **Out-of-plane fold.** A mirror redirects a beam out of `Z = 0` and hits a
   non-coplanar detector.
4. **Finite 3D apertures clip**, from both propagation directions, on circular,
   annular, rectangular, and polygonal openings.
5. **Detector cells have area.** Uniform illumination gives equal relative flux
   density across equal-area cells.

## Tolerance gates

**Planar equivalence** (§15.3), 3D traced in `Z = 0` against the original 2D:

| Quantity | Lines/conics | Even asphere |
|---|---:|---:|
| position | `<= 1e-10 mm` | `<= 1e-8 mm` |
| direction | `<= 1e-12` | `<= 1e-12` |
| OPL | `<= 1e-10 mm` | `<= 1e-8 mm` |

The asphere column is looser only until an independent solver tolerance is
demonstrated tighter. The current 2D asphere takes its normal by finite
difference, which is the known ~1e-9 floor; 3D uses the analytic derivative and
should beat it.

**Optiland cross-validation** (§15.4), extended from meridional to skew:

| Quantity | Sphere/plane/conic | Even asphere |
|---|---:|---:|
| direction norm difference | `<= 1e-12` | `<= 1e-9` |
| final position | `<= 1e-10 mm` | `<= 1e-7 mm` |
| OPD | `<= 1e-10 mm` | `<= 1e-7 mm` |

Required corpus: plane parallel plate, biconvex singlet, plano-convex singlet,
meniscus, asphere, air-spaced doublet, cemented doublet, tilted mirror,
finite-conjugate system, three wavelengths, and an off-axis field with sagittal
and tangential pupil samples.

**f32 and GPU parity** (§15.5):

| Comparison | Gate |
|---|---|
| CPU f32 vs f64 | `<= max(5e-2 mm, 5e-5 * scene_scale)` |
| GPU vs CPU f32 | `<= max(1e-2 mm, 1e-5 * scene_scale)` |

Branch type, TIR flags, detector cell, and termination reason must agree except
within a documented boundary tolerance. An unsupported primitive must get the
same capability classification on both backends.

**Energy** (§9.8): for an uncapped deterministic scene,

```text
launched ≈ escaped + absorbed_volume + absorbed_surface
         + blocked_aperture + terminating_detector_flux
```

to `1e-10` relative on analytic fixtures. Flux removed by a threshold or a cap
appears under `truncated_flux`; it never simply stops being in the ledger.

## Performance gates

| Gate | Budget | Owner |
|---|---|---|
| existing planar preview | `2,000 rays × 8` in `<= 8 ms` | every PR, no more than 10% regression |
| full-3D preview | `2,000 rays × 8 × 32 objects`, ghosts off, `<= 8 ms` compute, `<= 16 ms` frame | 3D-08 |
| full-3D stress | `8,000 rays × 8 × 32 objects`, ghosts off, `<= 16 ms` compute | 3D-08 |
| f64 settled trace | `10,000 rays × 24 × 32 objects`, ghosts on (max gen 4), `<= 250 ms` native | 3D-04 |

## Fixture corpora

| Path | Contains | PR |
|---|---|---|
| [`crates/oc-core/tests/data/ocanvas-v2/`](../../crates/oc-core/tests/data/ocanvas-v2/) | v2 schema fixture, generated by `--example emit_v2_fixture` and anti-drift checked | 3D-02 ✅ |
| [`web/public/catalog/openuc2/assets/*.ocanvas`](../../web/public/catalog/openuc2/assets/) | the shipped v1 corpus; all four load through the migration | 3D-02 ✅ |
| `validation/systems_3d.json` | shared 3D system + skew launch-ray spec, read by both engines | 3D-04 |
| `validation/optiland_trace_3d.py` | traces the corpus through real Optiland | 3D-04 |
| `validation/compare_3d.py` | ray-by-ray diff | 3D-04 |

The v1 corpus rule is the strict one: **every current example loads, and no
third dimension is invented doing it**. A migrated v1 body is `PlanarOnly`.

## Commands

### Runs today

Green as of PR 3D-00. The eight 3D suites report zero tests until their PR fills
them in, which is the point: the command list does not change shape as the epic
lands, so a suite cannot go missing without a command going red.

```sh
cargo test --workspace --exclude oc-py
cargo test -p oc-core --test scene3
cargo test -p oc-core --test geom3
cargo test -p oc-trace --test analytic3
cargo test -p oc-trace --test energy3
cargo test -p oc-trace --test planar_equivalence
cargo test -p oc-gpu --test cpu3_vs_f64
cargo test -p oc-gpu --features gpu --test gpu3_vs_cpu3
cargo test -p oc-gpu --release --test perf_regression
cargo clippy --workspace --all-targets --all-features

cd web && npm ci && npm run build && npm run test:e2e
cd crates/oc-py && pip install -e '.[test]' && pytest -q
```

### Arrives with the epic

The 3D validation corpus, which has no counterpart today (PR 3D-04):

```sh
python validation/optiland_trace_3d.py validation/systems_3d.json > optiland_3d.json
cargo run -p oc-trace --example audit_compare_3d -- validation/systems_3d.json > canvas_3d.json
python validation/compare_3d.py canvas_3d.json optiland_3d.json
```

### Not green, and not this epic's to fix

```sh
cargo fmt --all -- --check
```

**This fails on `main` at `bc44b8b`**, before any 3D work, in about twenty places
across `fragment.rs`, `geom.rs`, `material.rs`, and `scene.rs`. The cause is
simply that CI has never run it, so the tree has drifted from rustfmt's
preference (mostly `use` statements a character or two over `max_width`).

It is recorded here rather than fixed here: reformatting four core files inside
an architecture-contract PR would bury a 46-line diff under a few hundred lines
of whitespace, and the whole review value of this PR is that it is small enough
to read. It wants its own commit, and a CI step to go with it, or the drift
returns. New files added by this epic are rustfmt-clean.


## Per-PR checklist

Before opening any PR in this epic:

- [ ] scope matches exactly one PR in the plan; no later-PR feature pulled in
      without necessity;
- [ ] new public types have validation and serde tests;
- [ ] no non-finite user input reaches an unchecked numerical kernel;
- [ ] all coordinate conversions go through the shared module;
- [ ] stable ids are used for persistent references;
- [ ] f64 is the source of every numerical metric;
- [ ] capability reports cover unsupported paths;
- [ ] new f32 behaviour has f64 parity tests; new GPU behaviour has CPU f32
      parity tests;
- [ ] energy removed from a branch is accounted for;
- [ ] scene revision is attached to asynchronous results;
- [ ] existing planar benchmarks were run;
- [ ] `cargo test --workspace --exclude oc-py` passes;
- [ ] `cargo clippy --workspace --all-targets --all-features` passes;
- [ ] `cd web && npm run build` passes;
- [ ] Python tests pass for PRs touching `oc-py`;
- [ ] **documentation describes only implemented behaviour.**
