# Acceptance manifest

The evidence the 3D migration owes, listed before it exists.

A test suite that has not been written yet is still tracked here, with the file
it will live in and the PR that owes it. The point is that "3D works" decomposes
into a fixed list of measurements agreed at the start, so the finish line cannot
drift toward whatever happened to get implemented. Every command below is
runnable today. The suites began as gated placeholders reporting zero tests; with
3D-00 through 3D-09 landed, every suite in the table below is filled. What is
still owed is the 3D validation corpus, listed under fixtures.

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
| [`oc-core/tests/geom3.rs`](../../crates/oc-core/tests/geom3.rs) | `scene3` | 3D-03 ✅ | analytic intersection, analytic normals, containment, AABBs, dedup, transform invariance |
| [`oc-core/tests/geom3_fuzz.rs`](../../crates/oc-core/tests/geom3_fuzz.rs) | `scene3` | 3D-03 ✅ | seeded fuzz: no-NaN, `t > t_min`, unit normals, placement invariance, conservative AABB, scale invariance |
| [`oc-trace/tests/analytic3.rs`](../../crates/oc-trace/tests/analytic3.rs) | `trace3` | 3D-04 ✅ | 3D optical fixtures; the proof the engine is not merely meridional |
| [`oc-trace/tests/energy3.rs`](../../crates/oc-trace/tests/energy3.rs) | `trace3` | 3D-04 ✅ | the energy ledger closes |
| [`oc-trace/tests/planar_equivalence.rs`](../../crates/oc-trace/tests/planar_equivalence.rs) | `trace3` | 3D-05 ✅ | the 2D path is an exact reduction of the 3D one |
| [`oc-trace/tests/sources3.rs`](../../crates/oc-trace/tests/sources3.rs) | `trace3` | 3D-06 ✅ | each source distribution samples its own measure; flux splits to the launched total |
| [`oc-trace/tests/detectors3.rs`](../../crates/oc-trace/tests/detectors3.rs) | `trace3` | 3D-06 ✅ | disc and annulus detectors, area normalization |
| [`oc-trace/tests/apertures3.rs`](../../crates/oc-trace/tests/apertures3.rs) | `trace3` | 3D-06 ✅ | rectangular and polygonal openings clip from both directions |
| [`oc-gpu/tests/cpu3_vs_f64.rs`](../../crates/oc-gpu/tests/cpu3_vs_f64.rs) | `gpu3` | 3D-07 ✅ | 3D f32 CPU against the f64 reference |
| [`oc-gpu/tests/gpu3_vs_cpu3.rs`](../../crates/oc-gpu/tests/gpu3_vs_cpu3.rs) | `gpu3` | 3D-08 ✅ | 3D WGSL against the 3D f32 CPU mirror |
| [`oc-gpu/tests/perf3.rs`](../../crates/oc-gpu/tests/perf3.rs) | `gpu3` | 3D-08 ✅ | the full-3D preview performance gate |

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
| `validation/systems_3d.json` | shared 3D system + skew launch-ray spec, read by both engines | 3D-04, not written |
| `validation/optiland_trace_3d.py` | traces the corpus through real Optiland | 3D-04, not written |
| `validation/compare_3d.py` | ray-by-ray diff | 3D-04, not written |

3D-04 landed the tracer without this corpus, so the skew half of the Optiland
cross-validation is still owed. It is the last unmet release proof that is not
waiting on 3D-10.

The v1 corpus rule is the strict one: **every current example loads, and no
third dimension is invented doing it**. A migrated v1 body is `PlanarOnly`.

## Commands

### Runs today

Green as of 3D-09. The command list deliberately did not change shape as the epic
landed: the 3D suites started out reporting zero tests and now report real ones,
so a suite cannot go missing without a command going red.

```sh
cargo test --workspace --exclude oc-py
# The scene3-gated 3D suites (scene3, geom3, geom3_fuzz) report zero tests above,
# because the flag is off by default; CI turns it on in a scoped step so they run:
cargo test -p oc-core --features scene3
cargo clippy -p oc-core --features scene3 --all-targets -- -D warnings
# The trace3-gated suites (analytic3, energy3) the same way; trace3 implies scene3:
cargo test -p oc-trace --features trace3
cargo clippy -p oc-trace --features trace3 --all-targets -- -D warnings

cargo test -p oc-core --test scene3
cargo test -p oc-core --test geom3
cargo test -p oc-core --test geom3_fuzz
cargo test -p oc-trace --test analytic3
cargo test -p oc-trace --test energy3
cargo test -p oc-trace --test planar_equivalence
cargo test -p oc-gpu --test cpu3_vs_f64
cargo test -p oc-gpu --features gpu --test gpu3_vs_cpu3
cargo test -p oc-gpu --release --test perf_regression
cargo clippy --workspace --all-targets --all-features

cd web && npm ci && npm run build && npm run test:e2e
cd crates/oc-py && pip install -e '.[test]' && pytest -q

# The WebGPU preview in a real browser, against the CPU mirror. Needs a GPU, so
# it is not in CI; it also logs the CPU/GPU timing sweep.
cd web && OC_WEBGPU=1 npx playwright test --project=webgpu
# Where wgpu picks DX12 and FXC rejects shader3.wgsl (seen on Windows), steer
# the native GPU suites to another backend:
WGPU_BACKEND=vulkan cargo test -p oc-gpu --features gpu3 --test gpu3_vs_cpu3
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

**This fails on `main`.** It failed at `bc44b8b`, before any 3D work, in about
twenty places across `fragment.rs`, `geom.rs`, `material.rs`, and `scene.rs`.
Measured again after the 3D epic landed: **187 hunks across 37 files**, of which
8 are files the epic added (`build3.rs`, `scene3.rs`, `tracer3.rs`,
`sequence3.rs`, `scene3_trace.rs` among them). The earlier claim that new files
are rustfmt-clean no longer holds, and the cause is unchanged: CI never runs
`fmt`, so nothing stops the drift (mostly `use` statements a character or two
over `max_width`).

It is still recorded rather than fixed here. Reformatting 37 files inside a
documentation PR would bury the diff, and the reformat is worthless without the
CI step that keeps it from drifting back. Both want their own commit.


## Coating cases C01 to C34

The thin-film acceptance cases: C01 to C10 from the next-steps spec (M4b),
C11 to C34 from the coating-fidelity spec. Each row names the test that holds
the case; `docs/validation-coatings.md` is the generated table of the oracle
matrix (C32), and `docs/validation.md` records the numbers. The oracle cases
run in `crates/oc-py/tests/test_coating_fixtures.py` (frozen) and
`test_thinfilm_tmm.py` (live), from `coating_cases.py`.

| Case | What | Held by |
|---|---|---|
| C01 | zero layers is the bare interface | `oc-core/src/thinfilm.rs` tests; fixture `bare` against `tmm` and Optiland |
| C02 | a zero-thickness layer is its removal | `thinfilm.rs` `c02_*` |
| C03 | a layer split in two is the same layer | `thinfilm.rs` `c03_*` |
| C04 | the quarter-wave AR null | `thinfilm.rs` `c04_*` |
| C05 | one lossless layer, swept | fixture `single_quarter_wave` |
| C06 | lossless multilayers | fixtures `quarter_wave_mirror`, `edge_filter` |
| C07 | absorbing films | fixtures `aluminium_film`, `aluminium_under_dielectric` |
| C08 | the reversed asymmetric stack | `thinfilm.rs` `c08_*`; fixtures `asymmetric_absorbing_*` |
| C09 | grazing and TIR-adjacent stress | `thinfilm.rs` `c09_*` |
| C10 | preview grid against f64 | `oc-gpu/tests/cpu3_vs_f64.rs`, `coating_preview3.rs` |
| C11 | a dispersive catalogue stack | fixture `dispersive_catalogue_stack`; `oc-py/tests/test_film_catalog.py` |
| C12 | 5 um aluminium and 64-layer absorbing stacks | `thinfilm.rs` `c12_*`; fixture `thick_aluminium`; `test_thinfilm_stress.py` (mpmath) |
| C13 | silica on an opaque aluminium substrate | `thinfilm.rs` `c13_*`; fixture `silica_on_opaque_aluminium` |
| C14 | the TIR phase pin, and Optiland's divergence | `oc-py/tests/test_polarization_outputs.py` `test_c14_*` |
| C15 | per-layer absorption | `thinfilm.rs` `c15_*`; `test_thinfilm_tmm.py` `test_c15_*` |
| C16 | deposits against the object ledger | `oc-trace/tests/deposits_by_layer3.rs` `c16_*` |
| C17 | a coated absorbing slab against `inc_tmm` | `oc-py/tests/test_thinfilm_incoherent.py` `test_c17_*` |
| C18 | the plate Mueller matrix | `thinfilm.rs` `c18_*` |
| C19 | ψ and Δ against `tmm.ellips` | `test_polarization_outputs.py` `test_c19_*` |
| C20 | transmission retardance | `test_polarization_outputs.py` `test_c20_*` |
| C21 | reciprocal T, distinct R, reversed | `thinfilm.rs` `c08_*`; fixture `asymmetric_absorbing_reversed` |
| C22 | the two-sided preview bake | `oc-gpu/tests/coating_preview3.rs` `c22_*` |
| C23 | a body-default AR in the preview | `coating_preview3.rs` `c23_*` |
| C24 | angle edges to grazing | `thinfilm.rs` `c24_*`; `coating_preview3.rs` `c24_*` |
| C25 | the matched AR against the single layer and bare glass | `oc-core/src/synthesis.rs` `c25_*` |
| C26 | the V-coat null and its refusal | `synthesis.rs` `c26_*` |
| C27 | the solar band AR | `synthesis.rs` `c27_*` |
| C28 | measured tables | `oc-core/src/response_table.rs` `c28_*`; `oc-trace/tests/measured_tables3.rs` `c28_*` |
| C29 | the thermo-optic shift, and `Off` bit-identical | `oc-trace/tests/thermo3.rs` `c29_*` |
| C30 | v2 to v3 migration changes no number | `oc-core/tests/film_catalog3.rs` `c30_*` |
| C31 | mutations caught | `thinfilm.rs` `mutation` module and its tests |
| C32 | the validation report | `oc-py/tests/test_report_coatings.py` `test_c32_*`; CI step "coating validation report" |
| C33 | `Standard` is the pre-spec kernel bit for bit | `oc-trace/tests/fidelity3.rs` against `tests/data/pre_profile_corpus3.json` |
| C34 | the stamp, the preview's refusals, the round trip, no mixed tables | `fidelity3.rs`; `web/tests/fidelity.spec.ts`; `test_report_coatings.py` `test_c34_*` |

Every MUST in the coating-fidelity spec maps to one of these cases; the spec's
work packages name theirs in their exit criteria.

## Wave cases C35 onward

The scalar wave optics epic ([ADR 0003](adr/0003-scalar-wave-optics-engine.md),
milestones W0–W8). Like the rest of this manifest, the cases are listed before
the evidence exists. A row without ✅ is owed by the milestone it names, in the
file it names.

Every reference cited by a case is named in the test that holds it, with its
DOI where one exists. The master list is the Sources section of
`docs/wave-optics.md` (W1).

| Case | What | Milestone | Held by |
|---|---|---|---|
| C35 | `Cx` moved out of `thinfilm`, and no coating number changed | W0 ✅ | `oc-core/src/complex.rs` tests; C33's corpus (`oc-trace/tests/fidelity3.rs`) unchanged |
| C36 | the 2D transform: Parseval scale, kernel sign, centring | W0 ✅ | `oc-wave/tests/fft.rs` |
| C37 | every propagator conserves power or books the loss | W1 ✅ | `oc-wave/tests/propagate.rs` |
| C38 | Fraunhofer: the square aperture's sinc², the Airy pattern and its encircled energy | W1 ✅ | `oc-wave/tests/analytic_fields.rs` |
| C39 | the exact on-axis Rayleigh–Sommerfeld field of a circular aperture (Heurtley 1973) | W1 ✅ | `analytic_fields.rs` |
| C40 | Fresnel diffraction of a square against the Fresnel integrals; the Voelz–Roggemann regimes | W1 ✅ | `analytic_fields.rs`, `propagate.rs` |
| C41 | a Gaussian beam spreads as Kogelnik and Li say, and a thin lens focuses it where Self says | W1 ✅ | `analytic_fields.rs` |
| C42 | the exact Talbot distance of a cosine grating | W1 ✅ | `analytic_fields.rs` |
| C43 | direct Rayleigh–Sommerfeld against the band-limited angular spectrum | W1 ✅ | `propagate.rs` |
| C44 | chirp-z zoom against the direct DFT; the Airy zero sampled exactly | W2 ✅ | `oc-wave/tests/czt.rs` |
| C45 | the Fraunhofer PSF against Optiland's `ScalarFFTPSF` and `MMDFTPSF` (oracle, not validation) | W2 ✅ | `oc-py/tests/test_wave_oracles.py` |
| C46 | DMD orders on the direction-cosine grid; the sampled far field against the analytic envelope; the factorized against the sampled; unit-cell power | W3a ✅ | `oc-wave/tests/dmd.rs` |
| C47 | the blazed number against Popoff et al. 2024, Fig. 5 | W3a ✅ | `dmd.rs` |
| C48 | average photopic diffraction efficiency against TI DLPA083B Table 6-1 (a vendor simulation; six rows within ±3 pp, the 5.4 µm f/2.4 row +4.4 pp, held to ±5 pp and named) | W3a ✅ | `dmd.rs` |
| C49 | N-level phase quantization, dead space and wavelength detuning | W3b ✅ | `oc-wave/tests/slm.rs` |
| C50 | the LCoS director split; fringing against quadrature | W3b ✅ | `slm.rs` (fringing against direct quadrature in Rust) |
| C51 | holograms: Lee, superpixel (Goorden 2014), Gerchberg–Saxton, spiral phase | W3 (Lee generator, Gerchberg–Saxton and the spiral phase ✅; superpixel owed) | `dmd.rs`, `slm.rs` |
| C52 | modulators in the scene: the geometric ray limit, preview parity, the v3 round trip and the v2 loss | W4a ✅ | `oc-trace/tests/modulator3.rs`; `oc-gpu/tests/cpu3_vs_f64.rs`, `gpu3_vs_cpu3.rs` (the GPU suite run on Vulkan: one Windows machine's D3D12 compiler, FXC, rejects `shader3.wgsl` before and after this work) |
| C53 | the wave path: free space, apertures the trace never logged (their outer extent as the tracer has it), the 4f cutoff, the ledger, agreement with rays at large Fresnel number, a DMD's orders in the scene, an LCoS's unpolarized split, element outlines that clip the beam, a resumable job | W4b ✅; outer extent, outlines and the LCoS split from the wave-optics review (I1) ✅ | `oc-wave/tests/path.rs` |
| C54 | the app: DMD and SLM from the palette, the wave dock | W5 ✅ (the pattern is not drawn in the 3D view; the coarse field during a drag is not done) | `web/tests/wave.spec.ts`, `modulator.spec.ts` |
| C55 | real optics in the wave path: a plate and a singlet against Kogelnik–Li and Self, a tilted plate against the rays, a plate's optical delay on the field's phase, a lens rim against an iris, a singlet's PSF against Optiland's `HuygensPSF` (a fixture from the clone; oracle, not validation) | W6 ✅ (face-by-face field tracing, not a single phase screen: see wave-optics.md); the phase clock and the rim at the field's sampling from the review (I1) ✅ | `oc-wave/tests/path.rs`; `tools/fixtures/huygens_singlet.py` |
| C56 | ray-traced orders against the unit cell; `untraced_orders` in the ledger; TI through the tracer; a wave path that follows a chosen order, so a Fourier-plane iris can pick a Lee hologram's −1 order (moved from W4b: the chief ray leaves a DMD along the facet's reflection, and an iris off it would block the chain) | W7 (orders, the ledger, TI through the tracer and an LCoS staircase ✅; the Lee reconstruction through an iris owed) | `oc-trace/tests/modulator_orders3.rs` |
| C57 | tilted planes and the shifted angular spectrum | W8 ✅ (the tilted plane read through a bilinear spectrum: 5.9e-4, not the 1e-10 an exact method would give; used for tilted detectors, not yet for oblique elements) | `oc-wave/tests/propagate.rs`, `oc-wave/tests/path.rs` |
| C58 | interferometers on the wave path: a Mach–Zehnder's two ports carry all the power at every phase (unitary splitters), a Michelson's fringes λ/sin 2α apart and moved 2k₀d by its mirror, two incoherent lines washing the fringes out where their beats cancel; a continuum refused by name (an LCoS's two polarization beams were, until C61) | the wave-optics review, I2 ✅ | `oc-wave/tests/path.rs`; `oc-trace/src/sequence3.rs` (`extract_sequences3`) |
| C59 | planar layers on the wave path, plane wave by plane wave: an etalon transparent at δ = 7π and dimmest at 7.5π, s and p at 31°, a millimetre plate at its round-trip phase, a single-layer coating against its closed form in reflection and transmission, a plate as free space of t/n to 1.9e-7; planes of incidence that differ refused (until C61) | the wave-optics review, I3 ✅ | `oc-wave/src/planar.rs`, `oc-wave/tests/path.rs` |
| C61 | polarization on the wave path, two components per sample: Malus's law and crossed sheets' extinction, a retarder between crossed polarizers and circular light, tilted crossed polarizers against the f64 tracer, a half-wave plate in a Mach–Zehnder arm (Fresnel–Arago, the fringes restored by an analyzer), an LCoS's two beams added in power, coatings in crossed planes; the chain found through crossed polarizers | the wave-optics review, I5 ✅ | `oc-wave/src/path.rs`, `oc-wave/tests/path.rs`; `oc-trace/src/sequence3.rs` (unpolarized extraction) |
| C60 | sampling where the beam focuses: the paraxial step's exact error against the propagators' own difference, `Auto` by the smaller of two known errors, a pinhole at a lens's focus against exp(−2), a 10× expander against the ABCD law | the wave-optics review, I4 ✅ | `oc-wave/tests/propagate.rs`, `oc-wave/tests/path.rs` |

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
