# Validating against published data

A vendor's curve is a test only where the part is **determined by numbers
anybody can look up**. Where the design is proprietary, reproducing the curve
would be a test of whether we guessed the design, and a mismatch would say
nothing about the physics. That line runs through the middle of the catalogue,
so it is worth drawing it before reaching for a datasheet.

## Where a datasheet is an oracle

The part is fully specified by public numbers, so the comparison can fail for
one reason only: the model is wrong.

- **Uncoated windows.** A transmission curve is two Fresnel faces plus internal
  transmittance, and both halves are published: Sellmeier from the glass
  catalogue, `τᵢ` from the glass datasheet. Tests normal-incidence Fresnel, the
  two-surface series, Beer-Lambert and dispersion at once.
- **Waveplates.** The vendor states the crystal, the order and the design
  wavelength, which fixes the thickness; retardance against wavelength then
  follows from the crystal's dispersion equation. This is the sharpest test
  available of the crystal waveplate, and the suite runs it.
- **Prisms, Brewster windows, critical angles.** Determined by an index and an
  angle.
- **Coating synthesis against its own law.** An edge filter's blue shift with
  angle is the one relationship every filter vendor publishes, and it needs no
  proprietary information to check: see below.

## Where it is not

Semrock, Chroma, Alluxa and Edmund publish excellent numeric spectra and no
layer recipes. Our synthesized edge filter will not reproduce theirs, and it
should not be expected to. The same goes for broadband AR coatings and for
commercial polarizing cubes.

For those parts the right move is the opposite one: stop synthesizing and
**import the measurement**. `SpectralScalar::Tabulated` and
`AngularScalar::Tabulated` exist for exactly that, and a measured curve is a
better model of a specific part than any stack we could guess.

Two things to know before doing it:

- Nothing can author a measured spectrum from the app yet. The kernel holds
  one, `surfaceCoatingJSON` reports null for it, and only a hand-written scene
  reaches it. `setPolarizerExtinctionTable` is the one tabulated curve with a
  setter.
- **A tabulated curve has no angle.** A filter measured at 0° will report the
  same numbers at 45°, silently, which is the exact trap real coatings were
  built to escape. An importer needs to carry the measurement angle and say so
  away from it.

## What is not a datasheet test at all

Agreement with Optiland is a cross-engine check and it lives in
`crates/oc-py/tests`. It is worth having and it is not evidence about reality:
two engines can share a convention error, which is how the two found in the
polarization work were found by physical oracles rather than by each other.

## How the data is carried

No vendor data files are vendored. What the suite holds is a handful of
coefficients from published papers, reproduced inline in
`crates/oc-core/tests/reference_data.rs` with the citation, the retrieval date
and the valid range next to them. That keeps the suite offline, keeps CI off
the network, and keeps the oracle from rotting when a page moves.

The rule for adding one: it has to be small enough to read in the diff, and it
has to say where it came from.

## What the suite checks now

**The quartz constants.** `Waveplate::quartz` carries two indices; the
dispersion equation that produced them is Ghosh (1999), and the two agree. The
birefringence is checked ten times tighter than the indices, because the
difference alone sets the thickness a retardance needs.

**What a constant index costs a waveplate.** The kernel holds `n_o` and `n_e`
fixed, so a plate's retardance goes exactly as `λ₀/λ`. Real quartz is more
birefringent toward the blue, so its retardance rises faster. The whole
discrepancy is the ratio of the two birefringences, and it is now a number
rather than a caveat:

| wavelength | kernel | quartz | out by |
|---|---|---|---|
| 450 nm | 110.0° | 112.7° | −2.7° |
| 500 nm | 99.0° | 100.0° | −1.0° |
| 550 nm | 90.0° | 90.0° | 0 |
| 650 nm | 76.2° | 75.0° | +1.1° |
| 700 nm | 70.7° | 69.3° | +1.4° |

for a zero-order quarter wave cut at 550 nm. Under three degrees across the
visible, which is a percent or two of the state.

The same *fraction* on a multi-order plate is not a couple of degrees. A
tenth-order quarter wave at 450 nm comes out more than 100 degrees wrong,
because the error is a share of the whole retardance and that plate carries 41
times as much of it. **A constant index is fine for a zero-order plate and no
use to a multi-order one away from its design wavelength.** If multi-order
plates matter, the fix is to carry a dispersion equation for the crystal
instead of two numbers.

**The edge shift with angle.** Every filter vendor publishes

```text
λ(θ) = λ₀ · √(1 − (sin θ / n_eff)²)
```

with `n_eff` a single number standing in for the stack, quoted in the range 1.5
to 2.5. Nothing in the kernel knows that formula: the edge is measured where
the synthesized stack's reflectance crosses a half, at each angle, out of the
transfer matrix.

So the test is not whether we can evaluate the formula. It is whether **one**
effective index fits every angle. Fitted once at 45° on a silica-tantala
long-pass, it predicts the edge to within 0.06 nm from 10° to 50° and 0.6 nm at
60°. A stack that got the angle physics wrong would need a different `n_eff` at
every angle and the one-parameter fit would come apart.

One thing that fell out: the rule of thumb that `n_eff` is the geometric mean
of the two indices reads 1.730 for this pair, and the stack's own answer is
1.680. Three percent, which is 1.5 nm on the 45° edge. The geometric mean is a
rule for a cavity, not a law for an edge.

## Coatings against two oracles

The oracle matrix as one table, regenerated by
`crates/oc-py/tests/report_coatings.py`, is
[`validation-coatings.md`](validation-coatings.md); CI checks it and keeps it
as an artifact of each commit.


A datasheet validates; an oracle verifies. The thin-film kernel is held to two
independent libraries and to closed forms, on every pull request, by the
harness of the coating-fidelity spec's CF-00:

- **Byrnes `tmm` 0.2.0**, a transfer-matrix code that is not Macleod's
  admittance form, so agreement is evidence rather than one algorithm checked
  against its own transcription;
- **Optiland `thin_film`**, the admittance form the kernel mirrors on purpose.

`crates/oc-py/python/optiland_canvas/coatings.py` exposes the kernel
(`rt`, `amplitudes`, `indices`) and translates a stack for each oracle
(`to_tmm`, `to_optiland`). Every oracle is fed the complex indices the kernel
sampled, so a disagreement is the solver's and never a second reading of a
material table.

The cases live once, in `crates/oc-py/tests/coating_cases.py`.
`gen_coating_fixtures.py` freezes both oracles' answers into
`crates/oc-py/tests/data/coatings/*.json` with the oracle versions, the
generator's hash and the date; `test_coating_fixtures.py` holds the kernel to
them with no oracle installed. A scheduled CI job regenerates them with the
versions pinned in `tests/requirements-oracles.txt` and fails if they moved.
`test_thinfilm_tmm.py` runs `tmm` live, including 200 seeded random absorbing
stacks met from both sides through the critical angle.

| Comparison | Tolerance | Worst measured |
|---|---|---|
| Ten fixed cases, `tmm`, powers | 1e-9 | 8.7e-15 |
| Ten fixed cases, `tmm`, amplitudes (no map) | 1e-9 | 1.2e-14 |
| Ten fixed cases, Optiland, below the critical angle | 1e-11 | 1.2e-14 |
| 200 random absorbing stacks, `tmm` | 1e-9 | 3.4e-13 power, 4.7e-13 amplitude |

Closed-form and self-consistency cases in `thinfilm.rs`: C02 (a zero-thickness
layer is no layer, to 1e-14 in amplitude), C03 (a split layer is one layer),
C04 (the ideal quarter-wave AR reflects below 1e-28), C08 (a stack met from its
substrate is its mirror image; transmission reciprocal to 1e-12, reflection
not), C09 (no NaN on a grazing sweep, exact grazing and the critical angle
included). `every_listed_mistake_is_caught` switches five one-line mistakes
into the solver, test builds only, and requires the named check to fail on
each (C31).

What the harness found at the baseline:

- **Optiland is wrong in power past the critical angle when the stack
  absorbs** (attenuated total reflection): 0.69 against 0.48 for `R_s` at 45
  degrees inside glass, where the kernel and `tmm` agree to 3e-15. Frozen as a
  known divergence, version-gated.
- **The kernel's evanescent `t_p` had the wrong sign**: the tangential-to-whole
  rescaling used the principal cosine while the admittance used the decaying
  one. No power or traced ray read it. Fixed.
- **A plate's bare far face returned NaN in p at exactly the critical angle**
  (∞/∞). Fixed to its limit, total reflection.
- **A metal film thick enough that its phase passes Im δ ≈ 355 returned NaN**
  (5 µm of aluminium at 0.4 µm). CF-02's opacity rule closed it: the film is
  now the bulk metal to 1e-12 (C12).

### Thick absorbing stacks (CF-02)

`tmm` cannot judge a thick absorbing stack: its double-precision matrix
product overflows on every random 64-layer stack with k up to 7 and layers up
to 3 µm (1200 of 1200 NaN, measured on 0.2.0). The oracle for those is
`crates/oc-py/tests/mp_tmm.py`, Byrnes' formulation in 40-digit arithmetic,
with no opacity rule and no rescaling, itself held to `tmm` to 2e-15 on stacks
`tmm` can do.

| Comparison | Tolerance | Worst measured |
|---|---|---|
| 2000 random 64-layer absorbing stacks, n 1.3 to 4, k 0 to 7, d 1 to 3000 nm, 0/30/60/85 degrees, against `mp_tmm` | 1e-9 | 1.6e-13 (400 of them sampled for the number; all 2000 pass) |
| 5 µm aluminium against bulk Fresnel (C12) | 1e-12 | passes; against `tmm` 8.9e-16 |
| Silica on an opaque aluminium substrate against `tmm` (C13) | 1e-9 | 1.2e-15 power, 1.9e-15 amplitude |
| Bare glass at 89.99 degrees against Fresnel in 50 digits (C24) | 1e-11 | 1.3e-12, from re-deriving cos θ₀ near grazing; 89.9 degrees is inside 1e-12 |

### Film materials (CF-03)

The built-in film catalogue (`crates/oc-core/src/film_catalog.rs`) is
generated by `tools/ri_import.py` from the refractiveindex.info database as
Optiland packages it, and `crates/oc-py/tests/test_film_catalog.py` holds it to
the same files:

| Check | Tolerance | Result |
|---|---|---|
| Every tabulated n and k at every source knot between 0.25 and 2.6 µm, read back through the kernel | 1e-12 | exact (the numbers are the source's digits) |
| Every formula entry (HfO₂ Cauchy, ZrO₂-Y₂O₃ Sellmeier, ZnS Sellmeier) against Optiland's `MaterialFile` of the same file, 201 points over its range | 1e-12 | passes |
| C11: MgF₂ over TiO₂ over SiO₂ from the catalogue, 20 wavelengths 0.4 to 2.0 µm, against `tmm` and Optiland fed the same indices | 1e-9 | passes, after the square-root fix below |
| C30: every committed v2 document, and a scene with every coating kind, migrated to v3 | bit-identical | passes |

The C11 stack found a precision loss in the kernel's complex square root: for
a weakly absorbing film (k = 6e-4, MgF₂ in the UV tail) the half-angle formula
subtracted two nearly equal numbers and lost 6e-9 in T, where `tmm`, Optiland
and the 40-digit oracle agreed with each other. The root now takes the small
part from the large one near the real axis, and only there, so every
well-conditioned result kept its bits (C33 still holds).

### Absorption by layer (CF-04)

| Check | Tolerance | Result |
|---|---|---|
| Every layer's absorbed fraction against `tmm.absorp_in_each_layer`, every absorbing fixture case, both sides (C15) | 1e-9 | passes |
| R + T + Σ A_k + A_substrate = 1 on 2000 random absorbing stacks, both sides; A_k ≥ −1e-15 | 1e-12 | passes |
| Deposits per object against the object ledger, polarized and not (C16) | 1e-10 relative | passes |
| A lossless dielectric AR | exactly 0 W deposited | passes |

`check_c15` also catches the mutation "a layer's absorption read across the
wrong pair of interfaces" (C31).

### Slabs and plates (CF-05)

| Check | Tolerance | Result |
|---|---|---|
| A coated 2 mm absorbing slab with a rear two-layer AR, both sides, R, T and every element's absorption against `tmm.inc_tmm` (C17) | 1e-9 | 1.5e-15 |
| A `Plate` under `plate_model: Incoherent` against the same plate in `tmm.inc_tmm` | 1e-9 | passes |
| The plate Mueller matrix against its series summed to 200 passes; S0 against the incoherent powers; DOP of +45 degree light at most 1 + 1e-12 (C18) | 1e-12 | passes |
| `PhaseFree` plates against the pre-profile kernel (C33) | bit-identical | passes |

### The preview's coatings (CF-07)

| Check | Tolerance | Result |
|---|---|---|
| A silvered glass tube's bore, from the bore and from the glass, against the f64 kernel (C22) | 2e-3 | passes; the outside grid reused inside errs above 1e-2, as the mutation must |
| A Fresnel lens with an AR as its default response, 10 000 random wavelength, angle and side samples (C23) | 2e-3 | passes |
| An edge filter from 85 to 88.2 degrees (C24) | 5e-3 | passes; flat beyond 88.2 |
| `coating_model: Nominal` | the design point in every row | passes |

The grid used to be 64 rows uniform in `sin²θ`, and near grazing it was out by
up to 8e-2 on bare glass and a three-layer AR; the earlier figure of 7.6e-4
(C10) was measured on scenes that did not reach those angles.

### Substrate-adapted designs (CF-08)

| Check | Tolerance | Result |
|---|---|---|
| `Bbar` against the fitted single layer and bare glass, n_s 1.46 to 2.00, 400 to 700 nm, 0 to 30 degrees (C25) | Bbar ≤ single + 1e-3 ≤ bare | passes; Bbar 0.0056 / 0.0059 / 0.0090 / 0.0076 / 0.0073 / 0.0056 band R; on 1.72 it falls back to one layer, on 1.92 and 2.00 to two |
| The old three-layer preset on N-SF66 at its 650 nm design wavelength (C25) | ≥ 8% band R | 8.08% |
| A V-coat of SiO₂ over TiO₂ on N-BK7 (C26) | R(550 nm) ≤ 1e-10 | passes; two low-index films are refused by name |
| A solar `BandAr` on low-iron glass, 0.4 to 2.5 µm, 0 to 40 degrees, weighted by the G173 direct sun (C27) | ≤ single + 1e-3; ≤ 0.5 × bare | 0.0211 against 0.0213 single and 0.0445 bare: 0.475 of bare |
| Seven synthesized layer lists pinned to the bit | identical on Windows and Linux | thicknesses quantized to 1e-6 nm, so an ulp of a platform's `powf` cannot show |

### Measured tables (CF-09)

| Check | Tolerance | Result |
|---|---|---|
| A table bilinear in `(λ, cos θ)`, read at and between its knots (C28) | 1e-12 | passes |
| A one-angle vendor table at 45 degrees, a ray at 52, under `Strict` (C28) | the stated text | "Data supplied at 45 degrees only; this ray is at 52 degrees" |
| A table written and read back as CSV (C28) | identical | passes |
| A table from the transfer matrix against the matrix at its knots (C28) | 1e-9 | passes, exactly |
| A tilted table plate: each arm against the table, absorption the remainder | 1e-12 | passes |
| Two s/p tables in series, polarized and scalar (earlier spec §9.4) | 1e-12 | `0.5 (Ts1 Ts2 + Tp1 Tp2)` and the product of the means |
| The preview's grid against the table, 0 to 80 degrees | 1e-2 | 7.5e-3, the grid cutting the corner at the table's 5-degree knots |

### Coatings at a temperature (CF-10)

| Check | Tolerance | Result |
|---|---|---|
| A quarter-wave stack 50 K above its reference: the centre, where `Im r` crosses zero, against `λ₀ (1 + ΔT dn/n)(1 + α ΔT)` (C29) | 1e-9 relative | passes; the first-order `(dn/n + α) ΔT λ₀` differs by the product term, 2.5e-7 relative here |
| `temperature: Off` with a declared temperature against no temperature, solver and trace (C29) | bit-identical | passes |
| An object held at its reference temperature under `Declared` | bit-identical to the cold trace | passes |

## Worth adding

- An uncoated N-BK7 and fused-silica window against published internal
  transmittance, which would test bulk absorption. `AbsorptionModel::Tabulated`
  already takes `α` per mm and nothing authors one.
- A measured-spectrum importer, with the measurement angle attached.
- Film materials with dispersion. Coating layers are constant-index today, so
  the same question asked of the waveplate has not been asked of an AR coating.
