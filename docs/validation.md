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

## Worth adding

- An uncoated N-BK7 and fused-silica window against published internal
  transmittance, which would test bulk absorption. `AbsorptionModel::Tabulated`
  already takes `α` per mm and nothing authors one.
- A measured-spectrum importer, with the measurement angle attached.
- Film materials with dispersion. Coating layers are constant-index today, so
  the same question asked of the waveplate has not been asked of an AR coating.
