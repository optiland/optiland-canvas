# What a detector measures

A detector collects flux. What that flux *means* depends on what you divide it
by, and the divisor is the measure:

| measure | reading | binned by |
|---|---|---|
| power | flux in the cell | where light landed on the face |
| irradiance | flux per `mm²` | where light landed on the face |
| intensity | flux per steradian | the direction light arrived from |

The measure is a property of the detector, stored in the document, not a view
setting. An instrument measures one thing; which thing is part of the scene.

## The unit is relative

Flux is relative unless a source is calibrated, and nothing here calibrates
one. So irradiance reads *per mm² of the source's own flux*, not W/mm², and
intensity reads per steradian of the same. Turn the source's `flux` into
watts and the readings become watts per unit; until then they are ratios, and
the app says `flux/mm²` rather than `W/mm²` for exactly that reason.

This is the rule in [canonical-3d-kernel.md](architecture/canonical-3d-kernel.md#units):
nothing is called absolute irradiance without the calibration to justify it.

## Irradiance

Flux over the cell's area. The area is the **face's**, not the beam's
cross-section: a beam arriving at a slant spreads over more cells and each one
reads less, so the obliquity arrives on its own rather than being applied by
hand. If you want the irradiance normal to the beam, that is a different
quantity and it is not this one.

A line probe has no area. The v1 line detector migrates to `LineProbeLegacy`
and the scene refuses to give it a measure that needs a denominator, the same
way it refuses to give it a height nobody authored.

**A cell is a square of the bounding box.** A rectangle's grid is the
rectangle; a disc's and an annulus's is the box around them, so a cell at the
rim has part of itself outside the shape and reads the average over the whole
square. That is the cost of a rectangular grid on a round face and it is
visible at the rim, not hidden.

Until the measure needed a unit, the cell area for a round detector divided
the *disc's* area by the cell count, which put every per-cell density 4/π (27%)
high. The test that covered it pinned an identity that held either way. A disc
and a rectangle now have to read the same irradiance from the same patch of
light, which is the test that catches it.

## Radiant intensity

Intensity is an angular density, so no amount of dividing a face map produces
one. The detector fills a second histogram, binned by the direction light
arrived from:

- `[rings, sectors]`: polar rings out to the cone from the face normal, and
  azimuth sectors around it. `[24, 1]` is `I(θ)`, the polar curve on an LED
  datasheet.
- The polar angle is taken to the normal **line**, not the normal vector, so a
  detector reads the same whichever side light reaches it from. This is the
  convention `max_tilt_rad` already used.
- The resolution is the measure's own, not the face's. A detector can want a
  fine spot on its face and a coarse curve through it at the same time, and
  64 × 64 face cells reinterpreted as angle would be 4096 cells of noise.

A ring sector subtends

```text
Ω = Δφ (cos θ₀ − cos θ₁)
```

exactly. There is no quadrature here and no small-cell assumption, so a
two-ring histogram is as exact as a two-hundred-ring one, and the cells sum to
the cap's own `2π(1 − cos θ)`.

The face histogram keeps filling alongside it, so the mode costs no picture:
the hit map on the 3D detector face still draws, and the component inset shows
the polar map instead of the face because that is the picture the measure
makes.

### Light past the cone

A histogram covering part of the sky leaves the rest somewhere, and the
somewhere is a field: `outside_cone_flux`, reported next to the total. So a
histogram summing to less than `total_incident` is a fact you can read rather
than a discrepancy you have to find. The app prints the share as a percentage
when there is one.

## What the tests hold it to

The oracles are definitions, not datasheets, because intensity is defined
rather than measured (see [validation.md](validation.md) for where a datasheet
*is* an oracle):

- **A ring reports the intensity it was handed.** One ray per ring carrying
  exactly that ring's solid angle times a chosen intensity, read back to
  `1e-12`. No sampling, so this pins the divisor alone.
- **A Lambertian hemisphere falls as `cos θ`**, against the exact per-ring mean
  `(cos θ₀ + cos θ₁)/2` rather than the cosine at some midpoint, to 0.3%.
- **A cone uniform in solid angle reads flat and then stops** at its own half
  angle, to the same 0.3%.
- **The rings partition the cone** to `1e-12`, at 2, 7 and 64 rings.
- **The f32 preview bins directions like the f64 reference**, cell for cell.

### One thing that came out of writing them

The even-grid sampler aliases against the ring boundaries. Its strata are
fixed by `⌈√n⌉`, so a boundary falling inside a stratum takes the whole
stratum with it, and the innermost ring of a six-ring histogram sits 3.6% high
*however many rays are launched* — doubling the count doubles the strata and
moves nothing. The Sobol (0,2)-sequence has no such alignment to lose and
lands inside 0.05%. Worth knowing before trusting a grid-sampled angular
histogram anywhere else.

A ray arriving exactly on a ring or sector boundary is a knife edge the f64
and f32 tiers can round to opposite sides of. That is a property of binning,
not a defect in either, and the cross-tier test keeps its fan off the
boundaries so it measures agreement rather than float width.

## What is missing

- **Radiance** (`W/m²/sr`) needs both divisors at once, so it needs the two
  histograms crossed rather than kept side by side.
- **A bins control.** The face grid is fixed at 64 × 64 by
  `addAreaDetector` and nothing in the app changes it. The angular grid is
  reachable only through its default.
- **Intensity is the arriving beam's**, and equals a source's own intensity
  distribution only in the far field, where the source is small against the
  distance. Nothing warns when it is not.
- **A detector still absorbs or probes**; there is no mode that integrates over
  a sphere, which is what measuring total flux from a scattering object would
  want.
