# What a detector measures

A detector collects flux. What that flux *means* depends on what you divide it
by, and the divisor is the measure:

| measure | reading | binned by |
|---|---|---|
| power | flux in the pixel | where light landed on the face |
| irradiance | flux per `mm²` | where light landed on the face |
| intensity | flux per steradian | the direction light arrived from |

The measure is a property of the detector, stored in the document, not a view
setting. An instrument measures one thing; which thing is part of the scene.

## The face: its size and its pixels

A detector's face is a rectangle, a disc, or a ring, at its real size in
millimetres. The component view sets all three: the shape, the width and
height (or the diameter, and a ring's hole), and the pixel counts across and
down.

- **The pixel grid spans the face's bounding box.** For a rectangle that is
  the face; for a disc or a ring it is the square around it.
- **The pitch follows.** It is the extent over the count, per axis, so a
  12 × 4 mm face cut 24 × 8 has 500 µm square pixels, and pixels are square
  only when the two pitches agree. Resizing the face keeps the counts and
  moves the pitch, as swapping a lens for a bigger sensor of the same
  resolution would.
- **A face holds at most 1024 × 1024 pixels** (`MAX_DETECTOR_CELLS`, also the
  cap on an intensity histogram). A readout keeps two numbers per pixel and
  the tracer one readout per worker, and a million pixels need tens of
  millions of rays before any one pixel's reading rests on more than a few.
  A file asking for more is refused on load, and the count is checked before
  it is multiplied, so an absurd one cannot overflow.

**A pixel's reading rests on the rays that landed in it.** The component view
says how many that is on average over the lit pixels, and what it means:
sampling alone can move a pixel by about `±2/√k` with `k` rays. For a random
source one standard deviation is `1/√k`; a grid source, which is what the app's
sources are, can gain or lose a row and a column of rays in a pixel, which is
`2/√k`, and does it in a pattern rather than as noise. Below ten rays the view
says single pixels are sampling noise. More pixels resolve more only while the rays keep
up; the total flux on the face does not depend on the pixel count at all.

**The total is the number every view quotes.** The detector dock and the
component view both print the flux on the whole face from one
`detectorResultJSON` reply, in the same digits, and the view re-reads whenever
the dock does, including after a trace setting such as the ghost layer that
does not move the scene's revision. Before that, the view led with the
brightest pixel, which is a different number, and kept an old trace after a
ghost toggle, which is a stale one.

## The unit is relative

Flux is relative unless a source is calibrated, and nothing here calibrates
one. So irradiance reads *per mm² of the source's own flux*, not W/mm², and
intensity reads per steradian of the same. Turn the source's `flux` into
watts and the readings become watts per unit; until then they are ratios, and
the app says `flux/mm²` rather than `W/mm²` for exactly that reason.

This is the rule in [canonical-3d-kernel.md](architecture/canonical-3d-kernel.md#units):
nothing is called absolute irradiance without the calibration to justify it.

## Irradiance

A pixel's flux over the area it was collected on. The area is the **face's**, not the beam's
cross-section: a beam arriving at a slant spreads over more cells and each one
reads less, so the obliquity arrives on its own rather than being applied by
hand. If you want the irradiance normal to the beam, that is a different
quantity and it is not this one.

A line probe has no area. The v1 line detector migrates to `LineProbeLegacy`
and the scene refuses to give it a measure that needs a denominator, the same
way it refuses to give it a height nobody authored.

**A rim pixel reads over its part of the face.** On a disc or a ring the
pixels at the rim are only partly on the face, and light reaches a pixel only
through that part. So irradiance divides each pixel's flux by the area of the
pixel that lies on the face, computed exactly (the disc and square overlap has
a closed form; nothing is sampled). Under uniform light a rim pixel then reads
what a whole one does, and the readings times these areas sum back to the
flux collected, whatever the shape: the A12 identity in the next-steps spec.
Pixels wholly off the face, in the corners of a disc's box or in a ring's
hole, collect nothing and read 0.

This replaced two earlier answers, both wrong. The first divided the *disc's*
area by the pixel count, which put every reading 4/π (27%) high; the test that
covered it pinned an identity that held either way. The second divided by the
whole square, which read a rim pixel low by its uncovered share, a pixel 30% on
the face at 0.3 of the light on it.

**The peak skips slivers.** A rim pixel with a few hundredths of its square on
the face is right on average, but it rests on few rays, and one ray over a
sliver of area is a spike: a pixel 1% on the face reads a single ray a hundred
times higher than a whole pixel would. The reported peak irradiance therefore
leaves out pixels with less than half the covered area of the fullest pixel.
The readings themselves are untouched, and the pictures scale to that peak so
one sliver cannot darken the rest.

**The face mean** is the flux on the face over the face's area. It does not
depend on the pixel count, and it equals the irradiance only where the light
covers the face evenly: a small spot on a large face reads low in the mean and
high in its pixels.

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

For the face:

- **The overlap is exact**, against closed forms (a quarter disc, a half, a
  circular segment, the square the rim touches on four sides) to `1e-13`, and
  against an independent numerical integral over sixty rectangles the rim cuts
  every way it can.
- **The covered areas sum to the face's area** to `1e-12`, for rectangles,
  discs and rings at pixel counts from 1 × 1 to 64 × 64, square and not.
- **Rim pixels read the light on them.** A uniform beam over a disc and a
  ring: whole pixels read exactly 1, rim pixels at least a quarter on the face
  are within 0.3% (held to 2%), and the readings times the areas give back the
  total to `1e-12`.
- **The peak is not a sliver**, and the sliver's own reading is still there.
- **The setters refuse** a zero count, a grid past the cap, a fraction, a
  non-positive or non-finite size, and a ring whose hole is as wide as itself,
  and leave the scene as it was.
- **The dock and the component view quote the same flux** on both arms of a
  beam splitter, before and after the ghost layer is switched on (a browser
  test).

For intensity, the oracles are definitions, not datasheets, because intensity
is defined rather than measured (see [validation.md](validation.md) for where
a datasheet *is* an oracle):

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
*however many rays are launched*: doubling the count doubles the strata and
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
- **A pixel is a square of the grid, not a photosite.** There is no fill
  factor, crosstalk, read noise, or saturation, and a round face's pixels are
  the box's squares rather than anything a real round sensor has.
- **The response curve** (quantum efficiency against wavelength) is in the
  scene model and the readout's `totalSignal`, but nothing in the app edits it.
- **Intensity is the arriving beam's**, and equals a source's own intensity
  distribution only in the far field, where the source is small against the
  distance. Nothing warns when it is not.
- **A detector still absorbs or probes**; there is no mode that integrates over
  a sphere, which is what measuring total flux from a scattering object would
  want.
