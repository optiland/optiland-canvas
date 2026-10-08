# What a detector measures

A detector collects flux. What that flux *means* depends on what you divide it
by, and the divisor is the measure:

| measure | reading | binned by |
|---|---|---|
| power | flux in the pixel | where light landed on the face |
| irradiance | flux per `mm²` | where light landed on the face |
| intensity | flux per steradian | the direction light arrived from |
| illuminance | lux, and the pixel's colour | where light landed on the face |

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

## Illuminance: what the eye sees

Irradiance counts every watt the same. The eye does not: at 555 nm a watt is
683 lumens, at 700 nm it is 2.8. Illuminance weighs each ray by the CIE 1931
photopic curve `ȳ(λ)` and divides by the pixel's area in square metres:

```text
E = 683 lm/W · Σ Φ ȳ(λ) / A
```

which is what a light meter on the face would read. Beside it each pixel keeps
its colour, `Σ Φ · (x̄, ȳ, z̄)(λ)` on the same scale, so `Y` is the
illuminance and `(X, Y, Z)` is the colour a person would name
(`oc_core::colorimetry`, the CIE 1931 2° observer tabulated at 5 nm from
CVRL's 1 nm table). The pictures draw the face in that colour instead of the
heat ramp:

- **Exposure**: the eye adapts, so the brightest part of the face, the 99th
  percentile of each pixel's largest display channel, is drawn at full scale,
  and **exposure** (EV) moves it in stops. Exposing for luminance instead
  would clip a saturated red, which needs 1.7 times full red at a third of
  white's luminance.
- **Gamut**: spectral colours lie outside sRGB (the sodium yellow needs
  negative blue). Such a colour moves toward the grey of its own luminance
  until it fits: brightness and hue kept, saturation given up. A spectrum on
  screen therefore shows creases where its hue passes a primary, the same as
  any photograph of one.

Lux need watts. Under the sun the flux is watts and the unit reads `lx`;
with any other source it reads `lx (relative)`, like every other reading
here. What a visitor sees on a surface is its luminance, the illuminance
times its reflectance over π for a matte white; the picture's exposure takes
care of the scale.

## Radiant intensity

Intensity is an angular density, so no amount of dividing a face map produces
one. The detector fills a second histogram, binned by the direction light
arrived from, in one of two layouts:

- **Grid**, the default for a detector switched to intensity: `[across, down]`
  cells over `±cone` in each of the two angles `atan(x/|z|)` and
  `atan(y/|z|)`, laid out like the face's pixels. It is the far field on
  straight axes, the picture a camera in a lens's focal plane records, and a
  cross-section can be cut through it along either axis. It starts at 31 × 31,
  odd so that one cell sits on the normal.
- **Polar**: `[rings, sectors]`, polar rings out to the cone from the face
  normal and azimuth sectors around it. `[24, 1]` is `I(θ)`, the polar curve
  on an LED datasheet. A file that names no layout means this one, since it
  was the only one when those files were written.
- Both take their angles to the normal **line**, not the normal vector, so a
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

A grid cell is exact too. A direction `(p, q, 1)` in slope space covers
`dp dq / (1 + p² + q²)^{3/2}` of the sphere, and over the rectangle from the
origin to `(p, q)` that integrates to

```text
F(p, q) = atan(pq / √(1 + p² + q²))
```

the solid angle a rectangle subtends from above its corner. A cell between
slopes `p₀..p₁` and `q₀..q₁` is four values of `F`, and the grid sums to
`4F(tan c, tan c)` for a cone `c`, and to `2π` at 90°.

The face histogram keeps filling alongside it, so the mode costs no picture:
the hit map on the 3D detector face still draws, and the component inset shows
the far field instead of the face because that is the picture the measure
makes.

### Light past the cone

A histogram covering part of the sky leaves the rest somewhere, and the
somewhere is a field: `outside_cone_flux`, reported next to the total. So a
histogram summing to less than `total_incident` is a fact you can read rather
than a discrepancy you have to find. The app prints the share as a percentage
when there is one.

## Cross-sections

The component view can open a plot beside the picture (**beside:
cross-sections**) and widens its window to make room; closing the plot gives
the width back, unless you have resized the window since. The picture and the
plot sit side by side in a wide window and one above the other in a tall one.

Each cross-section is a dashed line across the picture: a row (across, at one
height) or a column (up, at one place across). Two start through the
brightest cell; **+ row** and **+ col** add more, up to six, and **− cut**
removes the last. Drag a line anywhere along it, or by its grip outside the
picture. The plot follows as it moves, from the numbers already in hand, so a
drag traces nothing. Each line reads whole cells: the dashed line is drawn at
the centre of the row or column it samples, since that is where its numbers
come from.

The readout lists each cut in its colour with its peak and its full width at
half that peak, interpolated between cell centres. On the face the positions
are mm; in a grid far field they are degrees. A polar histogram has no rows,
so the plot asks for the grid.

## Colour

**beside: colour** breaks the light on a detector out by wavelength. The
tracer keeps, for each spectral line, flux-weighted sums of where each ray
landed `(u, v)` and of its slopes `a = x/|z|`, `b = y/|z|` in the detector's
frame, with their squares and the cross terms `u·a` and `v·b`. Sums rather than
per-hit records, so every hit counts however many there are, and the worker
pool adds them up like every other number. From them, per line:

- **direction**: the mean arrival angle in each plane, and the RMS angular
  spread about it. What a collimator leaves as colour in its beam.
- **position**: the centroid. Lateral colour.
- **spot**: the RMS radius about the centroid.
- **focus**: how far along the normal the spot is smallest, positive the way
  light travels. Each ray's position there is `u + s·a`, so the spot's
  variance is a parabola in `s` with its least point at
  `s = −(cov(u,a) + cov(v,b)) / (var a + var b)`. Axial colour.

The plot shows the chosen one against wavelength, each point in its line's
colour, and the readout sets the numbers against each other: how far the
lines' directions differ, against how much each line spreads by itself.

Every line has a plane of least spot size, but it is a focus only for light
arriving as a cone. After a tube's walls have mixed it, or out of a
collimator, the plane is arithmetic. So focus is reported, and opens by
default, only when every line spreads over more than 20 mrad RMS (about f/25);
otherwise the panel opens on direction and says why.

The sums are kept per wavelength bin, 5 nm wide below 1 µm, 20 nm to 2.5 µm
and 100 nm beyond, each also holding its flux-weighted wavelength, so a source
of a few lines reads each line back exactly and a continuous one is not a sum
per ray. The panel merges a continuum's two hundred or so bins into at most 24
groups of equal flux, exactly, by adding their sums.

Each point carries an error bar: one standard error of the line's mean, its
spread over the square root of its rays. The readout quotes the typical one
next to the difference between lines, and says so when that difference is
within a few of them. Colour in a beam that is spread by other things (a
tube's walls, a sun's disc) is small against that spread, and needs many rays
to see: raise the analysis rays, or read it where the beam is still tight.

## Analysis rays

The ray-density control sets how many rays the preview traces and the picture
draws, which has to stay light enough to follow a drag. **analysis rays** in
the detector's component view traces the numbers with as many launch rays as
you type, up to a billion, in the background on the worker pool, while the
picture keeps the ray density's count. The field takes `250000`, `2.5e5`,
`250k` or `2.5M`; **100k**, **1M** and **10M** are a click away, and
**scene** (or an empty field) goes back to the ray density's own. A count
below the ray density's traces the ray density's. Each block of the trace traces its share of both: the
numbers' rays draw nothing, and the picture's rays are traced for their
segments alone, their hits and flux thrown away. The numbers are then the
denser trace's, to the bit, and the picture is the one the ray density alone
draws.

Each source gets its share of the count in proportion to the rays it emits
now: a source with an extent gets more origins, a point source more
directions, rounded to whole samples, so the count traced is within a few
percent of the one asked for, and the readout quotes the one traced. The flux
per ray falls to match, so every total stays what it was and only the noise
drops.

Past half a million rays the trace runs in passes of at most that many. Each
pass is the same point sets shifted by a random offset on the unit square (a
Cranley–Patterson rotation), their jitter and pairing drawn afresh, so no pass
repeats another and each is as even as the first. Every ray carries its
share of the passes' flux. The trace's 64 blocks run through the passes in
order, so a worker holds one pass in memory at a time, about 75 MB, and the
limit on the count is the time you give it, not the memory. A pool gives what
one worker gives, to the bit, however many passes. The setting applies to every detector and the dock,
is a trace setting like the ghost layer, and is not saved in the scene. The
readout says how many rays the numbers rest on.

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
- **Illuminance weighs each wavelength by the eye**: two rays in one pixel,
  2 W at 555 nm and 1 W at 700 nm, read `683 · (2 + 0.004102) / 4e-6` lux to
  `1e-12`, their colour is the two rays times the table's `x̄, ȳ, z̄`, and a
  pool of workers adds the colours up bit for bit as a single trace does
  (`detector_measure3`, `trace_job3`). The table itself: `ȳ(555) = 1`, equal
  energy gives equal `X, Y, Z` to 0.1%, and D65 white is sRGB white
  (`oc_core::colorimetry`).
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
- **The grid partitions its square of directions** to `1e-12`, at 3 × 3,
  7 × 5, 31 × 31 over 89° and 64 × 64 over 90°, symmetric about both axes.
- **A grid cell reports the intensity it was handed**, one ray down the
  middle of each cell, to `1e-12`.
- **The f32 preview bins directions like the f64 reference**, cell for cell.

For colour and the analysis rays:

- **The lines add up to the detector**: their flux and hits are the
  detector's, in order of wavelength.
- **Each line focuses where its sums say.** The default singlet's detector is
  moved to each line's predicted best focus and traced again; the spot it
  records there is the one the sums predicted to `1e-9` mm, and is smaller
  than 0.3 mm either side. Blue focuses before red.
- **A collimated beam has no focus**, zero spread, and the beam's angle.
- **More rays for the numbers leave the picture as it was**: the numbers are
  the denser scene's to the bit and the picture is the plain scene's, segment
  for segment and in order, traced whole or shared out to a pool.
- **Passes add rays and nothing else**: every later pass has pass 0's count
  and flux, shares no direction with it, and samples a cone as evenly (its
  mean cosine within `1e-5` of the exact one). Three passes of the Cooke
  triplet land the one-pass totals within 2% with at least twice the hits,
  and three workers handed the blocks out of order give the single job's
  numbers to the bit.
- **In the browser**, cross-sections widen the window and follow a dragged
  line, the colour panel's axial colour matches the readout, and a typed count
  of sixteen times the scene's rays lands more than eight times the hits with
  the picture's segment count unchanged.

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
