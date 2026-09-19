# Polarization

The kernel computed s and p separately from the day the transfer matrix landed,
and threw the difference away at every surface. A dichroic at 45 degrees can
reflect 78% of s and 2% of p and hand on a beam the tracer called unpolarized.

The error never showed at that element. For unpolarized light in, the mean is
the right flux, so the first dichroic in a path looked fine. It showed at the
**second** polarization-sensitive element, which in reality no longer sees
unpolarized light. Two tilted dichroics in series, the ordinary case in a
fluorescence path, was outside the model.

The conventions are in [`conventions.md`](physics/conventions.md). This page is
what the model does, what it refuses to do, and where it is still an
idealisation.

## What a ray carries

A power-weighted Stokes vector and the transverse frame it is written on.

`S0` **is** the ray's flux. There is no second copy of it, because two numbers
that mean the same thing eventually disagree and there is no way to tell which
one is wrong. Unpolarized light is `(Φ, 0, 0, 0)`; a single Jones vector cannot
represent that, which is why the state is Stokes.

Between interactions the frame is constant in world space: a straight ray in an
isotropic medium has nothing to turn it.

## What happens at a surface

1. Resolve the plane of incidence and write the arriving state on it.
2. Apply the element's Mueller matrix.
3. Hand both outgoing rays that same `ŝ` axis.

Step 3 costs nothing: the reflected and the transmitted ray both lie in the
plane of incidence, so `ŝ` is perpendicular to both of them. Only the ideal thin
lens needs anything else, because it bends a ray without a plane of incidence to
work in, and its frame is carried across by the shortest rotation.

Bulk absorption is polarization-blind, so a path through glass scales the state
and does not change it.

## What each response does

| Response | Model |
|---|---|
| Bare dielectric interface | Complex Fresnel amplitudes, so a partial reflection polarizes and a total internal reflection retards. |
| Thin-film coating | The stack's own complex amplitudes, from the same matrix that produces its powers. |
| Plate-backed coating | Powers only. The two faces are added incoherently, which is right for millimetres of glass and is not a Jones element, so there is no relative phase to report. |
| Mirror | One number. Treated as an ideal reflector: equal for s and p, no retardance. |
| `ExplicitRt` | Two numbers. Treated as an ideal grey attenuator, polarization-blind and phase-neutral. |
| Ideal thin lens | No polarization action. |
| Polarizer | Passes the field on its transmission axis, absorbs the perpendicular component, leaks its stated extinction, which is one number or a measured curve against angle. Optionally laminated in glass, whose two faces then do real Fresnel. |
| Polarizing cube | A quarter-wave stack cemented in the glass the Brewster condition demands, at 45 degrees. Transmits p, reflects s, and gets its contrast and its acceptance angle from the transfer matrix. |
| Retarder | Adds a phase to its slow axis. Either a number, or a real slab of uniaxial crystal, which moves with wavelength and angle. Lossless, unless it is laminated, when its faces reflect. |

**Power data cannot supply a phase.** `Rs`, `Rp`, `Ts` and `Tp` fix the
magnitudes of a diagonal Jones element and say nothing about the phase between
them. Where the kernel has only powers it declares an ideal phase-neutral model
and records the assumption, and it records it **only when the light arriving is
already polarized**, because that is when the assumption can be wrong. An ideal
mirror in unpolarized light assumes nothing, and a report that fired on every
scene with a mirror in it would not be worth reading.

## Elements with an axis

A polarizer's transmission axis and a waveplate's fast axis are directions in
the element. They do not turn with whatever ray happens to arrive, so unlike a
coating these do not work in the plane of incidence. The axis is the surface's
local `+X` turned by its angle in the surface's own plane, and what acts on a
ray is that direction projected into the plane the field lives in.

Projecting the axis out of the element is not the same as projecting local `+X`
and turning the result inside the ray's transverse plane. The two agree at
normal incidence and diverge as the square of the obliquity, and the difference
is the whole of an ideal element's angle dependence. Turning inside the
transverse plane keeps the angle between any two elements exactly what it was,
so a crossed pair would extinguish perfectly seen from every direction. Real
crossed polarizers leak off axis, which is why a display has a viewing angle:

| ray off axis | fraction of unpolarized flux passed by a crossed pair |
|---|---|
| 10 degrees | 1.2e-4 |
| 20 degrees | 1.9e-3 |
| 30 degrees | 1.0e-2 |

Those are the worst case over the azimuth, which lies at 45 degrees to both
axes. A ray leaning along either axis leaks nothing, because that axis is left
alone and the other is shortened without being turned.

That makes them standalone-surface responses. A body patch carrying one is
refused rather than given an axis nobody chose, because a curved patch of a
solid does not define one.

Both are zero thickness. They are the polarization action and not the crystal:
no walk-off, no ordinary and extraordinary rays, no optical path, and a
retardance that is the number you gave rather than one that follows from a
thickness and a birefringence at each wavelength.

## The glass around them

An ideal layer is not a part you can buy. A sheet polarizer is laminated in
glass, and those two faces reflect: less light on axis, and a different amount
of s than of p as soon as the element is tilted. `substrate_index` is that
glass. One is air, where a face reflects nothing and transmits everything, so
the faces drop out of the arithmetic rather than being skipped by a branch.

What it buys, for n = 1.5:

| ray off axis | passing a polarizer on s | on p |
|---|---|---|
| on axis | 0.4608 | 0.4608 |
| 20 degrees | 0.4540 | 0.4671 |
| 45 degrees | 0.4122 | 0.4916 |
| 60 degrees | 0.3390 | 0.4982 |

as fractions of unpolarized light arriving, against a bare layer's flat 0.5 at
every angle. By 45 degrees a polarizer that passes p gets a fifth more light
than one that passes s, which is the single largest thing angle does to a real
polarizer below its acceptance limit.

The oracle is Fresnel twice over, and it is exact rather than fitted: the two
faces are reversible, so a face meeting air from outside reflects exactly what
it reflects meeting glass at the refracted angle from inside. The tests compare
against `(1 - R)²` written out from the Fresnel formulas rather than against
anything the kernel computed.

Inside the glass the ray is straighter, so the axis projection shrinks. A
laminated polarizer holds its contrast off axis better than a bare film does,
and the model gets that for the same reason the real one does.

The element is still zero thickness, so the two faces sit in the same place.
There is no displacement, no optical path through the glass, and no bounce
between the faces. That bounce is bounded by the square of a face's reflectance
and crosses the polarizing layer twice more on the way, so a polarizer absorbs
most of it and a lossless retarder does not. The capability report says so
whenever an element has a substrate.

What the faces reflect is a ghost like any other: followed when ghosts are on,
named in the ledger when they are off, and absorbed in neither case.

## A polarizing beam splitter

Everything else here that polarizes does it by absorbing, and does it because
somebody said so. The cube does it by reflecting, and it is the only polarizing
part in the canvas whose behaviour is **derived**.

Its stack is a plain quarter-wave mirror. What makes it a polarizer is where it
is used: at the Brewster angle between its own layers, where the p reflectance
of every interface inside it is exactly zero. So p passes and s is reflected,
and neither number was typed in anywhere.

That condition fixes the glass. Brewster puts the angle in the high-index layer
at `tan θ_H = n_L/n_H`, and Snell carries that back out to whatever the stack is
cemented in. At 45 degrees, which is what a cube is:

```text
n₀ = √2 · n_H · n_L / √(n_H² + n_L²)
```

For titania on silica that is **1.754**, which is why a polarizing cube is made
of dense flint and not of crown. The builder takes the coating pair and makes
the cube out of the glass they demand; there is no glass argument, because
there is no choice. Cement the same twelve periods in ordinary crown and it
stops being a beam splitter and becomes a mirror, which a test checks.

Twelve periods, on axis, one unpolarized ray in:

| | flux | degree of polarization |
|---|---|---|
| transmitted | 0.4946 | 1.000000 |
| reflected | 0.4949 | 0.9995 |

The transmitted beam is the cleaner one, which is what a real cube's datasheet
says too: what keeps s out of it is every layer in the stack, while the
reflection carries a little p because the stack's outer interfaces are not at
Brewster even when its inner ones are.

**It has an acceptance angle nobody told it.** Launch light that is already s,
which the cube should reflect entirely, and measure what gets past it:

| off axis | s leaking through |
|---|---|
| 0° | below 1e-9 |
| 1° | 9.7e-5 |
| 2° | 3.9e-4 |
| 5° | 2.4e-3 |
| 10° | 9.4e-3 |

Quadratic in the angle, and it comes out of the transfer matrix rather than out
of a table. This is the difference between this part and the ideal polarizer
above, which needs a measured curve to have an acceptance angle at all.

The cube's outer faces carry `ar_residual`, the scalar antireflection
stand-in. That is not an approximation where they are used: at normal incidence
s and p are the same thing, and one number is the whole truth.

The cube and its diagonal are two objects. A solid with an interface inside it
is not something the scene model can hold, and two touching solids would put
two faces in the same place, which the tracer resolves by distance and cannot
tell apart. A surface strictly inside one solid has neither problem: media
resolve by containment, so the stack sees glass on both sides at 45 degrees,
which is exactly what a cube is.

## Where a polarizer stops working

Everything above gives the polarizing layer itself one contrast, held at every
angle. No real one does. A wire grid loses contrast off axis; a thin-film
polarizer or a MacNeille cube is built around a Brewster match at one angle and
gives up a couple of degrees away from it; a Glan prism has a hard acceptance
cone past which its ordinary ray stops being totally reflected.

None of that follows from anything the kernel can compute. A coating's angle
dependence falls out of its own stack, and an ideal layer has no stack. So the
extinction is either one number or **a curve against angle of incidence**,
which is how a datasheet gives it and the only honest way to say where a
polarizer stops being one:

```js
scene.setPolarizerExtinctionTable(id, [0, 10, 25, 40], [1e-4, 3e-4, 5e-3, 8e-2]);
```

Outside the angles measured the curve holds its endpoint. That is a known
understatement, not a model: a real polarizer is worse out there, not the same,
so the range somebody measured is the range to trust. The component view plots
the curve on a log scale with the samples marked, because it is data and a
reader should see where the measurements are and where the line is a guess
between them.

Typing a number into the extinction field puts it back to one value at every
angle, which is the ideal layer.

## A waveplate made of something

An ideal retarder is a number: the same retardance at every wavelength and from
every direction. No plate is. Give a retarder a crystal and the number becomes
the **nominal**, what the plate gives on axis at the wavelength it was cut for,
which is how a waveplate is specified and sold. Everywhere else the retardance
follows from the plate.

The thickness follows from the nominal and the birefringence. A quarter wave at
550 nm is 15.09 µm of quartz, 11.65 µm of magnesium fluoride, or 0.799 µm of
calcite, which is why nobody makes a zero-order calcite waveplate.

**Wavelength.** With the indices held constant the retardance is exactly `λ0/λ`
times the nominal, so a quarter wave cut for 550 nm retards by 110 degrees at
450 and the light that came out circular comes out visibly elliptical. That is
the larger half of a real plate's dispersion; the rest is the dispersion of the
birefringence itself, which is not modelled.

**Angle.** The two waves see different indices along the plate normal, and
tilting the plate moves those apart. What is conserved across every face is the
transverse index `n sin θ`, so the plate's own glass never enters, and the shift
depends on **which way** the plate is tilted:

| 15 degrees off axis, leaning | zero-order quarter wave | tenth-order quarter wave |
|---|---|---|
| across the optic axis | 91.28° | 142.63° |
| halfway between | 90.01° | 90.22° |
| along the optic axis | 88.73° | 37.80° |

Two things fall out of that table. A multi-order plate is not a quarter wave
any more 15 degrees off axis, because the error is a fraction of the whole
retardance and a tenth-order plate carries 41 times as much of it. And the
sensitivity nearly vanishes at 45 degrees to the axes, because the coefficient
on one side is `1/(2·n_o·n_e)` and on the other `1/(2·n_o²)`, and quartz barely
tells those apart. Tilt tuning works in the first geometry and does nothing in
the second.

The optic axis is the slow one in a positive crystal like quartz and the fast
one in a negative crystal like calcite. The element names its **fast** axis
either way, so which kind of crystal it is only shows in how the retardance
moves when it is tilted.

Still one ray in and one ray out: a real crystal sends the ordinary and the
extraordinary wave off at slightly different angles, so they emerge side by
side. Micrometres for a plate this thin, and the capability report says it is
not modelled.

## Two convention bugs, and how they were caught

Neither could be seen in any power the kernel computed.

**The direction of time.** The transfer matrix is Macleod's, written in
`exp(+iωt)`; the rest of the kernel works in `exp(-iωt)`. Left alone, every
circular state the kernel could produce would come out with the wrong
handedness. Caught by comparing a single layer against an explicit Airy sum,
whose round-trip factor carries a sign.

**The branch of a square root.** Past the critical angle the principal branch,
which is numpy's and so Optiland's, is the *growing* evanescent wave in that
convention. `R = 1` and `T = 0` on both branches, so nothing in the module could
tell, and the reflection phase of every total internal reflection had the wrong
sign: a Fresnel rhomb would have turned the light the wrong way.

The second is worth reporting upstream. `optiland.thin_film` takes the same
branch, so its complex `r` under total internal reflection carries the opposite
phase. Its powers are unaffected.

## The preview does not do this

`TraceConfig3.polarization` turns the state off, and the f32 preview runs with
it off: every element sees unpolarized light and hands on unpolarized light.

That is the **scalar-unpolarized approximation**. It is exactly right for one
interaction and wrong from the second polarization-sensitive element onward. In
the preview a crossed pair of polarizers is bright, because all a polarizer can
be there is a 50% attenuator; in the reference it is dark.

Two parity suites set the flag, so they compare like with like rather than
calling a difference of physics a precision drift. It is not a small difference
where it applies: a coated plate, whose front face polarizes the beam before its
back face transmits it, moves by 2e-2 of a ray's flux between the two models,
ten times the grid tolerance that suite exists to check.

## In the component view

A polarizing element is unlike everything else the component view edits. A lens
has a shape and a coating has a curve in wavelength; these have an **axis**, and
what they do depends on the state of the light reaching them rather than on any
property of the surface alone. There is no R or T to plot.

So the drawing is the element face on, with its axis across it and a handle to
turn it, and beside that:

- **Polarizer:** Malus' law, the fraction passed against the angle of an
  incoming fully linear state, with the flat line at one half that unpolarized
  light gets whatever the angle. Extinction is quoted the way a datasheet quotes
  it, as a contrast ratio.
- **Retarder:** the polarization ellipse that linear light at 45 degrees to the
  fast axis comes out as, on a dial whose angle is the retardance. The quarter
  wave is marked, and the readout names the setting as well as measuring it.

They are asked about before the coating editor, because a polarizing element
answers `surfaceCoatingJSON` too, with nulls where its R and T would be, and
would otherwise open a coating editor showing two empty curves.

## At a detector

Every hit records its state and the frame that state is written on, because
`S1` without an axis names nothing.

The aggregate is summed in the detector's own frame: each ray is rotated onto
the projection of the detector's local `+X` before being added. `S0` equals the
total flux and is always meaningful. The other three are only as meaningful as
that shared frame is, so the worst arrival angle is reported next to them, and
the readout says so above five degrees. Rays arriving from very different
directions have genuinely different transverse planes, and adding their `S1` is
arithmetic rather than measurement.

## What it costs, measured

Two dichroics in series at their band edge: the polarized and the scalar model
disagree by more than 5%.

One tilted slab of ordinary glass at 60 degrees: the polarized answer is higher
by a few parts per thousand, and matches `½(Ts₁Ts₂ + Tp₁Tp₂)` to 1e-9 where the
scalar model gives `½(Ts₁+Tp₁) · ½(Ts₂+Tp₂)`.

The default singlet in the app polarizes its own light by about 0.4%, which is
two glass surfaces met obliquely. Nothing in the canvas could report that
before.

## Still missing

- **One part, one object.** A polarizing cube is a body and a surface, because
  the scene model cannot hold an interface inside a solid. They share a label
  so the object list calls them one thing, and moving one moves only that one.
- **Source polarization.** The tracer accepts a state on a launch ray and the
  tests use it, but no source in a scene declares one. Every scene starts
  unpolarized and has to pass through a polarizer to become anything else.
- **Birefringence.** A ray splitting into ordinary and extraordinary is a
  different feature, and a large one. It is Optiland's to have, not the canvas's.
- **Dispersion of the birefringence.** A crystal waveplate's retardance moves
  as `1/λ`, which is most of it, but `n_o` and `n_e` are each held constant,
  and real quartz is more birefringent toward the blue. Measured against the
  published dispersion equation in
  [`validation.md`](validation.md): under three degrees across the visible for
  a zero-order quarter wave, and over a hundred for a tenth-order one at 450
  nm, because the error is a share of the whole retardance. Fine for a
  zero-order plate, no use to a multi-order one away from its design
  wavelength.
- **Walk-off.** The ordinary and extraordinary waves leave a real crystal side
  by side. One ray goes in here and one comes out.
- **Depolarization.** Nothing here depolarizes. Scatter and roughness would, and
  neither is modelled.
- **Where an acceptance angle comes from.** The contrast curve is data, not a
  model. The kernel will not derive a wire grid's angular response, and nothing
  it can compute about an ideal layer would.
- **Thickness.** A substrate is an index and not a plate: no displacement of a
  beam through a tilted element, no optical path, and no bounce between the two
  faces.
- **Coating phase in the preview.** The baked grid carries powers only. Adding
  amplitudes to it is possible and has not been measured to be worth it.
