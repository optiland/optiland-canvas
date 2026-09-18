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
| Polarizer | Passes the field on its transmission axis, absorbs the perpendicular component, leaks its stated extinction. |
| Retarder | Adds a phase to its slow axis. Lossless. |

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
coating these do not work in the plane of incidence: they use the surface's own
local `+X`, projected into the plane the ray's field lives in.

That makes them standalone-surface responses. A body patch carrying one is
refused rather than given an axis nobody chose, because a curved patch of a
solid does not define one.

Both are zero thickness. They are the polarization action and not the crystal:
no walk-off, no ordinary and extraordinary rays, no optical path, and a
retardance that is the number you gave rather than one that follows from a
thickness and a birefringence at each wavelength.

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

- **A polarizing beam splitter** as a part. The physics is there (a coating with
  very different s and p, in a cube), but nothing builds one, and a real PBS
  needs its prism geometry and an internal splitting response rather than a
  zero-thickness plate.
- **Source polarization.** The tracer accepts a state on a launch ray and the
  tests use it, but no source in a scene declares one. Every scene starts
  unpolarized and has to pass through a polarizer to become anything else.
- **Birefringence.** A ray splitting into ordinary and extraordinary is a
  different feature, and a large one. It is Optiland's to have, not the canvas's.
- **Dispersion in a retarder.** The retardance is one number, not a thickness
  times a birefringence, so a quarter wave is a quarter wave at every
  wavelength. Real waveplates are not.
- **Depolarization.** Nothing here depolarizes. Scatter and roughness would, and
  neither is modelled.
- **Coating phase in the preview.** The baked grid carries powers only. Adding
  amplitudes to it is possible and has not been measured to be worth it.
