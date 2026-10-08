# Coatings

A coating in this kernel is a stack of layers, and what it does to a ray is
worked out from those layers at the ray's own wavelength **and its own angle of
incidence**. That second variable is new, and it is the whole point.

## The bug this replaced

Until this landed, every surface response in the kernel was a number or a table
in wavelength. None of them knew what angle the light arrived at. So a dichroic
filter reported the same reflectance to a ray at 45 degrees as to one straight
down the axis.

That matters because a dichroic is almost never used on axis. Every beam
splitter and dichroic the app places sits at 45 degrees, because that is what
folds a beam. And a multilayer coating's band moves when you tilt it: the phase
thickness of every layer carries a `cos θ`, so at 45 degrees each layer looks
thinner and the whole reflectance band walks toward the blue, by roughly 8% for
a typical dielectric pair.

For a filter labelled 550 nm that is a shift of 40 to 50 nm. It is the
difference between passing a laser line and blocking it. The canvas was drawing
one and reporting the other.

## What it computes

For one wavelength and one angle, each layer gets a characteristic matrix

```
M_j = [ cos δ_j          i sin δ_j / η_j ]
      [ i η_j sin δ_j    cos δ_j         ]
```

with phase thickness `δ_j = 2π/λ · n_j · d_j · cos θ_j` and tilted optical
admittance `η_j = n_j cos θ_j` for s, `n̄_j² / (n_j cos θ_j)` for p. The stack
matrix is their ordered product from the incident side inward, and with
`[B, C]ᵀ = M · [1, η_sub]ᵀ`:

```
r = (η₀ B − C) / (η₀ B + C)      R = |r|²
t = 2 η₀ / (η₀ B + C)            T = |t|² · Re(η_sub)/Re(η₀)
```

`A = 1 − R − T` is what the layers absorbed. This is Macleod, *Thin-Film Optical
Filters*, chapter 2.6. Complex indices work, so metal films are ordinary members
of the model rather than a special case.

s and p are computed separately throughout, and the reference tracer now keeps
them apart all the way to the detector: see [`polarization.md`](polarization.md).
The same matrix also yields complex amplitudes, because a coating's retardance is
as real as its reflectance, and a quarter-wave stack near its band edge puts tens
of degrees between s and p.

## Why Optiland is the reference

The formulation follows `optiland.thin_film` deliberately closely: the same
forward branch for the propagation cosine, the same conjugated index in the p
admittance. That makes Optiland a numerical oracle rather than a second opinion.

`crates/oc-py/tests/test_thinfilm_parity.py` runs seven stacks through both
implementations and subtracts, holding them to 1e-11. The cases are picked for
what they can break: a bare interface, which must reduce to plain Fresnel; a
single quarter wave; the mirror stack and the symmetric-period edge filter the
canvas synthesizes; an aluminium film, which is the only case that exercises the
complex propagation cosine and the conjugated admittance at all; a metal under a
dielectric; and a stack lit through glass.

Agreement to a few percent would not catch a wrong branch. Agreement to 1e-11
cannot survive one.

## Edge filters are symmetric periods

A plain quarter-wave mirror, `(HL)^N H`, has a clean stopband and a violent
passband: about 20% mean reflectance with 37% peaks just past its edge. On a
filter whose entire job is to cut cleanly at one wavelength, that makes "where
does it cut" close to meaningless, and the first version of the edge solver
happily placed the 50% crossing on a ripple peak 35 nm from the real band edge.

So edge filters use the symmetric period `(0.5H L 0.5H)^N` (Herpin's equivalent
layer), which halves the ripple to about 10% mean and 12% peak. Adjacent half
layers merge, so it is stored as `0.5H (L H)^(N-1) L 0.5H`: the same film, fewer
rows in the editor.

**That 10% residual ripple is real and it stays.** Commercial edge filters get
under a percent with a computer-optimized matching section of tens of extra
layers, which this does not attempt. The component view plots the ripple rather
than drawing the clean edge nobody's coating actually has. A one-parameter
search over the outer layer thicknesses confirmed the plain symmetric period is
already the optimum of that family, so there is no cheap win being left on the
table.

## Placing the edge

A stack built at a given quarter-wave wavelength does not put its band edge
there: the edge is offset from the band centre by the index contrast.

Rather than a textbook width formula, the solver uses the fact that a
quarter-wave stack of non-dispersive materials is **exactly scale-invariant**.
Its whole reflectance curve is a function of `λ₀/λ`, so building the stack once,
measuring where its edge landed and rescaling by the ratio lands it on target in
a single step. A few repeats absorb any dispersion. Three spectral scans, where
a bisection would have wanted sixty.

The search window is held to ±40% of `λ₀`, because a quarter-wave stack also
reflects at `λ₀/3`, `λ₀/5` and so on, and those harmonics are just as strong: a
wider window finds one as the peak and rescales the design onto it.

## A plate is a piece of glass

A dichroic is a coating on a glass plate, not a coating floating in air, and for
a while this modelled it as a plane. That got the energy right and the geometry
wrong in four ways at once: the reflection left the mid-plane rather than the
front face, the transmitted beam carried on undisplaced, the glass added nothing
to the optical path, and the far face returned no beam at all. For a 3 mm plate
at 45 degrees the first two are about 2 mm and 1 mm, which is not a rounding
error on a 15 mm part.

So a dichroic is a **body**: a disc of N-BK7 with the stack on the face an axial
beam meets first and plain glass on the other. None of the four corrections is
special-cased. The exact tracer already resolved per-face responses and
refracted through solids, so pointing the geometry at reality was enough:

```
   incident          coated face        far face
   ----------------->|                  |
                     |\                 |
                     | \ refracted      |
                     |  \ path          |
                     |   \              |
                     |    +-------------|-----------> transmitted:
                     |                  |             parallel, displaced
                     |<-- reflected     |
                                        `--> and its own reflection,
                                             back through the glass
```

The displacement is `t·sin(θ − θ′)/cos θ′`, measured perpendicular to the beam.
`a_thick_plate_walks_the_transmitted_beam_sideways` checks the traced geometry
against that closed form rather than against yesterday's behaviour: 3 mm of
N-BK7 at 45 degrees moves the beam 1.0014 mm, and the formula says 1.0015.

Because the tracer now crosses the glass itself, the coating on that face is
backed by the **scene's** glass (`CoatingBacking::Medium`) rather than folding a
substrate into its own numbers. Folding one in as well would count the far face
twice.

`CoatingBacking::Plate` remains for the zero-thickness idealisation, where the
far face is combined **incoherently** (millimetres of glass are far beyond any
coherence length here, so intensities add and the bounces between the faces sum
as a geometric series). Documents written before plates had thickness carry that
shape, and it is what `coating_rt` returns.

`thickness_mm` gives the plate its glass: each pass through it is attenuated by
the substrate's own `k` along the slant path, `τ = exp(−4πk·d / (λ cos θ))`,
and the series becomes `R = r₁ + t₁t₁′ r₂τ² / (1 − r₁′ r₂τ²)`. Nothing else
reads it: the plate stays zero thickness, so the beam is not displaced. With
the layers on the far face and `Incidence::Substrate`, this is a second-surface
mirror; `a_thick_plate_absorbs_on_every_pass` checks both against the closed
form.

## The two tracers

| | Exact (`oc-trace`) | Preview (`oc-gpu`) |
|---|---|---|
| Coating | the transfer matrix, per hit | a baked grid, read bilinearly |
| Angle | the ray's own `cos_i` | interpolated in the ambient-medium `sin²θ` |
| Wavelength | exact | exact at the baked columns |
| Per-face | yes, always | revolved bodies only |

Running a multilayer matrix per ray in a shader is not on, so the stack is baked
onto a grid when the scene compiles. Two things keep that honest.

The CPU mirror and the WGSL shader read the *same* grid, so those two stay
bit-identical and the parity suite keeps its meaning. What drifts is the preview
against the exact matrix, and only where the curve is steep.

And the grid is baked at the wavelengths the scene actually emits, not over a
fixed spectral range. A source carries a few lines, so a three-line scene bakes
192 points instead of tens of thousands, every ray lands exactly on a sample in
wavelength, and angle is the only axis that interpolates at all. That matters
because the preview recompiles the scene on every frame of a drag. Moving to the
scene's own wavelengths dropped the worst drift from 1.0e-2 to 7.6e-4 of a ray's
flux, and made the bake 40 times cheaper.

The angle axis is uniform in `sin²θ` rather than in θ, because that is very
nearly what the band shift is linear in. It is measured **in the ambient
medium**, not in whatever medium the ray is in: a coated face of a plate is
crossed from both sides, and what both crossings share is `n sin θ`, which Snell
conserves. Indexing on it lets one grid serve a ray arriving through air and the
same ray coming back out through the glass.

Two things about that grid were wrong before plates had thickness, in ways only
a body exposes. It was baked with air behind the coating, which is right for a
plate in air and badly wrong for a film on glass, so it now bakes against the
body's own index at each wavelength; and the angle axis was the local one.
Together those cost 3.9e-2 of a ray's flux, now under 2e-3.

`cpu3_vs_f64` holds the coating's branch *structure* exactly and its flux to
2e-3, stated in the test rather than borrowed from the general flux tolerance.
The numbers the UI shows never come from this path.

## In the app

`＋ dichroic` builds a 3 mm coated glass plate, its stack solved for the 45
degrees it is hung at. The angle comes from the placement: a plate's normal runs
along `(cos φ, sin φ, 0)`, so an axial beam meets it at `arccos|cos φ|`.

That formula is worth stating because the wrong one is invisible at exactly 135
degrees, where two plausible formulas agree, and only a test at 180 degrees
caught it.

The component view's `film` mode plots the coating at the angle the surface is
actually used at, with the same stack's on-axis curve behind it in grey. The gap
between those two lines is the bug this replaced, drawn rather than described.
The readout gives the layer count, the angle, and where the built stack really
crosses 50% there, measured rather than assumed, so a design that cannot deliver
the edge it was asked for says so.

Under the plot the stack is drawn as its layers, left to right from the incident
medium inward, each as wide as it is thick and shaded by index. Every boundary
drags, changing the thickness of the layer to its left.

## Coating a lens

Select a lens in the component view and each face takes one of two
antireflection coatings, following whichever face is selected the same way `R`
and `k` do. A real lens is coated a face at a time and the two are often
different.

Both are solved against **that lens's own glass**: the same three layers on
crown and on flint are not the same coating, because what a stack reflects
depends on what it is sitting on. They are designed at 550 nm on axis, which is
where a general-purpose AR is centred and how a lens face is used. A face meant
for 45 degrees is a different part.

What it is worth, for the default singlet:

| face | reflects at 550 nm |
|---|---|
| bare crown | 4.2% |
| single quarter wave of MgF₂ | 1.3% |
| quarter-half-quarter | lower again, and flatter across the visible |

Through the default scene, to the detector:

| both faces | passes |
|---|---|
| bare | 0.917 |
| AR ×1 | 0.971 |
| AR ×3 | 0.987 |

The five percent that stops bouncing around inside the instrument is the
reason coatings exist.

**What an AR coat still reflects is a ghost.** It is the residual the coating
failed to suppress, stray light for exactly the reason bare glass's reflection
is, so the ghost switch decides whether it is traced, and with the switch off
it is named in the ledger as truncated flux. A dichroic's reflected band, a
mirror, and a polarizing splitter's s are the opposite case: the reflection is
the part doing its job, and it is followed whichever way the switch is set.
The design says which a stack is (`Coating::reflection_is_stray`), and a layer
edit keeps the design, so a hand-thinned AR coat is still an AR coat. A stack
with no design reads as a path. Until this was drawn, every coated face took
the dichroic's branch, and coating a lens made its reflections appear as
primary rays with the ghost layer off.

**Light leaving the lens meets the coating from behind.** The layers are
written from the air side inward, so a ray inside the glass meets them in the
other order, and the face's inside copy says so (`Coating::seen_from_behind`).
A single layer is its own mirror image and cannot tell. The three-layer AR is
not, and walked in the written order from inside it put the layer meant to
face the air against the glass: it reflected more than bare glass, and the
lens passed 0.911 of the light, less than with no coating at all. The preview
was never wrong about this. It bakes the grid from outside and reads a ray
from inside at the conjugate angle, which a lossless stack allows because it
reflects and transmits the same from either side; that is the property the
test holds the two walks to, to 1e-12.

**Past the critical angle the reflection is the ray.** Whether a reflection is
stray is a question only while light gets through. A ray inside the glass past
the critical angle cannot get out whatever the coating, because the air
carries only an evanescent wave, so a coated face reflects it totally, exactly
as bare glass does: a primary path, flagged as total internal reflection, and
never a ghost. The transfer matrix gives R = 1 there for a lossless stack, and
the test holds every such bounce to that. The preview cannot read this off its
grid, which is baked from outside and saturates at grazing, so it takes R = 1
outright. That is exact for every coating the app makes; a hand-written
absorbing stack would reflect a little less in the reference than in the
preview.

A coated lens is still a lens. What separates a plate from a lens is curvature,
not the film, so the object list and the component view keep calling it one and
it keeps the lens editor. That was not free: the view used to ask "does this
body carry a film" before "is this body a lens", which was right while only
plates carried films.

`ar_residual`, the single number that used to be the only way to say a lens was
coated, is still on every uncoated face. It is not a coating model and its own
doc comment says so.

## Checked against published data

The angle shift of an edge is the one relationship every filter vendor
publishes, and it can be checked without knowing anybody's stack: one effective
index, fitted once at 45 degrees, predicts this synthesis's edge to within 0.06
nm from 10 to 50 degrees. See [`validation.md`](validation.md), which also says
where a vendor curve is an oracle and where reproducing one would only be a
test of guessing their design.

## When a coating has no answer

A coating either gives a number or says why it cannot, by name. It never
reflects everything quietly and never returns NaN. The reasons, with the
sentence each one prints:

| Reason | When | Sentence |
|---|---|---|
| `Degenerate` | the stack's matrix is singular (refused under `numerics: Strict`; under `Permissive` the stack reflects everything and the result is flagged, as it always did) | the coating's transfer matrix is singular at this wavelength and angle, so it has no reflectance |
| `OutOfTable` | a layer's or the substrate's table does not cover the wavelength: always when the table's policy is `Error`, and under `Strict` whatever its policy | layer 3 has no optical constants at 0.2500 um: its table does not reach that wavelength |
| | the same for the substrate | the substrate has no optical constants at 0.2500 um: its table does not reach that wavelength |
| `NonPhysicalIndex` | n not positive or not finite; a thickness negative or not finite; k negative under `Strict` | layer 1 is not physical: the index must be positive and finite, the extinction coefficient not negative, and the thickness finite and not negative |
| `AbsorbingIncidentMedium` | an `Opaque`-backed coating lit from its substrate side | light cannot arrive through an absorbing substrate: an opaque-backed coating is lit from its film side only |
| `Overflow` | the arithmetic left the finite numbers despite the guards below | the coating's arithmetic overflowed, so it has no finite answer |
| `Incoherent` | amplitudes asked of a plate-backed coating | a plate-backed coating adds its two faces incoherently and has no single complex amplitude |

Under `Strict` the tracer turns each of these into an
`UnsupportedSurfaceResponse` that carries the sentence; under `Permissive`
the first two are answered as before and flagged in the capability report.

Two guards keep thick absorbing stacks finite. A layer whose phase thickness
has an imaginary part above 35 is **opaque**: it passes `exp(-70)` of the
power, so it becomes the substrate and nothing behind it counts. That is
Byrnes' limit in `tmm`, which clamps where the canvas truncates; the two differ
by less than `exp(-70)`. And the accumulated matrix is rescaled whenever an
entry passes `1e100`, which leaves `r` exact and lets `t` underflow to zero
where it used to overflow. Neither guard touches a stack that gave a finite
answer before it existed. 5 µm of aluminium used to return NaN; it now
reflects what bulk aluminium does, to 1e-12.

A ray at exact grazing (cos θ below 1e-9) is reflected entirely by rule, with
`r = -1` in both polarizations of the ray frame.

An `Opaque` backing is a semi-infinite absorbing substrate: a metal block, or a
mirror whose metal is far thicker than its skin depth. What the stack sends
into it is absorbed there; the tracer books it as surface absorption and spawns
no transmitted ray.

## Where a coating's absorption goes

Under `layer_absorption: PerLayer` (the `Reference` preset) the tracer says
which layer of a coating absorbed, and books a plate's slab as bulk
absorption:

- each layer's share comes from the net Poynting flux through the stack's
  interfaces, `F_k = Re(B_k C_k*) / Re(B_0 C_0*)` from the same characteristic
  matrices the reflectance uses, so `A_k = (1 − R)(F_{k−1} − F_k)` and
  `R + T + Σ A_k = 1` to rounding; a lossless layer absorbs exactly nothing;
- a plate's coating is met from the front and, on every return pass, from
  inside, and its slab takes the rest: under `PerLayer` that goes to the
  ledger's `absorbed_volume`, under `PerFace` it stays in `absorbed_surface`
  as it always did;
- an opaque substrate keeps what enters it, booked at the surface and
  deposited as `Bulk`;
- polarized light weights the s and p shares by its intensity in each.

`TraceConfig3.deposit_bodies` and `deposit_surfaces` choose which objects
record deposits, and each `Deposit3` says where: `Face`, `Film(k)` (written
order) or `Bulk`. The heliostat light pipe's `THERMAL=1` export carries this
as a seventh column. Per-layer absorption agrees with `tmm.absorp_in_each_layer`
to 1e-9 (C15) and the deposits sum to the ledger to 1e-10 (C16).

## What is still missing

- **Polarization** is no longer missing. It has its own page,
  [`polarization.md`](polarization.md).

  It belongs in this one's list of gaps because a coating is where it bites.
  The dichroic `＋ dichroic` builds, at 45 degrees, reflecting:

  | λ (nm) | R_s | R_p | unpolarized mean | degree of polarization |
  |---|---|---|---|---|
  | 460 | 0.9997 | 0.9937 | 0.9967 | 0.3% |
  | 520 | 0.9961 | 0.8668 | 0.9315 | 6.9% |
  | 540 | 0.9773 | 0.3493 | 0.6633 | 47% |
  | 560 | 0.7782 | 0.0203 | 0.3993 | **95%** |
  | 620 | 0.0969 | 0.0173 | 0.0571 | 70% |

  Deep in the band both polarizations saturate and the mean is harmless. At the
  edge, which is exactly where a dichroic is used, the reflected beam is 95%
  polarized. The reference tracer now carries that state on to whatever stands
  in the folded arm. The f32 preview still takes the mean, which is the right
  flux for one element and wrong for the second.

  A **polarizing beam splitter** as a part is still missing, and so is any
  source that emits polarized light without a polarizer in front of it.
- **Anything but an AR on a lens, from the app.** The core will put any design
  on either face of any revolved body, and the lens editor offers the two ARs,
  which is what a lens is coated with. A mirror or an edge filter on a lens
  face needs `setLensFaceCoating` from script.
- **Wedged plates.** Both faces are parallel. A wedge, which is how real
  beamsplitters push their ghost out of the way, is not authorable.
- **Scatter at the rim.** A ray that leaves through the edge of the glass is
  traced as glass, not as the ground edge it would really meet.
- **Dispersive films.** Layer indices are `SpectralScalar`, so a measured `n(λ)`
  serializes and traces, but the presets are constants. Real films disperse by a
  percent or two.
- **Optimized designs.** No needle optimization, no matching sections. The
  synthesized stacks are textbook quarter-wave designs and behave like them.
- **Optiland export.** A film has no place in a v1 `SurfaceSpec`, and
  `planar_reduce` refuses to reduce one rather than picking an angle and baking
  it in.

## The old dichroic

`SurfaceResponse::ExplicitRt` with a raised-cosine table is still in the kernel
and still ignores the angle. It has to: documents written before this exist, and
giving them different physics on load would be its own kind of lie. Nothing in
the UI creates one any more.

`coatings3.rs` tests both, and the difference between them at 45 degrees is
exactly the size of the bug.
