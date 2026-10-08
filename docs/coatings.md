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

### Slabs, and a plate's Mueller matrix

A coating can hold **slabs**: layers flagged `"incoherent": true`, far thicker
than any coherence length the canvas models (a 2 mm window is
`FilmLayer::slab(glass, 2.0)`). A coating with slabs is evaluated by the
incoherent transfer-matrix method of Byrnes' `inc_tmm` (Katsidis and
Siapkas): the films between two slabs are one coherent group, the slabs pass
intensity, and every element gets its absorbed share. That is how a window
with a different AR on each face is written: front films, a slab, rear films,
on a `Medium` backing. It agrees with `tmm.inc_tmm` and
`tmm.inc_absorp_in_each_layer` to 1.5e-15 from either side (C17). A slab sits
on the far medium; a plate-backed coating already is a slab and refuses more
(`SlabOnPlate`), and a body face refuses both, because the body is its own
substrate.

`plate_model` decides what a `Plate` backing is:

- `PhaseFree` (the default, `Standard`): the plate's own two-face series, as
  it always was; for polarized light the faces act as phase-free Jones
  elements and the result says it is approximated.
- `Incoherent` (`Reference`): the plate is its films, its slab and its far
  face through the same incoherent method, and it has a **Mueller matrix**,
  the faces' Mueller matrices with their phases added over every pass:
  `M_R = M(r_E) + M(t_E') M(r_O) (I − M(r_E') M(r_O) P²)⁻¹ M(t_E) P²` and
  `M_T = M(t_O) (I − M(r_E') M(r_O) P²)⁻¹ M(t_E) P`, with `P` the power one
  crossing of the slab keeps. It equals the series summed to 200 passes to
  1e-12, its S0 is the incoherent powers, and it never hands on more than
  fully polarized light (C18). The two models differ, for a lossy slab, by
  the slab's convention at its own faces, a few parts in a million.

## The two tracers

| | Exact (`oc-trace`) | Preview (`oc-gpu`) |
|---|---|---|
| Coating | the transfer matrix, per hit | a baked grid, read bilinearly |
| Angle | the ray's own `cos_i` | interpolated in the ambient-medium `sin²θ` |
| Wavelength | exact | exact at the baked columns |
| Per-face | yes, always | revolved bodies and tubes, with a grid per side when the stack absorbs |

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

The angle axis has 128 rows uniform in the angle, from 0 to 88.2 degrees
(`sin²θ = 0.999`), and holds the last row flat beyond it. It used to be 64
rows uniform in `sin²θ`, which put one row across 82.8 to 88.2 degrees, where a
coating's reflectance climbs towards 1: bare glass and a three-layer AR were
out by up to 8e-2 there, which nothing measured until CF-07 sampled the whole
range. Uniform in angle with 128 rows the worst drift is 5.4e-4 on bare glass,
5.8e-4 on the AR and 1.6e-3 on an eight-period edge filter, and a grid at 64
wavelengths is 64 kB. The angle is measured **in the ambient medium**, not in
whatever medium the ray is in: a coated face of a plate is crossed from both
sides, and what both crossings share is `n sin θ`, which Snell conserves.
Indexing on it lets one grid serve a ray arriving through air and the same ray
coming back out through the glass, **for a lossless stack**: a lossless stack
reflects and transmits the same from either side at conjugate angles.

An absorbing stack does not. Silver behind glass reflects less than silver in
air, so a coated face whose stack absorbs (or whose inside response is not the
outside stack met from behind) is baked twice: from outside, and from inside
with the body's glass as the incident medium. The shader and the CPU mirror
read the inside grid for a ray arriving from inside the body. Reusing the
outside grid there is wrong by more than 1e-2 on a silvered glass tube (C22).

A coating can also be a body's **default** response, which is what the
Fresnel lens's AR is: every face then carries the coating as its own response,
and faces that share one response share one bake (a lens with thousands of
grooves has one grid). Under `coating_model: Nominal` every row of a grid is
the coating's design point, which is what the reference answers then.

Baking a protected-silver face from both sides at 64 wavelengths takes 5 ms in
release (`bake_cost` in `coating_preview3.rs`); the budget is 20 ms.

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

Select a lens in the component view and each face takes an antireflection
coating, following whichever face is selected the same way `R` and `k` do. A
real lens is coated a face at a time and the two are often different.

Every one is solved against **that lens's own glass**: the same three layers on
crown and on flint are not the same coating, because what a stack reflects
depends on what it is sitting on. They are designed at 550 nm on axis, which is
where a general-purpose AR is centred and how a lens face is used. A face meant
for 45 degrees is a different part.

| button | design | what it is |
|---|---|---|
| AR ×1 | `SingleLayerAr` | a quarter wave of MgF₂ at the constant 1.38 |
| ×1 fit | `SingleLayerAr` + `adapt` | the catalogue film nearest √(n₀ n_s), a quarter wave |
| V | `VCoat` | SiO₂ over TiO₂ with an exact null at 550 nm, narrow |
| ×3 fit | `Bbar` + `adapt` | MgF₂, a half wave of Ta₂O₅, and a quarter matched to the glass |
| band | `BandAr` + `adapt` | the same three refined over 400 to 700 nm |
| ×3 nominal | `BroadbandArLegacy` | nominal, not adapted: the old preset, kept so old documents read the same |

The fitted designs (`crates/oc-core/src/synthesis.rs`) take their films from
the catalogue at their index at the design wavelength, and **are solved again
when the lens's glass changes**; a fixed recipe keeps its layers. A fitted AR
is never worse over its band than the fitted single layer: when the three
layers would be, it falls back to two, then one. On N-SF66 the matched
quarter wants an index near 1.8 that no catalogue film has, so `×3 fit` there
is MgF₂ over HfO₂.

Through the default scene, both faces coated, to the detector:

| both faces | crown (default singlet) | N-SF66 (n_d 1.923) |
|---|---|---|
| bare | 0.917 | 0.807 |
| AR ×1 | 0.971 | 0.987 |
| ×1 fit | 0.972 | 0.987 |
| V | 0.950 | 0.973 |
| ×3 fit | 0.991 | 0.990 |
| band | 0.996 | 0.991 |
| ×3 nominal | 0.987 | **0.936** |

The last row is the reason this table exists. The old three-layer preset is a
quarter-half-quarter of constant indices that ignores the glass: on N-SF66 it
passes less than one MgF₂ layer. At its 650 nm design wavelength it reflects
8.1% of 400 to 700 nm per face (the 8 October eyepiece study, held as a
regression in C25). The V-coat is a laser-line part: an exact null at one
wavelength, and more than a single layer over the visible.

MgF₂ is taken at the bulk crystal's ordinary index (1.378 at 550 nm), which a
hot-deposited AR layer sits near. The catalogue's MgF₂ film is a 44 nm sample
that reads 1.423; designed with that number, the fitted single layer passes
0.954 on crown, worse than the constant it replaces.

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

## Measured and vendor data

A vendor's curve is not a coating model and the canvas does not pretend it is
one. `SurfaceResponse::MeasuredTable` (`crates/oc-core/src/response_table.rs`)
carries R and T over wavelength and angle with what the numbers mean: s and p
apart or not, the side and media they were measured from, whether they include
the substrate, and where they came from. Lookup is bilinear in `(λ, cos θ)`.

A curve measured at 45 degrees is not a curve at 52 degrees, and nothing in the
data says what it would be. So outside its data a table either refuses or
reads its edge and says so:

- under `numerics: Strict`, or when the table says `Refuse`, the ray stops and
  the capability report carries the reason, for example "Data supplied at 45
  degrees only; this ray is at 52 degrees";
- otherwise it reads the nearest edge and the result is flagged;
- a one-angle table holds within half a degree of its angle;
- a ray from the side the data were not measured from, or between media more
  than 0.05 away from the ones they were measured between, is out of domain in
  the same way.

The tracer takes a table's s and p powers as a diagonal Jones element with no
phase between them, and says so; an unpolarized or scalar table gives both
polarizations the same number. Two s/p tables in series pass
`0.5 (Ts1 Ts2 + Tp1 Tp2)` of unpolarized light in the polarized trace and the
product of the means in the scalar one: the scalar mode's limitation, held in
`measured_tables3.rs`.

A table of a whole part (`FiniteAssembly`) already holds the substrate and its
rear face, so it goes on a zero-thickness surface and is refused on a body
face, where the glass would be counted twice.

The preview resamples a table onto the same 128-row grid as a coating, the
unpolarized mean, read at the edge outside the data: it cannot refuse one ray.
A table is piecewise linear in `cos θ`, and the grid cuts the corner at its
knots, up to 7.5e-3 on an eight-period edge filter tabulated every 5 degrees.

Import goes through `importMeasuredCsv(id, face, csv, map)`. A CSV says none
of what a table needs, so the map states it all: the wavelength column and its
unit, the angle (one stated angle, or a column for a table in long form), the
value columns and whether they are fractions or percent, the side, the scope
and the media. A transmission-only curve can be read with R declared as
1 − T, and the provenance records that declaration. `AngularTable::from_coating`
writes a table from the canvas's own transfer matrix, which reproduces it at
the knots.

## Coatings at a temperature

A film moves when it warms: its index by `dn/dT`, its thickness by the
expansion coefficient, and with them every edge in the stack. Under
`temperature: Declared` (`Reference` sets it when the scene has a temperature)
a film material with a thermo-optic model is evaluated at the scene's
temperature, or at an object's own (`setSceneTemperature`,
`setObjectTemperature`):

```text
n(λ, T) = n(λ) + dn/dT (T − T_ref)        d(T) = d (1 + α (T − T_ref))
```

One slope and one coefficient, constant over wavelength and temperature: a
first-order model, declared and not validated, and every result under it says
so. A quarter-wave stack's centre then moves by its optical thickness,
`λ₀ (1 + ΔT dn/n)(1 + α ΔT)`, which is what C29 holds the solver to.

No built-in catalogue entry carries a model. The refractiveindex.info database
gives none for these films, and a bulk crystal's `dn/dT` put on a film entry
would be a guess; the catalogue says "no thermo-optic data" for each. A film
without a model is held at its reference index under a declared temperature,
and the result names it. Under `temperature: Off` the scene's temperatures are
never read, and the trace is the untempered one bit for bit. The preview
refuses `Declared` and draws films at their reference.

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
| `Incoherent` | amplitudes asked of a plate-backed coating, or of one with slabs | a plate-backed coating adds its two faces incoherently and has no single complex amplitude |
| `SlabOnPlate` | slabs inside a plate-backed coating | a plate-backed coating cannot hold slabs: write it as its films and a slab on the far medium instead |

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
  on either face of any revolved body, and the lens editor offers the ARs,
  which is what a lens is coated with. A mirror, an enhanced metal or an edge
  filter on a lens face needs `setLensFaceCoating` from script.
- **Wedged plates.** Both faces are parallel. A wedge, which is how real
  beamsplitters push their ghost out of the way, is not authorable.
- **Scatter at the rim.** A ray that leaves through the edge of the glass is
  traced as glass, not as the ground edge it would really meet.
- **Optimized designs beyond the band AR.** `BandAr` is a bounded,
  deterministic Nelder-Mead over thicknesses only: no needle optimization, no
  matching sections, no material changes. A solar AR on low-iron glass reaches
  0.475 of bare glass over 0.4 to 2.5 µm and 0 to 40 degrees (C27); a lower
  floor needs a film below MgF₂, such as porous silica, which the catalogue
  does not have.
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
