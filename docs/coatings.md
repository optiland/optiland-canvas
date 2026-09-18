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

s and p are computed separately throughout. The tracer is unpolarized and
consumes their mean, but the two are kept apart because at 45 degrees a
dichroic's s and p edges sit tens of nanometres apart, and a mean that hid that
would be worse than no number at all.

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

## Plates have two surfaces

A dichroic is a coating on a glass plate, not a coating floating in air. The
light that gets through the stack still has to get out of the substrate, which
costs about 4% unless that face is anti-reflected.

So a `Plate` backing carries its substrate and its far face, combined
**incoherently**: millimetres of glass are far beyond any coherence length here,
so intensities add and the multiple bounces between the two faces sum as a
geometric series. Dropping that series would lose the second pass of a highly
reflective coating entirely.

The plate stays zero-thickness geometrically. The transmitted ray continues
undeviated with no lateral offset, which is the same approximation the splitter
has always made.

## The two tracers

| | Exact (`oc-trace`) | Preview (`oc-gpu`) |
|---|---|---|
| Coating | the transfer matrix, per hit | a baked grid, read bilinearly |
| Angle | the ray's own `cos_i` | interpolated in `sin²θ` |
| Wavelength | exact | exact at the baked columns |

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
nearly what the band shift is linear in.

`cpu3_vs_f64` holds the coating's branch *structure* exactly and its flux to
2e-3, stated in the test rather than borrowed from the general flux tolerance.
The numbers the UI shows never come from this path.

## In the app

`＋ dichroic` builds a real stack, solved for the 45 degrees the plate is hung
at. The angle comes from the placement: a plate's normal runs along
`(cos φ, sin φ, 0)`, so an axial beam meets it at `arccos|cos φ|`.

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

## What is still missing

- **Polarization.** s and p are computed and the component view could show them,
  but the tracer carries no polarization state, so it consumes their mean. A
  polarizing beam splitter cannot be represented.
- **Coatings on lens faces.** The response exists for a body surface and the
  exact tracer handles it, but the preview refuses one by name rather than
  guessing. Lens faces still use the scalar `ar_residual`.
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
