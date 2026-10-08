# Physical conventions

A convention is not a detail. A sign copied from the wrong book, or a square root
taken on the wrong branch, produces numbers that look entirely reasonable, pass
every energy check, and are wrong in a way nobody notices until a chiral element
sends light the other way round. Two such bugs were found in this kernel the week
this page was written, both invisible to every power the code computed, and both
caught only once something needed a phase.

So the conventions live here, stated as arithmetic rather than as words, each one
next to the test that holds the code to it.

This page covers what the physics needs. The world frame, its handedness and the
transform order are [`docs/architecture/coordinates.md`](../architecture/coordinates.md);
this page does not restate them.

## Units

| Quantity | Unit | Where it is converted |
|---|---|---|
| Length, position, aperture, optical path | mm | Never; mm is the kernel's only length |
| Vacuum wavelength | micrometres | At the source and at every UI edge |
| Coating layer thickness | nanometres | `thinfilm.rs`, once, at the phase thickness: `δ = 2π/λ · n · d/1000 · cos θ` |
| Angles in the kernel | radians | `to_radians` at the UI boundary, never inside |
| Angles in the UI and in authoring calls | degrees | |

Wavelength in micrometres against length in millimetres is a factor of 1000 that
appears in exactly one place, and it is written out there rather than folded into
a constant.

## What a ray carries

A ray is a **packet of flux**, not a radiance and not a field amplitude. It is
relative flux unless a source is calibrated, and nothing in the canvas calibrates
one. Increasing the sample count divides the same source flux into more packets;
it does not emit more light.

A ray also carries a **polarization state** (below), whose `S0` *is* that flux.
There is no second scalar: if the two ever disagreed, one of them would be a lie,
and there would be no way to tell which.

## Time, and the sign of an imaginary index

Fields vary as **`exp(-iωt)`**. A wave travelling along `+z` is `exp(i(kz - ωt))`,
so passing through thickness `d` multiplies the field by `exp(+ikd)`, and a
passive absorbing medium has index **`n + ik` with `k ≥ 0`**.

Two consequences that are easy to get wrong and impossible to see in a power:

**The transfer matrix is not in this convention.** `crates/oc-core/src/thinfilm.rs`
follows Macleod, who writes `exp(i(ωt - kz))` with index `n - ik`. The module
stores `n + ik` and conjugates on the way in, so its internal arithmetic is
Macleod's and its amplitudes are Macleod's. Converting them to the kernel's
convention is a conjugation, applied once, in `stack_amp`. Conjugation maps
Macleod's layer matrix term by term onto the other convention's, and `r` and `t`
are rational functions of that matrix with real coefficients, so conjugating the
result is the whole of the conversion. Held by
`a_single_layer_matches_the_airy_formula_and_fixes_the_sign`, which compares one
layer against an explicit two-interface Airy sum whose round-trip factor is
`exp(+2iδ)`.

**An evanescent medium needs the decaying branch.** Past the critical angle the
propagation cosine is a square root of a negative real number, and the principal
branch (numpy's, and so Optiland's) is the *growing* wave in Macleod's
convention. For a lossless stack every power is identical on both branches,
`R = 1` and `T = 0`, so nothing noticed until a reflection phase was needed.
`decaying_cos` puts the exit medium on the branch where `Im(n cos θ) ≤ 0`. A
finite layer needs no such correction: its matrix is even in `cos θ`.

The branch is not only a phase question. When the stack absorbs, the evanescent
field reaches into the absorbing layer and the reflected power depends on the
branch: attenuated total reflection off a metal film comes out wrong by tens of
percent on the growing one. Optiland 0.6.1 is on the growing branch there; the
canvas and `tmm` agree to rounding (the `asymmetric_absorbing_reversed` coating
fixture, and `test_optiland_diverges_past_the_critical_angle_when_the_stack_absorbs`).
The rescaling of the evanescent `t_p` from tangential to whole field uses the
same decaying cosine, so it does not flip the sign the admittance fixed.

**The two thin-film oracles, mapped.** Measured on one layer (n = 1.72, 137 nm
on 1.52, 550 nm, 0 and 35 degrees) in `test_tmm_works_in_the_canvas_ray_frame`:

| Oracle | Time | p basis | `t_p` | Map to the kernel |
|---|---|---|---|---|
| Byrnes `tmm` 0.2.0 | `exp(-iωt)`, `n + ik` | `r_p = -r_s` at normal incidence | whole field | none |
| Optiland `thin_film` | Macleod, `exp(+iωt)` | `r_p = +r_s` at normal incidence | tangential, `t` already conjugated | conjugate `r`; negate `r_p`; `t_p · cos θ₀ / cos θ_exit` |

**Transmittance into an absorbing exit medium.** The kernel computes
Macleod's `T = |t_M|² Re(η_exit) / Re(η₀)`; Byrnes' `tmm` computes
`T = |t|² Re(n_f cos θ_f) / Re(n₀ cos θ₀)` for s and
`T = |t|² Re(n_f conj(cos θ_f)) / Re(n₀ conj(cos θ₀))` for p. They are the
same number. Macleod's quantities are the conjugates of the `exp(-iωt)` ones
(index `n̄ = n − ik`), so his tilted admittances are `η_s = conj(n cos θ)` and
`η_p = conj(n / cos θ)`, and conjugation leaves every real part and modulus
alone.

- s: `Re(η_s) = Re(n cos θ)` and `|t_M| = |t|`, which is Byrnes' s form.
- p: Macleod's `t_M` is the tangential field, `t_M = t · cos θ_f / cos θ₀` up to
  conjugation, so `|t_M|² = |t|² |cos θ_f|² / |cos θ₀|²`; and
  `Re(n / cos θ) = Re(n conj(cos θ)) / |cos θ|²`. Then
  `|t_M|² Re(η_p,f) / Re(η_p,0) = |t|² Re(n_f conj(cos θ_f)) / Re(n₀ conj(cos θ₀))`,
  the `|cos θ|²` factors cancelling: Byrnes' p form.

Held by the `silica_on_opaque_aluminium` coating fixture (an absorbing exit
medium against `tmm`, R to 1e-15) and by the 2000-stack sweep of
`test_thinfilm_stress.py` against the arbitrary-precision oracle.

## Surfaces and sides

A geometric normal identifies which material is on which side. The tracer hands
every response a normal **oriented against the incoming ray** (`n̂ · d̂ < 0`), and
`n1`/`n2` are the media on the incoming and outgoing sides of *that* orientation.
A response must never reorient a normal for its own convenience: face-forwarding
a shading normal would swap the media and reverse the coating stack.

## Polarization

Implemented in [`crates/oc-core/src/polarization.rs`](../../crates/oc-core/src/polarization.rs).

### The state

A power-weighted Stokes vector `S = (S0, S1, S2, S3)`, with `S0` the packet's
flux. Unpolarized light is `(Φ, 0, 0, 0)`. A single Jones vector cannot represent
unpolarized light and is not used as the state.

Every physical state satisfies `S0 ≥ 0` and `S1² + S2² + S3² ≤ S0²`.
`Stokes::is_realizable` checks it, and every ideal element is tested to preserve
it.

### The frame

A ray carries a unit vector `x̂` perpendicular to its direction. The second axis
is implied:

```text
ŷ = d̂ × x̂        so        x̂ × ŷ = d̂
```

In an isotropic medium a straight ray has nothing to rotate its frame, so between
interactions `x̂` is constant in world space.

For a transverse field `E = Re{(Ex, Ey) exp(-iωt)}` written on that frame:

```text
S0 = |Ex|² + |Ey|²        S1 = |Ex|² - |Ey|²
S2 = 2 Re(Ex* Ey)         S3 = 2 Im(Ex* Ey)
```

`S1 > 0` is linear along `x̂`. `S2 > 0` is linear at +45° from `x̂` toward `ŷ`.
`S3 > 0` is the state whose field vector turns from `x̂` toward `ŷ` as time
advances; because `x̂ × ŷ = d̂`, an observer the beam travels **toward** sees that
as clockwise, which is right-handed in the traditional optical sense. The test
`a_right_circular_state_turns_the_way_the_convention_says` reconstructs the field
at two instants and checks the direction of the turn, so the sentence above is
pinned to arithmetic and not to the word "right".

A frame rotation by `θ` about the direction of travel:

```text
S1' =  cos2θ S1 + sin2θ S2
S2' = -sin2θ S1 + cos2θ S2
```

### The plane of incidence

```text
ŝ = normalize(d̂ × n̂)        p̂ = d̂ × ŝ
```

with `n̂` the ray-opposing normal. **Both** the incoming and the outgoing ray use
the same `ŝ`, because the plane of incidence is the same for both; `p̂` follows
each ray's own direction, so at normal incidence `p̂_reflected = -p̂_incident`.

That choice fixes the sign of every p coefficient in the kernel. It is the
convention in which

```text
r_p = (n₂cos θᵢ - n₁cos θₜ) / (n₂cos θᵢ + n₁cos θₜ)
```

which is **minus** what Macleod's admittance algebra produces. The flip happens
once, in `stack_amp`, and `at_normal_incidence_the_polarizations_are_equal_and_opposite`
catches both a missing flip and a doubled one.

Macleod's `t_p` is the amplitude of the *tangential* field, `cos θ_sub / cos θ₀`
of the whole. The whole-field value is what a polarization calculation needs, so
`t_p` is rescaled in the same place. Powers never need the rescaling, because the
same factor cancels against the admittance ratio.

At exactly normal incidence there is no plane of incidence. `incidence_axis`
returns `None` and the caller keeps the ray's own frame, which is the continuous
limit: at normal incidence `r_s = r_p` in the lab basis, so every azimuth gives
the same physical answer.

## Model validity

| Model | Supported domain | Outside it |
|---|---|---|
| Thin-film stack | Isotropic, non-magnetic, laterally uniform, coherent layers; real non-absorbing incident and exit media; absorbing layers allowed | Complex external media are outside the stated `R`/`T` normalisation; birefringent layers are not modelled at all |
| Thin-film amplitudes | `CoatingBacking::Medium` only | A `Plate` backing adds its two faces incoherently and has no single Jones matrix; `coating_amp_at` returns `None` rather than inventing a phase |
| Plate backing | Substrate far thicker than the coherence length | Thin substrates would interfere and do not here |
| Scalar responses (`Mirror`, `ExplicitRt`, `DielectricFresnel`) | Unpolarized light, one interaction | They carry no phase, so they cannot be given a Jones matrix; see below |
| GRIN | Not implemented; traces as a homogeneous medium at `base_index` | Any quantitative GRIN claim |

**Power data does not determine phase.** `Rs`, `Rp`, `Ts`, `Tp` fix the
magnitudes of a Jones element's diagonal and say nothing about the relative phase
between them. A measured table, or any response stored as powers, therefore
cannot be turned into a full polarization interaction. Where the kernel needs
one it must either declare an ideal phase-neutral attenuator as an explicit model
choice, and say so in the capability report, or refuse. It must not quietly
assume zero retardance and present the result as measured.
