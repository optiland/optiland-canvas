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

**The three thin-film solvers, mapped.** Measured, not remembered: one
layer (n = 1.72, 137 nm on 1.52, 550 nm, 0 and 35 degrees) in
`test_tmm_works_in_the_canvas_ray_frame`, and the Fresnel rhomb in
`test_polarization_outputs.py`.

| | Canvas (`FilmAmp`) | Byrnes `tmm` 0.2.0 | Optiland `thin_film` 0.6.1 |
|---|---|---|---|
| Time convention, index | `exp(-iωt)`, `n + ik` | `exp(-iωt)`, `n + ik` | Macleod, `exp(+iωt)`, `n − ik` inside; `t` conjugated on the way out, `r` not |
| `r_p` at normal incidence | `−r_s` (p̂ follows each ray) | `−r_s` | `+r_s` (Macleod's admittance) |
| Evanescent exit medium | decaying | decaying | principal root, the growing wave: phases wrong past the critical angle, and powers too when the stack absorbs |
| `t_p` | whole field | whole field | tangential field |
| Transmittance | `\|t\|²` times `t_power_scale` | `\|t\|² Re(n_f cos θ_f) / Re(n₀ cos θ₀)` (s), conjugated cosines (p) | Macleod's admittance ratio |
| Ellipsometric Δ | `arg(r_p / r_s)`: π at normal incidence on bare glass | `arg(−r_p / r_s)` (`tmm.ellips`): 0 there | not provided |
| Map to the canvas | none for amplitudes; `Δ = Δ_tmm + π` | conjugate `r`; negate `r_p`; `t_p · cos θ₀ / cos θ_exit`; below the critical angle only | |

The rows are held by `test_thinfilm_tmm.py` (amplitudes, no map),
`test_thinfilm_parity.py` and `coating_compare.py` (Optiland's map), the
`total_internal_reflection` and `asymmetric_absorbing_reversed` coating
fixtures (the evanescent branch), `test_c19_*` (Δ) and
`test_optiland_puts_the_rhomb_phase_on_the_growing_branch` (Optiland's
retardance past the critical angle is the negative of the true one).

## Fourier and propagation conventions

The scalar wave optics in `oc-wave` (ADR 0003) uses the same time dependence,
`exp(-iωt)`, and so does its main reference, Goodman's *Introduction to Fourier
Optics*, 4th ed. (W. H. Freeman/Macmillan Learning, 2017, ISBN
978-1-319-11916-4). Everything below follows from it.

| Quantity | Convention |
|---|---|
| Plane wave along `+z` | `exp(+ikz)`, `k = 2πn/λ` |
| Forward transform | `Û(f) = Σ U(x) exp(−i2π f·x) ΔxΔy`, the sampled Fourier integral |
| Inverse transform | `U(x) = Σ Û(f) exp(+i2π f·x) Δf_xΔf_y`, `Δf = 1/(NΔx)` |
| Parseval | `Σ\|U\|²ΔxΔy = Σ\|Û\|²Δf_xΔf_y`, exactly, with the two scales above |
| Field normalization | `\|U\|²` is flux per mm²; `Σ\|U\|²ΔA` is the field's flux, relative unless the source is calibrated |
| Sample layout | sample `i` at `center + (i − ⌊n/2⌋)·Δx`; `ifftshift` puts the window centre at the transform origin |
| Analytic carrier | the true field is `data · exp(ik(α₀x + β₀y))`, with `(α₀, β₀)` direction cosines held outside the samples |

The propagators that land with W1 add their transfer functions to this table:
the angular spectrum `H = exp(+i2πz√((n/λ)² − f²))`, the Fresnel impulse
response `h = e^{ikz}/(iλz)·exp(ik r²/2z)` and the thin lens
`t = exp(−ik r²/2f)` (so `f > 0` converges) are all Goodman's, in this
convention.

Held by `oc-wave/tests/fft.rs`:
- `fft2_is_unitary_under_parseval` pins the scales.
- `the_centered_transform_of_a_shifted_gaussian_is_a_gaussian_with_the_shift_phase`
  pins the sign of the kernel and the centring. A Gaussian shifted by `a`
  must come back as `exp(−πf²)·exp(−i2πfa)`, so a flipped kernel or a lost
  shift fails it.

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
| Wave: band-limited angular spectrum (`oc-wave`) | The first Rayleigh–Sommerfeld solution, scalar, homogeneous isotropic medium, between parallel planes; exact up to the critical distance `LΔx/λ`, band-limited beyond it with the removed power booked | Vector fields, high-NA focusing, coupling to structures; a window too small for the far field shows as booked band-limit loss |
| Wave: Fresnel transfer function and impulse response | The paraxial (Fresnel) approximation, each on its side of the critical distance (Voelz & Roggemann 2009); the zoomed Fresnel transform onto a new window | A field whose paraxial error, computed exactly from its spectrum, is not small (reported `Paraxial`); the other side of `z_c` (reported `OutsideRegime`) |
| Wave: Fraunhofer | Source Fresnel number below 0.1 | Anything nearer (reported) |
| Wave: polarizers and retarders on the wave path | The tracer's Jones matrix at the chief ray (faces, the layer's axis projected across the ray in its glass), applied to the field's two components (Jones 1941) | A field whose directions spread far from the chief ray's: the matrix is one for the whole field |
| Wave: refractive faces and curved mirrors on the wave path | Geometric field tracing across a thin slab (Wyrowski & Kuhn 2011), rays along a reference fixed by the geometry (the beam's paraxial `q`, the chief ray's or a modulator order's direction); a linear operator | A field departing far from the reference (reported: its lateral move θL and its phase kθ²L/2 across the slab) |
| Wave: thin elements (apertures, thin lens, phase screens) | A transmittance in one plane; apertures, element outlines and lens rims anti-aliased by exact cell area (of their projection along the beam, on an oblique plane) | The element's thickness; the sub-cell projection error of a sampled edge (booked as `subcell_projection`), the norm beyond the grid's band not measured apart from it; features below ~14λ, where Pommet, Moharam & Grann (1994) found scalar efficiencies off by more than ±5% for their binary gratings (a heuristic: their 8-level case needs a period above 20λ, and the error grows with index, depth, levels and incidence) |
| Wave: DMD (`oc_core::modulator`, `oc_wave::modulator`) | Thin element: each mirror reflects as its tilted plane, over its outline projected onto the array plane; orders in direction cosines; Harvey's normalization by default ([dmd.md](../dmd.md)) | Mirror shading at oblique incidence, edges, hinges, curvature; the facet's own surface integral, which the array-normal mask approximates (compared in [dmd.md](../dmd.md), flagged above 15° incidence); a mirror under ~14λ (flagged, a heuristic); the superpixel method (not implemented) |
| Wave: LCoS SLM | Thin element: phase from a table at `λ₀`, scaled `λ₀/λ` (flagged); dead space unmodulated; fringing as linear broadening; each polarization component a scalar field, the director's split applied to the pair at the chief ray ([slm.md](../slm.md)) | LC dynamics and flicker, birefringence dispersion, angle dependence of the phase; fringing beyond a box-averaged phase surrogate; an electrode under ~14λ (flagged, a heuristic) |
| DMD and SLM in the ray tracers (reference and preview) | The geometric limit: a ray reflects off the facet it hits, or is steered by the local phase gradient `(λ/2π)∇φ`; reported as approximated. Or, in the reference only, the orders ray model: the brightest orders of the local unit cell as child rays, the rest booked as untraced | In the geometric limit, diffraction (the wave path carries it); in the orders model, coherence between hits, an aperiodic pattern beyond an 8 × 8 window, fringing, a device not in air |
| Wave path (`oc_wave::path`) | A coherent beam along each chain of the chief ray to the detector, unpolarized, its samples carrying the whole phase; arms that meet on the detector added as fields, with unitary splitters (the Stokes signs); spectral lines added in power; a plane-parallel plate (round trips included) and a flat coating plane wave by plane wave, s or p of one plane of incidence; thin elements; a flat mirror by image theory; each face of a body and each curved surface by geometric field tracing across its sag, the field diffracting between faces; an oblique element by projection along the rays (`TiltProjection`) | Diffraction inside a face's slab; polarizing elements (refused); a chosen diffraction order (W7); the tilt projection's depth beyond `Δx²/λ` (reported); light on paths the chain does not follow (booked, never added back); feedback outside a plate; arms of different polarization (visibility an upper bound); planes of incidence that differ (refused); a lens's faces polarization-blind |

**Power data does not determine phase.** `Rs`, `Rp`, `Ts`, `Tp` fix the
magnitudes of a Jones element's diagonal and say nothing about the relative phase
between them. A measured table, or any response stored as powers, therefore
cannot be turned into a full polarization interaction. Where the kernel needs
one it must either declare an ideal phase-neutral attenuator as an explicit model
choice, and say so in the capability report, or refuse. It must not quietly
assume zero retardance and present the result as measured.
