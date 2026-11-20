# Scalar wave optics (`oc-wave`)

A complex scalar field sampled on a plane, and the methods that move it from
one plane to another by diffraction. The crate sits beside the ray tracer, not
inside it, and the two meet only through optical path length
([ADR 0004](architecture/adr/0004-scalar-wave-optics-engine.md)). Everything is
f64.

**What exists today:**
- the propagators, including a zoomed far field on any output grid;
- thin elements;
- analytic beams;
- a coherent wave path through a scene, along the chain the ray tracer finds;
- the report that accounts for every step.

All of it is usable from Rust, and the propagators from Python
(`optiland_canvas.wave`). In the app, the detector dock's wave mode runs the
wave path in a worker ([in the app](#in-the-app)).

The pixelated modulators built on it, the DMD and the LCoS SLM, have their own
pages: [dmd.md](dmd.md) and [slm.md](slm.md).

Every method below is a published one, and every validation names the source it
is held to. The numbers in brackets refer to [Sources](#sources) at the end.

## Conventions

The phase conventions are the kernel's, written out in
[`physics/conventions.md`](physics/conventions.md#fourier-and-propagation-conventions):

- time dependence `exp(-iωt)`;
- a forward plane wave is `exp(+ikz)`;
- the forward transform kernel is `exp(−i2πfx)`, scaled by `ΔxΔy` so that
  Parseval holds exactly.

`|U|²` is flux per mm², so a field's power is in the same relative units as a
ray's flux.

The field may carry an analytic carrier: a plane-wave tilt `(α₀, β₀)` in
direction cosines, held outside the samples so a steep beam does not need a
steep sampled phase. Propagators account for it.

## Propagators

`oc_wave::propagate(field, z_mm, Propagator, &mut Fft2)` returns the new field
and a [`WaveReport`](#the-report).

| `Propagator` | What it computes | Valid where | Source |
|---|---|---|---|
| `AngularSpectrum` | the first Rayleigh–Sommerfeld solution, by plane-wave decomposition: `H = exp(i2πz√(1/λ² − f²))` | homogeneous medium, any distance; band-limited by default | [1], [2] |
| `FresnelTf` | Fresnel transfer function, `H = e^{ikz} exp(−iπλz f²)` | paraxial; `z ≤ z_c` | [1], [3] |
| `FresnelIr` | Fresnel impulse response, `h = e^{ikz}/(iλz) exp(iπr²/λz)` | paraxial; `z ≥ z_c` | [1], [3] |
| `Fraunhofer { pad }` | one transform onto the far-field grid, pitch `λz/(PΔx)` | source Fresnel number below 0.1 | [1] |
| `RsDirect` | the Rayleigh–Sommerfeld integral by FFT convolution, Simpson-weighted | kernel sampled to the largest offset (below) | [4] |
| `Auto` | one of the above, chosen and recorded | — | [3] |

Two more evaluate the angular spectrum somewhere other than the input's own
grid:

| Function | What it computes | Valid where | Source |
|---|---|---|---|
| `asm_zoom(field, z, out, pad)` | the plane waves advanced `z` exactly and summed on any output window and pitch by chirp-z, so a focal spot is sampled far finer than the field and a window may sit off the axis | the padded window covers the output's offset (a field repeats a padded window away) | [1], [7], [9] |
| `asm_tilted(field, plane, out, pad)` | the field on any plane: tilted, shifted and distant at once, each plane wave read in the target frame, `(u, v, w) = Rᵀ(u′, v′, w′)`, with the Jacobian `\|∂(u, v)/∂(u′, v′)\|` and the phase to the plane's origin | the source spectrum is read between its samples by gridding, a Kaiser–Bessel kernel ten samples wide on the spectrum of the field padded at least twice [30], [31]: the kernel's aliasing, 2e-10 to 7e-10 of the peak from 0° to 60° in its test; each target frequency is met by two wavevectors, `w′ = ±√(1/λ² − u′² − v′²)`, and each that is a forward wave of the source is kept, so a plane steeply tilted holds every wave that crosses it; what it cannot hold is booked by cause (its evanescent waves as `evanescent`, waves beyond the target spectrum's span as `spectral_projection`, the rest off the window as `left_window`); waves within 6° of grazing it are kept and warned of (`OutsideRegime`); the power through the window takes one obliquity, the carrier's | [6], [7], [30], [31] |

**The critical distance** `z_c = LΔx/λ`, for a window `L` wide after padding,
divides the two Fresnel methods [3]. The window is taken as the propagators
pad it, `P = next_fast_len(2n)` samples a side, axis by axis, and `z_c` is
the smaller axis's (`padded_critical_distance_mm`). Up to `z_c` the angular spectrum's
transfer function is sampled finely enough everywhere and its band limit
removes nothing. Running a Fresnel method on the wrong side of `z_c` is allowed,
since comparisons need it, but the report carries an `OutsideRegime` warning.

**The band limit** [2] keeps the transfer function inside both ellipses of
Eqs. (18)–(19):

- `u²/u_limit² + λ²v² ≤ 1`;
- `λ²u² + v²/v_limit² ≤ 1`;
- with `u_limit = λ⁻¹[(2Δu·z)² + 1]^(−1/2)` (Eqs. 13, 20) and `Δu = 1/(PΔx)`.

These equations were read from the paper's text. The power the limit removes is
booked as `band_limit_removed` and reported as a `BandLimited` warning. It is
never renormalized back into the field. The paper's caveat holds: toward the
far field the angular spectrum needs a window much larger than the source
before its band limit stops cutting what matters. The tests show that, and
`Auto` hands such steps to the Fresnel impulse response.

**`Auto`'s rule:**
- the band-limited angular spectrum up to `z_c`, where it is exact and the
  band limit removes nothing;
- beyond `z_c`, whichever of the Fresnel impulse response and the
  band-limited angular spectrum has the smaller model-error indicator on
  this very field, both computed from its spectrum padded as the step pads
  it before either runs, and both written into the step's note;
- when both indicators exceed `STEP_TOLERANCE`, 1e-2 of the field, the step
  still runs with the smaller, and the report says it is not certified
  (`OutsideRegime`).

The indicators are what each model leaves out of the exact angular
spectrum on this field, not certified totals: a sampled Fresnel step's
quadrature, its kernel's truncation and its window are not among them (Q6
of the expert review). The propagators' convergence with pitch, window,
padding and output sampling is tested on its own, against fields no finite
spectrum shares (below).

The paraxial step's error is known exactly. With `κ = √(k² − |q|²)` and its
paraxial form `κ₂ = k − |q|²/2k`, the exact and the paraxial propagators
differ on a field `g` by

```text
‖(U − U₂)g‖² = Σ_q 4 sin²(z(κ − κ₂)/2) |ĝ(q)|²,   κ − κ₂ = −|q|⁴ / (2k(k + κ)²),
```

which follows from Parseval's theorem because both are diagonal in the plane
waves; the difference is written in its stable form. `paraxial_error` sums it
over the field's own spectrum, its carrier included, the field padded as the
step pads it (its spectrum sampled `1/(PΔx)` apart per axis, Q7 of the
expert review), so a narrow beam tilted far from its plane's normal is not
mistaken for a paraxial one, as a bound from the window's corner (the old
rule, `πρ⁴/(4λz³) < 0.1` rad [1]) would. An evanescent plane wave, which the
exact propagator damps and the paraxial one carries unchanged, counts by the
difference of the two transfer factors, `|e^{−z√(|q|² − k²)} − e^{izκ₂}|²`.
The band limit's indicator is the field it would remove, `band_limit_error`,
the same that the band-limited step books, to round-off. Every Fresnel
step, chosen or forced, says its paraxial error in its note, and above
`1e-4` raises the `Paraxial` warning with it. The identity holds to 5e-8
between the canvas's own two propagators, and to 2.1e-12 on a field with
evanescent content (tests below).

**Padding and wrap-around.** The angular spectrum and both Fresnel methods
zero-pad to `2×` the samples (rounded up to a size with only 2, 3 and 5 as
factors). That makes the convolution linear over the original window. A field
periodic in its window, such as a grating's Talbot carpet, can use
`AsmOptions { pad: 1, .. }` and propagate exactly. A propagated field with more
than a millionth of its power in the outer sixteenth of the padded window gets
a `WrapAround` warning. The power cropped off the window is booked as
`left_window`.

**Direct Rayleigh–Sommerfeld** [4] samples the kernel out to the largest
source-to-observation offset `L`. Its local frequency there must stay below
Nyquist: `Δx ≤ λ√(z² + L²)/(2L)`. This is the impulse-response condition of
[3] for a linear convolution. Outside it the kernel aliases into ghost
images, and the report warns `Undersampled` rather than pass them off as a
field. This method cannot separate power that diffracted out of its window
from quadrature error, so its ledger books that gap as `numerical` and says
why.

## Zoomed far field

`oc_wave::czt::fraunhofer_zoom(field, z_mm, out_grid, &mut Fft2)` evaluates the
Fraunhofer field on any output window and pitch. Typical uses are a focal spot
sampled finer than the FFT's `λz/(PΔx)`, or one diffraction order. It uses the
chirp-z transform: Bluestein's identity `mn = (m² + n² − (m − n)²)/2` turns the
Fourier sum into a convolution at FFT cost [9]. Y. Hu et al. [8] use it for
diffraction in the same way. A published Comment disputes their novelty for
scalar fields, not their correctness.

The output samples only part of the far field. The power outside it is booked
as `left_window`.

## Thin elements

`oc_wave::elements`:

- `apply_aperture` takes a rectangle, disc, annulus or polygon (`Shape`).
- `apply_thin_lens`: `t = exp(−ik r²/2f)`.
- `apply_phase`: a phase screen.
- `apply_transmittance`: a complex mask with `|t| ≤ 1`.

These are the thin-element approximation [1]: a transmittance applied in one
plane, nothing modelled inside the element.

**Apertures are anti-aliased by exact area.** An edge crossing a cell
transmits that cell's covered fraction `f`, computed exactly:

- by interval overlap for a rectangle;
- by integrating the circle's chord in closed form for a disc;
- by Sutherland–Hodgman clipping and the shoelace area for a polygon.

The edge error then falls as `Δx²`, which the Heurtley test below measures.
Physically the cell transmits `f·|U|²ΔA`, but the sample `f·U` carries only
`f²·|U|²ΔA`. The difference, `(f − f²)|U|²ΔA`, is the sub-cell projection
error of the sampled edge: the norm the cell average discards. It is booked
as `subcell_projection`, separately from the `absorbed_aperture` of the
opaque part. It is not the same claim as "beyond the grid's band": a cell
average is not a band projector (Q3 of the expert review). `edge_budget`
measures the two apart: the mask laid `s` times finer, its cells tiling the
grid's, its spectrum taken on the grid's own frequencies, and what the band
holds of it subtracted from the open area. For a 0.2 mm strip whose edges sit
on cell boundaries the sub-cell error is nought and 1.0% of the power still
lies beyond a 10 µm grid's band; a quarter into a cell the two are 1.9% and
1.0%; both halve with the cell (R19, test below). The samples keep the
coherent amplitude `f·U`; `√f·U` would close the power and change the field.

The thin lens warns `Undersampled` when its chirp changes by more than π
between neighbouring samples at the window edge.

## Beams

`oc_wave::beams`:

- **`GaussianBeam`:** Kogelnik and Li's fundamental mode [13], in this
  kernel's convention: `U = A·(q₀/q)·exp(ikr²/2q)·e^{ikz}`, with
  `q = z − iz_R` and `A = √(2P/πw₀²)`. They write `exp(+iωt)`, which
  conjugates every phase. It is a paraxial solution, so an exact propagator
  moves it differently by about the square of its divergence, `(λ/πw₀)²`.
- **`plane_wave`:** held as an analytic carrier, so any angle is sampled
  exactly.
- **`point_source`:** a diverging spherical wave.

## The wave path

`oc_wave::path` (feature `scene`) carries a coherent beam through a scene,
from a source to a rectangular detector, along every chain by which the
source's axial ray reaches it. The f64 tracer's chief ray decides where the
beam goes; the field decides what arrives. This is the coupling
ADR 0004 chose: rays for the layout, fields for the diffraction, joined by
optical path [17].

```rust
let spec = WavePathSpec {
    source, detector,
    beam: BeamProfile::Gaussian { waist_radius_mm: 0.3, waist_offset_mm: 0.0 },
    power: 1.0,
    grid_n: 512, window_mm: 4.0,      // the field across the beam
    device_samples_per_pixel: 4,      // when the beam is laid onto a DMD or SLM
    readout: Readout::Projected,      // or Readout::Diffracted { samples_per_pixel }, a floor on the points a pixel
    detector_model: DetectorModel::NormalFlux,   // or ParaxialArea
};
let result = wave_path(&scene, &spec, revision)?;   // power per detector pixel, the field, the report
// or, from the beam the source itself declares:
let spec = WavePathSpec::for_source(&scene, source, detector, 512, 4.0)?;
```

**How the path is planned.** `extract_wave_sequences3`, the wave path's form of
`extract_sequences3`
([ADR 0002](architecture/adr/0002-sequence-extraction-from-the-trace.md)), gives
each chain of surfaces the chief ray meets on its way to the detector, at
each of the source's lines; there are several where splitter arms recombine
on the detector (below). It differs from the plain extraction twice, so that a
chain stands for the field rather than for its axial ray (R17 and R18 of the
expert review). An aperture does not stop the axial ray: an annular stop's
centre blocks it, but the field passes around it, and the path lays the stop
on the field where the chain crosses it and notes that the axial ray is
stopped there. And under the geometric limit a DMD is traced once for each
mirror family its pattern holds, every mirror taken in that family, so each
family the field lights has its chain whatever the one pixel or gap the axial
ray meets. The tracer logs no event when a ray
passes an aperture's opening, so the path intersects the chief ray with every
aperture plane itself and inserts the crossings. The field rides in a frame
across the chief ray, its transverse axes carried round each turn by the
shortest rotation.

**Between events** the field propagates the chief ray's arc length with
`Propagator::Auto`. **At an event** it is handed from the plane across the
incoming ray to the plane across the outgoing one:

| Element | What the field does | Exact? |
|---|---|---|
| flat mirror | mirror image, amplitude `−√R`; what it does not reflect is absorbed | yes, in scalar optics (image theory [1]) |
| beam-splitter arm (`ExplicitRt`) | `−√R` reflected from its front, `+√R` from its back, `√T` transmitted, on the arm the chain took; the other arm's share is booked as `other_arm`, the rest as absorbed | as the element's R and T; the signs make it unitary (below) |
| grating | its order carries the grating's own phase, `exp(i m K·(r − r_g))` with `K = 2π/d` across the grooves from the surface's origin: with the chain along the tracer's order direction that is the order's momentum, and what is left, `exp(i m K·(p − r_g))`, moves with the grating; amplitude `√η`; the other orders are booked as `untracked_orders`, once when the order and the zero order both reach the detector | thin element |
| ideal thin lens | `exp(−ik\|r − c\|²/2f)` about the lens's centre; the chain's rays go along this phase's gradient, the same operator ([conventions](physics/conventions.md)) | thin element |
| a flat element's outline | the element acts inside its own shape, the edge weighted by exact cell coverage; light outside passes it by, on the chain where the chain goes straight on (a transmitted arm, a thin lens), off it as `passed_by` where the chain turns (a mirror, a grating's order) | thin element |
| aperture | as the tracer has it: light passes inside the opening, is stopped between the opening and the outer extent, and passes beyond the outer extent (an aperture with none is unbounded); each edge weighted by exact cell coverage, of the shape itself square to the beam and of its projection along the beam otherwise | thin element |
| DMD, LCoS SLM | laid onto the device plane at `device_samples_per_pixel`, reflected by the device model ([DMD](dmd.md), [SLM](slm.md)), taken up across the outgoing ray; at an LCoS, half of what it reflects follows the chain (below) | thin element |
| face of a refractive body, curved mirror or other curved surface | traced across the thin slab that holds the surface where the beam meets it (below), its rim laid on the field with exact cell coverage; between a body's faces the field diffracts inside the glass by the angular spectrum, in the glass's index | the slab is geometric: no diffraction across a surface's sag |
| plane-parallel plate (a body entered and left by two flat, parallel faces, bare or with a residual AR) | one step: the closed-form slab, round trips included, on each plane wave of the field, then the angular spectrum across the planes (below) | exact for a plate of infinite extent, s or p; its rim laid on the field |
| flat coated surface (`Coating`: a coated splitter, a dichroic) | each plane wave times the coating's complex `r` or `t` at its own angle, from the transfer matrix, then handed over like any flat element; a plate-backed coating's round trips as its plate declares (`PlateRoundTrips`): the first pass alone (the default), added in power, or summed as fields across its thickness | exact for the stack, s or p; the plate as it declares it, zero-thick in the scene's geometry |
| detector (`Readout::Projected` or `Readout::Diffracted { samples_per_pixel }`) | one readout for every arm: the field, carried to the detector's plane across its chief ray, is interpolated onto the detector's own sub-pixel grid, `s × s` points a pixel at the centres of equal parts of each pixel, by the zoomed angular spectrum (`asm_zoom`) where the face is square to the beam with its axes along the beam's and by the spectrum rotated onto the face (`asm_tilted`) otherwise, its carrier kept analytic; the field's spectrum is padded so that one period holds the whole face (and, on a face too wide for that within the target's widest padded grid, the field's own span on it), and the face beyond that period reads zero, where the sum would lay the field's copies (a focal spot on a detector many times its window); `s` holds the intensity's band on the face (the arms' directions' spread plus twice the envelope's band) at `READOUT_PER_PERIOD` (8) points a period for the band that holds all but 1e-6 of the envelope's energy and at `READOUT_PER_PERIOD_TAILS` (2.5) for its whole band, down to 1e-8 of its peak amplitude (a field cut by its window has spectral tails to its grid's Nyquist frequency, whose beat with the main lobe is of first order in the edge's amplitude: read at points that do not resolve them, a Michelson's arm gained 1e-4 of its power; resolved above two points a period, the sum over the points is the interpolant's power exactly), two at least, and `Diffracted` sets a floor of its `samples_per_pixel`; a pixel's power is the sum of its cells' integrals, the fringe between two arms' carriers integrated over each cell exactly (the rectangular pixel's sinc factors) and the envelopes by the midpoint rule, the irradiance by the `DetectorModel` (below) | exact scalar propagation; the midpoint rule on the envelopes, `(πfh)²/6` of an intensity component at frequency `f` for points `h` apart (2.6% at the band's edge, 0.64% at half of it); padded so that the field's periodic copies, a padded window apart, stay clear of the face (`Periodic` warns where 4096 samples a side cannot); the sample cap (`CoarsePreview` says what the points fell short of) |

When the beam leaves an element at another angle than it came, its
cross-section changes by `cos θ_out / cos θ_in`; the handover's amplitude
factor `√(cos θ_in / cos θ_out)` keeps the power right [19]. When an
element's plane is not square to the beam, the handover projects the field
along the rays rather than diffracting it across the plane's depth. That is
exact for a flat mirror. For a lens, an aperture or a modulator seen
obliquely, the report carries `TiltProjection` with the depth spanned, once
that depth exceeds `Δx²/λ`. An aperture and an element's outline are laid on
the field as their projection along the beam, whose edges are covered
exactly: a disc becomes an ellipse, drawn as a polygon of up to 1024 sides
with the disc's own area, departing from it by under a five-hundredth of a
sample. A tilted detector is not projected: the spectrum is rotated onto
its own plane (`asm_tilted` [6]). Elements on oblique planes still are.

**Through a lens: one face at a time.** A refractive body is not carried
through whole by its rays. Inside its Rayleigh range a beam's curvature does
not grow by the distance travelled (for a Gaussian, `dR/dz = 1 − z_R²/z²`),
so a 0.2 mm Gaussian transported by rays through a 10 mm plate would leave
as if the plate were 5.6 mm shorter rather than the 3.4 mm it is to the beam
(the plate test below holds the difference). Instead each face is a
transfer across a thin slab, planes a micrometre clear of the surface where
rays over the beam window meet it (found by probing the face with rays,
and along each direction by bisection for the face's edge, so a lens smaller
than the window is not missed), and between faces the field diffracts
inside the glass. Across a slab the f64
tracer carries one ray from each point of a grid, up to 257 × 257, through a
copy of the scene holding the element alone, along a reference wavefront
fixed by the geometry (below). Each ray gives
where it crosses the exit plane, its optical path and its transmittance; the
field that leaves is

```text
U_out(x') = U_in(x) · √(T / |∂x'/∂x|) · exp(ik₀ OPL(x)),
```

conserving power ray tube by ray tube (F. Wyrowski and M. Kuhn's geometric
field tracing [17]). Light whose ray misses the face is stopped as by a mount.
The edge of what passes, the rim, is found by bisection on traced rays along
720 directions from the chief ray and laid on the arriving field as a polygon
with exact cell coverage, so it diffracts at the field's own sampling, not the
ray grid's; the traced map is carried two rays past it. On a 1 mm lens rim in
a 2.8 mm beam, four field samples a ray, resolving the rim to the ray grid
instead stopped 3.9e-3 too much of the power and moved the focal spot by
6.8e-2 of its peak (the rim test below). A rim that is not star-shaped about
the chief ray is still resolved to the ray grid, and the report says so. The
faces' reflection (unpolarized) is booked as `other_arm`: light the chain does
not follow, not absorbed. A curved mirror is the same transfer with the exit
plane across the reflected ray. The report names the rays traced, how far
they moved across the beam, and how far the arriving field's directions
depart from the reference, with what that departure moves.

**A transfer that is one operator.** The rays a face is traced with follow a
reference wavefront that the field does not choose. The path carries the
beam's Gaussian parameter `q` across each axis of its frame from the declared
beam (a top-hat as the Gaussian of its radius) through every element before:
free space adds its length, a thin lens its power, a plate the free space its
glass is worth (`t/n`), and a traced face its paraxial matrices, from five
rays through it, in reduced form across the change of index
(`q̂' = (Aq̂ + B)/(Cq̂ + D)`, `q̂ = q/n` [13], [28]). The reference is the
wavefront `Re 1/q` about the chief ray. Traced for a reference and a window,
the map is the transfer's operator, the same for any field on that window: a
field's two polarization components take one map, and
`T(u₁ + u₂) = T(u₁) + T(u₂)` to round-off (5.7e-17). With a sphere fitted to
each field it was given, as before, the same two Gaussians tilted ±10 mrad
missed it by 0.15: their sum was traced along a direction that was neither's.

A wrong reference costs more than its lateral walk `θ·L` across the slab. A
field tilted `θ` from the rays it is traced along gathers the phase of their
straight run, where its own plane waves advance by `k cos θ`: a piston of
`kθ²L/2` too much. Against the exact plane-wave-by-plane-wave interface
(Goodman [1]; Born & Wolf [16]), a Gaussian tilted 10 mrad through a plane
face 0.4 mm deep is within 4.9e-4 traced about its own direction, and 0.15
off traced about the chief ray, its piston 0.1486 rad against the 0.1479 of
`k₀θ²/2·(L_air + L_glass/n)` (tests below). So each face's report gives the
departure, the steepest where the field holds a hundredth of its peak and the
power-weighted rms, with the move it makes across the slab and the phase it
gathers there (the local direction from the phase steps between adjacent
samples, each in (−π, π], so it aliases only past π a sample; the
difference across two samples aliased past π/2, R08 of the expert review).
A field of several directions can hide them from its phase: `1 + 0.5 cos(qx)`
has no phase gradient anywhere, and is three plane waves, two `q/k` off the
axis, which a slab of free space traced geometrically, as long as dephases
them by π, leaves `2/3` of the field's norm away from the angular spectrum
(R07). So the report also gives the rms direction of the field's angular
spectrum about the reference, the move that makes across the slab and the
dephasing about `kθ²L/2` it brings; where the move exceeds a sample the
transfer is reported not certified (`OutsideRegime`). Through a face that
changes nothing (index 1 on both sides), the geometric slab differs from
the angular spectrum by `√2` times that estimate for a Gaussian, the exact
first-order factor, whatever the face's place in the slab (R10).

A modulator's field holds several directions at once: its orders. Where a
traced face lies ahead, the field leaving a DMD or an LCoS is cut, in its
angular spectrum, into the cells of the device's lattice of orders
(directions `λ/p` apart in the device's plane [10]; about the specular
reflection for a DMD, whose facets only weight the orders, and about the
followed beam for an LCoS), and each order is carried apart as far as the
detector with a reference ray of its own: its direction, walked through free
space, turned by a thin lens and traced through each face. Each order then
crosses each face about its own direction, and the orders add back into one
field at the detector. At most nine are carried apart, each of at least 1e-4
of the power; a weaker one joins its nearest, and one that free space has
emptied rejoins the strongest. Through an N-BK7 singlet, R ±8 mm, a DMD
whose blaze lies 27 mrad from its nearest order crosses both faces with its
orders departing from their own references by at most 0.29 mrad where the
field holds a hundredth of its peak; the power-weighted rms, down to 1e-4 of
the peak, reads 3.3 mrad (3.8 by the spectrum), the weak orders folded at
the window's band edge (the difference across two samples read 0.36 by
folding their steps). Traced about the chief ray, its brightest orders would
have departed by their 27 and 36 mrad.

The orders carried apart are the lattice's. A hologram's own orders, inside
one cell of the lattice, share their cell's reference, and their departure
from it is in the report.

**One phase clock.** The samples carry the whole phase: every propagator
multiplies them by its full transfer function, `e^{ikz}` included, and a
traced face by `exp(ik₀·OPL)` with the optical path from its entry plane. The
field's `opl_ref_mm` records the reference ray's optical path to the frame
origin, for comparing a field with a trace; it is never applied again. So a
plate of thickness `t` delays the beam's centre by `k₀(n − 1)t` against free
space, less the Gouy phase its shorter effective length does not accumulate
(the plate-phase test below, within 2.6e-6 rad), and two fields that left one
source by different paths can be added as they are.

**Reading a field between its samples.** The transfer reads the arriving
field at the points its rays start from, which fall between samples; so do
the hand-over onto an oblique element and the resampling onto and off a
modulator. Bilinear interpolation of a complex field whose phase turns by `φ`
from one sample to the next lowers its amplitude in between,
`|(1 − s) + s·e^{iφ}|² = 1 − 2s(1 − s)(1 − cos φ)`, a power loss of about
`φ²/6` per axis averaged over a cell. A beam converging inside a lens turns
most of a radian a sample at its edge, which lost 3.7% of the power at the
singlet's second face and dimmed its pupil edge. So a smooth wavefront is
taken out of the samples first and put back exactly at the point read (a
transfer's reference; for the other hand-overs a sphere and tilt fitted to
the field, which only choose how it is interpolated); only the slowly varying
rest is interpolated. The singlet's
unexplained remainder fell from 3.7% to 0.04% of the launched power. What
interpolation still costs is booked at each step as `resampling`.

**Windows fitted to the beam.** One window for a whole path cannot both hold
a wide beam and resolve the focus it is brought to: a 1 mm beam focused by
f = 100 mm has a 20 µm waist, under two samples of a 6 mm window of 512. So
before each free-space step the path predicts the beam's size at the next
plane from the field's own second moments, its variance `σ²`, the variance
of its slopes `σ_θ²` and their cross moment `C = ⟨(x − x̄)θ⟩`. A slope is how
far a plane wave walks across the beam per unit length, `t/√(1 − |t|²)` for
direction cosines `t`, the transverse velocity `q/k_z` that only the
paraxial limit equates with `t` (Q16 of the expert review); the local slope
comes from the phase steps between adjacent samples, the fitted sphere taken
out first so that a steep wavefront is not understated. Then
`σ²(z) = σ² + 2zC + z²σ_θ²`, exact under paraxial propagation whatever the
beam's shape (the second-moment law; A. E. Siegman [25]): a predictor, not a
bound on the tails, which is why the step is checked after it runs (below).
The slopes' moments come from the samples' spectrum, exact for a field its
samples resolve. Behind a lens whose curvature turns more than π a sample at
the beam's edge the spectrum folds those plane waves back to smaller angles,
the slopes' spread comes out short of `C`'s, and a focus is predicted to
vanish: the app's default singlet, 256 samples over 4 mm at 0.42 µm, folded
about 1% of the power, gave `C²/σ²σ_θ² = 1.033` and a negative `σ²` at the
detector, and the window there shrank to 64λ and cut 16% of the light. Where
the local direction passes the samples' band on more than a millionth of the
power, the slopes' moments are taken from the same local directions as `C`,
with the diffraction part (the spread no local direction explains,
`⟨|∇A|²⟩/(k²⟨A²⟩)` in the paraxial limit) from the spectrum of the field
with its fitted sphere taken out, which the samples do resolve; that
beam's waist is then predicted within 5e-5 of a Gaussian's own. Either
way the spread is kept from breaking the uncertainty relation
`σ²σ_θ² − C² ≥ (λ/4π)²` (Siegman's `M² ≥ 1` [25]). When the beam, taken as six rms
radii to either side of its centroid, would fill less than a quarter of the
window, or on a window already narrowed more than four fifths of it, the
field is carried onto a window fitted to it, centred on its predicted
centroid, with the same sample count and never wider than the declared
window. A window so centred lies off the chief ray, and every step takes
it where it is: the zoomed and the rotated spectrum sum their plane waves
about the window's centre, where the transform is taken; a traced face's
ray grid and a plate's probe span the window wherever it lies; a
modulator's footprint is the window's own corners (the off-centre tests
below). The zoomed angular spectrum carries it, exactly, padded enough that
its plane waves' walk over the step stays inside; where that walk exceeds the
window and the step is longer than the window's critical distance (a narrow
window widening again, typically), the zoomed Fresnel transform, the Fresnel
integral as one chirp-z transform onto the new grid, its paraxial error
reported, but only where that error's indicator is within `STEP_TOLERANCE`
or no padding up to 8 (and the target's widest padded grid, 4096 samples a
side, 2048 in the browser) can hold the walk; else the
zoomed spectrum, padded. A Fresnel step taken past the tolerance is reported
not certified. After each step the path checks what the window kept: where
a window narrower than the declared one lost more than a millionth of the
launched power off its edge, the step is run again from the field before it
on a window twice as wide, and the one that loses less to the window, off
its edge and beyond its band together, is kept, with both losses in the
report (Q16); through the f/10 singlet of the Optiland comparison the second
window lost 1.2e-3 of the launched power where the first lost 2.9e-3. A coarser window holds fewer directions: the plane waves beyond its
band are taken out of the field first, and booked as `spectral_projection`,
for free space keeps every plane
wave and the new samples, read at their points, would fold them back in at
directions they do not have. A pinhole at a focus is then laid on the focus
at the focus's own sampling, and a 10× beam expander keeps its beam resolved
from 0.2 mm through a 50 µm focus to 2 mm (tests below). A modulator met by a
narrow beam, on a window finer than a few samples a pixel, is laid on a
device grid at least twice as fine as the beam's own (up to 32 samples a
pixel): read between device samples it did not have, the DMD's take-up onto a
0.1 mm beam's window resampled 7.5% of the power at four samples a pixel, and
1.0% at 32. A device grid finer than the window the field leaves on holds
directions that window cannot, as a coarser window does: the plane waves
beyond the outgoing window's band, about the reflected family's carrier, are
taken out of the device's field before the take-up and booked as
`spectral_projection`. Read between device samples with them still in, a DMD's
facets folded 17% of the power back into the band in the DMD-and-lens test
below (booked as `resampling`) and weighted its orders wrongly; taken out
first, the resampling there is 1.3e-4.

**What the path follows at a modulator.** Under the geometric limit the chief
ray, and so the beam, leaves a DMD along the reflection off a family's facet,
the chain carrying the family whose direction it leaves along, and an SLM
along the phase gradient there (or specularly, for the unmodulated arm).
Under the orders model (`RayModel::Orders`), as the tracer has it (the device
in air, its orders in closed form), each order the tracer follows is a chain,
and the chain owns its order: of every family's reflected field it keeps the
lattice cell of its order, about its direction on the lattice of the
pattern's unit cell where the axial ray meets it, the families' cells added as
fields and taken up about the order's own direction. What the chains share,
the device's absorption and its mask's band, is booked once, and the cells no
chain follows once, as `untracked_orders` (R16 of the expert review; before,
each order's chain carried a copy of the whole spectrum). Of the field
an LCoS meets, the component along its director is modulated and the one
across it is not [20]: the chain carries two components (below) and takes
into its beam the one that beam is made of. Of the other, the device takes
the same share (its fill and reflectance do not tell the beams apart), and
the rest goes into the other beam, booked as `other_family`. Where both beams
reach the detector, each is a chain of its own; they meet orthogonally
polarized and add in power, and the ledger books each beam's share once. The
field keeps
every diffraction order that falls inside its window; orders outside are
booked as `spectral_projection` or `left_window`. Under the geometric limit it
does not steer onto a chosen order; under the orders model each followed order
has its own chain, but a Fourier-plane filter that picks one order is not yet
a tested case.

**Planar layers, plane wave by plane wave.** A planar interface keeps every
plane wave's tangential wavevector, so each plane wave of a field meets it on
its own, at its own angle: in the field's angular spectrum the interface is a
multiplication, with no projection and no ray. A plane-parallel plate is so
treated as one step: the closed-form slab
`t₀₁t₁₂e^{iδ}/(1 − r₁₀r₁₂e^{2iδ})`, `δ = k₀n_g L cos θ_g`, relative to free
space across the same planes (`× e^{−ik₀n₀L cos θ₀}`), then the exact angular
spectrum from the plane before the plate to the plane behind it, the field
shifted onto the chief ray's exit. The round trips between the faces are in
the sum, so an etalon's multiple-beam fringes come out of it, and so does the
lateral shift a tilted plate gives the beam (the slope of the slab's phase
across the spectrum). A tilted plate's round trips leave the beam a
round-trip shift apart, `2L tan θ_g cos θ₀`; those that would land beyond the
window are summed out of the series and booked as `left_window`, never
wrapped back in. What that takes from the observable is booked exactly,
plane wave by plane wave, `|full|² − |kept|²`, and the step's note splits it
into the omitted round trips as a field of their own, `lead·x^{M+1}/(1 − x)`,
whose power has `|1 − x|²` below (summing the round trips' intensities,
`1 − |x|²`, would leave out how they interfere), and their interference with
the round trips kept, at most twice the product of the two norms (Q11 of the
expert review); the kept round trips are the beams summed one by one to
2.4e-16 (test below). A plate of lower index than its surroundings reflects
whole what meets it past its critical angle; across a thin gap the
evanescent wave would tunnel (frustrated total internal reflection, Born &
Wolf [16], §1.5.4), which is not modelled: the step refuses a gap thinner
than five decay lengths by name, and otherwise names the leak, at most
`e^{−2κL}`, it leaves out. The
amplitudes are in the tangential chart and power coordinates of a scattering
two-port: `r = (y₁ − y₂)/(y₁ + y₂)`, `t = 2√(y₁y₂)/(y₁ + y₂)`,
`y_s = n cos θ`, `y_p = n / cos θ`, unitary, with the Stokes signs of the
splitters. That two-port, `S = [[−√R, √T], [√T, √R]]`, has
`S†S = (R + T)·I`: passive for any excitation of both ports at once when
`R + T ≤ 1`, which a splitter given the same reflection sign from both sides
would not be (Q12 of the expert review). The signs are a convention: a basis
convention changed consistently moves no measured power, while a real
coating's relative phase, which the transfer matrix carries, moves an
interferometer's fringes. A flat coated surface multiplies each plane wave by
its coating's complex reflection or transmission, from `oc_core::thinfilm`'s
transfer matrix put into that chart (its ray-frame `r_p` negated, its
transmission scaled into power coordinates). A coating on a plate takes the
round trips between its two faces as the plate declares (finding 7 of the
expert review: a monochromatic source does not lose its coherence because a
substrate is thick, so the choice is the part's, and the report names it).
`FirstPass`, the default, is the stack's amplitude into the glass times the
back face's out of it, the light the faces send round the plate booked to
the other arm. `Incoherent` adds the round trips' powers,
`T = T₁T₂/(1 − R₁′R₂′)`, the first pass's field carrying them, as the ray
tracer does (Katsidis & Siapkas; Byrnes [coatings](coatings.md)).
`Coherent` sums them as fields, `t₁t₂e^{iδ}/(1 − r₁′r₂′e^{2iδ})` relative to
free space across the plate, `δ` from its thickness, so that a tilted plate
moves the beam across by its lateral shift and delays it by its phase; the
faces' smooth amplitudes are tabulated over the angle and the plate's phase
taken at each plane wave's own. The plate stays zero-thick in the scene's
geometry, so the field behind it is the plate's relative to free space
across the same thickness, and the report says so. A body that is not such a plate is still
traced face by face, both polarization components taking the tracer's
unpolarized transmittance.

**Two polarization components.** A chain that meets an element telling
polarizations apart (a polarizer, a retarder, an LCoS, or a flat coating or a
plate met obliquely) carries two fields, the components along its beam
frame's `x` and `y`: a Jones vector at every sample [26]. At each element the
pair is turned into that element's own s and p (`ŝ` normal to its plane of
incidence, `p̂ = ŝ × d` on either side, turned with the beam so that its part
along the surface is kept, the chart the amplitudes are written in), each
component carried through the element as a field of that polarization alone,
and the pair turned into the outgoing beam's frame. Square to the beam, the
element's s and p are the beam's own axes. A polarizer or a retarder then
applies its Jones matrix there, as the ray tracer applies it to a ray along
the chief ray: the element's front face, its layer on its own axis (its local
`+X` turned `axis_deg` towards `+Y`, projected across the direction inside
its glass, so that crossed polarizers leak off axis), its back face; one
matrix for the whole field, which the report names. A source that declares
its polarization (`Source3.polarization`, in the source's frame:
[polarization](polarization.md)) launches one coherent pair of components in
that state; an unpolarized source is two launches, along `x` and along `y`,
mutually incoherent and half its power each, added in power like spectral
lines [22], whatever the basis: the sum is the same in any (R15 and finding 4
of the expert review: light polarized along a half-wave plate's axis keeps
the Mach–Zehnder's fringes at visibility `cos 2θ`, where unpolarized light
loses them). A grating that declares its TM fractions (`tm`) carries the pair
in its grooves' basis, the component along them taking the TE fractions and
the one across them the TM ones, as the f64 tracer does; how a conical mount
mixes the two is left out, and the report names it. A crystal retarder
delays both its waves by its fast wave's slab against free space,
`k₀d(√(n_f² − n₀² sin²θ) − n₀ cos θ)`, on top of its retardance, as a plate
does (finding 5: the retardance alone does not say where an interferometer's
fringes sit); an ideal retarder has its retardance alone, and the report
says so. At an LCoS, the component the chain does not follow is reflected as
its own beam is, on a copy of the job's state, and what the device absorbs of
it is measured on it (Q14), not taken from the followed beam's share. A
coating met by a field of two components reports how far its s and p
amplitudes vary over the field's directions (Q13). Elements in different planes of incidence are so carried
through in turn, the s of one becoming a mix of the next one's s and p. Every
plane wave of a planar step takes its own angle's amplitude for its
polarization; that the plane waves off the plane of incidence see s and p
slightly rotated is left out, and the report names it. A chain whose elements
cannot tell polarizations apart carries one field standing for unpolarized
light.

**Interferometers: arms that meet.** Where splitter arms recombine, the
axial ray reaches the detector by several chains, each carrying the same
launched field along its own arm. Arms that meet at a flat element and leave
it as one beam (a Mach–Zehnder's second splitter, a Michelson's splitter on
the way back) **join there**: two chains that reach one flat element by
different paths, leave it from one point along one direction (within a
thousandth of the window, and 1e-6 in direction) and meet the same events
from there on are one field from that element on. The later chain runs up
to and through the element; its field is read onto the other's plane and
grid by the spectrum rotated onto it (`asm_tilted` [6]: the planes may lie a
little apart along the beam or across it, and their frames be turned about
it), its two components turned onto the other's axes, and the two are added.
The events ahead then run once, on the joint field, so an aperture behind a
recombining splitter clips the arms' sum (R04): in the Mach–Zehnder's X
port a disc takes the same share of what reaches it at every phase, and
nothing when the port is dark. Joins are taken longest shared path first, so
arms that meet early are one field before they meet others. Arms that meet
only on the detector (a Michelson with a tilted mirror) are added there, on
the detector's own grid, `s × s` points a pixel, before the power is formed:
the one readout every arm gets (above), so a second arm that carries almost
nothing changes the first's pixels by their cross term alone, of order the
root of its power, and nothing at all once it carries nothing (R12). Either
way it is the coherent sum of paths the scattering-network formulation asks
for (`T = C(I − SP)⁻¹SB`, here a network without feedback, so a finite sum
of paths). It needs three things the single chain did not:

- **One phase clock.** The samples carry the whole phase (above), and every
  arm starts from the same launched field, so the arms' fields on the same
  points of the face add as they are.
- **Unitary splitters.** A lossless splitter must send all its power out of
  its two ports whatever the phase between its inputs. Real amplitudes `√R`,
  `√T` on both arms do not: a Mach–Zehnder's two outputs would carry
  `1 + cos Δφ` of the input between them. The splitter is given the signs of a
  dielectric interface's two-port, the Stokes relations [23], [24]: reflection
  `−√R` from its front and `+√R` from its back, transmission `√T` both ways; a
  perfect conductor is that interface's limit, so a mirror reflects `−√R`.
  These are conventions where a real coating would fix the phases; they move
  fringes, never the power. The Mach–Zehnder test holds the sum to 1 at every
  phase.
- **A ledger for a network.** Each arm's own losses are counted, an event the
  arms share once (they carry one field until they part), each arm's readout
  onto the face with it. Where arms part at one element with one field (a
  splitter's two ports, a grating's two orders), the element's absorption is
  booked once and the shares no arm carries on once, `input·(1 − Σ shares)`
  less that absorption: a splitter whose two ports both reach the detector
  books no `other_arm`, and a grating whose two orders do books `1 − η − η₀`
  of what met it, not `(1 − η) + (1 − η₀)`. Where arms join, each keeps what
  it booked at the element, the reading of the joining arm's field books its
  own resampling, and their interference, the two arms' powers less the
  joint field's, goes once to the port they leave unread (`other_arm` for a
  splitter): for a unitary splitter that port then holds
  `(R + T)(P_a + P_b) − P` in all, which the Mach–Zehnder's other port, read
  on its own detector, matches to 3e-10. A splitter that is not unitary
  shows there as a negative `other_arm`, which the ledger's checks report.
  The light that leaves by the ports no arm reads where arms meet only on
  the detector is the remainder, `other_ports`, named and never a check; the
  checks (below) report it when it comes out negative.

The irradiance at a point of the face follows the spec's `DetectorModel`.
`NormalFlux`, the default, is the scalar flux through the face,
`Σ_c |U_c|² cos θ_c + Re Σ_{c≠c′} U_c U_c′* (cos θ_c + cos θ_c′)/2`: each
arm's own power times its obliquity, so that one arm alone reads as it would
by itself, and the cross terms with the mean obliquity, exact for coplanar
TE plane waves. `ParaxialArea` is `|Σ_c U_c|²`, every obliquity 1. Two arms
crossing at an angle make fringes of period `λ/|t₁ − t₂|`, `t` the
directions' components along the face; their carriers' fringe is integrated
over each readout cell exactly, so two plane waves give the rectangular
pixel's sinc factors to round-off (R13), and a period under two of the
readout's points, where the envelopes' part of the cross term is no longer
resolved, is flagged `Undersampled`. Arms of two components are added
component by component on the face's own axes (R14): an arm oblique to the
face has its transverse axes turned onto the face by the rotation that takes
its direction onto the face's normal, so the component across the plane of
incidence stays, the one in it lands along the face's in-plane direction and
each arm keeps its power (a projection would drop the part of the field along
the normal); the vector cross terms between arms of different directions are
not modelled, which the report names. So orthogonally polarized arms do not
interfere [27]: a half-wave plate in one arm of a Mach–Zehnder takes its
fringes away, and an analyzer behind it brings them back, on a detector
rolled about its normal or tilted about its own axis as on one square to the
beam. Where the chains carry one field, the arms' polarizations are taken as
the same and the fringes' visibility is an upper bound (named in the report).
An arm read out at a tilt goes through the tilted-plane spectrum, whose
reading between spectral samples (gridding, [30], [31]) is accurate to about
1e-9 of the field.

**Several lines.** A source of spectral lines is a set of mutually incoherent
modes [22]: each line runs coherently through every chain at its own
wavelength, its chains traced at that wavelength, with its share of the
power, and the lines add in power. So two lines' fringes wash out where they
fall in antiphase, which the two-line test holds. The field shown is the
strongest line's.

**Refused, by name:** a measured table (it gives R and T without their
phases), any response with no wave model, an aperture inside a face's slab or
a plate, a continuous spectrum (give the source lines: a sampling of it is the source's
to choose), and more than 16 lines. A refusal
is an error naming what was refused; nothing is skipped and nothing is run at
a wavelength the source did not ask for.

**In steps.** `WaveJob::new` plans the chain and launches the beam;
`WaveJob::step(n)` carries the field through `n` more events (the free space
up to each and the element there); `WaveJob::finish` reads the detector. A
worker can drop a stale job between steps. `wave_path` is the job run in one
step, and slicing changes no bit. The job keeps its own copy of the scene and
answers for the revision it started at.

**A recorded plan, and the path as one linear operator** (R09 and finding 3 of
the expert review). The path chooses from the field it meets: which
propagator a step runs, the window a field is carried onto and whether a step
is run again wider, the sphere a field is read about, which of a modulator's
orders are carried apart and which have emptied, which polarization
components are lit, whether a device's mask is band-limited, the rotated
spectrum's reach and the readout's band. As it runs it is therefore not one
linear operator: a sum of two fields of different widths is carried on
windows neither gets alone. Every such choice goes through one recorder
(`oc_wave::plan`), and `WavePathResult::plan` holds them in the order made,
with the path they were made on (the spec and every chain's events). A job
`replaying(&plan)` makes none of its own and is then one fixed linear
operator on the field it launches; `with_launch(&field)` launches any field
on the launch window instead of the spec's beam. A plan recorded on another
path is refused, and so is a run that asks for a choice the plan does not
hold there, both by name. A replayed plan suits the field that recorded it:
another field is carried exactly linearly but on that field's windows, which
need not suit it (below, the sum's own run reads its brightest pixel 4.1%
apart from the replayed one, its focus window sized for the sum rather than
for the narrower field that recorded the plan).

**Off-centre windows through a turn.** A window off the chief ray (a re-grid
centres its window on the beam) is not where it was once an element turns
the beam: a mirror turns it over, and a DMD met at 24° turns it over and
stretches it. The hand-over and a modulator's take-up centre the outgoing
window on the image of the incoming window's centre. Before, they kept the
incoming coordinates: found while measuring the plan above, a DMD's take-up
of a window 0.06 mm off the chief ray lost 8.2% of the power as
`resampling`, and a fold mirror's 7.2e-5 off its window's edge.

**Where the beam is described.** A source declares the coherent beam it
launches as its `coherence` (`CoherentBeam::Gaussian { waist_radius_mm,
waist_offset_mm }` or `TopHat { radius_mm }`), saved with the document;
`WavePathSpec::for_source` builds a spec from it, with the source's flux as the
beam's power, and refuses a source that declares none. The rays the source
emits are unaffected. A spec can also carry any beam directly.

### In the app

The source panel's **coherent beam** gives the first source its beam,
Gaussian or top-hat, and sets the field's window and its samples (128² to
1024²). The detector dock's **wave** button runs the path from that source to
the selected detector, or the first rectangular one, in a worker of its own:
an event per step, and a job whose scene has changed dropped at the next step.
The dock shows the power on each detector pixel, linear or on six decades, or
the arriving field across the beam, its phase as hue and its amplitude as
brightness. The numbers line gives the detected share and how many flags the
report raised; its tooltip holds the whole ledger, every step and every
warning. The status line says whether the picture is this scene's, the
previous scene's while the new one runs, or why the path was refused.

The path runs when an edit settles; while a handle is dragged the last result
stays up, dimmed. The plan's coarse field during a drag is not done.

### Wave-path validation

These are acceptance case C53, in `crates/oc-wave/tests/path.rs`.

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `a_gaussian_through_empty_space_spreads_as_kogelnik_and_li_say` | Kogelnik–Li after 300 mm of scene [13] | 1e-5 | radius 3.5e-7; field 2.3e-7 of peak |
| `a_thin_lens_in_the_scene_focuses_like_self` | Self's waist [14] | 1e-4 | 3.6e-8 |
| `a_fold_mirror_hands_on_the_mirror_image_exactly` | image theory [1]: folded path = straight path | 1e-9 | 6.1e-11 of peak |
| `an_aperture_on_the_path_is_found_although_the_trace_logs_no_event` | a Gaussian through a disc passes `1 − exp(−2a²/w²)` [13] | 1e-4 | 3.2e-5 |
| `a_4f_system_images_at_minus_f2_over_f1_and_its_iris_cuts_the_spectrum` | the image waist `w₀f₂/f₁` [1]; a Fourier-plane iris passes `1 − exp(−2r²/w_F²)`, `w_F = λf₁/(πw₀)` [13]; the ledger | 1e-4; 2e-3; 1e-6 | within 1e-4; 1.7e-5 (1.6e-3 before the window was fitted to the focus); 1.1e-13 |
| `wave_and_ray_agree_on_where_the_beam_lands` | the chief ray's landing point on a tilted fold | 0.05 pixel | 3.2e-4 mm (0.003 pixel) |
| `wave_power_agrees_with_rays_when_the_fresnel_number_is_large` | the f64 ray trace of the same scene, a uniform beam through a disc at `N_F` = 1264 | 1% | 0.12% |
| `a_dmd_in_the_scene_focuses_its_orders_f_lambda_over_p_apart` | orders `λ/p` apart in direction cosine [10], so `fλ/p` apart at a lens's focus | 1% | 1.000; resampling 1.3e-4, `numerical` 3.6e-13 |
| `slicing_a_wave_job_changes_no_bit` | the same path run in one step | bit-identical | identical |
| `a_replayed_plan_makes_the_path_one_linear_operator` | R09 of the expert review: two Gaussians of different waists and centres (60 and 100 µm), complex `a`, `b`, through a thin lens and an open stop at its focus (a re-grid there and back), a DMD (its orders carried apart) and a singlet's two traced faces, under the plan the first recorded: `T(aU + bV)` against `aT(U) + bT(V)`; a plan recorded on another path refused | 1e-12 of `\|T(U)\|` | 1.0e-15; 66 choices (7 windows, 17 propagators, 23 spheres, the orders, 2 readout bands); free, V's run makes 47 of them otherwise, and the sum's own run reads its brightest pixel 4.1% apart from the replayed one |
| `an_off_centre_window_leaves_a_mirror_and_a_dmd_where_its_light_goes` | a window centred 30 µm off the chief ray in the plane of incidence, across it, and on it, through a fold mirror and through a DMD lit at 24° | mirror: off its edge 2e-6 of the launched, resampling 1e-9; DMD: resampling 1e-3 | mirror: 9.2e-7, 7.7e-7, 2.2e-7 off its edge, resampling 1.2e-12; DMD: 3.7e-4 on and off the chief ray (8.2e-2 before, at 60 µm) |
| `a_glass_plate_shifts_a_gaussian_as_free_space_of_t_over_n` | a plate of thickness `t`, index `n` is free space `t/n` to a Gaussian [13]; faces anti-reflected, one closed-form step | 1e-5 | 1.9e-7 (implied index within 1.8e-5; face by face it was 2.4e-6 and 2.3e-4) |
| `a_gaussian_through_a_singlet_waists_where_self_predicts` | Self [14] with the singlet's focal length and principal planes traced by the f64 tracer (an N-BK7 equiconvex lens, R ±50 mm, 4 mm) | 1e-3 | waist position 2.5e-4 of its Rayleigh range (2.9 µm); radius 5.1e-5 |
| `a_tilted_plate_displaces_the_beam_where_the_rays_say` | the chief ray's landing point behind a plate at 30° | 0.05 pixel | 5.2e-5 mm |
| `a_singlet_psf_matches_optiland_huygens_psf` | Optiland's `HuygensPSF` of the same singlet at f/10, at its paraxial focus (oracle, below) | Strehl ±0.01; rms 1e-2 of the ideal peak | Strehl 0.8623 against 0.8642; rms 2.8e-4 |
| `stronger_na_and_off_axis_singlets_against_optiland_huygens_psf` | V02 of the expert review: Optiland's `HuygensPSF` (fixtures of `huygens_singlet.py --v02`) of the singlet at f/10 two degrees off axis; and at f/4 on and off axis on a 2048-sample window, whose 7.6 µm samples cannot hold the converging sphere (218 cycles/mm at the edge, Nyquist 66/mm) | f/10 at 2°: Strehl ±2%, rms 2e-3 of the oracle's peak, sums ±1%; f/4: the report must say the field after the face is undersampled | f/10 at 2°: 0.6120 against 0.6183, rms 6.1e-4, sums 1.002; f/4: flagged (6.6 rad between samples), the PSF's centre within 2% (0.0138 against 0.0141, at 2° 0.0108 against 0.0101) but the halo 5.6 times too faint (sums over the grid 0.18) |
| `an_f2_singlet_against_optiland_huygens_psf_on_4096_samples` (ignored by default: minutes and gigabytes) | V02 at f/2: a singlet a tenth the size (R ±5 mm, 1.6 mm thick) on a 3.1 mm window of 4096 samples, which holds its sphere (417 cycles/mm, Nyquist 658/mm), on and off axis | Strehl ±10%, rms 1e-2 of the oracle's peak, sums ±3% | on the axis 0.01151 against 0.01130, rms 2.0e-3, sums 1.015; at 2° 0.00877 against 0.00952, rms 5.0e-3, sums 1.009 |
| `a_tilted_detector_reads_the_beam_on_its_own_plane` | Kogelnik–Li [13] on a face tilted 30°, times the obliquity: each pixel's power against the 8 × 8 sub-pixel mean of the law, and the power caught | 2e-3 of the peak; power on the face 3e-3 | 8.4e-4; 1.8e-8 (−1.9e-3 under the bilinear spectrum reading before E3) |
| `an_aperture_whose_outer_extent_the_beam_passes_beyond_leaves_it_alone` | the tracer's aperture: light beyond the outer extent passes; against an open aperture at the same plane | bit-identical | identical |
| `an_annular_stop_whose_centre_blocks_the_axial_ray_lets_the_arago_spot_through` | R18 of the expert review: a top-hat beam 1 µm before an annulus 0.1 < r < 0.3 mm, whose centre stops the axial ray; on the axis 15, 25 and 40 mm behind it, Heurtley's exact field [5] taken for each edge by Babinet, `(z/R_in) e^{ikR_in} − (z/R_out) e^{ikR_out}`; the report names the stopped axial ray | 1e-2 of `max(|U/U₀|², 1)` | 1.8e-4, 3.5e-3, 3.5e-3 on 1.55, 3.97, 3.75 |
| `a_fold_mirror_smaller_than_the_beam_reflects_only_what_meets_it` | a 0.6 mm disc at 45° in a 1 mm Gaussian [13]: the power off its projected ellipse, by quadrature, passes beside it | 1e-4 | 1.5e-5 |
| `a_thin_lens_smaller_than_the_beam_lets_the_rest_pass_beside_it` | 10 µm behind a 0.5 mm thin lens: the lens's phase inside, free space's field outside, away from the edge | 2e-3 | 1.0e-3 inside; 3.0e-5 outside |
| `a_traced_ray_leaves_a_thin_lens_along_the_wave_s_phase_gradient` | Q22 of the expert review: the wave's lens phase (`apply_thin_lens`, [1]) read by a central difference on a plane wave, against the f64 tracer's ray at the same point and incidence; f = 20 and −35 mm, three points | 1e-9 in direction cosine | 1.8e-11 or better |
| `a_lens_rim_smaller_than_the_beam_stops_and_diffracts_as_an_iris_against_it` | a 1 mm iris 1.5 µm before a lens too large to clip; the lens rim itself, with exact cover | power 3e-4; spot 2e-3 of peak | 1.1e-4 (the back face's rim stops 9e-4 of what reaches it, which is real); 3.5e-4 |
| `an_lcos_on_the_path_sends_half_of_unpolarized_light_into_each_beam` | half of what the device meets in the unmodulated beam [20]; `sinc²(1/16)` of the modulated half in the first order [21] | 2e-3; 3e-3 | −5.9e-7; −2.0e-3 |
| `a_glass_plate_delays_the_beam_by_its_optical_path` | `k₀(n − 1)t` less the Gouy change of the effective length [13] | 1e-4 rad | 8.9e-9 rad (face by face, 2.6e-6) |
| `a_mach_zehnder_sends_all_the_power_out_of_its_two_ports_whatever_the_path_difference` | a lossless network is unitary: the two ports carry the input between them, `(1 ∓ cos Δφ)/2` each, at Δφ = 0, π/2, π, 3π/2 [23], [24] | sum 1e-3; share 2e-3 | sum 1.000; share within 1.0e-7; the network ledger's remainder 2.2e-14 |
| `a_michelson_with_a_tilted_mirror_makes_fringes_of_period_lambda_over_sin_2_alpha` | two arms crossing at 2α make fringes `λ/sin 2α` apart [1]; read from the power's spectrum across the face | 2e-3 | −7.1e-5 (the beams' curvature accounts for −3.5e-5) |
| `moving_a_michelson_mirror_by_d_shifts_its_fringes_by_two_k_d` | four steps of λ/8 are four of π/2: `P₀ + P₂ = P₁ + P₃`; visibility 1; equal arms dark [1] | 1e-3; 0.995 | −3.1e-8; 1.000 |
| `two_lines_wash_a_michelson_s_fringes_out_where_their_beats_cancel` | two incoherent lines [22]: visibility `\|cos(2πΔ(1/λ₁ − 1/λ₂))\|`, 1 at equal arms, 0 where the beats cancel | 0.99; 2e-2 | 0.9996; 1.6e-3 |
| `a_continuous_source_is_refused_and_lines_add_in_power` | the lines' own runs, weighted; a continuum refused by name | 1e-12 | 1.8e-15 |
| `an_lcos_whose_two_polarization_beams_both_reach_the_detector_adds_them_in_power` | the two beams are orthogonally polarized [20] and add in power [27]: the detector reads the same whatever uniform phase the device puts on one; the device meets `erf(√2a/w)²` [13] | 1e-9; 1e-3 | 0 (identical); −2.7e-4 |
| `a_thin_plate_is_an_etalon_transparent_at_seven_pi_and_dimmest_at_seven_and_a_half` | the closed-form slab at δ = 7π (`T = 1`, `t = (−1)⁷`) and 7.5π (`(1 − r²)²/(1 + r²)²`) [16] | 1e-6 | 6e-14 and 4e-14; the field 3.3e-7 from −e^{−ik₀L} |
| `an_oblique_etalon_transmits_s_and_p_as_their_own_closed_forms` | at 31°, δ = 7.5π: unpolarized light transmits `(T_s + T_p)/2` [16], [22] | 1e-5 | 1.3e-7 |
| `a_millimetre_plate_s_transmission_follows_its_round_trip_phase` | the closed form at δ = k₀nL ≈ 15 000 rad [16] | 1e-4 | −2.3e-5 (the beam's own angular spread, its plane waves' δ 5e-4 rad apart) |
| `a_coated_surface_reflects_and_transmits_as_its_single_layer_s_closed_form` | a quarter-wave MgF₂ layer on crown glass, by the single-layer formula computed apart from the transfer matrix [16]: `(R_s + R_p)/2` at 45°, `(1 − R_f)(1 − R_b)` through the plate | 1e-6 | 1.6e-8; 3.1e-11 |
| `a_coated_plate_takes_its_round_trips_as_it_declares` | R21 of the expert review and finding 7: a coated plate tilted 45°; summed as fields behind an ideal AR back face, 1 mm of glass moves the beam across by `d sin(θ₀ − θ_g)/cos θ_g` and delays the chief plane wave by `k₀d(n cos θ_g − cos θ₀)` [16]; through 10 µm with a bare back face each mode against its reference: the whole sandwich's transfer matrix (coherent), the ray side's incoherent plate, `T₁T₂` (first pass) | 1e-3, relative; 1e-5 rad (of 6339); 2e-4, 1e-5, 1e-5 | 8.0e-8; −1.5e-6 rad; −7.4e-6, −1.1e-7, −1.1e-7 |
| `a_plate_s_kept_round_trips_are_the_beams_summed_one_by_one` (a unit test of `planar.rs`) | R20: a plate of index 4 tilted 30°, 7 round trips kept: against the beams summed one by one; the first round trip's walk against `2d tan θ_g cos θ₀`; what is booked beyond the window, the tail's own power plus its interference with the kept round trips, the latter within twice the product of their norms | 1e-12; 1e-3; 1e-12 of the received | 2.4e-16; within 1e-3; holds |
| `a_thin_low_index_gap_met_past_its_critical_angle_is_refused` (unit test) | a gap of index 1 in glass of 1.5 met at 45°: 1 µm thick (3.5 decay lengths), refused by name; 10 µm (35), run, the leak it leaves out named | refused; named | refused; 34.7 decay lengths |
| `a_coated_plate_s_round_trips_meet_their_three_references` (unit test) | the coated plate's amplitudes at 0°, 30°, 60°, s and p, film side and glass side: coherent against the whole sandwich's transfer matrix (film and 0.5 mm of glass between air and air), incoherent against the ray side's plate, first pass against `T₁T₂` | 1e-10; 1e-12; 1e-14 | 1.8e-12; 7.8e-16; 1.1e-16 |
| `the_ideal_splitter_is_passive_for_any_excitation_of_both_ports` (a unit test of `path.rs`) | Q12: `S†S = (R + T)·I` for the splitter's two-port over R, T in steps of 0.1 with R + T ≤ 1; the same reflection sign from both sides gives an eigenvalue of 2 at R = T = 1/2 | 1e-15 | holds |
| `coatings_whose_planes_of_incidence_differ_reflect_s_of_one_as_p_of_the_other` | two 45° folds in crossed planes: `(R₁ₛR₂ₚ + R₁ₚR₂ₛ)/2`, each `R` by the single-layer formula [16], [26]; a shared plane's `(R₁ₛR₂ₛ + R₁ₚR₂ₚ)/2` is 12% away | 1e-5, relative | −1.1e-6 |
| `two_polarizers_pass_unpolarized_light_as_malus_s_law_says` | Malus [27]: `cos²θ/2` at 0°, 30°, 60°, 90°; crossed sheets of extinction ε pass ε [26] | 1e-9; 1e-6, relative | 1.0e-14; 2.0e-14 |
| `a_retarder_between_crossed_polarizers_passes_half_sin2_2phi_sin2_half_delta` | `sin²2φ·sin²(δ/2)/2` [26], [27] at five (φ, δ); circular light from a quarter-wave plate at 45°, `1/4` through any analyzer | 1e-9 | 9.6e-15; 4.8e-15 |
| `tilted_crossed_polarizers_leak_as_the_ray_tracer_says` | crossed at ±45° to the plane of incidence, met at 40°: the f64 tracer's polarized trace of the chief ray, bare and in glass of 1.5 (one model, two implementations); bare, `(sin²θ/(1 + cos²θ))²/2` from the axes' projection | 1e-6; 1e-9 | 1.8e-14 and 3.5e-14; 1.9e-14 |
| `a_polarized_source_through_polarizers_and_a_quarter_wave_plate_agrees_with_the_ray_tracer` | R15 of the expert review: a source linear at 30° through polarizers at 0° and 60°, `cos²30°·cos²60°` (Malus [27]); right circular (`(1, i)/√2`, e^{−iωt}) through a quarter-wave plate on `x̂`, linear at −45° [26]: an analyzer there passes all, at +45° none, and left circular the reverse; each against the f64 tracer launching the source's own rays | 1e-6 | 3.7e-15; 1.8e-14; 3.4e-32; 1.8e-14 |
| `a_linearly_polarized_source_keeps_the_half_wave_plate_mach_zehnder_s_fringes_at_cos_40` | R15 and finding 4: light polarized along a half-wave plate's axis reference, the plate at 20° in one arm: the X port `(1 − cos 40°·cos Δφ)/2` [27] | 1e-6 | 7.7e-8 |
| `a_crystal_retarder_delays_its_arm_by_its_slab_and_an_ideal_one_does_not` | finding 5: a zero-order quartz half-wave plate in one arm, lit along its fast axis, delays it by `k₀d(n_o − 1)`: the X port `(1 − cos φ)/2` at Δφ = 0; an ideal one delays nothing and says so | 1e-6 | −2.3e-10; 3.4e-16 |
| `a_grating_takes_its_tm_efficiency_across_its_grooves` | η_TE 0.6, η_TM 0.3 into the order: light along the grooves, across them, unpolarized (0.45); each against the f64 tracer | 1e-6 | −2.1e-7, −1.1e-7, −1.6e-7; the tracer the same |
| `a_half_wave_plate_in_one_arm_of_a_mach_zehnder_takes_its_fringes_away_and_a_polarizer_brings_them_back` | Fresnel–Arago [27]: half the launched in the X port at every Δφ; behind an analyzer at β, `1/4 − cos Δφ·cos 2(θ − β)/4` [26], θ = 20°, β = 20°, 50°, 65° | 1e-6 | 4.9e-8 |
| `a_dmd_s_orders_cross_a_singlet_each_about_its_own_direction` | the orders of a DMD whose blaze lies between two of them, through an N-BK7 singlet (R ±8 mm): each crosses each face about its own direction [10], [17]; the chief ray lies 27 mrad from the nearest | steepest departure 0.54 mrad; rms and spectral departure 5.4 mrad; `numerical` 1e-6 | 0.29 mrad; 3.3 and 3.8 mrad (the weak orders folded at the window's band edge; the difference across two samples read 0.36 by folding their steps); 5.9e-9 (resampling 9.9e-3 at the DMD's take-up) |
| `crossed_polarizers_before_a_lens_leave_it_a_dark_field_to_carry` | the chain is geometry: behind crossed polarizers a singlet carries an all-zero field, the ledger closed | 1e-12 | detected 1.7e-33 |
| `a_pinhole_at_a_lens_s_focus_passes_what_the_focused_gaussian_puts_through_it` | a waist on a lens of focal length f has radius λf/(πw₀) at the focal plane (bundle M30, [13]): a pinhole of that radius stops `exp(−2)` | 1e-5 | 7.0e-7 |
| `a_ten_times_beam_expander_widens_the_beam_as_the_abcd_law_says` | the radius at the second lens by the Gaussian ABCD law [13], through a 50 µm focus | 1e-5 | 4.2e-7 |
| `a_grating_order_leaves_with_its_momentum_and_its_phase` | the grating equation [1] (R01 of the expert review): a 20 l/mm grating met at 10°, η = 0.65; the order's field stays centred on the chief ray turned to the order, lands where the traced chief ray does, and takes the phase `e^{−i2πΔ/d}` when the grating moves Δ across its grooves | centroid 1e-4 mm; landing 6e-3 mm; phase 1e-8 | 2.3e-14 mm; 5.2e-4 mm; 1.2e-12 |
| `both_orders_of_a_grating_book_what_neither_carries_once` | the order and the zero order on one detector (R02): `untracked_orders = 1 − η − η₀` once, nothing left to the remainder | 1e-6; 2e-3 | 0.250; detected 0.750, remainder −1.1e-7 |
| `a_window_fitted_off_the_chief_ray_carries_the_field_where_it_is` | a half-blocked 0.1 mm beam whose fitted window lies 55 µm off the chief ray (R03): in free space its centroid on the detector against a fixed window's and against the second-moment law [25]; through a singlet, the power and centroid against a fixed window's | 2e-4 mm; 1e-2 mm (the detector holds 96.7% of the light); 2e-3 mm; 2e-2 (the windows hold different shares of the edge's light) | 7.2e-5 mm; 4.9e-3 mm; 3.1e-4 mm; 9.9e-3 |
| `a_rigidly_moved_scene_reads_the_same_detector` | gauge (R06): a folded path, a Mach–Zehnder and two Michelsons rotated 0.7 rad about (1, 2, 3) and moved, against themselves | 1e-9; 1e-5 of the launched (k·OPL round-off, 1e-7 rad) | 2.3e-12; 9.8e-11, 1.8e-8, 1.4e-7 |
| `a_vanishing_second_arm_changes_the_bright_arm_continuously` | R12 of the expert review: one readout for one arm or several, so a second arm carrying ε of the power changes the bright arm's pixels by their cross term alone, of order `√ε`, and the detected power as much; at ε = 0 one chain runs | 3√ε of the peak; 3√ε | 1.86√ε (1.9e-1, 1.9e-2, 1.9e-4 at ε = 1e-2, 1e-4, 1e-8); detected −9.3e-2, −9.3e-3, −9.3e-5 |
| `a_rolled_detector_reads_the_same_polarized_fringes` | R14: the half-wave-plate Mach–Zehnder on a detector rolled 30° about its normal, and on one tilted 30° about its own axis, against the square detector and the Fresnel–Arago law [26], [27] at Δφ = 0, π/2 (analyzer 50°), π (analyzer 20°) | 1e-6 | square against the law 5.0e-8; rolled 1.1e-11; tilted 7.9e-9 |
| `a_noncoplanar_mach_zehnder_conserves_power_across_its_ports` | R14: arms folded out of the plane reach the second splitter with frames turned about the beam; added on the detector's own axes, both ports carry the launched power, and a rigid roll of the whole scene leaves the Z port's pixels | 2e-3; 1e-5 of the launched | 1.000; 1.0e-10 (the arms joined at the second splitter) |
| `an_aperture_behind_a_recombining_splitter_clips_the_sum_of_the_arms` | R04: the Mach–Zehnder's arms join at the second splitter, so a disc of radius 0.4 mm behind it clips their sum: the same share of the X port at Δφ = π/2 and π, `exp(−2a²/w²)` [13], nothing when the port is dark (Δφ = 0); the Y port's own run against what the X run books to its unread port; a job sliced into single events against the whole | 1e-6; 2e-4 (the disc's edge cells on a 15.6 µm grid); 1e-6 of the launched; 1e-6; bit-identical | 5.4e-8; 9.2e-5; 1.8e-16; 2.8e-10; identical |
| `two_plane_waves_integrate_over_the_pixels_with_the_sinc_factors` (a unit test of `path.rs`) | R13: two plane waves on a 20 × 20 detector, as the arms' carriers and as their envelopes, against the rectangular pixel's closed form `A(w₁ + w₂)[1 + cos(2πΔf x_c)·sinc(Δf d)]`, for the scalar flux (cos 0.9, 0.7; 1, 1) and the paraxial area | 1e-12; the midpoint rule's `(x/sin x − 1)·sinc(Δf d)`, `x = πΔf h`, within 10% | 2.4e-15; 1.19e-2 (the rule's 1.20e-2) |
| `a_focal_spot_on_a_wide_detector_is_read_once_with_the_light_its_window_holds` | the app's default singlet at 0.42 µm, 256 samples over 4 mm, its focal spot on a strip 6 mm wide (24 × 1 pixels): the faces' Fresnel reflection (8.5% at n = 1.528) and the light the undersampled field folds away (the `Undersampled` warning), the spot once, on the two middle pixels, and no `Periodic` warning; before the fix the window there shrank to 64λ and the readout laid the spot's copies beside it | detected 0.88–0.915; left the window < 3%; middle pixels > 0.999 | 0.894; 2.1e-2; 1.000 |
| `an_undersampled_converging_beam_keeps_its_focus` (a unit test of `path.rs`) | a Gaussian (w 0.5 mm) converging 46.7 mm on at 0.42 µm, whose samples' spectrum folds 1% of its power at 256 over 4 mm: the waist its moments predict against Kogelnik–Li's from the complex beam parameter [13], and the same beam on 2048² read by its spectrum | 2% | 5e-5 at 256², 6e-5 at 2048² |
| `the_readout_cap_counts_its_samples_without_wrapping` (a unit test of `path.rs`) | the browser's refusal of the default scene: 64 × 64 pixels at 26 340 points each, whose 32-bit product wraps to 2 813 394 944 and scaled the points to 1017 a pixel; counted in f64 | within the cap | 2²² samples, 32 a pixel |

The transfer's own tests (acceptance case C62, `crates/oc-wave/src/transfer.rs`):

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `a_transfer_about_a_fixed_reference_is_linear_in_the_field` | `T(u₁ + u₂) = T(u₁) + T(u₂)`, two Gaussians tilted ±10 mrad through a lens face; with a sphere fitted to each field (before I6), the same | 1e-12 | 5.7e-17; 0.15 |
| `a_phase_step_of_0_8_pi_a_sample_reads_as_0_8_pi` (a unit test of `transfer.rs`) | R08 of the expert review: a smooth field whose phase steps 0.8π a sample; the adjacent steps read 0.8π, the difference across two samples −0.2π; the fitted wavefront the direction `0.8π/(kΔx)` | 1e-12; 1e-9, relative | 1.4e-14; −8.3e-15 |
| `a_field_with_no_phase_departure_can_depart_in_its_spectrum` (unit test) | R07: `1 + 0.5 cos(qx)` across a slab of free space that dephases its sidebands by π, traced through a face of index 1, against the angular spectrum [1]: `2/3` of its norm away; the phase metric reads no departure, the spectral one `q/3k`, and the sidebands' walk across the slab exceeds a sample | 1e-6; 1e-12; 1e-9, relative | 0.66667; 0; 0 |
| `a_face_that_changes_nothing_leaves_free_space_whatever_slab_holds_it` (unit test) | R10: a curved face of index 1, the same slab split two ways; margins of 1 µm to 1 mm, the error against the angular spectrum all the way `√2` times the note's estimate `k⟨θ²⟩D/2` for a Gaussian | 1e-10; √2 ± 0.03 | 5.4e-12; 1.4142 at every margin (errors 7.4e-4 to 2.9e-2) |
| `a_tilted_beam_through_a_plane_face_matches_the_exact_interface_when_traced_about_its_own_direction` | the plane interface plane wave by plane wave (0.1 mm of air, `√(1 − R)`, 0.3 mm of N-BK7) [1], [16]; traced about the chief ray, the piston `k₀θ²/2·(L_air + L_glass/n)` | 1e-3; 2% | 4.9e-4; 0.1486 rad against 0.1479 |
| `a_tilted_beam_crosses_a_curved_face_where_its_rays_do` | a Gaussian 0.3 mm off the axis and tilted 30 mrad through a face of R 8 mm: its centroid against its traced rays' (geometric field tracing [17]; a check of the map against the rays it is fitted to) | 3 nm | 1.4 nm (the rays' own centroid lies 11 nm from the central ray) |
| `each_lobe_traced_about_its_own_direction_matches_it_traced_alone` | the same lobe about its own direction and about the chief ray's: what one reference for all costs | departure reported | 0.15 relative |

Cases adopted from the wave-optics specification bundle v1.0.0 [29], its
frozen fixtures in the canvas's units (acceptance case C63,
`crates/oc-wave/tests/bundle.rs`):

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `b02_finite_modes_are_carried_by_the_exact_diagonal_transport` | B02: six periodic modes, two components, of a 128 × 96 µm cell on 32 × 24, 550 nm, each multiplied by `e^{id√(k² − \|q\|²)}`, at 3, 7 and 10 mm (the angular spectrum without padding or band limit [1]); 3 + 7 against 10; forward and back | 2e-11 (the bundle's finite-matrix tolerance) | 1.4e-11; 8.5e-12; 5.8e-16 |
| `b02_the_fft_spectrum_is_the_explicit_finite_dft` | B02: the FFT against the explicit DFT at every frequency | 2e-11 | 6.2e-16 |
| `b10_an_aperture_acts_at_its_plane_and_the_order_matters` | B10: `U_after M U_before` (7 mm, the integer circle of 6 steps, 11 mm; quadratic Fresnel on a 240 µm cell) by FFT against the explicit DFT; the misplaced order against the fixture's frozen 0.4329834561 | 2e-11 | 2.6e-15; 6.8e-14 |
| `b10_a_gaussian_through_a_circular_hole_passes_one_minus_e_to_the_minus_half` | B10, M34: `1 − e^{−2a²/w²}` for w = 0.1 mm, a = 0.05 mm, through exact cell coverage: what the mask does not stop | 1e-4, and second order in the spacing | −4.4e-5 at 2 µm, −1.1e-5 at 1 µm (the boundary quadrature of a sampled field; the bundle's 2e-10 is for a quadrature of the continuum) |
| `a_sinusoidal_phase_mask_puts_j_m_squared_into_its_orders` | M14: `e^{iβ cos Gx}` has coefficients `i^m J_m(β)` (Jacobi–Anger, DLMF 10.12.1 [15]); β = 2, orders to ±12, power and phase | 1e-10; 1e-7 rad | 1.8e-15; 1.2e-8 rad (the Gaussian cut at four radii) |
| `two_slits_make_the_far_field_of_m35` | M35: two slits' far field `2a sinc(aq/2) cos(dq/2)` [1], the sampled mask's cell factor `(πfΔx)/sin(πfΔx)` carried | 1e-9 of peak | 8.5e-16 |
| `the_efficiencies_match_du_s_table` (`tests/mie.rs`) | B30: the Mie efficiencies against Du's Table 1 [35], thirteen spheres, `x` 0.055 to 10⁴, `m` 0.75 to 10 + 10i | half a unit of the seventh digit | 0.99 of it at worst (x = 100, m = 1.33 + 10⁻⁵i) |
| `the_exterior_fields_match_an_independent_implementation_at_the_bundle_s_points` (same) | B30: the total exterior `E` and `H` and the efficiencies against miepython 3.3.0 [37] at the bundle's 36 points, index 1 and 1.5, kR = 10, 100, 1000 | fields 1e-8 of the incident; efficiencies 1e-10 | 1.5e-9; 5.1e-13 |
| `the_series_converges_and_the_fields_obey_maxwell_s_equations` (same) | B30: the fields at 1.2R with Wiscombe's terms + 16 against + 32 [36]; Faraday's law `∇ × E = iH` (these units) by central differences | 1e-12; 1e-6 | 0 to 3.5e-13; 1.9e-9 |
| `no_contrast_scatters_nothing_and_a_lossless_sphere_extinguishes_what_it_scatters` (unit test) | B30: every coefficient zero at m = 1; `Re a_n = \|a_n\|²` and `Q_ext = Q_sca` for m = 1.5 [34] | 1e-12; 2e-10; 1e-10 | 0; 7.7e-15; 1e-14 |
| `a_sphere_narrower_than_the_beam_is_refused_by_name` (`tests/path.rs`) | B40: a sphere of index 1.5, kR = 1000 and 100 at 550 nm, under a top-hat twice its radius: refused, the reason named | refused | refused |

The propagation tests of W8 (acceptance case C57, `crates/oc-wave/tests/propagate.rs`):

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `the_zoomed_angular_spectrum_lands_on_the_angular_spectrum_on_its_own_grid` | the FFT angular spectrum on the same points [1] | 1e-10 | 2.7e-14 of peak |
| `an_off_axis_window_holds_the_field_the_large_window_does` | a beam walked 0.6 mm sideways, against the FFT angular spectrum on a window four times as wide [7] | 1e-6 | 1.9e-12 of peak |
| `a_tilted_plane_holds_the_field_its_plane_waves_sum_to` | a plane tilted 20°, 2 mm along, against the plane waves of the same spectrum summed at its points [6], [30], [31]; with no tilt, against `asm_zoom`; rolled about its normal, against the field itself | 1e-7; 1e-10; 1e-9 | 4.1e-10 (the gridding kernel's aliasing, padded 2×); 2.5e-11; rolled 30° about its normal, against the Gaussian at the rolled points, 2.5e-11 |
| `the_rotated_spectrum_converges_over_tilt_and_bandwidth` | Q8: the rotated spectrum against the plane waves summed directly on planes tilted 0°, 20°, 40° and 60°, for beams of 0.1 and 0.025 mm [6], [30], [31] | 1e-7 of the peak | 2.2e-10 to 6.7e-10 |
| `the_rotated_spectrum_books_what_the_plane_cannot_hold_by_cause` | Q8: an evanescent grating's share, `ε²/(1 + ε²)`, booked as `evanescent`; a plane at 87° to the beam warned of grazing waves, one at 20° not | 1e-6 | 1.4e-16; warned |
| `an_off_centre_input_window_propagates_where_its_field_is` | a window centred 0.2 mm off the frame origin, with and without a carrier (R03 of the expert review): the zoomed spectrum against the FFT angular spectrum on a wide centred window [1], onto windows centred anywhere; the rotation onto a shifted parallel plane against the zoom | 1e-6; 1e-9 | 2.0e-12; 8.3e-13 |
| `a_translated_frame_and_a_translated_grid_hold_the_same_field` | gauge: one field encoded as a grid at `c` or as a frame moved to `c`, by the zoom and by the rotation onto one plane | 1e-12 | 3.2e-15; 5.2e-16 |
| `the_zoomed_fresnel_and_fraunhofer_transforms_take_the_grid_where_it_is` | the chirp-z Fresnel transform of an off-centre window against the zoomed angular spectrum (padded 8×, for its periodic copies); a shifted field's far field against the centred one's times the shift phase `e^{−i2πf·c}` [1] | 1e-4; 1e-10 | 4.2e-6; 8.3e-16 |
| `a_baked_carrier_propagates_like_an_analytic_one` | gauge (R06): the carrier analytic or folded into the samples, through `Auto` and through the zoom | 1e-10 | 8.2e-13; 2.6e-13 |
| `a_small_field_read_onto_a_wide_window_is_there_once` | a Gaussian 10 µm in radius read at z = 0, and through a plane tilted 20°, onto a window 15 of its padded periods wide: the sampled spectrum's sum repeats the field every period [1]; the window beyond the period must read zero and the field's power cross it once | zero beyond; power 1e-4 | 0 and 0; 1.000000000 and 1.0000058 |

The DMD case shows what the ledger is for. Of the launched power, 73.0%
reaches the detector and 11.7% falls in the dead space between mirrors
(`absorbed_element`): the physical buckets. The numerical ones: 8.1% is what
the mirrors reflect beyond the device grid's band and the band cut before the
take-up, orders the beam grid cannot hold (`spectral_projection`), 3.6e-6 the
sub-cell projection error (`subcell_projection`; with the band-limited mask
the mirrors' edges are in the spectrum, no longer averaged into each cell),
7.1% misses the detector (`left_window`), 1.3e-4 is lost resampling between
grids (`resampling`), and 3.7e-13 is left unaccounted (`numerical`).

## From Python

```python
from optiland_canvas import wave
res = wave.propagate(u, pitch_mm=0.005, wavelength_um=0.5, z_mm=10.0, method="auto")
res.field, res.pitch_mm, res.report["steps"][0]["method"], res.report["ledger"]
zoom = wave.fraunhofer_zoom(u, 0.005, 0.5, 1000.0, out_n=201, out_pitch_mm=0.01)
```

A field is a `numpy.complex128` array indexed `[y, x]`. It crosses into Rust as
raw bytes, interleaved little-endian `f64`, which is numpy's own layout, so
nothing is reordered or rescaled. The report crosses as the same JSON the Rust
side serializes. The module needs numpy and is imported on its own; the rest of
`optiland_canvas` does not need numpy.

## The report

Every call returns a `WaveReport`:

- the step taken, with its method, distance and sampling regime;
- `warnings`: `BandLimited`, `WrapAround`, `OutsideRegime`, `Paraxial`,
  `Undersampled`, and on a wave path `TiltProjection` and `Periodic` (a
  field whose span on a face, its window and its walk to the face, passes
  the period of the padded spectrum read onto it: its light beyond the
  period folds back in; the face beyond the period reads zero);
- `validity` flags;
- named approximations;
- a `WaveLedger`, where every step books what it changed to a cause, in two
  groups;
- `inconsistencies`: what the ledger booked that a passive model cannot
  (below); empty for a sound result.

```text
launched = physical + numerical

physical  = detected + absorbed_aperture + absorbed_element + evanescent
          + other_family + untracked_orders + other_arm + passed_by
          + other_ports
numerical = left_window + band_limit_removed + spectral_projection
          + subcell_projection + resampling + numerical
```

That it sums to what was launched is bookkeeping, not a check: `numerical`
is defined as what no step accounted for. The entries are the evidence. The
**physical** buckets say where the model sends the power: `other_arm`,
`passed_by`, `other_family` and `untracked_orders` are light on paths the
computation does not follow, never absorption. The **numerical** buckets say
what the computation took: a window's crop (`left_window`), a propagator's
band limit, a band cut before a coarser window or a modulator's take-up
(`spectral_projection`), the sub-cell projection error of a sampled edge
(`subcell_projection`), interpolation (`resampling`: 4.2e-4 of the power
through the f/10 singlet, 1.2e-3 on and off the LCoS of its test), and the
remainder `numerical`, which should be round-off: below 1e-10 of the launched
power on most tested paths, 5.9e-9 on the DMD-and-singlet path; the tests
print it.

What *is* checked, on every result (`WaveLedger::check`): no physical bucket
is negative, and no more is detected than was launched, beyond 1e-7 of the
launched power (`LEDGER_TOLERANCE`: a readout's band-limited interpolation
onto the face can gain a few 1e-9). Anything else is named in
`inconsistencies`, which every path test asserts empty, interferometers
included: an arm's readout is an interpolation accurate to about 1e-9 (the
chirp-z sum, or the gridded spectrum rotation on a tilted face), so the
Michelson that read 1.6e-5 more than was launched under the earlier
point-sampled readout now closes.

## Validation

These are acceptance cases C35–C45; the wave path's are [above](#wave-path-validation). The tests are in `crates/oc-wave/tests/`;
run them with `-- --nocapture` to print the measured values.

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `fft2_is_unitary_under_parseval` | Parseval [1] | 1e-13 | round-off |
| `the_centered_transform_of_a_shifted_gaussian_is_a_gaussian_with_the_shift_phase` | `F{e^{−π(x−a)²}} = e^{−πf²}e^{−i2πfa}` [1]: pins the sign and the centring | 1e-12 | passes |
| `every_propagator_conserves_power_when_nothing_is_band_limited` | `\|H\| = 1` [1] | 1e-12 | ASM 2.8e-14, Fresnel TF 1.7e-14, Fraunhofer 5.1e-14; impulse response 2.0e-11 (its sampled chirp is not exactly unitary [3]; gate 1e-6) |
| `asm_forward_then_back_returns_the_field` | reversibility [1] | 1e-12 | 1.4e-13 of peak |
| `rayleigh_sommerfeld_direct_agrees_with_band_limited_asm` | [4] against [2]: two discretizations of one integral (verification) | 1e-4 | 3.9e-12 of peak |
| `the_band_limit_reports_the_power_it_removes` | the ledger closes [2] | 1e-12 | 32.5% removed and booked |
| `the_paraxial_step_s_error_is_the_identity_s_exactly` | `‖(U − U₂)g‖² = Σ 4 sin²(z(κ − κ₂)/2)\|ĝ\|²` between the band-unlimited angular spectrum and the Fresnel transfer function on the same grid, a Gaussian tilted 0.15 rad; a forced Fresnel step's `Paraxial` warning carries the value | 1e-6 | 5.0e-8 |
| `the_regime_indicators_read_the_spectrum_as_the_step_pads_it` | Q7 of the expert review: the band-limit indicator against the field the band-limited step removes, on a 257 × 200 window padded to 540 × 400 [2] | 1e-9, relative | 6.7e-16 |
| `an_evanescent_wave_counts_by_the_difference_of_the_two_transfer_factors` | Q6: the identity with an evanescent grating (period 0.8λ) on a 3 µm Gaussian, the exact and the paraxial propagator on one padded grid, 2 µm apart | 1e-6, relative | 2.1e-12 |
| `a_step_neither_candidate_holds_is_run_but_not_certified` | Q6: a 3 µm slit 50 mm on, both indicators past 1e-2: the step runs with the smaller and is reported not certified | warned | 0.89 and 0.97; warned |
| `voelz_regime_picks_the_transfer_function_below_the_critical_distance` | `Auto` below and beyond `z_c`: beyond it the smaller error, a Gaussian's band limit removing 2.5e-8 against the paraxial step's 2.5e-6 | exact choice | angular spectrum |
| `an_aperture_passes_exactly_its_area_of_a_uniform_field` | exact cell coverage | 1e-12 | passes, all four shapes |
| `voelz_regime_picks_the_transfer_function_below_the_critical_distance` | `z_c = LΔx/λ` [3] | exact | passes |
| `fresnel_integrals_and_bessel_functions_match_the_tables` | DLMF ch. 7 and 10 [15] | 1e-10 | passes |
| `a_square_aperture_far_field_is_sinc_squared` | Fraunhofer sinc² [1] | rms 1e-3 of peak | rms 4.5e-4 (worst 2.2e-3, in the outer lobes, from the sampled aperture) |
| `a_circular_aperture_makes_the_airy_pattern` | `2J₁(v)/v`; encircled energy 0.8378 at the first dark ring [1], [16] | 2e-3; ±1e-3 | 2.8e-4 of peak; 0.8376 |
| `the_on_axis_field_of_a_circular_aperture_matches_heurtley` | the exact on-axis Rayleigh–Sommerfeld field [5], Fresnel numbers 10 to 0.25 | 5e-3 at a/Δx = 256; error ratio > 3 on halving Δx; Richardson-extrapolated 3e-4 | 2.6e-3 → 1.6e-6 across the range; ratio 4.04–4.05 (second order); extrapolated 3.9e-5 (N_F 10) to 2.6e-8 (N_F 0.25). Band-limited ASM 2.2e-3 at N_F 10; `Auto` (impulse response) 9.4e-5 to 5.8e-6 in the far field |
| `the_angular_spectrum_converges_on_exact_fields_as_each_sampling_is_refined` | R11 of the expert review: a Gaussian (w₀ 50 µm) 20 mm on against its continuous angular spectrum summed by quadrature [1], [33]; a circular aperture on its axis against Heurtley [5]; pitch, window, padding and output sampling refined one at a time | each refinement no worse; Gaussian 1e-9 of the peak, output sampling 1e-12; Heurtley second order in the pitch (each halving > 3×) | Gaussian, pitch 32 → 8 µm: 1.3e-3, 8.7e-12, 8.6e-13; window 0.256 → 1.024 mm: 0.62, 2.7e-12, 8.6e-13; pad 1 → 4: 1.1e-5, 2.7e-12, 1.4e-12; output pitch 4 → 1 µm: 4.6e-15 apart. Heurtley, a/Δx 64 → 256: 4.2e-2, 1.1e-2, 3.3e-3; window 4a → 16a: 4.22e-2 to 4.27e-2; pad 1 → 4: 7.8e-2, 4.2e-2, 4.3e-2 |
| `a_square_aperture_fresnel_pattern_matches_the_fresnel_integrals` | Fresnel diffraction of a square through C and S [1], [15] | 1.5e-3 at 201 samples across; ratio > 3 | 3.5e-3 → 8.4e-4 (ratio 4.2), each method in its regime |
| `an_edge_s_subcell_error_and_its_power_beyond_the_band_are_different_things` | R19 and Q3 of the expert review: a 0.2 mm strip, its edges on a cell boundary, a quarter in and at a cell's centre, on 10, 5 and 2.5 µm grids (`edge_budget`, the mask nine times finer): what the area passes, what the samples carry, the sub-cell error, the power beyond the band, the coherent error in the band | the area and `carried + subcell` to 1e-12; no sub-cell error on a boundary, a band cut all the same; each halves with the cell (±0.05, ±0.2); 9 against 27 times finer within 5% | 1e-16 sub-cell, 1.0e-2 beyond the band on a boundary at 10 µm; quarter: 1.9e-2 and 1.0e-2; ratios 2.000, 1.999; coherent error 3.6e-2 → 1.8e-2; 1.1% |
| `a_gaussian_beam_spreads_as_kogelnik_and_li_say` | `w(z)`, the complex field, Gouy phase [13] | 1e-5 | radius 3.5e-7; field 2.3e-7 of peak |
| `a_thin_lens_focuses_a_gaussian_where_self_predicts` | Self's waist position and size [14], cross-checked through the q-parameter | 1e-4 | waist 8.1e-10; wavefront flat to 2.9e-5 rad at w₀″ |
| `a_cosine_grating_reimages_at_the_exact_talbot_distance` | `z_T = λ/(1 − √(1 − λ²/p²))` and the half-period shift at `z_T/2` [12] | 1e-10 (the absolute phase `k·z_T ≈ 1.3e5` rad puts round-off near 1e-11) | 5.1e-12; 2.6e-12 |
| `chirp_z_zoom_matches_the_direct_dft` | the 2D Fourier sum written out [9] | 1e-11 | 4.6e-15 of peak |
| `fraunhofer_zoom_lands_on_the_fft_fraunhofer_on_its_own_grid` | the FFT Fraunhofer step | 1e-11 | 4.3e-14 of peak |
| `the_airy_first_zero_lands_at_1_21967_lambda_z_over_d` | `r₁ = j₁,₁λz/(πD)` [1], [15], [16], located to sub-sample precision through the zoom | 1e-4 at D/Δx = 128; ratio > 3 | 6.6e-5, 1.1e-5, 1.9e-6 at D/Δx = 32, 64, 128 (ratio 5.8) |
| `test_fraunhofer_matches_optiland_scalar_fft_psf_on_its_own_pupil` (Python) | Optiland `ScalarFFTPSF` on its own aberrated pupil (oracle, below) | 1e-10 | 5.4e-16 of peak |
| `test_zoomed_psf_matches_optiland_mmdft` (Python) | Optiland `MMDFTPSF`, pad size 373.7, off the FFT grid (oracle) | 1e-10 | 3.7e-15 of peak |

**What the tests show about the method, not just the code.** The aperture
tests (Heurtley, the Fresnel square) are limited by how the sampled aperture
represents the true one. They are held by their convergence: the error falls
fourfold each time `Δx` halves, and the Richardson extrapolation,
`(4U(Δx/2) − U(Δx))/3`, lands on Heurtley's exact value to `4e-5` or better.
A fixed gate on a single sampling would test only the gate.

## Oracles

`crates/oc-py/tests/test_wave_oracles.py` (acceptance case C45) hands Optiland's
own pupil array for an off-axis field of its Cooke triplet to both engines.
Optiland's `ScalarFFTPSF` and `MMDFTPSF` transform it, and so do the canvas's
Fraunhofer step and chirp-z zoom. The results agree to round-off. That shows the
two index, centre and scale the transform alike. Per
[validation.md](validation.md) it verifies the code; it is not evidence about
a lens.

**Optiland's pupil sign.** Optiland writes the pupil as `A·exp(−i2π·OPD)`, the
opposite sign to the canvas's `exp(+ik·OPL)`. Handing over Optiland's own array
cancels it here.

**A singlet's PSF (acceptance case C55).** `tools/fixtures/huygens_singlet.py`,
run with the clone's interpreter, writes
`crates/oc-wave/tests/fixtures/huygens_singlet.json`: Optiland's `HuygensPSF`
of an equiconvex singlet (R ±50 mm, 4 mm, index 1.5168) at f/10, stop at its
first surface, at its paraxial focus 47.706221 mm behind it, on 65 × 65 points.
The canvas builds the same lens in a scene, sends a uniform beam through an
iris at that surface, traces it face by face, and reads the focus with the
diffracted readout, one point a pixel at Optiland's own points. Both PSFs are
on axis and symmetric, so the two engines' opposite pupil signs cannot show.
The canvas's PSF is normalized by the peak of a perfect spherical wave
filling the exit pupil, `P·π·sin²α/λ²`, with `α` the half-angle the exit
pupil subtends at the paraxial focus. Both are traced: the exit pupil is
where a ray through the stop's centre crosses the axis in image space, 2.712
mm before the back vertex, and its edge is the marginal ray's height there,
2.5204 mm, so `sin α = 0.049928`. The marginal ray's own sine (0.050118) is
not the reference: the spherical aberration tilts that ray by its ray error,
and dividing by it read the PSF 0.76% low, uniformly. The Strehl ratios are
0.8623 (canvas) and 0.8642 (Optiland), 0.22% apart; the PSFs differ by
2.8e-4 rms of the ideal peak, and their sums over the grid by 0.24%. The
step from the lens toward the focus lost 3.0e-3 of the launched power off
the window fitted to the beam's moments, and was run again on one twice as
wide, which loses 1.2e-3 off its edge and beyond its band; on the first
window the canvas read 0.8644, 0.02% from Optiland, with an rms difference
of 3.7e-4. Two
things leave a residual of that order: the canvas's sampling (finer than its
2.9 µm pitch, the Strehl ratio drops by 0.06% at 1.5 µm), and the reference
itself (a uniform spherical cap peaks at `4π sin²(α/2)/λ²` per watt, 0.06%
above `π sin²α/λ²` at f/10).

Two things about the oracle itself. Its value is converged in pupil rays
(0.85039, 0.85029, 0.85027 at 256, 384 and 512 on an even grid). But
`strehl_ratio()` is the largest sample of the image grid, and an even grid has
no sample on the axis: on 64 points it reads 0.8503, on 65 the 0.8642 above.
The fixture uses an odd grid.

**Stronger NA and off axis (V02 of the expert review).** With `--v02` the
same script writes five more fixtures, `huygens_singlet_<tag>.json`: the
singlet at f/10 two degrees off axis, at f/4 on and off axis, and a lens a
tenth the size (R ±5 mm, made 1.6 mm thick so that its pupil fits well inside
where its faces meet) at f/2 on and off axis. Each records the image grid's
centre, which `HuygensPSF` takes as the mean of a six-ray trace, not the chief
ray; away from f/10 the grid holds the PSF's centre at half the diffraction
scale, not the whole blur, and 384 pupil rays. The canvas places its stop a
fortieth of the lens's thickness before the vertex (Optiland's is the vertex
plane), clear of the slab a tilted beam crosses the face in; the 2° beam's
footprint moves 0.14% of its radius. Its Strehl ratio is read as the oracle's,
the grid's largest sample. Measured: f/10 at 2° agrees as on the axis; f/2
agrees on a window of 4096 samples, which holds its converging sphere; f/4 at
full size does not, and the path says its field is undersampled (above, and
What it does not do).

**Which Optiland.** The oracle is Optiland's current development line, not a
release: [github.com/optiland/optiland](https://github.com/optiland/optiland)
`master` at `1e0513902c68` (committed 2026-10-14), cloned 2026-10-19 and
installed editable in its own environment. The test prints the checkout it ran
against. To reproduce:

```sh
git clone https://github.com/optiland/optiland.git optiland-main
python -m venv optiland-main/.venv
optiland-main/.venv/Scripts/python -m pip install -e optiland-main pytest numpy maturin
cd crates/oc-py && VIRTUAL_ENV=<that venv> <that venv>/Scripts/python -m maturin develop --release
<that venv>/Scripts/python -m pytest tests/test_wave_oracles.py -s
```

The coating fixtures keep their own pin, `optiland==0.6.1`
(`tests/requirements-oracles.txt`), because they were generated with it.

**A sphere: the Mie reference** (`oc_wave::mie`; the bundle's B30, V03 of the
expert review). The exact scattering of a plane wave by a homogeneous sphere
[34]: the coefficients `a_n`, `b_n`, the efficiencies, the amplitudes `S₁`,
`S₂`, and the total exterior `E` and `H` at any point outside, with
`e^{−iωt}`, the wave along `+z`, `E` along `+x`, and `H` in units where the
incident `H` is 1 along `+y`. The coefficients are written through
logarithmic derivatives, each by the recurrence in which it is stable
(`D_n(mx)`, `D_n(x)` downward, `ξ_n′/ξ_n` and `ψ_n/ξ_n` upward), so the orders
past `x`, whose coefficients are tiny but multiply Hankel functions that grow
fast near the sphere, are as accurate as the first. The downward start lies
`16 + 8|mx|^{1/3}` orders above `max(N, |mx|)`: its error falls only as
`exp(−(2√2/3)Δ^{3/2}/|mx|^{1/2})` past the turning point, so BHMIE's fixed 15
left `5e-5` in the fields at `kR = 100` and `2.6e-4` in `Q_ext` at `x = 10⁴`.
The series starts at Wiscombe's `⌈x + 4x^{1/3} + 2⌉` terms [36] and grows by
16 until the efficiencies hold, at most eight times, or is refused.

Two independent checks. Du's Table 1 [35]: thirteen spheres, absorbing and
not, `x` from 0.055 to 10⁴, `Q_ext` and `Q_sca` within half a unit of the
table's seventh digit. And miepython 3.3.0 [37], written apart from this one:
`tools/fixtures/mie_reference.py` freezes its efficiencies and its total
exterior `E` and `H` at the bundle's 36 points a case (r/R = 1.2, 2, 5; θ = 15°
to 135°; φ = 0°, 45°, 90°) for index 1 and 1.5 at kR = 10, 100 and 1000, in
`crates/oc-wave/tests/fixtures/mie_b30.json`; the two agree to 1.5e-9 of the
incident field. Run it with an interpreter that has miepython (a virtual
environment of its own), `python tools/fixtures/mie_reference.py`.

**The wave path against Mie (the bundle's B40) is not run.** A plane wave
wider than a sphere puts light both through it and beside it, so a plane
through the sphere holds a field partly in the glass and partly in air,
which the path's field (one medium per field) cannot be; and the sphere's two
faces meet at its equator, leaving no plane between them. The path refuses
the case by name (test below). For a lens narrower than the beam it stops the
light outside the rim, as a mount would; a sphere has no mount, so that
model would not be the sphere's either. The reference waits for a path that
carries the light beside a body as an arm of its own.

## What it does not do

- **Scalar diffraction.** No vector diffraction and no high-NA focusing: the
  field's two polarization components diffract each as a scalar field, and
  the longitudinal component is not carried. A modulator whose mirror or
  electrode is under 14 wavelengths raises a `ScalarLimit` flag: Pommet,
  Moharam and Grann [11] found scalar efficiencies off by more than ±5% for
  their binary gratings below about 14λ (their 8-level case below a period of
  20λ, the error growing with index, depth, levels and incidence). A heuristic,
  not a bound: no pitch above it certifies a device.
- **Oblique elements by projection.** A lens, an aperture or a modulator seen
  obliquely is handed over by projecting the field along the rays
  (`TiltProjection`, reported); only the detector's readout uses the
  tilted-plane spectrum.
- **Lenses face by face.** A face is traced across its sag in the geometric
  limit; diffraction is taken between faces, not inside a face's slab. Its
  rays follow the beam's paraxial wavefront, not the field's own: a field that
  departs from it (aberrations, a hologram's orders inside one lattice cell)
  is traced along it, and the report gives the departure.
- **A converging beam it cannot sample.** The samples carry the field's
  whole phase but for a linear carrier, so a lens's converging sphere must be
  resolved by the window across it: a singlet at f/4 and 12 mm across needs
  some 8192 samples, which the path does not reach; it reports the field
  undersampled and its PSF's halo comes out several times too faint (V02,
  above). No NA claim is made past the cases measured there: f/10 on and two
  degrees off the axis, and f/2 on a lens small enough for its window to
  hold its sphere.
- **A body narrower than the beam.** The light outside a lens's rim is
  stopped there, as by a mount; a body whose two faces meet across the window
  (a sphere) is refused. The Mie comparison of the bundle's B40 is not run
  for that reason.
- **On a wave path, the beam follows the chief ray.** Under a modulator's
  orders model each followed order is a chain of its own; elsewhere steering
  onto a chosen diffraction order is W7. Arms that recombine are added once: a cavity's
  round trips are summed only inside a plane-parallel plate, where they are
  in closed form; a cavity of two separate mirrors is not followed round.
- **One Jones matrix per element.** A polarizer, a retarder or an LCoS's
  director acts on the whole field as on its chief ray, a coating or a plate
  on each plane wave as in the chief ray's plane of incidence, and a lens's
  faces give both components the tracer's unpolarized transmittance.
- **No measured datasets.** Every reference above is a closed-form result or
  another implementation of the same integral. The DMD page compares with
  published device numbers, and says which are simulations.

## Performance

Measured with `cargo run -p oc-wave --release --example bench_wave`, single
thread, on an AMD Ryzen 9 9900X, 2026-10-19:

| Field (padded) | One band-limited angular-spectrum step |
|---|---|
| 128² (256²) | 2.9 ms |
| 256² (512²) | 11.7 ms |
| 512² (1024²) | 52 ms |
| 1024² (2048²) | 209 ms |

In the browser the path runs in a worker; `tests/wave-bench.spec.ts`
(opt-in, `OC_BENCH=1 npx playwright test --project=bench
tests/wave-bench.spec.ts`) times it at each grid size and records the page's
longest task meanwhile.
[`performance.md`](performance.md) holds the transform's own timings and the
cost of bridging the complex type.

## Sources

Bibliographic data checked 2026-10-19 against the publisher, a repository or
an index. Where a detail is still to be confirmed against the printed text, the
entry says so.

1. J. W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H. Freeman/Macmillan Learning (2017), ISBN 978-1-319-11916-4. (Section numbers to confirm against the printed edition.)
2. K. Matsushima, T. Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Opt. Express* 17(22), 19662–19673 (2009), doi:10.1364/OE.17.019662.
3. D. G. Voelz, M. C. Roggemann, "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences," *Appl. Opt.* 48(32), 6132–6142 (2009), doi:10.1364/AO.48.006132.
4. F. Shen, A. Wang, "Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction formula," *Appl. Opt.* 45(6), 1102–1110 (2006), doi:10.1364/AO.45.001102.
5. W. H. Heurtley, "Scalar Rayleigh–Sommerfeld and Kirchhoff diffraction integrals: a comparison of exact evaluations for axial points," *J. Opt. Soc. Am.* 63(8), 1003–1008 (1973). (DOI to confirm.)
6. K. Matsushima, H. Schimmel, F. Wyrowski, "Fast calculation method for optical diffraction on tilted planes by use of the angular spectrum of plane waves," *J. Opt. Soc. Am. A* 20(9), 1755–1762 (2003), doi:10.1364/JOSAA.20.001755 (resolved 2026-10-19).
7. K. Matsushima, "Shifted angular spectrum method for off-axis numerical propagation," *Opt. Express* 18(17), 18453–18463 (2010), doi:10.1364/OE.18.018453 (resolved 2026-10-19).
8. Y. Hu et al., "Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method," *Light Sci. Appl.* 9, 119 (2020), doi:10.1038/s41377-020-00362-z; Comment by Shao & Urbach (2021), doi:10.1038/s41377-020-00447-9.
9. L. R. Rabiner, R. W. Schafer, C. M. Rader, "The chirp z-transform algorithm," *IEEE Trans. Audio Electroacoust.* 17(2), 86–92 (1969), doi:10.1109/TAU.1969.1162034.
10. J. E. Harvey, C. L. Vernold, "Description of diffraction grating behavior in direction cosine space," *Appl. Opt.* 37(34), 8158–8160 (1998), doi:10.1364/AO.37.008158.
11. D. A. Pommet, M. G. Moharam, E. B. Grann, "Limits of scalar diffraction theory for diffractive phase elements," *J. Opt. Soc. Am. A* 11(6), 1827–1834 (1994), doi:10.1364/JOSAA.11.001827.
12. J. Wen, Y. Zhang, M. Xiao, "The Talbot effect: recent advances in classical optics, nonlinear optics, and quantum optics," *Adv. Opt. Photon.* 5, 83–130 (2013).
13. H. Kogelnik, T. Li, "Laser beams and resonators," *Appl. Opt.* 5(10), 1550–1567 (1966). (DOI to confirm.)
14. S. A. Self, "Focusing of spherical Gaussian beams," *Appl. Opt.* 22(5), 658–661 (1983).
15. NIST Digital Library of Mathematical Functions, ch. 7 (error functions and Fresnel integrals) and ch. 10 (Bessel functions), https://dlmf.nist.gov.
16. M. Born, E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press (1999): the Airy pattern; the theory of stratified media, a single film and a plane-parallel plate's multiple-beam transmission. (Section numbers to confirm.)
17. F. Wyrowski, M. Kuhn, "Introduction to field tracing," *J. Mod. Opt.* 58(5–6), 449–466 (2011), doi:10.1080/09500340.2010.532237. (The coupling of rays and fields; ADR 0004.)

18. Optiland, https://github.com/optiland/optiland, `master` at `1e0513902c68948a727394c6226f3580c7261c65` (2026-10-14), cloned 2026-10-19: the oracle for C45.

19. J. E. Harvey et al., "Linear systems formulation of non-paraxial scalar diffraction theory," *Proc. SPIE* 8122, 812205 (2011), doi:10.1117/12.894562. (The obliquity factor that keeps a beam's power through an element it leaves at another angle.)
20. Z. Zhang, Z. You, D. Chu, "Fundamentals of phase-only liquid crystal on silicon (LCOS) devices," *Light Sci. Appl.* 3, e213 (2014), doi:10.1038/lsa.2014.94. (An LCoS modulates the field along its director only.)
21. G. J. Swanson, "Binary optics technology: the theory and design of multi-level diffractive optical elements," MIT Lincoln Laboratory Technical Report 854 (1989), DTIC ADA213404. (An N-level staircase's first order, `sinc²(1/N)`.)
22. E. Wolf, "New theory of partial coherence in the space–frequency domain. Part I: spectra and cross spectra of steady-state sources," *J. Opt. Soc. Am.* 72(3), 343–351 (1982), doi:10.1364/JOSA.72.000343. (Mutually incoherent modes add in power.)
23. G. G. Stokes, "On the perfect blackness of the central spot in Newton's rings, and on the verification of Fresnel's formulae for the intensities of reflected and refracted rays," *Cambridge and Dublin Mathematical Journal* 4, 1–14 (1849). (The Stokes relations: `r' = −r`, `tt' = 1 − r²`.)
24. A. Zeilinger, "General properties of lossless beam splitters in interferometry," *Am. J. Phys.* 49(9), 882–883 (1981), doi:10.1119/1.12387. (A lossless splitter is a unitary two-port. DOI to confirm.)
25. A. E. Siegman, "New developments in laser resonators," *Proc. SPIE* 1224, 2–14 (1990). (The second-moment beam width and its quadratic law in z, for any beam. DOI to confirm.)
26. R. C. Jones, "A new calculus for the treatment of optical systems. I. Description and discussion of the calculus," *J. Opt. Soc. Am.* 31(7), 488–493 (1941), doi:10.1364/JOSA.31.000488. (A polarizer and a retarder as 2×2 matrices on the field's two components.)
27. E. Hecht, *Optics*, 5th ed., Pearson (2017), ch. 8 (polarization): Malus's law; a retarder between crossed polarizers; the Fresnel–Arago laws. (Section numbers to confirm against the edition.)
28. A. E. Siegman, *Lasers*, University Science Books (1986), ch. 15 and 20: ray matrices in reduced form across media of different index, and the ABCD law for a Gaussian beam's `q`. (Chapters to confirm.)
29. Wave-optics specification bundle v1.0.0 (`wave_optics_agent_bundle_v1.zip`, supplied by the project, 2026-10-19): `01_MODEL.md` M14, M34, M35; `03_BENCHMARKS.md` B02, B10; `config/benchmark_cases.json`, frozen fixtures. (A specification, not a published source; its own references are in `04_REFERENCES.md`.)
30. J. I. Jackson, C. H. Meyer, D. G. Nishimura, A. Macovski, "Selection of a convolution function for Fourier inversion using gridding," *IEEE Trans. Med. Imaging* 10(3), 473–478 (1991), doi:10.1109/42.97598. (Gridding: a spectrum read between its samples by a kernel on an oversampled grid, the field divided by the kernel's Fourier dual; the Kaiser–Bessel kernel.)
31. P. J. Beatty, D. G. Nishimura, J. M. Pauly, "Rapid gridding reconstruction with a minimal oversampling ratio," *IEEE Trans. Med. Imaging* 24(6), 799–808 (2005), doi:10.1109/TMI.2005.848376. (The kernel's shape parameter for a width and an oversampling, Eq. (5).)
32. L. Greengard, J.-Y. Lee, "Accelerating the nonuniform fast Fourier transform," *SIAM Rev.* 46(3), 443–454 (2004), doi:10.1137/S003614450343200X. (The same step as the interpolation of a type-2 non-uniform FFT.)
33. M. Abramowitz, I. A. Stegun, *Handbook of Mathematical Functions*, National Bureau of Standards (1964), Table 25.4. (The ten-point Gauss–Legendre nodes and weights the Gaussian's reference quadrature uses.)
34. C. F. Bohren, D. R. Huffman, *Absorption and Scattering of Light by Small Particles* (Wiley, 1983), ch. 4: Eqs. 4.45, 4.50, 4.53, 4.61, 4.62, 4.74; §4.8 (the logarithmic derivative's downward recurrence).
35. H. Du, "Mie-scattering calculation," *Appl. Opt.* 43(9), 1951–1956 (2004), doi:10.1364/AO.43.001951, Table 1 (values as reproduced in the scattnlay test suite, `tests/test_py.py`, retrieved 2026-10-21).
36. W. J. Wiscombe, *Mie Scattering Calculations: Advances in Technique and Fast, Vector-Speed Computer Codes*, NCAR Technical Note NCAR/TN-140+STR (1979; revised 1996). (The number of terms.)
37. S. Prahl, miepython 3.3.0, https://github.com/scottprahl/miepython (installed 2026-10-21 with numpy 2.5.4 and scipy 1.18.1). An independent implementation, used as an oracle for the efficiencies and the near fields.

The FFT is `rustfft` 6.4.1 (MIT OR Apache-2.0).
