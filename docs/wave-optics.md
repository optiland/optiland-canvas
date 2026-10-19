# Scalar wave optics (`oc-wave`)

A complex scalar field sampled on a plane, and the methods that move it from
one plane to another by diffraction. The crate sits beside the ray tracer, not
inside it, and the two meet only through optical path length
([ADR 0003](architecture/adr/0003-scalar-wave-optics-engine.md)). Everything is
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
| `asm_tilted(field, plane, out, pad)` | the field on any plane: tilted, shifted and distant at once, each plane wave read in the target frame, `(u, v, w) = Rᵀ(u′, v′, w′)`, with the Jacobian `\|∂(u, v)/∂(u′, v′)\|` and the phase to the plane's origin | the source spectrum is read between its samples (bilinear): padding samples it finer | [6], [7] |

**The critical distance** `z_c = LΔx/λ`, for a window `L` wide after padding,
divides the two Fresnel methods [3]. Up to `z_c` the angular spectrum's
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
  band-limited angular spectrum makes the smaller error on this very field,
  both known from its spectrum before either runs, and both written into the
  step's note.

The paraxial step's error is known exactly. With `κ = √(k² − |q|²)` and its
paraxial form `κ₂ = k − |q|²/2k`, the exact and the paraxial propagators
differ on a field `g` by

```text
‖(U − U₂)g‖² = Σ_q 4 sin²(z(κ − κ₂)/2) |ĝ(q)|²,   κ − κ₂ = −|q|⁴ / (2k(k + κ)²),
```

which follows from Parseval's theorem because both are diagonal in the plane
waves; the difference is written in its stable form. `paraxial_error` sums it
over the field's own spectrum, its carrier included, so a narrow beam tilted
far from its plane's normal is not mistaken for a paraxial one, as a bound
from the window's corner (the old rule, `πρ⁴/(4λz³) < 0.1` rad [1]) would.
The band limit's error is the field it would remove, `band_limit_error`.
Every Fresnel step, chosen or forced, says its paraxial error in its note,
and above `1e-4` raises the `Paraxial` warning with it. The identity holds
to 5e-8 between the canvas's own two propagators (test below).

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
`f²·|U|²ΔA`. The difference is edge diffraction into angles beyond the grid's
Nyquist band. It is booked as `beyond_nyquist`, separately from the
`absorbed_aperture` of the opaque part.

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
ADR 0003 chose: rays for the layout, fields for the diffraction, joined by
optical path [17].

```rust
let spec = WavePathSpec {
    source, detector,
    beam: BeamProfile::Gaussian { waist_radius_mm: 0.3, waist_offset_mm: 0.0 },
    power: 1.0,
    grid_n: 512, window_mm: 4.0,      // the field across the beam
    device_samples_per_pixel: 4,      // when the beam is laid onto a DMD or SLM
    readout: Readout::Projected,      // or Readout::Diffracted { samples_per_pixel }
};
let result = wave_path(&scene, &spec, revision)?;   // power per detector pixel, the field, the report
// or, from the beam the source itself declares:
let spec = WavePathSpec::for_source(&scene, source, detector, 512, 4.0)?;
```

**How the path is planned.** `extract_sequences3`
([ADR 0002](architecture/adr/0002-sequence-extraction-from-the-trace.md)) gives
each chain of surfaces the chief ray meets on its way to the detector, at
each of the source's lines; there are several where splitter arms recombine
on the detector (below). The tracer logs no event when a ray
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
| grating | its order's momentum along the grooves, amplitude `√η`; the other orders are booked as `untracked_orders` | thin element |
| ideal thin lens | `exp(−ik\|r − c\|²/2f)` about the lens's centre | thin element |
| a flat element's outline | the element acts inside its own shape, the edge weighted by exact cell coverage; light outside passes it by, on the chain where the chain goes straight on (a transmitted arm, a thin lens), off it as `passed_by` where the chain turns (a mirror, a grating's order) | thin element |
| aperture | as the tracer has it: light passes inside the opening, is stopped between the opening and the outer extent, and passes beyond the outer extent (an aperture with none is unbounded); each edge weighted by exact cell coverage, of the shape itself square to the beam and of its projection along the beam otherwise | thin element |
| DMD, LCoS SLM | laid onto the device plane at `device_samples_per_pixel`, reflected by the device model ([DMD](dmd.md), [SLM](slm.md)), taken up across the outgoing ray; at an LCoS, half of what it reflects follows the chain (below) | thin element |
| face of a refractive body, curved mirror or other curved surface | traced across the thin slab that holds the surface where the beam meets it (below), its rim laid on the field with exact cell coverage; between a body's faces the field diffracts inside the glass by the angular spectrum, in the glass's index | the slab is geometric: no diffraction across a surface's sag |
| plane-parallel plate (a body entered and left by two flat, parallel faces, bare or with a residual AR) | one step: the closed-form slab, round trips included, on each plane wave of the field, then the angular spectrum across the planes (below) | exact for a plate of infinite extent, s or p; its rim laid on the field |
| flat coated surface (`Coating`: a coated splitter, a dichroic) | each plane wave times the coating's complex `r` or `t` at its own angle, from the transfer matrix, then handed over like any flat element; a plate-backed coating's plate in series, without round trips, as the coating declares it incoherent | exact for the stack, s or p |
| detector, `Readout::Projected` | power per pixel: each field cell's `\|U\|²ΔA` binned into the pixel its projection lands in, in sub-cells of at most an eighth of a pixel | power conserved to round-off |
| detector, `Readout::Diffracted` | the last free-space step taken straight onto the detector by the zoomed angular spectrum (`asm_zoom`), at `s × s` points a pixel: a focal spot far smaller than the field's own samples is resolved, with no paraxial step | exact scalar propagation; the field repeats a window away when unpadded |

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
sample. A tilted detector read with `Readout::Diffracted`
is not projected: the spectrum is rotated onto its own plane (`asm_tilted`
[6]). Elements on oblique planes still are.

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
copy of the scene holding the element alone, along the arriving field's
wavefront (a sphere and tilt fitted to its phase gradient). Each ray gives
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
they moved across the beam, and how far the arriving field's rays depart
from the fitted sphere.

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
singlet's second face and dimmed its pupil edge. So the field's fitted sphere
and tilt are taken out of the samples first and put back exactly at the
point read; only the slowly varying rest is interpolated. The singlet's
unexplained remainder fell from 3.7% to 0.04% of the launched power. What
interpolation still costs is booked at each step as `resampling`.

**Windows fitted to the beam.** One window for a whole path cannot both hold
a wide beam and resolve the focus it is brought to: a 1 mm beam focused by
f = 100 mm has a 20 µm waist, under two samples of a 6 mm window of 512. So
before each free-space step the path predicts the beam's size at the next
plane from the field's own second moments, its variance `σ²`, its angular
variance `σ_θ²` and their cross moment `C = ⟨(x − x̄)θ⟩` (the local direction
`θ` from the phase gradient, the fitted sphere taken out first so that a
steep wavefront is not understated): under paraxial propagation
`σ²(z) = σ² + 2zC + z²σ_θ²` exactly, whatever the beam's shape (the
second-moment law; A. E. Siegman [25]). When the beam, six rms radii across,
would fill less than a quarter of the window, or on a window already
narrowed more than four fifths of it, the field is carried onto a window
fitted to it, centred on its predicted centroid, with the same sample count
and never wider than the declared window. Narrowing, the zoomed angular
spectrum carries it, exactly, padded enough that its plane waves' walk over
the step stays inside; widening from a narrow window past its critical
distance, the zoomed Fresnel transform, the Fresnel integral as one chirp-z
transform onto the new grid, its paraxial error reported. A pinhole at a
focus is then laid on the focus at the focus's own sampling, and a 10× beam
expander keeps its beam resolved from 0.2 mm through a 50 µm focus to 2 mm
(tests below).

**What the path follows at a modulator.** The chief ray, and so the beam,
leaves a DMD along the reflection off the facet it hits, and an SLM along the
phase gradient there (or specularly, for the unmodulated arm). Of the field
an LCoS meets, the component along its director is modulated and the one
across it is not [20]: the chain carries two components (below) and takes
into its beam the one that beam is made of. Of the other, the device takes
the same share (its fill and reflectance do not tell the beams apart), and
the rest goes into the other beam, booked as `other_family`. Where both beams
reach the detector, each is a chain of its own; they meet orthogonally
polarized and add in power, and the ledger books each beam's share once. The
field keeps
every diffraction order that falls inside its window; orders outside are
booked as `beyond_nyquist` or `left_window`. It does not steer onto a chosen
order: an iris on a Lee hologram's −1 order, off the chief ray, would block
the chief ray itself. The wave path does not follow an order yet; the ray
tracer's orders model (`RayModel::Orders`) traces them as rays.

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
round-trip shift apart; those that would land beyond the window are summed
out of the series and booked as `left_window`, never wrapped back in. The
amplitudes are in the tangential chart and power coordinates of a scattering
two-port: `r = (y₁ − y₂)/(y₁ + y₂)`, `t = 2√(y₁y₂)/(y₁ + y₂)`,
`y_s = n cos θ`, `y_p = n / cos θ`, unitary, with the Stokes signs of the
splitters. A flat coated surface multiplies each plane wave by its coating's
complex reflection or transmission, from `oc_core::thinfilm`'s transfer
matrix put into that chart (its ray-frame `r_p` negated, its transmission
scaled into power coordinates). A body that is not such a plate is still
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
matrix for the whole field, which the report names. An unpolarized source is
two launches, along `x` and along `y`, mutually incoherent and half its power
each, added in power like spectral lines [22], whatever the basis: the sum is
the same in any. Elements in different planes of incidence are so carried
through in turn, the s of one becoming a mix of the next one's s and p. Every
plane wave of a planar step takes its own angle's amplitude for its
polarization; that the plane waves off the plane of incidence see s and p
slightly rotated is left out, and the report names it. A chain whose elements
cannot tell polarizations apart carries one field standing for unpolarized
light.

**Interferometers: arms that meet.** Where splitter arms recombine on the
detector, the axial ray reaches it by several chains. Each carries the same
launched field along its own arm, and the arms' fields are added on the
detector's own grid, `s × s` points a pixel (two for `Readout::Projected`),
before the power is formed. That is the coherent sum of paths the
scattering-network formulation asks for (`T = C(I − SP)⁻¹SB`, here a network
without feedback, so a finite sum of paths). It needs three things the single
chain did not:

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
  arms share once (they carry one field until they part). What a splitter
  sends into another arm is not lost, so a splitter several arms meet books
  no `other_arm`; the light that leaves by the ports no arm reads (a
  recombining splitter's other output, what misses the detector once the arms
  meet) is then the remainder, `other_ports`, named and never a check.

Each arm's field is weighted on the face by the root of its obliquity, so that
one arm alone reads as it would by itself. Two arms crossing at an angle make
fringes of period `λ/|t₁ − t₂|`, `t` the directions' components along the face;
a period under two of the readout's points is flagged `Undersampled`. Arms of
two components are added component by component, so orthogonally polarized
arms do not interfere [27]: a half-wave plate in one arm of a Mach–Zehnder
takes its fringes away, and an analyzer behind it brings them back. Where the
chains carry one field, the arms' polarizations are taken as the same and the
fringes' visibility is an upper bound (named in the report). An arm read out
at a tilt goes through the tilted-plane spectrum, whose reading between
spectral samples is its approximation (5.9e-4 in its own test).

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
| `a_thin_lens_in_the_scene_focuses_like_self` | Self's waist [14] | 1e-4 | 8.1e-10 |
| `a_fold_mirror_hands_on_the_mirror_image_exactly` | image theory [1]: folded path = straight path | 1e-9 | 2.0e-11 of peak |
| `an_aperture_on_the_path_is_found_although_the_trace_logs_no_event` | a Gaussian through a disc passes `1 − exp(−2a²/w²)` [13] | 1e-4 | 3.2e-5 |
| `a_4f_system_images_at_minus_f2_over_f1_and_its_iris_cuts_the_spectrum` | the image waist `w₀f₂/f₁` [1]; a Fourier-plane iris passes `1 − exp(−2r²/w_F²)`, `w_F = λf₁/(πw₀)` [13]; the ledger | 1e-4; 2e-3; 1e-6 | within 1e-4; 1.8e-5 (1.6e-3 before the window was fitted to the focus); 8.7e-14 |
| `wave_and_ray_agree_on_where_the_beam_lands` | the chief ray's landing point on a tilted fold | 0.05 pixel | 3.2e-4 mm (0.003 pixel) |
| `wave_power_agrees_with_rays_when_the_fresnel_number_is_large` | the f64 ray trace of the same scene, a uniform beam through a disc at `N_F` = 1264 | 1% | 0.12% |
| `a_dmd_in_the_scene_focuses_its_orders_f_lambda_over_p_apart` | orders `λ/p` apart in direction cosine [10], so `fλ/p` apart at a lens's focus | 1% | 1.000; resampling 7.7e-6, `numerical` 2.8e-13 |
| `slicing_a_wave_job_changes_no_bit` | the same path run in one step | bit-identical | identical |
| `a_glass_plate_shifts_a_gaussian_as_free_space_of_t_over_n` | a plate of thickness `t`, index `n` is free space `t/n` to a Gaussian [13]; faces anti-reflected, one closed-form step | 1e-5 | 1.9e-7 (implied index within 1.8e-5; face by face it was 2.4e-6 and 2.3e-4) |
| `a_gaussian_through_a_singlet_waists_where_self_predicts` | Self [14] with the singlet's focal length and principal planes traced by the f64 tracer (an N-BK7 equiconvex lens, R ±50 mm, 4 mm) | 1e-3 | waist position 2.5e-4 of its Rayleigh range (2.9 µm); radius 5.1e-5 |
| `a_tilted_plate_displaces_the_beam_where_the_rays_say` | the chief ray's landing point behind a plate at 30° | 0.05 pixel | 5.2e-5 mm |
| `a_singlet_psf_matches_optiland_huygens_psf` | Optiland's `HuygensPSF` of the same singlet at f/10, at its paraxial focus (oracle, below) | Strehl ±0.01; rms 1e-2 of the ideal peak | Strehl 0.8653 against 0.8642; rms 1.2e-4 |
| `a_tilted_detector_reads_the_beam_on_its_own_plane` | Kogelnik–Li [13] at each point of a face tilted 30°, times the obliquity | 2e-3 of the peak; power on the face 2e-3 | 8.5e-4; −1.9e-3 |
| `an_aperture_whose_outer_extent_the_beam_passes_beyond_leaves_it_alone` | the tracer's aperture: light beyond the outer extent passes; against an open aperture at the same plane | bit-identical | identical |
| `a_fold_mirror_smaller_than_the_beam_reflects_only_what_meets_it` | a 0.6 mm disc at 45° in a 1 mm Gaussian [13]: the power off its projected ellipse, by quadrature, passes beside it | 1e-4 | 1.5e-5 |
| `a_thin_lens_smaller_than_the_beam_lets_the_rest_pass_beside_it` | 10 µm behind a 0.5 mm thin lens: the lens's phase inside, free space's field outside, away from the edge | 2e-3 | 1.0e-3 inside; 3.0e-5 outside |
| `a_lens_rim_smaller_than_the_beam_stops_and_diffracts_as_an_iris_against_it` | a 1 mm iris 1.5 µm before a lens too large to clip; the lens rim itself, with exact cover | power 3e-4; spot 2e-3 of peak | 1.1e-4 (the back face's rim stops 9e-4 of what reaches it, which is real); 3.5e-4 |
| `an_lcos_on_the_path_sends_half_of_unpolarized_light_into_each_beam` | half of what the device meets in the unmodulated beam [20]; `sinc²(1/16)` of the modulated half in the first order [21] | 2e-3; 3e-3 | −5.9e-7; −2.0e-3 |
| `a_glass_plate_delays_the_beam_by_its_optical_path` | `k₀(n − 1)t` less the Gouy change of the effective length [13] | 1e-4 rad | 8.9e-9 rad (face by face, 2.6e-6) |
| `a_mach_zehnder_sends_all_the_power_out_of_its_two_ports_whatever_the_path_difference` | a lossless network is unitary: the two ports carry the input between them, `(1 ∓ cos Δφ)/2` each, at Δφ = 0, π/2, π, 3π/2 [23], [24] | sum 1e-3; share 2e-3 | sum 1.000; share within 1e-7 |
| `a_michelson_with_a_tilted_mirror_makes_fringes_of_period_lambda_over_sin_2_alpha` | two arms crossing at 2α make fringes `λ/sin 2α` apart [1]; read from the power's spectrum across the face | 2e-3 | −7.1e-5 (the beams' curvature accounts for −3.5e-5) |
| `moving_a_michelson_mirror_by_d_shifts_its_fringes_by_two_k_d` | four steps of λ/8 are four of π/2: `P₀ + P₂ = P₁ + P₃`; visibility 1; equal arms dark [1] | 1e-3; 0.995 | 2.8e-7; 1.000 |
| `two_lines_wash_a_michelson_s_fringes_out_where_their_beats_cancel` | two incoherent lines [22]: visibility `\|cos(2πΔ(1/λ₁ − 1/λ₂))\|`, 1 at equal arms, 0 where the beats cancel | 0.99; 2e-2 | 0.9996; 1.6e-3 |
| `a_continuous_source_is_refused_and_lines_add_in_power` | the lines' own runs, weighted; a continuum refused by name | 1e-12 | 1.8e-15 |
| `an_lcos_whose_two_polarization_beams_both_reach_the_detector_adds_them_in_power` | the two beams are orthogonally polarized [20] and add in power [27]: the detector reads the same whatever uniform phase the device puts on one; the device meets `erf(√2a/w)²` [13] | 1e-9; 1e-3 | 0 (identical); −2.7e-4 |
| `a_thin_plate_is_an_etalon_transparent_at_seven_pi_and_dimmest_at_seven_and_a_half` | the closed-form slab at δ = 7π (`T = 1`, `t = (−1)⁷`) and 7.5π (`(1 − r²)²/(1 + r²)²`) [16] | 1e-6 | 6e-14 and 4e-14; the field 3.3e-7 from −e^{−ik₀L} |
| `an_oblique_etalon_transmits_s_and_p_as_their_own_closed_forms` | at 31°, δ = 7.5π: unpolarized light transmits `(T_s + T_p)/2` [16], [22] | 1e-5 | 1.3e-7 |
| `a_millimetre_plate_s_transmission_follows_its_round_trip_phase` | the closed form at δ = k₀nL ≈ 15 000 rad [16] | 1e-4 | −2.3e-5 (the beam's own angular spread, its plane waves' δ 5e-4 rad apart) |
| `a_coated_surface_reflects_and_transmits_as_its_single_layer_s_closed_form` | a quarter-wave MgF₂ layer on crown glass, by the single-layer formula computed apart from the transfer matrix [16]: `(R_s + R_p)/2` at 45°, `(1 − R_f)(1 − R_b)` through the plate | 1e-6 | 1.6e-8; 3.1e-11 |
| `coatings_whose_planes_of_incidence_differ_reflect_s_of_one_as_p_of_the_other` | two 45° folds in crossed planes: `(R₁ₛR₂ₚ + R₁ₚR₂ₛ)/2`, each `R` by the single-layer formula [16], [26]; a shared plane's `(R₁ₛR₂ₛ + R₁ₚR₂ₚ)/2` is 12% away | 1e-5, relative | −1.1e-6 |
| `two_polarizers_pass_unpolarized_light_as_malus_s_law_says` | Malus [27]: `cos²θ/2` at 0°, 30°, 60°, 90°; crossed sheets of extinction ε pass ε [26] | 1e-9; 1e-6, relative | 1.0e-14; 2.0e-14 |
| `a_retarder_between_crossed_polarizers_passes_half_sin2_2phi_sin2_half_delta` | `sin²2φ·sin²(δ/2)/2` [26], [27] at five (φ, δ); circular light from a quarter-wave plate at 45°, `1/4` through any analyzer | 1e-9 | 9.6e-15; 4.8e-15 |
| `tilted_crossed_polarizers_leak_as_the_ray_tracer_says` | crossed at ±45° to the plane of incidence, met at 40°: the f64 tracer's polarized trace of the chief ray, bare and in glass of 1.5 (one model, two implementations); bare, `(sin²θ/(1 + cos²θ))²/2` from the axes' projection | 1e-6; 1e-9 | 1.8e-14 and 3.5e-14; 1.9e-14 |
| `a_half_wave_plate_in_one_arm_of_a_mach_zehnder_takes_its_fringes_away_and_a_polarizer_brings_them_back` | Fresnel–Arago [27]: half the launched in the X port at every Δφ; behind an analyzer at β, `1/4 − cos Δφ·cos 2(θ − β)/4` [26], θ = 20°, β = 20°, 50°, 65° | 1e-6 | 4.9e-8 |
| `crossed_polarizers_before_a_lens_leave_it_a_dark_field_to_carry` | the chain is geometry: behind crossed polarizers a singlet carries an all-zero field, the ledger closed | 1e-12 | detected 1.7e-33 |
| `a_pinhole_at_a_lens_s_focus_passes_what_the_focused_gaussian_puts_through_it` | a waist on a lens of focal length f has radius λf/(πw₀) at the focal plane (bundle M30, [13]): a pinhole of that radius stops `exp(−2)` | 1e-5 | 5.7e-7 |
| `a_ten_times_beam_expander_widens_the_beam_as_the_abcd_law_says` | the radius at the second lens by the Gaussian ABCD law [13], through a 50 µm focus | 1e-5 | 3.9e-7 |

The propagation tests of W8 (acceptance case C57, `crates/oc-wave/tests/propagate.rs`):

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `the_zoomed_angular_spectrum_lands_on_the_angular_spectrum_on_its_own_grid` | the FFT angular spectrum on the same points [1] | 1e-10 | 2.7e-14 of peak |
| `an_off_axis_window_holds_the_field_the_large_window_does` | a beam walked 0.6 mm sideways, against the FFT angular spectrum on a window four times as wide [7] | 1e-6 | 1.9e-12 of peak |
| `a_tilted_plane_holds_the_field_its_plane_waves_sum_to` | a plane tilted 20°, 2 mm along, against the plane waves of the same spectrum summed at its points [6]; with no tilt, against `asm_zoom` | 2e-3; 1e-10 | 5.9e-4 (the bilinear reading of the spectrum, padded 4×); 8.3e-14 |

The DMD case shows what the ledger is for. Of the launched power, 66.8%
reaches the detector, 11.7% falls in the dead space between mirrors
(`absorbed_element`), 19.4% goes into orders beyond the beam grid's Nyquist
angle (`beyond_nyquist`), and 2.1% misses the detector (`left_window`);
7.7e-6 is lost resampling between grids (`resampling`), and 2.8e-13 is left
unaccounted (`numerical`).

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
  `Undersampled`, and on a wave path `TiltProjection`;
- `validity` flags;
- named approximations;
- a `WaveLedger`, where every step books what it changed to a cause:

```text
launched = detected + absorbed_aperture + absorbed_element + band_limit_removed
         + beyond_nyquist + evanescent + left_window + other_family
         + untracked_orders + other_arm + passed_by + resampling
         + other_ports + numerical
```

That it sums to what was launched is bookkeeping, not a check: `numerical`
is defined as what no step accounted for. The entries are the evidence.
`other_arm`, `passed_by`, `other_family` and `untracked_orders` are light on
paths the computation does not follow, never absorption; `resampling` is
what carrying a field onto another grid or plane cost (4.2e-4 of the power
through the f/10 singlet, 1.2e-3 on and off the LCoS of its test); and
`numerical`, round-off, is at most 7e-12 of the launched power on every path
the tests run.

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
| `voelz_regime_picks_the_transfer_function_below_the_critical_distance` | `Auto` below and beyond `z_c`: beyond it the smaller error, a Gaussian's band limit removing 2.5e-8 against the paraxial step's 2.5e-6 | exact choice | angular spectrum |
| `an_aperture_passes_exactly_its_area_of_a_uniform_field` | exact cell coverage | 1e-12 | passes, all four shapes |
| `voelz_regime_picks_the_transfer_function_below_the_critical_distance` | `z_c = LΔx/λ` [3] | exact | passes |
| `fresnel_integrals_and_bessel_functions_match_the_tables` | DLMF ch. 7 and 10 [15] | 1e-10 | passes |
| `a_square_aperture_far_field_is_sinc_squared` | Fraunhofer sinc² [1] | rms 1e-3 of peak | rms 4.5e-4 (worst 2.2e-3, in the outer lobes, from the sampled aperture) |
| `a_circular_aperture_makes_the_airy_pattern` | `2J₁(v)/v`; encircled energy 0.8378 at the first dark ring [1], [16] | 2e-3; ±1e-3 | 2.8e-4 of peak; 0.8376 |
| `the_on_axis_field_of_a_circular_aperture_matches_heurtley` | the exact on-axis Rayleigh–Sommerfeld field [5], Fresnel numbers 10 to 0.25 | 5e-3 at a/Δx = 256; error ratio > 3 on halving Δx; Richardson-extrapolated 3e-4 | 2.6e-3 → 1.6e-6 across the range; ratio 4.04–4.05 (second order); extrapolated 3.9e-5 (N_F 10) to 2.6e-8 (N_F 0.25). Band-limited ASM 2.2e-3 at N_F 10; `Auto` (impulse response) 9.4e-5 to 5.8e-6 in the far field |
| `a_square_aperture_fresnel_pattern_matches_the_fresnel_integrals` | Fresnel diffraction of a square through C and S [1], [15] | 1.5e-3 at 201 samples across; ratio > 3 | 3.5e-3 → 8.4e-4 (ratio 4.2), each method in its regime |
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
0.8653 (canvas) and 0.8642 (Optiland), 0.12% apart; the PSFs differ by
1.2e-4 rms of the ideal peak, and their sums over the grid by 0.08%. Two
things leave a residual of that order: the canvas's sampling (finer than its
2.9 µm pitch, the Strehl ratio drops by 0.06% at 1.5 µm), and the reference
itself (a uniform spherical cap peaks at `4π sin²(α/2)/λ²` per watt, 0.06%
above `π sin²α/λ²` at f/10).

Two things about the oracle itself. Its value is converged in pupil rays
(0.85039, 0.85029, 0.85027 at 256, 384 and 512 on an even grid). But
`strehl_ratio()` is the largest sample of the image grid, and an even grid has
no sample on the axis: on 64 points it reads 0.8503, on 65 the 0.8642 above.
The fixture uses an odd grid.

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

## What it does not do

- **Scalar diffraction.** No vector diffraction and no high-NA focusing: the
  field's two polarization components diffract each as a scalar field, and
  the longitudinal component is not carried. Scalar theory's error exceeds ±5% for features below about
  14 wavelengths [11].
- **Oblique elements by projection.** A lens, an aperture or a modulator seen
  obliquely is handed over by projecting the field along the rays
  (`TiltProjection`, reported); only the detector's diffracted readout uses
  the tilted-plane spectrum.
- **Lenses face by face.** A face is traced across its sag in the geometric
  limit; diffraction is taken between faces, not inside a face's slab.
- **On a wave path, the beam follows the chief ray.** Steering onto a chosen
  diffraction order is W7. Arms that recombine are added once: a cavity's
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
17. F. Wyrowski, M. Kuhn, "Introduction to field tracing," *J. Mod. Opt.* 58(5–6), 449–466 (2011), doi:10.1080/09500340.2010.532237. (The coupling of rays and fields; ADR 0003.)

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

The FFT is `rustfft` 6.4.1 (MIT OR Apache-2.0).
