# `oc-wave`: a physics-review brief

**What this is.** A self-contained description of the scalar wave optics in
Optiland Canvas: the `oc-wave` crate, the DMD and LCoS models it shares with
`oc-core`, and the ray-side pieces the wave path relies on. It is written for
a reviewer who will check the physics and the limits of every model **without
access to the repository**:
- every equation the code implements is stated;
- the code that implements it is quoted verbatim;
- every numerical test of the wave optics is listed with its reference, gate
  and measured value (§10). The unit and I/O tests are listed more briefly.

**Pinned code.** Branch `wave-optics`, commit `1af40e8` (local, not yet
published). Line numbers in excerpts refer to that commit. The brief adds no
code.

**What is asked of the reviewer.**

1. Check that each model's equations are right in this kernel's conventions
   (§3), and that the code implements them (the excerpts).
2. Check that each claim of exactness or of an error bound is justified, and
   name the conditions under which it fails.
3. Judge whether each test supports the claim it is attached to (§10 sorts
   the tests by the kind of evidence they give).
4. Answer the specific questions in §13, and add any problem the list misses.

§11.1 reports defects found while this brief was written. All are confirmed
by a reproduction and not yet fixed:
- a flat grating on the wave path loses its order's momentum, and the
  network ledger counts a grating's lost orders once per arm;
- the zoomed and tilted angular spectra ignore an input window centred off
  the frame origin;
- the diffracted readout's point sampling can misread the power, with no
  warning.

## Contents

1. [How to read this brief](#1-how-to-read-this-brief)
2. [System overview](#2-system-overview)
3. [Conventions](#3-conventions)
4. [Free-space propagation](#4-free-space-propagation)
5. [Thin elements and beams](#5-thin-elements-and-beams)
6. [Pixelated modulators: DMD and LCoS](#6-pixelated-modulators-dmd-and-lcos)
7. [The wave path through a scene](#7-the-wave-path-through-a-scene)
8. [The report and the ledger](#8-the-report-and-the-ledger)
9. [Ray-side models the wave path relies on](#9-ray-side-models-the-wave-path-relies-on)
10. [Evidence inventory](#10-evidence-inventory)
11. [Known defects, limitations and open findings](#11-known-defects-limitations-and-open-findings)
12. [Where the author is least sure](#12-where-the-author-is-least-sure)
13. [Questions for the reviewer](#13-questions-for-the-reviewer)
14. [Sources](#14-sources)
15. [Appendices](#15-appendices)

---

## 1. How to read this brief

**Excerpts.** Code is quoted exactly, with its path and line range at commit
`1af40e8`. A script checked every excerpt against `git show 1af40e8:<path>`
when the brief was built. The language is Rust. Two helpers recur:

- `Cx` is the kernel's complex type. `Cx::expi(φ)` is `e^{iφ}`, and
  `norm_sqr()` is `|z|²`.
- A `Vec3` is a 3D vector, with `dot` and `cross`.

**Units.** Lengths are in millimetres and vacuum wavelengths in micrometres.
The name `lam` in the code changes meaning:
- in the propagators, `planar.rs`, `regrid_plan` and `split_orders` it is the
  wavelength **in the medium, in mm** (`wavelength_in_medium_mm()`);
- in `ChainJob` (`self.lam`), `plate_of`, `slab`, `polarization_matters`,
  `transfer.rs`'s ray launches and the modulator functions it is the
  **vacuum wavelength in µm**;
- `lambda_um` is always the vacuum wavelength in µm.

Appendix D defines the helper functions and types the excerpts use.

**Project terms.**
- **W0–W8** are the project's first milestones (the wave core, the DMD and
  SLM models, the scene path, the app).
- **I1–I6** are the batches that followed a review against the bundle:
  - I1, fixes;
  - I2, interferometers;
  - I3, planar layers;
  - I4, sampling at foci;
  - I5, two polarization components;
  - I6, a linear lens transfer.
- In the specification bundle [Bundle], an **"M" number names either a
  section of its `01_MODEL.md` (M00–M20) or a tagged equation in it (M12–M40)**,
  and a "B" number names a benchmark in its `03_BENCHMARKS.md`. Appendix E
  states every bundle item the brief cites.

**Verification and validation.** The terms follow the specification bundle
the project supplied [Bundle], §6 of its `05_RESULTS_AND_CLAIMS.md`:

- *Verification* shows that the code computes the model it states, for
  example two discretizations of one integral that agree.
- *Validation* compares the model with an independent reference for the
  physics: a closed form, an exact solution, or a measurement.

No test here uses measured data (§10).

**Citations** are short keys, author–year where there is an author, resolved
in §14.

**Doc comments in excerpts** are quoted as they stand at the commit. Where one
no longer matches the code it sits on, the brief says so beside the excerpt
(and §11.3 lists them).

---

## 2. System overview

Optiland Canvas is an optical design tool: a Rust kernel compiled natively
and to WebAssembly, and a browser app. Its core is a non-sequential **f64
geometric ray tracer** (`oc-trace`). There a ray is a packet of flux with an
optical path length (OPL), never a field amplitude. `oc-wave` adds a complex
scalar field sampled on a plane, and the methods that move it.

The coupling is fixed by architecture decision ADR 0003:

- **Rays for the layout, fields for the diffraction.** Rays and fields meet
  only through the optical path. A field never feeds a ray's flux. A ray
  never carries a phase other than `k·OPL`.
- **Where the beam goes** comes from the ray tracer. A source's *axial ray*
  is traced through the scene with ghosts off. The ordered chain of surfaces
  it meets on its way to a chosen detector is the *sequence*.
- **What arrives** comes from the field. The field is carried along that
  chain: free space between the events, and an element model at each event.

```text
source (declares a coherent beam: Gaussian w₀, waist offset; or top-hat radius)
  │  axial ray traced by oc-trace ─► chain(s) of events to the detector
  ▼
launch field on an N×N grid across the axial ray
  │  for each event, in order:
  │    free space along the chief ray (ASM / Fresnel, automatic choice;
  │                                     re-gridded to fit the beam)
  │    element: aperture │ flat element (mirror, splitter, grating, thin lens)
  │             │ face of a body / curved surface (geometric field tracing)
  │             │ plane-parallel plate (closed-form slab, plane wave by plane wave)
  │             │ flat coating (TMM per plane wave) │ polarizer/retarder (Jones)
  │             │ DMD / LCoS (sampled near field on a device grid)
  ▼
detector readout: power per pixel (projected, or diffracted onto the face)
  │  several chains (interferometer arms): fields added on the detector grid
  │  several lines / two polarization launches: added in power
  ▼
WavePathResult: power per pixel, arriving field, WaveReport (steps, warnings,
                validity flags, named approximations, ledger of power by cause)
```

**Crate layout** (`crates/oc-wave/src`, lines at `1af40e8`):

| Module | Lines | Role |
|---|---|---|
| `lib.rs` | 70 | conventions, module list, the sample cap |
| `field.rs`, `grid.rs`, `fft.rs` | 205, 169, 233 | the sampled field, its grid, the scaled 2D FFT |
| `propagate/` (`asm`, `fresnel`, `fraunhofer`, `rs_direct`, `tilted`, `regime`, `mod`) | 1172 | free-space propagators and the automatic choice |
| `czt.rs` | 280 | chirp-z transform; zoomed Fraunhofer and Fresnel |
| `elements.rs` | 566 | apertures with exact cell coverage, thin lens, phase, transmittance |
| `beams.rs` | 202 | Gaussian beam, plane wave, point source |
| `modulator.rs` | 546 | DMD and LCoS sampled near field, factorized far fields |
| `design.rs` | 83 | Gerchberg–Saxton |
| `special.rs` | 103 | Bessel J_n and Fresnel integrals (for references) |
| `report.rs`, `error.rs` | 240, 36 | report, ledger, refusals |
| `path.rs` | 3536 | the wave path through a scene (feature `scene`) |
| `transfer.rs` | 1056 | geometric field tracing across a face |
| `planar.rs` | 424 | plates and coatings in the angular spectrum |

Outside the crate:

- `oc-core/src/modulator.rs` (1108 lines) describes the devices and computes
  their closed-form diffraction orders.
- `oc-core/src/thinfilm.rs` holds the coating transfer matrix (TMM).
- `oc-core/src/polarization.rs` holds the polarization frames and the
  A-plate retardance.
- `oc-trace` holds the ray tracer and the sequence extraction.

**Abbreviations** used throughout: ASM (angular spectrum method), TF and IR
(Fresnel transfer function and impulse response), RS (Rayleigh–Sommerfeld),
TMM (transfer-matrix method), AR (anti-reflection coating), SLM (spatial
light modulator), DMD (digital micromirror device), LCoS (liquid crystal on
silicon), TEA (thin-element approximation), ADR (architecture decision
record), RCWA (rigorous coupled-wave analysis), BEM (boundary-element
method), MMDFT (Optiland's matrix-multiply DFT PSF), `spp` (samples per
pixel).

**In the app.** The source panel declares the coherent beam, the window and
the samples. The detector dock's *wave* button runs the path in a Web
Worker, sliced into resumable steps, and shows the power per pixel or the
arriving field. The app always uses `WavePathSpec::for_source`:

- `Readout::Projected`;
- `device_samples_per_pixel = 4`;
- the grid is limited to 8–2048 samples across, and the browser caps a field
  at 2²² samples (2²⁶ natively). A larger request is refused, never
  truncated.

---

## 3. Conventions

<!-- excerpt crates/oc-wave/src/lib.rs 11 29 -->
*Excerpt: `crates/oc-wave/src/lib.rs`, lines 11–29, commit `1af40e8`.*

```rust
//! # Conventions
//!
//! These follow docs/physics/conventions.md, which fixes the time dependence
//! as `exp(-iωt)`, and Goodman's *Introduction to Fourier Optics*, 4th ed.
//! (W. H. Freeman/Macmillan Learning, 2017, ISBN 978-1-319-11916-4), which uses
//! the same one.
//!
//! - A plane wave travelling toward `+z` is `exp(+ikz)`.
//! - The forward transform kernel is `exp(-i2π(f_x x + f_y y))`, scaled by the
//!   sample area `ΔxΔy` so that it approximates the continuous Fourier
//!   integral; the inverse is scaled by the frequency cell `Δf_x Δf_y`. With
//!   those two scales Parseval's theorem holds exactly between the sampled
//!   field and its sampled spectrum, so power is comparable on both sides.
//! - Lengths are millimetres, wavelengths micrometres, as everywhere in the
//!   kernel. The factor of 1000 between them appears once, in
//!   [`Field2::wavelength_mm`].
//! - `|U|²` is flux per mm², so `Σ|U|²ΔA` is the flux the field carries. It is
//!   relative unless the source that launched it was calibrated, exactly like
//!   a ray's flux.
```

The comment's claim that the factor of 1000 between µm and mm "appears once"
is not true of the code. Several functions divide `lambda_um` by 1000 inline
(`beams.rs`, `launch_q`, `transfer.rs`, `dmd_far_field`), always correctly.

| Quantity | Convention |
|---|---|
| Time dependence | `exp(−iωt)` (Goodman's) |
| Plane wave along `+z` | `exp(+ikz)`, `k = 2πn/λ₀` |
| Forward transform | `Û(f) = Σ U(x) e^{−i2πf·x} ΔxΔy` |
| Inverse transform | `U(x) = Σ Û(f) e^{+i2πf·x} Δf_xΔf_y`, `Δf = 1/(NΔx)` |
| Parseval | `Σ|U|²ΔxΔy = Σ|Û|²Δf_xΔf_y`, exactly |
| Sample layout | sample `i` at `center + (i − ⌊n/2⌋)Δx`; row-major, `x` fastest |
| Thin lens | `t = exp(−ik r²/2f)`; `f > 0` converges |
| Normalization | `Σ|U|²ΔA` is the field's power, in the same relative units as ray flux |

<!-- excerpt crates/oc-wave/src/grid.rs 6 14 -->
*Excerpt: `crates/oc-wave/src/grid.rs`, lines 6–14, commit `1af40e8`.*

```rust
/// `n[0] × n[1]` samples at `pitch_mm`, centred on `center_mm` in the plane's
/// local `XY`.
///
/// Sample `i` along an axis sits at `center + (i − ⌊n/2⌋)·pitch`, so the
/// centre is a sample and the window runs from `−⌊n/2⌋` to `n − 1 − ⌊n/2⌋`
/// pitches about it. That is the layout an FFT expects after `ifftshift`, and
/// it puts zero spatial frequency at index `⌊n/2⌋` after `fftshift`.
///
/// Storage is row-major with `x` fastest: sample `(i, j)` is at `j·n[0] + i`.
```

**The field and its carrier.** A field is `data(x, y) · exp(ik(α₀x + β₀y))`.
The direction cosines `(α₀, β₀)` are an analytic plane-wave tilt held outside
the samples, so a steep beam needs no steep sampled phase. Every propagator
evaluates its transfer function at the absolute spatial frequency, carrier
included.

<!-- excerpt crates/oc-wave/src/field.rs 7 42 -->
*Excerpt: `crates/oc-wave/src/field.rs`, lines 7–42, commit `1af40e8`.*

```rust
/// `U(x, y)` on a [`Grid2`], at one vacuum wavelength, in one medium.
///
/// The true field is `data(x, y) · exp(ik(α₀x + β₀y))`, where `(α₀, β₀)` is
/// [`Field2::carrier`]: a plane-wave tilt held analytically instead of in the
/// samples. A beam crossing its own plane at 24° would otherwise need a phase
/// ramp of many radians per sample. Propagators account for the carrier;
/// everything that reads `data` directly sees the field with the tilt removed.
///
/// `frame` places the plane in the world: its local `+Z` is the normal the
/// field propagates along, and the samples lie over its local `X` and `Y`. That
/// is the surface convention `oc_core::grating` already uses.
#[derive(Debug, Clone, PartialEq)]
pub struct Field2 {
    pub grid: Grid2,
    /// Row-major, `x` fastest. `|U|²` is flux per mm².
    pub data: Vec<Cx>,
    /// Vacuum wavelength, µm.
    pub lambda_um: f64,
    /// Refractive index of the homogeneous medium the field propagates in.
    pub n_medium: f64,
    /// Direction cosines `(α₀, β₀)` of the analytic carrier, in the plane's
    /// local frame.
    pub carrier: [f64; 2],
    /// The plane in the world.
    pub frame: Isometry3,
    /// The optical path of the reference ray, the beam's axis, from where the
    /// field was launched to the frame origin, mm: what a ray reports there,
    /// for comparing a field with a trace.
    ///
    /// It is a record, never applied again: the samples carry the whole
    /// phase. Every propagator multiplies them by its full transfer function,
    /// `e^{ikz}` included, and a traced element by `exp(ik₀·OPL)` along its
    /// rays, so two fields that left one source by different paths and meet
    /// on one plane add as they are.
    pub opl_ref_mm: f64,
}
```

**One phase clock.** The samples carry the whole phase:

- every propagator multiplies them by its full transfer function, `e^{ikz}`
  included;
- a traced element multiplies them by `exp(ik₀·OPL)` along its rays.

`opl_ref_mm` records the reference ray's OPL from the launch to the frame
origin. It is never applied again. Two fields that left one source by
different paths can therefore be added as they are. That is what the
interferometer readout does (§7.11).

**Two index conventions.**
- A field's sample `i` is at `center + (i − ⌊n/2⌋)Δx` (`Grid2`), so the
  centre is a sample.
- A device's pixel `i` is centred at `(i − (N − 1)/2)p` (`SpatialModulator`),
  so for even `N` the device centre falls between pixels.

**What `|U|²` means.** The field's transverse samples are a scalar **power
coordinate**: the angular-spectrum step is unitary on them under Parseval,
so `Σ|U|²ΔA` is conserved by free space whatever the angles.

Reading `|U|²` as irradiance on a plane is a declared paraxial area detector,
in the bundle's terms. The power a non-paraxial plane-wave mix sends through a
plane carries each plane wave's obliquity, and `Σ|U|²ΔA` does not. On the
wave path the field's plane is always across the chief ray, so for beams near
the chief ray the difference is second order in the angular spread. Tilted
readout planes are treated separately (§4.6).

**Frames.** A field's `frame` places its plane in the world: local `+Z` is
the propagation normal, and the samples lie over local `X`, `Y`. On the wave
path, the frame across the chief ray turns at each event by the shortest
rotation from the incoming direction to the outgoing one (`Beam::turned`,
§7.1).

Two other frames differ from that convention:
- A source's optical axis is its local `+X` (the tracer's convention). The
  wave path builds the launch frame across that axis.
- A modulator's device frame has light arriving toward `−Z` (§6).

---

## 4. Free-space propagation

`oc_wave::propagate(field, z, Propagator, fft)` returns the field `z`
further along its frame's `+Z`, with a `WaveReport` (§8).

| Propagator | Transfer function or kernel | Valid where | Source |
|---|---|---|---|
| `AngularSpectrum` | `H = exp(i2πz√(1/λ² − f²))`, λ in the medium | any `z`, homogeneous medium; band-limited by default | [Goodman 2017], [Matsushima & Shimobaba 2009] |
| `FresnelTf` | `H = e^{ikz} exp(−iπλz f²)` | paraxial; `z ≤ z_c` | [Voelz & Roggemann 2009] |
| `FresnelIr` | `h = e^{ikz}/(iλz) · exp(iπr²/λz)` | paraxial; `z ≥ z_c` | same |
| `Fraunhofer{pad}` | one transform onto pitch `λz/(PΔx)` | source Fresnel number < 0.1 | [Goodman 2017] |
| `RsDirect` | first Rayleigh–Sommerfeld kernel by FFT convolution | kernel sampled to the largest offset | [Shen & Wang 2006] |
| `Auto` | the band-limited ASM or the Fresnel IR, chosen and recorded (never the TF) | — | [Voelz & Roggemann 2009] |

**Three critical distances.** `z_c = LΔx/λ` appears in three places with
three `L`:
- **the Fresnel steps' regime warning**: the padded width
  `next_fast_len(2n)Δx`, per axis;
- **`Auto`**: exactly `2nΔx` on axis 0;
- **the re-grid** (§7.2): the unpadded window `nΔx`.

### 4.1 Angular spectrum, band-limited

The field is zero-padded to `P = next_fast_len(2n)` samples per axis (the
smallest length ≥ 2n with only 2, 3 and 5 as prime factors). Padding turns
the circular convolution into a linear one over the original window. `pad: 1`
keeps it circular, which is exact for a field periodic in its window (the
Talbot test).

The band limit is [Matsushima & Shimobaba 2009], Eqs. (13) and (18)–(20):

```text
u_limit = 1 / (λ √((2Δu z)² + 1)),   Δu = 1/(PΔx)    (same in v)
keep only  u²/u_limit² + λ²v² ≤ 1   and   λ²u² + v²/v_limit² ≤ 1
```

Here `u, v` are absolute frequencies, carrier included. For a linear carrier
this is the carrier-conjugated projector of the bundle's section M10
(Eq. M20, Appendix E): the band is applied to the physical field, not to the
samples' own spectrum.

The bookkeeping:
- **Removed power** goes to `band_limit_removed` and raises a `BandLimited`
  warning when it exceeds 1e-12 of the launched power. It is never
  renormalized back into the field.
- **Evanescent components** decay as `exp(−2πz√(f² − 1/λ²))` going forward,
  and the decayed share goes to `evanescent`. For `z ≤ 0` they are removed
  and booked whole (`propagate` returns early at `z = 0`, so in practice
  this is the backward case).
- **Power cropped** to the original window goes to `left_window`.
- A **`WrapAround` warning** is raised when more than 1e-6 of the output power
  lies in the outer sixteenth of the padded window.

<!-- excerpt crates/oc-wave/src/propagate/asm.rs 80 144 -->
*Excerpt: `crates/oc-wave/src/propagate/asm.rs`, lines 80–144, commit `1af40e8`.*

```rust
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let df = du * dv;
    // Matsushima & Shimobaba Eqs. (13), (20).
    let u_lim = 1.0 / (lam * ((2.0 * du * z).powi(2) + 1.0).sqrt());
    let v_lim = 1.0 / (lam * ((2.0 * dv * z).powi(2) + 1.0).sqrt());
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);

    let mut removed = 0.0;
    let mut evanescent = 0.0;
    for ky in 0..p[1] {
        let fy = gp.fft_freq(1, ky) + v0;
        for kx in 0..p[0] {
            let fx = gp.fft_freq(0, kx) + u0;
            let a = &mut buf[ky * p[0] + kx];
            if opts.band_limit {
                // Eqs. (18), (19): inside both ellipses.
                let e1 = fx * fx / (u_lim * u_lim) + lam * lam * fy * fy;
                let e2 = lam * lam * fx * fx + fy * fy / (v_lim * v_lim);
                if e1 > 1.0 || e2 > 1.0 {
                    removed += a.norm_sqr();
                    *a = Cx::ZERO;
                    continue;
                }
            }
            let w = inv_l2 - fx * fx - fy * fy;
            if w >= 0.0 {
                *a *= Cx::expi(2.0 * PI * z * w.sqrt());
            } else if z > 0.0 {
                let d = (-2.0 * PI * z * (-w).sqrt()).exp();
                evanescent += a.norm_sqr() * (1.0 - d * d);
                *a *= d;
            } else {
                evanescent += a.norm_sqr();
                *a = Cx::ZERO;
            }
        }
    }

    fft.inverse(&mut buf, p, df);
    fftshift2(&mut buf, p);
    // With no padding the propagation is periodic on purpose (a grating's
    // Talbot carpet): power at the edge is the next period, not an artefact.
    if opts.pad > 1 {
        check_wrap(&buf, p, report);
    }
    let full = sum_sq(&buf) * gp.cell_area_mm2();
    let data = if opts.pad == 1 { buf } else { crop(&buf, p, n) };

    let mut out = u.clone();
    out.data = data;
    out.frame = advance(u, z);
    out.opl_ref_mm = u.opl_ref_mm + u.n_medium * z;
    let kept = out.power();

    report.ledger.band_limit_removed += removed * df;
    report.ledger.evanescent += evanescent * df;
    report.ledger.left_window += (full - kept).max(0.0);
    if removed > 0.0 {
        let frac = removed * df / report.ledger.launched.max(f64::MIN_POSITIVE);
        if frac > 1e-12 {
            report.warn(SamplingWarning::BandLimited { removed: frac });
        }
    }
```

### 4.2 Fresnel transfer function and impulse response

Both are padded twice. The critical distance is `z_c = LΔx/λ`, `L` the padded
width `next_fast_len(2n)Δx` [Voelz & Roggemann 2009]:

- below `z_c` the transfer function's chirp is sampled well;
- above it the impulse response's chirp is.

Running either method on the wrong side is allowed (comparisons need it) and
raises `OutsideRegime`. The impulse response's sampled chirp, cut off at the
padded window, is not unitary. Its net loss is booked as `left_window`, and
any gain falls into `numerical`.

<!-- excerpt crates/oc-wave/src/propagate/fresnel.rs 29 184 -->
*Excerpt: `crates/oc-wave/src/propagate/fresnel.rs`, lines 29–184, commit `1af40e8`.*

```rust
fn padded(u: &Field2) -> Result<([usize; 2], Grid2), WaveError> {
    let p = [next_fast_len(2 * u.grid.n[0]), next_fast_len(2 * u.grid.n[1])];
    let g = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    Ok((p, g))
}

fn regime_note(
    u: &Field2,
    g: &Grid2,
    z: f64,
    want: Sampling,
    method: Method,
    report: &mut WaveReport,
) -> Sampling {
    let lam = u.wavelength_in_medium_mm();
    // The coarser axis decides: both must be in regime.
    let s0 = voelz_sampling(g.pitch_mm[0], g.width_mm(0), lam, z);
    let s1 = voelz_sampling(g.pitch_mm[1], g.width_mm(1), lam, z);
    let s = if s0 == s1 {
        s0
    } else if s0 == Sampling::Critical {
        s1
    } else if s1 == Sampling::Critical {
        s0
    } else {
        // One axis each side: neither method is in regime on both.
        Sampling::Undersampled
    };
    if s != want && s != Sampling::Critical {
        let zc = critical_distance_mm(g.pitch_mm[0], g.width_mm(0), lam);
        report.warn(SamplingWarning::OutsideRegime {
            method,
            reason: format!(
                "z = {z} mm against a critical distance of {zc:.6} mm (Voelz & Roggemann 2009)"
            ),
        });
    }
    s
}

/// Fresnel transfer function.
pub(super) fn run_tf(
    u: &Field2,
    z: f64,
    fft: &mut Fft2,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    let (p, gp) = padded(u)?;
    let sampling = regime_note(u, &gp, z, Sampling::Oversampled, Method::FresnelTf, report);
    let mut buf = embed(u, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());

    let lam = u.wavelength_in_medium_mm();
    let k = u.k_mm();
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let global = Cx::expi(k * z);
    for ky in 0..p[1] {
        let fy = gp.fft_freq(1, ky) + v0;
        for kx in 0..p[0] {
            let fx = gp.fft_freq(0, kx) + u0;
            buf[ky * p[0] + kx] *= global * Cx::expi(-PI * lam * z * (fx * fx + fy * fy));
        }
    }
    finish(u, z, buf, p, &gp, fft, report, Method::FresnelTf, Some(sampling))
}

/// Fresnel impulse response.
pub(super) fn run_ir(
    u: &Field2,
    z: f64,
    fft: &mut Fft2,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    let (p, gp) = padded(u)?;
    let sampling = regime_note(u, &gp, z, Sampling::Undersampled, Method::FresnelIr, report);
    let lam = u.wavelength_in_medium_mm();
    let k = u.k_mm();
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);

    // The kernel on the padded grid, about its centre. With a carrier the
    // convolution of `data · e^{i2πu₀x}` with `h` is
    // `e^{i2πu₀x} · (data ⊛ h·e^{−i2πu₀x})`, so the kernel takes the
    // conjugate tilt and the samples keep theirs.
    let pre = Cx::expi(k * z) / Cx::new(0.0, lam * z);
    let mut h = vec![Cx::ZERO; p[0] * p[1]];
    for j in 0..p[1] {
        let y = (j as f64 - (p[1] / 2) as f64) * gp.pitch_mm[1];
        for i in 0..p[0] {
            let x = (i as f64 - (p[0] / 2) as f64) * gp.pitch_mm[0];
            h[j * p[0] + i] = pre
                * Cx::expi(PI * (x * x + y * y) / (lam * z) - 2.0 * PI * (u0 * x + v0 * y));
        }
    }
    ifftshift2(&mut h, p);
    fft.forward(&mut h, p, gp.cell_area_mm2());

    let mut buf = embed(u, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    for (a, hh) in buf.iter_mut().zip(h.iter()) {
        *a *= *hh;
    }
    finish(u, z, buf, p, &gp, fft, report, Method::FresnelIr, Some(sampling))
}

#[allow(clippy::too_many_arguments)]
fn finish(
    u: &Field2,
    z: f64,
    mut buf: Vec<Cx>,
    p: [usize; 2],
    gp: &Grid2,
    fft: &mut Fft2,
    report: &mut WaveReport,
    method: Method,
    sampling: Option<Sampling>,
) -> Result<Field2, WaveError> {
    fft.inverse(&mut buf, p, gp.freq_pitch(0) * gp.freq_pitch(1));
    fftshift2(&mut buf, p);
    check_wrap(&buf, p, report);
    let full = sum_sq(&buf) * gp.cell_area_mm2();
    let mut out = u.clone();
    out.data = crop(&buf, p, u.grid.n);
    out.frame = advance(u, z);
    out.opl_ref_mm = u.opl_ref_mm + u.n_medium * z;
    if method == Method::FresnelIr {
        // The impulse response is a sampled chirp cut off at the padded
        // window: it is not unitary. It can fail to carry power that would
        // diffract past the window's reach, and it can put spurious power in
        // the margin the crop discards. Only the net loss is light that left;
        // an excess is numerical, and `close` books it there.
        report.ledger.left_window += (report.ledger.launched - out.power()).max(0.0);
    } else {
        // |H| ≤ 1: what the crop discards is light that left the window.
        report.ledger.left_window += (full - out.power()).max(0.0);
    }
    report.steps.push(StepReport {
        method,
        z_mm: z,
        sampling,
        note: format!(
            "Fresnel {} on a {}×{} grid padded to {}×{}",
            if method == Method::FresnelTf {
                "transfer function"
            } else {
                "impulse response"
            },
            u.grid.n[0],
            u.grid.n[1],
            p[0],
            p[1]
        ),
    });
    Ok(out)
}
```

### 4.3 The paraxial step's exact error, and `Auto`

Both propagators are diagonal in the plane waves. With `q = 2πf`,
`κ = √(k² − |q|²)` and its paraxial form `κ₂ = k − |q|²/2k`, Parseval gives
the error of the paraxial step on a field `g` exactly:

```text
‖(U − U₂)g‖² = Σ_q 4 sin²(z(κ − κ₂)/2) |ĝ(q)|²,
κ − κ₂ = −|q|⁴ / (2k(k + κ)²)      (stable form)
```

This is the bundle's Eq. M32, in its section M13 (Appendix E). It is summed
over the field's own spectrum, carrier included. An evanescent component, which the paraxial propagator
carries unchanged, counts whole.

The result is returned relative to `‖g‖`. Every Fresnel step states it in its
note, and raises `Paraxial` above 1e-4. The band limit's error is defined as
the norm of the field the band limit would remove, relative to `‖g‖`.

**`Auto`'s rule:**
- At or below `z_c`, computed with the padded width taken as exactly `2nΔx`
  on axis 0, use the band-limited ASM.
- Beyond it, compute both errors from one FFT of the field. Use the Fresnel
  impulse response if its paraxial error is not larger, and the band-limited
  ASM otherwise.

<!-- excerpt crates/oc-wave/src/propagate/mod.rs 142 197 -->
*Excerpt: `crates/oc-wave/src/propagate/mod.rs`, lines 142–197, commit `1af40e8`.*

```rust
fn regime_errors(u: &Field2, z_mm: f64, fft: &mut Fft2) -> (f64, f64) {
    let n = u.grid.n;
    let mut buf = u.data.clone();
    crate::fft::ifftshift2(&mut buf, n);
    fft.forward(&mut buf, n, u.grid.cell_area_mm2());
    let lam = u.wavelength_in_medium_mm();
    let k = 2.0 * std::f64::consts::PI / lam;
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    // The band limit for the window padded twice, as the step would run.
    let (du, dv) = (1.0 / (2.0 * u.grid.width_mm(0)), 1.0 / (2.0 * u.grid.width_mm(1)));
    let u_lim = 1.0 / (lam * ((2.0 * du * z_mm).powi(2) + 1.0).sqrt());
    let v_lim = 1.0 / (lam * ((2.0 * dv * z_mm).powi(2) + 1.0).sqrt());
    let (mut err, mut cut, mut tot) = (0.0, 0.0, 0.0);
    for ky in 0..n[1] {
        let fy = u.grid.fft_freq(1, ky) + v0;
        for kx in 0..n[0] {
            let fx = u.grid.fft_freq(0, kx) + u0;
            let e = buf[ky * n[0] + kx].norm_sqr();
            tot += e;
            if fx * fx / (u_lim * u_lim) + lam * lam * fy * fy > 1.0 || lam * lam * fx * fx + fy * fy / (v_lim * v_lim) > 1.0 {
                cut += e;
            }
            let q2 = (2.0 * std::f64::consts::PI).powi(2) * (fx * fx + fy * fy);
            let w = k * k - q2;
            if w <= 0.0 {
                err += e;
                continue;
            }
            let dk = -q2 * q2 / (2.0 * k * (k + w.sqrt()).powi(2));
            err += 4.0 * (0.5 * z_mm * dk).sin().powi(2) * e;
        }
    }
    if tot > 0.0 {
        ((err / tot).sqrt(), (cut / tot).sqrt())
    } else {
        (0.0, 0.0)
    }
}

/// The [`Propagator::Auto`] rule, decided before anything runs so the choice
/// is reproducible and can be explained: from the sampling up to the critical
/// distance; beyond it, the smaller of the two errors this field would meet,
/// which are returned with the choice.
fn choose(u: &Field2, z_mm: f64, fft: &mut Fft2) -> (Propagator, Option<(f64, f64)>) {
    let padded_width = (2 * u.grid.n[0]) as f64 * u.grid.pitch_mm[0];
    let zc = critical_distance_mm(u.grid.pitch_mm[0], padded_width, u.wavelength_in_medium_mm());
    if z_mm.abs() <= zc {
        return (Propagator::AngularSpectrum(AsmOptions::default()), None);
    }
    let (ep, eb) = regime_errors(u, z_mm, fft);
    if ep <= eb {
        (Propagator::FresnelIr, Some((ep, eb)))
    } else {
        (Propagator::AngularSpectrum(AsmOptions::default()), Some((ep, eb)))
    }
}
```

<!-- excerpt crates/oc-wave/src/propagate/regime.rs 15 17 -->
*Excerpt: `crates/oc-wave/src/propagate/regime.rs`, lines 15–17, commit `1af40e8`.*

```rust
pub fn critical_distance_mm(pitch_mm: f64, width_mm: f64, lambda_mm: f64) -> f64 {
    width_mm * pitch_mm / lambda_mm
}
```

### 4.4 Fraunhofer

The step implements:

```text
U₂(x₂) = e^{ikz} e^{ik|x₂|²/2z} / (iλz) · F{U₁}(x₂/λz)
```

with one FFT. The output pitch is `λz/(PΔx)`, and the output grid is centred
at `z·(α₀, β₀)`, the point the carrier direction reaches (paraxially). Power
is conserved by the scaling. The step warns `OutsideRegime` when the Fresnel
number of the source's support exceeds 0.1. That number is `r²_max/(λz)`,
with the support taken as the samples above 1e-12 of the peak intensity,
measured from the frame axis.

<!-- excerpt crates/oc-wave/src/propagate/fraunhofer.rs 63 89 -->
*Excerpt: `crates/oc-wave/src/propagate/fraunhofer.rs`, lines 63–89, commit `1af40e8`.*

```rust
    let mut buf = super::embed(u, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    fftshift2(&mut buf, p);

    // Output grid: x₂ = λz·f, centred where the carrier points.
    let pitch2 = [lam * z.abs() * gp.freq_pitch(0), lam * z.abs() * gp.freq_pitch(1)];
    let c2 = [u.carrier[0] * z, u.carrier[1] * z];
    let g2 = Grid2::new(p, pitch2, c2)?;
    let (cx, cy) = (u.grid.center_mm[0], u.grid.center_mm[1]);
    let pre = Cx::expi(k * z) / Cx::new(0.0, lam * z);
    for j in 0..p[1] {
        for i in 0..p[0] {
            let (x2, y2) = g2.xy(i, j);
            // The transform was taken about the window centre; about the
            // frame axis it carries the shift phase exp(−i2πf·c).
            let (fx, fy) = (x2 / (lam * z), y2 / (lam * z));
            // Sample i holds F{data} at (i − P/2)/(PΔx); the true field's
            // spectrum is that shifted by the carrier, which the output grid's
            // centre absorbed.
            let shift = Cx::expi(-2.0 * PI * ((fx - u.carrier[0] / lam) * cx
                + (fy - u.carrier[1] / lam) * cy));
            let quad = Cx::expi(k * (x2 * x2 + y2 * y2) / (2.0 * z));
            buf[j * p[0] + i] *= pre * quad * shift;
        }
    }
    // No rescale: |pre|² = 1/(λz)² and ΔA₂ = (λz)²Δf, so Σ|U₂|²ΔA₂ = Σ|Û|²Δf.
```

`fraunhofer_zoom` evaluates the same expression on any output grid by
chirp-z ([Bluestein 1970]; [Rabiner, Schafer & Rader 1969]; used for
diffraction in the same way by [Hu et al. 2020]). The power outside the
output window is booked as `left_window`.

`czt2` (below) evaluates the 2D sum `Σ u(x)e^{−i2πf·x}ΔxΔy` with `x` the
samples' absolute coordinates in the frame, carrier excluded.

<!-- excerpt crates/oc-wave/src/czt.rs 34 115 -->
*Excerpt: `crates/oc-wave/src/czt.rs`, lines 34–115, commit `1af40e8`.*

```rust
pub fn czt_line(
    x: &[Cx],
    x0: f64,
    dx: f64,
    f0: f64,
    df: f64,
    m_out: usize,
    fft: &mut Fft2,
) -> Vec<Cx> {
    let n = x.len();
    if n == 0 || m_out == 0 {
        return vec![Cx::ZERO; m_out];
    }
    let l = next_fast_len(n + m_out - 1);
    let a = df * dx; // the chirp rate: e^{∓iπ a k²}
    // y_n = x_n e^{−i2π f₀ nΔx} e^{−iπ a n²}
    let mut y = vec![Cx::ZERO; l];
    for (k, v) in x.iter().enumerate() {
        let kf = k as f64;
        y[k] = *v * Cx::expi(-2.0 * PI * f0 * kf * dx - PI * a * kf * kf);
    }
    // v_k = e^{+iπ a k²} for k from −(n−1) to m−1, wrapped.
    let mut v = vec![Cx::ZERO; l];
    for (k, vk) in v.iter_mut().enumerate().take(m_out) {
        let kf = k as f64;
        *vk = Cx::expi(PI * a * kf * kf);
    }
    for k in 1..n {
        let kf = k as f64;
        v[l - k] = Cx::expi(PI * a * kf * kf);
    }
    fft.forward(&mut y, [l, 1], 1.0);
    fft.forward(&mut v, [l, 1], 1.0);
    for (p, q) in y.iter_mut().zip(&v) {
        *p *= *q;
    }
    fft.inverse(&mut y, [l, 1], 1.0 / l as f64);
    (0..m_out)
        .map(|m| {
            let mf = m as f64;
            // X_m = e^{−i2π(f₀ + mδf)x₀} · e^{−iπ a m²} · (y ⊛ v)_m, times Δx.
            y[m] * Cx::expi(-2.0 * PI * (f0 + mf * df) * x0 - PI * a * mf * mf) * dx
        })
        .collect()
}

/// The sampled continuous 2D Fourier transform of `u`'s samples (carrier not
/// included) about the frame origin, on the frequency grid `f₀ + m·δf` per
/// axis with `m_out` points per axis.
pub fn czt2(
    u: &Field2,
    f0: [f64; 2],
    df: [f64; 2],
    m_out: [usize; 2],
    fft: &mut Fft2,
) -> Vec<Cx> {
    let [nx, ny] = u.grid.n;
    let [mx, my] = m_out;
    let x0 = u.grid.coord(0, 0);
    let y0 = u.grid.coord(1, 0);
    let [dx, dy] = u.grid.pitch_mm;
    // Along x: ny rows → ny × mx.
    let mut rows = vec![Cx::ZERO; ny * mx];
    for j in 0..ny {
        let line = &u.data[j * nx..(j + 1) * nx];
        let out = czt_line(line, x0, dx, f0[0], df[0], mx, fft);
        rows[j * mx..(j + 1) * mx].copy_from_slice(&out);
    }
    // Along y: mx columns → my × mx.
    let mut out = vec![Cx::ZERO; my * mx];
    let mut col = vec![Cx::ZERO; ny];
    for i in 0..mx {
        for j in 0..ny {
            col[j] = rows[j * mx + i];
        }
        let c = czt_line(&col, y0, dy, f0[1], df[1], my, fft);
        for (m, v) in c.into_iter().enumerate() {
            out[m * mx + i] = v;
        }
    }
    out
}
```

<!-- excerpt crates/oc-wave/src/czt.rs 160 181 -->
*Excerpt: `crates/oc-wave/src/czt.rs`, lines 160–181, commit `1af40e8`.*

```rust
    // f = x₂/(λz), shifted back by the carrier: F_true(f) = F_data(f − u₀).
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let lz = lam * z_mm;
    let f0 = [out.coord(0, 0) / lz - u0, out.coord(1, 0) / lz - v0];
    let df = [out.pitch_mm[0] / lz, out.pitch_mm[1] / lz];
    let mut data = czt2(u, f0, df, out.n, fft);
    let pre = Cx::expi(k * z_mm) / Cx::new(0.0, lz);
    for j in 0..out.n[1] {
        for i in 0..out.n[0] {
            let (x, y) = out.xy(i, j);
            data[out.idx(i, j)] *= pre * Cx::expi(k * (x * x + y * y) / (2.0 * z_mm));
        }
    }
    let mut v = u.clone();
    v.grid = out;
    v.data = data;
    v.carrier = [0.0, 0.0];
    v.frame = advance(u, z_mm);
    v.opl_ref_mm = u.opl_ref_mm + u.n_medium * z_mm;
    report.ledger.detected = v.power();
    report.ledger.left_window = (report.ledger.launched - report.ledger.detected).max(0.0);
    report.ledger.close();
```

### 4.5 Direct Rayleigh–Sommerfeld

The kernel is the first Rayleigh–Sommerfeld solution in the `exp(−iωt)`
convention:

```text
h(x, y) = (1/2π) (z/r) (1/r − ik) e^{ikr}/r,   r = √(x² + y² + z²)
```

It is applied as a linear convolution by FFT on at least `2n − 1` samples
[Shen & Wang 2006]. The source carries Simpson weights on odd axes and
trapezoidal weights on even axes. The kernel's local frequency at the largest
offset `L = (n − 1)Δx` must stay below Nyquist, `Δx ≤ λ√(z² + L²)/(2L)`.
Otherwise the step warns `Undersampled`.

The output window is the input window. The method cannot tell power that
diffracted out of the window from quadrature error, so it books neither: the
whole gap lands in `numerical`, and the step's note says so. The carrier is
first folded into the samples (`bake_carrier`), because the kernel has no
carrier term.

<!-- excerpt crates/oc-wave/src/propagate/rs_direct.rs 28 147 -->
*Excerpt: `crates/oc-wave/src/propagate/rs_direct.rs`, lines 28–147, commit `1af40e8`.*

```rust
fn weights(n: usize) -> (Vec<f64>, &'static str) {
    if n == 1 {
        return (vec![1.0], "single sample");
    }
    if n % 2 == 1 {
        let w = (0..n)
            .map(|i| {
                if i == 0 || i == n - 1 {
                    1.0 / 3.0
                } else if i % 2 == 1 {
                    4.0 / 3.0
                } else {
                    2.0 / 3.0
                }
            })
            .collect();
        (w, "Simpson")
    } else {
        let w = (0..n)
            .map(|i| if i == 0 || i == n - 1 { 0.5 } else { 1.0 })
            .collect();
        (w, "trapezoidal")
    }
}

pub(super) fn run(
    u: &Field2,
    z: f64,
    fft: &mut Fft2,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    if z <= 0.0 {
        return Err(WaveError::OutOfRange(
            "direct Rayleigh–Sommerfeld integration needs z > 0".into(),
        ));
    }
    let mut src = u.clone();
    src.bake_carrier();
    let [nx, ny] = u.grid.n;
    // The kernel is sampled out to the largest source–observation offset,
    // `L ≈ (n − 1)Δx`. Its local frequency there is `L / (λ√(z² + L²))`
    // (the sine of the angle over λ), and it must stay below Nyquist,
    // `1/(2Δx)`: the impulse-response sampling condition of Voelz &
    // Roggemann (2009), extended by them to Rayleigh–Sommerfeld, for a
    // linear convolution.
    {
        let lam = u.wavelength_in_medium_mm();
        for axis in 0..2 {
            let dx = u.grid.pitch_mm[axis];
            let l = (u.grid.n[axis].max(2) - 1) as f64 * dx;
            let local = l / (lam * (z * z + l * l).sqrt());
            let step = 2.0 * PI * local * dx;
            if step > PI {
                report.warn(crate::report::SamplingWarning::Undersampled {
                    what: format!(
                        "Rayleigh–Sommerfeld kernel along {} at the largest offset (needs Δx ≤ {:.4e} mm)",
                        if axis == 0 { "x" } else { "y" },
                        lam * (z * z + l * l).sqrt() / (2.0 * l)
                    ),
                    max_phase_step_rad: step,
                });
            }
        }
    }
    let q = [next_fast_len(2 * nx - 1), next_fast_len(2 * ny - 1)];
    let (wx, rule_x) = weights(nx);
    let (wy, rule_y) = weights(ny);
    let [dx, dy] = u.grid.pitch_mm;
    let k = u.k_mm();

    let mut a = vec![Cx::ZERO; q[0] * q[1]];
    for j in 0..ny {
        for i in 0..nx {
            a[j * q[0] + i] = src.at(i, j) * (wx[i] * wy[j]);
        }
    }
    let mut h = vec![Cx::ZERO; q[0] * q[1]];
    let kernel = |x: f64, y: f64| {
        let r2 = x * x + y * y + z * z;
        let r = r2.sqrt();
        Cx::new(1.0 / r, -k) * Cx::expi(k * r) * (z / (2.0 * PI * r2))
    };
    for mj in -(ny as isize - 1)..=(ny as isize - 1) {
        let jj = mj.rem_euclid(q[1] as isize) as usize;
        for mi in -(nx as isize - 1)..=(nx as isize - 1) {
            let ii = mi.rem_euclid(q[0] as isize) as usize;
            h[jj * q[0] + ii] = kernel(mi as f64 * dx, mj as f64 * dy);
        }
    }
    fft.forward(&mut a, q, 1.0);
    fft.forward(&mut h, q, 1.0);
    for (x, y) in a.iter_mut().zip(h.iter()) {
        *x *= *y;
    }
    fft.inverse(&mut a, q, dx * dy / (q[0] * q[1]) as f64);

    let mut out = src;
    for j in 0..ny {
        for i in 0..nx {
            out.data[j * nx + i] = a[j * q[0] + i];
        }
    }
    out.frame = advance(u, z);
    out.opl_ref_mm = u.opl_ref_mm + u.n_medium * z;
    // The output window is the input window. This method cannot tell power
    // that diffracted out of it from quadrature error, so it books neither:
    // the ledger's close puts the whole gap under `numerical`, and the note
    // says why.
    report.steps.push(StepReport {
        method: Method::RsDirect,
        z_mm: z,
        sampling: None,
        note: format!(
            "direct Rayleigh–Sommerfeld, {rule_x}×{rule_y} weights, convolution grid {}×{}; \
             power leaving the window and quadrature error are not separated (both booked as numerical)",
            q[0], q[1]
        ),
    });
    Ok(out)
}
```

### 4.6 Zoomed and tilted angular spectra

**`asm_zoom(u, z, out, pad)`** advances each plane wave exactly and evaluates
the inverse transform on any output grid (window and pitch) by chirp-z, one
axis at a time. No band limit is applied, since the output is a window and
not a periodic copy. Without padding the field repeats a window away. For a
beam converging onto the output that is harmless, because each copy focuses a
window away. Every re-grid (§7.2) and the square diffracted readout (§7.13)
use it. Its comment "the transform is about the frame origin" holds only when
the input window is centred on the frame origin (§11.1.2).

<!-- excerpt crates/oc-wave/src/propagate/asm.rs 167 275 -->
*Excerpt: `crates/oc-wave/src/propagate/asm.rs`, lines 167–275, commit `1af40e8`.*

```rust
/// The angular-spectrum field `z` ahead on any output grid `out` (in the
/// field's own frame: its axes, any centre and pitch), so that a focal spot
/// can be sampled far finer than the field it comes from.
///
/// The spectrum is the FFT of the field, zero-padded `pad` times; each plane
/// wave is advanced by the exact transfer function of [`run`]; and the
/// inverse transform is evaluated on the output grid by chirp-z
/// ([`crate::czt::czt_line`]), one pass per axis, instead of by an inverse
/// FFT on the input's grid. No band limit is applied: the output is a window,
/// not a periodic copy of the input. Without padding the field is periodic in
/// its window, which for a beam converging to the output is harmless (each
/// periodic copy focuses a window away). Evanescent power is booked; power
/// that lands outside the output window is booked as having left it.
pub fn asm_zoom(
    u: &Field2,
    z: f64,
    out: Grid2,
    pad: usize,
    fft: &mut Fft2,
) -> Result<(Field2, WaveReport), WaveError> {
    if !(z.is_finite() && z > 0.0) {
        return Err(WaveError::OutOfRange("a zoomed angular spectrum goes forward, a positive distance".into()));
    }
    out.validate()?;
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();
    let n = u.grid.n;
    let p = if pad <= 1 { n } else { [next_fast_len(pad * n[0]), next_fast_len(pad * n[1])] };
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let mut buf = if pad <= 1 { u.data.clone() } else { embed(u, p) };
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    fftshift2(&mut buf, p);
    // After the shift, sample k sits at frequency (k − p/2)·Δf.
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let f_at = |k: usize, axis: usize| (k as f64 - (p[axis] / 2) as f64) * if axis == 0 { du } else { dv };
    let mut evanescent = 0.0;
    for ky in 0..p[1] {
        let fy = f_at(ky, 1) + v0;
        for kx in 0..p[0] {
            let fx = f_at(kx, 0) + u0;
            let a = &mut buf[ky * p[0] + kx];
            let w = inv_l2 - fx * fx - fy * fy;
            if w >= 0.0 {
                *a *= Cx::expi(2.0 * PI * z * w.sqrt());
            } else {
                let d = (-2.0 * PI * z * (-w).sqrt()).exp();
                evanescent += a.norm_sqr() * (1.0 - d * d);
                *a *= d;
            }
        }
    }
    // The centred output coordinates: the transform is about the frame
    // origin, and `out`'s samples sit at out.coord(axis, m).
    let (mx, my) = (out.n[0], out.n[1]);
    let (x0, y0) = (out.coord(0, 0), out.coord(1, 0));
    let (dx, dy) = (out.pitch_mm[0], out.pitch_mm[1]);
    // Along fx: p[1] rows of p[0] frequencies → p[1] × mx.
    let mut rows = vec![Cx::ZERO; p[1] * mx];
    for ky in 0..p[1] {
        let line = &buf[ky * p[0]..(ky + 1) * p[0]];
        let r = crate::czt::czt_line(line, f_at(0, 0), du, -x0, -dx, mx, fft);
        rows[ky * mx..(ky + 1) * mx].copy_from_slice(&r);
    }
    // Along fy: mx columns of p[1] frequencies → my × mx.
    let mut data = vec![Cx::ZERO; my * mx];
    let mut col = vec![Cx::ZERO; p[1]];
    for i in 0..mx {
        for ky in 0..p[1] {
            col[ky] = rows[ky * mx + i];
        }
        let c = crate::czt::czt_line(&col, f_at(0, 1), dv, -y0, -dy, my, fft);
        for (m, v) in c.into_iter().enumerate() {
            data[m * mx + i] = v;
        }
    }
    let mut v = u.clone();
    v.grid = out;
    v.data = data;
    v.frame = advance(u, z);
    v.opl_ref_mm = u.opl_ref_mm + u.n_medium * z;
    let df = du * dv;
    report.ledger.evanescent = evanescent * df;
    let kept = v.power();
    report.ledger.detected = kept;
    report.ledger.left_window = (report.ledger.launched - report.ledger.evanescent - kept).max(0.0);
    report.steps.push(StepReport {
        method: Method::AngularSpectrum { band_limited: false },
        z_mm: z,
        sampling: None,
        note: format!(
            "angular spectrum of a {}×{} field (padded to {}×{}) onto a {}×{} window of pitch {:.4e} mm by chirp-z",
            n[0], n[1], p[0], p[1], mx, my, dx
        ),
    });
    report.ledger.close();
    Ok((v, report))
}

/// The plane `z` further along the field's own normal.
pub(crate) fn advance(u: &Field2, z: f64) -> oc_core::Isometry3 {
    let step = u.frame.rotate(oc_core::Vec3::new(0.0, 0.0, z));
    let mut f = u.frame;
    f.pos += step;
    f
}
```

**`asm_tilted(u, plane, out, pad)`** gives the field on any plane: tilted,
shifted and distant at once [Matsushima, Schimmel & Wyrowski 2003];
[Matsushima 2010]. A plane wave is a plane wave in any frame:
- `A(u, v)` is the source field's angular spectrum, in source-frame
  frequencies.
- `A′(u′, v′)` is the field's spectrum in the target frame.
- `t_x, t_y, t_z` are the target frame's axes expressed in the source frame.

A target-frame wave vector `(u′, v′, w′)` is the source-frame vector
`u′t_x + v′t_y + w′t_z`. Then:

```text
(u, v, w) = u′ t_x + v′ t_y + w′ t_z,   w′ = ±√(1/λ² − u′² − v′²)
A′(u′, v′) = A(u, v) · |∂(u, v)/∂(u′, v′)| · exp(i2π (u, v, w)·Δo)
```

`Δo` is the target origin seen from the source origin. The source spectrum is
read between its FFT samples by **bilinear interpolation**, which is the
method's approximation. The detector readout that calls it pads by 4 when the
field is up to 512 across, by 2 up to 1024, and by 1 beyond.

Three simplifications in the ledger:
- **Waves dropped without a name.** Plane waves that would travel back
  across the source plane (`k_z ≤ 0`), target-frame evanescent waves, and
  reads outside the source spectrum are dropped. Their power is not booked by
  name. `detected` and `left_window` are *assigned*,
  `left_window = launched − detected`, so every loss the step does not
  otherwise count falls there.
- **One obliquity.** The power through the tilted window is `Σ|U|²ΔA` times
  one obliquity, the carrier direction's. Each plane wave's own obliquity is
  not used.
- **The readout overrides it.** The tilted detector readout (§7.13) forms the
  pixel powers again, with the chief ray's obliquity `|n_det·d|`.

<!-- excerpt crates/oc-wave/src/propagate/tilted.rs 56 110 -->
*Excerpt: `crates/oc-wave/src/propagate/tilted.rs`, lines 56–110, commit `1af40e8`.*

```rust
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);

    // The target frame's axes and origin in the source frame.
    let src = u.frame;
    let ax = |v: Vec3| src.quat().inverse() * target.rotate(v);
    let (tx, ty, tz) = (ax(Vec3::X), ax(Vec3::Y), ax(Vec3::Z));
    let d = src.quat().inverse() * (target.pos - src.pos);
    // A source-frame frequency from a target-frame one: k = u′ tx + v′ ty + w′ tz.
    let to_src = |up: f64, vp: f64, wp: f64| tx * up + ty * vp + tz * wp;
    // The bilinear spectrum, at absolute source frequency (fx, fy).
    let read = |fx: f64, fy: f64| -> Cx {
        let (gx, gy) = ((fx - u0) / du + (p[0] / 2) as f64, (fy - v0) / dv + (p[1] / 2) as f64);
        if !(gx >= 0.0 && gy >= 0.0 && gx <= (p[0] - 1) as f64 && gy <= (p[1] - 1) as f64) {
            return Cx::ZERO;
        }
        let (i0, j0) = ((gx.floor() as usize).min(p[0] - 2), (gy.floor() as usize).min(p[1] - 2));
        let (s, t) = (gx - i0 as f64, gy - j0 as f64);
        let at = |i: usize, j: usize| buf[j * p[0] + i];
        at(i0, j0) * ((1.0 - s) * (1.0 - t)) + at(i0 + 1, j0) * (s * (1.0 - t)) + at(i0, j0 + 1) * ((1.0 - s) * t) + at(i0 + 1, j0 + 1) * (s * t)
    };
    // The target spectrum on a grid as fine as the source's, over the band
    // the source holds, about the source's carrier seen in the target frame.
    let c = Vec3::new(u0, v0, (inv_l2 - u0 * u0 - v0 * v0).max(0.0).sqrt());
    let ct = Vec3::new(c.dot(tx), c.dot(ty), c.dot(tz));
    let (up0, vp0) = (ct.x, ct.y);
    // A face may look back along the beam (a detector facing it): forward
    // plane waves then have w′ < 0 in its frame.
    let sgn = if tz.z >= 0.0 { 1.0 } else { -1.0 };
    let (mx, my) = (p[0], p[1]);
    let f_at = |k: usize, axis: usize| (k as f64 - (if axis == 0 { mx } else { my } / 2) as f64) * if axis == 0 { du } else { dv };
    let mut spec = vec![Cx::ZERO; mx * my];
    for ky in 0..my {
        let vp = vp0 + f_at(ky, 1);
        for kx in 0..mx {
            let up = up0 + f_at(kx, 0);
            let w2 = inv_l2 - up * up - vp * vp;
            if w2 <= 0.0 {
                continue;
            }
            let wp = sgn * w2.sqrt();
            let k = to_src(up, vp, wp);
            if k.z <= 0.0 {
                continue;
            }
            // ∂(u, v)/∂(u′, v′) with w′ = √(1/λ² − u′² − v′²).
            let (dux, duy) = (tx.x - tz.x * up / wp, ty.x - tz.x * vp / wp);
            let (dvx, dvy) = (tx.y - tz.y * up / wp, ty.y - tz.y * vp / wp);
            let jac = (dux * dvy - duy * dvx).abs();
            let phase = 2.0 * PI * (k.x * d.x + k.y * d.y + k.z * d.z);
            spec[ky * mx + kx] = read(k.x, k.y) * Cx::expi(phase) * jac;
        }
    }
```

<!-- excerpt crates/oc-wave/src/propagate/tilted.rs 134 144 -->
*Excerpt: `crates/oc-wave/src/propagate/tilted.rs`, lines 134–144, commit `1af40e8`.*

```rust
    let mut v = u.clone();
    v.grid = out;
    v.data = data;
    v.carrier = [0.0, 0.0];
    v.frame = target;
    v.opl_ref_mm = u.opl_ref_mm + u.n_medium * d.z;
    // |U|² over a tilted plane is not power: the power crossing it is the
    // irradiance times the obliquity, taken here as the carrier direction's.
    let obliquity = (ct.z * lam).abs().min(1.0);
    report.ledger.detected = v.power() * obliquity;
    report.ledger.left_window = (report.ledger.launched - report.ledger.detected).max(0.0);
```

**`fresnel_zoom(u, z, out)`** is the Fresnel integral as one transform onto
any output grid. A re-grid uses it when the plane waves' walk exceeds the
current window and `z` exceeds that window's critical distance (§7.2). That
happens mostly when a window is widened after a focus, but also when one is
narrowed:

```text
U₂ = e^{ikz} e^{ik|x₂|²/2z}/(iλz) · F{U₁ e^{ik|x₁|²/2z}}(x₂/λz)
```

It is paraxial. It warns `Undersampled` when the input chirp
`k·|x|·Δx/z` exceeds π anywhere the field is non-zero.

<!-- excerpt crates/oc-wave/src/czt.rs 212 239 -->
*Excerpt: `crates/oc-wave/src/czt.rs`, lines 212–239, commit `1af40e8`.*

```rust
    let mut chirped = u.clone();
    let g = u.grid;
    let mut step: f64 = 0.0;
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, y) = g.xy(i, j);
            let idx = g.idx(i, j);
            chirped.data[idx] *= Cx::expi(k * (x * x + y * y) / (2.0 * z_mm));
            if u.data[idx].norm_sqr() > 0.0 {
                step = step.max(k * (x.abs() * g.pitch_mm[0]).max(y.abs() * g.pitch_mm[1]) / z_mm);
            }
        }
    }
    if step > std::f64::consts::PI {
        report.warn(SamplingWarning::Undersampled { what: "the Fresnel chirp of a zoomed step".into(), max_phase_step_rad: step });
    }
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let lz = lam * z_mm;
    let f0 = [out.coord(0, 0) / lz - u0, out.coord(1, 0) / lz - v0];
    let df = [out.pitch_mm[0] / lz, out.pitch_mm[1] / lz];
    let mut data = czt2(&chirped, f0, df, out.n, fft);
    let pre = Cx::expi(k * z_mm) / Cx::new(0.0, lz);
    for j in 0..out.n[1] {
        for i in 0..out.n[0] {
            let (x, y) = out.xy(i, j);
            data[out.idx(i, j)] *= pre * Cx::expi(k * (x * x + y * y) / (2.0 * z_mm));
        }
    }
```

---

## 5. Thin elements and beams

### 5.1 Apertures with exact cell coverage

A thin element is a transmittance applied in one plane, `U → t·U`
[Goodman 2017]. For an aperture each sample's transmittance is the **exact
covered fraction `f` of its cell**:

- by interval overlap for a rectangle;
- by the closed-form chord integral for a disc or an annulus;
- by Sutherland–Hodgman clipping and the shoelace area for a polygon. Cells
  whose neighbourhood the boundary does not touch are filled by a scanline
  test.

The edge error then falls as `Δx²`. The Heurtley test (§10) measures an error
ratio of 4.04 per halving of `Δx`.

**The ledger of an edge cell.** The opaque part `(1 − f)|U|²ΔA` is booked as
`absorbed_aperture`. A cell passing the fraction `f` physically transmits
`f|U|²ΔA`, but the sample `f·U` carries only `f²|U|²ΔA`. The code books the
difference, `(f − f²)|U|²ΔA`, as `beyond_nyquist`: power the edge diffracted
into angles the grid cannot hold.

The justification: let `m(x)` be the binary mask inside a cell of area `A`.
Then:
- `∫m² = ∫m = fA`;
- the cell average carries `f²A`;
- the remainder `m − f`, of zero mean over the cell, carries `(f − f²)A`.

That remainder is attributed to angles beyond Nyquist. The bundle's section
M14 asks for the discarded projection to be reported separately, and not
called absorption (Appendix E). Whether the attribution is right is asked in
§13.

<!-- excerpt crates/oc-wave/src/elements.rs 78 293 -->
*Excerpt: `crates/oc-wave/src/elements.rs`, lines 78–293, commit `1af40e8`.*

```rust
/// `∫ √(r² − x²) dx`.
fn semicircle_integral(r: f64, x: f64) -> f64 {
    let x = x.clamp(-r, r);
    0.5 * (x * (r * r - x * x).max(0.0).sqrt() + r * r * (x / r).asin())
}

/// Area of the disc of radius `r` about the origin inside `[x0,x1]×[y0,y1]`.
///
/// The disc's vertical extent at `x` is `[−s(x), s(x)]`, `s = √(r² − x²)`, so
/// the area is `∫ max(0, min(y1, s) − max(y0, −s)) dx`. Between the
/// breakpoints where `s` crosses `y0` or `y1` the min and max pick a fixed
/// branch, and each branch integrates in closed form.
fn disc_rect_area(r: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    let a = x0.max(-r);
    let b = x1.min(r);
    if a >= b || y0 >= y1 {
        return 0.0;
    }
    let mut cuts = vec![a, b];
    for y in [y0, y1] {
        if y.abs() < r {
            let xs = (r * r - y * y).sqrt();
            for x in [-xs, xs] {
                if x > a && x < b {
                    cuts.push(x);
                }
            }
        }
    }
    cuts.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let s = |x: f64| (r * r - x * x).max(0.0).sqrt();
    let mut area = 0.0;
    for w in cuts.windows(2) {
        let (p, q) = (w[0], w[1]);
        if q <= p {
            continue;
        }
        let m = 0.5 * (p + q);
        let sm = s(m);
        let top_is_s = sm < y1;
        let bot_is_s = -sm > y0;
        let top = if top_is_s { sm } else { y1 };
        let bot = if bot_is_s { -sm } else { y0 };
        if top <= bot {
            continue;
        }
        // ∫ top dx − ∫ bot dx over [p, q].
        let int_s = semicircle_integral(r, q) - semicircle_integral(r, p);
        let int_top = if top_is_s { int_s } else { y1 * (q - p) };
        let int_bot = if bot_is_s { -int_s } else { y0 * (q - p) };
        area += int_top - int_bot;
    }
    area.max(0.0)
}

fn shoelace(v: &[[f64; 2]]) -> f64 {
    let n = v.len();
    let mut a = 0.0;
    for i in 0..n {
        let [x0, y0] = v[i];
        let [x1, y1] = v[(i + 1) % n];
        a += x0 * y1 - x1 * y0;
    }
    0.5 * a
}

/// Clip a polygon to an axis-aligned rectangle (Sutherland–Hodgman) and take
/// the shoelace area. The rectangle is convex, so the clip is exact for any
/// simple subject polygon; degenerate edges it may leave on the cell border
/// contribute zero area.
fn polygon_rect_area(v: &[[f64; 2]], x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let mut poly: Vec<[f64; 2]> = v.to_vec();
    // (axis, bound, keep_greater)
    for (axis, bound, keep_ge) in [(0, x0, true), (0, x1, false), (1, y0, true), (1, y1, false)] {
        if poly.is_empty() {
            break;
        }
        let inside = |p: &[f64; 2]| if keep_ge { p[axis] >= bound } else { p[axis] <= bound };
        let mut out = Vec::with_capacity(poly.len() + 4);
        for i in 0..poly.len() {
            let cur = poly[i];
            let prev = poly[(i + poly.len() - 1) % poly.len()];
            let (ci, pi) = (inside(&cur), inside(&prev));
            if ci != pi {
                let t = (bound - prev[axis]) / (cur[axis] - prev[axis]);
                let mut x = [prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])];
                x[axis] = bound;
                out.push(x);
            }
            if ci {
                out.push(cur);
            }
        }
        poly = out;
    }
    if poly.len() < 3 {
        0.0
    } else {
        shoelace(&poly).abs()
    }
}

/// The covered fraction of each cell of `grid` by `shape`, exactly: the closed
/// forms above for a rectangle, a disc and an annulus, and for a polygon
/// [`polygon_coverage`].
pub fn coverage(grid: &Grid2, shape: &Shape) -> Result<Vec<f64>, WaveError> {
    shape.validate()?;
    if let Shape::Polygon(v) = shape {
        return Ok(polygon_coverage(grid, v));
    }
    let [dx, dy] = grid.pitch_mm;
    let cell = dx * dy;
    let mut t = Vec::with_capacity(grid.len());
    for j in 0..grid.n[1] {
        for i in 0..grid.n[0] {
            let (x, y) = grid.xy(i, j);
            t.push((shape.cell_overlap(x - 0.5 * dx, x + 0.5 * dx, y - 0.5 * dy, y + 0.5 * dy) / cell).clamp(0.0, 1.0));
        }
    }
    Ok(t)
}

/// The covered fraction of each cell of `grid` by a simple polygon (convex or
/// not, either orientation), exactly and in time that grows with the
/// boundary rather than the area. The cells within a cell of the boundary
/// take the polygon's area clipped to them; every other cell lies wholly on
/// one side, which its centre's side says, found by crossing each row's
/// centre line with the edges.
pub fn polygon_coverage(grid: &Grid2, poly: &[[f64; 2]]) -> Vec<f64> {
    let [nx, ny] = grid.n;
    let [dx, dy] = grid.pitch_mm;
    let mut f = vec![0.0; nx * ny];
    let n = poly.len();
    if n < 3 {
        return f;
    }
    // Cell (i, j) spans [ex + i·dx, ex + (i + 1)·dx] × [ey + j·dy, ey + (j + 1)·dy].
    let ex = grid.coord(0, 0) - 0.5 * dx;
    let ey = grid.coord(1, 0) - 0.5 * dy;
    // The window, two cells wider each way: an edge is walked only there.
    let (wx0, wx1) = (ex - 2.0 * dx, ex + (nx as f64 + 2.0) * dx);
    let (wy0, wy1) = (ey - 2.0 * dy, ey + (ny as f64 + 2.0) * dy);
    let mut edge = vec![false; nx * ny];
    let step = 0.25 * dx.min(dy);
    for k in 0..n {
        let (p, q) = (poly[k], poly[(k + 1) % n]);
        // Liang–Barsky: the part of the edge inside the widened window.
        let (ddx, ddy) = (q[0] - p[0], q[1] - p[1]);
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let mut keep = true;
        for (pp, qq) in [(-ddx, p[0] - wx0), (ddx, wx1 - p[0]), (-ddy, p[1] - wy0), (ddy, wy1 - p[1])] {
            if pp == 0.0 {
                if qq < 0.0 {
                    keep = false;
                }
            } else {
                let r = qq / pp;
                if pp < 0.0 {
                    t0 = t0.max(r);
                } else {
                    t1 = t1.min(r);
                }
            }
        }
        if !keep || t0 > t1 {
            continue;
        }
        let len = (ddx * ddx + ddy * ddy).sqrt() * (t1 - t0);
        let m = (len / step).ceil().max(1.0) as usize;
        for s in 0..=m {
            let t = t0 + (t1 - t0) * s as f64 / m as f64;
            let (x, y) = (p[0] + t * ddx, p[1] + t * ddy);
            let (ci, cj) = (((x - ex) / dx).floor() as i64, ((y - ey) / dy).floor() as i64);
            for j in cj - 1..=cj + 1 {
                for i in ci - 1..=ci + 1 {
                    if i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny {
                        edge[j as usize * nx + i as usize] = true;
                    }
                }
            }
        }
    }
    let mut xs: Vec<f64> = Vec::new();
    for j in 0..ny {
        let yc = ey + (j as f64 + 0.5) * dy;
        xs.clear();
        for k in 0..n {
            let (a, b) = (poly[k], poly[(k + 1) % n]);
            if (a[1] > yc) != (b[1] > yc) {
                xs.push(a[0] + (yc - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks_exact(2) {
            // The cells whose centre ex + (i + ½)dx lies in [left, right).
            let i0 = ((pair[0] - ex) / dx - 0.5).ceil().clamp(0.0, nx as f64) as usize;
            let i1 = ((pair[1] - ex) / dx - 0.5).ceil().clamp(0.0, nx as f64) as usize;
            for v in &mut f[j * nx + i0..j * nx + i1.max(i0)] {
                *v = 1.0;
            }
        }
    }
    let cell = dx * dy;
    for j in 0..ny {
        for i in 0..nx {
            if edge[j * nx + i] {
                let (x0, y0) = (ex + i as f64 * dx, ey + j as f64 * dy);
                f[j * nx + i] = (polygon_rect_area(poly, x0, x0 + dx, y0, y0 + dy) / cell).clamp(0.0, 1.0);
            }
        }
    }
    f
}
```

<!-- excerpt crates/oc-wave/src/elements.rs 330 348 -->
*Excerpt: `crates/oc-wave/src/elements.rs`, lines 330–348, commit `1af40e8`.*

```rust
/// Multiply `u` by the covered fractions `t`, one per sample, and book what
/// that does. A cell passing a fraction `t` of its area transmits `t·|U|²ΔA`;
/// the opaque rest, `(1 − t)·|U|²ΔA`, is returned as the first number. The
/// sample `t·U` carries only `t²·|U|²ΔA`: the edge diffracted the
/// difference, the second number, beyond the grid's Nyquist band.
pub fn apply_coverage(u: &mut Field2, t: &[f64]) -> Result<(f64, f64), WaveError> {
    if t.len() != u.data.len() {
        return Err(WaveError::Mismatch("mask and field differ in size".into()));
    }
    let da = u.grid.cell_area_mm2();
    let (mut stopped, mut beyond) = (0.0, 0.0);
    for (z, ti) in u.data.iter_mut().zip(t) {
        let i = z.norm_sqr() * da;
        stopped += (1.0 - ti) * i;
        beyond += (ti - ti * ti) * i;
        *z *= *ti;
    }
    Ok((stopped, beyond))
}
```

**A disc seen obliquely** becomes a polygon. Its circumradius is
`r√(θ/sin θ)`, `θ = 2π/n`, so the polygon has the disc's own area. The number
of sides is the smallest that keeps the polygon within a five-hundredth of a
sample of the circle, clamped to 64–1024:

<!-- excerpt crates/oc-wave/src/elements.rs 295 316 -->
*Excerpt: `crates/oc-wave/src/elements.rs`, lines 295–316, commit `1af40e8`.*

```rust
/// A regular polygon of `n` sides about `c` with the area of the disc of
/// radius `r`: its circumradius is `r·√(θ / sin θ)`, `θ = 2π/n`, so its edges
/// cross the circle and depart from it by at most about `r·(π/n)²/2`.
pub fn disc_polygon(c: [f64; 2], r: f64, n: usize) -> Vec<[f64; 2]> {
    let n = n.max(8);
    let th = 2.0 * PI / n as f64;
    let rr = r * (th / th.sin()).sqrt();
    (0..n)
        .map(|k| {
            let a = k as f64 * th;
            [c[0] + rr * a.cos(), c[1] + rr * a.sin()]
        })
        .collect()
}

/// Sides for a disc of radius `r` drawn as a polygon on a grid of `pitch`:
/// enough that the polygon departs from the circle by under a five-hundredth
/// of a sample, at most 1024.
pub fn disc_sides(r: f64, pitch: f64) -> usize {
    let tol = 2e-3 * pitch;
    ((PI * (r.abs() / (2.0 * tol)).sqrt()).ceil() as usize).clamp(64, 1024)
}
```

### 5.2 Thin lens, phase screen, transmittance

`apply_thin_lens` multiplies by `exp(−ik|r − c|²/2f)`. It warns
`Undersampled` when the chirp's phase step between neighbours at the window
edge exceeds π. That step is `k·r_max·Δx/|f|`, with `r_max` the largest
`max(|x − c_x|, |y − c_y|)` over the window and `Δx` the larger pitch.

The wave path does **not** use `apply_thin_lens`. Its thin lens is applied in
the hand-over (§7.3, `flat_element`), with its own undersampling check at the
window's half-diagonal. `apply_phase` multiplies by `e^{iφ}`.
`apply_transmittance` refuses `|t| > 1` and books the loss as
`absorbed_element`.

<!-- excerpt crates/oc-wave/src/elements.rs 394 421 -->
*Excerpt: `crates/oc-wave/src/elements.rs`, lines 394–421, commit `1af40e8`.*

```rust
pub fn apply_thin_lens(
    u: &mut Field2,
    f_mm: f64,
    c: [f64; 2],
    report: &mut WaveReport,
) -> Result<(), WaveError> {
    if !(f_mm.is_finite() && f_mm != 0.0) {
        return Err(WaveError::OutOfRange(format!("focal length {f_mm} mm")));
    }
    let k = u.k_mm();
    let g = u.grid;
    let mut rmax: f64 = 0.0;
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, y) = g.xy(i, j);
            let (dx, dy) = (x - c[0], y - c[1]);
            rmax = rmax.max(dx.abs().max(dy.abs()));
            let idx = g.idx(i, j);
            u.data[idx] *= Cx::expi(-k * (dx * dx + dy * dy) / (2.0 * f_mm));
        }
    }
    let step = k * rmax * g.pitch_mm[0].max(g.pitch_mm[1]) / f_mm.abs();
    if step > PI {
        report.warn(SamplingWarning::Undersampled {
            what: format!("thin lens f = {f_mm} mm"),
            max_phase_step_rad: step,
        });
    }
```

### 5.3 Beams

**Gaussian beam** [Kogelnik & Li 1966], written in the `exp(−iωt)`
convention (they use `exp(+iωt)`, which conjugates every phase):

```text
U(r, z) = A · (q₀/q) · exp(ik r²/2q) · e^{ikz},   q = z − i z_R,  q₀ = −i z_R,
z_R = π w₀² n/λ₀,   A = √(2P/(πw₀²))
```

It is a solution of the paraxial wave equation, not of the Helmholtz
equation. An exact propagator therefore moves it differently by a relative
amount of order `(λ/πw₀)²`.

<!-- excerpt crates/oc-wave/src/beams.rs 80 91 -->
*Excerpt: `crates/oc-wave/src/beams.rs`, lines 80–91, commit `1af40e8`.*

```rust
    /// The complex amplitude at `(x, y)`, `z_mm` from the waist.
    pub fn value(&self, x: f64, y: f64, z_mm: f64) -> Cx {
        let k = 2.0 * PI * self.n / (self.lambda_um / 1000.0);
        let zr = self.rayleigh_range_mm();
        let q = Cx::new(z_mm, -zr);
        let q0 = Cx::new(0.0, -zr);
        let r2 = (x - self.center[0]).powi(2) + (y - self.center[1]).powi(2);
        let a = (2.0 * self.power / (PI * self.w0_mm * self.w0_mm)).sqrt();
        // exp(ik r²/2q) = exp(i k r² / 2 · 1/q).
        let arg = Cx::new(0.0, k * r2 / 2.0) * (Cx::ONE / q);
        (q0 / q) * arg.exp() * Cx::expi(k * z_mm) * a
    }
```

**Plane wave**: uniform amplitude, its tilt held entirely as the carrier.

**Point source**: a point a distance `z` behind the plane, on the axis
through `c`, gives `exp(ik(r − z))·z/r` with `r = √(|ρ − c|² + z²)`. This is
the spherical wave `e^{ikr}/r`, scaled to unit amplitude on the axis. The
factor `e^{−ikz}` sets the axis sample's phase to zero.

**Launch on the wave path.**
- A Gaussian is sampled at `z = −waist_offset` from its waist.
- A top-hat of radius `a` is a disc of amplitude `√(P/πa²)`, weighted by
  exact cell coverage.
- The launch books what the samples do not hold:
  - a Gaussian's tail beyond the window as `left_window`;
  - a top-hat's shortfall as `beyond_nyquist`. That shortfall is the edge's
    `f − f²`, but also any part of the disc outside the window;
  - a sampled Gaussian holding slightly *more* than `P` as negative
    `resampling`.

The **reference Gaussian parameter** the path carries (§7.6) uses
Kogelnik and Li's sign, `q = z + i z_R`, with `Im q > 0`. `beams.rs` writes
the field with `q = z − i z_R`, its complex conjugate, as the `exp(−iωt)`
convention needs. Both have `Re(1/q) = z/(z² + z_R²) = 1/R(z)`, the
curvature, positive when diverging. A **top-hat's reference is the Gaussian
whose waist radius equals the disc's radius**, with its waist at the launch
plane.

<!-- excerpt crates/oc-wave/src/path.rs 913 955 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 913–955, commit `1af40e8`.*

```rust
/// The Gaussian parameter of the declared beam at the launch plane, in a
/// medium of index `n0`: a Gaussian's own, a top-hat's as the Gaussian of its
/// radius with its waist there (H. Kogelnik, T. Li, *Appl. Opt.* 5, 1550
/// (1966)).
fn launch_q(spec: &WavePathSpec, lam_um: f64, n0: f64) -> Cx {
    let lam = lam_um / 1000.0 / n0;
    match spec.beam {
        BeamProfile::Gaussian { waist_radius_mm, waist_offset_mm } => Cx::new(-waist_offset_mm, std::f64::consts::PI * waist_radius_mm * waist_radius_mm / lam),
        BeamProfile::TopHat { radius_mm } => Cx::new(0.0, std::f64::consts::PI * radius_mm * radius_mm / lam),
    }
}

/// The field a source launches on `grid`, with the ledger of the launch.
fn launch_field(spec: &WavePathSpec, grid: Grid2, lam: f64, n0: f64, power: f64, beam: &Beam) -> Result<(Field2, WaveLedger), WaveError> {
    let mut u = match spec.beam {
        BeamProfile::Gaussian { waist_radius_mm, waist_offset_mm } => {
            GaussianBeam { w0_mm: waist_radius_mm, lambda_um: lam, n: n0, power, center: [0.0, 0.0] }.sample(grid, -waist_offset_mm)?
        }
        BeamProfile::TopHat { radius_mm } => {
            // The disc's edge weighted by exact cell coverage, as an
            // aperture's is.
            let a = (power / (std::f64::consts::PI * radius_mm * radius_mm)).sqrt();
            let cover = crate::elements::coverage(&grid, &crate::elements::Shape::Disc { c: [0.0, 0.0], r: radius_mm })?;
            let mut u = Field2::zeros(grid, lam)?.in_medium(n0)?;
            for (z, c) in u.data.iter_mut().zip(&cover) {
                *z = Cx::real(a * c);
            }
            u
        }
    };
    u.frame = beam.iso();
    // The source launches its power; what the samples do not hold is the
    // Gaussian's tail beyond the window, or the top-hat's edge diffracted
    // beyond the grid's band.
    let mut ledger = WaveLedger { launched: power, ..Default::default() };
    let short = power - u.power();
    match spec.beam {
        BeamProfile::TopHat { .. } => ledger.beyond_nyquist += short,
        BeamProfile::Gaussian { .. } if short >= 0.0 => ledger.left_window += short,
        BeamProfile::Gaussian { .. } => ledger.resampling += short,
    }
    Ok((u, ledger))
}
```

---

## 6. Pixelated modulators: DMD and LCoS

The devices are described in `oc-core` (`SpatialModulator`: pixel counts, a
square pitch `p`, the kind, the pattern). Their periodic patterns have
closed-form orders there. `oc-wave` lays a device into a sampled field, or
computes a whole device's far field by factorization.

**Frames.** A device lies in its local `XY` plane, facing `+Z`. Light
arrives travelling toward `−Z`, and reflected orders leave toward `+Z`.
Pixel `(i, j)` is centred at `((i − (Nₓ−1)/2)p, (j − (Nᵧ−1)/2)p)`.

### 6.1 DMD geometry and its thin-element reflection

Each pixel holds a square mirror of side `a = ρp`, where `ρ` is the mirror
fill (side over pitch). The pixel's state sets its tilt about the hinge
(`DmdState::sign()` is +1, −1 and 0):
- **ON**: `+θ`;
- **OFF**: `−θ`;
- **flat** (parked): no tilt.

The hinge is the pixel diagonal on most TI devices, and a pixel side on the
5.4 µm devices [TI DLPA083B §4.2]. `hinge()` returns the hinge direction
`â` and the in-plane direction across it, `ŵ = ẑ × â`. The facet normal is
`+Z` rotated by the signed tilt about the hinge: `n = cos θ ẑ + sin θ (â × ẑ)`.

The **thin-element model** follows the pixel response of [Popoff et al.
2024, App. A], in its 2D, any-incidence form. A mirror reflects the incident
wave as its tilted plane would, restricted to its outline **projected onto
the array plane along the array normal**, with the phase zero at the
mirror's pivot:

```text
T(r) = √R · T_w · Σ_mirrors mask_s(r − r_ij) · exp(ik (t_spec,s − t_in)·(r − r_ij))
```

- `t` is a direction's tangential part (direction cosines in the array
  plane).
- `t_spec,s` is the incident direction reflected off the normal of state `s`.
- `R` is the mirror's reflectance, and `T_w` the window's single-pass power
  transmittance (two passes make `T_w` in amplitude).

The projected outline is the square compressed by `cos θ` across the hinge.
Its transform is:

```text
M̂(f) = cos θ · a² · sinc(a(Af)ₓ) · sinc(a(Af)ᵧ),   A = ââᵀ + cos θ ŵŵᵀ
```

with `ŵ` the in-plane direction across the hinge and `sinc(x) = sin(πx)/(πx)`.

Not modelled:
- a mirror shading its neighbour at oblique incidence;
- the edges, the hinge and the mirror's curvature;
- the gap between a facet and the array plane, beyond the phase above;
- thermal drift.

<!-- excerpt crates/oc-core/src/modulator.rs 549 596 -->
*Excerpt: `crates/oc-core/src/modulator.rs`, lines 549–596, commit `1af40e8`.*

```rust
/// The hinge axis and the in-plane direction across it.
pub fn hinge(axis: TiltAxis) -> (Vec3, Vec3) {
    match axis {
        TiltAxis::Diagonal => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            (Vec3::new(s, s, 0.0), Vec3::new(-s, s, 0.0))
        }
        TiltAxis::Side => (Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
    }
}

/// A mirror's unit normal in `state`: `+Z` rotated by the signed tilt about
/// the hinge (Rodrigues; `ẑ ⊥ â`, so `R ẑ = cos θ ẑ + sin θ (â × ẑ)`).
pub fn facet_normal(d: &DmdSpec, state: DmdState) -> Vec3 {
    let th = state.sign() * d.tilt_deg.to_radians();
    let (a, _) = hinge(d.tilt_axis);
    let z = Vec3::Z;
    z * th.cos() + a.cross(z) * th.sin()
}

/// Mirror reflection of a direction about a unit normal.
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - n * (2.0 * d.dot(n))
}

/// The Fourier transform of a mirror's outline projected onto the array
/// plane, at spatial frequency `f` (cycles/mm): `cos θ · a² · sinc(a(Af)ₓ)
/// sinc(a(Af)ᵧ)`, `a` the mirror side in mm (see the module docs).
pub fn mirror_ft(d: &DmdSpec, pitch_mm: f64, state: DmdState, f: [f64; 2]) -> f64 {
    let th = state.sign() * d.tilt_deg.to_radians();
    let a = d.mirror_fill * pitch_mm;
    let (h, w) = hinge(d.tilt_axis);
    let c = th.cos();
    // A f = â(â·f) + cos θ ŵ(ŵ·f); A is symmetric.
    let fa = h.x * f[0] + h.y * f[1];
    let fw = w.x * f[0] + w.y * f[1];
    let af = [h.x * fa + c * w.x * fw, h.y * fa + c * w.y * fw];
    c * a * a * sinc(a * af[0]) * sinc(a * af[1])
}

/// The normalized sinc, `sin(πx)/(πx)`.
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        1.0 - (PI * x).powi(2) / 6.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}
```

### 6.2 Closed-form orders of a periodic pattern

A pattern can be periodic over a unit cell of `Pₓ × Pᵧ` pixels, with periods
`Lₓ = Pₓp`, `Lᵧ = Pᵧp` and cell area `A_cell = LₓLᵧ`. The Fourier series of
`T` is then exact within the thin-element model. The coefficient of order
`(m, n)` is:

```text
c_mn = (√R T_w / A_cell) Σ_cell e^{−i2π G_mn·r_ij} · M̂_s(G_mn − (t_spec,s − t_in)/λ),
G_mn = (m/Lₓ, n/Lᵧ)
```

Order `(m, n)` leaves along `t_in + λ(m/Lₓ, n/Lᵧ)` in direction cosines
[Harvey & Vernold 1998], and it propagates when that tangential part is
inside the unit circle. The sum over *all* orders follows from Parseval on
the cell:

```text
Σ_all |c|² = R·T_w² · Σ_cell M̂_s(0) / A_cell
```

For a uniform ON pattern this is `R T_w² ρ² cos θ`.

There are two order models:
- **`Tea`** (the thin-element approximation): `η = |c_mn|²` as it is. The
  share the evanescent orders would hold is lost.
- **`HarveyRenormalized`** (the default): `η = |c_mn|² · Σ_all|c|² /
  Σ_prop|c|²`. The propagating orders share the evanescent orders' power in
  proportion. This is the non-paraxial scalar treatment of
  [Harvey et al. 2011], which TI cites for its DMD efficiencies
  [TI DLPA083B].

The excerpt also holds the LCoS branch, which §6.5 explains. `LcosBeam`
selects the director component (`Modulated`) or the orthogonal one
(`Unmodulated`). `lut.phase(g)` is the table's phase for grey level `g`, and
`g` here is the dead space's complex reflectance,
`gap_amplitude·e^{i·gap_phase_rad}`.

<!-- excerpt crates/oc-core/src/modulator.rs 796 911 -->
*Excerpt: `crates/oc-core/src/modulator.rs`, lines 796–911, commit `1af40e8`.*

```rust
    let lam = lambda_um / 1000.0;
    let p = device.pitch_mm();
    let [px, py] = cell.period;
    let (lx, ly) = (px as f64 * p, py as f64 * p);
    let t_in = [d_in.x, d_in.y];

    // Coefficient of order (m, n), and the analytic Σ|c|².
    let coeff: Box<dyn Fn(i32, i32) -> Cx>;
    let total: f64;
    match &device.kind {
        ModulatorKind::Dmd(d) => {
            let amp = d.reflectance.sqrt() * d.window_transmittance;
            let states: Vec<DmdState> = cell.values.iter().map(|v| DmdState::from_value(*v)).collect();
            // Specular tangential direction per state.
            let spec = |s: DmdState| {
                let r = reflect(d_in, facet_normal(d, s));
                [r.x, r.y]
            };
            let dd = d.clone();
            let specs: Vec<[f64; 2]> = states.iter().map(|s| spec(*s)).collect();
            let cell_area = lx * ly;
            total = amp * amp
                * states
                    .iter()
                    .map(|s| mirror_ft(&dd, p, *s, [0.0, 0.0]))
                    .sum::<f64>()
                / cell_area;
            coeff = Box::new(move |m, n| {
                let g = [m as f64 / lx, n as f64 / ly];
                let mut c = Cx::ZERO;
                for jj in 0..py {
                    for ii in 0..px {
                        let k = jj * px + ii;
                        let s = states[k];
                        let ts = specs[k];
                        let f = [g[0] - (ts[0] - t_in[0]) / lam, g[1] - (ts[1] - t_in[1]) / lam];
                        let r = [ii as f64 * p, jj as f64 * p];
                        c += Cx::expi(-2.0 * PI * (g[0] * r[0] + g[1] * r[1])) * mirror_ft(&dd, p, s, f);
                    }
                }
                c * (amp / cell_area)
            });
        }
        ModulatorKind::Lcos(l) => {
            if l.fringing_um > 0.0 {
                return err("an LCoS with fringing has no closed-form orders; sample it in oc-wave");
            }
            let rho = l.pixel_fill;
            let g = Cx::from_polar(l.gap_amplitude, l.gap_phase_rad);
            let scale = l.design_lambda_um / lambda_um;
            let pix: Vec<Cx> = cell
                .values
                .iter()
                .map(|v| {
                    let phi = match beam {
                        LcosBeam::Modulated => l.lut.phase(*v) * scale,
                        LcosBeam::Unmodulated => 0.0,
                    };
                    Cx::from_polar(l.reflectance.sqrt(), phi) - g
                })
                .collect();
            let nc = (px * py) as f64;
            total = rho * rho * l.reflectance + (1.0 - rho * rho) * l.gap_amplitude * l.gap_amplitude;
            coeff = Box::new(move |m, n| {
                let mut s = Cx::ZERO;
                for jj in 0..py {
                    for ii in 0..px {
                        let ph = -2.0 * PI * (m as f64 * ii as f64 / px as f64 + n as f64 * jj as f64 / py as f64);
                        s += pix[jj * px + ii] * Cx::expi(ph);
                    }
                }
                let mut c = s * (rho * rho * sinc(rho * m as f64 / px as f64) * sinc(rho * n as f64 / py as f64) / nc);
                if m == 0 && n == 0 {
                    c += g;
                }
                c
            });
        }
    }

    // Every order whose direction cosines stay inside the unit circle.
    let mmax = ((1.0 + t_in[0].abs()) * lx / lam).ceil() as i32 + 1;
    let nmax = ((1.0 + t_in[1].abs()) * ly / lam).ceil() as i32 + 1;
    let mut orders = Vec::new();
    let mut propagating = 0.0;
    for n in -nmax..=nmax {
        for m in -mmax..=mmax {
            let t = [t_in[0] + lam * m as f64 / lx, t_in[1] + lam * n as f64 / ly];
            let s2 = t[0] * t[0] + t[1] * t[1];
            if s2 >= 1.0 {
                continue;
            }
            let c = coeff(m, n);
            propagating += c.norm_sqr();
            orders.push(Order {
                m,
                n,
                dir: Vec3::new(t[0], t[1], (1.0 - s2).sqrt()),
                amplitude: c,
                efficiency: c.norm_sqr(),
            });
        }
    }
    if model == OrderModel::HarveyRenormalized && propagating > 0.0 {
        let k = total / propagating;
        for o in &mut orders {
            o.efficiency *= k;
        }
    }
    Ok(OrderTable {
        orders,
        total,
        propagating,
        model,
    })
}
```

### 6.3 DMD sampled near field (`dmd_reflect`)

A field on the device plane, whose carrier is the incident direction's
tangential part, is multiplied sample by sample. Each sample sums, over the
pixels that overlap its cell, the **exact area fraction of the projected
mirror outline** in that cell.

Each state reflects into its own direction, so a state is a **family**. The
family the chain follows has its specular direction `t_spec` held as the
output field's carrier. The samples then carry only a piston per mirror,
`exp(−ik(t_spec − t_in)·r_ij)`, and the in-mirror ramp is carried by the
carrier.

Without the carrier, a 7.56 µm mirror tilted 12° at 0.4 µm would need about
thirty samples per pixel to resolve its ramp. With it, four samples hold the
phase. The sampled far field converges at about second order (slope −1.87),
but from a large error, because of the mirror edges: 6.0e-2 at 4 samples per
pixel, 3.4e-3 at 8 and 8.4e-4 at 32 (§10.2, §11.4). The test calls these
"samples per mirror", but they are per pixel pitch (`Δx = p/per`).

`incident_direction(carrier)` is `(α, β, −√(1 − α² − β²))`, travelling
toward `−Z`. In `mirror_outline`, `a` is the mirror's **half**-side,
`ρp/2`.

The ledger, with `R_eff = R·T_w²` (`refl` in the code):
- `absorbed_element` takes the dead space, the light outside the device's
  pixel array, and `1 − R_eff` of what the mirrors meet (the mirror and two
  window passes);
- the other families' reflected power goes to `other_family`;
- `R_eff·cov_fam·|U|²ΔA − |v|²ΔA` goes to `beyond_nyquist`. As for an
  aperture, a sample spanning a mirror edge carries less than the cell
  reflects. The sum is clipped at zero, so a negative total (a sample
  carrying more than its cell reflects) falls into `numerical`.

<!-- excerpt crates/oc-wave/src/modulator.rs 76 103 -->
*Excerpt: `crates/oc-wave/src/modulator.rs`, lines 76–103, commit `1af40e8`.*

```rust
/// The outline of a mirror in `state`, projected onto the array plane,
/// relative to its pixel centre, mm.
fn mirror_outline(d: &DmdSpec, pitch_mm: f64, state: DmdState) -> Shape {
    let a = 0.5 * d.mirror_fill * pitch_mm;
    let n = facet_normal(d, state);
    // Project the tilted square: rotate each corner as the mirror rotates and
    // drop z. The rotation maps ẑ to n about the hinge; corners on the hinge
    // stay put, corners across it shrink by cos θ.
    let th = (n.z).clamp(-1.0, 1.0).acos();
    let c = th.cos();
    let (h, w) = match d.tilt_axis {
        oc_core::modulator::TiltAxis::Diagonal => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            ([s, s], [-s, s])
        }
        oc_core::modulator::TiltAxis::Side => ([0.0, 1.0], [-1.0, 0.0]),
    };
    let corners = [[-a, -a], [a, -a], [a, a], [-a, a]];
    let v = corners
        .iter()
        .map(|p| {
            let ph = p[0] * h[0] + p[1] * h[1];
            let pw = p[0] * w[0] + p[1] * w[1];
            [h[0] * ph + c * w[0] * pw, h[1] * ph + c * w[1] * pw]
        })
        .collect();
    Shape::Polygon(v)
}
```

<!-- excerpt crates/oc-wave/src/modulator.rs 119 196 -->
*Excerpt: `crates/oc-wave/src/modulator.rs`, lines 119–196, commit `1af40e8`.*

```rust
pub fn dmd_reflect(
    u: &Field2,
    device: &SpatialModulator,
    family: DmdState,
) -> Result<(Field2, WaveReport), WaveError> {
    device.validate().map_err(me)?;
    let d = dmd_spec(device)?;
    let d_in = incident_direction(u.carrier)?;
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    let amp = d.reflectance.sqrt() * d.window_transmittance;
    let refl = amp * amp;
    let k = u.k_mm();
    let t_in = [d_in.x, d_in.y];
    let spec = reflect(d_in, facet_normal(d, family));
    let t_s = [spec.x, spec.y];
    let outlines = [DmdState::Off, DmdState::On, DmdState::Flat].map(|s| (s, mirror_outline(d, p, s)));
    let outline = |s: DmdState| &outlines.iter().find(|(t, _)| *t == s).expect("all states").1;

    let g = u.grid;
    let [dx, dy] = g.pitch_mm;
    let da = dx * dy;
    let mut out = u.clone();
    let (mut absorbed, mut other, mut beyond) = (0.0, 0.0, 0.0);
    for j in 0..g.n[1] {
        let y = g.coord(1, j);
        let (y0, y1) = (y - 0.5 * dy, y + 0.5 * dy);
        for i in 0..g.n[0] {
            let x = g.coord(0, i);
            let (x0, x1) = (x - 0.5 * dx, x + 0.5 * dx);
            let idx = g.idx(i, j);
            let irr = u.data[idx].norm_sqr() * da;
            let mut acc = Cx::ZERO;
            let (mut cov_fam, mut cov_all) = (0.0, 0.0);
            for pj in pixel_range(y0, y1, ny, p) {
                for pi in pixel_range(x0, x1, nx, p) {
                    let c = device.pixel_center_mm(pi, pj);
                    let state = DmdState::from_value(device.value(pi, pj));
                    let cov = outline(state).cell_overlap(x0 - c[0], x1 - c[0], y0 - c[1], y1 - c[1]) / da;
                    if cov <= 0.0 {
                        continue;
                    }
                    cov_all += cov;
                    if state == family {
                        cov_fam += cov;
                        let piston = -k * ((t_s[0] - t_in[0]) * c[0] + (t_s[1] - t_in[1]) * c[1]);
                        acc += Cx::expi(piston) * cov;
                    }
                }
            }
            let v = u.data[idx] * acc * amp;
            out.data[idx] = v;
            absorbed += irr * ((1.0 - cov_all) + (1.0 - refl) * cov_all);
            other += irr * refl * (cov_all - cov_fam);
            beyond += irr * refl * cov_fam - v.norm_sqr() * da;
        }
    }
    out.carrier = t_s;
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();
    report.ledger.absorbed_element = absorbed;
    report.ledger.other_family = other;
    report.ledger.beyond_nyquist = beyond.max(0.0);
    report.ledger.detected = out.power();
    report.ledger.close();
    validity_flags(&mut report, p, u.lambda_um);
    report.approx("DMD as a thin element: mirror shading, edges, hinges and curvature not modelled");
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!(
            "DMD {}×{} at {} µm, {:?} family, carrier ({:.6}, {:.6})",
            nx, ny, device.pitch_um, family, t_s[0], t_s[1]
        ),
    });
    Ok((out, report))
}
```

**`ScalarLimit`** is flagged when `p < 14λ`. [Pommet, Moharam & Grann 1994]
find scalar theory's efficiency error above ±5% for features below about 14λ.
A 7.56 µm pitch is 14λ at 0.54 µm.

### 6.4 Factorized far field of a whole device

Every mirror of one state has the same response, so the far field is one
mirror's transform times the lattice sum over that state's mirrors:

```text
Û(f) = Σ_states √R T_w · M̂_s(f − t_s/λ) · Σ_{ij ∈ s} U_in(r_ij) e^{−i2πf·r_ij}
```

The lattice sum is a chirp-z transform at one sample per pixel. The result is
on a grid of direction cosines, and `|Û|²/λ²` is power per unit `dα dβ`.

The factorization assumes the incident field is a plane wave across each
pixel. It is exact within the thin-element model for a plane wave.
`dmd_far_field` takes only a plane wave, `U_in(r_ij) = e^{ik t_in·r_ij}`
with unit amplitude, so the general `U_in` above is the module's statement of
the method, not an input of this function.

<!-- excerpt crates/oc-wave/src/modulator.rs 401 485 -->
*Excerpt: `crates/oc-wave/src/modulator.rs`, lines 401–485, commit `1af40e8`.*

```rust
/// The lattice sum `Σ_ij v_ij e^{−i2πf·r_ij}` over the device's pixel centres
/// on a frequency grid, by chirp-z.
fn lattice_sum(
    device: &SpatialModulator,
    values: Vec<Cx>,
    f0: [f64; 2],
    df: [f64; 2],
    n: [usize; 2],
    fft: &mut Fft2,
) -> Result<Vec<Cx>, WaveError> {
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    // A Grid2 whose samples sit on the pixel centres: for even counts the
    // grid's centre sample is half a pitch off the device centre.
    let cx = if nx % 2 == 0 { -0.5 * p } else { 0.0 };
    let cy = if ny % 2 == 0 { -0.5 * p } else { 0.0 };
    let g = Grid2::new([nx, ny], [p, p], [-cx, -cy])?;
    let mut lat = Field2::zeros(g, 1.0)?;
    debug_assert!((g.coord(0, 0) - device.pixel_center_mm(0, 0)[0]).abs() < 1e-12 * p.max(1.0));
    lat.data = values;
    let mut out = czt2(&lat, f0, df, n, fft);
    let inv = 1.0 / (p * p);
    for v in &mut out {
        *v *= inv;
    }
    Ok(out)
}

/// The far field of a whole DMD under a plane wave of unit amplitude
/// travelling along `d_in`, on a grid of direction cosines, by the factorized
/// transform (see the module docs). `states` limits which mirror families
/// contribute (all three for the whole field).
pub fn dmd_far_field(
    device: &SpatialModulator,
    lambda_um: f64,
    d_in: Vec3,
    grid: AngularGrid,
    states: &[DmdState],
    fft: &mut Fft2,
) -> Result<AngularField, WaveError> {
    let AngularGrid { start, step, n } = grid;
    device.validate().map_err(me)?;
    let d = dmd_spec(device)?;
    let lam = lambda_um / 1000.0;
    let k = 2.0 * PI / lam;
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    let amp = d.reflectance.sqrt() * d.window_transmittance;
    let f0 = [start[0] / lam, start[1] / lam];
    let df = [step[0] / lam, step[1] / lam];
    let mut total = vec![Cx::ZERO; n[0] * n[1]];
    for &s in states {
        let spec = reflect(d_in, facet_normal(d, s));
        let ts = [spec.x, spec.y];
        let mut vals = vec![Cx::ZERO; nx * ny];
        let mut any = false;
        for j in 0..ny {
            for i in 0..nx {
                if DmdState::from_value(device.value(i, j)) == s {
                    let c = device.pixel_center_mm(i, j);
                    vals[j * nx + i] = Cx::expi(k * (d_in.x * c[0] + d_in.y * c[1]));
                    any = true;
                }
            }
        }
        if !any {
            continue;
        }
        let b = lattice_sum(device, vals, f0, df, n, fft)?;
        for jj in 0..n[1] {
            for ii in 0..n[0] {
                let f = [f0[0] + ii as f64 * df[0], f0[1] + jj as f64 * df[1]];
                let m = mirror_ft(d, p, s, [f[0] - ts[0] / lam, f[1] - ts[1] / lam]);
                total[jj * n[0] + ii] += b[jj * n[0] + ii] * (amp * m);
            }
        }
    }
    Ok(AngularField {
        start,
        step,
        n,
        lambda_mm: lam,
        data: total,
    })
}
```

**Projector efficiency** follows TI's method [TI DLPA083B §4.2]:
- an incoherent sum of plane waves over the illumination cone;
- each plane wave's orders, under `HarveyRenormalized`, collected inside the
  projection lens's cone about `+Z`;
- the result as a share of the reflected power.

The cone is an f-number about the direction that the ON mirrors send along
the normal (`on_axis_illumination`: `ẑ` reflected off the ON facet). It is
sampled uniformly in direction cosines by a Fibonacci spiral of exactly
`samples` directions (`cone_directions`). `within(axis, na)` sums the
efficiencies of the orders inside the cone `cos(angle to axis) ≥ √(1 − na²)`.

<!-- excerpt crates/oc-core/src/modulator.rs 750 766 -->
*Excerpt: `crates/oc-core/src/modulator.rs`, lines 750–766, commit `1af40e8`.*

```rust
impl OrderTable {
    /// The order carrying the most power.
    pub fn brightest(&self) -> Option<&Order> {
        self.orders
            .iter()
            .max_by(|a, b| a.efficiency.partial_cmp(&b.efficiency).unwrap())
    }
    /// Power fraction inside a cone of half-angle `asin(na)` about `axis`.
    pub fn within(&self, axis: Vec3, na: f64) -> f64 {
        let c = (1.0 - na * na).max(0.0).sqrt();
        self.orders
            .iter()
            .filter(|o| o.dir.dot(axis) >= c)
            .map(|o| o.efficiency)
            .sum()
    }
}
```

<!-- excerpt crates/oc-core/src/modulator.rs 913 978 -->
*Excerpt: `crates/oc-core/src/modulator.rs`, lines 913–978, commit `1af40e8`.*

```rust
/// The illumination direction that a DMD's ON mirrors send along `+Z`:
/// `ẑ` reflected off the ON facet and reversed into an incoming direction,
/// `ẑ − 2(ẑ·n)n`, which arrives at twice the tilt.
pub fn on_axis_illumination(d: &DmdSpec) -> Vec3 {
    let n = facet_normal(d, DmdState::On);
    reflect(Vec3::Z, n)
}

/// Directions filling a cone of numerical aperture `na` about `axis`
/// uniformly in direction cosines (equal étendue each), deterministically:
/// a Fibonacci spiral on the disk of radius `na` in the axis's own frame.
pub fn cone_directions(axis: Vec3, na: f64, count: usize) -> Vec<Vec3> {
    let a = axis.normalize();
    let helper = if a.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let u = a.cross(helper).normalize();
    let v = a.cross(u);
    let golden = PI * (3.0 - 5f64.sqrt());
    (0..count)
        .map(|i| {
            let r = na * ((i as f64 + 0.5) / count as f64).sqrt();
            let t = i as f64 * golden;
            let (s, c) = (r * t.cos(), r * t.sin());
            (a * (1.0 - s * s - c * c).sqrt() + u * s + v * c).normalize()
        })
        .collect()
}

/// The diffraction efficiency of an all-ON DMD in a projector, after Texas
/// Instruments DLPA083B §4.2: "the best way to approach broadband source
/// diffraction is to consider it as a combination of a large number of plane
/// waves varying in wavelength and direction ... combined incoherently";
/// the diffracted power is truncated by the projection lens's aperture.
///
/// Illumination fills a cone of f-number `illum_fno` about the direction the
/// ON mirrors send along `+Z`; the projection lens collects a cone of
/// f-number `proj_fno` about `+Z`. Each incident direction is a plane wave,
/// its orders come from [`unit_cell_orders`] under `model`, and the result is
/// the collected share of the reflected power, averaged over `samples`
/// directions: the window, the mirror reflectance and the fill factor are
/// TI's separate factors and are not in it.
pub fn dmd_projector_efficiency(
    device: &SpatialModulator,
    lambda_um: f64,
    illum_fno: f64,
    proj_fno: f64,
    samples: usize,
    model: OrderModel,
) -> Result<f64, ModulatorError> {
    let d = match &device.kind {
        ModulatorKind::Dmd(d) => d,
        _ => return err("a projector efficiency is for a DMD"),
    };
    if !(illum_fno > 0.5 && proj_fno > 0.5 && samples > 0) {
        return err("f-numbers above 0.5 and at least one sample");
    }
    let chief = on_axis_illumination(d);
    let na_ill = 1.0 / (2.0 * illum_fno);
    let na_proj = 1.0 / (2.0 * proj_fno);
    let cell = UnitCell::uniform(DmdState::On.value());
    let mut acc = 0.0;
    for dir in cone_directions(chief, na_ill, samples) {
        let t = unit_cell_orders(device, &cell, lambda_um, dir, model, LcosBeam::Modulated)?;
        acc += t.within(Vec3::Z, na_proj) / t.total;
    }
    Ok(acc / samples as f64)
}
```

### 6.5 LCoS: phase table, dead space, director, fringing

- **Pixels**: square electrodes of side `ρp`.
- **Phase table**: grey level `g` maps to phase `φ(g)` through a table at the
  design wavelength `λ₀`. `Linear{levels L, max φ_max}` gives
  `φ(g) = min(g, L−1)·φ_max/L`, so a full wave leaves the top level one step
  short of 2π.
- **Other wavelengths**: the phase scales as `λ₀/λ`, which ignores the
  dispersion of the birefringence (`PhaseDispersionApprox`).
- **Dead space**: reflects unmodulated, with complex amplitude `g`.
- **Polarization** [Zhang, You & Chu 2014]: only the component along the
  director is modulated. Of a linear polarization at `ψ` against director
  `δ`, `cos(ψ − δ)` of the amplitude is modulated and `sin(ψ − δ)` is not.
  The two beams are orthogonally polarized.

Closed-form orders for a periodic cell:

```text
c_mn = g·δ_m0 δ_n0 + (ρ²/(PₓPᵧ)) sinc(ρm/Pₓ) sinc(ρn/Pᵧ) Σ_ij (√R e^{iφ_ij} − g) e^{−i2π(mi/Pₓ + nj/Pᵧ)}
```

With no dead space this gives the N-level staircase's first order,
`sinc²(1/N)` [Swanson 1989]. That staircase is a blazed grating whose period
is `N` pixels in one direction, `Pₓ = N`, with phases `2πj/N`,
`j = 0 … N − 1`, from an `N`-level linear table. Dead space is a known source
of the zero order [Zhang et al. 2009]; for a survey of LCoS devices see
[Lazarev et al. 2019].

**Sampled near field** (`lcos_reflect`): each sample takes the area-weighted
mean of the reflectance over its cell, `√R e^{iφ}` on the electrodes and `g`
on the dead space. Light outside the pixel array is absorbed. In the
excerpt:
- `g_gap` is `g`;
- `half` is the electrode's half-side `ρp/2`;
- `w` is the fringing width in mm.

As for the DMD, the edge loss booked as `beyond_nyquist` is clipped at zero.

**Fringing** of width `w > 0` [Fringing-field model 2004];
[Persson, Engström & Goksör 2012]: the electrode phase at a sample is the **mean of
the pixel phases over a `w × w` window** about it. The means are taken over
the wrapped phases, so a 2π reset gets a flyback ramp. The phase is averaged,
not the complex field. Fringing has no closed form, so it is sampled only.

<!-- excerpt crates/oc-wave/src/modulator.rs 209 216 -->
*Excerpt: `crates/oc-wave/src/modulator.rs`, lines 209–216, commit `1af40e8`.*

```rust
/// The LCoS phase of pixel `(i, j)` at vacuum wavelength `lambda_um` for
/// `beam`: the table's phase scaled by `λ₀/λ`.
fn lcos_phase(device: &SpatialModulator, l: &LcosSpec, i: usize, j: usize, lambda_um: f64, beam: LcosBeam) -> f64 {
    match beam {
        LcosBeam::Modulated => l.lut.phase(device.value(i, j)) * l.design_lambda_um / lambda_um,
        LcosBeam::Unmodulated => 0.0,
    }
}
```

<!-- excerpt crates/oc-wave/src/modulator.rs 252 341 -->
*Excerpt: `crates/oc-wave/src/modulator.rs`, lines 252–341, commit `1af40e8`.*

```rust
    // 1D overlap weights of [c − w/2, c + w/2] with each pixel's full cell.
    let weights = |c: f64, n: usize| -> Vec<(usize, f64)> {
        let mut v = Vec::new();
        for i in pixel_range(c - 0.5 * w, c + 0.5 * w, n, p) {
            let left = -(n as f64) * p / 2.0 + i as f64 * p;
            let o = ((c + 0.5 * w).min(left + p) - (c - 0.5 * w).max(left)).max(0.0) / w;
            if o > 0.0 {
                v.push((i, o));
            }
        }
        v
    };

    for j in 0..grid.n[1] {
        let y = grid.coord(1, j);
        let (y0, y1) = (y - 0.5 * dy, y + 0.5 * dy);
        let wy = if w > 0.0 { weights(y, ny) } else { vec![] };
        for i in 0..grid.n[0] {
            let x = grid.coord(0, i);
            let (x0, x1) = (x - 0.5 * dx, x + 0.5 * dx);
            let idx = grid.idx(i, j);
            let irr = u.data[idx].norm_sqr() * da;
            // Electrode coverage per pixel, and the cell inside the device.
            let mut acc = Cx::ZERO;
            let mut cov_el = 0.0;
            let smooth_phase = if w > 0.0 && beam == LcosBeam::Modulated {
                let wx = weights(x, nx);
                let mut ph = 0.0;
                let mut tot = 0.0;
                for &(pj, vy) in &wy {
                    for &(pi, vx) in &wx {
                        ph += vx * vy * lcos_phase(device, l, pi, pj, lam, beam);
                        tot += vx * vy;
                    }
                }
                if tot > 0.0 {
                    Some(ph / tot)
                } else {
                    None
                }
            } else {
                None
            };
            let mut cov_dev = 0.0;
            for pj in pixel_range(y0, y1, ny, p) {
                for pi in pixel_range(x0, x1, nx, p) {
                    let c = device.pixel_center_mm(pi, pj);
                    let cell = Shape::Rect { c, h: [0.5 * p, 0.5 * p] };
                    cov_dev += cell.cell_overlap(x0, x1, y0, y1) / da;
                    let el = Shape::Rect { c, h: [half, half] };
                    let cov = el.cell_overlap(x0, x1, y0, y1) / da;
                    if cov <= 0.0 {
                        continue;
                    }
                    cov_el += cov;
                    let phi = smooth_phase.unwrap_or_else(|| lcos_phase(device, l, pi, pj, lam, beam));
                    acc += Cx::from_polar(l.reflectance.sqrt() * cov, phi);
                }
            }
            let cov_gap = (cov_dev - cov_el).max(0.0);
            acc += g_gap * cov_gap;
            let v = u.data[idx] * acc;
            out.data[idx] = v;
            // Physically reflected: R on electrodes, |g|² on the dead space.
            let refl = l.reflectance * cov_el + l.gap_amplitude * l.gap_amplitude * cov_gap;
            absorbed += irr * (cov_el + cov_gap - refl) + irr * (1.0 - cov_dev).max(0.0);
            beyond += irr * refl - v.norm_sqr() * da;
        }
    }
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();
    report.ledger.absorbed_element = absorbed;
    // A sample spanning several phases carries less than the cell reflects:
    // the difference went into angles beyond the grid's band.
    report.ledger.beyond_nyquist = beyond.max(0.0);
    report.ledger.detected = out.power();
    report.ledger.close();
    validity_flags(&mut report, p, lam);
    if beam == LcosBeam::Modulated && (lam - l.design_lambda_um).abs() > 1e-12 {
        report.flag(ValidityFlag::PhaseDispersionApprox);
    }
    report.approx("LCoS as a thin element: dead space unmodulated, fringing as linear broadening");
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!("LCoS {}×{} at {} µm, {:?} beam", nx, ny, device.pitch_um, beam),
    });
    Ok((out, report))
}
```

**LCoS factorized far field**: pixels `(√R e^{iφ} − g)` times the
electrode's transform, plus the dead space's `g` over the whole device.
Fringing is refused here.

<!-- excerpt crates/oc-wave/src/modulator.rs 512 538 -->
*Excerpt: `crates/oc-wave/src/modulator.rs`, lines 512–538, commit `1af40e8`.*

```rust
    let g = Cx::from_polar(l.gap_amplitude, l.gap_phase_rad);
    let a = l.pixel_fill * p;
    let f0 = [start[0] / lam, start[1] / lam];
    let df = [step[0] / lam, step[1] / lam];
    let mut vals = vec![Cx::ZERO; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let c = device.pixel_center_mm(i, j);
            let inc = Cx::expi(k * (d_in.x * c[0] + d_in.y * c[1]));
            let r = Cx::from_polar(l.reflectance.sqrt(), lcos_phase(device, l, i, j, lambda_um, beam));
            vals[j * nx + i] = inc * (r - g);
        }
    }
    let b = lattice_sum(device, vals, f0, df, n, fft)?;
    let (wx, wy) = (nx as f64 * p, ny as f64 * p);
    let mut out = vec![Cx::ZERO; n[0] * n[1]];
    for jj in 0..n[1] {
        for ii in 0..n[0] {
            let f = [f0[0] + ii as f64 * df[0], f0[1] + jj as f64 * df[1]];
            // The incident tilt runs across each electrode too: its transform
            // is taken at f − t_in/λ, as the dead space's is over the device.
            let fr = [f[0] - d_in.x / lam, f[1] - d_in.y / lam];
            let el = a * a * sinc(a * fr[0]) * sinc(a * fr[1]);
            let dev = wx * wy * sinc(wx * fr[0]) * sinc(wy * fr[1]);
            out[jj * n[0] + ii] = b[jj * n[0] + ii] * el + g * dev;
        }
    }
```

### 6.6 Patterns and design

**DMD patterns**:
- binary gratings;
- Lee holograms as [Gutiérrez-Cuevas & Popoff 2024] write them, Eq. (11):
  - for a target field `A e^{iφ}`, its amplitude `A` scaled to a peak of 1;
  - with a carrier of spatial frequency `ν` (one cycle per `period` pixels
    along the pattern's angle);
  - the mirror is ON where `cos(2πν·r − φ) ≥ cos(πq)`, with
    `q = arcsin(A)/π`, so that `A = sin(πq)`.

  The −1 order then carries `(A/π)e^{iφ}` (their Eq. (15)).

**Popoff's blazed number** `μ` [Popoff et al. 2024, Eq. (2)] measures how
well the mirror's specular direction falls on one order of the lattice. At
`μ = 1` one order collects the light; at `μ = 0` four orders share it.

**LCoS patterns**: blazed gratings, Fresnel lenses `−πr²/(λ₀f)`, vortices
`ℓ·atan2(y, x)`, wrapped sums of these, or any explicit grey map.

**Gerchberg–Saxton** [Gerchberg & Saxton 1972] runs between a source amplitude
and a Fraunhofer target on the FFT's own grid. Its error cannot increase per
iteration [Fienup 1982], and the test holds the code to that.

---

## 7. The wave path through a scene

`wave_path(scene, spec, revision)` carries a coherent beam from a source to a
rectangular detector. It follows every chain by which the source's axial ray
reaches the detector, at each of the source's spectral lines. It is a
`WaveJob` run in one step. The job can also be sliced, and slicing changes no
bit.

### 7.1 Planning: the chain, its events, the beam frame

`oc_trace::extract_sequences3` traces the source's axial ray at a line's
wavelength, with ghosts off (§9). Unlike the single-chain `extract_sequence3`,
which refuses several chains as `Ambiguous`, it returns **every** branch
lineage that reaches the detector:
- one chain for a simple system;
- several where splitter arms recombine, an interferometer.

The chain's hits become events:
- a **flat standalone surface** (mirror, splitter arm, grating, thin lens,
  polarizer, retarder, coating, modulator) is a `Hit`;
- a **face of a body or a curved surface** is a `Transfer` across a thin slab
  (§7.5);
- **two flat parallel faces of one body**, entered and left with the same
  medium outside, bare or with a residual AR, are a `Plate` (§7.8);
- the **detector** is the `End`.

The tracer logs no event where a ray passes an aperture's opening. The path
therefore intersects each chain segment with every aperture plane and
inserts `Aperture` events. An aperture plane that falls inside a face's slab
or a plate is refused by name.

The field rides in a frame across the chief ray: origin on the ray, `+Z`
along it. At each event the transverse axes are carried round by the
shortest rotation from the incoming direction to the outgoing one.

<!-- excerpt crates/oc-wave/src/path.rs 166 194 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 166–194, commit `1af40e8`.*

```rust
/// A frame across the chief ray: origin on it, `d` along it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Beam {
    pub(crate) origin: Vec3,
    pub(crate) x: Vec3,
    pub(crate) y: Vec3,
    pub(crate) d: Vec3,
}

impl Beam {
    pub(crate) fn iso(&self) -> Isometry3 {
        let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(self.x, self.y, self.d));
        Isometry3::new(self.origin, [q.x, q.y, q.z, q.w]).unwrap_or(Isometry3::from_pos(self.origin))
    }
    pub(crate) fn coords(&self, r: Vec3) -> (f64, f64) {
        let v = r - self.origin;
        (v.dot(self.x), v.dot(self.y))
    }
    pub(crate) fn point(&self, a: f64, b: f64) -> Vec3 {
        self.origin + self.x * a + self.y * b
    }
    /// The same beam going along `d2` from `origin`: its transverse axes
    /// carried by the shortest rotation, which keeps the frame right-handed.
    fn turned(&self, origin: Vec3, d2: Vec3) -> Beam {
        let q = glam::DQuat::from_rotation_arc(self.d, d2);
        let x = (q * self.x - d2 * d2.dot(q * self.x)).normalize();
        Beam { origin, x, y: d2.cross(x), d: d2 }
    }
}
```

In the planning code below:
- a `SeqHit3` is one hit of the chain, with its `point`, its `normal`, the
  `object` hit, the `action` taken (`Reflect`, `Transmit`, `Diffract`,
  `ThinLens`), and `n_before` and `n_after` (the indices before and after);
- `SeqHitKind` says whether the object is a `Body` (a solid with faces) or a
  standalone `Surface`;
- `plate_of` (§7.8) tests for a plane-parallel plate;
- `slab` (§7.5) finds a face's slab;
- `probe_scene` copies the scene with that one element alone.

<!-- excerpt crates/oc-wave/src/path.rs 976 1097 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 976–1097, commit `1af40e8`.*

```rust
        // The chain's events. Each face of a body, and each curved surface, is
        // a transfer across the thin slab that holds the surface where the
        // beam meets it; between faces the field diffracts inside the glass.
        let mut chain: Vec<(Vec3, Vec3, Ev)> = Vec::new();
        let mut resume = seq.launch.point;
        let mut k = 0;
        while k < seq.hits.len() {
            let h = &seq.hits[k];
            k += 1;
            let curved = h.kind == SeqHitKind::Surface
                && scene.surfaces.iter().any(|s| s.common.id == h.object && !matches!(s.geometry, SurfaceGeometry3::Plane { .. }));
            if h.kind != SeqHitKind::Body && !curved {
                chain.push((h.point, h.point, Ev::Hit(h.clone())));
                resume = h.point;
                continue;
            }
            // A plane-parallel plate is one closed-form step, its round trips
            // included (crate::planar).
            if h.kind == SeqHitKind::Body {
                if let Some(h2) = seq.hits.get(k) {
                    let next = seq.hits.get(k + 1).map_or(seq.terminal.point, |n| n.point);
                    if let Some((plate, entry, exit)) = plate_of(scene, h, h2, resume, next, 0.5 * spec.window_mm, lam) {
                        chain.push((entry, exit, Ev::Plate { first: h.clone(), second: h2.clone(), exit, plate }));
                        resume = exit;
                        k += 1;
                        continue;
                    }
                }
            }
            let next_point = seq.hits.get(k).map_or(seq.terminal.point, |n| n.point);
            let d_in = (h.point - resume).normalize();
            let d_out = (next_point - h.point).normalize();
            let probe = probe_scene(scene, h.object);
            let half = 0.5 * spec.window_mm;
            let (s_min, s_max) = slab(&probe, resume, d_in, h.point, d_out, half, lam);
            // A margin of a micrometre, so neither plane touches the surface.
            let entry = h.point + d_in * (s_min - 1e-3);
            let exit = h.point + d_out * (s_max + 1e-3);
            if (entry - resume).dot(d_in) <= 0.0 || (next_point - exit).dot(d_out) <= 0.0 {
                return Err(WaveError::OutOfRange(format!(
                    "#{} ({}): the surface where the beam meets it is too close to the next one on the path for a plane to fit between them",
                    h.object.0, h.label
                )));
            }
            chain.push((entry, exit, Ev::Transfer { hit: h.clone(), exit, d_out }));
            resume = exit;
        }
        chain.push((seq.terminal.point, seq.terminal.point, Ev::End));

        // Insert the aperture planes the chief ray crosses between events.
        let mut points: Vec<(Vec3, Vec3, Ev)> = Vec::new();
        let mut prev = seq.launch.point;
        for (p, after, ev) in chain {
            let seg = p - prev;
            let len = seg.length();
            if len > 0.0 {
                let dir = seg / len;
                let mut crossings: Vec<(f64, usize)> = Vec::new();
                for (ai, a) in scene.apertures.iter().enumerate() {
                    let t = &a.common.transform;
                    let o = t.transform_point(a.shape.plane_local.origin);
                    let n = t.rotate(a.shape.plane_local.normal);
                    if let Some(x) = onto_plane(prev, dir, o, n) {
                        let s = (x - prev).dot(dir);
                        // One at the event's own plane is applied just before it.
                        if s > 1e-9 && s <= len + 1e-9 {
                            crossings.push((s, ai));
                        }
                    }
                }
                crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (s, ai) in crossings {
                    let q = prev + dir * s;
                    points.push((q, q, Ev::Aperture(ai)));
                }
            }
            // An aperture inside a face's slab, or a plate, has no place in
            // the field's steps: refused by name rather than skipped.
            let inside: Vec<(Vec3, Vec3, &oc_core::seq3::SeqHit3)> = match &ev {
                Ev::Transfer { hit, .. } => vec![(p, hit.point, hit), (hit.point, after, hit)],
                Ev::Plate { first, .. } => vec![(p, after, first)],
                _ => Vec::new(),
            };
            for (from, to, hit) in inside {
                {
                    let len = (to - from).length();
                    if len <= 0.0 {
                        continue;
                    }
                    let dir = (to - from) / len;
                    for a in &scene.apertures {
                        let t = &a.common.transform;
                        let o = t.transform_point(a.shape.plane_local.origin);
                        let n = t.rotate(a.shape.plane_local.normal);
                        if let Some(x) = onto_plane(from, dir, o, n) {
                            let s = (x - from).dot(dir);
                            if s > 1e-9 && s < len - 1e-9 {
                                return Err(WaveError::OutOfRange(format!(
                                    "aperture #{} lies inside the slab the field is traced across at #{} ({}): move it clear of the surface",
                                    a.common.id.0, hit.object.0, hit.label
                                )));
                            }
                        }
                    }
                }
            }
            points.push((p, after, ev));
            prev = after;
        }
        let keys = points
            .iter()
            .map(|(p, _, ev)| {
                let at = format!("{:.6},{:.6},{:.6}", p.x, p.y, p.z);
                match ev {
                    Ev::Hit(h) => format!("hit {} {:?} {at}", h.object.0, h.action),
                    Ev::Aperture(ai) => format!("aperture {ai} {at}"),
                    Ev::Transfer { hit, .. } => format!("transfer {} {:?} {at}", hit.object.0, hit.action),
                    Ev::Plate { first, .. } => format!("plate {} {at}", first.object.0),
                    Ev::End => format!("end {at}"),
                }
            })
            .collect();
```

### 7.2 Free space between events, and windows fitted to the beam

Between events the field propagates the chief ray's arc length with
`Propagator::Auto` (§4.3). One window cannot both hold a wide beam and resolve
the focus it is brought to. So, before each free-space step, the path
predicts the beam's size at the next plane from the field's **second
moments**, per axis:

- the centroid `x̄` and the variance `σ²`;
- the mean direction `θ̄` and the angular variance `σ_θ²`, both from the
  field's FFT spectrum (direction cosine `λf`);
- the cross moment `C = ⟨(x − x̄)θ⟩`, with `θ = ∂φ/∂x / k` the local
  direction. This is taken from a sphere-plus-tilt fit, plus the
  neighbouring-sample phase difference of what the fit leaves.

Under paraxial propagation, for any beam shape
[Siegman 1990] (the second-moment law of the Wigner distribution):

```text
σ²(z) = σ² + 2zC + z²σ_θ²
```

**When to re-grid.** `needed = 6σ(z)` is taken as the new **half-width**:
- *shrink* when `max(needed, 16λ) < ¼` of the current half-width;
- *grow* (only on a window already narrowed) when `needed > 0.8` of the
  current half-width, to a half-width of `min(2·needed, declared half-width)`.

The new window is square, of `max(nₓ, nᵧ)` samples per side. It is centred
on the predicted centroid `x̄ + zθ̄`, and is never wider than the declared
window. The doc comments and the step notes speak of "six rms radii across"
the window, but the code uses `6σ` as the half-width (noted in §11.3).

**How to re-grid.** The plane waves walk `z·(|θ̄| + 4σ_θ)` over the step:
- if the walk fits the current window, the zoomed angular spectrum with 2×
  padding (exact);
- otherwise, if `z` exceeds the current window's critical distance
  `nΔx·Δx/λ`, unpadded and on axis 0 only, the zoomed Fresnel transform
  (paraxial, its error stated). The `regrid_plan` comment ties this to widening, but the code
  reaches it whenever the walk is large;
- otherwise the zoomed ASM, padded `⌈2·walk/width⌉` times (clamped 2–8).

**A coarser window holds fewer directions.** The plane waves beyond the new
window's band, `|f| > 1/(2Δx_new)` about the carrier, are taken out first by
`band_to` and booked as `beyond_nyquist`. The reason: free space keeps every
plane wave, and the new samples, read at their points, would fold those plane
waves back in at directions they do not have.

In the excerpts below:
- `self.u` is the field, and `self.uy` the second polarization component when
  the chain carries two (§7.10).
- `self.rest` holds a modulator's other orders carried apart (§7.7), each a
  `Lobe { u, uy, at }`.
- `self.q` holds the reference Gaussian parameter per frame axis (§7.6).
- `self.at` is `[a, b, tₓ, tᵧ]`, the reference ray's point and direction
  cosines across the beam, all zero for the chief ray.
- `Moments::with` combines two components' moments, and `sigma_at(axis, z)`
  is `σ(z)`.

<!-- excerpt crates/oc-wave/src/path.rs 383 496 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 383–496, commit `1af40e8`.*

```rust
/// A field's second moments, across its plane and in angle, per axis: its
/// centroid `x̄` and variance `σ²`, its mean direction `θ̄` (a direction
/// cosine, the carrier included) and angular variance `σ_θ²`, and the cross
/// moment `C = ⟨(x − x̄)θ⟩`, `θ = ∂φ/∂x / k` the local direction. Under
/// paraxial propagation a distance `z` they give the beam's size exactly,
/// `σ²(z) = σ² + 2zC + z²σ_θ²` (the second-moment law of the Wigner
/// distribution; A. E. Siegman, *Proc. SPIE* 1224, 2 (1990)), whatever the
/// beam's shape: what re-gridding needs to choose the window at the next
/// plane before the field is carried there.
struct Moments {
    power: f64,
    mean: [f64; 2],
    var: [f64; 2],
    tilt: [f64; 2],
    spread: [f64; 2],
    cross: [f64; 2],
}

impl Moments {
    fn of(u: &Field2, fft: &mut Fft2) -> Option<Moments> {
        let g = u.grid;
        let k = u.k_mm();
        let (mut p, mut mx, mut my) = (0.0, 0.0, 0.0);
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (x, y) = g.xy(i, j);
                let w = u.at(i, j).norm_sqr();
                p += w;
                mx += w * x;
                my += w * y;
            }
        }
        if p <= 0.0 || p.is_nan() {
            return None;
        }
        let (mx, my) = (mx / p, my / p);
        // The local direction θ = α + c·x + the rest: the fitted sphere and
        // tilt analytically, what they leave by the phase difference of
        // neighbouring samples (which a sphere turning most of a radian a
        // sample would make a finite difference understate).
        let ([c, al, be], _) = fit_sphere(u);
        let sm = Smooth::with(u, [c, al - u.carrier[0], be - u.carrier[1]]);
        let rest = |i: usize, j: usize| {
            let (x, y) = g.xy(i, j);
            sm.at(x, y) * Cx::expi(-k * (0.5 * c * (x * x + y * y) + (al - u.carrier[0]) * x + (be - u.carrier[1]) * y))
        };
        let (mut vx, mut vy, mut cx, mut cy) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (x, y) = g.xy(i, j);
                let w = u.at(i, j).norm_sqr();
                if w == 0.0 {
                    continue;
                }
                vx += w * (x - mx).powi(2);
                vy += w * (y - my).powi(2);
                let tx = if i > 0 && i + 1 < g.n[0] { (rest(i + 1, j) * rest(i - 1, j).conj()).arg() / (2.0 * g.pitch_mm[0] * k) } else { 0.0 };
                let ty = if j > 0 && j + 1 < g.n[1] { (rest(i, j + 1) * rest(i, j - 1).conj()).arg() / (2.0 * g.pitch_mm[1] * k) } else { 0.0 };
                cx += (x - mx) * w * (al + c * x + tx);
                cy += (y - my) * w * (be + c * y + ty);
            }
        }
        // The angular moments from the field's spectrum.
        let n = g.n;
        let mut buf = u.data.clone();
        crate::fft::ifftshift2(&mut buf, n);
        fft.forward(&mut buf, n, g.cell_area_mm2());
        let lam = u.wavelength_in_medium_mm();
        let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
        let (mut q, mut tx, mut ty, mut sx, mut sy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for ky in 0..n[1] {
            let fy = g.fft_freq(1, ky) + v0;
            for kx in 0..n[0] {
                let fx = g.fft_freq(0, kx) + u0;
                let w = buf[ky * n[0] + kx].norm_sqr();
                q += w;
                tx += w * lam * fx;
                ty += w * lam * fy;
                sx += w * (lam * fx).powi(2);
                sy += w * (lam * fy).powi(2);
            }
        }
        let (tx, ty) = (tx / q, ty / q);
        Some(Moments {
            power: p,
            mean: [mx, my],
            var: [vx / p, vy / p],
            tilt: [tx, ty],
            spread: [(sx / q - tx * tx).max(0.0), (sy / q - ty * ty).max(0.0)],
            cross: [cx / p, cy / p],
        })
    }

    /// The moments of two orthogonal components together: power-weighted,
    /// the variances about the common centroid.
    fn with(&self, o: &Moments) -> Moments {
        let (a, b) = (self.power, o.power);
        let t = (a + b).max(f64::MIN_POSITIVE);
        let w = |x: f64, y: f64| (a * x + b * y) / t;
        let mean = [w(self.mean[0], o.mean[0]), w(self.mean[1], o.mean[1])];
        let tilt = [w(self.tilt[0], o.tilt[0]), w(self.tilt[1], o.tilt[1])];
        let var = [0, 1].map(|k| w(self.var[k] + (self.mean[k] - mean[k]).powi(2), o.var[k] + (o.mean[k] - mean[k]).powi(2)));
        let spread = [0, 1].map(|k| w(self.spread[k] + (self.tilt[k] - tilt[k]).powi(2), o.spread[k] + (o.tilt[k] - tilt[k]).powi(2)));
        let cross = [0, 1].map(|k| {
            w(self.cross[k] + (self.mean[k] - mean[k]) * (self.tilt[k] - tilt[k]), o.cross[k] + (o.mean[k] - mean[k]) * (o.tilt[k] - tilt[k]))
        });
        Moments { power: t, mean, var, tilt, spread, cross }
    }

    /// The rms radius along `axis`, `z` on.
    fn sigma_at(&self, axis: usize, z: f64) -> f64 {
        (self.var[axis] + 2.0 * z * self.cross[axis] + z * z * self.spread[axis]).max(0.0).sqrt()
    }
}
```

<!-- excerpt crates/oc-wave/src/path.rs 516 566 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 516–566, commit `1af40e8`.*

```rust
/// `u` with the plane waves a window `out` cannot hold taken out, those
/// beyond its band about the field's carrier, and the power they carried;
/// `u` as it is when `out` is no coarser.
fn band_to(u: &Field2, out: &Grid2, fft: &mut Fft2) -> (Field2, f64) {
    let g = u.grid;
    let lim = [0.5 / out.pitch_mm[0], 0.5 / out.pitch_mm[1]];
    if out.pitch_mm[0] <= g.pitch_mm[0] * (1.0 + 1e-12) && out.pitch_mm[1] <= g.pitch_mm[1] * (1.0 + 1e-12) {
        return (u.clone(), 0.0);
    }
    let n = g.n;
    let mut b = u.data.clone();
    crate::fft::ifftshift2(&mut b, n);
    fft.forward(&mut b, n, g.cell_area_mm2());
    let df = g.freq_pitch(0) * g.freq_pitch(1);
    let mut removed = 0.0;
    for ky in 0..n[1] {
        let fy = g.fft_freq(1, ky);
        for kx in 0..n[0] {
            let fx = g.fft_freq(0, kx);
            if fx.abs() > lim[0] || fy.abs() > lim[1] {
                let a = &mut b[ky * n[0] + kx];
                removed += a.norm_sqr() * df;
                *a = Cx::ZERO;
            }
        }
    }
    if removed == 0.0 {
        return (u.clone(), 0.0);
    }
    fft.inverse(&mut b, n, df);
    crate::fft::fftshift2(&mut b, n);
    let mut v = u.clone();
    v.data = b;
    (v, removed)
}

/// A window a field is to be carried onto, and how.
struct Regrid {
    out: Grid2,
    how: RegridHow,
    half: f64,
    needed: f64,
}

#[derive(Clone, Copy)]
enum RegridHow {
    /// The zoomed angular spectrum, padded so many times.
    Zoom(usize),
    /// The zoomed Fresnel transform.
    Fresnel,
}
```

<!-- excerpt crates/oc-wave/src/path.rs 1252 1345 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1252–1345, commit `1af40e8`.*

```rust
    /// Carry the field `z` on onto a window fitted to the beam there, when
    /// the one it has would no longer suit it: when the beam, six of its rms
    /// radii across (`Moments`), would fill less than a quarter of the window
    /// (a focus, a pinhole's plane), or, on a window already narrowed, more
    /// than four fifths of it. The window is centred on the beam's predicted
    /// centroid, keeps its sample count, and never grows past the one the
    /// spec declares. `None` to stay on the window the field has.
    ///
    /// Narrowing, the zoomed angular spectrum carries it, exactly, padded
    /// enough that the plane waves' walk over `z` stays inside; widening from
    /// a narrow window past its critical distance, the zoomed Fresnel
    /// transform, whose paraxial error on the field is reported.
    fn regrid_plan(&mut self, z: f64) -> Result<Option<Regrid>, WaveError> {
        let mut fields: Vec<&Field2> = vec![&self.u];
        fields.extend(self.uy.as_ref());
        for l in &self.rest {
            fields.push(&l.u);
            fields.extend(l.uy.as_ref());
        }
        let mut m: Option<Moments> = None;
        for f in fields {
            if let Some(a) = Moments::of(f, &mut self.fft) {
                m = Some(match m {
                    Some(b) => b.with(&a),
                    None => a,
                });
            }
        }
        let Some(m) = m else { return Ok(None) };
        let g = self.u.grid;
        let n = g.n[0].max(g.n[1]);
        let half_now = 0.5 * g.width_mm(0).max(g.width_mm(1));
        let half_max = 0.5 * self.spec.window_mm;
        let needed = 6.0 * m.sigma_at(0, z).max(m.sigma_at(1, z));
        let lam = self.u.wavelength_in_medium_mm();
        // A window no narrower than a few dozen wavelengths.
        let target = needed.min(half_max).max(16.0 * lam);
        let shrink = target < 0.25 * half_now;
        let grow = half_now < half_max * (1.0 - 1e-9) && needed > 0.8 * half_now;
        if !(shrink || grow) {
            return Ok(None);
        }
        let half = if grow { (2.0 * needed).min(half_max) } else { target };
        let centre = [m.mean[0] + z * m.tilt[0], m.mean[1] + z * m.tilt[1]];
        let pitch = 2.0 * half / n as f64;
        let out = Grid2::new([n, n], [pitch, pitch], centre)?;
        // How far the field's plane waves walk across the beam over z.
        let walk = z * (m.tilt[0].abs().max(m.tilt[1].abs()) + 4.0 * m.spread[0].max(m.spread[1]).sqrt());
        let width_in = 2.0 * half_now;
        let zc = g.width_mm(0) * g.pitch_mm[0] / lam;
        let how = if walk <= width_in {
            RegridHow::Zoom(2)
        } else if z > zc {
            RegridHow::Fresnel
        } else {
            RegridHow::Zoom(((2.0 * walk / width_in).ceil() as usize).clamp(2, 8))
        };
        Ok(Some(Regrid { out, how, half, needed }))
    }

    /// Carry `u` `z` on onto the window a plan chose. A coarser window holds
    /// fewer directions: the plane waves beyond its band are taken out first
    /// and booked (`beyond_nyquist`), since free space keeps every plane wave
    /// and the new samples, read at their points, would fold them back in at
    /// directions they do not have.
    fn regrid_apply(&mut self, u: &Field2, z: f64, pl: &Regrid) -> Result<(Field2, WaveReport), WaveError> {
        let (banded, beyond) = band_to(u, &pl.out, &mut self.fft);
        let u = &banded;
        let ((v, mut r), how) = match pl.how {
            RegridHow::Zoom(2) => (crate::asm_zoom(u, z, pl.out, 2, &mut self.fft)?, "the zoomed angular spectrum"),
            RegridHow::Zoom(pad) => (crate::asm_zoom(u, z, pl.out, pad, &mut self.fft)?, "the zoomed angular spectrum, padded"),
            RegridHow::Fresnel => {
                let e = crate::propagate::paraxial_error(u, z, &mut self.fft);
                let mut step = crate::czt::fresnel_zoom(u, z, pl.out, &mut self.fft)?;
                if e > 1e-4 {
                    step.1.warn(SamplingWarning::Paraxial { field_error: e });
                }
                (step, "the zoomed Fresnel transform")
            }
        };
        let n = pl.out.n[0];
        r.ledger.beyond_nyquist += beyond;
        let cut = if beyond > 0.0 { format!("; {:.3e} of its power in directions beyond the new window's band taken out first", beyond / (beyond + u.power()).max(f64::MIN_POSITIVE)) } else { String::new() };
        r.steps.push(element_step(
            z,
            format!(
                "the field carried {z:.4} mm onto a new window {:.4e} mm across ({n}², pitch {:.4e} mm), where the beam is predicted {:.4e} mm across (six rms radii), by {how}{cut}",
                2.0 * pl.half,
                pl.out.pitch_mm[0],
                pl.needed
            ),
        ));
        Ok((v, r))
    }
```

<!-- excerpt crates/oc-wave/src/path.rs 2090 2161 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2090–2161, commit `1af40e8`.*

```rust
    /// The field `z` on to `p`, both components onto one window; the
    /// reference's wavefront and ray with it.
    fn free_space(&mut self, z: f64, p: Vec3) -> Result<(), WaveError> {
        let plan = self.regrid_plan(z)?;
        let (v, r) = match &plan {
            Some(pl) => self.regrid_apply(&self.u.clone(), z, pl)?,
            None => propagate(&self.u, z, Propagator::Auto, &mut self.fft)?,
        };
        merge(&mut self.report, r);
        self.u = v;
        if let Some(uy) = self.uy.take() {
            let (vy, ry) = match &plan {
                Some(pl) => self.regrid_apply(&uy, z, pl)?,
                None => propagate(&uy, z, Propagator::Auto, &mut self.fft)?,
            };
            let mut ry = ry;
            ry.steps.clear();
            merge(&mut self.report, ry);
            self.uy = Some(vy);
        }
        let mut lobes = std::mem::take(&mut self.rest);
        for l in &mut lobes {
            for f in std::iter::once(&mut l.u).chain(l.uy.as_mut()) {
                let (v, mut r) = match &plan {
                    Some(pl) => self.regrid_apply(f, z, pl)?,
                    None => propagate(f, z, Propagator::Auto, &mut self.fft)?,
                };
                r.steps.clear();
                merge(&mut self.report, r);
                *f = v;
            }
        }
        // An order free space has emptied, its power booked as it left,
        // rejoins the strongest.
        let pw = |u: &Field2, uy: &Option<Field2>| u.power() + uy.as_ref().map_or(0.0, |f| f.power());
        let total: f64 = pw(&self.u, &self.uy) + lobes.iter().map(|l| pw(&l.u, &l.uy)).sum::<f64>();
        let (spent, kept): (Vec<Lobe>, Vec<Lobe>) = lobes.into_iter().partition(|l| pw(&l.u, &l.uy) < 1e-9 * total);
        for l in spent {
            for (a, b) in self.u.data.iter_mut().zip(&l.u.data) {
                *a += *b;
            }
            if let (Some(uy), Some(ly)) = (&mut self.uy, &l.uy) {
                for (a, b) in uy.data.iter_mut().zip(&ly.data) {
                    *a += *b;
                }
            }
        }
        self.rest = kept;
        self.beam.origin = p;
        let iso = self.beam.iso();
        self.u.frame = iso;
        if let Some(uy) = &mut self.uy {
            uy.frame = iso;
        }
        for q in &mut self.q {
            *q += Cx::real(z);
        }
        let walk = |at: &mut [f64; 4]| {
            let tz = (1.0 - at[2] * at[2] - at[3] * at[3]).max(1e-12).sqrt();
            at[0] += z * at[2] / tz;
            at[1] += z * at[3] / tz;
        };
        walk(&mut self.at);
        for l in &mut self.rest {
            l.u.frame = iso;
            if let Some(uy) = &mut l.uy {
                uy.frame = iso;
            }
            walk(&mut l.at);
        }
        Ok(())
    }
```

### 7.3 The hand-over at a flat element

At a `Hit`, the field is handed from the plane across the incoming ray (`inc`,
direction `d`) to the plane across the outgoing ray (`out`, direction `d′`).
The element's plane passes through the chief ray's hit `p` with normal `n_e`.

For each output sample `r_out`:
1. go along `d′` back to the element plane, giving `r_e`;
2. go along `d` back to the incoming plane, giving `r_in`;
3. take the arriving field at `r_in`.

That point gets:

```text
v(r_out) = u(r_in) · F(r_e) · exp(ik (d − d′)·(r_e − p)) · √(cos θ_in / cos θ_out)
```

- `F` is the element's complex factor on its plane.
- The exponential is the change of reference wave, from `exp(ik d·r)` to
  `exp(ik d′·r)`. `F` must carry the element's whole phase on its plane
  (§11.1 shows a case where it does not).
- `cos θ = |d·n_e|`. The square root keeps the power through the element's
  plane when the beam's cross-section changes by `cos θ_out/cos θ_in` (a
  grating's anamorphic factor) [Harvey et al. 2011].

**A flat mirror** reflects with `d′ = d − 2(d·n)n`, so
`(d − d′)·(r_e − p) = 0` on the mirror plane, and `r_in` is exactly the image
of `r_out`. The hand-over is then image theory, exact in scalar optics
[Goodman 2017]: the reflected field is the mirror image of the incident one,
with no propagation across the plane's depth.

**Any other oblique element** (a thin lens, an aperture, a modulator, a
grating) is a *projection* along the rays: the field is not diffracted across
the depth its plane spans across the beam. `TiltProjection` is raised, with
that depth, once it exceeds `Δx²/λ`.

**Interpolation.** `u(r_in)` falls between samples. It is read bilinearly
after a sphere and tilt fitted to the field are removed, and they are then put
back exactly at the point read (`Smooth`, §7.14).

<!-- excerpt crates/oc-wave/src/path.rs 223 274 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 223–274, commit `1af40e8`.*

```rust
/// Hand the field `u` (across `inc`) through an element in the plane through
/// `inc.origin` with normal `n_e`, to the plane across `out`. `factor` is the
/// element's complex factor at a point of its plane, including any phase the
/// change of direction does not already carry. Returns the new field and the
/// largest depth the projection spanned, mm.
fn hand_over(u: &Field2, inc: &Beam, n_e: Vec3, out: &Beam, factor: &dyn Fn(Vec3) -> Cx) -> (Field2, f64) {
    let mut v = u.clone();
    let g = u.grid;
    let k = u.k_mm();
    let p = inc.origin;
    let mut depth: f64 = 0.0;
    let obliquity = obliquity(inc.d, out.d, n_e);
    let smooth = Smooth::new(u);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (a, b) = g.xy(i, j);
            let r_out = out.point(a, b);
            let val = match onto_plane(r_out, out.d, p, n_e) {
                Some(r_e) => {
                    let r_in = r_e - inc.d * (r_e - p).dot(inc.d);
                    depth = depth.max((r_out - r_e).length() + (r_e - r_in).length());
                    let (ai, bi) = inc.coords(r_in);
                    // The reference wave changes from exp(ik d·r) to
                    // exp(ik d'·r): what the new direction does not carry of
                    // the element's own phase is put on the samples.
                    smooth.at(ai, bi) * factor(r_e) * Cx::expi(k * (inc.d - out.d).dot(r_e - p)) * obliquity
                }
                None => Cx::ZERO,
            };
            v.data[g.idx(i, j)] = val;
        }
    }
    v.frame = out.iso();
    (v, depth)
}

/// The amplitude factor that conserves the power crossing an element's
/// plane when the beam leaves at another angle than it came:
/// `√(cos θ_in / cos θ_out)`. A beam's cross-section changes by
/// `cos θ_out / cos θ_in` on the way through (a grating's anamorphic factor),
/// while projection keeps the amplitude; this puts the power right. It is 1
/// for a mirror, and it is the obliquity the non-paraxial scalar treatment
/// carries by working in direction cosines (J. E. Harvey et al., *Proc. SPIE*
/// 8122, 812205 (2011)).
fn obliquity(d_in: Vec3, d_out: Vec3, n_e: Vec3) -> f64 {
    let (ci, co) = (d_in.dot(n_e).abs(), d_out.dot(n_e).abs());
    if co < 1e-12 {
        0.0
    } else {
        (ci / co).sqrt()
    }
}
```

**The element's factor and its sign conventions.** The signs make every
lossless element unitary, so that recombined arms conserve power [Stokes
1849]; [Zeilinger 1981]:

| Element | `F` inside its outline | What else it books |
|---|---|---|
| mirror (`Mirror{R(λ)}`), reflect | `−√R` | `1 − R` absorbed |
| splitter (`ExplicitRt{R, T}`), reflect | `−√R` from the front, `+√R` from the back | `min(T, 1 − R)` other arm, the rest absorbed |
| splitter, transmit | `+√T` both ways | `min(R, 1 − T)` other arm, the rest absorbed |
| grating, diffracted order | `√η` (the declared TE efficiency; see §11.2) | `1 − η` untracked orders |
| grating, zero order | `√η₀` | `1 − η₀` untracked orders |
| ideal thin lens | `exp(−ik|r − c|²_⊥/2f)` about the lens centre, in its plane | — |
| coating (`Coating`) | applied per plane wave before the hand-over (§7.9); then `F = 1` | other arm, absorbed |
| polarizer, retarder (two-component field) | `F = 1` in each component's pass; the Jones matrix after (§7.10) | — |
| measured table (`MeasuredTable`) | **refused**: R and T without their phases | — |

A perfect conductor is the limit of the dielectric interface, so a mirror
reflects `−√R`. A real coating would fix these phases otherwise. As
conventions they move fringes, never power.

In the excerpt:
- `from_front` is true when the beam meets the surface against its own local
  `+Z` (`d·ẑ_surface < 0`), and the surface's `response_front` then applies;
- `unit()` clamps a sampled value to [0, 1];
- `refuse(kind)` builds the "no wave model" refusal;
- `Loss` says where the power the element does not pass goes: absorbed, the
  other arm, or untracked orders.

<!-- excerpt crates/oc-wave/src/path.rs 1676 1705 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1676–1705, commit `1af40e8`.*

```rust
        // The amplitude inside the outline, a thin lens's focal length and
        // centre, and where the rest goes.
        //
        // The amplitudes' signs make every lossless element unitary, so that
        // arms recombined after it conserve power. A splitter is the two-port
        // of a dielectric interface (bundle M14; the Stokes relations):
        // reflection −√R from its front, +√R from its back, transmission √T
        // both ways; a perfect conductor is that interface's limit, so a
        // mirror reflects −√R. One phase per element is a convention a real
        // coating fixes otherwise, and it moves fringes, never the power.
        let (amp, lens, loss): (f64, Option<(f64, Vec3)>, Loss) = match (response, h.action) {
            (SurfaceResponse::Mirror { reflectivity }, SeqAction::Reflect) => {
                (-unit(reflectivity.sample(lam)).ok_or_else(|| refuse(kind))?.sqrt(), None, Loss::Absorbed)
            }
            (SurfaceResponse::ExplicitRt { reflectance, transmittance, .. }, SeqAction::Reflect | SeqAction::Transmit) => {
                let r = unit(reflectance.sample(lam)).ok_or_else(|| refuse(kind))?;
                let t = unit(transmittance.sample(lam)).ok_or_else(|| refuse(kind))?;
                if h.action == SeqAction::Reflect {
                    (if from_front { -r.sqrt() } else { r.sqrt() }, None, Loss::Arm(t.min(1.0 - r)))
                } else {
                    (t.sqrt(), None, Loss::Arm(r.min(1.0 - t)))
                }
            }
            (SurfaceResponse::Grating { efficiency, .. }, SeqAction::Diffract) => {
                (unit(efficiency.sample(lam)).ok_or_else(|| refuse(kind))?.sqrt(), None, Loss::Orders)
            }
            (SurfaceResponse::Grating { zero_order, .. }, _) => {
                (unit(zero_order.sample(lam)).ok_or_else(|| refuse(kind))?.sqrt(), None, Loss::Orders)
            }
            (SurfaceResponse::ThinLens { focal_length_mm }, _) => (1.0, Some((*focal_length_mm, s.common.transform.pos)), Loss::Absorbed),
```

**An element's outline.** A flat element's own shape clips the field. Its
edge is weighted by exact cell coverage of the shape's projection along the
beam (a disc becomes an area-preserving polygon). Light outside the outline
passes the element by:
- where the chain goes on through it (a transmitted arm, a thin lens), the
  light stays on the chain, handed over with `F = 1`;
- where the chain turns (a reflection, a diffracted order), the light leaves
  the chain and is booked as `passed_by`.

A cell with coverage `f` physically sends on `f|F|² + (1 − f)·pb` of its
power, where `pb` is 1 for a pass-by element and 0 otherwise. The sample
carries `|fF + (1 − f)pb|²`, and the difference is booked as
`beyond_nyquist`. Whatever the hand-over itself does not carry is booked as
`resampling`.

The outline's coverage uses `shape_coverage`. Square to the beam, a disc or
an annulus is covered by its closed form; otherwise every shape becomes a
polygon:

<!-- excerpt crates/oc-wave/src/path.rs 276 305 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 276–305, commit `1af40e8`.*

```rust
/// The covered fraction of each of `grid`'s cells by `shape`, a shape in an
/// element's plane, seen across the beam: `to_beam` takes a point of that
/// plane (its local mm) along the beam to the beam's plane. Square to the
/// beam, a disc or an annulus stays one and is covered by its closed form;
/// otherwise every shape is a polygon there (a disc one of
/// [`crate::elements::disc_sides`] sides with the disc's own area), covered
/// exactly. Either way the edge is resolved to the field's own sample.
fn shape_coverage(grid: &Grid2, shape: &PlanarShape2, square: bool, to_beam: &dyn Fn(f64, f64) -> [f64; 2]) -> Result<Vec<f64>, WaveError> {
    use crate::elements::{coverage, disc_polygon, disc_sides, Shape};
    let pitch = grid.pitch_mm[0].min(grid.pitch_mm[1]);
    let ring = |r: f64| -> Vec<[f64; 2]> { disc_polygon([0.0, 0.0], r, disc_sides(r, pitch)).iter().map(|p| to_beam(p[0], p[1])).collect() };
    let c = to_beam(0.0, 0.0);
    match shape {
        PlanarShape2::Disc { radius_mm } if square => coverage(grid, &Shape::Disc { c, r: *radius_mm }),
        PlanarShape2::Annulus { inner_mm, outer_mm } if square => {
            coverage(grid, &Shape::Annulus { c, r_in: *inner_mm, r_out: *outer_mm })
        }
        PlanarShape2::Disc { radius_mm } => coverage(grid, &Shape::Polygon(ring(*radius_mm))),
        PlanarShape2::Annulus { inner_mm, outer_mm } => {
            let outer = coverage(grid, &Shape::Polygon(ring(*outer_mm)))?;
            let inner = if *inner_mm > 0.0 { coverage(grid, &Shape::Polygon(ring(*inner_mm)))? } else { vec![0.0; outer.len()] };
            Ok(outer.iter().zip(&inner).map(|(o, i)| (o - i).clamp(0.0, 1.0)).collect())
        }
        PlanarShape2::Rectangle { half_width_mm, half_height_mm } => {
            let (w, h) = (*half_width_mm, *half_height_mm);
            coverage(grid, &Shape::Polygon(vec![to_beam(-w, -h), to_beam(w, -h), to_beam(w, h), to_beam(-w, h)]))
        }
        PlanarShape2::Polygon { vertices } => coverage(grid, &Shape::Polygon(vertices.iter().map(|v| to_beam(v.x, v.y)).collect())),
    }
}
```

<!-- excerpt crates/oc-wave/src/path.rs 1765 1893 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1765–1893, commit `1af40e8`.*

```rust
        let k = self.u.k_mm();
        let fin = move |r: Vec3| -> Cx {
            match lens {
                // Distance from the lens's centre within its plane.
                Some((f, c)) => {
                    let v = r - c;
                    let v = v - n_s * v.dot(n_s);
                    Cx::expi(-k * v.length_squared() / (2.0 * f))
                }
                None => Cx::real(amp),
            }
        };
        let pass_by = matches!(h.action, SeqAction::Transmit | SeqAction::ThinLens);
        let pb = if pass_by { 1.0 } else { 0.0 };

        // The outline across the beam.
        let g = self.u.grid;
        let square = n_s.dot(beam.d).abs() > 1.0 - 1e-12;
        let tf = s.common.transform;
        let to_beam = |x: f64, y: f64| {
            let w = tf.transform_point(Vec3::new(x, y, 0.0));
            let (a, b) = beam.coords(w - beam.d * (w - beam.origin).dot(beam.d));
            [a, b]
        };
        let cov = match &s.geometry {
            SurfaceGeometry3::Plane { shape } => shape_coverage(&g, shape, square, &to_beam)?,
            SurfaceGeometry3::SurfaceOfRevolution { .. } => vec![1.0; g.len()],
        };

        // Each sample's share: inside, where the element's factor acts; outside,
        // passed by. Physically a cell sends on f·|t|² + (1 − f)·pb of its
        // power; the sample (f·t + (1 − f)·pb)·U carries less where an edge
        // crosses it, and the difference is diffracted beyond the grid's band.
        let da = g.cell_area_mm2();
        let (mut inside, mut lost_in, mut beside, mut beyond, mut sent, mut total) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut clipped = false;
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let ix = g.idx(i, j);
                let pw = self.u.data[ix].norm_sqr() * da;
                if pw == 0.0 {
                    continue;
                }
                let f = cov[ix];
                clipped |= f < 1.0;
                let (a, b) = g.xy(i, j);
                let r_b = beam.point(a, b);
                let t = fin(onto_plane(r_b, beam.d, p, n_s).unwrap_or(r_b));
                let carried = (t * f + Cx::real((1.0 - f) * pb)).norm_sqr();
                inside += f * pw;
                lost_in += f * pw * (1.0 - t.norm_sqr());
                beside += (1.0 - f) * (1.0 - pb) * pw;
                beyond += (f * t.norm_sqr() + (1.0 - f) * pb - carried) * pw;
                sent += carried * pw;
                total += pw;
            }
        }
        let (v, depth) = if !clipped {
            hand_over(&self.u, &beam, n_s, out, &fin)
        } else {
            let mut u_in = self.u.clone();
            for (z, f) in u_in.data.iter_mut().zip(&cov) {
                *z *= *f;
            }
            let (mut v, depth) = hand_over(&u_in, &beam, n_s, out, &fin);
            if pass_by {
                let mut u_by = self.u.clone();
                for (z, f) in u_by.data.iter_mut().zip(&cov) {
                    *z *= 1.0 - *f;
                }
                let (v_by, _) = hand_over(&u_by, &beam, n_s, out, &|_| Cx::ONE);
                for (a, b) in v.data.iter_mut().zip(&v_by.data) {
                    *a += *b;
                }
                // The light beside the element keeps its direction, which the
                // outgoing plane holds as a phase ramp across its samples.
                let ramp = k * (beam.d - out.d).length() * g.pitch_mm[0].max(g.pitch_mm[1]);
                if ramp > std::f64::consts::PI {
                    self.report.warn(SamplingWarning::Undersampled {
                        what: format!("the light passing beside #{}", h.object.0),
                        max_phase_step_rad: ramp,
                    });
                }
            }
            (v, depth)
        };
        let l = &mut self.report.ledger;
        match loss {
            Loss::Absorbed => l.absorbed_element += lost_in,
            Loss::Arm(other) => {
                l.other_arm += inside * other;
                l.absorbed_element += (lost_in - inside * other).max(0.0);
            }
            Loss::Orders => l.untracked_orders += lost_in,
        }
        l.passed_by += beside;
        l.beyond_nyquist += beyond;
        l.resampling += sent - v.power();
        // A flat mirror's handover is image theory, exact at any tilt;
        // anything else seen obliquely is a projection.
        let exact = matches!(response, SurfaceResponse::Mirror { .. }) && h.action == SeqAction::Reflect;
        if !exact {
            self.tilt_warn(depth);
        }
        let mut note = match lens {
            Some((f, _)) => format!("thin lens #{} ({}), f = {f} mm", h.object.0, h.label),
            None => format!("{:?} at #{} ({}), amplitude {amp:+.6}", h.action, h.object.0, h.label),
        };
        if clipped {
            note += &format!(
                "; its outline clips the beam, edges weighted by exact cell coverage, {:.3e} of the power inside it{}",
                inside / f64::max(total, f64::MIN_POSITIVE),
                if pass_by { ", the rest passing beside it on the chain" } else { ", the rest passing beside it off the chain" }
            );
        }
        self.report.steps.push(element_step(0.0, note));
        if let Some((f, _)) = lens {
            let dx = g.pitch_mm[0];
            let edge = 0.5 * g.width_mm(0) * 2f64.sqrt();
            if k * edge * dx / f.abs() > std::f64::consts::PI {
                self.report.warn(SamplingWarning::Undersampled {
                    what: format!("thin lens #{} f = {f} mm", h.object.0),
                    max_phase_step_rad: k * edge * dx / f.abs(),
                });
            }
        }
        self.u = v;
        Ok(())
    }
```

### 7.4 Apertures

Apertures follow the tracer's semantics:
- light passes inside the clear opening;
- it is stopped between the opening and the outer extent;
- it passes beyond the outer extent. An aperture with no outer extent is
  unbounded.

The transmittance is `1 − cover(outer) + cover(opening)`, clamped to [0, 1].
Each cover is exact. Square to the beam, a disc or an annulus uses its closed
form; otherwise the projection along the beam is a polygon. An opening that
reaches past its outer extent is named as an approximation.

<!-- excerpt crates/oc-wave/src/path.rs 2336 2391 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2336–2391, commit `1af40e8`.*

```rust
            Ev::Aperture(ai) => {
                let a = &self.scene.apertures[ai];
                let t = &a.common.transform;
                let plane = &a.shape.plane_local;
                let n_e = t.rotate(plane.normal);
                let o_e = t.transform_point(plane.origin);
                let square = n_e.dot(beam.d).abs() > 1.0 - 1e-12;
                // A point of the aperture's plane, along the beam onto the
                // plane across it here.
                let to_beam = |pu: f64, pv: f64| {
                    let w = t.transform_point(plane.origin + plane.u * pu + plane.v * pv);
                    let (x, y) = beam.coords(w - beam.d * (w - beam.origin).dot(beam.d));
                    [x, y]
                };
                // As the tracer has it: light passes inside the opening, is
                // stopped outside it within the outer extent, and passes
                // beyond the outer extent; with none, the mask is unbounded.
                let g = self.u.grid;
                let open = shape_coverage(&g, &a.shape.clear_opening, square, &to_beam)?;
                let pass: Vec<f64> = match &a.shape.outer_extent {
                    None => open,
                    Some(ext) => {
                        let outer = shape_coverage(&g, ext, square, &to_beam)?;
                        let inside = shape_rim(&a.shape.clear_opening)
                            .into_iter()
                            .all(|p| oc_core::geom3::primitives::planar_shape_contains(ext, p));
                        if !inside {
                            self.report.approx(format!(
                                "aperture #{}: its opening reaches past its outer extent, where the two covers are combined as if it did not",
                                a.common.id.0
                            ));
                        }
                        outer.iter().zip(&open).map(|(o, c)| (1.0 - o + c).clamp(0.0, 1.0)).collect()
                    }
                };
                let (stopped, beyond) = crate::elements::apply_coverage(&mut self.u, &pass)?;
                self.report.ledger.absorbed_aperture += stopped;
                self.report.ledger.beyond_nyquist += beyond;
                let id = a.common.id.0;
                if !square {
                    // Not square to the beam, the opening is its projection
                    // along the beam: an approximation by the depth it spans.
                    self.tilt_warn(plane_depth(&beam, &g, o_e, n_e));
                }
                let note = if stopped > 0.0 {
                    format!(
                        "aperture #{} at its plane, edges weighted by exact cell coverage{}: {:.3e} of the power stopped",
                        id,
                        if square { "" } else { " of their projection along the beam" },
                        stopped / (stopped + self.u.power() + beyond).max(f64::MIN_POSITIVE)
                    )
                } else {
                    format!("aperture #{id} at its plane, clear of the beam")
                };
                self.report.steps.push(element_step(0.0, note));
            }
```

### 7.5 A refractive face or a curved surface: geometric field tracing

A lens is not carried through whole by its rays. Inside a beam's Rayleigh
range the wavefront curvature does not grow by the distance travelled.
Instead, **each face is a transfer across a thin slab**, and between faces
the field diffracts inside the glass by the angular spectrum, in the glass's
index.

**The slab.** Rays from the previous event's plane along `d_in` are traced on
24 azimuths and 8 rings out to the window's corners. Along each azimuth the
element's edge is found by bisection (40 steps). Each ray's first meeting
point with the surface is measured from the chief ray's hit:
- along `d_in`, for the entry plane, which is placed 1 µm before the
  earliest point;
- along `d_out`, for the exit plane, which is placed 1 µm past the farthest
  point.

If no plane fits before the next event, the path is refused.

<!-- excerpt crates/oc-wave/src/path.rs 621 677 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 621–677, commit `1af40e8`.*

```rust
/// How far, before the chief ray's hit along `d_in` and after it along
/// `d_out`, the surface reaches where rays over the beam window meet it:
/// rays from `from` along `d_in`, on rings out to the window's corners, and
/// where each first meets the element.
fn slab(probe: &Scene3, from: Vec3, d_in: Vec3, hit: Vec3, d_out: Vec3, half: f64, lam: f64) -> (f64, f64) {
    use oc_trace::{trace3, LaunchRay3, TraceConfig3};
    let cfg = TraceConfig3 { ghosts: false, polarization: false, bounds_mm: oc_trace::bounds_for_scene3(probe), ..TraceConfig3::default() };
    let x = if d_in.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let x = (x - d_in * d_in.dot(x)).normalize();
    let y = d_in.cross(x);
    let (mut s_min, mut s_max) = (0.0f64, 0.0f64);
    let r_max = half * std::f64::consts::SQRT_2;
    // Where the ray from `from`, `r` off the chief ray at azimuth `t`, first
    // meets the element: before the hit along `d_in`, after it along `d_out`.
    let meet = |r: f64, t: f64| -> Option<(f64, f64)> {
        let o = from + x * (r * t.cos()) + y * (r * t.sin());
        let ray = oc_core::vec3::Ray3::new(o, d_in).ok()?;
        let res = trace3(probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
        let root = res.segments.iter().find(|s| s.parent_path_id.is_none())?;
        // A root with a child met the element at its end.
        res.segments.iter().any(|s| s.parent_path_id == Some(root.path_id)).then(|| ((root.b - hit).dot(d_in), (root.b - hit).dot(d_out)))
    };
    // Rings out to the window's corners, and along each azimuth the edge of
    // the element where a ring first misses it, found by bisection: an
    // element smaller than the window reaches its greatest depth at its rim,
    // which no ring need meet.
    for k in 0..24 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / 24.0;
        let mut last = 0.0;
        for ring in 1..=8 {
            let r = r_max * ring as f64 / 8.0;
            if let Some((a, b)) = meet(r, t) {
                s_min = s_min.min(a);
                s_max = s_max.max(b);
                last = r;
                continue;
            }
            if r - last > r_max / 8.0 + 1e-12 {
                continue;
            }
            let (mut lo, mut hi) = (last, r);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if meet(mid, t).is_some() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            if let Some((a, b)) = meet(lo, t) {
                s_min = s_min.min(a);
                s_max = s_max.max(b);
            }
        }
    }
    (s_min, s_max)
}
```

**The map.** On a grid of `g × g` rays across the window (`g` odd,
`clamp(grid_n, 65, 257)`), each ray starts at an entry-plane point `(a, b)`.
It travels along the **reference direction** `t(a, b) = tilt + curv·((a, b) −
centre)` (§7.6), through a copy of the scene that holds the element alone. It
gives:
- where it crosses the exit plane after exactly one interaction;
- its OPL from the entry plane;
- its power transmittance `T` (the tracer's unpolarized Fresnel);
- its direction.

The crossing map `x′(a, b)` is fitted as affine and the OPL as quadratic.
Both are completed by bilinearly interpolated residuals on the ray grid. Then
[Wyrowski & Kuhn 2011]:

```text
U_out(x′) = U_in(x) · √(T(x) / |∂x′/∂x|) · exp(ik₀ · OPL(x)),   x = x′⁻¹(exit sample)
```

The inverse map is found by fixed-point iteration on the fitted affine map.
This conserves power ray tube by ray tube, less what the faces reflect. The
output field carries no carrier.

The phase is assembled from three OPLs:
- `chief.opl` is the OPL, from the entry plane, of the ray at the window
  centre along the reference direction;
- the map stores each ray's OPL relative to that one (`opl`);
- each sample gets `exp(ik₀(chief.opl + opl))`, its ray's whole OPL from the
  entry plane, so the samples stay on free space's clock.

`opl_axis`, the OPL of the axis ray traced along the frame normal, is only
added to the record `opl_ref_mm`. `map.at(a, b)` returns the landing point,
`opl`, `T`, and `det`, the map's Jacobian determinant `∂x′/∂x`.

The bookkeeping:
- **Rays that miss** the face, or do anything the chief ray did not, are
  stopped as by a mount (`absorbed_aperture`).
- **The rim** (the edge of what passes) is found by bisection along 720
  azimuths, to 1e-6 of the ray spacing. It is laid on the arriving field as a
  polygon with exact cell coverage, so it diffracts at the field's own
  sampling, and the map is carried two rays past it. A rim that is not
  star-shaped about the chief ray is resolved only to the ray grid, and named.
- **The faces' reflection** `1 − T` is booked as `other_arm`.

The construction is linear: the map is traced for a reference and a window,
and then applied to any field on that window.

<!-- excerpt crates/oc-wave/src/transfer.rs 640 747 -->
*Excerpt: `crates/oc-wave/src/transfer.rs`, lines 640–747, commit `1af40e8`.*

```rust
pub(crate) fn trace_map(probe: &Scene3, grid: Grid2, lam: f64, inc: &Beam, out: &Beam, rays: usize, r: &Reference) -> Result<TracedMap, WaveError> {
    let cfg = trace_config(probe);
    let trace = |a: f64, b: f64| -> Option<RayOut> { trace_with(probe, &cfg, lam, inc, out, a, b, r.dir(a, b)) };
    let chief = trace(0.0, 0.0)
        .ok_or_else(|| WaveError::OutOfRange("the chief ray does not cross the surface to its exit plane".into()))?;
    let opl_axis = if r.dir(0.0, 0.0) == [0.0, 0.0] { chief.opl } else { trace_with(probe, &cfg, lam, inc, out, 0.0, 0.0, [0.0, 0.0]).map_or(chief.opl, |x| x.opl) };

    // The ray grid: odd, so the chief ray is its centre, spanning the window.
    let g = (rays.max(9) | 1).min(1025);
    let w = grid.width_mm(0).max(grid.width_mm(1));
    let h = w / (g - 1) as f64;
    let half = (g - 1) as f64 / 2.0;
    let mut rays_out: Vec<Option<RayOut>> = Vec::with_capacity(g * g);
    for j in 0..g {
        for i in 0..g {
            let (a, b) = ((i as f64 - half) * h, (j as f64 - half) * h);
            rays_out.push(trace(a, b));
        }
    }
    let passed = rays_out.iter().filter(|r| r.is_some()).count();
    if passed < 9 {
        return Err(WaveError::OutOfRange("fewer than nine rays pass the element".into()));
    }
    // The rim at the field's own sampling, to be laid on the arriving field.
    let rim = if passed < g * g {
        let ok: Vec<bool> = rays_out.iter().map(|r| r.is_some()).collect();
        rim_polygon(&|a, b| trace(a, b).is_some(), &ok, g, h)
    } else {
        None
    };
    // Fit the map and the path on coordinates scaled to the window.
    let s = 0.5 * w;
    let mut rows_x = Vec::with_capacity(passed);
    let mut rows_y = Vec::with_capacity(passed);
    let mut rows_p = Vec::with_capacity(passed);
    for j in 0..g {
        for i in 0..g {
            if let Some(r) = rays_out[j * g + i] {
                let (uu, vv) = ((i as f64 - half) * h / s, (j as f64 - half) * h / s);
                rows_x.push(([uu, vv, 1.0], r.x));
                rows_y.push(([uu, vv, 1.0], r.y));
                rows_p.push(([1.0, uu, vv, uu * uu, uu * vv, vv * vv], r.opl - chief.opl));
            }
        }
    }
    let fail = || WaveError::OutOfRange("the traced map could not be fitted".into());
    let ax = lstsq(&rows_x).ok_or_else(fail)?;
    let ay = lstsq(&rows_y).ok_or_else(fail)?;
    let q = lstsq(&rows_p).ok_or_else(fail)?;
    let mut map = Map { g, h, s, ax, ay, q, rays: rays_out, rx: vec![0.0; g * g], ry: vec![0.0; g * g], rp: vec![0.0; g * g] };
    let mut walk: f64 = 0.0;
    for j in 0..g {
        for i in 0..g {
            let k = j * g + i;
            if let Some(r) = map.rays[k] {
                let (a, b) = ((i as f64 - half) * h, (j as f64 - half) * h);
                let (fx, fy) = map.fit_x(a, b);
                map.rx[k] = r.x - fx;
                map.ry[k] = r.y - fy;
                map.rp[k] = (r.opl - chief.opl) - map.fit_p(a, b);
                walk = walk.max(((r.x - a).powi(2) + (r.y - b).powi(2)).sqrt());
            }
        }
    }
    // Past a resolved rim, the map carried two rays on: each failed ray next
    // to a passing one takes its neighbours' mean residuals and transmittance.
    if rim.is_some() {
        for _ in 0..2 {
            let live: Vec<bool> = map.rays.iter().map(|r| r.is_some()).collect();
            for j in 0..g {
                for i in 0..g {
                    let k = j * g + i;
                    if live[k] {
                        continue;
                    }
                    let (mut n, mut sx, mut sy, mut sp, mut st) = (0.0, 0.0, 0.0, 0.0, 0.0);
                    for dj in -1i64..=1 {
                        for di in -1i64..=1 {
                            let (ii, jj) = (i as i64 + di, j as i64 + dj);
                            if ii < 0 || jj < 0 || ii >= g as i64 || jj >= g as i64 {
                                continue;
                            }
                            let kk = jj as usize * g + ii as usize;
                            if let (true, Some(r)) = (live[kk], map.rays[kk]) {
                                n += 1.0;
                                sx += map.rx[kk];
                                sy += map.ry[kk];
                                sp += map.rp[kk];
                                st += r.t;
                            }
                        }
                    }
                    if n > 0.0 {
                        let (a, b) = ((i as f64 - half) * h, (j as f64 - half) * h);
                        let (fx, fy) = map.fit_x(a, b);
                        let fp = map.fit_p(a, b);
                        let (rx, ry, rp) = (sx / n, sy / n, sp / n);
                        map.rx[k] = rx;
                        map.ry[k] = ry;
                        map.rp[k] = rp;
                        map.rays[k] = Some(RayOut { x: fx + rx, y: fy + ry, opl: chief.opl + fp + rp, t: st / n, dx: 0.0, dy: 0.0 });
                    }
                }
            }
        }
    }
    Ok(TracedMap { map, chief, opl_axis, rim, grid, reference: *r, passed, walk, slab: (out.origin - inc.origin).length() })
}
```

<!-- excerpt crates/oc-wave/src/transfer.rs 479 553 -->
*Excerpt: `crates/oc-wave/src/transfer.rs`, lines 479–553, commit `1af40e8`.*

```rust
/// The edge of the region of the entry plane whose rays pass the element, as
/// a polygon of 720 vertices about the chief ray: along each azimuth, the
/// first ray of the grid that fails brackets the edge, which bisection on
/// traced rays then finds to a millionth of the ray spacing. `None` when the
/// region is not star-shaped about the chief ray (a ray passes again beyond
/// the edge), or the bracket does not hold.
pub(crate) fn rim_polygon(pass: &dyn Fn(f64, f64) -> bool, grid: &[bool], g: usize, h: f64) -> Option<Vec<[f64; 2]>> {
    let half = (g - 1) as f64 / 2.0;
    let r_max = half * h * std::f64::consts::SQRT_2;
    let on_grid = |a: f64, b: f64| -> Option<bool> {
        let (i, j) = ((a / h + half).round(), (b / h + half).round());
        if i < 0.0 || j < 0.0 || i > (g - 1) as f64 || j > (g - 1) as f64 {
            return None;
        }
        Some(grid[j as usize * g + i as usize])
    };
    const M: usize = 720;
    let mut poly = Vec::with_capacity(M);
    for k in 0..M {
        let th = 2.0 * std::f64::consts::PI * k as f64 / M as f64;
        let (c, s) = (th.cos(), th.sin());
        let mut r_fail = None;
        let mut r = 0.0;
        while r <= r_max {
            match on_grid(r * c, r * s) {
                None => break,
                Some(true) => {}
                Some(false) => {
                    r_fail = Some(r);
                    break;
                }
            }
            r += 0.5 * h;
        }
        let Some(rf) = r_fail else {
            // Rays pass out to the window's edge along this azimuth.
            poly.push([2.0 * r_max * c, 2.0 * r_max * s]);
            continue;
        };
        let mut q = rf;
        while q <= r_max {
            if on_grid(q * c, q * s) == Some(true) {
                return None;
            }
            q += 0.5 * h;
        }
        let mut lo = (rf - 1.5 * h).max(0.0);
        while lo > 0.0 && !pass(lo * c, lo * s) {
            lo = (lo - h).max(0.0);
        }
        if !pass(lo * c, lo * s) {
            return None;
        }
        let mut hi = rf + 0.5 * h;
        let mut tries = 0;
        while pass(hi * c, hi * s) {
            hi += h;
            tries += 1;
            if tries > 4 {
                return None;
            }
        }
        while hi - lo > 1e-6 * h {
            let mid = 0.5 * (lo + hi);
            if pass(mid * c, mid * s) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let re = 0.5 * (lo + hi);
        poly.push([re * c, re * s]);
    }
    Some(poly)
}
```

<!-- excerpt crates/oc-wave/src/transfer.rs 749 816 -->
*Excerpt: `crates/oc-wave/src/transfer.rs`, lines 749–816, commit `1af40e8`.*

```rust
/// Carry `u`, on the window `m` was traced for, through it into a medium of
/// index `n_out` across `out`: linear in `u`.
pub(crate) fn apply(m: &TracedMap, u: &Field2, out: &Beam, n_out: f64) -> Result<(Field2, TransferStats), WaveError> {
    if u.grid != m.grid {
        return Err(WaveError::OutOfRange("a traced map applied to a field on another window".into()));
    }
    let lam = u.lambda_um;
    let gu = u.grid;
    let k_in = u.k_mm();
    let map = &m.map;
    let mut um = u.clone();
    let (rim_stopped, rim_beyond) = match &m.rim {
        Some(poly) => crate::elements::apply_coverage(&mut um, &crate::elements::polygon_coverage(&u.grid, poly))?,
        None => (0.0, 0.0),
    };
    let u = &um;
    let (angle, rms) = departure(u, &m.reference);

    // The arriving field, carrier included, read between its samples with
    // the reference's phase taken out (`Smooth`).
    let smooth = Smooth::about(u, &m.reference);
    let full = |a: f64, b: f64| smooth.at(a, b) * Cx::expi(k_in * (u.carrier[0] * a + u.carrier[1] * b));

    // Where the arriving power goes: stopped, reflected, out of the window.
    let cell = gu.cell_area_mm2();
    let (mut stopped, mut reflected, mut left) = (0.0, 0.0, 0.0);
    let (wx, wy) = (0.5 * gu.width_mm(0), 0.5 * gu.width_mm(1));
    for j in 0..gu.n[1] {
        for i in 0..gu.n[0] {
            let p = u.at(i, j).norm_sqr() * cell;
            if p == 0.0 {
                continue;
            }
            let (a, b) = gu.xy(i, j);
            match map.at(a, b) {
                None => stopped += p,
                Some((x, y, _, t, _)) => {
                    reflected += p * (1.0 - t.clamp(0.0, 1.0));
                    if (x - gu.center_mm[0]).abs() > wx || (y - gu.center_mm[1]).abs() > wy {
                        left += p * t.clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    // The field that leaves, sample by sample across the exit plane.
    let k0 = 2.0 * std::f64::consts::PI / (lam / 1000.0);
    let chief = m.chief;
    let mut v = u.clone();
    v.carrier = [0.0, 0.0];
    v.n_medium = n_out;
    v.frame = out.iso();
    v.opl_ref_mm = u.opl_ref_mm + m.opl_axis;
    let mut paths = vec![f64::NAN; gu.n[0] * gu.n[1]];
    for j in 0..gu.n[1] {
        for i in 0..gu.n[0] {
            let (x, y) = gu.xy(i, j);
            let idx = gu.idx(i, j);
            v.data[idx] = match map.inverse(x, y).and_then(|(a, b)| map.at(a, b).map(|r| (a, b, r))) {
                Some((a, b, (_, _, opl, t, det))) if det.abs() > 1e-12 => {
                    paths[idx] = opl;
                    full(a, b) * (t.clamp(0.0, 1.0) / det.abs()).sqrt() * Cx::expi(k0 * (chief.opl + opl))
                }
                _ => Cx::ZERO,
            };
        }
    }
```

**Departure from the reference.** The rays follow the reference, not the
field. A field component tilted `θ` from its rays is traced along the rays'
straight run, where its own plane waves advance by `k cos θ`. So it gathers
the wrong phase: a piston of about `kθ²L/2` over a slab of depth `L`, and a
lateral error of about `θL`.

Each face reports two departures of the arriving field's local directions from
the reference:
- the steepest, where the field holds at least a hundredth of its peak
  intensity;
- the power-weighted rms, where it holds at least 1e-4.

It also reports what those move across the slab. The reference's own phase is
taken out before the gradient is formed.

<!-- excerpt crates/oc-wave/src/transfer.rs 442 477 -->
*Excerpt: `crates/oc-wave/src/transfer.rs`, lines 442–477, commit `1af40e8`.*

```rust
/// How far `u`'s directions depart from a reference's, rad: the steepest
/// where the field holds a hundredth of its peak, and the power-weighted rms
/// where it holds a ten-thousandth. The reference's phase is taken out first,
/// so that only the departure, not the field's whole tilt, has to be resolved
/// by the samples.
pub(crate) fn departure(u: &Field2, r: &Reference) -> (f64, f64) {
    let gu = u.grid;
    let k = u.k_mm();
    let top = u.data.iter().fold(0.0f64, |m, c| m.max(c.norm_sqr()));
    let mut worst: f64 = 0.0;
    let (mut sw, mut s2) = (0.0, 0.0);
    if top <= 0.0 || top.is_nan() {
        return (0.0, 0.0);
    }
    let rest = |i: usize, j: usize| {
        let (a, b) = gu.xy(i, j);
        u.at(i, j) * Cx::expi(-ref_phase(k, r, u.carrier, a, b))
    };
    for j in 1..gu.n[1].saturating_sub(1) {
        for i in 1..gu.n[0].saturating_sub(1) {
            let w = u.at(i, j).norm_sqr();
            if w < 1e-4 * top {
                continue;
            }
            let gx = (rest(i + 1, j) * rest(i - 1, j).conj()).arg() / (2.0 * gu.pitch_mm[0]);
            let gy = (rest(i, j + 1) * rest(i, j - 1).conj()).arg() / (2.0 * gu.pitch_mm[1]);
            let d = (gx / k).hypot(gy / k);
            sw += w;
            s2 += w * d * d;
            if w >= 1e-2 * top {
                worst = worst.max(d);
            }
        }
    }
    (worst, (s2 / sw).sqrt())
}
```

<!-- excerpt crates/oc-wave/src/path.rs 2264 2289 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2264–2289, commit `1af40e8`.*

```rust
            Ev::Transfer { hit, exit, d_out } => {
                let first = &hit;
                let last = &hit;
                // The scene with the element alone: nothing else can meet its rays.
                let probe = probe_scene(&self.scene, hit.object);
                let out = beam.turned(exit, d_out);
                let before = self.u.power();
                let rays = self.spec.grid_n.clamp(65, 257);
                let r = self.reference();
                let map = match &self.cache {
                    Some((i, m)) if *i == idx && m.fits(self.u.grid, &r) => Arc::clone(m),
                    _ => {
                        let m = Arc::new(crate::transfer::trace_map(&probe, self.u.grid, self.u.lambda_um, &beam, &out, rays, &r)?);
                        self.cache = Some((idx, Arc::clone(&m)));
                        m
                    }
                };
                let (v, st) = crate::transfer::apply(&map, &self.u, &out, last.n_after)?;
                let l = &mut self.report.ledger;
                // The rim and the mount stop light; the faces reflect it into
                // a path the chain does not follow.
                l.absorbed_aperture += st.rim_stopped + st.stopped;
                l.beyond_nyquist += st.rim_beyond;
                l.other_arm += st.reflected;
                l.left_window += st.left_window;
                l.resampling += st.carried - st.stopped - st.reflected - st.left_window - v.power();
```

### 7.6 The reference wavefront: a Gaussian `q` through the geometry

The rays a face is traced with follow a wavefront fixed by the geometry, not
by the arriving field. This makes the transfer one linear operator,
`T(u₁ + u₂) = T(u₁) + T(u₂)`; with a sphere fitted to each field instead, it
was not.

The path carries the beam's Gaussian parameter per frame axis,
`q = z + i z_R` in the medium, from the declared beam through every element
before:

| Element | Update of `q` |
|---|---|
| free space of length `z` | `q ← q + z` |
| ideal thin lens | `1/q ← 1/q − 1/f` |
| plate | `q̂ ← q̂ + (d₁/n + L_g/n_g + d₂/n)` in reduced form `q̂ = q/n`; `L_g` is the chief ray's path in the glass, `d₁`, `d₂` its paths from the entry plane to the first face and from the second face to the exit plane |
| traced face | reduced ABCD: `q̂′ = (Aq̂ + B)/(Cq̂ + D)`, `q̂ = q/n` [Siegman 1986] |
| anything else (mirror, splitter, grating, aperture, DMD, LCoS) | unchanged |

The traced face's ABCD comes from five rays per face: the chief ray, one
offset by ε = 1 µm and one tilted by δ = 1e-4 per axis. Cross terms between
the axes are dropped.

The reference is the wavefront `Re(1/q)` about a reference ray. That ray is
the chief ray, or, for a modulator's order carried apart (§7.7), its own
point and direction. That ray is itself traced through each face and turned
by thin lenses. A top-hat's `q` is the Gaussian of its radius (§5.3).

<!-- excerpt crates/oc-wave/src/path.rs 2183 2256 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2183–2256, commit `1af40e8`.*

```rust
    /// The reference the next transfer's rays follow: the beam's paraxial
    /// wavefront about the reference ray.
    fn reference(&self) -> Reference {
        let c = |q: Cx| (Cx::ONE / q).re;
        Reference { curv: [c(self.q[0]), c(self.q[1])], tilt: [self.at[2], self.at[3]], centre: [self.at[0], self.at[1]] }
    }

    /// The reference's wavefront and ray through the element at an event,
    /// from the geometry alone: a transfer's paraxial matrices from rays
    /// traced through it, a thin lens's power, a plate as the free space its
    /// glass is worth (`t/n`); a flat element hands the beam's frame on as it
    /// is. `beam_in` and `n_in` are the beam's frame and medium before it.
    fn paraxial(&mut self, ev: &Ev, p: Vec3, beam_in: &Beam, n_in: f64) -> Result<(), WaveError> {
        match ev {
            Ev::Transfer { hit, .. } => {
                let probe = probe_scene(&self.scene, hit.object);
                let out = self.beam;
                let n_out = hit.n_after;
                if let Some(m) = crate::transfer::abcd(&probe, self.lam, beam_in, &out, n_in, n_out) {
                    for (q, m) in self.q.iter_mut().zip(m) {
                        *q = crate::transfer::q_through(*q, m, n_in, n_out);
                    }
                } else {
                    self.report.approx(format!("#{}: its paraxial matrices could not be traced, so the reference wavefront is carried through it unchanged", hit.object.0));
                }
                let mut ats: Vec<[f64; 4]> = vec![self.at];
                ats.extend(self.rest.iter().map(|l| l.at));
                for a in &mut ats {
                    if *a == [0.0; 4] {
                        continue;
                    }
                    match crate::transfer::ray_through(&probe, self.lam, beam_in, &out, a[0], a[1], [a[2], a[3]]) {
                        Some((x, t, _)) => *a = [x[0], x[1], t[0], t[1]],
                        None => self.report.approx(format!("#{}: the reference ray of one of the field's orders misses it, and is carried on as it was", hit.object.0)),
                    }
                }
                self.at = ats[0];
                for (l, a) in self.rest.iter_mut().zip(&ats[1..]) {
                    l.at = *a;
                }
            }
            Ev::Hit(h) => {
                let s = self.scene.surfaces.iter().find(|s| s.common.id == h.object);
                let lens = s.and_then(|s| {
                    let r = if beam_in.d.dot(s.common.transform.rotate(Vec3::Z)) < 0.0 { &s.response_front } else { &s.response_back };
                    match r {
                        SurfaceResponse::ThinLens { focal_length_mm } => Some(*focal_length_mm),
                        _ => None,
                    }
                });
                if let Some(f) = lens {
                    for q in &mut self.q {
                        *q = Cx::ONE / (Cx::ONE / *q - Cx::real(1.0 / f));
                    }
                    // The frame turns with the chief ray: a reference ray
                    // turns by its own height on the lens less the chief's.
                    for a in std::iter::once(&mut self.at).chain(self.rest.iter_mut().map(|l| &mut l.at)) {
                        a[2] -= a[0] / f;
                        a[3] -= a[1] / f;
                    }
                }
            }
            Ev::Plate { first, second, exit, plate } => {
                let (n_g, n_out) = (plate.n_glass, plate.n_out);
                let glass = (second.point - first.point).length();
                let worth = (first.point - p).length() / n_in + glass / n_g + (*exit - second.point).length() / n_out;
                for q in &mut self.q {
                    *q = (*q * (1.0 / n_in) + Cx::real(worth)) * n_out;
                }
            }
            _ => {}
        }
        Ok(())
    }
```

<!-- excerpt crates/oc-wave/src/transfer.rs 585 611 -->
*Excerpt: `crates/oc-wave/src/transfer.rs`, lines 585–611, commit `1af40e8`.*

```rust
/// The element's paraxial matrices about the chief ray, `[A, B, C, D]` for
/// each axis of the beam frames, in reduced angles (`n·θ`): from rays moved
/// `ε` across the entry plane and tilted `δ`, differenced against the chief
/// ray. Cross terms between the axes are left out.
pub(crate) fn abcd(probe: &Scene3, lam: f64, inc: &Beam, out: &Beam, n_in: f64, n_out: f64) -> Option<[[f64; 4]; 2]> {
    let (eps, del) = (1e-3, 1e-4);
    let o = trace_one(probe, lam, inc, out, 0.0, 0.0, [0.0, 0.0])?;
    let mut m = [[0.0; 4]; 2];
    for (axis, row) in m.iter_mut().enumerate() {
        let (pa, pb) = if axis == 0 { (eps, 0.0) } else { (0.0, eps) };
        let tilt = if axis == 0 { [del, 0.0] } else { [0.0, del] };
        let rx = trace_one(probe, lam, inc, out, pa, pb, [0.0, 0.0])?;
        let rt = trace_one(probe, lam, inc, out, 0.0, 0.0, tilt)?;
        let pos = |r: &RayOut| if axis == 0 { r.x - o.x } else { r.y - o.y };
        let dir = |r: &RayOut| if axis == 0 { r.dx - o.dx } else { r.dy - o.dy };
        *row = [pos(&rx) / eps, pos(&rt) / (n_in * del), n_out * dir(&rx) / eps, n_out * dir(&rt) / (n_in * del)];
    }
    Some(m)
}

/// A Gaussian beam parameter `q` (in a medium of index `n_in`) through
/// reduced paraxial matrices into one of index `n_out`: `q̂ = q/n`,
/// `q̂' = (A q̂ + B)/(C q̂ + D)` (A. E. Siegman, *Lasers* (1986), ch. 15).
pub(crate) fn q_through(q: Cx, m: [f64; 4], n_in: f64, n_out: f64) -> Cx {
    let qh = q * (1.0 / n_in);
    (qh * m[0] + Cx::real(m[1])) / (qh * m[2] + Cx::real(m[3])) * n_out
}
```

### 7.7 A DMD or an LCoS on the path

**Laying the beam onto the device.** The device grid has spacing `p/per`:

```text
per = max(device_samples_per_pixel, 2, min(⌈2p/Δx_beam⌉, 32))
```

so it is never coarser than half the beam's own sample spacing (up to 32
samples a pixel). It covers the beam window's footprint on the device plane,
plus two samples.

Each device sample at device-plane point `r` takes the beam's field at its
projection along the beam direction `d`, with no propagation across the
plane's depth. Its carrier is `t_d`, the beam direction's tangential part in
the device frame. On the device plane, the beam's reference wave
`e^{ik d·(r − o)}` (`o` the beam frame's origin) and the carrier's
`e^{ik t_d·r_local}` differ by a phase. Each sample is multiplied by
`exp(ik(d·(r − o) − t_d·r_local))`, so that sample times carrier is the
true field.

**The device model** is then applied, family by family:
- **A DMD**: the family of the pixel the chief ray hits (§6.3).
- **An LCoS**: the modulated beam when the chain diffracted there, the
  unmodulated beam otherwise (§6.5). The chain must carry two polarization
  components (§7.10).

**Band limit before the take-up.** The device grid is finer than the
outgoing window. The plane waves beyond the outgoing window's band are taken
out first, about the reflected family's carrier, and booked as
`beyond_nyquist`. Along a device axis `e` the outgoing window's sample
spacing is `Δx_beam/√(1 − (e·d′)²)`.

Before commit `1af40e8` this step was missing. In the DMD-and-lens test, the
DMD facets then folded 17.4% of the power back into the band through the
bilinear take-up, booked as `resampling`, and the weak orders were weighted
wrongly. With the cut, the resampling is 1.3e-4.

**The take-up.** Each outgoing sample is projected along `d′` onto the device
plane and read bilinearly, after the incident field's fitted sphere is
removed. It is multiplied by the inverse phase,
`exp(−ik(d′·(r_e − o′) − t_s·r_local))`, which turns the reflected family's
carrier `t_s` back into the outgoing beam's reference. It is then multiplied
by `√(cos θ_in/cos θ_out)`. The device's ledger is rescaled from
device-plane units by `P_beam/P_device`.

The device grid's extent uses the beam window's axis-0 width for both axes
(the windows are square).

In the take-up code's comment, "`p` apart" means the beam window's pitch
(`pb` in the code), not the device pitch.

<!-- excerpt crates/oc-wave/src/path.rs 3398 3536 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 3398–3536, commit `1af40e8`.*

```rust
    let n_s = t.rotate(Vec3::Z);
    // The device-plane grid: the beam window's footprint, at a few samples a
    // pixel, and no coarser than the beam's own samples (a window fitted to a
    // narrow beam is finer than a few samples a pixel, and the take-up would
    // read the device's facets between samples it does not have), up to 32.
    let p = device.pitch_mm();
    let per = spec.device_samples_per_pixel.max(2).max(((2.0 * p / u.grid.pitch_mm[0].min(u.grid.pitch_mm[1])).ceil() as usize).min(32));
    let dxd = p / per as f64;
    let half = 0.5 * u.grid.width_mm(0);
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for (a, b) in [(-half, -half), (half, -half), (-half, half), (half, half)] {
        if let Some(r) = onto_plane(beam.point(a, b), beam.d, beam.origin, n_s) {
            let l = t.inverse_transform_point(r);
            lo = [lo[0].min(l.x), lo[1].min(l.y)];
            hi = [hi[0].max(l.x), hi[1].max(l.y)];
        }
    }
    let n = [((hi[0] - lo[0]) / dxd).ceil() as usize + 2, ((hi[1] - lo[1]) / dxd).ceil() as usize + 2];
    let center = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let g = Grid2::new(n, [dxd, dxd], center)?;
    // The incident field on the device: data from the beam plane, carrier the
    // beam's direction seen in the device frame.
    let d_local = t.quat().inverse() * beam.d;
    let mut on_dev = Field2::zeros(g, u.lambda_um)?.in_medium(u.n_medium)?;
    on_dev.carrier = [d_local.x, d_local.y];
    let k = u.k_mm();
    let smooth = Smooth::new(u);
    for j in 0..n[1] {
        for i in 0..n[0] {
            let (x, y) = g.xy(i, j);
            let r = t.transform_point(Vec3::new(x, y, 0.0));
            let r_in = r - beam.d * (r - beam.origin).dot(beam.d);
            let (a, b) = beam.coords(r_in);
            // The beam's reference wave, exp(ik d·(r − o)), is the carrier
            // plus a constant on the device plane; take the constant out.
            let c = Cx::expi(k * (beam.d.dot(r - beam.origin) - (d_local.x * x + d_local.y * y)));
            on_dev.data[g.idx(i, j)] = smooth.at(a, b) * c;
        }
    }
    let before = u.power();
    let dev_in = on_dev.power();
    let (reflected, rep) = match &device.kind {
        ModulatorKind::Dmd(_) => {
            // The chain followed the family of the pixel the chief ray hit.
            let l = t.inverse_transform_point(h.point);
            let p = device.pitch_mm();
            let fi = ((l.x / p + device.pixels[0] as f64 / 2.0).floor().max(0.0) as usize).min(device.pixels[0] as usize - 1);
            let fj = ((l.y / p + device.pixels[1] as f64 / 2.0).floor().max(0.0) as usize).min(device.pixels[1] as usize - 1);
            dmd_reflect(&on_dev, device, DmdState::from_value(device.value(fi, fj)))?
        }
        ModulatorKind::Lcos(_) => {
            let beam_kind = if h.action == SeqAction::Diffract { LcosBeam::Modulated } else { LcosBeam::Unmodulated };
            // Of what the device reflects, the part along its director is
            // modulated and the part across it is not (Zhang, You & Chu 2014):
            // the chain brings the one component its beam takes
            // (`element_polarized`), the other booked there.
            if pol != Pol::Projected {
                return Err(WaveError::OutOfRange(format!(
                    "LCoS #{}: met by a field of one component, where it needs the two its beams split",
                    s.common.id.0
                )));
            }
            report.approx("LCoS on the wave path: the field's two polarization components split on the director, each into its own beam; the two never interfere");
            lcos_reflect(&on_dev, device, beam_kind)?
        }
    };
    // The device grid, finer than the outgoing window, holds plane waves that
    // window cannot: read between device samples, they would fold back in at
    // directions they do not have, and weight the orders wrongly. Take them
    // out first, about the reflected family's carrier, as a re-grid does,
    // and book them. The window's samples, `p` apart across the outgoing
    // ray, are `p/√(1 − (e·d)²)` apart along a device axis `e` on its plane.
    let scale = if dev_in > 0.0 { before / dev_in } else { 0.0 };
    let pb = u.grid.pitch_mm[0].max(u.grid.pitch_mm[1]);
    let along = |e: Vec3| pb / (1.0 - e.dot(out.d).powi(2)).max(1e-12).sqrt();
    let band = Grid2::new(reflected.grid.n, [along(t.rotate(Vec3::X)), along(t.rotate(Vec3::Y))], [0.0, 0.0])?;
    let (reflected, cut) = band_to(&reflected, &band, &mut Fft2::new());
    // Take the reflected field up across the outgoing ray. Its samples are
    // the incident ones times a piston per pixel, so the incident field's
    // sphere is the one to take out of them.
    let ([c_dev, a_dev, b_dev], _) = fit_sphere(&on_dev);
    let smooth_out = Smooth::with(&reflected, [c_dev, a_dev - on_dev.carrier[0], b_dev - on_dev.carrier[1]]);
    let mut v = u.clone();
    let gb = u.grid;
    let mut depth: f64 = 0.0;
    let ob = obliquity(beam.d, out.d, n_s);
    for j in 0..gb.n[1] {
        for i in 0..gb.n[0] {
            let (a, b) = gb.xy(i, j);
            let r_out = out.point(a, b);
            let val = match onto_plane(r_out, out.d, out.origin, n_s) {
                Some(r_e) => {
                    depth = depth.max((r_out - r_e).length());
                    let l = t.inverse_transform_point(r_e);
                    // Device-plane data carries the outgoing family's
                    // carrier; across the outgoing ray that carrier is the
                    // reference wave, and the constant is put back.
                    let c = Cx::expi(-k * (out.d.dot(r_e - out.origin) - (reflected.carrier[0] * l.x + reflected.carrier[1] * l.y)));
                    smooth_out.at(l.x, l.y) * c * ob
                }
                None => Cx::ZERO,
            };
            v.data[gb.idx(i, j)] = val;
        }
    }
    if depth > gb.pitch_mm[0] * gb.pitch_mm[0] / u.wavelength_in_medium_mm() {
        report.warn(SamplingWarning::TiltProjection { depth_mm: depth });
    }
    // The device's own ledger, in this beam's units: the device plane sees the
    // beam's power spread over its oblique footprint, so its samples sum to
    // more than the beam's, by the scale below. What the take-up across the
    // outgoing ray does not carry is resampling.
    report.ledger.absorbed_element += rep.ledger.absorbed_element * scale;
    report.ledger.other_family += rep.ledger.other_family * scale;
    report.ledger.beyond_nyquist += (rep.ledger.beyond_nyquist + cut) * scale;
    report.ledger.numerical += rep.ledger.numerical * scale;
    report.ledger.resampling += (rep.ledger.detected - cut) * scale - v.power();
    for f in rep.validity {
        report.flag(f);
    }
    for a in rep.approximated {
        report.approx(a);
    }
    report.steps.push(element_step(
        0.0,
        format!(
            "{} #{} ({}): laid on {}×{} device samples, {} per pixel; {:.3e} of the power, beyond the outgoing window's band, taken out before the take-up",
            oc_core::build3::response_kind(&s.response_front),
            h.object.0,
            h.label,
            n[0],
            n[1],
            per,
            cut * scale / before.max(f64::MIN_POSITIVE)
        ),
    ));
    Ok(v)
}
```

**Orders carried apart.** A modulator's field holds several directions at
once, its orders. Where a traced face lies ahead, the outgoing field's angular
spectrum is cut into the **cells of the device's lattice of orders**:
directions `λ/p` apart in the device plane [Harvey & Vernold 1998]. The
lattice is centred on:
- the incident beam's specular reflection off the array plane, for a DMD,
  whose facets only weight the orders;
- the followed beam, for an LCoS.

Each plane wave goes to the nearest lattice point. Then:
- At most 9 orders are kept, each holding at least 1e-4 of the power. Every
  other order's plane waves join the nearest kept order.
- Each kept order is a field of its own, with a reference ray along its own
  direction.
- They share the windows and propagate apart. An order that free space has
  emptied (under 1e-9 of the total) is added into the main field, which is
  the order that was strongest at the split.
- At each later event every order passes through the element about its own
  reference.
- They are summed back into one field when no traced face lies ahead, and at
  the detector.

The orders carried apart are the **lattice's**. A hologram's own orders,
inside one lattice cell, share their cell's reference, and their departure
from it is reported.

In `split_orders`:
- `nrm` is the device normal, and `ex`, `ey` its axes in the world;
- `centre` is the lattice's centre direction, with `(cx, cy)` its components
  along `ex`, `ey`;
- `step = λ/p` is the lattice spacing in direction cosine;
- `side` is the side of the device the orders leave on;
- `sx`, `sy` are the two components' spectra;
- `MAX_LOBES = 9`.

<!-- excerpt crates/oc-wave/src/path.rs 1967 2088 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1967–2088, commit `1af40e8`.*

```rust
    /// Where a modulator's field meets a traced element further on, carry
    /// its orders apart: the field's angular spectrum cut into the cells of
    /// the device's lattice of orders (directions `λ/p` apart in the device's
    /// plane [Harvey & Vernold 1998], about the specular reflection for a DMD,
    /// whose facets only weight the orders, and about the beam the chain
    /// follows for an LCoS), each order a field of its own with its own
    /// reference ray, its direction. A transfer then traces each order along
    /// its own direction rather than all of them along one; the fields add
    /// back to the one field at the detector.
    fn split_orders(&mut self, h: &oc_core::seq3::SeqHit3, beam_in: &Beam) -> Result<(), WaveError> {
        let scene = Arc::clone(&self.scene);
        let Some(s) = scene.surfaces.iter().find(|s| s.common.id == h.object) else { return Ok(()) };
        let t = s.common.transform;
        let nrm = t.rotate(Vec3::Z);
        let response = if beam_in.d.dot(nrm) < 0.0 { &s.response_front } else { &s.response_back };
        let SurfaceResponse::Modulator { device, .. } = response else { return Ok(()) };
        self.merge_lobes();
        self.at = [0.0; 4];
        let (ex, ey) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
        let out = self.beam;
        let centre = match device.kind {
            ModulatorKind::Dmd(_) => beam_in.d - nrm * (2.0 * beam_in.d.dot(nrm)),
            ModulatorKind::Lcos(_) => out.d,
        };
        let side = if out.d.dot(nrm) >= 0.0 { 1.0 } else { -1.0 };
        let lam = self.u.wavelength_in_medium_mm();
        let step = lam / device.pitch_mm();
        let (cx, cy) = (centre.dot(ex), centre.dot(ey));
        let g = self.u.grid;
        let n = g.n;
        let spectrum = |f: &Field2, fft: &mut Fft2| {
            let mut b = f.data.clone();
            crate::fft::ifftshift2(&mut b, n);
            fft.forward(&mut b, n, g.cell_area_mm2());
            b
        };
        let sx = spectrum(&self.u, &mut self.fft);
        let sy = self.uy.as_ref().map(|u| spectrum(u, &mut self.fft));
        // Each plane wave's order: the lattice point nearest its direction's
        // part along the device's plane.
        let (u0, v0) = (self.u.carrier[0] / lam, self.u.carrier[1] / lam);
        let mut order = vec![(0i64, 0i64); n[0] * n[1]];
        let mut power: std::collections::HashMap<(i64, i64), f64> = std::collections::HashMap::new();
        for ky in 0..n[1] {
            let fy = g.fft_freq(1, ky) + v0;
            for kx in 0..n[0] {
                let fx = g.fft_freq(0, kx) + u0;
                let (tx, ty) = (lam * fx, lam * fy);
                let tz = (1.0 - tx * tx - ty * ty).max(0.0).sqrt();
                let d = out.x * tx + out.y * ty + out.d * tz;
                let o = (((d.dot(ex) - cx) / step).round() as i64, ((d.dot(ey) - cy) / step).round() as i64);
                let k = ky * n[0] + kx;
                order[k] = o;
                *power.entry(o).or_default() += sx[k].norm_sqr() + sy.as_ref().map_or(0.0, |y| y[k].norm_sqr());
            }
        }
        let total: f64 = power.values().sum();
        let mut ranked: Vec<((i64, i64), f64)> = power.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        let kept: Vec<(i64, i64)> = ranked.iter().filter(|(_, p)| *p >= 1e-4 * total).take(MAX_LOBES).map(|x| x.0).collect();
        // An order's direction across the outgoing beam.
        let dir = |o: (i64, i64)| -> Option<[f64; 2]> {
            let (a, b) = (cx + o.0 as f64 * step, cy + o.1 as f64 * step);
            let z2 = 1.0 - a * a - b * b;
            (z2 > 0.0).then(|| {
                let d = ex * a + ey * b + nrm * (side * z2.sqrt());
                [d.dot(out.x), d.dot(out.y)]
            })
        };
        let kept: Vec<((i64, i64), [f64; 2])> = kept.into_iter().filter_map(|o| dir(o).map(|d| (o, d))).collect();
        if kept.len() <= 1 {
            // One order: the field is traced about its direction.
            if let Some((_, d)) = kept.first() {
                self.at = [0.0, 0.0, d[0], d[1]];
            }
            return Ok(());
        }
        // Every order joins its nearest kept one.
        let mut join: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();
        let nearest = |o: (i64, i64)| {
            kept.iter()
                .enumerate()
                .min_by_key(|(_, (k, _))| (k.0 - o.0).pow(2) + (k.1 - o.1).pow(2))
                .map_or(0, |(i, _)| i)
        };
        let df = g.freq_pitch(0) * g.freq_pitch(1);
        let mut lobe = |spec: &[Cx], which: usize, like: &Field2, fft: &mut Fft2| -> Field2 {
            let mut b: Vec<Cx> = spec
                .iter()
                .zip(&order)
                .map(|(z, o)| if *join.entry(*o).or_insert_with(|| nearest(*o)) == which { *z } else { Cx::ZERO })
                .collect();
            fft.inverse(&mut b, n, df);
            crate::fft::fftshift2(&mut b, n);
            let mut f = like.clone();
            f.data = b;
            f
        };
        let mut lobes = Vec::with_capacity(kept.len());
        for (i, (_, d)) in kept.iter().enumerate() {
            let u = lobe(&sx, i, &self.u, &mut self.fft);
            let uy = match (&sy, &self.uy) {
                (Some(y), Some(like)) => Some(lobe(y, i, like, &mut self.fft)),
                _ => None,
            };
            lobes.push(Lobe { u, uy, at: [0.0, 0.0, d[0], d[1]] });
        }
        let first = lobes.remove(0);
        self.u = first.u;
        self.uy = first.uy;
        self.at = first.at;
        self.rest = lobes;
        let count = self.rest.len() + 1;
        if let Some(st) = self.report.steps.last_mut() {
            st.note += &format!("; its field carried on as {count} of the device's orders apart, each traced about its own direction through the elements ahead");
        }
        self.report.approx(format!(
            "the field leaving modulator #{} carried as {count} of its orders apart (the device's lattice, λ/p apart; at most {MAX_LOBES}, each of at least 1e-4 of the power, a weaker order joining its nearest), each traced through the lenses ahead about its own direction; their sum is the field",
            h.object.0
        ));
        Ok(())
    }
```

<!-- excerpt crates/oc-wave/src/path.rs 1895 1965 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1895–1965, commit `1af40e8`.*

```rust
    fn event(&mut self, idx: usize) -> Result<(), WaveError> {
        let (p, _, ev) = self.points[idx].clone();
        // A diffracted readout takes the last step itself, onto the detector,
        // and so does an arm read out to be added to others.
        if matches!(ev, Ev::End) && (self.face || matches!(self.spec.readout, Readout::Diffracted { .. })) {
            self.merge_lobes();
            return Ok(());
        }
        // Free space up to this event, onto a window fitted to the beam there
        // when the one it has would no longer suit it; both components, and
        // every order carried apart, onto the same one.
        let z = (p - self.beam.origin).dot(self.beam.d);
        if z > 1e-12 {
            self.free_space(z, p)?;
        }
        let (beam_in, n_in) = (self.beam, self.u.n_medium);
        self.element_any(idx, &ev, p)?;
        if !self.rest.is_empty() {
            // The other orders through the same element, each about its own
            // reference ray; the report keeps the first one's step.
            let beam_out = self.beam;
            let n = self.report.steps.len();
            let mut lobes = std::mem::take(&mut self.rest);
            for l in &mut lobes {
                std::mem::swap(&mut self.u, &mut l.u);
                std::mem::swap(&mut self.uy, &mut l.uy);
                std::mem::swap(&mut self.at, &mut l.at);
                self.beam = beam_in;
                let r = self.element_any(idx, &ev, p);
                std::mem::swap(&mut self.u, &mut l.u);
                std::mem::swap(&mut self.uy, &mut l.uy);
                std::mem::swap(&mut self.at, &mut l.at);
                r?;
                self.report.steps.truncate(n);
            }
            self.rest = lobes;
            self.beam = beam_out;
        }
        if let Some(d) = self.departures.take() {
            self.report.approx(d.note());
        }
        self.paraxial(&ev, p, &beam_in, n_in)?;
        let transfer_ahead = self.points[idx + 1..].iter().any(|(_, _, e)| matches!(e, Ev::Transfer { .. }));
        if !transfer_ahead {
            self.merge_lobes();
        } else if let Ev::Hit(h) = &ev {
            self.split_orders(h, &beam_in)?;
        }
        if matches!(ev, Ev::End) {
            self.merge_lobes();
        }
        Ok(())
    }

    /// The orders carried apart added back into one field.
    fn merge_lobes(&mut self) {
        if self.rest.is_empty() {
            return;
        }
        self.at = [0.0; 4];
        for l in std::mem::take(&mut self.rest) {
            for (a, b) in self.u.data.iter_mut().zip(&l.u.data) {
                *a += *b;
            }
            if let (Some(uy), Some(ly)) = (&mut self.uy, &l.uy) {
                for (a, b) in uy.data.iter_mut().zip(&ly.data) {
                    *a += *b;
                }
            }
        }
    }
```

### 7.8 Plane-parallel plates: the closed-form slab, plane wave by plane wave

A planar interface keeps every plane wave's tangential wavevector, so in the
field's angular spectrum a plate is a multiplication.

The notation:
- medium 0 is outside before the plate, 1 the glass, 2 outside behind it
  (same index as 0);
- `r_ab`, `t_ab` are the interface amplitudes from medium `a` into `b`;
- `L` is the plate's thickness along its normal;
- `n_g`, `θ_g` are the glass's index and the plane wave's angle in it;
- `n₀`, `θ₀` are the outside medium's.

For each plane wave at `cos θ₀` outside (s or p):

```text
t_plate = t₀₁ t₁₂ e^{iδ} / (1 − x),   x = r₁₀ r₁₂ e^{2iδ},   δ = k₀ n_g L cos θ_g,   r₁₀ = −r₀₁
```

This is applied relative to free space across the plate's thickness,
`× e^{−ik₀n₀L cos θ₀}`. Then the exact angular spectrum carries each plane
wave from the plane before the plate to the plane behind it,
`exp(i2π(dist·√(1/λ² − f²) + f·s))` [Born & Wolf 1999], where:
- `dist` is the distance between the planes along the beam;
- `f` is the plane wave's transverse frequency;
- `s` is the chief ray's lateral shift between the planes, so the field
  lands on the exit plane's origin.

The round trips are in the sum, so an etalon's multiple-beam fringes and a
tilted plate's lateral shift both come out. A tilted plate's round trips are
displaced across the beam by `2L(n₀/n_g) sin θ₀ cos θ₀/cos θ_g`. Only
`m = ⌊0.4·window/shift⌋` of them are kept. The rest of the series, in power
`|t₀₁t₁₂|²|x|^{2(m+1)}/(1 − |x|²)`, is held as `PlateStats::beyond`. The
event then books it as `left_window`, so it is never wrapped back in.
`PlateStats::rest` is what the sum of the parts misses, booked as
`resampling`: the kept and dropped terms' interference, and round-off.

The rim of the plate is laid on the field with exact cell coverage. It is
found on a 65 × 65 probe grid and by bisection. A rim that is not
star-shaped about the chief ray is refused.

**Amplitudes, in the tangential chart and power coordinates** of a
scattering two-port, with the tilted admittances of [Macleod]:

```text
r = (y₁ − y₂)/(y₁ + y₂),   t = 2√(y₁y₂)/(y₁ + y₂),   y_s = n cos θ,   y_p = n / cos θ
```

This chart is unitary, `|r|² + |t|² = 1`. Its far side reflects `−r`, and at
normal incidence `r = (n₁ − n₂)/(n₁ + n₂)` for both polarizations. A residual
AR scales `r` by `√residual` and keeps the two-port unitary. Past the
critical angle, `|r| = 1` with its phase, and `t = 0`.

<!-- excerpt crates/oc-wave/src/planar.rs 78 112 -->
*Excerpt: `crates/oc-wave/src/planar.rs`, lines 78–112, commit `1af40e8`.*

```rust
/// `cos θ₂` from `cos θ₁` by Snell's law, on the decaying branch past the
/// critical angle (`exp(−iωt)`: `κ = +i|κ|`).
fn cos2(n1: f64, n2: f64, c1: f64) -> Cx {
    let s2 = (n1 / n2).powi(2) * (1.0 - c1 * c1).max(0.0);
    if s2 <= 1.0 {
        Cx::real((1.0 - s2).sqrt())
    } else {
        Cx::new(0.0, (s2 - 1.0).sqrt())
    }
}

/// A bare interface's two-port for a plane wave meeting it from index `n1`
/// at `cos θ₁ = c1`, in power coordinates: `(r, t)` for s or p. Past the
/// critical angle `|r| = 1` with its phase and `t = 0`.
pub fn interface(n1: f64, n2: f64, c1: f64, p: bool) -> (Cx, Cx) {
    let c1 = c1.abs().clamp(1e-12, 1.0);
    let c2 = cos2(n1, n2, c1);
    let (y1, y2) = if p { (Cx::real(n1 / c1), Cx::real(n2) / c2) } else { (Cx::real(n1 * c1), c2 * n2) };
    let r = (y1 - y2) / (y1 + y2);
    if c2.im != 0.0 {
        return (r, Cx::ZERO);
    }
    (r, Cx::real(2.0 * (y1.re * y2.re).sqrt() / (y1.re + y2.re)))
}

/// The same interface with its reflectance scaled by `residual` (a residual
/// AR), its transmission what the scaling leaves, keeping it unitary.
fn interface_ar(n1: f64, n2: f64, c1: f64, p: bool, residual: f64) -> (Cx, Cx) {
    let (r, t) = interface(n1, n2, c1, p);
    if t == Cx::ZERO {
        return (r, t);
    }
    let r = r * residual.clamp(0.0, 1.0).sqrt();
    (r, Cx::real((1.0 - r.norm_sqr()).max(0.0).sqrt()))
}
```

<!-- excerpt crates/oc-wave/src/planar.rs 150 250 -->
*Excerpt: `crates/oc-wave/src/planar.rs`, lines 150–250, commit `1af40e8`.*

```rust
/// The plate's transmission relative to free space for a plane wave at
/// `cos θ₀ = c0` outside, for s or p: `(full, kept, dropped)`, the whole
/// series, the series to `keep` round trips, and the power of the rest.
fn slab(plate: &Plate, k0: f64, c0: f64, p: bool, keep: Option<usize>) -> (Cx, Cx, f64) {
    let (n0, ng, l) = (plate.n_out, plate.n_glass, plate.thickness_mm);
    let (r01, t01) = interface_ar(n0, ng, c0, p, plate.residual[0]);
    let cg = cos2(n0, ng, c0).re;
    let (r12, t12) = interface_ar(ng, n0, cg, p, plate.residual[1]);
    // Face 1 from inside reflects −r₀₁ (unitary two-port).
    let r10 = -r01;
    let delta = k0 * ng * l * cg;
    let x = r10 * r12 * Cx::expi(2.0 * delta);
    let lead = t01 * t12 * Cx::expi(delta);
    let full = lead / (Cx::ONE - x);
    let (kept, dropped) = match keep {
        None => (full, 0.0),
        Some(m) => {
            let mut xm = Cx::ONE;
            for _ in 0..=m {
                xm *= x;
            }
            let ax = x.norm_sqr();
            (lead * (Cx::ONE - xm) / (Cx::ONE - x), lead.norm_sqr() * xm.norm_sqr() / (1.0 - ax).max(1e-300))
        }
    };
    let free = Cx::expi(-k0 * n0 * l * c0);
    (full * free, kept * free, dropped)
}

/// Carry `u`, across `inc` on a plane before the plate, through it to the
/// plane across `out` behind it, the plate's effect on each plane wave the
/// closed-form slab relative to free space, the free space between the two
/// planes the exact angular spectrum, and the field shifted onto `out`'s
/// origin (the chief ray's exit). Padded twice.
pub(crate) fn plate_step(u: &Field2, inc: &Beam, out: &Beam, plate: &Plate, pol: Pol, fft: &mut Fft2) -> Result<(Field2, PlateStats), WaveError> {
    let n = u.grid.n;
    let p = [next_fast_len(2 * n[0]), next_fast_len(2 * n[1])];
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let mut buf = embed(u, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    let lam0 = u.wavelength_mm();
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let k0 = 2.0 * PI / lam0;
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let df = du * dv;
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let dist = (out.origin - inc.origin).dot(inc.d);
    let lateral = out.origin - (inc.origin + inc.d * dist);
    let (sx, sy) = (lateral.dot(inc.x), lateral.dot(inc.y));
    // One round trip's shift across the beam, for the chief direction.
    let c_chief = inc.d.dot(plate.normal).abs().clamp(1e-12, 1.0);
    let cg_chief = cos2(plate.n_out, plate.n_glass, c_chief).re.max(1e-12);
    let sin0 = (1.0 - c_chief * c_chief).max(0.0).sqrt();
    let shift = 2.0 * plate.thickness_mm * (plate.n_out / plate.n_glass) * sin0 / cg_chief * c_chief;
    let window = u.grid.width_mm(0).min(u.grid.width_mm(1));
    let keep = (shift > 1e-9 * window).then(|| (0.4 * window / shift).floor() as usize);
    let mut st = PlateStats { round_trips: keep, round_trip_shift_mm: shift, received: u.power(), ..Default::default() };
    for ky in 0..p[1] {
        let fy = gp.fft_freq(1, ky) + v0;
        for kx in 0..p[0] {
            let fx = gp.fft_freq(0, kx) + u0;
            let a = &mut buf[ky * p[0] + kx];
            let w = inv_l2 - fx * fx - fy * fy;
            let e = a.norm_sqr() * df;
            if w <= 0.0 {
                st.evanescent += e;
                *a = Cx::ZERO;
                continue;
            }
            let dir = inc.x * (lam * fx) + inc.y * (lam * fy) + inc.d * (lam * w.sqrt());
            let c0 = dir.dot(plate.normal).abs();
            let (full, kept, dropped) = match pol.is_p() {
                Some(pp) => slab(plate, k0, c0, pp, keep),
                None => {
                    // Square to the beam s and p are one; off it an
                    // unpolarized field takes their mean (reported).
                    let (fs, ks, ds) = slab(plate, k0, c0, false, keep);
                    let (fp, kp, dp) = slab(plate, k0, c0, true, keep);
                    ((fs + fp) * 0.5, (ks + kp) * 0.5, 0.5 * (ds + dp))
                }
            };
            st.reflected += (1.0 - full.norm_sqr()).max(0.0) * e;
            st.beyond += dropped * e;
            *a *= kept * Cx::expi(2.0 * PI * (dist * w.sqrt() + fx * sx + fy * sy));
        }
    }
    fft.inverse(&mut buf, p, df);
    fftshift2(&mut buf, p);
    let whole: f64 = buf.iter().map(|z| z.norm_sqr()).sum::<f64>() * gp.cell_area_mm2();
    let data = crop(&buf, p, n);
    let mut v = u.clone();
    v.data = data;
    v.frame = out.iso();
    v.opl_ref_mm = u.opl_ref_mm + plate.chief_opl_mm;
    let kept_power = v.power();
    st.left_window = (whole - kept_power).max(0.0);
    st.rest = st.received - st.reflected - st.beyond - st.left_window - st.evanescent - kept_power;
    Ok((v, st))
}
```

The plate is recognized only when every condition holds:
- both faces are flat, which probe rays on 8 directions confirm at 0.3, 0.6,
  0.9 and 0.999 of the face's reach;
- they are parallel to 1e-12;
- the direction is unchanged and the medium is the same on both sides;
- both faces are `DielectricFresnel` (the bare Fresnel interface), each with
  an `ar_residual`: the fraction of the bare reflectance it keeps, 1 for
  bare glass and 0 for a perfect AR.

A body that is not such a plate is traced face by face (§7.5).

<!-- excerpt crates/oc-wave/src/path.rs 699 795 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 699–795, commit `1af40e8`.*

```rust
/// A plane-parallel plate where the chain enters a body by `h` and leaves it
/// by `h2`: both faces flat (every probe ray meets them along one normal),
/// parallel, bare or with a residual AR, the medium the same on both sides.
/// Returns the plate and the planes before and after it, or `None` for
/// anything else, which is then traced face by face.
fn plate_of(
    scene: &Scene3,
    h: &oc_core::seq3::SeqHit3,
    h2: &oc_core::seq3::SeqHit3,
    resume: Vec3,
    next: Vec3,
    half: f64,
    lam: f64,
) -> Option<(Plate, Vec3, Vec3)> {
    use oc_trace::{trace3, LaunchRay3, TraceConfig3, TraceEventKind};
    if h2.object != h.object || h2.kind != SeqHitKind::Body || h.action != SeqAction::Transmit || h2.action != SeqAction::Transmit {
        return None;
    }
    let n = h.normal.normalize();
    if n.dot(h2.normal.normalize()).abs() < 1.0 - 1e-12 {
        return None;
    }
    let d = (h.point - resume).normalize();
    let d_out = (next - h2.point).normalize();
    if d.dot(d_out) < 1.0 - 1e-12 || (h.n_before - h2.n_after).abs() > 1e-12 || (h.n_after - h2.n_before).abs() > 1e-12 {
        return None;
    }
    let body = scene.bodies.iter().find(|b| b.common.id == h.object)?;
    let residual = |r: &SurfaceResponse| match r {
        SurfaceResponse::DielectricFresnel { ar_residual } => ar_residual.sample(lam),
        _ => None,
    };
    let r1 = residual(&body.surface.for_patch(h.patch).from_outside)?;
    let r2 = residual(&body.surface.for_patch(h2.patch).from_inside)?;
    // The planes before the first face and behind the second, over the window.
    let x = if d.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let x = (x - d * d.dot(x)).normalize();
    let y = d.cross(x);
    let corners = [(-half, -half), (half, -half), (-half, half), (half, half)];
    let depth = |a: f64, b: f64| -(x * a + y * b).dot(n) / d.dot(n);
    let s_min = corners.iter().map(|&(a, b)| depth(a, b)).fold(0.0f64, f64::min);
    let s_max = corners.iter().map(|&(a, b)| depth(a, b)).fold(0.0f64, f64::max);
    let entry = h.point + d * (s_min - 1e-3);
    let exit = h2.point + d * (s_max + 1e-3);
    if (entry - resume).dot(d) <= 0.0 || (next - exit).dot(d) <= 0.0 {
        return None;
    }
    // Flat: along each of eight directions from the chief ray, out to where
    // the faces end (found by bisection, so a small element is not missed),
    // every ray meets the two faces square to n.
    let probe = probe_scene(scene, h.object);
    let cfg = TraceConfig3 { ghosts: false, polarization: false, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(&probe), ..TraceConfig3::default() };
    // The normals at the two faces where the ray from (a, b) meets them.
    let faces = |a: f64, b: f64| -> Option<(Vec3, Vec3)> {
        let ray = oc_core::vec3::Ray3::new(entry + x * a + y * b, d).ok()?;
        let res = trace3(&probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
        let hits: Vec<(oc_core::id::PatchId, Vec3)> = res
            .events
            .iter()
            .filter_map(|e| match &e.kind {
                TraceEventKind::Interact { surface, normal, .. } => Some((surface.patch, *normal)),
                _ => None,
            })
            .take(2)
            .collect();
        (hits.len() == 2 && hits[0].0 == h.patch && hits[1].0 == h2.patch).then(|| (hits[0].1, hits[1].1))
    };
    let r_max = half * std::f64::consts::SQRT_2;
    for k in 0..8 {
        let t = std::f64::consts::PI * k as f64 / 4.0;
        let (c, sn) = (t.cos(), t.sin());
        let (mut lo, mut hi) = (0.0, r_max);
        if faces(hi * c, hi * sn).is_none() {
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if faces(mid * c, mid * sn).is_some() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        } else {
            lo = hi;
        }
        for f in [0.3, 0.6, 0.9, 0.999] {
            let r = f * lo;
            let (n1, n2) = faces(r * c, r * sn)?;
            if n1.normalize().dot(n).abs() < 1.0 - 1e-9 || n2.normalize().dot(n).abs() < 1.0 - 1e-9 {
                return None;
            }
        }
    }
    let thickness_mm = (h2.point - h.point).dot(n).abs();
    let chief_opl_mm = h.n_before * (h.point - entry).length() + h.n_after * (h2.point - h.point).length() + h2.n_after * (exit - h2.point).length();
    let plate = Plate { normal: n, thickness_mm, n_glass: h.n_after, n_out: h.n_before, residual: [r1, r2], chief_opl_mm };
    Some((plate, entry, exit))
}
```

<!-- excerpt crates/oc-wave/src/path.rs 1165 1250 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1165–1250, commit `1af40e8`.*

```rust
    /// A plane-parallel plate: its rim laid on the field at the field's own
    /// sampling, then the slab's closed form plane wave by plane wave.
    fn plate_event(&mut self, first: &oc_core::seq3::SeqHit3, second: &oc_core::seq3::SeqHit3, exit: Vec3, plate: &Plate) -> Result<(), WaveError> {
        use oc_trace::{trace3, LaunchRay3, TraceConfig3, TraceEventKind};
        let beam = self.beam;
        let lam = self.lam;
        let before = self.u.power();
        // The rim: where a ray from the plane before the plate meets the two
        // faces in turn.
        let probe = probe_scene(&self.scene, first.object);
        let cfg = TraceConfig3 { ghosts: false, polarization: false, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(&probe), ..TraceConfig3::default() };
        let pass = |a: f64, b: f64| -> bool {
            let Ok(ray) = oc_core::vec3::Ray3::new(beam.point(a, b), beam.d) else { return false };
            let res = trace3(&probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
            let hits: Vec<_> = res
                .events
                .iter()
                .filter_map(|e| match &e.kind {
                    TraceEventKind::Interact { surface, .. } => Some(surface.patch),
                    _ => None,
                })
                .take(2)
                .collect();
            hits == [first.patch, second.patch]
        };
        let g = 65usize;
        let w = self.u.grid.width_mm(0).max(self.u.grid.width_mm(1));
        let h = w / (g - 1) as f64;
        let half = (g - 1) as f64 / 2.0;
        let grid: Vec<bool> = (0..g * g).map(|k| pass(((k % g) as f64 - half) * h, ((k / g) as f64 - half) * h)).collect();
        let rim = if grid.iter().all(|&p| p) {
            "no rim within the window".to_string()
        } else {
            let poly = crate::transfer::rim_polygon(&pass, &grid, g, h)
                .ok_or_else(|| WaveError::OutOfRange(format!("plate #{} ({}): the edge of what passes it is not star-shaped about the chief ray", first.object.0, first.label)))?;
            let cov = crate::elements::polygon_coverage(&self.u.grid, &poly);
            let (stopped, beyond) = crate::elements::apply_coverage(&mut self.u, &cov)?;
            self.report.ledger.absorbed_aperture += stopped;
            self.report.ledger.beyond_nyquist += beyond;
            format!("its rim laid on the field with exact cell coverage, {:.3e} of the power stopped there", stopped / before.max(f64::MIN_POSITIVE))
        };
        let out = beam.turned(exit, beam.d);
        let (v, st) = crate::planar::plate_step(&self.u, &beam, &out, plate, self.pol, &mut self.fft)?;
        let l = &mut self.report.ledger;
        l.other_arm += st.reflected;
        l.left_window += st.beyond + st.left_window;
        l.evanescent += st.evanescent;
        l.resampling += st.rest;
        let oblique = beam.d.dot(plate.normal).abs() < 1.0 - 1e-12;
        let pol = match self.pol {
            Pol::S(_) => "s",
            Pol::P(_) => "p",
            Pol::Unpolarized | Pol::Projected => "s = p",
        };
        let trips = match st.round_trips {
            None => "every round trip".to_string(),
            Some(m) => format!(
                "{m} round trip(s), each {:.4} mm across the beam; the rest, beyond the window, {:.3e} of the power",
                st.round_trip_shift_mm,
                st.beyond / before.max(f64::MIN_POSITIVE)
            ),
        };
        self.report.steps.push(element_step(
            0.0,
            format!(
                "plate #{} ({}): {:.6} mm of index {:.6}, the slab's closed form plane wave by plane wave ({pol}), {trips}; {rim}; {:.3e} reflected",
                first.object.0,
                first.label,
                plate.thickness_mm,
                plate.n_glass,
                st.reflected / before.max(f64::MIN_POSITIVE)
            ),
        ));
        if oblique && self.pol == Pol::Unpolarized {
            self.report.approx(format!("plate #{}: met obliquely by a field standing for unpolarized light, its s and p amplitudes averaged", first.object.0));
        }
        if oblique {
            self.report.approx(
                "planar layers: every plane wave takes its own angle's amplitude for the chain's s or p; plane waves off the plane of incidence see s and p slightly rotated, which is left out",
            );
        }
        self.u = v;
        self.beam = out;
        self.u.n_medium = plate.n_out;
        Ok(())
    }
```

### 7.9 Flat coated surfaces

A `Coating` (a stack of thin films, evaluated by `oc_core::thinfilm`'s
transfer matrix) on a flat standalone surface multiplies each plane wave of
the field by the stack's complex `r` (reflected arm) or `t` (transmitted
arm). The values are taken at that plane wave's own angle, from a 1025-entry
table in `cos θ` interpolated linearly.

`coating_amp_at` returns a `FilmAmp`. It holds `r_s`, `r_p`, `t_s`, `t_p` in
the ray frame:
- `ŝ = d̂ × n̂`;
- `p̂ = d̂ × ŝ`, following each ray;
- `t_p` rescaled from the tangential field to the whole field;
- `t_power_scale = Re(n_exit cos θ_exit)/Re(n₀ cos θ₀)`, so that `|t|²` times
  it is the transmittance.

These are put into the chart above:
- `r_p` is negated (the ray frame's `p̂` follows the reflected ray, so its
  `r_p = −r_s` at normal incidence);
- `t` is scaled by `√t_power_scale` into power coordinates.

`CoatingBacking` says what lies behind the stack:
- the same medium;
- an opaque substrate;
- a plate of a substrate, with an exit face.

`Incidence::Film` means light meets the stack first; `Substrate` means it
meets the glass first.

A coating on a plate (`CoatingBacking::Plate`) is coherent in its stack but
incoherent between the plate's two faces, as the scene declares it. The
transmitted amplitude is the stack's into the glass times the bare back
face's out of it, and the light the faces send round the plate is booked, not
added.

<!-- excerpt crates/oc-wave/src/planar.rs 252 368 -->
*Excerpt: `crates/oc-wave/src/planar.rs`, lines 252–368, commit `1af40e8`.*

```rust
/// Multiply each plane wave of `u` (across `inc`) by what an element in the
/// plane with normal `n_e` does to it: `f(cos θ)` gives the amplitude on the
/// chain's arm and the shares of the power that go into the other arm and
/// into the element. Returns the field and those two powers, and what left
/// the window. Padded twice; nothing propagates.
pub(crate) fn surface_filter(
    u: &Field2,
    inc: &Beam,
    n_e: Vec3,
    f: &dyn Fn(f64) -> (Cx, f64, f64),
    fft: &mut Fft2,
) -> Result<(Field2, f64, f64, f64), WaveError> {
    let n = u.grid.n;
    let p = [next_fast_len(2 * n[0]), next_fast_len(2 * n[1])];
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let mut buf = embed(u, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let df = du * dv;
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let (mut other, mut absorbed) = (0.0, 0.0);
    for ky in 0..p[1] {
        let fy = gp.fft_freq(1, ky) + v0;
        for kx in 0..p[0] {
            let fx = gp.fft_freq(0, kx) + u0;
            let a = &mut buf[ky * p[0] + kx];
            let w = (inv_l2 - fx * fx - fy * fy).max(0.0);
            let dir = inc.x * (lam * fx) + inc.y * (lam * fy) + inc.d * (lam * w.sqrt());
            let (c, o, ab) = f(dir.dot(n_e).abs());
            let e = a.norm_sqr() * df;
            other += o * e;
            absorbed += ab * e;
            *a *= c;
        }
    }
    fft.inverse(&mut buf, p, df);
    fftshift2(&mut buf, p);
    let whole: f64 = buf.iter().map(|z| z.norm_sqr()).sum::<f64>() * gp.cell_area_mm2();
    let mut v = u.clone();
    v.data = crop(&buf, p, n);
    let left = (whole - v.power()).max(0.0);
    Ok((v, other, absorbed, left))
}

/// A flat coating's amplitude on one arm, for a plane wave meeting it from
/// index `n0` at `cos θ₀ = c0`, in the chart above: the amplitude, the share
/// of the power that goes into the other arm (and, for a plate-backed
/// coating, into the plate's own round trips), and the share absorbed.
///
/// A coating on a plate (`CoatingBacking::Plate`) is, as the scene declares
/// it, a plate far thicker than any coherence length the canvas models: its
/// stack is coherent, its two faces are not, so the transmitted amplitude is
/// the stack's into the glass times the back face's out of it, and the light
/// the faces send round the plate is booked, not added.
pub fn coating_arm(c: &Coating, reflect: bool, n0: f64, lambda_um: f64, c0: f64, p: bool) -> Result<(Cx, f64, f64), WaveError> {
    let fe = |e: oc_core::thinfilm::FilmError| WaveError::OutOfRange(format!("the coating cannot be evaluated: {e:?}"));
    let chart = |a: &oc_core::thinfilm::FilmAmp| -> (Cx, Cx) {
        let s = a.t_power_scale.max(0.0).sqrt();
        if p {
            (-a.r_p, a.t_p * s)
        } else {
            (a.r_s, a.t_s * s)
        }
    };
    match &c.backing {
        CoatingBacking::Medium | CoatingBacking::Opaque { .. } => {
            let a = coating_amp_at(c, n0, n0, lambda_um, c0).map_err(fe)?;
            let (r, t) = chart(&a);
            let (rr, tt) = (r.norm_sqr(), t.norm_sqr());
            let lost = (1.0 - rr - tt).max(0.0);
            Ok(if reflect { (r, tt, lost) } else { (t, rr, lost) })
        }
        CoatingBacking::Plate { substrate, exit, .. } => {
            let ns = substrate.index_at(lambda_um).map(|z| z.re).unwrap_or(1.5).max(1e-9);
            let film = Coating { backing: CoatingBacking::Medium, ..c.clone() };
            let residual = match exit {
                ExitFace::Bare => 1.0,
                ExitFace::AntiReflective { residual } => *residual,
            };
            match c.incidence {
                Incidence::Film => {
                    // The stack from outside into the glass, then the back face.
                    let a = coating_amp_at(&film, n0, ns, lambda_um, c0).map_err(fe)?;
                    let (rf, tf) = chart(&a);
                    let cs = cos2(n0, ns, c0).re.max(1e-12);
                    let (rb, tb) = interface_ar(ns, n0, cs, p, residual);
                    let (rr, tt) = (rf.norm_sqr(), tf.norm_sqr());
                    let lost = (1.0 - rr - tt).max(0.0);
                    if reflect {
                        Ok((rf, tt, lost))
                    } else {
                        Ok((tf * tb, rr + tt * rb.norm_sqr(), lost))
                    }
                }
                Incidence::Substrate => {
                    // The bare back face into the glass, then the stack from the
                    // glass side, then (reflected) the back face again.
                    let (rb, tb) = interface_ar(n0, ns, c0, p, residual);
                    let cs = cos2(n0, ns, c0).re.max(1e-12);
                    let a = coating_amp_at(&film, ns, n0, lambda_um, cs).map_err(fe)?;
                    let (rf, tf) = chart(&a);
                    let (rr, tt) = (rf.norm_sqr(), tf.norm_sqr());
                    let lost = (1.0 - rr - tt).max(0.0) * tb.norm_sqr();
                    let tbb = tb.norm_sqr();
                    if reflect {
                        Ok((tb * rf * tb, rb.norm_sqr() + tbb * tt + tbb * rr * (1.0 - tbb), lost))
                    } else {
                        Ok((tb * tf, rb.norm_sqr() + tbb * rr, lost))
                    }
                }
            }
        }
    }
}
```

<!-- excerpt crates/oc-wave/src/path.rs 1712 1746 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1712–1746, commit `1af40e8`.*

```rust
            (SurfaceResponse::Coating { coating }, SeqAction::Reflect | SeqAction::Transmit) => {
                let reflect = h.action == SeqAction::Reflect;
                let n0 = h.n_before;
                let pol = self.pol;
                // The amplitude on a table of cosines, read between its
                // entries: it is smooth in the angle, and a transfer matrix
                // per plane wave would cost the field's size in stacks.
                let table: Vec<(Cx, f64, f64)> = (0..=1024)
                    .map(|i| {
                        let c = (i as f64 / 1024.0).max(1e-6);
                        let at = |pp: bool| crate::planar::coating_arm(coating, reflect, n0, lam, c, pp);
                        match pol.is_p() {
                            Some(pp) => at(pp),
                            None => at(false).and_then(|a| at(true).map(|b| ((a.0 + b.0) * 0.5, 0.5 * (a.1 + b.1), 0.5 * (a.2 + b.2)))),
                        }
                    })
                    .collect::<Result<_, _>>()?;
                let read = |c: f64| -> (Cx, f64, f64) {
                    let x = (c.clamp(0.0, 1.0) * 1024.0).min(1023.999999);
                    let (i, f) = (x.floor() as usize, x - x.floor());
                    let (a, b) = (table[i], table[i + 1]);
                    (a.0 * (1.0 - f) + b.0 * f, a.1 * (1.0 - f) + b.1 * f, a.2 * (1.0 - f) + b.2 * f)
                };
                let (v, other, absorbed, left) = crate::planar::surface_filter(&self.u, &beam, n_s, &read, &mut self.fft)?;
                let l = &mut self.report.ledger;
                l.other_arm += other;
                l.absorbed_element += absorbed;
                l.left_window += left;
                self.u = v;
                if beam.d.dot(n_s).abs() < 1.0 - 1e-12 && self.pol == Pol::Unpolarized {
                    self.report.approx(format!("coating #{}: met obliquely by a field standing for unpolarized light, its s and p amplitudes averaged", h.object.0));
                }
                self.report.approx("coatings on the path: each plane wave takes its own angle's amplitude for the chain's s or p, from the transfer matrix; a plate-backed coating's plate is incoherent, as it declares");
                (1.0, None, Loss::Absorbed)
            }
```

### 7.10 Polarization: two components per sample

**When two components are carried.** A chain carries two fields, the
components along its beam frame's `x` and `y` (a Jones vector at every sample
[Jones 1941]), when any chain meets one of these:
- a polarizer, a retarder or an LCoS;
- a flat coating met obliquely;
- a plate met obliquely.

**Unpolarized light** is then two mutually incoherent launches, along `x` and
along `y`, with half the power each, added in power [Wolf 1982]. The sum is
the same in any basis. Otherwise one field stands for unpolarized light.

**A stale comment.** The first paragraph of the doc comment below
("…s and p of the plane of incidence those elements share … Elements whose
planes of incidence differ mix s and p … refused by name") describes an
earlier design. The code that runs is the second paragraph's function. It
carries `x` and `y` components and turns them into each element's s and p,
and elements in different planes of incidence are supported (the test
`coatings_whose_planes_of_incidence_differ_reflect_s_of_one_as_p_of_the_other`).

<!-- excerpt crates/oc-wave/src/path.rs 3070 3106 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 3070–3106, commit `1af40e8`.*

```rust
/// The polarization modes a set of chains needs: one standing for
/// unpolarized light, unless an element on a chain tells s from p obliquely
/// (a plane-parallel plate, a flat coating), when an unpolarized source is
/// two mutually incoherent modes, s and p of the plane of incidence those
/// elements share, half its power each. Elements whose planes of incidence
/// differ mix s and p, which a scalar field cannot carry: refused by name.
/// Whether the chains' light must be carried as two polarization components:
/// when one of them meets a polarizer, a retarder or an LCoS, or a flat
/// coating or a plate obliquely, where s and p differ.
fn polarization_matters(scene: &Scene3, seqs: &[oc_core::seq3::SequencePrescription3], half: f64, lam: f64) -> bool {
    seqs.iter().any(|seq| {
        let mut prev = seq.launch.point;
        seq.hits.iter().enumerate().any(|(k, h)| {
            let d = (h.point - prev).normalize();
            prev = h.point;
            let n = h.normal.normalize();
            let oblique = d.dot(n).abs() < 1.0 - 1e-9;
            let surface = scene.surfaces.iter().find(|s| s.common.id == h.object);
            let response = surface.map(|s| if d.dot(s.common.transform.rotate(Vec3::Z)) < 0.0 { &s.response_front } else { &s.response_back });
            let telling = h.kind == SeqHitKind::Surface
                && response.is_some_and(|r| match r {
                    SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. } => true,
                    SurfaceResponse::Modulator { device, .. } => matches!(device.kind, ModulatorKind::Lcos(_)),
                    SurfaceResponse::Coating { .. } => oblique && surface.is_some_and(|s| matches!(s.geometry, SurfaceGeometry3::Plane { .. })),
                    _ => false,
                });
            let plate = h.kind == SeqHitKind::Body
                && oblique
                && seq.hits.get(k + 1).is_some_and(|h2| {
                    let next = seq.hits.get(k + 2).map_or(seq.terminal.point, |x| x.point);
                    let from = if k == 0 { seq.launch.point } else { seq.hits[k - 1].point };
                    plate_of(scene, h, h2, from, next, half, lam).is_some()
                });
            telling || plate
        })
    })
}
```

**At each element:**
1. The pair is turned into the element's own s and p: `ŝ` normal to the
   plane of incidence, `p̂ = ŝ × d` on either side, with `p̂`'s part along
   the surface kept across the element.
2. Each component is carried through the element as a field of that
   polarization alone.
3. The pair is turned into the outgoing beam's frame.

Square to the beam, the element's s and p are the beam's own axes. Every
plane wave of a planar step takes its own angle's amplitude for the chain's
s or p. That plane waves off the plane of incidence see s and p slightly
rotated is left out and named.

How the element functions call each other:
- `element_any` dispatches. A field of two components goes to
  `element_polarized`; an aperture runs twice, once per component; anything
  else goes to `element`.
- `element_polarized` turns the pair into the element's basis, runs `element`
  once per component with `self.pol` set to `Pol::S` or `Pol::P`, and turns
  the pair back.

The other names:
- `Pol::Unpolarized` is a one-field chain, or a square element where s = p.
- `Pol::Projected` is the one component an LCoS acts on, already projected
  out. It has nothing to do with `Readout::Projected`.
- `mix(u, v, a, b)` is `a·u + b·v` sample by sample (`mix_c` with complex
  weights).
- `layer` applies a polarizer's or a retarder's Jones matrix.

<!-- excerpt crates/oc-wave/src/path.rs 568 590 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 568–590, commit `1af40e8`.*

```rust
/// The polarization basis an element works in, met along `d_in` with normal
/// `n` and left along `d_out`: s and p of its plane of incidence, `ŝ` normal
/// to the plane and `p̂ = ŝ × d` in it, on either side; square to the beam,
/// the beam's own axes, whose field the element leaves as it is. Across the
/// element the s and p components keep their direction (`p̂` turns with the
/// beam, its part along the surface kept, as the tangential chart has it),
/// and an element's scalar amplitudes, which carry its signs, apply to each.
fn element_basis(inc: &Beam, n: Vec3, d_out: Vec3) -> ([Vec3; 2], [Vec3; 2]) {
    let n = n.normalize();
    let sn = inc.d.cross(n);
    if sn.length() < 1e-9 {
        return ([inc.x, inc.y], [inc.x, inc.y]);
    }
    let s_in = sn.normalize();
    let p_in = s_in.cross(inc.d);
    let s_out = (s_in - d_out * d_out.dot(s_in)).normalize();
    let mut p_out = s_out.cross(d_out);
    let along = |v: Vec3| v - n * n.dot(v);
    if along(p_out).dot(along(p_in)) < 0.0 {
        p_out = -p_out;
    }
    ([s_in, p_in], [s_out, p_out])
}
```

<!-- excerpt crates/oc-wave/src/path.rs 1438 1491 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1438–1491, commit `1af40e8`.*

```rust
        let (bin, bout) = element_basis(&beam0, n_e, d_out);
        let comps = [mix(&ux, &uy, beam0.x.dot(bin[0]), beam0.y.dot(bin[0])), mix(&ux, &uy, beam0.x.dot(bin[1]), beam0.y.dot(bin[1]))];
        let oblique = bin[0] != beam0.x || bin[1] != beam0.y;
        let pols = if oblique { [Pol::S(bin[0]), Pol::P(bin[0])] } else { [Pol::Unpolarized, Pol::Unpolarized] };
        // A component without power is carried as zeros, on the other's
        // outgoing window: an element fitted to a field has nothing to fit.
        let lit: Vec<bool> = comps.iter().map(|c| c.power() > 0.0).collect();
        let mut outs: [Option<Field2>; 2] = [None, None];
        let mut beam1 = beam0;
        let mut ran = false;
        for k in 0..2 {
            if !lit[k] && (ran || lit[1 - k]) {
                continue;
            }
            self.u = comps[k].clone();
            self.pol = pols[k];
            self.beam = beam0;
            let n = self.report.steps.len();
            self.element(idx, ev.clone(), p)?;
            if ran {
                self.report.steps.truncate(n);
            }
            ran = true;
            beam1 = self.beam;
            outs[k] = Some(self.u.clone());
        }
        self.pol = Pol::Unpolarized;
        let zeros = |f: &Field2| {
            let mut z = f.clone();
            z.data.iter_mut().for_each(|v| *v = Cx::ZERO);
            z
        };
        let [o1, o2] = outs;
        let (o1, o2) = match (o1, o2) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => {
                let b = zeros(&a);
                (a, b)
            }
            (None, Some(b)) => (zeros(&b), b),
            (None, None) => unreachable!("one component is always run"),
        };
        let out = beam1;
        let mut fx = mix(&o1, &o2, bout[0].dot(out.x), bout[1].dot(out.x));
        let mut fy = mix(&o1, &o2, bout[0].dot(out.y), bout[1].dot(out.y));
        if let Some(si) = layer {
            let (gx, gy) = self.layer(si, &out, &fx, &fy)?;
            fx = gx;
            fy = gy;
        }
        self.u = fx;
        self.uy = Some(fy);
        Ok(())
    }
```

**A polarizer or a retarder** applies **one Jones matrix for the whole
field**, the one the ray tracer applies to the chief ray:
1. the element's front face (Fresnel, diagonal in s and p);
2. its layer on its own axis (local `+X` turned `axis_deg` towards `+Y`),
   projected across the direction inside its glass, so that crossed
   polarizers leak off axis;
3. its back face.

Further details:
- A polarizer passes amplitude 1 along its axis and `√ε(θ)` across it,
  where `ε` is the extinction at the chief ray's angle.
- A retarder has the declared retardance, or, for a crystal, the A-plate
  retardance at the chief ray's angle. An A-plate has its optic axis in the
  plate's plane, and its retardance at oblique incidence comes from
  `oc_core::polarization::a_plate_retardance`.
- The faces' amplitudes come from `interface_amp`, the bare Fresnel
  interface in the ray frame.
- The **front** face's reflection is booked as `other_arm`. Everything else
  the element takes goes to `absorbed_element`: the back face's reflection,
  the polarizer's extinction and any substrate loss.
- There is no bounce between the faces, and no walk-off in a crystal. Both
  are named.

<!-- excerpt crates/oc-wave/src/path.rs 1501 1592 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1501–1592, commit `1af40e8`.*

```rust
    fn layer(&mut self, si: usize, out: &Beam, fx: &Field2, fy: &Field2) -> Result<(Field2, Field2), WaveError> {
        let s = &self.scene.surfaces[si];
        let t = s.common.transform;
        let nrm = t.rotate(Vec3::Z);
        let d = out.d;
        let response = if d.dot(nrm) < 0.0 { &s.response_front } else { &s.response_back };
        let (axis_deg, n_sub) = match response {
            SurfaceResponse::Polarizer { axis_deg, substrate_index, .. } | SurfaceResponse::Retarder { axis_deg, substrate_index, .. } => (*axis_deg, substrate_index.max(1.0)),
            _ => unreachable!("only polarizers and retarders are layers"),
        };
        let id = s.common.id.0;
        let n_here = fx.n_medium;
        let n_opp = if d.dot(nrm) < 0.0 { nrm } else { -nrm };
        let cos_i = (-n_opp.dot(d)).clamp(0.0, 1.0);
        let s_hat = oc_core::polarization::incidence_axis(d, n_opp).unwrap_or(out.x);
        let p_hat = s_hat.cross(d);
        // Into the element's glass by Snell's law.
        let eta = n_here / n_sub;
        let kk = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
        if kk < 0.0 {
            return Err(WaveError::OutOfRange(format!("#{id}: the field is past the critical angle of the element's own glass and never reaches its layer")));
        }
        let cos_t = kk.sqrt();
        let d_in = (d * eta + n_opp * (eta * cos_i - cos_t)).normalize();
        let p_glass = s_hat.cross(d_in);
        let front = oc_core::thinfilm::interface_amp(n_here, n_sub, cos_i);
        let back = oc_core::thinfilm::interface_amp(n_sub, n_here, cos_t);
        let (sa, ca) = axis_deg.to_radians().sin_cos();
        let axis = t.rotate(Vec3::X) * ca + t.rotate(Vec3::Y) * sa;
        let projected = axis - d_in * d_in.dot(axis);
        if projected.length() < 1e-9 {
            return Err(WaveError::OutOfRange(format!("#{id}: the beam runs along the element's own axis, which has no direction across it")));
        }
        let e1 = projected.normalize();
        let e2 = d_in.cross(e1);
        let (a, b, what) = match response {
            SurfaceResponse::Polarizer { extinction, .. } => {
                let aoi = cos_i.acos().to_degrees();
                let leak = extinction
                    .sample(aoi)
                    .ok_or_else(|| WaveError::OutOfRange(format!("polarizer #{id}: its extinction curve has no value at {aoi:.3}°")))?;
                (Cx::ONE, Cx::real(leak.clamp(0.0, 1.0).sqrt()), format!("polarizer, axis {axis_deg}°, extinction {leak:.3e}"))
            }
            SurfaceResponse::Retarder { retardance_deg, crystal, .. } => {
                let rad = match crystal {
                    None => retardance_deg.to_radians(),
                    Some(c) => {
                        // As the tracer has it: the optic axis is the slow one
                        // in a positive crystal, the fast one in a negative.
                        let optic = if c.n_e >= c.n_o { axis_deg + 90.0 } else { axis_deg };
                        let (so, co) = optic.to_radians().sin_cos();
                        let c_hat = t.rotate(Vec3::X) * co + t.rotate(Vec3::Y) * so;
                        let lean = d - n_opp * d.dot(n_opp);
                        let cos_axis = if lean.length() > 1e-12 { lean.normalize().dot(c_hat) } else { 0.0 };
                        let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
                        oc_core::polarization::a_plate_retardance(c.n_o, c.n_e, c.thickness_um(*retardance_deg), self.lam, n_here * sin_i, cos_axis).abs()
                    }
                };
                (Cx::ONE, Cx::expi(rad), format!("retarder, fast axis {axis_deg}°, retardance {:.4}°", rad.to_degrees()))
            }
            _ => unreachable!("only polarizers and retarders are layers"),
        };
        // The layer in the glass's (ŝ, p̂) basis, between the faces'.
        let c1 = [e1.dot(s_hat), e1.dot(p_glass)];
        let c2 = [e2.dot(s_hat), e2.dot(p_glass)];
        let lay = |i: usize, j: usize| a * (c1[i] * c1[j]) + b * (c2[i] * c2[j]);
        let tf = [front.t_s * front.t_power_scale.max(0.0).sqrt(), front.t_p * front.t_power_scale.max(0.0).sqrt()];
        let tb = [back.t_s * back.t_power_scale.max(0.0).sqrt(), back.t_p * back.t_power_scale.max(0.0).sqrt()];
        let j = |i: usize, k: usize| tb[i] * lay(i, k) * tf[k];
        let cs = mix(fx, fy, out.x.dot(s_hat), out.y.dot(s_hat));
        let cp = mix(fx, fy, out.x.dot(p_hat), out.y.dot(p_hat));
        let before = cs.power() + cp.power();
        let reflected = front.r_s.norm_sqr() * cs.power() + front.r_p.norm_sqr() * cp.power();
        let ns = mix_c(&cs, &cp, j(0, 0), j(0, 1));
        let np = mix_c(&cs, &cp, j(1, 0), j(1, 1));
        let gx = mix(&ns, &np, s_hat.dot(out.x), p_hat.dot(out.x));
        let gy = mix(&ns, &np, s_hat.dot(out.y), p_hat.dot(out.y));
        let after = gx.power() + gy.power();
        let l = &mut self.report.ledger;
        l.other_arm += reflected;
        l.absorbed_element += (before - after - reflected).max(0.0);
        if let Some(st) = self.report.steps.last_mut() {
            st.note += &format!("; its Jones matrix on the field's two components ({what}), at the chief ray's {:.3}°", cos_i.acos().to_degrees());
        }
        self.report.approx("polarizers and retarders on the wave path: one Jones matrix for the whole field, the chief ray's");
        if n_sub > 1.0 {
            self.report.approx(format!("#{id}: the element's faces added singly, at the chief ray's angle: no bounce between them"));
        }
        if matches!(response, SurfaceResponse::Retarder { crystal: Some(_), .. }) {
            self.report.approx(format!("#{id}: one wave through the crystal, no walk-off between its ordinary and extraordinary waves"));
        }
        Ok((gx, gy))
```

**An LCoS** works on the component along its director (modulated beam) or
across it (unmodulated beam), whichever the chain follows [Zhang, You & Chu
2014]:
- That component is projected out of the pair and run through the device
  (§7.7).
- The device's absorbed fraction `ρ`, measured on that component, is applied
  to the other component too ("the device takes the same share").
- The rest of the other component goes to `other_family`.
- The component leaves along the device's image of its direction (its part
  along the surface kept).

Where both beams reach the detector, each is a chain of its own. They are
orthogonally polarized and add in power.

<!-- excerpt crates/oc-wave/src/path.rs 1385 1437 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 1385–1437, commit `1af40e8`.*

```rust
        if let Some((si, modulated)) = lcos {
            // The director across the beam, and the component the beam the
            // chain follows takes: along it modulated, across it not (Zhang,
            // You & Chu 2014).
            let s = &self.scene.surfaces[si];
            let h_id = s.common.id.0;
            let t = s.common.transform;
            let n = t.rotate(Vec3::Z);
            let ModulatorKind::Lcos(l) = (match if beam0.d.dot(n) < 0.0 { &s.response_front } else { &s.response_back } {
                SurfaceResponse::Modulator { device, .. } => &device.kind,
                _ => unreachable!("an LCoS is noted"),
            }) else {
                unreachable!("an LCoS is noted")
            };
            let dl = l.director_deg.to_radians();
            let dir = t.rotate(Vec3::new(dl.cos(), dl.sin(), 0.0));
            let dir = (dir - beam0.d * beam0.d.dot(dir)).normalize();
            let keep = if modulated { dir } else { beam0.d.cross(dir) };
            let c = mix(&ux, &uy, beam0.x.dot(keep), beam0.y.dot(keep));
            let total = ux.power() + uy.power();
            let kept = c.power();
            let dropped = (total - kept).max(0.0);
            // The device takes the same share of either component (its fill
            // and reflectance do not tell the beams apart), measured on the
            // kept one, or on the other standing in for it when the kept one
            // is dark; the rest of the other component goes into the other
            // beam, which the chain does not follow.
            let before = self.report.ledger;
            let stand_in = kept <= 0.0 && dropped > 0.0;
            let ran = if stand_in { dropped } else { kept };
            self.u = if stand_in { mix(&ux, &uy, beam0.x.dot(beam0.d.cross(keep)), beam0.y.dot(beam0.d.cross(keep))) } else { c };
            self.pol = Pol::Projected;
            self.element(idx, ev.clone(), p)?;
            self.pol = Pol::Unpolarized;
            let rho = if ran > 0.0 { ((self.report.ledger.absorbed_element - before.absorbed_element) / ran).clamp(0.0, 1.0) } else { 0.0 };
            if stand_in {
                self.report.ledger = before;
                self.u.data.iter_mut().for_each(|v| *v = Cx::ZERO);
            }
            self.report.ledger.absorbed_element += dropped * rho;
            self.report.ledger.other_family += dropped * (1.0 - rho);
            self.arms[idx] = Some(Arm { object: h_id, shared: false, foreign_absorbed: dropped * rho, foreign_family: dropped * (1.0 - rho) });
            // The component leaves along the device's image of its direction:
            // its part along the surface kept, as the tangential chart has it.
            let out = self.beam;
            let kt = keep - n * n.dot(keep);
            let kn = -kt.dot(out.d) / n.dot(out.d);
            let e = (kt + n * kn).normalize();
            let o = self.u.clone();
            self.u = mix(&o, &o, e.dot(out.x), 0.0);
            self.uy = Some(mix(&o, &o, e.dot(out.y), 0.0));
            return Ok(());
        }
```

### 7.11 Interferometers: arms added as fields

Where several chains reach the detector, each carries the same launched field
along its own arm. Each arm is read out onto **the detector's own sub-pixel
grid**:
- `s × s` points per pixel, with `s = 2` for `Readout::Projected`;
- by the zoomed angular spectrum when the face is square to the beam with its
  axes along the beam's;
- by the tilted-plane spectrum otherwise.

The arms' fields are added before the power is formed:

```text
P_pixel = Σ_points |Σ_arms U_c √cos θ_c|² ΔA
```

Each arm is weighted by the root of its obliquity on the face. One arm alone
then reads as it would by itself under the diffracted readout, though not
under the projected one, which bins power geometrically. Two-component arms are added component
by component, so orthogonally polarized arms do not interfere [Hecht 2017].
Where the chains carry one field, the arms' polarizations are taken as the
same, and the fringes' visibility is an upper bound (named).

Two arms crossing at an angle make fringes of period `λ/|t₁ − t₂|`, with `t`
the directions' parts along the face. A period under two readout points
raises `Undersampled`.

This is the coherent sum of paths the scattering-network formulation asks
for, `T = C(I − SP)⁻¹SB` [Bundle, Eq. M00; Appendix E]:
- `S` holds the components' scattering;
- `P` the interconnections between them;
- `B` puts the source into the network;
- `C` reads the detector.

Here the network has no feedback, so the sum of paths is finite.

**Which readout applies.**
- With **one chain** per mode, the spec's readout applies: `Projected` or
  `Diffracted` (§7.13).
- With **several chains** (arms), each arm is read on the face's sub-pixel
  grid by the zoomed or tilted angular spectrum, whatever the spec's readout,
  and the fields are added. Under `Readout::Projected` that grid has `s = 2`
  points per pixel; under `Diffracted{s}` it has `s`.

The code below shows:
- the per-arm readout (`face`);
- the sum;
- how the network ledger knows which events the arms share. Each event has a
  key built from its object, its action and its point. A splitter
  (`ExplicitRt` or `Coating`) or an LCoS registers itself as an `Arm` at its
  event.

**The network's ledger:**
- `launched` is counted once.
- Each arm's own losses are counted, and an event the arms share is counted
  once, since they carry one field until they part.
- A splitter several arms meet books no `other_arm`, since that light reaches
  the detector by another arm.
- What leaves by ports no arm reads is the **remainder**,
  `other_ports = launched − accounted`. This covers a recombining splitter's
  other output, what misses the detector once the arms meet, and the
  readout's own window and evanescent losses. It is named, never checked, and
  nothing tests its sign (§11.1).

<!-- excerpt crates/oc-wave/src/path.rs 2405 2417 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2405–2417, commit `1af40e8`.*

```rust
                if let SurfaceResponse::Modulator { device, .. } = response {
                    self.u = modulator_step(&self.u, &beam, &out, s, device, &h, &self.spec, self.pol, &mut self.report)?;
                } else {
                    let (s, response) = (s.clone(), response.clone());
                    if matches!(response, SurfaceResponse::ExplicitRt { .. } | SurfaceResponse::Coating { .. }) {
                        self.arms[idx] = Some(Arm::splitter(h.object.0));
                    }
                    self.flat_element(&s, &response, &h, p, n_s, &out, from_front)?;
                }
                self.beam = out;
                self.u.frame = out.iso();
                self.u.n_medium = h.n_after;
            }
```

<!-- excerpt crates/oc-wave/src/path.rs 2713 2814 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2713–2814, commit `1af40e8`.*

```rust
impl ChainJob {
    /// The arm's field on the detector's own sub-pixel grid, `s × s` points a
    /// pixel at the centres of equal parts of it: the zoomed angular
    /// spectrum onto the face when it is square to the beam with its axes
    /// along the beam's, the spectrum rotated onto it otherwise
    /// ([`crate::asm_tilted`]). Every arm is read on the same points, so the
    /// fields can be added.
    fn face(mut self, s: usize) -> Result<FaceField, WaveError> {
        if self.next < self.points.len() {
            return Err(WaveError::OutOfRange("an arm read out before its chain is done".into()));
        }
        if let Some(uy) = self.uy.take() {
            let twin = self.twin(uy);
            let mut fx = self.face(s)?;
            let fy = twin.face(s)?;
            fx.arriving += fy.arriving;
            fx.data_y = Some(fy.data);
            return Ok(fx);
        }
        let beam = self.beam;
        let det = self
            .scene
            .detectors
            .iter()
            .find(|d| d.common.id.0 == self.spec.detector.0)
            .ok_or_else(|| WaveError::OutOfRange("no such detector".into()))?;
        let (hw, hh) = match det.geometry {
            DetectorGeometry3::Rectangle { half_width_mm, half_height_mm } => (half_width_mm, half_height_mm),
            _ => return Err(WaveError::OutOfRange("the wave readout needs a rectangular detector".into())),
        };
        let [nu, nv] = det.bins;
        let t = det.common.transform;
        let n_det = t.rotate(Vec3::Z);
        let (du, dv) = (2.0 * hw / nu as f64, 2.0 * hh / nv as f64);
        let z = (t.pos - beam.origin).dot(beam.d);
        if z <= 0.0 {
            return Err(WaveError::OutOfRange("the detector is behind the field's last plane".into()));
        }
        let s = s.max(1);
        let (ps, qs) = (du / s as f64, dv / s as f64);
        let fg = centred(Grid2::new([nu * s, nv * s], [ps, qs], [0.0, 0.0])?);
        let arriving = self.u.power();
        let n_max = self.u.grid.n[0].max(self.u.grid.n[1]);
        let square = n_det.dot(beam.d).abs() > 1.0 - 1e-12;
        let (eu, ev) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
        let along = |a: Vec3, b: Vec3| a.dot(b).abs() > 1.0 - 1e-12;
        let aligned = if square && along(eu, beam.x) && along(ev, beam.y) {
            Some(([nu * s, nv * s], [ps, qs]))
        } else if square && along(eu, beam.y) && along(ev, beam.x) {
            Some(([nv * s, nu * s], [qs, ps]))
        } else {
            None
        };
        let mut data = vec![Cx::ZERO; fg.len()];
        let rep = if let Some((n, pitch)) = aligned {
            let plane = beam.origin + beam.d * z;
            let c = beam.coords(t.pos - beam.d * (t.pos - plane).dot(beam.d));
            let out = centred(Grid2::new(n, pitch, [c.0, c.1])?);
            let pad = if n_max >= 1024 { 1 } else { 2 };
            let (zf, rep) = crate::asm_zoom(&self.u, z, out, pad, &mut self.fft)?;
            let k = zf.k_mm();
            for j in 0..out.n[1] {
                for i in 0..out.n[0] {
                    let (a, b) = out.xy(i, j);
                    let l = t.inverse_transform_point(plane + beam.x * a + beam.y * b);
                    let (iu, iv) = (((l.x + hw) / ps).floor(), ((l.y + hh) / qs).floor());
                    if iu < 0.0 || iv < 0.0 || iu >= (nu * s) as f64 || iv >= (nv * s) as f64 {
                        continue;
                    }
                    data[fg.idx(iu as usize, iv as usize)] = zf.data[out.idx(i, j)] * Cx::expi(k * (zf.carrier[0] * a + zf.carrier[1] * b));
                }
            }
            rep
        } else {
            let pad = if n_max <= 512 { 4 } else if n_max <= 1024 { 2 } else { 1 };
            let (zt, rep) = crate::asm_tilted(&self.u, t, fg, pad, &mut self.fft)?;
            data = zt.data;
            rep
        };
        let mut report = self.report;
        report.steps.extend(rep.steps);
        for w in rep.warnings {
            report.warn(w);
        }
        Ok(FaceField {
            grid: fg,
            data,
            cos: n_det.dot(beam.d).abs(),
            dir: beam.d,
            frame: t,
            lambda_um: self.lam,
            n_medium: self.u.n_medium,
            arriving,
            data_y: None,
            report,
            keys: self.keys,
            arms: self.arms,
            ledgers: self.ledgers,
            events: self.points.len(),
        })
    }
}
```

<!-- excerpt crates/oc-wave/src/path.rs 2854 2942 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2854–2942, commit `1af40e8`.*

```rust
/// Several arms of one coherent mode, added on the detector's grid.
///
/// The power a pixel catches is `Σ |Σ_c U_c √cos θ_c|² ΔA` over its points:
/// each arm's field weighted by the root of its obliquity on the face, so
/// that one arm alone reads as it would by itself. Two arms crossing at an
/// angle make fringes of period `λ/|t₁ − t₂|`, `t` the directions' components
/// along the face; a period under two of the readout's points is flagged.
///
/// The ledger counts each arm's losses, an event the arms share once (they
/// carry the same field up to where they part). What a splitter sends into
/// another arm is not lost, so a splitter met by several arms books no
/// `other_arm`; the light that leaves by the ports no arm reads (a
/// recombining splitter's other output, what misses the detector once the
/// arms meet) is then the remainder, `other_ports`.
fn combine_arms(scene: &Scene3, spec: &WavePathSpec, faces: Vec<FaceField>, revision: u64) -> Result<WavePathResult, WaveError> {
    let det = scene
        .detectors
        .iter()
        .find(|d| d.common.id.0 == spec.detector.0)
        .ok_or_else(|| WaveError::OutOfRange("no such detector".into()))?;
    let [nu, nv] = det.bins;
    let g = faces[0].grid;
    let s = g.n[0] / nu;
    let (ps, qs) = (g.pitch_mm[0], g.pitch_mm[1]);
    let mut data = vec![Cx::ZERO; g.len()];
    let two = faces.iter().any(|f| f.data_y.is_some());
    let mut data_y = vec![Cx::ZERO; if two { g.len() } else { 0 }];
    for f in &faces {
        let w = f.cos.sqrt();
        for (a, b) in data.iter_mut().zip(&f.data) {
            *a += *b * w;
        }
        if let Some(dy) = &f.data_y {
            for (a, b) in data_y.iter_mut().zip(dy) {
                *a += *b * w;
            }
        }
    }
    let mut power = vec![0.0; nu * nv];
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let k = g.idx(i, j);
            let y = if two { data_y[k].norm_sqr() } else { 0.0 };
            power[(j / s) * nu + i / s] += (data[k].norm_sqr() + y) * ps * qs;
        }
    }
    let mut report = WaveReport::default();
    for (c, f) in faces.iter().enumerate() {
        for st in &f.report.steps {
            let mut st = st.clone();
            st.note = format!("arm {}: {}", c + 1, st.note);
            report.steps.push(st);
        }
        for w in &f.report.warnings {
            report.warn(w.clone());
        }
        for v in &f.report.validity {
            report.flag(v.clone());
        }
        for a in &f.report.approximated {
            report.approx(a.clone());
        }
    }
    // The fringes between each pair of arms, against the readout's points.
    let n_det = det.common.transform.rotate(Vec3::Z);
    let lam_mm = faces[0].lambda_um / 1000.0 / faces[0].n_medium;
    for a in 0..faces.len() {
        for b in a + 1..faces.len() {
            let (da, db) = (faces[a].dir, faces[b].dir);
            let ta = da - n_det * da.dot(n_det);
            let tb = db - n_det * db.dot(n_det);
            let gap = (ta - tb).length();
            if gap > 0.0 {
                let period = lam_mm / gap;
                let h = ps.min(qs);
                if period < 2.0 * h {
                    report.warn(SamplingWarning::Undersampled {
                        what: format!(
                            "the fringes of arms {} and {} on detector #{}, {period:.4e} mm apart, against readout points {h:.4e} mm apart",
                            a + 1,
                            b + 1,
                            spec.detector.0
                        ),
                        max_phase_step_rad: 2.0 * std::f64::consts::PI * h / period,
                    });
                }
            }
        }
    }
```

<!-- excerpt crates/oc-wave/src/path.rs 2943 2991 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2943–2991, commit `1af40e8`.*

```rust
    // The network's ledger.
    let mut l = WaveLedger { launched: faces[0].ledgers[0].launched, ..Default::default() };
    add_losses(&mut l, &faces[0].ledgers[0]);
    let mut met: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for f in &faces {
        let mut objs: Vec<u64> = f.arms.iter().flatten().map(|a| a.object).collect();
        objs.sort_unstable();
        objs.dedup();
        for o in objs {
            *met.entry(o).or_default() += 1;
        }
    }
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut parted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for f in &faces {
        let mut history = String::new();
        for i in 0..f.keys.len() {
            let before = history.clone();
            history.push_str(&f.keys[i]);
            history.push('|');
            if !seen.insert(history.clone()) {
                continue;
            }
            let mut d = losses_between(&f.ledgers[i + 1], &f.ledgers[i]);
            if let Some(arm) = f.arms[i] {
                let (o, shared) = (arm.object, arm.shared);
                if met.get(&o).copied().unwrap_or(0) >= 2 {
                    // What went into the other arm reached the detector, and
                    // that arm books its own share.
                    if shared {
                        d.other_arm = 0.0;
                    } else {
                        d.absorbed_element -= arm.foreign_absorbed;
                        d.other_family -= arm.foreign_family;
                    }
                }
                // Where arms part at a splitter, they meet it with one field:
                // what it absorbs or passes by is counted once. At an LCoS
                // each beam meets the device with its own polarization, and
                // what the device takes of each is its own.
                if shared && !parted.insert(format!("{before}#{o}")) {
                    d = WaveLedger { resampling: d.resampling, numerical: d.numerical, ..Default::default() };
                }
            }
            add_losses(&mut l, &d);
        }
    }
    l.detected = power.iter().sum();
    l.other_ports = l.launched - l.accounted();
```

### 7.12 Spectral lines

A source of spectral lines is a set of mutually incoherent modes [Wolf
1982]. Each line runs coherently through every chain at its own wavelength,
with its chains traced at that wavelength and its share of the power, and the
lines add in power. So two lines' fringes wash out where they fall in
antiphase.

The limits:
- A continuous spectrum is refused: sampling it into lines is the source's
  choice to make, not the path's.
- More than 16 lines are refused.
- The field shown is the strongest line's.

<!-- excerpt crates/oc-wave/src/path.rs 3157 3201 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 3157–3201, commit `1af40e8`.*

```rust
        let mut lines: Vec<(f64, f64)> = match &src.spectrum {
            oc_core::scene::Spectrum::Monochromatic(l) => vec![(*l, 1.0)],
            oc_core::scene::Spectrum::Lines(v) => v.iter().copied().filter(|l| l.1 > 0.0).collect(),
            oc_core::scene::Spectrum::Continuum(_) => {
                return Err(WaveError::OutOfRange(format!(
                    "source #{} emits a continuous spectrum; the wave path runs each spectral line coherently and adds the lines in power: give it a single line or a few lines",
                    spec.source.0
                )))
            }
        };
        if lines.is_empty() {
            return Err(WaveError::OutOfRange(format!("source #{} emits no line of positive weight", spec.source.0)));
        }
        if lines.len() > MAX_WAVE_LINES {
            return Err(WaveError::OutOfRange(format!(
                "source #{} emits {} lines; the wave path runs at most {MAX_WAVE_LINES}, each a coherent computation: give it fewer",
                spec.source.0,
                lines.len()
            )));
        }
        lines.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = lines.iter().map(|l| l.1).sum();
        let shared = Arc::new(scene.clone());
        let mut modes = Vec::with_capacity(lines.len());
        for (lam, w) in lines {
            let seqs = oc_trace::extract_sequences3(scene, spec.source, spec.detector, Some(lam), revision)
                .map_err(|e| WaveError::OutOfRange(format!("no chain from the source to the detector: {e}")))?;
            let face = seqs.len() > 1;
            let pols: Vec<Option<[f64; 2]>> =
                if polarization_matters(scene, &seqs, 0.5 * spec.window_mm, lam) { vec![Some([1.0, 0.0]), Some([0.0, 1.0])] } else { vec![None] };
            let share = 1.0 / pols.len() as f64;
            for jones in pols {
                let chains = seqs
                    .iter()
                    .cloned()
                    .map(|seq| ChainJob::new(&shared, spec, seq, spec.power * w / total * share, face, jones).map(|j| Chain::Running(Box::new(j))))
                    .collect::<Result<Vec<_>, _>>()?;
                let label = match jones {
                    None => format!("λ {lam} µm"),
                    Some(j) => format!("λ {lam} µm, launched along {}", if j[1] == 0.0 { "x" } else { "y" }),
                };
                modes.push(Mode { lambda_um: lam, label, chains });
            }
        }
        Ok(WaveJob { scene: shared, spec: *spec, revision, modes })
```

### 7.13 Readouts

These readouts apply to a mode of one chain. Several arms are always read as
fields (§7.11).

**`Readout::Projected`** (the app's) bins power. Each field cell's `|U|²ΔA`
goes to the pixel its projection along the beam lands in, split into `s × s`
sub-cells with `s = clamp(⌈8·Δx/(cos θ·min(pixel pitch))⌉, 1, 16)`, `θ` the
beam's angle to the face's normal. Power is conserved to
round-off, and what misses the face is booked as `left_window`. The phase
plays no part: it is a geometric projection of the last plane's power.

<!-- excerpt crates/oc-wave/src/path.rs 2474 2518 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2474–2518, commit `1af40e8`.*

```rust
        // A detector integrates power, so the readout bins it: each field
        // cell's power |U|²ΔA goes to the pixel its projection along the beam
        // lands in, as `s × s` sub-cells so that each, stretched by 1/cos on
        // a tilted face, spans at most an eighth of a pixel (a cell snapped
        // whole to a pixel would move the light by up to half a cell). Power
        // is conserved to round-off; what misses the face left the window.
        let g = self.u.grid;
        let o = t.pos;
        let cell = g.pitch_mm[0].max(g.pitch_mm[1]) / cos.max(1e-6);
        let s = ((8.0 * cell / du.min(dv)).ceil() as usize).clamp(1, 16);
        let share = g.cell_area_mm2() / (s * s) as f64;
        let mut power = vec![0.0; nu * nv];
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let p = self.u.data[g.idx(i, j)].norm_sqr() * share;
                if p == 0.0 {
                    continue;
                }
                let (x, y) = g.xy(i, j);
                for sj in 0..s {
                    for si in 0..s {
                        let a = x + ((si as f64 + 0.5) / s as f64 - 0.5) * g.pitch_mm[0];
                        let b = y + ((sj as f64 + 0.5) / s as f64 - 0.5) * g.pitch_mm[1];
                        let Some(r) = onto_plane(beam.point(a, b), beam.d, o, n_det) else {
                            continue;
                        };
                        let l = t.inverse_transform_point(r);
                        if l.x.abs() > hw || l.y.abs() > hh {
                            continue;
                        }
                        let iu = (((l.x + hw) / du) as usize).min(nu - 1);
                        let iv = (((l.y + hh) / dv) as usize).min(nv - 1);
                        power[iv * nu + iu] += p;
                    }
                }
            }
        }
        let bins = det.bins;
        let on_detector: f64 = power.iter().sum();
        let in_face = on_detector;
        let arriving = self.u.power();
        self.report.ledger.left_window += (arriving - in_face).max(0.0);
        let arriving_power = arriving;
        self.report.ledger.detected = on_detector;
        self.report.ledger.close();
```

**`Readout::Diffracted { samples_per_pixel: s }`** takes the last free-space
step straight onto the detector, at `s × s` points a pixel:
- by `asm_zoom` onto the plane across the beam at the detector's centre, each
  pixel the mean irradiance of its points times its area and `cos`;
- for a face not square to the beam, by `asm_tilted` onto the face's own
  plane (§4.6).

A focal spot far smaller than the field's own samples is resolved, with no
paraxial step. Each pixel's power is the **mean of its points' irradiance**
times its area, a point sample of the integral. Where the irradiance has
structure finer than the points, this misreads the power (§11.1.3).
`centred(g)` moves an even axis's samples half a pitch, onto the centres of
the cells the window divides into.

<!-- excerpt crates/oc-wave/src/path.rs 2532 2686 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 2532–2686, commit `1af40e8`.*

```rust
impl ChainJob {
    /// The diffracted readout: the field carried from where it is straight
    /// onto a window of the plane across the beam at the detector's centre,
    /// at `s × s` points a pixel, then binned into the pixels.
    fn finish_diffracted(
        mut self,
        s: usize,
        (hw, hh): (f64, f64),
        [nu, nv]: [usize; 2],
        (du, dv): (f64, f64),
        revision: u64,
    ) -> Result<WavePathResult, WaveError> {
        let spec = self.spec;
        let beam = self.beam;
        let det = self
            .scene
            .detectors
            .iter()
            .find(|d| d.common.id.0 == spec.detector.0)
            .ok_or_else(|| WaveError::OutOfRange("no such detector".into()))?;
        let t = &det.common.transform;
        let n_det = t.rotate(Vec3::Z);
        let z = (t.pos - beam.origin).dot(beam.d);
        if z <= 0.0 {
            return Err(WaveError::OutOfRange("the detector is behind the field's last plane".into()));
        }
        // A face that is not square to the beam: the spectrum rotated onto the
        // face's own plane (Matsushima, Schimmel & Wyrowski 2003), on its own
        // subpixel grid; each pixel its mean irradiance over its area.
        if n_det.dot(beam.d).abs() <= 1.0 - 1e-12 {
            let s = s.max(1);
            let grid = centred(Grid2::new([nu * s, nv * s], [du / s as f64, dv / s as f64], [0.0, 0.0])?);
            // The rotation reads the spectrum between its samples: pad finer
            // where the field is small enough to afford it.
            let n_max = self.u.grid.n[0].max(self.u.grid.n[1]);
            let pad = if n_max <= 512 { 4 } else if n_max <= 1024 { 2 } else { 1 };
            let arriving = self.u.power();
            let (zf, mut rep) = crate::asm_tilted(&self.u, *t, grid, pad, &mut self.fft)?;
            let cos = n_det.dot(beam.d).abs();
            let mut power = vec![0.0; nu * nv];
            for j in 0..nv * s {
                for i in 0..nu * s {
                    power[(j / s) * nu + i / s] += zf.data[grid.idx(i, j)].norm_sqr();
                }
            }
            for v in &mut power {
                *v *= du * dv * cos / (s * s) as f64;
            }
            let on_detector: f64 = power.iter().sum();
            rep.ledger.left_window = (arriving - on_detector).max(0.0);
            rep.ledger.detected = on_detector;
            merge(&mut self.report, rep);
            self.report.ledger.detected = on_detector;
            self.report.ledger.close();
            self.u = zf;
            return Ok(WavePathResult { revision, detector: spec.detector, bins: [nu, nv], power, field: self.u, field_y: None, arriving, report: self.report });
        }
        // The face's corners, projected along the beam onto the plane across
        // it at the detector's centre, give the window.
        let plane = beam.origin + beam.d * z;
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        let mut depth: f64 = 0.0;
        for (u, v) in [(-hw, -hh), (hw, -hh), (-hw, hh), (hw, hh)] {
            let r = t.transform_point(Vec3::new(u, v, 0.0));
            depth = depth.max((r - plane).dot(beam.d).abs());
            let (a, b) = beam.coords(r - beam.d * (r - plane).dot(beam.d));
            lo = [lo[0].min(a), lo[1].min(b)];
            hi = [hi[0].max(a), hi[1].max(b)];
        }
        // A face square to the beam with its axes along the beam's is read
        // on its own subpixel grid, `s × s` points at the centres of equal
        // parts of each pixel (one point: the pixel's centre). Any other face
        // is read on a grid along the beam's axes, each pixel taking the mean
        // of the points that fall in it.
        let s = s.max(1);
        let (eu, ev) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
        let square = n_det.dot(beam.d).abs() > 1.0 - 1e-12;
        let along = |a: Vec3, b: Vec3| a.dot(b).abs() > 1.0 - 1e-12;
        // The face's axes along the beam's, either way round.
        let aligned = if square && along(eu, beam.x) && along(ev, beam.y) {
            Some(([nu * s, nv * s], [du / s as f64, dv / s as f64]))
        } else if square && along(eu, beam.y) && along(ev, beam.x) {
            Some(([nv * s, nu * s], [dv / s as f64, du / s as f64]))
        } else {
            None
        };
        let (out, pitch_a, pitch_b) = if let Some((n, pitch)) = aligned {
            let c = beam.coords(t.pos - beam.d * (t.pos - plane).dot(beam.d));
            (centred(Grid2::new(n, pitch, [c.0, c.1])?), pitch[0], pitch[1])
        } else {
            let pitch = du.min(dv) / s as f64;
            let m = [
                (((hi[0] - lo[0]) / pitch).ceil() as usize).clamp(1, 4096),
                (((hi[1] - lo[1]) / pitch).ceil() as usize).clamp(1, 4096),
            ];
            (Grid2::new(m, [pitch, pitch], [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])])?, pitch, pitch)
        };
        let pitch = pitch_a.min(pitch_b);
        let m = out.n;
        // Without padding the field repeats a window away, which a beam
        // converging onto the detector never reaches; small fields are padded.
        let pad = if self.u.grid.n[0].max(self.u.grid.n[1]) >= 1024 { 1 } else { 2 };
        let arriving = self.u.power();
        let (zf, mut rep) = crate::asm_zoom(&self.u, z, out, pad, &mut self.fft)?;
        if depth > pitch * pitch / zf.wavelength_in_medium_mm() {
            self.report.warn(SamplingWarning::TiltProjection { depth_mm: depth });
        }
        // Each pixel: the mean irradiance of its points, over its area seen
        // across the beam.
        let mut sum = vec![0.0; nu * nv];
        let mut count = vec![0usize; nu * nv];
        for j in 0..m[1] {
            for i in 0..m[0] {
                let (a, b) = out.xy(i, j);
                let r = plane + beam.x * a + beam.y * b;
                let Some(q) = onto_plane(r, beam.d, t.pos, n_det) else { continue };
                let l = t.inverse_transform_point(q);
                if l.x.abs() > hw || l.y.abs() > hh {
                    continue;
                }
                let iu = (((l.x + hw) / du) as usize).min(nu - 1);
                let iv = (((l.y + hh) / dv) as usize).min(nv - 1);
                sum[iv * nu + iu] += zf.data[out.idx(i, j)].norm_sqr();
                count[iv * nu + iu] += 1;
            }
        }
        let cos = n_det.dot(beam.d).abs();
        if count.contains(&0) {
            self.report.warn(SamplingWarning::Undersampled {
                what: format!("the diffracted readout of detector #{}: a pixel got no sample", spec.detector.0),
                max_phase_step_rad: 0.0,
            });
        }
        let power: Vec<f64> =
            sum.iter().zip(&count).map(|(s, &c)| if c > 0 { s / c as f64 * du * dv * cos } else { 0.0 }).collect();
        let on_detector: f64 = power.iter().sum();
        // What the window held but the face did not left the window too.
        rep.ledger.left_window += (zf.power() - on_detector).max(0.0);
        merge(&mut self.report, rep);
        self.report.ledger.detected = on_detector;
        self.report.ledger.close();
        self.u = zf;
        self.u.frame = beam.turned(plane, beam.d).iso();
        Ok(WavePathResult {
            revision,
            detector: spec.detector,
            bins: [nu, nv],
            power,
            field: self.u,
            field_y: None,
            arriving,
            report: self.report,
        })
    }
}
```

<!-- excerpt crates/oc-wave/src/path.rs 371 381 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 371–381, commit `1af40e8`.*

```rust
/// The grid's samples moved onto the centres of the cells it divides its
/// width into: `Grid2` puts sample `n/2` on the centre, which on an even axis
/// is half a pitch off every cell's centre.
fn centred(mut g: Grid2) -> Grid2 {
    for a in 0..2 {
        if g.n[a] % 2 == 0 {
            g.center_mm[a] += 0.5 * g.pitch_mm[a];
        }
    }
    g
}
```

### 7.14 Reading a field between its samples

Bilinear interpolation of a complex field whose phase turns by `φ` from one
sample to the next lowers its amplitude between them:

```text
|(1 − s) + s·e^{iφ}|² = 1 − 2s(1 − s)(1 − cos φ)
```

averaged over a cell, that is a power loss of `(1 − cos φ)/3 ≈ φ²/6` per
axis. A beam converging inside a lens turns most of a radian per sample at
its edge. Before this treatment (historical, from the commit history; not
checkable here), the f/10 singlet of
`a_singlet_psf_matches_optiland_huygens_psf` lost 3.7% of the power at its
second face. So a smooth wavefront is taken out of the samples first and
put back exactly at the point read:
- for a transfer, the transfer's reference;
- for the other hand-overs, a sphere and tilt fitted to the field by
  intensity-weighted least squares on the phase gradient.

Only the slowly varying rest is interpolated, and the cost is booked as
`resampling` at each step.

**Linearity.** The fitted sphere depends on the field. So hand-overs and
modulator take-ups are linear only up to their interpolation error. The
transfer (§7.5) is the operator whose linearity is tested.

<!-- excerpt crates/oc-wave/src/path.rs 196 211 -->
*Excerpt: `crates/oc-wave/src/path.rs`, lines 196–211, commit `1af40e8`.*

```rust
/// Bilinear sample of a field's data at local `(a, b)`; zero outside.
pub(crate) fn sample(u: &Field2, a: f64, b: f64) -> Cx {
    let g = u.grid;
    let fi = (a - g.center_mm[0]) / g.pitch_mm[0] + (g.n[0] / 2) as f64;
    let fj = (b - g.center_mm[1]) / g.pitch_mm[1] + (g.n[1] / 2) as f64;
    if !(fi >= 0.0 && fj >= 0.0 && fi <= (g.n[0] - 1) as f64 && fj <= (g.n[1] - 1) as f64) {
        return Cx::ZERO;
    }
    let (i0, j0) = (fi.floor() as usize, fj.floor() as usize);
    let (i1, j1) = ((i0 + 1).min(g.n[0] - 1), (j0 + 1).min(g.n[1] - 1));
    let (s, t) = (fi - i0 as f64, fj - j0 as f64);
    u.at(i0, j0) * ((1.0 - s) * (1.0 - t))
        + u.at(i1, j0) * (s * (1.0 - t))
        + u.at(i0, j1) * ((1.0 - s) * t)
        + u.at(i1, j1) * (s * t)
}
```

<!-- excerpt crates/oc-wave/src/transfer.rs 320 423 -->
*Excerpt: `crates/oc-wave/src/transfer.rs`, lines 320–423, commit `1af40e8`.*

```rust
/// A field's wavefront: direction cosines `t = (α + c·a, β + c·b)` across
/// the beam (carrier included), fitted by intensity-weighted least squares to
/// the phase gradient where the field is bright. Returns `[c, α, β]` and the
/// largest departure of a bright sample's direction from the fit; the
/// carrier alone when nothing is bright enough to fit.
pub(crate) fn fit_sphere(u: &Field2) -> ([f64; 3], f64) {
    let gu = u.grid;
    let k_in = u.k_mm();
    let mut top: f64 = 0.0;
    for c in &u.data {
        top = top.max(c.norm_sqr());
    }
    // (a, b, tx, ty, weight) at each bright sample.
    let mut grads: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for j in 1..gu.n[1].saturating_sub(1) {
        for i in 1..gu.n[0].saturating_sub(1) {
            let w = u.at(i, j).norm_sqr();
            if w < 1e-3 * top {
                continue;
            }
            let gx = (u.at(i + 1, j) * u.at(i - 1, j).conj()).arg() / (2.0 * gu.pitch_mm[0]);
            let gy = (u.at(i, j + 1) * u.at(i, j - 1).conj()).arg() / (2.0 * gu.pitch_mm[1]);
            let (a, b) = gu.xy(i, j);
            grads.push((a, b, gx / k_in + u.carrier[0], gy / k_in + u.carrier[1], w));
        }
    }
    // tx = c·a + α and ty = c·b + β, unknowns (c, α, β), weighted by intensity.
    let [curv, alpha, beta] = {
        let (mut m, mut v) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
        for &(a, b, tx, ty, w) in &grads {
            for (f, y) in [([a, 1.0, 0.0], tx), ([b, 0.0, 1.0], ty)] {
                for r in 0..3 {
                    v[r] += w * f[r] * y;
                    for q in 0..3 {
                        m[r][q] += w * f[r] * f[q];
                    }
                }
            }
        }
        square3(m, v).unwrap_or([0.0, u.carrier[0], u.carrier[1]])
    };
    let mut angle: f64 = 0.0;
    for &(a, b, tx, ty, _) in &grads {
        angle = angle.max(((tx - curv * a - alpha).powi(2) + (ty - curv * b - beta).powi(2)).sqrt());
    }
    ([curv, alpha, beta], angle)
}

/// A field read between its samples. Bilinear interpolation of a complex
/// field whose phase turns by `φ` from one sample to the next lowers its
/// amplitude between them: `|(1 − s) + s·e^{iφ}|² = 1 − 2s(1 − s)(1 − cos φ)`,
/// a power loss of `(1 − cos φ)/3 ≈ φ²/6` per axis averaged over a cell. A
/// beam converging at a few hundredths of a radian turns most of a radian
/// per sample at its edge, so the sphere and tilt the field is fitted to are
/// taken out of the samples first, and put back exactly at the point read:
/// what is interpolated is the slowly varying rest.
pub(crate) struct Smooth {
    d: Field2,
    k: f64,
    r: Reference,
    car: [f64; 2],
}

impl Smooth {
    /// `u` with its own fitted sphere taken out.
    pub(crate) fn new(u: &Field2) -> Self {
        let ([c, a, b], _) = fit_sphere(u);
        Self::with(u, [c, a - u.carrier[0], b - u.carrier[1]])
    }

    /// `u` with the phase `k·(α·a + β·b + c·(a² + b²)/2)` taken out, `s` =
    /// `[c, α, β]` relative to the carrier (the samples' own phase).
    pub(crate) fn with(u: &Field2, s: [f64; 3]) -> Self {
        Self::about(u, &Reference { curv: [s[0], s[0]], tilt: [s[1] + u.carrier[0], s[2] + u.carrier[1]], centre: [0.0, 0.0] })
    }

    /// `u` with a reference wavefront's phase taken out.
    pub(crate) fn about(u: &Field2, r: &Reference) -> Self {
        let k = u.k_mm();
        let mut d = u.clone();
        let g = u.grid;
        let car = u.carrier;
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (a, b) = g.xy(i, j);
                let idx = g.idx(i, j);
                d.data[idx] *= Cx::expi(-ref_phase(k, r, car, a, b));
            }
        }
        Self { d, k, r: *r, car }
    }

    /// The field's sample value at `(a, b)` (carrier excluded); zero outside.
    pub(crate) fn at(&self, a: f64, b: f64) -> Cx {
        crate::path::sample(&self.d, a, b) * Cx::expi(ref_phase(self.k, &self.r, self.car, a, b))
    }
}

/// A reference's phase at `(a, b)`, relative to a carrier: its gradient is
/// `k` times the reference's direction less the carrier.
fn ref_phase(k: f64, r: &Reference, car: [f64; 2], a: f64, b: f64) -> f64 {
    let (da, db) = (a - r.centre[0], b - r.centre[1]);
    k * ((r.tilt[0] - car[0]) * a + (r.tilt[1] - car[1]) * b + 0.5 * (r.curv[0] * da * da + r.curv[1] * db * db))
}
```

### 7.15 What is refused, by name

Each refusal is an error that names its reason; nothing is skipped:
- a measured R/T table (it gives no phase);
- any surface response with no wave model;
- an aperture inside a face's slab or a plate;
- a continuous spectrum, or more than 16 lines;
- a source with no declared coherent beam;
- a non-rectangular detector;
- an LCoS met by a one-component field;
- a plate whose rim is not star-shaped about the chief ray;
- a beam past the critical angle of a polarizer's or a retarder's substrate;
- a beam along a polarizer's or retarder's own axis;
- a polarizer whose extinction curve has no value at the angle met;
- a grid larger than the target allows;
- a face too close to the next event for a slab plane to fit;
- a traced face that fewer than nine rays pass, or that the chief ray does
  not cross to its exit plane, or whose map cannot be fitted;
- a thin lens of zero or non-finite focal length, and a transmittance above 1
  in magnitude.

---

## 8. The report and the ledger

Every call returns a `WaveReport`:

- `steps`: each with its method, distance, sampling regime, and a note in
  words;
- `warnings`: `BandLimited`, `WrapAround`, `OutsideRegime`, `Paraxial`,
  `Undersampled`, `TiltProjection`. `CoarsePreview` is defined but never
  raised;
- `validity` flags: `ScalarLimit`, `PhaseDispersionApprox`. `ThinElement` is
  defined but never raised;
- `approximated`: named approximations, as sentences;
- `rejected`: refusals;
- a `WaveLedger`, in which every step books what it changed to a named
  cause:

```text
launched = detected + absorbed_aperture + absorbed_element + band_limit_removed
         + beyond_nyquist + evanescent + left_window + other_family
         + untracked_orders + other_arm + passed_by + resampling
         + other_ports + numerical
```

| Entry | Meaning | Booked by |
|---|---|---|
| `absorbed_aperture` | stopped by an aperture, a rim, a mount | apertures, rims, rays that miss a face |
| `absorbed_element` | absorbed by an element | mirrors (`1 − R`); splitters; a DMD's dead space, light off the array and `1 − R·T_w²`; an LCoS's `1 − R`, dead-space loss and light off the array; polarizers and retarders, including the back face's reflection |
| `band_limit_removed` | taken out by the ASM band limit | §4.1 |
| `beyond_nyquist` | diffracted into angles the grid cannot hold | edge cells (`f − f²`), band cuts before re-gridding and take-up, a top-hat's edge |
| `evanescent` | decayed evanescent waves | ASM, zoomed ASM, plates |
| `left_window` | left the sampled window | crops; readouts; plate round trips beyond the window (`PlateStats::beyond`); in `asm_tilted`, every loss not otherwise counted |
| `other_family` | a modulator family not followed | DMD OFF/ON mirrors, LCoS other polarization |
| `untracked_orders` | grating orders not followed | flat gratings |
| `other_arm` | light on a path not followed | splitters, face reflections, polarizer faces |
| `passed_by` | light beside an element that turns the chain | small mirrors, gratings |
| `resampling` | gained (negative) or lost carrying a field onto another grid or plane | hand-overs, transfers, take-ups, plate remainders |
| `other_ports` | ports no arm reads, in a network of arms | the remainder in `combine_arms` |
| `numerical` | whatever no step accounted for | the remainder in `close()` |

**The closure is bookkeeping, not a check.** `close()` defines `numerical`
as `launched − Σ(everything else)`, so the ledger sums by construction. The
evidence is in the entries:
- `resampling` should be small;
- `numerical` should be round-off;
- the physical entries should match references (the aperture and pinhole
  tests compare `absorbed_aperture` with closed forms).

Some steps put more than round-off into a remainder:
- **Direct Rayleigh–Sommerfeld** books the power leaving its window in
  `numerical` (§4.5).
- **The Fresnel impulse response** books a net gain in `numerical`.
- **`asm_tilted`** assigns `left_window = launched − detected`, so its
  backward and evanescent waves and its out-of-band reads land there.
- **The DMD and LCoS near fields** clip their `beyond_nyquist` at zero, so a
  negative edge term moves into `numerical`.
- **`combine_arms`** puts everything not otherwise booked into
  `other_ports`, with no sign check. That includes each arm's readout losses,
  and any double count (§11.1).

<!-- excerpt crates/oc-wave/src/report.rs 82 161 -->
*Excerpt: `crates/oc-wave/src/report.rs`, lines 82–161, commit `1af40e8`.*

```rust
/// Where the power went, as a fraction of nothing in particular: the same
/// units as [`crate::Field2::power`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct WaveLedger {
    pub launched: f64,
    /// Still in the field at the end.
    pub detected: f64,
    pub absorbed_aperture: f64,
    pub absorbed_element: f64,
    /// Removed by a band limit.
    pub band_limit_removed: f64,
    /// Diffracted by a sharp edge into angles beyond the grid's Nyquist
    /// band: power the sampled field cannot carry. An aperture edge passing
    /// a fraction `f` of a cell transmits `f·|U|²ΔA` physically, but the
    /// sample, `f·U`, carries only `f²·|U|²ΔA`; the difference is here, not
    /// in `absorbed_aperture`.
    #[serde(default)]
    pub beyond_nyquist: f64,
    /// Carried by evanescent components that decayed.
    pub evanescent: f64,
    /// Left the sampled window.
    pub left_window: f64,
    /// Sent into a state family (a DMD's OFF mirrors, an LCoS's unmodulated
    /// polarization) that this computation did not follow.
    pub other_family: f64,
    /// Diffraction orders not followed.
    pub untracked_orders: f64,
    /// Sent by a splitter into its other arm, or reflected by a refracting
    /// face, on a path this computation does not follow. Not absorbed.
    #[serde(default)]
    pub other_arm: f64,
    /// Passed beside an element's outline, along a direction this
    /// computation does not follow (beside a fold mirror, a grating).
    #[serde(default)]
    pub passed_by: f64,
    /// Gained (negative) or lost carrying a field onto another grid or plane:
    /// interpolation between samples, a traced map's fit, a window's edge.
    #[serde(default)]
    pub resampling: f64,
    /// Left by ports a computation of several arms does not read: a
    /// recombining splitter's other output, and what misses the detector
    /// once the arms meet. It is the remainder after every arm's own losses
    /// are booked, so it is named, never a check.
    #[serde(default)]
    pub other_ports: f64,
    /// Round-off: whatever is left after every cause above.
    pub numerical: f64,
}

impl WaveLedger {
    /// Everything booked, `numerical` included.
    pub fn accounted(&self) -> f64 {
        self.detected
            + self.absorbed_aperture
            + self.absorbed_element
            + self.band_limit_removed
            + self.beyond_nyquist
            + self.evanescent
            + self.left_window
            + self.other_family
            + self.untracked_orders
            + self.other_arm
            + self.passed_by
            + self.resampling
            + self.other_ports
            + self.numerical
    }

    /// Book the unexplained remainder to `numerical` and return its size
    /// relative to what was launched.
    pub fn close(&mut self) -> f64 {
        self.numerical = 0.0;
        let gap = self.launched - self.accounted();
        self.numerical = gap;
        if self.launched > 0.0 {
            gap.abs() / self.launched
        } else {
            gap.abs()
        }
    }
```

---

## 9. Ray-side models the wave path relies on

### 9.1 Sequence extraction

The tracer's models in this section are described from its code and
documentation. Only the excerpts shown are quoted.

The axial ray starts at the source frame's origin along its local `+X`. For
each line the wave path forces the line's own wavelength. The ray is traced
with ghosts off, unpolarized, with a flux cut-off far below the default. A
chain through `k` lossy splitters carries about `0.5^k` of the launch flux,
and must not be culled. Deterministic
splitters (`ExplicitRt`) follow both arms, and a grating spawns its zero
order and its declared order. Each branch lineage that reaches the detector
is a chain. Later hits of the detector on the same lineage are the same
chain arriving again, not a second chain.

The module's header, below, was written for the single-chain entry
`extract_sequence3`, which returns "the one branch lineage" and refuses
several as `Ambiguous`. The wave path calls `extract_sequences3`, which shares
the same trace and returns all the chains (`all = true` in the code that
follows the header).

<!-- excerpt crates/oc-trace/src/sequence3.rs 1 23 -->
*Excerpt: `crates/oc-trace/src/sequence3.rs`, lines 1–23, commit `1af40e8`.*

```rust
//! Sequence extraction: trace a source's **axial ray** through the scene and
//! read off the ordered surface chain that reaches a detector.
//!
//! This is the inversion that lets a sequential consumer (an Optiland emitter,
//! a paraxial ladder, a diagnostic view) stop *predicting* where the beam goes
//! and instead *ask*: the non-sequential reference tracer walks the actual
//! scene, the event log records what happened, and the prescription is the one
//! branch lineage that ends on the requested detector. Where a static
//! beam-path compiler needs alignment tolerances, fold idealizations, and
//! unsupported-case errors, the extractor needs none of them — a folded path,
//! a curved mirror in it, a splitter arm are all just what the ray did.
//!
//! Failure is physical and named, in the [`crate::result3::CapabilityReport`]
//! spirit: the axial ray either reached the detector (a prescription, with
//! anything it cannot express in `notes`), or it did not
//! ([`SequenceError::NotReached`] carries every branch's actual fate), or it
//! reached it more than one way ([`SequenceError::Ambiguous`] names the
//! chains, and the caller disambiguates by declaring a tighter intent).
//!
//! Extraction traces **one** launch ray with ghosts off: Fresnel
//! back-reflections are not part of a designed sequence, while deterministic
//! splitters ([`SurfaceResponse::ExplicitRt`](oc_core::scene3::SurfaceResponse))
//! follow both arms and are disambiguated by which arm reaches the detector.
```

<!-- excerpt crates/oc-trace/src/sequence3.rs 195 203 -->
*Excerpt: `crates/oc-trace/src/sequence3.rs`, lines 195–203, commit `1af40e8`.*

```rust
pub fn extract_sequences3(
    scene: &Scene3,
    source: SourceId,
    detector: DetectorId,
    lambda_um: Option<f64>,
    revision: u64,
) -> Result<Vec<SequencePrescription3>, SequenceError> {
    extract_chains(scene, source, detector, lambda_um, revision, true)
}
```

### 9.2 Element rules in the tracer that the wave path inherits

- **Apertures**: opening and outer extent (§7.4).
- **Ideal thin lens**: a slope law, `d′_t = d_t − |d_z|·h/f`, then
  normalized. This is a law in tangents, while the wave's lens phase
  `exp(−ikr²/2f)` deflects by `−h/f` in direction cosine. For a chief ray
  that hits the lens off its centre, the frame turns by the tracer's angle,
  and the hand-over phase removes it from the lens's phase. The difference,
  of third order in `h/f`, is left on the samples as a residual tilt.

<!-- excerpt crates/oc-trace/src/tracer3.rs 2010 2044 -->
*Excerpt: `crates/oc-trace/src/tracer3.rs`, lines 2010–2044, commit `1af40e8`.*

```rust
    // The ideal thin lens is the one response whose outgoing direction depends
    // on WHERE the surface was hit, so it needs the surface's local frame and
    // is handled here rather than in the frame-blind `apply_response`.
    if let SurfaceResponse::ThinLens { focal_length_mm } = response {
        let f = *focal_length_mm;
        if !f.is_finite() || f == 0.0 {
            r.report
                .reject(CapabilityIssue::UnsupportedSurfaceResponse {
                    surface: surface_ref, reason: None
                });
            r.ledger.numerical_loss += flux_at;
            r.terminations.record(TerminationReason::UnsupportedPhysics);
            push_event(r, cfg, w, point, opl_at, lens_interact(Vec::new()));
            return;
        }
        r.report.note_exact(PhysicsFeature::ThinLens);
        let t = &s.common.transform;
        let p_local = t.inverse_transform_point(point);
        let d_local = t.quat().inverse() * w.ray.dir;
        // Slope law u' = u − h/f per transverse axis, direction-aware: with the
        // axial component preserved, d'_t = d_t − |d_z|·h_t/f sends a forward
        // parallel bundle exactly through (0, 0, +f) and a backward one through
        // (0, 0, −f), and images every axial conjugate exactly (1/s_o + 1/s_i =
        // 1/f). A ray grazing in the lens plane (|d_z| ≈ 0) passes undeviated.
        let dir = if d_local.z.abs() < 1e-12 {
            w.ray.dir
        } else {
            let bent = Vec3::new(
                d_local.x - d_local.z.abs() * p_local.x / f,
                d_local.y - d_local.z.abs() * p_local.y / f,
                d_local.z,
            )
            .normalize();
            t.quat() * bent
        };
```

- **Grating**: the grating equation in direction cosines,
  `t_out = t_in + mλ·(lines/mm)` across the grooves. The declared order and
  the zero order are traced. (The tracer's code for this is not excerpted.)
- **DMD, geometric ray model** (`RayModel::Geometric`, the default):
  - a ray finds the pixel it hits;
  - inside that mirror's outline, projected onto the array plane, it
    reflects off the pixel's facet with power `R·T_w²`;
  - in the dead space it is absorbed.

  The chief ray's outgoing direction, and so the wave path's outgoing frame,
  is the facet reflection of the pixel the chief ray hits.
- **LCoS, geometric ray model**:
  - the component along the director is steered by the local phase gradient,
    `Δt = (λ/2π)∇φ` (the grating equation for a phase profile), and gains
    `OPL = φλ/2π`;
  - the gradient is a central difference of the displayed phases, wrapped
    into `(−π, π]`, and one-sided at the array's edge;
  - the orthogonal component reflects specularly;
  - unpolarized light splits evenly;
  - the dead space reflects specularly with `|g|²`.

  A **two-level** grating has a gradient of ±π per pixel, whose wrapped sign
  is ambiguous. With this model the chief ray does not reach a detector placed
  on the first order (§11.4).
- **Orders ray model** (`RayModel::Orders{max_orders, min_efficiency}`):
  - each hit spawns the brightest orders of the local unit cell, from the
    closed form of §6.2 with Harvey's renormalization;
  - the orders not followed are booked as untraced;
  - the local unit cell is one pixel of a uniform pattern, or one period of a
    binary or blazed grating that repeats in whole pixels along a pixel axis;
  - anything else is an 8 × 8 window starting at the hit pixel (moved back
    inside the array at its edge) taken as one period (named).

  With this model a modulator spawns several chains, which the wave path then
  adds as arms.

<!-- excerpt crates/oc-core/src/modulator.rs 665 699 -->
*Excerpt: `crates/oc-core/src/modulator.rs`, lines 665–699, commit `1af40e8`.*

```rust
/// The unit cell the pattern repeats with about pixel `(i, j)`, and whether it
/// is the pattern's own: one pixel of a uniform pattern, or the period of a
/// binary or blazed grating that repeats in whole pixels along a pixel axis.
/// Anything else is an 8 × 8 window from that pixel treated as one period,
/// an approximation (`false`).
pub fn local_unit_cell(d: &SpatialModulator, i: usize, j: usize) -> (UnitCell, bool) {
    let [nx, ny] = [d.pixels[0] as usize, d.pixels[1] as usize];
    let window = |px: usize, py: usize| {
        let (i0, j0) = (i.min(nx.saturating_sub(px)), j.min(ny.saturating_sub(py)));
        let values = (0..py)
            .flat_map(|b| (0..px).map(move |a| (a, b)))
            .map(|(a, b)| d.value((i0 + a).min(nx - 1), (j0 + b).min(ny - 1)))
            .collect();
        UnitCell { period: [px, py], values }
    };
    let period = match &d.pattern {
        Pattern::Uniform { value } => return (UnitCell::uniform(*value), true),
        Pattern::Generated(PatternGenerator::BinaryGrating { period_px, angle_deg, .. })
        | Pattern::Generated(PatternGenerator::BlazedGrating { period_px, angle_deg }) => {
            let p = period_px.round();
            let whole = (period_px - p).abs() < 1e-9 && p >= 1.0;
            let a = angle_deg.rem_euclid(180.0);
            match (whole, a) {
                (true, a) if a.abs() < 1e-9 && (p as usize) <= nx => Some([p as usize, 1]),
                (true, a) if (a - 90.0).abs() < 1e-9 && (p as usize) <= ny => Some([1, p as usize]),
                _ => None,
            }
        }
        _ => None,
    };
    match period {
        Some([px, py]) => (window(px, py), true),
        None => (window(8.min(nx), 8.min(ny)), false),
    }
}
```

### 9.3 The probe scene for a transfer

A face is traced in a copy of the scene that holds that element alone, with
no apertures, detectors or sources. The trace has ghosts off and polarization
off. A face's transmittance is therefore the tracer's **unpolarized** Fresnel
`T`, and a two-component field gives both components that same `√T`.

---

## 10. Evidence inventory

All Rust tests were rerun at `1af40e8` in release mode for this brief:
- `cargo test -p oc-wave --features scene --release -- --nocapture --test-threads=1`;
- `cargo test -p oc-trace --features trace3 --release --test modulator3 --test modulator_orders3 -- --nocapture`.

All 112 `oc-wave` tests and the 9 `oc-trace` DMD/SLM tests pass. The
"Measured" column gives the values printed by that run. The Python tests were
not rerun: they need Optiland installed, and their values are the
documentation's.

**Classes:**
- **A**: an independent closed form or analytic result. Several of these
  compute the reference with the crate's own helpers:
  - `special::bessel_j` and `special::fresnel_cs`, themselves held to DLMF
    tables;
  - `oc_core::modulator::{sinc, mirror_ft, popoff_blazed_number}`;
  - `GaussianBeam::value`;
  - `Shape::area`;
  - the canvas ray tracer, for focal lengths and principal planes.

  `bundle.rs` uses its own power-series Bessel function, and the path tests
  write the single-layer, slab, ABCD and Malus formulas inline.
- **B**: an external oracle, Optiland (a fixture, or a live run). This is
  verification of a shared approximation, not validation of the physics.
- **C**: self-consistency: one model computed two ways, a round trip,
  Parseval, linearity, or another canvas implementation.
- **D**: a vendor's simulation (Texas Instruments DLPA083B).
- **—**: I/O or input validation, with no numerical reference.

| File | Tests | A | B | C | D | — |
|---|---|---|---|---|---|---|
| `oc-wave/tests/analytic_fields.rs` | 8 | 8 | | | | |
| `oc-wave/tests/propagate.rs` | 10 | 1 | | 9 | | |
| `oc-wave/tests/czt.rs` | 3 | 1 | | 2 | | |
| `oc-wave/tests/fft.rs` | 3 | 1 | | 2 | | |
| `oc-wave/tests/bundle.rs` | 6 | 4 | | 2 | | |
| `oc-wave/tests/dmd.rs` | 9 | 6 | | 2 | 1 | |
| `oc-wave/tests/slm.rs` | 9 | 7 | | 2 | | |
| `oc-wave/tests/path.rs` | 40 | 30 | 1 | 8 | | 1 |
| `oc-wave/src` unit tests | 24 | 13 | | 7 | | 4 |
| `oc-trace` DMD/SLM tests | 9 | 4 | | 3 | | 2 |
| `oc-py` Python tests (functions) | 6 | 2 | 2 | | 1 | 1 |
| **Total** | **127** | **77** | **3** | **37** | **2** | **8** |

**No test uses measured data.** Every reference is one of four kinds:
- a closed form or exact solution;
- another implementation of the same integral or model;
- Optiland;
- TI's simulation.

Two tests are classed A although their geometric inputs come from the canvas
tracer. These are the singlet waist test (focal length and principal planes)
and the PSF normalization (the exit pupil). The formula they are held to is
independent; the inputs are not.

The test names below are the Rust functions' names. Where the prose calls a
test by a short name: "the DMD-and-lens test" is
`a_dmd_in_the_scene_focuses_its_orders_f_lambda_over_p_apart`, and "the
DMD-and-singlet test" is
`a_dmd_s_orders_cross_a_singlet_each_about_its_own_direction`.

### 10.1 Propagators, elements, transforms

The rows below come from four files:
- `analytic_fields.rs`: the first eight rows;
- `propagate.rs`: from `every_propagator_conserves_power…` to
  `a_tilted_plane_holds…`;
- `czt.rs`: the three chirp-z rows;
- `fft.rs`: the last two rows.

| Test | Held against | Gate | Measured | Cl. |
|---|---|---|---|---|
| `fresnel_integrals_and_bessel_functions_match_the_tables` | [DLMF] ch. 7, 10: C(1), S(1), J₀(1), J₁(1), j₁,₁; C, S asymptote at 40 | 1e-10 (1e-8) | passes | A |
| `a_square_aperture_far_field_is_sinc_squared` | Fraunhofer of a 0.2 mm square, λ 0.5 µm, z 1 m: `w² sinc(wx/λz) sinc(wy/λz)/(iλz)`·phases | rms 1e-3 of peak | rms 4.54e-4; worst 2.22e-3 (outer lobes) | A |
| `a_circular_aperture_makes_the_airy_pattern` | `2J₁(v)/v`; encircled energy `1 − J₀² − J₁²` = 0.8378 at the first dark ring | 2e-3; 1e-3 | 2.82e-4; 0.8376 | A |
| `the_on_axis_field_of_a_circular_aperture_matches_heurtley` | exact on-axis RS field [Heurtley 1973], `e^{ikz} − (z/√(z²+a²))e^{ik√(z²+a²)}`, N_F 10 → 0.25 | direct RS 5e-3 at a/Δx 256; error ratio > 3 on halving Δx; Richardson 3e-4; ASM 3e-3; Auto 3e-3 | 2.57e-3 → 1.62e-6; ratio 4.040–4.048; Richardson 3.91e-5 → 2.59e-8; band-limited ASM 2.15e-3 at N_F 10; Auto (IR) 9.42e-5, 2.32e-5, 5.77e-6 at N_F 1, 0.5, 0.25 | A |
| `a_square_aperture_fresnel_pattern_matches_the_fresnel_integrals` | Fresnel diffraction of a 0.5 mm square by C and S, z 50 mm | 1.5e-3 at 201 across; ratio > 3 | 3.52e-3 (TF, 101 across) → 8.44e-4 (IR, 201 across), ratio 4.17. Each sampling uses the method in its regime, so the ratio is between two methods, not one method's convergence | A |
| `a_gaussian_beam_spreads_as_kogelnik_and_li_say` | `w(z)` and the complex field after 300 mm | 1e-5 | 3.53e-7; 2.33e-7 of peak | A |
| `a_thin_lens_focuses_a_gaussian_where_self_predicts` | [Self 1983]'s waist position and size; flat wavefront at the waist | 1e-4; 1e-3 rad | 8.05e-10; 2.89e-5 rad | A |
| `a_cosine_grating_reimages_at_the_exact_talbot_distance` | `z_T = λ/(1 − √(1 − λ²/p²))`; half-period shift at `z_T/2` [Wen, Zhang & Xiao 2013] | 1e-10 | 5.13e-12; 2.56e-12 | A |
| `every_propagator_conserves_power_when_nothing_is_band_limited` | `|H| = 1` | 1e-12 (IR 1e-6) | ASM 2.79e-14; TF 1.69e-14; Fraunhofer 5.06e-14; IR 1.97e-11 | C |
| `asm_forward_then_back_returns_the_field` | reversibility | 1e-12 | 1.42e-13 | C |
| `voelz_regime_picks_the_transfer_function_below_the_critical_distance` | the `Auto` choice below and beyond `z_c`; the Fresnel TF and IR regime warnings. Despite the name, `Auto` picks the band-limited ASM below `z_c` (never the TF) | exact choice | ASM at both distances; beyond `z_c` the band limit's error 2.45e-8 < paraxial 2.47e-6 | C |
| `the_paraxial_step_s_error_is_the_identity_s_exactly` | `‖(U − U₂)g‖` against the identity, Gaussian tilted 0.15 rad | 1e-6 relative | −5.03e-8 | C |
| `rayleigh_sommerfeld_direct_agrees_with_band_limited_asm` | two discretizations of one integral | 1e-4 of peak | 3.90e-12 | C |
| `the_band_limit_reports_the_power_it_removes` | ledger closure | 1e-12 | 32.5% removed, booked | C |
| `an_aperture_passes_exactly_its_area_of_a_uniform_field` | the shape's area, four shapes | 1e-12 | passes | A |
| `the_zoomed_angular_spectrum_lands_on_the_angular_spectrum_on_its_own_grid` | FFT ASM on the same points | 1e-10 | 2.74e-14 | C |
| `an_off_axis_window_holds_the_field_the_large_window_does` | a window 4× as wide | 1e-6 | 1.91e-12 | C |
| `a_tilted_plane_holds_the_field_its_plane_waves_sum_to` | direct plane-wave sum of the same spectrum on a plane tilted 20° | 2e-3 | 5.87e-4 (bilinear spectrum, pad 4); 8.27e-14 untilted | C |
| `chirp_z_zoom_matches_the_direct_dft` | the 2D sum written out | 1e-11 | 4.57e-15 | C |
| `fraunhofer_zoom_lands_on_the_fft_fraunhofer_on_its_own_grid` | the FFT Fraunhofer step | 1e-11 | 4.31e-14 | C |
| `the_airy_first_zero_lands_at_1_21967_lambda_z_over_d` | `r₁ = j₁,₁λz/(πD)` | 1e-4 at D/Δx 128; ratio > 3 | 6.64e-5, 1.10e-5, 1.89e-6 at D/Δx 32, 64, 128; ratio 5.83 | A |
| `fft2_is_unitary_under_parseval`, `forward_then_inverse_returns_the_field` | Parseval; round trip | 1e-13 | pass | C |
| `the_centered_transform_of_a_shifted_gaussian_is_a_gaussian_with_the_shift_phase` | `F{e^{−π(x−a)²}} = e^{−πf²}e^{−i2πfa}`: the kernel's sign and centring | 1e-12 | passes | A |

**Bundle cases** [Bundle], in the canvas's units:

| Test | Held against | Gate | Measured | Cl. |
|---|---|---|---|---|
| `b02_finite_modes_are_carried_by_the_exact_diagonal_transport` | six periodic modes × 2 components, each times `e^{id√(k²−|q|²)}`, at 3, 7, 10 mm; semigroup; adjoint | 2e-11 | 6.7e-12 to 1.42e-11; semigroup 8.5e-12, 8.1e-12; back 5.0e-16, 5.8e-16 | A (C for the semigroup and adjoint) |
| `b02_the_fft_spectrum_is_the_explicit_finite_dft` | explicit DFT | 2e-11 | 6.21e-16 | C |
| `b10_an_aperture_acts_at_its_plane_and_the_order_matters` | `U_after M U_before` by FFT against the explicit DFT; the misplaced order against the bundle's frozen 0.43298345607270855 | 2e-11 | 2.41e-15, 2.55e-15; 6.85e-14 | C + frozen value |
| `b10_a_gaussian_through_a_circular_hole_passes_one_minus_e_to_the_minus_half` | `1 − e^{−2a²/w²}`, w 0.1, a 0.05 mm | 1e-4, and second order | −4.43e-5 at 2 µm, −1.08e-5 at 1 µm (of it, 4.2e-3 and 2.1e-3 booked beyond Nyquist) | A |
| `a_sinusoidal_phase_mask_puts_j_m_squared_into_its_orders` | Jacobi–Anger `e^{iβ cos Gx}` → `i^m J_m(β)`, β = 2, orders ±12 | 1e-10 power; 1e-7 rad | ≤ 2.0e-15; phase 1.2e-8 rad (docs; not printed) | A |
| `two_slits_make_the_far_field_of_m35` | `2a sinc(aq/2) cos(dq/2)` with the sampled mask's cell factor | 1e-9 of peak | 8.47e-16 | A |

**Unit tests in `oc-wave/src`** (24; none print values):

| Test (file) | Held against | Gate | Cl. |
|---|---|---|---|
| `a_gaussian_carries_the_power_it_was_given_and_has_its_radius` (beams) | P = 2.5; `I(w₀)/I₀ = e^{−2}` | 1e-10; 1e-12 | A |
| `the_q_parameter_carries_kogelnik_and_lis_gouy_phase` (beams) | Gouy phase π/4 at `z = z_R` | 1e-12 | A |
| `czt_line_is_the_direct_sum` (czt) | the 1D sum written out | 1e-12 | C |
| `disc_overlap_matches_the_disc_and_its_quadrants` (elements) | πr², πr²/4, a strip; a cell against 4000² midpoints | 1e-14; 2e-6 | A |
| `polygon_coverage_is_the_clipped_area_on_every_cell` (elements) | per-cell clipping, four polygons | 1e-12 | C |
| `a_disc_polygon_has_the_disc_s_area_and_hugs_its_circle` (elements) | the disc's area; departure `< r(π/n)²/2` | 1e-12 | A |
| `polygon_overlap_of_a_triangle` (elements) | areas 2, 1, 3, 0 | 1e-15 | A |
| `next_fast_len_has_only_small_prime_factors` (fft) | 7 → 8, 97 → 100, 1025 → 1080 | exact | A |
| `the_shifts_undo_each_other_on_odd_and_even_sizes` (fft) | `ifftshift ∘ fftshift = id` | exact | C |
| `power_is_the_irradiance_summed_over_the_cell_area` (field) | `Σ|U|²ΔA` | 1e-12 | A |
| `the_wavelength_crosses_into_millimetres_once` (field) | `k = 2πn/λ` | 1e-9 | A |
| `a_baked_carrier_is_a_phase_ramp` (field) | `e^{ikαx}` | 1e-12 | A |
| `bad_wavelengths_and_media_are_refused` (field) | refusals | — | — |
| `the_centre_is_a_sample_and_the_fft_frequencies_wrap` (grid) | layout; FFT frequency order | exact | A |
| `bad_grids_are_refused_by_name` (grid) | refusals, `TooLarge` | — | — |
| `a_bare_interface_is_a_unitary_two_port_with_the_textbook_numbers` (planar) | bundle B03: `r = −0.2`, `R + T = 1`, Brewster `r_p = 0`, Stokes `r′ = −r`, `t′ = t`, total internal reflection | 1e-14 | A |
| `the_slab_matches_the_closed_form_and_its_series` (planar) | `t = (−1)⁷` at 7π; the 7.5π transmission; the one-round-trip series | 1e-12 to 1e-16 | A |
| `the_ledger_closes_by_booking_the_gap_to_numerical` (report) | bookkeeping | 1e-14 | — |
| `a_report_round_trips_through_json` (report) | I/O | exact | — |
| `gauss_legendre_integrates_polynomials_exactly` (special) | `∫t³⁰ = 2/31` | 1e-15 | A |
| four `transfer.rs` tests | the table below | | C |

| Test (`transfer.rs`) | Held against | Gate | Measured | Cl. |
|---|---|---|---|---|
| `a_transfer_about_a_fixed_reference_is_linear_in_the_field` | `T(u₁+u₂) = T(u₁) + T(u₂)` for Gaussians tilted ±10 mrad | 1e-12 | 5.69e-17; with a sphere fitted to each field (before I6) 0.152 | C |
| `a_tilted_beam_through_a_plane_face_matches_the_exact_interface_when_traced_about_its_own_direction` | exact plane-wave interface: ASM 0.1 mm in air, `√(1−R)`, ASM 0.3 mm in glass. Traced about the chief ray instead, the piston `k₀θ²/2·(L_air + L_glass/n)` | 1e-3; piston 2% | 4.92e-4; chief-ray trace 0.149, piston 0.1486 rad against 0.1479 | C + A |
| `a_tilted_beam_crosses_a_curved_face_where_its_rays_do` | the centroid of the traced rays, weighted by the Gaussian (R 8 mm face, 0.3 mm off axis, 30 mrad) | 3 nm | 1.42 nm (the rays' centroid is 11.05 nm from the central ray) | C |
| `each_lobe_traced_about_its_own_direction_matches_it_traced_alone` | the same lobe about the chief ray | departure reported | 0.149 relative; departure 1.0e-2 against 4.7e-17 rad | C |

### 10.2 Modulators

| Test | Held against | Gate | Measured | Cl. |
|---|---|---|---|---|
| `dmd_orders_sit_on_the_harvey_vernold_grid` | `t_in + λ(m, n)/p`; Harvey's sum equals the reflected total | 1e-12 | exact; 634 orders propagate at 532 nm, 7.56 µm | A + C |
| `the_blazed_number_matches_popoff_figure_5` | Popoff et al. 2024 Eq. (2), Fig. 5: 0.06, 0.04, 0.97, 1.00 | ±0.01 | 0.059, 0.036, 0.969, 0.998 | A (published) |
| `a_blazed_dmd_concentrates_one_order_and_an_anti_blazed_one_splits_four` | Popoff §2, qualitative | shares | 10.8 µm: 85.3%, second 2.4%; 7.6 µm: 26.0, 17.6, 17.6, 13.8% | A (qualitative) |
| `a_flat_state_dmd_puts_the_fill_factor_squared_in_the_zero_order` | rectangular grating: `η₀₀ = ρ⁴`, `Σ = ρ²` | 1e-12 | exact | A |
| `the_factorized_far_field_lands_on_the_closed_form_orders` | `|Û|/(N²p²) = |c_mn|` | 1e-12 | 7.98e-17 | C |
| `the_sampled_dmd_far_field_converges_on_the_factorized_one` | the same model two ways, at 4–32 samples per mirror (per pixel pitch, `Δx = p/per`) | slope < −1.5; 2e-3 at 32 | 5.98e-2, 3.38e-3, 2.94e-3, **4.88e-3**, 9.39e-4, 8.38e-4 at 4, 8, 12, 16, 24, 32; slope −1.87 (not monotonic: 16 is worse than 12) | C |
| `a_binary_grating_on_the_on_mirrors_puts_one_over_pi_squared_in_each_first_order` | the lattice factor `1/(P sin(π/P))` for a binary grating of period `P` pixels (→ 1/π, so the first order's share → 1/π²), and `|1 − M̂_off/M̂_on|` | 1e-10 | the squared factor 0.1026, 0.1014, 0.1013 for P = 16, 64, 256 (1/π² = 0.1013); OFF leak 5.5e-4 → 4.5e-4 | A / C |
| `ti_dlpa083_average_photopic_diffraction_efficiency` | TI DLPA083B Table 6-1 (a vendor simulation); mirror fill = √(TI fill) | ±3 pp (one row ±5) | 13.68: −0.19; 10.8: −0.26; 9.0: −1.78; 7.56: −1.08; 5.4 f/1.7: +1.94; **5.4 f/2.4: +4.44**; 7.56 f/3→f/2.4: +0.02 pp | D |
| `the_lee_generator_follows_gutierrez_cuevas_and_popoff_eq_11` | Eq. (11) | exact count | passes | A |
| `an_n_level_blazed_grating_reaches_sinc_squared_one_over_n` | `sinc²(1/N)`: 0.40528, 0.81057, 0.94964, 0.98721 | 1e-12 | exact | A |
| `fill_factor_and_dead_space_set_the_zero_order` | `η₀₀ = |g|²(1−ρ²)²`, `η₁₀ = ρ⁴ sinc²(ρ/N)` | 1e-12 | exact (η₀₀ 2.128e-2) | A |
| `a_kinoform_at_the_wrong_wavelength_loses_efficiency_as_sinc_squared` | the quantized geometric series; `sinc²(1 − λ₀/λ)` in the limit | 1e-12; 1e-4 | exact; 0.9675 at λ₀/λ = 0.9 and 1.1 | A |
| `the_lcos_modulates_only_the_director_component` | `cos²(ψ − δ)` | 1e-12 | 0.750 at 30° | A |
| `the_sampled_lcos_lands_on_the_closed_form_orders` | the closed form, edges on sample boundaries | 1e-10 | 2.67e-15 | C |
| `fringing_lowers_blazed_efficiency_and_matches_quadrature` | direct quadrature of the same model | 1e-3 | η₁ 0.950, 0.906, 0.860, 0.766 at w = 0, ¼, ½, 1 px; agreement ≤ 4.8e-5 | C |
| `a_spiral_phase_leaves_a_dark_core_of_charge_one` | `e^{iℓφ}`: dark core, winding ℓ [Allen et al. 1992] | 1e-10; ℓ = 1 | 8.3e-33; 1 | A |
| `gerchberg_saxton_error_never_increases` | Fienup's error-reduction property | non-increasing | 2.00 → 0.096 in 60 iterations | A |
| `pommet_flag_is_raised_below_14_wavelengths` | the 14λ rule | exact | passes | A |
| `an_on_mirror_reflects_a_ray_through_twice_its_tilt` (`oc-trace`) | facet reflection; `R·T_w²` | 1e-12 | passes | A |
| `a_dmd_in_the_ray_limit_absorbs_its_gaps` (`oc-trace`) | `ρ² cos θ` of the array | 2e-3 | 0.792000 against 0.792300 | A |
| `an_slm_steers_the_director_component_by_its_phase_gradient` (`oc-trace`) | grating equation `λ/Λ`; 1:1 split | 1e-12 | passes | A |
| `dmd_orders_traced_match_the_unit_cell_order_for_order` (`oc-trace`) | the closed-form table | 1e-12 | passes | C |
| `untraced_orders_are_in_the_ledger` (`oc-trace`) | the table's remainder; dead space `1 − total` | 1e-12 | 0.053257 = 0.053257; residual 1.3e-15 | C |
| `ti_dlpa083_through_the_tracer_matches_direct_integration` (`oc-trace`) | the canvas's own `dmd_projector_efficiency` (not TI's numbers) | 5e-3 | 0.882030 against 0.882032 | C |
| `an_lcos_blazed_grating_sends_the_director_half_into_its_first_order` (`oc-trace`) | `sinc²(1/8)`, Harvey-renormalized, half the unpolarized light; `sin θ = λ/(8p)` | 1e-12 | 0.475794068 = 0.475794068 | A (+C) |
| `test_blazed_staircase_first_order` (Python, ×4) | `sinc²(1/N)` | 1e-12 | (docs) passes | A |
| `test_dmd_projector_efficiency_is_near_ti_at_550_nm` (Python) | TI 84% at 550 nm | **0.08** (the docstring says 3 pp) | (docs) passes | D |
| `test_gerchberg_saxton_error_never_rises` (Python) | non-increasing | 1e-12 | (docs) passes | A |

### 10.3 The wave path (`oc-wave/tests/path.rs`)

| Test | Held against | Gate | Measured | Cl. |
|---|---|---|---|---|
| `a_gaussian_through_empty_space_spreads_as_kogelnik_and_li_say` | Kogelnik–Li after 300 mm of scene | 1e-5 | radius 3.53e-7; field 2.33e-7 of peak | A |
| `a_thin_lens_in_the_scene_focuses_like_self` | Self's waist | 1e-4 | 3.57e-8 | A |
| `a_fold_mirror_hands_on_the_mirror_image_exactly` | the same path unfolded | 1e-9 | 6.08e-11 | C |
| `an_aperture_on_the_path_is_found_although_the_trace_logs_no_event` | `e^{−2a²/w²}` blocked | 1e-4 | 3.17e-5 | A |
| `a_4f_system_images_at_minus_f2_over_f1_and_its_iris_cuts_the_spectrum` | image waist `w₀f₂/f₁`; Fourier-plane iris `e^{−2r²/w_F²}` | 1e-4; 2e-3; 1e-6 | 1.000; 1.74e-5; ledger numerical 1.13e-13 | A |
| `wave_and_ray_agree_on_where_the_beam_lands` | the chief ray on a tilted fold | 0.05 pixel | 3.2e-4 mm | C |
| `a_dmd_in_the_scene_focuses_its_orders_f_lambda_over_p_apart` | orders `fλ/p` apart at a lens's focus [Harvey & Vernold 1998] | 1% | 1.000; detected 72.1%; resampling 1.3e-4; numerical 3.6e-13 | A |
| `slicing_a_wave_job_changes_no_bit` | the same job in one step | bit-identical | identical | C |
| `wave_power_agrees_with_rays_when_the_fresnel_number_is_large` | the f64 ray trace of the same scene, `N_F` 1264 | 1% | 0.124% (wave against a²/R²: 0.132%; rays: 0.0074%) | C |
| `a_source_carries_its_coherent_beam_through_the_file_and_into_its_wave_path` | file round trip | exact | passes | — |
| `a_continuous_source_is_refused_and_lines_add_in_power` | the lines' own runs, weighted | 1e-12 | 1.81e-15 | C |
| `a_mach_zehnder_sends_all_the_power_out_of_its_two_ports_whatever_the_path_difference` | unitarity: `(1 ∓ cos Δφ)/2` at Δφ = 0, π/2, π, 3π/2 | sum 1e-3; share 2e-3 | sum 1.000; share ≤ 1.001e-7 | A |
| `a_michelson_with_a_tilted_mirror_makes_fringes_of_period_lambda_over_sin_2_alpha` | fringe frequency `sin 2α/λ`, α = 0.5 mrad | 2e-3 | −7.09e-5 | A |
| `moving_a_michelson_mirror_by_d_shifts_its_fringes_by_two_k_d` | λ/8 steps are π/2: `P₀ + P₂ = P₁ + P₃`; visibility 1; equal arms dark | 1e-3; 0.995 | −2.81e-7; 1.000 | A |
| `two_lines_wash_a_michelson_s_fringes_out_where_their_beats_cancel` | visibility `|cos(2πΔ(1/λ₁ − 1/λ₂))|`, with `Δ` the arm-length difference (the path difference is `2Δ`) and λ₂ = 1.001 λ₁ | 0.99; 2e-2 | 0.9996; 1.57e-3 | A |
| `a_gaussian_through_a_singlet_waists_where_self_predicts` | Self, with f and the principal planes traced by the canvas tracer (N-BK7, R ±50, 4 mm) | position 1e-3 of s″; radius 1e-3 | position −2.51e-4 of the Rayleigh range (the gate is on s″); radius −5.10e-5; faces reflect 8.2%; resampling −2.0e-7 | A |
| `a_glass_plate_shifts_a_gaussian_as_free_space_of_t_over_n` | a plate of thickness t is free space t/n to a Gaussian | 1e-5 | 1.90e-7 (implied index −1.78e-5) | A |
| `a_tilted_plate_displaces_the_beam_where_the_rays_say` | the chief ray's landing behind a 30° plate | 0.05 pixel | 5.22e-5 mm | A + C |
| `a_singlet_psf_matches_optiland_huygens_psf` | Optiland `HuygensPSF`, f/10 singlet, paraxial focus, 65² | Strehl ±0.01; rms 1e-2 of ideal peak | Strehl 0.86532 against 0.86425; rms 1.22e-4; grid sums 1.0008 | **B** |
| `a_tilted_detector_reads_the_beam_on_its_own_plane` | Kogelnik–Li on a face tilted 30°, times the obliquity | 2e-3 of peak; 2e-3 | 8.52e-4; −1.95e-3 | A |
| `an_aperture_whose_outer_extent_the_beam_passes_beyond_leaves_it_alone` | the same plane, open | bit-identical | identical | C |
| `a_fold_mirror_smaller_than_the_beam_reflects_only_what_meets_it` | Gaussian power off a 0.6 mm disc's projected ellipse at 45°, by quadrature | 1e-4 | 1.52e-5 | A |
| `a_thin_lens_smaller_than_the_beam_lets_the_rest_pass_beside_it` | 10 µm behind a 0.5 mm lens: its phase inside, free space outside | 2e-3 | 1.03e-3 inside; 2.96e-5 outside | A / C |
| `a_lens_rim_smaller_than_the_beam_stops_and_diffracts_as_an_iris_against_it` | a 1 mm iris just before a lens too large to clip | power 3e-4; spot 2e-3 of peak | 1.09e-4; 3.71e-4 | C |
| `an_lcos_on_the_path_sends_half_of_unpolarized_light_into_each_beam` | half the device's share unmodulated; `sinc²(1/16)` of the modulated half in the first order | 2e-3; 3e-3 | −5.88e-7; −2.06e-3 | A |
| `a_glass_plate_delays_the_beam_by_its_optical_path` | `k₀(n − 1)t` less the Gouy change of the effective length | 1e-4 rad | −8.89e-9 rad | A |
| `an_lcos_whose_two_polarization_beams_both_reach_the_detector_adds_them_in_power` | independent of a uniform phase on one beam; the device meets `erf(√2a/w)²` | 1e-9; 1e-3 | 0; −2.68e-4 | A / C |
| `a_thin_plate_is_an_etalon_transparent_at_seven_pi_and_dimmest_at_seven_and_a_half` | the closed-form slab at δ = 7π and 7.5π; the field `−e^{−ik₀L}` | 1e-6 | −6.07e-14; 4.04e-14; field 3.30e-7 | A |
| `an_oblique_etalon_transmits_s_and_p_as_their_own_closed_forms` | `(T_s + T_p)/2` at 31°, δ = 7.5π | 1e-5 | 1.33e-7 | A |
| `a_millimetre_plate_s_transmission_follows_its_round_trip_phase` | the closed form at a single-pass phase δ = k₀nL ≈ 15 000 rad (the name's "round-trip phase" is that δ) | 1e-4 | −2.27e-5 (the beam's own angular spread) | A |
| `a_coated_surface_reflects_and_transmits_as_its_single_layer_s_closed_form` | quarter-wave MgF₂ on crown, single-layer formula written apart: `(R_s+R_p)/2` at 45°; `(1−R_f)(1−R_b)` | 1e-6 | 1.63e-8; −3.11e-11 | A |
| `coatings_whose_planes_of_incidence_differ_reflect_s_of_one_as_p_of_the_other` | `(R₁ₛR₂ₚ + R₁ₚR₂ₛ)/2` for crossed planes | 1e-5 relative | −1.11e-6 | A |
| `a_pinhole_at_a_lens_s_focus_passes_what_the_focused_gaussian_puts_through_it` | a pinhole of radius `λf/(πw₀)` stops `e^{−2}` | 1e-5 | 5.68e-7 | A |
| `a_ten_times_beam_expander_widens_the_beam_as_the_abcd_law_says` | Gaussian ABCD law through a 50 µm focus | 1e-5 | −4.22e-7 | A |
| `two_polarizers_pass_unpolarized_light_as_malus_s_law_says` | `cos²θ/2`; crossed sheets pass ε | 1e-9; 1e-6 | ≤ 9.8e-15; −2.0e-14 | A |
| `a_retarder_between_crossed_polarizers_passes_half_sin2_2phi_sin2_half_delta` | `sin²2φ sin²(δ/2)/2`; circular light passes 1/4 | 1e-9 | ≤ 9.6e-15; ≤ 4.8e-15 | A |
| `tilted_crossed_polarizers_leak_as_the_ray_tracer_says` | the f64 tracer's polarized chief ray; bare `(sin²θ/(1+cos²θ))²/2` | 1e-6; 1e-9 | −1.8e-14, −3.5e-14; −1.9e-14 | C + A |
| `a_half_wave_plate_in_one_arm_of_a_mach_zehnder_takes_its_fringes_away_and_a_polarizer_brings_them_back` | Fresnel–Arago: ½ at every Δφ; behind an analyzer at β, `¼ − cos Δφ cos 2(θ−β)/4`. For unpolarized light, `θ` = 20° is the half-wave plate's fast axis (the average over input polarizations of `cos(ψ − β)cos(2θ − ψ − β)` is `½cos 2(θ − β)`), and β ∈ {20°, 50°, 65°} is the analyzer | 1e-6 | ≤ 4.92e-8 | A |
| `crossed_polarizers_before_a_lens_leave_it_a_dark_field_to_carry` | an all-zero field, the ledger closed | 1e-12 | passes | A |
| `a_dmd_s_orders_cross_a_singlet_each_about_its_own_direction` | each order about its own reference (nearest order 27 mrad from the chief ray) | rms 2.7 mrad; numerical 1e-6 | rms 0.36 mrad; numerical 5.87e-9; resampling 9.94e-3 | C |

### 10.4 External oracles (Python; not rerun here)

| Test | Held against | Gate | Measured (docs) | Cl. |
|---|---|---|---|---|
| `test_fraunhofer_matches_optiland_scalar_fft_psf_on_its_own_pupil` | Optiland `ScalarFFTPSF` on its own aberrated Cooke-triplet pupil | 1e-10 of peak | 5.4e-16 | B |
| `test_zoomed_psf_matches_optiland_mmdft` | Optiland `MMDFTPSF`, pad 373.7 | 1e-10 | 3.7e-15 | B |

These show that the two engines index, centre and scale the transform
alike. They are verification of code, not evidence about a lens.

The singlet PSF test (§10.3) has more to it:
- It is held to Optiland's `HuygensPSF` of the same lens. Both engines trace
  the same geometric approximation, so this too is verification.
- The canvas's PSF is normalized by the peak of a perfect spherical wave
  filling the exit pupil, `P·π sin²α/λ²`. The angle comes from a traced exit
  pupil, `sin α = 0.049928`.
- A uniform spherical cap actually peaks at `4π sin²(α/2)/λ²`, 0.06% above
  that at f/10.
- The oracle's Strehl ratio is the largest sample of its image grid. An even
  grid has no sample on the axis, so the fixture uses 65 points.

### 10.5 Cross-checks from the report (computed for the user's report, not tests)

These came from running the app's own scene files natively, against
references computed apart.

| System | Reference | Result |
|---|---|---|
| iris and thin lens (Airy) | `2J₁(v)/v`, with `P_iris = (a/R)²` of a top-hat | max difference 4.3e-4 of the peak; encircled energy within 3.4e-5 over six rings |
| f/10 singlet | Optiland `HuygensPSF` (as §10.3) | Strehl 0.8653 against 0.8642; rms 1.2e-4 |
| DMD and lens, orders at the focus (10 µm pitch, field pitch 2.5 µm, so 8 device samples per pixel by the rule of §7.7) | the factorized far field (§6.4) | the brightest two orders −1.0% and −1.7%; orders 90–120 mrad from the beam −12 to −13%; beyond 130 mrad −57 to −66%. At 16 and 24 device samples per pixel the two brightest are within 0.33% |
| LCoS blazed gratings, N = 2–16 levels, 8 µm pixels | `sinc²(1/N)`, Harvey-renormalized, Orders ray model | first-order shortfall at a 5 µm field pitch 0.3–7.2%; at 2.5 µm 0.1–2.7%; at 1.25 µm below about 6e-3 for N ≥ 3 and 7.8e-3 for N = 2 |
| Michelson, tilt sweep | `sin 2α/λ` | worst 5.6e-4 over the sweep (a 12 mm detector); mirror step 1.8e-5; visibility 1.6e-3 |
| Mach–Zehnder, half-wave plate | Jones calculus | ≤ 1.6e-6; port sum within 2.9e-9 |
| polarizers, retarder | Malus; `sin²2φ sin²(δ/2)/2` | ≤ 1e-14 |
| a 1 mm glass plate as an etalon, thickness swept by 0.45 µm (w₀ 0.5 mm) | at 0°: the closed form. At 31°: the closed form **averaged over the beam's own plane waves** | 2.3e-11 at 0°; 1.3e-6 at 31°. The Gaussian's fringe contrast at 31° is 0.073, against 0.158 for one plane wave. This differs from the oblique-etalon test of §10.3 (1.33e-7 against the plane-wave form) because that test's plate is 1.7 µm thick: its single-pass phase δ is 7.5π (24 rad), against about 15 000 rad for the 1 mm plate, so the beam's angular spread barely changes it |
| pinhole spatial filter | an exact non-paraxial reference: the Hankel transform of `exp(−βr²)`, `β = 1/w₀² + ik/2f`, each plane wave advanced by `e^{iz(k_z − k)}` | 2048 across: within 4.6e-7 of the exact transmission; 1024 across: up to 6.4e-5 (the app's 1024 run raises `Undersampled`). The paraxial and exact references differ by up to 6.3e-5 |
| DMD and singlet | each order about its own reference | rms departure 0.36 and 0.093 mrad at the two faces; the nearest order 27 mrad from the chief ray |

---

## 11. Known defects, limitations and open findings

### 11.1 Confirmed defects

Three defects were found while this brief was written. Each is confirmed by
a scratch program outside the repository, and none is fixed.

#### 11.1.1 A flat grating on the wave path loses its order's momentum

**What happens.** At a flat `Grating` met with `SeqAction::Diffract`, the
hand-over (§7.3) uses `F = √η`, a real constant. The hand-over then
multiplies by the reference change `exp(ik(d − d′)·(r_e − p))`.

For a grating, `d′ − d` has a tangential part on the grating plane: the
grating vector, `k(d′ − d)_t = mK`. The true element factor on the plane is
`√η·exp(i mK·(r_e − p))`, which the reference change should cancel. With
`F = √η` the samples keep `exp(−i mK·(r_e − p))`. So the field leaves along
the **incident** direction while its frame turns to the diffracted one.

The thin lens is handled correctly, because its `F` carries the lens's full
phase. A mirror and a straight-through arm have a zero tangential change, and
the DMD and LCoS take-up (§7.7) handles the change through the carrier. The
grating is the only element with the inconsistency.

**Reproduction** (a scratch program outside the repository, Appendix C):
- a transmission grating at x = 50 mm (`add_grating`'s defaults: the +1
  order at η = 0.65 and the zero order at 0.10), its zero order set to 0 so
  that only the +1 order is on the chain;
- a Gaussian of w₀ 0.3 mm at 632.8 nm;
- a detector square to the diffracted chief ray, 100 mm on, 8 × 8 mm
  (half-widths 4 mm) in 80 × 80 pixels of 0.1 mm;
- the power centroid computed on the detector's own pixels.

| Grating | Zero order η₀ | Readout | Power on the face | Centroid on the face (mm) | Where the zero order lands (mm) |
|---|---|---|---|---|---|
| 20 l/mm | 0 | Projected | 0.6500 | (+0.016, **−1.266**) | −1.266 |
| 20 l/mm | 0 | Diffracted | 0.6500 | (+0.025, −1.263) | −1.266 |
| 20 l/mm | 0.10 | Projected | **1.2595** | (+0.016, −1.266) | −1.266 |
| 100 l/mm | 0 | Projected | 0.6500 | (+0.060, −0.939) | −6.341 |
| 100 l/mm | 0 | Diffracted | **0.6546** | (+0.067, −0.936) | −6.341 |

**Reading the rows:**
- **20 l/mm.** The order lands where the zero order would: it has kept the
  incident direction.
- **100 l/mm.** The leftover ramp `exp(−ik tan θ·a)` turns 7.4 rad per
  sample of the 11.7 µm grid. It is aliased, and appears as a tilt of about
  −9.4 mrad, so the beam lands 0.94 mm off its chief ray. No warning is
  raised, because the hand-over does not check the phase step of what it
  puts on the samples.
- **100 l/mm, diffracted readout.** The face reads 0.6546 against the 0.65
  the order carries. The ledger then closes with `numerical ≈ −4.6e-3`, a
  gain of 0.7% that nothing flags. The cause is not established. It is
  probably the diffracted readout's point sampling (§11.1.3) of the aliased
  field, at 2 points per 0.1 mm pixel.
- **Both orders reach the detector** (`η₀ = 0.10`). They run as two arms and
  land on the same spot in phase. The face reads 1.2595 of the launched 1.0,
  against `(√0.10 + √0.65)² = 1.2599` for complete overlap. The network
  ledger closes with `other_ports = −1.5095`, which nothing checks (§7.11).

**A second, independent ledger defect** shows in the same run. The network
ledger counts an event that several arms share only once if the element is
registered as a splitter (`ExplicitRt`, `Coating`) or an LCoS. A grating is
not, and its two arms' hit keys differ (`Transmit` against `Diffract`). So
each arm books the other's order as untracked:
`untracked_orders = (1 − 0.10) + (1 − 0.65) = 1.25` instead of `0.25`. With
the momentum fixed and the orders landing apart, `other_ports` would still
close at about −1.0.

**Why the tests missed it:**
- no wave-path test puts a flat grating on the chain;
- `other_ports`, the network remainder, has no sign check;
- the hand-over's phase is not checked for aliasing.

**The fixes the author proposes** (not applied):
- For a diffracted order, set
  `F(r) = √η·exp(ik(d′ − d)·(r − p))`, or skip the reference-change factor
  for that element.
- Register a grating, like a splitter, as an element that several arms
  meet.
- Warn when `other_ports < −1e-9·launched`.
- Warn when a hand-over's phase step per sample exceeds π.
- Add tests that the order lands on its chief ray, and that a grating whose
  two orders both reach the detector conserves power.

#### 11.1.2 The zoomed and tilted angular spectra ignore an off-centre input window

**What happens.** `asm_zoom` (§4.6) and `asm_tilted` (§4.6) take the input's
spectrum with `embed`, `ifftshift2` and an FFT. That is the spectrum about the
input window's **centre** `c`, `Â_c(f) = Σ u(x) e^{−i2πf(x − c)}ΔA`. They
then evaluate the plane-wave sum at **absolute** output coordinates, measured
from the frame origin, with no `e^{−i2πf·c}` correction. The output at `x`
therefore holds the field at `x + c`: the result is displaced by `−c`.

`asm_zoom`'s comment "the transform is about the frame origin" holds only
for `c = 0`. The Fraunhofer step makes the correction (§4.4, "about the frame
axis it carries the shift phase"). `fresnel_zoom` is not affected, because
`czt2` uses absolute input coordinates.

**Reproduction** (Appendix C.2).
- A Gaussian of w₀ 50 µm at 632.8 nm, centred at `x = 0.3` mm, sampled on
  256² at 4 µm.
- It is propagated 5 mm onto a 128² window centred at `x = 0.3` mm, and
  compared with Kogelnik–Li at the output points.

| Input window centre | `asm_zoom` centroid | `asm_tilted` centroid (untilted target) | worst error / peak, both | plain ASM centroid |
|---|---|---|---|---|
| 0 | +0.30000 | +0.30000 | 2.8e-6, 2.0e-6 | +0.30000 |
| 0.3 mm | +0.05367 (the beam pushed toward the window's edge) | +0.05367 | **1.000**, **1.000** | +0.30000 |

**Where the wave path meets it.** A re-grid (§7.2) centres the new window on
the beam's predicted centroid, so after it `c ≠ 0` in general. Every later
`asm_zoom` or `asm_tilted` on that field is then displaced:
- a second re-grid;
- the diffracted readout;
- each arm's readout onto the detector grid.

Three other grids are laid out about the frame origin rather than the
window's centre, and would cover the wrong area of an off-centre field:
- a transfer's ray grid, which would book the samples it misses as "stopped
  by a mount";
- a plate's 65 × 65 rim probe;
- a modulator's device footprint (`±half` about the origin).

**Why it has not shown.** In the tested paths the beam stays on the chief
ray, so the windows' centres stay near 0. In the DMD-and-singlet scene, whose
orders lie 27–36 mrad off the chief ray, the last window's centre is 2.5 µm,
half a sample. No test moves an input window off the origin:
`an_off_axis_window_holds_the_field_the_large_window_does` moves only the
output.

**The fix the author proposes** (not applied):
- multiply the spectrum by `e^{−i2πf·c}` in `asm_zoom` and `asm_tilted`;
- lay the ray, probe and device grids about the window's centre;
- add a test with an input window off the origin.

#### 11.1.3 The diffracted readout's point sampling can misread the power, unflagged

**What happens.** `Readout::Diffracted { samples_per_pixel: s }` takes each
pixel's power as the **mean irradiance of its `s × s` points** times the
pixel's area (§7.13). Where the irradiance varies within a pixel on scales
finer than the point spacing, that mean is not the pixel's power. Examples
are fringes between orders, and speckle. The only warning is for a pixel
with no point at all.

**Reproduction** (Appendix C.2). The DMD-and-singlet scene (§10.3), read
with the diffracted readout. The pixels are 0.1875 mm; the orders' fringes are about
10 µm apart. The power arriving at the detector's plane is 0.724 of launched
(Projected readout: 0.72392).

| Points per pixel | Detected | `left_window` | `numerical` |
|---|---|---|---|
| 2 | **0.951** | 0.160 | **−0.387** |
| 8 | 0.722 | 3.5e-3 | −1.3e-3 |
| 32 | 0.722 | 1.7e-3 | 5.9e-9 |

At 2 points per pixel the detector reads 31% more power than arrives, and
the ledger hides the gain in `numerical` without a warning.

**What would help:**
- a warning when `|numerical|` exceeds a small fraction of the launched power
  on any path;
- a check of the readout grid's spacing against the arriving field's
  spectral width;
- or pixels integrated by the spectrum rather than point-sampled.

### 11.2 Approximations the code makes without naming them

Found while this brief was written:

1. **A grating's TM efficiencies are ignored on the wave path.** The declared
   TE `efficiency` and `zero_order` are used for any polarization. A grating
   does not count as an element that tells polarizations apart.
2. **The reference `q` misses some elements.** It is not changed by a
   grating's or a modulator's anamorphic magnification
   (`w_out/w_in = cos θ_out/cos θ_in` in the plane of incidence). It is also
   not changed by an LCoS pattern's lens power, such as a Fresnel-lens
   pattern. A later face then traces along a wrong reference, and its
   reported departure is the only sign of it.
3. **`asm_tilted`'s ledger simplifications** (§4.6) are not named in the
   report:
   - backward and evanescent waves and out-of-band reads go to
     `left_window`;
   - one obliquity, the carrier's, is used for the power.
4. **The hand-overs and modulator take-ups are not exactly linear.** Their
   interpolation sphere is fitted to the field (§7.14). Only the transfer is
   tested for linearity.
5. **`combine_arms` weights cross terms by `√(cos θ₁ cos θ₂)`.** The normal
   Poynting flux of two plane waves would give `(cos θ₁ + cos θ₂)/2` (§13,
   Q15). It also books each arm's readout losses into `other_ports`.
6. **The `Auto` rule's error estimates use a slightly different grid.**
   `regime_errors` evaluates the band limit for a padded width of exactly
   `2n`, while the step pads to `next_fast_len(2n)`. `z_c` uses `2nΔx` on
   axis 0 only.
7. **The Fraunhofer output is centred at `z·(α₀, β₀)`**, with direction
   cosines, not tangents: a paraxial centring.
8. **An oblique plate or coating on a one-field chain averages
   amplitudes.** It takes `(t_s + t_p)/2` in amplitude, where unpolarized
   light needs a power mean. The report names the averaging when it happens,
   but not as an error.

   This is reachable only where the obliquity tests disagree:
   - `1 − |d·n| < 1e-9` in `polarization_matters`;
   - `1e-12` in `plate_event` and in the coating branch;
   - `|d × n| < 1e-9` in `element_basis`.

   At such small angles `t_s ≈ t_p`, so the effect is negligible, but the
   rule itself is wrong. The deficit `|t_s − t_p|²/4` is then booked as
   reflected power.
9. **The thin lens's ray law is in tangents and its wave phase in direction
   cosines.** The difference, of third order in `h/f`, is left as a residual
   tilt (§9.2).
10. **An oblique thin lens is astigmatic, and three parts of the code
    disagree about how.**
    - The wave's lens phase is taken from the distance within the lens's
      plane, so across an oblique beam it is `−k(a²/cos²θ + b²)/2f`: focal
      lengths `f cos²θ` tangentially and `f` sagittally.
    - The reference `q` applies `1/f` on both axes.
    - The tracer's slope law in lens-local tangents gives a third pair.

    The path's undersampling check for the lens also ignores the `1/cos θ`
    stretch.
11. **A tilted plate's reference distance is the same on both axes.** The
    reference `q` adds `L_g/n_g` on both axes. Tangentially the paraxial
    distance a tilted plate is worth is `(L_g/n_g)·cos²θ₀/cos²θ_g`. Only the
    reference is affected; the plate's field step (§7.8) is exact.
12. **Two-component arms are added in frames that may differ.** Each arm's
    `x` and `y` are its own beam frame's axes, carried by the shortest
    rotation at every turn. For arms that do not stay in one plane, those
    axes can differ on the detector. The fields are added component by
    component, with no projection onto a common basis.
13. **Traced faces give both polarization components the same `√T`**, the
    tracer's unpolarized transmittance (§9.3), even on a chain that carries
    two components.

### 11.3 Documentation discrepancies found by the rerun

The repository's docs (`docs/wave-optics.md`, `docs/dmd.md`, `docs/slm.md`,
ADR 0003) differ from the code in these places. The brief uses the code and
the rerun.

- **Measured values that changed**, all still inside their gates:

  | Test | Docs | Now |
  |---|---|---|
  | fold mirror | 2.0e-11 | 6.08e-11 |
  | thin lens in the scene (the docs give the analytic test's value) | 8.1e-10 | 3.57e-8 |
  | 4f ledger `numerical` | 8.7e-14 | 1.13e-13 |
  | 10× expander | 3.9e-7 | 4.22e-7 |
  | lens-rim spot | 3.5e-4 | 3.71e-4 |
- **A false claim.** The docs say "`numerical`, round-off, is at most 7e-12
  of the launched power on every path the tests run". The DMD-and-singlet
  test gives 5.87e-9, against its gate of 1e-6.
- **ADR 0003 is not amended.** Item 5 ("fields are scalar … Jones fields are
  later work") is superseded by the two-component fields. Item 6 (the ledger
  "must close to 1e-12") is superseded by the finding that the closure is
  tautological (§8).
- **The window wording.** The comments and step notes say the window holds
  "six rms radii across"; the code sets the half-width to `6σ` (§7.2).
- **The projected readout's limit.** Its "sub-cells of at most an eighth of
  a pixel" holds only while a field cell spans at most two pixels (`s` is
  capped at 16).
- **The singlet waist test** prints `dz/z_R″` under the label "/ Rayleigh
  range", but its gate is on `|dz/s″|`.
- **The Python TI test**: the docstring says 3 percentage points, and the
  code allows 8.
- **A test name.** The `oc-trace` test named `ti_dlpa083_…` compares with the
  canvas's own closed form (class C), not with TI's numbers.
- **A gate quoted too tightly.** The Mach–Zehnder share is documented
  "within 1e-7". The largest is 1.001e-7, against a gate of 2e-3.
- **Stale doc comments in the code** (quoted in this brief, and annotated
  where they appear):
  - `polarization_matters` (§7.10) describes a superseded design that
    refused differing planes of incidence;
  - the `sequence3.rs` header (§9.1) describes the single-chain extraction;
  - `lib.rs` says the µm-to-mm factor "appears once" (§3);
  - `regrid_plan` ties the zoomed Fresnel re-grid to widening only (§7.2).

### 11.4 Open findings from the user's report (not fixed)

- **DMD device sampling at large order angles.** At 8 device samples per
  pixel (the rule's result for a 2.5 µm field pitch and 10 µm mirrors),
  orders 90–120 mrad from the beam carry 12–13% less power than the
  factorized table, and orders beyond 130 mrad 57–66% less. At 16–24 samples
  per pixel the brightest orders are within 0.33%. The sampled-against-
  factorized test is not monotonic in the sampling (16 samples per mirror is
  worse than 12). The app's rule, `per = max(4, 2, min(32, ⌈2p/Δx⌉))`, does
  not raise the sampling for wide-angle orders.
- **The browser's cap of 2²² samples.** It refuses a DMD on a fine field. The
  DMD-and-lens scene at 1024² over 2.56 mm needs 4.6 M device samples. The
  report's app screenshots used 512².
- **A two-level LCoS grating needs the traced-orders ray model.** With the
  geometric ray model, the chief ray of a two-level grating does not reach a
  detector on the first order (§9.2).

### 11.5 Limits by design

- **Scalar diffraction.** Each polarization component diffracts as a scalar
  field. There is no vector diffraction, no high-NA focusing, and no
  longitudinal component. Scalar theory's error exceeds ±5% for features
  below about 14λ [Pommet, Moharam & Grann 1994], and that is flagged for
  modulators only.
- **Oblique elements by projection.** Lenses, apertures and modulators are
  projected. Only the detector readouts use the tilted-plane spectrum: the
  diffracted readout, and every arm's readout in an interferometer.
- **Lenses face by face, in the geometric limit.** There is no diffraction
  across a face's sag. Rays follow the beam's paraxial wavefront, and the
  departure is reported.
- **The chief ray's chain.** The beam follows the chief ray. Steering onto a
  chosen order is not done. Arms that recombine are added once. A cavity of
  two separate mirrors is not followed round; round trips are summed only
  inside a plane-parallel plate.
- **One Jones matrix per element**, at the chief ray's angle.
- **Thin-element modulators.**
  - DMD: no shading, edges, hinges, curvature or thermal drift.
  - LCoS: no liquid-crystal dynamics, no dispersion of the birefringence,
    no dependence of the phase on the angle of incidence, and an unmodulated
    dead space.
- **Not built** (bundle items; [Bundle], Appendix E):
  - full vector fields at every sample, with the bundle's decoder from power
    coordinates to physical E and H (I5 stage b);
  - Gaussian frames and beamlets (section M11, benchmarks B21, B22);
  - a Mie reference for a dielectric sphere (section M17, B30), and the study
    of the canvas's curved-surface approximation against it (B40). These
    would be the only true curved-surface Maxwell validation of the lens
    transfer. The project has agreed to do them next;
  - a separate boundary-element solver (B31), which is not proposed;
  - the bundle's separate Python reference implementation of its milestone
    M1.
- **No measured datasets** anywhere in the suite.

---

## 12. Where the author is least sure

Each item gives the concern and the reason for it. Several concerns appear in
more than one place:

| Concern | §11.2 item | §12 item | §13 question |
|---|---|---|---|
| hand-over by projection | — | 1 | 2 |
| `f − f²` as `beyond_nyquist` | — | 2 | 3 |
| DMD interception and phase | — | 3 | 4 |
| renormalization of orders | — | 4 | 5 |
| LCoS polarization rules | — | 5 | 14 |
| reference `q` (top-hat; missing elements; astigmatism) | 2, 10, 11 | 6, 13 | 10 |
| orders carried apart | — | 7 | 23 |
| device sampling and the take-up's band cut | — | — | 17, 18 |
| window rule (and off-centre windows, §11.1.2) | — | 8 | 16 |
| slab depth | — | 9 | 9 |
| tilted-plane readout | 3 | 10 | 8 |
| arm weighting and arm frames | 5, 12 | — | 15, 24 |
| one field for unpolarized light | 8 | 12 | — |

1. **The hand-over at oblique non-mirror elements.** The field is projected
   along the rays, not diffracted across the plane's depth. `TiltProjection`
   fires when the depth exceeds `Δx²/λ`, the Fresnel length of one sample.
   Whether that threshold bounds the error is not shown.
2. **`f − f²` as `beyond_nyquist`.** The cell average is not an ideal
   low-pass filter. Part of the `m − f` remainder lies inside the band, and
   part of the cell average lies outside it. The attribution is plausible in
   total, not exact.
3. **The DMD's thin-element interception at oblique incidence.**
   - The mirror outline is projected along the array normal, so the
     reflected fraction is `ρ² cos θ`, whatever the incidence.
   - A tilted facet in isolation intercepts `ρ²|d_in·n|/|d_in·ẑ|` of the
     power arriving per pixel cell.
   - In TI's geometry (24° incidence onto 12° facets) that is
     `ρ² cos 12°/cos 24°`, a factor `1/cos 24° ≈ 1.095` above the thin
     element.
   - Shading by neighbours, which is not modelled, works the other way.
   - The ray tracer uses the same projected outline, so the two engines
     agree with each other. The projector efficiency is a share of the
     reflected power, so it is unaffected. The absolute reflected power is.
4. **The modulators' renormalization.** The DMD's sampled near field, which
   is what the wave path uses, has no Harvey renormalization: power in orders
   that become evanescent is lost. The ray tracer's order tables are
   renormalized. The two differ for orders near grazing.
5. **The LCoS polarization rules.** The device's absorbed fraction, measured
   on one component, is applied to the other ("the same share"). The
   outgoing polarization direction is the device's image of the kept
   direction. Both are reasonable for an ideal phase-only device, and neither
   is derived from a model of the liquid crystal.
6. **The top-hat's reference.** Its `q` is the Gaussian of its radius. After
   an aperture or for a small top-hat, the true wavefront curvature differs.
   The transfers then trace along a reference the field departs from, and
   the departure is reported, not removed.
7. **The orders carried apart.** At most 9 orders, each of at least 1e-4,
   with weaker orders joining their nearest. A joined order is traced about
   its neighbour's direction, `λ/p` away, and its departure is then of the
   order of `λ/p`. That is not separately reported per order.
8. **The second-moment window rule.** The rule is paraxial. Its cross
   moment comes from phase gradients, which fail near phase singularities (a
   vortex beam) and between separated lobes. A window chosen too small cuts
   the beam, and that power is booked as it leaves, not prevented.
9. **The slab of a traced face.** Its planes sit 1 µm from the surface where
   the window's rays meet it. Inside the slab there is no diffraction. A
   criterion for when the sag across the window is too deep for this is not
   implemented.
10. **The tilted-plane readout.** The bilinear reading of the spectrum gives
    5.9e-4 at 20° with 4× padding. How it grows with tilt and with the
    field's spectral width is untested.
11. **The Michelson fringe frequency.** The error is −7.1e-5. The beams'
    curvature accounts for −3.5e-5 by the docs' estimate; the rest is not
    explained.
12. **One field for unpolarized light.** A chain with no polarizing element
    gives faces the unpolarized power transmittance, applied as an amplitude
    `√T` to the one field. Arms through different faces then interfere as if
    s and p had the same phase. The visibility is named as an upper bound.
13. **The reference wavefront fixed by paraxial matrices.** It linearizes the
    transfer, but it ties the rays to a paraxial wavefront in fast systems.
    The f/10 PSF test is the only check of a real lens's aberrations, and
    against an oracle of the same approximation.

---

## 13. Questions for the reviewer

Each question is meant to be checkable from this brief. Q7 and Q19 also need
the cited papers, and Q21 needs the bundle items stated in Appendix E.

1. **§11.1**: Is the grating defect's analysis right?
   - Is `F = √η·exp(ik(d′ − d)·(r − p))` the correct factor for a diffracted
     order, for both transmission and reflection gratings?
   - Should the anamorphic factor `√(cos θ_in/cos θ_out)` then still apply,
     given that `η` is defined as a power ratio?
   - Is the network ledger's rule for shared events (§7.11) right once a
     grating is added to it?
2. **§7.3**: Is the projection hand-over, `r_out → r_e → r_in`, with the
   reference-change phase and `√(cos θ_in/cos θ_out)`, the correct
   local-plane-wave map for an element seen obliquely? Is the claim that it
   is exact for a flat mirror (image theory) correct?
3. **§5.1**: Is booking `(f − f²)|U|²ΔA` as power diffracted beyond the
   grid's band defensible? Is there a better split between in-band and
   out-of-band edge diffraction?
4. **§6.1, §12 item 3**: Should the DMD's thin-element mask be the facet projected
   along the array normal, along the incident direction, or along the
   reflected one? Which reproduces a tilted facet's interception and its
   specular phase? Is the facet phase `k(t_spec − t_in)·(r − r_ij)`, with
   zero at the pivot, the stationary-phase result for a facet out of the
   array plane?
5. **§6.2, §12 item 4**: Is Harvey's renormalization appropriate for the closed-form
   orders? Should the sampled near field used on the wave path apply an
   equivalent correction, or is losing the evanescent orders' power the
   honest choice there?
6. **§4.3**: The `Auto` rule compares the paraxial step's exact field error
   with the band limit's "error", taken as the norm of the field it removes.
   Is that a fair comparison? The band-limited step's remaining aliasing,
   and the impulse response's own sampling error (its kernel truncated at the
   padded window), are not counted.
7. **§4.1**: Matsushima and Shimobaba write `Δu = 1/(2S)` for a zero-padded
   window of width `2S`. Is `Δu = 1/(PΔx)`, with `P = next_fast_len(2n)`,
   the right reading when `P` is not exactly `2n`?
8. **§4.6**: Is `A′(u′, v′) = A(u, v)|∂(u, v)/∂(u′, v′)|` the right
   transformation for field values on the tilted plane? Is reporting the
   power through it as `Σ|U|²ΔA` times the carrier's obliquity acceptable for
   a beam of finite angular spread? Should the backward waves be booked by
   name? The bundle's B20 asks for waves with a negative target-normal
   component to be assigned to the negative direction, not discarded
   (Appendix E).
9. **§7.5**: Is `U_out = U_in √(T/|J|) e^{ik₀OPL}`, with rays along a
   reference that differs from the field's local direction by `θ`, correctly
   bounded by the reported walk `θL` and piston `kθ²L/2`? What criterion on
   the slab depth `L` (the sag across the window) should limit the method?
10. **§7.6, §11.2 item 2**: Which elements must update the reference `q` for
    the transfer to stay accurate? Gratings and modulators (anamorphic), and
    LCoS lens patterns, are missing. Is the Gaussian of a top-hat's radius an
    acceptable reference?
11. **§7.8**: Is the slab's closed form, applied relative to free space
    across the plate's thickness and then followed by the exact ASM between
    the beam-frame planes, correct for a tilted plate? Is cutting the round
    trips at `⌊0.4·window/shift⌋` and booking the rest's power without its
    interference with the kept terms justified? (`PlateStats::rest` holds
    that overlap.)
12. **§7.8–7.9**: Check the chart: `y_s = n cos θ`, `y_p = n/cos θ`,
    `r = (y₁ − y₂)/(y₁ + y₂)`, `t = 2√(y₁y₂)/(y₁ + y₂)`, the far-side
    reflection `−r`. Check the conversion of the coating's ray-frame
    amplitudes: `r_p` negated, `t` scaled by `√t_power_scale`. Check the
    splitters' `−√R` front / `+√R` back / `√T` signs, and the mirror's `−√R`.
    Are these mutually consistent, so that a lossless network stays unitary
    whichever element types are combined?
13. **§7.10**: How large is the error of one Jones matrix per element (the
    chief ray's) for a beam of numerical aperture NA? How large is the
    neglect of the s/p rotation for plane waves off the plane of incidence?
    Should either be flagged above some NA?
14. **§7.10, §12 item 5**: Are the LCoS polarization rules right for a
    reflective phase-only device: the "same share" absorption, and the
    outgoing polarization direction?
15. **§7.11**: Is weighting each arm by `√cos θ` before forming `|Σ|²` the
    right detector model for arms meeting a face at different angles? The
    cross term then carries `√(cos θ₁ cos θ₂)`, not `(cos θ₁ + cos θ₂)/2`.
16. **§7.2**: Is the second-moment law a safe basis for choosing windows?
    When does `σ²(z) = σ² + 2zC + z²σ_θ²`, with `C` from phase gradients,
    mislead? Should the window be chosen with a margin, or the power near
    its edge be checked after the step?
17. **§7.7**: Is the device-grid rule `per = max(spp, 2, min(32, ⌈2p/Δx⌉))`
    adequate, given §11.4? What should set the device sampling for orders at
    large angles: a bound on the facet's phase ramp per sample, or on the
    take-up's interpolation?
18. **§7.7**: Is the take-up's band cut, about the reflected family's
    carrier, the correct anti-aliasing step? For a DMD, which reflects
    several families into very different directions, should each family be
    cut about its own carrier?
19. **§6.5**: Is the fringing model (a `w × w` box mean of the *wrapped*
    phases, with a flyback at resets) a fair stand-in for the cited models?
    Is the phase's `λ₀/λ` scaling stated with the right caveat?
20. **§6.3, §11.5**: Is the 14λ scalar limit applied to the right feature
    size: the pitch, the mirror, or the gap?
21. **§10**: Which class-C tests should be replaced by independent
    references, and with what? Candidates:
    - a Mie solution for a dielectric sphere used as a ball lens, against
      which the canvas's face-by-face transfer is measured (bundle B30 and
      B40, Appendix E);
    - RCWA for the DMD and LCoS gratings;
    - measured device data.
22. **§9.2, §11.2 items 9–10**: Should the thin lens's ray law, wave phase
    and reference `q` be made the same? That would cover both direction
    cosines against tangents, and the astigmatism of an oblique lens, so that
    an off-centre or oblique chief ray leaves no residual tilt or
    inconsistent focus.
23. **§7.7**: Is cutting a modulator's angular spectrum into the cells of its
    lattice of orders, and tracing each cell about its own direction, a sound
    way to make a transfer of a field of several directions accurate? What
    error does an order joined to its neighbour carry? Would a finer cut than
    the lattice (for a hologram's own orders) be worth it?
24. **§7.11, §11.2 item 12**: For an interferometer whose arms do not stay in
    one plane, should each arm's two components be projected onto one common
    basis on the detector before they are added?

---

## 14. Sources

Bibliographic data as the repository's docs hold them, checked 2026-10-19
against the publisher, a repository or an index. "(to confirm)" marks a
detail the docs flag as not yet checked against the printed text.

- **[Allen et al. 1992]** L. Allen, M. W. Beijersbergen, R. J. C. Spreeuw,
  J. P. Woerdman, "Orbital angular momentum of light and the transformation
  of Laguerre-Gaussian laser modes," *Phys. Rev. A* 45, 8185–8189 (1992),
  doi:10.1103/PhysRevA.45.8185.
- **[Bluestein 1970]** L. I. Bluestein, *IEEE Trans. Audio Electroacoust.*
  18(4), 451–455 (1970).
- **[Born & Wolf 1999]** M. Born, E. Wolf, *Principles of Optics*, 7th ed.,
  Cambridge University Press (1999). Used for the Airy pattern and the theory
  of stratified media (a single film, a plate's multiple-beam transmission).
  Section numbers to confirm.
- **[Bundle]** The wave-optics specification bundle v1.0.0
  (`wave_optics_agent_bundle_v1.zip`, supplied by the project, 2026-10-19).
  It contains `01_MODEL.md` (M00–M20, M30–M40), `03_BENCHMARKS.md`
  (B01–B41), `05_RESULTS_AND_CLAIMS.md` and frozen fixtures. It is a
  specification, not a published source; its own references are in its
  `04_REFERENCES.md`.
- **[DLMF]** NIST Digital Library of Mathematical Functions,
  https://dlmf.nist.gov: ch. 7 (Fresnel integrals) and ch. 10 (Bessel
  functions).
- **[Fienup 1982]** J. R. Fienup, "Phase retrieval algorithms: a comparison,"
  *Appl. Opt.* 21(15), 2758–2769 (1982).
- **[Fringing-field model 2004]** "Fringing-field effect in liquid-crystal
  beam-steering devices: an approximate analytical model," *J. Opt. Soc. Am.
  A* 21(10), 1996–2008 (2004). Author list to confirm.
- **[Gerchberg & Saxton 1972]** R. W. Gerchberg, W. O. Saxton, "A practical
  algorithm for the determination of phase from image and diffraction plane
  pictures," *Optik* 35(2), 237–246 (1972).
- **[Goodman 2017]** J. W. Goodman, *Introduction to Fourier Optics*, 4th
  ed., W. H. Freeman/Macmillan Learning (2017), ISBN 978-1-319-11916-4.
  Section numbers to confirm.
- **[Gutiérrez-Cuevas & Popoff 2024]** R. Gutiérrez-Cuevas, S. M. Popoff,
  "Binary amplitude holograms for shaping complex light fields with digital
  micromirror devices," *J. Phys. Photonics* 6, 045022 (2024),
  doi:10.1088/2515-7647/ad8617 (arXiv:2311.16685v1, Eqs. (11), (15)). It
  follows W.-H. Lee, *Appl. Opt.* 18(21), 3661–3669 (1979) (DOI to confirm).
- **[Harvey & Vernold 1998]** J. E. Harvey, C. L. Vernold, "Description of
  diffraction grating behavior in direction cosine space," *Appl. Opt.*
  37(34), 8158–8160 (1998), doi:10.1364/AO.37.008158.
- **[Harvey et al. 2011]** J. E. Harvey et al., "Linear systems formulation
  of non-paraxial scalar diffraction theory," *Proc. SPIE* 8122, 812205
  (2011), doi:10.1117/12.894562.
- **[Hecht 2017]** E. Hecht, *Optics*, 5th ed., Pearson (2017), ch. 8: Malus's
  law, a retarder between crossed polarizers, the Fresnel–Arago laws.
  Section numbers to confirm.
- **[Heurtley 1973]** W. H. Heurtley, "Scalar Rayleigh–Sommerfeld and
  Kirchhoff diffraction integrals: a comparison of exact evaluations for
  axial points," *J. Opt. Soc. Am.* 63(8), 1003–1008 (1973). DOI to confirm.
- **[Hu et al. 2020]** Y. Hu et al., "Efficient full-path optical calculation
  of scalar and vector diffraction using the Bluestein method," *Light Sci.
  Appl.* 9, 119 (2020), doi:10.1038/s41377-020-00362-z. Comment by Shao &
  Urbach (2021), doi:10.1038/s41377-020-00447-9.
- **[Jones 1941]** R. C. Jones, "A new calculus for the treatment of optical
  systems. I.," *J. Opt. Soc. Am.* 31(7), 488–493 (1941),
  doi:10.1364/JOSA.31.000488.
- **[Kogelnik & Li 1966]** H. Kogelnik, T. Li, "Laser beams and resonators,"
  *Appl. Opt.* 5(10), 1550–1567 (1966). DOI to confirm.
- **[Lazarev et al. 2019]** G. Lazarev, P.-J. Chen, J. Strauß, N. Fontaine,
  A. Forbes, "Beyond the display: phase-only liquid crystal on Silicon
  devices and their applications in photonics," *Opt. Express* 27(11),
  16206–16249 (2019), doi:10.1364/OE.27.016206.
- **[Macleod]** H. A. Macleod, *Thin-Film Optical Filters*, CRC Press. The
  characteristic matrix of `oc_core::thinfilm`, used for coatings. The
  repository does not pin the edition.
- **[Matsushima & Shimobaba 2009]** K. Matsushima, T. Shimobaba,
  "Band-limited angular spectrum method for numerical simulation of
  free-space propagation in far and near fields," *Opt. Express* 17(22),
  19662–19673 (2009), doi:10.1364/OE.17.019662. Eqs. (13), (18)–(20) were
  read from the Kansai University repository copy.
- **[Matsushima, Schimmel & Wyrowski 2003]** K. Matsushima, H. Schimmel,
  F. Wyrowski, "Fast calculation method for optical diffraction on tilted
  planes by use of the angular spectrum of plane waves," *J. Opt. Soc. Am.
  A* 20(9), 1755–1762 (2003), doi:10.1364/JOSAA.20.001755.
- **[Matsushima 2010]** K. Matsushima, "Shifted angular spectrum method for
  off-axis numerical propagation," *Opt. Express* 18(17), 18453–18463
  (2010), doi:10.1364/OE.18.018453.
- **[Optiland]** https://github.com/optiland/optiland, `master` at
  `1e0513902c68948a727394c6226f3580c7261c65` (2026-10-14), cloned
  2026-10-19: the oracle for the PSF tests.
- **[Persson, Engström & Goksör 2012]** M. Persson, D. Engström, M. Goksör,
  "Reducing the effect of pixel crosstalk in phase only spatial light
  modulators," *Opt. Express* 20, 22334–22343 (2012),
  doi:10.1364/OE.20.022334.
- **[Pommet, Moharam & Grann 1994]** D. A. Pommet, M. G. Moharam,
  E. B. Grann, "Limits of scalar diffraction theory for diffractive phase
  elements," *J. Opt. Soc. Am. A* 11(6), 1827–1834 (1994),
  doi:10.1364/JOSAA.11.001827.
- **[Popoff et al. 2024]** S. M. Popoff, R. Gutiérrez-Cuevas, Y. Bromberg,
  M. W. Matthès, "A practical guide to digital micro-mirror devices (DMDs)
  for wavefront shaping," *J. Phys. Photonics* 6, 043001 (2024),
  doi:10.1088/2515-7647/ad6dc0 (arXiv:2311.17496). Eqs. (1)–(2), Fig. 5,
  §2, App. A.
- **[Rabiner, Schafer & Rader 1969]** L. R. Rabiner, R. W. Schafer,
  C. M. Rader, "The chirp z-transform algorithm," *IEEE Trans. Audio
  Electroacoust.* 17(2), 86–92 (1969), doi:10.1109/TAU.1969.1162034.
- **[Self 1983]** S. A. Self, "Focusing of spherical Gaussian beams," *Appl.
  Opt.* 22(5), 658–661 (1983).
- **[Shen & Wang 2006]** F. Shen, A. Wang, "Fast-Fourier-transform based
  numerical integration method for the Rayleigh–Sommerfeld diffraction
  formula," *Appl. Opt.* 45(6), 1102–1110 (2006), doi:10.1364/AO.45.001102.
- **[Siegman 1986]** A. E. Siegman, *Lasers*, University Science Books
  (1986), ch. 15 and 20: reduced ray matrices across media, the ABCD law for
  `q`. Chapters to confirm.
- **[Siegman 1990]** A. E. Siegman, "New developments in laser resonators,"
  *Proc. SPIE* 1224, 2–14 (1990): the second-moment width and its quadratic
  law in `z`. DOI to confirm.
- **[Stokes 1849]** G. G. Stokes, "On the perfect blackness of the central
  spot in Newton's rings, and on the verification of Fresnel's formulae for
  the intensities of reflected and refracted rays," *Cambridge and Dublin
  Mathematical Journal* 4, 1–14 (1849): `r′ = −r`, `tt′ = 1 − r²`.
- **[Swanson 1989]** G. J. Swanson, "Binary optics technology: the theory and
  design of multi-level diffractive optical elements," MIT Lincoln
  Laboratory Technical Report 854 (1989), DTIC ADA213404: an N-level
  staircase's first order, `sinc²(1/N)`. Equation location to confirm.
- **[TI DLPA083B]** Texas Instruments, *DMD Optical Efficiency for Visible
  Wavelengths*, application note DLPA083B (June 2019, rev. April 2023): Table
  2-1 (devices), Table 6-1 (efficiencies), §§4.2–4.3. A vendor simulation.
- **[Voelz & Roggemann 2009]** D. G. Voelz, M. C. Roggemann, "Digital
  simulation of scalar optical diffraction: revisiting chirp function
  sampling criteria and consequences," *Appl. Opt.* 48(32), 6132–6142
  (2009), doi:10.1364/AO.48.006132.
- **[Wen, Zhang & Xiao 2013]** J. Wen, Y. Zhang, M. Xiao, "The Talbot effect:
  recent advances in classical optics, nonlinear optics, and quantum
  optics," *Adv. Opt. Photon.* 5, 83–130 (2013).
- **[Wolf 1982]** E. Wolf, "New theory of partial coherence in the
  space–frequency domain. Part I," *J. Opt. Soc. Am.* 72(3), 343–351 (1982),
  doi:10.1364/JOSA.72.000343: mutually incoherent modes add in power.
- **[Wyrowski & Kuhn 2011]** F. Wyrowski, M. Kuhn, "Introduction to field
  tracing," *J. Mod. Opt.* 58(5–6), 449–466 (2011),
  doi:10.1080/09500340.2010.532237.
- **[Zeilinger 1981]** A. Zeilinger, "General properties of lossless beam
  splitters in interferometry," *Am. J. Phys.* 49(9), 882–883 (1981),
  doi:10.1119/1.12387: a lossless splitter is a unitary two-port. DOI to
  confirm.
- **[Zhang et al. 2009]** H. Zhang et al., "Elimination of a zero-order beam
  induced by a pixelated spatial light modulator for holographic
  projection," *Appl. Opt.* 48(30), 5834 (2009), doi:10.1364/AO.48.005834:
  the dead space as a source of the zero order.
- **[Zhang, You & Chu 2014]** Z. Zhang, Z. You, D. Chu, "Fundamentals of
  phase-only liquid crystal on silicon (LCOS) devices," *Light Sci. Appl.* 3,
  e213 (2014), doi:10.1038/lsa.2014.94: an LCoS modulates the field along
  its director only.

The FFT is `rustfft` 6.4.1 (MIT OR Apache-2.0).

---

## 15. Appendices

### A. Glossary

| Symbol or term | Meaning |
|---|---|
| `λ`, `λ₀` | wavelength in the medium; vacuum wavelength (a design wavelength for an LCoS table) |
| `k`, `k₀` | `2πn/λ₀`, `2π/λ₀` |
| `f`, `u`, `v` | spatial frequency (cycles/mm); `q = 2πf` its angular form |
| `κ`, `κ₂` | exact and paraxial axial wavenumbers, `√(k² − |q|²)` and `k − |q|²/2k` |
| `(α₀, β₀)`, carrier | direction cosines of the analytic plane-wave tilt held outside the samples |
| `t` | the tangential part of a direction (direction cosines in a plane) |
| `Δx`, `n`, `P` | sample pitch, samples per axis, padded samples per axis |
| `z_c` | critical distance `LΔx/λ` |
| `z_R`, `w₀`, `q` | Rayleigh range, waist radius, Gaussian beam parameter |
| `N_F` | Fresnel number `a²/(λz)` |
| `f` (coverage) | the covered fraction of a sample cell |
| `ρ`, `p`, `θ` (DMD) | mirror fill (side over pitch), pitch, tilt |
| `R`, `T_w` | mirror power reflectance; window single-pass power transmittance |
| `c_mn`, `η` | order coefficient; order efficiency |
| `g` (LCoS) | dead-space complex reflectance |
| `ψ`, `δ` (LCoS) | polarization angle; director angle |
| `w` (LCoS) | fringing width |
| `per` | device samples per pixel |
| `y_s`, `y_p` | tilted admittances `n cos θ`, `n/cos θ` |
| chain | one branch lineage of the axial ray from the source to the detector |
| event | a point of a chain: a hit, an aperture plane, a transfer, a plate, the end |
| family | the mirrors of one DMD state, or one LCoS polarization beam |
| arm | one of several chains that meet on the detector |
| lobe, order carried apart | one lattice cell of a modulator's angular spectrum, propagated as its own field |
| reference | the wavefront (from `q`) and ray a transfer's rays follow |
| departure | how far a field's local directions are from its reference's |
| power coordinate | a field whose `Σ|U|²ΔA` is conserved by free space (Parseval) |
| W0–W8, I1–I6 | the project's milestones and its post-review batches (§1) |
| M…, B… | bundle sections, equations and benchmarks (§1, Appendix E) |

**Symbols that are reused.** A symbol's meaning is set by its section. The
main reuses are:
- `L`: padded width (§4.2), largest offset (§4.5), cell period `Lₓ`, `Lᵧ`
  (§6.2), slab depth (§7.5), plate thickness (§7.8).
- `P`: padded samples (§4), beam power (§5.3), pattern period in pixels (§6.2,
  §10.2).
- `A`: amplitude (§5.3), angular spectrum (§4.6), cell area (§5.1), ABCD's
  `A`, Lee's target amplitude (§6.6).
- `s`: interpolation fraction (§7.14), sub-cells per axis (§7.11, §7.13),
  lateral shift (§7.8), s-polarization.
- `g`: grey level, LCoS dead-space reflectance, ray-grid size (§7.5).
- `δ`: LCoS director angle, Kronecker `δ_m0`, slab phase (§7.8), retardance,
  tilt step in `abcd`.
- `ρ`: mirror or electrode fill; the LCoS absorbed fraction (§7.10).
- `t`: tangential direction cosines, transmission amplitude, thickness.
- `w`: Gaussian radius, fringing width, `√(1/λ² − f²)` (§7.8).
- `q`: Gaussian parameter, `2πf` (§4.3), Lee's `q` (§6.6).
- `n`: refractive index, samples per axis.
- `N`: pixel count, the levels of a staircase, `N_F`.
- `C`: the second-moment cross term (§7.2), the network's detector map
  (§7.11).
- `T`: power transmittance, the network operator (§7.11).
- `ε`: polarizer extinction, the 1 µm offset in `abcd`.
- `λ₀`: vacuum wavelength; for an LCoS table, its design wavelength.
- `θ`: the DMD tilt, the local direction in §7.2, the angles of incidence
  and obliquities.
- `R`: reflectance, the Gaussian's `R(z)`, a face's radius.
- `x`: a coordinate, and the round-trip factor in §7.8.
- `m`: an order index, a mask, the round trips kept (§7.8).
- `a`: the mirror side (and, in `mirror_outline`, its half), aperture radii,
  a beam-frame coordinate.
- `D`: an aperture's diameter, ABCD's `D`, the slab's denominator.

### B. Reproducing the numbers (for the project, not the reviewer)

```sh
# all oc-wave tests, values printed
cargo test -p oc-wave --features scene --release -- --nocapture --test-threads=1
# the ray tracer's DMD and SLM tests
cargo test -p oc-trace --features trace3 --release --test modulator3 --test modulator_orders3 -- --nocapture
# the Optiland oracles (needs Optiland at the pinned commit, its venv, and maturin)
cd crates/oc-py && maturin develop --release && pytest tests/test_wave_oracles.py -s
```

### C. The grating reproduction (scratch program, not in the repository)

A Cargo project depending on `oc-core` (feature `scene3`) and `oc-wave`
(features `scene`, `parallel`) by path, at `1af40e8`:

```rust
// Scratch check (not in the repo): a transmission grating on the wave path.
// Detector square to the diffracted chief ray, 100 mm on. Report the power
// centroid in detector pixels, the arriving field's grid, and the ledger.
use oc_core::scene::Spectrum;
use oc_core::scene3::{Scene3, SpectralScalar, SurfaceResponse};
use oc_core::vec3::Vec3;
use oc_wave::path::{wave_path, BeamProfile, Readout, WavePathSpec};

fn run(lpmm: f64, zero: f64, readout: Readout, window: f64) {
    let lam = 0.6328;
    let mut s = Scene3::new();
    let src = s.add_collimated_beam(Vec3::ZERO, 1.0, 1.0, 1, Spectrum::Monochromatic(lam));
    let st = lam / 1000.0 * lpmm;
    let d = Vec3::new((1.0 - st * st).sqrt(), st, 0.0);
    s.add_grating(Vec3::new(50.0, 0.0, 0.0), 20.0, 20.0, 0.0, lpmm, true);
    if let SurfaceResponse::Grating { zero_order, .. } = &mut s.surfaces.last_mut().unwrap().response_front { *zero_order = SpectralScalar::Constant(zero); }
    if let SurfaceResponse::Grating { zero_order, .. } = &mut s.surfaces.last_mut().unwrap().response_back { *zero_order = SpectralScalar::Constant(zero); }
    let p = Vec3::new(50.0, 0.0, 0.0) + d * 100.0;
    let det = s.add_area_detector(p, -d, 4.0, 4.0, [80, 80]);
    s.validate().unwrap();
    let spec = WavePathSpec {
        source: src, detector: det,
        beam: BeamProfile::Gaussian { waist_radius_mm: 0.3, waist_offset_mm: 0.0 },
        power: 1.0, grid_n: 512, window_mm: window, device_samples_per_pixel: 4, readout,
    };
    match wave_path(&s, &spec, 1) {
        Ok(r) => {
            let [nu, nv] = r.bins;
            let (mut t, mut cu, mut cv) = (0.0, 0.0, 0.0);
            for v in 0..nv { for u in 0..nu {
                let w = r.power[v * nu + u];
                t += w; cu += w * (u as f64 + 0.5 - nu as f64 / 2.0) * 0.1; cv += w * (v as f64 + 0.5 - nv as f64 / 2.0) * 0.1;
            }}
            let l = r.report.ledger;
            println!("{lpmm:>5} l/mm, zero order {zero}, {readout:?}, window {window}: on detector {t:.4}, power centroid on the face ({:+.4}, {:+.4}) mm; field grid centre {:?}; detected {:.4} other_ports {:.4} untracked {:.4} left_window {:.2e}",
                cu / t.max(1e-300), cv / t.max(1e-300), r.field.grid.center_mm, l.detected, l.other_ports, l.untracked_orders, l.left_window);
        }
        Err(e) => println!("{lpmm} l/mm: refused: {e}"),
    }
}

fn main() {
    let lam = 0.6328_f64;
    for lpmm in [20.0, 100.0] {
        let st = lam / 1000.0 * lpmm;
        println!("-- {lpmm} l/mm: zero order lands {:+.4} mm from the first order's chief ray on this face", -100.0 * st / (1.0 - st * st).sqrt());
        run(lpmm, 0.0, Readout::Projected, 6.0);
        run(lpmm, 0.0, Readout::Diffracted { samples_per_pixel: 2 }, 6.0);
        run(lpmm, 0.10, Readout::Projected, 6.0);
    }
}
```

Its output at `1af40e8`:

```text
-- 20 l/mm: zero order lands -1.2657 mm from the first order's chief ray on this face
   20 l/mm, zero order 0, Projected, window 6: on detector 0.6500, power centroid on the face (+0.0159, -1.2658) mm; field grid centre [0.0, 0.0]; detected 0.6500 other_ports 0.0000 untracked 0.3500 left_window 2.31e-14
   20 l/mm, zero order 0, Diffracted { samples_per_pixel: 2 }, window 6: on detector 0.6500, power centroid on the face (+0.0250, -1.2627) mm; field grid centre [4.884981308350689e-15, 5.329070518200751e-15]; detected 0.6500 other_ports 0.0000 untracked 0.3500 left_window 2.91e-14
   20 l/mm, zero order 0.1, Projected, window 6: on detector 1.2595, power centroid on the face (+0.0160, -1.2657) mm; field grid centre [0.025, 0.025]; detected 1.2595 other_ports -1.5095 untracked 1.2500 left_window 2.31e-14
-- 100 l/mm: zero order lands -6.3407 mm from the first order's chief ray on this face
  100 l/mm, zero order 0, Projected, window 6: on detector 0.6500, power centroid on the face (+0.0596, -0.9390) mm; field grid centre [0.0, 0.0]; detected 0.6500 other_ports 0.0000 untracked 0.3500 left_window 2.31e-14
  100 l/mm, zero order 0, Diffracted { samples_per_pixel: 2 }, window 6: on detector 0.6546, power centroid on the face (+0.0670, -0.9364) mm; field grid centre [0.0, 0.0]; detected 0.6546 other_ports 0.0000 untracked 0.3500 left_window 5.89e-12
  100 l/mm, zero order 0.1, Projected, window 6: on detector 0.6500, power centroid on the face (+0.0596, -0.9390) mm; field grid centre [0.0, 0.0]; detected 0.6500 other_ports 0.0000 untracked 0.3500 left_window 2.31e-14
```

The detector's local axes are not the beam frame's, so the centroid's sign
and its axis follow the detector. In the third line, `untracked 1.2500` is
the second ledger defect of §11.1.

Printing the full ledger and the arriving field for the one-order runs gives:
- **20 l/mm**: the field's centroid in the beam frame is −1.2658 mm. The
  local direction at the window centre is −0.01198, which is `−tan θ`
  (−0.01266) plus the beam's own curvature.
- **100 l/mm**: the centroid is −0.9409 mm and the direction −0.00878. This
  is the aliased ramp: 7.38 rad per sample is 1.10 rad modulo 2π.
- **Both runs**: `band_limit_removed` 0, `left_window` 2.3e-14,
  `resampling` −3.8e-8 and −2.6e-7, and no warnings.

### C.2 The off-centre window check (scratch program, not in the repository)

The same Cargo project, a second binary. Its output at `1af40e8` is the table in §11.1.2.

```rust
// Scratch check (not in the repo): asm_zoom, fresnel_zoom (via czt) and the
// plain ASM on a field whose input window is centred off the frame origin.
use oc_wave::beams::GaussianBeam;
use oc_wave::{Fft2, Grid2, Propagator, propagate, asm_zoom, asm_tilted};

fn main() {
    let (lam, w0, z) = (0.6328, 0.05, 5.0);
    let b = GaussianBeam { w0_mm: w0, lambda_um: lam, n: 1.0, power: 1.0, center: [0.3, 0.0] };
    for c in [0.0f64, 0.3] {
        // the input window centred at (c, 0); the beam always at x = 0.3
        let g = Grid2::new([256, 256], [0.004, 0.004], [c, 0.0]).unwrap();
        let u = b.sample(g, 0.0).unwrap();
        let mut fft = Fft2::new();
        let out = Grid2::new([128, 128], [0.004, 0.004], [0.3, 0.0]).unwrap();
        let (v, _) = asm_zoom(&u, z, out, 2, &mut fft).unwrap();
        let (pv, _) = propagate(&u, z, Propagator::Auto, &mut fft).unwrap();
        let cen = |f: &oc_wave::Field2| {
            let (mut p, mut mx) = (0.0, 0.0);
            for j in 0..f.grid.n[1] { for i in 0..f.grid.n[0] { let (x, _) = f.grid.xy(i, j); let w = f.at(i, j).norm_sqr(); p += w; mx += w * x; } }
            mx / p
        };
        let mut worst: f64 = 0.0;
        let peak = b.value(0.3, 0.0, z).abs();
        for j in 0..out.n[1] { for i in 0..out.n[0] { let (x, y) = out.xy(i, j); worst = worst.max((v.at(i, j) - b.value(x, y, z)).abs()); } }
        let mut target = u.frame; target.pos += oc_core::Vec3::new(0.0, 0.0, z);
        let (t, _) = asm_tilted(&u, target, out, 2, &mut fft).unwrap();
        let mut worst_t: f64 = 0.0;
        for j in 0..out.n[1] { for i in 0..out.n[0] { let (x, y) = out.xy(i, j); worst_t = worst_t.max((t.at(i, j).abs() - b.value(x, y, z).abs()).abs()); } }
        println!("  asm_tilted (untilted target {z} mm on): centroid {:+.5} mm, worst |U| vs Kogelnik-Li / peak {:.3e}", cen(&t), worst_t / peak);
        println!("input window centre {c}: asm_zoom centroid {:+.5} mm (beam at +0.30000), worst vs Kogelnik-Li / peak {:.3e}; plain ASM on the input grid centroid {:+.5}", cen(&v), worst / peak, cen(&pv));
    }
}
```

The point-sampling check of §11.1.3 builds the scene of
`a_dmd_s_orders_cross_a_singlet_each_about_its_own_direction` (§10.3) unchanged, and runs it
with `Readout::Diffracted { samples_per_pixel: s }` for s = 2, 8, 32.

### D. Helpers and types the excerpts use

**Grids and transforms**

- **`Grid2 { n, pitch_mm, center_mm }`**
  - `coord(axis, i) = center + (i − ⌊n/2⌋)·pitch`; `xy(i, j)` gives both
    coordinates.
  - `idx(i, j) = j·n₀ + i` (row-major, `x` fastest).
  - `width_mm(axis) = n·pitch` (not `(n − 1)·pitch`), and
    `freq_pitch(axis) = 1/width_mm`.
  - `fft_freq(axis, k)` is bin `k`'s frequency in **FFT order**,
    `0, 1, …, ⌈n/2⌉ − 1, −⌊n/2⌋, …, −1`, times `freq_pitch`. After
    `fftshift2`, bin `k` sits at `(k − ⌊n/2⌋)·freq_pitch` instead.
  - `cell_area_mm2() = pitch₀·pitch₁`; `padded(f)` has `f` times the samples
    at the same pitch, so a window `f` times as wide.
  - `voelz_sampling(Δx, L, λ, z)` says which side of `z_c = LΔx/λ` a
    distance is on.
- **`Fft2::forward(data, n, scale)`** computes the unnormalized sum
  `Σ x·e^{−i2πkm/N}`, per axis, times `scale`. `inverse` computes
  `Σ X·e^{+i2πkm/N}` times `scale`, with **no** `1/N` inside. The two scales
  must therefore supply the normalization:
  - forward `ΔxΔy` with inverse `Δf_xΔf_y = 1/(NΔxΔy)` is the identity pair
    of §3;
  - direct RS uses forward 1 and inverse `ΔxΔy/(q₀q₁)`: the discrete
    convolution sum times the area element.
- **`fftshift2`** moves index 0 to `⌊n/2⌋`; **`ifftshift2`** undoes it, which
  puts a grid's centre sample at the transform's origin.
  **`next_fast_len(n)`** is the smallest length ≥ `n` with only 2, 3 and 5
  as prime factors.
- **`embed(u, p)`** places `u` in the middle of a zero array of `p` samples,
  its centre sample at `⌊p/2⌋`; **`crop(buf, p, n)`** takes the middle `n`
  back. **`sum_sq`** is `Σ|z|²`. **`check_wrap`** raises `WrapAround`
  (§4.1). **`advance(u, z)`** moves a field's frame `z` along its `+Z`.
- **`AsmOptions`**: `band_limit` (default true) and `pad` (default 2: pad to
  `next_fast_len(2n)`; 1: no padding, circular).
- **`czt_line`, `czt2`** (§4.4): `czt2(u, f₀, δf, m)` gives
  `Σ u(x) e^{−i2πf·x} ΔxΔy` at `f = f₀ + kδf`. Here `x` are the samples'
  absolute coordinates in the frame (`grid.coord`), and the carrier is not
  included.
- **`centred(g)`** moves an even axis's samples half a pitch, onto the
  centres of the cells the window divides into (§7.13).

**Ledger helpers**

- **`merge(total, step)`** adds a step report's **loss** entries into the
  running ledger, and appends its steps, warnings, flags and approximations.
  `launched` and `detected` are not added: a path keeps its own.
- **`add_losses(l, d)`** adds `d`'s loss entries, everything but `launched`
  and `detected`, into `l`. **`losses_between(after, before)`** is the
  entry-by-entry difference of two ledgers, losses only.

**Ray tracing**

- **`RayOut { x, y, opl, t, dx, dy }`**: one traced ray at the exit plane.
  - `x`, `y`: where it crosses, in the exit frame;
  - `opl`: its OPL from the entry plane;
  - `t`: its power transmittance;
  - `dx`, `dy`: its direction cosines across the exit frame, which give
    `abcd`'s C and D.
- **`trace_with`** launches a ray from `(a, b)` on the entry plane along the
  direction cosines `t` across the entry frame. It traces it through the
  probe scene and returns its `RayOut` after exactly one interaction.
  **`trace_one`** and **`ray_through`** call it.
- **`plane_depth(beam, grid, o, n)`** is the largest distance, along the
  beam, from the window's corners to an oblique plane: the depth a projection
  spans (§7.4).

**Equations not reproduced.** Three functions are used without their
equations stated here:
- `coating_amp_at`: the Macleod characteristic-matrix TMM of `oc-core`, in
  the ray frame, conjugated into `exp(−iωt)`. For an opaque backing its
  transmitted amplitudes are zero.
- `interface_amp`: the same TMM with no layer.
- `a_plate_retardance`: the retardance of a uniaxial plate with its optic
  axis in its plane, at the chief ray's angle and azimuth to the axis.

**Fields and frames**

- **`Field2`** (§3):
  - `data`, `grid`, `lambda_um` (vacuum, µm), `n_medium`, `carrier`, `frame`,
    `opl_ref_mm`;
  - `k_mm() = 2πn/λ₀` in rad/mm, and `wavelength_in_medium_mm()`;
  - `power() = Σ|U|²ΔA`, and `at(i, j)` is a sample;
  - `bake_carrier()` folds the carrier into the samples.
- **`Isometry3`**: a rigid transform from local to world coordinates.
  - `rotate(v)` maps a local direction to the world;
  - `transform_point(p)` maps a local point to the world;
  - `inverse_transform_point` maps a world point back to local;
  - `quat()` is the rotation, and `pos` is the origin in the world.
- **`Beam { origin, x, y, d }`**: a frame across the chief ray, with unit
  axes and `y = d × x`.
  - `coords(r) = ((r − origin)·x, (r − origin)·y)`;
  - `point(a, b) = origin + a·x + b·y`;
  - `turned(o, d′)` turns the axes to the new direction `d′` by the shortest
    rotation;
  - `iso()` is the `Isometry3`.
- **`onto_plane(r, dir, p, n)`**: where the line `r + s·dir` meets the plane
  through `p` with normal `n`; `None` when the line runs along the plane.
- **`sample(u, a, b)`**: bilinear interpolation of `u`'s samples at local
  `(a, b)`, zero outside the grid.
- **`Smooth`** (§7.14) reads a field between samples with a reference
  wavefront taken out. **`Reference { curv, tilt, centre }`** (§7.5) has
  `dir(a, b) = tilt + curv ⊙ ((a, b) − centre)`. **`fit_sphere(u)`** fits
  `t = (α + c·a, β + c·b)` to the phase gradient, weighted by intensity.

**Path state**

- **`ChainJob`** is one chain at one wavelength. Its state:
  - `u`, and `uy` when it carries two components;
  - `beam`, the frame;
  - `q[2]`, the reference parameter per frame axis;
  - `at = [a, b, tₓ, tᵧ]`, the reference ray;
  - `rest`, the orders carried apart (`Lobe { u, uy, at }`);
  - `pol` (`Pol`), the polarization the field stands for;
  - `face`, true when the chain is one of several arms, so it is read onto
    the detector grid;
  - `points`, the events, and `keys`, `arms` and `ledgers`, for the network
    ledger.
- **`WaveJob`** holds one `Mode` per line and polarization launch. Each mode
  holds its chains: running, read out (`Face`), or done. **`revision`** is
  the scene revision the job answers for. A job is never mixed with a
  changed scene.
- **`FaceField`**: one arm read onto the detector's sub-pixel grid. It holds
  `data` and `data_y`, `cos` (`|n_det·d|`), `dir`, and the arm's `keys`,
  `arms` and `ledgers`.
- **`Arm { object, shared, foreign_absorbed, foreign_family }`**: where a
  chain parts from others. `shared` is true for a splitter (one field meets
  it) and false for an LCoS (each beam meets it with its own polarization).
- **`Regrid { out, how, half, needed }`**, where `how` is
  `RegridHow::Zoom(pad)` or `RegridHow::Fresnel`. **`Departures`** collects a
  transfer's departure statistics for its note.
- **`TracedMap`**: the fitted affine map, the quadratic OPL and their
  bilinear residuals on the ray grid.
  - `at(a, b) → (x′, y′, opl relative to the chief ray, T, det ∂x′/∂x)`;
  - `inverse(x′, y′)` gives the entry point.

  **`TransferStats.carried`** is the arriving power after the rim.
- **`PlateStats`**:
  - `reflected`, which the event books as `other_arm`;
  - `beyond`, the round trips summed out, booked as `left_window`;
  - `left_window`, the crop;
  - `evanescent`;
  - `rest`, booked as `resampling`.
- **`Pol`**:
  - `Unpolarized`: one field for unpolarized light, or a square element;
  - `S(ŝ)`, `P(ŝ)`: the component being run through an oblique element;
  - `Projected`: the one component an LCoS acts on.

  `is_p()` gives `Some(true)` for `P` and `Some(false)` for `S`.
- **`Loss`** in `flat_element`: `Absorbed`; `Arm(f)`, a fraction to the
  other arm with the rest absorbed; `Orders`, untracked orders.
  **`unit()`** clamps a sampled value to [0, 1].

**Scene and devices**

- **`SeqHit3`**: one hit of a chain.
  - `point`, `normal`, `object`, `patch`, `label`;
  - `kind`: `Body` or `Surface`;
  - `action`: `Reflect`, `Transmit`, `Diffract`, `ThinLens`, …;
  - `n_before`, `n_after`.

  A surface's **front** is the side its local `+Z` faces. A beam with
  `d·ẑ < 0` meets the front and gets `response_front`.
- **`SpatialModulator { pixels, pitch_um, kind, pattern }`**:
  `pixel_center_mm(i, j)`; `value(i, j)` is a DMD state or an LCoS grey level.
  - **`incident_direction(carrier)`** = `(α, β, −√(1 − α² − β²))`.
  - **`Shape::cell_overlap`** is the exact area of a shape inside a
    rectangle.
  - **`pixel_range(lo, hi, n, p)`** gives the pixels whose cells may overlap
    `[lo, hi]`.
  - **`validity_flags`** raises `ScalarLimit` when `p < 14λ`.
  - **`lattice_sum`** is the chirp-z sum over pixel centres. It divides by
    `p²`, so it is a sum, not an integral.
- **`FilmAmp`**: §7.9. **`interface_amp(n₁, n₂, cos θ)`**: a bare interface's
  `FilmAmp`.

**Constants**

| Constant | Value |
|---|---|
| `MAX_LOBES` | 9 |
| `MAX_WAVE_LINES` | 16 |
| `MAX_WAVE_SAMPLES` | 2²² in the browser, 2²⁶ natively |
| `MAX_WAVE_GRID` (the app) | 2048 across |

### E. The bundle items this brief cites

The bundle [Bundle] is a specification supplied by the project. These are its
statements, condensed, with its own equation tags. Section M numbers and
equation M tags overlap: for example, Eq. M14 is in section M06.

- **Section M00, Eq. M00.** A linear coherent system operator `T`, and a
  detector's power `P_p = Tr(Q_p T Γ T†)` for a source second moment `Γ`. The
  system is assembled from component scattering operators `S` and
  interconnections `P`, with `a = Pb + Bs`, `b = Sa`, `y = Cb`. Hence
  `T = C(I − SP)⁻¹SB`. Thin screens, finite Gaussian transport, local
  planar treatments of curved interfaces and simplified detectors are
  *declared approximations*.
- **Section M06, Eqs. M12–M14** (planar scattering). Conserve `q`;
  `e_i + e_r = e_t`, `Y₁(e_i − e_r) = Y₂e_t`. For each TE/TM channel with real
  positive admittances, the power-coordinate two-port is
  `S = (1/(y₁ + y₂))·[[y₁ − y₂, 2√(y₁y₂)], [2√(y₁y₂), y₂ − y₁]]`, unitary.
  A textbook p reflection coefficient may differ by a sign from the chosen
  reflected unit vector. This is what the code's "bundle M14; the Stokes
  relations" refers to.
- **Section M09** (network closure, passivity). `(I − SP)⁻¹ = Σ(SP)^m` are
  coherent path amplitudes, and all coherent contributions interfere before
  power is formed. Passivity is `S†S ⪯ I` and `P†P ⪯ I`.
- **Section M10, Eq. M20** (exact carrier-coordinate change). For a real
  reference optical path `Φ`, `ψ = R_Φ v = e^{ik₀Φ}v`, and
  `T_residual = R_{Φout}⁻¹ T R_{Φin}`. A band projector is to be applied to
  the reconstructed physical `ψ`, or the conjugated projector to `v`. A
  cutoff on `v`'s own spectral disk is generally wrong.
- **Section M13, Eqs. M32–M33** (computable approximation errors).
  `‖(U − U₂)g‖² = ∫4 sin²[(d/2)(κ − κ₂)]|ĝ|² d²q ≤ d²‖(κ − κ₂)ĝ‖²`, for a
  quadratic `κ₂`; the Fresnel `κ₂` omits the leading phase `−d|q|⁴/(8k³)`.
  For a carrier `Φ = Φ₂ + R₃`, `‖(e^{ik₀Φ} − e^{ik₀Φ₂})g‖ ≤ k₀‖R₃g‖`.
- **Section M14, Eqs. M34–M35** (thin masks). The scalar screen is
  `(Mψ)(ρ) = m(ρ)ψ(ρ)` with `|m| ≤ 1`. Projected onto the propagating
  subspace it is only contractive, and the discarded projection norm is to
  be reported separately, **not as absorption**. A finite aperture acts at
  its plane, `T = U₂MU₁` with `[M, U] ≠ 0`.
  - Eq. M34: a Gaussian through a hole passes `1 − e^{−2a²/w²}`.
  - Eq. M35: two slits give `E_FF ∝ 2a sinc(aq/2) cos(dq/2)`, with
    `sinc(u) = sin u/u`.
  - A sinusoidal phase mask `e^{iβ cos Gx}` has coefficients `i^m J_m(β)`.
    Exact high-angle order powers are not to be inferred from that.
- **Section M15, Eqs. M36–M37** (coherence and detector).
  `Γ = Σ w_m u_m u_m†`. Each coherent mode follows all coherent paths, and
  uncorrelated modes add intensities. M1 uses "the separately declared
  paraxial area detector".
- **Section M16, Eq. M40** (slab). `t = (1 − r²)e^{iδ}/(1 − r²e^{2iδ})`, with
  `T = (1 − r²)²/D`, `D = (1 − r²)² + 4r² sin²δ`. At `δ = mπ`, `t = (−1)^m`.
- **Section M17, Eqs. M41–M42** (dielectric sphere). The Mie coefficients
  `a_l`, `b_l` with outgoing `h_l^(1)` for `exp(−iωt)`. For real `m, x`,
  `Re a_l = |a_l|²`, and `C_ext = C_sca` for a lossless sphere. It is "an
  exact special-geometry reference, not a general curved-lens validation".
- **Section M11** (Gaussian frames). Gaussian atoms
  `g_j = N_j exp{ik₀[p_j·ξ + ½ξᵀK_jξ]}` with `Im K_j ≻ 0`, analysed by
  stable SVD or QR with a singular cutoff.
- **B02**: six periodic modes on a 128 × 96 µm cell, 32 × 24 samples, carried
  by `exp{id√((nk₀)² − |q|²)}`, against an explicit Fourier sum, with
  semigroup and adjoint checks.
- **B03**: `n` 1 → 1.5. At normal incidence `r = −0.2`, `t_e = 0.8`,
  `R = 0.04`; Brewster TM `r = 0`; `S†S = I`.
- **B10**: `1 − e^{−1/2}` for w = 100 µm, a = 50 µm, to 2e-10 for a
  continuum quadrature. A binary Cartesian mask "is not expected to meet
  this". The finite placement `U_after M U_before` on a 24² cell is held
  against an explicit DFT. The misplaced order differs by about 0.4329834561.
- **B20**: tilted planes. The reference is the original 3D plane-wave sum.
  Waves with a **negative target-normal component must be assigned to the
  negative target direction, not discarded**.
- **B21, B22**: a Gaussian frame with redundant columns, and carrier
  coordinates with the transformed projector. Applying the original
  projector to the residual is "deliberately incorrect".
- **B30**: a dielectric sphere at λ₀ 550 nm, `n` 1 and 1.5,
  `kR = 10, 100, 1000`, a plane wave along `+z` with `E ∥ x`, checked at
  listed exterior points. The checks are convergence, `Re a_l = |a_l|²`,
  `C_ext = C_sca`, and complex E and H retained.
- **B31**: an optional separate Maxwell boundary solver against B30 at
  `kR = 10`.
- **B40**: the curved local approximation against Mie, on B30's `n = 1.5`
  family. Record the complex-field error, phase error, scattering error,
  passivity defect, runtime and memory. "No universal physical-accuracy
  pass threshold"; "Do not force agreement by renormalizing outgoing power".
