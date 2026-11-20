//! Free-space propagation of a [`Field2`] between parallel planes.
//!
//! Every propagator here is a published method, cited where it is
//! implemented:
//!
//! | [`Propagator`] | Method | Source |
//! |---|---|---|
//! | `AngularSpectrum` | angular spectrum of plane waves, optionally band-limited | Goodman, *Introduction to Fourier Optics* 4th ed. (2017); Matsushima & Shimobaba, *Opt. Express* 17, 19662–19673 (2009), doi:10.1364/OE.17.019662 |
//! | `FresnelTf` / `FresnelIr` | Fresnel transfer function / impulse response | Voelz & Roggemann, *Appl. Opt.* 48(32), 6132–6142 (2009), doi:10.1364/AO.48.006132 |
//! | `Fraunhofer` | one transform onto the far-field grid `λz/(NΔx)` | Goodman |
//! | `RsDirect` | direct Rayleigh–Sommerfeld convolution, Simpson-weighted | Shen & Wang, *Appl. Opt.* 45(6), 1102–1110 (2006), doi:10.1364/AO.45.001102 |
//!
//! Each call returns the propagated field and a [`WaveReport`] whose ledger
//! accounts for the input power: what arrived, what a band limit removed,
//! what decayed as evanescent waves, what left the window.

pub(crate) mod asm;
pub(crate) mod tilted;
mod fraunhofer;
mod fresnel;
mod regime;
mod rs_direct;

pub use asm::{asm_zoom, AsmOptions};
pub use tilted::asm_tilted;
pub use regime::{critical_distance_mm, fresnel_phase_error_rad, voelz_sampling};

/// The relative field error past which neither candidate of
/// [`Propagator::Auto`] certifies a step: the step still runs with the
/// smaller indicator, and the report says it is not certified
/// (`OutsideRegime`).
pub const STEP_TOLERANCE: f64 = 1e-2;

/// The critical distance of a window as the propagators pad it,
/// `next_fast_len(2n)` samples a side: per axis `PΔx²/λ`, the smaller of
/// the two (Voelz & Roggemann 2009; the padded length, axis by axis, as
/// Matsushima & Shimobaba's band limit uses it).
pub fn padded_critical_distance_mm(grid: &crate::Grid2, lambda_mm: f64) -> f64 {
    (0..2)
        .map(|a| critical_distance_mm(grid.pitch_mm[a], crate::fft::next_fast_len(2 * grid.n[a]) as f64 * grid.pitch_mm[a], lambda_mm))
        .fold(f64::INFINITY, f64::min)
}

use crate::report::{Method, SamplingWarning, StepReport};
use crate::{Cx, Field2, Fft2, WaveError, WaveReport};

/// Which propagator to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Propagator {
    /// Chosen per step and recorded in the report: the band-limited angular
    /// spectrum up to the critical distance, where it is exact and loses
    /// nothing; beyond it, whichever of the Fresnel impulse response and the
    /// band-limited angular spectrum makes the smaller error on this very
    /// field, both known from its spectrum before either runs: the paraxial
    /// step's exactly ([`paraxial_error`]), the band limit's as the field it
    /// would remove ([`band_limit_error`]).
    Auto,
    AngularSpectrum(AsmOptions),
    FresnelTf,
    FresnelIr,
    /// The far field on its own grid, pitch `λz/(PΔx)` for a window padded to
    /// `pad` times its samples.
    Fraunhofer { pad: usize },
    RsDirect,
}

/// Propagate `u` a distance `z_mm` along its plane's `+Z` (negative `z_mm`
/// goes back).
pub fn propagate(
    u: &Field2,
    z_mm: f64,
    how: Propagator,
    fft: &mut Fft2,
) -> Result<(Field2, WaveReport), WaveError> {
    if !z_mm.is_finite() {
        return Err(WaveError::OutOfRange(format!(
            "propagation distance {z_mm} mm is not finite"
        )));
    }
    let launched = u.power();
    let mut report = WaveReport::default();
    report.ledger.launched = launched;

    if z_mm == 0.0 && !matches!(how, Propagator::Fraunhofer { .. }) {
        report.ledger.detected = launched;
        report.steps.push(StepReport {
            method: Method::Element,
            z_mm: 0.0,
            sampling: None,
            note: "zero distance: the field is unchanged".into(),
        });
        report.ledger.close();
        return Ok((u.clone(), report));
    }

    let (how, paraxial) = match how {
        Propagator::Auto => choose(u, z_mm, fft),
        h => (h, None),
    };
    let out = match how {
        Propagator::AngularSpectrum(opts) => asm::run(u, z_mm, opts, fft, &mut report)?,
        Propagator::FresnelTf => fresnel::run_tf(u, z_mm, fft, &mut report)?,
        Propagator::FresnelIr => fresnel::run_ir(u, z_mm, fft, &mut report)?,
        Propagator::Fraunhofer { pad } => fraunhofer::run(u, z_mm, pad, fft, &mut report)?,
        Propagator::RsDirect => rs_direct::run(u, z_mm, fft, &mut report)?,
        Propagator::Auto => unreachable!("resolved above"),
    };
    if let Some((ep, eb)) = paraxial {
        if let Some(st) = report.steps.last_mut() {
            st.note += &format!(
                "; chosen on this field's model-error indicators, from its spectrum padded as the step pads it: the paraxial step's {ep:.2e}, the band limit's {eb:.2e} (the Fresnel step's quadrature, kernel truncation and window are not among them)"
            );
        }
        if ep.min(eb) > STEP_TOLERANCE {
            let method = report.steps.last().map_or(Method::Element, |s| s.method);
            report.warn(SamplingWarning::OutsideRegime {
                method,
                reason: format!(
                    "not certified: both candidates' indicators exceed {STEP_TOLERANCE:.0e} of the field (the paraxial step's {ep:.2e}, the band limit's {eb:.2e}); the smaller ran"
                ),
            });
        }
    }
    if matches!(how, Propagator::FresnelTf | Propagator::FresnelIr) {
        let e = paraxial.map_or_else(|| paraxial_error(u, z_mm, fft), |p| p.0);
        if paraxial.is_none() {
            if let Some(st) = report.steps.last_mut() {
                st.note += &format!("; the paraxial model's error indicator on this field {e:.2e}, from its padded spectrum (the step's own quadrature and kernel truncation not among it)");
            }
        }
        if e > 1e-4 {
            report.warn(SamplingWarning::Paraxial { field_error: e });
        }
    }
    report.ledger.detected = out.power();
    report.ledger.close();
    Ok((out, report))
}

/// The paraxial (Fresnel) propagator's error on this field over `z_mm`,
/// exactly: with `κ = √(k² − |q|²)` and its paraxial form `κ₂ = k − |q|²/2k`,
///
/// ```text
/// ‖(U − U₂)g‖² = Σ_q 4 sin²(z(κ − κ₂)/2) |ĝ(q)|²,   κ − κ₂ = −|q|⁴ / (2k(k + κ)²),
/// ```
///
/// the identity of the bundle's M13 (Eq. M32), the difference in its stable
/// form, summed over the field's own angular spectrum, its carrier included,
/// the field padded as the steps pad it (`next_fast_len(2n)` a side, so the
/// spectrum is sampled `1/(PΔx)` apart per axis). An evanescent plane wave,
/// which the exact propagator damps by `e^{−z√(|q|² − k²)}` and the paraxial
/// one carries unchanged, counts by the difference of the two transfer
/// factors, `|e^{−z√(|q|² − k²)} − e^{izκ₂}|²`. Returned relative to the
/// field's norm, `‖(U − U₂)g‖/‖g‖`: an indicator of the paraxial model's
/// error on this field, not a bound on a sampled Fresnel step's (whose
/// quadrature and kernel truncation it leaves out). Unlike a bound from the
/// window's corner it knows where the field's power actually is: a narrow
/// beam tilted far off its plane's normal is paraxial in nothing.
pub fn paraxial_error(u: &Field2, z_mm: f64, fft: &mut Fft2) -> f64 {
    regime_errors(u, z_mm, fft).0
}

/// The field a band-limited angular-spectrum step over `z_mm` would remove,
/// relative to the field's norm: the root of the power its spectrum holds
/// outside the band limit of Matsushima & Shimobaba (2009), Eqs. (13),
/// (18)–(20), for the window as `asm` pads it (`next_fast_len(2n)` a side,
/// `Δu = 1/(PΔx)` per axis).
pub fn band_limit_error(u: &Field2, z_mm: f64, fft: &mut Fft2) -> f64 {
    regime_errors(u, z_mm, fft).1
}

/// Both indicators from one transform of the field, padded as the steps pad
/// it.
fn regime_errors(u: &Field2, z_mm: f64, fft: &mut Fft2) -> (f64, f64) {
    let n = u.grid.n;
    let p = [crate::fft::next_fast_len(2 * n[0]), crate::fft::next_fast_len(2 * n[1])];
    let Ok(gp) = crate::Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm) else { return (0.0, 0.0) };
    let mut buf = embed(u, p);
    crate::fft::ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, u.grid.cell_area_mm2());
    let lam = u.wavelength_in_medium_mm();
    let k = 2.0 * std::f64::consts::PI / lam;
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    // The band limit for the window as the step pads it.
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let u_lim = 1.0 / (lam * ((2.0 * du * z_mm).powi(2) + 1.0).sqrt());
    let v_lim = 1.0 / (lam * ((2.0 * dv * z_mm).powi(2) + 1.0).sqrt());
    let (mut err, mut cut, mut tot) = (0.0, 0.0, 0.0);
    for ky in 0..p[1] {
        let fy = gp.fft_freq(1, ky) + v0;
        for kx in 0..p[0] {
            let fx = gp.fft_freq(0, kx) + u0;
            let e = buf[ky * p[0] + kx].norm_sqr();
            if e == 0.0 {
                continue;
            }
            tot += e;
            if fx * fx / (u_lim * u_lim) + lam * lam * fy * fy > 1.0 || lam * lam * fx * fx + fy * fy / (v_lim * v_lim) > 1.0 {
                cut += e;
            }
            let q2 = (2.0 * std::f64::consts::PI).powi(2) * (fx * fx + fy * fy);
            let w = k * k - q2;
            if w <= 0.0 {
                // |e^{−|z|κ} − e^{iz(k − q²/2k)}|², κ = √(q² − k²).
                let damp = (-z_mm.abs() * (-w).sqrt()).exp();
                let phase = z_mm * (k - q2 / (2.0 * k));
                err += (damp * damp + 1.0 - 2.0 * damp * phase.cos()) * e;
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
    crate::plan::decide("propagator", || choose_free(u, z_mm, fft))
}

fn choose_free(u: &Field2, z_mm: f64, fft: &mut Fft2) -> (Propagator, Option<(f64, f64)>) {
    let zc = padded_critical_distance_mm(&u.grid, u.wavelength_in_medium_mm());
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

// ---------------------------------------------------------------------------
// Shared plumbing: padding, cropping, and the window checks.
// ---------------------------------------------------------------------------

/// Copy `u` into the middle of a `p[0] × p[1]` zero array, keeping its centre
/// sample at index `⌊p/2⌋`.
pub(crate) fn embed(u: &Field2, p: [usize; 2]) -> Vec<Cx> {
    let [nx, ny] = u.grid.n;
    let ox = p[0] / 2 - nx / 2;
    let oy = p[1] / 2 - ny / 2;
    let mut buf = vec![Cx::ZERO; p[0] * p[1]];
    for j in 0..ny {
        let src = &u.data[j * nx..(j + 1) * nx];
        let dst = (j + oy) * p[0] + ox;
        buf[dst..dst + nx].copy_from_slice(src);
    }
    buf
}

/// The inverse of [`embed`]: the middle `n` of a `p` array.
pub(crate) fn crop(buf: &[Cx], p: [usize; 2], n: [usize; 2]) -> Vec<Cx> {
    let ox = p[0] / 2 - n[0] / 2;
    let oy = p[1] / 2 - n[1] / 2;
    let mut out = Vec::with_capacity(n[0] * n[1]);
    for j in 0..n[1] {
        let s = (j + oy) * p[0] + ox;
        out.extend_from_slice(&buf[s..s + n[0]]);
    }
    out
}

/// The output samples `x₀ + m·Δx`, `m` in `0..n`, that lie within half a
/// `period` of the point the sum is taken about (`x₀` is measured from it):
/// `m0..m1`. A spectrum sampled at `1/period` sums to a field that repeats
/// with that period; only the period about the field's own window is the
/// field, and beyond it the sum would place the field's copies. A point on
/// the period's edge, to round-off, is kept.
pub(crate) fn within_period(x0: f64, dx: f64, n: usize, period: f64) -> (usize, usize) {
    let half = 0.5 * period;
    let lo = ((-half - x0) / dx - 1e-9).ceil();
    let hi = ((half - x0) / dx + 1e-9).floor() + 1.0;
    let clamp = |v: f64, fallback: usize| if v.is_nan() { fallback } else { v.clamp(0.0, n as f64) as usize };
    let (m0, m1) = (clamp(lo, 0), clamp(hi, n));
    (m0, m1.max(m0))
}

/// `Σ|z|²` over a buffer (multiply by the cell area for power).
pub(crate) fn sum_sq(buf: &[Cx]) -> f64 {
    buf.iter().map(|z| z.norm_sqr()).sum()
}

/// The fraction of a buffer's `Σ|z|²` that sits in the outer sixteenth of its
/// window on either axis. A periodic transform wraps whatever reaches the
/// edge onto the opposite side, so power there means the result may be
/// contaminated.
pub(crate) fn edge_fraction(buf: &[Cx], p: [usize; 2]) -> f64 {
    let total = sum_sq(buf);
    if total == 0.0 {
        return 0.0;
    }
    let bx = (p[0] / 16).max(1);
    let by = (p[1] / 16).max(1);
    let mut edge = 0.0;
    for j in 0..p[1] {
        let ej = j < by || j >= p[1] - by;
        for i in 0..p[0] {
            if ej || i < bx || i >= p[0] - bx {
                edge += buf[j * p[0] + i].norm_sqr();
            }
        }
    }
    edge / total
}

/// Warn when more than a millionth of the power reached the window edge.
pub(crate) fn check_wrap(buf: &[Cx], p: [usize; 2], report: &mut WaveReport) {
    let f = edge_fraction(buf, p);
    if f > 1e-6 {
        report.warn(SamplingWarning::WrapAround { edge_fraction: f });
    }
}
