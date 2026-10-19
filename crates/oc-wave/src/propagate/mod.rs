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

use crate::report::{Method, SamplingWarning, StepReport};
use crate::{Cx, Field2, Fft2, WaveError, WaveReport};

/// Which propagator to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Propagator {
    /// Chosen per step and recorded in the report: the band-limited angular
    /// spectrum up to the critical distance, where it is exact and loses
    /// nothing; beyond it, the Fresnel impulse response when its paraxial
    /// phase error is below 0.1 rad, otherwise the band-limited angular
    /// spectrum with its loss booked.
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

    let how = match how {
        Propagator::Auto => choose(u, z_mm),
        h => h,
    };
    let out = match how {
        Propagator::AngularSpectrum(opts) => asm::run(u, z_mm, opts, fft, &mut report)?,
        Propagator::FresnelTf => fresnel::run_tf(u, z_mm, fft, &mut report)?,
        Propagator::FresnelIr => fresnel::run_ir(u, z_mm, fft, &mut report)?,
        Propagator::Fraunhofer { pad } => fraunhofer::run(u, z_mm, pad, fft, &mut report)?,
        Propagator::RsDirect => rs_direct::run(u, z_mm, fft, &mut report)?,
        Propagator::Auto => unreachable!("resolved above"),
    };
    report.ledger.detected = out.power();
    report.ledger.close();
    Ok((out, report))
}

/// The [`Propagator::Auto`] rule. Decided from the sampling alone, before
/// anything runs, so the choice is reproducible and can be explained.
fn choose(u: &Field2, z_mm: f64) -> Propagator {
    let padded_width = (2 * u.grid.n[0]) as f64 * u.grid.pitch_mm[0];
    let zc = critical_distance_mm(u.grid.pitch_mm[0], padded_width, u.wavelength_in_medium_mm());
    if z_mm.abs() <= zc {
        return Propagator::AngularSpectrum(AsmOptions::default());
    }
    let reach = 0.5 * padded_width.hypot((2 * u.grid.n[1]) as f64 * u.grid.pitch_mm[1]);
    if fresnel_phase_error_rad(reach, z_mm.abs(), u.wavelength_in_medium_mm()) < 0.1 {
        Propagator::FresnelIr
    } else {
        Propagator::AngularSpectrum(AsmOptions::default())
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
