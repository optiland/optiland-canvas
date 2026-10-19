//! Fresnel propagation: transfer function and impulse response.
//!
//! Both are the Fresnel (paraxial) approximation of the diffraction integral
//! (Goodman, *Introduction to Fourier Optics*, 4th ed. (2017)), in the
//! kernel's `exp(−iωt)` convention:
//!
//! - transfer function `H(f) = e^{ikz} exp(−iπλz(f_x² + f_y²))`;
//! - impulse response `h(x, y) = e^{ikz}/(iλz) · exp(iπ(x² + y²)/(λz))`.
//!
//! They differ in what they sample. D. G. Voelz and M. C. Roggemann, "Digital
//! simulation of scalar optical diffraction: revisiting chirp function
//! sampling criteria and consequences," *Appl. Opt.* 48(32), 6132–6142
//! (2009), doi:10.1364/AO.48.006132, show that the transfer function is the
//! accurate one while `Δx > λz/L` and the impulse response while
//! `Δx < λz/L`, `L` being the (padded) window. Running either on the wrong side
//! is allowed, because a comparison sometimes needs it, but the report says so
//! with an `OutsideRegime` warning.
//!
//! The window is zero-padded 2× as for the angular spectrum.

use super::{check_wrap, crop, embed, sum_sq};
use crate::fft::{fftshift2, ifftshift2, next_fast_len};
use crate::propagate::asm::advance;
use crate::propagate::regime::{critical_distance_mm, voelz_sampling};
use crate::report::{Method, Sampling, SamplingWarning, StepReport};
use crate::{Cx, Fft2, Field2, Grid2, WaveError, WaveReport};
use std::f64::consts::PI;

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
