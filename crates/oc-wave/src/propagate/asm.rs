//! The angular spectrum method, band-limited after Matsushima & Shimobaba.
//!
//! The field is decomposed into plane waves, each is advanced by
//! `H = exp(i2πz√(1/λ² − f_x² − f_y²))` (λ in the medium), and the waves are
//! summed again. That is the Rayleigh–Sommerfeld solution for a homogeneous
//! medium with no approximation in `z` (Goodman, *Introduction to Fourier
//! Optics*, 4th ed. (2017), the angular spectrum and its propagation).
//!
//! What is approximate is the sampling. The transform is periodic, so the
//! field is zero-padded (`pad` times its samples, 2 by default) to turn the
//! circular convolution into a linear one over the original window. And at
//! long distances the transfer function's phase changes faster than its
//! samples can follow, which aliases. K. Matsushima and T. Shimobaba, "Band-
//! limited angular spectrum method for numerical simulation of free-space
//! propagation in far and near fields," *Opt. Express* 17(22), 19662–19673
//! (2009), doi:10.1364/OE.17.019662, limit the bandwidth to where it does not:
//!
//! - Eq. (13): `u_limit = 1 / (λ √((2Δu z)² + 1))`, `Δu = 1/(2S)` for a
//!   zero-padded window `2S` wide (here `1/(PΔx)` for `P` padded samples);
//!   Eq. (20) is the same in `v`;
//! - Eqs. (18)–(19): the transfer function is kept inside both ellipses
//!   `u²/u_limit² + λ²v² ≤ 1` and `λ²u² + v²/v_limit² ≤ 1`.
//!
//! The equations were read from the paper's text (Kansai University
//! repository copy, retrieved 2026-10-19). The power the limit removes is
//! booked in the ledger and reported; it is never renormalized back in.
//!
//! An evanescent component (`f² > 1/λ²`) decays as `exp(−2π|z|√(f² − 1/λ²))`
//! going forward. Going backward it would grow, which no physical field does,
//! so it is removed and booked instead.

use super::{check_wrap, crop, embed, sum_sq};
use crate::fft::{fftshift2, ifftshift2, next_fast_len};
use crate::report::{Method, SamplingWarning, StepReport};
use crate::{Cx, Fft2, Field2, Grid2, WaveError, WaveReport};
use std::f64::consts::PI;

/// Options for the angular spectrum method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsmOptions {
    /// Apply the Matsushima–Shimobaba band limit.
    pub band_limit: bool,
    /// Zero padding factor: 2 makes the convolution linear over the window;
    /// 1 keeps it circular, which is exact for a field periodic in its
    /// window (a grating's Talbot carpet).
    pub pad: usize,
}

impl Default for AsmOptions {
    fn default() -> Self {
        Self {
            band_limit: true,
            pad: 2,
        }
    }
}

pub(super) fn run(
    u: &Field2,
    z: f64,
    opts: AsmOptions,
    fft: &mut Fft2,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    if opts.pad == 0 {
        return Err(WaveError::OutOfRange("padding factor 0".into()));
    }
    let n = u.grid.n;
    let p = if opts.pad == 1 {
        n
    } else {
        [next_fast_len(opts.pad * n[0]), next_fast_len(opts.pad * n[1])]
    };
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;

    let mut buf = if opts.pad == 1 { u.data.clone() } else { embed(u, p) };
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());

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
    report.steps.push(StepReport {
        method: Method::AngularSpectrum {
            band_limited: opts.band_limit,
        },
        z_mm: z,
        sampling: None,
        note: format!(
            "angular spectrum on a {}×{} grid padded to {}×{}{}",
            n[0],
            n[1],
            p[0],
            p[1],
            if opts.band_limit {
                format!("; band limit |u| ≤ {u_lim:.4} /mm, |v| ≤ {v_lim:.4} /mm")
            } else {
                String::new()
            }
        ),
    });
    Ok(out)
}

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
    // At `z = 0` it is the band-limited interpolation of the field onto
    // `out`, by the same sum.
    if !(z.is_finite() && z >= 0.0) {
        return Err(WaveError::OutOfRange("a zoomed angular spectrum goes forward, a positive distance or none".into()));
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
    // The output coordinates: the transform was taken about the input
    // window's centre `c` (`ifftshift2` puts it at the origin), so the
    // plane-wave sum is evaluated at `x − c`; `out`'s samples sit at
    // out.coord(axis, m) in the frame.
    let (mx, my) = (out.n[0], out.n[1]);
    let c = u.grid.center_mm;
    let (x0, y0) = (out.coord(0, 0) - c[0], out.coord(1, 0) - c[1]);
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
