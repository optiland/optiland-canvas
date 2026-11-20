//! The zoomed Fourier transform: any output window, any pitch.
//!
//! An FFT returns the spectrum on one fixed grid, `1/(NΔx)` apart. A focal
//! spot or a diffraction order often needs a few hundred samples over a small
//! patch of that spectrum, finer than the FFT's grid. The chirp-z transform
//! computes
//!
//! ```text
//! X(f_m) = Σ_n x_n exp(−i2π f_m x_n),   f_m = f₀ + m·δf,   x_n = x₀ + n·Δx,
//! ```
//!
//! for any `f₀`, `δf` and number of outputs, at FFT cost, by Bluestein's
//! identity `mn = (m² + n² − (m − n)²)/2`, which turns the sum into a
//! convolution with a chirp (L. R. Rabiner, R. W. Schafer, C. M. Rader, "The
//! chirp z-transform algorithm," *IEEE Trans. Audio Electroacoust.* 17(2),
//! 86–92 (1969), doi:10.1109/TAU.1969.1162034; L. I. Bluestein, *IEEE Trans.
//! Audio Electroacoust.* 18(4), 451–455 (1970)). Y. Hu et al., *Light Sci.
//! Appl.* 9, 119 (2020), doi:10.1038/s41377-020-00362-z, use it for exactly
//! this purpose in diffraction; a published Comment (Shao & Urbach 2021,
//! doi:10.1038/s41377-020-00447-9) disputes its novelty for scalar fields, not
//! its correctness.
//!
//! In 2D the transform is separable: one pass along `x`, one along `y`.

use crate::fft::next_fast_len;
use crate::propagate::asm::advance;
use crate::report::{Method, SamplingWarning, StepReport};
use crate::{Cx, Fft2, Field2, Grid2, WaveError, WaveReport};
use std::f64::consts::PI;

/// The sampled continuous Fourier transform `Σ x_n e^{−i2π f_m x_n} Δx` of one
/// line of samples at positions `x₀ + nΔx`, at `m_out` frequencies
/// `f₀ + m·δf`, by Bluestein's chirp convolution.
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

/// The Fraunhofer field on an arbitrary output grid `out`, `z_mm` ahead:
/// `U₂ = e^{ikz} e^{ik|x₂|²/2z}/(iλz) · F{U₁}(x₂/λz)`, the transform taken by
/// [`czt2`] so the output window and pitch are free.
///
/// Fraunhofer's caveat is the same as for the FFT version: above a source
/// Fresnel number of 0.1 the report says so. The output samples only part of
/// the far field, so the power it holds is not the input power; the ledger
/// books the rest as `left_window`.
pub fn fraunhofer_zoom(
    u: &Field2,
    z_mm: f64,
    out: Grid2,
    fft: &mut Fft2,
) -> Result<(Field2, WaveReport), WaveError> {
    if !(z_mm.is_finite() && z_mm > 0.0) {
        return Err(WaveError::OutOfRange(
            "a Fraunhofer step goes forward, a positive distance".into(),
        ));
    }
    out.validate()?;
    let lam = u.wavelength_in_medium_mm();
    let k = u.k_mm();
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();

    let peak = u.data.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
    let mut r2max: f64 = 0.0;
    for j in 0..u.grid.n[1] {
        for i in 0..u.grid.n[0] {
            if u.at(i, j).norm_sqr() > 1e-12 * peak {
                let (x, y) = u.grid.xy(i, j);
                r2max = r2max.max(x * x + y * y);
            }
        }
    }
    let nf = r2max / (lam * z_mm);
    if nf > 0.1 {
        report.warn(SamplingWarning::OutsideRegime {
            method: Method::Fraunhofer,
            reason: format!("Fresnel number of the source support {nf:.3} > 0.1"),
        });
    }

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
    report.steps.push(StepReport {
        method: Method::Fraunhofer,
        z_mm,
        sampling: None,
        note: format!(
            "Fraunhofer by chirp-z onto a {}×{} window, pitch {:.4e} × {:.4e} mm; source Fresnel number {nf:.3e}",
            out.n[0], out.n[1], out.pitch_mm[0], out.pitch_mm[1]
        ),
    });
    Ok((v, report))
}

/// The Fresnel field on an arbitrary output grid `out`, `z_mm` ahead: the
/// Fresnel integral as one transform, `U₂ = e^{ikz}e^{ik|x₂|²/2z}/(iλz) ·
/// F{U₁e^{ik|x₁|²/2z}}(x₂/λz)` (Goodman, the Fresnel diffraction integral
/// written as a Fourier transform), taken by [`czt2`] so the output window and
/// pitch are free: a beam that has left a focus can be followed onto a window
/// many times the one it came from. Paraxial, with the error of the
/// paraxial propagator (`crate::propagate::paraxial_error`); the input chirp
/// is sampled while `z` exceeds the input's critical distance, and the report
/// says where it is not.
pub fn fresnel_zoom(u: &Field2, z_mm: f64, out: Grid2, fft: &mut Fft2) -> Result<(Field2, WaveReport), WaveError> {
    if !(z_mm.is_finite() && z_mm > 0.0) {
        return Err(WaveError::OutOfRange("a Fresnel step onto a new grid goes forward, a positive distance".into()));
    }
    out.validate()?;
    let lam = u.wavelength_in_medium_mm();
    let k = u.k_mm();
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();
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
    let mut v = u.clone();
    v.grid = out;
    v.data = data;
    v.carrier = [0.0, 0.0];
    v.frame = advance(u, z_mm);
    v.opl_ref_mm = u.opl_ref_mm + u.n_medium * z_mm;
    report.ledger.detected = v.power();
    report.ledger.left_window = (report.ledger.launched - report.ledger.detected).max(0.0);
    report.ledger.close();
    report.steps.push(StepReport {
        method: Method::FresnelIr,
        z_mm,
        sampling: None,
        note: format!(
            "Fresnel by chirp-z onto a {}×{} window, pitch {:.4e} × {:.4e} mm",
            out.n[0], out.n[1], out.pitch_mm[0], out.pitch_mm[1]
        ),
    });
    Ok((v, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn czt_line_is_the_direct_sum() {
        let x: Vec<Cx> = (0..37).map(|k| Cx::new((k as f64 * 0.7).sin(), (k as f64).cos())).collect();
        let (x0, dx, f0, df, m) = (-0.3, 0.013, -11.0, 0.37, 53);
        let got = czt_line(&x, x0, dx, f0, df, m, &mut Fft2::new());
        for (mi, g) in got.iter().enumerate() {
            let f = f0 + mi as f64 * df;
            let want: Cx = x
                .iter()
                .enumerate()
                .map(|(n, v)| *v * Cx::expi(-2.0 * PI * f * (x0 + n as f64 * dx)) * dx)
                .sum();
            assert!((*g - want).abs() < 1e-12, "m = {mi}: {g:?} vs {want:?}");
        }
    }
}
