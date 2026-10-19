//! Fraunhofer propagation onto the far-field grid.
//!
//! `U₂(x₂) = e^{ikz} e^{ik|x₂|²/2z} / (iλz) · F{U₁}(x₂/λz)` (Goodman,
//! *Introduction to Fourier Optics*, 4th ed. (2017), the Fraunhofer
//! approximation), evaluated with one transform. The output pitch is
//! `λz/(PΔx)` for a window padded to `P` samples, so padding refines the far
//! field. The output grid is centred on the point the carrier direction
//! reaches, `z·(α₀, β₀)`, and carries no carrier itself.
//!
//! The approximation drops the source-plane quadratic phase
//! `exp(ik|ξ|²/2z)`. It is accurate when the Fresnel number of the source's
//! support, `a²/(λz)`, is small; above 0.1 the report says so.
//!
//! Power is conserved exactly: the scales make the transform unitary, and
//! `|λz|²` in the prefactor cancels the change of variables.

use crate::fft::{fftshift2, ifftshift2, next_fast_len};
use crate::propagate::asm::advance;
use crate::report::{Method, SamplingWarning, StepReport};
use crate::{Cx, Fft2, Field2, Grid2, WaveError, WaveReport};
use std::f64::consts::PI;

pub(super) fn run(
    u: &Field2,
    z: f64,
    pad: usize,
    fft: &mut Fft2,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    if z <= 0.0 {
        return Err(WaveError::OutOfRange(
            "a Fraunhofer step goes forward, a positive distance".into(),
        ));
    }
    if pad == 0 {
        return Err(WaveError::OutOfRange("padding factor 0".into()));
    }
    let n = u.grid.n;
    let p = [next_fast_len(pad * n[0]), next_fast_len(pad * n[1])];
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let lam = u.wavelength_in_medium_mm();
    let k = u.k_mm();

    // Fresnel number of the support, measured from the frame axis.
    let peak = u.data.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
    let mut r2max: f64 = 0.0;
    for j in 0..n[1] {
        for i in 0..n[0] {
            if u.at(i, j).norm_sqr() > 1e-12 * peak {
                let (x, y) = u.grid.xy(i, j);
                r2max = r2max.max(x * x + y * y);
            }
        }
    }
    let nf = r2max / (lam * z.abs());
    if nf > 0.1 {
        report.warn(SamplingWarning::OutsideRegime {
            method: Method::Fraunhofer,
            reason: format!("Fresnel number of the source support {nf:.3} > 0.1"),
        });
    }

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
    let mut out = u.clone();
    out.grid = g2;
    out.data = buf;
    out.carrier = [0.0, 0.0];
    out.frame = advance(u, z);
    out.opl_ref_mm = u.opl_ref_mm + u.n_medium * z;
    report.steps.push(StepReport {
        method: Method::Fraunhofer,
        z_mm: z,
        sampling: None,
        note: format!(
            "Fraunhofer onto a {}×{} far-field grid, pitch {:.6e} × {:.6e} mm; source Fresnel number {nf:.3e}",
            p[0], p[1], pitch2[0], pitch2[1]
        ),
    });
    Ok(out)
}
