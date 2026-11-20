//! Analytic fields to launch, and to test against.
//!
//! # The Gaussian beam
//!
//! H. Kogelnik and T. Li, "Laser beams and resonators," *Appl. Opt.* 5(10),
//! 1550–1567 (1966). Their fundamental mode, written in this kernel's
//! `exp(−iωt)` convention (they use `exp(+iωt)`, which conjugates every
//! phase), is
//!
//! ```text
//! U(r, z) = A · (q₀/q) · exp(ik r² / 2q) · exp(ikz),   q = z − i z_R,  q₀ = −i z_R,
//! z_R = π w₀² n / λ.
//! ```
//!
//! Expanding `1/q` recovers their radius, curvature and Gouy phase:
//! `w(z) = w₀√(1 + (z/z_R)²)`, `R(z) = z(1 + (z_R/z)²)`, and
//! `q₀/q = (w₀/w) exp(−i arctan(z/z_R))`. `A = √(2P/(πw₀²))` normalizes the
//! beam to carry power `P`, since `∬|U|² = A²πw₀²/2`.
//!
//! It is a paraxial solution: it satisfies the paraxial wave equation, not the
//! Helmholtz equation, so an exact propagator moves it differently by a
//! relative amount of order `(λ/πw₀)²`, the square of its divergence.

use crate::{Cx, Field2, Grid2, WaveError};
use std::f64::consts::PI;

/// Rayleigh range `π w₀² n / λ`, mm.
pub fn rayleigh_range_mm(w0_mm: f64, lambda_um: f64, n: f64) -> f64 {
    PI * w0_mm * w0_mm * n / (lambda_um / 1000.0)
}

/// `1/e²` intensity radius at `z` from the waist, mm.
pub fn gaussian_radius_mm(w0_mm: f64, lambda_um: f64, n: f64, z_mm: f64) -> f64 {
    let zr = rayleigh_range_mm(w0_mm, lambda_um, n);
    w0_mm * (1.0 + (z_mm / zr).powi(2)).sqrt()
}

/// Wavefront radius of curvature at `z` from the waist, mm (infinite at the
/// waist).
pub fn gaussian_curvature_mm(w0_mm: f64, lambda_um: f64, n: f64, z_mm: f64) -> f64 {
    let zr = rayleigh_range_mm(w0_mm, lambda_um, n);
    if z_mm == 0.0 {
        f64::INFINITY
    } else {
        z_mm * (1.0 + (zr / z_mm).powi(2))
    }
}

/// Gouy phase `arctan(z/z_R)`, rad.
pub fn gouy_phase_rad(w0_mm: f64, lambda_um: f64, n: f64, z_mm: f64) -> f64 {
    (z_mm / rayleigh_range_mm(w0_mm, lambda_um, n)).atan()
}

/// A Gaussian beam: its waist, wavelength, medium, power, and the axis
/// through `center` it travels along.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianBeam {
    pub w0_mm: f64,
    pub lambda_um: f64,
    pub n: f64,
    pub power: f64,
    pub center: [f64; 2],
}

impl GaussianBeam {
    pub fn new(w0_mm: f64, lambda_um: f64, n: f64, power: f64) -> Self {
        Self {
            w0_mm,
            lambda_um,
            n,
            power,
            center: [0.0, 0.0],
        }
    }

    pub fn rayleigh_range_mm(&self) -> f64 {
        rayleigh_range_mm(self.w0_mm, self.lambda_um, self.n)
    }

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

    /// The beam sampled on `grid`, `z_mm` from its waist (negative before it).
    pub fn sample(&self, grid: Grid2, z_mm: f64) -> Result<Field2, WaveError> {
        if !(self.w0_mm.is_finite()
            && self.w0_mm > 0.0
            && self.power.is_finite()
            && self.power >= 0.0
            && z_mm.is_finite())
        {
            return Err(WaveError::OutOfRange(format!(
                "Gaussian beam w₀ = {} mm, P = {}, z = {z_mm} mm",
                self.w0_mm, self.power
            )));
        }
        let u = Field2::from_fn(grid, self.lambda_um, |x, y| self.value(x, y, z_mm))?;
        u.in_medium(self.n)
    }
}

/// A sampled Gaussian beam, `z_mm` from its waist (negative before it). The
/// shorthand for [`GaussianBeam::sample`].
#[allow(clippy::too_many_arguments)]
pub fn gaussian(
    grid: Grid2,
    lambda_um: f64,
    n: f64,
    w0_mm: f64,
    z_mm: f64,
    power: f64,
    center: [f64; 2],
) -> Result<Field2, WaveError> {
    GaussianBeam {
        w0_mm,
        lambda_um,
        n,
        power,
        center,
    }
    .sample(grid, z_mm)
}

/// A uniform plane wave of irradiance `irradiance` travelling with direction
/// cosines `(α, β)` in the plane's frame. The tilt is held as the analytic
/// carrier, so any angle is sampled exactly.
pub fn plane_wave(
    grid: Grid2,
    lambda_um: f64,
    n: f64,
    dir: [f64; 2],
    irradiance: f64,
) -> Result<Field2, WaveError> {
    if !(dir[0].hypot(dir[1]) < 1.0 && irradiance >= 0.0 && irradiance.is_finite()) {
        return Err(WaveError::OutOfRange(format!(
            "plane wave direction {dir:?} must be propagating; irradiance {irradiance}"
        )));
    }
    let a = irradiance.sqrt();
    let mut u = Field2::from_fn(grid, lambda_um, |_, _| Cx::real(a))?.in_medium(n)?;
    u.carrier = dir;
    Ok(u)
}

/// A diverging spherical wave from a point `z_mm` behind the plane, on the
/// axis through `c`: `exp(ikr)/r`, scaled to unit amplitude on the axis.
/// Its phase steepens away from the axis, so it is only sampled where
/// `x/(λr) < 1/(2Δx)`; the caller's report should carry that check.
pub fn point_source(
    grid: Grid2,
    lambda_um: f64,
    n: f64,
    z_mm: f64,
    c: [f64; 2],
) -> Result<Field2, WaveError> {
    if !(z_mm.is_finite() && z_mm > 0.0) {
        return Err(WaveError::OutOfRange(format!("point source distance {z_mm} mm")));
    }
    let k = 2.0 * PI * n / (lambda_um / 1000.0);
    let u = Field2::from_fn(grid, lambda_um, |x, y| {
        let r = ((x - c[0]).powi(2) + (y - c[1]).powi(2) + z_mm * z_mm).sqrt();
        Cx::expi(k * (r - z_mm)) * (z_mm / r)
    })?;
    u.in_medium(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gaussian_carries_the_power_it_was_given_and_has_its_radius() {
        let g = Grid2::square(512, 0.004).unwrap();
        let u = gaussian(g, 0.6328, 1.0, 0.2, 0.0, 2.5, [0.0, 0.0]).unwrap();
        assert!((u.power() - 2.5).abs() < 1e-10);
        // On the waist the field is real and |U|² falls by e⁻² at r = w₀.
        let i0 = u.at(256, 256).norm_sqr();
        let iw = GaussianBeam::new(0.2, 0.6328, 1.0, 2.5).value(0.2, 0.0, 0.0).norm_sqr();
        assert!((iw / i0 - (-2.0f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn the_q_parameter_carries_kogelnik_and_lis_gouy_phase() {
        // Gouy phase at z = z_R is π/4 behind the plane wave e^{ikz}.
        let (w0, lam) = (0.1, 0.5);
        let zr = rayleigh_range_mm(w0, lam, 1.0);
        let k = 2.0 * PI / (lam / 1000.0);
        let v = GaussianBeam::new(w0, lam, 1.0, 1.0).value(0.0, 0.0, zr);
        let gouy = (v * Cx::expi(-k * zr)).arg();
        assert!((gouy + PI / 4.0).abs() < 1e-12, "{gouy}");
        assert!((gouy_phase_rad(w0, lam, 1.0, zr) - PI / 4.0).abs() < 1e-15);
    }
}
