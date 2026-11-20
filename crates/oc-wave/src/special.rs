//! Special functions for closed-form references: Bessel `J_n` and the Fresnel
//! integrals.
//!
//! These exist so the analytic tests can compute their expected values to
//! machine precision without a dependency. Definitions follow the NIST
//! Digital Library of Mathematical Functions (https://dlmf.nist.gov):
//!
//! - Bessel's integral, DLMF 10.9.2:
//!   `J_n(x) = (1/π) ∫₀^π cos(nτ − x sin τ) dτ` for integer `n`. Over a full
//!   period the integrand is smooth and periodic, so the trapezoidal rule
//!   converges geometrically; enough points past `|x|` give round-off
//!   accuracy.
//! - The Fresnel integrals, DLMF 7.2.7–7.2.8:
//!   `C(x) = ∫₀^x cos(πt²/2) dt`, `S(x) = ∫₀^x sin(πt²/2) dt`, by
//!   Gauss–Legendre quadrature on panels each spanning at most 0.5 in `t²`,
//!   so no panel holds more than a quarter cycle of the integrand.

use std::f64::consts::PI;

/// `J_n(x)` for integer `n`, by the trapezoidal rule on Bessel's integral.
pub fn bessel_j(n: i32, x: f64) -> f64 {
    // The integrand's Fourier content ends near |x| + |n|; 40 more points put
    // the aliasing error below round-off (the error decays like (ex/2m)^m).
    let m = 2 * ((x.abs() as usize) + n.unsigned_abs() as usize + 40);
    let h = 2.0 * PI / m as f64;
    let mut s = 0.0;
    for k in 0..m {
        let t = k as f64 * h;
        s += (n as f64 * t - x * t.sin()).cos();
    }
    s / m as f64
}

/// Gauss–Legendre nodes and weights on `[−1, 1]`, by Newton's method on the
/// Legendre polynomial from the Chebyshev guesses.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let pn = if n == 0 { 1.0 } else if n == 1 { z } else { p1 };
            let pm = if n == 1 { 1.0 } else { p0 };
            dp = n as f64 * (z * pn - pm) / (z * z - 1.0);
            let dz = pn / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// The Fresnel integrals `(C(x), S(x))`.
pub fn fresnel_cs(x: f64) -> (f64, f64) {
    let (nodes, weights) = gauss_legendre(16);
    let sign = x.signum();
    let a = x.abs();
    if a == 0.0 {
        return (0.0, 0.0);
    }
    // Panels with equal steps in t², so each holds the same phase change.
    let panels = ((a * a) / 0.5).ceil().max(1.0) as usize;
    let (mut c, mut s) = (0.0, 0.0);
    for p in 0..panels {
        let t0 = (a * a * p as f64 / panels as f64).sqrt();
        let t1 = (a * a * (p + 1) as f64 / panels as f64).sqrt();
        let (mid, half) = (0.5 * (t0 + t1), 0.5 * (t1 - t0));
        for (xi, wi) in nodes.iter().zip(&weights) {
            let t = mid + half * xi;
            let ph = 0.5 * PI * t * t;
            c += wi * half * ph.cos();
            s += wi * half * ph.sin();
        }
    }
    (sign * c, sign * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        let (x, w) = gauss_legendre(16);
        // ∫₋₁¹ t^30 dt = 2/31, the highest degree 16 points integrate exactly.
        let v: f64 = x.iter().zip(&w).map(|(t, wi)| wi * t.powi(30)).sum();
        assert!((v - 2.0 / 31.0).abs() < 1e-15);
        assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-15);
    }
}
