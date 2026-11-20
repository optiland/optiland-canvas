//! Scattering of a plane wave by a homogeneous sphere: the Mie solution, as a
//! reference for the wave path's curved-surface approximation (V03 of the
//! expert review, the bundle's B30 and B40).
//!
//! Conventions, those of C. F. Bohren and D. R. Huffman, *Absorption and
//! Scattering of Light by Small Particles* (Wiley, 1983), ch. 4: time
//! dependence `e^{−iωt}`; the incident wave travels along `+z` with `E` along
//! `+x` and unit amplitude; `θ` from `+z`, `φ` from `+x` toward `+y`; the
//! size parameter `x = kR` and the relative index `m`, both in the medium.
//! `H` is in units where the incident wave's `H` is 1 along `+y`: the SI
//! field times `ωμ/k` (`η₀/n` in a nonmagnetic medium of index `n`).
//!
//! The coefficients (BH Eq. 4.53) are written through logarithmic
//! derivatives, every one by the recurrence in which it is stable:
//! `D_n(z) = ψ_n′(z)/ψ_n(z)` downward (BH §4.8), `G_n(x) = ξ_n′(x)/ξ_n(x)`
//! upward, and the ratio `R_n = ψ_n(x)/ξ_n(x)` upward from those two:
//!
//! ```text
//! a_n = R_n (D_n(mx)/m − D_n(x)) / (D_n(mx)/m − G_n(x))
//! b_n = R_n (m D_n(mx) − D_n(x)) / (m D_n(mx) − G_n(x))
//! ```
//!
//! So orders far past `x`, whose coefficients are tiny but multiply Hankel
//! functions that grow fast near the sphere, are as accurate as the first
//! ones; an upward recurrence of `ψ_n` (BHMIE) is not past `x`. The series
//! starts at Wiscombe's `⌈x + 4x^{1/3} + 2⌉` terms (W. J. Wiscombe, NCAR
//! Technical Note TN-140+STR, 1979) and grows by 16 until it converges.

use crate::{Cx, WaveError};

/// A sphere's Mie coefficients `a_n`, `b_n`, `n = 1..=N`.
#[derive(Debug, Clone)]
pub struct MieSphere {
    /// Relative index `m` (complex: absorption in its imaginary part, `> 0`).
    pub m: Cx,
    /// Size parameter `x = kR`, `k` the wavenumber in the medium.
    pub x: f64,
    /// `a_n`, `b_n` for `n = 1..=a.len()`.
    pub a: Vec<Cx>,
    pub b: Vec<Cx>,
}

/// Wiscombe's number of terms, `⌈x + 4x^{1/3} + 2⌉`.
pub fn wiscombe_terms(x: f64) -> usize {
    (x + 4.0 * x.cbrt() + 2.0).ceil() as usize
}

/// `D_n(z) = ψ_n′(z)/ψ_n(z)` for `n = 0..=n_max`, by downward recurrence
/// (BH §4.8) from far enough above `n_max` and `|z|`. The start's error
/// shrinks by `(ψ_n/ψ_{n−1})²` a step, which past `|z|` falls only as
/// `exp(−(2√2/3)Δ^{3/2}/|z|^{1/2})` over `Δ` orders (the Airy regime), so a
/// fixed margin of 15 or 16 (BHMIE's) leaves `5e-5` at `|z| = 150` and more
/// beyond; `16 + 8|z|^{1/3}` orders take it to round-off at any size.
fn log_derivative_down(z: Cx, n_max: usize) -> Vec<Cx> {
    let start = n_max.max(z.abs().ceil() as usize) + 16 + (8.0 * z.abs().cbrt()).ceil() as usize;
    let mut d = vec![Cx::ZERO; start + 1];
    for n in (1..=start).rev() {
        let nz = Cx::real(n as f64) / z;
        d[n - 1] = nz - Cx::ONE / (d[n] + nz);
    }
    d.truncate(n_max + 1);
    d
}

impl MieSphere {
    /// The coefficients for `n_terms` orders.
    pub fn with_terms(m: Cx, x: f64, n_terms: usize) -> Result<Self, WaveError> {
        if !(x.is_finite() && x > 0.0 && m.is_finite() && m.abs() > 0.0 && m.im >= 0.0 && n_terms >= 1) {
            return Err(WaveError::OutOfRange(format!("a Mie sphere needs a positive size parameter and a passive index, got x = {x}, m = {m:?}")));
        }
        let n = n_terms;
        let dm = log_derivative_down(m * x, n);
        let dx = log_derivative_down(Cx::real(x), n);
        // G_n = ξ_n′/ξ_n upward from G_0 = i; R_n = ψ_n/ξ_n upward from
        // R_0 = ψ_0/ξ_0 = i sin x e^{−ix}.
        let mut g = Cx::I;
        let mut r = Cx::I * Cx::expi(-x) * x.sin();
        let (mut a, mut b) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for k in 1..=n {
            let nx = k as f64 / x;
            let g_prev = g;
            g = Cx::ONE / (Cx::real(nx) - g_prev) - Cx::real(nx);
            // ψ_{n−1}/ψ_n = D_n + n/x, ξ_{n−1}/ξ_n = G_n + n/x.
            r = r * (g + Cx::real(nx)) / (dx[k] + Cx::real(nx));
            let dmm = dm[k] / m;
            a.push(r * (dmm - dx[k]) / (dmm - g));
            let mdm = m * dm[k];
            b.push(r * (mdm - dx[k]) / (mdm - g));
        }
        Ok(MieSphere { m, x, a, b })
    }

    /// The coefficients, the series grown by 16 terms from Wiscombe's start
    /// until the scattering and extinction efficiencies move by less than
    /// `tol` relative, at most eight times; refused by name if it has not
    /// converged by then.
    pub fn new(m: Cx, x: f64, tol: f64) -> Result<Self, WaveError> {
        let mut n = wiscombe_terms(x);
        let mut prev = MieSphere::with_terms(m, x, n)?;
        for _ in 0..8 {
            n += 16;
            let next = MieSphere::with_terms(m, x, n)?;
            let (qa, qb) = (prev.efficiencies(), next.efficiencies());
            if (qa.0 - qb.0).abs() <= tol * qb.0.abs() && (qa.1 - qb.1).abs() <= tol * qb.1.abs() {
                return Ok(next);
            }
            prev = next;
        }
        Err(WaveError::OutOfRange(format!("the Mie series for x = {x}, m = {m:?} has not converged to {tol} after eight increments of 16 terms")))
    }

    /// `(Q_ext, Q_sca)`: the extinction and scattering cross sections over
    /// the geometric one, `πR²` (BH Eqs. 4.61, 4.62).
    pub fn efficiencies(&self) -> (f64, f64) {
        let (mut ext, mut sca) = (0.0, 0.0);
        for (k, (a, b)) in self.a.iter().zip(&self.b).enumerate() {
            let w = (2 * (k + 1) + 1) as f64;
            ext += w * (a.re + b.re);
            sca += w * (a.norm_sqr() + b.norm_sqr());
        }
        let s = 2.0 / (self.x * self.x);
        (s * ext, s * sca)
    }

    /// The scattering amplitudes `S₁(θ)`, `S₂(θ)` (BH Eq. 4.74).
    pub fn amplitudes(&self, theta: f64) -> (Cx, Cx) {
        let (pi, tau) = angular(self.a.len(), theta.cos());
        let (mut s1, mut s2) = (Cx::ZERO, Cx::ZERO);
        for k in 0..self.a.len() {
            let n = (k + 1) as f64;
            let w = (2.0 * n + 1.0) / (n * (n + 1.0));
            s1 += (self.a[k] * pi[k] + self.b[k] * tau[k]) * w;
            s2 += (self.a[k] * tau[k] + self.b[k] * pi[k]) * w;
        }
        (s1, s2)
    }

    /// The total field (incident plus scattered) at a point outside the
    /// sphere, `(x, y, z)` in units of `1/k` (that is, `kr` for the radius),
    /// in Cartesian components: `(E, H)` (BH Eq. 4.45 with the vector
    /// spherical harmonics of Eq. 4.50, `h_n^{(1)}` their radial function).
    /// Refused inside the sphere, where the internal field holds.
    pub fn exterior_field(&self, p: [f64; 3]) -> Result<([Cx; 3], [Cx; 3]), WaveError> {
        let rho = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if rho < self.x * (1.0 - 1e-12) {
            return Err(WaveError::OutOfRange(format!("the point at kr = {rho} lies inside the sphere of kR = {}", self.x)));
        }
        let theta = (p[2] / rho).clamp(-1.0, 1.0).acos();
        let phi = p[1].atan2(p[0]);
        let n = self.a.len();
        let (pi, tau) = angular(n, theta.cos());
        // ξ_n(ρ) = ρ h_n(ρ), upward from ξ_0 = −i e^{iρ}, ξ_{−1} = e^{iρ}
        // (dominant past ρ, so upward is stable), and ξ_n′ = ξ_{n−1} − n ξ_n/ρ.
        let mut xi_prev = Cx::expi(rho);
        let mut xi = Cx::new(0.0, -1.0) * Cx::expi(rho);
        let (sp, cp, st) = (phi.sin(), phi.cos(), theta.sin());
        // Spherical components of E and H scattered.
        let (mut er, mut et, mut ep) = (Cx::ZERO, Cx::ZERO, Cx::ZERO);
        let (mut hr, mut ht, mut hp) = (Cx::ZERO, Cx::ZERO, Cx::ZERO);
        let mut i_n = Cx::ONE;
        for k in 0..n {
            let nn = (k + 1) as f64;
            let xi_next = xi * ((2.0 * nn - 1.0) / rho) - xi_prev;
            (xi_prev, xi) = (xi, xi_next);
            let dxi = xi_prev - xi * (nn / rho);
            let z = xi / rho;
            let dz = dxi / rho;
            i_n *= Cx::I;
            let en = i_n * ((2.0 * nn + 1.0) / (nn * (nn + 1.0)));
            let (a, b) = (self.a[k], self.b[k]);
            // M_o1n, M_e1n, N_o1n, N_e1n (BH 4.50), radial function h_n.
            let mo = [Cx::ZERO, z * (cp * pi[k]), z * (-sp * tau[k])];
            let me = [Cx::ZERO, z * (-sp * pi[k]), z * (-cp * tau[k])];
            // The radial parts carry h_n(ρ)/ρ.
            let zr = z / rho;
            let no = [zr * (sp * nn * (nn + 1.0) * st * pi[k]), dz * (sp * tau[k]), dz * (cp * pi[k])];
            let ne = [zr * (cp * nn * (nn + 1.0) * st * pi[k]), dz * (cp * tau[k]), dz * (-sp * pi[k])];
            // E_s = Σ E_n (i a_n N_e1n − b_n M_o1n); H_s = Σ E_n (i b_n N_o1n + a_n M_e1n).
            let ia = Cx::I * a;
            let ib = Cx::I * b;
            er += en * (ia * ne[0] - b * mo[0]);
            et += en * (ia * ne[1] - b * mo[1]);
            ep += en * (ia * ne[2] - b * mo[2]);
            hr += en * (ib * no[0] + a * me[0]);
            ht += en * (ib * no[1] + a * me[1]);
            hp += en * (ib * no[2] + a * me[2]);
        }
        let (ct, cpp) = (theta.cos(), phi.cos());
        let to_xyz = |r: Cx, t: Cx, f: Cx| {
            [
                r * (st * cpp) + t * (ct * cpp) - f * sp,
                r * (st * sp) + t * (ct * sp) + f * cpp,
                r * ct - t * st,
            ]
        };
        let (mut e, mut h) = (to_xyz(er, et, ep), to_xyz(hr, ht, hp));
        // The incident wave, E along x and H along y.
        let inc = Cx::expi(p[2]);
        e[0] += inc;
        h[1] += inc;
        Ok((e, h))
    }
}

/// `π_n(μ)`, `τ_n(μ)` for `n = 1..=n` (BH Eq. 4.47 and the recurrences after it).
fn angular(n: usize, mu: f64) -> (Vec<f64>, Vec<f64>) {
    let (mut pi, mut tau) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut p_prev, mut p) = (0.0, 1.0);
    for k in 1..=n {
        let kk = k as f64;
        pi.push(p);
        tau.push(kk * mu * p - (kk + 1.0) * p_prev);
        let next = ((2.0 * kk + 1.0) * mu * p - (kk + 1.0) * p_prev) / kk;
        (p_prev, p) = (p, next);
    }
    (pi, tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_contrast_scatters_nothing_and_a_lossless_sphere_extinguishes_what_it_scatters() {
        // BH §4.4: at m = 1 every coefficient vanishes; for real m and x,
        // each has the form N/(N + iD), so Re a_n = |a_n|², and the
        // extinction and scattering cross sections agree (B30).
        for x in [10.0, 100.0, 1000.0] {
            let s = MieSphere::new(Cx::ONE, x, 1e-14).unwrap();
            let worst = s.a.iter().chain(&s.b).map(|c| c.abs()).fold(0.0, f64::max);
            assert!(worst < 1e-12, "x {x}: {worst}");
            let s = MieSphere::new(Cx::real(1.5), x, 1e-14).unwrap();
            let id = s.a.iter().chain(&s.b).map(|c| (c.re - c.norm_sqr()).abs()).fold(0.0, f64::max);
            let (ext, sca) = s.efficiencies();
            println!("measured: Mie, m = 1.5, x = {x}: worst |Re a − |a|²| = {id:.3e}; Q_ext − Q_sca = {:.3e}", ext - sca);
            assert!(id < 2e-10, "x {x}: {id}");
            assert!((ext - sca).abs() < 1e-10 * ext, "x {x}: {ext} {sca}");
        }
    }
}
