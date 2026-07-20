//! Interface physics: vector Snell refraction, reflection, and scalar Fresnel
//! reflectance. Pure functions, f64, no scene dependencies — these are the exact
//! building blocks the reference tracer is validated against (§7).

use oc_core::{Vec2, Vec3};

/// Reflect a unit direction `d` about a unit normal `n`.
#[inline]
pub fn reflect(d: Vec2, n: Vec2) -> Vec2 {
    d - 2.0 * d.dot(n) * n
}

/// Reflect a unit direction `d` about a unit normal `n`, in 3D.
///
/// The same formula as [`reflect`]; the reflection law is dimension-free, and
/// keeping one implementation per dimension is cheaper than a generic bound the
/// tracer would have to name at every call.
#[inline]
pub fn reflect3(d: Vec3, n: Vec3) -> Vec3 {
    d - 2.0 * d.dot(n) * n
}

/// Refract `d` (unit) in 3D crossing an interface from index `n1` into `n2`.
///
/// `n` must be the unit interface normal oriented to *oppose* the incident
/// direction (`n·d < 0`). Returns the transmitted unit direction, or `None` under
/// total internal reflection. The transmitted ray lies in the plane of incidence
/// by construction: it is a combination of `d` and `n`, and those two span that
/// plane.
pub fn refract3(d: Vec3, n: Vec3, n1: f64, n2: f64) -> Option<Vec3> {
    let eta = n1 / n2;
    let cosi = -n.dot(d); // > 0 by the orientation contract
    let k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if k < 0.0 {
        return None; // TIR
    }
    Some(eta * d + (eta * cosi - k.sqrt()) * n)
}

/// Refract `d` (unit) crossing an interface from index `n1` into `n2`.
///
/// `n` must be the unit interface normal oriented to *oppose* the incident
/// direction (`n·d < 0`, i.e. pointing back toward where the ray came from).
/// Returns the transmitted unit direction, or `None` under total internal
/// reflection.
pub fn refract(d: Vec2, n: Vec2, n1: f64, n2: f64) -> Option<Vec2> {
    let eta = n1 / n2;
    let cosi = -n.dot(d); // > 0 by the orientation contract
    let k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if k < 0.0 {
        return None; // TIR
    }
    Some(eta * d + (eta * cosi - k.sqrt()) * n)
}

/// The critical angle (radians) for the `n1 → n2` interface, or `None` when
/// `n1 ≤ n2` (no TIR possible).
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 > n2 {
        Some((n2 / n1).asin())
    } else {
        None
    }
}

/// Unpolarized Fresnel reflectance for light incident at cosine `cos_i` on an
/// `n1 → n2` interface. Returns `1.0` under total internal reflection.
///
/// `cos_i` is the cosine of the angle of incidence (positive).
pub fn reflectance(cos_i: f64, n1: f64, n2: f64) -> f64 {
    let cos_i = cos_i.clamp(-1.0, 1.0).abs();
    let sin_i2 = (1.0 - cos_i * cos_i).max(0.0);
    let sin_t2 = (n1 / n2) * (n1 / n2) * sin_i2;
    if sin_t2 >= 1.0 {
        return 1.0; // TIR
    }
    let cos_t = (1.0 - sin_t2).sqrt();
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
    0.5 * (rs * rs + rp * rp)
}

/// Beer–Lambert transmittance over a path of `length` in a medium with
/// attenuation coefficient `alpha` (per unit length).
#[inline]
pub fn beer_lambert(alpha: f64, length: f64) -> f64 {
    if alpha <= 0.0 {
        1.0
    } else {
        (-alpha * length).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build an incident unit direction at angle `theta` from the surface
    /// normal, and the opposing normal, for a horizontal interface.
    fn setup(theta: f64) -> (Vec2, Vec2) {
        // Interface is the x-axis; the ray travels downward (+ into the surface).
        let d = Vec2::new(theta.sin(), -theta.cos());
        let n = Vec2::new(0.0, 1.0); // opposes d (n·d = -cos theta < 0)
        (d, n)
    }

    #[test]
    fn snell_at_a_plane() {
        // Air (1.0) -> glass (1.5), 30° incidence.
        let theta_i = 30f64.to_radians();
        let (d, n) = setup(theta_i);
        let t = refract(d, n, 1.0, 1.5).unwrap();
        // Angle of transmitted ray from the normal (-y direction).
        let cos_t = -t.dot(-n); // t·(downward normal)
        let theta_t = cos_t.acos();
        // n1 sin i = n2 sin t.
        let lhs = 1.0 * theta_i.sin();
        let rhs = 1.5 * theta_t.sin();
        assert!((lhs - rhs).abs() < 1e-12, "Snell: {lhs} vs {rhs}");
    }

    #[test]
    fn tir_beyond_critical() {
        // Glass (1.5) -> air (1.0). Critical angle asin(1/1.5) ≈ 41.81°.
        let crit = critical_angle(1.5, 1.0).unwrap();
        assert!((crit - (1.0f64 / 1.5).asin()).abs() < 1e-15);

        // Just below critical: refraction exists.
        let (d, n) = setup(crit - 1e-3);
        assert!(refract(d, n, 1.5, 1.0).is_some());
        // Just above critical: TIR.
        let (d2, n2) = setup(crit + 1e-3);
        assert!(refract(d2, n2, 1.5, 1.0).is_none());
        // Reflectance is exactly 1 past critical.
        assert!((reflectance((crit + 1e-3).cos(), 1.5, 1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn fresnel_normal_incidence() {
        // R0 = ((n1 - n2)/(n1 + n2))². Air->glass 1->1.5 gives 0.04.
        let r = reflectance(1.0, 1.0, 1.5);
        assert!((r - 0.04).abs() < 1e-12, "R0 = {r}");
    }

    #[test]
    fn fresnel_at_brewster_kills_p() {
        // Brewster angle: tan θ_B = n2/n1. Rp = 0 there, so R = Rs/2.
        let (n1, n2) = (1.0_f64, 1.5_f64);
        let theta_b = (n2 / n1).atan();
        let cos_b = theta_b.cos();
        let r = reflectance(cos_b, n1, n2);

        // Compute Rs directly to compare.
        let sin_t2 = (n1 / n2) * (n1 / n2) * theta_b.sin().powi(2);
        let cos_t = (1.0 - sin_t2).sqrt();
        let rs = ((n1 * cos_b - n2 * cos_t) / (n1 * cos_b + n2 * cos_t)).powi(2);
        assert!((r - 0.5 * rs).abs() < 1e-12, "expected Rs/2, got R={r}");
        // Sanity: Brewster angle for 1->1.5 is ≈ 56.31°.
        assert!((theta_b - 56.309_932f64.to_radians()).abs() < 1e-6);
    }

    #[test]
    fn reflect_is_specular() {
        let (d, n) = setup(PI / 6.0);
        let r = reflect(d, n);
        // Angle in == angle out about the normal.
        assert!((r.dot(n) + d.dot(n)).abs() < 1e-15);
        assert!((r.length() - 1.0).abs() < 1e-15);
    }
}
