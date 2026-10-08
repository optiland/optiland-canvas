//! A ruled or holographic diffraction grating on a flat surface.
//!
//! The grooves run along the surface's local `Y`, so the grating vector lies
//! along local `X`, with `lines_per_mm` grooves per millimetre. A ray arriving
//! with direction cosines `(dx, dy, dz)` in the surface frame leaves in order
//! `m` with
//!
//! ```text
//! dx' = dx + m λ / d,   dy' = dy,   dz' = ±√(1 − dx'² − dy'²)
//! ```
//!
//! which is the grating equation `sin β = sin α + m λ / d` written for any
//! ray, in or out of the plane perpendicular to the grooves (conical
//! diffraction included). Both sides of the surface are in the same medium,
//! as for every zero-thickness surface, so `λ` is the wavelength in that
//! medium, which in air is the air wavelength every table quotes. A
//! reflection grating sends `dz'` back to the side the ray came from; a
//! transmission grating carries it on. When `dx'² + dy'² > 1` the order is
//! evanescent: it carries no light.
//!
//! The directions are exact. How much light goes into each order is not
//! computed: a real grating's efficiency depends on the groove profile, the
//! wavelength and the polarization, and needs a rigorous electromagnetic
//! calculation this kernel does not do. It is declared instead, one number or
//! a table in wavelength for the working order and one for the zero order
//! (the specular reflection, or the straight-through beam), and the rest,
//! which in a real grating goes into the other orders, is absorbed. The
//! zero order is traced because in an instrument it is a bright white beam
//! that has to go somewhere.
//!
//! A ruled metal grating treats the two polarizations differently, by tens of
//! percent near its blaze and with sharp anomalies elsewhere. The declared
//! numbers can say so: one pair for light polarized along the grooves (TE)
//! and one across them (TM). The f64 tracer then splits each ray's Stokes
//! state on the grooves, and the light leaving is partly polarized, which the
//! mirrors after it see. The f32 preview takes the mean of the two.

use crate::scene3::{Interpolation, OutOfRangePolicy, SpectralScalar};
use crate::vec3::Vec3;

/// The efficiency unpolarized light sees, the mean of a TE and a TM curve:
/// a constant when both are, else a table on both curves' wavelengths, which
/// a linear table reproduces exactly between them.
pub fn unpolarized(te: &SpectralScalar, tm: &SpectralScalar) -> SpectralScalar {
    let grid = |x: &SpectralScalar| match x {
        SpectralScalar::Constant(_) => Vec::new(),
        SpectralScalar::Tabulated { lambda_um, .. } => lambda_um.clone(),
    };
    let mut lambda_um = [grid(te), grid(tm)].concat();
    if lambda_um.is_empty() {
        let at = |x: &SpectralScalar| x.sample(0.55).unwrap_or(0.0);
        return SpectralScalar::Constant(0.5 * (at(te) + at(tm)));
    }
    lambda_um.sort_by(f64::total_cmp);
    lambda_um.dedup();
    let values = lambda_um
        .iter()
        .map(|&l| 0.5 * (te.sample(l).unwrap_or(0.0) + tm.sample(l).unwrap_or(0.0)))
        .collect();
    SpectralScalar::Tabulated {
        lambda_um,
        values,
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    }
}

/// The direction a ray leaves a grating in, in the grating's frame: `d` the
/// arriving direction (unit, local), `k_per_um` the order times the groove
/// density in grooves per micrometre (`m · lines_per_mm / 1000`), `lambda_um`
/// the wavelength. `None` when the order is evanescent.
pub fn diffract_local(d: Vec3, k_per_um: f64, lambda_um: f64, transmit: bool) -> Option<Vec3> {
    let tx = d.x + k_per_um * lambda_um;
    let ty = d.y;
    let t2 = tx * tx + ty * ty;
    if t2 >= 1.0 {
        return None;
    }
    let nz = (1.0 - t2).sqrt();
    // The side the ray travels toward: kept through a transmission grating,
    // turned back off a reflection grating.
    let ahead = if d.z >= 0.0 { 1.0 } else { -1.0 };
    let z = if transmit { ahead * nz } else { -ahead * nz };
    Some(Vec3::new(tx, ty, z))
}

/// `m · lines_per_mm / 1000`: the shift an order adds to a ray's direction
/// cosine across the grooves, per micrometre of wavelength.
pub fn k_per_um(lines_per_mm: f64, order: i32) -> f64 {
    order as f64 * lines_per_mm * 1e-3
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Littrow: a ray arriving at the blaze angle goes back along itself,
    /// where 2 d sin θ = m λ.
    #[test]
    fn littrow_returns_the_ray_along_itself() {
        let (lines, lambda): (f64, f64) = (1200.0, 0.5);
        let s = lambda * lines * 1e-3 / 2.0;
        // Arriving from +x toward the surface (dz < 0), tilted so that
        // sin α = -s along x.
        let d = Vec3::new(-s, 0.0, -(1.0 - s * s).sqrt());
        let out = diffract_local(d, k_per_um(lines, 1), lambda, false).unwrap();
        assert!((out + d).length() < 1e-12, "{out:?}");
    }

    /// Order 0 of a reflection grating is the mirror, and of a transmission
    /// grating the ray itself.
    #[test]
    fn the_zero_order_is_specular_or_straight() {
        let d = Vec3::new(0.3, -0.2, -(1.0f64 - 0.13).sqrt());
        let r = diffract_local(d, 0.0, 0.6, false).unwrap();
        assert!((r - Vec3::new(d.x, d.y, -d.z)).length() < 1e-15);
        let t = diffract_local(d, 0.0, 0.6, true).unwrap();
        assert!((t - d).length() < 1e-15);
    }

    /// Normal incidence on 600 lines/mm at 0.5 µm: sin β = 0.3, and past the
    /// grazing limit the order is gone.
    #[test]
    fn the_grating_equation_and_the_evanescent_limit() {
        let d = Vec3::new(0.0, 0.0, -1.0);
        let out = diffract_local(d, k_per_um(600.0, 1), 0.5, false).unwrap();
        assert!((out.x - 0.3).abs() < 1e-15 && out.z > 0.0);
        assert!((out.length() - 1.0).abs() < 1e-15);
        assert!(diffract_local(d, k_per_um(600.0, 4), 0.5, false).is_none());
    }
}
