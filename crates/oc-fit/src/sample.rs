//! Turn a fitted [`Edge`] back into points: a polyline for drawing the fitted
//! curve under the stroke (the complexity-slider preview), and the aperture rim
//! endpoints for extruding a surface into a solid element.

use crate::fits::{asphere_sag, conic_sag};
use oc_core::{Edge, Vec2};

/// Sample `edge` into a polyline of about `n` points, in the edge's own frame
/// (the same coordinates the fit was computed in). Endpoints are included.
pub fn sample_edge(edge: &Edge, n: usize) -> Vec<Vec2> {
    let n = n.max(2);
    match edge {
        Edge::Line { a, b } => (0..n)
            .map(|i| a.lerp(*b, i as f64 / (n - 1) as f64))
            .collect(),
        Edge::Arc { center, r, a0, a1 } => (0..n)
            .map(|i| {
                let a = a0 + (a1 - a0) * i as f64 / (n - 1) as f64;
                *center + Vec2::new(a.cos(), a.sin()) * *r
            })
            .collect(),
        Edge::Conic(cp) => {
            let axis = cp.axis.normalize();
            let tang = Vec2::new(-axis.y, axis.x);
            (0..n)
                .map(|i| {
                    let y = -cp.half_ap + 2.0 * cp.half_ap * i as f64 / (n - 1) as f64;
                    cp.vertex + y * tang + conic_sag(cp.c, cp.k, y) * axis
                })
                .collect()
        }
        Edge::Asphere { base, coeffs } => {
            let axis = base.axis.normalize();
            let tang = Vec2::new(-axis.y, axis.x);
            (0..n)
                .map(|i| {
                    let y = -base.half_ap + 2.0 * base.half_ap * i as f64 / (n - 1) as f64;
                    base.vertex + y * tang + asphere_sag(base.c, base.k, coeffs, y) * axis
                })
                .collect()
        }
        Edge::Spline { ctrl, .. } => ctrl.clone(),
    }
}

/// The two aperture-rim endpoints of a surface edge and the unit axis pointing
/// **into** the curvature (toward the centre of curvature) — enough to extrude
/// the surface into a closed element body.
pub fn surface_rim(edge: &Edge) -> (Vec2, Vec2, Vec2) {
    match edge {
        Edge::Line { a, b } => {
            let axis = {
                let d = (*b - *a).normalize_or_zero();
                Vec2::new(-d.y, d.x)
            };
            (*a, *b, axis)
        }
        Edge::Arc { center, r, a0, a1 } => {
            let p0 = *center + Vec2::new(a0.cos(), a0.sin()) * *r;
            let p1 = *center + Vec2::new(a1.cos(), a1.sin()) * *r;
            let mid = 0.5 * (p0 + p1);
            let axis = (*center - mid).normalize_or_zero();
            (p0, p1, axis)
        }
        Edge::Conic(cp) => {
            let axis = cp.axis.normalize();
            let tang = Vec2::new(-axis.y, axis.x);
            let ha = cp.half_ap;
            let p0 = cp.vertex - ha * tang + conic_sag(cp.c, cp.k, -ha) * axis;
            let p1 = cp.vertex + ha * tang + conic_sag(cp.c, cp.k, ha) * axis;
            (p0, p1, axis)
        }
        Edge::Asphere { base, coeffs } => {
            let axis = base.axis.normalize();
            let tang = Vec2::new(-axis.y, axis.x);
            let ha = base.half_ap;
            let p0 = base.vertex - ha * tang + asphere_sag(base.c, base.k, coeffs, -ha) * axis;
            let p1 = base.vertex + ha * tang + asphere_sag(base.c, base.k, coeffs, ha) * axis;
            (p0, p1, axis)
        }
        Edge::Spline { ctrl, .. } => {
            let p0 = *ctrl.first().unwrap_or(&Vec2::ZERO);
            let p1 = *ctrl.last().unwrap_or(&Vec2::ZERO);
            let d = (p1 - p0).normalize_or_zero();
            (p0, p1, Vec2::new(-d.y, d.x))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::ConicParams;

    #[test]
    fn samples_hit_the_endpoints() {
        let e = Edge::Line {
            a: Vec2::new(-3.0, 1.0),
            b: Vec2::new(4.0, 1.0),
        };
        let pts = sample_edge(&e, 10);
        assert_eq!(pts.len(), 10);
        assert!((pts[0] - Vec2::new(-3.0, 1.0)).length() < 1e-12);
        assert!((pts[9] - Vec2::new(4.0, 1.0)).length() < 1e-12);
    }

    #[test]
    fn conic_rim_is_at_the_aperture() {
        let cp = ConicParams {
            vertex: Vec2::ZERO,
            axis: Vec2::new(1.0, 0.0),
            c: 1.0 / 20.0,
            k: 0.0,
            half_ap: 6.0,
        };
        let (p0, p1, axis) = surface_rim(&Edge::Conic(cp));
        assert!((p0.y + 6.0).abs() < 1e-12 && (p1.y - 6.0).abs() < 1e-12);
        assert!((axis - Vec2::new(1.0, 0.0)).length() < 1e-12);
        // Rim points sit forward of the vertex (positive sag along +axis).
        assert!(p0.x > 0.0 && p1.x > 0.0);
    }
}
