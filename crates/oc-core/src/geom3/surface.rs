//! [`Intersect3`] for the zero-thickness surfaces: a planar shape and a surface
//! of revolution.
//!
//! A zero-thickness surface has no interior, so [`Intersect3::contains`] is
//! [`Containment::Outside`] everywhere: it never changes which medium a point is
//! in (§5.4). Its geometric normal follows the surface — `+Z` for a plane, and
//! the true surface normal (`+Z` at the vertex) for a surface of revolution,
//! which is the stable "front" side that two-sided response selection reads.

use super::primitives::{
    even_asphere_sag, planar_shape_intersect, revolution_intersect, Revolution,
};
use super::{ray_to_local, Containment, Intersect3, LocalHit3, ToleranceContext};
use crate::id::PatchId;
use crate::scene3::{PlanarShape2, RadialAperture, SagModel, SurfaceGeometry3};
use crate::vec::Vec2;
use crate::vec3::{Aabb3, Isometry3, Ray3, Vec3};

/// The `(c, k, coeffs)` a [`SagModel`] contributes to a surface of revolution.
fn sag_params(sag: &SagModel) -> (f64, f64, [f64; 8]) {
    match sag {
        SagModel::Plane => (0.0, 0.0, [0.0; 8]),
        SagModel::Conic {
            curvature_per_mm,
            conic,
        } => (*curvature_per_mm, *conic, [0.0; 8]),
        SagModel::EvenAsphere {
            curvature_per_mm,
            conic,
            coeffs,
        } => (*curvature_per_mm, *conic, *coeffs),
    }
}

/// A surface of revolution about local `+Z`, from a sag model and radial
/// aperture. The vertex sits at the local origin; sag runs along `+Z`.
fn revolution_of(sag: &SagModel, aperture: &RadialAperture) -> Revolution {
    let (c, k, coeffs) = sag_params(sag);
    Revolution {
        vertex: Vec3::ZERO,
        axis: Vec3::Z,
        c,
        k,
        coeffs,
        inner_r: aperture.inner_radius_mm,
        outer_r: aperture.outer_radius_mm,
    }
}

/// Conservative `XY` bounds of a planar shape, at `z = 0`.
fn planar_shape_bounds(shape: &PlanarShape2) -> (Vec2, Vec2) {
    match shape {
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => (
            Vec2::new(-half_width_mm, -half_height_mm),
            Vec2::new(*half_width_mm, *half_height_mm),
        ),
        PlanarShape2::Disc { radius_mm } => (Vec2::splat(-radius_mm), Vec2::splat(*radius_mm)),
        PlanarShape2::Annulus { outer_mm, .. } => (Vec2::splat(-outer_mm), Vec2::splat(*outer_mm)),
        PlanarShape2::Polygon { vertices } => {
            let mut lo = Vec2::splat(f64::INFINITY);
            let mut hi = Vec2::splat(f64::NEG_INFINITY);
            for v in vertices {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
            if lo.x > hi.x {
                (Vec2::ZERO, Vec2::ZERO)
            } else {
                (lo, hi)
            }
        }
    }
}

fn local_aabb(geom: &SurfaceGeometry3) -> Aabb3 {
    match geom {
        SurfaceGeometry3::Plane { shape } => {
            let (lo, hi) = planar_shape_bounds(shape);
            Aabb3 {
                min: Vec3::new(lo.x, lo.y, 0.0),
                max: Vec3::new(hi.x, hi.y, 0.0),
            }
        }
        SurfaceGeometry3::SurfaceOfRevolution {
            sag,
            radial_aperture,
        } => {
            let (c, k, coeffs) = sag_params(sag);
            let outer = radial_aperture.outer_radius_mm;
            let sag_edge = even_asphere_sag(c, k, &coeffs, outer);
            Aabb3 {
                min: Vec3::new(-outer, -outer, sag_edge.min(0.0)),
                max: Vec3::new(outer, outer, sag_edge.max(0.0)),
            }
        }
    }
}

fn transform_local_aabb(local: Aabb3, transform: &Isometry3) -> Aabb3 {
    let mut bb = Aabb3::EMPTY;
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { local.min.x } else { local.max.x },
            if i & 2 == 0 { local.min.y } else { local.max.y },
            if i & 4 == 0 { local.min.z } else { local.max.z },
        );
        bb.expand(transform.transform_point(corner));
    }
    bb
}

impl Intersect3 for SurfaceGeometry3 {
    fn aabb_world(&self, transform: &Isometry3) -> Aabb3 {
        transform_local_aabb(local_aabb(self), transform)
    }

    fn intersect(
        &self,
        ray_world: &Ray3,
        transform: &Isometry3,
        t_min_mm: f64,
        t_max_mm: f64,
        out: &mut dyn FnMut(LocalHit3),
    ) {
        let ray = ray_to_local(transform, ray_world);
        let mut emit = |raw: super::primitives::RawHit3| {
            out(LocalHit3 {
                t_mm: raw.t_mm,
                point_local: raw.point,
                normal_local: raw.normal,
                patch: PatchId(0),
                uv: raw.uv,
            });
        };
        match self {
            SurfaceGeometry3::Plane { shape } => {
                planar_shape_intersect(shape, &ray, t_min_mm, t_max_mm, &mut emit)
            }
            SurfaceGeometry3::SurfaceOfRevolution {
                sag,
                radial_aperture,
            } => {
                let surf = revolution_of(sag, radial_aperture);
                revolution_intersect(&surf, &ray, t_min_mm, t_max_mm, &mut emit)
            }
        }
    }

    /// A surface has no interior. Always [`Containment::Outside`].
    fn contains(
        &self,
        _point: Vec3,
        _transform: &Isometry3,
        _tol: &ToleranceContext,
    ) -> Containment {
        Containment::Outside
    }
}
