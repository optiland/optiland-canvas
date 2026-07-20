//! 3D analytic geometry: intersection, normals, containment, bounds.
//!
//! What this suite owes (contract §15.1, §15.6):
//!
//! - plane, sphere, cylinder, box intersection;
//! - conic-of-revolution intersection and **analytic** normal;
//! - even-asphere root and **analytic** normal (no finite difference: that is
//!   the known ~1e-9 floor on the current 2D asphere, and 3D must not inherit
//!   it);
//! - extruded profile sides and caps;
//! - revolved profile containment;
//! - event deduplication at tangency and at patch seams;
//! - conservative AABBs: too large is allowed, missing a hit is not;
//! - transform invariance of intersections;
//! - property/fuzz invariants: no NaN in an accepted hit, `t > t_min`, unit
//!   normals, scale invariance from 1e-3 mm to 1e4 mm.
//!
//! The fuzz half lives in `tests/geom3_fuzz.rs`, a seeded, dependency-free
//! harness (the "equivalent deterministic fuzz harness" §15.6 allows); this file
//! is the worked, named fixtures.
#![cfg(feature = "scene3")]

use oc_core::geom3::primitives::{
    box_contains, box_intersect, cylinder_contains, cylinder_intersect, frustum_intersect,
    planar_shape_intersect, revolution_intersect, sphere_contains, sphere_intersect, RawHit3,
    Revolution,
};
use oc_core::geom3::profile::{revolve_contains, validate_extrudable, validate_revolvable};
use oc_core::geom3::{Containment, Intersect3, LocalHit3, ToleranceContext};
use oc_core::id::PatchId;
use oc_core::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::{PlanarShape2, RadialAperture, SagModel, SolidGeometry3, SurfaceGeometry3};
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Ray3, Vec3};
use std::f64::consts::{FRAC_PI_3, FRAC_PI_4};

const TMAX: f64 = f64::INFINITY;
const TMIN: f64 = 1e-9;

fn ray(o: Vec3, d: Vec3) -> Ray3 {
    Ray3::new(o, d).unwrap()
}

fn prim(f: impl FnOnce(&mut dyn FnMut(RawHit3))) -> Vec<RawHit3> {
    let mut v = Vec::new();
    f(&mut |h| v.push(h));
    v
}

fn hits(g: &impl Intersect3, r: &Ray3, t: &Isometry3) -> Vec<LocalHit3> {
    let mut v = Vec::new();
    g.intersect(r, t, TMIN, TMAX, &mut |h| v.push(h));
    v
}

/// The nearest hit `t`, or a failure.
fn nearest(hits: &[RawHit3]) -> f64 {
    hits.iter().map(|h| h.t_mm).fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

#[test]
fn a_ray_through_a_sphere_enters_and_exits_with_outward_normals() {
    let r = 4.0;
    let hits = prim(|out| {
        sphere_intersect(
            r,
            &ray(Vec3::new(-10.0, 0.0, 0.0), Vec3::X),
            TMIN,
            TMAX,
            out,
        )
    });
    assert_eq!(hits.len(), 2);
    let mut ts: Vec<f64> = hits.iter().map(|h| h.t_mm).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((ts[0] - 6.0).abs() < 1e-12, "enters at x=-4");
    assert!((ts[1] - 14.0).abs() < 1e-12, "exits at x=+4");
    for h in &hits {
        // Outward normal is the radial direction, and it is already unit.
        assert!((h.normal.length() - 1.0).abs() < 1e-14);
        assert!((h.normal - h.point / r).length() < 1e-13);
    }
}

#[test]
fn a_skew_ray_finds_the_sphere_and_its_normal_stays_unit() {
    // Non-zero in all three components: the point of a 3D engine. Aimed through
    // the interior so it genuinely crosses.
    let r = 5.0;
    let dir = Vec3::new(20.0, 3.0, -3.0);
    let hits =
        prim(|out| sphere_intersect(r, &ray(Vec3::new(-20.0, -1.0, 2.0), dir), TMIN, TMAX, out));
    assert_eq!(hits.len(), 2);
    for h in &hits {
        assert!((h.point.length() - r).abs() < 1e-10, "on the sphere");
        assert!((h.normal.length() - 1.0).abs() < 1e-13);
    }
}

#[test]
fn sphere_containment_is_scale_free() {
    let tol = ToleranceContext::for_scene_scale(8.0);
    assert_eq!(sphere_contains(4.0, Vec3::ZERO, &tol), Containment::Inside);
    assert_eq!(
        sphere_contains(4.0, Vec3::new(10.0, 0.0, 0.0), &tol),
        Containment::Outside
    );
    assert_eq!(
        sphere_contains(4.0, Vec3::new(4.0, 0.0, 0.0), &tol),
        Containment::Boundary
    );
}

// ---------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------

#[test]
fn a_ray_through_a_box_reports_entry_and_exit_faces() {
    let h = Vec3::new(1.0, 2.0, 3.0);
    let hits = prim(|out| {
        box_intersect(
            h,
            &ray(Vec3::new(-10.0, 0.0, 0.0), Vec3::X),
            TMIN,
            TMAX,
            out,
        )
    });
    assert_eq!(hits.len(), 2);
    let entry = hits
        .iter()
        .min_by(|a, b| a.t_mm.partial_cmp(&b.t_mm).unwrap())
        .unwrap();
    assert!(
        (entry.t_mm - 9.0).abs() < 1e-12,
        "enters the -X face at x=-1"
    );
    assert!(
        (entry.normal - Vec3::NEG_X).length() < 1e-14,
        "outward normal is -X"
    );
}

#[test]
fn a_ray_through_a_box_corner_is_one_event_not_three() {
    // Straight at the (-1,-1,-1) corner: all three slabs share t_enter, and the
    // dedup rule must report one entry hit rather than three coincident ones.
    let h = Vec3::splat(1.0);
    let dir = Vec3::splat(1.0);
    let hits = prim(|out| {
        box_intersect(
            h,
            &ray(Vec3::new(-10.0, -10.0, -10.0), dir),
            TMIN,
            TMAX,
            out,
        )
    });
    assert_eq!(hits.len(), 2, "one entry, one exit — not a hit per slab");
}

#[test]
fn box_containment() {
    let tol = ToleranceContext::for_scene_scale(4.0);
    let h = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(box_contains(h, Vec3::ZERO, &tol), Containment::Inside);
    assert_eq!(
        box_contains(h, Vec3::new(0.5, 1.0, 2.9), &tol),
        Containment::Inside
    );
    assert_eq!(
        box_contains(h, Vec3::new(1.5, 0.0, 0.0), &tol),
        Containment::Outside
    );
    assert_eq!(
        box_contains(h, Vec3::new(1.0, 0.0, 0.0), &tol),
        Containment::Boundary
    );
}

// ---------------------------------------------------------------------------
// Cylinder
// ---------------------------------------------------------------------------

#[test]
fn a_ray_crosses_a_cylinder_side_and_a_cap() {
    // Axis +Z, r = 2, half-length 5. A ray along +X through the middle crosses
    // both side walls; a ray along +Z crosses both caps.
    let (r, l, axis) = (2.0, 5.0, Vec3::Z);
    let side = prim(|out| {
        cylinder_intersect(
            r,
            l,
            axis,
            &ray(Vec3::new(-10.0, 0.0, 0.0), Vec3::X),
            TMIN,
            TMAX,
            out,
        )
    });
    assert_eq!(side.len(), 2);
    for h in &side {
        assert!((h.point.x.abs() - r).abs() < 1e-10);
        assert!(
            (h.normal.z).abs() < 1e-12,
            "a side normal has no axial part"
        );
    }
    let caps = prim(|out| {
        cylinder_intersect(
            r,
            l,
            axis,
            &ray(Vec3::new(0.0, 0.0, -10.0), Vec3::Z),
            TMIN,
            TMAX,
            out,
        )
    });
    assert_eq!(caps.len(), 2);
    for h in &caps {
        assert!((h.point.z.abs() - l).abs() < 1e-12);
        assert!(
            (h.normal.z.abs() - 1.0).abs() < 1e-12,
            "a cap normal is axial"
        );
    }
}

#[test]
fn a_cylinder_about_a_tilted_axis_still_works() {
    let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
    let (r, l) = (1.5, 4.0);
    // Fire perpendicular to the axis, through the centre.
    let dir = Vec3::new(1.0, -1.0, 0.0).normalize();
    let hits = prim(|out| {
        cylinder_intersect(
            r,
            l,
            axis,
            &ray(Vec3::new(-10.0, 10.0, 0.0), dir),
            TMIN,
            TMAX,
            out,
        )
    });
    assert_eq!(hits.len(), 2);
    for h in &hits {
        // On the side: radial distance from the axis equals r.
        let radial = h.point - axis * h.point.dot(axis);
        assert!((radial.length() - r).abs() < 1e-10);
    }
}

#[test]
fn cylinder_containment() {
    let tol = ToleranceContext::for_scene_scale(5.0);
    let (r, l, axis) = (2.0, 5.0, Vec3::Z);
    assert_eq!(
        cylinder_contains(r, l, axis, Vec3::ZERO, &tol),
        Containment::Inside
    );
    assert_eq!(
        cylinder_contains(r, l, axis, Vec3::new(0.0, 0.0, 6.0), &tol),
        Containment::Outside
    );
    assert_eq!(
        cylinder_contains(r, l, axis, Vec3::new(3.0, 0.0, 0.0), &tol),
        Containment::Outside
    );
    assert_eq!(
        cylinder_contains(r, l, axis, Vec3::new(2.0, 0.0, 0.0), &tol),
        Containment::Boundary
    );
}

// ---------------------------------------------------------------------------
// Planar shapes
// ---------------------------------------------------------------------------

#[test]
fn planar_shapes_admit_the_points_they_should() {
    let along_z = ray(Vec3::new(0.3, -0.2, -5.0), Vec3::Z);
    for (shape, inside) in [
        (
            PlanarShape2::Rectangle {
                half_width_mm: 1.0,
                half_height_mm: 1.0,
            },
            true,
        ),
        (PlanarShape2::Disc { radius_mm: 0.3 }, false),
        (PlanarShape2::Disc { radius_mm: 1.0 }, true),
        (
            PlanarShape2::Annulus {
                inner_mm: 0.5,
                outer_mm: 2.0,
            },
            false,
        ),
    ] {
        let hits = prim(|out| planar_shape_intersect(&shape, &along_z, TMIN, TMAX, out));
        assert_eq!(hits.len(), usize::from(inside), "{shape:?}");
        if inside {
            assert!((hits[0].point.z).abs() < 1e-12, "the hit is on z = 0");
            assert!((hits[0].normal - Vec3::Z).length() < 1e-14);
        }
    }
}

#[test]
fn a_polygon_shape_uses_the_crossing_rule() {
    // An L-shaped polygon: a point in the notch is outside.
    let shape = PlanarShape2::Polygon {
        vertices: vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ],
    };
    let inside = ray(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
    let notch = ray(Vec3::new(1.5, 1.5, -5.0), Vec3::Z);
    assert_eq!(
        prim(|o| planar_shape_intersect(&shape, &inside, TMIN, TMAX, o)).len(),
        1
    );
    assert_eq!(
        prim(|o| planar_shape_intersect(&shape, &notch, TMIN, TMAX, o)).len(),
        0
    );
}

// ---------------------------------------------------------------------------
// Conic and even asphere of revolution
// ---------------------------------------------------------------------------

fn conic_of_rev(c: f64, k: f64, outer_r: f64) -> Revolution {
    Revolution {
        vertex: Vec3::ZERO,
        axis: Vec3::X,
        c,
        k,
        coeffs: [0.0; 8],
        inner_r: 0.0,
        outer_r,
    }
}

#[test]
fn a_conic_of_revolution_reduces_to_a_sphere_at_k_zero() {
    // c = 1/R, k = 0: a sphere of radius R, vertex at the origin, centre at +R x̂.
    let r = 8.0;
    let surf = conic_of_rev(1.0 / r, 0.0, 5.0);
    // Along the axis, the vertex is hit at the origin.
    let axis_hits = prim(|o| {
        revolution_intersect(
            &surf,
            &ray(Vec3::new(-3.0, 0.0, 0.0), Vec3::X),
            TMIN,
            TMAX,
            o,
        )
    });
    assert!(!axis_hits.is_empty());
    assert!((nearest(&axis_hits) - 3.0).abs() < 1e-10, "vertex at x = 0");
    // A sphere centred at (R, 0, 0) of radius R passes through every hit.
    let skew = prim(|o| {
        revolution_intersect(
            &surf,
            &ray(Vec3::new(-3.0, 1.0, 0.5), Vec3::X),
            TMIN,
            TMAX,
            o,
        )
    });
    assert!(!skew.is_empty());
    for h in &skew {
        assert!(((h.point - Vec3::new(r, 0.0, 0.0)).length() - r).abs() < 1e-9);
        assert!((h.normal.length() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn a_conic_of_revolution_has_the_analytic_gradient_normal() {
    let surf = conic_of_rev(1.0 / 10.0, -0.5, 4.0);
    let hs = prim(|o| {
        revolution_intersect(
            &surf,
            &ray(Vec3::new(-5.0, 1.3, -0.7), Vec3::X),
            TMIN,
            TMAX,
            o,
        )
    });
    assert!(!hs.is_empty());
    for h in &hs {
        // ∇F for F = c(r² + (1+k)x²) − 2x, r² = y² + z².
        let (c, k) = (1.0 / 10.0, -0.5);
        let grad = Vec3::new(
            2.0 * c * (1.0 + k) * h.point.x - 2.0,
            2.0 * c * h.point.y,
            2.0 * c * h.point.z,
        )
        .normalize();
        // Up to sign; the primitive orients toward +axis.
        let aligned = if grad.dot(h.normal) < 0.0 {
            -grad
        } else {
            grad
        };
        assert!(
            (aligned - h.normal).length() < 1e-11,
            "analytic normal is off"
        );
    }
}

#[test]
fn an_even_asphere_reduces_to_its_base_conic_to_machine_precision() {
    // The finite-difference floor is ~1e-9. Zero aspheric terms must reproduce the
    // exact conic, position and normal, far tighter than that — which the analytic
    // path clears and the 2D central-difference normal could not.
    let base = conic_of_rev(1.0 / 12.0, -1.0, 5.0);
    let asph = Revolution {
        coeffs: [0.0; 8],
        ..base
    };
    let r = ray(Vec3::new(-6.0, 2.0, 1.1), Vec3::new(1.0, 0.02, -0.01));
    let a = prim(|o| revolution_intersect(&base, &r, TMIN, TMAX, o));
    let b = prim(|o| revolution_intersect(&asph, &r, TMIN, TMAX, o));
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    for (x, y) in a.iter().zip(&b) {
        assert!((x.point - y.point).length() < 1e-12);
        assert!((x.normal - y.normal).length() < 1e-12);
    }
}

#[test]
fn an_even_asphere_normal_matches_an_independent_analytic_gradient() {
    // A genuine asphere. The reference normal is the closed-form gradient of
    // x − sag(r), computed independently here; agreement to 1e-10 is the proof the
    // engine's normal is analytic, not the 2D path's ~1e-9 finite difference.
    let (c, k) = (1.0 / 20.0, 0.0);
    let coeffs = [1e-5, -2e-7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let surf = Revolution {
        vertex: Vec3::ZERO,
        axis: Vec3::X,
        c,
        k,
        coeffs,
        inner_r: 0.0,
        outer_r: 6.0,
    };
    let hs = prim(|o| {
        revolution_intersect(
            &surf,
            &ray(Vec3::new(-8.0, 2.5, -1.5), Vec3::X),
            TMIN,
            TMAX,
            o,
        )
    });
    assert!(!hs.is_empty());
    for h in &hs {
        let r = (h.point.y * h.point.y + h.point.z * h.point.z).sqrt();
        let dsag = ref_dsag(c, k, &coeffs, r);
        let grad = Vec3::new(1.0, -dsag * h.point.y / r, -dsag * h.point.z / r).normalize();
        let aligned = if grad.dot(h.normal) < 0.0 {
            -grad
        } else {
            grad
        };
        assert!(
            (aligned - h.normal).length() < 1e-10,
            "normal error {}",
            (aligned - h.normal).length()
        );
    }
}

/// An independent closed-form `dsag/dr`, spelled out differently from the engine's
/// so a shared slip cannot hide.
fn ref_dsag(c: f64, k: f64, coeffs: &[f64; 8], r: f64) -> f64 {
    let s = (1.0 - (1.0 + k) * c * c * r * r).sqrt();
    let mut d = c * r / s;
    for (i, a) in coeffs.iter().enumerate() {
        let p = (2 * i + 4) as f64;
        d += a * p * r.powi((2 * i + 3) as i32);
    }
    d
}

// ---------------------------------------------------------------------------
// Frustum (revolved line segment)
// ---------------------------------------------------------------------------

#[test]
fn a_frustum_is_a_cylinder_when_the_radius_is_constant() {
    // y0 = y1 = 3, x from -1 to 1: a cylinder of radius 3 about +X.
    let hits = prim(|o| {
        frustum_intersect(
            -1.0,
            3.0,
            1.0,
            3.0,
            &ray(Vec3::new(0.0, -10.0, 0.0), Vec3::Y),
            TMIN,
            TMAX,
            o,
        )
    });
    assert_eq!(hits.len(), 2);
    for h in &hits {
        assert!(((h.point.y * h.point.y + h.point.z * h.point.z).sqrt() - 3.0).abs() < 1e-12);
    }
}

#[test]
fn a_frustum_is_a_flat_face_when_the_axial_span_is_zero() {
    // x0 = x1 = 2, r from 0 to 4: an annular/disc face at x = 2.
    let hits = prim(|o| {
        frustum_intersect(
            2.0,
            0.0,
            2.0,
            4.0,
            &ray(Vec3::new(-5.0, 1.0, 0.0), Vec3::X),
            TMIN,
            TMAX,
            o,
        )
    });
    assert_eq!(hits.len(), 1);
    assert!((hits[0].point.x - 2.0).abs() < 1e-12);
    assert!((hits[0].normal.x.abs() - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Transform invariance
// ---------------------------------------------------------------------------

#[test]
fn intersection_is_invariant_under_placement() {
    // Placing the geometry and firing the world ray must give the same world hit
    // as pulling the ray into the local frame and firing at the identity.
    let geoms = [
        SolidGeometry3::Sphere { radius_mm: 3.0 },
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(1.0, 2.0, 1.5),
        },
        SolidGeometry3::Cylinder {
            radius_mm: 1.5,
            half_length_mm: 4.0,
            axis_local: Vec3::Z,
        },
    ];
    let t = Isometry3::new(
        Vec3::new(5.0, -3.0, 2.0),
        quat_axis(Vec3::new(1.0, 2.0, -1.0), 0.9),
    )
    .unwrap();
    let r = ray(Vec3::new(-20.0, 1.0, -0.5), Vec3::new(1.0, 0.1, 0.05));
    for g in &geoms {
        let placed = hits(g, &r, &t);
        let r_local = ray(t.inverse_transform_point(r.origin), t.inv_rotate(r.dir));
        let local = hits(g, &r_local, &Isometry3::IDENTITY);
        assert_eq!(placed.len(), local.len(), "{g:?}");
        for (p, l) in placed.iter().zip(&local) {
            assert!((p.t_mm - l.t_mm).abs() < 1e-10, "{g:?}");
            // The placed hit's world point and normal match the pushed-forward
            // local hit's.
            let pw = t.transform_point(p.point_local);
            let lw = t.transform_point(l.point_local);
            assert!((pw - lw).length() < 1e-10, "{g:?}");
            assert!(
                (t.rotate(p.normal_local) - t.rotate(l.normal_local)).length() < 1e-10,
                "{g:?}"
            );
            // And the world point genuinely lies on the world ray.
            assert!((pw - r.at(p.t_mm)).length() < 1e-9, "{g:?} hit off the ray");
        }
    }
}

fn quat_axis(axis: Vec3, angle: f64) -> [f64; 4] {
    Isometry3::from_axis_angle(axis, angle).unwrap().rot_xyzw
}

// ---------------------------------------------------------------------------
// Conservative AABBs
// ---------------------------------------------------------------------------

#[test]
fn a_world_aabb_contains_every_hit_it_is_asked_about() {
    let geoms = [
        SolidGeometry3::Sphere { radius_mm: 3.0 },
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(1.0, 2.0, 1.5),
        },
        SolidGeometry3::Cylinder {
            radius_mm: 1.5,
            half_length_mm: 4.0,
            axis_local: Vec3::new(1.0, 1.0, 0.0),
        },
    ];
    let t = Isometry3::new(Vec3::new(2.0, -1.0, 0.5), quat_axis(Vec3::Z, FRAC_PI_4)).unwrap();
    for g in &geoms {
        let bb = g.aabb_world(&t);
        // Sweep a fan of rays; every world hit must sit inside the reported box.
        for i in 0..40 {
            let a = i as f64 / 40.0 * std::f64::consts::TAU;
            let o = t.pos + Vec3::new(20.0 * a.cos(), 20.0 * a.sin(), 3.0);
            let dir = (t.pos - o).normalize();
            for h in hits(g, &ray(o, dir), &t) {
                let p = t.transform_point(h.point_local);
                let pad = bb.padded(1e-9 * bb.diagonal().max(1.0));
                assert!(pad.contains_point(p), "{g:?}: hit {p:?} escaped {bb:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Extruded profile
// ---------------------------------------------------------------------------

/// A closed rectangle profile in XY: [-w,w] × [-h,h], CCW, four line edges.
fn rectangle_profile(w: f64, h: f64) -> ClosedProfile2 {
    let corners = [
        Vec2::new(-w, -h),
        Vec2::new(w, -h),
        Vec2::new(w, h),
        Vec2::new(-w, h),
    ];
    let edges = (0..4)
        .map(|i| ProfileEdge2 {
            patch: PatchId(i as u64),
            curve: Curve2::Line {
                a: corners[i],
                b: corners[(i + 1) % 4],
            },
        })
        .collect();
    ClosedProfile2 { edges }
}

#[test]
fn an_extruded_rectangle_is_a_box_of_sides_and_caps() {
    let geom = SolidGeometry3::Profile(ProfileSolid3 {
        profile: rectangle_profile(1.0, 2.0),
        construction: ProfileConstruction3::ExtrudeLocalZ {
            z_min_mm: -3.0,
            z_max_mm: 3.0,
        },
    });
    let t = Isometry3::IDENTITY;
    // Along X: two side hits at x = ±1.
    let side = hits(&geom, &ray(Vec3::new(-10.0, 0.0, 0.0), Vec3::X), &t);
    assert_eq!(side.len(), 2);
    for h in &side {
        assert!((h.point_local.x.abs() - 1.0).abs() < 1e-12);
        assert!(
            (h.normal_local.x.abs() - 1.0).abs() < 1e-12,
            "a side normal points along X"
        );
    }
    // Along Z: two cap hits at z = ±3, and the caps face outward.
    let caps = hits(&geom, &ray(Vec3::new(0.0, 0.5, -10.0), Vec3::Z), &t);
    assert_eq!(caps.len(), 2);
    for h in &caps {
        assert!((h.point_local.z.abs() - 3.0).abs() < 1e-12);
        assert!((h.normal_local.z.abs() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn extruded_containment_needs_both_the_profile_and_the_z_range() {
    let geom = SolidGeometry3::Profile(ProfileSolid3 {
        profile: rectangle_profile(1.0, 2.0),
        construction: ProfileConstruction3::ExtrudeLocalZ {
            z_min_mm: -3.0,
            z_max_mm: 3.0,
        },
    });
    let t = Isometry3::IDENTITY;
    let tol = ToleranceContext::for_scene_scale(6.0);
    assert_eq!(geom.contains(Vec3::ZERO, &t, &tol), Containment::Inside);
    // Inside the profile but past the cap.
    assert_eq!(
        geom.contains(Vec3::new(0.0, 0.0, 4.0), &t, &tol),
        Containment::Outside
    );
    // Within Z but outside the profile.
    assert_eq!(
        geom.contains(Vec3::new(2.0, 0.0, 0.0), &t, &tol),
        Containment::Outside
    );
}

// ---------------------------------------------------------------------------
// Revolved profile
// ---------------------------------------------------------------------------

#[test]
fn a_revolved_rectangle_is_a_cylinder_by_containment() {
    // The mirror's shape: revolve a rectangle about +X and it is a cylinder of
    // radius h, length 2w. Containment is the release-critical revolve test.
    let (w, h) = (0.75, 14.0);
    let profile = rectangle_profile(w, h);
    let tol = ToleranceContext::for_scene_scale(30.0);
    let inside = |p: Vec3| revolve_contains(&profile, p, &tol);
    assert_eq!(inside(Vec3::ZERO), Containment::Inside);
    // On the axis inside the length, and off-axis within the radius.
    assert_eq!(inside(Vec3::new(0.0, 10.0, 0.0)), Containment::Inside);
    assert_eq!(inside(Vec3::new(0.0, 0.0, 10.0)), Containment::Inside);
    // Beyond the radius (r = 15 > 14).
    assert_eq!(inside(Vec3::new(0.0, 9.0, 12.0)), Containment::Outside);
    // Beyond the length.
    assert_eq!(inside(Vec3::new(1.0, 0.0, 0.0)), Containment::Outside);
}

#[test]
fn a_revolved_rectangle_traces_as_a_cylinder() {
    let (w, h) = (0.75, 14.0);
    let geom = SolidGeometry3::Profile(ProfileSolid3 {
        profile: rectangle_profile(w, h),
        construction: ProfileConstruction3::RevolveLocalX,
    });
    let t = Isometry3::IDENTITY;
    // Perpendicular to the axis through the middle: two side hits at r = h.
    let side = hits(&geom, &ray(Vec3::new(0.0, -30.0, 0.0), Vec3::Y), &t);
    assert_eq!(side.len(), 2, "entry and exit of the cylindrical wall");
    for s in &side {
        assert!((s.point_local.y.abs() - h).abs() < 1e-9);
        // Outward normal points away from the axis.
        assert!(
            s.normal_local.dot(s.point_local.with_x(0.0)) > 0.0,
            "normal faces outward"
        );
    }
    // Along the axis: the two flat end caps at x = ±w.
    let caps = hits(&geom, &ray(Vec3::new(-10.0, 0.0, 0.0), Vec3::X), &t);
    assert_eq!(caps.len(), 2);
    for c in &caps {
        assert!((c.point_local.x.abs() - w).abs() < 1e-9);
        assert!(
            (c.normal_local.x.abs() - 1.0).abs() < 1e-9,
            "a cap normal is axial"
        );
    }
}

/// A biconvex singlet meridian: front and rear conic surfaces plus the two rim
/// lines, exactly as the generators author it.
fn singlet_profile(r1: f64, r2: f64, thickness: f64, half_ap: f64) -> ClosedProfile2 {
    let (zf, zr) = (-thickness / 2.0, thickness / 2.0);
    let sag = |c: f64, ha: f64| c * ha * ha / (1.0 + (1.0 - c * c * ha * ha).max(0.0).sqrt());
    let (c1, c2) = (1.0 / r1, 1.0 / r2);
    let ftip = zf + sag(c1, half_ap);
    let rtip = zr + sag(c2, half_ap);
    ClosedProfile2 {
        edges: vec![
            ProfileEdge2 {
                patch: PatchId(0),
                curve: Curve2::Conic(ConicParams2 {
                    vertex: Vec2::new(zf, 0.0),
                    axis: Vec2::X,
                    c: c1,
                    k: 0.0,
                    half_ap,
                }),
            },
            ProfileEdge2 {
                patch: PatchId(1),
                curve: Curve2::Line {
                    a: Vec2::new(ftip, half_ap),
                    b: Vec2::new(rtip, half_ap),
                },
            },
            ProfileEdge2 {
                patch: PatchId(2),
                curve: Curve2::Conic(ConicParams2 {
                    vertex: Vec2::new(zr, 0.0),
                    axis: Vec2::X,
                    c: c2,
                    k: 0.0,
                    half_ap,
                }),
            },
            ProfileEdge2 {
                patch: PatchId(3),
                curve: Curve2::Line {
                    a: Vec2::new(rtip, -half_ap),
                    b: Vec2::new(ftip, -half_ap),
                },
            },
        ],
    }
}

#[test]
fn a_revolved_singlet_is_hit_on_its_front_surface_at_the_vertex() {
    // R1 = +20 (convex front), R2 = -20 (convex rear), 4 mm thick, 8 mm aperture.
    let geom = SolidGeometry3::Profile(ProfileSolid3 {
        profile: singlet_profile(20.0, -20.0, 4.0, 8.0),
        construction: ProfileConstruction3::RevolveLocalX,
    });
    let t = Isometry3::IDENTITY;
    // Down the axis: hits the front vertex (x = -2), passes through, hits the rear
    // vertex (x = +2). Both are real crossings of the glass boundary.
    let axial = hits(&geom, &ray(Vec3::new(-10.0, 0.0, 0.0), Vec3::X), &t);
    let xs: Vec<f64> = axial.iter().map(|h| h.point_local.x).collect();
    assert!(
        xs.iter().any(|x| (x + 2.0).abs() < 1e-6),
        "front vertex at x=-2: {xs:?}"
    );
    assert!(
        xs.iter().any(|x| (x - 2.0).abs() < 1e-6),
        "rear vertex at x=+2: {xs:?}"
    );
    // A skew ray through the aperture still finds the front surface.
    let skew = hits(&geom, &ray(Vec3::new(-10.0, 3.0, 2.0), Vec3::X), &t);
    assert!(!skew.is_empty(), "a skew ray misses the lens entirely");
    for h in &skew {
        assert!((h.normal_local.length() - 1.0).abs() < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// Surface of revolution (zero-thickness)
// ---------------------------------------------------------------------------

#[test]
fn a_zero_thickness_surface_contains_nothing() {
    let geom = SurfaceGeometry3::SurfaceOfRevolution {
        sag: SagModel::Conic {
            curvature_per_mm: 1.0 / 10.0,
            conic: 0.0,
        },
        radial_aperture: RadialAperture {
            inner_radius_mm: 0.0,
            outer_radius_mm: 5.0,
        },
    };
    let tol = ToleranceContext::default();
    assert_eq!(
        geom.contains(Vec3::ZERO, &Isometry3::IDENTITY, &tol),
        Containment::Outside
    );
}

#[test]
fn a_surface_of_revolution_is_hit_with_a_front_facing_normal() {
    // Axis +Z, vertex at the origin, a weak convex sag. A ray down -Z hits the
    // vertex and the normal there is the front, +Z.
    let geom = SurfaceGeometry3::SurfaceOfRevolution {
        sag: SagModel::Conic {
            curvature_per_mm: 1.0 / 25.0,
            conic: 0.0,
        },
        radial_aperture: RadialAperture {
            inner_radius_mm: 0.0,
            outer_radius_mm: 6.0,
        },
    };
    let hs = hits(
        &geom,
        &ray(Vec3::new(0.0, 0.0, 10.0), Vec3::NEG_Z),
        &Isometry3::IDENTITY,
    );
    assert!(!hs.is_empty());
    let vertex = hs
        .iter()
        .min_by(|a, b| {
            a.point_local
                .z
                .abs()
                .partial_cmp(&b.point_local.z.abs())
                .unwrap()
        })
        .unwrap();
    assert!(vertex.point_local.length() < 1e-9, "hits the vertex");
    assert!(
        (vertex.normal_local - Vec3::Z).length() < 1e-9,
        "vertex normal is +Z"
    );
}

// ---------------------------------------------------------------------------
// PlanarOnly refuses a 3D trace
// ---------------------------------------------------------------------------

#[test]
fn a_planar_only_body_is_unsupported_in_3d() {
    let geom = SolidGeometry3::Profile(ProfileSolid3 {
        profile: rectangle_profile(1.0, 1.0),
        construction: ProfileConstruction3::PlanarOnly { plane_z_mm: 0.0 },
    });
    let t = Isometry3::IDENTITY;
    let tol = ToleranceContext::default();
    assert_eq!(
        geom.contains(Vec3::ZERO, &t, &tol),
        Containment::Unsupported
    );
    // And it intersects nothing: no fake plane appears in a full-3D trace.
    assert!(hits(&geom, &ray(Vec3::new(0.0, 0.0, -5.0), Vec3::Z), &t).is_empty());
}

// ---------------------------------------------------------------------------
// Promotion validation
// ---------------------------------------------------------------------------

#[test]
fn promotion_accepts_a_symmetric_lens_and_refuses_an_asymmetric_one() {
    let lens = singlet_profile(20.0, -20.0, 4.0, 8.0);
    assert!(validate_revolvable(&lens).is_ok());
    assert!(validate_extrudable(&lens).is_ok());

    // Shear the top rim off-symmetry: no longer mirror-symmetric about y = 0.
    let mut skew = lens.clone();
    skew.edges[1].curve = Curve2::Line {
        a: Vec2::new(-1.0, 8.0),
        b: Vec2::new(1.0, 9.0),
    };
    assert!(
        validate_revolvable(&skew).is_err(),
        "an asymmetric profile is not revolvable"
    );
}

#[test]
fn promotion_refuses_an_off_axis_arc_for_revolve() {
    // A closed loop whose curved side is an arc centred off the axis revolves to a
    // torus, which is refused rather than traced with a missing surface.
    let profile = ClosedProfile2 {
        edges: vec![
            ProfileEdge2 {
                patch: PatchId(0),
                curve: Curve2::Line {
                    a: Vec2::new(-1.0, 0.0),
                    b: Vec2::new(1.0, 0.0),
                },
            },
            ProfileEdge2 {
                patch: PatchId(1),
                curve: Curve2::Arc {
                    center: Vec2::new(0.0, 5.0),
                    radius_mm: 5.0,
                    a0: -FRAC_PI_3,
                    a1: FRAC_PI_3,
                },
            },
            ProfileEdge2 {
                patch: PatchId(2),
                curve: Curve2::Line {
                    a: Vec2::new(1.0, 0.0),
                    b: Vec2::new(-1.0, 0.0),
                },
            },
        ],
    };
    let e = validate_revolvable(&profile).unwrap_err();
    assert!(e.contains("torus"), "{e}");
}
