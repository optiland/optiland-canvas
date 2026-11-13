//! A branch that leaves a face close to grazing stays in the medium it is in.
//!
//! The tracer reads a branch's medium a short step (`8 × medium_abs_mm`) ahead
//! of its origin. Along a ray that leaves its face within asin(1/8) ≈ 7.2° of
//! grazing, that point is less than `medium_abs_mm` off the face, where
//! containment answers neither in nor out, and the branch used to be given the
//! ambient index. Total internal reflection along a light pipe's wall is that
//! case: the reflected ray then crossed the end face without refracting and
//! without a Fresnel loss, or crossed the next wall and left. A PMMA taper on
//! an LED put 7.7 % of its output at the wrong angle that way.
//!
//! - a straight pipe (a box), rays at internal angles either side of the
//!   threshold, one reflection to eleven: every wall hit is TIR, every segment
//!   inside runs at the glass's index, and the ray leaves the end face by Snell;
//! - a cone taper (a revolved profile) whose TIR leaves the wall 6.3° from
//!   grazing: the reflection turns the ray by twice the wall's angle, and the
//!   end face refracts it and takes its Fresnel cut.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::id::{LayerId, PatchId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::profile2::{
    ClosedProfile2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::{Body3, ObjectCommon, SolidGeometry3, SurfaceProps3};
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Ray3, Vec3};
use oc_trace::{trace3, Segment3};

/// A glass body; returns the index the tracer will use at the d line.
fn add_glass(scene: &mut Scene3, geometry: SolidGeometry3, transform: Isometry3) -> f64 {
    let material = Material::AbbePoint { nd: 1.5, vd: 64.0 };
    let n = material.index_at(LAMBDA_D, &scene.catalog).unwrap();
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = transform;
    scene.bodies.push(Body3 {
        common,
        geometry,
        material,
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    n
}

fn dir(s: &Segment3) -> Vec3 {
    (s.b - s.a).normalize()
}

/// The angle between `d` and the `+x` axis, in degrees.
fn off_axis_deg(d: Vec3) -> f64 {
    d.cross(Vec3::X).length().asin().to_degrees()
}

/// The ray in air that the end face at `x = 0` refracts into `inside`.
fn launched_into(inside: Vec3, n: f64, hit: Vec3) -> Ray3 {
    let s = inside.cross(Vec3::X).length() * n;
    let across = Vec3::new(0.0, inside.y, inside.z).normalize();
    let d = Vec3::X * (1.0 - s * s).sqrt() + across * s;
    Ray3::new(hit - d, d).unwrap()
}

/// Unpolarized Fresnel reflectance from `n1` into `n2`.
fn reflectance(n1: f64, n2: f64, cos_i: f64) -> f64 {
    let sin_t = n1 / n2 * (1.0 - cos_i * cos_i).sqrt();
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
    0.5 * (rs * rs + rp * rp)
}

/// Every segment between the entry and the exit runs at index `n`, and the last
/// leaves the `x = const` end face by Snell from the one before it.
fn assert_inside_then_out(segs: &[Segment3], n: f64, what: &str) {
    let k = segs.len();
    assert!(k >= 3, "{what}: expected entry, glass, exit; got {k} segments");
    for s in &segs[1..k - 1] {
        let rate = (s.opl1_mm - s.opl0_mm) / (s.b - s.a).length();
        assert!((rate - n).abs() < 1e-9, "{what}: a segment inside runs at index {rate}, not {n}");
    }
    let last = &segs[k - 1];
    let rate = (last.opl1_mm - last.opl0_mm) / (last.b - last.a).length();
    assert!((rate - 1.0).abs() < 1e-9, "{what}: the ray past the end face runs at index {rate}");
    let (s_in, s_out) = (
        dir(&segs[k - 2]).cross(Vec3::X).length(),
        dir(last).cross(Vec3::X).length(),
    );
    assert!((s_out - n * s_in).abs() < 1e-12, "{what}: the end face sends sin {s_out}, Snell says {}", n * s_in);
}

#[test]
fn tir_along_a_straight_pipe_keeps_every_ray_in_the_glass() {
    // A pipe along x from 0 to 100, walls at y = ±1. A ray entering on the axis
    // at internal angle θ meets the walls at x = (2k - 1) / tan θ, each time
    // leaving at θ from grazing.
    let (len, half) = (100.0, 1.0);
    for theta_deg in [1.0_f64, 3.0, 6.0, 7.0, 7.5, 9.0, 12.0] {
        let mut scene = Scene3::new();
        let n = add_glass(
            &mut scene,
            SolidGeometry3::Box { half_extents_mm: Vec3::new(len / 2.0, half, 10.0) },
            Isometry3::from_pos(Vec3::new(len / 2.0, 0.0, 0.0)),
        );
        let theta = theta_deg.to_radians();
        let inside = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let ray = launched_into(inside, n, Vec3::ZERO);
        let res = trace3(&scene, &[launch(ray, LAMBDA_D, 1.0)], &no_ghosts());
        let segs = primary_segments(&res);
        let what = format!("θ {theta_deg}°");

        let walls = ((len * theta.tan() + half) / (2.0 * half)).floor() as usize;
        let tir = segs.iter().filter(|s| s.flags.tir).count();
        assert_eq!(tir, walls, "{what}: {tir} TIR segments, the walls are met {walls} times");
        assert_inside_then_out(&segs, n, &what);
        assert!((segs[segs.len() - 1].a.x - len).abs() < 1e-9, "{what}: it does not leave by the end face");
        let out = off_axis_deg(dir(&segs[segs.len() - 1]));
        let want = (n * theta.sin()).asin().to_degrees();
        assert!((out - want).abs() < 1e-9, "{what}: leaves at {out}°, not {want}°");
    }
}

#[test]
fn tir_off_a_cone_tapers_wall_near_grazing_still_refracts_at_the_end() {
    // A PMMA-like taper about +x: radius 2 at x = 0 to 6 at x = 40, its wall 5.71°
    // off the axis. A ray inside at 12° meets the wall at 12 - 5.71 = 6.29° from
    // grazing, below the 7.2° the look-ahead cannot resolve, and leaves it at
    // 2 × 5.71 - 12 = -0.58° to the axis, straight to the end face.
    let (r1, r2, len) = (2.0, 6.0, 40.0);
    let edge = |a: (f64, f64), b: (f64, f64), k: u64| ProfileEdge2 {
        patch: PatchId(k),
        curve: Curve2::Line { a: Vec2::new(a.0, a.1), b: Vec2::new(b.0, b.1) },
    };
    let profile = ClosedProfile2 {
        edges: vec![
            edge((0.0, -r1), (0.0, r1), 1),
            edge((0.0, r1), (len, r2), 2),
            edge((len, r2), (len, -r2), 3),
            edge((len, -r2), (0.0, -r1), 4),
        ],
    };
    let mut scene = Scene3::new();
    let n = add_glass(
        &mut scene,
        SolidGeometry3::Profile(ProfileSolid3 { profile, construction: ProfileConstruction3::RevolveLocalX }),
        Isometry3::IDENTITY,
    );
    let wall = ((r2 - r1) / len).atan();
    let theta = 12f64.to_radians();
    let ray = launched_into(Vec3::new(theta.cos(), theta.sin(), 0.0), n, Vec3::new(0.0, 0.3, 0.0));
    let res = trace3(&scene, &[launch(ray, LAMBDA_D, 1.0)], &no_ghosts());
    let segs = primary_segments(&res);

    assert_eq!(segs.len(), 4, "entry, glass to the wall, wall to the end face, out");
    assert!(segs[2].flags.tir, "the wall reflection is TIR");
    let graze = (theta - wall).to_degrees();
    assert!(graze < (1.0f64 / 8.0).asin().to_degrees(), "the case is the one the look-ahead cannot resolve");
    let turned = off_axis_deg(dir(&segs[2]));
    assert!((turned - (2.0 * wall - theta).abs().to_degrees()).abs() < 1e-9, "TIR turns the ray to {turned}°");
    assert_inside_then_out(&segs, n, "taper");
    let cos_i = dir(&segs[2]).dot(Vec3::X);
    let t = segs[3].flux_start / segs[2].flux_end;
    let want = 1.0 - reflectance(n, 1.0, cos_i);
    assert!((t - want).abs() < 1e-5, "the end face passes {t}, Fresnel says {want}");
}
