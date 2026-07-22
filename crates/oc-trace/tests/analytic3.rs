//! 3D optical analytic fixtures: the proof the engine is not merely meridional.
//!
//! What this suite owes (contract §15.2):
//!
//! - arbitrary-azimuth Snell: `n1 sin θi = n2 sin θt` to 1e-12, with the
//!   transmitted ray in the plane of incidence and unit length;
//! - arbitrary 3D reflection: equal angles, tangential component preserved to
//!   1e-13;
//! - TIR just below and just above the critical angle, with a skew component;
//! - **spherical lens skew ray**: non-zero out-of-plane field and pupil
//!   coordinates through a centered singlet, refracting by Snell at each face;
//! - **tangential and sagittal fans distinct** for an off-axis field. This is
//!   the release proof (§20);
//! - out-of-plane fold: a mirror that puts light on a non-coplanar detector;
//! - a skew ray through a finite extruded prism against hand vector-Snell;
//! - aperture clipping, from both propagation directions;
//! - detector area normalization: uniform illumination gives equal flux density
//!   across equal-area cells.
//!
//! The against-Optiland cross-validation of the skew singlet lives in the
//! `validation/` harness (§15.4), which has an independent sequential tracer to
//! compare with; here the singlet is checked against Snell's law directly.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::material::LAMBDA_D;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{reflect3, refract3, trace3, TraceConfig3};

/// The sine of the angle between a unit direction and a unit normal.
fn sin_to_normal(dir: Vec3, normal: Vec3) -> f64 {
    dir.cross(normal).length()
}

// ---------------------------------------------------------------------------
// Interface physics, in 3D and at an arbitrary azimuth
// ---------------------------------------------------------------------------

#[test]
fn snell_holds_at_an_arbitrary_azimuth() {
    // A normal and an incident direction that both have all three components
    // non-zero: nothing here lives in a plane.
    let normal = Vec3::new(0.3, -0.7, 0.5).normalize();
    let d = Vec3::new(0.6, 0.5, -0.2).normalize();
    // Orient the normal to oppose the ray.
    let n = if d.dot(normal) < 0.0 { normal } else { -normal };
    let (n1, n2) = (1.0, 1.5);
    let t = refract3(d, n, n1, n2).expect("not TIR at this angle");

    assert!((t.length() - 1.0).abs() < 1e-13, "transmitted ray is unit");
    // n1 sin θi = n2 sin θt.
    let si = sin_to_normal(d, n);
    let st = sin_to_normal(t, n);
    assert!(
        (n1 * si - n2 * st).abs() < 1e-12,
        "Snell: {} vs {}",
        n1 * si,
        n2 * st
    );
    // The transmitted ray lies in the plane of incidence spanned by d and n.
    let plane_normal = d.cross(n).normalize();
    assert!(
        t.dot(plane_normal).abs() < 1e-13,
        "transmitted ray left the incidence plane"
    );
}

#[test]
fn reflection_preserves_the_tangential_component_in_3d() {
    let normal = Vec3::new(-0.2, 0.6, 0.77).normalize();
    let d = Vec3::new(0.5, 0.5, -0.5).normalize();
    let n = if d.dot(normal) < 0.0 { normal } else { -normal };
    let r = reflect3(d, n);
    assert!((r.length() - 1.0).abs() < 1e-14);
    // Equal angles: the normal components are opposite, tangential components
    // identical.
    assert!(
        (r.dot(n) + d.dot(n)).abs() < 1e-14,
        "angle of incidence != reflection"
    );
    let tang_in = d - d.dot(n) * n;
    let tang_out = r - r.dot(n) * n;
    assert!(
        (tang_in - tang_out).length() < 1e-13,
        "tangential component changed"
    );
}

#[test]
fn tir_switches_on_across_the_critical_angle_with_a_skew_component() {
    // Glass -> air, critical angle asin(1/1.5). The incidence plane is tilted so
    // the ray carries a genuine out-of-plane component.
    let (n1, n2) = (1.5_f64, 1.0_f64);
    let crit = (n2 / n1).asin();
    let normal = Vec3::new(0.0, 0.0, 1.0);
    // Build incident rays at crit ∓ ε, in a plane rotated 40° about the normal.
    let az = 40f64.to_radians();
    let tang = Vec3::new(az.cos(), az.sin(), 0.0);
    let below = (tang * (crit - 1e-3).sin() - normal * (crit - 1e-3).cos()).normalize();
    let above = (tang * (crit + 1e-3).sin() - normal * (crit + 1e-3).cos()).normalize();
    assert!(
        refract3(below, normal, n1, n2).is_some(),
        "should refract below critical"
    );
    assert!(
        refract3(above, normal, n1, n2).is_none(),
        "should TIR above critical"
    );
    // Both carry a real skew component (non-zero y).
    assert!(below.y.abs() > 1e-3 && above.y.abs() > 1e-3);
}

// ---------------------------------------------------------------------------
// A skew ray through a spherical singlet
// ---------------------------------------------------------------------------

#[test]
fn a_skew_ray_refracts_by_snell_at_both_faces_of_a_glass_sphere() {
    // A ball lens: one glass sphere. A ray with all three direction components
    // non-zero enters and exits; Snell must hold at each curved face, which a
    // meridional solver cannot claim for an out-of-plane ray.
    let mut scene = Scene3::new();
    let r_mm = 6.0;
    let n_glass = add_glass_sphere(&mut scene, Vec3::ZERO, r_mm, 1.5);

    let ray = Ray3::new(Vec3::new(-20.0, 1.5, 1.0), Vec3::new(1.0, 0.05, 0.03)).unwrap();
    let res = trace3(&scene, &[launch(ray, LAMBDA_D, 1.0)], &no_ghosts());

    // The primary transmitted chain: air -> glass -> air.
    let dirs = primary_dirs(&res);
    assert!(
        dirs.len() >= 3,
        "expected enter, cross, exit; got {}",
        dirs.len()
    );
    let pts = primary_points(&res);

    // Snell at the entry face.
    let n_entry = (pts[1] - Vec3::ZERO).normalize(); // outward sphere normal
    let si = sin_to_normal(dirs[0], n_entry);
    let st = sin_to_normal(dirs[1], n_entry);
    assert!(
        (1.0 * si - n_glass * st).abs() < 1e-9,
        "entry Snell: {} vs {}",
        si,
        n_glass * st
    );
    // Snell at the exit face.
    let n_exit = (pts[2] - Vec3::ZERO).normalize();
    let si2 = sin_to_normal(dirs[1], n_exit);
    let st2 = sin_to_normal(dirs[2], n_exit);
    assert!(
        (n_glass * si2 - 1.0 * st2).abs() < 1e-9,
        "exit Snell: {} vs {}",
        n_glass * si2,
        st2
    );
}

#[test]
fn tangential_and_sagittal_fans_are_distinct() {
    // The release proof (§20). An off-axis bundle through a glass sphere: a fan
    // offset in the meridional plane (tangential) stays in that plane, while a fan
    // offset perpendicular to it (sagittal) bends out of the plane. A meridional
    // engine cannot produce the second, so their distinctness is the sharpest test
    // that the migration happened.
    //
    // Optical axis +X, meridional plane XY (the field tilt is in Y), sagittal
    // direction Z.
    let mut scene = Scene3::new();
    add_glass_sphere(&mut scene, Vec3::ZERO, 6.0, 1.6);

    let field = Vec3::new(1.0, 0.12, 0.0).normalize(); // off-axis in Y
    let start = Vec3::new(-20.0, 0.0, 0.0);
    let cfg = no_ghosts();

    // Tangential marginal: pupil offset in Y (stays in the XY plane).
    let tan_ray = Ray3::new(start + Vec3::new(0.0, 2.0, 0.0), field).unwrap();
    // Sagittal marginal: pupil offset in Z (leaves the XY plane).
    let sag_ray = Ray3::new(start + Vec3::new(0.0, 0.0, 2.0), field).unwrap();

    let tan = primary_dirs(&trace3(&scene, &[launch(tan_ray, LAMBDA_D, 1.0)], &cfg));
    let sag = primary_dirs(&trace3(&scene, &[launch(sag_ray, LAMBDA_D, 1.0)], &cfg));

    let tan_exit = *tan.last().unwrap();
    let sag_exit = *sag.last().unwrap();

    // The tangential ray never leaves the meridional plane.
    assert!(
        tan_exit.z.abs() < 1e-12,
        "tangential ray left the meridional plane: {tan_exit:?}"
    );
    // The sagittal ray does leave it: it carries a real Z bend.
    assert!(
        sag_exit.z.abs() > 1e-3,
        "sagittal ray stayed in-plane; engine is meridional"
    );
    // And the two exit rays are genuinely different.
    assert!(
        (tan_exit - sag_exit).length() > 1e-3,
        "the two fans are indistinguishable"
    );
}

// ---------------------------------------------------------------------------
// Out-of-plane fold
// ---------------------------------------------------------------------------

#[test]
fn a_tilted_mirror_folds_the_beam_onto_a_non_coplanar_detector() {
    // A beam travelling +X meets a mirror tilted so it reflects the beam toward
    // +Z, out of the Z = 0 plane, and lands on a detector that is nowhere near it.
    let mut scene = Scene3::new();
    // Mirror normal bisects +X and -Z, so +X reflects to +Z.
    let mirror_normal = Vec3::new(-1.0, 0.0, 1.0).normalize();
    add_mirror(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        mirror_normal,
        8.0,
        1.0,
    );
    // Detector up at z = +20, facing down.
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 20.0),
        Vec3::NEG_Z,
        10.0,
        10.0,
        [8, 8],
    );

    let ray = Ray3::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::X).unwrap();
    let res = trace3(&scene, &[launch(ray, LAMBDA_D, 1.0)], &no_ghosts());

    let d = detector_named(&res, det);
    assert_eq!(d.hits, 1, "the folded beam missed the detector");
    // The reflected segment genuinely climbs out of z = 0.
    let refl = primary_points(&res);
    assert!(
        refl.last().unwrap().z > 15.0,
        "the beam did not fold upward"
    );
}

// ---------------------------------------------------------------------------
// Aperture clipping
// ---------------------------------------------------------------------------

#[test]
fn a_circular_aperture_passes_the_centre_and_blocks_the_edge_from_both_sides() {
    let mut scene = Scene3::new();
    // A circular opening of radius 2 in an opaque disc of radius 20, in the x = 0
    // plane (normal +X).
    add_circular_aperture(&mut scene, Vec3::ZERO, Vec3::X, 2.0, 20.0);
    let cfg = no_ghosts();

    // Forward, on axis: passes, then escapes.
    let on_axis = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &cfg,
    );
    assert_eq!(
        on_axis.ledger.blocked_aperture, 0.0,
        "the centre should pass"
    );
    assert!(on_axis.ledger.escaped > 0.9);

    // Forward, off axis (y = 5 > opening): blocked.
    let off_axis = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &cfg,
    );
    assert!(
        (off_axis.ledger.blocked_aperture - 1.0).abs() < 1e-12,
        "the edge should block"
    );

    // Backward, off axis: still blocked. An aperture is bidirectional.
    let backward = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(5.0, 5.0, 0.0), Vec3::NEG_X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &cfg,
    );
    assert!(
        (backward.ledger.blocked_aperture - 1.0).abs() < 1e-12,
        "aperture is not bidirectional"
    );
}

#[test]
fn an_annular_aperture_blocks_the_centre_and_passes_the_ring() {
    let mut scene = Scene3::new();
    // A ring opening 3..5 mm in an opaque disc of radius 20, plane x = 0.
    add_annular_aperture(&mut scene, Vec3::ZERO, Vec3::X, 3.0, 5.0, 20.0);
    let cfg = no_ghosts();

    // On axis (r = 0 < inner): the central obstruction blocks it.
    let centre = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &cfg,
    );
    assert!(
        (centre.ledger.blocked_aperture - 1.0).abs() < 1e-12,
        "the centre of an annulus should block, got {}",
        centre.ledger.blocked_aperture
    );

    // In the ring (r = 4): passes.
    let ring = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(-5.0, 4.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &cfg,
    );
    assert_eq!(ring.ledger.blocked_aperture, 0.0, "the ring should pass");
}

#[test]
fn a_mirror_reflection_is_not_flagged_as_tir() {
    // A ray strikes a plain mirror at 45°: the reflected ray is specular, not
    // total internal reflection, so no segment must carry the TIR flag.
    let mut scene = Scene3::new();
    add_mirror(&mut scene, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), 20.0, 1.0);

    let res = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(-5.0, 3.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &no_ghosts(),
    );

    assert!(res.segments.len() >= 2, "the ray should reflect off the mirror");
    assert!(
        res.segments.iter().all(|s| !s.flags.tir),
        "a mirror reflection must not be flagged as TIR"
    );
}

// ---------------------------------------------------------------------------
// Detector area normalization
// ---------------------------------------------------------------------------

#[test]
fn uniform_illumination_gives_uniform_flux_density() {
    // A rectangular detector flooded by a uniform grid of parallel rays. Equal-area
    // cells must see equal flux density (§15.2). This is the measurement a line
    // probe could never make.
    let mut scene = Scene3::new();
    let (hw, hh) = (4.0, 3.0);
    // Face +Z so the detector's local axes align with world X and Y; the grid then
    // maps one-to-one onto its cells.
    let det = add_rect_detector(&mut scene, Vec3::ZERO, Vec3::Z, hw, hh, [4, 3]);

    // A dense uniform grid over the face, each ray carrying equal flux, fired
    // straight down onto it.
    let mut launches = Vec::new();
    let (gx, gy) = (48, 36);
    for i in 0..gx {
        for j in 0..gy {
            let x = -hw + 2.0 * hw * (i as f64 + 0.5) / gx as f64;
            let y = -hh + 2.0 * hh * (j as f64 + 0.5) / gy as f64;
            let ray = Ray3::new(Vec3::new(x, y, 10.0), Vec3::NEG_Z).unwrap();
            launches.push(launch(ray, LAMBDA_D, 1.0));
        }
    }
    let res = trace3(&scene, &launches, &no_ghosts());
    let d = detector_named(&res, det);
    assert_eq!(d.hits, (gx * gy) as u64);

    let density = d.flux_density().expect("a rectangle has area");
    let mean = density.iter().sum::<f64>() / density.len() as f64;
    for (c, v) in density.iter().enumerate() {
        assert!(
            (v - mean).abs() < 0.05 * mean,
            "cell {c} density {v} strays from the mean {mean}"
        );
    }
}

// ---------------------------------------------------------------------------
// A skew ray through an extruded prism
// ---------------------------------------------------------------------------

#[test]
fn a_skew_ray_through_a_prism_matches_hand_vector_snell() {
    // A right-angle glass prism, extruded along Z. A skew ray (non-zero Z) enters
    // the first face and refracts; the tracer's transmitted direction must match a
    // hand vector-Snell computation at that face.
    let mut scene = Scene3::new();
    let n_glass = add_glass_prism(&mut scene, 1.5);

    // Enter the front face (x = -? ) travelling mostly +X with a Z skew.
    let ray = Ray3::new(Vec3::new(-10.0, 0.5, 0.4), Vec3::new(1.0, 0.0, 0.06)).unwrap();
    let res = trace3(&scene, &[launch(ray, LAMBDA_D, 1.0)], &no_ghosts());
    let dirs = primary_dirs(&res);
    let pts = primary_points(&res);
    assert!(dirs.len() >= 2, "the ray missed the prism");

    // The first face is the plane x = -2 (front of the extruded triangle), normal
    // -X in world. Hand vector-Snell there.
    let face_n = Vec3::NEG_X;
    let n = if dirs[0].dot(face_n) < 0.0 {
        face_n
    } else {
        -face_n
    };
    let expect = refract3(dirs[0], n, 1.0, n_glass).expect("front face refracts");
    assert!(
        (dirs[1] - expect).length() < 1e-9,
        "front-face refraction disagrees with hand Snell: {:?} vs {:?}",
        dirs[1],
        expect
    );
    let _ = pts;
}

// ---------------------------------------------------------------------------
// An explicit beamsplitter surface splits deterministically
// ---------------------------------------------------------------------------

#[test]
fn an_explicit_beamsplitter_splits_reflected_and_transmitted() {
    // A 50/50 plate: an explicit R/T surface. The transmitted branch continues
    // undeviated (same medium both sides), the reflected branch is specular, and
    // each carries half the flux.
    let mut scene = Scene3::new();
    add_beamsplitter(&mut scene, Vec3::ZERO, Vec3::X, 5.0, 0.5, 0.5);
    // A detector behind (transmitted) and one above after a fold would need a
    // mirror; here we read the ledger and segment flags instead.
    let ray = Ray3::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap();
    let cfg = TraceConfig3 {
        ghosts: true,
        ..TraceConfig3::default()
    };
    let res = trace3(&scene, &[launch(ray, LAMBDA_D, 1.0)], &cfg);

    // Two children leave the plate. One is specular-reflected (dir -X), one
    // transmitted undeviated (dir +X), each with half the flux.
    let leaving: Vec<_> = res
        .segments
        .iter()
        .filter(|s| s.a.x.abs() < 1e-9) // segments starting at the plate
        .collect();
    assert_eq!(
        leaving.len(),
        2,
        "expected a reflected and a transmitted branch"
    );
    let refl = leaving.iter().find(|s| (s.b.x - s.a.x) < 0.0).unwrap();
    let tran = leaving.iter().find(|s| (s.b.x - s.a.x) > 0.0).unwrap();
    assert!((refl.flux_start - 0.5).abs() < 1e-12, "reflected half");
    assert!((tran.flux_start - 0.5).abs() < 1e-12, "transmitted half");
}
