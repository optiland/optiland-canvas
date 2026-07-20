//! Aperture masks across their opening shapes, from both directions (§6.8,
//! PR 3D-06).
//!
//! The circular opening is pinned in `analytic3`; here the rectangle and polygon
//! carry the same contract: a ray through the clear opening passes with no event,
//! a ray outside it but inside the mask is absorbed, and the answer does not
//! depend on which way the ray travels — an aperture is a bidirectional physical
//! event, not a one-sided clip.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::material::LAMBDA_D;
use oc_core::scene3::Scene3;
use oc_core::vec::Vec2;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::trace3;

/// The flux blocked when a single ray at height `y` crosses the aperture plane at
/// `x = 0`, travelling in `dir`.
fn blocked(scene: &Scene3, y: f64, dir: Vec3) -> f64 {
    let x0 = if dir.x > 0.0 { -5.0 } else { 5.0 };
    let res = trace3(
        scene,
        &[launch(
            Ray3::new(Vec3::new(x0, y, 0.0), dir).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &no_ghosts(),
    );
    res.ledger.blocked_aperture
}

#[test]
fn a_rectangular_aperture_passes_the_opening_and_blocks_the_mask_both_ways() {
    let mut scene = Scene3::new();
    // A 2×1 (half-extent) opening in an opaque disc of radius 20, in x = 0.
    add_rect_aperture(&mut scene, Vec3::ZERO, Vec3::X, 2.0, 1.0, 20.0);
    scene.validate().unwrap();

    // Inside the opening (|y| < 1): passes.
    assert_eq!(
        blocked(&scene, 0.5, Vec3::X),
        0.0,
        "inside the opening should pass"
    );
    // Above the opening (y = 3 > 1) but inside the mask: blocked, both ways.
    assert!(
        (blocked(&scene, 3.0, Vec3::X) - 1.0).abs() < 1e-12,
        "forward edge should block"
    );
    assert!(
        (blocked(&scene, 3.0, Vec3::NEG_X) - 1.0).abs() < 1e-12,
        "aperture is not bidirectional"
    );
}

#[test]
fn a_polygon_aperture_blocks_outside_its_triangular_opening() {
    let mut scene = Scene3::new();
    // A triangle opening (roughly centred, spanning a few mm) in a disc mask.
    let triangle = vec![
        Vec2::new(-3.0, -2.0),
        Vec2::new(3.0, -2.0),
        Vec2::new(0.0, 3.0),
    ];
    add_polygon_aperture(&mut scene, Vec3::ZERO, Vec3::X, triangle, 20.0);
    scene.validate().unwrap();

    // Inside the triangle (near its centroid, y = 0): passes.
    assert_eq!(
        blocked(&scene, 0.0, Vec3::X),
        0.0,
        "inside the polygon should pass"
    );
    // Above the apex (y = 4 > 3): blocked, and bidirectionally.
    assert!(
        (blocked(&scene, 4.0, Vec3::X) - 1.0).abs() < 1e-12,
        "outside the polygon should block"
    );
    assert!(
        (blocked(&scene, 4.0, Vec3::NEG_X) - 1.0).abs() < 1e-12,
        "polygon aperture is not bidirectional"
    );
}
