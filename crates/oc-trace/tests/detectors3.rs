//! Area detectors across their shapes: rectangle, disc, and annulus (§6.9,
//! PR 3D-06).
//!
//! The rectangle uniformity and cell area are pinned in `analytic3`; here the
//! disc and annulus carry their weight — a disc reports an area-normalized
//! density, an annulus ignores the rays that pass through its hole, and a
//! rectangle splits a symmetric beam evenly across a 2D bin grid.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::material::LAMBDA_D;
use oc_core::scene3::Scene3;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::trace3;

#[test]
fn a_disc_detector_reports_an_area_normalized_density() {
    let mut scene = Scene3::new();
    let det = add_disc_detector(&mut scene, Vec3::new(10.0, 0.0, 0.0), -Vec3::X, 3.0, [6, 6]);
    scene.validate().unwrap();

    // Two rays inside the disc, at different radii.
    let rays = [
        launch(
            Ray3::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        ),
        launch(
            Ray3::new(Vec3::new(-5.0, 2.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        ),
    ];
    let res = trace3(&scene, &rays, &no_ghosts());
    let d = detector_named(&res, det);

    assert_eq!(d.hits, 2);
    let expected_cell = std::f64::consts::PI * 9.0 / 36.0;
    assert!(
        (d.cell_area_mm2.unwrap() - expected_cell).abs() < 1e-12,
        "cell area {:?}",
        d.cell_area_mm2
    );
    // Density is flux / area, and it exists because the disc has area.
    let density = d.flux_density().expect("disc has a flux density");
    assert_eq!(density.len(), 36);
    let total_density: f64 = density.iter().sum();
    assert!((total_density - 2.0 / expected_cell).abs() < 1e-9);
}

#[test]
fn an_annulus_detector_ignores_rays_through_its_hole() {
    let mut scene = Scene3::new();
    let det = add_annulus_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        2.0,
        5.0,
        [8, 8],
    );
    scene.validate().unwrap();

    // Down the axis: inside the hole (r = 0 < inner), so it is not collected — it
    // passes through and escapes.
    let hole = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &no_ghosts(),
    );
    assert_eq!(
        detector_named(&hole, det).hits,
        0,
        "the hole should not detect"
    );
    assert!(
        hole.ledger.escaped > 0.9,
        "the ray should pass through the hole"
    );

    // In the ring (inner < r < outer): collected.
    let ring = trace3(
        &scene,
        &[launch(
            Ray3::new(Vec3::new(-5.0, 3.5, 0.0), Vec3::X).unwrap(),
            LAMBDA_D,
            1.0,
        )],
        &no_ghosts(),
    );
    assert_eq!(detector_named(&ring, det).hits, 1, "the ring should detect");
    assert!((detector_named(&ring, det).total_incident - 1.0).abs() < 1e-12);
}

#[test]
fn a_rectangle_detector_splits_a_symmetric_beam_evenly_across_its_bins() {
    // Four launch rays, one into each quadrant of a 2×2 grid. By symmetry each
    // cell collects a quarter of the flux, whichever way the detector's local
    // frame is rolled.
    let mut scene = Scene3::new();
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        4.0,
        4.0,
        [2, 2],
    );
    scene.validate().unwrap();

    let rays: Vec<_> = [(2.0, 2.0), (2.0, -2.0), (-2.0, 2.0), (-2.0, -2.0)]
        .into_iter()
        .map(|(y, z)| {
            launch(
                Ray3::new(Vec3::new(-5.0, y, z), Vec3::X).unwrap(),
                LAMBDA_D,
                1.0,
            )
        })
        .collect();
    let res = trace3(&scene, &rays, &no_ghosts());
    let d = detector_named(&res, det);

    assert_eq!(d.hits, 4);
    assert_eq!(d.incident_flux.len(), 4);
    for (i, f) in d.incident_flux.iter().enumerate() {
        assert!((f - 1.0).abs() < 1e-12, "cell {i} got {f}, expected 1.0");
    }
}
