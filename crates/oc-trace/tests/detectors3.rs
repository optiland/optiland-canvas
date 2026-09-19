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
fn a_disc_detector_bins_into_the_squares_of_its_bounding_box() {
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
    // A 3 mm disc binned 6 by 6 has cells one millimetre square, because the
    // grid is the bounding box and not the disc. Sharing the disc's own area
    // out over the same cells would say 0.785 mm², and every density read off
    // it would be 4/π high.
    assert!(
        (d.cell_area_mm2.unwrap() - 1.0).abs() < 1e-12,
        "cell area {:?}",
        d.cell_area_mm2
    );
    let density = d.flux_density().expect("disc has a flux density");
    assert_eq!(density.len(), 36);
    let total_density: f64 = density.iter().sum();
    assert!((total_density - 2.0).abs() < 1e-9, "{total_density}");
}

/// Two shapes, the same light, the same reading.
///
/// A disc bins into the squares of its bounding box exactly as a rectangle
/// does, so a patch of light landing well inside both has to read the same
/// irradiance on both. It did not while the disc divided its own area by the
/// cell count: the disc came out 4/π high, and nothing in the suite asked.
#[test]
fn a_disc_and_a_rectangle_read_the_same_irradiance_from_the_same_light() {
    // One millimetre of flux spread evenly over one square millimetre, aimed
    // at the cell spanning (0,0) to (1,1) on both faces.
    let beam: Vec<_> = (0..8)
        .flat_map(|i| (0..8).map(move |j| (i, j)))
        .map(|(i, j)| {
            let y = (i as f64 + 0.5) / 8.0;
            let z = (j as f64 + 0.5) / 8.0;
            launch(
                Ray3::new(Vec3::new(-5.0, y, z), Vec3::X).unwrap(),
                LAMBDA_D,
                1.0 / 64.0,
            )
        })
        .collect();

    let read = |scene: &Scene3, det| {
        let res = trace3(scene, &beam, &no_ghosts());
        let d = detector_named(&res, det);
        assert_eq!(d.hits, 64, "the whole patch lands");
        let e = d.irradiance().expect("an area detector has irradiance");
        e.into_iter().fold(0.0f64, f64::max)
    };

    let mut disc = Scene3::new();
    let a = add_disc_detector(&mut disc, Vec3::new(10.0, 0.0, 0.0), -Vec3::X, 3.0, [6, 6]);
    disc.validate().unwrap();

    let mut rect = Scene3::new();
    let b = add_rect_detector(
        &mut rect,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        3.0,
        3.0,
        [6, 6],
    );
    rect.validate().unwrap();

    // One unit of flux over one square millimetre is one unit of irradiance,
    // and the shape of the surround has nothing to say about it.
    for (name, got) in [("disc", read(&disc, a)), ("rectangle", read(&rect, b))] {
        assert!((got - 1.0).abs() < 1e-12, "{name} reads {got}");
    }
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
