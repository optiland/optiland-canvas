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
    // A 3 mm disc binned 6 by 6 has pixels one millimetre square, because the
    // grid is the bounding box and not the disc. Sharing the disc's own area
    // out over the same pixels would say 0.785 mm², and every density read
    // off it would be 4/π high.
    assert!(
        (d.pixel_area_mm2.unwrap() - 1.0).abs() < 1e-12,
        "pixel area {:?}",
        d.pixel_area_mm2
    );
    let density = d.flux_density().expect("disc has a flux density");
    assert_eq!(density.len(), 36);
    // The ray down the axis lands in a whole pixel and reads its flux over
    // one square millimetre. The ray two millimetres out lands in a pixel the
    // rim cuts, and reads its flux over the part on the face (A12), so the
    // readings times the covered areas give back both rays.
    let whole = density
        .iter()
        .zip(&d.covered_area_mm2)
        .filter(|(e, a)| **e > 0.0 && (**a - 1.0).abs() < 1e-12)
        .count();
    assert_eq!(whole, 1, "one ray in a whole pixel");
    let collected: f64 = density.iter().zip(&d.covered_area_mm2).map(|(e, a)| e * a).sum();
    assert!((collected - 2.0).abs() < 1e-12, "{collected}");
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

// ---------------------------------------------------------------------------
// Pixels at a round face's rim (A12)
// ---------------------------------------------------------------------------

/// A uniform beam, `n` by `n` rays over the square `[-half, half]²` across
/// the world `y` and `z` axes, travelling along `+x`, each carrying its
/// square's share of one unit of flux per square millimetre.
fn uniform_beam(half: f64, n: usize) -> Vec<oc_trace::LaunchRay3> {
    let step = 2.0 * half / n as f64;
    (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| {
            let y = -half + (i as f64 + 0.5) * step;
            let z = -half + (j as f64 + 0.5) * step;
            launch(
                Ray3::new(Vec3::new(-5.0, y, z), Vec3::X).unwrap(),
                LAMBDA_D,
                step * step,
            )
        })
        .collect()
}

/// Under uniform light every pixel reads the same irradiance, rim pixels
/// included, and the readings times the covered areas give back the flux.
///
/// Before the covered areas, a rim pixel divided its flux by the whole square
/// and read low by its uncovered share: a pixel 30% on the face read 0.3 of
/// the light on it. The beam's rays are spaced so the pixel edges fall
/// between them, so whole pixels read exactly one; what is left is the beam
/// sampling the arc: 0.3% at worst on the disc and 0.2% on the annulus, held
/// to 2% here. The old division was 25 to 75% low on the same pixels.
#[test]
fn rim_pixels_read_the_irradiance_of_the_light_on_them() {
    type AddFace = fn(&mut Scene3) -> oc_core::id::ObjectId;
    let faces: [(&str, AddFace, usize); 2] = [
        (
            "disc",
            |s| add_disc_detector(s, Vec3::new(10.0, 0.0, 0.0), -Vec3::X, 3.0, [9, 9]),
            297, // 33 rays to a pixel's side
        ),
        (
            "annulus",
            |s| add_annulus_detector(s, Vec3::new(10.0, 0.0, 0.0), -Vec3::X, 1.2, 3.0, [8, 8]),
            296, // 37 rays to a pixel's side
        ),
    ];
    for (name, add, n) in faces {
        let mut scene = Scene3::new();
        let det = add(&mut scene);
        scene.validate().unwrap();
        let res = trace3(&scene, &uniform_beam(3.0, n), &no_ghosts());
        let d = detector_named(&res, det);
        let e = d.irradiance().expect("an area detector");
        let pixel = d.pixel_area_mm2.unwrap();

        // A12: exact, whatever the sampling.
        let back: f64 = e.iter().zip(&d.covered_area_mm2).map(|(e, a)| e * a).sum();
        assert!(
            (back - d.total_incident).abs() < 1e-12 * d.total_incident,
            "{name}: {back} vs {}",
            d.total_incident
        );

        let mut worst: f64 = 0.0;
        let mut rim = 0;
        for (i, (e, a)) in e.iter().zip(&d.covered_area_mm2).enumerate() {
            let share = a / pixel;
            if share >= 0.999_999_999 {
                assert!((e - 1.0).abs() < 1e-9, "{name}: whole pixel {i} reads {e}");
            } else if share >= 0.25 {
                rim += 1;
                worst = worst.max((e - 1.0).abs());
            }
        }
        assert!(rim >= 8, "{name}: the test needs rim pixels, found {rim}");
        assert!(worst < 0.02, "{name}: a rim pixel is {:.1}% off", worst * 100.0);
    }
}

/// The peak irradiance skips slivers, and only the peak does.
///
/// One ray into a whole pixel and one into the corner of a disc's box, where
/// the rim leaves a few hundredths of a square millimetre on the face. Both
/// readings are right, and the sliver's is thirty times the other, because a
/// single ray over a sliver of area is a spike. The peak reports the whole
/// pixel.
#[test]
fn the_peak_irradiance_is_not_a_sliver_at_the_rim() {
    use oc_core::scene3::DetectorMeasure;

    let mut scene = Scene3::new();
    let det = add_disc_detector(&mut scene, Vec3::new(10.0, 0.0, 0.0), -Vec3::X, 3.0, [6, 6]);
    set_detector_measure(&mut scene, det, DetectorMeasure::Irradiance);
    scene.validate().unwrap();
    let pose = scene.detectors[0].common.transform;
    let at = |u: f64, v: f64| {
        let p = pose.transform_point(Vec3::new(u, v, 0.0));
        launch(Ray3::new(p - Vec3::X * 5.0, Vec3::X).unwrap(), LAMBDA_D, 1.0)
    };
    // (2.05, 2.05) is on the disc, 2.9 mm out, in the pixel [2, 3] x [2, 3].
    let res = trace3(&scene, &[at(0.5, 0.5), at(2.05, 2.05)], &no_ghosts());
    let d = detector_named(&res, det);
    assert_eq!(d.hits, 2);

    let sliver = oc_core::geom3::primitives::disc_rect_overlap_area(3.0, 2.0, 3.0, 2.0, 3.0);
    assert!(sliver > 0.0 && sliver < 0.05, "{sliver}");
    let (readings, _) = d.readings();
    let spike = readings.iter().copied().fold(0.0, f64::max);
    assert!((spike - 1.0 / sliver).abs() < 1e-9, "the sliver reads {spike}");
    assert_eq!(d.peak_reading(), Some(1.0), "the peak is the whole pixel's");

    // Power divides by nothing, so it has no spike to skip.
    set_detector_measure(&mut scene, det, DetectorMeasure::Power);
    let res = trace3(&scene, &[at(0.5, 0.5), at(2.05, 2.05)], &no_ghosts());
    assert_eq!(detector_named(&res, det).peak_reading(), Some(1.0));
}
