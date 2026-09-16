//! The source sampler feeds the reference tracer (§6.10, PR 3D-06).
//!
//! `trace_scene3` emits launch rays from a scene's sources and traces them in one
//! call. These tests check the seam: a collimated beam's rays all reach an area
//! detector with their flux intact, and the ray count is exactly what the sampler
//! promised.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::id::LayerId;
use oc_core::scene3::{
    AngularDistribution3, SamplingSequence, Scene3, Source3, SourceSampling, SpatialDistribution3,
};
use oc_core::vec3::{Isometry3, Vec3};
use oc_trace::{launch_from_scene, trace_scene3};

fn collimated_disc(scene: &mut Scene3, at_x: f64, radius_mm: f64, samples: u32) {
    let id = scene.alloc_source_id();
    scene.sources.push(Source3 {
        id,
        transform: Isometry3::from_pos(Vec3::new(at_x, 0.0, 0.0)),
        spatial: SpatialDistribution3::Disc { radius_mm },
        angular: AngularDistribution3::Direction { local_dir: Vec3::X },
        spectrum: oc_core::scene::Spectrum::Monochromatic(oc_core::material::LAMBDA_D),
        flux: 1.0,
        samples: SourceSampling {
            spatial_samples: samples,
            angular_samples: 1,
            seed: 0,
            sequence: SamplingSequence::Grid,
        },
    });
    let _ = LayerId(0);
}

#[test]
fn a_collimated_disc_delivers_all_its_flux_to_a_detector() {
    let mut scene = Scene3::new();
    collimated_disc(&mut scene, -10.0, 2.0, 100);
    // A detector large enough to catch the whole 2 mm disc, facing the beam.
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        5.0,
        5.0,
        [16, 16],
    );
    scene.validate().expect("scene is valid");

    assert_eq!(launch_from_scene(&scene).len(), 100, "sampler ray count");

    let res = trace_scene3(&scene, &no_ghosts());
    let d = detector_named(&res, det);
    assert_eq!(d.hits, 100, "every ray should land");
    assert!(
        (d.total_incident - 1.0).abs() < 1e-12,
        "all flux collected: {}",
        d.total_incident
    );
    // The ledger balances: everything launched was detected.
    assert!(
        res.ledger.residual() < 1e-12,
        "ledger residual {}",
        res.ledger.residual()
    );
}

#[test]
fn a_wider_beam_spills_past_a_small_detector() {
    // Only the rays within the detector's half-width are collected; the rest
    // escape. The ledger still accounts for every photon.
    let mut scene = Scene3::new();
    collimated_disc(&mut scene, -10.0, 5.0, 400);
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        1.0,
        1.0,
        [8, 8],
    );
    scene.validate().expect("scene is valid");

    let res = trace_scene3(&scene, &no_ghosts());
    let d = detector_named(&res, det);
    assert!(
        d.hits > 0 && (d.hits as usize) < 400,
        "some but not all: {}",
        d.hits
    );
    assert!(d.total_incident < 1.0, "partial flux: {}", d.total_incident);
    assert!(
        res.ledger.residual() < 1e-12,
        "ledger must still close: {}",
        res.ledger.residual()
    );
}
