//! A detector that counts its front only: light arriving at its back goes on as if the face were not there.
//!
//! The scene is one arm of an interferometer: a beam travelling +X crosses a probe, meets a mirror that sends it
//! back, crosses the probe again from behind, and ends on a catcher. Counting both sides, the probe reads the beam
//! twice; counting its front, once, and the returning beam passes it unchanged.

mod common;
use common::*;

use oc_core::id::ObjectId;
use oc_core::material::LAMBDA_D;
use oc_core::scene3::{Detector3, DetectorSides, DetectorTermination};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::trace3;

/// The arm: the probe at x = 0 with its front to the light going out (−X), the mirror at x = 10, the catcher at
/// x = −20 facing the light coming back.
fn arm(sides: DetectorSides, termination: DetectorTermination) -> (Scene3, ObjectId, ObjectId) {
    let mut scene = Scene3::new();
    let probe = add_rect_detector(&mut scene, Vec3::ZERO, Vec3::NEG_X, 5.0, 5.0, [4, 4]);
    add_mirror(&mut scene, Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_X, 8.0, 1.0);
    let catcher = add_rect_detector(
        &mut scene,
        Vec3::new(-20.0, 0.0, 0.0),
        Vec3::X,
        5.0,
        5.0,
        [4, 4],
    );
    let d = scene
        .detectors
        .iter_mut()
        .find(|d| d.common.id == probe)
        .unwrap();
    d.sides = sides;
    d.termination = termination;
    (scene, probe, catcher)
}

fn beam() -> Vec<oc_trace::LaunchRay3> {
    vec![launch(
        Ray3::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::X).unwrap(),
        LAMBDA_D,
        1.0,
    )]
}

#[test]
fn a_probe_counting_both_sides_reads_the_beam_going_out_and_coming_back() {
    let (scene, probe, catcher) = arm(DetectorSides::Both, DetectorTermination::ProbeAndContinue);
    let res = trace3(&scene, &beam(), &no_ghosts());
    assert_eq!(detector_named(&res, probe).hits, 2);
    assert_eq!(detector_named(&res, catcher).hits, 1);
}

#[test]
fn a_probe_counting_its_front_reads_the_beam_once_and_lets_the_return_pass() {
    let (scene, probe, catcher) = arm(DetectorSides::Front, DetectorTermination::ProbeAndContinue);
    let res = trace3(&scene, &beam(), &no_ghosts());
    let p = detector_named(&res, probe);
    assert_eq!(
        p.hits, 1,
        "the probe counted the returning beam at its back"
    );
    assert!(
        (p.total_incident - 1.0).abs() < 1e-12,
        "the beam going out, all of it: {}",
        p.total_incident
    );
    assert_eq!(
        detector_named(&res, catcher).hits,
        1,
        "the returning beam did not go on"
    );
}

#[test]
fn a_front_only_face_that_stops_light_is_not_there_for_light_at_its_back() {
    // Turned away from the beam going out: the beam reaches its back, so it neither counts nor stops it.
    let (mut scene, probe, catcher) = arm(DetectorSides::Front, DetectorTermination::Terminate);
    scene
        .detectors
        .iter_mut()
        .find(|d| d.common.id == probe)
        .unwrap()
        .common
        .transform =
        oc_core::vec3::Isometry3::new(Vec3::ZERO, rotation_arc(Vec3::Z, Vec3::X)).unwrap();
    let res = trace3(&scene, &beam(), &no_ghosts());
    // The return arrives at its front and ends there; the catcher sees nothing.
    assert_eq!(detector_named(&res, probe).hits, 1);
    assert_eq!(detector_named(&res, catcher).hits, 0);
}

#[test]
fn a_scene_file_says_front_only_and_one_written_before_counts_both() {
    let (scene, probe, _) = arm(DetectorSides::Front, DetectorTermination::ProbeAndContinue);
    let d = scene
        .detectors
        .iter()
        .find(|d| d.common.id == probe)
        .unwrap();
    let json = serde_json::to_value(d).unwrap();
    assert_eq!(json["sides"], "Front");
    let mut old = json.clone();
    old.as_object_mut().unwrap().remove("sides");
    assert_eq!(
        serde_json::from_value::<Detector3>(old).unwrap().sides,
        DetectorSides::Both
    );
    // Both is what a file says by saying nothing.
    let both = Detector3 {
        sides: DetectorSides::Both,
        ..d.clone()
    };
    assert!(serde_json::to_value(&both).unwrap().get("sides").is_none());
}

fn rotation_arc(from: Vec3, to: Vec3) -> [f64; 4] {
    let q = glam::DQuat::from_rotation_arc(from, to);
    [q.x, q.y, q.z, q.w]
}
