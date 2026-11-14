//! `Scene3` schema, validation, and `.ocanvas` v1 → v2 migration.
//!
//! What this suite owes (contract §11.5, §15.7):
//!
//! - every v1 corpus file loads and re-emits as stable v2;
//! - v1 → v2 migration is deterministic byte-for-byte after normalization;
//! - v2 → v2 preserves all f64 values exactly through `serde_json`;
//! - a future `version` is rejected, not mis-parsed;
//! - malformed quaternions, non-finite geometry, invalid profiles, and
//!   duplicate ids are rejected;
//! - v2 legacy-planar objects retain identical v1 planar ray results;
//! - **no third dimension is invented**: a migrated v1 body is `PlanarOnly`,
//!   never silently revolved, extruded, or given a thickness;
//! - stable ids survive insertion and removal of unrelated objects (§3.8).
//!
//! The planar-ray-equivalence half lives in `oc-trace`
//! (`tests/planar_equivalence.rs`), which has a tracer to compare with.
#![cfg(feature = "scene3")]

use oc_core::id::{BodyId, LayerId, ObjectId, PatchId, SourceId};
use oc_core::migrate_v1::{load_any, migrate_v1_to_v2, LoadReport};
use oc_core::profile2::{Curve2, ProfileConstruction3};
use oc_core::scene::Scene;
use oc_core::scene3::{
    can_promote_to_extruded, can_promote_to_revolved, promote_body, AngularDistribution3, Body3,
    DetectorGeometry3, DetectorMeasure, ObjectCommon, PromotionCheck, Scene3, SceneError,
    SolidGeometry3, SpatialDistribution3, SurfaceResponse,
};
use oc_core::vec3::Vec3;

/// The real v1 corpus: the `.ocanvas` documents the browser actually ships.
///
/// Read from their committed location rather than copied here. A copy would be a
/// second corpus that drifts, and the claim under test is that *these* files
/// load.
const V1_CORPUS: &[(&str, &str)] = &[
    (
        "aperture",
        include_str!("../../../web/public/catalog/openuc2/assets/aperture.ocanvas"),
    ),
    (
        "detector",
        include_str!("../../../web/public/catalog/openuc2/assets/detector.ocanvas"),
    ),
    (
        "mirror",
        include_str!("../../../web/public/catalog/openuc2/assets/mirror.ocanvas"),
    ),
    (
        "source",
        include_str!("../../../web/public/catalog/openuc2/assets/source.ocanvas"),
    ),
];

// ---------------------------------------------------------------------------
// The v1 corpus loads
// ---------------------------------------------------------------------------

#[test]
fn every_v1_corpus_file_loads_through_the_migration() {
    for (name, json) in V1_CORPUS {
        let (s3, report) = load_any(json).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            matches!(report, LoadReport::MigratedFromV1(_)),
            "{name}: should have been recognised as v1"
        );
        s3.validate()
            .unwrap_or_else(|e| panic!("{name}: migrated scene does not validate: {e}"));
        assert_eq!(s3.version, 3, "{name}: migrated scenes are held as v3");
        assert_eq!(s3.dimension, 3, "{name}");
    }
}

#[test]
fn a_migrated_scene_re_emits_as_loadable_v3_and_v2() {
    for (name, json) in V1_CORPUS {
        let (s3, _) = load_any(json).unwrap();
        let out = s3.to_json();
        let (again, report) = load_any(&out).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(report, LoadReport::V3, "{name}: re-emitted doc is not v3");
        assert_eq!(again.to_json(), out, "{name}: v3 is not stable on re-emit");
        // And as v2, for older readers: nothing of v3 is used, so nothing is lost.
        let (v2, losses) = s3.to_json_v2();
        assert!(losses.is_empty(), "{name}: {losses:?}");
        let (_, report) = load_any(&v2).unwrap();
        assert_eq!(report, LoadReport::V2, "{name}");
    }
}

// ---------------------------------------------------------------------------
// No third dimension is invented
// ---------------------------------------------------------------------------

#[test]
fn no_migrated_body_gains_a_dimension_it_did_not_have() {
    // The exit criterion, and the one that matters most. A v1 element does not
    // record whether it is revolved or extruded, so nothing may decide for it.
    for (name, json) in V1_CORPUS {
        let (s3, _) = load_any(json).unwrap();
        for b in &s3.bodies {
            match b.construction() {
                Some(ProfileConstruction3::PlanarOnly { plane_z_mm }) => {
                    assert_eq!(plane_z_mm, 0.0, "{name}: v1 lives at z=0");
                }
                other => panic!("{name}: body was given a construction: {other:?}"),
            }
            assert!(!b.is_3d(), "{name}: body claims to be traceable in 3D");
        }
    }
}

#[test]
fn no_migrated_detector_gains_an_area() {
    // A v1 detector is a line segment. A line has no area, so its bins have no
    // mm² and its readings are not a flux density. Giving it a height would
    // invent a number every downstream irradiance claim would then be built on.
    let (s3, _) = load_any(V1_CORPUS[1].1).unwrap();
    assert_eq!(s3.detectors.len(), 1);
    let d = &s3.detectors[0];
    assert!(
        matches!(d.geometry, DetectorGeometry3::LineProbeLegacy { .. }),
        "detector was given an area: {:?}",
        d.geometry
    );
    assert!(!d.geometry.has_area());
}

/// A document written before detectors had a measure still reads power.
///
/// The field is new and every `.ocanvas` v2 in the wild predates it, so its
/// absence has to mean the thing those files meant. Serde's default does the
/// work; this is the test that says so, because a default that silently
/// changed would change every old scene's numbers without touching a byte of
/// them.
#[test]
fn a_detector_written_before_the_measure_still_reads_power() {
    let mut scene = Scene3::new();
    scene.add_area_detector(Vec3::new(10.0, 0.0, 0.0), -Vec3::X, 4.0, 4.0, [8, 8]);
    let json = serde_json::to_string(&scene).expect("serializes");

    // The field is there when it is written out.
    assert!(json.contains("measure"), "the measure is serialized");

    // And a document with the field struck out still loads, as power.
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parses");
    value["detectors"][0]
        .as_object_mut()
        .expect("a detector object")
        .remove("measure")
        .expect("there was a measure to remove");
    let older: Scene3 = serde_json::from_value(value).expect("an older document loads");
    assert_eq!(older.detectors[0].measure, DetectorMeasure::Power);
    older.validate().expect("and it is valid");
}

/// The direction grid is laid out the way the face grid is.
///
/// `bins[0]` runs fastest in both, so the same reader draws either picture:
/// the face map's columns are millimetres across the face and the direction
/// map's are degrees away from the normal.
#[test]
fn the_direction_grid_runs_rings_fastest() {
    let mut scene = Scene3::new();
    let det = scene.add_area_detector(Vec3::ZERO, -Vec3::X, 4.0, 4.0, [5, 4]);
    let d = scene
        .detectors
        .iter_mut()
        .find(|d| d.id() == det)
        .expect("the detector");
    d.measure = DetectorMeasure::Intensity {
        half_angle_deg: 50.0,
        bins: [5, 4],
        layout: Default::default(),
    };

    let d = &scene.detectors[0];
    let cells = d.direction_cell_solid_angles().expect("a direction grid");
    assert_eq!(cells.len(), 20);

    // Every sector holds the same set of ring areas, so the grid repeats with
    // a period of the ring count.
    for sector in 1..4 {
        for ring in 0..5 {
            assert!(
                (cells[sector * 5 + ring] - cells[ring]).abs() < 1e-15,
                "sector {sector} ring {ring} differs from sector 0"
            );
        }
    }
    // Rings grow outward: the cap nearest the axis is the smallest.
    for ring in 1..5 {
        assert!(cells[ring] > cells[ring - 1], "ring {ring} is not larger");
    }

    // A ray straight down the normal lands in ring 0, and one at 49 degrees in
    // the last ring, whichever side of the face it arrives from.
    for sign in [-1.0, 1.0] {
        let on_axis = d
            .direction_cell(Vec3::new(0.0, 0.0, sign))
            .expect("on axis is inside the cone");
        assert_eq!(on_axis % 5, 0, "on axis is ring 0");
        let t = 49f64.to_radians();
        let far = d
            .direction_cell(Vec3::new(t.sin(), 0.0, sign * t.cos()))
            .expect("49 degrees is inside a 50 degree cone");
        assert_eq!(far % 5, 4, "49 degrees is the outer ring");
    }
    // And one past the cone lands nowhere.
    let t = 51f64.to_radians();
    assert!(d.direction_cell(Vec3::new(t.sin(), 0.0, t.cos())).is_none());
}

#[test]
fn no_migrated_source_gains_a_solid_angle_or_an_area() {
    // A v1 collimated source is a line, not a rectangle. A v1 point fan is
    // equal-angle in one plane, not uniform in solid angle. Promoting either
    // would silently change the radiometry of every existing scene.
    let (s3, _) = load_any(V1_CORPUS[3].1).unwrap();
    assert_eq!(s3.sources.len(), 1);
    let s = &s3.sources[0];
    assert!(
        matches!(s.spatial, SpatialDistribution3::LineLegacy { .. }),
        "collimated source gained an area: {:?}",
        s.spatial
    );
    assert!(s.spatial.is_legacy());

    // And the point-fan half, built directly since the corpus has no point source.
    let mut v1 = Scene::new();
    v1.sources.push(oc_core::Source {
        kind: oc_core::SourceKind::Point { half_angle: 0.3 },
        transform: oc_core::Isometry2::IDENTITY,
        spectrum: oc_core::Spectrum::Monochromatic(0.5876),
        flux: 1.0,
        rays: 21,
    });
    let (s3, _) = migrate_v1_to_v2(&v1).unwrap();
    assert!(
        matches!(
            s3.sources[0].angular,
            AngularDistribution3::PlanarFanLegacy { half_angle_rad } if half_angle_rad == 0.3
        ),
        "point fan became a cone: {:?}",
        s3.sources[0].angular
    );
    assert!(s3.sources[0].angular.is_legacy());
}

#[test]
fn the_migration_report_says_the_dimension_is_unresolved() {
    // Silence would be the failure. The report is how the UI knows to say so.
    let (_, report) = load_any(V1_CORPUS[2].1).unwrap();
    let LoadReport::MigratedFromV1(r) = report else {
        panic!("expected a v1 migration")
    };
    assert!(r.has_unresolved_dimension());
    assert_eq!(r.planar_only_bodies.len(), 1);
    assert!(
        r.notes.iter().any(|n| n.contains("PlanarOnly")),
        "report does not explain itself: {:?}",
        r.notes
    );
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn migration_is_deterministic_byte_for_byte() {
    // Ids are allocated in a fixed traversal order, so the same v1 file always
    // produces the same v2 file. Without this an undo snapshot or a git diff
    // would churn on every load.
    for (name, json) in V1_CORPUS {
        let a = load_any(json).unwrap().0.to_json();
        let b = load_any(json).unwrap().0.to_json();
        assert_eq!(a, b, "{name}: migration is not deterministic");
    }
}

#[test]
fn migrated_ids_are_dense_and_start_at_zero() {
    // Not a requirement, but it pins the traversal order so a change to it is a
    // visible test failure rather than a silent renumbering of every document.
    let (s3, _) = load_any(V1_CORPUS[0].1).unwrap();
    // Two 4-edge absorber elements: body, 4 patches, body, 4 patches.
    assert_eq!(s3.bodies[0].common.id, ObjectId(0));
    assert_eq!(s3.bodies[1].common.id, ObjectId(5));
    assert_eq!(s3.next_id, 10);
}

// ---------------------------------------------------------------------------
// v2 round trip
// ---------------------------------------------------------------------------

/// A scene using every awkward value a document format has to survive.
fn hostile_scene() -> Scene3 {
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    // Values chosen to break a lossy writer: subnormals, the largest and
    // smallest normals, a value whose shortest decimal form is not its exact
    // one, and a negative zero.
    s3.env.wavelengths = vec![
        (0.1 + 0.2, 1.0 / 3.0),
        (f64::MIN_POSITIVE, 0.0),
        (1.7976931348623157e308, 5e-324),
        (0.5876, -0.0_f64.abs()),
    ];
    // One ULP above air's 1.000293, built from bits rather than written as a
    // literal so it cannot quietly round back to the round number. This is the
    // exact shape of the bug `float_roundtrip` exists for: the repo already hit
    // it once, when FusedSilica's computed Sellmeier terms came back one ULP off
    // and made the glass unplaceable.
    s3.env.ambient_index = f64::from_bits(1.000293_f64.to_bits() + 1);
    assert_ne!(
        s3.env.ambient_index, 1.000293,
        "the hostile value is not hostile"
    );
    s3
}

#[test]
fn v2_round_trips_every_f64_exactly() {
    // The format contract: a number that goes out comes back as the *same* f64,
    // not a near one. This is what `serde_json`'s `float_roundtrip` feature buys,
    // and it is not free, so it is asserted rather than assumed.
    let s = hostile_scene();
    let json = s.to_json();
    let back = Scene3::from_json(&json).unwrap();

    assert_eq!(
        back.env.ambient_index.to_bits(),
        s.env.ambient_index.to_bits()
    );
    for (a, b) in back.env.wavelengths.iter().zip(&s.env.wavelengths) {
        assert_eq!(a.0.to_bits(), b.0.to_bits(), "wavelength moved");
        assert_eq!(a.1.to_bits(), b.1.to_bits(), "weight moved");
    }
    // And the whole document is byte-stable, which is the stronger claim.
    assert_eq!(back.to_json(), json);
}

#[test]
fn a_third_round_trip_changes_nothing() {
    let s = hostile_scene();
    let a = s.to_json();
    let b = Scene3::from_json(&a).unwrap().to_json();
    let c = Scene3::from_json(&b).unwrap().to_json();
    assert_eq!(a, b);
    assert_eq!(b, c);
}

// ---------------------------------------------------------------------------
// The committed v2 fixture
// ---------------------------------------------------------------------------

/// The committed v2 schema fixture.
///
/// Generated by `cargo run -p oc-core --features scene3 --example
/// emit_v2_fixture`, never hand-edited.
const V2_FIXTURE: &str = include_str!("data/ocanvas-v2/mirror-migrated.ocanvas");

#[test]
fn the_committed_v2_fixture_still_matches_the_code() {
    // The anti-drift check. Without it the fixture is a snapshot of a schema that
    // used to exist, and it would keep passing the load test below while quietly
    // describing something the writer no longer emits.
    let v1 = Scene::from_json(V1_CORPUS[2].1).unwrap();
    let (s3, _) = migrate_v1_to_v2(&v1).unwrap();
    let (v2, losses) = s3.to_json_v2();
    assert!(losses.is_empty(), "{losses:?}");
    assert_eq!(
        v2.replace("\r\n", "\n"),
        V2_FIXTURE.replace("\r\n", "\n"),
        "the committed fixture is stale; regenerate it with \
         `cargo run -p oc-core --features scene3 --example emit_v2_fixture`"
    );
}

#[test]
fn the_committed_v2_fixture_loads_as_v2() {
    let (s3, report) = load_any(V2_FIXTURE).unwrap();
    assert_eq!(report, LoadReport::V2, "fixture was not read as v2");
    // Read as v2, held as v3 in memory.
    assert_eq!(s3.version, 3);
    assert_eq!(s3.dimension, 3);
    s3.validate().unwrap();
    // It is a real document, not an empty shell.
    assert_eq!(s3.bodies.len(), 1);
    assert!(!s3.bodies[0].is_3d(), "fixture body should be PlanarOnly");
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn a_document_from_the_future_is_refused_not_guessed_at() {
    let (s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    let json = s3.to_json().replace("\"version\": 3", "\"version\": 4");
    assert_eq!(
        load_any(&json).unwrap_err(),
        SceneError::UnsupportedVersion { found: 4 }
    );
}

#[test]
fn a_v2_document_claiming_to_be_2d_is_refused() {
    let (s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    let json = s3.to_json().replace("\"dimension\": 3", "\"dimension\": 2");
    assert_eq!(
        load_any(&json).unwrap_err(),
        SceneError::WrongDimension { found: 2 }
    );
}

#[test]
fn a_foreign_document_is_refused() {
    let json = r#"{"format":"zemax","version":1}"#;
    assert!(matches!(
        load_any(json).unwrap_err(),
        SceneError::WrongFormat { .. }
    ));
}

#[test]
fn a_duplicate_id_is_refused() {
    // The check stable ids exist to make possible: one flat sweep of the whole
    // document, because there is one id space.
    let (mut s3, _) = load_any(V1_CORPUS[0].1).unwrap();
    let victim = s3.bodies[1].common.id;
    s3.bodies[1].common.id = s3.bodies[0].common.id;
    assert_eq!(
        s3.validate().unwrap_err(),
        SceneError::DuplicateId {
            id: s3.bodies[0].common.id.0
        }
    );
    // Including across kinds, since bodies and patches share the space.
    s3.bodies[1].common.id = victim;
    let clash = s3.bodies[0].common.id.0;
    if let SolidGeometry3::Profile(p) = &mut s3.bodies[1].geometry {
        p.profile.edges[0].patch = PatchId(clash);
    }
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::DuplicateId { .. }
    ));
}

/// Rewrite one body's rotation quaternion in a serialized v2 document.
///
/// Through `serde_json` rather than string replacement: the pretty-printer's
/// whitespace is not part of the contract, and a test that greps for it passes
/// or fails for reasons that have nothing to do with the schema.
fn with_body_rotation(s3: &Scene3, rot: [f64; 4]) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&s3.to_json()).unwrap();
    v["bodies"][0]["common"]["transform"]["rot_xyzw"] = serde_json::json!(rot);
    serde_json::to_string(&v).unwrap()
}

#[test]
fn a_malformed_transform_is_refused_on_load() {
    let (s3, _) = load_any(V1_CORPUS[2].1).unwrap();

    // A zero quaternion names no rotation, and normalizing it divides by zero.
    // Refused, rather than loaded as a scene full of NaN.
    assert!(
        load_any(&with_body_rotation(&s3, [0.0, 0.0, 0.0, 0.0])).is_err(),
        "a zero quaternion loaded"
    );
    // Non-finite.
    let nan = with_body_rotation(&s3, [0.0, 0.0, 0.0, 1.0]).replace("1.0", "null");
    assert!(load_any(&nan).is_err(), "a null quaternion loaded");

    // An unnormalized one is *repaired*, not refused: it names a real rotation,
    // and the length carries no information a document could have meant. Left
    // alone it would scale the body by |q|².
    let doc = with_body_rotation(&s3, [0.0, 0.0, 0.0, 5.0]);
    let (loaded, _) = load_any(&doc).expect("a scalable quaternion should normalize");
    assert_eq!(
        loaded.bodies[0].common.transform.rot_xyzw,
        [0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn a_non_finite_value_is_refused() {
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    s3.env.ambient_index = f64::NAN;
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::NonFinite { .. }
    ));
    s3.env.ambient_index = 0.0;
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::OutOfRange { .. }
    ));
}

#[test]
fn a_dangling_reference_is_refused() {
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    s3.bodies[0].common.layer = LayerId(99);
    assert_eq!(
        s3.validate().unwrap_err(),
        SceneError::DanglingRef {
            what: "layer",
            id: 99
        }
    );
}

#[test]
fn a_patch_override_naming_no_patch_is_refused() {
    // The stale-reference class that v1's index ids made unavoidable. Here it is
    // a load error instead of an override that silently does nothing.
    use oc_core::scene3::{PatchResponseOverride, TwoSidedSurfaceResponse};
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    s3.bodies[0].surface.overrides.push(PatchResponseOverride {
        patch: PatchId(9999),
        response: TwoSidedSurfaceResponse::default(),
    });
    assert_eq!(
        s3.validate().unwrap_err(),
        SceneError::DanglingRef {
            what: "patch",
            id: 9999
        }
    );
}

#[test]
fn next_id_below_the_high_water_mark_is_refused_in_memory_and_repaired_on_load() {
    let (mut s3, _) = load_any(V1_CORPUS[0].1).unwrap();
    let high = s3.next_id - 1;
    s3.next_id = 0;
    // In memory it is a contradiction, so validate says so.
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::OutOfRange { .. }
    ));
    // On load it is bookkeeping a hand-edited file got wrong, so it is repaired.
    // The alternative is refusing a document whose *objects* are all fine.
    s3.repair_next_id();
    assert_eq!(s3.next_id, high + 1);
    assert!(s3.validate().is_ok());
}

#[test]
fn an_id_is_never_reused_after_the_object_is_gone() {
    // §3.8: removal does not renumber survivors, and the id does not come back.
    let (mut s3, _) = load_any(V1_CORPUS[0].1).unwrap();
    let survivor = s3.bodies[1].common.id;
    let removed = s3.bodies[0].common.id;
    s3.bodies.remove(0);
    assert_eq!(s3.bodies[0].common.id, survivor, "survivor was renumbered");
    let fresh = s3.alloc_object_id();
    assert_ne!(fresh, removed, "a removed id was handed out again");
    assert!(fresh.0 >= s3.next_id - 1);
    assert!(s3.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Spectral and response validation
// ---------------------------------------------------------------------------

#[test]
fn a_wavelength_table_must_be_a_function() {
    use oc_core::scene3::{Interpolation, OutOfRangePolicy, SpectralScalar};
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    let set = |s3: &mut Scene3, lambda: Vec<f64>, values: Vec<f64>| {
        s3.bodies[0].surface.default.from_outside = SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Tabulated {
                lambda_um: lambda,
                values,
                interpolation: Interpolation::Linear,
                out_of_range: OutOfRangePolicy::Error,
            },
        };
    };
    // Two samples at one wavelength make the table a relation, not a function:
    // which value you get would depend on how the search happened to land.
    set(&mut s3, vec![0.4, 0.4], vec![0.9, 0.1]);
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::InvalidTable { .. }
    ));
    // Decreasing.
    set(&mut s3, vec![0.6, 0.4], vec![0.9, 0.1]);
    assert!(s3.validate().is_err());
    // Mismatched lengths.
    set(&mut s3, vec![0.4, 0.5, 0.6], vec![0.9, 0.1]);
    assert!(s3.validate().is_err());
    // One sample cannot be interpolated.
    set(&mut s3, vec![0.4], vec![0.9]);
    assert!(s3.validate().is_err());
    // A reflectivity above 1 is a surface that returns more than it received.
    set(&mut s3, vec![0.4, 0.6], vec![0.9, 1.1]);
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::OutOfRange { .. }
    ));
    // And a good one passes, so the checks are not just refusing everything.
    set(&mut s3, vec![0.4, 0.6], vec![0.9, 0.95]);
    assert!(s3.validate().is_ok());
}

#[test]
fn r_plus_t_may_not_exceed_one() {
    use oc_core::scene3::SpectralScalar;
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    s3.bodies[0].surface.default.from_outside = SurfaceResponse::ExplicitRt {
        reflectance: SpectralScalar::Constant(0.6),
        transmittance: SpectralScalar::Constant(0.6),
    };
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::OutOfRange { .. }
    ));
    // 0.6 + 0.4 is exactly 1: the remainder absorbed is zero, which is legal.
    s3.bodies[0].surface.default.from_outside = SurfaceResponse::ExplicitRt {
        reflectance: SpectralScalar::Constant(0.6),
        transmittance: SpectralScalar::Constant(0.4),
    };
    assert!(s3.validate().is_ok());
}

#[test]
fn complementary_dichroic_curves_are_physical() {
    use oc_core::scene3::{Interpolation, OutOfRangePolicy, SpectralScalar};
    let step = |values: Vec<f64>| SpectralScalar::Tabulated {
        lambda_um: vec![0.505, 0.507],
        values,
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    };
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    // A dichroic: R and T are complementary step curves. Each peaks at 1 (at
    // DIFFERENT wavelengths), so a per-curve-maxima bound would reject this
    // physical surface; the pointwise union-grid check accepts it.
    s3.bodies[0].surface.default.from_outside = SurfaceResponse::ExplicitRt {
        reflectance: step(vec![1.0, 0.0]),
        transmittance: step(vec![0.0, 1.0]),
    };
    assert!(s3.validate().is_ok());
    // A pair that genuinely sums past 1 somewhere is still refused.
    s3.bodies[0].surface.default.from_outside = SurfaceResponse::ExplicitRt {
        reflectance: step(vec![1.0, 0.4]),
        transmittance: step(vec![0.0, 0.7]),
    };
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::OutOfRange { .. }
    ));
    // Mixed constant + table: bounded pointwise on the table's grid.
    s3.bodies[0].surface.default.from_outside = SurfaceResponse::ExplicitRt {
        reflectance: SpectralScalar::Constant(0.5),
        transmittance: step(vec![0.5, 0.2]),
    };
    assert!(s3.validate().is_ok());
}

#[test]
fn negative_absorption_is_refused() {
    use oc_core::scene3::AbsorptionModel;
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    // Negative absorption is gain. Nothing here models a laser.
    s3.bodies[0].absorption = AbsorptionModel::Constant { alpha_per_mm: -0.1 };
    assert!(matches!(
        s3.validate().unwrap_err(),
        SceneError::OutOfRange { .. }
    ));
}

// ---------------------------------------------------------------------------
// The v1 surface-spec migration table (§6.7)
// ---------------------------------------------------------------------------

#[test]
fn v1_surface_specs_migrate_by_the_table() {
    use oc_core::scene3::SpectralScalar;
    use oc_core::{Element, Material, SurfaceProps, SurfaceSpec};

    let build = |spec: SurfaceSpec| {
        let mut v1 = Scene::new();
        let mut e = Element::new(
            oc_core::ElementId(0),
            vec![oc_core::Edge::Line {
                a: oc_core::Vec2::ZERO,
                b: oc_core::Vec2::X,
            }],
            Material::AbbePoint {
                nd: 1.5168,
                vd: 64.17,
            },
        );
        e.surface = SurfaceProps {
            default: spec,
            overrides: vec![],
        };
        v1.elements.push(e);
        migrate_v1_to_v2(&v1).unwrap().0.bodies[0]
            .surface
            .default
            .from_outside
            .clone()
    };

    // mirror = true, reflectivity = x  ->  Mirror(Constant(x))
    let r = build(SurfaceSpec {
        mirror: true,
        reflectivity: 0.83,
        ..Default::default()
    });
    assert!(matches!(
        r,
        SurfaceResponse::Mirror { reflectivity: SpectralScalar::Constant(x) } if x == 0.83
    ));

    // absorb = true  ->  Absorber, and it wins over mirror.
    let r = build(SurfaceSpec {
        absorb: true,
        mirror: true,
        ..Default::default()
    });
    assert_eq!(r, SurfaceResponse::Absorber);

    // default dielectric, ar_residual = x  ->  DielectricFresnel(Constant(x))
    let r = build(SurfaceSpec {
        ar_residual: 5e-3,
        ..Default::default()
    });
    assert!(matches!(
        r,
        SurfaceResponse::DielectricFresnel { ar_residual: SpectralScalar::Constant(x) } if x == 5e-3
    ));
}

#[test]
fn v1_roughness_is_dropped_and_the_report_says_so() {
    // v1 reserved `roughness` for scatter and never read it. Carrying it into v2
    // would let the schema imply a scatter model that does not exist; dropping it
    // silently would lose an authored number. So it is dropped *and* reported.
    use oc_core::{Element, Material, SurfaceProps, SurfaceSpec};
    let mut v1 = Scene::new();
    let mut e = Element::new(
        oc_core::ElementId(0),
        vec![oc_core::Edge::Line {
            a: oc_core::Vec2::ZERO,
            b: oc_core::Vec2::X,
        }],
        Material::AbbePoint {
            nd: 1.5168,
            vd: 64.17,
        },
    );
    e.surface = SurfaceProps {
        default: SurfaceSpec {
            roughness: 0.25,
            ..Default::default()
        },
        overrides: vec![],
    };
    v1.elements.push(e);
    let (_, report) = migrate_v1_to_v2(&v1).unwrap();
    assert!(
        report.dropped.iter().any(|d| d.contains("roughness")),
        "roughness vanished without a word: {:?}",
        report.dropped
    );
}

#[test]
fn v1_spline_becomes_a_polyline_not_a_fake_bspline() {
    // v1's `Spline` is documented as "evaluated as a polyline through the control
    // points", so the rename changes no ray and stops the schema promising a
    // curve the tracer never computed (§6.3).
    use oc_core::{Element, Material, Vec2};
    let pts = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(2.0, 0.0),
    ];
    let mut v1 = Scene::new();
    v1.elements.push(Element::new(
        oc_core::ElementId(0),
        vec![oc_core::Edge::Spline {
            ctrl: pts.clone(),
            degree: 3,
        }],
        Material::AbbePoint {
            nd: 1.5168,
            vd: 64.17,
        },
    ));
    let (s3, _) = migrate_v1_to_v2(&v1).unwrap();
    let SolidGeometry3::Profile(p) = &s3.bodies[0].geometry else {
        panic!("not a profile")
    };
    match &p.profile.edges[0].curve {
        Curve2::Polyline { points } => assert_eq!(points, &pts, "control points moved"),
        other => panic!("spline became {other:?}"),
    }
}

#[test]
fn v1_geometry_values_survive_the_migration_bit_for_bit() {
    use oc_core::{ConicParams, Element, Material, Vec2};
    let base = ConicParams {
        vertex: Vec2::new(1.0 / 3.0, -0.1 - 0.2),
        axis: Vec2::new(1.0, 0.0),
        c: 1.0 / 51.68,
        k: -0.5,
        half_ap: 12.7,
    };
    let coeffs = [1e-7, -2.5e-11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut v1 = Scene::new();
    v1.elements.push(Element::new(
        oc_core::ElementId(0),
        vec![oc_core::Edge::Asphere { base, coeffs }],
        Material::AbbePoint {
            nd: 1.5168,
            vd: 64.17,
        },
    ));
    let (s3, _) = migrate_v1_to_v2(&v1).unwrap();
    let SolidGeometry3::Profile(p) = &s3.bodies[0].geometry else {
        panic!("not a profile")
    };
    let Curve2::EvenAsphere {
        base: b2,
        coeffs: c2,
    } = &p.profile.edges[0].curve
    else {
        panic!("asphere became something else")
    };
    assert_eq!(b2.c.to_bits(), base.c.to_bits());
    assert_eq!(b2.k.to_bits(), base.k.to_bits());
    assert_eq!(b2.vertex.x.to_bits(), base.vertex.x.to_bits());
    assert_eq!(b2.vertex.y.to_bits(), base.vertex.y.to_bits());
    assert_eq!(b2.half_ap.to_bits(), base.half_ap.to_bits());
    assert_eq!(c2, &coeffs);
}

#[test]
fn a_v1_transform_becomes_the_same_pose_in_3d() {
    use oc_core::{Element, Isometry2, Material, Vec2};
    let mut v1 = Scene::new();
    let mut e = Element::new(
        oc_core::ElementId(0),
        vec![oc_core::Edge::Line {
            a: Vec2::ZERO,
            b: Vec2::X,
        }],
        Material::AbbePoint {
            nd: 1.5168,
            vd: 64.17,
        },
    );
    e.transform = Isometry2::new(Vec2::new(3.0, -7.25), 0.9);
    v1.elements.push(e);
    let (s3, _) = migrate_v1_to_v2(&v1).unwrap();
    let t = s3.bodies[0].common.transform;

    // Position gains a zero z, not a guess.
    assert_eq!(t.pos, oc_core::Vec3::new(3.0, -7.25, 0.0));
    // A v1 in-plane rotation is a rotation about the axis normal to that plane,
    // so an in-plane point lands exactly where the 2D transform put it.
    for p in [Vec2::new(1.0, 0.0), Vec2::new(-2.5, 4.0), Vec2::ZERO] {
        let want = e_transform(&v1, p);
        let got = t.transform_point(oc_core::Vec3::new(p.x, p.y, 0.0));
        assert!(
            (got.x - want.x).abs() < 1e-12 && (got.y - want.y).abs() < 1e-12,
            "{p:?}"
        );
        assert_eq!(got.z, 0.0, "an in-plane point left the plane");
    }
}

fn e_transform(v1: &Scene, p: oc_core::Vec2) -> oc_core::Vec2 {
    v1.elements[0].transform.transform_point(p)
}

#[test]
fn an_identity_v1_transform_is_exactly_the_identity_in_3d() {
    // Not approximately. A fragment placed unrotated at the origin may not have
    // its geometry perturbed, which is the guarantee `Isometry2` already makes.
    let (s3, _) = load_any(V1_CORPUS[0].1).unwrap();
    for b in &s3.bodies {
        if b.common.transform.pos == oc_core::Vec3::ZERO {
            assert_eq!(b.common.transform.rot_xyzw, [0.0, 0.0, 0.0, 1.0]);
        }
    }
}

// ---------------------------------------------------------------------------
// Promotion
// ---------------------------------------------------------------------------

#[test]
fn promotion_is_the_only_way_a_body_becomes_3d() {
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    let id = s3.bodies[0].id();
    assert!(!s3.body(id).unwrap().is_3d());

    let report = promote_body(&mut s3, id, ProfileConstruction3::RevolveLocalX).unwrap();
    assert_eq!(
        report.from,
        ProfileConstruction3::PlanarOnly { plane_z_mm: 0.0 }
    );
    assert_eq!(report.to, ProfileConstruction3::RevolveLocalX);
    assert!(s3.body(id).unwrap().is_3d());
    assert!(s3.validate().is_ok());

    // The geometry engine now runs the real checks, so nothing is left undecided:
    // the mirror is a closed, y-symmetric rectangle and revolves to a cylinder.
    assert!(
        report.undecided.is_empty(),
        "the revolve validator ran, so no check should be undecided: {:?}",
        report.undecided
    );
}

#[test]
fn promotion_refuses_what_it_can_already_disprove() {
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    let id = s3.bodies[0].id();
    // z_min >= z_max bounds no volume, and that needs no geometry engine to see.
    let e = promote_body(
        &mut s3,
        id,
        ProfileConstruction3::ExtrudeLocalZ {
            z_min_mm: 5.0,
            z_max_mm: 5.0,
        },
    )
    .unwrap_err();
    assert!(format!("{e}").contains("bounds no volume"), "{e}");
    // And the body is untouched, so a refused promotion costs nothing.
    assert!(!s3.body(id).unwrap().is_3d());

    assert!(matches!(
        promote_body(&mut s3, BodyId(9999), ProfileConstruction3::RevolveLocalX),
        Err(oc_core::scene3::PromotionError::UnknownBody(9999))
    ));
}

#[test]
fn the_promotion_checks_now_decide() {
    // The geometry engine has landed, so a check that once reported `Undecided`
    // returns a real verdict. The mirror is a closed rectangle, symmetric about
    // y = 0, so it both revolves (to a cylinder) and extrudes.
    let (s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    let b = &s3.bodies[0];
    assert_eq!(can_promote_to_revolved(b), PromotionCheck::Ok);
    assert_eq!(can_promote_to_extruded(b, 0.0, 10.0), PromotionCheck::Ok);
    // An inverted extrusion is still refused before the geometry runs.
    assert!(can_promote_to_extruded(b, 10.0, 0.0).is_refusal());
}

#[test]
fn an_analytic_solid_needs_no_promotion() {
    let (mut s3, _) = load_any(V1_CORPUS[2].1).unwrap();
    let id = s3.alloc_object_id();
    s3.bodies.push(Body3 {
        common: ObjectCommon::new(id, LayerId(0)),
        geometry: SolidGeometry3::Sphere { radius_mm: 5.0 },
        material: oc_core::Material::AbbePoint {
            nd: 1.5168,
            vd: 64.17,
        },
        absorption: Default::default(),
        surface: Default::default(),
        priority: 0,
    });
    let b = s3.body(BodyId(id.0)).unwrap();
    assert!(b.is_3d(), "a sphere is 3D by construction");
    assert!(b.construction().is_none());
    assert!(can_promote_to_revolved(b).is_refusal());
    assert!(matches!(
        promote_body(&mut s3, BodyId(id.0), ProfileConstruction3::RevolveLocalX),
        Err(oc_core::scene3::PromotionError::NotProfileAuthored(_))
    ));
}

// ---------------------------------------------------------------------------
// The typed id views
// ---------------------------------------------------------------------------

#[test]
fn a_body_id_and_its_object_id_are_the_same_number() {
    // One id space; the newtypes are lenses over it, not parallel spaces to keep
    // in step.
    let (s3, _) = load_any(V1_CORPUS[0].1).unwrap();
    for b in &s3.bodies {
        assert_eq!(b.id().0, b.common.id.0);
        assert!(s3.body(b.id()).is_some());
    }
    let (s3, _) = load_any(V1_CORPUS[1].1).unwrap();
    for d in &s3.detectors {
        assert_eq!(d.id().0, d.common.id.0);
        assert!(s3.detector(d.id()).is_some());
    }
}

#[test]
fn ids_do_not_collide_across_kinds() {
    // A body, its patches, a source and a detector all draw from one counter, so
    // no two objects in a document ever share a number.
    let mut v1 = Scene::new();
    v1.elements.push(oc_core::Element::new(
        oc_core::ElementId(0),
        vec![oc_core::Edge::Line {
            a: oc_core::Vec2::ZERO,
            b: oc_core::Vec2::X,
        }],
        oc_core::Material::AbbePoint {
            nd: 1.5168,
            vd: 64.17,
        },
    ));
    v1.sources.push(oc_core::Source {
        kind: oc_core::SourceKind::Point { half_angle: 0.1 },
        transform: oc_core::Isometry2::IDENTITY,
        spectrum: oc_core::Spectrum::Monochromatic(0.5876),
        flux: 1.0,
        rays: 3,
    });
    v1.detectors.push(oc_core::Detector::new(
        oc_core::Vec2::new(10.0, -5.0),
        oc_core::Vec2::new(10.0, 5.0),
        7,
    ));
    let (s3, _) = migrate_v1_to_v2(&v1).unwrap();
    let mut ids = vec![
        s3.bodies[0].common.id.0,
        s3.sources[0].id.0,
        s3.detectors[0].common.id.0,
    ];
    if let SolidGeometry3::Profile(p) = &s3.bodies[0].geometry {
        ids.extend(p.profile.edges.iter().map(|e| e.patch.0));
    }
    let n = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), n, "ids collided across kinds");
    assert!(s3.validate().is_ok());
    assert_eq!(s3.source(SourceId(s3.sources[0].id.0)).unwrap().flux, 1.0);
}

// ---------------------------------------------------------------------------
// Spectral and absorption sampling (the tracer's consumers, §6.7)
// ---------------------------------------------------------------------------

#[test]
fn spectral_and_absorption_tables_interpolate_and_honour_the_range_policy() {
    use oc_core::scene3::{AbsorptionModel, Interpolation, OutOfRangePolicy, SpectralScalar};

    // A linear ramp 0.4 -> 0.8 over 0.5..0.7 um.
    let ramp = |out_of_range| SpectralScalar::Tabulated {
        lambda_um: vec![0.5, 0.7],
        values: vec![0.4, 0.8],
        interpolation: Interpolation::Linear,
        out_of_range,
    };

    // Interpolates at the midpoint.
    assert!((ramp(OutOfRangePolicy::Error).sample(0.6).unwrap() - 0.6).abs() < 1e-12);
    // Error policy refuses out of range rather than inventing a value.
    assert_eq!(ramp(OutOfRangePolicy::Error).sample(0.4), None);
    assert_eq!(ramp(OutOfRangePolicy::Error).sample(0.8), None);
    // Clamp holds the endpoint; Zero returns nothing.
    assert_eq!(ramp(OutOfRangePolicy::Clamp).sample(0.4), Some(0.4));
    assert_eq!(ramp(OutOfRangePolicy::Clamp).sample(0.9), Some(0.8));
    assert_eq!(ramp(OutOfRangePolicy::Zero).sample(0.4), Some(0.0));

    // Constants are wavelength-flat; absorption-free models return zero.
    assert_eq!(SpectralScalar::Constant(0.5).sample(1.0), Some(0.5));
    assert_eq!(AbsorptionModel::None.sample(0.5), Some(0.0));
    assert_eq!(
        AbsorptionModel::Constant { alpha_per_mm: 0.1 }.sample(0.5),
        Some(0.1)
    );
}

// ---------------------------------------------------------------------------
// transform_objects (CV-C): batch pose edits on the loaded scene
// ---------------------------------------------------------------------------

#[test]
fn transform_objects_moves_a_batch_together() {
    use oc_core::build3::SingletGeom;
    use oc_core::vec3::{Isometry3, Vec3};

    let mut scene = Scene3::new();
    let body = scene.add_singlet(Vec3::new(50.0, 0.0, 0.0), SingletGeom::biconvex(), 1.5168, 64.17);
    let src = scene.add_collimated_beam(
        Vec3::ZERO,
        1.0,
        0.2,
        4,
        oc_core::scene::Spectrum::Monochromatic(0.55),
    );
    let before_body = scene.bodies[0].common.transform;
    let before_src = scene.sources[0].transform;

    let delta = Isometry3::from_pos(Vec3::new(0.0, 7.0, -3.0));
    scene
        .transform_objects(&[body.0, src.raw()], delta)
        .expect("both ids resolve");

    // Each object composed the same delta, so their relative pose is intact.
    assert_eq!(scene.bodies[0].common.transform, delta.compose(before_body));
    assert_eq!(scene.sources[0].transform, delta.compose(before_src));
    let rel = |a: Isometry3, b: Isometry3| a.pos - b.pos;
    assert_eq!(
        rel(scene.bodies[0].common.transform, scene.sources[0].transform),
        rel(before_body, before_src)
    );
}

#[test]
fn transform_objects_unknown_id_is_atomic() {
    use oc_core::build3::SingletGeom;
    use oc_core::vec3::{Isometry3, Vec3};

    let mut scene = Scene3::new();
    let body = scene.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
    let before = scene.bodies[0].common.transform;

    let delta = Isometry3::from_pos(Vec3::new(5.0, 0.0, 0.0));
    let err = scene
        .transform_objects(&[body.0, 9999], delta)
        .expect_err("unknown id must refuse");
    assert!(matches!(err, SceneError::DanglingRef { id: 9999, .. }));
    // Nothing moved, including the id that did resolve.
    assert_eq!(scene.bodies[0].common.transform, before);
}

#[test]
fn transform_objects_duplicate_id_applies_once() {
    use oc_core::build3::SingletGeom;
    use oc_core::vec3::{Isometry3, Vec3};

    let mut scene = Scene3::new();
    let body = scene.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
    let delta = Isometry3::from_pos(Vec3::new(5.0, 0.0, 0.0));
    scene
        .transform_objects(&[body.0, body.0, body.0], delta)
        .expect("duplicates are a set, not a multiplier");
    assert_eq!(scene.bodies[0].common.transform.pos, Vec3::new(5.0, 0.0, 0.0));
}

#[test]
fn thin_lens_validates_focal_length() {
    use oc_core::scene3::{PlanarShape2, SurfaceGeometry3, SurfaceObject3, SurfaceResponse};

    let mut scene = Scene3::new();
    let id = scene.alloc_object_id();
    let resp = SurfaceResponse::ThinLens {
        focal_length_mm: 0.0,
    };
    scene.surfaces.push(SurfaceObject3 {
        common: ObjectCommon::new(id, oc_core::id::LayerId(0)),
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Disc { radius_mm: 10.0 },
        },
        response_front: resp.clone(),
        response_back: resp,
    });
    let err = scene.validate().expect_err("f = 0 is not a lens");
    assert!(matches!(err, SceneError::OutOfRange { .. }), "{err:?}");

    // A finite nonzero focal length (negative = diverging) validates.
    let lens = SurfaceResponse::ThinLens {
        focal_length_mm: -75.0,
    };
    scene.surfaces[0].response_front = lens.clone();
    scene.surfaces[0].response_back = lens;
    scene.validate().expect("diverging thin lens is valid");
}

#[test]
fn ideal_mirror_validates_powers_and_keeps_its_scene_format() {
    use oc_core::scene3::{PlanarShape2, SurfaceGeometry3, SurfaceObject3, SurfaceResponse};

    let mut scene = Scene3::new();
    let id = scene.alloc_object_id();
    let mirror = |px: f64, py: f64| SurfaceResponse::IdealMirror {
        power_x_per_mm: px,
        power_y_per_mm: py,
    };
    scene.surfaces.push(SurfaceObject3 {
        common: ObjectCommon::new(id, oc_core::id::LayerId(0)),
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Disc { radius_mm: 10.0 },
        },
        response_front: mirror(0.01, 0.01),
        response_back: SurfaceResponse::Absorber,
    });
    scene.validate().expect("a focusing ideal mirror is valid");
    // Flat, diverging, and different along the two axes are all mirrors.
    for (px, py) in [(0.0, 0.0), (-0.02, -0.02), (0.01, 0.025), (0.0, 0.05)] {
        scene.surfaces[0].response_front = mirror(px, py);
        scene.validate().unwrap_or_else(|e| panic!("({px}, {py}): {e:?}"));
    }
    for (px, py) in [(f64::NAN, 0.0), (0.0, f64::INFINITY)] {
        scene.surfaces[0].response_front = mirror(px, py);
        let err = scene.validate().expect_err("a power must be finite");
        assert!(matches!(err, SceneError::OutOfRange { .. }), "{err:?}");
    }

    // What a scene file carries, written out: the shape other tools write.
    let r = mirror(0.01, 0.025);
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(
        json,
        serde_json::json!({ "IdealMirror": { "power_x_per_mm": 0.01, "power_y_per_mm": 0.025 } })
    );
    assert_eq!(serde_json::from_value::<SurfaceResponse>(json).unwrap(), r);
    assert_eq!(oc_core::build3::response_kind(&r), "ideal-mirror");
}
