//! The sequential export of a 3D scene (`oc_trace::seq_export`).
//!
//! What this suite owes:
//!
//! - a coaxial singlet exports as its two faces, with the gaps, apertures and
//!   medium the scene gives them;
//! - the order is the order the light meets the faces, so a lens turned round
//!   exports its faces the other way, curvatures flipped;
//! - every aperture the path crosses is a surface where the path crosses it,
//!   and the one the scene declares as the stop, from either end, is the stop;
//! - stop declarations that disagree, or name an aperture off the path, are
//!   refused, and a scene that declares none says so with `stop: None`;
//! - everything a sequential system cannot hold is refused with the object it
//!   concerns, every reason at once;
//! - a point source's cone is a finite conjugate with an object-space NA.
#![cfg(feature = "trace3")]

use oc_core::build3::SingletGeom;
use oc_core::id::{ApertureId, DetectorId, SourceId};
use oc_core::scene::Spectrum;
use oc_core::scene3::{
    AngularDistribution3, PlanarShape2, PrimaryPathIntent, Scene3, SpatialDistribution3,
    SpectralScalar, SurfaceResponse,
};
use oc_core::seq::{FieldType, SeqAperture, SeqMaterial, SeqSystem};
use oc_core::vec3::Vec3;
use oc_trace::{default_path, export_seq, ExportError};

const EPS: f64 = 1e-9;

/// A disc beam of radius 5 mm from x = -70 along +X, a biconvex singlet at the
/// origin (R = ±50, 4 mm, semi-aperture 12), and a detector square to the axis
/// at `x_det`.
fn singlet_bench(x_det: f64) -> (Scene3, SourceId, DetectorId) {
    let mut s = Scene3::new();
    s.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
    let src = s.add_collimated_beam(
        Vec3::new(-70.0, 0.0, 0.0),
        5.0,
        5.0,
        9,
        Spectrum::Monochromatic(0.58756),
    );
    s.sources[0].spatial = SpatialDistribution3::Disc { radius_mm: 5.0 };
    let det = s.add_area_detector(Vec3::new(x_det, 0.0, 0.0), -Vec3::X, 12.0, 12.0, [8, 8]);
    s.validate().expect("valid");
    (s, src, det)
}

fn export(s: &Scene3, src: SourceId, det: DetectorId) -> SeqSystem {
    export_seq(s, src, det).unwrap_or_else(|e| panic!("{e}"))
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS * b.abs().max(1.0)
}

#[test]
fn a_singlet_exports_as_its_two_faces() {
    let (s, src, det) = singlet_bench(48.0);
    let seq = export(&s, src, det);
    assert_eq!(seq.surfaces.len(), 2);
    let (front, back) = (&seq.surfaces[0], &seq.surfaces[1]);
    assert!(close(front.radius.unwrap(), 50.0), "{:?}", front.radius);
    assert!(close(back.radius.unwrap(), -50.0), "{:?}", back.radius);
    assert!(close(front.thickness, 4.0));
    // Back vertex at x = 2, detector at x = 48.
    assert!(close(back.thickness, 46.0), "{}", back.thickness);
    assert_eq!(front.semi_aperture, 12.0);
    assert_eq!(
        front.material,
        SeqMaterial::Abbe {
            nd: 1.5168,
            vd: 64.17
        }
    );
    assert_eq!(back.material, SeqMaterial::Air);
    assert_eq!(seq.aperture, SeqAperture::Epd { value: 10.0 });
    assert_eq!(seq.fields, vec![0.0]);
    assert_eq!(seq.field_type, FieldType::Angle);
    assert_eq!(seq.object_distance, None);
    assert_eq!(
        seq.stop, None,
        "nothing declares a stop, and nothing picks one"
    );
    assert!(seq.notes.is_empty(), "{:?}", seq.notes);
}

/// A plano-convex lens, flat side to the light, then turned round. The trace
/// meets the curved face first the second time, and the exported curvature is
/// the one that face has seen from the light's side.
#[test]
fn a_lens_turned_round_exports_its_faces_in_the_order_the_light_meets_them() {
    let pc = SingletGeom {
        r1_mm: f64::INFINITY,
        r2_mm: -30.0,
        thickness_mm: 5.0,
        semi_aperture_mm: 10.0,
        conic1: 0.0,
        conic2: 0.0,
    };
    let bench = |turned: bool| {
        let mut s = Scene3::new();
        let lens = s.add_singlet(Vec3::ZERO, pc, 1.5168, 64.17);
        if turned {
            assert!(s.set_object_rotation_z_deg(lens.0, 180.0));
        }
        let src = s.add_collimated_beam(
            Vec3::new(-50.0, 0.0, 0.0),
            4.0,
            4.0,
            9,
            Spectrum::Monochromatic(0.58756),
        );
        s.sources[0].spatial = SpatialDistribution3::Disc { radius_mm: 4.0 };
        let det = s.add_area_detector(Vec3::new(80.0, 0.0, 0.0), -Vec3::X, 10.0, 10.0, [4, 4]);
        s.validate().unwrap();
        export(&s, src, det)
    };
    let flat_first = bench(false);
    assert_eq!(
        flat_first.surfaces[0].radius, None,
        "the flat face meets the light"
    );
    assert!(close(flat_first.surfaces[1].radius.unwrap(), -30.0));

    let curved_first = bench(true);
    // Convex toward the light: the centre of curvature lies ahead of it.
    assert!(close(curved_first.surfaces[0].radius.unwrap(), 30.0));
    assert_eq!(curved_first.surfaces[1].radius, None);
    for seq in [&flat_first, &curved_first] {
        assert!(close(seq.surfaces[0].thickness, 5.0));
        assert!(close(
            seq.surfaces[0].thickness + seq.surfaces[1].thickness,
            82.5
        ));
    }
}

/// Two singlets, a clear aperture in front and a stop between them.
fn interior_stop_bench() -> (Scene3, SourceId, DetectorId, u64, u64) {
    let mut s = Scene3::new();
    s.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
    s.add_singlet(
        Vec3::new(40.0, 0.0, 0.0),
        SingletGeom::biconvex(),
        1.5168,
        64.17,
    );
    let front = s.add_aperture(Vec3::new(-20.0, 0.0, 0.0), 8.0, 0.0);
    let stop = s.add_aperture(Vec3::new(20.0, 0.0, 0.0), 4.0, 0.0);
    let src = s.add_collimated_beam(
        Vec3::new(-70.0, 0.0, 0.0),
        3.0,
        3.0,
        9,
        Spectrum::Monochromatic(0.58756),
    );
    s.sources[0].spatial = SpatialDistribution3::Disc { radius_mm: 3.0 };
    let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::X, 12.0, 12.0, [8, 8]);
    s.validate().unwrap();
    (s, src, det, front, stop)
}

fn declare_path(s: &mut Scene3, src: SourceId, det: DetectorId, stop: Option<u64>) -> u64 {
    let id = s.alloc_path_intent_id();
    s.intent.primary_paths.push(PrimaryPathIntent {
        id,
        source: src,
        detector: det,
        stop: stop.map(ApertureId),
        label: "imaging".into(),
    });
    id.0
}

#[test]
fn an_interior_stop_is_the_stop_and_apertures_sit_where_the_path_crosses_them() {
    let (mut s, src, det, _front, stop) = interior_stop_bench();
    declare_path(&mut s, src, det, Some(stop));
    s.validate().unwrap();
    let seq = export(&s, src, det);

    // Front aperture, lens one, the stop, lens two.
    let semis: Vec<f64> = seq.surfaces.iter().map(|x| x.semi_aperture).collect();
    assert_eq!(semis, vec![8.0, 12.0, 12.0, 4.0, 12.0, 12.0]);
    assert_eq!(
        seq.stop,
        Some(3),
        "the declared aperture, between the lenses"
    );
    let gaps: Vec<f64> = seq.surfaces.iter().map(|x| x.thickness).collect();
    for (got, want) in gaps.iter().zip([18.0, 4.0, 18.0, 18.0, 4.0, 58.0]) {
        assert!(close(*got, want), "{gaps:?}");
    }
    // An aperture is flat and sits in the medium it is in.
    assert_eq!(seq.surfaces[3].radius, None);
    assert_eq!(seq.surfaces[3].material, SeqMaterial::Air);
    assert!(seq.notes.is_empty(), "{:?}", seq.notes);
}

#[test]
fn a_stop_named_by_the_aperture_is_the_same_stop() {
    let (mut s, src, det, _front, stop) = interior_stop_bench();
    let intent = declare_path(&mut s, src, det, None);
    s.apertures
        .iter_mut()
        .find(|a| a.common.id.0 == stop)
        .unwrap()
        .primary_stop_for
        .push(oc_core::id::PathIntentId(intent));
    s.validate().unwrap();
    assert_eq!(export(&s, src, det).stop, Some(3));
    // Named from both ends, consistently, is still one stop.
    s.intent.primary_paths[0].stop = Some(ApertureId(stop));
    assert_eq!(export(&s, src, det).stop, Some(3));
}

#[test]
fn stop_declarations_that_disagree_or_miss_the_path_are_refused() {
    let (mut s, src, det, front, stop) = interior_stop_bench();
    let intent = declare_path(&mut s, src, det, Some(stop));
    s.apertures
        .iter_mut()
        .find(|a| a.common.id.0 == front)
        .unwrap()
        .primary_stop_for
        .push(oc_core::id::PathIntentId(intent));
    s.validate().unwrap();
    match export_seq(&s, src, det) {
        Err(ExportError::Stop(why)) => {
            assert!(
                why.contains(&format!("#{front}")) && why.contains(&format!("#{stop}")),
                "{why}"
            )
        }
        other => panic!("{other:?}"),
    }

    // A stop off the path: an aperture well above the axis.
    let (mut s, src, det, _, _) = interior_stop_bench();
    let aside = s.add_aperture(Vec3::new(20.0, 60.0, 0.0), 4.0, 0.0);
    declare_path(&mut s, src, det, Some(aside));
    s.validate().unwrap();
    match export_seq(&s, src, det) {
        Err(ExportError::Stop(why)) => assert!(why.contains("not on the path"), "{why}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_decentred_or_tilted_lens_is_refused_by_name() {
    for (what, edit) in [("decentred", 0usize), ("tilted", 1usize)] {
        let (mut s, src, det) = singlet_bench(48.0);
        let lens = s.bodies[0].common.id.0;
        match edit {
            0 => assert!(s.set_object_position(lens, Vec3::new(0.0, 0.3, 0.0))),
            _ => assert!(s.set_object_rotation_z_deg(lens, 2.0)),
        }
        match export_seq(&s, src, det) {
            Err(ExportError::Unsupported(reasons)) => {
                assert!(
                    reasons
                        .iter()
                        .any(|r| r.contains(what) && r.contains(&format!("#{lens}"))),
                    "{reasons:?}"
                );
            }
            other => panic!("{what}: {other:?}"),
        }
    }
}

#[test]
fn a_fold_and_a_square_opening_are_both_named() {
    // A 45 degree mirror folding the beam up to a detector above it, and a
    // square aperture on the way.
    let mut s = Scene3::new();
    let src = s.add_collimated_beam(
        Vec3::new(-70.0, 0.0, 0.0),
        3.0,
        3.0,
        9,
        Spectrum::Monochromatic(0.58756),
    );
    s.sources[0].spatial = SpatialDistribution3::Disc { radius_mm: 3.0 };
    let square = s.add_aperture(Vec3::new(-40.0, 0.0, 0.0), 6.0, 0.0);
    s.apertures[0].shape.clear_opening = PlanarShape2::Rectangle {
        half_width_mm: 6.0,
        half_height_mm: 6.0,
    };
    let mirror_response = SurfaceResponse::Mirror {
        reflectivity: SpectralScalar::Constant(1.0),
    };
    let mirror = s.add_planar_element(
        Vec3::ZERO,
        10.0,
        135.0,
        mirror_response.clone(),
        mirror_response,
    );
    let det = s.add_area_detector(Vec3::new(0.0, 40.0, 0.0), -Vec3::Y, 12.0, 12.0, [8, 8]);
    s.validate().unwrap();
    match export_seq(&s, src, det) {
        Err(ExportError::Unsupported(reasons)) => {
            assert!(
                reasons
                    .iter()
                    .any(|r| r.contains(&format!("#{mirror}")) && r.contains("mirror")),
                "{reasons:?}"
            );
            // The square aperture is before the fold, on the stretch the
            // export can still measure, and it is named in the same refusal.
            assert!(
                reasons
                    .iter()
                    .any(|r| r.contains(&format!("#{square}")) && r.contains("not circular")),
                "{reasons:?}"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_point_source_is_a_finite_conjugate() {
    let (mut s, src, det) = singlet_bench(48.0);
    s.sources[0].transform = oc_core::vec3::Isometry3::from_pos(Vec3::new(-100.0, 0.0, 0.0));
    s.sources[0].spatial = SpatialDistribution3::Point;
    s.sources[0].angular = AngularDistribution3::UniformConeSolidAngle {
        half_angle_rad: 0.05,
    };
    s.validate().unwrap();
    let seq = export(&s, src, det);
    assert_eq!(seq.field_type, FieldType::ObjectHeight);
    // From the source at x = -100 to the front vertex at x = -2.
    assert!(
        close(seq.object_distance.unwrap(), 98.0),
        "{:?}",
        seq.object_distance
    );
    match seq.aperture {
        SeqAperture::ObjectNa { value } => assert!(close(value, 0.05f64.sin())),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_rectangular_beam_is_noted_not_hidden() {
    let (mut s, src, det) = singlet_bench(48.0);
    s.sources[0].spatial = SpatialDistribution3::Rectangle {
        half_width_mm: 6.0,
        half_height_mm: 2.0,
    };
    let seq = export(&s, src, det);
    assert_eq!(seq.aperture, SeqAperture::Epd { value: 12.0 });
    assert!(
        seq.notes.iter().any(|n| n.contains("rectangle")),
        "{:?}",
        seq.notes
    );
}

#[test]
fn the_primary_wavelength_is_the_heaviest_line_and_the_d_line_breaks_ties() {
    let (mut s, src, det) = singlet_bench(48.0);
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.45, 1.0), (0.55, 2.0), (0.65, 1.0)]);
    assert_eq!(export(&s, src, det).wavelengths, vec![0.55, 0.45, 0.65]);
    s.sources[0].spectrum =
        Spectrum::Lines(vec![(0.45, 1.0), (0.57, 1.0), (0.60, 1.0), (0.66, 1.0)]);
    // 0.60 is 12.4 nm from the d line and 0.57 is 17.6 nm from it.
    assert_eq!(
        export(&s, src, det).wavelengths,
        vec![0.60, 0.45, 0.57, 0.66]
    );
}

#[test]
fn the_path_to_export_is_the_scene_s_only_one_or_it_must_be_named() {
    let (mut s, src, det) = singlet_bench(48.0);
    assert_eq!(default_path(&s).unwrap(), (src, det));
    s.add_area_detector(Vec3::new(60.0, 0.0, 0.0), -Vec3::X, 12.0, 12.0, [8, 8]);
    assert!(matches!(default_path(&s), Err(ExportError::WhichPath(_))));
    declare_path(&mut s, src, det, None);
    assert_eq!(
        default_path(&s).unwrap(),
        (src, det),
        "a declared path settles it"
    );
}

/// The interior-stop bench, as the file the Python bridge's Optiland test
/// reads (`crates/oc-py/tests/data/interior-stop.ocanvas`). Built here so it is
/// the same scene this suite holds the export to, and compared rather than
/// trusted, so the two cannot drift apart. Regenerate with
/// `UPDATE_BRIDGE_FIXTURES=1 cargo test -p oc-trace --features trace3 --test seq_export3`.
#[test]
fn the_bridge_s_interior_stop_file_is_this_bench() {
    let (mut s, src, det, _front, stop) = interior_stop_bench();
    declare_path(&mut s, src, det, Some(stop));
    s.validate().unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../oc-py/tests/data/interior-stop.ocanvas");
    if std::env::var_os("UPDATE_BRIDGE_FIXTURES").is_some() {
        std::fs::write(&path, s.to_json() + "\n").unwrap();
    }
    let kept: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("the fixture exists")).unwrap();
    let now: serde_json::Value = serde_json::from_str(&s.to_json()).unwrap();
    assert_eq!(
        kept, now,
        "interior-stop.ocanvas no longer matches the bench; regenerate it"
    );
}

#[test]
fn a_fresnel_lens_is_refused_by_name() {
    let mut s = Scene3::new();
    let (nd, vd) = oc_core::fresnel_lens::PMMA_ND_VD;
    let lens = s
        .add_fresnel_lens(
            Vec3::ZERO,
            0.0,
            &oc_core::fresnel_lens::FresnelLens1::default(),
            oc_core::material::Material::AbbePoint { nd, vd },
        )
        .unwrap();
    let src = s.add_collimated_beam(
        Vec3::new(-70.0, 0.0, 0.0),
        5.0,
        5.0,
        9,
        Spectrum::Monochromatic(0.58756),
    );
    let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::X, 12.0, 12.0, [8, 8]);
    match export_seq(&s, src, det) {
        Err(ExportError::Unsupported(reasons)) => assert!(
            reasons
                .iter()
                .any(|r| r.contains("Fresnel lens") && r.contains(&format!("#{lens}"))),
            "{reasons:?}"
        ),
        other => panic!("{other:?}"),
    }
}
