//! A 3D scene built from a sequential prescription (`oc_core::seq_import`).
//!
//! What this suite owes:
//!
//! - a prescription's lenses become revolved bodies whose faces sit where its
//!   vertices are, and exporting the scene gives the prescription back:
//!   radii, gaps, glasses and the stop;
//! - a cemented surface is one face shared by two bodies, and both sides of
//!   it export with the right glass;
//! - the stop is an aperture the scene's path declares, and each field's beam
//!   is the paraxial entrance pupil, its chief ray through the pupil's centre;
//! - an object at a finite distance in a telecentric object space is a point
//!   per field with a cone about the axis, and exports as an object-space NA;
//! - aimed beams are built square to the field and the stop's mask reaches the
//!   barrel;
//! - a glass keeps its catalog formula; an ideal medium keeps its one index;
//! - an image inside a medium stays inside it;
//! - faces far apart in clear aperture get a shoulder, not a bowl;
//! - everything the scene cannot hold is refused, every reason at once.
#![cfg(feature = "trace3")]

use oc_core::material::{DispersionFormula, Material};
use oc_core::scene3::{
    AngularDistribution3, PlanarShape2, Scene3, SolidGeometry3, SpatialDistribution3,
};
use oc_core::seq::{AimedBeam, FieldType, SeqAperture, SeqMaterial, SeqSurface, SeqSystem};
use oc_core::seq_import::{import_seq, ImportOptions, Imported};
use oc_core::vec3::Vec3;
use oc_trace::{default_path, export_seq};

const EPS: f64 = 1e-9;

fn surface(radius: Option<f64>, thickness: f64, semi: f64, material: SeqMaterial) -> SeqSurface {
    SeqSurface {
        radius,
        conic: 0.0,
        thickness,
        semi_aperture: semi,
        coefficients: Vec::new(),
        material,
        reflective: false,
    }
}

/// Schott N-BK7 and F2 as refractiveindex.info publishes them (formula 2).
fn nbk7() -> SeqMaterial {
    SeqMaterial::Glass {
        name: "N-BK7".into(),
        reference: Some("schott".into()),
        formula: Some(DispersionFormula::Sellmeier2 {
            coefficients: vec![
                0.0,
                1.039_612_12,
                0.006_000_698_67,
                0.231_792_344,
                0.020_017_914_4,
                1.010_469_45,
                103.560_653,
            ],
        }),
        sellmeier: None,
    }
}

fn f2() -> SeqMaterial {
    SeqMaterial::Glass {
        name: "F2".into(),
        reference: Some("schott".into()),
        formula: Some(DispersionFormula::Sellmeier2 {
            coefficients: vec![
                0.0,
                1.345_333_59,
                0.009_977_438_71,
                0.209_073_176,
                0.047_045_076_7,
                0.937_357_162,
                111.886_764,
            ],
        }),
        sellmeier: None,
    }
}

fn system(surfaces: Vec<SeqSurface>, stop: Option<usize>) -> SeqSystem {
    SeqSystem {
        surfaces,
        stop,
        aperture: SeqAperture::Epd { value: 10.0 },
        fields: vec![0.0, 5.0],
        field_type: FieldType::Angle,
        object_distance: None,
        object_space_telecentric: false,
        image_semi_aperture: None,
        aimed_beams: Vec::new(),
        wavelengths: vec![0.587_56, 0.486_13, 0.656_27],
        notes: Vec::new(),
    }
}

/// A plane stop 10 mm in front of an N-BK7 singlet (R 60 / -60, 5 mm), image
/// 50 mm behind it.
fn stopped_singlet() -> SeqSystem {
    system(
        vec![
            surface(None, 10.0, 5.0, SeqMaterial::Air),
            surface(Some(60.0), 5.0, 8.0, nbk7()),
            surface(Some(-60.0), 50.0, 8.0, SeqMaterial::Air),
        ],
        Some(0),
    )
}

fn build(seq: &SeqSystem) -> Imported {
    import_seq(seq, &ImportOptions::default()).unwrap_or_else(|e| panic!("{e}"))
}

fn export(scene: &Scene3) -> SeqSystem {
    let (source, detector) = default_path(scene).expect("the scene declares its path");
    export_seq(scene, source, detector).unwrap_or_else(|e| panic!("{e}"))
}

fn assert_same_lens(back: &SeqSystem, seq: &SeqSystem) {
    assert_eq!(back.surfaces.len(), seq.surfaces.len());
    for (i, (b, s)) in back.surfaces.iter().zip(&seq.surfaces).enumerate() {
        match (b.radius, s.radius) {
            (Some(rb), Some(rs)) => assert!((rb - rs).abs() < EPS * rs.abs(), "S{i}: {rb} vs {rs}"),
            (None, None) => {}
            other => panic!("S{i}: radius {other:?}"),
        }
        assert!((b.thickness - s.thickness).abs() < EPS, "S{i} thickness");
        assert_eq!(
            std::mem::discriminant(&b.material),
            std::mem::discriminant(&s.material),
            "S{i} material"
        );
    }
    assert_eq!(back.stop, seq.stop);
}

#[test]
fn a_prescription_becomes_a_scene_that_exports_as_itself() {
    let seq = stopped_singlet();
    let built = build(&seq);
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    let scene = &built.scene;
    assert_eq!(scene.bodies.len(), 1);
    assert_eq!(scene.apertures.len(), 1);
    assert_eq!(scene.sources.len(), 2, "one beam per field");
    assert_eq!(scene.detectors.len(), 1);
    // The lens's faces sit at the prescription's vertices: x = 10 and 15.
    let body = &scene.bodies[0];
    let SolidGeometry3::Profile(p) = &body.geometry else {
        panic!("a revolved body");
    };
    let centre = body.common.transform.pos.x;
    assert!((centre - 12.5).abs() < EPS);
    let faces: Vec<_> = p
        .profile
        .edges
        .iter()
        .filter_map(|e| match &e.curve {
            oc_core::profile2::Curve2::Conic(c) => Some(c.vertex.x + centre),
            _ => None,
        })
        .collect();
    assert_eq!(faces.len(), 2);
    assert!((faces[0] - 10.0).abs() < EPS && (faces[1] - 15.0).abs() < EPS);

    let back = export(scene);
    assert_same_lens(&back, &seq);
    let SeqMaterial::Glass { name, formula, .. } = &back.surfaces[1].material else {
        panic!("glass behind S2");
    };
    assert_eq!(name, "N-BK7");
    assert!(formula.is_some(), "the glass keeps its formula");
    assert_eq!(back.wavelengths, seq.wavelengths, "the primary stays first");
    match back.aperture {
        SeqAperture::Epd { value } => assert!((value - 10.0).abs() < EPS),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_glass_is_the_catalog_formula_and_the_ideal_medium_one_index() {
    let mut seq = stopped_singlet();
    let scene = build(&seq).scene;
    let Material::Catalog(id) = scene.bodies[0].material else {
        panic!("a catalog glass");
    };
    let glass = scene.catalog.get(id).unwrap();
    let SeqMaterial::Glass {
        formula: Some(f), ..
    } = nbk7()
    else {
        unreachable!()
    };
    for lambda in [0.4, 0.58756, 0.75] {
        assert_eq!(glass.index(lambda), f.index(lambda));
    }
    assert_eq!(glass.reference.as_deref(), Some("schott"));

    seq.surfaces[1].material = SeqMaterial::Ideal { n: 4.002 };
    let scene = build(&seq).scene;
    let m = &scene.bodies[0].material;
    for lambda in [0.5, 4.2, 10.6] {
        assert_eq!(m.index_at(lambda, &scene.catalog), Some(4.002));
    }
    assert!(matches!(
        export(&scene).surfaces[1].material,
        SeqMaterial::Ideal { n } if n == 4.002
    ));
}

#[test]
fn a_cemented_face_is_one_surface_between_two_glasses() {
    let seq = system(
        vec![
            surface(Some(40.0), 6.0, 10.0, nbk7()),
            surface(Some(-30.0), 2.0, 10.0, f2()),
            surface(Some(-120.0), 60.0, 10.0, SeqMaterial::Air),
        ],
        Some(0),
    );
    let scene = build(&seq).scene;
    assert_eq!(scene.bodies.len(), 2);
    let back = export(&scene);
    // The stop sits on the first face's vertex, so it exports as its own
    // plane there, ahead of the lens.
    assert_eq!(back.surfaces.len(), 4);
    let names: Vec<String> = back
        .surfaces
        .iter()
        .map(|s| match &s.material {
            SeqMaterial::Glass { name, .. } => name.clone(),
            SeqMaterial::Air => "air".into(),
            other => format!("{other:?}"),
        })
        .collect();
    let glasses: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(
        glasses == ["air", "N-BK7", "F2", "air"] || glasses == ["N-BK7", "N-BK7", "F2", "air"],
        "{glasses:?}"
    );
    let r: Vec<Option<f64>> = back.surfaces.iter().map(|s| s.radius).collect();
    assert_eq!(r[0], None, "the stop is a plane");
    assert!((r[2].unwrap() + 30.0).abs() < EPS, "the cemented face");
}

#[test]
fn each_field_is_the_paraxial_pupil_with_its_chief_ray_through_the_centre() {
    // An interior stop: the lens images it forward into object space.
    let seq = system(
        vec![
            surface(Some(40.0), 5.0, 9.0, nbk7()),
            surface(Some(-40.0), 8.0, 9.0, SeqMaterial::Air),
            surface(None, 40.0, 4.0, SeqMaterial::Air),
        ],
        Some(2),
    );
    let scene = build(&seq).scene;
    let src = &scene.sources[1];
    let AngularDistribution3::Direction { local_dir } = src.angular else {
        panic!("collimated");
    };
    let d = src.transform.rotate(local_dir).normalize();
    assert!(
        (d.y - 5.0_f64.to_radians().sin()).abs() < 1e-12,
        "5 degrees up"
    );
    // Where the beam's centre line crosses the axis is the entrance pupil.
    let p = src.transform.pos;
    let x_ep = p.x - p.y * d.x / d.y;
    // The paraxial image of the stop through the lens, by hand: the stop at
    // 8 mm behind the rear vertex, seen through the thick lens.
    let n = match &seq.surfaces[0].material {
        SeqMaterial::Glass {
            formula: Some(f), ..
        } => f.index(0.587_56),
        _ => unreachable!(),
    };
    let (c1, c2, t) = (1.0 / 40.0, -1.0 / 40.0, 5.0);
    // (y, nu) from the stop backwards is the forward matrix to the stop.
    let mul = |a: [[f64; 2]; 2], b: [[f64; 2]; 2]| {
        [
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ]
    };
    let mut m = [[1.0, 0.0], [-(n - 1.0) * c1, 1.0]];
    m = mul([[1.0, t / n], [0.0, 1.0]], m);
    m = mul([[1.0, 0.0], [-(1.0 - n) * c2, 1.0]], m);
    m = mul([[1.0, 8.0], [0.0, 1.0]], m);
    let epl = m[0][1] / m[0][0];
    assert!((x_ep - epl).abs() < 1e-9, "pupil at {x_ep}, by hand {epl}");
    let SpatialDistribution3::Disc { radius_mm } = src.spatial else {
        panic!("a disc");
    };
    assert!((radius_mm - 5.0).abs() < EPS, "EPD 10");
}

#[test]
fn a_telecentric_object_is_a_point_per_field_with_a_cone_about_the_axis() {
    let mut seq = stopped_singlet();
    seq.object_distance = Some(40.0);
    seq.object_space_telecentric = true;
    seq.field_type = FieldType::ObjectHeight;
    seq.fields = vec![0.0, 3.0];
    seq.aperture = SeqAperture::ObjectNa { value: 0.1 };
    let scene = build(&seq).scene;
    for (s, h) in scene.sources.iter().zip([0.0, 3.0]) {
        assert!(matches!(s.spatial, SpatialDistribution3::Point));
        let AngularDistribution3::UniformConeSolidAngle { half_angle_rad } = s.angular else {
            panic!("a cone");
        };
        assert!((half_angle_rad - 0.1_f64.asin()).abs() < 1e-12);
        assert!(
            (s.transform.rotate(Vec3::X) - Vec3::X).length() < 1e-12,
            "about the axis"
        );
        assert!((s.transform.pos - Vec3::new(-40.0, h, 0.0)).length() < EPS);
    }
    let back = export(&scene);
    match back.aperture {
        SeqAperture::ObjectNa { value } => assert!((value - 0.1).abs() < 1e-12),
        other => panic!("{other:?}"),
    }
    assert!((back.object_distance.unwrap() - 40.0).abs() < EPS);
}

#[test]
fn aimed_beams_are_square_to_their_field_and_the_stop_is_a_barrel() {
    let mut seq = stopped_singlet();
    seq.aimed_beams = vec![
        AimedBeam {
            centre_mm: [0.0, 0.0],
            half_width_mm: 5.1,
            half_height_mm: 5.1,
        },
        AimedBeam {
            centre_mm: [-0.9, 0.0],
            half_width_mm: 4.0,
            half_height_mm: 5.0,
        },
    ];
    let built = build(&seq);
    assert!(
        built.notes.iter().any(|n| n.contains("2.00% wider")),
        "{:?}",
        built.notes
    );
    let scene = built.scene;
    assert!(matches!(
        scene.sources[0].spatial,
        SpatialDistribution3::Disc { radius_mm } if (radius_mm - 5.1).abs() < EPS
    ));
    let s = &scene.sources[1];
    assert!(matches!(
        s.spatial,
        SpatialDistribution3::Rectangle { half_width_mm, half_height_mm }
            if (half_width_mm - 4.0).abs() < EPS && (half_height_mm - 5.0).abs() < EPS
    ));
    let d = s.transform.rotate(Vec3::X);
    assert!(
        (d.y - 5.0_f64.to_radians().sin()).abs() < 1e-12,
        "square to the field"
    );
    // The centre line crosses the first vertex plane (x = 0) where it was
    // measured.
    let p = s.transform.pos;
    let y0 = p.y - p.x * d.y / d.x;
    assert!((y0 + 0.9).abs() < 1e-9);
    let mask = match scene.apertures[0].shape.outer_extent {
        Some(PlanarShape2::Disc { radius_mm }) => radius_mm,
        ref other => panic!("{other:?}"),
    };
    assert!(mask >= 8.0, "the mask reaches the lens's rim, {mask}");
}

#[test]
fn an_image_inside_a_medium_stays_inside_it() {
    // An eye's last medium runs to the retina.
    let seq = system(
        vec![
            surface(None, 2.0, 2.0, SeqMaterial::Air),
            surface(Some(8.0), 20.0, 5.0, SeqMaterial::Ideal { n: 1.336 }),
        ],
        Some(0),
    );
    let built = build(&seq);
    assert!(built.notes.iter().any(|n| n.contains("image lies inside")));
    let scene = built.scene;
    let det = scene.detectors[0].common.transform.pos;
    assert!((det.x - 22.0).abs() < EPS);
    let n = oc_trace_medium(&scene, det - Vec3::X * 1e-6);
    assert_eq!(n, Some(1.336), "the detector sits in the medium");
}

/// The index of the body containing `p`, if any.
fn oc_trace_medium(scene: &Scene3, p: Vec3) -> Option<f64> {
    use oc_core::geom3::{Containment, Intersect3, ToleranceContext};
    let tol = ToleranceContext::for_scene_scale(10.0);
    scene
        .bodies
        .iter()
        .find(|b| b.geometry.contains(p, &b.common.transform, &tol) == Containment::Inside)
        .and_then(|b| b.material.index_at(0.587_56, &scene.catalog))
}

#[test]
fn faces_far_apart_in_aperture_get_a_shoulder_and_close_ones_one_diameter() {
    let mut seq = stopped_singlet();
    // Close: 8 and 8.5 are cut to one diameter, the singlet layout the
    // inspector reads.
    seq.surfaces[2].semi_aperture = 8.5;
    let scene = build(&seq).scene;
    let id = scene.bodies[0].common.id.0;
    assert!(scene.singlet_shape(id).is_some());

    // Far: 4 and 8 get a shoulder, and the small face ends at its own
    // aperture.
    seq.surfaces[1].semi_aperture = 4.0;
    let scene = build(&seq).scene;
    let SolidGeometry3::Profile(p) = &scene.bodies[0].geometry else {
        unreachable!()
    };
    assert!(p.profile.edges.len() > 4);
    let front = p
        .profile
        .edges
        .iter()
        .find_map(|e| match &e.curve {
            oc_core::profile2::Curve2::Conic(c) if c.c > 0.0 => Some(c.half_ap),
            _ => None,
        })
        .unwrap();
    assert!((front - 4.0).abs() < EPS);
    // And the scene still exports as the same lens.
    assert_same_lens(&export(&scene), &seq);
}

#[test]
fn a_curved_surface_between_one_medium_changes_nothing_and_says_so() {
    let mut seq = stopped_singlet();
    seq.surfaces
        .insert(1, surface(Some(25.0), 0.0, 5.0, SeqMaterial::Air));
    seq.surfaces[0].thickness = 10.0;
    let built = build(&seq);
    assert_eq!(built.scene.bodies.len(), 1);
    assert!(
        built
            .notes
            .iter()
            .any(|n| n.contains("S2") && n.contains("left out")),
        "{:?}",
        built.notes
    );
}

#[test]
fn what_a_scene_cannot_hold_is_refused_with_every_reason() {
    let mut seq = stopped_singlet();
    seq.surfaces[1].reflective = true;
    seq.surfaces[2].thickness = -3.0;
    seq.surfaces[1].material = SeqMaterial::glass("NOT-A-GLASS");
    let err = import_seq(&seq, &ImportOptions::default()).unwrap_err();
    let all = err.0.join("\n");
    assert!(all.contains("S2 is a mirror"), "{all}");
    assert!(all.contains("virtual image"), "{all}");

    let mut seq = stopped_singlet();
    seq.surfaces[1].material = SeqMaterial::glass("NOT-A-GLASS");
    let err = import_seq(&seq, &ImportOptions::default()).unwrap_err();
    assert!(err.0[0].contains("no data"), "{:?}", err.0);

    let mut seq = stopped_singlet();
    seq.object_distance = Some(30.0);
    seq.aperture = SeqAperture::ImageFno { value: 4.0 };
    let err = import_seq(&seq, &ImportOptions::default()).unwrap_err();
    assert!(err.0.iter().any(|r| r.contains("f-number")), "{:?}", err.0);

    // Two different glasses under one name.
    let mut seq = system(
        vec![
            surface(Some(40.0), 6.0, 10.0, nbk7()),
            surface(Some(-40.0), 2.0, 10.0, SeqMaterial::Air),
            surface(Some(40.0), 6.0, 10.0, f2()),
            surface(Some(-40.0), 40.0, 10.0, SeqMaterial::Air),
        ],
        Some(0),
    );
    if let SeqMaterial::Glass { name, .. } = &mut seq.surfaces[2].material {
        *name = "N-BK7".into();
    }
    let err = import_seq(&seq, &ImportOptions::default()).unwrap_err();
    assert!(
        err.0.iter().any(|r| r.contains("two different glasses")),
        "{:?}",
        err.0
    );
}
