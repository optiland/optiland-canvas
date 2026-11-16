//! The 2D tracer is an exact reduction of the 3D one, or it is not used (§10.2,
//! §15.3).
//!
//! This is the suite that earns the right to keep the fast planar path. Each
//! positive fixture builds a v2 scene with a genuinely 3D body, reduces it to the
//! v1 scene the fast tracer consumes, and then traces both: the v1 scene in 2D,
//! and the v2 scene in 3D with the *same* rays lifted into `Z = 0`. Every event —
//! position, direction, flux, and OPL — must agree to the tolerances the contract
//! sets, or the reduction was a silent approximation and the fixture fails.
//!
//! The negative fixtures matter as much as the positive ones: the reducer must
//! **refuse** an out-of-plane mirror, a cone source, a tilted body, an area
//! detector, an aperture, and a non-canonical plane, rather than reduce them
//! anyway (§10.2 exit criterion).
//!
//! The reduction itself is exact: it copies geometry, materials, and priorities
//! and changes no number. What it is compared against is the *legacy* 2D tracer,
//! which still carries the fixed scene-scale epsilons §2 marks for later removal —
//! it advances each continuation ray by `EPS_ORIGIN = 1e-7` mm past a surface
//! (`tracer.rs`). So the quantities that are exact in both engines are compared
//! tightly, and the two that inherit that nudge (a continuation segment's *start*
//! point and its accumulated OPL at the *far* end) are compared at the epsilon
//! floor, with a note that the floor is the 2D tracer's, not the reducer's.
//!
//! | Quantity | Tolerance | Exact in both? |
//! |---|---:|---|
//! | hit point (segment end) | `<= 1e-9 mm` | yes |
//! | ray direction | `<= 1e-12` | yes (the nudge is along the ray) |
//! | flux | `<= 1e-12` | yes |
//! | OPL at segment start | `<= 1e-9 mm` | yes |
//! | OPL at segment end | `<= 5e-6 mm` | no — 2D `EPS_ORIGIN` floor |
#![cfg(feature = "trace3")]

use oc_core::id::ObjectId;
use oc_core::material::{Material, LAMBDA_D};
use oc_core::planar_reduce::{preview_mode, reduce_to_planar, PreviewMode, ReductionIssue};
use oc_core::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene::{
    Detector, Element, ElementId, Layer, Scene as Scene2, Source as Source2, SourceKind, Spectrum,
    SurfaceProps, SurfaceSpec,
};
use oc_core::scene3::{
    AngularDistribution3, Aperture3, ApertureInteraction, ApertureShape3, Body3, Detector3,
    DetectorGeometry3, DetectorResponse, DetectorTermination, ObjectCommon, PlanarShape2, Scene3,
    SolidGeometry3, Source3, SourceSampling, SpatialDistribution3, SpectralScalar,
    SurfaceGeometry3, SurfaceObject3, SurfaceProps3, SurfaceResponse,
};
use oc_core::vec::{Isometry2, Vec2};
use oc_core::vec3::{Isometry3, Plane3, Ray3, Vec3};
use oc_trace::{trace, trace3, LaunchRay3, Segment, Segment3, TraceConfig, TraceConfig3};

// ---------------------------------------------------------------------------
// Config: ghosts off on both sides, so each launch ray is a single primary
// chain and the comparison is unambiguous.
// ---------------------------------------------------------------------------

fn cfg2() -> TraceConfig {
    TraceConfig {
        ghosts: false,
        flux_eps: 1e-12,
        max_depth: 64,
        ..TraceConfig::default()
    }
}

fn cfg3() -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        flux_eps: 1e-12,
        max_depth: 64,
        // The v1 planar tracer has no polarization: every surface takes the mean
        // of the s and p powers and hands on unpolarized light. So the 3D
        // tracer runs the same model here, or this would be comparing two
        // different physics and calling the difference a reduction error.
        //
        // The difference is real and is not small enough to ignore: two glass
        // faces in a row transmit about 7e-5 more of a beam once the first
        // one's p enrichment is carried into the second.
        // `two_glass_faces_transmit_more_than_the_mean_of_means` measures it.
        polarization: false,
        ..TraceConfig3::default()
    }
}

// ---------------------------------------------------------------------------
// Lifting the reduced v1 source's rays into the plane, so both tracers see the
// identical launch.
// ---------------------------------------------------------------------------

fn launch_rays(s2: &Scene2) -> Vec<LaunchRay3> {
    s2.sources
        .iter()
        .flat_map(|s| s.emit())
        .map(|er| LaunchRay3 {
            ray: Ray3::new(
                Vec3::new(er.ray.origin.x, er.ray.origin.y, 0.0),
                Vec3::new(er.ray.dir.x, er.ray.dir.y, 0.0),
            )
            .expect("an in-plane ray is a valid 3D ray"),
            lambda_um: er.lambda_um,
            flux: er.flux,
            source: None,
            pol: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Chain comparison
// ---------------------------------------------------------------------------

/// Sort key: OPL at the segment start, then the start point, so parallel rays at
/// equal OPL still order deterministically by height.
fn key2(s: &Segment) -> (f64, f64, f64) {
    (s.opl0, s.a.x, s.a.y)
}
fn key3(s: &Segment3) -> (f64, f64, f64) {
    (s.opl0_mm, s.a.x, s.a.y)
}

fn sorted2(r: &oc_trace::TraceResult) -> Vec<Segment> {
    let mut v = r.segments.clone();
    v.sort_by(|a, b| key2(a).partial_cmp(&key2(b)).unwrap());
    v
}
fn sorted3(r: &oc_trace::TraceResult3) -> Vec<Segment3> {
    // Ghosts are off, so every segment is primary; sort to match.
    let mut v = r.segments.clone();
    v.sort_by(|a, b| key3(a).partial_cmp(&key3(b)).unwrap());
    v
}

/// Assert the 3D trace, restricted to the plane, reproduces the 2D trace event
/// for event (§15.3).
///
/// Segments pair up by sort key: with ghosts off each launch ray is one linear
/// chain, and the OPL-at-start is exact and distinct per stage, so it orders the
/// chain unambiguously. The two quantities the legacy 2D tracer nudges — a
/// continuation segment's *start* point and its OPL at the *far* end — are held
/// to the epsilon floor; everything else is exact (see the module header).
fn assert_planar_equivalent(r2: &oc_trace::TraceResult, r3: &oc_trace::TraceResult3) {
    const POS: f64 = 1e-9;
    const DIR: f64 = 1e-12;
    // OPL accumulates the 2D tracer's EPS_ORIGIN nudge once per surface crossing,
    // so it drifts by ~n·1e-7 mm per bounce — the legacy tracer's floor, not the
    // reducer's, which copies geometry unchanged.
    const OPL: f64 = 5e-6;
    const FLUX: f64 = 1e-12;
    const IN_PLANE: f64 = 1e-10;
    // Longer than any fixture but far shorter than the escape length (bounds =
    // 1e4 mm), so a segment that runs off to the scene bounds is recognised.
    const ESCAPE: f64 = 1e3;

    let a = sorted2(r2);
    let b = sorted3(r3);
    assert_eq!(
        a.len(),
        b.len(),
        "segment counts differ: 2D has {}, 3D has {}",
        a.len(),
        b.len()
    );
    assert!(!a.is_empty(), "no segments traced");

    for (i, (s2, s3)) in a.iter().zip(&b).enumerate() {
        // Every 3D event stays in the plane.
        assert!(
            s3.a.z.abs() <= IN_PLANE && s3.b.z.abs() <= IN_PLANE,
            "segment {i}: 3D event left the plane (a.z={}, b.z={})",
            s3.a.z,
            s3.b.z
        );

        // The first segment of each chain starts at the exact launch point.
        if s2.opl0 == 0.0 {
            assert!(
                (s3.a.x - s2.a.x).abs() <= POS && (s3.a.y - s2.a.y).abs() <= POS,
                "segment {i} launch: 2D {:?} vs 3D ({}, {})",
                s2.a,
                s3.a.x,
                s3.a.y
            );
        }

        // The end point is the exact surface hit — unless the segment escapes to
        // the scene bounds, where only the direction is meaningful.
        let escapes = (s2.b - s2.a).length() > ESCAPE;
        if !escapes {
            assert!(
                (s3.b.x - s2.b.x).abs() <= POS && (s3.b.y - s2.b.y).abs() <= POS,
                "segment {i} hit: 2D {:?} vs 3D ({}, {})",
                s2.b,
                s3.b.x,
                s3.b.y
            );
        }

        // Direction — exact, because the 2D nudge is along the ray.
        let d2 = (s2.b - s2.a).normalize();
        let d3 = (s3.b - s3.a).normalize();
        assert!(
            (d3.x - d2.x).abs() <= DIR && (d3.y - d2.y).abs() <= DIR && d3.z.abs() <= DIR,
            "segment {i} direction: 2D {:?} vs 3D ({}, {}, {})",
            d2,
            d3.x,
            d3.y,
            d3.z
        );

        // Flux (constant along a segment with no volume absorption).
        assert!(
            (s3.flux_start - s2.flux).abs() <= FLUX + FLUX * s2.flux.abs(),
            "segment {i} flux: 2D {} vs 3D {}",
            s2.flux,
            s3.flux_start
        );

        // OPL at both ends, floored by the 2D nudge.
        assert!(
            (s3.opl0_mm - s2.opl0).abs() <= OPL && (s3.opl1_mm - s2.opl1).abs() <= OPL,
            "segment {i} OPL: 2D [{}, {}] vs 3D [{}, {}]",
            s2.opl0,
            s2.opl1,
            s3.opl0_mm,
            s3.opl1_mm
        );
    }
}

/// The whole ritual: reduce, lift, trace both, compare. Also asserts the reduced
/// scene reports an exact preview and the 3D reference rejected nothing.
fn check_equivalent(s3: &Scene3) {
    s3.validate().expect("fixture scene is valid");
    let (s2, proof) = reduce_to_planar(s3, &Plane3::XY).expect("scene reduces");
    assert_eq!(proof.plane, Plane3::XY);

    let launch = launch_rays(&s2);
    assert!(!launch.is_empty(), "reduced scene produced no launch rays");

    let r2 = trace(&s2, &cfg2());
    let r3 = trace3(s3, &launch, &cfg3());

    // The reference must have traced the whole scene exactly — no rejected
    // capability, or the equivalence claim is vacuous.
    assert!(
        r3.capability_report.rejected.is_empty(),
        "3D reference rejected part of the scene: {:?}",
        r3.capability_report.rejected
    );

    assert_planar_equivalent(&r2, &r3);
}

// ---------------------------------------------------------------------------
// Scene builders
// ---------------------------------------------------------------------------

fn glass(nd: f64) -> Material {
    Material::AbbePoint { nd, vd: 64.0 }
}

fn push_body(
    s: &mut Scene3,
    geometry: SolidGeometry3,
    material: Material,
    t: Isometry3,
) -> ObjectId {
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, oc_core::id::LayerId(0));
    common.transform = t;
    s.bodies.push(Body3 {
        common,
        geometry,
        material,
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    id
}

/// A biconvex revolved singlet, authored exactly as the generators do: two conic
/// faces and two rim lines (see `geom3.rs::singlet_profile`).
fn revolved_singlet(s: &mut Scene3, r1: f64, r2: f64, thickness: f64, half_ap: f64, nd: f64) {
    let (zf, zr) = (-thickness / 2.0, thickness / 2.0);
    let sag = |c: f64, ha: f64| c * ha * ha / (1.0 + (1.0 - c * c * ha * ha).max(0.0).sqrt());
    let (c1, c2) = (1.0 / r1, 1.0 / r2);
    let ftip = zf + sag(c1, half_ap);
    let rtip = zr + sag(c2, half_ap);
    let edges = vec![
        ProfileEdge2 {
            patch: s.alloc_patch_id(),
            curve: Curve2::Conic(ConicParams2 {
                vertex: Vec2::new(zf, 0.0),
                axis: Vec2::X,
                c: c1,
                k: 0.0,
                half_ap,
            }),
        },
        ProfileEdge2 {
            patch: s.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(ftip, half_ap),
                b: Vec2::new(rtip, half_ap),
            },
        },
        ProfileEdge2 {
            patch: s.alloc_patch_id(),
            curve: Curve2::Conic(ConicParams2 {
                vertex: Vec2::new(zr, 0.0),
                axis: Vec2::X,
                c: c2,
                k: 0.0,
                half_ap,
            }),
        },
        ProfileEdge2 {
            patch: s.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(rtip, -half_ap),
                b: Vec2::new(ftip, -half_ap),
            },
        },
    ];
    push_body(
        s,
        SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            construction: ProfileConstruction3::RevolveLocalX,
        }),
        glass(nd),
        Isometry3::IDENTITY,
    );
}

/// A right-angle glass prism, extruded through the plane along local Z.
fn extruded_prism(s: &mut Scene3, nd: f64) {
    let verts = [
        Vec2::new(-2.0, -2.0),
        Vec2::new(-2.0, 2.0),
        Vec2::new(2.0, -2.0),
    ];
    let edges: Vec<ProfileEdge2> = (0..3)
        .map(|i| ProfileEdge2 {
            patch: s.alloc_patch_id(),
            curve: Curve2::Line {
                a: verts[i],
                b: verts[(i + 1) % 3],
            },
        })
        .collect();
    push_body(
        s,
        SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            construction: ProfileConstruction3::ExtrudeLocalZ {
                z_min_mm: -5.0,
                z_max_mm: 5.0,
            },
        }),
        glass(nd),
        Isometry3::IDENTITY,
    );
}

/// A collimated ray (or beam) at height `y`, aimed along +X.
fn collimated_source(s: &mut Scene3, y: f64, half_width: f64, rays: u32) {
    let id = s.alloc_source_id();
    s.sources.push(Source3 {
        window: Default::default(),
        id,
        transform: Isometry3::from_pos(Vec3::new(-30.0, y, 0.0)),
        spatial: SpatialDistribution3::LineLegacy {
            half_width_mm: half_width,
        },
        angular: AngularDistribution3::Direction { local_dir: Vec3::X },
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        samples: SourceSampling {
            spatial_samples: rays,
            angular_samples: 1,
            ..SourceSampling::default()
        },
    });
}

// ---------------------------------------------------------------------------
// Positive fixtures
// ---------------------------------------------------------------------------

#[test]
fn a_revolved_singlet_reduces_exactly_on_axis() {
    let mut s = Scene3::new();
    revolved_singlet(&mut s, 20.0, -20.0, 4.0, 8.0, 1.5);
    collimated_source(&mut s, 0.0, 0.0, 1);
    check_equivalent(&s);
}

#[test]
fn a_revolved_singlet_reduces_exactly_off_axis() {
    // The proof for conic faces: an off-axis ray refracts at a curved surface at a
    // real angle of incidence, in both engines, to 1e-10 mm.
    let mut s = Scene3::new();
    revolved_singlet(&mut s, 20.0, -20.0, 4.0, 8.0, 1.5);
    collimated_source(&mut s, 3.0, 0.0, 1);
    check_equivalent(&s);
}

#[test]
fn a_glass_sphere_reduces_to_its_great_circle() {
    let mut s = Scene3::new();
    push_body(
        &mut s,
        SolidGeometry3::Sphere { radius_mm: 8.0 },
        glass(1.5),
        Isometry3::IDENTITY,
    );
    collimated_source(&mut s, 3.0, 0.0, 1);
    check_equivalent(&s);
}

#[test]
fn an_extruded_prism_reduces_to_its_cross_section() {
    let mut s = Scene3::new();
    extruded_prism(&mut s, 1.5);
    collimated_source(&mut s, 0.5, 0.0, 1);
    check_equivalent(&s);
}

#[test]
fn a_glass_slab_reduces_for_a_whole_collimated_beam() {
    // A box (glass slab) and five parallel rays, so the comparison exercises the
    // multi-ray sort as well as a flat refracting face.
    let mut s = Scene3::new();
    push_body(
        &mut s,
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(2.0, 20.0, 20.0),
        },
        glass(1.5),
        Isometry3::IDENTITY,
    );
    collimated_source(&mut s, 0.0, 6.0, 5);
    check_equivalent(&s);
}

#[test]
fn a_revolved_singlet_reports_an_exact_preview() {
    let mut s = Scene3::new();
    revolved_singlet(&mut s, 20.0, -20.0, 4.0, 8.0, 1.5);
    collimated_source(&mut s, 0.0, 0.0, 1);
    assert_eq!(preview_mode(&s), PreviewMode::ExactPlanarReduction);
}

// ---------------------------------------------------------------------------
// Negative fixtures — the reducer refuses rather than faking (§10.2)
// ---------------------------------------------------------------------------

#[test]
fn an_out_of_plane_mirror_is_refused() {
    let mut s = Scene3::new();
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, oc_core::id::LayerId(0));
    // A mirror tilted so its normal has a Z component: it would fold light out of
    // the plane, and no planar reduction can represent that.
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, Vec3::new(0.0, 1.0, 1.0).normalize());
    common.transform = Isometry3::new(Vec3::ZERO, [q.x, q.y, q.z, q.w]).unwrap();
    s.surfaces.push(SurfaceObject3 {
        common,
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Rectangle {
                half_width_mm: 5.0,
                half_height_mm: 5.0,
            },
        },
        response_front: SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(1.0),
        },
        response_back: SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(1.0),
        },
    });
    match reduce_to_planar(&s, &Plane3::XY) {
        Err(ReductionIssue::ZeroThicknessSurface {
            out_of_plane: true, ..
        }) => {}
        other => panic!("expected an out-of-plane surface refusal, got {other:?}"),
    }
}

#[test]
fn a_cone_source_is_refused() {
    let mut s = Scene3::new();
    let id = s.alloc_source_id();
    s.sources.push(Source3 {
        window: Default::default(),
        id,
        transform: Isometry3::from_pos(Vec3::new(-10.0, 0.0, 0.0)),
        spatial: SpatialDistribution3::Point,
        angular: AngularDistribution3::UniformConeSolidAngle {
            half_angle_rad: 0.2,
        },
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        samples: SourceSampling::default(),
    });
    match reduce_to_planar(&s, &Plane3::XY) {
        Err(ReductionIssue::NonPlanarSource { .. }) => {}
        other => panic!("expected a non-planar source refusal, got {other:?}"),
    }
}

#[test]
fn a_tilted_body_is_refused() {
    let mut s = Scene3::new();
    // A slab rotated about +Y tips its faces out of the plane.
    let t = Isometry3::from_axis_angle(Vec3::Y, 0.3).unwrap();
    push_body(
        &mut s,
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(2.0, 5.0, 5.0),
        },
        glass(1.5),
        t,
    );
    match reduce_to_planar(&s, &Plane3::XY) {
        Err(ReductionIssue::OutOfPlaneObject { .. }) => {}
        other => panic!("expected an out-of-plane body refusal, got {other:?}"),
    }
}

#[test]
fn an_area_detector_is_refused() {
    let mut s = Scene3::new();
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, oc_core::id::LayerId(0));
    common.transform = Isometry3::from_pos(Vec3::new(20.0, 0.0, 0.0));
    s.detectors.push(Detector3 {
        common,
        geometry: DetectorGeometry3::Rectangle {
            half_width_mm: 5.0,
            half_height_mm: 5.0,
        },
        bins: [16, 16],
        response: DetectorResponse::Unity,
        termination: DetectorTermination::Terminate,
        measure: oc_core::scene3::DetectorMeasure::Power,
    });
    match reduce_to_planar(&s, &Plane3::XY) {
        Err(ReductionIssue::AreaDetector { .. }) => {}
        other => panic!("expected an area-detector refusal, got {other:?}"),
    }
}

#[test]
fn an_aperture_is_refused() {
    let mut s = Scene3::new();
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, oc_core::id::LayerId(0));
    common.transform = Isometry3::from_pos(Vec3::new(5.0, 0.0, 0.0));
    s.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Disc { radius_mm: 2.0 },
            outer_extent: Some(PlanarShape2::Disc { radius_mm: 10.0 }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });
    match reduce_to_planar(&s, &Plane3::XY) {
        Err(ReductionIssue::Aperture { .. }) => {}
        other => panic!("expected an aperture refusal, got {other:?}"),
    }
}

#[test]
fn a_non_canonical_plane_is_refused() {
    let mut s = Scene3::new();
    revolved_singlet(&mut s, 20.0, -20.0, 4.0, 8.0, 1.5);
    // A plane through the origin but tilted: not the Canvas Z = 0.
    let plane = Plane3::new(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 1.0, 1.0)).unwrap();
    match reduce_to_planar(&s, &plane) {
        Err(ReductionIssue::UnsupportedPlane { .. }) => {}
        other => panic!("expected an unsupported-plane refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Legacy compatibility: v1 → migrate → reduce reproduces v1 (§10.3)
// ---------------------------------------------------------------------------

/// A small v1 scene: a glass block, a collimated beam, and a line detector.
fn v1_scene() -> Scene2 {
    let mut s = Scene2::new();
    let half = 3.0;
    let boundary = vec![
        oc_core::Edge::Line {
            a: Vec2::new(-half, -half),
            b: Vec2::new(half, -half),
        },
        oc_core::Edge::Line {
            a: Vec2::new(half, -half),
            b: Vec2::new(half, half),
        },
        oc_core::Edge::Line {
            a: Vec2::new(half, half),
            b: Vec2::new(-half, half),
        },
        oc_core::Edge::Line {
            a: Vec2::new(-half, half),
            b: Vec2::new(-half, -half),
        },
    ];
    let mut e = Element::new(ElementId(0), boundary, glass(1.5));
    e.surface = SurfaceProps {
        default: SurfaceSpec::default(),
        overrides: Vec::new(),
    };
    s.add_element(e);
    s.sources.push(Source2 {
        kind: SourceKind::Collimated { half_width: 2.0 },
        transform: Isometry2::from_pos(Vec2::new(-20.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 5,
    });
    s.detectors.push(Detector::new(
        Vec2::new(20.0, -6.0),
        Vec2::new(20.0, 6.0),
        16,
    ));
    let _ = Layer::default();
    s
}

#[test]
fn migrating_then_reducing_a_v1_scene_reproduces_it() {
    // The round trip that makes legacy-planar compatibility exact rather than
    // merely close: reduce(migrate(v1)) must trace bit-for-bit like v1 did.
    let v1 = v1_scene();
    let (s3, _report) = oc_core::migrate_v1::migrate_v1_to_v2(&v1).expect("migrates");
    let (back, _proof) = reduce_to_planar(&s3, &Plane3::XY).expect("reduces");

    let ra = trace(&v1, &cfg2());
    let rb = trace(&back, &cfg2());

    let a = sorted2(&ra);
    let b = sorted2(&rb);
    assert_eq!(a.len(), b.len(), "segment counts differ after round trip");
    for (s1, s2) in a.iter().zip(&b) {
        assert!((s1.a - s2.a).length() <= 1e-12);
        assert!((s1.b - s2.b).length() <= 1e-12);
        assert!((s1.flux - s2.flux).abs() <= 1e-12);
        assert!((s1.opl1 - s2.opl1).abs() <= 1e-12);
    }

    // The detector reading survives too.
    assert_eq!(ra.detectors.len(), rb.detectors.len());
    let da = &ra.detectors[0];
    let db = &rb.detectors[0];
    assert_eq!(da.hits, db.hits);
    assert!((da.total_flux - db.total_flux).abs() <= 1e-12);
    assert_eq!(da.centroid_u().is_some(), db.centroid_u().is_some());
}

#[test]
fn an_unpromoted_v1_scene_previews_as_legacy() {
    let v1 = v1_scene();
    let (s3, _report) = oc_core::migrate_v1::migrate_v1_to_v2(&v1).expect("migrates");
    assert_eq!(preview_mode(&s3), PreviewMode::LegacyPlanarCompatibility);
}
