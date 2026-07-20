//! The 3D CPU f32 preview against the f64 reference (§15.5, PR 3D-07).
//!
//! The 3D successor to `cross_validate.rs`, the project's single most important
//! correctness guard. The f32 preview is an *approximation* — its job is to look
//! right fast — so it is held to a tolerance, not to equality: a ray must land
//! within `max(5e-2 mm, 5e-5 · scene_scale)` of where the f64 reference puts it,
//! the branch structure must match, and a primitive the preview cannot trace must
//! be **rejected with a capability issue**, never swapped for a lookalike (§12.3).
//!
//! Covers the objects the compiler admits: sphere, box, cylinder, and extruded
//! line profiles; mirror surfaces; and aperture-plus-detector clipping. The
//! revolved profile bodies are the one geometry still to land, and must show up
//! in `rejected` rather than as a chord — the profile-rejection test pins that.
#![cfg(feature = "gpu3")]

use oc_core::id::{LayerId, ObjectId, SourceId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::profile2::{
    ClosedProfile2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::{
    Aperture3, ApertureInteraction, ApertureShape3, Body3, Detector3, DetectorGeometry3,
    DetectorResponse, DetectorTermination, ObjectCommon, PlanarShape2, Scene3, SolidGeometry3,
    SpectralScalar, SurfaceGeometry3, SurfaceObject3, SurfaceProps3, SurfaceResponse,
};
use oc_core::source3::EmittedRay3;
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Plane3, Ray3, Vec3};
use oc_gpu::cpu3::{self, SegOut3};
use oc_gpu::scene3::{BuildOpts, GpuScene3};
use oc_trace::{trace3, LaunchRay3, Segment3, TraceConfig3};

/// A rotation carrying local +Z onto `normal`, placing a surface/detector face.
fn facing(pos: Vec3, normal: Vec3) -> Isometry3 {
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, normal.normalize());
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap()
}

fn add_mirror(scene: &mut Scene3, pos: Vec3, normal: Vec3, half: f64) -> ObjectId {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    let resp = SurfaceResponse::Mirror {
        reflectivity: SpectralScalar::Constant(1.0),
    };
    scene.surfaces.push(SurfaceObject3 {
        common,
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Rectangle {
                half_width_mm: half,
                half_height_mm: half,
            },
        },
        response_front: resp.clone(),
        response_back: resp,
    });
    id
}

fn add_circular_aperture(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    opening: f64,
    mask: f64,
) -> ObjectId {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Disc { radius_mm: opening },
            outer_extent: Some(PlanarShape2::Disc { radius_mm: mask }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });
    id
}

fn add_rect_detector(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    hw: f64,
    hh: f64,
    bins: [usize; 2],
) -> ObjectId {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.detectors.push(Detector3 {
        common,
        geometry: DetectorGeometry3::Rectangle {
            half_width_mm: hw,
            half_height_mm: hh,
        },
        bins,
        response: DetectorResponse::Unity,
        termination: DetectorTermination::Terminate,
    });
    id
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------

fn push_body(scene: &mut Scene3, geometry: SolidGeometry3, material: Material) -> ObjectId {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::IDENTITY;
    scene.bodies.push(Body3 {
        common,
        geometry,
        material,
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    id
}

fn glass(nd: f64) -> Material {
    Material::AbbePoint { nd, vd: 64.0 }
}

/// The same rays as both a v2 launch set (f32 path) and f64 launch rays.
fn rays(origins_dirs: &[(Vec3, Vec3)]) -> (Vec<EmittedRay3>, Vec<LaunchRay3>) {
    let mut emitted = Vec::new();
    let mut launch = Vec::new();
    for (o, d) in origins_dirs {
        let ray = Ray3::new(*o, *d).unwrap();
        emitted.push(EmittedRay3 {
            ray,
            lambda_um: LAMBDA_D,
            flux: 1.0,
            source: SourceId(0),
        });
        launch.push(LaunchRay3 {
            ray,
            lambda_um: LAMBDA_D,
            flux: 1.0,
            source: None,
        });
    }
    (emitted, launch)
}

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        flux_eps: 1e-9,
        max_depth: 32,
        ..TraceConfig3::default()
    }
}

// ---------------------------------------------------------------------------
// Extracting a comparable primary path
// ---------------------------------------------------------------------------

/// The ordered non-escape hit points, then the final direction, from the f64
/// reference (ghosts off ⇒ one linear chain).
fn f64_landing(res: &oc_trace::TraceResult3, ray_index: usize, n_rays: usize) -> (Vec<Vec3>, Vec3) {
    // Rays are launched independently; the tracer tags no ray index on segments,
    // so this fixture uses one ray per trace to keep the chain unambiguous.
    assert_eq!(n_rays, 1, "f64_landing expects one ray per trace");
    let _ = ray_index;
    let mut segs: Vec<Segment3> = res.segments.to_vec();
    segs.sort_by(|a, b| a.opl0_mm.partial_cmp(&b.opl0_mm).unwrap());
    landing_from(&segs.iter().map(|s| (s.a, s.b)).collect::<Vec<_>>())
}

/// The same, from the f32 preview's one-ray segment list. The endpoints are
/// already world millimetres in f64.
fn f32_landing(segs: &[SegOut3]) -> (Vec<Vec3>, Vec3) {
    let ab: Vec<(Vec3, Vec3)> = segs.iter().map(|s| (s.a, s.b)).collect();
    landing_from(&ab)
}

/// Given path segments in order, the interior hit points and the exit direction.
fn landing_from(ab: &[(Vec3, Vec3)]) -> (Vec<Vec3>, Vec3) {
    const ESCAPE: f64 = 1e3;
    let mut hits = Vec::new();
    for (a, b) in ab {
        if (*b - *a).length() < ESCAPE {
            hits.push(*b);
        }
    }
    let exit_dir = ab
        .last()
        .map(|(a, b)| (*b - *a).normalize())
        .unwrap_or(Vec3::X);
    (hits, exit_dir)
}

/// The §15.5 landing tolerance for a scene of the given scale.
fn pos_tol(scene_scale: f64) -> f64 {
    (5e-2_f64).max(5e-5 * scene_scale)
}

fn assert_landing_matches(f64_scene: &Scene3, origins_dirs: &[(Vec3, Vec3)]) {
    let scene_scale = GpuScene3::build(f64_scene, BuildOpts::default()).norm.scale;
    let tol = pos_tol(scene_scale);

    for od in origins_dirs {
        let (emitted, launch) = rays(std::slice::from_ref(od));

        let f64_res = trace3(f64_scene, &launch, &cfg());
        let (f64_hits, f64_dir) = f64_landing(&f64_res, 0, 1);

        let gpu = GpuScene3::build_with_rays(
            f64_scene,
            &emitted,
            BuildOpts {
                ghosts: false,
                flux_eps: 1e-9,
                max_ghost: 0,
            },
        );
        assert!(
            gpu.rejected.is_empty(),
            "unexpected rejects: {:?}",
            gpu.rejected
        );
        let f32_segs = &cpu3::trace_segments(&gpu, 32)[0];
        let (f32_hits, f32_dir) = f32_landing(f32_segs);

        assert_eq!(
            f64_hits.len(),
            f32_hits.len(),
            "hit count differs for ray {od:?}: f64 {}, f32 {}",
            f64_hits.len(),
            f32_hits.len()
        );
        for (i, (a, b)) in f64_hits.iter().zip(&f32_hits).enumerate() {
            let d = (*a - *b).length();
            assert!(d <= tol, "hit {i} for ray {od:?}: |Δ| = {d} mm > {tol}");
        }
        // Exit direction: an angular check scaled to the same landing tolerance.
        let ang = (f64_dir - f32_dir).length();
        assert!(ang < 1e-3, "exit direction for ray {od:?}: |Δ| = {ang}");
    }
}

// ---------------------------------------------------------------------------
// Parity fixtures
// ---------------------------------------------------------------------------

#[test]
fn a_glass_sphere_matches_f64_for_skew_rays() {
    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Sphere { radius_mm: 8.0 },
        glass(1.5),
    );
    scene.validate().unwrap();

    // Axial plus a spread of skew rays, all entering the sphere.
    let rays = [
        (Vec3::new(-20.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(-20.0, 3.0, 0.0), Vec3::X),
        (Vec3::new(-20.0, 0.0, 3.0), Vec3::X),
        (Vec3::new(-20.0, 2.5, -2.5), Vec3::X),
        (Vec3::new(-20.0, -4.0, 1.0), Vec3::X),
    ];
    assert_landing_matches(&scene, &rays);
}

#[test]
fn a_glass_cylinder_matches_f64_for_skew_rays() {
    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Cylinder {
            radius_mm: 6.0,
            half_length_mm: 15.0,
            axis_local: Vec3::Z,
        },
        glass(1.5),
    );
    scene.validate().unwrap();

    // The cylinder stands along Z; +X rays strike its curved wall.
    let rays = [
        (Vec3::new(-20.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(-20.0, 3.0, 0.0), Vec3::X),
        (
            Vec3::new(-20.0, 2.0, 4.0),
            Vec3::new(1.0, 0.0, 0.1).normalize(),
        ),
        (
            Vec3::new(-20.0, -3.5, 2.0),
            Vec3::new(1.0, 0.05, 0.0).normalize(),
        ),
    ];
    assert_landing_matches(&scene, &rays);
}

#[test]
fn an_extruded_prism_matches_f64() {
    let mut scene = Scene3::new();
    // A right-angle prism, extruded along Z through the plane.
    let verts = [
        Vec2::new(-3.0, -3.0),
        Vec2::new(-3.0, 3.0),
        Vec2::new(3.0, -3.0),
    ];
    let id = scene.alloc_object_id();
    let edges: Vec<ProfileEdge2> = (0..3)
        .map(|i| ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: verts[i],
                b: verts[(i + 1) % 3],
            },
        })
        .collect();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::IDENTITY;
    scene.bodies.push(Body3 {
        common,
        geometry: SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            construction: ProfileConstruction3::ExtrudeLocalZ {
                z_min_mm: -8.0,
                z_max_mm: 8.0,
            },
        }),
        material: glass(1.6),
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    scene.validate().unwrap();

    let rays = [
        (Vec3::new(-20.0, 0.5, 0.0), Vec3::X),
        (
            Vec3::new(-20.0, 1.0, 2.0),
            Vec3::new(1.0, 0.0, 0.05).normalize(),
        ),
        (
            Vec3::new(-20.0, -1.0, 0.0),
            Vec3::new(1.0, 0.05, 0.0).normalize(),
        ),
    ];
    assert_landing_matches(&scene, &rays);
}

#[test]
fn a_glass_box_matches_f64_for_skew_rays() {
    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(3.0, 10.0, 10.0),
        },
        glass(1.6),
    );
    scene.validate().unwrap();

    let rays = [
        (Vec3::new(-20.0, 0.0, 0.0), Vec3::X),
        (
            Vec3::new(-20.0, 2.0, 0.0),
            Vec3::new(1.0, 0.1, 0.0).normalize(),
        ),
        (
            Vec3::new(-20.0, 0.0, 2.0),
            Vec3::new(1.0, 0.0, 0.15).normalize(),
        ),
        (
            Vec3::new(-20.0, 1.0, -1.0),
            Vec3::new(1.0, 0.1, -0.1).normalize(),
        ),
    ];
    assert_landing_matches(&scene, &rays);
}

// ---------------------------------------------------------------------------
// Capability: unsupported geometry is rejected, not approximated (§12.3)
// ---------------------------------------------------------------------------

#[test]
fn a_profile_body_is_rejected_not_approximated() {
    let mut scene = Scene3::new();
    // A profile (revolved) body: the f32 preview does not trace curved profile
    // surfaces yet, so it must reject rather than draw a chord.
    let id = scene.alloc_object_id();
    let edges = vec![
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(-1.0, 0.0),
                b: Vec2::new(1.0, 0.0),
            },
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(1.0, 0.0),
                b: Vec2::new(1.0, 2.0),
            },
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(1.0, 2.0),
                b: Vec2::new(-1.0, 2.0),
            },
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(-1.0, 2.0),
                b: Vec2::new(-1.0, 0.0),
            },
        },
    ];
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::IDENTITY;
    scene.bodies.push(Body3 {
        common,
        geometry: SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            construction: ProfileConstruction3::RevolveLocalX,
        }),
        material: glass(1.5),
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    scene.validate().unwrap();

    let gpu = GpuScene3::build(&scene, BuildOpts::default());
    assert_eq!(gpu.rejected.len(), 1, "the profile body should be rejected");
    assert_eq!(gpu.rejected[0].object, id.0);
    assert!(gpu.bodies.is_empty(), "no body should be traced");
}

// ---------------------------------------------------------------------------
// Event objects: surfaces, apertures, detectors
// ---------------------------------------------------------------------------

#[test]
fn a_tilted_mirror_folds_like_f64() {
    let mut scene = Scene3::new();
    // A 45° mirror at the origin: its normal folds a +X beam toward +Y.
    add_mirror(
        &mut scene,
        Vec3::ZERO,
        Vec3::new(-1.0, 1.0, 0.0).normalize(),
        10.0,
    );
    scene.validate().unwrap();

    let rays = [
        (Vec3::new(-15.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(-15.0, 2.0, 0.0), Vec3::X),
        (Vec3::new(-15.0, 0.0, 2.0), Vec3::X),
    ];
    assert_landing_matches(&scene, &rays);
}

/// Compare the first detector's readout between the f64 reference and the f32
/// preview for a shared launch set.
fn assert_detector_matches(scene: &Scene3, det: ObjectId, origins_dirs: &[(Vec3, Vec3)]) {
    let (emitted, launch) = rays(origins_dirs);
    let f64_res = trace3(scene, &launch, &cfg());
    let f64_det = f64_res
        .detectors
        .iter()
        .find(|d| d.detector.0 == det.0)
        .expect("f64 detector");

    let gpu = GpuScene3::build_with_rays(
        scene,
        &emitted,
        BuildOpts {
            ghosts: false,
            flux_eps: 1e-9,
            max_ghost: 0,
        },
    );
    assert!(
        gpu.rejected.is_empty(),
        "unexpected rejects: {:?}",
        gpu.rejected
    );
    let f32 = cpu3::trace(&gpu, 32);
    let f32_det = f32.detector(det.0).expect("f32 detector");

    assert_eq!(f64_det.hits, f32_det.hits, "hit count differs");
    assert!(
        (f64_det.total_incident as f32 - f32_det.total).abs() < 1e-4,
        "total flux: f64 {} vs f32 {}",
        f64_det.total_incident,
        f32_det.total
    );
    assert_eq!(
        f64_det.incident_flux.len(),
        f32_det.incident.len(),
        "cell count differs"
    );
    for (i, (a, b)) in f64_det
        .incident_flux
        .iter()
        .zip(&f32_det.incident)
        .enumerate()
    {
        assert!(
            (*a as f32 - *b).abs() < 1e-4,
            "cell {i}: f64 {a} vs f32 {b}"
        );
    }
}

#[test]
fn an_aperture_and_detector_match_f64() {
    // A circular aperture clips a grid of rays; a detector behind it records what
    // gets through. Both the clipping and the binning must match f64.
    let mut scene = Scene3::new();
    add_circular_aperture(&mut scene, Vec3::new(0.0, 0.0, 0.0), Vec3::X, 3.0, 20.0);
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        6.0,
        6.0,
        [6, 6],
    );
    scene.validate().unwrap();

    // A grid of parallel rays spanning inside and outside the opening.
    let mut rays = Vec::new();
    for iy in -3..=3 {
        for iz in -3..=3 {
            rays.push((Vec3::new(-10.0, iy as f64 * 1.2, iz as f64 * 1.2), Vec3::X));
        }
    }
    assert_detector_matches(&scene, det, &rays);
}
