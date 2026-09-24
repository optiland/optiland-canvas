//! The 3D CPU f32 preview against the f64 reference (§15.5, PR 3D-07).
//!
//! The 3D successor to `cross_validate.rs`, the project's single most important
//! correctness guard. The f32 preview is an *approximation* — its job is to look
//! right fast — so it is held to a tolerance, not to equality: a ray must land
//! within `max(5e-2 mm, 5e-5 · scene_scale)` of where the f64 reference puts it,
//! the branch structure must match, and a primitive the preview cannot trace must
//! be **rejected with a capability issue**, never swapped for a lookalike (§12.3).
//!
//! Covers the objects the compiler admits: sphere, box, cylinder, extruded line
//! profiles, and revolved profiles (a spherical singlet and an even-asphere);
//! mirror surfaces; beam splitters and dichroics (explicit R/T, constant and
//! tabulated, with every out-of-range policy); and aperture-plus-detector
//! clipping. What is still out of reach — a curved-edge extrusion — must show up
//! in `rejected` rather than as a chord, which the rejection test pins.
#![cfg(feature = "gpu3")]

use oc_core::id::{LayerId, ObjectId, SourceId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::{
    Aperture3, ApertureInteraction, ApertureShape3, Body3, Detector3, DetectorGeometry3,
    DetectorResponse, DetectorTermination, Interpolation, ObjectCommon, OutOfRangePolicy,
    PlanarShape2, Scene3, SolidGeometry3, SpectralScalar, SurfaceGeometry3, SurfaceObject3,
    SurfaceProps3, SurfaceResponse,
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
    let resp = SurfaceResponse::Mirror {
        reflectivity: SpectralScalar::Constant(1.0),
    };
    add_surface(scene, pos, normal, half, resp)
}

/// A square zero-thickness surface with the same response on both faces.
fn add_surface(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    half: f64,
    resp: SurfaceResponse,
) -> ObjectId {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
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

/// A linear wavelength table.
fn table(lambda_um: &[f64], values: &[f64], out_of_range: OutOfRangePolicy) -> SpectralScalar {
    SpectralScalar::Tabulated {
        lambda_um: lambda_um.to_vec(),
        values: values.to_vec(),
        interpolation: Interpolation::Linear,
        out_of_range,
    }
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
        measure: oc_core::scene3::DetectorMeasure::Power,
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
    let at_d: Vec<_> = origins_dirs
        .iter()
        .map(|&(o, d)| (o, d, LAMBDA_D))
        .collect();
    rays_at(&at_d)
}

/// [`rays`] with a wavelength per ray: `(origin, dir, lambda_um)`.
fn rays_at(rays: &[(Vec3, Vec3, f64)]) -> (Vec<EmittedRay3>, Vec<LaunchRay3>) {
    let mut emitted = Vec::new();
    let mut launch = Vec::new();
    for &(o, d, lambda_um) in rays {
        let ray = Ray3::new(o, d).unwrap();
        emitted.push(EmittedRay3 {
            ray,
            lambda_um,
            flux: 1.0,
            source: SourceId(0),
        });
        launch.push(LaunchRay3 {
            ray,
            lambda_um,
            flux: 1.0,
            source: None,
            pol: None,
        });
    }
    (emitted, launch)
}

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        flux_eps: 1e-9,
        max_depth: 32,
        // The preview carries no polarization state: it averages s and p at
        // each surface and hands on unpolarized light. So the reference runs
        // the same model here. Comparing a polarization-aware f64 trace against
        // an unpolarized f32 one would measure the difference between two
        // physical models and call it a precision drift.
        //
        // The difference is not small where it applies: a coated plate, whose
        // front face polarizes the beam before its back face transmits it,
        // moves by 2e-2 of a ray's flux between the two models, which is ten
        // times the grid tolerance this suite is checking.
        // `oc-trace/tests/polarization3.rs` measures the gap deliberately.
        polarization: false,
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

/// A biconvex revolved singlet, authored as the generators do: two conic faces
/// and two rim lines (mirrors `geom3.rs::singlet_profile`).
fn revolved_singlet(
    scene: &mut Scene3,
    r1: f64,
    r2: f64,
    thickness: f64,
    half_ap: f64,
    nd: f64,
) -> ObjectId {
    let (zf, zr) = (-thickness / 2.0, thickness / 2.0);
    let sag = |c: f64, ha: f64| c * ha * ha / (1.0 + (1.0 - c * c * ha * ha).max(0.0).sqrt());
    let (c1, c2) = (1.0 / r1, 1.0 / r2);
    let ftip = zf + sag(c1, half_ap);
    let rtip = zr + sag(c2, half_ap);
    let mk_conic = |scene: &mut Scene3, vx: f64, c: f64| ProfileEdge2 {
        patch: scene.alloc_patch_id(),
        curve: Curve2::Conic(ConicParams2 {
            vertex: Vec2::new(vx, 0.0),
            axis: Vec2::X,
            c,
            k: 0.0,
            half_ap,
        }),
    };
    let e0 = mk_conic(scene, zf, c1);
    let e1 = ProfileEdge2 {
        patch: scene.alloc_patch_id(),
        curve: Curve2::Line {
            a: Vec2::new(ftip, half_ap),
            b: Vec2::new(rtip, half_ap),
        },
    };
    let e2 = mk_conic(scene, zr, c2);
    let e3 = ProfileEdge2 {
        patch: scene.alloc_patch_id(),
        curve: Curve2::Line {
            a: Vec2::new(rtip, -half_ap),
            b: Vec2::new(ftip, -half_ap),
        },
    };
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::IDENTITY;
    scene.bodies.push(Body3 {
        common,
        geometry: SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 {
                edges: vec![e0, e1, e2, e3],
            },
            construction: ProfileConstruction3::RevolveLocalX,
        }),
        material: glass(nd),
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    id
}

#[test]
fn a_revolved_singlet_matches_f64_for_skew_rays() {
    // The canonical optical primitive: a spherical biconvex singlet, its conic
    // surfaces of revolution refracting skew rays to within the §15.5 envelope.
    let mut scene = Scene3::new();
    revolved_singlet(&mut scene, 20.0, -20.0, 4.0, 8.0, 1.5);
    scene.validate().unwrap();

    let rays = [
        (Vec3::new(-25.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(-25.0, 3.0, 0.0), Vec3::X),
        (Vec3::new(-25.0, 0.0, 3.0), Vec3::X),
        (Vec3::new(-25.0, 2.5, -2.5), Vec3::X),
        (
            Vec3::new(-25.0, 4.0, 1.0),
            Vec3::new(1.0, -0.02, 0.0).normalize(),
        ),
    ];
    assert_landing_matches(&scene, &rays);
}

#[test]
fn a_revolved_asphere_matches_f64_for_skew_rays() {
    // The even-asphere path: a conic front with A4/A6 terms, solved by the
    // safeguarded Newton iteration, must still land within the envelope.
    let mut scene = Scene3::new();
    let half_ap = 8.0;
    let c1 = 1.0 / 25.0;
    let sag = |c: f64, ha: f64, a4: f64, a6: f64| {
        let base = c * ha * ha / (1.0 + (1.0 - c * c * ha * ha).max(0.0).sqrt());
        base + a4 * ha.powi(4) + a6 * ha.powi(6)
    };
    let (a4, a6) = (1e-5, 1e-8);
    let ftip = -2.0 + sag(c1, half_ap, a4, a6);
    let rtip =
        2.0 + (-1.0 / 25.0) * half_ap * half_ap / (1.0 + (1.0 - half_ap * half_ap / 625.0).sqrt());
    let mut coeffs = [0.0; 8];
    coeffs[0] = a4;
    coeffs[1] = a6;
    let edges = vec![
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::EvenAsphere {
                base: ConicParams2 {
                    vertex: Vec2::new(-2.0, 0.0),
                    axis: Vec2::X,
                    c: c1,
                    k: 0.0,
                    half_ap,
                },
                coeffs,
            },
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(ftip, half_ap),
                b: Vec2::new(rtip, half_ap),
            },
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Conic(ConicParams2 {
                vertex: Vec2::new(2.0, 0.0),
                axis: Vec2::X,
                c: -1.0 / 25.0,
                k: 0.0,
                half_ap,
            }),
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(rtip, -half_ap),
                b: Vec2::new(ftip, -half_ap),
            },
        },
    ];
    let id = scene.alloc_object_id();
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

    let rays = [
        (Vec3::new(-25.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(-25.0, 3.0, 0.0), Vec3::X),
        (Vec3::new(-25.0, 2.0, 3.0), Vec3::X),
    ];
    assert_landing_matches(&scene, &rays);
}

#[test]
fn a_curved_edge_extrusion_is_rejected_not_approximated() {
    let mut scene = Scene3::new();
    // An extrusion with a conic edge: the ruled curved side surface is the one
    // profile case still to land, so it must reject rather than draw a chord.
    let id = scene.alloc_object_id();
    let edges = vec![
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Conic(ConicParams2 {
                vertex: Vec2::new(-1.0, 0.0),
                axis: Vec2::X,
                c: 0.05,
                k: 0.0,
                half_ap: 3.0,
            }),
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(-1.0, 3.0),
                b: Vec2::new(1.0, 3.0),
            },
        },
        ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: Vec2::new(1.0, 3.0),
                b: Vec2::new(1.0, 0.0),
            },
        },
    ];
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::IDENTITY;
    scene.bodies.push(Body3 {
        common,
        geometry: SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            construction: ProfileConstruction3::ExtrudeLocalZ {
                z_min_mm: -4.0,
                z_max_mm: 4.0,
            },
        }),
        material: glass(1.5),
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    scene.validate().unwrap();

    let gpu = GpuScene3::build(&scene, BuildOpts::default());
    assert_eq!(
        gpu.rejected.len(),
        1,
        "the curved extrusion should be rejected"
    );
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

/// A round face's pixels read the same irradiance in both tiers, rim pixels
/// included.
///
/// The preview divides by the covered areas the scene model computes, not by
/// a copy of the geometry of its own, and computes them only when asked. So
/// the thing to hold is that asking gives the reference's numbers: a grid of
/// rays across a disc, kept off the pixel edges so neither tier's rounding
/// picks the pixel, lands in whole pixels and in pixels the rim cuts.
#[test]
fn a_round_face_reads_the_same_irradiance_in_both_tiers() {
    let mut scene = Scene3::new();
    let det = add_rect_detector(&mut scene, Vec3::new(20.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [6, 6]);
    scene.detectors[0].geometry = DetectorGeometry3::Disc { radius_mm: 3.0 };
    scene.validate().expect("valid");

    let offsets: Vec<f64> = (0..6).map(|i| i as f64 + 0.37 - 3.0).collect();
    let origins_dirs: Vec<(Vec3, Vec3)> = offsets
        .iter()
        .flat_map(|&y| offsets.iter().map(move |&z| (y, z)))
        .filter(|(y, z)| y * y + z * z < 8.5)
        .map(|(y, z)| (Vec3::new(0.0, y, z), Vec3::X))
        .collect();
    let (emitted, launch) = rays(&origins_dirs);

    let reference = trace3(&scene, &launch, &cfg());
    let f64_det = reference
        .detectors
        .iter()
        .find(|d| d.detector.0 == det.0)
        .expect("f64 detector");
    let gpu = GpuScene3::build_with_rays(
        &scene,
        &emitted,
        BuildOpts {
            ghosts: false,
            flux_eps: 1e-9,
            max_ghost: 0,
        },
    );
    let preview = cpu3::trace(&gpu, 32);
    let f32_det = preview.detector(det.0).expect("f32 detector");

    let e64 = f64_det.irradiance().expect("an area detector");
    let e32 = f32_det.density();
    assert_eq!(e64.len(), e32.len());
    let rim = f64_det
        .covered_area_mm2
        .iter()
        .zip(&e64)
        .filter(|(a, e)| **e > 0.0 && **a < 0.999)
        .count();
    assert!(rim >= 4, "rays reach rim pixels: {rim}");
    for (i, (a, b)) in e64.iter().zip(&e32).enumerate() {
        assert!(
            (*a as f32 - *b).abs() <= 1e-5 * a.abs().max(1.0) as f32,
            "pixel {i}: f64 {a} vs f32 {b}"
        );
    }
}

/// The preview bins directions the same way the reference does.
///
/// Irradiance and power come off the same face histogram the two tiers have
/// always agreed on, so they need nothing new. Intensity does not: it is a
/// second histogram over the arrival direction, written twice, once in f64
/// and once in f32. Two copies of a ring layout is exactly the drift this
/// suite exists to catch, so a fan of known angles goes in and both sides
/// have to put it in the same rings.
#[test]
fn the_preview_bins_arrival_directions_like_the_reference() {
    let mut scene = Scene3::new();
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(0.0, 0.0, 0.0),
        -Vec3::X,
        40.0,
        40.0,
        [6, 4],
    );
    scene
        .detectors
        .iter_mut()
        .for_each(|d| {
            d.measure = oc_core::scene3::DetectorMeasure::Intensity {
                half_angle_deg: 30.0,
                bins: [6, 4],
            }
        });
    scene.validate().unwrap();

    // A fan spread over polar angle and azimuth, so every ring and every
    // sector gets something, with a few rays past the cone to land in the
    // outside count.
    //
    // The angles are deliberately off every boundary. A ray arriving exactly
    // on a ring or sector edge is a knife-edge that the two tiers can round
    // to opposite sides of, which is a property of binning and not a defect
    // in either of them; a test that sat on one would be measuring float
    // width rather than agreement.
    let mut rays = Vec::new();
    for i in 0..14 {
        let theta = (i as f64 + 0.5) * 2.5f64.to_radians();
        for j in 0..9 {
            let phi = (13.0 + j as f64 * 37.0).to_radians();
            let dir = Vec3::new(
                theta.cos(),
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
            )
            .normalize();
            rays.push((dir * -50.0, dir));
        }
    }

    let (emitted, launch) = rays_at(&rays.iter().map(|(o, d)| (*o, *d, 1.0)).collect::<Vec<_>>());
    let f64_res = trace3(
        &scene,
        &launch,
        &TraceConfig3 {
            ghosts: false,
            ..cfg()
        },
    );
    let gpu = GpuScene3::build_with_rays(&scene, &emitted, opts(false));
    let f32_res = cpu3::trace(&gpu, 32);

    let a = &f64_res.detectors[0];
    let b = f32_res.detector(det.0).expect("f32 detector");
    assert_eq!(
        a.direction_flux.len(),
        b.direction.len(),
        "the two grids are the same size"
    );
    assert!(
        a.direction_flux.iter().filter(|f| **f > 0.0).count() > 15,
        "the fan reaches most of the grid"
    );
    assert!(a.outside_cone_flux > 0.0, "and some of it misses the cone");

    for (i, (x, y)) in a.direction_flux.iter().zip(&b.direction).enumerate() {
        assert!(
            flux_close(*x, *y),
            "direction cell {i}: f64 {x} vs f32 {y}"
        );
    }
    assert!(
        flux_close(a.outside_cone_flux, b.outside),
        "outside the cone: f64 {} vs f32 {}",
        a.outside_cone_flux,
        b.outside
    );

    // And the intensity itself, which is the flux over solid angles the scene
    // model handed to both.
    let ia = a.radiant_intensity().expect("f64 intensity");
    let ib = b.radiant_intensity().expect("f32 intensity");
    for (i, (x, y)) in ia.iter().zip(&ib).enumerate() {
        assert!(
            (x - *y as f64).abs() <= 1e-4 * x.abs().max(1.0),
            "intensity cell {i}: f64 {x} vs f32 {y}"
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

#[test]
fn a_thin_lens_matches_f64_for_skew_rays() {
    // The ideal-lens slope law runs in both engines; the f32 preview must land
    // where the f64 reference does, forward and backward.
    let mut scene = Scene3::new();
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(Vec3::ZERO, Vec3::X);
    let resp = SurfaceResponse::ThinLens {
        focal_length_mm: 50.0,
    };
    scene.surfaces.push(SurfaceObject3 {
        common,
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Rectangle {
                half_width_mm: 20.0,
                half_height_mm: 20.0,
            },
        },
        response_front: resp.clone(),
        response_back: resp,
    });
    scene.validate().unwrap();

    let rays = [
        (Vec3::new(-30.0, 0.001, 0.0), Vec3::X),
        (Vec3::new(-30.0, 5.0, 0.0), Vec3::X),
        (Vec3::new(-30.0, 0.0, -8.0), Vec3::X),
        (Vec3::new(-30.0, 7.0, -4.0), Vec3::X),
        (Vec3::new(30.0, 6.0, -2.0), Vec3::NEG_X),
    ];
    assert_landing_matches(&scene, &rays);
}

// ---------------------------------------------------------------------------
// Beam splitters and dichroics: explicit R/T, constant and tabulated
// ---------------------------------------------------------------------------

/// Build options matching [`cfg`], with the ghost switch set.
fn opts(ghosts: bool) -> BuildOpts {
    BuildOpts {
        ghosts,
        flux_eps: 1e-9,
        max_ghost: if ghosts { 8 } else { 0 },
    }
}

/// Relative agreement to 1e-5 (so absolute 1e-7 for fluxes below 1e-2).
fn flux_close(a: f64, b: f32) -> bool {
    let b = b as f64;
    (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1e-2)
}

/// Trace one ray through both engines and compare the whole branch tree.
///
/// Both engines keep a LIFO branch stack and push a split's children in the
/// same order, so one ray's segment lists line up one to one: same count, same
/// ghost and TIR flags, endpoints within the landing tolerance (an escape by
/// direction only), flux within [`flux_close`]. Returns the f32 segments.
fn assert_tree_matches(scene: &Scene3, ray: (Vec3, Vec3, f64), ghosts: bool) -> Vec<SegOut3> {
    let (emitted, launch) = rays_at(&[ray]);
    let f64_res = trace3(scene, &launch, &TraceConfig3 { ghosts, ..cfg() });
    let gpu = GpuScene3::build_with_rays(scene, &emitted, opts(ghosts));
    assert!(
        gpu.rejected.is_empty(),
        "unexpected rejects: {:?}",
        gpu.rejected
    );
    let tol = pos_tol(gpu.norm.scale);
    let f32_segs = cpu3::trace_segments(&gpu, 32).swap_remove(0);
    let f64_segs = &f64_res.segments;

    assert_eq!(
        f64_segs.len(),
        f32_segs.len(),
        "segment count for {ray:?}, ghosts {ghosts}:\nf64 {f64_segs:?}\nf32 {f32_segs:?}"
    );
    const ESCAPE: f64 = 1e3;
    for (i, (a, b)) in f64_segs.iter().zip(&f32_segs).enumerate() {
        let at = format!("seg {i} of {ray:?}, ghosts {ghosts}");
        assert_eq!(a.flags.ghost, b.is_ghost(), "{at}: ghost flag");
        assert_eq!(a.flags.tir, b.is_tir(), "{at}: TIR flag");
        let d0 = (a.a - b.a).length();
        assert!(d0 <= tol, "{at}: |Δstart| = {d0} mm > {tol}");
        let escapes = (a.b - a.a).length() > ESCAPE;
        assert_eq!(escapes, (b.b - b.a).length() > ESCAPE, "{at}: escape");
        if escapes {
            let ang = ((a.b - a.a).normalize() - (b.b - b.a).normalize()).length();
            assert!(ang < 1e-3, "{at}: escape direction |Δ| = {ang}");
        } else {
            let d1 = (a.b - b.b).length();
            assert!(d1 <= tol, "{at}: |Δend| = {d1} mm > {tol}");
        }
        assert!(
            flux_close(a.flux_end, b.flux),
            "{at}: flux f64 {} vs f32 {}",
            a.flux_end,
            b.flux
        );
    }
    f32_segs
}

/// [`assert_tree_matches`] without the flux comparison: same branch structure
/// and the same geometry, but each segment's flux left to the caller.
///
/// Only the thin-film coating needs this. Its preview reads a baked grid rather
/// than running the transfer matrix, so its flux carries interpolation error
/// that `flux_close` would reject — while the *paths* stay exact, because where
/// a ray goes does not depend on how much of it goes there.
fn assert_tree_geometry_matches(
    scene: &Scene3,
    ray: (Vec3, Vec3, f64),
    ghosts: bool,
) -> Vec<SegOut3> {
    let (emitted, launch) = rays_at(&[ray]);
    let f64_res = trace3(scene, &launch, &TraceConfig3 { ghosts, ..cfg() });
    let gpu = GpuScene3::build_with_rays(scene, &emitted, opts(ghosts));
    assert!(
        gpu.rejected.is_empty(),
        "unexpected rejects: {:?}",
        gpu.rejected
    );
    let tol = pos_tol(gpu.norm.scale);
    let f32_segs = cpu3::trace_segments(&gpu, 32).swap_remove(0);
    let f64_segs = &f64_res.segments;
    assert_eq!(
        f64_segs.len(),
        f32_segs.len(),
        "segment count for {ray:?}:
f64 {f64_segs:?}
f32 {f32_segs:?}"
    );
    for (i, (a, b)) in f64_segs.iter().zip(&f32_segs).enumerate() {
        let at = format!("seg {i} of {ray:?}");
        assert_eq!(a.flags.ghost, b.is_ghost(), "{at}: ghost flag");
        let d0 = (a.a - b.a).length();
        assert!(d0 <= tol, "{at}: |Δstart| = {d0} mm > {tol}");
        if (a.b - a.a).length() <= 1e3 {
            let d1 = (a.b - b.b).length();
            assert!(d1 <= tol, "{at}: |Δend| = {d1} mm > {tol}");
        }
    }
    f32_segs
}

/// Every listed detector's readout, f64 against f32, for one launch set: equal
/// hit counts, total and per-cell flux within [`flux_close`].
fn assert_detectors_match(
    scene: &Scene3,
    dets: &[ObjectId],
    rays: &[(Vec3, Vec3, f64)],
    ghosts: bool,
) {
    let (emitted, launch) = rays_at(rays);
    let f64_res = trace3(scene, &launch, &TraceConfig3 { ghosts, ..cfg() });
    let gpu = GpuScene3::build_with_rays(scene, &emitted, opts(ghosts));
    assert!(
        gpu.rejected.is_empty(),
        "unexpected rejects: {:?}",
        gpu.rejected
    );
    let f32_res = cpu3::trace(&gpu, 32);
    for det in dets {
        let a = f64_res
            .detectors
            .iter()
            .find(|d| d.detector.0 == det.0)
            .expect("f64 detector");
        let b = f32_res.detector(det.0).expect("f32 detector");
        assert_eq!(a.hits, b.hits, "detector {}: hit count", det.0);
        assert!(
            flux_close(a.total_incident, b.total),
            "detector {}: total f64 {} vs f32 {}",
            det.0,
            a.total_incident,
            b.total
        );
        assert_eq!(a.incident_flux.len(), b.incident.len(), "cell count");
        for (i, (x, y)) in a.incident_flux.iter().zip(&b.incident).enumerate() {
            assert!(
                flux_close(*x, *y),
                "detector {} cell {i}: f64 {x} vs f32 {y}",
                det.0
            );
        }
    }
}

/// A 45° fold at the origin (a +X beam reflects toward +Y) with a terminating
/// detector on each arm. Returns `(through, folded)`.
fn fold_with_two_arms(scene: &mut Scene3, resp: SurfaceResponse) -> (ObjectId, ObjectId) {
    let normal = Vec3::new(-1.0, 1.0, 0.0).normalize();
    add_surface(scene, Vec3::ZERO, normal, 10.0, resp);
    // Three bins a side keep every test ray's hit clear of a cell edge.
    let through = add_rect_detector(scene, Vec3::new(20.0, 0.0, 0.0), -Vec3::X, 8.0, 8.0, [3, 3]);
    let folded = add_rect_detector(scene, Vec3::new(0.0, 20.0, 0.0), -Vec3::Y, 8.0, 8.0, [3, 3]);
    (through, folded)
}

/// Parallel +X rays meeting the fold at a spread of heights, at one wavelength.
fn fold_rays(lambda_um: f64) -> Vec<(Vec3, Vec3, f64)> {
    [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (-1.5, 1.0)]
        .iter()
        .map(|&(y, z)| (Vec3::new(-15.0, y, z), Vec3::X, lambda_um))
        .collect()
}

fn splitter_50_50() -> SurfaceResponse {
    SurfaceResponse::ExplicitRt {
        reflectance: SpectralScalar::Constant(0.5),
        transmittance: SpectralScalar::Constant(0.5),
    }
}

#[test]
fn a_beam_splitter_keeps_its_reflected_arm_with_ghosts_off() {
    // A splitter's reflection is a primary path in tracer3, not a Fresnel ghost:
    // with ghosts off it must still be traced, and with ghosts on it must not be
    // flagged as one.
    let mut scene = Scene3::new();
    let (through, folded) = fold_with_two_arms(&mut scene, splitter_50_50());
    scene.validate().unwrap();

    let rays = fold_rays(LAMBDA_D);
    for ghosts in [false, true] {
        for &ray in &rays {
            // In, then through, then folded: the reflection is pushed first.
            let segs = assert_tree_matches(&scene, ray, ghosts);
            assert_eq!(segs.len(), 3, "{ray:?}");
            assert!(segs.iter().all(|s| !s.is_ghost()), "{ray:?}");
            assert!(
                segs[1].b.x > 19.0 && segs[2].b.y > 19.0,
                "{ray:?}: {segs:?}"
            );
        }
        assert_detectors_match(&scene, &[through, folded], &rays, ghosts);
    }

    // Each arm receives half of every ray.
    let (emitted, _) = rays_at(&rays);
    let gpu = GpuScene3::build_with_rays(&scene, &emitted, opts(false));
    let res = cpu3::trace(&gpu, 32);
    for det in [through, folded] {
        let d = res.detector(det.0).unwrap();
        assert_eq!(d.hits, rays.len() as u64);
        assert!((d.total - 0.5 * rays.len() as f32).abs() < 1e-6);
    }
}

#[test]
fn a_ghost_that_meets_a_splitter_stays_a_ghost_on_both_arms() {
    // A glass plate ahead of the splitter, ghosts on. The plate's double-bounce
    // ghosts travel forward into the splitter, whose children keep the parent's
    // ghost state and ghost depth (tracer3 spawns them with `w.ghost`).
    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(1.0, 8.0, 8.0),
        },
        glass(1.5),
    );
    scene.bodies[0].common.transform = Isometry3::from_pos(Vec3::new(-8.0, 0.0, 0.0));
    let (through, folded) = fold_with_two_arms(&mut scene, splitter_50_50());
    scene.validate().unwrap();

    let ray = (Vec3::new(-20.0, 0.5, 0.25), Vec3::X, LAMBDA_D);
    let segs = assert_tree_matches(&scene, ray, true);
    let ghost_on = |through_arm: bool| {
        segs.iter().any(|s| {
            s.is_ghost()
                && if through_arm {
                    s.b.x > 19.0
                } else {
                    s.b.y > 19.0
                }
        })
    };
    assert!(ghost_on(true), "no ghost on the through arm: {segs:?}");
    assert!(ghost_on(false), "no ghost on the folded arm: {segs:?}");
    assert_detectors_match(&scene, &[through, folded], &[ray], true);
}

#[test]
fn a_dichroic_splits_by_wavelength_like_f64() {
    // A long-pass dichroic: reflects below 0.53 µm, transmits above 0.57 µm,
    // a linear edge between, the end values held outside the table. R + T = 1.
    let edge = [0.40, 0.53, 0.57, 0.80];
    let dichroic = SurfaceResponse::ExplicitRt {
        reflectance: table(&edge, &[1.0, 1.0, 0.0, 0.0], OutOfRangePolicy::Clamp),
        transmittance: table(&edge, &[0.0, 0.0, 1.0, 1.0], OutOfRangePolicy::Clamp),
    };
    let mut scene = Scene3::new();
    let (through, folded) = fold_with_two_arms(&mut scene, dichroic);
    scene.validate().unwrap();

    // (wavelength, segments per ray): one arm off the edge, past either end of
    // the table included; both arms on the edge.
    let cases = [
        (0.35, 2),
        (0.45, 2),
        (0.52, 2),
        (0.545, 3),
        (0.55, 3),
        (0.56, 3),
        (0.65, 2),
        (0.95, 2),
    ];
    let mut all = Vec::new();
    for (lambda_um, n_segs) in cases {
        let rays = fold_rays(lambda_um);
        for &ray in &rays {
            let segs = assert_tree_matches(&scene, ray, false);
            assert_eq!(segs.len(), n_segs, "{ray:?}");
            if n_segs == 2 {
                let folded_arm = segs[1].b.y > 19.0;
                assert_eq!(folded_arm, lambda_um < 0.53, "{ray:?}: {segs:?}");
            }
        }
        assert_detectors_match(&scene, &[through, folded], &rays, false);
        all.extend(rays);
    }
    assert_detectors_match(&scene, &[through, folded], &all, false);
}

#[test]
fn out_of_range_splitter_policies_shape_the_branches_like_f64() {
    // Reflectance tabulated over 0.45..0.65 µm only, transmittance constant.
    // Outside the band tracer3 ends the branch under Error (no value), keeps
    // only the transmitted arm under Zero, and holds the end value under Clamp.
    let band = [0.45, 0.65];
    for (policy, outside) in [
        (OutOfRangePolicy::Error, 1),
        (OutOfRangePolicy::Zero, 2),
        (OutOfRangePolicy::Clamp, 3),
    ] {
        let resp = SurfaceResponse::ExplicitRt {
            reflectance: table(&band, &[0.3, 0.35], policy),
            transmittance: SpectralScalar::Constant(0.6),
        };
        let mut scene = Scene3::new();
        let (through, folded) = fold_with_two_arms(&mut scene, resp);
        scene.validate().unwrap();

        let mut all = Vec::new();
        for lambda_um in [0.4, 0.55, 0.8] {
            let inside = (band[0]..=band[1]).contains(&lambda_um);
            let rays = fold_rays(lambda_um);
            for &ray in &rays {
                let segs = assert_tree_matches(&scene, ray, false);
                let want = if inside { 3 } else { outside };
                assert_eq!(segs.len(), want, "{policy:?}, {ray:?}");
            }
            all.extend(rays);
        }
        assert_detectors_match(&scene, &[through, folded], &all, false);
    }
}

#[test]
fn out_of_range_mirror_and_window_tables_match_f64() {
    let band = [0.45, 0.65];
    // Mirror: no value (Error) ends the branch; Zero reflects nothing, and the
    // flux threshold drops that child, as tracer3's spawn does.
    for policy in [OutOfRangePolicy::Error, OutOfRangePolicy::Zero] {
        let mirror = SurfaceResponse::Mirror {
            reflectivity: table(&band, &[0.9, 0.95], policy),
        };
        let mut scene = Scene3::new();
        let (_, folded) = fold_with_two_arms(&mut scene, mirror);
        scene.validate().unwrap();

        let mut all = Vec::new();
        for lambda_um in [0.4, 0.55, 0.8] {
            let inside = (band[0]..=band[1]).contains(&lambda_um);
            let rays = fold_rays(lambda_um);
            for &ray in &rays {
                let segs = assert_tree_matches(&scene, ray, false);
                assert_eq!(segs.len(), if inside { 2 } else { 1 }, "{policy:?}");
            }
            all.extend(rays);
        }
        assert_detectors_match(&scene, &[folded], &all, false);
    }

    // Window (scalar dielectric): with no AR value tracer3 scales the Fresnel
    // reflection by 1 rather than dropping the branch. Both faces of a
    // zero-thickness surface see one index, so the beam passes whole either
    // way; the tree and the flux must still agree, ghosts on and off. (The
    // fallback itself is unit-tested in cpu3.)
    let window = SurfaceResponse::DielectricFresnel {
        ar_residual: table(&band, &[0.01, 0.02], OutOfRangePolicy::Error),
    };
    let mut scene = Scene3::new();
    add_surface(&mut scene, Vec3::ZERO, -Vec3::X, 10.0, window);
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(20.0, 0.0, 0.0),
        -Vec3::X,
        8.0,
        8.0,
        [3, 3],
    );
    scene.validate().unwrap();
    for ghosts in [false, true] {
        let mut all = Vec::new();
        for lambda_um in [0.4, 0.55, 0.8] {
            let rays = fold_rays(lambda_um);
            for &ray in &rays {
                let segs = assert_tree_matches(&scene, ray, ghosts);
                assert_eq!(segs.len(), 2, "{ray:?}");
                assert!((segs[1].flux - 1.0).abs() < 1e-6, "{ray:?}: {segs:?}");
            }
            all.extend(rays);
        }
        assert_detectors_match(&scene, &[det], &all, ghosts);
    }
}

#[test]
fn a_tabulated_mirror_matches_f64() {
    // A metal-like reflectivity curve, weaker in the blue, held at its ends.
    let reflectivity = table(
        &[0.40, 0.50, 0.60, 0.75],
        &[0.80, 0.93, 0.96, 0.975],
        OutOfRangePolicy::Clamp,
    );
    let mut scene = Scene3::new();
    let (_, folded) = fold_with_two_arms(&mut scene, SurfaceResponse::Mirror { reflectivity });
    scene.validate().unwrap();

    let mut all = Vec::new();
    for lambda_um in [0.38, 0.45, 0.5, LAMBDA_D, 0.7, 0.9] {
        let rays = fold_rays(lambda_um);
        for &ray in &rays {
            let segs = assert_tree_matches(&scene, ray, false);
            assert_eq!(segs.len(), 2, "{ray:?}");
        }
        all.extend(rays);
    }
    assert_detectors_match(&scene, &[folded], &all, false);

    // The curve itself: at 0.45 µm the mirror returns the interpolated 0.865.
    let rays = fold_rays(0.45);
    let (emitted, _) = rays_at(&rays);
    let gpu = GpuScene3::build_with_rays(&scene, &emitted, opts(false));
    let total = cpu3::trace(&gpu, 32).detector(folded.0).unwrap().total;
    let want = 0.865 * rays.len() as f32;
    assert!((total - want).abs() < 1e-5, "{total} vs {want}");
}

/// A thin-film coating in the preview against the exact transfer matrix.
///
/// This is the one response the preview does **not** evaluate exactly. Running
/// a multilayer matrix per ray in a shader is not on, so the stack is baked onto
/// a grid in wavelength and `sin²θ` when the scene compiles, and the preview
/// reads that grid bilinearly. The CPU mirror and the GPU share the grid, so
/// those two stay identical; what drifts is the preview against the f64
/// reference, and only where the curve is steep.
///
/// So this test holds the branch **structure** exactly — same segments, same
/// geometry, a ray goes the same ways — and holds the flux to a bound stated
/// here rather than to `flux_close`. The bound is the honest cost of the grid,
/// and the numbers the UI shows never come from this path.
#[test]
fn a_thin_film_coating_matches_f64_within_the_grid_tolerance() {
    let design = oc_core::thinfilm::CoatingDesign::long_pass(550.0, 45.0);
    let coating = oc_core::thinfilm::plate_coating(
        design,
        oc_core::thinfilm::materials::bk7(),
        oc_core::thinfilm::ExitFace::Bare,
        1.0,
    );
    let resp = SurfaceResponse::Coating { coating };
    let mut scene = Scene3::new();
    let (through, folded) = fold_with_two_arms(&mut scene, resp);
    scene.validate().unwrap();

    let mut worst: f64 = 0.0;
    // Across the band: deep in the stopband, on the edge where the curve is
    // steepest and the grid works hardest, and out in the passband ripple.
    for lambda_um in [0.40, 0.46, 0.50, 0.53, 0.55, 0.57, 0.62, 0.70] {
        let rays = fold_rays(lambda_um);
        for &ray in &rays {
            // Structure is exact: both arms exist, in the same order, landing in
            // the same places.
            let segs = assert_tree_geometry_matches(&scene, ray, false);
            assert_eq!(segs.len(), 3, "both arms at {lambda_um}: {segs:?}");
        }
        let (emitted, launch) = rays_at(&rays);
        let f64_res = trace3(
            &scene,
            &launch,
            &TraceConfig3 {
                ghosts: false,
                ..cfg()
            },
        );
        let gpu = GpuScene3::build_with_rays(&scene, &emitted, opts(false));
        assert!(gpu.rejected.is_empty(), "rejects: {:?}", gpu.rejected);
        let f32_res = cpu3::trace(&gpu, 32);
        for det in [through, folded] {
            let a = f64_res
                .detectors
                .iter()
                .find(|d| d.detector.0 == det.0)
                .expect("f64 detector");
            let b = f32_res.detector(det.0).expect("f32 detector");
            assert_eq!(a.hits, b.hits, "hit count at {lambda_um}");
            worst = worst.max((a.total_incident - b.total as f64).abs() / rays.len() as f64);
        }
    }
    // Stated as a fraction of one ray's launched flux. Measured at 7.6e-4 with
    // the grid baked at the scene's own wavelengths, so the bound leaves room
    // for the angle axis and nothing else.
    assert!(
        worst < 2e-3,
        "the baked grid drifts further from the matrix than documented: {worst}"
    );
    eprintln!("worst per-ray flux drift from the coating grid: {worst:.2e}");
}

/// An AR-coated lens face reflects ghosts in the preview, as it does in the
/// reference.
///
/// A body face with a coating reads its reflection off the baked grid, and
/// that branch used to be pushed as a primary path for every stack, so an AR
/// coat on a lens drew its reflections at full strength with ghosts switched
/// off. The branch structure has to match the reference in both states: no
/// reflection at all with ghosts off, and a ghost-flagged one with them on.
#[test]
fn an_ar_coated_lens_reflects_ghosts_like_f64() {
    use oc_core::build3::{BodyFace, SingletGeom};
    use oc_core::thinfilm::{CoatingDesign, DesignKind};

    let mut scene = Scene3::new();
    let lens = scene.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
    let ar = CoatingDesign {
        kind: DesignKind::BroadbandAr,
        lambda_nm: 550.0,
        aoi_deg: 0.0,
        periods: 1,
        n_high: 2.35,
        n_low: 1.38,
    };
    for face in [BodyFace::Front, BodyFace::Back] {
        assert!(scene.set_body_face_coating(lens.0, face, Some(ar)));
    }
    scene.validate().unwrap();

    for lambda_um in [0.48, 0.55, 0.62] {
        for y in [0.0, 1.5, -3.0] {
            let ray = (Vec3::new(-30.0, y, 0.5), Vec3::X, lambda_um);
            let off = assert_tree_geometry_matches(&scene, ray, false);
            assert!(
                off.iter().all(|g| !g.is_ghost()),
                "ghosts off draws no reflection: {off:?}"
            );
            // In, through the glass, out: nothing folds back.
            assert_eq!(off.len(), 3, "{ray:?}: {off:?}");

            let on = assert_tree_geometry_matches(&scene, ray, true);
            assert!(
                on.iter().any(|g| g.is_ghost()),
                "ghosts on draws the reflection, as a ghost: {on:?}"
            );
            assert_eq!(
                on.iter().filter(|g| !g.is_ghost()).count(),
                3,
                "and the primary path is the same either way"
            );
        }

        // A ray inside the glass past the critical angle: total internal
        // reflection at a coated face is the ray itself, a primary path in
        // both tiers with the switch either way, never a stray reflection.
        let t = 60f64.to_radians();
        let inside = (Vec3::ZERO, Vec3::new(t.cos(), t.sin(), 0.0), lambda_um);
        for ghosts in [false, true] {
            let segs = assert_tree_geometry_matches(&scene, inside, ghosts);
            assert!(segs[1].is_tir() && !segs[1].is_ghost(), "{segs:?}");
        }
    }
}

/// A coated plate with real thickness, previewed against the exact tracer.
///
/// The plate is a body: the coating is on one face and plain glass on the other,
/// so this is the first scene where the preview has to know *which face* it hit.
/// Get that wrong and the film lands on the far face, which still splits the
/// beam and still looks plausible, so the check is that the geometry matches
/// the reference exactly: same segment count, same refracted path through the
/// glass, same exit point.
#[test]
fn a_coated_plate_body_matches_f64() {
    let mut scene = Scene3::new();
    scene
        .add_plate_dichroic(Vec3::new(0.0, 0.0, 0.0), 10.0, 3.0, 135.0, 550.0, true)
        .expect("the plate is valid");
    let through = add_rect_detector(
        &mut scene,
        Vec3::new(20.0, 0.0, 0.0),
        -Vec3::X,
        8.0,
        8.0,
        [3, 3],
    );
    let folded = add_rect_detector(
        &mut scene,
        Vec3::new(0.0, 20.0, 0.0),
        -Vec3::Y,
        8.0,
        8.0,
        [3, 3],
    );
    scene.validate().unwrap();

    let mut worst: f64 = 0.0;
    for lambda_um in [0.45, 0.50, 0.55, 0.60, 0.65] {
        let rays: Vec<(Vec3, Vec3, f64)> = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (-1.5, 1.0)]
            .iter()
            .map(|&(y, z)| (Vec3::new(-15.0, y, z), Vec3::X, lambda_um))
            .collect();
        for &ray in &rays {
            let segs = assert_tree_geometry_matches(&scene, ray, false);
            assert!(
                segs.len() >= 3,
                "reflect, cross the glass, and carry on: {segs:?}"
            );
        }
        let (emitted, launch) = rays_at(&rays);
        let f64_res = trace3(
            &scene,
            &launch,
            &TraceConfig3 {
                ghosts: false,
                ..cfg()
            },
        );
        let gpu = GpuScene3::build_with_rays(&scene, &emitted, opts(false));
        assert!(gpu.rejected.is_empty(), "rejects: {:?}", gpu.rejected);
        let f32_res = cpu3::trace(&gpu, 32);
        for det in [through, folded] {
            let a = f64_res
                .detectors
                .iter()
                .find(|d| d.detector.0 == det.0)
                .expect("f64 detector");
            let b = f32_res.detector(det.0).expect("f32 detector");
            assert_eq!(a.hits, b.hits, "hit count at {lambda_um}");
            worst = worst.max((a.total_incident - b.total as f64).abs() / rays.len() as f64);
        }
    }
    assert!(
        worst < 2e-3,
        "the coating grid drifts further than documented: {worst}"
    );
}

// ---------------------------------------------------------------------------
// Fresnel lens: a hundred annular facets and risers, each an exact face
// ---------------------------------------------------------------------------

fn fresnel_scene(lens: &oc_core::fresnel_lens::FresnelLens1) -> Scene3 {
    let mut scene = Scene3::new();
    let (nd, vd) = oc_core::fresnel_lens::PMMA_ND_VD;
    scene
        .add_fresnel_lens(Vec3::ZERO, 0.0, lens, Material::AbbePoint { nd, vd })
        .expect("the lens builds");
    scene
}

#[test]
fn a_fresnel_lens_matches_f64_away_from_its_risers() {
    // Rays through the middle of a zone, skew ones included: the f32 facet must
    // be the f64 facet, shifted and clipped to its annulus.
    let lens = oc_core::fresnel_lens::FresnelLens1::default();
    let scene = fresnel_scene(&lens);
    let gpu = GpuScene3::build(&scene, BuildOpts::default());
    assert!(gpu.rejected.is_empty(), "rejected: {:?}", gpu.rejected);
    let mut rays = Vec::new();
    for j in 0..24 {
        let h = j as f64 + 0.5;
        let phi = 0.9 * j as f64;
        rays.push((
            Vec3::new(-30.0, h * phi.cos(), h * phi.sin()),
            Vec3::new(1.0, 0.004, -0.003).normalize(),
        ));
    }
    assert_landing_matches(&scene, &rays);
}

/// The fraction of a uniformly filled beam whose f32 path matches the f64 one
/// (same hit count, every hit within the landing tolerance, same exit
/// direction). A ray that lands within the medium probe of a riser can read the
/// wrong side in f32; that is the whole of the disagreement.
fn fresnel_agreement(lens: &oc_core::fresnel_lens::FresnelLens1, n: usize) -> f64 {
    let scene = fresnel_scene(lens);
    let tol = pos_tol(GpuScene3::build(&scene, BuildOpts::default()).norm.scale);
    let rc = lens.clear_radius_mm;
    let mut agree = 0usize;
    for i in 0..n {
        let r = 0.99 * rc * ((i as f64 + 0.5) / n as f64).sqrt();
        let phi = i as f64 * 2.399_963_229_728_653;
        let od = (
            Vec3::new(-30.0, r * phi.cos(), r * phi.sin()),
            Vec3::new(1.0, 0.002, 0.001).normalize(),
        );
        let (emitted, launch) = rays(std::slice::from_ref(&od));
        let (a_hits, a_dir) = f64_landing(&trace3(&scene, &launch, &cfg()), 0, 1);
        let gpu = GpuScene3::build_with_rays(
            &scene,
            &emitted,
            BuildOpts {
                ghosts: false,
                flux_eps: 1e-9,
                max_ghost: 0,
            },
        );
        let (b_hits, b_dir) = f32_landing(&cpu3::trace_segments(&gpu, 32)[0]);
        let same = a_hits.len() == b_hits.len()
            && a_hits
                .iter()
                .zip(&b_hits)
                .all(|(a, b)| (*a - *b).length() <= tol)
            && (a_dir - b_dir).length() < 1e-3;
        agree += same as usize;
    }
    agree as f64 / n as f64
}

#[test]
fn a_fresnel_lens_matches_f64_across_a_filled_beam() {
    for lens in [
        oc_core::fresnel_lens::FresnelLens1::default(),
        oc_core::fresnel_lens::FresnelLens1 {
            draft_deg: 5.0,
            facing: oc_core::fresnel_lens::FresnelFacing::MinusX,
            ..oc_core::fresnel_lens::FresnelLens1::default()
        },
    ] {
        let agree = fresnel_agreement(&lens, 600);
        eprintln!("f32 vs f64 path agreement, {lens:?}: {agree:.4}");
        assert!(agree >= 0.97, "only {agree} of the beam agrees");
    }
}

/// Per-ray cost of the f32 CPU mirror through a Fresnel lens against a plain
/// singlet of the same size. A measurement, not a check: run with
/// `cargo test --release -p oc-gpu --features gpu3 --test cpu3_vs_f64 fresnel_preview_cost -- --ignored --nocapture`.
#[test]
#[ignore]
fn fresnel_preview_cost() {
    let big = oc_core::fresnel_lens::FresnelLens1 {
        radius_mm: 100.0,
        clear_radius_mm: 50.0,
        grooves: oc_core::fresnel_lens::FresnelGrooves::ConstantPitch { pitch_mm: 0.5 },
        ..oc_core::fresnel_lens::FresnelLens1::default()
    };
    for lens in [oc_core::fresnel_lens::FresnelLens1::default(), big] {
        let rc = lens.clear_radius_mm;
        let n = 20_000;
        let od: Vec<(Vec3, Vec3)> = (0..n)
            .map(|i| {
                let r = 0.98 * rc * ((i as f64 + 0.5) / n as f64).sqrt();
                let phi = i as f64 * 2.399_963_229_728_653;
                (Vec3::new(-50.0, r * phi.cos(), r * phi.sin()), Vec3::X)
            })
            .collect();
        let (emitted, _) = rays(&od);
        let mut singlet = Scene3::new();
        singlet.add_singlet(
            Vec3::ZERO,
            oc_core::build3::SingletGeom {
                r1_mm: f64::INFINITY,
                r2_mm: -lens.radius_mm,
                thickness_mm: 2.0 + lens.radius_mm - (lens.radius_mm.powi(2) - rc * rc).sqrt(),
                semi_aperture_mm: rc,
                conic1: 0.0,
                conic2: 0.0,
            },
            1.4917,
            57.4,
        );
        for (name, scene) in [("fresnel", fresnel_scene(&lens)), ("singlet", singlet)] {
            let gpu = GpuScene3::build_with_rays(&scene, &emitted, BuildOpts::default());
            assert!(gpu.rejected.is_empty());
            let t0 = std::time::Instant::now();
            let segs = cpu3::trace_segments(&gpu, 32);
            let dt = t0.elapsed().as_secs_f64();
            let faces = gpu.bodies[0].revolve.as_ref().map_or(0, |r| r.faces.len());
            let contour = gpu.bodies[0]
                .revolve
                .as_ref()
                .map_or(0, |r| r.contour.len());
            eprintln!(
                "{name:8} R {} clear {rc}: {faces} faces, {contour} contour points, {:.2} us/ray ({} rays)",
                lens.radius_mm,
                1e6 * dt / n as f64,
                segs.len()
            );
        }
    }
}
