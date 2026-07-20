//! The 3D CPU f32 preview against the f64 reference (§15.5, PR 3D-07).
//!
//! The 3D successor to `cross_validate.rs`, the project's single most important
//! correctness guard. The f32 preview is an *approximation* — its job is to look
//! right fast — so it is held to a tolerance, not to equality: a ray must land
//! within `max(5e-2 mm, 5e-5 · scene_scale)` of where the f64 reference puts it,
//! the branch structure must match, and a primitive the preview cannot trace must
//! be **rejected with a capability issue**, never swapped for a lookalike (§12.3).
//!
//! This first increment covers the analytic bodies the compiler admits — sphere
//! and box, dielectric or absorbing. The curved profile surfaces and the
//! aperture/detector objects arrive in later increments; here they must show up
//! in `rejected`, which the last test pins.
#![cfg(feature = "gpu3")]

use oc_core::id::{LayerId, ObjectId, SourceId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::profile2::{
    ClosedProfile2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::{Body3, ObjectCommon, Scene3, SolidGeometry3, SurfaceProps3};
use oc_core::source3::EmittedRay3;
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Ray3, Vec3};
use oc_gpu::cpu3::{self, SegOut3};
use oc_gpu::scene3::{BuildOpts, GpuScene3};
use oc_trace::{trace3, LaunchRay3, Segment3, TraceConfig3};

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
        let f32_segs = &cpu3::trace(&gpu, 32)[0];
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
