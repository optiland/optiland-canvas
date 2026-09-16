//! The 3D WGSL shader against the 3D CPU f32 mirror (PR 3D-08, §15.5, §12.2).
//!
//! The CPU f32 path is the oracle here, not the f64 reference: this test isolates
//! *shader* disagreement from *precision* disagreement, which `cpu3_vs_f64` covers
//! separately. Comparing the shader straight to f64 would blur the two and make a
//! real WGSL bug look like rounding.
//!
//! What this suite owes:
//!
//! - GPU vs CPU f32 ray landing within `max(1e-2 mm, 1e-5 * scene_scale)` (§15.5);
//! - the same branch structure — segment count, TIR/ghost flags, termination;
//! - the same capability classification (rejects) on both backends;
//! - Rust and WGSL agree on the buffer layout — the compile-time asserts in
//!   `buffers3` cover Rust; a divergent field offset would make the trace diverge,
//!   which the landing checks below catch.
//!
//! Part 1 covers the analytic bodies (sphere, box, cylinder), a tilted mirror, and
//! an aperture-plus-detector. Extruded and revolved bodies join as their shader
//! parts land. The suite skips cleanly when no wgpu adapter is present.
#![cfg(feature = "gpu3")]

use oc_core::id::{LayerId, ObjectId};
use oc_core::material::Material;
use oc_core::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
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
use oc_gpu::gpu3::Gpu3Tracer;
use oc_gpu::scene3::{BuildOpts, GpuScene3};

const MAX_DEPTH: u32 = 32;
const MAX_SEGS: u32 = 32;
const LAMBDA_D: f64 = oc_core::material::LAMBDA_D;

fn opts() -> BuildOpts {
    BuildOpts {
        ghosts: false,
        flux_eps: 1e-9,
        max_ghost: 0,
    }
}

fn glass(nd: f64) -> Material {
    Material::AbbePoint { nd, vd: 64.0 }
}

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

fn facing(pos: Vec3, normal: Vec3) -> Isometry3 {
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, normal.normalize());
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap()
}

fn add_mirror(scene: &mut Scene3, pos: Vec3, normal: Vec3, half: f64) {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    let resp = SurfaceResponse::Mirror {
        reflectivity: SpectralScalar::Constant(0.9),
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
}

fn add_circular_aperture(scene: &mut Scene3, pos: Vec3, normal: Vec3, opening: f64, mask: f64) {
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
}

fn add_rect_detector(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    hw: f64,
    hh: f64,
    bins: [usize; 2],
) {
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
}

fn emitted(origins_dirs: &[(Vec3, Vec3)]) -> Vec<EmittedRay3> {
    origins_dirs
        .iter()
        .map(|(o, d)| EmittedRay3 {
            ray: Ray3::new(*o, *d).unwrap(),
            lambda_um: LAMBDA_D,
            flux: 1.0,
            source: oc_core::id::SourceId(0),
        })
        .collect()
}

/// The §15.5 GPU-vs-CPU landing tolerance for a scene of the given scale.
fn tol(scale: f64) -> f64 {
    (1e-2_f64).max(1e-5 * scale)
}

/// Compare the shader's per-ray segments to the CPU oracle's, within §15.5.
fn assert_gpu_matches_cpu(tracer: &Gpu3Tracer, scene: &Scene3, origins_dirs: &[(Vec3, Vec3)]) {
    let launch = emitted(origins_dirs);
    let gpu_scene = GpuScene3::build_with_rays(scene, &launch, opts());
    assert!(
        gpu_scene.rejected.is_empty(),
        "unexpected rejects: {:?}",
        gpu_scene.rejected
    );
    let pos_tol = tol(gpu_scene.norm.scale);

    let cpu = cpu3::trace_segments(&gpu_scene, MAX_DEPTH);
    let gpu = tracer.trace(&gpu_scene, MAX_DEPTH, MAX_SEGS);

    assert_eq!(cpu.len(), gpu.len(), "ray count");
    for (ri, (csegs, gsegs)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            csegs.len() as u32 <= MAX_SEGS,
            "ray {ri}: CPU produced {} segments, over the GPU budget",
            csegs.len()
        );
        assert_eq!(
            csegs.len(),
            gsegs.len(),
            "ray {ri} ({:?}): segment count CPU {} vs GPU {}",
            origins_dirs[ri],
            csegs.len(),
            gsegs.len()
        );
        for (i, (c, g)) in csegs.iter().zip(gsegs).enumerate() {
            assert_eq!(c.flags, g.flags, "ray {ri} seg {i}: flags");
            assert_seg(c, g, pos_tol, ri, i, origins_dirs[ri]);
        }
    }
}

fn assert_seg(c: &SegOut3, g: &SegOut3, pos_tol: f64, ri: usize, i: usize, od: (Vec3, Vec3)) {
    const ESCAPE: f64 = 100.0;
    let da = (c.a - g.a).length();
    assert!(
        da <= pos_tol,
        "ray {ri} seg {i} ({od:?}): |Δstart| = {da} mm > {pos_tol}"
    );

    // Direction agrees for every segment; the endpoint agrees for finite ones.
    // An escape endpoint is thousands of mm away, where f32 spreads past the
    // landing tolerance, so it is checked by direction only.
    let cd = (c.b - c.a).normalize();
    let gd = (g.b - g.a).normalize();
    let ang = (cd - gd).length();
    assert!(ang < 1e-3, "ray {ri} seg {i} ({od:?}): |Δdir| = {ang}");

    if (c.b - c.a).length() < ESCAPE {
        let db = (c.b - g.b).length();
        assert!(
            db <= pos_tol,
            "ray {ri} seg {i} ({od:?}): |Δend| = {db} mm > {pos_tol}"
        );
    }
    assert!(
        (c.flux - g.flux).abs() < 2e-3,
        "ray {ri} seg {i} ({od:?}): flux CPU {} vs GPU {}",
        c.flux,
        g.flux
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn tracer_or_skip() -> Option<Gpu3Tracer> {
    match Gpu3Tracer::new() {
        Some(t) => {
            eprintln!(
                "GPU3 backend: {:?} '{}'",
                t.adapter_info.backend, t.adapter_info.name
            );
            Some(t)
        }
        None => {
            eprintln!("no wgpu adapter; skipping 3D GPU parity");
            None
        }
    }
}

#[test]
fn sphere_box_cylinder_match_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };

    // Sphere.
    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Sphere { radius_mm: 8.0 },
        glass(1.5),
    );
    scene.validate().unwrap();
    assert_gpu_matches_cpu(
        &tracer,
        &scene,
        &[
            (Vec3::new(-20.0, 0.0, 0.0), Vec3::X),
            (Vec3::new(-20.0, 3.0, 0.0), Vec3::X),
            (Vec3::new(-20.0, 0.0, 3.0), Vec3::X),
            (Vec3::new(-20.0, 2.5, -2.5), Vec3::X),
        ],
    );

    // Box.
    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(3.0, 10.0, 10.0),
        },
        glass(1.6),
    );
    scene.validate().unwrap();
    assert_gpu_matches_cpu(
        &tracer,
        &scene,
        &[
            (Vec3::new(-20.0, 0.0, 0.0), Vec3::X),
            (
                Vec3::new(-20.0, 2.0, 0.0),
                Vec3::new(1.0, 0.1, 0.0).normalize(),
            ),
            (
                Vec3::new(-20.0, 1.0, -1.0),
                Vec3::new(1.0, 0.1, -0.1).normalize(),
            ),
        ],
    );

    // Cylinder along Z.
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
    assert_gpu_matches_cpu(
        &tracer,
        &scene,
        &[
            (Vec3::new(-20.0, 0.0, 0.0), Vec3::X),
            (Vec3::new(-20.0, 3.0, 0.0), Vec3::X),
            (
                Vec3::new(-20.0, 2.0, 4.0),
                Vec3::new(1.0, 0.0, 0.1).normalize(),
            ),
        ],
    );
}

#[test]
fn an_extruded_prism_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let mut scene = Scene3::new();
    // A right-angle prism extruded along Z.
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
    assert_gpu_matches_cpu(
        &tracer,
        &scene,
        &[
            (Vec3::new(-20.0, 0.5, 0.0), Vec3::X),
            (
                Vec3::new(-20.0, 1.0, 2.0),
                Vec3::new(1.0, 0.0, 0.05).normalize(),
            ),
            (
                Vec3::new(-20.0, -1.0, 0.0),
                Vec3::new(1.0, 0.05, 0.0).normalize(),
            ),
        ],
    );
}

/// A biconvex revolved singlet: two conic faces and two rim lines (mirrors
/// `cpu3_vs_f64`'s `revolved_singlet`).
fn revolved_singlet(scene: &mut Scene3, r1: f64, r2: f64, thickness: f64, half_ap: f64, nd: f64) {
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
}

#[test]
fn a_revolved_singlet_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let mut scene = Scene3::new();
    revolved_singlet(&mut scene, 20.0, -20.0, 4.0, 8.0, 1.5);
    scene.validate().unwrap();
    assert_gpu_matches_cpu(
        &tracer,
        &scene,
        &[
            (Vec3::new(-25.0, 0.0, 0.0), Vec3::X),
            (Vec3::new(-25.0, 3.0, 0.0), Vec3::X),
            (Vec3::new(-25.0, 0.0, 3.0), Vec3::X),
            (Vec3::new(-25.0, 2.5, -2.5), Vec3::X),
        ],
    );
}

#[test]
fn a_revolved_asphere_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
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
    assert_gpu_matches_cpu(
        &tracer,
        &scene,
        &[
            (Vec3::new(-25.0, 0.0, 0.0), Vec3::X),
            (Vec3::new(-25.0, 3.0, 0.0), Vec3::X),
            (Vec3::new(-25.0, 2.0, 3.0), Vec3::X),
        ],
    );
}

#[test]
fn a_tilted_mirror_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let mut scene = Scene3::new();
    add_mirror(
        &mut scene,
        Vec3::ZERO,
        Vec3::new(-1.0, 1.0, 0.0).normalize(),
        10.0,
    );
    scene.validate().unwrap();
    assert_gpu_matches_cpu(
        &tracer,
        &scene,
        &[
            (Vec3::new(-15.0, 0.0, 0.0), Vec3::X),
            (Vec3::new(-15.0, 2.0, 0.0), Vec3::X),
            (Vec3::new(-15.0, 0.0, 2.0), Vec3::X),
        ],
    );
}

#[test]
fn an_aperture_and_detector_match_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let mut scene = Scene3::new();
    add_circular_aperture(&mut scene, Vec3::new(0.0, 0.0, 0.0), Vec3::X, 3.0, 20.0);
    add_rect_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        6.0,
        6.0,
        [6, 6],
    );
    scene.validate().unwrap();

    let mut rays = Vec::new();
    for iy in -3..=3 {
        for iz in -3..=3 {
            rays.push((Vec3::new(-10.0, iy as f64 * 1.2, iz as f64 * 1.2), Vec3::X));
        }
    }
    assert_gpu_matches_cpu(&tracer, &scene, &rays);
}
