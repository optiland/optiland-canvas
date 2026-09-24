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
//! It covers the analytic bodies (sphere, box, cylinder), extruded and revolved
//! bodies, a tilted mirror, an aperture-plus-detector, beam splitters (ghosts off
//! and on), and dichroics and other tabulated responses. The struct layout
//! itself is checked without a device in `shader3_layout`. The suite skips
//! cleanly when no wgpu adapter is present.
#![cfg(feature = "gpu3")]

use oc_core::id::{LayerId, ObjectId};
use oc_core::material::Material;
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
        measure: oc_core::scene3::DetectorMeasure::Power,
    });
}

fn emitted(origins_dirs: &[(Vec3, Vec3)]) -> Vec<EmittedRay3> {
    emitted_at(&at_d(origins_dirs))
}

/// [`emitted`] with a wavelength per ray: `(origin, dir, lambda_um)`.
fn emitted_at(rays: &[(Vec3, Vec3, f64)]) -> Vec<EmittedRay3> {
    rays.iter()
        .map(|&(o, d, lambda_um)| EmittedRay3 {
            ray: Ray3::new(o, d).unwrap(),
            lambda_um,
            flux: 1.0,
            source: oc_core::id::SourceId(0),
        })
        .collect()
}

fn at_d(origins_dirs: &[(Vec3, Vec3)]) -> Vec<(Vec3, Vec3, f64)> {
    origins_dirs
        .iter()
        .map(|&(o, d)| (o, d, LAMBDA_D))
        .collect()
}

/// The §15.5 GPU-vs-CPU landing tolerance for a scene of the given scale.
fn tol(scale: f64) -> f64 {
    (1e-2_f64).max(1e-5 * scale)
}

/// Compare the shader's per-ray segments to the CPU oracle's, within §15.5.
fn assert_gpu_matches_cpu(tracer: &Gpu3Tracer, scene: &Scene3, origins_dirs: &[(Vec3, Vec3)]) {
    assert_gpu_matches_cpu_at(tracer, scene, &at_d(origins_dirs), opts());
}

/// [`assert_gpu_matches_cpu`] with a wavelength per ray and explicit build
/// options. The segment budget grows to hold the CPU mirror's longest tree (a
/// ghost tree can outgrow [`MAX_SEGS`]), and the shader must fit it.
fn assert_gpu_matches_cpu_at(
    tracer: &Gpu3Tracer,
    scene: &Scene3,
    rays: &[(Vec3, Vec3, f64)],
    build: BuildOpts,
) {
    let launch = emitted_at(rays);
    let gpu_scene = GpuScene3::build_with_rays(scene, &launch, build);
    assert!(
        gpu_scene.rejected.is_empty(),
        "unexpected rejects: {:?}",
        gpu_scene.rejected
    );
    let pos_tol = tol(gpu_scene.norm.scale);

    let cpu = cpu3::trace_segments(&gpu_scene, MAX_DEPTH);
    let longest = cpu.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let budget = longest.next_power_of_two().max(MAX_SEGS);
    let full = tracer.trace_checked(&gpu_scene, MAX_DEPTH, budget);
    assert!(
        full.is_complete(),
        "budget {budget}: {} rays over budget, {} lost a branch to the stack",
        full.budget_truncated,
        full.stack_truncated
    );
    let gpu = full.segments;

    assert_eq!(cpu.len(), gpu.len(), "ray count");
    for (ri, (csegs, gsegs)) in cpu.iter().zip(&gpu).enumerate() {
        let od = (rays[ri].0, rays[ri].1);
        assert_eq!(
            csegs.len(),
            gsegs.len(),
            "ray {ri} ({:?}): segment count CPU {} vs GPU {}",
            rays[ri],
            csegs.len(),
            gsegs.len()
        );
        for (i, (c, g)) in csegs.iter().zip(gsegs).enumerate() {
            assert_eq!(c.flags, g.flags, "ray {ri} seg {i}: flags");
            assert_eq!(c.lambda_um, g.lambda_um, "ray {ri} seg {i}: wavelength");
            assert_seg(c, g, pos_tol, ri, i, od);
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

/// With ghosts on, a ray splits at every interface, so its tree can outgrow the
/// shader's per-ray segment budget. The shader must trace the CPU mirror's tree
/// when it fits, and must report a budget that is too small instead of returning
/// a thinner tree.
#[test]
fn ghost_trees_match_cpu_and_a_short_budget_is_reported() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };

    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Sphere { radius_mm: 8.0 },
        glass(1.5),
    );
    scene.validate().unwrap();
    let rays = [
        (Vec3::new(-20.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(-20.0, 3.0, 0.0), Vec3::X),
        (Vec3::new(-20.0, 2.5, -2.5), Vec3::X),
    ];
    let ghosts = BuildOpts {
        ghosts: true,
        flux_eps: 1e-9,
        max_ghost: 2,
    };
    let gpu_scene = GpuScene3::build_with_rays(&scene, &emitted(&rays), ghosts);
    assert!(gpu_scene.rejected.is_empty(), "{:?}", gpu_scene.rejected);

    let cpu = cpu3::trace_segments(&gpu_scene, MAX_DEPTH);
    let longest = cpu.iter().map(Vec::len).max().unwrap();
    assert!(
        longest > 2,
        "the fixture must branch; longest ray has {longest} segments"
    );

    // Too small a budget is reported, never silently clipped.
    let short = tracer.trace_checked(&gpu_scene, MAX_DEPTH, 2);
    assert!(
        short.budget_truncated > 0,
        "a 2-segment budget must be reported"
    );
    assert_eq!(short.stack_truncated, 0);

    // A budget that holds the tree reproduces the CPU mirror, ghosts included.
    let budget = (longest as u32).next_power_of_two().max(MAX_SEGS);
    let full = tracer.trace_checked(&gpu_scene, MAX_DEPTH, budget);
    assert!(
        full.is_complete(),
        "budget {budget}: {} rays over budget",
        full.budget_truncated
    );
    let pos_tol = tol(gpu_scene.norm.scale);
    assert_eq!(cpu.len(), full.segments.len(), "ray count");
    for (ri, (csegs, gsegs)) in cpu.iter().zip(&full.segments).enumerate() {
        assert_eq!(csegs.len(), gsegs.len(), "ray {ri}: segment count");
        for (i, (c, g)) in csegs.iter().zip(gsegs).enumerate() {
            assert_eq!(c.flags, g.flags, "ray {ri} seg {i}: flags");
            assert_seg(c, g, pos_tol, ri, i, rays[ri]);
        }
    }
    assert!(
        full.segments
            .iter()
            .flatten()
            .any(|s| s.flags & cpu3::SEG_GHOST != 0),
        "the fixture must trace at least one ghost branch"
    );
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

// ---------------------------------------------------------------------------
// Beam splitters, dichroics, tabulated responses
// ---------------------------------------------------------------------------

fn ghost_opts(ghosts: bool) -> BuildOpts {
    BuildOpts {
        ghosts,
        flux_eps: 1e-9,
        max_ghost: if ghosts { 8 } else { 0 },
    }
}

/// A square zero-thickness surface with the same response on both faces.
fn add_surface(scene: &mut Scene3, pos: Vec3, normal: Vec3, half: f64, resp: SurfaceResponse) {
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
}

fn table(lambda_um: &[f64], values: &[f64], out_of_range: OutOfRangePolicy) -> SpectralScalar {
    SpectralScalar::Tabulated {
        lambda_um: lambda_um.to_vec(),
        values: values.to_vec(),
        interpolation: Interpolation::Linear,
        out_of_range,
    }
}

/// A 45° fold at the origin (+X reflects toward +Y) with a terminating detector
/// on each arm, as in `cpu3_vs_f64`. An off-path polygon absorber is added
/// first, so the fold's response tables do not start at `verts[0]`.
fn fold_with_two_arms(scene: &mut Scene3, resp: SurfaceResponse) {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(Vec3::new(0.0, -40.0, 0.0), Vec3::Y);
    scene.surfaces.push(SurfaceObject3 {
        common,
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Polygon {
                vertices: vec![
                    Vec2::new(-3.0, -3.0),
                    Vec2::new(3.0, -3.0),
                    Vec2::new(0.0, 3.0),
                ],
            },
        },
        response_front: SurfaceResponse::Absorber,
        response_back: SurfaceResponse::Absorber,
    });
    add_surface(
        scene,
        Vec3::ZERO,
        Vec3::new(-1.0, 1.0, 0.0).normalize(),
        10.0,
        resp,
    );
    add_rect_detector(scene, Vec3::new(20.0, 0.0, 0.0), -Vec3::X, 8.0, 8.0, [3, 3]);
    add_rect_detector(scene, Vec3::new(0.0, 20.0, 0.0), -Vec3::Y, 8.0, 8.0, [3, 3]);
}

/// Parallel +X rays meeting the fold at a spread of heights, at each wavelength.
fn fold_rays(lambdas: &[f64]) -> Vec<(Vec3, Vec3, f64)> {
    let heights = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (-1.5, 1.0)];
    lambdas
        .iter()
        .flat_map(|&l| {
            heights
                .iter()
                .map(move |&(y, z)| (Vec3::new(-15.0, y, z), Vec3::X, l))
        })
        .collect()
}

fn splitter_50_50() -> SurfaceResponse {
    SurfaceResponse::ExplicitRt {
        reflectance: SpectralScalar::Constant(0.5),
        transmittance: SpectralScalar::Constant(0.5),
    }
}

/// An AR coat's reflection is a ghost in the shader too, on a lens face and on
/// a zero-thickness window alike.
///
/// Both coating paths in the shader used to push the reflection as a primary
/// ray for every stack, so an AR-coated lens drew its reflections with ghosts
/// switched off. The shader reads the stray bit the preview compiles from the
/// coating; this holds it to the CPU mirror with the switch both ways.
#[test]
fn an_ar_coat_reflects_ghosts_like_cpu() {
    use oc_core::build3::{BodyFace, SingletGeom};
    use oc_core::thinfilm::{materials, plate_coating, CoatingDesign, DesignKind, ExitFace};

    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let ar = CoatingDesign {
        kind: DesignKind::BroadbandAr,
        lambda_nm: 550.0,
        aoi_deg: 0.0,
        periods: 1,
        n_high: 2.35,
        n_low: 1.38,
    };

    // A coated lens: the body-face path.
    let mut lens = Scene3::new();
    let id = lens.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
    for face in [BodyFace::Front, BodyFace::Back] {
        assert!(lens.set_body_face_coating(id.0, face, Some(ar)));
    }
    lens.validate().unwrap();

    // An AR-coated window: the zero-thickness surface path.
    let mut window = Scene3::new();
    let coating = plate_coating(ar, materials::bk7(), ExitFace::Bare, 1.0);
    add_surface(
        &mut window,
        Vec3::ZERO,
        -Vec3::X,
        10.0,
        SurfaceResponse::Coating { coating },
    );
    window.validate().unwrap();

    let rays: Vec<(Vec3, Vec3, f64)> = [0.0, 1.5, -3.0]
        .iter()
        .flat_map(|&y| {
            [0.48, 0.55, 0.62]
                .into_iter()
                .map(move |l| (Vec3::new(-30.0, y, 0.5), Vec3::X, l))
        })
        .collect();
    // The lens also gets one ray inside the glass past the critical angle,
    // where the coated face reflects totally and the reflection is the ray.
    let t = 60f64.to_radians();
    let mut lens_rays = rays.clone();
    lens_rays.push((Vec3::ZERO, Vec3::new(t.cos(), t.sin(), 0.0), 0.55));
    for (scene, rays) in [(&lens, &lens_rays), (&window, &rays)] {
        for ghosts in [false, true] {
            assert_gpu_matches_cpu_at(&tracer, scene, rays, ghost_opts(ghosts));
        }
    }
}

#[test]
fn a_beam_splitter_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    // Both arms are primary paths: traced with ghosts off, unflagged with
    // ghosts on.
    let mut scene = Scene3::new();
    fold_with_two_arms(&mut scene, splitter_50_50());
    scene.validate().unwrap();
    for ghosts in [false, true] {
        assert_gpu_matches_cpu_at(&tracer, &scene, &fold_rays(&[LAMBDA_D]), ghost_opts(ghosts));
    }

    // A glass plate ahead of the splitter, ghosts on: the plate's forward
    // ghosts must stay ghosts on both arms.
    let mut scene = Scene3::new();
    push_body(
        &mut scene,
        SolidGeometry3::Box {
            half_extents_mm: Vec3::new(1.0, 8.0, 8.0),
        },
        glass(1.5),
    );
    scene.bodies[0].common.transform = Isometry3::from_pos(Vec3::new(-8.0, 0.0, 0.0));
    fold_with_two_arms(&mut scene, splitter_50_50());
    scene.validate().unwrap();
    let rays = [(Vec3::new(-20.0, 0.5, 0.25), Vec3::X, LAMBDA_D)];
    assert_gpu_matches_cpu_at(&tracer, &scene, &rays, ghost_opts(true));
}

#[test]
fn dichroics_and_tabulated_responses_match_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let lambdas = [0.35, 0.45, 0.52, 0.545, 0.55, 0.56, 0.65, 0.95];

    // A long-pass dichroic with a linear edge, end values held.
    let edge = [0.40, 0.53, 0.57, 0.80];
    let dichroic = SurfaceResponse::ExplicitRt {
        reflectance: table(&edge, &[1.0, 1.0, 0.0, 0.0], OutOfRangePolicy::Clamp),
        transmittance: table(&edge, &[0.0, 0.0, 1.0, 1.0], OutOfRangePolicy::Clamp),
    };
    let mut scene = Scene3::new();
    fold_with_two_arms(&mut scene, dichroic);
    scene.validate().unwrap();
    assert_gpu_matches_cpu_at(&tracer, &scene, &fold_rays(&lambdas), opts());

    // Every out-of-range policy on a splitter's reflectance table.
    for policy in [
        OutOfRangePolicy::Error,
        OutOfRangePolicy::Zero,
        OutOfRangePolicy::Clamp,
    ] {
        let resp = SurfaceResponse::ExplicitRt {
            reflectance: table(&[0.45, 0.65], &[0.3, 0.35], policy),
            transmittance: SpectralScalar::Constant(0.6),
        };
        let mut scene = Scene3::new();
        fold_with_two_arms(&mut scene, resp);
        scene.validate().unwrap();
        assert_gpu_matches_cpu_at(&tracer, &scene, &fold_rays(&lambdas), opts());
    }

    // Tabulated mirrors: no value outside the band (Error), and a held curve.
    for (xs, ys, policy) in [
        (vec![0.45, 0.65], vec![0.9, 0.95], OutOfRangePolicy::Error),
        (
            vec![0.40, 0.50, 0.60, 0.75],
            vec![0.80, 0.93, 0.96, 0.975],
            OutOfRangePolicy::Clamp,
        ),
    ] {
        let mirror = SurfaceResponse::Mirror {
            reflectivity: table(&xs, &ys, policy),
        };
        let mut scene = Scene3::new();
        fold_with_two_arms(&mut scene, mirror);
        scene.validate().unwrap();
        assert_gpu_matches_cpu_at(&tracer, &scene, &fold_rays(&lambdas), opts());
    }

    // A window whose AR table has no value outside its band, ghosts on and off.
    let window = SurfaceResponse::DielectricFresnel {
        ar_residual: table(&[0.45, 0.65], &[0.01, 0.02], OutOfRangePolicy::Error),
    };
    let mut scene = Scene3::new();
    add_surface(&mut scene, Vec3::ZERO, -Vec3::X, 10.0, window);
    add_rect_detector(
        &mut scene,
        Vec3::new(20.0, 0.0, 0.0),
        -Vec3::X,
        8.0,
        8.0,
        [3, 3],
    );
    scene.validate().unwrap();
    for ghosts in [false, true] {
        assert_gpu_matches_cpu_at(&tracer, &scene, &fold_rays(&lambdas), ghost_opts(ghosts));
    }
}

/// A real thin-film coating on the GPU against the CPU mirror.
///
/// This is the only test that runs the shader's coating branch at all. Both
/// sides read the same baked grid, so unlike every other physics comparison
/// here this one is a check on the *indexing*: the shader walks a wavelength
/// axis and a `sin²θ` row by hand, and an off-by-one in either would read a
/// neighbouring column and quietly route light the wrong way.
#[test]
fn a_thin_film_coating_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let design = oc_core::thinfilm::CoatingDesign::long_pass(550.0, 45.0);
    let coating = oc_core::thinfilm::plate_coating(
        design,
        oc_core::thinfilm::materials::bk7(),
        oc_core::thinfilm::ExitFace::Bare,
        1.0,
    );
    let mut scene = Scene3::new();
    fold_with_two_arms(&mut scene, SurfaceResponse::Coating { coating });
    scene.validate().unwrap();
    // Across the edge, where the grid's two axes both matter, and out either
    // side of it where they do not.
    let lambdas = [0.42, 0.50, 0.53, 0.55, 0.57, 0.62, 0.70];
    assert_gpu_matches_cpu_at(&tracer, &scene, &fold_rays(&lambdas), opts());
}

/// A coated plate body on the GPU against the CPU mirror.
///
/// The plate is a solid whose two faces do different things, so the shader has
/// to pick the response by face. Getting that wrong still splits the beam and
/// still looks plausible, which is exactly why it needs a test: the CPU mirror
/// reads the same baked grid, so any difference here is the shader's face
/// indexing and nothing else.
#[test]
fn a_coated_plate_body_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let mut scene = Scene3::new();
    scene
        .add_plate_dichroic(Vec3::new(0.0, 0.0, 0.0), 10.0, 3.0, 135.0, 550.0, true)
        .expect("the plate is valid");
    add_rect_detector(
        &mut scene,
        Vec3::new(20.0, 0.0, 0.0),
        -Vec3::X,
        8.0,
        8.0,
        [3, 3],
    );
    add_rect_detector(
        &mut scene,
        Vec3::new(0.0, 20.0, 0.0),
        -Vec3::Y,
        8.0,
        8.0,
        [3, 3],
    );
    scene.validate().unwrap();
    // Across the edge, where both grid axes matter, and either side of it.
    let lambdas = [0.45, 0.50, 0.53, 0.55, 0.58, 0.62, 0.70];
    assert_gpu_matches_cpu_at(&tracer, &scene, &fold_rays(&lambdas), opts());
}

/// A tube's bore reflects in the shader as it does in the CPU mirror: a mirror
/// bore, a silver one read from the coating grid, and a black end ring, for
/// rays that zig-zag and rays that spiral.
#[test]
fn a_lined_tube_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    for bore in [
        oc_core::tube::BoreFinish::Mirror { reflectivity: 0.9 },
        oc_core::tube::BoreFinish::Metal {
            metal: oc_core::tube::BoreMetal::Silver,
            protected: true,
        },
    ] {
        let mut scene = Scene3::new();
        scene
            .add_tube(
                Vec3::ZERO,
                0.0,
                oc_core::tube::TubeDims {
                    inner_diameter_mm: 10.0,
                    outer_diameter_mm: 12.0,
                    length_mm: 100.0,
                },
                bore,
            )
            .expect("a tube");
        add_rect_detector(&mut scene, Vec3::new(70.0, 0.0, 0.0), -Vec3::X, 40.0, 40.0, [3, 3]);
        scene.validate().unwrap();
        let slope = |deg: f64, z: f64| {
            let t = deg.to_radians();
            (Vec3::new(-60.0, 0.0, z), Vec3::new(t.cos(), t.sin(), 0.0), 0.55)
        };
        let rays = [
            (Vec3::new(-60.0, 0.0, 0.0), Vec3::X, 0.55),
            slope(15.0, 0.0),
            slope(15.0, 3.0),
            slope(22.0, -2.0),
            (Vec3::new(-60.0, 5.5, 0.0), Vec3::X, 0.55),
        ];
        assert_gpu_matches_cpu_at(&tracer, &scene, &rays, opts());
    }
}

// ---------------------------------------------------------------------------
// Fresnel lens: annular sag faces (`inner_r > 0`) in the shader
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
fn a_fresnel_lens_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let lens = oc_core::fresnel_lens::FresnelLens1::default();
    let scene = fresnel_scene(&lens);
    // Mid-zone rays, skew ones included: every facet the shader meets must be
    // the CPU mirror's, clipped to the same annulus.
    let rays: Vec<(Vec3, Vec3)> = (0..24)
        .map(|j| {
            let h = j as f64 + 0.5;
            let phi = 0.9 * j as f64;
            (
                Vec3::new(-30.0, h * phi.cos(), h * phi.sin()),
                Vec3::new(1.0, 0.004, -0.003).normalize(),
            )
        })
        .collect();
    assert_gpu_matches_cpu(&tracer, &scene, &rays);
}

#[test]
fn a_filled_beam_through_a_fresnel_lens_matches_cpu() {
    let Some(tracer) = tracer_or_skip() else {
        return;
    };
    let lens = oc_core::fresnel_lens::FresnelLens1 {
        draft_deg: 4.0,
        ..oc_core::fresnel_lens::FresnelLens1::default()
    };
    let scene = fresnel_scene(&lens);
    let n = 2000;
    let rays: Vec<(Vec3, Vec3, f64)> = (0..n)
        .map(|i| {
            let r = 24.5 * ((i as f64 + 0.5) / n as f64).sqrt();
            let phi = i as f64 * 2.399_963_229_728_653;
            (
                Vec3::new(-30.0, r * phi.cos(), r * phi.sin()),
                Vec3::X,
                oc_core::material::LAMBDA_D,
            )
        })
        .collect();
    let gpu_scene = GpuScene3::build_with_rays(&scene, &emitted_at(&rays), opts());
    assert!(gpu_scene.rejected.is_empty(), "{:?}", gpu_scene.rejected);
    let pos_tol = tol(gpu_scene.norm.scale);
    let cpu = cpu3::trace_segments(&gpu_scene, MAX_DEPTH);
    let full = tracer.trace_checked(&gpu_scene, MAX_DEPTH, MAX_SEGS);
    let gpu = full.segments;
    assert_eq!(cpu.len(), gpu.len());
    let agree = cpu
        .iter()
        .zip(&gpu)
        .filter(|(c, g)| {
            c.len() == g.len()
                && c.iter().zip(g.iter()).all(|(c, g)| {
                    let dir = ((c.b - c.a).normalize() - (g.b - g.a).normalize()).length();
                    (c.a - g.a).length() <= pos_tol && dir < 1e-3
                })
        })
        .count() as f64
        / n as f64;
    eprintln!("GPU vs CPU mirror, filled beam through a Fresnel lens: {agree:.4} agree");
    assert!(agree >= 0.99, "only {agree} of the beam agrees");
}
