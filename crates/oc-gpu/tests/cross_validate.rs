//! Golden-scene cross-validation (spec §10.1) — the highest correctness risk in
//! the project. The interactive f32 path (CPU **and** GPU) must land rays in the
//! same place as the exact f64 reference tracer.
//!
//! For each golden scene we launch the same rays through three tracers and
//! compare where each ray's *exit* ray crosses a measurement plane:
//! - `oc-trace` (f64, reference, ghosts off) — truth.
//! - `oc-gpu` CPU f32 — the fallback backend.
//! - `oc-gpu` GPU f32 (wgpu) — the interactive backend (skipped if no adapter).

use oc_core::{
    ConicParams, Detector, Edge, Element, ElementId, EmittedRay, Isometry2, Material, Ray, Scene,
    Source, SourceKind, Spectrum, Vec2, LAMBDA_D,
};
#[cfg(feature = "gpu")]
use oc_gpu::gpu::GpuTracer;
use oc_gpu::{scene::GpuScene, trace_cpu};
use oc_trace::{trace, TraceConfig};

/// Where the supporting line of segment `(a,b)` crosses the plane `x = plane`.
fn cross_at_x(a: Vec2, b: Vec2, plane: f64) -> Option<f64> {
    let dx = b.x - a.x;
    if dx.abs() < 1e-9 {
        return None;
    }
    let t = (plane - a.x) / dx;
    Some(a.y + t * (b.y - a.y))
}

/// The exit segment is the one whose start is furthest along +x.
fn exit_seg_f64(segs: &[oc_trace::Segment]) -> Option<(Vec2, Vec2)> {
    segs.iter()
        .max_by(|p, q| p.a.x.partial_cmp(&q.a.x).unwrap())
        .map(|s| (s.a, s.b))
}

/// Reference crossing for a single launch ray, traced individually (ghosts off).
fn ref_crossing(golden: &Scene, er: &EmittedRay, plane: f64) -> Option<f64> {
    let mut s = Scene::new();
    s.env = golden.env.clone();
    s.catalog = golden.catalog.clone();
    for e in &golden.elements {
        s.elements.push(e.clone());
    }
    let angle = er.ray.dir.y.atan2(er.ray.dir.x);
    s.sources.push(Source {
        kind: SourceKind::Point { half_angle: 0.0 },
        transform: Isometry2::new(er.ray.origin, angle),
        spectrum: Spectrum::Monochromatic(er.lambda_um),
        flux: 1.0,
        rays: 1,
    });
    let cfg = TraceConfig {
        ghosts: false,
        ..TraceConfig::default()
    };
    let segs = trace(&s, &cfg).segments;
    let (a, b) = exit_seg_f64(&segs)?;
    cross_at_x(a, b, plane)
}

fn interactive_crossing(segs: &[oc_gpu::Segment], plane: f64) -> Option<f64> {
    let s = segs
        .iter()
        .max_by(|p, q| p.a.x.partial_cmp(&q.a.x).unwrap())?;
    cross_at_x(
        Vec2::new(s.a.x as f64, s.a.y as f64),
        Vec2::new(s.b.x as f64, s.b.y as f64),
        plane,
    )
}

struct Golden {
    name: &'static str,
    scene: Scene,
    plane: f64,
    max_depth: u32,
}

fn singlet_scene() -> Golden {
    let (r, tc, ha) = (50.0, 4.0, 10.0);
    let c = 1.0 / r;
    let front = ConicParams {
        vertex: Vec2::new(-tc / 2.0, 0.0),
        axis: Vec2::new(1.0, 0.0),
        c,
        k: 0.0,
        half_ap: ha,
    };
    let rear = ConicParams {
        vertex: Vec2::new(tc / 2.0, 0.0),
        axis: Vec2::new(-1.0, 0.0),
        c,
        k: 0.0,
        half_ap: ha,
    };
    let sag = c * ha * ha / (1.0 + (1.0 - c * c * ha * ha).sqrt());
    let (ftip, rtip) = (-tc / 2.0 + sag, tc / 2.0 - sag);
    let lens = Element::new(
        ElementId(0),
        vec![
            Edge::Conic(front),
            Edge::Line {
                a: Vec2::new(ftip, ha),
                b: Vec2::new(rtip, ha),
            },
            Edge::Conic(rear),
            Edge::Line {
                a: Vec2::new(rtip, -ha),
                b: Vec2::new(ftip, -ha),
            },
        ],
        Material::AbbePoint { nd: 1.5, vd: 60.0 },
    );
    let mut scene = Scene::new();
    scene.add_element(lens);
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 6.0 },
        transform: Isometry2::from_pos(Vec2::new(-60.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 33,
    });
    Golden {
        name: "singlet-lens",
        scene,
        plane: 45.0,
        max_depth: 6,
    }
}

fn mirror_scene() -> Golden {
    let (radius, ha) = (8.0, 4.0);
    let c = 1.0 / radius;
    let parab = ConicParams {
        vertex: Vec2::ZERO,
        axis: Vec2::new(-1.0, 0.0),
        c,
        k: -1.0,
        half_ap: ha,
    };
    let sag = c * ha * ha / 2.0;
    let tip_x = -sag;
    let mirror = Element::new(
        ElementId(0),
        vec![
            Edge::Conic(parab),
            Edge::Line {
                a: Vec2::new(tip_x, ha),
                b: Vec2::new(5.0, ha),
            },
            Edge::Line {
                a: Vec2::new(5.0, ha),
                b: Vec2::new(5.0, -ha),
            },
            Edge::Line {
                a: Vec2::new(5.0, -ha),
                b: Vec2::new(tip_x, -ha),
            },
        ],
        Material::Mirror { reflectivity: 1.0 },
    );
    let mut scene = Scene::new();
    scene.add_element(mirror);
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 3.0 },
        transform: Isometry2::from_pos(Vec2::new(-20.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 16, // even -> no exactly-on-axis ray
    });
    Golden {
        name: "parabolic-mirror",
        scene,
        plane: -10.0,
        max_depth: 4,
    }
}

fn prism_scene() -> Golden {
    // An equilateral-ish glass prism; collimated rays enter the left face.
    let tri = vec![
        Edge::Line {
            a: Vec2::new(-6.0, -5.0),
            b: Vec2::new(6.0, -5.0),
        },
        Edge::Line {
            a: Vec2::new(6.0, -5.0),
            b: Vec2::new(0.0, 6.0),
        },
        Edge::Line {
            a: Vec2::new(0.0, 6.0),
            b: Vec2::new(-6.0, -5.0),
        },
    ];
    let prism = Element::new(ElementId(0), tri, Material::AbbePoint { nd: 1.6, vd: 40.0 });
    let mut scene = Scene::new();
    scene.add_element(prism);
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 2.0 },
        transform: Isometry2::from_pos(Vec2::new(-20.0, -1.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 9,
    });
    Golden {
        name: "prism",
        scene,
        plane: 30.0,
        max_depth: 6,
    }
}

/// Compare CPU-f32 vs the f64 reference across all golden scenes.
#[test]
fn cpu_f32_matches_f64_reference() {
    let mut worst = 0.0f64;
    for g in [singlet_scene(), mirror_scene(), prism_scene()] {
        let launch: Vec<EmittedRay> = g.scene.sources.iter().flat_map(|s| s.emit()).collect();
        let gpu_scene = GpuScene::build(&g.scene, LAMBDA_D);
        assert_eq!(gpu_scene.rays.len(), launch.len(), "{}: ray count", g.name);

        let cpu = trace_cpu(&gpu_scene, g.max_depth);

        let mut scene_worst = 0.0f64;
        let mut compared = 0;
        for (i, er) in launch.iter().enumerate() {
            let (Some(rf), Some(rc)) = (
                ref_crossing(&g.scene, er, g.plane),
                interactive_crossing(&cpu[i], g.plane),
            ) else {
                continue;
            };
            let dev = (rf - rc).abs();
            scene_worst = scene_worst.max(dev);
            compared += 1;
        }
        assert!(compared > 0, "{}: nothing compared", g.name);
        println!(
            "{:>18}: {} rays, worst |Δy| = {:.3e} mm at plane x={}",
            g.name, compared, scene_worst, g.plane
        );
        // f32 vs f64 at mm scale: 50 µm envelope is comfortably tight and still
        // catches any real algorithmic divergence.
        assert!(
            scene_worst < 5e-2,
            "{}: CPU f32 diverged from f64 reference by {:.4} mm",
            g.name,
            scene_worst
        );
        worst = worst.max(scene_worst);
    }
    println!("overall worst |Δy| = {worst:.3e} mm");
}

/// Compare the GPU backend against the CPU f32 backend (identical math ⇒ must
/// agree to f32 transcendental noise). Skips cleanly if no adapter is present.
#[cfg(feature = "gpu")]
#[test]
fn gpu_matches_cpu_f32() {
    let Some(tracer) = GpuTracer::new() else {
        eprintln!("no wgpu adapter; skipping GPU cross-check");
        return;
    };
    eprintln!(
        "GPU backend: {:?} '{}'",
        tracer.adapter_info.backend, tracer.adapter_info.name
    );

    let mut worst = 0.0f64;
    for g in [singlet_scene(), mirror_scene(), prism_scene()] {
        let gpu_scene = GpuScene::build(&g.scene, LAMBDA_D);
        let max_segs = gpu_scene.max_segs as usize;

        let cpu_raw = oc_gpu::cpu::trace(&gpu_scene, g.max_depth);
        let gpu_raw = tracer.trace(&gpu_scene, g.max_depth);
        assert_eq!(cpu_raw.len(), gpu_raw.len(), "{}: buffer size", g.name);

        let cpu = oc_gpu::segments_per_ray(&cpu_raw, gpu_scene.rays.len(), max_segs);
        let gpu = oc_gpu::segments_per_ray(&gpu_raw, gpu_scene.rays.len(), max_segs);

        let mut scene_worst = 0.0f64;
        for i in 0..gpu_scene.rays.len() {
            if let (Some(cc), Some(gc)) = (
                interactive_crossing(&cpu[i], g.plane),
                interactive_crossing(&gpu[i], g.plane),
            ) {
                scene_worst = scene_worst.max((cc - gc).abs());
            }
        }
        println!(
            "{:>18}: GPU vs CPU worst |Δy| = {:.3e} mm",
            g.name, scene_worst
        );
        assert!(
            scene_worst < 1e-2,
            "{}: GPU diverged from CPU f32 by {:.4} mm",
            g.name,
            scene_worst
        );
        worst = worst.max(scene_worst);
    }
    println!("GPU vs CPU overall worst |Δy| = {worst:.3e} mm");
}

/// A detector-centroid sanity check that the interactive picture is physically
/// sane: the singlet focuses the collimated beam near the axis.
#[test]
fn singlet_focuses_on_axis() {
    let g = singlet_scene();
    let gpu_scene = GpuScene::build(&g.scene, LAMBDA_D);
    let cpu = trace_cpu(&gpu_scene, g.max_depth);
    // Every exit ray should cross the focal plane close to y = 0.
    let mut max_abs_y = 0.0f64;
    for segs in &cpu {
        if let Some(y) = interactive_crossing(segs, 51.3) {
            max_abs_y = max_abs_y.max(y.abs());
        }
    }
    assert!(max_abs_y < 0.5, "spot spread too large: {max_abs_y} mm");
    // Keep Detector in the public surface exercised.
    let _ = Detector::new(Vec2::new(51.3, -1.0), Vec2::new(51.3, 1.0), 32);
    let _ = Ray::new(Vec2::ZERO, Vec2::X);
}

// ---------------------------------------------------------------------------
// M2: spectral dispersion, ghosts, TIR flag.
// ---------------------------------------------------------------------------

use oc_core::{LAMBDA_C, LAMBDA_F};
use oc_gpu::{BuildOpts, SEG_GHOST, SEG_TIR};

/// Where the exit ray crosses the optical axis (y = 0) — the focus, per ray.
fn exit_axis_x(segs: &[oc_gpu::Segment]) -> Option<f64> {
    let s = segs
        .iter()
        .max_by(|p, q| p.a.x.partial_cmp(&q.a.x).unwrap())?;
    let dy = (s.b.y - s.a.y) as f64;
    if dy.abs() < 1e-9 {
        return None;
    }
    let t = -s.a.y as f64 / dy;
    Some(s.a.x as f64 + t * (s.b.x - s.a.x) as f64)
}

/// A dispersive singlet illuminated by an F/d/C beam. The interactive spectral
/// trace must (a) match the f64 reference per ray, and (b) actually disperse —
/// blue (F) focuses nearer than red (C).
#[test]
fn spectral_dispersion_matches_reference() {
    // Real crown so dispersion is physical (N-BK7 from the builtin catalog).
    let scene = {
        let (r, tc, ha) = (50.0, 4.0, 8.0);
        let c = 1.0 / r;
        let front = ConicParams {
            vertex: Vec2::new(-tc / 2.0, 0.0),
            axis: Vec2::new(1.0, 0.0),
            c,
            k: 0.0,
            half_ap: ha,
        };
        let rear = ConicParams {
            vertex: Vec2::new(tc / 2.0, 0.0),
            axis: Vec2::new(-1.0, 0.0),
            c,
            k: 0.0,
            half_ap: ha,
        };
        let sag = c * ha * ha / (1.0 + (1.0 - c * c * ha * ha).sqrt());
        let (ftip, rtip) = (-tc / 2.0 + sag, tc / 2.0 - sag);
        let mut s = Scene::new();
        let bk7 = Material::Catalog(s.catalog.find("N-BK7").unwrap());
        s.add_element(Element::new(
            ElementId(0),
            vec![
                Edge::Conic(front),
                Edge::Line {
                    a: Vec2::new(ftip, ha),
                    b: Vec2::new(rtip, ha),
                },
                Edge::Conic(rear),
                Edge::Line {
                    a: Vec2::new(rtip, -ha),
                    b: Vec2::new(ftip, -ha),
                },
            ],
            bk7,
        ));
        s.sources.push(Source {
            kind: SourceKind::Collimated { half_width: 4.0 },
            transform: Isometry2::from_pos(Vec2::new(-60.0, 0.0)),
            spectrum: Spectrum::Lines(vec![(LAMBDA_F, 1.0), (LAMBDA_D, 1.0), (LAMBDA_C, 1.0)]),
            flux: 1.0,
            rays: 11,
        });
        s
    };

    let launch: Vec<EmittedRay> = scene.sources.iter().flat_map(|s| s.emit()).collect();
    let gs = GpuScene::build(&scene, LAMBDA_D);
    assert_eq!(gs.rays.len(), launch.len());
    let cpu = trace_cpu(&gs, 6);

    // (a) Per-ray agreement with the reference (each ray at its own λ).
    let mut worst = 0.0f64;
    for (i, er) in launch.iter().enumerate() {
        if let (Some(rf), Some(rc)) = (
            ref_crossing(&scene, er, 90.0),
            interactive_crossing(&cpu[i], 90.0),
        ) {
            worst = worst.max((rf - rc).abs());
        }
    }
    assert!(worst < 5e-2, "spectral f32 vs f64 worst {worst:.4} mm");

    // (b) Mean focus per line — must be ordered F < d < C (normal dispersion).
    let focus = |lam: f64| -> f64 {
        let mut sum = 0.0;
        let mut n = 0;
        for (i, er) in launch.iter().enumerate() {
            if (er.lambda_um - lam).abs() < 1e-9 {
                if let Some(x) = exit_axis_x(&cpu[i]) {
                    sum += x;
                    n += 1;
                }
            }
        }
        sum / n as f64
    };
    let (ff, fd, fc) = (focus(LAMBDA_F), focus(LAMBDA_D), focus(LAMBDA_C));
    println!("spectral foci: F={ff:.3}  d={fd:.3}  C={fc:.3} mm (worst Δ vs ref {worst:.2e})");
    assert!(ff < fd && fd < fc, "expected F < d < C, got {ff} {fd} {fc}");
}

/// With ghosts on, the interactive tracer emits the Fresnel back-reflection and
/// its flux matches the analytic normal-incidence reflectance R₀.
#[test]
fn ghost_reflectance_matches_fresnel() {
    let nd = 1.5;
    let mut scene = Scene::new();
    // Flat glass slab; front face at x = -5, normal along x.
    let hx = 5.0;
    scene.add_element(Element::new(
        ElementId(0),
        vec![
            Edge::Line {
                a: Vec2::new(-hx, -20.0),
                b: Vec2::new(hx, -20.0),
            },
            Edge::Line {
                a: Vec2::new(hx, -20.0),
                b: Vec2::new(hx, 20.0),
            },
            Edge::Line {
                a: Vec2::new(hx, 20.0),
                b: Vec2::new(-hx, 20.0),
            },
            Edge::Line {
                a: Vec2::new(-hx, 20.0),
                b: Vec2::new(-hx, -20.0),
            },
        ],
        Material::AbbePoint { nd, vd: 60.0 },
    ));
    let flux_per_ray = 0.25;
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 3.0 },
        transform: Isometry2::from_pos(Vec2::new(-30.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: flux_per_ray * 4.0,
        rays: 4,
    });

    let gs = GpuScene::build_with(
        &scene,
        BuildOpts {
            ghosts: true,
            max_segs: 24,
            flux_eps: 1e-6,
            max_ghost: 1,
        },
    );
    let per_ray = trace_cpu(&gs, 16);

    // R₀ = ((n-1)/(n+1))²; AbbePoint index at the d line is exactly nd.
    let r0 = ((nd - 1.0) / (nd + 1.0)).powi(2);

    // The strongest backward-travelling ghost is the front-face reflection.
    let mut best_ghost = 0.0f64;
    let mut any_ghost = false;
    for segs in &per_ray {
        for s in segs {
            if s.is_ghost() {
                any_ghost = true;
                let dir_x = (s.b.x - s.a.x) as f64;
                if dir_x < -0.5 {
                    best_ghost = best_ghost.max(s.flux as f64);
                }
            }
        }
    }
    assert!(any_ghost, "ghosts on but no ghost segments produced");
    let ratio = best_ghost / flux_per_ray;
    println!("ghost front reflectance = {ratio:.5} (Fresnel R₀ = {r0:.5})");
    assert!((ratio - r0).abs() < 1e-3, "ghost flux {ratio} vs R₀ {r0}");
}

/// A ray inside glass striking a face beyond the critical angle is flagged TIR.
#[test]
fn tir_surface_is_flagged() {
    let nd = 1.5;
    let crit = (1.0f64 / nd).asin();
    let mut scene = Scene::new();
    scene.add_element(Element::new(
        ElementId(0),
        vec![
            Edge::Line {
                a: Vec2::new(-5.0, -100.0),
                b: Vec2::new(5.0, -100.0),
            },
            Edge::Line {
                a: Vec2::new(5.0, -100.0),
                b: Vec2::new(5.0, 100.0),
            },
            Edge::Line {
                a: Vec2::new(5.0, 100.0),
                b: Vec2::new(-5.0, 100.0),
            },
            Edge::Line {
                a: Vec2::new(-5.0, 100.0),
                b: Vec2::new(-5.0, -100.0),
            },
        ],
        Material::AbbePoint { nd, vd: 60.0 },
    ));
    let theta = crit + 5f64.to_radians();
    scene.sources.push(Source {
        kind: SourceKind::Point { half_angle: 0.0 },
        transform: Isometry2::new(Vec2::ZERO, theta),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 1,
    });

    let gs = GpuScene::build(&scene, LAMBDA_D);
    let per_ray = trace_cpu(&gs, 8);
    let tir = per_ray.iter().flatten().any(|s| s.flags & SEG_TIR != 0);
    assert!(tir, "expected a TIR-flagged segment");
    // The unused import guard.
    let _ = SEG_GHOST;
}
