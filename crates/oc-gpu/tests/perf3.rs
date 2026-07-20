//! Full-3D preview performance gate (§16.2, PR 3D-08).
//!
//! The 3D counterpart of `perf_regression.rs`'s GPU path. It measures the WebGPU
//! 3D tracer ([`Gpu3Tracer`]) at the spec's budget point and fails if a real GPU
//! misses the interactive SLA:
//!
//! - 2,000 launch rays × 8 interactions × 32 analytic objects, ghosts off:
//!   ≤ 8 ms trace compute;
//! - 8,000 launch rays at the same depth: ≤ 16 ms.
//!
//! The measured time is wall-clock `trace()` — upload + dispatch + read-back — so
//! it is *stricter* than pure compute; passing it passes the compute budget with
//! margin. Perf is only meaningful in `--release`; a debug build skips the
//! assertions. The test skips cleanly when no wgpu adapter is present, and only
//! *gates* real (non-CPU) adapters — a software fallback is reported, not failed.
#![cfg(feature = "gpu3")]

use std::hint::black_box;
use std::time::Instant;

use oc_core::id::{LayerId, SourceId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::scene3::{Body3, ObjectCommon, Scene3, SolidGeometry3, SurfaceProps3};
use oc_core::source3::EmittedRay3;
use oc_core::vec3::{Isometry3, Ray3, Vec3};
use oc_gpu::gpu3::Gpu3Tracer;
use oc_gpu::scene3::{BuildOpts, GpuScene3};

const DEPTH: u32 = 8;
const MAX_SEGS: u32 = 16;
const N_OBJECTS: usize = 32;

/// The §16.2 budgets: 2k×8 ≤ 8 ms, 8k×8 ≤ 16 ms on a real GPU.
const WORKLOADS: [(u32, f64); 2] = [(2000, 8.0), (8000, 16.0)];

/// A stack of 32 thin glass plates along +X: a collimated beam refracts through
/// them in sequence, so a ray takes several interactions and every nearest-event
/// scan loops all 32 objects — the cost §16.2 gates.
fn plate_scene() -> Scene3 {
    let mut scene = Scene3::new();
    for i in 0..N_OBJECTS {
        let id = scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        let x = -30.0 + i as f64 * 2.0;
        common.transform = Isometry3::from_pos(Vec3::new(x, 0.0, 0.0));
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Box {
                half_extents_mm: Vec3::new(0.5, 15.0, 15.0),
            },
            material: Material::AbbePoint { nd: 1.5, vd: 64.0 },
            absorption: Default::default(),
            surface: SurfaceProps3::default(),
            priority: 0,
        });
    }
    scene
}

/// `n` collimated +X rays on a square grid spanning the plates.
fn collimated(n: u32) -> Vec<EmittedRay3> {
    let side = (n as f64).sqrt().ceil() as u32;
    let mut rays = Vec::new();
    for iy in 0..side {
        for iz in 0..side {
            if rays.len() as u32 >= n {
                break;
            }
            let y = -10.0 + 20.0 * iy as f64 / (side - 1) as f64;
            let z = -10.0 + 20.0 * iz as f64 / (side - 1) as f64;
            rays.push(EmittedRay3 {
                ray: Ray3::new(Vec3::new(-60.0, y, z), Vec3::X).unwrap(),
                lambda_um: LAMBDA_D,
                flux: 1.0,
                source: SourceId(0),
            });
        }
    }
    rays
}

/// Best (minimum) single-call time in ms; min is the least-noisy "how fast can
/// this run" estimator.
fn best_ms(budget_s: f64, reps: u32, mut f: impl FnMut()) -> f64 {
    f(); // warm
    let mut best = f64::INFINITY;
    let start = Instant::now();
    let mut n = 0;
    loop {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        n += 1;
        if n >= reps || start.elapsed().as_secs_f64() > budget_s {
            break;
        }
    }
    best
}

#[test]
fn full_3d_preview_meets_the_16_2_budget() {
    if cfg!(debug_assertions) {
        eprintln!(
            "perf3: SKIPPED in debug build (timings are meaningless). \
             Run with `cargo test -p oc-gpu --features gpu3 --release --test perf3`."
        );
        return;
    }
    let Some(tracer) = Gpu3Tracer::new() else {
        eprintln!("no wgpu adapter; skipping 3D preview perf gate");
        return;
    };
    let info = &tracer.adapter_info;
    let is_real_gpu = info.device_type != wgpu::DeviceType::Cpu;
    eprintln!(
        "GPU3 backend: {:?} '{}' (real GPU: {is_real_gpu})",
        info.backend, info.name
    );

    let opts = BuildOpts {
        ghosts: false,
        flux_eps: 1e-4,
        max_ghost: 0,
    };
    let scene = plate_scene();
    let mut violations = Vec::new();

    println!("\n=== full-3D preview perf (§16.2, {N_OBJECTS} objects, depth {DEPTH}) ===");
    for &(rays, budget_ms) in &WORKLOADS {
        let gpu_scene = GpuScene3::build_with_rays(&scene, &collimated(rays), opts);
        assert!(gpu_scene.rejected.is_empty());
        let ms = best_ms(0.6, 120, || {
            black_box(tracer.trace(black_box(&gpu_scene), DEPTH, MAX_SEGS));
        });
        let sla = if is_real_gpu {
            if ms > budget_ms {
                violations.push(format!(
                    "{rays}×{DEPTH}: {ms:.3} ms > {budget_ms} ms budget"
                ));
                "FAIL"
            } else {
                "ok"
            }
        } else {
            "-"
        };
        println!("{rays:>6} rays  {ms:>8.3} ms   (budget {budget_ms} ms)  {sla}");
    }
    println!();

    assert!(
        violations.is_empty(),
        "full-3D preview missed the §16.2 budget:\n  - {}",
        violations.join("\n  - ")
    );
}
