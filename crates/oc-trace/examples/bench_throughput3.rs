//! Throughput benchmark for the **3D** `trace3` reference kernel on a biconvex
//! singlet — the 3D counterpart of `bench_throughput.rs`.
//!
//! `bench_throughput.rs` measures the 2D meridional tracer, whose per-ray work
//! is a slice through the system rather than a full solid-angle trace. That
//! makes its rays/s incomparable to a 3D Monte Carlo tracer. This example runs
//! the same optic through `trace3` so the numbers can be put side by side with
//! other 3D non-sequential engines.
//!
//! Same geometry as the 2D bench: biconvex R = +/-50, tc = 6, n = 1.5,
//! semi-aperture 16 mm, collimated square beam, detector downstream.
//!
//! `Mrays/s` counts launch (primary) rays; `Mseg/s` counts traced segments.
//!
//! Run:
//!   cargo run --release -p oc-trace --features trace3 --example bench_throughput3
//!   RAYON_NUM_THREADS=1 cargo run --release -p oc-trace --features trace3 \
//!       --example bench_throughput3

use oc_core::{build3::SingletGeom, scene3::Scene3, Spectrum, Vec3, LAMBDA_D};
use oc_trace::{trace_scene3, TraceConfig3};
use std::time::Instant;

fn build_scene(nrays: u32, ghosts: bool) -> Scene3 {
    let _ = ghosts;
    let mut scene = Scene3::new();
    let geom = SingletGeom {
        r1_mm: 50.0,
        r2_mm: -50.0,
        thickness_mm: 6.0,
        semi_aperture_mm: 16.0,
        conic1: 0.0,
        conic2: 0.0,
    };
    scene.add_singlet(Vec3::new(0.0, 0.0, 0.0), geom, 1.5, 50.0);
    scene.add_collimated_beam(
        Vec3::new(-25.0, 0.0, 0.0),
        16.0,
        16.0,
        nrays,
        Spectrum::Monochromatic(LAMBDA_D),
    );
    scene.add_area_detector(
        Vec3::new(60.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        20.0,
        20.0,
        [256, 256],
    );
    scene
}

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    println!("# oc-trace THROUGHPUT (3D trace3 kernel, biconvex n=1.5, rayon threads={threads}, best of 3)");
    println!(
        "{:>10} {:>8} {:>11} {:>13} {:>13} {:>13}",
        "nrays", "ghosts", "wall_s", "Mrays/s", "Mseg/s", "segments"
    );

    for ghosts in [false, true] {
        for nrays_u in [10_000u32, 100_000, 1_000_000, 4_000_000] {
            let scene = build_scene(nrays_u, ghosts);
            let cfg = TraceConfig3 {
                ghosts,
                max_depth: 24,
                max_ghost_bounces: 8,
                max_branches: u64::MAX,
                record_detector_hits: false,
                ..Default::default()
            };

            // Warm once, then best of 3.
            let warm = trace_scene3(&scene, &cfg);
            let nrays = nrays_u as u64;
            let mut best = f64::INFINITY;
            let mut segs = warm.segments.len() as u64;
            for _ in 0..3 {
                let t0 = Instant::now();
                let res = trace_scene3(&scene, &cfg);
                let dt = t0.elapsed().as_secs_f64();
                if dt < best {
                    best = dt;
                    segs = res.segments.len() as u64;
                }
            }
            println!(
                "{:>10} {:>8} {:>11.5} {:>13.2} {:>13.2} {:>13}",
                nrays,
                ghosts,
                best,
                nrays as f64 / best / 1e6,
                segs as f64 / best / 1e6,
                segs
            );
        }
    }
}
