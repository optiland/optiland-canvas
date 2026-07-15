//! Throughput benchmark for the `oc-trace` reference tracer on a biconvex
//! singlet.
//!
//! Traces a collimated bundle of N parallel rays (half_width 16 mm) through the
//! same biconvex singlet used in `validation/systems.json`
//! (R = +/-50, tc = 6, n = 1.5 constant), sweeping N and reporting rays/second.
//!
//! Two modes bracket the energy-split cost:
//!   - ghosts=on : Fresnel back-reflections spawned at each interface
//!   - ghosts=off: single transmitted path
//!
//! `rays/s` counts launch (primary) rays; `Mseg/s` counts traced segments
//! (branches), the closer analogue to per-segment throughput.
//!
//! Run (uses rayon threads by default; pin with RAYON_NUM_THREADS=1):
//!   cargo run --release -p oc-trace --example bench_throughput

use oc_core::{
    ConicParams, Edge, Element, ElementId, Isometry2, Material, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_D,
};
use oc_trace::{trace, TraceConfig};
use std::time::Instant;

fn biconvex(r: f64, tc: f64, half_ap: f64, material: Material) -> Element {
    let c = 1.0 / r;
    let front = ConicParams { vertex: Vec2::new(-tc / 2.0, 0.0), axis: Vec2::new(1.0, 0.0), c, k: 0.0, half_ap };
    let rear = ConicParams { vertex: Vec2::new(tc / 2.0, 0.0), axis: Vec2::new(-1.0, 0.0), c, k: 0.0, half_ap };
    let sag = c * half_ap * half_ap / (1.0 + (1.0 - c * c * half_ap * half_ap).sqrt());
    let (ftip, rtip) = (-tc / 2.0 + sag, tc / 2.0 - sag);
    Element::new(
        ElementId(0),
        vec![
            Edge::Conic(front),
            Edge::Line { a: Vec2::new(ftip, half_ap), b: Vec2::new(rtip, half_ap) },
            Edge::Conic(rear),
            Edge::Line { a: Vec2::new(rtip, -half_ap), b: Vec2::new(ftip, -half_ap) },
        ],
        material,
    )
}

fn build_scene(nrays: u32) -> Scene {
    let mut scene = Scene::new();
    // Constant index n=1.5 at the d line (AbbePoint returns exactly nd there).
    scene.add_element(biconvex(50.0, 6.0, 16.0, Material::AbbePoint { nd: 1.5, vd: 50.0 }));
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 16.0 },
        transform: Isometry2::from_pos(Vec2::new(-25.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: nrays,
    });
    scene.detectors.push(oc_core::Detector::new(
        Vec2::new(60.0, -20.0),
        Vec2::new(60.0, 20.0),
        256,
    ));
    scene
}

fn bench(nrays: u32, ghosts: bool, repeats: u32) -> (f64, u64) {
    let scene = build_scene(nrays);
    let cfg = TraceConfig { ghosts, ..TraceConfig::default() };
    let mut best = f64::INFINITY;
    let mut branches = 0u64;
    for _ in 0..repeats {
        let t = Instant::now();
        let res = trace(&scene, &cfg);
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
            branches = res.branches;
        }
    }
    (best, branches)
}

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    let counts: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let counts = if counts.is_empty() {
        vec![10_000, 100_000, 1_000_000, 4_000_000, 10_000_000]
    } else {
        counts
    };

    println!("# oc-trace throughput  (biconvex n=1.5, rayon threads={threads}, best of 3)");
    println!(
        "{:>10}  {:>7}  {:>10}  {:>12}  {:>12}  {:>12}",
        "nrays", "ghosts", "wall_s", "Mrays/s", "Mseg/s", "segments"
    );
    for &ghosts in &[false, true] {
        for &n in &counts {
            let (wall, branches) = bench(n, ghosts, 3);
            let mrays = n as f64 / wall / 1e6;
            let mseg = branches as f64 / wall / 1e6;
            println!(
                "{:>10}  {:>7}  {:>10.5}  {:>12.2}  {:>12.2}  {:>12}",
                n, ghosts, wall, mrays, mseg, branches
            );
        }
    }
}
