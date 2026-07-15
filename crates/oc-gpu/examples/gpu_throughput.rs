//! Throughput of the **interactive** `oc-gpu` tracer (the f32 GPU/CPU path that
//! was optimized), on the same biconvex singlet as `bench_throughput.rs`.
//!
//! Unlike the f64 reference (`oc-trace`), `oc-gpu` materializes every ray segment
//! (`n_rays * max_segs`), so it is a visualizer, not a batch renderer: it is
//! measured up to ~1e6 rays, where the segment read-back caps out.
//!
//! Run: cargo run --release -p oc-gpu --example gpu_throughput

use oc_core::{
    ConicParams, Edge, Element, ElementId, Isometry2, Material, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_D,
};
use oc_gpu::{cpu, gpu::GpuTracer, scene::GpuScene};
use std::time::Instant;

const DEPTH: u32 = 8;

fn biconvex(r: f64, tc: f64, ha: f64, material: Material) -> Element {
    let c = 1.0 / r;
    let front = ConicParams { vertex: Vec2::new(-tc / 2.0, 0.0), axis: Vec2::new(1.0, 0.0), c, k: 0.0, half_ap: ha };
    let rear = ConicParams { vertex: Vec2::new(tc / 2.0, 0.0), axis: Vec2::new(-1.0, 0.0), c, k: 0.0, half_ap: ha };
    let sag = c * ha * ha / (1.0 + (1.0 - c * c * ha * ha).sqrt());
    let (ftip, rtip) = (-tc / 2.0 + sag, tc / 2.0 - sag);
    Element::new(
        ElementId(0),
        vec![
            Edge::Conic(front),
            Edge::Line { a: Vec2::new(ftip, ha), b: Vec2::new(rtip, ha) },
            Edge::Conic(rear),
            Edge::Line { a: Vec2::new(rtip, -ha), b: Vec2::new(ftip, -ha) },
        ],
        material,
    )
}

fn scene(nrays: u32) -> Scene {
    let mut s = Scene::new();
    s.add_element(biconvex(50.0, 6.0, 16.0, Material::AbbePoint { nd: 1.5, vd: 50.0 }));
    s.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 16.0 },
        transform: Isometry2::from_pos(Vec2::new(-25.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: nrays,
    });
    s
}

fn best_ms(mut f: impl FnMut(), reps: u32) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    // Capped at 500k: oc-gpu materializes n_rays*max_segs segments, and wgpu's
    // default max storage buffer is 256 MB (~666k rays at 12 segs x 32 B). This
    // cap is itself the point: the interactive tracer is not a batch renderer.
    let counts: [u32; 5] = [1_000, 10_000, 100_000, 300_000, 500_000];
    let gpu = GpuTracer::new();
    if let Some(g) = &gpu {
        println!("# oc-gpu GPU adapter: {} ({:?})", g.adapter_info.name, g.adapter_info.backend);
    } else {
        println!("# no GPU adapter");
    }
    println!("{:>10}  {:>10}  {:>12}  {:>10}  {:>12}", "nrays", "gpu_ms", "gpu_Mrays/s", "cpu_ms", "cpu_Mrays/s");
    for &n in &counts {
        let gs = GpuScene::build(&scene(n), LAMBDA_D);
        let (gms, gmr) = if let Some(g) = &gpu {
            g.trace(&gs, DEPTH); // warm
            let ms = best_ms(|| { let _ = g.trace(&gs, DEPTH); }, 3);
            (ms, n as f64 / ms / 1e3)
        } else {
            (f64::NAN, f64::NAN)
        };
        let cms = best_ms(|| { let _ = cpu::trace(&gs, DEPTH); }, 3);
        let cmr = n as f64 / cms / 1e3;
        println!("{:>10}  {:>10.4}  {:>12.2}  {:>10.4}  {:>12.2}", n, gms, gmr, cms, cmr);
    }
}
