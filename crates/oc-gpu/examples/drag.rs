//! M1 "it moves": drag a lens across the scene and watch the light follow —
//! and measure the interactive frame budget against §7 (≤ 8 ms trace at
//! 2k rays × 8 bounces). No "run" button; every frame is a full retrace.
//!
//! Run: `cargo run -p oc-gpu --release --example drag`

use oc_core::{
    ConicParams, Edge, Element, ElementId, Isometry2, Material, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_D,
};
use oc_gpu::{gpu::GpuTracer, scene::GpuScene, segments_per_ray};
use std::time::Instant;

fn singlet(y_offset: f64) -> Element {
    let (r, tc, ha) = (50.0, 4.0, 12.0);
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
    let mut e = Element::new(
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
        Material::Catalog(oc_core::GlassId(0)), // N-BK7 (index 0 in builtin catalog)
    );
    e.transform = Isometry2::from_pos(Vec2::new(0.0, y_offset));
    e
}

fn build_scene(y_offset: f64, rays: u32) -> Scene {
    let mut scene = Scene::new();
    scene.add_element(singlet(y_offset));
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 11.0 },
        transform: Isometry2::from_pos(Vec2::new(-60.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays,
    });
    scene
}

fn main() {
    let Some(tracer) = GpuTracer::new() else {
        eprintln!("no wgpu adapter available");
        return;
    };
    println!(
        "GPU: {:?} '{}'\n",
        tracer.adapter_info.backend, tracer.adapter_info.name
    );

    // "Drag" the lens vertically; report where the focused spot lands each frame.
    println!("Dragging the lens — the focus tracks it (no recompute button):");
    for step in 0..=6 {
        let y = -6.0 + 2.0 * step as f64;
        let scene = build_scene(y, 33);
        let gs = GpuScene::build(&scene, LAMBDA_D);
        let raw = tracer.trace(&gs, 8);
        let per_ray = segments_per_ray(&raw, gs.rays.len(), gs.max_segs as usize);
        // Focus proxy: mean exit-ray crossing of the plane x = 45.
        let mut sum = 0.0;
        let mut n = 0;
        for segs in &per_ray {
            if let Some(s) = segs
                .iter()
                .max_by(|a, b| a.a.x.partial_cmp(&b.a.x).unwrap())
            {
                let dx = (s.b.x - s.a.x) as f64;
                if dx.abs() > 1e-6 {
                    let t = (45.0 - s.a.x as f64) / dx;
                    sum += s.a.y as f64 + t * (s.b.y - s.a.y) as f64;
                    n += 1;
                }
            }
        }
        println!(
            "  lens y = {y:+.1} mm  →  focused spot y = {:+.3} mm",
            sum / n as f64
        );
    }

    // Frame-budget check against §7: 2k rays × 8 bounces.
    println!("\nInteractive frame budget (§7 target: ≤ 8 ms trace @ 2k×8):");
    for &rays in &[1000u32, 2000, 8000] {
        let scene = build_scene(0.0, rays);
        let gs = GpuScene::build(&scene, LAMBDA_D);
        // Warm up (shader/pipeline caches).
        let _ = tracer.trace(&gs, 8);
        let frames = 60;
        let t0 = Instant::now();
        for _ in 0..frames {
            let _ = tracer.trace(&gs, 8);
        }
        let per_frame = t0.elapsed().as_secs_f64() * 1000.0 / frames as f64;
        println!(
            "  {rays:>5} rays × 8 bounces: {per_frame:6.3} ms/frame  ({:.0} fps){}",
            1000.0 / per_frame,
            if rays == 2000 && per_frame <= 8.0 {
                "  ✅ within budget"
            } else {
                ""
            }
        );
    }
    println!("\n(Note: M1 rebuilds buffers each frame; persistent buffers + progressive\naccumulation on release are the M6 perf task.)");
}
