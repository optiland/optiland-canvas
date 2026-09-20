//! What a gallery scene costs to trace: the f64 reference and the f32 CPU
//! preview, with and without ghosts, on the settings the app uses.
//!
//! ```text
//! RAYON_NUM_THREADS=1 cargo run --release -p oc-gpu --example bench_gallery -- web/public/gallery
//! ```
//! One thread is what one browser worker gets; the app's pool of workers
//! shares the reference out (docs/performance.md).

use oc_core::scene3::Scene3;
use oc_gpu::scene3::{BuildOpts, GpuScene3};
use oc_trace::{depth_for_scene3, trace_scene3, TraceConfig3};
use std::time::Instant;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "web/public/gallery".into());
    let only: Vec<String> = std::env::args().skip(2).collect();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("gallery dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ocanvas"))
        .filter(|p| {
            only.is_empty()
                || only.iter().any(|o| {
                    p.file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .contains(o.as_str())
                })
        })
        .collect();
    files.sort();
    println!(
        "{:30} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "design", "rays", "f64 ms", "f64 segs", "f64+g ms", "g segs", "f32 ms", "f32+g ms"
    );
    for f in files {
        let scene = Scene3::from_json(&std::fs::read_to_string(&f).unwrap()).unwrap();
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let rays: usize = scene.sources.iter().map(|s| s.emit().len()).sum();
        let f64_run = |ghosts: bool| {
            let cfg = TraceConfig3 {
                ghosts,
                max_ghost_bounces: 2,
                max_depth: depth_for_scene3(&scene, ghosts, 2),
                ..TraceConfig3::default()
            };
            let t = Instant::now();
            let res = trace_scene3(&scene, &cfg);
            (t.elapsed().as_secs_f64() * 1e3, res.segments.len())
        };
        let f32_run = |ghosts: bool| {
            let t = Instant::now();
            let job = GpuScene3::build(
                &scene,
                BuildOpts {
                    ghosts,
                    flux_eps: 1e-4,
                    max_ghost: 2,
                },
            );
            let segs = oc_gpu::cpu3::trace_segments(&job, depth_for_scene3(&scene, ghosts, 2));
            let n: usize = segs.iter().map(|s| s.len()).sum();
            (t.elapsed().as_secs_f64() * 1e3, n)
        };
        if std::env::var("F32HASH").is_ok() {
            // The same fingerprint for the f32 preview's segments.
            for ghosts in [false, true] {
                let job = GpuScene3::build(
                    &scene,
                    BuildOpts {
                        ghosts,
                        flux_eps: 1e-4,
                        max_ghost: 2,
                    },
                );
                let t = Instant::now();
                let segs = oc_gpu::cpu3::trace_segments(&job, depth_for_scene3(&scene, ghosts, 2));
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let mut n = 0;
                for ray in &segs {
                    for sg in ray {
                        n += 1;
                        for b in format!("{sg:?}").bytes() {
                            h ^= b as u64;
                            h = h.wrapping_mul(0x100_0000_01b3);
                        }
                    }
                }
                println!("{name:30} f32 ghosts {ghosts:5} {ms:>9.1} ms {n:>9} segs  {h:016x}");
            }
            continue;
        }
        if std::env::var("HASH").is_ok() {
            // A fingerprint of the whole exact result, bit for bit: every
            // segment's ends and flux, every detector cell, and the ledger.
            for ghosts in [false, true] {
                let cfg = TraceConfig3 {
                    ghosts,
                    max_ghost_bounces: 2,
                    max_depth: depth_for_scene3(&scene, ghosts, 2),
                    polarization: std::env::var("NOPOL").is_err(),
                    ..TraceConfig3::default()
                };
                let t = Instant::now();
                let res = trace_scene3(&scene, &cfg);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let mut eat = |v: f64| {
                    for b in v.to_bits().to_le_bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100_0000_01b3);
                    }
                };
                let mut segs: Vec<_> = res
                    .segments
                    .iter()
                    .map(|s| {
                        [
                            s.a.x,
                            s.a.y,
                            s.a.z,
                            s.b.x,
                            s.b.y,
                            s.b.z,
                            s.flux_end,
                            s.lambda_um,
                        ]
                    })
                    .collect();
                segs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if let Ok(path) = std::env::var("DUMP") {
                    let text: Vec<String> = segs
                        .iter()
                        .map(|s| {
                            s.iter()
                                .map(|v| format!("{v:e}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                        .collect();
                    std::fs::write(format!("{path}.{name}.{ghosts}.txt"), text.join("\n")).unwrap();
                }
                for s in &segs {
                    s.iter().for_each(|v| eat(*v));
                }
                for d in &res.detectors {
                    d.incident_flux.iter().for_each(|v| eat(*v));
                    eat(d.total_incident);
                }
                eat(res.ledger.truncated_flux);
                eat(res.ledger.escaped);
                println!(
                    "{name:30} ghosts {ghosts:5} {:>9.1} ms {:>9} segs  {h:016x}",
                    ms,
                    segs.len()
                );
            }
            continue;
        }
        if let Ok(budget) = std::env::var("JOB") {
            // The app's ghost reference, stepped as its worker steps it, for at
            // most `JOB` seconds: how far it gets, and the time it would take.
            let budget: f64 = budget.parse().unwrap_or(60.0);
            let cfg = TraceConfig3 {
                ghosts: true,
                max_ghost_bounces: 2,
                max_depth: depth_for_scene3(&scene, true, 2),
                record_detector_hits: true,
                max_records: 8192,
                segment_floor: 1e-5,
                ..TraceConfig3::default()
            };
            let t = Instant::now();
            let mut job = oc_trace::TraceJob3::new(scene.clone(), cfg);
            while !job.step(1) && t.elapsed().as_secs_f64() < budget {}
            let secs = t.elapsed().as_secs_f64();
            let (done, total) = job.progress();
            let segs = job.segments_so_far();
            let whole = if done > 0 {
                secs * total as f64 / done as f64
            } else {
                f64::NAN
            };
            println!(
                "{name:30} ghosts: {done}/{total} rays in {secs:.1} s, {segs} segs kept, whole ~{whole:.1} s"
            );
            continue;
        }
        if std::env::var("WHY").is_ok() {
            for depth in [24, depth_for_scene3(&scene, true, 2)] {
                let cfg = TraceConfig3 {
                    ghosts: true,
                    max_ghost_bounces: 2,
                    max_depth: depth,
                    ..TraceConfig3::default()
                };
                let t = Instant::now();
                let res = trace_scene3(&scene, &cfg);
                let top = res
                    .segments
                    .iter()
                    .map(|s| s.flux_start)
                    .fold(0.0, f64::max);
                for floor in [1e-3, 1e-4, 1e-5, 1e-6] {
                    let n = res
                        .segments
                        .iter()
                        .filter(|s| s.flux_start >= floor * top)
                        .count();
                    println!("  segments at or above {floor:e} of the brightest: {n}");
                }
                let tir = res.segments.iter().filter(|s| s.flags.tir).count();
                let ghost = res.segments.iter().filter(|s| s.flags.ghost).count();
                let det: f64 = res.detectors.iter().map(|d| d.total_incident).sum();
                println!(
                    "{name} depth {depth}: {:.0} ms, {} segs ({} ghost, {} tir), detected {:.6}, truncated {:.3e}\n  {:?}",
                    t.elapsed().as_secs_f64() * 1e3,
                    res.segments.len(),
                    ghost,
                    tir,
                    det,
                    res.ledger.truncated_flux,
                    res.terminations
                );
            }
            continue;
        }
        let (a, sa) = f64_run(false);
        let (b, sb) = f64_run(true);
        let (c, _) = f32_run(false);
        let (d, _) = f32_run(true);
        println!("{name:30} {rays:>5} {a:>9.1} {sa:>9} {b:>9.1} {sb:>9} {c:>9.1} {d:>9.1}");
    }
}
