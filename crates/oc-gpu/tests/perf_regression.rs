//! Performance-regression gate (companion to the correctness gate in
//! `cross_validate.rs`). It measures every tracer path the project ships and
//! **fails if a change makes one slower than the committed baseline**:
//!
//! - **f64 CPU** — the exact reference tracer (`oc-trace`, rayon-parallel).
//! - **f32 CPU** — the no-WebGPU fallback (`oc-gpu::cpu`).
//! - **f32 GPU** — the interactive `wgpu` backend, measured on **every** adapter
//!   `wgpu` enumerates (discrete GPU, integrated GPU, software fallback), not
//!   just the auto-selected one.
//!
//! ## How the baseline stays portable
//!
//! Absolute wall-clock times are machine-specific, so a committed baseline of
//! raw milliseconds would be meaningless on another box. Instead:
//!
//! - **CPU paths are normalized to "work units"** = `trace_time / calibration_time`,
//!   where the calibration is a fixed, optimizer-resistant arithmetic loop run
//!   in the same process. A machine twice as fast runs *both* twice as fast, so
//!   the ratio is stable; only an actual code regression moves it. These units
//!   are gated on every machine.
//! - **GPU paths are keyed by `backend:adapter-name`.** Each adapter is gated
//!   against its own committed baseline (reference machine), and — for real
//!   (non-CPU) adapters — against the project's hard SLA of ≤ 8 ms at 2k×8
//!   (spec §7). An adapter with no committed entry is measured and reported but
//!   not failed (record it with `OC_PERF_BLESS=1`).
//!
//! ## Running / updating
//!
//! ```sh
//! cargo test -p oc-gpu --release --test perf_regression -- --nocapture   # gate + print table
//! OC_PERF_BLESS=1 cargo test -p oc-gpu --release --test perf_regression   # re-record baseline
//! ```
//!
//! Perf is only meaningful in `--release`; a debug build **skips** all
//! assertions (and prints a note) so the ordinary `cargo test` stays green.
//! For the least-noisy numbers add `-- --test-threads=1`. Tolerances default to
//! 35 % (CPU) / 60 % (GPU) and can be overridden with `OC_PERF_TOL`.
//!
//! ## Why the sampling looks generous
//!
//! `min` is the right estimator here, but it is only as good as its sample
//! count, and a rep cap that binds on a sub-millisecond workload throws that
//! away. This gate used to cap CPU paths at 200 reps: at ~0.25 ms a trace that
//! is 48 ms of sampling, and the resulting `min` moved by **20-40 % between
//! runs** on an idle machine. The tolerance is 35 %, so `f64_ref_2000x8` (the
//! shortest path, and the one whose baseline has no headroom) failed roughly one
//! run in three for no reason at all.
//!
//! So the reps are governed by the time budget, not a cap. The cap remains only
//! where a single rep is genuinely slow (a software GPU adapter traces in tens of
//! milliseconds, and there is no point spending a whole budget on one). This
//! costs a few seconds and takes the spread to ~6 %.
//!
//! Note that this cannot mask a regression: more samples only ever lower a `min`,
//! and a slower code path raises it regardless of how well it is sampled. The
//! committed baselines stay valid for the same reason.

use std::hint::black_box;
use std::time::Instant;

use oc_core::{
    ConicParams, Edge, Element, ElementId, GlassId, Isometry2, Material, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_D,
};
use oc_trace::{trace as trace_f64, TraceConfig};

use oc_gpu::scene::GpuScene;

const BASELINE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/perf_baseline.json");

/// Workloads (launch-ray count × bounce depth). 2k×8 is the spec §7 budget point.
const RAY_COUNTS: [u32; 2] = [2000, 8000];
const DEPTH: u32 = 8;

/// Hard interactive SLA on a real GPU (spec §7): ≤ 8 ms to trace 2k×8.
const GPU_SLA_MS: f64 = 8.0;
const SLA_RAYS: u32 = 2000;

const CPU_TOL_DEFAULT: f64 = 0.35;
const GPU_TOL_DEFAULT: f64 = 0.60;

/// Sampling budget for one CPU measurement.
///
/// The CPU paths are sub-millisecond, so this is thousands of reps for `min` to
/// choose from. Measured spread on an idle 24-thread box: ~42 % at the old
/// 200-rep cap, ~10 % at 0.4 s uncapped, ~6 % here.
const CPU_BUDGET_S: f64 = 0.8;
/// Warmup reps before a CPU measurement. The f64 reference is rayon-parallel and
/// its pool parks between calls, so the first reps pay a wake-up that has nothing
/// to do with the tracer.
const CPU_WARMUP: u32 = 10;

// ------------------------------------------------------------------ scene ----

/// A biconvex N-BK7 singlet — the same shape the `drag` demo and cross-validation
/// use, so the perf workload matches the correctness workload. N-BK7 exercises
/// the Sellmeier dispersion path in the tracer inner loop.
fn singlet() -> Element {
    let (r, tc, ha) = (50.0_f64, 4.0_f64, 12.0_f64);
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
    Element::new(
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
        Material::Catalog(GlassId(0)), // N-BK7 (index 0 in the builtin catalog)
    )
}

fn build_scene(rays: u32) -> Scene {
    let mut scene = Scene::new();
    scene.add_element(singlet());
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 11.0 },
        transform: Isometry2::from_pos(Vec2::new(-60.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays,
    });
    scene
}

// ------------------------------------------------------------ measurement ----

/// Fixed, optimizer-resistant scalar-FP workload. Its runtime is the machine's
/// "unit of arithmetic speed"; CPU trace costs are divided by it so the baseline
/// is machine-independent. Loop-carried dependencies block vectorization/elision.
fn calibration_ns() -> f64 {
    const ITERS: u64 = 5_000_000;
    let mut a = black_box(1.000_000_1_f64);
    let mut b = black_box(0.999_999_9_f32);
    let t0 = Instant::now();
    for _ in 0..ITERS {
        a = (a * 1.000_000_3 + 1e-12).sqrt();
        a = a.mul_add(0.5, 0.500_000_1);
        b = (b * 1.000_000_7_f32 + 1e-9_f32).sqrt();
        b = b.mul_add(0.5_f32, 0.500_000_1_f32);
    }
    let ns = t0.elapsed().as_nanos() as f64;
    black_box((a, b));
    ns
}

/// Best (minimum) single-iteration time in nanoseconds.
///
/// `min` is the least-noisy estimator of "how fast can this run": jitter only
/// ever makes a sample slower, so the floor is the signal. It is only as good as
/// its sample count, though, which is what `max_reps` is dangerous for.
///
/// **`max_reps` must not bind on a fast path.** `budget_s` already bounds the
/// wall time on its own, because the budget is checked after every rep, so a cap
/// buys nothing there and costs the estimate its samples. Pass `u32::MAX` unless
/// a single rep is slow enough that finishing the budget would be a waste
/// (a software GPU adapter, which traces in tens of milliseconds). See the
/// module docs for what capping a 0.25 ms workload at 200 reps did to this gate.
fn best_ns(budget_s: f64, warmup: u32, max_reps: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = f64::INFINITY;
    let start = Instant::now();
    let mut reps = 0u32;
    loop {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_nanos() as f64);
        reps += 1;
        if reps >= max_reps || start.elapsed().as_secs_f64() > budget_s {
            break;
        }
    }
    best
}

// --------------------------------------------------------------- baseline ----

fn load_baseline() -> serde_json::Value {
    std::fs::read_to_string(BASELINE_PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}

fn tol(env_key_default: f64) -> f64 {
    std::env::var("OC_PERF_TOL")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(env_key_default)
}

// ------------------------------------------------------------------- test ----

#[test]
fn perf_regression() {
    if cfg!(debug_assertions) {
        eprintln!(
            "perf_regression: SKIPPED in debug build (timings are meaningless). \
             Run with `cargo test -p oc-gpu --release --test perf_regression`."
        );
        return;
    }

    let bless = std::env::var("OC_PERF_BLESS").is_ok();
    let baseline = load_baseline();
    let cpu_tol = tol(CPU_TOL_DEFAULT);
    let gpu_tol = tol(GPU_TOL_DEFAULT);

    let mut recorded = serde_json::Map::new();
    let mut cpu_out = serde_json::Map::new();
    let mut gpu_out = serde_json::Map::new();
    let mut violations: Vec<String> = Vec::new();

    // Machine speed unit (min of a few runs).
    let cal = (0..5).map(|_| calibration_ns()).fold(f64::INFINITY, f64::min);
    println!("\n=== oc-gpu performance gate (release) ===");
    println!("calibration unit: {:.0} ns  (CPU costs normalized to this)\n", cal);

    // ---- CPU paths: normalized work units ----
    println!(
        "{:<26}{:>12}{:>12}{:>14}",
        "cpu path", "time", "units", "baseline"
    );
    for &rays in &RAY_COUNTS {
        let scene = build_scene(rays);
        let cfg = TraceConfig {
            ghosts: false,
            max_depth: DEPTH,
            ..TraceConfig::default()
        };
        // f64 reference tracer.
        {
            let key = format!("f64_ref_{rays}x{DEPTH}");
            let ns = best_ns(CPU_BUDGET_S, CPU_WARMUP, u32::MAX, || {
                black_box(trace_f64(black_box(&scene), black_box(&cfg)));
            });
            let units = ns / cal;
            let base = baseline["cpu"][&key]["units"].as_f64();
            report_cpu(&key, ns, units, base, cpu_tol, &mut violations);
            cpu_out.insert(key, serde_json::json!({ "units": units }));
        }
        // f32 CPU fallback.
        {
            let key = format!("f32_cpu_{rays}x{DEPTH}");
            let gs = GpuScene::build(&scene, LAMBDA_D);
            let ns = best_ns(CPU_BUDGET_S, CPU_WARMUP, u32::MAX, || {
                black_box(oc_gpu::cpu::trace(black_box(&gs), DEPTH));
            });
            let units = ns / cal;
            let base = baseline["cpu"][&key]["units"].as_f64();
            report_cpu(&key, ns, units, base, cpu_tol, &mut violations);
            cpu_out.insert(key, serde_json::json!({ "units": units }));
        }
    }

    // ---- GPU paths: per-adapter, keyed by backend:name ----
    #[cfg(feature = "gpu")]
    {
        use oc_gpu::gpu::GpuTracer;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        println!(
            "\n{:<40}{:>10}{:>12}{:>10}",
            "gpu adapter (2k×8 unless noted)", "ms", "baseline", "SLA"
        );
        for adapter in adapters {
            let info = adapter.get_info();
            let key = format!("{:?}:{}", info.backend, info.name);
            let Some(tracer) = GpuTracer::from_adapter(&adapter) else {
                println!("{key:<40}{:>10}", "(no device)");
                continue;
            };
            let is_real_gpu = info.device_type != wgpu::DeviceType::Cpu;
            let mut entry = serde_json::Map::new();
            for &rays in &RAY_COUNTS {
                let gs = GpuScene::build(&build_scene(rays), LAMBDA_D);
                let _ = tracer.trace(&gs, DEPTH); // warm pipeline

                // A real adapter traces in ~0.1 ms, so the budget governs and
                // the cap must not bind. A software adapter takes tens of ms,
                // which is the one case where a cap earns its keep.
                let (budget, warmup, reps) = if is_real_gpu {
                    (0.5, 5, u32::MAX)
                } else {
                    (1.0, 2, 20)
                };
                let ns = best_ns(budget, warmup, reps, || {
                    black_box(tracer.trace(black_box(&gs), DEPTH));
                });
                let ms = ns / 1.0e6;
                let subkey = format!("ms_{rays}x{DEPTH}");
                let base = baseline["gpu"][&key][&subkey].as_f64();

                // SLA gate at the 2k×8 point on real GPUs.
                let sla = if is_real_gpu && rays == SLA_RAYS {
                    if ms > GPU_SLA_MS {
                        violations.push(format!(
                            "GPU SLA: {key} traced {rays}x{DEPTH} in {ms:.3} ms > {GPU_SLA_MS} ms budget"
                        ));
                        "FAIL"
                    } else {
                        "ok"
                    }
                } else {
                    "-"
                };
                // Per-adapter regression gate (only if this adapter is baselined).
                if let Some(b) = base {
                    if ms > b * (1.0 + gpu_tol) {
                        violations.push(format!(
                            "GPU regression: {key} {subkey} = {ms:.3} ms > {:.3} ms (baseline {b:.3} +{:.0}%)",
                            b * (1.0 + gpu_tol),
                            gpu_tol * 100.0
                        ));
                    }
                }
                if rays == SLA_RAYS {
                    let base_str = base.map(|b| format!("{b:.3}")).unwrap_or_else(|| "-".into());
                    println!("{key:<40}{ms:>10.3}{base_str:>12}{sla:>10}");
                }
                entry.insert(subkey, serde_json::json!(ms));
            }
            gpu_out.insert(key, serde_json::Value::Object(entry));
        }
    }
    #[cfg(not(feature = "gpu"))]
    println!("\n(gpu feature disabled — GPU paths not measured)");

    recorded.insert("note".into(), serde_json::json!("machine-normalized CPU units; per-adapter GPU ms. Regenerate with OC_PERF_BLESS=1."));
    recorded.insert("cpu".into(), serde_json::Value::Object(cpu_out));
    recorded.insert("gpu".into(), serde_json::Value::Object(gpu_out));

    if bless {
        let pretty = serde_json::to_string_pretty(&serde_json::Value::Object(recorded)).unwrap();
        std::fs::write(BASELINE_PATH, pretty).expect("write baseline");
        println!("\nOC_PERF_BLESS=1 → wrote baseline to {BASELINE_PATH}");
        return;
    }

    println!();
    assert!(
        violations.is_empty(),
        "performance regression(s) detected:\n  - {}\n\nIf this is an intentional/hardware change, re-record with \
         OC_PERF_BLESS=1 (see test docs).",
        violations.join("\n  - ")
    );
}

fn report_cpu(
    key: &str,
    ns: f64,
    units: f64,
    base: Option<f64>,
    tol: f64,
    violations: &mut Vec<String>,
) {
    let base_str = base.map(|b| format!("{b:.3}")).unwrap_or_else(|| "-".into());
    println!(
        "{:<26}{:>10.3}ms{:>12.3}{:>14}",
        key,
        ns / 1.0e6,
        units,
        base_str
    );
    if let Some(b) = base {
        if units > b * (1.0 + tol) {
            violations.push(format!(
                "CPU regression: {key} = {units:.3} units > {:.3} (baseline {b:.3} +{:.0}%)",
                b * (1.0 + tol),
                tol * 100.0
            ));
        }
    }
}
