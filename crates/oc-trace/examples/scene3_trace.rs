//! CV-A — the headless Scene3 trace runner: `.ocanvas` v2 in, results JSON out.
//!
//! The cross-repository verification seam. optikit-core's materializer emits a
//! Scene3 document (`SPEC-OPTIKIT-CORE-CANVAS-INTEGRATION` §9); this runner loads
//! it, traces it through the f64 reference tracer ([`trace_scene3`] — the same
//! authority the browser calls on idle), and prints the trace as JSON so a Python
//! harness can compare it against an Optiland oracle ray-by-ray (KIT-05) or assert
//! a design's chain contract (KIT-06). It is the headless analogue of the wasm
//! `traceProjected3D`, and it extends the `audit_compare` example pattern to the
//! full 3D scene.
//!
//! Every displayed number in the emitted JSON is f64 straight from the reference
//! result; nothing here recomputes or downcasts. Segment order is sorted to a
//! stable key so two runs of one scene agree regardless of trace parallelism
//! (the `parallel` feature reduces per-thread, so vector order is not stable on
//! its own).
//!
//! Run:
//! ```text
//! cargo run -p oc-trace --features trace3 --example scene3_trace -- \
//!     path/to/scene3.ocanvas [--pretty] [--no-ghosts] [--max-depth N] \
//!     [--records] [--cells] [--summary]
//! ```
//! `-` reads the scene from stdin. Output goes to stdout.
#![cfg(feature = "trace3")]

use oc_core::scene3::Scene3;
use oc_trace::{DetectorResult3, Segment3, TraceConfig3, TraceResult3};
use serde_json::{json, Value};
use std::io::Read;

/// What the runner emits, distinct from the trace physics: the segment buffer and
/// per-cell arrays are large and optional, everything else is always present.
struct Options {
    input: String,
    pretty: bool,
    /// Emit the full segment list (default). `--summary` suppresses it, keeping
    /// only counts, detectors, and the ledger.
    segments: bool,
    /// Emit per-cell `incident_flux`/`detected_signal` arrays per detector.
    cells: bool,
    /// Keep and emit per-hit detector records (`cfg.record_detector_hits`).
    records: bool,
    cfg: TraceConfig3,
}

impl Options {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut input: Option<String> = None;
        let mut pretty = false;
        let mut segments = true;
        let mut cells = false;
        let mut records = false;
        // Ghosts default on: the reference is the exact path (§3.6). A harness
        // comparing a single refractive chain passes `--no-ghosts`.
        let mut cfg = TraceConfig3::default();
        let mut it = args.peekable();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--pretty" => pretty = true,
                "--summary" => segments = false,
                "--cells" => cells = true,
                "--records" => records = true,
                "--ghosts" => cfg.ghosts = true,
                "--no-ghosts" => cfg.ghosts = false,
                "--max-depth" => cfg.max_depth = take_parse(&mut it, "--max-depth")?,
                "--flux-eps" => cfg.flux_eps = take_parse(&mut it, "--flux-eps")?,
                "--bounds" => cfg.bounds_mm = take_parse(&mut it, "--bounds")?,
                "--max-branches" => cfg.max_branches = take_parse(&mut it, "--max-branches")?,
                "--max-records" => cfg.max_records = take_parse(&mut it, "--max-records")?,
                s if s.starts_with("--") => return Err(format!("unknown flag {s}")),
                s => {
                    if input.replace(s.to_string()).is_some() {
                        return Err("more than one input path given".into());
                    }
                }
            }
        }
        cfg.record_detector_hits = records;
        Ok(Options {
            input: input.ok_or("usage: scene3_trace <scene.ocanvas | -> [flags]")?,
            pretty,
            segments,
            cells,
            records,
            cfg,
        })
    }
}

fn take_parse<T: std::str::FromStr>(
    it: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    it.next()
        .ok_or_else(|| format!("{flag} needs a value"))?
        .parse()
        .map_err(|_| format!("{flag}: not a valid value"))
}

fn v3(v: oc_core::vec3::Vec3) -> Value {
    json!([v.x, v.y, v.z])
}

fn v2(v: oc_core::vec::Vec2) -> Value {
    json!([v.x, v.y])
}

/// A traced segment, world-frame f64. Path ids let a chain-contract test group a
/// branch; `ghost`/`tir` flag the non-primary walks.
fn segment_json(s: &Segment3) -> Value {
    json!({
        "a": v3(s.a),
        "b": v3(s.b),
        "flux_start": s.flux_start,
        "flux_end": s.flux_end,
        "lambda_um": s.lambda_um,
        "opl0_mm": s.opl0_mm,
        "opl1_mm": s.opl1_mm,
        "path_id": s.path_id.raw(),
        "parent_path_id": s.parent_path_id.map(|p| p.raw()),
        "ghost": s.flags.ghost,
        "tir": s.flags.tir,
    })
}

/// A detector readout. Centroid and mean OPL are the f64 numbers EMB-E checks the
/// embedded kernel against; the per-cell arrays and per-hit records are opt-in.
fn detector_json(d: &DetectorResult3, cells: bool, records: bool) -> Value {
    let mut o = json!({
        "detector": d.detector.raw(),
        "bins": d.bins,
        "pixel_area_mm2": d.pixel_area_mm2,
        "face_area_mm2": d.face_area_mm2,
        "hits": d.hits,
        "total_incident": d.total_incident,
        "total_signal": d.total_signal,
        "centroid_uv_mm": d.centroid_uv().map(v2),
        "mean_opl_mm": d.mean_opl_mm(),
    });
    if cells {
        o["incident_flux"] = json!(d.incident_flux);
        o["detected_signal"] = json!(d.detected_signal);
        o["covered_area_mm2"] = json!(d.covered_area_mm2);
    }
    if records {
        o["records"] = Value::Array(
            d.records
                .iter()
                .map(|h| {
                    json!({
                        "point_world": v3(h.point_world),
                        "uv_mm": v2(h.uv_mm),
                        "incident_dir_local": v3(h.incident_dir_local),
                        "lambda_um": h.lambda_um,
                        "incident_flux": h.incident_flux,
                        "signal": h.signal,
                        "opl_mm": h.opl_mm,
                        "path_id": h.path_id.raw(),
                        "parent_path_id": h.parent_path_id.map(|p| p.raw()),
                        "depth": h.depth,
                        "ghost_depth": h.ghost_depth,
                    })
                })
                .collect(),
        );
    }
    o
}

fn ledger_json(l: &oc_trace::EnergyLedger) -> Value {
    json!({
        "launched": l.launched,
        "detected_incident": l.detected_incident,
        "detected_signal": l.detected_signal,
        "escaped": l.escaped,
        "absorbed_volume": l.absorbed_volume,
        "absorbed_surface": l.absorbed_surface,
        "blocked_aperture": l.blocked_aperture,
        "truncated_flux": l.truncated_flux,
        "numerical_loss": l.numerical_loss,
        "residual": l.residual(),
    })
}

fn terminations_json(t: &oc_trace::TerminationSummary) -> Value {
    json!({
        "detector": t.detector,
        "absorbed_volume": t.absorbed_volume,
        "absorbed_surface": t.absorbed_surface,
        "aperture_blocked": t.aperture_blocked,
        "escaped_bounds": t.escaped_bounds,
        "flux_threshold": t.flux_threshold,
        "max_depth": t.max_depth,
        "max_ghost_depth": t.max_ghost_depth,
        "max_branches": t.max_branches,
        "numerical_failure": t.numerical_failure,
        "unsupported_physics": t.unsupported_physics,
    })
}

/// The capability report is the honesty field: a non-empty `rejected` means some
/// flux could not be traced exactly, so a harness must not trust the numbers. The
/// issue variants are not `Serialize`, so they are `Debug`-rendered as strings.
fn capability_json(r: &oc_trace::CapabilityReport) -> Value {
    json!({
        "clean": r.is_clean(),
        "rejected": r.rejected.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>(),
        "approximated": r.approximated.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
        "exact": r.exact.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>(),
    })
}

/// Convert a trace result into the runner's JSON, sorting segments to a stable
/// key so the output is reproducible under parallel tracing.
fn result_json(res: &TraceResult3, scene: &Scene3, opts: &Options) -> Value {
    let mut out = json!({
        "engine": "optiland-canvas",
        "runner": "cv-a",
        "revision": res.revision,
        "scene": {
            "bodies": scene.bodies.len(),
            "surfaces": scene.surfaces.len(),
            "apertures": scene.apertures.len(),
            "detectors": scene.detectors.len(),
            "sources": scene.sources.len(),
        },
        "config": {
            "ghosts": opts.cfg.ghosts,
            "max_depth": opts.cfg.max_depth,
            "flux_eps": opts.cfg.flux_eps,
            "bounds_mm": opts.cfg.bounds_mm,
        },
        "segment_count": res.segments.len(),
        "detectors": res.detectors.iter().map(|d| detector_json(d, opts.cells, opts.records)).collect::<Vec<_>>(),
        "ledger": ledger_json(&res.ledger),
        "terminations": terminations_json(&res.terminations),
        "capability": capability_json(&res.capability_report),
    });
    if opts.segments {
        let mut segs: Vec<&Segment3> = res.segments.iter().collect();
        segs.sort_by(|a, b| seg_key(a).partial_cmp(&seg_key(b)).unwrap());
        out["segments"] = Value::Array(segs.into_iter().map(segment_json).collect());
    }
    out
}

/// A total order over segments independent of trace scheduling: path id first,
/// then the two endpoints. Endpoints are finite (the scene is finite by §9), so
/// `partial_cmp` never sees a NaN.
fn seg_key(s: &Segment3) -> (u64, u64, f64, f64, f64, f64, f64, f64) {
    (
        s.path_id.raw(),
        s.parent_path_id.map(|p| p.raw()).unwrap_or(u64::MAX),
        s.a.x,
        s.a.y,
        s.a.z,
        s.b.x,
        s.b.y,
        s.b.z,
    )
}

fn run(opts: &Options) -> Result<Value, String> {
    let text = if opts.input == "-" {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("read stdin: {e}"))?;
        s
    } else {
        std::fs::read_to_string(&opts.input).map_err(|e| format!("read {}: {e}", opts.input))?
    };
    let scene = Scene3::from_json(&text).map_err(|e| format!("parse Scene3: {e:?}"))?;
    let res = oc_trace::trace_scene3(&scene, &opts.cfg);
    Ok(result_json(&res, &scene, opts))
}

fn main() {
    let opts = match Options::parse(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let value = match run(&opts) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let text = if opts.pretty {
        serde_json::to_string_pretty(&value)
    } else {
        serde_json::to_string(&value)
    }
    .expect("serialize result");
    println!("{text}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::scene::Spectrum;
    use oc_core::material::LAMBDA_D;
    use oc_core::scene3::{
        AngularDistribution3, Detector3, DetectorGeometry3, DetectorResponse, DetectorTermination,
        ObjectCommon, SamplingSequence, Source3, SourceSampling, SpatialDistribution3,
    };
    use oc_core::id::LayerId;
    use oc_core::vec3::{Isometry3, Vec3};

    /// A collimated 2 mm disc on axis at x = -10 firing +X into a 5×5 mm detector
    /// at x = +10 facing the beam. Every ray lands; the centroid sits at the
    /// origin. Mirrors the `sources3` suite's seam test, self-contained here.
    fn disc_into_detector() -> Scene3 {
        let mut scene = Scene3::new();
        let sid = scene.alloc_source_id();
        scene.sources.push(Source3 {
            window: Default::default(),
            id: sid,
            transform: Isometry3::from_pos(Vec3::new(-10.0, 0.0, 0.0)),
            spatial: SpatialDistribution3::Disc { radius_mm: 2.0 },
            angular: AngularDistribution3::Direction { local_dir: Vec3::X },
            spectrum: Spectrum::Monochromatic(LAMBDA_D),
            flux: 1.0,
            samples: SourceSampling {
                spatial_samples: 64,
                angular_samples: 1,
                seed: 0,
                sequence: SamplingSequence::Grid,
            },
        });
        let q = glam::DQuat::from_rotation_arc(Vec3::Z, -Vec3::X);
        let mut common = ObjectCommon::new(scene.alloc_object_id(), LayerId(0));
        common.transform = Isometry3::new(Vec3::new(10.0, 0.0, 0.0), [q.x, q.y, q.z, q.w]).unwrap();
        scene.detectors.push(Detector3 {
            common,
            geometry: DetectorGeometry3::Rectangle {
                half_width_mm: 5.0,
                half_height_mm: 5.0,
            },
            bins: [16, 16],
            response: DetectorResponse::Unity,
            termination: DetectorTermination::Terminate,
            sides: Default::default(),
        });
        scene
    }

    fn opts_for(scene_json: &str, records: bool) -> (Scene3, Options) {
        let scene = Scene3::from_json(scene_json).expect("scene parses");
        let mut cfg = TraceConfig3::default();
        cfg.ghosts = false;
        cfg.record_detector_hits = records;
        (
            scene,
            Options {
                input: "-".into(),
                pretty: false,
                segments: true,
                cells: true,
                records,
                cfg,
            },
        )
    }

    #[test]
    fn emits_the_expected_top_level_shape() {
        let (scene, opts) = opts_for(&disc_into_detector().to_json(), false);
        let res = oc_trace::trace_scene3(&scene, &opts.cfg);
        let v = result_json(&res, &scene, &opts);
        for key in [
            "engine",
            "runner",
            "revision",
            "scene",
            "config",
            "segment_count",
            "segments",
            "detectors",
            "ledger",
            "terminations",
            "capability",
        ] {
            assert!(v.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(v["runner"], "cv-a");
        assert_eq!(v["scene"]["sources"], 1);
        assert_eq!(v["scene"]["detectors"], 1);
        assert!(v["segment_count"].as_u64().unwrap() > 0, "some rays traced");
        assert!(v["capability"]["clean"].as_bool().unwrap(), "clean trace");
    }

    #[test]
    fn detector_centroid_is_on_axis_and_all_flux_lands() {
        let (scene, opts) = opts_for(&disc_into_detector().to_json(), false);
        let res = oc_trace::trace_scene3(&scene, &opts.cfg);
        let v = result_json(&res, &scene, &opts);
        let det = &v["detectors"][0];
        assert_eq!(det["hits"].as_u64().unwrap(), 64, "every ray lands");
        assert!(
            (det["total_incident"].as_f64().unwrap() - 1.0).abs() < 1e-12,
            "all flux collected"
        );
        let c = det["centroid_uv_mm"].as_array().expect("centroid present");
        assert!(c[0].as_f64().unwrap().abs() < 1e-9, "centroid u on axis");
        assert!(c[1].as_f64().unwrap().abs() < 1e-9, "centroid v on axis");
    }

    #[test]
    fn segment_order_is_stable_across_runs() {
        let (scene, opts) = opts_for(&disc_into_detector().to_json(), false);
        let a = result_json(&oc_trace::trace_scene3(&scene, &opts.cfg), &scene, &opts);
        let b = result_json(&oc_trace::trace_scene3(&scene, &opts.cfg), &scene, &opts);
        assert_eq!(a["segments"], b["segments"], "sorted segments reproduce");
    }

    #[test]
    fn records_flag_turns_on_per_hit_output() {
        let (scene, opts) = opts_for(&disc_into_detector().to_json(), true);
        let res = oc_trace::trace_scene3(&scene, &opts.cfg);
        let v = result_json(&res, &scene, &opts);
        let recs = v["detectors"][0]["records"].as_array().expect("records present");
        assert_eq!(recs.len(), 64, "one record per landed ray");
        assert!(recs[0].get("opl_mm").is_some(), "record carries OPL");
    }
}
