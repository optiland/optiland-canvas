//! The analysis job and its first analysis, `detectorStats`.
//!
//! What this suite owes:
//!
//! - a job gives the same result to the bit however it is stepped: one ray at
//!   a time, in odd batches, or all at once;
//! - its pooled numbers are the kernel's own readout of the same rays, to the
//!   bit;
//! - on a beam whose answer is known, each value is within a few standard
//!   errors of it, and across many independent runs the standard errors are
//!   the size of the actual errors;
//! - a number with no noise says so: its standard error is zero;
//! - a request the scene cannot answer is refused with the reason.

use oc_analysis::{AnalysisJob, AnalysisRequest};
use oc_core::scene::Spectrum;
use oc_core::scene3::{SamplingSequence, Scene3};
use oc_core::vec3::Vec3;
use oc_trace::{TraceConfig3, TraceJob3};
use serde_json::Value;

fn cfg(revision: u64) -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        revision,
        ..TraceConfig3::default()
    }
}

fn request(text: &str) -> AnalysisRequest {
    AnalysisRequest::from_json(text).unwrap()
}

fn run(scene: &Scene3, req: &str, batch: u64) -> Value {
    let mut job = AnalysisJob::new(scene, cfg(3), request(req)).unwrap();
    let mut steps = 0;
    while !job.step(batch) {
        steps += 1;
        assert!(job.result().is_none(), "no result before the last ray");
    }
    if batch == 1 {
        assert!(steps > 100, "{steps} steps of one ray");
    }
    job.result().unwrap()
}

/// A square collimated beam, half-width `beam` mm, monochromatic, sampled on
/// a stratified grid of `n` × `n`, onto a square detector of half-width
/// `face` mm 50 mm on.
fn beam_scene(beam: f64, face: f64, n: u32, seed: u64) -> Scene3 {
    let mut s = Scene3::new();
    s.add_collimated_beam(Vec3::ZERO, beam, beam, n * n, Spectrum::Monochromatic(0.55));
    let src = &mut s.sources[0].samples;
    src.sequence = SamplingSequence::Stratified;
    src.seed = seed;
    s.add_area_detector(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, face, face, [8, 8]);
    s
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[test]
fn stepped_one_ray_at_a_time_is_stepped_all_at_once() {
    let scene = Scene3::default_scene();
    let req = r#"{"kind":"detectorStats","passes":8}"#;
    let whole = run(&scene, req, u64::MAX);
    let text = serde_json::to_string(&whole).unwrap();
    for batch in [1, 37, 1000] {
        let stepped = serde_json::to_string(&run(&scene, req, batch)).unwrap();
        assert_eq!(stepped, text, "batches of {batch}");
    }
}

/// The pooled values are what the kernel's own job gives for the same rays,
/// to the bit: the analysis adds nothing to the trace.
#[test]
fn the_pooled_values_are_the_kernels_readout() {
    let scene = Scene3::default_scene();
    let r = run(&scene, r#"{"kind":"detectorStats","passes":4}"#, u64::MAX);
    let mut job = TraceJob3::with_passes(scene.clone(), 4, None, cfg(3));
    while !job.step(usize::MAX) {}
    let t = job.finish();
    let d = &r["result"]["detectors"][0];
    assert_eq!(num(&d["flux"]["value"]).to_bits(), t.detectors[0].total_incident.to_bits());
    assert_eq!(d["hits"].as_u64().unwrap(), t.detectors[0].hits);
    let lines = d["lines"].as_array().unwrap();
    assert_eq!(lines.len(), t.detectors[0].spectral.len());
    for (l, s) in lines.iter().zip(&t.detectors[0].spectral) {
        assert_eq!(num(&l["flux"]["value"]).to_bits(), s.flux.to_bits());
        let rms = s.rms_uv().unwrap();
        assert_eq!(num(&l["rmsUv"][0]["value"]).to_bits(), rms[0].to_bits());
        assert_eq!(num(&l["rmsUv"][1]["value"]).to_bits(), rms[1].to_bits());
    }
}

#[test]
fn the_envelope_states_what_the_numbers_rest_on() {
    let scene = Scene3::default_scene();
    let r = run(&scene, r#"{"kind":"detectorStats","rays":3000,"passes":4}"#, u64::MAX);
    assert_eq!(r["kind"], "detectorStats");
    assert_eq!(r["revision"], 3);
    assert_eq!(r["fidelity"]["preset"], "Standard");
    assert!(r["fidelity"]["solverVersion"].is_string());
    assert!(r["approximations"].is_array());
    assert_eq!(r["rejected"], Value::Array(vec![]));
    assert_eq!(r["sampling"]["passes"], 4);
    let rays = r["sampling"]["rays"].as_u64().unwrap();
    assert!((2900..=3100).contains(&rays), "{rays} rays for 3000");
    let d = &r["result"]["detectors"][0];
    assert!(num(&d["flux"]["stderr"]) >= 0.0);
    for l in d["lines"].as_array().unwrap() {
        for k in ["centroidUv", "rmsUv"] {
            for a in 0..2 {
                assert!(num(&l[k][a]["stderr"]) >= 0.0, "{k}[{a}] states its error");
            }
        }
    }
}

/// The whole beam lands: every pass gives the full flux, and the error says
/// there is no noise. The spot of a uniform square of half-width `a` has an
/// RMS of `a/√3` along each axis and is centred: within a few standard errors.
#[test]
fn a_beam_that_lands_whole_reads_its_known_spot() {
    let scene = beam_scene(5.0, 12.0, 40, 1);
    let r = run(&scene, r#"{"kind":"detectorStats","passes":16}"#, u64::MAX);
    let d = &r["result"]["detectors"][0];
    assert!((num(&d["flux"]["value"]) - 1.0).abs() < 1e-12);
    assert!(num(&d["flux"]["stderr"]) < 1e-12, "{}", d["flux"]);
    let l = &d["lines"][0];
    let want = 5.0 / 3f64.sqrt();
    for a in 0..2 {
        let (v, se) = (num(&l["rmsUv"][a]["value"]), num(&l["rmsUv"][a]["stderr"]));
        assert!(se > 0.0 && se < 1e-2, "rms[{a}] stderr {se}");
        assert!((v - want).abs() < 5.0 * se + 1e-9, "rms[{a}] {v} ± {se}, want {want}");
        let (c, se) = (num(&l["centroidUv"][a]["value"]), num(&l["centroidUv"][a]["stderr"]));
        assert!(c.abs() < 5.0 * se + 1e-9, "centroid[{a}] {c} ± {se}");
    }
}

/// A detector smaller than the beam catches `(face/beam)²` of it. Over many
/// independent runs (each source seed draws its passes afresh) the actual
/// error over the stated one has a mean square near one: the standard error
/// is the size of the error, neither hiding noise nor inflating it.
#[test]
fn the_standard_error_is_the_size_of_the_error() {
    let (beam, face) = (5.0, 3.1);
    let want = (face / beam) * (face / beam);
    let runs = 40;
    let mut z2 = 0.0;
    for seed in 0..runs {
        let scene = beam_scene(beam, face, 12, 100 + seed);
        let r = run(&scene, r#"{"kind":"detectorStats","passes":8}"#, u64::MAX);
        let f = &r["result"]["detectors"][0]["flux"];
        let (v, se) = (num(&f["value"]), num(&f["stderr"]));
        assert!(se > 0.0, "a partial detector is noisy");
        z2 += ((v - want) / se).powi(2);
    }
    let mean = z2 / runs as f64;
    // Eight passes: (v - want)/se follows Student's t with 7 degrees of
    // freedom, mean square 7/5. The bounds are wide enough for 40 runs.
    assert!((0.4..3.5).contains(&mean), "mean squared z {mean}");
}

#[test]
fn a_request_the_scene_cannot_answer_is_refused() {
    let scene = Scene3::default_scene();
    let err = |req: &str, scene: &Scene3| {
        AnalysisJob::new(scene, cfg(0), request(req)).err().unwrap()
    };
    assert!(err(r#"{"kind":"detectorStats","params":{"detectors":[999]}}"#, &scene).contains("999"));
    assert!(err(r#"{"kind":"detectorStats","passes":3}"#, &scene).contains("divides"));
    let mut dark = scene.clone();
    dark.sources.clear();
    assert!(err(r#"{"kind":"detectorStats"}"#, &dark).contains("no rays"));
    let mut blind = scene.clone();
    blind.detectors.clear();
    assert!(err(r#"{"kind":"detectorStats"}"#, &blind).contains("no detector"));
}

#[test]
fn progress_counts_rays_and_passes() {
    let scene = Scene3::default_scene();
    let mut job = AnalysisJob::new(&scene, cfg(0), request(r#"{"kind":"detectorStats","passes":4}"#)).unwrap();
    let p = job.progress();
    assert_eq!((p.traced, p.passes_done, p.passes), (0, 0, 4));
    job.step(p.rays / 2);
    let half = job.progress();
    assert_eq!((half.traced, half.passes_done), (p.rays / 2, 2));
    while !job.step(1000) {}
    let end = job.progress();
    assert_eq!((end.traced, end.passes_done), (p.rays, 4));
}
