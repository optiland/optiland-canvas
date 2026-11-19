//! `uniformity` with `"sampling": "focused"`: a pilot, then most of the rays
//! on the part of each source's domain that reaches the region.
//!
//! What this suite owes:
//!
//! - focused and forward read the same field within their combined standard
//!   errors, on a beam over a small tile and on a lens imaging a Lambertian
//!   disc onto one;
//! - on those tiles focused reaches a much lower standard error for the
//!   same rays, the pilot's included;
//! - a focused job gives the same JSON however it is stepped;
//! - forward is the default, and a request that says so reads the same to
//!   the bit; a focused job with nothing to cut reads the forward numbers;
//! - a bad request is refused with the reason.

use oc_analysis::{AnalysisJob, AnalysisRequest};
use oc_core::build3::SingletGeom;
use oc_core::scene::Spectrum;
use oc_core::scene3::{AngularDistribution3, SamplingSequence, Scene3, SpatialDistribution3};
use oc_core::vec3::Vec3;
use oc_trace::TraceConfig3;
use serde_json::Value;

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        revision: 3,
        ..TraceConfig3::default()
    }
}

fn job(scene: &Scene3, req: &str) -> Result<AnalysisJob, String> {
    AnalysisJob::new(scene, cfg(), AnalysisRequest::from_json(req)?)
}

fn run(scene: &Scene3, req: &str, batch: u64) -> Value {
    let mut job = job(scene, req).unwrap();
    while !job.step(batch) {
        assert!(job.result().is_none(), "no result before the job is done");
    }
    let p = job.progress();
    assert!(p.traced <= p.rays, "{p:?}");
    job.result().unwrap()
}

fn req(scene: &Scene3, rays: u64, passes: u32, params: &str) -> String {
    format!(
        r#"{{"kind":"uniformity","rays":{rays},"passes":{passes},"params":{{"detector":{},{params}}}}}"#,
        scene.detectors[0].common.id.0
    )
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

fn est(v: &Value) -> (f64, f64) {
    (num(&v["value"]), num(&v["stderr"]))
}

/// A square collimated beam, half-width `beam` mm, stratified, onto a square
/// detector of half-width `face` mm.
fn beam_scene(beam: f64, face: f64, n: u32, seed: u64) -> Scene3 {
    let mut s = Scene3::new();
    s.add_collimated_beam(Vec3::ZERO, beam, beam, n * n, Spectrum::Monochromatic(0.55));
    let src = &mut s.sources[0].samples;
    src.sequence = SamplingSequence::Stratified;
    src.seed = seed;
    s.add_area_detector(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, face, face, [8, 8]);
    s
}

/// A Lambertian disc 3 mm in radius imaged about 1:1 by a singlet 100 mm
/// on, onto a detector at the image.
fn imaging_scene(seed: u64) -> Scene3 {
    let mut s = Scene3::new();
    s.add_collimated_beam(Vec3::ZERO, 1.0, 1.0, 1000, Spectrum::Monochromatic(0.55));
    let src = &mut s.sources[0];
    src.spatial = SpatialDistribution3::Disc { radius_mm: 3.0 };
    src.angular = AngularDistribution3::LambertianHemisphere;
    src.samples.angular_samples = 4;
    src.samples.sequence = SamplingSequence::Stratified;
    src.samples.seed = seed;
    s.add_singlet(
        Vec3::new(100.0, 0.0, 0.0),
        SingletGeom::biconvex(),
        1.5168,
        64.17,
    );
    s.add_area_detector(Vec3::new(200.0, 0.0, 0.0), -Vec3::X, 5.0, 5.0, [8, 8]);
    s
}

/// Forward with `rays`, and focused with the same rays in all: a quarter
/// to the pilot, the rest to the trace.
fn both(scene: &Scene3, rays: u64, params: &str) -> (Value, Value) {
    let fwd = run(scene, &req(scene, rays, 16, params), u64::MAX);
    let foc = run(
        scene,
        &req(
            scene,
            rays - rays / 4,
            16,
            &format!(r#"{params},"sampling":"focused","pilotRays":{}"#, rays / 4),
        ),
        u64::MAX,
    );
    let s = &foc["sampling"];
    assert!(
        num(&s["rays"]) + num(&s["pilotRays"]) <= rays as f64 * 1.001,
        "{s}"
    );
    (fwd, foc)
}

/// The two read the same field: the mean, every cell, the CV and min/max
/// within their combined errors. Returns the factor by which focused's
/// median cell error is below forward's.
fn agree(fwd: &Value, foc: &Value) -> f64 {
    let (a, b) = (&fwd["result"], &foc["result"]);
    let close = |k: &str, tol: f64| {
        let ((x, sx), (y, sy)) = (est(&a[k]), est(&b[k]));
        let z = (x - y) / (sx * sx + sy * sy).sqrt().max(1e-300);
        assert!(
            z.abs() <= tol,
            "{k}: forward {x} ± {sx}, focused {y} ± {sy}"
        );
    };
    close("mean", 3.0);
    close("cv", 3.5);
    close("minMax", 3.5);
    // Over the cells, the squared difference over its variance averages
    // about one.
    let cells = |r: &Value, k: &str| -> Vec<f64> {
        r["irradiance"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(num)
            .collect()
    };
    let (xa, sa, xb, sb) = (
        cells(a, "value"),
        cells(a, "stderr"),
        cells(b, "value"),
        cells(b, "stderr"),
    );
    let n = xa.len() as f64;
    let chi2: f64 = (0..xa.len())
        .map(|c| (xa[c] - xb[c]).powi(2) / (sa[c] * sa[c] + sb[c] * sb[c]))
        .sum::<f64>()
        / n;
    assert!(chi2 < 2.5, "mean squared cell z {chi2}");
    num(&a["medianCellRelStderr"]) / num(&b["medianCellRelStderr"])
}

#[test]
fn a_beam_on_a_small_tile_reads_the_same_with_fewer_rays() {
    let scene = beam_scene(12.0, 10.0, 60, 3);
    let (fwd, foc) = both(
        &scene,
        262_144,
        r#""cells":[4,4],"regionMm":[[2,3],[-1,0]]"#,
    );
    let gain = agree(&fwd, &foc);
    eprintln!(
        "beam: cell stderr {gain:.2}x lower, {:.1}x fewer rays; {}",
        gain * gain,
        foc["sampling"]
    );
    assert!(gain > 3.0, "cell stderr only {gain}x lower");
    // The beam's irradiance is known: 1/576 per mm² in every cell.
    let b = &foc["result"];
    let want = 1.0 / 576.0;
    let (mean, se) = est(&b["mean"]);
    assert!(
        (mean - want).abs() <= 3.0 * se,
        "mean {mean} ± {se}, want {want}"
    );
    let v = b["irradiance"]["value"].as_array().unwrap();
    let e = b["irradiance"]["stderr"].as_array().unwrap();
    let chi2 = v
        .iter()
        .zip(e)
        .map(|(v, e)| ((num(v) - want) / num(e)).powi(2))
        .sum::<f64>()
        / v.len() as f64;
    assert!(
        chi2 < 2.5,
        "mean squared cell z {chi2} about the known irradiance"
    );
    let f = &foc["sampling"]["focus"][0];
    assert_eq!(f["focused"], true);
    assert_eq!(
        f["angular"],
        Value::Null,
        "a collimated beam has no directions to cut"
    );
    assert!(num(&f["domainShare"]) < 0.05, "{f}");
}

#[test]
fn an_imaging_tile_reads_the_same_with_fewer_rays() {
    let scene = imaging_scene(1);
    let (fwd, foc) = both(
        &scene,
        524_288,
        r#""cells":[4,4],"regionMm":[[0.5,2.5],[0.5,2.5]]"#,
    );
    let gain = agree(&fwd, &foc);
    eprintln!(
        "imaging: cell stderr {gain:.2}x lower, {:.1}x fewer rays; {}",
        gain * gain,
        foc["sampling"]
    );
    assert!(gain > 2.0, "cell stderr only {gain}x lower");
    let f = &foc["sampling"]["focus"][0];
    assert_eq!(f["focused"], true);
    assert!(num(&f["domainShare"]) < 0.1, "{f}");
}

#[test]
fn stepped_in_any_batches_it_gives_the_same_json() {
    let scene = beam_scene(12.0, 10.0, 30, 2);
    for params in [
        r#""cells":[4,4],"regionMm":[[2,4],[-1,1]],"sampling":"focused","pilotRays":16384"#,
        r#""cells":[4,4],"regionMm":[[2,4],[-1,1]],"sampling":"focused","pilotRays":16384,"target":{"metric":"cv","stderr":0.5}"#,
        // A pilot too small to see the tile: the job traces forward.
        r#""cells":[4,4],"regionMm":[[2,2.5],[-1,-0.5]],"sampling":"focused","pilotRays":4096"#,
    ] {
        let r = req(&scene, 32_000, 16, params);
        let whole = serde_json::to_string(&run(&scene, &r, u64::MAX)).unwrap();
        for batch in [1, 37, 1000] {
            let stepped = serde_json::to_string(&run(&scene, &r, batch)).unwrap();
            assert_eq!(stepped, whole, "{params}, batches of {batch}");
        }
        // Uneven steps, the last one asking for every ray.
        let mut j = job(&scene, &r).unwrap();
        j.step(5);
        j.step(4099);
        j.step(u64::MAX);
        assert!(j.done());
        assert_eq!(
            serde_json::to_string(&j.result().unwrap()).unwrap(),
            whole,
            "{params}, uneven"
        );
    }
}

#[test]
fn forward_is_the_default_and_a_focused_job_with_nothing_to_cut_reads_forward() {
    let scene = beam_scene(5.0, 8.0, 24, 1);
    let plain = run(&scene, &req(&scene, 9000, 16, r#""cells":[5,4]"#), 1000);
    let said = run(
        &scene,
        &req(&scene, 9000, 16, r#""cells":[5,4],"sampling":"forward""#),
        1000,
    );
    assert_eq!(
        serde_json::to_string(&said).unwrap(),
        serde_json::to_string(&plain).unwrap()
    );
    assert_eq!(
        plain["sampling"].as_object().unwrap().len(),
        4,
        "{}",
        plain["sampling"]
    );

    // A point aimed along the axis: every ray the same, nothing to cut.
    let mut point = scene.clone();
    point.sources[0].spatial = SpatialDistribution3::Point;
    let p = run(&point, &req(&point, 9000, 16, r#""cells":[5,4]"#), u64::MAX);
    let f = run(
        &point,
        &req(&point, 9000, 16, r#""cells":[5,4],"sampling":"focused""#),
        u64::MAX,
    );
    assert_eq!(f["result"], p["result"]);
    let s = &f["sampling"];
    assert_eq!(
        (s["mode"].as_str(), s["pilotRays"].as_u64()),
        (Some("focused"), Some(0))
    );
    assert_eq!(s["focus"][0]["reason"], "nothing to cut");
}

#[test]
fn a_bad_request_is_refused_with_its_reason() {
    let scene = beam_scene(5.0, 8.0, 10, 1);
    let err = |params: &str| job(&scene, &req(&scene, 1000, 8, params)).err().unwrap();
    assert!(err(r#""cells":[4,4],"sampling":"backward""#).contains("\"forward\" or \"focused\""));
    assert!(err(r#""cells":[4,4],"pilotRays":8192"#).contains("\"focused\" alone"));
    assert!(
        err(r#""cells":[4,4],"sampling":"forward","pilotRays":8192"#).contains("\"focused\" alone")
    );
    assert!(err(r#""cells":[4,4],"sampling":"focused","pilotRays":100"#).contains("from 4096"));
    assert!(
        err(r#""cells":[4,4],"sampling":"focused","pilotRays":1000000000000"#)
            .contains("pilotRays")
    );
    assert!(
        err(r#""cells":[4,4],"sampling":"focused","pilotRays":2.5"#).contains("uniformity params")
    );
    assert!(err(r#""cells":[4,4],"sampling":"focused","pilot":8192"#).contains("unknown field"));
    assert!(err(r#""cells":[4,4],"sampling":true"#).contains("uniformity params"));
}
