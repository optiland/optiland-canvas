//! `uniformity`: how even the light is over a region of a detector.
//!
//! What this suite owes:
//!
//! - the same request gives the same JSON however the job is stepped, with a
//!   noise target or without, stopped early or at the ray limit;
//! - each cell's pooled flux is the kernel's readout of the same rays;
//! - a uniform field reads a CV of 0 and a min/max of 1 within their standard
//!   errors, and the errors fall as the rays grow;
//! - a field of known shape reads its known CV, min/max and falloff;
//! - over many runs the CV's standard error is the size of its error;
//! - a loose target stops early and says so, an impossible one runs to the
//!   ray limit and says so;
//! - a request the scene cannot answer is refused with the reason.

use oc_analysis::{AnalysisJob, AnalysisRequest};
use oc_core::scene::Spectrum;
use oc_core::scene3::{SamplingSequence, Scene3};
use oc_core::vec3::Vec3;
use oc_trace::{TraceConfig3, TraceJob3};
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
    job.result().unwrap()
}

/// A square collimated beam, half-width `beam` mm, sampled on a stratified
/// grid of `n` × `n`, onto a square detector of half-width `face` mm.
fn beam_scene(beam: f64, face: f64, n: u32, seed: u64) -> Scene3 {
    let mut s = Scene3::new();
    s.add_collimated_beam(Vec3::ZERO, beam, beam, n * n, Spectrum::Monochromatic(0.55));
    let src = &mut s.sources[0].samples;
    src.sequence = SamplingSequence::Stratified;
    src.seed = seed;
    s.add_area_detector(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, face, face, [8, 8]);
    s
}

fn det(scene: &Scene3) -> u64 {
    scene.detectors[0].common.id.0
}

fn req(scene: &Scene3, rays: u64, passes: u32, params: &str) -> String {
    format!(
        r#"{{"kind":"uniformity","rays":{rays},"passes":{passes},"params":{{"detector":{},{params}}}}}"#,
        det(scene)
    )
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// `(value, stderr)` of an estimate.
fn est(v: &Value) -> (f64, f64) {
    (num(&v["value"]), num(&v["stderr"]))
}

#[test]
fn stepped_in_any_batches_it_gives_the_same_json() {
    let scene = beam_scene(5.0, 8.0, 24, 1);
    for params in [
        r#""cells":[5,4]"#,
        r#""cells":[4,3],"regionMm":[[0,8],[-5,5]]"#,
        r#""cells":[6,6],"target":{"metric":"cv","stderr":0.5}"#,
        r#""cells":[6,6],"target":{"metric":"minMax","stderr":1e-12}"#,
    ] {
        let r = req(&scene, 9000, 16, params);
        let whole = serde_json::to_string(&run(&scene, &r, u64::MAX)).unwrap();
        for batch in [1, 37, 1000] {
            let stepped = serde_json::to_string(&run(&scene, &r, batch)).unwrap();
            assert_eq!(stepped, whole, "{params}, batches of {batch}");
        }
    }
}

/// With no region the cells are the detector's grid: each cell's flux is the
/// kernel's own pixel, to the bit.
#[test]
fn each_cell_is_the_kernels_pixel() {
    let scene = beam_scene(5.0, 8.0, 24, 1);
    let r = run(&scene, &req(&scene, 9000, 4, r#""cells":[5,4]"#), u64::MAX);
    let mut grid = scene.clone();
    grid.detectors[0].bins = [5, 4];
    let mut traced = oc_analysis::sampled_for(&grid, 9000 / 4);
    traced.detectors[0].measure = Default::default();
    let mut kernel = TraceJob3::with_passes(traced, 4, None, cfg());
    while !kernel.step(usize::MAX) {}
    let t = kernel.finish();
    let d = &t.detectors[0];
    let cells = r["result"]["irradiance"]["value"].as_array().unwrap();
    for (c, v) in cells.iter().enumerate() {
        let want = d.incident_flux[c] / d.covered_area_mm2[c];
        assert_eq!(num(v).to_bits(), want.to_bits(), "cell {c}");
    }
    assert_eq!(r["result"]["hits"].as_u64().unwrap(), d.hits);
    assert_eq!(r["result"]["stop"], Value::Null);
}

/// A beam wider than the face lights it evenly: the CV is 0 and min/max is 1
/// within their standard errors, the mean is the beam's irradiance, and four
/// times the rays bring every error down by about half or more (the sampling
/// is stratified, which converges faster than 1/√rays).
#[test]
fn a_uniform_field_reads_even_and_the_noise_falls_with_the_rays() {
    let scene = beam_scene(12.0, 10.0, 40, 1);
    let read = |rays: u64| run(&scene, &req(&scene, rays, 16, r#""cells":[8,8]"#), u64::MAX);
    let (small, big) = (read(40_000), read(160_000));
    for r in [&small, &big] {
        let b = &r["result"];
        let (cv, se) = est(&b["cv"]);
        assert!(se > 0.0 && cv <= 3.0 * se, "cv {cv} ± {se}");
        let (mm, se) = est(&b["minMax"]);
        assert!(
            se > 0.0 && (mm - 1.0).abs() <= 3.0 * se,
            "min/max {mm} ± {se}"
        );
        let (mean, se) = est(&b["mean"]);
        let want = 1.0 / (24.0 * 24.0);
        assert!(
            (mean - want).abs() <= 4.0 * se + 1e-12,
            "mean {mean} ± {se}, want {want}"
        );
        for k in ["edge", "corner"] {
            let (f, se) = est(&b["falloff"][k]);
            assert!((f - 1.0).abs() <= 3.0 * se, "{k} falloff {f} ± {se}");
        }
        assert_eq!(
            (b["cellsDark"].as_u64(), b["cellsSparse"].as_u64()),
            (Some(0), Some(0))
        );
    }
    let shrink = |k: &str| {
        let a = small["result"][k]["stderr"].as_f64().unwrap();
        let b = big["result"][k]["stderr"].as_f64().unwrap();
        b / a
    };
    for k in ["minMax", "mean"] {
        let s = shrink(k);
        assert!((0.15..0.7).contains(&s), "{k} stderr ×{s} for 4× the rays");
    }
    let cell =
        num(&big["result"]["medianCellRelStderr"]) / num(&small["result"]["medianCellRelStderr"]);
    assert!(
        (0.15..0.7).contains(&cell),
        "cell stderr ×{cell} for 4× the rays"
    );
    assert_eq!(small["sampling"]["rays"], 40_000);
}

/// A beam over half the region: half the cells at the beam's irradiance, half
/// dark. CV 1, min/max 0 with no noise, the mean half the beam's, and the
/// edge and corners as bright as the centre (each half lit).
#[test]
fn a_half_lit_region_reads_its_known_shape() {
    let scene = beam_scene(5.0, 10.0, 40, 1);
    let r = run(
        &scene,
        &req(
            &scene,
            64_000,
            16,
            r#""cells":[4,3],"regionMm":[[0,10],[-5,5]]"#,
        ),
        u64::MAX,
    );
    let b = &r["result"];
    assert_eq!(b["regionMm"], serde_json::json!([[0.0, 10.0], [-5.0, 5.0]]));
    let (cv, se) = est(&b["cv"]);
    assert!((cv - 1.0).abs() <= 3.0 * se && se < 1e-3, "cv {cv} ± {se}");
    assert_eq!(
        b["minMax"],
        serde_json::json!({"value": 0.0, "stderr": 0.0})
    );
    let (mean, se) = est(&b["mean"]);
    assert!((mean - 0.005).abs() <= 4.0 * se, "mean {mean} ± {se}");
    for k in ["edge", "corner"] {
        let (f, se) = est(&b["falloff"][k]);
        assert!((f - 1.0).abs() <= 3.0 * se, "{k} falloff {f} ± {se}");
    }
    assert_eq!(b["cellsDark"], 6);
    assert_eq!(b["cellsSparse"], 0);
    let lit = b["irradiance"]["litPasses"].as_array().unwrap();
    assert!(lit.iter().all(|l| l == 0 || l == 16), "{lit:?}");
}

/// Three columns, the beam's edge through the middle one: irradiance 1, ½ and
/// 0 of the beam's, a CV of √(2/3). Over many runs the CV's error over its
/// stated standard error has a mean square near one.
#[test]
fn the_cv_standard_error_is_the_size_of_its_error() {
    let want = (2.0f64 / 3.0).sqrt();
    let runs = 30;
    let mut z2 = 0.0;
    for seed in 0..runs {
        let scene = beam_scene(5.0, 10.0, 14, 200 + seed);
        let r = run(
            &scene,
            &req(
                &scene,
                6_400,
                8,
                r#""cells":[3,1],"regionMm":[[0,10],[-5,5]]"#,
            ),
            u64::MAX,
        );
        let (cv, se) = est(&r["result"]["cv"]);
        assert!(se > 0.0, "an edge through a cell is noisy");
        z2 += ((cv - want) / se).powi(2);
    }
    let mean = z2 / runs as f64;
    assert!((0.3..3.5).contains(&mean), "mean squared z {mean}");
}

/// A loose target is met at the first check (8 passes) and the job stops; an
/// impossible one runs every ray and says it fell short. Either way the
/// result says how far it got, and stepping does not move it.
#[test]
fn a_noise_target_stops_the_job_or_states_it_was_not_reached() {
    let scene = beam_scene(12.0, 10.0, 40, 1);
    let loose = req(
        &scene,
        256_000,
        64,
        r#""cells":[8,8],"target":{"metric":"cv","stderr":0.05}"#,
    );
    let r = run(&scene, &loose, u64::MAX);
    let stop = &r["result"]["stop"];
    assert_eq!(
        (stop["metric"].as_str(), stop["reached"].as_bool()),
        (Some("cv"), Some(true))
    );
    assert_eq!(
        (
            stop["passes"].as_u64(),
            stop["rays"].as_u64(),
            stop["rayLimit"].as_u64()
        ),
        (Some(8), Some(32_000), Some(256_000))
    );
    assert!(num(&stop["stderr"]) <= 0.05);
    assert_eq!(
        num(&stop["stderr"]),
        num(&r["result"]["cv"]["stderr"]),
        "the noise stated is the metric's"
    );
    assert_eq!(
        (
            r["sampling"]["rays"].as_u64(),
            r["sampling"]["passes"].as_u64()
        ),
        (Some(32_000), Some(8))
    );
    let (mean, se) = est(&r["result"]["mean"]);
    assert!(
        (mean - 1.0 / 576.0).abs() <= 4.0 * se,
        "an early stop still reads the whole flux: {mean} ± {se}"
    );

    let tight = req(
        &scene,
        64_000,
        16,
        r#""cells":[8,8],"target":{"metric":"minMax","stderr":1e-9}"#,
    );
    let r2 = run(&scene, &tight, u64::MAX);
    let stop = &r2["result"]["stop"];
    assert_eq!(stop["reached"], false);
    assert_eq!(
        (stop["passes"].as_u64(), stop["rays"].as_u64()),
        (Some(16), Some(64_000))
    );
    assert!(num(&stop["stderr"]) > 1e-9);

    for (r, text) in [(&r, &loose), (&r2, &tight)] {
        let whole = serde_json::to_string(r).unwrap();
        for batch in [7, 4999] {
            assert_eq!(
                serde_json::to_string(&run(&scene, text, batch)).unwrap(),
                whole,
                "batches of {batch}"
            );
        }
    }

    let mut early = job(&scene, &loose).unwrap();
    while !early.step(10_000) {}
    let p = early.progress();
    assert_eq!((p.traced, p.passes_done, p.rays), (32_000, 8, 256_000));
}

/// A region on a fine pixel grid: a tile a twentieth of the face, 4 cells
/// across, read as the kernel's pixels on a grid snapped to it.
#[test]
fn a_small_tile_is_read_on_a_grid_snapped_to_it() {
    let scene = beam_scene(12.0, 10.0, 60, 3);
    let r = run(
        &scene,
        &req(
            &scene,
            64_000,
            16,
            r#""cells":[4,4],"regionMm":[[2,3],[-1,0]]"#,
        ),
        u64::MAX,
    );
    let b = &r["result"];
    let reg = &b["regionMm"];
    for (k, want) in [[2.0, 3.0], [-1.0, 0.0]].iter().enumerate() {
        for e in 0..2 {
            assert!((num(&reg[k][e]) - want[e]).abs() < 1e-12, "{reg}");
        }
    }
    assert_eq!(b["cellSizeMm"][0], 0.25);
    let (mean, se) = est(&b["mean"]);
    assert!((mean - 1.0 / 576.0).abs() <= 4.0 * se, "{mean} ± {se}");
}

/// Few hits are not hidden: a sparse field marks the cells some pass missed.
#[test]
fn cells_some_pass_missed_are_counted_as_sparse() {
    let scene = beam_scene(12.0, 10.0, 6, 1);
    let r = run(
        &scene,
        &req(&scene, 640, 16, r#""cells":[16,16]"#),
        u64::MAX,
    );
    let b = &r["result"];
    let sparse = b["cellsSparse"].as_u64().unwrap() + b["cellsDark"].as_u64().unwrap();
    assert!(sparse > 100, "{sparse} of 256 cells some pass missed");
    let lit = b["irradiance"]["litPasses"].as_array().unwrap();
    assert!(lit.iter().any(|l| num(l) < 16.0));
}

#[test]
fn a_bad_request_is_refused_with_its_reason() {
    let scene = beam_scene(5.0, 8.0, 10, 1);
    let err = |params: &str, passes: u32| {
        job(&scene, &req(&scene, 1000, passes, params))
            .err()
            .unwrap()
    };
    assert!(err(r#""cells":[4,4]"#, 2).contains("at least 4"));
    assert!(err(r#""cells":[0,4]"#, 8).contains("cells"));
    assert!(err(r#""cells":[300,300]"#, 8).contains("65536"));
    assert!(err(r#""cells":[4,4],"regionMm":[[-9,0],[0,1]]"#, 8).contains("off the face"));
    assert!(err(r#""cells":[4,4],"regionMm":[[1,0],[0,1]]"#, 8).contains("low < high"));
    assert!(
        err(r#""cells":[4,4],"target":{"metric":"rms","stderr":0.1}"#, 8)
            .contains("\"cv\" or \"minMax\"")
    );
    assert!(err(r#""cells":[4,4],"target":{"metric":"cv","stderr":0}"#, 8).contains("above 0"));
    assert!(err(r#""cells":[4,4],"target":{"metric":"cv"}"#, 8).contains("stderr"));
    let missing = r#"{"kind":"uniformity","params":{"detector":999,"cells":[4,4]}}"#;
    assert!(job(&scene, missing)
        .err()
        .unwrap()
        .contains("no detector has id 999"));

    // A disc's grid is the square around it: a region in its corner has no
    // cell on the face.
    let mut disc = scene.clone();
    disc.detectors[0].geometry = oc_core::scene3::DetectorGeometry3::Disc { radius_mm: 8.0 };
    let corner = format!(
        r#"{{"kind":"uniformity","params":{{"detector":{},"cells":[2,2],"regionMm":[[7.5,8],[7.5,8]]}}}}"#,
        det(&disc)
    );
    assert!(job(&disc, &corner)
        .err()
        .unwrap()
        .contains("off the detector's face"));
}
