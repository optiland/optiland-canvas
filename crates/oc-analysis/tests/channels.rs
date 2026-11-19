//! `channels`: each source traced alone, and per channel its signal, its
//! leak and the crosstalk table.
//!
//! What this suite owes:
//!
//! - a job gives the same result to the bit however it is stepped;
//! - each crosstalk entry is the kernel's readout of that source traced
//!   alone, to the bit;
//! - where geometry or an ideal filter keeps a source's light off a detector,
//!   the leak is zero with no noise; where a filter passes both, the leak is
//!   the known share;
//! - across many independent runs the standard errors of the signal, the leak
//!   and their ratio are the size of the actual errors;
//! - a request the scene cannot answer is refused with the reason.

use oc_analysis::{sampled_for, source_alone, AnalysisJob, AnalysisRequest};
use oc_core::build3::DichroicSpec;
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
    while !job.step(batch) {
        assert!(job.result().is_none(), "no result before the last ray");
    }
    job.result().unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// A square collimated beam along +X from `pos`, half-width `half` mm, on a
/// stratified `n` × `n` grid. Returns its source id.
fn beam(s: &mut Scene3, pos: Vec3, half: f64, n: u32, lambda_um: f64, seed: u64) -> u64 {
    let id = s.add_collimated_beam(pos, half, half, n * n, Spectrum::Monochromatic(lambda_um));
    let src = &mut s.sources.last_mut().unwrap().samples;
    src.sequence = SamplingSequence::Stratified;
    src.seed = seed;
    id.0
}

/// An ideal edge filter: transmits all on one side of `edge_nm`, absorbs the
/// rest, reflects nothing.
fn edge(edge_nm: f64, long_pass: bool) -> DichroicSpec {
    DichroicSpec {
        edge_nm,
        width_nm: 0.5,
        long_pass,
        r_max: 0.0,
        t_max: 1.0,
    }
}

/// Two beams side by side, 20 mm apart, each onto its own detector: the
/// geometry keeps each source off the other's detector.
fn side_by_side() -> (Scene3, [u64; 2], [u64; 2]) {
    let mut s = Scene3::new();
    let a = beam(&mut s, Vec3::new(0.0, 10.0, 0.0), 2.0, 8, 0.45, 1);
    let b = beam(&mut s, Vec3::new(0.0, -10.0, 0.0), 2.0, 8, 0.60, 2);
    let da = s
        .add_area_detector(Vec3::new(50.0, 10.0, 0.0), -Vec3::X, 4.0, 4.0, [4, 4])
        .0;
    let db = s
        .add_area_detector(Vec3::new(50.0, -10.0, 0.0), -Vec3::X, 4.0, 4.0, [4, 4])
        .0;
    (s, [a, b], [da, db])
}

/// A blue (0.45 µm) and a red (0.60 µm) beam on one axis, split 50/50. The
/// transmitted half crosses an ideal short-pass at 500 nm to detector 1; the
/// reflected half crosses an ideal long-pass at 400 nm, which passes both, to
/// detector 2.
fn split_and_filtered() -> (Scene3, [u64; 2], [u64; 2]) {
    let mut s = Scene3::new();
    let blue = beam(&mut s, Vec3::ZERO, 2.0, 8, 0.45, 1);
    let red = beam(&mut s, Vec3::ZERO, 2.0, 8, 0.60, 2);
    s.add_beam_splitter(Vec3::new(20.0, 0.0, 0.0), 10.0, 45.0, 0.5, 0.5)
        .unwrap();
    s.add_dichroic(Vec3::new(35.0, 0.0, 0.0), 10.0, 0.0, edge(500.0, false))
        .unwrap();
    let d1 = s
        .add_area_detector(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 5.0, 5.0, [4, 4])
        .0;
    s.add_dichroic(Vec3::new(20.0, -15.0, 0.0), 10.0, 90.0, edge(400.0, true))
        .unwrap();
    let d2 = s
        .add_area_detector(Vec3::new(20.0, -30.0, 0.0), Vec3::Y, 5.0, 5.0, [4, 4])
        .0;
    (s, [blue, red], [d1, d2])
}

fn channel<'a>(r: &'a Value, name: &str) -> &'a Value {
    r["result"]["channels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no channel {name}"))
}

/// `{value, stderr}` is `want` with no noise.
fn exact(e: &Value, want: f64, what: &str) {
    assert!(
        (num(&e["value"]) - want).abs() < 1e-12,
        "{what}: {e}, want {want}"
    );
    assert!(num(&e["stderr"]) < 1e-12, "{what}: {e} has noise");
}

fn split_request(scene: &(Scene3, [u64; 2], [u64; 2]), passes: u32) -> String {
    let (_, [blue, red], [d1, d2]) = scene;
    format!(
        r#"{{"kind":"channels","passes":{passes},"params":{{"channels":[
            {{"name":"blue","detector":{d1},"sources":[{blue}]}},
            {{"name":"red","detector":{d2},"sources":[{red}]}},
            {{"name":"red band","detector":{d2},"sources":[{red}],"bandUm":[0.55,0.65]}},
            {{"name":"blue band","detector":{d2},"sources":[{blue}],"bandUm":[0.4,0.5]}}]}}}}"#
    )
}

#[test]
fn stepped_one_ray_at_a_time_is_stepped_all_at_once() {
    let scene = split_and_filtered();
    let req = split_request(&scene, 8);
    let whole = serde_json::to_string(&run(&scene.0, &req, u64::MAX)).unwrap();
    for batch in [1, 37, 1000] {
        let stepped = serde_json::to_string(&run(&scene.0, &req, batch)).unwrap();
        assert_eq!(stepped, whole, "batches of {batch}");
    }
}

/// Each entry of the table is what the kernel's own job gives for that
/// source traced alone, at the same rays, to the bit; a channel's signal and
/// leak are sums of entries.
#[test]
fn each_entry_is_the_kernels_readout_of_its_source_alone() {
    let (scene, _, _) = split_and_filtered();
    let req = r#"{"kind":"channels","rays":3000,"passes":4}"#;
    let r = run(&scene, req, u64::MAX);
    let traced = sampled_for(&scene, 3000u64.div_ceil(4));
    let table = &r["result"]["crosstalk"]["flux"];
    for s in 0..2 {
        let mut job = TraceJob3::with_passes(source_alone(&traced, s), 4, None, cfg(3));
        while !job.step(usize::MAX) {}
        let t = job.finish();
        for d in 0..2 {
            let v = num(&table[s][d]["value"]);
            assert_eq!(
                v.to_bits(),
                t.detectors[d].total_incident.to_bits(),
                "source {s} detector {d}"
            );
        }
        let src = &r["result"]["sources"][s];
        assert_eq!(src["source"], traced.sources[s].id.0);
        assert_eq!(num(&src["launched"]).to_bits(), t.ledger.launched.to_bits());
    }
    // No band, every source the channel's own: the signal is the column sum.
    let c = &r["result"]["channels"][1];
    let want = num(&table[0][1]["value"]) + num(&table[1][1]["value"]);
    assert_eq!(num(&c["signal"]["value"]).to_bits(), want.to_bits());
    exact(&c["leak"], 0.0, "leak with every source own");
}

/// The rays are split across the sources as each source's own count says:
/// every source sampled by the same factor.
#[test]
fn the_rays_are_split_across_the_sources() {
    let (mut scene, _, _) = side_by_side();
    // The second source emits four times the rays of the first.
    scene.sources[1].samples.spatial_samples = 4 * 64;
    let r = run(
        &scene,
        r#"{"kind":"channels","rays":16000,"passes":8}"#,
        u64::MAX,
    );
    let rays: Vec<u64> = r["result"]["sources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["rays"].as_u64().unwrap())
        .collect();
    assert_eq!(
        rays.iter().sum::<u64>(),
        r["sampling"]["rays"].as_u64().unwrap()
    );
    assert_eq!(rays, vec![400 * 8, 1600 * 8]);
    assert_eq!(r["sampling"]["raysPerPass"], 2000);
}

#[test]
fn geometry_that_keeps_a_source_off_a_detector_reads_no_leak() {
    let (scene, [a, b], [da, db]) = side_by_side();
    let req = format!(
        r#"{{"kind":"channels","passes":8,"params":{{"channels":[
            {{"name":"A","detector":{da},"sources":[{a}]}},{{"name":"B","detector":{db},"sources":[{b}]}}]}}}}"#
    );
    let r = run(&scene, &req, u64::MAX);
    for name in ["A", "B"] {
        let c = channel(&r, name);
        exact(&c["signal"], 1.0, name);
        exact(&c["leak"], 0.0, name);
        exact(&c["ratio"], 0.0, name);
    }
    let t = &r["result"]["crosstalk"];
    assert_eq!(t["sources"], serde_json::json!([a, b]));
    assert_eq!(t["channels"], serde_json::json!(["A", "B"]));
    exact(&t["flux"][0][0], 1.0, "a on A");
    exact(&t["flux"][0][1], 0.0, "a on B");
    exact(&t["flux"][1][0], 0.0, "b on A");
    exact(&t["flux"][1][1], 1.0, "b on B");
}

/// With no params: one channel per detector, seeing every source, no band.
#[test]
fn the_default_is_a_channel_per_detector() {
    let (scene, [a, b], [da, db]) = side_by_side();
    let r = run(&scene, r#"{"kind":"channels","passes":4}"#, u64::MAX);
    let chans = r["result"]["channels"].as_array().unwrap();
    assert_eq!(chans.len(), 2);
    for (c, d) in chans.iter().zip([da, db]) {
        assert_eq!(c["detector"], d);
        assert_eq!(c["name"], format!("detector {d}"));
        assert_eq!(c["sources"], serde_json::json!([a, b]));
        assert_eq!(c["bandUm"], Value::Null);
        exact(&c["signal"], 1.0, "signal");
        exact(&c["leak"], 0.0, "leak");
    }
}

/// The short-pass keeps the red beam off detector 1: no leak. The long-pass
/// at 400 nm passes both: the red channel on detector 2 leaks the blue half
/// beam, a ratio of one. A band on the detected light takes the leak out
/// again.
#[test]
fn a_filter_that_passes_both_leaks_the_known_share() {
    let scene = split_and_filtered();
    let r = run(&scene.0, &split_request(&scene, 8), u64::MAX);
    let blue = channel(&r, "blue");
    exact(&blue["signal"], 0.5, "blue signal");
    exact(&blue["leak"], 0.0, "blue leak");
    let red = channel(&r, "red");
    exact(&red["signal"], 0.5, "red signal");
    exact(&red["leak"], 0.5, "red leak");
    exact(&red["ratio"], 1.0, "red ratio");
    let band = channel(&r, "red band");
    assert_eq!(band["bandUm"], serde_json::json!([0.55, 0.65]));
    exact(&band["signal"], 0.5, "red band signal");
    exact(&band["leak"], 0.0, "red band leak");
    let blue_band = channel(&r, "blue band");
    exact(&blue_band["signal"], 0.5, "blue band signal");
    exact(&blue_band["leak"], 0.0, "blue band leak");
    // The table: source by channel, in band.
    let t = &r["result"]["crosstalk"]["flux"];
    let want = [[0.5, 0.5, 0.0, 0.5], [0.0, 0.5, 0.5, 0.0]];
    for (s, row) in want.iter().enumerate() {
        for (c, w) in row.iter().enumerate() {
            exact(&t[s][c], *w, &format!("source {s} channel {c}"));
        }
    }
}

/// Two beams on one axis, half-widths 4 and 5 mm, onto a detector of
/// half-width 3.1 mm: it catches (3.1/4)² of the first, its signal, and
/// (3.1/5)² of the second, its leak; their ratio is 16/25. Over many
/// independent runs the actual error over the stated one has a mean square
/// near one for all three.
#[test]
fn the_standard_errors_are_the_size_of_the_errors() {
    let face = 3.1;
    let want = [
        (face / 4.0f64).powi(2),
        (face / 5.0f64).powi(2),
        16.0 / 25.0,
    ];
    let runs = 40;
    let mut z2 = [0.0; 3];
    for seed in 0..runs {
        let mut s = Scene3::new();
        let a = beam(&mut s, Vec3::ZERO, 4.0, 12, 0.45, 100 + seed);
        beam(&mut s, Vec3::ZERO, 5.0, 12, 0.60, 500 + seed);
        let d = s
            .add_area_detector(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, face, face, [4, 4])
            .0;
        let req = format!(
            r#"{{"kind":"channels","passes":8,"params":{{"channels":[{{"detector":{d},"sources":[{a}]}}]}}}}"#
        );
        let r = run(&s, &req, u64::MAX);
        let c = &r["result"]["channels"][0];
        for (k, f) in ["signal", "leak", "ratio"].iter().enumerate() {
            let (v, se) = (num(&c[f]["value"]), num(&c[f]["stderr"]));
            assert!(se > 0.0, "{f} is noisy");
            z2[k] += ((v - want[k]) / se).powi(2);
        }
    }
    for (k, f) in ["signal", "leak", "ratio"].iter().enumerate() {
        let mean = z2[k] / runs as f64;
        // Eight passes: Student's t with 7 degrees of freedom, mean square
        // 7/5; the bounds are wide enough for 40 runs.
        assert!((0.4..3.5).contains(&mean), "{f}: mean squared z {mean}");
    }
}

#[test]
fn a_request_the_scene_cannot_answer_is_refused() {
    let (scene, [a, _], [d1, _]) = split_and_filtered();
    let err = |params: &str| {
        let req = format!(r#"{{"kind":"channels","params":{params}}}"#);
        AnalysisJob::new(&scene, cfg(0), request(&req))
            .err()
            .unwrap()
    };
    let ch = |body: &str| err(&format!(r#"{{"channels":[{body}]}}"#));
    assert!(ch(r#"{"detector":999}"#).contains("no detector has id 999"));
    assert!(ch(&format!(r#"{{"detector":{d1},"sources":[999]}}"#)).contains("no source has id 999"));
    assert!(ch(&format!(r#"{{"detector":{d1},"sources":[]}}"#)).contains("sees no source"));
    assert!(ch(&format!(r#"{{"detector":{d1},"sources":[{a},{a}]}}"#)).contains("twice"));
    for band in ["[0.6,0.5]", "[0.5,0.5]", "[0,0.5]", "[-0.1,0.5]"] {
        let e = ch(&format!(r#"{{"detector":{d1},"bandUm":{band}}}"#));
        assert!(e.contains("bandUm"), "{band}: {e}");
    }
    assert!(ch(&format!(
        r#"{{"name":"x","detector":{d1}}},{{"name":"x","detector":{d1}}}"#
    ))
    .contains("two channels"));
    assert!(ch(&format!(r#"{{"name":" ","detector":{d1}}}"#)).contains("empty"));
    assert!(err(r#"{"channels":[]}"#).contains("at least one channel"));
    let passes = AnalysisJob::new(&scene, cfg(0), request(r#"{"kind":"channels","passes":3}"#))
        .err()
        .unwrap();
    assert!(passes.contains("divides"), "{passes}");
    let mut blind = scene.clone();
    blind.detectors.clear();
    let e = AnalysisJob::new(&blind, cfg(0), request(r#"{"kind":"channels"}"#))
        .err()
        .unwrap();
    assert!(e.contains("no detector"), "{e}");
}

#[test]
fn progress_counts_rays_over_every_source() {
    let (scene, _, _) = split_and_filtered();
    let mut job =
        AnalysisJob::new(&scene, cfg(0), request(r#"{"kind":"channels","passes":4}"#)).unwrap();
    let p = job.progress();
    // Two sources of 64 rays, four passes each.
    assert_eq!((p.traced, p.rays, p.passes_done, p.passes), (0, 512, 0, 4));
    job.step(256);
    let half = job.progress();
    assert_eq!((half.traced, half.passes_done), (256, 2));
    while !job.step(100) {}
    let end = job.progress();
    assert_eq!((end.traced, end.passes_done), (512, 4));
    assert_eq!(job.result().unwrap()["kind"], "channels");
}
