//! `budget`: where the light went, in all and part by part.
//!
//! What this suite owes:
//!
//! - a job gives the same result to the bit however it is stepped, whole or
//!   by source;
//! - the sinks add up to the launched flux, to rounding, and so do the parts;
//!   the totals are the kernel's own ledger, to the bit;
//! - the parts' values are the kernel's own object ledger, to the bit, each
//!   with a standard error from the passes;
//! - an ideal absorbing filter of known transmission reads its absorption
//!   and the detected rest, with no noise, part by part too; a beam partly
//!   outside a stop reads the share the stop blocks, with noise; a beam that
//!   misses everything escapes whole;
//! - a request the scene cannot answer is refused with the reason.

use oc_analysis::{sampled_for, AnalysisJob, AnalysisRequest};
use oc_core::build3::SingletGeom;
use oc_core::scene::Spectrum;
use oc_core::scene3::{AbsorptionModel, SamplingSequence, Scene3};
use oc_core::vec3::Vec3;
use oc_trace::{ObjectLedger3, ObjectRef3, TraceConfig3, TraceJob3};
use serde_json::Value;

/// The rounding the sinks may leave against the launched flux, relative.
const CLOSURE: f64 = 1e-12;

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

fn run_with(scene: &Scene3, c: TraceConfig3, req: &str, batch: u64) -> Value {
    let mut job = AnalysisJob::new(scene, c, request(req)).unwrap();
    while !job.step(batch) {
        assert!(job.result().is_none(), "no result before the last ray");
    }
    job.result().unwrap()
}

fn run(scene: &Scene3, req: &str, batch: u64) -> Value {
    run_with(scene, cfg(3), req, batch)
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// A square collimated beam along +X from `pos`, half-width `half` mm, on a
/// stratified `n` × `n` grid. Returns its source id.
fn beam(s: &mut Scene3, pos: Vec3, half: f64, n: u32, seed: u64) -> u64 {
    let id = s.add_collimated_beam(pos, half, half, n * n, Spectrum::Monochromatic(0.55));
    let src = &mut s.sources.last_mut().unwrap().samples;
    src.sequence = SamplingSequence::Stratified;
    src.seed = seed;
    id.0
}

/// An ideal absorbing filter across the axis at `x`: transmits `t`, absorbs
/// the rest, reflects nothing.
fn filter(s: &mut Scene3, pos: Vec3, t: f64) -> u64 {
    s.add_beam_splitter(pos, 10.0, 0.0, 0.0, t).unwrap()
}

fn detector(s: &mut Scene3, pos: Vec3, half: f64) -> u64 {
    s.add_area_detector(pos, -Vec3::X, half, half, [4, 4]).0
}

/// `{value, stderr}` is `want` with no noise.
fn exact(e: &Value, want: f64, what: &str) {
    assert!(
        (num(&e["value"]) - want).abs() < 1e-12,
        "{what}: {e}, want {want}"
    );
    assert!(num(&e["stderr"]) < 1e-12, "{what}: {e} has noise");
}

fn part<'a>(b: &'a Value, kind: &str, id: u64) -> &'a Value {
    b["parts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["kind"] == kind && p["object"] == id)
        .unwrap_or_else(|| panic!("no {kind} {id} in {}", b["parts"]))
}

/// The sinks add up to what was launched, the parts too, and the stopping
/// detectors to what was detected.
fn closes(b: &Value) {
    let launched = num(&b["launched"]);
    let c = &b["closure"];
    assert!(num(&c["relative"]) < CLOSURE, "sinks: {c}");
    assert!(
        num(&c["partsResidual"]).abs() < CLOSURE * launched,
        "parts: {c}"
    );
    let sum: f64 = [
        "detected",
        "absorbedSurface",
        "absorbedVolume",
        "blocked",
        "escaped",
        "truncated",
        "numerical",
    ]
    .iter()
    .map(|k| num(&b["totals"][k]["value"]))
    .sum();
    assert!((sum - num(&c["accounted"])).abs() < CLOSURE * launched);
    let shares: f64 = b["shares"].as_object().unwrap().values().map(num).sum();
    assert!((shares - 1.0).abs() < CLOSURE, "shares sum to {shares}");
    let caught: f64 = b["detectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| d["terminates"] == true)
        .map(|d| num(&d["flux"]["value"]))
        .sum();
    assert!((caught - num(&b["totals"]["detected"]["value"])).abs() < CLOSURE * launched);
}

/// A beam through a stop of radius 2 mm, then a filter that transmits 30%,
/// onto a detector; a lens of absorbing glass first, for bulk absorption.
fn train() -> (Scene3, [u64; 4]) {
    let mut s = Scene3::new();
    beam(&mut s, Vec3::new(-30.0, 0.0, 0.0), 3.0, 16, 1);
    let lens = s.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
    s.bodies[0].absorption = AbsorptionModel::Constant { alpha_per_mm: 0.02 };
    let stop = s.add_aperture(Vec3::new(20.0, 0.0, 0.0), 2.0, 0.0);
    let f = filter(&mut s, Vec3::new(30.0, 0.0, 0.0), 0.3);
    let d = detector(&mut s, Vec3::new(40.0, 0.0, 0.0), 10.0);
    (s, [lens.0, stop, f, d])
}

#[test]
fn stepped_one_ray_at_a_time_is_stepped_all_at_once() {
    let (mut scene, _) = train();
    beam(&mut scene, Vec3::new(-30.0, 30.0, 0.0), 1.0, 4, 2);
    for req in [
        r#"{"kind":"budget","passes":8}"#,
        r#"{"kind":"budget","passes":8,"params":{"bySource":true}}"#,
    ] {
        let whole = serde_json::to_string(&run(&scene, req, u64::MAX)).unwrap();
        for batch in [1, 37, 1000] {
            let stepped = serde_json::to_string(&run(&scene, req, batch)).unwrap();
            assert_eq!(stepped, whole, "{req}: batches of {batch}");
        }
    }
}

/// With ghosts on and a short depth limit, every sink is in play: the sinks
/// and the parts still close, and the totals are the kernel's own ledger for
/// the same rays (the object ledger changes nothing else).
#[test]
fn the_sinks_close_and_are_the_kernels_ledger() {
    let (scene, ids) = train();
    let c = TraceConfig3 {
        ghosts: true,
        max_depth: 3,
        ..cfg(3)
    };
    let r = run_with(
        &scene,
        c,
        r#"{"kind":"budget","rays":4096,"passes":4}"#,
        u64::MAX,
    );
    let b = &r["result"];
    closes(b);
    let mut job = TraceJob3::with_passes(sampled_for(&scene, 1024), 4, None, c);
    while !job.step(usize::MAX) {}
    let l = job.finish().ledger;
    let t = &b["totals"];
    for (k, want) in [
        ("detected", l.detected_incident),
        ("absorbedSurface", l.absorbed_surface),
        ("absorbedVolume", l.absorbed_volume),
        ("blocked", l.blocked_aperture),
        ("escaped", l.escaped),
        ("truncated", l.truncated_flux),
        ("numerical", l.numerical_loss),
    ] {
        assert_eq!(num(&t[k]["value"]).to_bits(), want.to_bits(), "{k}");
    }
    assert_eq!(num(&b["launched"]).to_bits(), l.launched.to_bits());
    for k in [
        "absorbedVolume",
        "blocked",
        "truncated",
        "detected",
        "absorbedSurface",
    ] {
        assert!(num(&t[k]["value"]) > 0.0, "{k} is in play: {t}");
    }
    assert!(num(&b["branches"]["maxDepth"]) > 0.0, "{}", b["branches"]);
    // Largest loss first.
    let lost: Vec<f64> = b["parts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| num(&p["lost"]["value"]))
        .collect();
    assert!(lost.windows(2).all(|w| w[0] >= w[1]), "{lost:?}");

    // The parts are the kernel's object ledger for the same rays, both bands
    // and a body's faces added up in the ledger's order.
    let mut job = TraceJob3::with_passes(
        sampled_for(&scene, 1024),
        4,
        None,
        TraceConfig3 {
            object_ledger: true,
            ..c
        },
    );
    while !job.step(usize::MAX) {}
    let objects = job.finish().objects;
    let kernel = |of: &dyn Fn(ObjectRef3) -> bool| {
        let mut l = ObjectLedger3::default();
        for e in objects.iter().filter(|e| of(e.object)) {
            l.add(&e.ledger);
        }
        l
    };
    type Field = (&'static str, fn(&ObjectLedger3) -> f64);
    let fields: [Field; 8] = [
        ("arrived", |l| l.incident),
        ("absorbedSurface", |l| l.absorbed_surface),
        ("absorbedVolume", |l| l.absorbed_volume),
        ("blocked", |l| l.blocked),
        ("detected", |l| l.detected),
        ("escaped", |l| l.escaped),
        ("truncated", |l| l.truncated),
        ("numerical", |l| l.numerical),
    ];
    let (lens, stop, f, d) = (ids[0], ids[1], ids[2], ids[3]);
    for (kind, id, want) in [
        (
            "body",
            lens,
            kernel(&|o| matches!(o, ObjectRef3::Body(0, _))),
        ),
        ("aperture", stop, kernel(&|o| o == ObjectRef3::Aperture(0))),
        ("surface", f, kernel(&|o| o == ObjectRef3::Surface(0))),
        ("detector", d, kernel(&|o| o == ObjectRef3::Detector(0))),
    ] {
        let p = part(b, kind, id);
        assert_eq!(p["hits"], want.hits, "{kind} hits are a plain count");
        for (k, f) in fields {
            assert_eq!(
                num(&p[k]["value"]).to_bits(),
                f(&want).to_bits(),
                "{kind} {k}"
            );
        }
        let lost = want.absorbed_surface + want.absorbed_volume + want.blocked;
        assert_eq!(
            num(&p["lost"]["value"]).to_bits(),
            lost.to_bits(),
            "{kind} lost"
        );
    }
    let faces = part(b, "body", lens)["faces"].as_array().unwrap();
    assert!(!faces.is_empty(), "the lens has faces");
    for face in faces {
        let patch = face["patch"].as_u64().unwrap();
        let want = kernel(&|o| o == ObjectRef3::Body(0, patch));
        for (k, f) in fields {
            assert_eq!(
                num(&face[k]["value"]).to_bits(),
                f(&want).to_bits(),
                "face {patch} {k}"
            );
        }
    }
    // Noise: a ray-traced loss has a spread from pass to pass.
    let lens_row = part(b, "body", lens);
    for k in ["arrived", "absorbedVolume", "lost"] {
        assert!(
            num(&lens_row[k]["stderr"]) > 0.0,
            "lens {k}: {}",
            lens_row[k]
        );
    }
    assert!(num(&part(b, "aperture", stop)["blocked"]["stderr"]) > 0.0);
}

#[test]
fn an_ideal_filter_absorbs_its_known_share() {
    let mut s = Scene3::new();
    let src = beam(&mut s, Vec3::ZERO, 2.0, 8, 1);
    let f = filter(&mut s, Vec3::new(20.0, 0.0, 0.0), 0.3);
    let d = detector(&mut s, Vec3::new(50.0, 0.0, 0.0), 4.0);
    let r = run(&s, r#"{"kind":"budget","passes":8}"#, u64::MAX);
    let b = &r["result"];
    closes(b);
    exact(&b["totals"]["detected"], 0.3, "detected");
    exact(&b["totals"]["absorbedSurface"], 0.7, "absorbed");
    for k in [
        "absorbedVolume",
        "blocked",
        "escaped",
        "truncated",
        "numerical",
    ] {
        exact(&b["totals"][k], 0.0, k);
    }
    exact(&b["detectors"][0]["flux"], 0.3, "on the detector");
    assert_eq!(b["detectors"][0]["detector"], d);
    // Part by part, with no noise either.
    let p = part(b, "surface", f);
    exact(&p["lost"], 0.7, "the filter's loss");
    exact(&p["absorbedSurface"], 0.7, "the filter's absorption");
    exact(&p["arrived"], 1.0, "at the filter");
    assert_eq!(p["hits"], 64 * 8, "hits are a plain count");
    assert_eq!(b["parts"][0]["object"], f, "the filter loses most");
    let pd = part(b, "detector", d);
    exact(&pd["detected"], 0.3, "the detector's");
    exact(&pd["lost"], 0.0, "the detector's loss");
    // Nothing is lost before the filter: the launch books no row.
    assert!(b["parts"]
        .as_array()
        .unwrap()
        .iter()
        .all(|p| p["kind"] != "source"));
    let _ = src;
}

/// A square beam of half-width 4 mm through a stop of radius 2 mm: the stop
/// passes π·2²/8² of it and blocks the rest.
#[test]
fn a_stop_blocks_the_share_outside_it() {
    let mut s = Scene3::new();
    beam(&mut s, Vec3::ZERO, 4.0, 32, 1);
    let stop = s.add_aperture(Vec3::new(20.0, 0.0, 0.0), 2.0, 0.0);
    detector(&mut s, Vec3::new(50.0, 0.0, 0.0), 5.0);
    let r = run(&s, r#"{"kind":"budget","passes":8}"#, u64::MAX);
    let b = &r["result"];
    closes(b);
    let want = 1.0 - std::f64::consts::PI * 4.0 / 64.0;
    let blocked = &b["totals"]["blocked"];
    let (v, se) = (num(&blocked["value"]), num(&blocked["stderr"]));
    assert!(
        se > 0.0 && (v - want).abs() < 5.0 * se.max(1e-3),
        "blocked {v} ± {se}, want {want}"
    );
    assert!((v + num(&b["totals"]["detected"]["value"]) - 1.0).abs() < CLOSURE);
    let p = part(b, "aperture", stop);
    assert_eq!(num(&p["blocked"]["value"]).to_bits(), v.to_bits());
    assert!(
        (num(&p["arrived"]["value"]) - v).abs() < 1e-12,
        "a stop's arrivals are what it blocked"
    );
    // The stop is the only thing that blocks, so its noise is the total's.
    let pse = num(&p["blocked"]["stderr"]);
    assert!(
        pse > 0.0 && (pse - se).abs() < 1e-12,
        "{} vs {se}",
        p["blocked"]
    );
    assert!((num(&b["shares"]["blocked"]) - v).abs() < 1e-15);
}

/// A beam that meets nothing leaves the scene whole, from the launch.
#[test]
fn a_beam_that_misses_everything_escapes() {
    let mut s = Scene3::new();
    let src = beam(&mut s, Vec3::ZERO, 2.0, 8, 1);
    detector(&mut s, Vec3::new(50.0, 30.0, 0.0), 4.0);
    let r = run(&s, r#"{"kind":"budget","passes":4}"#, u64::MAX);
    let b = &r["result"];
    closes(b);
    exact(&b["totals"]["escaped"], 1.0, "escaped");
    exact(&b["totals"]["detected"], 0.0, "detected");
    exact(&b["detectors"][0]["flux"], 0.0, "on the detector");
    let p = part(b, "source", src);
    assert_eq!(p["label"].as_str(), Some("launch"));
    exact(&p["escaped"], 1.0, "escaped from the launch");
    exact(&p["lost"], 0.0, "lost at the launch");
    assert_eq!(b["branches"]["escaped"], 64 * 4);
}

/// By source: each traced alone, each with its own budget, the scene's the
/// sum of theirs.
#[test]
fn by_source_each_source_has_its_own_budget() {
    let mut s = Scene3::new();
    let a = beam(&mut s, Vec3::ZERO, 2.0, 8, 1);
    let b = beam(&mut s, Vec3::new(0.0, 30.0, 0.0), 2.0, 8, 2);
    filter(&mut s, Vec3::new(20.0, 0.0, 0.0), 0.3);
    detector(&mut s, Vec3::new(50.0, 0.0, 0.0), 4.0);
    let r = run(
        &s,
        r#"{"kind":"budget","passes":4,"params":{"bySource":true}}"#,
        u64::MAX,
    );
    let all = &r["result"];
    closes(all);
    assert_eq!(num(&all["launched"]), 2.0);
    exact(&all["totals"]["detected"], 0.3, "detected");
    exact(&all["totals"]["escaped"], 1.0, "escaped");
    let src = all["sources"].as_array().unwrap();
    assert_eq!(
        (src[0]["source"].as_u64(), src[1]["source"].as_u64()),
        (Some(a), Some(b))
    );
    assert_eq!(src[0]["rays"], 256);
    for x in src {
        closes(x);
    }
    exact(&src[0]["totals"]["absorbedSurface"], 0.7, "a absorbed");
    exact(&src[0]["totals"]["escaped"], 0.0, "a escaped");
    exact(&src[1]["totals"]["escaped"], 1.0, "b escaped");
    exact(&part(&src[1], "source", b)["escaped"], 1.0, "b's launch");
    exact(
        &part(all, "source", b)["escaped"],
        1.0,
        "b's launch, in all",
    );
    // Without bySource there is no per-source table.
    let whole = run(&s, r#"{"kind":"budget","passes":4}"#, u64::MAX);
    assert!(whole["result"].get("sources").is_none());
}

#[test]
fn a_request_the_scene_cannot_answer_is_refused() {
    let (scene, _) = train();
    let err = |req: &str| match AnalysisRequest::from_json(req) {
        Err(e) => e,
        Ok(r) => AnalysisJob::new(&scene, cfg(0), r).err().unwrap(),
    };
    assert!(err(r#"{"kind":"budget","passes":3}"#).contains("divides"));
    assert!(err(r#"{"kind":"budget","params":{"parts":true}}"#).contains("unknown field"));
    assert!(err(r#"{"kind":"budget","params":{"bySource":"yes"}}"#).contains("budget params"));
    let mut dark = scene.clone();
    dark.sources.clear();
    let e = AnalysisJob::new(&dark, cfg(0), request(r#"{"kind":"budget"}"#))
        .err()
        .unwrap();
    assert!(e.contains("emits no rays"), "{e}");
}
