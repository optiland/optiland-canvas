//! `fluorescence`: light a dye absorbs and emits again, traced in two stages.
//!
//! What this suite owes:
//!
//! - a job gives the same result to the bit however it is stepped;
//! - what a dye absorbs is the kernel's own deposit total, to the bit, and
//!   what it emits is the quantum yield times that, pass by pass;
//! - a thin slab in a collimated beam absorbs `1 − exp(−αd)`;
//! - a point-like dye emits the same into every direction: detectors around
//!   it each catch the solid-angle share of the emission;
//! - a dye absorbs only where its spectrum says: a source off its band
//!   excites nothing;
//! - an emission filter that blocks the band leaves no emission at the
//!   camera, while the excitation that leaks through is still read;
//! - the emission converges on the analytic answer as the voxels are refined,
//!   and the bias of a coarse grid is stated;
//! - a request the scene cannot answer is refused with the reason.

use oc_analysis::{sampled_for, source_alone, AnalysisJob, AnalysisRequest};
use oc_core::build3::DichroicSpec;
use oc_core::id::LayerId;
use oc_core::material::Material;
use oc_core::scene::Spectrum;
use oc_core::scene3::{
    AbsorptionModel, Body3, Interpolation, ObjectCommon, OutOfRangePolicy, SamplingSequence,
    Scene3, SolidGeometry3, SurfaceProps3,
};
use oc_core::vec3::{Isometry3, Vec3};
use oc_trace::{DepositLayer, ObjectRef3, TraceConfig3, TraceJob3};
use serde_json::Value;
use std::f64::consts::PI;

/// The excitation line the dyes absorb, and the emission they give.
const EXCITE: f64 = 0.488;
const EMISSION: &str = r#"{"lambdaUm":[0.54,0.55,0.56],"weight":[1,2,1]}"#;

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
    let p = job.progress();
    assert_eq!(p.traced, p.rays, "every planned ray is traced");
    job.result().unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

fn val(e: &Value) -> (f64, f64) {
    (num(&e["value"]), num(&e["stderr"]))
}

/// A dye's absorption, `alpha` per mm across 480 to 500 nm, nothing outside
/// 470 to 510 nm: it takes the 488 nm line and not its own emission.
fn absorbs(alpha: f64) -> AbsorptionModel {
    AbsorptionModel::Tabulated {
        lambda_um: vec![0.47, 0.48, 0.50, 0.51],
        alpha_per_mm: vec![0.0, alpha, alpha, 0.0],
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Zero,
    }
}

/// A body of index 1 (so its faces neither reflect nor bend) at `center`.
fn body(
    s: &mut Scene3,
    center: Vec3,
    geometry: SolidGeometry3,
    absorption: AbsorptionModel,
) -> u64 {
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::from_pos(center);
    s.bodies.push(Body3 {
        common,
        geometry,
        material: Material::AbbePoint { nd: 1.0, vd: 50.0 },
        absorption,
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    id.0
}

fn cuboid(s: &mut Scene3, center: Vec3, half: Vec3, absorption: AbsorptionModel) -> u64 {
    body(
        s,
        center,
        SolidGeometry3::Box {
            half_extents_mm: half,
        },
        absorption,
    )
}

/// A square beam along +X from `pos`, half-width `half`, `n` × `n` rays on a
/// stratified grid. Its source id.
fn beam(s: &mut Scene3, pos: Vec3, half: f64, n: u32, lambda: f64, seed: u64) -> u64 {
    let id = s.add_collimated_beam(pos, half, half, n * n, Spectrum::Monochromatic(lambda));
    let src = &mut s.sources.last_mut().unwrap().samples;
    src.sequence = SamplingSequence::Stratified;
    src.seed = seed;
    id.0
}

fn detector(s: &mut Scene3, pos: Vec3, normal: Vec3, half: f64) -> u64 {
    s.add_area_detector(pos, normal, half, half, [4, 4]).0
}

/// A 1 mm slab of dye across a 2 mm beam, a camera behind it on the axis and
/// a detector beside it. The dye's id, and the camera's and the side's.
fn slab_scene(alpha: f64, seed: u64) -> (Scene3, u64, u64, u64) {
    let mut s = Scene3::new();
    beam(&mut s, Vec3::ZERO, 1.0, 12, EXCITE, seed);
    let dye = cuboid(
        &mut s,
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::new(0.5, 3.0, 3.0),
        absorbs(alpha),
    );
    let camera = detector(&mut s, Vec3::new(30.0, 0.0, 0.0), -Vec3::X, 5.0);
    let side = detector(&mut s, Vec3::new(10.0, 8.0, 0.0), -Vec3::Y, 4.0);
    (s, dye, camera, side)
}

fn req(dye: u64, rays: u64, passes: u32, extra: &str) -> String {
    format!(
        r#"{{"kind":"fluorescence","rays":{rays},"passes":{passes},"params":{{"dyes":[{{"body":{dye},"quantumYield":0.8,"emission":{EMISSION},"name":"dye"}}]{extra}}}}}"#
    )
}

fn detector_of(r: &Value, id: u64) -> &Value {
    r["result"]["detectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["detector"] == id)
        .unwrap_or_else(|| panic!("no detector {id}"))
}

#[test]
fn stepped_in_any_batches_it_gives_the_same_json() {
    // A sphere of dye, so the paths through it differ and every number has
    // noise, behind a beam partly past it.
    let mut s = Scene3::new();
    beam(&mut s, Vec3::ZERO, 2.5, 8, EXCITE, 3);
    let dye = body(
        &mut s,
        Vec3::new(10.0, 0.0, 0.0),
        SolidGeometry3::Sphere { radius_mm: 2.0 },
        absorbs(0.3),
    );
    detector(&mut s, Vec3::new(30.0, 0.0, 0.0), -Vec3::X, 5.0);
    detector(&mut s, Vec3::new(10.0, 8.0, 0.0), -Vec3::Y, 4.0);
    let req = req(dye, 4000, 4, r#","voxels":4"#);
    let whole = serde_json::to_string(&run(&s, &req, u64::MAX)).unwrap();
    for batch in [1, 37, 1000] {
        let stepped = serde_json::to_string(&run(&s, &req, batch)).unwrap();
        assert_eq!(stepped, whole, "batches of {batch}");
    }
    let r: Value = serde_json::from_str(&whole).unwrap();
    assert_eq!(r["kind"], "fluorescence");
    let samp = &r["sampling"];
    assert_eq!(samp["passes"], 4);
    assert_eq!(samp["stderrIncludesExcitation"], true);
    let (e, m) = (
        samp["excitation"]["rays"].as_u64().unwrap(),
        samp["emission"]["rays"].as_u64().unwrap(),
    );
    assert_eq!(samp["rays"].as_u64().unwrap(), e + m);
    assert!(m >= 1990 && e == 2000, "excitation {e}, emission {m}");
}

/// What the dye absorbed is what the kernel deposits in it, tracing the
/// source alone at the same rays, to the bit; and what it emits is the
/// quantum yield times that, the pooled value and every pass.
#[test]
fn absorbed_is_the_kernels_deposit_and_emitted_is_the_yield_of_it() {
    let mut s = Scene3::new();
    beam(&mut s, Vec3::ZERO, 2.5, 8, EXCITE, 5);
    let dye = body(
        &mut s,
        Vec3::new(10.0, 0.0, 0.0),
        SolidGeometry3::Sphere { radius_mm: 2.0 },
        absorbs(0.3),
    );
    detector(&mut s, Vec3::new(30.0, 0.0, 0.0), -Vec3::X, 5.0);
    let (rays, passes) = (16_000u64, 8u32);
    let r = run(&s, &req(dye, rays, passes, ""), u64::MAX);
    let d = &r["result"]["dyes"][0];

    let traced = sampled_for(&s, (rays as f64 / passes as f64 * 0.5).round() as u64);
    let c = TraceConfig3 {
        record_detector_hits: false,
        record_events: false,
        segment_floor: f64::INFINITY,
        deposit_bodies: [0, 1],
        ..cfg(3)
    };
    let mut job = TraceJob3::with_passes(source_alone(&traced, 0), passes, None, c);
    while !job.step(usize::MAX) {}
    let t = job.finish();
    let want = t
        .deposits
        .iter()
        .filter(|x| x.layer == DepositLayer::Bulk && matches!(x.object, ObjectRef3::Body(0, _)))
        .fold(0.0, |a, x| a + x.flux);
    assert!(want > 0.1);
    assert_eq!(num(&d["absorbed"]["value"]).to_bits(), want.to_bits());
    assert_eq!(
        num(&d["bySource"][0]["absorbed"]["value"]).to_bits(),
        want.to_bits()
    );

    let (a, a_se) = val(&d["absorbed"]);
    let (e, e_se) = val(&d["emitted"]);
    assert!(a_se > 0.0, "the sphere's absorption is noisy");
    // The per-pass values are proportional, so their spread is too.
    assert!((e - 0.8 * a).abs() < 1e-12 * a, "emitted {e}, absorbed {a}");
    assert!(
        (e_se - 0.8 * a_se).abs() < 1e-9 * a_se,
        "stderr {e_se} vs {a_se}"
    );
    // One line, at the source's own wavelength.
    let lines = d["bySource"][0]["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 1);
    assert!((num(&lines[0]["meanLambdaUm"]) - EXCITE).abs() < 1e-12);
    assert_eq!(
        num(&lines[0]["absorbed"]["value"]).to_bits(),
        want.to_bits()
    );
}

#[test]
fn a_thin_slab_absorbs_one_minus_exp_of_alpha_d() {
    for alpha in [0.05, 0.4, 2.0] {
        let (s, dye, camera, _) = slab_scene(alpha, 1);
        let r = run(&s, &req(dye, 8000, 4, ""), u64::MAX);
        let want = 1.0 - (-alpha * 1.0f64).exp();
        let (a, se) = val(&r["result"]["dyes"][0]["absorbed"]);
        assert!(
            (a - want).abs() < 1e-9 + 5.0 * se,
            "alpha {alpha}: {a} ± {se}, want {want}"
        );
        // The rest goes on to the camera: the excitation leak, with no noise.
        let (leak, leak_se) = val(&detector_of(&r, camera)["excitation"]["flux"]);
        assert!(
            (leak - (1.0 - want)).abs() < 1e-9 && leak_se < 1e-12,
            "leak {leak} ± {leak_se}"
        );
        assert_eq!(
            r["result"]["excitation"]["flux"][0][0]["value"],
            detector_of(&r, camera)["excitation"]["flux"]["value"]
        );
    }
}

/// The solid angle a square of half-side `a` subtends at a point `d` from
/// its centre on its axis.
fn square_solid_angle(a: f64, d: f64) -> f64 {
    4.0 * (a * a / (a * a + d * d)).asin()
}

#[test]
fn a_point_like_dye_emits_the_same_into_every_direction() {
    let mut s = Scene3::new();
    beam(&mut s, Vec3::new(-5.0, 0.0, 0.0), 0.05, 6, EXCITE, 9);
    let dye = cuboid(&mut s, Vec3::ZERO, Vec3::splat(0.1), absorbs(2.0));
    let (dist, half) = (20.0, 5.0);
    let around = [Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z, -Vec3::X];
    let ids: Vec<u64> = around
        .iter()
        .map(|&n| detector(&mut s, n * dist, -n, half))
        .collect();
    let r = run(
        &s,
        &req(dye, 240_000, 8, r#","emissionShare":0.8"#),
        u64::MAX,
    );
    let emitted = num(&r["result"]["dyes"][0]["emitted"]["value"]);
    assert!((emitted - 0.8 * (1.0 - (-0.4f64).exp())).abs() < 1e-9);
    let share = square_solid_angle(half, dist) / (4.0 * PI);
    for id in ids {
        let d = detector_of(&r, id);
        let (f, se) = val(&d["emission"]["flux"]);
        let want = emitted * share;
        assert!(se > 0.0 && se < 0.05 * want, "{id}: {f} ± {se}");
        assert!(
            (f - want).abs() < 5.0 * se,
            "detector {id}: {f} ± {se}, want {want}"
        );
        // No excitation reaches the detectors off the beam.
        assert_eq!(num(&d["excitation"]["flux"]["value"]), 0.0);
        // The emission arrives at its own wavelengths.
        for l in d["emission"]["lines"].as_array().unwrap() {
            let lam = num(&l["meanLambdaUm"]);
            assert!((0.535..=0.565).contains(&lam), "{lam}");
        }
    }
}

#[test]
fn a_source_outside_the_absorption_band_excites_nothing() {
    let (mut s, dye, camera, side) = slab_scene(0.4, 1);
    let blue = s.sources[0].id.0;
    let red = beam(&mut s, Vec3::new(0.0, 0.0, 0.0), 1.0, 12, 0.60, 2);
    let r = run(&s, &req(dye, 16_000, 4, ""), u64::MAX);
    let by = r["result"]["dyes"][0]["bySource"].as_array().unwrap();
    assert_eq!(
        (by[0]["source"].as_u64(), by[1]["source"].as_u64()),
        (Some(blue), Some(red))
    );
    assert!(num(&by[0]["absorbed"]["value"]) > 0.3);
    assert_eq!(val(&by[1]["absorbed"]), (0.0, 0.0));
    assert!(by[1]["lines"].as_array().unwrap().is_empty());
    // Each source's excitation leak, read apart.
    let ex = &r["result"]["excitation"];
    assert_eq!(ex["sources"], serde_json::json!([blue, red]));
    assert!(
        (num(&ex["flux"][1][0]["value"]) - 1.0).abs() < 1e-9,
        "red passes whole"
    );

    // The red source alone: nothing absorbed, nothing emitted, no emission
    // traced, and every emission number a plain zero.
    s.sources.remove(0);
    let r = run(&s, &req(dye, 8000, 4, ""), u64::MAX);
    let d = &r["result"]["dyes"][0];
    assert_eq!(val(&d["absorbed"]), (0.0, 0.0));
    assert_eq!(val(&d["emitted"]), (0.0, 0.0));
    assert_eq!(d["emissionRays"], 0);
    assert_eq!(r["sampling"]["emission"]["rays"], 0);
    for id in [camera, side] {
        assert_eq!(val(&detector_of(&r, id)["emission"]["flux"]), (0.0, 0.0));
    }
}

#[test]
fn an_emission_filter_blocks_the_emission_and_passes_the_leak() {
    let (s, dye, camera, side) = slab_scene(0.4, 1);
    let open = run(&s, &req(dye, 16_000, 4, ""), u64::MAX);
    let (f, se) = val(&detector_of(&open, camera)["emission"]["flux"]);
    assert!(
        f > 5.0 * se && f > 0.0,
        "the camera sees emission without the filter: {f} ± {se}"
    );

    // A short-pass at 520 nm: the 488 nm excitation through, the 540 to
    // 560 nm emission absorbed.
    let mut s = s;
    let spec = DichroicSpec {
        edge_nm: 520.0,
        width_nm: 10.0,
        long_pass: false,
        r_max: 0.0,
        t_max: 1.0,
    };
    s.add_dichroic(Vec3::new(20.0, 0.0, 0.0), 10.0, 0.0, spec)
        .unwrap();
    let r = run(&s, &req(dye, 16_000, 4, ""), u64::MAX);
    let cam = detector_of(&r, camera);
    assert_eq!(val(&cam["emission"]["flux"]), (0.0, 0.0));
    assert_eq!(val(&cam["emission"]["signal"]), (0.0, 0.0));
    let (leak, _) = val(&cam["excitation"]["flux"]);
    assert!(
        (leak - (-0.4f64).exp()).abs() < 1e-9,
        "the leak is still read: {leak}"
    );
    // The side detector, out of the filter's way, still sees the emission.
    assert!(num(&detector_of(&r, side)["emission"]["flux"]["value"]) > 0.0);
    // The table: dye by detector.
    let table = &r["result"]["emission"];
    assert_eq!(table["dyes"], serde_json::json!(["dye"]));
    assert_eq!(table["flux"][0][0]["value"], 0.0);
}

/// The solid angle of the rectangle `[x1, x2] × [y1, y2]` in a plane `h`
/// away, seen from the origin.
fn rectangle_solid_angle(x1: f64, x2: f64, y1: f64, y2: f64, h: f64) -> f64 {
    let f = |x: f64, y: f64| (x * y / (h * (h * h + x * x + y * y).sqrt())).atan();
    f(x2, y2) - f(x1, y2) - f(x2, y1) + f(x1, y1)
}

/// The share of isotropic emission spread along `[0, l]` of a line, by
/// `weight`, that a square detector of half-side `a` catches: the square in
/// a plane `h` from the line and parallel to it, spanning `[0, 2a]` along it
/// and `[-a, a]` across it. Simpson's rule, fine enough to be exact here.
fn line_share(l: f64, a: f64, h: f64, weight: impl Fn(f64) -> f64) -> f64 {
    let n = 20_000;
    let dx = l / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let x = i as f64 * dx;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * weight(x) * rectangle_solid_angle(-x, 2.0 * a - x, -a, a, h) / (4.0 * PI);
    }
    sum * dx / 3.0
}

/// A 10 mm bar of dye lit across its whole length by a sheet of light, and a
/// square detector 1.5 mm from it beside its first half. Each ray crosses
/// 0.2 mm of dye, so the dye absorbs evenly along the bar. Emission from a
/// voxel starts at the centroid of what it absorbed, not spread along it:
/// the coarser the grid, the further off. The result converges on the
/// analytic answer as the voxels shrink.
#[test]
fn the_emission_converges_as_the_voxels_are_refined() {
    let (alpha, l, a, h, qy) = (2.0, 10.0, 2.5, 1.5, 0.8);
    let mut s = Scene3::new();
    // The sheet: 10 mm along the bar (local Y), 0.02 mm thick, along +X.
    s.add_collimated_beam(
        Vec3::new(-1.0, l / 2.0, 0.0),
        l / 2.0,
        0.01,
        4096,
        Spectrum::Monochromatic(EXCITE),
    );
    s.sources[0].samples.sequence = SamplingSequence::Stratified;
    let dye = cuboid(
        &mut s,
        Vec3::new(0.0, l / 2.0, 0.0),
        Vec3::new(0.1, l / 2.0, 0.1),
        absorbs(alpha),
    );
    let det = detector(&mut s, Vec3::new(0.0, a, h), -Vec3::Z, a);
    let absorbed = 1.0 - (-alpha * 0.2f64).exp();
    let want = qy * absorbed * line_share(l, a, h, |_| 1.0 / l);
    let mut bias = Vec::new();
    for n in [1, 2, 4, 8, 16, 32, 64] {
        let extra = format!(r#","voxels":[1,{n},1],"emissionShare":0.9"#);
        let r = run(&s, &req(dye, 160_000, 8, &extra), u64::MAX);
        let (e, _) = val(&r["result"]["dyes"][0]["emitted"]);
        assert!(
            (e - qy * absorbed).abs() < 1e-9,
            "every voxel emits its own: {e}"
        );
        let (f, se) = val(&detector_of(&r, det)["emission"]["flux"]);
        let (b, bse) = ((f - want) / want, se / want);
        eprintln!(
            "voxels {n:>2} along the bar ({:.3} mm): emission {f:.6} ± {se:.6}, want {want:.6}, bias {:+.3}% ± {:.3}%",
            l / n as f64,
            100.0 * b,
            100.0 * bse
        );
        bias.push((n, b, bse));
    }
    // One voxel is far off, and each halving of the voxels cuts the bias by
    // about four (the centroid's error is second order in the voxel's size)
    // until it is under the noise.
    let (_, b1, s1) = bias[0];
    assert!(b1.abs() > 10.0 * s1, "one voxel is biased: {b1} ± {s1}");
    for w in bias.windows(2) {
        let ((n0, b0, s0), (n1, b1, s1)) = (w[0], w[1]);
        if b0.abs() > 10.0 * s0 {
            assert!(
                b1.abs() < 0.5 * b0.abs() + 3.0 * s1,
                "{n0} to {n1} voxels: {b0} to {b1}"
            );
        }
    }
    for &(n, b, se) in &bias[4..] {
        assert!(b.abs() < 4.0 * se, "{n} voxels: bias {b} ± {se}");
    }
}

/// The kernel books a segment's bulk absorption at the middles of four equal
/// steps along it ([`oc_trace::Deposit3`]): along a long path through the
/// dye, that is where the emission starts, however fine the voxels. Down a
/// 10 mm bar, that puts it at 1.25, 3.75, 6.25 and 8.75 mm, and the job reads
/// what emission from those four points gives, not a continuous line.
#[test]
fn along_a_long_path_the_emission_starts_where_the_kernel_deposits() {
    let (alpha, l, a, h, qy) = (0.1, 10.0, 2.5, 1.5, 0.8);
    let mut s = Scene3::new();
    beam(&mut s, Vec3::new(-1.0, 0.0, 0.0), 0.01, 4, EXCITE, 4);
    let dye = cuboid(
        &mut s,
        Vec3::new(l / 2.0, 0.0, 0.0),
        Vec3::new(l / 2.0, 0.1, 0.1),
        absorbs(alpha),
    );
    let det = detector(&mut s, Vec3::new(a, h, 0.0), -Vec3::Y, a);
    let step = l / 4.0;
    let four: f64 = (0..4)
        .map(|k| {
            let x = step * (k as f64 + 0.5);
            let p = (-alpha * step * k as f64).exp() - (-alpha * step * (k + 1) as f64).exp();
            p * rectangle_solid_angle(-x, 2.0 * a - x, -a, a, h) / (4.0 * PI)
        })
        .sum::<f64>()
        * qy;
    let line = qy * line_share(l, a, h, |x| alpha * (-alpha * x).exp());
    let r = run(
        &s,
        &req(dye, 160_000, 8, r#","voxels":[32,1,1],"emissionShare":0.9"#),
        u64::MAX,
    );
    let (f, se) = val(&detector_of(&r, det)["emission"]["flux"]);
    eprintln!(
        "four deposit points {four:.6}, a continuous line {line:.6} ({:+.2}%), read {f:.6} ± {se:.6}",
        100.0 * (four - line) / line
    );
    assert!(
        (f - four).abs() < 4.0 * se,
        "{f} ± {se}, the four points give {four}"
    );
    assert!(
        (four - line).abs() > 10.0 * se,
        "the steps show at this resolution"
    );
}

#[test]
fn a_request_the_scene_cannot_answer_is_refused() {
    let (mut s, dye, _, _) = slab_scene(0.4, 1);
    let clear = cuboid(
        &mut s,
        Vec3::new(50.0, 0.0, 0.0),
        Vec3::splat(1.0),
        AbsorptionModel::None,
    );
    let err = |req: String, scene: &Scene3| {
        AnalysisJob::new(scene, cfg(0), request(&req))
            .err()
            .unwrap()
    };
    let one = |dye: String| format!(r#"{{"kind":"fluorescence","params":{{"dyes":[{dye}]}}}}"#);
    let spec = |body: u64, qy: &str, emission: &str| {
        format!(r#"{{"body":{body},"quantumYield":{qy},"emission":{emission}}}"#)
    };
    assert!(err(one(spec(999, "0.5", EMISSION)), &s).contains("no body has id 999"));
    assert!(err(one(spec(clear, "0.5", EMISSION)), &s).contains("absorbs nothing"));
    for qy in ["0", "-0.1", "1.5"] {
        assert!(
            err(one(spec(dye, qy, EMISSION)), &s).contains("quantum yield"),
            "qy {qy}"
        );
    }
    for emission in [
        r#"{"lambdaUm":[0.5,0.6],"weight":[1]}"#,
        r#"{"lambdaUm":[],"weight":[]}"#,
        r#"{"lambdaUm":[0.5],"weight":[-1]}"#,
        r#"{"lambdaUm":[0.5],"weight":[0]}"#,
        r#"{"lambdaUm":[0],"weight":[1]}"#,
    ] {
        assert!(
            err(one(spec(dye, "0.5", emission)), &s).contains("emission"),
            "{emission}"
        );
    }
    let two = format!(
        r#"{{"kind":"fluorescence","params":{{"dyes":[{},{}]}}}}"#,
        spec(dye, "0.5", EMISSION),
        spec(dye, "0.5", EMISSION)
    );
    assert!(err(two, &s).contains("two dyes"));
    let with = |extra: &str| {
        format!(
            r#"{{"kind":"fluorescence","params":{{"dyes":[{}]{extra}}}}}"#,
            spec(dye, "0.5", EMISSION)
        )
    };
    assert!(err(with(r#","voxels":0"#), &s).contains("voxels"));
    assert!(err(with(r#","voxels":[1024,1024,1]"#), &s).contains("voxels"));
    assert!(err(with(r#","emissionShare":1"#), &s).contains("emissionShare"));
    assert!(err(with(r#","emissionShare":0"#), &s).contains("emissionShare"));
    assert!(
        err(r#"{"kind":"fluorescence","params":{"dyes":[]}}"#.into(), &s)
            .contains("at least one dye")
    );
    let mut blind = s.clone();
    blind.detectors.clear();
    assert!(err(with(""), &blind).contains("no detector"));
}
