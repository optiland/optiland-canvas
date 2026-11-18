//! The reference trace done in batches (`TraceJob3`), and the nearest-hit walk
//! checked against the brute-force search it replaced.
//!
//! What this suite owes:
//!
//! - a job run to the end gives the same trace however it was batched, to the
//!   bit, and the same trace `trace3` gives: the same segments, and the same
//!   detector readouts and ledger up to the order a parallel sum adds them in;
//! - a job stopped part way holds the rays it traced and nothing else;
//! - a pool of workers sharing the blocks out, each block's numbers encoded
//!   and posted to the one that adds them up, gives the job's trace to the
//!   bit, hit records and segment order included;
//! - each block's object ledger, read in place before the merge, adds up to
//!   the merged trace's, row by row, to the bit;
//! - each block's deposits, taken out before the merge, are the merged
//!   trace's deposits in order, and a merge of taken parts holds none;
//! - the nearest-first walk that passes over boxes beyond the nearest hit
//!   finds the events the brute-force search finds, to the bit, on every
//!   design in the gallery, with ghosts on the smaller ones.
#![cfg(feature = "trace3")]

use oc_core::scene3::Scene3;
use oc_trace::{
    depth_for_scene3, launch_from_scene, trace3, EncodedPart3, ObjectEntry3, ObjectLedger3,
    ObjectRef3, Segment3, TraceBlocks3, TraceConfig3, TraceJob3, TraceMerge3, TraceResult3,
};

const GALLERY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/public/gallery");

fn design(id: &str) -> Scene3 {
    let text = std::fs::read_to_string(format!("{GALLERY}/{id}.ocanvas")).unwrap();
    Scene3::from_json(&text).unwrap()
}

fn cfg(scene: &Scene3, ghosts: bool) -> TraceConfig3 {
    TraceConfig3 {
        ghosts,
        max_ghost_bounces: 2,
        max_depth: depth_for_scene3(scene, ghosts, 2),
        record_detector_hits: true,
        ..TraceConfig3::default()
    }
}

fn run(scene: &Scene3, cfg: TraceConfig3, batch: usize) -> TraceResult3 {
    let mut job = TraceJob3::new(scene.clone(), cfg);
    while !job.step(batch) {}
    job.finish()
}

/// Every segment, as bits, sorted: the picture, whatever order it was traced in.
fn segments(r: &TraceResult3) -> Vec<[u64; 8]> {
    let mut v: Vec<[u64; 8]> = r
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
            .map(f64::to_bits)
        })
        .collect();
    v.sort_unstable();
    v
}

/// The readouts and the ledger, as bits.
fn numbers(r: &TraceResult3) -> Vec<u64> {
    let mut v = Vec::new();
    for d in &r.detectors {
        v.extend(d.incident_flux.iter().map(|x| x.to_bits()));
        v.push(d.total_incident.to_bits());
        v.push(d.hits);
    }
    let l = &r.ledger;
    v.extend([l.launched, l.escaped, l.truncated_flux, l.detected_incident].map(f64::to_bits));
    v
}

fn assert_same(a: &TraceResult3, b: &TraceResult3, what: &str) {
    assert_eq!(segments(a), segments(b), "{what}: segments");
    assert_eq!(numbers(a), numbers(b), "{what}: readouts and ledger");
    assert_eq!(a.terminations, b.terminations, "{what}: terminations");
}

#[test]
fn a_job_gives_the_same_trace_however_it_is_batched() {
    for (id, ghosts) in [
        ("cooke-triplet", true),
        ("double-gauss", false),
        ("uv-projection-lens", false),
    ] {
        let scene = design(id);
        let whole = run(&scene, cfg(&scene, ghosts), usize::MAX);
        for batch in [1, 7, 64] {
            assert_same(
                &run(&scene, cfg(&scene, ghosts), batch),
                &whole,
                &format!("{id}, batches of {batch}"),
            );
        }
    }
}

#[test]
fn a_finished_job_is_the_trace_trace3_gives() {
    for (id, ghosts) in [("cemented-achromat", true), ("petzval-lens", false)] {
        let scene = design(id);
        let job = run(&scene, cfg(&scene, ghosts), 16);
        let one = trace3(&scene, &launch_from_scene(&scene), &cfg(&scene, ghosts));
        // The same rays make the same segments.
        assert_eq!(segments(&job), segments(&one), "{id}: segments");
        assert_eq!(job.terminations, one.terminations, "{id}: terminations");
        // A parallel trace adds a detector's cells in whatever order its
        // threads finish; the sums agree to that order's rounding.
        for (a, b) in job.detectors.iter().zip(&one.detectors) {
            assert_eq!(a.hits, b.hits);
            for (x, y) in a.incident_flux.iter().zip(&b.incident_flux) {
                assert!(
                    (x - y).abs() <= 1e-12 * y.abs().max(1e-300),
                    "{id}: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn a_job_stopped_part_way_holds_the_rays_it_traced() {
    let scene = design("cooke-triplet");
    let mut job = TraceJob3::new(scene.clone(), cfg(&scene, false));
    let (_, all) = job.progress();
    assert!(!job.step(10));
    assert_eq!(job.progress(), (10, all));
    let part = job.finish();
    assert!((part.ledger.launched - 10.0 * scene.sources[0].flux / 64.0 / 3.0).abs() < 1e-12);
    assert!(part.segments.len() < 10 * 20);
}

/// A floor on the segments a trace records thins the picture and nothing
/// else: every number is the whole trace's.
#[test]
fn a_segment_floor_leaves_out_dim_segments_and_no_number() {
    let scene = design("cooke-triplet");
    let all = run(&scene, cfg(&scene, true), usize::MAX);
    let floor = 1e-5;
    let thin = run(
        &scene,
        TraceConfig3 {
            segment_floor: floor,
            ..cfg(&scene, true)
        },
        usize::MAX,
    );
    assert_eq!(numbers(&thin), numbers(&all));
    assert_eq!(thin.terminations, all.terminations);
    let brightest = launch_from_scene(&scene)
        .iter()
        .map(|l| l.flux)
        .fold(0.0, f64::max);
    let kept: Vec<_> = all
        .segments
        .iter()
        .filter(|s| s.flux_end >= floor * brightest)
        .cloned()
        .collect();
    let thin_only = TraceResult3 {
        segments: kept,
        ..thin.clone()
    };
    assert_eq!(segments(&thin), segments(&thin_only));
    assert_eq!(
        thin.segments.len() as u64 + thin.segments_omitted,
        all.segments.len() as u64
    );
    assert!(
        thin.segments_omitted > all.segments.len() as u64 / 3,
        "the ghosts are mostly dim"
    );
}

#[test]
fn the_nearest_first_walk_finds_what_the_brute_force_search_finds() {
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{GALLERY}/index.json")).unwrap())
            .unwrap();
    let small_enough_for_ghosts = [
        "cemented-achromat",
        "edmund-49-847",
        "cooke-triplet",
        "navarro-wide-angle-eye",
    ];
    for d in index["designs"].as_array().unwrap() {
        if d["status"] != "ported" {
            continue;
        }
        let id = d["id"].as_str().unwrap();
        let scene = design(id);
        for ghosts in [false, true] {
            if ghosts && !small_enough_for_ghosts.contains(&id) {
                continue;
            }
            let brute = TraceConfig3 {
                brute_force_search: true,
                ..cfg(&scene, ghosts)
            };
            assert_same(
                &run(&scene, cfg(&scene, ghosts), usize::MAX),
                &run(&scene, brute, usize::MAX),
                &format!("{id}, ghosts {ghosts}"),
            );
        }
    }
}

/// Every number a trace reports, as bits: each detector's sums, cells and hit
/// records, the ledger, and the omitted-segment count.
fn every_number(r: &TraceResult3) -> Vec<u64> {
    let mut v = Vec::new();
    for d in &r.detectors {
        v.push(d.hits);
        let sums = [
            d.total_incident,
            d.total_signal,
            d.flux_uv.x,
            d.flux_uv.y,
            d.flux_opl,
            d.stokes_local.s0,
            d.stokes_local.s1,
            d.stokes_local.s2,
            d.stokes_local.s3,
            d.max_tilt_rad,
            d.outside_cone_flux,
            d.spectral_untracked_flux,
        ];
        v.extend(sums.map(f64::to_bits));
        v.push(d.spectral.len() as u64);
        for line in &d.spectral {
            v.extend(line.to_array().map(f64::to_bits));
        }
        for cells in [&d.incident_flux, &d.detected_signal, &d.direction_flux] {
            v.push(cells.len() as u64);
            v.extend(cells.iter().map(|x| x.to_bits()));
        }
        v.push(d.colour_xyz.len() as u64);
        v.extend(d.colour_xyz.iter().flatten().map(|x| x.to_bits()));
        v.push(d.records.len() as u64);
        for h in &d.records {
            let fields = [
                h.point_world.x,
                h.point_world.y,
                h.point_world.z,
                h.uv_mm.x,
                h.uv_mm.y,
                h.incident_dir_local.x,
                h.incident_dir_local.y,
                h.incident_dir_local.z,
                h.lambda_um,
                h.incident_flux,
                h.signal,
                h.opl_mm,
                h.stokes.s0,
                h.stokes.s1,
                h.stokes.s2,
                h.stokes.s3,
                h.frame_x.x,
                h.frame_x.y,
                h.frame_x.z,
            ];
            v.extend(fields.map(f64::to_bits));
            v.extend([h.path_id.0, h.parent_path_id.map_or(u64::MAX, |p| p.0)]);
            v.extend([h.depth as u64, h.ghost_depth as u64]);
        }
    }
    let l = &r.ledger;
    let ledger = [
        l.launched,
        l.detected_incident,
        l.detected_signal,
        l.escaped,
        l.absorbed_volume,
        l.absorbed_surface,
        l.blocked_aperture,
        l.truncated_flux,
        l.numerical_loss,
    ];
    v.extend(ledger.map(f64::to_bits));
    v.push(r.segments_omitted);
    v
}

/// Segments as bits, in the order they were recorded.
fn in_order(segs: &[Segment3]) -> Vec<[u64; 9]> {
    segs.iter()
        .map(|s| {
            let flags = s.flags.ghost as u64 + 2 * s.flags.tir as u64;
            let v = [
                s.a.x,
                s.a.y,
                s.a.z,
                s.b.x,
                s.b.y,
                s.b.z,
                s.flux_start,
                s.flux_end,
            ];
            let b = v.map(f64::to_bits);
            [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], flags]
        })
        .collect()
}

#[test]
fn a_pool_sharing_the_blocks_out_gives_the_jobs_trace_to_the_bit() {
    for (id, ghosts) in [
        ("cooke-triplet", true),
        ("double-gauss", false),
        ("cemented-achromat", true),
    ] {
        let scene = design(id);
        let cfg = TraceConfig3 {
            segment_floor: 1e-5,
            // Few enough that one block fills them, so later blocks skip
            // their records and the merge must not miss them.
            max_records: 5,
            ..cfg(&scene, ghosts)
        };
        let job = run(&scene, cfg, 5);

        // Three workers, each with its own copy of the scene. The blocks are
        // traced out of order: the first, then the rest back to front.
        let workers: Vec<TraceBlocks3> = (0..3)
            .map(|_| TraceBlocks3::new(scene.clone(), cfg))
            .collect();
        let blocks = workers[0].blocks();
        assert!(blocks > 4, "{id}: enough blocks to share");
        let mut order = vec![0];
        order.extend((1..blocks).rev());
        let mut parts: Vec<Option<(EncodedPart3, Vec<Segment3>)>> =
            (0..blocks).map(|_| None).collect();
        let mut full: Vec<usize> = Vec::new();
        let mut skipped = 0;
        for (k, &b) in order.iter().enumerate() {
            let w = &workers[k % 3];
            // A block after one that filled every detector's records alone
            // keeps none: no merge could keep them.
            let record = !full.iter().any(|&f| f < b);
            skipped += !record as usize;
            let mut part = w.part(b, record);
            while !w.step(&mut part, 3) {}
            if part.records_full(cfg.max_records) {
                full.push(b);
            }
            parts[b] = Some((part.encode(), part.segments().to_vec()));
        }
        assert!(skipped > 0, "{id}: later blocks skipped their records");

        let mut merge = TraceMerge3::new(&workers[1]);
        let (second, _) = parts[1].as_ref().unwrap();
        assert!(merge.add_encoded(second).is_err(), "{id}: out of order");
        let mut picture = Vec::new();
        for p in &parts {
            let (numbers, segs) = p.as_ref().unwrap();
            merge.add_encoded(numbers).unwrap();
            picture.extend_from_slice(segs);
        }
        assert!(merge.done());
        let pooled = merge.finish();

        assert_eq!(every_number(&pooled), every_number(&job), "{id}: numbers");
        assert_eq!(pooled.terminations, job.terminations, "{id}: terminations");
        assert_eq!(
            pooled.capability_report, job.capability_report,
            "{id}: report"
        );
        assert!(pooled.segments.is_empty());
        assert_eq!(in_order(&picture), in_order(&job.segments), "{id}: picture");
    }
}

/// `scene` with every source emitting `k` times the rays: what the page's
/// analysis ray count traces its numbers with.
fn with_more_rays(scene: &Scene3, k: u32) -> Scene3 {
    let mut more = scene.clone();
    for src in &mut more.sources {
        src.samples.spatial_samples *= k;
    }
    more
}

/// Numbers traced with more rays than the picture shows: the numbers are the
/// denser scene's, to the bit, and the picture is the scene's own, segment
/// for segment and in order, whether one job traces it or a pool of workers
/// shares the blocks out.
#[test]
fn more_rays_for_the_numbers_leave_the_picture_as_it_was() {
    for (id, ghosts) in [("cooke-triplet", true), ("double-gauss", false)] {
        let scene = design(id);
        let more = with_more_rays(&scene, 4);
        let cfg = TraceConfig3 {
            segment_floor: 1e-5,
            ..cfg(&scene, ghosts)
        };
        let plain = run(&scene, cfg, usize::MAX);
        let dense = run(&more, cfg, usize::MAX);

        for batch in [usize::MAX, 7] {
            let mut job = TraceJob3::with_picture(more.clone(), Some(&scene), cfg);
            // Progress counts the numbers' rays, the ones the status line quotes.
            assert_eq!(
                job.progress().1,
                TraceBlocks3::new(more.clone(), cfg).rays()
            );
            while !job.step(batch) {}
            let split = job.finish();
            let mut numbers_only = every_number(&split);
            let mut dense_numbers = every_number(&dense);
            // The count of segments left out below the floor is the
            // picture's, like the segments themselves.
            numbers_only.pop();
            dense_numbers.pop();
            assert_eq!(numbers_only, dense_numbers, "{id}, batch {batch}: numbers");
            assert_eq!(split.terminations, dense.terminations, "{id}: terminations");
            assert_eq!(
                split.segments_omitted, plain.segments_omitted,
                "{id}: omitted"
            );
            assert_eq!(
                in_order(&split.segments),
                in_order(&plain.segments),
                "{id}: picture"
            );
        }

        // The same through a pool: each block traces its share of both.
        let workers: Vec<TraceBlocks3> = (0..2)
            .map(|_| TraceBlocks3::with_picture(more.clone(), Some(&scene), cfg))
            .collect();
        let mut merge = TraceMerge3::new(&workers[0]);
        let mut picture = Vec::new();
        for b in 0..workers[0].blocks() {
            let w = &workers[b % 2];
            let mut part = w.part(b, true);
            while !w.step(&mut part, 5) {}
            picture.extend_from_slice(part.segments());
            merge.add_encoded(&part.encode()).unwrap();
        }
        let pooled = merge.finish();
        let mut a = every_number(&pooled);
        let mut b = every_number(&dense);
        a.pop();
        b.pop();
        assert_eq!(a, b, "{id}: pooled numbers");
        assert_eq!(
            in_order(&picture),
            in_order(&plain.segments),
            "{id}: pooled picture"
        );
    }
}

/// Rays past one pass come as more passes, each the scene's point sets
/// shifted afresh: the totals stay one pass's, the hits grow with the
/// passes, and a pool whose workers hop between passes gives what one job
/// gives, to the bit.
#[test]
fn passes_add_rays_and_a_pool_agrees_with_a_job() {
    // Its detectors asked for illuminance, so the colour sums travel through
    // the pool with the rest.
    let mut scene = design("cooke-triplet");
    for d in &mut scene.detectors {
        if d.geometry.has_area() {
            d.measure = oc_core::scene3::DetectorMeasure::Illuminance;
        }
    }
    assert!(scene.detectors.iter().any(|d| d.measure == oc_core::scene3::DetectorMeasure::Illuminance));
    let cfg = cfg(&scene, false);
    let one = run(&scene, cfg, usize::MAX);

    let mut job = TraceJob3::with_passes(scene.clone(), 3, None, cfg);
    assert_eq!(job.progress().1, 3 * TraceBlocks3::new(scene.clone(), cfg).rays());
    while !job.step(7) {}
    let passes = job.finish();
    assert!((passes.ledger.launched - one.ledger.launched).abs() <= 1e-12 * one.ledger.launched);
    for (a, b) in passes.detectors.iter().zip(&one.detectors) {
        assert!(a.hits >= 2 * b.hits, "{} hits against {}", a.hits, b.hits);
        assert!(
            (a.total_incident - b.total_incident).abs() <= 0.02 * b.total_incident.max(1e-12),
            "{} against {}",
            a.total_incident,
            b.total_incident
        );
    }

    // Three workers, handed blocks round the table: each makes the passes it
    // reaches, in whatever order they come.
    let workers: Vec<TraceBlocks3> = (0..3)
        .map(|_| TraceBlocks3::with_passes(scene.clone(), 3, None, cfg))
        .collect();
    let mut merge = TraceMerge3::new(&workers[0]);
    for b in 0..workers[0].blocks() {
        let w = &workers[(b * 7) % 3];
        let mut part = w.part(b, true);
        while !w.step(&mut part, 11) {}
        merge.add_encoded(&part.encode()).unwrap();
    }
    let pooled = merge.finish();
    let mut a = every_number(&pooled);
    let mut b = every_number(&passes);
    a.pop();
    b.pop();
    assert_eq!(a, b, "pooled numbers");
}

#[test]
fn a_blocks_object_ledger_adds_up_to_the_merged_one() {
    let scene = design("cemented-achromat");
    let on = TraceConfig3 {
        object_ledger: true,
        ..cfg(&scene, true)
    };
    let blocks = TraceBlocks3::new(scene.clone(), on);
    let mut merge = TraceMerge3::new(&blocks);
    let mut sum: std::collections::BTreeMap<(ObjectRef3, bool), ObjectLedger3> =
        std::collections::BTreeMap::new();
    let mut rows = 0;
    for b in 0..blocks.blocks() {
        let mut part = blocks.part(b, true);
        while !blocks.step(&mut part, 1 << 20) {}
        for e in part.objects() {
            sum.entry((e.object, e.in_band)).or_default().add(&e.ledger);
            rows += 1;
        }
        merge.add(part).unwrap();
    }
    let merged = merge.finish().objects;
    assert!(rows > merged.len(), "several blocks met the same objects");
    let summed: Vec<ObjectEntry3> = sum
        .into_iter()
        .map(|((object, in_band), ledger)| ObjectEntry3 {
            object,
            in_band,
            ledger,
        })
        .collect();
    assert_eq!(summed, merged);

    // Without the object ledger a block has none to give.
    let off = TraceBlocks3::new(scene.clone(), cfg(&scene, true));
    let mut part = off.part(0, true);
    while !off.step(&mut part, 1 << 20) {}
    assert_eq!(part.objects().count(), 0);
}

#[test]
fn a_blocks_deposits_taken_out_are_the_merged_ones_in_order() {
    let mut scene = design("cemented-achromat");
    for b in &mut scene.bodies {
        b.absorption = oc_core::scene3::AbsorptionModel::Constant { alpha_per_mm: 0.05 };
    }
    let on = TraceConfig3 {
        deposit_bodies: [0, scene.bodies.len()],
        ..cfg(&scene, true)
    };
    let whole = run(&scene, on, usize::MAX).deposits;
    assert!(!whole.is_empty(), "the absorbing lens takes light");
    let blocks = TraceBlocks3::new(scene.clone(), on);
    let mut merge = TraceMerge3::new(&blocks);
    let mut taken = Vec::new();
    for b in 0..blocks.blocks() {
        let mut part = blocks.part(b, true);
        while !blocks.step(&mut part, 1 << 20) {}
        taken.extend(part.take_deposits());
        assert!(part.take_deposits().is_empty(), "taken once");
        merge.add(part).unwrap();
    }
    assert_eq!(taken, whole);
    let merged = merge.finish();
    assert!(merged.deposits.is_empty(), "the merge holds none");
    // Nothing else moved.
    assert_eq!(numbers(&merged), numbers(&run(&scene, on, usize::MAX)));
}
