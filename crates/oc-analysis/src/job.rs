//! An analysis run a batch of launch rays at a time.
//!
//! A job owns its copy of the scene, traces it in independent passes through
//! the kernel's block API ([`TraceBlocks3`]), hands each block's readouts to
//! the analysis before the block is merged, and builds the result from the
//! merged trace and the per-pass readings once every ray is traced.
//!
//! Most analyses trace the scene once. One that must tell the sources apart
//! (`channels`, and `budget` by source) traces it once per source, each trace the scene with that
//! source alone, one after the other. Each trace is its own run of blocks.
//!
//! The kernel cuts a trace into blocks and adds them up in block order, so a
//! block's sums do not depend on how the job was stepped. A job's passes are
//! whole runs of blocks, so its per-pass readings do not either: a job stepped
//! one ray at a time gives the result a job stepped all at once gives, to the
//! bit (`tests/job.rs`).
//!
//! A `fluorescence` job traces in two stages: each source alone to excite
//! its dyes, and then, once every excitation block is in, one trace per pass
//! and dye of the light they emit, each built from that pass's absorption
//! ([`crate::fluorescence`]). The emission traces are built one at a time, as
//! the job reaches each, so it holds one of them at once.
//!
//! A job may stop before its last ray: one with a noise target
//! (`uniformity`) checks it as each checkpoint's last pass is merged, and
//! stops there once the target is met. A check reads whole passes only, so a
//! job stops at the same pass however it was stepped, and its result is the
//! same to the bit.

use oc_core::scene3::{Scene3, SpatialDistribution3};
use oc_trace::{TraceBlocks3, TraceConfig3, TraceMerge3, TracePart3, TraceResult3};
use serde::Serialize;
use serde_json::{json, Value};

use crate::budget::Budget;
use crate::channels::Channels;
use crate::detector_stats::DetectorStats;
use crate::fluorescence::{emission_share, Fluorescence};
use crate::focus::{Pilot, DEFENSIVE_SHARE, MAX_PILOT_RAYS, MIN_PILOT_RAYS};
use crate::request::{AnalysisRequest, Kind, MAX_PASS_RAYS};
use crate::uniformity::Uniformity;

/// What a job has traced so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    /// Launch rays traced, and in all.
    pub traced: u64,
    pub rays: u64,
    /// Passes complete, and in all. A job of several traces counts the
    /// passes each has done and gives their mean, rounded down.
    pub passes_done: u32,
    pub passes: u32,
}

/// The analysis a job runs, holding its per-pass readings.
enum Analysis {
    DetectorStats(DetectorStats),
    Channels(Channels),
    Uniformity(Uniformity),
    Budget(Budget),
    Fluorescence(Fluorescence),
}

/// One trace of a job: a scene traced in passes, a block at a time.
struct Run {
    blocks: TraceBlocks3,
    part: TracePart3,
    /// The blocks added up so far; taken when the last one is in.
    merge: Option<TraceMerge3>,
    blocks_per_pass: usize,
    /// Launch rays in the blocks already merged, and the blocks.
    merged_rays: u64,
    merged_blocks: usize,
    /// The merged trace, once every block is in.
    total: Option<TraceResult3>,
}

impl Run {
    fn new(scene: Scene3, passes: u32, cfg: TraceConfig3) -> Result<Self, String> {
        let blocks = TraceBlocks3::with_passes(scene, passes, None, cfg);
        if blocks.blocks() % passes as usize != 0 {
            return Err(format!(
                "{passes} passes do not split the trace's {} blocks evenly; ask for a number of passes that divides {}",
                blocks.blocks(),
                blocks.blocks()
            ));
        }
        let blocks_per_pass = blocks.blocks() / passes as usize;
        let part = blocks.part(0, false);
        let merge = TraceMerge3::new(&blocks);
        Ok(Self {
            blocks,
            part,
            merge: Some(merge),
            blocks_per_pass,
            merged_rays: 0,
            merged_blocks: 0,
            total: None,
        })
    }

    fn done(&self) -> bool {
        self.total.is_some()
    }

    fn rays(&self) -> u64 {
        self.blocks.rays() as u64
    }

    fn traced(&self) -> u64 {
        let current = if self.done() { 0 } else { self.part.progress().0 as u64 };
        self.merged_rays + current
    }

    fn passes_done(&self) -> usize {
        self.merged_blocks / self.blocks_per_pass
    }

    /// Launch rays in the blocks merged: every ray once the run is done,
    /// unless it stopped early.
    fn merged(&self) -> u64 {
        self.merged_rays
    }
}

/// An analysis of one scene, traced a batch of launch rays at a time.
pub struct AnalysisJob {
    request: AnalysisRequest,
    analysis: Analysis,
    /// The traces, run in order.
    runs: Vec<Run>,
    /// The run being traced; `runs.len()` once every run is in.
    current: usize,
    /// The scene as traced, for labels in the result.
    scene: Scene3,
    /// The settings every trace of the job runs under.
    cfg: TraceConfig3,
    /// The pilot of a focused job, traced before the runs.
    pilot: Option<Pilot>,
    /// What the pilot cut, once it is done.
    focus: Option<Value>,
    /// A fluorescence job's emission trace in hand, and the next one's
    /// index: the emission traces follow the runs.
    emission: Option<Run>,
    next_emission: usize,
    /// The settings the emission traces run under: no deposits.
    emission_cfg: TraceConfig3,
}

/// The launch rays one pass of `scene` emits: every origin with every
/// direction, once per line of a line spectrum.
pub fn launch_count(scene: &Scene3) -> u64 {
    scene
        .sources
        .iter()
        .map(|src| {
            src.spectrum.ray_multiplicity() as u64
                * src.samples.spatial_samples.max(1) as u64
                * src.samples.angular_samples.max(1) as u64
        })
        .sum()
}

/// `scene` with its sources sampled for about `per_pass` launch rays in each
/// pass. A source with an extent gets more (or fewer) origins; a point
/// source, which has one, more directions. Flux per ray changes to match, so
/// every total stays what it was and only the noise moves.
pub fn sampled_for(scene: &Scene3, per_pass: u64) -> Scene3 {
    let own = launch_count(scene);
    let mut out = scene.clone();
    if own == 0 || own == per_pass {
        return out;
    }
    let k = per_pass as f64 / own as f64;
    for src in &mut out.sources {
        let s = &mut src.samples;
        let scale = |n: u32| ((n.max(1) as f64 * k).round().max(1.0)).min(u32::MAX as f64) as u32;
        if matches!(src.spatial, SpatialDistribution3::Point) {
            s.angular_samples = scale(s.angular_samples);
        } else {
            s.spatial_samples = scale(s.spatial_samples);
        }
    }
    out
}

/// `scene` with its `i`th source alone: what a `channels` job traces for
/// that source.
pub fn source_alone(scene: &Scene3, i: usize) -> Scene3 {
    let mut out = scene.clone();
    out.sources = vec![scene.sources[i].clone()];
    out
}

impl AnalysisJob {
    /// Begin `request` on `scene`, traced under `cfg` (whose `revision` the
    /// result carries). The job records no segments and no hit records: it
    /// reads sums. Errors name what is wrong with the request.
    pub fn new(scene: &Scene3, cfg: TraceConfig3, request: AnalysisRequest) -> Result<Self, String> {
        let passes = request.passes;
        let own = launch_count(scene);
        if own == 0 {
            return Err("the scene emits no rays".into());
        }
        let per_pass = request
            .rays
            .map_or(own, |r| r.div_ceil(passes as u64))
            .min(MAX_PASS_RAYS);
        // A fluorescence job splits the rays between its two stages; without
        // a count, the excitation traces the scene's own.
        let (per_pass, emission_per_pass) = match &request.kind {
            Kind::Fluorescence(p) => {
                let share = emission_share(p)?;
                let exc = match request.rays {
                    Some(r) => (r as f64 / passes as f64 * (1.0 - share)).round().max(1.0) as u64,
                    None => own,
                };
                let em = match request.rays {
                    Some(r) => (r as f64 / passes as f64 * share).round(),
                    None => (own as f64 * share / (1.0 - share)).round(),
                };
                (exc.min(MAX_PASS_RAYS), (em.max(1.0) as u64).min(MAX_PASS_RAYS))
            }
            _ => (per_pass, 0),
        };
        let traced = sampled_for(scene, per_pass);
        let analysis = match &request.kind {
            Kind::DetectorStats(p) => Analysis::DetectorStats(DetectorStats::new(&traced, p, passes)?),
            Kind::Channels(p) => Analysis::Channels(Channels::new(&traced, p, passes)?),
            Kind::Uniformity(p) => Analysis::Uniformity(Uniformity::new(&traced, p, passes)?),
            Kind::Budget(p) => Analysis::Budget(Budget::new(&traced, p, passes)?),
            Kind::Fluorescence(p) => {
                // Each dye's share of the emission rays of a pass.
                let each = (emission_per_pass / p.dyes.len().max(1) as u64).max(1);
                Analysis::Fluorescence(Fluorescence::new(&traced, p, passes, each)?)
            }
        };
        let cfg = TraceConfig3 {
            record_detector_hits: false,
            record_events: false,
            // No segment is as bright as infinity: nothing is drawn.
            segment_floor: f64::INFINITY,
            // Where the light was lost, part by part: changes no other number.
            object_ledger: matches!(analysis, Analysis::Budget(_)),
            ..cfg
        };
        let emission_cfg = cfg;
        // A fluorescence job's excitation records where its dyes absorb.
        let cfg = match &analysis {
            Analysis::Fluorescence(f) => TraceConfig3 {
                deposit_bodies: f.deposit_bodies(),
                ..cfg
            },
            _ => cfg,
        };
        // The scenes traced: the scene, or each source alone. Each is traced
        // under the settings of the whole scene (its bounds and depth come
        // in `cfg`), so a source alone travels exactly as among the others.
        let scenes = match &analysis {
            Analysis::DetectorStats(_) => vec![traced.clone()],
            Analysis::Channels(_) | Analysis::Fluorescence(_) => {
                (0..traced.sources.len()).map(|i| source_alone(&traced, i)).collect()
            }
            Analysis::Uniformity(u) => vec![u.traced_scene(&traced)],
            Analysis::Budget(b) if b.by_source() => {
                (0..traced.sources.len()).map(|i| source_alone(&traced, i)).collect()
            }
            Analysis::Budget(_) => vec![traced.clone()],
        };
        let runs = scenes
            .into_iter()
            .map(|s| Run::new(s, passes, cfg))
            .collect::<Result<Vec<_>, _>>()?;
        // A focused job traces its pilot first, on the scene as traced.
        let pilot = match &analysis {
            Analysis::Uniformity(u) => u.focused.map(|rays| {
                // A quarter of the job's own rays, unless the request says.
                let budget = rays.unwrap_or_else(|| {
                    (launch_count(&traced) * passes as u64 / 4).clamp(MIN_PILOT_RAYS, MAX_PILOT_RAYS)
                });
                Pilot::new(&u.traced_scene(&traced), cfg, budget)
            }),
            _ => None,
        };
        let mut job = Self {
            request,
            analysis,
            runs,
            current: 0,
            scene: traced,
            cfg,
            pilot,
            focus: None,
            emission: None,
            next_emission: 0,
            emission_cfg,
        };
        if job.pilot.as_ref().is_some_and(Pilot::done) {
            job.focus_runs();
        }
        job.advance();
        Ok(job)
    }

    /// The request the job runs.
    pub fn request(&self) -> &AnalysisRequest {
        &self.request
    }

    /// Trace the next `rays` launch rays (at least one). True once every ray
    /// is traced and the result is ready.
    pub fn step(&mut self, rays: u64) -> bool {
        let mut left = rays.max(1);
        while left > 0 && !self.done() {
            if let (Some(pilot), Analysis::Uniformity(u)) = (self.pilot.as_mut(), &self.analysis) {
                if !pilot.done() {
                    left = left.saturating_sub(pilot.step(left, u));
                    if pilot.done() {
                        self.focus_runs();
                    }
                    continue;
                }
            }
            if self.current >= self.runs.len() {
                // The runs are in: a fluorescence job's emission traces.
                left = left.saturating_sub(self.step_emission(left));
                continue;
            }
            let run = &mut self.runs[self.current];
            let (before, each) = run.part.progress();
            let before = before as u64;
            // No more than the block holds, so the kernel's count of rays to
            // trace cannot overflow.
            run.blocks.step(&mut run.part, left.min(each as u64 - before) as usize);
            left = left.saturating_sub(run.part.progress().0 as u64 - before);
            self.advance();
        }
        self.done()
    }

    /// Hand the part in hand to the analysis and the merge while it is
    /// complete, begin the next block, finish a run once every block is in,
    /// and go on to the next run.
    fn advance(&mut self) {
        while let Some(run) = self.runs.get_mut(self.current) {
            let mut stopped = false;
            let Some(merge) = run.merge.as_mut() else {
                self.current += 1;
                continue;
            };
            while run.part.done() && !merge.done() {
                let block = merge.next_block();
                let pass = block / run.blocks_per_pass;
                // A pass is whole blocks: the block starts in the pass it is
                // counted to.
                let per_pass = (run.blocks.rays() / self.request.passes as usize) as u64;
                debug_assert_eq!(run.merged_rays / per_pass.max(1), pass as u64);
                match &mut self.analysis {
                    Analysis::DetectorStats(a) => a.add_block(pass, run.part.detectors()),
                    Analysis::Channels(a) => a.add_block(self.current, pass, run.part.detectors()),
                    Analysis::Uniformity(a) => a.add_block(pass, run.part.detectors()),
                    Analysis::Budget(a) => {
                        a.add_block(
                            self.current,
                            pass,
                            run.part.ledger(),
                            run.part.detectors(),
                            run.part.objects(),
                        )
                    }
                    Analysis::Fluorescence(a) => a.add_excitation_block(self.current, pass, &mut run.part),
                }
                let next = (block + 1).min(run.blocks.blocks() - 1);
                let part = std::mem::replace(&mut run.part, run.blocks.part(next, false));
                run.merged_rays += part.progress().1 as u64;
                run.merged_blocks += 1;
                merge.add(part).expect("the job adds its blocks in order");
                // A pass is complete: a job with a target may stop on it.
                if run.merged_blocks % run.blocks_per_pass == 0 {
                    if let Analysis::Uniformity(a) = &mut self.analysis {
                        if a.stop_after(run.merged_blocks / run.blocks_per_pass) {
                            stopped = true;
                            break;
                        }
                    }
                }
            }
            if !merge.done() && !stopped {
                return;
            }
            run.total = run.merge.take().map(TraceMerge3::finish);
            self.current += 1;
        }
        // Every excitation block is in: the emission traces are fixed.
        if let Analysis::Fluorescence(f) = &mut self.analysis {
            f.plan();
            self.begin_emission();
        }
    }

    /// Begin the next emission trace that has light to trace, unless one is
    /// in hand. One with none (the dye absorbed nothing in that pass) is
    /// passed over: it reads zero.
    fn begin_emission(&mut self) {
        let Analysis::Fluorescence(f) = &mut self.analysis else { return };
        while self.emission.is_none() && self.next_emission < f.emission_traces() {
            match f.emission_scene(self.next_emission, &self.scene) {
                Some(scene) => {
                    let run = Run::new(scene, 1, self.emission_cfg).expect("one pass divides any blocks");
                    self.emission = Some(run);
                }
                None => self.next_emission += 1,
            }
        }
    }

    /// Trace up to `rays` launch rays of the emission trace in hand; the rays
    /// traced. Blocks are added in order, as for a run.
    fn step_emission(&mut self, rays: u64) -> u64 {
        let Some(run) = self.emission.as_mut() else { return rays.max(1) };
        let (before, each) = run.part.progress();
        let before = before as u64;
        run.blocks.step(&mut run.part, rays.min(each as u64 - before).max(1) as usize);
        let traced = run.part.progress().0 as u64 - before;
        let Analysis::Fluorescence(f) = &mut self.analysis else { return traced };
        let k = self.next_emission;
        let merge = run.merge.as_mut().expect("a trace in hand is not merged");
        while run.part.done() && !merge.done() {
            let block = merge.next_block();
            f.add_emission_block(k, run.part.ledger(), run.part.detectors());
            let next = (block + 1).min(run.blocks.blocks() - 1);
            let part = std::mem::replace(&mut run.part, run.blocks.part(next, false));
            run.merged_rays += part.progress().1 as u64;
            run.merged_blocks += 1;
            merge.add(part).expect("the job adds its blocks in order");
        }
        if merge.done() {
            let total = run.merge.take().map(TraceMerge3::finish).expect("merged");
            f.finish_emission(&total);
            self.emission = None;
            self.next_emission += 1;
            self.begin_emission();
        }
        traced
    }

    /// The emission rays traced so far.
    fn emission_traced(&self) -> u64 {
        let Analysis::Fluorescence(f) = &self.analysis else { return 0 };
        f.emission_rays_before(self.next_emission) + self.emission.as_ref().map_or(0, Run::traced)
    }

    /// The pilot is done: trace the scene it found in place of the one the
    /// job began with, unless it cut nothing.
    fn focus_runs(&mut self) {
        let Some(pilot) = &self.pilot else { return };
        let (scene, mut report) = pilot.focused();
        let cut = report
            .as_array()
            .is_some_and(|r| r.iter().any(|s| s["focused"] == true));
        if cut {
            match Run::new(scene, self.request.passes, self.cfg) {
                Ok(run) => self.runs[0] = run,
                Err(_) => {
                    // Too few rays for the pieces to split into the passes:
                    // the job traces the sources as they are.
                    for s in report.as_array_mut().into_iter().flatten() {
                        *s = json!({ "source": s["source"], "focused": false, "reason": "too few rays to cut" });
                    }
                }
            }
        }
        self.focus = Some(report);
        self.advance();
    }

    /// Whether every ray is traced.
    pub fn done(&self) -> bool {
        let emitted = match &self.analysis {
            Analysis::Fluorescence(f) => self.emission.is_none() && self.next_emission >= f.emission_traces(),
            _ => true,
        };
        self.pilot.as_ref().map_or(true, Pilot::done) && self.current >= self.runs.len() && emitted
    }

    pub fn progress(&self) -> Progress {
        let passes_done = self.runs.iter().map(Run::passes_done).sum::<usize>() / self.runs.len().max(1);
        let (pilot_traced, pilot_rays) = self.pilot.as_ref().map_or((0, 0), |p| (p.traced(), p.total()));
        // A fluorescence job: its emission traces too, a pass done once each
        // dye's is, and the passes done the mean of its two stages'.
        let (em_traced, em_rays, passes_done) = match &self.analysis {
            Analysis::Fluorescence(f) => {
                let dyes = (f.emission_traces() / self.request.passes as usize).max(1);
                let em_done = if self.current >= self.runs.len() {
                    (self.next_emission / dyes).min(self.request.passes as usize)
                } else {
                    0
                };
                (self.emission_traced(), f.emission_rays_total(), (passes_done + em_done) / 2)
            }
            _ => (0, 0, passes_done),
        };
        Progress {
            traced: pilot_traced + self.runs.iter().map(Run::traced).sum::<u64>() + em_traced,
            rays: pilot_rays + self.runs.iter().map(Run::rays).sum::<u64>() + em_rays,
            passes_done: passes_done as u32,
            passes: self.request.passes,
        }
    }

    /// The result envelope, once [`AnalysisJob::done`]; `None` before.
    ///
    /// ```json
    /// { "kind": "detectorStats", "revision": 4,
    ///   "fidelity": { "profile": {..}, "preset": "Standard", "text": "..",
    ///                 "approximations": [..], "solverVersion": ".." },
    ///   "approximations": [{ "feature": "Mirror", "detail": ".." }],
    ///   "rejected": [],
    ///   "sampling": { "rays": 200000, "passes": 16, "raysPerPass": 12500,
    ///                 "stderr": "batchMeans" },
    ///   "result": { .. per kind .. } }
    /// ```
    ///
    /// A job of several traces states what any of them was traced with, or
    /// could not be, once each.
    pub fn result(&self) -> Option<Value> {
        if !self.done() {
            return None;
        }
        let totals: Vec<&TraceResult3> = self.runs.iter().filter_map(|r| r.total.as_ref()).collect();
        let first = *totals.first()?;
        let body = match &self.analysis {
            Analysis::DetectorStats(a) => a.result(&self.scene, first),
            Analysis::Channels(a) => {
                let rays: Vec<u64> = self.runs.iter().map(Run::rays).collect();
                a.result(&self.scene, &totals, &rays)
            }
            Analysis::Uniformity(a) => {
                let run = &self.runs[0];
                a.result(&self.scene, first, run.merged(), run.rays())
            }
            Analysis::Budget(a) => {
                let rays: Vec<u64> = self.runs.iter().map(Run::rays).collect();
                a.result(&self.scene, &totals, &rays)
            }
            Analysis::Fluorescence(a) => {
                let rays: Vec<u64> = self.runs.iter().map(Run::rays).collect();
                a.result(&self.scene, &totals, &rays)
            }
        };
        let stamp = &first.fidelity;
        let mut approximated = Vec::new();
        let mut rejected = Vec::new();
        for t in &totals {
            let cap = &t.capability_report;
            for a in &cap.approximated {
                if !approximated.contains(a) {
                    approximated.push(a.clone());
                }
            }
            for r in &cap.rejected {
                if !rejected.contains(r) {
                    rejected.push(r.clone());
                }
            }
        }
        if let Analysis::Fluorescence(f) = &self.analysis {
            let (a, r) = f.reports();
            for x in a {
                if !approximated.contains(x) {
                    approximated.push(x.clone());
                }
            }
            for x in r {
                if !rejected.contains(x) {
                    rejected.push(x.clone());
                }
            }
        }
        // The rays and passes traced: all of them, unless the job stopped on
        // its target.
        let rays: u64 = self.runs.iter().map(Run::merged).sum();
        let passes = self.runs.iter().map(Run::passes_done).min().unwrap_or(0) as u64;
        let mut sampling = json!({
            "rays": rays,
            "passes": passes,
            "raysPerPass": rays / passes.max(1),
            "stderr": "batchMeans",
        });
        if let Analysis::Fluorescence(f) = &self.analysis {
            // Both stages: `rays` is their sum, and each says its own. Each
            // emission pass is built from its excitation pass, so the errors
            // carry the noise of both.
            let s = sampling.as_object_mut().expect("an object");
            let excitation = rays;
            let emission = f.sampling();
            let all = excitation + emission["rays"].as_u64().unwrap_or(0);
            s.insert("rays".into(), json!(all));
            s.insert("raysPerPass".into(), json!(all / passes.max(1)));
            s.insert(
                "excitation".into(),
                json!({ "rays": excitation, "raysPerPass": excitation / passes.max(1), "traces": self.runs.len() }),
            );
            s.insert("emission".into(), emission);
            s.insert("stderrIncludesExcitation".into(), json!(true));
        }
        if let Some(pilot) = &self.pilot {
            // A focused job: what its pilot traced and what it cut. The
            // pilot's light is in no number: `rays` are the job's own.
            let s = sampling.as_object_mut().expect("an object");
            s.insert("mode".into(), json!("focused"));
            s.insert("pilotRays".into(), json!(pilot.traced()));
            s.insert("pilot".into(), pilot.report());
            s.insert("defensiveShare".into(), json!(DEFENSIVE_SHARE));
            s.insert("focus".into(), self.focus.clone().unwrap_or(Value::Null));
        }
        Some(json!({
            "kind": self.request.kind.name(),
            "revision": first.revision,
            "fidelity": {
                "profile": stamp.profile,
                "preset": stamp.preset.map(|p| p.name()),
                "text": stamp.text(),
                "approximations": stamp.approximations,
                "solverVersion": stamp.solver_version,
            },
            "approximations": approximated,
            "rejected": rejected,
            "sampling": sampling,
            "result": body,
        }))
    }
}
