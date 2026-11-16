//! An analysis run a batch of launch rays at a time.
//!
//! A job owns its copy of the scene, traces it in independent passes through
//! the kernel's block API ([`TraceBlocks3`]), hands each block's readouts to
//! the analysis before the block is merged, and builds the result from the
//! merged trace and the per-pass readings once every ray is traced.
//!
//! Most analyses trace the scene once. One that must tell the sources apart
//! (`channels`) traces it once per source, each trace the scene with that
//! source alone, one after the other. Each trace is its own run of blocks.
//!
//! The kernel cuts a trace into blocks and adds them up in block order, so a
//! block's sums do not depend on how the job was stepped. A job's passes are
//! whole runs of blocks, so its per-pass readings do not either: a job stepped
//! one ray at a time gives the result a job stepped all at once gives, to the
//! bit (`tests/job.rs`).

use oc_core::scene3::{Scene3, SpatialDistribution3};
use oc_trace::{TraceBlocks3, TraceConfig3, TraceMerge3, TracePart3, TraceResult3};
use serde::Serialize;
use serde_json::{json, Value};

use crate::channels::Channels;
use crate::detector_stats::DetectorStats;
use crate::request::{AnalysisRequest, Kind, MAX_PASS_RAYS};

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
}

/// One trace of a job: a scene traced in passes, a block at a time.
struct Run {
    blocks: TraceBlocks3,
    part: TracePart3,
    /// The blocks added up so far; taken when the last one is in.
    merge: Option<TraceMerge3>,
    blocks_per_pass: usize,
    /// Launch rays in the blocks already merged.
    merged_rays: u64,
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
        let blocks_in = self.merge.as_ref().map_or(self.blocks.blocks(), TraceMerge3::next_block);
        blocks_in / self.blocks_per_pass
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
        let traced = sampled_for(scene, per_pass);
        let analysis = match &request.kind {
            Kind::DetectorStats(p) => Analysis::DetectorStats(DetectorStats::new(&traced, p, passes)?),
            Kind::Channels(p) => Analysis::Channels(Channels::new(&traced, p, passes)?),
        };
        let cfg = TraceConfig3 {
            record_detector_hits: false,
            record_events: false,
            // No segment is as bright as infinity: nothing is drawn.
            segment_floor: f64::INFINITY,
            ..cfg
        };
        // The scenes traced: the scene, or each source alone. Each is traced
        // under the settings of the whole scene (its bounds and depth come
        // in `cfg`), so a source alone travels exactly as among the others.
        let scenes = match &analysis {
            Analysis::DetectorStats(_) => vec![traced.clone()],
            Analysis::Channels(_) => (0..traced.sources.len()).map(|i| source_alone(&traced, i)).collect(),
        };
        let runs = scenes
            .into_iter()
            .map(|s| Run::new(s, passes, cfg))
            .collect::<Result<Vec<_>, _>>()?;
        let mut job = Self {
            request,
            analysis,
            runs,
            current: 0,
            scene: traced,
        };
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
            let run = &mut self.runs[self.current];
            let before = run.part.progress().0 as u64;
            run.blocks.step(&mut run.part, left.min(usize::MAX as u64) as usize);
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
                }
                let next = (block + 1).min(run.blocks.blocks() - 1);
                let part = std::mem::replace(&mut run.part, run.blocks.part(next, false));
                run.merged_rays += part.progress().1 as u64;
                merge.add(part).expect("the job adds its blocks in order");
            }
            if !merge.done() {
                return;
            }
            run.total = run.merge.take().map(TraceMerge3::finish);
            self.current += 1;
        }
    }

    /// Whether every ray is traced.
    pub fn done(&self) -> bool {
        self.current >= self.runs.len()
    }

    pub fn progress(&self) -> Progress {
        let passes_done = self.runs.iter().map(Run::passes_done).sum::<usize>() / self.runs.len().max(1);
        Progress {
            traced: self.runs.iter().map(Run::traced).sum(),
            rays: self.runs.iter().map(Run::rays).sum(),
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
        let rays: u64 = self.runs.iter().map(Run::rays).sum();
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
            "sampling": {
                "rays": rays,
                "passes": self.request.passes,
                "raysPerPass": rays / self.request.passes as u64,
                "stderr": "batchMeans",
            },
            "result": body,
        }))
    }
}
