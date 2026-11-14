//! An analysis run a batch of launch rays at a time.
//!
//! A job owns its copy of the scene, traces it in independent passes through
//! the kernel's block API ([`TraceBlocks3`]), hands each block's readouts to
//! the analysis before the block is merged, and builds the result from the
//! merged trace and the per-pass readings once every ray is traced.
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

use crate::detector_stats::DetectorStats;
use crate::request::{AnalysisRequest, Kind, MAX_PASS_RAYS};

/// What a job has traced so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    /// Launch rays traced, and in all.
    pub traced: u64,
    pub rays: u64,
    /// Passes complete, and in all.
    pub passes_done: u32,
    pub passes: u32,
}

/// The analysis a job runs, holding its per-pass readings.
enum Analysis {
    DetectorStats(DetectorStats),
}

/// An analysis of one scene, traced a batch of launch rays at a time.
pub struct AnalysisJob {
    request: AnalysisRequest,
    analysis: Analysis,
    blocks: TraceBlocks3,
    part: TracePart3,
    /// The blocks added up so far; taken when the last one is in.
    merge: Option<TraceMerge3>,
    blocks_per_pass: usize,
    /// Launch rays in the blocks already merged.
    merged_rays: u64,
    /// The scene as traced, for labels in the result.
    scene: Scene3,
    /// The merged trace, once every block is in.
    total: Option<TraceResult3>,
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
        };
        let cfg = TraceConfig3 {
            record_detector_hits: false,
            record_events: false,
            // No segment is as bright as infinity: nothing is drawn.
            segment_floor: f64::INFINITY,
            ..cfg
        };
        let blocks = TraceBlocks3::with_passes(traced.clone(), passes, None, cfg);
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
        let mut job = Self {
            request,
            analysis,
            blocks,
            part,
            merge: Some(merge),
            blocks_per_pass,
            merged_rays: 0,
            scene: traced,
            total: None,
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
            let before = self.part.progress().0 as u64;
            self.blocks.step(&mut self.part, left.min(usize::MAX as u64) as usize);
            left = left.saturating_sub(self.part.progress().0 as u64 - before);
            self.advance();
        }
        self.done()
    }

    /// Hand the part in hand to the analysis and the merge while it is
    /// complete, begin the next block, and finish once every block is in.
    fn advance(&mut self) {
        let Some(merge) = self.merge.as_mut() else {
            return;
        };
        while self.part.done() && !merge.done() {
            let block = merge.next_block();
            let pass = block / self.blocks_per_pass;
            // A pass is whole blocks: the block starts in the pass it is
            // counted to.
            let per_pass = (self.blocks.rays() / self.request.passes as usize) as u64;
            debug_assert_eq!(self.merged_rays / per_pass.max(1), pass as u64);
            match &mut self.analysis {
                Analysis::DetectorStats(a) => a.add_block(pass, self.part.detectors()),
            }
            let next = (block + 1).min(self.blocks.blocks() - 1);
            let part = std::mem::replace(&mut self.part, self.blocks.part(next, false));
            self.merged_rays += part.progress().1 as u64;
            merge.add(part).expect("the job adds its blocks in order");
        }
        if merge.done() {
            self.total = self.merge.take().map(TraceMerge3::finish);
        }
    }

    /// Whether every ray is traced.
    pub fn done(&self) -> bool {
        self.total.is_some()
    }

    pub fn progress(&self) -> Progress {
        let current = if self.done() { 0 } else { self.part.progress().0 as u64 };
        let traced = self.merged_rays + current;
        let blocks_in = self.merge.as_ref().map_or(self.blocks.blocks(), TraceMerge3::next_block);
        Progress {
            traced,
            rays: self.blocks.rays() as u64,
            passes_done: (blocks_in / self.blocks_per_pass) as u32,
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
    pub fn result(&self) -> Option<Value> {
        let total = self.total.as_ref()?;
        let body = match &self.analysis {
            Analysis::DetectorStats(a) => a.result(&self.scene, total),
        };
        let stamp = &total.fidelity;
        let cap = &total.capability_report;
        Some(json!({
            "kind": self.request.kind.name(),
            "revision": total.revision,
            "fidelity": {
                "profile": stamp.profile,
                "preset": stamp.preset.map(|p| p.name()),
                "text": stamp.text(),
                "approximations": stamp.approximations,
                "solverVersion": stamp.solver_version,
            },
            "approximations": cap.approximated,
            "rejected": cap.rejected,
            "sampling": {
                "rays": self.blocks.rays(),
                "passes": self.request.passes,
                "raysPerPass": self.blocks.rays() / self.request.passes as usize,
                "stderr": "batchMeans",
            },
            "result": body,
        }))
    }
}
