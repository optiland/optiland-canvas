//! The f64 full-3D non-sequential reference tracer (§9).
//!
//! The **authority**: CPU, f64, exact, deterministic. Any 3D number the UI shows
//! must come from here, and the fast f32/GPU preview is validated against it
//! (§3.3). The shape of a step:
//!
//! 1. Find the nearest event across bodies, surfaces, apertures, and detectors
//!    (§9.2), grouped within an event tolerance and resolved by a fixed policy
//!    (§9.3) — never by vector order.
//! 2. Resolve the media on both sides of a boundary by containment plus priority
//!    (§9.4), never a mutable medium stack.
//! 3. Attenuate over the traversed path (Beer–Lambert), accumulate OPL (§9.5).
//! 4. Interact: 3D Snell, scalar Fresnel, exact TIR, or the surface's explicit
//!    response, splitting deterministic reflected and transmitted branches
//!    (§9.6).
//! 5. Account every photon: the [`EnergyLedger`] balances, and flux removed by a
//!    threshold or a cap is named, not dropped (§9.8).
//!
//! Deterministic reference transport means Russian roulette stays off: a branch
//! is followed until it is absorbed, detected, blocked, escapes, or hits a hard
//! cap, and a cap deposits the remaining flux in `truncated_flux`.

use oc_core::geom3::solid::SolidPrep;
use oc_core::geom3::{Containment, Intersect3, ToleranceContext};
use oc_core::id::{ObjectId, PathId, SurfaceRef};
use oc_core::scene3::{
    Aperture3, Detector3, DetectorGeometry3, DetectorTermination, PlanarShape2, Scene3,
    SolidGeometry3, SurfaceObject3, SurfaceResponse,
};
use oc_core::vec::Vec2;
use oc_core::vec3::{Aabb3, Isometry3, Ray3, Vec3};

use std::ops::Range;

use crate::bvh3::Bvh3;
use crate::fresnel::{beer_lambert, reflect3, refract3};
use crate::result3::*;
use oc_core::polarization::{self, Mueller, PolRay, Stokes};

/// Reference-tracer configuration. The reference is the exact path, so its
/// defaults follow ghosts as far as the depth cap allows; the interactive preset
/// lives in the f32 path (§3.6).
#[derive(Debug, Clone, Copy)]
pub struct TraceConfig3 {
    pub max_depth: u32,
    pub flux_eps: f64,
    pub ghosts: bool,
    pub max_ghost_bounces: u32,
    pub bounds_mm: f64,
    pub max_branches: u64,
    /// Keep per-hit detector records (bounded by `max_records`).
    pub record_detector_hits: bool,
    pub max_records: usize,
    /// Record the interaction-level event log ([`TraceEvent3`]): which scene
    /// object each branch met, what it did there, and which children it
    /// spawned. Off by default — the log is for sequence extraction and
    /// diagnostics, not the render path.
    pub record_events: bool,
    /// The scene revision, copied into the result so a stale trace is discarded
    /// rather than shown (§3.3).
    pub revision: u64,
    /// Record a segment only when the flux it ends with is at least this
    /// fraction of the brightest launch ray's; 0 records every one. A budget
    /// for the picture, which cannot show a segment much dimmer than 1e-5 of
    /// the brightest, and for memory: a ghost-lit twelve-element lens makes 57
    /// million segments, most of them far dimmer than that. The ledger, the
    /// detectors and the event log see every branch whatever this is, and
    /// [`TraceResult3::segments_omitted`] counts what was left out.
    pub segment_floor: f64,
    /// Ask every object whose box the ray crosses for its hits at every step,
    /// rather than walking the boxes nearest first and passing over those
    /// beyond the nearest hit. The same events either way, to the bit; this is
    /// the oracle the faster walk is tested against (`tests/nearest3.rs`).
    pub brute_force_search: bool,
    /// Carry a polarization state on every ray, and let each interface act on
    /// it. On by default, because it is the physics.
    ///
    /// Turning it off is the **scalar-unpolarized approximation**: every
    /// element sees unpolarized light, so it uses the mean of the s and p
    /// powers and hands on unpolarized light. That is exactly right for one
    /// interaction and wrong from the second polarization-sensitive element
    /// onward. It is what the f32 preview does, so the parity suite runs the
    /// reference this way to compare like with like.
    pub polarization: bool,
    /// Book every removal against the object where it happened
    /// ([`TraceResult3::objects`]). Off by default: it costs a map update per
    /// event, and it changes no number the trace reports otherwise.
    pub object_ledger: bool,
    /// The band, µm, `[lo, hi)`, the object ledger keeps apart from the rest.
    pub ledger_band_um: [f64; 2],
    /// The bodies, by scene index `[lo, hi)`, whose hits every branch counts
    /// ([`DetectorHit3::counted`]): the sections of a light pipe, say. Empty by
    /// default.
    pub count_bodies: [usize; 2],
    /// The bodies, by scene index `[lo, hi)`, whose absorbed power is recorded
    /// where it happened ([`TraceResult3::deposits`]): the thermal load of a
    /// lens. Empty by default.
    pub deposit_bodies: [usize; 2],
    /// Record where each stop or mask blocks light, too.
    pub deposit_apertures: bool,
    /// Record what standalone coated surfaces absorb, too, not only bodies
    /// (coating-fidelity spec CF-04).
    pub deposit_surfaces: bool,
    /// How much coating physics the trace pays for (coating-fidelity spec
    /// §5). The default reproduces the kernel as it was before the profile
    /// existed. `polarization` above is the older spelling of one switch and
    /// stays as an alias: the trace is polarized only when both say so.
    pub fidelity: oc_core::fidelity::FidelityProfile,
}

impl TraceConfig3 {
    /// Is the trace polarized? The deprecated `polarization` flag and the
    /// profile's `coating_model` must both say so.
    pub fn polarized(&self) -> bool {
        self.polarization && self.fidelity.polarized()
    }

    /// This config with a profile applied: the profile's ghost setting, when
    /// it has one, replaces the config's own.
    pub fn with_fidelity(mut self, p: oc_core::fidelity::FidelityProfile) -> Self {
        self.fidelity = p;
        if let Some(g) = p.ghosts {
            self.ghosts = g > 0;
            if g > 0 {
                self.max_ghost_bounces = g;
            }
        }
        self
    }

    /// The profile the trace actually runs under, the alias and the ghost
    /// setting folded in: what its results are stamped with.
    pub fn effective_profile(&self) -> oc_core::fidelity::FidelityProfile {
        use oc_core::fidelity::CoatingModel;
        let mut p = self.fidelity;
        if !self.polarization && p.coating_model == CoatingModel::Polarized {
            p.coating_model = CoatingModel::Unpolarized;
        }
        p.ghosts = Some(if self.ghosts { self.max_ghost_bounces } else { 0 });
        p
    }
}

impl Default for TraceConfig3 {
    fn default() -> Self {
        Self {
            max_depth: 24,
            flux_eps: 1e-6,
            ghosts: true,
            max_ghost_bounces: 8,
            bounds_mm: 1.0e4,
            max_branches: 100_000,
            record_detector_hits: false,
            max_records: 4096,
            record_events: false,
            segment_floor: 0.0,
            brute_force_search: false,
            revision: 0,
            polarization: true,
            object_ledger: false,
            ledger_band_um: [0.39, 0.70],
            count_bodies: [0, 0],
            deposit_bodies: [0, 0],
            deposit_apertures: false,
            deposit_surfaces: false,
            fidelity: oc_core::fidelity::FidelityProfile::default(),
        }
    }
}

/// The interaction depth a scene needs, sized to the scene rather than fixed
/// (§4.2).
///
/// A primary ray crosses each optical face once, so its path is at most the
/// scene's face count long; following ghosts multiplies that by the ghost
/// generations. A floor keeps a small folded scene usable and a ceiling bounds
/// the work per ray. A fixed number is what this replaces: the default of 24
/// stopped every ray of a 21-element lithography lens halfway through it.
pub fn depth_for_scene3(scene: &Scene3, ghosts: bool, max_ghost_bounces: u32) -> u32 {
    let faces: usize = scene
        .bodies
        .iter()
        .map(|b| match &b.geometry {
            SolidGeometry3::Profile(p) => p.profile.edges.len(),
            // A light pipe is walked by reflection, not crossed: a ray at
            // 80 degrees to its axis meets the bore about six times per bore
            // diameter of length, so that is what it is budgeted.
            SolidGeometry3::Tube {
                inner_radius_mm,
                half_length_mm,
                ..
            } => 4 + (6.0 * half_length_mm / inner_radius_mm).ceil().min(1024.0) as usize,
            _ => 6,
        })
        .sum::<usize>()
        + scene.surfaces.len()
        + scene.apertures.len()
        + scene.detectors.len();
    let generations = if ghosts { max_ghost_bounces } else { 0 };
    u32::try_from(faces)
        .unwrap_or(u32::MAX)
        .saturating_mul(generations.saturating_add(1))
        .saturating_add(16)
        .clamp(32, 1024)
}

/// How far a ray may fly between events, sized to the scene rather than fixed.
///
/// A ray that meets nothing within this is an escape, so it has to reach
/// across the whole scene: the 10 m default is a bench, and a heliostat field
/// is tens of metres with its target at the far end. Four scene diagonals, and
/// never less than the default, so a ray leaving the scene at its far corner
/// still draws as leaving it.
pub fn bounds_for_scene3(scene: &Scene3) -> f64 {
    use oc_core::geom3::Intersect3;
    let mut bb = oc_core::vec3::Aabb3::EMPTY;
    for b in &scene.bodies {
        let w = b.geometry.aabb_world(&b.common.transform);
        if w.min.is_finite() && w.max.is_finite() {
            bb.expand(w.min);
            bb.expand(w.max);
        }
    }
    let placed = scene
        .surfaces
        .iter()
        .map(|s| s.common.transform.pos)
        .chain(scene.apertures.iter().map(|a| a.common.transform.pos))
        .chain(scene.detectors.iter().map(|d| d.common.transform.pos))
        .chain(scene.sources.iter().map(|s| s.transform.pos));
    for p in placed {
        if p.is_finite() {
            bb.expand(p);
        }
    }
    let diagonal = if bb.min.is_finite() && bb.max.is_finite() {
        (bb.max - bb.min).length()
    } else {
        0.0
    };
    (4.0 * diagonal).max(TraceConfig3::default().bounds_mm)
}

/// The launch rays a single source emits, as the tracer's [`LaunchRay3`].
///
/// The sampler lives in `oc-core` ([`Source3::emit`](oc_core::source3)); this is
/// the thin wrapper that tags each ray with its source and hands it to transport.
pub fn launch_from_source(source: &oc_core::scene3::Source3) -> Vec<LaunchRay3> {
    launch_from_source_pass(source, 0)
}

/// Pass `pass` of a source's launch rays ([`Source3::emit_pass`](oc_core::scene3::Source3::emit_pass)).
pub fn launch_from_source_pass(source: &oc_core::scene3::Source3, pass: u32) -> Vec<LaunchRay3> {
    source
        .emit_pass(pass)
        .into_iter()
        .map(|e| LaunchRay3 {
            ray: e.ray,
            lambda_um: e.lambda_um,
            flux: e.flux,
            source: Some(e.source),
            pol: None,
        })
        .collect()
}

/// The state a launch ray starts in.
///
/// A source that declares no polarization emits unpolarized light, which looks
/// the same in every frame; the frame is still chosen deterministically, so two
/// runs of one scene agree on the numbers the state would carry once something
/// polarizes it. A declared state is rescaled to the ray's flux, so `flux` stays
/// the single source of truth for how much light there is.
fn launch_state(lr: &LaunchRay3) -> PolRay {
    match lr.pol {
        Some(p) if p.stokes.s0 > 0.0 => PolRay {
            stokes: p.stokes.scaled(lr.flux / p.stokes.s0),
            x: p.x,
        },
        _ => PolRay::unpolarized(lr.ray.dir, lr.flux),
    }
}

/// Every launch ray a scene's sources emit, deterministically (§6.10).
pub fn launch_from_scene(scene: &Scene3) -> Vec<LaunchRay3> {
    launch_from_scene_pass(scene, 0)
}

/// Pass `pass` of every launch ray a scene's sources emit.
pub fn launch_from_scene_pass(scene: &Scene3, pass: u32) -> Vec<LaunchRay3> {
    scene
        .sources
        .iter()
        .flat_map(|s| launch_from_source_pass(s, pass))
        .collect()
}

/// Pass `pass` of `passes`: every ray carrying its share of the passes' flux.
fn launch_pass(scene: &Scene3, pass: u32, passes: u32) -> Vec<LaunchRay3> {
    let mut rays = launch_from_scene_pass(scene, pass);
    if passes > 1 {
        let k = 1.0 / passes as f64;
        for r in &mut rays {
            r.flux *= k;
        }
    }
    rays
}

/// Trace a scene end to end, emitting launch rays from its sources.
///
/// The production entry point: the caller supplies a scene, and the sampler and
/// transport run together. [`trace3`] stays available for tests that drive the
/// engine with hand-built rays.
pub fn trace_scene3(scene: &Scene3, cfg: &TraceConfig3) -> TraceResult3 {
    let launch = launch_from_scene(scene);
    trace3(scene, &launch, cfg)
}

/// Trace a set of explicit launch rays through the scene.
///
/// Sources are turned into launch rays by [`launch_from_scene`]; this entry point
/// takes them explicitly so the transport engine can be exercised on hand-built
/// geometry.
pub fn trace3(scene: &Scene3, launch: &[LaunchRay3], cfg: &TraceConfig3) -> TraceResult3 {
    let prep = Prepared::new(scene, launch, cfg);
    let (objs, bvh, tol) = (&prep.objs, &prep.bvh, &prep.tol);
    let identity = || prep.empty();

    // A few chunks per thread, each traced into its own readout. Left to
    // itself, rayon splits a long launch list into hundreds of pieces, and
    // every piece holds a copy of every detector: with a 2600 x 330 pixel
    // detector that is tens of gigabytes.
    #[cfg(feature = "parallel")]
    let merged = {
        use rayon::prelude::*;
        let chunk = launch.len().div_ceil(4 * rayon::current_num_threads()).max(1);
        launch
            .par_chunks(chunk)
            .map(|rays| {
                let mut acc = identity();
                for lr in rays {
                    trace_one(scene, objs, bvh, tol, cfg, *lr, &mut acc);
                }
                acc
            })
            .reduce(identity, |mut a, b| {
                a.result.merge(b.result);
                a
            })
    };
    #[cfg(not(feature = "parallel"))]
    let merged = {
        let mut acc = identity();
        for lr in launch {
            trace_one(scene, objs, bvh, tol, cfg, *lr, &mut acc);
        }
        acc
    };
    merged.finish(cfg)
}

/// What a trace of a scene builds before its first ray: the objects a ray can
/// meet, their hierarchy, the scene's tolerances, and the empty readouts.
struct Prepared {
    objs: EventObjects,
    bvh: Bvh3,
    tol: ToleranceContext,
    blank: Vec<DetectorResult3>,
    report: CapabilityReport,
    /// The flux below which a segment is not recorded
    /// ([`TraceConfig3::segment_floor`] of the brightest launch ray).
    segment_floor: f64,
}

impl Prepared {
    fn new(scene: &Scene3, launch: &[LaunchRay3], cfg: &TraceConfig3) -> Self {
        let objs = EventObjects::build(scene);
        let bvh = Bvh3::build(&objs.boxes);
        let tol = ToleranceContext::from_aabb(&objs.scene_aabb);

        // A scene with a PlanarOnly body cannot be traced authoritatively: it
        // has no 3D extent, so a full-3D trace of it is flagged rather than
        // silently missing the object (§3.2, §7.7).
        let mut report = CapabilityReport::default();
        for b in &scene.bodies {
            if !b.is_3d() {
                report.reject(CapabilityIssue::PlanarOnlyObject {
                    object: b.common.id,
                });
            }
        }
        // Each detector's empty readout, built once: the exact rim areas in it
        // are worth computing once per trace, not once per worker.
        let blank = scene.detectors.iter().map(DetectorResult3::new).collect();
        // No segment is brighter than the ray it continues, so the brightest
        // launch ray sets the scale before a ray is traced.
        let brightest = launch.iter().map(|l| l.flux).fold(0.0, f64::max);
        Self {
            objs,
            bvh,
            tol,
            blank,
            report,
            segment_floor: cfg.segment_floor * brightest,
        }
    }

    fn empty(&self) -> Local {
        let mut result = LocalResult::new(self.blank.clone(), self.report.clone());
        result.segment_floor = self.segment_floor;
        Local { result }
    }
}

impl Local {
    fn finish(self, cfg: &TraceConfig3) -> TraceResult3 {
        let mut r = self.result;
        let fidelity = oc_core::fidelity::FidelityStamp::new(cfg.effective_profile());
        r.report.fidelity = Some(fidelity.clone());
        TraceResult3 {
            fidelity,
            revision: cfg.revision,
            segments: r.segments,
            segments_omitted: r.segments_omitted,
            detectors: r.detectors,
            ledger: r.ledger,
            terminations: r.terminations,
            capability_report: r.report,
            events: r.events,
            objects: r
                .objects
                .into_iter()
                .map(|((object, in_band), ledger)| ObjectEntry3 { object, in_band, ledger })
                .collect(),
            deposits: r.deposits,
        }
    }
}

/// How many blocks of launch rays a reference trace is cut into (fewer when
/// there are fewer rays). Each block is summed on its own, and the blocks are
/// added up in block order, so a trace comes out the same, to the bit, whether
/// one worker traces every block ([`TraceJob3`]) or a pool of workers shares
/// them out ([`TraceBlocks3`]). A worker holds one block's segments at a
/// time, so more blocks is less memory per worker: a block of a ghost-lit
/// twenty-element lens is about 150 thousand segments.
pub const TRACE_BLOCKS: usize = 64;

fn blocks_for(rays: usize) -> usize {
    rays.clamp(1, TRACE_BLOCKS)
}

/// The launch rays of block `i` of a trace of `rays` rays. In u64, since a
/// trace of many passes can hold more rays than a 32-bit `usize` can
/// multiply by the block count.
fn block_rays(rays: usize, i: usize) -> Range<usize> {
    let b = blocks_for(rays) as u64;
    let (rays, i) = (rays as u64, i as u64);
    (i * rays / b) as usize..((i + 1) * rays / b) as usize
}

/// A reference trace prepared once and traced a block at a time, for a pool
/// of workers to share. Each worker prepares its own copy of the scene, traces
/// the blocks it is handed into [`TracePart3`]s, and one [`TraceMerge3`] adds
/// the parts up in block order.
pub struct TraceBlocks3 {
    scene: Scene3,
    cfg: TraceConfig3,
    /// How many times the scene's rays are traced, each time shifted
    /// ([`Source3::emit_pass`](oc_core::scene3::Source3::emit_pass)).
    passes: u32,
    /// The launch rays in one pass.
    per_pass: usize,
    /// One pass's launch rays, and which: a trace of many passes holds one
    /// at a time. Its blocks run through the passes in order, so a worker
    /// makes each pass it reaches about once.
    launch: std::cell::RefCell<(u32, Vec<LaunchRay3>)>,
    /// Launch rays traced for the picture alone, when the numbers are traced
    /// with more rays than the picture should show; empty when the launch
    /// rays draw themselves. See [`TraceBlocks3::with_picture`].
    picture: Vec<LaunchRay3>,
    prep: Prepared,
}

impl TraceBlocks3 {
    pub fn new(scene: Scene3, cfg: TraceConfig3) -> Self {
        Self::with_picture(scene, None, cfg)
    }

    /// A trace whose numbers come from `scene` and whose picture comes from
    /// `picture`, the same scene sampled with fewer rays.
    ///
    /// Tracing more rays sharpens every number and would thicken the picture
    /// by as much, which past a point shows nothing more and costs memory in
    /// every worker. So the numbers' rays draw nothing, and each block also
    /// traces its share of `picture`'s rays for their segments alone: their
    /// hits, flux and ledger are thrown away. The picture is then the one
    /// `picture` traced by itself would draw, and the numbers are `scene`'s,
    /// to the bit.
    pub fn with_picture(scene: Scene3, picture: Option<&Scene3>, cfg: TraceConfig3) -> Self {
        Self::with_passes(scene, 1, picture, cfg)
    }

    /// A trace of `passes` passes of `scene`'s rays, each carrying its share
    /// of the flux, so every total is what one pass gives and the noise falls
    /// as one over the root of the passes. The numbers take as many rays as
    /// anyone has time for, while a worker holds one pass of them.
    ///
    /// The [`TRACE_BLOCKS`] blocks run across the passes in order: pass 0's
    /// rays, then pass 1's, cut into blocks as one long list. A block's sum
    /// does not depend on which worker traces it, so a pool gives what one
    /// job gives, to the bit, whatever the passes.
    pub fn with_passes(scene: Scene3, passes: u32, picture: Option<&Scene3>, cfg: TraceConfig3) -> Self {
        let passes = passes.max(1);
        let first = launch_pass(&scene, 0, passes);
        let picture = picture.map(launch_from_scene).unwrap_or_default();
        let drawn = if picture.is_empty() { &first } else { &picture };
        let prep = Prepared::new(&scene, drawn, &cfg);
        Self {
            scene,
            cfg,
            passes,
            per_pass: first.len(),
            launch: std::cell::RefCell::new((0, first)),
            picture,
            prep,
        }
    }

    /// The number of blocks.
    pub fn blocks(&self) -> usize {
        blocks_for(self.rays())
    }

    /// The number of launch rays, in all blocks and every pass.
    pub fn rays(&self) -> usize {
        self.per_pass.saturating_mul(self.passes as usize)
    }

    /// The number of passes.
    pub fn passes(&self) -> u32 {
        self.passes
    }

    /// Begin block `block`. A part begun without `record_hits` keeps no
    /// per-hit detector records. A merge keeps only the first
    /// [`TraceConfig3::max_records`] of each detector, in block order, so once
    /// one block has filled every detector's by itself, no later block's
    /// records can reach the result, and a pool skips them.
    pub fn part(&self, block: usize, record_hits: bool) -> TracePart3 {
        let rays = block_rays(self.rays(), block);
        let mut acc = self.prep.empty();
        acc.result.record_hits = record_hits;
        // The picture's rays split into as many blocks as the numbers', so a
        // block holds its share of both.
        let blocks = self.blocks();
        let p = self.picture.len();
        let pic_rays = block * p / blocks..(block + 1) * p / blocks;
        let pic = (!self.picture.is_empty()).then(|| {
            acc.result.draw = false;
            let mut pic = self.prep.empty();
            pic.result.record_hits = false;
            pic
        });
        TracePart3 {
            block,
            start: rays.start,
            next: rays.start,
            end: rays.end,
            acc,
            pic,
            pic_start: pic_rays.start,
            pic_next: pic_rays.start,
            pic_end: pic_rays.end,
        }
    }

    /// Trace up to `rays` more launch rays of `part` (at least one): its
    /// picture's first, then its numbers'. True once its block is traced.
    pub fn step(&self, part: &mut TracePart3, rays: usize) -> bool {
        let mut left = rays.max(1);
        let Prepared { objs, bvh, tol, .. } = &self.prep;
        if let Some(pic) = part.pic.as_mut() {
            let end = (part.pic_next + left).min(part.pic_end);
            for lr in &self.picture[part.pic_next..end] {
                trace_one(&self.scene, objs, bvh, tol, &self.cfg, *lr, pic);
            }
            left -= end - part.pic_next;
            part.pic_next = end;
            if part.pic_next >= part.pic_end {
                // The picture's share is in: its segments are the block's,
                // and nothing else it traced is kept.
                let pic = part.pic.take().expect("the picture is in hand");
                part.acc.result.segments = pic.result.segments;
                part.acc.result.segments_omitted += pic.result.segments_omitted;
            }
            if left == 0 {
                return part.done();
            }
        }
        let end = (part.next + left).min(part.end);
        while part.next < end {
            // The pass `part.next` is in, and its rays up to `end`.
            let pass = part.next / self.per_pass;
            let base = pass * self.per_pass;
            let stop = end.min(base + self.per_pass);
            let pass = pass as u32;
            if self.launch.borrow().0 != pass {
                *self.launch.borrow_mut() = (pass, launch_pass(&self.scene, pass, self.passes));
            }
            let launch = self.launch.borrow();
            let rays = &launch.1;
            // A pass whose sampler dropped a degenerate ray is that much shorter.
            let (lo, hi) = ((part.next - base).min(rays.len()), (stop - base).min(rays.len()));
            for lr in &rays[lo..hi] {
                trace_one(&self.scene, objs, bvh, tol, &self.cfg, *lr, &mut part.acc);
            }
            part.next = stop;
        }
        part.done()
    }
}

/// One block's share of a reference trace ([`TraceBlocks3::part`]).
pub struct TracePart3 {
    block: usize,
    start: usize,
    next: usize,
    end: usize,
    acc: Local,
    /// The picture's rays of this block while they are traced, when the
    /// picture has rays of its own ([`TraceBlocks3::with_picture`]).
    pic: Option<Local>,
    pic_start: usize,
    pic_next: usize,
    pic_end: usize,
}

/// A [`TracePart3`]'s numbers, to post from one worker to another: every sum
/// and count as an f64 (a count is exact up to 2^53) and the capability report
/// as JSON, which `serde_json` round-trips to the bit. Its segments stay with
/// the worker that traced them, which draws them into the picture; so does the
/// event log, which a split trace does not keep.
pub struct EncodedPart3 {
    pub numbers: Vec<f64>,
    pub report: String,
}

/// Bumped whenever [`EncodedPart3::numbers`] changes layout.
const PART_LAYOUT: f64 = 4.0;
/// One detector hit record, as f64s.
const HIT_FIELDS: usize = 24;

impl TracePart3 {
    pub fn block(&self) -> usize {
        self.block
    }

    pub fn done(&self) -> bool {
        self.next >= self.end && self.pic.is_none()
    }

    /// Launch rays traced so far, and in the block, the picture's included.
    pub fn progress(&self) -> (usize, usize) {
        (
            self.next - self.start + self.pic_next - self.pic_start,
            self.end - self.start + self.pic_end - self.pic_start,
        )
    }

    /// The block's segments, in launch order: its share of the picture.
    pub fn segments(&self) -> &[Segment3] {
        &self.acc.result.segments
    }

    /// Whether this block alone recorded `max_records` hits on every detector
    /// (see [`TraceBlocks3::part`]).
    pub fn records_full(&self, max_records: usize) -> bool {
        self.acc
            .result
            .detectors
            .iter()
            .all(|d| d.records.len() >= max_records)
    }

    /// The part's numbers, for a [`TraceMerge3`] in another worker.
    pub fn encode(&self) -> EncodedPart3 {
        let r = &self.acc.result;
        let mut n = vec![
            PART_LAYOUT,
            self.block as f64,
            r.segments_omitted as f64,
            r.detectors.len() as f64,
        ];
        let l = &r.ledger;
        n.extend([
            l.launched,
            l.detected_incident,
            l.detected_signal,
            l.escaped,
            l.absorbed_volume,
            l.absorbed_surface,
            l.blocked_aperture,
            l.truncated_flux,
            l.numerical_loss,
        ]);
        let t = &r.terminations;
        n.extend(
            [
                t.detector,
                t.absorbed_volume,
                t.absorbed_surface,
                t.aperture_blocked,
                t.escaped_bounds,
                t.flux_threshold,
                t.max_depth,
                t.max_ghost_depth,
                t.max_branches,
                t.numerical_failure,
                t.unsupported_physics,
            ]
            .map(|c| c as f64),
        );
        for d in &r.detectors {
            n.extend([
                d.hits as f64,
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
            ]);
            for v in [&d.incident_flux, &d.detected_signal, &d.direction_flux] {
                n.push(v.len() as f64);
                n.extend_from_slice(v);
            }
            n.push(d.colour_xyz.len() as f64);
            n.extend(d.colour_xyz.iter().flatten());
            n.push(d.spectral_untracked_flux);
            n.push(d.spectral.len() as f64);
            for line in &d.spectral {
                n.extend(line.to_array());
            }
            n.push(d.records.len() as f64);
            for h in &d.records {
                let parent = h.parent_path_id.map_or(-1.0, |p| p.0 as f64);
                n.extend([
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
                    h.path_id.0 as f64,
                    parent,
                    h.depth as f64,
                    h.ghost_depth as f64,
                    h.stokes.s0,
                    h.stokes.s1,
                    h.stokes.s2,
                    h.stokes.s3,
                    h.frame_x.x,
                    h.frame_x.y,
                    h.frame_x.z,
                    0.0,
                ]);
            }
        }
        EncodedPart3 {
            numbers: n,
            report: serde_json::to_string(&r.report).expect("a capability report serializes"),
        }
    }
}

/// The parts of a reference trace added up in block order.
pub struct TraceMerge3 {
    cfg: TraceConfig3,
    blocks: usize,
    next: usize,
    /// Each detector's empty readout, for decoding a part into.
    blank: Vec<DetectorResult3>,
    total: Local,
}

impl TraceMerge3 {
    pub fn new(of: &TraceBlocks3) -> Self {
        Self {
            cfg: of.cfg,
            blocks: of.blocks(),
            next: 0,
            blank: of.prep.blank.clone(),
            total: of.prep.empty(),
        }
    }

    /// The block the merge takes next.
    pub fn next_block(&self) -> usize {
        self.next
    }

    pub fn done(&self) -> bool {
        self.next >= self.blocks
    }

    /// Add the next block's part. A part out of order is refused: the sum
    /// depends on the order it is added in.
    pub fn add(&mut self, part: TracePart3) -> Result<(), String> {
        self.check(part.block)?;
        self.take(part.acc.result);
        Ok(())
    }

    /// Add the next block's part, as another worker encoded it. The merged
    /// trace then holds no segments of that block.
    pub fn add_encoded(&mut self, part: &EncodedPart3) -> Result<(), String> {
        let r = self.decode(part)?;
        self.take(r);
        Ok(())
    }

    /// The trace of the blocks added so far.
    pub fn finish(self) -> TraceResult3 {
        self.total.finish(&self.cfg)
    }

    fn check(&self, block: usize) -> Result<(), String> {
        if block != self.next || self.done() {
            return Err(format!(
                "block {block} arrived; the merge takes block {} next",
                self.next
            ));
        }
        Ok(())
    }

    fn take(&mut self, r: LocalResult) {
        let total = &mut self.total.result;
        total.merge(r);
        for d in &mut total.detectors {
            d.records.truncate(self.cfg.max_records);
        }
        self.next += 1;
    }

    fn decode(&self, part: &EncodedPart3) -> Result<LocalResult, String> {
        let bad = |what: &str| format!("a malformed trace part: {what}");
        let mut it = part.numbers.iter().copied();
        let mut next = || it.next().ok_or_else(|| bad("too short"));
        if next()? != PART_LAYOUT {
            return Err(bad("another layout"));
        }
        let block = next()? as usize;
        self.check(block)?;
        let report: CapabilityReport =
            serde_json::from_str(&part.report).map_err(|e| bad(&e.to_string()))?;
        let mut r = LocalResult::new(self.blank.clone(), report);
        r.segments_omitted = next()? as u64;
        if next()? as usize != r.detectors.len() {
            return Err(bad("another detector count"));
        }
        let l = &mut r.ledger;
        for f in [
            &mut l.launched,
            &mut l.detected_incident,
            &mut l.detected_signal,
            &mut l.escaped,
            &mut l.absorbed_volume,
            &mut l.absorbed_surface,
            &mut l.blocked_aperture,
            &mut l.truncated_flux,
            &mut l.numerical_loss,
        ] {
            *f = next()?;
        }
        let t = &mut r.terminations;
        for c in [
            &mut t.detector,
            &mut t.absorbed_volume,
            &mut t.absorbed_surface,
            &mut t.aperture_blocked,
            &mut t.escaped_bounds,
            &mut t.flux_threshold,
            &mut t.max_depth,
            &mut t.max_ghost_depth,
            &mut t.max_branches,
            &mut t.numerical_failure,
            &mut t.unsupported_physics,
        ] {
            *c = next()? as u64;
        }
        for d in &mut r.detectors {
            d.hits = next()? as u64;
            d.total_incident = next()?;
            d.total_signal = next()?;
            d.flux_uv = Vec2::new(next()?, next()?);
            d.flux_opl = next()?;
            d.stokes_local = Stokes {
                s0: next()?,
                s1: next()?,
                s2: next()?,
                s3: next()?,
            };
            d.max_tilt_rad = next()?;
            d.outside_cone_flux = next()?;
            for v in [
                &mut d.incident_flux,
                &mut d.detected_signal,
                &mut d.direction_flux,
            ] {
                if next()? as usize != v.len() {
                    return Err(bad("another detector shape"));
                }
                for x in v.iter_mut() {
                    *x = next()?;
                }
            }
            if next()? as usize != d.colour_xyz.len() {
                return Err(bad("another detector shape"));
            }
            for c in d.colour_xyz.iter_mut() {
                *c = [next()?, next()?, next()?];
            }
            d.spectral_untracked_flux = next()?;
            let lines = next()? as usize;
            if lines > crate::result3::MAX_SPECTRAL_LINES {
                return Err(bad("too many spectral lines"));
            }
            let mut x = [0.0; crate::result3::SPECTRAL_FIELDS];
            for _ in 0..lines {
                for v in x.iter_mut() {
                    *v = next()?;
                }
                d.spectral.push(crate::result3::SpectralSums3::from_array(&x));
            }
            let records = next()? as usize;
            let mut h = [0.0; HIT_FIELDS];
            for _ in 0..records {
                for x in h.iter_mut() {
                    *x = next()?;
                }
                d.records.push(DetectorHit3 {
                    point_world: Vec3::new(h[0], h[1], h[2]),
                    uv_mm: Vec2::new(h[3], h[4]),
                    incident_dir_local: Vec3::new(h[5], h[6], h[7]),
                    lambda_um: h[8],
                    incident_flux: h[9],
                    signal: h[10],
                    opl_mm: h[11],
                    path_id: PathId(h[12] as u64),
                    parent_path_id: (h[13] >= 0.0).then(|| PathId(h[13] as u64)),
                    depth: h[14] as u32,
                    ghost_depth: h[15] as u32,
                    stokes: Stokes {
                        s0: h[16],
                        s1: h[17],
                        s2: h[18],
                        s3: h[19],
                    },
                    frame_x: Vec3::new(h[20], h[21], h[22]),
                    counted: 0,
                });
            }
        }
        if next().is_ok() {
            return Err(bad("too long"));
        }
        Ok(r)
    }
}

/// A reference trace done a batch of launch rays at a time.
///
/// For a caller that has to stay responsive while it traces: a browser worker
/// returns to its event loop between batches, sees that the scene has changed,
/// and drops the job rather than finishing a trace nobody will read. The job
/// owns its copy of the scene, so nothing can edit what it is tracing.
///
/// It traces the blocks of [`TRACE_BLOCKS`] in order and adds each up as it
/// completes, so it gives what a pool sharing the blocks out gives, to the
/// bit, and what [`trace3`] gives up to the order a parallel sum adds in
/// (`tests/trace_job3.rs`).
pub struct TraceJob3 {
    blocks: TraceBlocks3,
    part: TracePart3,
    merge: TraceMerge3,
    /// Launch rays in the blocks already merged.
    merged_rays: usize,
}

impl TraceJob3 {
    /// A job tracing every ray `scene`'s sources emit.
    pub fn new(scene: Scene3, cfg: TraceConfig3) -> Self {
        Self::with_picture(scene, None, cfg)
    }

    /// A job whose numbers come from `scene` and whose picture comes from
    /// `picture` ([`TraceBlocks3::with_picture`]).
    pub fn with_picture(scene: Scene3, picture: Option<&Scene3>, cfg: TraceConfig3) -> Self {
        Self::with_passes(scene, 1, picture, cfg)
    }

    /// A job of `passes` passes of `scene`'s rays ([`TraceBlocks3::with_passes`]).
    pub fn with_passes(scene: Scene3, passes: u32, picture: Option<&Scene3>, cfg: TraceConfig3) -> Self {
        let blocks = TraceBlocks3::with_passes(scene, passes, picture, cfg);
        let part = blocks.part(0, true);
        let merge = TraceMerge3::new(&blocks);
        let mut job = Self {
            blocks,
            part,
            merge,
            merged_rays: 0,
        };
        job.advance();
        job
    }

    /// Trace the next `rays` launch rays (at least one). True once every ray
    /// is traced.
    pub fn step(&mut self, rays: usize) -> bool {
        let mut left = rays.max(1);
        while left > 0 && !self.done() {
            let before = self.part.progress().0;
            self.blocks.step(&mut self.part, left);
            left = left.saturating_sub(self.part.progress().0 - before);
            self.advance();
        }
        self.done()
    }

    /// Merge the part in hand while it is complete, and begin the next block.
    fn advance(&mut self) {
        while self.part.done() && !self.merge.done() {
            let next = self.merge.next_block() + 1;
            let fresh = self.blocks.part(next.min(self.blocks.blocks() - 1), true);
            let part = std::mem::replace(&mut self.part, fresh);
            self.merged_rays += part.end - part.start;
            self.merge
                .add(part)
                .expect("the job adds its blocks in order");
        }
    }

    pub fn done(&self) -> bool {
        self.merge.done()
    }

    /// Launch rays traced so far, and in all.
    pub fn progress(&self) -> (usize, usize) {
        let current = if self.done() {
            0
        } else {
            self.part.next - self.part.start
        };
        (self.merged_rays + current, self.blocks.rays())
    }

    /// The segments recorded so far: a picture of the rays traced so far.
    pub fn segments_so_far(&self) -> usize {
        let current = if self.done() {
            0
        } else {
            self.part.segments().len()
        };
        self.merge.total.result.segments.len() + current
    }

    /// The trace: complete once [`TraceJob3::done`], and otherwise the part
    /// of it the rays traced so far make up.
    pub fn finish(self) -> TraceResult3 {
        let mut merge = self.merge;
        if !merge.done() && self.part.next > self.part.start {
            merge
                .add(self.part)
                .expect("the part in hand is the next block");
        }
        merge.finish()
    }
}

// ---------------------------------------------------------------------------
// Per-thread accumulation
// ---------------------------------------------------------------------------

struct Local {
    result: LocalResult,
}

struct LocalResult {
    segments: Vec<Segment3>,
    /// The flux below which a segment is not recorded, and how many were not.
    segment_floor: f64,
    segments_omitted: u64,
    /// Keep per-hit records, when the trace asks for them. Off for a block
    /// whose records no merge would keep ([`TraceBlocks3::part`]).
    record_hits: bool,
    /// Record segments. Off for rays traced for their numbers alone, when
    /// the picture has rays of its own ([`TraceBlocks3::with_picture`]).
    draw: bool,
    detectors: Vec<DetectorResult3>,
    ledger: EnergyLedger,
    terminations: TerminationSummary,
    report: CapabilityReport,
    events: Vec<TraceEvent3>,
    objects: std::collections::BTreeMap<(ObjectRef3, bool), ObjectLedger3>,
    deposits: Vec<Deposit3>,
    /// What the coating just met absorbed, layer by layer, for the deposits
    /// of `after_interaction` (`layer_absorption: PerLayer`).
    pending_layers: Vec<(DepositLayer, f64)>,
    next_path: u64,
}

impl LocalResult {
    fn new(detectors: Vec<DetectorResult3>, report: CapabilityReport) -> Self {
        Self {
            segments: Vec::new(),
            segment_floor: 0.0,
            segments_omitted: 0,
            record_hits: true,
            draw: true,
            detectors,
            ledger: EnergyLedger::default(),
            terminations: TerminationSummary::default(),
            report,
            events: Vec::new(),
            objects: std::collections::BTreeMap::new(),
            deposits: Vec::new(),
            pending_layers: Vec::new(),
            next_path: 0,
        }
    }

    fn merge(&mut self, o: LocalResult) {
        self.segments.extend(o.segments);
        self.segments_omitted += o.segments_omitted;
        for (a, b) in self.detectors.iter_mut().zip(&o.detectors) {
            a.merge(b);
        }
        self.ledger.merge(&o.ledger);
        self.terminations.merge(&o.terminations);
        self.report.merge(&o.report);
        self.events.extend(o.events);
        for (k, v) in o.objects {
            self.objects.entry(k).or_default().add(&v);
        }
        self.deposits.extend(o.deposits);
    }

    fn new_path(&mut self) -> PathId {
        let id = PathId(self.next_path);
        self.next_path += 1;
        id
    }
}

struct RayWork {
    ray: Ray3,
    lambda: f64,
    /// Flux **and** polarization. `pol.stokes.s0` is the flux: one number, not
    /// two that could drift apart.
    pol: PolRay,
    depth: u32,
    ghost_depth: u32,
    opl: f64,
    path_id: PathId,
    parent: Option<PathId>,
    from_tir: bool,
    ghost: bool,
    /// The object that sent this branch on its way, for the object ledger.
    from: ObjectRef3,
    /// Hits on `TraceConfig3::count_hits_on` since launch.
    counted: u32,
}

impl RayWork {
    fn flux(&self) -> f64 {
        self.pol.stokes.s0
    }

    /// The state after the medium has taken its cut. Bulk absorption in an
    /// isotropic medium is polarization-blind, so this is a pure scaling.
    fn state_at(&self, flux: f64) -> Stokes {
        let s0 = self.pol.stokes.s0;
        if s0 > 0.0 {
            self.pol.stokes.scaled(flux / s0)
        } else {
            Stokes::unpolarized(flux)
        }
    }
}

// ---------------------------------------------------------------------------
// Event objects and their world boxes
// ---------------------------------------------------------------------------

/// Everything a ray can meet, flattened for the BVH. Indices point back into the
/// scene's vectors (which are presentation order only; identity is by id).
struct EventObjects {
    boxes: Vec<Aabb3>,
    kinds: Vec<ObjKind>,
    scene_aabb: Aabb3,
    /// Each scene body's world box, by body index (empty for a body with no
    /// 3D extent), so a medium lookup skips a body without asking it.
    body_boxes: Vec<Aabb3>,
    /// What each body's intersection reuses from step to step, by body index.
    body_preps: Vec<SolidPrep>,
}

enum ObjKind {
    Body(usize),
    Surface(usize),
    Aperture(usize),
    Detector(usize),
}

impl EventObjects {
    fn build(scene: &Scene3) -> Self {
        let mut boxes = Vec::new();
        let mut kinds = Vec::new();
        let mut body_boxes = Vec::with_capacity(scene.bodies.len());
        for (i, b) in scene.bodies.iter().enumerate() {
            if !b.is_3d() {
                body_boxes.push(Aabb3::EMPTY);
                continue; // no 3D extent; flagged in the report, contributes no events
            }
            let bb = b.geometry.aabb_world(&b.common.transform);
            boxes.push(bb);
            body_boxes.push(bb);
            kinds.push(ObjKind::Body(i));
        }
        for (i, s) in scene.surfaces.iter().enumerate() {
            boxes.push(s.geometry.aabb_world(&s.common.transform));
            kinds.push(ObjKind::Surface(i));
        }
        for (i, a) in scene.apertures.iter().enumerate() {
            boxes.push(aperture_aabb(a));
            kinds.push(ObjKind::Aperture(i));
        }
        for (i, d) in scene.detectors.iter().enumerate() {
            if let Some(bb) = detector_aabb(d) {
                boxes.push(bb);
                kinds.push(ObjKind::Detector(i));
            }
        }
        let mut scene_aabb = Aabb3::EMPTY;
        for b in &boxes {
            scene_aabb = scene_aabb.union(b);
        }
        if scene_aabb.is_empty() {
            scene_aabb = Aabb3 {
                min: Vec3::splat(-1.0),
                max: Vec3::splat(1.0),
            };
        }
        Self {
            boxes,
            kinds,
            scene_aabb,
            body_boxes,
            body_preps: scene.bodies.iter().map(|b| b.geometry.prepare()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// The event a step resolves to
// ---------------------------------------------------------------------------

struct Event {
    t: f64,
    point: Vec3,
    kind: EventKind,
}

enum EventKind {
    /// A closed body's surface. `outward` is the stable outward normal (§5.4).
    Body {
        body: usize,
        outward: Vec3,
        patch: oc_core::id::PatchId,
    },
    /// A zero-thickness surface. `normal` is the local `+Z` front normal in world.
    Surface { surf: usize, normal: Vec3 },
    /// An opaque aperture mask (the ray is outside the clear opening).
    ApertureBlocked { ap: usize },
    Detector {
        det: usize,
        uv: Vec2,
        dir_local: Vec3,
    },
}

// ---------------------------------------------------------------------------
// Trace one launch ray and its whole branch tree
// ---------------------------------------------------------------------------

fn trace_one(
    scene: &Scene3,
    objs: &EventObjects,
    bvh: &Bvh3,
    tol: &ToleranceContext,
    cfg: &TraceConfig3,
    lr: LaunchRay3,
    acc: &mut Local,
) {
    let r = &mut acc.result;
    r.ledger.launched += lr.flux;
    let min_flux = cfg.flux_eps * lr.flux;
    let skip = tol.medium_abs_mm.max(1e-12);
    // The medium is sampled a few boundary-bands off a surface, not one: at
    // exactly `medium_abs_mm` the point lands in the containment `Boundary` band,
    // which is neither inside nor outside, so the body's index would not be read
    // and a refractive interface would see n1 == n2 and never bend.
    let medium_eps = tol.medium_abs_mm * 8.0;

    let root = r.new_path();
    let mut stack: Vec<RayWork> = Vec::with_capacity(32);
    stack.push(RayWork {
        ray: lr.ray,
        lambda: lr.lambda_um,
        // A source with no declared polarization emits unpolarized light, whose
        // state is the same in every frame; the frame still has to be chosen
        // deterministically so two runs agree on the numbers it would carry if
        // something polarized it later.
        pol: launch_state(&lr),
        depth: 0,
        ghost_depth: 0,
        opl: 0.0,
        path_id: root,
        parent: None,
        from_tir: false,
        ghost: false,
        from: ObjectRef3::Source,
        counted: 0,
    });

    let mut branches = 0u64;
    while let Some(w) = stack.pop() {
        branches += 1;
        if branches > cfg.max_branches {
            // Deposit the popped branch and everything still queued, so the cap
            // subtracts a named quantity rather than a silent one.
            r.ledger.truncated_flux += w.flux() + stack.iter().map(|x| x.flux()).sum::<f64>();
            r.terminations.record(TerminationReason::MaxBranches);
            if cfg.object_ledger {
                book(r, cfg, w.from, w.lambda).truncated += w.flux();
                for x in &stack {
                    book(r, cfg, x.from, x.lambda).truncated += x.flux();
                }
            }
            break;
        }
        let culled = if w.depth > cfg.max_depth {
            Some(TerminationReason::MaxDepth)
        } else if w.ghost && w.ghost_depth > cfg.max_ghost_bounces {
            Some(TerminationReason::MaxGhostDepth)
        } else if w.flux() < min_flux {
            Some(TerminationReason::FluxThreshold)
        } else {
            None
        };
        if let Some(reason) = culled {
            deposit_truncated(r, &w, reason);
            if cfg.object_ledger {
                book(r, cfg, w.from, w.lambda).truncated += w.flux();
            }
            continue;
        }

        // The medium this branch travels through: containment + priority, sampled
        // just ahead of the origin so a ray leaving a surface reads the medium it
        // is entering.
        let here = region_at(scene, objs, w.ray.origin + medium_eps * w.ray.dir, tol);
        let n_here = region_index(scene, here, w.lambda);
        let alpha = match region_absorption(scene, here, w.lambda) {
            Ok(a) => a,
            Err(issue) => {
                r.report.reject(issue);
                0.0
            }
        };

        // Nearest event, and any probe detectors passed on the way to it.
        let Some(ev) = nearest_event(scene, objs, bvh, cfg, &w, skip) else {
            // Nothing ahead: the branch leaves the scene.
            let end = w.ray.at(cfg.bounds_mm);
            let flux_end = w.flux() * beer_lambert(alpha, cfg.bounds_mm);
            record_absorption(r, w.flux() - flux_end);
            push_segment(r, &w, end, flux_end, w.opl + n_here * cfg.bounds_mm);
            push_event(
                r,
                cfg,
                &w,
                end,
                w.opl + n_here * cfg.bounds_mm,
                TraceEventKind::Escaped,
            );
            r.ledger.escaped += flux_end;
            r.terminations.record(TerminationReason::EscapedBounds);
            if cfg.object_ledger {
                let b = book(r, cfg, w.from, w.lambda);
                b.absorbed_volume += (w.flux() - flux_end).max(0.0);
                b.escaped += flux_end;
            }
            continue;
        };

        let l = ev.t;
        let flux_at = w.flux() * beer_lambert(alpha, l);
        record_absorption(r, w.flux() - flux_at);
        if cfg.object_ledger && w.flux() > flux_at {
            book(r, cfg, w.from, w.lambda).absorbed_volume += w.flux() - flux_at;
        }
        if w.flux() > flux_at {
            deposit_segment(r, cfg, &w, alpha, l);
        }
        let opl_at = w.opl + n_here * l;
        push_segment(r, &w, ev.point, flux_at, opl_at);

        match ev.kind {
            EventKind::Detector { det, uv, dir_local } => {
                let terminates = matches!(
                    scene.detectors[det].termination,
                    DetectorTermination::Terminate
                );
                push_event(
                    r,
                    cfg,
                    &w,
                    ev.point,
                    opl_at,
                    TraceEventKind::DetectorHit {
                        detector: scene.detectors[det].id(),
                        uv,
                        terminal: terminates,
                    },
                );
                record_detector(
                    r, scene, det, &w, ev.point, uv, dir_local, flux_at, opl_at, cfg,
                );
                if cfg.object_ledger {
                    let b = book(r, cfg, ObjectRef3::Detector(det), w.lambda);
                    b.hits += 1;
                    b.incident += flux_at;
                    if terminates {
                        b.detected += flux_at;
                    }
                }
                if terminates {
                    r.ledger.detected_incident += flux_at;
                    r.ledger.detected_signal += flux_at * detector_qe(scene, det, w.lambda);
                    r.terminations.record(TerminationReason::Detector);
                } else {
                    // Probe: the flux continues, unchanged, just past the face.
                    continue_probe(&mut stack, &w, ev.point, opl_at, flux_at, skip);
                }
            }
            EventKind::ApertureBlocked { ap } => {
                r.ledger.blocked_aperture += flux_at;
                if cfg.object_ledger {
                    let b = book(r, cfg, ObjectRef3::Aperture(ap), w.lambda);
                    b.hits += 1;
                    b.incident += flux_at;
                    b.blocked += flux_at;
                }
                if cfg.deposit_apertures {
                    r.deposits.push(Deposit3 { object: ObjectRef3::Aperture(ap), point: ev.point, lambda_um: w.lambda, flux: flux_at, layer: DepositLayer::Face });
                }
                r.report.note_exact(PhysicsFeature::Aperture);
                push_event(
                    r,
                    cfg,
                    &w,
                    ev.point,
                    opl_at,
                    TraceEventKind::ApertureBlocked {
                        aperture: scene.apertures[ap].id(),
                    },
                );
                r.terminations.record(TerminationReason::ApertureBlocked);
            }
            EventKind::Surface { surf, normal } => {
                let (before, n0) = (r.ledger, stack.len());
                interact_surface(
                    scene, &mut stack, r, cfg, &w, surf, ev.point, normal, n_here, flux_at, opl_at,
                    min_flux,
                );
                after_interaction(r, cfg, &mut stack[n0..], ObjectRef3::Surface(surf), &w, flux_at, &before, ev.point);
            }
            EventKind::Body {
                body,
                outward,
                patch,
            } => {
                let (before, n0) = (r.ledger, stack.len());
                interact_body(
                    scene, objs, &mut stack, r, cfg, tol, &w, body, ev.point, outward, patch, here,
                    n_here, flux_at, opl_at, medium_eps, min_flux,
                );
                after_interaction(r, cfg, &mut stack[n0..], ObjectRef3::Body(body, patch.0), &w, flux_at, &before, ev.point);
            }
        }
    }
}

/// The object ledger's row for `obj` in `lambda`'s band.
fn book<'a>(r: &'a mut LocalResult, cfg: &TraceConfig3, obj: ObjectRef3, lambda: f64) -> &'a mut ObjectLedger3 {
    let [lo, hi] = cfg.ledger_band_um;
    r.objects.entry((obj, lambda >= lo && lambda < hi)).or_default()
}

/// After an interaction at `obj`: its children leave from `obj` and count it
/// when it is counted, and whatever the interaction took out of the ledger
/// (absorption, culled children, numerical losses) is booked against it.
#[allow(clippy::too_many_arguments)] // the result, the config, the ray before and after, and where it hit
fn after_interaction(
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    children: &mut [RayWork],
    obj: ObjectRef3,
    w: &RayWork,
    flux_at: f64,
    before: &EnergyLedger,
    point: Vec3,
) {
    let counts = matches!(obj, ObjectRef3::Body(i, _) if (cfg.count_bodies[0]..cfg.count_bodies[1]).contains(&i));
    // Absorbed at the face of a recorded body or surface (a coating, a black
    // face): one deposit for the face, or, under `layer_absorption:
    // PerLayer`, one per coating layer and one for the substrate, which the
    // coating left in `pending_layers`.
    let wanted = match obj {
        ObjectRef3::Body(i, _) => (cfg.deposit_bodies[0]..cfg.deposit_bodies[1]).contains(&i),
        ObjectRef3::Surface(_) => cfg.deposit_surfaces,
        _ => false,
    };
    let recorded = cfg.fidelity.layer_absorption != oc_core::fidelity::LayerAbsorption::Off;
    let pending = std::mem::take(&mut r.pending_layers);
    if wanted && recorded {
        if pending.is_empty() {
            let took = r.ledger.absorbed_surface - before.absorbed_surface;
            if took > 0.0 {
                r.deposits.push(Deposit3 { object: obj, point, lambda_um: w.lambda, flux: took, layer: DepositLayer::Face });
            }
        } else {
            for (layer, flux) in pending {
                if flux > 0.0 {
                    r.deposits.push(Deposit3 { object: obj, point, lambda_um: w.lambda, flux, layer });
                }
            }
        }
    }
    for c in children.iter_mut() {
        c.from = obj;
        c.counted = w.counted + u32::from(counts);
    }
    if !cfg.object_ledger {
        return;
    }
    let l = r.ledger;
    let b = book(r, cfg, obj, w.lambda);
    b.hits += 1;
    b.incident += flux_at;
    b.absorbed_surface += l.absorbed_surface - before.absorbed_surface;
    b.absorbed_volume += l.absorbed_volume - before.absorbed_volume;
    b.blocked += l.blocked_aperture - before.blocked_aperture;
    b.detected += l.detected_incident - before.detected_incident;
    b.escaped += l.escaped - before.escaped;
    b.truncated += l.truncated_flux - before.truncated_flux;
    b.numerical += l.numerical_loss - before.numerical_loss;
}

/// The bulk absorption of the segment `w` travelled before its event, `l` mm
/// at `alpha` per mm, recorded in four steps when it ran inside a recorded
/// body (the body it entered through).
fn deposit_segment(r: &mut LocalResult, cfg: &TraceConfig3, w: &RayWork, alpha: f64, l: f64) {
    let ObjectRef3::Body(i, _) = w.from else { return };
    if !(cfg.deposit_bodies[0]..cfg.deposit_bodies[1]).contains(&i) {
        return;
    }
    const STEPS: usize = 4;
    for k in 0..STEPS {
        let (t0, t1) = (l * k as f64 / STEPS as f64, l * (k + 1) as f64 / STEPS as f64);
        let took = w.flux() * (beer_lambert(alpha, t0) - beer_lambert(alpha, t1));
        if took > 0.0 {
            r.deposits.push(Deposit3 { object: w.from, point: w.ray.at(0.5 * (t0 + t1)), lambda_um: w.lambda, flux: took, layer: DepositLayer::Bulk });
        }
    }
}

/// Deposit a culled branch's flux into `truncated_flux` and record the reason.
fn deposit_truncated(r: &mut LocalResult, w: &RayWork, reason: TerminationReason) {
    r.ledger.truncated_flux += w.flux();
    r.terminations.record(reason);
}

fn record_absorption(r: &mut LocalResult, absorbed: f64) {
    if absorbed > 0.0 {
        r.ledger.absorbed_volume += absorbed;
    }
}

/// Append an event-log entry, when the config asks for one.
fn push_event(
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    w: &RayWork,
    point: Vec3,
    opl: f64,
    kind: TraceEventKind,
) {
    if cfg.record_events {
        r.events.push(TraceEvent3 {
            path_id: w.path_id,
            parent: w.parent,
            depth: w.depth,
            ghost: w.ghost,
            point,
            opl_mm: opl,
            lambda_um: w.lambda,
            kind,
        });
    }
}

fn push_segment(r: &mut LocalResult, w: &RayWork, b: Vec3, flux_end: f64, opl1: f64) {
    if !r.draw {
        return;
    }
    if flux_end < r.segment_floor {
        r.segments_omitted += 1;
        return;
    }
    r.segments.push(Segment3 {
        a: w.ray.origin,
        b,
        flux_start: w.flux(),
        flux_end,
        lambda_um: w.lambda,
        opl0_mm: w.opl,
        opl1_mm: opl1,
        path_id: w.path_id,
        parent_path_id: w.parent,
        flags: SegmentFlags {
            ghost: w.ghost,
            tir: w.from_tir,
        },
    });
}

// ---------------------------------------------------------------------------
// Nearest-event search
// ---------------------------------------------------------------------------

fn nearest_event(
    scene: &Scene3,
    objs: &EventObjects,
    bvh: &Bvh3,
    cfg: &TraceConfig3,
    w: &RayWork,
    skip: f64,
) -> Option<Event> {
    let mut best: Option<Event> = None;
    let mut best_t = cfg.bounds_mm;

    // The nearest event wins. Two at exactly the same distance, as the two
    // copies of a face that two bodies share can be, are settled by the lower
    // object id: a fixed policy, so which one the walk reached first does not
    // decide which body's normal the ray is refracted by (§9.3).
    let consider = |ev: Event, best: &mut Option<Event>, best_t: &mut f64| {
        let wins = ev.t < *best_t
            || (ev.t == *best_t
                && best
                    .as_ref()
                    .is_some_and(|b| event_object(scene, &ev.kind) < event_object(scene, &b.kind)));
        if wins {
            *best_t = ev.t;
            *best = Some(ev);
        }
    };

    let mut visit = |oi: usize, _: f64| -> f64 {
        match &objs.kinds[oi] {
            ObjKind::Body(bi) => {
                let b = &scene.bodies[*bi];
                b.geometry.intersect_prepared(
                    &objs.body_preps[*bi],
                    &w.ray,
                    &b.common.transform,
                    skip,
                    best_t,
                    &mut |h| {
                        let ev = Event {
                            t: h.t_mm,
                            point: b.common.transform.transform_point(h.point_local),
                            kind: EventKind::Body {
                                body: *bi,
                                outward: b.common.transform.rotate(h.normal_local),
                                patch: h.patch,
                            },
                        };
                        consider(ev, &mut best, &mut best_t);
                    },
                );
            }
            ObjKind::Surface(si) => {
                let s = &scene.surfaces[*si];
                s.geometry
                    .intersect(&w.ray, &s.common.transform, skip, best_t, &mut |h| {
                        let ev = Event {
                            t: h.t_mm,
                            point: s.common.transform.transform_point(h.point_local),
                            kind: EventKind::Surface {
                                surf: *si,
                                normal: s.common.transform.rotate(h.normal_local),
                            },
                        };
                        consider(ev, &mut best, &mut best_t);
                    });
            }
            ObjKind::Aperture(ai) => {
                if let Some((t, point)) =
                    aperture_block(&scene.apertures[*ai], &w.ray, skip, best_t)
                {
                    consider(
                        Event {
                            t,
                            point,
                            kind: EventKind::ApertureBlocked { ap: *ai },
                        },
                        &mut best,
                        &mut best_t,
                    );
                }
            }
            ObjKind::Detector(di) => {
                if let Some((t, point, uv, dir_local)) =
                    detector_hit(&scene.detectors[*di], &w.ray, skip, best_t)
                {
                    consider(
                        Event {
                            t,
                            point,
                            kind: EventKind::Detector {
                                det: *di,
                                uv,
                                dir_local,
                            },
                        },
                        &mut best,
                        &mut best_t,
                    );
                }
            }
        }
        best_t
    };
    if cfg.brute_force_search {
        // The differential-test oracle: every object whose box the ray
        // crosses at all, asked in turn.
        let mut all: Vec<usize> = Vec::new();
        bvh.candidates(&w.ray, skip, cfg.bounds_mm, &mut |oi| all.push(oi));
        for oi in all {
            visit(oi, 0.0);
        }
    } else {
        bvh.nearest(&w.ray, skip, cfg.bounds_mm, &mut visit);
    }

    best
}

/// The id of the object an event is on, the key that settles a tie.
fn event_object(scene: &Scene3, kind: &EventKind) -> u64 {
    match kind {
        EventKind::Body { body, .. } => scene.bodies[*body].common.id.0,
        EventKind::Surface { surf, .. } => scene.surfaces[*surf].common.id.0,
        EventKind::ApertureBlocked { ap } => scene.apertures[*ap].common.id.0,
        EventKind::Detector { det, .. } => scene.detectors[*det].common.id.0,
    }
}

// ---------------------------------------------------------------------------
// Medium resolution (§9.4)
// ---------------------------------------------------------------------------

/// The body whose material occupies `point`: highest priority, ties broken by
/// highest id (a stable key, never vector order, §9.2). `None` is the ambient.
///
/// A body whose world box does not hold the point is passed over without the
/// containment test: the box encloses the body (the same guarantee the ray
/// search rests on), so the test could only have said outside. The answer is
/// the one asking every body gives, found without solving every body's curves.
fn region_at(
    scene: &Scene3,
    objs: &EventObjects,
    point: Vec3,
    tol: &ToleranceContext,
) -> Option<usize> {
    let mut best: Option<(usize, i32, u64)> = None;
    for (i, b) in scene.bodies.iter().enumerate() {
        if !b.is_3d() || !objs.body_boxes[i].contains_point(point) {
            continue;
        }
        let inside =
            b.geometry
                .contains_prepared(&objs.body_preps[i], point, &b.common.transform, tol);
        if inside == Containment::Inside {
            let key = (b.priority, b.common.id.0);
            if best.map_or(true, |(_, p, id)| key > (p, id)) {
                best = Some((i, b.priority, b.common.id.0));
            }
        }
    }
    best.map(|(i, _, _)| i)
}

fn region_index(scene: &Scene3, region: Option<usize>, lambda: f64) -> f64 {
    match region {
        None => scene.env.ambient_index,
        Some(i) => scene.bodies[i]
            .material
            .index_at(lambda, &scene.catalog)
            .unwrap_or(scene.env.ambient_index),
    }
}

fn region_absorption(
    scene: &Scene3,
    region: Option<usize>,
    lambda: f64,
) -> Result<f64, CapabilityIssue> {
    let (model, object) = match region {
        None => (&scene.env.ambient_absorption, None),
        Some(i) => (&scene.bodies[i].absorption, Some(scene.bodies[i].common.id)),
    };
    model
        .sample(lambda)
        .ok_or_else(|| CapabilityIssue::WavelengthOutOfRange {
            object: object.unwrap_or(ObjectId(0)),
            lambda_um: lambda,
        })
}

// ---------------------------------------------------------------------------
// Surface interaction (§9.6)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn interact_body(
    scene: &Scene3,
    objs: &EventObjects,
    stack: &mut Vec<RayWork>,
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    tol: &ToleranceContext,
    w: &RayWork,
    body: usize,
    point: Vec3,
    outward: Vec3,
    patch: oc_core::id::PatchId,
    here: Option<usize>,
    n_here: f64,
    flux_at: f64,
    opl_at: f64,
    medium_eps: f64,
    min_flux: f64,
) {
    let b = &scene.bodies[body];
    r.report.note_exact(match b.geometry {
        SolidGeometry3::Profile(_) => PhysicsFeature::ProfileBody,
        _ => PhysicsFeature::AnalyticSolid,
    });

    // Side selection uses the stable outward normal, not the incident-oriented
    // copy (§5.4): the ray is entering when it runs against `outward`.
    let entering = w.ray.dir.dot(outward) < 0.0;
    let two_sided = b.surface.for_patch(patch);
    let response = if entering {
        &two_sided.from_outside
    } else {
        &two_sided.from_inside
    };

    // The medium on the far side of the boundary (§9.4).
    let n_oppose = if entering { outward } else { -outward };
    let far = region_at(scene, objs, point - medium_eps * n_oppose, tol);
    let n_far = region_index(scene, far, w.lambda);
    let _ = here;
    // The medium is found off the drawn face; the light sees the wavy one.
    let n_oppose = lean_normal(n_oppose, w.ray.dir, point, scene.slope_error_rad(b.common.id));

    apply_response(
        stack,
        r,
        cfg,
        w,
        response,
        SurfaceRef::new(b.common.id, patch),
        point,
        n_oppose,
        n_here,
        n_far,
        true,
        flux_at,
        opl_at,
        min_flux,
    );
}

/// The normal a hit sees on a face with slope error `sigma` (radians RMS per
/// axis, [`oc_core::scene3::FigureError`]): the drawn normal `n`, which
/// opposes `dir`, leaned by a Gaussian draw in the two directions across it.
///
/// The draw is hashed from the hit's own point and direction, so a trace
/// repeats exactly and two rays never share one. A lean that would turn the
/// face away from the ray, or send the mirror image into the face, is drawn
/// again: at grazing incidence a real wavy face hides those slopes behind its
/// own crests. After eight tries the drawn normal is kept.
fn lean_normal(n: Vec3, dir: Vec3, point: Vec3, sigma: f64) -> Vec3 {
    if sigma <= 0.0 {
        return n;
    }
    fn mix(z: u64) -> u64 {
        // splitmix64's finaliser.
        let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let mut h = [point.x, point.y, point.z, dir.x, dir.y, dir.z]
        .iter()
        .fold(0u64, |acc, v| mix(acc ^ v.to_bits()));
    let mut uniform = || {
        h = mix(h);
        // (0, 1]: never 0, which the logarithm below cannot take.
        ((h >> 11) as f64 + 1.0) * (1.0 / (1u64 << 53) as f64)
    };
    let t1 = n.any_orthonormal_vector();
    let t2 = n.cross(t1);
    for _ in 0..8 {
        // Box-Muller: two independent unit Gaussians.
        let (u1, u2) = (uniform(), uniform());
        let radius = sigma * (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        let m = (n + t1 * (radius * c) + t2 * (radius * s)).normalize();
        if dir.dot(m) < 0.0 && reflect3(dir, m).dot(n) > 0.0 {
            return m;
        }
    }
    n
}

#[allow(clippy::too_many_arguments)]
fn interact_surface(
    scene: &Scene3,
    stack: &mut Vec<RayWork>,
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    w: &RayWork,
    surf: usize,
    point: Vec3,
    normal: Vec3,
    n_here: f64,
    flux_at: f64,
    opl_at: f64,
    min_flux: f64,
) {
    let s: &SurfaceObject3 = &scene.surfaces[surf];
    r.report.note_exact(PhysicsFeature::ZeroThicknessSurface);
    // Front is the local `+Z` side; the ray is incident on the front when it runs
    // against that normal.
    let from_front = w.ray.dir.dot(normal) < 0.0;
    let response = if from_front {
        &s.response_front
    } else {
        &s.response_back
    };
    let n_oppose = lean_normal(
        if from_front { normal } else { -normal },
        w.ray.dir,
        point,
        scene.slope_error_rad(s.common.id),
    );
    let surface_ref = SurfaceRef::new(s.common.id, oc_core::id::PatchId(0));
    let lens_interact = |children: Vec<(ChildRole, PathId)>| TraceEventKind::Interact {
        surface: surface_ref,
        body: false,
        normal: n_oppose,
        n1: n_here,
        n2: n_here,
        children,
    };
    // The ideal thin lens is the one response whose outgoing direction depends
    // on WHERE the surface was hit, so it needs the surface's local frame and
    // is handled here rather than in the frame-blind `apply_response`.
    if let SurfaceResponse::ThinLens { focal_length_mm } = response {
        let f = *focal_length_mm;
        if !f.is_finite() || f == 0.0 {
            r.report
                .reject(CapabilityIssue::UnsupportedSurfaceResponse {
                    surface: surface_ref, reason: None
                });
            r.ledger.numerical_loss += flux_at;
            r.terminations.record(TerminationReason::UnsupportedPhysics);
            push_event(r, cfg, w, point, opl_at, lens_interact(Vec::new()));
            return;
        }
        r.report.note_exact(PhysicsFeature::ThinLens);
        let t = &s.common.transform;
        let p_local = t.inverse_transform_point(point);
        let d_local = t.quat().inverse() * w.ray.dir;
        // Slope law u' = u − h/f per transverse axis, direction-aware: with the
        // axial component preserved, d'_t = d_t − |d_z|·h_t/f sends a forward
        // parallel bundle exactly through (0, 0, +f) and a backward one through
        // (0, 0, −f), and images every axial conjugate exactly (1/s_o + 1/s_i =
        // 1/f). A ray grazing in the lens plane (|d_z| ≈ 0) passes undeviated.
        let dir = if d_local.z.abs() < 1e-12 {
            w.ray.dir
        } else {
            let bent = Vec3::new(
                d_local.x - d_local.z.abs() * p_local.x / f,
                d_local.y - d_local.z.abs() * p_local.y / f,
                d_local.z,
            )
            .normalize();
            t.quat() * bent
        };
        // Full transmission: no reflection branch, no absorption, no added OPL,
        // and no polarization action either. The bend is not a reflection, so
        // there is no plane of incidence to work in; the frame is carried across
        // it by the shortest rotation, which is what parallel transport along a
        // bent ray means.
        let pol = PolRay {
            stokes: w.state_at(flux_at),
            x: polarization::transport_frame(w.pol.x, w.ray.dir, dir),
        };
        let child = spawn(
            stack, r, cfg, w, point, dir, pol, opl_at, false, false, min_flux,
        );
        let children = child
            .map(|id| (ChildRole::ThinLens, id))
            .into_iter()
            .collect();
        push_event(r, cfg, w, point, opl_at, lens_interact(children));
        return;
    }
    // A grating sends light along its grating vector, which is a direction in
    // the element like a lens's axes: handled here, in the surface's frame.
    if let SurfaceResponse::Grating {
        lines_per_mm,
        order,
        efficiency,
        zero_order,
        transmissive,
        tm,
    } = response
    {
        // A table that promises more than all of the light gives all of it,
        // in proportion.
        let pair = |e: &oc_core::scene3::SpectralScalar, z: &oc_core::scene3::SpectralScalar| {
            let (a, b) = (e.sample(w.lambda)?.clamp(0.0, 1.0), z.sample(w.lambda)?.clamp(0.0, 1.0));
            let over = (a + b).max(1.0);
            Some((a / over, b / over))
        };
        let te = pair(efficiency, zero_order);
        let tm_pair = match tm {
            Some(t) => pair(&t.efficiency, &t.zero_order),
            None => te,
        };
        let (Some((te_m, te_0)), Some((tm_m, tm_0))) = (te, tm_pair) else {
            r.report
                .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface: surface_ref, reason: None });
            r.ledger.numerical_loss += flux_at;
            r.terminations.record(TerminationReason::UnsupportedPhysics);
            push_event(r, cfg, w, point, opl_at, lens_interact(Vec::new()));
            return;
        };
        r.report.note_exact(PhysicsFeature::Grating);
        r.report.note_approx(
            PhysicsFeature::Grating,
            if tm.is_some() {
                "declared order efficiencies per polarization, with no retardance, and the other orders absorbed"
            } else {
                "declared order efficiencies: polarization-blind, and the other orders absorbed"
            },
        );
        let t = &s.common.transform;
        let d_local = t.quat().inverse() * w.ray.dir;
        let k = oc_core::grating::k_per_um(*lines_per_mm, *order);
        let stokes = w.state_at(flux_at);
        // TE's axis on a ray: the grooves, seen across it.
        let grooves = t.rotate(Vec3::Y);
        let te_axis = |d: Vec3| {
            let p = grooves - d * d.dot(grooves);
            (p.length() > 1e-9).then(|| p.normalize())
        };
        let branch = |eta_te: f64, eta_tm: f64, dir: Vec3| -> PolRay {
            match (tm.is_some(), te_axis(w.ray.dir), te_axis(dir)) {
                // The grating splits the state on its grooves, as a
                // diattenuator would: amplitude √η along them and across.
                (true, Some(a_in), Some(a_out)) => {
                    let m = Mueller::from_diagonal_jones(
                        oc_core::thinfilm::Cx::real(eta_te.sqrt()),
                        oc_core::thinfilm::Cx::real(eta_tm.sqrt()),
                    );
                    let s_in = stokes.rotated(polarization::frame_angle(w.ray.dir, w.pol.x, a_in));
                    let out = if cfg.polarized() {
                        m.apply(&s_in)
                    } else {
                        Stokes::unpolarized(m.unpolarized_gain() * flux_at)
                    };
                    PolRay { stokes: out, x: a_out }
                }
                // Scaled as it is: an ideal grating that treats both
                // polarizations alike, which the declared numbers assume.
                _ => PolRay {
                    stokes: stokes.scaled(0.5 * (eta_te + eta_tm)),
                    x: polarization::transport_frame(w.pol.x, w.ray.dir, dir),
                },
            }
        };
        let (eta_m, eta_0) = (te_m.max(tm_m), te_0.max(tm_0));
        let mut children = Vec::new();
        let mut kept = 0.0;
        if eta_m > 0.0 {
            if let Some(dl) = oc_core::grating::diffract_local(d_local, k, w.lambda, *transmissive) {
                let dir = (t.quat() * dl).normalize();
                let pol = branch(te_m, tm_m, dir);
                kept += pol.stokes.s0;
                if let Some(id) = spawn(stack, r, cfg, w, point, dir, pol, opl_at, false, false, min_flux) {
                    children.push((ChildRole::Diffract, id));
                }
            }
        }
        if eta_0 > 0.0 {
            let (dir, role) = if *transmissive {
                (w.ray.dir, ChildRole::Transmit)
            } else {
                (reflect3(w.ray.dir, n_oppose), ChildRole::Reflect)
            };
            let pol = branch(te_0, tm_0, dir);
            kept += pol.stokes.s0;
            if let Some(id) = spawn(stack, r, cfg, w, point, dir, pol, opl_at, false, false, min_flux) {
                children.push((role, id));
            }
        }
        // The other orders, and an evanescent working order, are absorbed.
        r.ledger.absorbed_surface += (flux_at - kept).max(0.0);
        if children.is_empty() {
            r.terminations.record(TerminationReason::AbsorbedSurface);
        }
        push_event(r, cfg, w, point, opl_at, lens_interact(children));
        return;
    }
    // A polarizer and a retarder act on the state in **their own** frame, not
    // in the plane of incidence: a waveplate's fast axis is fixed in the
    // element and does not turn with the ray that happens to arrive. So like
    // the thin lens they need the surface's local frame, and are handled here.
    if let SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. } = response {
        let interact_ev = |children: Vec<(ChildRole, PathId)>| TraceEventKind::Interact {
            surface: surface_ref,
            body: false,
            normal: n_oppose,
            n1: n_here,
            n2: n_here,
            children,
        };
        let dir = w.ray.dir;
        let (axis_deg, n_sub) = match response {
            SurfaceResponse::Polarizer {
                axis_deg,
                substrate_index,
                ..
            }
            | SurfaceResponse::Retarder {
                axis_deg,
                substrate_index,
                ..
            } => (*axis_deg, substrate_index.max(1.0)),
            _ => unreachable!("matched above"),
        };

        // The element's own faces. The polarizing layer is laminated in glass
        // of index `substrate_index`, and a ray meets that glass going in and
        // again coming out. An index of 1 is a bare layer, where a face
        // reflects nothing and transmits everything, so none of this changes an
        // answer.
        //
        // The element is still zero thickness, so the two faces sit in the same
        // place: no displacement, no optical path through the glass, and no
        // bounce between them. That bounce is bounded by the square of a face's
        // reflectance and crosses the polarizing layer twice more, so a
        // polarizer absorbs most of it and a lossless retarder does not.
        let cos_i = -n_oppose.dot(dir).clamp(-1.0, 1.0);
        let front = oc_core::thinfilm::interface_amp(n_here, n_sub, cos_i);

        // Work on the plane of incidence, which the faces are diagonal in. The
        // element's own frame and this one share `ŝ`: it is perpendicular to
        // the plane of incidence, so it is the same inside the glass as
        // outside, and only `p̂` turns with the ray and turns back on the way
        // out.
        let s_hat = polarization::incidence_axis(dir, n_oppose).unwrap_or(w.pol.x);
        let stokes_in = w
            .state_at(flux_at)
            .rotated(polarization::frame_angle(dir, w.pol.x, s_hat));
        let make = |m: &Mueller| -> Stokes {
            if cfg.polarized() {
                m.apply(&stokes_in)
            } else {
                // Without a state to act on, all an element like this can do is
                // its average: half for a polarizer facing unpolarized light,
                // all of it for a lossless retarder. The preview runs this, and
                // it is why a crossed pair looks bright there and is dark here.
                Stokes::unpolarized(m.unpolarized_gain() * flux_at)
            }
        };

        let Some(dir_in) = refract3(dir, n_oppose, n_here, n_sub) else {
            // The element sits in something denser than its own glass and the
            // ray is past the critical angle at the front face, so it never
            // reaches the polarizing layer at all.
            let stokes = make(&Mueller::from_diagonal_jones(front.r_s, front.r_p));
            r.ledger.absorbed_surface += (flux_at - stokes.s0).max(0.0);
            let child = spawn_tir(
                stack,
                r,
                cfg,
                w,
                point,
                reflect3(dir, n_oppose),
                PolRay { stokes, x: s_hat },
                opl_at,
                min_flux,
            );
            let children = child.map(|id| (ChildRole::Tir, id)).into_iter().collect();
            push_event(r, cfg, w, point, opl_at, interact_ev(children));
            return;
        };
        let cos_t = -n_oppose.dot(dir_in).clamp(-1.0, 1.0);
        let back = oc_core::thinfilm::interface_amp(n_sub, n_here, cos_t);

        // The axis is a direction in the element's own plane: its local `+X`
        // turned by `axis_deg` towards its local `+Y`, the plane its normal
        // (local `+Z`) stands on. What acts on the ray is that direction
        // projected into the plane the field lives in, which inside the glass
        // is set by the refracted direction. A substrate straightens the ray
        // and so shrinks the projection, which is one reason a laminated
        // polarizer holds its contrast off axis better than a bare film.
        //
        // Building the axis first and projecting second is not the same as
        // projecting `+X` and turning the result inside the ray's transverse
        // plane. The two agree at normal incidence and diverge as the square of
        // the obliquity, and the difference is the whole of an ideal element's
        // angle dependence: turning inside the transverse plane preserves the
        // angle between any two elements exactly, so a crossed pair would
        // extinguish perfectly seen from every direction. Real crossed
        // polarizers leak off axis, which is why a display has a viewing angle.
        let (sin_a, cos_a) = axis_deg.to_radians().sin_cos();
        let t = &s.common.transform;
        let axis = t.rotate(Vec3::X) * cos_a + t.rotate(Vec3::Y) * sin_a;
        let projected = axis - dir_in * dir_in.dot(axis);
        if projected.length() < 1e-9 {
            // The ray runs along the element's own axis: there is no projection
            // and no meaningful transmission axis to apply.
            r.report
                .reject(CapabilityIssue::UnsupportedSurfaceResponse {
                    surface: surface_ref, reason: None
                });
            r.ledger.numerical_loss += flux_at;
            r.terminations.record(TerminationReason::UnsupportedPhysics);
            push_event(r, cfg, w, point, opl_at, interact_ev(Vec::new()));
            return;
        }
        let frame_x = projected.normalize();

        let (layer, feature) = match response {
            SurfaceResponse::Polarizer { extinction, .. } => {
                // A datasheet's contrast curve is read at this ray's own angle
                // of incidence. One number is the ideal layer, which holds its
                // contrast wherever it is used, and no real polarizer does.
                let Some(leak) = extinction.sample(cos_i.acos().to_degrees()) else {
                    // The curve does not reach this angle and its policy says
                    // to refuse, so there is no contrast to apply.
                    r.report
                        .reject(CapabilityIssue::UnsupportedSurfaceResponse {
                            surface: surface_ref, reason: None
                        });
                    r.ledger.numerical_loss += flux_at;
                    r.terminations.record(TerminationReason::UnsupportedPhysics);
                    push_event(r, cfg, w, point, opl_at, interact_ev(Vec::new()));
                    return;
                };
                (
                    // Amplitude on the blocked axis is the square root of the
                    // power that leaks: extinction is quoted as a power ratio.
                    Mueller::from_diagonal_jones(
                        oc_core::thinfilm::Cx::real(1.0),
                        oc_core::thinfilm::Cx::real(leak.clamp(0.0, 1.0).sqrt()),
                    ),
                    PhysicsFeature::Polarizer,
                )
            }
            SurfaceResponse::Retarder {
                retardance_deg,
                crystal,
                ..
            } => {
                let nominal = retardance_deg.to_radians();
                let rad = match crystal {
                    // An ideal retarder: the number given, wherever the ray
                    // came from and whatever colour it is.
                    None => nominal,
                    Some(c) => {
                        // A slab of crystal instead. The transverse index
                        // `n sin θ` is conserved across every parallel face, so
                        // it is the ray's own here and the element's glass, if
                        // it has any, does not come into it.
                        let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
                        let thickness = c.thickness_um(*retardance_deg);
                        // The optic axis is the slow one in a positive crystal
                        // and the fast one in a negative crystal, and the
                        // element always names its fast axis.
                        let optic = if c.n_e >= c.n_o {
                            axis_deg + 90.0
                        } else {
                            axis_deg
                        };
                        let (so, co) = optic.to_radians().sin_cos();
                        let c_hat = t.rotate(Vec3::X) * co + t.rotate(Vec3::Y) * so;
                        // Which way the ray leans, in the plate's own plane.
                        let lean = dir - n_oppose * dir.dot(n_oppose);
                        let cos_axis = if lean.length() > 1e-12 {
                            lean.normalize().dot(c_hat)
                        } else {
                            0.0
                        };
                        polarization::a_plate_retardance(
                            c.n_o,
                            c.n_e,
                            thickness,
                            w.lambda,
                            n_here * sin_i,
                            cos_axis,
                        )
                        .abs()
                    }
                };
                (Mueller::linear_retarder(rad), PhysicsFeature::Retarder)
            }
            _ => unreachable!("matched above"),
        };
        r.report.note_exact(feature);
        if n_sub > 1.0 {
            r.report.note_approx(
                feature,
                "element faces added singly: no bounce between them",
            );
        }
        if matches!(
            response,
            SurfaceResponse::Retarder {
                crystal: Some(_),
                ..
            }
        ) {
            // A real crystal sends the ordinary and the extraordinary wave off
            // at slightly different angles inside the plate, so they come out
            // side by side rather than on top of each other. Micrometres for a
            // thin plate, and not modelled: one ray goes in and one comes out.
            r.report
                .note_approx(feature, "one ray through the crystal: no walk-off");
        }

        // Front face, then the layer on its own axis, then the back face. The
        // order is not decoration: the two faces commute with each other and
        // neither commutes with the layer.
        let faces = |a: &oc_core::thinfilm::FilmAmp| {
            Mueller::from_diagonal_jones(a.t_s, a.t_p).scaled(a.t_power_scale)
        };
        let turned = layer.rotated(polarization::frame_angle(dir_in, s_hat, frame_x));
        let m_t = faces(&back).times(&turned.times(&faces(&front)));
        let stokes_t = make(&m_t);
        let stokes_r = make(&Mueller::from_diagonal_jones(front.r_s, front.r_p));
        r.ledger.absorbed_surface += (flux_at - stokes_t.s0 - stokes_r.s0).max(0.0);

        let mut children = Vec::new();
        if let Some(id) = spawn(
            stack,
            r,
            cfg,
            w,
            point,
            dir,
            PolRay {
                stokes: stokes_t,
                x: s_hat,
            },
            opl_at,
            false,
            false,
            min_flux,
        ) {
            children.push((ChildRole::Transmit, id));
        }
        if stokes_r.s0 > 0.0 {
            if cfg.ghosts {
                if let Some(id) = spawn(
                    stack,
                    r,
                    cfg,
                    w,
                    point,
                    reflect3(dir, n_oppose),
                    PolRay {
                        stokes: stokes_r,
                        x: s_hat,
                    },
                    opl_at,
                    true,
                    false,
                    min_flux,
                ) {
                    children.push((ChildRole::Reflect, id));
                }
            } else {
                // Ghosts off: the reflected flux is not followed, but it is
                // still named rather than lost.
                r.ledger.truncated_flux += stokes_r.s0;
            }
        }
        push_event(r, cfg, w, point, opl_at, interact_ev(children));
        return;
    }
    // A zero-thickness surface has the same medium on both sides (§6.6): no
    // refraction, transmission continues undeviated.
    apply_response(
        stack,
        r,
        cfg,
        w,
        response,
        surface_ref,
        point,
        n_oppose,
        n_here,
        n_here,
        false,
        flux_at,
        opl_at,
        min_flux,
    );
}

/// Apply a [`SurfaceResponse`] at a point, splitting deterministic branches and
/// depositing whatever flux does not continue.
#[allow(clippy::too_many_arguments)]
fn apply_response(
    stack: &mut Vec<RayWork>,
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    w: &RayWork,
    response: &SurfaceResponse,
    surface: SurfaceRef,
    point: Vec3,
    n_oppose: Vec3,
    n1: f64,
    n2: f64,
    medium_transition: bool,
    flux_at: f64,
    opl_at: f64,
    min_flux: f64,
) {
    let cos_i = -n_oppose.dot(w.ray.dir).clamp(-1.0, 1.0);
    let sample = |s: &oc_core::scene3::SpectralScalar| s.sample(w.lambda);

    // The plane of incidence, and the arriving state written on it. At normal
    // incidence there is no such plane, and the ray keeps its own frame: that is
    // the continuous limit, because there the two polarizations describe the
    // same physical reflection and every azimuth gives the same answer.
    let s_hat = polarization::incidence_axis(w.ray.dir, n_oppose).unwrap_or(w.pol.x);
    let stokes_in = w
        .state_at(flux_at)
        .rotated(polarization::frame_angle(w.ray.dir, w.pol.x, s_hat));
    // Everything that leaves this point leaves on the same s axis: the reflected
    // and the transmitted ray both lie in the plane of incidence, so it is
    // perpendicular to both of them.
    //
    // With polarization off, every element instead sees unpolarized light and
    // hands on unpolarized light. That is the scalar-unpolarized approximation:
    // the right flux for one interaction, and wrong from the second
    // polarization-sensitive element onward. The preview runs it, so the parity
    // tests can compare like with like.
    // A phase-neutral element acting on *unpolarized* light assumes nothing:
    // the mean is the right flux and the output is unpolarized either way. The
    // assumption only bites once the light arriving is polarized, so that is
    // when the report says so, rather than decorating every scene with a mirror
    // in it.
    let phase_assumed = cfg.polarized() && stokes_in.dop() > 1e-12;
    let out = |m: &Mueller| -> PolRay {
        let stokes = if cfg.polarized() {
            m.apply(&stokes_in)
        } else {
            Stokes::unpolarized(m.unpolarized_gain() * flux_at)
        };
        PolRay { stokes, x: s_hat }
    };
    // `medium_transition` doubles as the body marker: bodies are the interfaces
    // that transition media, zero-thickness surfaces never do (§6.6).
    let interact = |children: Vec<(ChildRole, PathId)>| TraceEventKind::Interact {
        surface,
        body: medium_transition,
        normal: n_oppose,
        n1,
        n2,
        children,
    };

    match response {
        SurfaceResponse::Absorber => {
            r.ledger.absorbed_surface += flux_at;
            r.terminations.record(TerminationReason::AbsorbedSurface);
            push_event(r, cfg, w, point, opl_at, interact(Vec::new()));
        }
        SurfaceResponse::Mirror { reflectivity } => {
            r.report.note_exact(PhysicsFeature::Mirror);
            let refl = match sample(reflectivity) {
                Some(v) => v.clamp(0.0, 1.0),
                None => {
                    r.report
                        .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface, reason: None });
                    r.ledger.numerical_loss += flux_at;
                    r.terminations.record(TerminationReason::UnsupportedPhysics);
                    push_event(r, cfg, w, point, opl_at, interact(Vec::new()));
                    return;
                }
            };
            // An ideal reflector: the same coefficient for s and p and no phase
            // between them. A real metal mirror has both, and this response is a
            // number rather than a stack, so the model choice is recorded rather
            // than assumed.
            if phase_assumed {
                r.report.note_approx(
                    PhysicsFeature::Mirror,
                    "ideal reflector: equal s and p, no retardance",
                );
            }
            let m = Mueller::attenuator(refl);
            let child_pol = out(&m);
            r.ledger.absorbed_surface += flux_at - child_pol.stokes.s0;
            let dir = reflect3(w.ray.dir, n_oppose);
            let child = spawn(
                stack, r, cfg, w, point, dir, child_pol, opl_at, false,
                false, // a mirror reflection is specular, not total internal reflection
                min_flux,
            );
            let children = child
                .map(|id| (ChildRole::Reflect, id))
                .into_iter()
                .collect();
            push_event(r, cfg, w, point, opl_at, interact(children));
        }
        SurfaceResponse::DielectricFresnel { ar_residual } => {
            r.report.note_exact(PhysicsFeature::DielectricFresnel);
            let ar = sample(ar_residual).unwrap_or(1.0).clamp(0.0, 1.0);
            // Amplitudes rather than the scalar mean: an interface is a diagonal
            // Jones element, and past the critical angle the phase it puts
            // between s and p is the whole of what it does.
            let amp = oc_core::thinfilm::interface_amp(n1, n2, cos_i);
            match refract3(w.ray.dir, n_oppose, n1, n2) {
                None => {
                    // Total internal reflection: R = 1, primary path. The
                    // retardance is real physics, and it is what a Fresnel rhomb
                    // is built out of.
                    let m = Mueller::from_diagonal_jones(amp.r_s, amp.r_p);
                    let dir = reflect3(w.ray.dir, n_oppose);
                    let child = spawn_tir(stack, r, cfg, w, point, dir, out(&m), opl_at, min_flux);
                    let children = child.map(|id| (ChildRole::Tir, id)).into_iter().collect();
                    push_event(r, cfg, w, point, opl_at, interact(children));
                }
                Some(dir_t) => {
                    let pw = amp.powers();
                    let m_r = Mueller::from_diagonal_jones(amp.r_s, amp.r_p).scaled(ar);
                    // A lossless interface has R + T = 1 in each polarization
                    // separately, so suppressing the reflection with `ar` has to
                    // hand that flux to the transmitted branch rather than lose
                    // it. The two polarizations need different factors, which is
                    // a rescaled diagonal Jones element and not a grey one.
                    let keep = |rp: f64, tp: f64| {
                        if tp > 0.0 {
                            ((1.0 - ar * rp) / tp).max(0.0).sqrt()
                        } else {
                            1.0
                        }
                    };
                    let m_t = Mueller::from_diagonal_jones(
                        amp.t_s * keep(pw.r_s, pw.t_s),
                        amp.t_p * keep(pw.r_p, pw.t_p),
                    )
                    .scaled(amp.t_power_scale);
                    let pol_r = out(&m_r);
                    let pol_t = out(&m_t);
                    let mut children = Vec::new();
                    if let Some(id) = spawn(
                        stack, r, cfg, w, point, dir_t, pol_t, opl_at, false, false, min_flux,
                    ) {
                        children.push((ChildRole::Transmit, id));
                    }
                    // Whatever neither branch carries away was absorbed here.
                    // Taking it from the states rather than from the nominal
                    // coefficients keeps the ledger balanced by construction.
                    r.ledger.absorbed_surface +=
                        (flux_at - pol_r.stokes.s0 - pol_t.stokes.s0).max(0.0);
                    if cfg.ghosts && pol_r.stokes.s0 > 0.0 {
                        let dir_r = reflect3(w.ray.dir, n_oppose);
                        if let Some(id) = spawn(
                            stack, r, cfg, w, point, dir_r, pol_r, opl_at, true, false, min_flux,
                        ) {
                            children.push((ChildRole::Reflect, id));
                        }
                    } else if pol_r.stokes.s0 > 0.0 {
                        // Ghosts off: the reflected flux is not followed, but it is
                        // still named rather than lost.
                        r.ledger.truncated_flux += pol_r.stokes.s0;
                    }
                    push_event(r, cfg, w, point, opl_at, interact(children));
                }
            }
        }
        SurfaceResponse::ExplicitRt {
            reflectance: rr,
            transmittance: tt,
        } => {
            r.report.note_exact(PhysicsFeature::ExplicitRt);
            let (refl, tr) = match (sample(rr), sample(tt)) {
                (Some(a), Some(b)) => (a.clamp(0.0, 1.0), b.clamp(0.0, 1.0)),
                _ => {
                    r.report
                        .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface, reason: None });
                    r.ledger.numerical_loss += flux_at;
                    r.terminations.record(TerminationReason::UnsupportedPhysics);
                    push_event(r, cfg, w, point, opl_at, interact(Vec::new()));
                    return;
                }
            };
            // Two numbers, and two numbers cannot describe a polarization
            // interaction: powers fix the magnitudes of a diagonal Jones element
            // and say nothing about the phase between them, nor about whether s
            // and p differ at all. This treats the surface as an ideal grey
            // attenuator, equal for both polarizations and phase-neutral, which
            // is a declared model choice and not something the data establishes.
            if phase_assumed {
                r.report.note_approx(
                    PhysicsFeature::ExplicitRt,
                    "power-only response: treated as polarization-blind and phase-neutral",
                );
            }
            let m_r = Mueller::attenuator(refl);
            let m_t = Mueller::attenuator(tr);
            let pol_r = out(&m_r);
            let pol_t = out(&m_t);
            r.ledger.absorbed_surface += (flux_at - pol_r.stokes.s0 - pol_t.stokes.s0).max(0.0);
            let mut children = Vec::new();
            // Reflected branch (specular).
            if refl > 0.0 {
                let dir_r = reflect3(w.ray.dir, n_oppose);
                if let Some(id) = spawn(
                    stack, r, cfg, w, point, dir_r, pol_r, opl_at, w.ghost, false, min_flux,
                ) {
                    children.push((ChildRole::Reflect, id));
                }
            }
            // Transmitted branch: Snell across a real medium transition, else
            // undeviated (§9.6).
            if tr > 0.0 {
                let dir_t = if medium_transition {
                    refract3(w.ray.dir, n_oppose, n1, n2).unwrap_or(w.ray.dir)
                } else {
                    w.ray.dir
                };
                if let Some(id) = spawn(
                    stack, r, cfg, w, point, dir_t, pol_t, opl_at, w.ghost, false, min_flux,
                ) {
                    children.push((ChildRole::Transmit, id));
                }
            }
            push_event(r, cfg, w, point, opl_at, interact(children));
        }
        // A real coating: the one response that reads `cos_i`. Everything else
        // here answers the same for a ray at 45° as for one down the axis, which
        // is what made a tilted dichroic report the wrong edge.
        SurfaceResponse::Coating { coating } => {
            r.report.note_exact(PhysicsFeature::ThinFilmCoating);
            // The far side is the medium the transmitted ray would enter, which
            // is the substrate for a coating sitting straight on glass. For a
            // zero-thickness surface both sides are the same medium and the
            // coating carries its own plate.
            // Under `coating_model: Nominal` every ray sees the design point:
            // no wavelength, no angle. Phenomenological, and the stamp says so.
            let (lambda, cos_i) = if cfg.fidelity.coating_model == oc_core::fidelity::CoatingModel::Nominal {
                let (l, c, designed) = coating.nominal_point();
                if !designed {
                    r.report.note_approx(
                        PhysicsFeature::ThinFilmCoating,
                        "nominal: a stack with no design record answers at 0.55 um, normal incidence",
                    );
                }
                (l, c)
            } else {
                (w.lambda, cos_i)
            };
            let ctx = oc_core::thinfilm::FilmContext::from_profile(&cfg.fidelity);
            let rt = oc_core::thinfilm::coating_eval(coating, n1, n2, lambda, cos_i, &ctx);
            let (refl, tr) = match &rt {
                Ok((v, notes)) => {
                    if !notes.clamped.is_empty() {
                        r.report.note_approx(
                            PhysicsFeature::ThinFilmCoating,
                            "a film table was read outside its range and answered by its out-of-range policy",
                        );
                    }
                    if notes.degenerate {
                        r.report.note_approx(
                            PhysicsFeature::ThinFilmCoating,
                            "a degenerate stack reflected everything by rule",
                        );
                    }
                    (v.r().clamp(0.0, 1.0), v.t().clamp(0.0, 1.0))
                }
                Err(e) => {
                    r.report.reject(CapabilityIssue::UnsupportedSurfaceResponse {
                        surface,
                        reason: Some(e.to_string()),
                    });
                    r.ledger.numerical_loss += flux_at;
                    r.terminations.record(TerminationReason::UnsupportedPhysics);
                    push_event(r, cfg, w, point, opl_at, interact(Vec::new()));
                    return;
                }
            };
            // A coherent stack is a diagonal Jones element and has amplitudes.
            // A plate-backed one adds its two faces incoherently, which is right
            // for millimetres of glass and is not a Jones element at all: the
            // relative phase is exactly what the incoherent sum throws away. In
            // that case the powers still separate s from p, and only the phase
            // between them is declared absent.
            let amp = oc_core::thinfilm::coating_amp_eval(coating, n1, n2, lambda, cos_i, &ctx);
            let (m_r, m_t) = match &amp {
                Ok((a, _)) => (
                    Mueller::from_diagonal_jones(a.r_s, a.r_p),
                    Mueller::from_diagonal_jones(a.t_s, a.t_p).scaled(a.t_power_scale),
                ),
                Err(_) => {
                    if phase_assumed {
                        r.report.note_approx(
                            PhysicsFeature::ThinFilmCoating,
                            "plate-backed stack: faces added incoherently, no relative phase",
                        );
                    }
                    let (v, _) = rt.as_ref().expect("checked above");
                    (
                        Mueller::from_diagonal_jones(
                            oc_core::thinfilm::Cx::real(v.r_s.max(0.0).sqrt()),
                            oc_core::thinfilm::Cx::real(v.r_p.max(0.0).sqrt()),
                        ),
                        Mueller::from_diagonal_jones(
                            oc_core::thinfilm::Cx::real(v.t_s.max(0.0).sqrt()),
                            oc_core::thinfilm::Cx::real(v.t_p.max(0.0).sqrt()),
                        ),
                    )
                }
            };
            let pol_r = out(&m_r);
            let pol_t = out(&m_t);
            let absorbed = (flux_at - pol_r.stokes.s0 - pol_t.stokes.s0).max(0.0);
            if cfg.fidelity.layer_absorption == oc_core::fidelity::LayerAbsorption::PerLayer && absorbed > 0.0 {
                // Where in the coating it was absorbed: each layer, and the
                // substrate. Polarized light weights the s and p splits by
                // its intensity in each; the shares are scaled to the total
                // the ledger books, so they sum to it exactly.
                let (i_s, i_p) = if cfg.polarized() {
                    (0.5 * (stokes_in.s0 + stokes_in.s1), 0.5 * (stokes_in.s0 - stokes_in.s1))
                } else {
                    (0.5 * flux_at, 0.5 * flux_at)
                };
                match oc_core::thinfilm::coating_power_flow_eval(coating, n1, n2, lambda, cos_i, &ctx) {
                    Ok((p, _)) => {
                        let films: Vec<f64> =
                            p.a_layer_s.iter().zip(&p.a_layer_p).map(|(a, b)| a * i_s + b * i_p).collect();
                        let bulk = p.a_substrate_s * i_s + p.a_substrate_p * i_p;
                        let sum = films.iter().sum::<f64>() + bulk;
                        let k = if sum > 0.0 { absorbed / sum } else { 0.0 };
                        let slab = matches!(coating.backing, oc_core::thinfilm::CoatingBacking::Plate { .. });
                        // A plate's slab is bulk absorption; an opaque
                        // substrate's is booked at the surface, tagged bulk.
                        if slab {
                            r.ledger.absorbed_volume += bulk * k;
                            r.ledger.absorbed_surface += absorbed - bulk * k;
                        } else {
                            r.ledger.absorbed_surface += absorbed;
                        }
                        r.pending_layers = films
                            .iter()
                            .enumerate()
                            .map(|(i, a)| (DepositLayer::Film(i.min(u16::MAX as usize) as u16), a * k))
                            .chain(std::iter::once((DepositLayer::Bulk, bulk * k)))
                            .collect();
                    }
                    Err(_) => r.ledger.absorbed_surface += absorbed,
                }
            } else {
                r.ledger.absorbed_surface += absorbed;
            }
            let mut children = Vec::new();
            // Past the critical angle nothing gets through, whatever the stack:
            // the far medium carries only an evanescent wave, and the matrix
            // returns R = 1 for a lossless coating. The reflection is then the
            // whole ray and a primary path, exactly as total internal
            // reflection at bare glass is, so it is never stray.
            let tir = medium_transition && refract3(w.ray.dir, n_oppose, n1, n2).is_none();
            // Otherwise what an AR coat reflects is stray light, a ghost exactly
            // as bare glass's reflection is; what any other stack reflects is a
            // path. See `Coating::reflection_is_stray`.
            let stray = !tir && coating.reflection_is_stray();
            if tir {
                let dir_r = reflect3(w.ray.dir, n_oppose);
                if let Some(id) = spawn_tir(stack, r, cfg, w, point, dir_r, pol_r, opl_at, min_flux)
                {
                    children.push((ChildRole::Tir, id));
                }
            } else if refl > 0.0 {
                if stray && !cfg.ghosts {
                    // Not followed, but named rather than lost, as for bare glass.
                    r.ledger.truncated_flux += pol_r.stokes.s0;
                } else {
                    let dir_r = reflect3(w.ray.dir, n_oppose);
                    if let Some(id) = spawn(
                        stack,
                        r,
                        cfg,
                        w,
                        point,
                        dir_r,
                        pol_r,
                        opl_at,
                        w.ghost || stray,
                        false,
                        min_flux,
                    ) {
                        children.push((ChildRole::Reflect, id));
                    }
                }
            }
            if tr > 0.0 {
                let dir_t = if medium_transition {
                    refract3(w.ray.dir, n_oppose, n1, n2).unwrap_or(w.ray.dir)
                } else {
                    w.ray.dir
                };
                if let Some(id) = spawn(
                    stack, r, cfg, w, point, dir_t, pol_t, opl_at, w.ghost, false, min_flux,
                ) {
                    children.push((ChildRole::Transmit, id));
                }
            }
            push_event(r, cfg, w, point, opl_at, interact(children));
        }
        // A thin lens on a body patch has no meaning: the bend depends on the
        // hit position in a lens plane the patch does not define. A polarizer or
        // a retarder on one has none either: their axis is a direction in the
        // element, and a curved patch of a solid does not define one. Standalone
        // surfaces intercept all three in `interact_surface`; anything else
        // refuses rather than picking an axis.
        SurfaceResponse::ThinLens { .. }
        | SurfaceResponse::Grating { .. }
        | SurfaceResponse::Polarizer { .. }
        | SurfaceResponse::Retarder { .. } => {
            r.report
                .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface, reason: None });
            r.ledger.numerical_loss += flux_at;
            r.terminations.record(TerminationReason::UnsupportedPhysics);
            push_event(r, cfg, w, point, opl_at, interact(Vec::new()));
        }
    }
}

/// Push a child branch, or deposit its flux in `truncated_flux` if it is born
/// already below threshold — so a split conserves flux exactly. Returns the
/// child's path id when a branch was actually pushed, so the event log can name
/// it.
#[allow(clippy::too_many_arguments)]
fn spawn(
    stack: &mut Vec<RayWork>,
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    parent: &RayWork,
    point: Vec3,
    dir: Vec3,
    pol: PolRay,
    opl: f64,
    ghost: bool,
    from_tir: bool,
    min_flux: f64,
) -> Option<PathId> {
    let flux = pol.stokes.s0;
    let depth = parent.depth + 1;
    let ghost_depth = parent.ghost_depth + u32::from(ghost && !parent.ghost);
    let is_ghost = parent.ghost || ghost;
    if flux < min_flux {
        r.ledger.truncated_flux += flux;
        r.terminations.record(TerminationReason::FluxThreshold);
        return None;
    }
    if depth > cfg.max_depth {
        r.ledger.truncated_flux += flux;
        r.terminations.record(TerminationReason::MaxDepth);
        return None;
    }
    let path_id = r.new_path();
    let ray = match Ray3::new(point, dir) {
        Ok(ray) => ray,
        Err(_) => {
            r.ledger.numerical_loss += flux;
            r.terminations.record(TerminationReason::NumericalFailure);
            return None;
        }
    };
    stack.push(RayWork {
        ray,
        lambda: parent.lambda,
        pol,
        depth,
        ghost_depth,
        opl,
        path_id,
        parent: Some(parent.path_id),
        from_tir,
        ghost: is_ghost,
        // The caller names the object once the interaction is over.
        from: parent.from,
        counted: parent.counted,
    });
    Some(path_id)
}

#[allow(clippy::too_many_arguments)]
fn spawn_tir(
    stack: &mut Vec<RayWork>,
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    parent: &RayWork,
    point: Vec3,
    dir: Vec3,
    pol: PolRay,
    opl: f64,
    min_flux: f64,
) -> Option<PathId> {
    // TIR is the primary path (R = 1), so it does not increment the ghost budget.
    spawn(
        stack,
        r,
        cfg,
        parent,
        point,
        dir,
        pol,
        opl,
        parent.ghost,
        true,
        min_flux,
    )
}

/// A probe detector continues the same branch just past the face, unchanged.
fn continue_probe(
    stack: &mut Vec<RayWork>,
    w: &RayWork,
    point: Vec3,
    opl: f64,
    flux: f64,
    _skip: f64,
) {
    if let Ok(ray) = Ray3::new(point, w.ray.dir) {
        stack.push(RayWork {
            ray,
            lambda: w.lambda,
            pol: PolRay {
                stokes: w.state_at(flux),
                x: w.pol.x,
            },
            depth: w.depth,
            ghost_depth: w.ghost_depth,
            opl,
            path_id: w.path_id,
            parent: w.parent,
            from_tir: w.from_tir,
            ghost: w.ghost,
            from: w.from,
            counted: w.counted,
        });
    }
}

// ---------------------------------------------------------------------------
// Detectors and apertures
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn record_detector(
    r: &mut LocalResult,
    scene: &Scene3,
    det: usize,
    w: &RayWork,
    point: Vec3,
    uv: Vec2,
    dir_local: Vec3,
    flux: f64,
    opl: f64,
    cfg: &TraceConfig3,
) {
    r.report.note_exact(PhysicsFeature::AreaDetector);
    let d = &scene.detectors[det];
    let qe = detector_qe(scene, det, w.lambda);
    let stokes = w.state_at(flux);

    // The detector's own frame: its local `+X`, projected into the plane this
    // ray's field lives in. Every hit is rotated onto that before being summed,
    // so the aggregate is an addition of comparable components rather than of
    // numbers that happen to sit in the same struct. How far that common frame
    // can be trusted is the tilt recorded beside it.
    let axis_world = d.common.transform.rotate(Vec3::X);
    let dir = w.ray.dir;
    let along = dir.dot(axis_world);
    let projected = axis_world - dir * along;
    let det_x = if projected.length() > 1e-9 {
        projected.normalize()
    } else {
        // The ray runs along the detector's own X: no projection survives, so
        // fall back to the ray's frame, which at least names an axis.
        w.pol.x
    };
    let normal_world = d.common.transform.rotate(Vec3::Z);
    let tilt = dir.dot(normal_world).abs().clamp(0.0, 1.0).acos();

    let record = cfg.record_detector_hits && r.record_hits;
    let out = &mut r.detectors[det];
    out.hits += 1;
    out.total_incident += flux;
    out.total_signal += flux * qe;
    out.flux_uv += uv * flux;
    out.flux_opl += opl * flux;
    out.stokes_local = out
        .stokes_local
        .plus(&stokes.rotated(polarization::frame_angle(dir, w.pol.x, det_x)));
    out.max_tilt_rad = out.max_tilt_rad.max(tilt);
    if let Some(cell) = detector_cell(d, uv) {
        out.incident_flux[cell] += flux;
        out.detected_signal[cell] += flux * qe;
        if let Some(c) = out.colour_xyz.get_mut(cell) {
            let cmf = oc_core::colorimetry::cie_1931(w.lambda);
            for k in 0..3 {
                c[k] += flux * cmf[k];
            }
        }
    }
    // The angular histogram, when the detector was asked for one. A ray past
    // the cone is counted somewhere rather than nowhere.
    if d.measure.cone_rad().is_some() {
        match d.direction_cell(dir_local) {
            Some(cell) => out.direction_flux[cell] += flux,
            None => out.outside_cone_flux += flux,
        }
    }
    out.add_spectral(w.lambda, flux, uv, dir_local);
    if record && out.records.len() < cfg.max_records {
        out.records.push(DetectorHit3 {
            point_world: point,
            uv_mm: uv,
            incident_dir_local: dir_local,
            lambda_um: w.lambda,
            incident_flux: flux,
            signal: flux * qe,
            opl_mm: opl,
            path_id: w.path_id,
            parent_path_id: w.parent,
            depth: w.depth,
            ghost_depth: w.ghost_depth,
            stokes,
            frame_x: w.pol.x,
            counted: w.counted,
        });
    }
}

fn detector_qe(scene: &Scene3, det: usize, lambda: f64) -> f64 {
    scene.detectors[det]
        .response
        .efficiency_at(lambda)
        .unwrap_or(0.0)
}

/// Intersect a detector face (local `z = 0`), returning `(t, world point, uv,
/// incident dir in local frame)`. Line probes have no area in 3D and are skipped.
fn detector_hit(
    d: &Detector3,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
) -> Option<(f64, Vec3, Vec2, Vec3)> {
    if matches!(d.geometry, DetectorGeometry3::LineProbeLegacy { .. }) {
        return None;
    }
    let (t, uv) = plane_face_hit(&d.common.transform, ray, tmin, tmax, |p| {
        detector_contains(d, p)
    })?;
    let point = ray.at(t);
    let dir_local = d.common.transform.inv_rotate(ray.dir);
    Some((t, point, uv, dir_local))
}

fn detector_contains(d: &Detector3, p: Vec2) -> bool {
    match d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => p.x.abs() <= half_width_mm && p.y.abs() <= half_height_mm,
        DetectorGeometry3::Disc { radius_mm } => p.length_squared() <= radius_mm * radius_mm,
        DetectorGeometry3::Annulus { inner_mm, outer_mm } => {
            let r2 = p.length_squared();
            r2 >= inner_mm * inner_mm && r2 <= outer_mm * outer_mm
        }
        DetectorGeometry3::LineProbeLegacy { .. } => false,
    }
}

/// The `(t, uv)` of a ray crossing an object's local `z = 0` plane, gated by a
/// membership test. Shared by detectors and (a plane offset aside) apertures.
fn plane_face_hit(
    transform: &Isometry3,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    inside: impl Fn(Vec2) -> bool,
) -> Option<(f64, Vec2)> {
    let o = transform.inverse_transform_point(ray.origin);
    let dir = transform.inv_rotate(ray.dir);
    if dir.z.abs() < 1e-12 {
        return None;
    }
    let t = -o.z / dir.z;
    if t <= tmin || t > tmax {
        return None;
    }
    let p = o + dir * t;
    let uv = Vec2::new(p.x, p.y);
    inside(uv).then_some((t, uv))
}

/// If the ray is blocked by an aperture (outside the clear opening, inside the
/// mask), return `(t, world point)`. A ray through the opening has no event.
fn aperture_block(a: &Aperture3, ray: &Ray3, tmin: f64, tmax: f64) -> Option<(f64, Vec3)> {
    // The aperture plane lives in the object's local frame; compose it with the
    // placement so the ray meets it where the author put it.
    let plane = &a.shape.plane_local;
    let o = a.common.transform.inverse_transform_point(ray.origin);
    let dir = a.common.transform.inv_rotate(ray.dir);
    let denom = dir.dot(plane.normal);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = (plane.origin - o).dot(plane.normal) / denom;
    if t <= tmin || t > tmax {
        return None;
    }
    let uv = plane.to_local(o + dir * t);
    if planar_contains(&a.shape.clear_opening, uv) {
        return None; // passes through the opening: no event
    }
    let masked = match &a.shape.outer_extent {
        Some(ext) => planar_contains(ext, uv),
        None => true, // unbounded opaque mask
    };
    masked.then(|| (t, ray.at(t)))
}

fn planar_contains(shape: &PlanarShape2, p: Vec2) -> bool {
    oc_core::geom3::primitives::planar_shape_contains(shape, p)
}

// ---------------------------------------------------------------------------
// World AABBs for apertures and detectors
// ---------------------------------------------------------------------------

fn planar_shape_extent(shape: &PlanarShape2) -> f64 {
    match shape {
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => half_width_mm.hypot(*half_height_mm),
        PlanarShape2::Disc { radius_mm } => *radius_mm,
        PlanarShape2::Annulus { outer_mm, .. } => *outer_mm,
        PlanarShape2::Polygon { vertices } => {
            vertices.iter().map(|v| v.length()).fold(0.0, f64::max)
        }
    }
}

fn aperture_aabb(a: &Aperture3) -> Aabb3 {
    // A finite mask bounds within its outer extent; an unbounded mask is bounded
    // only loosely, which is fine — the BVH box may be large, it may not miss.
    let ext = a
        .shape
        .outer_extent
        .as_ref()
        .map(planar_shape_extent)
        .unwrap_or_else(|| planar_shape_extent(&a.shape.clear_opening) * 4.0);
    let plane = &a.shape.plane_local;
    let mut bb = Aabb3::EMPTY;
    for su in [-1.0, 1.0] {
        for sv in [-1.0, 1.0] {
            let local = plane.origin + plane.u * (su * ext) + plane.v * (sv * ext);
            bb.expand(a.common.transform.transform_point(local));
        }
    }
    bb.padded(1e-6 * ext.max(1.0))
}

fn detector_aabb(d: &Detector3) -> Option<Aabb3> {
    let ext = match d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => Vec2::new(half_width_mm, half_height_mm),
        DetectorGeometry3::Disc { radius_mm } => Vec2::splat(radius_mm),
        DetectorGeometry3::Annulus { outer_mm, .. } => Vec2::splat(outer_mm),
        DetectorGeometry3::LineProbeLegacy { .. } => return None,
    };
    let mut bb = Aabb3::EMPTY;
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            let local = Vec3::new(sx * ext.x, sy * ext.y, 0.0);
            bb.expand(d.common.transform.transform_point(local));
        }
    }
    Some(bb.padded(1e-6 * ext.max_element().max(1.0)))
}

/// The cell index a local `(u, v)` hit falls in, row-major over `bins`. A
/// rectangle maps to an even grid (equal-area cells, the normalization test);
/// disc and annulus fall back to their bounding box.
fn detector_cell(d: &Detector3, uv: Vec2) -> Option<usize> {
    let (nu, nv) = (d.bins[0].max(1), d.bins[1].max(1));
    let (fx, fy) = match d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => (
            (uv.x + half_width_mm) / (2.0 * half_width_mm),
            (uv.y + half_height_mm) / (2.0 * half_height_mm),
        ),
        DetectorGeometry3::Disc { radius_mm } => (
            (uv.x + radius_mm) / (2.0 * radius_mm),
            (uv.y + radius_mm) / (2.0 * radius_mm),
        ),
        DetectorGeometry3::Annulus { outer_mm, .. } => (
            (uv.x + outer_mm) / (2.0 * outer_mm),
            (uv.y + outer_mm) / (2.0 * outer_mm),
        ),
        DetectorGeometry3::LineProbeLegacy { .. } => return None,
    };
    if !(0.0..=1.0).contains(&fx) || !(0.0..=1.0).contains(&fy) {
        return None;
    }
    let iu = ((fx * nu as f64) as usize).min(nu - 1);
    let iv = ((fy * nv as f64) as usize).min(nv - 1);
    Some(iv * nu + iu)
}
