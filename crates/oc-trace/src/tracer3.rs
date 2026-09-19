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

use oc_core::geom3::{Containment, Intersect3, ToleranceContext};
use oc_core::id::{DetectorId, ObjectId, PathId, SurfaceRef};
use oc_core::scene3::{
    Aperture3, Detector3, DetectorGeometry3, DetectorTermination, PlanarShape2, Scene3,
    SolidGeometry3, SurfaceObject3, SurfaceResponse,
};
use oc_core::vec::Vec2;
use oc_core::vec3::{Aabb3, Isometry3, Ray3, Vec3};

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
            revision: 0,
            polarization: true,
        }
    }
}

/// The launch rays a single source emits, as the tracer's [`LaunchRay3`].
///
/// The sampler lives in `oc-core` ([`Source3::emit`](oc_core::source3)); this is
/// the thin wrapper that tags each ray with its source and hands it to transport.
pub fn launch_from_source(source: &oc_core::scene3::Source3) -> Vec<LaunchRay3> {
    source
        .emit()
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
    scene.sources.iter().flat_map(launch_from_source).collect()
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
    let objs = EventObjects::build(scene);
    let bvh = Bvh3::build(&objs.boxes);
    let tol = ToleranceContext::from_aabb(&objs.scene_aabb);

    // A scene with a PlanarOnly body cannot be traced authoritatively: it has no
    // 3D extent, so a full-3D trace of it is flagged rather than silently missing
    // the object (§3.2, §7.7).
    let mut base_report = CapabilityReport::default();
    for b in &scene.bodies {
        if !b.is_3d() {
            base_report.reject(CapabilityIssue::PlanarOnlyObject {
                object: b.common.id,
            });
        }
    }

    let identity = || Local {
        result: LocalResult::new(scene, cfg, base_report.clone()),
    };

    #[cfg(feature = "parallel")]
    let merged = {
        use rayon::prelude::*;
        launch
            .par_iter()
            .fold(identity, |mut acc, lr| {
                trace_one(scene, &objs, &bvh, &tol, cfg, *lr, &mut acc);
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
            trace_one(scene, &objs, &bvh, &tol, cfg, *lr, &mut acc);
        }
        acc
    };

    let r = merged.result;
    TraceResult3 {
        revision: cfg.revision,
        segments: r.segments,
        detectors: r.detectors,
        ledger: r.ledger,
        terminations: r.terminations,
        capability_report: r.report,
        events: r.events,
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
    detectors: Vec<DetectorResult3>,
    ledger: EnergyLedger,
    terminations: TerminationSummary,
    report: CapabilityReport,
    events: Vec<TraceEvent3>,
    next_path: u64,
}

impl LocalResult {
    fn new(scene: &Scene3, _cfg: &TraceConfig3, report: CapabilityReport) -> Self {
        let detectors = scene
            .detectors
            .iter()
            .map(|d| DetectorResult3::new(DetectorId(d.common.id.0), d.bins, detector_cell_area(d)))
            .collect();
        Self {
            segments: Vec::new(),
            detectors,
            ledger: EnergyLedger::default(),
            terminations: TerminationSummary::default(),
            report,
            events: Vec::new(),
            next_path: 0,
        }
    }

    #[cfg(feature = "parallel")]
    fn merge(&mut self, o: LocalResult) {
        self.segments.extend(o.segments);
        for (a, b) in self.detectors.iter_mut().zip(&o.detectors) {
            a.merge(b);
        }
        self.ledger.merge(&o.ledger);
        self.terminations.merge(&o.terminations);
        self.report.merge(&o.report);
        self.events.extend(o.events);
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
        for (i, b) in scene.bodies.iter().enumerate() {
            if !b.is_3d() {
                continue; // no 3D extent; flagged in the report, contributes no events
            }
            boxes.push(b.geometry.aabb_world(&b.common.transform));
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
    });

    let mut branches = 0u64;
    while let Some(w) = stack.pop() {
        branches += 1;
        if branches > cfg.max_branches {
            // Deposit the popped branch and everything still queued, so the cap
            // subtracts a named quantity rather than a silent one.
            r.ledger.truncated_flux += w.flux() + stack.iter().map(|x| x.flux()).sum::<f64>();
            r.terminations.record(TerminationReason::MaxBranches);
            break;
        }
        if w.depth > cfg.max_depth {
            deposit_truncated(r, &w, TerminationReason::MaxDepth);
            continue;
        }
        if w.ghost && w.ghost_depth > cfg.max_ghost_bounces {
            deposit_truncated(r, &w, TerminationReason::MaxGhostDepth);
            continue;
        }
        if w.flux() < min_flux {
            deposit_truncated(r, &w, TerminationReason::FluxThreshold);
            continue;
        }

        // The medium this branch travels through: containment + priority, sampled
        // just ahead of the origin so a ray leaving a surface reads the medium it
        // is entering.
        let here = region_at(scene, w.ray.origin + medium_eps * w.ray.dir, tol);
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
            continue;
        };

        let l = ev.t;
        let flux_at = w.flux() * beer_lambert(alpha, l);
        record_absorption(r, w.flux() - flux_at);
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
                interact_surface(
                    scene, &mut stack, r, cfg, &w, surf, ev.point, normal, n_here, flux_at, opl_at,
                    min_flux,
                );
            }
            EventKind::Body {
                body,
                outward,
                patch,
            } => {
                interact_body(
                    scene, &mut stack, r, cfg, tol, &w, body, ev.point, outward, patch, here,
                    n_here, flux_at, opl_at, medium_eps, min_flux,
                );
            }
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

    let consider = |ev: Event, best: &mut Option<Event>, best_t: &mut f64| {
        if ev.t < *best_t {
            *best_t = ev.t;
            *best = Some(ev);
        }
    };

    bvh.candidates(&w.ray, skip, best_t, &mut |oi| match &objs.kinds[oi] {
        ObjKind::Body(bi) => {
            let b = &scene.bodies[*bi];
            b.geometry
                .intersect(&w.ray, &b.common.transform, skip, best_t, &mut |h| {
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
                });
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
            if let Some((t, point)) = aperture_block(&scene.apertures[*ai], &w.ray, skip, best_t) {
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
    });

    best
}

// ---------------------------------------------------------------------------
// Medium resolution (§9.4)
// ---------------------------------------------------------------------------

/// The body whose material occupies `point`: highest priority, ties broken by
/// highest id (a stable key, never vector order, §9.2). `None` is the ambient.
fn region_at(scene: &Scene3, point: Vec3, tol: &ToleranceContext) -> Option<usize> {
    let mut best: Option<(usize, i32, u64)> = None;
    for (i, b) in scene.bodies.iter().enumerate() {
        if !b.is_3d() {
            continue;
        }
        if b.geometry.contains(point, &b.common.transform, tol) == Containment::Inside {
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
    let far = region_at(scene, point - medium_eps * n_oppose, tol);
    let n_far = region_index(scene, far, w.lambda);
    let _ = here;

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
    let n_oppose = if from_front { normal } else { -normal };
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
                    surface: surface_ref,
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
            if cfg.polarization {
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
            r.report.reject(CapabilityIssue::UnsupportedSurfaceResponse {
                surface: surface_ref,
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
                    r.report.reject(CapabilityIssue::UnsupportedSurfaceResponse {
                        surface: surface_ref,
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
            r.report
                .note_approx(feature, "element faces added singly: no bounce between them");
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
    let phase_assumed = cfg.polarization && stokes_in.dop() > 1e-12;
    let out = |m: &Mueller| -> PolRay {
        let stokes = if cfg.polarization {
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
                        .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface });
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
                        .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface });
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
            let rt = oc_core::thinfilm::coating_rt_at(coating, n1, n2, w.lambda, cos_i);
            let (refl, tr) = match rt {
                Some(v) => (v.r().clamp(0.0, 1.0), v.t().clamp(0.0, 1.0)),
                None => {
                    r.report
                        .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface });
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
            let amp = oc_core::thinfilm::coating_amp_at(coating, n1, n2, w.lambda, cos_i);
            let (m_r, m_t) = match &amp {
                Some(a) => (
                    Mueller::from_diagonal_jones(a.r_s, a.r_p),
                    Mueller::from_diagonal_jones(a.t_s, a.t_p).scaled(a.t_power_scale),
                ),
                None => {
                    if phase_assumed {
                        r.report.note_approx(
                            PhysicsFeature::ThinFilmCoating,
                            "plate-backed stack: faces added incoherently, no relative phase",
                        );
                    }
                    let v = rt.as_ref().expect("checked above");
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
            r.ledger.absorbed_surface += (flux_at - pol_r.stokes.s0 - pol_t.stokes.s0).max(0.0);
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
                if let Some(id) =
                    spawn_tir(stack, r, cfg, w, point, dir_r, pol_r, opl_at, min_flux)
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
        | SurfaceResponse::Polarizer { .. }
        | SurfaceResponse::Retarder { .. } => {
            r.report
                .reject(CapabilityIssue::UnsupportedSurfaceResponse { surface });
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
    }
    if cfg.record_detector_hits && out.records.len() < cfg.max_records {
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

fn detector_cell_area(d: &Detector3) -> Option<f64> {
    let (nu, nv) = (d.bins[0].max(1), d.bins[1].max(1));
    let total = match d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => 4.0 * half_width_mm * half_height_mm,
        DetectorGeometry3::Disc { radius_mm } => std::f64::consts::PI * radius_mm * radius_mm,
        DetectorGeometry3::Annulus { inner_mm, outer_mm } => {
            std::f64::consts::PI * (outer_mm * outer_mm - inner_mm * inner_mm)
        }
        DetectorGeometry3::LineProbeLegacy { .. } => return None,
    };
    Some(total / (nu * nv) as f64)
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
