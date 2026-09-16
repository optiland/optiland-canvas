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
use crate::fresnel::{beer_lambert, reflect3, reflectance, refract3};
use crate::result3::*;

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
    /// The scene revision, copied into the result so a stale trace is discarded
    /// rather than shown (§3.3).
    pub revision: u64,
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
            revision: 0,
        }
    }
}

/// Trace a set of explicit launch rays through the scene.
///
/// Sources are turned into launch rays by PR 3D-06; here the caller supplies them
/// so the transport engine is exercised on hand-built geometry.
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
            next_path: 0,
        }
    }

    fn merge(&mut self, o: LocalResult) {
        self.segments.extend(o.segments);
        for (a, b) in self.detectors.iter_mut().zip(&o.detectors) {
            a.merge(b);
        }
        self.ledger.merge(&o.ledger);
        self.terminations.merge(&o.terminations);
        self.report.merge(&o.report);
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
    flux: f64,
    depth: u32,
    ghost_depth: u32,
    opl: f64,
    path_id: PathId,
    parent: Option<PathId>,
    from_tir: bool,
    ghost: bool,
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
        flux: lr.flux,
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
            r.ledger.truncated_flux += w.flux + stack.iter().map(|x| x.flux).sum::<f64>();
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
        if w.flux < min_flux {
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
            let flux_end = w.flux * beer_lambert(alpha, cfg.bounds_mm);
            record_absorption(r, w.flux - flux_end);
            push_segment(r, &w, end, flux_end, w.opl + n_here * cfg.bounds_mm);
            r.ledger.escaped += flux_end;
            r.terminations.record(TerminationReason::EscapedBounds);
            continue;
        };

        let l = ev.t;
        let flux_at = w.flux * beer_lambert(alpha, l);
        record_absorption(r, w.flux - flux_at);
        let opl_at = w.opl + n_here * l;
        push_segment(r, &w, ev.point, flux_at, opl_at);

        match ev.kind {
            EventKind::Detector { det, uv, dir_local } => {
                let terminates = matches!(
                    scene.detectors[det].termination,
                    DetectorTermination::Terminate
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
                let _ = ap;
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
    r.ledger.truncated_flux += w.flux;
    r.terminations.record(reason);
}

fn record_absorption(r: &mut LocalResult, absorbed: f64) {
    if absorbed > 0.0 {
        r.ledger.absorbed_volume += absorbed;
    }
}

fn push_segment(r: &mut LocalResult, w: &RayWork, b: Vec3, flux_end: f64, opl1: f64) {
    r.segments.push(Segment3 {
        a: w.ray.origin,
        b,
        flux_start: w.flux,
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
    // A zero-thickness surface has the same medium on both sides (§6.6): no
    // refraction, transmission continues undeviated.
    apply_response(
        stack,
        r,
        cfg,
        w,
        response,
        SurfaceRef::new(s.common.id, oc_core::id::PatchId(0)),
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

    match response {
        SurfaceResponse::Absorber => {
            r.ledger.absorbed_surface += flux_at;
            r.terminations.record(TerminationReason::AbsorbedSurface);
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
                    return;
                }
            };
            r.ledger.absorbed_surface += flux_at * (1.0 - refl);
            let dir = reflect3(w.ray.dir, n_oppose);
            spawn(
                stack,
                r,
                cfg,
                w,
                point,
                dir,
                flux_at * refl,
                opl_at,
                false,
                true,
                min_flux,
            );
        }
        SurfaceResponse::DielectricFresnel { ar_residual } => {
            r.report.note_exact(PhysicsFeature::DielectricFresnel);
            let ar = sample(ar_residual).unwrap_or(1.0).clamp(0.0, 1.0);
            match refract3(w.ray.dir, n_oppose, n1, n2) {
                None => {
                    // Total internal reflection: R = 1, primary path.
                    let dir = reflect3(w.ray.dir, n_oppose);
                    spawn_tir(stack, r, cfg, w, point, dir, flux_at, opl_at, min_flux);
                }
                Some(dir_t) => {
                    let refl = (reflectance(cos_i, n1, n2) * ar).clamp(0.0, 1.0);
                    let tr = 1.0 - refl;
                    spawn(
                        stack,
                        r,
                        cfg,
                        w,
                        point,
                        dir_t,
                        flux_at * tr,
                        opl_at,
                        false,
                        false,
                        min_flux,
                    );
                    if cfg.ghosts && refl > 0.0 {
                        let dir_r = reflect3(w.ray.dir, n_oppose);
                        spawn(
                            stack,
                            r,
                            cfg,
                            w,
                            point,
                            dir_r,
                            flux_at * refl,
                            opl_at,
                            true,
                            false,
                            min_flux,
                        );
                    } else if refl > 0.0 {
                        // Ghosts off: the reflected flux is not followed, but it is
                        // still named rather than lost.
                        r.ledger.truncated_flux += flux_at * refl;
                    }
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
                    return;
                }
            };
            r.ledger.absorbed_surface += flux_at * (1.0 - refl - tr).max(0.0);
            // Reflected branch (specular).
            if refl > 0.0 {
                let dir_r = reflect3(w.ray.dir, n_oppose);
                spawn(
                    stack,
                    r,
                    cfg,
                    w,
                    point,
                    dir_r,
                    flux_at * refl,
                    opl_at,
                    w.ghost,
                    false,
                    min_flux,
                );
            }
            // Transmitted branch: Snell across a real medium transition, else
            // undeviated (§9.6).
            if tr > 0.0 {
                let dir_t = if medium_transition {
                    refract3(w.ray.dir, n_oppose, n1, n2).unwrap_or(w.ray.dir)
                } else {
                    w.ray.dir
                };
                spawn(
                    stack,
                    r,
                    cfg,
                    w,
                    point,
                    dir_t,
                    flux_at * tr,
                    opl_at,
                    w.ghost,
                    false,
                    min_flux,
                );
            }
        }
    }
}

/// Push a child branch, or deposit its flux in `truncated_flux` if it is born
/// already below threshold — so a split conserves flux exactly.
#[allow(clippy::too_many_arguments)]
fn spawn(
    stack: &mut Vec<RayWork>,
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    parent: &RayWork,
    point: Vec3,
    dir: Vec3,
    flux: f64,
    opl: f64,
    ghost: bool,
    from_tir: bool,
    min_flux: f64,
) {
    let depth = parent.depth + 1;
    let ghost_depth = parent.ghost_depth + u32::from(ghost && !parent.ghost);
    let is_ghost = parent.ghost || ghost;
    if flux < min_flux {
        r.ledger.truncated_flux += flux;
        r.terminations.record(TerminationReason::FluxThreshold);
        return;
    }
    if depth > cfg.max_depth {
        r.ledger.truncated_flux += flux;
        r.terminations.record(TerminationReason::MaxDepth);
        return;
    }
    let path_id = r.new_path();
    let ray = match Ray3::new(point, dir) {
        Ok(ray) => ray,
        Err(_) => {
            r.ledger.numerical_loss += flux;
            r.terminations.record(TerminationReason::NumericalFailure);
            return;
        }
    };
    stack.push(RayWork {
        ray,
        lambda: parent.lambda,
        flux,
        depth,
        ghost_depth,
        opl,
        path_id,
        parent: Some(parent.path_id),
        from_tir,
        ghost: is_ghost,
    });
}

#[allow(clippy::too_many_arguments)]
fn spawn_tir(
    stack: &mut Vec<RayWork>,
    r: &mut LocalResult,
    cfg: &TraceConfig3,
    parent: &RayWork,
    point: Vec3,
    dir: Vec3,
    flux: f64,
    opl: f64,
    min_flux: f64,
) {
    // TIR is the primary path (R = 1), so it does not increment the ghost budget.
    spawn(
        stack,
        r,
        cfg,
        parent,
        point,
        dir,
        flux,
        opl,
        parent.ghost,
        true,
        min_flux,
    );
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
            flux,
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
    let out = &mut r.detectors[det];
    out.hits += 1;
    out.total_incident += flux;
    out.total_signal += flux * qe;
    out.flux_uv += uv * flux;
    out.flux_opl += opl * flux;
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
