//! The f32 CPU 3D tracer: the oracle the WGSL shader is validated against (§12,
//! PR 3D-07).
//!
//! It mirrors the f64 reference ([`oc_trace::trace3`]) in miniature — nearest-hit
//! transport, media by containment plus priority, vector Snell with scalar
//! Fresnel and exact TIR, deterministic reflected/transmitted branches — but in
//! f32 and in the normalized frame of [`GpuScene3`], so it does exactly what the
//! shader will do and can stand in for it wherever no GPU is present.
//!
//! It traces the objects the compiler admits: the analytic bodies (sphere, box,
//! dielectric or absorbing), zero-thickness surfaces (mirror, absorber, scalar
//! dielectric, explicit R/T), finite aperture masks, and area detectors. The
//! curved profile bodies (conic and asphere of revolution, extrusions) are the
//! one remaining piece; until they land the compiler rejects them by name rather
//! than the tracer guessing.

use glam::{DVec3, Vec2, Vec3};

use crate::scene3::{
    GpuAperture3, GpuBody3, GpuDetector3, GpuResponse, GpuScene3, GpuSurface3, OBJ_BOX, OBJ_SPHERE,
    SURF_ABSORBER,
};

/// Segment flag bits (shared meaning with the 2D path).
pub const SEG_TIR: u32 = 2;
pub const SEG_GHOST: u32 = 4;

/// A traced segment in world millimetres.
#[derive(Debug, Clone, Copy)]
pub struct SegOut3 {
    pub a: DVec3,
    pub b: DVec3,
    pub flux: f32,
    pub lambda_um: f32,
    pub flags: u32,
}

impl SegOut3 {
    pub fn is_tir(&self) -> bool {
        self.flags & SEG_TIR != 0
    }
    pub fn is_ghost(&self) -> bool {
        self.flags & SEG_GHOST != 0
    }
}

/// An area detector's accumulated readout across every launch ray.
#[derive(Debug, Clone)]
pub struct DetReadout3 {
    pub object: u64,
    pub bins: [usize; 2],
    pub cell_area_mm2: f32,
    /// Incident flux per cell, row-major over `bins`.
    pub incident: Vec<f32>,
    pub hits: u64,
    pub total: f32,
}

impl DetReadout3 {
    /// Relative flux density per cell (`flux / area`).
    pub fn density(&self) -> Vec<f32> {
        self.incident
            .iter()
            .map(|f| f / self.cell_area_mm2)
            .collect()
    }
}

/// The result of an f32 preview trace.
#[derive(Debug, Clone)]
pub struct Cpu3Result {
    /// Segments per launch ray, in world millimetres.
    pub segments: Vec<Vec<SegOut3>>,
    /// One readout per compiled detector, in scene order.
    pub detectors: Vec<DetReadout3>,
}

impl Cpu3Result {
    /// A detector readout by object id.
    pub fn detector(&self, object: u64) -> Option<&DetReadout3> {
        self.detectors.iter().find(|d| d.object == object)
    }
}

/// Self-hit offset and medium-probe distance, in the normalized frame. f32 across
/// a ~unit-scale scene resolves this comfortably.
const EPS: f32 = 1e-4;
/// Branch-count safety cap per launch ray.
const MAX_BRANCHES: usize = 4096;

#[derive(Clone, Copy)]
struct Work {
    o: Vec3,
    d: Vec3,
    flux: f32,
    depth: u32,
    gdepth: u32,
    /// Index and attenuation of the medium this branch travels in.
    medium_n: f32,
    medium_alpha: f32,
    flags: u32,
}

/// Trace every launch ray, returning segments and detector readouts.
pub fn trace(scene: &GpuScene3, max_depth: u32) -> Cpu3Result {
    let mut detectors: Vec<DetReadout3> = scene
        .detectors
        .iter()
        .map(|d| DetReadout3 {
            object: d.object,
            bins: d.bins,
            cell_area_mm2: d.cell_area_mm2,
            incident: vec![0.0; d.bins[0].max(1) * d.bins[1].max(1)],
            hits: 0,
            total: 0.0,
        })
        .collect();

    let segments = scene
        .rays
        .iter()
        .map(|r| trace_one(scene, *r, max_depth, &mut detectors))
        .collect();

    Cpu3Result {
        segments,
        detectors,
    }
}

/// Trace segments only, for the many call sites that do not read detectors.
pub fn trace_segments(scene: &GpuScene3, max_depth: u32) -> Vec<Vec<SegOut3>> {
    trace(scene, max_depth).segments
}

fn trace_one(
    scene: &GpuScene3,
    ray: crate::scene3::GpuRay3,
    max_depth: u32,
    detectors: &mut [DetReadout3],
) -> Vec<SegOut3> {
    let (o, d, flux, lambda) = (ray.origin, ray.dir, ray.flux, ray.lambda_um);
    let min_flux = scene.flux_eps * flux;
    let mut out = Vec::new();

    let (n0, a0) = region_optics(scene, o, lambda);
    let mut stack = vec![Work {
        o,
        d,
        flux,
        depth: 0,
        gdepth: 0,
        medium_n: n0,
        medium_alpha: a0,
        flags: 0,
    }];

    let mut branches = 0usize;
    while let Some(w) = stack.pop() {
        branches += 1;
        if branches > MAX_BRANCHES || w.depth >= max_depth || w.flux < min_flux {
            continue;
        }

        let Some(ev) = nearest_event(scene, w.o, w.d) else {
            let end = w.o + w.d * scene.bounds_norm;
            out.push(seg(scene, &w, end, w.flux, lambda));
            continue;
        };

        let point = w.o + w.d * ev.t;
        let flux_end = w.flux * beer_lambert(w.medium_alpha, scene.norm.world_len(ev.t));
        out.push(seg(scene, &w, point, flux_end, lambda));

        match ev.kind {
            EvKind::Body { body, normal } => interact_body(
                scene, &mut stack, &w, body, point, normal, flux_end, lambda, max_depth, min_flux,
            ),
            EvKind::Surface { surf, normal } => interact_surface(
                scene, &mut stack, &w, surf, point, normal, flux_end, max_depth, min_flux,
            ),
            EvKind::Aperture => { /* blocked: the branch ends here */ }
            EvKind::Detector { det, uv } => {
                record_detector(&mut detectors[det], &scene.detectors[det], uv, flux_end);
                if !scene.detectors[det].terminate {
                    // Probe: continue undeviated, just past the face.
                    push(
                        &mut stack,
                        Work {
                            o: point + w.d * EPS,
                            ..w
                        },
                        max_depth,
                        min_flux,
                    );
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Interactions
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn interact_body(
    scene: &GpuScene3,
    stack: &mut Vec<Work>,
    w: &Work,
    body: usize,
    point: Vec3,
    hit_normal: Vec3,
    flux_end: f32,
    lambda: f32,
    max_depth: u32,
    min_flux: f32,
) {
    if scene.bodies[body].surface == SURF_ABSORBER {
        return; // swallowed at the surface
    }
    let n = if hit_normal.dot(w.d) > 0.0 {
        -hit_normal
    } else {
        hit_normal
    };
    let (n1, _a1) = region_optics(scene, point + n * EPS, lambda);
    let (n2, a2) = region_optics(scene, point - n * EPS, lambda);

    match refract(w.d, n, n1, n2) {
        None => {
            let dir = reflect(w.d, n);
            push(
                stack,
                Work {
                    o: point + dir * EPS,
                    d: dir,
                    flux: flux_end,
                    depth: w.depth + 1,
                    gdepth: w.gdepth,
                    medium_n: w.medium_n,
                    medium_alpha: w.medium_alpha,
                    flags: w.flags | SEG_TIR,
                },
                max_depth,
                min_flux,
            );
        }
        Some(td) => {
            let cos_i = (-n.dot(w.d)).clamp(0.0, 1.0);
            let r = reflectance(cos_i, n1, n2).clamp(0.0, 1.0);
            push(
                stack,
                Work {
                    o: point + td * EPS,
                    d: td,
                    flux: flux_end * (1.0 - r),
                    depth: w.depth + 1,
                    gdepth: w.gdepth,
                    medium_n: n2,
                    medium_alpha: a2,
                    flags: w.flags,
                },
                max_depth,
                min_flux,
            );
            if scene.ghosts && w.gdepth < scene.max_ghost && r > 0.0 {
                let dir = reflect(w.d, n);
                push(
                    stack,
                    Work {
                        o: point + dir * EPS,
                        d: dir,
                        flux: flux_end * r,
                        depth: w.depth + 1,
                        gdepth: w.gdepth + 1,
                        medium_n: w.medium_n,
                        medium_alpha: w.medium_alpha,
                        flags: w.flags | SEG_GHOST,
                    },
                    max_depth,
                    min_flux,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn interact_surface(
    scene: &GpuScene3,
    stack: &mut Vec<Work>,
    w: &Work,
    surf: usize,
    point: Vec3,
    normal: Vec3,
    flux_end: f32,
    max_depth: u32,
    min_flux: f32,
) {
    let s: &GpuSurface3 = &scene.surfaces[surf];
    // Front is the local +Z side; incident on the front when running against it.
    let from_front = w.d.dot(normal) < 0.0;
    let resp = if from_front { s.front } else { s.back };
    let n_oppose = if from_front { normal } else { -normal };

    // A zero-thickness surface has the same medium on both sides (§6.6): no
    // refraction, transmission continues undeviated.
    let transmit = |stack: &mut Vec<Work>, flux: f32| {
        push(
            stack,
            Work {
                o: point + w.d * EPS,
                d: w.d,
                flux,
                depth: w.depth + 1,
                gdepth: w.gdepth,
                medium_n: w.medium_n,
                medium_alpha: w.medium_alpha,
                flags: w.flags,
            },
            max_depth,
            min_flux,
        );
    };
    let reflect_branch = |stack: &mut Vec<Work>, flux: f32, ghost: bool| {
        let dir = reflect(w.d, n_oppose);
        let mut flags = w.flags;
        let mut gdepth = w.gdepth;
        if ghost {
            flags |= SEG_GHOST;
            gdepth += 1;
        }
        push(
            stack,
            Work {
                o: point + dir * EPS,
                d: dir,
                flux,
                depth: w.depth + 1,
                gdepth,
                medium_n: w.medium_n,
                medium_alpha: w.medium_alpha,
                flags,
            },
            max_depth,
            min_flux,
        );
    };

    match resp {
        GpuResponse::Absorber => { /* terminate */ }
        GpuResponse::Mirror { r } => {
            reflect_branch(stack, flux_end * r.clamp(0.0, 1.0), false);
        }
        GpuResponse::Dielectric { ar } => {
            // Same medium both sides: Fresnel at n1 = n2 is ~0, so the transmitted
            // ray is the whole beam undeviated; the AR-scaled sliver reflects.
            let cos_i = (-n_oppose.dot(w.d)).clamp(0.0, 1.0);
            let r = (reflectance(cos_i, w.medium_n, w.medium_n) * ar).clamp(0.0, 1.0);
            transmit(stack, flux_end * (1.0 - r));
            if scene.ghosts && w.gdepth < scene.max_ghost && r > 0.0 {
                reflect_branch(stack, flux_end * r, true);
            }
        }
        GpuResponse::ExplicitRt { r, t } => {
            let (r, t) = (r.clamp(0.0, 1.0), t.clamp(0.0, 1.0));
            transmit(stack, flux_end * t);
            if scene.ghosts && w.gdepth < scene.max_ghost && r > 0.0 {
                reflect_branch(stack, flux_end * r, true);
            }
        }
    }
}

fn record_detector(out: &mut DetReadout3, d: &GpuDetector3, uv: Vec2, flux: f32) {
    out.hits += 1;
    out.total += flux;
    if let Some(cell) = d.shape.cell(uv, d.bins) {
        out.incident[cell] += flux;
    }
}

fn push(stack: &mut Vec<Work>, w: Work, max_depth: u32, min_flux: f32) {
    if w.depth <= max_depth && w.flux >= min_flux {
        stack.push(w);
    }
}

fn seg(scene: &GpuScene3, w: &Work, end_norm: Vec3, flux_end: f32, lambda: f32) -> SegOut3 {
    SegOut3 {
        a: scene.norm.world_point(w.o),
        b: scene.norm.world_point(end_norm),
        flux: flux_end,
        lambda_um: lambda,
        flags: w.flags,
    }
}

// ---------------------------------------------------------------------------
// Nearest event
// ---------------------------------------------------------------------------

enum EvKind {
    Body { body: usize, normal: Vec3 },
    Surface { surf: usize, normal: Vec3 },
    Aperture,
    Detector { det: usize, uv: Vec2 },
}

struct Event {
    t: f32,
    kind: EvKind,
}

fn nearest_event(scene: &GpuScene3, o: Vec3, d: Vec3) -> Option<Event> {
    let mut best: Option<Event> = None;
    let mut consider = |t: f32, kind: EvKind| {
        if best.as_ref().map(|e| t < e.t).unwrap_or(true) {
            best = Some(Event { t, kind });
        }
    };

    for (bi, b) in scene.bodies.iter().enumerate() {
        if let Some((t, nl)) = body_hit(b, o, d) {
            consider(
                t,
                EvKind::Body {
                    body: bi,
                    normal: (b.rot * nl).normalize(),
                },
            );
        }
    }
    for (si, s) in scene.surfaces.iter().enumerate() {
        if let Some((t, normal)) = surface_hit(s, o, d) {
            consider(t, EvKind::Surface { surf: si, normal });
        }
    }
    for a in &scene.apertures {
        if let Some(t) = aperture_hit(a, o, d) {
            consider(t, EvKind::Aperture);
        }
    }
    for (di, det) in scene.detectors.iter().enumerate() {
        if let Some((t, uv)) = detector_hit(det, o, d) {
            consider(t, EvKind::Detector { det: di, uv });
        }
    }
    best
}

fn body_hit(b: &GpuBody3, o: Vec3, d: Vec3) -> Option<(f32, Vec3)> {
    let ol = b.rot.transpose() * (o - b.pos);
    let dl = b.rot.transpose() * d;
    match b.kind {
        OBJ_SPHERE => sphere_hit(ol, dl, b.params.x, EPS),
        OBJ_BOX => box_hit(ol, dl, b.params, EPS),
        _ => None,
    }
}

/// A zero-thickness surface's `(t, world normal)` — its local `z = 0` plane,
/// gated by the face shape.
fn surface_hit(s: &GpuSurface3, o: Vec3, d: Vec3) -> Option<(f32, Vec3)> {
    let (t, uv) = plane_face(s.rot, s.pos, o, d)?;
    s.shape
        .contains(uv)
        .then(|| (t, (s.rot * Vec3::Z).normalize()))
}

/// A detector face hit: `(t, local uv)`.
fn detector_hit(det: &GpuDetector3, o: Vec3, d: Vec3) -> Option<(f32, Vec2)> {
    let (t, uv) = plane_face(det.rot, det.pos, o, d)?;
    det.shape.contains(uv).then_some((t, uv))
}

/// The `(t, local uv)` of a ray crossing an object's local `z = 0` plane.
fn plane_face(rot: glam::Mat3, pos: Vec3, o: Vec3, d: Vec3) -> Option<(f32, Vec2)> {
    let ol = rot.transpose() * (o - pos);
    let dl = rot.transpose() * d;
    if dl.z.abs() < 1e-12 {
        return None;
    }
    let t = -ol.z / dl.z;
    if t <= EPS {
        return None;
    }
    let p = ol + dl * t;
    Some((t, Vec2::new(p.x, p.y)))
}

/// If a ray is blocked by an aperture (outside the opening, inside the mask),
/// its `t`. A ray through the opening has no event.
fn aperture_hit(a: &GpuAperture3, o: Vec3, d: Vec3) -> Option<f32> {
    let denom = d.dot(a.normal);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = (a.origin - o).dot(a.normal) / denom;
    if t <= EPS {
        return None;
    }
    let rel = o + d * t - a.origin;
    let uv = Vec2::new(rel.dot(a.u), rel.dot(a.v));
    if a.clear.contains(uv) {
        return None; // passes through
    }
    let masked = a.outer.as_ref().map(|s| s.contains(uv)).unwrap_or(true);
    masked.then_some(t)
}

// ---------------------------------------------------------------------------
// Body geometry
// ---------------------------------------------------------------------------

/// Nearest sphere intersection at `t > tmin`, with the outward local normal.
fn sphere_hit(o: Vec3, d: Vec3, r: f32, tmin: f32) -> Option<(f32, Vec3)> {
    let b = o.dot(d);
    let c = o.dot(o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let (t0, t1) = (-b - s, -b + s);
    let t = if t0 > tmin {
        t0
    } else if t1 > tmin {
        t1
    } else {
        return None;
    };
    let p = o + d * t;
    Some((t, p / r))
}

/// Nearest box intersection at `t > tmin`, with the outward local face normal.
fn box_hit(o: Vec3, d: Vec3, half: Vec3, tmin: f32) -> Option<(f32, Vec3)> {
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    let mut enter_axis = 0usize;
    let mut exit_axis = 0usize;
    for i in 0..3 {
        let (oi, di, hi) = (o[i], d[i], half[i]);
        if di.abs() < 1e-20 {
            if oi < -hi || oi > hi {
                return None;
            }
            continue;
        }
        let inv = 1.0 / di;
        let mut t0 = (-hi - oi) * inv;
        let mut t1 = (hi - oi) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        if t0 > t_enter {
            t_enter = t0;
            enter_axis = i;
        }
        if t1 < t_exit {
            t_exit = t1;
            exit_axis = i;
        }
        if t_enter > t_exit {
            return None;
        }
    }
    let (t, ax) = if t_enter > tmin {
        (t_enter, enter_axis)
    } else if t_exit > tmin {
        (t_exit, exit_axis)
    } else {
        return None;
    };
    let p = o + d * t;
    let mut nl = Vec3::ZERO;
    nl[ax] = p[ax].signum();
    Some((t, nl))
}

/// A point's medium optics: index and attenuation of the highest-priority body
/// containing it, else the ambient (§9.4).
fn region_optics(scene: &GpuScene3, p: Vec3, lambda: f32) -> (f32, f32) {
    let mut best: Option<&GpuBody3> = None;
    for b in &scene.bodies {
        if contains(b, p) {
            match best {
                Some(cur) if cur.priority >= b.priority => {}
                _ => best = Some(b),
            }
        }
    }
    match best {
        Some(b) if b.surface != SURF_ABSORBER => (b.index(lambda), b.alpha_per_mm),
        _ => (scene.ambient, 0.0),
    }
}

fn contains(b: &GpuBody3, p: Vec3) -> bool {
    let pl = b.rot.transpose() * (p - b.pos);
    match b.kind {
        OBJ_SPHERE => pl.length_squared() <= b.params.x * b.params.x,
        OBJ_BOX => pl.x.abs() <= b.params.x && pl.y.abs() <= b.params.y && pl.z.abs() <= b.params.z,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Optics (f32, unpolarized)
// ---------------------------------------------------------------------------

fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    (d - 2.0 * d.dot(n) * n).normalize()
}

/// Refracted direction, or `None` at total internal reflection. `n` opposes `d`.
fn refract(d: Vec3, n: Vec3, n1: f32, n2: f32) -> Option<Vec3> {
    let eta = n1 / n2;
    let cos_i = -n.dot(d);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        None
    } else {
        Some((eta * d + (eta * cos_i - k.sqrt()) * n).normalize())
    }
}

fn reflectance(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let ci = cos_i.abs().clamp(0.0, 1.0);
    let s2 = 1.0 - ci * ci;
    let st2 = (n1 / n2) * (n1 / n2) * s2;
    if st2 >= 1.0 {
        return 1.0;
    }
    let ct = (1.0 - st2).sqrt();
    let rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    let rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    0.5 * (rs * rs + rp * rp)
}

fn beer_lambert(alpha_per_mm: f32, len_mm: f64) -> f32 {
    if alpha_per_mm <= 0.0 {
        1.0
    } else {
        (-alpha_per_mm as f64 * len_mm).exp() as f32
    }
}
