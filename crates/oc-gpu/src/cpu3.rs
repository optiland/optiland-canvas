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
//! cylinder), extruded line profiles (prisms), revolved profiles (conic and
//! even-asphere surfaces of revolution, plus their frustum rims), zero-thickness
//! surfaces (mirror, absorber, scalar dielectric, explicit R/T, thin lens, with
//! constant or tabulated response scalars), finite aperture masks, and area
//! detectors. The conic of revolution is an exact quadratic and the asphere a
//! safeguarded Newton with an analytic derivative (§7.3–7.4). What the compiler
//! still cannot trace exactly it rejects by name — an off-axis arc, a
//! curved-edge extrusion, tabulated volume absorption — rather than guessing.

use glam::{DVec3, Vec2, Vec3};

use crate::scene3::{
    even_asphere_dsag, even_asphere_sag, GpuAperture3, GpuBody3, GpuDetector3, GpuExtrude,
    GpuResponse, GpuRevolve, GpuScene3, GpuSurface3, RevFace, Spectral32, OBJ_BOX, OBJ_CYLINDER,
    OBJ_EXTRUDE, OBJ_REVOLVE, OBJ_SPHERE, SURF_ABSORBER,
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
                scene, &mut stack, &w, surf, point, normal, flux_end, lambda, max_depth, min_flux,
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
                    flags: w.flags & !SEG_TIR, // transmitted: clear the TIR mark
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
                        flags: (w.flags & !SEG_TIR) | SEG_GHOST, // Fresnel ghost, not TIR
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
    lambda: f32,
    max_depth: u32,
    min_flux: f32,
) {
    let s: &GpuSurface3 = &scene.surfaces[surf];
    // Front is the local +Z side; incident on the front when running against it.
    let from_front = w.d.dot(normal) < 0.0;
    let resp = if from_front { &s.front } else { &s.back };
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
                flags: w.flags & !SEG_TIR, // TIR is per-interaction, not inherited
            },
            max_depth,
            min_flux,
        );
    };
    let reflect_branch = |stack: &mut Vec<Work>, flux: f32, ghost: bool| {
        let dir = reflect(w.d, n_oppose);
        let mut flags = w.flags & !SEG_TIR; // specular reflection is not TIR
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
            // A table with no value at this wavelength ends the branch (tracer3
            // names it as unsupported physics).
            let Some(r) = r.sample(lambda) else {
                return;
            };
            reflect_branch(stack, flux_end * r.clamp(0.0, 1.0), false);
        }
        GpuResponse::Dielectric { ar } => {
            // Same medium both sides: Fresnel at n1 = n2 is ~0, so the transmitted
            // ray is the whole beam undeviated; the AR-scaled sliver reflects.
            let cos_i = (-n_oppose.dot(w.d)).clamp(0.0, 1.0);
            let ar = ar_at(ar, lambda);
            let r = (reflectance(cos_i, w.medium_n, w.medium_n) * ar).clamp(0.0, 1.0);
            transmit(stack, flux_end * (1.0 - r));
            if scene.ghosts && w.gdepth < scene.max_ghost && r > 0.0 {
                reflect_branch(stack, flux_end * r, true);
            }
        }
        GpuResponse::ExplicitRt { r, t } => {
            // A beam splitter or dichroic: both branches are primary paths, as in
            // tracer3. Each keeps the parent's ghost state and ghost depth and
            // is followed whether or not ghosts are on. Either scalar missing at
            // this wavelength ends the branch.
            let (Some(r), Some(t)) = (r.sample(lambda), t.sample(lambda)) else {
                return;
            };
            let (r, t) = (r.clamp(0.0, 1.0), t.clamp(0.0, 1.0));
            // Reflected first, like tracer3, so both pop in the same order.
            if r > 0.0 {
                reflect_branch(stack, flux_end * r, false);
            }
            if t > 0.0 {
                transmit(stack, flux_end * t);
            }
        }
        GpuResponse::CoatingGrid { lambdas, rt } => {
            // A real thin-film stack, read off the grid that was baked from the
            // transfer matrix when the scene compiled. Both branches are primary
            // paths, exactly as for `ExplicitRt`; the only difference is that
            // `R` and `T` now depend on the angle the ray arrived at.
            let cos_i = (-n_oppose.dot(w.d)).clamp(0.0, 1.0);
            let (r, t) = crate::scene3::sample_coating_grid(lambdas, rt, lambda, cos_i);
            if r > 0.0 {
                reflect_branch(stack, flux_end * r, false);
            }
            if t > 0.0 {
                transmit(stack, flux_end * t);
            }
        }
        GpuResponse::ThinLens { f } => {
            // Ideal-lens slope law u' = u − h/f in the surface's local frame
            // (direction-aware; mirrors tracer3's f64 branch). Full
            // transmission, no reflection, no absorption.
            let f = *f;
            let rt = s.rot.transpose();
            let pl = rt * (point - s.pos);
            let dl = rt * w.d;
            let dir = if dl.z.abs() < 1e-6 || f == 0.0 {
                w.d
            } else {
                s.rot
                    * Vec3::new(
                        dl.x - dl.z.abs() * pl.x / f,
                        dl.y - dl.z.abs() * pl.y / f,
                        dl.z,
                    )
                    .normalize()
            };
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
                    flags: w.flags & !SEG_TIR,
                },
                max_depth,
                min_flux,
            );
        }
    }
}

/// The AR residual at `lambda`, clamped to `[0, 1]`. A table with no value there
/// leaves the Fresnel reflection unscaled (`1`), as tracer3 does.
fn ar_at(ar: &Spectral32, lambda: f32) -> f32 {
    ar.sample(lambda).unwrap_or(1.0).clamp(0.0, 1.0)
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
        OBJ_CYLINDER => cylinder_hit(ol, dl, b.params.x, b.params.y, EPS),
        OBJ_EXTRUDE => extrude_hit(b.extrude.as_ref()?, ol, dl, EPS),
        OBJ_REVOLVE => revolve_hit(b.revolve.as_ref()?, ol, dl, EPS),
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
        OBJ_CYLINDER => {
            pl.x * pl.x + pl.y * pl.y <= b.params.x * b.params.x && pl.z.abs() <= b.params.y
        }
        OBJ_EXTRUDE => b
            .extrude
            .as_ref()
            .map(|e| extrude_contains(e, pl))
            .unwrap_or(false),
        OBJ_REVOLVE => b
            .revolve
            .as_ref()
            .map(|r| revolve_contains(r, pl))
            .unwrap_or(false),
        _ => false,
    }
}

/// Is a local point inside the revolved body? Map to `(x, r)` and test the
/// sampled meridian contour.
fn revolve_contains(rev: &GpuRevolve, p: Vec3) -> bool {
    let r = (p.y * p.y + p.z * p.z).sqrt();
    point_in_polygon(&rev.contour, Vec2::new(p.x, r))
}

/// Nearest revolved-profile intersection at `t > tmin`, with the outward local
/// normal. Each face is exact; the normal is oriented by a containment probe,
/// because a single surface cannot say which side is glass.
fn revolve_hit(rev: &GpuRevolve, o: Vec3, d: Vec3, tmin: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    let keep = |best: &mut Option<(f32, Vec3)>, t: f32, n: Vec3| {
        if t > tmin && best.map(|(bt, _)| t < bt).unwrap_or(true) {
            *best = Some((t, orient_rev(rev, n, o + d * t)));
        }
    };
    for face in &rev.faces {
        match face {
            RevFace::Sag {
                vertex_x,
                axis_sign,
                c,
                k,
                coeffs,
                outer_r,
                is_conic,
            } => sag_hit(
                *vertex_x,
                *axis_sign,
                *c,
                *k,
                coeffs,
                *outer_r,
                *is_conic,
                o,
                d,
                tmin,
                &mut |t, n| keep(&mut best, t, n),
            ),
            RevFace::Frustum { x0, y0, x1, y1 } => {
                frustum_hit(*x0, *y0, *x1, *y1, o, d, tmin, &mut |t, n| {
                    keep(&mut best, t, n)
                })
            }
            RevFace::Arc {
                center_x,
                r,
                a0,
                a1,
            } => arc_hit(*center_x, *r, *a0, *a1, o, d, tmin, &mut |t, n| {
                keep(&mut best, t, n)
            }),
        }
    }
    best
}

fn orient_rev(rev: &GpuRevolve, n: Vec3, p: Vec3) -> Vec3 {
    if revolve_contains(rev, p + n * EPS) {
        -n
    } else {
        n
    }
}

/// The outward normal of a conic/asphere of revolution at `p` (analytic, §7.4).
/// `vertex` is on the axis; radial is the component perpendicular to `+X`.
fn sag_normal(vertex_x: f32, axis_sign: f32, c: f32, k: f32, coeffs: &[f32; 8], p: Vec3) -> Vec3 {
    let rad = Vec3::new(0.0, p.y, p.z);
    let r = rad.length();
    let axis = Vec3::new(axis_sign, 0.0, 0.0);
    let n = if r > 0.0 {
        axis - rad / r * even_asphere_dsag(c, k, coeffs, r)
    } else {
        axis
    };
    let _ = vertex_x;
    n.normalize()
}

/// Intersect a conic (exact quadratic) or even-asphere (safeguarded Newton)
/// surface of revolution about `+X`, reporting each aperture-clipped hit.
#[allow(clippy::too_many_arguments)]
fn sag_hit(
    vertex_x: f32,
    axis_sign: f32,
    c: f32,
    k: f32,
    coeffs: &[f32; 8],
    outer_r: f32,
    is_conic: bool,
    o: Vec3,
    d: Vec3,
    tmin: f32,
    out: &mut dyn FnMut(f32, Vec3),
) {
    let ov = o - Vec3::new(vertex_x, 0.0, 0.0);
    let z0 = ov.x * axis_sign;
    let dz = d.x * axis_sign;
    let o_rad = Vec3::new(0.0, ov.y, ov.z);
    let d_rad = Vec3::new(0.0, d.y, d.z);

    // The sag envelope: keep the near nappe out to the aperture; reject the far
    // side of the whole revolution.
    let sag_edge = even_asphere_sag(c, k, coeffs, outer_r);
    let z_lo = sag_edge.min(0.0);
    let z_hi = sag_edge.max(0.0);
    let z_tol = 1e-4 * (1.0 + outer_r.abs());
    let r_tol = 1e-5 * (1.0 + outer_r);

    let accept = |t: f32, out: &mut dyn FnMut(f32, Vec3)| {
        if t <= tmin {
            return;
        }
        let p = o + d * t;
        let r = (p.y * p.y + p.z * p.z).sqrt();
        let z = (p - Vec3::new(vertex_x, 0.0, 0.0)).x * axis_sign;
        if r <= outer_r + r_tol && z >= z_lo - z_tol && z <= z_hi + z_tol {
            out(t, sag_normal(vertex_x, axis_sign, c, k, coeffs, p));
        }
    };

    if is_conic {
        let k1 = 1.0 + k;
        let aa = c * (d_rad.length_squared() + k1 * dz * dz);
        let bb = c * (2.0 * o_rad.dot(d_rad) + 2.0 * k1 * z0 * dz) - 2.0 * dz;
        let cc = c * (o_rad.length_squared() + k1 * z0 * z0) - 2.0 * z0;
        for t in quadratic_roots(aa, bb, cc) {
            accept(t, out);
        }
        return;
    }

    // Even asphere: seed from the base conic, then Newton-refine the residual
    // axial(t) − sag(r(t)) with the analytic derivative (§7.4).
    let r_of = |t: f32| (o_rad + d_rad * t).length();
    let residual = |t: f32| (z0 + t * dz) - even_asphere_sag(c, k, coeffs, r_of(t));
    let dresidual = |t: f32| {
        let rad = o_rad + d_rad * t;
        let r = rad.length();
        let dr = if r > 0.0 { rad.dot(d_rad) / r } else { 0.0 };
        dz - even_asphere_dsag(c, k, coeffs, r) * dr
    };

    let k1 = 1.0 + k;
    let aa = c * (d_rad.length_squared() + k1 * dz * dz);
    let bb = c * (2.0 * o_rad.dot(d_rad) + 2.0 * k1 * z0 * dz) - 2.0 * dz;
    let cc = c * (o_rad.length_squared() + k1 * z0 * z0) - 2.0 * z0;
    let mut seeds: Vec<f32> = quadratic_roots(aa, bb, cc).into_iter().collect();
    if seeds.is_empty() {
        // A coarse march over the aperture span, so a strongly aspheric surface
        // with no nearby conic root is still bracketed.
        let start = tmin.max(0.0);
        let step = outer_r.max(0.05) * 0.1;
        let mut pt = start;
        let mut pv = residual(pt);
        for i in 1..=200 {
            let t = start + step * i as f32;
            let v = residual(t);
            if pv * v < 0.0 {
                seeds.push(0.5 * (pt + t));
            }
            pt = t;
            pv = v;
        }
    }
    for seed in seeds {
        if let Some(t) = refine_root(seed, tmin, &residual, &dresidual) {
            accept(t, out);
        }
    }
}

/// Real roots of `aa·t² + bb·t + cc`, numerically stable. Handles the degenerate
/// linear case (a plane, or a ray along the axis).
fn quadratic_roots(aa: f32, bb: f32, cc: f32) -> Vec<f32> {
    if aa.abs() < 1e-20 {
        return if bb.abs() > 1e-20 {
            vec![-cc / bb]
        } else {
            Vec::new()
        };
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return Vec::new();
    }
    let sq = disc.sqrt();
    let q = -0.5 * (bb + bb.signum() * sq);
    let mut roots = vec![q / aa];
    if q.abs() > 1e-20 {
        roots.push(cc / q);
    }
    roots
}

/// Safeguarded Newton for `f(t) = 0` near `seed`, bisecting when a step leaves a
/// sign-change bracket (§7.4). f32 tolerances.
fn refine_root(
    seed: f32,
    tmin: f32,
    f: &dyn Fn(f32) -> f32,
    df: &dyn Fn(f32) -> f32,
) -> Option<f32> {
    let mut t = seed;
    let span = 1e-2_f32;
    let (mut lo, mut hi) = (seed - span, seed + span);
    let (flo, fhi) = (f(lo), f(hi));
    let bracketed = flo * fhi <= 0.0;
    let mut flo = flo;

    for _ in 0..48 {
        let ft = f(t);
        if ft.abs() < 1e-7 {
            break;
        }
        let dft = df(t);
        let mut nt = if dft.abs() > 1e-30 {
            t - ft / dft
        } else {
            f32::INFINITY
        };
        if !nt.is_finite() || (bracketed && (nt < lo || nt > hi)) {
            nt = 0.5 * (lo + hi);
        }
        if bracketed {
            let fnt = f(nt);
            if flo * fnt <= 0.0 {
                hi = nt;
            } else {
                lo = nt;
                flo = fnt;
            }
        }
        let step = (nt - t).abs();
        t = nt;
        if step < 1e-8 * (1.0 + t.abs()) {
            break;
        }
    }
    let _ = fhi;
    (f(t).abs() < 1e-5 && t > tmin).then_some(t)
}

/// A line segment `(x0,y0)–(x1,y1)` revolved about `+X`: a cone, cylinder, or flat
/// annular face. Ported from `oc_core::geom3::primitives::frustum_intersect`.
#[allow(clippy::too_many_arguments)]
fn frustum_hit(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    o: Vec3,
    d: Vec3,
    tmin: f32,
    out: &mut dyn FnMut(f32, Vec3),
) {
    let (x_lo, x_hi) = (x0.min(x1), x0.max(x1));
    let flat = (x1 - x0).abs() <= 1e-9 * (1.0 + x0.abs() + x1.abs());

    if flat {
        if d.x.abs() < 1e-12 {
            return;
        }
        let t = (x0 - o.x) / d.x;
        if t <= tmin {
            return;
        }
        let p = o + d * t;
        let r = (p.y * p.y + p.z * p.z).sqrt();
        let (r_lo, r_hi) = (y0.min(y1), y0.max(y1));
        let pad = 1e-9 * (1.0 + r_hi);
        if r >= r_lo - pad && r <= r_hi + pad {
            out(t, Vec3::X);
        }
        return;
    }

    let m = (y1 - y0) / (x1 - x0);
    let b = y0 - m * x0;
    let g0 = m * o.x + b;
    let aa = d.y * d.y + d.z * d.z - m * m * d.x * d.x;
    let bb = 2.0 * (o.y * d.y + o.z * d.z) - 2.0 * m * d.x * g0;
    let cc = o.y * o.y + o.z * o.z - g0 * g0;
    let pad = 1e-6 * (1.0 + x_hi.abs() + x_lo.abs());
    for t in quadratic_roots(aa, bb, cc) {
        if t <= tmin {
            continue;
        }
        let p = o + d * t;
        let r_line = m * p.x + b;
        if r_line >= -pad && p.x >= x_lo - pad && p.x <= x_hi + pad {
            out(t, Vec3::new(-m * r_line, p.y, p.z).normalize());
        }
    }
}

/// An axis-centred arc revolved about `+X`: a spherical zone.
#[allow(clippy::too_many_arguments)]
fn arc_hit(
    center_x: f32,
    r: f32,
    a0: f32,
    a1: f32,
    o: Vec3,
    d: Vec3,
    tmin: f32,
    out: &mut dyn FnMut(f32, Vec3),
) {
    let c3 = Vec3::new(center_x, 0.0, 0.0);
    let oc = o - c3;
    let b = oc.dot(d);
    let disc = b * b - (oc.length_squared() - r * r);
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    for t in [-b - sq, -b + sq] {
        if t <= tmin {
            continue;
        }
        let p = o + d * t;
        let rad = (p.y * p.y + p.z * p.z).sqrt();
        let ang = rad.atan2(p.x - center_x);
        if angle_in_sweep(ang, a0, a1) {
            out(t, (p - c3) / r);
        }
    }
}

/// Is `ang` within the CCW sweep from `a0` to `a1`? Mirrors the profile arc's
/// unwrapped-difference convention.
fn angle_in_sweep(ang: f32, a0: f32, a1: f32) -> bool {
    use std::f32::consts::TAU;
    let sweep = (a1 - a0).rem_euclid(TAU);
    if sweep >= TAU - 1e-6 {
        return true;
    }
    let rel = (ang - a0).rem_euclid(TAU);
    rel <= sweep + 1e-6
}

/// Is a local point inside the extruded body?
fn extrude_contains(e: &GpuExtrude, p: Vec3) -> bool {
    p.z >= e.z_min && p.z <= e.z_max && point_in_polygon(&e.verts, Vec2::new(p.x, p.y))
}

fn point_in_polygon(verts: &[Vec2], p: Vec2) -> bool {
    let mut inside = false;
    let n = verts.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (verts[i], verts[j]);
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if p.x < a.x + t * (b.x - a.x) {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Nearest finite-cylinder intersection at `t > tmin` (canonical `+Z` axis), with
/// the outward local normal. Side wall plus the two end caps.
fn cylinder_hit(o: Vec3, d: Vec3, r: f32, hl: f32, tmin: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    let keep = |best: &mut Option<(f32, Vec3)>, t: f32, n: Vec3| {
        if t > tmin && best.map(|(bt, _)| t < bt).unwrap_or(true) {
            *best = Some((t, n));
        }
    };

    let a = d.x * d.x + d.y * d.y;
    if a > 1e-20 {
        let b = 2.0 * (o.x * d.x + o.y * d.y);
        let c = o.x * o.x + o.y * o.y - r * r;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            for t in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                let z = o.z + t * d.z;
                if z.abs() <= hl {
                    let p = o + d * t;
                    keep(&mut best, t, Vec3::new(p.x, p.y, 0.0) / r);
                }
            }
        }
    }
    if d.z.abs() > 1e-20 {
        for (zc, nz) in [(hl, 1.0_f32), (-hl, -1.0_f32)] {
            let t = (zc - o.z) / d.z;
            let p = o + d * t;
            if p.x * p.x + p.y * p.y <= r * r {
                keep(&mut best, t, Vec3::new(0.0, 0.0, nz));
            }
        }
    }
    best
}

/// Nearest extruded-profile intersection at `t > tmin`, with the outward local
/// normal: ruled side faces plus the two end caps.
fn extrude_hit(e: &GpuExtrude, o: Vec3, d: Vec3, tmin: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    let keep = |best: &mut Option<(f32, Vec3)>, t: f32, n: Vec3| {
        if t > tmin && best.map(|(bt, _)| t < bt).unwrap_or(true) {
            *best = Some((t, n));
        }
    };

    let n = e.verts.len();
    let o2 = Vec2::new(o.x, o.y);
    let d2 = Vec2::new(d.x, d.y);
    if d2.length() > 1e-12 {
        for i in 0..n {
            let (a, b) = (e.verts[i], e.verts[(i + 1) % n]);
            if let Some((t, n2)) = segment_hit(a, b, o2, d2, tmin) {
                let z = o.z + t * d.z;
                if z >= e.z_min && z <= e.z_max {
                    let p = o + d * t;
                    let nrm = orient_out(e, Vec3::new(n2.x, n2.y, 0.0), p);
                    keep(&mut best, t, nrm);
                }
            }
        }
    }
    if d.z.abs() > 1e-12 {
        for (z, nz) in [(e.z_min, -1.0_f32), (e.z_max, 1.0_f32)] {
            let t = (z - o.z) / d.z;
            if t > tmin {
                let p = o + d * t;
                if point_in_polygon(&e.verts, Vec2::new(p.x, p.y)) {
                    keep(&mut best, t, Vec3::new(0.0, 0.0, nz));
                }
            }
        }
    }
    best
}

/// Orient a side-face normal outward: away from the body's interior.
fn orient_out(e: &GpuExtrude, n: Vec3, p: Vec3) -> Vec3 {
    if extrude_contains(e, p + n * EPS) {
        -n
    } else {
        n
    }
}

/// Intersect the 2D ray `o + t·d` (`d` need not be unit) with segment `a–b`.
/// Returns `(t, unit edge normal)` when `t > tmin` and the hit is on the segment.
fn segment_hit(a: Vec2, b: Vec2, o: Vec2, d: Vec2, tmin: f32) -> Option<(f32, Vec2)> {
    let edge = b - a;
    let denom = cross2(d, edge);
    if denom.abs() < 1e-20 {
        return None;
    }
    let ao = a - o;
    let t = cross2(ao, edge) / denom;
    let u = cross2(ao, d) / denom;
    if t <= tmin || !(0.0..=1.0).contains(&u) {
        return None;
    }
    Some((t, Vec2::new(edge.y, -edge.x).normalize()))
}

#[inline]
fn cross2(p: Vec2, q: Vec2) -> f32 {
    p.x * q.y - p.y * q.x
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

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::scene3::OutOfRangePolicy;

    #[test]
    fn a_missing_ar_residual_leaves_fresnel_unscaled() {
        // tracer3: `sample(ar_residual).unwrap_or(1.0).clamp(0.0, 1.0)`.
        let table = |policy| Spectral32::Table {
            lambda: vec![0.5, 0.6],
            values: vec![0.2, 0.4],
            policy,
        };
        assert_eq!(ar_at(&table(OutOfRangePolicy::Error), 0.7), 1.0);
        assert_eq!(ar_at(&table(OutOfRangePolicy::Zero), 0.7), 0.0);
        assert_eq!(ar_at(&table(OutOfRangePolicy::Clamp), 0.7), 0.4);
        assert!((ar_at(&table(OutOfRangePolicy::Error), 0.55) - 0.3).abs() < 1e-6);
        assert_eq!(ar_at(&Spectral32::Const(1.5), 0.55), 1.0);
    }
}
