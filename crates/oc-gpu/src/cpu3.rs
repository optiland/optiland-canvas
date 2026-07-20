//! The f32 CPU 3D tracer: the oracle the WGSL shader is validated against (§12,
//! PR 3D-07).
//!
//! It mirrors the f64 reference ([`oc_trace::trace3`]) in miniature — nearest-hit
//! transport, media by containment plus priority, vector Snell with scalar
//! Fresnel and exact TIR, deterministic reflected/transmitted branches — but in
//! f32 and in the normalized frame of [`GpuScene3`], so it does exactly what the
//! shader will do and can stand in for it wherever no GPU is present.
//!
//! This first increment traces the analytic bodies the compiler admits — sphere
//! and box, dielectric or absorbing. It is the arithmetic core; apertures,
//! detectors, and the curved profile surfaces layer on in later increments, and
//! until they do the compiler rejects them by name rather than the tracer
//! guessing.

use glam::{DVec3, Vec3};

use crate::scene3::{GpuBody3, GpuScene3, OBJ_BOX, OBJ_SPHERE, SURF_ABSORBER};

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

/// Trace every launch ray, returning each one's segments in world millimetres.
pub fn trace(scene: &GpuScene3, max_depth: u32) -> Vec<Vec<SegOut3>> {
    scene
        .rays
        .iter()
        .map(|r| trace_one(scene, r.origin, r.dir, r.flux, r.lambda_um, max_depth))
        .collect()
}

fn trace_one(
    scene: &GpuScene3,
    o: Vec3,
    d: Vec3,
    flux: f32,
    lambda: f32,
    max_depth: u32,
) -> Vec<SegOut3> {
    let min_flux = scene.flux_eps * flux;
    let mut out = Vec::new();

    // The launch medium: whatever the origin sits in.
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

        let hit = nearest_hit(scene, w.o, w.d);
        let Some(h) = hit else {
            // Escape.
            let end = w.o + w.d * scene.bounds_norm;
            out.push(seg(scene, w, end, w.flux, lambda));
            continue;
        };

        let point = w.o + w.d * h.t;
        let flux_end = w.flux * beer_lambert(w.medium_alpha, scene.norm.world_len(h.t));
        out.push(seg(scene, w, point, flux_end, lambda));

        let body = &scene.bodies[h.body];
        if body.surface == SURF_ABSORBER {
            continue; // swallowed at the surface
        }

        // Orient the normal to oppose the incident ray, then read the media just
        // inside and just outside the interface (containment + priority).
        let n = if h.normal.dot(w.d) > 0.0 {
            -h.normal
        } else {
            h.normal
        };
        let (n1, _a1) = region_optics(scene, point + n * EPS, lambda);
        let (n2, a2) = region_optics(scene, point - n * EPS, lambda);

        match refract(w.d, n, n1, n2) {
            None => {
                // Total internal reflection: always followed, flagged.
                let dir = reflect(w.d, n);
                push(
                    &mut stack,
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
                // Transmitted branch: enters the far medium.
                push(
                    &mut stack,
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
                // Reflected (ghost) branch: stays in the near medium.
                if scene.ghosts && w.gdepth < scene.max_ghost && r > 0.0 {
                    let dir = reflect(w.d, n);
                    push(
                        &mut stack,
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
    out
}

fn push(stack: &mut Vec<Work>, w: Work, max_depth: u32, min_flux: f32) {
    if w.depth <= max_depth && w.flux >= min_flux {
        stack.push(w);
    }
}

fn seg(scene: &GpuScene3, w: Work, end_norm: Vec3, flux_end: f32, lambda: f32) -> SegOut3 {
    SegOut3 {
        a: scene.norm.world_point(w.o),
        b: scene.norm.world_point(end_norm),
        flux: flux_end,
        lambda_um: lambda,
        flags: w.flags,
    }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

struct Hit {
    t: f32,
    normal: Vec3, // world (outward)
    body: usize,
}

fn nearest_hit(scene: &GpuScene3, o: Vec3, d: Vec3) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for (bi, b) in scene.bodies.iter().enumerate() {
        // Into the body's local frame (rotation-only pullback plus translation).
        let ol = b.rot.transpose() * (o - b.pos);
        let dl = b.rot.transpose() * d;
        let local = match b.kind {
            OBJ_SPHERE => sphere_hit(ol, dl, b.params.x, EPS),
            OBJ_BOX => box_hit(ol, dl, b.params, EPS),
            _ => None,
        };
        if let Some((t, nl)) = local {
            if best.as_ref().map(|h| t < h.t).unwrap_or(true) {
                best = Some(Hit {
                    t,
                    normal: (b.rot * nl).normalize(),
                    body: bi,
                });
            }
        }
    }
    best
}

/// Nearest sphere intersection at `t > tmin`, with the outward local normal.
fn sphere_hit(o: Vec3, d: Vec3, r: f32, tmin: f32) -> Option<(f32, Vec3)> {
    let b = o.dot(d);
    let c = o.dot(o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = -b - s;
    let t1 = -b + s;
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
    let mut tmin_hit = f32::NEG_INFINITY;
    let mut tmax_hit = f32::INFINITY;
    let mut axis = 0usize;
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
        if t0 > tmin_hit {
            tmin_hit = t0;
            axis = i;
        }
        tmax_hit = tmax_hit.min(t1);
        if tmin_hit > tmax_hit {
            return None;
        }
    }
    // Nearest crossing beyond tmin: the entry face, or the exit face from inside.
    let (t, ax) = if tmin_hit > tmin {
        (tmin_hit, axis)
    } else if tmax_hit > tmin {
        // Exiting: find which axis the exit belongs to.
        let mut exit_axis = 0usize;
        let mut best = f32::INFINITY;
        for i in 0..3 {
            if d[i].abs() < 1e-20 {
                continue;
            }
            let inv = 1.0 / d[i];
            let t1 = ((half[i]) - o[i]) * inv;
            let t0 = (-half[i] - o[i]) * inv;
            let te = t0.max(t1);
            if te < best {
                best = te;
                exit_axis = i;
            }
        }
        (tmax_hit, exit_axis)
    } else {
        return None;
    };
    let p = o + d * t;
    let mut nl = Vec3::ZERO;
    nl[ax] = p[ax].signum();
    Some((t, nl))
}

/// A point's medium optics: the refractive index and attenuation of the
/// highest-priority body containing it, else the ambient (§9.4).
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
