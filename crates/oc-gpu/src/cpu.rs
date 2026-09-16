//! CPU f32 interactive tracer — a line-for-line mirror of `shader.wgsl`.
//!
//! The no-WebGPU fallback (§4.3) and the oracle the GPU path is validated
//! against. M2: spectral `n(λ)`, stack-based non-sequential trace with ghosts,
//! and TIR flagging.

use crate::scene::{
    GpuScene, GpuSeg, DISP_CAUCHY, DISP_SELLMEIER, KIND_LINE, MAT_ABSORBER, MAT_DIELECTRIC,
    MAT_MIRROR, SEG_GHOST, SEG_TIR, SEG_VALID,
};
use glam::Vec2;

// Non-axis-aligned containment ray direction (1, 1/π). Must match the literal
// in `shader.wgsl` so CPU and GPU count boundary crossings identically.
const CONTAIN_DIR: [f32; 2] = [1.0, std::f32::consts::FRAC_1_PI];
const EPS_O: f32 = 1e-5;
const EPS_M: f32 = 1e-4;
const STACK: usize = 48;

fn perp(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}
fn perp_dot(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn line_hit(a: Vec2, b: Vec2, o: Vec2, d: Vec2, tmin: f32) -> f32 {
    let e = b - a;
    let det = perp_dot(e, d);
    if det.abs() < 1e-12 {
        return -1.0;
    }
    let rhs = a - o;
    let t = perp_dot(e, rhs) / det;
    let u = perp_dot(d, rhs) / det;
    if t > tmin && (0.0..=1.0).contains(&u) {
        t
    } else {
        -1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn conic_accept(
    t: f32,
    tmin: f32,
    y0: f32,
    dy: f32,
    z0: f32,
    dz: f32,
    ha: f32,
    z_lo: f32,
    z_hi: f32,
    ztol: f32,
) -> bool {
    if t <= tmin {
        return false;
    }
    let y = y0 + t * dy;
    let z = z0 + t * dz;
    y.abs() <= ha + 1e-5 && z >= z_lo - ztol && z <= z_hi + ztol
}

struct Conic {
    vertex: Vec2,
    axis: Vec2,
    c: f32,
    k: f32,
    ha: f32,
}

impl Conic {
    fn from_prim(p: &[f32; 8]) -> Self {
        Conic {
            vertex: Vec2::new(p[0], p[1]),
            axis: Vec2::new(p[2], p[3]).normalize(),
            c: p[4],
            k: p[5],
            ha: p[6],
        }
    }

    fn coeffs(&self, o: Vec2, d: Vec2) -> ([f32; 3], [f32; 4], [f32; 3]) {
        let pp = perp(self.axis);
        let oo = o - self.vertex;
        let y0 = oo.dot(pp);
        let z0 = oo.dot(self.axis);
        let dy = d.dot(pp);
        let dz = d.dot(self.axis);
        let k1 = 1.0 + self.k;
        let aa = self.c * dy * dy + self.c * k1 * dz * dz;
        let bb = 2.0 * self.c * y0 * dy + 2.0 * self.c * k1 * z0 * dz - 2.0 * dz;
        let cc = self.c * y0 * y0 + self.c * k1 * z0 * z0 - 2.0 * z0;
        let disc_e = 1.0 - k1 * self.c * self.c * self.ha * self.ha;
        let sag_edge = self.c * self.ha * self.ha / (1.0 + disc_e.max(0.0).sqrt());
        let z_lo = sag_edge.min(0.0);
        let z_hi = sag_edge.max(0.0);
        let ztol = 1e-4 * (1.0 + self.ha.abs());
        ([aa, bb, cc], [y0, z0, dy, dz], [z_lo, z_hi, ztol])
    }

    fn hit(&self, o: Vec2, d: Vec2, tmin: f32) -> f32 {
        let ([aa, bb, cc], [y0, z0, dy, dz], [z_lo, z_hi, ztol]) = self.coeffs(o, d);
        let acc = |t: f32| conic_accept(t, tmin, y0, dy, z0, dz, self.ha, z_lo, z_hi, ztol);
        let mut best = -1.0f32;
        if aa.abs() < 1e-12 {
            if bb.abs() > 1e-12 {
                let t = -cc / bb;
                if acc(t) {
                    best = t;
                }
            }
        } else {
            let disc = bb * bb - 4.0 * aa * cc;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let sgn = if bb >= 0.0 { 1.0 } else { -1.0 };
                let q = -0.5 * (bb + sgn * sq);
                let t1 = q / aa;
                if acc(t1) {
                    best = t1;
                }
                if q.abs() > 1e-20 {
                    let t2 = cc / q;
                    if acc(t2) && (best < 0.0 || t2 < best) {
                        best = t2;
                    }
                }
            }
        }
        best
    }

    fn count(&self, o: Vec2, d: Vec2) -> u32 {
        let ([aa, bb, cc], [y0, z0, dy, dz], [z_lo, z_hi, ztol]) = self.coeffs(o, d);
        let acc = |t: f32| conic_accept(t, 1e-7, y0, dy, z0, dz, self.ha, z_lo, z_hi, ztol);
        let mut n = 0u32;
        if aa.abs() < 1e-12 {
            if bb.abs() > 1e-12 && acc(-cc / bb) {
                n += 1;
            }
        } else {
            let disc = bb * bb - 4.0 * aa * cc;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let sgn = if bb >= 0.0 { 1.0 } else { -1.0 };
                let q = -0.5 * (bb + sgn * sq);
                if acc(q / aa) {
                    n += 1;
                }
                if q.abs() > 1e-20 && acc(cc / q) {
                    n += 1;
                }
            }
        }
        n
    }

    fn normal(&self, pt: Vec2) -> Vec2 {
        let pp = perp(self.axis);
        let oo = pt - self.vertex;
        let y = oo.dot(pp);
        let z = oo.dot(self.axis);
        let k1 = 1.0 + self.k;
        ((2.0 * self.c * y) * pp + (2.0 * self.c * k1 * z - 2.0) * self.axis).normalize()
    }
}

fn prim_hit(scene: &GpuScene, pi: usize, o: Vec2, d: Vec2, tmin: f32) -> f32 {
    let pr = &scene.prims[pi];
    if pr.kind == KIND_LINE {
        line_hit(
            Vec2::new(pr.p[0], pr.p[1]),
            Vec2::new(pr.p[2], pr.p[3]),
            o,
            d,
            tmin,
        )
    } else {
        Conic::from_prim(&pr.p).hit(o, d, tmin)
    }
}

fn prim_normal(scene: &GpuScene, pi: usize, pt: Vec2) -> Vec2 {
    let pr = &scene.prims[pi];
    if pr.kind == KIND_LINE {
        let e = Vec2::new(pr.p[2] - pr.p[0], pr.p[3] - pr.p[1]);
        perp(e).normalize()
    } else {
        Conic::from_prim(&pr.p).normal(pt)
    }
}

fn cross_count(scene: &GpuScene, e: usize, pt: Vec2) -> u32 {
    let dir = Vec2::new(CONTAIN_DIR[0], CONTAIN_DIR[1]).normalize();
    let el = &scene.elems[e];
    let mut count = 0u32;
    for pi in el.edge_start as usize..(el.edge_start + el.n_edges) as usize {
        let pr = &scene.prims[pi];
        if pr.kind == KIND_LINE {
            if line_hit(
                Vec2::new(pr.p[0], pr.p[1]),
                Vec2::new(pr.p[2], pr.p[3]),
                pt,
                dir,
                1e-7,
            ) > 0.0
            {
                count += 1;
            }
        } else {
            count += Conic::from_prim(&pr.p).count(pt, dir);
        }
    }
    count
}

fn region_elem(scene: &GpuScene, pt: Vec2) -> i32 {
    let mut best = -1i32;
    let mut best_pri = i32::MIN + 1;
    for e in 0..scene.elems.len() {
        if cross_count(scene, e, pt) & 1 == 1 && (best < 0 || scene.elems[e].priority >= best_pri) {
            best = e as i32;
            best_pri = scene.elems[e].priority;
        }
    }
    best
}

fn element_index(scene: &GpuScene, e: i32, lambda: f32) -> f32 {
    if e < 0 {
        return scene.ambient;
    }
    let el = &scene.elems[e as usize];
    if el.kind != MAT_DIELECTRIC {
        return scene.ambient;
    }
    if el.disp == DISP_SELLMEIER {
        let l2 = lambda * lambda;
        let n2 = 1.0
            + el.dp[0] * l2 / (l2 - el.dp[3])
            + el.dp[1] * l2 / (l2 - el.dp[4])
            + el.dp[2] * l2 / (l2 - el.dp[5]);
        n2.sqrt()
    } else if el.disp == DISP_CAUCHY {
        el.dp[0] + el.dp[1] / (lambda * lambda)
    } else {
        el.index_const
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

fn refract(d: Vec2, n: Vec2, n1: f32, n2: f32) -> Option<Vec2> {
    let eta = n1 / n2;
    let cosi = -n.dot(d);
    let kk = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if kk < 0.0 {
        None
    } else {
        Some(eta * d + (eta * cosi - kk.sqrt()) * n)
    }
}

fn reflectv(d: Vec2, n: Vec2) -> Vec2 {
    d - 2.0 * d.dot(n) * n
}

#[derive(Clone, Copy)]
struct Work {
    o: Vec2,
    d: Vec2,
    flux: f32,
    /// Total interface interactions on this branch (capped by `max_depth`).
    depth: u32,
    /// Ghost reflection generations in this branch's ancestry (capped by
    /// `scene.max_ghost`). The primary/transmitted/TIR/mirror path keeps its
    /// parent's `gdepth`; only a Fresnel back-reflection increments it.
    gdepth: u32,
    /// Refractive index of the medium this ray travels in, carried on the ray
    /// (like a current-material id) instead of re-deriving it by a full-scene
    /// region test at the top of every bounce. Must stay in lockstep with
    /// `shader.wgsl`.
    medium: f32,
}

/// Trace all launch rays on the CPU in f32; returns a `n_rays * max_segs`
/// segment array (same layout the GPU produces).
pub fn trace(scene: &GpuScene, max_depth: u32) -> Vec<GpuSeg> {
    let max_segs = scene.max_segs as usize;
    let mut out = vec![GpuSeg::zeroed(); scene.rays.len() * max_segs];

    for (rid, r) in scene.rays.iter().enumerate() {
        let base = rid * max_segs;
        let lambda = r.lambda;
        let min_flux = scene.flux_eps * r.flux;

        let mut stack = [Work {
            o: Vec2::ZERO,
            d: Vec2::X,
            flux: 0.0,
            depth: 0,
            gdepth: 0,
            medium: 0.0,
        }; STACK];
        let mut sp = 0usize;
        let push = |stack: &mut [Work; STACK], sp: &mut usize, w: Work| {
            if w.depth >= max_depth || w.flux < min_flux {
                return;
            }
            if *sp < STACK {
                stack[*sp] = w;
                *sp += 1;
            }
        };
        let launch_o = Vec2::new(r.ox, r.oy);
        let launch_d = Vec2::new(r.dx, r.dy).normalize();
        // Only the launch ray needs a from-scratch medium lookup.
        let launch_medium = element_index(
            scene,
            region_elem(scene, launch_o + EPS_M * launch_d),
            lambda,
        );
        push(
            &mut stack,
            &mut sp,
            Work {
                o: launch_o,
                d: launch_d,
                flux: r.flux,
                depth: 0,
                gdepth: 0,
                medium: launch_medium,
            },
        );

        let mut seg_count = 0usize;
        while sp > 0 && seg_count < max_segs {
            sp -= 1;
            let w = stack[sp];
            if w.depth >= max_depth || w.flux < min_flux {
                continue;
            }
            let (o, d) = (w.o, w.d);
            let n_here = w.medium;

            let mut best_t = scene.bounds;
            let mut best_pi = usize::MAX;
            let mut best_elem = -1i32;
            for e in 0..scene.elems.len() {
                let el = &scene.elems[e];
                for pi in el.edge_start as usize..(el.edge_start + el.n_edges) as usize {
                    let t = prim_hit(scene, pi, o, d, EPS_O);
                    if t > 0.0 && t < best_t {
                        best_t = t;
                        best_pi = pi;
                        best_elem = e as i32;
                    }
                }
            }

            let gflag = if w.gdepth > 0 { SEG_GHOST } else { 0 };

            if best_elem < 0 {
                let b = o + best_t * d;
                out[base + seg_count] = seg(o, b, w.flux, lambda, SEG_VALID | gflag);
                seg_count += 1;
                continue;
            }

            let hit = o + best_t * d;
            let si = base + seg_count;
            out[si] = seg(o, hit, w.flux, lambda, SEG_VALID | gflag);
            seg_count += 1;

            let mut nrm = prim_normal(scene, best_pi, hit);
            if nrm.dot(d) > 0.0 {
                nrm = -nrm;
            }

            let far = region_elem(scene, hit - EPS_M * nrm);
            let hit_el = scene.elems[best_elem as usize];

            let far_absorb = far >= 0 && scene.elems[far as usize].kind == MAT_ABSORBER;
            if hit_el.kind == MAT_ABSORBER || far_absorb {
                continue;
            }

            let mut mirror = -1.0f32;
            if hit_el.kind == MAT_MIRROR {
                mirror = hit_el.reflectivity;
            } else if far >= 0 && scene.elems[far as usize].kind == MAT_MIRROR {
                mirror = scene.elems[far as usize].reflectivity;
            }

            if mirror >= 0.0 {
                let dr = reflectv(d, nrm);
                // Reflection stays in the current medium.
                push(
                    &mut stack,
                    &mut sp,
                    Work {
                        o: hit + EPS_O * dr,
                        d: dr,
                        flux: w.flux * mirror,
                        depth: w.depth + 1,
                        gdepth: w.gdepth,
                        medium: w.medium,
                    },
                );
                continue;
            }

            let n1 = n_here;
            let n2 = element_index(scene, far, lambda);
            let dr = reflectv(d, nrm);
            match refract(d, nrm, n1, n2) {
                None => {
                    // Total internal reflection: reflected ray stays in n_here.
                    out[si].flags |= SEG_TIR;
                    push(
                        &mut stack,
                        &mut sp,
                        Work {
                            o: hit + EPS_O * dr,
                            d: dr,
                            flux: w.flux,
                            depth: w.depth + 1,
                            gdepth: w.gdepth,
                            medium: w.medium,
                        },
                    );
                }
                Some(dt) => {
                    let cosi = -nrm.dot(d);
                    let rr = reflectance(cosi, n1, n2).clamp(0.0, 1.0);
                    // Transmitted ray enters the far medium (n2).
                    push(
                        &mut stack,
                        &mut sp,
                        Work {
                            o: hit + EPS_O * dt,
                            d: dt,
                            flux: w.flux * (1.0 - rr),
                            depth: w.depth + 1,
                            gdepth: w.gdepth,
                            medium: n2,
                        },
                    );
                    if scene.ghosts && rr > 0.0 && w.gdepth < scene.max_ghost {
                        // Ghost (back-reflection) stays in n_here; one more
                        // reflection generation (§4.2).
                        push(
                            &mut stack,
                            &mut sp,
                            Work {
                                o: hit + EPS_O * dr,
                                d: dr,
                                flux: w.flux * rr,
                                depth: w.depth + 1,
                                gdepth: w.gdepth + 1,
                                medium: w.medium,
                            },
                        );
                    }
                }
            }
        }
    }
    out
}

fn seg(a: Vec2, b: Vec2, flux: f32, lambda: f32, flags: u32) -> GpuSeg {
    GpuSeg {
        ax: a.x,
        ay: a.y,
        bx: b.x,
        by: b.y,
        flux,
        lambda,
        flags,
        _pad: 0,
    }
}

use bytemuck::Zeroable;
