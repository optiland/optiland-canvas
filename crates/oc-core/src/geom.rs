//! Ray/edge geometry: intersection and normals for every [`Edge`] variant.
//!
//! All intersection routines operate in the edge's **local** frame. The scene
//! layer (see [`crate::scene`]) transforms rays into an element's local frame
//! before calling these, so the math here is transform-free.
//!
//! Sign conventions:
//! - Ray direction is a unit vector, so the ray parameter `t` is a distance.
//! - Returned `normal` is a unit geometric normal. Its orientation is *not*
//!   guaranteed to face any particular side; the tracer flips it to oppose the
//!   incident direction. Medium identity is resolved by containment + priority
//!   (§4.2), never by normal orientation.

use crate::vec::{perp, Vec2};
use serde::{Deserialize, Serialize};

/// A ray: an origin and a **unit** direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec2,
    pub dir: Vec2,
}

impl Ray {
    /// Construct a ray, normalizing `dir`.
    pub fn new(origin: Vec2, dir: Vec2) -> Self {
        Self {
            origin,
            dir: dir.normalize(),
        }
    }

    #[inline]
    pub fn at(&self, t: f64) -> Vec2 {
        self.origin + t * self.dir
    }
}

/// A single ray/edge intersection in the edge's local frame.
#[derive(Debug, Clone, Copy)]
pub struct EdgeHit {
    /// Ray parameter (distance, since `dir` is unit).
    pub t: f64,
    /// Intersection point.
    pub point: Vec2,
    /// Unit geometric normal (orientation unspecified; caller re-orients).
    pub normal: Vec2,
}

/// Parameters for a conic-of-revolution cross-section (its meridional curve).
///
/// The implicit form in local coordinates `(y, z)` — `y` transverse to `axis`,
/// `z` along `axis`, both measured from `vertex` — is
///
/// ```text
///     c * (y² + (1 + k) z²) - 2 z = 0
/// ```
///
/// which is exact for planes (`c = 0`), spheres (`k = 0`), parabolas
/// (`k = -1`), ellipses (`-1 < k < 0`) and hyperbolas (`k < -1`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConicParams {
    /// Vertex position (local frame).
    pub vertex: Vec2,
    /// Axis direction (need not be unit; normalized on use).
    pub axis: Vec2,
    /// Curvature `c = 1/R` at the vertex. `c = 0` is a plane.
    pub c: f64,
    /// Conic constant `k`.
    pub k: f64,
    /// Half aperture: hits with `|y| > half_ap` are rejected.
    pub half_ap: f64,
}

/// A boundary edge. Element boundaries are closed CCW loops of these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Edge {
    Line {
        a: Vec2,
        b: Vec2,
    },
    Arc {
        center: Vec2,
        r: f64,
        /// Start/end angles (radians). Swept CCW from `a0` to `a1`.
        a0: f64,
        a1: f64,
    },
    Conic(ConicParams),
    /// Base conic plus even aspheric terms: `sag += Σ coeffs[i]·y^(2i+4)`.
    Asphere {
        base: ConicParams,
        coeffs: [f64; 8],
    },
    /// Freeform escape hatch. For M0 it is evaluated as a polyline through the
    /// control points; true B-spline evaluation is deferred to the fit work.
    Spline {
        ctrl: Vec<Vec2>,
        degree: u8,
    },
}

const EDGE_EPS: f64 = 1e-12;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec2,
    pub max: Vec2,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec2::new(f64::INFINITY, f64::INFINITY),
        max: Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
    };

    pub fn from_point(p: Vec2) -> Self {
        Aabb { min: p, max: p }
    }

    pub fn expand(&mut self, p: Vec2) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn union(mut self, other: &Aabb) -> Aabb {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self
    }

    /// Grow the box by `pad` in all directions.
    pub fn padded(&self, pad: f64) -> Aabb {
        Aabb {
            min: self.min - Vec2::splat(pad),
            max: self.max + Vec2::splat(pad),
        }
    }

    pub fn center(&self) -> Vec2 {
        0.5 * (self.min + self.max)
    }

    /// Slab test: does `ray` enter this box at any `t > tmin` and before `tmax`?
    pub fn intersects_ray(&self, ray: &Ray, tmin: f64, tmax: f64) -> bool {
        let inv = Vec2::new(1.0 / ray.dir.x, 1.0 / ray.dir.y);
        let t0 = (self.min - ray.origin) * inv;
        let t1 = (self.max - ray.origin) * inv;
        let tsmall = t0.min(t1);
        let tbig = t0.max(t1);
        let enter = tsmall.max_element().max(tmin);
        let exit = tbig.min_element().min(tmax);
        enter <= exit
    }
}

impl Edge {
    /// Conservative local-frame bounding box for this edge.
    pub fn local_aabb(&self) -> Aabb {
        let mut bb = Aabb::EMPTY;
        match self {
            Edge::Line { a, b } => {
                bb.expand(*a);
                bb.expand(*b);
            }
            Edge::Arc { center, r, .. } => {
                // Conservative: the full circle box.
                bb.expand(*center - Vec2::splat(*r));
                bb.expand(*center + Vec2::splat(*r));
            }
            Edge::Conic(cp) => sample_conic_bounds(cp, &[0.0; 8], &mut bb),
            Edge::Asphere { base, coeffs } => sample_conic_bounds(base, coeffs, &mut bb),
            Edge::Spline { ctrl, .. } => {
                for p in ctrl {
                    bb.expand(*p);
                }
            }
        }
        bb
    }
}

fn sample_conic_bounds(cp: &ConicParams, coeffs: &[f64; 8], bb: &mut Aabb) {
    let axis = cp.axis.normalize();
    let p = perp(axis);
    let n = 24;
    for i in 0..=n {
        let y = -cp.half_ap + 2.0 * cp.half_ap * (i as f64 / n as f64);
        let sag = asphere_sag(cp, coeffs, y);
        bb.expand(cp.vertex + y * p + sag * axis);
    }
}

impl Edge {
    /// Enumerate every intersection with `ray` at parameter `> tmin`, in local
    /// coordinates, invoking `push` for each. Avoids allocation.
    pub fn intersect<F: FnMut(EdgeHit)>(&self, ray: &Ray, tmin: f64, push: &mut F) {
        match self {
            Edge::Line { a, b } => line_intersect(*a, *b, ray, tmin, push),
            Edge::Arc { center, r, a0, a1 } => {
                arc_intersect(*center, *r, *a0, *a1, ray, tmin, push)
            }
            Edge::Conic(cp) => conic_intersect(cp, ray, tmin, push),
            Edge::Asphere { base, coeffs } => asphere_intersect(base, coeffs, ray, tmin, push),
            Edge::Spline { ctrl, .. } => spline_intersect(ctrl, ray, tmin, push),
        }
    }
}

fn line_intersect<F: FnMut(EdgeHit)>(a: Vec2, b: Vec2, ray: &Ray, tmin: f64, push: &mut F) {
    let e = b - a;
    let det = e.perp_dot(ray.dir);
    if det.abs() < EDGE_EPS {
        return; // parallel
    }
    let rhs = a - ray.origin;
    let t = e.perp_dot(rhs) / det;
    let u = ray.dir.perp_dot(rhs) / det;
    if t > tmin && (0.0..=1.0).contains(&u) {
        push(EdgeHit {
            t,
            point: ray.at(t),
            normal: perp(e).normalize(),
        });
    }
}

/// Is `ang` within the CCW sweep from `a0` to `a1`?
fn angle_in_sweep(ang: f64, a0: f64, a1: f64) -> bool {
    let tau = std::f64::consts::TAU;
    let norm = |x: f64| x.rem_euclid(tau);
    let span = norm(a1 - a0);
    let rel = norm(ang - a0);
    rel <= span + 1e-12
}

fn arc_intersect<F: FnMut(EdgeHit)>(
    center: Vec2,
    r: f64,
    a0: f64,
    a1: f64,
    ray: &Ray,
    tmin: f64,
    push: &mut F,
) {
    let f = ray.origin - center;
    let b = f.dot(ray.dir);
    let disc = b * b - (f.length_squared() - r * r);
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    for t in [-b - sq, -b + sq] {
        if t > tmin {
            let point = ray.at(t);
            let rel = point - center;
            if angle_in_sweep(rel.y.atan2(rel.x), a0, a1) {
                push(EdgeHit {
                    t,
                    point,
                    normal: rel / r,
                });
            }
        }
    }
}

/// Shared conic root-finder used by `Conic` and as the seed for `Asphere`.
fn conic_intersect<F: FnMut(EdgeHit)>(cp: &ConicParams, ray: &Ray, tmin: f64, push: &mut F) {
    let axis = cp.axis.normalize();
    let p = perp(axis);
    let o = ray.origin - cp.vertex;
    let (c, k) = (cp.c, cp.k);

    let y0 = o.dot(p);
    let z0 = o.dot(axis);
    let dy = ray.dir.dot(p);
    let dz = ray.dir.dot(axis);
    let k1 = 1.0 + k;

    let aa = c * dy * dy + c * k1 * dz * dz;
    let bb = 2.0 * c * y0 * dy + 2.0 * c * k1 * z0 * dz - 2.0 * dz;
    let cc = c * y0 * y0 + c * k1 * z0 * z0 - 2.0 * z0;

    // The implicit conic is the *whole* surface of revolution (both nappes of a
    // sphere/hyperboloid). The physical optical surface is only the cap near the
    // vertex, so bound the accepted sag `z` to the aperture's sag envelope —
    // otherwise a ray can "hit" the far side of a large sphere. (A parabola has
    // no far nappe, so this is a no-op for it; a plane collapses it to z ≈ 0.)
    let ha = cp.half_ap;
    let sag_edge = {
        let d = 1.0 - k1 * c * c * ha * ha;
        c * ha * ha / (1.0 + d.max(0.0).sqrt())
    };
    let z_lo = sag_edge.min(0.0);
    let z_hi = sag_edge.max(0.0);
    let z_tol = 1e-7 * (1.0 + ha.abs());

    let mut consider = |t: f64| {
        if t > tmin {
            let y = y0 + t * dy;
            let z = z0 + t * dz;
            if y.abs() <= ha + 1e-12 && z >= z_lo - z_tol && z <= z_hi + z_tol {
                // ∇F = (2 c y, 2 c (1+k) z − 2) in (p, axis) coordinates.
                let n = (2.0 * c * y) * p + (2.0 * c * k1 * z - 2.0) * axis;
                push(EdgeHit {
                    t,
                    point: ray.at(t),
                    normal: n.normalize(),
                });
            }
        }
    };

    if aa.abs() < EDGE_EPS {
        // Degenerate to linear (e.g. plane, or ray along the axis of a sphere).
        if bb.abs() > EDGE_EPS {
            consider(-cc / bb);
        }
        return;
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    // Numerically stable quadratic roots.
    let q = -0.5 * (bb + bb.signum() * sq);
    consider(q / aa);
    if q.abs() > EDGE_EPS {
        consider(cc / q);
    }
}

/// Sag of an aspheric profile at transverse coordinate `y` (local).
fn asphere_sag(base: &ConicParams, coeffs: &[f64; 8], y: f64) -> f64 {
    let c = base.c;
    let r2 = y * y;
    let denom = 1.0 + (1.0 - (1.0 + base.k) * c * c * r2).max(0.0).sqrt();
    let mut sag = c * r2 / denom;
    let mut ypow = r2 * r2; // y^4
    for &a in coeffs {
        sag += a * ypow;
        ypow *= r2;
    }
    sag
}

/// Asphere intersection: seed from the base conic roots, then Newton-refine the
/// residual `z(t) − sag(y(t))` along the ray. Reduces to the exact conic when
/// all `coeffs` are zero.
fn asphere_intersect<F: FnMut(EdgeHit)>(
    base: &ConicParams,
    coeffs: &[f64; 8],
    ray: &Ray,
    tmin: f64,
    push: &mut F,
) {
    let axis = base.axis.normalize();
    let p = perp(axis);
    let o = ray.origin - base.vertex;
    let (y0, z0) = (o.dot(p), o.dot(axis));
    let (dy, dz) = (ray.dir.dot(p), ray.dir.dot(axis));

    let residual = |t: f64| (z0 + t * dz) - asphere_sag(base, coeffs, y0 + t * dy);

    // Collect base-conic seeds.
    let mut seeds: Vec<f64> = Vec::new();
    conic_intersect(base, ray, tmin - 1.0, &mut |h| seeds.push(h.t));
    if seeds.is_empty() {
        // Fall back to a coarse march to bracket a root near the aperture.
        let step = base.half_ap.max(1.0) * 0.1;
        let mut prev_t = tmin.max(0.0);
        let mut prev = residual(prev_t);
        for i in 1..=200 {
            let t = tmin.max(0.0) + step * i as f64;
            let cur = residual(t);
            if prev * cur < 0.0 {
                seeds.push(0.5 * (prev_t + t));
            }
            prev_t = t;
            prev = cur;
        }
    }

    for mut t in seeds {
        // Newton with numerical derivative; a few iterations converge for the
        // smooth near-conic profiles this variant is meant for.
        for _ in 0..24 {
            let f = residual(t);
            let h = 1e-7 * (t.abs() + 1.0);
            let dfdt = (residual(t + h) - residual(t - h)) / (2.0 * h);
            if dfdt.abs() < 1e-18 {
                break;
            }
            let dt = f / dfdt;
            t -= dt;
            if dt.abs() < 1e-13 {
                break;
            }
        }
        if t > tmin && residual(t).abs() < 1e-9 {
            let y = y0 + t * dy;
            if y.abs() <= base.half_ap + 1e-12 {
                // Surface normal ∝ (−dsag/dy, 1) in (p, axis) coords.
                let hh = 1e-7 * (y.abs() + 1.0);
                let dsag = (asphere_sag(base, coeffs, y + hh) - asphere_sag(base, coeffs, y - hh))
                    / (2.0 * hh);
                let n = (-dsag) * p + axis;
                push(EdgeHit {
                    t,
                    point: ray.at(t),
                    normal: n.normalize(),
                });
            }
        }
    }
}

/// Polyline evaluation of a spline (M0 escape hatch).
fn spline_intersect<F: FnMut(EdgeHit)>(ctrl: &[Vec2], ray: &Ray, tmin: f64, push: &mut F) {
    for w in ctrl.windows(2) {
        line_intersect(w[0], w[1], ray, tmin, push);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(edge: &Edge, ray: &Ray) -> Vec<EdgeHit> {
        let mut v = Vec::new();
        edge.intersect(ray, 1e-9, &mut |h| v.push(h));
        v
    }

    #[test]
    fn line_hit_and_normal() {
        // Vertical segment x = 2, y in [-1, 1]. Ray along +x from origin.
        let edge = Edge::Line {
            a: Vec2::new(2.0, -1.0),
            b: Vec2::new(2.0, 1.0),
        };
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let hits = collect(&edge, &ray);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].t - 2.0).abs() < 1e-12);
        // Normal parallel to x-axis.
        assert!(hits[0].normal.x.abs() > 0.999);
    }

    #[test]
    fn conic_plane_equals_line() {
        // c = 0 conic through vertex (2,0) with axis +x is the plane x = 2.
        let cp = ConicParams {
            vertex: Vec2::new(2.0, 0.0),
            axis: Vec2::new(1.0, 0.0),
            c: 0.0,
            k: 0.0,
            half_ap: 5.0,
        };
        let edge = Edge::Conic(cp);
        let ray = Ray::new(Vec2::new(0.0, 0.3), Vec2::new(1.0, 0.0));
        let hits = collect(&edge, &ray);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].t - 2.0).abs() < 1e-12);
    }

    #[test]
    fn conic_sphere_matches_circle() {
        // Sphere: c = 1/R, k = 0, vertex at origin, axis +x. R = 4 => center at (4,0).
        let r = 4.0;
        let cp = ConicParams {
            vertex: Vec2::new(0.0, 0.0),
            axis: Vec2::new(1.0, 0.0),
            c: 1.0 / r,
            k: 0.0,
            half_ap: 3.9,
        };
        let edge = Edge::Conic(cp);
        // Ray along axis hits vertex at t=... origin at (-2,0) going +x hits x=0 (vertex).
        let ray = Ray::new(Vec2::new(-2.0, 0.0), Vec2::new(1.0, 0.0));
        let hits = collect(&edge, &ray);
        assert!(!hits.is_empty());
        let near = hits.iter().map(|h| h.t).fold(f64::INFINITY, f64::min);
        assert!((near - 2.0).abs() < 1e-10, "vertex hit at t=2, got {near}");
    }

    #[test]
    fn asphere_reduces_to_conic() {
        let base = ConicParams {
            vertex: Vec2::ZERO,
            axis: Vec2::new(1.0, 0.0),
            c: 1.0 / 4.0,
            k: 0.0,
            half_ap: 3.9,
        };
        let conic = Edge::Conic(base);
        let asph = Edge::Asphere {
            base,
            coeffs: [0.0; 8],
        };
        let ray = Ray::new(Vec2::new(-2.0, 0.5), Vec2::new(1.0, 0.0));
        let a: Vec<f64> = collect(&conic, &ray).iter().map(|h| h.t).collect();
        let b: Vec<f64> = collect(&asph, &ray).iter().map(|h| h.t).collect();
        assert!(!a.is_empty() && !b.is_empty());
        let amin = a.iter().cloned().fold(f64::INFINITY, f64::min);
        let bmin = b.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((amin - bmin).abs() < 1e-7, "conic {amin} vs asphere {bmin}");
    }

    #[test]
    fn arc_semicircle() {
        // Full-ish arc, radius 3 centered at origin, sweep 0..pi (upper half).
        let edge = Edge::Arc {
            center: Vec2::ZERO,
            r: 3.0,
            a0: 0.0,
            a1: std::f64::consts::PI,
        };
        // Ray from (0,-5) going +y crosses the upper semicircle at (0,3).
        let ray = Ray::new(Vec2::new(0.0, -5.0), Vec2::new(0.0, 1.0));
        let hits = collect(&edge, &ray);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].point - Vec2::new(0.0, 3.0)).length() < 1e-12);
    }
}
