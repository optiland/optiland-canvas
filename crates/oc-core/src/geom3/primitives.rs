//! Transform-free analytic primitives in a local frame.
//!
//! Each routine takes a ray already pulled into the shape's local frame (see
//! [`super::ray_to_local`]) and reports [`RawHit3`]s through a callback, exactly
//! as [`crate::geom`] does in 2D. Keeping the math transform-free is what lets
//! the body layer place a shape with a rigid transform and get an identical
//! answer to the untransformed shape with a transformed ray — the property the
//! transform-invariance tests exist to pin.
//!
//! Every normal here is **analytic**. The 2D asphere takes its normal by finite
//! difference, which is the known ~1e-9 floor; the 3D path derives `dsag/dr` in
//! closed form and must beat it (contract §7.4, the `geom3` suite doc).

use super::ToleranceContext;
use crate::scene3::PlanarShape2;
use crate::vec::Vec2;
use crate::vec3::{Ray3, Vec3};

/// A direction component below this is treated as parallel to the axis it
/// indexes. The direction is a unit vector, so this is a true angle floor rather
/// than a scene-scale quantity: `1e-12` is roughly a picoradian of tilt.
const PARALLEL_EPS: f64 = 1e-12;

/// One intersection in a primitive's local frame, before an object identity or a
/// world transform is attached.
///
/// The primitive knows *which face* of itself was hit but not its
/// [`PatchId`](crate::id::PatchId): a box has no authored patches, and a profile
/// body maps a face to an edge's patch at the layer that owns both. So the face
/// index travels here and the body layer turns it into a patch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawHit3 {
    pub t_mm: f64,
    pub point: Vec3,
    /// Unit, outward for a closed primitive, and not yet flipped to oppose the
    /// ray (§5.4).
    pub normal: Vec3,
    pub uv: Option<Vec2>,
    /// Which face of a multi-face primitive: `0` for a single surface, `0..6` for
    /// a box, side/`+cap`/`-cap` = `0`/`1`/`2` for a cylinder.
    pub face: u32,
}

/// Is `t` a forward hit inside the half-open interval the tracer asked for?
#[inline]
fn in_range(t: f64, tmin: f64, tmax: f64) -> bool {
    t > tmin && t <= tmax
}

// ---------------------------------------------------------------------------
// Planar shapes: a surface in the local plane z = 0, normal +Z
// ---------------------------------------------------------------------------

/// Is `(x, y)` inside a planar shape? The boundary counts as inside.
///
/// Shared by planar surfaces, box and extrusion caps, aperture openings, and
/// detector faces, so "what the shape contains" has exactly one definition.
pub fn planar_shape_contains(shape: &PlanarShape2, p: Vec2) -> bool {
    match shape {
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => p.x.abs() <= *half_width_mm && p.y.abs() <= *half_height_mm,
        PlanarShape2::Disc { radius_mm } => p.length_squared() <= radius_mm * radius_mm,
        PlanarShape2::Annulus { inner_mm, outer_mm } => {
            let r2 = p.length_squared();
            r2 >= inner_mm * inner_mm && r2 <= outer_mm * outer_mm
        }
        PlanarShape2::Polygon { vertices } => point_in_polygon(vertices, p),
    }
}

/// The area where a disc of radius `r` centred on the origin overlaps the
/// axis-aligned rectangle `[x0, x1] × [y0, y1]`, mm².
///
/// Exact, not sampled. Across `x` the disc is the chord `[-h(x), h(x)]` with
/// `h = √(r² − x²)`, and the rectangle clips it to `[y0, y1]`. Cutting `x`
/// wherever the arc crosses `y0` or `y1` leaves pieces on which each limit is
/// either a straight edge or the arc throughout, and `∫h dx` has the closed
/// form `½(x h + r² asin(x/r))`. So the answer is a sum of those, with no
/// quadrature and nothing assumed about the rectangle being small.
///
/// A detector needs this for the pixels at a round face's rim: the part of a
/// pixel on the face is what its flux was collected over.
pub fn disc_rect_overlap_area(r: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let (a, b) = (x0.max(-r), x1.min(r));
    if r.is_nan() || r <= 0.0 || b <= a || y1 <= y0 {
        return 0.0;
    }
    // Whole rectangle inside the disc, or none of it: no arc to integrate.
    let far_x = x0.abs().max(x1.abs());
    let far_y = y0.abs().max(y1.abs());
    if far_x * far_x + far_y * far_y <= r * r {
        return (x1 - x0) * (y1 - y0);
    }
    let near_x = if x0 > 0.0 { x0 } else if x1 < 0.0 { -x1 } else { 0.0 };
    let near_y = if y0 > 0.0 { y0 } else if y1 < 0.0 { -y1 } else { 0.0 };
    if near_x * near_x + near_y * near_y >= r * r {
        return 0.0;
    }

    let half_chord = |x: f64| (r * r - x * x).max(0.0).sqrt();
    let chord_integral = |x: f64| {
        let x = x.clamp(-r, r);
        0.5 * (x * half_chord(x) + r * r * (x / r).asin())
    };
    // A line the arc only touches (`|y| = r`) still needs its cut at `x = 0`:
    // without it the piece's middle lands on the tangent point, where the arc
    // and the edge agree, and the whole piece reads as the edge.
    let mut cuts = vec![a, b];
    for y in [y0, y1] {
        if y.abs() <= r {
            let xc = half_chord(y);
            cuts.extend([-xc, xc].into_iter().filter(|x| *x > a && *x < b));
        }
    }
    cuts.sort_by(f64::total_cmp);

    let mut area = 0.0;
    for w in cuts.windows(2) {
        let (p, q) = (w[0], w[1]);
        if q <= p {
            continue;
        }
        // Which limit is the arc holds across the whole piece, so the middle
        // decides it.
        let h = half_chord(0.5 * (p + q));
        let top_is_arc = h < y1;
        let bottom_is_arc = -h > y0;
        let top = if top_is_arc { h } else { y1 };
        let bottom = if bottom_is_arc { -h } else { y0 };
        if top <= bottom {
            continue;
        }
        let arc = chord_integral(q) - chord_integral(p);
        let len = q - p;
        let upper = if top_is_arc { arc } else { y1 * len };
        let lower = if bottom_is_arc { -arc } else { y0 * len };
        area += upper - lower;
    }
    area.max(0.0)
}

/// Even-odd point-in-polygon, robust to a horizontal-edge coincidence.
///
/// Counts crossings of a ray from `p` along `+x`. The `(vi.y > p.y) != (vj.y > p.y)`
/// guard is the standard trick that makes each edge belong to exactly one of its
/// endpoints, so a ray grazing a shared vertex is counted once rather than zero
/// or twice.
fn point_in_polygon(vertices: &[Vec2], p: Vec2) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let vi = vertices[i];
        let vj = vertices[j];
        if (vi.y > p.y) != (vj.y > p.y) {
            let x_cross = vi.x + (p.y - vi.y) / (vj.y - vi.y) * (vj.x - vi.x);
            if p.x < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Intersect the local plane `z = 0`, then test membership. Normal is `+Z`; the
/// `(u, v)` reported is the hit's `(x, y)`.
pub fn planar_shape_intersect(
    shape: &PlanarShape2,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    if ray.dir.z.abs() < PARALLEL_EPS {
        return; // in or parallel to the plane: no isolated crossing
    }
    let t = -ray.origin.z / ray.dir.z;
    if !in_range(t, tmin, tmax) {
        return;
    }
    let p = ray.at(t);
    let uv = Vec2::new(p.x, p.y);
    if planar_shape_contains(shape, uv) {
        out(RawHit3 {
            t_mm: t,
            point: p,
            normal: Vec3::Z,
            uv: Some(uv),
            face: 0,
        });
    }
}

// ---------------------------------------------------------------------------
// Sphere, centred at the local origin
// ---------------------------------------------------------------------------

pub fn sphere_intersect(
    radius_mm: f64,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    // |o + t d|² = r², with |d| = 1: t² + 2(o·d) t + (|o|² - r²) = 0.
    let b = ray.origin.dot(ray.dir);
    let c = ray.origin.length_squared() - radius_mm * radius_mm;
    let disc = b * b - c;
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    for t in [-b - sq, -b + sq] {
        if in_range(t, tmin, tmax) {
            let p = ray.at(t);
            out(RawHit3 {
                t_mm: t,
                point: p,
                normal: p / radius_mm, // outward, already unit
                uv: None,
                face: 0,
            });
        }
    }
}

pub fn sphere_contains(radius_mm: f64, p: Vec3, tol: &ToleranceContext) -> super::Containment {
    classify(p.length() - radius_mm, tol)
}

// ---------------------------------------------------------------------------
// Axis-aligned box, centred at the local origin
// ---------------------------------------------------------------------------

pub fn box_intersect(
    half_extents_mm: Vec3,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    let h = half_extents_mm;
    let mut t_enter = f64::NEG_INFINITY;
    let mut enter_axis = 0usize;
    let mut enter_sign = 0.0;
    let mut t_exit = f64::INFINITY;
    let mut exit_axis = 0usize;
    let mut exit_sign = 0.0;

    for i in 0..3 {
        let o = ray.origin[i];
        let d = ray.dir[i];
        if d.abs() < PARALLEL_EPS {
            // Parallel to this slab: a miss unless the origin is already between
            // its faces. No hit is contributed either way.
            if o < -h[i] || o > h[i] {
                return;
            }
            continue;
        }
        let inv = 1.0 / d;
        // `-h` face first (outward normal `-e_i`), then `+h` (outward `+e_i`).
        let (mut t_near, mut s_near) = ((-h[i] - o) * inv, -1.0);
        let (mut t_far, mut s_far) = ((h[i] - o) * inv, 1.0);
        if t_near > t_far {
            std::mem::swap(&mut t_near, &mut t_far);
            std::mem::swap(&mut s_near, &mut s_far);
        }
        // Strict `>` keeps ties on an edge or corner to the first axis, so a ray
        // through a corner reports one entry hit, not three coincident ones.
        if t_near > t_enter {
            t_enter = t_near;
            enter_axis = i;
            enter_sign = s_near;
        }
        if t_far < t_exit {
            t_exit = t_far;
            exit_axis = i;
            exit_sign = s_far;
        }
        if t_enter > t_exit {
            return;
        }
    }

    for (t, axis, sign) in [
        (t_enter, enter_axis, enter_sign),
        (t_exit, exit_axis, exit_sign),
    ] {
        if in_range(t, tmin, tmax) {
            let mut n = Vec3::ZERO;
            n[axis] = sign;
            out(RawHit3 {
                t_mm: t,
                point: ray.at(t),
                normal: n,
                uv: None,
                face: (axis as u32) * 2 + if sign > 0.0 { 0 } else { 1 },
            });
        }
    }
}

pub fn box_contains(half_extents_mm: Vec3, p: Vec3, tol: &ToleranceContext) -> super::Containment {
    // Signed distance to the box surface: negative inside, positive outside. The
    // usual `max(|p| - h)` is exact outside and on the faces, and a good-enough
    // inside estimate for a boundary test.
    let q = p.abs() - half_extents_mm;
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.max_element().min(0.0);
    classify(outside + inside, tol)
}

// ---------------------------------------------------------------------------
// Finite cylinder about an arbitrary local axis, centred at the local origin
// ---------------------------------------------------------------------------

/// The component of `v` perpendicular to the unit `axis`.
#[inline]
fn radial(v: Vec3, axis: Vec3) -> Vec3 {
    v - axis * v.dot(axis)
}

pub fn cylinder_intersect(
    radius_mm: f64,
    half_length_mm: f64,
    axis_unit: Vec3,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    let a = axis_unit;
    let l = half_length_mm;
    let r = radius_mm;

    // Side: |radial(o + t d)|² = r².
    let o_rad = radial(ray.origin, a);
    let d_rad = radial(ray.dir, a);
    let aa = d_rad.length_squared();
    if aa > PARALLEL_EPS * PARALLEL_EPS {
        let bb = 2.0 * o_rad.dot(d_rad);
        let cc = o_rad.length_squared() - r * r;
        let disc = bb * bb - 4.0 * aa * cc;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            // Stable roots (§7.3 style): avoid cancellation in `-b ± √disc`.
            let q = -0.5 * (bb + bb.signum() * sq);
            for t in [q / aa, cc / q] {
                if t.is_finite() && in_range(t, tmin, tmax) {
                    let p = ray.at(t);
                    let axial = p.dot(a);
                    if axial.abs() <= l {
                        let n = radial(p, a);
                        out(RawHit3 {
                            t_mm: t,
                            point: p,
                            normal: n / r, // outward, |radial| = r here
                            uv: None,
                            face: 0,
                        });
                    }
                }
            }
        }
    }

    // Caps at axial = ±l.
    let da = ray.dir.dot(a);
    if da.abs() >= PARALLEL_EPS {
        for (s, face) in [(l, 1u32), (-l, 2u32)] {
            let t = (s - ray.origin.dot(a)) / da;
            if in_range(t, tmin, tmax) {
                let p = ray.at(t);
                if radial(p, a).length_squared() <= r * r {
                    out(RawHit3 {
                        t_mm: t,
                        point: p,
                        normal: a * s.signum(), // outward along the axis
                        uv: None,
                        face,
                    });
                }
            }
        }
    }
}

pub fn cylinder_contains(
    radius_mm: f64,
    half_length_mm: f64,
    axis_unit: Vec3,
    p: Vec3,
    tol: &ToleranceContext,
) -> super::Containment {
    let axial = p.dot(axis_unit).abs() - half_length_mm;
    let rad = radial(p, axis_unit).length() - radius_mm;
    // Signed distance to the capped cylinder: the larger of the two constraints
    // outside, their max inside.
    let d = if axial <= 0.0 && rad <= 0.0 {
        axial.max(rad)
    } else {
        let ca = axial.max(0.0);
        let cr = rad.max(0.0);
        (ca * ca + cr * cr).sqrt()
    };
    classify(d, tol)
}

// ---------------------------------------------------------------------------
// Conic and even-asphere of revolution about an arbitrary local axis
// ---------------------------------------------------------------------------

/// A surface of revolution: `axial = sag(r)` about `axis` through `vertex`.
///
/// `axis` is unit and points from the vertex toward increasing sag. `c = 1/R` is
/// the vertex curvature, `k` the conic constant, `coeffs` the even aspheric terms
/// `A4, A6, …` (all zero for a pure conic). `inner_r`/`outer_r` clip the radial
/// aperture: a hit outside `[inner_r, outer_r]` is not on the physical surface.
#[derive(Debug, Clone, Copy)]
pub struct Revolution {
    pub vertex: Vec3,
    pub axis: Vec3,
    pub c: f64,
    pub k: f64,
    pub coeffs: [f64; 8],
    pub inner_r: f64,
    pub outer_r: f64,
}

/// Sag of an even-aspheric profile at aperture coordinate `r`.
///
/// `conic_sag(r) + Σ coeffs[i]·r^(2i+4)`, the standard form, even in `r`. Shared
/// verbatim by the 2D profile-curve path and the 3D surface of revolution so the
/// two cannot drift: `r` is the transverse coordinate in 2D and the radial
/// distance in 3D, but the polynomial is identical.
pub(crate) fn even_asphere_sag(c: f64, k: f64, coeffs: &[f64; 8], r: f64) -> f64 {
    let r2 = r * r;
    let denom = 1.0 + (1.0 - (1.0 + k) * c * c * r2).max(0.0).sqrt();
    let mut sag = c * r2 / denom;
    let mut rp = r2 * r2; // r^4
    for &a in coeffs {
        sag += a * rp;
        rp *= r2;
    }
    sag
}

/// `d/dr` of [`even_asphere_sag`], analytic. Odd in `r`, and `0` at `r = 0`,
/// which is why the vertex normal is exactly the axis and needs no special limit.
///
/// Deriving this in closed form — rather than by the central difference the 2D
/// engine uses — is the whole point of §7.4: the finite-difference normal is the
/// known ~1e-9 floor, and the analytic one clears it.
pub(crate) fn even_asphere_dsag(c: f64, k: f64, coeffs: &[f64; 8], r: f64) -> f64 {
    let r2 = r * r;
    let s = (1.0 - (1.0 + k) * c * c * r2).max(0.0).sqrt();
    // d/dr of the conic sag c r² / (1 + s). Stays finite as (1+k)c²r² → 1 because
    // `s` is clamped at 0 and the term is guarded below.
    let mut d = if s > 0.0 { c * r / s } else { 0.0 };
    let mut rp = r2 * r; // r^3
    for (i, &a) in coeffs.iter().enumerate() {
        let power = (2 * i + 4) as f64;
        d += a * power * rp;
        rp *= r2;
    }
    d
}

impl Revolution {
    fn is_pure_conic(&self) -> bool {
        self.coeffs.iter().all(|a| *a == 0.0)
    }

    /// Sag of the profile at radial distance `r`. Even in `r`.
    fn sag(&self, r: f64) -> f64 {
        even_asphere_sag(self.c, self.k, &self.coeffs, r)
    }

    /// `dsag/dr`, analytic.
    fn dsag_dr(&self, r: f64) -> f64 {
        even_asphere_dsag(self.c, self.k, &self.coeffs, r)
    }

    /// The outward-ish geometric normal at a surface point, analytic.
    ///
    /// From `G = axial - sag(r) = 0`: `∇G = axis - (dsag/dr)·r̂`. At `r = 0` the
    /// radial term vanishes and the normal is the axis, so the apex is not a
    /// special case in the arithmetic (§7.4).
    fn normal_at(&self, p: Vec3) -> Vec3 {
        let d = p - self.vertex;
        let rad = radial(d, self.axis);
        let r = rad.length();
        let n = if r > 0.0 {
            self.axis - rad / r * self.dsag_dr(r)
        } else {
            self.axis
        };
        n.normalize()
    }

    /// Radial distance and axial coordinate (sag direction) of a point.
    fn cyl_coords(&self, p: Vec3) -> (f64, f64) {
        let d = p - self.vertex;
        let axial = d.dot(self.axis);
        (radial(d, self.axis).length(), axial)
    }

    fn in_aperture(&self, r: f64) -> bool {
        r >= self.inner_r - 1e-12 * (1.0 + self.outer_r)
            && r <= self.outer_r + 1e-12 * (1.0 + self.outer_r)
    }
}

/// Intersect a conic or even-asphere of revolution.
///
/// A pure conic (`coeffs` all zero) is solved by its exact quadratic; adding
/// aspheric terms switches to a safeguarded Newton seeded from the base-conic
/// roots. Either way the normal is analytic, so the asphere does not inherit the
/// 2D path's finite-difference floor.
pub fn revolution_intersect(
    surf: &Revolution,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    if surf.is_pure_conic() {
        conic_intersect(surf, ray, tmin, tmax, out);
    } else {
        asphere_intersect(surf, ray, tmin, tmax, out);
    }
}

/// Exact conic-of-revolution roots. `F = c(r² + (1+k)z²) − 2z`, with `z` the
/// axial (sag) coordinate and `r²` the radial distance squared.
fn conic_intersect(
    surf: &Revolution,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    let axis = surf.axis;
    let (c, k1) = (surf.c, 1.0 + surf.k);
    let o = ray.origin - surf.vertex;

    let z0 = o.dot(axis);
    let dz = ray.dir.dot(axis);
    let o_rad = radial(o, axis);
    let d_rad = radial(ray.dir, axis);

    let aa = c * (d_rad.length_squared() + k1 * dz * dz);
    let bb = c * (2.0 * o_rad.dot(d_rad) + 2.0 * k1 * z0 * dz) - 2.0 * dz;
    let cc = c * (o_rad.length_squared() + k1 * z0 * z0) - 2.0 * z0;

    // The sag envelope: the axial span the physical cap occupies out to its
    // aperture. The implicit surface is the whole revolution (both nappes of a
    // sphere or hyperboloid); without this bound a ray could "hit" the far side.
    let sag_edge = surf.sag(surf.outer_r);
    let z_lo = sag_edge.min(0.0);
    let z_hi = sag_edge.max(0.0);
    let z_tol = 1e-7 * (1.0 + surf.outer_r.abs());

    let mut consider = |t: f64| {
        if !in_range(t, tmin, tmax) {
            return;
        }
        let p = ray.at(t);
        let (r, z) = surf.cyl_coords(p);
        if surf.in_aperture(r) && z >= z_lo - z_tol && z <= z_hi + z_tol {
            out(RawHit3 {
                t_mm: t,
                point: p,
                normal: surf.normal_at(p),
                uv: None,
                face: 0,
            });
        }
    };

    if aa.abs() < PARALLEL_EPS {
        // Degenerate quadratic: a plane (c = 0), or a ray along the axis.
        if bb.abs() > PARALLEL_EPS {
            consider(-cc / bb);
        }
        return;
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    let q = -0.5 * (bb + bb.signum() * sq);
    consider(q / aa);
    if q.abs() > PARALLEL_EPS {
        consider(cc / q);
    }
}

/// Even-asphere-of-revolution roots: seed from the base conic, then Newton-refine
/// the residual `axial(t) − sag(r(t))` with an **analytic** derivative, falling
/// back to bisection when a step leaves the bracket (§7.4).
fn asphere_intersect(
    surf: &Revolution,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    let axis = surf.axis;
    let o = ray.origin - surf.vertex;
    let z0 = o.dot(axis);
    let dz = ray.dir.dot(axis);
    let o_rad = radial(o, axis);
    let d_rad = radial(ray.dir, axis);

    // r(t) and its derivative, both analytic.
    let r_of = |t: f64| (o_rad + d_rad * t).length();
    let residual = |t: f64| (z0 + t * dz) - surf.sag(r_of(t));
    let dresidual = |t: f64| {
        let rad = o_rad + d_rad * t;
        let r = rad.length();
        let dr = if r > 0.0 { rad.dot(d_rad) / r } else { 0.0 };
        dz - surf.dsag_dr(r) * dr
    };

    // Seeds: base-conic roots bracket the near cap. Fall back to a coarse march
    // over the aperture span so a highly aspheric surface with no nearby conic
    // root is still bracketed rather than missed.
    let base = Revolution {
        coeffs: [0.0; 8],
        ..*surf
    };
    let mut seeds: Vec<f64> = Vec::new();
    conic_intersect(&base, ray, tmin - surf.outer_r.max(1.0), tmax, &mut |h| {
        seeds.push(h.t_mm)
    });
    if seeds.is_empty() {
        let start = tmin.max(0.0);
        let step = surf.outer_r.max(1.0) * 0.1;
        let mut pt = start;
        let mut pv = residual(pt);
        for i in 1..=200 {
            let t = start + step * i as f64;
            if t > tmax {
                break;
            }
            let v = residual(t);
            if pv * v < 0.0 {
                seeds.push(0.5 * (pt + t));
            }
            pt = t;
            pv = v;
        }
    }

    for seed in seeds {
        if let Some(t) = refine_root(seed, tmin, tmax, &residual, &dresidual) {
            let p = ray.at(t);
            let (r, _) = surf.cyl_coords(p);
            if surf.in_aperture(r) {
                out(RawHit3 {
                    t_mm: t,
                    point: p,
                    normal: surf.normal_at(p),
                    uv: None,
                    face: 0,
                });
            }
        }
    }
}

/// Safeguarded Newton for `f(t) = 0` near `seed`, converging in both `|f|` and
/// `|Δt|` and bisecting whenever a Newton step leaves a known sign-change
/// bracket. Returns `None` if it does not converge inside `(tmin, tmax]`.
fn refine_root(
    seed: f64,
    tmin: f64,
    tmax: f64,
    f: &dyn Fn(f64) -> f64,
    df: &dyn Fn(f64) -> f64,
) -> Option<f64> {
    let mut t = seed;
    // Establish a bracket around the seed by probing outward; the residual is
    // smooth here, so a modest span suffices.
    let span = (tmax - tmin).abs().clamp(1.0, 1e6) * 1e-3;
    let (mut lo, mut hi) = (seed - span, seed + span);
    let (mut flo, mut fhi) = (f(lo), f(hi));
    let bracketed = flo * fhi <= 0.0;

    for _ in 0..64 {
        let ft = f(t);
        if ft.abs() < 1e-13 {
            return in_range(t, tmin, tmax).then_some(t);
        }
        let dft = df(t);
        let mut step = if dft.abs() > 1e-300 {
            ft / dft
        } else {
            f64::INFINITY
        };
        let mut nt = t - step;
        if !nt.is_finite() || (bracketed && (nt < lo || nt > hi)) {
            // Newton left the bracket (or blew up): bisect instead.
            nt = 0.5 * (lo + hi);
            step = t - nt;
        }
        if bracketed {
            let fnt = f(nt);
            if flo * fnt <= 0.0 {
                hi = nt;
                fhi = fnt;
            } else {
                lo = nt;
                flo = fnt;
            }
            let _ = fhi;
        }
        t = nt;
        if step.abs() < 1e-14 * (1.0 + t.abs()) {
            break;
        }
    }
    (f(t).abs() < 1e-9 && in_range(t, tmin, tmax)).then_some(t)
}

// ---------------------------------------------------------------------------
// Frustum: a line segment revolved about local +X
// ---------------------------------------------------------------------------

/// Intersect the surface of revolution of the segment `(x0, y0)–(x1, y1)` about
/// local `+X`, with `y0, y1 >= 0` (the caller clips a meridian at the axis).
///
/// This is one analytic surface with three faces of the same family: a cylinder
/// when the radius is constant, a flat annular face when the axial span is zero,
/// and a cone otherwise. §7.2 requires it directly — it is the rim of every
/// revolved lens — and it is exact, never a chord.
///
/// The normal is geometric and its sign is arbitrary; a closed body re-orients it
/// outward with a containment probe, because a segment alone cannot say which
/// side is glass.
#[allow(clippy::too_many_arguments)] // a segment, a ray interval, and a sink
pub fn frustum_intersect(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    let (x_lo, x_hi) = (x0.min(x1), x0.max(x1));
    let flat = (x1 - x0).abs() <= 1e-12 * (1.0 + x0.abs() + x1.abs());

    if flat {
        // A radial face at axial x0: the plane X = x0, banded in radius.
        if ray.dir.x.abs() < PARALLEL_EPS {
            return;
        }
        let t = (x0 - ray.origin.x) / ray.dir.x;
        if !in_range(t, tmin, tmax) {
            return;
        }
        let p = ray.at(t);
        let r = (p.y * p.y + p.z * p.z).sqrt();
        let (r_lo, r_hi) = (y0.min(y1), y0.max(y1));
        let pad = 1e-12 * (1.0 + r_hi);
        if r >= r_lo - pad && r <= r_hi + pad {
            out(RawHit3 {
                t_mm: t,
                point: p,
                normal: Vec3::X,
                uv: None,
                face: 0,
            });
        }
        return;
    }

    // r(x) = m x + b along the segment; the surface is py² + pz² = (m px + b)².
    let m = (y1 - y0) / (x1 - x0);
    let b = y0 - m * x0;
    let o = ray.origin;
    let d = ray.dir;
    let g0 = m * o.x + b;
    let aa = d.y * d.y + d.z * d.z - m * m * d.x * d.x;
    let bb = 2.0 * (o.y * d.y + o.z * d.z) - 2.0 * m * d.x * g0;
    let cc = o.y * o.y + o.z * o.z - g0 * g0;

    let mut consider = |t: f64| {
        if !in_range(t, tmin, tmax) {
            return;
        }
        let p = ray.at(t);
        let r_line = m * p.x + b;
        // Keep the physical nappe (r = m x + b >= 0) and the segment's axial span.
        let pad = 1e-9 * (1.0 + x_hi.abs() + x_lo.abs());
        if r_line >= -pad && p.x >= x_lo - pad && p.x <= x_hi + pad {
            let n = Vec3::new(-m * r_line, p.y, p.z);
            out(RawHit3 {
                t_mm: t,
                point: p,
                normal: n.normalize(),
                uv: None,
                face: 0,
            });
        }
    };

    if aa.abs() < PARALLEL_EPS {
        if bb.abs() > PARALLEL_EPS {
            consider(-cc / bb);
        }
        return;
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    let q = -0.5 * (bb + bb.signum() * sq);
    consider(q / aa);
    if q.abs() > PARALLEL_EPS {
        consider(cc / q);
    }
}

// ---------------------------------------------------------------------------
// Boundary classification
// ---------------------------------------------------------------------------

/// Turn a signed distance to a surface into a [`Containment`](super::Containment),
/// with the boundary band set by the tolerance context.
#[inline]
fn classify(signed_distance: f64, tol: &ToleranceContext) -> super::Containment {
    if signed_distance.abs() <= tol.medium_abs_mm {
        super::Containment::Boundary
    } else if signed_distance < 0.0 {
        super::Containment::Inside
    } else {
        super::Containment::Outside
    }
}
