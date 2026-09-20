//! Profile → 3D construction: revolve and extrude a closed 2D profile.
//!
//! A [`ClosedProfile2`] is a loop of curves in the local `XY` plane, `+X` the
//! optical axis. It does not say what it *is*: the same loop is a revolved lens
//! or an extruded prism, and the two trace differently (§3.2, `profile2`). The
//! construction rule decides, and this module turns a profile plus a rule into a
//! traceable body.
//!
//! # Revolve
//!
//! Containment is the clean case, and it needs no meridian bookkeeping at all: a
//! point `(x, y, z)` is inside the revolved solid exactly when `(x, √(y²+z²))` is
//! inside the full 2D profile. The profile is mirror-symmetric about `y = 0`, so
//! testing the non-negative radius against the whole loop is the same as testing
//! it against the upper meridian — and [`profile_contains_2d`] is then shared,
//! unchanged, with the extrusion caps.
//!
//! Intersection does walk the meridian: each edge's `y >= 0` portion becomes one
//! surface of revolution (a conic edge → a conic of revolution, a line → a
//! frustum), and an edge lying entirely below the axis is skipped because its
//! mirror above the axis is a separate edge that already covers that surface.
//!
//! # Extrude
//!
//! The profile is swept along `Z` between finite limits. Each edge rules a side
//! surface, solved by intersecting the ray's `XY` projection with the 2D curve
//! and keeping the hits whose `Z` lands in range; the two end caps are the planar
//! region the profile bounds (§7.6).
//!
//! Every normal here is analytic, the even asphere included: the 3D path must not
//! inherit the 2D engine's finite-difference floor (§7.4).

use super::primitives::{even_asphere_dsag, even_asphere_sag, frustum_intersect, RawHit3};
use super::{Containment, LocalHit3, ToleranceContext};
use crate::id::PatchId;
use crate::profile2::{ClosedProfile2, ConicParams2, Curve2};
use crate::vec::{perp, Vec2};
use crate::vec3::{Aabb3, Ray3, Vec3};

/// A direction component below this counts as parallel; see
/// [`super::primitives`] for why this is an angle floor rather than a length.
const PARALLEL_EPS: f64 = 1e-12;

/// The two extrusion caps are not authored edges and cannot carry a per-patch
/// response override, so they take sentinel patch ids well clear of any allocated
/// one (§3.8: an override only ever names an edge).
const CAP_MIN_PATCH: PatchId = PatchId(u64::MAX - 1);
const CAP_MAX_PATCH: PatchId = PatchId(u64::MAX);

// ---------------------------------------------------------------------------
// 2D curve intersection, in the profile plane
// ---------------------------------------------------------------------------

/// One ray/curve intersection in the profile plane, with a **unit**-direction
/// ray so `s` is a distance.
#[derive(Debug, Clone, Copy)]
struct Hit2 {
    s: f64,
    /// Unit, geometric; the caller re-orients. The 3D caller recovers the point
    /// from `t = s / |d_xy|` against its own ray, so only `s` and the in-plane
    /// normal travel here.
    normal: Vec2,
}

/// Enumerate every intersection of the unit-direction 2D ray with a profile
/// curve at `s > smin`. Analytic throughout, the even asphere included.
fn curve2_intersect_unit(
    curve: &Curve2,
    o: Vec2,
    dir: Vec2,
    smin: f64,
    push: &mut dyn FnMut(Hit2),
) {
    match curve {
        Curve2::Line { a, b } => line2(*a, *b, o, dir, smin, push),
        Curve2::Arc {
            center,
            radius_mm,
            a0,
            a1,
        } => arc2(*center, *radius_mm, *a0, *a1, o, dir, smin, push),
        Curve2::Conic(cp) => conic2(cp, &[0.0; 8], o, dir, smin, push),
        Curve2::EvenAsphere { base, coeffs } => asphere2(base, coeffs, o, dir, smin, push),
        Curve2::Polyline { points } => {
            for w in points.windows(2) {
                line2(w[0], w[1], o, dir, smin, push);
            }
        }
    }
}

fn line2(a: Vec2, b: Vec2, o: Vec2, dir: Vec2, smin: f64, push: &mut dyn FnMut(Hit2)) {
    let e = b - a;
    let det = e.perp_dot(dir);
    if det.abs() < PARALLEL_EPS {
        return;
    }
    let rhs = a - o;
    let s = e.perp_dot(rhs) / det;
    let u = dir.perp_dot(rhs) / det;
    if s > smin && (0.0..=1.0).contains(&u) {
        push(Hit2 {
            s,
            normal: perp(e).normalize(),
        });
    }
}

/// Sweep test copied from [`crate::geom`]: the arc means the **unwrapped**
/// difference `a1 - a0`, so a full turn is a circle and a zero sweep is a point.
fn angle_in_sweep(ang: f64, a0: f64, a1: f64) -> bool {
    let tau = std::f64::consts::TAU;
    let sweep = a1 - a0;
    if sweep.abs() >= tau - 1e-9 {
        return true;
    }
    let span = sweep.rem_euclid(tau);
    let rel = (ang - a0).rem_euclid(tau);
    rel <= span + 1e-12
}

#[allow(clippy::too_many_arguments)] // arc geometry, ray, and sink
fn arc2(
    center: Vec2,
    r: f64,
    a0: f64,
    a1: f64,
    o: Vec2,
    dir: Vec2,
    smin: f64,
    push: &mut dyn FnMut(Hit2),
) {
    let f = o - center;
    let b = f.dot(dir);
    let disc = b * b - (f.length_squared() - r * r);
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    for s in [-b - sq, -b + sq] {
        if s > smin {
            let rel = (o + s * dir) - center;
            if angle_in_sweep(rel.y.atan2(rel.x), a0, a1) {
                push(Hit2 { s, normal: rel / r });
            }
        }
    }
}

/// Conic (`coeffs` all zero) or even asphere in the profile plane. The conic is
/// solved exactly; the asphere seeds from it and Newton-refines with an analytic
/// derivative. Shared so the two share their aperture and sag-envelope logic.
fn conic2(
    cp: &ConicParams2,
    coeffs: &[f64; 8],
    o: Vec2,
    dir: Vec2,
    smin: f64,
    push: &mut dyn FnMut(Hit2),
) {
    let pure_conic = coeffs.iter().all(|a| *a == 0.0);
    let axis = cp.axis.normalize();
    let p = perp(axis);
    let rel = o - cp.vertex;
    let y0 = rel.dot(p);
    let z0 = rel.dot(axis);
    let dy = dir.dot(p);
    let dz = dir.dot(axis);
    let ha = cp.half_ap;

    // The physical cap's axial span out to the aperture, so the far nappe of a
    // sphere or hyperboloid is never accepted (as in `crate::geom`).
    let sag_edge = even_asphere_sag(cp.c, cp.k, coeffs, ha);
    let z_lo = sag_edge.min(0.0);
    let z_hi = sag_edge.max(0.0);
    let z_tol = 1e-7 * (1.0 + ha.abs());

    let normal_at = |y: f64, z: f64| {
        if pure_conic {
            // ∇F = (2c y, 2c(1+k)z − 2) in (p, axis).
            ((2.0 * cp.c * y) * p + (2.0 * cp.c * (1.0 + cp.k) * z - 2.0) * axis).normalize()
        } else {
            // Surface normal ∝ (−dsag/dy, 1) in (p, axis).
            let ds = even_asphere_dsag(cp.c, cp.k, coeffs, y);
            ((-ds) * p + axis).normalize()
        }
    };
    let accept = |s: f64, push: &mut dyn FnMut(Hit2)| {
        if s <= smin {
            return;
        }
        let y = y0 + s * dy;
        let z = z0 + s * dz;
        if y.abs() <= ha + 1e-12 && z >= z_lo - z_tol && z <= z_hi + z_tol {
            push(Hit2 {
                s,
                normal: normal_at(y, z),
            });
        }
    };

    if pure_conic {
        let k1 = 1.0 + cp.k;
        let aa = cp.c * dy * dy + cp.c * k1 * dz * dz;
        let bb = 2.0 * cp.c * y0 * dy + 2.0 * cp.c * k1 * z0 * dz - 2.0 * dz;
        let cc = cp.c * y0 * y0 + cp.c * k1 * z0 * z0 - 2.0 * z0;
        if aa.abs() < PARALLEL_EPS {
            if bb.abs() > PARALLEL_EPS {
                accept(-cc / bb, push);
            }
            return;
        }
        let disc = bb * bb - 4.0 * aa * cc;
        if disc < 0.0 {
            return;
        }
        let sq = disc.sqrt();
        let q = -0.5 * (bb + bb.signum() * sq);
        accept(q / aa, push);
        if q.abs() > PARALLEL_EPS {
            accept(cc / q, push);
        }
        return;
    }

    // Asphere: seed from the base conic, refine on residual z(s) − sag(y(s)).
    let sag = |y: f64| even_asphere_sag(cp.c, cp.k, coeffs, y);
    let residual = |s: f64| (z0 + s * dz) - sag(y0 + s * dy);
    let dresidual = |s: f64| dz - even_asphere_dsag(cp.c, cp.k, coeffs, y0 + s * dy) * dy;

    let mut seeds = Vec::new();
    let base = ConicParams2 { half_ap: ha, ..*cp };
    conic2(&base, &[0.0; 8], o, dir, smin - ha.max(1.0), &mut |h| {
        seeds.push(h.s)
    });
    if seeds.is_empty() {
        let step = ha.max(1.0) * 0.1;
        let start = smin.max(0.0);
        let mut ps = start;
        let mut pv = residual(ps);
        for i in 1..=200 {
            let s = start + step * i as f64;
            let v = residual(s);
            if pv * v < 0.0 {
                seeds.push(0.5 * (ps + s));
            }
            ps = s;
            pv = v;
        }
    }
    for mut s in seeds {
        for _ in 0..48 {
            let f = residual(s);
            let d = dresidual(s);
            if d.abs() < 1e-300 {
                break;
            }
            let step = f / d;
            s -= step;
            if step.abs() < 1e-14 * (1.0 + s.abs()) {
                break;
            }
        }
        if s > smin && residual(s).abs() < 1e-9 {
            let y = y0 + s * dy;
            if y.abs() <= ha + 1e-12 {
                push(Hit2 {
                    s,
                    normal: normal_at(y, 0.0),
                });
            }
        }
    }
}

fn asphere2(
    base: &ConicParams2,
    coeffs: &[f64; 8],
    o: Vec2,
    dir: Vec2,
    smin: f64,
    push: &mut dyn FnMut(Hit2),
) {
    conic2(base, coeffs, o, dir, smin, push);
}

// ---------------------------------------------------------------------------
// Point in the closed 2D profile
// ---------------------------------------------------------------------------

/// Is `p` inside the closed profile? Even-odd crossing count.
///
/// The probe direction is deliberately off-axis: a lens rim is a horizontal line
/// and a cap a vertical one, and a ray along either would run through a coincident
/// edge and miscount. An irrational-ish direction meets each edge transversally.
/// Parity is direction-independent for a point that is not on the boundary, which
/// is the only case a containment test must get right.
pub fn profile_contains_2d(profile: &ClosedProfile2, p: Vec2) -> bool {
    let dir = Vec2::new(0.998_654_2, 0.051_857_3).normalize();
    let mut crossings = 0usize;
    for edge in &profile.edges {
        curve2_intersect_unit(&edge.curve, p, dir, 0.0, &mut |_| crossings += 1);
    }
    crossings % 2 == 1
}

/// Signed-distance-free boundary test: how close is `p` to the nearest edge,
/// along the probe direction, relative to the tolerance band?
fn near_profile_boundary(profile: &ClosedProfile2, p: Vec2, tol: &ToleranceContext) -> bool {
    // A cheap boundary check that does not need per-curve distance: fire a short
    // probe both ways and see whether an edge sits within the band. Good enough to
    // report `Boundary` for the medium sampler, which never queries exactly on a
    // surface in practice.
    let band = tol.medium_abs_mm;
    let mut near = false;
    for dir in [Vec2::X, Vec2::Y, perp(Vec2::X)] {
        for edge in &profile.edges {
            curve2_intersect_unit(&edge.curve, p, dir, -band, &mut |h| {
                if h.s.abs() <= band {
                    near = true;
                }
            });
        }
    }
    near
}

// ---------------------------------------------------------------------------
// Extrude
// ---------------------------------------------------------------------------

/// Conservative local bounding box of an extruded profile.
pub fn extrude_aabb(profile: &ClosedProfile2, z_min: f64, z_max: f64) -> Aabb3 {
    let (lo, hi) = profile_xy_bounds(profile);
    Aabb3 {
        min: Vec3::new(lo.x, lo.y, z_min),
        max: Vec3::new(hi.x, hi.y, z_max),
    }
}

/// Classify a local point against an extruded body.
pub fn extrude_contains(
    profile: &ClosedProfile2,
    z_min: f64,
    z_max: f64,
    p: Vec3,
    tol: &ToleranceContext,
) -> Containment {
    let xy = Vec2::new(p.x, p.y);
    let inside_xy = profile_contains_2d(profile, xy);
    let band = tol.medium_abs_mm;
    let on_cap = (p.z - z_min).abs() <= band || (p.z - z_max).abs() <= band;
    let within_z = p.z >= z_min - band && p.z <= z_max + band;

    if !within_z {
        return Containment::Outside;
    }
    if (on_cap && inside_xy) || (within_z && near_profile_boundary(profile, xy, tol)) {
        return Containment::Boundary;
    }
    if inside_xy && p.z > z_min + band && p.z < z_max - band {
        Containment::Inside
    } else {
        Containment::Outside
    }
}

/// Intersect an extruded body: ruled sides plus two end caps.
pub fn extrude_intersect(
    profile: &ClosedProfile2,
    z_min: f64,
    z_max: f64,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(LocalHit3),
) {
    let tol = ToleranceContext::from_aabb(&extrude_aabb(profile, z_min, z_max));
    let probe = 8.0 * tol.medium_abs_mm;
    let orient = |n: Vec3, p: Vec3| -> Vec3 {
        if extrude_contains(profile, z_min, z_max, p + n * probe, &tol) == Containment::Outside {
            n
        } else {
            -n
        }
    };

    // Sides: intersect the XY projection with each curve, keep hits whose Z lands
    // between the caps. The projected direction is not unit, so a 2D distance `s`
    // becomes a 3D parameter `t = s / |d_xy|`.
    let d_xy = Vec2::new(ray.dir.x, ray.dir.y);
    let len_xy = d_xy.length();
    if len_xy >= PARALLEL_EPS {
        let o_xy = Vec2::new(ray.origin.x, ray.origin.y);
        let u_xy = d_xy / len_xy;
        for edge in &profile.edges {
            let patch = edge.patch;
            curve2_intersect_unit(&edge.curve, o_xy, u_xy, tmin * len_xy, &mut |h| {
                let t = h.s / len_xy;
                if t <= tmin || t > tmax {
                    return;
                }
                let p = ray.at(t);
                if p.z < z_min || p.z > z_max {
                    return;
                }
                let n = Vec3::new(h.normal.x, h.normal.y, 0.0).normalize();
                out(LocalHit3 {
                    t_mm: t,
                    point_local: p,
                    normal_local: orient(n, p),
                    patch,
                    uv: None,
                });
            });
        }
    }

    // Caps: the profile-bounded region of each plane Z = z_min, z_max.
    if ray.dir.z.abs() >= PARALLEL_EPS {
        for (z, patch, outward) in [
            (z_min, CAP_MIN_PATCH, Vec3::NEG_Z),
            (z_max, CAP_MAX_PATCH, Vec3::Z),
        ] {
            let t = (z - ray.origin.z) / ray.dir.z;
            if t <= tmin || t > tmax {
                continue;
            }
            let p = ray.at(t);
            if profile_contains_2d(profile, Vec2::new(p.x, p.y)) {
                out(LocalHit3 {
                    t_mm: t,
                    point_local: p,
                    normal_local: outward,
                    patch,
                    uv: Some(Vec2::new(p.x, p.y)),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Revolve
// ---------------------------------------------------------------------------

/// Conservative local bounding box of a revolved profile (about `+X`).
pub fn revolve_aabb(profile: &ClosedProfile2) -> Aabb3 {
    let (lo, hi) = profile_xy_bounds(profile);
    let r = lo.y.abs().max(hi.y.abs());
    Aabb3 {
        min: Vec3::new(lo.x, -r, -r),
        max: Vec3::new(hi.x, r, r),
    }
}

/// Classify a local point against a revolved body: map to `(x, r)` and test the
/// full profile, which for a mirror-symmetric loop is the meridian test.
pub fn revolve_contains(profile: &ClosedProfile2, p: Vec3, tol: &ToleranceContext) -> Containment {
    let r = (p.y * p.y + p.z * p.z).sqrt();
    let xr = Vec2::new(p.x, r);
    if near_profile_boundary(profile, xr, tol) {
        return Containment::Boundary;
    }
    if profile_contains_2d(profile, xr) {
        Containment::Inside
    } else {
        Containment::Outside
    }
}

/// Intersect a revolved body by walking each edge's non-negative-`r` meridian.
pub fn revolve_intersect(
    profile: &ClosedProfile2,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(LocalHit3),
) {
    revolve_intersect_with(profile, &revolve_tolerance(profile), ray, tmin, tmax, out)
}

/// The tolerances a revolved body's hits are oriented with: its own size's.
/// It depends on the profile alone, and finding it samples every face, so a
/// tracer that meets the body at every step works it out once
/// ([`revolve_intersect_with`]).
pub fn revolve_tolerance(profile: &ClosedProfile2) -> ToleranceContext {
    ToleranceContext::from_aabb(&revolve_aabb(profile))
}

/// [`revolve_intersect`] with the body's [`revolve_tolerance`] already found.
pub fn revolve_intersect_with(
    profile: &ClosedProfile2,
    tol: &ToleranceContext,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(LocalHit3),
) {
    let tol = *tol;
    let probe = 8.0 * tol.medium_abs_mm;
    // Outward when a point just off the hit along `n` is outside the body:
    // what `revolve_contains` calls `Outside`, neither inside nor on the
    // boundary band. Asked inside-first, since a point inside needs no
    // boundary probes to be not outside.
    let orient = |n: Vec3, p: Vec3| -> Vec3 {
        let q = p + n * probe;
        let xr = Vec2::new(q.x, (q.y * q.y + q.z * q.z).sqrt());
        let outside =
            !profile_contains_2d(profile, xr) && !near_profile_boundary(profile, xr, &tol);
        if outside {
            n
        } else {
            -n
        }
    };

    for edge in &profile.edges {
        let patch = edge.patch;
        let mut emit = |raw: RawHit3| {
            out(LocalHit3 {
                t_mm: raw.t_mm,
                point_local: raw.point,
                normal_local: orient(raw.normal, raw.point),
                patch,
                uv: raw.uv,
            });
        };
        revolve_edge(&edge.curve, ray, tmin, tmax, &mut emit);
    }
}

/// Revolve one profile edge's `y >= 0` portion about `+X`, reporting geometric
/// (un-oriented) hits. An edge entirely below the axis is skipped: its mirror
/// above the axis is a separate edge covering the same surface.
fn revolve_edge(curve: &Curve2, ray: &Ray3, tmin: f64, tmax: f64, out: &mut dyn FnMut(RawHit3)) {
    use super::primitives::{revolution_intersect, Revolution};
    match curve {
        Curve2::Line { a, b } => revolve_segment(*a, *b, ray, tmin, tmax, out),
        Curve2::Polyline { points } => {
            for w in points.windows(2) {
                revolve_segment(w[0], w[1], ray, tmin, tmax, out);
            }
        }
        Curve2::Conic(cp) => {
            let surf = Revolution {
                vertex: Vec3::new(cp.vertex.x, 0.0, 0.0),
                axis: conic_axis3(cp),
                c: cp.c,
                k: cp.k,
                coeffs: [0.0; 8],
                inner_r: 0.0,
                outer_r: cp.half_ap,
            };
            revolution_intersect(&surf, ray, tmin, tmax, out);
        }
        Curve2::EvenAsphere { base, coeffs } => {
            let surf = Revolution {
                vertex: Vec3::new(base.vertex.x, 0.0, 0.0),
                axis: conic_axis3(base),
                c: base.c,
                k: base.k,
                coeffs: *coeffs,
                inner_r: 0.0,
                outer_r: base.half_ap,
            };
            revolution_intersect(&surf, ray, tmin, tmax, out);
        }
        Curve2::Arc {
            center,
            radius_mm,
            a0,
            a1,
        } => revolve_arc(*center, *radius_mm, *a0, *a1, ray, tmin, tmax, out),
    }
}

/// The 3D revolution axis of a profile conic: `±X`, matching the 2D axis. A valid
/// revolve has every conic on the optical axis, so this is a unit `±X`.
fn conic_axis3(cp: &ConicParams2) -> Vec3 {
    let a = cp.axis.normalize();
    Vec3::new(a.x, a.y, 0.0).normalize()
}

/// Revolve the `y >= 0` part of segment `a–b`, clipping at the axis.
fn revolve_segment(
    a: Vec2,
    b: Vec2,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    let (mut a, mut b) = (a, b);
    if a.y < 0.0 && b.y < 0.0 {
        return; // entirely below the axis; the mirror edge covers it
    }
    // Clip a straddling segment to the axis so only its non-negative-r part is
    // revolved.
    if a.y < 0.0 {
        let u = a.y / (a.y - b.y);
        a = Vec2::new(a.x + u * (b.x - a.x), 0.0);
    } else if b.y < 0.0 {
        let u = b.y / (b.y - a.y);
        b = Vec2::new(b.x + u * (a.x - b.x), 0.0);
    }
    frustum_intersect(a.x, a.y, b.x, b.y, ray, tmin, tmax, out);
}

/// Revolve a profile arc about `+X`. Only an axis-centred arc is supported here —
/// it revolves to a spherical zone, solved exactly. An off-axis arc revolves to a
/// torus, which [`can_promote_to_revolved`](crate::scene3::can_promote_to_revolved)
/// refuses rather than let it trace with a missing surface.
#[allow(clippy::too_many_arguments)] // arc geometry, ray interval, and sink
fn revolve_arc(
    center: Vec2,
    r: f64,
    a0: f64,
    a1: f64,
    ray: &Ray3,
    tmin: f64,
    tmax: f64,
    out: &mut dyn FnMut(RawHit3),
) {
    if center.y.abs() > 1e-9 * (1.0 + center.x.abs()) {
        return; // off-axis: refused at promotion, so this is unreachable in a trace
    }
    // A sphere of radius r centred at (center.x, 0, 0), restricted to the arc's
    // meridian sweep.
    let c3 = Vec3::new(center.x, 0.0, 0.0);
    let o = ray.origin - c3;
    let b = o.dot(ray.dir);
    let disc = b * b - (o.length_squared() - r * r);
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    for t in [-b - sq, -b + sq] {
        if t <= tmin || t > tmax {
            continue;
        }
        let p = ray.at(t);
        let rad = (p.y * p.y + p.z * p.z).sqrt();
        // Meridian angle of the hit: r plays the role of the profile's y.
        let ang = rad.atan2(p.x - center.x);
        if angle_in_sweep(ang, a0, a1) {
            let n = (p - c3) / r;
            out(RawHit3 {
                t_mm: t,
                point: p,
                normal: n,
                uv: None,
                face: 0,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Promotion validation (§6.4, §6.5)
// ---------------------------------------------------------------------------

/// Can this profile be revolved about `+X` into a closed body? (§6.4)
///
/// The geometric half of [`can_promote_to_revolved`](crate::scene3::can_promote_to_revolved):
/// the loop closes, is mirror-symmetric about `y = 0`, every conic sits on the
/// optical axis, every arc is axis-centred (an off-axis arc revolves to a torus,
/// which the intersector does not yet solve and so must be refused, not traced
/// with a hole), and the boundary does not cross itself. The last two are sampled
/// — dense enough for authored geometry, and stated as sampled rather than
/// claimed exact.
pub fn validate_revolvable(profile: &ClosedProfile2) -> Result<(), String> {
    let scale = profile_scale(profile);
    // Curve-type capability first: whether a surface can be a surface of
    // revolution at all is independent of how the loop closes, and gives the
    // most specific message.
    for edge in &profile.edges {
        match &edge.curve {
            Curve2::Conic(cp) => on_axis_conic(cp)?,
            Curve2::EvenAsphere { base, .. } => on_axis_conic(base)?,
            Curve2::Arc { center, .. } => {
                if center.y.abs() > 1e-6 * scale {
                    return Err(
                        "an off-axis arc revolves to a torus, which is not yet supported"
                            .to_string(),
                    );
                }
            }
            Curve2::Line { .. } | Curve2::Polyline { .. } => {}
        }
    }
    if !is_closed(profile, scale) {
        return Err("the profile is not a closed loop".to_string());
    }
    if !is_mirror_symmetric_y(profile, scale) {
        return Err("the profile is not mirror-symmetric about y = 0".to_string());
    }
    if profile_self_intersects(profile) {
        return Err("the profile crosses itself".to_string());
    }
    Ok(())
}

/// Can this profile be extruded along `Z` into a closed body? (§6.5)
///
/// Closure and non-self-intersection of the `XY` loop; the finite-`Z` limits are
/// checked by the caller. No symmetry is required — an extrusion has no axis.
pub fn validate_extrudable(profile: &ClosedProfile2) -> Result<(), String> {
    let scale = profile_scale(profile);
    if !is_closed(profile, scale) {
        return Err("the profile is not a closed loop".to_string());
    }
    if profile_self_intersects(profile) {
        return Err("the profile crosses itself".to_string());
    }
    Ok(())
}

fn on_axis_conic(cp: &ConicParams2) -> Result<(), String> {
    let a = cp.axis.normalize();
    if a.y.abs() > 1e-9 {
        return Err("a lens surface is off the optical axis (+X)".to_string());
    }
    Ok(())
}

/// A length scale for the profile's tolerances: its bounding diagonal, floored.
fn profile_scale(profile: &ClosedProfile2) -> f64 {
    let (lo, hi) = profile_xy_bounds(profile);
    (hi - lo).length().max(1.0)
}

/// The two endpoints of a curve, in traversal order for lines and arcs and at the
/// `±half_ap` rim for conics. Only proximity is read downstream, so the rim order
/// is not load-bearing.
fn curve_endpoints(curve: &Curve2) -> (Vec2, Vec2) {
    match curve {
        Curve2::Line { a, b } => (*a, *b),
        Curve2::Arc {
            center,
            radius_mm,
            a0,
            a1,
        } => (
            *center + *radius_mm * Vec2::new(a0.cos(), a0.sin()),
            *center + *radius_mm * Vec2::new(a1.cos(), a1.sin()),
        ),
        Curve2::Conic(cp) => conic_rim(cp, &[0.0; 8]),
        Curve2::EvenAsphere { base, coeffs } => conic_rim(base, coeffs),
        Curve2::Polyline { points } => (
            *points.first().unwrap_or(&Vec2::ZERO),
            *points.last().unwrap_or(&Vec2::ZERO),
        ),
    }
}

fn conic_rim(cp: &ConicParams2, coeffs: &[f64; 8]) -> (Vec2, Vec2) {
    let axis = cp.axis.normalize();
    let p = perp(axis);
    let sag = even_asphere_sag(cp.c, cp.k, coeffs, cp.half_ap);
    (
        cp.vertex - cp.half_ap * p + sag * axis,
        cp.vertex + cp.half_ap * p + sag * axis,
    )
}

/// Do consecutive edges (with wrap-around) share an endpoint? A necessary
/// condition for a single closed loop, and the one authored profiles satisfy.
fn is_closed(profile: &ClosedProfile2, scale: f64) -> bool {
    let n = profile.edges.len();
    if n < 2 {
        return false;
    }
    let tol = 1e-6 * scale;
    (0..n).all(|i| {
        let (a0, a1) = curve_endpoints(&profile.edges[i].curve);
        let (b0, b1) = curve_endpoints(&profile.edges[(i + 1) % n].curve);
        let d = [
            (a0 - b0).length(),
            (a0 - b1).length(),
            (a1 - b0).length(),
            (a1 - b1).length(),
        ];
        d.iter().cloned().fold(f64::INFINITY, f64::min) <= tol
    })
}

/// Sample a curve into points, endpoints included.
fn sample_curve(curve: &Curve2, n: usize) -> Vec<Vec2> {
    let mut pts = Vec::new();
    match curve {
        Curve2::Line { a, b } => {
            for i in 0..=n {
                pts.push(a.lerp(*b, i as f64 / n as f64));
            }
        }
        Curve2::Arc {
            center,
            radius_mm,
            a0,
            a1,
        } => {
            for i in 0..=n {
                let a = a0 + (a1 - a0) * (i as f64 / n as f64);
                pts.push(*center + *radius_mm * Vec2::new(a.cos(), a.sin()));
            }
        }
        Curve2::Conic(cp) => sample_conic(cp, &[0.0; 8], n, &mut pts),
        Curve2::EvenAsphere { base, coeffs } => sample_conic(base, coeffs, n, &mut pts),
        Curve2::Polyline { points } => pts.extend_from_slice(points),
    }
    pts
}

fn sample_conic(cp: &ConicParams2, coeffs: &[f64; 8], n: usize, pts: &mut Vec<Vec2>) {
    let axis = cp.axis.normalize();
    let p = perp(axis);
    for i in 0..=n {
        let y = -cp.half_ap + 2.0 * cp.half_ap * (i as f64 / n as f64);
        let sag = even_asphere_sag(cp.c, cp.k, coeffs, y);
        pts.push(cp.vertex + y * p + sag * axis);
    }
}

/// Is the profile symmetric under `y → −y`? Sampled: every boundary sample has a
/// mirror sample within tolerance. Exact for lines and on-axis conics, whose
/// samples reflect onto samples of the mirror edge.
fn is_mirror_symmetric_y(profile: &ClosedProfile2, scale: f64) -> bool {
    let tol = 1e-6 * scale;
    let cloud: Vec<Vec2> = profile
        .edges
        .iter()
        .flat_map(|e| sample_curve(&e.curve, 16))
        .collect();
    cloud.iter().all(|p| {
        let mirror = Vec2::new(p.x, -p.y);
        cloud.iter().any(|q| (*q - mirror).length() <= tol)
    })
}

/// Does the closed profile cross itself? Sampled into a polygon, then tested for
/// a proper intersection between non-adjacent segments.
fn profile_self_intersects(profile: &ClosedProfile2) -> bool {
    let mut poly: Vec<Vec2> = Vec::new();
    for e in &profile.edges {
        let s = sample_curve(&e.curve, 12);
        // Drop the shared endpoint with the previous edge to avoid a false
        // adjacency crossing at the seam.
        let start = if poly.last().is_some_and(|last| {
            s.first()
                .is_some_and(|f| (*last - *f).length() < 1e-9 * (1.0 + last.length()))
        }) {
            1
        } else {
            0
        };
        poly.extend_from_slice(&s[start.min(s.len())..]);
    }
    let n = poly.len();
    if n < 4 {
        return false;
    }
    for i in 0..n {
        let a0 = poly[i];
        let a1 = poly[(i + 1) % n];
        for j in (i + 2)..n {
            // Skip the pair that wraps back to segment i.
            if i == 0 && j == n - 1 {
                continue;
            }
            let b0 = poly[j];
            let b1 = poly[(j + 1) % n];
            if segments_cross(a0, a1, b0, b1) {
                return true;
            }
        }
    }
    false
}

/// Proper segment intersection: the two open segments cross transversally. Shared
/// endpoints and collinear touches do not count, so a closed loop's own seams are
/// not flagged.
fn segments_cross(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> bool {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let denom = d1.perp_dot(d2);
    if denom.abs() < 1e-12 {
        return false; // parallel or collinear
    }
    let t = (b0 - a0).perp_dot(d2) / denom;
    let u = (b0 - a0).perp_dot(d1) / denom;
    let eps = 1e-9;
    t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps
}

// ---------------------------------------------------------------------------
// Shared 2D bounds
// ---------------------------------------------------------------------------

/// Conservative `XY` bounds of the whole profile.
fn profile_xy_bounds(profile: &ClosedProfile2) -> (Vec2, Vec2) {
    let mut lo = Vec2::splat(f64::INFINITY);
    let mut hi = Vec2::splat(f64::NEG_INFINITY);
    let mut grow = |p: Vec2| {
        lo = lo.min(p);
        hi = hi.max(p);
    };
    for edge in &profile.edges {
        match &edge.curve {
            Curve2::Line { a, b } => {
                grow(*a);
                grow(*b);
            }
            Curve2::Arc {
                center, radius_mm, ..
            } => {
                // The full-circle box: conservative for any sweep.
                grow(*center - Vec2::splat(*radius_mm));
                grow(*center + Vec2::splat(*radius_mm));
            }
            Curve2::Conic(cp) => conic_bounds(cp, &[0.0; 8], &mut grow),
            Curve2::EvenAsphere { base, coeffs } => conic_bounds(base, coeffs, &mut grow),
            Curve2::Polyline { points } => {
                for p in points {
                    grow(*p);
                }
            }
        }
    }
    if lo.x > hi.x {
        // An empty profile: a degenerate box at the origin rather than an
        // inverted one, so callers do not propagate infinities.
        return (Vec2::ZERO, Vec2::ZERO);
    }
    (lo, hi)
}

/// Sample a conic/asphere edge across its aperture into a bounds accumulator.
fn conic_bounds(cp: &ConicParams2, coeffs: &[f64; 8], grow: &mut dyn FnMut(Vec2)) {
    let axis = cp.axis.normalize();
    let p = perp(axis);
    let n = 24;
    for i in 0..=n {
        let y = -cp.half_ap + 2.0 * cp.half_ap * (i as f64 / n as f64);
        let sag = even_asphere_sag(cp.c, cp.k, coeffs, y);
        grow(cp.vertex + y * p + sag * axis);
    }
}
