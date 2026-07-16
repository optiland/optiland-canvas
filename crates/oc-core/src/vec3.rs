//! f64 3D primitives: vector, rigid transform, ray, box, plane.
//!
//! The 3D counterpart of [`crate::vec`] and the `Aabb`/`Ray` half of
//! [`crate::geom`]. Everything is f64: the reference path is the authority, and
//! it is exact end to end (`docs/architecture/canonical-3d-kernel.md`).
//!
//! Rotations are unit quaternions rather than a matrix or Euler angles. Euler
//! angles gimbal-lock and have twelve conventions to get wrong; a matrix is nine
//! numbers holding six constraints, so a serialized one drifts out of SO(3) and
//! starts scaling the scene. A quaternion is four numbers holding one
//! constraint, and that constraint is cheap to restore.
//!
//! The constraint is restored, not assumed. Every constructor and every load
//! normalizes, so a document cannot introduce a rotation that quietly scales.

use crate::vec::Vec2;
use glam::DQuat;
use serde::{Deserialize, Serialize};

/// 3D point/vector in f64. Alias for [`glam::DVec3`].
pub type Vec3 = glam::DVec3;

/// The shortest vector that may be normalized.
///
/// Below this a direction carries no reliable orientation, and normalizing it
/// amplifies whatever noise produced it. Components near `1e-200` are worse than
/// unreliable: their squares underflow to zero, so the length is `0`, and the
/// division returns `NaN` rather than a wrong answer. Both are rejected instead.
pub const MIN_NORM: f64 = 1e-15;

/// A rotation that cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isometry3Error {
    /// A component was `NaN` or infinite.
    NonFinite,
    /// The quaternion is shorter than [`MIN_NORM`] and names no rotation.
    DegenerateRotation,
}

impl std::fmt::Display for Isometry3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "transform component is not finite"),
            Self::DegenerateRotation => {
                write!(
                    f,
                    "quaternion norm is below {MIN_NORM:e} and names no rotation"
                )
            }
        }
    }
}

impl std::error::Error for Isometry3Error {}

/// A rigid body transform: rotate about the origin, then translate.
///
/// The 3D [`crate::vec::Isometry2`]. Local-to-world by convention, matching the
/// 2D type and every `transform` field in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Isometry3Wire", into = "Isometry3Wire")]
pub struct Isometry3 {
    /// Unit quaternion, local-to-world, `[x, y, z, w]`.
    ///
    /// Always normalized and sign-canonical: see [`Isometry3::new`]. Assigning
    /// this field directly bypasses both, so build one through a constructor.
    pub rot_xyzw: [f64; 4],
    /// Translation in millimetres.
    pub pos: Vec3,
}

/// The serialized shape. Exists so that `try_from` runs the same validation on
/// load that a constructor runs in memory: a hand-edited `.ocanvas` carrying a
/// zero or unnormalized quaternion is a load error, not a scaled scene.
#[derive(Serialize, Deserialize)]
struct Isometry3Wire {
    rot_xyzw: [f64; 4],
    pos: Vec3,
}

impl TryFrom<Isometry3Wire> for Isometry3 {
    type Error = Isometry3Error;
    fn try_from(w: Isometry3Wire) -> Result<Self, Self::Error> {
        Isometry3::new(w.pos, w.rot_xyzw)
    }
}

impl From<Isometry3> for Isometry3Wire {
    fn from(i: Isometry3) -> Self {
        Isometry3Wire {
            rot_xyzw: i.rot_xyzw,
            pos: i.pos,
        }
    }
}

impl Default for Isometry3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Isometry3 {
    pub const IDENTITY: Self = Self {
        rot_xyzw: [0.0, 0.0, 0.0, 1.0],
        pos: Vec3::ZERO,
    };

    /// Build from a translation and a quaternion `[x, y, z, w]`.
    ///
    /// Rejects non-finite components and a quaternion shorter than [`MIN_NORM`],
    /// then normalizes and canonicalizes the sign.
    pub fn new(pos: Vec3, rot_xyzw: [f64; 4]) -> Result<Self, Isometry3Error> {
        if !pos.is_finite() || !rot_xyzw.iter().all(|c| c.is_finite()) {
            return Err(Isometry3Error::NonFinite);
        }
        let [x, y, z, w] = rot_xyzw;
        let norm = (x * x + y * y + z * z + w * w).sqrt();
        if norm < MIN_NORM {
            return Err(Isometry3Error::DegenerateRotation);
        }
        Ok(Self {
            rot_xyzw: canonicalize([x / norm, y / norm, z / norm, w / norm]),
            pos,
        })
    }

    /// Translation only.
    pub fn from_pos(pos: Vec3) -> Self {
        Self {
            rot_xyzw: [0.0, 0.0, 0.0, 1.0],
            pos,
        }
    }

    /// Rotation of `angle_rad` about `axis`, no translation.
    pub fn from_axis_angle(axis: Vec3, angle_rad: f64) -> Result<Self, Isometry3Error> {
        if !axis.is_finite() || !angle_rad.is_finite() {
            return Err(Isometry3Error::NonFinite);
        }
        if axis.length() < MIN_NORM {
            return Err(Isometry3Error::DegenerateRotation);
        }
        let q = DQuat::from_axis_angle(axis.normalize(), angle_rad);
        Self::new(Vec3::ZERO, [q.x, q.y, q.z, q.w])
    }

    /// The rotation as a glam quaternion.
    #[inline]
    pub fn quat(&self) -> DQuat {
        let [x, y, z, w] = self.rot_xyzw;
        DQuat::from_xyzw(x, y, z, w)
    }

    /// Build from an already-valid quaternion, normalizing and canonicalizing.
    /// Private: the public surface goes through the checked constructors.
    fn from_quat(q: DQuat, pos: Vec3) -> Self {
        let n = q.normalize();
        Self {
            rot_xyzw: canonicalize([n.x, n.y, n.z, n.w]),
            pos,
        }
    }

    /// Rotate a direction (no translation).
    #[inline]
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        self.quat() * v
    }

    /// Inverse-rotate a direction.
    #[inline]
    pub fn inv_rotate(&self, v: Vec3) -> Vec3 {
        self.quat().conjugate() * v
    }

    /// Local space to world space.
    #[inline]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.pos + self.rotate(p)
    }

    /// World space to local space.
    #[inline]
    pub fn inverse_transform_point(&self, p: Vec3) -> Vec3 {
        self.inv_rotate(p - self.pos)
    }

    /// Compose: `self ∘ inner`, applying `inner` first.
    ///
    /// The 3D counterpart of [`crate::vec::Isometry2::compose`], and it carries
    /// the same guarantee: composing with [`Isometry3::IDENTITY`] returns the
    /// operand untouched, so placing an unrotated fragment at the origin cannot
    /// perturb its geometry.
    pub fn compose(&self, inner: Isometry3) -> Isometry3 {
        if *self == Self::IDENTITY {
            return inner;
        }
        if inner == Self::IDENTITY {
            return *self;
        }
        Self::from_quat(
            self.quat() * inner.quat(),
            self.pos + self.rotate(inner.pos),
        )
    }

    /// The transform that undoes this one.
    pub fn inverse(&self) -> Isometry3 {
        let inv = self.quat().conjugate();
        Self::from_quat(inv, -(inv * self.pos))
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.pos.is_finite() && self.rot_xyzw.iter().all(|c| c.is_finite())
    }
}

/// Pick one of the two quaternions naming a rotation, so that equal rotations
/// serialize equally.
///
/// `q` and `-q` are the same rotation. Without a rule, a scene saved twice can
/// differ on disk while describing the same layout, which breaks the diffable
/// format contract and makes byte-comparison tests flap.
///
/// Prefer `w > 0`; on `w == 0` (a half turn, where both choices have `w == 0`)
/// take the sign of the first non-zero of `x, y, z`.
fn canonicalize(mut q: [f64; 4]) -> [f64; 4] {
    let negate = if q[3] < 0.0 {
        true
    } else if q[3] > 0.0 {
        false
    } else {
        // w == 0, which is also true of -0.0.
        match q.iter().take(3).find(|c| **c != 0.0) {
            Some(c) => *c < 0.0,
            // Unreachable: a zero quaternion is rejected before this runs.
            None => false,
        }
    };
    if negate {
        for c in q.iter_mut() {
            *c = -*c;
        }
    }
    // -0.0 == 0.0 compares equal but serializes as "-0.0". Pin the sign so the
    // document is stable.
    for c in q.iter_mut() {
        if *c == 0.0 {
            *c = 0.0;
        }
    }
    q
}

/// A ray that cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ray3Error {
    NonFinite,
    /// The direction is shorter than [`MIN_NORM`] and points nowhere.
    DegenerateDirection,
}

impl std::fmt::Display for Ray3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "ray component is not finite"),
            Self::DegenerateDirection => {
                write!(f, "ray direction is shorter than {MIN_NORM:e}")
            }
        }
    }
}

impl std::error::Error for Ray3Error {}

/// A ray with a unit direction.
///
/// The direction is unit length by construction, which is what makes `t` a
/// length in millimetres everywhere downstream. Tolerances are quoted in mm, so
/// an unnormalized direction would silently rescale every one of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray3 {
    pub origin: Vec3,
    /// Unit length, guaranteed by [`Ray3::new`].
    pub dir: Vec3,
}

impl Ray3 {
    pub fn new(origin: Vec3, dir: Vec3) -> Result<Self, Ray3Error> {
        if !origin.is_finite() || !dir.is_finite() {
            return Err(Ray3Error::NonFinite);
        }
        if dir.length() < MIN_NORM {
            return Err(Ray3Error::DegenerateDirection);
        }
        Ok(Self {
            origin,
            dir: dir.normalize(),
        })
    }

    /// The point at path length `t_mm`.
    #[inline]
    pub fn at(&self, t_mm: f64) -> Vec3 {
        self.origin + self.dir * t_mm
    }
}

/// An axis-aligned bounding box. The 3D [`crate::geom::Aabb`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb3 {
    /// The identity for [`Aabb3::union`]: inverted bounds, so the first point
    /// expanded into it wins on both sides.
    pub const EMPTY: Aabb3 = Aabb3 {
        min: Vec3::splat(f64::INFINITY),
        max: Vec3::splat(f64::NEG_INFINITY),
    };

    pub fn from_point(p: Vec3) -> Self {
        Aabb3 { min: p, max: p }
    }

    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn union(mut self, other: &Aabb3) -> Aabb3 {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self
    }

    /// Grow by `pad` in all directions.
    pub fn padded(&self, pad: f64) -> Aabb3 {
        Aabb3 {
            min: self.min - Vec3::splat(pad),
            max: self.max + Vec3::splat(pad),
        }
    }

    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    /// Corner-to-corner length. `scene_scale` in the tolerance policy is this
    /// over the whole scene.
    pub fn diagonal(&self) -> f64 {
        (self.max - self.min).length()
    }

    /// Is this box empty (no point has been expanded into it)?
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    pub fn contains_point(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    /// Slab test: does `ray` enter this box at any `t > tmin` and before `tmax`?
    ///
    /// A bound must never omit a true hit, so this is written to be permissive
    /// exactly where the arithmetic goes undefined.
    ///
    /// A direction component of zero gives an infinite slab bound, which is
    /// fine. The trap is a ray that is *both* parallel to a slab and exactly on
    /// it: `(bound - origin)` is `0`, the reciprocal is `inf`, and `0 * inf` is
    /// `NaN`. Such a ray lies in the closed slab for every `t`, so the slab must
    /// not constrain the interval at all.
    ///
    /// The comparison-based updates below get that for free: every comparison
    /// against `NaN` is false, so a `NaN` slab leaves `enter` and `exit`
    /// untouched. Folding with `min`/`max` instead would not. Those return the
    /// non-`NaN` operand, which here is the *near* bound of `-inf`, so `exit`
    /// would collapse to `-inf` and the box would report a miss on a ray that
    /// grazes its face.
    pub fn intersects_ray(&self, ray: &Ray3, tmin: f64, tmax: f64) -> bool {
        let (mut enter, mut exit) = (tmin, tmax);
        for axis in 0..3 {
            let inv = 1.0 / ray.dir[axis];
            let mut near = (self.min[axis] - ray.origin[axis]) * inv;
            let mut far = (self.max[axis] - ray.origin[axis]) * inv;
            if near > far {
                std::mem::swap(&mut near, &mut far);
            }
            if near > enter {
                enter = near;
            }
            if far < exit {
                exit = far;
            }
            if enter > exit {
                return false;
            }
        }
        true
    }
}

/// A plane that cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane3Error {
    NonFinite,
    /// An axis is shorter than [`MIN_NORM`].
    DegenerateAxis,
    /// `u` and `v` are not perpendicular, so they span no orthonormal frame.
    NotOrthogonal,
}

impl std::fmt::Display for Plane3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "plane component is not finite"),
            Self::DegenerateAxis => write!(f, "plane axis is shorter than {MIN_NORM:e}"),
            Self::NotOrthogonal => write!(f, "plane axes u and v are not perpendicular"),
        }
    }
}

impl std::error::Error for Plane3Error {}

/// An oriented plane with an in-plane basis.
///
/// `u`, `v`, `normal` are orthonormal and right-handed (`u × v = normal`), so
/// the 2D coordinates a view reports are a rigid picture of the 3D scene rather
/// than a sheared one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Plane3Wire", into = "Plane3Wire")]
pub struct Plane3 {
    pub origin: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub normal: Vec3,
}

/// The serialized shape, validated on load like [`Isometry3Wire`]. `normal` is
/// stored rather than derived so a document is readable on its own, and
/// recomputed on load so a document cannot disagree with itself.
#[derive(Serialize, Deserialize)]
struct Plane3Wire {
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    normal: Vec3,
}

impl TryFrom<Plane3Wire> for Plane3 {
    type Error = Plane3Error;
    fn try_from(w: Plane3Wire) -> Result<Self, Self::Error> {
        let p = Plane3::new(w.origin, w.u, w.v)?;
        if !w.normal.is_finite() {
            return Err(Plane3Error::NonFinite);
        }
        // A stored normal that disagrees with u × v is a broken document. It is
        // refused rather than silently replaced by the derived one, because the
        // disagreement means the author's intent is unknown.
        if (p.normal - w.normal.normalize_or_zero()).length() > 1e-9 {
            return Err(Plane3Error::NotOrthogonal);
        }
        Ok(p)
    }
}

impl From<Plane3> for Plane3Wire {
    fn from(p: Plane3) -> Self {
        Plane3Wire {
            origin: p.origin,
            u: p.u,
            v: p.v,
            normal: p.normal,
        }
    }
}

impl Default for Plane3 {
    fn default() -> Self {
        Self::XY
    }
}

impl Plane3 {
    /// `Z = 0`, screen horizontal `+X`, screen vertical `+Y`. The default Canvas
    /// view, and the plane the whole v1 scene lives in.
    pub const XY: Plane3 = Plane3 {
        origin: Vec3::ZERO,
        u: Vec3::X,
        v: Vec3::Y,
        normal: Vec3::Z,
    };

    /// Build from an origin and two in-plane axes.
    ///
    /// Normalizes `u` and `v` and derives `normal = u × v`. Non-perpendicular
    /// axes are refused rather than orthogonalized: a caller who passed a
    /// sheared pair meant something, and quietly moving `v` would answer a
    /// question they did not ask.
    pub fn new(origin: Vec3, u: Vec3, v: Vec3) -> Result<Self, Plane3Error> {
        if !origin.is_finite() || !u.is_finite() || !v.is_finite() {
            return Err(Plane3Error::NonFinite);
        }
        if u.length() < MIN_NORM || v.length() < MIN_NORM {
            return Err(Plane3Error::DegenerateAxis);
        }
        let (u, v) = (u.normalize(), v.normalize());
        if u.dot(v).abs() > 1e-12 {
            return Err(Plane3Error::NotOrthogonal);
        }
        Ok(Plane3 {
            origin,
            u,
            v,
            normal: u.cross(v).normalize(),
        })
    }

    /// Signed distance from the plane, positive on the `normal` side.
    #[inline]
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        (p - self.origin).dot(self.normal)
    }

    /// World point to in-plane `(u, v)` coordinates. Any component along the
    /// normal is dropped, so this is a projection and not a round trip.
    #[inline]
    pub fn to_local(&self, p: Vec3) -> Vec2 {
        let d = p - self.origin;
        Vec2::new(d.dot(self.u), d.dot(self.v))
    }

    /// In-plane `(u, v)` coordinates back to a world point on the plane.
    #[inline]
    pub fn from_local(&self, uv: Vec2) -> Vec3 {
        self.origin + self.u * uv.x + self.v * uv.y
    }

    /// The closest point on the plane.
    #[inline]
    pub fn project(&self, p: Vec3) -> Vec3 {
        p - self.normal * self.signed_distance(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, PI};

    fn sample_isometries() -> Vec<Isometry3> {
        vec![
            Isometry3::IDENTITY,
            Isometry3::from_pos(Vec3::new(3.0, -2.0, 7.5)),
            Isometry3::from_axis_angle(Vec3::Z, 0.7).unwrap(),
            Isometry3::from_axis_angle(Vec3::new(1.0, 2.0, -3.0), 1.9).unwrap(),
            Isometry3::new(
                Vec3::new(-11.0, 4.5, 0.25),
                [
                    0.18257418583505536,
                    0.3651483716701107,
                    0.5477225575051661,
                    0.7302967433402214,
                ],
            )
            .unwrap(),
        ]
    }

    // ---- Isometry3: validation ----

    #[test]
    fn a_non_finite_transform_is_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                Isometry3::new(Vec3::new(bad, 0.0, 0.0), [0.0, 0.0, 0.0, 1.0]),
                Err(Isometry3Error::NonFinite)
            );
            assert_eq!(
                Isometry3::new(Vec3::ZERO, [bad, 0.0, 0.0, 1.0]),
                Err(Isometry3Error::NonFinite)
            );
        }
    }

    #[test]
    fn a_quaternion_too_short_to_normalize_is_refused() {
        // Zero names no rotation, and normalizing it is a division by zero that
        // would return NaN rather than an error.
        assert_eq!(
            Isometry3::new(Vec3::ZERO, [0.0, 0.0, 0.0, 0.0]),
            Err(Isometry3Error::DegenerateRotation)
        );
        assert_eq!(
            Isometry3::new(Vec3::ZERO, [1e-16, 0.0, 0.0, 0.0]),
            Err(Isometry3Error::DegenerateRotation)
        );
        // Just above the floor is accepted and normalizes to a real rotation.
        let ok = Isometry3::new(Vec3::ZERO, [1e-14, 0.0, 0.0, 0.0]).unwrap();
        assert!((ok.rot_xyzw[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_unnormalized_quaternion_is_normalized_rather_than_scaling_the_scene() {
        // The hazard this closes: a rotation of norm 2 applied to a point scales
        // it by 4. The layout would look plausible and every distance would be
        // wrong.
        let iso = Isometry3::new(Vec3::ZERO, [0.0, 0.0, 0.0, 2.0]).unwrap();
        assert_eq!(iso.rot_xyzw, [0.0, 0.0, 0.0, 1.0]);

        let iso = Isometry3::new(Vec3::ZERO, [0.0, 0.0, 3.0, 3.0]).unwrap();
        let p = Vec3::new(2.0, -5.0, 1.0);
        assert!((iso.rotate(p).length() - p.length()).abs() < 1e-13);
    }

    // ---- Isometry3: canonicalization ----

    #[test]
    fn a_rotation_and_its_negation_serialize_identically() {
        // q and -q are the same rotation. Without a sign rule the same layout
        // saves two ways, which breaks the diffable-format contract.
        for [x, y, z, w] in [
            [0.0, 0.0, 0.0, 1.0],
            [
                0.18257418583505536,
                0.3651483716701107,
                0.5477225575051661,
                0.7302967433402214,
            ],
            [FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2],
            // w == 0: a half turn, where neither choice has w > 0 and the rule
            // falls to the first non-zero of x, y, z.
            [1.0, 0.0, 0.0, 0.0],
            [0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0],
        ] {
            let a = Isometry3::new(Vec3::ZERO, [x, y, z, w]).unwrap();
            let b = Isometry3::new(Vec3::ZERO, [-x, -y, -z, -w]).unwrap();
            assert_eq!(
                a.rot_xyzw, b.rot_xyzw,
                "{:?} vs {:?}",
                a.rot_xyzw, b.rot_xyzw
            );
            // Same rotation either way, which is why the sign was free to pick.
            let p = Vec3::new(1.0, 2.0, 3.0);
            assert!((a.rotate(p) - b.rotate(p)).length() < 1e-15);
        }
    }

    #[test]
    fn canonical_form_prefers_positive_w() {
        let iso = Isometry3::new(Vec3::ZERO, [0.0, 0.0, 0.0, -1.0]).unwrap();
        assert_eq!(iso.rot_xyzw, [0.0, 0.0, 0.0, 1.0]);
        for iso in sample_isometries() {
            assert!(iso.rot_xyzw[3] >= 0.0, "{:?}", iso.rot_xyzw);
        }
    }

    #[test]
    fn on_a_half_turn_the_first_non_zero_axis_is_positive() {
        // w == 0 for both choices, so w cannot break the tie.
        let iso = Isometry3::new(Vec3::ZERO, [-1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(iso.rot_xyzw, [1.0, 0.0, 0.0, 0.0]);
        let iso = Isometry3::new(Vec3::ZERO, [0.0, -FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]).unwrap();
        assert_eq!(iso.rot_xyzw, [0.0, FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0]);
    }

    #[test]
    fn no_component_serializes_as_negative_zero() {
        // -0.0 == 0.0 but writes as "-0.0", so two equal rotations would produce
        // different bytes.
        for iso in sample_isometries() {
            for c in iso.rot_xyzw {
                assert!(!(c == 0.0 && c.is_sign_negative()), "{:?}", iso.rot_xyzw);
            }
        }
        let iso = Isometry3::new(Vec3::ZERO, [0.0, 0.0, 0.0, -1.0]).unwrap();
        assert!(iso.rot_xyzw.iter().all(|c| !c.is_sign_negative()));
    }

    // ---- Isometry3: algebra ----

    #[test]
    fn transform_round_trips() {
        // §5.1 asks for 1e-13 relative.
        for iso in sample_isometries() {
            for p in [
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-4.0, 7.5, 2.25),
                Vec3::new(1e3, -1e3, 5e2),
            ] {
                let back = iso.inverse_transform_point(iso.transform_point(p));
                let tol = 1e-13 * p.length().max(1.0);
                assert!((back - p).length() <= tol, "{iso:?} {p:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn rotation_is_orthonormal() {
        for iso in sample_isometries() {
            let v = Vec3::new(2.0, -5.0, 1.5);
            assert!((iso.inv_rotate(iso.rotate(v)) - v).length() < 1e-13);
            assert!((iso.rotate(v).length() - v.length()).abs() < 1e-13);
        }
    }

    #[test]
    fn compose_matches_sequential_application() {
        let outer = Isometry3::new(
            Vec3::new(10.0, -3.0, 1.0),
            [0.0, 0.0, (0.3_f64).sin(), (0.3_f64).cos()],
        )
        .unwrap();
        let inner = Isometry3::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), -1.1).unwrap();
        let inner = Isometry3::new(Vec3::new(-2.0, 5.0, 0.5), inner.rot_xyzw).unwrap();
        let both = outer.compose(inner);
        for p in [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-4.0, 7.5, 3.0),
        ] {
            let sequential = outer.transform_point(inner.transform_point(p));
            assert!(
                (both.transform_point(p) - sequential).length() < 1e-12,
                "{p:?}"
            );
        }
        let d = Vec3::new(0.6, -0.8, 0.0);
        assert!((both.rotate(d) - outer.rotate(inner.rotate(d))).length() < 1e-12);
    }

    #[test]
    fn compose_with_identity_is_exact() {
        // Same guarantee as Isometry2: placing an unrotated fragment at the
        // origin may not perturb its geometry at all.
        for iso in sample_isometries() {
            assert_eq!(Isometry3::IDENTITY.compose(iso), iso);
            assert_eq!(iso.compose(Isometry3::IDENTITY), iso);
        }
    }

    #[test]
    fn compose_is_associative() {
        let a = Isometry3::from_axis_angle(Vec3::X, 0.3).unwrap();
        let b = Isometry3::new(Vec3::new(-5.0, 0.5, 2.0), [0.0, 0.6, 0.0, 0.8]).unwrap();
        let c = Isometry3::from_axis_angle(Vec3::new(1.0, -1.0, 2.0), -0.4).unwrap();
        let p = Vec3::new(3.3, -1.2, 0.75);
        let left = a.compose(b).compose(c).transform_point(p);
        let right = a.compose(b.compose(c)).transform_point(p);
        assert!((left - right).length() < 1e-12);
    }

    #[test]
    fn inverse_undoes_the_transform() {
        for iso in sample_isometries() {
            let round = iso.compose(iso.inverse());
            for p in [Vec3::ZERO, Vec3::new(2.0, -3.0, 4.0)] {
                assert!((round.transform_point(p) - p).length() < 1e-12, "{iso:?}");
            }
            // And the inverse agrees with the point-wise inverse.
            let p = Vec3::new(-7.0, 1.5, 0.5);
            let a = iso.inverse().transform_point(p);
            let b = iso.inverse_transform_point(p);
            assert!((a - b).length() < 1e-12);
        }
    }

    #[test]
    fn a_quarter_turn_about_z_maps_x_to_y() {
        // A concrete orientation check, so a sign error cannot hide behind
        // round trips that would pass for a mirrored frame too.
        let iso = Isometry3::from_axis_angle(Vec3::Z, FRAC_PI_2).unwrap();
        assert!((iso.rotate(Vec3::X) - Vec3::Y).length() < 1e-15);
        assert!((iso.rotate(Vec3::Y) + Vec3::X).length() < 1e-15);
        assert!((iso.rotate(Vec3::Z) - Vec3::Z).length() < 1e-15);
    }

    #[test]
    fn rotation_is_right_handed() {
        // det = +1. A left-handed rotation would mirror the scene and still pass
        // every length and round-trip check above.
        for iso in sample_isometries() {
            let (x, y, z) = (
                iso.rotate(Vec3::X),
                iso.rotate(Vec3::Y),
                iso.rotate(Vec3::Z),
            );
            assert!(
                (x.dot(y.cross(z)) - 1.0).abs() < 1e-13,
                "{:?}",
                iso.rot_xyzw
            );
        }
    }

    // ---- Isometry3: serde ----

    #[test]
    fn isometry_serde_round_trips_exactly() {
        for iso in sample_isometries() {
            let json = serde_json::to_string(&iso).unwrap();
            let back: Isometry3 = serde_json::from_str(&json).unwrap();
            assert_eq!(back, iso, "{json}");
        }
    }

    #[test]
    fn a_document_carrying_a_bad_quaternion_is_refused_on_load() {
        // The same validation a constructor runs. A hand-edited scene cannot
        // introduce a rotation that scales.
        let zero = r#"{"rot_xyzw":[0,0,0,0],"pos":[0,0,0]}"#;
        assert!(serde_json::from_str::<Isometry3>(zero).is_err());
        let nan = r#"{"rot_xyzw":[0,0,0,1],"pos":[0,"x",0]}"#;
        assert!(serde_json::from_str::<Isometry3>(nan).is_err());
    }

    #[test]
    fn an_unnormalized_document_loads_normalized() {
        let j = r#"{"rot_xyzw":[0,0,0,5],"pos":[1,2,3]}"#;
        let iso: Isometry3 = serde_json::from_str(j).unwrap();
        assert_eq!(iso.rot_xyzw, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(iso.pos, Vec3::new(1.0, 2.0, 3.0));
    }

    // ---- Ray3 ----

    #[test]
    fn a_ray_direction_is_normalized() {
        let r = Ray3::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0)).unwrap();
        assert_eq!(r.dir, Vec3::Z);
        assert!((r.dir.length() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn t_is_a_length_in_millimetres() {
        // The reason the direction is normalized: every tolerance downstream is
        // quoted in mm, and an unnormalized direction rescales all of them.
        let r = Ray3::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 17.0)).unwrap();
        assert_eq!(r.at(4.0), Vec3::new(1.0, 2.0, 7.0));
        assert!(((r.at(4.0) - r.origin).length() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn an_invalid_ray_is_refused() {
        assert_eq!(
            Ray3::new(Vec3::ZERO, Vec3::ZERO),
            Err(Ray3Error::DegenerateDirection)
        );
        assert_eq!(
            Ray3::new(Vec3::ZERO, Vec3::splat(1e-16)),
            Err(Ray3Error::DegenerateDirection)
        );
        assert_eq!(
            Ray3::new(Vec3::new(f64::NAN, 0.0, 0.0), Vec3::X),
            Err(Ray3Error::NonFinite)
        );
        assert_eq!(
            Ray3::new(Vec3::ZERO, Vec3::new(f64::INFINITY, 0.0, 0.0)),
            Err(Ray3Error::NonFinite)
        );
    }

    // ---- Aabb3 ----

    #[test]
    fn empty_is_the_union_identity() {
        let b = Aabb3::from_point(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(Aabb3::EMPTY.union(&b), b);
        assert!(Aabb3::EMPTY.is_empty());
        assert!(!b.is_empty());
    }

    #[test]
    fn expand_and_union_agree() {
        let pts = [
            Vec3::new(1.0, -2.0, 3.0),
            Vec3::new(-4.0, 5.0, 0.0),
            Vec3::new(0.5, 0.5, 9.0),
        ];
        let mut a = Aabb3::EMPTY;
        for p in pts {
            a.expand(p);
        }
        let b = pts
            .iter()
            .fold(Aabb3::EMPTY, |acc, p| acc.union(&Aabb3::from_point(*p)));
        assert_eq!(a, b);
        assert_eq!(a.min, Vec3::new(-4.0, -2.0, 0.0));
        assert_eq!(a.max, Vec3::new(1.0, 5.0, 9.0));
    }

    #[test]
    fn center_diagonal_and_padding() {
        let b = Aabb3 {
            min: Vec3::new(-1.0, -2.0, -2.0),
            max: Vec3::new(1.0, 2.0, 2.0),
        };
        assert_eq!(b.center(), Vec3::ZERO);
        assert!((b.diagonal() - 6.0).abs() < 1e-15); // (2,4,4) -> 6
        assert_eq!(b.padded(1.0).min, Vec3::new(-2.0, -3.0, -3.0));
        assert!(b.contains_point(Vec3::ZERO));
        assert!(b.contains_point(Vec3::new(1.0, 2.0, 2.0))); // the boundary is inside
        assert!(!b.contains_point(Vec3::new(1.1, 0.0, 0.0)));
    }

    #[test]
    fn the_slab_test_finds_a_hit_and_rejects_a_miss() {
        let b = Aabb3 {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let hit = Ray3::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap();
        assert!(b.intersects_ray(&hit, 0.0, f64::INFINITY));
        let miss = Ray3::new(Vec3::new(-5.0, 3.0, 0.0), Vec3::X).unwrap();
        assert!(!b.intersects_ray(&miss, 0.0, f64::INFINITY));
        // Behind the ray, and outside the t window.
        let behind = Ray3::new(Vec3::new(5.0, 0.0, 0.0), Vec3::X).unwrap();
        assert!(!b.intersects_ray(&behind, 0.0, f64::INFINITY));
        assert!(!b.intersects_ray(&hit, 0.0, 1.0));
    }

    #[test]
    fn a_ray_grazing_a_face_is_not_lost_to_a_nan() {
        // The conservativeness case, and the one worth having. A ray parallel to
        // a slab *and* exactly on it makes `(bound - origin) * inv` into
        // `0 * inf = NaN`. Fold that with min/max and `exit` collapses to -inf,
        // so the box reports a miss on a ray that touches it, and whatever the
        // box contained is skipped. A bound may be too large; it may not lose a
        // hit.
        let b = Aabb3 {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        for (origin, dir) in [
            // On the +Y face, running along it.
            (Vec3::new(-5.0, 1.0, 0.0), Vec3::X),
            // On the -Y face.
            (Vec3::new(-5.0, -1.0, 0.0), Vec3::X),
            // On an edge: parallel to and exactly on two slabs at once.
            (Vec3::new(-5.0, 1.0, 1.0), Vec3::X),
            // Same, on a different axis.
            (Vec3::new(0.0, -5.0, -1.0), Vec3::Y),
            // Through the middle of a face, parallel to the other two slabs.
            (Vec3::new(-5.0, 0.0, 0.0), Vec3::X),
        ] {
            let r = Ray3::new(origin, dir).unwrap();
            assert!(
                b.intersects_ray(&r, 0.0, f64::INFINITY),
                "lost a grazing hit: {origin:?} along {dir:?}"
            );
        }
    }

    #[test]
    fn a_ray_parallel_to_a_slab_but_outside_it_still_misses() {
        // The other half: being permissive about NaN must not make the test
        // useless. This ray is parallel to the Y slabs and clear of them.
        let b = Aabb3 {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        for y in [1.0000001, -1.0000001, 50.0] {
            let r = Ray3::new(Vec3::new(-5.0, y, 0.0), Vec3::X).unwrap();
            assert!(
                !b.intersects_ray(&r, 0.0, f64::INFINITY),
                "false hit at y={y}"
            );
        }
    }

    // ---- Plane3 ----

    #[test]
    fn the_default_plane_is_the_canvas_view() {
        assert_eq!(Plane3::default(), Plane3::XY);
        assert_eq!(Plane3::XY.normal, Vec3::Z);
        // The whole v1 scene lives here, so a v1 point maps to itself.
        assert_eq!(
            Plane3::XY.to_local(Vec3::new(3.0, 4.0, 0.0)),
            Vec2::new(3.0, 4.0)
        );
        assert_eq!(Plane3::XY.signed_distance(Vec3::new(0.0, 0.0, 2.5)), 2.5);
    }

    #[test]
    fn a_plane_derives_a_right_handed_normal() {
        let p = Plane3::new(Vec3::ZERO, Vec3::X, Vec3::Y).unwrap();
        assert_eq!(p.normal, Vec3::Z);
        let p = Plane3::new(Vec3::ZERO, Vec3::Y, Vec3::X).unwrap();
        assert_eq!(p.normal, -Vec3::Z);
        for p in [
            Plane3::XY,
            Plane3::new(Vec3::new(1.0, 2.0, 3.0), Vec3::Z, Vec3::X).unwrap(),
            Plane3::new(
                Vec3::ZERO,
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            )
            .unwrap(),
        ] {
            assert!((p.u.cross(p.v) - p.normal).length() < 1e-15);
            assert!((p.u.length() - 1.0).abs() < 1e-15);
            assert!((p.v.length() - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn a_sheared_or_degenerate_plane_is_refused() {
        assert_eq!(
            Plane3::new(Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0)),
            Err(Plane3Error::NotOrthogonal)
        );
        assert_eq!(
            Plane3::new(Vec3::ZERO, Vec3::X, Vec3::X),
            Err(Plane3Error::NotOrthogonal)
        );
        assert_eq!(
            Plane3::new(Vec3::ZERO, Vec3::ZERO, Vec3::Y),
            Err(Plane3Error::DegenerateAxis)
        );
        assert_eq!(
            Plane3::new(Vec3::new(f64::NAN, 0.0, 0.0), Vec3::X, Vec3::Y),
            Err(Plane3Error::NonFinite)
        );
    }

    #[test]
    fn local_coordinates_round_trip_on_the_plane() {
        let p = Plane3::new(
            Vec3::new(1.0, -2.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        for uv in [Vec2::ZERO, Vec2::new(3.0, -4.0), Vec2::new(1e3, 2.5)] {
            let back = p.to_local(p.from_local(uv));
            assert!((back - uv).length() < 1e-12, "{uv:?} -> {back:?}");
            // A point built from local coordinates lies on the plane.
            assert!(p.signed_distance(p.from_local(uv)).abs() < 1e-12);
        }
    }

    #[test]
    fn projection_drops_only_the_normal_component() {
        let p = Plane3::XY;
        let q = Vec3::new(3.0, 4.0, 9.0);
        assert_eq!(p.project(q), Vec3::new(3.0, 4.0, 0.0));
        assert_eq!(p.signed_distance(q), 9.0);
        // to_local is a projection, not a round trip: height is gone.
        assert_eq!(p.to_local(q), p.to_local(p.project(q)));
    }

    #[test]
    fn plane_serde_round_trips_and_a_disagreeing_normal_is_refused() {
        let p = Plane3::new(Vec3::new(1.0, 2.0, 3.0), Vec3::Z, Vec3::X).unwrap();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Plane3>(&json).unwrap(), p);

        // u x v is +Z here, so a stored -Z is a document that contradicts itself.
        let bad = r#"{"origin":[0,0,0],"u":[1,0,0],"v":[0,1,0],"normal":[0,0,-1]}"#;
        assert_eq!(
            serde_json::from_str::<Plane3>(bad).unwrap_err().to_string(),
            Plane3Error::NotOrthogonal.to_string()
        );
        let sheared = r#"{"origin":[0,0,0],"u":[1,0,0],"v":[1,1,0],"normal":[0,0,1]}"#;
        assert!(serde_json::from_str::<Plane3>(sheared).is_err());
    }

    #[test]
    fn a_rotated_plane_still_measures_height_along_its_own_normal() {
        let iso = Isometry3::from_axis_angle(Vec3::X, FRAC_PI_3).unwrap();
        let p = Plane3::new(Vec3::ZERO, iso.rotate(Vec3::X), iso.rotate(Vec3::Y)).unwrap();
        assert!((p.normal - iso.rotate(Vec3::Z)).length() < 1e-15);
        let above = p.normal * 2.0;
        assert!((p.signed_distance(above) - 2.0).abs() < 1e-14);
        assert!(p.to_local(above).length() < 1e-14);
    }

    #[test]
    fn a_half_turn_is_its_own_inverse() {
        let iso = Isometry3::from_axis_angle(Vec3::Y, PI).unwrap();
        let twice = iso.compose(iso);
        for p in [Vec3::X, Vec3::new(1.0, 2.0, 3.0)] {
            assert!((twice.transform_point(p) - p).length() < 1e-14, "{p:?}");
        }
    }
}
