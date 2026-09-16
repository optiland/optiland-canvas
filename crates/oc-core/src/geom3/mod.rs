//! 3D analytic geometry: intersection, normals, containment, and bounds.
//!
//! The 3D counterpart of [`crate::geom`], and the geometric half of the
//! canonical kernel (`docs/architecture/canonical-3d-kernel.md`). Everything is
//! f64: the reference path is the authority and it is exact end to end.
//!
//! # The contract
//!
//! Every traceable geometry implements [`Intersect3`]. The three methods answer
//! the three questions the tracer asks of a shape, and each carries a rule that
//! a reviewer should reject a PR for breaking (§7.1):
//!
//! - **`intersect`** returns *every* forward hit in the requested interval, at
//!   `t > t_min`, with a unit normal in the stable convention of §5.4. No hidden
//!   tessellation, and roots at a tangency are deduplicated rather than doubled.
//! - **`aabb_world`** is conservative: it may be too large, but it MUST NOT omit
//!   a valid hit. A box that clips a grazing ray is a silent dropped interaction.
//! - **`contains`** classifies a point against the *closed* body, because the
//!   medium at a point is resolved by containment and priority, never by which
//!   way a normal happens to face (§5.4).
//!
//! # Local and world frames
//!
//! A shape does its math in its own local frame. [`Intersect3::intersect`] pulls
//! the world ray back through the object transform, solves locally, and reports a
//! [`LocalHit3`]; the tracer lifts it to a world-space [`SurfaceHit3`] with
//! [`LocalHit3::to_surface_hit`]. Splitting it this way keeps every primitive
//! transform-free, which is the property the transform-invariance tests pin: a
//! rotated sphere and a sphere-with-a-rotated-ray must return the same hit.
//!
//! `t_mm` is the one quantity that needs no lifting. A rigid transform preserves
//! distance and leaves the direction unit, so the path length is identical in
//! both frames.

pub mod primitives;
pub mod profile;
pub mod solid;
pub mod surface;

use crate::id::{ObjectId, PatchId};
use crate::vec::Vec2;
use crate::vec3::{Aabb3, Isometry3, Ray3, Vec3};

/// Where a point sits relative to a closed body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Containment {
    Inside,
    Outside,
    /// On the surface within [`ToleranceContext::medium_abs_mm`]. The tracer
    /// samples just off a boundary rather than on it, so this is a diagnostic
    /// answer, not a normal transport state.
    Boundary,
    /// The body cannot answer. A `PlanarOnly` profile is the case that matters:
    /// its third dimension is unresolved, so it has no volume to be inside of,
    /// and a full-3D trace of a scene containing one must refuse rather than
    /// invent a depth (§3.2, §7.7).
    Unsupported,
}

/// One intersection, in the intersected geometry's local frame.
///
/// Local because [`Intersect3::intersect`] solves in local coordinates; lifting
/// the point and normal to world is [`LocalHit3::to_surface_hit`]. The normal is
/// oriented by the stable convention of §5.4 — outward for a closed body, local
/// `+Z` for a zero-thickness surface — and is **not** yet flipped to oppose the
/// incident ray. That flip is a temporary copy the interaction math makes; the
/// stored orientation is what two-sided response selection reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalHit3 {
    pub t_mm: f64,
    pub point_local: Vec3,
    pub normal_local: Vec3,
    /// The surface hit. For a profile body this is the authored [`PatchId`] of
    /// the edge that became this surface, so a per-patch response override and a
    /// path record still resolve after unrelated edits (§3.8). For an analytic
    /// solid, which carries no authored patches, it is a stable per-face index.
    pub patch: PatchId,
    /// Surface parameters, where the surface has a natural pair. Planar shapes
    /// report `(u, v)` in the plane; curved surfaces may leave it `None`.
    pub uv: Option<Vec2>,
}

impl LocalHit3 {
    /// Lift to a world-space [`SurfaceHit3`] and tag it with its object.
    pub fn to_surface_hit(&self, transform: &Isometry3, object: ObjectId) -> SurfaceHit3 {
        SurfaceHit3 {
            t_mm: self.t_mm,
            point: transform.transform_point(self.point_local),
            geometric_normal: transform.rotate(self.normal_local),
            object,
            patch: self.patch,
            uv: self.uv,
        }
    }
}

/// One intersection in world coordinates, tagged with the object it belongs to.
///
/// The form the reference tracer consumes (§5). `geometric_normal` is a unit
/// world-space normal in the stable convention of §5.4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceHit3 {
    pub t_mm: f64,
    pub point: Vec3,
    pub geometric_normal: Vec3,
    pub object: ObjectId,
    pub patch: PatchId,
    pub uv: Option<Vec2>,
}

/// A traceable 3D geometry.
///
/// Implemented in local coordinates; the transform is threaded through so a
/// caller never has to place a shape by hand. See the module docs for the frame
/// convention and the three rules the methods hold to.
pub trait Intersect3 {
    /// A world-space box that contains every point the shape can be hit at.
    ///
    /// Conservative: too large is allowed; missing a hit is not.
    fn aabb_world(&self, transform: &Isometry3) -> Aabb3;

    /// Report every hit with `t_min_mm < t <= t_max_mm`, in local coordinates.
    ///
    /// `out` is called once per hit and never allocates (§16.4). Hits need not
    /// arrive sorted; the tracer groups them.
    fn intersect(
        &self,
        ray_world: &Ray3,
        transform: &Isometry3,
        t_min_mm: f64,
        t_max_mm: f64,
        out: &mut dyn FnMut(LocalHit3),
    );

    /// Classify a world point against the closed body.
    ///
    /// A zero-thickness surface has no interior and answers [`Containment::Outside`]
    /// everywhere: it never changes which medium a point is in.
    fn contains(
        &self,
        point_world: Vec3,
        transform: &Isometry3,
        tol: &ToleranceContext,
    ) -> Containment;
}

/// Pull a world ray back into an object's local frame.
///
/// A rigid transform sends a unit direction to a unit direction, so the result
/// is a valid [`Ray3`] and its `t` means the same length it did in world space.
pub(crate) fn ray_to_local(transform: &Isometry3, ray: &Ray3) -> Ray3 {
    Ray3 {
        origin: transform.inverse_transform_point(ray.origin),
        dir: transform.inv_rotate(ray.dir),
    }
}

/// Scale-derived tolerances (§8).
///
/// Fixed global epsilons break at both ends: a value tuned for millimetres calls
/// two microscope surfaces coincident, and one tuned for microns never lands a
/// hit on a metre-scale layout. Every tolerance here is derived from one number,
/// the scene scale, so a scene and a geometrically similar copy of it a thousand
/// times larger reach the same decisions (the property the scale-invariance fuzz
/// tests pin, from `1e-3 mm` to `1e4 mm`).
///
/// The constants MAY be tuned by tests, but every backend derives its tolerances
/// from this one policy so they cannot silently disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToleranceContext {
    /// The scene AABB diagonal, floored at 1 mm.
    pub scene_scale_mm: f64,
    /// Coincidence at the level of a point's own representation error.
    pub origin_abs_mm: f64,
    /// How far off a boundary the tracer samples to read the medium.
    pub medium_abs_mm: f64,
    /// How close two hits must be to count as one event.
    pub event_abs_mm: f64,
    pub relative: f64,
    pub angular_rad: f64,
}

impl ToleranceContext {
    /// Derive tolerances from a scene scale in millimetres.
    pub fn for_scene_scale(scene_scale_mm: f64) -> Self {
        let scale = if scene_scale_mm.is_finite() {
            scene_scale_mm.max(1.0)
        } else {
            1.0
        };
        // The relative term (`1e-10 * scale`) dominates at every scale: 64 ulp at
        // scene magnitude is `64 * 2.2e-16 * scale`, twelve orders below it. The
        // ulp floor is carried anyway so the policy stays honest if a caller ever
        // tightens the relative constant toward machine precision.
        let ulp = 64.0 * scale * f64::EPSILON;
        let origin = (1e-10 * scale).max(ulp);
        Self {
            scene_scale_mm: scale,
            origin_abs_mm: origin,
            medium_abs_mm: (1e-9 * scale).max(8.0 * origin),
            event_abs_mm: (1e-10 * scale).max(4.0 * origin),
            relative: 1e-12,
            angular_rad: 1e-12,
        }
    }

    /// Derive tolerances from a scene bounding box.
    pub fn from_aabb(bb: &Aabb3) -> Self {
        Self::for_scene_scale(bb.diagonal())
    }
}

impl Default for ToleranceContext {
    /// A unit-scale context, for tests and single-object queries.
    fn default() -> Self {
        Self::for_scene_scale(1.0)
    }
}
