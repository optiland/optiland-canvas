//! 2D authoring profiles, and the rule that turns one into a solid.
//!
//! Drawing stays 2D. A profile is what the freehand tool, the fitter, and the
//! lens inspector produce, and none of that changes. What changes is that a
//! profile no longer *is* the object: it is a cross-section plus an explicit
//! statement of how that cross-section becomes a body.
//!
//! **A drawn outline does not say what it is.** The same closed curve is a
//! revolved lens or an extruded prism depending only on the author's intent, and
//! the two trace differently. So the intent is stored, and where it is unknown
//! the object says so ([`ProfileConstruction3::PlanarOnly`]) rather than guessing
//! (contract §3.2).

use crate::id::PatchId;
use crate::vec::Vec2;
use serde::{Deserialize, Serialize};

/// A conic section in the profile plane.
///
/// The v2 counterpart of [`crate::geom::ConicParams`], field for field, so v1
/// migration copies rather than converts and no value moves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConicParams2 {
    /// Vertex position, profile frame.
    pub vertex: Vec2,
    /// Axis direction. Need not be unit; normalized on use.
    pub axis: Vec2,
    /// Curvature `c = 1/R` at the vertex. `c = 0` is a plane.
    pub c: f64,
    /// Conic constant `k`.
    pub k: f64,
    /// Half aperture: hits beyond this are rejected.
    pub half_ap: f64,
}

impl ConicParams2 {
    pub fn is_finite(&self) -> bool {
        self.vertex.is_finite()
            && self.axis.is_finite()
            && self.c.is_finite()
            && self.k.is_finite()
            && self.half_ap.is_finite()
    }
}

/// One curve of a closed profile.
///
/// The v1 [`crate::geom::Edge`] with one deliberate difference: there is no
/// `Spline`. See [`Curve2::Polyline`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Curve2 {
    Line {
        a: Vec2,
        b: Vec2,
    },
    /// A circular arc swept CCW from `a0` to `a1`.
    ///
    /// The **unwrapped** difference `a1 - a0` is the sweep, carrying v1's
    /// semantics unchanged: `a1 == a0` is empty, `a1 == a0 + TAU` is a full
    /// circle.
    Arc {
        center: Vec2,
        radius_mm: f64,
        a0: f64,
        a1: f64,
    },
    Conic(ConicParams2),
    /// Base conic plus even aspheric terms: `sag += Σ coeffs[i]·y^(2i+4)`.
    EvenAsphere {
        base: ConicParams2,
        coeffs: [f64; 8],
    },
    /// A polyline through its points.
    ///
    /// This is where v1's `Spline { ctrl, degree }` lands, and the rename is the
    /// point. That variant was never a spline: it is documented as "evaluated as
    /// a polyline through the control points", so `degree` was read by nothing
    /// and the name promised curvature the tracer never delivered. Migrating it
    /// to `Polyline` changes no ray and stops the schema making a claim the
    /// physics does not (§6.3).
    ///
    /// A real B-spline arrives with the PR that implements B-spline intersection,
    /// as a separate variant, or not at all.
    Polyline {
        points: Vec<Vec2>,
    },
    /// A portion of an on-axis sag curve: the points `base.vertex + sag(y)·axis
    /// + y·perp(axis)` for profile height `y` from `y0` to `y1`, with `sag` the
    /// even asphere of `base` and `coeffs` (a conic when they are all zero).
    ///
    /// What a Fresnel facet is: an exact annular piece of a base surface shifted
    /// along the axis, not a chord of it. `y0` and `y1` are signed heights in the
    /// profile, so the facet above the axis and its mirror below are two edges,
    /// as a lens rim is. `base.half_ap` is not read; the segment's own `y0..y1`
    /// is its extent. Revolved, a segment above the axis is an annular surface
    /// of revolution between radii `|y0|` and `|y1|`.
    SagSegment {
        base: ConicParams2,
        coeffs: [f64; 8],
        y0: f64,
        y1: f64,
    },
}

impl Curve2 {
    pub fn is_finite(&self) -> bool {
        match self {
            Curve2::Line { a, b } => a.is_finite() && b.is_finite(),
            Curve2::Arc {
                center,
                radius_mm,
                a0,
                a1,
            } => center.is_finite() && radius_mm.is_finite() && a0.is_finite() && a1.is_finite(),
            Curve2::Conic(c) => c.is_finite(),
            Curve2::EvenAsphere { base, coeffs } => {
                base.is_finite() && coeffs.iter().all(|c| c.is_finite())
            }
            Curve2::Polyline { points } => points.iter().all(|p| p.is_finite()),
            Curve2::SagSegment {
                base,
                coeffs,
                y0,
                y1,
            } => {
                base.is_finite()
                    && coeffs.iter().all(|c| c.is_finite())
                    && y0.is_finite()
                    && y1.is_finite()
            }
        }
    }
}

/// A curve plus the stable identity of the surface it becomes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileEdge2 {
    /// Stable across insertion, removal, and reorder. A per-patch response
    /// override and a stop designation both point here, and neither may follow a
    /// vector index (§3.8).
    pub patch: PatchId,
    pub curve: Curve2,
}

/// A closed loop of profile curves.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ClosedProfile2 {
    pub edges: Vec<ProfileEdge2>,
}

impl ClosedProfile2 {
    pub fn is_finite(&self) -> bool {
        self.edges.iter().all(|e| e.curve.is_finite())
    }
}

/// How a 2D profile becomes a 3D body.
///
/// There is no default. That is the design: a missing rule is a distinct state
/// ([`PlanarOnly`](ProfileConstruction3::PlanarOnly)) rather than a silent pick.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProfileConstruction3 {
    /// **Compatibility only. No 3D body exists.**
    ///
    /// The profile is known, the third dimension is not. Every v1 body loads
    /// like this, because a v1 document never recorded whether its outline was
    /// revolved or extruded and no amount of reading it will say.
    ///
    /// Such a body displays and traces in the planar compatibility path exactly
    /// as it did in v1. A full-3D trace refuses it with a typed capability issue
    /// until somebody promotes it on purpose. The cliff is deliberate: the
    /// alternative is inventing a depth nobody authored (§3.2).
    PlanarOnly {
        /// The plane the profile lives in. `0` for a migrated v1 body.
        plane_z_mm: f64,
    },

    /// Revolve a validated meridional profile about local `+X`.
    ///
    /// What a lens is. Validation is in the geometry PR, not here.
    RevolveLocalX,

    /// Extrude the closed XY profile between finite local-Z limits.
    ///
    /// What a prism is. The limits are finite because an infinite extrusion is
    /// not an object: it has no volume to bound and no caps to close.
    ExtrudeLocalZ { z_min_mm: f64, z_max_mm: f64 },
}

impl ProfileConstruction3 {
    /// Is this a real 3D body?
    pub fn is_3d(&self) -> bool {
        !matches!(self, ProfileConstruction3::PlanarOnly { .. })
    }

    pub fn is_finite(&self) -> bool {
        match self {
            ProfileConstruction3::PlanarOnly { plane_z_mm } => plane_z_mm.is_finite(),
            ProfileConstruction3::RevolveLocalX => true,
            ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
                z_min_mm.is_finite() && z_max_mm.is_finite()
            }
        }
    }
}

/// A body authored as a 2D profile plus a construction rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileSolid3 {
    pub profile: ClosedProfile2,
    pub construction: ProfileConstruction3,
}
