//! Vector and rigid-transform primitives.
//!
//! The reference tracer is f64 end to end, so `Vec2` aliases glam's [`DVec2`].
//! The spec writes geometry fields as `Vec2`; here that name is f64-backed.

use serde::{Deserialize, Serialize};

/// 2D point/vector in f64. Alias for [`glam::DVec2`].
pub type Vec2 = glam::DVec2;

/// A rigid body transform: rotation about the origin followed by translation.
///
/// Stored as an angle (radians) plus a translation so the serialized form is
/// human-readable and diffable (§6). Composing/inverting is cheap.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Isometry2 {
    /// Rotation angle in radians.
    pub rot: f64,
    /// Translation.
    pub pos: Vec2,
}

impl Default for Isometry2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Isometry2 {
    pub const IDENTITY: Self = Self {
        rot: 0.0,
        pos: Vec2::ZERO,
    };

    pub fn new(pos: Vec2, rot: f64) -> Self {
        Self { rot, pos }
    }

    pub fn from_pos(pos: Vec2) -> Self {
        Self { rot: 0.0, pos }
    }

    /// Rotate a direction vector by `rot` (no translation).
    #[inline]
    pub fn rotate(&self, v: Vec2) -> Vec2 {
        let (s, c) = self.rot.sin_cos();
        Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
    }

    /// Inverse-rotate a direction vector.
    #[inline]
    pub fn inv_rotate(&self, v: Vec2) -> Vec2 {
        let (s, c) = self.rot.sin_cos();
        Vec2::new(c * v.x + s * v.y, -s * v.x + c * v.y)
    }

    /// Map a point from local space to world space.
    #[inline]
    pub fn transform_point(&self, p: Vec2) -> Vec2 {
        self.pos + self.rotate(p)
    }

    /// Map a point from world space to local space.
    #[inline]
    pub fn inverse_transform_point(&self, p: Vec2) -> Vec2 {
        self.inv_rotate(p - self.pos)
    }
}

/// Perpendicular (rotate +90°): `(x, y) -> (-y, x)`.
#[inline]
pub fn perp(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isometry_round_trips() {
        let iso = Isometry2::new(Vec2::new(3.0, -2.0), 0.7);
        let p = Vec2::new(1.5, 4.2);
        let back = iso.inverse_transform_point(iso.transform_point(p));
        assert!((back - p).length() < 1e-15);
    }

    #[test]
    fn rotate_is_orthonormal() {
        let iso = Isometry2::new(Vec2::ZERO, 1.234);
        let v = Vec2::new(2.0, -5.0);
        assert!((iso.inv_rotate(iso.rotate(v)) - v).length() < 1e-15);
        assert!((iso.rotate(v).length() - v.length()).abs() < 1e-15);
    }
}
