//! The canonical openUC2 frame, and the only place axes are swapped (§6).
//!
//! Coordinate ambiguity is a release blocker: the physical icon, the CAD, the
//! ports, the optical geometry, and the rays all have to land in the same place.
//! So every mapping between frames lives here as a named function, and nothing
//! anywhere else is allowed to reorder an axis inline.
//!
//! # The frames
//!
//! Canonical openUC2, right-handed, millimetres:
//!
//! | Axis | Meaning |
//! |---|---|
//! | `+X` | nominal forward optical direction |
//! | `+Y` | lateral, in the module/grid plane |
//! | `+Z` | vertical, and the layer stacking direction |
//!
//! Canvas is the `XY` plane of that frame, unchanged: `canvas.x = X_uc2`,
//! `canvas.y = Y_uc2`. `Z_uc2` is *not* a Canvas coordinate — the tracer is 2D,
//! so height is carried by the selected layer and by port pose metadata.
//!
//! Three.js keeps its own convention (`+Y` up, `+Z` nominally forward), which is
//! a relabelling of the same right-handed frame:
//! `X_three = Y_uc2`, `Y_three = Z_uc2`, `Z_three = X_uc2`.

use oc_core::{Isometry2, Vec2};
use serde::{Deserialize, Serialize};

/// A point in the canonical openUC2 frame, in millimetres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Uc2Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Uc2Point {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn from_mm(v: [f64; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }

    pub fn to_mm(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// A point in the Three.js mechanical frame, in millimetres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreePoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ThreePoint {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn to_mm(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

/// openUC2 → Canvas. Drops `z`: the tracer is 2D and height is a layer, not a
/// coordinate.
pub fn uc2_to_canvas(p: Uc2Point) -> Vec2 {
    Vec2::new(p.x, p.y)
}

/// Canvas → openUC2, restoring the height the 2D scene never carried.
pub fn canvas_to_uc2(v: Vec2, z: f64) -> Uc2Point {
    Uc2Point::new(v.x, v.y, z)
}

/// openUC2 → Three.js: `X_three = Y_uc2`, `Y_three = Z_uc2`, `Z_three = X_uc2`.
pub fn uc2_to_three(p: Uc2Point) -> ThreePoint {
    ThreePoint::new(p.y, p.z, p.x)
}

/// Three.js → openUC2, the exact inverse of [`uc2_to_three`].
pub fn three_to_uc2(t: ThreePoint) -> Uc2Point {
    Uc2Point::new(t.z, t.x, t.y)
}

/// A whole-quadrant module rotation about its anchor-cell centre.
///
/// The grid admits only these four, so rotation is an integer and a placement
/// can never accumulate a fractional angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum Quadrant {
    R0,
    R90,
    R180,
    R270,
}

impl Quadrant {
    /// From a quarter-turn count. Rejects out-of-range rather than wrapping: a
    /// setup asking for rotation 7 is malformed, not secretly asking for 3.
    pub fn from_quarter_turns(q: u8) -> Result<Self, InvalidQuadrant> {
        match q {
            0 => Ok(Quadrant::R0),
            1 => Ok(Quadrant::R90),
            2 => Ok(Quadrant::R180),
            3 => Ok(Quadrant::R270),
            other => Err(InvalidQuadrant(other)),
        }
    }

    pub fn quarter_turns(self) -> u8 {
        match self {
            Quadrant::R0 => 0,
            Quadrant::R90 => 1,
            Quadrant::R180 => 2,
            Quadrant::R270 => 3,
        }
    }

    pub fn degrees(self) -> i32 {
        self.quarter_turns() as i32 * 90
    }

    /// The rotation in radians, built from `FRAC_PI_2` rather than from a
    /// degrees-to-radians conversion, so a quarter turn is exact.
    pub fn radians(self) -> f64 {
        self.quarter_turns() as f64 * std::f64::consts::FRAC_PI_2
    }

    /// Rotate an integer cell offset about the anchor, in the grid plane.
    ///
    /// Cells are exact integers and must stay exact: rotating through floating
    /// point and rounding back would be a way to lose a cell at some angle.
    /// A quarter turn CCW is `(dx, dy) -> (-dy, dx)`; the layer index is
    /// untouched, since a module does not rotate out of its own layer.
    pub fn rotate_cell(self, cell: [i32; 3]) -> [i32; 3] {
        let [dx, dy, dz] = cell;
        match self {
            Quadrant::R0 => [dx, dy, dz],
            Quadrant::R90 => [-dy, dx, dz],
            Quadrant::R180 => [-dx, -dy, dz],
            Quadrant::R270 => [dy, -dx, dz],
        }
    }
}

impl TryFrom<u8> for Quadrant {
    type Error = InvalidQuadrant;
    fn try_from(q: u8) -> Result<Self, Self::Error> {
        Quadrant::from_quarter_turns(q)
    }
}

impl From<Quadrant> for u8 {
    fn from(q: Quadrant) -> u8 {
        q.quarter_turns()
    }
}

/// A rotation that is not a whole quarter turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidQuadrant(pub u8);

impl std::fmt::Display for InvalidQuadrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rotation {} is not a quarter turn; the grid allows 0, 1, 2, or 3",
            self.0
        )
    }
}

impl std::error::Error for InvalidQuadrant {}

/// The grid a setup is placed on: data, not a constant.
///
/// The pitch is a property of an openUC2 product line, not of the ray tracer, so
/// it arrives from the catalogue and is never hardcoded in Canvas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridProfile {
    pub id: String,
    /// Cell-to-cell spacing along `X`, `Y`, and `Z`/layer.
    pub pitch_mm: [f64; 3],
    pub cell_size_mm: [f64; 3],
    /// Height of the mounting plane above a layer's base.
    pub mounting_plane_offset_mm: f64,
    pub allowed_rotations_deg: Vec<i32>,
}

impl GridProfile {
    /// The default openUC2 profile: a 50 mm in-plane grid on a 55 mm layer pitch.
    pub fn openuc2_core_50() -> Self {
        Self {
            id: "openuc2-core-50".to_string(),
            pitch_mm: [50.0, 50.0, 55.0],
            cell_size_mm: [50.0, 50.0, 50.0],
            mounting_plane_offset_mm: 5.0,
            allowed_rotations_deg: vec![0, 90, 180, 270],
        }
    }

    /// Where a cell's module origin sits in the canonical frame.
    ///
    /// A module's frame origin is the centre of its anchor cell **on the
    /// mounting plane**, so `z` is the layer's plane rather than the cube's
    /// mid-height: §6.4 forbids inferring optical height from cube dimensions,
    /// because ports carry their own explicit `z`.
    pub fn cell_origin(&self, cell: [i32; 3]) -> Uc2Point {
        Uc2Point::new(
            cell[0] as f64 * self.pitch_mm[0],
            cell[1] as f64 * self.pitch_mm[1],
            cell[2] as f64 * self.pitch_mm[2] + self.mounting_plane_offset_mm,
        )
    }

    /// Which cell a Canvas point falls in, on `layer`.
    ///
    /// The inverse of [`cell_origin`](Self::cell_origin) in the plane, and the
    /// snap that turns a pointer into a placement (§15.6). A cell origin is the
    /// cell's *centre*, so this rounds rather than floors: the cell you are in is
    /// the one whose centre is nearest, and the boundary falls halfway between.
    ///
    /// `z` is passed in rather than derived: height is a layer choice, and the 2D
    /// tracer has no third coordinate to read one from.
    pub fn cell_at(&self, canvas: Vec2, layer: i32) -> [i32; 3] {
        let round = |v: f64, pitch: f64| -> i32 {
            if pitch.abs() < 1e-9 || !v.is_finite() {
                return 0;
            }
            (v / pitch).round() as i32
        };
        [
            round(canvas.x, self.pitch_mm[0]),
            round(canvas.y, self.pitch_mm[1]),
            layer,
        ]
    }

    /// Is `q` permitted by this profile?
    pub fn allows(&self, q: Quadrant) -> bool {
        self.allowed_rotations_deg.contains(&q.degrees())
    }

    /// The Canvas placement transform for a module at `anchor` rotated by `q`.
    ///
    /// This is the single bridge from "grid cell plus quarter turn" to the rigid
    /// [`Isometry2`] that `oc-core` places a fragment with. A module fragment is
    /// authored about its own anchor, so the rotation needs no separate centre.
    pub fn placement(&self, anchor: [i32; 3], q: Quadrant) -> Isometry2 {
        Isometry2::new(uc2_to_canvas(self.cell_origin(anchor)), q.radians())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canonical_point_maps_as_specified() {
        // §6.5: (100, 150, 55) -> Canvas (100, 150), Three.js (150, 55, 100).
        let p = Uc2Point::new(100.0, 150.0, 55.0);
        assert_eq!(uc2_to_canvas(p), Vec2::new(100.0, 150.0));
        assert_eq!(uc2_to_three(p).to_mm(), [150.0, 55.0, 100.0]);
    }

    #[test]
    fn uc2_canvas_uc2_round_trips() {
        let p = Uc2Point::new(12.5, -37.25, 55.0);
        let back = canvas_to_uc2(uc2_to_canvas(p), p.z);
        assert!((back.x - p.x).abs() < 1e-12);
        assert!((back.y - p.y).abs() < 1e-12);
        assert!((back.z - p.z).abs() < 1e-12);
    }

    #[test]
    fn uc2_three_uc2_round_trips() {
        for p in [
            Uc2Point::new(100.0, 150.0, 55.0),
            Uc2Point::new(-3.5, 0.0, 12.25),
            Uc2Point::ZERO,
        ] {
            let back = three_to_uc2(uc2_to_three(p));
            assert!((back.x - p.x).abs() < 1e-12, "{p:?}");
            assert!((back.y - p.y).abs() < 1e-12, "{p:?}");
            assert!((back.z - p.z).abs() < 1e-12, "{p:?}");
        }
    }

    #[test]
    fn the_three_mapping_is_a_relabelling_not_a_reflection() {
        // Both frames are right-handed, so the mapping must preserve handedness.
        // A cyclic axis permutation does; swapping two axes would not, and would
        // silently mirror every module.
        let x = uc2_to_three(Uc2Point::new(1.0, 0.0, 0.0)).to_mm();
        let y = uc2_to_three(Uc2Point::new(0.0, 1.0, 0.0)).to_mm();
        let z = uc2_to_three(Uc2Point::new(0.0, 0.0, 1.0)).to_mm();
        // det[x y z] = +1 for a right-handed relabelling.
        let det = x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
            + x[2] * (y[0] * z[1] - y[1] * z[0]);
        assert!((det - 1.0).abs() < 1e-12, "handedness flipped: det = {det}");
    }

    #[test]
    fn quadrant_angles_are_exact() {
        use std::f64::consts::{FRAC_PI_2, PI};
        assert_eq!(Quadrant::R0.radians(), 0.0);
        assert_eq!(Quadrant::R90.radians(), FRAC_PI_2);
        assert_eq!(Quadrant::R180.radians(), PI);
        assert_eq!(Quadrant::R270.radians(), 3.0 * FRAC_PI_2);
    }

    #[test]
    fn an_out_of_range_rotation_is_rejected_not_wrapped() {
        // A setup asking for rotation 7 is malformed; silently reading it as 3
        // would place the module somewhere the author never wrote.
        assert_eq!(Quadrant::from_quarter_turns(4), Err(InvalidQuadrant(4)));
        assert_eq!(Quadrant::from_quarter_turns(7), Err(InvalidQuadrant(7)));
        assert_eq!(Quadrant::from_quarter_turns(3), Ok(Quadrant::R270));
    }

    #[test]
    fn cell_rotation_is_exact_and_cyclic() {
        let cell = [2, 1, 0];
        assert_eq!(Quadrant::R0.rotate_cell(cell), [2, 1, 0]);
        assert_eq!(Quadrant::R90.rotate_cell(cell), [-1, 2, 0]);
        assert_eq!(Quadrant::R180.rotate_cell(cell), [-2, -1, 0]);
        assert_eq!(Quadrant::R270.rotate_cell(cell), [1, -2, 0]);
        // Four quarter turns is the identity, exactly.
        let mut c = cell;
        for _ in 0..4 {
            c = Quadrant::R90.rotate_cell(c);
        }
        assert_eq!(c, cell);
    }

    #[test]
    fn cell_rotation_keeps_a_module_in_its_layer() {
        // A module turns on its layer; it does not rotate out of it.
        for q in [Quadrant::R0, Quadrant::R90, Quadrant::R180, Quadrant::R270] {
            assert_eq!(q.rotate_cell([1, 2, 3])[2], 3);
        }
    }

    #[test]
    fn cell_rotation_agrees_with_the_canvas_rotation() {
        // The two must not drift: rotating a cell index and rotating that cell's
        // centre in Canvas have to land in the same place, or the footprint and
        // the optics come apart.
        let grid = GridProfile::openuc2_core_50();
        for q in [Quadrant::R0, Quadrant::R90, Quadrant::R180, Quadrant::R270] {
            for cell in [[1, 0, 0], [2, 3, 0], [-1, 2, 0]] {
                let by_cell = uc2_to_canvas(grid.cell_origin(q.rotate_cell(cell)));
                let by_angle =
                    Isometry2::new(Vec2::ZERO, q.radians()).transform_point(uc2_to_canvas(grid.cell_origin(cell)));
                assert!(
                    (by_cell - by_angle).length() < 1e-12,
                    "{q:?} {cell:?}: {by_cell:?} vs {by_angle:?}"
                );
            }
        }
    }

    #[test]
    fn cell_origin_uses_pitch_and_the_mounting_plane() {
        let grid = GridProfile::openuc2_core_50();
        assert_eq!(grid.cell_origin([0, 0, 0]), Uc2Point::new(0.0, 0.0, 5.0));
        assert_eq!(grid.cell_origin([2, 1, 0]), Uc2Point::new(100.0, 50.0, 5.0));
        // The layer pitch (55) is not the in-plane pitch (50), and the mounting
        // plane offset rides on top of it.
        assert_eq!(grid.cell_origin([0, 0, 1]), Uc2Point::new(0.0, 0.0, 60.0));
        assert_eq!(grid.cell_origin([0, 0, 2]), Uc2Point::new(0.0, 0.0, 115.0));
    }

    #[test]
    fn a_point_snaps_to_the_cell_it_is_in() {
        let grid = GridProfile::openuc2_core_50();
        assert_eq!(grid.cell_at(Vec2::new(0.0, 0.0), 0), [0, 0, 0]);
        // Anywhere inside the cell picks the cell: its centre is the origin, so
        // it spans -25..25.
        assert_eq!(grid.cell_at(Vec2::new(24.9, -24.9), 0), [0, 0, 0]);
        assert_eq!(grid.cell_at(Vec2::new(100.0, 50.0), 0), [2, 1, 0]);
        assert_eq!(grid.cell_at(Vec2::new(-100.0, -50.0), 3), [-2, -1, 3]);
    }

    #[test]
    fn snapping_is_the_inverse_of_the_cell_origin() {
        // The round trip that makes a drag land where it looks like it will.
        let grid = GridProfile::openuc2_core_50();
        for cell in [[0, 0, 0], [3, -2, 0], [-7, 11, 2]] {
            let o = grid.cell_origin(cell);
            let back = grid.cell_at(uc2_to_canvas(o), cell[2]);
            assert_eq!(back, cell);
        }
    }

    #[test]
    fn a_non_finite_point_snaps_to_the_origin_rather_than_panicking() {
        // A pointer event cannot produce one, but an `as i32` cast on a NaN is a
        // silent 0 anyway; being explicit costs nothing.
        let grid = GridProfile::openuc2_core_50();
        assert_eq!(grid.cell_at(Vec2::new(f64::NAN, 0.0), 0), [0, 0, 0]);
        assert_eq!(grid.cell_at(Vec2::new(f64::INFINITY, 0.0), 0), [0, 0, 0]);
    }

    #[test]
    fn placement_puts_the_module_at_its_anchor_cell() {
        let grid = GridProfile::openuc2_core_50();
        let t = grid.placement([2, 1, 0], Quadrant::R90);
        assert_eq!(t.pos, Vec2::new(100.0, 50.0));
        assert_eq!(t.rot, std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn the_default_profile_allows_only_quarter_turns() {
        let grid = GridProfile::openuc2_core_50();
        for q in [Quadrant::R0, Quadrant::R90, Quadrant::R180, Quadrant::R270] {
            assert!(grid.allows(q));
        }
    }
}
