//! The canonical Canvas world frame, and the only place its axes are swapped.
//!
//! Coordinate ambiguity is a release blocker: the sketch, the 3D preview, the
//! Optiland handoff, and the rays all have to land in the same place. So every
//! mapping between the Canvas frame and an external one lives here as a named
//! function, and **nothing anywhere else is allowed to reorder an axis inline**
//! (see `docs/architecture/coordinates.md`).
//!
//! # The canonical frame
//!
//! Right-handed, millimetres, `X × Y = Z`:
//!
//! | Axis | Meaning |
//! |---|---|
//! | `+X` | nominal forward optical direction; the Canvas horizontal axis |
//! | `+Y` | the Canvas vertical/lateral axis |
//! | `+Z` | out of the Canvas plane; the openUC2 vertical/layer direction |
//!
//! The default Canvas view is the plane `Z = 0`, screen horizontal `+X`, screen
//! vertical `+Y`. This is the frame `oc-openuc2` already calls canonical, so
//! openUC2's `+X_uc2/+Y_uc2/+Z_uc2` and Canvas world `X/Y/Z` are the same axes
//! under different names.
//!
//! # Status
//!
//! **Stub.** PR 3D-00 fixes the contract and the tests; PR 3D-01 implements the
//! bodies. Everything here is behind the off-by-default `scene3` feature, so the
//! default workspace neither builds nor runs it.

use glam::DVec3 as Vec3;

/// Canvas → Optiland.
///
/// Optiland's nominal optical axis is `+Z` where Canvas uses `+X`, so the frames
/// differ by a rotation:
///
/// ```text
/// optiland.x = -canvas.z
/// optiland.y =  canvas.y
/// optiland.z =  canvas.x
/// ```
///
/// This is a signed axis permutation with determinant `+1`, so it is a rotation
/// and not a reflection: a right-handed system stays right-handed, and a chiral
/// layout does not silently mirror on the way to Optiland. The negation on `z`
/// is what buys that. Mapping `canvas.z` to `optiland.x` unsigned would swap two
/// axes and flip handedness.
///
/// It is also the mapping the repository already assumes everywhere the two
/// engines are compared: `canvas +x ↔ optiland +z`, transverse `y ↔ y`
/// ([`crate::seq`], `validation/`). Extending it to three dimensions had to
/// preserve that, or every existing cross-validation number would move.
///
/// The map is linear and orthogonal, so it applies unchanged to **points,
/// directions, and normals** alike. Points are additionally in millimetres on
/// both sides, so there is no scale factor.
pub fn canvas_to_optiland(canvas: Vec3) -> Vec3 {
    let _ = canvas;
    unimplemented!("PR 3D-01 implements the Canvas → Optiland mapping")
}

/// Optiland → Canvas, the exact inverse of [`canvas_to_optiland`]:
///
/// ```text
/// canvas.x =  optiland.z
/// canvas.y =  optiland.y
/// canvas.z = -optiland.x
/// ```
pub fn optiland_to_canvas(optiland: Vec3) -> Vec3 {
    let _ = optiland;
    unimplemented!("PR 3D-01 implements the Optiland → Canvas mapping")
}

/// Canvas → Three.js.
///
/// Three.js keeps its own convention (`+Y` up, `+Z` nominally forward), which is
/// a relabelling of the same right-handed frame:
///
/// ```text
/// three.x = canvas.y
/// three.y = canvas.z
/// three.z = canvas.x
/// ```
///
/// This is the mapping `oc-openuc2` already ships as `uc2_to_three`, unchanged.
/// It is a cyclic permutation, so it too has determinant `+1`.
pub fn canvas_to_three(canvas: Vec3) -> Vec3 {
    let _ = canvas;
    unimplemented!("PR 3D-01 implements the Canvas → Three.js mapping")
}

/// Three.js → Canvas, the exact inverse of [`canvas_to_three`]:
///
/// ```text
/// canvas.x = three.z
/// canvas.y = three.x
/// canvas.z = three.y
/// ```
pub fn three_to_canvas(three: Vec3) -> Vec3 {
    let _ = three;
    unimplemented!("PR 3D-01 implements the Three.js → Canvas mapping")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Determinant of the matrix whose columns are the images of the basis
    /// vectors. `+1` is a rotation, `-1` is a reflection.
    fn determinant(map: impl Fn(Vec3) -> Vec3) -> f64 {
        let x = map(Vec3::X);
        let y = map(Vec3::Y);
        let z = map(Vec3::Z);
        x.dot(y.cross(z))
    }

    const SAMPLES: [Vec3; 5] = [
        Vec3::ZERO,
        Vec3::new(100.0, 150.0, 55.0),
        Vec3::new(-3.5, 0.0, 12.25),
        Vec3::new(1e-3, -1e4, 7.0),
        Vec3::new(0.1, 0.2, 0.3),
    ];

    #[test]
    fn canvas_to_optiland_maps_the_axes_as_specified() {
        assert_eq!(canvas_to_optiland(Vec3::X), Vec3::Z);
        assert_eq!(canvas_to_optiland(Vec3::Y), Vec3::Y);
        assert_eq!(canvas_to_optiland(Vec3::Z), -Vec3::X);
        assert_eq!(
            canvas_to_optiland(Vec3::new(2.0, 3.0, 5.0)),
            Vec3::new(-5.0, 3.0, 2.0)
        );
    }

    #[test]
    fn the_optiland_mapping_keeps_the_meridional_relationship() {
        // `canvas +x ↔ optiland +z`, transverse `y ↔ y` is not a new choice: it
        // is what `seq.rs` and the whole `validation/` corpus already assume. If
        // the 3D extension broke it, every committed cross-validation number
        // would move, and the sequential handoff would start mirroring designs.
        assert_eq!(canvas_to_optiland(Vec3::X), Vec3::Z);
        assert_eq!(canvas_to_optiland(Vec3::Y), Vec3::Y);
    }

    #[test]
    fn canvas_to_three_maps_the_axes_as_specified() {
        assert_eq!(canvas_to_three(Vec3::X), Vec3::Z);
        assert_eq!(canvas_to_three(Vec3::Y), Vec3::X);
        assert_eq!(canvas_to_three(Vec3::Z), Vec3::Y);
        // The canonical openUC2 point (§6.5 of the openUC2 spec), which
        // `oc-openuc2` already pins to Three.js `(150, 55, 100)`. The two
        // modules must not drift once 3D-01 makes openUC2 call through here.
        assert_eq!(
            canvas_to_three(Vec3::new(100.0, 150.0, 55.0)),
            Vec3::new(150.0, 55.0, 100.0)
        );
    }

    #[test]
    fn both_mappings_are_rotations_not_reflections() {
        // A reflection would mirror every chiral layout on the way out and look
        // entirely plausible doing it. Handedness is checked, never assumed.
        assert_eq!(determinant(canvas_to_optiland), 1.0);
        assert_eq!(determinant(canvas_to_three), 1.0);
        assert_eq!(determinant(optiland_to_canvas), 1.0);
        assert_eq!(determinant(three_to_canvas), 1.0);
    }

    // The round trips are asserted *exactly*, not within a tolerance. Both maps
    // are signed axis permutations, so in f64 they are exact by construction:
    // the only arithmetic is a sign flip. A tolerance here would hide an
    // implementation that had stopped being a permutation.

    #[test]
    fn canvas_optiland_canvas_round_trips_exactly() {
        for p in SAMPLES {
            assert_eq!(optiland_to_canvas(canvas_to_optiland(p)), p, "{p:?}");
        }
    }

    #[test]
    fn optiland_canvas_optiland_round_trips_exactly() {
        for p in SAMPLES {
            assert_eq!(canvas_to_optiland(optiland_to_canvas(p)), p, "{p:?}");
        }
    }

    #[test]
    fn canvas_three_canvas_round_trips_exactly() {
        for p in SAMPLES {
            assert_eq!(three_to_canvas(canvas_to_three(p)), p, "{p:?}");
        }
    }

    #[test]
    fn three_canvas_three_round_trips_exactly() {
        for p in SAMPLES {
            assert_eq!(canvas_to_three(three_to_canvas(p)), p, "{p:?}");
        }
    }

    #[test]
    fn the_mappings_preserve_length_and_angle() {
        // Orthogonality is what lets one map serve points, directions, and
        // normals alike: a unit normal must arrive as a unit normal, and an
        // incidence angle must survive the trip to Optiland unchanged.
        //
        // Tolerance, not equality, and only here: permuting the components
        // reorders the sums inside `dot`, and f64 addition is not associative.
        // The claim under test is orthogonality, not a bit pattern.
        let a = Vec3::new(0.6, -0.8, 0.0);
        let b = Vec3::new(1.0, 2.0, -3.0);
        for map in [
            canvas_to_optiland as fn(Vec3) -> Vec3,
            optiland_to_canvas,
            canvas_to_three,
            three_to_canvas,
        ] {
            assert!((map(a).length() - a.length()).abs() < 1e-15);
            assert!((map(a).dot(map(b)) - a.dot(b)).abs() < 1e-15);
        }
    }
}
