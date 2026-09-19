//! A detector's face: its size, its pixels, and the area each pixel covers.
//!
//! What this suite owes:
//!
//! - the disc and rectangle overlap is exact, against closed forms and against
//!   an independent numerical integral;
//! - the covered pixel areas of a rectangle, a disc and an annulus sum to the
//!   face's own area, at any pixel count, square or not (the A12 identity's
//!   geometric half; the flux half is in `oc-trace`'s `detectors3`);
//! - pitch and pixel area follow the size and the counts;
//! - the size and pixel setters refuse what does not validate and leave the
//!   scene as it was, and a file cannot ask for a grid that overflows.
#![cfg(feature = "scene3")]

use std::f64::consts::PI;

use oc_core::geom3::primitives::disc_rect_overlap_area as overlap;
use oc_core::id::LayerId;
use oc_core::scene3::{
    grid_fits, Detector3, DetectorGeometry3, DetectorMeasure, DetectorResponse,
    DetectorTermination, ObjectCommon, Scene3, MAX_DETECTOR_CELLS,
};
use oc_core::vec3::Vec3;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

// ---------------------------------------------------------------------------
// The overlap
// ---------------------------------------------------------------------------

#[test]
fn the_disc_rectangle_overlap_matches_its_closed_forms() {
    let r = 2.5;
    let whole = PI * r * r;
    let cases = [
        (
            "the whole disc",
            overlap(r, -10.0, 10.0, -10.0, 10.0),
            whole,
        ),
        // The square the rim touches on all four sides: the arc meets each
        // edge at one point only, which the first version read as the edge.
        ("its bounding square", overlap(r, -r, r, -r, r), whole),
        ("a strip the rim touches", overlap(r, -r, r, -1.0, r), {
            let a = -1.0f64;
            whole - (r * r * (-a / r).acos() + a * (r * r - a * a).sqrt())
        }),
        ("a quarter", overlap(r, 0.0, r, 0.0, r), whole / 4.0),
        ("a half", overlap(r, -r, r, 0.0, 10.0), whole / 2.0),
        (
            "a square well inside",
            overlap(r, -0.3, 0.4, -0.2, 0.5),
            0.7 * 0.7,
        ),
        ("wholly outside", overlap(r, r, 2.0 * r, 0.0, 1.0), 0.0),
        ("touching a corner", overlap(r, 1.5, 3.0, 2.0, 3.0), 0.0),
        ("empty", overlap(r, 1.0, 1.0, 0.0, 1.0), 0.0),
    ];
    for (name, got, want) in cases {
        assert!(close(got, want, 1e-13), "{name}: {got} vs {want}");
    }
    // A strip from x = a to the rim is the circular segment
    // r² acos(a/r) − a √(r² − a²).
    for a in [-2.0, -0.7, 0.0, 0.9, 2.2, 2.49] {
        let got = overlap(r, a, 5.0, -5.0, 5.0);
        let want = r * r * (a / r).acos() - a * (r * r - a * a).sqrt();
        assert!(close(got, want, 1e-13), "segment from {a}: {got} vs {want}");
    }
}

/// The closed form against a different way of computing the same area: the
/// chord at each `x` clipped to the rectangle, summed over a fine midpoint
/// grid. The two share no code, so agreement is agreement about the geometry.
#[test]
fn the_overlap_agrees_with_a_fine_numerical_integral() {
    let r = 1.7;
    let numeric = |x0: f64, x1: f64, y0: f64, y1: f64| {
        let n = 400_000;
        let dx = (x1 - x0) / n as f64;
        (0..n)
            .map(|i| {
                let x = x0 + (i as f64 + 0.5) * dx;
                if x.abs() >= r {
                    return 0.0;
                }
                let h = (r * r - x * x).sqrt();
                (y1.min(h) - y0.max(-h)).max(0.0) * dx
            })
            .sum::<f64>()
    };
    // A fixed pseudo-random spread of rectangles across the rim, so the arc
    // cuts them every way it can.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 4.4 - 2.2
    };
    for _ in 0..60 {
        let (a, b, c, d) = (next(), next(), next(), next());
        let (x0, x1) = (a.min(b), a.max(b));
        let (y0, y1) = (c.min(d), c.max(d));
        let got = overlap(r, x0, x1, y0, y1);
        let want = numeric(x0, x1, y0, y1);
        assert!(
            (got - want).abs() < 1e-7,
            "[{x0:.3}, {x1:.3}] x [{y0:.3}, {y1:.3}]: {got} vs {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// The covered areas
// ---------------------------------------------------------------------------

fn detector(geometry: DetectorGeometry3, bins: [usize; 2]) -> Detector3 {
    let mut s = Scene3::new();
    let id = s.add_area_detector(Vec3::ZERO, Vec3::X, 1.0, 1.0, bins);
    assert!(
        s.set_detector_geometry(id.0, geometry),
        "the face validates"
    );
    s.detectors.remove(0)
}

#[test]
fn a_faces_covered_pixels_sum_to_its_area() {
    let faces = [
        DetectorGeometry3::Rectangle {
            half_width_mm: 3.5,
            half_height_mm: 1.25,
        },
        DetectorGeometry3::Disc { radius_mm: 2.0 },
        DetectorGeometry3::Annulus {
            inner_mm: 0.8,
            outer_mm: 3.0,
        },
    ];
    for face in faces {
        for bins in [[1, 1], [2, 2], [7, 3], [9, 9], [5, 12], [64, 64], [33, 17]] {
            let d = detector(face.clone(), bins);
            let cells = d.covered_cell_areas_mm2().expect("an area detector");
            assert_eq!(cells.len(), bins[0] * bins[1]);
            let pixel = d.pixel_area_mm2().unwrap();
            for (i, a) in cells.iter().enumerate() {
                assert!(
                    *a >= 0.0 && *a <= pixel * (1.0 + 1e-12),
                    "{face:?} {bins:?} pixel {i}: {a} of {pixel}"
                );
            }
            let sum: f64 = cells.iter().sum();
            let face_area = d.face_area_mm2().unwrap();
            assert!(
                close(sum, face_area, 1e-12),
                "{face:?} {bins:?}: {sum} vs {face_area}"
            );
        }
    }
}

#[test]
fn a_disc_cut_in_four_gives_each_pixel_a_quarter() {
    let d = detector(DetectorGeometry3::Disc { radius_mm: 3.0 }, [2, 2]);
    for a in d.covered_cell_areas_mm2().unwrap() {
        assert!(close(a, PI * 9.0 / 4.0, 1e-14), "{a}");
    }
    // The pixels are still the squares of the box.
    assert!(close(d.pixel_area_mm2().unwrap(), 9.0, 1e-15));
}

#[test]
fn the_pitch_follows_the_size_and_the_counts() {
    let d = detector(
        DetectorGeometry3::Rectangle {
            half_width_mm: 4.0,
            half_height_mm: 2.0,
        },
        [16, 4],
    );
    assert_eq!(d.grid_extent_mm(), Some([8.0, 4.0]));
    assert_eq!(d.pixel_pitch_mm(), Some([0.5, 1.0]));
    assert_eq!(d.pixel_area_mm2(), Some(0.5));
    assert_eq!(d.face_area_mm2(), Some(32.0));

    let disc = detector(DetectorGeometry3::Disc { radius_mm: 5.0 }, [20, 10]);
    assert_eq!(disc.grid_extent_mm(), Some([10.0, 10.0]));
    assert_eq!(
        disc.pixel_pitch_mm(),
        Some([0.5, 1.0]),
        "a disc's box is square"
    );
}

// ---------------------------------------------------------------------------
// The setters
// ---------------------------------------------------------------------------

#[test]
fn pixel_counts_are_set_within_the_cap_and_refused_past_it() {
    let mut s = Scene3::new();
    let id = s
        .add_area_detector(Vec3::ZERO, Vec3::X, 2.0, 2.0, [64, 64])
        .0;
    let bins = |s: &Scene3| s.detectors[0].bins;

    assert!(s.set_detector_pixels(id, [128, 32]));
    assert_eq!(bins(&s), [128, 32]);
    assert!(
        s.set_detector_pixels(id, [1024, 1024]),
        "the cap itself is allowed"
    );

    for refused in [[0, 8], [8, 0], [2048, 1024], [MAX_DETECTOR_CELLS + 1, 1]] {
        assert!(!s.set_detector_pixels(id, refused), "{refused:?}");
        assert_eq!(bins(&s), [1024, 1024], "{refused:?} left the scene alone");
    }
    assert!(!s.set_detector_pixels(id + 99, [8, 8]), "no such detector");
    s.validate().expect("still valid");
}

/// A count read from a file is checked before it is multiplied, so an absurd
/// one is an error and not an overflow.
#[test]
fn a_file_cannot_ask_for_a_grid_that_overflows() {
    assert!(!grid_fits([usize::MAX, 2]));
    assert!(grid_fits([1, MAX_DETECTOR_CELLS]));

    let mut s = Scene3::new();
    s.add_area_detector(Vec3::ZERO, Vec3::X, 2.0, 2.0, [8, 8]);
    s.detectors[0].bins = [usize::MAX, 2];
    assert!(s.validate().is_err());

    s.detectors[0].bins = [8, 8];
    s.detectors[0].measure = DetectorMeasure::Intensity {
        half_angle_deg: 30.0,
        bins: [usize::MAX, 3],
    };
    assert!(s.validate().is_err(), "the angular grid is capped too");
}

#[test]
fn a_new_face_keeps_everything_else_and_refuses_a_bad_size() {
    let mut s = Scene3::new();
    let id = s
        .add_area_detector(Vec3::new(5.0, 1.0, 0.0), -Vec3::X, 2.0, 2.0, [16, 8])
        .0;
    s.detectors[0].measure = DetectorMeasure::Irradiance;
    let pose = s.detectors[0].common.transform;

    assert!(s.set_detector_geometry(id, DetectorGeometry3::Disc { radius_mm: 4.0 }));
    let d = &s.detectors[0];
    assert_eq!(d.geometry, DetectorGeometry3::Disc { radius_mm: 4.0 });
    assert_eq!(d.bins, [16, 8], "the pixel counts stay");
    assert_eq!(
        d.measure,
        DetectorMeasure::Irradiance,
        "so does the measure"
    );
    assert_eq!(d.common.transform, pose, "and the pose");

    let refused = [
        DetectorGeometry3::Rectangle {
            half_width_mm: 0.0,
            half_height_mm: 1.0,
        },
        DetectorGeometry3::Rectangle {
            half_width_mm: f64::NAN,
            half_height_mm: 1.0,
        },
        DetectorGeometry3::Disc { radius_mm: -1.0 },
        DetectorGeometry3::Annulus {
            inner_mm: 3.0,
            outer_mm: 3.0,
        },
        DetectorGeometry3::LineProbeLegacy {
            a_local: Vec3::ZERO,
            b_local: Vec3::X,
            bins: 4,
        },
    ];
    for g in refused {
        assert!(!s.set_detector_geometry(id, g.clone()), "{g:?}");
        assert_eq!(
            s.detectors[0].geometry,
            DetectorGeometry3::Disc { radius_mm: 4.0 },
            "{g:?} left the face alone"
        );
    }
}

#[test]
fn a_line_probe_has_no_face_to_resize_or_pixelate() {
    let mut s = Scene3::new();
    let id = s.alloc_object_id();
    s.detectors.push(Detector3 {
        common: ObjectCommon::new(id, LayerId(0)),
        geometry: DetectorGeometry3::LineProbeLegacy {
            a_local: Vec3::ZERO,
            b_local: Vec3::X,
            bins: 8,
        },
        bins: [8, 1],
        response: DetectorResponse::Unity,
        termination: DetectorTermination::Terminate,
        measure: DetectorMeasure::Power,
    });
    s.validate().expect("a line probe is valid");
    assert!(!s.set_detector_pixels(id.0, [4, 4]));
    assert!(!s.set_detector_geometry(id.0, DetectorGeometry3::Disc { radius_mm: 1.0 }));
    let d = &s.detectors[0];
    assert_eq!(d.pixel_pitch_mm(), None);
    assert_eq!(d.covered_cell_areas_mm2(), None);
    assert_eq!(d.face_area_mm2(), None);
}

#[test]
fn only_an_intensity_detector_has_angular_bins_to_set() {
    let mut s = Scene3::new();
    let id = s.add_area_detector(Vec3::ZERO, Vec3::X, 2.0, 2.0, [8, 8]).0;
    assert!(
        !s.set_detector_direction_bins(id, 12, 4),
        "power bins no directions"
    );

    s.detectors[0].measure = DetectorMeasure::Intensity {
        half_angle_deg: 40.0,
        bins: [24, 1],
    };
    assert!(s.set_detector_direction_bins(id, 12, 4));
    assert_eq!(
        s.detectors[0].measure,
        DetectorMeasure::Intensity {
            half_angle_deg: 40.0,
            bins: [12, 4],
        },
        "the cone stays"
    );
    assert!(!s.set_detector_direction_bins(id, 0, 4));
    assert!(!s.set_detector_direction_bins(id, 2048, 1024));
    assert!(matches!(
        s.detectors[0].measure,
        DetectorMeasure::Intensity { bins: [12, 4], .. }
    ));
}
