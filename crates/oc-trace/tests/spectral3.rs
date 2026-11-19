//! A detector's hits broken out by wavelength: the sums its colour
//! diagnostics come from.
//!
//! Each spectral line keeps flux-weighted sums of the arrival position and
//! the arrival slope. From those come the line's centroid and spot size, its
//! mean direction and spread, and the plane along the normal where its spot is
//! smallest. The oracle for all of them is the detector itself: move it to the
//! plane the sums predict and trace again, and the spot it records has to be
//! the one predicted, because after the last surface every ray is a straight
//! line and the sums hold everything a straight line needs.
//!
//! The same sums kept per axis give each axis's spot and focus on its own:
//! a cylindrical lens focuses in one plane only, and crossed cylinders of
//! different power focus the face's `u` and `v` at different planes.
#![cfg(feature = "trace3")]

mod common;

use oc_core::id::LayerId;
use oc_core::material::{Material, LAMBDA_D};
use oc_core::profile2::{
    ClosedProfile2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene::Spectrum;
use oc_core::scene3::{Body3, ObjectCommon, Scene3, SolidGeometry3, SurfaceProps3};
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Vec3};
use oc_trace::result3::SPECTRAL_FIELDS;
use oc_trace::{trace_scene3, SpectralSums3, TraceConfig3};

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        ..TraceConfig3::default()
    }
}

/// The default scene (a white collimated beam into a biconvex N-BK7 singlet)
/// with its detector moved `shift` mm down the axis, the way light travels.
fn scene_shifted(shift: f64) -> Scene3 {
    let mut s = Scene3::default_scene();
    s.detectors[0].common.transform.pos.x += shift;
    s
}

#[test]
fn the_lines_add_up_to_the_detector() {
    let r = trace_scene3(&scene_shifted(0.0), &cfg());
    let d = &r.detectors[0];
    assert!(d.spectral.len() > 1, "a white beam has more than one line");
    let flux: f64 = d.spectral.iter().map(|l| l.flux).sum();
    let hits: u64 = d.spectral.iter().map(|l| l.hits).sum();
    assert!((flux - d.total_incident).abs() <= 1e-12 * d.total_incident);
    assert_eq!(hits, d.hits);
    assert_eq!(d.spectral_untracked_flux, 0.0);
    assert!(
        d.spectral
            .windows(2)
            .all(|w| w[0].lambda_um < w[1].lambda_um),
        "in order of wavelength"
    );
}

/// Every line focuses where its sums say, and blue before red.
#[test]
fn each_line_focuses_where_its_sums_say() {
    let at0 = trace_scene3(&scene_shifted(0.0), &cfg());
    let lines = &at0.detectors[0].spectral;
    let mut foci = Vec::new();
    for line in lines {
        let s = line.best_focus_mm().expect("a converging beam has a focus");
        // The spot the sums predict at `s`: each ray's position there is
        // `u + s·a`, so its variance is a quadratic in `s`.
        let w = line.flux;
        let var = |p: f64, pp: f64| pp / w - (p / w) * (p / w);
        let cov = |p: f64, q: f64, pq: f64| pq / w - (p / w) * (q / w);
        let predicted = (var(line.u, line.uu)
            + var(line.v, line.vv)
            + 2.0 * s * (cov(line.u, line.a, line.ua) + cov(line.v, line.b, line.vb))
            + s * s * (var(line.a, line.aa) + var(line.b, line.bb)))
        .max(0.0)
        .sqrt();

        // Move the detector there and look.
        let rms_at = |shift: f64| {
            let r = trace_scene3(&scene_shifted(shift), &cfg());
            let l = r.detectors[0]
                .spectral
                .iter()
                .find(|x| x.lambda_um == line.lambda_um)
                .copied()
                .unwrap();
            assert_eq!(l.hits, line.hits, "every ray still lands at {shift} mm");
            l.rms_radius().unwrap()
        };
        let there = rms_at(s);
        assert!(
            (there - predicted).abs() < 1e-9,
            "{} nm: the sums predict {predicted} mm at {s} mm, the detector reads {there}",
            line.mean_lambda_um() * 1e3
        );
        // And it is the smallest spot along the axis.
        for d in [-0.3, 0.3] {
            assert!(
                rms_at(s + d) > there,
                "{} nm: smaller at {} mm",
                line.mean_lambda_um() * 1e3,
                s + d
            );
        }
        foci.push(s);
    }
    // A positive singlet brings the shorter wavelengths to focus first.
    assert!(
        foci.windows(2).all(|w| w[0] < w[1]),
        "focus grows with wavelength: {foci:?}"
    );
}

/// A collimated beam onto a detector: every line's rays are parallel, so no
/// line has a focus, and the mean angle is the beam's.
#[test]
fn a_collimated_beam_has_no_focus_and_the_beams_angle() {
    let mut s = Scene3::default_scene();
    s.bodies.clear();
    let tilt = 0.02_f64;
    s.sources[0].angular = oc_core::scene3::AngularDistribution3::Direction {
        local_dir: oc_core::vec3::Vec3::new(tilt.cos(), tilt.sin(), 0.0),
    };
    let r = trace_scene3(&s, &cfg());
    for line in &r.detectors[0].spectral {
        assert_eq!(line.best_focus_mm(), None);
        assert!(line.rms_slope().unwrap() < 1e-12);
        let a = line.mean_slope().unwrap();
        let angle = a.x.atan().abs().max(a.y.atan().abs());
        assert!(
            (angle - tilt).abs() < 1e-12,
            "{angle} for a {tilt} rad beam"
        );
    }
}

// ---------------------------------------------------------------------------
// Per axis
// ---------------------------------------------------------------------------

/// The sums of rays that are straight lines through a line focus at `fu` mm
/// in `u` and another at `fv` mm in `v`, laid on a grid of slopes: a ray of
/// slopes `(a, b)` lands at `(c + a·(−fu), d + b·(−fv))`.
fn astigmatic_sums(fu: f64, fv: f64) -> SpectralSums3 {
    let (cu, cv) = (0.7, -0.4);
    let mut x = [0.0; SPECTRAL_FIELDS];
    x[0] = LAMBDA_D;
    for i in 0..9 {
        for j in 0..7 {
            let a = 0.01 * (i as f64 - 4.0) + 0.002;
            let b = 0.03 * (j as f64 - 3.0) - 0.001;
            let (u, v) = (cu - fu * a, cv - fv * b);
            let w = 1.0 + 0.1 * ((i + 2 * j) % 3) as f64;
            for (k, t) in [
                (1, w * LAMBDA_D),
                (2, 1.0),
                (3, w),
                (4, w * u),
                (5, w * v),
                (6, w * u * u),
                (7, w * v * v),
                (8, w * a),
                (9, w * b),
                (10, w * a * a),
                (11, w * b * b),
                (12, w * u * a),
                (13, w * v * b),
            ] {
                x[k] += t;
            }
        }
    }
    SpectralSums3::from_array(&x)
}

/// Rays through two line foci: each axis finds its own, the one focus is their
/// mean weighted by each axis's slope variance, and the spot sizes add in
/// quadrature to the radius.
#[test]
fn each_axis_finds_its_own_line_focus() {
    let (fu, fv) = (12.5, -3.25);
    let l = astigmatic_sums(fu, fv);
    let [su, sv] = l.best_focus_uv_mm().unwrap().map(Option::unwrap);
    assert!((su - fu).abs() < 1e-9, "u focus {su}, built at {fu}");
    assert!((sv - fv).abs() < 1e-9, "v focus {sv}, built at {fv}");

    let w = l.flux;
    let var = |p: f64, pp: f64| pp / w - (p / w) * (p / w);
    let (va, vb) = (var(l.a, l.aa), var(l.b, l.bb));
    let mean = (va * su + vb * sv) / (va + vb);
    let one = l.best_focus_mm().unwrap();
    assert!((one - mean).abs() < 1e-9, "{one} against the weighted {mean}");
    assert!((one - fu).abs() > 1.0 && (one - fv).abs() > 1.0, "the one focus is at neither");

    let [ru, rv] = l.rms_uv().unwrap();
    let r = l.rms_radius().unwrap();
    assert!((ru.hypot(rv) - r).abs() < 1e-12 * r, "hypot({ru}, {rv}) against {r}");
}

/// Rays parallel in one plane: that axis has no focus, the other has.
#[test]
fn an_axis_of_parallel_rays_has_no_focus() {
    let mut x = astigmatic_sums(8.0, 0.0).to_array();
    // Every ray's `b` the same: take its spread out of the sums.
    let (w, b) = (x[3], 0.02);
    x[9] = w * b;
    x[11] = w * b * b;
    x[13] = x[5] * b;
    let l = SpectralSums3::from_array(&x);
    let [su, sv] = l.best_focus_uv_mm().unwrap();
    assert!((su.unwrap() - 8.0).abs() < 1e-9);
    assert_eq!(sv, None);
    // With one axis parallel, the one focus is the other axis's.
    assert!((l.best_focus_mm().unwrap() - 8.0).abs() < 1e-9);
}

/// The cylinders' glass, and its index at the d line the beam is traced at,
/// as the tracer reads it.
const ND: f64 = 1.5168;
fn n_d() -> f64 {
    Material::AbbePoint { nd: ND, vd: 64.0 }
        .index_at(LAMBDA_D, &Scene3::new().catalog)
        .unwrap()
}

/// A plano-convex cylindrical lens of focal length `f_mm` and centre
/// thickness `t_mm`: its curved face at `x_front` toward a beam along +X, its
/// flat face behind. Curved in the plane of X and `power_axis` (Y or Z),
/// straight along the third.
fn add_cylinder(s: &mut Scene3, x_front: f64, f_mm: f64, t_mm: f64, power_axis: Vec3) {
    let r = f_mm * (n_d() - 1.0);
    let h = 6.0_f64; // half height of the lens
    let alpha = (h / r).asin();
    let centre = Vec2::new(r, 0.0);
    let on_arc = |angle: f64| Vec2::new(centre.x + r * angle.cos(), centre.y + r * angle.sin());
    let (top, bottom) = (on_arc(std::f64::consts::PI - alpha), on_arc(std::f64::consts::PI + alpha));
    let curves = [
        Curve2::Arc {
            center: centre,
            radius_mm: r,
            a0: std::f64::consts::PI - alpha,
            a1: std::f64::consts::PI + alpha,
        },
        Curve2::Line { a: bottom, b: Vec2::new(t_mm, -h) },
        Curve2::Line { a: Vec2::new(t_mm, -h), b: Vec2::new(t_mm, h) },
        Curve2::Line { a: Vec2::new(t_mm, h), b: top },
    ];
    let edges = curves
        .into_iter()
        .map(|curve| ProfileEdge2 {
            patch: s.alloc_patch_id(),
            curve,
        })
        .collect();
    // The profile is drawn in local XY and extruded along local Z; a quarter
    // turn about X carries its curved plane from XY onto XZ.
    let turn = if power_axis == Vec3::Z {
        std::f64::consts::FRAC_PI_2
    } else {
        0.0
    };
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::from_axis_angle(Vec3::X, turn).unwrap();
    common.transform.pos = Vec3::new(x_front, 0.0, 0.0);
    s.bodies.push(Body3 {
        common,
        geometry: SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            construction: ProfileConstruction3::ExtrudeLocalZ {
                z_min_mm: -h,
                z_max_mm: h,
            },
        }),
        material: Material::AbbePoint { nd: ND, vd: 64.0 },
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
}

/// A square collimated beam, 3 mm across, of 21 × 21 rays along +X at the d
/// line, onto crossed cylinders: `f = 100` mm curved in Y at `x = 0..4`, and
/// `f = 50` mm curved in Z at `x = 10..14`. A detector facing back down the
/// axis at `x = det_x`.
fn crossed_cylinders(det_x: f64) -> Scene3 {
    let mut s = Scene3::new();
    s.add_collimated_beam(
        Vec3::new(-20.0, 0.0, 0.0),
        1.5,
        1.5,
        21 * 21,
        Spectrum::Monochromatic(LAMBDA_D),
    );
    add_cylinder(&mut s, 0.0, 100.0, 4.0, Vec3::Y);
    add_cylinder(&mut s, 10.0, 50.0, 4.0, Vec3::Z);
    common::add_rect_detector(&mut s, Vec3::new(det_x, 0.0, 0.0), -Vec3::X, 10.0, 10.0, [8, 8]);
    s
}

/// Crossed cylinders focus Y and Z at the planes paraxial optics puts them.
///
/// Along Y: the first lens's back focus, `f − t/n` past its flat face, pushed
/// on by `t(1 − 1/n)` through the second lens, a slab in that plane, so
/// `x = 108 − 8/n`. Along Z: the second lens's alone, `x = 14 + 50 − 4/n`.
/// The per-axis offsets land on these within 0.1 mm: both read 0.032 mm
/// short, the lenses' spherical aberration, which a beam half as wide cuts to
/// a quarter (0.008 mm). The face's `u` is the detector's local X; facing −X, that is
/// world Z, and `v` is world Y. Moving the detector to each predicted plane,
/// that axis's spot is the one its sums predict, to 1e-9 mm, and is wider
/// 0.3 mm either side, and the one focus falls between the two.
#[test]
fn crossed_cylinders_focus_each_axis_at_its_own_plane() {
    let det_x = 40.0;
    let r = trace_scene3(&crossed_cylinders(det_x), &cfg());
    let d = &r.detectors[0];
    assert_eq!(d.spectral.len(), 1);
    let line = d.spectral[0];
    assert_eq!(line.hits, 21 * 21, "every ray lands");

    // Which world axis each of the face's axes is.
    let q = crossed_cylinders(det_x).detectors[0].common.transform;
    let (u_axis, v_axis) = (q.rotate(Vec3::X), q.rotate(Vec3::Y));
    assert!((u_axis.z.abs() - 1.0).abs() < 1e-12, "u along world Z: {u_axis:?}");
    assert!((v_axis.y.abs() - 1.0).abs() < 1e-12, "v along world Y: {v_axis:?}");

    let n = n_d();
    let y_focus = 108.0 - 8.0 / n;
    let z_focus = 14.0 + 50.0 - 4.0 / n;
    let [su, sv] = line.best_focus_uv_mm().unwrap().map(Option::unwrap);
    assert!(
        (det_x + su - z_focus).abs() < 0.1,
        "u (world Z) focuses at x = {}, paraxially {z_focus}",
        det_x + su
    );
    assert!(
        (det_x + sv - y_focus).abs() < 0.1,
        "v (world Y) focuses at x = {}, paraxially {y_focus}",
        det_x + sv
    );
    let one = line.best_focus_mm().unwrap();
    assert!(su < one && one < sv, "the one focus {one} lies between {su} and {sv}");

    // The spot along an axis `s` mm on, from the sums: `var u + 2s·cov(u,a) + s²·var a`.
    let w = line.flux;
    let var = |p: f64, pp: f64| pp / w - (p / w) * (p / w);
    let cov = |p: f64, q: f64, pq: f64| pq / w - (p / w) * (q / w);
    let predict = |s: f64| {
        [
            (var(line.u, line.uu) + 2.0 * s * cov(line.u, line.a, line.ua) + s * s * var(line.a, line.aa))
                .max(0.0)
                .sqrt(),
            (var(line.v, line.vv) + 2.0 * s * cov(line.v, line.b, line.vb) + s * s * var(line.b, line.bb))
                .max(0.0)
                .sqrt(),
        ]
    };
    let rms_uv_at = |s: f64| {
        let r = trace_scene3(&crossed_cylinders(det_x + s), &cfg());
        let l = r.detectors[0].spectral[0];
        assert_eq!(l.hits, line.hits, "every ray still lands {s} mm on");
        l.rms_uv().unwrap()
    };
    for (k, s) in [(0, su), (1, sv)] {
        let there = rms_uv_at(s)[k];
        let predicted = predict(s)[k];
        assert!(
            (there - predicted).abs() < 1e-9,
            "axis {k}: the sums predict {predicted} mm at {s} mm, the detector reads {there}"
        );
        for ds in [-0.3, 0.3] {
            assert!(rms_uv_at(s + ds)[k] > there, "axis {k}: narrower at {} mm", s + ds);
        }
    }
}

/// One cylinder: the plane it does not curve in keeps the beam's width and
/// has no focus, while the one focus is the curved plane's and the radius
/// hides which axis it belongs to.
#[test]
fn one_cylinder_focuses_one_axis() {
    let mut s = Scene3::new();
    s.add_collimated_beam(
        Vec3::new(-20.0, 0.0, 0.0),
        1.5,
        1.5,
        21 * 21,
        Spectrum::Monochromatic(LAMBDA_D),
    );
    add_cylinder(&mut s, 0.0, 100.0, 4.0, Vec3::Y);
    common::add_rect_detector(&mut s, Vec3::new(60.0, 0.0, 0.0), -Vec3::X, 10.0, 10.0, [8, 8]);
    let r = trace_scene3(&s, &cfg());
    let line = r.detectors[0].spectral[0];
    let [su, sv] = line.best_focus_uv_mm().unwrap();
    assert_eq!(su, None, "parallel in Z: no focus along u");
    let sv = sv.unwrap();
    assert!((60.0 + sv - (104.0 - 4.0 / n_d())).abs() < 0.1, "v focuses at x = {}", 60.0 + sv);
    assert!((line.best_focus_mm().unwrap() - sv).abs() < 1e-9);
    // A grid of 21 across 3 mm: its RMS width, `1.5·√((1 − 1/21²)/3)`, kept.
    let [ru, rv] = line.rms_uv().unwrap();
    let grid = 1.5 * ((1.0 - 1.0 / 441.0) / 3.0_f64).sqrt();
    assert!((ru - grid).abs() < 1e-9, "u keeps the beam's {grid} mm: {ru}");
    assert!(rv < ru, "v has narrowed toward its focus: {rv}");
    let rr = line.rms_radius().unwrap();
    assert!((ru.hypot(rv) - rr).abs() < 1e-12 * rr);
}
