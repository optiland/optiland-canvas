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
#![cfg(feature = "trace3")]

use oc_core::scene3::Scene3;
use oc_trace::{trace_scene3, TraceConfig3};

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
