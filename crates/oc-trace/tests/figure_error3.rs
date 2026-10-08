//! Slope error in the f64 reference: a mirror with a Gaussian slope error of
//! `σ` per axis spreads a reflected parallel beam by `2σ` per axis, the same
//! way on every run, and a mirror drawn true reflects it exactly.

#![cfg(feature = "trace3")]

use oc_core::id::{ObjectId, SourceId};
use oc_core::scene3::{FigureError, Scene3};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, LaunchRay3, TraceConfig3};

/// A disc mirror at the origin facing +X, and a 64 x 64 grid of rays coming
/// down -X onto it.
fn mirror(slope_rms_mrad: Option<f64>) -> (Scene3, Vec<LaunchRay3>) {
    let mut s = Scene3::new();
    let id = s
        .add_planar_mirror(Vec3::new(0.0, -40.0, 0.0), Vec3::new(0.0, 40.0, 0.0), 1.0)
        .unwrap();
    if let Some(m) = slope_rms_mrad {
        s.figure_errors.push(FigureError { object: ObjectId(id), slope_rms_mrad: m });
    }
    let mut rays = Vec::new();
    for i in 0..64 {
        for j in 0..64 {
            let (y, z) = (-25.0 + 50.0 * i as f64 / 63.0, -25.0 + 50.0 * j as f64 / 63.0);
            rays.push(LaunchRay3 {
                ray: Ray3::new(Vec3::new(50.0, y, z), -Vec3::X).unwrap(),
                lambda_um: 0.55,
                flux: 1.0,
                source: Some(SourceId(1)),
                pol: None,
            });
        }
    }
    (s, rays)
}

/// The directions of the reflected segments.
fn reflected(s: &Scene3, rays: &[LaunchRay3]) -> Vec<Vec3> {
    let res = trace3(s, rays, &TraceConfig3::default());
    res.segments
        .iter()
        .map(|g| (g.b - g.a).normalize())
        .filter(|d| d.x > 0.0)
        .collect()
}

#[test]
fn a_slope_error_spreads_a_reflected_beam_by_twice_itself() {
    let (s, rays) = mirror(Some(2.0));
    s.validate().unwrap();
    let out = reflected(&s, &rays);
    assert_eq!(out.len(), rays.len());
    let n = out.len() as f64;
    let rms = |f: &dyn Fn(&Vec3) -> f64| (out.iter().map(|d| f(d).powi(2)).sum::<f64>() / n).sqrt();
    let (sy, sz) = (rms(&|d| d.y), rms(&|d| d.z));
    // 2σ = 4 mrad per axis; 4096 draws hold the RMS to about 1%.
    for s in [sy, sz] {
        assert!((s / 4e-3 - 1.0).abs() < 0.04, "rms {sy:.6} {sz:.6}");
    }
    // The same hits draw the same leans.
    assert_eq!(out, reflected(&s, &rays));
}

#[test]
fn a_mirror_drawn_true_reflects_exactly() {
    let (s, rays) = mirror(None);
    for d in reflected(&s, &rays) {
        assert!((d - Vec3::X).length() < 1e-12, "{d:?}");
    }
}

#[test]
fn a_figure_error_names_a_real_object_and_a_sane_slope() {
    let (mut s, _) = mirror(None);
    s.validate().unwrap();
    s.figure_errors.push(FigureError { object: ObjectId(9_999), slope_rms_mrad: 1.0 });
    assert!(s.validate().is_err());
    assert!(mirror(Some(-1.0)).0.validate().is_err());
    assert!(mirror(Some(f64::NAN)).0.validate().is_err());
    assert!(mirror(Some(1.5)).0.validate().is_ok());
}
