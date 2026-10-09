//! A tube carries light down its bore by reflection, and what each bounce
//! costs is exactly what its finish says.
//!
//! The geometry is checked by counting: a ray entering a mirror-lined bore at a
//! known slope meets the wall a number of times that follows from nothing but
//! the slope, the bore and the length, so the flux out is `R^N` to rounding. A
//! skew ray, which spirals rather than zig-zags, keeps its distance from the
//! axis and hits the wall once per chord of the circle at that distance, which
//! is the same count done in 3D. A metal bore is checked against the transfer
//! matrix at the ray's own angle, which is the reason a light pipe is lined
//! with a coating rather than a number.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::scene3::Scene3;
use oc_core::thinfilm::coating_rt_at;
use oc_core::tube::{BoreFinish, BoreMetal, TubeDims};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, TraceConfig3};

const LAMBDA: f64 = 0.55;

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        record_detector_hits: true,
        ..no_ghosts()
    }
}

/// A tube about the x axis centred on the origin, bore radius `ri`, wall to
/// `ri + 1`, `2 hl` long, and a wide detector at `x = hl + 20` facing back.
fn pipe(ri: f64, hl: f64, bore: BoreFinish) -> (Scene3, u64, oc_core::id::ObjectId) {
    let mut s = Scene3::default();
    let tube = s
        .add_tube(
            Vec3::ZERO,
            0.0,
            TubeDims {
                inner_diameter_mm: 2.0 * ri,
                outer_diameter_mm: 2.0 * ri + 2.0,
                length_mm: 2.0 * hl,
            },
            bore,
        )
        .expect("a tube");
    let det = add_rect_detector(
        &mut s,
        Vec3::new(hl + 20.0, 0.0, 0.0),
        -Vec3::X,
        200.0,
        200.0,
        [4, 4],
    );
    s.validate().expect("the pipe validates");
    (s, tube, det)
}

fn ray(origin: Vec3, dir: Vec3) -> Ray3 {
    Ray3::new(origin, dir.normalize()).expect("a unit direction")
}

/// What fraction of one launched ray reached the detector.
fn delivered(s: &Scene3, det: oc_core::id::ObjectId, r: Ray3) -> f64 {
    let res = trace3(s, &[launch(r, LAMBDA, 1.0)], &cfg());
    assert!(
        res.ledger.residual().abs() < 1e-12,
        "the ledger balances: {:?}",
        res.ledger
    );
    detector_named(&res, det).total_incident / res.ledger.launched
}

/// Down the axis nothing is touched, whatever the bore is made of.
#[test]
fn a_ray_down_the_axis_touches_nothing() {
    let (s, _, det) = pipe(5.0, 50.0, BoreFinish::Black);
    let got = delivered(&s, det, ray(Vec3::new(-60.0, 0.0, 0.0), Vec3::X));
    assert!((got - 1.0).abs() < 1e-12, "all of it arrives: {got}");
}

/// A meridional ray zig-zags: unfolded, it crosses a wall at every odd
/// multiple of the bore radius, so the count is known before tracing.
#[test]
fn a_meridional_ray_pays_one_reflectance_per_bounce() {
    let (ri, hl, r) = (5.0, 50.0, 0.9);
    let (s, _, det) = pipe(ri, hl, BoreFinish::Mirror { reflectivity: r });
    let theta = 15f64.to_radians();
    let start = Vec3::new(-60.0, 0.0, 0.0);
    // Unfolded height at the exit plane, and the walls it crossed on the way.
    let rise = (hl - start.x) * theta.tan();
    let bounces = (0..).take_while(|k| ri * (2 * k + 1) as f64 <= rise).count();
    assert_eq!(bounces, 3, "the geometry the test was written for");
    let got = delivered(&s, det, ray(start, Vec3::new(theta.cos(), theta.sin(), 0.0)));
    let want = r.powi(bounces as i32);
    assert!((got - want).abs() < 1e-12, "R^{bounces} = {want}, got {got}");
}

/// A skew ray spirals. Seen down the axis it runs along chords of the bore,
/// every chord the same length because a reflection off a cylinder keeps the
/// ray's distance from the axis. So the count is the transverse path over the
/// chord, the same arithmetic in a plane that does not contain the axis.
#[test]
fn a_skew_ray_spirals_and_pays_per_chord() {
    let (ri, hl, r) = (5.0, 50.0, 0.8);
    let (s, _, det) = pipe(ri, hl, BoreFinish::Mirror { reflectivity: r });
    let (theta, b) = (15f64.to_radians(), 3.0);
    let start = Vec3::new(-60.0, 0.0, b);
    let half_chord = (ri * ri - b * b).sqrt();
    let travel = (hl - start.x) * theta.tan();
    // First wall at one half chord, then one every full chord.
    let bounces = (0..)
        .take_while(|k| half_chord * (2 * k + 1) as f64 <= travel)
        .count();
    assert_eq!(bounces, 4, "the geometry the test was written for");
    let got = delivered(&s, det, ray(start, Vec3::new(theta.cos(), theta.sin(), 0.0)));
    let want = r.powi(bounces as i32);
    assert!((got - want).abs() < 1e-12, "R^{bounces} = {want}, got {got}");
}

/// One bounce off a silver bore at 60 degrees costs what the transfer matrix
/// says a 150 nm silver film costs at 60 degrees, for unpolarized light.
#[test]
fn a_metal_bore_reflects_by_the_transfer_matrix_at_the_rays_own_angle() {
    for (metal, protected) in [
        (BoreMetal::Silver, false),
        (BoreMetal::Silver, true),
        (BoreMetal::Aluminium, false),
    ] {
        let finish = BoreFinish::Metal { metal, protected };
        let (s, tube, det) = pipe(5.0, 10.0, finish);
        let theta = 30f64.to_radians();
        let got = delivered(
            &s,
            det,
            ray(Vec3::new(-11.0, 0.0, 0.0), Vec3::new(theta.cos(), theta.sin(), 0.0)),
        );
        let bore = s.bodies.iter().find(|b| b.common.id.0 == tube).unwrap();
        let coating = match &bore.surface.overrides[0].response.from_outside {
            oc_core::scene3::SurfaceResponse::Coating { coating } => coating.clone(),
            other => panic!("a metal bore is a coating, not {other:?}"),
        };
        // Angle of incidence on the wall is 90 degrees minus the slope.
        let aoi = std::f64::consts::FRAC_PI_2 - theta;
        let want = coating_rt_at(&coating, 1.0, 1.0, LAMBDA, aoi.cos()).unwrap().r();
        assert!(
            (got - want).abs() < 1e-9,
            "{metal:?} protected={protected}: R(60°) = {want}, got {got}"
        );
        assert!(want > 0.85 && want < 1.0, "a metal mirror: {want}");
    }
}

/// Silver's reflectance is not one number: at grazing incidence it climbs
/// toward one, which is why a long light pipe works at all.
#[test]
fn silver_is_more_reflective_at_grazing_incidence() {
    let c = oc_core::thinfilm::metal_mirror(oc_core::thinfilm::materials::silver(), None);
    let r = |aoi_deg: f64| {
        coating_rt_at(&c, 1.0, 1.0, LAMBDA, aoi_deg.to_radians().cos())
            .unwrap()
            .r()
    };
    let (normal, grazing) = (r(0.0), r(85.0));
    assert!(normal > 0.97 && normal < 0.99, "silver at normal: {normal}");
    assert!(grazing > normal, "grazing {grazing} beats normal {normal}");
}

/// The outside of a tube and its end rings are black.
#[test]
fn the_wall_and_the_end_rings_absorb() {
    let (s, _, det) = pipe(5.0, 50.0, BoreFinish::Mirror { reflectivity: 1.0 });
    // Into the end ring, inside the wall's thickness.
    let end = trace3(
        &s,
        &[launch(ray(Vec3::new(-60.0, 5.5, 0.0), Vec3::X), LAMBDA, 1.0)],
        &cfg(),
    );
    // Down onto the outer wall from above.
    let side = trace3(
        &s,
        &[launch(ray(Vec3::new(0.0, 20.0, 0.0), -Vec3::Y), LAMBDA, 1.0)],
        &cfg(),
    );
    for res in [end, side] {
        assert_eq!(detector_named(&res, det).total_incident, 0.0);
        assert!((res.ledger.absorbed_surface - 1.0).abs() < 1e-12, "{:?}", res.ledger);
    }
}

/// A tube can be turned and resized in place, and its bore refinished,
/// without the scene losing track of which face is the bore.
#[test]
fn a_tube_round_trips_its_dimensions_and_finish() {
    let (mut s, tube, _) = pipe(5.0, 50.0, BoreFinish::Black);
    let dims = TubeDims {
        inner_diameter_mm: 20.0,
        outer_diameter_mm: 24.0,
        length_mm: 300.0,
    };
    assert!(s.set_tube_dims(tube, dims));
    let finish = BoreFinish::Metal {
        metal: BoreMetal::Silver,
        protected: true,
    };
    assert!(s.set_tube_bore(tube, finish.clone()));
    assert_eq!(s.tube_info(tube), Some((dims, Some(finish))));
    // An empty wall is refused and changes nothing.
    let bad = TubeDims {
        outer_diameter_mm: 20.0,
        ..dims
    };
    assert!(!s.set_tube_dims(tube, bad));
    assert_eq!(s.tube_info(tube).unwrap().0, dims);
    // And the document survives a save.
    let json = serde_json::to_string(&s).unwrap();
    let back: Scene3 = serde_json::from_str(&json).unwrap();
    back.validate().expect("the reloaded pipe validates");
    assert_eq!(back.tube_info(tube), s.tube_info(tube));
}

/// A long pipe is walked to its end: a steep ray down a 3 m, 50 mm bore
/// meets the wall sixty times, past the 32 events a bench scene is given,
/// and the depth the scene asks for has to cover it.
#[test]
fn a_long_pipe_is_walked_to_its_end() {
    let (ri, hl, r) = (25.0, 1500.0, 0.98);
    let (s, _, det) = pipe(ri, hl, BoreFinish::Mirror { reflectivity: r });
    let theta = 45f64.to_radians();
    let start = Vec3::new(-hl - 1.0, 0.0, 0.0);
    let rise = (hl - start.x) * theta.tan();
    let bounces = (0..).take_while(|k| ri * (2 * k + 1) as f64 <= rise).count();
    assert_eq!(bounces, 60, "the geometry the test was written for");
    let cfg = TraceConfig3 {
        max_depth: oc_trace::depth_for_scene3(&s, false, 0),
        bounds_mm: oc_trace::bounds_for_scene3(&s),
        ..cfg()
    };
    let res = trace3(
        &s,
        &[launch(ray(start, Vec3::new(theta.cos(), theta.sin(), 0.0)), LAMBDA, 1.0)],
        &cfg,
    );
    assert_eq!(res.ledger.truncated_flux, 0.0, "{:?}", res.ledger);
    let got = detector_named(&res, det).total_incident;
    let want = r.powi(bounces as i32);
    assert!((got - want).abs() < 1e-12, "R^{bounces} = {want}, got {got}");
}
