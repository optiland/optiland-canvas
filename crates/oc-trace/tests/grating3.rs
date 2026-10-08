//! A diffraction grating in the f64 reference: each order leaves where the
//! grating equation puts it, carries the declared share of the flux, and the
//! rest is absorbed at the surface, with the ledger balanced.

#![cfg(feature = "trace3")]

use oc_core::id::SourceId;
use oc_core::scene3::Scene3;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, LaunchRay3, TraceConfig3};

fn launch(origin: Vec3, dir: Vec3, lambda_um: f64) -> LaunchRay3 {
    LaunchRay3 {
        ray: Ray3::new(origin, dir).unwrap(),
        lambda_um,
        flux: 1.0,
        source: Some(SourceId(1)),
        pol: None,
    }
}

/// A 600 lines/mm reflection grating facing +X, grooves along world Z, lit
/// straight on at 0.5 µm: the first order leaves at sin β = 0.3 in the XY
/// plane with 65% of the light, the zero order goes straight back with 10%,
/// and 25% is absorbed.
#[test]
fn a_reflection_grating_sends_each_order_where_the_equation_says() {
    let mut s = Scene3::new();
    s.add_grating(Vec3::ZERO, 20.0, 20.0, 0.0, 600.0, false);
    let beta = 0.3f64.asin();
    let first = Vec3::new(beta.cos(), beta.sin(), 0.0);
    let d1 = s.add_area_detector(first * 100.0, -first, 5.0, 5.0, [1, 1]);
    s.validate().unwrap();
    let rays = [launch(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 0.5)];
    let cfg = TraceConfig3 { record_detector_hits: true, ..TraceConfig3::default() };
    let res = trace3(&s, &rays, &cfg);
    let det = res.detectors.iter().find(|d| d.detector.0 == d1.0).unwrap();
    assert!((det.total_incident - 0.65).abs() < 1e-12, "{}", det.total_incident);
    assert!((res.ledger.absorbed_surface - 0.25).abs() < 1e-12);
    // The zero order goes back along the beam and escapes with its 10%.
    assert!((res.ledger.escaped - 0.10).abs() < 1e-12, "{}", res.ledger.escaped);
    // The first-order ray lands on its detector's centre: the direction is
    // the grating equation's.
    assert_eq!(det.records.len(), 1);
    assert!(det.records[0].uv_mm.length() < 1e-9, "{:?}", det.records[0].uv_mm);
}

/// The directions, for skew rays and both kinds, against the grating
/// equation written out: `dx' = dx + m λ / d` across the grooves, `dy'`
/// kept, and the zero order the mirror or the straight beam.
#[test]
fn skew_rays_keep_their_component_along_the_grooves() {
    for transmissive in [false, true] {
        let mut s = Scene3::new();
        let id = s.add_grating(Vec3::ZERO, 40.0, 40.0, 0.0, 1200.0, transmissive);
        s.validate().unwrap();
        let (lines, lambda) = (1200.0, 0.55);
        let d = Vec3::new(-0.9, 0.2, 0.3).normalize();
        let cfg = TraceConfig3 { record_events: true, ..TraceConfig3::default() };
        let res = trace3(&s, &[launch(Vec3::new(20.0, -4.0, -6.0) - d * 1.0, d, lambda)], &cfg);
        // Local frame of the grating: X across the grooves = world Y, Y along
        // the grooves = world Z, Z the normal = world X.
        let across = d.y + lines * 1e-3 * lambda;
        let along = d.z;
        let nz = (1.0 - across * across - along * along).sqrt();
        let expect = Vec3::new(if transmissive { -nz } else { nz }, across, along);
        let segs = &res.segments;
        let got = segs.iter().skip(1).map(|g| (g.b - g.a).normalize()).collect::<Vec<_>>();
        assert!(
            got.iter().any(|g| (*g - expect).length() < 1e-12),
            "transmissive {transmissive}: {got:?} vs {expect:?} (grating #{id})",
        );
        let zero = if transmissive { d } else { Vec3::new(-d.x, d.y, d.z) };
        assert!(got.iter().any(|g| (*g - zero).length() < 1e-12), "zero order");
    }
}

/// A grating that diffracts 80% of light polarized along its grooves and 40%
/// across them sends 60% of unpolarized light into its order, a third
/// polarized along the grooves; light polarized along them gets the full 80%.
#[test]
fn a_grating_splits_the_polarizations_on_its_grooves() {
    use oc_core::polarization::{PolRay, Stokes};
    use oc_core::scene3::{GratingTm, SpectralScalar, SurfaceResponse};
    let mut s = Scene3::new();
    let id = s.add_grating(Vec3::ZERO, 20.0, 20.0, 0.0, 600.0, false);
    let g = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
    let SurfaceResponse::Grating { efficiency, zero_order, tm, .. } = &mut g.response_front else {
        panic!("a grating")
    };
    *efficiency = SpectralScalar::Constant(0.8);
    *zero_order = SpectralScalar::Constant(0.1);
    *tm = Some(GratingTm {
        efficiency: SpectralScalar::Constant(0.4),
        zero_order: SpectralScalar::Constant(0.3),
    });
    let beta = 0.3f64.asin();
    let first = Vec3::new(beta.cos(), beta.sin(), 0.0);
    let d1 = s.add_area_detector(first * 100.0, -first, 5.0, 5.0, [1, 1]);
    s.validate().unwrap();
    let cfg = TraceConfig3 { record_detector_hits: true, ..TraceConfig3::default() };

    let res = trace3(&s, &[launch(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 0.5)], &cfg);
    let det = res.detectors.iter().find(|d| d.detector.0 == d1.0).unwrap();
    assert!((det.total_incident - 0.6).abs() < 1e-12, "{}", det.total_incident);
    let hit = &det.records[0];
    assert!((hit.stokes.dop() - 1.0 / 3.0).abs() < 1e-12, "{:?}", hit.stokes);
    // Along the grooves (world Z): the state's frame axis lies along them
    // and the excess is on it.
    assert!(hit.frame_x.cross(Vec3::Z).length() < 1e-12, "{:?}", hit.frame_x);
    assert!(hit.stokes.s1 > 0.0);
    // Zero order: (0.1 + 0.3) / 2 escapes back along the beam.
    assert!((res.ledger.escaped - 0.2).abs() < 1e-12, "{}", res.ledger.escaped);

    let mut te = launch(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 0.5);
    te.pol = Some(PolRay { stokes: Stokes { s0: 1.0, s1: 1.0, s2: 0.0, s3: 0.0 }, x: Vec3::Z });
    let res = trace3(&s, &[te], &cfg);
    let det = res.detectors.iter().find(|d| d.detector.0 == d1.0).unwrap();
    assert!((det.total_incident - 0.8).abs() < 1e-12, "{}", det.total_incident);
}
