//! Measured and supplied coating tables in the reference tracer
//! (coating-fidelity spec CF-09, C28, and the scalar rule of the earlier
//! spec's §9.4).
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::fidelity::{FidelityProfile, Numerics};
use oc_core::response_table::{
    AngularTable, PhaseInformation, PowerResolution, ReflectionRole, TableMedia, TableScope, TableSide,
};
use oc_core::scene3::SurfaceResponse;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, CapabilityIssue, TraceConfig3};

fn air() -> TableMedia {
    TableMedia { incident: 1.0, exit: 1.0 }
}

/// A 45-degree vendor filter: R and T at 45 degrees only, both sides alike.
fn filter_45() -> AngularTable {
    let mut t = AngularTable::one_angle(vec![0.45, 0.55, 0.65], 45.0, vec![0.1, 0.3, 0.8], vec![0.85, 0.6, 0.15], air());
    t.side = TableSide::Either;
    t
}

fn table_pair(t: &AngularTable) -> (SurfaceResponse, SurfaceResponse) {
    let copy = |from| SurfaceResponse::MeasuredTable { table: Box::new(t.clone()), from };
    (copy(TableSide::Front), copy(TableSide::Back))
}

/// A plate tilted to 135 degrees at the origin: a beam along +X meets its
/// front at 45 degrees and reflects to +Y. Detectors catch both arms.
fn tilted(t: &AngularTable) -> (Scene3, oc_core::id::ObjectId, oc_core::id::ObjectId) {
    let mut s = Scene3::default();
    let (front, back) = table_pair(t);
    s.add_planar_element(Vec3::ZERO, 20.0, 135.0, front, back);
    let through = add_rect_detector(&mut s, Vec3::new(40.0, 0.0, 0.0), -Vec3::X, 40.0, 40.0, [1, 1]);
    let folded = add_rect_detector(&mut s, Vec3::new(0.0, 40.0, 0.0), -Vec3::Y, 40.0, 40.0, [1, 1]);
    (s, through, folded)
}

fn beam(deg: f64, lambda_um: f64) -> oc_trace::LaunchRay3 {
    let d = Vec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
    launch(Ray3::new(Vec3::new(-30.0, 0.0, 0.0), d).unwrap(), lambda_um, 1.0)
}

fn strict() -> TraceConfig3 {
    TraceConfig3 { fidelity: FidelityProfile { numerics: Numerics::Strict, ..Default::default() }, ..TraceConfig3::default() }
}

/// On its data the table is the surface: each arm gets the table's R or T,
/// and what is neither is absorbed at the surface.
#[test]
fn a_table_splits_the_beam_by_its_data() {
    let t = filter_45();
    let (s, through, folded) = tilted(&t);
    for (i, lam) in [0.45, 0.5, 0.55, 0.6, 0.65].into_iter().enumerate() {
        let res = trace3(&s, &[beam(0.0, lam)], &strict());
        assert!(res.capability_report.rejected.is_empty(), "{:?}", res.capability_report.rejected);
        let p = t.powers_clamped(lam, 45f64.to_radians().cos());
        let tt = detector_named(&res, through).total_incident;
        let rr = detector_named(&res, folded).total_incident;
        assert!((tt - p.t()).abs() < 1e-12, "line {i}: T {tt} vs {}", p.t());
        assert!((rr - p.r()).abs() < 1e-12, "line {i}: R {rr} vs {}", p.r());
        assert!((res.ledger.absorbed_surface - (1.0 - p.r() - p.t())).abs() < 1e-12);
    }
}

/// C28: a ray at 52 degrees on a 45-degree table is refused under `Strict`
/// with the sentence the spec names, and read at the edge and flagged under
/// `Permissive`. A divergent beam loses exactly its rays outside half a
/// degree.
#[test]
fn c28_a_one_angle_table_refuses_a_ray_at_another_angle() {
    let (s, through, _) = tilted(&filter_45());
    // The plate is at 135 degrees, so a ray at +7 degrees meets it at 52.
    let res = trace3(&s, &[beam(7.0, 0.55)], &strict());
    let reasons: Vec<String> = res
        .capability_report
        .rejected
        .iter()
        .filter_map(|r| match r {
            CapabilityIssue::UnsupportedSurfaceResponse { reason, .. } => reason.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(reasons, vec!["Data supplied at 45 degrees only; this ray is at 52 degrees".to_string()]);
    assert!((res.ledger.numerical_loss - 1.0).abs() < 1e-12);
    assert_eq!(detector_named(&res, through).total_incident, 0.0);

    let lax = trace3(&s, &[beam(7.0, 0.55)], &TraceConfig3::default());
    assert!(lax.capability_report.rejected.is_empty());
    assert!(lax
        .capability_report
        .approximated
        .iter()
        .any(|a| a.detail.contains("outside its data")));
    assert!((detector_named(&lax, through).total_incident - 0.6).abs() < 1e-12);

    // A fan from -2 to +2 degrees: under Strict only the rays within half a
    // degree of 45 get through.
    let fan: Vec<_> = (0..9).map(|i| beam(-2.0 + 0.5 * i as f64, 0.55)).collect();
    let res = trace3(&s, &fan, &strict());
    let kept = 3.0; // -0.5, 0 and +0.5 degrees
    assert!((detector_named(&res, through).total_incident - 0.6 * kept).abs() < 1e-12);
    assert!((res.ledger.numerical_loss - (9.0 - kept)).abs() < 1e-12);
}

/// A table measured from the front, met from the back, is named.
#[test]
fn a_one_sided_table_names_the_other_side() {
    let mut t = filter_45();
    t.side = TableSide::Front;
    let (s, _, _) = tilted(&t);
    // From +X towards -X: the back of a plate whose front faces -X.
    let d = Vec3::new(-1.0, 0.0, 0.0);
    let ray = launch(Ray3::new(Vec3::new(30.0, 0.0, 0.0), d).unwrap(), 0.55, 1.0);
    let res = trace3(&s, &[ray], &strict());
    assert!(res.capability_report.rejected.iter().any(|r| matches!(
        r,
        CapabilityIssue::UnsupportedSurfaceResponse { reason: Some(m), .. }
            if m == "Data supplied for the front side only; this ray meets the back"
    )));
}

/// Two s/p tables in series (the earlier spec's §9.4): the polarized trace
/// carries the polarization the first makes into the second and passes
/// `0.5 (Ts1 Ts2 + Tp1 Tp2)` of unpolarized light; the scalar-unpolarized
/// trace takes the mean at each and passes the product of the means. The two
/// are not equal, which is the limitation the scalar mode is labelled with.
#[test]
fn the_scalar_mode_takes_the_mean_at_each_surface() {
    let sp = |ts: f64, tp: f64| AngularTable {
        lambda_um: vec![0.5, 0.6],
        aoi_deg: vec![40.0, 50.0],
        power_resolution: PowerResolution::Sp,
        rs: Some(vec![1.0 - ts; 4]),
        rp: Some(vec![1.0 - tp; 4]),
        ts: Some(vec![ts; 4]),
        tp: Some(vec![tp; 4]),
        r: None,
        t: None,
        phase_information: PhaseInformation::Unknown,
        scope: TableScope::Interface,
        side: TableSide::Either,
        media: air(),
        provenance: None,
        out_of_domain: None,
        reflection: ReflectionRole::Path,
    };
    let (a, b) = (sp(0.9, 0.3), sp(0.7, 0.2));
    let mut s = Scene3::default();
    let (f1, b1) = table_pair(&a);
    let (f2, b2) = table_pair(&b);
    s.add_planar_element(Vec3::ZERO, 10.0, 135.0, f1, b1);
    s.add_planar_element(Vec3::new(20.0, 0.0, 0.0), 10.0, 135.0, f2, b2);
    let det = add_rect_detector(&mut s, Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 10.0, 10.0, [1, 1]);
    let ray = || [beam(0.0, 0.55)];
    let polarized = trace3(&s, &ray(), &TraceConfig3::default());
    let scalar = trace3(&s, &ray(), &TraceConfig3 { polarization: false, ..TraceConfig3::default() });
    let got_pol = detector_named(&polarized, det).total_incident;
    let got_sca = detector_named(&scalar, det).total_incident;
    let want_pol = 0.5 * (0.9 * 0.7 + 0.3 * 0.2);
    let want_sca = 0.5 * (0.9 + 0.3) * 0.5 * (0.7 + 0.2);
    assert!((got_pol - want_pol).abs() < 1e-12, "polarized {got_pol} vs {want_pol}");
    assert!((got_sca - want_sca).abs() < 1e-12, "scalar {got_sca} vs {want_sca}");
    assert!((want_pol - want_sca).abs() > 0.05);
    // The polarized trace says the phase was not in the data.
    assert!(polarized.capability_report.approximated.iter().any(|x| x.detail.contains("no phase")));
}

/// A table of a whole part cannot sit on a lens face; an interface table can,
/// and is read from each side by its own copy.
#[test]
fn a_whole_part_table_is_refused_on_a_body_face() {
    use oc_core::build3::BodyFace;
    let mut s = Scene3::default_scene();
    let lens = s.bodies[0].common.id.0;
    let mut whole = filter_45();
    whole.scope = TableScope::FiniteAssembly;
    let err = s.set_object_table(lens, Some(BodyFace::Front), whole).unwrap_err();
    assert!(err.contains("whole part"), "{err}");
    let mut face = AngularTable::one_angle(vec![0.4, 0.7], 0.0, vec![0.01, 0.01], vec![0.99, 0.99], TableMedia { incident: 1.0, exit: 1.52 });
    face.aoi_deg = vec![0.0];
    face.side = TableSide::Front;
    s.set_object_table(lens, Some(BodyFace::Front), face).unwrap();
    s.validate().unwrap();
    assert!(s.object_table(lens).is_some());
}
