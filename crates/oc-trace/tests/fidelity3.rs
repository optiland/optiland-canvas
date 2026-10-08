//! The fidelity profile in the tracer: C33 (profile equivalence) and the
//! trace half of C34 (every result carries its profile).
#![cfg(feature = "trace3")]

mod common;
mod corpus3;

use common::*;
use oc_core::fidelity::{CoatingModel, FidelityProfile, LayerAbsorption, Preset};
use oc_core::scene::Spectrum;
use oc_core::scene3::{Scene3, SurfaceResponse};
use oc_core::thinfilm::{coating_rt_at, film_coating, CoatingDesign, DesignKind};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, trace_scene3, TraceConfig3};

const GOLDEN: &str = include_str!("data/pre_profile_corpus3.json");

/// C33, first half: `Standard` is the pre-profile kernel, to the bit, on every
/// corpus scene and configuration. The golden file was recorded at 212c747,
/// before the profile existed (`record_corpus3.rs`).
#[test]
fn standard_reproduces_the_pre_profile_kernel_bit_for_bit() {
    let golden: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    let mut compared = 0;
    for case in corpus3::cases() {
        for (name, cfg) in corpus3::configs() {
            let cfg = TraceConfig3 { fidelity: Preset::Standard.profile(None, false), ..cfg };
            let got = corpus3::fingerprint(&case.trace(&cfg));
            let want: Vec<String> = serde_json::from_value(golden[case.name][name].clone())
                .unwrap_or_else(|e| panic!("no golden for {} / {name}: {e}", case.name));
            assert_eq!(got, want, "{} under {name}", case.name);
            compared += 1;
        }
    }
    assert_eq!(compared, 20);
}

/// A coated plate standing alone in a beam, its coating a single-layer AR
/// designed for 30 degrees, and a detector on each arm.
fn ar_window() -> (Scene3, oc_core::id::ObjectId, oc_core::id::ObjectId, oc_core::thinfilm::Coating) {
    let coating = film_coating(
        CoatingDesign {
            kind: DesignKind::QuarterWaveStack,
            lambda_nm: 600.0,
            aoi_deg: 30.0,
            periods: 3,
            n_high: 2.35,
            n_low: 1.46,
            adapt: None,
        },
        1.0,
        1.0,
    );
    let mut s = Scene3::new();
    let id = s.add_planar_mirror(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 0.5).unwrap();
    let surf = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
    surf.response_front = SurfaceResponse::Coating { coating: coating.clone() };
    surf.response_back = SurfaceResponse::Coating { coating: coating.seen_from_behind() };
    let through = add_rect_detector(&mut s, Vec3::new(60.0, 0.0, 0.0), -Vec3::X, 200.0, 200.0, [2, 2]);
    let back = add_rect_detector(&mut s, Vec3::new(-60.0, 0.0, 0.0), Vec3::X, 200.0, 200.0, [2, 2]);
    s.validate().expect("valid");
    (s, through, back, coating)
}

/// C33, second half: under `Fast` a designed coating answers with its
/// design-point response at every wavelength and angle.
#[test]
fn fast_answers_with_the_design_point_at_every_wavelength_and_angle() {
    let (s, through, back, coating) = ar_window();
    let fast = TraceConfig3::default().with_fidelity(Preset::Fast.profile(Some(0), false));
    let (lam_d, cos_d, designed) = coating.nominal_point();
    assert!(designed);
    let want = coating_rt_at(&coating, 1.0, 1.0, lam_d, cos_d).unwrap();
    for deg in [0.0f64, 20.0, 45.0] {
        for lam in [0.42, 0.6, 0.9] {
            let d = Vec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
            let ray = Ray3::new(Vec3::new(-50.0, -50.0 * deg.to_radians().tan(), 0.0), d).unwrap();
            let r = trace3(&s, &[launch(ray, lam, 1.0)], &fast);
            let t = detector_named(&r, through).total_incident;
            assert!((t - want.t()).abs() < 1e-12, "{deg} deg, {lam} um: T {t} vs {}", want.t());
            if deg < 30.0 {
                let rr = detector_named(&r, back).total_incident;
                assert!((rr - want.r()).abs() < 1e-12, "{deg} deg, {lam} um: R {rr} vs {}", want.r());
            }
            // And the full model does depend on wavelength, or the test proves nothing.
            let full = trace3(&s, &[launch(ray, lam, 1.0)], &TraceConfig3::default());
            if (lam - 0.6).abs() > 0.1 {
                assert!((detector_named(&full, through).total_incident - t).abs() > 1e-3);
            }
            assert!(r.fidelity.text().contains("nominal, not verified"));
        }
    }
}

/// The spec's own reason for the polarized model, under the profile: two
/// dichroics in series differ by more than 5% between `Unpolarized` and the
/// polarized model that `Standard` and `Reference` both run.
#[test]
fn the_polarized_profiles_differ_from_unpolarized_on_two_dichroics() {
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.56, 1.0)]);
    s.add_plate_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 3.0, 135.0, 560.0, true).unwrap();
    s.add_plate_dichroic(Vec3::new(-35.0, 40.0, 0.0), 15.0, 3.0, -45.0, 560.0, true).unwrap();
    let det = add_rect_detector(&mut s, Vec3::new(25.0, 40.0, 0.0), -Vec3::X, 20.0, 20.0, [4, 4]);
    let base = TraceConfig3 { ghosts: false, ..TraceConfig3::default() };
    let flux = |p: FidelityProfile| detector_named(&trace_scene3(&s, &base.with_fidelity(p)), det).total_incident;
    let unpolarized = flux(FidelityProfile { coating_model: CoatingModel::Unpolarized, ..Default::default() });
    for preset in [Preset::Standard, Preset::Reference] {
        let polarized = flux(preset.profile(None, false));
        let gap = (polarized - unpolarized).abs() / unpolarized;
        assert!(gap > 0.05, "{preset:?} against Unpolarized: {gap:.3}");
    }
}

/// The deprecated alias and the switch are one thing.
#[test]
fn the_polarization_flag_is_an_alias_for_the_coating_model() {
    let off = TraceConfig3 { polarization: false, ..TraceConfig3::default() };
    assert_eq!(off.effective_profile().coating_model, CoatingModel::Unpolarized);
    let switch = TraceConfig3::default()
        .with_fidelity(FidelityProfile { coating_model: CoatingModel::Unpolarized, ..Default::default() });
    let s = corpus3::cases().remove(1).scene;
    let a = corpus3::fingerprint(&trace_scene3(&s, &off));
    let b = corpus3::fingerprint(&trace_scene3(&s, &switch));
    assert_eq!(a, b);
}

/// C34, trace half: a result and its capability report carry the profile it
/// ran under, ghosts included.
#[test]
fn every_result_carries_its_profile() {
    let s = Scene3::default_scene();
    for preset in Preset::ALL {
        let cfg = TraceConfig3::default().with_fidelity(preset.profile(None, false));
        let r = trace_scene3(&s, &cfg);
        assert_eq!(r.fidelity.preset, Some(preset), "{}", r.fidelity.text());
        assert_eq!(r.capability_report.fidelity.as_ref(), Some(&r.fidelity));
        assert!(r.fidelity.text().starts_with(&format!("computed with {}", preset.name())));
        assert!(r.fidelity.solver_version.contains("thin film"));
    }
    let custom = TraceConfig3 { polarization: false, ghosts: false, ..TraceConfig3::default() };
    let r = trace_scene3(&s, &custom);
    assert_eq!(r.fidelity.preset, None, "unpolarized is custom");
    assert_eq!(r.fidelity.profile.ghosts, Some(0));
    let json = serde_json::to_string(&r.capability_report).unwrap();
    assert!(json.contains("\"fidelity\""), "{json}");
}

/// `layer_absorption: Off` keeps the ledger and drops the surface deposits.
#[test]
fn absorption_off_records_no_surface_deposit_and_moves_no_flux() {
    let case = corpus3::cases().into_iter().find(|c| c.name == "mirror_and_window").unwrap();
    let on = TraceConfig3 { deposit_bodies: [0, 64], ..TraceConfig3::default() };
    let off = on.with_fidelity(FidelityProfile { layer_absorption: LayerAbsorption::Off, ..Default::default() });
    let a = case.trace(&on);
    let b = case.trace(&off);
    assert!(!a.deposits.is_empty());
    assert!(b.deposits.is_empty(), "{} deposits under Off", b.deposits.len());
    assert_eq!(format!("{:?}", a.ledger), format!("{:?}", b.ledger));
}
