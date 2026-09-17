//! Authored beam splitters and dichroic filters route light as specified.
//!
//! `build3` makes both as plates with an explicit reflectance and
//! transmittance. These tests fold the default scene's beam with one and check
//! where the flux lands: the reflected arm on a detector above the plate, the
//! transmitted arm through the lens to its focus, every wavelength on the side
//! its coating sends it to, and the ledger balanced.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::build3::DichroicSpec;
use oc_core::id::ObjectId;
use oc_core::scene::Spectrum;
use oc_core::scene3::Scene3;
use oc_core::vec3::Vec3;
use oc_trace::{trace_scene3, TraceConfig3};

/// The default scene (a beam along +X from x = -70, a singlet at the origin, a
/// detector near its focus) with the source emitting `nm` at equal weight, a
/// plate added by `add` at x = -35, and a detector at y = +40 for what the
/// plate reflects upward. Returns the scene and the up and focus detectors.
fn folded(
    nm: &[f64],
    add: impl FnOnce(&mut Scene3) -> Option<u64>,
) -> (Scene3, ObjectId, ObjectId) {
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(nm.iter().map(|w| (w / 1000.0, 1.0)).collect());
    add(&mut s).expect("the plate is valid");
    let focus = s.detectors[0].common.id;
    let up = add_rect_detector(
        &mut s,
        Vec3::new(-35.0, 40.0, 0.0),
        -Vec3::Y,
        20.0,
        20.0,
        [8, 8],
    );
    s.validate().expect("the folded scene validates");
    (s, up, focus)
}

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        record_detector_hits: true,
        ..no_ghosts()
    }
}

/// A plate at 135° faces the incoming beam and folds its reflection to +Y.
const TILT: f64 = 135.0;

#[test]
fn a_beam_splitter_divides_the_beam_between_its_arms() {
    let (s, up, focus) = folded(&[587.56], |s| {
        s.add_beam_splitter(Vec3::new(-35.0, 0.0, 0.0), 15.0, TILT, 0.3, 0.7)
    });
    let res = trace_scene3(&s, &cfg());
    let launched = res.ledger.launched;
    let up = detector_named(&res, up);
    let focus = detector_named(&res, focus);
    // Every ray meets the plate, so the reflected arm is exactly 30 %.
    assert!(
        (up.total_incident - 0.3 * launched).abs() < 1e-12 * launched,
        "up arm {} of {launched}",
        up.total_incident
    );
    assert_eq!(up.hits, 21, "one reflected ray per launched ray");
    // The transmitted 70 % crosses the lens, which reflects a few percent away.
    assert!(focus.total_incident < 0.7 * launched && focus.total_incident > 0.6 * launched);
    assert!(res.ledger.residual() < 1e-12, "ledger {:?}", res.ledger);
}

#[test]
fn a_long_pass_dichroic_reflects_blue_and_passes_red() {
    let (s, up, focus) = folded(&[450.0, 650.0], |s| {
        s.add_dichroic(
            Vec3::new(-35.0, 0.0, 0.0),
            15.0,
            TILT,
            DichroicSpec::default(),
        )
    });
    let res = trace_scene3(&s, &cfg());
    let per_line = res.ledger.launched / 2.0;
    let up = detector_named(&res, up);
    let focus = detector_named(&res, focus);
    assert!(
        (up.total_incident - 0.98 * per_line).abs() < 1e-12,
        "{}",
        up.total_incident
    );
    assert!(up
        .records
        .iter()
        .all(|h| (h.lambda_um - 0.45).abs() < 1e-12));
    assert!(!focus.records.is_empty());
    assert!(focus
        .records
        .iter()
        .all(|h| (h.lambda_um - 0.65).abs() < 1e-12));
    assert!(focus.total_incident < 0.95 * per_line && focus.total_incident > 0.85 * per_line);
    // The blue line's 2 % and the red line's 5 % are absorbed at the plate.
    let absorbed = res.ledger.absorbed_surface;
    assert!(
        (absorbed - 0.07 * per_line).abs() < 1e-12,
        "absorbed {absorbed}"
    );
    assert!(res.ledger.residual() < 1e-12);
}

#[test]
fn a_short_pass_dichroic_swaps_the_arms() {
    let spec = DichroicSpec {
        long_pass: false,
        ..DichroicSpec::default()
    };
    let (s, up, focus) = folded(&[450.0, 650.0], |s| {
        s.add_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, TILT, spec)
    });
    let res = trace_scene3(&s, &cfg());
    let up = detector_named(&res, up);
    let focus = detector_named(&res, focus);
    assert!(up
        .records
        .iter()
        .all(|h| (h.lambda_um - 0.65).abs() < 1e-12));
    assert!(focus
        .records
        .iter()
        .all(|h| (h.lambda_um - 0.45).abs() < 1e-12));
    assert!((up.total_incident - 0.98 * res.ledger.launched / 2.0).abs() < 1e-12);
}

#[test]
fn a_line_on_the_edge_splits_by_the_tabulated_fractions() {
    // 550 nm sits mid-transition: R = 0.49, T = 0.475.
    let (s, up, _) = folded(&[550.0], |s| {
        s.add_dichroic(
            Vec3::new(-35.0, 0.0, 0.0),
            15.0,
            TILT,
            DichroicSpec::default(),
        )
    });
    let res = trace_scene3(&s, &cfg());
    let up = detector_named(&res, up);
    let launched = res.ledger.launched;
    assert!((up.total_incident - 0.49 * launched).abs() < 1e-9 * launched);
    assert!((res.ledger.absorbed_surface - 0.035 * launched).abs() < 1e-9 * launched);
}
