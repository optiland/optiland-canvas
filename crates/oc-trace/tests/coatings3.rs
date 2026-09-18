//! A dichroic built as a real thin-film stack routes light by its actual
//! reflectance, at the angle it is actually used at.
//!
//! The plate these tests fold the beam with is the same one `splitters3` uses,
//! but its coating is a stack of layers rather than a table in wavelength, so
//! the transfer matrix runs at each ray's own angle of incidence. What that
//! buys, and what it costs anyone who was relying on the old behaviour, is the
//! subject of the tests below.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::build3::DichroicSpec;
use oc_core::id::ObjectId;
use oc_core::scene::Spectrum;
use oc_core::scene3::Scene3;
use oc_core::thinfilm::CoatingDesign;
use oc_core::vec3::Vec3;
use oc_trace::{trace_scene3, TraceConfig3};

/// A plate at 135° faces the incoming beam and folds its reflection to +Y.
const TILT: f64 = 135.0;

/// The default scene with the source emitting `nm`, a plate added at x = −35,
/// and a detector at y = +40 catching whatever the plate sends upward.
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

/// What fraction of the launched flux went up the reflected arm.
fn reflected_fraction(nm: f64, edge_nm: f64, long_pass: bool) -> f64 {
    let (s, up, _) = folded(&[nm], |s| {
        s.add_film_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, TILT, edge_nm, long_pass)
    });
    let res = trace_scene3(&s, &cfg());
    detector_named(&res, up).total_incident / res.ledger.launched
}

/// A long-pass film dichroic specified for 500 nm reflects blue and passes red,
/// with the plate used at the 45° its design was solved for.
#[test]
fn a_film_dichroic_routes_colour_by_its_edge() {
    let blue = reflected_fraction(450.0, 500.0, true);
    let red = reflected_fraction(650.0, 500.0, true);
    assert!(
        blue > 0.95,
        "blue must reflect off a 500 nm long-pass: {blue}"
    );
    assert!(red < 0.25, "red must mostly pass it: {red}");
}

/// The fix, end to end.
///
/// Two plates in the same place, at the same 45°, with the same nominal 550 nm
/// long-pass edge. One was designed for the 45° it is used at; the other was
/// designed for normal incidence and then hung at 45°, which is the mistake the
/// old table-driven dichroic made for every plate in every scene. Their bands
/// are about 40 nm apart, so there is a wide stretch of the spectrum they route
/// in opposite directions.
#[test]
fn the_band_really_does_move_when_the_plate_is_tilted() {
    let designed_for_use = |nm: f64| reflected_fraction(nm, 550.0, true);
    let designed_on_axis = |nm: f64| {
        let (s, up, _) = folded(&[nm], |s| {
            let id = s.add_film_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, TILT, 550.0, true)?;
            // The same edge, solved for a plate sitting square to the beam, then
            // used at 45° anyway.
            assert!(s.set_surface_coating_design(id, CoatingDesign::long_pass(550.0, 0.0)));
            Some(id)
        });
        let res = trace_scene3(&s, &cfg());
        detector_named(&res, up).total_incident / res.ledger.launched
    };

    // Deep in both bands, they agree: blue reflects either way.
    assert!(designed_for_use(460.0) > 0.95 && designed_on_axis(460.0) > 0.95);
    // At 530 nm they disagree about which arm the light belongs in, which is
    // the whole bug: one sends it up the reflected arm, the other passes it.
    let used = designed_for_use(530.0);
    let naive = designed_on_axis(530.0);
    assert!(
        used > 0.8,
        "designed for 45°, 530 nm is inside the band: {used}"
    );
    assert!(
        naive < 0.2,
        "designed on axis and tilted, its band has already let 530 nm go: {naive}"
    );
    assert!(
        used - naive > 0.6,
        "the two must disagree strongly: {used} vs {naive}"
    );
}

/// Energy still balances with a stack in the scene: whatever a coating does not
/// reflect or transmit, it absorbs, and the ledger has to see all three.
#[test]
fn a_coated_plate_keeps_the_ledger_balanced() {
    let (s, _, _) = folded(&[450.0, 550.0, 650.0], |s| {
        s.add_film_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, TILT, 520.0, true)
    });
    let res = trace_scene3(&s, &cfg());
    assert!(res.ledger.residual() < 1e-12, "ledger {:?}", res.ledger);
}

/// The old table-driven dichroic still works and still ignores the angle. It has
/// to: documents written before this existed carry one, and silently giving them
/// different physics on load would be its own kind of lie. The two coexist, and
/// the difference between them at 45° is exactly the size of the bug.
#[test]
fn the_table_dichroic_still_traces_and_still_ignores_the_angle() {
    let table = |nm: f64| {
        let (s, up, _) = folded(&[nm], |s| {
            s.add_dichroic(
                Vec3::new(-35.0, 0.0, 0.0),
                15.0,
                TILT,
                DichroicSpec {
                    edge_nm: 550.0,
                    width_nm: 20.0,
                    long_pass: true,
                    r_max: 0.98,
                    t_max: 0.95,
                },
            )
        });
        let res = trace_scene3(&s, &cfg());
        detector_named(&res, up).total_incident / res.ledger.launched
    };
    // Its edge is wherever the table says, regardless of the 45° it sits at.
    assert!(table(530.0) > 0.9, "inside the table's band");
    assert!(table(570.0) < 0.1, "outside it");
}
