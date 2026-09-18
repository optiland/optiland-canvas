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

/// A coated plate with real thickness puts the beams where a real plate does.
///
/// This is the whole reason a plate is a body rather than a plane. Three things
/// are checked against closed forms, not against the previous behaviour:
///
/// 1. the transmitted beam leaves **parallel** to the incident beam, which is
///    what plane-parallel faces guarantee;
/// 2. it is displaced sideways by `t·sin(θ−θ′)/cos θ′`, which for a 3 mm N-BK7
///    plate at 45° is almost exactly a millimetre;
/// 3. it carries the glass in its optical path length.
#[test]
fn a_thick_plate_walks_the_transmitted_beam_sideways() {
    use oc_core::scene3::{AngularDistribution3, SpatialDistribution3};
    use oc_core::thinfilm::lateral_shift_mm;

    const T_MM: f64 = 3.0;
    let mut s = Scene3::default_scene();
    // One ray, on the axis, so its own displacement is the thing measured and
    // not the spread of a bundle.
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.65, 1.0)]);
    s.sources[0].spatial = SpatialDistribution3::Point;
    s.sources[0].angular = AngularDistribution3::Direction { local_dir: Vec3::X };
    s.add_plate_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, T_MM, TILT, 500.0, true)
        .expect("the plate is valid");
    s.validate().expect("the scene validates");

    let res = trace_scene3(&s, &cfg());
    // Between the plate and the lens: the beam is travelling free, so its height
    // and direction there are the plate's doing and nothing else's.
    let onward: Vec<_> = res
        .segments
        .iter()
        .filter(|g| {
            // The segment that leaves the far face and runs free toward the
            // lens: it starts just past the plate and travels along +X.
            let d = g.b - g.a;
            d.x > 0.5 && g.a.x > -34.5 && g.a.x < -30.0
        })
        .collect();
    assert!(
        !onward.is_empty(),
        "something must get through: {:?}",
        res.segments
    );

    // 1. Parallel: plane-parallel faces cannot change a beam's direction.
    for g in &onward {
        let d = (g.b - g.a).normalize();
        assert!(
            d.y.abs() < 1e-12 && d.z.abs() < 1e-12,
            "the transmitted beam must leave parallel to the incident one: {d:?}"
        );
    }

    // 2. Displaced: the ray went in on y = 0 and comes out somewhere else.
    let shift = onward.iter().map(|g| g.a.y.abs()).fold(0.0f64, f64::max);
    let want = lateral_shift_mm(T_MM, 1.0, 1.5168, 45.0);
    assert!(
        (want - 1.0).abs() < 0.05,
        "sanity: a 3 mm plate at 45 degrees shifts about 1 mm, not {want}"
    );
    assert!(
        (shift - want).abs() < 0.02,
        "traced shift {shift} mm against the closed form {want} mm"
    );
    // And it is a real displacement, not noise.
    assert!(shift > 0.5, "the beam really did move: {shift} mm");
}

/// The plate's far face returns its own reflection, and the glass shows up in
/// the optical path. Neither existed when the plate was a plane.
#[test]
fn a_thick_plate_has_a_second_surface() {
    const T_MM: f64 = 4.0;
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.65, 1.0)]);
    s.add_plate_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, T_MM, TILT, 500.0, true)
        .expect("the plate is valid");
    s.validate().expect("the scene validates");

    let with_ghosts = TraceConfig3 {
        ghosts: true,
        max_ghost_bounces: 2,
        record_detector_hits: true,
        ..cfg()
    };
    let res = trace_scene3(&s, &with_ghosts);
    // A ghost branch exists that leaves the plate region upward, which can only
    // be the far face reflecting back out through the coated one.
    let second_surface = res
        .segments
        .iter()
        .any(|g| g.flags.ghost && (g.b - g.a).y.abs() > 1.0);
    assert!(second_surface, "the far face must return a beam of its own");
    assert!(res.ledger.residual() < 1e-12, "ledger {:?}", res.ledger);
}


// ---------------------------------------------------------------------------
// A coating on a lens, which is what a coating is mostly for
// ---------------------------------------------------------------------------

/// An AR coating on a lens face does what an AR coating does: a face that
/// reflected four percent reflects about one, and the light goes through
/// instead of bouncing around inside the instrument.
///
/// The oracle is the single-layer quarter-wave formula, `((n0·ns − nf²) /
/// (n0·ns + nf²))²`, which for magnesium fluoride on this crown is 1.3% where
/// bare glass is 4.2%. Two faces, so the transmission goes from about 0.92 to
/// about 0.97.
#[test]
fn an_ar_coating_on_a_lens_face_passes_more_light() {
    use oc_core::build3::BodyFace;
    use oc_core::thinfilm::DesignKind;

    let n_glass = 1.5168f64;
    let n_film = 1.38f64;
    let bare_r = ((1.0 - n_glass) / (1.0 + n_glass)).powi(2);
    let ar_r = ((n_glass - n_film * n_film) / (n_glass + n_film * n_film)).powi(2);
    assert!(bare_r > 0.041 && bare_r < 0.043, "bare face: {bare_r}");
    assert!(ar_r > 0.012 && ar_r < 0.014, "coated face: {ar_r}");

    let ar = CoatingDesign {
        kind: DesignKind::SingleLayerAr,
        lambda_nm: 550.0,
        aoi_deg: 0.0,
        periods: 1,
        n_high: 2.35,
        n_low: n_film,
    };

    let measure = |coat: bool| -> f64 {
        let mut s = Scene3::default_scene();
        let lens = s
            .bodies
            .first()
            .map(|b| b.common.id.0)
            .expect("the default scene has a singlet");
        if coat {
            assert!(
                s.set_body_face_coating(lens, BodyFace::Front, Some(ar)),
                "front face coats"
            );
            assert!(
                s.set_body_face_coating(lens, BodyFace::Back, Some(ar)),
                "back face coats"
            );
        }
        let det = s.detectors[0].common.id;
        s.validate().expect("valid");
        let res = trace_scene3(&s, &TraceConfig3::default());
        detector_named(&res, det).total_incident
    };

    let bare = measure(false);
    let coated = measure(true);
    assert!(coated > bare, "coating a lens helps: {coated} vs {bare}");
    // Both faces, so roughly (1 - ar_r)² / (1 - bare_r)² more light. The beam
    // is not collimated and the faces are curved, so the rays meet them at a
    // spread of angles and the ratio is near that rather than on it.
    let want = ((1.0 - ar_r) / (1.0 - bare_r)).powi(2);
    let got = coated / bare;
    assert!(
        (got - want).abs() < 0.01,
        "and by about the right amount: {got} vs {want}"
    );
}

/// Coating a face is a thing you can undo, and it lands on the face you named.
#[test]
fn a_lens_face_can_be_coated_and_stripped_one_side_at_a_time() {
    use oc_core::build3::BodyFace;
    use oc_core::thinfilm::DesignKind;

    let ar = CoatingDesign {
        kind: DesignKind::SingleLayerAr,
        lambda_nm: 550.0,
        aoi_deg: 0.0,
        periods: 1,
        n_high: 2.35,
        n_low: 1.38,
    };
    let mut s = Scene3::default_scene();
    let lens = s.bodies.first().map(|b| b.common.id.0).expect("a singlet");

    assert!(s.body_face_coating(lens, BodyFace::Front).is_none());
    assert!(s.set_body_face_coating(lens, BodyFace::Front, Some(ar)));
    assert!(s.body_face_coating(lens, BodyFace::Front).is_some());
    // And only that face: the back is still bare glass.
    assert!(s.body_face_coating(lens, BodyFace::Back).is_none());
    s.validate().expect("a coated lens is a valid scene");

    assert!(s.set_body_face_coating(lens, BodyFace::Back, Some(ar)));
    assert!(s.body_face_coating(lens, BodyFace::Back).is_some());
    assert!(s.set_body_face_coating(lens, BodyFace::Front, None));
    assert!(s.body_face_coating(lens, BodyFace::Front).is_none());
    assert!(
        s.body_face_coating(lens, BodyFace::Back).is_some(),
        "stripping one face leaves the other alone"
    );

    // A detector is not a body with faces, and neither is a nonexistent id.
    let det = s.detectors[0].common.id.0;
    assert!(!s.set_body_face_coating(det, BodyFace::Front, Some(ar)));
    assert!(!s.set_body_face_coating(9_999_999, BodyFace::Front, Some(ar)));
}
