//! A heliostat puts sunlight on its target at every hour, and the sun that
//! lights it carries the watts the sky does.
//!
//! The mirror is turned by the rig, not by the test: the test only sets the
//! site and the clock, traces, and checks where the light went. The target is
//! out of the canvas plane on purpose, so the sun, the mirror and the target
//! are a genuinely three-dimensional arrangement that no in-plane shortcut
//! could get right.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::heliostat::{HelioTarget, LocalTime, Site};
use oc_core::id::ObjectId;
use oc_core::scene3::Scene3;
use oc_core::vec3::Vec3;
use oc_trace::{trace_scene3, TraceConfig3};

/// Almería, the Plataforma Solar, on the June solstice, clocks on CEST.
fn almeria() -> Site {
    Site {
        latitude_deg: 37.094,
        longitude_deg: -2.358,
        elevation_m: 500.0,
        pressure_mbar: 950.0,
        temperature_c: 20.0,
    }
}

fn at(hour: u32, minute: u32) -> LocalTime {
    LocalTime {
        year: 2026,
        month: 6,
        day: 21,
        hour,
        minute,
        second: 0.0,
        utc_offset_h: 2.0,
    }
}

/// Sized to the field, as the app sizes it: the target is 22 m away, past the
/// 10 m a bench scene needs.
fn cfg_for(s: &Scene3) -> TraceConfig3 {
    TraceConfig3 {
        bounds_mm: oc_trace::bounds_for_scene3(s),
        ..no_ghosts()
    }
}

/// A 1 m square heliostat at the origin, a target 20 m north (the canvas's
/// -X with the default south-right bearing) and 8 m up, and 3 m out of the
/// canvas plane, where a 3 m detector faces the mirror: a flat mirror's spot is
/// its own outline, blurred by the sun's disc, and it has to land whole.
fn field(time: LocalTime) -> (Scene3, ObjectId, ObjectId, Vec3) {
    let mut s = Scene3::new();
    s.add_sun(almeria(), time, 180.0);
    let target = Vec3::new(-20_000.0, 8_000.0, 3_000.0);
    let det = add_rect_detector(&mut s, target, -target.normalize(), 1500.0, 1500.0, [16, 16]);
    let mirror = s
        .add_heliostat(Vec3::ZERO, 1000.0, 1000.0, 0.95, HelioTarget::Object(det))
        .expect("a heliostat");
    s.validate().expect("the field validates");
    (s, mirror, det, target)
}

/// Morning, noon and evening: the rig turns the mirror, and the reflected
/// spot's centroid sits on the target's centre every time.
#[test]
fn the_spot_stays_on_the_target_all_day() {
    for (h, m) in [(8, 30), (11, 0), (14, 20), (18, 45)] {
        let (mut s, _, det, _) = field(at(h, m));
        // Enough rays that the centroid is the optics and not the sampling:
        // a flat mirror maps its own centroid onto the target's exactly.
        s.solar.as_mut().unwrap().bands = 1;
        s.sources[0].samples.spatial_samples = 4096;
        s.sources[0].samples.angular_samples = 16;
        let state = s.resolve_solar().expect("resolves");
        assert!(state.is_up(), "{h}:{m:02} the sun is up");
        let res = trace_scene3(&s, &cfg_for(&s));
        let d = detector_named(&res, det);
        assert!(d.total_incident > 0.0, "{h}:{m:02}: light arrives");
        let c = d.flux_uv / d.total_incident;
        // The sun's disc blurs the spot by ±4.65 mrad × 21.7 m ≈ ±100 mm;
        // the centroid of a symmetric blur does not move.
        assert!(
            c.length() < 5.0,
            "{h}:{m:02}: spot centroid {c:?} mm off the target centre"
        );
    }
}

/// The law of reflection, solved for the mirror: the normal makes equal
/// angles with the sun and the target, and lies in the plane of the two.
/// And the mount is azimuth-elevation, so the mirror's width stays level.
#[test]
fn the_normal_bisects_sun_and_target_and_the_width_stays_level() {
    for (h, m) in [(7, 40), (13, 0), (19, 30)] {
        let (mut s, mirror, _, target) = field(at(h, m));
        let state = s.resolve_solar().unwrap();
        let hs = state.heliostats[0];
        assert!(hs.tracking);
        let t = target.normalize();
        let (a, b) = (hs.normal.dot(state.sun_dir), hs.normal.dot(t));
        assert!((a - b).abs() < 1e-12, "equal angles: {a} vs {b}");
        let coplanar = hs.normal.dot(state.sun_dir.cross(t));
        assert!(coplanar.abs() < 1e-12, "in the plane: {coplanar}");
        let m = s.surfaces.iter().find(|x| x.common.id == mirror).unwrap();
        let width = m.common.transform.rotate(Vec3::X);
        assert!(width.y.abs() < 1e-12, "the width is level: {width:?}");
        assert!(
            (hs.incidence_deg - a.acos().to_degrees()).abs() < 1e-9,
            "incidence reported"
        );
    }
}

/// The sun source is calibrated: a detector square to the sun that catches
/// the whole beam reads DNI times the beam's area times the band's share.
#[test]
fn the_sun_delivers_dni_times_area() {
    let mut s = Scene3::new();
    s.add_sun(almeria(), at(12, 0), 180.0);
    let det = add_disc_detector(&mut s, Vec3::ZERO, Vec3::Y, 5_000.0, [4, 4]);
    let rig = s.solar.as_mut().unwrap();
    rig.aim_at = Some(det);
    rig.beam_radius_mm = 400.0;
    let state = s.resolve_solar().unwrap();
    // Turn the detector to face the sun and trace.
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, state.sun_dir);
    s.detectors[0].common.transform =
        oc_core::vec3::Isometry3::new(Vec3::ZERO, [q.x, q.y, q.z, q.w]).unwrap();
    s.resolve_solar().unwrap();
    let res = trace_scene3(&s, &cfg_for(&s));
    let got = detector_named(&res, det).total_incident;
    let share = oc_core::solar::am15_fraction(0.3, 2.5);
    let want = 900.0 * share * std::f64::consts::PI * 0.4 * 0.4;
    assert!(
        (got - want).abs() < 1e-9 * want,
        "{got} W vs DNI x area x share = {want} W"
    );
    assert!((state.source_flux_w - want).abs() < 1e-9 * want);
}

/// What a heliostat delivers is DNI times its reflectivity times the area the
/// sun sees, which is the mirror's area times the cosine of the incidence.
/// Sampled, so held to the sampling error of a few thousand rays.
#[test]
fn the_cosine_loss_is_the_incidence_cosine() {
    for (h, m) in [(9, 0), (16, 0)] {
        let (mut s, _, det, _) = field(at(h, m));
        s.sources[0].samples.spatial_samples = 16_384;
        s.sources[0].samples.angular_samples = 1;
        let state = s.resolve_solar().unwrap();
        let res = trace_scene3(&s, &cfg_for(&s));
        let got = detector_named(&res, det).total_incident;
        let cos = state.heliostats[0].incidence_deg.to_radians().cos();
        let share = oc_core::solar::am15_fraction(0.3, 2.5);
        let want = 900.0 * share * 0.95 * 1.0 * cos;
        let rel = (got - want).abs() / want;
        assert!(
            rel < 0.01,
            "{h}:00: {got:.2} W vs {want:.2} W (cos {cos:.4}), {rel:.4}"
        );
    }
}

/// No sun, no light, whatever the clock says.
#[test]
fn the_sun_below_the_horizon_is_dark() {
    let (mut s, _, det, _) = field(at(23, 30));
    let state = s.resolve_solar().unwrap();
    assert!(!state.is_up());
    assert_eq!(s.sources[0].flux, 0.0);
    let res = trace_scene3(&s, &cfg_for(&s));
    assert_eq!(detector_named(&res, det).total_incident, 0.0);
}

/// Deleting a heliostat's mirror or target drops it from the rig, and the
/// scene still validates; deleting the sun's source drops the rig.
#[test]
fn deleting_what_a_rig_names_repairs_the_rig() {
    let (mut s, mirror, det, _) = field(at(12, 0));
    assert!(s.delete_object(det.0));
    s.resolve_solar();
    assert!(s.solar.as_ref().unwrap().heliostats.is_empty());
    s.validate().expect("valid after the target went");
    let _ = mirror;
    let src = s.sources[0].id.0;
    assert!(s.delete_object(src));
    s.resolve_solar();
    assert!(s.solar.is_none());
    s.validate().expect("valid after the sun went");
}

/// A rig survives a save, and a document with no sun has no `solar` key.
#[test]
fn the_rig_round_trips_and_stays_out_of_sunless_documents() {
    let (s, _, _, _) = field(at(10, 0));
    let json = serde_json::to_string(&s).unwrap();
    let back: Scene3 = serde_json::from_str(&json).unwrap();
    back.validate().unwrap();
    assert_eq!(back.solar, s.solar);
    let plain = serde_json::to_string(&Scene3::new()).unwrap();
    assert!(!plain.contains("solar"));
}


/// A field, not a mirror: the beam covers every heliostat, and each one
/// delivers its own cosine share onto the shared target.
#[test]
fn every_heliostat_in_a_field_is_lit() {
    let (mut s, first, det, _) = field(at(10, 0));
    let second = s
        .add_heliostat(
            Vec3::new(0.0, 0.0, -5_000.0),
            1000.0,
            1000.0,
            0.95,
            HelioTarget::Object(det),
        )
        .expect("a second heliostat");
    s.solar.as_mut().unwrap().bands = 1;
    s.sources[0].samples.spatial_samples = 65_536;
    s.sources[0].samples.angular_samples = 1;
    let state = s.resolve_solar().unwrap();
    assert_eq!(state.heliostats.len(), 2);
    let res = trace_scene3(&s, &cfg_for(&s));
    let got = detector_named(&res, det).total_incident;
    let share = oc_core::solar::am15_fraction(0.3, 2.5);
    let want: f64 = state
        .heliostats
        .iter()
        .map(|h| 900.0 * share * 0.95 * h.incidence_deg.to_radians().cos())
        .sum();
    let rel = (got - want).abs() / want;
    assert!(rel < 0.02, "{got:.1} W from two mirrors vs {want:.1} W, {rel:.4}");
    let _ = (first, second);
}

/// The sun's distance is the user's to choose, over a field too: a disc far
/// up-sun stands above a tall target, as the real sun does, and the mirror
/// gets the same watts. Asked for less than a beam radius, the disc stays a
/// radius out so it clears the mirrors.
#[test]
fn the_sun_stands_as_far_up_sun_as_asked() {
    let (mut s, _, det, target) = field(at(13, 0));
    s.solar.as_mut().unwrap().bands = 1;
    s.sources[0].samples.spatial_samples = 65_536;
    s.sources[0].samples.angular_samples = 1;
    s.resolve_solar().unwrap();
    let near_w = detector_named(&trace_scene3(&s, &cfg_for(&s)), det).total_incident;
    let r = s.solar.as_ref().unwrap().beam_radius_mm;

    s.solar.as_mut().unwrap().standoff_mm = 30_000.0;
    let state = s.resolve_solar().unwrap();
    let at_ = s.sources[0].transform.pos;
    assert!((at_.length() - 30_000.0).abs() < 1.0, "disc {:.0} mm out", at_.length());
    assert!(at_.dot(state.sun_dir) > 29_999.0, "the disc sits up-sun");
    assert!(at_.y > target.y, "the disc stands above the target, y {:.0}", at_.y);
    let far_w = detector_named(&trace_scene3(&s, &cfg_for(&s)), det).total_incident;
    assert!((far_w - near_w).abs() / near_w < 0.01, "{far_w:.1} W far vs {near_w:.1} W near");

    s.solar.as_mut().unwrap().standoff_mm = 0.0;
    s.resolve_solar().unwrap();
    let back = s.solar.as_ref().unwrap().standoff_mm;
    assert!((back - r.max(200.0)).abs() < 1e-9, "floor {back:.1} mm, radius {r:.1} mm");
}

/// A model drawn upside down (the room turned over about +X, a roof
/// heliostat below the target in world coordinates) says so, and the sun
/// comes from world -Y: the same sun turned with the room, the same
/// incidence, the same watts on the target.
#[test]
fn an_upside_down_model_has_its_sun_below() {
    let turn = |v: Vec3| Vec3::new(v.x, -v.y, -v.z);
    let watts = |upside_down: bool| {
        let mut s = Scene3::new();
        s.add_sun(almeria(), at(13, 0), 180.0);
        let target = Vec3::new(-20_000.0, 8_000.0, 3_000.0);
        let (target, n) = if upside_down { (turn(target), turn(-target.normalize())) } else { (target, -target.normalize()) };
        let det = add_rect_detector(&mut s, target, n, 1500.0, 1500.0, [16, 16]);
        s.add_heliostat(Vec3::ZERO, 1000.0, 1000.0, 0.95, HelioTarget::Object(det)).expect("a heliostat");
        let rig = s.solar.as_mut().unwrap();
        rig.upside_down = upside_down;
        rig.bands = 1;
        s.sources[0].samples.spatial_samples = 65_536;
        s.sources[0].samples.angular_samples = 1;
        let state = s.resolve_solar().unwrap();
        let back: Scene3 = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.solar.as_ref().unwrap().upside_down, upside_down, "the flag round-trips");
        let w = detector_named(&trace_scene3(&s, &cfg_for(&s)), det).total_incident;
        (state.sun_dir, state.heliostats[0].incidence_deg, w)
    };
    let (sun, inc, w) = watts(false);
    let (sun_down, inc_down, w_down) = watts(true);
    assert!(sun.y > 0.0 && sun_down.y < 0.0, "sun {sun:?}, upside down {sun_down:?}");
    assert!((turn(sun) - sun_down).length() < 1e-12, "the sun turns with the room");
    assert!((inc - inc_down).abs() < 1e-9, "incidence {inc} vs {inc_down}");
    assert!((w - w_down).abs() / w < 0.01, "{w:.1} W upright vs {w_down:.1} W upside down");
}

/// A new sun is the measured one: every wavelength of G173, a limb-darkened
/// disc with a clear day's aureole. A file written before those choices reads
/// as the flat disc and six lines it was written with.
#[test]
fn a_new_sun_is_measured_and_an_old_file_keeps_its_sun() {
    use oc_core::scene::Spectrum;
    use oc_core::scene3::AngularDistribution3;
    let (s, _, _, _) = field(at(12, 0));
    let src = &s.sources[0];
    assert!(matches!(src.spectrum, Spectrum::Continuum(ref v) if v.len() == 1661));
    match src.angular {
        AngularDistribution3::Sun { disc, csr, .. } => {
            assert_eq!(disc, oc_core::solar::sunshape::SunDisc::LimbDarkened);
            assert_eq!(csr, oc_core::heliostat::DEFAULT_CSR);
        }
        other => panic!("{other:?}"),
    }
    // The same rig as a file from before: the new keys taken out.
    let mut json: serde_json::Value = serde_json::to_value(&s).unwrap();
    let rig = json["solar"].as_object_mut().unwrap();
    for k in ["continuous", "disc", "csr"] {
        rig.remove(k);
    }
    let mut old: Scene3 = serde_json::from_value(json).unwrap();
    old.resolve_solar().unwrap();
    assert!(matches!(old.sources[0].spectrum, Spectrum::Lines(ref v) if v.len() == 6));
    assert!(matches!(old.sources[0].angular, AngularDistribution3::UniformConeSolidAngle { .. }));
}

/// Whatever the sun's shape and spectrum, it carries the DNI times its beam's
/// area times the band's share: the aureole and the limb move light around,
/// and the continuum shares it out, but none of them makes or loses any.
#[test]
fn the_shape_and_the_spectrum_keep_the_watts() {
    let (mut s, _, _, _) = field(at(12, 0));
    let watts = |s: &mut Scene3| s.resolve_solar().unwrap().source_flux_w;
    let measured = watts(&mut s);
    let rig = s.solar.as_mut().unwrap();
    rig.continuous = false;
    rig.disc = oc_core::solar::sunshape::SunDisc::Uniform;
    rig.csr = 0.0;
    let flat = watts(&mut s);
    assert!((measured - flat).abs() < 1e-9 * flat, "{measured} vs {flat}");
    let emitted: f64 = oc_trace::launch_from_scene(&s).iter().map(|l| l.flux).sum();
    assert!((emitted - flat).abs() < 1e-9 * flat);
}

/// The object ledger books every removal once: summed over objects and bands
/// it is the energy ledger, it moves no other number, and what it books on
/// the mirror is the mirror's absorption.
#[test]
fn the_object_ledger_is_the_ledger_object_by_object() {
    let (mut s, mirror, det, _) = field(at(12, 0));
    s.sources[0].samples.spatial_samples = 2048;
    s.sources[0].samples.angular_samples = 4;
    s.resolve_solar().expect("resolves");
    let plain = trace_scene3(&s, &cfg_for(&s));
    let cfg = TraceConfig3 { object_ledger: true, ..cfg_for(&s) };
    let booked = trace_scene3(&s, &cfg);
    assert_eq!(plain.ledger, booked.ledger, "the ledger is the same with or without the books");
    let d = |r: &oc_trace::TraceResult3| r.detectors.iter().find(|x| x.detector.0 == det.0).unwrap().total_incident;
    assert_eq!(d(&plain), d(&booked));
    assert!(plain.objects.is_empty());
    let removed: f64 = booked.objects.iter().map(|e| e.ledger.removed()).sum();
    let l = booked.ledger;
    assert!((removed - l.accounted()).abs() < 1e-9 * l.launched, "{removed} vs {}", l.accounted());
    // The mirror's row holds the mirror's absorption, and it saw light.
    let mi = s.surfaces.iter().position(|x| x.common.id == mirror).expect("the mirror is a surface");
    let m: f64 = booked
        .objects
        .iter()
        .filter(|e| e.object == oc_trace::ObjectRef3::Surface(mi))
        .map(|e| e.ledger.absorbed_surface)
        .sum();
    assert!(m > 0.0 && (m - l.absorbed_surface).abs() < 1e-9 * l.launched, "mirror {m} of {}", l.absorbed_surface);
    // Nothing named, nothing counted.
    for r in &booked.detectors.iter().find(|x| x.detector.0 == det.0).unwrap().records {
        assert_eq!(r.counted, 0);
    }
}
