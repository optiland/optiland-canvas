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
