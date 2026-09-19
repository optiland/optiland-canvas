//! Polarization in transport: what changes once a ray carries a state.
//!
//! Every test here fails if the tracer goes back to averaging s and p at each
//! surface and handing on unpolarized light. That model gets one interaction
//! exactly right and is wrong from the second polarization-sensitive element
//! onward, which is the case the canvas actually builds: a dichroic folds a
//! beam and something else stands in the folded arm.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::polarization::{PolRay, Stokes};
use oc_core::scene::Spectrum;
use oc_core::scene3::Scene3;
use oc_core::thinfilm::interface_amp;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, trace_scene3, LaunchRay3, TraceConfig3};

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        record_detector_hits: true,
        ..no_ghosts()
    }
}

/// The same configuration with the polarization state switched off, which is
/// the scalar-unpolarized approximation the preview runs.
fn scalar_cfg() -> TraceConfig3 {
    TraceConfig3 {
        polarization: false,
        ..cfg()
    }
}

fn launch_polarized(ray: Ray3, lambda_um: f64, stokes: Stokes, x: Vec3) -> LaunchRay3 {
    LaunchRay3 {
        ray,
        lambda_um,
        flux: stokes.s0,
        source: None,
        pol: Some(PolRay { stokes, x }),
    }
}

// ---------------------------------------------------------------------------
// The difference the model makes
// ---------------------------------------------------------------------------

/// The spec's own example, measured rather than asserted.
///
/// Unpolarized light through two tilted interfaces transmits
/// `½(Ts₁Ts₂ + Tp₁Tp₂)`, not `½(Ts₁+Tp₁) · ½(Ts₂+Tp₂)`. The first surface
/// enriches the beam in p, and p transmits better, so the true answer is the
/// larger of the two. It is the same effect that makes a pile of plates a
/// polarizer.
///
/// The two expressions differ by a quarter of a percent for a single tilted
/// slab of ordinary glass, which is small and is not nothing, and it grows with
/// every extra surface.
#[test]
fn two_glass_faces_transmit_more_than_the_mean_of_means() {
    let deg = 60.0f64;
    let n1 = 1.0;

    // A slab tilted so a horizontal ray meets it at `deg`, and a detector past
    // it. The beam crosses two interfaces: air to glass, then glass to air.
    //
    // At the d line the traced index is the nd the slab was built with, so the
    // closed form below uses the same number the tracer does and the two can be
    // compared at machine precision rather than at the size of a dispersion
    // step.
    let mut s = Scene3::new();
    let normal = Vec3::new(-deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
    let n2 = add_tilted_slab(&mut s, Vec3::ZERO, normal, 4.0, 1.5168);
    let det = add_rect_detector(
        &mut s,
        Vec3::new(60.0, 0.0, 0.0),
        -Vec3::X,
        40.0,
        40.0,
        [4, 4],
    );
    s.validate().expect("valid");

    let launch = vec![launch(
        Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap(),
        oc_core::material::LAMBDA_D,
        1.0,
    )];

    let polarized = trace3(&s, &launch, &cfg());
    let scalar = trace3(&s, &launch, &scalar_cfg());
    let got_pol = detector_named(&polarized, det).total_incident;
    let got_scalar = detector_named(&scalar, det).total_incident;

    // The closed form, from the interface coefficients alone.
    let cos_i = deg.to_radians().cos();
    let first = interface_amp(n1, n2, cos_i).powers();
    // Inside the glass the ray runs at the refracted angle.
    let sin_t = n1 * deg.to_radians().sin() / n2;
    let second = interface_amp(n2, n1, (1.0 - sin_t * sin_t).sqrt()).powers();
    let want_pol = 0.5 * (first.t_s * second.t_s + first.t_p * second.t_p);
    let want_scalar = 0.5 * (first.t_s + first.t_p) * 0.5 * (second.t_s + second.t_p);

    assert!(
        (got_pol - want_pol).abs() < 1e-9,
        "polarized: traced {got_pol} vs closed form {want_pol}"
    );
    assert!(
        (got_scalar - want_scalar).abs() < 1e-9,
        "scalar: traced {got_scalar} vs closed form {want_scalar}"
    );
    assert!(
        got_pol > got_scalar,
        "carrying the state must transmit more, not less: {got_pol} vs {got_scalar}"
    );
    let gap = (got_pol - got_scalar) / got_scalar;
    assert!(
        gap > 2e-3,
        "the gap at {deg} degrees should be a few parts per thousand, got {gap:.2e}"
    );
}

/// A tilted slab turns unpolarized light into partly polarized light, in the
/// direction physics says: the transmitted beam is enriched in p, so its `S1`
/// on the incidence frame is negative (p is the `ŷ` axis of that frame).
#[test]
fn a_tilted_slab_polarizes_what_goes_through_it() {
    let deg = 60.0f64;
    let mut s = Scene3::new();
    let normal = Vec3::new(-deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
    add_tilted_slab(&mut s, Vec3::ZERO, normal, 4.0, 1.5168);
    // The plane of incidence is the z = 0 plane, so ŝ runs along z. A detector
    // whose local +X is z reads S1 on that axis directly.
    let det = add_rect_detector(
        &mut s,
        Vec3::new(60.0, 0.0, 0.0),
        -Vec3::X,
        40.0,
        40.0,
        [4, 4],
    );
    s.validate().expect("valid");

    let launch = vec![launch(
        Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap(),
        0.5876,
        1.0,
    )];
    let res = trace3(&s, &launch, &cfg());
    let d = detector_named(&res, det);
    let hit = d.records.first().expect("one hit");

    assert!(hit.stokes.dop() > 0.09, "DOP is only {}", hit.stokes.dop());
    assert!(
        hit.stokes.is_realizable(1e-12),
        "the state must be one light can be in: {:?}",
        hit.stokes
    );
    assert!(
        hit.stokes.s3.abs() < 1e-12,
        "two dielectric interfaces add no circularity: S3 = {}",
        hit.stokes.s3
    );

    // Which polarization survived: project onto the incidence frame's ŝ, which
    // is the z axis here. More p than s means less along ŝ.
    let s_axis = Vec3::Z;
    let angle = oc_core::polarization::frame_angle(Vec3::X, hit.frame_x, s_axis);
    let on_s = hit.stokes.rotated(angle);
    assert!(
        on_s.s1 < 0.0,
        "the transmitted beam must favour p, so S1 on the s axis is negative: {}",
        on_s.s1
    );
}

/// With no polarization-sensitive element in the way, switching the model must
/// change nothing at all. A mirror and a thin lens are both polarization-blind,
/// so the two runs have to agree to the last bit.
#[test]
fn the_two_models_agree_when_nothing_can_tell_them_apart() {
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.55, 1.0)]);
    add_mirror(
        &mut s,
        Vec3::new(-20.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        20.0,
        0.9,
    );
    let det = add_rect_detector(
        &mut s,
        Vec3::new(-20.0, 40.0, 0.0),
        -Vec3::Y,
        20.0,
        20.0,
        [4, 4],
    );
    s.validate().expect("valid");

    let a = trace_scene3(&s, &cfg());
    let b = trace_scene3(&s, &scalar_cfg());
    let (fa, fb) = (
        detector_named(&a, det).total_incident,
        detector_named(&b, det).total_incident,
    );
    assert!(
        (fa - fb).abs() < 1e-15,
        "a mirror cannot distinguish the models: {fa} vs {fb}"
    );
    // And an unpolarized beam off an ideal mirror is still unpolarized.
    assert!(detector_named(&a, det).stokes_local.dop() < 1e-12);
}

// ---------------------------------------------------------------------------
// Elements acting on a state that was already polarized
// ---------------------------------------------------------------------------

/// Brewster's angle, in the tracer rather than in a formula.
///
/// A plate at Brewster is perfectly transparent to light polarized in the plane
/// of incidence, and not only at its front face: the refracted angle inside the
/// glass *is* the internal Brewster angle, so the back face passes p as well.
/// That is what a Brewster window in a laser cavity is for, and it is a
/// two-surface statement that a per-surface average cannot make.
///
/// Light polarized the other way reflects off both faces, and the plate returns
/// the geometric sum of them, `2R/(1+R)`. Averaging the two powers would give
/// both launches the same answer, which is the whole point.
#[test]
fn brewster_suppresses_one_polarization_and_not_the_other() {
    let n2 = 1.5168f64;
    let brewster = n2.atan();
    let deg = brewster.to_degrees();

    // A slab tilted so the ray meets the front face at Brewster, ghosts on so
    // the reflection is followed, and a detector to catch it.
    let mut s = Scene3::new();
    let normal = Vec3::new(-deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
    add_tilted_slab(&mut s, Vec3::ZERO, normal, 4.0, n2);
    // Reflecting +X off a normal tilted by `brewster` sends the beam along
    // (-cos 2t, sin 2t): still forward, because twice Brewster is past 90.
    let out = Vec3::new(-(2.0 * brewster).cos(), (2.0 * brewster).sin(), 0.0);
    let det = add_rect_detector(&mut s, out * 60.0, -out, 40.0, 40.0, [4, 4]);
    s.validate().expect("valid");

    let ray = Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap();
    // The plane of incidence is z = 0, so ŝ is along z and p̂ is in the plane.
    let s_axis = Vec3::Z;
    let p_axis = ray.dir.cross(s_axis);

    let ghosts = TraceConfig3 {
        ghosts: true,
        ..cfg()
    };
    // The d line, where the traced index is the nd the slab was built with, so
    // `brewster` above is the angle the tracer actually sees.
    let d_line = oc_core::material::LAMBDA_D;
    let in_plane = trace3(
        &s,
        &[launch_polarized(
            ray,
            d_line,
            Stokes::linear(1.0, 0.0),
            p_axis,
        )],
        &ghosts,
    );
    let out_of_plane = trace3(
        &s,
        &[launch_polarized(
            ray,
            d_line,
            Stokes::linear(1.0, 0.0),
            s_axis,
        )],
        &ghosts,
    );

    let refl_p = detector_named(&in_plane, det).total_incident;
    let refl_s = detector_named(&out_of_plane, det).total_incident;
    assert!(
        refl_p < 1e-12,
        "a Brewster plate reflects no p, off either face: got {refl_p}"
    );

    // Both faces reflect s, and the plate's faces are parallel, so the second
    // one's beam leaves alongside the first and lands on the same detector. The
    // bounces sum to `R + T R T + T R R R T + ... = 2R/(1+R)`, using `T = 1-R`
    // and the two faces having equal `R` by reversibility.
    let r1 = interface_amp(1.0, n2, brewster.cos()).powers().r_s;
    let want_s = 2.0 * r1 / (1.0 + r1);
    // The trace stops at the ghost budget, so it holds a truncated series and
    // must land just *under* the infinite sum, with the remainder named in the
    // ledger rather than lost. Each round trip inside the plate costs a factor
    // R^2, about 2.4%, so eight surface hits leave a part in ten million.
    let short_by = want_s - refl_s;
    assert!(
        short_by > 0.0 && short_by < 1e-6,
        "s must reflect the plate series, less what the ghost cap dropped:          {refl_s} vs {want_s}"
    );
    assert!(refl_s > 0.25, "and it is not a small number: {refl_s}");
}

/// Two dichroics in series, which is the case the scalar model cannot do.
///
/// The first one is where the error used to hide: with unpolarized light in,
/// the mean is the right flux, so nothing looks wrong. The second one is lit by
/// the polarized beam the first made, and its answer depends on a state the old
/// model had already thrown away.
#[test]
fn two_dichroics_in_series_disagree_with_the_mean_model() {
    // Both plates fold the beam, each near its own band edge, where s and p are
    // furthest apart.
    let edge = 560.0;
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(vec![(edge / 1000.0, 1.0)]);
    s.add_plate_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 3.0, 135.0, edge, true)
        .expect("first plate");
    // The reflected arm runs up +Y; the second plate stands in it and folds it
    // back along +X. A normal at -45 degrees does that; at +45 it would send the
    // beam backwards instead.
    s.add_plate_dichroic(Vec3::new(-35.0, 40.0, 0.0), 15.0, 3.0, -45.0, edge, true)
        .expect("second plate");
    let det = add_rect_detector(
        &mut s,
        Vec3::new(25.0, 40.0, 0.0),
        -Vec3::X,
        20.0,
        20.0,
        [4, 4],
    );
    s.validate().expect("valid");

    let polarized = trace_scene3(&s, &cfg());
    let scalar = trace_scene3(&s, &scalar_cfg());
    let a = detector_named(&polarized, det).total_incident;
    let b = detector_named(&scalar, det).total_incident;
    assert!(
        a > 0.0 && b > 0.0,
        "both models must deliver light: {a}, {b}"
    );

    let gap = (a - b).abs() / b.max(1e-30);
    assert!(
        gap > 0.05,
        "two dichroics at their edge must disagree by more than a rounding error, got {gap:.3}"
    );

    // And the beam arriving is strongly polarized, which is why they disagree.
    let dop = detector_named(&polarized, det).stokes_local.dop();
    assert!(dop > 0.5, "the folded beam should be well polarized: {dop}");
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// Rotating the whole scene, the launch included, must change nothing physical.
///
/// This is the test that catches a frame built out of a fixed world axis, which
/// works perfectly until the geometry is not in the plane somebody assumed.
#[test]
fn a_rigid_rotation_of_everything_changes_no_physics() {
    let deg = 55.0f64;
    let build = |rot: glam::DQuat| {
        let mut s = Scene3::new();
        let normal = rot * Vec3::new(-deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
        add_tilted_slab(&mut s, Vec3::ZERO, normal, 4.0, 1.5168);
        let dir = rot * Vec3::X;
        let det = add_rect_detector(&mut s, dir * 60.0, -dir, 40.0, 40.0, [4, 4]);
        s.validate().expect("valid");
        let ray = Ray3::new(dir * -60.0, dir).unwrap();
        (s, det, ray)
    };

    let (s0, d0, r0) = build(glam::DQuat::IDENTITY);
    let base = trace3(&s0, &[launch(r0, 0.5876, 1.0)], &cfg());
    let hit0 = detector_named(&base, d0).records[0];

    for (axis, angle) in [
        (Vec3::Z, 0.7),
        (Vec3::Y, -1.1),
        (Vec3::new(1.0, 1.0, 1.0).normalize(), 2.0),
    ] {
        let rot = glam::DQuat::from_axis_angle(axis, angle);
        let (s1, d1, r1) = build(rot);
        let turned = trace3(&s1, &[launch(r1, 0.5876, 1.0)], &cfg());
        let hit1 = detector_named(&turned, d1).records[0];

        assert!(
            (hit1.incident_flux - hit0.incident_flux).abs() < 1e-12,
            "flux moved under a rigid rotation: {} vs {}",
            hit1.incident_flux,
            hit0.incident_flux
        );
        assert!(
            (hit1.stokes.dop() - hit0.stokes.dop()).abs() < 1e-12,
            "degree of polarization moved: {} vs {}",
            hit1.stokes.dop(),
            hit0.stokes.dop()
        );
        // The frames differ, so S1 and S2 differ; the polarized flux does not.
        assert!(
            (hit1.stokes.polarized_flux() - hit0.stokes.polarized_flux()).abs() < 1e-12,
            "polarized flux moved"
        );
    }
}

/// Straight down the axis there is no plane of incidence, and the answer must
/// not depend on which azimuth the limit is approached from.
#[test]
fn normal_incidence_has_no_preferred_azimuth() {
    let mut s = Scene3::new();
    add_glass_slab(&mut s, -2.0, 2.0, 1.5168);
    let det = add_rect_detector(
        &mut s,
        Vec3::new(40.0, 0.0, 0.0),
        -Vec3::X,
        20.0,
        20.0,
        [4, 4],
    );
    s.validate().expect("valid");

    let ray = Ray3::new(Vec3::new(-40.0, 0.0, 0.0), Vec3::X).unwrap();
    let mut fluxes = Vec::new();
    for deg in [0.0f64, 30.0, 45.0, 75.0, 90.0] {
        let a = deg.to_radians();
        // Fully linear, at a different azimuth each time.
        let x = Vec3::new(0.0, a.cos(), a.sin());
        let res = trace3(
            &s,
            &[launch_polarized(ray, 0.5876, Stokes::linear(1.0, 0.0), x)],
            &cfg(),
        );
        let d = detector_named(&res, det);
        fluxes.push(d.total_incident);
        let hit = d.records[0];
        assert!(
            (hit.stokes.dop() - 1.0).abs() < 1e-12,
            "a normal-incidence slab depolarizes nothing: {}",
            hit.stokes.dop()
        );
    }
    let first = fluxes[0];
    for f in &fluxes {
        assert!(
            (f - first).abs() < 1e-12,
            "azimuth changed the answer at normal incidence: {fluxes:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// The ledger still balances
// ---------------------------------------------------------------------------

/// Flux is now the `S0` of a Stokes vector rather than a number of its own, and
/// the ledger has to come out the same.
#[test]
fn the_ledger_balances_with_a_state_on_every_ray() {
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.52, 1.0), (0.58, 1.0)]);
    s.add_plate_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 3.0, 135.0, 550.0, true)
        .expect("plate");
    s.validate().expect("valid");

    for c in [
        TraceConfig3 {
            ghosts: true,
            ..cfg()
        },
        scalar_cfg(),
    ] {
        let res = trace_scene3(&s, &c);
        let l = &res.ledger;
        let out = l.absorbed_volume
            + l.absorbed_surface
            + l.detected_incident
            + l.escaped
            + l.blocked_aperture
            + l.truncated_flux
            + l.numerical_loss;
        let residual = (l.launched - out).abs() / l.launched;
        assert!(residual < 1e-12, "ledger residual {residual:.3e}");
    }
}

// ---------------------------------------------------------------------------
// Ideal elements, in a scene rather than in a matrix
// ---------------------------------------------------------------------------

/// A bare bench: nothing but what the test puts on it, one unpolarized ray down
/// the axis, and a detector at the end. Returns the fraction of the launched
/// flux that arrives and the state it arrives in.
///
/// Every plate faces the beam (tilt 0, so its normal is `+X`), which puts its
/// local `+X` along world `-Z`. Only relative angles matter to any of these
/// tests, so that is a fact to know rather than one to work around.
fn bench(build: impl FnOnce(&mut Scene3)) -> (f64, oc_core::polarization::Stokes) {
    let mut s = Scene3::new();
    build(&mut s);
    let det = add_rect_detector(
        &mut s,
        Vec3::new(60.0, 0.0, 0.0),
        -Vec3::X,
        20.0,
        20.0,
        [4, 4],
    );
    s.validate().expect("valid");
    let launch = vec![launch(
        Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap(),
        0.55,
        1.0,
    )];
    let res = trace3(&s, &launch, &cfg());
    let d = detector_named(&res, det);
    (d.total_incident, d.stokes_local)
}

/// The same bench under the scalar-unpolarized model the preview runs.
fn scalar_bench(build: impl FnOnce(&mut Scene3)) -> f64 {
    let mut s = Scene3::new();
    build(&mut s);
    let det = add_rect_detector(
        &mut s,
        Vec3::new(60.0, 0.0, 0.0),
        -Vec3::X,
        20.0,
        20.0,
        [4, 4],
    );
    s.validate().expect("valid");
    let launch = vec![launch(
        Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap(),
        0.55,
        1.0,
    )];
    detector_named(&trace3(&s, &launch, &scalar_cfg()), det).total_incident
}

/// The elements' own plane written in world coordinates. A plate that faces
/// `+X` carries its local `+X` along world `-Z` and its local `+Y` along world
/// `+Y`, which is the fact the bench above records.
fn plate_x() -> Vec3 {
    -Vec3::Z
}

fn plate_y() -> Vec3 {
    Vec3::Y
}

/// A direction `tilt_deg` off the bench axis, leaning towards `azimuth_deg`
/// measured in the elements' own plane from their local `+X`.
fn skew_dir(tilt_deg: f64, azimuth_deg: f64) -> Vec3 {
    let (st, ct) = tilt_deg.to_radians().sin_cos();
    let (sa, ca) = azimuth_deg.to_radians().sin_cos();
    (Vec3::X * ct + (plate_x() * ca + plate_y() * sa) * st).normalize()
}

/// An element's axis as a ray travelling along `dir` sees it: the direction in
/// the element, projected into the plane the field lives in.
fn seen_axis(axis_deg: f64, dir: Vec3) -> Vec3 {
    let (sa, ca) = axis_deg.to_radians().sin_cos();
    let axis = plate_x() * ca + plate_y() * sa;
    (axis - dir * dir.dot(axis)).normalize()
}

/// The bench again, with the ray arriving off axis. Every element on it still
/// faces `+X`; the ray does not. Returns the arriving fraction and the
/// direction it arrived along, which is what the oracle needs.
fn skew_bench(
    tilt_deg: f64,
    azimuth_deg: f64,
    build: impl FnOnce(&mut Scene3),
) -> (f64, Vec3) {
    let dir = skew_dir(tilt_deg, azimuth_deg);
    let mut s = Scene3::new();
    build(&mut s);
    let det = add_rect_detector(&mut s, dir * 60.0, -dir, 20.0, 20.0, [4, 4]);
    s.validate().expect("valid");
    let launch = vec![launch(Ray3::new(dir * -60.0, dir).unwrap(), 0.55, 1.0)];
    let res = trace3(&s, &launch, &cfg());
    (detector_named(&res, det).total_incident, dir)
}

fn add_pol(s: &mut Scene3, x: f64, axis_deg: f64, extinction: f64) {
    s.add_polarizer(Vec3::new(x, 0.0, 0.0), 15.0, 0.0, axis_deg, extinction)
        .expect("polarizer");
}

/// The bench at a wavelength of the test's choosing, since a real waveplate is
/// only the plate it was specified as at one of them.
fn bench_at(lambda_um: f64, build: impl FnOnce(&mut Scene3)) -> (f64, oc_core::polarization::Stokes) {
    let mut s = Scene3::new();
    build(&mut s);
    let det = add_rect_detector(
        &mut s,
        Vec3::new(60.0, 0.0, 0.0),
        -Vec3::X,
        20.0,
        20.0,
        [4, 4],
    );
    s.validate().expect("valid");
    let launch = vec![launch(
        Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap(),
        lambda_um,
        1.0,
    )];
    let res = trace3(&s, &launch, &cfg());
    let d = detector_named(&res, det);
    (d.total_incident, d.stokes_local)
}

/// One skew ray in a state of the test's choosing, written on a frame the test
/// picks, so what reaches the detector can be read against an exact answer
/// without a polarizer's own projection getting in the way.
fn skew_state_bench(
    tilt_deg: f64,
    azimuth_deg: f64,
    stokes: oc_core::polarization::Stokes,
    frame_x: Vec3,
    build: impl FnOnce(&mut Scene3),
) -> oc_core::polarization::Stokes {
    let dir = skew_dir(tilt_deg, azimuth_deg);
    let mut s = Scene3::new();
    build(&mut s);
    let det = add_rect_detector(&mut s, dir * 60.0, -dir, 20.0, 20.0, [4, 4]);
    s.validate().expect("valid");
    let launch = vec![launch_polarized(
        Ray3::new(dir * -60.0, dir).unwrap(),
        0.55,
        stokes,
        frame_x,
    )];
    detector_named(&trace3(&s, &launch, &cfg()), det).stokes_local
}

/// A waveplate cut from crystal quartz, facing the bench axis.
fn add_quartz_wave(s: &mut Scene3, x: f64, axis_deg: f64, nominal_deg: f64, order: u32) {
    let id = s
        .add_retarder(Vec3::new(x, 0.0, 0.0), 15.0, 0.0, axis_deg, nominal_deg)
        .expect("retarder");
    assert!(
        s.set_retarder_crystal(id, Some(oc_core::scene3::Waveplate::quartz(order, 0.55))),
        "crystal"
    );
}

/// A perfect polarizer laminated in glass of index `n_sub`, facing the bench
/// axis.
fn add_pol_in_glass(s: &mut Scene3, x: f64, axis_deg: f64, n_sub: f64) {
    let id = s
        .add_polarizer(Vec3::new(x, 0.0, 0.0), 15.0, 0.0, axis_deg, 0.0)
        .expect("polarizer");
    assert!(s.set_polarizing_substrate(id, n_sub), "substrate");
}

/// A polarizer whose contrast came off a datasheet: a curve against angle.
fn add_pol_table(s: &mut Scene3, x: f64, axis_deg: f64, aoi: &[f64], leak: &[f64]) {
    let id = s
        .add_polarizer(Vec3::new(x, 0.0, 0.0), 15.0, 0.0, axis_deg, 0.0)
        .expect("polarizer");
    assert!(
        s.set_polarizer_extinction_table(id, aoi, leak),
        "the table is well formed"
    );
}

/// The power reflectances `(s, p)` of one air-to-glass face, straight from
/// Fresnel and not from the kernel, so a test that uses it is an oracle and not
/// a mirror.
fn face_reflectances(n: f64, aoi_deg: f64) -> (f64, f64) {
    let ci = aoi_deg.to_radians().cos();
    let st = aoi_deg.to_radians().sin() / n;
    let ct = (1.0 - st * st).sqrt();
    let rs = (ci - n * ct) / (ci + n * ct);
    let rp = (n * ci - ct) / (n * ci + ct);
    (rs * rs, rp * rp)
}

fn add_wave(s: &mut Scene3, x: f64, axis_deg: f64, retardance_deg: f64) {
    s.add_retarder(Vec3::new(x, 0.0, 0.0), 15.0, 0.0, axis_deg, retardance_deg)
        .expect("retarder");
}

/// Malus, built out of two parts in a scene.
///
/// The first polarizer halves unpolarized light, which is where the flux goes;
/// the second passes the cosine squared of what is left. A model that averages
/// s and p at each surface would pass a quarter of the light at every angle,
/// crossed included.
#[test]
fn two_polarizers_obey_malus_law() {
    let (open, st) = bench(|s| {
        add_pol(s, -20.0, 0.0, 0.0);
        add_pol(s, 0.0, 0.0, 0.0);
    });
    assert!(
        (open - 0.5).abs() < 1e-12,
        "parallel polarizers pass half of unpolarized light: {open}"
    );
    assert!(
        (st.dop() - 1.0).abs() < 1e-12,
        "and what passes is fully polarized: {}",
        st.dop()
    );

    for deg in [0.0f64, 22.5, 30.0, 45.0, 60.0, 90.0] {
        let (got, _) = bench(|s| {
            add_pol(s, -20.0, 0.0, 0.0);
            add_pol(s, 0.0, deg, 0.0);
        });
        let want = 0.5 * deg.to_radians().cos().powi(2);
        assert!(
            (got - want).abs() < 1e-12,
            "at {deg} degrees: {got} vs {want}"
        );
    }
}

/// Crossed polarizers pass nothing, and an imperfect pair passes its extinction.
#[test]
fn crossed_polarizers_go_dark() {
    let (dark, _) = bench(|s| {
        add_pol(s, -20.0, 0.0, 0.0);
        add_pol(s, 0.0, 90.0, 0.0);
    });
    assert!(dark < 1e-15, "crossed polarizers pass nothing: {dark}");

    // A pair of sheet polarizers with a 1e-4 power extinction. Unpolarized in:
    // half the flux sits on each axis. The component the first plate passes
    // whole is then blocked to 1e-4, and the one it blocks to 1e-4 passes the
    // second whole, so both halves contribute the same product.
    let (leaked, _) = bench(|s| {
        add_pol(s, -20.0, 0.0, 1e-4);
        add_pol(s, 0.0, 90.0, 1e-4);
    });
    let want = 0.5 * (1e-4 + 1e-4);
    assert!(
        (leaked - want).abs() < 1e-15,
        "a real sheet leaks its extinction: {leaked} vs {want}"
    );
}

/// The three-polarizer paradox: inserting a third plate between two crossed
/// ones lets light through.
///
/// Nothing in a flux-only model can produce this. It happens because the middle
/// polarizer does not filter the beam, it rewrites its state, and the third one
/// then has something to pass.
#[test]
fn a_third_polarizer_between_two_crossed_ones_lets_light_through() {
    let (got, _) = bench(|s| {
        add_pol(s, -30.0, 0.0, 0.0);
        add_pol(s, -15.0, 45.0, 0.0);
        add_pol(s, 0.0, 90.0, 0.0);
    });
    // Half through the first, then the cosine squared of 45 degrees at each of
    // the other two.
    assert!(
        (got - 0.125).abs() < 1e-12,
        "three polarizers pass an eighth: {got}"
    );

    // The same two without the middle one pass nothing, which is the half of
    // the paradox that makes it one.
    let (dark, _) = bench(|s| {
        add_pol(s, -30.0, 0.0, 0.0);
        add_pol(s, 0.0, 90.0, 0.0);
    });
    assert!(dark < 1e-15, "and without it, nothing: {dark}");
}

/// One quarter-wave plate at 45 degrees to the incoming linear state produces
/// circular light: fully polarized, none of it linear, all of it in S3.
#[test]
fn a_quarter_wave_plate_makes_circular_light_in_a_scene() {
    let (flux, st) = bench(|s| {
        add_pol(s, -20.0, 0.0, 0.0);
        add_wave(s, 0.0, 45.0, 90.0);
    });
    assert!(
        (flux - 0.5).abs() < 1e-12,
        "a retarder is lossless: {flux} of the half the polarizer passed"
    );
    assert!((st.dop() - 1.0).abs() < 1e-12, "still fully polarized");
    assert!(
        st.dolp() < 1e-12,
        "and no longer linear at all: DoLP {}",
        st.dolp()
    );
    assert!(
        (st.s3.abs() - st.s0).abs() < 1e-12,
        "all of it is circular: S3 {} of S0 {}",
        st.s3,
        st.s0
    );
}

/// Two quarter waves at the same angle make a half wave, which turns a linear
/// state at 0 into one at 90, where a crossed analyzer passes all of it.
#[test]
fn two_quarter_waves_rotate_linear_light_through_a_crossed_analyzer() {
    let (got, _) = bench(|s| {
        add_pol(s, -30.0, 0.0, 0.0);
        add_wave(s, -20.0, 45.0, 90.0);
        add_wave(s, -10.0, 45.0, 90.0);
        add_pol(s, 0.0, 90.0, 0.0);
    });
    assert!(
        (got - 0.5).abs() < 1e-12,
        "the crossed analyzer should now pass everything reaching it: {got}"
    );
}

/// A half-wave plate turns a linear state by twice the angle between them, and
/// loses nothing doing it.
#[test]
fn a_half_wave_plate_rotates_and_does_not_absorb() {
    for deg in [10.0f64, 22.5, 45.0] {
        let (got, _) = bench(|s| {
            add_pol(s, -30.0, 0.0, 0.0);
            add_wave(s, -15.0, deg, 180.0);
            // An analyzer at twice the plate's angle should pass all of it.
            add_pol(s, 0.0, 2.0 * deg, 0.0);
        });
        assert!(
            (got - 0.5).abs() < 1e-12,
            "a half wave at {deg} rotates to {}: passed {got}",
            2.0 * deg
        );
    }
}

/// With the state switched off, a polarizer can only be a 50% attenuator, so a
/// crossed pair looks bright. That is the preview's model, and the reason a
/// number has to say which model produced it.
#[test]
fn the_scalar_model_cannot_cross_two_polarizers() {
    let got = scalar_bench(|s| {
        add_pol(s, -20.0, 0.0, 0.0);
        add_pol(s, 0.0, 90.0, 0.0);
    });
    assert!(
        (got - 0.25).abs() < 1e-12,
        "the scalar model passes a quarter whatever the angles: {got}"
    );
}

/// A polarizer on a body patch has no axis to speak of, so the tracer refuses
/// it rather than picking one.
#[test]
fn a_polarizer_on_a_solid_is_refused() {
    let mut s = Scene3::new();
    add_glass_slab(&mut s, -2.0, 2.0, 1.5168);
    let b = s.bodies.last_mut().unwrap();
    let r = oc_core::scene3::SurfaceResponse::Polarizer {
        axis_deg: 0.0,
        extinction: oc_core::scene3::AngularScalar::Constant(0.0),
        substrate_index: 1.0,
    };
    b.surface.default.from_outside = r.clone();
    b.surface.default.from_inside = r;
    s.validate()
        .expect("the scene is well formed; the physics is what refuses");

    let launch = vec![launch(
        Ray3::new(Vec3::new(-40.0, 0.0, 0.0), Vec3::X).unwrap(),
        0.55,
        1.0,
    )];
    let res = trace3(&s, &launch, &cfg());
    assert!(
        !res.capability_report.rejected.is_empty(),
        "a polarizing element on a solid must be reported, not guessed at"
    );
}

/// Crossed polarizers leak off axis, which is why a display has a viewing
/// angle.
///
/// Two ideal plates crossed in their own plane are not crossed in the
/// transverse plane of a ray that arrives skew: projecting two perpendicular
/// directions onto a tilted plane does not keep them perpendicular. What leaks
/// is Malus against whatever is left of the right angle, and by thirty degrees
/// off axis that is a percent of the light.
///
/// A model that turned the axis inside the ray's own transverse plane instead
/// of projecting it out of the element would keep the pair exactly crossed from
/// every direction and report a hard zero here.
#[test]
fn crossed_polarizers_leak_off_axis() {
    for tilt in [10.0f64, 20.0, 30.0] {
        let (got, dir) = skew_bench(tilt, 45.0, |s| {
            add_pol(s, -8.0, 0.0, 0.0);
            add_pol(s, 8.0, 90.0, 0.0);
        });
        // Half the unpolarized light survives the first plate, and the second
        // passes the cosine squared of the angle between the axes this ray
        // sees.
        let want = 0.5 * seen_axis(0.0, dir).dot(seen_axis(90.0, dir)).powi(2);
        assert!(
            (got - want).abs() < 1e-12,
            "{tilt} degrees off axis: {got} vs {want}"
        );
        assert!(got > 1e-5, "and it is a leak, not a rounding: {got}");
    }
}

/// The leak comes from the geometry rather than from a fudge factor: it
/// vanishes when the ray leans along either axis and peaks between them.
///
/// Leaning along an axis leaves that axis alone and shortens the other without
/// turning it, so the two stay perpendicular and the pair stays dark.
#[test]
fn the_off_axis_leak_follows_the_azimuth() {
    let crossed = |az: f64| {
        skew_bench(30.0, az, |s| {
            add_pol(s, -8.0, 0.0, 0.0);
            add_pol(s, 8.0, 90.0, 0.0);
        })
        .0
    };
    for az in [0.0f64, 90.0, 180.0, 270.0] {
        let got = crossed(az);
        assert!(got < 1e-15, "leaning along an axis at {az} leaks {got}");
    }
    let between = crossed(45.0);
    assert!(between > 5e-3, "and leaning between them leaks: {between}");
}

/// A polarizer whose axis leans out of the ray's transverse plane still passes
/// half of unpolarized light, because an ideal one passes all of whatever lies
/// along the axis it can see and none of the rest whatever the angle.
#[test]
fn one_polarizer_off_axis_still_halves_unpolarized_light() {
    for tilt in [0.0f64, 15.0, 30.0, 45.0] {
        let (got, _) = skew_bench(tilt, 45.0, |s| add_pol(s, 0.0, 30.0, 0.0));
        assert!(
            (got - 0.5).abs() < 1e-12,
            "at {tilt} degrees off axis: {got}"
        );
    }
}


// ---------------------------------------------------------------------------
// The glass a polarizing element is carried in
// ---------------------------------------------------------------------------

/// A polarizer in glass loses what its faces reflect before it has done
/// anything a polarizer does.
///
/// Two air-to-glass faces pass 0.9216 of ordinary crown at normal incidence,
/// which is part of why a real sheet polarizer's transmission for its own pass
/// axis is well under one half.
#[test]
fn the_faces_of_a_laminated_polarizer_cost_what_they_reflect() {
    let (bare, _) = bench(|s| add_pol(s, 0.0, 0.0, 0.0));
    assert!((bare - 0.5).abs() < 1e-12, "a bare layer halves it: {bare}");

    let n = 1.5;
    let (got, _) = bench(|s| add_pol_in_glass(s, 0.0, 0.0, n));
    let (rs, _) = face_reflectances(n, 0.0);
    let want = 0.5 * (1.0 - rs).powi(2);
    assert!((got - want).abs() < 1e-12, "in glass: {got} vs {want}");
    assert!((want - 0.4608).abs() < 5e-5, "and that number is 0.4608: {want}");
}

/// Tilt a laminated polarizer and its transmission splits by which axis it
/// passes: the faces let p through better than s, and by forty-five degrees
/// that is a fifth more light for a polarizer that passes p than for one that
/// passes s.
///
/// This is the angle dependence an ideal layer does not have and a real part
/// cannot avoid. The oracle is Fresnel twice over, which is exact here because
/// the two faces are reversible: a face meeting air at some angle from outside
/// reflects what it reflects meeting glass at the refracted angle from inside.
#[test]
fn a_laminated_polarizer_transmits_by_which_axis_it_passes() {
    let n = 1.5;
    // Leaning in the plane of the element's local `+X` puts the s direction on
    // its local `+Y`, which is an axis of 90 degrees, and p on its local `+X`.
    for aoi in [0.0f64, 20.0, 45.0, 60.0] {
        let (got_s, _) = skew_bench(aoi, 0.0, |s| add_pol_in_glass(s, 0.0, 90.0, n));
        let (got_p, _) = skew_bench(aoi, 0.0, |s| add_pol_in_glass(s, 0.0, 0.0, n));
        let (rs, rp) = face_reflectances(n, aoi);
        let want_s = 0.5 * (1.0 - rs).powi(2);
        let want_p = 0.5 * (1.0 - rp).powi(2);
        assert!(
            (got_s - want_s).abs() < 1e-12,
            "s axis at {aoi}: {got_s} vs {want_s}"
        );
        assert!(
            (got_p - want_p).abs() < 1e-12,
            "p axis at {aoi}: {got_p} vs {want_p}"
        );
    }
    let s45 = skew_bench(45.0, 0.0, |s| add_pol_in_glass(s, 0.0, 90.0, n)).0;
    let p45 = skew_bench(45.0, 0.0, |s| add_pol_in_glass(s, 0.0, 0.0, n)).0;
    let ratio = p45 / s45;
    assert!(ratio > 1.15, "p beats s by a fifth at 45 degrees: {ratio}");
    // A bare layer has none of this: both axes pass exactly half.
    let bare_s = skew_bench(45.0, 0.0, |s| add_pol(s, 0.0, 90.0, 0.0)).0;
    let bare_p = skew_bench(45.0, 0.0, |s| add_pol(s, 0.0, 0.0, 0.0)).0;
    assert!(
        (bare_s - 0.5).abs() < 1e-12 && (bare_p - 0.5).abs() < 1e-12,
        "bare: {bare_s} and {bare_p}"
    );
}

/// What a laminated face reflects is a ghost like any other: followed when
/// ghosts are on, named when they are off, and absorbed in neither case.
#[test]
fn what_a_laminated_face_reflects_is_not_absorbed() {
    let n = 1.5;
    let (rs, _) = face_reflectances(n, 0.0);
    for ghosts in [false, true] {
        let mut s = Scene3::new();
        add_pol_in_glass(&mut s, 0.0, 0.0, n);
        add_rect_detector(
            &mut s,
            Vec3::new(60.0, 0.0, 0.0),
            -Vec3::X,
            20.0,
            20.0,
            [4, 4],
        );
        s.validate().expect("valid");
        let launch = vec![launch(
            Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap(),
            0.55,
            1.0,
        )];
        let res = trace3(&s, &launch, &TraceConfig3 { ghosts, ..cfg() });
        let l = &res.ledger;
        let out = l.absorbed_volume
            + l.absorbed_surface
            + l.detected_incident
            + l.escaped
            + l.blocked_aperture
            + l.truncated_flux
            + l.numerical_loss;
        let residual = (l.launched - out).abs() / l.launched;
        assert!(residual < 1e-12, "ledger with ghosts {ghosts}: {residual:.3e}");
        if ghosts {
            assert!(
                (l.escaped - rs).abs() < 1e-12,
                "the reflection goes back out: {} vs {rs}",
                l.escaped
            );
        } else {
            assert!(
                (l.truncated_flux - rs).abs() < 1e-12,
                "and is named when it is not followed: {} vs {rs}",
                l.truncated_flux
            );
        }
    }
}

/// A waveplate in glass is no longer lossless, and that is the point: the
/// retardance is still exact and the faces still take their cut.
#[test]
fn a_laminated_waveplate_keeps_its_retardance_and_loses_its_faces() {
    let n = 1.5;
    let (rs, _) = face_reflectances(n, 0.0);
    let faces = (1.0 - rs).powi(2);

    let (flux, st) = bench(|s| {
        add_pol(s, -20.0, 0.0, 0.0);
        let id = s
            .add_retarder(Vec3::new(0.0, 0.0, 0.0), 15.0, 0.0, 45.0, 90.0)
            .expect("retarder");
        assert!(s.set_polarizing_substrate(id, n), "substrate");
    });
    assert!(
        (flux - 0.5 * faces).abs() < 1e-12,
        "the faces take their cut: {flux} vs {}",
        0.5 * faces
    );
    // Still a quarter wave: fully polarized, and none of it linear.
    assert!((st.dop() - 1.0).abs() < 1e-12, "fully polarized: {}", st.dop());
    assert!(st.dolp() < 1e-12, "and circular: {}", st.dolp());
}


// ---------------------------------------------------------------------------
// A waveplate made of something
// ---------------------------------------------------------------------------

/// A quartz quarter wave is a quarter wave at the wavelength it was cut for and
/// nothing so tidy anywhere else.
///
/// With the indices held constant the retardance is exactly the nominal times
/// `λ0/λ`, so a plate cut for 550 nm gives 110 degrees at 450 and the light
/// that came out circular comes out elliptical. An ideal retarder in the same
/// place gives 90 degrees at every colour.
#[test]
fn a_quartz_quarter_wave_is_not_a_quarter_wave_in_blue() {
    let scene = |s: &mut Scene3| {
        add_pol(s, -20.0, 0.0, 0.0);
        add_quartz_wave(s, 0.0, 45.0, 90.0, 0);
    };
    // At the design wavelength: circular, which is a full S3 and no linear
    // part at all.
    let (_, st) = bench_at(0.55, scene);
    assert!((st.dop() - 1.0).abs() < 1e-12, "dop {}", st.dop());
    assert!(st.dolp() < 1e-12, "circular at 550: dolp {}", st.dolp());

    // At 450 nm the same plate retards by 110 degrees.
    let want = 90.0f64.to_radians() * 0.55 / 0.45;
    let (flux, st) = bench_at(0.45, |s| {
        add_pol(s, -20.0, 0.0, 0.0);
        add_quartz_wave(s, 0.0, 45.0, 90.0, 0);
    });
    assert!((st.dop() - 1.0).abs() < 1e-12, "still fully polarized");
    assert!(
        ((st.s3 / st.s0).abs() - want.sin()).abs() < 1e-12,
        "S3 says {} and the plate says {}",
        (st.s3 / st.s0).abs(),
        want.sin()
    );
    assert!(
        (st.dolp() - want.cos().abs()).abs() < 1e-12,
        "and the linear part says {} against {}",
        st.dolp(),
        want.cos().abs()
    );
    assert!(
        (flux - 0.5).abs() < 1e-12,
        "a bare plate is still lossless: {flux}"
    );

    // An ideal retarder does not care what colour the light is.
    let (_, ideal) = bench_at(0.45, |s| {
        add_pol(s, -20.0, 0.0, 0.0);
        add_wave(s, 0.0, 45.0, 90.0);
    });
    assert!(ideal.dolp() < 1e-12, "ideal at 450: dolp {}", ideal.dolp());
}

/// A zero-order plate holds its retardance off axis and a multi-order one does
/// not, because the error is a fraction of the whole retardance rather than of
/// the quarter wave anybody can see.
///
/// The input is launched already linear at 45 degrees to the plate's axis *as
/// this ray sees it*, so the only thing being read is what the plate did. The
/// expected retardance comes from the closed form with the geometry worked out
/// here: the transverse index is the ray's own `sin θ`, and the optic axis of a
/// positive crystal sits across the fast axis the element names.
#[test]
fn a_multi_order_plate_slips_off_axis_where_a_zero_order_one_holds() {
    let tilt = 15.0f64;
    // Leaning along the fast axis, which is across the optic axis of a
    // positive crystal and so the direction the retardance moves furthest in.
    // Halfway between the two the shift very nearly cancels, because the
    // coefficients either side are 1/(2·n_o·n_e) and 1/(2·n_o²) and quartz
    // barely tells those apart.
    let azimuth = 45.0f64;
    let fast = 45.0f64;
    let dir = skew_dir(tilt, azimuth);
    let seen = seen_axis(fast, dir);
    // Linear at 45 degrees to the axis the plate presents to this ray.
    let input = oc_core::polarization::Stokes::linear(1.0, std::f64::consts::FRAC_PI_4);

    let q = tilt.to_radians().sin();
    let cos_axis = (azimuth - (fast + 90.0)).to_radians().cos();
    let quartz = oc_core::scene3::Waveplate::quartz(0, 0.55);

    for order in [0u32, 10] {
        let plate = oc_core::scene3::Waveplate::quartz(order, 0.55);
        let want = oc_core::polarization::a_plate_retardance(
            quartz.n_o,
            quartz.n_e,
            plate.thickness_um(90.0),
            0.55,
            q,
            cos_axis,
        );
        let st = skew_state_bench(tilt, azimuth, input, seen, |s| {
            add_quartz_wave(s, 0.0, fast, 90.0, order);
        });
        assert!((st.dop() - 1.0).abs() < 1e-12, "lossless and pure");
        assert!(
            ((st.s3 / st.s0).abs() - want.sin().abs()).abs() < 1e-9,
            "order {order}: S3 {} vs sin of {} degrees",
            (st.s3 / st.s0).abs(),
            want.to_degrees()
        );
        assert!(
            (st.dolp() - want.cos().abs()).abs() < 1e-9,
            "order {order}: dolp {} vs cos of {} degrees",
            st.dolp(),
            want.to_degrees()
        );
    }

    // And the claim in plain terms: at fifteen degrees off axis the zero-order
    // plate is still making nearly circular light and the tenth-order one is
    // not making it at all.
    let zero = skew_state_bench(tilt, azimuth, input, seen, |s| {
        add_quartz_wave(s, 0.0, fast, 90.0, 0);
    });
    let tenth = skew_state_bench(tilt, azimuth, input, seen, |s| {
        add_quartz_wave(s, 0.0, fast, 90.0, 10);
    });
    assert!(zero.dolp() < 0.05, "zero order holds: {}", zero.dolp());
    assert!(tenth.dolp() > 0.5, "tenth order does not: {}", tenth.dolp());
}


// ---------------------------------------------------------------------------
// Contrast off a datasheet
// ---------------------------------------------------------------------------

/// A polarizer can be told where it stops working, because nothing the kernel
/// computes about an ideal layer will ever say so.
///
/// A wire grid loses contrast off axis, a thin-film polarizer gives up a couple
/// of degrees from its design angle, and a Glan prism has a hard acceptance
/// cone. None of that follows from a stack, so it comes from measurement. The
/// light launched here is already linear along the axis the polarizer blocks,
/// so what arrives *is* the leak and nothing else.
#[test]
fn a_polarizer_can_lose_its_contrast_off_axis() {
    // Four angles off a plausible wire-grid datasheet.
    let aoi = [0.0f64, 10.0, 25.0, 40.0];
    let leak = [1e-4f64, 3e-4, 5e-3, 8e-2];
    let blocked = oc_core::polarization::Stokes::linear(1.0, std::f64::consts::FRAC_PI_2);

    for (deg, want) in aoi.iter().zip(leak) {
        let dir = skew_dir(*deg, 0.0);
        let st = skew_state_bench(*deg, 0.0, blocked, seen_axis(0.0, dir), |s| {
            add_pol_table(s, 0.0, 0.0, &aoi, &leak);
        });
        assert!(
            (st.s0 - want).abs() < 1e-12,
            "at {deg} degrees the datasheet says {want} and the trace says {}",
            st.s0
        );
    }

    // Between samples the curve is interpolated, not stepped.
    let dir = skew_dir(17.5, 0.0);
    let st = skew_state_bench(17.5, 0.0, blocked, seen_axis(0.0, dir), |s| {
        add_pol_table(s, 0.0, 0.0, &aoi, &leak);
    });
    let want = 3e-4 + (5e-3 - 3e-4) * (17.5 - 10.0) / (25.0 - 10.0);
    assert!((st.s0 - want).abs() < 1e-12, "at 17.5: {} vs {want}", st.s0);

    // Past the last measurement the curve holds its endpoint, which is the
    // honest thing to do with a range nobody measured.
    let dir = skew_dir(55.0, 0.0);
    let st = skew_state_bench(55.0, 0.0, blocked, seen_axis(0.0, dir), |s| {
        add_pol_table(s, 0.0, 0.0, &aoi, &leak);
    });
    assert!((st.s0 - 8e-2).abs() < 1e-12, "past the table: {}", st.s0);

    // And an ideal layer keeps its contrast wherever it is used, which is the
    // thing a datasheet disagrees with.
    for deg in [0.0f64, 25.0, 55.0] {
        let dir = skew_dir(deg, 0.0);
        let st = skew_state_bench(deg, 0.0, blocked, seen_axis(0.0, dir), |s| {
            add_pol(s, 0.0, 0.0, 1e-4);
        });
        assert!((st.s0 - 1e-4).abs() < 1e-15, "ideal at {deg}: {}", st.s0);
    }
}

/// A contrast curve that could not have come from a measurement is refused
/// rather than repaired.
#[test]
fn a_contrast_curve_has_to_be_a_curve() {
    let mut s = Scene3::new();
    let id = s
        .add_polarizer(Vec3::ZERO, 15.0, 0.0, 0.0, 0.0)
        .expect("polarizer");

    let bad: [(&[f64], &[f64], &str); 5] = [
        (&[0.0], &[1e-4], "one sample interpolates nothing"),
        (&[0.0, 10.0], &[1e-4], "the arrays disagree in length"),
        (&[10.0, 0.0], &[1e-4, 1e-3], "the angles run backwards"),
        (&[0.0, 90.0], &[1e-4, 1e-3], "90 degrees is grazing, not an angle of use"),
        (&[0.0, 10.0], &[1e-4, 1.5], "an extinction above 1 is more light than arrived"),
    ];
    for (aoi, leak, why) in bad {
        assert!(
            !s.set_polarizer_extinction_table(id, aoi, leak),
            "refused: {why}"
        );
    }
    // And the element still has the contrast it started with.
    assert!(matches!(
        s.polarizer_extinction(id),
        Some(oc_core::scene3::AngularScalar::Constant(v)) if *v == 0.0
    ));
    assert!(s.set_polarizer_extinction_table(id, &[0.0, 10.0], &[1e-4, 1e-3]));
    assert!(matches!(
        s.polarizer_extinction(id),
        Some(oc_core::scene3::AngularScalar::Tabulated { .. })
    ));
    s.validate().expect("a measured polarizer is a valid scene");

    // Setting one number again puts it back to an ideal layer.
    assert!(s.set_polarizer_extinction(id, 1e-4));
    assert!(matches!(
        s.polarizer_extinction(id),
        Some(oc_core::scene3::AngularScalar::Constant(_))
    ));
}


// ---------------------------------------------------------------------------
// The one polarizing part whose behaviour is derived rather than declared
// ---------------------------------------------------------------------------

/// Build a polarizing cube at the origin, with a detector past it and another
/// above it, and trace one unpolarized ray in along `+X`.
fn pbs_bench(periods: u32) -> (oc_core::polarization::Stokes, oc_core::polarization::Stokes) {
    let mut s = Scene3::new();
    s.add_polarizing_beam_splitter(Vec3::ZERO, 8.0, 135.0, 550.0, periods, 2.35, 1.46)
        .expect("a cube");
    let through = add_rect_detector(
        &mut s,
        Vec3::new(60.0, 0.0, 0.0),
        -Vec3::X,
        20.0,
        20.0,
        [4, 4],
    );
    let up = add_rect_detector(
        &mut s,
        Vec3::new(0.0, 60.0, 0.0),
        -Vec3::Y,
        20.0,
        20.0,
        [4, 4],
    );
    s.validate().expect("valid");
    let launch = vec![launch(
        Ray3::new(Vec3::new(-60.0, 0.0, 0.0), Vec3::X).unwrap(),
        0.55,
        1.0,
    )];
    let res = trace3(
        &s,
        &launch,
        &TraceConfig3 {
            ghosts: true,
            ..cfg()
        },
    );
    (
        detector_named(&res, through).stokes_local,
        detector_named(&res, up).stokes_local,
    )
}

/// The cube and its diagonal are two objects that say they are one part.
///
/// A solid with an interface inside it is not something the scene model can
/// hold yet, so the part is a body and a surface. They share a label, so the
/// object list calls them one thing, and moving one moves only that one.
#[test]
fn a_polarizing_cube_is_two_objects_with_one_name() {
    let mut s = Scene3::new();
    let (cube, diagonal) = s
        .add_polarizing_beam_splitter(Vec3::ZERO, 8.0, 135.0, 550.0, 12, 2.35, 1.46)
        .expect("a cube");
    assert_ne!(cube, diagonal);
    let label = |id: u64| -> String {
        s.bodies
            .iter()
            .map(|b| &b.common)
            .chain(s.surfaces.iter().map(|x| &x.common))
            .find(|c| c.id.0 == id)
            .map(|c| c.label.clone())
            .unwrap_or_default()
    };
    assert_eq!(label(cube), label(diagonal));
    assert!(label(cube).contains("polarizing"), "{}", label(cube));

    // A cube smaller than a millimetre across is not a part, and neither is
    // one whose coating pair cannot make a glass.
    assert!(s
        .add_polarizing_beam_splitter(Vec3::ZERO, 0.1, 135.0, 550.0, 12, 2.35, 1.46)
        .is_none());
    assert!(s
        .add_polarizing_beam_splitter(Vec3::ZERO, 8.0, f64::NAN, 550.0, 12, 2.35, 1.46)
        .is_none());
}

/// A polarizing cube sorts unpolarized light into two beams that are polarized
/// the opposite ways, and it does it by reflecting rather than by absorbing.
///
/// Nothing about this is declared. The stack is a plain quarter-wave mirror;
/// what makes it a polarizer is that it sits at the Brewster angle between its
/// own layers, where every p reflection inside it vanishes. The contrast comes
/// out of the transfer matrix at the ray's own angle, which is why this part
/// has an angular acceptance and the ideal polarizer needs a datasheet to have
/// one.
#[test]
fn a_polarizing_cube_sorts_the_light_by_polarization() {
    let (through, up) = pbs_bench(12);

    // Both beams get out, and between them they carry nearly everything: the
    // cube's faces are AR coated and the stack absorbs nothing.
    assert!(
        through.s0 > 0.45 && up.s0 > 0.45,
        "a fair split: {} and {}",
        through.s0,
        up.s0
    );
    assert!(
        through.s0 + up.s0 > 0.95,
        "and almost all of it arrives: {}",
        through.s0 + up.s0
    );

    // The transmitted beam is the cleaner one, which is what a real cube's
    // datasheet says too: p passes almost perfectly, and what keeps s out of
    // it is every layer in the stack. The reflection is behind it because the
    // stack's outer interfaces are not at Brewster even when its inner ones
    // are, so a little p comes back with the s.
    assert!(
        through.dop() > 0.9999,
        "what goes through is pure: {}",
        through.dop()
    );
    assert!(up.dop() > 0.99, "the reflection nearly is: {}", up.dop());
    assert!(
        through.dop() > up.dop(),
        "transmission beats reflection: {} vs {}",
        through.dop(),
        up.dop()
    );

    // Both are linear. A dielectric stack away from total internal reflection
    // has no circular component to give.
    for st in [through, up] {
        assert!(st.s3.abs() / st.s0 < 1e-6, "linear: {}", st.s3 / st.s0);
    }
}

/// Thin the stack and the transmitted beam stops being pure, because what
/// keeps s out of it is the number of layers.
#[test]
fn a_polarizing_cube_needs_its_layers() {
    let thin = pbs_bench(1).0;
    let thick = pbs_bench(12).0;
    assert!(
        thin.dop() < thick.dop(),
        "one period is not a polarizer: {} against {}",
        thin.dop(),
        thick.dop()
    );
    assert!(thick.dop() - thin.dop() > 0.05, "and the gap is not noise");
}


/// The cube has an angular acceptance, and nobody typed it in.
///
/// This is the difference between a part whose behaviour is derived and one
/// whose behaviour is declared. An ideal polarizer holds its contrast at every
/// angle until a datasheet says otherwise; this one gives it up on its own,
/// because the Brewster match its layers rely on only holds at the angle it
/// was cut for.
///
/// The light launched here is already s, which the cube is supposed to reflect
/// entirely, so whatever reaches the detector past it is the extinction.
#[test]
fn a_polarizing_cube_has_an_acceptance_angle_it_was_never_told() {
    let leak = |deg: f64| -> f64 {
        let t = deg.to_radians();
        let dir = Vec3::new(t.cos(), 0.0, -t.sin()).normalize();
        let mut sc = Scene3::new();
        sc.add_polarizing_beam_splitter(Vec3::ZERO, 8.0, 135.0, 550.0, 12, 2.35, 1.46)
            .expect("a cube");
        let det = add_rect_detector(&mut sc, dir * 60.0, -dir, 20.0, 20.0, [4, 4]);
        sc.validate().expect("valid");
        // s is perpendicular to the plane of incidence, which for a diagonal
        // standing in the X-Y plane means s lies along Z.
        let launch = vec![launch_polarized(
            Ray3::new(dir * -60.0, dir).unwrap(),
            0.55,
            oc_core::polarization::Stokes::linear(1.0, 0.0),
            Vec3::Z,
        )];
        let res = trace3(
            &sc,
            &launch,
            &TraceConfig3 {
                ghosts: true,
                ..cfg()
            },
        );
        detector_named(&res, det).total_incident
    };

    assert!(leak(0.0) < 1e-9, "on axis nothing gets through: {}", leak(0.0));

    // And off axis the leak grows as the square of the angle, which is what a
    // stack coming off its design condition does.
    let (a, b, c, d) = (leak(1.0), leak(2.0), leak(5.0), leak(10.0));
    assert!(a > 1e-5, "one degree already leaks: {a:.3e}");
    assert!(d > 5e-3, "ten degrees is not a polarizer any more: {d:.3e}");
    for (lo, hi, at) in [(a, b, "1 to 2"), (c, d, "5 to 10")] {
        let ratio = hi / lo;
        assert!(
            (ratio - 4.0).abs() < 0.4,
            "doubling the angle quadruples the leak, {at} degrees: {ratio:.3}"
        );
    }
}
