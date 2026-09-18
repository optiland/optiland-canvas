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
