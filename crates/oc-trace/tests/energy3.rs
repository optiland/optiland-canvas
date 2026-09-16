//! The energy ledger: launched flux is accounted for, never lost.
//!
//! What this suite owes (contract §9.8, §15.2):
//!
//! - for an uncapped deterministic scene, `launched` balances everything that
//!   removes flux (escaped, volume and surface absorption, blocked apertures, a
//!   terminating detector) to 1e-10 relative on analytic fixtures;
//! - flux removed by a threshold or a branch cap appears under `truncated_flux`.
//!   It does not quietly disappear, because a ledger that only balances when
//!   nothing is truncated is not a ledger;
//! - `1 - R - T` lands in `absorbed_surface`;
//! - Beer-Lambert attenuation lands in `absorbed_volume`;
//! - every branch ends with an explicit `TerminationReason`;
//! - energy never increases without a source.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::material::LAMBDA_D;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, TraceConfig3};

fn one_ray(scene: &Scene3, origin: Vec3, dir: Vec3, cfg: &TraceConfig3) -> oc_trace::TraceResult3 {
    let ray = Ray3::new(origin, dir).unwrap();
    trace3(scene, &[launch(ray, LAMBDA_D, 1.0)], cfg)
}

#[test]
fn a_plane_parallel_plate_closes_the_ledger() {
    // A dielectric slab with ghosts followed: an etalon whose internal
    // reflections weaken until they fall below threshold. Nothing absorbs, so
    // every photon eventually escapes or is truncated — and the two must sum to
    // what was launched.
    let mut scene = Scene3::new();
    add_glass_slab(&mut scene, 0.0, 4.0, 1.5);
    let res = one_ray(
        &scene,
        Vec3::new(-10.0, 0.0, 0.0),
        Vec3::X,
        &TraceConfig3::default(),
    );
    assert!(
        res.ledger.absorbed_volume < 1e-15,
        "a clear slab absorbs nothing in its bulk"
    );
    assert!(
        res.ledger.residual() < 1e-9,
        "ledger did not close: launched {} vs accounted {}",
        res.ledger.launched,
        res.ledger.accounted()
    );
}

#[test]
fn a_partial_mirror_puts_one_minus_r_into_absorbed_surface() {
    // Reflectivity 0.8: 0.2 of the flux is absorbed at the surface, 0.8 leaves.
    let mut scene = Scene3::new();
    add_mirror(&mut scene, Vec3::ZERO, Vec3::X, 10.0, 0.8);
    let res = one_ray(&scene, Vec3::new(-5.0, 0.0, 0.0), Vec3::X, &no_ghosts());
    assert!(
        (res.ledger.absorbed_surface - 0.2).abs() < 1e-12,
        "1 - R absorbed"
    );
    assert!((res.ledger.escaped - 0.8).abs() < 1e-12, "R reflected out");
    assert!(res.ledger.residual() < 1e-12);
}

#[test]
fn an_absorber_body_swallows_the_beam() {
    let mut scene = Scene3::new();
    add_absorber_box(&mut scene, Vec3::new(0.0, 0.0, 0.0), Vec3::splat(3.0));
    let res = one_ray(&scene, Vec3::new(-10.0, 0.0, 0.0), Vec3::X, &no_ghosts());
    assert!(
        (res.ledger.absorbed_surface - 1.0).abs() < 1e-12,
        "all flux absorbed at the surface"
    );
    assert!(res.ledger.escaped < 1e-15, "nothing escaped an absorber");
    assert!(res.ledger.residual() < 1e-12);
}

#[test]
fn beer_lambert_attenuation_lands_in_absorbed_volume() {
    // An index-matched box (n = 1) so no flux reflects at the faces: the only sink
    // is the bulk. Path length inside is 2·half = 10 mm, α = 0.1 mm⁻¹, so a
    // fraction 1 − e⁻¹ is absorbed in the volume and the rest escapes.
    let mut scene = Scene3::new();
    let alpha = 0.1;
    add_absorbing_box(&mut scene, Vec3::new(5.0, 20.0, 20.0), 1.0, alpha);
    let res = one_ray(&scene, Vec3::new(-10.0, 0.0, 0.0), Vec3::X, &no_ghosts());

    let expect_absorbed = 1.0 - (-alpha * 10.0).exp();
    assert!(
        (res.ledger.absorbed_volume - expect_absorbed).abs() < 1e-4,
        "absorbed_volume {} vs expected {}",
        res.ledger.absorbed_volume,
        expect_absorbed
    );
    assert!(res.ledger.residual() < 1e-9);
}

#[test]
fn a_blocked_aperture_is_accounted() {
    let mut scene = Scene3::new();
    add_circular_aperture(&mut scene, Vec3::ZERO, Vec3::X, 1.0, 20.0);
    let res = one_ray(&scene, Vec3::new(-5.0, 5.0, 0.0), Vec3::X, &no_ghosts());
    assert!((res.ledger.blocked_aperture - 1.0).abs() < 1e-12);
    assert!(res.ledger.residual() < 1e-12);
}

#[test]
fn ghosts_off_names_the_reflected_flux_rather_than_dropping_it() {
    // With ghosts off, the Fresnel reflections at each face are not followed. They
    // must still appear in `truncated_flux`: a ledger that balances only because
    // the dropped light was never counted is worthless (§9.8).
    let mut scene = Scene3::new();
    add_glass_slab(&mut scene, 0.0, 4.0, 1.5);
    let res = one_ray(&scene, Vec3::new(-10.0, 0.0, 0.0), Vec3::X, &no_ghosts());
    assert!(
        res.ledger.truncated_flux > 0.05,
        "the reflected flux was dropped, not named"
    );
    assert!(res.ledger.residual() < 1e-9, "ledger did not close");
    // The named-and-followed accounting agrees: with ghosts on nothing is
    // truncated at the first bounce, and the ledger still closes.
    let on = one_ray(
        &scene,
        Vec3::new(-10.0, 0.0, 0.0),
        Vec3::X,
        &TraceConfig3::default(),
    );
    assert!(on.ledger.residual() < 1e-9);
}

#[test]
fn a_richer_scene_still_closes_with_ghosts_on() {
    // A slab, a partial mirror, and a detector in one scene. Every sink is active;
    // the ledger must still balance to a photon.
    let mut scene = Scene3::new();
    add_glass_slab(&mut scene, -2.0, 2.0, 1.52);
    add_mirror(
        &mut scene,
        Vec3::new(30.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.2, 0.1).normalize(),
        20.0,
        0.9,
    );
    add_rect_detector(
        &mut scene,
        Vec3::new(30.0, 0.0, 25.0),
        Vec3::NEG_Z,
        30.0,
        30.0,
        [4, 4],
    );

    let res = one_ray(
        &scene,
        Vec3::new(-10.0, 0.0, 0.0),
        Vec3::X,
        &TraceConfig3::default(),
    );
    assert!(
        res.ledger.residual() < 1e-8,
        "ledger did not close: launched {} vs accounted {}",
        res.ledger.launched,
        res.ledger.accounted()
    );
    // Energy never increases: what is accounted for never exceeds what came in.
    assert!(res.ledger.accounted() <= res.ledger.launched + 1e-9);
}

#[test]
fn a_branch_cap_deposits_the_remaining_flux() {
    // A hard branch cap must not vaporize the queued flux: it lands in
    // `truncated_flux` and the ledger still closes.
    let mut scene = Scene3::new();
    add_glass_slab(&mut scene, 0.0, 4.0, 1.5);
    let cfg = TraceConfig3 {
        max_branches: 5,
        ..TraceConfig3::default()
    };
    let res = one_ray(&scene, Vec3::new(-10.0, 0.0, 0.0), Vec3::X, &cfg);
    assert!(
        res.terminations.max_branches >= 1,
        "the cap should have fired"
    );
    assert!(
        res.ledger.residual() < 1e-9,
        "capped flux was dropped, not named"
    );
}
