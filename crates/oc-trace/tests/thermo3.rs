//! The first-order thermo-optic model (coating-fidelity spec CF-10, C29):
//! a quarter-wave stack moves with its optical thickness, `Off` is the
//! untempered path bit for bit, and a declared temperature reaches the trace
//! per object.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::fidelity::{FidelityProfile, Temperature};
use oc_core::migrate_v2::for_each_coating_mut;
use oc_core::scene::Spectrum;
use oc_core::thinfilm::{
    coating_amp_eval, coating_eval, Coating, CoatingBacking, FilmContext, FilmLayer, FilmMaterial, Incidence, ThermoOptic,
};
use oc_core::vec3::Vec3;
use oc_trace::{trace_scene3, TraceConfig3};

const T_REF: f64 = 20.0;

/// `(HL)^6 H` quarter waves at 550 nm, H 2.35 and L 1.46, each with a
/// thermo-optic model whose `dn/dT` is `rel` times its index, so both layers'
/// optical thicknesses move by the same factor and the stack stays a
/// quarter-wave stack at a new centre.
fn stack(rel: f64, cte: f64) -> Coating {
    let film = |n: f64| {
        let mut m = FilmMaterial::constant(if n > 2.0 { "H" } else { "L" }, n);
        m.thermal = Some(ThermoOptic { dn_dt_per_k: rel * n, cte_per_k: cte, t_ref_c: T_REF });
        FilmLayer::new(m, 550.0 / (4.0 * n))
    };
    let mut layers = Vec::new();
    for _ in 0..6 {
        layers.push(film(2.35));
        layers.push(film(1.46));
    }
    layers.push(film(2.35));
    Coating { layers, backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None }
}

fn at(t: Option<f64>) -> FilmContext {
    FilmContext { temperature_c: t, ..FilmContext::default() }
}

/// The centre of the stop band: where the reflected amplitude is real. A
/// quarter-wave stack's reflection phase passes through π there, so `Im r`
/// changes sign; bisected to the last bit.
fn centre(c: &Coating, ctx: &FilmContext) -> f64 {
    let im = |l: f64| coating_amp_eval(c, 1.0, 1.52, l, 1.0, ctx).unwrap().0.r_s.im;
    let (mut a, mut b) = (0.53, 0.57);
    assert!(im(a).signum() != im(b).signum(), "the bracket holds one centre");
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if m <= a || m >= b {
            break;
        }
        if im(m).signum() == im(a).signum() {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// C29: 50 K above its reference the centre moves by the optical thickness,
/// `λ₀ (1 + ΔT dn/n)(1 + α ΔT)`, to 1e-9 relative; the first-order form
/// `(dn/n + α) ΔT λ₀` is that less the product term.
#[test]
fn c29_a_quarter_wave_stack_moves_with_its_optical_thickness() {
    let (rel, cte) = (2.0e-5, 5.0e-6);
    let c = stack(rel, cte);
    let cold = centre(&c, &at(None));
    assert!((cold - 0.55).abs() < 1e-12, "the cold centre is the design: {cold}");
    let dt = 50.0;
    let warm = centre(&c, &at(Some(T_REF + dt)));
    let want = cold * (1.0 + rel * dt) * (1.0 + cte * dt);
    assert!(((warm - want) / want).abs() < 1e-9, "warm {warm} vs {want}");
    let first_order = cold * (rel + cte) * dt;
    let product = cold * rel * cte * dt * dt;
    assert!(((warm - cold) - first_order - product).abs() < 1e-12);
    // At the reference temperature nothing moves.
    assert_eq!(centre(&c, &at(Some(T_REF))), cold);
}

/// Under `Off` the context carries no temperature whatever the scene
/// declares, and a material's thermal model is never read: bit-identical to
/// the same stack with no thermal data at all.
#[test]
fn c29_off_is_the_untempered_path_bit_for_bit() {
    let off = FidelityProfile::default();
    assert_eq!(off.temperature, Temperature::Off);
    let ctx = FilmContext::from_profile(&off).at_temperature(&off, Some(90.0));
    assert_eq!(ctx.temperature_c, None);
    let with = stack(2.0e-5, 5.0e-6);
    let mut without = with.clone();
    for l in &mut without.layers {
        l.material.thermal = None;
    }
    for i in 0..60 {
        let lam = 0.4 + 0.005 * i as f64;
        for cos in [1.0, 0.9, 0.7] {
            let a = coating_eval(&with, 1.0, 1.52, lam, cos, &ctx).unwrap().0;
            let b = coating_eval(&without, 1.0, 1.52, lam, cos, &FilmContext::default()).unwrap().0;
            assert_eq!(a, b, "{lam} {cos}");
        }
    }
}

/// A film plate whose layers carry thermal data, in the default scene.
fn warm_scene() -> (Scene3, u64) {
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.54, 1.0), (0.56, 1.0), (0.58, 1.0)]);
    let id = s.add_film_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 135.0, 560.0, false).expect("a dichroic");
    for_each_coating_mut(&mut s, |c| {
        for l in &mut c.layers {
            let n = l.material.nominal_index().re;
            l.material.thermal = Some(ThermoOptic { dn_dt_per_k: 1.0e-4 * n, cte_per_k: 1.0e-5, t_ref_c: T_REF });
        }
    });
    (s, id)
}

fn through(s: &Scene3, cfg: &TraceConfig3) -> f64 {
    let det = s.detectors[0].common.id;
    detector_named(&trace_scene3(s, cfg), det).total_incident
}

/// In the trace: a declared temperature under `Off` changes no bit; under
/// `Declared` it moves the dichroic's edge and the stamp says "declared, not
/// validated"; an object held at its reference temperature is the cold
/// trace again.
#[test]
fn a_declared_temperature_reaches_the_trace_per_object() {
    let (cold, id) = warm_scene();
    let mut hot = cold.clone();
    hot.temperature_c = Some(T_REF + 150.0);
    let off = TraceConfig3::default();
    assert_eq!(through(&cold, &off).to_bits(), through(&hot, &off).to_bits());

    let declared = TraceConfig3 {
        fidelity: FidelityProfile { temperature: Temperature::Declared, ..FidelityProfile::default() },
        ..TraceConfig3::default()
    };
    let (a, b) = (through(&cold, &declared), through(&hot, &declared));
    assert!((a - b).abs() > 1e-3, "150 K moves the edge: {a} vs {b}");
    let res = trace_scene3(&hot, &declared);
    assert!(res.fidelity.text().contains("temperature declared, not validated"), "{}", res.fidelity.text());

    // The plate itself held at the reference: the cold trace.
    assert!(hot.set_object_temperature(id, Some(T_REF)));
    assert_eq!(through(&hot, &declared).to_bits(), a.to_bits());

    // Declared with no temperature anywhere says the films stay put.
    let res = trace_scene3(&cold, &declared);
    assert!(res
        .capability_report
        .approximated
        .iter()
        .any(|x| x.detail.contains("none is set")));
}

/// A film with no thermo-optic data is held at its reference and the result
/// says so; the temperatures round-trip through v3 and are dropped, with a
/// note, by the v2 export.
#[test]
fn films_without_data_are_named_and_v2_drops_temperatures() {
    let mut s = Scene3::default_scene();
    s.add_film_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 135.0, 560.0, false).unwrap();
    s.temperature_c = Some(60.0);
    let declared = TraceConfig3 {
        fidelity: FidelityProfile { temperature: Temperature::Declared, ..FidelityProfile::default() },
        ..TraceConfig3::default()
    };
    let res = trace_scene3(&s, &declared);
    assert!(res
        .capability_report
        .approximated
        .iter()
        .any(|x| x.detail.contains("no thermo-optic data")));
    let back = Scene3::from_json(&s.to_json()).unwrap();
    assert_eq!(back.temperature_c, Some(60.0));
    let (_, losses) = s.to_json_v2();
    assert!(losses.iter().any(|l| l.contains("declared temperatures are dropped")), "{losses:?}");
}
