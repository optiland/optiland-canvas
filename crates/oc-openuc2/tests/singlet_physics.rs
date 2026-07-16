//! Does the generated singlet actually obey the lensmaker's equation?
//!
//! This is the cross-check that makes the radius editable in good conscience. The
//! generator reports a paraxial focal length from
//! `1/f = (n-1)[c1 - c2 + (n-1) t c1 c2 / n]`; the tracer refracts rays through
//! the geometry it built and knows nothing about that formula. They are two
//! independent answers to one question, and the only reason either can be
//! trusted is that they agree.
//!
//! Paraxial theory is a small-angle limit, so the rays here are launched close to
//! the axis. That is not a way of hiding error — it is what the prediction claims
//! to describe. A ray at the rim of a 24 mm singlet misses the paraxial focus by
//! millimetres of honest spherical aberration, and
//! [`spherical_aberration_grows_with_aperture`] asserts that it does.

use oc_openuc2::generator::{Generator, SingletParams, DESIGN_LAMBDA_UM};
use oc_core::{Isometry2, Scene, Source, SourceKind, Spectrum, Vec2};
use oc_trace::{trace, TraceConfig};

/// A lens at the origin, lit by a collimated beam of half-width `h` along +x.
///
/// Monochromatic at the design wavelength: the prediction uses one index, so
/// lighting it with three lines would compare a number against a smear.
fn bench(p: &SingletParams, h: f64, rays: u32) -> Scene {
    let fragment = Generator::Singlet
        .build(&params_of(p))
        .expect("the generator builds it");
    let mut scene = Scene::new();
    scene
        .insert_fragment(&fragment, Isometry2::IDENTITY)
        .expect("it inserts");
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: h },
        transform: Isometry2::from_pos(Vec2::new(-40.0, 0.0)),
        spectrum: Spectrum::Lines(vec![(DESIGN_LAMBDA_UM, 1.0)]),
        flux: 1.0,
        rays,
    });
    scene
}

fn params_of(p: &SingletParams) -> std::collections::BTreeMap<String, serde_json::Value> {
    let radius = |r: f64| -> serde_json::Value {
        // JSON has no infinity; a plane travels as the string (§9.5).
        if r.is_infinite() {
            serde_json::json!(if r > 0.0 { "inf" } else { "-inf" })
        } else {
            serde_json::json!(r)
        }
    };
    let mut m = std::collections::BTreeMap::new();
    m.insert("material".into(), serde_json::json!(p.material));
    m.insert("diameter_mm".into(), serde_json::json!(p.diameter_mm));
    m.insert(
        "center_thickness_mm".into(),
        serde_json::json!(p.center_thickness_mm),
    );
    m.insert("radius_1_mm".into(), radius(p.radius_1_mm));
    m.insert("radius_2_mm".into(), radius(p.radius_2_mm));
    m
}

fn singlet(r1: f64, r2: f64, diameter: f64) -> SingletParams {
    SingletParams {
        material: "N-BK7".to_string(),
        diameter_mm: diameter,
        center_thickness_mm: 5.0,
        radius_1_mm: r1,
        radius_2_mm: r2,
    }
}

/// Where the outermost emerging ray crosses the axis, measured in the lens frame.
///
/// The last segment of the ray that started furthest off-axis: after the lens it
/// is heading for the focus, so extending it to `y = 0` is the crossing. Read off
/// the trace rather than off a detector, because a detector would quantize it.
fn axis_crossing(scene: &Scene) -> f64 {
    let cfg = TraceConfig {
        bounds: 2000.0,
        // No ghosts: a faint internal reflection also converges somewhere, and it
        // is not the focus anyone means.
        ghosts: false,
        ..TraceConfig::default()
    };
    let out = trace(scene, &cfg);

    // A ray's path here is: source -> front surface (parallel, d.y = 0), through
    // the glass (converging), then out to the bounds (converging harder). The one
    // to extend is the last, so among the converging segments take the one
    // furthest downstream — the lens is only millimetres thick, so picking by a
    // fixed x would be picking by luck.
    let mut best: Option<(f64, f64)> = None; // (start x, crossing x)
    for s in &out.segments {
        let d = s.b - s.a;
        if d.x <= 0.0 || s.a.y == 0.0 || d.y.abs() < 1e-12 {
            continue;
        }
        // Converging: heading back toward the axis it started off.
        if (s.a.y > 0.0) == (d.y > 0.0) {
            continue;
        }
        let x = s.a.x - s.a.y * d.x / d.y;
        if best.map_or(true, |(bx, _)| s.a.x > bx) {
            best = Some((s.a.x, x));
        }
    }
    best.expect("a converging ray leaving the lens").1
}

#[test]
fn a_biconvex_singlet_focuses_where_the_lensmakers_equation_says() {
    // The headline check: the geometry the generator built bends light to the
    // focus the generator predicts, and neither computation knows about the other.
    let p = singlet(100.0, -100.0, 24.0);
    let predicted = p.paraxial().unwrap();
    // 1 mm off axis: paraxial is a small-angle claim, so test it in that regime.
    let traced = axis_crossing(&bench(&p, 1.0, 3));

    let err = (traced - predicted.rear_focus_x_mm).abs();
    assert!(
        err < 0.05,
        "traced focus {traced:.4} vs predicted {:.4} (efl {:.3}, bfd {:.3}, n {:.5})",
        predicted.rear_focus_x_mm,
        predicted.efl_mm,
        predicted.bfd_mm,
        predicted.nd
    );
}

#[test]
fn a_plano_convex_singlet_focuses_where_predicted_too() {
    // A different shape, so the agreement is not an artefact of a symmetric lens
    // where the thickness terms happen to cancel.
    let p = singlet(51.68, f64::INFINITY, 24.0);
    let predicted = p.paraxial().unwrap();
    let traced = axis_crossing(&bench(&p, 1.0, 3));
    assert!(
        (traced - predicted.rear_focus_x_mm).abs() < 0.05,
        "traced {traced:.4} vs predicted {:.4}",
        predicted.rear_focus_x_mm
    );
}

#[test]
fn halving_the_radius_roughly_halves_the_focal_length() {
    // The property behind the editable field: a stronger curve is a stronger
    // lens. For a thin biconvex, 1/f scales as 2(n-1)/R, so halving R about
    // halves f — "about", because the thickness term is not linear.
    let long = singlet(200.0, -200.0, 24.0).paraxial().unwrap().efl_mm;
    let short = singlet(100.0, -100.0, 24.0).paraxial().unwrap().efl_mm;
    let ratio = long / short;
    assert!((ratio - 2.0).abs() < 0.1, "f({long:.2}) / f({short:.2}) = {ratio:.3}");
}

#[test]
fn the_traced_focus_moves_when_the_radius_does() {
    // And the tracer agrees with the prediction *at both radii*, which is the
    // whole point of being allowed to type a new one.
    for r in [80.0, 120.0, 200.0] {
        let p = singlet(r, -r, 24.0);
        let predicted = p.paraxial().unwrap();
        let traced = axis_crossing(&bench(&p, 1.0, 3));
        assert!(
            (traced - predicted.rear_focus_x_mm).abs() < 0.05,
            "R = {r}: traced {traced:.4} vs predicted {:.4}",
            predicted.rear_focus_x_mm
        );
    }
}

#[test]
fn spherical_aberration_grows_with_aperture() {
    // The paraxial number is a small-angle claim, and this is what that costs. A
    // marginal ray at the rim crosses the axis *short* of the paraxial focus, by
    // more than the near-axis ray does. If this ever failed, either the trace had
    // stopped being a real spherical surface or the prediction had grown a fudge
    // factor to make the headline test pass.
    let p = singlet(100.0, -100.0, 24.0);
    let f = p.paraxial().unwrap().rear_focus_x_mm;

    let near = (axis_crossing(&bench(&p, 1.0, 3)) - f).abs();
    let rim = (axis_crossing(&bench(&p, 11.0, 3)) - f).abs();

    assert!(rim > near, "rim error {rim:.4} should exceed paraxial {near:.4}");
    assert!(
        axis_crossing(&bench(&p, 11.0, 3)) < f,
        "a spherical singlet's marginal focus is short of paraxial, not long"
    );
}

#[test]
fn the_generated_lens_is_the_lens_the_fixture_always_had() {
    // The fixture's singlet used to be a hand-authored `.ocanvas`; it is generated
    // now, from the same numbers. This is what makes that switch a fact rather
    // than a hope: the two agree edge for edge, so the demo, its committed
    // `.uc2canvas`, and every number in the C2–C5 tests are untouched by it.
    use oc_core::{Scene, SceneFragment};
    use oc_openuc2::test_support::lens_fragment;

    let authored = SceneFragment::from_scene(&Scene::from_json(&lens_fragment()).unwrap()).unwrap();
    let generated = Generator::Singlet
        .build(&params_of(&singlet(100.0, -100.0, 24.0)))
        .unwrap();

    assert_eq!(generated.elements, authored.elements);
}

#[test]
fn the_prediction_uses_the_same_dispersion_the_tracer_refracts_with() {
    // Both read the Sellmeier coefficients; `nd` is only a cache (§10.3). If the
    // prediction ever read the cache instead, this would drift the moment a glass
    // entry's cache disagreed with its coefficients.
    let scene = Scene::new();
    let id = scene.catalog.find("N-BK7").unwrap();
    let glass = scene.catalog.get(id).unwrap();
    let p = singlet(100.0, -100.0, 24.0).paraxial().unwrap();
    assert_eq!(p.nd, glass.sell.index(DESIGN_LAMBDA_UM));
    assert_eq!(p.lambda_um, DESIGN_LAMBDA_UM);
}
