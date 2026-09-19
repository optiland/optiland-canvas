//! The kernel against published reference data.
//!
//! A vendor's curve is only a test where the part is **determined by numbers
//! anybody can look up**. Where the design is proprietary, reproducing the
//! curve would be a test of guessing the design, so nothing here tries; see
//! `docs/validation.md` for where that line falls and what to do on the other
//! side of it.
//!
//! What is here is the material data our synthesized parts are built out of.
//! Every constant is quoted with where it came from, and no data files are
//! vendored: these are a handful of coefficients from a published paper,
//! reproduced for verification.
#![cfg(feature = "scene3")]

use oc_core::polarization::{a_plate_retardance, a_plate_thickness};
use oc_core::scene3::Waveplate;
use std::f64::consts::FRAC_PI_2;

// ---------------------------------------------------------------------------
// Crystalline quartz
// ---------------------------------------------------------------------------
//
// Sellmeier coefficients for α-quartz, ordinary and extraordinary, from
//
//   G. Ghosh, "Dispersion-equation coefficients for the refractive index and
//   birefringence of calcite and quartz crystals", Opt. Commun. 163, 95-102
//   (1999). doi:10.1016/S0030-4018(99)00091-7
//
// as tabulated by refractiveindex.info (retrieved 2026-09-19), valid over
// 0.198-2.053 µm. The form is
//
//   n² − 1 = A + B₁λ²/(λ² − C₁) + B₂λ²/(λ² − C₂)
//
// with λ in µm. Six numbers each, reproduced here rather than fetched, so the
// suite runs offline and the oracle cannot rot.

/// `(A, B₁, C₁, B₂, C₂)` for the ordinary ray.
const QUARTZ_O: (f64, f64, f64, f64, f64) =
    (0.28604141, 1.07044083, 1.00585997e-2, 1.10202242, 100.0);
/// The same for the extraordinary ray.
const QUARTZ_E: (f64, f64, f64, f64, f64) =
    (0.28851804, 1.09509924, 1.02101864e-2, 1.15662475, 100.0);

fn sellmeier((a, b1, c1, b2, c2): (f64, f64, f64, f64, f64), lambda_um: f64) -> f64 {
    let l2 = lambda_um * lambda_um;
    (1.0 + a + b1 * l2 / (l2 - c1) + b2 * l2 / (l2 - c2)).sqrt()
}

/// The sodium D line, which is where the two constants the kernel carries are
/// quoted.
const SODIUM_D_UM: f64 = 0.5893;

/// The constants [`Waveplate::quartz`] is built on are the ones anybody would
/// look up, and they agree with the dispersion equation that produced them.
///
/// What matters for a waveplate is not either index but their difference,
/// because that alone sets the thickness a retardance needs. So the
/// birefringence is checked ten times tighter than the indices.
#[test]
fn the_quartz_constants_agree_with_the_published_dispersion() {
    let plate = Waveplate::quartz(0, 0.55);
    let (o, e) = (
        sellmeier(QUARTZ_O, SODIUM_D_UM),
        sellmeier(QUARTZ_E, SODIUM_D_UM),
    );
    assert!(
        (plate.n_o - o).abs() < 1e-4,
        "n_o: kernel {} against Ghosh {o:.6}",
        plate.n_o
    );
    assert!(
        (plate.n_e - e).abs() < 1e-4,
        "n_e: kernel {} against Ghosh {e:.6}",
        plate.n_e
    );
    let (ours, theirs) = (plate.n_e - plate.n_o, e - o);
    assert!(
        (ours - theirs).abs() / theirs < 3e-3,
        "birefringence: kernel {ours:.6} against Ghosh {theirs:.6}"
    );
}

/// How far a constant-index waveplate drifts from a real one across the
/// visible, in degrees, and it is not a guess.
///
/// The kernel holds `n_o` and `n_e` fixed, so a plate's retardance goes
/// exactly as `λ₀/λ`. A real plate's birefringence rises toward the blue as
/// well, so its retardance rises faster than that. The whole of the
/// discrepancy is the ratio of the two birefringences, and this pins it:
///
/// | λ | kernel | quartz | out by |
/// |---|---|---|---|
/// | 450 nm | 110.0° | 112.7° | −2.7° |
/// | 500 nm | 99.0° | 100.0° | −1.0° |
/// | 550 nm | 90.0° | 90.0° | 0 |
/// | 650 nm | 76.2° | 75.0° | +1.1° |
/// | 700 nm | 70.7° | 69.3° | +1.4° |
///
/// A couple of degrees on a quarter wave, which is a percent or two of the
/// state and is why the docs call the missing piece the dispersion of the
/// birefringence rather than something that does not matter.
#[test]
fn a_constant_index_quarter_wave_drifts_from_quartz_by_degrees() {
    let design = 0.55;
    let plate = Waveplate::quartz(0, design);
    let thickness = plate.thickness_um(90.0);
    let dn0 = sellmeier(QUARTZ_E, design) - sellmeier(QUARTZ_O, design);

    let mut worst: f64 = 0.0;
    for (nm, want_err) in [
        (450.0, -2.679),
        (500.0, -1.031),
        (550.0, 0.0),
        (650.0, 1.116),
        (700.0, 1.422),
    ] {
        let lambda = nm / 1000.0;
        let ours = a_plate_retardance(plate.n_o, plate.n_e, thickness, lambda, 0.0, 0.0);
        // The same plate made of real quartz: the thickness is whatever the
        // design wavelength fixed, and the birefringence is the one the
        // dispersion equation gives at the wavelength being traced.
        let dn = sellmeier(QUARTZ_E, lambda) - sellmeier(QUARTZ_O, lambda);
        let real = ours * dn / dn0;
        let err = (ours - real).to_degrees();
        assert!(
            (err - want_err).abs() < 0.01,
            "at {nm} nm we are out by {err:.3} degrees, expected {want_err}"
        );
        worst = worst.max(err.abs());
    }
    assert!(
        worst > 2.5 && worst < 3.0,
        "worst case across the visible: {worst:.3} degrees"
    );

    // And the same fraction on a multi-order plate is not a couple of degrees
    // any more. The error is a share of the whole retardance, so a tenth-order
    // quarter wave at 450 nm is most of a half wave out, and a constant index
    // is no use to it at all.
    let multi = Waveplate::quartz(10, design);
    let thick = multi.thickness_um(90.0);
    let lambda = 0.45;
    let ours = a_plate_retardance(multi.n_o, multi.n_e, thick, lambda, 0.0, 0.0);
    let dn = sellmeier(QUARTZ_E, lambda) - sellmeier(QUARTZ_O, lambda);
    let err = (ours * (1.0 - dn / dn0)).to_degrees().abs();
    assert!(
        err > 100.0,
        "a tenth-order plate at 450 nm is out by {err:.1} degrees"
    );
}

/// The thickness the kernel picks for a quarter wave is the thickness a real
/// one is ground to.
///
/// 15.1 µm of quartz for a true zero-order quarter wave at 550 nm, which is
/// why nobody sells one unmounted: it is a sixth the thickness of a human
/// hair, and a real "zero-order" plate is usually two thick plates subtracted
/// or one cemented to a window.
#[test]
fn a_zero_order_quarter_wave_is_as_thin_as_it_ought_to_be() {
    let d = a_plate_thickness(
        sellmeier(QUARTZ_O, 0.55),
        sellmeier(QUARTZ_E, 0.55),
        0.55,
        FRAC_PI_2,
        0,
    );
    assert!((d - 15.01).abs() < 0.1, "{d:.3} µm");
    // Ten orders up it is milled to something you can hold, and that is the
    // trade: 41 times the thickness, 41 times the sensitivity.
    let multi = a_plate_thickness(
        sellmeier(QUARTZ_O, 0.55),
        sellmeier(QUARTZ_E, 0.55),
        0.55,
        FRAC_PI_2,
        10,
    );
    assert!((multi / d - 41.0).abs() < 1e-9, "{}", multi / d);
}

// ---------------------------------------------------------------------------
// The law every filter vendor publishes
// ---------------------------------------------------------------------------

/// A thin-film edge blue-shifts with angle by
///
/// ```text
///   λ(θ) = λ₀ · √(1 − (sin θ / n_eff)²)
/// ```
///
/// which is what a filter datasheet tells you to use and what anybody
/// mounting a dichroic at 45 degrees relies on. `n_eff` is a single number
/// standing in for the whole stack, and vendors quote it in the range 1.5 to
/// 2.5.
///
/// Nothing in the kernel knows that formula. The edge comes out of the
/// transfer matrix at each angle, measured where the reflectance crosses a
/// half. So the test is not whether we can evaluate the formula, it is whether
/// **one** effective index fits every angle: fit `n_eff` once at 45 degrees,
/// then predict 10 through 60 and see how far off it lands.
///
/// It lands within 0.06 nm out to 50 degrees and 0.6 nm at 60. A stack that
/// got the angle physics wrong would need a different `n_eff` at every angle,
/// and the one-parameter fit would fall apart.
#[test]
fn one_effective_index_fits_the_edge_at_every_angle() {
    use oc_core::thinfilm::{crossing_nm, film_coating, BandSide, CoatingDesign, DesignKind};

    // Silica and tantala, the ordinary pair for a visible edge filter, on
    // crown.
    let (n_low, n_high) = (1.46f64, 2.05f64);
    let design = CoatingDesign {
        kind: DesignKind::LongPass,
        lambda_nm: 550.0,
        aoi_deg: 0.0,
        periods: 8,
        n_high,
        n_low,
    };
    let c = film_coating(design, 1.0, 1.5168);
    let edge = |aoi: f64| {
        crossing_nm(&c, 1.0, 1.5168, aoi, 0.5, 300.0, 900.0, BandSide::Long)
            .expect("the band has a long edge")
    };

    let at_normal = edge(0.0);
    assert!(
        (at_normal - 550.0).abs() < 1.0,
        "the synthesis put the edge where it was asked: {at_normal:.1} nm"
    );

    // One effective index, fitted from one angle.
    let ratio = edge(45.0) / at_normal;
    let n_eff = (0.5f64 / (1.0 - ratio * ratio)).sqrt();
    assert!(
        (1.5..=2.5).contains(&n_eff),
        "in the range vendors quote: {n_eff:.4}"
    );
    assert!(
        n_eff > n_low && n_eff < n_high,
        "and between the two materials: {n_eff:.4}"
    );

    for (aoi, tol) in [
        (10.0f64, 0.2),
        (20.0, 0.2),
        (30.0, 0.2),
        (40.0, 0.2),
        (50.0, 0.2),
        (60.0, 1.0),
    ] {
        let want = at_normal * (1.0 - (aoi.to_radians().sin() / n_eff).powi(2)).sqrt();
        let got = edge(aoi);
        assert!(
            (got - want).abs() < tol,
            "at {aoi} degrees: matrix {got:.2} nm against the formula's {want:.2} nm"
        );
    }

    // The rule of thumb that `n_eff` is the geometric mean of the two indices
    // is a rule for a cavity, not a law for an edge. It reads 1.730 here and
    // the stack's own answer is 1.680, three percent under, which is a 1.5 nm
    // error in the 45 degree edge. Worth knowing before trusting it.
    let rule = (n_low * n_high).sqrt();
    assert!(
        rule > n_eff && (rule - n_eff) / n_eff < 0.05,
        "the geometric mean {rule:.4} overshoots the measured {n_eff:.4}"
    );
}
