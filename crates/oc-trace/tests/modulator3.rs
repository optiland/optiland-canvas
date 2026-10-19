//! DMDs and LCoS SLMs in the f64 reference, in the geometric limit
//! (acceptance case C52): a micromirror reflects off its own facet, the dead
//! space absorbs, a phase pattern steers by its gradient, and the ledger
//! balances. Their diffraction is the wave engine's; the report says so.
//!
//! The DMD geometry follows Texas Instruments DLPA083B (2019, rev. 2023)
//! §4.2; the phase-gradient law is the grating equation in direction cosines
//! (J. E. Harvey, C. L. Vernold, *Appl. Opt.* 37(34), 8158 (1998),
//! doi:10.1364/AO.37.008158); the director split is Zhang, You & Chu,
//! *Light Sci. Appl.* 3, e213 (2014), doi:10.1038/lsa.2014.94.

#![cfg(feature = "trace3")]

use oc_core::id::SourceId;
use oc_core::modulator::{
    DmdSpec, LcosSpec, ModulatorKind, Pattern, PatternGenerator, PhaseLut, SpatialModulator, TiltAxis,
};
use oc_core::scene3::{Scene3, SurfaceResponse};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, LaunchRay3, PhysicsFeature, TraceConfig3};

fn launch(origin: Vec3, dir: Vec3, lambda_um: f64) -> LaunchRay3 {
    LaunchRay3 {
        ray: Ray3::new(origin, dir).unwrap(),
        lambda_um,
        flux: 1.0,
        source: Some(SourceId(1)),
        pol: None,
    }
}

fn dmd(pixels: u32, fill: f64, refl: f64, window: f64, value: u16) -> SpatialModulator {
    SpatialModulator {
        pixels: [pixels, pixels],
        pitch_um: 10.0,
        kind: ModulatorKind::Dmd(DmdSpec {
            tilt_deg: 12.0,
            tilt_axis: TiltAxis::Side,
            mirror_fill: fill,
            reflectance: refl,
            window_transmittance: window,
        }),
        pattern: Pattern::Uniform { value },
    }
}

/// A side-tilt DMD facing +X with its hinge along world Z: an ON mirror's
/// normal leans 12° toward +Y, so a ray arriving along −X leaves at 24° in
/// the XY plane, carrying `R·T_w²` of the flux; the rest is absorbed.
#[test]
fn an_on_mirror_reflects_a_ray_through_twice_its_tilt() {
    let mut s = Scene3::new();
    s.add_modulator(Vec3::ZERO, 0.0, dmd(15, 0.95, 0.9, 0.98, 1)).unwrap();
    let out = Vec3::new(24f64.to_radians().cos(), 24f64.to_radians().sin(), 0.0);
    let det = s.add_area_detector(out * 100.0, -out, 5.0, 5.0, [1, 1]);
    s.validate().unwrap();
    let cfg = TraceConfig3 { record_detector_hits: true, ..TraceConfig3::default() };
    let res = trace3(&s, &[launch(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 0.55)], &cfg);
    let d = res.detectors.iter().find(|d| d.detector.0 == det.0).unwrap();
    let want = 0.9 * 0.98 * 0.98;
    assert!((d.total_incident - want).abs() < 1e-12, "{}", d.total_incident);
    assert!(d.records[0].uv_mm.length() < 1e-9, "{:?}", d.records[0].uv_mm);
    assert!((res.ledger.absorbed_surface - (1.0 - want)).abs() < 1e-12);
    assert!(res.capability_report.exact.contains(&PhysicsFeature::Micromirror));
    assert!(res.capability_report.approximated.iter().any(|a| a.feature == PhysicsFeature::Micromirror));
    // OFF mirrors send it 24° the other way.
    let mut s = Scene3::new();
    s.add_modulator(Vec3::ZERO, 0.0, dmd(15, 0.95, 1.0, 1.0, 0)).unwrap();
    let off = Vec3::new(out.x, -out.y, 0.0);
    let det = s.add_area_detector(off * 100.0, -off, 5.0, 5.0, [1, 1]);
    let res = trace3(&s, &[launch(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 0.55)], &TraceConfig3::default());
    let d = res.detectors.iter().find(|d| d.detector.0 == det.0).unwrap();
    assert!((d.total_incident - 1.0).abs() < 1e-12);
}

/// Mirrors of side ρp tilted θ about one side cover ρ²cos θ of the array
/// plane. A stratified grid of parallel rays over whole pixels finds that
/// share reflected and the rest absorbed in the gaps, the ledger balanced.
#[test]
fn a_dmd_in_the_ray_limit_absorbs_its_gaps() {
    let (rho, theta) = (0.9f64, 12f64.to_radians());
    let mut s = Scene3::new();
    s.add_modulator(Vec3::ZERO, 0.0, dmd(4, rho, 1.0, 1.0, 1)).unwrap();
    s.validate().unwrap();
    // 4 × 4 pixels of 10 µm: rays over the array, 200 a pixel each way.
    let n = 800;
    let p = 0.010;
    let rays: Vec<LaunchRay3> = (0..n * n)
        .map(|k| {
            let (i, j) = (k % n, k / n);
            let y = -2.0 * p + (i as f64 + 0.5) * 4.0 * p / n as f64;
            let z = -2.0 * p + (j as f64 + 0.5) * 4.0 * p / n as f64;
            let mut l = launch(Vec3::new(10.0, y, z), -Vec3::X, 0.55);
            l.flux = 1.0 / (n * n) as f64;
            l
        })
        .collect();
    let res = trace3(&s, &rays, &TraceConfig3::default());
    let reflected = 1.0 - res.ledger.absorbed_surface;
    let want = rho * rho * theta.cos();
    eprintln!("measured: DMD reflected share {reflected:.6} vs ρ²cos θ = {want:.6}");
    assert!((reflected - want).abs() < 2e-3, "{reflected} vs {want}");
    let accounted = res.ledger.escaped + res.ledger.absorbed_surface + res.ledger.numerical_loss;
    assert!((accounted - 1.0).abs() < 1e-10, "{:?}", res.ledger);
}

fn slm(pattern: Pattern) -> SpatialModulator {
    SpatialModulator {
        pixels: [64, 64],
        pitch_um: 8.0,
        kind: ModulatorKind::Lcos(LcosSpec {
            director_deg: 0.0,
            lut: PhaseLut::Linear { levels: 256, max_phase_rad: 2.0 * std::f64::consts::PI },
            design_lambda_um: 0.633,
            pixel_fill: 1.0,
            gap_amplitude: 0.0,
            gap_phase_rad: 0.0,
            fringing_um: 0.0,
            reflectance: 1.0,
        }),
        pattern,
    }
}

/// A blazed grating of period 16 pixels steers the director component by
/// the grating equation, `Δt = λ/Λ` (the phase-gradient law), adding the
/// pixel's retardation to the optical path; the orthogonal half reflects
/// straight back. Unpolarized light splits evenly.
#[test]
fn an_slm_steers_the_director_component_by_its_phase_gradient() {
    let mut s = Scene3::new();
    let id = s
        .add_modulator(Vec3::ZERO, 0.0, slm(Pattern::Generated(PatternGenerator::BlazedGrating { period_px: 16.0, angle_deg: 0.0 })))
        .unwrap();
    s.validate().unwrap();
    let lam = 0.633;
    let cfg = TraceConfig3 { record_events: true, ..TraceConfig3::default() };
    // Hit the middle of a pixel away from the reset.
    let y = 0.004 + 0.003 * 0.008;
    let res = trace3(&s, &[launch(Vec3::new(20.0, y, 0.0), -Vec3::X, lam)], &cfg);
    let dirs: Vec<Vec3> = res.segments.iter().skip(1).map(|g| (g.b - g.a).normalize()).collect();
    // Local X of the device is world Y: the steered ray's Y cosine is λ/Λ.
    let t = lam / 1000.0 / (16.0 * 0.008);
    let steered = Vec3::new((1.0 - t * t).sqrt(), t, 0.0);
    assert!(dirs.iter().any(|d| (*d - steered).length() < 1e-12), "{dirs:?} vs {steered:?} (#{id})");
    assert!(dirs.iter().any(|d| (*d - Vec3::X).length() < 1e-12), "the orthogonal half goes straight back");
    let flux: Vec<f64> = res.segments.iter().skip(1).map(|g| g.flux_start).collect();
    assert!(flux.iter().all(|f| (f - 0.5).abs() < 1e-12), "{flux:?}");
    assert!(res.capability_report.exact.contains(&PhysicsFeature::PhaseModulator));
}

/// A scene with a DMD showing an explicit bitmap round-trips through its
/// file; written as v2 the DMD becomes an absorber, and the loss is named.
#[test]
fn a_modulator_document_round_trips_and_writing_v2_notes_the_loss() {
    let mut s = Scene3::new();
    let mut d = dmd(32, 0.95, 0.89, 0.98, 1);
    d.pattern = Pattern::Explicit((0..32 * 32).map(|k| ((k % 32) / 4 % 2) as u16).collect());
    s.add_modulator(Vec3::ZERO, 10.0, d).unwrap();
    s.validate().unwrap();
    let json = s.to_json();
    let back = Scene3::from_json(&json).unwrap();
    assert_eq!(back.surfaces, s.surfaces);
    assert_eq!(back.to_json(), json);
    let (v2, losses) = oc_core::migrate_v2::to_v2(&s);
    assert!(matches!(v2.surfaces[0].response_front, SurfaceResponse::Absorber));
    assert!(losses.iter().any(|l| l.contains("DMD")), "{losses:?}");
}
