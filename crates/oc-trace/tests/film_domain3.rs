//! Coatings at the edge of their domain, in the tracer (coating-fidelity spec
//! CF-02): an opaque substrate transmits nothing and books what it takes as
//! absorption, and under `numerics: Strict` a coating that has no answer is
//! refused with its reason.
#![cfg(feature = "trace3")]

mod common;

use common::*;
use oc_core::fidelity::{FidelityProfile, Numerics};
use oc_core::scene3::{Interpolation, OutOfRangePolicy, Scene3, SpectralScalar, SurfaceResponse};
use oc_core::thinfilm::{coating_rt_at, Coating, CoatingBacking, FilmLayer, FilmMaterial, Incidence};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::result3::CapabilityIssue;
use oc_trace::{trace3, TraceConfig3};

/// A standalone surface at the origin facing -X, both faces `front`/`back`,
/// a detector behind it and one in front.
fn bench(front: Coating) -> (Scene3, oc_core::id::ObjectId, oc_core::id::ObjectId) {
    let mut s = Scene3::new();
    let id = s.add_planar_mirror(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 0.5).unwrap();
    let surf = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
    surf.response_front = SurfaceResponse::Coating { coating: front };
    surf.response_back = SurfaceResponse::Absorber;
    let behind = add_rect_detector(&mut s, Vec3::new(60.0, 0.0, 0.0), -Vec3::X, 200.0, 200.0, [2, 2]);
    let before = add_rect_detector(&mut s, Vec3::new(-60.0, 0.0, 0.0), Vec3::X, 200.0, 200.0, [2, 2]);
    s.validate().expect("valid");
    (s, behind, before)
}

fn overcoated_aluminium() -> Coating {
    Coating {
        layers: vec![FilmLayer::new(FilmMaterial::constant("SiO2", 1.46), 94.0)],
        backing: CoatingBacking::Opaque { substrate: FilmMaterial::absorbing("Al", 0.958, 6.69) },
        incidence: Incidence::Film,
        design: None,
    }
}

/// C13, the tracer's half: no transmitted ray, the reflection the kernel
/// says, the rest absorbed, and the ledger closed.
#[test]
fn an_opaque_substrate_transmits_nothing_and_books_the_rest_as_absorbed() {
    let coating = overcoated_aluminium();
    let (s, behind, before) = bench(coating.clone());
    for deg in [0.0f64, 25.0, 50.0] {
        let d = Vec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
        let ray = Ray3::new(Vec3::new(-40.0, -40.0 * deg.to_radians().tan(), 0.0), d).unwrap();
        let r = trace3(&s, &[launch(ray, 0.55, 1.0)], &TraceConfig3 { ghosts: false, ..TraceConfig3::default() });
        let want = coating_rt_at(&coating, 1.0, 1.0, 0.55, deg.to_radians().cos()).unwrap();
        assert_eq!(detector_named(&r, behind).hits, 0, "{deg} deg: something went through");
        let back = detector_named(&r, before).total_incident;
        assert!((back - want.r()).abs() < 1e-12, "{deg} deg: R {back} vs {}", want.r());
        assert!((r.ledger.absorbed_surface - (1.0 - want.r())).abs() < 1e-12);
        assert!(r.ledger.residual().abs() < 1e-10, "{:?}", r.ledger);
        assert!(r.capability_report.rejected.is_empty());
    }
}

/// A protected-silver film read far outside its table: answered (clamped and
/// flagged) under `Permissive`, refused with the reason under `Strict`.
#[test]
fn strict_refuses_a_coating_outside_its_table_and_says_why() {
    let silver_table = |values: Vec<f64>| SpectralScalar::Tabulated {
        lambda_um: vec![0.3, 0.5, 0.9],
        values,
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    };
    let coating = Coating {
        layers: vec![FilmLayer::new(
            FilmMaterial::scalars("Ag", silver_table(vec![0.05, 0.05, 0.04]), silver_table(vec![1.3, 3.0, 6.0])),
            150.0,
        )],
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: None,
    };
    let (s, _, _) = bench(coating);
    let ray = Ray3::new(Vec3::new(-40.0, 0.0, 0.0), Vec3::X).unwrap();
    let launch = [launch(ray, 1.6, 1.0)];
    let base = TraceConfig3 { ghosts: false, ..TraceConfig3::default() };

    let permissive = trace3(&s, &launch, &base);
    assert!(permissive.capability_report.rejected.is_empty());
    assert!(
        permissive.capability_report.approximated.iter().any(|a| a.detail.contains("outside its range")),
        "{:?}",
        permissive.capability_report.approximated
    );

    let strict = trace3(&s, &launch, &base.with_fidelity(FidelityProfile { numerics: Numerics::Strict, ..Default::default() }));
    let reason = strict
        .capability_report
        .rejected
        .iter()
        .find_map(|i| match i {
            CapabilityIssue::UnsupportedSurfaceResponse { reason: Some(r), .. } => Some(r.clone()),
            _ => None,
        })
        .expect("a refusal with its reason");
    assert!(reason.starts_with("layer 1 has no optical constants at 1.6000 um"), "{reason}");
    assert!((strict.ledger.numerical_loss - 1.0).abs() < 1e-12, "the refused ray is named, not lost");
}

/// C18, the tracer's half: polarized light through a coated plate is
/// approximated under `PhaseFree` and says so, and under `Incoherent` it goes
/// through the plate's Mueller matrix with no such note; the power is the
/// same either way to the slab's lossy-face convention.
#[test]
fn an_incoherent_plate_carries_polarization_without_an_approximation() {
    use oc_core::fidelity::PlateModel;
    use oc_core::polarization::{PolRay, Stokes};
    use oc_core::thinfilm::ExitFace;
    let plate = Coating {
        layers: vec![
            FilmLayer::new(FilmMaterial::constant("TiO2", 2.35), 58.0),
            FilmLayer::new(FilmMaterial::constant("SiO2", 1.46), 94.0),
        ],
        backing: CoatingBacking::Plate {
            substrate: FilmMaterial::absorbing("glass", 1.5168, 2e-6),
            exit: ExitFace::Bare,
            thickness_mm: 3.0,
            round_trips: Default::default(),
        },
        incidence: Incidence::Film,
        design: None,
    };
    let (s, behind, _) = bench(plate);
    let d = Vec3::new(45f64.to_radians().cos(), 45f64.to_radians().sin(), 0.0);
    let ray = Ray3::new(Vec3::new(-40.0, -40.0, 0.0), d).unwrap();
    let mut lr = launch(ray, 0.55, 1.0);
    lr.pol = Some(PolRay { stokes: Stokes::linear(1.0, std::f64::consts::FRAC_PI_4), x: Vec3::Z });
    let base = TraceConfig3 { ghosts: false, ..TraceConfig3::default() };
    let note = "plate-backed stack: faces added incoherently, no relative phase";

    let phase_free = trace3(&s, &[lr], &base);
    assert!(phase_free.capability_report.approximated.iter().any(|a| a.detail == note));

    let incoherent = trace3(&s, &[lr], &base.with_fidelity(FidelityProfile { plate_model: PlateModel::Incoherent, ..Default::default() }));
    assert!(!incoherent.capability_report.approximated.iter().any(|a| a.detail == note));
    let a = detector_named(&phase_free, behind);
    let b = detector_named(&incoherent, behind);
    assert!((a.total_incident - b.total_incident).abs() < 1e-5, "{} vs {}", a.total_incident, b.total_incident);
    assert!(b.stokes_local.dop() <= 1.0 + 1e-12);
}
