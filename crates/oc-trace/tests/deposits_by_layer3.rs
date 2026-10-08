//! Absorbed power by coating layer in the tracer (coating-fidelity spec
//! CF-04, C16): every coated object can say where it absorbed, layer by layer,
//! and the deposits add up to the ledger.
#![cfg(feature = "trace3")]

mod common;

use common::*;
use oc_core::fidelity::{FidelityProfile, LayerAbsorption};
use oc_core::scene3::{Scene3, SurfaceResponse};
use oc_core::thinfilm::{
    film_coating, materials, Coating, CoatingBacking, CoatingDesign, DesignKind, ExitFace, FilmLayer, FilmMaterial,
    Incidence,
};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, DepositLayer, ObjectRef3, TraceConfig3};

fn surface_with(front: Coating) -> Scene3 {
    let mut s = Scene3::new();
    let id = s.add_planar_mirror(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 0.5).unwrap();
    let surf = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
    surf.response_back = SurfaceResponse::Coating { coating: front.seen_from_behind() };
    surf.response_front = SurfaceResponse::Coating { coating: front };
    add_rect_detector(&mut s, Vec3::new(60.0, 0.0, 0.0), -Vec3::X, 200.0, 200.0, [2, 2]);
    add_rect_detector(&mut s, Vec3::new(-60.0, 0.0, 0.0), Vec3::X, 200.0, 200.0, [2, 2]);
    s.validate().expect("valid");
    s
}

fn per_layer(polarized: bool) -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        polarization: polarized,
        deposit_surfaces: true,
        object_ledger: true,
        ..TraceConfig3::default()
    }
    .with_fidelity(FidelityProfile { layer_absorption: LayerAbsorption::PerLayer, ..Default::default() })
}

fn rays() -> Vec<oc_trace::LaunchRay3> {
    [0.0f64, 20.0, 45.0]
        .iter()
        .flat_map(|&deg| {
            [0.4, 0.55, 0.8].map(|lam| {
                let d = Vec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
                launch(Ray3::new(Vec3::new(-40.0, -40.0 * deg.to_radians().tan(), 0.0), d).unwrap(), lam, 1.0)
            })
        })
        .collect()
}

/// C16: a silver film under silica on a standalone surface. Its deposits are
/// in the silver layer, and per object they sum to what the ledger says the
/// object absorbed.
#[test]
fn c16_a_standalone_surface_deposits_layer_by_layer() {
    let coating = Coating {
        layers: vec![FilmLayer::new(materials::silica(), 90.0), FilmLayer::new(materials::silver(), 40.0)],
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: None,
    };
    let s = surface_with(coating);
    for polarized in [true, false] {
        let r = trace3(&s, &rays(), &per_layer(polarized));
        assert!(!r.deposits.is_empty());
        assert!(r.deposits.iter().all(|d| d.layer == DepositLayer::Film(1)), "only the silver absorbs");
        let deposited: f64 = r.deposits.iter().map(|d| d.flux).sum();
        let booked: f64 = r
            .objects
            .iter()
            .filter(|e| matches!(e.object, ObjectRef3::Surface(_)))
            .map(|e| e.ledger.absorbed_surface + e.ledger.absorbed_volume)
            .sum();
        assert!(booked > 0.0);
        assert!((deposited - booked).abs() <= 1e-10 * booked, "{deposited} deposited, {booked} booked");
        assert!(r.ledger.residual().abs() < 1e-10, "{:?}", r.ledger);
    }
}

/// A plate's slab absorbs in the bulk: under `PerLayer` its share is booked
/// as volume absorption and deposited as `Bulk`; the film's as surface.
#[test]
fn c16_a_plate_puts_its_slab_in_the_volume() {
    let mut coating = Coating {
        layers: vec![FilmLayer::new(materials::silica(), 90.0), FilmLayer::new(materials::aluminium(), 6.0)],
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: None,
    };
    coating.backing = CoatingBacking::Plate {
        substrate: FilmMaterial::absorbing("absorbing glass", 1.5168, 2e-5),
        exit: ExitFace::Bare,
        thickness_mm: 5.0,
    };
    let s = surface_with(coating);
    let r = trace3(&s, &rays(), &per_layer(true));
    assert!(r.ledger.absorbed_volume > 0.0, "the slab's loss is volume absorption");
    let bulk: f64 = r.deposits.iter().filter(|d| d.layer == DepositLayer::Bulk).map(|d| d.flux).sum();
    let film: f64 = r.deposits.iter().filter(|d| d.layer == DepositLayer::Film(1)).map(|d| d.flux).sum();
    assert!((bulk - r.ledger.absorbed_volume).abs() <= 1e-10 * bulk.max(1e-30), "{bulk} vs {}", r.ledger.absorbed_volume);
    assert!((film - r.ledger.absorbed_surface).abs() <= 1e-10 * film.max(1e-30));
    assert!(r.ledger.residual().abs() < 1e-10);

    // Under PerFace (Standard) the same light is one surface number, as before.
    let face = trace3(&s, &rays(), &TraceConfig3 { ghosts: false, deposit_surfaces: true, ..TraceConfig3::default() });
    assert_eq!(face.ledger.absorbed_volume, 0.0);
    assert!((face.ledger.absorbed_surface - (r.ledger.absorbed_surface + r.ledger.absorbed_volume)).abs() < 1e-12);
    assert!(face.deposits.iter().all(|d| d.layer == DepositLayer::Face));
}

/// A lossless dielectric AR deposits nothing at all.
#[test]
fn c16_a_dielectric_ar_deposits_nothing() {
    let ar = film_coating(
        CoatingDesign { kind: DesignKind::BroadbandAr, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38 },
        1.0,
        1.0,
    );
    let s = surface_with(ar);
    let r = trace3(&s, &rays(), &per_layer(true));
    assert!(r.deposits.is_empty(), "{:?}", r.deposits);
    // The ledger books 1 - R - T, which for a lossless stack is rounding.
    assert!(r.ledger.absorbed_surface < 1e-15, "{}", r.ledger.absorbed_surface);
}
