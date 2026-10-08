//! Film materials, the scene catalogue and document v3 (coating-fidelity spec
//! CF-03): C30, the catalogue round trip, and the lossy v2 export.
#![cfg(feature = "scene3")]

use oc_core::document::{read_document, Document};
use oc_core::fidelity::{FidelityProfile, FilmMaterials};
use oc_core::film_catalog::FilmCatalogEntry;
use oc_core::migrate_v2::coatings;
use oc_core::scene3::{Scene3, SceneError, SurfaceResponse};
use oc_core::thinfilm::{
    coating_eval, coating_rt_at, film_coating, materials, metal_mirror, Coating, CoatingBacking, CoatingDesign,
    DesignKind, FilmContext, FilmLayer, Incidence, SourceKind,
};
use oc_core::vec3::Vec3;

/// A scene with every kind of coating the canvas builds, from the nominal
/// constants and the tabulated metals: what a v2 document holds.
fn coated_v2_scene() -> Scene3 {
    let mut s = Scene3::default_scene();
    s.add_film_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 135.0, 560.0, false).unwrap();
    s.add_plate_dichroic(Vec3::new(-35.0, 40.0, 0.0), 15.0, 3.0, -45.0, 560.0, true).unwrap();
    let mirror = metal_mirror(materials::silver(), Some((materials::silica(), 90.0)));
    s.add_coated_plate(Vec3::new(-45.0, 0.0, 0.0), 15.0, 3.0, 120.0, mirror);
    let ar = film_coating(
        CoatingDesign { kind: DesignKind::BroadbandAr, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38 },
        1.0,
        1.5168,
    );
    s.add_coated_plate(Vec3::new(-25.0, 0.0, 0.0), 15.0, 3.0, 160.0, ar);
    s
}

/// Every coating's response over a grid of wavelengths and angles.
fn responses(s: &Scene3) -> Vec<String> {
    let mut out = Vec::new();
    for c in coatings(s) {
        for lam in [0.3, 0.45, 0.55, 0.7, 1.2, 2.4] {
            for deg in [0.0f64, 30.0, 60.0] {
                let rt = coating_rt_at(c, 1.0, 1.5168, lam, deg.to_radians().cos());
                out.push(format!("{rt:?}"));
            }
        }
    }
    out
}

/// C30: a v2 document read into v3 traces exactly as it did, its inline
/// materials are recorded as nominal and catalogued, and written back as v2
/// it is the document it was.
#[test]
fn c30_migration_from_v2_changes_no_number() {
    let original = coated_v2_scene();
    let (v2, losses) = original.to_json_v2();
    assert!(losses.is_empty(), "{losses:?}");
    assert!(v2.contains("\"version\": 2") && !v2.contains("film_catalog"));

    // Read without migrating (the v2 reader's view) and with.
    let raw: Scene3 = serde_json::from_str(&v2).unwrap();
    let migrated = Scene3::from_json(&v2).unwrap();
    assert_eq!(migrated.version, 3);
    assert!(!coatings(&raw).is_empty());
    assert_eq!(responses(&raw), responses(&migrated), "migration moved a number");

    // Every inline material became a nominal catalogue entry its layers name.
    assert!(!migrated.film_catalog.is_empty());
    for c in coatings(&migrated) {
        for l in &c.layers {
            let id = l.catalog_ref.expect("a migrated layer names its entry");
            let e = migrated.film_catalog.iter().find(|e| e.id == id).unwrap();
            assert_eq!(e.material, l.material);
            assert_eq!(l.material.provenance.as_ref().unwrap().source_kind, SourceKind::Nominal);
        }
    }
    // One entry per distinct material, not one per layer.
    let layers: usize = coatings(&migrated).iter().map(|c| c.layers.len()).sum();
    assert!(migrated.film_catalog.len() < layers);

    // Written back as v2 it is what was read, and the document layer keeps
    // a v2 file v2.
    let (back, losses) = migrated.to_json_v2();
    assert!(losses.is_empty(), "{losses:?}");
    let a: serde_json::Value = serde_json::from_str(&v2).unwrap();
    let b: serde_json::Value = serde_json::from_str(&back).unwrap();
    assert_eq!(a, b);
    let doc = read_document(&v2).unwrap();
    assert!(matches!(doc, Document::V2(_)));
    assert_eq!(serde_json::from_str::<serde_json::Value>(&doc.to_json()).unwrap(), a);
}

/// Every v2 document committed in the repository migrates without moving a
/// coating's number.
#[test]
fn c30_every_committed_v2_document_migrates_bit_identically() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut stack = vec![root.join("web/public"), root.join("crates"), root.join("examples")];
    let mut checked = 0;
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if !p.ends_with("target") && !p.ends_with("node_modules") {
                    stack.push(p);
                }
                continue;
            }
            if p.extension().and_then(|x| x.to_str()) != Some("ocanvas") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap();
            let Ok(raw) = serde_json::from_str::<Scene3>(&text) else { continue };
            if raw.version != 2 {
                continue;
            }
            let migrated = Scene3::from_json(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert_eq!(responses(&raw), responses(&migrated), "{}", p.display());
            checked += 1;
        }
    }
    assert!(checked >= 5, "only {checked} v2 documents found");
}

/// The catalogue is the truth: a layer that names an entry takes the
/// entry's material on load, whatever its inline copy says.
#[test]
fn a_layer_takes_its_material_from_the_catalogue() {
    let mut s = Scene3::default_scene();
    let silica = materials::catalog::silica();
    s.film_catalog.push(FilmCatalogEntry { id: 7, material: silica.clone() });
    let mut layer = FilmLayer::new(materials::silica(), 94.0); // the nominal constant inline
    layer.catalog_ref = Some(7);
    let coating = Coating { layers: vec![layer], backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None };
    let id = s.add_planar_mirror(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 0.5).unwrap();
    let surf = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
    surf.response_front = SurfaceResponse::Coating { coating };

    let back = Scene3::from_json(&s.to_json()).unwrap();
    let c = coatings(&back).into_iter().next().unwrap();
    assert_eq!(c.layers[0].material, silica, "the catalogue entry replaced the inline constant");
    assert_eq!(back.film_catalog.len(), 1);

    // A layer naming an entry that is not there is refused by name.
    let broken = s.to_json().replace("\"ref\": 7", "\"ref\": 8");
    match Scene3::from_json(&broken) {
        Err(SceneError::OutOfRange { field, reason, .. }) => {
            assert!(field.contains("ref") && reason.contains("entry 8"), "{field}: {reason}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// The profile round-trips through v3 and is dropped, with a note, by the v2
/// export; so are a formula, the provenance and an opaque substrate.
#[test]
fn the_v2_export_says_what_it_drops() {
    let mut s = Scene3::default_scene();
    s.fidelity = FidelityProfile { film_materials: FilmMaterials::Nominal, ..Default::default() };
    let hafnia = materials::catalog::hafnia();
    let coating = Coating {
        layers: vec![FilmLayer::new(hafnia, 80.0)],
        backing: CoatingBacking::Opaque { substrate: materials::catalog::aluminium() },
        incidence: Incidence::Film,
        design: None,
    };
    let id = s.add_planar_mirror(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 0.5).unwrap();
    let surf = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
    surf.response_front = SurfaceResponse::Coating { coating: coating.clone() };

    let back = Scene3::from_json(&s.to_json()).unwrap();
    assert_eq!(back.fidelity, s.fidelity);

    let (v2, losses) = s.to_json_v2();
    let all = losses.join(" | ");
    assert!(all.contains("fidelity profile"), "{all}");
    assert!(all.contains("dispersion formula"), "{all}");
    assert!(all.contains("provenance"), "{all}");
    assert!(all.contains("opaque"), "{all}");
    let old: Scene3 = serde_json::from_str(&v2).unwrap();
    assert_eq!(old.version, 2);
    // The v2 stand-in for the opaque substrate reflects what the original
    // does, to the precision of the sampled formula.
    let c2 = coatings(&old).into_iter().next().unwrap();
    let a = coating_rt_at(&coating, 1.0, 1.0, 0.55, 1.0).unwrap();
    let b = coating_rt_at(c2, 1.0, 1.0, 0.55, 1.0).unwrap();
    assert!((a.r() - b.r()).abs() < 1e-4, "{} vs {}", a.r(), b.r());
}

/// `film_materials: Nominal` evaluates each material at its nominal constant;
/// a constant material is its constant either way, which is why Standard,
/// dispersive, reproduces the constant-built corpora.
#[test]
fn nominal_materials_are_the_nominal_constants() {
    let ctx = FilmContext { materials: FilmMaterials::Nominal, ..FilmContext::default() };
    let dispersive = FilmContext::default();
    let constant = Coating {
        layers: vec![FilmLayer::new(materials::titania(), 60.0), FilmLayer::new(materials::silica(), 94.0)],
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: None,
    };
    for lam in [0.4, 0.55, 1.0] {
        let a = coating_eval(&constant, 1.0, 1.52, lam, 0.9, &ctx).unwrap();
        let b = coating_eval(&constant, 1.0, 1.52, lam, 0.9, &dispersive).unwrap();
        assert_eq!(a, b, "a constant is its own nominal");
    }
    let measured = Coating {
        layers: vec![FilmLayer::new(materials::catalog::titania(), 60.0), FilmLayer::new(materials::catalog::silica(), 94.0)],
        ..constant
    };
    let at = |lam: f64, c: &FilmContext| coating_eval(&measured, 1.0, 1.52, lam, 1.0, c).unwrap().0.r();
    // At 0.55 um, where the nominal is taken, the two agree; elsewhere they do not.
    assert!((at(0.55, &ctx) - at(0.55, &dispersive)).abs() < 1e-12);
    assert!((at(0.4, &ctx) - at(0.4, &dispersive)).abs() > 1e-4);
}
