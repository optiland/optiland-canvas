//! Document format v3, and the way between it and v2 (coating-fidelity spec
//! CF-03, §10).
//!
//! v3 adds a film catalogue with provenance, layers that name their catalogue
//! entry, materials with a formula, a form, a source and a nominal value, the
//! opaque backing and the fidelity profile. Every addition is optional on
//! read, so a v3 reader reads v2 unchanged.
//!
//! Reading v2 migrates in memory: each inline layer material is recorded as
//! what it was (`source_kind: Nominal`, form `Bulk`) and becomes an anonymous
//! catalogue entry its layers name. Nothing a coating evaluates changes, so
//! `coating_rt_at` is bit-identical before and after (C30).
//!
//! Writing v2 is the reverse: [`to_v2`] drops every v3 addition. For a scene
//! read from v2 that is exact. For one that uses v3 it is lossy, and says
//! what it dropped: a formula becomes the table sampled from it, an opaque
//! substrate becomes a millimetre of that material on the scene's far medium,
//! provenance and the profile go.

use crate::scene3::{Scene3, SceneError, SurfaceResponse};
use crate::thinfilm::{Coating, CoatingBacking, FilmLayer, FilmMaterial, Provenance};

/// Run `f` on every coating in the scene: standalone surfaces, both sides,
/// and every body's default and per-face responses.
pub fn for_each_coating_mut(scene: &mut Scene3, mut f: impl FnMut(&mut Coating)) {
    let mut visit = |r: &mut SurfaceResponse| {
        if let SurfaceResponse::Coating { coating } = r {
            f(coating);
        }
    };
    for s in &mut scene.surfaces {
        visit(&mut s.response_front);
        visit(&mut s.response_back);
    }
    for b in &mut scene.bodies {
        visit(&mut b.surface.default.from_outside);
        visit(&mut b.surface.default.from_inside);
        for o in &mut b.surface.overrides {
            visit(&mut o.response.from_outside);
            visit(&mut o.response.from_inside);
        }
    }
}

/// Every coating in the scene, read-only, in the order
/// [`for_each_coating_mut`] visits them.
pub fn coatings(scene: &Scene3) -> Vec<&Coating> {
    fn visit(r: &SurfaceResponse) -> Option<&Coating> {
        match r {
            SurfaceResponse::Coating { coating } => Some(coating),
            _ => None,
        }
    }
    let mut out = Vec::new();
    for s in &scene.surfaces {
        out.extend(visit(&s.response_front));
        out.extend(visit(&s.response_back));
    }
    for b in &scene.bodies {
        out.extend(visit(&b.surface.default.from_outside));
        out.extend(visit(&b.surface.default.from_inside));
        for o in &b.surface.overrides {
            out.extend(visit(&o.response.from_outside));
            out.extend(visit(&o.response.from_inside));
        }
    }
    out
}

const V2_NOTE: &str = "an inline material from a v2 document";

/// Migrate a scene read from a v2 document to v3, in place.
pub fn migrate_v2_to_v3(scene: &mut Scene3) {
    let mut catalog = std::mem::take(&mut scene.film_catalog);
    let mut next = catalog.iter().map(|e| e.id + 1).max().unwrap_or(1);
    for_each_coating_mut(scene, |c| {
        for layer in &mut c.layers {
            if layer.catalog_ref.is_some() {
                continue;
            }
            if layer.material.provenance.is_none() {
                layer.material.provenance = Some(Box::new(Provenance::nominal(V2_NOTE)));
            }
            let id = match catalog.iter().find(|e| e.material == layer.material) {
                Some(e) => e.id,
                None => {
                    let id = next;
                    next += 1;
                    catalog.push(crate::film_catalog::FilmCatalogEntry { id, material: layer.material.clone() });
                    id
                }
            };
            layer.catalog_ref = Some(id);
        }
        match &mut c.backing {
            CoatingBacking::Plate { substrate, .. } | CoatingBacking::Opaque { substrate } => {
                if substrate.provenance.is_none() {
                    substrate.provenance = Some(Box::new(Provenance::nominal(V2_NOTE)));
                }
            }
            CoatingBacking::Medium => {}
        }
    });
    scene.film_catalog = catalog;
    scene.version = crate::scene3::OCANVAS_V3_VERSION;
}

/// Put each catalogue entry's material into every layer that names it. The
/// catalogue is the truth; the inline copy is for readers that know no
/// catalogue. Fails on a layer naming an entry that is not there.
pub fn resolve_film_refs(scene: &mut Scene3) -> Result<(), SceneError> {
    let catalog = scene.film_catalog.clone();
    let mut missing = None;
    for_each_coating_mut(scene, |c| {
        for layer in &mut c.layers {
            if let Some(id) = layer.catalog_ref {
                match catalog.iter().find(|e| e.id == id) {
                    Some(e) => layer.material = e.material.clone(),
                    None => missing = Some(id),
                }
            }
        }
    });
    match missing {
        Some(id) => Err(SceneError::OutOfRange {
            object: None,
            field: "coating.layers[].ref".to_string(),
            reason: format!("names film catalogue entry {id}, which the scene does not have"),
        }),
        None => Ok(()),
    }
}

/// A v2 reader's copy of a material: its two curves and its name.
fn v2_material(m: &FilmMaterial, what: &str, losses: &mut Vec<String>) -> FilmMaterial {
    if m.formula.is_some() {
        losses.push(format!("{what}: the dispersion formula of {:?} is written as the table sampled from it", m.name));
    }
    FilmMaterial::scalars(&m.name, m.n.clone(), m.k.clone())
}

/// The scene as a v2 document, and what v2 could not say.
///
/// Empty losses means the v2 document holds everything the scene does, which
/// is the case for any scene read from v2 and not edited with a v3 feature.
/// A v2 reader knows the six original design kinds and no `adapt` record.
/// The layers are written as they are either way, since they are what the
/// transfer matrix reads; only the recipe goes. A substrate-adapted AR is
/// written as `SingleLayerAr`, which keeps its residual reflection a ghost in
/// v2; an enhanced metal loses its record and reads as a path, which a mirror
/// is.
fn design_to_v2(c: &mut Coating, losses: &mut Vec<String>) {
    use crate::thinfilm::DesignKind as K;
    let Some(d) = c.design.as_mut() else {
        return;
    };
    match d.kind {
        K::VCoat | K::Bbar | K::BandAr => {
            losses.push(format!(
                "a {:?} design is written as SingleLayerAr: v2 has no such kind; the layers are kept, the recipe is not",
                d.kind
            ));
            d.kind = K::SingleLayerAr;
            d.adapt = None;
        }
        K::EnhancedMetal { .. } => {
            losses.push("an enhanced-metal design record is dropped: v2 has no such kind; the layers are kept".into());
            c.design = None;
        }
        _ if d.adapt.is_some() => {
            losses.push(format!("the substrate adaptation of a {:?} design is dropped; the layers are kept", d.kind));
            d.adapt = None;
        }
        _ => {}
    }
}

/// A v2 reader has no measured table. Each is written as an angle-free
/// `ExplicitRt` of its unpolarized powers at its first angle, refusing outside
/// its wavelengths, with a loss line naming the angle.
fn tables_to_v2(scene: &mut Scene3, losses: &mut Vec<String>) {
    use crate::scene3::{Interpolation, OutOfRangePolicy, SpectralScalar};
    let mut visit = |r: &mut SurfaceResponse| {
        let SurfaceResponse::MeasuredTable { table, .. } = r else {
            return;
        };
        let aoi = table.aoi_deg[0];
        let cos = aoi.to_radians().cos();
        let at: Vec<_> = table.lambda_um.iter().map(|&l| table.powers_clamped(l, cos)).collect();
        let curve = |values: Vec<f64>| SpectralScalar::Tabulated {
            lambda_um: table.lambda_um.clone(),
            values,
            interpolation: Interpolation::Linear,
            out_of_range: OutOfRangePolicy::Error,
        };
        losses.push(format!(
            "a measured table is written as an angle-free reflectance and transmittance at {aoi} degrees, unpolarized"
        ));
        *r = SurfaceResponse::ExplicitRt {
            reflectance: curve(at.iter().map(|p| p.r()).collect()),
            transmittance: curve(at.iter().map(|p| p.t()).collect()),
        };
    };
    for s in &mut scene.surfaces {
        visit(&mut s.response_front);
        visit(&mut s.response_back);
    }
    for b in &mut scene.bodies {
        visit(&mut b.surface.default.from_outside);
        visit(&mut b.surface.default.from_inside);
        for o in &mut b.surface.overrides {
            visit(&mut o.response.from_outside);
            visit(&mut o.response.from_inside);
        }
    }
}

pub fn to_v2(scene: &Scene3) -> (Scene3, Vec<String>) {
    let mut s = scene.clone();
    let mut losses = Vec::new();
    let sourced = |m: &FilmMaterial| {
        m.provenance.as_ref().is_some_and(|p| p.note != V2_NOTE) || m.nominal.is_some() || m.form != crate::thinfilm::FilmForm::Bulk
    };
    for_each_coating_mut(&mut s, |c| {
        design_to_v2(c, &mut losses);
        for (i, layer) in c.layers.iter_mut().enumerate() {
            if sourced(&layer.material) {
                losses.push(format!("layer {} ({}): provenance, form and nominal value dropped", i + 1, layer.material.name));
            }
            if layer.incoherent {
                losses.push(format!(
                    "layer {} is a {} mm slab of {}; v2 has no slabs and traces it as a coherent film",
                    i + 1,
                    layer.thickness_nm / 1e6,
                    layer.material.name
                ));
            }
            let m = v2_material(&layer.material, &format!("layer {}", i + 1), &mut losses);
            *layer = FilmLayer::new(m, layer.thickness_nm);
        }
        match &mut c.backing {
            CoatingBacking::Plate { substrate, .. } => {
                *substrate = v2_material(substrate, "plate substrate", &mut losses);
            }
            CoatingBacking::Opaque { substrate } => {
                losses.push(format!(
                    "the opaque {} substrate is written as 1 mm of it on the scene's far medium",
                    substrate.name
                ));
                let m = v2_material(substrate, "opaque substrate", &mut losses);
                c.layers.push(FilmLayer::new(m, 1.0e6));
                c.backing = CoatingBacking::Medium;
            }
            CoatingBacking::Medium => {}
        }
    });
    tables_to_v2(&mut s, &mut losses);
    if !s.film_catalog.is_empty() {
        s.film_catalog.clear();
    }
    if !s.fidelity.is_default() {
        losses.push(format!("the fidelity profile ({}) is dropped; v2 traces as Standard", s.fidelity.describe()));
        s.fidelity = crate::fidelity::FidelityProfile::default();
    }
    s.version = crate::scene3::OCANVAS_V2_VERSION;
    losses.dedup();
    (s, losses)
}
