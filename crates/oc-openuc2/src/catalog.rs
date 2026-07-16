//! The generated static catalogue (§11).
//!
//! The browser and the tests both consume normalized JSON produced by the module
//! repository's compiler; nothing here reads `module.yaml`, and nothing here
//! talks to GitHub. A catalogue is pinned to a source commit, so a setup that
//! resolves today resolves identically tomorrow.
//!
//! Fragments arrive separately. `catalog.json` references a module's optics by
//! content-addressed asset path, so the document alone cannot place anything;
//! call [`Catalog::attach_fragment`] with each asset's `.ocanvas` text. That
//! split is what lets the browser lazy-load assets and the tests stay in memory,
//! without either inventing its own idea of what a module's optics are.

use std::collections::BTreeMap;

use oc_core::{Scene, SceneFragment};
use serde::{Deserialize, Serialize};

use crate::coordinates::GridProfile;
use crate::errors::Error;

/// The document tag of a generated catalogue.
pub const CATALOG_FORMAT: &str = "openuc2-canvas-catalog";

/// Where a catalogue came from. Pinned, never "latest".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSource {
    pub repository: String,
    /// Full commit sha of the module repository.
    pub commit: String,
}

/// How ready a module is, along four independent axes (§9.1).
///
/// Independent on purpose: a module can be perfectly good for layout while
/// having no optics at all, and collapsing these into one status is how a
/// layout-only cube starts looking like a simulated one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    #[serde(default = "unknown")]
    pub layout: String,
    #[serde(default = "unknown")]
    pub optical: String,
    #[serde(default = "unknown")]
    pub cad: String,
    #[serde(default = "unknown")]
    pub control: String,
}

fn unknown() -> String {
    "absent".to_string()
}

impl Capabilities {
    /// Does this module claim any Canvas optics at all?
    pub fn has_optics(&self) -> bool {
        !matches!(self.optical.as_str(), "absent" | "")
    }

    /// Are the optics declared exact rather than approximate?
    pub fn optics_validated(&self) -> bool {
        self.optical == "validated"
    }
}

/// The cells a module occupies, as integer offsets from its anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Footprint {
    #[serde(default)]
    pub anchor_cell: [i32; 3],
    pub cells: Vec<[i32; 3]>,
}

impl Footprint {
    /// A single-cell module.
    pub fn single() -> Self {
        Self {
            anchor_cell: [0, 0, 0],
            cells: vec![[0, 0, 0]],
        }
    }

    /// Cell offsets relative to the anchor, which is what rotation acts on.
    pub fn offsets(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        self.cells.iter().map(|c| {
            [
                c[0] - self.anchor_cell[0],
                c[1] - self.anchor_cell[1],
                c[2] - self.anchor_cell[2],
            ]
        })
    }
}

/// An optical or mechanical connector.
///
/// Advisory in the MVP: ports produce diagnostics, never automatic optical
/// topology. Free-space tracing does not consult them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Port {
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub face: Option<String>,
    #[serde(default)]
    pub pose: Option<PortPose>,
    #[serde(default)]
    pub aperture_mm: Option<f64>,
    #[serde(default)]
    pub compatibility: Vec<String>,
}

/// A port's position and outward normal in the canonical module frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortPose {
    pub position_mm: [f64; 3],
    /// The **outward** connector normal, unit length.
    pub direction: [f64; 3],
}

/// Module parameters. Values stay JSON: the adapter does not interpret them, and
/// §9.5 forbids executing anything a descriptor carries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Parameters {
    #[serde(default)]
    pub static_: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub variable: BTreeMap<String, serde_json::Value>,
}

/// Asset paths, relative to the catalogue root.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assets {
    #[serde(default)]
    pub icon: Option<String>,
    #[serde(default)]
    pub thumbnail: Option<String>,
    #[serde(default)]
    pub glb: Option<String>,
    /// The module's Canvas optics, as an `.ocanvas` fragment.
    #[serde(default)]
    pub canvas_fragment: Option<String>,
}

/// Product identity for the BOM.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Commercial {
    #[serde(default)]
    pub vendor: String,
    #[serde(default)]
    pub sku: String,
    #[serde(default)]
    pub price_eur: f64,
}

/// One normalized module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleEntry {
    pub id: String,
    pub uuid: String,
    pub version: String,
    pub name: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default = "Footprint::single")]
    pub footprint: Footprint,
    #[serde(default)]
    pub ports: Vec<Port>,
    #[serde(default)]
    pub parameters: Parameters,
    pub capabilities: Capabilities,
    #[serde(default)]
    pub assets: Assets,
    #[serde(default)]
    pub commercial: Commercial,
    #[serde(default)]
    pub source_path: Option<String>,
}

impl ModuleEntry {
    /// This module's pinned identity.
    pub fn module_ref(&self) -> ModuleRef {
        ModuleRef {
            uuid: self.uuid.clone(),
            version: self.version.clone(),
        }
    }
}

/// A pinned module identity.
///
/// Resolution is by `(uuid, version)` and never by display name: names are not
/// unique and are not stable, and a setup that resolved by name would silently
/// re-point at a different product when one was renamed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModuleRef {
    pub uuid: String,
    pub version: String,
}

impl ModuleRef {
    pub fn new(uuid: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            uuid: uuid.into(),
            version: version.into(),
        }
    }
}

/// The catalogue document, as generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogDoc {
    pub format: String,
    pub version: String,
    #[serde(default)]
    pub generated_at: Option<String>,
    pub source: CatalogSource,
    #[serde(default)]
    pub descriptor_schema_version: Option<String>,
    #[serde(default)]
    pub grid_profiles: Vec<GridProfile>,
    #[serde(default)]
    pub modules: Vec<ModuleEntry>,
}

/// A loaded catalogue, plus whatever fragment assets have been attached.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    doc: Option<CatalogDoc>,
    /// Asset path -> parsed fragment.
    fragments: BTreeMap<String, SceneFragment>,
    /// `(uuid, version)` -> index into `doc.modules`, built once at load.
    index: BTreeMap<ModuleRef, usize>,
}

impl Catalog {
    /// Parse a generated `catalog.json`.
    ///
    /// Rejects a document that is not a catalogue, and a duplicate
    /// `(uuid, version)`: the whole point of pinned resolution is that an
    /// identity names exactly one module, and a duplicate would make which one
    /// you get depend on iteration order.
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let doc: CatalogDoc = serde_json::from_str(json)
            .map_err(|e| Error::Format(format!("catalogue is not valid JSON: {e}")))?;
        if doc.format != CATALOG_FORMAT {
            return Err(Error::Format(format!(
                "not an openUC2 catalogue: format = {:?} (expected {CATALOG_FORMAT:?})",
                doc.format
            )));
        }

        let mut index = BTreeMap::new();
        for (i, m) in doc.modules.iter().enumerate() {
            if let Some(prev) = index.insert(m.module_ref(), i) {
                return Err(Error::Format(format!(
                    "catalogue lists {} at version {} twice (entries {} and {}); \
                     a pinned identity must name exactly one module",
                    m.uuid, m.version, prev, i
                )));
            }
        }

        Ok(Self {
            doc: Some(doc),
            fragments: BTreeMap::new(),
            index,
        })
    }

    /// Attach an asset's `.ocanvas` text, parsed as a module fragment.
    ///
    /// The document is read as a fragment, so its ambient medium and viewport are
    /// dropped and the destination scene's stay authoritative (§10.1).
    pub fn attach_fragment(&mut self, asset_path: &str, ocanvas: &str) -> Result<(), Error> {
        let scene = Scene::from_json(ocanvas).map_err(|e| {
            Error::Format(format!("fragment {asset_path:?} is not a valid .ocanvas: {e}"))
        })?;
        let fragment = SceneFragment::from_scene(&scene)?;
        self.fragments.insert(asset_path.to_string(), fragment);
        Ok(())
    }

    pub fn doc(&self) -> Option<&CatalogDoc> {
        self.doc.as_ref()
    }

    pub fn source(&self) -> Option<&CatalogSource> {
        self.doc.as_ref().map(|d| &d.source)
    }

    pub fn modules(&self) -> &[ModuleEntry] {
        self.doc.as_ref().map(|d| d.modules.as_slice()).unwrap_or(&[])
    }

    pub fn grid_profiles(&self) -> &[GridProfile] {
        self.doc
            .as_ref()
            .map(|d| d.grid_profiles.as_slice())
            .unwrap_or(&[])
    }

    pub fn grid_profile(&self, id: &str) -> Option<&GridProfile> {
        self.grid_profiles().iter().find(|g| g.id == id)
    }

    /// Resolve a pinned identity.
    ///
    /// Returns `None` rather than falling back to another version. §12.3 forbids
    /// substituting the newest: a setup pinned to 1.0.0 that silently became
    /// 2.0.0 would change the optics of a design without telling anyone.
    pub fn resolve(&self, r: &ModuleRef) -> Option<&ModuleEntry> {
        let i = *self.index.get(r)?;
        self.doc.as_ref().map(|d| &d.modules[i])
    }

    /// Every version of a module, oldest listing order preserved. For a UI that
    /// wants to tell the user what *is* available when a pin fails.
    pub fn versions_of(&self, uuid: &str) -> Vec<&ModuleEntry> {
        self.modules().iter().filter(|m| m.uuid == uuid).collect()
    }

    /// The parsed fragment for a module, if it declares one and the asset has
    /// been attached.
    pub fn fragment_for(&self, entry: &ModuleEntry) -> Option<&SceneFragment> {
        let path = entry.assets.canvas_fragment.as_ref()?;
        self.fragments.get(path)
    }

    /// Does this module declare a fragment whose asset is missing?
    pub fn fragment_asset_missing<'a>(&self, entry: &'a ModuleEntry) -> Option<&'a str> {
        let path = entry.assets.canvas_fragment.as_deref()?;
        if self.fragments.contains_key(path) {
            None
        } else {
            Some(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{minimal_catalog_json, LENS_UUID};

    #[test]
    fn a_catalogue_parses_and_indexes() {
        let cat = Catalog::from_json(&minimal_catalog_json()).unwrap();
        assert_eq!(cat.modules().len(), 5);
        assert_eq!(cat.source().unwrap().repository, "openUC2/openUC2-optikit-modules");
        assert!(cat.grid_profile("openuc2-core-50").is_some());
    }

    #[test]
    fn resolution_is_by_pinned_identity() {
        let cat = Catalog::from_json(&minimal_catalog_json()).unwrap();
        let found = cat.resolve(&ModuleRef::new(LENS_UUID, "1.0.0")).unwrap();
        assert_eq!(found.id, "lens-pcx-nbk7-1x1");
        // A version the catalogue does not have does not fall back to another.
        assert!(cat.resolve(&ModuleRef::new(LENS_UUID, "9.9.9")).is_none());
        assert!(cat.resolve(&ModuleRef::new("no-such-uuid", "1.0.0")).is_none());
    }

    #[test]
    fn a_newer_version_never_substitutes_for_a_pinned_one() {
        // Two versions of the same module: pinning must select exactly one.
        let mut doc: serde_json::Value =
            serde_json::from_str(&minimal_catalog_json()).unwrap();
        let mut v2 = doc["modules"][0].clone();
        v2["version"] = serde_json::json!("2.0.0");
        v2["name"] = serde_json::json!("Lens mk2");
        doc["modules"].as_array_mut().unwrap().push(v2);
        let cat = Catalog::from_json(&doc.to_string()).unwrap();

        assert_eq!(cat.resolve(&ModuleRef::new(LENS_UUID, "1.0.0")).unwrap().version, "1.0.0");
        assert_eq!(cat.resolve(&ModuleRef::new(LENS_UUID, "2.0.0")).unwrap().name, "Lens mk2");
        assert_eq!(cat.versions_of(LENS_UUID).len(), 2);
        // And a pin at a version that does not exist stays unresolved even though
        // newer versions are sitting right there.
        assert!(cat.resolve(&ModuleRef::new(LENS_UUID, "1.5.0")).is_none());
    }

    #[test]
    fn a_duplicate_pinned_identity_is_rejected() {
        let mut doc: serde_json::Value =
            serde_json::from_str(&minimal_catalog_json()).unwrap();
        let dup = doc["modules"][0].clone();
        doc["modules"].as_array_mut().unwrap().push(dup);
        let err = Catalog::from_json(&doc.to_string()).unwrap_err();
        assert!(format!("{err}").contains("twice"), "{err}");
    }

    #[test]
    fn a_foreign_document_is_rejected() {
        let err = Catalog::from_json(r#"{"format": "ocanvas", "version": "1.0.0",
            "source": {"repository": "", "commit": ""}}"#)
            .unwrap_err();
        assert!(format!("{err}").contains("not an openUC2 catalogue"), "{err}");
    }

    #[test]
    fn capabilities_are_independent() {
        let cat = Catalog::from_json(&minimal_catalog_json()).unwrap();
        let cube = cat.modules().iter().find(|m| m.id == "cube-empty-1x1").unwrap();
        // Layout-ready but optically absent: the pair must not collapse.
        assert_eq!(cube.capabilities.layout, "ready");
        assert!(!cube.capabilities.has_optics());
        assert!(!cube.capabilities.optics_validated());

        let lens = cat.resolve(&ModuleRef::new(LENS_UUID, "1.0.0")).unwrap();
        assert!(lens.capabilities.has_optics());
        assert!(lens.capabilities.optics_validated());
    }

    #[test]
    fn footprint_offsets_are_relative_to_the_anchor() {
        let f = Footprint {
            anchor_cell: [1, 1, 0],
            cells: vec![[1, 1, 0], [2, 1, 0]],
        };
        let offsets: Vec<_> = f.offsets().collect();
        assert_eq!(offsets, vec![[0, 0, 0], [1, 0, 0]]);
    }
}
