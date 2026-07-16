//! The `.uc2canvas` setup document (§12).
//!
//! A setup is what the user saves: which catalogue modules are assembled, where
//! they sit, and any custom Canvas geometry drawn alongside them. It is
//! deliberately *not* an `.ocanvas` scene — the scene is derived from this and
//! regenerated, never edited in parallel (§5.1).
//!
//! What a setup stores about a module is its **pinned identity** plus enough
//! fallback data to still show something useful when the catalogue no longer has
//! it. Module-derived geometry is never serialized here: it comes from the
//! catalogue, and duplicating it is how the two drift apart.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::catalog::ModuleRef;
use crate::errors::Error;

/// The document tag of a setup.
pub const SETUP_FORMAT: &str = "openuc2-canvas-setup";
/// The setup schema version this build writes.
pub const SETUP_VERSION: &str = "1.0.0";

/// Free-text document metadata. Never affects scene generation (§12.3.8).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetupMetadata {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub modified: Option<String>,
}

/// The catalogue this setup was authored against, pinned.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogPin {
    #[serde(default)]
    pub repository: String,
    #[serde(default)]
    pub commit: String,
    #[serde(default)]
    pub catalog_sha256: String,
    pub grid_profile: String,
}

/// View state. Additive and backward compatible (§12.3.7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewState {
    #[serde(default)]
    pub active_layer: i32,
    #[serde(default = "default_trace_plane")]
    pub trace_plane: String,
    #[serde(default = "yes")]
    pub show_footprints: bool,
    #[serde(default)]
    pub show_ports: bool,
    #[serde(default = "yes")]
    pub show_optical_geometry: bool,
}

fn default_trace_plane() -> String {
    "XY".to_string()
}
fn yes() -> bool {
    true
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            active_layer: 0,
            trace_plane: default_trace_plane(),
            show_footprints: true,
            show_ports: false,
            show_optical_geometry: true,
        }
    }
}

/// What the setup remembers about a module in case the catalogue loses it.
///
/// Enough to draw a labelled footprint and tell the user what is missing. It is
/// not enough to trace, and deliberately so: a placeholder must look like a
/// placeholder, not like a working module.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceFallback {
    #[serde(default)]
    pub module_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub footprint_cells: Vec<[i32; 3]>,
}

/// Where an instance sits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    pub anchor_grid: [i32; 3],
    #[serde(default)]
    pub rotation_quadrants: u8,
}

/// One placed module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetupInstance {
    pub instance_id: String,
    pub module_uuid: String,
    pub module_version: String,
    #[serde(default)]
    pub fallback: InstanceFallback,
    pub placement: Placement,
    /// Overrides keyed by parameter name. A `BTreeMap` so the serialized order
    /// is the key order and a diff means something.
    #[serde(default)]
    pub parameter_overrides: BTreeMap<String, serde_json::Value>,
    #[serde(default = "yes")]
    pub enabled: bool,
}

impl SetupInstance {
    pub fn module_ref(&self) -> ModuleRef {
        ModuleRef::new(self.module_uuid.clone(), self.module_version.clone())
    }
}

/// Custom Canvas content drawn alongside the modules.
///
/// Held as a raw `.ocanvas` document so the setup format never has to track the
/// scene format's shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CustomCanvas {
    #[serde(default)]
    pub ocanvas: serde_json::Value,
}

/// The `.uc2canvas` document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenUc2Setup {
    pub format: String,
    pub version: String,
    #[serde(default)]
    pub metadata: SetupMetadata,
    pub catalog: CatalogPin,
    #[serde(default)]
    pub view: ViewState,
    #[serde(default)]
    pub instances: Vec<SetupInstance>,
    #[serde(default)]
    pub custom_canvas: CustomCanvas,
}

impl OpenUc2Setup {
    /// An empty setup against `grid_profile`.
    pub fn new(grid_profile: impl Into<String>) -> Self {
        Self {
            format: SETUP_FORMAT.to_string(),
            version: SETUP_VERSION.to_string(),
            metadata: SetupMetadata::default(),
            catalog: CatalogPin {
                grid_profile: grid_profile.into(),
                ..Default::default()
            },
            view: ViewState::default(),
            instances: Vec::new(),
            custom_canvas: CustomCanvas::default(),
        }
    }

    /// Parse a `.uc2canvas` document.
    ///
    /// Rejects a foreign format and a future major version rather than
    /// mis-reading either. Migrations are explicit and versioned (§12.3.10);
    /// there are none yet, so a newer major is simply refused.
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let setup: OpenUc2Setup = serde_json::from_str(json)
            .map_err(|e| Error::Format(format!("setup is not valid JSON: {e}")))?;
        if setup.format != SETUP_FORMAT {
            return Err(Error::Format(format!(
                "not an openUC2 setup: format = {:?} (expected {SETUP_FORMAT:?})",
                setup.format
            )));
        }
        setup.check_version()?;

        let mut seen = std::collections::BTreeSet::new();
        for i in &setup.instances {
            if !seen.insert(i.instance_id.clone()) {
                return Err(Error::Format(format!(
                    "setup lists instance {:?} twice; instance ids address objects \
                     and must be unique",
                    i.instance_id
                )));
            }
        }
        Ok(setup)
    }

    /// Refuse a document a newer writer produced.
    fn check_version(&self) -> Result<(), Error> {
        let major = |v: &str| -> u32 {
            v.split('.').next().unwrap_or("0").parse().unwrap_or(0)
        };
        let ours = major(SETUP_VERSION);
        let theirs = major(&self.version);
        if theirs > ours {
            return Err(Error::Format(format!(
                "setup version {} is newer than this build supports ({SETUP_VERSION})",
                self.version
            )));
        }
        Ok(())
    }

    /// Serialize deterministically.
    ///
    /// Struct field order is declaration order and every map is a `BTreeMap`, so
    /// the same setup always produces the same bytes and a diff shows only what
    /// the user actually changed.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("setup serializes")
    }

    /// The next free `m-NNNN` instance id.
    pub fn next_instance_id(&self) -> String {
        let mut n = 1;
        loop {
            let candidate = format!("m-{n:04}");
            if !self.instances.iter().any(|i| i.instance_id == candidate) {
                return candidate;
            }
            n += 1;
        }
    }

    pub fn instance(&self, id: &str) -> Option<&SetupInstance> {
        self.instances.iter().find(|i| i.instance_id == id)
    }

    pub fn instance_mut(&mut self, id: &str) -> Option<&mut SetupInstance> {
        self.instances.iter_mut().find(|i| i.instance_id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> OpenUc2Setup {
        let mut s = OpenUc2Setup::new("openuc2-core-50");
        s.metadata.name = "Folded singlet demo".to_string();
        s.catalog.commit = "abc123".to_string();
        s.instances.push(SetupInstance {
            instance_id: "m-0001".to_string(),
            module_uuid: "u-1".to_string(),
            module_version: "1.0.0".to_string(),
            fallback: InstanceFallback {
                module_id: "lens-pcx-nbk7-1x1".to_string(),
                name: "Lens".to_string(),
                color: Some("#7CC142".to_string()),
                footprint_cells: vec![[0, 0, 0]],
            },
            placement: Placement {
                anchor_grid: [2, 1, 0],
                rotation_quadrants: 1,
            },
            parameter_overrides: BTreeMap::new(),
            enabled: true,
        });
        s
    }

    #[test]
    fn setup_round_trips() {
        let s = sample();
        let back = OpenUc2Setup::from_json(&s.to_json()).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn serialization_is_deterministic() {
        let s = sample();
        assert_eq!(s.to_json(), s.to_json());
        // And stable across a round trip, which is what makes a diff meaningful.
        let back = OpenUc2Setup::from_json(&s.to_json()).unwrap();
        assert_eq!(back.to_json(), s.to_json());
    }

    #[test]
    fn parameter_overrides_serialize_in_key_order() {
        // Insertion order must not leak into the file: two setups with the same
        // overrides added in different orders must produce identical bytes.
        let mut a = sample();
        let mut b = sample();
        for (k, v) in [("zoom", 2), ("aperture", 1), ("focus", 3)] {
            a.instances[0]
                .parameter_overrides
                .insert(k.to_string(), serde_json::json!(v));
        }
        for (k, v) in [("focus", 3), ("zoom", 2), ("aperture", 1)] {
            b.instances[0]
                .parameter_overrides
                .insert(k.to_string(), serde_json::json!(v));
        }
        assert_eq!(a.to_json(), b.to_json());
    }

    #[test]
    fn a_foreign_document_is_rejected() {
        let err = OpenUc2Setup::from_json(
            r#"{"format": "ocanvas", "version": "1.0.0", "catalog": {"grid_profile": "g"}}"#,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("not an openUC2 setup"), "{err}");
    }

    #[test]
    fn a_future_major_version_is_rejected() {
        let mut s = sample();
        s.version = "2.0.0".to_string();
        let err = OpenUc2Setup::from_json(&s.to_json()).unwrap_err();
        assert!(format!("{err}").contains("newer than this build"), "{err}");
    }

    #[test]
    fn a_future_minor_version_still_loads() {
        // View state is additive, so a 1.x document written by a newer build must
        // still open; unknown fields are simply ignored.
        let mut s = sample();
        s.version = "1.9.0".to_string();
        assert!(OpenUc2Setup::from_json(&s.to_json()).is_ok());
    }

    #[test]
    fn duplicate_instance_ids_are_rejected() {
        let mut s = sample();
        let dup = s.instances[0].clone();
        s.instances.push(dup);
        let err = OpenUc2Setup::from_json(&s.to_json()).unwrap_err();
        assert!(format!("{err}").contains("twice"), "{err}");
    }

    #[test]
    fn missing_optional_sections_default() {
        // The smallest document a hand-editor might write.
        let s = OpenUc2Setup::from_json(
            r#"{"format": "openuc2-canvas-setup", "version": "1.0.0",
                "catalog": {"grid_profile": "openuc2-core-50"}}"#,
        )
        .unwrap();
        assert!(s.instances.is_empty());
        assert_eq!(s.view.active_layer, 0);
        assert_eq!(s.view.trace_plane, "XY");
        assert!(s.view.show_optical_geometry);
    }

    #[test]
    fn instance_ids_do_not_collide() {
        let mut s = sample();
        assert_eq!(s.next_instance_id(), "m-0002");
        s.instances[0].instance_id = "m-0002".to_string();
        assert_eq!(s.next_instance_id(), "m-0001");
    }
}
