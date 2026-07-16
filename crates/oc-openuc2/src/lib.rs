//! # oc-openuc2
//!
//! The openUC2/OptiKit adapter: catalogue, setup, and module placement on top of
//! the generic Canvas core.
//!
//! This crate is the *only* place in the workspace that knows what an openUC2
//! cube, SKU, or grid cell is. `oc-core`, `oc-trace`, `oc-gpu`, and the
//! `.ocanvas` format stay domain-free (§5.2), and nothing here owns ray physics.
//!
//! # The three models
//!
//! | Model | Answers | Lives in |
//! |---|---|---|
//! | Module descriptor | what a reusable openUC2 object *is* | the module repository |
//! | Setup ([`OpenUc2Setup`]) | which objects are assembled, and where | `.uc2canvas` |
//! | Scene (`oc_core::Scene`) | what the light does | derived, regenerated |
//!
//! The scene is **derived**. It is materialized from the setup plus the
//! catalogue and never maintained in parallel with them, so the two cannot
//! disagree about where a lens is.
//!
//! # Getting started
//!
//! ```no_run
//! use oc_openuc2::{Catalog, GridProfile, ModuleRef, OpenUc2Project, Quadrant};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let (catalog_json, lens_ocanvas) = ("", "");
//! // The catalogue is a generated, pinned artifact: no live GitHub, ever (§5.4).
//! let mut catalog = Catalog::from_json(catalog_json)?;
//! catalog.attach_fragment("assets/lens.ocanvas", lens_ocanvas)?;
//!
//! let mut project = OpenUc2Project::new(catalog, GridProfile::openuc2_core_50())?;
//! let lens = ModuleRef::new("…uuid…", "1.0.0");
//! project.place_module(&lens, [1, 0, 0], Quadrant::R0)?;
//!
//! // The scene is derived; trace it with oc-trace as usual.
//! let _ = project.scene();
//! # Ok(())
//! # }
//! ```
//!
//! # What this crate will not do
//!
//! - **Substitute a module version.** A pin that no longer resolves becomes an
//!   unresolved placeholder, never a different version of the same product.
//! - **Trace more than one layer.** Canvas is 2D. Modules off the active layer
//!   are mechanical context and say so in [`warnings`](OpenUc2Project::warnings).
//! - **Claim fidelity it does not have.** A module with no optics is reported as
//!   layout-only rather than quietly contributing nothing.

pub mod catalog;
pub mod coordinates;
pub mod errors;
pub mod project;
pub mod setup;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

pub use catalog::{
    Assets, Capabilities, Catalog, CatalogDoc, CatalogSource, Commercial, Footprint, ModuleEntry,
    ModuleRef, Parameters, Port, PortPose, CATALOG_FORMAT,
};
pub use coordinates::{
    canvas_to_uc2, three_to_uc2, uc2_to_canvas, uc2_to_three, GridProfile, InvalidQuadrant,
    Quadrant, ThreePoint, Uc2Point,
};
pub use errors::{Error, PlacementError, ProjectWarning};
pub use project::{Bom, BomLine, FlatExport, MaterializedInstance, OpenUc2Project, SceneObjectId};
pub use setup::{
    CatalogPin, CustomCanvas, InstanceFallback, OpenUc2Setup, Placement, SetupInstance,
    SetupMetadata, ViewState, SETUP_FORMAT, SETUP_VERSION,
};
