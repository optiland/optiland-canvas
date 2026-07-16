//! `Scene3` schema, validation, and `.ocanvas` v1 → v2 migration.
//!
//! **Scaffolding.** Empty until PR 3D-02 fills it in; gated so that
//! `cargo test -p oc-core --test scene3` is a stable command from today and
//! reports zero tests until then.
//!
//! What this suite owes (contract §11.5, §15.7):
//!
//! - every v1 corpus file loads and re-emits as stable v2;
//! - v1 → v2 migration is deterministic byte-for-byte after normalization;
//! - v2 → v2 preserves all f64 values exactly through `serde_json`;
//! - a future `version` is rejected, not mis-parsed;
//! - malformed quaternions, non-finite geometry, invalid profiles, and
//!   duplicate ids are rejected;
//! - v2 legacy-planar objects retain identical v1 planar ray results;
//! - **no third dimension is invented**: a migrated v1 body is `PlanarOnly`,
//!   never silently revolved, extruded, or given a thickness;
//! - stable ids survive insertion and removal of unrelated objects (§3.8).
#![cfg(feature = "scene3")]
