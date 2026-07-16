//! The 3D CPU f32 preview against the f64 reference.
//!
//! **Scaffolding.** Empty until PR 3D-07 fills it in; gated so that
//! `cargo test -p oc-gpu --test cpu3_vs_f64` is a stable command from today and
//! reports zero tests until then.
//!
//! The 2D pair of this test (`cross_validate.rs`) is described in the README as
//! the project's single most important correctness guard. This is its 3D
//! successor and inherits that standing.
//!
//! What this suite owes (contract §15.5):
//!
//! - ray landing within `max(5e-2 mm, 5e-5 * scene_scale)` of f64;
//! - branch type, TIR flags, detector cell, and termination reason agree,
//!   except within a documented boundary tolerance;
//! - an unsupported primitive is **rejected with a capability issue**, not
//!   approximated. No arc becomes a chord and no asphere becomes its base
//!   conic (§12.3);
//! - CPU and GPU classify the same unsupported primitive the same way.
#![cfg(feature = "gpu3")]
