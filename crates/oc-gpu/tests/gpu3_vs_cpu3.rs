//! The 3D WGSL shader against the 3D CPU f32 mirror.
//!
//! **Scaffolding.** Empty until PR 3D-08 fills it in; gated so that
//! `cargo test -p oc-gpu --features gpu --test gpu3_vs_cpu3` is a stable
//! command from today and reports zero tests until then.
//!
//! The CPU f32 path is the oracle here, not the f64 reference: this test
//! isolates *shader* disagreement from *precision* disagreement, which
//! `cpu3_vs_f64` covers separately. Comparing the shader straight to f64 would
//! blur the two and make a real WGSL bug look like rounding.
//!
//! What this suite owes (contract §15.5, §12.2):
//!
//! - GPU vs CPU f32 ray landing within `max(1e-2 mm, 1e-5 * scene_scale)`;
//! - Rust and WGSL agree on buffer size and alignment, checked at compile time,
//!   and on contents, checked at runtime;
//! - branch type, TIR flags, detector cell, and termination reason agree;
//! - the same capability classification on both backends.
#![cfg(feature = "gpu3")]
