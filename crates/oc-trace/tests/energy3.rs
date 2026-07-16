//! The energy ledger: launched flux is accounted for, never lost.
//!
//! **Scaffolding.** Empty until PR 3D-04 fills it in; gated so that
//! `cargo test -p oc-trace --test energy3` is a stable command from today and
//! reports zero tests until then.
//!
//! What this suite owes (contract §9.8, §15.2):
//!
//! - for an uncapped deterministic scene,
//!   `launched ≈ escaped + absorbed_volume + absorbed_surface +
//!   blocked_aperture + terminating_detector_flux` to 1e-10 relative on
//!   analytic fixtures;
//! - flux removed by a threshold or a branch cap appears under
//!   `truncated_flux`. It does not quietly disappear, because a ledger that
//!   only balances when nothing is truncated is not a ledger;
//! - `1 - R - T` lands in `absorbed_surface`;
//! - Beer-Lambert attenuation lands in `absorbed_volume`;
//! - every branch ends with an explicit `TerminationReason`;
//! - energy never increases without a source.
#![cfg(feature = "trace3")]
