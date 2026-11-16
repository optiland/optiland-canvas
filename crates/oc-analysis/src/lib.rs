//! # oc-analysis
//!
//! Analyses read off the f64 reference trace ([ADR 0003]): numbers a designer
//! asks of a whole system, each run as a job a browser worker steps a batch of
//! launch rays at a time, and each result stating what it rests on.
//!
//! The kernel crate `oc-trace` stays the one transport engine. This crate
//! drives it through its public API and changes nothing about how a ray
//! travels: it chooses which rays to launch, reads the sums a trace keeps, and
//! turns them into figures.
//!
//! Every result carries, in one envelope ([`AnalysisJob::result`]):
//!
//! - the scene revision it was traced at, so a stale result is discarded;
//! - the fidelity stamp of the profile it was traced under;
//! - the tracer's approximations and rejections from the capability report;
//! - for each statistical number, its standard error ([`stats`]).
//!
//! The analyses this build runs are [`request::KINDS`].
//!
//! [ADR 0003]: ../../docs/architecture/adr/0003-analysis-crate.md

mod budget;
mod channels;
mod detector_stats;
mod focus;
pub mod job;
pub mod request;
pub mod stats;
mod uniformity;

pub use job::{launch_count, sampled_for, source_alone, AnalysisJob, Progress};
pub use request::{AnalysisRequest, BudgetParams, ChannelSpec, ChannelsParams, DetectorStatsParams, Kind, UniformityParams, UniformityTarget, DEFAULT_PASSES, KINDS, MAX_PASS_RAYS};
pub use stats::{standard_error, Estimate};
pub use uniformity::MAX_UNIFORMITY_CELLS;
