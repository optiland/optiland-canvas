//! # oc-trace
//!
//! The **reference tracer**: CPU, f64, exact — the source of truth against which
//! the (future) interactive f32/GPU path is cross-validated (§4.1). It is
//! non-sequential: rays split at every interface (transmit + reflect), media are
//! resolved by containment + priority rather than a stack, and ghosts fall out
//! for free (§4.2).

pub mod bvh;
pub mod fresnel;
pub mod tracer;

pub use fresnel::{beer_lambert, critical_angle, reflect, reflectance, refract};
pub use tracer::{trace, DetectorResult, Segment, TraceConfig, TraceResult};
