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

#[cfg(feature = "trace3")]
pub mod bvh3;
#[cfg(feature = "trace3")]
pub mod result3;
#[cfg(feature = "trace3")]
pub mod tracer3;

pub use fresnel::{
    beer_lambert, critical_angle, reflect, reflect3, reflectance, refract, refract3,
};
pub use tracer::{trace, DetectorResult, Segment, TraceConfig, TraceResult};

#[cfg(feature = "trace3")]
pub use result3::{
    Approximation, CapabilityIssue, CapabilityReport, DetectorHit3, DetectorResult3, EnergyLedger,
    LaunchRay3, PhysicsFeature, Segment3, SegmentFlags, TerminationReason, TerminationSummary,
    TraceResult3,
};
#[cfg(feature = "trace3")]
pub use tracer3::{launch_from_scene, launch_from_source, trace3, trace_scene3, TraceConfig3};
