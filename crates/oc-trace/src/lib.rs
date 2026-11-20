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
pub mod sequence3;
#[cfg(feature = "trace3")]
pub mod seq_export;
#[cfg(feature = "trace3")]
pub mod tracer3;

pub use fresnel::{
    beer_lambert, critical_angle, reflect, reflect3, reflectance, refract, refract3,
};
pub use tracer::{trace, DetectorResult, Segment, TraceConfig, TraceResult};

#[cfg(feature = "trace3")]
pub use result3::{
    Approximation, CapabilityIssue, CapabilityReport, ChildRole, DetectorHit3, DetectorResult3,
    Deposit3, DepositLayer, EnergyLedger, LaunchRay3, ObjectEntry3, ObjectLedger3, ObjectRef3, PhysicsFeature, Segment3,
    SegmentFlags, SpectralSums3,
    TerminationReason, TerminationSummary, TraceEvent3, TraceEventKind, TraceResult3,
};
#[cfg(feature = "trace3")]
pub use sequence3::{axial_launch, extract_for_intent, extract_sequence3, extract_sequences3, extract_wave_sequences3, SequenceError};
#[cfg(feature = "trace3")]
pub use seq_export::{default_path, export_seq, ExportError};
#[cfg(feature = "trace3")]
pub use tracer3::{
    bounds_for_scene3, depth_for_scene3, launch_from_scene, launch_from_scene_pass, launch_from_source,
    launch_from_source_pass, trace3, trace_scene3, EncodedPart3,
    TraceBlocks3, TraceConfig3, TraceJob3, TraceMerge3, TracePart3, TRACE_BLOCKS,
};
