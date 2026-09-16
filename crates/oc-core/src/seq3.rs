//! The 3D sequence prescription: the ordered surface chain one source's axial
//! ray actually threads on its way to one detector.
//!
//! The 3D sibling of [`crate::seq::SeqSystem`], and the same contract stance:
//! a **neutral, engine-free description** that downstream tools (the optikit
//! emitter, an Optiland exporter, a diagnostic view) consume without knowing
//! anything about the tracer that produced it. Where `SeqSystem` flattens the
//! planar canvas into a rotationally-symmetric prescription, this type does
//! *not* flatten anything: every hit carries its full world-space pose context
//! (point, normal) and the chain may fold, split, and pass through glass. What
//! the sequential idealization cannot express is *reported* in [`notes`] rather
//! than silently dropped — the `SeqSystem::notes` rule, verbatim.
//!
//! The prescription is **derived, never authored**: it is the output of tracing
//! a scene's axial ray (`oc-trace::sequence3`), the inversion that replaces
//! static beam-path bookkeeping with physics. Consumers treat it as a cache of
//! "where did the beam actually go" keyed by the scene revision that produced
//! it.
//!
//! [`notes`]: SequencePrescription3::notes

use serde::{Deserialize, Serialize};

use crate::id::{DetectorId, ObjectId, PatchId, PathIntentId, SourceId};
use crate::vec::Vec2;
use crate::vec3::Vec3;

/// Format tag of the serialized prescription.
pub const SEQ3_FORMAT: &str = "oc-seq3";
/// Version of the serialized prescription layout.
pub const SEQ3_VERSION: u32 = 1;

/// What the chain's continuing branch did at a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeqAction {
    /// Continued through the interface: Snell refraction across a real medium
    /// transition, or undeviated passage where both sides share a medium.
    Transmit,
    /// Specular reflection: a mirror, or the reflected arm of a splitter.
    Reflect,
    /// Total internal reflection (R = 1, the primary path).
    Tir,
    /// The ideal thin-lens bend of a standalone surface.
    ThinLens,
}

/// Which kind of scene object the hit surface belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeqHitKind {
    /// A patch of a closed body ([`crate::scene3::Body3`]).
    Body,
    /// A zero-thickness standalone surface ([`crate::scene3::SurfaceObject3`]).
    Surface,
}

/// One surface interaction along the chain, in beam order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeqHit3 {
    /// The object the surface belongs to. Stable scene identity, never an index.
    pub object: ObjectId,
    /// The patch within the object (`PatchId(0)` for standalone surfaces).
    pub patch: PatchId,
    pub kind: SeqHitKind,
    /// The object's label at extraction time. A human courtesy for notes and
    /// diagnostics; identity is `object`/`patch`.
    pub label: String,
    /// World-space hit point in mm.
    pub point: Vec3,
    /// Unit normal at the hit, oriented **against** the incident ray (the
    /// tracer's incident-opposing convention).
    pub normal: Vec3,
    pub action: SeqAction,
    /// True when the interaction spawned more than one followed branch (a real
    /// splitter, or Fresnel with ghosts enabled) — the chain took one arm and
    /// [`SequencePrescription3::notes`] names it.
    pub split: bool,
    /// Refractive index of the medium the ray arrived through, at the chain's
    /// wavelength.
    pub n_before: f64,
    /// Refractive index on the far side of the interface. Equal to `n_before`
    /// at a zero-thickness surface or a reflection.
    pub n_after: f64,
    /// Geometric distance from the previous chain point (the launch origin for
    /// the first hit) in mm. Media between events are homogeneous, so segments
    /// are straight and this is exact.
    pub arc_mm: f64,
    /// Cumulative geometric arc length from the launch origin, in mm.
    pub cum_arc_mm: f64,
    /// Cumulative optical path length at the hit, in mm.
    pub opl_mm: f64,
}

/// The chain's launch ray: the source's axial ray.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeqLaunch3 {
    /// World-space launch point in mm (the source frame's origin).
    pub point: Vec3,
    /// Unit launch direction (the source frame's `+X`, its optical axis).
    pub dir: Vec3,
    /// The wavelength the chain was traced at, in µm: the source's
    /// highest-weight spectral line.
    pub lambda_um: f64,
}

/// Where and how the chain ended on the target detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeqTerminal3 {
    pub detector: DetectorId,
    /// Hit position on the detector face, local `(u, v)` in mm.
    pub uv_mm: Vec2,
    /// World-space hit point in mm.
    pub point: Vec3,
    /// Geometric distance from the last hit (or the launch origin) in mm.
    pub arc_mm: f64,
    /// Total geometric arc length of the chain, launch to detector, in mm.
    pub cum_arc_mm: f64,
    /// Total optical path length, launch to detector, in mm.
    pub opl_mm: f64,
}

/// The ordered surface chain from one source's axial ray to one detector.
///
/// Serialized with [`Self::to_json`]; the wire format is versioned by
/// [`SEQ3_FORMAT`]/[`SEQ3_VERSION`] like the `.ocanvas` document itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequencePrescription3 {
    pub format: String,
    pub version: u32,
    /// The scene revision this prescription was extracted at. A consumer holding
    /// a newer scene discards the prescription rather than trusting it (§3.3's
    /// stale-result rule, applied to sequences).
    pub revision: u64,
    /// The intent this extraction served, when it was named by one.
    pub intent: Option<PathIntentId>,
    pub source: SourceId,
    pub detector: DetectorId,
    pub launch: SeqLaunch3,
    /// Surface interactions in beam order. Probe-detector passages are not
    /// hits; they are reported in [`notes`](Self::notes).
    pub hits: Vec<SeqHit3>,
    pub terminal: SeqTerminal3,
    /// Everything the chain encountered that the hit list cannot express:
    /// splitter arms not taken, probe passages, the ghost policy in force.
    /// The [`crate::seq::SeqSystem::notes`] rule: reported, never faked.
    pub notes: Vec<String>,
}

impl SequencePrescription3 {
    /// Serialize to the versioned JSON wire form.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("prescription serialization is infallible")
    }

    /// Parse a prescription back from JSON.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
}
