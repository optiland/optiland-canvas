//! The 3D trace result: segments, detector readouts, the energy ledger, and the
//! capability report (§9.8, §9.9, §17).
//!
//! Two rules shape these types. First, **every launched photon is accounted
//! for**: the [`EnergyLedger`] balances launched flux against everything that
//! could remove it, and flux killed by a threshold or a cap lands in
//! `truncated_flux` rather than vanishing. Second, **fidelity is reported, never
//! faked**: a scene the tracer cannot handle exactly produces a
//! [`CapabilityReport`] with a typed [`CapabilityIssue`], not a quietly
//! substituted geometry (§3.5).

use oc_core::id::{
    ApertureId, BodyId, DetectorId, ObjectId, PatchId, PathId, SourceId, SurfaceRef,
};
use oc_core::vec::Vec2;
use oc_core::vec3::{Ray3, Vec3};

/// One photon bundle launched into the scene.
///
/// The tracer takes launch rays explicitly; turning a [`Source3`](oc_core::scene3::Source3)
/// into a deterministic set of these is PR 3D-06's job, so the transport engine
/// can be validated on hand-built rays without a sampler in the way.
#[derive(Debug, Clone, Copy)]
pub struct LaunchRay3 {
    pub ray: Ray3,
    pub lambda_um: f64,
    pub flux: f64,
    /// The source this ray came from, for diagnostics. Not required for transport.
    pub source: Option<SourceId>,
}

/// Flags on a traced segment, for rendering and diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentFlags {
    /// A Fresnel back-reflection (or its descendants): the non-primary path.
    pub ghost: bool,
    /// Born of total internal reflection.
    pub tir: bool,
}

/// A traced ray segment (§9.9). `opl0`/`opl1` are the cumulative optical path
/// length in mm at the segment's ends, the propagation clock a wavefront view
/// reads.
#[derive(Debug, Clone, Copy)]
pub struct Segment3 {
    pub a: Vec3,
    pub b: Vec3,
    pub flux_start: f64,
    pub flux_end: f64,
    pub lambda_um: f64,
    pub opl0_mm: f64,
    pub opl1_mm: f64,
    pub path_id: PathId,
    pub parent_path_id: Option<PathId>,
    pub flags: SegmentFlags,
}

/// One recorded detector hit (§9.7).
#[derive(Debug, Clone, Copy)]
pub struct DetectorHit3 {
    pub point_world: Vec3,
    /// Position on the detector face, local `(u, v)` in mm.
    pub uv_mm: Vec2,
    /// Incident direction in the detector's local frame.
    pub incident_dir_local: Vec3,
    pub lambda_um: f64,
    pub incident_flux: f64,
    /// Flux after the detector's quantum efficiency.
    pub signal: f64,
    pub opl_mm: f64,
    pub path_id: PathId,
    pub parent_path_id: Option<PathId>,
    pub depth: u32,
    pub ghost_depth: u32,
}

/// An area detector's accumulated readout (§6.9, §9.7).
///
/// Flux density is `incident / area`, and the area is a real `mm²` — the thing a
/// line probe never had, and the reason [`DetectorGeometry3::LineProbeLegacy`](oc_core::scene3::DetectorGeometry3)
/// reports no density (§3.2).
#[derive(Debug, Clone)]
pub struct DetectorResult3 {
    pub detector: DetectorId,
    pub bins: [usize; 2],
    /// Area of one cell in `mm²`, or `None` for a line probe with no area.
    pub cell_area_mm2: Option<f64>,
    /// Incident optical flux per cell, row-major over `bins`.
    pub incident_flux: Vec<f64>,
    /// Detected signal per cell, after quantum efficiency.
    pub detected_signal: Vec<f64>,
    pub hits: u64,
    pub total_incident: f64,
    pub total_signal: f64,
    /// Flux-weighted sum of `(u, v)`; divide by `total_incident` for the centroid.
    pub flux_uv: Vec2,
    /// Flux-weighted sum of optical path length.
    pub flux_opl: f64,
    /// Bounded per-hit records, if requested.
    pub records: Vec<DetectorHit3>,
}

impl DetectorResult3 {
    pub fn new(detector: DetectorId, bins: [usize; 2], cell_area_mm2: Option<f64>) -> Self {
        let cells = (bins[0].max(1)) * (bins[1].max(1));
        Self {
            detector,
            bins,
            cell_area_mm2,
            incident_flux: vec![0.0; cells],
            detected_signal: vec![0.0; cells],
            hits: 0,
            total_incident: 0.0,
            total_signal: 0.0,
            flux_uv: Vec2::ZERO,
            flux_opl: 0.0,
            records: Vec::new(),
        }
    }

    /// Flux-weighted centroid in local `(u, v)` mm, or `None` with no flux.
    pub fn centroid_uv(&self) -> Option<Vec2> {
        (self.total_incident > 0.0).then(|| self.flux_uv / self.total_incident)
    }

    /// Flux-weighted mean optical path length, or `None` with no flux.
    pub fn mean_opl_mm(&self) -> Option<f64> {
        (self.total_incident > 0.0).then(|| self.flux_opl / self.total_incident)
    }

    /// Relative flux density per cell (`flux / area`), or `None` for a line probe.
    pub fn flux_density(&self) -> Option<Vec<f64>> {
        let area = self.cell_area_mm2?;
        Some(self.incident_flux.iter().map(|f| f / area).collect())
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn merge(&mut self, other: &DetectorResult3) {
        for (a, b) in self.incident_flux.iter_mut().zip(&other.incident_flux) {
            *a += *b;
        }
        for (a, b) in self.detected_signal.iter_mut().zip(&other.detected_signal) {
            *a += *b;
        }
        self.hits += other.hits;
        self.total_incident += other.total_incident;
        self.total_signal += other.total_signal;
        self.flux_uv += other.flux_uv;
        self.flux_opl += other.flux_opl;
        self.records.extend_from_slice(&other.records);
    }
}

/// Where a branch's flux ended up (§9.8). For an uncapped deterministic scene,
/// `launched` balances the rest to `1e-10` relative on analytic fixtures.
///
/// The invariant is not "energy is conserved" — a scene absorbs, a detector
/// swallows, an aperture blocks. It is that *no flux leaves the accounting*: what
/// a threshold or a cap removes is named in `truncated_flux`, so a ledger that
/// balances is a real statement and not an artefact of quietly dropped light.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EnergyLedger {
    pub launched: f64,
    pub detected_incident: f64,
    pub detected_signal: f64,
    pub escaped: f64,
    pub absorbed_volume: f64,
    pub absorbed_surface: f64,
    pub blocked_aperture: f64,
    pub truncated_flux: f64,
    pub numerical_loss: f64,
}

impl EnergyLedger {
    /// Everything the ledger accounts for besides what was launched. A
    /// terminating detector's incident flux is a sink like any other.
    pub fn accounted(&self) -> f64 {
        self.detected_incident
            + self.escaped
            + self.absorbed_volume
            + self.absorbed_surface
            + self.blocked_aperture
            + self.truncated_flux
            + self.numerical_loss
    }

    /// Relative imbalance `|launched − accounted| / launched`, or the absolute
    /// residual when nothing was launched.
    pub fn residual(&self) -> f64 {
        let r = (self.launched - self.accounted()).abs();
        if self.launched > 0.0 {
            r / self.launched
        } else {
            r
        }
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn merge(&mut self, o: &EnergyLedger) {
        self.launched += o.launched;
        self.detected_incident += o.detected_incident;
        self.detected_signal += o.detected_signal;
        self.escaped += o.escaped;
        self.absorbed_volume += o.absorbed_volume;
        self.absorbed_surface += o.absorbed_surface;
        self.blocked_aperture += o.blocked_aperture;
        self.truncated_flux += o.truncated_flux;
        self.numerical_loss += o.numerical_loss;
    }
}

/// Why a branch stopped (§9.8). Every branch ends with exactly one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerminationReason {
    Detector,
    AbsorbedVolume,
    AbsorbedSurface,
    ApertureBlocked,
    EscapedBounds,
    FluxThreshold,
    MaxDepth,
    MaxGhostDepth,
    MaxBranches,
    NumericalFailure,
    UnsupportedPhysics,
}

/// A count of branch terminations by reason.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerminationSummary {
    pub detector: u64,
    pub absorbed_volume: u64,
    pub absorbed_surface: u64,
    pub aperture_blocked: u64,
    pub escaped_bounds: u64,
    pub flux_threshold: u64,
    pub max_depth: u64,
    pub max_ghost_depth: u64,
    pub max_branches: u64,
    pub numerical_failure: u64,
    pub unsupported_physics: u64,
}

impl TerminationSummary {
    pub(crate) fn record(&mut self, reason: TerminationReason) {
        use TerminationReason::*;
        match reason {
            Detector => self.detector += 1,
            AbsorbedVolume => self.absorbed_volume += 1,
            AbsorbedSurface => self.absorbed_surface += 1,
            ApertureBlocked => self.aperture_blocked += 1,
            EscapedBounds => self.escaped_bounds += 1,
            FluxThreshold => self.flux_threshold += 1,
            MaxDepth => self.max_depth += 1,
            MaxGhostDepth => self.max_ghost_depth += 1,
            MaxBranches => self.max_branches += 1,
            NumericalFailure => self.numerical_failure += 1,
            UnsupportedPhysics => self.unsupported_physics += 1,
        }
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn merge(&mut self, o: &TerminationSummary) {
        self.detector += o.detector;
        self.absorbed_volume += o.absorbed_volume;
        self.absorbed_surface += o.absorbed_surface;
        self.aperture_blocked += o.aperture_blocked;
        self.escaped_bounds += o.escaped_bounds;
        self.flux_threshold += o.flux_threshold;
        self.max_depth += o.max_depth;
        self.max_ghost_depth += o.max_ghost_depth;
        self.max_branches += o.max_branches;
        self.numerical_failure += o.numerical_failure;
        self.unsupported_physics += o.unsupported_physics;
    }
}

/// A physics feature a backend can run (§3.5). Coarse on purpose: the report
/// exists to say what was and was not handled, not to enumerate every constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsFeature {
    AnalyticSolid,
    ProfileBody,
    ZeroThicknessSurface,
    Aperture,
    AreaDetector,
    DielectricFresnel,
    Mirror,
    ExplicitRt,
    ThinLens,
    VolumeAbsorption,
    SpectralResponse,
}

/// A feature that ran, but not exactly (§3.5). None are emitted yet — the
/// reference path is exact or it refuses — but the field exists so a later
/// approximate backend has somewhere honest to record itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Approximation {
    pub feature: PhysicsFeature,
    pub detail: String,
}

/// A scene, or part of one, a backend cannot handle exactly (§17).
///
/// The tracer emits these instead of substituting a superficially similar
/// geometry or surface. A backend that cannot trace a required feature rejects
/// the scene; it never traces it wrong and stays quiet.
#[derive(Debug, Clone, PartialEq)]
pub enum CapabilityIssue {
    /// A profile body whose third dimension is unresolved (§3.2).
    PlanarOnlyObject {
        object: ObjectId,
    },
    UnsupportedGeometry {
        object: ObjectId,
        kind: String,
    },
    UnsupportedSurfaceResponse {
        surface: SurfaceRef,
    },
    UnsupportedSourceDistribution {
        source: SourceId,
    },
    InvalidClosedBody {
        body: BodyId,
        reason: String,
    },
    /// Coincident physical interfaces the event policy cannot order (§9.3).
    AmbiguousEvent {
        point: Vec3,
        objects: Vec<ObjectId>,
    },
    /// A spectral or absorption table asked for a wavelength it does not cover,
    /// under the error policy (§6.7).
    WavelengthOutOfRange {
        object: ObjectId,
        lambda_um: f64,
    },
    NonFiniteInput {
        object: Option<ObjectId>,
        field: String,
    },
    NumericalNonConvergence {
        object: ObjectId,
        patch: PatchId,
    },
    BlockedByStop {
        aperture: ApertureId,
    },
}

/// A backend's structured statement of what it can and cannot do for a scene
/// (§3.5). For the reference tracer, `rejected` is the load-bearing field: a
/// non-empty `rejected` means some flux could not be traced honestly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CapabilityReport {
    pub supported: Vec<PhysicsFeature>,
    pub exact: Vec<PhysicsFeature>,
    pub approximated: Vec<Approximation>,
    pub rejected: Vec<CapabilityIssue>,
}

impl CapabilityReport {
    /// Did every required feature trace exactly?
    pub fn is_clean(&self) -> bool {
        self.rejected.is_empty() && self.approximated.is_empty()
    }

    pub(crate) fn note_exact(&mut self, f: PhysicsFeature) {
        if !self.exact.contains(&f) {
            self.exact.push(f);
        }
        if !self.supported.contains(&f) {
            self.supported.push(f);
        }
    }

    pub(crate) fn reject(&mut self, issue: CapabilityIssue) {
        if !self.rejected.contains(&issue) {
            self.rejected.push(issue);
        }
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn merge(&mut self, o: &CapabilityReport) {
        for f in &o.exact {
            self.note_exact(*f);
        }
        for f in &o.supported {
            if !self.supported.contains(f) {
                self.supported.push(*f);
            }
        }
        for a in &o.approximated {
            if !self.approximated.contains(a) {
                self.approximated.push(a.clone());
            }
        }
        for i in &o.rejected {
            self.reject(i.clone());
        }
    }
}

/// The result of a full 3D trace (§9.9).
#[derive(Debug, Clone)]
pub struct TraceResult3 {
    /// The scene revision that produced these numbers, so a stale result can be
    /// discarded rather than shown (§3.3). The tracer copies it from the config;
    /// the caller that owns the scene sets it.
    pub revision: u64,
    pub segments: Vec<Segment3>,
    pub detectors: Vec<DetectorResult3>,
    pub ledger: EnergyLedger,
    pub terminations: TerminationSummary,
    pub capability_report: CapabilityReport,
}
