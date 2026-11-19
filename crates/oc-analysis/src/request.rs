//! What an analysis is asked to do: the request envelope.
//!
//! Every request has the same outer shape, and only `params` depends on the
//! kind:
//!
//! ```json
//! { "kind": "detectorStats", "rays": 200000, "passes": 16,
//!   "params": { "detectors": [7] } }
//! ```
//!
//! - `kind` names the analysis ([`KINDS`]).
//! - `rays` is how many launch rays to trace in all, optional. Without it each
//!   pass traces the rays the scene's sources emit.
//! - `passes` is how many independent passes the rays are traced in, optional
//!   ([`DEFAULT_PASSES`]). The spread between passes is where every standard
//!   error comes from, so there are at least two.
//! - `params` holds what the kind needs, optional when the kind needs nothing.
//!
//! Unknown fields are refused rather than ignored, at both levels: a request
//! that asks for something this build does not do is told so.

use serde::Deserialize;

/// The analyses this build runs, by the name a request gives.
pub const KINDS: [&str; 5] = ["detectorStats", "channels", "uniformity", "budget", "fluorescence"];

/// Passes when a request names none. Sixteen passes put the standard error
/// itself within about a fifth of its true value.
pub const DEFAULT_PASSES: u32 = 16;

/// The most launch rays one pass holds, as for the reference trace: a job
/// keeps one pass of launch rays in memory.
pub const MAX_PASS_RAYS: u64 = 524_288;

/// A request, parsed and checked against the kinds this build knows.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisRequest {
    pub kind: Kind,
    /// Launch rays in all, if the request named a count.
    pub rays: Option<u64>,
    pub passes: u32,
}

/// An analysis and its parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    DetectorStats(DetectorStatsParams),
    Channels(ChannelsParams),
    Uniformity(UniformityParams),
    Budget(BudgetParams),
    Fluorescence(FluorescenceParams),
}

impl Kind {
    /// The name a request and a result give the kind.
    pub fn name(&self) -> &'static str {
        match self {
            Kind::DetectorStats(_) => "detectorStats",
            Kind::Channels(_) => "channels",
            Kind::Uniformity(_) => "uniformity",
            Kind::Budget(_) => "budget",
            Kind::Fluorescence(_) => "fluorescence",
        }
    }
}

/// `detectorStats`: which detectors to read, by id; every detector when
/// `None`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorStatsParams {
    #[serde(default)]
    pub detectors: Option<Vec<u64>>,
}

/// `channels`: the channels to read; one per detector, seeing every source,
/// when `None`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelsParams {
    #[serde(default)]
    pub channels: Option<Vec<ChannelSpec>>,
}

/// One channel: a detector, the sources it is meant to see, and the band of
/// wavelengths it counts.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChannelSpec {
    /// The channel's name; the detector's label (or `detector <id>`) when
    /// absent. Names are unique within a request.
    #[serde(default)]
    pub name: Option<String>,
    /// The detector the channel reads, by id.
    pub detector: u64,
    /// The sources whose light is the channel's signal, by id; every source
    /// when absent. The light of any other source is leak.
    #[serde(default)]
    pub sources: Option<Vec<u64>>,
    /// `[low, high]` in µm: only light detected at these wavelengths counts,
    /// both ends included. Every wavelength when absent.
    #[serde(default)]
    pub band_um: Option<[f64; 2]>,
}

/// `uniformity`: how even the light is over a region of one detector, read
/// on a grid of cells.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct UniformityParams {
    /// The detector read, by id.
    pub detector: u64,
    /// Cells across the region, `[along u, along v]`.
    pub cells: [u64; 2],
    /// `[[u low, u high], [v low, v high]]` in mm, in the detector's local
    /// frame; the face's whole grid when absent.
    #[serde(default)]
    pub region_mm: Option<[[f64; 2]; 2]>,
    /// Stop once this metric's standard error is this small; trace every ray
    /// when absent.
    #[serde(default)]
    pub target: Option<UniformityTarget>,
    /// `"forward"` (the default) traces the sources as they are;
    /// `"focused"` traces a pilot first and spends most of the rays on the
    /// part of each source's domain that reaches the region.
    #[serde(default)]
    pub sampling: Option<String>,
    /// Launch rays the pilot of a `"focused"` job traces (optional).
    #[serde(default)]
    pub pilot_rays: Option<u64>,
}

/// The noise a `uniformity` job traces down to.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UniformityTarget {
    /// `"cv"` or `"minMax"`.
    pub metric: String,
    /// The standard error to reach, in the metric's own unit (a fraction).
    pub stderr: f64,
}

/// `budget`: where the light went.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct BudgetParams {
    /// Trace each source alone, and give each its own budget beside the
    /// scene's. False when absent.
    #[serde(default)]
    pub by_source: bool,
}

/// `fluorescence`: the dyes in the scene, and how the job splits its work
/// between exciting them and tracing what they emit.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct FluorescenceParams {
    /// The dyes, at least one; each a body of the scene.
    pub dyes: Vec<DyeSpec>,
    /// The grid each dye's absorbed light is binned on before it is
    /// re-emitted: one count for every axis, or `[x, y, z]`, over the body's
    /// world bounding box. [`crate::fluorescence::DEFAULT_VOXELS`] a side when
    /// absent.
    #[serde(default)]
    pub voxels: Option<Voxels>,
    /// The share of `rays` the emission traces take, the rest exciting;
    /// one half when absent. Strictly between 0 and 1.
    #[serde(default)]
    pub emission_share: Option<f64>,
}

/// A voxel grid: the same count along every axis, or one per axis.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Voxels {
    Cube(u32),
    Grid([u32; 3]),
}

impl Voxels {
    /// `[x, y, z]`.
    pub fn counts(self) -> [u32; 3] {
        match self {
            Voxels::Cube(n) => [n; 3],
            Voxels::Grid(g) => g,
        }
    }
}

/// One dye: the body it fills, how much of what it absorbs it emits, and at
/// which wavelengths. What it absorbs is the body's own absorption model,
/// `ln 10 · ε(λ) · c` per mm.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DyeSpec {
    /// The body, by object id.
    pub body: u64,
    /// Watts emitted per watt absorbed, in (0, 1].
    pub quantum_yield: f64,
    /// The emission spectrum.
    pub emission: EmissionSpectrum,
    /// The dye's name, unique in the request; the body's label, or
    /// `body <id>`, when absent.
    #[serde(default)]
    pub name: Option<String>,
}

/// A dye's emission spectrum: relative power at each wavelength (µm).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct EmissionSpectrum {
    pub lambda_um: Vec<f64>,
    pub weight: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    kind: String,
    #[serde(default)]
    rays: Option<f64>,
    #[serde(default)]
    passes: Option<f64>,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

impl AnalysisRequest {
    /// Parse a request from its JSON. The error says what is wrong with it.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let e: Envelope =
            serde_json::from_str(text).map_err(|e| format!("not an analysis request: {e}"))?;
        let rays = match e.rays {
            None => None,
            Some(r) if r.is_finite() && r >= 1.0 && r.fract() == 0.0 => Some(r as u64),
            Some(r) => return Err(format!("rays must be a whole number from 1 up, not {r}")),
        };
        let passes = match e.passes {
            None => DEFAULT_PASSES,
            Some(p) if p.is_finite() && p >= 2.0 && p.fract() == 0.0 && p <= 1024.0 => p as u32,
            Some(p) => {
                return Err(format!(
                    "passes must be a whole number from 2 up, not {p}: the standard errors come from the spread between passes"
                ))
            }
        };
        let params = e.params.unwrap_or(serde_json::Value::Object(Default::default()));
        let kind = match e.kind.as_str() {
            "detectorStats" => Kind::DetectorStats(
                serde_json::from_value(params)
                    .map_err(|err| format!("detectorStats params: {err}"))?,
            ),
            "channels" => Kind::Channels(
                serde_json::from_value(params).map_err(|err| format!("channels params: {err}"))?,
            ),
            "uniformity" => Kind::Uniformity(
                serde_json::from_value(params)
                    .map_err(|err| format!("uniformity params: {err}"))?,
            ),
            "budget" => Kind::Budget(
                serde_json::from_value(params).map_err(|err| format!("budget params: {err}"))?,
            ),
            "fluorescence" => Kind::Fluorescence(
                serde_json::from_value(params)
                    .map_err(|err| format!("fluorescence params: {err}"))?,
            ),
            other => {
                return Err(format!(
                    "unknown analysis kind {other:?}; this build runs {}",
                    KINDS.join(", ")
                ))
            }
        };
        Ok(Self { kind, rays, passes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_request_takes_the_defaults() {
        let r = AnalysisRequest::from_json(r#"{"kind":"detectorStats"}"#).unwrap();
        assert_eq!(r.kind, Kind::DetectorStats(DetectorStatsParams::default()));
        assert_eq!((r.rays, r.passes), (None, DEFAULT_PASSES));
    }

    #[test]
    fn a_channels_request_reads_its_channels() {
        let r = AnalysisRequest::from_json(
            r#"{"kind":"channels","rays":9000,"passes":8,"params":{"channels":[
                {"name":"EGFP","detector":6,"sources":[2],"bandUm":[0.5,0.55]},{"detector":7}]}}"#,
        )
        .unwrap();
        let Kind::Channels(p) = r.kind else { panic!("{:?}", r.kind) };
        let c = p.channels.unwrap();
        assert_eq!(c[0].name.as_deref(), Some("EGFP"));
        assert_eq!((c[0].detector, c[0].sources.as_deref(), c[0].band_um), (6, Some(&[2][..]), Some([0.5, 0.55])));
        assert_eq!((c[1].name.as_deref(), c[1].sources.as_deref(), c[1].band_um), (None, None, None));
        let bare = AnalysisRequest::from_json(r#"{"kind":"channels"}"#).unwrap();
        assert_eq!(bare.kind, Kind::Channels(ChannelsParams::default()));
    }

    #[test]
    fn a_uniformity_request_reads_its_region_and_target() {
        let r = AnalysisRequest::from_json(
            r#"{"kind":"uniformity","rays":64000,"passes":64,"params":{"detector":6,"cells":[20,10],
                "regionMm":[[-5,5],[-2.5,2.5]],"target":{"metric":"cv","stderr":0.005}}}"#,
        )
        .unwrap();
        let Kind::Uniformity(p) = r.kind else { panic!("{:?}", r.kind) };
        assert_eq!((p.detector, p.cells), (6, [20, 10]));
        assert_eq!(p.region_mm, Some([[-5.0, 5.0], [-2.5, 2.5]]));
        let t = p.target.unwrap();
        assert_eq!((t.metric.as_str(), t.stderr), ("cv", 0.005));
        assert_eq!((p.sampling, p.pilot_rays), (None, None));
        let r = AnalysisRequest::from_json(
            r#"{"kind":"uniformity","params":{"detector":6,"cells":[2,2],"sampling":"focused","pilotRays":8192}}"#,
        )
        .unwrap();
        let Kind::Uniformity(p) = r.kind else { panic!("{:?}", r.kind) };
        assert_eq!((p.sampling.as_deref(), p.pilot_rays), (Some("focused"), Some(8192)));
    }

    #[test]
    fn a_budget_request_reads_by_source() {
        let r = AnalysisRequest::from_json(r#"{"kind":"budget","params":{"bySource":true}}"#).unwrap();
        assert_eq!(r.kind, Kind::Budget(BudgetParams { by_source: true }));
        let bare = AnalysisRequest::from_json(r#"{"kind":"budget"}"#).unwrap();
        assert_eq!(bare.kind, Kind::Budget(BudgetParams::default()));
    }

    #[test]
    fn a_fluorescence_request_reads_its_dyes() {
        let r = AnalysisRequest::from_json(
            r#"{"kind":"fluorescence","rays":64000,"passes":16,"params":{"dyes":[
                {"body":5,"quantumYield":0.9,"emission":{"lambdaUm":[0.5,0.51],"weight":[1,2]},"name":"FITC"}],
                "voxels":[8,8,2],"emissionShare":0.25}}"#,
        )
        .unwrap();
        let Kind::Fluorescence(p) = r.kind else { panic!("{:?}", r.kind) };
        assert_eq!((p.dyes[0].body, p.dyes[0].quantum_yield, p.dyes[0].name.as_deref()), (5, 0.9, Some("FITC")));
        assert_eq!(p.dyes[0].emission.weight, vec![1.0, 2.0]);
        assert_eq!(p.voxels.map(Voxels::counts), Some([8, 8, 2]));
        assert_eq!(p.emission_share, Some(0.25));
        let r = AnalysisRequest::from_json(
            r#"{"kind":"fluorescence","params":{"dyes":[{"body":5,"quantumYield":1,"emission":{"lambdaUm":[0.5],"weight":[1]}}],"voxels":4}}"#,
        )
        .unwrap();
        let Kind::Fluorescence(p) = r.kind else { panic!("{:?}", r.kind) };
        assert_eq!((p.voxels.map(Voxels::counts), p.emission_share, p.dyes[0].name.as_deref()), (Some([4, 4, 4]), None, None));
    }

    #[test]
    fn a_bad_request_is_refused_with_its_reason() {
        let bad = |s: &str| AnalysisRequest::from_json(s).unwrap_err();
        assert!(bad(r#"{"kind":"psf"}"#).contains("detectorStats"));
        assert!(bad(r#"{"kind":"detectorStats","passes":1}"#).contains("spread"));
        assert!(bad(r#"{"kind":"detectorStats","rays":2.5}"#).contains("whole"));
        assert!(bad(r#"{"kind":"detectorStats","ray":10}"#).contains("unknown field"));
        assert!(bad(r#"{"kind":"detectorStats","params":{"detector":[1]}}"#).contains("unknown field"));
        assert!(bad("[]").contains("not an analysis request"));
        assert!(bad(r#"{"kind":"channels","params":{"channels":[{"sources":[1]}]}}"#).contains("detector"));
        assert!(bad(r#"{"kind":"channels","params":{"channels":[{"detector":1,"band":[0.4,0.5]}]}}"#).contains("unknown field"));
        assert!(bad(r#"{"kind":"channels","params":{"channels":[{"detector":1,"bandUm":[0.4]}]}}"#).contains("channels params"));
        assert!(bad(r#"{"kind":"uniformity","params":{"detector":1}}"#).contains("cells"));
        assert!(bad(r#"{"kind":"uniformity"}"#).contains("uniformity params"));
        assert!(bad(r#"{"kind":"uniformity","params":{"detector":1,"cells":[4,4],"region":[[0,1],[0,1]]}}"#).contains("unknown field"));
        assert!(bad(r#"{"kind":"budget","params":{"perSource":true}}"#).contains("unknown field"));
        assert!(bad(r#"{"kind":"budget","params":{"bySource":1}}"#).contains("budget params"));
        assert!(bad(r#"{"kind":"fluorescence"}"#).contains("fluorescence params"));
        assert!(bad(r#"{"kind":"fluorescence","params":{"dyes":[{"body":1,"qy":1}]}}"#).contains("unknown field"));
        assert!(bad(r#"{"kind":"fluorescence","params":{"dyes":[],"voxels":[1,2]}}"#).contains("fluorescence params"));
    }
}
