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
pub const KINDS: [&str; 2] = ["detectorStats", "channels"];

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
}

impl Kind {
    /// The name a request and a result give the kind.
    pub fn name(&self) -> &'static str {
        match self {
            Kind::DetectorStats(_) => "detectorStats",
            Kind::Channels(_) => "channels",
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
    }
}
