//! `detectorStats`: what each detector received, per wavelength, with the
//! noise of every number.
//!
//! For each detector read: the hits, the incident flux and the detected
//! signal; and for each wavelength bin ([`oc_trace::result3::spectral_bin`])
//! the hits, the flux, the flux-weighted centroid and the RMS spot size along
//! each of the face's axes. Each value is the pooled one, from every ray of
//! every pass, and is exactly what the kernel's own readout gives for the same
//! rays; each standard error is the spread of the per-pass readings
//! ([`crate::stats`]). Hit counts are counts of what happened and carry none.

use std::collections::BTreeMap;

use oc_core::scene3::Scene3;
use oc_trace::result3::SPECTRAL_FIELDS;
use oc_trace::{DetectorResult3, SpectralSums3, TraceResult3};
use serde_json::{json, Value};

use crate::request::DetectorStatsParams;
use crate::stats::Estimate;

/// One pass's sums on one detector.
#[derive(Debug, Clone, Default)]
struct PassReading {
    incident: f64,
    signal: f64,
    /// Each wavelength bin's sums, keyed by the bits of its centre (positive,
    /// so they sort as the wavelengths do).
    lines: BTreeMap<u64, [f64; SPECTRAL_FIELDS]>,
}

/// The per-pass sums of the detectors a `detectorStats` job reads.
#[derive(Debug, Clone)]
pub(crate) struct DetectorStats {
    /// The detectors read, as indices into the scene's (and the trace's)
    /// detector list, in the order the request named them.
    index: Vec<usize>,
    /// `[pass][detector read]`.
    passes: Vec<Vec<PassReading>>,
}

impl DetectorStats {
    pub(crate) fn new(scene: &Scene3, p: &DetectorStatsParams, passes: u32) -> Result<Self, String> {
        let index = match &p.detectors {
            None => (0..scene.detectors.len()).collect(),
            Some(ids) => ids
                .iter()
                .map(|id| {
                    scene
                        .detectors
                        .iter()
                        .position(|d| d.common.id.0 == *id)
                        .ok_or_else(|| format!("detectorStats: no detector has id {id}"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        if index.is_empty() {
            return Err("detectorStats: the scene has no detector to read".into());
        }
        let blank = vec![PassReading::default(); index.len()];
        Ok(Self {
            index,
            passes: vec![blank; passes as usize],
        })
    }

    /// Add one block's readouts to pass `pass`. Blocks come in order, so each
    /// pass's sums are added in one fixed order however the job was stepped.
    pub(crate) fn add_block(&mut self, pass: usize, detectors: &[DetectorResult3]) {
        for (reading, &i) in self.passes[pass].iter_mut().zip(&self.index) {
            let d = &detectors[i];
            reading.incident += d.total_incident;
            reading.signal += d.total_signal;
            for s in &d.spectral {
                let sum = reading
                    .lines
                    .entry(s.lambda_um.to_bits())
                    .or_insert([0.0; SPECTRAL_FIELDS]);
                let x = s.to_array();
                // The bin centre is a key, not a sum.
                sum[0] = x[0];
                for k in 1..SPECTRAL_FIELDS {
                    sum[k] += x[k];
                }
            }
        }
    }

    /// The result: pooled values from the merged trace `total`, standard
    /// errors from the passes.
    pub(crate) fn result(&self, scene: &Scene3, total: &TraceResult3) -> Value {
        // Each launch ray carries 1/passes of its flux, so a pass's sum times
        // the passes is that pass's own estimate of the total.
        let k = self.passes.len() as f64;
        let detectors: Vec<Value> = self
            .index
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                let d = &total.detectors[i];
                let per_pass = |f: &dyn Fn(&PassReading) -> f64| -> Vec<f64> {
                    self.passes.iter().map(|p| k * f(&p[j])).collect()
                };
                let lines: Vec<Value> = d
                    .spectral
                    .iter()
                    .map(|s| self.line(j, s, k))
                    .collect();
                json!({
                    "detector": d.detector.0,
                    "label": scene.detectors[i].common.label,
                    "hits": d.hits,
                    "flux": Estimate::new(d.total_incident, &per_pass(&|p| p.incident)),
                    "signal": Estimate::new(d.total_signal, &per_pass(&|p| p.signal)),
                    "lines": lines,
                })
            })
            .collect();
        json!({ "detectors": detectors })
    }

    /// One wavelength bin of detector `j`, its pooled sums `s`.
    fn line(&self, j: usize, s: &SpectralSums3, k: f64) -> Value {
        let key = s.lambda_um.to_bits();
        let sums: Vec<SpectralSums3> = self
            .passes
            .iter()
            .map(|p| {
                p[j].lines
                    .get(&key)
                    .map(SpectralSums3::from_array)
                    .unwrap_or_default()
            })
            .collect();
        let flux: Vec<f64> = sums.iter().map(|x| k * x.flux).collect();
        // A centroid or a spot size exists only in a pass the line reached.
        let lit: Vec<&SpectralSums3> = sums.iter().filter(|x| x.flux > 0.0).collect();
        let centroid: Vec<[f64; 2]> = lit
            .iter()
            .filter_map(|x| x.centroid().map(|c| [c.x, c.y]))
            .collect();
        let rms: Vec<[f64; 2]> = lit.iter().filter_map(|x| x.rms_uv()).collect();
        let axis = |v: Option<f64>, from: &[[f64; 2]], a: usize| -> Value {
            match v {
                Some(v) => json!(Estimate::new(
                    v,
                    &from.iter().map(|x| x[a]).collect::<Vec<_>>()
                )),
                None => Value::Null,
            }
        };
        let c = s.centroid();
        let r = s.rms_uv();
        json!({
            "lambdaUm": s.lambda_um,
            "meanLambdaUm": s.mean_lambda_um(),
            "hits": s.hits,
            "flux": Estimate::new(s.flux, &flux),
            "centroidUv": [axis(c.map(|c| c.x), &centroid, 0), axis(c.map(|c| c.y), &centroid, 1)],
            "rmsUv": [axis(r.map(|r| r[0]), &rms, 0), axis(r.map(|r| r[1]), &rms, 1)],
        })
    }
}
