//! `channels`: what each channel of a multi-source scene passes, and how
//! much of the other sources' light leaks into it.
//!
//! A channel is a detector, the sources it is meant to see, and optionally a
//! band of wavelengths it counts. The job traces each source alone
//! ([`crate::job::source_alone`]), so the light on every detector is known by
//! the source it came from. For each source and channel it reads the flux the
//! detector received from that source, in the channel's band: the crosstalk
//! table. A channel's signal is the sum of its own sources' entries, its leak
//! the sum of the others', and its ratio leak over signal.
//!
//! Each value is the pooled one, from every ray of every pass, and each entry
//! is exactly what the kernel's readout of that source alone gives; each
//! standard error is the spread of the per-pass readings ([`crate::stats`]).
//! The sources are traced independently, so a pass's signal is the sum of its
//! sources' pass readings, and its spread is the signal's error.

use std::collections::BTreeMap;

use oc_core::scene3::Scene3;
use oc_trace::{DetectorResult3, TraceResult3};
use serde_json::{json, Value};

use crate::request::{ChannelSpec, ChannelsParams};
use crate::stats::Estimate;

/// One pass's sums on one detector, from one source.
#[derive(Debug, Clone, Default)]
struct PassReading {
    incident: f64,
    /// Each wavelength bin's flux, keyed by the bits of its centre (positive,
    /// so they sort as the wavelengths do).
    lines: BTreeMap<u64, f64>,
}

/// A channel, checked against the scene.
#[derive(Debug, Clone)]
struct Channel {
    name: String,
    /// Index into the detectors read.
    read: usize,
    /// Whether each source (in scene order) is the channel's own.
    own: Vec<bool>,
    band: Option<[f64; 2]>,
}

/// The per-pass sums of a `channels` job.
#[derive(Debug, Clone)]
pub(crate) struct Channels {
    channels: Vec<Channel>,
    /// The detectors read, as indices into the scene's (and each trace's)
    /// detector list, each once.
    index: Vec<usize>,
    /// `[source][pass][detector read]`.
    passes: Vec<Vec<Vec<PassReading>>>,
}

fn bad(what: String) -> String {
    format!("channels: {what}")
}

impl Channels {
    pub(crate) fn new(scene: &Scene3, p: &ChannelsParams, passes: u32) -> Result<Self, String> {
        let sources: Vec<u64> = scene.sources.iter().map(|s| s.id.0).collect();
        if scene.detectors.is_empty() {
            return Err(bad("the scene has no detector to read".into()));
        }
        let detector = |id: u64| {
            scene
                .detectors
                .iter()
                .position(|d| d.common.id.0 == id)
                .ok_or_else(|| bad(format!("no detector has id {id}")))
        };
        // Each channel asked for, with its detector's index.
        let specs: Vec<(usize, ChannelSpec)> = match &p.channels {
            None => scene
                .detectors
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let spec = ChannelSpec {
                        name: None,
                        detector: d.common.id.0,
                        sources: None,
                        band_um: None,
                    };
                    (i, spec)
                })
                .collect(),
            Some(list) if list.is_empty() => return Err(bad("name at least one channel".into())),
            Some(list) => list
                .iter()
                .map(|c| Ok((detector(c.detector)?, c.clone())))
                .collect::<Result<_, String>>()?,
        };
        let mut index: Vec<usize> = Vec::new();
        let mut channels: Vec<Channel> = Vec::new();
        // Named channels first, so a default name gives way to a given one.
        let given: Vec<String> = specs.iter().filter_map(|(_, c)| c.name.clone()).collect();
        for (k, n) in given.iter().enumerate() {
            if n.trim().is_empty() {
                return Err(bad("a channel's name is empty".into()));
            }
            if given[..k].contains(n) {
                return Err(bad(format!("two channels are named {n:?}")));
            }
        }
        for (i, spec) in specs {
            let ChannelSpec {
                name,
                sources: own,
                band_um: band,
                ..
            } = spec;
            let d = &scene.detectors[i];
            let name = match name {
                Some(n) => n,
                None => {
                    let id = d.common.id.0;
                    let base = if d.common.label.trim().is_empty() {
                        format!("detector {id}")
                    } else {
                        d.common.label.clone()
                    };
                    let taken = |n: &str| {
                        given.iter().any(|g| g == n) || channels.iter().any(|c| c.name == n)
                    };
                    if taken(&base) {
                        format!("{base} ({id})")
                    } else {
                        base
                    }
                }
            };
            let own = match own {
                None => vec![true; sources.len()],
                Some(ids) if ids.is_empty() => {
                    return Err(bad(format!(
                        "channel {name:?} sees no source; name at least one"
                    )))
                }
                Some(ids) => {
                    let mut own = vec![false; sources.len()];
                    for id in ids {
                        let s = sources.iter().position(|&s| s == id).ok_or_else(|| {
                            bad(format!("channel {name:?}: no source has id {id}"))
                        })?;
                        if own[s] {
                            return Err(bad(format!("channel {name:?} lists source {id} twice")));
                        }
                        own[s] = true;
                    }
                    own
                }
            };
            if let Some([lo, hi]) = band {
                if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && lo < hi) {
                    return Err(bad(format!(
                        "channel {name:?}: bandUm must be [low, high] in µm with 0 < low < high, not [{lo}, {hi}]"
                    )));
                }
            }
            let read = match index.iter().position(|&j| j == i) {
                Some(r) => r,
                None => {
                    index.push(i);
                    index.len() - 1
                }
            };
            channels.push(Channel {
                name,
                read,
                own,
                band,
            });
        }
        let blank = vec![PassReading::default(); index.len()];
        Ok(Self {
            channels,
            index,
            passes: vec![vec![blank; passes as usize]; sources.len()],
        })
    }

    /// Add one block of source `source`'s trace to pass `pass`. Blocks come in
    /// order, so each pass's sums are added in one fixed order however the
    /// job was stepped.
    pub(crate) fn add_block(&mut self, source: usize, pass: usize, detectors: &[DetectorResult3]) {
        for (reading, &i) in self.passes[source][pass].iter_mut().zip(&self.index) {
            let d = &detectors[i];
            reading.incident += d.total_incident;
            for s in &d.spectral {
                *reading.lines.entry(s.lambda_um.to_bits()).or_insert(0.0) += s.flux;
            }
        }
    }

    /// Source `s`'s flux on channel `c`: the pooled value from its merged
    /// trace `total`, and each pass's own estimate of it.
    ///
    /// In a band, a wavelength bin counts when its flux-weighted mean
    /// wavelength, over the pooled trace, is in the band; each pass then adds
    /// the same bins.
    fn entry(&self, s: usize, c: &Channel, total: &TraceResult3) -> (f64, Vec<f64>) {
        // Each launch ray carries 1/passes of its flux, so a pass's sum times
        // the passes is that pass's own estimate of the total.
        let k = self.passes[s].len() as f64;
        let d = &total.detectors[self.index[c.read]];
        let readings = self.passes[s].iter().map(|p| &p[c.read]);
        match c.band {
            None => (d.total_incident, readings.map(|p| k * p.incident).collect()),
            Some([lo, hi]) => {
                let bins: Vec<_> = d
                    .spectral
                    .iter()
                    .filter(|b| (lo..=hi).contains(&b.mean_lambda_um()))
                    .collect();
                let value = bins.iter().fold(0.0, |a, b| a + b.flux);
                let per_pass = readings
                    .map(|p| {
                        k * bins.iter().fold(0.0, |a, b| {
                            a + p.lines.get(&b.lambda_um.to_bits()).copied().unwrap_or(0.0)
                        })
                    })
                    .collect();
                (value, per_pass)
            }
        }
    }

    /// The result: pooled values from each source's merged trace `totals`
    /// (in scene order, launch rays `rays`), standard errors from the passes.
    pub(crate) fn result(&self, scene: &Scene3, totals: &[&TraceResult3], rays: &[u64]) -> Value {
        let m = self.passes.first().map_or(0, Vec::len);
        // [source][channel]
        let entries: Vec<Vec<(f64, Vec<f64>)>> = totals
            .iter()
            .enumerate()
            .map(|(s, t)| self.channels.iter().map(|c| self.entry(s, c, t)).collect())
            .collect();
        let sum = |c: usize, own: bool| -> (f64, Vec<f64>) {
            let ch = &self.channels[c];
            let mut value = 0.0;
            let mut per_pass = vec![0.0; m];
            for (s, row) in entries.iter().enumerate() {
                if ch.own[s] == own {
                    value += row[c].0;
                    for (a, b) in per_pass.iter_mut().zip(&row[c].1) {
                        *a += b;
                    }
                }
            }
            (value, per_pass)
        };
        let source_ids: Vec<u64> = scene.sources.iter().map(|s| s.id.0).collect();
        let channels: Vec<Value> = self
            .channels
            .iter()
            .enumerate()
            .map(|(c, ch)| {
                let (signal, sp) = sum(c, true);
                let (leak, lp) = sum(c, false);
                // Leak over signal; its error to first order is the spread of
                // (leak - ratio * signal) / signal over the passes.
                let ratio = (signal > 0.0).then(|| {
                    let r = leak / signal;
                    let z: Vec<f64> = lp
                        .iter()
                        .zip(&sp)
                        .map(|(l, s)| (l - r * s) / signal)
                        .collect();
                    Estimate::new(r, &z)
                });
                let i = self.index[ch.read];
                let own: Vec<u64> = source_ids
                    .iter()
                    .zip(&ch.own)
                    .filter_map(|(id, &o)| o.then_some(*id))
                    .collect();
                json!({
                    "name": ch.name,
                    "detector": scene.detectors[i].common.id.0,
                    "label": scene.detectors[i].common.label,
                    "sources": own,
                    "bandUm": ch.band,
                    "signal": Estimate::new(signal, &sp),
                    "leak": Estimate::new(leak, &lp),
                    "ratio": ratio,
                })
            })
            .collect();
        let flux: Vec<Vec<Estimate>> = entries
            .iter()
            .map(|row| row.iter().map(|(v, pp)| Estimate::new(*v, pp)).collect())
            .collect();
        let sources: Vec<Value> = source_ids
            .iter()
            .zip(totals)
            .zip(rays)
            .map(|((id, t), r)| json!({ "source": id, "rays": r, "launched": t.ledger.launched }))
            .collect();
        json!({
            "sources": sources,
            "channels": channels,
            "crosstalk": {
                "sources": source_ids,
                "channels": self.channels.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
                "flux": flux,
            },
        })
    }
}
