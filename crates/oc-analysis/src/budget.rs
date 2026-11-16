//! `budget`: where the light went.
//!
//! Every watt launched ends in one of the kernel's ledger sinks: a detector
//! that stops it, a surface or a coating that absorbs it, a body's bulk, a
//! stop that blocks it, the scene's bounds, a depth, ghost, branch or flux
//! limit that culls it, or a numerical loss. The job reads that ledger
//! ([`oc_trace::EnergyLedger`]) for the totals, each detector's readout for
//! what it caught, and the object ledger ([`oc_trace::ObjectLedger3`]) for
//! where, part by part, the light was lost.
//!
//! Every figure carries a standard error from the passes: the totals from
//! each block's ledger ([`oc_trace::TracePart3::ledger`]), the detectors from
//! its readouts, and the parts, faces included, from its object ledger
//! ([`oc_trace::TracePart3::objects`]). The values themselves are the pooled
//! ones, read off the merged result; the passes give only the spread. A
//! part's `hits` is a plain count.
//!
//! The sinks partition what was launched, so the totals close: launched less
//! their sum is the kernel's own residual, which is rounding. The parts close
//! the same way, since the object ledger books each removal once.

use std::collections::BTreeMap;

use oc_core::scene3::{DetectorTermination, Scene3};
use oc_trace::{
    DetectorResult3, EnergyLedger, ObjectEntry3, ObjectLedger3, ObjectRef3, TraceResult3,
};
use serde_json::{json, Value};

use crate::request::BudgetParams;
use crate::stats::Estimate;

/// The ledger's sinks, by the names the result gives them. Together they are
/// everything the ledger accounts for beside what was launched.
const SINKS: [&str; 7] = [
    "detected",
    "absorbedSurface",
    "absorbedVolume",
    "blocked",
    "escaped",
    "truncated",
    "numerical",
];

fn sinks(l: &EnergyLedger) -> [f64; 7] {
    [
        l.detected_incident,
        l.absorbed_surface,
        l.absorbed_volume,
        l.blocked_aperture,
        l.escaped,
        l.truncated_flux,
        l.numerical_loss,
    ]
}

/// One pass's sums from one trace.
#[derive(Debug, Clone, Default)]
struct PassReading {
    sinks: [f64; 7],
    /// Each detector's incident flux, in scene order.
    detectors: Vec<f64>,
    /// The object ledger, part by part.
    parts: BTreeMap<PartKey, Part>,
}

/// The per-pass sums of a `budget` job.
#[derive(Debug, Clone)]
pub(crate) struct Budget {
    by_source: bool,
    /// `[trace][pass]`: one trace, or one per source.
    passes: Vec<Vec<PassReading>>,
}

/// A part of the scene as the result names it: its kind's order, then its
/// index in the scene's list of that kind (or the trace, for the launch).
type PartKey = (u8, usize);

/// What one part did to the light, and each of a body's faces.
#[derive(Debug, Clone, Default)]
struct Part {
    ledger: ObjectLedger3,
    faces: BTreeMap<u64, ObjectLedger3>,
}

impl Budget {
    pub(crate) fn new(scene: &Scene3, p: &BudgetParams, passes: u32) -> Result<Self, String> {
        let traces = if p.by_source { scene.sources.len() } else { 1 };
        let blank = PassReading {
            sinks: [0.0; 7],
            detectors: vec![0.0; scene.detectors.len()],
            parts: BTreeMap::new(),
        };
        Ok(Self {
            by_source: p.by_source,
            passes: vec![vec![blank; passes as usize]; traces],
        })
    }

    /// Whether each source is traced alone.
    pub(crate) fn by_source(&self) -> bool {
        self.by_source
    }

    /// Add one block of trace `trace` to pass `pass`. Blocks come in order,
    /// so each pass's sums are added in one fixed order however the job was
    /// stepped.
    pub(crate) fn add_block(
        &mut self,
        trace: usize,
        pass: usize,
        ledger: &EnergyLedger,
        detectors: &[DetectorResult3],
        objects: impl Iterator<Item = ObjectEntry3>,
    ) {
        let by_source = self.by_source;
        let r = &mut self.passes[trace][pass];
        for e in objects {
            book(&mut r.parts, by_source, trace, &e);
        }
        for (a, b) in r.sinks.iter_mut().zip(sinks(ledger)) {
            *a += b;
        }
        for (a, d) in r.detectors.iter_mut().zip(detectors) {
            *a += d.total_incident;
        }
    }

    /// The result: pooled values from each trace's merged result `totals`
    /// (launch rays `rays`), standard errors from the passes.
    pub(crate) fn result(&self, scene: &Scene3, totals: &[&TraceResult3], rays: &[u64]) -> Value {
        let all: Vec<usize> = (0..totals.len()).collect();
        let mut out = self.summary(scene, totals, &all);
        if self.by_source {
            let sources: Vec<Value> = scene
                .sources
                .iter()
                .enumerate()
                .map(|(s, src)| {
                    let mut v = self.summary(scene, totals, &[s]);
                    v["source"] = json!(src.id.0);
                    v["rays"] = json!(rays[s]);
                    v
                })
                .collect();
            out["sources"] = Value::Array(sources);
        }
        out
    }

    /// The budget of the traces `of`, added up in order.
    fn summary(&self, scene: &Scene3, totals: &[&TraceResult3], of: &[usize]) -> Value {
        // Each launch ray carries 1/passes of its flux, so a pass's sum times
        // the passes is that pass's own estimate of the total.
        let m = self.passes[0].len();
        let k = m as f64;
        let per_pass = |f: &dyn Fn(&PassReading) -> f64| -> Vec<f64> {
            (0..m)
                .map(|p| k * of.iter().fold(0.0, |a, &t| a + f(&self.passes[t][p])))
                .collect()
        };
        let pooled =
            |f: &dyn Fn(&TraceResult3) -> f64| of.iter().fold(0.0, |a, &t| a + f(totals[t]));

        let launched = pooled(&|t| t.ledger.launched);
        let share = |v: f64| (launched > 0.0).then(|| v / launched);
        let mut sums = serde_json::Map::new();
        let mut shares = serde_json::Map::new();
        let mut accounted = 0.0;
        for (c, name) in SINKS.iter().enumerate() {
            let v = pooled(&|t| sinks(&t.ledger)[c]);
            accounted += v;
            sums.insert(
                name.to_string(),
                json!(Estimate::new(v, &per_pass(&|p| p.sinks[c]))),
            );
            shares.insert(name.to_string(), json!(share(v)));
        }

        let detectors: Vec<Value> = scene
            .detectors
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let flux = pooled(&|t| t.detectors[i].total_incident);
                json!({
                    "detector": d.common.id.0,
                    "label": d.common.label,
                    "terminates": matches!(d.termination, DetectorTermination::Terminate),
                    "hits": of.iter().map(|&t| totals[t].detectors[i].hits).sum::<u64>(),
                    "flux": Estimate::new(flux, &per_pass(&|p| p.detectors[i])),
                    "share": share(flux),
                })
            })
            .collect();

        let parts = self.parts(totals, of);
        let parts_removed = parts.iter().fold(0.0, |a, (_, p)| a + p.ledger.removed());
        let mut rows: Vec<(f64, Value)> = parts
            .iter()
            .map(|(key, p)| {
                // The part in each pass, the traces `of` added up in order; a
                // pass that never met it read nothing.
                let passes: Vec<Part> = (0..m)
                    .map(|pass| {
                        let mut sum = Part::default();
                        for &t in of {
                            if let Some(q) = self.passes[t][pass].parts.get(key) {
                                sum.add(q);
                            }
                        }
                        sum
                    })
                    .collect();
                (
                    lost(&p.ledger),
                    part_json(scene, *key, p, &passes, k, &share),
                )
            })
            .collect();
        // Largest loss first; a tie keeps the scene's order.
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut branches = oc_trace::TerminationSummary::default();
        for &t in of {
            let s = &totals[t].terminations;
            branches.detector += s.detector;
            branches.absorbed_volume += s.absorbed_volume;
            branches.absorbed_surface += s.absorbed_surface;
            branches.aperture_blocked += s.aperture_blocked;
            branches.escaped_bounds += s.escaped_bounds;
            branches.flux_threshold += s.flux_threshold;
            branches.max_depth += s.max_depth;
            branches.max_ghost_depth += s.max_ghost_depth;
            branches.max_branches += s.max_branches;
            branches.numerical_failure += s.numerical_failure;
            branches.unsupported_physics += s.unsupported_physics;
        }
        let b = &branches;
        json!({
            "launched": launched,
            "totals": sums,
            "shares": shares,
            "closure": {
                "accounted": accounted,
                "residual": launched - accounted,
                "relative": share((launched - accounted).abs()),
                "parts": parts_removed,
                "partsResidual": launched - parts_removed,
            },
            "detectors": detectors,
            "parts": rows.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
            "branches": {
                "detector": b.detector,
                "absorbedSurface": b.absorbed_surface,
                "absorbedVolume": b.absorbed_volume,
                "blocked": b.aperture_blocked,
                "escaped": b.escaped_bounds,
                "fluxThreshold": b.flux_threshold,
                "maxDepth": b.max_depth,
                "maxGhostDepth": b.max_ghost_depth,
                "maxBranches": b.max_branches,
                "numericalFailure": b.numerical_failure,
                "unsupportedPhysics": b.unsupported_physics,
            },
        })
    }

    /// The object ledger of the traces `of`, both bands together, a body's
    /// faces gathered under it. The launch of a source traced alone is that
    /// source's own row.
    fn parts(&self, totals: &[&TraceResult3], of: &[usize]) -> BTreeMap<PartKey, Part> {
        let mut parts: BTreeMap<PartKey, Part> = BTreeMap::new();
        for &t in of {
            for e in &totals[t].objects {
                book(&mut parts, self.by_source, t, e);
            }
        }
        parts
    }
}

impl Part {
    fn add(&mut self, o: &Part) {
        self.ledger.add(&o.ledger);
        for (f, l) in &o.faces {
            self.faces.entry(*f).or_default().add(l);
        }
    }
}

/// Book one row of trace `trace`'s object ledger under its part, and its face
/// for a body.
fn book(parts: &mut BTreeMap<PartKey, Part>, by_source: bool, trace: usize, e: &ObjectEntry3) {
    let (key, face) = match e.object {
        ObjectRef3::Source => ((0, if by_source { trace } else { usize::MAX }), None),
        ObjectRef3::Surface(i) => ((1, i), None),
        ObjectRef3::Body(i, patch) => ((2, i), Some(patch)),
        ObjectRef3::Aperture(i) => ((3, i), None),
        ObjectRef3::Detector(i) => ((4, i), None),
    };
    let p = parts.entry(key).or_default();
    p.ledger.add(&e.ledger);
    if let Some(f) = face {
        p.faces.entry(f).or_default().add(&e.ledger);
    }
}

/// What a part took out of the light and kept: absorbed or blocked.
fn lost(l: &ObjectLedger3) -> f64 {
    l.absorbed_surface + l.absorbed_volume + l.blocked
}

/// One flux of an object ledger, by the name the result gives it.
type Field = (&'static str, fn(&ObjectLedger3) -> f64);

/// A ledger's figures, each with its standard error from the same ledger in
/// each pass, `passes`, a pass's figure scaled by `k` (the passes) to be that
/// pass's own estimate of the whole.
fn ledger_json(
    l: &ObjectLedger3,
    passes: &[&ObjectLedger3],
    k: f64,
) -> serde_json::Map<String, Value> {
    let fields: [Field; 9] = [
        ("arrived", |l| l.incident),
        ("lost", lost),
        ("absorbedSurface", |l| l.absorbed_surface),
        ("absorbedVolume", |l| l.absorbed_volume),
        ("blocked", |l| l.blocked),
        ("detected", |l| l.detected),
        ("escaped", |l| l.escaped),
        ("truncated", |l| l.truncated),
        ("numerical", |l| l.numerical),
    ];
    let mut m = serde_json::Map::new();
    m.insert("hits".into(), json!(l.hits));
    for (name, f) in fields {
        let per_pass: Vec<f64> = passes.iter().map(|p| k * f(p)).collect();
        m.insert(name.into(), json!(Estimate::new(f(l), &per_pass)));
    }
    m
}

fn part_json(
    scene: &Scene3,
    (kind, i): PartKey,
    p: &Part,
    passes: &[Part],
    k: f64,
    share: &dyn Fn(f64) -> Option<f64>,
) -> Value {
    let (kind, common) = match kind {
        0 => ("source", None),
        1 => ("surface", Some(&scene.surfaces[i].common)),
        2 => ("body", Some(&scene.bodies[i].common)),
        3 => ("aperture", Some(&scene.apertures[i].common)),
        _ => ("detector", Some(&scene.detectors[i].common)),
    };
    let (id, label) = match common {
        Some(c) => (json!(c.id.0), json!(c.label)),
        // The launch: light lost before its first interaction. Its source is
        // known when it was traced alone, or when there is only one.
        None => {
            let src = if i == usize::MAX {
                (scene.sources.len() == 1).then(|| scene.sources[0].id.0)
            } else {
                Some(scene.sources[i].id.0)
            };
            (json!(src), json!("launch"))
        }
    };
    let mut row = serde_json::Map::new();
    row.insert("object".into(), id);
    row.insert("kind".into(), json!(kind));
    row.insert("label".into(), label);
    let ledgers: Vec<&ObjectLedger3> = passes.iter().map(|q| &q.ledger).collect();
    row.extend(ledger_json(&p.ledger, &ledgers, k));
    row.insert("lostShare".into(), json!(share(lost(&p.ledger))));
    if kind == "body" {
        let faces: Vec<Value> = p
            .faces
            .iter()
            .map(|(patch, l)| {
                let none = ObjectLedger3::default();
                let faces: Vec<&ObjectLedger3> = passes
                    .iter()
                    .map(|q| q.faces.get(patch).unwrap_or(&none))
                    .collect();
                let mut f = serde_json::Map::new();
                f.insert("patch".into(), json!(patch));
                f.extend(ledger_json(l, &faces, k));
                Value::Object(f)
            })
            .collect();
        row.insert("faces".into(), Value::Array(faces));
    }
    Value::Object(row)
}
