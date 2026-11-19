//! `uniformity`: how even the light is over a region of one detector.
//!
//! The region is cut into a grid of cells, and each cell's irradiance is its
//! flux over the part of it on the face. From the cells: the mean irradiance,
//! the ratio of the least to the most (min/max), the coefficient of variation
//! (CV), and the falloff from the centre to the edge and the corners, each
//! with its standard error.
//!
//! **The grid is the kernel's.** The job traces the scene with the detector's
//! pixel grid set so that every cell is a whole number of pixels: the cells
//! themselves when the region is the whole face, and a finer grid, snapped to
//! the region, otherwise ([`snap`]). A pixel grid bins hits and moves no ray,
//! so the trace is the scene's own. Each cell's flux is the sum of its pixels'
//! in the kernel's readout.
//!
//! **Noise in, noise out.** A noisy cell is as likely high as low, so the
//! spread of noisy cells overstates the field's own, and the least and most of
//! them overstate its extremes. The passes are split into two halves, the even
//! and the odd ones, whose noise is independent:
//!
//! - the CV's variance is the mean product of each cell's deviation in one
//!   half with its deviation in the other, which noise does not bias;
//! - min/max picks the least and the most lit cells in one half and reads them
//!   in the other, and the other way round, and takes the mean of the two.
//!
//! Each standard error is first-order, from the spread of the passes in each
//! half ([`crate::stats`]); min/max's is conditional on the cells picked.
//!
//! **Stopping on a noise target.** A job with a target checks, after 8, 16, 32
//! passes and so on, whether the target metric's standard error is down to
//! it, and stops there if so. A check reads whole passes only, so where a job
//! stops does not depend on how it was stepped.

use oc_core::scene3::{DetectorMeasure, Scene3};
use oc_trace::{DetectorResult3, TraceResult3};
use serde_json::{json, Value};

use crate::focus::{MAX_PILOT_RAYS, MIN_PILOT_RAYS};
use crate::request::UniformityParams;
use crate::stats::{standard_error, Estimate};

/// The most cells a `uniformity` grid may hold. Each pass keeps one number
/// per cell.
pub const MAX_UNIFORMITY_CELLS: u64 = 1 << 16;

/// The most pixels the traced grid has along one axis when it is snapped to a
/// region: `1024²` is the kernel's own cap on a detector grid.
const MAX_FINE: usize = 1024;

/// The passes after which a job with a target checks it: powers of two from
/// 8. Few checks keep the stop from chasing a lucky low reading.
pub fn is_checkpoint(passes: usize) -> bool {
    passes >= 8 && passes.is_power_of_two()
}

/// The metric a target is set on.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Metric {
    Cv,
    MinMax,
}

impl Metric {
    fn name(self) -> &'static str {
        match self {
            Metric::Cv => "cv",
            Metric::MinMax => "minMax",
        }
    }
}

/// One axis of the traced grid: `fine` pixels across the face's grid, the
/// region starting at pixel `start`, `per_cell` pixels to a cell.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Axis {
    fine: usize,
    start: usize,
    per_cell: usize,
}

/// The pixel grid along one axis of extent `w` (centred on 0) that puts the
/// edges of `n` equal cells over `[lo, hi]` closest to where they were asked,
/// and the region it reads.
///
/// The fewest pixels whose edges land within a thousandth of a cell of the
/// ones asked for, or failing that, of up to [`MAX_FINE`] pixels, the closest.
/// The whole extent is `n` pixels, exactly.
fn snap(w: f64, lo: f64, hi: f64, n: usize) -> Axis {
    let (a, b) = (lo + 0.5 * w, hi + 0.5 * w);
    let tol = 1e-3 * (b - a) / n as f64;
    let mut best: Option<(f64, Axis)> = None;
    for fine in n..=MAX_FINE.max(n) {
        let p = w / fine as f64;
        let per_cell = (((b - a) / (n as f64 * p)).round() as usize).max(1);
        if n * per_cell > fine {
            continue;
        }
        let start = ((a / p).round().max(0.0) as usize).min(fine - n * per_cell);
        let end = start + n * per_cell;
        let err = (start as f64 * p - a).abs().max((end as f64 * p - b).abs());
        let axis = Axis {
            fine,
            start,
            per_cell,
        };
        if best.as_ref().map_or(true, |(e, _)| err < *e) {
            best = Some((err, axis));
        }
        if err <= tol {
            break;
        }
    }
    best.map(|(_, a)| a).expect("n pixels always fit")
}

/// What a check, or the result, reads off the passes so far.
struct Metrics {
    cv: Option<Estimate>,
    min_max: Option<Estimate>,
}

/// The per-pass sums of a `uniformity` job.
#[derive(Debug, Clone)]
pub(crate) struct Uniformity {
    /// The detector read, as an index into the scene's detector list.
    det: usize,
    u: Axis,
    v: Axis,
    cells: [usize; 2],
    /// The region read, `[[u low, u high], [v low, v high]]` mm, and the one
    /// asked for.
    region: [[f64; 2]; 2],
    asked: Option<[[f64; 2]; 2]>,
    /// Each cell's area on the face, mm²; 0 for a cell off it.
    area: Vec<f64>,
    target: Option<(Metric, f64)>,
    /// Each cell's flux: `[pass][cell]`.
    passes: Vec<Vec<f64>>,
    /// Where the job stopped on its target, in passes, and the standard
    /// error it read there.
    stopped: Option<(usize, f64)>,
    /// For a focused job, the pilot rays it asked for, if it named them;
    /// `None` for a forward one.
    pub(crate) focused: Option<Option<u64>>,
}

fn bad(what: String) -> String {
    format!("uniformity: {what}")
}

impl Uniformity {
    pub(crate) fn new(scene: &Scene3, p: &UniformityParams, passes: u32) -> Result<Self, String> {
        if passes < 4 {
            return Err(bad(format!(
                "{passes} passes; ask for at least 4: the noise is read from two halves of the passes"
            )));
        }
        let det = scene
            .detectors
            .iter()
            .position(|d| d.common.id.0 == p.detector)
            .ok_or_else(|| bad(format!("no detector has id {}", p.detector)))?;
        let d = &scene.detectors[det];
        let [w, h] = d.grid_extent_mm().ok_or_else(|| {
            bad(format!(
                "detector {} is a line probe, with no area",
                p.detector
            ))
        })?;
        let [nu, nv] = p.cells;
        if nu == 0 || nv == 0 || nu.saturating_mul(nv) > MAX_UNIFORMITY_CELLS {
            return Err(bad(format!(
                "cells must be [across u, across v], each at least 1 and {MAX_UNIFORMITY_CELLS} in all, not [{nu}, {nv}]"
            )));
        }
        let (nu, nv) = (nu as usize, nv as usize);
        if nu > MAX_FINE || nv > MAX_FINE {
            return Err(bad(format!(
                "at most {MAX_FINE} cells along an axis, not [{nu}, {nv}]"
            )));
        }
        let whole = [[-0.5 * w, 0.5 * w], [-0.5 * h, 0.5 * h]];
        let asked = p.region_mm;
        let r = asked.unwrap_or(whole);
        for (k, (axis, ext)) in r.iter().zip([w, h]).enumerate() {
            let [lo, hi] = *axis;
            let name = ["u", "v"][k];
            let slack = 1e-9 * ext;
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(bad(format!(
                    "regionMm along {name} must be [low, high] with low < high, not [{lo}, {hi}]"
                )));
            }
            if lo < -0.5 * ext - slack || hi > 0.5 * ext + slack {
                return Err(bad(format!(
                    "regionMm along {name} is [{lo}, {hi}], off the face's grid of [{}, {}]",
                    -0.5 * ext,
                    0.5 * ext
                )));
            }
        }
        let (u, v) = if asked.is_some() {
            (snap(w, r[0][0], r[0][1], nu), snap(h, r[1][0], r[1][1], nv))
        } else {
            let whole = |n| Axis {
                fine: n,
                start: 0,
                per_cell: 1,
            };
            (whole(nu), whole(nv))
        };
        let edge = |a: &Axis, ext: f64, i: usize| -0.5 * ext + ext * i as f64 / a.fine as f64;
        let region = [
            [edge(&u, w, u.start), edge(&u, w, u.start + nu * u.per_cell)],
            [edge(&v, h, v.start), edge(&v, h, v.start + nv * v.per_cell)],
        ];
        let fine_area = d
            .geometry
            .covered_cell_areas_mm2([u.fine, v.fine])
            .expect("an area detector has cell areas");
        let mut area = vec![0.0; nu * nv];
        for_each_pixel(&u, &v, [nu, nv], |pixel, cell| {
            area[cell] += fine_area[pixel]
        });
        if area.iter().all(|&a| a <= 0.0) {
            return Err(bad(
                "every cell of the region is off the detector's face".into()
            ));
        }
        let target = match &p.target {
            None => None,
            Some(t) => {
                let metric = match t.metric.as_str() {
                    "cv" => Metric::Cv,
                    "minMax" => Metric::MinMax,
                    other => {
                        return Err(bad(format!(
                            "a target is on \"cv\" or \"minMax\", not {other:?}"
                        )))
                    }
                };
                if !(t.stderr.is_finite() && t.stderr > 0.0) {
                    return Err(bad(format!(
                        "a target's stderr must be above 0, not {}",
                        t.stderr
                    )));
                }
                Some((metric, t.stderr))
            }
        };
        let focused = match (p.sampling.as_deref(), p.pilot_rays) {
            (None | Some("forward"), None) => None,
            (None | Some("forward"), Some(_)) => {
                return Err(bad(
                    "pilotRays is for \"sampling\": \"focused\" alone".into(),
                ))
            }
            (Some("focused"), rays) => {
                if let Some(r) = rays.filter(|r| !(MIN_PILOT_RAYS..=MAX_PILOT_RAYS).contains(r)) {
                    return Err(bad(format!(
                        "pilotRays must be from {MIN_PILOT_RAYS} to {MAX_PILOT_RAYS}, not {r}"
                    )));
                }
                Some(rays)
            }
            (Some(other), _) => {
                return Err(bad(format!(
                    "sampling is \"forward\" or \"focused\", not {other:?}"
                )))
            }
        };
        Ok(Self {
            focused,
            det,
            u,
            v,
            cells: [nu, nv],
            region,
            asked,
            area,
            target,
            passes: vec![vec![0.0; nu * nv]; passes as usize],
            stopped: None,
        })
    }

    /// `scene` as the job traces it: the detector's pixel grid set to the one
    /// the cells are cut from, and its measure to plain power, which bins
    /// nothing besides. Neither moves a ray.
    pub(crate) fn traced_scene(&self, scene: &Scene3) -> Scene3 {
        let mut out = scene.clone();
        let d = &mut out.detectors[self.det];
        d.bins = [self.u.fine, self.v.fine];
        d.measure = DetectorMeasure::Power;
        out
    }

    /// Add one block's readouts to pass `pass`. Blocks come in order, so each
    /// pass's sums are added in one fixed order however the job was stepped.
    pub(crate) fn add_block(&mut self, pass: usize, detectors: &[DetectorResult3]) {
        let flux = &detectors[self.det].incident_flux;
        let sums = &mut self.passes[pass];
        for_each_pixel(&self.u, &self.v, self.cells, |pixel, cell| {
            sums[cell] += flux[pixel]
        });
    }

    /// The light one block's readouts put into the region, for the pilot of
    /// a focused job.
    pub(crate) fn region_flux(&self, detectors: &[DetectorResult3]) -> f64 {
        let flux = &detectors[self.det].incident_flux;
        let mut sum = 0.0;
        for_each_pixel(&self.u, &self.v, self.cells, |pixel, _| sum += flux[pixel]);
        sum
    }

    /// Whether a job with a target stops after `done` whole passes: at a
    /// checkpoint, once the target metric's standard error is down to it.
    pub(crate) fn stop_after(&mut self, done: usize) -> bool {
        let Some((metric, want)) = self.target else {
            return false;
        };
        if !is_checkpoint(done) || done >= self.passes.len() {
            return false;
        }
        let m = self.metrics(done);
        let se = match metric {
            Metric::Cv => m.cv,
            Metric::MinMax => m.min_max,
        }
        .and_then(|e| e.stderr);
        match se {
            Some(se) if se <= want => {
                self.stopped = Some((done, se));
                true
            }
            _ => false,
        }
    }

    /// The cells on the face.
    fn on_face(&self) -> Vec<usize> {
        (0..self.area.len())
            .filter(|&c| self.area[c] > 0.0)
            .collect()
    }

    /// Each cell's irradiance from each of the first `q` passes, `[pass][cell]`.
    /// Each launch ray carries 1/passes of its flux, so a pass's sum times the
    /// passes is that pass's own estimate.
    fn per_pass(&self, q: usize) -> Vec<Vec<f64>> {
        let k = self.passes.len() as f64;
        self.passes[..q]
            .iter()
            .map(|p| {
                p.iter()
                    .zip(&self.area)
                    .map(|(f, a)| if *a > 0.0 { k * f / a } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    /// The CV and min/max from the first `q` passes, split into the even and
    /// the odd ones.
    fn metrics(&self, q: usize) -> Metrics {
        let x = self.per_pass(q);
        let face = self.on_face();
        let even: Vec<&Vec<f64>> = x.iter().step_by(2).collect();
        let odd: Vec<&Vec<f64>> = x.iter().skip(1).step_by(2).collect();
        let mean_of = |half: &[&Vec<f64>]| -> Vec<f64> {
            let h = half.len() as f64;
            (0..self.area.len())
                .map(|c| half.iter().map(|p| p[c]).sum::<f64>() / h)
                .collect()
        };
        let (a, b) = (mean_of(&even), mean_of(&odd));
        Metrics {
            cv: cv(&face, &even, &odd, &a, &b),
            min_max: min_max(&face, &even, &odd, &a, &b),
        }
    }

    /// The result, from the merged trace `total` of the passes traced and
    /// the per-pass sums. `rays` is the launch rays traced and `limit` the
    /// most the job would have.
    pub(crate) fn result(
        &self,
        scene: &Scene3,
        total: &TraceResult3,
        rays: u64,
        limit: u64,
    ) -> Value {
        let q = self.stopped.map_or(self.passes.len(), |(q, _)| q);
        let k = self.passes.len() as f64 / q as f64;
        let [nu, nv] = self.cells;
        let d = &total.detectors[self.det];
        let mut pooled = vec![0.0; nu * nv];
        for_each_pixel(&self.u, &self.v, self.cells, |pixel, cell| {
            pooled[cell] += d.incident_flux[pixel]
        });
        // The merge holds the first `q` passes, each ray at 1/passes of its
        // flux: scaled up, it is their mean.
        let pooled: Vec<f64> = pooled
            .iter()
            .zip(&self.area)
            .map(|(f, a)| if *a > 0.0 { k * f / a } else { 0.0 })
            .collect();
        let x = self.per_pass(q);
        let face = self.on_face();
        let cell_se: Vec<Option<f64>> = (0..nu * nv)
            .map(|c| {
                (self.area[c] > 0.0)
                    .then(|| standard_error(&x.iter().map(|p| p[c]).collect::<Vec<_>>()))
                    .flatten()
            })
            .collect();
        let lit: Vec<usize> = (0..nu * nv)
            .map(|c| x.iter().filter(|p| p[c] > 0.0).count())
            .collect();
        let dark = face.iter().filter(|&&c| lit[c] == 0).count();
        let sparse = face.iter().filter(|&&c| lit[c] > 0 && lit[c] < q).count();
        let mut rel: Vec<f64> = face
            .iter()
            .filter(|&&c| pooled[c] > 0.0)
            .filter_map(|&c| cell_se[c].map(|s| s / pooled[c]))
            .collect();
        rel.sort_by(f64::total_cmp);
        let median_rel = (!rel.is_empty()).then(|| {
            let n = rel.len();
            if n % 2 == 1 {
                rel[n / 2]
            } else {
                0.5 * (rel[n / 2 - 1] + rel[n / 2])
            }
        });

        let group_mean = |cells: &[usize], of: &dyn Fn(usize) -> f64| -> f64 {
            cells.iter().map(|&c| of(c)).sum::<f64>() / cells.len() as f64
        };
        let mean = Estimate::new(
            group_mean(&face, &|c| pooled[c]),
            &x.iter()
                .map(|p| group_mean(&face, &|c| p[c]))
                .collect::<Vec<_>>(),
        );
        // Ratio of two groups' mean irradiance, pooled; its error to first
        // order is the spread of (top - ratio * bottom) / bottom over the
        // passes.
        let ratio = |top: &[usize], bottom: &[usize]| -> Option<Estimate> {
            if top.is_empty() || bottom.is_empty() {
                return None;
            }
            let (t, b) = (
                group_mean(top, &|c| pooled[c]),
                group_mean(bottom, &|c| pooled[c]),
            );
            (b > 0.0).then(|| {
                let r = t / b;
                let z: Vec<f64> = x
                    .iter()
                    .map(|p| (group_mean(top, &|c| p[c]) - r * group_mean(bottom, &|c| p[c])) / b)
                    .collect();
                Estimate::new(r, &z)
            })
        };
        let (centre, edge, corner) = self.falloff_groups();
        let falloff = if nu >= 3 && nv >= 3 {
            json!({ "edge": ratio(&edge, &centre), "corner": ratio(&corner, &centre) })
        } else {
            Value::Null
        };
        let pick = |better: &dyn Fn(f64, f64) -> bool| -> Option<[usize; 2]> {
            let mut best: Option<usize> = None;
            for &c in &face {
                if best.map_or(true, |b| better(pooled[c], pooled[b])) {
                    best = Some(c);
                }
            }
            best.map(|c| [c % nu, c / nu])
        };
        let m = self.metrics(q);
        let stop = self.target.map(|(metric, want)| {
            let se = match self.stopped {
                Some((_, se)) => Some(se),
                None => match metric {
                    Metric::Cv => m.cv,
                    Metric::MinMax => m.min_max,
                }
                .and_then(|e| e.stderr),
            };
            json!({
                "metric": metric.name(),
                "target": want,
                "stderr": se,
                "reached": se.is_some_and(|s| s <= want),
                "passes": q,
                "rays": rays,
                "rayLimit": limit,
            })
        });
        let cell = |c: usize, v: Value| if self.area[c] > 0.0 { v } else { Value::Null };
        json!({
            "detector": scene.detectors[self.det].common.id.0,
            "label": scene.detectors[self.det].common.label,
            "hits": d.hits,
            "cells": self.cells,
            "regionMm": self.region,
            "regionAskedMm": self.asked,
            "cellSizeMm": [
                (self.region[0][1] - self.region[0][0]) / nu as f64,
                (self.region[1][1] - self.region[1][0]) / nv as f64,
            ],
            "irradiance": {
                "value": (0..nu * nv).map(|c| cell(c, json!(pooled[c]))).collect::<Vec<_>>(),
                "stderr": (0..nu * nv).map(|c| cell(c, json!(cell_se[c]))).collect::<Vec<_>>(),
                "litPasses": (0..nu * nv).map(|c| cell(c, json!(lit[c]))).collect::<Vec<_>>(),
            },
            "mean": mean,
            "minMax": m.min_max,
            "cv": m.cv,
            "falloff": falloff,
            "minCell": pick(&|a, b| a < b),
            "maxCell": pick(&|a, b| a > b),
            "cellsOnFace": face.len(),
            "cellsDark": dark,
            "cellsSparse": sparse,
            "medianCellRelStderr": median_rel,
            "stop": stop,
        })
    }

    /// The centre cells (the middle one, or the two or four about the middle
    /// of an even count), the border ring and the four corners, on the face.
    fn falloff_groups(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let [nu, nv] = self.cells;
        let mid = |n: usize| {
            if n % 2 == 1 {
                vec![n / 2]
            } else {
                vec![n / 2 - 1, n / 2]
            }
        };
        let on = |c: &usize| self.area[*c] > 0.0;
        let mut centre: Vec<usize> = Vec::new();
        for j in mid(nv) {
            for i in mid(nu) {
                centre.push(j * nu + i);
            }
        }
        let edge: Vec<usize> = (0..nu * nv)
            .filter(|c| {
                let (i, j) = (c % nu, c / nu);
                i == 0 || j == 0 || i == nu - 1 || j == nv - 1
            })
            .filter(on)
            .collect();
        let mut corner = vec![0, nu - 1, (nv - 1) * nu, nv * nu - 1];
        corner.dedup();
        centre.retain(on);
        corner.retain(on);
        (centre, edge, corner)
    }
}

/// Call `f(pixel, cell)` for every pixel of the traced grid in the region,
/// in pixel order.
fn for_each_pixel(u: &Axis, v: &Axis, cells: [usize; 2], mut f: impl FnMut(usize, usize)) {
    let [nu, nv] = cells;
    for j in 0..nv * v.per_cell {
        let row = (v.start + j) * u.fine + u.start;
        let cell_row = (j / v.per_cell) * nu;
        for i in 0..nu * u.per_cell {
            f(row + i, cell_row + i / u.per_cell);
        }
    }
}

/// The noise-free CV: the variance across cells is the mean product of each
/// cell's deviation in one half with its deviation in the other, over the
/// product of the halves' means. Its error is first-order, each half's passes
/// moving it through the other half's deviations.
///
/// The CV is the root of that, 0 when noise leaves it at or below 0; its
/// standard error is the distance from there to the root of the value plus
/// one standard error, so a CV of 0 reads 0 with the noise it could hide.
fn cv(
    face: &[usize],
    even: &[&Vec<f64>],
    odd: &[&Vec<f64>],
    a: &[f64],
    b: &[f64],
) -> Option<Estimate> {
    let n = face.len() as f64;
    let mean = |x: &[f64]| face.iter().map(|&c| x[c]).sum::<f64>() / n;
    let (ma, mb) = (mean(a), mean(b));
    if ma <= 0.0 || mb <= 0.0 {
        return None;
    }
    let s = face.iter().map(|&c| (a[c] - ma) * (b[c] - mb)).sum::<f64>() / n;
    let c2 = s / (ma * mb);
    // d c2 / d (a cell of one half), per pass of that half: the other half's
    // deviation over both means, less c2 over this half's mean, over n.
    let z = |half: &[&Vec<f64>], other: &[f64], mo: f64, mh: f64| -> Vec<f64> {
        half.iter()
            .map(|p| {
                face.iter()
                    .map(|&c| p[c] * ((other[c] - mo) / (mh * mo) - c2 / mh))
                    .sum::<f64>()
                    / n
            })
            .collect()
    };
    let se_a = standard_error(&z(even, b, mb, ma))?;
    let se_b = standard_error(&z(odd, a, ma, mb))?;
    let se2 = (se_a * se_a + se_b * se_b).sqrt();
    let value = c2.max(0.0).sqrt();
    Some(Estimate {
        value,
        stderr: Some((c2.max(0.0) + se2).sqrt() - value),
    })
}

/// Cross-read min/max: the least and the most lit cells picked in one half,
/// their ratio read in the other, both ways round, and the mean taken. Its
/// error is first-order and conditional on the cells picked.
fn min_max(
    face: &[usize],
    even: &[&Vec<f64>],
    odd: &[&Vec<f64>],
    a: &[f64],
    b: &[f64],
) -> Option<Estimate> {
    let extremes = |x: &[f64]| -> (usize, usize) {
        let (mut lo, mut hi) = (face[0], face[0]);
        for &c in face {
            if x[c] < x[lo] {
                lo = c;
            }
            if x[c] > x[hi] {
                hi = c;
            }
        }
        (lo, hi)
    };
    // Picked from `pick`, read in `read` (its passes and their mean).
    let one = |pick: &[f64], read: &[&Vec<f64>], mean: &[f64]| -> Option<(f64, f64)> {
        let (lo, hi) = extremes(pick);
        let top = mean[hi];
        if top <= 0.0 {
            return None;
        }
        let r = mean[lo] / top;
        let z: Vec<f64> = read.iter().map(|p| (p[lo] - r * p[hi]) / top).collect();
        Some((r, standard_error(&z)?))
    };
    let (r1, s1) = one(a, odd, b)?;
    let (r2, s2) = one(b, even, a)?;
    Some(Estimate {
        value: 0.5 * (r1 + r2),
        stderr: Some(0.5 * (s1 * s1 + s2 * s2).sqrt()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_whole_face_is_its_own_grid_and_a_region_snaps_to_pixels() {
        // A tile a twentieth of the face, 16 cells across, starting on a
        // pixel edge: 320 pixels put every edge where it was asked.
        let w = 20.0;
        let a = snap(w, -10.0 + 3.0 * 1.0, -10.0 + 4.0 * 1.0, 16);
        assert_eq!(a.fine % 320, 0, "{a:?}");
        let p = w / a.fine as f64;
        assert!((a.start as f64 * p - 3.0).abs() < 1e-12);
        assert!(((a.start + 16 * a.per_cell) as f64 * p - 4.0).abs() < 1e-12);
        // An awkward region lands within a thousandth of a cell, or as close as
        // 1024 pixels allow.
        let b = snap(10.0, -1.2345, 2.71, 7);
        let p = 10.0 / b.fine as f64;
        let cell = (2.71 + 1.2345) / 7.0;
        assert!((b.start as f64 * p - (5.0 - 1.2345)).abs() <= 0.5 * p.max(1e-3 * cell));
        assert!(b.fine <= MAX_FINE);
    }

    #[test]
    fn checkpoints_are_powers_of_two_from_eight() {
        let at: Vec<usize> = (1..=64).filter(|&q| is_checkpoint(q)).collect();
        assert_eq!(at, [8, 16, 32, 64]);
    }
}
