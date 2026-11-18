//! `fluorescence`: light a dye absorbs and emits again, traced in two coupled
//! stages through the real optics.
//!
//! **The physics.** Excitation light enters a dye body; the dye absorbs it
//! by Beer–Lambert along the body's own absorption, `α(λ) = ln 10 · ε(λ) · c`
//! per mm; every watt it absorbs gives `quantumYield` watts of emission, at
//! the point of absorption, the same in every direction over 4π, unpolarized,
//! with the dye's emission spectrum; and that light is traced on through the
//! optics like any other.
//!
//! **Two stages.** Linear fluorescence is a sum over the points of
//! absorption: the light emitted at a point is proportional to the power
//! absorbed there, and nothing about how it is traced on depends on how the
//! excitation got there. So, in expectation, tracing the excitation and
//! re-emitting at every absorption in one trace gives what these two stages
//! give:
//!
//! 1. **Excitation.** Each source of the scene is traced alone (so what each
//!    leaks onto a detector, and what each puts into each dye, is known by
//!    source), with the kernel recording where each dye body absorbs
//!    ([`oc_trace::TraceConfig3::deposit_bodies`]). Each block's bulk
//!    deposits in a dye are taken out of the block as it comes in
//!    ([`oc_trace::TracePart3::take_deposits`]) and binned, pass by pass, on
//!    a grid of voxels over the body's bounding box.
//! 2. **Emission.** Once every excitation block is in, each pass of each dye
//!    becomes a scene of its own: the scene with its sources replaced by one
//!    point source per lit voxel, at the voxel's flux-weighted centroid,
//!    uniform over the whole sphere (a cone of half-angle π), with the dye's
//!    emission spectrum and a flux of the voxel's absorbed power times the
//!    quantum yield. A point inside a body starts in that body's medium, so
//!    the dye's own absorption takes its share of the emission on the way
//!    out: the inner-filter effect, in both stages.
//!
//! No transport changes: the emission is traced from ordinary scene sources.
//!
//! **Statistics.** Emission pass `p` is built from excitation pass `p`'s own
//! deposits, so each emission pass is an independent estimate of the whole
//! chain, and the batch-means standard error of anything emitted carries the
//! excitation's noise as well as the emission's.
//!
//! **Binning.** The only approximation the stages add is where in its voxel
//! the light is emitted: at the centroid of the absorption, not spread over
//! where each watt was taken. The total emitted is exact (each voxel emits
//! its own absorbed power); only where it starts moves, by less than a voxel.
//!
//! **Ignored.** Emission absorbed again by a dye is lost, not re-emitted;
//! dipole orientation and polarization memory (every emission is isotropic
//! and unpolarized); saturation and photobleaching (the response is linear);
//! and anything particular to emission close to an interface.

use std::collections::BTreeMap;

use oc_core::geom3::Intersect3;
use oc_core::id::SourceId;
use oc_core::scene::Spectrum;
use oc_core::scene3::{
    AbsorptionModel, AngularDistribution3, SamplingSequence, Scene3, Source3, SourceSampling,
    SpatialDistribution3,
};
use oc_core::vec3::{Isometry3, Vec3};
use oc_trace::result3::spectral_bin;
use oc_trace::{
    Approximation, CapabilityIssue, DepositLayer, DetectorResult3, EnergyLedger, ObjectRef3,
    TracePart3, TraceResult3,
};
use serde_json::{json, Value};

use crate::request::FluorescenceParams;
use crate::stats::Estimate;

/// Voxels along each axis of a dye's bounding box when the request names
/// none.
pub const DEFAULT_VOXELS: u32 = 16;

/// The most voxels a dye's grid may have, and along one axis.
pub const MAX_VOXELS: u64 = 1 << 16;
const MAX_AXIS: u32 = 1024;

/// The emission share of the rays when the request names none.
const DEFAULT_EMISSION_SHARE: f64 = 0.5;

fn bad(what: String) -> String {
    format!("fluorescence: {what}")
}

/// The share of the rays the emission takes, checked.
pub(crate) fn emission_share(p: &FluorescenceParams) -> Result<f64, String> {
    match p.emission_share {
        None => Ok(DEFAULT_EMISSION_SHARE),
        Some(s) if s.is_finite() && s > 0.0 && s < 1.0 => Ok(s),
        Some(s) => Err(bad(format!(
            "emissionShare must be strictly between 0 and 1, not {s}"
        ))),
    }
}

/// A dye, checked against the scene, and the grid its absorption is binned on.
#[derive(Debug, Clone)]
struct Dye {
    name: String,
    /// The body's index in the scene, and its id.
    body: usize,
    id: u64,
    qy: f64,
    /// `(λ µm, relative power)`.
    spectrum: Vec<(f64, f64)>,
    /// The grid: its low corner, a voxel's size, and the counts.
    lo: Vec3,
    size: Vec3,
    counts: [u32; 3],
}

impl Dye {
    /// The voxel `point` is in. A deposit is inside the body, so inside its
    /// box; one a rounding outside is put in the voxel at the edge.
    fn voxel(&self, point: Vec3) -> u32 {
        let axis = |x: f64, lo: f64, size: f64, n: u32| -> u32 {
            let i = ((x - lo) / size).floor();
            if i.is_finite() {
                i.clamp(0.0, (n - 1) as f64) as u32
            } else {
                0
            }
        };
        let [nx, ny, nz] = self.counts;
        let ix = axis(point.x, self.lo.x, self.size.x, nx);
        let iy = axis(point.y, self.lo.y, self.size.y, ny);
        let iz = axis(point.z, self.lo.z, self.size.z, nz);
        ix + nx * (iy + ny * iz)
    }
}

/// What one dye absorbed of one source's light: pooled over every block in
/// order, and pass by pass; and by wavelength bin.
#[derive(Debug, Clone, Default)]
struct Absorbed {
    pooled: f64,
    passes: Vec<f64>,
    /// Keyed by the bits of the bin's centre: pooled flux, pooled flux × λ,
    /// and pass by pass.
    lines: BTreeMap<u64, (f64, f64, Vec<f64>)>,
}

/// One voxel's sums in one pass: flux, and flux × x, y, z.
type Cell = [f64; 4];

/// One pass's sums on one detector, from one trace.
#[derive(Debug, Clone, Default)]
struct Reading {
    hits: u64,
    incident: f64,
    signal: f64,
    /// Keyed by the bits of the bin's centre: flux, and flux × λ.
    lines: BTreeMap<u64, (f64, f64)>,
}

impl Reading {
    fn add(&mut self, d: &DetectorResult3, lines: bool) {
        self.hits += d.hits;
        self.incident += d.total_incident;
        self.signal += d.total_signal;
        if lines {
            for s in &d.spectral {
                let e = self
                    .lines
                    .entry(s.lambda_um.to_bits())
                    .or_insert((0.0, 0.0));
                e.0 += s.flux;
                e.1 += s.lw;
            }
        }
    }
}

/// The per-pass sums of a `fluorescence` job, both stages.
#[derive(Debug, Clone)]
pub(crate) struct Fluorescence {
    dyes: Vec<Dye>,
    passes: usize,
    /// The emission rays each pass of each dye aims at.
    emission_rays: u64,
    seed: u64,
    /// `[dye][source]`.
    absorbed: Vec<Vec<Absorbed>>,
    /// `[dye][pass]`: each lit voxel's sums, until its emission is built.
    cells: Vec<Vec<BTreeMap<u32, Cell>>>,
    /// `[source][pass][detector]`: the excitation each detector received.
    excitation: Vec<Vec<Vec<Reading>>>,
    /// `[pass * dyes + dye]`: the launch rays and the lit voxels of each
    /// emission trace, once the excitation is in.
    plan: Option<Vec<(u64, usize)>>,
    /// `[dye][pass][detector]`: the emission each detector received.
    emission: Vec<Vec<Vec<Reading>>>,
    /// `[dye][pass]`: the flux each emission trace launched.
    launched: Vec<Vec<f64>>,
    /// What any emission trace was traced with, or could not be.
    approximated: Vec<Approximation>,
    rejected: Vec<CapabilityIssue>,
}

/// A seed apart from every other, for voxel `v` of emission trace `k`.
fn mix(seed: u64, k: u64, v: u64) -> u64 {
    let mut z = seed
        ^ (k + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (v + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The rays each lit voxel of `cells` gets of about `n`: its share of the
/// power, at least one, rounded to a square. Every ray of a trace then
/// carries about the same flux; a voxel's flux is its own, however many rays
/// carry it.
///
/// A square because a voxel's directions are a stratified `m × m` grid of
/// the sphere's unit square, one jittered point per cell: any other count
/// leaves the grid's last row part empty, and the directions it would have
/// held short of light.
fn allocate(cells: &BTreeMap<u32, Cell>, n: u64) -> Vec<(u32, Cell, u32)> {
    let total = cells.values().fold(0.0, |a, c| a + c[0]);
    cells
        .iter()
        .filter(|(_, c)| c[0] > 0.0)
        .map(|(&v, c)| {
            let m = (n as f64 * c[0] / total)
                .sqrt()
                .round()
                .clamp(1.0, 65_535.0) as u32;
            (v, *c, m * m)
        })
        .collect()
}

impl Fluorescence {
    pub(crate) fn new(
        scene: &Scene3,
        p: &FluorescenceParams,
        passes: u32,
        emission_rays: u64,
    ) -> Result<Self, String> {
        emission_share(p)?;
        if scene.detectors.is_empty() {
            return Err(bad("the scene has no detector to read".into()));
        }
        if p.dyes.is_empty() {
            return Err(bad("name at least one dye".into()));
        }
        let counts = p.voxels.map_or([DEFAULT_VOXELS; 3], |v| v.counts());
        if counts.iter().any(|&n| n == 0 || n > MAX_AXIS)
            || counts.iter().map(|&n| n as u64).product::<u64>() > MAX_VOXELS
        {
            return Err(bad(format!(
                "voxels must be 1 to {MAX_AXIS} along each axis and at most {MAX_VOXELS} in all, not {counts:?}"
            )));
        }
        let given: Vec<String> = p.dyes.iter().filter_map(|d| d.name.clone()).collect();
        for (k, n) in given.iter().enumerate() {
            if n.trim().is_empty() {
                return Err(bad("a dye's name is empty".into()));
            }
            if given[..k].contains(n) {
                return Err(bad(format!("two dyes are named {n:?}")));
            }
        }
        let mut dyes: Vec<Dye> = Vec::new();
        for spec in &p.dyes {
            let id = spec.body;
            let body = scene
                .bodies
                .iter()
                .position(|b| b.common.id.0 == id)
                .ok_or_else(|| bad(format!("no body has id {id}")))?;
            if dyes.iter().any(|d| d.body == body) {
                return Err(bad(format!("body {id} is named as two dyes")));
            }
            let b = &scene.bodies[body];
            if matches!(b.absorption, AbsorptionModel::None) {
                return Err(bad(format!(
                    "body {id} absorbs nothing: give it the dye's absorption, ln 10 · ε(λ) · c per mm"
                )));
            }
            let qy = spec.quantum_yield;
            if !(qy.is_finite() && qy > 0.0 && qy <= 1.0) {
                return Err(bad(format!(
                    "the quantum yield of the dye in body {id} must be in (0, 1], not {qy}"
                )));
            }
            let e = &spec.emission;
            let spectrum_bad = e.lambda_um.is_empty()
                || e.lambda_um.len() != e.weight.len()
                || e.lambda_um.iter().any(|l| !(l.is_finite() && *l > 0.0))
                || e.weight.iter().any(|w| !(w.is_finite() && *w >= 0.0))
                || e.weight.iter().sum::<f64>() <= 0.0;
            if spectrum_bad {
                return Err(bad(format!(
                    "the emission of the dye in body {id} must give as many weights as wavelengths, at least one, every wavelength above 0 µm and every weight at or above 0, not all 0"
                )));
            }
            let bb = b.geometry.aabb_world(&b.common.transform);
            let extent = bb.max - bb.min;
            if !(bb.min.is_finite() && bb.max.is_finite())
                || extent.x <= 0.0
                || extent.y <= 0.0
                || extent.z <= 0.0
            {
                return Err(bad(format!("body {id} has no finite bounds to bin on")));
            }
            let size = Vec3::new(
                extent.x / counts[0] as f64,
                extent.y / counts[1] as f64,
                extent.z / counts[2] as f64,
            );
            let name = match &spec.name {
                Some(n) => n.clone(),
                None => {
                    let base = if b.common.label.trim().is_empty() {
                        format!("body {id}")
                    } else {
                        b.common.label.clone()
                    };
                    let taken =
                        |n: &str| given.iter().any(|g| g == n) || dyes.iter().any(|d| d.name == n);
                    if taken(&base) {
                        format!("{base} ({id})")
                    } else {
                        base
                    }
                }
            };
            dyes.push(Dye {
                name,
                body,
                id,
                qy,
                spectrum: e
                    .lambda_um
                    .iter()
                    .copied()
                    .zip(e.weight.iter().copied())
                    .collect(),
                lo: bb.min,
                size,
                counts,
            });
        }
        let m = passes as usize;
        let (n_dyes, n_src, n_det) = (dyes.len(), scene.sources.len(), scene.detectors.len());
        let absorbed = Absorbed {
            passes: vec![0.0; m],
            ..Default::default()
        };
        // The emission's points are drawn afresh for every source seed.
        let seed = scene
            .sources
            .iter()
            .fold(0x005E_EDF1_u64, |a, s| mix(a, s.id.0, s.samples.seed));
        Ok(Self {
            dyes,
            passes: m,
            emission_rays: emission_rays.max(1),
            seed,
            absorbed: vec![vec![absorbed; n_src]; n_dyes],
            cells: vec![vec![BTreeMap::new(); m]; n_dyes],
            excitation: vec![vec![vec![Reading::default(); n_det]; m]; n_src],
            plan: None,
            emission: vec![vec![vec![Reading::default(); n_det]; m]; n_dyes],
            launched: vec![vec![0.0; m]; n_dyes],
            approximated: Vec::new(),
            rejected: Vec::new(),
        })
    }

    /// The scene indices `[lo, hi)` the excitation records deposits in: every
    /// dye's body, and any body between, whose deposits are dropped.
    pub(crate) fn deposit_bodies(&self) -> [usize; 2] {
        let lo = self.dyes.iter().map(|d| d.body).min().unwrap_or(0);
        let hi = self.dyes.iter().map(|d| d.body + 1).max().unwrap_or(0);
        [lo, hi]
    }

    /// Add one block of source `source`'s excitation trace to pass `pass`,
    /// taking its deposits out of the part. Blocks come in order, so each
    /// sum is added in one fixed order however the job was stepped.
    pub(crate) fn add_excitation_block(
        &mut self,
        source: usize,
        pass: usize,
        part: &mut TracePart3,
    ) {
        for d in part.take_deposits() {
            if d.layer != DepositLayer::Bulk {
                continue;
            }
            let ObjectRef3::Body(i, _) = d.object else {
                continue;
            };
            let Some(k) = self.dyes.iter().position(|y| y.body == i) else {
                continue;
            };
            let a = &mut self.absorbed[k][source];
            a.pooled += d.flux;
            a.passes[pass] += d.flux;
            let m = self.passes;
            let line = a
                .lines
                .entry(spectral_bin(d.lambda_um).to_bits())
                .or_insert_with(|| (0.0, 0.0, vec![0.0; m]));
            line.0 += d.flux;
            line.1 += d.flux * d.lambda_um;
            line.2[pass] += d.flux;
            let v = self.dyes[k].voxel(d.point);
            let c = self.cells[k][pass].entry(v).or_insert([0.0; 4]);
            c[0] += d.flux;
            c[1] += d.flux * d.point.x;
            c[2] += d.flux * d.point.y;
            c[3] += d.flux * d.point.z;
        }
        for (r, d) in self.excitation[source][pass]
            .iter_mut()
            .zip(part.detectors())
        {
            r.add(d, false);
        }
    }

    /// The emission traces, `passes × dyes` of them, in the order the job
    /// traces them: pass by pass, each dye in turn.
    pub(crate) fn emission_traces(&self) -> usize {
        self.passes * self.dyes.len()
    }

    /// The excitation is in: fix each emission trace's rays.
    pub(crate) fn plan(&mut self) {
        if self.plan.is_some() {
            return;
        }
        let n = self.dyes.len();
        let plan = (0..self.emission_traces())
            .map(|k| {
                let cells = &self.cells[k % n][k / n];
                let a = allocate(cells, self.emission_rays);
                (a.iter().map(|x| x.2 as u64).sum(), a.len())
            })
            .collect();
        self.plan = Some(plan);
    }

    /// The emission rays in all: those planned, or before the plan those
    /// aimed at.
    pub(crate) fn emission_rays_total(&self) -> u64 {
        match &self.plan {
            Some(p) => p.iter().map(|x| x.0).sum(),
            None => self.emission_rays * self.emission_traces() as u64,
        }
    }

    /// The rays of the emission traces before trace `k`.
    pub(crate) fn emission_rays_before(&self, k: usize) -> u64 {
        self.plan
            .as_ref()
            .map_or(0, |p| p[..k.min(p.len())].iter().map(|x| x.0).sum())
    }

    /// Emission trace `k`: `base` with its sources replaced by the lit voxels
    /// of that dye in that pass; `None` when the dye absorbed nothing in it.
    /// The voxels are dropped once the scene is built.
    pub(crate) fn emission_scene(&mut self, k: usize, base: &Scene3) -> Option<Scene3> {
        let n = self.dyes.len();
        let (dye, pass) = (k % n, k / n);
        let cells = std::mem::take(&mut self.cells[dye][pass]);
        let alloc = allocate(&cells, self.emission_rays);
        if alloc.is_empty() {
            return None;
        }
        let d = &self.dyes[dye];
        let spectrum = if d.spectrum.len() == 1 {
            Spectrum::Monochromatic(d.spectrum[0].0)
        } else {
            Spectrum::Continuum(d.spectrum.clone())
        };
        let mut scene = base.clone();
        scene.sources = alloc
            .into_iter()
            .map(|(v, c, rays)| {
                let at = Vec3::new(c[1] / c[0], c[2] / c[0], c[3] / c[0]);
                Source3 {
                    id: SourceId(v as u64 + 1),
                    transform: Isometry3::from_pos(at),
                    spatial: SpatialDistribution3::Point,
                    angular: AngularDistribution3::UniformConeSolidAngle {
                        half_angle_rad: std::f64::consts::PI,
                    },
                    spectrum: spectrum.clone(),
                    flux: d.qy * c[0],
                    samples: SourceSampling {
                        spatial_samples: 1,
                        angular_samples: rays,
                        seed: mix(self.seed, k as u64, v as u64),
                        sequence: SamplingSequence::Stratified,
                    },
                    window: Default::default(),
                }
            })
            .collect();
        Some(scene)
    }

    /// Add one block of emission trace `k`.
    pub(crate) fn add_emission_block(
        &mut self,
        k: usize,
        ledger: &EnergyLedger,
        detectors: &[DetectorResult3],
    ) {
        let n = self.dyes.len();
        let (dye, pass) = (k % n, k / n);
        self.launched[dye][pass] += ledger.launched;
        for (r, d) in self.emission[dye][pass].iter_mut().zip(detectors) {
            r.add(d, true);
        }
    }

    /// Emission trace `k` is in: keep what it was traced with.
    pub(crate) fn finish_emission(&mut self, total: &TraceResult3) {
        let cap = &total.capability_report;
        for a in &cap.approximated {
            if !self.approximated.contains(a) {
                self.approximated.push(a.clone());
            }
        }
        for r in &cap.rejected {
            if !self.rejected.contains(r) {
                self.rejected.push(r.clone());
            }
        }
    }

    /// What the emission traces were traced with, or could not be.
    pub(crate) fn reports(&self) -> (&[Approximation], &[CapabilityIssue]) {
        (&self.approximated, &self.rejected)
    }

    /// What `sampling` says of the emission.
    pub(crate) fn sampling(&self) -> Value {
        let n = self.dyes.len();
        let plan = self.plan.clone().unwrap_or_default();
        let rays: u64 = plan.iter().map(|x| x.0).sum();
        let lit: u64 = plan.iter().map(|x| x.1 as u64).sum();
        json!({
            "rays": rays,
            "raysPerPass": rays / self.passes.max(1) as u64,
            "traces": plan.len(),
            "voxelSources": lit,
            "dyes": n,
        })
    }

    /// The result: excitation values pooled from each source's merged trace
    /// `totals` (launch rays `rays`), emission values summed over the passes'
    /// traces, standard errors from the passes.
    pub(crate) fn result(&self, scene: &Scene3, totals: &[&TraceResult3], rays: &[u64]) -> Value {
        let m = self.passes;
        let k = m as f64;
        let scale = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| k * x).collect() };
        let add = |a: &mut Vec<f64>, b: &[f64]| {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        };
        let source_ids: Vec<u64> = scene.sources.iter().map(|s| s.id.0).collect();
        let detector_ids: Vec<u64> = scene.detectors.iter().map(|d| d.common.id.0).collect();
        let plan = self.plan.clone().unwrap_or_default();
        let n = self.dyes.len();

        let dyes: Vec<Value> = self
            .dyes
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut absorbed = 0.0;
                let mut absorbed_pp = vec![0.0; m];
                let by_source: Vec<Value> = self.absorbed[i]
                    .iter()
                    .zip(&source_ids)
                    .map(|(a, id)| {
                        absorbed += a.pooled;
                        add(&mut absorbed_pp, &a.passes);
                        let lines: Vec<Value> = a
                            .lines
                            .iter()
                            .map(|(key, (f, fl, pp))| {
                                json!({
                                    "lambdaUm": f64::from_bits(*key),
                                    "meanLambdaUm": if *f > 0.0 { fl / f } else { f64::from_bits(*key) },
                                    "absorbed": Estimate::new(*f, &scale(pp)),
                                })
                            })
                            .collect();
                        json!({
                            "source": id,
                            "absorbed": Estimate::new(a.pooled, &scale(&a.passes)),
                            "lines": lines,
                        })
                    })
                    .collect();
                let emitted = self.launched[i].iter().fold(0.0, |a, x| a + x);
                let (rays, lit) = (0..m).fold((0u64, 0u64), |(r, l), p| {
                    let x = plan.get(p * n + i).copied().unwrap_or((0, 0));
                    (r + x.0, l + x.1 as u64)
                });
                json!({
                    "name": d.name,
                    "body": d.id,
                    "label": scene.bodies[d.body].common.label,
                    "quantumYield": d.qy,
                    "absorbed": Estimate::new(absorbed, &scale(&absorbed_pp)),
                    "emitted": Estimate::new(emitted, &scale(&self.launched[i])),
                    "bySource": by_source,
                    "voxels": d.counts,
                    "voxelSizeMm": [d.size.x, d.size.y, d.size.z],
                    "voxelsLitPerPass": lit as f64 / m.max(1) as f64,
                    "emissionRays": rays,
                })
            })
            .collect();

        // [dye][detector]: emission, pooled and per pass.
        let emission = |f: &dyn Fn(&Reading) -> f64| -> Vec<Vec<(f64, Vec<f64>)>> {
            self.emission
                .iter()
                .map(|per_pass| {
                    (0..detector_ids.len())
                        .map(|j| {
                            let pp: Vec<f64> = per_pass.iter().map(|r| f(&r[j])).collect();
                            (pp.iter().fold(0.0, |a, x| a + x), scale(&pp))
                        })
                        .collect()
                })
                .collect()
        };
        let em_flux = emission(&|r| r.incident);
        let em_signal = emission(&|r| r.signal);
        // [source][detector]: excitation, pooled from the merged traces.
        let excitation = |pooled: &dyn Fn(&DetectorResult3) -> f64,
                          f: &dyn Fn(&Reading) -> f64|
         -> Vec<Vec<(f64, Vec<f64>)>> {
            totals
                .iter()
                .zip(&self.excitation)
                .map(|(t, per_pass)| {
                    (0..detector_ids.len())
                        .map(|j| {
                            let pp: Vec<f64> = per_pass.iter().map(|r| f(&r[j])).collect();
                            (pooled(&t.detectors[j]), scale(&pp))
                        })
                        .collect()
                })
                .collect()
        };
        let ex_flux = excitation(&|d| d.total_incident, &|r| r.incident);
        let ex_signal = excitation(&|d| d.total_signal, &|r| r.signal);
        // A column of a table, summed over its rows in order.
        let column = |t: &[Vec<(f64, Vec<f64>)>], j: usize| -> Estimate {
            let mut v = 0.0;
            let mut pp = vec![0.0; m];
            for row in t {
                v += row[j].0;
                add(&mut pp, &row[j].1);
            }
            Estimate::new(v, &pp)
        };
        let table = |t: &[Vec<(f64, Vec<f64>)>]| -> Vec<Vec<Estimate>> {
            t.iter()
                .map(|row| row.iter().map(|(v, pp)| Estimate::new(*v, pp)).collect())
                .collect()
        };

        let detectors: Vec<Value> = scene
            .detectors
            .iter()
            .enumerate()
            .map(|(j, det)| {
                // Every dye's emission by wavelength bin.
                let mut lines: BTreeMap<u64, (f64, f64, Vec<f64>)> = BTreeMap::new();
                let mut hits = 0;
                for per_pass in &self.emission {
                    for (p, r) in per_pass.iter().enumerate() {
                        hits += r[j].hits;
                        for (key, (f, fl)) in &r[j].lines {
                            let e = lines
                                .entry(*key)
                                .or_insert_with(|| (0.0, 0.0, vec![0.0; m]));
                            e.0 += f;
                            e.1 += fl;
                            e.2[p] += f;
                        }
                    }
                }
                let lines: Vec<Value> = lines
                    .iter()
                    .map(|(key, (f, fl, pp))| {
                        json!({
                            "lambdaUm": f64::from_bits(*key),
                            "meanLambdaUm": if *f > 0.0 { fl / f } else { f64::from_bits(*key) },
                            "flux": Estimate::new(*f, &scale(pp)),
                        })
                    })
                    .collect();
                let ex_hits: u64 = totals.iter().map(|t| t.detectors[j].hits).sum();
                json!({
                    "detector": det.common.id.0,
                    "label": det.common.label,
                    "emission": {
                        "flux": column(&em_flux, j),
                        "signal": column(&em_signal, j),
                        "hits": hits,
                        "lines": lines,
                    },
                    "excitation": {
                        "flux": column(&ex_flux, j),
                        "signal": column(&ex_signal, j),
                        "hits": ex_hits,
                    },
                })
            })
            .collect();

        let sources: Vec<Value> = source_ids
            .iter()
            .zip(totals)
            .zip(rays)
            .map(|((id, t), r)| json!({ "source": id, "rays": r, "launched": t.ledger.launched }))
            .collect();
        let names: Vec<&str> = self.dyes.iter().map(|d| d.name.as_str()).collect();
        json!({
            "dyes": dyes,
            "sources": sources,
            "detectors": detectors,
            "emission": {
                "dyes": names,
                "detectors": detector_ids,
                "flux": table(&em_flux),
                "signal": table(&em_signal),
            },
            "excitation": {
                "sources": source_ids,
                "detectors": detector_ids,
                "flux": table(&ex_flux),
                "signal": table(&ex_signal),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_go_to_voxels_by_power_and_every_lit_one_gets_one() {
        let mut cells = BTreeMap::new();
        cells.insert(3, [3.0, 0.0, 0.0, 0.0]);
        cells.insert(5, [1.0, 0.0, 0.0, 0.0]);
        cells.insert(9, [1e-9, 0.0, 0.0, 0.0]);
        cells.insert(11, [0.0, 0.0, 0.0, 0.0]);
        let a = allocate(&cells, 400);
        let rays: Vec<(u32, u32)> = a.iter().map(|x| (x.0, x.2)).collect();
        // 300 and 100 rays' worth: 17² and 10².
        assert_eq!(rays, vec![(3, 289), (5, 100), (9, 1)]);
    }
}
