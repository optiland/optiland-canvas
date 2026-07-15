//! The non-sequential reference tracer (§4.2).
//!
//! This is the **source of truth**: CPU, f64, exact. It is deliberately not the
//! fast path — any number the UI displays must come from here (§4.1). The
//! algorithm follows the spec exactly:
//!
//! 1. BVH over the scene; nearest hit.
//! 2. Media on both sides are resolved by containment + priority (§4.2), never
//!    a medium stack.
//! 3. Fresnel split: spawn both transmitted and reflected rays; TIR ⇒ R = 1.
//! 4. Beer–Lambert absorption over the traversed path.
//! 5. Terminate on: detector hit, bounds exit, `flux < ε`, or `depth > D`.

use oc_core::{Material, Ray, Region, Scene, Vec2};

use crate::bvh::Bvh;
use crate::fresnel::{beer_lambert, reflect, reflectance, refract};

/// Epsilon offsets. Scenes are assumed to be at a mm-ish scale; these are safe
/// for that range and are the consistent offsets §10.3 calls for.
const EPS_ORIGIN: f64 = 1e-7;
const EPS_MEDIUM: f64 = 1e-6;

/// Tracer configuration. `max_depth` and `flux_eps` are the two knobs the UI
/// exposes to trade completeness against speed (§4.2).
#[derive(Debug, Clone, Copy)]
pub struct TraceConfig {
    /// Hard depth cap `D`.
    pub max_depth: u32,
    /// Flux threshold `ε`, as a fraction of each ray's launch flux.
    pub flux_eps: f64,
    /// Spawn weak Fresnel back-reflections (ghosts) at transmitting interfaces.
    /// TIR and mirror reflections are always followed regardless of this flag.
    pub ghosts: bool,
    /// Ghost reflection-generation cap `G` (§4.2): how many nested Fresnel
    /// back-reflections to follow. The primary transmitted path is not counted —
    /// its length is the system's surface count, bounded only by `max_depth`.
    pub max_ghost_bounces: u32,
    /// Russian-roulette flux fraction below which ghost branches are randomly
    /// killed (unbiased). `0.0` disables RR — the default for the *reference*
    /// tracer, which must be deterministic.
    pub rr_cutoff: f64,
    /// Distance a ray travels before it is declared to have left the scene.
    pub bounds: f64,
    /// Global safety cap on total ray branches per launch ray.
    pub max_branches: u64,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            max_depth: 24,
            flux_eps: 1e-4,
            ghosts: true,
            // The reference tracer is the exact one, so by default it follows
            // ghosts as far as the depth cap allows (energy-conserving). The
            // interactive path overrides this with a small, user-set cap (§4.2).
            max_ghost_bounces: 64,
            rr_cutoff: 0.0,
            bounds: 1.0e4,
            max_branches: 100_000,
        }
    }
}

impl TraceConfig {
    /// The interactive preset (§4.1): shallow, cheap, approximate. (Semantics
    /// only — the fast f32/GPU path lives in `oc-gpu`.)
    pub fn interactive() -> Self {
        Self {
            max_depth: 8,
            ghosts: false,
            ..Self::default()
        }
    }
}

/// A traced ray segment, for rendering (flux-weighted alpha, §5.5).
///
/// `opl0`/`opl1` are the cumulative optical path length (Σ nᵢ·dᵢ, in mm) at the
/// segment's start and end — the propagation "clock" the wavefront scrub reads
/// (§5.5). Since propagation time is OPL/c, clipping every segment at a common
/// OPL draws the wavefront at that instant.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub a: Vec2,
    pub b: Vec2,
    pub flux: f64,
    pub lambda_um: f64,
    pub opl0: f64,
    pub opl1: f64,
}

/// Accumulated detector readout (§3.3).
#[derive(Debug, Clone)]
pub struct DetectorResult {
    /// Irradiance histogram along the segment.
    pub bins: Vec<f64>,
    /// Number of ray hits.
    pub hits: u64,
    /// Total flux collected.
    pub total_flux: f64,
    /// Flux-weighted sum of position `u ∈ [0,1]` (divide by `total_flux` for
    /// the centroid).
    pub flux_u: f64,
    /// Flux-weighted sum of optical path length.
    pub flux_opl: f64,
}

impl DetectorResult {
    fn new(bins: usize) -> Self {
        Self {
            bins: vec![0.0; bins.max(1)],
            hits: 0,
            total_flux: 0.0,
            flux_u: 0.0,
            flux_opl: 0.0,
        }
    }

    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    fn merge(&mut self, other: &DetectorResult) {
        for (a, b) in self.bins.iter_mut().zip(&other.bins) {
            *a += *b;
        }
        self.hits += other.hits;
        self.total_flux += other.total_flux;
        self.flux_u += other.flux_u;
        self.flux_opl += other.flux_opl;
    }

    /// Flux-weighted centroid position `∈ [0,1]`, or `None` with no hits.
    pub fn centroid_u(&self) -> Option<f64> {
        (self.total_flux > 0.0).then(|| self.flux_u / self.total_flux)
    }

    /// Mean optical path length of collected rays, or `None` with no hits.
    pub fn mean_opl(&self) -> Option<f64> {
        (self.total_flux > 0.0).then(|| self.flux_opl / self.total_flux)
    }
}

/// The result of a full trace.
#[derive(Debug, Clone)]
pub struct TraceResult {
    pub segments: Vec<Segment>,
    pub detectors: Vec<DetectorResult>,
    pub branches: u64,
}

/// Trace an entire scene. Parallel over launch rays (rayon).
pub fn trace(scene: &Scene, cfg: &TraceConfig) -> TraceResult {
    let boxes: Vec<_> = scene
        .elements
        .iter()
        .map(|e| e.aabb().padded(1e-6))
        .collect();
    let bvh = Bvh::build(&boxes);

    let launch: Vec<_> = scene.sources.iter().flat_map(|s| s.emit()).collect();
    let ndet = scene.detectors.len();

    let identity = || Local {
        segments: Vec::new(),
        detectors: scene
            .detectors
            .iter()
            .map(|d| DetectorResult::new(d.bins))
            .collect(),
        branches: 0,
    };

    // Parallel over launch rays where rayon is available (native); serial
    // otherwise (e.g. wasm32 without threads — the browser reference path).
    #[cfg(feature = "parallel")]
    let merged = {
        use rayon::prelude::*;
        launch
            .par_iter()
            .fold(identity, |mut acc, er| {
                trace_one(scene, &bvh, cfg, *er, &mut acc);
                acc
            })
            .reduce(identity, |mut a, b| {
                a.segments.extend(b.segments);
                for (x, y) in a.detectors.iter_mut().zip(&b.detectors) {
                    x.merge(y);
                }
                a.branches += b.branches;
                a
            })
    };
    #[cfg(not(feature = "parallel"))]
    let merged = {
        let mut acc = identity();
        for er in &launch {
            trace_one(scene, &bvh, cfg, *er, &mut acc);
        }
        acc
    };

    debug_assert_eq!(merged.detectors.len(), ndet);
    TraceResult {
        segments: merged.segments,
        detectors: merged.detectors,
        branches: merged.branches,
    }
}

struct Local {
    segments: Vec<Segment>,
    detectors: Vec<DetectorResult>,
    branches: u64,
}

struct RayWork {
    ray: Ray,
    lambda: f64,
    flux: f64,
    depth: u32,
    /// Ghost reflection generations in this branch's ancestry (§4.2). Only a
    /// Fresnel back-reflection increments it; the primary path keeps its parent's.
    gdepth: u32,
    /// Accumulated optical path length so far.
    opl: f64,
}

/// Deterministic per-launch-ray PRNG (xorshift64) for Russian roulette. Seeded
/// from ray geometry so results are reproducible run to run.
struct Rng(u64);
impl Rng {
    fn seed(v: f64, w: f64) -> Self {
        let s = (v.to_bits() ^ (w.to_bits().rotate_left(17))) | 1;
        Rng(s)
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // Top 53 bits -> [0,1).
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Trace one launch ray and its entire branch tree into `acc`.
fn trace_one(
    scene: &Scene,
    bvh: &Bvh,
    cfg: &TraceConfig,
    er: oc_core::EmittedRay,
    acc: &mut Local,
) {
    let min_flux = cfg.flux_eps * er.flux;
    let mut rng = Rng::seed(er.ray.origin.x + er.lambda_um, er.ray.dir.y);

    let mut stack: Vec<RayWork> = Vec::with_capacity(32);
    stack.push(RayWork {
        ray: er.ray,
        lambda: er.lambda_um,
        flux: er.flux,
        depth: 0,
        gdepth: 0,
        opl: 0.0,
    });

    let mut branches = 0u64;
    while let Some(w) = stack.pop() {
        branches += 1;
        if branches > cfg.max_branches {
            break;
        }
        if w.depth > cfg.max_depth || w.flux < min_flux {
            continue;
        }

        // Medium this ray currently travels through (for Beer–Lambert + n1).
        let start_region = scene.region_at(w.ray.origin + EPS_MEDIUM * w.ray.dir);
        let n_here = scene.region_index(start_region, w.lambda);
        let alpha = region_absorption(scene, start_region, w.lambda);

        // Nearest element edge hit.
        let mut best_t = cfg.bounds;
        let mut best: Option<oc_core::ElementHit> = None;
        let mut best_elem = usize::MAX;
        bvh.candidates(&w.ray, EPS_ORIGIN, best_t, &mut |ei| {
            scene.elements[ei].intersect(&w.ray, EPS_ORIGIN, &mut |h| {
                if h.t < best_t {
                    best_t = h.t;
                    best = Some(h);
                    best_elem = ei;
                }
            });
        });

        // Nearest detector hit (detectors terminate the branch).
        let mut det_hit: Option<(usize, f64, f64)> = None; // (index, t, u)
        for (di, d) in scene.detectors.iter().enumerate() {
            if let Some((t, u)) = d.intersect(&w.ray, EPS_ORIGIN) {
                if t < best_t && det_hit.map_or(true, |(_, pt, _)| t < pt) {
                    det_hit = Some((di, t, u));
                }
            }
        }

        // Detector wins if closer than any element.
        if let Some((di, t, u)) = det_hit {
            if best.is_none() || t < best_t {
                let end = w.ray.at(t);
                let flux = w.flux * beer_lambert(alpha, t);
                acc.segments.push(Segment {
                    a: w.ray.origin,
                    b: end,
                    flux,
                    lambda_um: w.lambda,
                    opl0: w.opl,
                    opl1: w.opl + n_here * t,
                });
                record_detector(acc, di, u, flux, w.opl + n_here * t);
                continue;
            }
        }

        let Some(hit) = best else {
            // No hit: ray leaves the scene.
            acc.segments.push(Segment {
                a: w.ray.origin,
                b: w.ray.at(cfg.bounds),
                flux: w.flux,
                lambda_um: w.lambda,
                opl0: w.opl,
                opl1: w.opl + n_here * cfg.bounds,
            });
            continue;
        };

        // Attenuate over the traversed path, then record the segment.
        let flux_at_hit = w.flux * beer_lambert(alpha, hit.t);
        let opl_at_hit = w.opl + n_here * hit.t;
        acc.segments.push(Segment {
            a: w.ray.origin,
            b: hit.point,
            flux: flux_at_hit,
            lambda_um: w.lambda,
            opl0: w.opl,
            opl1: opl_at_hit,
        });

        // Orient the normal to oppose the incident direction.
        let mut n = hit.normal;
        if n.dot(w.ray.dir) > 0.0 {
            n = -n;
        }

        // Regions on both sides (§4.2): near side is +n, far side is −n.
        let far_region = scene.region_at(hit.point - EPS_MEDIUM * n);
        let surf = scene.elements[best_elem].surface.for_edge(hit.edge_index);
        let elem_mat = &scene.elements[best_elem].material;

        // Absorbing surface / entering an absorber: terminate.
        let far_is_absorber = matches!(far_region, Region::In(e) if matches!(e.material, Material::Absorber))
            || matches!(elem_mat, Material::Absorber);
        if surf.absorb || far_is_absorber {
            continue;
        }

        let cos_i = -n.dot(w.ray.dir); // in [0,1]

        // Reflective interface: element is a mirror, edge forces mirror, or the
        // far side is a mirror body.
        let mirror_reflectivity = mirror_reflectivity(elem_mat, &far_region, &surf);
        if let Some(refl) = mirror_reflectivity {
            let dir_r = reflect(w.ray.dir, n);
            push_branch(
                &mut stack,
                cfg,
                &mut rng,
                hit.point,
                dir_r,
                w.lambda,
                flux_at_hit * refl,
                w.depth + 1,
                w.gdepth,
                opl_at_hit,
                min_flux,
                true, // primary path
            );
            continue;
        }

        // Dielectric interface. n1 = medium here, n2 = far medium.
        let n1 = n_here;
        let n2 = scene.region_index(far_region, w.lambda);

        match refract(w.ray.dir, n, n1, n2) {
            None => {
                // Total internal reflection — R = 1, primary path.
                let dir_r = reflect(w.ray.dir, n);
                push_branch(
                    &mut stack,
                    cfg,
                    &mut rng,
                    hit.point,
                    dir_r,
                    w.lambda,
                    flux_at_hit,
                    w.depth + 1,
                    w.gdepth,
                    opl_at_hit,
                    min_flux,
                    true,
                );
            }
            Some(dir_t) => {
                let r = (reflectance(cos_i, n1, n2) * surf.ar_residual).clamp(0.0, 1.0);
                let t = 1.0 - r;
                // Transmitted branch (primary).
                push_branch(
                    &mut stack,
                    cfg,
                    &mut rng,
                    hit.point,
                    dir_t,
                    w.lambda,
                    flux_at_hit * t,
                    w.depth + 1,
                    w.gdepth,
                    opl_at_hit,
                    min_flux,
                    true,
                );
                // Reflected branch (ghost) — only if ghosts are enabled and we
                // are within the ghost-generation budget (§4.2).
                if cfg.ghosts && r > 0.0 && w.gdepth < cfg.max_ghost_bounces {
                    let dir_r = reflect(w.ray.dir, n);
                    push_branch(
                        &mut stack,
                        cfg,
                        &mut rng,
                        hit.point,
                        dir_r,
                        w.lambda,
                        flux_at_hit * r,
                        w.depth + 1,
                        w.gdepth + 1,
                        opl_at_hit,
                        min_flux,
                        false,
                    );
                }
            }
        }
    }

    acc.branches += branches;
}

/// Push a new branch, applying epsilon offset, flux/depth culling, and — for
/// non-primary (ghost) branches — Russian roulette.
#[allow(clippy::too_many_arguments)]
fn push_branch(
    stack: &mut Vec<RayWork>,
    cfg: &TraceConfig,
    rng: &mut Rng,
    point: Vec2,
    dir: Vec2,
    lambda: f64,
    mut flux: f64,
    depth: u32,
    gdepth: u32,
    opl: f64,
    min_flux: f64,
    primary: bool,
) {
    if depth > cfg.max_depth || flux < min_flux {
        return;
    }
    // Russian roulette on faint ghost branches: keep unbiased by scaling up
    // survivors (§4.2). Never applied to primary paths.
    if !primary && cfg.rr_cutoff > 0.0 && flux < cfg.rr_cutoff {
        let p_survive = (flux / cfg.rr_cutoff).clamp(0.0, 1.0);
        if rng.next_f64() > p_survive {
            return;
        }
        flux /= p_survive;
    }
    stack.push(RayWork {
        ray: Ray::new(point + EPS_ORIGIN * dir, dir),
        lambda,
        flux,
        depth,
        gdepth,
        opl,
    });
}

/// Reflectivity if this interface reflects (mirror element / edge / far body),
/// else `None` (a dielectric interface).
fn mirror_reflectivity(
    elem_mat: &Material,
    far_region: &Region<'_>,
    surf: &oc_core::SurfaceSpec,
) -> Option<f64> {
    if surf.mirror {
        return Some(surf.reflectivity);
    }
    if let Material::Mirror { reflectivity } = elem_mat {
        return Some(*reflectivity);
    }
    if let Region::In(e) = far_region {
        if let Material::Mirror { reflectivity } = e.material {
            return Some(reflectivity);
        }
    }
    None
}

/// Volume absorption coefficient of a region (Beer–Lambert). Seeded materials
/// are transparent (α = 0); this is the hook colored/absorbing glass plugs into.
fn region_absorption(_scene: &Scene, _region: Region<'_>, _lambda: f64) -> f64 {
    0.0
}

fn record_detector(acc: &mut Local, di: usize, u: f64, flux: f64, opl: f64) {
    let det = &mut acc.detectors[di];
    det.hits += 1;
    det.total_flux += flux;
    det.flux_u += u * flux;
    det.flux_opl += opl * flux;
    let nb = det.bins.len();
    let bin = ((u * nb as f64) as usize).min(nb - 1);
    det.bins[bin] += flux;
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::{Isometry2, Source, SourceKind, Spectrum, LAMBDA_D};

    #[test]
    fn empty_scene_ray_exits() {
        let mut scene = Scene::new();
        scene.sources.push(Source {
            kind: SourceKind::Point { half_angle: 0.0 },
            transform: Isometry2::from_pos(Vec2::ZERO),
            spectrum: Spectrum::Monochromatic(LAMBDA_D),
            flux: 1.0,
            rays: 1,
        });
        let res = trace(&scene, &TraceConfig::default());
        assert_eq!(res.segments.len(), 1);
        // Segment runs to the bounds distance.
        let s = res.segments[0];
        assert!((s.b - s.a).length() > 1000.0);
    }
}
