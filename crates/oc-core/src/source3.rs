//! Turning a [`Source3`] into deterministic launch rays (§6.10).
//!
//! The reference tracer takes launch rays explicitly, so that transport can be
//! validated on hand-built geometry without a sampler in the way (see
//! [`Source3::emit`]'s callers). This module is the sampler that feeds it in
//! production: it expands a source's spatial extent, angular spread, and spectrum
//! into a set of [`EmittedRay3`], and it does so **deterministically** — the same
//! seed and configuration yield the same rays, bit for bit, every run (§3.4,
//! §6.10).
//!
//! # Frame
//!
//! A source emits about its local `+X`, the optical axis, with the transverse
//! aperture in the local `YZ` plane. So a collimated rectangle spreads its origins
//! across `YZ` and aims every ray along `+X`; a point cone keeps one origin and
//! spreads directions about `+X`. The source transform then places and aims the
//! whole bundle, exactly as a v1 source's `Isometry2` did in the plane.
//!
//! # Measure
//!
//! Each distribution samples its *natural* measure, so a stochastic estimate
//! converges to the physical integral:
//!
//! - an area source is uniform in area (`mm²`), including the disc via a
//!   concentric map that does not clump at the centre;
//! - [`AngularDistribution3::UniformConeSolidAngle`] is uniform in **solid angle**
//!   (steradians), not in polar angle — the two differ, and a fan that is uniform
//!   in angle is a different source;
//! - [`AngularDistribution3::LambertianHemisphere`] is cosine-weighted, so its
//!   radiance is constant with angle.
//!
//! The legacy [`AngularDistribution3::PlanarFanLegacy`] and
//! [`SpatialDistribution3::LineLegacy`] are the degenerate planar distributions v1
//! authored, kept so old files sample as they did, and marked as having measure
//! only in the plane (§6.10).

use crate::id::SourceId;
use crate::scene3::{AngularDistribution3, SamplingSequence, Source3, SpatialDistribution3};
use crate::vec3::{Ray3, Vec3};
use std::f64::consts::TAU;

/// One ray a source launches: its geometry, wavelength, and the share of the
/// source's flux it carries.
///
/// The 3D analogue of [`crate::scene::EmittedRay`]. The tracer wraps it into its
/// own `LaunchRay3`; keeping the sampler in `oc-core` lets the scene model own ray
/// generation without depending on the tracer crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmittedRay3 {
    pub ray: Ray3,
    pub lambda_um: f64,
    pub flux: f64,
    pub source: SourceId,
}

impl Source3 {
    /// Expand this source into its deterministic set of launch rays (§6.10).
    ///
    /// The ray count is `spatial_samples × angular_samples` per spectral line: a
    /// point source keeps `spatial_samples = 1` and spreads directions, a
    /// collimated source keeps `angular_samples = 1` and spreads origins, and a
    /// source with both extents takes the product. Flux is split evenly across the
    /// rays of each spectral line, weighted by the line, so the total emitted flux
    /// is `flux` regardless of sample count.
    pub fn emit(&self) -> Vec<EmittedRay3> {
        let spectral = self.spectrum.samples();
        let n_spatial = self.samples.spatial_samples.max(1) as usize;
        let n_angular = self.samples.angular_samples.max(1) as usize;

        // Two independent low-discrepancy streams, so the spatial and angular
        // patterns do not correlate. Stratified jitter is seeded from the source
        // seed plus a per-stream salt.
        let origins_local = spatial_samples(
            &self.spatial,
            n_spatial,
            self.samples.sequence,
            self.samples.seed,
        );
        let dirs_local = angular_samples(
            &self.angular,
            n_angular,
            self.samples.sequence,
            self.samples.seed ^ 0x9E37_79B9_7F4A_7C15,
        );

        let mut out = Vec::with_capacity(spectral.len() * n_spatial * n_angular);
        for (lambda, weight) in &spectral {
            let flux_each = self.flux * weight / (n_spatial * n_angular) as f64;
            for o_local in &origins_local {
                let origin = self.transform.transform_point(*o_local);
                for d_local in &dirs_local {
                    let dir = self.transform.rotate(*d_local);
                    // A degenerate direction (a zero-length legacy dir, say) would
                    // make no ray; skip it rather than emit a NaN.
                    let Ok(ray) = Ray3::new(origin, dir) else {
                        continue;
                    };
                    out.push(EmittedRay3 {
                        ray,
                        lambda_um: *lambda,
                        flux: flux_each,
                        source: self.id,
                    });
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Spatial sampling — local origins in the YZ aperture plane
// ---------------------------------------------------------------------------

fn spatial_samples(
    dist: &SpatialDistribution3,
    n: usize,
    seq: SamplingSequence,
    seed: u64,
) -> Vec<Vec3> {
    match dist {
        SpatialDistribution3::Point => vec![Vec3::ZERO; n],
        SpatialDistribution3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => unit_square(n, seq, seed)
            .into_iter()
            .map(|(u, v)| {
                Vec3::new(
                    0.0,
                    (2.0 * u - 1.0) * half_width_mm,
                    (2.0 * v - 1.0) * half_height_mm,
                )
            })
            .collect(),
        SpatialDistribution3::Disc { radius_mm } => unit_square(n, seq, seed)
            .into_iter()
            .map(|(u, v)| {
                let (dy, dz) = concentric_disc(u, v);
                Vec3::new(0.0, radius_mm * dy, radius_mm * dz)
            })
            .collect(),
        // v1 compatibility: a line along local +Y in the plane, no area.
        SpatialDistribution3::LineLegacy { half_width_mm } => unit_interval(n, seq, seed)
            .into_iter()
            .map(|u| Vec3::new(0.0, (2.0 * u - 1.0) * half_width_mm, 0.0))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Angular sampling — local directions about +X
// ---------------------------------------------------------------------------

fn angular_samples(
    dist: &AngularDistribution3,
    n: usize,
    seq: SamplingSequence,
    seed: u64,
) -> Vec<Vec3> {
    match dist {
        AngularDistribution3::Direction { local_dir } => {
            vec![local_dir.normalize_or_zero(); n]
        }
        AngularDistribution3::UniformConeSolidAngle { half_angle_rad } => {
            let cos_max = half_angle_rad.cos();
            unit_square(n, seq, seed)
                .into_iter()
                .map(|(u, v)| {
                    // Uniform in solid angle: cosθ uniform in [cos α, 1].
                    let cos_t = 1.0 - u * (1.0 - cos_max);
                    dir_from_cos(cos_t, TAU * v)
                })
                .collect()
        }
        AngularDistribution3::LambertianHemisphere => unit_square(n, seq, seed)
            .into_iter()
            .map(|(u, v)| {
                // Cosine-weighted: cosθ = √(1 − u), so radiance is constant.
                let cos_t = (1.0 - u).sqrt();
                dir_from_cos(cos_t, TAU * v)
            })
            .collect(),
        // v1 compatibility: an equal-angle fan in the local XY plane.
        AngularDistribution3::PlanarFanLegacy { half_angle_rad } => unit_interval(n, seq, seed)
            .into_iter()
            .map(|u| {
                let ang = (2.0 * u - 1.0) * half_angle_rad;
                Vec3::new(ang.cos(), ang.sin(), 0.0)
            })
            .collect(),
    }
}

/// A unit direction at polar angle `acos(cos_t)` from `+X` and azimuth `phi` in
/// the `YZ` plane.
fn dir_from_cos(cos_t: f64, phi: f64) -> Vec3 {
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    Vec3::new(cos_t, sin_t * phi.cos(), sin_t * phi.sin())
}

/// Shirley's concentric map: `[0,1]²` to the unit disc, area-preserving and
/// without the polar clumping a naive `(√r, θ)` map produces.
fn concentric_disc(u: f64, v: f64) -> (f64, f64) {
    let a = 2.0 * u - 1.0;
    let b = 2.0 * v - 1.0;
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0);
    }
    let (r, theta) = if a * a > b * b {
        (a, std::f64::consts::FRAC_PI_4 * (b / a))
    } else {
        (
            b,
            std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4 * (a / b),
        )
    };
    (r * theta.cos(), r * theta.sin())
}

// ---------------------------------------------------------------------------
// Low-discrepancy sequences (§6.10). Deterministic for a fixed seed.
// ---------------------------------------------------------------------------

/// `n` samples in `[0, 1)`.
fn unit_interval(n: usize, seq: SamplingSequence, seed: u64) -> Vec<f64> {
    match seq {
        SamplingSequence::Grid => (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect(),
        SamplingSequence::Stratified => {
            let mut rng = Rng::new(seed);
            (0..n)
                .map(|i| (i as f64 + rng.next_f64()) / n as f64)
                .collect()
        }
        SamplingSequence::Sobol => (0..n).map(|i| vdc_base2(i as u32 + 1)).collect(),
    }
}

/// `n` samples in `[0, 1)²`.
fn unit_square(n: usize, seq: SamplingSequence, seed: u64) -> Vec<(f64, f64)> {
    match seq {
        SamplingSequence::Grid => grid_2d(n),
        SamplingSequence::Stratified => {
            let mut rng = Rng::new(seed);
            let cols = (n as f64).sqrt().ceil().max(1.0) as usize;
            let rows = n.div_ceil(cols);
            let mut out = Vec::with_capacity(n);
            'fill: for r in 0..rows {
                for c in 0..cols {
                    if out.len() == n {
                        break 'fill;
                    }
                    let u = (c as f64 + rng.next_f64()) / cols as f64;
                    let v = (r as f64 + rng.next_f64()) / rows as f64;
                    out.push((u, v));
                }
            }
            out
        }
        // A (0,2)-sequence: van der Corput base 2 in the first dimension, the
        // classic Sobol second dimension in the other.
        SamplingSequence::Sobol => (0..n)
            .map(|i| {
                let k = i as u32 + 1;
                (vdc_base2(k), sobol_dim2(k))
            })
            .collect(),
    }
}

/// An even grid over `[0,1)²`, `⌈√n⌉` columns, filled row-major to exactly `n`.
fn grid_2d(n: usize) -> Vec<(f64, f64)> {
    let cols = (n as f64).sqrt().ceil().max(1.0) as usize;
    let rows = n.div_ceil(cols);
    let mut out = Vec::with_capacity(n);
    'fill: for r in 0..rows {
        for c in 0..cols {
            if out.len() == n {
                break 'fill;
            }
            out.push((
                (c as f64 + 0.5) / cols as f64,
                (r as f64 + 0.5) / rows as f64,
            ));
        }
    }
    out
}

/// Van der Corput radical inverse in base 2: reverse the bits of `i`.
fn vdc_base2(i: u32) -> f64 {
    i.reverse_bits() as f64 / 4_294_967_296.0
}

/// The second dimension of the Sobol' (0,2)-sequence (primitive polynomial
/// `x + 1`, all direction numbers one, applied in Gray-code order).
fn sobol_dim2(mut i: u32) -> f64 {
    let mut result = 0u32;
    let mut v = 1u32 << 31;
    while i != 0 {
        if i & 1 != 0 {
            result ^= v;
        }
        v ^= v >> 1;
        i >>= 1;
    }
    result as f64 / 4_294_967_296.0
}

/// A small deterministic PRNG (splitmix64) for stratified jitter. Seeded from the
/// source seed so a run is reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point.
        Rng(seed ^ 0xDEAD_BEEF_CAFE_F00D)
    }
    fn next_f64(&mut self) -> f64 {
        // splitmix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Spectrum;
    use crate::scene3::SourceSampling;
    use crate::vec3::Isometry3;

    fn source(
        spatial: SpatialDistribution3,
        angular: AngularDistribution3,
        samples: SourceSampling,
    ) -> Source3 {
        Source3 {
            id: SourceId(0),
            transform: Isometry3::IDENTITY,
            spatial,
            angular,
            spectrum: Spectrum::Monochromatic(0.5),
            flux: 1.0,
            samples,
        }
    }

    fn sampling(
        n_spatial: u32,
        n_angular: u32,
        seq: SamplingSequence,
        seed: u64,
    ) -> SourceSampling {
        SourceSampling {
            spatial_samples: n_spatial,
            angular_samples: n_angular,
            seed,
            sequence: seq,
        }
    }

    #[test]
    fn a_point_direction_source_launches_one_aimed_ray_per_sample() {
        let s = source(
            SpatialDistribution3::Point,
            AngularDistribution3::Direction { local_dir: Vec3::X },
            sampling(1, 4, SamplingSequence::Grid, 0),
        );
        let rays = s.emit();
        assert_eq!(rays.len(), 4);
        for r in &rays {
            assert!((r.ray.dir - Vec3::X).length() < 1e-15);
            assert!(r.ray.origin.length() < 1e-15);
        }
    }

    #[test]
    fn flux_is_conserved_across_samples_and_spectrum() {
        let mut s = source(
            SpatialDistribution3::Rectangle {
                half_width_mm: 2.0,
                half_height_mm: 1.0,
            },
            AngularDistribution3::Direction { local_dir: Vec3::X },
            sampling(16, 1, SamplingSequence::Grid, 0),
        );
        s.flux = 7.5;
        s.spectrum = Spectrum::Lines(vec![(0.45, 1.0), (0.55, 3.0)]);
        let rays = s.emit();
        let total: f64 = rays.iter().map(|r| r.flux).sum();
        assert!((total - 7.5).abs() < 1e-12, "total flux {total}");
    }

    #[test]
    fn a_collimated_rectangle_fills_its_area_and_stays_parallel() {
        let s = source(
            SpatialDistribution3::Rectangle {
                half_width_mm: 3.0,
                half_height_mm: 2.0,
            },
            AngularDistribution3::Direction { local_dir: Vec3::X },
            sampling(64, 1, SamplingSequence::Grid, 0),
        );
        let rays = s.emit();
        let mut mean = Vec3::ZERO;
        for r in &rays {
            assert!((r.ray.dir - Vec3::X).length() < 1e-15, "not parallel");
            assert!(r.ray.origin.x.abs() < 1e-15, "aperture is not in YZ");
            assert!(r.ray.origin.y.abs() <= 3.0 + 1e-12 && r.ray.origin.z.abs() <= 2.0 + 1e-12);
            mean += r.ray.origin;
        }
        mean /= rays.len() as f64;
        // A symmetric grid is centred on the source.
        assert!(mean.length() < 1e-9, "beam is not centred: {mean:?}");
    }

    #[test]
    fn a_cone_is_uniform_in_solid_angle() {
        // For a cone of half-angle α sampled uniformly in solid angle, the mean of
        // cosθ is (1 + cos α)/2. A fan uniform in *angle* would give a different
        // mean, so this is the test that the measure is right.
        let alpha = 0.5_f64;
        let s = source(
            SpatialDistribution3::Point,
            AngularDistribution3::UniformConeSolidAngle {
                half_angle_rad: alpha,
            },
            sampling(1, 4096, SamplingSequence::Sobol, 0),
        );
        let rays = s.emit();
        let mut sum_cos = 0.0;
        for r in &rays {
            // cosθ is the +X component of the (unit) direction.
            assert!(r.ray.dir.x >= alpha.cos() - 1e-9, "outside the cone");
            sum_cos += r.ray.dir.x;
        }
        let mean = sum_cos / rays.len() as f64;
        let expected = (1.0 + alpha.cos()) / 2.0;
        assert!(
            (mean - expected).abs() < 2e-3,
            "mean cosθ {mean}, expected {expected}"
        );
    }

    #[test]
    fn a_lambertian_hemisphere_is_cosine_weighted() {
        // Cosine-weighted sampling of the hemisphere has E[cosθ] = 2/3.
        let s = source(
            SpatialDistribution3::Point,
            AngularDistribution3::LambertianHemisphere,
            sampling(1, 8192, SamplingSequence::Sobol, 0),
        );
        let rays = s.emit();
        let mean: f64 = rays.iter().map(|r| r.ray.dir.x).sum::<f64>() / rays.len() as f64;
        assert!((mean - 2.0 / 3.0).abs() < 5e-3, "mean cosθ {mean}");
    }

    #[test]
    fn a_disc_source_samples_its_area_without_clumping() {
        let s = source(
            SpatialDistribution3::Disc { radius_mm: 5.0 },
            AngularDistribution3::Direction { local_dir: Vec3::X },
            sampling(4096, 1, SamplingSequence::Sobol, 0),
        );
        let rays = s.emit();
        let mut inside = 0usize; // within half-radius
        for r in &rays {
            let rr = (r.ray.origin.y.powi(2) + r.ray.origin.z.powi(2)).sqrt();
            assert!(rr <= 5.0 + 1e-9, "outside the disc: {rr}");
            if rr <= 5.0 / std::f64::consts::SQRT_2 {
                inside += 1;
            }
        }
        // Half the area is within r/√2, so ~half the (area-uniform) samples are.
        let frac = inside as f64 / rays.len() as f64;
        assert!((frac - 0.5).abs() < 0.03, "area fraction {frac}");
    }

    #[test]
    fn sampling_is_deterministic_for_a_fixed_seed_and_varies_with_it() {
        let make = |seed| {
            source(
                SpatialDistribution3::Disc { radius_mm: 1.0 },
                AngularDistribution3::UniformConeSolidAngle {
                    half_angle_rad: 0.3,
                },
                sampling(8, 8, SamplingSequence::Stratified, seed),
            )
            .emit()
        };
        let a = make(42);
        let b = make(42);
        let c = make(43);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x == y), "same seed differs");
        assert!(
            a.iter().zip(&c).any(|(x, y)| x != y),
            "different seed matches"
        );
    }

    #[test]
    fn the_transform_places_and_aims_the_bundle() {
        // Aim down -Y by rotating +X onto -Y, and offset the origin.
        let q = glam::DQuat::from_rotation_arc(Vec3::X, -Vec3::Y);
        let s = Source3 {
            transform: Isometry3::new(Vec3::new(1.0, 2.0, 3.0), [q.x, q.y, q.z, q.w]).unwrap(),
            ..source(
                SpatialDistribution3::Point,
                AngularDistribution3::Direction { local_dir: Vec3::X },
                sampling(1, 1, SamplingSequence::Grid, 0),
            )
        };
        let rays = s.emit();
        assert_eq!(rays.len(), 1);
        assert!((rays[0].ray.origin - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-12);
        assert!((rays[0].ray.dir - (-Vec3::Y)).length() < 1e-12);
    }
}
