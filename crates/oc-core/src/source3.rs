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

/// Salts that keep a source's sample streams apart under stratified jitter.
const ANGULAR_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const PAIRING_SALT: u64 = 0xD1B5_4A32_D192_ED03;
/// Salts that set one pass of a source's rays apart from the next
/// ([`Source3::emit_pass`]).
const PASS_SALT: u64 = 0xA24B_AED4_963E_E407;
const ROTATION_SALT: u64 = 0x5851_F42D_4C95_7F2D;
const SPECTRAL_SALT: u64 = 0x8CB9_2BA7_2F3D_8DD7;

impl Source3 {
    /// Expand this source into its deterministic set of launch rays (§6.10).
    ///
    /// A source with `spatial_samples × angular_samples = n` launches `n` rays
    /// per spectral line, or `n` rays in all for a [`Spectrum::Continuum`](crate::scene::Spectrum::Continuum),
    /// whose rays each carry their own wavelength. Flux is split evenly across
    /// the rays of each line, weighted by the line, so the total emitted flux
    /// is `flux` regardless of sample count.
    ///
    /// Where the rays come from and where they go:
    ///
    /// - A source that spreads only one of them (a point source spreads
    ///   directions, a collimated beam origins) takes every origin with every
    ///   direction, as it always has.
    /// - A source that spreads both, such as the sun, a Lambertian disc, or a
    ///   cone from an area, gives **every ray its own direction**: `n` origins
    ///   and `n` directions, each set low-discrepancy on its own, paired by a
    ///   fixed shuffle. Taking the product instead would send every origin out
    ///   along the same `angular_samples` directions, and a sun of two
    ///   directions is two points in the sky, not a disc: after a light pipe,
    ///   which keeps each ray's angle to its axis, that shows up as rings.
    ///   The shuffle is Kollig and Keller's padding of two stratified sets,
    ///   which keeps each set as even as it was.
    /// - A continuum draws each ray's wavelength the same way, from a third
    ///   stratified set in the spectrum's power, shuffled apart from the other
    ///   two. A limb-darkened sun then draws each ray's direction for that
    ///   ray's own wavelength.
    pub fn emit(&self) -> Vec<EmittedRay3> {
        self.emit_pass(0)
    }

    /// Pass `pass` of this source's rays: [`Source3::emit`] for pass 0, and
    /// for any other the same point sets shifted by a random offset on the
    /// unit square (a Cranley–Patterson rotation), their jitter and pairing
    /// drawn afresh.
    ///
    /// Each pass is as even as the first and independent of it, so the mean
    /// of several passes is an unbiased estimate whose noise falls as one over
    /// the root of the passes. That lets a trace take more rays than it can
    /// hold at once, a pass at a time, without repeating any.
    pub fn emit_pass(&self, pass: u32) -> Vec<EmittedRay3> {
        if self.spectrum.is_continuum() || self.spreads_both() {
            return self.emit_paired(pass);
        }
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
            pass,
        );

        let mut out = Vec::with_capacity(spectral.len() * n_spatial * n_angular);
        for (lambda, weight) in &spectral {
            // Per line: a limb-darkened sun is a different shape in each.
            let dirs_local = angular_samples(
                &self.angular,
                n_angular,
                self.samples.sequence,
                self.samples.seed ^ ANGULAR_SALT,
                pass,
                *lambda,
            );
            let flux_each = self.flux * weight / (n_spatial * n_angular) as f64;
            for o_local in &origins_local {
                let origin = self.transform.transform_point(*o_local);
                for d_local in &dirs_local {
                    self.push(&mut out, origin, *d_local, *lambda, flux_each);
                }
            }
        }
        out
    }

    /// Does this source spread both its origins and its directions, over more
    /// than one sample each? The v1 kinds never do: they keep their product.
    fn spreads_both(&self) -> bool {
        let spatial = !matches!(self.spatial, SpatialDistribution3::Point) && !self.spatial.is_legacy();
        let angular = !matches!(self.angular, AngularDistribution3::Direction { .. }) && !self.angular.is_legacy();
        spatial && angular && self.samples.spatial_samples > 1 && self.samples.angular_samples > 1
    }

    /// Every ray its own origin and direction, and under a continuum its own
    /// wavelength (see [`Source3::emit`]).
    fn emit_paired(&self, pass: u32) -> Vec<EmittedRay3> {
        let n = (self.samples.spatial_samples.max(1) as usize) * (self.samples.angular_samples.max(1) as usize);
        let (seq, seed) = (self.samples.sequence, self.samples.seed);
        let origins = spatial_samples(&self.spatial, n, seq, seed, pass);
        let dirs = unit_square(n, seq, seed ^ ANGULAR_SALT, pass);
        let spectral = self.spectrum.samples();
        let mut out = Vec::with_capacity(n * self.spectrum.ray_multiplicity());
        let ray = |out: &mut Vec<EmittedRay3>, i: usize, d: usize, lambda: f64, flux: f64| {
            let (u, v) = dirs[d];
            let dir = angular_dir(&self.angular, u, v, lambda);
            self.push(out, self.transform.transform_point(origins[i]), dir, lambda, flux);
        };
        if self.spectrum.is_continuum() {
            // Each ray one band, as often as its power says: a stratified
            // number through the cumulative power.
            let mut cdf = Vec::with_capacity(spectral.len());
            let mut acc = 0.0;
            for (_, w) in &spectral {
                acc += w;
                cdf.push(acc);
            }
            let picks = unit_interval(n, seq, seed ^ SPECTRAL_SALT, pass);
            let pair_d = shuffle(n, pass_seed(seed ^ PAIRING_SALT, pass));
            let pair_l = shuffle(n, pass_seed(seed ^ SPECTRAL_SALT, pass));
            let flux_each = self.flux / n as f64;
            for i in 0..n {
                let u = picks[pair_l[i]] * acc;
                let k = cdf.partition_point(|c| *c <= u).min(spectral.len() - 1);
                ray(&mut out, i, pair_d[i], spectral[k].0, flux_each);
            }
        } else {
            for (line, (lambda, weight)) in spectral.iter().enumerate() {
                let pair_d = shuffle(n, pass_seed(seed ^ PAIRING_SALT ^ (line as u64).wrapping_mul(ANGULAR_SALT), pass));
                let flux_each = self.flux * weight / n as f64;
                for (i, d) in pair_d.into_iter().enumerate() {
                    ray(&mut out, i, d, *lambda, flux_each);
                }
            }
        }
        out
    }

    fn push(&self, out: &mut Vec<EmittedRay3>, origin: Vec3, d_local: Vec3, lambda: f64, flux: f64) {
        let dir = self.transform.rotate(d_local);
        // A degenerate direction (a zero-length legacy dir, say) would make no
        // ray; skip it rather than emit a NaN.
        let Ok(ray) = Ray3::new(origin, dir) else {
            return;
        };
        out.push(EmittedRay3 {
            ray,
            lambda_um: lambda,
            flux,
            source: self.id,
        });
    }
}

/// A stream's seed for pass `pass`: its own for pass 0, a fresh one otherwise.
fn pass_seed(seed: u64, pass: u32) -> u64 {
    if pass == 0 {
        seed
    } else {
        seed ^ (pass as u64).wrapping_mul(PASS_SALT)
    }
}

/// The offset pass `pass` shifts a stream's points by, modulo 1: none for
/// pass 0, two uniform numbers otherwise.
fn rotation(seed: u64, pass: u32) -> (f64, f64) {
    if pass == 0 {
        return (0.0, 0.0);
    }
    let mut rng = Rng::new(pass_seed(seed, pass) ^ ROTATION_SALT);
    (rng.next_f64(), rng.next_f64())
}

/// `x + s` wrapped into `[0, 1)`.
fn wrap(x: f64, s: f64) -> f64 {
    let y = x + s;
    if y >= 1.0 { y - 1.0 } else { y }
}

/// A fixed shuffle of `0..n` (Fisher–Yates on the source's own generator).
fn shuffle(n: usize, seed: u64) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed);
    for i in (1..n).rev() {
        let j = ((rng.next_f64() * (i + 1) as f64) as usize).min(i);
        p.swap(i, j);
    }
    p
}

// ---------------------------------------------------------------------------
// Spatial sampling — local origins in the YZ aperture plane
// ---------------------------------------------------------------------------

fn spatial_samples(
    dist: &SpatialDistribution3,
    n: usize,
    seq: SamplingSequence,
    seed: u64,
    pass: u32,
) -> Vec<Vec3> {
    match dist {
        SpatialDistribution3::Point => vec![Vec3::ZERO; n],
        SpatialDistribution3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => unit_square(n, seq, seed, pass)
            .into_iter()
            .map(|(u, v)| {
                Vec3::new(
                    0.0,
                    (2.0 * u - 1.0) * half_width_mm,
                    (2.0 * v - 1.0) * half_height_mm,
                )
            })
            .collect(),
        SpatialDistribution3::Disc { radius_mm } => unit_square(n, seq, seed, pass)
            .into_iter()
            .map(|(u, v)| {
                let (dy, dz) = concentric_disc(u, v);
                Vec3::new(0.0, radius_mm * dy, radius_mm * dz)
            })
            .collect(),
        // v1 compatibility: a line along local +Y in the plane, no area.
        SpatialDistribution3::LineLegacy { half_width_mm } => unit_interval(n, seq, seed, pass)
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
    pass: u32,
    lambda_um: f64,
) -> Vec<Vec3> {
    match dist {
        AngularDistribution3::Direction { local_dir } => {
            vec![local_dir.normalize_or_zero(); n]
        }
        // v1 compatibility: an equal-angle fan in the local XY plane.
        AngularDistribution3::PlanarFanLegacy { half_angle_rad } => unit_interval(n, seq, seed, pass)
            .into_iter()
            .map(|u| {
                let ang = (2.0 * u - 1.0) * half_angle_rad;
                Vec3::new(ang.cos(), ang.sin(), 0.0)
            })
            .collect(),
        _ => unit_square(n, seq, seed, pass)
            .into_iter()
            .map(|(u, v)| angular_dir(dist, u, v, lambda_um))
            .collect(),
    }
}

/// The direction a point `(u, v)` of the unit square maps to, about `+X`.
fn angular_dir(dist: &AngularDistribution3, u: f64, v: f64, lambda_um: f64) -> Vec3 {
    match *dist {
        AngularDistribution3::Direction { local_dir } => local_dir.normalize_or_zero(),
        AngularDistribution3::UniformConeSolidAngle { half_angle_rad } => {
            // Uniform in solid angle: cosθ uniform in [cos α, 1].
            let cos_t = 1.0 - u * (1.0 - half_angle_rad.cos());
            dir_from_cos(cos_t, TAU * v)
        }
        AngularDistribution3::LambertianHemisphere => {
            // Cosine-weighted: cosθ = √(1 − u), so radiance is constant.
            dir_from_cos((1.0 - u).sqrt(), TAU * v)
        }
        AngularDistribution3::Sun {
            half_angle_rad,
            disc,
            csr,
        } => {
            let shape = crate::solar::sunshape::SunShape {
                half_angle_rad,
                disc,
                csr,
            };
            dir_from_cos(shape.polar_angle(u, lambda_um).cos(), TAU * v)
        }
        AngularDistribution3::PlanarFanLegacy { half_angle_rad } => {
            let ang = (2.0 * u - 1.0) * half_angle_rad;
            Vec3::new(ang.cos(), ang.sin(), 0.0)
        }
        AngularDistribution3::RadialProfile { half_angle_rad, radiance } => {
            dir_from_cos(radial_polar_angle(half_angle_rad, &radiance, u).cos(), TAU * v)
        }
    }
}

/// The polar angle at fraction `u` of a [`AngularDistribution3::RadialProfile`]'s
/// power: radiance linear between its nodes, weighted by `sin θ` for the
/// ring's solid angle, inverted on a fine grid of each span.
fn radial_polar_angle(half: f64, radiance: &[f64; crate::scene3::RADIAL_NODES], u: f64) -> f64 {
    const SUB: usize = 16;
    let n = radiance.len() - 1;
    let h = half / n as f64 / SUB as f64;
    let at = |t: f64| {
        let x = (t / half * n as f64).clamp(0.0, n as f64);
        let i = (x as usize).min(n - 1);
        let f = x - i as f64;
        (radiance[i] * (1.0 - f) + radiance[i + 1] * f) * t.sin()
    };
    // Trapezoids on the fine grid: the power up to each grid angle.
    let mut cdf = Vec::with_capacity(n * SUB + 1);
    cdf.push(0.0);
    for k in 0..n * SUB {
        let (a, b) = (k as f64 * h, (k + 1) as f64 * h);
        cdf.push(cdf[k] + 0.5 * (at(a) + at(b)) * h);
    }
    let want = u.clamp(0.0, 1.0) * cdf[n * SUB];
    let k = cdf.partition_point(|c| *c < want).clamp(1, n * SUB);
    let (c0, c1) = (cdf[k - 1], cdf[k]);
    let f = if c1 > c0 { (want - c0) / (c1 - c0) } else { 0.0 };
    ((k - 1) as f64 + f) * h
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

/// `n` samples in `[0, 1)`, shifted for pass `pass` ([`Source3::emit_pass`]).
fn unit_interval(n: usize, seq: SamplingSequence, seed: u64, pass: u32) -> Vec<f64> {
    let points = unit_interval_first(n, seq, pass_seed(seed, pass));
    if pass == 0 {
        return points;
    }
    let (s, _) = rotation(seed, pass);
    points.into_iter().map(|x| wrap(x, s)).collect()
}

/// `n` samples in `[0, 1)²`, shifted for pass `pass` ([`Source3::emit_pass`]).
fn unit_square(n: usize, seq: SamplingSequence, seed: u64, pass: u32) -> Vec<(f64, f64)> {
    let points = unit_square_first(n, seq, pass_seed(seed, pass));
    if pass == 0 {
        return points;
    }
    let (su, sv) = rotation(seed, pass);
    points.into_iter().map(|(u, v)| (wrap(u, su), wrap(v, sv))).collect()
}

/// `n` samples in `[0, 1)`: pass 0's.
fn unit_interval_first(n: usize, seq: SamplingSequence, seed: u64) -> Vec<f64> {
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

/// `n` samples in `[0, 1)²`: pass 0's.
fn unit_square_first(n: usize, seq: SamplingSequence, seed: u64) -> Vec<(f64, f64)> {
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
    /// A flat radial profile is a uniform cone: its polar angle at power
    /// fraction `u` is the cone's, `cos θ = 1 - u (1 - cos α)`; and a profile
    /// that falls to zero at its middle node puts nothing past it.
    #[test]
    fn a_radial_profile_samples_its_power() {
        use crate::scene3::RADIAL_NODES;
        let half = 0.05;
        for u in [0.0, 0.1, 0.37, 0.5, 0.81, 1.0] {
            let t = super::radial_polar_angle(half, &[1.0; RADIAL_NODES], u);
            let want = (1.0 - u * (1.0 - half.cos())).acos();
            assert!((t - want).abs() < 2e-6, "u {u}: {t} vs {want}");
        }
        let mut half_lit = [0.0; RADIAL_NODES];
        half_lit[..12].fill(1.0);
        // Linear down to zero at node 12.
        let edge = half * 12.0 / (RADIAL_NODES - 1) as f64;
        assert!(super::radial_polar_angle(half, &half_lit, 1.0) <= edge + 1e-12);
    }

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
            coherence: None,
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

    /// The sun that made rings: a disc of origins and a cone of directions,
    /// 256 by 2. Every ray now has its own direction, and together they fill
    /// the cone evenly instead of standing on two points of it.
    #[test]
    fn a_source_spreading_both_gives_every_ray_its_own_direction() {
        let alpha = 4.65e-3_f64;
        let s = source(
            SpatialDistribution3::Disc { radius_mm: 500.0 },
            AngularDistribution3::UniformConeSolidAngle { half_angle_rad: alpha },
            sampling(256, 2, SamplingSequence::Sobol, 0),
        );
        let rays = s.emit();
        assert_eq!(rays.len(), 512);
        let mut dirs: Vec<[u64; 3]> = rays.iter().map(|r| [r.ray.dir.x.to_bits(), r.ray.dir.y.to_bits(), r.ray.dir.z.to_bits()]).collect();
        dirs.sort_unstable();
        dirs.dedup();
        assert_eq!(dirs.len(), 512, "one direction per ray");
        let mut origins: Vec<[u64; 3]> = rays.iter().map(|r| [r.ray.origin.x.to_bits(), r.ray.origin.y.to_bits(), r.ray.origin.z.to_bits()]).collect();
        origins.sort_unstable();
        origins.dedup();
        assert_eq!(origins.len(), 512, "one origin per ray");
        // Uniform in solid angle: a quarter of the rays inside half the cone.
        let inside = rays.iter().filter(|r| r.ray.dir.x.acos() < 0.5 * alpha).count();
        assert!((inside as f64 / 512.0 - 0.25).abs() < 0.02, "{inside} of 512 inside half the cone");
        let flux: f64 = rays.iter().map(|r| r.flux).sum();
        assert!((flux - 1.0).abs() < 1e-12);
    }

    /// A continuum gives each ray one wavelength, as often as its power says,
    /// and launches no more rays than its samples.
    #[test]
    fn a_continuum_draws_one_wavelength_per_ray_by_power() {
        let mut s = source(
            SpatialDistribution3::Disc { radius_mm: 5.0 },
            AngularDistribution3::Direction { local_dir: Vec3::X },
            sampling(1000, 1, SamplingSequence::Sobol, 0),
        );
        s.flux = 3.0;
        s.spectrum = Spectrum::Continuum(vec![(0.45, 1.0), (0.55, 3.0), (0.65, 6.0)]);
        let rays = s.emit();
        assert_eq!(rays.len(), 1000);
        let flux: f64 = rays.iter().map(|r| r.flux).sum();
        assert!((flux - 3.0).abs() < 1e-12);
        for (lambda, share) in [(0.45, 0.1), (0.55, 0.3), (0.65, 0.6)] {
            let n = rays.iter().filter(|r| r.lambda_um == lambda).count();
            // Stratified: each band within a ray of its share.
            assert!((n as f64 / 1000.0 - share).abs() <= 2e-3, "{lambda}: {n}");
        }
    }

    /// A limb-darkened sun is smaller in the blue than in the red, because
    /// its limb is darker there: each ray's direction follows its wavelength.
    #[test]
    fn a_limb_darkened_sun_is_smaller_in_the_blue() {
        let mut s = source(
            SpatialDistribution3::Disc { radius_mm: 100.0 },
            AngularDistribution3::Sun {
                half_angle_rad: 4.65e-3,
                disc: crate::solar::sunshape::SunDisc::LimbDarkened,
                csr: 0.0,
            },
            sampling(512, 8, SamplingSequence::Sobol, 0),
        );
        s.spectrum = Spectrum::Continuum(vec![(0.4, 1.0), (1.0, 1.0)]);
        let rays = s.emit();
        let rms = |lambda: f64| {
            let a: Vec<f64> = rays.iter().filter(|r| r.lambda_um == lambda).map(|r| r.ray.dir.x.min(1.0).acos()).collect();
            (a.iter().map(|t| t * t).sum::<f64>() / a.len() as f64).sqrt()
        };
        assert!(rms(0.4) < 0.97 * rms(1.0), "blue {} vs red {}", rms(0.4), rms(1.0));
        assert!(rays.iter().all(|r| r.ray.dir.x.acos() <= 4.65e-3 * (1.0 + 1e-9)));
    }

    /// A source spreading only one of the two keeps the product it always
    /// had, ray for ray.
    #[test]
    fn a_source_spreading_one_keeps_its_product() {
        let s = source(
            SpatialDistribution3::Point,
            AngularDistribution3::UniformConeSolidAngle { half_angle_rad: 0.1 },
            sampling(1, 16, SamplingSequence::Sobol, 0),
        );
        let rays = s.emit();
        let expect = angular_samples(&s.angular, 16, SamplingSequence::Sobol, ANGULAR_SALT, 0, 0.5);
        assert_eq!(rays.len(), 16);
        for (r, d) in rays.iter().zip(&expect) {
            assert!((r.ray.dir - *d).length() < 1e-15);
        }
    }

    /// A later pass is the same count and flux of rays, none of them the
    /// first pass's, and still an even sample: the mean of cosθ over a cone
    /// uniform in solid angle is (1 + cos α)/2 in every pass.
    #[test]
    fn every_pass_is_a_fresh_even_sample() {
        let s = source(
            SpatialDistribution3::Disc { radius_mm: 1.0 },
            AngularDistribution3::UniformConeSolidAngle { half_angle_rad: 0.2 },
            sampling(64, 16, SamplingSequence::Sobol, 0),
        );
        let first = s.emit();
        assert_eq!(s.emit_pass(0), first);
        let expect = (1.0 + 0.2f64.cos()) / 2.0;
        for pass in 1..4 {
            let rays = s.emit_pass(pass);
            assert_eq!(rays.len(), first.len());
            let flux: f64 = rays.iter().map(|r| r.flux).sum();
            assert!((flux - s.flux).abs() <= 1e-12 * s.flux);
            assert!(rays.iter().all(|r| first.iter().all(|f| f.ray.dir != r.ray.dir)));
            let mean = rays.iter().map(|r| r.ray.dir.x).sum::<f64>() / rays.len() as f64;
            assert!((mean - expect).abs() < 1e-5, "pass {pass}: {mean} vs {expect}");
        }
    }
}
