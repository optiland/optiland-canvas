//! Property/fuzz invariants for the 3D geometry (§15.6).
//!
//! A seeded, dependency-free harness — the "equivalent deterministic fuzz
//! harness" the contract allows in place of `proptest`. Determinism is the point:
//! a CI failure must reproduce from the same seed every time, so the generator is
//! a fixed-seed SplitMix64 and nothing here reads the clock or a global RNG.
//!
//! The invariants, over random transforms, conics, aspheres, rays, and scene
//! scales from `1e-3 mm` to `1e4 mm`:
//!
//! - no NaN or infinity in an accepted hit;
//! - an accepted `t` is strictly greater than `t_min`;
//! - a reported normal is unit length;
//! - a hit is invariant under the placement transform;
//! - the world AABB never omits a hit it was asked about;
//! - a geometrically similar scene reaches the same decision at any scale.
#![cfg(feature = "scene3")]

use oc_core::geom3::primitives::{revolution_intersect, RawHit3, Revolution};
use oc_core::geom3::{Intersect3, LocalHit3};
use oc_core::id::PatchId;
use oc_core::profile2::{
    ClosedProfile2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::SolidGeometry3;
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Ray3, Vec3};

/// SplitMix64: a tiny, well-distributed PRNG with an explicit seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn vec(&mut self, lo: f64, hi: f64) -> Vec3 {
        Vec3::new(self.range(lo, hi), self.range(lo, hi), self.range(lo, hi))
    }
    /// A unit direction, rejection-sampled off the origin.
    fn dir(&mut self) -> Vec3 {
        loop {
            let v = self.vec(-1.0, 1.0);
            let l = v.length();
            if (0.2..=1.0).contains(&l) {
                return v / l;
            }
        }
    }
    fn transform(&mut self, span: f64) -> Isometry3 {
        // A random quaternion (normalized by the constructor) and translation.
        let q = self.vec(-1.0, 1.0);
        let w = self.range(-1.0, 1.0);
        Isometry3::new(self.vec(-span, span), [q.x, q.y, q.z, w]).unwrap_or(Isometry3::IDENTITY)
    }
}

const TMIN: f64 = 1e-9;

/// A centred rectangle profile in `XY`: closed, symmetric about `y = 0`, so it is
/// both revolvable (to a cylinder) and extrudable.
fn rect_profile(w: f64, h: f64) -> ClosedProfile2 {
    let c = [
        Vec2::new(-w, -h),
        Vec2::new(w, -h),
        Vec2::new(w, h),
        Vec2::new(-w, h),
    ];
    ClosedProfile2 {
        edges: (0..4)
            .map(|i| ProfileEdge2 {
                patch: PatchId(i as u64),
                curve: Curve2::Line {
                    a: c[i],
                    b: c[(i + 1) % 4],
                },
            })
            .collect(),
    }
}

fn collect(g: &impl Intersect3, r: &Ray3, t: &Isometry3) -> Vec<LocalHit3> {
    let mut v = Vec::new();
    g.intersect(r, t, TMIN, f64::INFINITY, &mut |h| v.push(h));
    v
}

/// A random well-conditioned solid at a given scale (mm). Conic and asphere
/// surfaces of revolution get their own sweep in
/// [`a_revolution_surface_never_returns_a_bad_hit`]; here the analytic solids and
/// the two profile constructions carry the transform, AABB, and scale invariants.
fn random_solid(rng: &mut Rng, scale: f64) -> SolidGeometry3 {
    match rng.next() % 5 {
        0 => SolidGeometry3::Sphere {
            radius_mm: rng.range(0.2, 2.0) * scale,
        },
        1 => SolidGeometry3::Box {
            half_extents_mm: rng.vec(0.2, 1.5) * scale,
        },
        2 => SolidGeometry3::Cylinder {
            radius_mm: rng.range(0.2, 1.0) * scale,
            half_length_mm: rng.range(0.5, 2.0) * scale,
            axis_local: rng.dir(),
        },
        3 => SolidGeometry3::Profile(ProfileSolid3 {
            profile: rect_profile(rng.range(0.3, 1.5) * scale, rng.range(0.3, 1.5) * scale),
            construction: ProfileConstruction3::RevolveLocalX,
        }),
        _ => {
            let half = rng.range(0.5, 2.0) * scale;
            SolidGeometry3::Profile(ProfileSolid3 {
                profile: rect_profile(rng.range(0.3, 1.5) * scale, rng.range(0.3, 1.5) * scale),
                construction: ProfileConstruction3::ExtrudeLocalZ {
                    z_min_mm: -half,
                    z_max_mm: half,
                },
            })
        }
    }
}

#[test]
fn every_accepted_hit_is_finite_forward_and_unit_normalled() {
    let mut rng = Rng(0xC0FF_EE12_3456_7890);
    for scale in [1e-3, 1.0, 1e2, 1e4] {
        for _ in 0..600 {
            let g = random_solid(&mut rng, scale);
            let t = rng.transform(4.0 * scale);
            let origin = rng.vec(-8.0, 8.0) * scale + t.pos;
            let ray = match Ray3::new(origin, rng.dir()) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for h in collect(&g, &ray, &t) {
                assert!(h.t_mm.is_finite() && h.t_mm > TMIN, "t = {}", h.t_mm);
                assert!(h.point_local.is_finite(), "non-finite point");
                assert!(h.normal_local.is_finite(), "non-finite normal");
                assert!(
                    (h.normal_local.length() - 1.0).abs() < 1e-9,
                    "normal not unit: {}",
                    h.normal_local.length()
                );
            }
        }
    }
}

#[test]
fn a_revolution_surface_never_returns_a_bad_hit() {
    // The asphere Newton path is the one with numerical risk, so it gets its own
    // sweep: random conics and aspheres, random rays, at three scales.
    let mut rng = Rng(0xA5A5_1111_2222);
    for scale in [1e-3, 1.0, 1e4] {
        for _ in 0..1500 {
            let outer = rng.range(0.3, 2.0) * scale;
            let c = rng.range(-0.7, 0.7) / outer;
            let surf = Revolution {
                vertex: Vec3::ZERO,
                axis: rng.dir(),
                c,
                k: rng.range(-3.0, 1.0),
                coeffs: {
                    let mut a = [0.0; 8];
                    a[0] = rng.range(-2e-4, 2e-4) / scale.max(1.0);
                    a[1] = rng.range(-1e-6, 1e-6) / (scale * scale).max(1.0);
                    a
                },
                inner_r: 0.0,
                outer_r: outer,
            };
            let ray = match Ray3::new(rng.vec(-6.0, 6.0) * scale, rng.dir()) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let mut hits: Vec<RawHit3> = Vec::new();
            revolution_intersect(&surf, &ray, TMIN, f64::INFINITY, &mut |h| hits.push(h));
            for h in hits {
                assert!(h.t_mm.is_finite() && h.t_mm > TMIN);
                assert!(h.point.is_finite() && h.normal.is_finite());
                assert!((h.normal.length() - 1.0).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn a_hit_is_invariant_under_placement() {
    let mut rng = Rng(0xBEEF_9999_0000);
    for scale in [1e-2, 1.0, 1e3] {
        for _ in 0..500 {
            let g = random_solid(&mut rng, scale);
            let t = rng.transform(5.0 * scale);
            let ray = match Ray3::new(rng.vec(-10.0, 10.0) * scale, rng.dir()) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let placed = collect(&g, &ray, &t);
            // Pull the ray into the local frame and fire at the identity.
            let local_ray =
                Ray3::new(t.inverse_transform_point(ray.origin), t.inv_rotate(ray.dir)).unwrap();
            let local = collect(&g, &local_ray, &Isometry3::IDENTITY);
            assert_eq!(placed.len(), local.len());
            for (p, l) in placed.iter().zip(&local) {
                let tol = 1e-7 * scale.max(1.0);
                assert!((p.t_mm - l.t_mm).abs() <= tol);
                // Both local hits describe the same local point (the placed one
                // pulled the ray by `t` internally; the manual one did it by hand).
                assert!((p.point_local - l.point_local).length() <= tol);
            }
        }
    }
}

#[test]
fn the_world_aabb_never_omits_a_hit() {
    let mut rng = Rng(0x1357_2468_ACE0);
    for scale in [1e-3, 1.0, 1e4] {
        for _ in 0..800 {
            let g = random_solid(&mut rng, scale);
            let t = rng.transform(4.0 * scale);
            let ray = match Ray3::new(rng.vec(-8.0, 8.0) * scale + t.pos, rng.dir()) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let bb = g.aabb_world(&t);
            let pad = bb.padded(1e-6 * bb.diagonal().max(scale));
            for h in collect(&g, &ray, &t) {
                let world = t.transform_point(h.point_local);
                assert!(
                    pad.contains_point(world),
                    "hit escaped the AABB at scale {scale}"
                );
            }
        }
    }
}

#[test]
fn geometry_is_scale_invariant() {
    // A sphere and ray, scaled uniformly by `s`, must produce a hit `t` scaled by
    // the same `s`. Fixed epsilons would break this; the tolerance policy does not.
    let mut rng = Rng(0x0FED_CBA9_8765);
    for _ in 0..300 {
        let r = rng.range(0.5, 3.0);
        let origin = rng.vec(-6.0, 6.0);
        let dir = rng.dir();
        let base = SolidGeometry3::Sphere { radius_mm: r };
        let base_ray = match Ray3::new(origin, dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let base_hits = collect(&base, &base_ray, &Isometry3::IDENTITY);
        for s in [1e-3, 1e3] {
            let scaled = SolidGeometry3::Sphere { radius_mm: r * s };
            let scaled_ray = Ray3::new(origin * s, dir).unwrap();
            let scaled_hits = collect(&scaled, &scaled_ray, &Isometry3::IDENTITY);
            assert_eq!(
                base_hits.len(),
                scaled_hits.len(),
                "hit count changed with scale"
            );
            for (b, sc) in base_hits.iter().zip(&scaled_hits) {
                let rel = ((sc.t_mm - b.t_mm * s) / (b.t_mm * s)).abs();
                assert!(rel < 1e-9, "t did not scale: rel error {rel}");
            }
        }
    }
}
