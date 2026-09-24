//! The Fresnel lens (OpticStudio's "Fresnel 1"), checked against the smooth
//! lens it is cut from.
//!
//! - **Exactness.** A facet is the base surface shifted along the axis, so a
//!   collimated ray entering the flat side leaves a facet at height `h` in the
//!   same direction the smooth plano-convex lens sends it at `h`: same normal,
//!   same internal direction. Checked to 1e-12, skew rays and an aspheric,
//!   conic base included.
//! - **Paraxial focus.** `h / tan(u')` tends to `R / (n - 1)`, and a detector at
//!   the focus sees a spot, not a disc.
//! - **Groove losses.** With a draft angle, rays inside the glass meet the
//!   leaning risers, and the fraction that does is the drafted area's share of
//!   the aperture.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use std::collections::HashSet;

use oc_core::fresnel_lens::{fresnel_profile, FresnelGrooves, FresnelLens1, PMMA_ND_VD};
use oc_core::id::{LayerId, PatchId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::{AbsorptionModel, Body3, ObjectCommon, SolidGeometry3, SurfaceProps3};
use oc_core::vec::Vec2;
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, TraceConfig3, TraceEventKind};

fn pmma() -> Material {
    let (nd, vd) = PMMA_ND_VD;
    Material::AbbePoint { nd, vd }
}

fn pmma_index() -> f64 {
    let (nd, vd) = PMMA_ND_VD;
    oc_core::material::Cauchy::from_abbe(nd, vd).index(LAMBDA_D)
}

fn fresnel_scene(lens: &FresnelLens1) -> (Scene3, u64) {
    let mut s = Scene3::new();
    let id = s
        .add_fresnel_lens(Vec3::ZERO, 0.0, lens, pmma())
        .expect("the lens builds");
    (s, id)
}

/// The smooth plano-convex lens the Fresnel lens is cut from: the same flat
/// back face, and one whole face with the central facet's curvature, conic and
/// aspheric terms, set far enough out that the glass has positive thickness at
/// the rim. Where the face sits along the axis does not change a ray's
/// direction, only where it leaves.
fn smooth_scene(lens: &FresnelLens1) -> Scene3 {
    let p = fresnel_profile(lens).unwrap();
    let (face, coeffs) = p
        .edges
        .iter()
        .find_map(|e| match &e.curve {
            Curve2::Conic(cp) => Some((*cp, [0.0; 8])),
            Curve2::EvenAsphere { base, coeffs } => Some((*base, *coeffs)),
            _ => None,
        })
        .unwrap();
    let Curve2::Line { a: back_lo, .. } = p.edges[0].curve else {
        unreachable!()
    };
    let rc = lens.clear_radius_mm;
    let x_b = back_lo.x;
    let base = ConicParams2 {
        vertex: Vec2::new(x_b + 20.0, 0.0),
        half_ap: rc,
        ..face
    };
    let rim = oc_core::geom3::profile::sag_segment_point(&base, &coeffs, rc).x;
    let curve = if coeffs.iter().all(|a| *a == 0.0) {
        Curve2::Conic(base)
    } else {
        Curve2::EvenAsphere { base, coeffs }
    };
    let curves = [
        Curve2::Line {
            a: Vec2::new(x_b, -rc),
            b: Vec2::new(x_b, rc),
        },
        Curve2::Line {
            a: Vec2::new(x_b, rc),
            b: Vec2::new(rim, rc),
        },
        curve,
        Curve2::Line {
            a: Vec2::new(rim, -rc),
            b: Vec2::new(x_b, -rc),
        },
    ];
    let mut s = Scene3::new();
    let edges = curves
        .into_iter()
        .map(|curve| ProfileEdge2 {
            patch: s.alloc_patch_id(),
            curve,
        })
        .collect();
    let id = s.alloc_object_id();
    s.bodies.push(Body3 {
        common: ObjectCommon::new(id, LayerId(0)),
        geometry: SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            construction: ProfileConstruction3::RevolveLocalX,
        }),
        material: pmma(),
        absorption: AbsorptionModel::None,
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    s.validate().expect("the smooth lens validates");
    s
}

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        polarization: false,
        ..no_ghosts()
    }
}

/// The direction a single collimated ray leaves the scene in: its last
/// segment, the one that starts furthest along the axis.
fn exit_direction(scene: &Scene3, h: f64, phi: f64) -> Vec3 {
    let ray = Ray3::new(Vec3::new(-50.0, h * phi.cos(), h * phi.sin()), Vec3::X).unwrap();
    let r = trace3(scene, &[launch(ray, LAMBDA_D, 1.0)], &cfg());
    let last = r
        .segments
        .iter()
        .max_by(|a, b| a.a.x.total_cmp(&b.a.x))
        .expect("the ray is traced");
    (last.b - last.a).normalize()
}

#[test]
fn a_facet_bends_a_ray_exactly_as_the_smooth_lens_does_at_the_same_height() {
    let aspheric = FresnelLens1 {
        radius_mm: 45.0,
        conic: -0.8,
        aspheric: [3e-6, -2e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        grooves: FresnelGrooves::ConstantPitch { pitch_mm: 0.8 },
        ..FresnelLens1::default()
    };
    let depth = FresnelLens1 {
        grooves: FresnelGrooves::ConstantDepth { depth_mm: 0.2 },
        draft_deg: 2.0,
        ..FresnelLens1::default()
    };
    let mut worst: f64 = 0.0;
    for lens in [FresnelLens1::default(), aspheric, depth] {
        let fresnel = fresnel_scene(&lens).0;
        let smooth = smooth_scene(&lens);
        // Skip heights within a hair of a zone boundary or a drafted riser, where
        // the Fresnel lens has a riser the smooth lens does not.
        let p = fresnel_profile(&lens).unwrap();
        let mut steps: Vec<(f64, f64)> = p
            .edges
            .iter()
            .filter_map(|e| match e.curve {
                Curve2::Line { a, b } if a.y > 0.0 && b.y > 0.0 && a.y < lens.clear_radius_mm => {
                    Some((a.y.min(b.y) - 1e-3, a.y.max(b.y) + 1e-3))
                }
                _ => None,
            })
            .collect();
        steps.push((lens.clear_radius_mm - 1e-3, f64::INFINITY));
        let mut checked = 0;
        for i in 0..240 {
            let h = 0.05 + (lens.clear_radius_mm - 0.1) * (i as f64 + 0.5) / 240.0;
            if steps.iter().any(|(lo, hi)| h >= *lo && h <= *hi) {
                continue;
            }
            let phi = 0.7 * i as f64;
            let a = exit_direction(&fresnel, h, phi);
            let b = exit_direction(&smooth, h, phi);
            // The ray was bent, and bent the same way.
            assert!(a.x < 1.0 - 1e-9, "no refraction at h = {h}");
            let err = (a - b).length();
            worst = worst.max(err);
            assert!(err < 1e-12, "h {h}: fresnel {a:?} smooth {b:?} ({err:e})");
            checked += 1;
        }
        assert!(checked > 150, "only {checked} heights checked");
    }
    eprintln!("max exit-direction difference, Fresnel vs smooth: {worst:e}");
}

#[test]
fn the_focal_length_tends_to_r_over_n_minus_one() {
    let lens = FresnelLens1::default();
    let (scene, _) = fresnel_scene(&lens);
    let n = pmma_index();
    let f_parax = lens.paraxial_efl_mm(n).unwrap();
    assert!((f_parax - 50.0 / (n - 1.0)).abs() < 1e-12);
    for (h, tol) in [(0.05, 1e-5), (1.5, 1e-3), (2.5, 3e-3), (3.5, 6e-3)] {
        let d = exit_direction(&scene, h, 0.0);
        let f = h / (-d.y / d.x);
        let rel = f / f_parax - 1.0;
        eprintln!("h {h:4}: f(h) = {f:.6} mm, paraxial {f_parax:.6} mm, {rel:+.2e}");
        assert!(rel.abs() < tol, "h {h}: f {f} vs {f_parax}");
        // Spherical aberration of a convex face: the zones focus shorter.
        assert!(h < 0.1 || rel < 0.0);
    }
}

#[test]
fn a_detector_at_the_focus_sees_a_spot() {
    let lens = FresnelLens1::default();
    let (mut scene, _) = fresnel_scene(&lens);
    let n = pmma_index();
    let f = lens.paraxial_efl_mm(n).unwrap();
    // The central facet's vertex is where the paraxial rays leave; the
    // principal plane of a plano-convex lens met flat side first sits there.
    let p = fresnel_profile(&lens).unwrap();
    let vertex = p
        .edges
        .iter()
        .find_map(|e| match &e.curve {
            Curve2::Conic(cp) => Some(cp.vertex.x),
            _ => None,
        })
        .unwrap();
    let x_focus = vertex + f;
    scene.add_area_detector(
        Vec3::new(x_focus, 0.0, 0.0),
        Vec3::NEG_X,
        5.0,
        5.0,
        [64, 64],
    );
    let beam = 5.0;
    let mut launch_set = Vec::new();
    for i in 0..400 {
        let r = beam * ((i as f64 + 0.5) / 400.0).sqrt();
        let phi = i as f64 * 2.399_963_229_728_653;
        let ray = Ray3::new(Vec3::new(-50.0, r * phi.cos(), r * phi.sin()), Vec3::X).unwrap();
        launch_set.push(launch(ray, LAMBDA_D, 1.0));
    }
    let r = trace3(
        &scene,
        &launch_set,
        &TraceConfig3 {
            record_detector_hits: true,
            ..cfg()
        },
    );
    let det = &r.detectors[0];
    assert_eq!(det.hits, 400, "every ray reaches the focal plane");
    let rms = (det
        .records
        .iter()
        .map(|h| h.uv_mm.length_squared())
        .sum::<f64>()
        / det.records.len() as f64)
        .sqrt();
    let max = det
        .records
        .iter()
        .map(|h| h.uv_mm.length())
        .fold(0.0_f64, f64::max);
    eprintln!("focal spot of a {beam} mm beam: rms {rms:.4} mm, max {max:.4} mm");
    // A 5 mm beam converges to well under a tenth of a millimetre.
    assert!(rms < 0.05 && max < 0.15, "rms {rms}, max {max}");
}

/// The patches of the upper risers of a body's profile, with the radial band
/// each covers.
fn risers(scene: &Scene3, id: u64) -> Vec<(PatchId, f64, f64)> {
    let b = scene.bodies.iter().find(|b| b.common.id.0 == id).unwrap();
    let SolidGeometry3::Profile(p) = &b.geometry else {
        unreachable!()
    };
    let n = p.profile.edges.len();
    p.profile.edges[2..n - 1]
        .iter()
        .filter_map(|e| match e.curve {
            Curve2::Line { a, b } if a.y > 0.0 => Some((e.patch, a.y.min(b.y), a.y.max(b.y))),
            _ => None,
        })
        .collect()
}

/// The fraction of a uniformly filled collimated beam whose first meeting with
/// the grooved face is a riser rather than a facet.
fn riser_fraction(lens: &FresnelLens1, rays: usize) -> (f64, f64) {
    let (scene, id) = fresnel_scene(lens);
    let bands = risers(&scene, id);
    let riser_patches: HashSet<PatchId> = bands.iter().map(|b| b.0).collect();
    let rc = lens.clear_radius_mm;
    let launch_set: Vec<_> = (0..rays)
        .map(|i| {
            let r = rc * ((i as f64 + 0.5) / rays as f64).sqrt();
            let phi = i as f64 * 2.399_963_229_728_653;
            let ray = Ray3::new(Vec3::new(-50.0, r * phi.cos(), r * phi.sin()), Vec3::X).unwrap();
            launch(ray, LAMBDA_D, 1.0)
        })
        .collect();
    let r = trace3(
        &scene,
        &launch_set,
        &TraceConfig3 {
            record_events: true,
            ..cfg()
        },
    );
    // Each ray's second interaction (depth 1) is where it meets the grooved
    // face from inside.
    let hits_riser = r
        .events
        .iter()
        .filter(|e| e.depth == 1)
        .filter(|e| match &e.kind {
            TraceEventKind::Interact { surface, .. } => riser_patches.contains(&surface.patch),
            _ => false,
        })
        .count();
    let traced = hits_riser as f64 / rays as f64;
    // Rays inside the glass run parallel to the axis, so the ones a drafted
    // riser catches are those under it: its radial band's share of the area.
    let geometric = bands
        .iter()
        .map(|(_, lo, hi)| (hi * hi - lo * lo) / (rc * rc))
        .sum::<f64>();
    (traced, geometric)
}

#[test]
fn without_draft_no_collimated_ray_meets_a_riser() {
    let (traced, geometric) = riser_fraction(&FresnelLens1::default(), 4000);
    assert_eq!(geometric, 0.0, "a zero-draft riser has no radial width");
    assert_eq!(traced, 0.0);
}

#[test]
fn a_drafted_riser_catches_the_rays_under_it() {
    let lens = FresnelLens1 {
        draft_deg: 5.0,
        ..FresnelLens1::default()
    };
    let rays = 20_000;
    let (traced, geometric) = riser_fraction(&lens, rays);
    // The rule of thumb: each riser shadows `depth · tan(draft)` of its pitch.
    let tan = lens.draft_deg.to_radians().tan();
    let rc = lens.clear_radius_mm;
    let sag = |r: f64| 50.0 - (2500.0 - r * r).sqrt();
    let thumb = (1..25)
        .map(|j| {
            let r = j as f64;
            let depth = sag(r + 1.0) - sag(r);
            2.0 * r * depth * tan / (rc * rc)
        })
        .sum::<f64>();
    eprintln!(
        "riser loss at a {} degree draft: traced {:.4}, drafted area {:.4}, depth*tan/pitch estimate {:.4}",
        lens.draft_deg, traced, geometric, thumb
    );
    let sigma = (geometric * (1.0 - geometric) / rays as f64).sqrt();
    assert!(
        (traced - geometric).abs() < 4.0 * sigma + 1e-4,
        "traced {traced} vs geometric {geometric}"
    );
    assert!(traced > 0.005, "some rays do meet the risers");
}

/// Per-ray cost of the reference trace through a Fresnel lens against the
/// smooth lens it is cut from. A measurement, not a check: run with
/// `cargo test --release -p oc-trace --features trace3 --test fresnel3 -- --ignored --nocapture`.
#[test]
#[ignore]
fn fresnel_trace_cost() {
    let big = FresnelLens1 {
        radius_mm: 100.0,
        clear_radius_mm: 50.0,
        grooves: FresnelGrooves::ConstantPitch { pitch_mm: 0.5 },
        ..FresnelLens1::default()
    };
    for lens in [FresnelLens1::default(), big] {
        let rays = 20_000;
        let rc = lens.clear_radius_mm;
        let launch_set: Vec<_> = (0..rays)
            .map(|i| {
                let r = 0.98 * rc * ((i as f64 + 0.5) / rays as f64).sqrt();
                let phi = i as f64 * 2.399_963_229_728_653;
                let ray =
                    Ray3::new(Vec3::new(-50.0, r * phi.cos(), r * phi.sin()), Vec3::X).unwrap();
                launch(ray, LAMBDA_D, 1.0)
            })
            .collect();
        let edges = fresnel_profile(&lens).unwrap().edges.len();
        for (name, scene) in [
            ("fresnel", fresnel_scene(&lens).0),
            ("smooth", smooth_scene(&lens)),
        ] {
            let t0 = std::time::Instant::now();
            let r = trace3(&scene, &launch_set, &cfg());
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "{name:8} R {} clear {} ({edges} edges): {:.2} us/ray ({} segments)",
                lens.radius_mm,
                rc,
                1e6 * dt / rays as f64,
                r.segments.len()
            );
        }
    }
}
