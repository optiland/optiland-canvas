//! The f32 preview's coatings against the f64 kernel (coating-fidelity spec
//! CF-07): a grid per side for an absorbing stack (C22), a coating as a
//! body's default response (C23), the edge of the angle range (C24), and a
//! nominal coating under `Fast`.
#![cfg(feature = "gpu3")]

use oc_core::fidelity::{CoatingModel, FidelityProfile};
use oc_core::fresnel_lens::FresnelLens1;
use oc_core::material::Material;
use oc_core::scene::Spectrum;
use oc_core::scene3::{Scene3, SurfaceResponse, TwoSidedSurfaceResponse};
use oc_core::thinfilm::{coating_rt_at, film_coating, Coating, CoatingDesign, DesignKind};
use oc_core::tube::{BoreFinish, BoreMetal, TubeDims};
use oc_core::vec3::Vec3;
use oc_gpu::scene3::{sample_coating_grid, BuildOpts, GpuResponse, GpuScene3};

const LINES: [f64; 4] = [0.45, 0.55, 0.65, 0.8];

fn lit(mut s: Scene3) -> Scene3 {
    s.add_collimated_beam(
        Vec3::new(-200.0, 0.0, 0.0),
        5.0,
        5.0,
        4,
        Spectrum::Lines(LINES.iter().map(|&l| (l, 1.0)).collect()),
    );
    s
}

/// A coated face's wavelengths, its outside grid, and its inside grid if it has one.
type Grids = (Vec<f32>, Vec<[f32; 2]>, Option<Vec<[f32; 2]>>);

/// A body face's coating grids.
fn grid(g: &GpuScene3, body: usize, face: usize) -> Grids {
    match &g.bodies[body].face_resp[face] {
        Some(GpuResponse::CoatingGrid { lambdas, rt, rt_inside, .. }) => {
            (lambdas.clone(), rt.to_vec(), rt_inside.as_ref().map(|v| v.to_vec()))
        }
        other => panic!("face {face} is not a coating grid: {other:?}"),
    }
}

/// Angles in ambient air, up to the last baked row (88.2 degrees).
fn angles(n: usize) -> Vec<f64> {
    (0..n).map(|i| f64::from(oc_gpu::scene3::COAT_THETA_MAX) * i as f64 / (n - 1) as f64).collect()
}

/// C22: a glass tube silvered on its bore. Light in the bore meets silver in
/// air; light in the glass wall meets silver behind glass, which reflects
/// less. The preview bakes both, each within 2e-3 of the kernel; reusing the
/// outside grid for a ray from inside is wrong by more than 1e-2, which is
/// the mutation C22 must catch.
#[test]
fn c22_an_absorbing_stack_is_baked_for_each_side() {
    let mut s = Scene3::new();
    let tube = s
        .add_tube(
            Vec3::ZERO,
            0.0,
            TubeDims { inner_diameter_mm: 20.0, outer_diameter_mm: 24.0, length_mm: 100.0 },
            BoreFinish::Metal { metal: BoreMetal::Silver, protected: false },
        )
        .unwrap();
    let n_glass = 1.5168;
    {
        let b = s.bodies.iter_mut().find(|b| b.common.id.0 == tube).unwrap();
        b.material = Material::AbbePoint { nd: n_glass, vd: 1e9 };
        b.surface.default = TwoSidedSurfaceResponse {
            from_outside: SurfaceResponse::DielectricFresnel { ar_residual: oc_core::scene3::SpectralScalar::Constant(1.0) },
            from_inside: SurfaceResponse::DielectricFresnel { ar_residual: oc_core::scene3::SpectralScalar::Constant(1.0) },
        };
    }
    let s = lit(s);
    let g = GpuScene3::build(&s, BuildOpts::default());
    assert!(g.rejected.is_empty(), "{:?}", g.rejected);
    let bore = oc_core::geom3::primitives::tube_face::INNER as usize;
    let (lambdas, outside, inside) = grid(&g, 0, bore);
    let inside = inside.expect("silver absorbs, so the bore has an inside grid");
    let silver = match &s.bodies[0].surface.overrides[0].response.from_outside {
        SurfaceResponse::Coating { coating } => coating.clone(),
        _ => panic!("the bore is a coating"),
    };
    let behind: Coating = silver.seen_from_behind();
    let (mut worst_in, mut worst_out, mut worst_reused) = (0.0f64, 0.0f64, 0.0f64);
    for th in angles(40) {
        for &lam in &LINES {
            let (cos_a, sin_a) = (th.cos(), th.sin());
            // From the bore (air): silver in air.
            let want = coating_rt_at(&silver, 1.0, n_glass, lam, cos_a).unwrap();
            let (r, _) = sample_coating_grid(&lambdas, &outside, lam as f32, cos_a as f32, 1.0, 1.0);
            worst_out = worst_out.max((r as f64 - want.r()).abs());
            // From the wall (glass): silver behind glass, the same n sin θ.
            let sin_g = sin_a / n_glass;
            let cos_g = (1.0 - sin_g * sin_g).sqrt();
            let want = coating_rt_at(&behind, n_glass, 1.0, lam, cos_g).unwrap();
            let (r, _) = sample_coating_grid(&lambdas, &inside, lam as f32, cos_g as f32, n_glass as f32, 1.0);
            worst_in = worst_in.max((r as f64 - want.r()).abs());
            let (r, _) = sample_coating_grid(&lambdas, &outside, lam as f32, cos_g as f32, n_glass as f32, 1.0);
            worst_reused = worst_reused.max((r as f64 - want.r()).abs());
        }
    }
    assert!(worst_out <= 2e-3, "outside {worst_out:.2e}");
    assert!(worst_in <= 2e-3, "inside {worst_in:.2e}");
    assert!(worst_reused > 1e-2, "the outside grid reused inside errs by only {worst_reused:.2e}");
}

/// C23: a Fresnel lens with an AR coat as its default response previews
/// (it used to be refused), and over 10 000 random samples of wavelength,
/// angle and side the grid stays within 2e-3 of the kernel.
#[test]
fn c23_a_coating_as_a_body_default_previews() {
    let mut s = Scene3::new();
    let pmma = 1.4917;
    let id = s.add_fresnel_lens(Vec3::ZERO, 0.0, &FresnelLens1::default(), Material::AbbePoint { nd: pmma, vd: 1e9 }).unwrap();
    let ar = film_coating(
        CoatingDesign { kind: DesignKind::BroadbandAr, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38 },
        1.0,
        pmma,
    );
    {
        let b = s.bodies.iter_mut().find(|b| b.common.id.0 == id).unwrap();
        b.surface.default = TwoSidedSurfaceResponse {
            from_outside: SurfaceResponse::Coating { coating: ar.clone() },
            from_inside: SurfaceResponse::Coating { coating: ar.seen_from_behind() },
        };
    }
    let s = lit(s);
    let g = GpuScene3::build(&s, BuildOpts::default());
    assert!(g.rejected.is_empty(), "{:?}", g.rejected);
    let faces = g.bodies[0].face_resp.len();
    assert!(faces > 10 && g.bodies[0].face_resp.iter().all(|f| matches!(f, Some(GpuResponse::CoatingGrid { .. }))));
    let (lambdas, rt, inside) = grid(&g, 0, 0);
    assert!(inside.is_none(), "a lossless AR serves both sides from one grid");
    // Every groove shares that one grid.
    if let (Some(GpuResponse::CoatingGrid { rt: a, .. }), Some(GpuResponse::CoatingGrid { rt: b, .. })) =
        (&g.bodies[0].face_resp[0], &g.bodies[0].face_resp[faces - 1])
    {
        assert!(std::sync::Arc::ptr_eq(a, b));
    }
    let behind = ar.seen_from_behind();
    let mut seed = 0x2357_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut worst = 0.0f64;
    for _ in 0..10_000 {
        let lam = LINES[(next() * LINES.len() as f64) as usize % LINES.len()];
        let theta = next() * f64::from(oc_gpu::scene3::COAT_THETA_MAX);
        let (sin_a, cos_a) = (theta.sin(), theta.cos());
        let (got, want) = if next() < 0.5 {
            let want = coating_rt_at(&ar, 1.0, pmma, lam, cos_a).unwrap();
            (sample_coating_grid(&lambdas, &rt, lam as f32, cos_a as f32, 1.0, 1.0), want)
        } else {
            let sin_g = sin_a / pmma;
            let cos_g = (1.0 - sin_g * sin_g).sqrt();
            let want = coating_rt_at(&behind, pmma, 1.0, lam, cos_g).unwrap();
            (sample_coating_grid(&lambdas, &rt, lam as f32, cos_g as f32, pmma as f32, 1.0), want)
        };
        worst = worst.max((got.0 as f64 - want.r()).abs()).max((got.1 as f64 - want.t()).abs());
    }
    assert!(worst <= 2e-3, "worst drift {worst:.2e}");
}

/// C24, the preview's half: within 5e-3 of the kernel from 85 degrees to the
/// last baked row at 88.2, and flat beyond it.
#[test]
fn c24_the_preview_holds_to_88_degrees_and_is_flat_beyond() {
    let mut s = Scene3::new();
    let id = s.add_planar_mirror(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 0.5).unwrap();
    let dichroic = film_coating(
        CoatingDesign { kind: DesignKind::LongPass, lambda_nm: 600.0, aoi_deg: 45.0, periods: 8, n_high: 2.35, n_low: 1.46 },
        1.0,
        1.0,
    );
    {
        let surf = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
        surf.response_front = SurfaceResponse::Coating { coating: dichroic.clone() };
        surf.response_back = SurfaceResponse::Coating { coating: dichroic.seen_from_behind() };
    }
    let s = lit(s);
    let g = GpuScene3::build(&s, BuildOpts::default());
    let (lambdas, rt) = match &g.surfaces[0].front {
        GpuResponse::CoatingGrid { lambdas, rt, .. } => (lambdas.clone(), rt.to_vec()),
        other => panic!("{other:?}"),
    };
    let mut worst = 0.0f64;
    for i in 0..=64 {
        let deg = 85.0 + 3.2 * i as f64 / 64.0;
        for &lam in &LINES {
            let cos = deg.to_radians().cos();
            let want = coating_rt_at(&dichroic, 1.0, 1.0, lam, cos).unwrap();
            let got = sample_coating_grid(&lambdas, &rt, lam as f32, cos as f32, 1.0, 1.0);
            worst = worst.max((got.0 as f64 - want.r()).abs());
        }
    }
    assert!(worst <= 5e-3, "85 to 88.2 degrees: {worst:.2e}");
    // Beyond the last row the grid holds it.
    let at = |deg: f64| sample_coating_grid(&lambdas, &rt, 0.55, deg.to_radians().cos() as f32, 1.0, 1.0);
    assert_eq!(at(88.5), at(89.9));
}

/// Under `coating_model: Nominal` every row of the grid is the design point,
/// which is what the reference tracer then answers.
#[test]
fn a_nominal_coating_bakes_a_flat_grid() {
    let mut s = Scene3::new();
    let id = s.add_planar_mirror(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 0.5).unwrap();
    let dichroic = film_coating(
        CoatingDesign { kind: DesignKind::LongPass, lambda_nm: 600.0, aoi_deg: 45.0, periods: 8, n_high: 2.35, n_low: 1.46 },
        1.0,
        1.0,
    );
    {
        let surf = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap();
        surf.response_front = SurfaceResponse::Coating { coating: dichroic.clone() };
        surf.response_back = SurfaceResponse::Coating { coating: dichroic.seen_from_behind() };
    }
    let mut s = lit(s);
    s.fidelity = FidelityProfile { coating_model: CoatingModel::Nominal, ..Default::default() };
    let g = GpuScene3::build(&s, BuildOpts::default());
    let rt = match &g.surfaces[0].front {
        GpuResponse::CoatingGrid { rt, .. } => rt.to_vec(),
        other => panic!("{other:?}"),
    };
    let (lam_d, cos_d, _) = dichroic.nominal_point();
    let want = coating_rt_at(&dichroic, 1.0, 1.0, lam_d, cos_d).unwrap();
    for v in rt.iter() {
        assert!((v[0] as f64 - want.r()).abs() < 1e-6 && (v[1] as f64 - want.t()).abs() < 1e-6, "{v:?}");
    }
}

/// What a coated face costs to bake: protected silver (tabulated metal under
/// a silica overcoat) at 64 wavelengths, both sides, 128 angles each. The
/// spec's budget is 20 ms per face; run in release:
/// `cargo test --release -p oc-gpu --features gpu3 --test coating_preview3 bake_cost -- --ignored --nocapture`.
#[test]
#[ignore = "timing; run in release"]
fn bake_cost() {
    let mut s = Scene3::new();
    s.add_tube(
            Vec3::ZERO,
            0.0,
            TubeDims { inner_diameter_mm: 20.0, outer_diameter_mm: 24.0, length_mm: 100.0 },
            BoreFinish::Metal { metal: BoreMetal::Silver, protected: true },
        )
        .unwrap();
    let lines: Vec<(f64, f64)> = (0..64).map(|i| (0.3 + 1.2 * i as f64 / 63.0, 1.0)).collect();
    s.add_collimated_beam(Vec3::new(-200.0, 0.0, 0.0), 5.0, 5.0, 1, Spectrum::Lines(lines));
    let t0 = std::time::Instant::now();
    let reps = 20;
    for _ in 0..reps {
        std::hint::black_box(GpuScene3::build(&s, BuildOpts::default()));
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("one protected-silver face, both sides, 64 wavelengths x 128 angles: {ms:.2} ms");
    assert!(ms < 20.0 || cfg!(debug_assertions), "{ms:.2} ms per face");
}
