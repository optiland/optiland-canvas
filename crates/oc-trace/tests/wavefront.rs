//! Wavefront (OPD) validation against the main Optiland engine.
//!
//! A wavefront is a surface of constant optical path length (equivalently
//! constant phase, or — since propagation time is OPL/c — constant time): every
//! point on it is reached by the same accumulated OPL = Σ nᵢ·dᵢ. That is exactly
//! what the M4 wavefront scrub draws, and what `Segment::opl0`/`opl1` carry. The
//! wavefront aberration across the pupil is the **optical path difference**,
//! OPD(ray) = OPL(ray) − OPL(chief), and it is the physically meaningful,
//! translation-invariant quantity to validate.
//!
//! The reference here is Optiland, whose `RealRays.opd` accumulates the same
//! optical path — `rays.opd += |t|·n_pre` at every surface
//! (`optiland/surfaces/standard_surface.py`). The golden OPD values below were
//! produced by tracing the identical systems through the real Optiland engine
//! (Python); see `validation/README.md` to reproduce. The two agree to ~5e-14 mm
//! (≈1e-10 waves at the d line, i.e. machine precision) — the OPD cancels the
//! reference tracer's deliberate 1e-7 mm ray-spawn offset, since every ray
//! crosses the same surfaces in the same media. The assertion tolerance (1e-11)
//! leaves margin for platform floating-point variation.

use oc_core::{
    ConicParams, Edge, Element, ElementId, Isometry2, Material, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_D,
};
use oc_trace::{trace, TraceConfig};

/// Signed-radius singlet (R1, R2), centre thickness `tc`, front vertex at the
/// local origin's −tc/2. Convention matches `validation/audit_compare` and
/// Optiland: both surfaces have axis +x and *signed* curvature c = 1/R, so the
/// sign of the radius alone sets which way each face bulges.
fn singlet(r1: f64, r2: f64, tc: f64, half_ap: f64, nd: f64) -> Element {
    let cv = |r: f64| if r.abs() >= 1e30 { 0.0 } else { 1.0 / r };
    let (c1, c2) = (cv(r1), cv(r2));
    let front = ConicParams {
        vertex: Vec2::new(-tc / 2.0, 0.0),
        axis: Vec2::new(1.0, 0.0),
        c: c1,
        k: 0.0,
        half_ap,
    };
    let rear = ConicParams {
        vertex: Vec2::new(tc / 2.0, 0.0),
        axis: Vec2::new(1.0, 0.0),
        c: c2,
        k: 0.0,
        half_ap,
    };
    let sag = |c: f64| c * half_ap * half_ap / (1.0 + (1.0 - c * c * half_ap * half_ap).sqrt());
    let ftip = -tc / 2.0 + sag(c1);
    let rtip = tc / 2.0 + sag(c2);
    Element::new(
        ElementId(0),
        vec![
            Edge::Conic(front),
            Edge::Line {
                a: Vec2::new(ftip, half_ap),
                b: Vec2::new(rtip, half_ap),
            },
            Edge::Conic(rear),
            Edge::Line {
                a: Vec2::new(rtip, -half_ap),
                b: Vec2::new(ftip, -half_ap),
            },
        ],
        Material::AbbePoint { nd, vd: 60.0 },
    )
}

/// Optical path length from the launch plane (`x = launch_x`) to the image plane
/// (`x = image_x`) for a collimated ray at height `y`, through `lens`. The exit
/// segment leaves the last surface in air (n = 1), so the residual air path to
/// the image plane is added geometrically.
fn opl_to_image(lens: &Element, y: f64, launch_x: f64, image_x: f64) -> f64 {
    let mut scene = Scene::new();
    scene.add_element(lens.clone());
    scene.sources.push(Source {
        kind: SourceKind::Point { half_angle: 0.0 },
        transform: Isometry2::new(Vec2::new(launch_x, y), 0.0),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 1,
    });
    let cfg = TraceConfig {
        ghosts: false,
        ..TraceConfig::default()
    };
    let segs = trace(&scene, &cfg).segments;
    // Exit segment: the one whose start is furthest along +x (the last surface
    // the ray leaves). `opl0` is the OPL at that surface; extend it in air.
    let s = segs
        .iter()
        .max_by(|a, b| a.a.x.partial_cmp(&b.a.x).unwrap())
        .expect("at least one segment");
    let d = (s.b - s.a).normalize();
    assert!(d.x.abs() > 1e-9, "exit ray must reach the image plane");
    s.opl0 + (image_x - s.a.x) / d.x
}

/// Assert the canvas OPD across the pupil matches Optiland's golden OPD.
fn assert_opd_matches(
    lens: &Element,
    launch_x: f64,
    image_x: f64,
    ys: &[f64],
    golden_opd: &[f64],
    tol: f64,
) {
    let opl_ref = opl_to_image(lens, ys[0], launch_x, image_x); // chief = axial
    let mut worst: f64 = 0.0;
    for (i, &y) in ys.iter().enumerate() {
        let opd = opl_to_image(lens, y, launch_x, image_x) - opl_ref;
        let err = (opd - golden_opd[i]).abs();
        worst = worst.max(err);
        assert!(
            err < tol,
            "y={y}: canvas OPD {opd:.12e} vs Optiland {:.12e} (Δ {err:.2e} mm)",
            golden_opd[i]
        );
    }
    println!("worst OPD Δ vs Optiland: {worst:.2e} mm");
}

// Biconvex R1=50, R2=−50, tc=6, n=1.5, half_ap=16. Front vertex at x=0 (lens
// centre at +3): launch plane 25 mm before it, image plane at x=60. Classic
// undercorrected spherical aberration — OPD grows ≈ y⁴.
#[test]
fn wavefront_opd_matches_optiland_biconvex() {
    let mut lens = singlet(50.0, -50.0, 6.0, 16.0, 1.5);
    lens.transform = Isometry2::from_pos(Vec2::new(3.0, 0.0));
    let ys = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
    // Optiland RealRays.opd at the image plane, relative to the axial ray (mm).
    let golden = [
        0.0,
        4.016333320735e-3,
        1.801910248157e-2,
        4.822559092892e-2,
        1.063039462914e-1,
        2.118188057069e-1,
        3.969391066745e-1,
        7.157422000440e-1,
    ];
    assert_opd_matches(&lens, -25.0, 60.0, &ys, &golden, 1e-11);
}

// Plano-convex, curved front R1=30, flat rear, tc=5, n=1.5168, half_ap=16.
// Front vertex at x=0 (lens centre at +2.5), launch 25 mm before, image at x=70.
#[test]
fn wavefront_opd_matches_optiland_planoconvex() {
    let mut lens = singlet(30.0, 1e99, 5.0, 16.0, 1.5168);
    lens.transform = Isometry2::from_pos(Vec2::new(2.5, 0.0));
    let ys = [0.0, 3.0, 6.0, 9.0, 12.0];
    let golden = [
        0.0,
        1.414156365247e-2,
        6.229078837441e-2,
        1.635215819558e-1,
        3.571760098197e-1,
    ];
    assert_opd_matches(&lens, -25.0, 70.0, &ys, &golden, 1e-11);
}
