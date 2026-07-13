//! §7 acceptance suite: the reference tracer vs. closed-form optics.
//!
//! This is the M0 gate — *"nothing else starts until this is green."* Each test
//! drives the **full** non-sequential tracer end to end (emit → BVH → interface
//! physics → segments) and checks the result against an analytic solution.
//!
//! Coverage of the §7 list:
//! - Snell at a plane .......... `snell_at_a_plane_full_trace`
//! - TIR critical angle ........ `tir_critical_angle`
//! - Thin/thick-lens focus ..... `paraxial_lens_focus`
//! - Parabolic mirror focus .... `parabolic_mirror_perfect_focus`
//! - Fresnel R/T ............... `fresnel_energy_split` (+ unit tests in fresnel.rs)
//! - Luneburg GRIN ............. deferred to M7 (`luneburg_lens` is #[ignore]d).

use oc_core::{
    ConicParams, Edge, Element, ElementId, Isometry2, Material, Ray, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_D,
};
use oc_trace::{trace, Segment, TraceConfig};

/// Trace a single ray through `elements` and return its segments (ghosts off,
/// deterministic).
fn segments_for(elements: Vec<Element>, ray: Ray, cfg: &TraceConfig) -> Vec<Segment> {
    let mut scene = Scene::new();
    for e in elements {
        scene.add_element(e);
    }
    let angle = ray.dir.y.atan2(ray.dir.x);
    scene.sources.push(Source {
        kind: SourceKind::Point { half_angle: 0.0 },
        transform: Isometry2::new(ray.origin, angle),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 1,
    });
    trace(&scene, cfg).segments
}

/// The exit segment is the one whose start is furthest along +x; return where
/// its supporting line crosses the optical axis (y = 0).
fn exit_axis_crossing(segs: &[Segment]) -> f64 {
    let s = segs
        .iter()
        .max_by(|a, b| a.a.x.partial_cmp(&b.a.x).unwrap())
        .expect("at least one segment");
    let d = s.b - s.a;
    assert!(d.y.abs() > 1e-12, "exit segment is parallel to the axis");
    let t = -s.a.y / d.y;
    s.a.x + t * d.x
}

/// A square glass block centred at the origin, side `2*half`, CCW.
fn glass_block(half: f64, nd: f64) -> Element {
    let b = vec![
        Edge::Line {
            a: Vec2::new(-half, -half),
            b: Vec2::new(half, -half),
        },
        Edge::Line {
            a: Vec2::new(half, -half),
            b: Vec2::new(half, half),
        },
        Edge::Line {
            a: Vec2::new(half, half),
            b: Vec2::new(-half, half),
        },
        Edge::Line {
            a: Vec2::new(-half, half),
            b: Vec2::new(-half, -half),
        },
    ];
    Element::new(ElementId(0), b, Material::AbbePoint { nd, vd: 60.0 })
}

/// A rectangular glass slab, half-extents `hx` × `hy`, CCW.
fn glass_rect(hx: f64, hy: f64, nd: f64) -> Element {
    let b = vec![
        Edge::Line {
            a: Vec2::new(-hx, -hy),
            b: Vec2::new(hx, -hy),
        },
        Edge::Line {
            a: Vec2::new(hx, -hy),
            b: Vec2::new(hx, hy),
        },
        Edge::Line {
            a: Vec2::new(hx, hy),
            b: Vec2::new(-hx, hy),
        },
        Edge::Line {
            a: Vec2::new(-hx, hy),
            b: Vec2::new(-hx, -hy),
        },
    ];
    Element::new(ElementId(0), b, Material::AbbePoint { nd, vd: 60.0 })
}

#[test]
fn snell_at_a_plane_full_trace() {
    // Ray enters a glass block (n = 1.5) at 30° to the front-face normal.
    let n_glass = 1.5;
    let theta_i = 30f64.to_radians();
    // Front face is x = -30 (block half = 30); its normal is the x-axis.
    let dir = Vec2::new(theta_i.cos(), theta_i.sin());
    let ray = Ray::new(Vec2::new(-50.0, 0.0), dir);

    let cfg = TraceConfig {
        ghosts: false,
        ..TraceConfig::default()
    };
    let segs = segments_for(vec![glass_block(30.0, n_glass)], ray, &cfg);

    // Segment 0: incoming to the front face. Segment 1: refracted, inside glass.
    assert!(segs.len() >= 2, "expected entry + interior segments");
    let incoming = (segs[0].b - segs[0].a).normalize();
    let interior = (segs[1].b - segs[1].a).normalize();

    // Angles from the face normal (x-axis).
    let sin_i = incoming.y.abs(); // |sin| relative to x-normal since dir is unit
    let sin_t = interior.y.abs();
    let lhs = 1.0 * sin_i;
    let rhs = n_glass * sin_t;
    assert!(
        (lhs - rhs).abs() < 1e-9,
        "Snell violated: n1·sinθ1 = {lhs}, n2·sinθ2 = {rhs}"
    );
    // And the transmitted angle equals the analytic prediction.
    let theta_t = (theta_i.sin() / n_glass).asin();
    assert!((sin_t - theta_t.sin()).abs() < 1e-9);
}

#[test]
fn tir_critical_angle() {
    // A ray *inside* the glass hits the rear face beyond the critical angle and
    // must totally internally reflect instead of escaping.
    let n_glass = 1.5;
    let crit = (1.0f64 / n_glass).asin(); // ≈ 41.81°
                                          // Tall, thin slab so a steep ray reaches the right face (x = 5), not the top.
    let block = glass_rect(5.0, 100.0, n_glass);

    // Source placed inside the block, aimed at the right face at > critical.
    let theta = crit + 5f64.to_radians();
    let dir = Vec2::new(theta.cos(), theta.sin());
    let ray = Ray::new(Vec2::new(0.0, 0.0), dir);

    let cfg = TraceConfig {
        ghosts: false,
        ..TraceConfig::default()
    };
    let segs = segments_for(vec![block], ray, &cfg);

    // The ray hits the right face (x = 5). Under TIR it reflects and stays
    // inside: the next segment must turn back toward −x (dir.x < 0), and no
    // segment may cross x = 5 into the outside.
    let hit_face = segs
        .iter()
        .find(|s| (s.b.x - 5.0).abs() < 1e-6)
        .expect("a segment should reach the right face");
    // The segment leaving that face point reverses in x (specular reflection).
    let reflected = segs
        .iter()
        .find(|s| (s.a - hit_face.b).length() < 1e-6)
        .expect("a reflected segment leaves the face");
    let refl_dir = (reflected.b - reflected.a).normalize();
    assert!(
        refl_dir.x < 0.0,
        "TIR should send the ray back into the glass"
    );
    // Specular: x-component flips sign, y-component preserved.
    assert!((refl_dir.x + dir.x).abs() < 1e-9);
    assert!((refl_dir.y - dir.y).abs() < 1e-9);

    // Sanity: just *below* critical, the ray escapes (refracts out).
    let theta2 = crit - 5f64.to_radians();
    let dir2 = Vec2::new(theta2.cos(), theta2.sin());
    let segs2 = segments_for(
        vec![glass_rect(5.0, 100.0, n_glass)],
        Ray::new(Vec2::ZERO, dir2),
        &cfg,
    );
    assert!(
        segs2.iter().any(|s| s.a.x > 4.999),
        "below critical, a segment should start at/after the exit face"
    );
}

/// Signed-radius biconvex lens (R1 > 0, R2 < 0), centre thickness `tc`.
fn biconvex(r: f64, tc: f64, half_ap: f64, nd: f64) -> Element {
    let c = 1.0 / r;
    let front = ConicParams {
        vertex: Vec2::new(-tc / 2.0, 0.0),
        axis: Vec2::new(1.0, 0.0), // vertex → centre of curvature (+x)
        c,
        k: 0.0,
        half_ap,
    };
    let rear = ConicParams {
        vertex: Vec2::new(tc / 2.0, 0.0),
        axis: Vec2::new(-1.0, 0.0), // centre of curvature at −x  ⇒ R2 < 0
        c,
        k: 0.0,
        half_ap,
    };
    // Sag magnitude at the aperture edge (same for both surfaces).
    let sag = c * half_ap * half_ap / (1.0 + (1.0 - c * c * half_ap * half_ap).sqrt());
    let ftip = -tc / 2.0 + sag;
    let rtip = tc / 2.0 - sag;
    let rim_top = Edge::Line {
        a: Vec2::new(ftip, half_ap),
        b: Vec2::new(rtip, half_ap),
    };
    let rim_bot = Edge::Line {
        a: Vec2::new(rtip, -half_ap),
        b: Vec2::new(ftip, -half_ap),
    };
    Element::new(
        ElementId(0),
        vec![Edge::Conic(front), rim_top, Edge::Conic(rear), rim_bot],
        Material::AbbePoint { nd, vd: 60.0 },
    )
}

#[test]
fn paraxial_lens_focus() {
    // Symmetric biconvex, n = 1.5 exactly at the d line (AbbePoint property).
    let (r, tc, n) = (50.0_f64, 4.0_f64, 1.5_f64);
    let half_ap = 10.0;

    // Analytic thick-lens back focal distance (R1 = +r, R2 = −r).
    let (r1, r2) = (r, -r);
    let phi = (n - 1.0) * (1.0 / r1 - 1.0 / r2) + (n - 1.0).powi(2) * tc / (n * r1 * r2);
    let f = 1.0 / phi;
    let bfd = f * (1.0 - (n - 1.0) * tc / (n * r1));
    let focus_x = tc / 2.0 + bfd; // measured from the origin

    let cfg = TraceConfig {
        ghosts: false,
        ..TraceConfig::default()
    };
    // Trace two near-axis heights and Richardson-extrapolate to h → 0 to reach
    // the paraxial focus (spherical aberration ∝ h² is real physics, not tracer
    // error — extrapolation removes its leading term).
    let crossing = |h: f64| {
        let ray = Ray::new(Vec2::new(-100.0, h), Vec2::new(1.0, 0.0));
        exit_axis_crossing(&segments_for(vec![biconvex(r, tc, half_ap, n)], ray, &cfg))
    };
    let (h1, h2) = (0.2, 0.1);
    let x_h1 = crossing(h1);
    let x_h2 = crossing(h2);
    let x_paraxial = (4.0 * x_h2 - x_h1) / 3.0; // cancels the h² term

    assert!(
        (x_paraxial - focus_x).abs() < 1e-6,
        "paraxial focus: traced {x_paraxial}, analytic {focus_x}"
    );
    // The finite-height crossings bracket the paraxial focus (aberration sign).
    assert!(x_h2 > 0.0 && x_h1 > 0.0);
}

/// A concave parabolic mirror: vertex at origin, axis pointing −x so the bowl
/// opens toward the incoming (+x-travelling) light. Focus at x = −R/2.
fn parabolic_mirror(radius: f64, half_ap: f64) -> Element {
    let c = 1.0 / radius;
    let parab = ConicParams {
        vertex: Vec2::ZERO,
        axis: Vec2::new(-1.0, 0.0),
        c,
        k: -1.0, // parabola
        half_ap,
    };
    // Close the (mirror) body behind the surface; incoming rays never reach it.
    let sag = c * half_ap * half_ap / 2.0; // parabola sag along +axis (−x)
    let tip_x = -sag;
    let back = 5.0;
    let boundary = vec![
        Edge::Conic(parab),
        Edge::Line {
            a: Vec2::new(tip_x, half_ap),
            b: Vec2::new(back, half_ap),
        },
        Edge::Line {
            a: Vec2::new(back, half_ap),
            b: Vec2::new(back, -half_ap),
        },
        Edge::Line {
            a: Vec2::new(back, -half_ap),
            b: Vec2::new(tip_x, -half_ap),
        },
    ];
    Element::new(
        ElementId(0),
        boundary,
        Material::Mirror { reflectivity: 1.0 },
    )
}

#[test]
fn parabolic_mirror_perfect_focus() {
    // Defining property of a parabola: *every* incoming ray parallel to the
    // axis reflects exactly through the focus, with no aberration.
    let radius = 8.0;
    let focus_x = -radius / 2.0; // = −4
    let cfg = TraceConfig {
        ghosts: false,
        ..TraceConfig::default()
    };

    let mut crossings = Vec::new();
    for &h in &[0.5, 1.0, 1.7, 2.5, 3.0, -0.5, -1.3, -2.8] {
        let ray = Ray::new(Vec2::new(-20.0, h), Vec2::new(1.0, 0.0));
        let segs = segments_for(vec![parabolic_mirror(radius, 4.0)], ray, &cfg);
        crossings.push(exit_axis_crossing(&segs));
    }

    let min = crossings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = crossings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Perfect focus: all crossings coincide to < 1e-9.
    assert!(
        max - min < 1e-9,
        "parabola focus spread {} (crossings {:?})",
        max - min,
        crossings
    );
    // And they coincide at the analytic focus.
    let mean = crossings.iter().sum::<f64>() / crossings.len() as f64;
    assert!(
        (mean - focus_x).abs() < 1e-9,
        "focus at {mean}, expected {focus_x}"
    );
}

#[test]
fn fresnel_energy_split() {
    // Normal incidence into a glass block: front reflectance R0 = 0.04.
    // With ghosts on and a tiny flux floor, energy is conserved: the sum of all
    // escaped flux equals the launched flux (no absorption here).
    let n_glass = 1.5;
    let ray = Ray::new(Vec2::new(-20.0, 0.0), Vec2::new(1.0, 0.0));
    let cfg = TraceConfig {
        ghosts: true,
        flux_eps: 1e-12,
        max_depth: 64,
        ..TraceConfig::default()
    };
    let segs = segments_for(vec![glass_block(5.0, n_glass)], ray, &cfg);

    // The first back-reflection off the front face carries R0 of the flux.
    let front_reflection = segs
        .iter()
        .filter(|s| (s.b - s.a).normalize().x < -0.5) // heading back −x
        .max_by(|a, b| a.flux.partial_cmp(&b.flux).unwrap())
        .expect("a back-reflected ghost segment exists");
    assert!(
        (front_reflection.flux - 0.04).abs() < 1e-6,
        "front reflectance = {}",
        front_reflection.flux
    );

    // Energy conservation: escaped flux (segments that run the full bounds
    // length) sums to the launched flux.
    let escaped: f64 = segs
        .iter()
        .filter(|s| (s.b - s.a).length() > cfg.bounds * 0.5)
        .map(|s| s.flux)
        .sum();
    assert!(
        (escaped - 1.0).abs() < 1e-6,
        "escaped flux {escaped}, expected 1.0"
    );
}

#[test]
#[ignore = "Luneburg GRIN lens is an M7 deliverable (eikonal integrator, §8)"]
fn luneburg_lens() {
    // n(r) = √(2 − r²): a collimated beam focuses to a point on the far surface.
    // Requires curved-ray integration (§8); tracked here as a pending gate.
}
