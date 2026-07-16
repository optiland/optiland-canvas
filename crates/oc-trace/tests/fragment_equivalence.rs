//! §13.4 acceptance: a placed [`SceneFragment`] traces exactly like the scene it
//! came from.
//!
//! # What is actually asserted
//!
//! The obvious test — build a scene "directly at world coordinates", build the
//! same scene as a fragment plus transform, compare — is circular here: the only
//! honest way to author the direct scene is to compose the transform onto each
//! object, which is exactly what `insert_fragment` does. It would compare
//! `compose` against itself and pass no matter what.
//!
//! So these tests assert the property that actually matters, and that a real bug
//! would break: **rigid invariance**. Geometric optics is invariant under rigid
//! motion, so tracing a module placed at `T` must yield the identity-placed trace
//! with `T` applied to it — same flux, same optical path length, same detector
//! histogram, positions moved by `T` and directions rotated by `T`.
//!
//! That catches every failure this layer can have: an element left behind, a
//! source whose aim did not rotate, a detector whose endpoints did not move, a
//! glass id that silently rebound to a different glass. Each is a divergence no
//! amount of `compose`-vs-`compose` agreement would reveal.
//!
//! Coverage of the §13.3 geometry list: line, arc, conic, asphere, and spline
//! each get a fragment; sources and detectors ride along in the pilot module.

use oc_core::{
    spherical_edge, ConicParams, Detector, Edge, Element, ElementId, Isometry2, Material, Scene,
    SceneFragment, Source, SourceKind, Spectrum, Transform2, Vec2, LAMBDA_D,
};
use oc_trace::{trace, TraceConfig, TraceResult};

/// Agreement required between a placed trace and the rigidly-moved reference.
#[derive(Clone, Copy)]
struct Tol {
    pos: f64,
    dir: f64,
    flux: f64,
    opl: f64,
}

/// §13.4 targets: exact to coordinate precision.
///
/// `pos` is an absolute floor that [`close`] widens in proportion to the
/// coordinate's own magnitude — §13.4 asks for 1e-12 mm "where the direct
/// representation permits", and at the 1e4 mm trace bounds one f64 ULP is
/// already ~1.8e-12, so an absolute 1e-12 is below the representable step.
const EXACT: Tol = Tol {
    pos: 1e-12,
    dir: 1e-12,
    flux: 1e-12,
    // OPL sums path lengths of order 1e4 mm, so it carries the reference
    // tracer's accumulated rounding rather than a coordinate-only error.
    opl: 1e-9,
};

/// Aspheres cannot meet [`EXACT`], for reasons that predate scene fragments.
///
/// `Edge::Asphere` has no closed form: `asphere_intersect` runs Newton on a
/// *numerically* differentiated residual, accepts a root at `|residual| < 1e-9`,
/// and finite-differences the surface normal with a step of `h = 1e-7`. A
/// central difference has a roundoff floor of about `eps/h ≈ 2e-9`, so the
/// normal — and hence every refracted direction downstream of it — is only good
/// to ~1e-9 regardless of frame. Re-expressing the same ray in a rotated frame
/// perturbs the iterates in the last bits, and the solver amplifies that to its
/// own accuracy rather than to f64's.
///
/// This is the "analytic asphere normals" debt in §17.6, and closing it would
/// tighten these numbers. Until then, asserting 1e-12 here would be asserting
/// something untrue about the tracer.
const ASPHERE: Tol = Tol {
    pos: 1e-4,
    dir: 1e-7,
    flux: 1e-7,
    opl: 1e-4,
};

/// Is `got` within `tol` of `want`?
///
/// `tol` is an absolute floor near the origin, widened by ~64 ULP at `want`'s
/// own magnitude. Both terms are needed. The §13.4 target of 1e-12 mm is a
/// *coordinate-only* tolerance, and the transform itself does meet it —
/// `oc-core`'s `compose` tests pin that directly, to 1e-12 and to bit-equality
/// under identity. What these tests compare is the far end of a full trace: a
/// deep ghost segment has refracted and reflected a dozen or more times, each
/// interaction rounding through `sqrt`/`normalize`, and the survivors land
/// hundreds of mm from the origin where one ULP is already ~5e-14. Agreement is
/// therefore bounded by the tracer's accumulated rounding at scene scale, which
/// is what this allowance describes.
fn close(got: Vec2, want: Vec2, tol: f64) -> bool {
    (got - want).length() <= tol + 64.0 * f64::EPSILON * want.length()
}

/// The placements a module realistically sees: identity, pure translation, the
/// openUC2 90° quadrant rotations on the 50 mm grid, and an awkward angle to
/// keep the tests honest about the general case.
fn placements() -> Vec<(&'static str, Transform2)> {
    use std::f64::consts::{FRAC_PI_2, PI};
    vec![
        ("identity", Transform2::IDENTITY),
        (
            "translate",
            Isometry2::from_pos(Vec2::new(150.0, -100.0)),
        ),
        ("rot 90", Isometry2::new(Vec2::new(50.0, 50.0), FRAC_PI_2)),
        ("rot 180", Isometry2::new(Vec2::new(-50.0, 100.0), PI)),
        (
            "rot 270",
            Isometry2::new(Vec2::new(0.0, -50.0), -FRAC_PI_2),
        ),
        (
            "oblique",
            Isometry2::new(Vec2::new(12.5, -7.25), 0.937_123),
        ),
    ]
}

fn trace_placed(fragment: &SceneFragment, t: Transform2, cfg: &TraceConfig) -> TraceResult {
    let mut scene = Scene::new();
    scene
        .insert_fragment(fragment, t)
        .expect("fragment places cleanly");
    trace(&scene, cfg)
}

/// Deterministic (no Russian roulette), ghosts followed to the depth cap.
///
/// `bounds` is tightened from the 1e4 mm default to suit these ~100 mm modules.
/// A ray that escapes is drawn to exactly `bounds`, so its endpoint is
/// `origin + bounds * dir`: at the default, a few-ULP direction difference is
/// levered into ~5e-11 mm of endpoint difference and the §13.4 target of 1e-12
/// becomes unreachable for reasons that have nothing to do with placement.
/// Bounding the trace near the scene's own size keeps the assertion strict and
/// about the thing under test.
fn config() -> TraceConfig {
    TraceConfig {
        bounds: 300.0,
        ..TraceConfig::default()
    }
}

/// Assert that placing `fragment` at each pose produces the identity-placed
/// trace, rigidly moved by that pose.
fn assert_rigid_invariance(fragment: &SceneFragment, label: &str, tol: Tol) {
    assert_rigid_invariance_with(fragment, label, tol, &config());
}

fn assert_rigid_invariance_with(
    fragment: &SceneFragment,
    label: &str,
    tol: Tol,
    cfg: &TraceConfig,
) {
    let base = trace_placed(fragment, Transform2::IDENTITY, cfg);
    assert!(
        !base.segments.is_empty(),
        "{label}: the reference placement traced nothing, so this proves nothing"
    );

    for (name, t) in placements() {
        let moved = trace_placed(fragment, t, cfg);
        assert_eq!(
            moved.segments.len(),
            base.segments.len(),
            "{label} @ {name}: segment count changed under a rigid motion"
        );

        for (i, (b, m)) in base.segments.iter().zip(&moved.segments).enumerate() {
            let want_a = t.transform_point(b.a);
            let want_b = t.transform_point(b.b);
            assert!(
                close(m.a, want_a, tol.pos),
                "{label} @ {name}: segment {i} start is {:?}, expected {want_a:?}",
                m.a
            );
            assert!(
                close(m.b, want_b, tol.pos),
                "{label} @ {name}: segment {i} end is {:?}, expected {want_b:?}",
                m.b
            );

            // Direction must rotate with the placement.
            let base_dir = b.b - b.a;
            if base_dir.length() > 1e-9 {
                let want = t.rotate(base_dir.normalize());
                let got = (m.b - m.a).normalize();
                assert!(
                    (got - want).length() <= tol.dir,
                    "{label} @ {name}: segment {i} direction is {got:?}, expected {want:?}"
                );
            }

            // Flux, wavelength, and optical path are invariant under rigid motion.
            assert!(
                (m.flux - b.flux).abs() <= tol.flux,
                "{label} @ {name}: segment {i} flux {} != {}",
                m.flux,
                b.flux
            );
            assert_eq!(
                m.lambda_um, b.lambda_um,
                "{label} @ {name}: segment {i} wavelength changed"
            );
            assert!(
                (m.opl0 - b.opl0).abs() <= tol.opl && (m.opl1 - b.opl1).abs() <= tol.opl,
                "{label} @ {name}: segment {i} OPL [{}, {}] != [{}, {}]",
                m.opl0,
                m.opl1,
                b.opl0,
                b.opl1
            );
        }

        // Detector readout is what the user actually sees a number for.
        assert_eq!(moved.detectors.len(), base.detectors.len());
        for (i, (b, m)) in base.detectors.iter().zip(&moved.detectors).enumerate() {
            assert_eq!(
                m.hits, b.hits,
                "{label} @ {name}: detector {i} hit count changed"
            );
            assert!(
                (m.total_flux - b.total_flux).abs() <= tol.flux,
                "{label} @ {name}: detector {i} total flux {} != {}",
                m.total_flux,
                b.total_flux
            );
            assert!(
                (m.flux_u - b.flux_u).abs() <= tol.flux,
                "{label} @ {name}: detector {i} centroid moved"
            );
            assert!(
                (m.flux_opl - b.flux_opl).abs() <= tol.opl,
                "{label} @ {name}: detector {i} flux-weighted OPL changed"
            );
            for (k, (bb, mm)) in b.bins.iter().zip(&m.bins).enumerate() {
                assert!(
                    (mm - bb).abs() <= tol.flux,
                    "{label} @ {name}: detector {i} bin {k} is {mm}, expected {bb}"
                );
            }
        }
    }
}

// --- Scene building ---------------------------------------------------------

fn sag(c: f64, k: f64, y: f64) -> f64 {
    let d = 1.0 - (1.0 + k) * c * c * y * y;
    c * y * y / (1.0 + d.max(0.0).sqrt())
}

/// A collimated source aimed along local +x, upstream of the optics.
fn collimated_source(spectrum: Spectrum) -> Source {
    Source {
        kind: SourceKind::Collimated { half_width: 6.0 },
        transform: Isometry2::from_pos(Vec2::new(-40.0, 0.0)),
        spectrum,
        flux: 1.0,
        rays: 7,
    }
}

/// Bin count for every detector in these tests.
///
/// **Odd on purpose — do not "round" this to 16.** Binning is a floor:
/// `bin = (u * nb) as usize`. These scenes are symmetric about the optical axis,
/// so the axial ray lands at exactly `u = 0.5`; with an even `nb` that is exactly
/// a bin boundary (`0.5 * 16 = 8.0`), and a one-ULP shift from the placement
/// drops the whole ray into bin 7 instead. That is a real discontinuity in the
/// histogram, but it is a property of binning any knife-edge sample, not
/// something rigid placement should be blamed for — and `flux_u`, the continuous
/// centroid, agrees to 1e-12 across it either way.
///
/// An odd `nb` maps `u = 0.5` to the *centre* of the middle bin, which is where
/// the physics being tested here actually lives.
const BINS: usize = 15;

/// A collimated source narrow enough to hit only a flat face of
/// [`spline_hexagon`], which spans y ∈ [-4.5, 4.5].
fn narrow_source() -> Source {
    Source {
        kind: SourceKind::Collimated { half_width: 3.0 },
        transform: Isometry2::from_pos(Vec2::new(-40.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 7,
    }
}

/// A detector across the local axis, downstream of the optics.
fn detector() -> Detector {
    Detector::new(Vec2::new(60.0, -15.0), Vec2::new(60.0, 15.0), BINS)
}

/// Wrap `elements` into a module: source upstream, detector downstream.
///
/// Built as a `Scene` and read back with [`SceneFragment::from_scene`] — the
/// same path a real `optics/canvas.ocanvas` module file takes.
fn module(elements: Vec<Element>) -> SceneFragment {
    module_with_spectrum(elements, Spectrum::Monochromatic(LAMBDA_D))
}

fn module_with_spectrum(elements: Vec<Element>, spectrum: Spectrum) -> SceneFragment {
    module_with_source(elements, collimated_source(spectrum))
}

fn module_with_source(elements: Vec<Element>, source: Source) -> SceneFragment {
    let mut scene = Scene::new();
    for e in elements {
        scene.add_element(e);
    }
    scene.sources.push(source);
    scene.detectors.push(detector());
    SceneFragment::from_scene(&scene).expect("module scene is a valid fragment")
}

fn glass() -> Material {
    Material::AbbePoint { nd: 1.5168, vd: 64.17 }
}

/// A square block: exercises `Edge::Line`.
fn line_block() -> Element {
    let h = 8.0;
    let boundary = vec![
        Edge::Line {
            a: Vec2::new(-h, -h),
            b: Vec2::new(h, -h),
        },
        Edge::Line {
            a: Vec2::new(h, -h),
            b: Vec2::new(h, h),
        },
        Edge::Line {
            a: Vec2::new(h, h),
            b: Vec2::new(-h, h),
        },
        Edge::Line {
            a: Vec2::new(-h, h),
            b: Vec2::new(-h, -h),
        },
    ];
    Element::new(ElementId(0), boundary, glass())
}

/// A disc built from two semicircles: exercises `Edge::Arc` sweep handling
/// across a seam, where the two edges meet at absolute angles.
fn arc_disc() -> Element {
    use std::f64::consts::{PI, TAU};
    Element::new(
        ElementId(0),
        vec![
            Edge::Arc {
                center: Vec2::ZERO,
                r: 10.0,
                a0: 0.0,
                a1: PI,
            },
            Edge::Arc {
                center: Vec2::ZERO,
                r: 10.0,
                a0: PI,
                a1: TAU,
            },
        ],
        glass(),
    )
}

/// A disc bounded by a single full-turn arc.
///
/// The one-edge circle is the representation a catalogue module would naturally
/// use for a round optic, and its sweep angles are absolute, so it is the first
/// thing a naive per-geometry rotation would break. Placement carries it in the
/// element transform instead, leaving `a0`/`a1` untouched.
fn full_circle_disc() -> Element {
    Element::new(
        ElementId(0),
        vec![Edge::Arc {
            center: Vec2::ZERO,
            r: 10.0,
            a0: 0.0,
            a1: std::f64::consts::TAU,
        }],
        glass(),
    )
}

/// A biconvex singlet: exercises `Edge::Conic` (plus `Edge::Line` rims).
fn conic_lens(material: Material) -> Element {
    let (r1, r2, thickness, half_ap) = (50.0, -50.0, 5.0, 10.0);
    let (zf, zr) = (-thickness / 2.0, thickness / 2.0);
    let front = spherical_edge(Vec2::new(zf, 0.0), Vec2::X, r1, half_ap);
    let rear = spherical_edge(Vec2::new(zr, 0.0), Vec2::X, r2, half_ap);
    let ftip = zf + sag(1.0 / r1, 0.0, half_ap);
    let rtip = zr + sag(1.0 / r2, 0.0, half_ap);
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
        vec![front, rim_top, rear, rim_bot],
        material,
    )
}

/// An aspheric singlet: exercises `Edge::Asphere`, whose normals are
/// finite-differenced and so are sensitive to frame handling.
fn asphere_lens() -> Element {
    let half_ap = 10.0;
    let mut coeffs = [0.0; 8];
    coeffs[0] = 1.0e-6; // y^4
    coeffs[1] = -2.0e-9; // y^6
    let base = ConicParams {
        vertex: Vec2::new(-2.5, 0.0),
        axis: Vec2::X,
        c: 1.0 / 40.0,
        k: -0.6,
        half_ap,
    };
    let front = Edge::Asphere { base, coeffs };
    let rear = spherical_edge(Vec2::new(2.5, 0.0), Vec2::X, -60.0, half_ap);
    let ftip = -2.5 + sag(base.c, base.k, half_ap)
        + coeffs[0] * half_ap.powi(4)
        + coeffs[1] * half_ap.powi(6);
    let rtip = 2.5 + sag(-1.0 / 60.0, 0.0, half_ap);
    let rim_top = Edge::Line {
        a: Vec2::new(ftip, half_ap),
        b: Vec2::new(rtip, half_ap),
    };
    let rim_bot = Edge::Line {
        a: Vec2::new(rtip, -half_ap),
        b: Vec2::new(ftip, -half_ap),
    };
    Element::new(ElementId(0), vec![front, rim_top, rear, rim_bot], glass())
}

/// A closed polyline: exercises `Edge::Spline`.
///
/// Two details are load-bearing, and both exist to keep this a test of geometry
/// placement rather than of epsilon robustness:
///
/// 1. The loop is closed by repeating the first point — `spline_intersect` walks
///    `ctrl.windows(2)` and does not close it for you.
/// 2. The vertices are offset by 30°, which puts a **flat face perpendicular to
///    the optical axis** on each side. Without the offset the faces meet the
///    axis at 60°, rays refract steeply into the interior, and — since TIR is
///    followed whether or not ghosts are on — the polygon becomes a chaotic
///    billiard. Corners are dispersive, so a ray that eventually lands within
///    `EPS_ORIGIN` (1e-7) of a vertex resolves to a different edge in a rotated
///    frame and jumps by that epsilon: the vertex degeneracy `Element::contains`
///    already flags as a known robustness risk, and the "geometry fuzzing" /
///    "scale-aware epsilons" debt in §17.6. Oriented flat-on, the hexagon
///    refracts like a slab and no ray goes near a corner.
///
/// [`narrow_source`] keeps every ray inside the flat faces, which span
/// y ∈ [-4.5, 4.5] at this radius.
fn spline_hexagon() -> Element {
    use std::f64::consts::{PI, TAU};
    let r = 9.0;
    let mut ctrl: Vec<Vec2> = (0..6)
        .map(|i| {
            let a = TAU * i as f64 / 6.0 + PI / 6.0;
            Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect();
    ctrl.push(ctrl[0]);
    Element::new(
        ElementId(0),
        vec![Edge::Spline { ctrl, degree: 1 }],
        glass(),
    )
}

/// A 45° fold mirror, placed by its own element transform — so a placed module
/// stacks the outer transform onto a non-identity local one.
fn fold_mirror() -> Element {
    let (hx, hy) = (0.75, 12.0);
    let boundary = vec![
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
    let mut e = Element::new(
        ElementId(0),
        boundary,
        Material::Mirror { reflectivity: 0.96 },
    );
    e.transform = Isometry2::new(Vec2::new(30.0, 0.0), std::f64::consts::FRAC_PI_4);
    e
}

// --- Per-geometry invariance (§13.3 list) -----------------------------------

#[test]
fn line_geometry_places_rigidly() {
    assert_rigid_invariance(&module(vec![line_block()]), "line block", EXACT);
}

#[test]
fn arc_geometry_places_rigidly() {
    assert_rigid_invariance(&module(vec![arc_disc()]), "arc disc", EXACT);
}

#[test]
fn a_one_edge_circle_places_rigidly() {
    assert_rigid_invariance(&module(vec![full_circle_disc()]), "full-circle disc", EXACT);
}

#[test]
fn a_placed_one_edge_circle_is_still_a_circle() {
    // The invariance check above compares a placed circle against a placed
    // circle, so it would also hold if placement quietly turned both into the
    // degenerate point the old sweep produced. Pin the absolute behaviour: a
    // placed circle refracts light and holds its medium at its new centre.
    let fragment = module(vec![full_circle_disc()]);
    let t = Isometry2::new(Vec2::new(120.0, -60.0), 0.7);
    let mut scene = Scene::new();
    scene.insert_fragment(&fragment, t).unwrap();

    assert!(
        scene.elements[0].contains(t.transform_point(Vec2::ZERO)),
        "the disc must still hold its medium at the placed centre"
    );
    let out = trace(&scene, &config());
    assert!(out.detectors[0].hits > 0, "no light reached the detector");
    assert!(out.detectors[0].total_flux > 0.0);
}

#[test]
fn conic_geometry_places_rigidly() {
    assert_rigid_invariance(&module(vec![conic_lens(glass())]), "conic lens", EXACT);
}

#[test]
fn asphere_geometry_places_rigidly() {
    assert_rigid_invariance(&module(vec![asphere_lens()]), "asphere lens", ASPHERE);
}

#[test]
fn spline_geometry_places_rigidly() {
    assert_rigid_invariance(
        &module_with_source(vec![spline_hexagon()], narrow_source()),
        "spline hexagon",
        EXACT,
    );
}

#[test]
fn spline_geometry_refracts_where_it_is_placed() {
    // Guard the fixture: light must actually enter, refract, and land, or the
    // invariance above would hold vacuously.
    let fragment = module_with_source(vec![spline_hexagon()], narrow_source());
    let out = trace_placed(
        &fragment,
        Isometry2::new(Vec2::new(30.0, -20.0), 0.4),
        &config(),
    );
    assert!(out.detectors[0].hits > 0, "no light reached the detector");
    assert!(out.detectors[0].total_flux > 0.0);
}

#[test]
fn mirror_with_a_local_transform_places_rigidly() {
    // The outer placement composes onto the mirror's own 45° pose.
    assert_rigid_invariance(&module(vec![fold_mirror()]), "fold mirror", EXACT);
}

// --- The pilot module (§16.2) -----------------------------------------------

#[test]
fn spectral_singlet_and_fold_mirror_place_rigidly() {
    // The folded-singlet demo in miniature: three-line source, catalog-glass
    // singlet, 45° fold mirror, detector — refraction, dispersion, reflection,
    // and Fresnel ghosts all at once.
    let mut scene = Scene::new();
    let nbk7 = scene.catalog.find("N-BK7").expect("builtin catalog has N-BK7");
    scene.add_element(conic_lens(Material::Catalog(nbk7)));
    scene.add_element(fold_mirror());
    scene.sources.push(collimated_source(Spectrum::Lines(vec![
        (0.4861, 1.0),
        (0.5876, 1.0),
        (0.6563, 1.0),
    ])));
    // The fold sends light along -y, so the detector sits below the axis.
    scene
        .detectors
        .push(Detector::new(Vec2::new(15.0, -40.0), Vec2::new(45.0, -40.0), BINS));

    let fragment = SceneFragment::from_scene(&scene).unwrap();
    assert_rigid_invariance(&fragment, "spectral folded singlet", EXACT);
}

#[test]
fn the_pilot_module_actually_exercises_the_physics_it_claims() {
    // Guard against the invariance tests passing vacuously: assert the pilot
    // really does refract, disperse, reflect, and land flux on the detector.
    let mut scene = Scene::new();
    let nbk7 = scene.catalog.find("N-BK7").unwrap();
    scene.add_element(conic_lens(Material::Catalog(nbk7)));
    scene.add_element(fold_mirror());
    scene.sources.push(collimated_source(Spectrum::Lines(vec![
        (0.4861, 1.0),
        (0.5876, 1.0),
        (0.6563, 1.0),
    ])));
    scene
        .detectors
        .push(Detector::new(Vec2::new(15.0, -40.0), Vec2::new(45.0, -40.0), BINS));

    let fragment = SceneFragment::from_scene(&scene).unwrap();
    let mut placed = Scene::new();
    placed
        .insert_fragment(&fragment, Isometry2::new(Vec2::new(100.0, 50.0), 0.5))
        .unwrap();
    let out = trace(&placed, &config());

    assert!(out.detectors[0].hits > 0, "no light reached the detector");
    assert!(out.detectors[0].total_flux > 0.0);
    let lambdas: std::collections::BTreeSet<u64> = out
        .segments
        .iter()
        .map(|s| (s.lambda_um * 1e4).round() as u64)
        .collect();
    assert_eq!(lambdas.len(), 3, "all three spectral lines should propagate");
}

// --- Sources and detectors --------------------------------------------------

#[test]
fn a_point_source_fan_places_rigidly() {
    let mut scene = Scene::new();
    scene.add_element(conic_lens(glass()));
    scene.sources.push(Source {
        kind: SourceKind::Point { half_angle: 0.12 },
        transform: Isometry2::from_pos(Vec2::new(-40.0, 0.0)),
        spectrum: Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        rays: 9,
    });
    scene.detectors.push(detector());
    let fragment = SceneFragment::from_scene(&scene).unwrap();
    assert_rigid_invariance(&fragment, "point-source fan", EXACT);
}

#[test]
fn an_absorber_places_rigidly() {
    let mut aperture = line_block();
    aperture.material = Material::Absorber;
    assert_rigid_invariance(&module(vec![aperture]), "absorber", EXACT);
}

// --- Multi-module scenes ----------------------------------------------------

#[test]
fn two_placements_of_one_module_are_independent() {
    // Two instances of the same module at different poses must each trace as
    // that module does alone — the shared fragment must not be mutated by
    // placement, and the two must not interfere through the merged catalog.
    let fragment = module(vec![conic_lens(glass())]);
    let a = Isometry2::from_pos(Vec2::new(0.0, 0.0));
    let b = Isometry2::new(Vec2::new(0.0, 300.0), std::f64::consts::FRAC_PI_2);

    let mut both = Scene::new();
    both.insert_fragment(&fragment, a).unwrap();
    both.insert_fragment(&fragment, b).unwrap();
    let combined = trace(&both, &config());

    let alone = trace_placed(&fragment, a, &config());
    assert_eq!(combined.detectors.len(), 2);
    // The first instance's detector is unaffected by the second's presence.
    assert_eq!(combined.detectors[0].hits, alone.detectors[0].hits);
    assert!((combined.detectors[0].total_flux - alone.detectors[0].total_flux).abs() <= EXACT.flux);
}

#[test]
fn placing_a_module_does_not_mutate_the_fragment() {
    let fragment = module(vec![conic_lens(glass())]);
    let before = fragment.clone();
    let mut scene = Scene::new();
    scene
        .insert_fragment(&fragment, Isometry2::new(Vec2::new(9.0, 9.0), 1.0))
        .unwrap();
    assert_eq!(fragment, before, "a catalogue fragment is reusable, so placement must not consume it");
}

