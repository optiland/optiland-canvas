//! The ideal mirror in the f64 reference: a flat surface that reflects as a
//! mirror of the powers it is given, with no aberration of its own.
//!
//! A collimated beam comes to one point at `f` along the reflected axis at any
//! angle of incidence, two powers make two line foci along the right axes, a
//! zero power is the flat mirror exactly, nothing is lost, and a body patch
//! that asks for one is refused.
//!
//! The tolerance on every focus is 1e-9 mm. The construction aims each ray at
//! the focus exactly, so what is left is f64 rounding over a 100 mm path: the
//! worst miss measured is under 1e-13 mm. A real curved mirror would miss by
//! millimetres here: a sphere used at 45° puts its two line foci at `f cos θ`
//! and `f / cos θ`, 70.7 and 141 mm for `f = 100`. Astigmatism or coma left
//! in the bend would show many orders of magnitude over the tolerance, and
//! the tolerance is still four orders over the rounding, so it does not
//! depend on the platform's last bits.

#![cfg(feature = "trace3")]

mod common;

use common::*;
use oc_core::id::{LayerId, ObjectId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::polarization::{PolRay, Stokes};
use oc_core::scene3::{
    Body3, ObjectCommon, PlanarShape2, SolidGeometry3, SpectralScalar, SurfaceGeometry3,
    SurfaceObject3, SurfaceProps3, SurfaceResponse,
};
use oc_core::vec3::{Isometry3, Ray3, Vec3};
use oc_trace::{trace3, CapabilityIssue, LaunchRay3, TraceConfig3, TraceResult3};

/// The rounding bound on a focus, mm (see the module notes).
const TOL_MM: f64 = 1e-9;

/// A frame at `pos` whose local `+Z` is `z` and local `+X` is `x` made square
/// to it.
fn framed(pos: Vec3, x: Vec3, z: Vec3) -> Isometry3 {
    let z = z.normalize();
    let x = (x - z * x.dot(z)).normalize();
    let y = z.cross(x);
    let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(x, y, z));
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap()
}

fn ideal(px: f64, py: f64) -> SurfaceResponse {
    SurfaceResponse::IdealMirror {
        power_x_per_mm: px,
        power_y_per_mm: py,
    }
}

/// A square standalone surface with the same response on both faces.
fn add_surface(scene: &mut Scene3, transform: Isometry3, half: f64, response: SurfaceResponse) -> ObjectId {
    let id = scene.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = transform;
    scene.surfaces.push(SurfaceObject3 {
        common,
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Rectangle {
                half_width_mm: half,
                half_height_mm: half,
            },
        },
        response_front: response.clone(),
        response_back: response,
    });
    id
}

/// The mirror's geometry for a beam along world `+X` meeting it at `aoi_deg`,
/// turned in the world XY plane: the normal facing the beam, local X in the
/// plane of incidence (tangential) and local Y along world Z (sagittal).
/// Returns the frame and the reflected axis.
fn mirror_at(aoi_deg: f64) -> (Isometry3, Vec3) {
    let t = aoi_deg.to_radians();
    let normal = Vec3::new(-t.cos(), t.sin(), 0.0);
    let local_x = Vec3::Z.cross(normal);
    let axis = Vec3::X - normal * (2.0 * Vec3::X.dot(normal));
    (framed(Vec3::ZERO, local_x, normal), axis)
}

/// A detector square to `axis`, `dist` mm down it from the origin, facing
/// back up it, its `u` along `u_world` made square to the axis.
fn add_square_detector(scene: &mut Scene3, axis: Vec3, dist: f64, u_world: Vec3) -> ObjectId {
    let id = add_rect_detector(scene, axis * dist, -axis, 20.0, 20.0, [4, 4]);
    let d = scene.detectors.iter_mut().find(|d| d.common.id == id).unwrap();
    d.common.transform = framed(axis * dist, u_world, -axis);
    id
}

/// A 9 × 9 collimated grid along `+X`, 8 mm square, from `x = -20`.
fn beam() -> Vec<LaunchRay3> {
    let mut rays = Vec::new();
    for i in -4..=4 {
        for j in -4..=4 {
            let o = Vec3::new(-20.0, i as f64, j as f64);
            rays.push(launch(Ray3::new(o, Vec3::X).unwrap(), LAMBDA_D, 1.0));
        }
    }
    rays
}

fn cfg() -> TraceConfig3 {
    TraceConfig3 {
        record_detector_hits: true,
        max_records: 1 << 16,
        ..TraceConfig3::default()
    }
}

/// Every ray's leg after the mirror, as (start, unit direction).
fn reflected_legs(res: &TraceResult3) -> Vec<(Vec3, Vec3)> {
    res.segments
        .iter()
        .filter(|s| s.parent_path_id.is_some())
        .map(|s| (s.a, (s.b - s.a).normalize()))
        .collect()
}

/// How far the line through `a` along unit `d` passes from `p`.
fn miss(a: Vec3, d: Vec3, p: Vec3) -> f64 {
    (p - a).cross(d).length()
}

/// Equal powers at 0°, 30°, 45° and 60°: every reflected ray of the beam runs
/// through the one point `f` down the reflected axis, and the readout's focus
/// along each of the detector's axes is that point.
///
/// The detector stands square to the axis 10 mm short of the focus, so both
/// axes read +10 mm. A diverging mirror is checked the same way: every leg,
/// run backward, comes from the virtual focus `|f|` behind the surface.
#[test]
fn equal_powers_focus_a_collimated_beam_to_one_point_at_any_angle() {
    for (aoi, f) in [(0.0, 100.0), (30.0, 100.0), (45.0, 100.0), (60.0, 100.0), (45.0, -80.0)] {
        let (frame, axis) = mirror_at(aoi);
        let mut s = Scene3::new();
        add_surface(&mut s, frame, 15.0, ideal(1.0 / f, 1.0 / f));
        let dist = if f > 0.0 { f - 10.0 } else { 40.0 };
        let det = add_square_detector(&mut s, axis, dist, frame.rotate(Vec3::X));
        s.validate().unwrap();
        let res = trace3(&s, &beam(), &cfg());

        let legs = reflected_legs(&res);
        assert_eq!(legs.len(), 81, "{aoi}°: one reflected leg per ray");
        let focus = axis * f;
        let worst = legs.iter().map(|(a, d)| miss(*a, *d, focus)).fold(0.0, f64::max);
        eprintln!("{aoi}°, f = {f}: worst miss of the focus {worst:.3e} mm");
        assert!(worst < TOL_MM, "{aoi}°, f = {f}: a ray misses the focus by {worst} mm");

        let line = detector_named(&res, det).spectral[0];
        assert_eq!(line.hits, 81, "{aoi}°: every ray lands");
        let [su, sv] = line.best_focus_uv_mm().unwrap().map(Option::unwrap);
        let want = f - dist;
        eprintln!("{aoi}°, f = {f}: focus along u {su:.12} mm, along v {sv:.12} mm, want {want}");
        assert!((su - want).abs() < TOL_MM, "{aoi}°: u focus {su}, want {want}");
        assert!((sv - want).abs() < TOL_MM, "{aoi}°: v focus {sv}, want {want}");
    }
}

/// At the focus itself, the spot of the whole 8 mm beam is a point, on the
/// axis. (Short of it the spot is off centre at an oblique angle: the
/// footprint is tilted to the axis, so its near edge is further from the
/// focus than its far edge, and the converging cone is not centred on it.)
#[test]
fn the_spot_at_the_focus_is_a_point() {
    for aoi in [0.0, 30.0, 45.0, 60.0] {
        let (frame, axis) = mirror_at(aoi);
        let mut s = Scene3::new();
        add_surface(&mut s, frame, 15.0, ideal(0.01, 0.01));
        let det = add_square_detector(&mut s, axis, 100.0, frame.rotate(Vec3::X));
        s.validate().unwrap();
        let res = trace3(&s, &beam(), &cfg());
        let line = detector_named(&res, det).spectral[0];
        let r = line.rms_radius().unwrap();
        eprintln!("{aoi}°: RMS spot radius at the focus {r:.3e} mm");
        assert!(r < TOL_MM, "{aoi}°: RMS radius {r} mm at the focus");
        let c = line.centroid().unwrap().length();
        assert!(c < TOL_MM, "{aoi}°: the point is {c} mm off the axis");
    }
}

/// Two powers, at 45°: `f_x` along the mirror's local X (here tangential) and
/// `f_y` along local Y (sagittal) come to two line foci at their own focal
/// lengths. With the detector's `u` on local X carried square to the axis and
/// 30 mm down it, `u` reads `f_x - 30` and `v` reads `f_y - 30`; swapping the
/// powers swaps the readings. At each line focus the spot has no width
/// across that line and the full beam's worth along it.
#[test]
fn two_powers_make_two_line_foci_along_their_own_axes() {
    let (frame, axis) = mirror_at(45.0);
    for (fx, fy) in [(100.0, 40.0), (40.0, 100.0)] {
        let trace_at = |dist: f64| {
            let mut s = Scene3::new();
            add_surface(&mut s, frame, 15.0, ideal(1.0 / fx, 1.0 / fy));
            let det = add_square_detector(&mut s, axis, dist, frame.rotate(Vec3::X));
            s.validate().unwrap();
            let res = trace3(&s, &beam(), &cfg());
            detector_named(&res, det).spectral[0]
        };

        // Every leg crosses the line through each focus that runs along the
        // other axis: the x focus is a line along local Y, the y one along X.
        let mut s = Scene3::new();
        add_surface(&mut s, frame, 15.0, ideal(1.0 / fx, 1.0 / fy));
        s.validate().unwrap();
        let res = trace3(&s, &beam(), &cfg());
        let e_x = {
            let x = frame.rotate(Vec3::X);
            (x - axis * x.dot(axis)).normalize()
        };
        let e_y = axis.cross(e_x);
        for (a, d) in reflected_legs(&res) {
            // Where the leg is `fx` down the axis, it is on the axis in x.
            let at_fx = a + d * ((fx - a.dot(axis)) / d.dot(axis));
            let at_fy = a + d * ((fy - a.dot(axis)) / d.dot(axis));
            assert!(at_fx.dot(e_x).abs() < TOL_MM, "x height {} at f_x", at_fx.dot(e_x));
            assert!(at_fy.dot(e_y).abs() < TOL_MM, "y height {} at f_y", at_fy.dot(e_y));
        }

        let line = trace_at(30.0);
        let [su, sv] = line.best_focus_uv_mm().unwrap().map(Option::unwrap);
        eprintln!("f_x = {fx}, f_y = {fy}: focus along u {su:.12} mm, along v {sv:.12} mm past 30 mm");
        assert!((su - (fx - 30.0)).abs() < TOL_MM, "u focus {su}, want {}", fx - 30.0);
        assert!((sv - (fy - 30.0)).abs() < TOL_MM, "v focus {sv}, want {}", fy - 30.0);

        let [ru, rv] = trace_at(fx).rms_uv().unwrap();
        eprintln!("at f_x = {fx}: RMS u {ru:.3e} mm, v {rv:.3} mm");
        assert!(ru < TOL_MM && rv > 0.5, "at f_x: {ru} across, {rv} along");
        let [ru, rv] = trace_at(fy).rms_uv().unwrap();
        eprintln!("at f_y = {fy}: RMS u {ru:.3} mm, v {rv:.3e} mm");
        assert!(rv < TOL_MM && ru > 0.5, "at f_y: {rv} across, {ru} along");
    }
}

/// Both powers zero is a `Mirror` of reflectivity 1, to the bit: the same
/// legs, detector readouts, polarization states, ledger, terminations, events
/// and capability report, for unpolarized and polarized light, with the
/// polarization switched on and off.
#[test]
fn zero_power_is_the_flat_mirror_bit_for_bit() {
    let scene = |response: SurfaceResponse| {
        let (frame, axis) = mirror_at(35.0);
        let mut s = Scene3::new();
        add_surface(&mut s, frame, 15.0, response);
        add_square_detector(&mut s, axis, 60.0, Vec3::Z);
        s.validate().unwrap();
        s
    };
    let flat = scene(SurfaceResponse::Mirror {
        reflectivity: SpectralScalar::Constant(1.0),
    });
    let zero = scene(ideal(0.0, 0.0));

    let mut rays = beam();
    // Linearly polarized at 30° to world Z, and a skew ray.
    let d = Vec3::new(1.0, 0.1, -0.05).normalize();
    rays.push(LaunchRay3 {
        ray: Ray3::new(Vec3::new(-20.0, 1.5, 2.5), d).unwrap(),
        lambda_um: LAMBDA_D,
        flux: 1.0,
        source: None,
        pol: Some(PolRay {
            stokes: Stokes { s0: 1.0, s1: 0.5, s2: 0.75f64.sqrt(), s3: 0.0 },
            x: Vec3::Z,
        }),
    });
    for polarization in [true, false] {
        let cfg = TraceConfig3 { polarization, record_events: true, ..cfg() };
        let (a, b) = (trace3(&flat, &rays, &cfg), trace3(&zero, &rays, &cfg));
        assert_eq!(format!("{:?}", a.segments), format!("{:?}", b.segments));
        assert_eq!(format!("{:?}", a.detectors), format!("{:?}", b.detectors));
        assert_eq!(format!("{:?}", a.ledger), format!("{:?}", b.ledger));
        assert_eq!(format!("{:?}", a.terminations), format!("{:?}", b.terminations));
        assert_eq!(format!("{:?}", a.events), format!("{:?}", b.events));
        assert_eq!(
            format!("{:?}", a.capability_report),
            format!("{:?}", b.capability_report)
        );
    }
}

/// Full reflection: every launched unit lands on the detector, nothing is
/// absorbed or lost, and the mirror makes one branch per ray even with ghosts
/// on. Polarized light keeps its degree of polarization, and its frame stays
/// square to the bent ray.
#[test]
fn an_ideal_mirror_reflects_all_the_light_in_one_branch() {
    for aoi in [0.0, 30.0, 45.0, 60.0] {
        let (frame, axis) = mirror_at(aoi);
        let mut s = Scene3::new();
        add_surface(&mut s, frame, 15.0, ideal(0.01, 0.025));
        let det = add_square_detector(&mut s, axis, 30.0, Vec3::Z);
        s.validate().unwrap();
        let mut rays = beam();
        for r in &mut rays {
            r.pol = Some(PolRay {
                stokes: Stokes { s0: 1.0, s1: 0.0, s2: 1.0, s3: 0.0 },
                x: Vec3::Y,
            });
        }
        let res = trace3(&s, &rays, &TraceConfig3 { ghosts: true, ..cfg() });
        assert_eq!(res.segments.len(), 2 * 81, "{aoi}°: one leg in, one out");
        let l = &res.ledger;
        assert_eq!(l.launched, 81.0);
        assert!((l.detected_incident - 81.0).abs() < 1e-12, "{aoi}°: {}", l.detected_incident);
        assert_eq!(l.absorbed_surface, 0.0, "{aoi}°");
        assert_eq!(l.numerical_loss + l.truncated_flux + l.escaped, 0.0, "{aoi}°");
        assert!(res.capability_report.rejected.is_empty());
        let face = s.detectors.iter().find(|d| d.common.id == det).unwrap().common.transform;
        let records = &detector_named(&res, det).records;
        assert_eq!(records.len(), 81);
        for h in records {
            assert!((h.stokes.dop() - 1.0).abs() < 1e-12, "{aoi}°: {:?}", h.stokes);
            let arriving = face.rotate(h.incident_dir_local);
            assert!(h.frame_x.dot(arriving).abs() < 1e-12, "{aoi}°: frame off the ray");
            assert!((h.frame_x.length() - 1.0).abs() < 1e-12);
        }
    }
}

/// An ideal mirror authored on a body's patch is refused: the bend needs a
/// plane the patch does not define. The ray ends there, its flux booked as
/// numerical loss, and the report names the surface.
#[test]
fn an_ideal_mirror_on_a_body_patch_is_rejected() {
    let mut s = Scene3::new();
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::IDENTITY;
    let mut surface = SurfaceProps3::default();
    surface.default.from_outside = ideal(0.01, 0.01);
    s.bodies.push(Body3 {
        common,
        geometry: SolidGeometry3::Box {
            half_extents_mm: Vec3::new(5.0, 10.0, 10.0),
        },
        material: Material::AbbePoint { nd: 1.5168, vd: 64.17 },
        absorption: Default::default(),
        surface,
        priority: 0,
    });
    s.validate().unwrap();
    let ray = launch(Ray3::new(Vec3::new(-20.0, 1.0, 2.0), Vec3::X).unwrap(), LAMBDA_D, 1.0);
    let res = trace3(&s, &[ray], &cfg());
    assert!(
        res.capability_report.rejected.iter().any(|r| matches!(
            r,
            CapabilityIssue::UnsupportedSurfaceResponse { surface, .. } if surface.object.0 == id.0
        )),
        "{:?}",
        res.capability_report.rejected
    );
    assert_eq!(res.ledger.numerical_loss, 1.0);
    assert_eq!(res.segments.len(), 1, "nothing leaves the patch");
}
