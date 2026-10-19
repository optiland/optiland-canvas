//! What a detector measures: power, irradiance, radiant intensity,
//! illuminance.
//!
//! The bins hold flux whichever it is. The measure is the divisor, and the
//! divisor is the whole content of the thing: area makes irradiance, solid
//! angle makes intensity. So these tests are about the divisor being right,
//! and they get their oracles from the definitions rather than from a vendor.
#![cfg(feature = "trace3")]

mod common;
use common::*;

use oc_core::id::LayerId;
use oc_core::material::LAMBDA_D;
use oc_core::scene3::{
    AngularDistribution3, DetectorMeasure, SamplingSequence, Scene3, Source3, SourceSampling,
    SpatialDistribution3,
};
use oc_core::vec3::{Isometry3, Ray3, Vec3};
use oc_trace::{trace3, trace_scene3};
use std::f64::consts::TAU;

/// A point source at the origin emitting along `+X`, one unit of flux.
///
/// The sequence is Sobol and that is not a detail. An even grid lays its
/// samples on `⌈√n⌉` strata of the polar coordinate, and a ring boundary that
/// falls inside a stratum takes the whole stratum with it: the innermost ring
/// of a six-ring histogram comes out 3.6% high on a grid however many rays are
/// launched, because doubling the count doubles the strata and moves nothing.
/// The (0,2)-sequence has no such alignment to lose and lands inside 0.05%.
fn point_source(scene: &mut Scene3, angular: AngularDistribution3, samples: u32) {
    let id = scene.alloc_source_id();
    scene.sources.push(Source3 {
        id,
        transform: Isometry3::from_pos(Vec3::ZERO),
        spatial: SpatialDistribution3::Point,
        angular,
        spectrum: oc_core::scene::Spectrum::Monochromatic(LAMBDA_D),
        flux: 1.0,
        samples: SourceSampling {
            spatial_samples: 1,
            angular_samples: samples,
            seed: 0,
            sequence: SamplingSequence::Sobol,
        },
        coherence: None,
    });
    let _ = LayerId(0);
}

/// Every ring reads back the intensity it was handed.
///
/// One ray per ring, aimed down the middle of it, carrying exactly that ring's
/// solid angle times a chosen intensity. Read it back and the intensity has to
/// be the one chosen, on every ring, because the only thing between the two is
/// the divisor. No sampling and no statistics: if the solid angle were wrong
/// anywhere the ring would say so to the last digit.
#[test]
fn a_ring_reports_the_intensity_it_was_given() {
    const RINGS: usize = 8;
    let cone_deg: f64 = 40.0;
    let cone = cone_deg.to_radians();
    let i0 = 3.25;

    let mut scene = Scene3::new();
    let det = add_rect_detector(&mut scene, Vec3::ZERO, -Vec3::X, 5.0, 5.0, [RINGS, 1]);
    set_detector_measure(
        &mut scene,
        det,
        DetectorMeasure::Intensity {
            half_angle_deg: cone_deg,
        bins: [RINGS, 1],
        layout: Default::default(),
        },
    );
    scene.validate().unwrap();

    // One azimuth sector, so a ring's solid angle is the full annulus.
    let omega = |k: usize| {
        let t0 = cone * k as f64 / RINGS as f64;
        let t1 = cone * (k + 1) as f64 / RINGS as f64;
        TAU * (t0.cos() - t1.cos())
    };

    let rays: Vec<_> = (0..RINGS)
        .map(|k| {
            let theta = cone * (k as f64 + 0.5) / RINGS as f64;
            let dir = Vec3::new(theta.cos(), theta.sin(), 0.0);
            launch(
                Ray3::new(dir * -30.0, dir).unwrap(),
                LAMBDA_D,
                i0 * omega(k),
            )
        })
        .collect();

    let res = trace3(&scene, &rays, &no_ghosts());
    let d = detector_named(&res, det);
    assert_eq!(d.hits, RINGS as u64, "one ray per ring lands");

    let intensity = d.radiant_intensity().expect("the detector measures it");
    for (k, got) in intensity.iter().enumerate() {
        assert!(
            (got - i0).abs() < 1e-12,
            "ring {k} reads {got} for an intensity of {i0}"
        );
    }
}

/// The histogram's solid angles add up to the cone it covers.
///
/// `2π(1 − cos θ)` is the cap's own solid angle, and the rings are a partition
/// of the cap, so the sum is not an approximation that gets better with more
/// rings. Two rings are as exact as two hundred.
#[test]
fn the_rings_partition_the_cone_exactly() {
    for (cone_deg, rings, sectors) in [(15.0, 2, 1), (45.0, 7, 5), (90.0, 64, 64)] {
        let mut scene = Scene3::new();
        let det = add_rect_detector(&mut scene, Vec3::ZERO, -Vec3::X, 5.0, 5.0, [rings, sectors]);
        set_detector_measure(
            &mut scene,
            det,
            DetectorMeasure::Intensity {
                half_angle_deg: cone_deg,
            bins: [rings, sectors],
            layout: Default::default(),
        },
    );
        scene.validate().unwrap();

        let d = &scene.detectors[0];
        let cells = d.direction_cell_solid_angles().expect("a direction grid");
        assert_eq!(cells.len(), rings * sectors);
        let sum: f64 = cells.iter().sum();
        let cap = TAU * (1.0 - cone_deg.to_radians().cos());
        assert!(
            (sum - cap).abs() < 1e-12,
            "{rings}x{sectors} over {cone_deg} degrees sums to {sum}, cap is {cap}"
        );
        assert!(cells.iter().all(|w| *w > 0.0), "no empty cell");
    }
}

/// A cosine emitter reads as a cosine.
///
/// Lambertian is the definition every LED datasheet's polar plot is drawn
/// against: radiance constant, so intensity falls as `cos θ`. The exact value
/// for a ring is its flux-weighted mean, `(cos θ₀ + cos θ₁)/2`, which is what
/// this compares against rather than the cosine at some midpoint.
///
/// The residual left is the sampler's and it is under a third of a percent;
/// see [`point_source`] for why the sequence had to be the Sobol one.
#[test]
fn a_lambertian_source_falls_off_as_the_cosine() {
    const RINGS: usize = 6;
    let cone_deg: f64 = 60.0;
    let cone: f64 = cone_deg.to_radians();

    let mut scene = Scene3::new();
    point_source(&mut scene, AngularDistribution3::LambertianHemisphere, 65536);
    // Wide enough that the cone lands on it well inside the edges.
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(20.0, 0.0, 0.0),
        -Vec3::X,
        400.0,
        400.0,
        [RINGS, 1],
    );
    set_detector_measure(
        &mut scene,
        det,
        DetectorMeasure::Intensity {
            half_angle_deg: cone_deg,
        bins: [RINGS, 1],
        layout: Default::default(),
        },
    );
    scene.validate().unwrap();

    let res = trace_scene3(&scene, &no_ghosts());
    let d = detector_named(&res, det);
    let intensity = d.radiant_intensity().expect("the detector measures it");

    // A Lambertian hemisphere of flux Φ has I(0) = Φ/π on axis.
    let i0 = 1.0 / std::f64::consts::PI;
    for (k, got) in intensity.iter().enumerate() {
        let t0 = cone * k as f64 / RINGS as f64;
        let t1 = cone * (k + 1) as f64 / RINGS as f64;
        let want = i0 * (t0.cos() + t1.cos()) / 2.0;
        assert!(
            (got - want).abs() / want < 3e-3,
            "ring {k} reads {got:.6}, the cosine law says {want:.6}"
        );
    }
    // And it falls, ring after ring, which a flat emitter would not.
    for k in 1..RINGS {
        assert!(
            intensity[k] < intensity[k - 1],
            "ring {k} is not below ring {}",
            k - 1
        );
    }
}

/// A source uniform in solid angle reads flat, and stops where it stops.
///
/// The counterpart to the cosine: `UniformConeSolidAngle` spreads flux evenly
/// over steradians, so its intensity is `Φ / 2π(1 − cos α)` inside the cone and
/// nothing outside. A detector cone wider than the source's therefore reads a
/// plateau and then a cliff, and the cliff is where the source's own half
/// angle is.
#[test]
fn a_cone_uniform_in_solid_angle_reads_flat_and_then_stops() {
    const RINGS: usize = 8;
    let source_deg: f64 = 20.0;
    let det_deg = 40.0;

    let mut scene = Scene3::new();
    point_source(
        &mut scene,
        AngularDistribution3::UniformConeSolidAngle {
            half_angle_rad: source_deg.to_radians(),
        },
        65536,
    );
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(20.0, 0.0, 0.0),
        -Vec3::X,
        400.0,
        400.0,
        [RINGS, 1],
    );
    set_detector_measure(
        &mut scene,
        det,
        DetectorMeasure::Intensity {
            half_angle_deg: det_deg,
        bins: [RINGS, 1],
        layout: Default::default(),
        },
    );
    scene.validate().unwrap();

    let res = trace_scene3(&scene, &no_ghosts());
    let d = detector_named(&res, det);
    let intensity = d.radiant_intensity().expect("the detector measures it");

    let want = 1.0 / (TAU * (1.0 - source_deg.to_radians().cos()));
    // The detector's rings are 5 degrees wide, so rings 0 to 3 are inside the
    // source's 20 and rings 4 to 7 are past it.
    for (k, got) in intensity.iter().take(4).enumerate() {
        assert!(
            (got - want).abs() / want < 3e-3,
            "ring {k} reads {got:.4}, flat would be {want:.4}"
        );
    }
    for (k, got) in intensity.iter().enumerate().skip(4) {
        assert!(
            *got == 0.0,
            "ring {k} is past the source's cone and reads {got}"
        );
    }
}

/// Light past the cone is named, not dropped.
///
/// An angular histogram that only covers part of the sky leaves the rest
/// somewhere, and the somewhere is a field rather than a rounding error. Flux
/// in the cone plus flux outside it has to be the flux that arrived.
#[test]
fn flux_past_the_cone_is_counted_where_it_went() {
    let mut scene = Scene3::new();
    point_source(&mut scene, AngularDistribution3::LambertianHemisphere, 4096);
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(20.0, 0.0, 0.0),
        -Vec3::X,
        400.0,
        400.0,
        [8, 4],
    );
    // A narrow cone on a wide emitter, so most of the light misses the
    // histogram and lands in the field that says so.
    set_detector_measure(
        &mut scene,
        det,
        DetectorMeasure::Intensity {
            half_angle_deg: 20.0,
        bins: [8, 4],
        layout: Default::default(),
        },
    );
    scene.validate().unwrap();

    let res = trace_scene3(&scene, &no_ghosts());
    let d = detector_named(&res, det);
    let binned: f64 = d.direction_flux.iter().sum();
    assert!(
        (binned + d.outside_cone_flux - d.total_incident).abs() < 1e-12,
        "{binned} in the cone plus {} outside is not {}",
        d.outside_cone_flux,
        d.total_incident
    );
    assert!(
        d.outside_cone_flux > 0.5 * d.total_incident,
        "most of a hemisphere is outside a 20 degree cone: {}",
        d.outside_cone_flux / d.total_incident
    );
    // And the face histogram carried on filling, so the mode costs no picture.
    let face: f64 = d.incident_flux.iter().sum();
    assert!(
        (face - d.total_incident).abs() < 1e-12,
        "the face still holds every hit: {face}"
    );
}

/// Power and irradiance are the same light and different numbers.
///
/// The measure changes what a reading means, not what the tracer collects, so
/// the two readings off one detector differ by exactly one cell area.
#[test]
fn irradiance_is_power_divided_by_the_cell() {
    let mut scene = Scene3::new();
    let det = add_rect_detector(
        &mut scene,
        Vec3::new(10.0, 0.0, 0.0),
        -Vec3::X,
        8.0,
        8.0,
        [8, 8],
    );
    scene.validate().unwrap();
    let rays = [launch(
        Ray3::new(Vec3::new(-5.0, 0.5, 0.5), Vec3::X).unwrap(),
        LAMBDA_D,
        2.0,
    )];

    let res = trace3(&scene, &rays, &no_ghosts());
    let d = detector_named(&res, det);
    let (power, power_unit) = d.readings();
    assert_eq!(power_unit, "flux");

    // The same scene, asked for irradiance instead.
    set_detector_measure(&mut scene, det, DetectorMeasure::Irradiance);
    scene.validate().unwrap();
    let res = trace3(&scene, &rays, &no_ghosts());
    let d = detector_named(&res, det);
    let (irradiance, unit) = d.readings();
    assert_eq!(unit, "flux/mm\u{b2}");

    // A 16 mm face binned eight ways has cells 2 mm square, so the two
    // readings differ by the four square millimetres between them.
    let area = d.pixel_area_mm2.unwrap();
    assert!((area - 4.0).abs() < 1e-12, "pixel area {area}");
    for (p, e) in power.iter().zip(&irradiance) {
        assert!((p / area - e).abs() < 1e-12);
    }
    assert!(
        (power.iter().sum::<f64>() - 2.0).abs() < 1e-12,
        "two units of flux arrived"
    );
    assert!(
        (irradiance.iter().sum::<f64>() - 0.5).abs() < 1e-12,
        "and two units over four square millimetres is half a unit of each"
    );
}

/// Illuminance is the light meter's reading: 683 lm/W times flux times
/// `V(λ)`, over the cell in square metres, and the colour beside it is the
/// same flux times the three colour-matching functions.
///
/// Two rays in one 2 mm cell: 2 W at 555 nm, where `V` is 1, and 1 W at
/// 700 nm, where it is 0.004102. The cell's lux follow from the table by hand;
/// a pool of workers must add them up the same way.
#[test]
fn illuminance_weighs_each_wavelength_by_the_eye() {
    use oc_core::colorimetry::{cie_1931, LM_PER_W};
    let mut scene = Scene3::new();
    let det = add_rect_detector(&mut scene, Vec3::new(10.0, 0.0, 0.0), -Vec3::X, 8.0, 8.0, [8, 8]);
    set_detector_measure(&mut scene, det, DetectorMeasure::Illuminance);
    scene.validate().unwrap();
    let ray = |l: f64, flux: f64| launch(Ray3::new(Vec3::new(-5.0, 0.5, 0.5), Vec3::X).unwrap(), l, flux);
    let rays = [ray(0.555, 2.0), ray(0.700, 1.0)];
    let res = trace3(&scene, &rays, &no_ghosts());
    let d = detector_named(&res, det);
    let (lux, unit) = d.readings();
    assert_eq!(unit, "lx");

    let m2 = 4.0e-6;
    let want = LM_PER_W * (2.0 * 1.0 + 1.0 * 0.004102) / m2;
    let lit = lux.iter().position(|v| *v > 0.0).expect("one lit cell");
    assert!((lux[lit] / want - 1.0).abs() < 1e-12, "{} lx against {want}", lux[lit]);
    assert_eq!(lux.iter().filter(|v| **v > 0.0).count(), 1);

    let colour = d.colour_per_m2().unwrap();
    let (a, b) = (cie_1931(0.555), cie_1931(0.700));
    for k in 0..3 {
        let want = LM_PER_W * (2.0 * a[k] + b[k]) / m2;
        assert!((colour[lit][k] - want).abs() <= 1e-9 * want.abs().max(1.0), "channel {k}");
    }
    assert_eq!(colour[lit][1], lux[lit], "Y is the illuminance");

    // Power still counts watts, whatever the eye makes of them.
    assert!((d.total_incident - 3.0).abs() < 1e-12);
    // Another measure keeps no colour.
    set_detector_measure(&mut scene, det, DetectorMeasure::Irradiance);
    let res = trace3(&scene, &rays, &no_ghosts());
    assert!(detector_named(&res, det).colour_per_m2().is_none());
}

/// A line has no area, so it cannot be asked for one.
///
/// The v1 line probe is kept so old documents still trace, and it is kept
/// honest by never being given a height nobody authored. Asking it for a flux
/// density is the same mistake, and the scene refuses it instead of inventing
/// a denominator.
#[test]
fn a_line_probe_cannot_be_asked_for_irradiance() {
    use oc_core::scene3::{
        Detector3, DetectorGeometry3, DetectorResponse, DetectorTermination, ObjectCommon,
    };

    let mut scene = Scene3::new();
    let id = scene.alloc_object_id();
    scene.detectors.push(Detector3 {
        common: ObjectCommon::new(id, LayerId(0)),
        geometry: DetectorGeometry3::LineProbeLegacy {
            a_local: Vec3::new(0.0, -5.0, 0.0),
            b_local: Vec3::new(0.0, 5.0, 0.0),
            bins: 16,
        },
        bins: [16, 1],
        response: DetectorResponse::Unity,
        termination: DetectorTermination::Terminate,
        measure: DetectorMeasure::Power,
    });
    scene.validate().expect("a line probe measuring flux is fine");

    scene.detectors[0].measure = DetectorMeasure::Irradiance;
    let err = scene.validate().expect_err("but not one measuring per mm2");
    assert!(
        format!("{err}").contains("area"),
        "the refusal should say why: {err}"
    );
}

/// The cone has to be a cone.
#[test]
fn an_intensity_cone_outside_its_range_is_refused() {
    for bad in [0.0, -10.0, 90.001, f64::NAN] {
        let mut scene = Scene3::new();
        let det = add_rect_detector(&mut scene, Vec3::ZERO, -Vec3::X, 5.0, 5.0, [4, 4]);
        set_detector_measure(
            &mut scene,
            det,
            DetectorMeasure::Intensity {
                half_angle_deg: bad,
            bins: [4, 4],
            layout: Default::default(),
        },
    );
        assert!(
            scene.validate().is_err(),
            "a {bad} degree cone should be refused"
        );
    }
    // And the two ends of the range are not.
    for good in [0.001, 90.0] {
        let mut scene = Scene3::new();
        let det = add_rect_detector(&mut scene, Vec3::ZERO, -Vec3::X, 5.0, 5.0, [4, 4]);
        set_detector_measure(
            &mut scene,
            det,
            DetectorMeasure::Intensity {
                half_angle_deg: good,
            bins: [4, 4],
            layout: Default::default(),
        },
    );
        scene.validate().expect("a real cone");
    }
}

/// A grid measure over `cone_deg` with `bins` cells, on a detector at the
/// origin facing back down `-X`.
fn grid_detector(cone_deg: f64, bins: [usize; 2]) -> (Scene3, oc_core::id::ObjectId) {
    let mut scene = Scene3::new();
    let det = add_rect_detector(&mut scene, Vec3::ZERO, -Vec3::X, 5.0, 5.0, [8, 8]);
    set_detector_measure(
        &mut scene,
        det,
        DetectorMeasure::Intensity {
            half_angle_deg: cone_deg,
            bins,
            layout: oc_core::scene3::DirectionLayout::Grid,
        },
    );
    scene.validate().unwrap();
    (scene, det)
}

/// The grid's cells add up to the square of directions it covers: four times
/// the solid angle a `tan c` by `tan c` rectangle subtends from above its
/// corner at unit height, and the whole hemisphere at a 90° cone.
#[test]
fn the_grid_partitions_its_square_of_directions_exactly() {
    for (cone_deg, bins) in [
        (15.0, [3, 3]),
        (45.0, [7, 5]),
        (89.0, [31, 31]),
        (90.0, [64, 64]),
    ] {
        let (scene, _) = grid_detector(cone_deg, bins);
        let cells = scene.detectors[0].direction_cell_solid_angles().unwrap();
        assert_eq!(cells.len(), bins[0] * bins[1]);
        let t = f64::tan(cone_deg.to_radians());
        let square = if cone_deg == 90.0 {
            TAU
        } else {
            4.0 * (t * t / (1.0 + 2.0 * t * t).sqrt()).atan()
        };
        let sum: f64 = cells.iter().sum();
        assert!(
            (sum - square).abs() < 1e-12,
            "{bins:?} over {cone_deg}: {sum} vs {square}"
        );
        assert!(cells.iter().all(|w| *w > 0.0), "no empty cell");
        // Symmetric about both axes.
        let [nu, nv] = bins;
        for j in 0..nv {
            for i in 0..nu {
                let w = cells[j * nu + i];
                assert!((w - cells[j * nu + (nu - 1 - i)]).abs() < 1e-15);
                assert!((w - cells[(nv - 1 - j) * nu + i]).abs() < 1e-15);
            }
        }
    }
}

/// Every grid cell reads back the intensity it was handed: one ray down the
/// middle of each cell, carrying that cell's solid angle times a chosen
/// intensity, lands in that cell and reads that intensity.
#[test]
fn a_grid_cell_reports_the_intensity_it_was_given() {
    let (cone_deg, bins) = (30.0_f64, [5usize, 3usize]);
    let (scene, det) = grid_detector(cone_deg, bins);
    let d = &scene.detectors[0];
    let omega = d.direction_cell_solid_angles().unwrap();
    let cone = cone_deg.to_radians();
    let i0 = 2.5;
    let mut rays = Vec::new();
    for j in 0..bins[1] {
        for i in 0..bins[0] {
            let au = -cone + 2.0 * cone * (i as f64 + 0.5) / bins[0] as f64;
            let av = -cone + 2.0 * cone * (j as f64 + 0.5) / bins[1] as f64;
            let local = Vec3::new(au.tan(), av.tan(), 1.0).normalize();
            assert_eq!(d.direction_cell(local), Some(j * bins[0] + i));
            let dir = d.common.transform.rotate(local);
            rays.push(launch(
                Ray3::new(dir * -30.0, dir).unwrap(),
                LAMBDA_D,
                i0 * omega[j * bins[0] + i],
            ));
        }
    }
    let res = trace3(&scene, &rays, &no_ghosts());
    let r = detector_named(&res, det);
    assert_eq!(r.hits, rays.len() as u64);
    for (k, got) in r.radiant_intensity().unwrap().iter().enumerate() {
        assert!((got - i0).abs() < 1e-12, "cell {k} reads {got}");
    }
    // Past the cone in either angle: counted, binned nowhere.
    let local = Vec3::new(f64::tan(cone + 0.01), 0.0, 1.0).normalize();
    assert_eq!(d.direction_cell(local), None);
}
