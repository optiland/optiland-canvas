//! Optiland's sample designs, as the canvas gallery ships them, held to
//! Optiland's own trace.
//!
//! `optiland_canvas.gallery` writes each design as a scene in
//! `web/public/gallery`, and beside them `checks.json`: rays Optiland traced
//! through the design at every field and wavelength (the chief ray, both
//! meridional marginal rays and a sagittal one), with where each crosses the
//! image plane. For every one of them this suite checks
//!
//! - that it is a ray the scene's source for that field emits: a collimated
//!   field's ray runs in the beam's direction and crosses the beam's disc at
//!   the pupil radius Optiland sampled; a point object's ray leaves the point
//!   at the angle that pupil radius makes inside its cone. A design that aims
//!   its rays through the real stop gets beams that overfill each field's
//!   pupil, trimmed by the stop, and there the ray lies inside its beam;
//! - that the canvas's reference tracer, started there, lands it on the image
//!   detector where Optiland lands it.
//!
//! The glasses are the catalogs' own formulas and the surfaces Optiland's, so
//! the two traces solve the same geometry, and the landing points agree to the
//! rounding of two independent intersection solvers. The first check is what
//! makes the second one about the scene rather than about the ray: a source
//! placed wrong fails it even where the lens is right.
#![cfg(feature = "trace3")]

use oc_core::scene3::{AngularDistribution3, Scene3, SpatialDistribution3};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{depth_for_scene3, trace3, LaunchRay3, TraceConfig3};
use serde_json::Value;

/// Where the gallery lives, from this crate.
const GALLERY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/public/gallery");

/// Landing points agree to this, relative to the system's length. Measured:
/// about 1e-15 for every design but the aspheric singlet, whose r^2 term is
/// refitted and lands within 5e-12. This leaves room for another platform's
/// rounding, not for a wrong surface.
const LANDING: f64 = 1e-10;
/// A ray is a source's ray to this: its direction to 1e-12, its crossing of
/// the source's disc to this relative to the disc.
const ON_SOURCE: f64 = 1e-9;

fn vec3(v: &Value) -> Vec3 {
    let a = v.as_array().expect("a 3-vector");
    Vec3::new(
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    )
}

fn read(name: &str) -> String {
    std::fs::read_to_string(format!("{GALLERY}/{name}"))
        .unwrap_or_else(|e| panic!("{GALLERY}/{name}: {e}"))
}

/// The length the landing tolerance scales with: the scene's extent along
/// the axis.
fn length(scene: &Scene3) -> f64 {
    let xs = scene
        .bodies
        .iter()
        .map(|b| b.common.transform.pos.x)
        .chain(scene.detectors.iter().map(|d| d.common.transform.pos.x))
        .chain(scene.sources.iter().map(|s| s.transform.pos.x));
    let (lo, hi) = xs.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
        (lo.min(x), hi.max(x))
    });
    (hi - lo).max(1.0)
}

/// Where the ray starts in the scene, having checked it is one `source`
/// emits. `Err` says how it is not. `aimed` says the source overfills a pupil
/// the stop trims, so the ray need only lie inside the beam.
fn launch_point(
    scene: &Scene3,
    source: usize,
    pupil: (f64, f64),
    aimed: bool,
    origin: Vec3,
    dir: Vec3,
) -> Result<Vec3, String> {
    let s = &scene.sources[source];
    let p = pupil.0.hypot(pupil.1);
    // A collimated source's ray, where it crosses the beam's own plane,
    // square to the source's local X, in the source's frame.
    let on_beam = |local_dir: Vec3| -> Result<(Vec3, Vec3), String> {
        let beam = s.transform.rotate(local_dir.normalize());
        if (beam - dir).length() > 1e-12 {
            return Err(format!("runs along {dir:?}, and the beam along {beam:?}"));
        }
        let n = s.transform.rotate(Vec3::X);
        let t = (s.transform.pos - origin).dot(n) / dir.dot(n);
        let at = origin + dir * t;
        Ok((at, s.transform.inverse_transform_point(at)))
    };
    match (s.spatial, s.angular) {
        (
            SpatialDistribution3::Disc { radius_mm },
            AngularDistribution3::Direction { local_dir },
        ) => {
            let (at, local) = on_beam(local_dir)?;
            let from_centre = local.y.hypot(local.z);
            let slack = ON_SOURCE * radius_mm.max(1.0);
            let fits = if aimed {
                from_centre <= radius_mm + slack
            } else {
                (from_centre - p * radius_mm).abs() <= slack
            };
            if !fits {
                return Err(format!(
                    "crosses the disc {from_centre} mm from its centre, of radius {radius_mm} mm, \
                     at pupil radius {p}"
                ));
            }
            Ok(at)
        }
        (
            SpatialDistribution3::Rectangle {
                half_width_mm,
                half_height_mm,
            },
            AngularDistribution3::Direction { local_dir },
        ) if aimed => {
            let (at, local) = on_beam(local_dir)?;
            let slack = ON_SOURCE * half_width_mm.max(half_height_mm).max(1.0);
            if local.y.abs() > half_width_mm + slack || local.z.abs() > half_height_mm + slack {
                return Err(format!(
                    "crosses the beam at ({}, {}) mm, outside its {half_width_mm} x \
                     {half_height_mm} mm half-extents",
                    local.y, local.z
                ));
            }
            Ok(at)
        }
        (
            SpatialDistribution3::Point,
            AngularDistribution3::UniformConeSolidAngle { half_angle_rad },
        ) => {
            let at = s.transform.pos;
            if (origin - at).length() > ON_SOURCE * at.length().max(1.0) {
                return Err(format!("starts at {origin:?}, and the source is at {at:?}"));
            }
            // A pupil radius p is a direction whose sine off the cone's axis
            // is p times the cone's: Optiland samples the object-space NA.
            let axis = s.transform.rotate(Vec3::X);
            let angle = dir.dot(axis).clamp(-1.0, 1.0).acos();
            let expected = (p * half_angle_rad.sin()).asin();
            if (angle - expected).abs() > 1e-12 {
                return Err(format!(
                    "leaves {angle} rad off the cone's axis, where pupil radius {p} is {expected}"
                ));
            }
            Ok(at)
        }
        other => Err(format!(
            "the source is {other:?}, which the gallery does not build"
        )),
    }
}

#[test]
fn every_gallery_design_traces_where_optiland_traces_it() {
    let checks: Value = serde_json::from_str(&read("checks.json")).expect("checks.json");
    let designs = checks["designs"].as_array().expect("designs");
    assert!(!designs.is_empty(), "the gallery has no designs");
    let mut failures = Vec::new();
    let mut total = 0;
    for d in designs {
        let id = d["id"].as_str().unwrap();
        let aimed = d["aimed"].as_bool().unwrap_or(false);
        let scene = Scene3::from_json(&read(d["file"].as_str().unwrap()))
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        // The depth the app traces this scene to, which a 21-element lens
        // needs more of than the default.
        let cfg = TraceConfig3 {
            ghosts: false,
            record_detector_hits: true,
            max_depth: depth_for_scene3(&scene, false, 0),
            ..TraceConfig3::default()
        };
        let scale = length(&scene);
        let mut worst = 0.0_f64;
        for r in d["rays"].as_array().unwrap() {
            total += 1;
            let pupil = r["pupil"].as_array().unwrap();
            let pupil = (pupil[0].as_f64().unwrap(), pupil[1].as_f64().unwrap());
            let lambda = r["wavelength_um"].as_f64().unwrap();
            let what = format!(
                "{id}, field {} at {lambda} µm, pupil {pupil:?}",
                r["field"].as_f64().unwrap()
            );
            let dir = vec3(&r["dir"]).normalize();
            let source = r["source"].as_u64().unwrap() as usize;
            let start = match launch_point(&scene, source, pupil, aimed, vec3(&r["origin"]), dir) {
                Ok(p) => p,
                Err(why) => {
                    failures.push(format!("{what}: not a ray of its source: {why}"));
                    continue;
                }
            };
            let launch = LaunchRay3 {
                ray: Ray3::new(start, dir).unwrap(),
                lambda_um: lambda,
                flux: 1.0,
                source: None,
                pol: None,
            };
            let res = trace3(&scene, &[launch], &cfg);
            assert!(
                res.capability_report.rejected.is_empty(),
                "{what}: {:?}",
                res.capability_report.rejected
            );
            let hits: Vec<Vec3> = res
                .detectors
                .iter()
                .flat_map(|det| det.records.iter().map(|h| h.point_world))
                .collect();
            let [hit] = hits.as_slice() else {
                failures.push(format!("{what}: {} detector hits, not one", hits.len()));
                continue;
            };
            let off = (*hit - vec3(&r["image"])).length();
            worst = worst.max(off / scale);
            if off > LANDING * scale {
                failures.push(format!(
                    "{what}: lands {off:.3e} mm from Optiland's point ({hit:?})"
                ));
            }
        }
        println!("{id:32} worst landing {worst:.2e} of its length");
    }
    assert!(total > 0, "checks.json has no rays");
    assert!(
        failures.is_empty(),
        "{} of {total} rays:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Every gallery scene is a document the app opens: it validates, its one
/// declared path exports, and the stop that path declares is on it.
#[test]
fn every_gallery_scene_exports_its_declared_path() {
    let index: Value = serde_json::from_str(&read("index.json")).expect("index.json");
    for d in index["designs"].as_array().unwrap() {
        if d["status"] != "ported" {
            continue;
        }
        let id = d["id"].as_str().unwrap();
        let scene = Scene3::from_json(&read(d["file"].as_str().unwrap()))
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        scene.validate().unwrap_or_else(|e| panic!("{id}: {e}"));
        let (source, detector) =
            oc_trace::default_path(&scene).unwrap_or_else(|e| panic!("{id}: {e}"));
        let seq =
            oc_trace::export_seq(&scene, source, detector).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(
            seq.stop.is_some(),
            "{id}: the declared stop is lost on export"
        );
    }
}
