//! The coated scenes the profile-equivalence check (C33) traces, and the
//! fingerprint it compares.
//!
//! Built only from scene API that existed before the fidelity profile, so the
//! same module compiles at the pre-profile commit and records the numbers the
//! kernel gave then (`record_corpus3.rs`). `fidelity3.rs` traces the same
//! scenes under `Standard` and requires those numbers to the bit.

#![allow(dead_code)]

use oc_core::scene::Spectrum;
use oc_core::scene3::Scene3;
use oc_core::thinfilm::{film_coating, materials, metal_mirror, CoatingDesign, DesignKind};
use oc_core::tube::{BoreFinish, BoreMetal, TubeDims};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, trace_scene3, LaunchRay3, TraceConfig3, TraceResult3};

use crate::common::{add_rect_detector, launch};

/// The configurations every scene is traced under: the pre-profile default,
/// the scalar-unpolarized alias, ghosts off, and the deposits recorded.
pub fn configs() -> Vec<(&'static str, TraceConfig3)> {
    vec![
        ("default", TraceConfig3::default()),
        ("scalar", TraceConfig3 { polarization: false, ..TraceConfig3::default() }),
        ("no_ghosts", TraceConfig3 { ghosts: false, ..TraceConfig3::default() }),
        (
            "deposits",
            TraceConfig3 {
                object_ledger: true,
                deposit_bodies: [0, 64],
                ..TraceConfig3::default()
            },
        ),
    ]
}

/// A scene and how to launch into it: from its own sources, or explicit rays.
pub struct Case {
    pub name: &'static str,
    pub scene: Scene3,
    pub rays: Option<Vec<LaunchRay3>>,
}

impl Case {
    pub fn trace(&self, cfg: &TraceConfig3) -> TraceResult3 {
        match &self.rays {
            Some(r) => trace3(&self.scene, r, cfg),
            None => trace_scene3(&self.scene, cfg),
        }
    }
}

fn lines() -> Spectrum {
    Spectrum::Lines(vec![(0.45, 1.0), (0.55, 1.0), (0.65, 0.8)])
}

fn ar(kind: DesignKind) -> oc_core::thinfilm::Coating {
    film_coating(
        CoatingDesign { kind, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38, adapt: None },
        1.0,
        1.5168,
    )
}

pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();

    // The default document: a singlet and a white collimated beam.
    out.push(Case { name: "singlet", scene: Scene3::default_scene(), rays: None });

    // Two dichroic plates in series, folding the beam at their edge.
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = Spectrum::Lines(vec![(0.56, 1.0), (0.52, 0.5), (0.60, 0.5)]);
    s.add_plate_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 3.0, 135.0, 560.0, true).expect("first plate");
    s.add_plate_dichroic(Vec3::new(-35.0, 40.0, 0.0), 15.0, 3.0, -45.0, 560.0, true).expect("second plate");
    add_rect_detector(&mut s, Vec3::new(25.0, 40.0, 0.0), -Vec3::X, 20.0, 20.0, [4, 4]);
    out.push(Case { name: "two_dichroics", scene: s, rays: None });

    // A thin-film dichroic on a plate, both arms caught.
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = lines();
    s.add_film_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 135.0, 560.0, false).expect("film dichroic");
    add_rect_detector(&mut s, Vec3::new(-35.0, 40.0, 0.0), -Vec3::Y, 30.0, 30.0, [4, 4]);
    out.push(Case { name: "film_dichroic", scene: s, rays: None });

    // A protected silver mirror on a plate, tilted, and an AR-coated window
    // with a three-layer coat that is not its own mirror image.
    let mut s = Scene3::default_scene();
    s.sources[0].spectrum = lines();
    let mirror = metal_mirror(materials::silver(), Some((materials::silica(), 90.0)));
    s.add_coated_plate(Vec3::new(-45.0, 0.0, 0.0), 15.0, 3.0, 120.0, mirror);
    s.add_coated_plate(Vec3::new(-25.0, 0.0, 0.0), 15.0, 3.0, 160.0, ar(DesignKind::BroadbandArLegacy));
    add_rect_detector(&mut s, Vec3::new(-45.0, 60.0, 0.0), -Vec3::Y, 60.0, 60.0, [4, 4]);
    out.push(Case { name: "mirror_and_window", scene: s, rays: None });

    // A silver light pipe walked at an angle, from both its bores' sides.
    let mut s = Scene3::default();
    s.add_tube(
        Vec3::ZERO,
        0.0,
        TubeDims { inner_diameter_mm: 10.0, outer_diameter_mm: 12.0, length_mm: 200.0 },
        BoreFinish::Metal { metal: BoreMetal::Silver, protected: true },
    )
    .expect("a tube");
    add_rect_detector(&mut s, Vec3::new(120.0, 0.0, 0.0), -Vec3::X, 200.0, 200.0, [4, 4]);
    let mut rays = Vec::new();
    for (i, deg) in [3.0f64, 8.0, 15.0, 25.0].into_iter().enumerate() {
        for lam in [0.35, 0.55, 1.2] {
            let d = Vec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.1 * i as f64).normalize();
            rays.push(launch(Ray3::new(Vec3::new(-120.0, 0.0, 0.0), d).unwrap(), lam, 1.0));
        }
    }
    out.push(Case { name: "silver_tube", scene: s, rays: Some(rays) });

    out
}

/// What C33 compares: every ledger line, every detector's totals and Stokes
/// vector, and the sum of the deposits, each as its shortest round-trip
/// decimal, so equal strings are equal bits.
pub fn fingerprint(r: &TraceResult3) -> Vec<String> {
    let l = &r.ledger;
    let mut out = vec![
        format!("launched {:?}", l.launched),
        format!("detected_incident {:?}", l.detected_incident),
        format!("detected_signal {:?}", l.detected_signal),
        format!("escaped {:?}", l.escaped),
        format!("absorbed_volume {:?}", l.absorbed_volume),
        format!("absorbed_surface {:?}", l.absorbed_surface),
        format!("blocked_aperture {:?}", l.blocked_aperture),
        format!("truncated_flux {:?}", l.truncated_flux),
        format!("numerical_loss {:?}", l.numerical_loss),
    ];
    for d in &r.detectors {
        let s = d.stokes_local;
        out.push(format!(
            "detector {} incident {:?} signal {:?} hits {} stokes {:?} {:?} {:?} {:?}",
            d.detector.0, d.total_incident, d.total_signal, d.hits, s.s0, s.s1, s.s2, s.s3
        ));
    }
    let deposited: f64 = r.deposits.iter().map(|d| d.flux).sum();
    out.push(format!("deposits {} {:?}", r.deposits.len(), deposited));
    out
}
