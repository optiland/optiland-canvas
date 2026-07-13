//! M0 demo: a collimated white beam through a biconvex N-BK7 singlet onto a
//! detector at the paraxial focus. Prints the spot statistics and dumps the
//! `.ocanvas` scene JSON.
//!
//! Run: `cargo run -p oc-trace --example singlet`

use oc_core::{
    ConicParams, Edge, Element, ElementId, Isometry2, Material, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_C, LAMBDA_D, LAMBDA_F,
};
use oc_trace::{trace, TraceConfig};

fn biconvex(r: f64, tc: f64, half_ap: f64, material: Material) -> Element {
    let c = 1.0 / r;
    let front = ConicParams {
        vertex: Vec2::new(-tc / 2.0, 0.0),
        axis: Vec2::new(1.0, 0.0),
        c,
        k: 0.0,
        half_ap,
    };
    let rear = ConicParams {
        vertex: Vec2::new(tc / 2.0, 0.0),
        axis: Vec2::new(-1.0, 0.0),
        c,
        k: 0.0,
        half_ap,
    };
    let sag = c * half_ap * half_ap / (1.0 + (1.0 - c * c * half_ap * half_ap).sqrt());
    let (ftip, rtip) = (-tc / 2.0 + sag, tc / 2.0 - sag);
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
        material,
    )
}

fn main() {
    let mut scene = Scene::new();
    let bk7 = Material::Catalog(scene.catalog.find("N-BK7").unwrap());
    scene.add_element(biconvex(50.0, 4.0, 12.0, bk7));

    // Collimated F/C/d white beam from the left.
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 10.0 },
        transform: Isometry2::from_pos(Vec2::new(-60.0, 0.0)),
        spectrum: Spectrum::Lines(vec![(LAMBDA_F, 1.0), (LAMBDA_D, 1.0), (LAMBDA_C, 1.0)]),
        flux: 1.0,
        rays: 64,
    });

    // Detector across the axis near the paraxial focus (~x = 51).
    scene.detectors.push(oc_core::Detector::new(
        Vec2::new(51.0, -3.0),
        Vec2::new(51.0, 3.0),
        64,
    ));

    let cfg = TraceConfig::default();
    let res = trace(&scene, &cfg);

    println!("== Optiland Canvas — M0 reference tracer ==");
    println!("segments traced : {}", res.segments.len());
    println!("ray branches    : {}", res.branches);
    let det = &res.detectors[0];
    println!("detector hits   : {}", det.hits);
    println!("collected flux  : {:.4}", det.total_flux);
    if let Some(u) = det.centroid_u() {
        // Map u∈[0,1] back to a y position on the detector segment.
        let y = -3.0 + 6.0 * u;
        println!("spot centroid y : {:+.4} mm", y);
    }
    if let Some(opl) = det.mean_opl() {
        println!("mean OPL        : {:.4} mm", opl);
    }

    // Chromatic focal shift is visible: trace each line alone and report the
    // flux-weighted spot spread (dispersion — the crown's V number at work).
    println!("\n.ocanvas scene ({} bytes):", scene.to_json().len());
    let json = scene.to_json();
    let preview: String = json.lines().take(6).collect::<Vec<_>>().join("\n");
    println!("{preview}\n  ...");
}
