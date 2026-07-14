//! Audit harness: trace the shared `systems.json` sample systems through the
//! Optiland Canvas reference tracer and emit per-ray exit state as JSON.
//!
//! This mirrors the sequential rotationally-symmetric systems that the sibling
//! Python harness traces through the main Optiland engine, so the two outputs
//! can be compared ray-by-ray. Axis convention: canvas +x <-> optiland +z,
//! transverse y <-> y.
//!
//! Run: `cargo run -p oc-trace --example audit_compare -- <path/to/systems.json>`

use oc_core::{
    ConicParams, Edge, Element, ElementId, Isometry2, Material, Scene, Source, SourceKind,
    Spectrum, Vec2, LAMBDA_D,
};
use oc_trace::{trace, TraceConfig};
use serde_json::{json, Value};
use std::f64::consts::PI;

/// Even-asphere sag (base conic + Σ coeffs[i]·y^(2i+4)); reduces to the conic
/// when `coeffs` is empty. `c = 1/R`, plane when `c == 0`.
fn sag(c: f64, k: f64, coeffs: &[f64], y: f64) -> f64 {
    let r2 = y * y;
    let mut s = if c == 0.0 {
        0.0
    } else {
        c * r2 / (1.0 + (1.0 - (1.0 + k) * c * c * r2).max(0.0).sqrt())
    };
    let mut ypow = r2 * r2; // y^4
    for &a in coeffs {
        s += a * ypow;
        ypow *= r2;
    }
    s
}

fn curvature(radius: f64) -> f64 {
    if radius.abs() >= 1e30 {
        0.0
    } else {
        1.0 / radius
    }
}

/// Build a front conic/asphere edge (axis +x) at the given vertex.
fn front_edge(vertex_x: f64, radius: f64, k: f64, coeffs: &[f64], half_ap: f64) -> Edge {
    let cp = ConicParams {
        vertex: Vec2::new(vertex_x, 0.0),
        axis: Vec2::new(1.0, 0.0),
        c: curvature(radius),
        k,
        half_ap,
    };
    if coeffs.is_empty() {
        Edge::Conic(cp)
    } else {
        let mut arr = [0.0f64; 8];
        for (i, v) in coeffs.iter().enumerate().take(8) {
            arr[i] = *v;
        }
        Edge::Asphere { base: cp, coeffs: arr }
    }
}

/// Coefficient list for a surface JSON value.
fn coeffs_of(s: &Value) -> Vec<f64> {
    s["coeffs"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect())
        .unwrap_or_default()
}

/// Build a single glass element bounded by the left surface (vertex at `zl`)
/// and the right surface (vertex at `zr`), both axis +x with signed radius.
#[allow(clippy::too_many_arguments)]
fn build_element(
    id: u32,
    zl: f64,
    left: &Value,
    zr: f64,
    right: &Value,
    half_ap: f64,
    material: Material,
) -> Element {
    let rl = left["radius"].as_f64().unwrap();
    let kl = left["conic"].as_f64().unwrap();
    let cl = coeffs_of(left);
    let rr = right["radius"].as_f64().unwrap();
    let kr = right["conic"].as_f64().unwrap();
    let cr = coeffs_of(right);

    let front = front_edge(zl, rl, kl, &cl, half_ap);
    let rear = front_edge(zr, rr, kr, &cr, half_ap);
    let ftip = zl + sag(curvature(rl), kl, &cl, half_ap);
    let rtip = zr + sag(curvature(rr), kr, &cr, half_ap);
    let rim_top = Edge::Line {
        a: Vec2::new(ftip, half_ap),
        b: Vec2::new(rtip, half_ap),
    };
    let rim_bot = Edge::Line {
        a: Vec2::new(rtip, -half_ap),
        b: Vec2::new(ftip, -half_ap),
    };
    Element::new(ElementId(id), vec![front, rim_top, rear, rim_bot], material)
}

/// Build one closed glass element per glass region (a maximal run where the
/// medium after a surface has n>1). Air regions become gaps; adjacent glass
/// regions (cemented) become separate, boundary-sharing elements. Surface `i`
/// vertex sits at the cumulative thickness of surfaces `0..i` (surface 0 at z=0).
/// `index_for(n_after)` maps a region index value to a canvas [`Material`].
fn build_elements(
    surfaces: &[Value],
    half_ap: f64,
    index_for: &dyn Fn(f64) -> Material,
) -> Vec<Element> {
    // Cumulative vertex z of each surface.
    let mut zs = vec![0.0f64; surfaces.len()];
    for i in 1..surfaces.len() {
        zs[i] = zs[i - 1] + surfaces[i - 1]["thickness"].as_f64().unwrap();
    }
    let mut elements = Vec::new();
    let mut id = 0u32;
    for i in 0..surfaces.len() - 1 {
        let n_after = surfaces[i]["n_after"].as_f64().unwrap();
        if n_after > 1.0 + 1e-12 {
            elements.push(build_element(
                id,
                zs[i],
                &surfaces[i],
                zs[i + 1],
                &surfaces[i + 1],
                half_ap,
                index_for(n_after),
            ));
            id += 1;
        }
    }
    elements
}

/// Concave conic mirror: vertex at origin, axis pointing -x so the bowl opens
/// toward incoming +x light (mirrors the analytic-suite builder).
fn build_mirror(radius: f64, conic: f64, half_ap: f64) -> Element {
    let c = curvature(radius);
    let parab = ConicParams {
        vertex: Vec2::ZERO,
        axis: Vec2::new(-1.0, 0.0),
        c,
        k: conic,
        half_ap,
    };
    let s = sag(c, conic, &[], half_ap); // sag along +axis (-x)
    let tip_x = -s;
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
    Element::new(ElementId(0), boundary, Material::Mirror { reflectivity: 1.0 })
}

/// Trace a single ray through `elements` and return the exit segment as
/// (origin, unit direction). The exit segment is the one whose start point is
/// furthest along +x (the last physical interface the ray leaves).
fn trace_exit(elements: &[Element], lambda_um: f64, origin: Vec2, dir: Vec2) -> (Vec2, Vec2) {
    let mut scene = Scene::new();
    for e in elements {
        scene.add_element(e.clone());
    }
    let angle = dir.y.atan2(dir.x);
    scene.sources.push(Source {
        kind: SourceKind::Point { half_angle: 0.0 },
        transform: Isometry2::new(origin, angle),
        spectrum: Spectrum::Monochromatic(lambda_um),
        flux: 1.0,
        rays: 1,
    });
    let cfg = TraceConfig {
        ghosts: false,
        ..TraceConfig::default()
    };
    let segs = trace(&scene, &cfg).segments;
    let s = segs
        .iter()
        .max_by(|a, b| a.a.x.partial_cmp(&b.a.x).unwrap())
        .expect("at least one segment");
    let d = (s.b - s.a).normalize();
    (s.a, d)
}

fn plane_crossing(o: Vec2, d: Vec2, x_plane: f64) -> Option<f64> {
    if d.x.abs() < 1e-15 {
        return None;
    }
    let t = (x_plane - o.x) / d.x;
    Some(o.y + t * d.y)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: audit_compare <systems.json>");
    let text = std::fs::read_to_string(&path).expect("read systems.json");
    let spec: Value = serde_json::from_str(&text).expect("parse json");
    let z_start = spec["z_start"].as_f64().unwrap();

    let mut out_systems = Vec::new();

    // Constant-index / mirror systems (traced at the d line; AbbePoint returns
    // exactly nd there, giving a constant index independent of Vd).
    for sysv in spec["systems"].as_array().unwrap() {
        let name = sysv["name"].as_str().unwrap();
        let kind = sysv["kind"].as_str().unwrap();
        let half_ap = sysv["half_ap"].as_f64().unwrap();
        let image_z = sysv["image_z"].as_f64().unwrap();

        let elements: Vec<Element> = if kind == "mirror" {
            vec![build_mirror(
                sysv["radius"].as_f64().unwrap(),
                sysv["conic"].as_f64().unwrap(),
                half_ap,
            )]
        } else {
            let surfaces = sysv["surfaces"].as_array().unwrap();
            // Constant-index: AbbePoint returns exactly nd at the d line.
            let index_for = |n: f64| Material::AbbePoint { nd: n, vd: 50.0 };
            build_elements(surfaces, half_ap, &index_for)
        };

        let mut rays_out = Vec::new();
        for rayv in sysv["rays"].as_array().unwrap() {
            let y = rayv["y"].as_f64().unwrap();
            let ang = rayv["angle_deg"].as_f64().unwrap() * PI / 180.0;
            let origin = Vec2::new(z_start, y);
            let dir = Vec2::new(ang.cos(), ang.sin());
            let (o, d) = trace_exit(&elements, LAMBDA_D, origin, dir);
            let img_y = plane_crossing(o, d, image_z);
            rays_out.push(json!({
                "y_in": y, "angle_deg": rayv["angle_deg"],
                "exit_x": o.x, "exit_y": o.y,
                "dir_x": d.x, "dir_y": d.y,
                "img_y": img_y,
            }));
        }
        out_systems.push(json!({"name": name, "rays": rays_out}));
    }

    // Dispersion systems (real catalog glass, traced at each wavelength).
    let mut out_disp = Vec::new();
    for sysv in spec["dispersion_systems"].as_array().unwrap() {
        let name = sysv["name"].as_str().unwrap();
        let glass = sysv["glass"].as_str().unwrap();
        let half_ap = sysv["half_ap"].as_f64().unwrap();
        let image_z = sysv["image_z"].as_f64().unwrap();
        let surfaces = sysv["surfaces"].as_array().unwrap();

        let scene_probe = Scene::new();
        let gid = scene_probe.catalog.find(glass).expect("glass in catalog");
        let material = Material::Catalog(gid);
        // Record the index the canvas Sellmeier model reports at each wavelength.
        let mut index_report = Vec::new();

        // Singlet: one glass element spanning surface 0 -> surface 1.
        let tc = surfaces[0]["thickness"].as_f64().unwrap();
        let elements = vec![build_element(
            0,
            0.0,
            &surfaces[0],
            tc,
            &surfaces[1],
            half_ap,
            material.clone(),
        )];

        let mut rays_out = Vec::new();
        for wl in sysv["wavelengths_um"].as_array().unwrap() {
            let lambda = wl.as_f64().unwrap();
            let n_here = scene_probe
                .catalog
                .get(gid)
                .unwrap()
                .sell
                .index(lambda);
            index_report.push(json!({"lambda_um": lambda, "n": n_here}));
            for rayv in sysv["rays"].as_array().unwrap() {
                let y = rayv["y"].as_f64().unwrap();
                let ang = rayv["angle_deg"].as_f64().unwrap() * PI / 180.0;
                let origin = Vec2::new(z_start, y);
                let dir = Vec2::new(ang.cos(), ang.sin());
                let (o, d) = trace_exit(&elements, lambda, origin, dir);
                let img_y = plane_crossing(o, d, image_z);
                rays_out.push(json!({
                    "lambda_um": lambda, "y_in": y, "angle_deg": rayv["angle_deg"],
                    "exit_x": o.x, "exit_y": o.y,
                    "dir_x": d.x, "dir_y": d.y,
                    "img_y": img_y,
                }));
            }
        }
        out_disp.push(json!({"name": name, "index": index_report, "rays": rays_out}));
    }

    let out = json!({"engine": "optiland-canvas", "systems": out_systems, "dispersion": out_disp});
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
