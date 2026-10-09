//! What each fidelity switch costs (coating-fidelity spec §11), for
//! `bench_throughput3 --fidelity-table`.

use oc_core::fidelity::{
    CoatingModel, FidelityProfile, FilmMaterials, LayerAbsorption, Numerics, PlateModel, Preset, Temperature,
};
use oc_core::scene::Spectrum;
use oc_core::scene3::{Scene3, SpatialDistribution3};
use oc_core::Vec3;
use oc_trace::{trace_scene3, TraceConfig3};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("bench-scenes")
}

/// The two small scenes: the default document, and two dichroic plates in
/// series folding the beam at their edge (`tests/polarization3.rs`).
pub fn write_scenes() {
    let singlet = Scene3::default_scene();
    let mut two = Scene3::default_scene();
    two.sources[0].spectrum = Spectrum::Lines(vec![(0.52, 1.0), (0.56, 1.0), (0.60, 1.0)]);
    two.add_plate_dichroic(Vec3::new(-35.0, 0.0, 0.0), 15.0, 3.0, 135.0, 560.0, true).unwrap();
    two.add_plate_dichroic(Vec3::new(-35.0, 40.0, 0.0), 15.0, 3.0, -45.0, 560.0, true).unwrap();
    two.add_area_detector(Vec3::new(25.0, 40.0, 0.0), -Vec3::X, 20.0, 20.0, [16, 16]);
    for (name, s) in [("singlet.ocanvas", singlet), ("two-dichroics.ocanvas", two)] {
        std::fs::write(dir().join(name), s.to_json()).unwrap();
        println!("wrote {}", dir().join(name).display());
    }
}

/// `scene` with its sources resampled to about `rays` launch rays.
fn with_rays(mut scene: Scene3, rays: f64) -> Scene3 {
    let own: f64 = scene
        .sources
        .iter()
        .map(|s| {
            s.spectrum.ray_multiplicity() as f64
                * s.samples.spatial_samples.max(1) as f64
                * s.samples.angular_samples.max(1) as f64
        })
        .sum();
    if own <= 0.0 {
        return scene;
    }
    let k = rays / own;
    for src in &mut scene.sources {
        let s = &mut src.samples;
        let grow = |n: u32| ((n.max(1) as f64 * k).round().max(1.0)) as u32;
        if matches!(src.spatial, SpatialDistribution3::Point) {
            s.angular_samples = grow(s.angular_samples);
        } else {
            s.spatial_samples = grow(s.spatial_samples);
        }
    }
    scene
}

fn best_of_3(scene: &Scene3, p: FidelityProfile) -> f64 {
    let cfg = TraceConfig3 {
        deposit_bodies: [0, scene.bodies.len()],
        ..TraceConfig3::default()
    }
    .with_fidelity(p);
    let _ = trace_scene3(scene, &cfg);
    (0..3)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(trace_scene3(scene, &cfg));
            t0.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

/// One row: a switch value, the value it is measured against, the two
/// profiles, and the §11 ceiling.
struct Row {
    switch: &'static str,
    value: &'static str,
    against: &'static str,
    on: FidelityProfile,
    off: FidelityProfile,
    ceiling: Option<f64>,
}

fn rows() -> Vec<Row> {
    let fast = Preset::Fast.profile(Some(0), false);
    let unpolarized = FidelityProfile { coating_model: CoatingModel::Unpolarized, ..fast };
    let polarized = FidelityProfile { coating_model: CoatingModel::Polarized, ..fast };
    let per_face = FidelityProfile { layer_absorption: LayerAbsorption::PerFace, ..fast };
    let row = |switch, value, against, on, off, ceiling| Row { switch, value, against, on, off, ceiling };
    vec![
        row("coating_model", "Nominal", "Fast", fast, fast, None),
        row("coating_model", "Unpolarized", "Nominal", unpolarized, fast, None),
        row("coating_model", "Polarized", "Unpolarized", polarized, unpolarized, Some(1.5)),
        row("film_materials", "Nominal", "Fast", fast, fast, None),
        row(
            "film_materials",
            "Dispersive",
            "Nominal",
            FidelityProfile { film_materials: FilmMaterials::Dispersive, ..unpolarized },
            unpolarized,
            Some(1.15),
        ),
        row("layer_absorption", "Off", "Fast", fast, fast, None),
        row("layer_absorption", "PerFace", "Off", per_face, fast, None),
        row(
            "layer_absorption",
            "PerLayer",
            "PerFace",
            FidelityProfile { layer_absorption: LayerAbsorption::PerLayer, ..fast },
            per_face,
            Some(1.3),
        ),
        row("plate_model", "PhaseFree", "Fast", fast, fast, None),
        row(
            "plate_model",
            "Incoherent",
            "PhaseFree",
            FidelityProfile { plate_model: PlateModel::Incoherent, ..polarized },
            polarized,
            Some(1.05),
        ),
        row("temperature", "Off", "Fast", fast, fast, None),
        row(
            "temperature",
            "Declared",
            "Off",
            FidelityProfile { temperature: Temperature::Declared, ..unpolarized },
            unpolarized,
            Some(1.05),
        ),
        row("numerics", "Permissive", "Fast", fast, fast, None),
        row(
            "numerics",
            "Strict",
            "Permissive",
            FidelityProfile { numerics: Numerics::Strict, ..unpolarized },
            unpolarized,
            Some(1.02),
        ),
    ]
}

pub fn table(write_costs: bool) {
    let rays: f64 = std::env::var("BENCH_RAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000.0);
    let mut scenes = Vec::new();
    for n in ["heliostat-light-pipe", "two-dichroics", "singlet"] {
        let path = dir().join(format!("{n}.ocanvas"));
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let s = Scene3::from_json(&text).unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
                scenes.push((n, with_rays(s, rays)));
            }
            Err(_) => eprintln!("{} is missing; see bench_throughput3's docs for how to write it", path.display()),
        }
    }
    println!(
        "Relative trace time of each switch value against the value named, about {rays} launch rays per scene, best of 3, ghosts off, deposits recorded on every body.\n"
    );
    let mut head = "| Switch value | against |".to_string();
    let mut rule = "|---|---|".to_string();
    for (n, _) in &scenes {
        head.push_str(&format!(" {n} |"));
        rule.push_str("---|");
    }
    println!("{head} worst | ceiling |");
    println!("{rule}---|---|");
    let mut json_rows = Vec::new();
    for r in rows() {
        let mut worst: f64 = 0.0;
        let mut line = format!("| `{}: {}` | {} |", r.switch, r.value, r.against);
        for (_, s) in &scenes {
            let ratio = if r.on == r.off { 1.0 } else { best_of_3(s, r.on) / best_of_3(s, r.off) };
            worst = worst.max(ratio);
            line.push_str(&format!(" {ratio:.3} |"));
        }
        let ceiling = r
            .ceiling
            .map_or(String::new(), |c| format!("{c}x{}", if worst > c { " EXCEEDED" } else { "" }));
        println!("{line} {worst:.3} | {ceiling} |");
        json_rows.push(format!(
            "  {{\"switch\": \"{}\", \"value\": \"{}\", \"against\": \"{}\", \"relative\": {:.3}}}",
            r.switch, r.value, r.against, worst
        ));
    }
    if write_costs {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/public/fidelity-costs.json");
        let names: Vec<String> = scenes.iter().map(|(n, _)| format!("\"{n}\"")).collect();
        let text = format!(
            "{{\n \"comment\": \"Relative trace time of each fidelity switch value against the value named, the worst over the bench scenes. Written by bench_throughput3 --fidelity-table --write-costs; a value with no row has not been measured.\",\n \"rays\": {rays},\n \"scenes\": [{}],\n \"rows\": [\n{}\n ]\n}}\n",
            names.join(", "),
            json_rows.join(",\n")
        );
        std::fs::write(&path, text).unwrap();
        eprintln!("wrote {}", path.display());
    }
}
