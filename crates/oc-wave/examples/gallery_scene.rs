//! The scene-coupled datasets of the validation gallery (`tools/wave_gallery.py`):
//!
//! ```sh
//! cargo run -p oc-wave --release --features scene --example gallery_scene -- <outdir>
//! ```
//!
//! `f10_singlet`: a real singlet's PSF through the wave path, against
//! Optiland's `HuygensPSF` of the same lens (the fixture
//! `crates/oc-wave/tests/fixtures/huygens_singlet.json`, generated from the
//! Optiland clone by `tools/fixtures/huygens_singlet.py`). The scene is the one
//! `a_singlet_psf_matches_optiland_huygens_psf` builds.

use oc_core::build3::SingletGeom;
use oc_core::scene::Spectrum;
use oc_core::scene3::{PlanarShape2, Scene3};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, LaunchRay3, TraceConfig3};
use oc_wave::path::{wave_path, BeamProfile, Readout, WavePathSpec};
use serde_json::{json, Value};
use std::path::Path;

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn write(dir: &Path, name: &str, key: &str, data: &[f64], shape: &[usize], arrays: &mut serde_json::Map<String, Value>, desc: &str) -> R<()> {
    let file = format!("{name}__{key}.f64");
    let mut bytes = Vec::with_capacity(8 * data.len());
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(&file), bytes)?;
    arrays.insert(key.into(), json!({ "file": file, "dtype": "<f8", "shape": shape, "description": desc }));
    Ok(())
}

fn main() -> R<()> {
    let dir = std::env::args().nth(1).ok_or("usage: gallery_scene <outdir>")?;
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir)?;
    let t0 = std::time::Instant::now();
    let fx: Value = serde_json::from_str(include_str!("../tests/fixtures/huygens_singlet.json"))?;
    let num = |k: &str| fx[k].as_f64().ok_or_else(|| format!("fixture: {k}"));
    let (lam, epd, image) = (num("lambda_um")?, num("epd_mm")?, num("image_distance_from_back_vertex_mm")?);
    let grid: Vec<f64> = fx["grid_mm"].as_array().ok_or("grid")?.iter().filter_map(Value::as_f64).collect();
    let oracle: Vec<f64> = fx["psf_rows_y_then_x"].as_array().ok_or("psf")?.iter()
        .flat_map(|r| r.as_array().into_iter().flatten().filter_map(Value::as_f64).collect::<Vec<_>>())
        .collect();
    let n = grid.len();
    let spacing = grid[1] - grid[0];

    let mut s = Scene3::new();
    let src = s.add_collimated_beam(Vec3::new(97.9, 0.0, 0.0), 3.0, 3.0, 1, Spectrum::Monochromatic(lam));
    let g = SingletGeom { r1_mm: 50.0, r2_mm: -50.0, thickness_mm: 4.0, semi_aperture_mm: 12.0, conic1: 0.0, conic2: 0.0 };
    s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
    s.add_aperture(Vec3::new(97.999, 0.0, 0.0), 0.5 * epd, 0.0);
    s.apertures.last_mut().ok_or("aperture")?.shape.clear_opening = PlanarShape2::Disc { radius_mm: 0.5 * epd };
    let width = n as f64 * spacing;
    let det = s.add_area_detector(Vec3::new(102.0 + image, 0.0, 0.0), -Vec3::X, 0.5 * width, 0.5 * width, [n, n]);
    s.validate()?;
    let spec = WavePathSpec {
        source: src,
        detector: det,
        beam: BeamProfile::TopHat { radius_mm: 2.8 },
        power: 1.0,
        grid_n: 2048,
        window_mm: 6.0,
        device_samples_per_pixel: 4,
        readout: Readout::Diffracted { samples_per_pixel: 1 },
    };
    let r = wave_path(&s, &spec, 1)?;

    // The ideal peak for this pupil, P·π·sin²α/λ², α the half-angle the exit
    // pupil subtends at the paraxial focus (as the test takes it): the exit
    // pupil where a ray through the stop's centre crosses the axis, its edge
    // the marginal ray's height there, both traced.
    let mut lens = s.clone();
    lens.sources.clear();
    lens.apertures.clear();
    lens.detectors.clear();
    let last = |o: Vec3, d: Vec3| -> R<(Vec3, Vec3)> {
        let ray = Ray3::new(o, d.normalize())?;
        let tr = trace3(&lens, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &TraceConfig3 { ghosts: false, polarization: false, ..TraceConfig3::default() });
        let l = tr.segments.iter().max_by(|a, b| a.opl0_mm.total_cmp(&b.opl0_mm)).ok_or("no segment")?;
        Ok((l.a, (l.b - l.a).normalize()))
    };
    let (pc, dc) = last(Vec3::new(97.999, 0.0, 0.0), Vec3::new(1.0, 1e-3, 0.0))?;
    let x_xp = pc.x - pc.y / dc.y * dc.x;
    let (pm, dm) = last(Vec3::new(90.0, 0.5 * epd, 0.0), Vec3::X)?;
    let h_xp = (pm.y + (x_xp - pm.x) * dm.y / dm.x).abs();
    let na = h_xp / h_xp.hypot(102.0 + image - x_xp);
    let na_marginal = dm.y.abs();
    let lam_mm = lam / 1000.0;
    let i0 = r.arriving * std::f64::consts::PI * na * na / (lam_mm * lam_mm);
    let canvas: Vec<f64> = r.power.iter().map(|p| 100.0 * p / (spacing * spacing) / i0).collect();
    // The diffraction-limited Airy pattern of the same NA, for reference.
    let airy: Vec<f64> = (0..n * n)
        .map(|k| {
            let (x, y) = (grid[k % n], grid[k / n]);
            let v = 2.0 * std::f64::consts::PI * na * (x * x + y * y).sqrt() / lam_mm;
            let a = if v < 1e-9 { 1.0 } else { 2.0 * oc_wave::special::bessel_j(1, v) / v };
            100.0 * a * a
        })
        .collect();
    let c = n / 2;
    let rms = (canvas.iter().zip(&oracle).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / (n * n) as f64).sqrt() / 100.0;
    let mut arrays = serde_json::Map::new();
    let name = "f10_singlet";
    write(dir, name, "grid_mm", &grid, &[n], &mut arrays, "image coordinates (Optiland's points)")?;
    write(dir, name, "canvas", &canvas, &[n, n], &mut arrays, "canvas PSF, % of the diffraction-limited peak, rows y")?;
    write(dir, name, "oracle", &oracle, &[n, n], &mut arrays, "Optiland HuygensPSF, % of the diffraction-limited peak, rows y")?;
    write(dir, name, "airy", &airy, &[n, n], &mut arrays, "diffraction-limited Airy pattern of the same NA, %")?;
    let meta = json!({
        "dataset": name,
        "generator": "crates/oc-wave/examples/gallery_scene.rs",
        "oracle": fx["oracle"],
        "parameters": { "lambda_um": lam, "epd_mm": epd, "f_number": fx["f_number"], "image_distance_mm": image, "na": na, "na_marginal_ray": na_marginal, "exit_pupil_before_back_vertex_mm": 102.0 - x_xp, "exit_pupil_radius_mm": h_xp, "grid_n": 2048, "window_mm": 6.0 },
        "errors": {
            "strehl_canvas": canvas[c * n + c] / 100.0,
            "strehl_oracle": fx["strehl"],
            "rms_rel_ideal_peak": rms,
            "steps": r.report.steps.iter().map(|s| s.note.clone()).collect::<Vec<_>>(),
            "ledger": r.report.ledger,
        },
        "compute_seconds": t0.elapsed().as_secs_f64(),
        "arrays": Value::Object(arrays),
    });
    std::fs::write(dir.join(format!("{name}.json")), serde_json::to_string_pretty(&meta)?)?;
    eprintln!("[gallery_scene] {name} done in {:.1} s: Strehl {:.6} vs {}", t0.elapsed().as_secs_f64(), canvas[c * n + c] / 100.0, fx["strehl"]);
    Ok(())
}
