//! Where a mirror's few percent go: the reflectance of the light pipe's
//! mirrors and of the alternatives, weighted by the Jena direct-sun spectrum
//! (300 to 2500 nm, as the sun rig emits), at one angle of incidence.
//!
//! - the model's mirror: silver behind 4 mm low-iron glass (`realism::
//!   behind_glass`, fitted to a measured Flabeg solar mirror);
//! - the same with lossless glass, which leaves the silver's own loss (the
//!   fit puts the real silver's shortfall into the glass's absorption, so
//!   this split is the model's, not a measurement);
//! - bare silver and enhanced silver (tantala/silica pair) as front-surface
//!   mirrors, the tube's bore coating.
//!
//! `AOI_DEG` (12.5, the fold mirror's) sets the angle. `SPECTRUM_OUT=<csv>`
//! writes R(lambda) for each.
//!
//! `cargo run --release -p oc-trace --features trace3 --example mirror_loss`

use oc_core::scene3::{Scene3, SpectralScalar, SurfaceResponse};
use oc_core::thinfilm::{coating_rt, materials, metal_mirror, Coating, CoatingBacking};
use oc_core::vec3::Vec3;

#[path = "shared/realism.rs"]
pub(crate) mod realism;

const SITE_SPECTRUM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/heliostat-lightpipe/jena-direct-g173-hsrs.csv");

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The coating `realism::behind_glass` puts on a mirror's lit face.
fn second_surface(glass_mm: f64) -> Coating {
    let mut s = Scene3::new();
    let id = s.add_planar_mirror(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.95).unwrap();
    realism::behind_glass(&mut s, id, 1.0, glass_mm);
    match &s.surfaces.iter().find(|x| x.common.id.0 == id).unwrap().response_front {
        SurfaceResponse::Coating { coating } => coating.clone(),
        other => panic!("behind_glass left {other:?}"),
    }
}

fn main() {
    let aoi = env_f64("AOI_DEG", 12.5).to_radians().cos();
    let text = std::fs::read_to_string(SITE_SPECTRUM).expect("the site spectrum");
    let rows: Vec<[f64; 2]> = oc_core::solar::parse_spectrum_table(&text).unwrap().rows.into_iter().filter(|r| (300.0..=2500.0).contains(&r[0])).collect();

    let real = second_surface(4.0);
    let mut lossless = real.clone();
    if let CoatingBacking::Plate { substrate, .. } = &mut lossless.backing {
        substrate.k = SpectralScalar::Constant(0.0);
    }
    let mirrors: Vec<(&str, Coating)> = vec![
        ("silver behind 4 mm low-iron glass (the model's mirrors)", real),
        ("same, glass made lossless (the silver's own loss)", lossless),
        ("silver behind 1 mm low-iron glass", second_surface(1.0)),
        ("bare silver, front surface", metal_mirror(materials::silver(), None)),
        ("enhanced silver, front surface (the tube's bore)", realism::enhanced_silver()),
    ];

    // Bands to say where the loss sits.
    let bands = [(300.0, 400.0, "UV 300-400"), (400.0, 700.0, "visible 400-700"), (700.0, 2500.0, "IR 700-2500")];
    let mut curves = Vec::new();
    println!("angle of incidence {:.1} deg, Jena direct sun 300-2500 nm", aoi.acos().to_degrees());
    for (name, c) in &mirrors {
        let r: Vec<f64> = rows.iter().map(|row| coating_rt(c, 1.0, row[0] * 1e-3, aoi).map_or(f64::NAN, |x| x.r())).collect();
        // Trapezoid sums of the sun's power and of what the mirror loses.
        let (mut sun, mut lost) = (0.0, 0.0);
        let mut lost_band = [0.0; 3];
        for i in 1..rows.len() {
            let dl = rows[i][0] - rows[i - 1][0];
            let e = 0.5 * (rows[i][1] + rows[i - 1][1]) * dl;
            let l = 0.5 * ((1.0 - r[i]) * rows[i][1] + (1.0 - r[i - 1]) * rows[i - 1][1]) * dl;
            sun += e;
            lost += l;
            let mid = 0.5 * (rows[i][0] + rows[i - 1][0]);
            for (k, (lo, hi, _)) in bands.iter().enumerate() {
                if mid >= *lo && mid < *hi {
                    lost_band[k] += l;
                }
            }
        }
        let parts: Vec<String> = bands.iter().zip(lost_band).map(|((_, _, b), l)| format!("{b} {:.1}%", 100.0 * l / sun)).collect();
        println!("{:>6.2}% reflected, {:>5.2}% lost ({}): {name}", 100.0 * (1.0 - lost / sun), 100.0 * lost / sun, parts.join(", "));
        curves.push(r);
    }
    if let Ok(out) = std::env::var("SPECTRUM_OUT") {
        let mut csv = String::from("lambda_nm,sun_w_m2_nm,r_glass_4mm,r_lossless_glass,r_glass_1mm,r_bare_silver,r_enhanced_silver\n");
        for (i, row) in rows.iter().enumerate() {
            let r: Vec<String> = curves.iter().map(|c| format!("{:.5}", c[i])).collect();
            csv.push_str(&format!("{},{},{}\n", row[0], row[1], r.join(",")));
        }
        std::fs::write(&out, csv).expect("write the spectra");
        eprintln!("wrote {out}");
    }
}
