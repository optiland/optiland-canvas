//! The faces of the clear PMMA light pipe, and the coatings that could go on
//! its outer face: reflectance against angle, weighted by the traced sun
//! spectrum, from the same transfer-matrix code the ray tracer uses.
//!
//! Light in the pipe meets the bore at a few degrees from grazing. Most of it
//! is reflected; what is transmitted enters the wall at just under the
//! critical angle (42 deg), crosses it, and meets the outer face from inside,
//! where bare PMMA lets part of it out. A mirror on the outer face sends it
//! back. This prints, as CSV on stdout:
//!
//! - `bore`: air to PMMA, bare, against the grazing angle (deg from the surface);
//! - `outer_*`: PMMA to air through the outer face, against the angle of
//!   incidence inside the wall (deg from the normal), bare and coated;
//! - `bore_silver`: the silver tube's enhanced silver, for reference.
//!
//! `cargo run --release -p oc-trace --example pmma_tube_faces > faces.csv`

use oc_core::thinfilm::{coating_rt_at, materials, metal_mirror, Coating, CoatingBacking, Incidence};

#[path = "shared/realism.rs"]
pub(crate) mod realism;

/// The wall of the deck tube.
const WALL_MM: f64 = 5.0;

const SPECTRUM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/heliostat-lightpipe/jena-direct-g173-hsrs.csv");

/// The clear wall as one effective bore surface (as `pmma_wall_coating` in
/// heliostat_dom_probes): air, the bare bore face, 5 mm of absorbing PMMA, and the
/// outer face, bare or under 150 nm of metal.
fn wall(outer: &str, pmma: &oc_core::material::Glass) -> Coating {
    use oc_core::scene3::{AbsorptionModel, Interpolation, OutOfRangePolicy, SpectralScalar};
    use oc_core::thinfilm::{ExitFace, FilmLayer, FilmMaterial};
    let AbsorptionModel::Tabulated { lambda_um, alpha_per_mm, .. } = realism::absorption("pmma") else { panic!("PMMA absorption") };
    let table = |values: Vec<f64>| SpectralScalar::Tabulated {
        lambda_um: lambda_um.clone(),
        values,
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    };
    let film = FilmMaterial::scalars(
        "PMMA",
        table(lambda_um.iter().map(|&l| pmma.index(l)).collect()),
        table(lambda_um.iter().zip(&alpha_per_mm).map(|(&l, &a)| a * l * 1e-3 / (4.0 * std::f64::consts::PI)).collect()),
    );
    let layers = match outer {
        "bare" => Vec::new(),
        "silver" => vec![FilmLayer::new(materials::silver(), 150.0)],
        _ => vec![FilmLayer::new(materials::aluminium_measured(), 150.0)],
    };
    Coating { layers, backing: CoatingBacking::Plate { substrate: film, exit: ExitFace::Bare, thickness_mm: WALL_MM }, incidence: Incidence::Substrate, design: None }
}

fn main() {
    let text = std::fs::read_to_string(SPECTRUM).expect("the site spectrum");
    let rows = oc_core::solar::parse_spectrum_table(&text).expect("a spectrum").rows;
    // Weights on a 5 nm grid, 300 to 2500 nm.
    let grid: Vec<(f64, f64)> = (0..=440)
        .map(|i| {
            let nm = 300.0 + 5.0 * i as f64;
            let j = rows.partition_point(|r| r[0] < nm).min(rows.len() - 1);
            (nm / 1e3, rows[j][1].max(0.0))
        })
        .collect();
    let wsum: f64 = grid.iter().map(|g| g.1).sum();
    let pmma = realism::pmma();
    let bare = Coating { layers: Vec::new(), backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None };
    let silver = metal_mirror(materials::silver(), None);
    let alu = metal_mirror(materials::aluminium_measured(), None);
    let enh = realism::enhanced_silver();
    // The same per-bounce sandwich written out, per polarization, so that the bore face can carry
    // a coating too (the Plate backing has only a bare or scaled exit face): R = R1 + T1 T1' R2 a² /
    // (1 − R1' R2 a²), a = exp(−α t / cos θt), for s and p apart, then the mean. `sandwich_bare`
    // must equal `wall_bare` above: the check that the two agree.
    let ar = {
        let mut c = oc_core::thinfilm::film_coating(
            oc_core::thinfilm::CoatingDesign {
                kind: oc_core::thinfilm::DesignKind::SingleLayerAr,
                lambda_nm: 550.0,
                aoi_deg: 0.0,
                periods: 1,
                n_high: 2.35,
                n_low: 1.38,
                adapt: None,
            },
            1.0,
            pmma.index(0.55),
        );
        c.design = None;
        c
    };
    let alpha = {
        let oc_core::scene3::AbsorptionModel::Tabulated { lambda_um, alpha_per_mm, .. } = realism::absorption("pmma") else { panic!("PMMA absorption") };
        move |l: f64| {
            let j = lambda_um.partition_point(|&x| x < l).clamp(1, lambda_um.len() - 1);
            let f = ((l - lambda_um[j - 1]) / (lambda_um[j] - lambda_um[j - 1])).clamp(0.0, 1.0);
            alpha_per_mm[j - 1] + f * (alpha_per_mm[j] - alpha_per_mm[j - 1])
        }
    };

    // Solar-weighted unpolarized R, A (= 1 - R - T), with the light arriving
    // through `n_in` and the far side `n_far` (both functions of wavelength).
    let avg = |c: &Coating, from_pmma: bool, cos_i: f64| -> (f64, f64) {
        let (mut r, mut a) = (0.0, 0.0);
        for &(l, w) in &grid {
            let np = pmma.index(l);
            let (n_in, n_far) = if from_pmma { (np, 1.0) } else { (1.0, np) };
            let rt = coating_rt_at(c, n_in, n_far, l, cos_i).expect("in range");
            r += w * rt.r();
            a += w * (1.0 - rt.r() - rt.t()).max(0.0);
        }
        (r / wsum, a / wsum)
    };

    println!("face,angle_deg,angle_kind,R,A");
    for i in 0..=100 {
        let g = 0.05 * i as f64; // grazing angle, deg
        let cos_i = g.to_radians().sin();
        let (r, a) = avg(&bare, false, cos_i);
        println!("bore,{g:.2},grazing,{r:.6},{a:.6}");
        let (r, a) = avg(&enh, false, cos_i);
        println!("bore_silver,{g:.2},grazing,{r:.6},{a:.6}");
        let (r, a) = avg(&ar, false, cos_i);
        println!("bore_ar,{g:.2},grazing,{r:.6},{a:.6}");
        // The bore and the 5 mm wall together, per bounce: every reflection between
        // the two faces summed incoherently, PMMA absorption on each crossing.
        for outer in ["bare", "aluminium", "silver"] {
            let c = wall(outer, &pmma);
            let (mut r, mut ab) = (0.0, 0.0);
            for &(l, w) in &grid {
                let rt = coating_rt_at(&c, 1.0, 1.0, l, cos_i).expect("in range");
                r += w * rt.r();
                ab += w * (1.0 - rt.r() - rt.t()).max(0.0);
            }
            println!("wall_{outer},{g:.2},grazing,{:.6},{:.6}", r / wsum, ab / wsum);
        }
    }
    let cases: [(&str, &Coating, &Coating); 6] = [
        ("bare", &bare, &bare),
        ("aluminium", &bare, &alu),
        ("silver", &bare, &silver),
        ("ar_bore", &ar, &bare),
        ("ar_outer", &bare, &ar),
        ("ar_both", &ar, &ar),
    ];
    for i in 0..=100 {
        let g = 0.05 * i as f64;
        let cos_i = g.to_radians().sin();
        for (name, bore_c, outer_c) in cases {
            let mut r = 0.0;
            for &(l, w) in &grid {
                let np = pmma.index(l);
                let sin_t = (1.0 - cos_i * cos_i).sqrt() / np;
                let cos_t = (1.0 - sin_t * sin_t).sqrt();
                let a2 = (-2.0 * alpha(l) * WALL_MM / cos_t).exp();
                let f1 = coating_rt_at(bore_c, 1.0, np, l, cos_i).expect("bore from air");
                let b1 = coating_rt_at(&bore_c.seen_from_behind(), np, 1.0, l, cos_t).expect("bore from the wall");
                let f2 = coating_rt_at(&outer_c.seen_from_behind(), np, 1.0, l, cos_t).expect("outer from the wall");
                let pol = |r1: f64, t1: f64, t1b: f64, r1b: f64, r2: f64| r1 + t1 * t1b * r2 * a2 / (1.0 - r1b * r2 * a2);
                let rs = pol(f1.r_s, f1.t_s, b1.t_s, b1.r_s, f2.r_s);
                let rp = pol(f1.r_p, f1.t_p, b1.t_p, b1.r_p, f2.r_p);
                r += w * 0.5 * (rs + rp);
            }
            println!("sandwich_{name},{g:.2},grazing,{:.6},0", r / wsum);
        }
    }
    for i in 0..=400 {
        let th = 30.0 + 0.05 * i as f64; // angle of incidence inside the wall, deg (30 to 50)
        let cos_i = th.to_radians().cos();
        for (name, c) in [("outer_bare", &bare), ("outer_silver", &silver), ("outer_aluminium", &alu), ("outer_ar", &ar)] {
            let c = if name == "outer_bare" { c.clone() } else { c.seen_from_behind() };
            let (r, a) = avg(&c, true, cos_i);
            println!("{name},{th:.2},incidence,{r:.6},{a:.6}");
        }
    }
}
