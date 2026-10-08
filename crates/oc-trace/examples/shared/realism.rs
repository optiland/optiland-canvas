//! What the light pipe, the spectrograph and the melting experiment are made
//! of, as a vendor would sell it: glass that absorbs, mirror coatings with
//! their layers, faces that are not quite the shape they are drawn as.
//!
//! Every part here replaced an idealisation (a constant reflectance, a
//! lossless glass, a perfect figure), and each can be put back to see what it
//! cost: `IDEAL=1` restores all of them, `REAL_<PART>=0` one of them, where
//! `<PART>` is `GLASS` (bulk absorption), `PMMA` (Beadie's index instead of
//! Sultanova's), `MIRRORS` (coating stacks instead of constants), `SLOPE`
//! (figure errors), `HELIO_GLASS` (second-surface mirrors behind real glass). `GHOSTS=1` also
//! traces every glass face's reflections ([`ghosts`]).
//! `FIDELITY=Fast|Standard|Reference` picks the coating physics the trace
//! pays for ([`fidelity`]); Standard, the default, is the kernel as it was.
//!
//! The measured data sit in examples/heliostat-lightpipe/materials, written
//! by its absorption.py from the refractiveindex.info database (CC0).

// Each example takes the parts it builds.
#![allow(dead_code)]

use oc_core::material::{Glass, Sellmeier};
use oc_core::scene3::{
    AbsorptionModel, FigureError, Interpolation, OutOfRangePolicy, Scene3, SurfaceResponse,
};
use oc_core::thinfilm::{materials, metal_mirror, Coating, CoatingBacking, FilmLayer, Incidence};

const MATERIALS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/heliostat-lightpipe/materials/");

/// Whether `part` is modelled as built (the default) or as the old ideal.
pub fn on(part: &str) -> bool {
    if std::env::var_os("IDEAL").is_some() {
        return false;
    }
    std::env::var(format!("REAL_{part}")).map_or(true, |v| v != "0")
}

/// Bulk absorption from `materials/<name>.csv` (`lambda_um,alpha_per_mm`),
/// held at its end values outside the table.
///
/// `REAL_ABS_<NAME>=0` takes one material's away, its name in capitals with
/// every other character an underscore: `REAL_ABS_N_SF66`, `REAL_ABS_PMMA`,
/// `REAL_ABS_SOLAR_GLASS`.
pub fn absorption(name: &str) -> AbsorptionModel {
    let key: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' }).collect();
    if !on("GLASS") || !on(&format!("ABS_{key}")) {
        return AbsorptionModel::None;
    }
    let path = format!("{MATERIALS}{name}.csv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (mut lambda_um, mut alpha_per_mm) = (Vec::new(), Vec::new());
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut it = line.split(',').map(|v| v.trim().parse::<f64>());
        if let (Some(Ok(l)), Some(Ok(a))) = (it.next(), it.next()) {
            lambda_um.push(l);
            alpha_per_mm.push(a.max(0.0));
        }
    }
    AbsorptionModel::Tabulated {
        lambda_um,
        alpha_per_mm,
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    }
}

/// Give body `id` the absorption of `materials/<name>.csv`.
pub fn absorb(s: &mut Scene3, id: u64, name: &str) {
    let a = absorption(name);
    if let Some(b) = s.bodies.iter_mut().find(|b| b.common.id.0 == id) {
        b.absorption = a;
    }
}

/// PMMA's index: a three-term Sellmeier fitted to Beadie et al., Appl. Opt.
/// 54, F139 (2015), measured on cast PMMA over 0.42 to 1.62 µm (fit within
/// 7e-5; Abbe number 58.7). `REAL_PMMA=0`: Sultanova et al. (2009), whose
/// one-term fit to 0.44-1.05 µm gives 53.
pub fn pmma() -> Glass {
    let sell = if on("PMMA") {
        Sellmeier { b: [1.191_955_85, 0.026_246_238_8, 0.0], c: [0.010_177_581_5, 18.195_663, 0.0] }
    } else {
        Sellmeier { b: [1.1819, 0.0, 0.0], c: [0.011_313, 0.0, 0.0] }
    };
    Glass::sellmeier("PMMA", sell)
}

/// Low-iron soda-lime glass, the float glass of a solar mirror: Rubin's index
/// (Sol. Energy Mater. 12, 275, 1985) as a Sellmeier, within 8e-4.
pub fn low_iron_glass() -> Glass {
    Glass::sellmeier(
        "Low-iron soda-lime",
        Sellmeier { b: [1.290_138_05, 50.0, 0.0], c: [0.008_680_101_51, 5_142.254_16, 0.0] },
    )
}

/// Silver under a quarter-wave pair of SiO2 and Ta2O5 at 520 nm, the usual
/// "enhanced silver": 99.1% of the light by the eye's weighting at normal
/// incidence, 98.7% at 45 degrees, 97.6% at 380 nm, and silver's own 99% in
/// the infrared. Ta2O5 rather than TiO2 because TiO2 absorbs below 400 nm.
pub fn enhanced_silver() -> Coating {
    Coating {
        layers: vec![
            FilmLayer::new(materials::tantala(), 520.0 / (4.0 * 2.10)),
            FilmLayer::new(materials::silica(), 520.0 / (4.0 * 1.46)),
            FilmLayer::new(materials::silver(), 150.0),
        ],
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: None,
    }
}

/// Aluminium under a half-wave of silica: 91% of the light, 89% of the sun's
/// power, with aluminium's dip to 77% near 830 nm. What a hot, UV-lit lamp
/// reflector is coated with.
pub fn protected_aluminium() -> Coating {
    metal_mirror(materials::aluminium_measured(), Some((materials::silica(), 550.0 / (2.0 * 1.46))))
}

/// Aluminium under the same pair as [`enhanced_silver`]: 95% of the light.
pub fn enhanced_aluminium() -> Coating {
    Coating {
        layers: vec![
            FilmLayer::new(materials::tantala(), 520.0 / (4.0 * 2.10)),
            FilmLayer::new(materials::silica(), 520.0 / (4.0 * 1.46)),
            FilmLayer::new(materials::aluminium_measured(), 150.0),
        ],
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: None,
    }
}

/// A mirror face coated with `coating`, or the constant `ideal` it replaced.
pub fn mirror(coating: Coating, ideal: f64) -> SurfaceResponse {
    if on("MIRRORS") {
        SurfaceResponse::Coating { coating }
    } else {
        SurfaceResponse::Mirror { reflectivity: oc_core::scene3::SpectralScalar::Constant(ideal) }
    }
}

/// Give object `id` a slope error of `mrad` RMS per axis.
pub fn slope(s: &mut Scene3, id: u64, mrad: f64) {
    if on("SLOPE") && mrad > 0.0 {
        s.figure_errors.retain(|f| f.object.0 != id);
        s.figure_errors.push(FigureError { object: oc_core::id::ObjectId(id), slope_rms_mrad: mrad });
    }
}

/// Make planar mirror `id` a second-surface mirror, lit on its `side` (`1.0`
/// its local `+Z`, `-1.0` its `-Z`): silver on the back of a
/// `thickness_mm` sheet of low-iron float glass, as a heliostat's facets and
/// a large flat are built. The light meets the bare glass first (4%
/// reflected, in the same direction), crosses it, is reflected by silver as
/// seen from glass, crosses back, and bounces between the two incoherently;
/// each pass is absorbed as the glass's iron absorbs (most near 1 µm). The
/// 2 to 3 mm sideways step the glass gives the beam is left out.
/// `REAL_HELIO_GLASS=0` (or `IDEAL=1`) keeps the constant the mirror had.
pub fn behind_glass(s: &mut Scene3, id: u64, side: f64, thickness_mm: f64) {
    if !on("HELIO_GLASS") {
        return;
    }
    let coating = Coating {
        layers: vec![FilmLayer::new(materials::silver(), 150.0)],
        backing: CoatingBacking::Plate { substrate: low_iron_film(), exit: oc_core::thinfilm::ExitFace::Bare, thickness_mm },
        incidence: Incidence::Substrate,
        design: None,
    };
    let m = s.surfaces.iter_mut().find(|x| x.common.id.0 == id).expect("the mirror");
    let (lit, dark) = if side > 0.0 { (&mut m.response_front, &mut m.response_back) } else { (&mut m.response_back, &mut m.response_front) };
    *lit = SurfaceResponse::Coating { coating };
    *dark = SurfaceResponse::Absorber;
}

/// [`low_iron_glass`] as a film material: its index, and the `k` that
/// materials/solar-glass.csv implies, `k = α λ / 4π`. That table is fitted so
/// a 4 mm silvered sheet reproduces DLR's measured Flabeg solar mirror at 8
/// degrees (0.952 of the sun's power; Sutter et al., Solar Energy 2018): it
/// holds the glass's iron and water bands and whatever the real silver and
/// its paint fall short of Babar and Weaver's.
fn low_iron_film() -> oc_core::thinfilm::FilmMaterial {
    let AbsorptionModel::Tabulated { lambda_um, alpha_per_mm, .. } = absorption("solar-glass") else {
        // Absorption off: the same glass, lossless.
        return oc_core::thinfilm::FilmMaterial::constant("Low-iron soda-lime", 1.5231);
    };
    let g = low_iron_glass();
    let table = |values: Vec<f64>| oc_core::scene3::SpectralScalar::Tabulated {
        lambda_um: lambda_um.clone(),
        values,
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    };
    oc_core::thinfilm::FilmMaterial {
        name: "Low-iron soda-lime".into(),
        n: table(lambda_um.iter().map(|&l| g.index(l)).collect()),
        k: table(lambda_um.iter().zip(&alpha_per_mm).map(|(&l, &a)| a * l * 1e-3 / (4.0 * std::f64::consts::PI)).collect()),
    }
}

/// Coat the bore of tube `id` with `coating`, in place of the one it was
/// built with.
pub fn bore(s: &mut Scene3, id: u64, coating: Coating) {
    if !on("MIRRORS") {
        return;
    }
    let patch = oc_core::id::PatchId(oc_core::geom3::primitives::tube_face::INNER as u64);
    let b = s.bodies.iter_mut().find(|b| b.common.id.0 == id).expect("the tube");
    for o in b.surface.overrides.iter_mut().filter(|o| o.patch == patch) {
        o.response = oc_core::scene3::TwoSidedSurfaceResponse {
            from_inside: SurfaceResponse::Coating { coating: coating.seen_from_behind() },
            from_outside: SurfaceResponse::Coating { coating: coating.clone() },
        };
    }
}

/// A reproducible unit Gaussian for draw `k` of stream `stream`: the same
/// installation errors on every run.
pub fn gauss(stream: u64, k: u64) -> f64 {
    let mix = |z: u64| {
        let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let h1 = mix(stream.wrapping_mul(0x1000_0000_01B3) ^ mix(k));
    let h2 = mix(h1);
    let u1 = ((h1 >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (h2 >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Whether the trace follows the light each glass face reflects, and through
/// how many such reflections in a row: `(ghosts, max_ghost_bounces)`.
///
/// Off, the tracer keeps only the light a dielectric face transmits. That
/// was harmless while every mirror was a front-surface constant, but a
/// second-surface mirror returns its 4% front-face reflection and the light
/// its front face bounces back inside along the main beam, and dropping them
/// costs it 8% it does not lose. [`behind_glass`] sums those inside the
/// mirror's own response instead, so this stays off unless `GHOSTS=1` asks
/// for every lens's reflections too (ten times the trace time). Two bounces
/// hold a second-surface mirror's.
pub fn ghosts() -> (bool, u32) {
    if std::env::var_os("GHOSTS").is_some() {
        (true, std::env::var("GHOST_BOUNCES").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
    } else {
        (false, 0)
    }
}

/// The coating-fidelity profile the examples trace under: the preset named by
/// `FIDELITY` (`Fast`, `Standard` or `Reference`), `Standard` when unset.
/// `Fast` turns ghosts off whatever `GHOSTS` says; the other two leave
/// [`ghosts`] in charge. An unknown name panics rather than tracing something
/// nobody asked for.
pub fn fidelity() -> oc_core::fidelity::FidelityProfile {
    use oc_core::fidelity::Preset;
    let name = std::env::var("FIDELITY").unwrap_or_else(|_| "Standard".into());
    let preset = Preset::ALL
        .into_iter()
        .find(|p| p.name().eq_ignore_ascii_case(&name))
        .unwrap_or_else(|| panic!("FIDELITY={name}: not Fast, Standard or Reference"));
    preset.profile(if preset == Preset::Fast { Some(0) } else { None }, false)
}

/// Tungsten's normal spectral emissivity at 3200 K, 0.3 to 2.5 µm: de Vos
/// (Physica 20, 690, 1954), as NASA TN D-1088 tabulates it to 2800 K,
/// carried on linearly in temperature. Where Forsythe and Worthing (1925)
/// measured at 3200 K, 0.467 and 0.665 µm, it agrees within 0.006.
const W_EMISSIVITY: [(f64, f64); 16] = [
    (0.3, 0.447), (0.4, 0.455), (0.5, 0.441), (0.6, 0.427), (0.7, 0.410), (0.8, 0.391), (0.9, 0.374), (1.0, 0.361),
    (1.1, 0.348), (1.25, 0.331), (1.4, 0.317), (1.6, 0.301), (1.8, 0.287), (2.0, 0.275), (2.2, 0.263), (2.5, 0.248),
];

/// A tungsten filament of colour temperature `t_colour` K: relative spectral
/// radiance against wavelength in µm. Tungsten's emissivity falls toward the
/// red, so the filament is cooler than its colour (3131 K for 3200, 3369 K
/// for 3450) and puts 14% less of its power into the infrared for its light
/// than a grey body would. `REAL_TUNGSTEN=0`: Planck at the colour
/// temperature.
pub fn tungsten(t_colour: f64) -> impl Fn(f64) -> f64 {
    let planck = |um: f64, t: f64| um.powi(-5) / ((14_387.77 / (um * t)).exp() - 1.0);
    let eps = |um: f64| {
        let i = W_EMISSIVITY.partition_point(|p| p.0 < um).clamp(1, W_EMISSIVITY.len() - 1);
        let ((x0, y0), (x1, y1)) = (W_EMISSIVITY[i - 1], W_EMISSIVITY[i]);
        y0 + (y1 - y0) * ((um - x0) / (x1 - x0)).clamp(0.0, 1.0)
    };
    let real = on("TUNGSTEN");
    // The true temperature whose blue-to-red ratio (450 / 650 nm) is the
    // colour temperature's.
    let t_true = if real {
        let want = planck(0.45, t_colour) / planck(0.65, t_colour);
        (0..=300)
            .map(|k| t_colour - k as f64)
            .min_by(|a, b| {
                let r = |t: f64| (planck(0.45, t) * eps(0.45) / (planck(0.65, t) * eps(0.65)) - want).abs();
                r(*a).total_cmp(&r(*b))
            })
            .unwrap_or(t_colour)
    } else {
        t_colour
    };
    move |um: f64| if real { planck(um, t_true) * eps(um) } else { planck(um, t_true) }
}

/// The xenon short-arc lamp's spectrum, relative, against wavelength in µm,
/// 0.24 to 2.36 µm in 5 nm bands: Newport's measured curve for its 1600 W
/// ozone-free lamp (materials/xenon-oriel-62711.csv), the infrared lines at
/// 0.8-1.0 µm included. `None` with `REAL_XENON=0`.
pub fn xenon_spectrum() -> Option<Vec<(f64, f64)>> {
    if !on("XENON") {
        return None;
    }
    let path = format!("{MATERIALS}xenon-oriel-62711.csv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    Some(
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let mut it = l.split(',').map(|v| v.trim().parse::<f64>());
                match (it.next(), it.next()) {
                    (Some(Ok(a)), Some(Ok(b))) => Some((a, b.max(0.0))),
                    _ => None,
                }
            })
            .collect(),
    )
}

/// The spectrograph's grating as a real one: a ruled aluminium 1800
/// lines/mm grating (MKS/Richardson 53-*-290R, 500 nm Littrow blaze) at the
/// spectrograph's 50 degrees, by polarization, from
/// materials/grating-1800-al-alpha50.csv: TE first, TE zero, and the TM pair.
///
/// The table is a rigorous coupled-wave calculation of the groove whose
/// facet angle reproduces the vendor's measured near-Littrow curves (30.5
/// degrees, not the 26.7 the ruling engine was set to), scaled by what the
/// real ruling then still loses (TE 0.968, TM 0.870). At 50 degrees the
/// grating turns the light 37 degrees, far from Littrow, and there light
/// polarized along the grooves goes mostly into the zero order: 48% into the
/// first order at 550 nm against 81% for the other polarization, 57% of the
/// light on average where the declared table said 79%. Below 490.6 nm the
/// second order opens and takes a fifth to two thirds of the TM light.
/// `None` with `REAL_GRATING=0`.
pub fn grating() -> Option<(oc_core::scene3::SpectralScalar, oc_core::scene3::SpectralScalar, oc_core::scene3::GratingTm)> {
    if !on("GRATING") {
        return None;
    }
    let path = format!("{MATERIALS}grating-1800-al-alpha50.csv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split(',').map(|v| v.trim().parse::<f64>().ok()).collect::<Option<Vec<f64>>>())
        .filter(|r| r.len() == 5)
        .collect();
    let col = |i: usize| oc_core::scene3::SpectralScalar::Tabulated {
        lambda_um: rows.iter().map(|r| r[0]).collect(),
        values: rows.iter().map(|r| r[i].clamp(0.0, 1.0)).collect(),
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    };
    Some((col(1), col(2), oc_core::scene3::GratingTm { efficiency: col(3), zero_order: col(4) }))
}
