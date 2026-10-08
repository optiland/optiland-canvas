//! The heliostat light pipe on its own, with a measuring plane before and
//! after every element, for the DOM power budget.
//!
//! The optics are the full system's (`heliostat_lightpipe.rs`, same
//! realism switches): sun → heliostat → 45° pick-up mirror → 1 m PMMA
//! Fresnel → N-SF66 eyepiece → 360 mm wall hole → 380 mm tube entrance stop
//! → 37 m tube in 3 m sections with the 200 mm gap at 32 m. The spectrograph
//! and the melting experiment are left out: nothing takes light from the gap,
//! and the tube's end only meets the exit detector.
//!
//! Two tubes, chosen by the first argument:
//! - `silver`: the full system's, aluminium lined with enhanced silver.
//! - `pmma`: the same 380 / 400 mm tube in clear PMMA, nothing on it. The
//!   bore guides by the bare surface's own reflection, which at grazing
//!   incidence is most of the light; what enters the wall is lost sideways
//!   or runs on inside it by total internal reflection. Every PMMA face is a
//!   coating with no layers: the transfer matrix then gives the bare
//!   interface's Fresnel coefficients at each ray's angle and wavelength,
//!   and the reflection is followed as a path, not dropped as a ghost.
//!
//! Every detector records and lets the light on, except the last. The sun's
//! launch disc sits 3 m up-sun of the heliostat instead of 0.85 m, so the
//! first detector fits between it and the mirror without crossing the
//! reflected beam; the disc keeps its size, and 2 m more of the sun's
//! 4.7 mrad spread moves its edge by 10 mm, inside its 41 mm margin.
//!
//! `cargo run --release -p oc-trace --features trace3 --example heliostat_dom_probes -- silver out.ocanvas`
//!
//! `VERIFY=<spatial samples>` traces the scene (131072 is 262 thousand rays)
//! and prints the budget; `BUDGET_OUT=<file.csv>` writes it.

use oc_core::build3::{BodyFace, SingletGeom};
use oc_core::fresnel_lens::{FresnelFacing, FresnelGrooves, FresnelLens1};
use oc_core::heliostat::{HelioTarget, LocalTime, Site};
use oc_core::id::ObjectId;
use oc_core::link::LinkMode;
use oc_core::material::{Glass, Material, Sellmeier};
use oc_core::scene3::{DetectorGeometry3, DetectorTermination, PlanarShape2, Scene3, SurfaceResponse};
use oc_core::thinfilm::{Coating, CoatingBacking, CoatingDesign, DesignKind, Incidence};
use oc_core::tube::{BoreFinish, BoreMetal, TubeDims};
use oc_core::vec3::{Isometry3, Vec3};
use oc_trace::{bounds_for_scene3, depth_for_scene3, TraceConfig3};

#[path = "shared/realism.rs"]
pub(crate) mod realism;

// The full system's numbers (heliostat_lightpipe.rs, spectrograph.rs).
const Y: f64 = 5_000.0;
const SUBSTRATE: f64 = 3.0;
const TO_EYEPIECE: f64 = 1_990.781_703_327_229_3;
const EYE_T: f64 = 10.0;
const EYE_SEMI: f64 = 190.0;
const TO_HOLE: f64 = 20.0;
const HOLE_RADIUS: f64 = 180.0;
const TO_TUBE: f64 = 50.0;
const TUBE_LEN: f64 = 37_000.0;
/// The tube's bore and wall: `INNER_D` (inner diameter, 380 mm) and
/// `WALL` (10 mm), the same for every section. The wall hole, the eyepiece
/// and everything before them stay as they are.
fn bore_radius() -> f64 {
    0.5 * env_f64("INNER_D", 380.0)
}

fn outer_radius() -> f64 {
    bore_radius() + env_f64("WALL", 10.0)
}
const SPEC_AT: f64 = 32_000.0;
const TUBE_GAP: f64 = 200.0;
/// The heliostat's half-sizes: `HELIO_W` across the mirror's elevation axis
/// (horizontal) and `HELIO_H` along it in the vertical plane through the
/// normal, the plane of incidence for a vertical beam and for every layout
/// here at noon (default 1150 x 1150 mm). Bartenbach's are 1200 x 1500, 3000
/// and 3500.
fn helio_half_w() -> f64 {
    0.5 * env_f64("HELIO_W", 1_150.0)
}
fn helio_half_h() -> f64 {
    0.5 * env_f64("HELIO_H", 1_150.0)
}
/// How far up-sun of the heliostat the sun's beam starts, and where the
/// first detector stands.
/// `SUN_STANDOFF` (3000 mm). At grazing incidence part of a close disc lies
/// behind the heliostat stop's plane and its light never meets the stop: the
/// June roof-down case (76 deg) needs 6000.
fn sun_standoff() -> f64 {
    env_f64("SUN_STANDOFF", 3_000.0)
}
/// `SUN_PROBE` (2000 mm): a high fold's falling beam crosses it at 2 m.
fn sun_probe() -> f64 {
    env_f64("SUN_PROBE", 2_000.0)
}
/// Each mirror's stop is the mirror's own size, 1 mm in front of it, and its
/// face probe sits between them, 0.5 mm in front: a ray passing the stop is
/// one the mirror catches, give or take 1 mm x tan(incidence) at the edge.
const STOP_GAP: f64 = 1.0;
const PROBE_GAP: f64 = 0.5;
const SITE_SPECTRUM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/heliostat-lightpipe/jena-direct-g173-hsrs.csv"
);

/// The heliostat's height above the pick-up mirror (`DROP`, 15000 mm). No
/// drawing gives it as a dimension. Bartenbach's section (studio klv memo of
/// 2 June 2026, p. 9) scales to about 15 m against the 37 m tube, and the
/// transport budget (Blahnik, 6 Sept 2026) takes 15 m from the same drawing.
/// `DROP=5000` gives the earlier runs.
fn drop_mm() -> f64 {
    env_f64("DROP", 15_000.0)
}

/// Where the heliostat stands and which way it sends the light (`LAYOUT`).
/// The pick-up mirror is fixed on the tube's axis; the room's up is world
/// `+Y` for `below-up`, `-Y` for the roof layouts (the sun rig's
/// `upside_down`), and north is world `+X`, the way the tube runs.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Layout {
    /// `DROP` below the pick-up mirror, beam straight up.
    BelowUp,
    /// On the roof `DROP` above the pick-up mirror, beam straight down.
    RoofDown,
    /// On the roof `DROP` above the tube's axis and north of the pick-up
    /// mirror, beam down and back south at `SLANT_DEG` (60) below the
    /// horizontal. 45 would send it through the Fresnel's lower edge.
    RoofSlant,
    /// On the roof `DROP` above the axis, beam up and north at `FOLD_DEG` (65)
    /// to a fixed fold mirror `FOLD_DIST` (5000) mm away, which sends it
    /// straight down onto the pick-up mirror.
    RoofFold,
}

fn layout() -> Layout {
    match std::env::var("LAYOUT").as_deref() {
        Ok("below-up") => Layout::BelowUp,
        Ok("roof-down") => Layout::RoofDown,
        Ok("roof-slant") => Layout::RoofSlant,
        Ok("roof-fold") => Layout::RoofFold,
        Ok(other) => panic!("LAYOUT: below-up, roof-down, roof-slant or roof-fold, not {other}"),
        // `UPSIDE_DOWN=0` is the earlier spelling of below-up.
        Err(_) if env_f64("UPSIDE_DOWN", 1.0) < 0.5 => Layout::BelowUp,
        Err(_) => Layout::RoofDown,
    }
}

/// Half-size of the heliostat stop's mask: the 854 mm sun beam's footprint
/// on the mirror's plane, 854 / cos(incidence), and some more. The fold
/// layout's stays clear of the beam coming down 2.1 m away.
fn helio_mask(lay: Layout) -> f64 {
    match lay {
        Layout::BelowUp => 1_100.0,
        Layout::RoofDown => 4_000.0, // 854 / cos 76 deg: the June noon sun
        Layout::RoofSlant => 1_500.0,
        Layout::RoofFold => 1_250.0,
    }
}

/// The fold mirror's radius (`FOLD_R`, 850 mm): the heliostat's beam, 1150 x 910 mm at 38 deg,
/// plus 5 m of the sun's spread and the slope errors.
fn fold_r() -> f64 {
    env_f64("FOLD_R", 850.0)
}

/// A flat round mirror at `at` turning light travelling `d_in` into `d_out`
/// (both in the XY plane), its lit face local `-Z`, silvered behind 4 mm
/// glass with a 1 mrad slope error, as the pick-up mirror is.
fn turning_mirror(s: &mut Scene3, at: Vec3, d_in: Vec3, d_out: Vec3, radius: f64, name: &str) -> u64 {
    let n = (d_out - d_in).normalize();
    let along = Vec3::new(-n.y, n.x, 0.0) * radius;
    let id = s.add_planar_mirror(at - along, at + along, 0.95).expect("a turning mirror");
    label(s, id, name);
    realism::behind_glass(s, id, -1.0, 4.0);
    realism::slope(s, id, env_f64("PICKUP_SLOPE", 1.0));
    id
}

#[derive(Clone, Copy, PartialEq)]
enum TubeKind {
    Silver,
    Pmma,
}

fn site_rows() -> Vec<[f64; 2]> {
    let text = std::fs::read_to_string(SITE_SPECTRUM).expect("the site spectrum (run site_spectrum.py)");
    oc_core::solar::parse_spectrum_table(&text).expect("a spectrum table").rows
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `FRESNEL_R` and `FRESNEL_A="a4 a6 .. a18"` give another Fresnel lens
/// (fresnel_variants.py in D:\tmp\dom-probes); the built F/3 lens by default.
fn fresnel() -> FresnelLens1 {
    let mut aspheric = [
        -2.872_550_532_174_384_5e-11,
        3.766_883_695_772_587_6e-18,
        -5.667_889_508_372_192e-25,
        9.138_076_524_559_708e-32,
        -1.511_533_763_305_906_2e-38,
        1.721_125_494_621_016e-45,
        1.326_265_720_908_974_4e-51,
        -1.365_679_261_586_851_6e-57,
    ];
    if let Ok(v) = std::env::var("FRESNEL_A") {
        let a: Vec<f64> = v.split_whitespace().map(|x| x.parse().expect("FRESNEL_A: eight numbers")).collect();
        aspheric.copy_from_slice(&a);
    }
    FresnelLens1 {
        radius_mm: env_f64("FRESNEL_R", 1_470.147_217_069_780_3),
        conic: -1.0,
        aspheric,
        clear_radius_mm: 500.0,
        grooves: FresnelGrooves::ConstantPitch { pitch_mm: 1.0 },
        substrate_mm: SUBSTRATE,
        facing: FresnelFacing::PlusX,
        draft_deg: env_f64("DRAFT", if realism::on("DRAFT") { 3.0 } else { 0.0 }),
    }
}

/// The clear PMMA wall, `wall_mm` thick, as one effective bore surface: the
/// light meets the bare bore face, crosses the wall (absorbed as PMMA absorbs,
/// along the slant path), meets the outer face, bare or under a 150 nm metal
/// film, and the bounces between the two faces add incoherently. What the
/// outer face transmits leaves the pipe. The same construction as a solar
/// mirror's silver behind glass ([`realism::behind_glass`]).
///
/// Left out against the full trace (`pmma` with `TUBE_OUTER`): the few-mm
/// step along the axis of the light that went through the wall, the light
/// that runs along inside the wall and leaves through the end rings, and any
/// light the wall carries across a joint. It needs no branch threshold: each
/// bore hit is one reflection.
fn pmma_wall_coating(outer: &str, wall_mm: f64) -> Coating {
    use oc_core::thinfilm::{materials, ExitFace, FilmLayer, FilmMaterial};
    use oc_core::scene3::{AbsorptionModel, Interpolation, OutOfRangePolicy, SpectralScalar};
    let pmma = realism::pmma();
    let AbsorptionModel::Tabulated { lambda_um, alpha_per_mm, .. } = realism::absorption("pmma") else {
        panic!("the PMMA absorption table");
    };
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
        "aluminium" => vec![FilmLayer::new(materials::aluminium_measured(), 150.0)],
        other => panic!("TUBE_EFFECTIVE: bare, silver or aluminium, not {other}"),
    };
    Coating {
        layers,
        backing: oc_core::thinfilm::CoatingBacking::Plate { substrate: film, exit: ExitFace::Bare, thickness_mm: wall_mm },
        incidence: oc_core::thinfilm::Incidence::Substrate,
        design: None,
    }
}

/// A bare dielectric face: no layers, so the transfer matrix is the plain
/// Fresnel interface, and no design, so its reflection is a path.
fn bare() -> oc_core::scene3::TwoSidedSurfaceResponse {
    let c = Coating { layers: Vec::new(), backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None };
    oc_core::scene3::TwoSidedSurfaceResponse {
        from_inside: oc_core::scene3::SurfaceResponse::Coating { coating: c.seen_from_behind() },
        from_outside: oc_core::scene3::SurfaceResponse::Coating { coating: c },
    }
}

/// One measuring plane: where, which way it faces (toward the oncoming
/// light), its shape, and its pixel grid.
struct Probe {
    name: &'static str,
    id: ObjectId,
    size: String,
}

fn probe(
    s: &mut Scene3,
    at: Vec3,
    facing: Vec3,
    shape: DetectorGeometry3,
    bins: usize,
    last: bool,
    name: &'static str,
) -> Probe {
    let id = s.add_area_detector(at, facing, 1.0, 1.0, [bins, bins]);
    let size = match &shape {
        DetectorGeometry3::Disc { radius_mm } => format!("disc {:.0} mm dia.", 2.0 * radius_mm),
        DetectorGeometry3::Rectangle { half_width_mm, half_height_mm } => {
            format!("square {:.0} x {:.0} mm", 2.0 * half_width_mm, 2.0 * half_height_mm)
        }
        _ => String::new(),
    };
    let d = s.detectors.iter_mut().find(|d| d.common.id.0 == id.0).unwrap();
    d.geometry = shape;
    d.termination = if last { DetectorTermination::Terminate } else { DetectorTermination::ProbeAndContinue };
    d.common.label = format!("P: {name}");
    Probe { name, id: ObjectId(id.0), size }
}

/// A probe 2 mm in front of mirror `parent`'s lit face (`side` +1 its local
/// `+Z`, -1 its `-Z`), its size, locked to it. The light reaching the mirror
/// crosses it forward, the reflected light backward.
fn face_probe(s: &mut Scene3, parent: u64, side: f64, name: &'static str) -> Probe {
    let m = s.surfaces.iter().find(|x| x.common.id.0 == parent).expect("the mirror");
    let shape = match &m.geometry {
        oc_core::scene3::SurfaceGeometry3::Plane { shape: PlanarShape2::Rectangle { half_width_mm, half_height_mm } } => {
            DetectorGeometry3::Rectangle { half_width_mm: *half_width_mm, half_height_mm: *half_height_mm }
        }
        oc_core::scene3::SurfaceGeometry3::Plane { shape: PlanarShape2::Disc { radius_mm } } => disc(*radius_mm),
        other => panic!("{name}: a flat mirror, not {other:?}"),
    };
    // Turned half a turn about local X when the lit face is -Z, so the
    // probe's +Z faces the oncoming light either way.
    let flip = if side > 0.0 { [0.0, 0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0, 0.0] };
    let at = m.common.transform.compose(Isometry3::new(Vec3::new(0.0, 0.0, PROBE_GAP * side), flip).unwrap());
    let mut p = probe(s, Vec3::ZERO, Vec3::Z, shape, 115, false, name);
    p.size.push_str(" (in: fwd, out: back)");
    s.detectors.iter_mut().find(|d| d.common.id == p.id).unwrap().common.transform = at;
    if let Err(e) = s.link_object(p.id.0, parent, LinkMode::Follow) {
        eprintln!("{name}: not locked to its mirror ({e})");
    }
    p
}

/// The eyepiece's AR coating on every face: a quarter wave of MgF2 at 650 nm,
/// or with `EYE_AR=bbar` the three-layer broadband AR (MgF2, TiO2, Al2O3) at
/// `EYE_AR_NM` (650).
fn eye_ar() -> CoatingDesign {
    let kind = match std::env::var("EYE_AR").as_deref() {
        Ok("bbar") => DesignKind::BroadbandAr,
        Ok(other) if other != "single" => panic!("EYE_AR: single or bbar, not {other}"),
        _ => DesignKind::SingleLayerAr,
    };
    CoatingDesign { kind, lambda_nm: env_f64("EYE_AR_NM", 650.0), aoi_deg: 0.0, periods: 1, n_high: 2.1, n_low: 1.38 }
}

/// An air-spaced doublet eyepiece (`EYE_DESIGN`): two glasses (Sellmeier),
/// four radii and conics front to back, the two thicknesses and the gap at the
/// axis, all mm.
struct EyeDoublet {
    glasses: Vec<Glass>,
    r: Vec<f64>,
    k: Vec<f64>,
    t: [f64; 2],
    gap: f64,
}

impl EyeDoublet {
    fn read(path: &str) -> Self {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).expect("EYE_DESIGN")).expect("EYE_DESIGN json");
        let nums = |x: &serde_json::Value| x.as_array().expect("array").iter().map(|n| n.as_f64().expect("number")).collect::<Vec<f64>>();
        let glasses = v["glasses"]
            .as_array()
            .expect("glasses")
            .iter()
            .map(|g| {
                let (b, c) = (nums(&g["b"]), nums(&g["c"]));
                Glass::sellmeier(g["name"].as_str().expect("name"), Sellmeier { b: [b[0], b[1], b[2]], c: [c[0], c[1], c[2]] })
            })
            .collect();
        let t = nums(&v["t"]);
        Self { glasses, r: nums(&v["R"]), k: nums(&v["k"]), t: [t[0], t[1]], gap: v["gap"].as_f64().expect("gap") }
    }
}

fn disc(r: f64) -> DetectorGeometry3 {
    DetectorGeometry3::Disc { radius_mm: r }
}

fn square(h: f64) -> DetectorGeometry3 {
    DetectorGeometry3::Rectangle { half_width_mm: h, half_height_mm: h }
}

fn build(kind: TubeKind) -> (Scene3, Vec<Probe>) {
    let mut s = Scene3::new();
    let pmma = realism::pmma();
    let sf66 = Glass::sellmeier(
        "N-SF66",
        Sellmeier { b: [2.024_597_6, 0.470_187_196, 2.599_704_33], c: [0.014_705_322_5, 0.069_299_827_6, 161.817_601] },
    );
    // `EYE_GLASS=nbk7|f2` (with `EYE_R`, `EYE_K` from fresnel_eyepiece.py) swaps the eyepiece's
    // glass for the chromatic comparison; N-SF66 (the built one) by default.
    let eye_glass = match std::env::var("EYE_GLASS").as_deref() {
        Ok("nbk7") => Glass::sellmeier(
            "N-BK7",
            Sellmeier { b: [1.039_612_12, 0.231_792_344, 1.010_469_45], c: [0.006_000_698_67, 0.020_017_914_4, 103.560_653] },
        ),
        Ok("f2") => Glass::sellmeier(
            "F2",
            Sellmeier { b: [1.345_333_59, 0.209_073_176, 0.937_357_162], c: [0.009_977_438_71, 0.047_045_076_7, 111.886_764] },
        ),
        Ok(other) if other != "nsf66" => panic!("EYE_GLASS: nsf66, nbk7 or f2, not {other}"),
        _ => sf66.clone(),
    };
    let pmma_id = s.catalog.put(&pmma);
    let sf66_id = s.catalog.put(&eye_glass);

    // Jena, Deutsches Optisches Museum; March equinox, solar noon unless
    // `SUN_DATE="month day hour minute"` (CET) says otherwise.
    let site = Site {
        latitude_deg: 50.92781,
        longitude_deg: 11.57928,
        elevation_m: 160.0,
        pressure_mbar: 1013.25 * (-160.0_f64 / 8434.0).exp(),
        temperature_c: 10.0,
    };
    let date: Vec<u32> =
        std::env::var("SUN_DATE").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let (month, day, hour, minute) = if let [mo, d, h, mi] = date[..] { (mo, d, h, mi) } else { (3, 20, 12, 21) };
    let time = LocalTime { year: 2026, month, day, hour, minute, second: 0.0, utc_offset_h: 1.0 };
    s.add_sun(site, time, 0.0);
    // A roof layout is the room turned over about +X: the heliostat stands
    // at world Y 0 or so, above the tube in the room, and the sun comes from
    // world -Y.
    let lay = layout();
    let upside_down = lay != Layout::BelowUp;
    s.solar.as_mut().unwrap().upside_down = upside_down;
    let up = if upside_down { -Vec3::Y } else { Vec3::Y };
    let pick_at = Vec3::new(0.0, Y, 0.0);
    // A point `north` mm north of the pick-up mirror and `h` mm above it.
    let room = |north: f64, h: f64| pick_at + Vec3::X * north + up * h;
    let slant = env_f64("SLANT_DEG", 60.0).to_radians();
    let (fold_deg, fold_d) = (env_f64("FOLD_DEG", 65.0).to_radians(), env_f64("FOLD_DIST", 5_000.0));
    let h = drop_mm();
    let (helio_at, fold_at) = match lay {
        Layout::BelowUp => (room(0.0, -h), None),
        Layout::RoofDown => (room(0.0, h), None),
        Layout::RoofSlant => (room(h / slant.tan(), h), None),
        // `PICK_OFF` (0): the pick-up stands this far north of the point under
        // the fold, and the falling leg leans north to it.
        Layout::RoofFold => {
            let off = env_f64("PICK_OFF", 0.0);
            (room(-fold_d * fold_deg.cos() - off, h), Some(room(-off, h + fold_d * fold_deg.sin())))
        }
    };
    let rows = site_rows();
    let total = oc_core::solar::table_power(&rows, rows[0][0] / 1e3, rows[rows.len() - 1][0] / 1e3);
    if let Some(rig) = s.solar.as_mut() {
        rig.spectrum = Some(oc_core::heliostat::SunSpectrum { name: "jena-direct-g173-hsrs.csv".into(), rows });
        rig.dni_w_m2 = total;
        // `CSR` (0.05): the share of the direct sunlight in the aureole (the sensitivity study).
        rig.csr = env_f64("CSR", rig.csr);
        // The DIN sun's turbidity is fitted to CAMS McClear, whose direct light is the disc
        // alone: the disc keeps all of it and the aureole comes on top (`DNI_DISC_ONLY=0`: the
        // disc and the aureole share it, as before 8 Oct).
        rig.dni_disc_only = std::env::var("DNI_DISC_ONLY").as_deref() != Ok("0");
        // `SUN_DISC=uniform|limb|buie` (limb): the disc's brightness profile.
        if let Some(d) = std::env::var("SUN_DISC").ok().and_then(|v| oc_core::solar::sunshape::SunDisc::parse(&v)) {
            rig.disc = d;
        }
        // `BAND_UM="lo hi"`: trace only this band (the chromatic study traces one colour at a time).
        if let Ok(v) = std::env::var("BAND_UM") {
            let b: Vec<f64> = v.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            rig.band_um = [b[0], b[1]];
        }
    }

    // Pick-up mirror: fixed, silvered 4 mm low-iron glass, turning what
    // comes down (or up) into the tube along +X. The fold mirror, when there
    // is one, is the same glass.
    let feed = fold_at.unwrap_or(helio_at);
    let pick_r = env_f64("PICK_R", 640.0 * std::f64::consts::SQRT_2);
    let pickup = turning_mirror(&mut s, pick_at, (pick_at - feed).normalize(), Vec3::X, pick_r, "Pick-up mirror, 1810 mm disc");
    let fold = fold_at
        .map(|f| turning_mirror(&mut s, f, (f - helio_at).normalize(), (pick_at - f).normalize(), fold_r(), "Fold mirror, fixed"));

    // Heliostat, 1150 mm square unless HELIO_W / HELIO_H say otherwise.
    // `TRACK_X_MRAD`, `TRACK_Z_MRAD` (0): a tracking error, the heliostat's normal off by this
    // much in the two directions across its beam. The mirror then aims at a point 2 e D beside its
    // target (D the distance to it): the reflected beam turns by twice the normal's error.
    let (track_x, track_z) = (env_f64("TRACK_X_MRAD", 0.0) * 1e-3, env_f64("TRACK_Z_MRAD", 0.0) * 1e-3);
    let helio_target = if track_x == 0.0 && track_z == 0.0 {
        HelioTarget::Object(ObjectId(fold.unwrap_or(pickup)))
    } else {
        let aim = fold_at.unwrap_or(pick_at);
        let (b, d) = ((aim - helio_at).normalize(), (aim - helio_at).length());
        let seed = if b.dot(Vec3::X).abs() < 0.9 { Vec3::X } else { Vec3::Z };
        let e1 = (seed - b * b.dot(seed)).normalize();
        let e2 = b.cross(e1);
        HelioTarget::Point(aim + (e1 * track_x + e2 * track_z) * (2.0 * d))
    };
    let heliostat = s
        .add_heliostat(helio_at, 2.0 * helio_half_w(), 2.0 * helio_half_h(), 0.94, helio_target)
        .expect("heliostat");
    realism::behind_glass(&mut s, heliostat.0, 1.0, 4.0);
    realism::slope(&mut s, heliostat.0, env_f64("HELIO_SLOPE", 1.2));
    // `HELIO_DARK=1`, the stray-light test: a black heliostat, so whatever
    // still reaches the Fresnel went round its stop.
    if env_f64("HELIO_DARK", 0.0) > 0.5 {
        let m = s.surfaces.iter_mut().find(|x| x.common.id.0 == heliostat.0).unwrap();
        m.response_front = SurfaceResponse::Absorber;
        m.response_back = SurfaceResponse::Absorber;
    }
    let mask = env_f64("HELIO_MASK", helio_mask(lay));
    stop_on(
        &mut s,
        heliostat.0,
        STOP_GAP,
        PlanarShape2::Rectangle { half_width_mm: helio_half_w(), half_height_mm: helio_half_h() },
        PlanarShape2::Rectangle { half_width_mm: mask, half_height_mm: mask },
        "Heliostat stop",
    );
    stop_on(
        &mut s,
        pickup,
        -STOP_GAP,
        PlanarShape2::Disc { radius_mm: pick_r },
        PlanarShape2::Disc { radius_mm: 1_400.0 },
        "Pick-up mirror stop",
    );
    if let Some(f) = fold {
        stop_on(&mut s, f, -STOP_GAP, PlanarShape2::Disc { radius_mm: fold_r() }, PlanarShape2::Disc { radius_mm: fold_r() + 450.0 }, "Fold mirror stop");
    }

    // Fresnel lens, 1 m after the pick-up mirror, MgF2 on both faces.
    let x_fresnel = 1_000.0;
    let f = s.add_fresnel_lens(Vec3::new(x_fresnel, Y, 0.0), 0.0, &fresnel(), Material::Catalog(pmma_id)).expect("Fresnel");
    let to_eye = env_f64("TO_EYE", TO_EYEPIECE);
    let no_eye = env_f64("NO_EYE", 0.0) > 0.5;
    // `EYE_DESIGN=<json>`: an air-spaced doublet eyepiece instead of the singlet
    // (eyepiece_doublet.py in the study folder writes it).
    let doublet = std::env::var("EYE_DESIGN").ok().map(|p| EyeDoublet::read(&p));
    let eye_len = doublet.as_ref().map_or(EYE_T, |d| d.t[0] + d.gap + d.t[1]);
    label(&mut s, f, &format!("Fresnel, PMMA, {:.0} mm before the wall hole, 1000 mm clear, single-layer AR both faces", to_eye + eye_len + TO_HOLE));
    realism::absorb(&mut s, f, "pmma");
    // `FRESNEL_SLOPE` (0 mrad): a figure error on the Fresnel lens's faces, for the tolerance study.
    realism::slope(&mut s, f, env_f64("FRESNEL_SLOPE", 0.0));
    let film = oc_core::thinfilm::film_coating(
        CoatingDesign { kind: DesignKind::SingleLayerAr, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38 },
        1.0,
        pmma.index(0.55),
    );
    if let Some(body) = s.bodies.iter_mut().find(|q| q.common.id.0 == f) {
        body.surface.default = oc_core::scene3::TwoSidedSurfaceResponse {
            from_inside: oc_core::scene3::SurfaceResponse::Coating { coating: film.seen_from_behind() },
            from_outside: oc_core::scene3::SurfaceResponse::Coating { coating: film },
        };
    }
    let valleys = x_fresnel + 0.5 * SUBSTRATE;
    // An absorber just before the Fresnel lens, open over its 1 m clear
    // aperture: the beam beside the lens ends here, not on the wall 2 m on
    // (`FRESNEL_MASK=0` leaves it out). It reaches 1.4 m from the axis, short
    // of the slanted layout's beam.
    if env_f64("FRESNEL_MASK", 1.0) > 0.5 {
        let m = s.add_aperture(Vec3::new(x_fresnel - 5.0, Y, 0.0), fresnel().clear_radius_mm, 0.0);
        if let Some(a) = s.apertures.iter_mut().find(|a| a.common.id.0 == m) {
            a.shape.outer_extent = Some(PlanarShape2::Disc { radius_mm: 1_400.0 });
        }
        label(&mut s, m, "Fresnel mask, 1000 mm opening, absorber to 2800 mm");
    }

    // Eyepiece: plano-concave N-SF66, hyperbolic face toward the Fresnel,
    // MgF2 at 650 nm on both faces.
    // `TO_EYE` moves the eyepiece (and the wall behind it) toward the Fresnel
    // lens; `NO_EYE=1` leaves the eyepiece out, the wall where it was.
    let vertex = valleys + to_eye;
    // `EYE_SEMI` (190 mm): a faster Fresnel lens needs a deeper eyepiece, and
    // its rim then meets the converging beam earlier, where it is wider.
    let eye_semi = env_f64("EYE_SEMI", EYE_SEMI);
    let rim_sag = {
        let (c, k) = (1.0 / env_f64("EYE_R", -928.717_076_232_186_2), env_f64("EYE_K", -3.687_298_942_825_485));
        if no_eye { 0.0 } else { c * eye_semi * eye_semi / (1.0 + (1.0 - (1.0 + k) * c * c * eye_semi * eye_semi).sqrt()) }
    };
    if let (Some(d), false) = (&doublet, no_eye) {
        let mut at = vertex;
        for (i, g) in d.glasses.iter().enumerate() {
            let id = s.catalog.put(g);
            let geom = SingletGeom {
                r1_mm: d.r[2 * i],
                r2_mm: d.r[2 * i + 1],
                thickness_mm: d.t[i],
                semi_aperture_mm: eye_semi,
                conic1: d.k[2 * i],
                conic2: d.k[2 * i + 1],
            };
            let eye = s.add_singlet(Vec3::new(at + 0.5 * d.t[i], Y, 0.0), geom, g.nd, g.vd);
            if let Some(b) = s.bodies.iter_mut().find(|b| b.common.id.0 == eye.0) {
                b.material = Material::Catalog(id);
                b.common.label = format!("Eyepiece {}, {}, 380 mm dia.", ["front", "back"][i], g.name);
            }
            realism::absorb(&mut s, eye.0, &g.name);
            for face in [BodyFace::Front, BodyFace::Back] {
                let ar = eye_ar();
                assert!(s.set_body_face_coating(eye.0, face, Some(ar)), "AR coating");
            }
            at += d.t[i] + d.gap;
        }
    } else if !no_eye {
        let eye = s.add_singlet(
            Vec3::new(vertex + 0.5 * EYE_T, Y, 0.0),
            SingletGeom {
                r1_mm: env_f64("EYE_R", -928.717_076_232_186_2),
                r2_mm: f64::INFINITY,
                thickness_mm: EYE_T,
                semi_aperture_mm: eye_semi,
                conic1: env_f64("EYE_K", -3.687_298_942_825_485),
                conic2: 0.0,
            },
            eye_glass.nd,
            eye_glass.vd,
        );
        if let Some(b) = s.bodies.iter_mut().find(|b| b.common.id.0 == eye.0) {
            b.material = Material::Catalog(sf66_id);
            b.common.label = format!("Eyepiece, {} hyperbolic, 380 mm dia.", eye_glass.name);
        }
        // Each eyepiece glass absorbs as SCHOTT's internal transmittance says (F2: the data of N-F2,
        // its lead-free equivalent; materials/*.csv).
        let abs_name = match eye_glass.name.as_str() {
            "N-BK7" => "N-BK7",
            "F2" => "N-F2",
            _ => "N-SF66",
        };
        realism::absorb(&mut s, eye.0, abs_name);
        // `EYE_SLOPE` (0 mrad): a figure error on the eyepiece's faces, for the tolerance study.
        realism::slope(&mut s, eye.0, env_f64("EYE_SLOPE", 0.0));
        for face in [BodyFace::Front, BodyFace::Back] {
            assert!(s.set_body_face_coating(eye.0, face, Some(eye_ar())), "AR coating");
        }
    }

    // Wall hole and the tube.
    let hole_x = vertex + eye_len + TO_HOLE;
    let hole = s.add_aperture(Vec3::new(hole_x, Y, 0.0), HOLE_RADIUS, 0.0);
    // The wall itself, 4 x 4 m: light that passed beside the Fresnel ends
    // here (it stays within 1 m of the axis), and the slanted layout's beam
    // passes above it.
    if let Some(a) = s.apertures.iter_mut().find(|a| a.common.id.0 == hole) {
        a.shape.outer_extent = Some(PlanarShape2::Rectangle { half_width_mm: 2_000.0, half_height_mm: 2_000.0 });
    }
    label(&mut s, hole, "Wall hole, 360 mm, wall 4 x 4 m");
    let entry = hole_x + TO_TUBE;
    let gap_at = entry + SPEC_AT;
    let half_gap = 0.5 * TUBE_GAP;
    let section = if realism::on("TUBE") { env_f64("TUBE_SECTION", 3_000.0) } else { f64::INFINITY };
    let (off_rms, tilt_rms) = (env_f64("TUBE_OFFSET", 0.5), env_f64("TUBE_TILT", 0.1e-3));
    let mut k = 0u64;
    for (from, to, name) in [(entry, gap_at - half_gap, "0-32 m"), (gap_at + half_gap, entry + TUBE_LEN, "32-37 m")] {
        let n = ((to - from) / section).ceil().max(1.0) as usize;
        let joint = if n > 1 { 2.0 } else { 0.0 };
        let len = (to - from - joint * (n - 1) as f64) / n as f64;
        for i in 0..n {
            let a = from + i as f64 * (len + joint);
            let finish = match kind {
                TubeKind::Silver => BoreFinish::Metal { metal: BoreMetal::Silver, protected: true },
                TubeKind::Pmma => BoreFinish::Black,
            };
            let tube = s
                .add_tube(
                    Vec3::new(a + 0.5 * len, Y, 0.0),
                    0.0,
                    TubeDims { inner_diameter_mm: 2.0 * bore_radius(), outer_diameter_mm: 2.0 * outer_radius(), length_mm: len },
                    finish,
                )
                .expect("tube");
            // The same 3 m sections, joints and laying errors (same seeds)
            // for both tubes.
            if section.is_finite() {
                let g = |j: u64| realism::gauss(17, 4 * k + j);
                let at = Vec3::new(a + 0.5 * len, Y + off_rms * g(0), off_rms * g(1));
                let turn = glam::DQuat::from_rotation_z(tilt_rms * g(2)) * glam::DQuat::from_rotation_y(tilt_rms * g(3));
                let b = s.bodies.iter_mut().find(|b| b.common.id.0 == tube).unwrap();
                b.common.transform = Isometry3::new(at, [turn.x, turn.y, turn.z, turn.w]).expect("a section's pose");
                k += 1;
            }
            match kind {
                TubeKind::Silver => {
                    // `TUBE_EFFECTIVE=bare|silver|aluminium`: the clear PMMA wall as one
                    // effective bore surface, for the coating study (see pmma_wall_coating).
                    match std::env::var("TUBE_EFFECTIVE").ok() {
                        Some(outer) => realism::bore(&mut s, tube, pmma_wall_coating(&outer, env_f64("WALL", 10.0))),
                        None => realism::bore(&mut s, tube, realism::enhanced_silver()),
                    }
                    label(&mut s, tube, &format!("Light pipe {name}, section {} of {n}, enhanced silver", i + 1));
                }
                TubeKind::Pmma => {
                    let b = s.bodies.iter_mut().find(|b| b.common.id.0 == tube).unwrap();
                    b.material = Material::Catalog(pmma_id);
                    b.surface.default = bare();
                    b.surface.overrides.clear();
                    // `TUBE_OUTER=silver|aluminium`: a 150 nm metal film on the
                    // outer face (a second-surface mirror seen from the wall); bare
                    // by default. The light the bore lets into the wall meets this
                    // face just below the critical angle and leaves through bare PMMA.
                    let outer = match std::env::var("TUBE_OUTER").as_deref() {
                        Ok("silver") => Some(oc_core::thinfilm::metal_mirror(oc_core::thinfilm::materials::silver(), None)),
                        Ok("aluminium") => Some(oc_core::thinfilm::metal_mirror(oc_core::thinfilm::materials::aluminium_measured(), None)),
                        Ok("bare") | Err(_) => None,
                        Ok(other) => panic!("TUBE_OUTER: bare, silver or aluminium, not {other}"),
                    };
                    let outer_set = outer.is_some();
                    if let Some(c) = outer {
                        b.surface.overrides.push(oc_core::scene3::PatchResponseOverride {
                            patch: oc_core::id::PatchId(oc_core::geom3::primitives::tube_face::OUTER as u64),
                            response: oc_core::scene3::TwoSidedSurfaceResponse {
                                from_inside: SurfaceResponse::Coating { coating: c.seen_from_behind() },
                                from_outside: SurfaceResponse::Coating { coating: c },
                            },
                        });
                    }
                    // `TUBE_AR=bore|outer|both`: the single-layer AR of the Fresnel lens
                    // (MgF2, quarter-wave at 550 nm, normal incidence) on the PMMA faces. No
                    // design, so the tracer follows its reflection as it does the bare face's.
                    use oc_core::geom3::primitives::tube_face::{INNER, OUTER};
                    let ar_faces: &[u32] = match std::env::var("TUBE_AR").as_deref() {
                        Ok("bore") => &[INNER],
                        Ok("outer") => &[OUTER],
                        Ok("both") => &[INNER, OUTER],
                        Ok("none") | Err(_) => &[],
                        Ok(other) => panic!("TUBE_AR: none, bore, outer or both, not {other}"),
                    };
                    assert!(!(ar_faces.contains(&OUTER) && outer_set), "TUBE_AR outer and TUBE_OUTER both coat the outer face");
                    for &face in ar_faces {
                        let mut c = oc_core::thinfilm::film_coating(
                            CoatingDesign { kind: DesignKind::SingleLayerAr, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38 },
                            1.0,
                            pmma.index(0.55),
                        );
                        c.design = None;
                        b.surface.overrides.push(oc_core::scene3::PatchResponseOverride {
                            patch: oc_core::id::PatchId(face as u64),
                            response: oc_core::scene3::TwoSidedSurfaceResponse {
                                from_inside: SurfaceResponse::Coating { coating: c.seen_from_behind() },
                                from_outside: SurfaceResponse::Coating { coating: c },
                            },
                        });
                    }
                    realism::absorb(&mut s, tube, "pmma");
                    label(&mut s, tube, &format!("Light pipe {name}, section {} of {n}, clear PMMA", i + 1));
                }
            }
            realism::slope(&mut s, tube, env_f64("BORE_SLOPE", 1.0));
        }
    }
    // `TUBE_STOP_R=0.2` closes the tube's entrance: the leak test, after
    // which the 32 m and 37 m probes must read nothing.
    let mouth = s.add_aperture(Vec3::new(entry - 1.0, Y, 0.0), env_f64("TUBE_STOP_R", bore_radius()), 0.0);
    if let Some(a) = s.apertures.iter_mut().find(|a| a.common.id.0 == mouth) {
        a.shape.outer_extent = Some(PlanarShape2::Disc { radius_mm: 750.0 });
    }
    // The 32 m and 37 m probes take only what leaves a tube's end: each sits
    // 1 mm past it, the tube's outside size, behind a black mask 0.5 mm past
    // it that opens 2 mm wider than the tube (the sections lie up to ~1.5 mm
    // off the line) and is dark out to 3 m.
    let first_end = gap_at - half_gap;
    let last_end = entry + TUBE_LEN;
    for (x, name) in [(first_end, "Mask after the 0-32 m length"), (last_end, "Mask after the tube's end")] {
        let m = s.add_aperture(Vec3::new(x + 0.5, Y, 0.0), outer_radius() + 2.0, 0.0);
        if let Some(a) = s.apertures.iter_mut().find(|a| a.common.id.0 == m) {
            a.shape.outer_extent = Some(PlanarShape2::Disc { radius_mm: 3_000.0 });
        }
        label(&mut s, m, name);
    }
    label(&mut s, mouth, &format!("Tube entrance stop, {:.0} mm, mask 1500 mm", 2.0 * bore_radius()));

    // The sun's beam: as wide as the full system makes it, 3 m up-sun.
    s.resolve_solar().expect("sun resolves");
    if let Some(rig) = s.solar.as_mut() {
        rig.aim_at = Some(ObjectId(heliostat.0));
        rig.standoff_mm = sun_standoff();
        // `SUN_DISC_R` (mm): a disc just covering the mirror as the sun sees it,
        // W x L cos(theta), in place of one covering its diagonal. A long
        // mirror at grazing incidence otherwise needs a stop's mask so wide
        // that it reaches the light pipe below.
        if let Ok(v) = std::env::var("SUN_DISC_R") {
            rig.beam_radius_mm = v.parse().expect("SUN_DISC_R in mm");
        }
    }
    s.resolve_solar().expect("sun resolves");
    let sun_src = s.solar.as_ref().unwrap().source;
    let sun_at = s.sources.iter().find(|x| x.id == sun_src).unwrap().transform.pos;
    let sun = (sun_at - helio_at).normalize();
    let to_target = (fold_at.unwrap_or(pick_at) - helio_at).normalize();
    // Probe 3 3 m along the heliostat's beam, or halfway to a closer fold.
    let probe3_at = (0.5 * (fold_at.unwrap_or(pick_at) - helio_at).length()).min(3_000.0);
    let beam_r = s.solar.as_ref().unwrap().beam_radius_mm;

    // The probes, in the order the light meets them.
    let mx = -Vec3::X;
    let p = vec![
        probe(&mut s, helio_at + sun * sun_probe(), sun, disc(beam_r + 50.0), 180, false, "1 sun beam, up-sun"),
        face_probe(&mut s, heliostat.0, 1.0, "2 heliostat face"),
        probe(&mut s, helio_at + to_target * probe3_at, -to_target, square(1_000.0), 200, false, "3 after heliostat"),
        face_probe(&mut s, pickup, -1.0, "4 pick-up mirror face"),
        probe(&mut s, Vec3::new(x_fresnel - 40.0, Y, 0.0), mx, square(900.0), 180, false, "5 after pick-up, whole beam"),
        probe(&mut s, Vec3::new(x_fresnel - 10.0, Y, 0.0), mx, disc(500.0), 100, false, "6 before Fresnel, clear aperture"),
        probe(&mut s, Vec3::new(valleys + 20.0, Y, 0.0), mx, disc(500.0), 100, false, "7 after Fresnel, clear aperture"),
        probe(&mut s, Vec3::new(vertex + rim_sag.min(0.0) - 25.0, Y, 0.0), mx, disc(230.0), 92, false, "8 before eyepiece"),
        probe(&mut s, Vec3::new(vertex + eye_len + 10.0, Y, 0.0), mx, disc(230.0), 92, false, "9 after eyepiece, before wall hole"),
        probe(&mut s, Vec3::new(hole_x + 20.0, Y, 0.0), mx, disc(230.0), 92, false, "10 after wall hole, before tube stop"),
        probe(&mut s, Vec3::new(entry + 1.0, Y, 0.0), mx, disc(bore_radius() - 0.1), 76, false, "11 inside tube, 1 mm"),
        probe(&mut s, Vec3::new(first_end + 1.0, Y, 0.0), mx, disc(outer_radius() + 2.0), 80, false, "12 tube 32 m (end of first length)"),
        probe(&mut s, Vec3::new(last_end + 1.0, Y, 0.0), mx, disc(outer_radius() + 2.0), 80, true, "13 tube end 37 m"),
    ];

    // 2048 rays in the file, as the full system's.
    if let Some(src) = s.sources.iter_mut().find(|x| x.id == sun_src) {
        src.samples.spatial_samples = 1024;
        src.samples.angular_samples = 2;
    }
    let state = s.resolve_solar().expect("sun resolves");
    // `DNI_MODEL=din5034-2`: the direct sun of DIN 5034-2's clear sky, as in
    // Bartenbach's study, I0 (1 + 0.033 cos(2 pi J / 365)) exp(-T_L / (0.9 + 9.4 sin h)),
    // with the month's Linke turbidity for Jena: the median, at solar noon, of
    // what CAMS McClear's clear sky implies over 2004-2025 at the museum.
    // `LINKE_SITE=erfurt` takes Bartenbach's Erfurt values (their slide 25)
    // instead; `LINKE_TL` overrides both. It replaces the table's broadband
    // 900.1 W/m²; the beam still carries the traced band's share, 280 to 2600 nm.
    if std::env::var("DNI_MODEL").as_deref() == Ok("din5034-2") {
        const LINKE_JENA: [f64; 12] = [2.44, 2.62, 3.11, 3.59, 3.93, 4.21, 4.21, 4.06, 3.56, 3.10, 2.65, 2.44];
        const LINKE_ERFURT: [f64; 12] = [2.4, 2.6, 3.4, 2.8, 4.4, 4.3, 5.2, 5.2, 4.5, 3.1, 2.9, 2.3];
        const DAYS_BEFORE: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let erfurt = std::env::var("LINKE_SITE").as_deref() == Ok("erfurt");
        let tl = env_f64("LINKE_TL", if erfurt { LINKE_ERFURT } else { LINKE_JENA }[month as usize - 1]);
        let j = f64::from(DAYS_BEFORE[month as usize - 1] + day);
        let h = state.elevation_deg.to_radians();
        let broadband = 1367.0 * (1.0 + 0.033 * (2.0 * std::f64::consts::PI * j / 365.0).cos()) * (-tl / (0.9 + 9.4 * h.sin())).exp();
        s.solar.as_mut().unwrap().dni_w_m2 = broadband;
        eprintln!(
            "direct sun: DIN 5034-2, T_L {tl:.2} ({}), day {j}, elevation {:.2}°: {broadband:.1} W/m² broadband ({total:.1} in the table)",
            if erfurt { "Erfurt" } else { "Jena" },
            state.elevation_deg
        );
        s.resolve_solar().expect("sun resolves");
    }
    s.validate().expect("scene validates");
    eprintln!("layout {lay:?}: heliostat at ({:.0}, {:.0}) mm, fold mirror {:?}", helio_at.x, helio_at.y, fold_at.map(|f| (f.x.round(), f.y.round())));
    eprintln!(
        "tube {:.0} / {:.0} mm; sun beam disc {:.1} mm radius, {:.0} mm up-sun; Fresnel valleys x {valleys:.1}, eyepiece vertex x {vertex:.1}, wall hole x {hole_x:.1}, tube x {entry:.1}..{:.1}, gap {:.1}..{:.1}",
        2.0 * bore_radius(),
        2.0 * outer_radius(),
        beam_r,
        sun_standoff(),
        entry + TUBE_LEN,
        gap_at - half_gap,
        gap_at + half_gap
    );
    (s, p)
}

fn stop_on(s: &mut Scene3, parent: u64, gap_mm: f64, opening: PlanarShape2, outer: PlanarShape2, name: &str) -> u64 {
    let at = s
        .surfaces
        .iter()
        .find(|x| x.common.id.0 == parent)
        .expect("the stop's mirror")
        .common
        .transform
        .compose(Isometry3::from_pos(Vec3::new(0.0, 0.0, gap_mm)));
    let id = s.add_aperture(Vec3::ZERO, 1.0, 0.0);
    let a = s.apertures.iter_mut().find(|x| x.common.id.0 == id).expect("the stop");
    a.common.transform = at;
    a.shape.clear_opening = opening;
    a.shape.outer_extent = Some(outer);
    s.link_object(id, parent, LinkMode::Follow).expect("lock the stop");
    label(s, id, name);
    id
}

fn label(s: &mut Scene3, id: u64, text: &str) {
    if let Some(c) = s
        .bodies
        .iter_mut()
        .map(|b| &mut b.common)
        .chain(s.surfaces.iter_mut().map(|x| &mut x.common))
        .chain(s.apertures.iter_mut().map(|x| &mut x.common))
        .find(|c| c.id.0 == id)
    {
        c.label = text.into();
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let kind = match args.next().as_deref() {
        Some("silver") => TubeKind::Silver,
        Some("pmma") => TubeKind::Pmma,
        other => panic!("first argument: silver or pmma, not {other:?}"),
    };
    let out = args.next().unwrap_or_else(|| "dom-probes.ocanvas".into());
    let (mut s, probes) = build(kind);
    std::fs::write(&out, s.to_json()).expect("write the scene");
    eprintln!("wrote {out}");

    let Ok(n) = std::env::var("VERIFY") else { return };
    let n: u32 = n.parse().expect("VERIFY is a sample count");
    let src = s.solar.as_ref().unwrap().source;
    if let Some(src) = s.sources.iter_mut().find(|x| x.id == src) {
        src.samples.spatial_samples = n;
    }
    let state = s.resolve_solar().unwrap();
    // `POINT_SUN=1`: a sun of 0.01 mrad, the test of what the sun's size does to a metric.
    if env_f64("POINT_SUN", 0.0) > 0.5 {
        if let Some(src) = s.sources.iter_mut().find(|x| x.id == src) {
            src.angular = oc_core::scene3::AngularDistribution3::UniformConeSolidAngle { half_angle_rad: 1e-5 };
        }
    }
    eprintln!(
        "sun: az {:.2}°, el {:.2}°, beam {:.1} W; heliostat incidence {:.2}°",
        state.azimuth_deg, state.elevation_deg, state.source_flux_w, state.heliostats[0].incidence_deg
    );
    let (ghosts, max_ghost_bounces) = realism::ghosts();
    // `BUDGET=1`: the energy budget. The ledger object by object, 390-700 nm
    // apart from the rest (objects.csv), each probe's forward spectrum in 1 nm
    // bins (spectra.f64), and every ray reaching the last probe with its
    // position, direction, wavelength, power and its reflections in the tube
    // (exit_rays.npy). It changes no other number.
    let budget = env_f64("BUDGET", 0.0) > 0.5;
    let tubes: Vec<usize> = s
        .bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.geometry, oc_core::scene3::SolidGeometry3::Tube { .. }))
        .map(|(i, _)| i)
        .collect();
    let tube_range = match (tubes.first(), tubes.last()) {
        (Some(&a), Some(&b)) => {
            assert_eq!(b + 1 - a, tubes.len(), "the tube's sections follow each other in the scene");
            [a, b + 1]
        }
        _ => [0, 0],
    };
    // `THERMAL=1`: where the Fresnel lens and the eyepiece absorb, and where the
    // stops and masks block, every deposit in its object's own frame
    // (deposits.npy + deposits.json). It changes no other number.
    let thermal = env_f64("THERMAL", 0.0) > 0.5;
    let lenses: Vec<usize> = s
        .bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.common.label.starts_with("Fresnel") || b.common.label.starts_with("Eyepiece"))
        .map(|(i, _)| i)
        .collect();
    let lens_range = match (lenses.first(), lenses.last()) {
        (Some(&a), Some(&b)) if thermal => {
            assert_eq!(b + 1 - a, lenses.len(), "the lenses follow each other in the scene");
            [a, b + 1]
        }
        _ => [0, 0],
    };
    let cfg = TraceConfig3 {
        deposit_bodies: lens_range,
        deposit_apertures: thermal,
        object_ledger: budget,
        ledger_band_um: [0.39, 0.70],
        count_bodies: if budget { tube_range } else { [0, 0] },
        ghosts,
        max_ghost_bounces,
        // A ray in the clear tube splits at every wall it meets; branches
        // under 1e-4 of their ray's flux are dropped (and counted as
        // truncated), which keeps the records of a PMMA run in memory.
        // `FLUX_EPS`, `MAX_DEPTH` override.
        flux_eps: env_f64("FLUX_EPS", 1e-4),
        max_depth: env_f64("MAX_DEPTH", depth_for_scene3(&s, ghosts, max_ghost_bounces) as f64) as u32,
        // `MAX_BRANCHES` (100000, the tracer's default): the branches one launched ray
        // may spawn before the rest of its tree is dropped. A mirrored PMMA wall splits
        // a ray at every crossing; check `branches ended by` for this cap.
        max_branches: env_f64("MAX_BRANCHES", 100_000.0) as u64,
        bounds_mm: bounds_for_scene3(&s),
        record_detector_hits: true,
        max_records: usize::MAX,
        segment_floor: f64::INFINITY,
        ..TraceConfig3::default()
    };
    // `FIDELITY=Fast|Standard|Reference`: the coating physics the trace pays for.
    let cfg = cfg.with_fidelity(realism::fidelity());
    // `PASSES` passes of the source's rays, each an independent, evenly
    // spread set (a Cranley-Patterson shift of the same point sets) carrying
    // 1/PASSES of the flux. Each pass's records are folded into the sums
    // below and dropped, so the ray count is bounded by time, not memory.
    let passes: u32 = env_f64("PASSES", 1.0) as u32;
    let mut acc: Vec<Acc> = probes
        .iter()
        .map(|p| {
            let d = s.detectors.iter().find(|d| d.common.id == p.id).unwrap();
            let (hu, hv) = match d.geometry {
                DetectorGeometry3::Rectangle { half_width_mm, half_height_mm } => (half_width_mm, half_height_mm),
                DetectorGeometry3::Disc { radius_mm } => (radius_mm, radius_mm),
                _ => (1.0, 1.0),
            };
            Acc::new(hu, hv)
        })
        .collect();
    let mut ledger = [0.0f64; 7];
    // Branch terminations by cause, summed over passes: which cap a truncation came from.
    let mut ends = [0u64; 4];
    let mut objects: std::collections::BTreeMap<(oc_trace::ObjectRef3, bool), oc_trace::ObjectLedger3> = Default::default();
    // The last probe's rays: x y z (world mm), u v (face mm), dx dy dz (world), lambda (um), W, tube hits.
    let mut exit_rays: Vec<[f64; 11]> = Vec::new();
    let last = probes.last().map(|p| p.id);
    let mut deposits: Vec<oc_trace::Deposit3> = Vec::new();
    let (mut rays, mut trace_s) = (0usize, 0.0f64);
    let t_all = std::time::Instant::now();
    // `PASS_OFFSET`: start at another shift, for independent replicates of a run.
    let k0: u32 = env_f64("PASS_OFFSET", 0.0) as u32;
    for k in k0..k0 + passes {
        let mut launch = oc_trace::launch_from_scene_pass(&s, k);
        for r in &mut launch {
            r.flux /= passes as f64;
        }
        rays += launch.len();
        let t = std::time::Instant::now();
        let res = oc_trace::trace3(&s, &launch, &cfg);
        trace_s += t.elapsed().as_secs_f64();
        drop(launch);
        for (p, a) in probes.iter().zip(acc.iter_mut()) {
            let d = res.detectors.iter().find(|d| d.detector.0 == p.id.0).unwrap();
            assert_eq!(d.records.len() as u64, d.hits, "{}: every hit recorded", p.name);
            a.add(d);
            if budget {
                a.add_spectrum(d);
                if Some(p.id) == last {
                    let t = s.detectors.iter().find(|x| x.common.id == p.id).unwrap().common.transform;
                    for h in &d.records {
                        let dw = t.rotate(h.incident_dir_local);
                        let q = h.point_world;
                        exit_rays.push([q.x, q.y, q.z, h.uv_mm.x, h.uv_mm.y, dw.x, dw.y, dw.z, h.lambda_um, h.incident_flux, h.counted as f64]);
                    }
                }
            }
        }
        for e in &res.objects {
            objects.entry((e.object, e.in_band)).or_default().add(&e.ledger);
        }
        deposits.extend_from_slice(&res.deposits);
        let tm = &res.terminations;
        for (x, y) in ends.iter_mut().zip([tm.flux_threshold, tm.max_depth, tm.max_branches, tm.max_ghost_depth]) {
            *x += y;
        }
        let l = &res.ledger;
        for (x, y) in ledger.iter_mut().zip([l.launched, l.escaped, l.absorbed_volume, l.absorbed_surface, l.blocked_aperture, l.truncated_flux, l.numerical_loss]) {
            *x += y;
        }
        if passes > 1 && (k + 1) % 5 == 0 {
            let el = t_all.elapsed().as_secs_f64();
            eprintln!("pass {}/{passes}: {rays} rays, {el:.0} s, about {:.0} s to go", k + 1, el / (k + 1) as f64 * (passes - k - 1) as f64);
        }
    }
    let wall_s = t_all.elapsed().as_secs_f64();
    eprintln!(
        "traced {rays} rays in {passes} pass(es): {trace_s:.1} s tracing, {wall_s:.1} s with the sums ({:.3} M rays/s), depth budget {}",
        rays as f64 / wall_s / 1e6,
        cfg.max_depth
    );
    let launched = ledger[0];
    eprintln!(
        "branches ended by: flux threshold {}, max depth {}, max branches {}, max ghost depth {}",
        ends[0], ends[1], ends[2], ends[3]
    );

    // Per probe: flux crossing toward the light's way (forward) and back,
    // the visible share, and for the tube planes, inside the bore against
    // inside the wall's ring.
    let mut csv = String::from("probe,size,forward_w,backward_w,pct_of_sun,pct_of_previous,vis_380_780_w,bore_w,wall_ring_w,r_rms_mm,r_max_mm,angle_rms_deg,angle_max_deg,hits\n");
    eprintln!(
        "{:<36} {:>22} {:>9} {:>8} {:>7} {:>7} {:>9} {:>8} {:>8} {:>7} {:>7} {:>6} {:>6}",
        "probe", "size", "fwd W", "back W", "% sun", "% prev", "vis W", "r rms", "r max", "a rms", "a max", "bore W", "wall W"
    );
    // The residual focusing per probe inside the tube, as the radial angle at the bore's edge.
    for (p, a) in probes.iter().zip(&acc) {
        if ["10 ", "11 "].iter().any(|t| p.name.starts_with(t)) && a.srr > 0.0 {
            eprintln!("focus residual at {}: {:.3} mrad at r = {:.1} mm", p.name, 1e3 * a.srt / a.srr * bore_radius(), bore_radius());
            eprintln!("rms angle at {}: {:.4} mrad over {:.3} W (inside the bore, within 3 deg)", p.name, 1e3 * (a.saa / a.sw).sqrt(), a.sw);
        }
    }
    let mut prev = launched;
    for (p, a) in probes.iter().zip(&acc) {
        let fwd = a.fwd;
        let (r_rms, a_rms) = ((a.r2 / fwd.max(1e-300)).sqrt(), (a.a2 / fwd.max(1e-300)).sqrt().to_degrees());
        let in_tube = ["11 ", "12 ", "13 "].iter().any(|t| p.name.starts_with(t));
        let (bore, ring) = if in_tube { (a.bore, a.ring) } else { (f64::NAN, f64::NAN) };
        eprintln!(
            "{:<36} {:>22} {:9.2} {:8.3} {:7.2} {:7.2} {:9.2} {:8.1} {:8.1} {:7.3} {:7.3} {:6.1} {:6.2}",
            p.name,
            p.size,
            fwd,
            a.back,
            100.0 * fwd / launched,
            100.0 * fwd / prev,
            a.vis,
            r_rms,
            a.rmax,
            a_rms,
            a.amax.to_degrees(),
            bore,
            ring
        );
        csv.push_str(&format!(
            "\"{}\",\"{}\",{fwd:.4},{:.5},{:.3},{:.3},{:.4},{bore:.4},{ring:.5},{r_rms:.2},{:.2},{a_rms:.4},{:.4},{}\n",
            p.name,
            p.size,
            a.back,
            100.0 * fwd / launched,
            100.0 * fwd / prev,
            a.vis,
            a.rmax,
            a.amax.to_degrees(),
            a.hits
        ));
        // After a mirror's face probe, what follows is measured against
        // what the mirror sent on.
        prev = if p.size.contains("(in:") { a.back } else { fwd };
    }
    eprintln!(
        "ledger W: launched {:.2}, escaped {:.2}, absorbed volume {:.2}, absorbed surface {:.2}, blocked by stops {:.2}, truncated {:.3}, numerical {:.3}",
        ledger[0], ledger[1], ledger[2], ledger[3], ledger[4], ledger[5], ledger[6]
    );

    // `OUT_DIR`: budget.csv, the radiant intensity histograms, meta.json
    // with the timing.
    if let Ok(dir) = std::env::var("OUT_DIR") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).expect("the output folder");
        std::fs::write(dir.join("budget.csv"), csv).expect("write the budget");
        // Radial: forward flux by angle to the probe's normal.
        let edges = radial_edges();
        let mut rad = String::from("probe,theta_lo_deg,theta_hi_deg,flux_w\n");
        for (p, a) in probes.iter().zip(&acc) {
            for (j, f) in a.radial.iter().enumerate() {
                rad.push_str(&format!("\"{}\",{},{},{:.6e}\n", p.name, edges[j], edges[j + 1], f));
            }
        }
        std::fs::write(dir.join("intensity_radial.csv"), rad).expect("write the radial histograms");
        // Maps: forward flux by (atan(dx/-dz), atan(dy/-dz)) in the probe's
        // frame, MAP_N x MAP_N bins over +-MAP_DEG, row = y, f64 little
        // endian, one map per probe in probe order.
        let mut bytes = Vec::with_capacity(acc.len() * MAP_N * MAP_N * 8);
        for a in &acc {
            for v in &a.map {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(dir.join("intensity_maps.f64"), bytes).expect("write the maps");
        // Per probe, f64 little endian, row = v: the forward flux by position
        // on the face (SP_N x SP_N over its size), then by direction
        // (ANG_N x ANG_N): angles from the beam's mean direction m along e1
        // (local x made square to m) and e2 = e1 x m, over a window 1.5 times
        // the angle holding 95% of its power.
        let mut bytes = Vec::new();
        let mut info = Vec::new();
        for (p, a) in probes.iter().zip(&acc) {
            for v in a.sp.iter().chain(&a.ang) {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let t = s.detectors.iter().find(|d| d.common.id == p.id).unwrap().common.transform;
            let (ux, vx) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
            let (m, _, _, half) = a.win.unwrap_or((-Vec3::Z, Vec3::X, Vec3::Y, 1.0));
            let tilt = (-m.z).clamp(-1.0, 1.0).acos().to_degrees();
            info.push(format!(
                "{{\"name\": \"{}\", \"size\": \"{}\", \"fwd_w\": {}, \"half_u_mm\": {}, \"half_v_mm\": {}, \"mean_dir_from_normal_deg\": {tilt}, \"ang_half_deg\": {half}, \"u_world\": [{:.4}, {:.4}, {:.4}], \"v_world\": [{:.4}, {:.4}, {:.4}], \"sp_out_w\": {}, \"ang_out_w\": {}}}",
                p.name, p.size, a.fwd, a.hu, a.hv, ux.x, ux.y, ux.z, vx.x, vx.y, vx.z, a.sp_out, a.ang_out
            ));
        }
        std::fs::write(dir.join("probe_maps.f64"), bytes).expect("write the probe maps");
        std::fs::write(dir.join("probe_maps.json"), format!("{{\"n_sp\": {SP_N}, \"n_ang\": {ANG_N}, \"probes\": [\n{}\n]}}\n", info.join(",\n")))
            .expect("write the probe map meta");
        let meta = format!(
            "{{\"probes\": [{}], \"map_n\": {MAP_N}, \"map_deg\": {MAP_DEG}, \"map_outside_w\": [{}], \"rays\": {rays}, \"passes\": {passes}, \"trace_s\": {trace_s:.2}, \"wall_s\": {wall_s:.2}, \"inner_d_mm\": {}, \"outer_d_mm\": {}, \"wall_mm\": {}, \"tube\": \"{}\", \"ledger_w\": {{\"launched\": {}, \"escaped\": {}, \"absorbed_volume\": {}, \"absorbed_surface\": {}, \"blocked\": {}, \"truncated\": {}, \"numerical\": {}}}}}\n",
            probes.iter().map(|p| format!("\"{}\"", p.name)).collect::<Vec<_>>().join(", "),
            acc.iter().map(|a| format!("{:.6e}", a.map_out)).collect::<Vec<_>>().join(", "),
            2.0 * bore_radius(),
            2.0 * outer_radius(),
            outer_radius() - bore_radius(),
            if kind == TubeKind::Silver { "silver" } else { "pmma" },
            ledger[0],
            ledger[1],
            ledger[2],
            ledger[3],
            ledger[4],
            ledger[5],
            ledger[6]
        );
        std::fs::write(dir.join("meta.json"), meta).expect("write the meta");
        if budget {
            write_budget(dir, &s, &probes, &acc, &objects, &exit_rays, tube_range);
        }
        if thermal {
            write_deposits(dir, &s, &deposits);
        }
        eprintln!("wrote {}", dir.display());
    }
}

/// The budget files (`BUDGET=1`): objects.csv, spectra.f64 + spectra.json, exit_rays.npy.
fn write_budget(
    dir: &std::path::Path,
    s: &Scene3,
    probes: &[Probe],
    acc: &[Acc],
    objects: &std::collections::BTreeMap<(oc_trace::ObjectRef3, bool), oc_trace::ObjectLedger3>,
    exit_rays: &[[f64; 11]],
    tube_range: [usize; 2],
) {
    use oc_trace::ObjectRef3 as O;
    let name = |o: &O| -> (String, String) {
        let (kind, common, extra) = match *o {
            O::Source => return ("source".into(), "Sun".into()),
            O::Surface(i) => ("surface", Some(&s.surfaces[i].common), String::new()),
            O::Body(i, patch) => {
                let face = if matches!(s.bodies[i].geometry, oc_core::scene3::SolidGeometry3::Tube { .. }) {
                    use oc_core::geom3::primitives::tube_face as f;
                    match patch as u32 {
                        f::INNER => "bore",
                        f::OUTER => "outer wall",
                        f::END_MINUS => "entry ring",
                        f::END_PLUS => "exit ring",
                        _ => "patch",
                    }
                    .to_string()
                } else {
                    String::new()
                };
                let c = &s.bodies[i].common;
                let base = if c.label.is_empty() { format!("body {}", c.id.0) } else { c.label.clone() };
                return ("body".into(), if face.is_empty() { base } else { format!("{base} | {face}") });
            }
            O::Aperture(i) => ("aperture", Some(&s.apertures[i].common), String::new()),
            O::Detector(i) => (
                "detector",
                Some(&s.detectors[i].common),
                probes.iter().find(|p| p.id == s.detectors[i].common.id).map_or(String::new(), |p| p.name.to_string()),
            ),
        };
        let c = common.unwrap();
        let label = if !extra.is_empty() { extra } else if !c.label.is_empty() { c.label.clone() } else { format!("{kind} {}", c.id.0) };
        (kind.into(), label)
    };
    let mut csv = String::from("kind,index,label,band,hits,incident_w,absorbed_surface_w,absorbed_volume_w,blocked_w,detected_w,escaped_w,truncated_w,numerical_w\n");
    for ((o, vis), l) in objects {
        let (kind, label) = name(o);
        let index = match *o {
            O::Source => 0,
            O::Surface(i) | O::Body(i, _) | O::Aperture(i) | O::Detector(i) => i,
        };
        csv.push_str(&format!(
            "{kind},{index},\"{}\",{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            label.replace('"', "'"),
            if *vis { "390-700" } else { "other" },
            l.hits,
            l.incident,
            l.absorbed_surface,
            l.absorbed_volume,
            l.blocked,
            l.detected,
            l.escaped,
            l.truncated,
            l.numerical
        ));
    }
    std::fs::write(dir.join("objects.csv"), csv).expect("write the object ledger");
    let mut bytes = Vec::new();
    for a in acc {
        for v in &a.spec {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("spectra.f64"), bytes).expect("write the spectra");
    std::fs::write(
        dir.join("spectra.json"),
        format!(
            "{{\"lambda0_nm\": {SPEC_NM0}, \"step_nm\": 1, \"bins\": {SPEC_N}, \"probes\": [{}], \"tube_bodies\": [{}, {}]}}\n",
            probes.iter().map(|p| format!("\"{}\"", p.name)).collect::<Vec<_>>().join(", "),
            tube_range[0],
            tube_range[1]
        ),
    )
    .expect("write the spectra meta");
    // A version 1.0 .npy: little-endian f64, C order, (n, 11).
    let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, 11), }}", exit_rays.len());
    let pad = 64 - (10 + header.len() + 1) % 64;
    let mut npy = b"\x93NUMPY\x01\x00".to_vec();
    npy.extend_from_slice(&((header.len() + pad + 1) as u16).to_le_bytes());
    npy.extend_from_slice(header.as_bytes());
    npy.extend(std::iter::repeat(b' ').take(pad));
    npy.push(b'\n');
    for r in exit_rays {
        for v in r {
            npy.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("exit_rays.npy"), npy).expect("write the exit rays");
    eprintln!("budget: {} object rows, {} exit rays", objects.len(), exit_rays.len());
}

/// The thermal files (`THERMAL=1`): deposits.npy, one row per deposit:
/// object number (deposits.json names them), x y z in that object's frame
/// (mm), wavelength (um), W.
fn write_deposits(dir: &std::path::Path, s: &Scene3, deposits: &[oc_trace::Deposit3]) {
    use oc_trace::ObjectRef3 as O;
    let mut names: Vec<(O, String, String)> = Vec::new();
    let mut rows: Vec<[f64; 6]> = Vec::with_capacity(deposits.len());
    for d in deposits {
        let key = match d.object {
            O::Body(i, _) => O::Body(i, 0),
            o => o,
        };
        let k = match names.iter().position(|n| n.0 == key) {
            Some(k) => k,
            None => {
                let (label, frame) = match key {
                    O::Body(i, _) => (s.bodies[i].common.label.clone(), format!("{:?}", s.bodies[i].common.transform)),
                    O::Aperture(i) => (s.apertures[i].common.label.clone(), format!("{:?}", s.apertures[i].common.transform)),
                    O::Surface(i) => (s.surfaces[i].common.label.clone(), String::new()),
                    _ => ("other".into(), String::new()),
                };
                names.push((key, label, frame));
                names.len() - 1
            }
        };
        let t = match key {
            O::Body(i, _) => s.bodies[i].common.transform,
            O::Aperture(i) => s.apertures[i].common.transform,
            O::Surface(i) => s.surfaces[i].common.transform,
            _ => Isometry3::from_pos(Vec3::ZERO),
        };
        let p = t.inverse_transform_point(d.point);
        rows.push([k as f64, p.x, p.y, p.z, d.lambda_um, d.flux]);
    }
    let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, 6), }}", rows.len());
    let pad = 64 - (10 + header.len() + 1) % 64;
    let mut npy = b"\x93NUMPY\x01\x00".to_vec();
    npy.extend_from_slice(&((header.len() + pad + 1) as u16).to_le_bytes());
    npy.extend_from_slice(header.as_bytes());
    npy.extend(std::iter::repeat(b' ').take(pad));
    npy.push(b'\n');
    for r in &rows {
        for v in r {
            npy.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("deposits.npy"), npy).expect("write the deposits");
    let list: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(k, n)| format!("{{\"index\": {k}, \"label\": \"{}\", \"frame\": \"{}\"}}", n.1.replace('"', "'"), n.2.replace('"', "'")))
        .collect();
    std::fs::write(dir.join("deposits.json"), format!("{{\"objects\": [\n{}\n]}}\n", list.join(",\n"))).expect("write the deposit names");
    eprintln!("thermal: {} deposits on {} objects", rows.len(), names.len());
}

/// The budget's spectra: 1 nm bins from 280 nm.
const SPEC_NM0: f64 = 280.0;
const SPEC_N: usize = 3720;
const MAP_N: usize = 200;
const SP_N: usize = 40;
const ANG_N: usize = 64;
const MAP_DEG: f64 = 5.0;

/// Radial bin edges in degrees: 0.02 to 5, 0.5 to 30, 5 to 90.
fn radial_edges() -> Vec<f64> {
    let mut e: Vec<f64> = (0..=250).map(|i| i as f64 * 0.02).collect();
    e.extend((1..=50).map(|i| 5.0 + i as f64 * 0.5));
    e.extend((1..=12).map(|i| 30.0 + i as f64 * 5.0));
    e
}

/// One probe's sums over every pass.
struct Acc {
    fwd: f64,
    back: f64,
    vis: f64,
    bore: f64,
    ring: f64,
    r2: f64,
    a2: f64,
    rmax: f64,
    amax: f64,
    hits: u64,
    edges: Vec<f64>,
    radial: Vec<f64>,
    map: Vec<f64>,
    map_out: f64,
    hu: f64,
    hv: f64,
    sp: Vec<f64>,
    sp_out: f64,
    /// The direction window: the beam's mean direction, the two axes square
    /// to it, and the half width in degrees, set by the first pass.
    win: Option<(Vec3, Vec3, Vec3, f64)>,
    ang: Vec<f64>,
    ang_out: f64,
    /// The residual focusing of the beam: flux-weighted sums of r^2 and of r times the
    /// radial ray angle (rad), over rays inside the bore and within 3 deg of the axis.
    /// Their ratio is the radial angle per mm of radius; the chromatic study prints it.
    srr: f64,
    srt: f64,
    /// Over the same rays: their flux and the flux-weighted sum of the squared angle to the
    /// axis (rad^2), so the rms angle without binning (the collimation study).
    sw: f64,
    saa: f64,
    /// The forward flux in 1 nm bins from `SPEC_NM0` (budget mode only).
    spec: Vec<f64>,
}

impl Acc {
    fn new(hu: f64, hv: f64) -> Self {
        let edges = radial_edges();
        let n = edges.len() - 1;
        Acc {
            fwd: 0.0,
            back: 0.0,
            vis: 0.0,
            bore: 0.0,
            ring: 0.0,
            r2: 0.0,
            a2: 0.0,
            rmax: 0.0,
            amax: 0.0,
            hits: 0,
            edges,
            radial: vec![0.0; n],
            map: vec![0.0; MAP_N * MAP_N],
            map_out: 0.0,
            hu,
            hv,
            sp: vec![0.0; SP_N * SP_N],
            sp_out: 0.0,
            win: None,
            ang: vec![0.0; ANG_N * ANG_N],
            ang_out: 0.0,
            srr: 0.0,
            srt: 0.0,
            sw: 0.0,
            saa: 0.0,
            spec: vec![0.0; SPEC_N],
        }
    }

    fn add_spectrum(&mut self, d: &oc_trace::DetectorResult3) {
        for h in d.records.iter().filter(|h| h.incident_dir_local.z <= 0.0) {
            let k = ((h.lambda_um * 1e3 - SPEC_NM0).floor().max(0.0) as usize).min(SPEC_N - 1);
            self.spec[k] += h.incident_flux;
        }
    }

    fn add(&mut self, d: &oc_trace::DetectorResult3) {
        self.hits += d.hits;
        let step = 2.0 * MAP_DEG / MAP_N as f64;
        let (rb, ro) = (bore_radius(), outer_radius());
        if self.win.is_none() {
            // The beam's mean direction, and 1.5 times the angle from it that
            // holds 95% of the forward flux (99% would follow the sun's aureole).
            let fw: Vec<(Vec3, f64)> =
                d.records.iter().filter(|h| h.incident_dir_local.z < 0.0).map(|h| (h.incident_dir_local, h.incident_flux)).collect();
            let w: f64 = fw.iter().map(|q| q.1).sum();
            if w > 0.0 {
                let m = fw.iter().fold(Vec3::ZERO, |acc, q| acc + q.0 * q.1).normalize();
                let e1 = (Vec3::X - m * Vec3::X.dot(m)).normalize();
                let e2 = e1.cross(m);
                let mut r: Vec<(f64, f64)> = fw.iter().map(|q| (q.0.dot(m).clamp(-1.0, 1.0).acos().to_degrees(), q.1)).collect();
                r.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut c = 0.0;
                let r95 = r
                    .iter()
                    .find(|q| {
                        c += q.1;
                        c >= 0.95 * w
                    })
                    .map_or(1.0, |q| q.0);
                self.win = Some((m, e1, e2, (1.5 * r95).max(0.05)));
            }
        }
        let (ns, na) = (SP_N as f64, ANG_N as f64);
        for h in &d.records {
            let w = h.incident_flux;
            let dir = h.incident_dir_local;
            if dir.z > 0.0 {
                self.back += w;
                continue;
            }
            self.fwd += w;
            let r = h.uv_mm.length();
            let a = (-dir.z).clamp(-1.0, 1.0).acos();
            if r < rb && a < 3f64.to_radians() {
                let (tx, ty) = (dir.x / -dir.z, dir.y / -dir.z);
                self.srr += w * r * r;
                self.srt += w * (h.uv_mm.x * tx + h.uv_mm.y * ty);
                self.sw += w;
                self.saa += w * (tx * tx + ty * ty);
            }
            self.r2 += w * r * r;
            self.a2 += w * a * a;
            self.rmax = self.rmax.max(r);
            self.amax = self.amax.max(a);
            if (0.38..0.78).contains(&h.lambda_um) {
                self.vis += w;
            }
            if r < rb {
                self.bore += w;
            } else if r < ro {
                self.ring += w;
            }
            let deg = a.to_degrees();
            let j = self.edges.partition_point(|&e| e <= deg).saturating_sub(1).min(self.radial.len() - 1);
            self.radial[j] += w;
            let (tx, ty) = ((dir.x / -dir.z).atan().to_degrees(), (dir.y / -dir.z).atan().to_degrees());
            let (ix, iy) = (((tx + MAP_DEG) / step).floor(), ((ty + MAP_DEG) / step).floor());
            if (0.0..MAP_N as f64).contains(&ix) && (0.0..MAP_N as f64).contains(&iy) {
                self.map[iy as usize * MAP_N + ix as usize] += w;
            } else {
                self.map_out += w;
            }
            let (iu, iv) = (((h.uv_mm.x + self.hu) / (2.0 * self.hu) * ns).floor(), ((h.uv_mm.y + self.hv) / (2.0 * self.hv) * ns).floor());
            if (0.0..ns).contains(&iu) && (0.0..ns).contains(&iv) {
                self.sp[iv as usize * SP_N + iu as usize] += w;
            } else {
                self.sp_out += w;
            }
            if let Some((m, e1, e2, half)) = self.win {
                let p = dir.dot(m);
                let (au, av) = ((dir.dot(e1) / p).atan().to_degrees(), (dir.dot(e2) / p).atan().to_degrees());
                let (iu, iv) = (((au + half) / (2.0 * half) * na).floor(), ((av + half) / (2.0 * half) * na).floor());
                if p > 0.0 && (0.0..na).contains(&iu) && (0.0..na).contains(&iv) {
                    self.ang[iv as usize * ANG_N + iu as usize] += w;
                } else {
                    self.ang_out += w;
                }
            }
        }
    }
}
