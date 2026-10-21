//! The projection spectrograph fed from the light pipe at 32 m, shared by the
//! demonstrator scene and its bench scene.
//!
//! The room: the tube's axis at `AXIS_Y`, the ceiling 250 mm above it (70 mm
//! above the tube's top), the floor 2.97 m below the ceiling, a table 0.55 m
//! high. Three
//! spectra land on the table side by side, each 375 to 690 nm over 2.25 m and
//! 180 mm across, 50 mm apart: the sun's in the middle, a halogen lamp's on
//! one side, a sodium lamp's on the other. Many of the numbers below are the
//! code's defaults; the locked design (round 4) sets fitted values through
//! the environment, from D:/tmp/spectrograph-bright/current-best/states.
//!
//! The sun's light, in order:
//!
//! 1. **Condenser pick-off**: an off-axis paraboloid strip on the tube's axis
//!    in a 200 mm gap between two lengths of tube, a few millimetres wide and
//!    as long as the bore, ground on a 6 mm glass bar. It takes a strip of the
//!    tube's light sideways.
//! 2. **Condenser mirror C1**, an off-axis ellipsoid used 6 degrees off its
//!    axis, and the **slit**, an off-axis paraboloid strip 0.45 mm wide that
//!    follows the arc the pick-off's image lies on. The three mirrors image the
//!    pick-off onto the slit at f/8.4, so the slit is lit by a strip of the
//!    whole bore.
//! 3. **UV/IR cut filter**, 375 to 690 nm, 30 mm past the slit.
//! 4. **Periscope P1/P2** drops the light to the collimator's height without
//!    turning the slit's image.
//! 5. **Collimator M1**, an off-axis paraboloid, f 1.8 m, its return beam 18
//!    degrees off the incoming one; then **fold F2**.
//! 6. **Grating**: ruled, 1800 lines/mm, the light arriving 50 degrees off its
//!    normal; the zero order goes into a dump 600 mm out, and a black mount
//!    behind the grating takes what the lamps send past it.
//! 7. **Camera M2**, a weakly aspheric sphere with the grating near its
//!    centre of curvature, leaning about 8.7 degrees so its light passes the
//!    grating; **M3**, a weakly convex fold that takes out the lean's bow.
//! 8. **Fold F4**, then the **field lens** at the intermediate spectrum: two
//!    PMMA Fresnels face to face, which send the camera's pupil into the
//!    projection lens.
//! 9. **Fold F5** and the **projection lens**: a Cooke triplet cut to a slab,
//!    N-SK16 / N-F2 with every face an even asphere, designed in Optiland,
//!    f 832 mm, throwing the spectrum about 2.3 times larger onto the table.
//!    `IDEAL_PROJ` and `IDEAL_FIELD` swap in thin lenses for comparison.
//!
//! The lamps ([`Lamp`], [`lamp_arm`]) have slits of their own in the sun
//! slit's focal surface, one either side of it along its length: an Osram
//! HLX 64625 halogen in an ellipsoidal reflector and an Osram Na/10 sodium
//! lamp behind a condenser, each with its own filter, folded into the shared
//! periscope.
//!
//! What the parts are made of is [`super::realism`]'s: every mirror enhanced
//! silver (the halogen's reflector enhanced aluminium), the grating's
//! efficiency by polarization from a rigorous calculation fitted to the
//! vendor's measurement, the glasses absorbing as Schott and the PMMA maker
//! measured them, the Fresnels' risers drafted 3 degrees. `IDEAL=1` puts back
//! the constants the design was done with.

// Placing an optic takes a position, two directions, a size and a name.
#![allow(clippy::too_many_arguments)]

use oc_core::scene3::{
    Aperture3, ApertureInteraction, ApertureShape3, DetectorGeometry3, DetectorTermination,
    Interpolation, ObjectCommon, OutOfRangePolicy, PlanarShape2, RadialAperture, SagModel, Scene3,
    SpectralScalar, SurfaceGeometry3, SurfaceObject3, SurfaceResponse,
};
use oc_core::build3::SingletGeom;
use oc_core::id::{LayerId, ObjectId, SourceId};
use oc_core::material::{Glass, Material, Sellmeier};
use oc_core::scene::Spectrum;
use oc_core::scene3::{AngularDistribution3, SamplingSequence, Source3, SourceSampling, SpatialDistribution3};
use oc_core::vec3::{Isometry3, Plane3, Vec3};

/// The tube's axis height, its bore and outside (355 and 360 mm across),
/// and the room around it. The room was laid out for a 400 mm tube 50 mm
/// under the ceiling and stays where it was.
pub const AXIS_Y: f64 = 5_000.0;
pub const BORE_RADIUS: f64 = 177.5;
pub const TUBE_OUTER_RADIUS: f64 = 180.0;
pub const CEILING_Y: f64 = AXIS_Y + 250.0;
pub const FLOOR_Y: f64 = CEILING_Y - 2_970.0;
pub const TABLE_Y: f64 = FLOOR_Y + 550.0;
/// The gap in the tube the pick-off's light leaves through: the condenser's
/// beam leaves the bore CONDENSER_ANGLE_DEG off square to the tube, so it
/// crosses the wall up to about 70 mm downstream of the pick-off.
pub const TUBE_GAP: f64 = 200.0;

/// The spectrum on the table.
pub const SPECTRUM_LENGTH: f64 = 2_250.0;
pub const SPECTRUM_HEIGHT: f64 = 180.0;
/// The table detector's half length, the spectrum and a margin.
pub const TABLE_HALF_LENGTH: f64 = 0.5 * SPECTRUM_LENGTH + 75.0;
/// Three spectra lie side by side on the table, the sun's in the middle and
/// a lamp's either side of it, this far apart.
pub const BAND_GAP: f64 = 50.0;
/// `LAMPS=1`: the lamps' slits and arms beside the sun's slit, and every
/// part after them sized for the three bands.
fn lamps_on() -> bool {
    std::env::var_os("LAMPS").is_some()
}
/// The halogen lamp's slit: its spectrum has no lines to resolve, and a
/// wider slit lets more of it through.
pub fn continuum_slit_width() -> f64 {
    knob("HAL_SLIT_W", 20.0)
}
/// From Fraunhofer's L (382.0 nm), K and H (393.4, 396.8) to B (686.7). The
/// second order of the bluest light passed lands at 750 nm, past the red end.
pub const BLUE_UM: f64 = 0.375;
pub const RED_UM: f64 = 0.690;

/// Half-angle of the light in the tube: the sun's disc through the
/// Galilean's magnification of 2.97.
pub const TUBE_HALF_ANGLE: f64 = 0.8 * std::f64::consts::PI / 180.0;
/// The collimator's focal length, and the beam it makes: the projection
/// lens takes a cone this beam fills, and the slit is lit to fill it.
pub const F_COLL: f64 = 1_800.0;
pub const BEAM: f64 = 215.0;
/// The beam's width along the dispersion, `BEAM_ALONG` (default `BEAM`):
/// wider takes more light through a narrower slit (the tube's bore already
/// fills the slit's height, so the height can take no more), a longer
/// grating and a faster camera along the spectrum.
pub fn beam_along() -> f64 {
    knob("BEAM_ALONG", BEAM)
}
/// Slit width across the dispersion, as the light sees it: 0.22 nm.
/// `SLIT_W` overrides it: narrower darkens the lines and dims the table in
/// proportion.
pub fn slit_width() -> f64 {
    knob("SLIT_W", 1.1)
}
/// The condenser that lights the slit at the collimator's f-number from the
/// tube's f/36: a strip of the tube's light, picked off in the gap, is imaged
/// onto the slit this much smaller, the tube's cone growing as much. The
/// pick-off focuses the tube's directions onto the condenser mirror
/// `CONDENSER_F` away; that mirror images the pick-off onto the slit.
const CONDENSER_F: f64 = 2_500.0;
/// The condenser's beam from the pick-off to its mirror runs this far off
/// straight out of the tube, downstream: clear of the projection path's
/// folds, which rise beside the grating, and of the lens's leg and F5, which
/// reach back toward the tube.
const CONDENSER_ANGLE_DEG: f64 = 20.0;
/// The condenser mirror sends the light back to the slit only this far off
/// the way it came, so it works near its axis: at a 90 degree fold its field
/// aberrations sent the slit's ends' light past the 1.1 mm slit.
const C1_FOLD_DEG: f64 = 6.0;
/// The pick-off's strip is larger than the slit's image needs by this much,
/// so the slit, not the pick-off, sets what passes.
const PICKOFF_OVERFILL: f64 = 1.05;
/// The field Fresnels' groove pitch while the layout is fitted, mm.
const FIT_PITCH: f64 = 2.0;
/// Every mirror's reflectance: a broadband dielectric HR coat, 99% over
/// 375-690 nm (protected silver gives 97-98% and falls in the blue).
const MIRROR_R: f64 = 0.99;
/// The UV/IR cut filter's transmittance in its band, a hard-coated
/// interference filter.
const FILTER_T: f64 = 0.95;
/// The light's path from the slit to F1, the periscope's drop included:
/// M1 then sits 620 mm past F1, which puts the grating, and M3 and F4 above
/// it, clear of the periscope and the condenser.
const STRIP_TO_F1: f64 = 1_180.0;
const FILTER_AT: f64 = 30.0;
/// The return beam's tilt off the collimator: the 215 mm beam passes F1
/// with 15 mm to spare. So far off axis a sphere would add 90 mm of
/// astigmatism, so M1 is an off-axis paraboloid.
const GAMMA_DEG: f64 = 18.0;
/// F2 far enough from the grating that it sits outside the diffracted fan.
const F2_TO_GRATING: f64 = 750.0;
/// The zero-order dump's distance from the grating, along the zero order.
const ZERO_DUMP_AT: f64 = 600.0;
pub const LINES_PER_MM: f64 = 1_800.0;
/// The arriving light's angle off the grating's normal.
const ALPHA_DEG: f64 = 50.0;
/// The camera mirror's lean, which drops its return under the fan. The
/// spectrum bows 0.31 mm per degree of it, so it is as small as the clearance
/// under the grating allows.
const CAMERA_TILT_DEG: f64 = 8.0;
/// The grating's distance from the camera mirror, as a fraction of the
/// mirror's radius, where the spectrum focuses on a plane.
const GRATING_AT: f64 = 0.46;
/// The fold's distance from the camera mirror, as a fraction of its focal
/// length, and its convex radius, as a multiple of the camera's radius. The
/// weak curvature takes out the bow the camera's lean puts in the spectrum
/// (a numerical search over the three).
const FOLD_AT: f64 = 0.8;
const FOLD_RADIUS_RATIO: f64 = 4.05;

/// The turn of everything past F1 about the vertical through `pivot`:
/// `TURN_DEG` degrees, right-handed about +Y. At 90 the spectrum runs across
/// the tube (-Z) instead of along it, the camera's long mirror with it, and
/// the camera's arm runs along the tube.
fn downstream_turn(pivot: Vec3) -> oc_core::vec3::Isometry3 {
    let r = oc_core::vec3::Isometry3::from_axis_angle(Vec3::Y, knob("TURN_DEG", 0.0).to_radians()).expect("a turn");
    oc_core::vec3::Isometry3::from_pos(pivot).compose(r.compose(oc_core::vec3::Isometry3::from_pos(-pivot)))
}

/// The table's axes in the world: along the spectrum (the unturned `X`) and
/// across the bands (the unturned `Z`; the table detector's `v` runs the
/// other way).
pub fn table_axes() -> (Vec3, Vec3) {
    let r = downstream_turn(Vec3::ZERO);
    (r.rotate(Vec3::X), r.rotate(Vec3::Z))
}

/// The table's normal toward the optics: up, or down with `TABLE_UP`.
pub fn table_normal() -> Vec3 {
    if knob("TABLE_UP", 0.0) > 0.5 {
        -Vec3::Y
    } else {
        Vec3::Y
    }
}

/// Every optic's reach toward the table: its nearest and farthest points'
/// distance from the table plane, nearest first. The detectors and the
/// table's own probes are left out.
pub fn from_table(s: &Scene3, table_centre: Vec3) -> Vec<(String, f64, f64)> {
    use oc_core::geom3::Intersect3;
    let n = table_normal();
    let mut out = Vec::new();
    let mut add = |label: &str, bb: oc_core::vec3::Aabb3| {
        let mut near = f64::INFINITY;
        let mut far = f64::NEG_INFINITY;
        for cx in [bb.min.x, bb.max.x] {
            for cy in [bb.min.y, bb.max.y] {
                for cz in [bb.min.z, bb.max.z] {
                    let d = (Vec3::new(cx, cy, cz) - table_centre).dot(n);
                    near = near.min(d);
                    far = far.max(d);
                }
            }
        }
        out.push((label.to_string(), near, far));
    };
    for o in &s.surfaces {
        add(&o.common.label, o.geometry.aabb_world(&o.common.transform));
    }
    for o in &s.bodies {
        add(&o.common.label, o.geometry.aabb_world(&o.common.transform));
    }
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

/// A layout constant, or its override from the environment, for trying
/// layouts without rebuilding.
fn knob(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
/// A patent lens's focal length and f-number, when a patent lens is in use
/// ([`DESIGNED`] is `None`).
const F_PROJ: f64 = 800.0;
const PROJ_F_NUMBER: f64 = 4.0;
/// Everything after the slit sits this far below the tube's axis, so the
/// folded projection path above the camera stays under the ceiling. A
/// periscope just past the slit takes the light down: its two flat folds
/// reflect in one vertical plane, so the slit's image moves down unturned.
/// (Dropped in the condenser, the pick-off strip would lie tilted to the
/// axis the condenser mirror images it along, its ends out of focus on the
/// slit; dropped at the collimator, the folds out of one plane would turn
/// the lines.)
const CAMERA_DROP: f64 = 694.0;
/// The periscope's upper fold, this far past the slit.
const PERISCOPE_AT: f64 = 120.0;
/// The projection path is folded flat under the ceiling. The field lens, a
/// pair of Fresnels, sits this far before the intermediate spectrum (clear
/// of M3 below it, its grooves out of focus); fold F4 this far past it turns
/// the light horizontal into the projection lens; fold F5 this far past the
/// lens's last vertex sends it down to the table. The field lens has to
/// refract: M2's lean tilts the spectrum's focus about 7 degrees off square
/// to its chief rays, and only a lens turns the rays square to the focus
/// without turning the focus too. A mirror keeps the angle, and the
/// projection's magnification of 6 would tilt the table's image 36 degrees.
const FIELD_BEFORE: f64 = 15.0;
const F4_AFTER: f64 = 70.0;
const F5_GAP: f64 = 160.0;

/// The projection lens: Schneider's compact wide-angle objective, example 1
/// of Kurt Behrens's patent (US 4,773,745, 1988), f = 100, back focus 81.9,
/// "image angle is 70 degrees for an aperture ratio of 1:5.6" with
/// "comparatively small lens diameters": the Super-Symmar HM's form, eight
/// elements in six groups. Radii and gaps from the object side (the
/// table's), each gap followed by the glass after it (`None` for air), with
/// nd and Vd as the patent gives them. The patent draws no diaphragm; it
/// sits in the middle of the seventh gap, before the thick doublet, where
/// the field traces best (in the fifth it traces four times worse at 25
/// degrees). Optiland reproduces f 100.003 and back focus 81.921.
const SCH_R: [f64; 14] = [
    39.152, 20.221, 29.616, 87.298, 29.998, 48.217, 55.911, 321.496, -19.009, -27.981, -117.436, -44.848, -28.475, 131.465,
];
const SCH_D: [f64; 13] = [2.517, 3.510, 6.623, 4.967, 7.000, 1.987, 3.312, 19.312, 1.987, 0.338, 5.841, 2.583, 2.517];
const SCH_GLASS: [Option<(f64, f64)>; 13] = [
    Some((1.464_5, 65.77)),
    None,
    Some((1.755_0, 52.30)),
    Some((1.625_4, 35.56)),
    None,
    Some((1.772_5, 49.60)),
    None,
    Some((1.620_4, 60.33)),
    Some((1.666_8, 33.01)),
    None,
    Some((1.620_4, 60.33)),
    None,
    Some((1.478_7, 58.70)),
];
/// The gap, counted from 0, whose middle holds the diaphragm.
const SCH_STOP_GAP: usize = 6;
/// Clear radii at f = 100, object side first: what f/5.6 pencils fill out
/// to 28 degrees off the axis (traced in Optiland), 0.5 over. The table's
/// corners are about 25 degrees off.
const SCH_SEMI: [f64; 14] = [19.45, 16.5, 16.41, 15.05, 12.29, 9.76, 9.21, 9.97, 14.3, 15.53, 16.15, 16.75, 16.76, 18.75];
/// Each glass's Sellmeier curve (formula 2, `C` in µm²), fitted within
/// 2.3e-5 over 350 to 800 nm to Optiland's Buchdahl model at the patent's nd
/// and Vd: the patent gives no dispersion beyond Vd, and a two-term Cauchy
/// curve drifts in the violet.
const SCH_SELLMEIER: [[f64; 6]; 8] = [
    [1.112_568_620, 0.008_119_604, 0.005_799_101, 0.072_998_556, 0.765_134_174, 134.422_689],
    [1.389_686_040, 0.007_763_475, 0.624_949_552, 0.019_592_310, 2.122_945_915, 174.741_524],
    [1.068_796_689, 0.004_753_253, 0.502_031_168, 0.036_599_928, 1.031_441_273, 105.732_051],
    [1.414_744_718, 0.007_380_413, 0.655_725_812, 0.021_991_907, 2.292_667_872, 186.279_060],
    [1.202_986_086, 0.006_467_460, 0.380_683_360, 0.019_222_003, 1.153_373_141, 117.525_189],
    [1.207_051_406, 0.005_678_981, 0.488_347_844, 0.041_356_692, 1.184_930_821, 106.578_802],
    [1.202_986_086, 0.006_467_460, 0.380_683_360, 0.019_222_003, 1.153_373_141, 117.525_189],
    [1.134_844_269, 0.008_784_441, 0.020_504_242, 0.050_821_112, 1.102_596_576, 193.069_354],
];

/// The Super-Angulon, the example of Klemt and Macher's patent for
/// Schneider (US 2,897,725, 1959), f = 100 at f/4, back focus 36.8, "for
/// high-quality image projection upon a relatively wide area": at f/4 it
/// holds 55 µm rms from 0 to 28 degrees (Optiland), where the compact
/// wide-angle above falls apart past f/5.6. Radii and gaps from the object
/// side, the diaphragm in the middle of the sixth gap.
const SA_R: [f64; 13] = [
    222.31, 48.08, 45.07, 62.18, 29.81, -175.16, 550.46, 27.95, -27.30, -74.58, -36.37, -46.76, -136.37,
];
const SA_D: [f64; 12] = [4.47, 13.91, 35.70, 6.91, 39.66, 7.90, 2.57, 19.98, 28.55, 31.23, 16.20, 4.97];
const SA_GLASS: [Option<(f64, f64)>; 12] = [
    Some((1.658_30, 57.49)),
    Some((1.464_50, 65.79)),
    None,
    Some((1.589_04, 53.16)),
    Some((1.607_29, 49.25)),
    None,
    Some((1.529_44, 51.80)),
    Some((1.617_72, 49.78)),
    Some((1.784_70, 26.10)),
    None,
    Some((1.533_75, 55.37)),
    Some((1.658_44, 50.84)),
];
/// Each glass's Sellmeier curve, fitted as for the lens above.
const SA_SELLMEIER: [[f64; 6]; 9] = [
    [1.249_872_990, 0.007_156_728, 0.451_536_737, 0.018_531_460, 1.372_186_834, 134.230_637],
    [1.114_461_922, 0.008_261_072, 0.003_744_489, 0.080_436_247, 0.749_401_353, 135.373_759],
    [1.424_582_167, 0.009_703_657, 0.054_619_919, 0.040_554_537, 1.070_772_048, 140.474_619],
    [1.174_336_185, 0.007_112_342, 0.357_763_924, 0.026_184_810, 1.074_044_796, 132.833_054],
    [1.164_227_053, 0.008_473_920, 0.134_288_745, 0.031_729_462, 0.916_508_668, 142.343_085],
    [1.205_938_104, 0.007_209_407, 0.359_160_244, 0.025_854_182, 1.060_826_787, 127.025_150],
    [1.498_310_415, 0.007_099_336, 0.559_498_097, 0.052_967_659, 1.646_791_027, 102.908_098],
    [1.292_231_910, 0.009_634_176, 0.021_573_733, 0.050_754_546, 0.970_514_316, 149.676_182],
    [1.235_108_971, 0.006_903_327, 0.459_896_086, 0.023_422_607, 1.196_886_923, 126.578_012],
];
/// Clear radii at f = 100: what f/4 pencils fill out to 28 degrees (traced
/// in Optiland), 0.5 over.
const SA_SEMI: [f64; 13] = [40.53, 34.74, 31.11, 29.02, 25.82, 19.94, 16.79, 17.15, 17.76, 23.39, 25.23, 30.38, 34.63];

/// A projection lens's prescription.
struct Prescription {
    name: &'static str,
    r: &'static [f64],
    d: &'static [f64],
    glass: &'static [Option<(f64, f64)>],
    sellmeier: &'static [[f64; 6]],
    /// The gap, counted from 0, whose middle holds the diaphragm.
    stop_gap: usize,
    semi: &'static [f64],
}
#[allow(dead_code)]
const SCHNEIDER_WIDE: Prescription = Prescription {
    name: "Schneider compact wide-angle (US 4,773,745)",
    r: &SCH_R,
    d: &SCH_D,
    glass: &SCH_GLASS,
    sellmeier: &SCH_SELLMEIER,
    stop_gap: SCH_STOP_GAP,
    semi: &SCH_SEMI,
};
const SUPER_ANGULON: Prescription = Prescription {
    name: "Super-Angulon (US 2,897,725)",
    r: &SA_R,
    d: &SA_D,
    glass: &SA_GLASS,
    sellmeier: &SA_SELLMEIER,
    stop_gap: 5,
    semi: &SA_SEMI,
};
/// A patent lens, used when [`DESIGNED`] is `None`.
const LENS: Prescription = SUPER_ANGULON;

/// One surface of a lens designed for this projector, in the light's order:
/// radius (`None` for the diaphragm), the gap after it, the clear radius, the
/// glass after it (an index into [`TRIPLET_SELLMEIER`]), the conic, and the
/// r⁴ and r⁶ terms.
#[derive(Clone, Copy)]
struct Row {
    r: Option<f64>,
    d: f64,
    semi: f64,
    glass: Option<usize>,
    k: f64,
    a: [f64; 2],
}
/// Schott's N-SK16 and N-F2 (B1, C1, B2, C2, B3, C3).
const TRIPLET_SELLMEIER: [[f64; 6]; 2] = [
    [1.343_177_74, 0.007_046_873_39, 0.241_144_399, 0.022_900_5, 0.994_317_969, 92.750_852_6],
    [1.397_570_37, 0.009_959_061_43, 0.159_201_403, 0.054_693_175_2, 1.268_654_3, 119.248_346],
];
/// The slab triplet: a Cooke triplet with every face an even asphere,
/// designed in Optiland for this projector (f 832 mm). It takes object NA
/// 0.093 along the spectrum and 0.64 of that across it from the
/// intermediate spectrum 1060 mm ahead, and puts it on the table 2480 mm
/// behind at x2.17. Each field point is one wavelength (375 nm at one end,
/// 690 at the other), so it carries no colour correction. On the table it
/// blurs 0.24 mm rms across the lines (median; 0.45 worst) and 0.44 along
/// them, with 0.11% distortion. Everything stays within 180 mm of the
/// diaphragm, so a 175 mm tall slab of each element carries the light;
/// round, the elements are 330 mm across. Clear radii: traced over the
/// whole field, 3 mm over.
const TRIPLET: [Row; 7] = [
    Row { r: Some(381.557_840), d: 45.368_834, semi: 158.3, glass: Some(0), k: 1.066_569, a: [-2.563_213_1e-9, -4.923_819_7e-15] },
    Row { r: Some(-1_584.451_691), d: 85.790_350, semi: 156.1, glass: None, k: -1.770_487, a: [2.187_320_5e-9, -1.119_037_8e-14] },
    Row { r: Some(-370.943_278), d: 10.000_380, semi: 96.8, glass: Some(1), k: -0.327_652, a: [1.264_077_8e-9, -6.047_845_8e-14] },
    Row { r: Some(431.676_627), d: 1.012_416, semi: 94.1, glass: None, k: -1.607_813, a: [-1.824_182_7e-9, 5.589_919_0e-14] },
    Row { r: None, d: 82.172_209, semi: 91.0, glass: None, k: 0.0, a: [0.0, 0.0] },
    Row { r: Some(3_365.500_561), d: 49.093_601, semi: 162.6, glass: Some(0), k: -1.432_928, a: [-1.809_109_5e-9, 1.264_370_0e-14] },
    Row { r: Some(-345.022_059), d: 0.0, semi: 165.1, glass: None, k: 0.320_070, a: [1.146_369_5e-9, 4.644_928_3e-16] },
];
/// The slab triplet redesigned for the long camera (`LENS_V1`): the same
/// Cooke form, every face an even asphere, reoptimized in Optiland at NA
/// 0.108 along the spectrum (0.7 of it across), 16% more than [`TRIPLET`],
/// so the beams at the spectrum's ends, which cross the diaphragm up to
/// 30 mm off its centre, still pass. f 860 mm, x2.19 over 2520 mm; on the
/// table 0.05 to 0.5 mm rms along the spectrum at every field. Its slab is
/// 110 mm half-height (use `LENS_SLAB=113`).
const TRIPLET_V1: [Row; 7] = [
    Row { r: Some(356.431_499), d: 94.228_922, semi: 180.0, glass: Some(0), k: 0.327_563, a: [-5.276_148_2e-10, 2.845_330_7e-14] },
    Row { r: Some(-1_192.236_568), d: 65.945_978, semi: 164.4, glass: None, k: -4.802_456, a: [5.461_934_5e-9, -4.629_965_9e-14] },
    Row { r: Some(-421.942_796), d: 10.000_000, semi: 115.3, glass: Some(1), k: 0.138_729, a: [6.172_723_2e-9, -6.858_710_5e-14] },
    Row { r: Some(391.141_405), d: 2.002_324, semi: 110.0, glass: None, k: -5.0, a: [4.687_642_5e-9, 6.858_710_5e-14] },
    Row { r: None, d: 106.375_341, semi: 105.7, glass: None, k: 0.0, a: [0.0, 0.0] },
    Row { r: Some(2_006.066_760), d: 55.116_865, semi: 193.8, glass: Some(0), k: -1.979_480, a: [-2.441_677_5e-9, 1.416_598_3e-14] },
    Row { r: Some(-460.445_966), d: 0.0, semi: 196.0, glass: None, k: 0.385_558, a: [8.081_778_7e-10, -1.366_713_4e-14] },
];

/// `LENS_FILE=<json>`: a projection lens from a prescription file instead of
/// the built-in ones: `{"rows": [{"r": radius or null for the diaphragm,
/// "t": gap after, "semi": clear radius, "glass": "N-SK16" | "N-F2" | "air",
/// "k": conic, "a": [r^4, r^6, ...]}, ...]}`, the intermediate spectrum's side
/// first. Rows: (radius, gap, clear radius, glass index, conic, terms).
type LensRow = (Option<f64>, f64, f64, Option<usize>, f64, Vec<f64>);

fn file_lens() -> Option<Vec<LensRow>> {
    let path = std::env::var("LENS_FILE").ok()?;
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("read LENS_FILE")).expect("LENS_FILE is JSON");
    let rows = v["rows"].as_array().expect("LENS_FILE has rows");
    Some(
        rows.iter()
            .map(|q| {
                let glass = match q["glass"].as_str() {
                    Some("N-SK16") => Some(0),
                    Some("N-F2") => Some(1),
                    _ => None,
                };
                let a: Vec<f64> = q["a"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default();
                (q["r"].as_f64(), q["t"].as_f64().unwrap_or(0.0), q["semi"].as_f64().unwrap_or(100.0), glass, q["k"].as_f64().unwrap_or(0.0), a)
            })
            .collect(),
    )
}

/// The designed lens's rows with `LENS_TWEAK` applied: twelve numbers, a
/// curvature change (in 1e-4 per mm) for each of its six glass faces, then a
/// conic change for each; a refit against the traced system
/// ([`lens_fit`]).
fn designed_rows(rows: &[Row]) -> Vec<Row> {
    let tweak: Vec<f64> = std::env::var("LENS_TWEAK").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let mut out = if knob("LENS_V1", 0.0) > 0.5 { TRIPLET_V1.to_vec() } else { rows.to_vec() };
    if tweak.len() == 12 {
        let faces: Vec<usize> = (0..out.len()).filter(|&i| out[i].r.is_some()).collect();
        for (j, &i) in faces.iter().enumerate().take(6) {
            let c = 1.0 / out[i].r.unwrap() + tweak[j] * 1e-4;
            out[i].r = Some(1.0 / c);
            out[i].k += tweak[6 + j];
        }
    }
    out
}

/// The triplet's glass is cut to a horizontal slab this tall (half), the
/// beam's traced 87.3 mm and 3 over; round, it would reach 165 mm.
const TRIPLET_SLAB: f64 = 90.3;
/// The lens in use: the designed triplet, or `None` for the patent [`LENS`].
const DESIGNED: Option<&[Row]> = Some(&TRIPLET);
pub const PROJ_NAME: &str = match DESIGNED {
    Some(_) => "slab triplet (aspheric, designed in Optiland)",
    None => LENS.name,
};

/// What [`add`] placed, and where the spectrum lands.
pub struct Spectrograph {
    /// The pick-off's centre on the tube's axis, and its strip as the tube's
    /// light sees it: height, width.
    pub pickoff: Vec3,
    pub pickoff_size: (f64, f64),
    /// The strip's run as the tube's light sees it, square to the tube.
    pub pickoff_along: Vec3,
    /// How far the tilted strip reaches upstream of its centre along the
    /// tube: a source that stands in for the tube goes beyond it.
    pub pickoff_depth: f64,
    /// How far the strip's arc bows across, as the tube sees it.
    pub pickoff_bow: f64,
    /// The slit's centre.
    pub slit: Vec3,
    pub slit_height: f64,
    /// Under the projection lens, on the table.
    pub table_centre: Vec3,
    /// The intermediate spectrum's middle, on its level plane; the light
    /// crosses it going up.
    pub intermediate: Vec3,
    /// The table's full detector, and the entrance probe after the filter.
    pub table: ObjectId,
    pub entrance: ObjectId,
    pub design: Design,
    /// The lamps' sources, and each spectrum's edges on the table, in the
    /// table detector's `v`, the sun's first.
    pub lamp_sources: Vec<SourceId>,
    pub bands: Vec<(&'static str, f64, f64)>,
    /// How far each lamp's slit bows across its length, mm.
    pub lamp_slit_bows: Vec<f64>,
    /// Heights of the folded projection path's parts, world `Y`.
    #[allow(dead_code)] // copied into the report's own record, read from there
    pub clearances: Vec<(&'static str, f64)>,
}

/// The first-order numbers the layout is built from.
#[derive(Clone, Copy, Debug)]
pub struct Design {
    pub alpha_deg: f64,
    pub beta_blue_deg: f64,
    pub beta_red_deg: f64,
    pub f_camera: f64,
    pub fold_radius: f64,
    pub f_field: f64,
    /// From the intermediate spectrum to the projection lens's first
    /// principal plane (`object_distance`), its first vertex, and its
    /// entrance pupil as the field lens sees it.
    pub object_distance: f64,
    pub f_proj: f64,
    /// The projection lens's exit pupil radius, and its f-number from it.
    pub exit_radius: f64,
    pub proj_f_number: f64,
    pub vertex_distance: f64,
    pub pupil_distance: f64,
    pub lens_length: f64,
    /// The lens's largest clear radius.
    pub lens_semi: f64,
    /// Its glass's half-height across the bands, the slab it is cut to.
    pub lens_slab: f64,
    /// How far the traced focus moved the lens from the paraxial place,
    /// along the light.
    pub focus_shift: f64,
    /// The field Fresnels' base radius, as a multiple of the thin-lens one,
    /// and their conic: fitted by tracing so the chief rays cross the
    /// projection lens's diaphragm at its centre.
    pub fresnel_scale: f64,
    pub fresnel_conic: f64,
    /// And their centre's shift along the spectrum: PMMA bends the violet
    /// end harder than the red.
    pub fresnel_shift: f64,
    /// And across the bands.
    pub fresnel_shift_v: f64,
    /// Fold F4's curvature, 1/mm (0 flat). A sphere at 45 degrees has 0.35 R
    /// across the bands (the plane it folds in) and 0.71 R along the
    /// spectrum: with the round Fresnels, two powers, for a camera pupil
    /// that lies nearer across the bands than along the spectrum.
    pub f4_curv: f64,
    /// The grooves' pitch, mm: coarser while fitting, where it moves no
    /// chief ray but costs facets to build.
    pub fresnel_pitch: f64,
    /// The camera mirror's conic and its r^4 and r^6 terms, and the convex
    /// fold's r^4 and r^6 terms, each term as its sag in mm at 500 mm from
    /// the vertex: fitted by tracing onto the table (`camera_fit`).
    pub m2_asph: [f64; 3],
    pub m3_asph: [f64; 2],
    /// The field Fresnels' base r^4, r^6, r^8 terms, as sag in mm at 500 mm:
    /// fitted with their radius, conic and shift so the camera's pupil lands
    /// on the projection lens's diaphragm ([`pupil_fit`]).
    pub fresnel_asph: [f64; 3],
    /// How far the condenser mirror swings round the slit's axis, radians:
    /// the collimator's drop turns the slit's image on the table, and the
    /// slit turns back as much, solved by tracing so a line stands square
    /// across the band.
    pub slit_turn: f64,
    /// A probe at the diaphragm, for that fit.
    pub stop_probe: bool,
    /// A table detector of one cell, for scenes that only probe with rays.
    pub probe_table: bool,
    /// The beam's f-number at the intermediate spectrum, and the one the
    /// projection lens takes there.
    pub camera_f_number: f64,
    #[allow(dead_code)] // part of the design record; nothing prints it yet
    pub camera_f_number_along: f64,
    pub lens_takes_f_number: f64,
    pub magnification: f64,
    pub throw: f64,
    pub slit_height: f64,
    pub resolution_nm: f64,
    /// Peak to valley of the intermediate spectrum's best focus about its
    /// plane, 400 to 700 nm, along the axis.
    pub focus_pv: f64,
    /// The dark gap between the sun's slit and a lamp's, at the slit.
    pub slit_gap: f64,
    /// How far off straight down the fold sends the light, degrees.
    pub fold_lean_deg: f64,
}

fn sized_for_lamps() -> bool {
    lamps_on() || knob("SIZE_FOR_LAMPS", 0.0) != 0.0
}

impl Design {
    /// Half the height of the three slits together, at the slit; the sun's
    /// alone while the lamps are off (`SIZE_FOR_LAMPS=1` sizes for all three
    /// anyway, as v1 was).
    pub fn field_half_height(&self) -> f64 {
        if sized_for_lamps() {
            1.5 * self.slit_height + self.slit_gap
        } else {
            0.5 * self.slit_height + 2.0
        }
    }

    /// The three spectra and their gaps, across the table; the sun's alone
    /// while the lamps are off.
    pub fn table_height(&self) -> f64 {
        if sized_for_lamps() {
            3.0 * SPECTRUM_HEIGHT + 2.0 * BAND_GAP
        } else {
            SPECTRUM_HEIGHT + 20.0
        }
    }
}

fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - n * (2.0 * d.dot(n))
}

/// Where the line `p + t d` meets the sphere, the far root from inside or
/// the near one from outside.
fn sphere_hit(p: Vec3, d: Vec3, c: Vec3, r: f64, far: bool) -> Vec3 {
    let oc = p - c;
    let b = d.dot(oc);
    let q = (b * b - (oc.length_squared() - r * r)).sqrt();
    p + d * if far { -b + q } else { -b - q }
}

/// The midpoint of the shortest segment between two lines.
fn closest(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> Vec3 {
    let w = p1 - p2;
    let (a, b, c, d, e) = (d1.dot(d1), d1.dot(d2), d2.dot(d2), d1.dot(w), d2.dot(w));
    let den = a * c - b * b;
    let (s, t) = ((b * e - c * d) / den, (a * e - b * d) / den);
    (p1 + d1 * s + p2 + d2 * t) * 0.5
}

/// The grating's frame, with the fan's centre along `-Z`: its normal, its
/// grating vector (local `X`), its groove direction (local `Y`), and the
/// direction the light arrives in.
struct GratingFrame {
    n: Vec3,
    gx: Vec3,
    gy: Vec3,
    arrive: Vec3,
    d_um: f64,
    beta_c: f64,
}

impl GratingFrame {
    fn new() -> Self {
        let d_um = 1e3 / LINES_PER_MM;
        let a = ALPHA_DEG.to_radians();
        let beta = |l: f64| (l / d_um - a.sin()).asin();
        let beta_c = 0.5 * (beta(BLUE_UM) + beta(RED_UM));
        // A reflected order leaves at sin b gx + cos b n; the fan's centre is
        // -Z with e1 = -Z, e2 = +X, and F2 sits on the +X side.
        let (e1, e2) = (-Vec3::Z, Vec3::X);
        let n = e1 * beta_c.cos() - e2 * beta_c.sin();
        let gx = e1 * beta_c.sin() + e2 * beta_c.cos();
        let arrive = gx * (-a.sin()) - n * a.cos();
        Self {
            n,
            gx,
            gy: n.cross(gx),
            arrive,
            d_um,
            beta_c,
        }
    }

    /// The first order of `lambda_um` for light arriving along `d`.
    fn diffract(&self, d: Vec3, lambda_um: f64) -> Vec3 {
        let local = Vec3::new(d.dot(self.gx), d.dot(self.gy), d.dot(self.n));
        let o = oc_core::grating::diffract_local(local, 1.0 / self.d_um, lambda_um, false)
            .expect("a propagating first order");
        self.gx * o.x + self.gy * o.y + self.n * o.z
    }
}

/// The camera side for a camera mirror of focal length `f`, relative to the
/// grating's centre.
struct Camera {
    r: f64,
    m2: Vec3,
    m2_normal: Vec3,
    fold: Vec3,
    fold_normal: Vec3,
    fold_r: f64,
    /// The intermediate spectrum's plane is horizontal at this height.
    image_y: f64,
    /// Where 400 and 700 nm from the slit's centre cross that plane.
    blue: Vec3,
    red: Vec3,
    focus_pv: f64,
    /// The image's height per radian of slit angle.
    height_per_rad: f64,
    /// How far before the image plane, along the light, the chief rays after
    /// the fold diverge from (negative: they converge past it), and the point
    /// itself.
    pupil_above: f64,
    pupil: Vec3,
    /// The light's direction after the fold, a little off straight up.
    out: Vec3,
    /// The sine of the focus's tilt along the slit's height.
    height_tilt: f64,
}

impl Camera {
    /// The camera whose fold sends the light down tilted just enough that
    /// the spectrum's focus lies level along the slit's height as well as
    /// along the spectrum. M2's lean tilts the focus along the height, about
    /// 19 degrees; tilting the light after the fold by as much turns the
    /// tilted focus level, and the field lens turns the light back into the
    /// projection lens.
    fn new(g: &GratingFrame, f: f64) -> Self {
        let err = |lean: f64| {
            let c = Self::leaned(g, f, lean);
            (c.height_tilt, c)
        };
        let (mut b0, mut b1) = (0.0, 0.1);
        let (mut e0, mut c) = err(b0);
        let (mut e1, c1) = err(b1);
        c = if e1.abs() < e0.abs() { c1 } else { c };
        for _ in 0..12 {
            if (e1 - e0).abs() < 1e-15 {
                break;
            }
            let b2 = b1 - e1 * (b1 - b0) / (e1 - e0);
            (b0, e0) = (b1, e1);
            b1 = b2;
            let (e, cc) = err(b1);
            e1 = e;
            c = cc;
            if e1.abs() < 1e-9 {
                break;
            }
        }
        c
    }

    fn leaned(g: &GratingFrame, f: f64, lean: f64) -> Self {
        let r = 2.0 * f;
        let e = knob("CAMERA_TILT", CAMERA_TILT_DEG).to_radians();
        // M2 leans up: its return passes over the grating, and the light
        // leaves M3 upward, toward the ceiling.
        let ret = Vec3::new(0.0, (2.0 * e).sin(), (2.0 * e).cos());
        let m2_normal = (ret + Vec3::Z).normalize();
        let m2 = -Vec3::Z * (knob("GRATING_AT", GRATING_AT) * r);
        let c2 = m2 + m2_normal * r;
        let fold = m2 + ret * (knob("FOLD_AT", FOLD_AT) * f);
        let out = Vec3::new(0.0, lean.cos(), lean.sin());
        let fold_normal = (out - ret).normalize();
        let fold_r = knob("FOLD_RATIO", FOLD_RADIUS_RATIO) * r;
        let c3 = fold - fold_normal * fold_r;
        let run = |p: Vec3, d: Vec3| {
            let q = sphere_hit(p, d, c2, r, true);
            let d = reflect(d, (c2 - q) / r);
            let q2 = sphere_hit(q, d, c3, fold_r, false);
            (q2, reflect(d, (q2 - c3) / fold_r))
        };
        let lambdas: Vec<f64> = (0..13).map(|i| BLUE_UM + (RED_UM - BLUE_UM) * i as f64 / 12.0).collect();
        let foci: Vec<Vec3> = lambdas
            .iter()
            .map(|&l| {
                let d = g.diffract(g.arrive, l);
                let beta = d.dot(g.gx).asin();
                let h = 0.35 * beam_along() * beta.cos() / ALPHA_DEG.to_radians().cos();
                let across = d.cross(Vec3::Y).normalize() * h;
                let (p1, d1) = run(across, d);
                let (p2, d2) = run(-across, d);
                closest(p1, d1, p2, d2)
            })
            .collect();
        let image_y = foci.iter().map(|p| p.y).sum::<f64>() / foci.len() as f64;
        let (lo, hi) = foci.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p.y), hi.max(p.y))
        });
        let on_plane = |l: f64, arrive: Vec3| {
            let (p, d) = run(Vec3::ZERO, g.diffract(arrive, l));
            (p + d * ((image_y - p.y) / d.y), p, d)
        };
        let (blue, pb, db) = on_plane(BLUE_UM, g.arrive);
        let (red, pr, dr) = on_plane(RED_UM, g.arrive);
        let lambda_c = g.d_um * (g.beta_c.sin() + ALPHA_DEG.to_radians().sin());
        let tilt = 1e-3;
        let (c0, _, _) = on_plane(lambda_c, g.arrive);
        let (c1, _, _) = on_plane(lambda_c, (g.arrive + Vec3::Y * tilt).normalize());
        let pupil = closest(pb, db, pr, dr);
        // The focus of the middle wavelength from a slit point off the axis,
        // above and below: the line between them should lie level.
        let focus_at = |phi: f64| {
            let arrive = (g.arrive + Vec3::Y * phi).normalize();
            let d = g.diffract(arrive, lambda_c);
            let h = 0.35 * beam_along();
            let across = d.cross(Vec3::Y).normalize() * h;
            let (p1, d1) = run(across, d);
            let (p2, d2) = run(-across, d);
            closest(p1, d1, p2, d2)
        };
        let v = focus_at(0.04) - focus_at(-0.04);
        Self {
            r,
            m2,
            m2_normal,
            fold,
            fold_normal,
            fold_r,
            image_y,
            blue,
            red,
            focus_pv: hi - lo,
            height_per_rad: (c1 - c0).length() / tilt,
            pupil_above: image_y - pupil.y,
            pupil,
            out,
            height_tilt: v.y / v.length(),
        }
    }

    fn length(&self) -> f64 {
        (self.red - self.blue).length()
    }
}

/// The projection lens turned round, the intermediate spectrum's side first,
/// scaled to `F_PROJ`, with its first-order numbers.
pub struct ProjLens {
    /// Surfaces in the light's order: radius, the gap after, clear radius and
    /// the medium after (the diaphragm is the plane one).
    surfaces: Vec<(Option<f64>, f64, f64, oc_core::seq::SeqMaterial)>,
    stop: usize,
    pub f: f64,
    /// From the first vertex, along the light: the principal planes, the
    /// entrance pupil seen from the intermediate spectrum, and the last
    /// vertex.
    h1: f64,
    h2: f64,
    pupil: f64,
    length: f64,
    /// The diaphragm's radius for `PROJ_F_NUMBER` as the table side sees it,
    /// and the entrance pupil's radius, the diaphragm seen from the
    /// intermediate spectrum.
    stop_radius: f64,
    pupil_radius: f64,
    /// The exit pupil's radius, the diaphragm seen from the table.
    pub exit_radius: f64,
    /// The glass's half-height across the bands: the slab it is cut to.
    pub slab: f64,
    /// Each surface's conic and r⁴, r⁶ terms, in step with `surfaces`.
    asph: Vec<(f64, Vec<f64>)>,
}

impl ProjLens {
    pub fn new() -> Self {
        use oc_core::material::DispersionFormula;
        use oc_core::seq::SeqMaterial;
        let glass = |k: usize| {
            let c = LENS.sellmeier[k];
            SeqMaterial::Glass {
                name: format!("{PROJ_NAME}, glass {}", k + 1),
                reference: None,
                formula: Some(DispersionFormula::Sellmeier2 {
                    coefficients: vec![0.0, c[0], c[1], c[2], c[3], c[4], c[5]],
                }),
                sellmeier: None,
            }
        };
        // The glass after each patent surface, numbered in the patent's order.
        let mut after = Vec::new();
        let mut k = 0;
        for g in LENS.glass {
            after.push(g.map(|_| {
                k += 1;
                glass(k - 1)
            }));
        }
        // Scaled once to f = 100, then to F_PROJ from the traced focal length.
        let build = |scale: f64| {
            let mut out = Vec::new();
            // Reversed: surface i turns round (R -> -R); the medium after it
            // is the one the patent has before it, and the gap after it the
            // patent's gap before it.
            for i in (0..LENS.r.len()).rev() {
                let medium = if i == 0 { SeqMaterial::Air } else { after[i - 1].clone().unwrap_or(SeqMaterial::Air) };
                let gap = if i == 0 { 0.0 } else { LENS.d[i - 1] * scale };
                if i == LENS.stop_gap + 1 {
                    // This surface is followed by half the diaphragm's gap,
                    // then the diaphragm, then the other half.
                    out.push((Some(-LENS.r[i] * scale), 0.5 * gap, LENS.semi[i] * scale, medium));
                    out.push((None, 0.5 * gap, 0.0, SeqMaterial::Air));
                    continue;
                }
                out.push((Some(-LENS.r[i] * scale), gap, LENS.semi[i] * scale, medium));
            }
            out
        };
        let tri = |k: usize| {
            let c = TRIPLET_SELLMEIER[k];
            SeqMaterial::Glass {
                name: ["N-SK16", "N-F2"][k].into(),
                reference: None,
                formula: Some(DispersionFormula::Sellmeier2 {
                    coefficients: vec![0.0, c[0], c[1], c[2], c[3], c[4], c[5]],
                }),
                sellmeier: None,
            }
        };
        let from_file = file_lens();
        let (mut surfaces, asph) = match (&from_file, DESIGNED) {
            (Some(rows), _) => (
                rows.iter().map(|q| (q.0, q.1, q.2, q.3.map_or(SeqMaterial::Air, tri))).collect::<Vec<_>>(),
                rows.iter().map(|q| (q.4, q.5.clone())).collect::<Vec<_>>(),
            ),
            (None, Some(rows)) => {
                let rows = designed_rows(rows);
                (
                    rows.iter().map(|q| (q.r, q.d, q.semi * knob("SEMI_SCALE", 1.0), q.glass.map_or(SeqMaterial::Air, tri))).collect::<Vec<_>>(),
                    rows.iter().map(|q| (q.k, q.a.to_vec())).collect::<Vec<_>>(),
                )
            }
            (None, None) => {
                let surfaces = build(1.0);
                let stop = surfaces.iter().position(|s| s.0.is_none()).unwrap();
                let f100 = Self::paraxial(&surfaces, stop).0;
                let surfaces = build(F_PROJ / f100);
                let n = surfaces.len();
                (surfaces, vec![(0.0, Vec::new()); n])
            }
        };
        let stop = surfaces.iter().position(|s| s.0.is_none()).unwrap();
        let (f, h1, h2, pupil, length, d_stop) = Self::paraxial(&surfaces, stop);
        let stop_radius = match (&from_file, DESIGNED) {
            (Some(rows), _) => knob("STOP_R", rows[stop].2),
            (None, Some(rows)) => knob("STOP_R", designed_rows(rows)[stop].semi),
            (None, None) => d_stop * 0.5 * f / PROJ_F_NUMBER,
        };
        // The entrance pupil: a ray parallel to the axis from the
        // intermediate spectrum's side meets the diaphragm at `a` times its
        // height.
        let mut a = 1.0;
        let mut n0 = 1.0;
        let (mut y, mut nu) = (1.0, 0.0);
        for (i, (r, t, _, m)) in surfaces.iter().enumerate() {
            if i == stop {
                a = y;
                break;
            }
            let n1 = match m {
                oc_core::seq::SeqMaterial::Glass { formula: Some(fm), .. } => fm.index(0.587_56),
                _ => 1.0,
            };
            nu -= (n1 - n0) * r.map_or(0.0, |r| 1.0 / r) * y;
            y += t * nu / n1;
            n0 = n1;
        }
        surfaces[stop].2 = stop_radius;
        let slab = match (&from_file, DESIGNED) {
            (Some(_), _) | (None, Some(_)) => knob("LENS_SLAB", TRIPLET_SLAB),
            (None, None) => surfaces.iter().map(|q| q.2).fold(0.0, f64::max),
        };
        Self {
            surfaces,
            stop,
            f,
            h1,
            h2,
            pupil,
            length,
            stop_radius,
            pupil_radius: stop_radius / a,
            exit_radius: stop_radius / d_stop,
            slab,
            asph,
        }
    }

    /// Paraxial numbers at the d line: the focal length, the principal
    /// planes, the entrance pupil from the first vertex, the last vertex, and
    /// the `D` term of the diaphragm-to-last-vertex matrix (the table side's
    /// pupil is the diaphragm divided by it).
    fn paraxial(surfaces: &[(Option<f64>, f64, f64, oc_core::seq::SeqMaterial)], stop: usize) -> (f64, f64, f64, f64, f64, f64) {
        use oc_core::seq::SeqMaterial;
        type M = [[f64; 2]; 2];
        let mul = |a: M, b: M| -> M {
            [
                [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
            ]
        };
        let index = |m: &SeqMaterial| match m {
            SeqMaterial::Glass {
                formula: Some(f), ..
            } => f.index(0.587_56),
            _ => 1.0,
        };
        let id: M = [[1.0, 0.0], [0.0, 1.0]];
        // (y, n u) through surface i's refraction, then the gap after it.
        let (mut to_stop, mut from_stop, mut all) = (id, id, id);
        let mut n0 = 1.0;
        let mut length = 0.0;
        let last = surfaces.len() - 1;
        for (i, (r, t, _, m)) in surfaces.iter().enumerate() {
            let n1 = index(m);
            let c = r.map_or(0.0, |r| 1.0 / r);
            let refr: M = [[1.0, 0.0], [-(n1 - n0) * c, 1.0]];
            let gap = if i == last { 0.0 } else { *t };
            let tr: M = [[1.0, gap / n1], [0.0, 1.0]];
            let step = mul(tr, refr);
            all = mul(step, all);
            if i < stop {
                to_stop = mul(step, to_stop);
            } else {
                // The diaphragm's own refraction is nothing.
                from_stop = mul(step, from_stop);
            }
            if i != last {
                length += t;
            }
            n0 = n1;
        }
        let [[a, _], [c, d]] = all;
        let f = -1.0 / c;
        let h1 = (d - 1.0) / c;
        let h2 = length + (1.0 - a) / c;
        // A ray through the diaphragm's centre crosses the axis ahead of the
        // lens at the pupil: rays (1, 0) and (0, 1) at the first vertex meet
        // the diaphragm at heights A and B.
        let pupil = to_stop[0][1] / to_stop[0][0];
        (f, h1, h2, pupil, length, from_stop[1][1])
    }
}

/// Solve the camera's focal length for `SPECTRUM_LENGTH` of spectrum on the
/// table. The throw runs from the projection lens along the horizontal leg
/// to the last fold and down to the table, so it depends on how high the
/// intermediate spectrum sits, which depends on the camera. The field lens,
/// just before the intermediate spectrum, draws it in a little.
fn solve() -> (GratingFrame, Camera, Design) {
    let g = GratingFrame::new();
    let lens = ProjLens::new();
    let (mut f, mut u, mut f_field) = (600.0, 420.0, 400.0);
    let mut cam = Camera::new(&g, f);
    let mut throw = 0.0;
    let mut k = 1.0;
    for _ in 0..40 {
        cam = Camera::new(&g, f);
        k = 1.0 / (1.0 + FIELD_BEFORE / f_field);
        throw = (lens.length - lens.h2 + F5_GAP) + (AXIS_Y - knob("CAMERA_DROP", CAMERA_DROP) + cam.image_y + knob("F4_AFTER", F4_AFTER) - TABLE_Y);
        u = 1.0 / (1.0 / lens.f - 1.0 / throw);
        let (a, b) = (cam.pupil_above - FIELD_BEFORE, FIELD_BEFORE + u - lens.h1 + lens.pupil);
        f_field = 1.0 / (1.0 / a + 1.0 / b);
        f *= SPECTRUM_LENGTH / (throw / u) / (k * cam.length());
    }
    // `F_CAMERA`: the camera held at a focal length a joint camera-and-relay
    // design was made for, whatever lens and heights the loop above sees.
    if let Some(fc) = std::env::var("F_CAMERA").ok().and_then(|v| v.parse::<f64>().ok()) {
        f = fc;
        cam = Camera::new(&g, f);
    }
    // With the camera held, the relay magnifies what the camera gives to
    // the spectrum's length (M3, the field lens and F4 are sized by it).
    let magnification = if std::env::var_os("F_CAMERA").is_some() { SPECTRUM_LENGTH / (k * cam.length()) } else { throw / u };
    // The ideal lens (`IDEAL_PROJ`) takes whatever conjugates that
    // magnification needs, and the field lens follows it.
    if std::env::var_os("F_CAMERA").is_some() && std::env::var_os("IDEAL_PROJ").is_some() {
        u = throw / magnification;
        let (a, b) = (cam.pupil_above - FIELD_BEFORE, FIELD_BEFORE + u - lens.h1 + lens.pupil);
        f_field = 1.0 / (1.0 / a + 1.0 / b);
    }
    if std::env::var_os("RELAY_DIAG").is_some() {
        eprintln!("SOLVEDIAG f_camera {f:.9} slit_height {:.9}", SPECTRUM_HEIGHT / (k * cam.height_per_rad * magnification / F_COLL));
    }
    let a = ALPHA_DEG.to_radians();
    let beta = |l: f64| (l / g.d_um - a.sin()).asin().to_degrees();
    // The slit's height on the table per radian, through the field lens.
    let height = k * cam.height_per_rad * magnification / F_COLL;
    let pupil_distance = u - lens.h1 + lens.pupil;
    let design = Design {
        alpha_deg: ALPHA_DEG,
        beta_blue_deg: beta(BLUE_UM),
        beta_red_deg: beta(RED_UM),
        f_camera: f,
        fold_radius: cam.fold_r,
        f_field,
        object_distance: u,
        f_proj: lens.f,
        exit_radius: lens.exit_radius,
        proj_f_number: 0.5 * lens.f / lens.exit_radius,
        vertex_distance: u - lens.h1,
        pupil_distance,
        lens_length: lens.length,
        lens_semi: lens.surfaces.iter().map(|q| q.2).fold(0.0, f64::max),
        lens_slab: lens.slab,
        focus_shift: 0.0,
        fresnel_scale: 1.0,
        fresnel_conic: -1.0,
        fresnel_shift: 0.0,
        fresnel_shift_v: 0.0,
        f4_curv: 0.0,
        fresnel_pitch: 0.5,
        m2_asph: [0.0; 3],
        m3_asph: [0.0; 2],
        fresnel_asph: [0.0; 3],
        slit_turn: 0.0,
        stop_probe: false,
        probe_table: false,
        camera_f_number: f / BEAM,
        camera_f_number_along: f / beam_along(),
        // The cone the lens's entrance pupil takes from the intermediate
        // spectrum, which the field lens sends into it whole.
        lens_takes_f_number: pupil_distance / (2.0 * lens.pupil_radius),
        magnification,
        throw,
        // `SLIT_HEIGHT` (mm) likewise.
        slit_height: knob("SLIT_HEIGHT", SPECTRUM_HEIGHT / height),
        resolution_nm: slit_width() / F_COLL * g.d_um * 1e3 * a.cos(),
        focus_pv: cam.focus_pv,
        fold_lean_deg: cam.out.z.atan2(cam.out.y).to_degrees(),
        slit_gap: BAND_GAP / height,
    };
    (g, cam, design)
}

/// The rigid frame with local `Z` along `z` and local `Y` as near `y` as a
/// right-handed frame allows.
pub fn frame(pos: Vec3, z: Vec3, y: Vec3) -> Isometry3 {
    let z = z.normalize();
    let y = (y - z * y.dot(z)).normalize();
    let x = y.cross(z);
    let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(x, y, z));
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).expect("a unit frame")
}

fn surface(
    s: &mut Scene3,
    at: Isometry3,
    geometry: SurfaceGeometry3,
    front: SurfaceResponse,
    back: SurfaceResponse,
    label: &str,
) -> ObjectId {
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = at;
    common.label = label.into();
    s.surfaces.push(SurfaceObject3 {
        common,
        geometry,
        response_front: front,
        response_back: back,
    });
    id
}

fn rect(hw: f64, hh: f64) -> SurfaceGeometry3 {
    SurfaceGeometry3::Plane {
        shape: PlanarShape2::Rectangle {
            half_width_mm: hw,
            half_height_mm: hh,
        },
    }
}

/// A sphere of radius `r` (positive concave toward local `+Z`), cut to a
/// `hw` x `hh` rectangle about its vertex.
/// [`sphere_strip`] with a conic and r^4, r^6 terms given as sag in mm at
/// 500 mm from the vertex.
fn asphere_strip(r: f64, k: f64, s4: f64, s6: f64, hw: f64, hh: f64) -> SurfaceGeometry3 {
    let mut g = sphere_strip(r, hw, hh);
    if let SurfaceGeometry3::SurfaceOfRevolution { sag, .. } = &mut g {
        let mut coeffs = [0.0; 8];
        coeffs[0] = s4 / 500f64.powi(4);
        coeffs[1] = s6 / 500f64.powi(6);
        *sag = SagModel::EvenAsphere {
            curvature_per_mm: 1.0 / r,
            conic: k,
            coeffs,
        };
    }
    g
}

/// An [`asphere_strip`] with an r^8 term too, as sag in mm at 500 mm.
fn with_s8(mut g: SurfaceGeometry3, s8: f64) -> SurfaceGeometry3 {
    if let SurfaceGeometry3::SurfaceOfRevolution { sag: SagModel::EvenAsphere { coeffs, .. }, .. } = &mut g {
        coeffs[2] = s8 / 500f64.powi(8);
    }
    g
}

/// A strip's outline grown by `plus` mm on its local +Y side and `minus`
/// on its -Y side (negative trims).
fn lopsided(mut g: SurfaceGeometry3, plus: f64, minus: f64) -> SurfaceGeometry3 {
    if plus == 0.0 && minus == 0.0 {
        return g;
    }
    if let SurfaceGeometry3::SurfaceOfRevolution { radial_aperture, outline, .. } = &mut g {
        if let Some(PlanarShape2::Rectangle { half_width_mm: hw, half_height_mm: hh }) = *outline {
            let (lo, hi) = (-hh - minus, hh + plus);
            *outline = Some(PlanarShape2::Polygon {
                vertices: vec![oc_core::vec::Vec2::new(-hw, lo), oc_core::vec::Vec2::new(hw, lo), oc_core::vec::Vec2::new(hw, hi), oc_core::vec::Vec2::new(-hw, hi)],
            });
            radial_aperture.outer_radius_mm = hw.hypot(lo.abs().max(hi.abs())) + 1.0;
        }
    }
    g
}

fn sphere_strip(r: f64, hw: f64, hh: f64) -> SurfaceGeometry3 {
    SurfaceGeometry3::SurfaceOfRevolution {
        sag: SagModel::Conic {
            curvature_per_mm: 1.0 / r,
            conic: 0.0,
        },
        radial_aperture: RadialAperture {
            inner_radius_mm: 0.0,
            outer_radius_mm: hw.hypot(hh) + 1.0,
        },
        outline: Some(PlanarShape2::Rectangle {
            half_width_mm: hw,
            half_height_mm: hh,
        }),
    }
}

/// A piece of a conic of revolution: vertex `v`, axis `axis` toward its
/// focus, vertex radius `r` and conic `k`, cut to the rectangle `hw` x `hh`
/// about `at` in the plane with normal `n` (`hh` along `along`), seen down
/// the axis. Returns its frame and shape.
fn conic_piece(
    v: Vec3,
    axis: Vec3,
    r: f64,
    k: f64,
    at: Vec3,
    n: Vec3,
    along: Vec3,
    hw: f64,
    hh: f64,
) -> (Isometry3, SurfaceGeometry3) {
    let hint = if axis.cross(along).length() > 0.1 { along } else { n };
    let fr = frame(v, axis, hint);
    let along = (along - n * along.dot(n)).normalize();
    let across = n.cross(along).normalize();
    let mut vertices = Vec::new();
    let mut reach: f64 = 0.0;
    for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let p = fr.inverse_transform_point(at + across * (a * hw) + along * (b * hh));
        vertices.push(oc_core::vec::Vec2::new(p.x, p.y));
        reach = reach.max(p.x.hypot(p.y));
    }
    // Counter-clockwise.
    let area: f64 = (0..4).map(|i| {
        let (p, q) = (vertices[i], vertices[(i + 1) % 4]);
        p.x * q.y - q.x * p.y
    }).sum();
    if area < 0.0 {
        vertices.reverse();
    }
    let geometry = SurfaceGeometry3::SurfaceOfRevolution {
        sag: SagModel::Conic {
            curvature_per_mm: 1.0 / r,
            conic: k,
        },
        radial_aperture: RadialAperture {
            inner_radius_mm: 0.0,
            outer_radius_mm: reach + 1.0,
        },
        outline: Some(PlanarShape2::Polygon { vertices }),
    };
    (fr, geometry)
}

/// The paraboloid with focus `focus` that sends light from it out along
/// `out`, through `at`: its vertex, axis and vertex radius.
fn paraboloid(focus: Vec3, out: Vec3, at: Vec3) -> (Vec3, Vec3, f64) {
    let q = at - focus;
    let fp = 0.5 * (q.length() - q.dot(out));
    (focus - out * fp, out, 2.0 * fp)
}

/// The ellipsoid with foci `f1` and `f2` through `at`: its vertex on
/// `at`'s side, its axis toward the centre, vertex radius and conic.
fn ellipsoid(f1: Vec3, f2: Vec3, at: Vec3) -> (Vec3, Vec3, f64, f64) {
    let a = 0.5 * ((at - f1).length() + (at - f2).length());
    let c = 0.5 * (f2 - f1).length();
    let e = (f2 - f1).normalize();
    let centre = (f1 + f2) * 0.5;
    let side = (at - centre).dot(e).signum();
    let b2 = a * a - c * c;
    (centre + e * (side * a), -e * side, b2 / a, -(c / a).powi(2))
}

/// From the condenser mirror to the slit: the tube's cone, f/36, grows to the
/// collimator's, as the pick-off's strip shrinks onto the slit. The pick-off
/// is CONDENSER_F out from the tube, level with the mirror.
fn condenser_gap() -> f64 {
    CONDENSER_F * TUBE_HALF_ANGLE.tan() / (0.5 * BEAM / F_COLL)
}

/// From the condenser mirror to the pick-off: level, CONDENSER_ANGLE_DEG
/// from straight out of the tube toward downstream (the mirror sits out and
/// along from the pick-off).
fn condenser_w() -> Vec3 {
    let b = CONDENSER_ANGLE_DEG.to_radians();
    Vec3::new(-b.sin(), 0.0, b.cos())
}

/// From the condenser mirror to the slit, unturned: back toward the
/// pick-off, C1_FOLD_DEG aside, on the side the slit's light then leaves
/// the condenser's beam by.
fn condenser_u0() -> Vec3 {
    let w = condenser_w();
    let a = C1_FOLD_DEG.to_radians();
    w * a.cos() + w.cross(Vec3::Y) * a.sin()
}

/// `v` turned `t` about X.
fn turn_x(v: Vec3, t: f64) -> Vec3 {
    Vec3::new(v.x, v.y * t.cos() - v.z * t.sin(), v.y * t.sin() + v.z * t.cos())
}

/// `v` turned `t` about the collimator's line of sight (Z).
fn turn_z(v: Vec3, t: f64) -> Vec3 {
    Vec3::new(v.x * t.cos() - v.y * t.sin(), v.x * t.sin() + v.y * t.cos(), v.z)
}

/// From the slit to the pick-off's centre on the tube's axis.
fn pickoff_offset() -> Vec3 {
    condenser_w() * CONDENSER_F - condenser_u0() * condenser_gap()
}

/// The condenser mirror's centre, the whole condenser turned `turn` about
/// the slit's line of sight: every chief ray to the slit comes from it.
fn condenser_mirror(slit: Vec3, turn: f64) -> Vec3 {
    slit - turn_z(condenser_u0(), turn) * condenser_gap()
}

/// The slit's run, square to the condenser mirror's direction.
fn slit_along(turn: f64) -> Vec3 {
    Vec3::new(-turn.sin(), turn.cos(), 0.0)
}

/// The slit's point `h` along it, `dr` out across it. The slit is a strip of
/// the paraboloid with the condenser mirror at its focus, and it lies where
/// the pick-off's image does, all of it as far from that mirror as its
/// centre: on the paraboloid those points make a circle in the plane square
/// to the paraboloid's axis (the collimator's line of sight), round the
/// mirror's foot in that plane.
///
/// That circle curves the slit, as seen down the collimator's line of sight,
/// about four times as much as straightens the spectrum's lines. So the slit
/// follows the paraboloid along a flatter curve instead: its cut with a
/// plane through the slit's centre and run, tilted about the run, which down
/// the axis is a circle of radius `rc / κ` ([`slit_curve`]) tangent to the
/// old one. Its ends then lie 0.6 mm or so nearer or farther than the
/// pick-off's image, 0.07 mm of blur in the slit's f/8 cone.
fn slit_point(slit: Vec3, turn: f64, h: f64, dr: f64) -> Vec3 {
    let c = condenser_mirror(slit, turn);
    let foot = c + Vec3::Z * (slit - c).z;
    let rc = (slit - foot).length();
    if std::env::var_os("STRAIGHT_SLIT").is_some() {
        return slit + slit_along(turn) * h + (slit - foot).normalize() * dr;
    }
    let k = slit_curve();
    let radial = (slit - foot).normalize();
    let mut t = Vec3::Z.cross(radial).normalize();
    if t.dot(slit_along(turn)) < 0.0 {
        t = -t;
    }
    // Down the axis: the slit's centre, then along the flatter circle.
    let flat = if k.abs() < 1e-9 {
        slit + t * h + radial * dr
    } else {
        let r = rc / k;
        let a = (h / r).atan();
        slit - radial * r + (radial * a.cos() + t * a.sin()) * (r + dr)
    };
    // Onto the paraboloid: focus `c`, light from it leaving along -Z.
    let q = slit - c;
    let fp = 0.5 * (q.length() + q.z);
    let rho2 = (flat - foot).length_squared() - (flat - foot).z.powi(2);
    Vec3::new(flat.x, flat.y, c.z + fp - rho2 / (4.0 * fp))
}

/// The slit's curvature as a fraction of the circle round the condenser
/// mirror's foot (1 is that circle, 0 straight): fitted with the slit's turn
/// by [`straighten_lines`], or `SLIT_K`.
static LAST_LAMP_LOSS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static SLIT_CURVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x3FF0_0000_0000_0000);

fn slit_curve() -> f64 {
    f64::from_bits(SLIT_CURVE.load(std::sync::atomic::Ordering::Relaxed))
}

fn set_slit_curve(k: f64) {
    SLIT_CURVE.store(k.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// The sun's chief ray to the slit's point `h` along it: a start just past
/// the condenser mirror, and the direction.
fn sun_ray(slit: Vec3, turn: f64, h: f64) -> (Vec3, Vec3) {
    let c = condenser_mirror(slit, turn);
    let dir = (slit_point(slit, turn, h, 0.0) - c).normalize();
    (c + dir * 5.0, dir)
}

/// The pick-off strip's run on the mirror: the direction whose image in the
/// condenser mirror lies along the slit (the mirror image inverts, which only
/// flips the sign).
fn pickoff_run(slit: Vec3, turn: f64) -> Vec3 {
    let c1 = condenser_mirror(slit, turn);
    let pickoff = slit + pickoff_offset();
    let to_c1 = (c1 - pickoff).normalize();
    let n_c1 = (-to_c1 + (slit - c1).normalize()).normalize();
    let n_p = (to_c1 - Vec3::X).normalize();
    let u = reflect(slit_along(turn), n_c1);
    (u - to_c1 * (u.dot(n_p) / to_c1.dot(n_p))).normalize()
}

/// The pick-off point whose light the condenser mirror's centre sends to
/// the slit's point `h` along it, `dr` out across it (at the slit): back
/// along that chief ray, reflected in the condenser mirror, to where it
/// meets the pick-off's paraboloid (focus the condenser mirror, axis -X).
fn pickoff_point(slit: Vec3, turn: f64, h: f64, dr: f64) -> Vec3 {
    let c1 = condenser_mirror(slit, turn);
    let pickoff = slit + pickoff_offset();
    let n_c1 = (-(c1 - pickoff).normalize() + (slit - c1).normalize()).normalize();
    let dir = (slit_point(slit, turn, h, dr) - c1).normalize();
    let e = -reflect(dir, n_c1);
    // The paraboloid through the pick-off's centre: |X - F| = (X - F).o + 2 fp.
    let o = -Vec3::X;
    let q = pickoff - c1;
    let fp = 0.5 * (q.length() - q.dot(o));
    c1 + e * (2.0 * fp / (1.0 - e.dot(o)))
}

/// The pick-off: a strip of the paraboloid with the condenser mirror at its
/// focus, on the arc [`pickoff_point`] traces, PICKOFF_OVERFILL over the
/// slit's conjugate.
fn pickoff_piece(slit: Vec3, slit_height: f64, turn: f64) -> (Isometry3, SurfaceGeometry3) {
    let c1 = condenser_mirror(slit, turn);
    let pickoff = slit + pickoff_offset();
    let (v, a, r) = paraboloid(c1, -Vec3::X, pickoff);
    let fr = frame(v, a, Vec3::Y);
    let n = 24;
    let hs: Vec<f64> = (0..=n).map(|i| PICKOFF_OVERFILL * slit_height * (i as f64 / n as f64 - 0.5)).collect();
    let at = |h: f64, dr: f64| {
        let p = fr.inverse_transform_point(pickoff_point(slit, turn, h, dr));
        oc_core::vec::Vec2::new(p.x, p.y)
    };
    let w = 0.5 * PICKOFF_OVERFILL * slit_width();
    let mut vertices: Vec<oc_core::vec::Vec2> = hs.iter().map(|&h| at(h, w)).collect();
    vertices.extend(hs.iter().rev().map(|&h| at(h, -w)));
    let area: f64 = (0..vertices.len()).map(|i| {
        let (p, q) = (vertices[i], vertices[(i + 1) % vertices.len()]);
        p.x * q.y - q.x * p.y
    }).sum();
    if area < 0.0 {
        vertices.reverse();
    }
    let reach = vertices.iter().map(|q| q.length()).fold(0.0, f64::max);
    let geometry = SurfaceGeometry3::SurfaceOfRevolution {
        sag: SagModel::Conic {
            curvature_per_mm: 1.0 / r,
            conic: -1.0,
        },
        radial_aperture: RadialAperture {
            inner_radius_mm: 0.0,
            outer_radius_mm: reach + 1.0,
        },
        outline: Some(PlanarShape2::Polygon { vertices }),
    };
    (fr, geometry)
}

/// How far the pick-off's arc bows off the straight line through its ends,
/// as the tube's light sees it.
fn pickoff_bow(slit: Vec3, slit_height: f64, turn: f64) -> f64 {
    let h = 0.5 * PICKOFF_OVERFILL * slit_height;
    let (a, b, c) = (pickoff_point(slit, turn, -h, 0.0), pickoff_point(slit, turn, h, 0.0), pickoff_point(slit, turn, 0.0, 0.0));
    let mid = (a + b) * 0.5;
    ((c - mid) - Vec3::X * (c - mid).x).length()
}

/// The pick-off's strip as the tube's light sees it: height and width.
fn pickoff_size(slit_height: f64, turn: f64) -> (f64, f64) {
    let gap = condenser_gap();
    let m = gap / (pickoff_offset() + turn_z(condenser_u0(), turn) * gap).length();
    (PICKOFF_OVERFILL * slit_height / m, PICKOFF_OVERFILL * slit_width() / m)
}

/// Every mirror of the spectrograph: enhanced silver, a quarter-wave pair of
/// SiO2 and Ta2O5 over silver (99.1% of the light at normal incidence, 98.7%
/// at 45 degrees, 97.6% at 380 nm, and a percent apart between s and p
/// there), or the constant `MIRROR_R` with `REAL_MIRRORS=0`.
fn hr() -> SurfaceResponse {
    super::realism::mirror(super::realism::enhanced_silver(), MIRROR_R)
}

fn table(points: &[(f64, f64)]) -> SpectralScalar {
    SpectralScalar::Tabulated {
        lambda_um: points.iter().map(|p| p.0).collect(),
        values: points.iter().map(|p| p.1).collect(),
        interpolation: Interpolation::Linear,
        out_of_range: OutOfRangePolicy::Clamp,
    }
}

/// A flat mirror at `pos` turning `from` into `to`, `hw` across the plane of
/// the turn and `hh` along `along`.
fn fold(s: &mut Scene3, pos: Vec3, from: Vec3, to: Vec3, along: Vec3, hw: f64, hh: f64, label: &str) -> ObjectId {
    let n = (to - from).normalize();
    surface(s, frame(pos, n, along), rect(hw, hh), hr(), SurfaceResponse::Absorber, label)
}

/// A detector face of `hw` x `hh` at `pos`, facing `normal`, its `u` axis
/// along `u`.
pub fn detector(
    s: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    u: Vec3,
    hw: f64,
    hh: f64,
    bins: [usize; 2],
    termination: DetectorTermination,
    label: &str,
) -> ObjectId {
    let id = s.add_area_detector(pos, normal, hw, hh, bins);
    let d = s.detectors.iter_mut().find(|d| d.common.id.0 == id.0).unwrap();
    // Local X along u, local Z along the normal: v = Z x u.
    d.common.transform = frame(pos, normal, normal.cross(u));
    d.geometry = DetectorGeometry3::Rectangle {
        half_width_mm: hw,
        half_height_mm: hh,
    };
    d.termination = termination;
    d.common.label = label.into();
    ObjectId(id.0)
}

/// A lamp beside the sun: what it emits and how its arm is built.
pub struct Lamp {
    pub name: &'static str,
    /// The emitting area, across the dispersion and along the slit, mm.
    across: f64,
    along: f64,
    /// Radiance in narrow bands, `(λ µm, W/(m² sr))`, 380 to 720 nm.
    radiance: Vec<(f64, f64)>,
    /// The condenser's focal length and clear radius, mm.
    condenser_f: f64,
    condenser_semi: f64,
    pub slit_width: f64,
    /// The wavelength whose image the curved slit makes straight and puts
    /// under the same wavelength of the sun's spectrum.
    lambda_ref: f64,
    /// An ellipsoidal cold mirror in place of the condenser: the coil at its
    /// near focus `throw` mm from the slit along the light, the mirror as
    /// wide as makes the cone reach the slit at `na`. The emitter is then a
    /// cylinder `across` thick and `along` long, radiating from its side.
    reflector: Option<(f64, f64)>,
}

/// The lamps beside the sun: the emission lamp (`EMISSION=hg` for mercury,
/// sodium otherwise) above, the halogen below.
pub fn lamps() -> Vec<Lamp> {
    let emission = if std::env::var("EMISSION").is_ok_and(|v| v == "hg") { Lamp::mercury() } else { Lamp::sodium() };
    vec![emission, Lamp::halogen()]
}

/// The eye's response at `lambda_um`, CIE 1931 `ȳ`.
fn v_lambda(lambda_um: f64) -> f64 {
    oc_core::colorimetry::cie_1931(lambda_um)[1]
}

impl Lamp {
    /// Osram Na/10, a low-pressure sodium spectral lamp: a discharge 15 mm
    /// tall and 10 wide in a clear bulb, burning upright, luminance 12 stilb
    /// (1.2e5 cd/m², Osram's data sheet). All of its visible light is in the
    /// D lines, D2 twice D1 (NIST); the neon that starts it fades once the
    /// sodium is warm. Upright, its 15 mm lie across the dispersion.
    pub fn sodium() -> Self {
        let lines = [(0.588_995, 2.0), (0.589_592, 1.0)];
        let lm_per_w: f64 = lines.iter().map(|l| l.1 * oc_core::colorimetry::LM_PER_W * v_lambda(l.0)).sum::<f64>()
            / lines.iter().map(|l| l.1).sum::<f64>();
        let total = 1.2e5 / lm_per_w;
        let w: f64 = lines.iter().map(|l| l.1).sum();
        Self {
            name: "Sodium spectral lamp, Osram Na/10",
            across: 15.0,
            along: 10.0,
            radiance: lines.iter().map(|l| (l.0, total * l.1 / w)).collect(),
            condenser_f: 100.0,
            condenser_semi: 52.0,
            slit_width: knob("EMI_SLIT_W", slit_width()),
            lambda_ref: 0.589_29,
            reflector: None,
        }
    }

    /// Osram's mercury spectral lamp: a discharge 20 mm tall and 3 wide,
    /// burning upright, luminance 150 stilb (1.5e6 cd/m², 110 cd; Osram's
    /// data sheet via lamptech). Its visible lines with the strengths NIST
    /// measured for low-pressure mercury-argon lamps, in watts, 435.8 nm =
    /// 10000 (Sansonetti, Salit and Reader, Appl. Opt. 35, 74, 1996). Upright,
    /// its 3 mm lie along the slit and fill f/33 through a 100 mm
    /// condenser, short of the collimator's f/22.5; its 20 mm overfill
    /// across.
    pub fn mercury() -> Self {
        let lines = [
            (0.404_656_5, 4_400.0),
            (0.407_783_7, 270.0),
            (0.434_750_6, 34.0),
            (0.435_833_5, 10_000.0),
            (0.546_075_0, 10_000.0),
            (0.576_961_0, 1_100.0),
            (0.579_067_0, 1_200.0),
        ];
        let lm: f64 = lines.iter().map(|l| oc_core::colorimetry::LM_PER_W * v_lambda(l.0) * l.1).sum();
        let scale = 1.5e6 / lm;
        Self {
            name: "Mercury spectral lamp, Osram Hg",
            across: 20.0,
            along: 3.0,
            radiance: lines.iter().map(|l| (l.0, l.1 * scale)).collect(),
            condenser_f: 100.0,
            condenser_semi: 52.0,
            slit_width: slit_width(),
            lambda_ref: 0.546_075,
            reflector: None,
        }
    }

    /// Osram HLX 64625, a 100 W 12 V halogen lamp: a coiled filament 4.2 mm
    /// long and 2.3 mm across, 3600 lm, colour temperature 3450 K (Osram's
    /// data sheet). The coil is taken as a cylinder radiating evenly from its
    /// side, 3.8e7 cd/m², with tungsten's spectrum at that colour. The
    /// flux is measured outside the capsule, so its losses are in it.
    /// The coil lies along the slit at the near focus of an ellipsoidal cold
    /// mirror whose far focus is the slit: the mirror takes the coil's whole
    /// back half and images it 17 to 33 times onto the slit at the
    /// collimator's f/8.4 (a 500 mm throw, 60 mm across). A condenser at the
    /// coil's distance would fill only 2.3 / 100 of the collimator's 0.12
    /// rad across and 4.2 / 100 along it, about 7 %.
    pub fn halogen() -> Self {
        const T: f64 = 3_450.0;
        let (d, l) = (2.3e-3, 4.2e-3);
        let luminance = 3_600.0 / (std::f64::consts::PI * std::f64::consts::PI * d * l);
        // Tungsten at that colour, its emissivity included ([`super::realism::tungsten`]).
        let planck = super::realism::tungsten(T);
        // Every 0.1 nm: each band is traced at its one wavelength, and bands 1 nm
        // apart (7 mm on the table) put a ripple on the halogen's band.
        let bands: Vec<(f64, f64)> = (0..3350).map(|i| (0.37005 + 0.0001 * i as f64, planck(0.37005 + 0.0001 * i as f64))).collect();
        let lm: f64 = bands.iter().map(|b| oc_core::colorimetry::LM_PER_W * v_lambda(b.0) * b.1).sum();
        Self {
            name: "Halogen lamp, Osram HLX 64625, 100 W",
            across: 2.3,
            along: 4.2,
            radiance: bands.iter().map(|b| (b.0, b.1 * luminance / lm)).collect(),
            condenser_f: 100.0,
            condenser_semi: 52.0,
            slit_width: continuum_slit_width(),
            lambda_ref: 0.55,
            reflector: Some((knob("HAL_THROW", 500.0), knob("HAL_NA", 0.5 * BEAM / F_COLL))),
        }
    }

    /// Luminance, cd/m².
    pub fn luminance(&self) -> f64 {
        self.radiance.iter().map(|b| oc_core::colorimetry::LM_PER_W * v_lambda(b.0) * b.1).sum()
    }
}

/// The lamp's slit, solved by tracing: its centre, and the slit's middle
/// line `(u, w)` in its plane (`u` along world `X`, `w` along world `Y`).
struct LampSlit {
    centre: Vec3,
    line: Vec<(f64, f64)>,
    /// The band's edges on the table, in the table detector's `v`.
    band: (f64, f64),
    /// Where the slit is in focus, along the light: `a + b u` from the
    /// centre. The plate is tilted to follow it.
    depth: (f64, f64),
    /// Along the slit, across it, and the way the light leaves.
    run: Vec3,
    across: Vec3,
    dir: Vec3,
}

/// What [`optics`] placed.
struct Placed {
    table: ObjectId,
    entrance: ObjectId,
    table_centre: Vec3,
    intermediate: Vec3,
    /// Each lamp's slit centre, by side (up the sun's slit, then down).
    lamp_slit: [Vec3; 2],
    /// The lamp slits' frame: along the slit, across it, and the way their
    /// light leaves (toward the periscope), as the sun's slit has them.
    lamp_run: Vec3,
    lamp_across: Vec3,
    lamp_dir: Vec3,
    /// Heights of the folded projection path's parts, world `Y`.
    clearances: Vec<(&'static str, f64)>,
    /// The diaphragm probe, its centre and the lens's axis.
    stop_probe: Option<(ObjectId, Vec3, Vec3)>,
    slit_turn: f64,
    /// The field lens's centre, where the projection lens's axis meets the
    /// intermediate spectrum's level, and its first vertex (unfolded, as if
    /// F4 were not there), all turned.
    field_centre: Vec3,
    lens_axis0: Vec3,
    lens_vertex_unfolded: Vec3,
}

/// Where a ray of `lambda_um` from `from` along `dir` lands on the table.
fn land(s: &Scene3, table: ObjectId, from: Vec3, dir: Vec3, lambda_um: f64) -> Option<Vec3> {
    land_many(s, table, &[(from, dir, lambda_um)])[0]
}

/// Where each probe `(from, dir, λ µm)` lands on the detector `table`, in
/// one trace: preparing the scene costs far more than a handful of rays.
/// Each probe is five rays a few hundredths of a millimetre apart, told
/// apart by wavelengths a picometre-scale step apart, and the median of
/// where they land: a single ray can strike a Fresnel's riser, which bends
/// it somewhere no other ray goes.
fn land_many(s: &Scene3, table: ObjectId, probes: &[(Vec3, Vec3, f64)]) -> Vec<Option<Vec3>> {
    let offsets = [(0.0, 0.0), (0.017, 0.011), (-0.013, 0.019), (0.021, -0.015), (-0.019, -0.023)];
    let step = 1e-9;
    let tag = |k: usize, i: usize, l: f64| l + step * (k * offsets.len() + i) as f64;
    let mut rays = Vec::new();
    for (k, &(from, dir, l)) in probes.iter().enumerate() {
        let dir = dir.normalize();
        let e1 = dir.cross(if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y }).normalize();
        let e2 = dir.cross(e1);
        for (i, (a, b)) in offsets.iter().enumerate() {
            if let Ok(ray) = oc_core::vec3::Ray3::new(from + e1 * *a + e2 * *b, dir) {
                rays.push(oc_trace::LaunchRay3 {
                    ray,
                    lambda_um: tag(k, i, l),
                    flux: 1.0,
                    source: None,
                    pol: None,
                });
            }
        }
    }
    // No ghosts: the probes follow each ray's own path, and the lens's
    // faces would split every ray a dozen ways.
    let ta = std::time::Instant::now();
    let bounds_mm = oc_trace::bounds_for_scene3(s);
    let tb = std::time::Instant::now();
    let cfg = oc_trace::TraceConfig3 {
        record_detector_hits: true,
        ghosts: false,
        max_depth: 64,
        bounds_mm,
        ..oc_trace::TraceConfig3::default()
    };
    let res = oc_trace::trace3(s, &rays, &cfg);
    if std::env::var_os("TIMING").is_some() {
        eprintln!("TIMING land: bounds {:.0?}, trace {:.0?} for {} rays", tb - ta, tb.elapsed(), rays.len());
    }
    let Some(d) = res.detectors.iter().find(|d| d.detector.0 == table.0) else {
        return vec![None; probes.len()];
    };
    // Each ray's own path is its strongest hit; the lens's surfaces split
    // off weak reflections that can reach the table too.
    let mut best: std::collections::HashMap<i64, (f64, Vec3)> = Default::default();
    for h in &d.records {
        // The tag nearest this record, whatever the probe's wavelength.
        for (k, &(_, _, l)) in probes.iter().enumerate() {
            let t = ((h.lambda_um - l) / step).round();
            let base = (k * offsets.len()) as f64;
            if t >= base && t < base + offsets.len() as f64 && (h.lambda_um - l - t * step).abs() < 0.3 * step {
                let e = best.entry(t as i64).or_insert((f64::NEG_INFINITY, h.point_world));
                if h.incident_flux > e.0 {
                    *e = (h.incident_flux, h.point_world);
                }
            }
        }
    }
    (0..probes.len())
        .map(|k| {
            let mut hits: Vec<Vec3> = (0..offsets.len())
                .filter_map(|i| best.get(&((k * offsets.len() + i) as i64)).map(|q| q.1))
                .collect();
            if hits.len() < 3 {
                return None;
            }
            let mut median = |f: fn(&Vec3) -> f64| {
                hits.sort_by(|a, b| f(a).total_cmp(&f(b)));
                f(&hits[hits.len() / 2])
            };
            Some(Vec3::new(median(|p| p.x), median(|p| p.y), median(|p| p.z)))
        })
        .collect()
}

/// The table detector's `v` of a world point: local `Y` is `-Z`.
fn table_v(p: Vec3, centre: Vec3) -> f64 {
    (p - centre).dot(table_normal().cross(table_axes().0))
}

/// The rigid frame with local `X` along `x` and local `Y` as near `y` as a
/// right-handed frame allows: how a singlet or a source is aimed.
pub fn frame_x(pos: Vec3, x: Vec3, y: Vec3) -> Isometry3 {
    let x = x.normalize();
    let y = (y - x * y.dot(x)).normalize();
    let z = x.cross(y);
    let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(x, y, z));
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).expect("a unit frame")
}

/// Solve a lamp's slit on the placed optics. Its band sits `BAND_GAP` past
/// the sun's on the table and `SPECTRUM_HEIGHT` tall; along it, every point
/// of the slit is moved across so that `lambda_ref` lands where the sun's
/// does from the strip's centre. A straight slit would give lines bowed by
/// the grating (the bow grows with the square of the height off the axis,
/// several millimetres on the table this far out); the curved one gives
/// straight lines, level with the sun's.
fn lamp_slit(s: &Scene3, p: &Placed, strip: Vec3, sun_half: f64, lamp: &Lamp, side: usize) -> LampSlit {
    let c = p.lamp_slit[side];
    let (run, acr, dir) = (p.lamp_run, p.lamp_across, p.lamp_dir);
    // Along the spectrum on the table.
    let al = |q: Vec3| (q - p.table_centre).dot(table_axes().0);
    let at = |u: f64, w: f64| {
        land(s, p.table, c + run * u + acr * w - dir * 5.0, dir, lamp.lambda_ref).unwrap_or_else(|| {
            let lr = oc_trace::LaunchRay3 {
                ray: oc_core::vec3::Ray3::new(c + run * u + acr * w - dir * 5.0, dir).unwrap(),
                lambda_um: lamp.lambda_ref,
                flux: 1.0,
                source: None,
                pol: None,
            };
            let cfg = oc_trace::TraceConfig3 { max_depth: 40, bounds_mm: oc_trace::bounds_for_scene3(s), ..oc_trace::TraceConfig3::default() };
            for q in oc_trace::trace3(s, &[lr], &cfg).segments.iter().filter(|q| q.flux_start > 0.05) {
                eprintln!("LAMPPATH {:.1?} -> {:.1?} flux {:.3}", q.a, q.b, q.flux_start);
            }
            for o in &s.surfaces {
                eprintln!("OBJ {:.1?} {}", o.common.transform.pos, o.common.label);
            }
            panic!("{}: no path from the slit at u {u:.2}, w {w:.3}", lamp.name)
        })
    };
    let (from, sun_dir) = sun_ray(strip, p.slit_turn, 0.0);
    let sun = land(s, p.table, from, sun_dir, lamp.lambda_ref).expect("the sun's centre reaches the table");
    // The strip's own edge is where a ray stops meeting it: aim just
    // inside and scale out.
    let sun_c = table_v(sun, p.table_centre);
    let sun_v = |h: f64| {
        let (from, sun_dir) = sun_ray(strip, p.slit_turn, 0.98 * h);
        let q = land(s, p.table, from, sun_dir, lamp.lambda_ref).expect("the slit's end reaches the table");
        sun_c + (table_v(q, p.table_centre) - sun_c) / 0.98
    };
    let v0 = table_v(at(0.0, 0.0), p.table_centre);
    let sign = (v0 - table_v(sun, p.table_centre)).signum();
    let near_sun = if (sun_v(sun_half) - sun_v(-sun_half)).signum() == sign {
        sun_v(sun_half)
    } else {
        sun_v(-sun_half)
    };
    let near = near_sun + sign * BAND_GAP;
    let far = near + sign * SPECTRUM_HEIGHT;
    // The band's ends: v is close to linear in u.
    // Where the slit's point `u` lands, if its chief ray gets through.
    let at_opt = |u: f64, w: f64| land(s, p.table, c + run * u + acr * w - dir * 5.0, dir, lamp.lambda_ref);
    let solve_u = |target: f64| {
        let (mut u0, mut u1) = (-20.0, 20.0);
        let (mut f0, mut f1) = (table_v(at(u0, 0.0), p.table_centre) - target, table_v(at(u1, 0.0), p.table_centre) - target);
        for _ in 0..6 {
            if (f1 - f0).abs() < 1e-12 {
                break;
            }
            // Within the slit's reach: a wild secant step leaves the arm.
            let mut u2 = (u1 - f1 * (u1 - u0) / (f1 - f0)).clamp(-0.8 * sun_half - 20.0, 0.8 * sun_half + 20.0);
            // A point whose light is lost: back toward the last good one.
            let mut f2 = at_opt(u2, 0.0);
            for _ in 0..8 {
                if f2.is_some() {
                    break;
                }
                u2 = 0.5 * (u2 + u1);
                f2 = at_opt(u2, 0.0);
            }
            let Some(q) = f2 else { break };
            (u0, f0) = (u1, f1);
            u1 = u2;
            f1 = table_v(q, p.table_centre) - target;
            if f1.abs() < 1e-4 {
                break;
            }
        }
        if std::env::var_os("SPEC_DIAG").is_some() && f1.abs() > 1.0 {
            eprintln!("LAMP {}: the band's edge at {target:.1} is out of reach, {f1:+.1} mm short at u {u1:.1}", lamp.name);
        }
        u1
    };
    if std::env::var_os("LAMP_PATH").is_some() {
        let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(c - dir * 5.0, dir).unwrap(), lambda_um: lamp.lambda_ref, flux: 1.0, source: None, pol: None };
        let cfg = oc_trace::TraceConfig3 { max_depth: 40, ghosts: false, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(s), ..oc_trace::TraceConfig3::default() };
        let mut names: std::collections::HashMap<u64, String> = Default::default();
        for o in &s.surfaces {
            names.insert(o.common.id.0, o.common.label.clone());
        }
        for o in &s.bodies {
            names.insert(o.common.id.0, o.common.label.clone());
        }
        for o in &s.apertures {
            names.insert(o.common.id.0, o.common.label.clone());
        }
        eprintln!("LAMPPATH start {:.1?} dir {:.3?} run {:.3?}; sun slit {:.1?}", c, dir, run, strip);
        for o in s.surfaces.iter().filter(|o| o.common.label.starts_with("Periscope")) {
            eprintln!("LAMPPATH {} at {:.1?}", o.common.label, o.common.transform.pos);
        }
        for e in &oc_trace::trace3(s, &[lr], &cfg).events {
            let what = match &e.kind {
                oc_trace::TraceEventKind::Interact { surface, .. } => names.get(&surface.object.0).cloned().unwrap_or_default(),
                oc_trace::TraceEventKind::ApertureBlocked { aperture } => format!("BLOCKED {}", names.get(&aperture.0).cloned().unwrap_or_default()),
                oc_trace::TraceEventKind::Escaped => "ESCAPED".into(),
                k => format!("{k:?}").chars().take(40).collect(),
            };
            eprintln!("LAMPPATH {} @ {:.0?}: {}", lamp.name.split(',').next().unwrap_or(""), e.point, what.chars().take(70).collect::<String>());
        }
    }
    if std::env::var_os("SPEC_DIAG").is_some() {
        let probe = |u: f64| land(s, p.table, c + run * u - dir * 5.0, dir, lamp.lambda_ref).map(|q| (al(q), table_v(q, p.table_centre)));
        eprintln!(
            "LAMP {}: sun centre v {sun_c:.1}, sun edges {:.1} {:.1}; band wanted {near:.1} to {far:.1}; slit at u -40/-20/0/20/40 lands {:?} {:?} {:?} {:?} {:?}",
            lamp.name, sun_v(sun_half), sun_v(-sun_half), probe(-40.0), probe(-20.0), probe(0.0), probe(20.0), probe(40.0)
        );
    }
    let (ua, ub) = (solve_u(near), solve_u(far));
    let n = 24;
    let line: Vec<(f64, f64)> = (0..=n)
        .map(|i| {
            let u = ua + (ub - ua) * i as f64 / n as f64;
            let (mut w0, mut w1) = (0.0, 0.2);
            let (Some(a0), Some(a1)) = (at_opt(u, w0), at_opt(u, w1)) else { return (u, 0.0) };
            let (mut f0, mut f1) = (al(a0) - al(sun), al(a1) - al(sun));
            for _ in 0..6 {
                let w2 = w1 - f1 * (w1 - w0) / (f1 - f0);
                let Some(a2) = at_opt(u, w2) else { break };
                (w0, f0) = (w1, f1);
                w1 = w2;
                f1 = al(a2) - al(sun);
                if f1.abs() < 1e-4 {
                    break;
                }
            }
            (u, w1)
        })
        .collect();
    // Focus: two rays through a point of the slit, 10 mrad either side
    // across it, land together when the point is in focus. The collimator's
    // field curves, so the best place moves along the slit's length.
    let pair = |u: f64, w: f64, dz: f64| -> Option<f64> {
        let q = c + run * u + acr * w + dir * dz;
        let x = |a: f64| {
            let d = (dir + acr * a).normalize();
            land(s, p.table, q - d * 5.0, d, lamp.lambda_ref).map(al)
        };
        Some(x(0.006)? - x(-0.006)?)
    };
    // Points whose probes are lost are left out.
    let focus: Vec<(f64, f64)> = line
        .iter()
        .step_by(4)
        .filter_map(|&(u, w)| {
            let (mut z0, mut z1) = (0.0, -20.0);
            let (mut f0, mut f1) = (pair(u, w, z0)?, pair(u, w, z1)?);
            for _ in 0..6 {
                if (f1 - f0).abs() < 1e-12 {
                    break;
                }
                let z2 = (z1 - f1 * (z1 - z0) / (f1 - f0)).clamp(-150.0, 60.0);
                let Some(f2) = pair(u, w, z2) else { break };
                (z0, f0) = (z1, f1);
                z1 = z2;
                f1 = f2;
                if f1.abs() < 1e-4 {
                    break;
                }
            }
            Some((u, z1))
        })
        .collect();
    let n = focus.len() as f64;
    let (su, sz) = focus.iter().fold((0.0, 0.0), |(a, b), q| (a + q.0, b + q.1));
    let (mu, mz) = (su / n, sz / n);
    let b = focus.iter().map(|q| (q.0 - mu) * (q.1 - mz)).sum::<f64>() / focus.iter().map(|q| (q.0 - mu).powi(2)).sum::<f64>();
    LampSlit {
        centre: c,
        line,
        band: (near.min(far), near.max(far)),
        depth: (mz - b * mu, b),
        run,
        across: acr,
        dir,
    }
}

/// Place everything but the lamps: the sun's slit, the spectrograph, the
/// lamps' folds and the table.
fn optics(s: &mut Scene3, slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) -> Placed {
    let (x, y, z) = (Vec3::X, Vec3::Y, Vec3::Z);

    // 1. The condenser, three mirrors, each a conic that images one point
    // onto another exactly however far off its axis it is used. The pick-off,
    // a strip of a paraboloid on the tube's axis in the gap, turns the tube's
    // +X light to -Z and focuses its directions (the sun's disc) onto the
    // condenser mirror. That mirror, a piece of an ellipsoid with the
    // pick-off and the slit at its foci, turns the light along +X and images
    // the strip onto the slit, smaller by the gap over CONDENSER_F: the
    // tube's f/36 becomes the collimator's f-number. The slit, a strip of a
    // paraboloid with the condenser mirror at its focus, turns the light to
    // -Z with every point's cone centred on -Z, as the tube's light was.
    let d1 = -z;
    let c1 = condenser_mirror(slit, d.slit_turn);
    let along = slit_along(d.slit_turn);
    let pickoff = slit + pickoff_offset();
    let (ph, pw) = pickoff_size(d.slit_height, d.slit_turn);
    let root2 = std::f64::consts::SQRT_2;
    // The condenser mirror's normal, and the pick-off strip's run: the
    // direction whose image lies along the slit (the mirror image inverts,
    // which only flips the sign).
    let to_c1 = (c1 - pickoff).normalize();
    let n_c1 = (-to_c1 + (slit - c1).normalize()).normalize();
    let n_p = (to_c1 - x).normalize();
    let along_p = pickoff_run(slit, d.slit_turn);
    let (v, a, r) = paraboloid(c1, -x, pickoff);
    let (fr, geom) = pickoff_piece(slit, d.slit_height, d.slit_turn);
    let _ = (v, a, r, root2);
    if std::env::var_os("SPEC_DIAG").is_some() {
        let across_p = n_p.cross(along_p).normalize();
        eprintln!(
            "PICKOFF run {:.4?}, across {:.4?}, normal {:.4?}; ends reach x {:+.1} and {:+.1} mm from the centre, edges {:+.1} {:+.1}",
            along_p, across_p, n_p, along_p.x * 0.5 * ph, -along_p.x * 0.5 * ph, across_p.x * 0.5 * pw * root2, -across_p.x * 0.5 * pw * root2
        );
    }
    // The strip is ground on a glass bar 6 mm deep (`PICKOFF_DEPTH`), which
    // shades the light running on down the tube to the melting experiment:
    // its back, a black copy of the face that far behind it, stands for it.
    if super::realism::on("MOUNTS") {
        let mut back = fr;
        back.pos -= n_p * knob("PICKOFF_DEPTH", 6.0);
        surface(s, back, geom.clone(), SurfaceResponse::Absorber, SurfaceResponse::Absorber, "Condenser pick-off, back of its glass bar (black)");
    }
    surface(
        s,
        fr,
        geom,
        hr(),
        SurfaceResponse::Absorber,
        &format!("Condenser pick-off, off-axis paraboloid, f {:.0} mm, {ph:.0} x {pw:.1} mm as the tube's light sees it", (c1 - pickoff).length()),
    );
    let disc = (pickoff - c1).length() * TUBE_HALF_ANGLE.tan() * 1.25;
    let (v, a, r, k) = ellipsoid(pickoff, slit, c1);
    let (fr, geom) = conic_piece(v, a, r, k, c1, n_c1, along, disc * root2, disc);
    surface(
        s,
        fr,
        geom,
        hr(),
        SurfaceResponse::Absorber,
        &format!("Condenser mirror, off-axis ellipsoid, {:.0} mm across", 2.0 * disc),
    );
    // The slit: every chief ray from the condenser mirror meets the
    // paraboloid on the circle round its focus, and the pick-off's image
    // lies on it too, so the strip follows that arc, the slit width across as
    // the collimator sees it down the axis.
    let (v, a, r) = paraboloid(c1, d1, slit);
    let fr = frame(v, a, y);
    let gap = condenser_gap();
    let arc = |h: f64, dr: f64| {
        let p = fr.inverse_transform_point(slit_point(slit, d.slit_turn, h, dr));
        oc_core::vec::Vec2::new(p.x, p.y)
    };
    let n = 24;
    let hs: Vec<f64> = (0..=n).map(|i| d.slit_height * (i as f64 / n as f64 - 0.5)).collect();
    let mut vertices: Vec<oc_core::vec::Vec2> = hs.iter().map(|&h| arc(h, 0.5 * slit_width())).collect();
    vertices.extend(hs.iter().rev().map(|&h| arc(h, -0.5 * slit_width())));
    let area: f64 = (0..vertices.len()).map(|i| {
        let (p, q) = (vertices[i], vertices[(i + 1) % vertices.len()]);
        p.x * q.y - q.x * p.y
    }).sum();
    if area < 0.0 {
        vertices.reverse();
    }
    let geom = SurfaceGeometry3::SurfaceOfRevolution {
        sag: SagModel::Conic {
            curvature_per_mm: 1.0 / r,
            conic: -1.0,
        },
        radial_aperture: RadialAperture {
            inner_radius_mm: 0.0,
            outer_radius_mm: gap + slit_width() + 1.0,
        },
        outline: Some(PlanarShape2::Polygon { vertices }),
    };
    surface(
        s,
        fr,
        geom,
        hr(),
        SurfaceResponse::Absorber,
        &format!("Spectrograph slit: off-axis paraboloid strip on an arc of radius {gap:.0} mm, {:.0} x {} mm", d.slit_height, slit_width()),
    );

    // 2. UV and IR cut.
    let filter = uv_ir_cut();
    let half_h = 0.5 * d.slit_height + FILTER_AT * 0.5 * BEAM / F_COLL + 5.0;
    let half_w = 0.5 * slit_width() + FILTER_AT * 0.5 * beam_along() / F_COLL + 5.0;
    surface(
        s,
        frame(slit + d1 * FILTER_AT, -d1, y),
        rect(half_w, half_h),
        filter.clone(),
        filter,
        "UV/IR cut filter, 375-690 nm",
    );
    let entrance = detector(
        s,
        slit + d1 * (FILTER_AT + 20.0),
        -d1,
        x,
        half_w,
        half_h,
        [6, 18],
        DetectorTermination::ProbeAndContinue,
        "Spectrograph entrance (after the filter)",
    );

    // The periscope takes the light CAMERA_DROP down, both folds in the
    // vertical plane the light runs in.
    let slope = TUBE_HALF_ANGLE.tan().max(0.5 * BEAM.max(beam_along()) / F_COLL);
    let p_p1 = slit + d1 * knob("PERISCOPE_AT", PERISCOPE_AT);
    let p_p2 = p_p1 - y * knob("CAMERA_DROP", CAMERA_DROP);
    for (at, from, to, reach, name) in [
        (p_p1, d1, -y, knob("PERISCOPE_AT", PERISCOPE_AT), "Periscope P1, down"),
        (p_p2, -y, d1, knob("PERISCOPE_AT", PERISCOPE_AT) + knob("CAMERA_DROP", CAMERA_DROP), "Periscope P2, level again"),
    ] {
        let c = reach * slope;
        let across = if lamps_on() { 0.5 * continuum_slit_width() } else { 0.5 * slit_width() };
        fold(s, at, from, to, x, (d.field_half_height() + c + 5.0) * std::f64::consts::SQRT_2, across + c + 5.0, name);
    }

    // 3. F1 turns it along the tube. It is only as tall as the sun's light
    // there, so the lamps' light passes above and below it.
    let p_f1 = p_p2 + d1 * (STRIP_TO_F1 - knob("PERISCOPE_AT", PERISCOPE_AT) - knob("CAMERA_DROP", CAMERA_DROP));
    let d2 = x;
    let cone = STRIP_TO_F1 * slope;
    let f1_hw = (0.5 * slit_width() + cone + 5.0) * std::f64::consts::SQRT_2;
    // Everything past F1 is built along the tube, then turned as one piece
    // about the vertical through F1 ([`downstream_turn`]); F1 sends the light
    // the turned way, and where that is straight on, there is no F1.
    let turned = downstream_turn(p_f1);
    let d2_turned = turned.rotate(d2);
    if (d2_turned - d1).length() > 1e-6 {
        fold(s, p_f1, d1, d2_turned, y, f1_hw, 0.5 * d.slit_height + cone + 1.5, "Fold F1");
    }
    let built = (s.bodies.len(), s.surfaces.len(), s.apertures.len(), s.detectors.len());

    // The lamps' slits: in the collimator's focal surface with the sun's,
    // one past each end of it, as far from its centre as a band and a gap
    // on the table. Their light leaves the way the sun's does, into the
    // periscope; their arms stand behind them, toward the tube.
    let lamp_run = slit_along(d.slit_turn);
    let lamp_across = d1.cross(lamp_run).normalize();
    let mut lamp_slit = [Vec3::ZERO; 2];
    for (i, side) in [1.0, -1.0].into_iter().enumerate() {
        lamp_slit[i] = slit + lamp_run * (side * (d.slit_height + d.slit_gap));
    }

    // 4. The collimator, as tall as all three slits and the lamps' light.
    let gamma = knob("GAMMA_DEG", GAMMA_DEG).to_radians();
    // `COLL_PHI` (degrees): which way M1 turns the light back, round the
    // incoming axis: -90 sideways (as built), 180 straight down. The
    // paraboloid's aberration for points up the slit is astigmatism whose
    // axes lie along its turn's plane and across it; with the slit across
    // that plane (as built) the lines blur diagonally, growing with the
    // height up the slit; with the slit in it, the blur is a focus shift
    // that the table's tilt takes up. About 138 keeps the slit's image
    // parallel to the grooves through F2.
    let phi = knob("COLL_PHI", -90.0).to_radians();
    let d3 = Vec3::new(-gamma.cos(), gamma.sin() * phi.cos(), gamma.sin() * phi.sin());
    let p_m1 = p_f1 + d2 * (F_COLL - STRIP_TO_F1);
    let field = d.field_half_height();
    // The slit as M1 sees it, through F1.
    let focus = p_f1 - d2 * STRIP_TO_F1;
    let (v, a, r) = paraboloid(focus, d3, p_m1);
    let m1_half = (0.5 * BEAM.max(beam_along()) / (0.5 * gamma).cos() + 20.0, field + 0.5 * BEAM.max(beam_along()) + 20.0);
    let m1_half = if knob("COLL_PHI", -90.0) == -90.0 { m1_half } else { (m1_half.0.max(m1_half.1), m1_half.0.max(m1_half.1)) };
    let (fr, geom) = conic_piece(v, a, r, -1.0, p_m1, (d3 - d2).normalize(), y, m1_half.0, m1_half.1);
    surface(
        s,
        fr,
        geom,
        hr(),
        SurfaceResponse::Absorber,
        &format!("Collimator M1, off-axis paraboloid, f {F_COLL:.0} mm, {BEAM:.0} mm beam"),
    );

    // 5. F2 onto the grating, 6. the grating at the pupil, F_COLL after M1.
    let p_f2 = p_m1 + d3 * (F_COLL - knob("F2_TO_GRATING", F2_TO_GRATING));
    let turn = 0.5 * (-d3.dot(g.arrive)).acos();
    let f2_half = (
        0.5 * BEAM.max(beam_along()) / turn.cos() + 5.0,
        0.5 * BEAM.max(beam_along()) + field * knob("F2_TO_GRATING", F2_TO_GRATING) / F_COLL + 5.0,
    );
    // Turned out of the horizontal, F2's long side lies along the light's
    // turn: along the fold's plane, across it the beam's width.
    let (f2_along, f2_half) = if knob("COLL_PHI", -90.0) == -90.0 {
        (y, f2_half)
    } else {
        let plane_n = d3.cross(g.arrive).normalize();
        (plane_n, (0.5 * BEAM.max(beam_along()) / turn.cos() + field * knob("F2_TO_GRATING", F2_TO_GRATING) / F_COLL + 10.0, 0.5 * BEAM.max(beam_along()) + field * knob("F2_TO_GRATING", F2_TO_GRATING) / F_COLL + 10.0))
    };
    // `F2_PAIR="L1 L2"`: two flats instead of F2, for M1 turned in the slit's
    // plane (`COLL_PHI=180`, the light back and down). F2a, L1 after M1,
    // folds in that vertical plane, level again and back along the incoming
    // axis beneath it; F2b, L2 further, folds in the horizontal plane onto
    // the grating. A fold in the vertical plane keeps the slit's image in
    // it, one in the horizontal plane keeps vertical vertical: the slit
    // reaches the grating along its grooves, and everything past it keeps
    // its orientation.
    let pair: Vec<f64> = std::env::var("F2_PAIR").map(|v| v.split_whitespace().filter_map(|q| q.parse().ok()).collect()).unwrap_or_default();
    let p_g = if let [l1, l2] = pair[..] {
        let spread = |dist: f64| field * dist / F_COLL;
        // `F2_W="x y z"`: the light's way from F2a to F2b (unturned frame),
        // one of the family that keeps the slit along the grooves (level
        // along +X by default); the folds' long sides lie in their planes.
        let w = std::env::var("F2_W").ok().map(|v| v.split_whitespace().filter_map(|q| q.parse::<f64>().ok()).collect::<Vec<_>>()).filter(|v| v.len() == 3).map_or(d2, |v| Vec3::new(v[0], v[1], v[2]).normalize());
        let p_a = p_m1 + d3 * l1;
        let inc_a = 0.5 * (-d3.dot(w)).acos();
        let across = |a: Vec3, b: Vec3| {
            let c = a.cross(b);
            if c.length() < 1e-9 { z } else { c.normalize() }
        };
        // Sized to the beam's own axes: the slit lies in M1's plane, so at
        // F2a the field spreads level, across its fold, and the beam's long
        // (dispersion) axis lies in the fold; at F2b the dispersion lies
        // level again, across its fold, and the field in it.
        let (half_disp, half_slit) = (0.5 * beam_along(), 0.5 * BEAM);
        fold(s, p_a, d3, w, across(d3, w), half_disp / inc_a.cos() + 10.0, half_slit + spread(F_COLL - l1) + 10.0, &format!("Fold F2a, {:.1} deg incidence", inc_a.to_degrees()));
        let p_b = p_a + w * l2;
        let inc_b = 0.5 * (-w.dot(g.arrive)).acos();
        fold(s, p_b, w, g.arrive, across(w, g.arrive), (half_slit + spread(F_COLL - l1 - l2)) / inc_b.cos() + 10.0, half_disp + 10.0, &format!("Fold F2b, {:.1} deg incidence, onto the grating", inc_b.to_degrees()));
        eprintln!("F2PAIR grating {:.0} mm below M1; F2a inc {:.1}, F2b inc {:.1}", -(p_b + g.arrive * (F_COLL - l1 - l2) - p_m1).y, inc_a.to_degrees(), inc_b.to_degrees());
        p_b + g.arrive * (F_COLL - l1 - l2)
    } else {
        fold(s, p_f2, d3, g.arrive, f2_along, f2_half.0, f2_half.1, &format!("Fold F2, {:.1} deg incidence", turn.to_degrees()));
        p_f2 + g.arrive * knob("F2_TO_GRATING", F2_TO_GRATING)
    };
    // A blazed grating for the middle of the band, protected aluminium:
    // first order near 80% at the blaze, falling toward the ends. That was
    // the declared table; the real one is [`super::realism::grating`].
    let first = table(&[
        (0.36, 0.50),
        (0.40, 0.65),
        (0.45, 0.75),
        (0.50, 0.79),
        (0.55, 0.80),
        (0.60, 0.78),
        (0.65, 0.74),
        (0.70, 0.68),
        (0.72, 0.65),
    ]);
    let zero = table(&[(0.36, 0.20), (0.45, 0.10), (0.55, 0.08), (0.72, 0.14)]);
    // Local X = gx, local Y = the grooves, local Z = the normal.
    let grating_hw = 0.5 * beam_along() / ALPHA_DEG.to_radians().cos() + 8.0;
    surface(
        s,
        frame(p_g, g.n, g.gy),
        rect(grating_hw, 0.5 * BEAM + 10.0),
        match super::realism::grating() {
            Some((te_first, te_zero, tm)) => SurfaceResponse::Grating {
                lines_per_mm: LINES_PER_MM,
                order: 1,
                efficiency: te_first,
                zero_order: te_zero,
                transmissive: false,
                tm: Some(tm),
            },
            None => SurfaceResponse::Grating {
                lines_per_mm: LINES_PER_MM,
                order: 1,
                efficiency: first,
                zero_order: zero,
                transmissive: false,
                tm: None,
            },
        },
        SurfaceResponse::Absorber,
        "Reflection grating, 1800 lines/mm",
    );
    // The grating's mount, black, just behind it: what the lamps send past
    // the grating ends here. It stays below the light rising from M3.
    surface(
        s,
        frame(p_g - g.n * 2.0, g.n, g.gy),
        rect(grating_hw + 20.0, 0.5 * BEAM + 20.0),
        SurfaceResponse::Absorber,
        SurfaceResponse::Absorber,
        "Grating mount (black)",
    );
    // The zero order's dump, far enough out that the first order, 45 degrees
    // or more away and up to 1.55 times as wide as the beam, clears it: at
    // 150 mm it took a quarter of the first order's light, most of the blue.
    let zero_dir = reflect(g.arrive, g.n);
    surface(
        s,
        frame(p_g + zero_dir * knob("ZERO_DUMP_AT", ZERO_DUMP_AT), -zero_dir, y),
        rect(0.5 * BEAM.max(beam_along()) + 30.0, 0.5 * BEAM.max(beam_along()) + 30.0),
        SurfaceResponse::Absorber,
        SurfaceResponse::Absorber,
        "Zero-order dump",
    );

    // 7. The camera mirror, as wide as the fan and as tall as the three
    // slits need.
    let m2_frame = frame(p_g + cam.m2, cam.m2_normal, y);
    let fan = |p: Vec3| m2_frame.inverse_transform_point(p_g + p);
    let m2_hit = |l: f64| {
        let dir = g.diffract(g.arrive, l);
        let c2 = cam.m2 + cam.m2_normal * cam.r;
        sphere_hit(Vec3::ZERO, dir, c2, cam.r, true)
    };
    let widest = fan(m2_hit(BLUE_UM)).x.abs().max(fan(m2_hit(RED_UM)).x.abs());
    let blue_beam = beam_along() * d.beta_blue_deg.to_radians().cos() / ALPHA_DEG.to_radians().cos();
    let m2_hw = widest + 0.5 * blue_beam + 8.0 + knob("M2_EXTRA", 0.0);
    let m2_hh = 0.5 * BEAM + knob("GRATING_AT", GRATING_AT) * cam.r * field / F_COLL + 8.0;
    surface(
        s,
        m2_frame,
        with_s8(asphere_strip(cam.r, d.m2_asph[0], d.m2_asph[1], d.m2_asph[2], m2_hw, m2_hh), knob("M2_S8", 0.0)),
        hr(),
        SurfaceResponse::Absorber,
        &format!("Camera M2, spherical, f {:.0} mm", d.f_camera),
    );

    // 8. The convex fold, local X along the dispersion.
    let m3_frame = frame(p_g + cam.fold, cam.fold_normal, cam.fold_normal.cross(x));
    let gap = (1.0 - knob("FOLD_AT", FOLD_AT)) * d.f_camera;
    let m3_hw = 0.65 * cam.length() + 0.5 * blue_beam * gap / d.f_camera + 8.0;
    let incidence = cam.fold_normal.dot(cam.out).acos();
    let m3_hh = (0.5 * BEAM * gap / d.f_camera + 0.6 * d.table_height() / d.magnification) / incidence.cos() + 5.0 + knob("M3_EXTRA", 0.0);
    surface(
        s,
        m3_frame,
        // `M3_K`, `M3_S8`: M3's conic and r^8 sag at 500 mm, as a joint
        // camera-and-relay design gives them.
        // `M3_UP`, `M3_DOWN`: M3 grown (or trimmed) on its local +Y and -Y
        // sides alone, for the lamps' bands.
        lopsided(with_s8(asphere_strip(-cam.fold_r, knob("M3_K", 0.0), d.m3_asph[0], d.m3_asph[1], m3_hw, m3_hh), knob("M3_S8", 0.0)), knob("M3_UP", 0.0), knob("M3_DOWN", 0.0)),
        hr(),
        SurfaceResponse::Absorber,
        &format!("Fold M3, weakly convex, R {:.0} mm, straightens the spectrum", cam.fold_r),
    );

    // 9. The field lens just before the intermediate spectrum, 10. the
    // projection lens, folded flat under the ceiling. Placed first as if the
    // light ran straight up from M3, then everything past fold F4 is
    // reflected in F4's plane: the lens lies horizontal along +Z, back
    // toward the tube, and F5 past it sends the light down to the table.
    // The field lens is off centre: its centre lies on the line from the
    // camera's pupil to the projection lens's entrance pupil, so it also
    // bends the light, tilted after M3, square onto the lens's axis.
    let mid = p_g + (cam.blue + cam.red) * 0.5;
    // `RELAY_TILT` (degrees): the relay's axis tilted about the spectrum's
    // length against the level intermediate spectrum. The camera sees its
    // focus obliquely (the light leaves M3 leaned) and draws it as a
    // trapezoid, each wavelength's line leaning by an amount growing along
    // the spectrum; a relay tilted the other way draws the opposite
    // trapezoid. `F5_TILT` tilts the light onto the table to keep it in
    // focus (Scheimpflug).
    let up = turn_x(y, knob("RELAY_TILT", 0.0).to_radians());
    let y_field = mid.y - FIELD_BEFORE;
    // The field lens, off centre and just before the image, draws the image
    // toward its own centre: the projection lens's axis goes through the
    // image where it lands, `axis0`.
    let pupil = p_g + cam.pupil;
    let k = 1.0 / (1.0 + FIELD_BEFORE / d.f_field);
    let (mut axis0, mut centre) = (mid, mid);
    for _ in 0..10 {
        let p_ep = axis0 + up * (d.pupil_distance + d.focus_shift);
        centre = pupil + (p_ep - pupil) * ((y_field - pupil.y) / (p_ep.y - pupil.y)) + x * d.fresnel_shift + z * d.fresnel_shift_v;
        let drawn = Vec3::new(centre.x, mid.y, centre.z) + (mid - Vec3::new(centre.x, mid.y, centre.z)) * k;
        axis0 = Vec3::new(drawn.x, mid.y, drawn.z);
    }
    // `FIELD_AT` and `LENS_AT` ("x y z"): the field lens's centre and the
    // projection lens's first vertex (as if F4 were not there) in the relay's
    // frame, x along the spectrum, z up the relay's axis, y = z x x, from the
    // intermediate spectrum's middle: where a relay designed against the
    // camera's own rays puts them.
    let relay_y = up.cross(x).normalize();
    let in_relay = |v: &[f64]| mid + x * v[0] + relay_y * v[1] + up * v[2];
    let triple = |name: &str| -> Option<Vec<f64>> {
        let v: Vec<f64> = std::env::var(name).ok()?.split_whitespace().filter_map(|q| q.parse().ok()).collect();
        (v.len() == 3).then_some(v)
    };
    if let Some(v) = triple("FIELD_AT") {
        centre = in_relay(&v);
    }
    let lens_at = triple("LENS_AT");
    if let Some(v) = &lens_at {
        axis0 = mid + x * v[0] + relay_y * v[1];
    }
    let off = centre - Vec3::new(mid.x, y_field, mid.z);
    let field_hw = 0.5 * cam.length() + off.x.abs() + 15.0;
    let field_hh = 0.5 * d.table_height() / d.magnification + off.z.abs() + 10.0;
    let fresnel_bottom = field_lens(s, centre, up, d, field_hw.hypot(field_hh));

    // The lens's leg runs back toward the tube, so F5 and the light it sends
    // down to the table are clear of the camera mirror M2, which hangs out
    // on the far side.
    let leg = z * knob("LEG_SIGN", 1.0);
    let p4 = axis0 + up * knob("F4_AFTER", F4_AFTER);
    let n4 = (leg - up).normalize();
    let unfold = |p: Vec3| p - n4 * (2.0 * (p - p4).dot(n4));
    // The beam at F4: the intermediate spectrum and its cone.
    let f4_hh = 0.5 * cam.length() + 20.0 + knob("F4_EXTRA", 0.0);
    let f4_hw = (0.5 * d.table_height() / d.magnification + 12.0 + knob("F4_EXTRA_W", 0.0)) * std::f64::consts::SQRT_2;
    if d.f4_curv == 0.0 {
        fold(s, p4, up, leg, x, f4_hw, f4_hh, "Fold F4, the light horizontal into the projection lens");
    } else {
        let n = (leg - up).normalize();
        surface(
            s,
            frame(p4, n, x),
            asphere_strip(1.0 / d.f4_curv, 0.0, 0.0, 0.0, f4_hw, f4_hh),
            hr(),
            SurfaceResponse::Absorber,
            &format!("Fold F4, spherical R {:.0} mm, the light horizontal into the projection lens", 1.0 / d.f4_curv),
        );
    }
    let p_vertex = match &lens_at {
        Some(v) => unfold(in_relay(v)),
        None => unfold(axis0 + up * (d.vertex_distance + d.focus_shift)),
    };
    // `LENS_TILT="about_x about_y"` (radians): the lens's axis tilted in the
    // relay's frame, about x (along the spectrum) and y, then folded by F4.
    let leg = match std::env::var("LENS_TILT").ok().map(|v| v.split_whitespace().filter_map(|q| q.parse::<f64>().ok()).collect::<Vec<_>>()) {
        Some(v) if v.len() == 2 => {
            let a = x * v[1].sin() - relay_y * (v[0].sin() * v[1].cos()) + up * (v[0].cos() * v[1].cos());
            let a = a.normalize();
            (a - n4 * (2.0 * a.dot(n4))).normalize()
        }
        _ => leg,
    };
    // `IDEAL_PROJ`: an ideal thin lens instead, for comparison, with the same
    // conjugates: object and image distances each stretched by their share
    // of the principal planes' gap, and the focal length with them, so the
    // magnification and the table stay where they are.
    let stop_probe = if std::env::var_os("IDEAL_PROJ").is_some() {
        let lens = ProjLens::new();
        let (u, v) = (d.object_distance, d.throw);
        let k = 1.0 + (lens.h2 - lens.h1) / (u + v);
        let at = unfold(axis0 + up * (u * k + d.focus_shift));
        let ideal = SurfaceResponse::ThinLens { focal_length_mm: u * v / (u + v) * k };
        surface(
            s,
            frame(at, leg, x),
            SurfaceGeometry3::Plane { shape: PlanarShape2::Disc { radius_mm: 260.0 } },
            ideal.clone(),
            ideal,
            "Projection lens (ideal thin lens, comparison)",
        );
        None
    } else {
        projection_lens(s, p_vertex, leg, x, d.stop_probe)
    };
    // F5 past the lens: the beam fans out from the lens's exit pupil, taken
    // as a third of the lens's length inside its last vertex, over the
    // table's length and height.
    let p5 = p_vertex + leg * (d.lens_length + F5_GAP);
    let reach = F5_GAP + d.lens_length / 3.0;
    let exit = d.exit_radius;
    let f5_hh = reach * (0.5 * SPECTRUM_LENGTH + 50.0) / d.throw + exit + 10.0 + knob("F5_EXTRA", 0.0);
    // Across the bands the beam fills only the camera's cone of the lens's.
    let exit_v = exit * (d.lens_takes_f_number / d.camera_f_number).min(1.0) * 1.1;
    let f5_hw = (reach * (0.5 * d.table_height() + 30.0) / d.throw + exit_v + 10.0 + knob("F5_EXTRA", 0.0)) * std::f64::consts::SQRT_2;
    // `TABLE_UP`: F5 sends the light up and the table lies as far above it
    // as it would below. The room is this model upside down: the table on
    // the floor side, the projection lens its nearest optic, the camera
    // beyond it.
    let to_table = turn_x(-table_normal(), knob("F5_TILT", 0.0).to_radians());
    fold(s, p5, leg, to_table, x, f5_hw, f5_hh, if to_table.y > 0.0 { "Fold F5, the light to the table (model upside down)" } else { "Fold F5, the light down to the table" });

    // How the folded path sits in the room.
    let semi = d.lens_slab;
    let m3_up = cam.fold_normal.cross(x).normalize();
    let clearances = vec![
        ("M3's top edge", p_g.y + cam.fold.y + m3_hh * m3_up.y.abs()),
        ("the field lens's lowest point", fresnel_bottom),
        ("F4's lowest point", p4.y - 0.5 * f4_hw * std::f64::consts::SQRT_2),
        ("the projection lens's top (glass, slab-cut)", p_vertex.y + semi),
        ("the projection lens's bottom (glass, slab-cut)", p_vertex.y - semi),
        ("F5's lowest point, the opening", p5.y - 0.5 * f5_hw * std::f64::consts::SQRT_2),
    ];

    let table_y = if to_table.y > 0.0 { 2.0 * p5.y - TABLE_Y } else { TABLE_Y };
    // Where the lens's axis meets the table.
    let table_centre = p5 + to_table * ((table_y - p5.y) / to_table.y);
    if std::env::var_os("RELAY_DIAG").is_some() {
        eprintln!(
            "RELAYDIAG mid {:.1?} p4 {:.1?} vertex {:.1?} leg {:.4?} p5 {:.1?} to_table {:.4?} table_y {:.1} centre {:.1?}; lens length {:.1}; F5 to table {:.1} mm",
            mid, p4, p_vertex, leg, p5, to_table, table_y, table_centre, d.lens_length, (table_centre - p5).length()
        );
    }
    let half_v = 0.5 * d.table_height() + 60.0;
    let table = detector(
        s,
        table_centre + table_normal() * 0.1,
        table_normal(),
        x,
        TABLE_HALF_LENGTH,
        half_v,
        // While fitting, one cell: the fits read the hits, and a trace
        // builds every detector's readout once per worker.
        if d.probe_table { [1, 1] } else { [(2.0 * TABLE_HALF_LENGTH * knob("TABLE_PX_PER_MM", 1.0)) as usize, (half_v as usize)] },
        DetectorTermination::Terminate,
        "Table: the three spectra",
    );
    // What a visitor sees: lux, and each 1 x 2 mm pixel's colour.
    if let Some(det) = s.detectors.iter_mut().find(|t| t.common.id.0 == table.0) {
        det.measure = oc_core::scene3::DetectorMeasure::Illuminance;
    }
    // Turn what was built past F1.
    for o in s.bodies.iter_mut().skip(built.0) {
        o.common.transform = turned.compose(o.common.transform);
    }
    for o in s.surfaces.iter_mut().skip(built.1) {
        o.common.transform = turned.compose(o.common.transform);
    }
    for o in s.apertures.iter_mut().skip(built.2) {
        o.common.transform = turned.compose(o.common.transform);
    }
    for o in s.detectors.iter_mut().skip(built.3) {
        o.common.transform = turned.compose(o.common.transform);
    }
    let stop_probe = stop_probe.map(|(id, c, a)| (id, turned.transform_point(c), turned.rotate(a)));
    let table_centre = turned.transform_point(table_centre);
    let mid = turned.transform_point(mid);
    Placed {
        table,
        entrance,
        table_centre,
        intermediate: mid,
        lamp_slit,
        lamp_run,
        lamp_across,
        lamp_dir: d1,
        clearances,
        stop_probe,
        slit_turn: d.slit_turn,
        field_centre: turned.transform_point(centre),
        lens_axis0: turned.transform_point(axis0),
        lens_vertex_unfolded: turned.transform_point(axis0 + up * (d.vertex_distance + d.focus_shift)),
    }
}

/// The `FRESNEL_A` term `i` (r^4, r^6, r^8), if given.
fn fresnel_a(i: usize) -> Option<f64> {
    let v: Vec<f64> = std::env::var("FRESNEL_A").ok()?.split_whitespace().filter_map(|q| q.parse().ok()).collect();
    v.get(i).copied()
}

/// The field lens: two PMMA Fresnels face to face, as an overhead
/// projector's stage lens is, each of twice the focal length so each facet
/// bends the light half as far. Paraboloidal bases (conic -1), grooves
/// 0.5 mm apart, 2 mm substrates, centred at `centre` with the light along
/// `axis`. Returns the pair's lowest point.
fn field_lens(s: &mut Scene3, centre: Vec3, axis: Vec3, d: &Design, radius: f64) -> f64 {
    let f = d.f_field;
    use oc_core::fresnel_lens::{FresnelFacing, FresnelGrooves, FresnelLens1};
    // The light pipe's PMMA: the catalogue holds one glass per name, so a
    // second "PMMA" here would replace the 1 m Fresnel's.
    let glass = super::realism::pmma();
    let n = glass.index(0.55);
    let pmma = s.catalog.put(&glass);
    let clear = radius + 5.0;
    if std::env::var_os("IDEAL_FIELD").is_some() {
        let lens = SurfaceResponse::ThinLens { focal_length_mm: f };
        surface(s, frame(centre, axis, Vec3::Z), SurfaceGeometry3::Plane { shape: PlanarShape2::Disc { radius_mm: clear } }, lens.clone(), lens, "Field lens (ideal, comparison)");
        return centre.y;
    }
    for (i, (facing, at)) in [(FresnelFacing::PlusX, -6.0), (FresnelFacing::MinusX, 6.0)].into_iter().enumerate() {
        let lens = FresnelLens1 {
            // `FRESNEL_R`, `FRESNEL_K`, `FRESNEL_A="r^4 r^6 r^8"`: each
            // Fresnel's profile as a relay design gives it, in mm.
            radius_mm: knob("FRESNEL_R", d.fresnel_scale * (n - 1.0) * 2.0 * f),
            conic: knob("FRESNEL_K", d.fresnel_conic),
            aspheric: [
                fresnel_a(0).unwrap_or(d.fresnel_asph[0] / 500f64.powi(4)),
                fresnel_a(1).unwrap_or(d.fresnel_asph[1] / 500f64.powi(6)),
                fresnel_a(2).unwrap_or(d.fresnel_asph[2] / 500f64.powi(8)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            clear_radius_mm: clear,
            grooves: FresnelGrooves::ConstantPitch { pitch_mm: d.fresnel_pitch },
            substrate_mm: 2.0,
            facing,
            draft_deg: if super::realism::on("DRAFT") { 3.0 } else { 0.0 },
        };
        let pos = centre + axis * at;
        let id = s.add_fresnel_lens(pos, 0.0, &lens, Material::Catalog(pmma)).expect("the field Fresnel builds");
        super::realism::absorb(s, id, "pmma");
        // Both faces carry the lens's broadband AR coat.
        let ar = oc_core::thinfilm::CoatingDesign {
            kind: oc_core::thinfilm::DesignKind::BroadbandArLegacy,
            lambda_nm: 550.0,
            aoi_deg: 0.0,
            periods: 1,
            n_high: 2.35,
            n_low: 1.38,
            adapt: None,
        };
        let film = oc_core::thinfilm::film_coating(ar, 1.0, n);
        if let Some(body) = s.bodies.iter_mut().find(|q| q.common.id.0 == id) {
            body.surface.default = oc_core::scene3::TwoSidedSurfaceResponse {
                from_inside: SurfaceResponse::Coating {
                    coating: film.seen_from_behind(),
                },
                from_outside: SurfaceResponse::Coating { coating: film },
            };
            body.common.transform = frame_x(pos, axis, Vec3::Z);
            body.common.label = format!(
                "Field lens, Fresnel {} of 2, PMMA, f {:.0} mm (the pair f {f:.0}), {:.0} mm across",
                i + 1,
                2.0 * f,
                2.0 * clear
            );
        }
    }
    // Horizontal discs: the lowest point is the lower substrate's back.
    centre.y - (axis.y.abs() * 3.0 + clear * (1.0 - axis.y * axis.y).sqrt())
}

/// The projection lens, built from its prescription with its first vertex
/// at `first_vertex`, the light running along `axis`: the glass bodies, the
/// diaphragm, and a black barrel end at each vertex so nothing passes
/// beside the glass.
fn projection_lens(s: &mut Scene3, first_vertex: Vec3, axis: Vec3, up: Vec3, probe: bool) -> Option<(ObjectId, Vec3, Vec3)> {
    use oc_core::seq::{FieldType, SeqAperture, SeqSurface, SeqSystem};
    let lens = ProjLens::new();
    let n = lens.surfaces.len();
    let seq = SeqSystem {
        surfaces: lens
            .surfaces
            .iter()
            .enumerate()
            .map(|(i, (r, t, semi, m))| SeqSurface {
                radius: *r,
                conic: lens.asph[i].0,
                thickness: if i == n - 1 { 100.0 } else { *t },
                semi_aperture: *semi,
                coefficients: lens.asph[i].1.clone(),
                material: m.clone(),
                reflective: false,
            })
            .collect(),
        stop: Some(lens.stop),
        aperture: SeqAperture::Epd { value: 10.0 },
        fields: vec![0.0],
        field_type: FieldType::Angle,
        object_distance: None,
        object_space_telecentric: false,
        image_semi_aperture: None,
        aimed_beams: Vec::new(),
        wavelengths: vec![0.587_56],
        notes: Vec::new(),
    };
    let built = oc_core::seq_import::import_seq(&seq, &Default::default()).unwrap_or_else(|e| panic!("{e}"));
    // The prescription's axis is +X from its first vertex.
    let place = frame_x(first_vertex, axis, up);
    // Fresh ids for every body and face, one per face even where two
    // cemented bodies share it.
    let mut patches = std::collections::HashMap::new();
    let mut patch = |s: &mut Scene3, old: oc_core::id::PatchId| *patches.entry(old.0).or_insert_with(|| oc_core::id::PatchId(s.alloc_object_id().0));
    // Where each surface's vertex is, and whether air meets it: those faces
    // carry a broadband AR coat, the multicoating Schneider sold these lenses
    // with; the cemented faces are bare.
    let mut at = 0.0;
    let mut air_faces = Vec::new();
    let mut before_air = true;
    for (r, t, _, m) in &lens.surfaces {
        let after_air = matches!(m, oc_core::seq::SeqMaterial::Air);
        if r.is_some() && (before_air || after_air) {
            air_faces.push(at);
        }
        at += t;
        before_air = after_air;
    }
    let ar = oc_core::thinfilm::CoatingDesign {
        kind: oc_core::thinfilm::DesignKind::BroadbandArLegacy,
        lambda_nm: 550.0,
        aoi_deg: 0.0,
        periods: 1,
        n_high: 2.35,
        n_low: 1.38,
        adapt: None,
    };
    let catalog = &built.scene.catalog;
    for (k, mut b) in built.scene.bodies.clone().into_iter().enumerate() {
        let n_glass = b.material.index_at(0.55, catalog).unwrap_or(1.5);
        if let oc_core::scene3::SolidGeometry3::Profile(p) = &b.geometry {
            let centre = b.common.transform.pos.x;
            for e in &p.profile.edges {
                let vertex = match &e.curve {
                    oc_core::profile2::Curve2::Conic(c) => Some(c.vertex.x),
                    oc_core::profile2::Curve2::EvenAsphere { base, .. } => Some(base.vertex.x),
                    _ => None,
                };
                if let Some(v) = vertex {
                    if air_faces.iter().any(|z| (v + centre - z).abs() < 1e-6) {
                        let film = oc_core::thinfilm::film_coating(ar, 1.0, n_glass);
                        b.surface.overrides.push(oc_core::scene3::PatchResponseOverride {
                            patch: e.patch,
                            response: oc_core::scene3::TwoSidedSurfaceResponse {
                                from_inside: SurfaceResponse::Coating {
                                    coating: film.seen_from_behind(),
                                },
                                from_outside: SurfaceResponse::Coating { coating: film },
                            },
                        });
                    }
                }
            }
        }
        if let oc_core::scene3::SolidGeometry3::Profile(p) = &mut b.geometry {
            for e in &mut p.profile.edges {
                e.patch = patch(s, e.patch);
            }
        }
        for o in &mut b.surface.overrides {
            o.patch = patch(s, o.patch);
        }
        if let Material::Catalog(g) = b.material {
            let glass = built.scene.catalog.get(g).expect("the lens's glass").clone();
            // Schott's internal transmittance for the lens's glass.
            for name in ["N-SK16", "N-F2", "N-BK7"] {
                if glass.name.contains(name) {
                    b.absorption = super::realism::absorption(name);
                }
            }
            b.material = Material::Catalog(s.catalog.put(&glass));
        }
        b.common.id = s.alloc_object_id();
        b.common.transform = place.compose(b.common.transform);
        b.common.label = format!("Projection lens, {PROJ_NAME}, element {}", k + 1);
        s.bodies.push(b);
    }
    // `LENS_SLAB_CUT=1`: the round blanks cut to the slab, as the glass would
    // be made. The tracer resolves a medium by the highest-priority body that
    // holds the point, so a block of air-index glass above the slab and one
    // below, over the lens's whole length, take the glass away there; their
    // inner faces are the slab's flat cut faces.
    if std::env::var_os("LENS_SLAB_CUT").is_some() {
        let reach = lens.surfaces.iter().map(|q| q.2).fold(0.0, f64::max) + 10.0;
        for side in [1.0, -1.0] {
            let id = s.alloc_object_id();
            let mut common = ObjectCommon::new(id, LayerId(0));
            common.transform = place.compose(Isometry3::from_pos(Vec3::new(0.5 * lens.length, 0.0, side * (lens.slab + 0.5 * reach))));
            common.label = format!("Projection lens, slab cut {}", if side > 0.0 { "one side" } else { "other side" });
            s.bodies.push(oc_core::scene3::Body3 {
                common,
                geometry: oc_core::scene3::SolidGeometry3::Box { half_extents_mm: Vec3::new(0.5 * lens.length + 30.0, reach, 0.5 * reach) },
                material: Material::AbbePoint { nd: 1.0, vd: 1e9 },
                absorption: oc_core::scene3::AbsorptionModel::None,
                surface: oc_core::scene3::SurfaceProps3::default(),
                priority: 1,
            });
        }
    }
    for mut a in built.scene.apertures {
        a.common.id = s.alloc_object_id();
        a.common.transform = place.compose(a.common.transform);
        a.primary_stop_for.clear();
        a.common.label = format!("Projection lens diaphragm, {:.1} mm across", 2.0 * lens.stop_radius);
        // Slab-cut, the diaphragm's round mask would hang below the glass
        // into the camera's light: a thin ring only.
        if std::env::var_os("LENS_SLAB_CUT").is_some() {
            if let PlanarShape2::Disc { radius_mm } = a.shape.clear_opening {
                a.shape.outer_extent = Some(PlanarShape2::Disc { radius_mm: radius_mm + 2.0 });
            }
        }
        s.apertures.push(a);
    }
    let first = lens.surfaces[0].2;
    let last = lens.surfaces[n - 1].2;
    for (at, r, name) in [(-1.0, first, "top"), (lens.length + 1.0, last, "bottom")] {
        let id = s.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = frame(first_vertex + axis * at, axis, up);
        common.label = format!("Projection lens barrel, {name}");
        s.apertures.push(Aperture3 {
            common,
            // The slab: `up` (local Y) runs along the spectrum, local X
            // across the bands, where the glass is cut.
            shape: ApertureShape3 {
                plane_local: Plane3::XY,
                clear_opening: PlanarShape2::Rectangle {
                    half_width_mm: lens.slab,
                    half_height_mm: r,
                },
                // The black ring only round the slab: a mask as wide as the
                // glass is tall would hang into the camera's rising light.
                outer_extent: Some(PlanarShape2::Rectangle {
                    half_width_mm: lens.slab + 10.0,
                    half_height_mm: r + 10.0,
                }),
            },
            interaction: ApertureInteraction::Absorb,
            primary_stop_for: Vec::new(),
        });
    }
    probe.then(|| {
        let at = first_vertex + axis * lens.surfaces[..lens.stop].iter().map(|q| q.1).sum::<f64>();
        let id = detector(
            s,
            at - axis * 0.01,
            axis,
            up,
            lens.stop_radius * 3.0,
            lens.stop_radius * 3.0,
            [1, 1],
            DetectorTermination::ProbeAndContinue,
            "Diaphragm probe",
        );
        (id, at, axis)
    })
}

/// A hard-coated UV/IR cut filter, as its datasheet gives it: 95% from 375
/// to 690 nm, edges 11-13 nm wide. A table, so blind to the angle, which here
/// costs nothing: the filter sits in an f/8.4 cone, and 3.4 degrees moves a
/// dielectric edge by about 0.4 nm.
fn uv_ir_cut() -> SurfaceResponse {
    SurfaceResponse::ExplicitRt {
        reflectance: SpectralScalar::Constant(0.0),
        transmittance: table(&[
            (0.350, 0.0),
            (0.364, 0.0),
            (0.375, FILTER_T),
            (0.690, FILTER_T),
            (0.703, 0.0),
            (0.720, 0.0),
        ]),
    }
}

/// N-BK7, Schott's catalog Sellmeier.
fn n_bk7() -> Glass {
    Glass::sellmeier(
        "N-BK7",
        Sellmeier {
            b: [1.039_612_12, 0.231_792_344, 1.010_469_45],
            c: [0.006_000_698_67, 0.020_017_914_4, 103.560_653],
        },
    )
}

/// A lamp arm: the lamp, its condenser, a UV/IR cut filter and the curved
/// slit. Köhler illumination: the lamp sits at the condenser's focus, so
/// each point of the lamp leaves as a parallel beam and the slit is lit
/// evenly along its length, every point of it seeing the whole lamp.
fn lamp_arm(s: &mut Scene3, lamp: &Lamp, ls: &LampSlit, glass: oc_core::material::GlassId) -> Vec<SourceId> {
    // x along the slit, y across it, z the way its light leaves.
    let (x, y, z) = (ls.run, ls.across, ls.dir);
    let (u_lo, u_hi) = ls.line.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (w_lo, w_hi) = ls.line.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let (u_mid, w_mid) = (0.5 * (u_lo + u_hi), 0.5 * (w_lo + w_hi));
    let hw = 0.5 * lamp.slit_width;

    // The slit: a black plate with the curved opening.
    let mut vertices: Vec<oc_core::vec::Vec2> = ls.line.iter().map(|p| oc_core::vec::Vec2::new(p.0, p.1 + hw)).collect();
    vertices.extend(ls.line.iter().rev().map(|p| oc_core::vec::Vec2::new(p.0, p.1 - hw)));
    // The plate is tilted about its width to follow the focus: its local X
    // runs along the slit and away from the lamp by b per millimetre.
    let (a, b) = ls.depth;
    let along = (x + z * b).normalize();
    let stretch = (1.0 + b * b).sqrt();
    let vertices: Vec<oc_core::vec::Vec2> = vertices.iter().map(|v| oc_core::vec::Vec2::new(v.x * stretch, v.y)).collect();
    let (u_lo, u_hi) = (u_lo * stretch, u_hi * stretch);
    let id = s.alloc_object_id();
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = frame(ls.centre + z * a, along.cross(y), y);
    common.label = format!("{} slit, curved, {:.0} x {} mm", lamp.name.split(',').next().unwrap_or(lamp.name), u_hi - u_lo, lamp.slit_width);
    s.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Polygon { vertices },
            // Over the fold's whole beam, so no light passes beside the
            // slit, and short on +y: the sun's condenser beam passes the slit
            // there, some 50 mm off.
            outer_extent: Some({
                let (ua, ub) = (-(u_lo.abs().max(u_hi.abs()) + 12.0), u_lo.abs().max(u_hi.abs()) + 12.0);
                let (va, vb) = (w_lo - hw - 25.0, w_hi + hw + 8.0);
                PlanarShape2::Polygon {
                    vertices: vec![oc_core::vec::Vec2::new(ua, va), oc_core::vec::Vec2::new(ub, va), oc_core::vec::Vec2::new(ub, vb), oc_core::vec::Vec2::new(ua, vb)],
                }
            }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });

    // A stop 100 mm past the slit, toward the periscope, before the lamp's
    // and the sun's cones meet: the slit's own curve, widened by what the
    // collimator's cone spreads over 100 mm, so it passes the cone the
    // collimator takes from every point of the slit and stops what the lamp
    // sends wider. The slit's chief rays run along +Z, so the curve is the
    // same there. A rectangle round the whole bow would pass 2.5 times the
    // cone across the slit near its middle, which the camera turns into
    // blur at the blue end.
    let reach = 100.0;
    let spread = reach * 0.5 * BEAM / F_COLL;
    let (first, last) = (ls.line[0].0, ls.line[ls.line.len() - 1].0);
    let out_u = |u: f64| if u == first { u - spread * (last - first).signum() } else if u == last { u + spread * (last - first).signum() } else { u };
    let mut stop: Vec<oc_core::vec::Vec2> = ls.line.iter().map(|p| oc_core::vec::Vec2::new(out_u(p.0), p.1 + hw + spread)).collect();
    stop.extend(ls.line.iter().rev().map(|p| oc_core::vec::Vec2::new(out_u(p.0), p.1 - hw - spread)));
    let sid = s.alloc_object_id();
    let mut common = ObjectCommon::new(sid, LayerId(0));
    common.transform = frame(ls.centre + z * (a + b * u_mid + reach), z, y);
    common.label = format!("{} arm stop, f/{:.1}", lamp.name.split(',').next().unwrap_or("lamp"), F_COLL / BEAM);
    s.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Polygon { vertices: stop },
            outer_extent: Some(PlanarShape2::Rectangle {
                half_width_mm: u_lo.abs().max(u_hi.abs()) / stretch + spread + 8.0,
                half_height_mm: w_lo.abs().max(w_hi.abs()) + hw + spread + 15.0,
            }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });

    // Behind the slit a flat turns the arm to run along -y, away from the
    // sun's condenser beam, which passes the slit on its +y side and crosses
    // behind it farther back; the lens, the filter and the lamp sit on that
    // leg. The lens: plano-convex, the hyperbolic face (conic -n²) toward the
    // lamp, which takes light from its focus to parallel inside the glass
    // with no spherical aberration at 550 nm. For a spectral lamp it is the
    // condenser, the lamp at its focus. Behind a reflector it is the field
    // lens: it images the reflector to infinity, so every point of the slit
    // sends its cone along the collimator's chief ray.
    let slit_mid = ls.centre + x * u_mid + y * w_mid + z * (a + b * u_mid);
    let lamp_short = lamp.name.split(',').next().unwrap_or("lamp");
    let fold_back = knob("LAMP_FOLD", 35.0);
    let fold_at = slit_mid - z * fold_back;
    let across = knob("LAMP_MASK_W", hw + 8.0);
    let along = 0.5 * (u_hi - u_lo) / stretch + 10.0;
    surface(
        s,
        frame(fold_at, (z - y).normalize(), x),
        rect(across * std::f64::consts::SQRT_2 + 3.0, along),
        hr(),
        SurfaceResponse::Absorber,
        &format!("{lamp_short} arm fold, flat"),
    );
    let lens_gap = knob("LAMP_LENS", 60.0);
    let flat = fold_at - y * lens_gap;
    // From the slit to the lens along the light.
    let to_lens = fold_back + lens_gap;
    let n = n_bk7().index(0.55);
    let rr = lamp.condenser_semi;
    // The reflector for a lamp that has one: the coil `throw` from the slit
    // along the light, the ellipsoid's far focus where the field lens images
    // it onto the slit, its rim in the coil's plane (its semi-latus rectum)
    // sized so the cone reaches the slit at `na`.
    let ellipse = lamp.reflector.map(|(throw, na)| {
        let to_coil = throw - to_lens;
        let (mut depth, mut out) = (10.0, (0.0, 0.0, 0.0, 0.0));
        for _ in 0..4 {
            let f = to_coil + 0.5 * depth;
            let virt = to_lens * f / (f - to_lens);
            let c = 0.5 * (to_coil + virt);
            let rim = na * 2.0 * c * to_lens / virt;
            let ea = 0.5 * (rim + (rim * rim + 4.0 * c * c).sqrt());
            depth = ea - c;
            out = (rim, depth, -(c / ea).powi(2), f);
        }
        (flat - y * to_coil, out.0, out.1, out.2, out.3)
    });
    let f = ellipse.map_or(lamp.condenser_f, |e| e.4);
    let r = f * (n - 1.0);
    let k = -n * n;
    let c = 1.0 / r;
    let sag = c * rr * rr / (1.0 + (1.0 - (1.0 + k) * c * c * rr * rr).sqrt());
    let t = sag + 4.0;
    let lens = s.add_singlet(
        flat - y * (0.5 * t),
        SingletGeom {
            r1_mm: r,
            r2_mm: f64::INFINITY,
            thickness_mm: t,
            semi_aperture_mm: rr,
            conic1: k,
            conic2: 0.0,
        },
        1.5168,
        64.17,
    );
    if let Some(b) = s.bodies.iter_mut().find(|b| b.common.id.0 == lens.0) {
        b.common.transform = frame_x(flat - y * (0.5 * t), y, z);
        b.material = Material::Catalog(glass);
        b.absorption = super::realism::absorption("N-BK7");
        let role = if ellipse.is_some() { "Field lens" } else { "Condenser" };
        b.common.label = format!("{role}, N-BK7 plano-hyperbolic, f {f:.0} mm, {:.0} mm across, {lamp_short}", 2.0 * rr);
    }
    // Both faces broadband AR coated, as the projection lens's air faces
    // (bare, 4% lost at each). `LAMP_AR=0` leaves them bare.
    if knob("LAMP_AR", 1.0) > 0.5 {
        let ar = oc_core::thinfilm::CoatingDesign {
            kind: oc_core::thinfilm::DesignKind::BroadbandArLegacy,
            lambda_nm: 550.0,
            aoi_deg: 0.0,
            periods: 1,
            n_high: 2.35,
            n_low: 1.38,
            adapt: None,
        };
        for face in [oc_core::build3::BodyFace::Front, oc_core::build3::BodyFace::Back] {
            assert!(s.set_body_face_coating(lens.0, face, Some(ar)), "the lamp lens's AR coat");
        }
        if let Some(b) = s.bodies.iter_mut().find(|b| b.common.id.0 == lens.0) {
            b.common.label.push_str(", broadband AR");
        }
    }
    let filter = uv_ir_cut();
    surface(
        s,
        frame(flat - y * (t + 10.0), y, x),
        rect(rr, rr),
        filter.clone(),
        filter,
        &format!("UV/IR cut filter, {lamp_short}"),
    );

    if let Some((coil, rim, depth, conic, _)) = ellipse {
        // The mask at the housing's mouth: only light that can reach the
        // slit goes on (the coil's image overfills the slit across three
        // times).
        let mouth = flat - y * (t + 25.0);
        let id = s.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = frame(mouth, y, x);
        common.label = format!("{lamp_short} arm mask, {:.0} x {:.0} mm", 2.0 * along, 2.0 * across);
        let hr = knob("HOUSING_R", 45.0);
        s.apertures.push(Aperture3 {
            common,
            shape: ApertureShape3 {
                plane_local: Plane3::XY,
                clear_opening: PlanarShape2::Rectangle {
                    half_width_mm: across,
                    half_height_mm: along,
                },
                outer_extent: Some(PlanarShape2::Rectangle {
                    half_width_mm: hr + 4.0,
                    half_height_mm: hr + 4.0,
                }),
            },
            interaction: ApertureInteraction::Absorb,
            primary_stop_for: Vec::new(),
        });
        return reflector_lamp(s, lamp, coil, rim, depth, conic, mouth - y * 1.0, x, z, y);
    }

    // The lamp at the condenser's focus. Only the cone the condenser takes
    // is traced; a Lambertian emitter's radiance is the same in every
    // direction, so the cone carries pi sin^2 of its half angle.
    let lamp_at = flat - y * (t + lamp.condenser_f);
    let half = ((rr + 0.5 * lamp.along.max(lamp.across)) / lamp.condenser_f).atan();
    let area_m2 = lamp.across * lamp.along * 1e-6;
    let power: f64 = lamp.radiance.iter().map(|b| b.1).sum();
    let sid = s.alloc_source_id();
    s.sources.push(Source3 {
        id: sid,
        // Local X along the light, local Y along the slit: the rectangle's
        // width is along the slit, its height across.
        transform: frame_x(lamp_at, y, x),
        spatial: SpatialDistribution3::Rectangle {
            half_width_mm: 0.5 * lamp.along,
            half_height_mm: 0.5 * lamp.across,
        },
        angular: AngularDistribution3::UniformConeSolidAngle { half_angle_rad: half },
        spectrum: Spectrum::Continuum(lamp.radiance.clone()),
        flux: power * area_m2 * std::f64::consts::PI * half.sin().powi(2),
        samples: SourceSampling {
            spatial_samples: 2048,
            angular_samples: 2,
            seed: 7 + sid.0,
            sequence: SamplingSequence::Sobol,
        },
        coherence: None,
        polarization: None,
    });
    vec![sid]
}

/// The coil in its ellipsoidal cold mirror, and the black housing from the
/// mirror to `lens_back` (the arm's mask), so no light leaves but through
/// it. `z` runs along the mirror's axis toward the slit. The coil is a 12-sided prism along the
/// slit (`x`), each face a Lambertian rectangle: from any direction it shows
/// its true outline, which a flat emitter would not. It radiates all round;
/// its front half goes straight on and mostly lands on the slit's plate.
#[allow(clippy::too_many_arguments)]
fn reflector_lamp(s: &mut Scene3, lamp: &Lamp, coil: Vec3, rim: f64, depth: f64, conic: f64, lens_back: Vec3, x: Vec3, y: Vec3, z: Vec3) -> Vec<SourceId> {
    const FACES: usize = 12;
    let short = lamp.name.split(',').next().unwrap_or("lamp");
    // The lamp's neck passes through the mirror's vertex.
    let neck = 6.0;
    let geometry = SurfaceGeometry3::SurfaceOfRevolution {
        sag: SagModel::Conic {
            curvature_per_mm: 1.0 / rim,
            conic,
        },
        radial_aperture: RadialAperture {
            inner_radius_mm: neck,
            outer_radius_mm: rim,
        },
        outline: None,
    };
    surface(
        s,
        frame(coil - z * depth, z, y),
        geometry,
        super::realism::mirror(super::realism::enhanced_aluminium(), knob("COLD_MIRROR_R", 0.95)),
        SurfaceResponse::Absorber,
        &format!("{short} ellipsoidal cold mirror, {:.0} mm across, foci the coil and the slit", 2.0 * rim),
    );
    // The coil and its supports shade the returning light near the axis.
    let r_coil = 0.5 * lamp.across;
    surface(
        s,
        frame(coil + z * (r_coil + 0.3), z, y),
        SurfaceGeometry3::Plane {
            shape: PlanarShape2::Disc { radius_mm: 3.0 },
        },
        SurfaceResponse::Absorber,
        SurfaceResponse::Absorber,
        &format!("{short} coil and supports, shade"),
    );
    // The housing: black inside and out, closed behind the mirror.
    let (from, to) = (coil - z * (depth + 3.0), lens_back - z * 1.0);
    let inner = knob("HOUSING_R", 45.0);
    let id = s
        .add_tube(
            from + (to - from) * 0.5,
            0.0,
            oc_core::tube::TubeDims {
                inner_diameter_mm: 2.0 * inner,
                outer_diameter_mm: 2.0 * inner + 4.0,
                length_mm: (to - from).length(),
            },
            oc_core::tube::BoreFinish::Black,
        )
        .expect("housing");
    if let Some(b) = s.bodies.iter_mut().find(|b| b.common.id.0 == id) {
        b.common.transform = frame_x(from + (to - from) * 0.5, z, y);
        b.common.label = format!("{short} housing, black, {:.0} mm long", (to - from).length());
    }
    surface(
        s,
        frame(from, z, y),
        SurfaceGeometry3::Plane {
            shape: PlanarShape2::Disc { radius_mm: inner + 2.0 },
        },
        SurfaceResponse::Absorber,
        SurfaceResponse::Absorber,
        &format!("{short} housing, back"),
    );
    let power: f64 = lamp.radiance.iter().map(|b| b.1).sum();
    let chord = lamp.across * (std::f64::consts::PI / FACES as f64).sin();
    let apothem = r_coil * (std::f64::consts::PI / FACES as f64).cos();
    (0..FACES)
        .map(|i| {
            let ang = std::f64::consts::TAU * (i as f64 + 0.5) / FACES as f64;
            let nrm = y * ang.cos() + z * ang.sin();
            let sid = s.alloc_source_id();
            s.sources.push(Source3 {
                id: sid,
                transform: frame_x(coil + nrm * apothem, nrm, x),
                spatial: SpatialDistribution3::Rectangle {
                    half_width_mm: 0.5 * lamp.along,
                    half_height_mm: 0.5 * chord,
                },
                angular: AngularDistribution3::LambertianHemisphere,
                spectrum: Spectrum::Continuum(lamp.radiance.clone()),
                flux: power * lamp.along * chord * 1e-6 * std::f64::consts::PI,
                samples: SourceSampling {
                    // Twice the condenser lamp's rays over the whole coil:
                    // half of them leave forward, most of those are lost.
                    spatial_samples: (2 * 2048 / FACES) as u32,
                    angular_samples: 2,
                    seed: 7 + sid.0,
                    sequence: SamplingSequence::Sobol,
                },
                coherence: None,
                polarization: None,
            });
            sid
        })
        .collect()
}

struct Timing(std::time::Instant, std::time::Instant);
impl Drop for Timing {
    fn drop(&mut self) {
        if std::env::var_os("TIMING").is_some() {
            eprintln!("TIMING build {:.0?}, probes {:.0?}", self.1 - self.0, self.1.elapsed());
        }
    }
}

/// Nelder-Mead from `x0` with first steps `step`, `iters` rounds.
fn nelder_mead<const N: usize>(f: impl Fn([f64; N]) -> f64, x0: [f64; N], step: [f64; N], iters: usize) -> [f64; N] {
    let mut pts: Vec<[f64; N]> = vec![x0];
    for i in 0..N {
        let mut p = x0;
        p[i] += step[i];
        pts.push(p);
    }
    let mut val: Vec<f64> = pts.iter().map(|p| f(*p)).collect();
    let at = |c: &[f64; N], w: &[f64; N], t: f64| {
        let mut q = [0.0; N];
        for i in 0..N {
            q[i] = c[i] + t * (w[i] - c[i]);
        }
        q
    };
    for _ in 0..iters {
        let mut idx: Vec<usize> = (0..=N).collect();
        idx.sort_by(|a, b| val[*a].total_cmp(&val[*b]));
        let (b, m, w) = (idx[0], idx[N - 1], idx[N]);
        let mut c = [0.0; N];
        for &k in &idx[..N] {
            for i in 0..N {
                c[i] += pts[k][i] / N as f64;
            }
        }
        let r = at(&c, &pts[w], -1.0);
        let fr = f(r);
        if fr < val[b] {
            let e = at(&c, &pts[w], -2.0);
            let fe = f(e);
            (pts[w], val[w]) = if fe < fr { (e, fe) } else { (r, fr) };
        } else if fr < val[m] {
            (pts[w], val[w]) = (r, fr);
        } else {
            let k = at(&c, &pts[w], 0.5);
            let fk = f(k);
            if fk < val[w] {
                (pts[w], val[w]) = (k, fk);
            } else {
                for &j in &idx[1..] {
                    pts[j] = at(&pts[b], &pts[j], 0.5);
                    val[j] = f(pts[j]);
                }
            }
        }
    }
    let b = (0..=N).min_by(|a, b| val[*a].total_cmp(&val[*b])).unwrap();
    pts[b]
}

/// Fit the field Fresnels' profile by tracing: the chief rays from the
/// sun's slit, over the spectrum and the slit's height, should cross the
/// projection lens's diaphragm at its centre, or the diaphragm cuts them. A
/// thin-lens radius bends the edge of so fast a Fresnel too hard, and PMMA
/// bends the violet end harder than the red. Nelder-Mead over the radius's
/// scale, the conic and the centre's shift along the spectrum.
fn aim(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) -> (f64, f64, f64) {
    let miss = |q: [f64; 3]| {
        let t0 = std::time::Instant::now();
        let mut sc = Scene3::new();
        let p = optics(
            &mut sc,
            slit,
            g,
            cam,
            &Design {
                fresnel_scale: q[0],
                fresnel_conic: q[1],
                fresnel_shift: q[2],
                fresnel_pitch: FIT_PITCH,
                probe_table: true,
                stop_probe: true,
                ..*d
            },
        );
        let (probe, c, axis) = p.stop_probe.expect("the probe");
        let t1 = std::time::Instant::now();
        let _timing = Timing(t0, t1);
        let mut probes = Vec::new();
        for h in [-0.45, 0.0, 0.45] {
            for l in [BLUE_UM, 0.42, 0.47, 0.53, 0.58, 0.64, RED_UM] {
                let (from, dir) = sun_ray(slit, d.slit_turn, h * d.slit_height);
                probes.push((from, dir, l));
            }
        }
        land_many(&sc, probe, &probes)
            .iter()
            .map(|q| match q {
                Some(q) => {
                    let r = *q - c;
                    (r - axis * r.dot(axis)).length_squared()
                }
                None => 1e4,
            })
            .sum::<f64>()
    };
    let q = nelder_mead(miss, [d.fresnel_scale, d.fresnel_conic, d.fresnel_shift], [0.05, 0.3, 5.0], 90);
    (q[0], q[1], q[2])
}

/// Turn the slit until a line stands square across the band: the middle
/// wavelength from the slit's two ends lands at the same place along the
/// spectrum. Secant on the turn.
fn square_lines(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) -> f64 {
    let lean = |turn: f64| {
        let mut sc = Scene3::new();
        let dd = Design { slit_turn: turn, ..*d };
        let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..dd });
        let probes: Vec<(Vec3, Vec3, f64)> = [0.45, -0.45]
            .iter()
            .map(|&h| {
                let (from, dir) = sun_ray(slit, turn, h * dd.slit_height);
                (from, dir, 0.53)
            })
            .collect();
        let x = land_many(&sc, p.table, &probes);
        let u = table_axes().0;
        x[0].map_or(f64::NAN, |q| q.dot(u)) - x[1].map_or(f64::NAN, |q| q.dot(u))
    };
    let (mut t0, mut t1) = (d.slit_turn, d.slit_turn + 0.03);
    let (mut e0, mut e1) = (lean(t0), lean(t1));
    if !e0.is_finite() || !e1.is_finite() {
        eprintln!("the slit's ends miss the table at turn {t0:.4}: {e0} {e1}; the slit stays unturned");
        return d.slit_turn;
    }
    for _ in 0..8 {
        if (e1 - e0).abs() < 1e-12 || e1.abs() < 0.02 {
            break;
        }
        let t2 = t1 - e1 * (t1 - t0) / (e1 - e0);
        let e2 = lean(t2);
        if !e2.is_finite() {
            break;
        }
        (t0, e0) = (t1, e1);
        (t1, e1) = (t2, e2);
    }
    t1
}

/// The camera mirrors' aspheric terms and the projection's focus, fitted
/// together by tracing onto the table: for three slit points and nine
/// wavelengths, the chief ray and two rings of eight filling the
/// collimator's cone; the merit is each group's spread along the spectrum
/// (plus a twentieth of it across), the spread that blurs lines. The camera
/// works its fan of 34 degrees at f/8, the grating 0.46 of its radius out:
/// as spheres, the ends of the spectrum carry field coma and astigmatism.
/// Each group's spread along the spectrum on the table (plus a twentieth of
/// it across), mm², as [`camera_fit`] weighs it: three slit points by nine
/// wavelengths, each a chief ray and two rings of eight in the
/// collimator's cone.
fn along_spread(slit: Vec3, g: &GratingFrame, cam: &Camera, dd: &Design) -> Vec<(f64, Option<Vec3>)> {
    let a = 0.5 * BEAM / F_COLL;
    let aw = 0.5 * beam_along() / F_COLL;
    let mut probes = Vec::new();
    for h in [-0.8, 0.0, 0.8] {
        let (_, chief) = sun_ray(slit, dd.slit_turn, h * 0.5 * dd.slit_height);
        let from = slit_point(slit, dd.slit_turn, h * 0.5 * dd.slit_height, 0.0);
        let e1 = chief.cross(Vec3::Y).normalize();
        let e2 = e1.cross(chief);
        for i in 0..9 {
            let l = 0.378 + (0.688 - 0.378) * i as f64 / 8.0;
            probes.push((from - chief * 20.0, chief, l));
            for (ring, off) in [(0.45, 0.0), (0.9, 0.5)] {
                for k in 0..8 {
                    let ang = (k as f64 + off) * std::f64::consts::FRAC_PI_4;
                    let dir = (chief + (e1 * (ang.cos() * aw) + e2 * (ang.sin() * a)) * ring).normalize();
                    probes.push((from - dir * 20.0, dir, l));
                }
            }
        }
    }
    let per = 17;
    let mut sc = Scene3::new();
    let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..*dd });
    land_many(&sc, p.table, &probes)
        .chunks(per)
        .map(|grp| {
            let all: Vec<Vec3> = grp.iter().flatten().copied().collect();
            if all.len() < per / 2 {
                return (100.0, None);
            }
            // Strays, more than 8 mm from the group's median along the
            // spectrum, count 16 mm² each instead of their distance.
            let (u, w) = table_axes();
            let mut a: Vec<f64> = all.iter().map(|p| p.dot(u)).collect();
            a.sort_by(f64::total_cmp);
            let med = a[a.len() / 2];
            let pts: Vec<Vec3> = all.iter().copied().filter(|p| (p.dot(u) - med).abs() < 8.0).collect();
            let strays = (per - pts.len()) as f64;
            if pts.len() < per / 2 {
                return (100.0, None);
            }
            let n = pts.len() as f64;
            let c = pts.iter().fold(Vec3::ZERO, |a, b| a + *b) / n;
            let vx = pts.iter().map(|p| (*p - c).dot(u).powi(2)).sum::<f64>() / n;
            let vz = pts.iter().map(|p| (*p - c).dot(w).powi(2)).sum::<f64>() / n;
            // Rays lost on the way (a mirror's edge, the diaphragm) count
            // 9 mm² each: light the table never gets.
            let lost = (per - all.len()) as f64;
            if std::env::var_os("SPREAD_DETAIL").is_some() {
                eprintln!("SPREAD blur {:.2} mm along, {:.2} across; {strays} strays, {lost} lost", vx.sqrt(), vz.sqrt());
            }
            ((vx + 0.05 * vz) * n / per as f64 + (16.0 * strays + 9.0 * lost) / per as f64, Some(c))
        })
        .collect()
}

/// How far the spectrum's middle row strays from a straight line across
/// the table, mm², per group: its centroids' `z` against the line fitted
/// through them.
fn bow(groups: &[(f64, Option<Vec3>)]) -> f64 {
    let (u, w) = table_axes();
    let pts: Vec<Vec3> = groups[9..18].iter().filter_map(|g| g.1).map(|c| Vec3::new(c.dot(u), 0.0, c.dot(w))).collect();
    if pts.len() < 3 {
        return 1e4;
    }
    let n = pts.len() as f64;
    let (mx, mz) = (pts.iter().map(|p| p.x).sum::<f64>() / n, pts.iter().map(|p| p.z).sum::<f64>() / n);
    let sxx: f64 = pts.iter().map(|p| (p.x - mx).powi(2)).sum();
    let sxz: f64 = pts.iter().map(|p| (p.x - mx) * (p.z - mz)).sum();
    let b = sxz / sxx;
    pts.iter().map(|p| (p.z - mz - b * (p.x - mx)).powi(2)).sum::<f64>() / n
}

/// The camera's layout fitted with its mirrors' shapes: the grating's
/// distance from M2 (as a fraction of M2's radius), the convex fold's
/// distance (a fraction of the focal length) and radius (a multiple of
/// M2's), M2's conic, the fold's r^4 term and the projection's focus. The
/// camera is solved afresh for each try, so the spectrum keeps its length.
/// Meant with `IDEAL_FIELD` and `IDEAL_PROJ`: the camera alone. Leaves the
/// fit in the environment (`GRATING_AT`, `FOLD_AT`, `FOLD_RATIO`, `CAM`) for
/// the rest of the run, and returns the focus.
fn layout_fit(slit: Vec3, d0: &Design, iters: usize) -> f64 {
    let set = |q: &[f64; 7]| {
        std::env::set_var("GRATING_AT", q[0].to_string());
        std::env::set_var("FOLD_AT", q[1].to_string());
        std::env::set_var("FOLD_RATIO", q[2].to_string());
        std::env::set_var("CAMERA_TILT", q[6].to_string());
    };
    type Groups = (Vec<(f64, Option<Vec3>)>, f64);
    let groups = |q: [f64; 7]| -> Option<Groups> {
        if !(0.35..=1.15).contains(&q[0]) || !(0.55..=0.93).contains(&q[1]) || !(0.5..=8.0).contains(&q[2]) || !(3.0..=14.0).contains(&q[6]) {
            return None;
        }
        set(&q);
        let (g, cam, d) = solve();
        let dd = Design {
            m2_asph: [q[3], 0.0, 0.0],
            m3_asph: [q[4], 0.0],
            focus_shift: q[5],
            fresnel_scale: d0.fresnel_scale,
            fresnel_conic: d0.fresnel_conic,
            fresnel_shift: d0.fresnel_shift,
            slit_turn: d0.slit_turn,
            ..d
        };
        // `LAMP_W`: with the lamps on, what losing all their light costs.
        let lost = if lamps_on() { knob("LAMP_W", 2.0) * lamp_loss(slit, &g, &cam, &dd) } else { 0.0 };
        LAST_LAMP_LOSS.store(lost.to_bits(), std::sync::atomic::Ordering::Relaxed);
        Some((along_spread(slit, &g, &cam, &dd), d.f_field))
    };
    // The spectrum's bow weighs a twentieth: 4 mm of it costs what 0.9 mm
    // of blur does.
    // `FIELD_W`: how much a strong field lens costs. The camera's exit pupil
    // where the projection lens's entrance pupil is needs none; a strong
    // one, used 800 mm off its centre, images the pupil badly and the
    // projection lens clips the light.
    let fw = knob("FIELD_W", 0.0);
    // `LAYOUT_LEAN`: how much each wavelength's line leaning across the
    // band costs, per mm² between the slit's two ends (0.8 of its height).
    let lw = knob("LAYOUT_LEAN", 0.0);
    let lean = |v: &[(f64, Option<Vec3>)]| {
        let u = table_axes().0;
        (0..9)
            .map(|i| match (v[i].1, v[18 + i].1) {
                (Some(a), Some(b)) => (a - b).dot(u).powi(2),
                _ => 100.0,
            })
            .sum::<f64>()
            / 9.0
    };
    let lamp_cost = || f64::from_bits(LAST_LAMP_LOSS.load(std::sync::atomic::Ordering::Relaxed));
    let merit = |q: [f64; 7]| groups(q).map_or(1e6, |(v, ff)| v.iter().map(|g| g.0).sum::<f64>() / v.len() as f64 + 0.05 * bow(&v) + fw * (1000.0 / ff).powi(2) + lw * lean(&v) + lamp_cost());
    let show = |q: [f64; 7], what: &str| {
        let Some((v, ff)) = groups(q) else { return };
        eprintln!("LAYOUT {what}: field lens f {ff:.0} mm, lean rms {:.2} mm, lamp loss cost {:.3}", lean(&v).sqrt(), lamp_cost());
        let rows: Vec<String> = (0..9)
            .map(|i| format!("{:.0}: {:.2} {:.2} {:.2}", (0.378 + 0.31 * i as f64 / 8.0) * 1e3, v[i].0.sqrt(), v[9 + i].0.sqrt(), v[18 + i].0.sqrt()))
            .collect();
        let zs: Vec<String> = v[9..18].iter().map(|g| g.1.map_or("miss".into(), |c| format!("{:.1}", c.dot(table_axes().1)))).collect();
        eprintln!("LAYOUT {what}: GRATING_AT={} FOLD_AT={} FOLD_RATIO={} CAMERA_TILT={} k2 {:.3} s4' {:.3} focus {:.2}; merit {:.4}, bow rms {:.2} mm", q[0], q[1], q[2], q[6], q[3], q[4], q[5], merit(q), bow(&v).sqrt());
        eprintln!("LAYOUT {what} spectrum z per nm: {}", zs.join(" "));
        eprintln!("LAYOUT {what} rms per nm (slit -0.8, 0, +0.8): {}", rows.join(" | "));
    };
    let start = |q: [f64; 7]| {
        // The focus for this layout alone, on a grid.
        let mut best = q;
        let mut m = f64::INFINITY;
        for k in -20..=20 {
            let mut t = q;
            t[5] = 2.0 * k as f64;
            let e = merit(t);
            if e < m {
                (m, best) = (e, t);
            }
        }
        (m, best)
    };
    // A few starts across the layouts the scans found promising.
    let cam_env: Vec<f64> = std::env::var("CAM").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let (k2, s4) = if let [k, _, _, s, _] = cam_env[..] { (k, s) } else { (0.0, 0.0) };
    let tilt = knob("CAMERA_TILT", CAMERA_TILT_DEG);
    let mut seeds: Vec<(f64, [f64; 7])> = [
        [knob("GRATING_AT", GRATING_AT), knob("FOLD_AT", FOLD_AT), knob("FOLD_RATIO", FOLD_RADIUS_RATIO), k2, s4, 0.0, tilt],
        [1.0, 0.87, 1.0, 0.0, 0.0, 0.0, 5.0],
        [1.0, 0.87, 1.0, 0.0, 0.0, 0.0, 11.0],
        [1.0, 0.8, 1.3, 0.0, 0.0, 0.0, 8.0],
    ]
    .into_iter()
    .map(start)
    .collect();
    seeds.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (m, q) in &seeds {
        eprintln!("LAYOUT seed {:?}: merit {m:.4}", q);
    }
    let mut x = seeds[0].1;
    show(x, "start");
    for round in 0..knob("LAYOUT_ROUNDS", 3.0) as usize {
        x = nelder_mead(merit, x, [0.05, 0.03, 0.15, 0.3, 0.5, 3.0, 1.5], iters);
        show(x, &format!("round {round}"));
    }
    set(&x);
    std::env::set_var("CAM", format!("{} 0 0 {} 0", x[3], x[4]));
    x[5]
}

/// The share of the lamps' light that is lost before the table: from each
/// lamp slit's two ends and middle, at the band's ends and middle in
/// wavelength, the chief ray and four more at 0.8 of the collimator's cone,
/// on the optics as placed (sized for the three bands).
fn lamp_loss(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) -> f64 {
    let mut sc = Scene3::new();
    let p = optics(&mut sc, slit, g, cam, &Design { probe_table: true, ..*d });
    let a = 0.8 * 0.5 * BEAM / F_COLL;
    let mut probes = Vec::new();
    for c in p.lamp_slit {
        for u in [-0.47, 0.0, 0.47] {
            let from = c + p.lamp_run * (u * d.slit_height);
            for l in [0.378, 0.533, 0.688] {
                for (du, dw) in [(0.0, 0.0), (a, 0.0), (-a, 0.0), (0.0, a), (0.0, -a)] {
                    let dir = (p.lamp_dir + p.lamp_run * du + p.lamp_across * dw).normalize();
                    probes.push((from - dir * 5.0, dir, l));
                }
            }
        }
    }
    let hits = land_many(&sc, p.table, &probes);
    if std::env::var_os("LAMP_LOSS_DIAG").is_some() {
        for (k, grp) in hits.chunks(15).enumerate() {
            let (side, u) = (k / 3, [-0.47, 0.0, 0.47][k % 3]);
            let miss: Vec<usize> = grp.chunks(5).map(|c| c.iter().filter(|h| h.is_none()).count()).collect();
            eprintln!("LAMPLOSS slit {} at u {u:+.2}: misses of 5 at 378/533/688 nm {:?}", ["upper", "lower"][side], miss);
        }
        // Where the outer ends' chief rays at 533 nm go.
        let mut names: std::collections::HashMap<u64, String> = Default::default();
        for o in &sc.surfaces {
            names.insert(o.common.id.0, o.common.label.clone());
        }
        for o in &sc.bodies {
            names.insert(o.common.id.0, o.common.label.clone());
        }
        for o in &sc.apertures {
            names.insert(o.common.id.0, o.common.label.clone());
        }
        for (side, u, du, dw) in [(0usize, 0.47, 0.0, 0.0), (1, -0.47, 0.0, 0.0), (1, 0.0, a, 0.0), (1, 0.0, -a, 0.0), (1, 0.0, 0.0, a), (1, 0.0, 0.0, -a)] {
            let from = p.lamp_slit[side] + p.lamp_run * (u * d.slit_height);
            let dir = (p.lamp_dir + p.lamp_run * du + p.lamp_across * dw).normalize();
            let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(from - dir * 5.0, dir).unwrap(), lambda_um: 0.533, flux: 1.0, source: None, pol: None };
            let cfg = oc_trace::TraceConfig3 { max_depth: 40, ghosts: false, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(&sc), ..oc_trace::TraceConfig3::default() };
            let chain: Vec<String> = oc_trace::trace3(&sc, &[lr], &cfg)
                .events
                .iter()
                .map(|e| match &e.kind {
                    oc_trace::TraceEventKind::Interact { surface, .. } => format!("{} @ ({:.0}, {:.0}, {:.0})", names.get(&surface.object.0).map_or("?", |q| q.split(',').next().unwrap_or("")), e.point.x, e.point.y, e.point.z),
                    oc_trace::TraceEventKind::ApertureBlocked { aperture } => format!("BLOCKED {}", names.get(&aperture.0).cloned().unwrap_or_default()),
                    oc_trace::TraceEventKind::Escaped => format!("ESCAPED @ ({:.0}, {:.0}, {:.0})", e.point.x, e.point.y, e.point.z),
                    k => format!("{k:?}").chars().take(30).collect(),
                })
                .collect();
            eprintln!("LAMPLOSS path {} u {u:+.2} cone {du:+.3} {dw:+.3}: {}", ["upper", "lower"][side], chain.join(" -> "));
        }
    }
    hits.iter().filter(|h| h.is_none()).count() as f64 / hits.len() as f64
}

/// The projection lens refitted against the traced system: its six faces'
/// curvatures and conics (`LENS_TWEAK`), with the focus and the field
/// Fresnels' conic and scale, onto [`along_spread`] and the spectrum's
/// [`bow`]. The layout is solved afresh for each try (the lens's focal
/// length sets the camera's). Leaves `LENS_TWEAK` and `FIT` in the
/// environment.
fn lens_fit(slit: Vec3, d0: &Design, iters: usize) {
    let tweak0: Vec<f64> = std::env::var("LENS_TWEAK").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let mut x0 = [0.0; 15];
    if tweak0.len() == 12 {
        x0[..12].copy_from_slice(&tweak0);
    }
    (x0[12], x0[13], x0[14]) = (d0.focus_shift, d0.fresnel_conic, d0.fresnel_scale);
    let set = |q: &[f64; 15]| {
        let t: Vec<String> = q[..12].iter().map(|v| v.to_string()).collect();
        std::env::set_var("LENS_TWEAK", t.join(" "));
    };
    let groups = |q: [f64; 15]| -> Vec<(f64, Option<Vec3>)> {
        set(&q);
        let (g, cam, d) = solve();
        let dd = Design {
            focus_shift: q[12],
            fresnel_conic: q[13],
            fresnel_scale: q[14],
            fresnel_shift: d0.fresnel_shift,
            fresnel_asph: d0.fresnel_asph,
            fresnel_shift_v: d0.fresnel_shift_v,
            f4_curv: d0.f4_curv,
            slit_turn: d0.slit_turn,
            m2_asph: d0.m2_asph,
            m3_asph: d0.m3_asph,
            ..d
        };
        along_spread(slit, &g, &cam, &dd)
    };
    let merit = |q: [f64; 15]| {
        let v = groups(q);
        v.iter().map(|g| g.0).sum::<f64>() / v.len() as f64 + 0.05 * bow(&v)
    };
    let show = |q: [f64; 15], what: &str| {
        let v = groups(q);
        let rows: Vec<String> = (0..9)
            .map(|i| format!("{:.0}: {:.2} {:.2} {:.2}", (0.378 + 0.31 * i as f64 / 8.0) * 1e3, v[i].0.sqrt(), v[9 + i].0.sqrt(), v[18 + i].0.sqrt()))
            .collect();
        let t: Vec<String> = q[..12].iter().map(|v| format!("{v:.4}")).collect();
        eprintln!("LENSFIT {what}: merit {:.4}, bow rms {:.2} mm; LENS_TWEAK=\"{}\" focus {:.2} fresnel conic {:.3} scale {:.4}", merit(q), bow(&v).sqrt(), t.join(" "), q[12], q[13], q[14]);
        eprintln!("LENSFIT {what} rms per nm (slit -0.8, 0, +0.8): {}", rows.join(" | "));
    };
    // The focus first, on a grid, for the lens as it is.
    let mut x = x0;
    let mut best = f64::INFINITY;
    for k in -25..=25 {
        let mut t = x0;
        t[12] = x0[12] + 2.0 * k as f64;
        let m = merit(t);
        if m < best {
            (best, x) = (m, t);
        }
    }
    show(x, "start");
    let step = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 3.0, 1.0, 0.01];
    for round in 0..knob("LENS_ROUNDS", 3.0) as usize {
        x = nelder_mead(merit, x, step, iters);
        show(x, &format!("round {round}"));
    }
    set(&x);
    std::env::set_var("FIT", format!("{} {} {} {} {}", x[14], x[13], d0.fresnel_shift, x[12], d0.slit_turn));
}

fn camera_fit(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, iters: usize) -> ([f64; 3], [f64; 2], f64) {
    let a = 0.5 * BEAM / F_COLL;
    let aw = 0.5 * beam_along() / F_COLL;
    let mut probes = Vec::new();
    for h in [-0.8, 0.0, 0.8] {
        let (_, chief) = sun_ray(slit, d.slit_turn, h * 0.5 * d.slit_height);
        let from = slit_point(slit, d.slit_turn, h * 0.5 * d.slit_height, 0.0);
        let e1 = chief.cross(Vec3::Y).normalize();
        let e2 = e1.cross(chief);
        for i in 0..9 {
            let l = 0.378 + (0.688 - 0.378) * i as f64 / 8.0;
            probes.push((from - chief * 20.0, chief, l));
            for (ring, off) in [(0.45, 0.0), (0.9, 0.5)] {
                for k in 0..8 {
                    let ang = (k as f64 + off) * std::f64::consts::FRAC_PI_4;
                    let dir = (chief + (e1 * (ang.cos() * aw) + e2 * (ang.sin() * a)) * ring).normalize();
                    probes.push((from - dir * 20.0, dir, l));
                }
            }
        }
    }
    let groups = |q: [f64; 6]| -> Vec<f64> {
        let dd = Design {
            m2_asph: [q[0], q[1], q[2]],
            m3_asph: [q[3], q[4]],
            focus_shift: q[5],
            ..*d
        };
        along_spread(slit, g, cam, &dd).into_iter().map(|x| x.0).collect()
    };
    let merit = |q: [f64; 6]| {
        let v = groups(q);
        v.iter().sum::<f64>() / v.len() as f64
    };
    let show = |q: [f64; 6], what: &str| {
        let v = groups(q);
        let rows: Vec<String> = (0..9)
            .map(|i| {
                let l = 0.378 + (0.688 - 0.378) * i as f64 / 8.0;
                format!("{:.0}: {:.2} {:.2} {:.2}", l * 1e3, v[i].sqrt(), v[9 + i].sqrt(), v[18 + i].sqrt())
            })
            .collect();
        eprintln!("CAMFIT {what} rms per nm (slit -0.8, 0, +0.8): {}", rows.join(" | "));
    };
    let x0 = [d.m2_asph[0], d.m2_asph[1], d.m2_asph[2], d.m3_asph[0], d.m3_asph[1], d.focus_shift];
    eprintln!("CAMFIT start: merit {:.4} (rms along {:.3} mm)", merit(x0), merit(x0).sqrt());
    show(x0, "before");
    // `FOCUS_ONLY`: the projection's focus alone, golden section over +-40 mm.
    if std::env::var_os("FOCUS_ONLY").is_some() {
        let at = |f: f64| merit([x0[0], x0[1], x0[2], x0[3], x0[4], f]);
        let r = 0.5 * (5.0_f64.sqrt() - 1.0);
        // A grid first: missed rays make the merit jump.
        let best = (-20..=20).map(|k| x0[5] + 2.0 * k as f64).min_by(|p, q| at(*p).total_cmp(&at(*q))).unwrap();
        let (mut a, mut b) = (best - 2.0, best + 2.0);
        let (mut c, mut e) = (b - r * (b - a), a + r * (b - a));
        let (mut fc, mut fe) = (at(c), at(e));
        while b - a > 0.05 {
            if fc < fe {
                (b, e, fe) = (e, c, fc);
                c = b - r * (b - a);
                fc = at(c);
            } else {
                (a, c, fc) = (c, e, fe);
                e = a + r * (b - a);
                fe = at(e);
            }
        }
        let mut x = x0;
        x[5] = 0.5 * (a + b);
        eprintln!("CAMFIT focus only: {:.2} mm, merit {:.4} (rms {:.3} mm)", x[5], merit(x), merit(x).sqrt());
        show(x, "after");
        return (d.m2_asph, d.m3_asph, x[5]);
    }
    if iters == 0 {
        return (d.m2_asph, d.m3_asph, d.focus_shift);
    }
    let mut x = x0;
    // Restarts: Nelder-Mead stalls on long valleys.
    for round in 0..3 {
        x = nelder_mead(merit, x, [0.5, 0.3, 0.1, 0.3, 0.1, 3.0], iters);
        let m = merit(x);
        eprintln!("CAMFIT round {round}: merit {m:.4} (rms {:.3} mm), CAM=\"{} {} {} {} {}\" focus {:.3}", m.sqrt(), x[0], x[1], x[2], x[3], x[4], x[5]);
    }
    show(x, "after");
    ([x[0], x[1], x[2]], [x[3], x[4]], x[5])
}

/// The slit's curvature and turn together, so lines stand straight and
/// square across the band: at five wavelengths, the slit's two ends and
/// middle land in line. The merit is each line's sag (the middle against the
/// ends' mean) and lean (one end against the other), squared, on the table.
fn straighten_lines(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) -> (f64, f64) {
    let merit = |q: [f64; 2]| {
        set_slit_curve(q[0]);
        let mut sc = Scene3::new();
        let dd = Design { slit_turn: q[1], ..*d };
        let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..dd });
        let mut probes = Vec::new();
        for l in [0.39, 0.46, 0.53, 0.60, 0.68] {
            for h in [0.45, 0.0, -0.45] {
                let (from, dir) = sun_ray(slit, q[1], h * dd.slit_height);
                probes.push((from, dir, l));
            }
        }
        let x = land_many(&sc, p.table, &probes);
        x.chunks(3)
            .map(|w| match (w[0], w[1], w[2]) {
                (Some(a), Some(b), Some(c)) => {
                    let u = table_axes().0;
                    (b.dot(u) - 0.5 * (a.dot(u) + c.dot(u))).powi(2) + (a.dot(u) - c.dot(u)).powi(2)
                }
                _ => 1e4,
            })
            .sum::<f64>()
    };
    // Each wavelength's sag and lean on the table, mm.
    let lines = |q: [f64; 2]| -> String {
        set_slit_curve(q[0]);
        let mut sc = Scene3::new();
        let dd = Design { slit_turn: q[1], ..*d };
        let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..dd });
        let mut probes = Vec::new();
        let ls = [0.39, 0.46, 0.53, 0.60, 0.68];
        for l in ls {
            for h in [0.45, 0.0, -0.45] {
                let (from, dir) = sun_ray(slit, q[1], h * dd.slit_height);
                probes.push((from, dir, l));
            }
        }
        let u = table_axes().0;
        let x = land_many(&sc, p.table, &probes);
        x.chunks(3)
            .zip(ls)
            .map(|(w, l)| match (w[0], w[1], w[2]) {
                (Some(a), Some(b), Some(c)) => format!("{:.0}: sag {:+.2} lean {:+.2}", l * 1e3, b.dot(u) - 0.5 * (a.dot(u) + c.dot(u)), a.dot(u) - c.dot(u)),
                _ => format!("{:.0}: miss", l * 1e3),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };
    eprintln!("LINES before: {}", lines([slit_curve(), d.slit_turn]));
    let q = nelder_mead(merit, [slit_curve(), d.slit_turn], [0.2, 0.01], 60);
    eprintln!("LINES: slit curve {:.4}, turn {:.5}, merit {:.4} mm^2", q[0], q[1], merit(q));
    eprintln!("LINES after: {}", lines(q));
    set_slit_curve(q[0]);
    (q[0], q[1])
}

/// The field Fresnels fitted to image the camera's pupil onto the
/// projection lens's diaphragm: their radius (as a multiple of the thin
/// lens's), conic, shift along the spectrum and r^4, r^6, r^8 terms. For
/// three slit points and seven wavelengths, the chief ray and a ring of
/// sixteen at the rim of the collimator's cone; the merit is their mean
/// square distance from the diaphragm's centre (a beam centred and as small
/// as the pupil's image), a ray that misses counting 200 mm.
fn pupil_fit(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, iters: usize) -> (f64, f64, f64, [f64; 3]) {
    let a = 0.5 * BEAM / F_COLL;
    let aw = 0.5 * beam_along() / F_COLL;
    let mut probes = Vec::new();
    for h in [-0.8, 0.0, 0.8] {
        let (_, chief) = sun_ray(slit, d.slit_turn, h * 0.5 * d.slit_height);
        let from = slit_point(slit, d.slit_turn, h * 0.5 * d.slit_height, 0.0);
        let e1 = chief.cross(Vec3::Y).normalize();
        let e2 = e1.cross(chief);
        for l in [0.378, 0.42, 0.47, 0.53, 0.59, 0.64, 0.688] {
            probes.push((from - chief * 20.0, chief, l));
            for k in 0..16 {
                let ang = k as f64 * std::f64::consts::PI / 8.0;
                let dir = (chief + (e1 * (ang.cos() * aw) + e2 * (ang.sin() * a)) * 0.95).normalize();
                probes.push((from - dir * 20.0, dir, l));
            }
        }
    }
    let merit = |q: [f64; 6]| {
        let mut sc = Scene3::new();
        let dd = Design {
            fresnel_scale: q[0],
            fresnel_conic: q[1],
            fresnel_shift: q[2],
            fresnel_asph: [q[3], q[4], q[5]],
            fresnel_pitch: FIT_PITCH,
            probe_table: true,
            stop_probe: true,
            ..*d
        };
        let p = optics(&mut sc, slit, g, cam, &dd);
        let (probe, c, axis) = p.stop_probe.expect("the probe");
        land_many(&sc, probe, &probes)
            .iter()
            .map(|q| match q {
                Some(q) => {
                    let r = *q - c;
                    (r - axis * r.dot(axis)).length_squared()
                }
                None => 4e4,
            })
            .sum::<f64>()
            / probes.len() as f64
    };
    let x0 = [d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.fresnel_asph[0], d.fresnel_asph[1], d.fresnel_asph[2]];
    eprintln!("PUPILFIT start: rms {:.1} mm", merit(x0).sqrt());
    let mut x = x0;
    for round in 0..3 {
        x = nelder_mead(merit, x, [0.03, 1.0, 5.0, 1.0, 0.5, 0.2], iters);
        eprintln!("PUPILFIT round {round}: rms {:.1} mm; FIT scale {} conic {} shift {}; FRESNEL_ASPH=\"{} {} {}\"", merit(x).sqrt(), x[0], x[1], x[2], x[3], x[4], x[5]);
    }
    (x[0], x[1], x[2], [x[3], x[4], x[5]])
}

/// Each wavelength's line lean on the table, mm: where the slit's two ends
/// (0.8 of its height) land along the spectrum, one against the other, from
/// [`along_spread`]'s groups.
fn leans(v: &[(f64, Option<Vec3>)]) -> Vec<f64> {
    let u = table_axes().0;
    (0..9)
        .map(|i| match (v[i].1, v[18 + i].1) {
            (Some(a), Some(b)) => (a - b).dot(u),
            _ => f64::NAN,
        })
        .collect()
}

/// The keystone taken out: the relay's tilt against the intermediate
/// spectrum (`RELAY_TILT`), the light's tilt onto the table (`F5_TILT`), the
/// projection's focus and the slit's turn, onto [`along_spread`], the
/// spectrum's [`bow`] and the lines' lean (`KEY_LEAN` per mm², default 1).
/// Leaves both tilts in the environment.
fn keystone_fit(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, iters: usize) -> Design {
    let set = |q: &[f64; 4]| {
        std::env::set_var("RELAY_TILT", q[0].to_string());
        std::env::set_var("F5_TILT", q[1].to_string());
    };
    let with = |q: &[f64; 4]| Design { focus_shift: q[2], slit_turn: q[3], ..*d };
    let lw = knob("KEY_LEAN", 1.0);
    let merit = |q: [f64; 4]| {
        set(&q);
        let v = along_spread(slit, g, cam, &with(&q));
        let l = leans(&v);
        let lean = l.iter().map(|x| if x.is_nan() { 100.0 } else { x * x }).sum::<f64>() / 9.0;
        v.iter().map(|g| g.0).sum::<f64>() / v.len() as f64 + 0.05 * bow(&v) + lw * lean
    };
    let show = |q: [f64; 4], what: &str| {
        let m = merit(q);
        let v = along_spread(slit, g, cam, &with(&q));
        let l: Vec<String> = leans(&v).iter().zip(0..9).map(|(x, i)| format!("{:.0}: {x:+.2}", (0.378 + 0.31 * i as f64 / 8.0) * 1e3)).collect();
        let rows: Vec<String> = (0..9).map(|i| format!("{:.2}", (v[i].0 + v[9 + i].0 + v[18 + i].0).sqrt() / 3f64.sqrt())).collect();
        eprintln!("KEYSTONE {what}: RELAY_TILT={} F5_TILT={} focus {:.2} turn {:.5}; merit {m:.4}, bow rms {:.2} mm", q[0], q[1], q[2], q[3], bow(&v).sqrt());
        eprintln!("KEYSTONE {what} lean per nm (mm over 0.8 of the band): {}", l.join(" | "));
        eprintln!("KEYSTONE {what} blur rms per nm: {}", rows.join(" "));
    };
    let mut x = [knob("RELAY_TILT", 0.0), knob("F5_TILT", 0.0), d.focus_shift, d.slit_turn];
    show(x, "start");
    for round in 0..knob("KEY_ROUNDS", 3.0) as usize {
        x = nelder_mead(merit, x, [1.0, 2.0, 3.0, 0.005], iters);
        show(x, &format!("round {round}"));
    }
    set(&x);
    with(&x)
}

/// The field Fresnels fitted on the table: their radius (a multiple of the
/// thin lens's), conic, shift along the spectrum and across the bands, and
/// r^4, r^6, r^8 terms, with the projection's focus, onto [`along_spread`]
/// (blur, strays and lost light) and the spectrum's [`bow`].
fn fresnel_fit(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, iters: usize) -> Design {
    let with = |q: &[f64; 8]| Design {
        fresnel_scale: q[0],
        fresnel_conic: q[1],
        fresnel_shift: q[2],
        fresnel_shift_v: q[3],
        fresnel_asph: [q[4], q[5], q[6]],
        focus_shift: q[7],
        ..*d
    };
    let merit = |q: [f64; 8]| {
        let v = along_spread(slit, g, cam, &with(&q));
        v.iter().map(|g| g.0).sum::<f64>() / v.len() as f64 + 0.05 * bow(&v)
    };
    let show = |q: [f64; 8], what: &str| {
        let v = along_spread(slit, g, cam, &with(&q));
        let rows: Vec<String> = (0..9)
            .map(|i| format!("{:.0}: {:.2} {:.2} {:.2}", (0.378 + 0.31 * i as f64 / 8.0) * 1e3, v[i].0.sqrt(), v[9 + i].0.sqrt(), v[18 + i].0.sqrt()))
            .collect();
        eprintln!("FRESNELFIT {what}: merit {:.4}; FIT scale {} conic {} shift {} FRESNEL_V={} FRESNEL_ASPH=\"{} {} {}\" focus {}", merit(q), q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
        eprintln!("FRESNELFIT {what} rms per nm (slit -0.8, 0, +0.8): {}", rows.join(" | "));
    };
    let mut x = [d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.fresnel_shift_v, d.fresnel_asph[0], d.fresnel_asph[1], d.fresnel_asph[2], d.focus_shift];
    show(x, "start");
    for round in 0..knob("FRESNEL_ROUNDS", 3.0) as usize {
        x = nelder_mead(merit, x, [0.03, 1.0, 8.0, 8.0, 0.5, 0.3, 0.1, 3.0], iters);
        show(x, &format!("round {round}"));
    }
    with(&x)
}

/// [`aim`] with more freedom: the Fresnels' radius, conic, shift along the
/// spectrum and across the bands, and r^4, r^6, r^8 terms, so every chief
/// ray (three slit points, nine wavelengths) crosses the diaphragm at its
/// centre.
fn aim_chief(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, iters: usize) -> Design {
    // F4's curvature is fitted too with `AIM_F4`, in 1e-5 per mm.
    let f4 = std::env::var_os("AIM_F4").is_some();
    let with = |q: &[f64; 8]| Design {
        fresnel_scale: q[0],
        fresnel_conic: q[1],
        fresnel_shift: q[2],
        fresnel_shift_v: q[3],
        fresnel_asph: [q[4], q[5], q[6]],
        f4_curv: if f4 { q[7] * 1e-5 } else { d.f4_curv },
        ..*d
    };
    let mut probes = Vec::new();
    for h in [-0.45, 0.0, 0.45] {
        for i in 0..9 {
            let l = 0.378 + (0.688 - 0.378) * i as f64 / 8.0;
            let (from, dir) = sun_ray(slit, d.slit_turn, h * d.slit_height);
            probes.push((from, dir, l));
        }
    }
    let miss = |q: [f64; 8]| {
        let mut sc = Scene3::new();
        let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, stop_probe: true, ..with(&q) });
        let (probe, c, axis) = p.stop_probe.expect("the probe");
        let off = land_many(&sc, probe, &probes)
            .iter()
            .map(|q| match q {
                Some(q) => {
                    let r = *q - c;
                    (r - axis * r.dot(axis)).length_squared()
                }
                None => 1e4,
            })
            .sum::<f64>()
            / probes.len() as f64;
        // The spectrum kept straight: the slit's middle row on the table,
        // its bow (mm²) weighted `AIM_BOW` (default 20).
        let table: Vec<(f64, Option<Vec3>)> = land_many(&sc, p.table, &probes).into_iter().map(|q| (0.0, q)).collect();
        // And each wavelength's line square across the band: the slit's two
        // ends (rows 0 and 2) land level along the spectrum, weighted
        // `AIM_LEAN` (default 5) per mm².
        let u = table_axes().0;
        let lean = (0..9)
            .map(|i| match (table[i].1, table[18 + i].1) {
                (Some(a), Some(b)) => (a - b).dot(u).powi(2),
                _ => 1e3,
            })
            .sum::<f64>()
            / 9.0;
        off + knob("AIM_BOW", 20.0) * bow(&table) + knob("AIM_LEAN", 5.0) * lean
    };
    let mut x = [d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.fresnel_shift_v, d.fresnel_asph[0], d.fresnel_asph[1], d.fresnel_asph[2], d.f4_curv * 1e5];
    eprintln!("AIMCHIEF start: rms {:.2} mm off the diaphragm's centre", miss(x).sqrt());
    for round in 0..3 {
        x = nelder_mead(miss, x, [0.03, 1.0, 8.0, 8.0, 0.5, 0.3, 0.1, 3.0], iters);
        eprintln!("AIMCHIEF round {round}: rms {:.2} mm; FIT scale {} conic {} shift {} FRESNEL_V={} FRESNEL_ASPH=\"{} {} {}\" F4_CURV={}", miss(x).sqrt(), x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] * 1e-5);
    }
    with(&x)
}

/// Focus the projection lens by tracing: move it along its axis until pairs
/// of rays from the sun's strip, through opposite sides of the lens's pupil,
/// land closest together on the table, over the spectrum and the strip's
/// height. The paraxial place is off by the lens's spherical aberration and
/// its field's curve, and 1 mm at the lens is about 130 mm on the table.
fn focus(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) -> f64 {
    let blur = |shift: f64| {
        let mut sc = Scene3::new();
        let p = optics(&mut sc, slit, g, cam, &Design { focus_shift: shift, fresnel_pitch: FIT_PITCH, probe_table: true, ..*d });
        // For each slit point and wavelength, the chief ray and a ring of 8
        // at 0.7 of the cone's half angle; the merit is the mean square
        // spread about each group's centroid, over the rays that land.
        let a = 0.7 * 0.5 * BEAM / F_COLL;
        let mut probes = Vec::new();
        for h in [-0.8, 0.0, 0.8] {
            let (_, chief) = sun_ray(slit, d.slit_turn, h * 0.5 * d.slit_height);
            let from = slit_point(slit, d.slit_turn, h * 0.5 * d.slit_height, 0.0);
            let e1 = chief.cross(Vec3::Y).normalize();
            let e2 = e1.cross(chief);
            for l in [0.40, 0.46, 0.53, 0.60, 0.67] {
                for k in 0..9 {
                    let t = if k == 0 {
                        Vec3::ZERO
                    } else {
                        let ang = (k - 1) as f64 * std::f64::consts::FRAC_PI_4;
                        (e1 * ang.cos() + e2 * ang.sin()) * a
                    };
                    let dir = (chief + t).normalize();
                    probes.push((from - dir * 20.0, dir, l));
                }
            }
        }
        let hits = land_many(&sc, p.table, &probes);
        let (mut sum, mut n, mut groups) = (0.0, 0usize, 0usize);
        for g in hits.chunks(9) {
            let pts: Vec<Vec3> = g.iter().flatten().copied().collect();
            if pts.len() < 5 {
                continue;
            }
            let c = pts.iter().fold(Vec3::ZERO, |a, b| a + *b) / pts.len() as f64;
            sum += pts.iter().map(|q| (*q - c).length_squared()).sum::<f64>();
            n += pts.len();
            groups += 1;
        }
        if 2 * groups < hits.len() / 9 {
            return 1e8;
        }
        sum / n as f64
    };
    // Golden section over +-8 mm.
    let r = 0.5 * (5.0_f64.sqrt() - 1.0);
    let (mut a, mut b) = (-0.04 * d.f_proj, 0.04 * d.f_proj);
    let (mut c, mut e) = (b - r * (b - a), a + r * (b - a));
    let (mut fc, mut fe) = (blur(c), blur(e));
    while b - a > 0.02 {
        if fc < fe {
            (b, e, fe) = (e, c, fc);
            c = b - r * (b - a);
            fc = blur(c);
        } else {
            (a, c, fc) = (c, e, fe);
            e = a + r * (b - a);
            fe = blur(e);
        }
    }
    0.5 * (a + b)
}

/// Build the spectrograph with its pick-off at `pickoff` on the tube's axis,
/// the two lamps beside its slit, and the table under the projection.
pub fn add(s: &mut Scene3, pickoff: Vec3) -> Spectrograph {
    let slit = pickoff - pickoff_offset();
    // `CAMERA_FIT2=<iterations>`: the camera laid out for the intermediate
    // spectrum alone (see [`camera_fit2`]).
    if let Ok(n) = std::env::var("CAMERA_FIT2") {
        std::env::remove_var("CAMERA_FIT2");
        let (_, _, mut d0) = solve();
        let cam_terms: Vec<f64> = std::env::var("CAM").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if let [k2, a, b, c, e] = cam_terms[..] {
            (d0.m2_asph, d0.m3_asph) = ([k2, a, b], [c, e]);
        }
        let fit: Vec<f64> = std::env::var("FIT").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if let [_, _, _, _, turn] = fit[..] {
            d0.slit_turn = turn;
        }
        camera_fit2(slit, &d0, n.parse().unwrap_or(100));
    }
    // `LAYOUT_FIT=<iterations>`: the camera's layout first (see
    // [`layout_fit`]); it leaves its result in the environment, and the
    // focus in `FIT`, so the rest of this run and the next `add` build it.
    if let Ok(n) = std::env::var("LAYOUT_FIT") {
        std::env::remove_var("LAYOUT_FIT");
        let (_, _, d0) = solve();
        let mut d0 = d0;
        let fit: Vec<f64> = std::env::var("FIT").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if let [scale, conic, shift, _, turn] = fit[..] {
            (d0.fresnel_scale, d0.fresnel_conic, d0.fresnel_shift, d0.slit_turn) = (scale, conic, shift, turn);
        }
        let focus = layout_fit(slit, &d0, n.parse().unwrap_or(120));
        std::env::set_var("FIT", format!("{} {} {} {} {}", d0.fresnel_scale, d0.fresnel_conic, d0.fresnel_shift, focus, d0.slit_turn));
    }
    // `LENS_FIT=<iterations>`: the projection lens refitted in the traced
    // system (see [`lens_fit`]), from `FIT`.
    if let Ok(n) = std::env::var("LENS_FIT") {
        std::env::remove_var("LENS_FIT");
        let (_, _, mut d0) = solve();
        let fit: Vec<f64> = std::env::var("FIT").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if let [scale, conic, shift, focus, turn] = fit[..] {
            (d0.fresnel_scale, d0.fresnel_conic, d0.fresnel_shift, d0.focus_shift, d0.slit_turn) = (scale, conic, shift, focus, turn);
        }
        let cam_terms: Vec<f64> = std::env::var("CAM").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if let [k2, a, b, c, e] = cam_terms[..] {
            (d0.m2_asph, d0.m3_asph) = ([k2, a, b], [c, e]);
        }
        if let Some(k) = std::env::var("SLIT_K").ok().and_then(|v| v.parse::<f64>().ok()) {
            set_slit_curve(k);
        }
        let fa: Vec<f64> = std::env::var("FRESNEL_ASPH").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if let [a, b, c] = fa[..] {
            d0.fresnel_asph = [a, b, c];
        }
        d0.fresnel_shift_v = knob("FRESNEL_V", 0.0);
        d0.f4_curv = knob("F4_CURV", 0.0);
        lens_fit(slit, &d0, n.parse().unwrap_or(300));
    }
    let (g, cam, mut d) = solve();
    // `FIT="scale conic shift focus turn"` reuses a fit, to try changes
    // quickly.
    let fit: Vec<f64> = std::env::var("FIT").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    if let [scale, conic, shift, focus_shift, turn] = fit[..] {
        (d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.focus_shift, d.slit_turn) = (scale, conic, shift, focus_shift, turn);
    } else {
        (d.fresnel_scale, d.fresnel_conic, d.fresnel_shift) = aim(slit, &g, &cam, &d);
        d.focus_shift = focus(slit, &g, &cam, &d);
        d.slit_turn = square_lines(slit, &g, &cam, &d);
        (d.fresnel_scale, d.fresnel_conic, d.fresnel_shift) = aim(slit, &g, &cam, &d);
        d.focus_shift = focus(slit, &g, &cam, &d);
    }
    // `CAM="k2 s4 s6 s4' s6'"` reuses a camera fit; `CAM_FIT=<iterations>`
    // fits the camera mirrors, then the field lens, focus and turn again.
    let cam_terms: Vec<f64> = std::env::var("CAM").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    if let [k2, a, b, c, e] = cam_terms[..] {
        (d.m2_asph, d.m3_asph) = ([k2, a, b], [c, e]);
    }
    if let Some(k) = std::env::var("SLIT_K").ok().and_then(|v| v.parse::<f64>().ok()) {
        set_slit_curve(k);
    }
    let fa: Vec<f64> = std::env::var("FRESNEL_ASPH").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    if let [a, b, c] = fa[..] {
        d.fresnel_asph = [a, b, c];
    }
    d.fresnel_shift_v = knob("FRESNEL_V", 0.0);
    d.f4_curv = knob("F4_CURV", 0.0);
    // `RESQUARE=1`: the slit's turn solved again (after a change upstream
    // that turns the slit's image, as `COLL_PHI` does).
    if std::env::var_os("RESQUARE").is_some() {
        d.slit_turn = square_lines(slit, &g, &cam, &d);
        eprintln!("RESQUARE slit turn {}", d.slit_turn);
    }
    // `AIM_CHIEF=<iterations>`: the Fresnels fitted so the chief rays cross
    // the diaphragm's centre (see [`aim_chief`]).
    if let Ok(n) = std::env::var("AIM_CHIEF") {
        std::env::remove_var("AIM_CHIEF");
        d = aim_chief(slit, &g, &cam, &d, n.parse().unwrap_or(200));
        std::env::set_var("F4_CURV", d.f4_curv.to_string());
        std::env::set_var("FRESNEL_ASPH", format!("{} {} {}", d.fresnel_asph[0], d.fresnel_asph[1], d.fresnel_asph[2]));
        std::env::set_var("FRESNEL_V", d.fresnel_shift_v.to_string());
        std::env::set_var("FIT", format!("{} {} {} {} {}", d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.focus_shift, d.slit_turn));
    }
    // `FRESNEL_FIT=<iterations>`: the field Fresnels and focus fitted on the
    // table (see [`fresnel_fit`]).
    if let Ok(n) = std::env::var("FRESNEL_FIT") {
        std::env::remove_var("FRESNEL_FIT");
        d = fresnel_fit(slit, &g, &cam, &d, n.parse().unwrap_or(200));
        std::env::set_var("FRESNEL_ASPH", format!("{} {} {}", d.fresnel_asph[0], d.fresnel_asph[1], d.fresnel_asph[2]));
        std::env::set_var("FRESNEL_V", d.fresnel_shift_v.to_string());
        std::env::set_var("FIT", format!("{} {} {} {} {}", d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.focus_shift, d.slit_turn));
    }
    // `PUPIL_FIT=<iterations>`: the field Fresnels fitted for the pupil's
    // image (see [`pupil_fit`]), in place of the chief rays' aim.
    if let Ok(n) = std::env::var("PUPIL_FIT") {
        std::env::remove_var("PUPIL_FIT");
        let (sc, k, sh, asph) = pupil_fit(slit, &g, &cam, &d, n.parse().unwrap_or(200));
        (d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.fresnel_asph) = (sc, k, sh, asph);
        std::env::set_var("FRESNEL_ASPH", format!("{} {} {}", asph[0], asph[1], asph[2]));
    }
    if let Ok(n) = std::env::var("KEYSTONE_FIT") {
        std::env::remove_var("KEYSTONE_FIT");
        d = keystone_fit(slit, &g, &cam, &d, n.parse().unwrap_or(80));
        std::env::set_var("FIT", format!("{} {} {} {} {}", d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.focus_shift, d.slit_turn));
    }
    if let Ok(n) = std::env::var("CAM_FIT") {
        let n: usize = n.parse().unwrap_or(150);
        if std::env::var_os("LINE_FIT").is_some() {
            (_, d.slit_turn) = straighten_lines(slit, &g, &cam, &d);
            if std::env::var_os("NO_AIM").is_none() {
            (d.fresnel_scale, d.fresnel_conic, d.fresnel_shift) = aim(slit, &g, &cam, &d);
        }
        }
        let (m2, m3, f) = camera_fit(slit, &g, &cam, &d, n);
        (d.m2_asph, d.m3_asph, d.focus_shift) = (m2, m3, f);
        if std::env::var_os("NO_AIM").is_none() {
            (d.fresnel_scale, d.fresnel_conic, d.fresnel_shift) = aim(slit, &g, &cam, &d);
        }
        if std::env::var_os("LINE_FIT").is_some() {
            (_, d.slit_turn) = straighten_lines(slit, &g, &cam, &d);
        } else {
            d.slit_turn = square_lines(slit, &g, &cam, &d);
        }
        let (m2, m3, f) = camera_fit(slit, &g, &cam, &d, n / 2);
        (d.m2_asph, d.m3_asph, d.focus_shift) = (m2, m3, f);
    }
    if let Ok(path) = std::env::var("CAM_EXPORT") {
        camera_export(slit, &g, &cam, &d, &path);
    }
    // `CAM_FATE="nm ..."`, `CAM_FATE_H="h ..."` (fractions of the slit's
    // half height): every pair.
    if let Ok(v) = std::env::var("CAM_FATE") {
        let hs: Vec<f64> = std::env::var("CAM_FATE_H").map(|q| q.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_else(|_| vec![0.0]);
        for l in v.split_whitespace().filter_map(|x| x.parse::<f64>().ok()) {
            for &h in &hs {
                camera_fate(slit, &g, &cam, &d, l / 1e3, h);
            }
        }
    }
    if std::env::var_os("CAM_SPOTS").is_some() {
        camera_spots(slit, &g, &cam, &d);
    }
    eprintln!("SLIT_K={}", slit_curve());
    // `PUPIL_REPORT`: where each beam crosses the projection lens's
    // diaphragm, across (along the spectrum's image) and up (across the
    // bands, the slab's height), against its 91 mm radius.
    if std::env::var_os("PUPIL_REPORT").is_some() {
        let mut sc = Scene3::new();
        let p = optics(&mut sc, slit, &g, &cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, stop_probe: true, ..d });
        if let Some((probe, c, axis)) = p.stop_probe {
            let up = Vec3::Y;
            let across = axis.cross(up).normalize();
            let a = 0.5 * BEAM / F_COLL;
            let aw = 0.5 * beam_along() / F_COLL;
            for h in [-0.8, 0.0, 0.8] {
                let mut line = Vec::new();
                for l in [0.378, 0.42, 0.47, 0.53, 0.59, 0.64, 0.688] {
                    let (_, chief) = sun_ray(slit, d.slit_turn, h * 0.5 * d.slit_height);
                    let from = slit_point(slit, d.slit_turn, h * 0.5 * d.slit_height, 0.0);
                    let e1 = chief.cross(Vec3::Y).normalize();
                    let e2 = e1.cross(chief);
                    let mut probes = vec![(from - chief * 20.0, chief, l)];
                    for k in 0..16 {
                        let ang = k as f64 * std::f64::consts::PI / 8.0;
                        let dir = (chief + (e1 * (ang.cos() * aw) + e2 * (ang.sin() * a)) * 0.95).normalize();
                        probes.push((from - dir * 20.0, dir, l));
                    }
                    let hits = land_many(&sc, probe, &probes);
                    let pts: Vec<(f64, f64)> = hits.iter().flatten().map(|q| ((*q - c).dot(across), (*q - c).dot(up))).collect();
                    let chief_at = hits[0].map(|q| ((q - c).dot(across), (q - c).dot(up)));
                    let reach = pts.iter().map(|q| q.0.hypot(q.1)).fold(0.0, f64::max);
                    let top = pts.iter().map(|q| q.1.abs()).fold(0.0, f64::max);
                    line.push(format!(
                        "{:.0}: chief {} reach {reach:.0} up {top:.0} ({}/17)",
                        l * 1e3,
                        chief_at.map_or("miss".into(), |q| format!("({:+.0},{:+.0})", q.0, q.1)),
                        pts.len()
                    ));
                }
                eprintln!("PUPIL h {h:+.1}: {}", line.join(" | "));
            }
        }
    }
    eprintln!("CAM=\"{} {} {} {} {}\"", d.m2_asph[0], d.m2_asph[1], d.m2_asph[2], d.m3_asph[0], d.m3_asph[1]);
    eprintln!("FIT=\"{} {} {} {} {}\"", d.fresnel_scale, d.fresnel_conic, d.fresnel_shift, d.focus_shift, d.slit_turn);
    eprintln!("FRESNEL_ASPH=\"{} {} {}\" FRESNEL_V={}", d.fresnel_asph[0], d.fresnel_asph[1], d.fresnel_asph[2], d.fresnel_shift_v);
    if std::env::var_os("SPEC_DIAG").is_some() {
        diagnose(slit, &g, &cam, &d);
    }
    // Solve the lamps' slits on the optics alone, then place everything.
    let mut scratch = Scene3::new();
    let p = optics(&mut scratch, slit, &g, &cam, &Design { probe_table: true, ..d });
    // `SUN_ONLY` leaves the lamps out, for a quick look at the sun's band.
    let lamps: Vec<Lamp> = if !lamps_on() || std::env::var_os("SUN_ONLY").is_some() { Vec::new() } else { lamps() };
    let slits: Vec<LampSlit> = lamps
        .iter()
        .enumerate()
        .map(|(i, l)| lamp_slit(&scratch, &p, slit, 0.5 * d.slit_height, l, i))
        .collect();
    let p = optics(s, slit, &g, &cam, &d);
    let glass = s.catalog.put(&n_bk7());
    let mut bands = vec![("Sun", -0.5 * SPECTRUM_HEIGHT, 0.5 * SPECTRUM_HEIGHT)];
    let mut sources = Vec::new();
    for (lamp, ls) in lamps.iter().zip(&slits) {
        sources.extend(lamp_arm(s, lamp, ls, glass));
        bands.push((lamp.name, ls.band.0, ls.band.1));
    }
    // The sun's band from its own edges, for the numbers.
    if let Some(sun) = bands.first_mut() {
        let v = |h: f64| {
            let (from, dir) = sun_ray(slit, d.slit_turn, h);
            land(s, p.table, from, dir, 0.55).map(|q| table_v(q, p.table_centre))
        };
        let (c, a, b) = (v(0.0), v(0.49 * d.slit_height), v(-0.49 * d.slit_height));
        if let (Some(c), Some(a), Some(b)) = (c, a, b) {
            let (a, b) = (c + (a - c) / 0.98, c + (b - c) / 0.98);
            *sun = ("Sun", a.min(b), a.max(b));
        }
    }
    Spectrograph {
        pickoff,
        pickoff_size: pickoff_size(d.slit_height, d.slit_turn),
        pickoff_along: {
            let a = pickoff_run(slit, d.slit_turn);
            (a - Vec3::X * a.x).normalize()
        },
        pickoff_bow: pickoff_bow(slit, d.slit_height, d.slit_turn),
        pickoff_depth: {
            let (ph, _) = pickoff_size(d.slit_height, d.slit_turn);
            let reach = (pickoff_run(slit, d.slit_turn).x * 0.5 * ph).abs();
            if 2.0 * reach + 10.0 > TUBE_GAP {
                eprintln!("warning: the pick-off strip reaches {reach:.0} mm along the tube either side, beyond the {TUBE_GAP:.0} mm gap");
            }
            let (ph, pw) = pickoff_size(d.slit_height, d.slit_turn);
            let run = pickoff_run(slit, d.slit_turn);
            let pickoff = slit + pickoff_offset();
            let c1 = condenser_mirror(slit, d.slit_turn);
            let n_p = ((c1 - pickoff).normalize() - Vec3::X).normalize();
            let across = n_p.cross(run).normalize();
            (run.x * 0.5 * ph).abs() + (across.x * 0.5 * pw * std::f64::consts::SQRT_2).abs()
        },
        slit,
        slit_height: d.slit_height,
        table_centre: p.table_centre,
        intermediate: p.intermediate,
        table: p.table,
        entrance: p.entrance,
        clearances: p.clearances.clone(),
        design: d,
        lamp_sources: sources,
        bands,
        lamp_slit_bows: slits
            .iter()
            .map(|l| {
                let (lo, hi) = l.line.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), q| (a.min(q.1), b.max(q.1)));
                hi - lo
            })
            .collect(),
    }
}

/// The camera alone on one level plane at the intermediate spectrum: for
/// each slit point (0.8 of its height either side, and the middle) and
/// wavelength, a grid over the collimator's elliptical cone lands on two
/// level planes before the field lens. Returns, per group, each ray's place
/// along the spectrum on the lower plane and its slope per mm up, and the
/// rays lost on the way; and the planes' heights.
type CameraLines = (Vec<Vec<(f64, f64)>>, usize, f64, f64);

fn camera_lines(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, lambdas: &[f64], n: usize) -> CameraLines {
    let mut sc = Scene3::new();
    let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..*d });
    let u_al = table_axes().0;
    let (ya, yb) = (p.intermediate.y - knob("PROBE_LO", 45.0), p.intermediate.y - knob("PROBE_HI", 20.0));
    let plane = |sc: &mut Scene3, y: f64| detector(sc, Vec3::new(p.intermediate.x, y, p.intermediate.z), Vec3::Y, u_al, 900.0, 200.0, [1, 1], DetectorTermination::ProbeAndContinue, "camera probe");
    let (pa, pb) = (plane(&mut sc, ya), plane(&mut sc, yb));
    let (a, aw) = (0.5 * BEAM / F_COLL, 0.5 * beam_along() / F_COLL);
    let mut pupil = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (u, v) = (2.0 * (i as f64 + 0.5) / n as f64 - 1.0, 2.0 * (j as f64 + 0.5) / n as f64 - 1.0);
            if u * u + v * v <= 1.0 {
                pupil.push((u, v));
            }
        }
    }
    let mut probes = Vec::new();
    for h in [-0.8, 0.0, 0.8] {
        let (_, chief) = sun_ray(slit, d.slit_turn, h * 0.5 * d.slit_height);
        let from = slit_point(slit, d.slit_turn, h * 0.5 * d.slit_height, 0.0);
        let e1 = chief.cross(Vec3::Y).normalize();
        let e2 = e1.cross(chief);
        for &l in lambdas {
            for &(u, v) in &pupil {
                let dir = (chief + e1 * (u * aw) + e2 * (v * a)).normalize();
                probes.push((from - dir * 20.0, dir, l));
            }
        }
    }
    let (ha, hb) = (land_many(&sc, pa, &probes), land_many(&sc, pb, &probes));
    let mut lost = 0;
    let groups = ha
        .chunks(pupil.len())
        .zip(hb.chunks(pupil.len()))
        .map(|(ga, gb)| {
            ga.iter()
                .zip(gb)
                .filter_map(|(qa, qb)| match (qa, qb) {
                    (Some(qa), Some(qb)) => Some((qa.dot(u_al), (qb.dot(u_al) - qa.dot(u_al)) / (yb - ya))),
                    _ => {
                        lost += 1;
                        None
                    }
                })
                .collect()
        })
        .collect();
    (groups, lost, ya, yb)
}

/// `CAMERA_FIT2=<iterations>`: the camera's layout for the intermediate
/// spectrum itself (no relay): the grating's distance from M2, the fold's
/// distance and radius, M2's lean, M2's conic and the fold's r^4 term, onto
/// the spread along the spectrum on the one level plane that suits every
/// group best, a lost ray counting 1 mm². Leaves the layout in the
/// environment.
fn camera_fit2(slit: Vec3, d0: &Design, iters: usize) {
    let lambdas = [0.378, 0.43, 0.486, 0.53, 0.589, 0.64, 0.688];
    let set = |q: &[f64; 6]| {
        std::env::set_var("GRATING_AT", q[0].to_string());
        std::env::set_var("FOLD_AT", q[1].to_string());
        std::env::set_var("FOLD_RATIO", q[2].to_string());
        std::env::set_var("CAMERA_TILT", q[3].to_string());
    };
    type Eval = (f64, f64, usize, Vec<f64>, Vec<f64>);
    let eval = |q: [f64; 6]| -> Option<Eval> {
        if !(0.6..=1.2).contains(&q[0]) || !(0.5..=0.97).contains(&q[1]) || !(0.3..=20.0).contains(&q[2]) || !(1.5..=knob("CAMFIT2_TILT_MAX", 10.0)).contains(&q[3]) {
            return None;
        }
        set(&q);
        let (g, cam, d) = solve();
        let dd = Design { m2_asph: [q[4], 0.0, 0.0], m3_asph: [q[5], 0.0], slit_turn: d0.slit_turn, ..d };
        let (groups, lost, ya, yb) = camera_lines(slit, &g, &cam, &dd, &lambdas, 8);
        // The common plane: t from the lower plane, least total spread.
        let (mut num, mut den) = (0.0, 0.0);
        for gr in &groups {
            let m = gr.len() as f64;
            if m < 3.0 {
                continue;
            }
            let mx = gr.iter().map(|r| r.0).sum::<f64>() / m;
            let ms = gr.iter().map(|r| r.1).sum::<f64>() / m;
            num += gr.iter().map(|r| (r.0 - mx) * (r.1 - ms)).sum::<f64>();
            den += gr.iter().map(|r| (r.1 - ms).powi(2)).sum::<f64>();
        }
        let tt = -num / den;
        // Each group's own best focus, mm above the common plane.
        let foci: Vec<f64> = groups
            .iter()
            .map(|gr| {
                let m = gr.len() as f64;
                if m < 3.0 {
                    return f64::NAN;
                }
                let mx = gr.iter().map(|r| r.0).sum::<f64>() / m;
                let ms = gr.iter().map(|r| r.1).sum::<f64>() / m;
                let c = gr.iter().map(|r| (r.0 - mx) * (r.1 - ms)).sum::<f64>();
                let v = gr.iter().map(|r| (r.1 - ms).powi(2)).sum::<f64>();
                -c / v - tt
            })
            .collect();
        let rms: Vec<f64> = groups
            .iter()
            .map(|gr| {
                let m = gr.len() as f64;
                if m < 3.0 {
                    return 10.0;
                }
                let xs: Vec<f64> = gr.iter().map(|r| r.0 + tt * r.1).collect();
                let c = xs.iter().sum::<f64>() / m;
                (xs.iter().map(|x| (x - c).powi(2)).sum::<f64>() / m).sqrt()
            })
            .collect();
        let total: usize = groups.iter().map(|g| g.len()).sum::<usize>() + lost;
        let merit = rms.iter().map(|r| r * r).sum::<f64>() / rms.len() as f64 + lost as f64 / total as f64;
        Some((merit, tt + (ya - yb) + 20.0, lost, rms, foci))
    };
    let merit = |q: [f64; 6]| eval(q).map_or(1e6, |e| e.0);
    let show = |q: [f64; 6], what: &str| {
        let Some((m, plane, lost, rms, foci)) = eval(q) else { return };
        let f: Vec<String> = foci.chunks(lambdas.len()).map(|c| c.iter().map(|x| format!("{x:+.1}")).collect::<Vec<_>>().join(" ")).collect();
        let r: Vec<String> = rms.chunks(lambdas.len()).map(|c| c.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" ")).collect();
        eprintln!("CAMFIT2 {what}: GRATING_AT={} FOLD_AT={} FOLD_RATIO={} CAMERA_TILT={} CAM=\"{} 0 0 {} 0\"; merit {m:.5}, plane {plane:+.1} mm, {lost} lost", q[0], q[1], q[2], q[3], q[4], q[5]);
        eprintln!("CAMFIT2 {what} rms along (mm), 378..688 nm, slit -0.8 | 0 | +0.8: {}", r.join(" | "));
        eprintln!("CAMFIT2 {what} own best focus (mm above the plane), same order: {}", f.join(" | "));
    };
    let mut x = [knob("GRATING_AT", GRATING_AT), knob("FOLD_AT", FOLD_AT), knob("FOLD_RATIO", FOLD_RADIUS_RATIO), knob("CAMERA_TILT", CAMERA_TILT_DEG), d0.m2_asph[0], d0.m3_asph[0]];
    show(x, "start");
    for round in 0..knob("CAMFIT2_ROUNDS", 3.0) as usize {
        x = nelder_mead(merit, x, [0.03, 0.03, 0.3, 0.5, 0.2, 0.3], iters);
        show(x, &format!("round {round}"));
    }
    set(&x);
    std::env::set_var("CAM", format!("{} 0 0 {} 0", x[4], x[5]));
}

/// `CAM_EXPORT=<file>`: the camera's rays just before the field lens, for
/// designing the relay against them. For nine wavelengths, five slit points
/// (0.9 and 0.45 of its half height either side, and the middle) and a grid
/// over the collimator's elliptical cone, each ray as it crosses the level
/// plane 90 mm under the intermediate spectrum, in the relay's frame: origin
/// the intermediate spectrum's middle, z up the relay's axis (`RELAY_TILT`
/// included), x along the spectrum, y = z x x. Writes
/// `nm,h,u,v,x,y,z,dx,dy,dz`, and the header comments carry the path along
/// the relay's axis to F4, the lens's first vertex, F5 and the table.
fn camera_export(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, path: &str) {
    let mut sc = Scene3::new();
    let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..*d });
    let rot = downstream_turn(Vec3::ZERO);
    let ez = rot.rotate(turn_x(Vec3::Y, knob("RELAY_TILT", 0.0).to_radians())).normalize();
    let ex0 = table_axes().0;
    let ex = (ex0 - ez * ex0.dot(ez)).normalize();
    let ey = ez.cross(ex);
    let o = p.intermediate;
    let u_al = table_axes().0;
    let (ya, yb) = (o.y - knob("PROBE_LO", 45.0), o.y - knob("PROBE_HI", 20.0));
    let plane = |sc: &mut Scene3, y: f64| detector(sc, Vec3::new(o.x, y, o.z), Vec3::Y, u_al, 900.0, 200.0, [1, 1], DetectorTermination::ProbeAndContinue, "camera export probe");
    let (pa, pb) = (plane(&mut sc, ya), plane(&mut sc, yb));
    let (a, aw) = (0.5 * BEAM / F_COLL, 0.5 * beam_along() / F_COLL);
    let n = 10;
    let mut pupil = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (u, v) = (2.0 * (i as f64 + 0.5) / n as f64 - 1.0, 2.0 * (j as f64 + 0.5) / n as f64 - 1.0);
            if u * u + v * v <= 1.0 {
                pupil.push((u, v));
            }
        }
    }
    pupil.push((0.0, 0.0));
    let lambdas: Vec<f64> = (0..9).map(|i| 0.378 + (0.688 - 0.378) * i as f64 / 8.0).collect();
    let hs = [-0.9, -0.45, 0.0, 0.45, 0.9];
    let mut probes = Vec::new();
    let mut tags = Vec::new();
    for &h in &hs {
        let (_, chief) = sun_ray(slit, d.slit_turn, h * 0.5 * d.slit_height);
        let from = slit_point(slit, d.slit_turn, h * 0.5 * d.slit_height, 0.0);
        // The cone's long axis across the slit, its short one along it (the
        // pick-off's image turns with the slit).
        let e1 = chief.cross(slit_along(d.slit_turn)).normalize();
        let e2 = e1.cross(chief);
        for &l in &lambdas {
            for &(u, v) in &pupil {
                let dir = (chief + e1 * (u * aw) + e2 * (v * a)).normalize();
                probes.push((from - dir * 20.0, dir, l));
                tags.push((l, h, u, v));
            }
        }
    }
    let (ha, hb) = (land_many(&sc, pa, &probes), land_many(&sc, pb, &probes));
    // Where the relay as built puts each ray on the table, to check a model
    // of it against the tracer.
    let land = land_many(&sc, p.table, &probes);
    let (tu, tw) = table_axes();
    let rel = |q: Vec3| {
        let r = q - o;
        (r.dot(ex), r.dot(ey), r.dot(ez))
    };
    let (fc, a0, lv) = (rel(p.field_centre), rel(p.lens_axis0), rel(p.lens_vertex_unfolded));
    let mut csv = String::new();
    csv.push_str(&format!("# relay frame: origin {:.3} {:.3} {:.3}; ez {:.6} {:.6} {:.6}; ex {:.6} {:.6} {:.6}\n", o.x, o.y, o.z, ez.x, ez.y, ez.z, ex.x, ex.y, ex.z));
    csv.push_str(&format!(
        "# F4 at {:.1} mm up the axis; the lens's first vertex {:.1} mm (unfolded); throw from its last vertex to the table {:.1} mm; lens length {:.1}; table tilt F5_TILT {:.4} deg; magnification {:.4}; slit height {:.3} mm\n",
        knob("F4_AFTER", F4_AFTER),
        d.vertex_distance + d.focus_shift,
        d.throw,
        d.lens_length,
        knob("F5_TILT", 0.0),
        d.magnification,
        d.slit_height
    ));
    csv.push_str(&format!(
        "# field lens centre {:.4} {:.4} {:.4}; lens axis at the intermediate {:.4} {:.4} {:.4}; first vertex unfolded {:.4} {:.4} {:.4}; field f {:.3}, scale {:.6}, conic {:.6}, asph {:.8} {:.8} {:.8}\n",
        fc.0, fc.1, fc.2, a0.0, a0.1, a0.2, lv.0, lv.1, lv.2, d.f_field, d.fresnel_scale, d.fresnel_conic, d.fresnel_asph[0], d.fresnel_asph[1], d.fresnel_asph[2]
    ));
    csv.push_str("nm,h,u,v,x,y,z,dx,dy,dz,tu,tw\n");
    let mut kept = 0;
    for (((qa, qb), q), (l, h, u, v)) in ha.iter().zip(&hb).zip(&land).zip(&tags) {
        if let (Some(qa), Some(qb)) = (qa, qb) {
            let r = *qa - o;
            let dvec = (*qb - *qa).normalize();
            let (a, b) = q.map_or((f64::NAN, f64::NAN), |q| ((q - p.table_centre).dot(tu), (q - p.table_centre).dot(tw)));
            csv.push_str(&format!(
                "{:.1},{h},{u:.4},{v:.4},{:.5},{:.5},{:.5},{:.8},{:.8},{:.8},{a:.4},{b:.4}\n",
                l * 1e3,
                r.dot(ex),
                r.dot(ey),
                r.dot(ez),
                dvec.dot(ex),
                dvec.dot(ey),
                dvec.dot(ez)
            ));
            kept += 1;
        }
    }
    std::fs::write(path, csv).expect("write the camera's rays");
    eprintln!("CAMEXPORT {kept} of {} rays to {path}", probes.len());
    if let Ok(jpath) = std::env::var("JOINT_EXPORT") {
        joint_export(&sc, &probes, &tags, rel, &(ex, ey, ez, o), &jpath);
    }
}

/// `JOINT_EXPORT=<file>` (with `CAM_EXPORT`): the same rays where they leave
/// the grating, and the camera mirrors M2 and M3 as built, all in the relay's
/// frame: the start of a model that traces camera and relay together.
fn joint_export(
    sc: &Scene3,
    probes: &[(Vec3, Vec3, f64)],
    tags: &[(f64, f64, f64, f64)],
    rel: impl Fn(Vec3) -> (f64, f64, f64),
    axes: &(Vec3, Vec3, Vec3, Vec3),
    path: &str,
) {
    use oc_trace::TraceEventKind;
    let (ex, ey, ez, _) = *axes;
    let step = 1e-11;
    let rays: Vec<oc_trace::LaunchRay3> = probes
        .iter()
        .enumerate()
        .filter_map(|(k, &(from, dir, l))| {
            oc_core::vec3::Ray3::new(from, dir.normalize()).ok().map(|ray| oc_trace::LaunchRay3 { ray, lambda_um: l + step * k as f64, flux: 1.0, source: None, pol: None })
        })
        .collect();
    let cfg = oc_trace::TraceConfig3 { ghosts: false, max_depth: 64, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(sc), ..oc_trace::TraceConfig3::default() };
    let ev = oc_trace::trace3(sc, &rays, &cfg).events;
    let id_of = |start: &str| sc.surfaces.iter().find(|o| o.common.label.starts_with(start)).map(|o| o.common.id.0);
    let (grating, m2, m3) = (id_of("Reflection grating"), id_of("Camera M2"), id_of("Fold M3"));
    // Per probe: the grating's hit, then M2's and M3's.
    let mut hits: std::collections::HashMap<usize, [Option<Vec3>; 3]> = Default::default();
    for e in &ev {
        if let TraceEventKind::Interact { surface, .. } = &e.kind {
            let id = Some(surface.object.0);
            let slot = if id == grating { 0 } else if id == m2 { 1 } else if id == m3 { 2 } else { continue };
            // Which probe: its wavelength tag past the nearest probe wavelength.
            let l = tags.iter().map(|t| t.0).min_by(|a, b| (a - e.lambda_um).abs().total_cmp(&(b - e.lambda_um).abs())).unwrap_or(0.0);
            let k = ((e.lambda_um - l) / step).round();
            if k >= 0.0 && (k as usize) < probes.len() && probes[k as usize].2 == l {
                let k = k as usize;
                let h = hits.entry(k).or_insert([None; 3]);
                if h[slot].is_none() {
                    h[slot] = Some(e.point);
                }
            }
        }
    }
    let mut out = String::new();
    for (name, id) in [("M2", m2), ("M3", m3)] {
        let o = sc.surfaces.iter().find(|o| Some(o.common.id.0) == id).expect("the camera mirror");
        let t = o.common.transform;
        let (v, z, y) = (rel(t.transform_point(Vec3::ZERO)), t.rotate(Vec3::Z), t.rotate(Vec3::Y));
        let (c, k, a) = match &o.geometry {
            SurfaceGeometry3::SurfaceOfRevolution { sag: SagModel::EvenAsphere { curvature_per_mm, conic, coeffs }, .. } => (*curvature_per_mm, *conic, coeffs.to_vec()),
            SurfaceGeometry3::SurfaceOfRevolution { sag: SagModel::Conic { curvature_per_mm, conic }, .. } => (*curvature_per_mm, *conic, vec![]),
            _ => panic!("a surface of revolution"),
        };
        out.push_str(&format!(
            "# {name} vertex {:.6} {:.6} {:.6}; axis {:.9} {:.9} {:.9}; y {:.9} {:.9} {:.9}; c {:.12e}; k {:.9}; a {}\n",
            v.0,
            v.1,
            v.2,
            z.dot(ex),
            z.dot(ey),
            z.dot(ez),
            y.dot(ex),
            y.dot(ey),
            y.dot(ez),
            c,
            k,
            a.iter().map(|q| format!("{q:.9e}")).collect::<Vec<_>>().join(" ")
        ));
    }
    out.push_str("nm,h,u,v,x,y,z,dx,dy,dz,m2x,m2y,m2z,m3x,m3y,m3z\n");
    let mut kept = 0;
    for (k, &(l, h, u, v)) in tags.iter().enumerate() {
        let Some([Some(g), Some(p2), Some(p3)]) = hits.get(&k).copied() else { continue };
        let (gx, gy, gz) = rel(g);
        let dv = (p2 - g).normalize();
        let (a2, b2, c2) = rel(p2);
        let (a3, b3, c3) = rel(p3);
        out.push_str(&format!(
            "{:.1},{h},{u:.4},{v:.4},{gx:.6},{gy:.6},{gz:.6},{:.10},{:.10},{:.10},{a2:.6},{b2:.6},{c2:.6},{a3:.6},{b3:.6},{c3:.6}\n",
            l * 1e3,
            dv.dot(ex),
            dv.dot(ey),
            dv.dot(ez)
        ));
        kept += 1;
    }
    std::fs::write(path, out).expect("write the joint export");
    eprintln!("JOINTEXPORT {kept} of {} rays to {path}", probes.len());
}

/// `CAM_FATE=<nm>`: where the camera's rays from the slit's centre that miss
/// the intermediate spectrum end, one wavelength: the object each lost ray
/// last met, tallied.
fn camera_fate(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design, l: f64, h: f64) {
    use oc_trace::TraceEventKind;
    let mut sc = Scene3::new();
    let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..*d });
    let u_al = table_axes().0;
    // `CAM_FATE_TABLE=1`: follow the rays on to the table instead.
    let probe = if std::env::var_os("CAM_FATE_TABLE").is_some() {
        p.table
    } else {
        detector(&mut sc, Vec3::new(p.intermediate.x, p.intermediate.y - 20.0, p.intermediate.z), Vec3::Y, u_al, 1200.0, 400.0, [1, 1], DetectorTermination::Terminate, "camera fate probe")
    };
    let (a, aw) = (0.5 * BEAM / F_COLL, 0.5 * beam_along() / F_COLL);
    let hq = h * 0.5 * d.slit_height;
    let (_, chief) = sun_ray(slit, d.slit_turn, hq);
    let from = slit_point(slit, d.slit_turn, hq, 0.0);
    let e1 = chief.cross(slit_along(d.slit_turn)).normalize();
    let e2 = e1.cross(chief);
    let n = 15;
    let mut rays = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (u, v) = (2.0 * (i as f64 + 0.5) / n as f64 - 1.0, 2.0 * (j as f64 + 0.5) / n as f64 - 1.0);
            if u * u + v * v <= 1.0 {
                let dir = (chief + e1 * (u * aw) + e2 * (v * a)).normalize();
                if let Ok(ray) = oc_core::vec3::Ray3::new(from - dir * 20.0, dir) {
                    rays.push(oc_trace::LaunchRay3 { ray, lambda_um: l + 1e-9 * rays.len() as f64, flux: 1.0, source: None, pol: None });
                }
            }
        }
    }
    let cfg = oc_trace::TraceConfig3 { ghosts: false, max_depth: 64, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(&sc), ..oc_trace::TraceConfig3::default() };
    let ev = oc_trace::trace3(&sc, &rays, &cfg).events;
    let mut names: std::collections::HashMap<u64, String> = Default::default();
    for o in &sc.surfaces {
        names.insert(o.common.id.0, o.common.label.clone());
    }
    for o in &sc.bodies {
        names.insert(o.common.id.0, o.common.label.clone());
    }
    for o in &sc.apertures {
        names.insert(o.common.id.0, o.common.label.clone());
    }
    let mut last: std::collections::HashMap<i64, String> = Default::default();
    let mut reached: std::collections::HashSet<i64> = Default::default();
    let mut off_m3: std::collections::HashSet<i64> = Default::default();
    let mut chain: std::collections::HashMap<i64, Vec<String>> = Default::default();
    for e in &ev {
        let k = ((e.lambda_um - l) / 1e-9).round() as i64;
        match &e.kind {
            TraceEventKind::Interact { surface, children, .. } => chain.entry(k).or_default().push(format!("{}{} @ ({:.0}, {:.0}, {:.0})", names.get(&surface.object.0).map_or("?", |q| q.split(',').next().unwrap_or("")), if children.len() > 1 { format!(" [{} children]", children.len()) } else { String::new() }, e.point.x, e.point.y, e.point.z)),
            TraceEventKind::Escaped => chain.entry(k).or_default().push(format!("ESCAPED @ ({:.0}, {:.0}, {:.0})", e.point.x, e.point.y, e.point.z)),
            TraceEventKind::ApertureBlocked { aperture } => chain.entry(k).or_default().push(format!("BLOCKED by {}", names.get(&aperture.0).cloned().unwrap_or_default())),
            _ => {}
        }
        match &e.kind {
            // Only the light coming off M3 counts at the probe under the
            // intermediate: the M2 to M3 beam can cross it too.
            TraceEventKind::DetectorHit { detector, .. } if detector.0 == probe.0 => {
                if std::env::var_os("CAM_FATE_TABLE").is_some() || off_m3.contains(&k) {
                    reached.insert(k);
                }
            }
            TraceEventKind::Interact { surface, .. } => {
                let name = names.get(&surface.object.0).map_or("?".into(), |q| q.split(',').next().unwrap_or("").to_string());
                if name.starts_with("Fold M3") {
                    off_m3.insert(k);
                }
                last.insert(k, name);
            }
            TraceEventKind::ApertureBlocked { aperture } => {
                last.insert(k, format!("blocked: {}", names.get(&aperture.0).cloned().unwrap_or_default()));
            }
            TraceEventKind::Escaped => {
                let prev = last.get(&k).cloned().unwrap_or_default();
                last.insert(k, format!("escaped after {prev}"));
            }
            _ => {}
        }
    }
    let mut tally: std::collections::HashMap<String, usize> = Default::default();
    let mut shown = 0;
    for k in 0..rays.len() as i64 {
        if !reached.contains(&k) && shown < 3 && std::env::var_os("CAM_FATE_WHO").is_some() && chain.get(&k).is_some_and(|c| std::env::var_os("CAM_FATE_ANY").is_some() || c.iter().any(|q| q.contains("rating"))) {
            shown += 1;
            eprintln!("CAMFATE lost ray {k}: {}", chain.get(&k).map_or(String::new(), |c| c.join(" -> ")));
        }
        if !reached.contains(&k) {
            *tally.entry(last.get(&k).cloned().unwrap_or_else(|| "nothing".into())).or_default() += 1;
        }
    }
    let mut v: Vec<_> = tally.into_iter().collect();
    v.sort_by_key(|q| std::cmp::Reverse(q.1));
    eprintln!("CAMFATE {:.0} nm h {h:+.2}: {} of {} reach; lost: {:?}", l * 1e3, reached.len(), rays.len(), v);
}

/// `CAM_SPOTS`: the camera alone, at the intermediate spectrum. From the
/// slit's centre, a grid over the collimator's (elliptical) cone at each of
/// five wavelengths lands on two level planes before the field lens; each
/// ray's line through them gives the best focus (least spread along the
/// spectrum) and, there, the spread along and across. The along spread is
/// split by its fit over the pupil: straight terms (focus, astigmatism),
/// squared (coma-like), cubed (spherical-like) and what is left.
fn camera_spots(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) {
    let mut sc = Scene3::new();
    let p = optics(&mut sc, slit, g, cam, &Design { fresnel_pitch: FIT_PITCH, probe_table: true, ..*d });
    let (u_al, w_ac) = table_axes();
    let (ya, yb) = (p.intermediate.y - knob("PROBE_LO", 45.0), p.intermediate.y - knob("PROBE_HI", 20.0));
    let plane = |sc: &mut Scene3, y: f64| {
        detector(sc, Vec3::new(p.intermediate.x, y, p.intermediate.z), Vec3::Y, u_al, 900.0, 200.0, [1, 1], DetectorTermination::ProbeAndContinue, "camera spot probe")
    };
    let (pa, pb) = (plane(&mut sc, ya), plane(&mut sc, yb));
    let (a, aw) = (0.5 * BEAM / F_COLL, 0.5 * beam_along() / F_COLL);
    // `CAM_SPOTS_H`: the slit point, a fraction of its half height.
    let hs = knob("CAM_SPOTS_H", 0.0) * 0.5 * d.slit_height;
    let (_, chief) = sun_ray(slit, d.slit_turn, hs);
    let from = slit_point(slit, d.slit_turn, hs, 0.0);
    let e1 = chief.cross(Vec3::Y).normalize();
    let e2 = e1.cross(chief);
    let n = 13;
    let mut pupil = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (u, v) = (2.0 * i as f64 / (n - 1) as f64 - 1.0, 2.0 * j as f64 / (n - 1) as f64 - 1.0);
            if u * u + v * v <= 1.0 {
                pupil.push((u, v));
            }
        }
    }
    for l in [0.382, 0.45, 0.53, 0.61, 0.687] {
        let probes: Vec<(Vec3, Vec3, f64)> = pupil
            .iter()
            .map(|&(u, v)| {
                let dir = (chief + e1 * (u * aw) + e2 * (v * a)).normalize();
                (from - dir * 20.0, dir, l)
            })
            .collect();
        let (ha, hb) = (land_many(&sc, pa, &probes), land_many(&sc, pb, &probes));
        let rays: Vec<(f64, f64, f64, f64, f64, f64)> = pupil
            .iter()
            .zip(ha.iter().zip(&hb))
            .filter_map(|(&(u, v), (qa, qb))| {
                let (qa, qb) = ((*qa)?, (*qb)?);
                let (xa, xb) = (qa.dot(u_al), qb.dot(u_al));
                let (za, zb) = (qa.dot(w_ac), qb.dot(w_ac));
                Some((u, v, xa, (xb - xa) / (yb - ya), za, (zb - za) / (yb - ya)))
            })
            .collect();
        let m = rays.len() as f64;
        if m < 10.0 {
            eprintln!("CAMSPOT {:.0} nm: {} of {} rays through", l * 1e3, rays.len(), pupil.len());
            continue;
        }
        type Ray6 = (f64, f64, f64, f64, f64, f64);
        let mean = |f: &dyn Fn(&Ray6) -> f64| rays.iter().map(f).sum::<f64>() / m;
        let (mx, ms) = (mean(&|r| r.2), mean(&|r| r.3));
        let cov = mean(&|r| (r.2 - mx) * (r.3 - ms));
        let var_s = mean(&|r| (r.3 - ms).powi(2));
        let t_best = -cov / var_s;
        let ex: Vec<f64> = rays.iter().map(|r| r.2 + t_best * r.3).collect();
        let ez: Vec<f64> = rays.iter().map(|r| r.4 + t_best * r.5).collect();
        let rms = |e: &[f64]| {
            let c = e.iter().sum::<f64>() / m;
            (e.iter().map(|x| (x - c).powi(2)).sum::<f64>() / m).sqrt()
        };
        // Least squares of the along error over the pupil, groups of terms
        // added in turn: what each group takes out.
        let basis = |u: f64, v: f64| -> [f64; 10] { let r2 = u * u + v * v; [1.0, u, v, u * u, v * v, u * v, u * r2, v * r2, u * u * u, v * v * v] };
        let fit_rms = |k: usize| {
            let mut ata = vec![vec![0.0; k]; k];
            let mut atb = vec![0.0; k];
            for (r, e) in rays.iter().zip(&ex) {
                let b = basis(r.0, r.1);
                for i in 0..k {
                    atb[i] += b[i] * e;
                    for j in 0..k {
                        ata[i][j] += b[i] * b[j];
                    }
                }
            }
            // Gaussian elimination.
            for c in 0..k {
                let piv = (c..k).max_by(|&i, &j| ata[i][c].abs().total_cmp(&ata[j][c].abs())).unwrap();
                ata.swap(c, piv);
                atb.swap(c, piv);
                for i in c + 1..k {
                    let f = ata[i][c] / ata[c][c];
                    // Row c is above row i, so the split borrows both at once.
                    let (above, from_i) = ata.split_at_mut(i);
                    for (a, b) in from_i[0][c..k].iter_mut().zip(&above[c][c..k]) {
                        *a -= f * b;
                    }
                    atb[i] -= f * atb[c];
                }
            }
            let mut x = vec![0.0; k];
            for i in (0..k).rev() {
                x[i] = (atb[i] - (i + 1..k).map(|j| ata[i][j] * x[j]).sum::<f64>()) / ata[i][i];
            }
            let res: Vec<f64> = rays.iter().zip(&ex).map(|(r, e)| e - basis(r.0, r.1)[..k].iter().zip(&x).map(|(b, c)| b * c).sum::<f64>()).collect();
            (rms(&res), x)
        };
        let (r0, (r1, _), (r2, q), (r3, _)) = (rms(&ex), fit_rms(3), fit_rms(6), fit_rms(10));
        eprintln!(
            "CAMSPOT {:.0} nm: {}/{} rays; best focus {:+.1} mm from the upper plane; along rms {:.3} mm (after straight terms {:.3}, squared {:.3}, cubed {:.3}); across rms {:.2}; u2 {:+.3} v2 {:+.3} uv {:+.3}",
            l * 1e3, rays.len(), pupil.len(), t_best + (ya - yb), r0, r1, r2, r3, rms(&ez), q[3], q[4], q[5]
        );
    }
}

/// The pick-off's strip, as the tube's light sees it, in `n` pieces along
/// its arc: each piece's centre, its run across the tube (no X), and its
/// half length along that run.
pub fn pickoff_pieces(sp: &Spectrograph, n: usize) -> Vec<(Vec3, Vec3, f64)> {
    let turn = sp.design.slit_turn;
    let h_all = 0.5 * PICKOFF_OVERFILL * sp.slit_height;
    let flat = |v: Vec3| v - Vec3::X * v.x;
    (0..n)
        .map(|i| {
            let h = h_all * (2.0 * (i as f64 + 0.5) / n as f64 - 1.0);
            let dh = h_all / n as f64;
            let (a, b) = (pickoff_point(sp.slit, turn, h - dh, 0.0), pickoff_point(sp.slit, turn, h + dh, 0.0));
            let run = flat(b - a);
            (pickoff_point(sp.slit, turn, h, 0.0), run.normalize(), 0.5 * run.length())
        })
        .collect()
}

/// The slit as a source sees it (`SLIT_SOURCE` in the bench), in `n` equal
/// pieces along its arc: each piece's centre, the direction its light
/// leaves in (from the condenser mirror, as the condenser lights it), and
/// the slit's run there.
pub fn slit_emitters(sp: &Spectrograph, n: usize) -> Vec<(Vec3, Vec3, Vec3)> {
    let turn = sp.design.slit_turn;
    let c1 = condenser_mirror(sp.slit, turn);
    (0..n)
        .map(|i| {
            let h = sp.slit_height * ((i as f64 + 0.5) / n as f64 - 0.5);
            let dh = 0.5 * sp.slit_height / n as f64;
            let p = slit_point(sp.slit, turn, h, 0.0);
            let run = (slit_point(sp.slit, turn, h + dh, 0.0) - slit_point(sp.slit, turn, h - dh, 0.0)).normalize();
            (p, (p - c1).normalize(), run)
        })
        .collect()
}

/// `SPEC_DIAG`: the design's numbers, where the chief rays cross the
/// projection lens's diaphragm, and the spots on the table. Each spot is
/// 400 rays from one point of the slit, filling the cone from the condenser
/// mirror, one wavelength each (picometres apart, so each ray's own strongest
/// hit is told from the lens's ghosts): where the median lands, the half
/// width holding half the rays, and the share within 3 mm.
fn diagnose(slit: Vec3, g: &GratingFrame, cam: &Camera, d: &Design) {
    eprintln!("SPEC {d:?}");
    let mut sc = Scene3::new();
    let p = optics(&mut sc, slit, g, cam, &Design { stop_probe: true, probe_table: true, ..*d });
    let probe_id = p.stop_probe.map_or(p.table, |q| q.0);
    let _ = probe_id;
    for (w, h) in &p.clearances {
        eprintln!("SPEC {w}: {:.0} mm above the table, {:.0} under the ceiling", h - TABLE_Y, CEILING_Y - h);
    }
    let (probe, c, axis) = p.stop_probe.unwrap_or((p.table, p.table_centre, Vec3::Y));
    let c1 = condenser_mirror(slit, d.slit_turn);
    let cfg = oc_trace::TraceConfig3 {
        record_detector_hits: true,
        max_depth: 40,
        bounds_mm: oc_trace::bounds_for_scene3(&sc),
        ..oc_trace::TraceConfig3::default()
    };
    if std::env::var_os("SPEC_STRIP").is_some() {
        // Rays straight down the tube onto points along the pick-off's arc:
        // where each lands on the slit, and what stops it.
        let cfg_e = oc_trace::TraceConfig3 { ghosts: false, record_events: true, ..cfg };
        for f in [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
            let h = f * PICKOFF_OVERFILL * d.slit_height;
            let target = pickoff_point(slit, d.slit_turn, h, 0.0);
            let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(target - Vec3::X * 25.0, Vec3::X).unwrap(), lambda_um: 0.53, flux: 1.0, source: None, pol: None };
            let res = oc_trace::trace3(&sc, &[lr], &cfg_e);
            let hits: Vec<String> = res.events.iter().filter_map(|e| match &e.kind {
                oc_trace::TraceEventKind::Interact { surface, .. } => sc.surfaces.iter().find(|o| o.common.id.0 == surface.object.0).map(|o| format!("{} @ ({:.1}, {:.1}, {:.1})", o.common.label.split(',').next().unwrap_or(""), e.point.x - slit.x, e.point.y - slit.y, e.point.z - slit.z)),
                oc_trace::TraceEventKind::Escaped => Some("escaped".into()),
                _ => None,
            }).take(4).collect();
            eprintln!("STRIP h {h:+.1} (target y {:+.1} z {:+.1} from the slit): {}", target.y - slit.y, target.z - slit.z, hits.join(" -> "));
        }
    }
    if std::env::var_os("SPEC_STRIP").is_some() {
        // The bench source's rectangle as a grid, straight and tilted: the
        // light along the slit at the entrance.
        let (ph, pw) = pickoff_size(d.slit_height, d.slit_turn);
        let bow = pickoff_bow(slit, d.slit_height, d.slit_turn);
        let pick = slit + pickoff_offset();
        for (tilt, name) in [(Vec3::ZERO, "straight"), (Vec3::Y * 0.012, "tilted +y"), (-Vec3::Y * 0.012, "tilted -y"), (Vec3::Z * 0.012, "tilted +z"), (-Vec3::Z * 0.012, "tilted -z")] {
            let mut rays = Vec::new();
            for i in 0..121 {
                for j in 0..15 {
                    let yy = (i as f64 / 120.0 - 0.5) * (ph + 4.0);
                    let zz = (j as f64 / 14.0 - 0.5) * (pw + 1.0 + 2.0 * bow);
                    let dir = (Vec3::X + tilt).normalize();
                    rays.push(oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(pick + Vec3::new(-25.0, yy, zz), dir).unwrap(), lambda_um: 0.53, flux: 1.0, source: None, pol: None });
                }
            }
            let res = oc_trace::trace3(&sc, &rays, &cfg);
            let mut bins = [0usize; 18];
            if let Some(det) = res.detectors.iter().find(|q| q.detector.0 == p.entrance.0) {
                for r in &det.records {
                    let v = r.point_world.y - slit.y;
                    let k = ((v / 46.8 + 1.0) * 9.0).floor();
                    if (0.0..18.0).contains(&k) {
                        bins[k as usize] += 1;
                    }
                }
            }
            eprintln!("SOURCE {name}: {:?}", bins);
            if name == "straight" || name == "tilted +y" {
                let cfg_e = oc_trace::TraceConfig3 { ghosts: false, record_events: true, ..cfg };
                for yy in [-170.0, -150.0, 150.0] {
                    let dir = (Vec3::X + tilt).normalize();
                    let target = pickoff_point(slit, d.slit_turn, -yy / 4.28, 0.0);
                    let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(target - dir * 25.0, dir).unwrap(), lambda_um: 0.53, flux: 1.0, source: None, pol: None };
                    let ev = oc_trace::trace3(&sc, &[lr], &cfg_e).events;
                    let chain: Vec<String> = ev.iter().filter_map(|e| match &e.kind {
                        oc_trace::TraceEventKind::Interact { surface, .. } => sc.surfaces.iter().find(|o| o.common.id.0 == surface.object.0).map(|o| format!("{} @ ({:.1}, {:.1}, {:.1})", o.common.label.split(',').next().unwrap_or(""), e.point.x - slit.x, e.point.y - slit.y, e.point.z - slit.z)),
                        oc_trace::TraceEventKind::DetectorHit { detector, .. } => Some(format!("detector {} @ y {:.1}", detector.0, e.point.y - slit.y)),
                        oc_trace::TraceEventKind::Escaped => Some("escaped".into()),
                        _ => None,
                    }).take(6).collect();
                    eprintln!("SOURCE {name} y {yy}: {}", chain.join(" -> "));
                }
            }
        }
    }
    if std::env::var_os("SPEC_COLL").is_some() {
        // Collimation: rays from slit points over the whole cone; after M1
        // each point's rays should be parallel.
        for h in [-0.45, 0.0, 0.45] {
            let (_, chief) = sun_ray(slit, d.slit_turn, h * d.slit_height);
            let target = slit_point(slit, d.slit_turn, h * d.slit_height, 0.0);
            let e1 = chief.cross(Vec3::Y).normalize();
            let e2 = e1.cross(chief);
            let mut dirs = Vec::new();
            for (a, b) in [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.7, 0.7), (-0.7, -0.7)] {
                let dir = (chief + (e1 * (a * beam_along() / BEAM) + e2 * b) * (0.9 * 0.5 * BEAM / F_COLL)).normalize();
                let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(target - dir * 20.0, dir).unwrap(), lambda_um: 0.53, flux: 1.0, source: None, pol: None };
                let segs = oc_trace::trace3(&sc, &[lr], &cfg).segments;
                // The segment leaving M1: the one that starts near M1's x.
                if let Some(q) = segs.iter().filter(|q| q.flux_start > 0.05).find(|q| q.a.x > slit.x + (F_COLL - STRIP_TO_F1) - 100.0) {
                    dirs.push(((q.b - q.a).normalize(), q.a));
                }
            }
            if let Some((d0, _)) = dirs.first().copied() {
                let dev: Vec<String> = dirs.iter().map(|(v, a)| format!("{:+.3} mrad @ ({:.0},{:.0},{:.0})", (v.cross(d0)).length() * 1e3 * (v - d0).dot(Vec3::Y).signum(), a.x, a.y, a.z)).collect();
                eprintln!("COLL h {h:+}: {}", dev.join(", "));
            }
        }
    }
    if std::env::var_os("SPEC_PATH").is_some() {
        let pick = slit + pickoff_offset();
        let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(pick - Vec3::X * 20.0, Vec3::X).unwrap(), lambda_um: 0.53, flux: 1.0, source: None, pol: None };
        for q in oc_trace::trace3(&sc, &[lr], &cfg).segments.iter().filter(|q| q.flux_start > 0.05).take(6) {
            eprintln!("PICKPATH {:.1?} -> {:.1?} flux {:.3}", q.a, q.b, q.flux_start);
        }
        for (tilt, name) in [(Vec3::Z, "across"), (Vec3::Y, "up")] {
            let (_, chief) = sun_ray(slit, d.slit_turn, 0.0);
            let e = (tilt - chief * tilt.dot(chief)).normalize();
            let dir = (chief + e * (0.8 * 0.5 * BEAM / F_COLL)).normalize();
            let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(slit - dir * 20.0, dir).unwrap(), lambda_um: 0.53, flux: 1.0, source: None, pol: None };
            for q in oc_trace::trace3(&sc, &[lr], &cfg).segments.iter().filter(|q| q.flux_start > 0.05) {
                eprintln!("CONEPATH {name}: {:.1?} -> {:.1?} flux {:.3}", q.a, q.b, q.flux_start);
            }
        }
        for h in [0.0, 0.45] {
            let (from, dir) = sun_ray(slit, d.slit_turn, h * d.slit_height);
            let lr = oc_trace::LaunchRay3 { ray: oc_core::vec3::Ray3::new(from, dir).unwrap(), lambda_um: 0.53, flux: 1.0, source: None, pol: None };
            let res = oc_trace::trace3(&sc, &[lr], &cfg);
            for q in res.segments.iter().filter(|q| q.flux_start > 0.05) {
                eprintln!("PATH h {h}: {:.1?} -> {:.1?} flux {:.3}", q.a, q.b, q.flux_start);
            }
        }
        for o in sc.surfaces.iter() {
            use oc_core::geom3::Intersect3;
            let bb = o.geometry.aabb_world(&o.common.transform);
            eprintln!("OBJ {:.1?} {} [x {:.0}..{:.0}, y {:.0}..{:.0}, z {:.0}..{:.0}]", o.common.transform.pos, o.common.label, bb.min.x, bb.max.x, bb.min.y, bb.max.y, bb.min.z, bb.max.z);
        }
    }
    for l in [BLUE_UM, 0.45, 0.53, 0.61, RED_UM] {
        for h in [-0.45, 0.0, 0.45] {
            let (from, dir) = sun_ray(slit, d.slit_turn, h * d.slit_height);
            let at = land(&sc, probe, from, dir, l).map(|q| {
                let r = q - c;
                r - axis * r.dot(axis)
            });
            let n = 400;
            let target = slit_point(slit, d.slit_turn, h * d.slit_height, 0.0);
            let r_disc = (target - c1).length() * 0.5 * BEAM / F_COLL;
            let e1 = dir.cross(Vec3::Y).normalize();
            let e2 = e1.cross(dir);
            let rays: Vec<oc_trace::LaunchRay3> = (0..n)
                .filter_map(|i| {
                    let (a, r) = (i as f64 * 2.399_963, r_disc * ((i as f64 + 0.5) / n as f64).sqrt());
                    let start = c1 + dir * 15.0 + e1 * (r * a.cos()) + e2 * (r * a.sin());
                    Some(oc_trace::LaunchRay3 {
                        ray: oc_core::vec3::Ray3::new(start, (target - start).normalize()).ok()?,
                        lambda_um: l + 1e-9 * i as f64,
                        flux: 1.0,
                        source: None,
                        pol: None,
                    })
                })
                .collect();
            let res = oc_trace::trace3(&sc, &rays, &cfg);
            if std::env::var_os("SPEC_LOSS").is_some() {
                // Each ray's fate from its own events (its wavelength tag):
                // the table, or else the object its first-order branch
                // stopped on.
                use oc_trace::TraceEventKind;
                let cfg_e = oc_trace::TraceConfig3 { ghosts: false, record_events: true, ..cfg };
                let ev = oc_trace::trace3(&sc, &rays, &cfg_e).events;
                let mut names: std::collections::HashMap<u64, String> = Default::default();
                for o in &sc.surfaces {
                    names.insert(o.common.id.0, o.common.label.clone());
                }
                for o in &sc.bodies {
                    names.insert(o.common.id.0, o.common.label.clone());
                }
                for o in &sc.apertures {
                    names.insert(o.common.id.0, o.common.label.clone());
                }
                let mut fate: std::collections::HashMap<i64, Vec<String>> = Default::default();
                let mut chain: std::collections::HashMap<i64, Vec<String>> = Default::default();
                for e in &ev {
                    let k = ((e.lambda_um - l) / 1e-9).round() as i64;
                    if let TraceEventKind::Interact { surface, .. } = &e.kind {
                        chain.entry(k).or_default().push(format!("{} @ ({:.0}, {:.0}, {:.0})", names.get(&surface.object.0).map_or("?", |q| q.split(',').next().unwrap_or("")), e.point.x, e.point.y, e.point.z));
                    }
                    let what = match &e.kind {
                        TraceEventKind::Interact { surface, children, .. } if children.is_empty() => names.get(&surface.object.0).cloned().unwrap_or_default(),
                        TraceEventKind::DetectorHit { detector, .. } if detector.0 == p.table.0 => "TABLE".into(),
                        TraceEventKind::ApertureBlocked { aperture } => names.get(&aperture.0).cloned().unwrap_or_default(),
                        TraceEventKind::Escaped => format!("escaped near ({:.0}, {:.0}, {:.0})", (e.point.x / 50.0).round() * 50.0, (e.point.y / 50.0).round() * 50.0, (e.point.z / 50.0).round() * 50.0),
                        _ => continue,
                    };
                    fate.entry(k).or_default().push(what);
                }
                let mut tally: std::collections::HashMap<String, usize> = Default::default();
                let who = std::env::var("SPEC_LOSS_WHO").ok();
                let mut shown = 0;
                for (k, v) in &fate {
                    let f = if v.iter().any(|q| q == "TABLE") {
                        "TABLE".to_string()
                    } else {
                        v.iter().find(|q| !q.starts_with("Zero-order")).cloned().unwrap_or_else(|| v[0].clone())
                    };
                    if let Some(w) = &who {
                        if f.contains(w.as_str()) && shown < 3 && h == 0.0 {
                            shown += 1;
                            let r = &rays[*k as usize];
                            eprintln!("WHO {:.0} nm ray {k} from {:.1?} dir {:.4?}: {}", l * 1e3, r.ray.origin, r.ray.dir, chain.get(k).map_or(String::new(), |c| c.join(" -> ")));
                        }
                    }
                    *tally.entry(f).or_default() += 1;
                }
                let mut t: Vec<_> = tally.into_iter().collect();
                t.sort_by_key(|q| std::cmp::Reverse(q.1));
                let line: Vec<String> = t.iter().take(14).map(|(w, k)| format!("{k} {w}")).collect();
                let to_table: f64 = res.detectors.iter().find(|q| q.detector.0 == p.table.0).map_or(0.0, |q| q.records.iter().map(|r| r.incident_flux).sum());
                eprintln!("LOSS {:.0} nm h {h:+}: {:.3} of the launched flux on the table; {}", l * 1e3, to_table / rays.len() as f64, line.join(" | "));
            }
            if std::env::var_os("SPEC_ENDS").is_some() && h == 0.0 {
                // Where each ray's strong path ends, 20 mm cells.
                let mut last: std::collections::HashMap<i64, (f64, Vec3)> = Default::default();
                for q in res.segments.iter().filter(|q| q.flux_start > 0.05) {
                    let k = ((q.lambda_um - l) / 1e-9).round() as i64;
                    let e = last.entry(k).or_insert((f64::NEG_INFINITY, q.b));
                    if q.opl1_mm > e.0 {
                        *e = (q.opl1_mm, q.b);
                    }
                }
                let mut ends: std::collections::BTreeMap<(i64, i64, i64), usize> = Default::default();
                for (_, b) in last.values() {
                    *ends.entry(((b.x / 20.0).round() as i64 * 20, (b.y / 20.0).round() as i64 * 20, (b.z / 20.0).round() as i64 * 20)).or_default() += 1;
                }
                let mut v: Vec<_> = ends.into_iter().collect();
                v.sort_by_key(|q| std::cmp::Reverse(q.1));
                for ((x, y, z), k) in v.iter().take(12) {
                    eprintln!("ENDS {:.0} nm: {k} rays end near ({x}, {y}, {z})", l * 1e3);
                }
            }
            if std::env::var_os("SPEC_ENDS").is_some() && h == 0.0 {
                // The beam's footprint in the diaphragm's plane.
                let stop_r = ProjLens::new().stop_radius;
                let mut rs: Vec<(f64, f64)> = Vec::new();
                let mut seen = std::collections::HashSet::new();
                if let Some(pr) = res.detectors.iter().find(|q| q.detector.0 == probe.0) {
                    for q in &pr.records {
                        let k = ((q.lambda_um - l) / 1e-9).round() as i64;
                        if q.incident_flux > 0.05 && seen.insert(k) {
                            let r = q.point_world - c;
                            let r = r - axis * r.dot(axis);
                            rs.push((r.x, r.y));
                        }
                    }
                }
                let n = rs.len() as f64;
                let (mx, my) = (rs.iter().map(|q| q.0).sum::<f64>() / n, rs.iter().map(|q| q.1).sum::<f64>() / n);
                let (wx, wy) = (rs.iter().map(|q| q.0).fold(0.0f64, |a, b| a.max((b - mx).abs())), rs.iter().map(|q| q.1).fold(0.0f64, |a, b| a.max((b - my).abs())));
                let outside = rs.iter().filter(|q| q.0.hypot(q.1) > stop_r).count();
                eprintln!("STOP {:.0} nm: {} rays at the diaphragm, centred at ({mx:+.1}, {my:+.1}), reaching +-{wx:.1} across and +-{wy:.1} up; {outside} outside its radius {stop_r:.1}", l * 1e3, rs.len());
            }
            let recs = res.detectors.iter().find(|q| q.detector.0 == p.table.0).map(|q| q.records.clone()).unwrap_or_default();
            let mut best: std::collections::HashMap<i64, (f64, Vec3)> = Default::default();
            for q in recs {
                let k = ((q.lambda_um - l) / 1e-9).round() as i64;
                let e = best.entry(k).or_insert((0.0, q.point_world));
                if q.incident_flux > e.0 {
                    *e = (q.incident_flux, q.point_world);
                }
            }
            let mut fl: Vec<f64> = best.values().map(|q| q.0).collect();
            fl.sort_by(f64::total_cmp);
            let med = fl.get(fl.len() / 2).copied().unwrap_or(0.0);
            let pts: Vec<Vec3> = best.values().filter(|q| q.0 > 0.5 * med).map(|q| q.1).collect();
            if pts.len() < 8 {
                eprintln!("SPOT {:.0} nm h {h:+}: {} rays reach the table", l * 1e3, pts.len());
                continue;
            }
            let mut xs: Vec<f64> = pts.iter().map(|q| q.x).collect();
            let mut zs: Vec<f64> = pts.iter().map(|q| q.z).collect();
            xs.sort_by(f64::total_cmp);
            zs.sort_by(f64::total_cmp);
            let k = xs.len();
            let (mx, mz) = (xs[k / 2], zs[k / 2]);
            let within = pts.iter().filter(|q| (q.x - mx).abs() < 3.0 && (q.z - mz).abs() < 3.0).count();
            eprintln!(
                "SPOT {:.0} nm h {h:+}: at x {:+.1} v {:+.1}; half the rays within x +-{:.2} v +-{:.2} mm; {:.0}% within 3 mm; {:.0}% reach; chief {:.1} mm off the diaphragm's centre",
                l * 1e3,
                mx - p.table_centre.x,
                p.table_centre.z - mz,
                0.5 * (xs[3 * k / 4] - xs[k / 4]),
                0.5 * (zs[3 * k / 4] - zs[k / 4]),
                100.0 * within as f64 / n as f64,
                100.0 * k as f64 / n as f64,
                at.map_or(f64::NAN, |v| v.length()),
            );
        }
    }
}
