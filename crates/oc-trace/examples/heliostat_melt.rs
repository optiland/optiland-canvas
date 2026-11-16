//! The heliostat light pipe's other experiment: melting metal with the light
//! that leaves the tube's end at 37 m.
//!
//! The light pipe is the spectrograph's, built by `heliostat_lightpipe`: the
//! sun over Jena, the heliostat, the pick-up mirror, the 1 m Fresnel and its
//! N-SF66 eyepiece, the 37 m silver tube with the spectrograph's gap at 32 m.
//! Its exit face now lets the light on to the melting optics:
//!
//! - a flat fold mirror at 45°, `FOLD_GAP` mm past the tube's end, which sends
//!   the beam straight down;
//! - then, by `MELT`:
//!   - `dish` (default): a paraboloidal dish under the sample, facing up,
//!     which brings the light back up to a focus on the sample's underside.
//!     The sample sits on the dish's axis, so it also shades the middle of
//!     the descending beam.
//!   - `lens`: a PMMA Fresnel lens above the sample, plano-hyperbolic, flat
//!     face up, which focuses the descending beam down onto it.
//!
//! For days without sun, a lamp module sits under the dish's middle
//! ([`add_lamp_module`]): a xenon short-arc lamp in an ellipsoidal reflector
//! that focuses through a hole in the dish onto the crucible. `LAMP=xenon`
//! (or `halogen`, for comparison) traces the lamp with the sun away and the
//! hole open; `MODULE=1` traces the sun with the module in place and the hole
//! closed by its lid.
//!
//! The sample's face is a probe that records every ray crossing it; the
//! analysis (`melt_analysis.py`) keeps the rays arriving from the focusing
//! optic's side, refocuses them, and reads the power on the sample, the flux
//! and the spectrum. `MELT_OUT=<file.csv>` writes those records, `EXIT_OUT`
//! the tube's exit beam, `SCENE_OUT` the scene (default `heliostat-melt.ocanvas`).
//!
//! Lengths in mm, the beam along +X in the tube. Up and down are the room's
//! ([`up`]): with `TABLE_UP` (the round 4 state) the model is the room upside
//! down, world -Y up, as the spectrograph is.
//!
//! `cargo run --release -p oc-trace --features trace3 --example heliostat_melt`

#[path = "heliostat_lightpipe.rs"]
#[allow(dead_code, unused_imports)]
mod lightpipe;

use lightpipe::{env_f64, realism, spectrograph as sp};
use oc_core::fresnel_lens::{FresnelFacing, FresnelGrooves, FresnelLens1};
use oc_core::id::{LayerId, ObjectId, SourceId};
use oc_core::material::{Glass, Material};
use oc_core::scene::Spectrum;
use oc_core::scene3::{
    AngularDistribution3, DetectorTermination, ObjectCommon, PlanarShape2, RadialAperture, SagModel, SamplingSequence, Scene3, Source3,
    SourceSampling, SpatialDistribution3, SurfaceGeometry3, SurfaceObject3, SurfaceResponse,
};
use oc_core::vec3::{Isometry3, Vec3};
use oc_trace::{bounds_for_scene3, depth_for_scene3, trace_scene3, TraceConfig3};

/// The tube's axis height (the pick-up mirror's centre, as the light pipe has it).
const AXIS: f64 = sp::AXIS_Y;

/// Up in the room, as the spectrograph has it: world +Y, or -Y with
/// `TABLE_UP` (the round 4 state), where the model is the room upside down.
/// The melting experiment follows, so both hang off the same tube the
/// same way up.
fn up() -> Vec3 {
    sp::table_normal()
}

/// `p` moved to `h` mm above the room's floor, which lies `AXIS - FLOOR_Y`
/// below the tube's axis in the room.
fn room(p: Vec3, h: f64) -> Vec3 {
    Vec3::new(p.x, AXIS, p.z) + up() * (sp::FLOOR_Y + h - AXIS)
}

/// A surface at `at` (local +Z its front), absorbing behind.
fn surface(s: &mut Scene3, at: Isometry3, geometry: SurfaceGeometry3, front: SurfaceResponse, back: SurfaceResponse, label: &str) -> ObjectId {
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

/// PMMA, as the light pipe's.
fn pmma() -> Glass {
    realism::pmma()
}

/// The fold: a flat first-surface mirror at 45° whose centre sits on the
/// tube's axis `gap` mm past its end, turning the beam from +X to straight
/// down. Rectangular, `FOLD_ACROSS` by `FOLD_ALONG` mm (default 440 x 640,
/// sized for a 380 mm bore; the 355 mm bore's beam and its stray light at
/// 45° with 30-40 mm to spare). Enhanced silver on float glass, flat to 1 mrad (`FOLD_SLOPE`);
/// `REAL_MIRRORS=0` makes it the constant `FOLD_R` (0.95).
fn add_fold(s: &mut Scene3, x_end: f64, gap: f64) -> Vec3 {
    let centre = Vec3::new(x_end + gap, AXIS, 0.0);
    let n = (-Vec3::X - up()).normalize();
    let (across, along) = (0.5 * env_f64("FOLD_ACROSS", 440.0), 0.5 * env_f64("FOLD_ALONG", 640.0));
    let id = surface(
        s,
        sp::frame(centre, n, Vec3::Z),
        SurfaceGeometry3::Plane {
            shape: PlanarShape2::Rectangle {
                half_width_mm: along,
                half_height_mm: across,
            },
        },
        realism::mirror(realism::enhanced_silver(), env_f64("FOLD_R", 0.95)),
        SurfaceResponse::Absorber,
        &format!("Fold mirror, flat, 45 deg, {:.0} x {:.0} mm, enhanced silver", 2.0 * across, 2.0 * along),
    );
    realism::slope(s, id.0, env_f64("FOLD_SLOPE", 1.0));
    centre
}

/// The dish: a paraboloid of focal length `f` (`DISH_F`, default 220 mm, a
/// 51 deg rim angle on the 420 mm aperture; 190 to 330 mm all put 99% on a
/// 20 mm crucible, and the shorter keeps more of it when the heliostat
/// tracks badly: 385 W against 373 W at f 254 with a 4 mrad error),
/// `DISH_D` mm across (default 420), vertex `f` below the sample,
/// facing up. Enhanced silver on an electroformed nickel dish, its figure
/// true to 2 mrad RMS (`DISH_SLOPE`; NASA TN D-4072 measured 1.45 radially
/// and 2.32 around on one); `REAL_MIRRORS=0` makes it the constant `DISH_R`
/// (0.92).
fn add_dish(s: &mut Scene3, sample: Vec3, hole: f64) -> f64 {
    let f = env_f64("DISH_F", 220.0);
    let rim = 0.5 * env_f64("DISH_D", 420.0);
    let vertex = sample - up() * f;
    let rim_deg = 2.0 * (rim / (2.0 * f)).atan().to_degrees();
    let id = surface(
        s,
        sp::frame(vertex, up(), Vec3::Z),
        SurfaceGeometry3::SurfaceOfRevolution {
            sag: SagModel::Conic {
                curvature_per_mm: 1.0 / (2.0 * f),
                conic: -1.0,
            },
            radial_aperture: RadialAperture {
                inner_radius_mm: hole,
                outer_radius_mm: rim,
            },
            outline: None,
        },
        realism::mirror(realism::enhanced_silver(), env_f64("DISH_R", 0.92)),
        SurfaceResponse::Absorber,
        &format!("Dish, paraboloid, f {f:.0} mm, {:.0} mm across, rim angle {rim_deg:.0} deg, hole {:.0} mm, enhanced silver", 2.0 * rim, 2.0 * hole),
    );
    realism::slope(s, id.0, env_f64("DISH_SLOPE", 2.0));
    rim_deg
}

/// The lens: a PMMA Fresnel, flat face up toward the descending beam, its
/// grooves down toward the sample, plano-hyperbolic (conic -n^2, which takes
/// a parallel beam to one point at the design wavelength), focal length
/// `LENS_F` (450 mm) at `LENS_NM` (750 nm, the middle of the beam's power),
/// `LENS_D` across (450 mm), grooves 0.5 mm apart, a 3 mm substrate. Its
/// valleys sit `LENS_F` above the sample.
fn add_lens(s: &mut Scene3, sample: Vec3) -> f64 {
    let f = env_f64("LENS_F", 450.0);
    let lam = env_f64("LENS_NM", 750.0) * 1e-3;
    let glass = pmma();
    let n = glass.index(lam);
    let substrate = 3.0;
    let lens = FresnelLens1 {
        radius_mm: f * (n - 1.0),
        conic: -n * n,
        aspheric: [0.0; 8],
        clear_radius_mm: 0.5 * env_f64("LENS_D", 450.0),
        grooves: FresnelGrooves::ConstantPitch { pitch_mm: 0.5 },
        substrate_mm: substrate,
        facing: FresnelFacing::PlusX,
        draft_deg: 0.0,
    };
    let id_glass = s.catalog.put(&glass);
    let axis = -up();
    let pos = sample - axis * (f + 0.5 * substrate);
    let id = s.add_fresnel_lens(pos, 0.0, &lens, Material::Catalog(id_glass)).expect("the melting Fresnel builds");
    if let Some(b) = s.bodies.iter_mut().find(|q| q.common.id.0 == id) {
        b.common.transform = sp::frame_x(pos, axis, Vec3::Z);
        b.common.label = format!("Melting lens, PMMA Fresnel, plano-hyperbolic, f {f:.0} mm at {:.0} nm, {:.0} mm across", lam * 1e3, 2.0 * lens.clear_radius_mm);
    }
    f
}

/// A lamp's emitter: Lambertian cylinders about the lamp's axis, each
/// `(diameter, length, share of the luminous flux)`, a spectrum, and the
/// luminous flux it is scaled to.
struct LampModel {
    name: &'static str,
    zones: Vec<(f64, f64, f64)>,
    spectrum: Vec<(f64, f64)>,
    lumens: f64,
}

/// Planck's spectrum at `t` K every 10 nm from `lo` to `hi` um, scaled so
/// it carries `lumens`; returns the bands (W each) and the total in watts.
fn planck_bands(t: f64, lo: f64, hi: f64, lumens: f64) -> (Vec<(f64, f64)>, f64) {
    let n = ((hi - lo) / 0.01).round() as usize;
    let bands: Vec<(f64, f64)> = (0..=n)
        .map(|i| {
            let um = lo + 0.01 * i as f64;
            (um, um.powi(-5) / ((14_387.77 / (um * t)).exp() - 1.0))
        })
        .collect();
    lumen_scaled(bands, lumens)
}

/// Equal-width bands `(um, relative power)` scaled so they carry `lumens`;
/// the bands (W each) and their total in watts.
fn lumen_scaled(bands: Vec<(f64, f64)>, lumens: f64) -> (Vec<(f64, f64)>, f64) {
    let lm: f64 = bands.iter().map(|b| oc_core::colorimetry::LM_PER_W * oc_core::colorimetry::cie_1931(b.0)[1] * b.1).sum();
    let bands: Vec<(f64, f64)> = bands.iter().map(|b| (b.0, b.1 * lumens / lm)).collect();
    let w = bands.iter().map(|b| b.1).sum();
    (bands, w)
}

impl LampModel {
    /// Osram XBO 1600 W/HSC OFR, a xenon short-arc lamp (1550 W, 60 000 lm,
    /// 5500 cd, 6000 K, mean luminance 70 000 cd/cm2 over a luminous area
    /// 1.0 x 3.2 mm, ozone free, 2000 h; Osram's XBO catalogue). The core
    /// of that size at that luminance carries 2240 cd, 41% of the lamp's
    /// 5500; the other 59% is taken as a dimmer glow filling the 3.8 mm gap,
    /// 3.5 mm across. `XE_MODEL=glow` puts it all in the glow (the
    /// pessimistic bound). Each zone radiates from its side, so its
    /// intensity falls as the sine toward the axis, where the electrodes
    /// stand: Osram's 60 000 lm over its 5500 cd is 10.9 sr, between the
    /// side-lit cylinder's 9.9 and a sphere's 12.6. Spectrum: Newport's
    /// measured 1600 W xenon lamp, 52 lm per watt of radiation, so the lamp
    /// radiates about 1150 W, three quarters of it past 700 nm.
    ///
    /// `XE=1000`: the XBO 1000 W/HSC OFR instead (32 000 lm, 3000 cd,
    /// 60 000 cd/cm2 over 1.1 x 2.8 mm: a core with 62% of the light, the
    /// glow 3.0 mm across in its 3.3 mm gap).
    fn xenon() -> Self {
        let small = std::env::var("XE").is_ok_and(|v| v == "1000");
        let (core, glow, lumens, name) = if small {
            ((1.1, 2.8, 0.62), (3.0, 3.5), 32_000.0, "Xenon short-arc lamp, Osram XBO 1000 W/HSC OFR")
        } else {
            ((1.0, 3.2, 0.41), (3.5, 4.0), 60_000.0, "Xenon short-arc lamp, Osram XBO 1600 W/HSC OFR")
        };
        let zones = if std::env::var("XE_MODEL").is_ok_and(|v| v == "glow") {
            vec![(glow.0, glow.1, 1.0)]
        } else {
            vec![core, (glow.0, glow.1, 1.0 - core.2)]
        };
        // Newport's measured 1600 W xenon spectrum (`REAL_XENON=0`: Planck
        // at 6000 K), scaled to the lamp's luminous flux. Osram's flux is
        // measured outside the bulb, so the bulb's losses are in it; the bulb
        // is centred on the arc and bends its rays by well under a milliradian.
        // `XE_MAX_UM` drops the spectrum past that wavelength: the bound for
        // infrared that comes from the hot anode and bulb, not the arc.
        let cut = env_f64("XE_MAX_UM", 9.9);
        let spectrum = match realism::xenon_spectrum() {
            Some(rows) => lumen_scaled(rows.into_iter().filter(|r| r.0 <= cut).collect(), lumens).0,
            None => planck_bands(6_000.0, 0.30, 2.50, lumens).0,
        };
        Self { name, zones, spectrum, lumens }
    }

    /// A 1000 W, 3200 K tungsten-halogen projection lamp with a compact
    /// coiled-coil filament 6 x 16 mm (Newport/Oriel 6315: 27 500 lm, 300 h),
    /// its coil along the reflector's axis. Tungsten at that colour
    /// ([`realism::tungsten`]), 320 nm to 2.7 um (the quartz cuts the rest).
    fn halogen() -> Self {
        let w = realism::tungsten(3_200.0);
        Self {
            name: "Halogen lamp, 1000 W 3200 K, coil 6 x 16 mm",
            zones: vec![(6.0, 16.0, 1.0)],
            spectrum: lumen_scaled((0..=238).map(|i| 0.32 + 0.01 * i as f64).map(|um| (um, w(um))).collect(), 27_500.0).0,
            lumens: 27_500.0,
        }
    }

    fn watts(&self) -> f64 {
        self.spectrum.iter().map(|b| b.1).sum()
    }
}

/// The lamp module under the dish's middle. The dish has a hole `LAMP_HOLE`
/// mm in radius (90) about its vertex. Under it, an ellipsoidal reflector
/// whose far focus is the crucible and whose near focus, `LAMP_D` mm
/// (390) under the crucible, holds the lamp upright on the axis, its base
/// through the reflector's vertex `LAMP_VTX` mm (45) behind the arc. The
/// reflector is cut where its light just clears the hole's rim, so all of
/// it converges through the hole onto the crucible's underside. Protected
/// aluminium, what a reflector this hot and this deep in the arc's UV is
/// coated with (a cinema cold mirror would throw the infrared away), its
/// figure true to 2 mrad (`LAMP_SLOPE`); `REAL_MIRRORS=0` makes it the
/// constant `LAMP_R` (0.92). A black top plate 30 mm under the hole, its own hole
/// just wide enough for the cone, stops the lamp's direct light from
/// rising beside the dish; the lamp's upper stem (141 mm above the arc, 27
/// mm across, its top 29 mm under the dish's vertex at `LAMP_D` 390) shades
/// the cone's middle.
///
/// In sunlight the hole is closed by a lid that is the dish's own middle,
/// the same paraboloid (`MODULE=1` traces that); it drops into the 30 mm
/// between the dish and the top plate and swings aside. Returns the sources.
fn add_lamp_module(s: &mut Scene3, sample: Vec3, lamp: Option<&LampModel>) -> Vec<SourceId> {
    let f = env_f64("DISH_F", 220.0);
    let hole = env_f64("LAMP_HOLE", 90.0);
    let vertex = sample - up() * f;
    let hole_y = hole * hole / (4.0 * f);
    // The black top plate, 30 mm under the hole's rim, its hole the cone's
    // width there.
    let plate_y = hole_y - 30.0;
    let plate_r = hole * (f - plate_y) / (f - hole_y) + 1.0;
    surface(
        s,
        sp::frame(vertex + up() * plate_y, up(), Vec3::Z),
        SurfaceGeometry3::Plane {
            shape: PlanarShape2::Annulus {
                inner_mm: plate_r,
                outer_mm: 400.0,
            },
        },
        SurfaceResponse::Absorber,
        SurfaceResponse::Absorber,
        &format!("Lamp module top plate, black, {:.0} mm hole", 2.0 * plate_r),
    );
    let Some(lamp) = lamp else {
        // The lid: the dish's middle, closing the hole, 0.5 mm gap all round.
        let id = surface(
            s,
            sp::frame(vertex, up(), Vec3::Z),
            SurfaceGeometry3::SurfaceOfRevolution {
                sag: SagModel::Conic {
                    curvature_per_mm: 1.0 / (2.0 * f),
                    conic: -1.0,
                },
                radial_aperture: RadialAperture {
                    inner_radius_mm: 0.0,
                    outer_radius_mm: hole - 0.5,
                },
                outline: None,
            },
            realism::mirror(realism::enhanced_silver(), env_f64("DISH_R", 0.92)),
            SurfaceResponse::Absorber,
            &format!("Lamp lid, the dish's middle, {:.0} mm across", 2.0 * (hole - 0.5)),
        );
        realism::slope(s, id.0, env_f64("DISH_SLOPE", 2.0));
        return Vec::new();
    };
    // The ellipsoid: foci the arc (f1) and the crucible's underside (f2).
    let d = env_f64("LAMP_D", 390.0);
    let vtx = env_f64("LAMP_VTX", 45.0);
    let (c, a) = (0.5 * d, 0.5 * d + vtx);
    let e = c / a;
    let p = a * (1.0 - e * e);
    let f1 = sample - up() * d;
    // The mouth: walking up the ellipse from the vertex, the last point whose
    // ray to f2 still passes inside the hole's rim.
    let t2 = (hole / (f - hole_y)).atan();
    let mut mouth = (0.0, 0.0);
    for i in (1..20_000).rev() {
        let phi = std::f64::consts::PI * i as f64 / 20_000.0;
        let r1 = p / (1.0 - e * phi.cos());
        let (x, y) = (r1 * phi.sin(), r1 * phi.cos());
        if x.atan2(d - y) > t2 {
            break;
        }
        mouth = (x, y);
    }
    let neck = 18.0;
    let reflector = surface(
        s,
        sp::frame(f1 - up() * vtx, up(), Vec3::Z),
        SurfaceGeometry3::SurfaceOfRevolution {
            sag: SagModel::Conic {
                curvature_per_mm: a / (a * a - c * c),
                conic: -e * e,
            },
            radial_aperture: RadialAperture {
                inner_radius_mm: neck,
                outer_radius_mm: mouth.0,
            },
            outline: None,
        },
        realism::mirror(realism::protected_aluminium(), env_f64("LAMP_R", 0.92)),
        SurfaceResponse::Absorber,
        &format!("Lamp reflector, ellipsoid, foci the arc and the crucible {d:.0} mm apart, {:.0} mm across, protected aluminium", 2.0 * mouth.0),
    );
    realism::slope(s, reflector.0, env_f64("LAMP_SLOPE", 2.0));
    // The upper stem and base, seen from the crucible: a disc at its top.
    surface(
        s,
        sp::frame(f1 + up() * 141.0, -up(), Vec3::Z),
        SurfaceGeometry3::Plane {
            shape: PlanarShape2::Disc { radius_mm: 13.5 },
        },
        SurfaceResponse::Absorber,
        SurfaceResponse::Absorber,
        "Lamp's upper stem and base (shade)",
    );
    eprintln!(
        "lamp module: hole r {hole:.0} mm (cone {:.1} deg), reflector mouth r {:.0} mm at {:.0} mm under the crucible, arc {d:.0} mm under it",
        t2.to_degrees(),
        mouth.0,
        d - mouth.1
    );
    // The emitter: each zone a 12-sided prism of Lambertian faces about the
    // lamp's axis, carrying its share of the lamp's radiant power.
    const FACES: usize = 12;
    let watts = lamp.watts();
    eprintln!("lamp: {}, {:.0} lm, {watts:.0} W of radiation", lamp.name, lamp.lumens);
    let per = env_f64("LAMP_RAYS", 4096.0) as u32;
    let mut ids = Vec::new();
    for &(dia, len, share) in &lamp.zones {
        let chord = dia * (std::f64::consts::PI / FACES as f64).sin();
        let apothem = 0.5 * dia * (std::f64::consts::PI / FACES as f64).cos();
        for i in 0..FACES {
            let ang = std::f64::consts::TAU * (i as f64 + 0.5) / FACES as f64;
            let nrm = Vec3::X * ang.cos() + Vec3::Z * ang.sin();
            let sid = s.alloc_source_id();
            s.sources.push(Source3 {
                window: Default::default(),
                id: sid,
                transform: sp::frame_x(f1 + nrm * apothem, nrm, Vec3::Y),
                spatial: SpatialDistribution3::Rectangle {
                    half_width_mm: 0.5 * len,
                    half_height_mm: 0.5 * chord,
                },
                angular: AngularDistribution3::LambertianHemisphere,
                spectrum: Spectrum::Continuum(lamp.spectrum.clone()),
                flux: watts * share / FACES as f64,
                samples: SourceSampling {
                    spatial_samples: per,
                    angular_samples: 4,
                    seed: 11 + sid.0,
                    sequence: SamplingSequence::Sobol,
                },
            });
            ids.push(sid);
        }
    }
    ids
}

fn main() {
    let built = lightpipe::build();
    let mut s = built.scene;
    let exit = built.detectors.iter().find(|(n, _)| *n == "tube exit 37 m").expect("the exit face").1;
    let x_end = {
        let d = s.detectors.iter_mut().find(|d| d.common.id.0 == exit.0).unwrap();
        d.termination = DetectorTermination::ProbeAndContinue;
        d.common.label = "Tube exit, 1 mm past its end (probe)".into();
        d.common.transform.pos.x - 1.0
    };
    let concept = std::env::var("MELT").unwrap_or_else(|_| "dish".into());
    // The sample's underside or top, `SAMPLE_H` mm above the floor (1000).
    let sample = Vec3::new(x_end + env_f64("FOLD_GAP", 350.0), 0.0, 0.0);
    let sample = room(sample, env_f64("SAMPLE_H", 1_000.0));
    let fold = add_fold(&mut s, x_end, env_f64("FOLD_GAP", 350.0));
    // A probe across the descending column, 300 mm under the fold.
    let column = sp::detector(&mut s, fold - up() * 300.0, up(), Vec3::X, 260.0, 260.0, [1, 1], DetectorTermination::ProbeAndContinue, "Column, 300 mm under the fold (probe)");
    // The crucible, 20 mm across (`SAMPLE_R` its radius).
    let r_sample = env_f64("SAMPLE_R", 10.0);
    // `LAMP=xenon` or `LAMP=halogen`: the sun is away and the lamp module
    // under the dish lights the crucible (dish concept only).
    let lamp = match std::env::var("LAMP").as_deref() {
        Ok("xenon") => Some(LampModel::xenon()),
        Ok("halogen") => Some(LampModel::halogen()),
        _ => None,
    };
    let mut lamp_sources = Vec::new();
    let (face, from_below, what) = match concept.as_str() {
        "lens" => {
            let f = add_lens(&mut s, sample);
            // The sample's top, facing up into the cone.
            let face = sp::detector(&mut s, sample, up(), Vec3::X, 40.0, 40.0, [1, 1], DetectorTermination::ProbeAndContinue, "Sample plane (probe)");
            (face, false, format!("lens f {f:.0}"))
        }
        _ => {
            let module = lamp.is_some() || std::env::var_os("MODULE").is_some();
            let rim_deg = add_dish(&mut s, sample, if module { env_f64("LAMP_HOLE", 90.0) } else { 0.0 });
            if module {
                lamp_sources = add_lamp_module(&mut s, sample, lamp.as_ref());
            }
            // The sample shades the descending beam: an absorbing disc just
            // above the face that records the rising light.
            surface(
                &mut s,
                sp::frame(sample + up() * 1.0, up(), Vec3::Z),
                SurfaceGeometry3::Plane {
                    shape: PlanarShape2::Disc { radius_mm: r_sample },
                },
                SurfaceResponse::Absorber,
                SurfaceResponse::Absorber,
                &format!("Sample, {:.0} mm across (absorbs)", 2.0 * r_sample),
            );
            let face = sp::detector(&mut s, sample, -up(), Vec3::X, 40.0, 40.0, [1, 1], DetectorTermination::ProbeAndContinue, "Sample plane, underside (probe)");
            // The crucible's holder: a ceramic rod 8 mm across at its
            // mid-height, run out sideways past the beam (`REAL_MOUNTS=0`
            // leaves the crucible floating).
            if realism::on("MOUNTS") {
                let id = s.alloc_object_id();
                let mut common = ObjectCommon::new(id, LayerId(0));
                common.transform = Isometry3::from_pos(sample + up() * 10.0 + Vec3::Z * (r_sample + 160.0));
                common.label = "Crucible holder, ceramic rod, 8 mm across (black)".into();
                let black = oc_core::scene3::TwoSidedSurfaceResponse { from_outside: SurfaceResponse::Absorber, from_inside: SurfaceResponse::Absorber };
                s.bodies.push(oc_core::scene3::Body3 {
                    common,
                    geometry: oc_core::scene3::SolidGeometry3::Cylinder { radius_mm: 4.0, half_length_mm: 160.0, axis_local: Vec3::Z },
                    material: Material::Absorber,
                    absorption: oc_core::scene3::AbsorptionModel::None,
                    surface: oc_core::scene3::SurfaceProps3 { default: black, overrides: Vec::new() },
                    priority: 0,
                });
            }
            (face, true, format!("dish, rim {rim_deg:.0} deg"))
        }
    };

    // The sun's rays: `VERIFY` spatial samples (131072 default), two
    // directions each.
    let n = env_f64("VERIFY", 131_072.0) as u32;
    let src = s.solar.as_ref().unwrap().source;
    if let Some(src) = s.sources.iter_mut().find(|x| x.id == src) {
        src.samples.spatial_samples = n;
    }
    let state = s.resolve_solar().expect("sun resolves");
    // `HELIO_ERR_MRAD="ax az"`: the heliostat off its aim, the mirror turned
    // about its pivot by `ax` mrad about world X and `az` about world Z after
    // it has followed the sun (a tracking error).
    let err: Vec<f64> = std::env::var("HELIO_ERR_MRAD").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    if let [ax, az] = err[..] {
        let hel = s.solar.as_ref().unwrap().heliostats[0].mirror;
        if let Some(m) = s.surfaces.iter_mut().find(|q| q.common.id.0 == hel.0) {
            let t = m.common.transform;
            let r = Isometry3::from_axis_angle(Vec3::X, ax * 1e-3).unwrap().compose(Isometry3::from_axis_angle(Vec3::Z, az * 1e-3).unwrap());
            let mut turned = r.compose(t);
            turned.pos = t.pos;
            m.common.transform = turned;
        }
        // The heliostat's glass turns with it.
        s.resolve_links();
        eprintln!("heliostat turned {ax} mrad about X, {az} mrad about Z off its aim");
    }
    if !lamp_sources.is_empty() {
        s.sources.retain(|x| x.id != src);
        eprintln!("the sun is away: {} lamp emitters", lamp_sources.len());
    }
    eprintln!(
        "sun: az {:.2} deg, el {:.2} deg, beam {:.1} W; heliostat incidence {:.2} deg",
        state.azimuth_deg, state.elevation_deg, state.source_flux_w, state.heliostats[0].incidence_deg
    );
    let (ghosts, max_ghost_bounces) = realism::ghosts();
    let cfg = TraceConfig3 {
        ghosts,
        max_ghost_bounces,
        max_depth: depth_for_scene3(&s, ghosts, max_ghost_bounces),
        bounds_mm: bounds_for_scene3(&s),
        record_detector_hits: true,
        max_records: 20_000_000,
        ..TraceConfig3::default()
    };
    // `FIDELITY=Fast|Standard|Reference`: the coating physics the trace pays for.
    let cfg = cfg.with_fidelity(realism::fidelity());
    let t0 = std::time::Instant::now();
    let res = trace_scene3(&s, &cfg);
    eprintln!("melting system ({concept}: {what}), traced in {:.1?}", t0.elapsed());
    let get = |id: ObjectId| res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
    let frame_of = |id: ObjectId| s.detectors.iter().find(|q| q.common.id.0 == id.0).unwrap().common.transform;
    eprintln!("  tube exit:        {:7.1} W", get(exit).total_incident);
    eprintln!("  under the fold:   {:7.1} W", get(column).total_incident);
    // The sample's face: only light arriving from the focusing optic.
    let d = get(face);
    let fr = frame_of(face);
    let mut csv = String::from("x_mm,y_mm,z_mm,dx,dy,dz,lambda_um,flux_w\n");
    let (mut p_in, mut kept) = (0.0, 0usize);
    for h in &d.records {
        let dir = fr.rotate(h.incident_dir_local);
        if (dir.dot(up()) > 0.0) != from_below {
            continue;
        }
        p_in += h.incident_flux;
        kept += 1;
        let p = h.point_world - sample;
        csv.push_str(&format!("{:.4},{:.4},{:.4},{:.8},{:.8},{:.8},{:.5},{:.6e}\n", p.x, p.y, p.z, dir.x, dir.y, dir.z, h.lambda_um, h.incident_flux));
    }
    let r2 = r_sample * r_sample;
    let on: f64 = d
        .records
        .iter()
        .filter(|h| (fr.rotate(h.incident_dir_local).dot(up()) > 0.0) == from_below && (h.point_world - sample).length_squared() - (h.point_world.y - sample.y).powi(2) <= r2)
        .map(|h| h.incident_flux)
        .sum();
    eprintln!("  at the sample's plane, from the optic: {p_in:7.1} W ({kept} rays); within r {r_sample:.1} mm: {on:.1} W");
    // The crucible is a cup, `CUP_H` mm tall (20), as wide as the disc: light
    // that passes beside its bottom still strikes its side wall on the way
    // up. Nothing else stands there, so each ray's own line decides it.
    // `REAL_CUP=0` counts the bottom alone, the flat disc of before.
    let cup_h = if realism::on("CUP") { env_f64("CUP_H", 20.0) } else { 0.0 };
    let wall: f64 = d
        .records
        .iter()
        .filter(|_| from_below)
        .filter_map(|h| {
            let dir = fr.rotate(h.incident_dir_local);
            let rise = dir.dot(up());
            let p = h.point_world - sample;
            let (q, v) = (p - up() * p.dot(up()), dir - up() * rise);
            if rise <= 0.0 || q.length_squared() <= r2 {
                return None;
            }
            // |q + v t| = r for the first t > 0, and its height rise * t.
            let (a, b, c) = (v.length_squared(), 2.0 * q.dot(v), q.length_squared() - r2);
            let disc = b * b - 4.0 * a * c;
            if a <= 0.0 || disc < 0.0 {
                return None;
            }
            let t = (-b - disc.sqrt()) / (2.0 * a);
            (t > 0.0 && rise * t <= cup_h).then_some(h.incident_flux)
        })
        .sum();
    eprintln!("  on the crucible, a cup {cup_h:.0} mm tall: {:.1} W ({on:.1} on its bottom, {wall:.1} on its side)", on + wall);
    if d.records.len() >= cfg.max_records {
        eprintln!("  warning: the record limit was reached");
    }
    if let Ok(path) = std::env::var("MELT_OUT") {
        std::fs::write(&path, csv).expect("write the sample's rays");
        eprintln!("wrote {path} (positions from the sample's centre)");
    }
    let out = std::env::var("SCENE_OUT").unwrap_or_else(|_| "heliostat-melt.ocanvas".into());
    let mut file = s.clone();
    if let Some(src) = file.sources.iter_mut().find(|x| x.id == src) {
        src.samples.spatial_samples = 1024;
    }
    std::fs::write(&out, file.to_json()).expect("write the scene");
    eprintln!("wrote {out}");
}
