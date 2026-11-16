//! Build the heliostat light-pipe demonstrator, trace it with the f64
//! reference, and write it as an `.ocanvas` v2 file the canvas opens.
//!
//! Sun (Jena, the Deutsches Optisches Museum, March equinox, solar noon) →
//! tracking heliostat on the ground →
//! fixed 45° pick-up mirror 5 m above it → 1 m further, a 1 m PMMA Fresnel
//! lens (flat to the sun, grooves away, F/3) → a dense-flint (N-SF66)
//! negative lens where the converging beam is 355 mm across, which with the
//! Fresnel makes a Galilean afocal of magnification 2.97 → a 37 m
//! silver-lined tube, 355 mm bore, 360 mm outside, run through a 360 mm hole
//! in the wall, its mouth flush with the wall.
//!
//! Two experiments share the light. At 32 m a projection spectrograph takes
//! a 0.75 mm slit of it through a gap in the tube and throws the solar
//! spectrum onto a table (docs/spectrograph.md); a metal-melting experiment
//! at 37 m takes everything that leaves the tube's end, caught by a detector
//! 1 mm past it. The sun's spectrum is G173 with the TSIS-1 solar lines
//! inside 380-720 nm (examples/heliostat-lightpipe/site_spectrum.py).
//!
//! After writing the scene this traces it (`VERIFY=<spatial samples>` sets
//! how many, 131072 is 262 thousand rays), takes the tube's gain at 32 m from
//! it, and writes and traces the spectrograph's bench scene (`BENCH_OUT`, and
//! `BENCH=<spatial samples>` for its trace): where each line lands, how much
//! light reaches the table, and the sodium D lines' profile, written to
//! `PROFILE_OUT` (default `sodium-d-profile.csv`).
//!
//! The eyepiece's glass is what corrects the Fresnel's colour. In a Galilean
//! of magnification `M` the eyepiece cancels the objective's primary colour
//! when its Abbe number is the objective's over `M`: 53 / 2.97 ≈ 18 here, and
//! N-SF66 (20.9) is the nearest real glass. Across 0.45 to 2.3 µm the exit
//! angle then stays within 1.4 mrad, against 7.4 mrad with an N-BK7 eyepiece
//! and ±13.9 mrad from the sun's own disc.
//!
//! The optical numbers come from examples/heliostat-lightpipe/design.py: each
//! Fresnel zone aims at one focus 3 m past the valleys at 0.6 µm, fitted as a
//! paraboloid plus A4..A18; the eyepiece face is the Cartesian surface (a
//! hyperboloid) that turns the cone into a parallel beam.
//!
//! `DIAG=1` adds probes along the path and prints ray angles instead of
//! writing the file; `POINT_SUN=1` shrinks the sun to a point, which
//! separates the optics' own errors from the sun's size.
//!
//! `cargo run --release -p oc-trace --features trace3 --example heliostat_lightpipe -- out.ocanvas`

use oc_core::build3::{BodyFace, SingletGeom};
use oc_core::fresnel_lens::{FresnelFacing, FresnelGrooves, FresnelLens1};
use oc_core::heliostat::{HelioTarget, LocalTime, Site};
use oc_core::id::ObjectId;
use oc_core::material::{Glass, Material, Sellmeier};
use oc_core::link::LinkMode;
use oc_core::scene3::{DetectorGeometry3, DetectorTermination, PlanarShape2, Scene3};
use oc_core::thinfilm::{CoatingDesign, DesignKind};
use oc_core::tube::{BoreFinish, BoreMetal, TubeDims};
use oc_core::vec3::{Isometry3, Vec3};
use oc_trace::{bounds_for_scene3, depth_for_scene3, trace_scene3, TraceConfig3};

/// Height of the beam line, the pick-up mirror's centre.
const Y: f64 = 5_000.0;
const SUBSTRATE: f64 = 3.0;
/// Fresnel valleys to the eyepiece vertex: where the whole beam, sun disc
/// included, is 355 mm across, the tube's bore.
const TO_EYEPIECE: f64 = 1_990.781_703_327_229_3;
const EYE_T: f64 = 10.0;
const EYE_SEMI: f64 = 190.0;
/// Eyepiece back face to the wall, where the tube's mouth is.
const TO_HOLE: f64 = 20.0;
const TUBE_LEN: f64 = 37_000.0;
const BORE_RADIUS: f64 = spectrograph::BORE_RADIUS;
/// Where along the tube the spectrograph takes its slit of light.
const SPEC_AT: f64 = 32_000.0;
/// The sun's spectrum: G173 with the TSIS-1 solar lines inside 380-720 nm
/// (examples/heliostat-lightpipe/site_spectrum.py).
const SITE_SPECTRUM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/heliostat-lightpipe/jena-direct-g173-hsrs.csv"
);

#[path = "shared/spectrograph.rs"]
pub(crate) mod spectrograph;
#[path = "shared/realism.rs"]
pub(crate) mod realism;

/// The site spectrum's rows, `[nm, W/m²/nm]`.
fn site_rows() -> Vec<[f64; 2]> {
    let text = std::fs::read_to_string(SITE_SPECTRUM).expect("the site spectrum (run site_spectrum.py)");
    oc_core::solar::parse_spectrum_table(&text).expect("a spectrum table").rows
}

fn fresnel() -> FresnelLens1 {
    FresnelLens1 {
        radius_mm: 1_470.147_217_069_780_3,
        conic: -1.0,
        aspheric: [
            -2.872_550_532_174_384_5e-11,
            3.766_883_695_772_587_6e-18,
            -5.667_889_508_372_192e-25,
            9.138_076_524_559_708e-32,
            -1.511_533_763_305_906_2e-38,
            1.721_125_494_621_016e-45,
            1.326_265_720_908_974_4e-51,
            -1.365_679_261_586_851_6e-57,
        ],
        clear_radius_mm: 500.0,
        grooves: FresnelGrooves::ConstantPitch { pitch_mm: 1.0 },
        substrate_mm: SUBSTRATE,
        facing: FresnelFacing::PlusX,
        // A moulded lens's risers lean a few degrees off the axis so the
        // mould releases it; 3 degrees here (`REAL_DRAFT=0`: vertical).
        draft_deg: env_f64("DRAFT", if realism::on("DRAFT") { 3.0 } else { 0.0 }),
    }
}

pub(crate) fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// PMMA: Beadie et al. 2015 ([`realism::pmma`]). N-SF66: Schott's catalog
/// Sellmeier.
fn glasses() -> (Glass, Glass) {
    let pmma = realism::pmma();
    let sf66 = Glass::sellmeier(
        "N-SF66",
        Sellmeier {
            b: [2.024_597_6, 0.470_187_196, 2.599_704_33],
            c: [0.014_705_322_5, 0.069_299_827_6, 161.817_601],
        },
    );
    (pmma, sf66)
}

pub(crate) struct Built {
    pub(crate) slit: Vec3,
    pub(crate) scene: Scene3,
    pub(crate) detectors: Vec<(&'static str, ObjectId)>,
}

pub(crate) fn build() -> Built {
    let mut s = Scene3::new();
    let (pmma, sf66) = glasses();
    let pmma_id = s.catalog.put(&pmma);
    let sf66_id = s.catalog.put(&sf66);

    // Jena, the Deutsches Optisches Museum at Carl-Zeiss-Platz 12
    // (50°55'40.1" N, 11°34'45.4" E, about 160 m up), on the March equinox at
    // solar noon, 12:21 CET: the sun about 39° up in the south. +X points
    // north, so the sun stands up and to the left of the heliostat. The
    // pressure is the standard atmosphere's at 160 m; the temperature only
    // moves the refraction, by arcseconds this high.
    let site = Site {
        latitude_deg: 50.92781,
        longitude_deg: 11.57928,
        elevation_m: 160.0,
        pressure_mbar: 1013.25 * (-160.0_f64 / 8434.0).exp(),
        temperature_c: 10.0,
    };
    // `SUN_DATE="month day hour minute"` (local standard time, CET) moves
    // the sun: the heliostat follows it, and only the geometry changes, the
    // spectrum stays G173's.
    let date: Vec<u32> = std::env::var("SUN_DATE").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let (month, day, hour, minute) = if let [mo, d, h, mi] = date[..] { (mo, d, h, mi) } else { (3, 20, 12, 21) };
    let time = LocalTime {
        year: 2026,
        month,
        day,
        hour,
        minute,
        second: 0.0,
        utc_offset_h: 1.0,
    };
    s.add_sun(site, time, 0.0);
    // The sun's spectrum, resolved enough to hold the solar lines the
    // spectrograph shows; its total is the DNI.
    let rows = site_rows();
    let total = oc_core::solar::table_power(&rows, rows[0][0] / 1e3, rows[rows.len() - 1][0] / 1e3);
    if let Some(rig) = s.solar.as_mut() {
        rig.spectrum = Some(oc_core::heliostat::SunSpectrum {
            name: "jena-direct-g173-hsrs.csv".into(),
            rows,
        });
        rig.dni_w_m2 = total;
    }

    // Pick-up mirror: fixed, 45°, sends the rising beam along +X.
    let pickup = s
        .add_planar_mirror(
            Vec3::new(-640.0, Y - 640.0, 0.0),
            Vec3::new(640.0, Y + 640.0, 0.0),
            0.95,
        )
        .expect("pick-up mirror");
    label(&mut s, pickup, "Pick-up mirror");
    // Silvered 4 mm low-iron glass, as a large flat is made, flat to 1 mrad.
    realism::behind_glass(&mut s, pickup, -1.0, 4.0);
    realism::slope(&mut s, pickup, env_f64("PICKUP_SLOPE", 1.0));

    // Heliostat on the ground, aimed at the pick-up mirror.
    let heliostat = s.add_heliostat(
        Vec3::ZERO,
        1_150.0,
        1_150.0,
        0.94,
        HelioTarget::Object(ObjectId(pickup)),
    )
    .expect("heliostat");
    // A solar mirror: silver behind 4 mm of low-iron glass, its figure true
    // to 1.2 mrad RMS, what Sandia measured on a glass-faceted heliostat
    // (Mitchell & Zhu, Solar Energy 2020).
    realism::behind_glass(&mut s, heliostat.0, 1.0, 4.0);
    realism::slope(&mut s, heliostat.0, env_f64("HELIO_SLOPE", 1.2));

    // Stops that take out the light the optics would not collect, each 5 mm
    // in front of its mirror and locked to it, so the heliostat's turns as it
    // tracks. Each opens 10 mm past its mirror's edge: a ray crosses the
    // stop's plane up to gap × tan(incidence) from where it meets the mirror,
    // 9 mm at 60°, so no ray the mirror catches is clipped. The sun's beam is a disc 1.05 times the heliostat's half
    // diagonal, 854 mm, so sunlight passing beside the heliostat, and
    // heliostat light passing beside the pick-up mirror, would otherwise run
    // on through the scene.
    stop_on(
        &mut s,
        heliostat.0,
        5.0,
        PlanarShape2::Rectangle { half_width_mm: 585.0, half_height_mm: 585.0 },
        PlanarShape2::Rectangle { half_width_mm: 1_100.0, half_height_mm: 1_100.0 },
        "Heliostat stop",
    );
    // The pick-up mirror reflects on both faces; the rising beam meets the
    // one facing down, its local -Z.
    stop_on(
        &mut s,
        pickup,
        -5.0,
        PlanarShape2::Disc { radius_mm: 640.0 * std::f64::consts::SQRT_2 + 10.0 },
        PlanarShape2::Disc { radius_mm: 1_400.0 },
        "Pick-up mirror stop",
    );

    // The Fresnel lens, 1 m after the pick-up mirror.
    let x_fresnel = 1_000.0;
    let f = s
        .add_fresnel_lens(
            Vec3::new(x_fresnel, Y, 0.0),
            0.0,
            &fresnel(),
            Material::Catalog(pmma_id),
        )
        .expect("Fresnel");
    label(&mut s, f, "Fresnel, PMMA F/3 (grooves away from the sun)");
    realism::absorb(&mut s, f, "pmma");
    // Both faces, grooves and risers included, carry an AR coat: bare, the
    // lens loses about 9% of the sun's light, 4% at each face. A quarter
    // wave of MgF2 at 550 nm, whose index lies between air's and PMMA's,
    // reflects less than bare PMMA at every wavelength, so the melting
    // experiment's infrared keeps its light too; the three-layer broadband
    // AR the field Fresnels carry gains 8% over 500-700 nm but loses 24% at
    // 370 nm and 35-40% beyond 1.1 µm (`FRESNEL_AR=2` for it).
    // `FRESNEL_AR=0` leaves the lens bare.
    let fresnel_ar = env_f64("FRESNEL_AR", 1.0);
    if fresnel_ar > 0.5 {
        let broadband = fresnel_ar > 1.5;
        let ar = CoatingDesign {
            kind: if broadband { DesignKind::BroadbandArLegacy } else { DesignKind::SingleLayerAr },
            lambda_nm: 550.0,
            aoi_deg: 0.0,
            periods: 1,
            n_high: 2.35,
            n_low: 1.38,
            adapt: None,
        };
        let film = oc_core::thinfilm::film_coating(ar, 1.0, pmma.index(0.55));
        if let Some(body) = s.bodies.iter_mut().find(|q| q.common.id.0 == f) {
            body.surface.default = oc_core::scene3::TwoSidedSurfaceResponse {
                from_inside: oc_core::scene3::SurfaceResponse::Coating { coating: film.seen_from_behind() },
                from_outside: oc_core::scene3::SurfaceResponse::Coating { coating: film },
            };
            body.common.label.push_str(if broadband { ", broadband AR both faces" } else { ", single-layer AR both faces" });
        }
    }
    let valleys = x_fresnel + 0.5 * SUBSTRATE;

    // Eyepiece: plano-concave N-SF66, the hyperbolic face toward the Fresnel.
    // `EYE_SHIFT` moves the eyepiece along the axis from where the design
    // put it, which is how the telescope is focused when it is assembled.
    let vertex = valleys + TO_EYEPIECE + env_f64("EYE_SHIFT", 0.0);
    let eye = s.add_singlet(
        Vec3::new(vertex + 0.5 * EYE_T, Y, 0.0),
        SingletGeom {
            r1_mm: -928.717_076_232_186_2,
            r2_mm: f64::INFINITY,
            thickness_mm: EYE_T,
            semi_aperture_mm: EYE_SEMI,
            conic1: -3.687_298_942_825_485,
            conic2: 0.0,
        },
        sf66.nd,
        sf66.vd,
    );
    if let Some(b) = s.bodies.iter_mut().find(|b| b.common.id.0 == eye.0) {
        b.material = Material::Catalog(sf66_id);
        b.common.label = "Eyepiece, N-SF66 hyperbolic".into();
    }
    // A dense flint browns in the violet: 10 mm passes half the light at
    // 400 nm, and the eyepiece is 10 to 29 mm thick.
    realism::absorb(&mut s, eye.0, "N-SF66");
    // A quarter wave of MgF2 on both faces. Bare N-SF66 (n 1.92) reflects
    // about 10% at each, the largest loss in the system, and MgF2 (1.38) is
    // almost exactly the ideal single layer for it, sqrt(1.92) = 1.386. The
    // three-layer broadband AR assumes a crown near 1.5 and does worse here.
    let ar_nm = env_f64("AR_NM", 650.0);
    if ar_nm > 0.0 {
        for face in [BodyFace::Front, BodyFace::Back] {
            let ar = CoatingDesign {
                kind: DesignKind::SingleLayerAr,
                lambda_nm: ar_nm,
                aoi_deg: 0.0,
                periods: 1,
                n_high: 2.1,
                n_low: 1.38,
                adapt: None,
            };
            assert!(s.set_body_face_coating(eye.0, face, Some(ar)), "AR coating");
        }
    }

    // The light pipe, through the wall's 360 mm hole, its mouth flush with
    // the wall; in two lengths with a gap at 32 m, where the spectrograph's
    // pick-off takes its strip of light out sideways.
    let entry = vertex + EYE_T + TO_HOLE;
    let gap_at = entry + SPEC_AT;
    let half_gap = 0.5 * spectrograph::TUBE_GAP;
    // Built as a duct is: `TUBE_SECTION` (3 m) lengths, 2 mm apart at their
    // joints, each laid within installation tolerance (centre 0.5 mm RMS off
    // the line each way, axis 0.1 mrad), lined with enhanced silver whose
    // rolled sheet is wavy to `BORE_SLOPE` (1 mrad RMS). Between supports
    // 3 m apart an aluminium tube of this size sags 0.03 mm, which is left
    // out. `REAL_TUBE=0`: one perfect length each side of the gap.
    let section = if realism::on("TUBE") { env_f64("TUBE_SECTION", 3_000.0) } else { f64::INFINITY };
    let (off_rms, tilt_rms) = (env_f64("TUBE_OFFSET", 0.5), env_f64("TUBE_TILT", 0.1e-3));
    let mut k = 0u64;
    for (from, to, name) in [
        (entry, gap_at - half_gap, "0-32 m"),
        (gap_at + half_gap, entry + TUBE_LEN, "32-37 m"),
    ] {
        let n = ((to - from) / section).ceil().max(1.0) as usize;
        let joint = if n > 1 { 2.0 } else { 0.0 };
        let len = (to - from - joint * (n - 1) as f64) / n as f64;
        for i in 0..n {
            let a = from + i as f64 * (len + joint);
            let tube = s
                .add_tube(
                    Vec3::new(a + 0.5 * len, Y, 0.0),
                    0.0,
                    TubeDims {
                        inner_diameter_mm: 2.0 * BORE_RADIUS,
                        outer_diameter_mm: 2.0 * spectrograph::TUBE_OUTER_RADIUS,
                        length_mm: len,
                    },
                    BoreFinish::Metal {
                        metal: BoreMetal::Silver,
                        protected: true,
                    },
                )
                .expect("tube");
            if section.is_finite() {
                let g = |j: u64| realism::gauss(17, 4 * k + j);
                let at = Vec3::new(a + 0.5 * len, Y + off_rms * g(0), off_rms * g(1));
                let turn = glam::DQuat::from_rotation_z(tilt_rms * g(2)) * glam::DQuat::from_rotation_y(tilt_rms * g(3));
                let b = s.bodies.iter_mut().find(|b| b.common.id.0 == tube).unwrap();
                b.common.transform = Isometry3::new(at, [turn.x, turn.y, turn.z, turn.w]).expect("a section's pose");
                k += 1;
            }
            realism::bore(&mut s, tube, realism::enhanced_silver());
            realism::slope(&mut s, tube, env_f64("BORE_SLOPE", 1.0));
            label(&mut s, tube, &format!("Light pipe {name}, section {} of {n}, enhanced silver", i + 1));
        }
    }
    // The bore's mouth: light outside it strikes the tube's 2.5 mm rim or
    // the wall. Dark out to 1.5 m across.
    let mouth = s.add_aperture(Vec3::new(entry - 1.0, Y, 0.0), BORE_RADIUS, 0.0);
    if let Some(a) = s.apertures.iter_mut().find(|a| a.common.id.0 == mouth) {
        a.shape.outer_extent = Some(PlanarShape2::Disc { radius_mm: 750.0 });
    }
    label(&mut s, mouth, "Tube mouth in the wall, 355 mm bore, wall dark to 1500 mm");

    // Detectors, all facing the oncoming light (-X).
    let probe = DetectorTermination::ProbeAndContinue;
    let d1 = rect(
        &mut s,
        x_fresnel - 40.0,
        600.0,
        120,
        probe,
        "Before Fresnel",
    );
    let d2 = rect(&mut s, valleys + 20.0, 550.0, 110, probe, "After Fresnel");
    // A measuring plane across the whole bore just before the slit: it
    // records and lets every ray on, so it stands for no object.
    let d3 = disc(&mut s, gap_at - 10.0, probe, "Tube, 32 m (whole bore)");
    // The spectrograph, its slit on the axis in the gap.
    let spec = spectrograph::add(&mut s, Vec3::new(gap_at, Y, 0.0));
    // The melting experiment: the light leaves the tube's end by itself, and
    // this face 1 mm past it catches all of it, 600 mm across against the
    // 360 mm tube.
    let exit = s.add_area_detector(
        Vec3::new(entry + TUBE_LEN + 1.0, Y, 0.0),
        -Vec3::X,
        1.0,
        1.0,
        [120, 120],
    );
    if let Some(d) = s.detectors.iter_mut().find(|d| d.common.id.0 == exit.0) {
        d.geometry = DetectorGeometry3::Disc { radius_mm: 300.0 };
        // `EXIT_PASS=1`: the face records and lets the light on, to optics
        // placed after the tube.
        d.termination = if std::env::var_os("EXIT_PASS").is_some() { DetectorTermination::ProbeAndContinue } else { DetectorTermination::Terminate };
        d.common.label = "Melting experiment, 1 mm past the tube's end".into();
    }
    let d4 = ObjectId(exit.0);

    // Diagnostic probes, only when asked for (not in the written file).
    let mut extra = Vec::new();
    if std::env::var_os("DIAG").is_some() {
        let d = rect(
            &mut s,
            vertex - 25.0,
            230.0,
            92,
            probe,
            "diag: before eyepiece",
        );
        extra.push(("before eyepiece", d));
        for (name, at) in [
            ("tube +1 mm", 1.0),
            ("tube 5 m", 5_000.0),
            ("tube 10 m", 10_000.0),
            ("tube 20 m", 20_000.0),
        ] {
            extra.push((name, disc(&mut s, entry + at, probe, name)));
        }
    }

    // 2048 rays, each with its own origin, direction and wavelength (the sun
    // is a continuum of G173's 1661 intervals, limb-darkened, with a 5%
    // aureole): few enough that the optics stay visible under the rays. The
    // component view's analysis rays trace the numbers with more.
    let src = s.solar.as_ref().unwrap().source;
    if let Some(src) = s.sources.iter_mut().find(|x| x.id == src) {
        src.samples.spatial_samples = 1024;
        src.samples.angular_samples = 2;
    }
    s.resolve_solar().expect("sun resolves");
    s.validate().expect("scene validates");
    Built {
        slit: spec.pickoff,
        scene: s,
        detectors: [
            vec![("before Fresnel", d1), ("after Fresnel", d2)],
            extra,
            vec![
                ("tube 32 m bore", d3),
                ("spectrograph entrance", spec.entrance),
                ("table", spec.table),
                ("tube exit 37 m", d4),
            ],
        ]
        .concat(),
    }
}

/// An absorbing stop `gap_mm` in front of surface `parent` along its local
/// `+Z`, parallel to it, open over `opening` and dark out to `outer`, locked
/// to it so it moves and turns with it.
fn stop_on(
    s: &mut Scene3,
    parent: u64,
    gap_mm: f64,
    opening: PlanarShape2,
    outer: PlanarShape2,
    name: &str,
) -> u64 {
    let at = s
        .surfaces
        .iter()
        .find(|x| x.common.id.0 == parent)
        .expect("the stop's mirror")
        .common
        .transform
        .compose(Isometry3::from_pos(Vec3::new(0.0, 0.0, gap_mm)));
    let id = s.add_aperture(Vec3::ZERO, 1.0, 0.0);
    let a = s
        .apertures
        .iter_mut()
        .find(|x| x.common.id.0 == id)
        .expect("the stop");
    a.common.transform = at;
    a.shape.clear_opening = opening;
    a.shape.outer_extent = Some(outer);
    s.link_object(id, parent, LinkMode::Follow).expect("lock the stop");
    label(s, id, name);
    id
}

pub(crate) fn label(s: &mut Scene3, id: u64, text: &str) {
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

fn rect(
    s: &mut Scene3,
    x: f64,
    half: f64,
    bins: usize,
    t: DetectorTermination,
    name: &str,
) -> ObjectId {
    let id = s.add_area_detector(Vec3::new(x, Y, 0.0), -Vec3::X, half, half, [bins, bins]);
    let d = s
        .detectors
        .iter_mut()
        .find(|d| d.common.id.0 == id.0)
        .unwrap();
    d.termination = t;
    d.common.label = name.into();
    ObjectId(id.0)
}

/// Just inside the bore, 5 mm pixels.
fn disc(s: &mut Scene3, x: f64, t: DetectorTermination, name: &str) -> ObjectId {
    let id = s.add_area_detector(Vec3::new(x, Y, 0.0), -Vec3::X, 1.0, 1.0, [76, 76]);
    let d = s
        .detectors
        .iter_mut()
        .find(|d| d.common.id.0 == id.0)
        .unwrap();
    d.geometry = DetectorGeometry3::Disc {
        radius_mm: BORE_RADIUS - 0.1,
    };
    d.termination = t;
    d.common.label = name.into();
    ObjectId(id.0)
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "heliostat-lightpipe.ocanvas".into());
    let Built { slit, scene, detectors } = build();
    if std::env::var_os("DIAG").is_none() {
        std::fs::write(&out, scene.to_json()).expect("write the scene");
        eprintln!("wrote {out}");
    }

    // The bench alone. Without `TUBE_GAIN` it takes a straight line through
    // the gain the ideal parts gave in the old 380 mm bore, with the
    // Fresnel's MgF2 coat (bare: 4.82 and 5.76).
    // `BENCH_GAIN="g400 g700"` sets it instead.
    if std::env::var_os("BENCH_ONLY").is_some() {
        let g: Vec<f64> = std::env::var("BENCH_GAIN").map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        let (g4, g7) = if let [a, b] = g[..] { (a, b) } else { (5.05, 6.03) };
        if let Ok(path) = std::env::var("TUBE_PROFILE") {
            read_profile(&path);
        }
        // `TUBE_GAIN=<file>`: the bands a whole run wrote (`TUBE_GAIN_OUT`).
        let gain: Vec<(f64, f64)> = match std::env::var("TUBE_GAIN") {
            Ok(path) => std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{path}: {e}"))
                .lines()
                .filter_map(|l| {
                    let v: Vec<f64> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                    (v.len() == 2).then(|| (v[0], v[1]))
                })
                .collect(),
            Err(_) => vec![(0.4, g4), (0.7, g7)],
        };
        run_bench(slit, &site_rows(), &gain);
        return;
    }
    let mut s = scene;
    if let Ok(n) = std::env::var("VERIFY") {
        let n: u32 = n.parse().expect("VERIFY is a sample count");
        let src = s.solar.as_ref().unwrap().source;
        if let Some(src) = s.sources.iter_mut().find(|x| x.id == src) {
            src.samples.spatial_samples = n;
        }
    }
    let state = s.resolve_solar().unwrap();
    eprintln!(
        "sun: az {:.2}°, el {:.2}°, beam {:.1} W; heliostat incidence {:.2}°",
        state.azimuth_deg,
        state.elevation_deg,
        state.source_flux_w,
        state.heliostats[0].incidence_deg
    );
    if std::env::var_os("POINT_SUN").is_some() {
        s.sources[0].angular = oc_core::scene3::AngularDistribution3::UniformConeSolidAngle {
            half_angle_rad: 1e-7,
        };
    }
    let (ghosts, max_ghost_bounces) = realism::ghosts();
    let cfg = TraceConfig3 {
        ghosts,
        max_ghost_bounces,
        max_depth: depth_for_scene3(&s, ghosts, max_ghost_bounces),
        bounds_mm: bounds_for_scene3(&s),
        record_detector_hits: true,
        max_records: 200_000,
        ..TraceConfig3::default()
    };
    // `FIDELITY=Fast|Standard|Reference`: the coating physics the trace pays for.
    let cfg = cfg.with_fidelity(realism::fidelity());
    let t0 = std::time::Instant::now();
    let res = trace_scene3(&s, &cfg);
    eprintln!(
        "traced in {:.2?}, depth budget {}",
        t0.elapsed(),
        cfg.max_depth
    );
    let detectors_all = detectors.clone();
    let detectors_exit = detectors.iter().copied().find(|(n, _)| *n == "tube exit 37 m");
    for (name, id) in detectors {
        let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
        // Radial extent and angle to the axis, flux weighted.
        let (mut w, mut r2, mut a2, mut rmax, mut amax) = (0.0, 0.0, 0.0, 0.0f64, 0.0f64);
        for h in &d.records {
            let r = h.uv_mm.length();
            let a = (-h.incident_dir_local.z).clamp(-1.0, 1.0).acos();
            w += h.incident_flux;
            r2 += h.incident_flux * r * r;
            a2 += h.incident_flux * a * a;
            rmax = rmax.max(r);
            amax = amax.max(a);
        }
        eprintln!(
            "{name:>16}: {:8.2} W  ({} hits)  r rms {:.1} max {:.1} mm, angle rms {:.2} max {:.2} deg",
            d.total_incident,
            d.hits,
            (r2 / w).sqrt(),
            rmax,
            (a2 / w).sqrt().to_degrees(),
            amax.to_degrees()
        );
        if name == "tube +1 mm" {
            // A continuum gives every ray its own wavelength: 100 nm bands.
            let band = |l: f64| (l * 10.0).floor() / 10.0;
            let mut lams: Vec<f64> = d.records.iter().map(|h| band(h.lambda_um)).collect();
            lams.sort_by(f64::total_cmp);
            lams.dedup();
            for l in lams {
                let (mut w, mut a2) = (0.0, 0.0);
                for h in d.records.iter().filter(|h| band(h.lambda_um) == l) {
                    let a = (-h.incident_dir_local.z).clamp(-1.0, 1.0).acos();
                    w += h.incident_flux;
                    a2 += h.incident_flux * a * a;
                }
                eprintln!(
                    "{:>20} {l:.3} um: {w:7.2} W, angle rms {:.3} deg",
                    "",
                    (a2 / w).sqrt().to_degrees()
                );
            }
        }
    }
    // The exit's far field as a histogram of the angle to the tube's axis: a
    // disc of directions fills it from 0 up, and a few directions shared by
    // every ray would stand in it as rings.
    if let Some((_, id)) = detectors_exit {
        let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
        let mut h = [0.0f64; 16];
        let mut all = 0.0;
        for r in &d.records {
            let a = (-r.incident_dir_local.z).clamp(-1.0, 1.0).acos().to_degrees();
            h[((a / 0.1) as usize).min(15)] += r.incident_flux;
            all += r.incident_flux;
        }
        let row: Vec<String> = h.iter().map(|v| format!("{:.0}", 100.0 * v / all)).collect();
        eprintln!("exit, % of flux per 0.1 deg from the axis: {}", row.join(" "));
    }
    eprintln!("{:#?}", res.terminations);
    // `EXIT_OUT=<file.csv>`: every ray the tube's exit detector caught, in
    // world coordinates (mm, unit direction), with its wavelength and flux.
    if let (Ok(path), Some((_, id))) = (std::env::var("EXIT_OUT"), detectors_exit) {
        let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
        let frame = s.detectors.iter().find(|q| q.common.id.0 == id.0).unwrap().common.transform;
        let mut csv = String::from("x_mm,y_mm,z_mm,dx,dy,dz,lambda_um,flux_w
");
        for h in &d.records {
            let dir = frame.rotate(h.incident_dir_local);
            let p = h.point_world;
            csv.push_str(&format!("{:.4},{:.4},{:.4},{:.8},{:.8},{:.8},{:.5},{:.6e}
", p.x, p.y, p.z, dir.x, dir.y, dir.z, h.lambda_um, h.incident_flux));
        }
        std::fs::write(&path, csv).expect("write the exit rays");
        eprintln!("wrote {} exit rays ({:.2} W) to {path}", d.records.len(), d.total_incident);
    }
    spectra(&s, &res, &detectors_all);
    let bore = detectors_all.iter().find(|(n, _)| *n == "tube 32 m bore").unwrap().1;
    let rows = site_rows();
    let gain = tube_gain(&res, bore, &rows);
    tube_profile(&res, bore);
    eprintln!(
        "tube gain at 32 m (irradiance over the sun's): {}",
        gain.iter().map(|(l, g)| format!("{:.0}:{:.2}", l * 1e3, g)).collect::<Vec<_>>().join(" ")
    );
    // The demonstrator's trace holds every segment it drew; the bench needs
    // none of it.
    drop(res);
    drop(s);
    let gain = smooth_gain(&gain);
    eprintln!(
        "  smoothed: {}",
        gain.iter().map(|(l, g)| format!("{:.0}:{:.2}", l * 1e3, g)).collect::<Vec<_>>().join(" ")
    );
    // Written for `BENCH_ONLY`, which reads it back from `TUBE_GAIN`.
    if let Ok(path) = std::env::var("TUBE_GAIN_OUT") {
        let text: String = gain.iter().map(|(l, g)| format!("{l:.4} {g:.6}
")).collect();
        std::fs::write(&path, text).expect("write the tube gain");
    }
    // `NO_BENCH=1`: the light pipe only.
    if std::env::var_os("NO_BENCH").is_some() {
        return;
    }
    run_bench(slit, &rows, &gain);
}

/// The spectrum at 32 m against what reaches the melting experiment, in
/// 200 nm bands, and what the spectrograph takes.
fn spectra(s: &Scene3, res: &oc_trace::TraceResult3, dets: &[(&str, ObjectId)]) {
    const EDGES: [f64; 12] = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5];
    let band = |l: f64| EDGES.windows(2).position(|w| l >= w[0] && l < w[1]);
    let get = |name: &str| {
        let id = dets.iter().find(|(n, _)| *n == name).unwrap().1;
        let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
        let mut w = [0.0f64; 11];
        for l in &d.spectral {
            if let Some(b) = band(l.mean_lambda_um()) {
                w[b] += l.flux;
            }
        }
        (w, d.total_incident)
    };
    let mut launched = [0.0f64; 11];
    for r in oc_trace::launch_from_scene(s) {
        if let Some(b) = band(r.lambda_um) {
            launched[b] += r.flux;
        }
    }
    let (bore, bore_t) = get("tube 32 m bore");
    let (_, entrance_t) = get("spectrograph entrance");
    let (exit, exit_t) = get("tube exit 37 m");
    eprintln!(
        "32 m bore {:.1} W; the spectrograph takes {:.3} W after its filter; melting experiment {:.2} W",
        bore_t, entrance_t, exit_t
    );
    eprintln!(" band nm    sun W  bore %  exit %  exit/sun");
    for b in 0..11 {
        eprintln!(
            "{:>4}-{:<4} {:8.2} {:7.2} {:7.2} {:8.3}",
            (EDGES[b] * 1e3) as u32,
            (EDGES[b + 1] * 1e3) as u32,
            launched[b],
            100.0 * bore[b] / bore_t,
            100.0 * exit[b] / exit_t,
            exit[b] / launched[b].max(1e-300)
        );
    }
}

/// The CIE 1924 photopic curve, as the usual one-Gaussian fit (within a few
/// percent between 450 and 650 nm, where the spectrum's light is).
fn photopic(lambda_um: f64) -> f64 {
    1.019 * (-285.4 * (lambda_um - 0.559).powi(2)).exp()
}

/// The tube's spectral irradiance at 32 m over the sun's, in 20 nm bands
/// from 360 to 740 nm: how the heliostat, the Galilean and 32 m of silver
/// concentrate and colour the light. `(band centre µm, gain)`.
fn tube_gain(res: &oc_trace::TraceResult3, bore: ObjectId, rows: &[[f64; 2]]) -> Vec<(f64, f64)> {
    let d = res.detectors.iter().find(|d| d.detector.0 == bore.0).unwrap();
    let area_m2 = std::f64::consts::PI * (BORE_RADIUS * 1e-3).powi(2);
    (18..37)
        .map(|k| {
            let (lo, hi) = (k as f64 * 0.020, (k + 1) as f64 * 0.020);
            let flux: f64 = d
                .spectral
                .iter()
                .filter(|l| l.lambda_um >= lo && l.lambda_um < hi)
                .map(|l| l.flux)
                .sum();
            let sun = oc_core::solar::table_power(rows, lo, hi);
            (0.5 * (lo + hi), flux / area_m2 / sun)
        })
        .collect()
}

/// The bands' gains with single stray bands taken out: each the median of
/// itself and its two neighbours.
///
/// The bands carry a few percent of sampling noise and, now and then, one
/// band 10-15% off its neighbours that moves to another band when anything
/// changes: the sun's sampler ties a ray's wavelength to where on the mirror
/// it starts, so a band can draw its rays from a stretch of the beam that
/// loses more. A median of three removes such a band and keeps any monotonic
/// run as it is, the eyepiece's violet edge included (0 at 370 nm, 2 at 410,
/// 5 at 550, through the dense flint's absorption), which a straight line
/// through the bands, the old smoothing, cannot follow. `GAIN_FIT=line` puts
/// the line back.
fn smooth_gain(gain: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if std::env::var("GAIN_FIT").is_ok_and(|v| v == "line") {
        let n = gain.len() as f64;
        let (mx, my) = (gain.iter().map(|g| g.0).sum::<f64>() / n, gain.iter().map(|g| g.1).sum::<f64>() / n);
        let sxy: f64 = gain.iter().map(|g| (g.0 - mx) * (g.1 - my)).sum();
        let sxx: f64 = gain.iter().map(|g| (g.0 - mx).powi(2)).sum();
        let slope = sxy / sxx;
        return gain.iter().map(|g| (g.0, my + slope * (g.0 - mx))).collect();
    }
    (0..gain.len())
        .map(|i| {
            if i == 0 || i + 1 == gain.len() {
                return gain[i];
            }
            let mut v = [gain[i - 1].1, gain[i].1, gain[i + 1].1];
            v.sort_by(f64::total_cmp);
            (gain[i].0, v[1])
        })
        .collect()
}

fn gain_at(gain: &[(f64, f64)], l: f64) -> f64 {
    let i = gain.partition_point(|g| g.0 < l);
    if i == 0 {
        return gain[0].1;
    }
    if i >= gain.len() {
        return gain[gain.len() - 1].1;
    }
    let ((x0, y0), (x1, y1)) = (gain[i - 1], gain[i]);
    y0 + (y1 - y0) * (l - x0) / (x1 - x0)
}

/// The tube's light at 32 m as a beam: its angular profile, measured by the
/// whole run ([`tube_profile`]) or read from `TUBE_PROFILE`.
static PROFILE: std::sync::OnceLock<(f64, [f64; oc_core::scene3::RADIAL_NODES])> = std::sync::OnceLock::new();

/// The directions the bench's sun leaves in: the measured profile when
/// there is one (`REAL_FEED=0` ignores it), else the sun's limb-darkened
/// disc through the Galilean's 2.97, which is all the tube would carry with
/// true optics and no wall.
fn tube_angular() -> oc_core::scene3::AngularDistribution3 {
    match PROFILE.get() {
        Some(&(half_angle_rad, radiance)) if realism::on("FEED") => {
            oc_core::scene3::AngularDistribution3::RadialProfile { half_angle_rad, radiance }
        }
        _ => oc_core::scene3::AngularDistribution3::Sun {
            half_angle_rad: spectrograph::TUBE_HALF_ANGLE,
            disc: oc_core::solar::sunshape::SunDisc::LimbDarkened,
            csr: 0.0,
        },
    }
}

/// The angular profile of the light crossing the bore at 32 m, from the
/// whole run's records: radiance at [`oc_core::scene3::RADIAL_NODES`]
/// angles out to `PROFILE_DEG` (2.5 degrees) from the axis, from rays going
/// downstream. It holds the sun's disc as the Galilean magnified it, the
/// eyepiece's aberration, the mirrors' slope errors and what the wall's
/// bounces fold back in. Written to `TUBE_PROFILE_OUT` (`tube-profile.txt`), which
/// `TUBE_PROFILE` reads back for `BENCH_ONLY`.
fn tube_profile(res: &oc_trace::TraceResult3, bore: ObjectId) {
    use oc_core::scene3::RADIAL_NODES;
    let d = res.detectors.iter().find(|d| d.detector.0 == bore.0).unwrap();
    let half = env_f64("PROFILE_DEG", 2.5).to_radians();
    let n = RADIAL_NODES - 1;
    let step = half / n as f64;
    // Power in a ring of half a step either side of each node.
    let mut ring = [0.0f64; RADIAL_NODES];
    let (mut inside, mut all) = (0.0, 0.0);
    for h in &d.records {
        let cos_t = -h.incident_dir_local.z;
        if cos_t <= 0.0 {
            continue;
        }
        let t = cos_t.clamp(-1.0, 1.0).acos();
        all += h.incident_flux;
        let k = (t / step).round() as usize;
        if k < RADIAL_NODES {
            ring[k] += h.incident_flux;
            inside += h.incident_flux;
        }
    }
    let mut radiance = [0.0f64; RADIAL_NODES];
    for k in 0..RADIAL_NODES {
        let (a, b) = (((k as f64) - 0.5).max(0.0) * step, ((k as f64) + 0.5).min(n as f64 + 0.5) * step);
        radiance[k] = ring[k] / (a.cos() - b.cos());
    }
    let peak = radiance.iter().cloned().fold(0.0, f64::max).max(1e-300);
    for v in &mut radiance {
        *v /= peak;
    }
    eprintln!(
        "tube profile at 32 m: {:.1}% of the downstream light within {:.2} deg; radiance by node: {}",
        100.0 * inside / all.max(1e-300),
        half.to_degrees(),
        radiance.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" ")
    );
    let path = std::env::var("TUBE_PROFILE_OUT").unwrap_or_else(|_| "tube-profile.txt".into());
    let text = format!("{}
{}
", half, radiance.iter().map(|v| format!("{v:.6e}")).collect::<Vec<_>>().join(" "));
    std::fs::write(&path, text).expect("write the tube profile");
    let _ = PROFILE.set((half, radiance));
}

/// Read a profile [`tube_profile`] wrote.
fn read_profile(path: &str) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines();
    let half: f64 = lines.next().and_then(|l| l.trim().parse().ok()).expect("the profile's half angle");
    let v: Vec<f64> = lines.next().unwrap_or("").split_whitespace().filter_map(|x| x.parse().ok()).collect();
    let radiance: [f64; oc_core::scene3::RADIAL_NODES] = v.try_into().expect("a profile of RADIAL_NODES values");
    let _ = PROFILE.set((half, radiance));
}

/// The spectrograph on its own, fed by what the tube carries at 32 m: a
/// source the size of the slit, lit with the tube's spectrum (the sun's
/// times [`tube_gain`]) over `lo_um..hi_um`, and with its directions (the
/// sun's limb-darkened disc through the Galilean's 2.97, which the tube
/// keeps). The aureole is left out: at up to 6 degrees it misses the
/// collimator.
fn bench(slit: Vec3, rows: &[[f64; 2]], gain: &[(f64, f64)], lo_um: f64, hi_um: f64, samples: u32) -> (Scene3, spectrograph::Spectrograph) {
    use oc_core::scene3::{AngularDistribution3, SamplingSequence, Source3, SourceSampling, SpatialDistribution3};
    let mut s = Scene3::new();
    let spec = spectrograph::add(&mut s, slit);
    let tube: Vec<[f64; 2]> = rows
        .iter()
        .filter(|r| r[0] >= 370.0 && r[0] <= 730.0)
        .map(|r| [r[0], r[1] * gain_at(gain, r[0] / 1e3)])
        .collect();
    let lines = oc_core::solar::table_continuum(&tube, lo_um, hi_um);
    let power: f64 = lines.iter().map(|l| l.1).sum();
    // The source lights the pick-off's strip, with a margin, as the tube does,
    // from upstream of the whole strip: the strip leans along the tube.
    let (hw, hh) = (0.5 * spec.pickoff_size.0 + 2.0, 0.5 * spec.pickoff_size.1 + 0.5 + spec.pickoff_bow);
    let id = s.alloc_source_id();
    s.sources.push(Source3 {
        window: Default::default(),
        id,
        // Local X along the tube, local Y (the width) along the strip.
        transform: spectrograph::frame_x(spec.pickoff - Vec3::X * (spec.pickoff_depth + 20.0), Vec3::X, spec.pickoff_along),
        spatial: SpatialDistribution3::Rectangle {
            half_width_mm: hw,
            half_height_mm: hh,
        },
        angular: tube_angular(),
        spectrum: oc_core::scene::Spectrum::Continuum(lines),
        flux: power * 4.0 * hw * hh * 1e-6,
        samples: SourceSampling {
            spatial_samples: samples,
            angular_samples: 2,
            seed: 0,
            sequence: SamplingSequence::Sobol,
        },
    });
    // `SUN_PIECES=<n>`: the same light from `n` rectangles that follow the
    // strip's arc ([`sun_pieces`]).
    if let Some(k) = std::env::var("SUN_PIECES").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&k| k > 0) {
        sun_pieces(&mut s, &spec, id, k);
    }
    // `SLIT_SOURCE`: the slit itself shines instead, with the tube's
    // radiance (the pick-off and condenser mirrors' two reflections taken
    // off) into a round cone as wide as the collimator's beam along the
    // dispersion, a little over: the condenser, made anamorphic, would fill
    // the collimator's wider beam along the spectrum; this skips it.
    if std::env::var_os("SLIT_SOURCE").is_some() {
        let n = 24;
        let pieces = spectrograph::slit_emitters(&spec, n);
        let half = 1.05 * 0.5 * spectrograph::BEAM.max(spectrograph::beam_along()) / spectrograph::F_COLL;
        let (hw, hh) = (0.5 * spec.slit_height / n as f64 + 0.2, 0.5 * spectrograph::slit_width() + 0.15);
        let t = spectrograph::TUBE_HALF_ANGLE;
        let base = s.sources.iter().find(|x| x.id == id).unwrap().clone();
        s.sources.retain(|x| x.id != id);
        for (p, dir, run) in pieces {
            let mut src = base.clone();
            src.id = s.alloc_source_id();
            src.transform = spectrograph::frame_x(p - dir * 1.0, dir, run);
            src.spatial = SpatialDistribution3::Rectangle { half_width_mm: hw, half_height_mm: hh };
            src.angular = AngularDistribution3::UniformConeSolidAngle { half_angle_rad: half };
            src.flux = power * 4.0 * hw * hh * 1e-6 * (half.sin() / t.sin()).powi(2) * 0.99 * 0.99;
            src.samples.spatial_samples = (samples / n as u32).max(1);
            s.sources.push(src);
        }
    }
    // The lamps: the same wavelengths, and more rays, since most of theirs
    // miss the slit or the collimator's f/36.
    for src in s.sources.iter_mut().filter(|x| spec.lamp_sources.contains(&x.id)) {
        if let oc_core::scene::Spectrum::Continuum(bands) = &mut src.spectrum {
            let all: f64 = bands.iter().map(|b| b.1).sum();
            bands.retain(|b| b.0 >= lo_um && b.0 <= hi_um);
            let kept: f64 = bands.iter().map(|b| b.1).sum();
            src.flux *= kept / all;
        }
        // Each source's share as it was built (2048 for a whole lamp).
        src.samples.spatial_samples = (samples as f64 * env_f64("LAMP_RAYS", 16.0) * src.samples.spatial_samples as f64 / 2048.0) as u32;
    }
    // A lamp with no light in the band is left out; `BENCH_SUN=1` leaves
    // every lamp out, their arms still there.
    s.sources.retain(|x| !matches!(&x.spectrum, oc_core::scene::Spectrum::Continuum(b) if b.is_empty()));
    if std::env::var_os("BENCH_SUN").is_some() {
        s.sources.retain(|x| !spec.lamp_sources.contains(&x.id));
    }
    // `BENCH_LAMPS=1`: the lamps alone.
    if std::env::var_os("BENCH_LAMPS").is_some() {
        s.sources.retain(|x| spec.lamp_sources.contains(&x.id));
    }
    // `BENCH_DROP="a|b"`: objects whose label holds any of these left out.
    if let Ok(v) = std::env::var("BENCH_DROP") {
        let keys: Vec<&str> = v.split('|').collect();
        let hit = |l: &str| keys.iter().any(|k| l.contains(k));
        s.surfaces.retain(|o| !hit(&o.common.label));
        s.bodies.retain(|o| !hit(&o.common.label));
        s.apertures.retain(|o| !hit(&o.common.label));
        s.detectors.retain(|o| !hit(&o.common.label));
    }
    (s, spec)
}

/// The bench's sun, source `id`, as `k` rectangles that follow the pick-off
/// strip's arc, each as wide as the strip and a millimetre either side, with
/// the same irradiance: the one rectangle holds the whole bow, and most of
/// its rays miss. The sun band's lux comes out 0.4 to 0.9% under the one
/// rectangle's (edge light the margin leaves out), so the numbers keep the
/// rectangle and the table picture takes the pieces.
fn sun_pieces(s: &mut Scene3, spec: &spectrograph::Spectrograph, id: oc_core::id::SourceId, k: usize) {
    use oc_core::scene3::SpatialDistribution3;
    let Some(base) = s.sources.iter().find(|x| x.id == id).cloned() else { return };
    let SpatialDistribution3::Rectangle { half_width_mm, half_height_mm } = base.spatial else { return };
    let irradiance = base.flux / (4.0 * half_width_mm * half_height_mm * 1e-6);
    let samples = base.samples.spatial_samples;
    s.sources.retain(|x| x.id != id);
    let across = 0.5 * spec.pickoff_size.1 + 1.0;
    let x0 = spec.pickoff.x - (spec.pickoff_depth + 20.0);
    let pieces = spectrograph::pickoff_pieces(spec, k);
    let last = pieces.len() - 1;
    for (i, (p, run, half)) in pieces.into_iter().enumerate() {
        // The two end pieces reach 2 mm past the strip's ends, as the one did.
        let (lo, hi) = (if i == 0 { 2.0 } else { 0.0 }, if i == last { 2.0 } else { 0.0 });
        let hw = half + 0.5 * (lo + hi);
        let centre = Vec3::new(x0, p.y, p.z) + run * (0.5 * (hi - lo));
        let mut src = base.clone();
        src.id = s.alloc_source_id();
        src.transform = spectrograph::frame_x(centre, Vec3::X, run);
        src.spatial = SpatialDistribution3::Rectangle { half_width_mm: hw, half_height_mm: across };
        src.flux = irradiance * 4.0 * hw * across * 1e-6;
        src.samples.spatial_samples = (samples / k as u32).max(1);
        s.sources.push(src);
    }
}

/// Trace `s` in passes of at most about 2 million rays, each with its own
/// sampling seed and its share of the flux, and add the detectors' readings
/// up: the lamps' rays in one go would not fit in memory. Sums the readings
/// the bench reads (flux and colour per pixel, totals, spectral sums), not
/// the per-hit records.
fn trace_passes(s: &Scene3, cfg: &TraceConfig3) -> oc_trace::TraceResult3 {
    // Each source in as many passes as its own rays need, at most half a
    // million a pass: split evenly, a source of a few hundred rays (the
    // sun's pieces) would trace one ray a pass and sample its cone badly.
    let own: Vec<u64> = s
        .sources
        .iter()
        .map(|x| ((x.samples.spatial_samples as f64 * x.samples.angular_samples as f64 * x.spectrum.ray_multiplicity() as f64) / 5.0e5).ceil().max(1.0) as u64)
        .collect();
    let passes = own.iter().copied().max().unwrap_or(1);
    let mut out: Option<oc_trace::TraceResult3> = None;
    for k in 0..passes {
        let mut p = s.clone();
        let mut i = 0;
        p.sources.retain(|_| {
            i += 1;
            k < own[i - 1]
        });
        for (src, &n) in p.sources.iter_mut().zip(own.iter().filter(|&&n| k < n)) {
            src.samples.spatial_samples = (src.samples.spatial_samples as u64).div_ceil(n) as u32;
            src.samples.seed += 7_919 * k;
            src.flux /= n as f64;
        }
        let r = trace_scene3(&p, cfg);
        match out.as_mut() {
            None => out = Some(r),
            Some(o) => {
                for (a, b) in o.detectors.iter_mut().zip(&r.detectors) {
                    a.hits += b.hits;
                    a.total_incident += b.total_incident;
                    for (x, y) in a.incident_flux.iter_mut().zip(&b.incident_flux) {
                        *x += y;
                    }
                    for (x, y) in a.colour_xyz.iter_mut().zip(&b.colour_xyz) {
                        for i in 0..3 {
                            x[i] += y[i];
                        }
                    }
                    for y in &b.spectral {
                        match a.spectral.iter_mut().find(|x| x.lambda_um == y.lambda_um) {
                            Some(x) => {
                                x.flux += y.flux;
                                x.lw += y.lw;
                                x.hits += y.hits;
                                x.v += y.v;
                                x.vv += y.vv;
                                x.u += y.u;
                                x.uu += y.uu;
                            }
                            None => a.spectral.push(*y),
                        }
                    }
                }
            }
        }
    }
    out.expect("at least one pass")
}

/// Each spectrum's lux on the table, averaged over 2.5 m and its height, and
/// its brightest 20 mm along.
fn band_lux(t: &oc_trace::DetectorResult3, bands: &[(&str, f64, f64)]) {
    let Some(colour) = t.colour_per_m2() else { return };
    let (nu, nv) = (t.bins[0], t.bins[1]);
    let (hw, hh) = (spectrograph::TABLE_HALF_LENGTH, 0.5 * nv as f64 * 2.0);
    // `TABLE_ROWS=1`: the mean lux of every sixth row, to check the bands.
    if std::env::var_os("TABLE_ROWS").is_some() {
        for j in (0..nv).step_by(6) {
            let v = -hh + (j as f64 + 0.5) * 2.0 * hh / nv as f64;
            eprintln!("ROW {v:+7.1} {:8.1}", (0..nu).map(|i| colour[j * nu + i][1]).sum::<f64>() / nu as f64);
        }
    }
    for (name, lo, hi) in bands {
        let rows: Vec<usize> = (0..nv)
            .filter(|&j| {
                let v = -hh + (j as f64 + 0.5) * 2.0 * hh / nv as f64;
                v >= *lo && v <= *hi
            })
            .collect();
        let mut cols = vec![0.0; nu];
        for &j in &rows {
            for (i, c) in cols.iter_mut().enumerate() {
                *c += colour[j * nu + i][1];
            }
        }
        let cols: Vec<f64> = cols.iter().map(|c| c / rows.len().max(1) as f64).collect();
        let px = 2.0 * hw / nu as f64;
        let mean = cols.iter().sum::<f64>() * px / spectrograph::SPECTRUM_LENGTH;
        let win = (20.0 / px) as usize;
        let best = cols.windows(win).map(|w| w.iter().sum::<f64>() / win as f64).fold(0.0, f64::max);
        eprintln!("  {name}: {mean:.1} lux over {:.2} m x {:.0} mm, brightest 20 mm {best:.0} lux", spectrograph::SPECTRUM_LENGTH / 1e3, hi - lo);
    }
}

/// Where a ray of `lambda_um` from the slit's centre lands on the table.
fn landing(s: &Scene3, slit: Vec3, table: ObjectId, lambda_um: f64) -> Option<Vec3> {
    // A ray that meets a Fresnel's riser goes astray, or lands far off: the
    // median, coordinate by coordinate, of five launched within half a
    // millimetre.
    let mut hits: Vec<Vec3> = [Vec3::ZERO, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z].iter().filter_map(|o| landing_from(s, slit + *o * 0.5, table, lambda_um)).collect();
    if hits.len() < 3 {
        return hits.first().copied();
    }
    let mut median = |f: fn(&Vec3) -> f64| {
        hits.sort_by(|a, b| f(a).total_cmp(&f(b)));
        f(&hits[hits.len() / 2])
    };
    Some(Vec3::new(median(|v| v.x), median(|v| v.y), median(|v| v.z)))
}

fn landing_from(s: &Scene3, slit: Vec3, table: ObjectId, lambda_um: f64) -> Option<Vec3> {
    let lr = oc_trace::LaunchRay3 {
        ray: oc_core::vec3::Ray3::new(slit - Vec3::X * 20.0, Vec3::X).ok()?,
        lambda_um,
        flux: 1.0,
        source: None,
        pol: None,
    };
    let cfg = TraceConfig3 {
        record_detector_hits: true,
        max_depth: 64,
        bounds_mm: bounds_for_scene3(s),
        ..TraceConfig3::default()
    };
    let res = oc_trace::trace3(s, &[lr], &cfg);
    let d = res.detectors.iter().find(|d| d.detector.0 == table.0)?;
    // The projection lens's faces split off weak reflections that reach the
    // table too: the ray's own path is the strongest hit.
    d.records.iter().max_by(|a, b| a.incident_flux.total_cmp(&b.incident_flux)).map(|h| h.point_world)
}

/// Where the bench's rays stop: each branch's last event, counted by the
/// object that ended it.
fn losses(s: &Scene3) {
    use oc_trace::TraceEventKind;
    let cfg = TraceConfig3 {
        ghosts: false,
        max_depth: depth_for_scene3(s, false, 0).max(64),
        bounds_mm: bounds_for_scene3(s),
        segment_floor: f64::INFINITY,
        record_events: true,
        ..TraceConfig3::default()
    };
    let res = trace_scene3(s, &cfg);
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
    for o in &s.detectors {
        names.insert(o.common.id.0, o.common.label.clone());
    }
    let mut ends: std::collections::HashMap<String, usize> = Default::default();
    let mut total = 0;
    // A launch ray's events come together; the last object each path met.
    let mut last: std::collections::HashMap<u64, String> = Default::default();
    for e in &res.events {
        if let TraceEventKind::Interact { surface, .. } = &e.kind {
            last.insert(e.path_id.0, names.get(&surface.object.0).cloned().unwrap_or_default());
        }
        let what = match &e.kind {
            TraceEventKind::Escaped => format!("escaped after {}", e.parent.map_or(last.get(&(e.path_id.0)), |_| last.get(&(e.path_id.0))).cloned().unwrap_or_else(|| "nothing".into())),
            TraceEventKind::Interact { surface, children, .. } if children.is_empty() => names.get(&surface.object.0).cloned().unwrap_or_else(|| format!("object {}", surface.object.0)),
            TraceEventKind::DetectorHit { detector, terminal: true, .. } => names.get(&detector.0).cloned().unwrap_or_else(|| format!("detector {}", detector.0)),
            TraceEventKind::ApertureBlocked { aperture } => names.get(&aperture.0).cloned().unwrap_or_else(|| format!("aperture {}", aperture.0)),
            _ => continue,
        };
        *ends.entry(what).or_default() += 1;
        total += 1;
        if !matches!(e.kind, TraceEventKind::Interact { .. }) {
            last.remove(&(e.path_id.0));
        }
    }
    let mut v: Vec<_> = ends.into_iter().collect();
    v.sort_by_key(|q| std::cmp::Reverse(q.1));
    eprintln!("LOSSES: {total} branch ends, {:.3} W launched, {:.3} W detected", res.ledger.launched, res.ledger.detected_incident);
    for (what, k) in v.iter().take(25) {
        eprintln!("LOSSES {:5.1}% {what}", 100.0 * *k as f64 / total as f64);
    }
}

/// The point `along` mm from `centre` along the spectrum, at height `y`.
fn on_table(centre: Vec3, along: f64, y: f64) -> Vec3 {
    let mut p = centre + spectrograph::table_axes().0 * along;
    p.y = y;
    p
}

/// How far `p` lies from `centre` along the spectrum, and across the bands
/// (the unturned `z`).
fn table_uw(p: Vec3, centre: Vec3) -> (f64, f64) {
    let (u, w) = spectrograph::table_axes();
    ((p - centre).dot(u), (p - centre).dot(w))
}

/// The sodium doublet up close: the tube's sunlight over 587.5 to 591 nm
/// (`NA_ZOOM` rays per pass, default 64 Mi), then each D line alone, onto a
/// 40 x 200 mm patch of 0.05 x 0.5 mm pixels centred on the doublet. Writes
/// each pixel grid (`NA_OUT` prefix) and where the two lines land.
fn na_zoom(slit: Vec3, rows: &[[f64; 2]], gain: &[(f64, f64)], cfg: &dyn Fn(&Scene3) -> TraceConfig3) {
    let n = env_f64("NA_ZOOM", 67_108_864.0).max(1.0) as u32;
    let prefix = std::env::var("NA_OUT").unwrap_or_else(|_| "na-zoom".into());
    let (d2, d1) = (0.588_995, 0.589_592);
    let (hw, hh, bins) = (20.0, 100.0, [800usize, 400usize]);
    let (s0, spec) = bench(slit, rows, gain, 0.5875, 0.591, n);
    let centre = spec.table_centre;
    let at = |l: f64| landing(&s0, spec.pickoff, spec.table, l).map(|p| table_uw(p, centre).0).unwrap_or(f64::NAN);
    let (x2, x1) = (at(d2), at(d1));
    let xc = 0.5 * (x1 + x2);
    let mut marks = String::from("lambda_um,x_mm\n");
    for l in [0.5875, d2, 0.58929, d1, 0.591] {
        marks.push_str(&format!("{l},{:.4}\n", at(l) - xc));
    }
    std::fs::write(format!("{prefix}-landings.csv"), marks).expect("write the landings");
    eprintln!("NA_ZOOM D2 lands at x {:+.3}, D1 at {:+.3} mm from their mean ({:.3} mm apart)", x2 - xc, x1 - xc, x1 - x2);
    let spectra: [(&str, Option<f64>); 3] = [("sun", None), ("d2", Some(d2)), ("d1", Some(d1))];
    for (tag, mono) in spectra {
        let mut s = s0.clone();
        if let Some(l) = mono {
            for src in &mut s.sources {
                src.spectrum = oc_core::scene::Spectrum::Monochromatic(l);
                src.samples.spatial_samples /= 4;
            }
        }
        let id = spectrograph::detector(
            &mut s,
            on_table(centre, xc, centre.y + spectrograph::table_normal().y * 0.3),
            spectrograph::table_normal(),
            spectrograph::table_axes().0,
            hw,
            hh,
            bins,
            DetectorTermination::ProbeAndContinue,
            "Table: sodium D zoom",
        );
        let t0 = std::time::Instant::now();
        let res = trace_passes(&s, &cfg(&s));
        let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
        // Irradiance per pixel, W/m2; rows run along -Z (v = Y x X).
        let px_m2 = (2.0 * hw / bins[0] as f64) * (2.0 * hh / bins[1] as f64) * 1e-6;
        let mut csv = format!("# {} x {} pixels over {} x {} mm, u fastest along +X from x {:+.3}, v along -Z; W/m2; centred on {:.3} mm\n", bins[0], bins[1], 2.0 * hw, 2.0 * hh, -hw, xc);
        for j in 0..bins[1] {
            let row: Vec<String> = (0..bins[0]).map(|i| format!("{:.5e}", d.incident_flux[j * bins[0] + i] / px_m2)).collect();
            csv.push_str(&row.join(","));
            csv.push('\n');
        }
        std::fs::write(format!("{prefix}-{tag}.csv"), csv).expect("write the zoom grid");
        let rays: u64 = s.sources.iter().map(|x| x.samples.spatial_samples as u64 * x.samples.angular_samples as u64).sum();
        eprintln!("NA_ZOOM {tag}: {rays} rays in {:.1?}, {} hits, {:.4e} W on the patch", t0.elapsed(), d.hits, d.total_incident);
    }
}

/// Each Fraunhofer line up close (`LINE_NM`, nm): the tube's sunlight over
/// `LINE_HALF` nm either side (default 1.5), then the line alone, each onto a
/// patch `LINE_HW` mm either side along (default 12) by 200 mm across, of
/// 0.1 x 2 mm pixels. `LINE_ZOOM` rays per sunlight pass (a quarter for the
/// line alone). Writes `{LINE_OUT}-{nm}-sun.csv` and `-mono.csv`, and the
/// dispersion there in `{LINE_OUT}-lines.csv`.
fn line_zoom(slit: Vec3, rows: &[[f64; 2]], gain: &[(f64, f64)], cfg: &dyn Fn(&Scene3) -> TraceConfig3) {
    let n = env_f64("LINE_ZOOM", 8_388_608.0).max(1.0) as u32;
    let prefix = std::env::var("LINE_OUT").unwrap_or_else(|_| "lines".into());
    let half = env_f64("LINE_HALF", 1.5) * 1e-3;
    let hw = env_f64("LINE_HW", 12.0);
    let (hh, bins) = (100.0, [(20.0 * hw) as usize, 100usize]);
    let nms: Vec<f64> = std::env::var("LINE_NM")
        .unwrap_or_else(|_| "382.04 393.37 486.13 518.36 588.995 656.28 686.72".into())
        .split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect();
    let mut index = String::from("nm,x_mm,mm_per_nm\n");
    for nm in nms {
        let l = nm * 1e-3;
        let (s0, spec) = bench(slit, rows, gain, l - half, l + half, n);
        let centre = spec.table_centre;
        let at = |l: f64| landing(&s0, spec.pickoff, spec.table, l).map(|p| table_uw(p, centre).0).unwrap_or(f64::NAN);
        let xc = at(l);
        let disp = (at(l + 0.0005) - at(l - 0.0005)).abs();
        index.push_str(&format!("{nm},{xc:.3},{disp:.4}\n"));
        for (tag, mono) in [("sun", false), ("mono", true)] {
            let mut s = s0.clone();
            if mono {
                for src in &mut s.sources {
                    src.spectrum = oc_core::scene::Spectrum::Monochromatic(l);
                    src.samples.spatial_samples /= 4;
                }
            }
            let id = spectrograph::detector(
                &mut s,
                on_table(centre, xc, centre.y + spectrograph::table_normal().y * 0.3),
                spectrograph::table_normal(),
                spectrograph::table_axes().0,
                hw,
                hh,
                bins,
                DetectorTermination::ProbeAndContinue,
                "Table: line zoom",
            );
            let t0 = std::time::Instant::now();
            let res = trace_passes(&s, &cfg(&s));
            let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
            let px_m2 = (2.0 * hw / bins[0] as f64) * (2.0 * hh / bins[1] as f64) * 1e-6;
            let mut csv = format!("# {} x {} pixels over {} x {} mm, u fastest along the spectrum from {:+.3}, v across; W/m2; centred on {nm} nm at {xc:.3} mm\n", bins[0], bins[1], 2.0 * hw, 2.0 * hh, -hw);
            for j in 0..bins[1] {
                let row: Vec<String> = (0..bins[0]).map(|i| format!("{:.5e}", d.incident_flux[j * bins[0] + i] / px_m2)).collect();
                csv.push_str(&row.join(","));
                csv.push('\n');
            }
            std::fs::write(format!("{prefix}-{nm}-{tag}.csv"), csv).expect("write the line grid");
            eprintln!("LINE_ZOOM {nm} {tag}: {:.1?}, {} hits, {:.4e} W on the patch; {disp:.3} mm/nm", t0.elapsed(), d.hits, d.total_incident);
        }
    }
    std::fs::write(format!("{prefix}-lines.csv"), index).expect("write the line index");
}

/// Where each part of the band focuses: one wavelength at a time onto planes
/// parallel to the table from 60 mm above it to 60 mm below (the table lets
/// the light through), each cut into 12 slices across the band. Prints each
/// slice's line centroid, rms width and FWHM at the table, and the height
/// where its rms width is least.
fn focus_scan(slit: Vec3, rows: &[[f64; 2]], gain: &[(f64, f64)], cfg: &dyn Fn(&Scene3) -> TraceConfig3) {
    let n = env_f64("FOCUS_SCAN", 8_388_608.0).max(1.0) as u32;
    let lambdas: Vec<f64> = std::env::var("SCAN_NM")
        .unwrap_or_else(|_| "375 450 530 589 640 690".into())
        .split_whitespace()
        .filter_map(|x| x.parse::<f64>().ok())
        .map(|x| x / 1e3)
        .collect();
    // `SCAN_AT=mid`: at the intermediate spectrum instead, whose light goes
    // up through its level plane; the Fresnels sit 9 to 21 mm below it.
    let mid = std::env::var("SCAN_AT").is_ok_and(|v| v == "mid");
    let (hw, hh, nu, ns) = if mid { (12.0, 50.0, 1200usize, 12usize) } else { (25.0, 100.0, 1000usize, 12usize) };
    let du = 2.0 * hw / nu as f64;
    let step = if mid { 3.0 } else { 15.0 };
    let heights: Vec<f64> = if mid { (-10..=4).map(|k| k as f64 * step).collect() } else { (-4..=4).map(|k| k as f64 * step).collect() };
    let (mut s0, spec) = bench(slit, rows, gain, 0.37, 0.70, n);
    let (centre, base_y, target) = if mid {
        let c = spec.intermediate;
        let id = spectrograph::detector(&mut s0, c, Vec3::Y, spectrograph::table_axes().0, 900.0, 150.0, [1, 1], DetectorTermination::ProbeAndContinue, "intermediate spectrum");
        (c, c.y + 0.2, id)
    } else {
        (spec.table_centre, spec.table_centre.y + spectrograph::table_normal().y * 0.3, spec.table)
    };
    let mut out = String::from("lambda_nm,dy_mm,slice,z_mm,centroid_mm,rms_mm,fwhm_mm,flux\n");
    for &l in &lambdas {
        let Some(p) = landing(&s0, spec.pickoff, target, l) else {
            eprintln!("FOCUS {:.0} nm: the probe ray misses", l * 1e3);
            continue;
        };
        let mut s = s0.clone();
        for src in &mut s.sources {
            src.spectrum = oc_core::scene::Spectrum::Monochromatic(l);
        }
        if let Some(t) = s.detectors.iter_mut().find(|t| t.common.id.0 == spec.table.0) {
            t.termination = DetectorTermination::ProbeAndContinue;
        }
        let ids: Vec<ObjectId> = heights
            .iter()
            .map(|dy| {
                spectrograph::detector(
                    &mut s,
                    on_table(centre, table_uw(p, centre).0, base_y + if mid { *dy } else { spectrograph::table_normal().y * dy }),
                    if mid { Vec3::Y } else { spectrograph::table_normal() },
                    spectrograph::table_axes().0,
                    hw,
                    hh,
                    [nu, 2 * ns + 4],
                    DetectorTermination::ProbeAndContinue,
                    &format!("focus scan {dy:+.0}"),
                )
            })
            .collect();
        let res = trace_passes(&s, &cfg(&s));
        // Per plane and slice: centroid, rms, FWHM along the spectrum.
        type Spot = (f64, f64, f64, f64, f64);
        let mut table: Vec<Vec<Spot>> = Vec::new();
        for (&dy, id) in heights.iter().zip(&ids) {
            let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
            if d.hits == 0 {
                eprintln!("FOCUS   plane {dy:+.0}: no hits (patch {:+.1} along, y {:+.1})", table_uw(p, centre).0, base_y + dy);
            }
            let nv = 2 * ns + 4;
            let dv = 2.0 * hh / nv as f64;
            // The rows the light reaches, split into `ns` slices.
            let row_flux: Vec<f64> = (0..nv).map(|j| d.incident_flux[j * nu..(j + 1) * nu].iter().sum()).collect();
            let top = row_flux.iter().cloned().fold(0.0, f64::max);
            let lit: Vec<usize> = (0..nv).filter(|&j| row_flux[j] > 0.3 * top).collect();
            let mut per = Vec::new();
            for chunk in lit.chunks(lit.len().div_ceil(ns).max(1)) {
                let mut prof = vec![0.0; nu];
                for &j in chunk {
                    for (a, b) in prof.iter_mut().zip(&d.incident_flux[j * nu..(j + 1) * nu]) {
                        *a += b;
                    }
                }
                let xs = |i: usize| -hw + (i as f64 + 0.5) * du;
                let w: f64 = prof.iter().sum();
                if w <= 0.0 {
                    continue;
                }
                let c = prof.iter().enumerate().map(|(i, f)| f * xs(i)).sum::<f64>() / w;
                let rms = (prof.iter().enumerate().map(|(i, f)| f * (xs(i) - c).powi(2)).sum::<f64>() / w).sqrt();
                let m = prof.iter().cloned().fold(0.0, f64::max);
                let above: Vec<usize> = (0..nu).filter(|&i| prof[i] >= 0.5 * m).collect();
                let fw = (above[above.len() - 1] - above[0] + 1) as f64 * du;
                // v runs along -Z: the slice's height across the band, +Z up.
                let z = -(chunk.iter().map(|&j| -hh + (j as f64 + 0.5) * dv).sum::<f64>() / chunk.len() as f64);
                per.push((z, c, rms, fw, w));
                out.push_str(&format!("{:.1},{dy:.0},{},{z:.1},{c:.4},{rms:.4},{fw:.3},{w:.5e}\n", l * 1e3, per.len() - 1));
            }
            table.push(per);
        }
        let at0 = heights.iter().position(|&h| h == 0.0).unwrap();
        eprintln!("FOCUS {:.0} nm lands at x {:+.1}: per slice z, centroid, rms, FWHM at the table; best-focus height and rms there", l * 1e3, table_uw(p, centre).0);
        let c0 = table[at0].iter().map(|q| q.1).sum::<f64>() / table[at0].len().max(1) as f64;
        for (k, q) in table[at0].iter().enumerate() {
            // A parabola through the least rms and its neighbours.
            let r: Vec<f64> = table.iter().map(|t| t.get(k).map_or(f64::INFINITY, |x| x.2)).collect();
            let i = (0..r.len()).min_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
            let (best, rbest) = if i > 0 && i + 1 < r.len() {
                let (a, b, c) = (r[i - 1], r[i], r[i + 1]);
                let den = a - 2.0 * b + c;
                let t = if den > 0.0 { 0.5 * (a - c) / den } else { 0.0 };
                (heights[i] + t * step, b - 0.25 * (a - c) * t)
            } else {
                (heights[i], r[i])
            };
            eprintln!("FOCUS   z {:+6.1}: centroid {:+6.2}  rms {:.2}  FWHM {:.2}  | best at dy {:+5.1}, rms {:.2}", q.0, q.1 - c0, q.2, q.3, best, rbest);
        }
    }
    let path = std::env::var("SCAN_OUT").unwrap_or_else(|_| "focus-scan.csv".into());
    std::fs::write(&path, out).expect("write the focus scan");
    eprintln!("wrote {path}");
}

fn run_bench(slit: Vec3, rows: &[[f64; 2]], gain: &[(f64, f64)]) {
    // The bench draws nothing: no segments. Kept, they cost about 120 bytes
    // each, a dozen per ray, tens of gigabytes at these ray counts.
    let cfg = |s: &Scene3| TraceConfig3 {
        ghosts: false,
        max_depth: depth_for_scene3(s, false, 0).max(64),
        bounds_mm: bounds_for_scene3(s),
        segment_floor: f64::INFINITY,
        brute_force_search: std::env::var_os("BRUTE").is_some(),
        ..TraceConfig3::default()
    };
    let mut sc0 = Scene3::new();
    let sp = spectrograph::add(&mut sc0, slit);
    let d = sp.design;
    eprintln!(
        "spectrograph: pick-off {:.0} x {:.1} mm lighting a {:.1} x {} mm slit ({:.3} nm) at f/{:.1}; collimator f {:.0}, {:.0} mm beam; light onto the grating at {:.1} deg, {:.0} nm leaves at {:.1}, {:.0} at {:.1}; camera mirror f {:.1} (f/{:.1}), convex fold R {:.1} sending the light {:.1} deg off vertical, focus flat to {:.2} mm",
        sp.pickoff_size.0, sp.pickoff_size.1, d.slit_height, spectrograph::slit_width(), d.resolution_nm, spectrograph::F_COLL / spectrograph::BEAM,
        spectrograph::F_COLL, spectrograph::BEAM, d.alpha_deg, spectrograph::BLUE_UM * 1e3, d.beta_blue_deg, spectrograph::RED_UM * 1e3, d.beta_red_deg,
        d.f_camera, d.camera_f_number, d.fold_radius, d.fold_lean_deg, d.focus_pv
    );
    eprintln!(
        "  field lens f {:.1} (Fresnel pair, radius x{:.3}, conic {:.3}); the lens takes f/{:.2} at the intermediate spectrum, the camera gives f/{:.2}",
        d.f_field, d.fresnel_scale, d.fresnel_conic, d.lens_takes_f_number, d.camera_f_number
    );
    eprintln!(
        "  projection lens: {}, f {:.1} mm, {:.0} mm long, {:.0} mm across, f/{:.2} at the diaphragm, horizontal; first vertex {:.1} mm past the intermediate spectrum; x{:.2} from {:.1} mm over a {:.0} mm throw; traced focus {:+.2} mm from paraxial",
        spectrograph::PROJ_NAME,
        d.f_proj, d.lens_length, 2.0 * d.lens_semi, d.proj_f_number, d.vertex_distance + d.focus_shift, d.magnification, d.object_distance, d.throw, d.focus_shift
    );
    // Each optic's nearest and farthest point from the table, nearest
    // first: in the room, height above the table (the model may be upside
    // down, `TABLE_UP`). The tube's axis and the ceiling are given the same
    // way.
    let axis_h = (Vec3::new(0.0, spectrograph::AXIS_Y, 0.0) - sp.table_centre).dot(spectrograph::table_normal());
    // The ceiling lies beyond the tube, as far past its axis as in the room.
    let ceiling_h = axis_h + (spectrograph::CEILING_Y - spectrograph::AXIS_Y);
    eprintln!("  from the table: the tube's axis {axis_h:.0} mm, the ceiling {ceiling_h:.0} mm");
    for (what, near, far) in spectrograph::from_table(&sc0, sp.table_centre).iter().take(env_f64("CLEAR_N", 8.0) as usize) {
        eprintln!("  CLEAR {near:6.0} .. {far:6.0} mm from the table: {what}");
    }
    for ((name, lo, hi), bow) in sp.bands.iter().skip(1).zip(&sp.lamp_slit_bows) {
        eprintln!("  {name}: band {lo:+.1} to {hi:+.1} mm across the table, its slit bowed {bow:.2} mm");
    }
    eprintln!("  sun: band {:+.1} to {:+.1} mm", sp.bands[0].1, sp.bands[0].2);
    if std::env::var_os("DESIGN_ONLY").is_some() {
        return;
    }
    if std::env::var_os("NA_ZOOM").is_some() {
        na_zoom(slit, rows, gain, &cfg);
        return;
    }
    if std::env::var_os("LINE_ZOOM").is_some() {
        line_zoom(slit, rows, gain, &cfg);
        return;
    }
    if std::env::var_os("FOCUS_SCAN").is_some() {
        focus_scan(slit, rows, gain, &cfg);
        return;
    }
    for l in spectrograph::lamps() {
        eprintln!("  {}: {:.3e} cd/m2, slit {} mm", l.name, l.luminance(), l.slit_width);
    }

    // The whole visible, for the picture, the light and the scale.
    let n = env_f64("BENCH", 1_048_576.0) as u32;
    let (lo_um, hi_um) = (spectrograph::BLUE_UM - 0.005, spectrograph::RED_UM + 0.005);
    let (mut s, spec) = bench(slit, rows, gain, lo_um, hi_um, n);
    if std::env::var_os("LOSSES").is_some() {
        // `LOSS_NM="lo hi"` narrows it to a few wavelengths.
        let w: Vec<f64> = std::env::var("LOSS_NM").map(|v| v.split_whitespace().filter_map(|x| x.parse::<f64>().ok()).map(|x| x / 1e3).collect()).unwrap_or_default();
        let (a, b) = if let [a, b] = w[..] { (a, b) } else { (lo_um, hi_um) };
        // `LOSS_SUN=1`: the sun's rays alone, the lamps' arms still there.
        let (mut sl, spl) = bench(slit, rows, gain, a, b, 20_000);
        if std::env::var_os("LOSS_SUN").is_some() {
            sl.sources.retain(|x| !spl.lamp_sources.contains(&x.id));
        }
        losses(&sl);
        if std::env::var_os("LOSSES_ONLY").is_some() {
            return;
        }
    }
    // Where each landmark falls, from the slit's centre.
    let marks = [
        ("375", 0.375),
        ("Fraunhofer L 382.0", 0.38204),
        ("Ca II K 393.4", 0.39337),
        ("Ca II H 396.8", 0.39685),
        ("H-beta 486.1", 0.48613),
        ("Mg b 517.3", 0.51727),
        ("Na D 589.3", 0.58929),
        ("H-alpha 656.3", 0.65628),
        ("Fraunhofer B 686.7", 0.68672),
        ("690", 0.690),
    ];
    let centre = spec.table_centre;
    let mut x_d = 0.0;
    for (name, l) in marks {
        if let Some(p) = landing(&s, spec.pickoff, spec.table, l) {
            let (a, w) = table_uw(p, centre);
            eprintln!("  {name:>14} nm lands at {a:+8.1} mm along, {w:+6.1} mm across");
            if name.starts_with("Na") {
                x_d = a;
            }
        }
    }
    // A fine strip of pixels over the sodium lines: 0.25 mm across.
    let fine = spectrograph::detector(
        &mut s,
        on_table(centre, x_d, centre.y + spectrograph::table_normal().y * 0.3),
        spectrograph::table_normal(),
        spectrograph::table_axes().0,
        30.0,
        100.0,
        [240, 20],
        DetectorTermination::ProbeAndContinue,
        "Table: sodium D lines, 0.25 mm pixels",
    );
    // Strips of 0.25 mm pixels over the mercury lamp's lines, across the
    // middle half of its band: each line's peak in lux, and where it stands
    // against where the same wavelength from the sun's slit lands.
    let (_, lo, hi) = spec.bands.get(1).copied().unwrap_or(("", 0.0, 0.0));
    let hg_all: &[(&str, f64)] = if std::env::var("EMISSION").is_ok_and(|v| v == "hg") {
        &[("Hg 404.7", 0.404_656_5), ("Hg 435.8", 0.435_833_5), ("Hg 546.1", 0.546_075), ("Hg 577.0 + 579.1", 0.578_014)]
    } else {
        &[("Na D2 589.0 + D1 589.6", 0.589_29)]
    };
    let hg_lines: &[(&str, f64)] = if spec.bands.len() > 1 && std::env::var_os("NO_STRIPS").is_none() { hg_all } else { &[] };
    let hg_strips: Vec<(&str, f64, f64, ObjectId)> = hg_lines
        .iter()
        .filter_map(|&(name, l)| {
            let x = table_uw(landing(&s, spec.pickoff, spec.table, l)?, centre).0;
            let id = spectrograph::detector(
                &mut s,
                on_table(centre, x, centre.y + spectrograph::table_normal().y * 0.3) + spectrograph::table_normal().cross(spectrograph::table_axes().0) * (0.5 * (lo + hi)),
                spectrograph::table_normal(),
                spectrograph::table_axes().0,
                30.0,
                0.25 * (hi - lo),
                [240, 1],
                DetectorTermination::ProbeAndContinue,
                &format!("Table: {name} nm, 0.25 mm pixels"),
            );
            Some((name, l, x, id))
        })
        .collect();
    // The file draws 4096 rays, few enough to see the optics under them;
    // the component view's analysis rays trace the numbers with more.
    let mut file = s.clone();
    for src in &mut file.sources {
        src.samples.spatial_samples = 2048;
    }
    let bench_out = std::env::var("BENCH_OUT").unwrap_or_else(|_| "spectrograph-bench.ocanvas".into());
    std::fs::write(&bench_out, file.to_json()).expect("write the bench scene");
    eprintln!("wrote {bench_out}");
    let t0 = std::time::Instant::now();
    let res = trace_passes(&s, &cfg(&s));
    let get = |id: ObjectId| res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
    let (table, entrance) = (get(spec.table), get(spec.entrance));
    let lumens: f64 = table.spectral.iter().map(|l| 683.0 * photopic(l.mean_lambda_um()) * l.flux).sum();
    let area = spectrograph::SPECTRUM_LENGTH * 1e-3 * 0.18;
    let (mut w, mut vv, mut v) = (0.0, 0.0, 0.0);
    for l in &table.spectral {
        w += l.flux;
        v += l.v;
        vv += l.vv;
    }
    let rms_v = (vv / w - (v / w).powi(2)).max(0.0).sqrt();
    eprintln!(
        "bench, {:.0}-{:.0} nm, {} rays in {:.1?}: {:.3} W enters, {:.3} W on the table, {:.1} lm, {:.0} lux over {:.2} x 0.18 m; spectrum {:.0} mm high",
        lo_um * 1e3,
        hi_um * 1e3,
        s.sources.iter().map(|x| x.samples.spatial_samples as u64 * x.samples.angular_samples as u64).sum::<u64>(),
        t0.elapsed(),
        entrance.total_incident,
        table.total_incident,
        lumens,
        lumens / area,
        spectrograph::SPECTRUM_LENGTH / 1e3,
        rms_v * 12f64.sqrt()
    );
    if std::env::var_os("ENTRANCE_PROFILE").is_some() {
        let (nu, nv) = (entrance.bins[0], entrance.bins[1]);
        let rows: Vec<String> = (0..nv).map(|j| format!("{:.3}", (0..nu).map(|i| entrance.incident_flux[j * nu + i]).sum::<f64>())).collect();
        eprintln!("ENTRANCE along the slit ({nv} bins): {}", rows.join(" "));
    }
    // Brightest 20 nm on the table, in lux.
    let mut best = (0.0, 0.0);
    for k in 19..35 {
        let (lo, hi) = (k as f64 * 0.02, (k + 1) as f64 * 0.02);
        let lm: f64 = table
            .spectral
            .iter()
            .filter(|l| l.lambda_um >= lo && l.lambda_um < hi)
            .map(|l| 683.0 * photopic(l.mean_lambda_um()) * l.flux)
            .sum();
        if lm > best.1 {
            best = (lo, lm);
        }
    }
    // The strip that band covers on the table: its share of 2.5 m.
    let strip_m = spectrograph::SPECTRUM_LENGTH * 1e-3 * 0.02 / (spectrograph::RED_UM - spectrograph::BLUE_UM);
    eprintln!(
        "  brightest 20 nm, {:.0}-{:.0} nm: {:.0} lux",
        best.0 * 1e3,
        best.0 * 1e3 + 20.0,
        best.1 / (strip_m * 0.18)
    );
    let _ = fine;
    band_lux(table, &spec.bands);
    for (name, l, _, id) in &hg_strips {
        let d = get(*id);
        let col_m2 = 0.25e-3 * 0.5 * (hi - lo) * 1e-3;
        let lux: Vec<f64> = d.incident_flux.iter().map(|f| f / col_m2 * 683.0 * photopic(*l)).collect();
        let top = lux.iter().cloned().fold(0.0, f64::max);
        // Local peaks above a third of the brightest, at their pixel centres.
        let peaks: Vec<String> = (1..lux.len() - 1)
            .filter(|&i| lux[i] >= top / 3.0 && lux[i] >= lux[i - 1] && lux[i] > lux[i + 1])
            .map(|i| format!("{:.1} lux at {:+.2} mm", lux[i], (i as f64 + 0.5) * 0.25 - 30.0))
            .collect();
        eprintln!("  {name} nm, {} hits: {}", d.hits, peaks.join(", "));
        // `STRIP_OUT=<prefix>`: the strip's lux per 0.25 mm pixel.
        if let Ok(prefix) = std::env::var("STRIP_OUT") {
            let rows: Vec<String> = lux.iter().enumerate().map(|(i, v)| format!("{:.3},{v:.4}", (i as f64 + 0.5) * 0.25 - 30.0)).collect();
            let _ = std::fs::write(format!("{prefix}{:.2}.csv", l * 1e3), format!("# mm,lux\n{}\n", rows.join("\n")));
        }
    }

    // The table as a visitor sees it: every pixel's colour, with more rays
    // than the numbers above need, for a picture of 312 thousand pixels.
    // `TABLE_TILE=<nm>`: the spectrum in windows that wide, each traced with
    // `TABLE_RAYS` of its own, their pictures added up: a window's light
    // lands only on its own stretch of the table, so every stretch gets the
    // rays the whole table got in one go.
    let n_img = env_f64("TABLE_RAYS", 8_388_608.0) as u32;
    let tile = env_f64("TABLE_TILE", 0.0) * 1e-3;
    let windows: Vec<(f64, f64)> = if tile > 0.0 {
        let k = ((hi_um - lo_um) / tile).ceil() as usize;
        (0..k).map(|i| (lo_um + i as f64 * tile, (lo_um + (i + 1) as f64 * tile).min(hi_um))).collect()
    } else {
        vec![(lo_um, hi_um)]
    };
    // With windows, each also lights a detector of its own over just its
    // stretch of the table (`TILE_MARGIN` mm either side, default 35: the
    // halogen's 23 mm slit spreads a wavelength over about 40 mm), with
    // `TILE_PX_PER_MM` pixels along (default 2), 2 mm across as the table;
    // pasted together they make `{TABLE_OUT}.tiles.f32`, finer than the
    // table's own detector may be.
    let per_mm = env_f64("TILE_PX_PER_MM", 2.0);
    let margin = env_f64("TILE_MARGIN", 35.0);
    let half_len = spectrograph::TABLE_HALF_LENGTH;
    let nu_all = (2.0 * half_len * per_mm).round() as usize;
    let mut mosaic: Vec<[f32; 3]> = Vec::new();
    let mut mosaic_rows = 0usize;
    let mut picture: Option<(Vec<[f64; 3]>, [usize; 2])> = None;
    let t_img = std::time::Instant::now();
    for (i, &(a, b)) in windows.iter().enumerate() {
        let (mut s_img, spec_img) = bench(slit, rows, gain, a, b, n_img);
        let mut tile_at = None;
        if windows.len() > 1 {
            // The sun in `TILE_SUN_PIECES` pieces (default 32, 0 for the one
            // rectangle): about ten times the rays on the table.
            let k = env_f64("TILE_SUN_PIECES", 32.0) as usize;
            let sun = s_img.sources.iter().find(|x| !spec_img.lamp_sources.contains(&x.id)).map(|x| x.id);
            if let (true, Some(sun), false) = (k > 0, sun, std::env::var_os("SUN_PIECES").is_some()) {
                sun_pieces(&mut s_img, &spec_img, sun, k);
            }
            // The lamps at `TILE_LAMP_RAYS` times the sun's rays (default 2)
            // rather than the numbers' `LAMP_RAYS`: their bands carry no lines.
            let f = env_f64("TILE_LAMP_RAYS", 2.0) / env_f64("LAMP_RAYS", 16.0);
            for src in s_img.sources.iter_mut().filter(|x| spec_img.lamp_sources.contains(&x.id)) {
                src.samples.spatial_samples = ((src.samples.spatial_samples as f64 * f) as u32).max(1);
            }
            let c = spec_img.table_centre;
            let x_at = |l: f64| landing(&s_img, spec_img.pickoff, spec_img.table, l).map(|p| table_uw(p, c).0);
            let (Some(xa), Some(xb)) = (x_at(a), x_at(b)) else { panic!("no landing for {a}..{b}") };
            let hh = match s_img.detectors.iter().find(|d| d.common.id.0 == spec_img.table.0).map(|d| &d.geometry) {
                Some(DetectorGeometry3::Rectangle { half_height_mm, .. }) => *half_height_mm,
                _ => panic!("the table is a rectangle"),
            };
            let nv = hh as usize;
            let i0 = (((xa.min(xb) - margin + half_len) * per_mm).floor().max(0.0)) as usize;
            let i1 = (((xa.max(xb) + margin + half_len) * per_mm).ceil() as usize).min(nu_all);
            let hw = 0.5 * (i1 - i0) as f64 / per_mm;
            let xc = -half_len + 0.5 * (i0 + i1) as f64 / per_mm;
            let id = spectrograph::detector(
                &mut s_img,
                on_table(c, xc, c.y + spectrograph::table_normal().y * 0.3),
                spectrograph::table_normal(),
                spectrograph::table_axes().0,
                hw,
                hh,
                [i1 - i0, nv],
                DetectorTermination::ProbeAndContinue,
                "Table: one window's stretch",
            );
            if let Some(det) = s_img.detectors.iter_mut().find(|t| t.common.id.0 == id.0) {
                det.measure = oc_core::scene3::DetectorMeasure::Illuminance;
            }
            if mosaic.is_empty() {
                mosaic = vec![[0.0; 3]; nu_all * nv];
                mosaic_rows = nv;
            }
            tile_at = Some((id, i0));
        }
        let res = trace_passes(&s_img, &cfg(&s_img));
        if let Some((id, i0)) = tile_at {
            let d = res.detectors.iter().find(|d| d.detector.0 == id.0).unwrap();
            if let Some(colour) = d.colour_per_m2() {
                let nu = d.bins[0];
                for j in 0..d.bins[1].min(mosaic_rows) {
                    for k in 0..nu {
                        let (m, c) = (&mut mosaic[j * nu_all + i0 + k], colour[j * nu + k]);
                        (0..3).for_each(|q| m[q] += c[q] as f32);
                    }
                }
            }
        }
        let t = res.detectors.iter().find(|d| d.detector.0 == spec_img.table.0).unwrap();
        let Some(colour) = t.colour_per_m2() else { continue };
        match &mut picture {
            None => picture = Some((colour, t.bins)),
            Some((sum, _)) => sum.iter_mut().zip(&colour).for_each(|(p, c)| (0..3).for_each(|k| p[k] += c[k])),
        }
        if windows.len() > 1 {
            eprintln!("  table window {}/{} {:.0}-{:.0} nm: {} hits, {:.1?} so far", i + 1, windows.len(), a * 1e3, b * 1e3, t.hits, t_img.elapsed());
        }
    }
    if let Some((colour, bins)) = picture {
        let mut csv = format!("# {} x {} pixels, u fastest, v upward; X,Y,Z in lux
", bins[0], bins[1]);
        for c in &colour {
            csv.push_str(&format!("{:.6e},{:.6e},{:.6e}
", c[0], c[1], c[2]));
        }
        let table_out = std::env::var("TABLE_OUT").unwrap_or_else(|_| "spectrograph-table-xyz.csv".into());
        std::fs::write(&table_out, csv).expect("write the table's colours");
        if !mosaic.is_empty() {
            let bytes: Vec<u8> = mosaic.iter().flat_map(|c| c.iter().flat_map(|x| x.to_le_bytes())).collect();
            std::fs::write(format!("{table_out}.tiles.f32"), bytes).expect("write the tiles");
            eprintln!("  tiles: {nu_all} x {mosaic_rows} pixels of {} x 2 mm, f32 X,Y,Z lux, u fastest; wrote {table_out}.tiles.f32", 1.0 / per_mm);
        }
        let peak = colour.iter().map(|c| c[1]).fold(0.0, f64::max);
        eprintln!("  table picture from {} windows of {} samples: {} pixels, brightest {:.0} lux; wrote {table_out}", windows.len(), n_img, colour.len(), peak);
    }

    // The sodium lines with many rays: 586-593 nm only.
    let (mut s, spec) = bench(slit, rows, gain, 0.586, 0.593, n);
    let fine = spectrograph::detector(
        &mut s,
        on_table(centre, x_d, centre.y + spectrograph::table_normal().y * 0.3),
        spectrograph::table_normal(),
        spectrograph::table_axes().0,
        30.0,
        100.0,
        [240, 20],
        DetectorTermination::ProbeAndContinue,
        "Table: sodium D lines, 0.25 mm pixels",
    );
    let res = trace_passes(&s, &cfg(&s));
    let d = res.detectors.iter().find(|d| d.detector.0 == fine.0).unwrap();
    let _ = spec;
    // Across the dispersion, summed over the middle half of the height.
    let (nu, nv) = (240usize, 20usize);
    let mut profile = vec![0.0f64; nu];
    for iv in nv / 4..3 * nv / 4 {
        for (p, v) in profile.iter_mut().zip(&d.incident_flux[iv * nu..(iv + 1) * nu]) {
            *p += v;
        }
    }
    let mut sorted = profile.clone();
    sorted.sort_by(f64::total_cmp);
    let cont = sorted[(0.9 * nu as f64) as usize];
    let mut csv = String::from("x_mm,relative\n");
    for (i, p) in profile.iter().enumerate() {
        csv.push_str(&format!("{:.3},{:.4}\n", (i as f64 + 0.5) * 0.25 - 30.0, p / cont));
    }
    let profile_out = std::env::var("PROFILE_OUT").unwrap_or_else(|_| "sodium-d-profile.csv".into());
    std::fs::write(&profile_out, csv).expect("write the sodium profile");
    // The deepest point within 6 mm either side of the doublet's centre:
    // D2 on the blue side, D1 on the red.
    let smooth: Vec<f64> = (0..nu)
        .map(|i| {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(nu - 1));
            profile[a..=b].iter().sum::<f64>() / (b - a + 1) as f64
        })
        .collect();
    let min_in = |r: std::ops::Range<usize>| r.min_by(|&a, &b| smooth[a].total_cmp(&smooth[b])).unwrap();
    let (m1, m2) = (min_in(nu / 2 - 24..nu / 2), min_in(nu / 2..nu / 2 + 24));
    let between = smooth[m1..=m2].iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "  sodium D, {} hits on the strip: dips {:.0}% and {:.0}% deep, {:.2} mm apart, the continuum between them back to {:.0}%",
        d.hits,
        100.0 * (1.0 - smooth[m1] / cont),
        100.0 * (1.0 - smooth[m2] / cont),
        (m2 - m1) as f64 * 0.25,
        100.0 * between / cont
    );
}
