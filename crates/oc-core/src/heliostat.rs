//! The sun, and mirrors that follow it.
//!
//! A heliostat is a mirror turned so that sunlight leaving it goes to a fixed
//! target. Where the sun is follows from where and when ([`crate::solar`]),
//! and the mirror's normal follows from the sun and the target: it bisects the
//! direction to the sun and the direction to the target, which is the law of
//! reflection solved for the mirror instead of the ray.
//!
//! None of this reaches the tracer. A [`SolarRig`] is authored intent, like a
//! stop designation: [`Scene3::resolve_solar`] turns it into an ordinary
//! source and ordinary mirror orientations, and the tracer, the preview and
//! the exporters see nothing but those. A saved document carries both, so a
//! reader that knows nothing of rigs still traces the scene as it was last
//! resolved.
//!
//! # Frames
//!
//! The canvas is a vertical section: world `+Y` is up. Which way the ground
//! runs is [`SolarRig::x_bearing_deg`], the compass bearing of world `+X`, and
//! world `+Z` is then 90 degrees clockwise of it, which is what a right-handed
//! frame with `+Y` up demands. With the default bearing of 180 the canvas is
//! the north-south plane seen from the west, south to the right: the plane the
//! noon sun stands in, north of the tropics.
//!
//! A model built the other way up, the room turned over about world `+X`
//! (a heliostat on the roof drawn below the tube it feeds), sets
//! [`SolarRig::upside_down`]: room up is then world `-Y`, world `+Z` is 90
//! degrees anticlockwise of `+X`, and the sun comes from the `-Y` side. The
//! turn is a rotation, not a mirror image, so the sun's east-west sense is
//! kept.
//!
//! # Units
//!
//! The sun is the one calibrated source. Its flux is the direct normal
//! irradiance times the area of its beam times the share of the spectrum it
//! carries, in watts, so what a detector reads under it is watts too.

use crate::id::{ObjectId, SourceId};
use crate::scene::Spectrum;
use crate::scene3::{
    AngularDistribution3, PlanarShape2, SamplingSequence, Scene3, SceneError, Source3,
    SourceSampling, SpatialDistribution3, SpectralScalar, SurfaceGeometry3, SurfaceObject3,
    SurfaceResponse,
};
use crate::solar::{self, SolarSite, UtcInstant};
use crate::vec3::{Isometry3, Vec3};
use serde::{Deserialize, Serialize};

/// A local civil time: what a clock on the site reads, and how far that clock
/// is from UTC.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: f64,
    /// Hours ahead of UTC: `+1` for Central European winter time.
    pub utc_offset_h: f64,
}

impl LocalTime {
    fn instant(&self) -> UtcInstant {
        UtcInstant::from_local(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.utc_offset_h,
        )
    }

    fn is_valid(&self) -> bool {
        (1..=12).contains(&self.month)
            && (1..=31).contains(&self.day)
            && self.hour < 24
            && self.minute < 60
            && (0.0..60.0).contains(&self.second)
            && self.utc_offset_h.abs() <= 14.0
    }
}

/// Where on Earth the scene stands.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Site {
    pub latitude_deg: f64,
    /// East positive.
    pub longitude_deg: f64,
    pub elevation_m: f64,
    /// Annual means are good enough here: they move the refraction correction,
    /// which is a few arcminutes at the horizon and nothing at noon.
    pub pressure_mbar: f64,
    pub temperature_c: f64,
}

impl Site {
    fn solar_site(&self) -> SolarSite {
        SolarSite {
            latitude_deg: self.latitude_deg,
            longitude_deg: self.longitude_deg,
            elevation_m: self.elevation_m,
            pressure_mbar: self.pressure_mbar,
            temperature_c: self.temperature_c,
            ..SolarSite::default()
        }
    }
}

/// Where a heliostat sends the light.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HelioTarget {
    /// A fixed world point, mm.
    Point(Vec3),
    /// The origin of a placed object: a detector, the mouth of a tube. The
    /// heliostat follows it when it moves.
    Object(ObjectId),
}

/// One heliostat: which mirror follows the sun, and where it sends the light.
///
/// The mount is azimuth-elevation: the mirror's long edge (local `X`) stays
/// horizontal, which is what a pedestal heliostat does and what decides how
/// the reflected image of a rectangular mirror rotates over the day.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Heliostat {
    /// A standalone planar surface. Its position is the pivot.
    pub mirror: ObjectId,
    pub target: HelioTarget,
}

/// The sun, when, where, and the mirrors that follow it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolarRig {
    pub site: Site,
    pub time: LocalTime,
    /// Compass bearing of world `+X`, degrees clockwise from north.
    pub x_bearing_deg: f64,
    /// The scene is the room turned over about world `+X`: up is world
    /// `-Y`. See the module's frames.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub upside_down: bool,
    /// Direct normal irradiance, W/m². The ASTM G173 direct spectrum carries
    /// 900.1, a clear-sky noon at mid latitudes 800 to 1000.
    pub dni_w_m2: f64,
    /// The band the sun source emits, µm, and how many lines stand for it
    /// when the spectrum is sampled as lines.
    pub band_um: [f64; 2],
    pub bands: u32,
    /// Sample the spectrum continuously: every ray draws its own wavelength
    /// from the table at the table's resolution (a [`Spectrum::Continuum`]),
    /// instead of every ray going out once per line. False in a file written
    /// before the choice, which meant lines.
    #[serde(default)]
    pub continuous: bool,
    /// How bright the disc is across its face. Uniform in a file written
    /// before the choice.
    #[serde(default)]
    pub disc: solar::sunshape::SunDisc,
    /// Circumsolar ratio: the share of the direct light, within 43.6 mrad of
    /// the sun's centre, that comes from the aureole around the disc. 0 in a
    /// file written before the choice.
    #[serde(default)]
    pub csr: f64,
    /// [`Self::dni_w_m2`] is the disc alone, as a radiative-transfer model
    /// gives it (CAMS McClear, libRadtran: no circumsolar light), and the
    /// aureole comes on top: the source carries `DNI / (1 − csr)`. False, the
    /// default, reads the DNI as a pyrheliometer's, aureole included, which
    /// the disc and the aureole then share.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dni_disc_only: bool,
    /// A measured or modelled spectrum in place of ASTM G173, for a site and
    /// a day (SMARTS output, a spectroradiometer): rows `[nm, W/m²/nm]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spectrum: Option<SunSpectrum>,
    /// The source this rig drives. Its placement, spread, spectrum and flux
    /// are overwritten by every resolve; its sampling is left alone.
    pub source: SourceId,
    /// What the sun beam is centred on. `None` with heliostats in the rig
    /// covers all of them: the beam centres on the field and grows to reach
    /// every mirror at any orientation, and [`Self::beam_radius_mm`] and
    /// [`Self::standoff_mm`] are rewritten to say what it chose. `None` with no
    /// heliostats keeps the beam where it is and only turns it.
    pub aim_at: Option<ObjectId>,
    /// Radius of the sun beam's disc, mm: big enough to fill what it lights,
    /// and no bigger, or rays are spent on the ground.
    pub beam_radius_mm: f64,
    /// How far up-sun of what it lights the beam starts, mm. Over a field
    /// ([`Self::aim_at`] `None`) this is a floor: the beam starts at least
    /// one beam radius out.
    pub standoff_mm: f64,
    pub heliostats: Vec<Heliostat>,
}

/// A spectrum imported for the sun, and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SunSpectrum {
    /// What the user called it: the file's name.
    pub name: String,
    /// `[nm, W/m²/nm]`, wavelength increasing.
    pub rows: Vec<[f64; 2]>,
}

impl SolarRig {
    /// The spectral table the sun emits: the imported one, else ASTM G173-03
    /// direct + circumsolar.
    pub fn table(&self) -> &[[f64; 2]] {
        self.spectrum
            .as_ref()
            .map_or(solar::g173_direct(), |t| t.rows.as_slice())
    }

    pub(crate) fn validate(&self, scene: &Scene3) -> Result<(), SceneError> {
        let range = |field: &str, reason: &str| SceneError::OutOfRange {
            object: None,
            field: format!("solar.{field}"),
            reason: reason.to_string(),
        };
        let s = &self.site;
        let finite = [
            s.latitude_deg,
            s.longitude_deg,
            s.elevation_m,
            s.pressure_mbar,
            s.temperature_c,
            self.time.second,
            self.time.utc_offset_h,
            self.x_bearing_deg,
            self.dni_w_m2,
            self.band_um[0],
            self.band_um[1],
            self.beam_radius_mm,
            self.standoff_mm,
        ];
        if finite.iter().any(|v| !v.is_finite()) {
            return Err(SceneError::NonFinite {
                object: None,
                field: "solar".to_string(),
            });
        }
        if s.latitude_deg.abs() > 90.0 {
            return Err(range("site.latitude_deg", "is past a pole"));
        }
        if !self.time.is_valid() {
            return Err(range("time", "is not a clock reading"));
        }
        if self.dni_w_m2 < 0.0 {
            return Err(range("dni_w_m2", "is negative"));
        }
        if self.band_um[0] >= self.band_um[1] || self.bands == 0 {
            return Err(range("band_um", "is empty"));
        }
        if !self.csr.is_finite() || !(0.0..=solar::sunshape::MAX_CSR).contains(&self.csr) {
            return Err(range("csr", "is outside [0, 0.849]"));
        }
        if let Some(t) = &self.spectrum {
            let ok = t.rows.len() >= 2
                && t.rows.iter().all(|r| r[0].is_finite() && r[1].is_finite() && r[0] > 0.0 && r[1] >= 0.0)
                && t.rows.windows(2).all(|w| w[1][0] > w[0][0]);
            if !ok {
                return Err(range("spectrum", "is not a table of increasing wavelengths"));
            }
        }
        if self.beam_radius_mm <= 0.0 || self.standoff_mm < 0.0 {
            return Err(range("beam", "has no size or a negative standoff"));
        }
        if !scene.sources.iter().any(|x| x.id == self.source) {
            return Err(SceneError::DanglingRef {
                what: "solar source",
                id: self.source.0,
            });
        }
        let exists = |id: ObjectId| object_origin(scene, id).is_some();
        if let Some(a) = self.aim_at {
            if !exists(a) {
                return Err(SceneError::DanglingRef {
                    what: "solar aim",
                    id: a.0,
                });
            }
        }
        for h in &self.heliostats {
            let planar = scene.surfaces.iter().any(|x| {
                x.common.id == h.mirror && matches!(x.geometry, SurfaceGeometry3::Plane { .. })
            });
            if !planar {
                return Err(SceneError::DanglingRef {
                    what: "heliostat mirror",
                    id: h.mirror.0,
                });
            }
            if let HelioTarget::Object(t) = h.target {
                if !exists(t) {
                    return Err(SceneError::DanglingRef {
                        what: "heliostat target",
                        id: t.0,
                    });
                }
            }
        }
        Ok(())
    }

    /// World `+X`, `+Y`, `+Z` in East-North-Up.
    fn world_axes_enu(&self) -> [Vec3; 3] {
        let b = self.x_bearing_deg.to_radians();
        let x = Vec3::new(b.sin(), b.cos(), 0.0);
        let y = Vec3::new(0.0, 0.0, self.up_sign());
        [x, y, x.cross(y)]
    }

    /// `+1` when world `+Y` is up, `-1` when the scene is upside down.
    pub fn up_sign(&self) -> f64 {
        if self.upside_down {
            -1.0
        } else {
            1.0
        }
    }

    /// Up in the room, in world coordinates.
    pub fn up(&self) -> Vec3 {
        Vec3::Y * self.up_sign()
    }

    /// An East-North-Up vector in world coordinates.
    fn enu_to_world(&self, enu: Vec3) -> Vec3 {
        let [x, y, z] = self.world_axes_enu();
        Vec3::new(enu.dot(x), enu.dot(y), enu.dot(z))
    }
}

/// What a resolve found, for the inspector.
#[derive(Debug, Clone, PartialEq)]
pub struct SolarState {
    /// Degrees clockwise from north.
    pub azimuth_deg: f64,
    /// Apparent, refraction included.
    pub elevation_deg: f64,
    /// Unit vector from the scene toward the sun, world coordinates.
    pub sun_dir: Vec3,
    /// The sun's angular radius at this date, radians.
    pub half_angle_rad: f64,
    /// Watts the sun source carries.
    pub source_flux_w: f64,
    pub heliostats: Vec<HelioState>,
}

impl SolarState {
    /// Is the sun's centre above the horizon?
    pub fn is_up(&self) -> bool {
        self.elevation_deg > 0.0
    }
}

/// One heliostat after a resolve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelioState {
    pub mirror: ObjectId,
    /// The mirror normal it was turned to, world.
    pub normal: Vec3,
    /// Angle of incidence of the sun on the mirror, degrees. Its cosine is the
    /// cosine efficiency: the share of the mirror's area the sun sees.
    pub incidence_deg: f64,
    /// False when the sun and the target lie in opposite directions from the
    /// pivot, where no mirror orientation works and the last one is kept.
    pub tracking: bool,
}

/// The position of a placed object or source, by id.
fn object_origin(scene: &Scene3, id: ObjectId) -> Option<Vec3> {
    let n = id.0;
    scene
        .bodies
        .iter()
        .map(|b| &b.common)
        .chain(scene.surfaces.iter().map(|s| &s.common))
        .chain(scene.apertures.iter().map(|a| &a.common))
        .chain(scene.detectors.iter().map(|d| &d.common))
        .find(|c| c.id.0 == n)
        .map(|c| c.transform.pos)
}

/// A rotation whose columns are the given orthonormal axes.
fn frame(x: Vec3, y: Vec3, z: Vec3) -> [f64; 4] {
    let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(x, y, z)).normalize();
    [q.x, q.y, q.z, q.w]
}

/// The azimuth-elevation mount's frame for a mirror normal: `+Z` the normal,
/// `+X` horizontal. `None` for a normal straight up or down, where
/// horizontal is every direction and the previous orientation is kept.
fn az_el_frame(normal: Vec3, up: Vec3) -> Option<(Vec3, Vec3, Vec3)> {
    let x = up.cross(normal);
    if x.length() < 1e-9 {
        return None;
    }
    let x = x.normalize();
    Some((x, normal.cross(x), normal))
}

/// The circumsolar ratio a new sun starts with. Buie et al. (2003) found
/// clear-sky CSRs mostly between 0.01 and 0.1 in the measurements behind their
/// model; 0.05 is the middle of that, a clear day with some haze, and the
/// value concentrator studies commonly assume. Measure or look up the site's
/// for anything that depends on it.
pub const DEFAULT_CSR: f64 = 0.05;

impl Scene3 {
    /// Recompute everything a [`SolarRig`] decides: the sun source's
    /// placement, spread, spectrum and flux, and each heliostat's orientation.
    ///
    /// References to objects that no longer exist are dropped first: a
    /// heliostat whose mirror was deleted goes, and so does one whose target
    /// was, and a rig whose source was deleted goes altogether. That is the
    /// repair a delete needs, and the only one; anything else invalid leaves
    /// the scene as it was and returns `None`.
    pub fn resolve_solar(&mut self) -> Option<SolarState> {
        self.prune_solar();
        let rig = self.solar.clone()?;
        if rig.validate(self).is_err() {
            return None;
        }
        let pos = solar::solar_position(&rig.site.solar_site(), &rig.time.instant());
        let [e, n, u] = solar::sun_direction_enu(&pos);
        let sun = rig.enu_to_world(Vec3::new(e, n, u)).normalize();
        let half_angle = pos.half_angle_rad();

        let mut helios = Vec::with_capacity(rig.heliostats.len());
        for h in &rig.heliostats {
            let target = match h.target {
                HelioTarget::Point(p) => p,
                HelioTarget::Object(id) => object_origin(self, id)?,
            };
            let Some(m) = self.surfaces.iter_mut().find(|s| s.common.id == h.mirror) else {
                continue;
            };
            let pivot = m.common.transform.pos;
            let to_target = (target - pivot).normalize_or_zero();
            let bisector = sun + to_target;
            let axes = (to_target != Vec3::ZERO && bisector.length() > 1e-9)
                .then(|| bisector.normalize())
                .and_then(|n| az_el_frame(n, rig.up()));
            if let Some((x, y, z)) = axes {
                if let Ok(t) = Isometry3::new(pivot, frame(x, y, z)) {
                    m.common.transform = t;
                }
            }
            let normal = m.common.transform.rotate(Vec3::Z);
            helios.push(HelioState {
                mirror: h.mirror,
                normal,
                incidence_deg: sun.dot(normal).clamp(-1.0, 1.0).acos().to_degrees(),
                tracking: axes.is_some(),
            });
        }

        // The beam: a disc square to the sun, up-sun of what it lights,
        // emitting down-sun into the sun's own cone. Over a field the disc
        // starts at least one radius out, so it clears every mirror, and
        // farther when the rig asks: a sun drawn above a 5 m tower reads
        // right to someone looking at the layout.
        let (radius, standoff, centre) = match (rig.aim_at, self.field_cover(&rig)) {
            (None, Some((c, r))) => (r, rig.standoff_mm.max(r).max(200.0), Some(c)),
            (aim, _) => (
                rig.beam_radius_mm,
                rig.standoff_mm,
                aim.and_then(|id| object_origin(self, id)),
            ),
        };
        let centre = centre.map(|c| c + sun * standoff);
        if let Some(r) = self.solar.as_mut() {
            r.beam_radius_mm = radius;
            r.standoff_mm = standoff;
        }
        let table = rig.table();
        let [lo, hi] = rig.band_um;
        let spectrum = if rig.continuous {
            Spectrum::Continuum(solar::table_continuum(table, lo, hi))
        } else {
            Spectrum::Lines(solar::table_lines(table, rig.bands as usize, lo, hi))
        };
        // The DNI is the whole table's; the beam carries the band's share.
        let fraction = solar::table_fraction(table, lo, hi);
        let area_m2 = std::f64::consts::PI * radius * radius * 1e-6;
        // No light from a sun below the horizon, whatever the clock says.
        let up = pos.elevation_deg > 0.0;
        // A disc-only DNI leaves the aureole's share to be added.
        let aureole = if rig.dni_disc_only { 1.0 / (1.0 - rig.csr) } else { 1.0 };
        let flux = if up {
            rig.dni_w_m2 * aureole * fraction * area_m2
        } else {
            0.0
        };
        let src = self.sources.iter_mut().find(|s| s.id == rig.source)?;
        let q = glam::DQuat::from_rotation_arc(Vec3::X, -sun);
        let at = centre.unwrap_or(src.transform.pos);
        if let Ok(t) = Isometry3::new(at, [q.x, q.y, q.z, q.w]) {
            src.transform = t;
        }
        src.spatial = SpatialDistribution3::Disc { radius_mm: radius };
        // A uniform disc with no aureole is the cone it always was.
        src.angular = if rig.disc == solar::sunshape::SunDisc::Uniform && rig.csr == 0.0 {
            AngularDistribution3::UniformConeSolidAngle {
                half_angle_rad: half_angle,
            }
        } else {
            AngularDistribution3::Sun {
                half_angle_rad: half_angle,
                disc: rig.disc,
                csr: rig.csr,
            }
        };
        src.spectrum = spectrum;
        src.flux = flux;

        Some(SolarState {
            azimuth_deg: pos.azimuth_deg,
            elevation_deg: pos.elevation_deg,
            sun_dir: sun,
            half_angle_rad: half_angle,
            source_flux_w: flux,
            heliostats: helios,
        })
    }

    /// The disc that covers every heliostat, `(centre, radius)`: centred on
    /// the pivots' mean, reaching the farthest pivot plus that mirror's half
    /// diagonal, which covers it at any orientation. `None` without
    /// heliostats. A few percent of margin, because a disc sampled at its rim
    /// should not decide whether a mirror's corner is lit.
    fn field_cover(&self, rig: &SolarRig) -> Option<(Vec3, f64)> {
        let mirrors: Vec<(Vec3, f64)> = rig
            .heliostats
            .iter()
            .filter_map(|h| self.surfaces.iter().find(|s| s.common.id == h.mirror))
            .map(|m| {
                let reach = match &m.geometry {
                    SurfaceGeometry3::Plane {
                        shape:
                            PlanarShape2::Rectangle {
                                half_width_mm,
                                half_height_mm,
                            },
                    } => half_width_mm.hypot(*half_height_mm),
                    SurfaceGeometry3::Plane {
                        shape: PlanarShape2::Disc { radius_mm },
                    } => *radius_mm,
                    SurfaceGeometry3::Plane {
                        shape: PlanarShape2::Annulus { outer_mm, .. },
                    } => *outer_mm,
                    SurfaceGeometry3::Plane {
                        shape: PlanarShape2::Polygon { vertices },
                    } => vertices.iter().map(|v| v.length()).fold(0.0, f64::max),
                    SurfaceGeometry3::SurfaceOfRevolution {
                        radial_aperture, ..
                    } => radial_aperture.outer_radius_mm,
                };
                (m.common.transform.pos, reach)
            })
            .collect();
        if mirrors.is_empty() {
            return None;
        }
        let centre = mirrors.iter().map(|(p, _)| *p).sum::<Vec3>() / mirrors.len() as f64;
        let radius = mirrors
            .iter()
            .map(|(p, r)| (*p - centre).length() + r)
            .fold(0.0, f64::max);
        Some((centre, 1.05 * radius))
    }

    /// Drop the rig's references to objects that are gone. See
    /// [`Self::resolve_solar`].
    fn prune_solar(&mut self) {
        let Some(mut rig) = self.solar.take() else {
            return;
        };
        if !self.sources.iter().any(|s| s.id == rig.source) {
            return;
        }
        let live = |scene: &Scene3, id: ObjectId| object_origin(scene, id).is_some();
        rig.heliostats.retain(|h| {
            self.surfaces.iter().any(|s| s.common.id == h.mirror)
                && match h.target {
                    HelioTarget::Object(t) => live(self, t),
                    HelioTarget::Point(_) => true,
                }
        });
        if rig.aim_at.is_some_and(|a| !live(self, a)) {
            rig.aim_at = None;
        }
        self.solar = Some(rig);
    }

    /// Put the sun in the scene: a calibrated source and a rig to drive it,
    /// clear sky over `site` at `time`, the canvas's `+X` bearing
    /// `x_bearing_deg`. Replaces any rig already there (a scene has one sun)
    /// and returns the new source's id.
    pub fn add_sun(&mut self, site: Site, time: LocalTime, x_bearing_deg: f64) -> SourceId {
        if let Some(old) = self.solar.take() {
            self.sources.retain(|s| s.id != old.source);
        }
        let id = self.alloc_source_id();
        self.sources.push(Source3 {
            window: Default::default(),
            id,
            transform: Isometry3::IDENTITY,
            spatial: SpatialDistribution3::Disc { radius_mm: 100.0 },
            angular: AngularDistribution3::UniformConeSolidAngle {
                half_angle_rad: solar::SUN_HALF_ANGLE_RAD,
            },
            spectrum: Spectrum::Monochromatic(crate::material::LAMBDA_D),
            flux: 0.0,
            samples: SourceSampling {
                spatial_samples: 256,
                angular_samples: 4,
                seed: 0,
                sequence: SamplingSequence::Sobol,
            },
        });
        self.solar = Some(Box::new(SolarRig {
            site,
            time,
            x_bearing_deg,
            upside_down: false,
            dni_w_m2: 900.0,
            band_um: [0.3, 2.5],
            bands: 6,
            // A new sun is the measured one: every wavelength of the table,
            // a limb-darkened disc, and a clear day's aureole.
            continuous: true,
            disc: solar::sunshape::SunDisc::LimbDarkened,
            csr: DEFAULT_CSR,
            dni_disc_only: false,
            spectrum: None,
            source: id,
            aim_at: None,
            beam_radius_mm: 100.0,
            standoff_mm: 500.0,
            heliostats: Vec::new(),
        }));
        self.resolve_solar();
        id
    }

    /// Add a flat rectangular heliostat mirror pivoting at `pivot`, `width` ×
    /// `height` mm, silvered on its front at `reflectivity`, following the
    /// sun onto `target`. The first heliostat also becomes what the sun beam
    /// is aimed at, sized to cover it. `None` without a sun in the scene.
    pub fn add_heliostat(
        &mut self,
        pivot: Vec3,
        width_mm: f64,
        height_mm: f64,
        reflectivity: f64,
        target: HelioTarget,
    ) -> Option<ObjectId> {
        self.solar.as_ref()?;
        if !(pivot.is_finite() && width_mm > 0.0 && height_mm > 0.0) {
            return None;
        }
        let id = self.alloc_object_id();
        let mut common = crate::scene3::ObjectCommon::new(id, crate::id::LayerId(0));
        common.transform = Isometry3::from_pos(pivot);
        common.label = "Heliostat".to_string();
        self.surfaces.push(SurfaceObject3 {
            common,
            geometry: SurfaceGeometry3::Plane {
                shape: PlanarShape2::Rectangle {
                    half_width_mm: 0.5 * width_mm,
                    half_height_mm: 0.5 * height_mm,
                },
            },
            response_front: SurfaceResponse::Mirror {
                reflectivity: SpectralScalar::Constant(reflectivity.clamp(0.0, 1.0)),
            },
            response_back: SurfaceResponse::Absorber,
        });
        let rig = self.solar.as_mut()?;
        // The beam goes back to covering the field, whatever it was aimed at.
        rig.aim_at = None;
        rig.heliostats.push(Heliostat { mirror: id, target });
        self.resolve_solar();
        Some(id)
    }
}
