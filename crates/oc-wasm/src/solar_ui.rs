//! The editor's handle on tubes, the sun and heliostats.
//!
//! Thin: every decision is in `oc_core::tube` and `oc_core::heliostat`, and
//! each call here parses what the page sends, makes one edit, and bumps the
//! revision (which re-resolves the sun). Reads come back as JSON the page
//! parses.

use oc_core::heliostat::{HelioTarget, LocalTime, Site};
use oc_core::id::ObjectId;
use oc_core::tube::{BoreFinish, BoreMetal, TubeDims};
use oc_core::vec3::Vec3;
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::Canvas;

/// A bore finish from the card's select value and its reflectance field:
/// `black`, `mirror`, or a metal name with `-protected` for the overcoat.
fn bore_from(kind: &str, reflectivity: f64) -> Option<BoreFinish> {
    match kind {
        "black" => Some(BoreFinish::Black),
        "mirror" => Some(BoreFinish::Mirror { reflectivity }),
        other => {
            let (name, protected) = match other.strip_suffix("-protected") {
                Some(n) => (n, true),
                None => (other, false),
            };
            Some(BoreFinish::Metal {
                metal: BoreMetal::parse(name)?,
                protected,
            })
        }
    }
}

fn bore_json(b: &Option<BoreFinish>) -> serde_json::Value {
    match b {
        Some(BoreFinish::Black) => json!({ "kind": "black" }),
        Some(BoreFinish::Mirror { reflectivity }) => {
            json!({ "kind": "mirror", "reflectivity": reflectivity })
        }
        Some(BoreFinish::Metal { metal, protected }) => json!({
            "kind": if *protected { format!("{}-protected", metal.name()) } else { metal.name().to_string() },
        }),
        None => json!({ "kind": "custom" }),
    }
}

#[wasm_bindgen]
impl Canvas {
    // --- tubes ---------------------------------------------------------------

    /// Add a tube centred at `(x, y)` in the canvas plane, its axis along `+X`
    /// turned by `tilt_deg`. Diameters and length in mm; `bore` as
    /// [`bore_from`] reads it. Returns the new body id, or -1.
    #[wasm_bindgen(js_name = addTube)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_tube(
        &mut self,
        x: f64,
        y: f64,
        tilt_deg: f64,
        inner_diameter_mm: f64,
        outer_diameter_mm: f64,
        length_mm: f64,
        bore: &str,
        reflectivity: f64,
    ) -> f64 {
        let Some(finish) = bore_from(bore, reflectivity) else {
            return -1.0;
        };
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let dims = TubeDims {
            inner_diameter_mm,
            outer_diameter_mm,
            length_mm,
        };
        match scene.add_tube(Vec3::new(x, y, 0.0), tilt_deg, dims, finish) {
            Some(id) => {
                self.bump();
                id as f64
            }
            None => -1.0,
        }
    }

    /// A tube's `{inner, outer, length, bore: {kind, reflectivity?}}`, or
    /// `null` if `id` is not a tube.
    #[wasm_bindgen(js_name = tubeJSON)]
    pub fn tube_json(&self, id: f64) -> String {
        let info = self.scene3.as_ref().and_then(|s| s.tube_info(id as u64));
        match info {
            Some((d, bore)) => json!({
                "inner": d.inner_diameter_mm,
                "outer": d.outer_diameter_mm,
                "length": d.length_mm,
                "bore": bore_json(&bore),
            })
            .to_string(),
            None => "null".into(),
        }
    }

    /// Resize a tube. False, and nothing changed, if the numbers make no tube.
    #[wasm_bindgen(js_name = setTubeDims)]
    pub fn set_tube_dims(&mut self, id: f64, inner: f64, outer: f64, length: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let dims = TubeDims {
            inner_diameter_mm: inner,
            outer_diameter_mm: outer,
            length_mm: length,
        };
        let ok = scene.set_tube_dims(id as u64, dims);
        if ok {
            self.bump();
        }
        ok
    }

    /// Refinish a tube's bore.
    #[wasm_bindgen(js_name = setTubeBore)]
    pub fn set_tube_bore(&mut self, id: f64, bore: &str, reflectivity: f64) -> bool {
        let Some(finish) = bore_from(bore, reflectivity) else {
            return false;
        };
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let ok = scene.set_tube_bore(id as u64, finish);
        if ok {
            self.bump();
        }
        ok
    }

    // --- the sun ---------------------------------------------------------------

    /// Put the sun in the scene, replacing any sun already there. Local civil
    /// time on the site's clock, `utc_offset_h` hours ahead of UTC; the canvas's
    /// `+X` points at compass bearing `x_bearing_deg`. Returns the source id.
    #[wasm_bindgen(js_name = addSun)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_sun(
        &mut self,
        latitude_deg: f64,
        longitude_deg: f64,
        elevation_m: f64,
        year: i32,
        month: u32,
        day: u32,
        minutes_of_day: f64,
        utc_offset_h: f64,
        x_bearing_deg: f64,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let site = Site {
            latitude_deg,
            longitude_deg,
            elevation_m,
            pressure_mbar: 1013.25 * (-elevation_m / 8434.0).exp(),
            temperature_c: 15.0,
        };
        let time = clock(year, month, day, minutes_of_day, utc_offset_h);
        let id = scene.add_sun(site, time, x_bearing_deg);
        self.bump();
        id.0 as f64
    }

    /// Move the sun's clock. `minutes_of_day` is local, 0 to 1440, so a time
    /// slider sweeps the day without touching the date. False without a sun or
    /// for a date that is not one.
    #[wasm_bindgen(js_name = setSunTime)]
    pub fn set_sun_time(
        &mut self,
        year: i32,
        month: u32,
        day: u32,
        minutes_of_day: f64,
        utc_offset_h: f64,
    ) -> bool {
        let Some(rig) = self.scene3.as_mut().and_then(|s| s.solar.as_mut()) else {
            return false;
        };
        let before = rig.time;
        rig.time = clock(year, month, day, minutes_of_day, utc_offset_h);
        if self.scene3.as_ref().is_some_and(|s| s.validate().is_err()) {
            if let Some(r) = self.scene3.as_mut().and_then(|s| s.solar.as_mut()) {
                r.time = before;
            }
            return false;
        }
        self.bump();
        true
    }

    /// Say which way up the model is drawn: `true` when it is the room turned
    /// over about `+X` (up is world `-Y`), and the sun comes from below.
    #[wasm_bindgen(js_name = setSunUpsideDown)]
    pub fn set_sun_upside_down(&mut self, upside_down: bool) -> bool {
        self.edit_rig(|rig| rig.upside_down = upside_down)
    }

    /// Move the site and turn the canvas on the ground.
    #[wasm_bindgen(js_name = setSunSite)]
    pub fn set_sun_site(
        &mut self,
        latitude_deg: f64,
        longitude_deg: f64,
        elevation_m: f64,
        x_bearing_deg: f64,
    ) -> bool {
        self.edit_rig(|rig| {
            rig.site.latitude_deg = latitude_deg;
            rig.site.longitude_deg = longitude_deg;
            rig.site.elevation_m = elevation_m;
            rig.site.pressure_mbar = 1013.25 * (-elevation_m / 8434.0).exp();
            rig.x_bearing_deg = x_bearing_deg;
        })
    }

    /// The sky: direct normal irradiance, the band the sun emits (µm) and how
    /// many lines stand for it.
    #[wasm_bindgen(js_name = setSunSky)]
    pub fn set_sun_sky(&mut self, dni_w_m2: f64, lo_um: f64, hi_um: f64, bands: u32) -> bool {
        self.edit_rig(|rig| {
            rig.dni_w_m2 = dni_w_m2;
            rig.band_um = [lo_um, hi_um];
            rig.bands = bands;
        })
    }

    /// The sun as it looks: its disc (`"uniform"`, `"limb"` for Neckel &
    /// Labs's measured limb darkening, `"buie"`) and the circumsolar ratio,
    /// the share of the direct light in the aureole around it (0 to 0.849).
    /// False without a sun, for an unknown disc, or for a ratio out of range.
    #[wasm_bindgen(js_name = setSunShape)]
    pub fn set_sun_shape(&mut self, disc: &str, csr: f64) -> bool {
        let Some(disc) = oc_core::solar::sunshape::SunDisc::parse(disc) else {
            return false;
        };
        self.edit_rig(|rig| {
            rig.disc = disc;
            rig.csr = csr;
        })
    }

    /// How the sun's spectrum is sampled: `continuous`, every ray its own
    /// wavelength at the table's resolution, or `bands` lines.
    #[wasm_bindgen(js_name = setSunSpectral)]
    pub fn set_sun_spectral(&mut self, continuous: bool, bands: u32) -> bool {
        self.edit_rig(|rig| {
            rig.continuous = continuous;
            rig.bands = bands;
        })
    }

    /// Use a spectrum read from `text` (see
    /// [`oc_core::solar::parse_spectrum_table`]) in place of ASTM G173, named
    /// `name`, and take the DNI to be its total. Returns what was read, or
    /// `error: ` and why nothing changed.
    #[wasm_bindgen(js_name = importSunSpectrum)]
    pub fn import_sun_spectrum(&mut self, name: &str, text: &str) -> String {
        let parsed = match oc_core::solar::parse_spectrum_table(text) {
            Ok(p) => p,
            Err(e) => return format!("error: {e}"),
        };
        let rows = parsed.rows;
        let total = oc_core::solar::table_power(&rows, rows[0][0] / 1000.0, rows[rows.len() - 1][0] / 1000.0);
        let ok = self.edit_rig(|rig| {
            rig.spectrum = Some(oc_core::heliostat::SunSpectrum {
                name: name.to_string(),
                rows,
            });
            rig.dni_w_m2 = total;
        });
        if ok {
            parsed.summary
        } else {
            "error: there is no sun to give it to".into()
        }
    }

    /// Go back to ASTM G173-03 and its 900 W/m².
    #[wasm_bindgen(js_name = clearSunSpectrum)]
    pub fn clear_sun_spectrum(&mut self) -> bool {
        self.edit_rig(|rig| {
            if rig.spectrum.take().is_some() {
                rig.dni_w_m2 = 900.0;
            }
        })
    }

    /// The sun beam: its radius, and how far up-sun it starts.
    #[wasm_bindgen(js_name = setSunBeam)]
    pub fn set_sun_beam(&mut self, radius_mm: f64, standoff_mm: f64) -> bool {
        self.edit_rig(|rig| {
            rig.beam_radius_mm = radius_mm;
            rig.standoff_mm = standoff_mm;
        })
    }

    /// Everything the sun panel shows: the rig as authored, where the sun is,
    /// what the source carries, and each heliostat's state. `null` without a
    /// sun.
    #[wasm_bindgen(js_name = solarJSON)]
    pub fn solar_json(&mut self) -> String {
        let Some(scene) = self.scene3.as_mut() else {
            return "null".into();
        };
        let Some(state) = scene.resolve_solar() else {
            return "null".into();
        };
        let Some(rig) = scene.solar.as_ref() else {
            return "null".into();
        };
        let t = rig.time;
        let helios: Vec<_> = state
            .heliostats
            .iter()
            .map(|h| {
                let target = rig
                    .heliostats
                    .iter()
                    .find(|x| x.mirror == h.mirror)
                    .map(|x| match x.target {
                        HelioTarget::Object(o) => json!({ "object": o.0 }),
                        HelioTarget::Point(p) => json!({ "point": [p.x, p.y, p.z] }),
                    });
                json!({
                    "mirror": h.mirror.0,
                    "target": target,
                    "incidence": h.incidence_deg,
                    "cosine": h.incidence_deg.to_radians().cos(),
                    "tracking": h.tracking,
                })
            })
            .collect();
        json!({
            "source": rig.source.0,
            "site": {
                "lat": rig.site.latitude_deg,
                "lon": rig.site.longitude_deg,
                "elevation": rig.site.elevation_m,
            },
            "time": {
                "year": t.year, "month": t.month, "day": t.day,
                "minutes": t.hour as f64 * 60.0 + t.minute as f64 + t.second / 60.0,
                "utcOffset": t.utc_offset_h,
            },
            "bearing": rig.x_bearing_deg,
            "upsideDown": rig.upside_down,
            "dni": rig.dni_w_m2,
            "band": rig.band_um,
            "bands": rig.bands,
            "continuous": rig.continuous,
            "disc": rig.disc.name(),
            "csr": rig.csr,
            "spectrum": rig.spectrum.as_ref().map(|t| json!({
                "name": t.name,
                "rows": t.rows.len(),
                "from": t.rows.first().map(|r| r[0]),
                "to": t.rows.last().map(|r| r[0]),
            })),
            "wavelengths": scene.sources.iter().find(|s| s.id == rig.source).map_or(0, |s| s.spectrum.samples().len()),
            "beamRadius": rig.beam_radius_mm,
            "standoff": rig.standoff_mm,
            "azimuth": state.azimuth_deg,
            "elevationAngle": state.elevation_deg,
            "up": state.is_up(),
            "sunDir": [state.sun_dir.x, state.sun_dir.y, state.sun_dir.z],
            "halfAngleMrad": state.half_angle_rad * 1e3,
            "fluxW": state.source_flux_w,
            "heliostats": helios,
        })
        .to_string()
    }

    // --- heliostats -----------------------------------------------------------

    /// Add a flat `width` × `height` mm heliostat pivoting at `(x, y, z)`,
    /// sending the sun to object `target_id`, or, when that is negative, to
    /// the world point `(tx, ty, tz)`. Adds a sun first if there is none (solar
    /// noon at the March equinox over Jena's Deutsches Optisches Museum, so it
    /// lights something straight away).
    /// Returns the mirror's id, or -1.
    #[wasm_bindgen(js_name = addHeliostat)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_heliostat(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        width_mm: f64,
        height_mm: f64,
        reflectivity: f64,
        target_id: f64,
        tx: f64,
        ty: f64,
        tz: f64,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        if scene.solar.is_none() {
            // Carl-Zeiss-Platz 12, Jena: 50°55'40.1" N, 11°34'45.4" E, 160 m.
            let site = Site {
                latitude_deg: 50.92781,
                longitude_deg: 11.57928,
                elevation_m: 160.0,
                pressure_mbar: 1013.25 * (-160.0_f64 / 8434.0).exp(),
                temperature_c: 10.0,
            };
            scene.add_sun(site, clock(2026, 3, 20, 12.0 * 60.0 + 21.0, 1.0), 180.0);
        }
        let target = if target_id >= 0.0 {
            HelioTarget::Object(ObjectId(target_id as u64))
        } else {
            HelioTarget::Point(Vec3::new(tx, ty, tz))
        };
        let id = scene.add_heliostat(
            Vec3::new(x, y, z),
            width_mm,
            height_mm,
            reflectivity,
            target,
        );
        match id {
            Some(id) if scene.validate().is_ok() => {
                self.bump();
                id.0 as f64
            }
            Some(id) => {
                // A target that does not exist: undo rather than keep a
                // heliostat the document cannot load.
                scene.delete_object(id.0);
                scene.resolve_solar();
                -1.0
            }
            None => -1.0,
        }
    }

    /// Point a heliostat at another object, or at a world point when
    /// `target_id` is negative.
    #[wasm_bindgen(js_name = setHeliostatTarget)]
    pub fn set_heliostat_target(
        &mut self,
        mirror: f64,
        target_id: f64,
        tx: f64,
        ty: f64,
        tz: f64,
    ) -> bool {
        let target = if target_id >= 0.0 {
            HelioTarget::Object(ObjectId(target_id as u64))
        } else {
            HelioTarget::Point(Vec3::new(tx, ty, tz))
        };
        let m = ObjectId(mirror as u64);
        self.edit_rig(|rig| {
            if let Some(h) = rig.heliostats.iter_mut().find(|h| h.mirror == m) {
                h.target = target;
            }
        })
    }

    /// Stop a mirror following the sun. It keeps the orientation it had.
    #[wasm_bindgen(js_name = releaseHeliostat)]
    pub fn release_heliostat(&mut self, mirror: f64) -> bool {
        let m = ObjectId(mirror as u64);
        self.edit_rig(|rig| rig.heliostats.retain(|h| h.mirror != m))
    }
}

impl Canvas {
    /// One edit to the rig, kept only if the scene still validates.
    fn edit_rig(&mut self, f: impl FnOnce(&mut oc_core::heliostat::SolarRig)) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let Some(rig) = scene.solar.as_mut() else {
            return false;
        };
        let before = rig.clone();
        f(rig);
        if scene.validate().is_err() {
            scene.solar = Some(before);
            return false;
        }
        self.bump();
        true
    }
}

/// A local clock reading from minutes past midnight.
fn clock(year: i32, month: u32, day: u32, minutes_of_day: f64, utc_offset_h: f64) -> LocalTime {
    let m = minutes_of_day.clamp(0.0, 1439.999);
    let hour = (m / 60.0).floor();
    let minute = (m - hour * 60.0).floor();
    LocalTime {
        year,
        month,
        day,
        hour: hour as u32,
        minute: minute as u32,
        second: ((m - hour * 60.0 - minute) * 60.0).clamp(0.0, 59.999),
        utc_offset_h,
    }
}
