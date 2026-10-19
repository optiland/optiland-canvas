//! The editor's handle on DMDs and LCoS SLMs (W5).
//!
//! Thin, like the other element files: `oc_core::modulator` is the physics
//! and `Scene3::add_modulator` the placement. Edits answer `{"ok":true}` or
//! `{"error":"..."}` and bump the revision only when they hold; reads come
//! back as JSON. A device's rectangle follows its pixel array, so an edit
//! that changes the count or the pitch resizes the surface too.

use oc_core::modulator::{
    local_unit_cell, on_axis_illumination, unit_cell_orders, DmdSpec, DmdState, LcosBeam, LcosSpec, ModulatorKind,
    OrderModel, Pattern, PhaseLut, RayModel, SpatialModulator, TiltAxis, TI_DMD_PRESETS,
};
use oc_core::scene3::{PlanarShape2, SurfaceGeometry3, SurfaceResponse};
use oc_core::vec3::Vec3;
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::{json_str, Canvas};

fn reply(r: Result<(), String>) -> String {
    match r {
        Ok(()) => "{\"ok\":true}".into(),
        Err(e) => format!("{{\"error\":{}}}", json_str(&e)),
    }
}

/// A default array about 10 mm across, whatever the pitch: big enough to see
/// and to catch a beam, small enough for the preview. The count is the
/// user's to set; no device resolution is assumed.
fn default_pixels(pitch_um: f64) -> [u32; 2] {
    let n = (10_000.0 / pitch_um).round().clamp(16.0, 4096.0) as u32;
    [n, (n * 3).div_ceil(4)]
}

/// A DMD with a preset's pitch, tilt and hinge axis (TI DLPA083B Table 2-1),
/// its mirror fill the square root of TI's on-state fill factor (an area
/// fraction that includes shading; docs/dmd.md), ideal mirrors and window,
/// all mirrors ON. `preset` out of range is a 7.56 µm, 12° diagonal device.
fn dmd_device(preset: i32) -> SpatialModulator {
    let (pitch, tilt, axis, fill) = usize::try_from(preset)
        .ok()
        .and_then(|i| TI_DMD_PRESETS.get(i))
        .map_or((7.56, 12.0, TiltAxis::Diagonal, 0.94), |p| (p.1, p.2, p.3, p.5));
    SpatialModulator {
        pixels: default_pixels(pitch),
        pitch_um: pitch,
        kind: ModulatorKind::Dmd(DmdSpec {
            tilt_deg: tilt,
            tilt_axis: axis,
            mirror_fill: fill.sqrt(),
            reflectance: 1.0,
            window_transmittance: 1.0,
        }),
        pattern: Pattern::Uniform { value: 1 },
    }
}

/// A generic phase-only LCoS, every parameter in plain view: 8 µm pixels, a
/// linear 8-bit table spanning a wave at 633 nm, fill 0.9, a dead space of
/// amplitude 0.5, no fringing, reflectance 0.9. No vendor's numbers.
fn slm_device() -> SpatialModulator {
    SpatialModulator {
        pixels: default_pixels(8.0),
        pitch_um: 8.0,
        kind: ModulatorKind::Lcos(LcosSpec {
            director_deg: 0.0,
            lut: PhaseLut::Linear { levels: 256, max_phase_rad: 2.0 * std::f64::consts::PI },
            design_lambda_um: 0.633,
            pixel_fill: 0.9,
            gap_amplitude: 0.5,
            gap_phase_rad: 0.0,
            fringing_um: 0.0,
            reflectance: 0.9,
        }),
        pattern: Pattern::Uniform { value: 0 },
    }
}

/// The optics and the array of a device, without its pattern: what
/// `setModulatorDevice` takes, `{"pixels", "pitch_um", "kind"}`.
fn device_edit(json: &str) -> Result<([u32; 2], f64, ModulatorKind), String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("not a device: {e}"))?;
    let pixels: [u32; 2] = serde_json::from_value(v["pixels"].clone()).map_err(|e| format!("pixels: {e}"))?;
    let pitch = v["pitch_um"].as_f64().ok_or("pitch_um: not a number")?;
    let kind: ModulatorKind = serde_json::from_value(v["kind"].clone()).map_err(|e| format!("kind: {e}"))?;
    Ok((pixels, pitch, kind))
}

impl Canvas {
    fn modulator(&self, id: f64) -> Option<&SpatialModulator> {
        self.modulator_and_model(id).map(|(d, _)| d)
    }

    fn modulator_and_model(&self, id: f64) -> Option<(&SpatialModulator, RayModel)> {
        let s = self.scene3.as_ref()?.surfaces.iter().find(|s| s.common.id.0 == id as u64)?;
        match &s.response_front {
            SurfaceResponse::Modulator { device, ray_model } => Some((device, *ray_model)),
            _ => None,
        }
    }

    /// Apply `f` to a copy of the device; keep it, and resize the surface to
    /// its array, only if the device and the scene validate.
    fn edit_modulator(&mut self, id: f64, f: impl FnOnce(&mut SpatialModulator) -> Result<(), String>) -> String {
        let Some(scene) = self.scene3.as_mut() else {
            return reply(Err("no 3D scene".into()));
        };
        let Some(i) = scene.surfaces.iter().position(|s| s.common.id.0 == id as u64) else {
            return reply(Err(format!("no surface #{id}")));
        };
        let SurfaceResponse::Modulator { device, ray_model } = &scene.surfaces[i].response_front else {
            return reply(Err(format!("surface #{id} is not a DMD or SLM")));
        };
        let mut d = (**device).clone();
        let ray_model = *ray_model;
        if let Err(e) = f(&mut d) {
            return reply(Err(e));
        }
        if let Err(e) = d.validate() {
            return reply(Err(e.0));
        }
        let before = scene.surfaces[i].clone();
        let p = d.pitch_mm();
        let (hw, hh) = (0.5 * d.pixels[0] as f64 * p, 0.5 * d.pixels[1] as f64 * p);
        scene.surfaces[i].response_front = SurfaceResponse::Modulator { device: Box::new(d), ray_model };
        scene.surfaces[i].geometry =
            SurfaceGeometry3::Plane { shape: PlanarShape2::Rectangle { half_width_mm: hw, half_height_mm: hh } };
        if let Err(e) = scene.validate() {
            scene.surfaces[i] = before;
            return reply(Err(format!("{e:?}")));
        }
        self.bump();
        reply(Ok(()))
    }
}

#[wasm_bindgen]
impl Canvas {
    /// Add a DMD at `(x, y)` tilted `tilt_deg` in the meridional plane, with
    /// TI preset `preset` (an index into `modulatorPresetsJSON`; out of range
    /// for a generic 7.56 µm device). Returns the new id, or -1.
    #[wasm_bindgen(js_name = addDmd)]
    pub fn add_dmd(&mut self, x: f64, y: f64, tilt_deg: f64, preset: i32) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_modulator(Vec3::new(x, y, 0.0), tilt_deg, dmd_device(preset)) {
            Some(id) => {
                self.bump();
                id as f64
            }
            None => -1.0,
        }
    }

    /// Add a generic phase-only LCoS SLM at `(x, y)`, tilted `tilt_deg`.
    /// Returns the new id, or -1.
    #[wasm_bindgen(js_name = addSlm)]
    pub fn add_slm(&mut self, x: f64, y: f64, tilt_deg: f64) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_modulator(Vec3::new(x, y, 0.0), tilt_deg, slm_device()) {
            Some(id) => {
                self.bump();
                id as f64
            }
            None => -1.0,
        }
    }

    /// The device on surface `id`, or `null`: `{kind: "dmd"|"slm", device,
    /// widthMm, heightMm}`. `device` is the scene's own JSON, except that an
    /// imported pattern is summarized as `{"Explicit": {"count": n}}`.
    #[wasm_bindgen(js_name = modulatorJSON)]
    pub fn modulator_json(&self, id: f64) -> String {
        let Some((d, model)) = self.modulator_and_model(id) else {
            return "null".into();
        };
        let mut device = serde_json::to_value(d).unwrap_or(serde_json::Value::Null);
        if let Pattern::Explicit(v) = &d.pattern {
            device["pattern"] = json!({ "Explicit": { "count": v.len() } });
        }
        let p = d.pitch_mm();
        json!({
            "kind": match d.kind { ModulatorKind::Dmd(_) => "dmd", ModulatorKind::Lcos(_) => "slm" },
            "device": device,
            "widthMm": d.pixels[0] as f64 * p,
            "heightMm": d.pixels[1] as f64 * p,
            "rayModel": model,
        })
        .to_string()
    }

    /// How the ray tracer treats the device: `"Geometric"` or
    /// `{"Orders":{"max_orders":n,"min_efficiency":e}}`.
    #[wasm_bindgen(js_name = setModulatorRayModel)]
    pub fn set_modulator_ray_model(&mut self, id: f64, json: &str) -> String {
        let model: RayModel = match serde_json::from_str(json) {
            Ok(m) => m,
            Err(e) => return reply(Err(format!("not a ray model: {e}"))),
        };
        if let Err(e) = model.validate() {
            return reply(Err(e.0));
        }
        let Some(scene) = self.scene3.as_mut() else {
            return reply(Err("no 3D scene".into()));
        };
        let Some(s) = scene.surfaces.iter_mut().find(|s| s.common.id.0 == id as u64) else {
            return reply(Err(format!("no surface #{id}")));
        };
        let SurfaceResponse::Modulator { ray_model, .. } = &mut s.response_front else {
            return reply(Err(format!("surface #{id} is not a DMD or SLM")));
        };
        *ray_model = model;
        self.bump();
        reply(Ok(()))
    }

    /// Replace the device's array and optics, keeping its pattern:
    /// `{"pixels":[nx,ny],"pitch_um":p,"kind":{...}}` as in the scene's JSON.
    #[wasm_bindgen(js_name = setModulatorDevice)]
    pub fn set_modulator_device(&mut self, id: f64, json: &str) -> String {
        let (pixels, pitch_um, kind) = match device_edit(json) {
            Ok(e) => e,
            Err(err) => return reply(Err(err)),
        };
        self.edit_modulator(id, move |d| {
            d.pixels = pixels;
            d.pitch_um = pitch_um;
            d.kind = kind;
            Ok(())
        })
    }

    /// Replace the pattern: `{"Uniform":{"value":v}}` or `{"Generated":{...}}`.
    /// An imported bitmap comes in through `importModulatorBitmap`.
    #[wasm_bindgen(js_name = setModulatorPattern)]
    pub fn set_modulator_pattern(&mut self, id: f64, json: &str) -> String {
        let p: Pattern = match serde_json::from_str(json) {
            Ok(p) => p,
            Err(err) => return reply(Err(format!("not a pattern: {err}"))),
        };
        if matches!(p, Pattern::Explicit(_)) {
            return reply(Err("an explicit pattern comes from an imported bitmap".into()));
        }
        self.edit_modulator(id, move |d| {
            d.pattern = p;
            Ok(())
        })
    }

    /// Show an RGBA bitmap `w × h` (row 0 at the top), resampled to the
    /// device's pixels by nearest neighbour. A DMD turns a mirror ON where the
    /// pixel's luminance is at least `threshold` (0–255); an SLM maps the
    /// luminance onto its grey levels.
    #[wasm_bindgen(js_name = importModulatorBitmap)]
    pub fn import_modulator_bitmap(&mut self, id: f64, w: u32, h: u32, rgba: &[u8], threshold: f64) -> String {
        if w == 0 || h == 0 || rgba.len() != 4 * w as usize * h as usize {
            return reply(Err(format!("a {w} × {h} bitmap needs {} bytes, not {}", 4 * w as u64 * h as u64, rgba.len())));
        }
        let rgba = rgba.to_vec();
        self.edit_modulator(id, move |d| {
            let [nx, ny] = [d.pixels[0] as usize, d.pixels[1] as usize];
            let levels = match &d.kind {
                ModulatorKind::Dmd(_) => None,
                ModulatorKind::Lcos(l) => Some(l.lut.levels()),
            };
            let mut v = vec![0u16; nx * ny];
            for j in 0..ny {
                // Device rows run up local Y; image rows run down.
                let sy = ((ny - 1 - j) * h as usize / ny).min(h as usize - 1);
                for i in 0..nx {
                    let sx = (i * w as usize / nx).min(w as usize - 1);
                    let k = 4 * (sy * w as usize + sx);
                    let lum = 0.299 * rgba[k] as f64 + 0.587 * rgba[k + 1] as f64 + 0.114 * rgba[k + 2] as f64;
                    v[j * nx + i] = match levels {
                        None => u16::from(lum >= threshold),
                        Some(n) => (lum / 255.0 * (n - 1) as f64).round() as u16,
                    };
                }
            }
            d.pattern = Pattern::Explicit(v);
            Ok(())
        })
    }

    /// `[w, h]` of the picture `modulatorPreviewRGBA` draws for `max_side`.
    #[wasm_bindgen(js_name = modulatorPreviewSize)]
    pub fn modulator_preview_size(&self, id: f64, max_side: u32) -> Vec<u32> {
        let Some(d) = self.modulator(id) else {
            return vec![0, 0];
        };
        let s = (max_side.max(1) as f64 / d.pixels[0].max(d.pixels[1]) as f64).min(1.0);
        vec![((d.pixels[0] as f64 * s).round() as u32).max(1), ((d.pixels[1] as f64 * s).round() as u32).max(1)]
    }

    /// The pattern as RGBA, at most `max_side` pixels on its longer side, row
    /// 0 at the top: a DMD's ON mirrors light, OFF dark, flat mid-grey; an
    /// SLM's phase as grey from 0 (black) to 2π (white).
    #[wasm_bindgen(js_name = modulatorPreviewRGBA)]
    pub fn modulator_preview_rgba(&self, id: f64, max_side: u32) -> Vec<u8> {
        let Some(d) = self.modulator(id) else {
            return Vec::new();
        };
        let size = self.modulator_preview_size(id, max_side);
        let (w, h) = (size[0] as usize, size[1] as usize);
        let [nx, ny] = [d.pixels[0] as usize, d.pixels[1] as usize];
        let mut out = vec![255u8; 4 * w * h];
        for r in 0..h {
            let j = ny - 1 - ((r as f64 + 0.5) * ny as f64 / h as f64).floor().min(ny as f64 - 1.0) as usize;
            for c in 0..w {
                let i = ((c as f64 + 0.5) * nx as f64 / w as f64).floor().min(nx as f64 - 1.0) as usize;
                let g = match &d.kind {
                    ModulatorKind::Dmd(_) => match DmdState::from_value(d.value(i, j)) {
                        DmdState::On => 225,
                        DmdState::Off => 40,
                        DmdState::Flat => 130,
                    },
                    ModulatorKind::Lcos(l) => {
                        let phi = l.lut.phase(d.value(i, j)).rem_euclid(2.0 * std::f64::consts::PI);
                        (phi / (2.0 * std::f64::consts::PI) * 255.0).round() as u8
                    }
                };
                let k = 4 * (r * w + c);
                out[k] = g;
                out[k + 1] = g;
                out[k + 2] = g;
            }
        }
        out
    }

    /// The propagating diffraction orders at `lambda_um`, for the editor's
    /// plot, from the pattern's unit cell in closed form
    /// (`oc_core::modulator::unit_cell_orders`, Harvey's renormalization). A
    /// DMD is lit along the direction its ON mirrors send onto the normal; an
    /// SLM at normal incidence, its modulated beam. `{incidence: [α, β],
    /// orders: [{m, n, alpha, beta, efficiency}], propagating, total, note,
    /// pitchOverLambda}`, direction cosines in the device's frame; orders
    /// below 10⁻⁶ are left out.
    #[wasm_bindgen(js_name = modulatorOrdersJSON)]
    pub fn modulator_orders_json(&self, id: f64, lambda_um: f64) -> String {
        let Some(d) = self.modulator(id) else {
            return "null".into();
        };
        let d_in = match &d.kind {
            ModulatorKind::Dmd(spec) => on_axis_illumination(spec),
            ModulatorKind::Lcos(_) => Vec3::new(0.0, 0.0, -1.0),
        };
        let (cell, exact) = local_unit_cell(d, d.pixels[0] as usize / 2, d.pixels[1] as usize / 2);
        let note = (!exact).then_some(
            "the pattern does not repeat in whole pixels along a pixel axis: these orders take the 8 × 8 pixels at the centre as one period, as the ray tracer does",
        );
        match unit_cell_orders(d, &cell, lambda_um, d_in, OrderModel::HarveyRenormalized, LcosBeam::Modulated) {
            Ok(t) => {
                let orders: Vec<_> = t
                    .orders
                    .iter()
                    .filter(|o| o.efficiency >= 1e-6)
                    .map(|o| json!({ "m": o.m, "n": o.n, "alpha": o.dir.x, "beta": o.dir.y, "efficiency": o.efficiency }))
                    .collect();
                json!({
                    "incidence": [d_in.x, d_in.y],
                    "orders": orders,
                    "propagating": t.propagating,
                    "total": t.total,
                    "period": cell.period,
                    "note": note,
                    "pitchOverLambda": d.pitch_um / lambda_um,
                })
                .to_string()
            }
            Err(e) => json!({ "error": e.0 }).to_string(),
        }
    }

    /// TI's DMD presets (DLPA083B Table 2-1): `[{name, pitchUm, tiltDeg,
    /// axis, fNumber, onStateFill}]`.
    #[wasm_bindgen(js_name = modulatorPresetsJSON)]
    pub fn modulator_presets_json(&self) -> String {
        let rows: Vec<_> = TI_DMD_PRESETS
            .iter()
            .map(|(name, pitch, tilt, axis, f, fill)| {
                json!({
                    "name": name,
                    "pitchUm": pitch,
                    "tiltDeg": tilt,
                    "axis": match axis { TiltAxis::Diagonal => "Diagonal", TiltAxis::Side => "Side" },
                    "fNumber": f,
                    "onStateFill": fill,
                })
            })
            .collect();
        json!(rows).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas() -> Canvas {
        let mut c = Canvas::new();
        c.new_default_scene3();
        c
    }

    #[test]
    fn a_dmd_is_added_read_edited_and_refused_by_name() {
        let mut c = canvas();
        let id = c.add_dmd(0.0, 0.0, 180.0, 3);
        assert!(id >= 0.0);
        let v: serde_json::Value = serde_json::from_str(&c.modulator_json(id)).unwrap();
        assert_eq!(v["kind"], "dmd");
        assert_eq!(v["device"]["pitch_um"], 7.56);
        let rev = c.revision;
        let r = c.set_modulator_pattern(id, r#"{"Generated":{"BinaryGrating":{"period_px":4.0,"duty":0.5,"angle_deg":0.0,"phase":0.0}}}"#);
        assert_eq!(r, "{\"ok\":true}");
        assert!(c.revision > rev);
        // A fill of 1.5 is refused, and nothing changes.
        let mut dev = v["device"].clone();
        dev["kind"]["Dmd"]["mirror_fill"] = json!(1.5);
        let rev = c.revision;
        let r = c.set_modulator_device(id, &dev.to_string());
        assert!(r.contains("error"), "{r}");
        assert_eq!(c.revision, rev);
        // Changing the count resizes the rectangle.
        dev["kind"]["Dmd"]["mirror_fill"] = json!(0.9);
        dev["pixels"] = json!([100, 50]);
        assert_eq!(c.set_modulator_device(id, &dev.to_string()), "{\"ok\":true}");
        let v: serde_json::Value = serde_json::from_str(&c.modulator_json(id)).unwrap();
        assert!((v["widthMm"].as_f64().unwrap() - 0.756).abs() < 1e-12);
        // A period-4 grating at 0° has a 4 × 1 cell and orders on its grid.
        let o: serde_json::Value = serde_json::from_str(&c.modulator_orders_json(id, 0.633)).unwrap();
        assert_eq!(o["period"], json!([4, 1]));
        // The ray model: the geometric limit until set; refused when silly.
        assert_eq!(serde_json::from_str::<serde_json::Value>(&c.modulator_json(id)).unwrap()["rayModel"], "Geometric");
        assert_eq!(c.set_modulator_ray_model(id, r#"{"Orders":{"max_orders":16,"min_efficiency":0.001}}"#), "{\"ok\":true}");
        assert!(c.set_modulator_ray_model(id, r#"{"Orders":{"max_orders":0,"min_efficiency":0.001}}"#).contains("error"));
        assert!(o["orders"].as_array().unwrap().len() > 4);
        assert_eq!(crate::surface_object_kind(&c.scene3.as_ref().unwrap().surfaces.last().unwrap().clone()), "dmd");
    }

    #[test]
    fn a_bitmap_lands_on_the_pixels_the_right_way_up() {
        let mut c = canvas();
        let id = c.add_slm(0.0, 0.0, 180.0);
        let mut dev: serde_json::Value = serde_json::from_str(&c.modulator_json(id)).unwrap();
        dev["device"]["pixels"] = json!([2, 2]);
        assert_eq!(c.set_modulator_device(id, &dev["device"].to_string()), "{\"ok\":true}");
        // Top row white, bottom row black.
        let rgba = [255u8, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255];
        assert_eq!(c.import_modulator_bitmap(id, 2, 2, &rgba, 128.0), "{\"ok\":true}");
        let d = c.modulator(id).unwrap();
        // Row j = 1 is the top of the device (+Y): the white row, level 255.
        assert_eq!(d.value(0, 1), 255);
        assert_eq!(d.value(0, 0), 0);
        let pic = c.modulator_preview_rgba(id, 64);
        assert_eq!(pic.len(), 16);
        assert!(pic[0] >= 250, "the picture's top row is the device's top row: {}", pic[0]);
        assert!(c.import_modulator_bitmap(id, 3, 3, &rgba, 128.0).contains("error"));
    }
}
