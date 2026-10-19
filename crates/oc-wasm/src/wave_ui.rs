//! The coherent wave path in the app (W5): a source's coherent beam, and a
//! wave job a worker runs in steps.
//!
//! `oc_wave::path` is the physics. A job is begun for a source and a
//! detector, stepped an event at a time so the worker can yield between
//! steps, and finished into a result the page reads as JSON plus typed
//! arrays. Like the reference, the job answers for the revision it began at:
//! a job whose scene has moved on is dropped, never finished.

use oc_core::id::{DetectorId, SourceId};
use oc_core::scene3::CoherentBeam;
use oc_wave::path::{WaveJob, WavePathResult, WavePathSpec};
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::{json_str, Canvas};

fn error(e: impl std::fmt::Display) -> String {
    format!("{{\"error\":{}}}", json_str(&e.to_string()))
}

/// The most samples across a wave field the app will hold: 2048² complex
/// samples, padded twice for the angular spectrum, is 128 MB in the browser.
const MAX_WAVE_GRID: u32 = 2048;

#[wasm_bindgen]
impl Canvas {
    /// Give source `id` a coherent beam (`{"Gaussian":{"waist_radius_mm":w,
    /// "waist_offset_mm":z}}` or `{"TopHat":{"radius_mm":r}}`), or take it
    /// away (`null`). `{"ok":true}` or `{"error":...}`.
    #[wasm_bindgen(js_name = setSourceCoherence)]
    pub fn set_source_coherence(&mut self, id: f64, json: &str) -> String {
        let beam: Option<CoherentBeam> = match serde_json::from_str(json) {
            Ok(b) => b,
            Err(e) => return error(format!("not a coherent beam: {e}")),
        };
        if let Some(b) = &beam {
            if let Err(e) = b.check() {
                return error(e);
            }
        }
        let Some(scene) = self.scene3.as_mut() else {
            return error("no 3D scene");
        };
        let Some(src) = scene.sources.iter_mut().find(|s| s.id.0 == id as u64) else {
            return error(format!("no source #{id}"));
        };
        src.coherence = beam;
        self.bump();
        "{\"ok\":true}".into()
    }

    /// Source `id`'s coherent beam, or `null`.
    #[wasm_bindgen(js_name = sourceCoherenceJSON)]
    pub fn source_coherence_json(&self, id: f64) -> String {
        self.scene3
            .as_ref()
            .and_then(|s| s.sources.iter().find(|s| s.id.0 == id as u64))
            .and_then(|s| s.coherence)
            .map_or("null".into(), |b| serde_json::to_string(&b).unwrap_or_else(|_| "null".into()))
    }

    /// What a wave path can run between: `{sources: [{id, coherence}],
    /// detectors: [{id, label, rectangle, widthMm, heightMm, bins}]}`. Only a
    /// rectangular detector can be read out.
    #[wasm_bindgen(js_name = waveTargetsJSON)]
    pub fn wave_targets_json(&self) -> String {
        let Some(scene) = self.scene3.as_ref() else {
            return "null".into();
        };
        let sources: Vec<_> = scene.sources.iter().map(|s| json!({ "id": s.id.0, "coherence": s.coherence })).collect();
        let detectors: Vec<_> = scene
            .detectors
            .iter()
            .map(|d| {
                let rect = match d.geometry {
                    oc_core::scene3::DetectorGeometry3::Rectangle { half_width_mm, half_height_mm } => {
                        Some((2.0 * half_width_mm, 2.0 * half_height_mm))
                    }
                    _ => None,
                };
                json!({
                    "id": d.common.id.0,
                    "label": d.common.label,
                    "rectangle": rect.is_some(),
                    "widthMm": rect.map(|r| r.0),
                    "heightMm": rect.map(|r| r.1),
                    "bins": d.bins,
                })
            })
            .collect();
        json!({ "sources": sources, "detectors": detectors }).to_string()
    }

    /// Begin a wave path from `source` (its own coherent beam, its flux the
    /// power) to `detector`, the field `grid_n` samples across `window_mm`.
    /// `{"ok":true,"events":n,"arms":k,"lines":m}` (an interferometer's arms
    /// meeting on the detector, the source's spectral lines) or
    /// `{"error":...}`, which names what the path cannot carry.
    #[wasm_bindgen(js_name = waveBegin)]
    pub fn wave_begin(&self, source: f64, detector: f64, grid_n: u32, window_mm: f64) -> String {
        *self.wave_result.borrow_mut() = None;
        *self.wave_job.borrow_mut() = None;
        let Some(scene) = self.scene3.as_ref() else {
            return error("no 3D scene");
        };
        if !(8..=MAX_WAVE_GRID).contains(&grid_n) {
            return error(format!("a wave field of {grid_n} samples across (8 to {MAX_WAVE_GRID})"));
        }
        let spec = match WavePathSpec::for_source(
            scene,
            SourceId(source as u64),
            DetectorId(detector as u64),
            grid_n as usize,
            window_mm,
        ) {
            Ok(s) => s,
            Err(e) => return error(e),
        };
        match WaveJob::new(scene, &spec, self.revision) {
            Ok(job) => {
                let (events, arms, lines) = (job.events(), job.arms(), job.lines());
                *self.wave_job.borrow_mut() = Some(job);
                json!({ "ok": true, "events": events, "arms": arms, "lines": lines }).to_string()
            }
            Err(e) => error(e),
        }
    }

    /// Carry the begun job through up to `events` more of its events.
    /// `{"done":bool,"fraction":f}`, or `{"error":...}` when the path meets
    /// an element it has no model for, or when the scene has changed since
    /// the job began (the job is then dropped).
    #[wasm_bindgen(js_name = waveStep)]
    pub fn wave_step(&self, events: u32) -> String {
        let mut slot = self.wave_job.borrow_mut();
        let Some(job) = slot.as_mut() else {
            return error("no wave job");
        };
        if job.revision() != self.revision {
            *slot = None;
            return error("the scene changed since the wave job began");
        }
        match job.step(events.max(1) as usize) {
            Ok(false) => {
                let f = job.done() as f64 / job.events().max(1) as f64;
                json!({ "done": false, "fraction": f }).to_string()
            }
            Ok(true) => {
                let job = slot.take().expect("held above");
                match job.finish() {
                    Ok(r) => {
                        *self.wave_result.borrow_mut() = Some(r);
                        json!({ "done": true, "fraction": 1.0 }).to_string()
                    }
                    Err(e) => error(e),
                }
            }
            Err(e) => {
                *slot = None;
                error(e)
            }
        }
    }

    /// The finished wave path, or `null`: `{revision, detector, bins, report,
    /// field: {n, pitchMm, lambdaUm}, power}`; `power` is the total on the
    /// detector. Per-pixel numbers come from `wavePower`.
    #[wasm_bindgen(js_name = waveResultJSON)]
    pub fn wave_result_json(&self) -> String {
        let r = self.wave_result.borrow();
        let Some(r) = r.as_ref() else {
            return "null".into();
        };
        json!({
            "revision": r.revision,
            "detector": r.detector.0,
            "bins": r.bins,
            "report": r.report,
            "field": { "n": r.field.grid.n, "pitchMm": r.field.grid.pitch_mm, "lambdaUm": r.field.lambda_um, "polarized": r.field_y.is_some() },
            "power": r.power.iter().sum::<f64>(),
        })
        .to_string()
    }

    /// The power in each detector pixel, row-major, `u` fastest.
    #[wasm_bindgen(js_name = wavePower)]
    pub fn wave_power(&self) -> Vec<f64> {
        self.wave_result.borrow().as_ref().map_or_else(Vec::new, |r: &WavePathResult| r.power.clone())
    }

    /// `|U|²` of the field arriving at the detector, across the chief ray,
    /// row-major, `x` fastest; of a field of two polarization components,
    /// their sum.
    #[wasm_bindgen(js_name = waveFieldIntensity)]
    pub fn wave_field_intensity(&self) -> Vec<f64> {
        self.wave_result.borrow().as_ref().map_or_else(Vec::new, |r| match &r.field_y {
            Some(y) => r.field.data.iter().zip(&y.data).map(|(a, b)| a.norm_sqr() + b.norm_sqr()).collect(),
            None => r.field.data.iter().map(|c| c.norm_sqr()).collect(),
        })
    }

    /// The phase of the same field, radians in `(−π, π]`, carrier excluded;
    /// of a field of two polarization components, the `x` one's.
    #[wasm_bindgen(js_name = wavePhase)]
    pub fn wave_phase(&self) -> Vec<f64> {
        self.wave_result.borrow().as_ref().map_or_else(Vec::new, |r| r.field.data.iter().map(|c| c.arg()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wave_path_runs_in_steps_and_a_scene_edit_drops_it() {
        use oc_core::scene::Spectrum;
        use oc_core::scene3::Scene3;
        use oc_core::vec3::Vec3;
        let mut c = Canvas::new();
        let mut s = Scene3::new();
        let src = s.add_collimated_beam(Vec3::ZERO, 1.0, 1.0, 1, Spectrum::Monochromatic(0.6328)).0 as f64;
        let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [16, 16]).0 as f64;
        c.scene3 = Some(s);
        let t: serde_json::Value = serde_json::from_str(&c.wave_targets_json()).unwrap();
        assert_eq!(t["detectors"][0]["rectangle"], true);
        // No beam declared yet: refused by name.
        assert!(c.wave_begin(src, det, 64, 4.0).contains("no coherent beam"));
        assert_eq!(
            c.set_source_coherence(src, r#"{"Gaussian":{"waist_radius_mm":0.5,"waist_offset_mm":0.0}}"#),
            "{\"ok\":true}"
        );
        assert!(c.source_coherence_json(src).contains("Gaussian"));
        assert!(c.set_source_coherence(src, r#"{"TopHat":{"radius_mm":-1}}"#).contains("error"));
        let b: serde_json::Value = serde_json::from_str(&c.wave_begin(src, det, 64, 4.0)).unwrap();
        assert_eq!(b["ok"], true, "{b}");
        loop {
            let s: serde_json::Value = serde_json::from_str(&c.wave_step(1)).unwrap();
            assert!(s.get("error").is_none(), "{s}");
            if s["done"] == true {
                break;
            }
        }
        let r: serde_json::Value = serde_json::from_str(&c.wave_result_json()).unwrap();
        assert_eq!(r["revision"].as_u64(), Some(c.revision));
        assert_eq!(c.wave_power().len(), 256);
        assert!((r["power"].as_f64().unwrap() - 1.0).abs() < 1e-6, "the whole beam lands: {r}");
        assert_eq!(c.wave_field_intensity().len(), 64 * 64);
        // An edit after the job began drops it.
        assert!(c.wave_begin(src, det, 64, 4.0).contains("ok"));
        c.bump();
        assert!(c.wave_step(1).contains("scene changed"));
    }
}
