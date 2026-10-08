//! The fidelity profile in the app (coating-fidelity spec §5, CF-01): read and
//! set it, see what the preview makes of it, look up what each switch costs,
//! and run the final check.
//!
//! The profile is stored in the scene (`Scene3::fidelity`), so it is saved with
//! the document and a reopened design traces as it was traced. The app's ghost
//! toggle and the profile's ghost setting are kept as one: a profile that sets
//! ghosts sets the toggle, and the toggle writes back into a profile that has a
//! setting of its own.

use oc_core::fidelity::{FidelityProfile, FidelityStamp, Preset};
use oc_core::scene3::Scene3;
use oc_trace::result3::ObjectRef3;
use oc_trace::{TraceConfig3, TraceResult3};
use wasm_bindgen::prelude::*;

use crate::{json_str, Canvas};

/// The measured cost of each switch value (`docs/performance.md`), as
/// `bench_throughput3 --fidelity-table` writes it.
const COSTS: &str = include_str!("../../../web/public/fidelity-costs.json");

/// The visible band the final check quotes a detector's flux in, µm.
const VISIBLE_UM: [f64; 2] = [0.38, 0.78];

fn profile_json(p: &FidelityProfile) -> String {
    serde_json::to_string(p).unwrap_or_else(|_| "null".into())
}

pub(crate) fn stamp_json(s: &FidelityStamp) -> String {
    let approx: Vec<String> = s.approximations.iter().map(|a| json_str(a)).collect();
    format!(
        "{{\"profile\":{},\"preset\":{},\"text\":{},\"approximations\":[{}],\"solverVersion\":{}}}",
        profile_json(&s.profile),
        s.preset.map_or("null".to_string(), |p| json_str(p.name())),
        json_str(&s.text()),
        approx.join(","),
        json_str(&s.solver_version)
    )
}

impl Canvas {
    /// The scene's profile, or the default when no scene is loaded.
    pub(crate) fn fidelity(&self) -> FidelityProfile {
        self.scene3.as_ref().map_or_else(FidelityProfile::default, |s| s.fidelity)
    }

    /// Make the app's ghost toggle agree with a profile that sets ghosts.
    pub(crate) fn sync_ghosts_from_profile(&mut self) {
        if let Some(g) = self.fidelity().ghosts {
            self.ghosts = g > 0;
            if g > 0 {
                self.max_ghost = g.min(8);
            }
        }
    }

    /// Write the ghost toggle back into a profile that has its own setting.
    pub(crate) fn sync_profile_from_ghosts(&mut self) {
        let g = if self.ghosts { self.max_ghost } else { 0 };
        if let Some(scene) = self.scene3.as_mut() {
            if scene.fidelity.ghosts.is_some() {
                scene.fidelity.ghosts = Some(g);
            }
        }
    }

    /// Does the scene declare a temperature, for itself or for any object?
    /// `Reference` declares the temperature only then.
    fn has_temperature(&self) -> bool {
        self.scene3.as_ref().is_some_and(|s| s.temperature_c.is_some() || !s.object_temperatures.is_empty())
    }

    /// One trace of `scene` with `cfg`, at the reference's analysis ray count,
    /// and every object's ledger kept, for the final check.
    fn check_trace(&self, scene: &Scene3, cfg: TraceConfig3) -> TraceResult3 {
        let cfg = TraceConfig3 {
            object_ledger: true,
            ledger_band_um: VISIBLE_UM,
            segment_floor: 1.0,
            ..cfg
        };
        let mut job = match self.analysis_scene(scene) {
            Some((more, passes)) => oc_trace::TraceJob3::with_passes(more, passes, Some(scene), cfg),
            None => oc_trace::TraceJob3::new(scene.clone(), cfg),
        };
        while !job.step(usize::MAX) {}
        job.finish()
    }
}

/// The per-object sums the final check compares.
struct CheckRow {
    flux: f64,
    visible: f64,
    dop: f64,
}

fn detector_rows(scene: &Scene3, r: &TraceResult3) -> Vec<(u64, CheckRow)> {
    scene
        .detectors
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let res = r.detectors.iter().find(|x| x.detector.0 == d.common.id.0);
            let visible: f64 = r
                .objects
                .iter()
                .filter(|e| e.object == ObjectRef3::Detector(i) && e.in_band)
                .map(|e| e.ledger.incident)
                .sum();
            let row = CheckRow {
                flux: res.map_or(0.0, |x| x.total_incident),
                visible,
                dop: res.map_or(0.0, |x| x.stokes_local.dop()),
            };
            (d.common.id.0, row)
        })
        .collect()
}

/// Absorbed power per scene object, by id: surface and volume together.
fn absorbed_by_object(scene: &Scene3, r: &TraceResult3) -> Vec<(u64, f64)> {
    let mut out: Vec<(u64, f64)> = Vec::new();
    for e in &r.objects {
        let id = match e.object {
            ObjectRef3::Body(i, _) => scene.bodies.get(i).map(|b| b.common.id.0),
            ObjectRef3::Surface(i) => scene.surfaces.get(i).map(|s| s.common.id.0),
            _ => None,
        };
        let Some(id) = id else { continue };
        let took = e.ledger.absorbed_surface + e.ledger.absorbed_volume;
        match out.iter_mut().find(|(k, _)| *k == id) {
            Some((_, v)) => *v += took,
            None => out.push((id, took)),
        }
    }
    out
}

fn notes(r: &TraceResult3) -> Vec<String> {
    let cap = &r.capability_report;
    cap.approximated
        .iter()
        .map(|a| format!("approximated: {:?}: {}", a.feature, a.detail))
        .chain(cap.rejected.iter().map(|i| format!("rejected: {i:?}")))
        .collect()
}

#[wasm_bindgen]
impl Canvas {
    /// The scene's fidelity profile and everything the panel shows about it:
    /// `{profile, preset, stamp, preview: {message, refused}, presets}`.
    /// `presets` holds the profile each preset would set, ghosts as the app
    /// has them now.
    #[wasm_bindgen(js_name = fidelityJSON)]
    pub fn fidelity_json(&self) -> String {
        let p = self.fidelity();
        let cfg = self.cfg3();
        let stamp = FidelityStamp::new(cfg.effective_profile());
        let preview = oc_gpu::scene3::preview_fidelity(&cfg.effective_profile());
        let refused: Vec<String> = preview.refused.iter().map(|s| json_str(s)).collect();
        let presets: Vec<String> = Preset::ALL
            .iter()
            .map(|pr| format!("{}:{}", json_str(pr.name()), profile_json(&pr.profile(None, self.has_temperature()))))
            .collect();
        format!(
            "{{\"profile\":{},\"preset\":{},\"stamp\":{},\"preview\":{{\"message\":{},\"refused\":[{}]}},\"presets\":{{{}}}}}",
            profile_json(&p),
            stamp.preset.map_or("null".to_string(), |x| json_str(x.name())),
            stamp_json(&stamp),
            preview.message.as_deref().map_or("null".to_string(), json_str),
            refused.join(","),
            presets.join(",")
        )
    }

    /// Set the scene's profile from its JSON (the document form; a `preset`
    /// field is ignored, the switches are the truth). Bumps the revision, so
    /// every cached result computed under the old profile is dropped. Throws
    /// on JSON that is not a profile, or when no scene is loaded.
    #[wasm_bindgen(js_name = setFidelity)]
    pub fn set_fidelity(&mut self, json: &str) -> Result<(), JsError> {
        let p: FidelityProfile =
            serde_json::from_str(json).map_err(|e| JsError::new(&format!("not a fidelity profile: {e}")))?;
        let Some(scene) = self.scene3.as_mut() else {
            return Err(JsError::new("no 3D scene loaded"));
        };
        scene.fidelity = p;
        self.sync_ghosts_from_profile();
        self.bump();
        Ok(())
    }

    /// Set the scene's profile to a named preset: `Fast`, `Standard` or
    /// `Reference`. `Fast` turns ghosts off; the other two leave the ghost
    /// toggle as it is.
    #[wasm_bindgen(js_name = setFidelityPreset)]
    pub fn set_fidelity_preset(&mut self, name: &str) -> Result<(), JsError> {
        let preset = Preset::ALL
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| JsError::new(&format!("no preset named {name:?}")))?;
        let p = preset.profile(None, self.has_temperature());
        let p = if preset == Preset::Fast { FidelityProfile { ghosts: Some(0), ..p } } else { p };
        let Some(scene) = self.scene3.as_mut() else {
            return Err(JsError::new("no 3D scene loaded"));
        };
        scene.fidelity = p;
        self.sync_ghosts_from_profile();
        self.bump();
        Ok(())
    }

    /// The temperature the scene's coatings are at, °C, or `NaN` when none is
    /// declared (every film at its reference). Read only under
    /// `temperature: Declared` (CF-10).
    #[wasm_bindgen(js_name = sceneTemperatureC)]
    pub fn scene_temperature_c(&self) -> f64 {
        self.scene3.as_ref().and_then(|s| s.temperature_c).unwrap_or(f64::NAN)
    }

    /// Declare the scene's temperature, °C, or clear it with `NaN`. False
    /// with no scene, or for a temperature at or below absolute zero.
    #[wasm_bindgen(js_name = setSceneTemperature)]
    pub fn set_scene_temperature(&mut self, c: f64) -> bool {
        if c.is_finite() && c <= -273.15 || c.is_infinite() {
            return false;
        }
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        scene.temperature_c = (!c.is_nan()).then_some(c);
        self.bump();
        true
    }

    /// Object `id`'s own temperature, °C, or `NaN` when it takes the scene's.
    #[wasm_bindgen(js_name = objectTemperatureC)]
    pub fn object_temperature_c(&self, id: f64) -> f64 {
        self.scene3
            .as_ref()
            .and_then(|s| s.object_temperatures.iter().find(|t| t.object.0 == id as u64).map(|t| t.temperature_c))
            .unwrap_or(f64::NAN)
    }

    /// Give object `id` a temperature of its own, °C, or clear it with `NaN`.
    /// False for an id that is not a body or surface, or a temperature at or
    /// below absolute zero.
    #[wasm_bindgen(js_name = setObjectTemperature)]
    pub fn set_object_temperature(&mut self, id: f64, c: f64) -> bool {
        if c.is_finite() && c <= -273.15 || c.is_infinite() {
            return false;
        }
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_object_temperature(id as u64, (!c.is_nan()).then_some(c)) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// The scene's profile as a short hash, for the keys of anything that
    /// caches a traced result (the reference pool's requests among them).
    #[wasm_bindgen(js_name = fidelityKey)]
    pub fn fidelity_key(&self) -> String {
        format!("{:016x}", self.fidelity().key())
    }

    /// The measured relative cost of each switch value, from
    /// `docs/performance.md`. A value with no row has not been measured, and
    /// the panel says so rather than showing a number.
    #[wasm_bindgen(js_name = fidelityCostsJSON)]
    pub fn fidelity_costs_json(&self) -> String {
        COSTS.to_string()
    }

    /// The final check (spec §5.4): one f64 trace under `Reference`, without
    /// changing the scene's profile, against one under the scene's own.
    /// `{current, reference, detectors: [{id, flux, visible, dop}, ...],
    /// objects: [{id, absorbed}, ...], onlyInCurrent, onlyInReference}`, where
    /// each number is a pair `[current, reference]`. `null` with no scene.
    #[wasm_bindgen(js_name = finalCheckJSON)]
    pub fn final_check_json(&self) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        let current_cfg = self.cfg3();
        let reference_cfg = self.cfg3().with_fidelity(FidelityProfile {
            ghosts: None,
            ..Preset::Reference.profile(None, self.has_temperature())
        });
        let a = self.check_trace(scene, current_cfg);
        let b = self.check_trace(scene, reference_cfg);
        let pair = |x: f64, y: f64| format!("[{},{}]", crate::json_num(x), crate::json_num(y));

        let da = detector_rows(scene, &a);
        let db = detector_rows(scene, &b);
        let dets: Vec<String> = da
            .iter()
            .zip(&db)
            .map(|((id, x), (_, y))| {
                format!(
                    "{{\"id\":{id},\"flux\":{},\"visible\":{},\"dop\":{}}}",
                    pair(x.flux, y.flux),
                    pair(x.visible, y.visible),
                    pair(x.dop, y.dop)
                )
            })
            .collect();

        let oa = absorbed_by_object(scene, &a);
        let ob = absorbed_by_object(scene, &b);
        let mut ids: Vec<u64> = oa.iter().chain(&ob).map(|(k, _)| *k).collect();
        ids.sort_unstable();
        ids.dedup();
        let get = |v: &[(u64, f64)], id: u64| v.iter().find(|(k, _)| *k == id).map_or(0.0, |(_, x)| *x);
        let objs: Vec<String> = ids
            .iter()
            .map(|&id| format!("{{\"id\":{id},\"absorbed\":{}}}", pair(get(&oa, id), get(&ob, id))))
            .collect();

        let (na, nb) = (notes(&a), notes(&b));
        let only = |x: &[String], y: &[String]| -> String {
            let v: Vec<String> = x.iter().filter(|n| !y.contains(n)).map(|n| json_str(n)).collect();
            format!("[{}]", v.join(","))
        };
        format!(
            "{{\"current\":{},\"reference\":{},\"detectors\":[{}],\"objects\":[{}],\"onlyInCurrent\":{},\"onlyInReference\":{}}}",
            stamp_json(&a.fidelity),
            stamp_json(&b.fidelity),
            dets.join(","),
            objs.join(","),
            only(&na, &nb),
            only(&nb, &na)
        )
    }
}
