//! The analysis job (ADR 0003): one interface for every analysis
//! `oc-analysis` runs, stepped a batch of launch rays at a time.
//!
//! ```text
//! analysisBegin(requestJSON)  -> launch rays in all
//! analysisStep(rays)          -> fraction traced; 1 once the result is ready
//! analysisProgressJSON()      -> where the job is
//! analysisResultJSON()        -> the result envelope, or null
//! analysisCancel()            -> drop the job
//! ```
//!
//! It follows the reference's rule (`referenceBegin` / `referenceStep`): a job
//! belongs to the scene, its revision, profile and ghost setting, it was begun
//! for, and the first call after any of those moves drops it. A dropped job's
//! result is never served.

use oc_analysis::{AnalysisJob, AnalysisRequest};
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::Canvas;

/// What an analysis depends on besides its request: the scene's revision and
/// the trace settings that move no revision.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct AnalysisKey {
    revision: u64,
    fidelity: u64,
    ghosts: bool,
    max_ghost: u32,
}

/// The job in hand, or why the last one went.
#[derive(Default)]
pub(crate) struct AnalysisSlot {
    job: Option<(AnalysisKey, AnalysisJob)>,
    /// Why the last job was dropped: `"cancelled"` or `"sceneChanged"`.
    dropped: Option<&'static str>,
}

#[wasm_bindgen]
impl Canvas {
    /// Begin the analysis `requestJSON` asks for (`oc_analysis::request`) on
    /// the 3D scene as it is, replacing any job in hand. Returns the launch
    /// rays it traces in all. Throws, with the reason, for a request this
    /// build cannot run or this scene cannot answer, and with no 3D scene.
    #[wasm_bindgen(js_name = analysisBegin)]
    pub fn analysis_begin(&self, request_json: &str) -> Result<f64, JsError> {
        self.begin_analysis(request_json).map_err(|e| JsError::new(&e))
    }

    /// Trace up to `rays` more launch rays of the job in hand (at least one).
    /// Returns the fraction traced; at 1 the result is ready. A job begun for
    /// a scene that has changed since is dropped, and 0 returned; so is 0
    /// with no job.
    #[wasm_bindgen(js_name = analysisStep)]
    pub fn analysis_step(&self, rays: u32) -> f64 {
        if !self.analysis_current() {
            return 0.0;
        }
        let mut slot = self.analysis.borrow_mut();
        let Some((_, job)) = slot.job.as_mut() else {
            return 0.0;
        };
        if job.step(rays as u64) {
            return 1.0;
        }
        let p = job.progress();
        p.traced as f64 / p.rays.max(1) as f64
    }

    /// Where the job is, as JSON: `state` is `idle` (no job), `running`,
    /// `done` (the result is ready) or `dropped`, with the `reason`
    /// (`cancelled`, `sceneChanged`). A job has its `kind`, the `revision` it
    /// traces, and its `traced` and `rays`, `passesDone` and `passes`.
    #[wasm_bindgen(js_name = analysisProgressJSON)]
    pub fn analysis_progress_json(&self) -> String {
        self.analysis_current();
        let slot = self.analysis.borrow();
        let v = match (&slot.job, slot.dropped) {
            (Some((key, job)), _) => {
                let p = job.progress();
                json!({
                    "state": if job.done() { "done" } else { "running" },
                    "kind": job.request().kind.name(),
                    "revision": key.revision,
                    "traced": p.traced,
                    "rays": p.rays,
                    "passesDone": p.passes_done,
                    "passes": p.passes,
                })
            }
            (None, Some(reason)) => json!({ "state": "dropped", "reason": reason }),
            (None, None) => json!({ "state": "idle" }),
        };
        v.to_string()
    }

    /// The result envelope of the job in hand once it is complete
    /// (`oc_analysis::AnalysisJob::result`), or `null`: before then, with no
    /// job, or once the scene has changed.
    #[wasm_bindgen(js_name = analysisResultJSON)]
    pub fn analysis_result_json(&self) -> String {
        if !self.analysis_current() {
            return "null".into();
        }
        self.analysis
            .borrow()
            .job
            .as_ref()
            .and_then(|(_, job)| job.result())
            .map_or_else(|| "null".into(), |v| v.to_string())
    }

    /// Drop the job in hand and its result. False when there was none.
    #[wasm_bindgen(js_name = analysisCancel)]
    pub fn analysis_cancel(&self) -> bool {
        let mut slot = self.analysis.borrow_mut();
        let had = slot.job.take().is_some();
        if had {
            slot.dropped = Some("cancelled");
        }
        had
    }
}

impl Canvas {
    fn analysis_key(&self) -> AnalysisKey {
        AnalysisKey {
            revision: self.revision,
            fidelity: self.fidelity().key(),
            ghosts: self.ghosts,
            max_ghost: self.max_ghost,
        }
    }

    /// [`Canvas::analysis_begin`], with its error as text.
    pub(crate) fn begin_analysis(&self, request_json: &str) -> Result<f64, String> {
        let scene = self.scene3.as_ref().ok_or("no 3D scene loaded")?;
        let request = AnalysisRequest::from_json(request_json)?;
        let job = AnalysisJob::new(scene, self.cfg3(), request)?;
        let rays = job.progress().rays as f64;
        *self.analysis.borrow_mut() = AnalysisSlot {
            job: Some((self.analysis_key(), job)),
            dropped: None,
        };
        Ok(rays)
    }

    /// Whether there is a job for the scene as it is. One for a scene that has
    /// changed since is dropped here.
    fn analysis_current(&self) -> bool {
        let key = self.analysis_key();
        let mut slot = self.analysis.borrow_mut();
        match &slot.job {
            Some((k, _)) if *k == key => true,
            Some(_) => {
                slot.job = None;
                slot.dropped = Some("sceneChanged");
                false
            }
            None => false,
        }
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

    fn progress(c: &Canvas) -> serde_json::Value {
        serde_json::from_str(&c.analysis_progress_json()).unwrap()
    }

    const REQ: &str = r#"{"kind":"detectorStats","passes":4}"#;

    #[test]
    fn a_job_runs_to_its_result() {
        let c = canvas();
        assert_eq!(progress(&c)["state"], "idle");
        let rays = c.begin_analysis(REQ).unwrap();
        assert!(rays > 0.0);
        assert_eq!(c.analysis_result_json(), "null");
        let f = c.analysis_step(10);
        assert!(f > 0.0 && f < 1.0, "{f}");
        assert_eq!(progress(&c)["state"], "running");
        while c.analysis_step(100) < 1.0 {}
        let p = progress(&c);
        assert_eq!(p["state"], "done");
        assert_eq!(p["traced"], p["rays"]);
        let r: serde_json::Value = serde_json::from_str(&c.analysis_result_json()).unwrap();
        assert_eq!(r["kind"], "detectorStats");
        assert_eq!(r["revision"].as_f64().unwrap(), c.scene_revision());
        assert!(r["result"]["detectors"][0]["flux"]["stderr"].is_number());
        // Stepping a finished job keeps its result.
        assert_eq!(c.analysis_step(1), 1.0);
    }

    #[test]
    fn cancel_drops_the_job() {
        let c = canvas();
        c.begin_analysis(REQ).unwrap();
        c.analysis_step(10);
        assert!(c.analysis_cancel());
        assert!(!c.analysis_cancel());
        assert_eq!(c.analysis_step(1000), 0.0);
        assert_eq!(c.analysis_result_json(), "null");
        let p = progress(&c);
        assert_eq!((p["state"].as_str(), p["reason"].as_str()), (Some("dropped"), Some("cancelled")));
    }

    #[test]
    fn a_scene_change_drops_the_job_and_its_result() {
        let mut c = canvas();
        c.begin_analysis(REQ).unwrap();
        c.analysis_step(10);
        c.bump();
        assert_eq!(c.analysis_step(1000), 0.0);
        let p = progress(&c);
        assert_eq!((p["state"].as_str(), p["reason"].as_str()), (Some("dropped"), Some("sceneChanged")));

        // A finished result goes stale too, and so does one traced before a
        // trace setting moved.
        c.begin_analysis(REQ).unwrap();
        while c.analysis_step(1000) < 1.0 {}
        assert_ne!(c.analysis_result_json(), "null");
        c.set_ghosts(!c.ghosts_enabled());
        assert_eq!(c.analysis_result_json(), "null");
        assert_eq!(progress(&c)["reason"], "sceneChanged");
    }

    #[test]
    fn a_bad_request_says_why_and_keeps_nothing() {
        let c = canvas();
        assert!(c.begin_analysis(r#"{"kind":"mtf"}"#).unwrap_err().contains("unknown analysis kind"));
        assert_eq!(progress(&c)["state"], "idle");
        assert!(Canvas::new().begin_analysis(REQ).unwrap_err().contains("no 3D scene"));
    }

    /// The job steps the same scene the reference traces, under the same
    /// settings: its pooled flux is the reference readout's at the same rays.
    #[test]
    fn the_job_traces_under_the_canvas_settings() {
        let c = canvas();
        c.begin_analysis(r#"{"kind":"detectorStats","passes":2}"#).unwrap();
        while c.analysis_step(u32::MAX) < 1.0 {}
        let r: serde_json::Value = serde_json::from_str(&c.analysis_result_json()).unwrap();
        let scene = c.scene3.clone().unwrap();
        let mut job = oc_trace::TraceJob3::with_passes(scene, 2, None, c.cfg3());
        while !job.step(usize::MAX) {}
        let t = job.finish();
        let flux = r["result"]["detectors"][0]["flux"]["value"].as_f64().unwrap();
        assert_eq!(flux.to_bits(), t.detectors[0].total_incident.to_bits());
    }
}
