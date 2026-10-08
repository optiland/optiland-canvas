//! The film catalogue in the app (coating-fidelity spec CF-03): what each
//! film material is, where its data came from, and the wavelengths it covers.

use oc_core::film_catalog;
use oc_core::thinfilm::FilmMaterial;
use wasm_bindgen::prelude::*;

use crate::Canvas;

fn material_json(m: &FilmMaterial) -> String {
    serde_json::to_string(m).unwrap_or_else(|_| "null".into())
}

#[wasm_bindgen]
impl Canvas {
    /// The film catalogue: `{builtin: [{key, material, thermalNote}], scene:
    /// [{id, material, thermalNote}]}`. Each material carries its curves or
    /// formula, its form (film, bulk, crystal), its provenance (source, paper,
    /// url, licence, range), its nominal index and its thermo-optic model, in
    /// the document's own form; `thermalNote` is "no thermo-optic data" for a
    /// material without one.
    #[wasm_bindgen(js_name = filmCatalogJSON)]
    pub fn film_catalog_json(&self) -> String {
        // Every entry says whether it has a thermo-optic model, and one that
        // has none says so in words (CF-10).
        let thermal = |m: &oc_core::thinfilm::FilmMaterial| {
            if m.thermal.is_some() { "null" } else { "\"no thermo-optic data\"" }
        };
        let builtin: Vec<String> = film_catalog::builtin()
            .into_iter()
            .map(|(key, m)| format!("{{\"key\":\"{key}\",\"material\":{},\"thermalNote\":{}}}", material_json(&m), thermal(&m)))
            .collect();
        let scene: Vec<String> = self
            .scene3
            .as_ref()
            .map(|s| {
                s.film_catalog
                    .iter()
                    .map(|e| {
                        format!("{{\"id\":{},\"material\":{},\"thermalNote\":{}}}", e.id, material_json(&e.material), thermal(&e.material))
                    })
                    .collect()
            })
            .unwrap_or_default();
        format!("{{\"builtin\":[{}],\"scene\":[{}]}}", builtin.join(","), scene.join(","))
    }
}

#[wasm_bindgen]
impl Canvas {
    /// A coated face's ellipsometric angles and transmission retardance at
    /// each wavelength in `nm`, for light arriving at `aoi_deg`:
    /// `[ψ, Δ, ret_t, ψ, Δ, ret_t, ...]` in degrees (coating-fidelity spec
    /// CF-06). `tan ψ · e^{iΔ} = r_p / r_s` on the ray-frame amplitudes; `ret_t`
    /// is `arg t_p − arg t_s`. `NaN` where the coating has no amplitude (a
    /// plate, a coating with slabs) or no answer; empty for an id with no
    /// coating.
    #[wasm_bindgen(js_name = surfaceEllipsometryF64)]
    pub fn surface_ellipsometry_f64(&self, id: f64, back: bool, nm: &[f64], aoi_deg: f64) -> Vec<f64> {
        use oc_core::scene3::SurfaceResponse;
        let Some(scene) = self.scene3.as_ref() else {
            return Vec::new();
        };
        let ambient = scene.env.ambient_index;
        // A body's coated face sits on the body's glass; a surface's coating
        // between ambient media.
        let (coating, on_body) = match scene.surface_responses(id as u64) {
            Some((front, back_r)) => match if back { back_r } else { front } {
                SurfaceResponse::Coating { coating } => (coating, false),
                _ => return Vec::new(),
            },
            None => match scene.coating_for_object(id as u64) {
                Some(c) => (c, true),
                None => return Vec::new(),
            },
        };
        let cos_i = aoi_deg.to_radians().cos();
        let mut out = Vec::with_capacity(nm.len() * 3);
        for &w in nm {
            let um = w / 1000.0;
            let (n0, n_far) = if on_body { scene.coated_face_media(id as u64, um) } else { (ambient, ambient) };
            match oc_core::thinfilm::coating_amp_at(coating, n0, n_far, um, cos_i) {
                Ok(a) => {
                    let (psi, delta) = a.ellipsometric();
                    out.extend([psi.to_degrees(), delta.to_degrees(), a.transmission_retardance().to_degrees()]);
                }
                Err(_) => out.extend([f64::NAN; 3]),
            }
        }
        out
    }
}

/// Bins of the angle-of-incidence histogram, degrees.
const AOI_BIN_DEG: f64 = 2.0;
const AOI_BINS: usize = 45;

#[wasm_bindgen]
impl Canvas {
    /// The angles at which the scene's rays meet an object's coated face, and
    /// what the coating does to them (coating-fidelity spec CF-07): one f64
    /// trace of the scene's own rays with the event log on, every hit on the
    /// face binned by its angle of incidence in 2 degree bins, weighted by the
    /// flux it carried, on each side the light can arrive from. With them,
    /// the face's R and T at each wavelength in `nm` averaged over that
    /// histogram, which is the curve the rays actually see, beside the curve
    /// at a single angle.
    ///
    /// `{binDeg, front: {weights, flux, hits, meanDeg, r, t}, back: {...}}`,
    /// `r` and `t` null on a side no ray reached; `null` for an id with no
    /// coating or no scene.
    #[wasm_bindgen(js_name = faceIncidenceJSON)]
    pub fn face_incidence_json(&self, id: f64, nm: &[f64]) -> String {
        use oc_core::scene3::SurfaceResponse;
        use oc_trace::TraceEventKind;
        let Some(scene) = self.scene3.as_ref() else {
            return "null".into();
        };
        let id = id as u64;
        let surface = scene.surfaces.iter().find(|s| s.common.id.0 == id);
        let body = scene.bodies.iter().find(|b| b.common.id.0 == id);
        if surface.is_none() && body.is_none() {
            return "null".into();
        }
        let cfg = oc_trace::TraceConfig3 { record_events: true, segment_floor: 0.0, ..self.cfg3() };
        let res = oc_trace::trace_scene3(scene, &cfg);
        // The segment that ends at each event carries the incoming direction
        // and flux.
        let mut by_path: std::collections::HashMap<u64, &oc_trace::Segment3> = std::collections::HashMap::new();
        for s in &res.segments {
            by_path.insert(s.path_id.0, s);
        }
        let ambient = scene.env.ambient_index;
        let mut weights = [vec![0.0f64; AOI_BINS], vec![0.0f64; AOI_BINS]];
        let mut hits = [0u64; 2];
        let mut flux = [0.0f64; 2];
        let mut coating_of: [Option<oc_core::thinfilm::Coating>; 2] = [None, None];
        for ev in &res.events {
            let TraceEventKind::Interact { surface: sref, normal, n1, .. } = &ev.kind else { continue };
            if sref.object.0 != id {
                continue;
            }
            let Some(seg) = by_path.get(&ev.path_id.0) else { continue };
            let d = (seg.b - seg.a).normalize();
            if !d.is_finite() {
                continue;
            }
            let cos = (-d.dot(*normal)).clamp(-1.0, 1.0).abs();
            // Which side, and that side's coating.
            let (side, coating) = match (surface, body) {
                (Some(s), _) => {
                    let front_n = s.common.transform.quat() * oc_core::vec3::Vec3::Z;
                    let front = normal.dot(front_n) > 0.0;
                    let r = if front { &s.response_front } else { &s.response_back };
                    (usize::from(!front), r)
                }
                (None, Some(b)) => {
                    let two = b.surface.for_patch(sref.patch);
                    let outside = (n1 - ambient).abs() < 1e-9;
                    (usize::from(!outside), if outside { &two.from_outside } else { &two.from_inside })
                }
                _ => continue,
            };
            let SurfaceResponse::Coating { coating } = coating else { continue };
            coating_of[side].get_or_insert_with(|| coating.clone());
            let deg = cos.acos().to_degrees();
            let bin = ((deg / AOI_BIN_DEG) as usize).min(AOI_BINS - 1);
            weights[side][bin] += seg.flux_end;
            hits[side] += 1;
            flux[side] += seg.flux_end;
        }
        let side_json = |k: usize| -> String {
            let total = flux[k];
            let mean = if total > 0.0 {
                weights[k].iter().enumerate().map(|(i, w)| w * (i as f64 + 0.5) * AOI_BIN_DEG).sum::<f64>() / total
            } else {
                f64::NAN
            };
            let (r, t) = match (&coating_of[k], total > 0.0) {
                (Some(c), true) => {
                    let mut r = Vec::with_capacity(nm.len());
                    let mut t = Vec::with_capacity(nm.len());
                    for &w in nm {
                        let um = w / 1000.0;
                        let (n_in, n_out) = match (surface, k) {
                            (Some(_), _) => (ambient, ambient),
                            (None, 0) => scene.coated_face_media(id, um),
                            (None, _) => {
                                let (a, g) = scene.coated_face_media(id, um);
                                (g, a)
                            }
                        };
                        let (mut sr, mut st) = (0.0, 0.0);
                        for (i, wt) in weights[k].iter().enumerate() {
                            if *wt == 0.0 {
                                continue;
                            }
                            let cos = ((i as f64 + 0.5) * AOI_BIN_DEG).to_radians().cos();
                            if let Ok(v) = oc_core::thinfilm::coating_rt_at(c, n_in, n_out, um, cos) {
                                sr += wt * v.r();
                                st += wt * v.t();
                            }
                        }
                        r.push(crate::json_num(sr / total));
                        t.push(crate::json_num(st / total));
                    }
                    (format!("[{}]", r.join(",")), format!("[{}]", t.join(",")))
                }
                _ => ("null".to_string(), "null".to_string()),
            };
            let w: Vec<String> = weights[k].iter().map(|v| crate::json_num(*v)).collect();
            format!(
                "{{\"weights\":[{}],\"flux\":{},\"hits\":{},\"meanDeg\":{},\"r\":{},\"t\":{}}}",
                w.join(","),
                crate::json_num(total),
                hits[k],
                crate::json_num(mean),
                r,
                t
            )
        };
        format!("{{\"binDeg\":{AOI_BIN_DEG},\"front\":{},\"back\":{}}}", side_json(0), side_json(1))
    }
}
