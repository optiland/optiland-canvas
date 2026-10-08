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
    /// The film catalogue: `{builtin: [{key, material}], scene: [{id,
    /// material}]}`. Each material carries its curves or formula, its form
    /// (film, bulk, crystal), its provenance (source, paper, url, licence,
    /// range) and its nominal index, in the document's own form.
    #[wasm_bindgen(js_name = filmCatalogJSON)]
    pub fn film_catalog_json(&self) -> String {
        let builtin: Vec<String> = film_catalog::builtin()
            .into_iter()
            .map(|(key, m)| format!("{{\"key\":\"{key}\",\"material\":{}}}", material_json(&m)))
            .collect();
        let scene: Vec<String> = self
            .scene3
            .as_ref()
            .map(|s| {
                s.film_catalog
                    .iter()
                    .map(|e| format!("{{\"id\":{},\"material\":{}}}", e.id, material_json(&e.material)))
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
