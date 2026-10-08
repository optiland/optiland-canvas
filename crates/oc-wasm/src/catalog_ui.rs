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
