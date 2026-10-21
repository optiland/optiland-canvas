//! Coating authoring in the app (coating-fidelity spec CF-11): read an
//! object's stack with its report, edit it layer by layer, pick materials
//! from the catalogue, and plot s and p against wavelength or angle.
//!
//! Every edit goes through `Scene3::edit_coating`, which applies it to every
//! copy of the coating and validates the scene, undoing what it refuses. The
//! edits answer `{"ok":true}` or `{"error":"..."}`, so the editor can show
//! the reason.

use oc_core::coating_edit::{describe_design, StackEdit};
use oc_core::film_catalog;
use oc_core::thinfilm::{coating_eval, CoatingBacking, ExitFace, FilmContext, FilmLayer, FilmMaterial};
use wasm_bindgen::prelude::*;

use crate::{json_num, json_str, Canvas};

fn reply(r: Result<(), String>) -> String {
    match r {
        Ok(()) => "{\"ok\":true}".into(),
        Err(e) => format!("{{\"error\":{}}}", json_str(&e)),
    }
}

/// The built-in catalogue key a material came from, matched by name.
fn key_of(m: &FilmMaterial) -> Option<&'static str> {
    film_catalog::builtin().into_iter().find(|(_, b)| b.name == m.name).map(|(k, _)| k)
}

impl Canvas {
    fn edit(&mut self, id: f64, edit: StackEdit) -> String {
        let Some(scene) = self.scene3.as_mut() else {
            return reply(Err("no 3D scene".into()));
        };
        let r = scene.edit_coating(id as u64, &edit);
        if r.is_ok() {
            self.bump();
        }
        reply(r)
    }

    fn material(key: &str) -> Result<FilmMaterial, String> {
        film_catalog::builtin_named(key).ok_or_else(|| format!("no catalogue entry {key:?}"))
    }
}

#[wasm_bindgen]
impl Canvas {
    /// Object `id`'s coating as the stack editor shows it, or `null`:
    /// `{layers: [{name, key, n, k, thicknessNm, quarterWaves, incoherent,
    /// covers, rangeNm}], backing: {kind, substrate, thicknessMm}, design,
    /// lambdaNm, report: [...]}`. `n` and `k` are at `lambdaNm`, the design
    /// wavelength (550 nm without a design); `quarterWaves` is the layer's
    /// optical thickness there; `covers` says whether its data reach every
    /// wavelength the scene launches. `report` is
    /// `Scene3::coating_report`: the design record, the materials out of
    /// their data, and the refusals a `Strict` trace would meet.
    #[wasm_bindgen(js_name = filmStackJSON)]
    pub fn film_stack_json(&self, id: f64) -> String {
        let Some(scene) = self.scene3.as_ref() else {
            return "null".into();
        };
        let Some(c) = scene.coating_for_object(id as u64) else {
            return "null".into();
        };
        let lam = c.design.map_or(0.55, |d| d.lambda_nm / 1000.0);
        let launch = scene.launch_wavelengths_um();
        let layers: Vec<String> = c
            .layers
            .iter()
            .map(|l| {
                let n = l.material.index_at(lam).unwrap_or(oc_core::thinfilm::Cx { re: f64::NAN, im: f64::NAN });
                let range = l.material.provenance.as_ref().and_then(|p| p.range_um);
                format!(
                    "{{\"name\":{},\"key\":{},\"n\":{},\"k\":{},\"thicknessNm\":{},\"quarterWaves\":{},\"incoherent\":{},\"covers\":{},\"rangeNm\":{}}}",
                    json_str(&l.material.name),
                    key_of(&l.material).map_or("null".into(), json_str),
                    json_num(n.re),
                    json_num(n.im),
                    json_num(l.thickness_nm),
                    json_num(n.re * l.thickness_nm / (lam * 1000.0 / 4.0)),
                    l.incoherent,
                    launch.iter().all(|&w| l.material.covers(w)),
                    range.map_or("null".into(), |[a, b]| format!("[{},{}]", json_num(a * 1000.0), json_num(b * 1000.0)))
                )
            })
            .collect();
        let backing = match &c.backing {
            CoatingBacking::Medium => "{\"kind\":\"medium\",\"substrate\":null,\"thicknessMm\":null}".to_string(),
            CoatingBacking::Plate { substrate, thickness_mm, .. } => format!(
                "{{\"kind\":\"plate\",\"substrate\":{},\"thicknessMm\":{}}}",
                json_str(&substrate.name),
                json_num(*thickness_mm)
            ),
            CoatingBacking::Opaque { substrate } => {
                format!("{{\"kind\":\"opaque\",\"substrate\":{},\"thicknessMm\":null}}", json_str(&substrate.name))
            }
        };
        let report: Vec<String> = scene.coating_report(id as u64).unwrap_or_default().iter().map(|s| json_str(s)).collect();
        format!(
            "{{\"layers\":[{}],\"backing\":{},\"design\":{},\"lambdaNm\":{},\"report\":[{}]}}",
            layers.join(","),
            backing,
            c.design.as_ref().map_or("null".into(), |d| json_str(&describe_design(d))),
            json_num(lam * 1000.0),
            report.join(",")
        )
    }

    /// Insert a layer of catalogue entry `key`, `nm` thick, before layer
    /// `index` (`index` = the layer count appends).
    #[wasm_bindgen(js_name = insertCoatingLayer)]
    pub fn insert_coating_layer(&mut self, id: f64, index: usize, key: &str, nm: f64) -> String {
        match Self::material(key) {
            Ok(m) => self.edit(id, StackEdit::Insert { index, layer: FilmLayer::new(m, nm) }),
            Err(e) => reply(Err(e)),
        }
    }

    /// Remove layer `index`.
    #[wasm_bindgen(js_name = removeCoatingLayer)]
    pub fn remove_coating_layer(&mut self, id: f64, index: usize) -> String {
        self.edit(id, StackEdit::Remove { index })
    }

    /// Move layer `from` to position `to`.
    #[wasm_bindgen(js_name = moveCoatingLayer)]
    pub fn move_coating_layer(&mut self, id: f64, from: usize, to: usize) -> String {
        self.edit(id, StackEdit::Move { from, to })
    }

    /// Make layer `index` of catalogue entry `key`, keeping its thickness.
    #[wasm_bindgen(js_name = setCoatingLayerMaterial)]
    pub fn set_coating_layer_material(&mut self, id: f64, index: usize, key: &str) -> String {
        match Self::material(key) {
            Ok(material) => self.edit(id, StackEdit::Material { index, material }),
            Err(e) => reply(Err(e)),
        }
    }

    /// Set what is behind the stack: `medium` (the far side's own medium),
    /// `plate` (a plate of catalogue entry `key`, `mm` thick, bare behind) or
    /// `opaque` (a semi-infinite absorbing substrate of `key`). A body face
    /// refuses a plate: the body is its substrate.
    #[wasm_bindgen(js_name = setCoatingBacking)]
    pub fn set_coating_backing(&mut self, id: f64, kind: &str, key: &str, mm: f64) -> String {
        let backing = match kind {
            "medium" => Ok(CoatingBacking::Medium),
            "plate" => Self::material(key).map(|substrate| CoatingBacking::Plate {
                substrate,
                exit: ExitFace::Bare,
                thickness_mm: if mm.is_finite() { mm } else { 0.0 },
                round_trips: Default::default(),
            }),
            "opaque" => Self::material(key).map(|substrate| CoatingBacking::Opaque { substrate }),
            other => Err(format!("no backing {other:?}: medium, plate or opaque")),
        };
        match backing {
            Ok(b) => self.edit(id, StackEdit::Backing(b)),
            Err(e) => reply(Err(e)),
        }
    }

    /// s and p reflectance and transmittance of object `id`'s coating (or
    /// measured table) at each pair `(nm[i], deg[i])`, from the front or the
    /// back: `[Rs, Rp, Ts, Tp, ...]`, `NaN` where it has no answer (a table
    /// outside its data, a refused stack). The same arrays plot a sweep in
    /// wavelength at one angle or in angle at one wavelength. Under the
    /// scene's own profile and temperature.
    #[wasm_bindgen(js_name = coatingSpF64)]
    pub fn coating_sp_f64(&self, id: f64, back: bool, nm: &[f64], deg: &[f64]) -> Vec<f64> {
        let Some(scene) = self.scene3.as_ref() else {
            return Vec::new();
        };
        let id = id as u64;
        let n = nm.len().min(deg.len());
        let mut out = Vec::with_capacity(4 * n);
        if let Some(t) = scene.object_table(id) {
            for i in 0..n {
                let um = nm[i] / 1000.0;
                match t.outside(um, deg[i]) {
                    Some(_) => out.extend([f64::NAN; 4]),
                    None => {
                        let p = t.powers_clamped(um, deg[i].to_radians().cos());
                        out.extend([p.rs, p.rp, p.ts, p.tp]);
                    }
                }
            }
            return out;
        }
        let Some(c) = scene.coating_for_object(id) else {
            return Vec::new();
        };
        let ctx = FilmContext::from_profile(&scene.fidelity).at_temperature(&scene.fidelity, scene.film_temperature_c(id));
        let is_surface = scene.surfaces.iter().any(|s| s.common.id.0 == id);
        let behind = c.seen_from_behind();
        for i in 0..n {
            let um = nm[i] / 1000.0;
            let (n0, nf) = if is_surface {
                (scene.env.ambient_index, scene.env.ambient_index)
            } else {
                scene.coated_face_media(id, um)
            };
            let (coat, a, b) = if back { (&behind, nf, n0) } else { (c, n0, nf) };
            match coating_eval(coat, a, b, um, deg[i].to_radians().cos(), &ctx) {
                Ok((v, _)) => out.extend([v.r_s, v.r_p, v.t_s, v.t_p]),
                Err(_) => out.extend([f64::NAN; 4]),
            }
        }
        out
    }

    /// Every distinct wavelength the scene's sources launch, nm, sorted.
    #[wasm_bindgen(js_name = sceneWavelengthsJSON)]
    pub fn scene_wavelengths_json(&self) -> String {
        let v: Vec<String> = self
            .scene3
            .as_ref()
            .map(|s| s.launch_wavelengths_um())
            .unwrap_or_default()
            .iter()
            .map(|l| json_num(l * 1000.0))
            .collect();
        format!("[{}]", v.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer edit through the app's calls changes the curve, survives a v3
    /// save and load, and a refused edit leaves the stack as it was.
    #[test]
    fn a_stack_edit_changes_the_curve_and_survives_a_save() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.add_film_dichroic(-35.0, 0.0, 0.0, 15.0, 135.0, 560.0, false);
        assert!(id >= 0.0, "a film dichroic");
        let stack = |c: &Canvas| serde_json::from_str::<serde_json::Value>(&c.film_stack_json(id)).unwrap();
        let n0 = stack(&c)["layers"].as_array().unwrap().len();
        let nm = [500.0, 560.0, 620.0];
        let deg = [45.0; 3];
        let before = c.coating_sp_f64(id, false, &nm, &deg);
        assert_eq!(c.insert_coating_layer(id, 0, "magnesium_fluoride_bulk", 100.0), r#"{"ok":true}"#);
        let after = c.coating_sp_f64(id, false, &nm, &deg);
        assert_ne!(before, after);
        assert_eq!(stack(&c)["layers"].as_array().unwrap().len(), n0 + 1);
        let saved = c.scene3_json();
        c.load_scene3_json(&saved).unwrap();
        assert_eq!(stack(&c)["layers"].as_array().unwrap().len(), n0 + 1);
        assert_eq!(c.coating_sp_f64(id, false, &nm, &deg), after);
        // A refused edit names the reason and changes nothing.
        let err = c.remove_coating_layer(id, 999);
        assert!(err.contains("there is no layer 1000"), "{err}");
        assert_eq!(c.coating_sp_f64(id, false, &nm, &deg), after);
        assert!(c.set_coating_layer_material(id, 0, "unobtainium").contains("no catalogue entry"));
    }

    /// A material with no data at a launched wavelength is named in the
    /// report, with the refusal a Strict trace would give.
    #[test]
    fn the_report_names_an_out_of_range_material() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.add_film_dichroic(-35.0, 0.0, 0.0, 15.0, 135.0, 560.0, false);
        // A 2.4 µm line: past hafnia's data, which stop at 2 µm.
        assert!(c.set_source_wavelengths(&[550.0, 2400.0], &[]));
        assert_eq!(c.set_coating_layer_material(id, 0, "hafnia"), r#"{"ok":true}"#);
        let v: serde_json::Value = serde_json::from_str(&c.film_stack_json(id)).unwrap();
        let report: Vec<String> = v["report"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
        assert!(report.iter().any(|r| r.starts_with("layer 1 (HfO2 (film)) has no data at 2400 nm")), "{report:?}");
        assert!(report.iter().any(|r| r.contains("refused at 2400 nm: layer 1 has no optical constants")), "{report:?}");
        assert_eq!(v["layers"][0]["covers"], false);
    }
}
