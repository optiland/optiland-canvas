//! The Fresnel lens for the browser: add one, read its recipe back, rebuild it
//! from an edited recipe. The geometry is `oc_core::fresnel_lens`; this is only
//! the JS surface over it.

use oc_core::fresnel_lens::{FresnelLens1, PMMA_ND_VD};
use oc_core::material::{Cauchy, Material, LAMBDA_D};
use oc_core::profile2::Curve2;
use oc_core::scene3::SolidGeometry3;
use oc_core::vec3::Vec3;
use wasm_bindgen::prelude::*;

use crate::Canvas;

#[wasm_bindgen]
impl Canvas {
    /// Add a Fresnel lens at `(x, y)`, its axis turned `tilt_deg` from `+X`:
    /// PMMA, an R 50 base, 25 mm clear radius, 1 mm pitch, 2 mm substrate,
    /// grooves facing `+X` (about a 100 mm focal length). Returns the new body
    /// id, or `-1` with no scene. Bumps the revision.
    #[wasm_bindgen(js_name = addFresnelLens)]
    pub fn add_fresnel_lens(&mut self, x: f64, y: f64, tilt_deg: f64) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let (nd, vd) = PMMA_ND_VD;
        match scene.add_fresnel_lens(
            Vec3::new(x, y, 0.0),
            tilt_deg,
            &FresnelLens1::default(),
            Material::AbbePoint { nd, vd },
        ) {
            Ok(id) => {
                self.revision += 1;
                id as f64
            }
            Err(_) => -1.0,
        }
    }

    /// A Fresnel lens's recipe and what it makes, or `null` if `id` is not one.
    ///
    /// `{recipe, nd, vd, n, efl, zones, outline}`: `recipe` is the
    /// `FresnelLens1` as serde writes it (the form [`Canvas::set_fresnel_lens`]
    /// reads), `nd`/`vd` the glass when it is an Abbe point, `n` its index at
    /// the d line, `efl` the paraxial `R / (n - 1)` (`null` for a flat base),
    /// `zones` the groove count, and `outline` the closed meridional profile as
    /// `[x, y, ...]` in the lens's local frame, sampled in loop order.
    #[wasm_bindgen(js_name = fresnelLensJSON)]
    pub fn fresnel_lens_json(&self, id: f64) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        let Some(lens) = scene.fresnel_lens(id as u64) else {
            return "null".into();
        };
        let Some(body) = scene.bodies.iter().find(|b| b.common.id.0 == id as u64) else {
            return "null".into();
        };
        let SolidGeometry3::Profile(p) = &body.geometry else {
            return "null".into();
        };
        let num = |v: f64| {
            if v.is_finite() {
                format!("{v}")
            } else {
                "null".into()
            }
        };
        let (nd, vd) = match body.material {
            Material::AbbePoint { nd, vd } => (nd, vd),
            _ => (f64::NAN, f64::NAN),
        };
        let n = body
            .material
            .index_at(LAMBDA_D, &scene.catalog)
            .unwrap_or_else(|| Cauchy::from_abbe(nd, vd).index(LAMBDA_D));
        let efl = lens.paraxial_efl_mm(n).unwrap_or(f64::NAN);
        let zones = 1 + p
            .profile
            .edges
            .iter()
            .filter(|e| matches!(e.curve, Curve2::SagSegment { y0, y1, .. } if y0 + y1 > 0.0))
            .count();
        let mut outline: Vec<String> = Vec::new();
        for e in &p.profile.edges {
            let pts: Vec<oc_core::vec::Vec2> = match &e.curve {
                Curve2::Line { a, b } => vec![*a, *b],
                Curve2::SagSegment {
                    base,
                    coeffs,
                    y0,
                    y1,
                } => (0..=6)
                    .map(|i| {
                        let t = y0 + (y1 - y0) * i as f64 / 6.0;
                        oc_core::geom3::profile::sag_segment_point(base, coeffs, t)
                    })
                    .collect(),
                Curve2::Conic(c) | Curve2::EvenAsphere { base: c, .. } => {
                    let coeffs = match &e.curve {
                        Curve2::EvenAsphere { coeffs, .. } => *coeffs,
                        _ => [0.0; 8],
                    };
                    // The loop runs from the upper facets to the lower ones.
                    (0..=24)
                        .map(|i| {
                            let t = c.half_ap - 2.0 * c.half_ap * i as f64 / 24.0;
                            oc_core::geom3::profile::sag_segment_point(c, &coeffs, t)
                        })
                        .collect()
                }
                _ => Vec::new(),
            };
            for q in pts {
                outline.push(num(q.x));
                outline.push(num(q.y));
            }
        }
        let recipe = serde_json::to_string(&lens).unwrap_or_else(|_| "null".into());
        format!(
            "{{\"recipe\":{recipe},\"nd\":{},\"vd\":{},\"n\":{},\"efl\":{},\"zones\":{zones},\"outline\":[{}]}}",
            num(nd),
            num(vd),
            num(n),
            num(efl),
            outline.join(",")
        )
    }

    /// Rebuild a Fresnel lens from an edited recipe (the `recipe` object of
    /// [`Canvas::fresnel_lens_json`], as JSON). Returns `""` on success, which
    /// bumps the revision, or why the recipe was refused, with the scene
    /// unchanged.
    #[wasm_bindgen(js_name = setFresnelLens)]
    pub fn set_fresnel_lens(&mut self, id: f64, recipe_json: &str) -> String {
        let Some(scene) = self.scene3.as_mut() else {
            return "no scene".into();
        };
        let lens: FresnelLens1 = match serde_json::from_str(recipe_json) {
            Ok(l) => l,
            Err(e) => return format!("not a Fresnel recipe: {e}"),
        };
        match scene.set_fresnel_lens(id as u64, &lens) {
            Ok(()) => {
                self.revision += 1;
                String::new()
            }
            Err(why) => why,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresnel_lens_round_trips_through_its_recipe_json() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.add_fresnel_lens(10.0, 0.0, 0.0);
        assert!(id >= 0.0);
        let info: serde_json::Value = serde_json::from_str(&c.fresnel_lens_json(id)).unwrap();
        assert_eq!(info["zones"], 25);
        assert!((info["efl"].as_f64().unwrap() - 101.69).abs() < 0.1);
        let mut recipe = info["recipe"].clone();
        recipe["grooves"] = serde_json::json!({ "ConstantPitch": { "pitch_mm": 0.5 } });
        assert_eq!(c.set_fresnel_lens(id, &recipe.to_string()), "");
        let info: serde_json::Value = serde_json::from_str(&c.fresnel_lens_json(id)).unwrap();
        assert_eq!(info["zones"], 50);
        recipe["clear_radius_mm"] = serde_json::json!(-1.0);
        assert!(!c.set_fresnel_lens(id, &recipe.to_string()).is_empty());
        assert_eq!(c.fresnel_lens_json(0.0), "null");
    }
}
