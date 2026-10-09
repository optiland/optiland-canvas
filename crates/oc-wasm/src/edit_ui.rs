//! The card's handle on aperture shapes and on links between elements.
//!
//! Thin, like `solar_ui`: the rules live in `oc_core` (`Scene3::set_aperture_shape`,
//! `oc_core::link`), and each call here parses what the page sends, makes one
//! edit, and bumps the revision (which re-resolves every link). Sizes cross
//! this boundary as diameters and full widths, the way a drawing gives them;
//! the scene stores radii and half widths.

use oc_core::link::LinkMode;
use oc_core::scene3::PlanarShape2;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use crate::Canvas;

fn outline_json(s: &PlanarShape2) -> Value {
    match s {
        PlanarShape2::Disc { radius_mm } => json!({ "kind": "circle", "d": 2.0 * radius_mm }),
        PlanarShape2::Annulus { inner_mm, outer_mm } => {
            json!({ "kind": "ring", "dIn": 2.0 * inner_mm, "d": 2.0 * outer_mm })
        }
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => json!({ "kind": "rect", "w": 2.0 * half_width_mm, "h": 2.0 * half_height_mm }),
        PlanarShape2::Polygon { .. } => json!({ "kind": "polygon" }),
    }
}

/// An outline from the page's `{kind, d, dIn, w, h}`. `Ok(None)` is the
/// unbounded mask.
fn outline_from(v: &Value) -> Result<Option<PlanarShape2>, String> {
    let num = |k: &str| {
        v.get(k)
            .and_then(Value::as_f64)
            .filter(|x| x.is_finite() && *x > 0.0)
            .ok_or_else(|| format!("{k} must be a positive size"))
    };
    Ok(Some(match v.get("kind").and_then(Value::as_str) {
        Some("circle") => PlanarShape2::Disc {
            radius_mm: 0.5 * num("d")?,
        },
        Some("ring") => {
            let (a, b) = (0.5 * num("dIn")?, 0.5 * num("d")?);
            if a >= b {
                return Err("a ring's inner diameter must be below its outer one".into());
            }
            PlanarShape2::Annulus {
                inner_mm: a,
                outer_mm: b,
            }
        }
        Some("rect") => PlanarShape2::Rectangle {
            half_width_mm: 0.5 * num("w")?,
            half_height_mm: 0.5 * num("h")?,
        },
        Some("none") => return Ok(None),
        other => return Err(format!("unknown outline {other:?}")),
    }))
}

#[wasm_bindgen]
impl Canvas {
    /// An aperture's opening and mask, or `null` if `id` is not one:
    /// `{opening, mask}`, each `{kind: "circle", d}`, `{kind: "ring", dIn, d}`
    /// or `{kind: "rect", w, h}` in mm (full sizes); the mask may also be
    /// `{kind: "none"}`, unbounded in its plane.
    #[wasm_bindgen(js_name = apertureShapeJSON)]
    pub fn aperture_shape_json(&self, id: f64) -> String {
        let Some((opening, mask)) = self
            .scene3
            .as_ref()
            .and_then(|s| s.aperture_shape(id as u64))
        else {
            return "null".into();
        };
        json!({
            "opening": outline_json(&opening),
            "mask": mask.as_ref().map_or(json!({ "kind": "none" }), outline_json),
        })
        .to_string()
    }

    /// Give an aperture a new opening and mask, in the form
    /// [`Canvas::aperture_shape_json`] reads. `""` on success (the revision
    /// bumps), or why it was refused, with the scene unchanged.
    #[wasm_bindgen(js_name = setApertureShape)]
    pub fn set_aperture_shape(&mut self, id: f64, json: &str) -> String {
        let Some(scene) = self.scene3.as_mut() else {
            return "no scene".into();
        };
        let v: Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => return format!("not an aperture shape: {e}"),
        };
        let opening = match v.get("opening").map(outline_from) {
            Some(Ok(Some(o))) => o,
            Some(Ok(None)) => return "the opening needs a size".into(),
            Some(Err(e)) => return e,
            None => return "no opening".into(),
        };
        let mask = match v.get("mask").map(outline_from) {
            Some(Ok(m)) => m,
            Some(Err(e)) => return e,
            None => return "no mask".into(),
        };
        match scene.set_aperture_shape(id as u64, opening, mask) {
            Ok(()) => {
                self.bump();
                String::new()
            }
            Err(e) => e,
        }
    }

    /// Lock element `child` to element `parent`: `mode` is `"follow"`
    /// (rigidly attached where it is), `"front"` (parallel to the parent on
    /// its axis, `gap_mm` in front) or `"face"` (turns to point at it). `""`
    /// on success (the revision bumps), or why it was refused, with the scene
    /// unchanged.
    #[wasm_bindgen(js_name = linkObject)]
    pub fn link_object(&mut self, child: f64, parent: f64, mode: &str, gap_mm: f64) -> String {
        let Some(scene) = self.scene3.as_mut() else {
            return "no scene".into();
        };
        let Some(mode) = LinkMode::parse(mode, gap_mm) else {
            return format!("unknown link mode {mode:?}");
        };
        match scene.link_object(child as u64, parent as u64, mode) {
            Ok(()) => {
                self.bump();
                String::new()
            }
            Err(e) => e,
        }
    }

    /// Release `child` from its link, where it stands. False if it had none.
    #[wasm_bindgen(js_name = unlinkObject)]
    pub fn unlink_object(&mut self, child: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let done = scene.unlink_object(child as u64);
        if done {
            self.bump();
        }
        done
    }

    /// `{parent, mode, gap}` for an element locked to another (`gap` in mm,
    /// `null` unless the mode is `front`), else `null`.
    #[wasm_bindgen(js_name = linkJSON)]
    pub fn link_json(&self, child: f64) -> String {
        match self.scene3.as_ref().and_then(|s| s.link_of(child as u64)) {
            Some(l) => {
                let gap = match l.mode {
                    LinkMode::Front { gap_mm } => json!(gap_mm),
                    _ => Value::Null,
                };
                json!({ "parent": l.parent.0, "mode": l.mode.name(), "gap": gap }).to_string()
            }
            None => "null".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_aperture_takes_any_opening_in_any_mask_that_holds_it() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.add_aperture(-40.0, 0.0, 5.0, 0.0);
        let rect =
            r#"{"opening":{"kind":"rect","w":30,"h":10},"mask":{"kind":"rect","w":200,"h":60}}"#;
        assert_eq!(c.set_aperture_shape(id, rect), "");
        let back: Value = serde_json::from_str(&c.aperture_shape_json(id)).unwrap();
        assert_eq!(
            back["opening"],
            json!({ "kind": "rect", "w": 30.0, "h": 10.0 })
        );
        assert_eq!(
            back["mask"],
            json!({ "kind": "rect", "w": 200.0, "h": 60.0 })
        );
        // A ring in an unbounded mask.
        let ring = r#"{"opening":{"kind":"ring","dIn":4,"d":20},"mask":{"kind":"none"}}"#;
        assert_eq!(c.set_aperture_shape(id, ring), "");
        // Too big for its mask: refused, and nothing changes.
        let big = r#"{"opening":{"kind":"circle","d":80},"mask":{"kind":"circle","d":60}}"#;
        assert!(!c.set_aperture_shape(id, big).is_empty());
        let still: Value = serde_json::from_str(&c.aperture_shape_json(id)).unwrap();
        assert_eq!(still["opening"]["kind"], "ring");
        assert_eq!(c.aperture_shape_json(-7.0), "null");
    }

    #[test]
    fn a_linked_element_follows_through_an_edit() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let ap = c.add_aperture(-40.0, 0.0, 5.0, 0.0);
        let lens = first_body(&c);
        assert_eq!(c.link_object(ap, lens, "follow", 0.0), "");
        assert_eq!(
            c.link_json(ap),
            json!({ "parent": lens as u64, "mode": "follow", "gap": null }).to_string()
        );
        c.set_object_position(lens, 10.0, 5.0, 0.0);
        let ap_pos = c.scene3.as_ref().unwrap().apertures[0].common.transform.pos;
        assert!(
            (ap_pos.x + 30.0).abs() < 1e-9 && (ap_pos.y - 5.0).abs() < 1e-9,
            "{ap_pos:?}"
        );
        assert!(c.unlink_object(ap));
        assert_eq!(c.link_json(ap), "null");
        assert!(!c.link_object(ap, ap, "follow", 0.0).is_empty());
        assert!(!c.link_object(ap, lens, "sideways", 0.0).is_empty());
        // In front of the lens, 12 mm out along its axis.
        assert_eq!(c.link_object(ap, lens, "front", 12.0), "");
        let v: Value = serde_json::from_str(&c.link_json(ap)).unwrap();
        assert_eq!(v["gap"], json!(12.0));
    }

    fn first_body(c: &Canvas) -> f64 {
        c.scene3.as_ref().unwrap().bodies[0].common.id.0 as f64
    }
}
