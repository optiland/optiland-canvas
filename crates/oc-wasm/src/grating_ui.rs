//! The editor's handle on diffraction gratings.
//!
//! Thin, like the other element files: `oc_core::grating` is the physics and
//! `Scene3::add_grating` / `set_grating` the edits. Each call here makes one
//! edit and bumps the revision; reads come back as JSON.

use oc_core::scene3::{SpectralScalar, SurfaceResponse};
use oc_core::vec3::Vec3;
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::Canvas;

/// A spectral scalar as the editor shows it: its value at 550 nm, and the
/// table when there is one.
fn scalar_json(v: &SpectralScalar) -> (serde_json::Value, serde_json::Value) {
    match v {
        SpectralScalar::Constant(x) => (json!(x), serde_json::Value::Null),
        SpectralScalar::Tabulated {
            lambda_um, values, ..
        } => (
            json!(v.sample(0.55)),
            json!({ "lambdaUm": lambda_um, "values": values }),
        ),
    }
}

#[wasm_bindgen]
impl Canvas {
    /// Add a plane diffraction grating at `(x, y)`: `width` across the grooves
    /// by `height` along them, tilted `tilt_deg` in the meridional plane, its
    /// grooves out of that plane so it disperses in it. `transmissive` makes a
    /// transmission grating. Returns the new id, or -1 with no 3D scene.
    #[wasm_bindgen(js_name = addGrating)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_grating(
        &mut self,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        tilt_deg: f64,
        lines_per_mm: f64,
        transmissive: bool,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let id = scene.add_grating(
            Vec3::new(x, y, 0.0),
            width,
            height,
            tilt_deg,
            lines_per_mm.clamp(1.0, 20_000.0),
            transmissive,
        );
        self.bump();
        id as f64
    }

    /// A grating's recipe: `{linesPerMm, order, transmissive, efficiency,
    /// zeroOrder, efficiencyTable, zeroOrderTable}`, the two efficiencies at
    /// 550 nm and their tables when they are tabulated (else `null`). `null`
    /// if `id` is not a grating.
    #[wasm_bindgen(js_name = gratingJSON)]
    pub fn grating_json(&self, id: f64) -> String {
        let Some((front, _)) = self
            .scene3
            .as_ref()
            .and_then(|s| s.surface_responses(id as u64))
        else {
            return "null".into();
        };
        let SurfaceResponse::Grating {
            lines_per_mm,
            order,
            efficiency,
            zero_order,
            transmissive,
            ..
        } = front
        else {
            return "null".into();
        };
        let (eta, eta_table) = scalar_json(efficiency);
        let (zero, zero_table) = scalar_json(zero_order);
        json!({
            "linesPerMm": lines_per_mm,
            "order": order,
            "transmissive": transmissive,
            "efficiency": eta,
            "zeroOrder": zero,
            "efficiencyTable": eta_table,
            "zeroOrderTable": zero_table,
        })
        .to_string()
    }

    /// Change a grating's groove density, order and kind, keeping its
    /// efficiencies. False if `id` is not a grating or the values are out of
    /// range.
    #[wasm_bindgen(js_name = setGrating)]
    pub fn set_grating(&mut self, id: f64, lines_per_mm: f64, order: i32, transmissive: bool) -> bool {
        let ok = self
            .scene3
            .as_mut()
            .is_some_and(|s| s.set_grating(id as u64, lines_per_mm, order, transmissive, None, None));
        if ok {
            self.bump();
        }
        ok
    }

    /// Set a grating's efficiencies as constants: `efficiency` into its order,
    /// `zeroOrder` into the zero order, each a power fraction (together at
    /// most 1). A tabulated curve they replace is gone. NaN keeps that one.
    #[wasm_bindgen(js_name = setGratingEfficiency)]
    pub fn set_grating_efficiency(&mut self, id: f64, efficiency: f64, zero_order: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let Some((SurfaceResponse::Grating {
            lines_per_mm,
            order,
            transmissive,
            ..
        }, _)) = scene.surface_responses(id as u64)
        else {
            return false;
        };
        let (l, o, t) = (*lines_per_mm, *order, *transmissive);
        let keep = |v: f64| (!v.is_nan()).then_some(v);
        let ok = scene.set_grating(id as u64, l, o, t, keep(efficiency), keep(zero_order));
        if ok {
            self.bump();
        }
        ok
    }
}

#[cfg(test)]
mod tests {
    use crate::Canvas;

    #[test]
    fn a_grating_is_added_read_and_edited() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.add_grating(40.0, 0.0, 30.0, 20.0, 180.0, 600.0, false);
        assert!(id >= 0.0);
        let g: serde_json::Value = serde_json::from_str(&c.grating_json(id)).unwrap();
        assert_eq!(g["linesPerMm"], 600.0);
        assert_eq!(g["order"], 1);
        assert_eq!(g["transmissive"], false);
        assert!(g["efficiencyTable"].is_null());

        assert!(c.set_grating(id, 1200.0, -1, true));
        let g: serde_json::Value = serde_json::from_str(&c.grating_json(id)).unwrap();
        assert_eq!((g["linesPerMm"].as_f64(), g["order"].as_i64(), g["transmissive"].as_bool()), (Some(1200.0), Some(-1), Some(true)));
        // Order 0 is the zero order already, and more than all of the light
        // is not a grating.
        assert!(!c.set_grating(id, 1200.0, 0, true));
        assert!(!c.set_grating_efficiency(id, 0.8, 0.3));
        assert!(c.set_grating_efficiency(id, 0.8, f64::NAN));
        let g: serde_json::Value = serde_json::from_str(&c.grating_json(id)).unwrap();
        assert_eq!(g["efficiency"], 0.8);
        assert_eq!(g["zeroOrder"], 0.1);
        // Not a grating: null.
        assert_eq!(c.grating_json(-5.0), "null");
    }
}
