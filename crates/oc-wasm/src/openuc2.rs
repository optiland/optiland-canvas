//! openUC2 bindings (§15.2).
//!
//! Coarse project operations only. JS says "place this module at that cell" and
//! asks for segments back; it never builds module geometry, and it never holds
//! the instance list. The project lives in Rust (§0.3), and this file is a thin
//! seam over [`oc_openuc2`] — no logic that belongs in the adapter leaks up here.
//!
//! Views cross the boundary as JSON strings. They are **read-only derived data**
//! for rendering a list or a badge: the browser may hold them as ephemeral UI
//! state, but the moment it wants to know something authoritative it asks again.
//!
//! Tracing reuses the same `colored_segments` and reference tracer the freeform
//! Canvas uses. Both modes therefore draw rays through one code path, and the
//! rule that displayed numbers come from f64 holds here for free (§4.1).

use oc_fit::sample_edge;
use oc_openuc2::{
    generator::{Generator, SingletParams},
    Catalog, GridProfile, ModuleRef, OpenUc2Project, Quadrant, SceneObjectId,
};
use wasm_bindgen::prelude::*;

use crate::{auto_max_depth, colored_segments};
use oc_core::{Material, Vec2};
use oc_trace::{trace as ref_trace, TraceConfig};

/// The openUC2 mode's handle.
///
/// Holds the catalogue and, once started, the project. Kept separate from
/// [`crate::Canvas`] so the freeform route is untouched by any of this.
#[wasm_bindgen]
pub struct OpenUc2 {
    catalog: Catalog,
    project: Option<OpenUc2Project>,
    ghosts: bool,
    max_ghost: u32,
}

impl Default for OpenUc2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Report an adapter error to JS as a structured value rather than a string, so
/// the UI can say what went wrong instead of printing a Rust `Debug`.
fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// What an element is made of, for the overlay to colour by.
///
/// Presentation only: how a shape is drawn never decides how it traces.
fn material_name(m: &Material) -> &'static str {
    match m {
        Material::Mirror { .. } => "mirror",
        Material::Absorber => "absorber",
        Material::Catalog(_) => "glass",
        _ => "other",
    }
}

#[wasm_bindgen]
impl OpenUc2 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> OpenUc2 {
        OpenUc2 {
            catalog: Catalog::default(),
            project: None,
            ghosts: false,
            max_ghost: 2,
        }
    }

    /// Load a generated `catalog.json`.
    ///
    /// The document arrives already fetched: this crate never makes a network
    /// request, so there is no way for it to quietly pull a newer revision
    /// (§5.4).
    #[wasm_bindgen(js_name = loadCatalog)]
    pub fn load_catalog(&mut self, json: &str) -> Result<(), JsValue> {
        self.catalog = Catalog::from_json(json).map_err(js_err)?;
        self.project = None;
        Ok(())
    }

    /// Attach one fragment asset's `.ocanvas` text.
    #[wasm_bindgen(js_name = attachFragment)]
    pub fn attach_fragment(&mut self, asset_path: &str, ocanvas: &str) -> Result<(), JsValue> {
        self.catalog
            .attach_fragment(asset_path, ocanvas)
            .map_err(js_err)
    }

    /// Start an empty project on a catalogue grid profile.
    #[wasm_bindgen(js_name = newProject)]
    pub fn new_project(&mut self, grid_profile_id: &str) -> Result<(), JsValue> {
        let grid = self
            .catalog
            .grid_profile(grid_profile_id)
            .cloned()
            .unwrap_or_else(GridProfile::openuc2_core_50);
        self.project =
            Some(OpenUc2Project::new(self.catalog.clone(), grid).map_err(js_err)?);
        Ok(())
    }

    /// Open a `.uc2canvas` document.
    #[wasm_bindgen(js_name = loadSetup)]
    pub fn load_setup(&mut self, json: &str) -> Result<(), JsValue> {
        self.project = Some(OpenUc2Project::load(self.catalog.clone(), json).map_err(js_err)?);
        Ok(())
    }

    /// Serialize the setup.
    #[wasm_bindgen(js_name = saveSetup)]
    pub fn save_setup(&self) -> String {
        self.project.as_ref().map(|p| p.save()).unwrap_or_default()
    }

    fn project_mut(&mut self) -> Result<&mut OpenUc2Project, JsValue> {
        self.project
            .as_mut()
            .ok_or_else(|| JsValue::from_str("no project: call newProject first"))
    }

    /// Place a catalogue module. Returns the new instance id.
    #[wasm_bindgen(js_name = placeModule)]
    #[allow(clippy::too_many_arguments)]
    pub fn place_module(
        &mut self,
        uuid: &str,
        version: &str,
        gx: i32,
        gy: i32,
        gz: i32,
        rotation: u8,
    ) -> Result<String, JsValue> {
        let rot = Quadrant::from_quarter_turns(rotation).map_err(js_err)?;
        let module = ModuleRef::new(uuid, version);
        self.project_mut()?
            .place_module(&module, [gx, gy, gz], rot)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = moveModule)]
    pub fn move_module(&mut self, instance: &str, gx: i32, gy: i32, gz: i32) -> Result<(), JsValue> {
        self.project_mut()?
            .move_module(instance, [gx, gy, gz])
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = rotateModule)]
    pub fn rotate_module(&mut self, instance: &str, rotation: u8) -> Result<(), JsValue> {
        let rot = Quadrant::from_quarter_turns(rotation).map_err(js_err)?;
        self.project_mut()?.rotate_module(instance, rot).map_err(js_err)
    }

    #[wasm_bindgen(js_name = removeModule)]
    pub fn remove_module(&mut self, instance: &str) -> Result<(), JsValue> {
        self.project_mut()?.remove_module(instance).map_err(js_err)
    }

    #[wasm_bindgen(js_name = duplicateModule)]
    pub fn duplicate_module(
        &mut self,
        instance: &str,
        gx: i32,
        gy: i32,
        gz: i32,
    ) -> Result<String, JsValue> {
        self.project_mut()?
            .duplicate_module(instance, [gx, gy, gz])
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = setActiveLayer)]
    pub fn set_active_layer(&mut self, layer: i32) -> Result<(), JsValue> {
        self.project_mut()?.set_active_layer(layer).map_err(js_err)
    }

    // --- Generated optics (§9.5, §15.7) ------------------------------------

    /// What an instance's optics were built from, and what may be changed.
    ///
    /// `{generator, editable: [...], values: {...}, paraxial: {...}}`, or `null`
    /// for a module whose optics are a stored fragment. `editable` is the
    /// generator's own list, so the UI can only offer fields that actually reach
    /// the geometry (§15.7).
    ///
    /// `paraxial` is the lensmaker's equation for the *current* numbers, computed
    /// without the tracer. It is here so the picture can be checked against
    /// arithmetic rather than admired: if the rays cross the axis where this says
    /// they should, both are probably right.
    #[wasm_bindgen(js_name = parametersJson)]
    pub fn parameters_json(&self, instance: &str) -> String {
        let Some(p) = self.project.as_ref() else {
            return "null".into();
        };
        let (Some(g), Some(values)) = (p.generator_of(instance), p.resolved_parameters(instance))
        else {
            return "null".into();
        };
        let paraxial = match g {
            Generator::Singlet => SingletParams::from_json(&values)
                .and_then(|s| s.paraxial())
                .ok()
                .and_then(|x| serde_json::to_value(x).ok()),
        };
        let overridden: Vec<&str> = p
            .setup()
            .instance(instance)
            .map(|i| i.parameter_overrides.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        serde_json::json!({
            "generator": g.name(),
            "editable": g.parameter_names(),
            "values": values,
            "overridden": overridden,
            "paraxial": paraxial,
        })
        .to_string()
    }

    /// Change one parameter. Rebuilds the module's optics; the rays follow.
    ///
    /// The value arrives as JSON text so a number and the string `"inf"` — a
    /// plane, which JSON cannot hold as a number — travel the same way.
    #[wasm_bindgen(js_name = setParameter)]
    pub fn set_parameter(&mut self, instance: &str, key: &str, value_json: &str) -> Result<(), JsValue> {
        let value: serde_json::Value = serde_json::from_str(value_json)
            .map_err(|e| JsValue::from_str(&format!("{value_json:?} is not JSON: {e}")))?;
        self.project_mut()?
            .set_parameter_override(instance, key, value)
            .map_err(js_err)
    }

    /// Drop an override, returning the parameter to the descriptor's design.
    #[wasm_bindgen(js_name = clearParameter)]
    pub fn clear_parameter(&mut self, instance: &str, key: &str) -> Result<(), JsValue> {
        self.project_mut()?
            .clear_parameter_override(instance, key)
            .map_err(js_err)
    }

    // --- Picking and snapping (§15.6) --------------------------------------

    /// The grid cell a Canvas point falls in, on the active layer.
    ///
    /// The snap lives here because the pitch is grid-profile data (§6.4), not a
    /// constant. A browser that divided by its own 50 would be wrong the moment a
    /// setup used a different profile, and would be wrong silently.
    #[wasm_bindgen(js_name = cellAt)]
    pub fn cell_at(&self, x: f64, y: f64) -> Vec<i32> {
        match self.project.as_ref() {
            Some(p) => p.grid().cell_at(Vec2::new(x, y), p.active_layer()).to_vec(),
            None => vec![0, 0, 0],
        }
    }

    /// The instance standing on a cell, or `""` for none.
    #[wasm_bindgen(js_name = instanceAtCell)]
    pub fn instance_at_cell(&self, gx: i32, gy: i32, gz: i32) -> String {
        self.project
            .as_ref()
            .and_then(|p| p.instance_at_cell([gx, gy, gz]))
            .unwrap_or_default()
            .to_string()
    }

    /// Select the module on a cell. Returns its id, or `""` when the click landed
    /// on empty grid, which clears the selection.
    #[wasm_bindgen(js_name = selectAtCell)]
    pub fn select_at_cell(&mut self, gx: i32, gy: i32, gz: i32) -> Result<String, JsValue> {
        Ok(self
            .project_mut()?
            .select_at_cell([gx, gy, gz])
            .unwrap_or_default()
            .to_string())
    }

    /// Select a module by name, for a caller that already knows which one it
    /// means. Returns `""` if there is no such instance.
    #[wasm_bindgen(js_name = selectInstance)]
    pub fn select_instance_by_id(&mut self, instance: &str) -> Result<String, JsValue> {
        Ok(self
            .project_mut()?
            .select_instance(instance)
            .unwrap_or_default()
            .to_string())
    }

    /// The active grid profile, so the UI can draw cells at the right pitch
    /// rather than assuming the default one.
    #[wasm_bindgen(js_name = gridJson)]
    pub fn grid_json(&self) -> String {
        match self.project.as_ref() {
            Some(p) => serde_json::to_string(p.grid()).unwrap_or_else(|_| "null".into()),
            None => "null".into(),
        }
    }

    #[wasm_bindgen(js_name = selectedInstance)]
    pub fn selected_instance(&self) -> String {
        self.project
            .as_ref()
            .and_then(|p| p.selected_instance())
            .unwrap_or_default()
            .to_string()
    }

    #[wasm_bindgen(js_name = clearSelection)]
    pub fn clear_selection(&mut self) {
        if let Some(p) = self.project.as_mut() {
            p.clear_selection();
        }
    }

    /// Which instance would block moving `instance` to a cell, or `""` for none.
    ///
    /// A question, not an attempt: the drag preview asks it on every pointer move
    /// and must not move anything to find out.
    #[wasm_bindgen(js_name = blockerFor)]
    pub fn blocker_for(&self, instance: &str, gx: i32, gy: i32, gz: i32) -> String {
        self.project
            .as_ref()
            .and_then(|p| p.blocker_for(instance, [gx, gy, gz]))
            .unwrap_or_default()
    }

    /// Which instance would block a *copy* of `instance` at a cell.
    ///
    /// The original blocks here and does not in `blockerFor`: looking for
    /// somewhere to duplicate to has to ask this one, or it will offer a cell the
    /// original is standing on.
    #[wasm_bindgen(js_name = blockerForCopy)]
    pub fn blocker_for_copy(&self, instance: &str, gx: i32, gy: i32, gz: i32) -> String {
        self.project
            .as_ref()
            .and_then(|p| p.blocker_for_copy(instance, [gx, gy, gz]))
            .unwrap_or_default()
    }

    // --- History (§14.8) ---------------------------------------------------

    /// Begin one settled operation: everything until `endGesture` is one undo
    /// step. Wrap a drag in this, or every pointer move becomes its own step.
    #[wasm_bindgen(js_name = beginGesture)]
    pub fn begin_gesture(&mut self) {
        if let Some(p) = self.project.as_mut() {
            p.begin_gesture();
        }
    }

    #[wasm_bindgen(js_name = endGesture)]
    pub fn end_gesture(&mut self) {
        if let Some(p) = self.project.as_mut() {
            p.end_gesture();
        }
    }

    /// Step back one settled operation. `false` when there was nothing to undo.
    #[wasm_bindgen]
    pub fn undo(&mut self) -> Result<bool, JsValue> {
        self.project_mut()?.undo().map_err(js_err)
    }

    #[wasm_bindgen]
    pub fn redo(&mut self) -> Result<bool, JsValue> {
        self.project_mut()?.redo().map_err(js_err)
    }

    #[wasm_bindgen(js_name = canUndo)]
    pub fn can_undo(&self) -> bool {
        self.project.as_ref().is_some_and(|p| p.can_undo())
    }

    #[wasm_bindgen(js_name = canRedo)]
    pub fn can_redo(&self) -> bool {
        self.project.as_ref().is_some_and(|p| p.can_redo())
    }

    #[wasm_bindgen(js_name = activeLayer)]
    pub fn active_layer(&self) -> i32 {
        self.project.as_ref().map(|p| p.active_layer()).unwrap_or(0)
    }

    /// The flattened `.ocanvas`, for Optiland handoff.
    #[wasm_bindgen(js_name = exportFlatOcanvas)]
    pub fn export_flat_ocanvas(&self) -> Result<String, JsValue> {
        let p = self
            .project
            .as_ref()
            .ok_or_else(|| JsValue::from_str("no project"))?;
        Ok(p.export_flat_ocanvas().map_err(js_err)?.ocanvas)
    }

    /// The flattened `.ocanvas` *and* what is not faithful about it.
    ///
    /// `{ocanvas, warnings: [{kind, text}]}`. The caveats travel beside the file
    /// rather than inside it, because `.ocanvas` v1 has nowhere to put them and
    /// inventing a field would couple the generic format to openUC2 (§5.3, §12.4).
    /// Exporting without showing them would hand someone a scene whose idealized
    /// parts look like measured ones (§15.9).
    #[wasm_bindgen(js_name = exportFlatOcanvasJson)]
    pub fn export_flat_ocanvas_json(&self) -> Result<String, JsValue> {
        let p = self
            .project
            .as_ref()
            .ok_or_else(|| JsValue::from_str("no project"))?;
        let export = p.export_flat_ocanvas().map_err(js_err)?;
        let warnings: Vec<_> = export
            .warnings
            .iter()
            .map(|w| {
                let mut v = serde_json::to_value(w).unwrap_or(serde_json::Value::Null);
                if let Some(o) = v.as_object_mut() {
                    o.insert("text".into(), serde_json::Value::String(w.to_string()));
                }
                v
            })
            .collect();
        Ok(serde_json::json!({
            "ocanvas": export.ocanvas,
            "warnings": warnings,
        })
        .to_string())
    }

    /// The module list as CSV (§15.8).
    #[wasm_bindgen(js_name = bomCsv)]
    pub fn bom_csv(&self) -> String {
        self.project
            .as_ref()
            .map(|p| p.bom().to_csv())
            .unwrap_or_default()
    }

    // --- Views (read-only, derived) ----------------------------------------

    /// Every catalogue module, flattened for the library list.
    #[wasm_bindgen(js_name = catalogView)]
    pub fn catalog_view(&self) -> String {
        serde_json::to_string(&self.catalog.view()).unwrap_or_else(|_| "[]".into())
    }

    /// Catalogue provenance: where it came from and at which commit.
    ///
    /// Exposed because the UI has to be able to say so. An unpinned or fixture
    /// catalogue must not look like a real one (§5.4).
    #[wasm_bindgen(js_name = catalogSourceJson)]
    pub fn catalog_source_json(&self) -> String {
        match self.catalog.source() {
            Some(s) => serde_json::to_string(s).unwrap_or_else(|_| "null".into()),
            None => "null".into(),
        }
    }

    /// The coverage summary (§11.4): what the catalogue does and does not cover.
    #[wasm_bindgen(js_name = coverageJson)]
    pub fn coverage_json(&self) -> String {
        serde_json::to_string(&self.catalog.coverage()).unwrap_or_else(|_| "null".into())
    }

    #[wasm_bindgen(js_name = gridProfilesJson)]
    pub fn grid_profiles_json(&self) -> String {
        serde_json::to_string(self.catalog.grid_profiles()).unwrap_or_else(|_| "[]".into())
    }

    /// The placed instances, for the layout list.
    ///
    /// Derived from the project on every call rather than pushed to JS and kept:
    /// the setup is authoritative, and a mirrored copy is a copy that goes stale.
    #[wasm_bindgen(js_name = instancesView)]
    pub fn instances_view(&self) -> String {
        let Some(p) = self.project.as_ref() else {
            return "[]".into();
        };
        let rows: Vec<_> = p
            .setup()
            .instances
            .iter()
            .map(|i| {
                let entry = p.catalog().resolve(&i.module_ref());
                serde_json::json!({
                    "instance_id": i.instance_id,
                    "module_uuid": i.module_uuid,
                    "module_version": i.module_version,
                    "name": entry.map(|e| e.name.as_str()).unwrap_or(&i.fallback.name),
                    "module_id": entry.map(|e| e.id.as_str()).unwrap_or(&i.fallback.module_id),
                    "color": entry.and_then(|e| e.color.as_deref()).or(i.fallback.color.as_deref()),
                    "anchor_grid": i.placement.anchor_grid,
                    "rotation_quadrants": i.placement.rotation_quadrants,
                    "enabled": i.enabled,
                    "resolved": entry.is_some(),
                    "materialized": p.instance(&i.instance_id).is_some(),
                })
            })
            .collect();
        serde_json::to_string(&rows).unwrap_or_else(|_| "[]".into())
    }

    #[wasm_bindgen(js_name = bomJson)]
    pub fn bom_json(&self) -> String {
        match self.project.as_ref() {
            Some(p) => serde_json::to_string(&p.bom()).unwrap_or_else(|_| "null".into()),
            None => "null".into(),
        }
    }

    /// Fidelity and resolution warnings, as structured rows plus their text.
    #[wasm_bindgen(js_name = warningsJson)]
    pub fn warnings_json(&self) -> String {
        let Some(p) = self.project.as_ref() else {
            return "[]".into();
        };
        let rows: Vec<_> = p
            .warnings()
            .iter()
            .map(|w| {
                let mut v = serde_json::to_value(w).unwrap_or(serde_json::Value::Null);
                if let Some(o) = v.as_object_mut() {
                    o.insert("text".into(), serde_json::Value::String(w.to_string()));
                }
                v
            })
            .collect();
        serde_json::to_string(&rows).unwrap_or_else(|_| "[]".into())
    }

    /// The occupied grid cells, for drawing the footprint overlay.
    #[wasm_bindgen(js_name = footprintsJson)]
    pub fn footprints_json(&self) -> String {
        let Some(p) = self.project.as_ref() else {
            return "[]".into();
        };
        let grid = p.grid();
        let rows: Vec<_> = p
            .setup()
            .instances
            .iter()
            .filter(|i| i.enabled)
            .filter_map(|i| {
                let entry = p.catalog().resolve(&i.module_ref())?;
                let rot = Quadrant::from_quarter_turns(i.placement.rotation_quadrants).ok()?;
                let cells: Vec<[f64; 2]> = entry
                    .footprint
                    .offsets()
                    .map(|o| {
                        let r = rot.rotate_cell(o);
                        let c = grid.cell_origin([
                            i.placement.anchor_grid[0] + r[0],
                            i.placement.anchor_grid[1] + r[1],
                            i.placement.anchor_grid[2] + r[2],
                        ]);
                        [c.x, c.y]
                    })
                    .collect();
                Some(serde_json::json!({
                    "instance_id": i.instance_id,
                    "color": entry.color.as_deref().unwrap_or("#4a6a8a"),
                    "layer": i.placement.anchor_grid[2],
                    "active": i.placement.anchor_grid[2] == p.active_layer(),
                    "cells_mm": cells,
                    "cell_size_mm": [grid.cell_size_mm[0], grid.cell_size_mm[1]],
                }))
            })
            .collect();
        serde_json::to_string(&rows).unwrap_or_else(|_| "[]".into())
    }

    /// Every placed port and what it is or is not connected to (§14.6).
    ///
    /// Advisory: nothing here reaches the tracer. It is what lets the UI say that
    /// a cube is one cell out or mounted facing away, which no other overlay can.
    #[wasm_bindgen(js_name = portsJson)]
    pub fn ports_json(&self) -> String {
        let Some(p) = self.project.as_ref() else {
            return "[]".into();
        };
        serde_json::to_string(&p.ports()).unwrap_or_else(|_| "[]".into())
    }

    /// The optical geometry, as world-space polylines with their owners.
    ///
    /// This is the *optics* overlay (§15.5), which is a different thing from the
    /// footprint: the footprint is where the cube is, this is what the light
    /// actually meets. Only the active layer has objects at all, so this is
    /// already layer-filtered by construction.
    ///
    /// Rebuilt per call, because a drag moves the geometry every frame. It is a
    /// handful of elements at a few dozen samples each; the trace it accompanies
    /// costs far more.
    #[wasm_bindgen(js_name = opticalOverlayJson)]
    pub fn optical_overlay_json(&self) -> String {
        let Some(p) = self.project.as_ref() else {
            return "[]".into();
        };
        let scene = p.scene();
        let owner = |o: SceneObjectId| p.owner_of(o).unwrap_or_default().to_string();
        let mut rows: Vec<serde_json::Value> = Vec::new();

        for (i, e) in scene.elements.iter().enumerate() {
            // Sampled and transformed the same way the freeform route draws an
            // element, so a curved surface looks like itself rather than like the
            // chord across it.
            let mut pts: Vec<f64> = Vec::new();
            for edge in &e.boundary {
                for q in sample_edge(edge, 24) {
                    let w = e.transform.transform_point(q);
                    pts.push(w.x);
                    pts.push(w.y);
                }
            }
            if pts.len() >= 2 {
                pts.push(pts[0]);
                pts.push(pts[1]);
            }
            rows.push(serde_json::json!({
                "kind": "element",
                "id": i,
                "instance": owner(SceneObjectId::Element(i as u32)),
                "material": material_name(&e.material),
                "points": pts,
            }));
        }
        for (i, d) in scene.detectors.iter().enumerate() {
            rows.push(serde_json::json!({
                "kind": "detector",
                "id": i,
                "instance": owner(SceneObjectId::Detector(i as u32)),
                "points": [d.a.x, d.a.y, d.b.x, d.b.y],
            }));
        }
        for (i, s) in scene.sources.iter().enumerate() {
            let o = s.transform.pos;
            let dir = s.transform.rotate(Vec2::X);
            rows.push(serde_json::json!({
                "kind": "source",
                "id": i,
                "instance": owner(SceneObjectId::Source(i as u32)),
                "points": [o.x, o.y],
                "direction": [dir.x, dir.y],
            }));
        }
        serde_json::to_string(&rows).unwrap_or_else(|_| "[]".into())
    }

    // --- Rendering ---------------------------------------------------------

    /// Interactive f32 segments, for the picture. Same call the freeform Canvas
    /// makes, on the project's derived scene.
    #[wasm_bindgen(js_name = traceColored)]
    pub fn trace_colored(&self) -> Vec<f32> {
        match self.project.as_ref() {
            Some(p) => colored_segments(p.scene(), self.ghosts, self.max_ghost),
            None => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = setGhosts)]
    pub fn set_ghosts(&mut self, on: bool) {
        self.ghosts = on;
    }

    /// Reference (f64) detector readout as JSON — the only source of a displayed
    /// number (§4.1). `null` when there is no such detector.
    #[wasm_bindgen(js_name = detectorJson)]
    pub fn detector_json(&self, idx: usize) -> String {
        let Some(p) = self.project.as_ref() else {
            return "null".into();
        };
        let scene = p.scene();
        if idx >= scene.detectors.len() {
            return "null".into();
        }
        let cfg = TraceConfig {
            ghosts: self.ghosts,
            max_depth: auto_max_depth(scene, self.max_ghost, self.ghosts),
            max_ghost_bounces: self.max_ghost,
            ..TraceConfig::default()
        };
        let res = ref_trace(scene, &cfg);
        let d = &res.detectors[idx];
        if d.hits == 0 {
            return "null".into();
        }
        serde_json::json!({
            "hits": d.hits,
            "flux": d.total_flux,
            "centroid": d.flux_u / d.total_flux,
            "opl": d.flux_opl / d.total_flux,
            "bins": d.bins,
        })
        .to_string()
    }

    #[wasm_bindgen(js_name = detectorCount)]
    pub fn detector_count(&self) -> usize {
        self.project
            .as_ref()
            .map(|p| p.scene().detectors.len())
            .unwrap_or(0)
    }

    /// The derived scene, for anything that wants to read it directly.
    #[wasm_bindgen(js_name = sceneJson)]
    pub fn scene_json(&self) -> String {
        self.project
            .as_ref()
            .map(|p| p.scene().to_json())
            .unwrap_or_else(|| "null".into())
    }
}
