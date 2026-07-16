//! The project: setup plus catalogue, materialized into a Canvas scene (§14).
//!
//! # Who owns what
//!
//! The setup is authoritative for *which modules are assembled and where*. The
//! catalogue is authoritative for *what a module is*. The scene is derived from
//! those two and is regenerated, never edited alongside them (§5.1). Nothing
//! here owns ray physics.
//!
//! # Why ids need such care
//!
//! `oc-core` ids are positions in the scene's vectors, so removing an object
//! renumbers every object after it. This project holds two things keyed by those
//! ids — each instance's `InsertedFragment` and the reverse `ownership` map — so
//! a destructive scene edit that is not followed by a remap leaves an instance
//! pointing at another instance's geometry. That is not a crash; it is a module
//! that silently moves the wrong lens. Every destructive path therefore routes
//! through [`OpenUc2Project::apply_scene_id_remap`], and no other code in this
//! crate is allowed to touch ids after a removal.

use std::collections::{BTreeMap, HashMap};

use oc_core::{
    DetectorId, ElementId, IdRemap, InsertedFragment, Isometry2, ObjectIdSet, Scene, SceneFragment,
    SourceId,
};

use crate::catalog::{Catalog, Capabilities, Footprint, ModuleEntry, ModuleRef};
use crate::coordinates::{GridProfile, Quadrant};
use crate::errors::{Error, PlacementError, ProjectWarning};
use crate::history::History;
use crate::ports::{self, PlacedPort, PortSource};
use crate::setup::{InstanceFallback, OpenUc2Setup, Placement, SetupInstance};

/// A Canvas object, whichever list it lives in.
///
/// One key across all three kinds, because ownership is a property of the object
/// and not of its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum SceneObjectId {
    Element(u32),
    Source(u32),
    Detector(u32),
}

/// A placed module that has been turned into scene objects.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedInstance {
    /// The pinned module this came from.
    pub module: ModuleRef,
    /// The catalogue's id, for messages.
    pub module_id: String,
    /// Exactly what core insertion returned, in fragment order. This is the
    /// local-to-scene map, and it is what a remap must be applied to.
    pub inserted: InsertedFragment,
}

impl MaterializedInstance {
    /// Every scene object this instance owns.
    pub fn ids(&self) -> ObjectIdSet {
        self.inserted.ids()
    }
}

/// A BOM line: one product, however many times it appears.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BomLine {
    pub quantity: usize,
    pub module_id: String,
    pub name: String,
    pub version: String,
    pub vendor: String,
    pub sku: String,
    pub unit_price_eur: f64,
    pub subtotal_eur: f64,
    /// False when the catalogue could not resolve the pin, in which case the
    /// commercial fields are whatever the setup remembered.
    pub resolved: bool,
}

/// The aggregated module list.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Bom {
    pub lines: Vec<BomLine>,
    pub total_eur: f64,
    /// Custom Canvas objects are not products and are counted separately (§5.7).
    pub custom_object_count: usize,
}

/// A flattened scene, with whatever fidelity caveats apply to it.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatExport {
    /// The `.ocanvas` document.
    pub ocanvas: String,
    /// Reported alongside the file, not inside it: `.ocanvas` v1 has nowhere to
    /// put them, and inventing a field would couple the generic format to
    /// openUC2 (§5.3).
    pub warnings: Vec<ProjectWarning>,
}

/// An openUC2 project.
#[derive(Debug)]
pub struct OpenUc2Project {
    catalog: Catalog,
    setup: OpenUc2Setup,
    grid: GridProfile,
    scene: Scene,
    /// Custom geometry, kept apart from catalogue-derived geometry so it can be
    /// saved separately and kept out of the BOM.
    custom: SceneFragment,
    custom_inserted: InsertedFragment,
    instances: BTreeMap<String, MaterializedInstance>,
    ownership: HashMap<SceneObjectId, String>,
    /// The object the user picked, when they picked one.
    selected: Option<SceneObjectId>,
    /// The module that selection *means* (§14.7).
    ///
    /// Held rather than derived from `selected`, because a module is selectable
    /// when it owns no object to point at: an empty structural cube takes up a
    /// cell and can be picked, moved, and deleted like anything else, and a
    /// selection that could only name geometry could never name it.
    selected_module: Option<String>,
    warnings: Vec<ProjectWarning>,
    /// Setups, not scenes: the scene is derived, so a snapshot of the setup is a
    /// snapshot of everything (§14.8).
    history: History,
}

impl OpenUc2Project {
    /// An empty project on `grid`.
    pub fn new(catalog: Catalog, grid: GridProfile) -> Result<Self, Error> {
        let setup = OpenUc2Setup::new(grid.id.clone());
        let mut p = Self {
            catalog,
            setup,
            grid,
            scene: Scene::new(),
            custom: SceneFragment::default(),
            custom_inserted: InsertedFragment::default(),
            instances: BTreeMap::new(),
            ownership: HashMap::new(),
            selected: None,
            selected_module: None,
            warnings: Vec::new(),
            history: History::default(),
        };
        p.stamp_catalog_pin();
        p.rematerialize()?;
        Ok(p)
    }

    /// Open a `.uc2canvas` document against `catalog`.
    ///
    /// The grid profile comes from the setup and must exist in the catalogue:
    /// guessing one would silently move every module.
    pub fn load(catalog: Catalog, json: &str) -> Result<Self, Error> {
        let setup = OpenUc2Setup::from_json(json)?;
        let grid = catalog
            .grid_profile(&setup.catalog.grid_profile)
            .cloned()
            .ok_or_else(|| Error::UnknownGridProfile(setup.catalog.grid_profile.clone()))?;

        let custom = load_custom(&setup)?;
        let mut p = Self {
            catalog,
            setup,
            grid,
            scene: Scene::new(),
            custom,
            custom_inserted: InsertedFragment::default(),
            instances: BTreeMap::new(),
            ownership: HashMap::new(),
            selected: None,
            selected_module: None,
            warnings: Vec::new(),
            history: History::default(),
        };
        p.rematerialize()?;
        Ok(p)
    }

    /// Serialize the setup.
    pub fn save(&self) -> String {
        self.setup.to_json()
    }

    fn stamp_catalog_pin(&mut self) {
        if let Some(src) = self.catalog.source() {
            self.setup.catalog.repository = src.repository.clone();
            self.setup.catalog.commit = src.commit.clone();
        }
        self.setup.catalog.grid_profile = self.grid.id.clone();
    }

    // --- Reads -------------------------------------------------------------

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn setup(&self) -> &OpenUc2Setup {
        &self.setup
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn grid(&self) -> &GridProfile {
        &self.grid
    }

    pub fn warnings(&self) -> &[ProjectWarning] {
        &self.warnings
    }

    pub fn active_layer(&self) -> i32 {
        self.setup.view.active_layer
    }

    /// The instance owning `obj`, if any. Custom objects have no owner.
    pub fn owner_of(&self, obj: SceneObjectId) -> Option<&str> {
        self.ownership.get(&obj).map(|s| s.as_str())
    }

    /// The materialized form of an instance.
    pub fn instance(&self, id: &str) -> Option<&MaterializedInstance> {
        self.instances.get(id)
    }

    /// Every instance that currently has scene objects.
    pub fn materialized_ids(&self) -> impl Iterator<Item = &str> {
        self.instances.keys().map(|s| s.as_str())
    }

    // --- Selection ---------------------------------------------------------

    /// Select a Canvas object. Selecting any object owned by an instance selects
    /// the whole module (§14.7), so the useful answer is the owning instance.
    pub fn select_object(&mut self, obj: SceneObjectId) -> Option<&str> {
        self.selected = Some(obj);
        self.selected_module = self.owner_of(obj).map(str::to_string);
        self.selected_instance()
    }

    /// Select a module by name.
    ///
    /// The primitive the other two selection paths go through. Callers that
    /// already know the instance use this: addressing it by a cell instead would
    /// depend on the module standing on the cell you named it by, and a footprint
    /// is not required to include its own anchor.
    pub fn select_instance(&mut self, instance: &str) -> Option<&str> {
        if self.setup.instance(instance).is_none() {
            self.clear_selection();
            return None;
        }
        // Point at one of the module's objects when it has any, so the selection
        // still survives a removal remap the way §14.4 requires.
        self.selected = self.first_object_of(instance);
        self.selected_module = Some(instance.to_string());
        self.selected_instance()
    }

    /// Select the module on `cell`, or clear the selection when there is none.
    ///
    /// The pick path for the layout view, where a click lands on the grid rather
    /// than on a ray. Custom objects are not modules and are never hit by this.
    pub fn select_at_cell(&mut self, cell: [i32; 3]) -> Option<&str> {
        match self.instance_at_cell(cell).map(str::to_string) {
            Some(id) => self.select_instance(&id),
            None => {
                self.clear_selection();
                None
            }
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
        self.selected_module = None;
    }

    pub fn selected_object(&self) -> Option<SceneObjectId> {
        self.selected
    }

    /// The selected module.
    pub fn selected_instance(&self) -> Option<&str> {
        self.selected_module.as_deref()
    }

    /// Any one object an instance owns, in fragment order.
    fn first_object_of(&self, instance: &str) -> Option<SceneObjectId> {
        let m = self.instances.get(instance)?;
        m.inserted
            .elements
            .first()
            .map(|e| SceneObjectId::Element(e.0))
            .or_else(|| m.inserted.sources.first().map(|s| SceneObjectId::Source(s.0)))
            .or_else(|| {
                m.inserted
                    .detectors
                    .first()
                    .map(|d| SceneObjectId::Detector(d.0))
            })
    }

    /// The instance occupying a grid cell, if any.
    ///
    /// The pick path for the layout: a module is selected by clicking where it
    /// physically is, and a structural cube with no optics has to be as pickable
    /// as a lens. Selecting by scene object could never reach one, because it owns
    /// no objects to hit (§14.7).
    pub fn instance_at_cell(&self, cell: [i32; 3]) -> Option<&str> {
        self.setup
            .instances
            .iter()
            .filter(|i| i.enabled)
            .find(|i| {
                let Ok(rot) = Quadrant::from_quarter_turns(i.placement.rotation_quadrants) else {
                    return false;
                };
                Self::cells_for(&self.footprint_of(i), i.placement.anchor_grid, rot)
                    .contains(&cell)
            })
            .map(|i| i.instance_id.as_str())
    }

    /// Which instance blocks `anchor`, if a module were placed there.
    ///
    /// A query, not an attempt: a drag preview asks this every pointer move and
    /// must not mutate anything to find out. `ignore` is the instance being
    /// dragged, which never collides with itself.
    pub fn blocker_at(
        &self,
        footprint: &Footprint,
        anchor: [i32; 3],
        rot: Quadrant,
        ignore: Option<&str>,
    ) -> Option<String> {
        let cells = Self::cells_for(footprint, anchor, rot);
        let occupied = self.occupied(ignore);
        cells.iter().find_map(|c| occupied.get(c).cloned())
    }

    /// Where a placed instance could not be moved to.
    ///
    /// The same question as [`blocker_at`](Self::blocker_at) for a module that is
    /// already on the grid, so a drag does not have to re-resolve its footprint.
    /// The module does not block itself: a drag that ends where it started, or
    /// that overlaps its own old cells on the way, is not a collision.
    pub fn blocker_for(&self, instance: &str, anchor: [i32; 3]) -> Option<String> {
        self.blocker_for_instance(instance, anchor, Some(instance))
    }

    /// Where a *copy* of an instance could not go.
    ///
    /// Unlike [`blocker_for`](Self::blocker_for), the original blocks. A duplicate
    /// is a new module and inherits no right to the cells the original is standing
    /// on, so asking the move question here would offer a cell that
    /// [`duplicate_module`](Self::duplicate_module) then refuses.
    pub fn blocker_for_copy(&self, instance: &str, anchor: [i32; 3]) -> Option<String> {
        self.blocker_for_instance(instance, anchor, None)
    }

    fn blocker_for_instance(
        &self,
        instance: &str,
        anchor: [i32; 3],
        ignore: Option<&str>,
    ) -> Option<String> {
        let inst = self.setup.instance(instance)?;
        let rot = Quadrant::from_quarter_turns(inst.placement.rotation_quadrants).ok()?;
        let footprint = self.footprint_of(inst);
        self.blocker_at(&footprint, anchor, rot, ignore)
    }

    /// Every placed port, with what it is or is not connected to (§14.6).
    ///
    /// Derived on demand and advisory throughout: nothing here reaches the tracer.
    pub fn ports(&self) -> Vec<PlacedPort> {
        let sources: Vec<PortSource<'_>> = self
            .setup
            .instances
            .iter()
            .filter(|i| i.enabled)
            .filter_map(|i| {
                let entry = self.catalog.resolve(&i.module_ref())?;
                let rotation = Quadrant::from_quarter_turns(i.placement.rotation_quadrants).ok()?;
                Some(PortSource {
                    instance: i.instance_id.as_str(),
                    module_id: entry.id.as_str(),
                    ports: &entry.ports,
                    anchor: i.placement.anchor_grid,
                    rotation,
                })
            })
            .collect();
        ports::graph(&self.grid, &sources)
    }

    // --- History (§14.8) ---------------------------------------------------

    /// Begin one settled operation.
    ///
    /// Everything until [`end_gesture`](Self::end_gesture) becomes a single undo
    /// step (§15.6). For a drag, which is one `move_module` per pointer move and
    /// would otherwise bury every earlier edit.
    pub fn begin_gesture(&mut self) {
        self.history.begin_gesture();
    }

    /// Settle the open gesture. A gesture that changed nothing records nothing.
    pub fn end_gesture(&mut self) {
        // Cloned because the comparison needs the setup while the history is
        // borrowed mutably; a settle happens once per drag, not once per frame.
        let current = self.setup.clone();
        self.history.end_gesture(&current);
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Step back one settled operation. `Ok(false)` when there is nothing to undo.
    pub fn undo(&mut self) -> Result<bool, Error> {
        let Some(previous) = self.history.peek_undo().cloned() else {
            return Ok(false);
        };
        let current = self.setup.clone();
        // Restore first: `adopt` leaves the project untouched if it fails, and the
        // stacks must not move unless the state actually did.
        self.adopt(previous)?;
        self.history.commit_undo(current);
        Ok(true)
    }

    /// Step forward one undone operation.
    pub fn redo(&mut self) -> Result<bool, Error> {
        let Some(next) = self.history.peek_redo().cloned() else {
            return Ok(false);
        };
        let current = self.setup.clone();
        self.adopt(next)?;
        self.history.commit_redo(current);
        Ok(true)
    }

    /// The state a mutation is about to change, when history needs it.
    ///
    /// `None` inside a gesture that already holds its snapshot, which is what
    /// keeps a drag from cloning the setup on every pointer move.
    fn before(&self) -> Option<OpenUc2Setup> {
        if self.history.wants_before() {
            Some(self.setup.clone())
        } else {
            None
        }
    }

    /// Record a successful mutation. Called only once the edit has committed: a
    /// refused placement is not an undo step.
    fn record(&mut self, before: Option<OpenUc2Setup>) {
        if let Some(b) = before {
            self.history.record(b);
        }
    }

    /// Become `setup`, regenerating everything derived from it.
    ///
    /// Atomic. If the new state cannot be materialized the old one is restored,
    /// so a failed undo leaves a working project rather than a half-applied one.
    fn adopt(&mut self, setup: OpenUc2Setup) -> Result<(), Error> {
        let previous = std::mem::replace(&mut self.setup, setup);
        match self.reload_derived() {
            Ok(()) => Ok(()),
            Err(e) => {
                // This state materialized once already, so putting it back cannot
                // fail for a reason the first attempt did not already hit.
                self.setup = previous;
                self.reload_derived()?;
                Err(e)
            }
        }
    }

    /// Rebuild everything the setup implies: the custom fragment and the scene.
    fn reload_derived(&mut self) -> Result<(), Error> {
        self.custom = load_custom(&self.setup)?;
        self.rematerialize()
    }

    // --- Placement ---------------------------------------------------------

    /// The grid cells an instance would occupy.
    fn cells_for(footprint: &Footprint, anchor: [i32; 3], rot: Quadrant) -> Vec<[i32; 3]> {
        footprint
            .offsets()
            .map(|o| {
                let r = rot.rotate_cell(o);
                [anchor[0] + r[0], anchor[1] + r[1], anchor[2] + r[2]]
            })
            .collect()
    }

    /// Cells occupied by every enabled instance except `ignore`.
    ///
    /// Cells carry their layer in `z`, so "same cell" already means "same cell on
    /// the same layer" and no separate layer check is needed. Note this consults
    /// *every* enabled instance, not just the active layer: two modules on
    /// different layers do not collide, but two on the same inactive layer still
    /// do. Mechanical occupancy is not a function of what the tracer is looking
    /// at.
    fn occupied(&self, ignore: Option<&str>) -> BTreeMap<[i32; 3], String> {
        let mut map = BTreeMap::new();
        for inst in &self.setup.instances {
            if !inst.enabled || Some(inst.instance_id.as_str()) == ignore {
                continue;
            }
            let Ok(rot) = Quadrant::from_quarter_turns(inst.placement.rotation_quadrants) else {
                continue;
            };
            let footprint = self.footprint_of(inst);
            for c in Self::cells_for(&footprint, inst.placement.anchor_grid, rot) {
                map.insert(c, inst.instance_id.clone());
            }
        }
        map
    }

    /// The footprint to use for an instance: the catalogue's when it resolves,
    /// the setup's remembered cells when it does not.
    ///
    /// An unresolved module still occupies space. Letting a placeholder collapse
    /// to nothing would let another module be placed on top of it, and the
    /// collision would reappear the moment the catalogue came back.
    fn footprint_of(&self, inst: &SetupInstance) -> Footprint {
        match self.catalog.resolve(&inst.module_ref()) {
            Some(e) => e.footprint.clone(),
            None if !inst.fallback.footprint_cells.is_empty() => Footprint {
                anchor_cell: [0, 0, 0],
                cells: inst.fallback.footprint_cells.clone(),
            },
            None => Footprint::single(),
        }
    }

    fn check_free(&self, cells: &[[i32; 3]], ignore: Option<&str>) -> Result<(), PlacementError> {
        let occupied = self.occupied(ignore);
        for c in cells {
            if let Some(owner) = occupied.get(c) {
                return Err(PlacementError::Collision {
                    occupied_by: owner.clone(),
                    cell: *c,
                });
            }
        }
        Ok(())
    }

    /// Place a catalogue module.
    ///
    /// Atomic: resolution, rotation, and collision are all checked before the
    /// scene or the setup is touched.
    pub fn place_module(
        &mut self,
        module: &ModuleRef,
        anchor: [i32; 3],
        rot: Quadrant,
    ) -> Result<String, PlacementError> {
        let entry = self
            .catalog
            .resolve(module)
            .ok_or_else(|| PlacementError::UnknownModule {
                uuid: module.uuid.clone(),
                version: module.version.clone(),
            })?;
        if !self.grid.allows(rot) {
            return Err(PlacementError::RotationNotAllowed {
                degrees: rot.degrees(),
            });
        }

        let instance_id = self.setup.next_instance_id();
        let inst = SetupInstance {
            instance_id: instance_id.clone(),
            module_uuid: module.uuid.clone(),
            module_version: module.version.clone(),
            fallback: fallback_for(entry),
            placement: Placement {
                anchor_grid: anchor,
                rotation_quadrants: rot.quarter_turns(),
            },
            parameter_overrides: BTreeMap::new(),
            enabled: true,
        };

        let cells = Self::cells_for(&entry.footprint, anchor, rot);
        self.check_free(&cells, None)?;
        let before = self.before();

        // Materialize before committing to the setup: insertion can still fail on
        // a bad fragment, and a setup entry with no objects would be a module the
        // user can see but not select.
        if self.should_materialize(&inst) {
            let placed = self.materialize_one(&inst).map_err(PlacementError::Fragment)?;
            if let Some(m) = placed {
                self.instances.insert(instance_id.clone(), m);
            }
        }
        self.setup.instances.push(inst);
        self.rebuild_ownership();
        self.recompute_warnings();
        self.record(before);
        Ok(instance_id)
    }

    /// Move a placed module to a new anchor cell.
    ///
    /// Uses a delta transform on the instance's owned batch, so ids do not
    /// change and one drag stays one batch.
    pub fn move_module(&mut self, instance: &str, anchor: [i32; 3]) -> Result<(), PlacementError> {
        self.reposition(instance, Some(anchor), None)
    }

    /// Rotate a placed module about its anchor cell.
    pub fn rotate_module(&mut self, instance: &str, rot: Quadrant) -> Result<(), PlacementError> {
        self.reposition(instance, None, Some(rot))
    }

    fn reposition(
        &mut self,
        instance: &str,
        anchor: Option<[i32; 3]>,
        rot: Option<Quadrant>,
    ) -> Result<(), PlacementError> {
        let inst = self
            .setup
            .instance(instance)
            .ok_or_else(|| PlacementError::UnknownModule {
                uuid: instance.to_string(),
                version: String::new(),
            })?;
        let old_anchor = inst.placement.anchor_grid;
        let old_rot = Quadrant::from_quarter_turns(inst.placement.rotation_quadrants)
            .unwrap_or(Quadrant::R0);
        let new_anchor = anchor.unwrap_or(old_anchor);
        let new_rot = rot.unwrap_or(old_rot);

        if !self.grid.allows(new_rot) {
            return Err(PlacementError::RotationNotAllowed {
                degrees: new_rot.degrees(),
            });
        }

        let footprint = self.footprint_of(inst);
        let cells = Self::cells_for(&footprint, new_anchor, new_rot);
        // Ignore this instance's own cells: a module never collides with itself.
        self.check_free(&cells, Some(instance))?;
        // Inside a drag's gesture this clones nothing: the snapshot was taken on
        // the first move and the rest of the drag folds into it (§15.6).
        let before = self.before();

        let old_t = self.grid.placement(old_anchor, old_rot);
        let new_t = self.grid.placement(new_anchor, new_rot);

        // Commit the setup first, then move the objects. Both are infallible from
        // here: the batch is this instance's own, and the transform is finite.
        if let Some(i) = self.setup.instance_mut(instance) {
            i.placement.anchor_grid = new_anchor;
            i.placement.rotation_quadrants = new_rot.quarter_turns();
        }

        let layer_changed = old_anchor[2] != new_anchor[2];
        if layer_changed {
            // Crossing layers changes whether the module is in the traced plane at
            // all, so its objects must be built or dropped rather than moved.
            self.rematerialize().map_err(PlacementError::from)?;
            self.record(before);
            return Ok(());
        }

        if let Some(m) = self.instances.get(instance) {
            let delta = new_t.compose(inverse(old_t));
            let ids = m.ids();
            self.scene
                .transform_objects(&ids, delta)
                .map_err(PlacementError::Fragment)?;
        }
        self.recompute_warnings();
        self.record(before);
        Ok(())
    }

    /// Remove a module and everything it owns.
    ///
    /// This is the path §14.4 spells out, and the order matters: the scene edit
    /// renumbers survivors, so the remap has to reach every retained id before
    /// anything reads one again.
    pub fn remove_module(&mut self, instance: &str) -> Result<(), Error> {
        if self.setup.instance(instance).is_none() {
            return Err(Error::UnknownInstance(instance.to_string()));
        }
        let before = self.before();

        if let Some(m) = self.instances.remove(instance) {
            let remap = self.scene.remove_objects(&m.ids())?;
            self.apply_scene_id_remap(&remap);
        }
        self.setup.instances.retain(|i| i.instance_id != instance);
        // The module is gone, so a selection naming it is gone too. A layout-only
        // module owns no object, so the remap above could not have cleared this.
        if self.selected_module.as_deref() == Some(instance) {
            self.clear_selection();
        }
        self.rebuild_ownership();
        self.recompute_warnings();
        self.record(before);
        Ok(())
    }

    /// Copy a module to a new anchor.
    pub fn duplicate_module(&mut self, instance: &str, anchor: [i32; 3]) -> Result<String, Error> {
        let inst = self
            .setup
            .instance(instance)
            .ok_or_else(|| Error::UnknownInstance(instance.to_string()))?;
        let module = inst.module_ref();
        let rot = Quadrant::from_quarter_turns(inst.placement.rotation_quadrants)?;
        let overrides = inst.parameter_overrides.clone();

        let new_id = self.place_module(&module, anchor, rot)?;
        if let Some(i) = self.setup.instance_mut(&new_id) {
            i.parameter_overrides = overrides;
        }
        Ok(new_id)
    }

    /// Choose the traced layer.
    ///
    /// Canvas traces one horizontal plane at a time (§5.6), so this rebuilds the
    /// scene: modules leaving the active layer lose their objects and modules
    /// entering it gain them.
    pub fn set_active_layer(&mut self, layer: i32) -> Result<(), Error> {
        if self.setup.view.active_layer == layer {
            return Ok(());
        }
        // Undoable, because the active layer is part of the snapshot: it decides
        // what materializes. Undoing a placement made on another layer without
        // going back to that layer would restore a module the user cannot see.
        let before = self.before();
        self.setup.view.active_layer = layer;
        self.rematerialize()?;
        self.record(before);
        Ok(())
    }

    /// Set a parameter override.
    ///
    /// Recorded and carried into the BOM, but not yet bound to geometry: no
    /// module in the MVP regenerates its optics from a parameter, and pretending
    /// otherwise would show a lens that reads 50 mm and traces 25.
    pub fn set_parameter_override(
        &mut self,
        instance: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), Error> {
        if self.setup.instance(instance).is_none() {
            return Err(Error::UnknownInstance(instance.to_string()));
        }
        let before = self.before();
        let i = self
            .setup
            .instance_mut(instance)
            .ok_or_else(|| Error::UnknownInstance(instance.to_string()))?;
        i.parameter_overrides.insert(key.to_string(), value);
        self.record(before);
        Ok(())
    }

    /// Replace the custom (non-catalogue) Canvas content.
    ///
    /// Undoable through the same stack as module edits: §14.8 requires that
    /// custom Canvas edits and module edits never produce divergent state, and
    /// two histories over one project is exactly how that happens.
    pub fn set_custom_canvas(&mut self, ocanvas: &str) -> Result<(), Error> {
        let scene = Scene::from_json(ocanvas)
            .map_err(|e| Error::Format(format!("custom canvas is not a valid .ocanvas: {e}")))?;
        let parsed: serde_json::Value = serde_json::from_str(ocanvas)
            .map_err(|e| Error::Format(format!("custom canvas is not valid JSON: {e}")))?;
        // Both parses are done before anything is committed, so a bad document
        // cannot leave the fragment and the setup describing different content.
        let before = self.before();
        self.custom = SceneFragment::from_scene(&scene)?;
        self.setup.custom_canvas.ocanvas = parsed;
        self.rematerialize()?;
        self.record(before);
        Ok(())
    }

    // --- Materialization ---------------------------------------------------

    /// Should this instance contribute optics right now?
    fn should_materialize(&self, inst: &SetupInstance) -> bool {
        inst.enabled && inst.placement.anchor_grid[2] == self.setup.view.active_layer
    }

    /// Insert one instance's fragment. `Ok(None)` means the module resolved but
    /// has nothing optical to place, which is normal for a structural cube.
    fn materialize_one(
        &mut self,
        inst: &SetupInstance,
    ) -> Result<Option<MaterializedInstance>, oc_core::FragmentError> {
        let module = inst.module_ref();
        let Some(entry) = self.catalog.resolve(&module) else {
            return Ok(None);
        };
        let module_id = entry.id.clone();
        let Some(fragment) = self.catalog.fragment_for(entry).cloned() else {
            return Ok(None);
        };
        let rot = Quadrant::from_quarter_turns(inst.placement.rotation_quadrants)
            .unwrap_or(Quadrant::R0);
        let t = self.grid.placement(inst.placement.anchor_grid, rot);
        let inserted = self.scene.insert_fragment(&fragment, t)?;
        Ok(Some(MaterializedInstance {
            module,
            module_id,
            inserted,
        }))
    }

    /// Rebuild the scene from the setup.
    ///
    /// The scene is derived, so the honest way to change what is in it is to
    /// regenerate it. Every object id changes, so the picked *object* is dropped
    /// rather than remapped: it no longer exists in any sense that survives a
    /// rebuild. The picked *module* does survive, when the setup still has it —
    /// undo, redo, and a layer switch should not silently deselect what the user
    /// is working on.
    fn rematerialize(&mut self) -> Result<(), Error> {
        self.scene = Scene::new();
        self.instances.clear();
        self.ownership.clear();
        self.selected = None;

        // Custom content goes in first, so its ids are stable while modules come
        // and go at the end of the vectors.
        self.custom_inserted = if self.custom.is_empty() {
            InsertedFragment::default()
        } else {
            let custom = self.custom.clone();
            self.scene.insert_fragment(&custom, Isometry2::IDENTITY)?
        };

        let instances = self.setup.instances.clone();
        for inst in &instances {
            if !self.should_materialize(inst) {
                continue;
            }
            if let Some(m) = self.materialize_one(inst)? {
                self.instances.insert(inst.instance_id.clone(), m);
            }
        }
        self.rebuild_ownership();

        // Re-point the surviving module selection at whatever the rebuild gave it.
        // A module off the active layer has no objects now, which is exactly why
        // the module and the object are tracked separately.
        let module = self
            .selected_module
            .take()
            .filter(|id| self.setup.instance(id).is_some());
        self.selected = module.as_deref().and_then(|id| self.first_object_of(id));
        self.selected_module = module;

        self.recompute_warnings();
        Ok(())
    }

    /// Rebuild the reverse map from the instances' own id lists.
    ///
    /// Derived from `instances` rather than maintained beside it, so the two
    /// cannot disagree.
    fn rebuild_ownership(&mut self) {
        self.ownership.clear();
        for (id, m) in &self.instances {
            for e in &m.inserted.elements {
                self.ownership.insert(SceneObjectId::Element(e.0), id.clone());
            }
            for s in &m.inserted.sources {
                self.ownership.insert(SceneObjectId::Source(s.0), id.clone());
            }
            for d in &m.inserted.detectors {
                self.ownership.insert(SceneObjectId::Detector(d.0), id.clone());
            }
        }
    }

    /// Push every retained scene id through `remap`.
    ///
    /// The single place ids are rewritten after a destructive scene edit (§14.4).
    /// It has to cover *everything* holding an id: each surviving instance's
    /// `InsertedFragment`, the custom batch, the reverse ownership map, and the
    /// selection. Anything missed here is a stale id that will read as another
    /// instance's object.
    fn apply_scene_id_remap(&mut self, remap: &IdRemap) {
        for m in self.instances.values_mut() {
            m.inserted = remap_inserted(&m.inserted, remap);
        }
        self.custom_inserted = remap_inserted(&self.custom_inserted, remap);

        // The selection may itself have been removed, in which case it is cleared
        // rather than pointed somewhere plausible.
        self.selected = self.selected.and_then(|o| remap_object(o, remap));

        self.rebuild_ownership();
    }

    fn recompute_warnings(&mut self) {
        let mut out = Vec::new();
        let active = self.setup.view.active_layer;
        for inst in &self.setup.instances {
            if !inst.enabled {
                continue;
            }
            let id = inst.instance_id.clone();
            let Some(entry) = self.catalog.resolve(&inst.module_ref()) else {
                out.push(ProjectWarning::UnresolvedModule {
                    instance: id,
                    uuid: inst.module_uuid.clone(),
                    version: inst.module_version.clone(),
                    fallback_id: inst.fallback.module_id.clone(),
                });
                continue;
            };
            let module_id = entry.id.clone();
            let caps: Capabilities = entry.capabilities.clone();
            let declares_fragment = entry.assets.canvas_fragment.is_some();
            let missing_asset = self.catalog.fragment_asset_missing(entry).map(str::to_string);

            if inst.placement.anchor_grid[2] != active {
                out.push(ProjectWarning::InactiveLayer {
                    instance: id.clone(),
                    module_id: module_id.clone(),
                    layer: inst.placement.anchor_grid[2],
                    active_layer: active,
                });
                continue;
            }
            if let Some(asset) = missing_asset {
                out.push(ProjectWarning::MissingFragmentAsset {
                    instance: id.clone(),
                    module_id: module_id.clone(),
                    asset,
                });
                continue;
            }
            if !caps.has_optics() || !declares_fragment {
                out.push(ProjectWarning::LayoutOnly {
                    instance: id.clone(),
                    module_id: module_id.clone(),
                    optical: caps.optical.clone(),
                });
            } else if !caps.optics_validated() {
                out.push(ProjectWarning::IdealizedOptics {
                    instance: id.clone(),
                    module_id: module_id.clone(),
                    fidelity: caps.optical.clone(),
                });
            }
        }
        self.warnings = out;
    }

    // --- Products ----------------------------------------------------------

    /// Aggregate enabled instances into a module list.
    ///
    /// Keyed by `(uuid, version, parameter_overrides)`: two instances of the same
    /// product configured differently are different line items, and two versions
    /// of one product are never merged. Custom objects are counted but never
    /// priced — they are not something anyone can order (§5.7).
    pub fn bom(&self) -> Bom {
        /// What distinguishes one line item from another: the pinned product,
        /// plus its configuration.
        type LineKey = (String, String, String);
        /// How many, and where to read the product details from.
        type LineGroup<'a> = (usize, Option<&'a ModuleEntry>, &'a SetupInstance);

        let mut groups: BTreeMap<LineKey, LineGroup<'_>> = BTreeMap::new();
        for inst in &self.setup.instances {
            if !inst.enabled {
                continue;
            }
            let overrides = serde_json::to_string(&inst.parameter_overrides).unwrap_or_default();
            let key = (
                inst.module_uuid.clone(),
                inst.module_version.clone(),
                overrides,
            );
            let entry = self.catalog.resolve(&inst.module_ref());
            groups
                .entry(key)
                .and_modify(|g| g.0 += 1)
                .or_insert((1, entry, inst));
        }

        let mut lines: Vec<BomLine> = groups
            .into_values()
            .map(|(quantity, entry, inst)| match entry {
                Some(e) => {
                    let unit = e.commercial.price_eur;
                    BomLine {
                        quantity,
                        module_id: e.id.clone(),
                        name: e.name.clone(),
                        version: e.version.clone(),
                        vendor: e.commercial.vendor.clone(),
                        sku: e.commercial.sku.clone(),
                        unit_price_eur: unit,
                        subtotal_eur: unit * quantity as f64,
                        resolved: true,
                    }
                }
                None => BomLine {
                    quantity,
                    module_id: inst.fallback.module_id.clone(),
                    name: inst.fallback.name.clone(),
                    version: inst.module_version.clone(),
                    vendor: String::new(),
                    sku: String::new(),
                    unit_price_eur: 0.0,
                    subtotal_eur: 0.0,
                    resolved: false,
                },
            })
            .collect();
        lines.sort_by(|a, b| {
            (&a.module_id, &a.version).cmp(&(&b.module_id, &b.version))
        });

        let total_eur = lines.iter().map(|l| l.subtotal_eur).sum();
        Bom {
            lines,
            total_eur,
            custom_object_count: self.custom.len(),
        }
    }

    /// Flatten to an `.ocanvas` for Optiland handoff (§12.4).
    ///
    /// This is the derived optical artifact, not a save file: it carries the
    /// geometry and none of the product identity, and cannot rebuild the BOM.
    /// That is what `.uc2canvas` is for.
    pub fn export_flat_ocanvas(&self) -> Result<FlatExport, Error> {
        Ok(FlatExport {
            ocanvas: self.scene.to_json(),
            warnings: self.warnings.clone(),
        })
    }
}

/// `t` undone.
fn inverse(t: Isometry2) -> Isometry2 {
    let rot = -t.rot;
    let inv = Isometry2::new(oc_core::Vec2::ZERO, rot);
    Isometry2::new(-inv.rotate(t.pos), rot)
}

fn remap_object(o: SceneObjectId, remap: &IdRemap) -> Option<SceneObjectId> {
    match o {
        SceneObjectId::Element(i) => remap
            .element(ElementId(i))
            .map(|n| SceneObjectId::Element(n.0)),
        SceneObjectId::Source(i) => remap.source(SourceId(i)).map(|n| SceneObjectId::Source(n.0)),
        SceneObjectId::Detector(i) => remap
            .detector(DetectorId(i))
            .map(|n| SceneObjectId::Detector(n.0)),
    }
}

/// Push an insertion's ids through a remap, preserving fragment order.
///
/// Order is the local-to-scene map, so it must survive: dropping a removed id
/// would silently renumber the fragment's own locals. A surviving instance never
/// has removed objects, so nothing is actually dropped here.
fn remap_inserted(f: &InsertedFragment, remap: &IdRemap) -> InsertedFragment {
    InsertedFragment {
        elements: f.elements.iter().filter_map(|i| remap.element(*i)).collect(),
        sources: f.sources.iter().filter_map(|i| remap.source(*i)).collect(),
        detectors: f
            .detectors
            .iter()
            .filter_map(|i| remap.detector(*i))
            .collect(),
    }
}

fn fallback_for(e: &ModuleEntry) -> InstanceFallback {
    InstanceFallback {
        module_id: e.id.clone(),
        name: e.name.clone(),
        color: e.color.clone(),
        footprint_cells: e.footprint.cells.clone(),
    }
}

fn load_custom(setup: &OpenUc2Setup) -> Result<SceneFragment, Error> {
    if setup.custom_canvas.ocanvas.is_null()
        || setup.custom_canvas.ocanvas.as_object().is_some_and(|o| o.is_empty())
    {
        return Ok(SceneFragment::default());
    }
    let text = setup.custom_canvas.ocanvas.to_string();
    let scene = Scene::from_json(&text)
        .map_err(|e| Error::Format(format!("custom canvas is not a valid .ocanvas: {e}")))?;
    Ok(SceneFragment::from_scene(&scene)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

    fn project() -> OpenUc2Project {
        OpenUc2Project::new(pilot_catalog(), GridProfile::openuc2_core_50()).unwrap()
    }

    fn lens() -> ModuleRef {
        ModuleRef::new(LENS_UUID, "1.0.0")
    }
    fn cube() -> ModuleRef {
        ModuleRef::new(CUBE_UUID, "1.0.0")
    }
    fn source() -> ModuleRef {
        ModuleRef::new(SOURCE_UUID, "1.0.0")
    }
    fn mirror() -> ModuleRef {
        ModuleRef::new(MIRROR_UUID, "1.0.0")
    }

    // --- Placement ---------------------------------------------------------

    #[test]
    fn placing_a_module_materializes_its_optics() {
        let mut p = project();
        let id = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        assert_eq!(id, "m-0001");
        assert_eq!(p.scene().elements.len(), 1);
        // The lens lands at its anchor cell: grid [1,0,0] on a 50 mm pitch.
        assert_eq!(p.scene().elements[0].transform.pos, oc_core::Vec2::new(50.0, 0.0));
        assert!(p.instance(&id).is_some());
    }

    #[test]
    fn a_layout_only_module_places_without_optics() {
        // An empty cube is useful for layout and must not be rejected, but it
        // must not silently look like it does something either.
        let mut p = project();
        let id = p.place_module(&cube(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(p.scene().elements.is_empty());
        assert!(p.instance(&id).is_none(), "no objects to own");
        assert!(matches!(
            p.warnings(),
            [ProjectWarning::LayoutOnly { .. }]
        ));
        // It still occupies its cell.
        assert_eq!(
            p.place_module(&cube(), [0, 0, 0], Quadrant::R0),
            Err(PlacementError::Collision {
                occupied_by: id,
                cell: [0, 0, 0]
            })
        );
    }

    #[test]
    fn placing_an_unknown_module_is_refused() {
        let mut p = project();
        let err = p
            .place_module(&ModuleRef::new(LENS_UUID, "9.9.9"), [0, 0, 0], Quadrant::R0)
            .unwrap_err();
        assert!(matches!(err, PlacementError::UnknownModule { .. }));
        assert!(p.setup().instances.is_empty(), "placement must be atomic");
    }

    #[test]
    fn the_glass_id_survives_placement() {
        // The lens fragment carries N-BK7 from the builtin catalogue. If insertion
        // copied the id positionally the element would come out as some other
        // glass, and the trace would be quietly wrong rather than failing.
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let scene = p.scene();
        let id = match scene.elements[0].material {
            oc_core::Material::Catalog(g) => g,
            ref m => panic!("expected a catalog glass, got {m:?}"),
        };
        assert_eq!(scene.catalog.get(id).unwrap().name, "N-BK7");
    }

    // --- Collision ---------------------------------------------------------

    #[test]
    fn two_modules_cannot_share_a_cell() {
        let mut p = project();
        let first = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        let err = p.place_module(&cube(), [1, 0, 0], Quadrant::R0).unwrap_err();
        assert_eq!(
            err,
            PlacementError::Collision {
                occupied_by: first,
                cell: [1, 0, 0]
            }
        );
        assert_eq!(p.setup().instances.len(), 1, "refusal must not mutate");
        assert_eq!(p.scene().elements.len(), 1);
    }

    #[test]
    fn modules_on_different_layers_do_not_collide() {
        // Same in-plane cell, different layer: mechanically fine.
        let mut p = project();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        assert!(p.place_module(&cube(), [1, 0, 1], Quadrant::R0).is_ok());
    }

    #[test]
    fn collision_ignores_disabled_instances() {
        let mut p = project();
        let id = p.place_module(&cube(), [0, 0, 0], Quadrant::R0).unwrap();
        p.setup.instance_mut(&id).unwrap().enabled = false;
        assert!(p.place_module(&lens(), [0, 0, 0], Quadrant::R0).is_ok());
    }

    /// A catalogue whose rail module spans two cells.
    fn rail_project() -> (OpenUc2Project, ModuleRef) {
        const RAIL: &str = "22222222-2222-4222-8222-000000000001";
        let mut doc: serde_json::Value =
            serde_json::from_str(&minimal_catalog_json()).unwrap();
        doc["modules"]
            .as_array_mut()
            .unwrap()
            .push(two_cell_module_json(RAIL));
        let cat = Catalog::from_json(&doc.to_string()).unwrap();
        (
            OpenUc2Project::new(cat, GridProfile::openuc2_core_50()).unwrap(),
            ModuleRef::new(RAIL, "1.0.0"),
        )
    }

    #[test]
    fn a_multi_cell_module_occupies_its_rotated_cells() {
        // The rail spans [0,0,0] and [1,0,0]. Rotated 90° about its anchor it
        // spans [0,0,0] and [0,1,0] instead, so what it blocks changes with it.
        let (mut p, rail) = rail_project();
        p.place_module(&rail, [0, 0, 0], Quadrant::R90).unwrap();

        // The un-rotated second cell is now free...
        assert!(p.place_module(&cube(), [1, 0, 0], Quadrant::R0).is_ok());
        // ...and the rotated one is taken.
        assert!(matches!(
            p.place_module(&cube(), [0, 1, 0], Quadrant::R0),
            Err(PlacementError::Collision { .. })
        ));
    }

    #[test]
    fn a_multi_cell_module_blocks_both_cells_unrotated() {
        let (mut p, rail) = rail_project();
        p.place_module(&rail, [0, 0, 0], Quadrant::R0).unwrap();
        assert!(matches!(
            p.place_module(&cube(), [1, 0, 0], Quadrant::R0),
            Err(PlacementError::Collision { cell, .. }) if cell == [1, 0, 0]
        ));
    }

    #[test]
    fn rotating_into_an_occupied_cell_is_refused() {
        let (mut p, rail) = rail_project();
        let id = p.place_module(&rail, [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&cube(), [0, 1, 0], Quadrant::R0).unwrap();
        // Rotating the rail 90° would put its far cell on top of the cube.
        assert!(matches!(
            p.rotate_module(&id, Quadrant::R90),
            Err(PlacementError::Collision { .. })
        ));
        // And the refusal leaves the rail where it was.
        assert_eq!(
            p.setup().instance(&id).unwrap().placement.rotation_quadrants,
            0
        );
    }

    #[test]
    fn a_module_does_not_collide_with_itself() {
        let (mut p, rail) = rail_project();
        let id = p.place_module(&rail, [0, 0, 0], Quadrant::R0).unwrap();
        // Sliding one cell along its own axis overlaps its old cells.
        assert!(p.move_module(&id, [1, 0, 0]).is_ok());
    }

    // --- Move and rotate ---------------------------------------------------

    #[test]
    fn moving_a_module_moves_its_objects_and_keeps_their_ids() {
        let mut p = project();
        let id = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let before = p.instance(&id).unwrap().inserted.clone();

        p.move_module(&id, [2, 1, 0]).unwrap();

        assert_eq!(p.scene().elements[0].transform.pos, oc_core::Vec2::new(100.0, 50.0));
        assert_eq!(
            p.instance(&id).unwrap().inserted,
            before,
            "a move is a transform, not a rebuild: ids must not change"
        );
        assert_eq!(p.setup().instance(&id).unwrap().placement.anchor_grid, [2, 1, 0]);
    }

    #[test]
    fn rotating_a_module_composes_onto_its_own_pose() {
        // The mirror fragment is authored at 45°. Rotating the module 90° must
        // give 135°, not replace the module's internal pose with 90°.
        let mut p = project();
        let id = p.place_module(&mirror(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!((p.scene().elements[0].transform.rot - std::f64::consts::FRAC_PI_4).abs() < 1e-12);

        p.rotate_module(&id, Quadrant::R90).unwrap();
        let expected = std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2;
        assert!((p.scene().elements[0].transform.rot - expected).abs() < 1e-12);
    }

    #[test]
    fn a_move_and_a_fresh_placement_agree() {
        // Moving to a pose must land exactly where placing there would have.
        let mut moved = project();
        let id = moved.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        moved.move_module(&id, [3, 2, 0]).unwrap();
        moved.rotate_module(&id, Quadrant::R270).unwrap();

        let mut direct = project();
        direct.place_module(&lens(), [3, 2, 0], Quadrant::R270).unwrap();

        let a = moved.scene().elements[0].transform;
        let b = direct.scene().elements[0].transform;
        assert!((a.pos - b.pos).length() < 1e-12, "{a:?} vs {b:?}");
        assert!((a.rot - b.rot).abs() < 1e-12, "{a:?} vs {b:?}");
    }

    #[test]
    fn moving_across_layers_rebuilds_the_optics() {
        // Leaving the active layer is not a move, it is leaving the traced plane.
        let mut p = project();
        let id = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        assert_eq!(p.scene().elements.len(), 1);

        p.move_module(&id, [0, 0, 1]).unwrap();
        assert!(p.scene().elements.is_empty(), "layer 1 is not traced");
        assert!(p.instance(&id).is_none());
        assert!(matches!(
            p.warnings(),
            [ProjectWarning::InactiveLayer { layer: 1, active_layer: 0, .. }]
        ));

        p.move_module(&id, [0, 0, 0]).unwrap();
        assert_eq!(p.scene().elements.len(), 1, "and coming back restores it");
    }

    // --- Removal and the id remap (§14.4) ----------------------------------

    #[test]
    fn removing_a_module_removes_exactly_its_objects() {
        let mut p = project();
        p.place_module(&source(), [0, 0, 0], Quadrant::R0).unwrap();
        let lens_id = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        assert_eq!(p.scene().elements.len(), 1);
        assert_eq!(p.scene().sources.len(), 1);

        p.remove_module(&lens_id).unwrap();
        assert!(p.scene().elements.is_empty(), "the lens is gone");
        assert_eq!(p.scene().sources.len(), 1, "the source is not");
        assert!(p.instance(&lens_id).is_none());
        assert!(p.setup().instance(&lens_id).is_none());
    }

    #[test]
    fn deleting_an_early_instance_remaps_every_later_one() {
        // The regression this whole remap contract exists for. Three lenses; drop
        // the first. Scene ids 1 and 2 slide down to 0 and 1, and if the surviving
        // instances keep their old ids they now name each other's geometry — a
        // module that silently moves the wrong lens.
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let b = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        let c = p.place_module(&lens(), [2, 0, 0], Quadrant::R0).unwrap();
        assert_eq!(p.instance(&a).unwrap().inserted.elements[0], ElementId(0));
        assert_eq!(p.instance(&b).unwrap().inserted.elements[0], ElementId(1));
        assert_eq!(p.instance(&c).unwrap().inserted.elements[0], ElementId(2));

        p.remove_module(&a).unwrap();

        // Ids slid down...
        assert_eq!(p.instance(&b).unwrap().inserted.elements[0], ElementId(0));
        assert_eq!(p.instance(&c).unwrap().inserted.elements[0], ElementId(1));
        // ...and still name the right geometry: b was at x=50, c at x=100.
        let pos = |i: &str| p.scene().elements[p.instance(i).unwrap().inserted.elements[0].index()].transform.pos;
        assert_eq!(pos(&b), oc_core::Vec2::new(50.0, 0.0));
        assert_eq!(pos(&c), oc_core::Vec2::new(100.0, 0.0));
    }

    #[test]
    fn the_reverse_ownership_map_survives_a_deletion() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let b = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        assert_eq!(p.owner_of(SceneObjectId::Element(1)), Some(b.as_str()));

        p.remove_module(&a).unwrap();

        // Element 0 is now b's, and nothing still claims the vacated id.
        assert_eq!(p.owner_of(SceneObjectId::Element(0)), Some(b.as_str()));
        assert_eq!(p.owner_of(SceneObjectId::Element(1)), None);
    }

    #[test]
    fn moving_a_survivor_after_a_deletion_moves_the_right_module() {
        // The end-to-end shape of the bug: delete, then drag. Without the remap
        // this moves whatever now sits at the stale id.
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let b = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        let c = p.place_module(&lens(), [2, 0, 0], Quadrant::R0).unwrap();

        p.remove_module(&a).unwrap();
        p.move_module(&c, [5, 0, 0]).unwrap();

        let pos = |i: &str| p.scene().elements[p.instance(i).unwrap().inserted.elements[0].index()].transform.pos;
        assert_eq!(pos(&c), oc_core::Vec2::new(250.0, 0.0), "c moved");
        assert_eq!(pos(&b), oc_core::Vec2::new(50.0, 0.0), "b did not");
    }

    #[test]
    fn selection_is_cleared_when_its_module_is_deleted() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        assert_eq!(p.select_object(SceneObjectId::Element(0)), Some(a.as_str()));

        p.remove_module(&a).unwrap();
        assert_eq!(p.selected_object(), None, "the object no longer exists");
        assert_eq!(p.selected_instance(), None);
    }

    #[test]
    fn selection_is_remapped_when_another_module_is_deleted() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let b = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        p.select_object(SceneObjectId::Element(1));
        assert_eq!(p.selected_instance(), Some(b.as_str()));

        p.remove_module(&a).unwrap();

        // The selection followed its object down rather than pointing at whatever
        // took the old id.
        assert_eq!(p.selected_object(), Some(SceneObjectId::Element(0)));
        assert_eq!(p.selected_instance(), Some(b.as_str()));
    }

    #[test]
    fn removing_every_module_leaves_no_orphans() {
        let mut p = project();
        let ids: Vec<_> = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
            .iter()
            .map(|a| p.place_module(&lens(), *a, Quadrant::R0).unwrap())
            .collect();
        for id in &ids {
            p.remove_module(id).unwrap();
        }
        assert!(p.scene().elements.is_empty());
        assert!(p.setup().instances.is_empty());
        assert_eq!(p.materialized_ids().count(), 0);
        assert!(p.owner_of(SceneObjectId::Element(0)).is_none());
    }

    #[test]
    fn removing_an_unknown_instance_is_refused() {
        let mut p = project();
        assert_eq!(
            p.remove_module("m-9999"),
            Err(Error::UnknownInstance("m-9999".to_string()))
        );
    }

    // --- Layers ------------------------------------------------------------

    #[test]
    fn only_the_active_layer_is_materialized() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [0, 0, 1], Quadrant::R0).unwrap();
        assert_eq!(p.scene().elements.len(), 1, "layer 0 only");

        p.set_active_layer(1).unwrap();
        assert_eq!(p.scene().elements.len(), 1, "now layer 1 only");
        assert_eq!(p.active_layer(), 1);
        // The layer-0 module is now the one flagged as context.
        assert!(matches!(
            p.warnings(),
            [ProjectWarning::InactiveLayer { layer: 0, active_layer: 1, .. }]
        ));
    }

    #[test]
    fn switching_layers_clears_the_selected_object() {
        // Every id changes on a rebuild, so a retained *object* selection could
        // only be wrong. The module the user picked is still theirs, though.
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.select_object(SceneObjectId::Element(0));
        p.set_active_layer(1).unwrap();
        assert_eq!(p.selected_object(), None, "the lens has no objects up here");
        assert_eq!(p.selected_instance(), Some(a.as_str()));

        // ...and coming back re-points it at the rebuilt geometry.
        p.set_active_layer(0).unwrap();
        assert_eq!(p.selected_object(), Some(SceneObjectId::Element(0)));
    }

    // --- Picking (§14.7) ---------------------------------------------------

    #[test]
    fn a_cell_names_the_module_standing_on_it() {
        let mut p = project();
        let a = p.place_module(&lens(), [2, 1, 0], Quadrant::R0).unwrap();
        assert_eq!(p.instance_at_cell([2, 1, 0]), Some(a.as_str()));
        assert_eq!(p.instance_at_cell([2, 2, 0]), None, "the next cell over");
        assert_eq!(p.instance_at_cell([2, 1, 1]), None, "another layer");
    }

    #[test]
    fn every_cell_of_a_multi_cell_module_picks_it() {
        // A rail is one module, not two. Clicking its far end must not miss.
        let (mut p, rail) = rail_project();
        let id = p.place_module(&rail, [0, 0, 0], Quadrant::R90).unwrap();

        // Rotated 90°, the second cell went from +x to +y.
        assert_eq!(p.instance_at_cell([0, 0, 0]), Some(id.as_str()));
        assert_eq!(p.instance_at_cell([0, 1, 0]), Some(id.as_str()));
        assert_eq!(p.instance_at_cell([1, 0, 0]), None);
    }

    #[test]
    fn a_layout_only_module_is_selectable() {
        // An empty cube owns no scene object, so an object-addressed selection
        // could never name it. It still occupies a cell and must pick like
        // anything else.
        let mut p = project();
        let c = p.place_module(&cube(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(p.instance(&c).is_none(), "the cube really has no objects");

        assert_eq!(p.select_at_cell([0, 0, 0]), Some(c.as_str()));
        assert_eq!(p.selected_instance(), Some(c.as_str()));
        assert_eq!(p.selected_object(), None);
    }

    #[test]
    fn picking_an_optical_module_selects_it_by_one_of_its_objects() {
        let mut p = project();
        let a = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        assert_eq!(p.select_at_cell([1, 0, 0]), Some(a.as_str()));
        assert_eq!(p.selected_object(), Some(SceneObjectId::Element(0)));
    }

    #[test]
    fn picking_empty_space_clears_the_selection() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.select_at_cell([0, 0, 0]);
        assert!(p.selected_instance().is_some());

        assert_eq!(p.select_at_cell([9, 9, 0]), None);
        assert_eq!(p.selected_instance(), None);
        assert_eq!(p.selected_object(), None);
    }

    #[test]
    fn deleting_a_layout_only_module_clears_its_selection() {
        // It owns no object, so the removal remap cannot be what clears this.
        let mut p = project();
        let c = p.place_module(&cube(), [0, 0, 0], Quadrant::R0).unwrap();
        p.select_at_cell([0, 0, 0]);
        p.remove_module(&c).unwrap();
        assert_eq!(p.selected_instance(), None);
    }

    // --- Collision preview -------------------------------------------------

    #[test]
    fn a_module_whose_anchor_cell_is_not_its_own_is_still_selectable() {
        // `offsets()` subtracts the anchor, and nothing requires the anchor to be
        // among the cells. Addressing a module by its anchor cell would miss this
        // one entirely, which is why selection has an instance-named primitive.
        let mut doc: serde_json::Value = serde_json::from_str(&minimal_catalog_json()).unwrap();
        let mut m = two_cell_module_json("u-offset");
        // Occupies anchor+[1,0,0] only: the anchor cell itself stays empty.
        m["footprint"] = serde_json::json!({"anchor_cell": [0, 0, 0], "cells": [[1, 0, 0]]});
        doc["modules"].as_array_mut().unwrap().push(m);
        let cat = Catalog::from_json(&doc.to_string()).unwrap();
        let mut p = OpenUc2Project::new(cat, GridProfile::openuc2_core_50()).unwrap();

        let id = p
            .place_module(&ModuleRef::new("u-offset", "1.0.0"), [0, 0, 0], Quadrant::R0)
            .unwrap();
        assert_eq!(p.instance_at_cell([0, 0, 0]), None, "the anchor is empty");
        assert_eq!(p.instance_at_cell([1, 0, 0]), Some(id.as_str()));

        assert_eq!(p.select_instance(&id), Some(id.as_str()));
        assert_eq!(p.selected_instance(), Some(id.as_str()));
    }

    #[test]
    fn selecting_a_module_that_is_gone_clears_rather_than_holds_it() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.select_instance(&a);
        assert_eq!(p.select_instance("m-9999"), None);
        assert_eq!(p.selected_instance(), None, "not left on the old module");
    }

    #[test]
    fn a_copy_is_blocked_by_the_original_but_a_move_is_not() {
        // The distinction the two queries exist for. A duplicate is a new module:
        // it inherits none of the original's right to the cells it stands on, so
        // offering it one would be offering a cell `duplicate_module` refuses.
        let (mut p, rail) = rail_project();
        let id = p.place_module(&rail, [0, 0, 0], Quadrant::R0).unwrap();
        // The rail spans [0,0,0] and [1,0,0].
        assert_eq!(p.blocker_for(&id, [1, 0, 0]), None, "moving onto itself is fine");
        assert_eq!(
            p.blocker_for_copy(&id, [1, 0, 0]),
            Some(id.clone()),
            "a copy there would overlap the original"
        );
        // And the query agrees with what duplicating actually does.
        assert!(p.duplicate_module(&id, [1, 0, 0]).is_err());
        assert_eq!(p.blocker_for_copy(&id, [2, 0, 0]), None);
        assert!(p.duplicate_module(&id, [2, 0, 0]).is_ok());
    }

    #[test]
    fn a_blocked_cell_names_its_occupant_without_touching_anything() {
        let mut p = project();
        let a = p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        let b = p.place_module(&lens(), [3, 0, 0], Quadrant::R0).unwrap();
        let scene_before = p.scene().to_json();

        assert_eq!(p.blocker_for(&b, [1, 0, 0]), Some(a.clone()));
        assert_eq!(p.blocker_for(&b, [2, 0, 0]), None);
        // A module never blocks itself: a drag that ends where it started is fine.
        assert_eq!(p.blocker_for(&b, [3, 0, 0]), None);

        assert_eq!(p.scene().to_json(), scene_before, "a query mutates nothing");
        assert_eq!(p.setup().instances.len(), 2);
    }

    // --- History (§14.8) ---------------------------------------------------

    #[test]
    fn undo_takes_back_a_placement() {
        let mut p = project();
        assert!(!p.can_undo(), "a new project has nothing to undo");

        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(p.can_undo());
        assert_eq!(p.scene().elements.len(), 1);

        assert!(p.undo().unwrap());
        assert_eq!(p.setup().instances.len(), 0);
        assert_eq!(p.scene().elements.len(), 0, "the geometry went with it");
        assert!(p.can_redo());

        assert!(p.redo().unwrap());
        assert_eq!(p.setup().instances.len(), 1);
        assert_eq!(p.scene().elements.len(), 1);
    }

    #[test]
    fn undoing_nothing_is_not_an_error() {
        let mut p = project();
        assert!(!p.undo().unwrap(), "nothing to undo");
        assert!(!p.redo().unwrap(), "nothing to redo");
    }

    #[test]
    fn undo_restores_the_scene_exactly() {
        // The scene is derived, so undo regenerates rather than replays. The proof
        // is that the regenerated scene is the same bytes, not merely similar.
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let before = p.scene().to_json();

        let b = p.place_module(&mirror(), [1, 0, 0], Quadrant::R0).unwrap();
        p.rotate_module(&b, Quadrant::R90).unwrap();
        p.undo().unwrap();
        p.undo().unwrap();

        assert_eq!(p.scene().to_json(), before);
    }

    #[test]
    fn a_refused_placement_is_not_an_undo_step() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.undo().unwrap();
        assert!(p.can_redo());

        // A collision changes nothing, so it must not consume the redo stack or
        // add a step that undoes to the state it is already in.
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(p.place_module(&cube(), [0, 0, 0], Quadrant::R0).is_err());
        assert!(p.undo().unwrap());
        assert_eq!(p.setup().instances.len(), 0, "one undo cleared the lens");
    }

    #[test]
    fn a_drag_is_one_undo_step() {
        // §15.6: one settled operation, one undo event. A drag is a move per
        // pointer event, and without the gesture each would be its own step.
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();

        p.begin_gesture();
        for x in 1..=6 {
            p.move_module(&a, [x, 0, 0]).unwrap();
        }
        p.end_gesture();

        assert_eq!(p.setup().instance(&a).unwrap().placement.anchor_grid, [6, 0, 0]);
        p.undo().unwrap();
        assert_eq!(
            p.setup().instance(&a).unwrap().placement.anchor_grid,
            [0, 0, 0],
            "one undo returns to where the drag started, not to cell 5"
        );

        // And the step underneath the drag is the placement itself: the six moves
        // left nothing else on the stack.
        p.undo().unwrap();
        assert!(p.setup().instances.is_empty());
        assert!(!p.can_undo());
    }

    #[test]
    fn a_drag_that_ends_where_it_started_is_not_an_edit() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.undo().unwrap();
        p.redo().unwrap();
        assert!(!p.can_redo());

        p.begin_gesture();
        p.move_module(&a, [3, 0, 0]).unwrap();
        p.move_module(&a, [0, 0, 0]).unwrap();
        p.end_gesture();

        // Picking a module up and putting it back must not cost an undo step.
        p.undo().unwrap();
        assert_eq!(p.setup().instances.len(), 0, "the undo reached the placement");
    }

    #[test]
    fn edits_outside_a_gesture_are_one_step_each() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.move_module(&a, [1, 0, 0]).unwrap();
        p.rotate_module(&a, Quadrant::R90).unwrap();

        p.undo().unwrap();
        assert_eq!(p.setup().instance(&a).unwrap().placement.rotation_quadrants, 0);
        p.undo().unwrap();
        assert_eq!(p.setup().instance(&a).unwrap().placement.anchor_grid, [0, 0, 0]);
        p.undo().unwrap();
        assert!(p.setup().instances.is_empty());
    }

    #[test]
    fn a_new_edit_after_an_undo_drops_the_redo() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        p.undo().unwrap();
        assert!(p.can_redo());

        p.place_module(&cube(), [5, 0, 0], Quadrant::R0).unwrap();
        assert!(!p.can_redo(), "that future is unreachable now");
    }

    #[test]
    fn undo_covers_removal() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.remove_module(&a).unwrap();
        assert_eq!(p.scene().elements.len(), 0);

        p.undo().unwrap();
        assert_eq!(p.setup().instances.len(), 1, "the module came back");
        assert_eq!(p.scene().elements.len(), 1, "and so did its optics");
    }

    #[test]
    fn undo_covers_the_active_layer() {
        // The active layer decides what materializes, so it is part of the state a
        // snapshot has to restore.
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.set_active_layer(1).unwrap();
        assert_eq!(p.scene().elements.len(), 0, "the lens is on layer 0");

        p.undo().unwrap();
        assert_eq!(p.active_layer(), 0);
        assert_eq!(p.scene().elements.len(), 1);
    }

    #[test]
    fn a_duplicate_is_one_undo_step() {
        // `duplicate_module` places through `place_module`, which records for it.
        // Two steps here would mean the user undoing a copy twice.
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.duplicate_module(&a, [1, 0, 0]).unwrap();
        assert_eq!(p.setup().instances.len(), 2);

        p.undo().unwrap();
        assert_eq!(p.setup().instances.len(), 1, "one undo, one copy gone");
    }

    // --- Ports (§14.6) -----------------------------------------------------

    #[test]
    fn a_placed_modules_ports_land_in_the_scene_frame() {
        let mut p = project();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        let ports = p.ports();
        assert_eq!(ports.len(), 2, "the lens fixture declares in and out");

        let out = ports.iter().find(|q| q.port_id == "optical-out").unwrap();
        // Cell 1 spans x = 25..75, so the +X face is at 75.
        assert!((out.position_mm[0] - 75.0).abs() < 1e-12, "{out:?}");
    }

    #[test]
    fn a_module_with_no_ports_contributes_none() {
        let mut p = project();
        p.place_module(&cube(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(p.ports().is_empty());
    }

    #[test]
    fn ports_follow_a_module_when_it_moves() {
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let before = p.ports()[0].position_mm;
        p.move_module(&a, [1, 0, 0]).unwrap();
        let after = p.ports()[0].position_mm;
        assert!((after[0] - before[0] - 50.0).abs() < 1e-12, "one cell along x");
    }

    // --- Unresolved modules ------------------------------------------------

    /// A setup pinned to a module the catalogue does not have.
    fn setup_with_missing_module() -> String {
        let mut p = project();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        let mut setup: serde_json::Value = serde_json::from_str(&p.save()).unwrap();
        setup["instances"][0]["module_version"] = serde_json::json!("9.9.9");
        setup.to_string()
    }

    #[test]
    fn a_missing_module_becomes_a_placeholder() {
        let p = OpenUc2Project::load(pilot_catalog(), &setup_with_missing_module()).unwrap();
        // The instance survives, carrying what the setup remembered...
        assert_eq!(p.setup().instances.len(), 1);
        assert_eq!(p.setup().instances[0].fallback.module_id, "lens-pcx-nbk7-1x1");
        // ...but contributes no optics, and says why.
        assert!(p.scene().elements.is_empty());
        assert!(matches!(
            p.warnings(),
            [ProjectWarning::UnresolvedModule { version, .. }] if version == "9.9.9"
        ));
    }

    #[test]
    fn a_missing_version_is_never_swapped_for_a_present_one() {
        // v1.0.0 is right there in the catalogue. Loading a setup pinned to 9.9.9
        // must not quietly use it: that would change the design's optics without
        // telling anyone.
        let p = OpenUc2Project::load(pilot_catalog(), &setup_with_missing_module()).unwrap();
        assert!(p.scene().elements.is_empty());
        assert_eq!(p.setup().instances[0].module_version, "9.9.9");
    }

    #[test]
    fn an_unresolved_placeholder_still_occupies_its_cells() {
        // Otherwise another module could be placed on top of it, and the collision
        // would reappear the moment the catalogue came back.
        let mut p = OpenUc2Project::load(pilot_catalog(), &setup_with_missing_module()).unwrap();
        assert!(matches!(
            p.place_module(&cube(), [1, 0, 0], Quadrant::R0),
            Err(PlacementError::Collision { .. })
        ));
    }

    #[test]
    fn a_declared_fragment_whose_asset_is_missing_is_reported() {
        // The catalogue says the lens has optics, but the asset was never
        // attached. Silence here would look exactly like a layout-only module.
        let cat = Catalog::from_json(&minimal_catalog_json()).unwrap();
        let mut p = OpenUc2Project::new(cat, GridProfile::openuc2_core_50()).unwrap();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(p.scene().elements.is_empty());
        assert!(matches!(
            p.warnings(),
            [ProjectWarning::MissingFragmentAsset { asset, .. }] if asset == "assets/lens.ocanvas"
        ));
    }

    #[test]
    fn an_idealized_module_says_so() {
        let mut p = project();
        p.place_module(&source(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(matches!(
            p.warnings(),
            [ProjectWarning::IdealizedOptics { fidelity, .. }] if fidelity == "idealized"
        ));
    }

    #[test]
    fn a_validated_module_is_quiet() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        assert!(p.warnings().is_empty(), "{:?}", p.warnings());
    }

    // --- Save and load -----------------------------------------------------

    #[test]
    fn a_setup_round_trips_through_the_project() {
        let mut p = project();
        p.place_module(&source(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&mirror(), [2, 0, 0], Quadrant::R0).unwrap();
        let saved = p.save();

        let reopened = OpenUc2Project::load(pilot_catalog(), &saved).unwrap();
        assert_eq!(reopened.save(), saved, "reopening must not perturb the file");
        assert_eq!(reopened.setup().instances.len(), 3);
        // And the scene comes back the same, because it is derived from the file.
        assert_eq!(reopened.scene().elements.len(), p.scene().elements.len());
        assert_eq!(reopened.scene().sources.len(), p.scene().sources.len());
    }

    #[test]
    fn the_saved_setup_pins_the_catalogue() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let setup = OpenUc2Setup::from_json(&p.save()).unwrap();
        assert_eq!(setup.catalog.repository, "example/optiland-canvas-dev-fixture");
        assert_eq!(setup.catalog.commit, "0000000000000000000000000000000000000000");
        assert_eq!(setup.catalog.grid_profile, "openuc2-core-50");
    }

    #[test]
    fn loading_against_an_unknown_grid_profile_is_refused() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        let mut setup: serde_json::Value = serde_json::from_str(&p.save()).unwrap();
        setup["catalog"]["grid_profile"] = serde_json::json!("openuc2-imaginary-70");
        let err = OpenUc2Project::load(pilot_catalog(), &setup.to_string()).unwrap_err();
        assert_eq!(
            err,
            Error::UnknownGridProfile("openuc2-imaginary-70".to_string())
        );
    }

    #[test]
    fn placement_is_deterministic_across_a_reload() {
        let mut p = project();
        p.place_module(&lens(), [2, 3, 0], Quadrant::R90).unwrap();
        let before = p.scene().to_json();
        let reopened = OpenUc2Project::load(pilot_catalog(), &p.save()).unwrap();
        assert_eq!(reopened.scene().to_json(), before);
    }

    // --- BOM ---------------------------------------------------------------

    #[test]
    fn the_bom_aggregates_by_product() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&cube(), [2, 0, 0], Quadrant::R0).unwrap();

        let bom = p.bom();
        assert_eq!(bom.lines.len(), 2);
        let lens_line = bom.lines.iter().find(|l| l.module_id == "lens-pcx-nbk7-1x1").unwrap();
        assert_eq!(lens_line.quantity, 2);
        assert_eq!(lens_line.unit_price_eur, 24.0);
        assert_eq!(lens_line.subtotal_eur, 48.0);
        assert_eq!(lens_line.sku, "EXAMPLE-LENS-01");
        assert_eq!(bom.total_eur, 48.0 + 6.0);
    }

    #[test]
    fn the_bom_splits_on_parameter_overrides() {
        // Two of the same product configured differently are two line items: they
        // are not interchangeable when someone goes to build it.
        let mut p = project();
        let a = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        p.set_parameter_override(&a, "coating", serde_json::json!("AR-VIS"))
            .unwrap();

        let bom = p.bom();
        assert_eq!(bom.lines.len(), 2);
        assert!(bom.lines.iter().all(|l| l.quantity == 1));
    }

    #[test]
    fn the_bom_counts_instances_on_every_layer() {
        // The BOM is what you order, not what you trace. A module on an inactive
        // layer is still a module you have to buy.
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [0, 0, 1], Quadrant::R0).unwrap();
        assert_eq!(p.bom().lines[0].quantity, 2);
        assert_eq!(p.scene().elements.len(), 1, "but only one is traced");
    }

    #[test]
    fn the_bom_excludes_disabled_instances() {
        let mut p = project();
        let id = p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.setup.instance_mut(&id).unwrap().enabled = false;
        assert!(p.bom().lines.is_empty());
    }

    #[test]
    fn an_unresolved_module_is_in_the_bom_but_not_priced() {
        // It is still something the design needs; it just cannot be costed.
        let p = OpenUc2Project::load(pilot_catalog(), &setup_with_missing_module()).unwrap();
        let bom = p.bom();
        assert_eq!(bom.lines.len(), 1);
        assert!(!bom.lines[0].resolved);
        assert_eq!(bom.lines[0].module_id, "lens-pcx-nbk7-1x1");
        assert_eq!(bom.lines[0].subtotal_eur, 0.0);
        assert_eq!(bom.total_eur, 0.0);
    }

    #[test]
    fn the_bom_is_deterministic() {
        let mut p = project();
        p.place_module(&cube(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();
        let a = p.bom();
        assert_eq!(a, p.bom());
        // Sorted by module id, not by placement order.
        let ids: Vec<_> = a.lines.iter().map(|l| l.module_id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }

    // --- Export ------------------------------------------------------------

    #[test]
    fn the_flat_export_is_a_loadable_scene() {
        let mut p = project();
        p.place_module(&source(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [1, 0, 0], Quadrant::R0).unwrap();

        let export = p.export_flat_ocanvas().unwrap();
        let scene = Scene::from_json(&export.ocanvas).expect("exports a valid .ocanvas");
        assert_eq!(scene.elements.len(), 1);
        assert_eq!(scene.sources.len(), 1);
        // The glass came through as a glass, not as an index into someone else's
        // catalogue.
        let id = match scene.elements[0].material {
            oc_core::Material::Catalog(g) => g,
            ref m => panic!("expected a catalog glass, got {m:?}"),
        };
        assert_eq!(scene.catalog.get(id).unwrap().name, "N-BK7");
    }

    #[test]
    fn the_flat_export_reports_fidelity_beside_the_file() {
        // `.ocanvas` v1 has nowhere to put a warning, and adding one would couple
        // the generic format to openUC2. So they travel next to it.
        let mut p = project();
        p.place_module(&source(), [0, 0, 0], Quadrant::R0).unwrap();
        let export = p.export_flat_ocanvas().unwrap();
        assert!(matches!(
            export.warnings.as_slice(),
            [ProjectWarning::IdealizedOptics { .. }]
        ));
        assert!(!export.ocanvas.contains("idealized"), "not inside the file");
    }

    #[test]
    fn the_flat_export_carries_only_the_active_layer() {
        let mut p = project();
        p.place_module(&lens(), [0, 0, 0], Quadrant::R0).unwrap();
        p.place_module(&lens(), [0, 0, 1], Quadrant::R0).unwrap();
        let scene = Scene::from_json(&p.export_flat_ocanvas().unwrap().ocanvas).unwrap();
        assert_eq!(scene.elements.len(), 1, "Canvas traces one layer at a time");
    }
}
