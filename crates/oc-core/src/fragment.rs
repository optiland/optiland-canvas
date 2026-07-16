//! Generic scene fragments and rigid batch transforms (§13).
//!
//! A [`SceneFragment`] is a self-contained piece of a scene — elements, sources,
//! detectors, and the glasses and layers they reference — expressed in its own
//! local frame and placeable into a [`Scene`] under a rigid [`Transform2`].
//!
//! Nothing here is domain-specific: a fragment is just "some scene content plus
//! where to put it". It is also purely an API abstraction, so the frozen
//! `.ocanvas` v1 format is untouched — a full document can be *read* as a
//! fragment by discarding its document-only state (see
//! [`SceneFragment::from_scene`]).
//!
//! # Placement is exact, and free
//!
//! [`Element`] and [`Source`] each already carry an [`Isometry2`], so placing
//! one folds the outer transform into that field with [`Isometry2::compose`].
//! Boundary [`Edge`]s are never rewritten. Lines, arcs, conics, aspheres, and
//! splines therefore all place through one code path, with no per-geometry
//! rewriting and no precision loss — an arc's sweep angles and a conic's axis
//! are carried by the element transform rather than re-derived. [`Detector`] is
//! the only type that stores bare points with no transform of its own, so it is
//! the only geometry mapped directly.
//!
//! # Ids are positions, so merging must remap them
//!
//! Ids in `oc-core` are indices into [`Scene`]'s vectors, and two fields store
//! such indices *into scene-level tables*: [`Material::Catalog`] holds a
//! [`GlassId`] into [`Scene::catalog`], and [`Element::layer`] holds a
//! [`LayerId`] into [`Scene::layers`]. Combining content authored against
//! different tables must therefore remap both, or `GlassId(2)` — N-SF11 in the
//! builtin catalog — silently becomes an unrelated glass in the destination.
//! Glasses merge by name via [`GlassCatalog::intern`] and report conflicts;
//! layers merge by name and, being presentation-only, do not.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;

use crate::geom::Edge;
use crate::material::{Glass, GlassId, Material};
use crate::scene::{Detector, DetectorId, Element, ElementId, Layer, LayerId, Scene, Source, SourceId};
use crate::vec::Isometry2;

/// A rigid placement: rotation about the origin, then translation.
///
/// This is the spec's `Transform2`. It is an alias rather than a new type
/// because [`Isometry2`] already *is* exactly this transform and is the type
/// [`Element`] and [`Source`] store. A separate struct would add a conversion at
/// every boundary and let the two definitions drift apart.
///
/// Scale and shear are unrepresentable by construction, which is the intended
/// constraint for module placement.
pub type Transform2 = Isometry2;

/// Why a fragment or batch operation could not be carried out.
///
/// Every variant means the operation was **refused with the scene unchanged**;
/// nothing here is a partial-success or best-effort report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentError {
    /// Geometry, a transform, or a detector endpoint was NaN or infinite.
    NonFinite {
        /// Which part of the fragment: `"element boundary"`, `"transform"`, …
        what: &'static str,
        /// Index of the offending object within its own list.
        index: usize,
    },
    /// An element's [`Material::Catalog`] id is outside the fragment's glasses.
    UnknownGlass { element: usize, id: GlassId },
    /// An element's [`Element::layer`] is outside the fragment's layers.
    UnknownLayer { element: usize, id: LayerId },
    /// Two glasses share a name but disagree on dispersion, so neither the
    /// fragment's nor the destination's optics can be preserved by merging.
    GlassConflict { name: String },
    /// A material that cannot be carried through a rigid transform.
    UnsupportedMaterial { element: usize, what: &'static str },
    /// A batch operation named an object the scene does not contain.
    UnknownObject { what: &'static str, id: u32 },
}

impl fmt::Display for FragmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FragmentError::NonFinite { what, index } => {
                write!(f, "{what} of object {index} is not finite")
            }
            FragmentError::UnknownGlass { element, id } => write!(
                f,
                "element {element} references glass {}, which the fragment does not carry",
                id.0
            ),
            FragmentError::UnknownLayer { element, id } => write!(
                f,
                "element {element} references layer {}, which the fragment does not carry",
                id.0
            ),
            FragmentError::GlassConflict { name } => write!(
                f,
                "glass {name:?} exists in both the fragment and the scene with different \
                 dispersion data; rename one, as merging would change the optics of the other"
            ),
            FragmentError::UnsupportedMaterial { element, what } => write!(
                f,
                "element {element} has material {what}, which cannot be rigidly transformed"
            ),
            FragmentError::UnknownObject { what, id } => {
                write!(f, "no such {what} in this scene: {id}")
            }
        }
    }
}

impl std::error::Error for FragmentError {}

/// Is every field of this edge finite?
fn edge_is_finite(edge: &Edge) -> bool {
    match edge {
        Edge::Line { a, b } => a.is_finite() && b.is_finite(),
        Edge::Arc { center, r, a0, a1 } => {
            center.is_finite() && r.is_finite() && a0.is_finite() && a1.is_finite()
        }
        Edge::Conic(cp) => {
            cp.vertex.is_finite()
                && cp.axis.is_finite()
                && cp.c.is_finite()
                && cp.k.is_finite()
                && cp.half_ap.is_finite()
        }
        Edge::Asphere { base, coeffs } => {
            base.vertex.is_finite()
                && base.axis.is_finite()
                && base.c.is_finite()
                && base.k.is_finite()
                && base.half_ap.is_finite()
                && coeffs.iter().all(|c| c.is_finite())
        }
        Edge::Spline { ctrl, .. } => ctrl.iter().all(|p| p.is_finite()),
    }
}

/// A set of scene objects, addressed by id.
///
/// Ordered and deduplicating, so a batch built from the same objects always
/// applies in the same order regardless of how it was assembled.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectIdSet {
    pub elements: BTreeSet<u32>,
    pub sources: BTreeSet<u32>,
    pub detectors: BTreeSet<u32>,
}

impl ObjectIdSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_element(mut self, id: ElementId) -> Self {
        self.elements.insert(id.0);
        self
    }

    pub fn with_source(mut self, id: SourceId) -> Self {
        self.sources.insert(id.0);
        self
    }

    pub fn with_detector(mut self, id: DetectorId) -> Self {
        self.detectors.insert(id.0);
        self
    }

    /// Total number of objects named.
    pub fn len(&self) -> usize {
        self.elements.len() + self.sources.len() + self.detectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What [`Scene::insert_fragment`] added.
///
/// Each vector is the fragment's **local-to-world id map**: the fragment's
/// `elements[i]` became `inserted.elements[i]`, and likewise for sources and
/// detectors. Insertion preserves fragment order, so the local index *is* the
/// lookup key and no side table is needed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InsertedFragment {
    pub elements: Vec<ElementId>,
    pub sources: Vec<SourceId>,
    pub detectors: Vec<DetectorId>,
}

impl InsertedFragment {
    /// Every object the insertion created, as a batch for
    /// [`Scene::transform_objects`] or [`Scene::remove_objects`].
    pub fn ids(&self) -> ObjectIdSet {
        ObjectIdSet {
            elements: self.elements.iter().map(|i| i.0).collect(),
            sources: self.sources.iter().map(|i| i.0).collect(),
            detectors: self.detectors.iter().map(|i| i.0).collect(),
        }
    }

    /// Total number of objects inserted.
    pub fn len(&self) -> usize {
        self.elements.len() + self.sources.len() + self.detectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// How ids moved during [`Scene::remove_objects`].
///
/// Removing from the middle of a vector shifts everything after it, so ids that
/// a caller is holding go stale. Each lookup maps a **pre-removal** id to its
/// post-removal id, or `None` if that object was the one removed. A caller
/// tracking object ownership must push its stored ids through this.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdRemap {
    elements: Vec<Option<ElementId>>,
    sources: Vec<Option<SourceId>>,
    detectors: Vec<Option<DetectorId>>,
}

impl IdRemap {
    /// Where `old` ended up, or `None` if it was removed.
    pub fn element(&self, old: ElementId) -> Option<ElementId> {
        self.elements.get(old.index()).copied().flatten()
    }

    pub fn source(&self, old: SourceId) -> Option<SourceId> {
        self.sources.get(old.index()).copied().flatten()
    }

    pub fn detector(&self, old: DetectorId) -> Option<DetectorId> {
        self.detectors.get(old.index()).copied().flatten()
    }

    /// Push a whole batch through the remap, dropping removed objects.
    pub fn remap_set(&self, ids: &ObjectIdSet) -> ObjectIdSet {
        ObjectIdSet {
            elements: ids
                .elements
                .iter()
                .filter_map(|&i| self.element(ElementId(i)))
                .map(|i| i.0)
                .collect(),
            sources: ids
                .sources
                .iter()
                .filter_map(|&i| self.source(SourceId(i)))
                .map(|i| i.0)
                .collect(),
            detectors: ids
                .detectors
                .iter()
                .filter_map(|&i| self.detector(DetectorId(i)))
                .map(|i| i.0)
                .collect(),
        }
    }
}

/// Scene content in a local frame, ready to be placed.
///
/// `glasses` and `layers` are the fragment's own tables; ids inside `elements`
/// index into them, not into any destination scene. [`Scene::insert_fragment`]
/// remaps both on the way in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneFragment {
    pub elements: Vec<Element>,
    pub sources: Vec<Source>,
    pub detectors: Vec<Detector>,
    /// Glasses referenced by `elements`, indexed by the fragment's own
    /// [`GlassId`]s.
    pub glasses: Vec<Glass>,
    /// Layers referenced by `elements`, indexed by the fragment's own
    /// [`LayerId`]s.
    pub layers: Vec<Layer>,
}

impl SceneFragment {
    /// Read a scene as a fragment.
    ///
    /// Document-only state is dropped, which is what makes any `.ocanvas` file
    /// usable as a fragment: the format header, and [`Scene::env`] — the
    /// destination scene's ambient medium is authoritative, so a fragment never
    /// carries its own.
    pub fn from_scene(scene: &Scene) -> Result<Self, FragmentError> {
        let fragment = Self {
            elements: scene.elements.clone(),
            sources: scene.sources.clone(),
            detectors: scene.detectors.clone(),
            glasses: scene.catalog.iter().map(|(_, g)| g.clone()).collect(),
            layers: scene.layers.clone(),
        };
        fragment.validate()?;
        Ok(fragment)
    }

    /// Reject anything that cannot be placed, before any of it is placed.
    fn validate(&self) -> Result<(), FragmentError> {
        for (i, e) in self.elements.iter().enumerate() {
            if !e.transform.is_finite() {
                return Err(FragmentError::NonFinite {
                    what: "element transform",
                    index: i,
                });
            }
            if !e.boundary.iter().all(edge_is_finite) {
                return Err(FragmentError::NonFinite {
                    what: "element boundary",
                    index: i,
                });
            }
            match &e.material {
                Material::Catalog(id) if id.0 as usize >= self.glasses.len() => {
                    return Err(FragmentError::UnknownGlass {
                        element: i,
                        id: *id,
                    })
                }
                // A `GrinField` stores an axis-aligned box in *world* space, which
                // a rotation cannot carry — the rotated field is not an AABB. Since
                // GRIN propagation is unimplemented anyway (the field only supplies
                // a constant fallback index), refuse it rather than place an
                // element whose material region silently stays behind. §13.3
                // requires a typed error over leaving geometry untransformed.
                Material::Grin(_) => {
                    return Err(FragmentError::UnsupportedMaterial {
                        element: i,
                        what: "Grin",
                    })
                }
                _ => {}
            }
            if e.layer.index() >= self.layers.len() {
                return Err(FragmentError::UnknownLayer {
                    element: i,
                    id: e.layer,
                });
            }
        }
        for (i, s) in self.sources.iter().enumerate() {
            if !s.transform.is_finite() {
                return Err(FragmentError::NonFinite {
                    what: "source transform",
                    index: i,
                });
            }
        }
        for (i, d) in self.detectors.iter().enumerate() {
            if !d.a.is_finite() || !d.b.is_finite() {
                return Err(FragmentError::NonFinite {
                    what: "detector endpoint",
                    index: i,
                });
            }
        }
        Ok(())
    }

    /// This fragment rigidly moved by `transform`.
    ///
    /// Elements and sources compose the transform into the [`Isometry2`] they
    /// already carry; detector endpoints are mapped directly. Nothing is
    /// resampled or re-fitted, so the result is exact.
    pub fn transformed(&self, transform: Transform2) -> Result<Self, FragmentError> {
        if !transform.is_finite() {
            return Err(FragmentError::NonFinite {
                what: "transform",
                index: 0,
            });
        }
        self.validate()?;

        let mut out = self.clone();
        for e in &mut out.elements {
            e.transform = transform.compose(e.transform);
        }
        for s in &mut out.sources {
            s.transform = transform.compose(s.transform);
        }
        for d in &mut out.detectors {
            d.a = transform.transform_point(d.a);
            d.b = transform.transform_point(d.b);
        }
        Ok(out)
    }

    /// Number of objects in the fragment.
    pub fn len(&self) -> usize {
        self.elements.len() + self.sources.len() + self.detectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Scene {
    /// Place `fragment` into this scene under `transform`.
    ///
    /// The fragment's glasses and layers are merged into the scene's tables and
    /// every id inside the inserted content is remapped to match. Only glasses
    /// and layers actually *referenced* by the fragment's elements are merged,
    /// so carrying a full catalog around costs nothing and cannot raise a
    /// spurious conflict.
    ///
    /// Atomic: every fallible step runs against staged copies, so on `Err` the
    /// scene is untouched.
    pub fn insert_fragment(
        &mut self,
        fragment: &SceneFragment,
        transform: Transform2,
    ) -> Result<InsertedFragment, FragmentError> {
        // Validates the fragment as a side effect, so everything below may index
        // `glasses`/`layers` freely.
        let placed = fragment.transformed(transform)?;

        // Stage the merged tables; commit only once nothing can fail.
        let mut catalog = self.catalog.clone();
        let mut layers = self.layers.clone();
        let mut glass_map: BTreeMap<u32, GlassId> = BTreeMap::new();
        let mut layer_map: BTreeMap<u32, LayerId> = BTreeMap::new();

        for e in &placed.elements {
            if let Material::Catalog(id) = &e.material {
                if let Entry::Vacant(slot) = glass_map.entry(id.0) {
                    let glass = &placed.glasses[id.0 as usize];
                    let merged = catalog.intern(glass).ok_or_else(|| {
                        FragmentError::GlassConflict {
                            name: glass.name.clone(),
                        }
                    })?;
                    slot.insert(merged);
                }
            }
            if let Entry::Vacant(slot) = layer_map.entry(e.layer.0) {
                let layer = &placed.layers[e.layer.index()];
                // Layers are presentation, not physics: a name match reuses the
                // destination's visibility/opacity rather than conflicting.
                let merged = match layers.iter().position(|l| l.name == layer.name) {
                    Some(i) => LayerId(i as u32),
                    None => {
                        layers.push(layer.clone());
                        LayerId((layers.len() - 1) as u32)
                    }
                };
                slot.insert(merged);
            }
        }

        // Commit. Nothing below can fail.
        self.catalog = catalog;
        self.layers = layers;

        let mut inserted = InsertedFragment::default();
        for mut e in placed.elements {
            if let Material::Catalog(id) = &mut e.material {
                *id = glass_map[&id.0];
            }
            e.layer = layer_map[&e.layer.0];
            // `add_element` assigns the id from the vector position.
            inserted.elements.push(self.add_element(e));
        }
        for s in placed.sources {
            inserted.sources.push(SourceId(self.sources.len() as u32));
            self.sources.push(s);
        }
        for d in placed.detectors {
            inserted.detectors.push(DetectorId(self.detectors.len() as u32));
            self.detectors.push(d);
        }
        Ok(inserted)
    }

    /// Remove every object in `ids`.
    ///
    /// Returns an [`IdRemap`], because ids are vector positions: removing one
    /// object renumbers those after it. Callers holding ids — an ownership map,
    /// a selection — must push them through the remap.
    ///
    /// Atomic: ids are all checked before anything is removed.
    pub fn remove_objects(&mut self, ids: &ObjectIdSet) -> Result<IdRemap, FragmentError> {
        for &i in &ids.elements {
            if i as usize >= self.elements.len() {
                return Err(FragmentError::UnknownObject {
                    what: "element",
                    id: i,
                });
            }
        }
        for &i in &ids.sources {
            if i as usize >= self.sources.len() {
                return Err(FragmentError::UnknownObject {
                    what: "source",
                    id: i,
                });
            }
        }
        for &i in &ids.detectors {
            if i as usize >= self.detectors.len() {
                return Err(FragmentError::UnknownObject {
                    what: "detector",
                    id: i,
                });
            }
        }

        let mut remap = IdRemap::default();

        let mut kept = Vec::with_capacity(self.elements.len() - ids.elements.len());
        for (i, mut e) in self.elements.drain(..).enumerate() {
            if ids.elements.contains(&(i as u32)) {
                remap.elements.push(None);
            } else {
                let id = ElementId(kept.len() as u32);
                // Ids are positional; renumber so `Element::id` keeps matching.
                e.id = id;
                kept.push(e);
                remap.elements.push(Some(id));
            }
        }
        self.elements = kept;

        let mut kept = Vec::with_capacity(self.sources.len() - ids.sources.len());
        for (i, s) in self.sources.drain(..).enumerate() {
            if ids.sources.contains(&(i as u32)) {
                remap.sources.push(None);
            } else {
                remap.sources.push(Some(SourceId(kept.len() as u32)));
                kept.push(s);
            }
        }
        self.sources = kept;

        let mut kept = Vec::with_capacity(self.detectors.len() - ids.detectors.len());
        for (i, d) in self.detectors.drain(..).enumerate() {
            if ids.detectors.contains(&(i as u32)) {
                remap.detectors.push(None);
            } else {
                remap.detectors.push(Some(DetectorId(kept.len() as u32)));
                kept.push(d);
            }
        }
        self.detectors = kept;

        Ok(remap)
    }

    /// Rigidly move every object in `ids` by `delta`.
    ///
    /// This is one batch: a module drag transforms all of its objects together
    /// and preserves their relative geometry exactly, because each object simply
    /// composes `delta` onto the transform it already carries.
    ///
    /// Atomic: the whole batch is checked before any of it moves.
    pub fn transform_objects(
        &mut self,
        ids: &ObjectIdSet,
        delta: Transform2,
    ) -> Result<(), FragmentError> {
        if !delta.is_finite() {
            return Err(FragmentError::NonFinite {
                what: "transform",
                index: 0,
            });
        }
        for &i in &ids.elements {
            let e = self
                .elements
                .get(i as usize)
                .ok_or(FragmentError::UnknownObject {
                    what: "element",
                    id: i,
                })?;
            if matches!(e.material, Material::Grin(_)) {
                return Err(FragmentError::UnsupportedMaterial {
                    element: i as usize,
                    what: "Grin",
                });
            }
        }
        for &i in &ids.sources {
            if i as usize >= self.sources.len() {
                return Err(FragmentError::UnknownObject {
                    what: "source",
                    id: i,
                });
            }
        }
        for &i in &ids.detectors {
            if i as usize >= self.detectors.len() {
                return Err(FragmentError::UnknownObject {
                    what: "detector",
                    id: i,
                });
            }
        }

        for &i in &ids.elements {
            let e = &mut self.elements[i as usize];
            e.transform = delta.compose(e.transform);
        }
        for &i in &ids.sources {
            let s = &mut self.sources[i as usize];
            s.transform = delta.compose(s.transform);
        }
        for &i in &ids.detectors {
            let d = &mut self.detectors[i as usize];
            d.a = delta.transform_point(d.a);
            d.b = delta.transform_point(d.b);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{abbe_number, Sellmeier};
    use crate::scene::{SourceKind, Spectrum};
    use crate::vec::Vec2;
    use std::f64::consts::FRAC_PI_2;

    fn square(half: f64) -> Vec<Edge> {
        vec![
            Edge::Line {
                a: Vec2::new(-half, -half),
                b: Vec2::new(half, -half),
            },
            Edge::Line {
                a: Vec2::new(half, -half),
                b: Vec2::new(half, half),
            },
            Edge::Line {
                a: Vec2::new(half, half),
                b: Vec2::new(-half, half),
            },
            Edge::Line {
                a: Vec2::new(-half, half),
                b: Vec2::new(-half, -half),
            },
        ]
    }

    fn elem(material: Material) -> Element {
        Element::new(ElementId(0), square(1.0), material)
    }

    /// A fragment holding one element of `material`, with `glasses`/`layers`
    /// as its own local tables.
    fn fragment_of(material: Material, glasses: Vec<Glass>, layers: Vec<Layer>) -> SceneFragment {
        SceneFragment {
            elements: vec![elem(material)],
            glasses,
            layers,
            ..Default::default()
        }
    }

    fn builtin_glass(name: &str) -> Glass {
        let cat = crate::material::GlassCatalog::builtin();
        cat.get(cat.find(name).unwrap()).unwrap().clone()
    }

    fn made_up_glass(name: &str) -> Glass {
        let sell = Sellmeier {
            b: [1.05, 0.22, 1.01],
            c: [0.006, 0.02, 103.0],
        };
        Glass {
            name: name.to_string(),
            nd: sell.index(crate::material::LAMBDA_D),
            vd: abbe_number(&sell),
            sell,
        }
    }

    // --- The reason id remapping exists ------------------------------------

    #[test]
    fn insert_remaps_glass_ids_to_the_destination_catalog() {
        // The fragment's GlassId(0) is N-SF11. In the destination's builtin
        // catalog, GlassId(0) is N-BK7. Inserting without remapping would leave
        // the element pointing at GlassId(0) and silently turn it into N-BK7 —
        // wrong optics, no error.
        let mut scene = Scene::new();
        let fragment = fragment_of(
            Material::Catalog(GlassId(0)),
            vec![builtin_glass("N-SF11")],
            vec![Layer::default()],
        );

        let inserted = scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        let id = match scene.elements[inserted.elements[0].index()].material {
            Material::Catalog(id) => id,
            ref m => panic!("expected a catalog glass, got {m:?}"),
        };
        assert_eq!(
            scene.catalog.get(id).unwrap().name,
            "N-SF11",
            "the element must still be N-SF11 after placement"
        );
        assert_eq!(id, scene.catalog.find("N-SF11").unwrap());
        assert_ne!(id, GlassId(0), "the local id must not survive verbatim");
    }

    #[test]
    fn insert_appends_a_glass_the_destination_lacks() {
        let mut scene = Scene::new();
        let before = scene.catalog.len();
        let fragment = fragment_of(
            Material::Catalog(GlassId(0)),
            vec![made_up_glass("HOUSE-CROWN")],
            vec![Layer::default()],
        );

        scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        assert_eq!(scene.catalog.len(), before + 1);
        let id = scene.catalog.find("HOUSE-CROWN").expect("glass was added");
        assert!(matches!(scene.elements[0].material, Material::Catalog(g) if g == id));
    }

    #[test]
    fn insert_merges_only_referenced_glasses() {
        // The fragment carries a whole catalog but references one glass. Merging
        // the unreferenced rest would pollute the destination and could raise a
        // conflict for a glass nobody uses.
        let mut scene = Scene::new();
        scene.catalog = crate::material::GlassCatalog::default();
        let fragment = fragment_of(
            Material::Catalog(GlassId(2)),
            vec![
                made_up_glass("UNUSED-A"),
                made_up_glass("UNUSED-B"),
                builtin_glass("N-BK7"),
            ],
            vec![Layer::default()],
        );

        scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        assert_eq!(scene.catalog.len(), 1, "only the referenced glass merges");
        assert!(scene.catalog.find("N-BK7").is_some());
        assert!(scene.catalog.find("UNUSED-A").is_none());
    }

    #[test]
    fn insert_rejects_a_conflicting_glass_and_leaves_the_scene_untouched() {
        let mut scene = Scene::new();
        let catalog_before = scene.catalog.len();
        // Same name as the builtin N-BK7, but different dispersion.
        let fragment = fragment_of(
            Material::Catalog(GlassId(0)),
            vec![made_up_glass("N-BK7")],
            vec![Layer::default()],
        );

        let err = scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap_err();

        assert_eq!(
            err,
            FragmentError::GlassConflict {
                name: "N-BK7".to_string()
            }
        );
        assert!(scene.elements.is_empty(), "insert must be atomic");
        assert_eq!(scene.catalog.len(), catalog_before);
    }

    #[test]
    fn insert_remaps_layer_ids() {
        let mut scene = Scene::new();
        let mut fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        fragment.layers = vec![Layer {
            name: "Module optics".to_string(),
            visible: true,
            opacity: 1.0,
        }];
        fragment.elements[0].layer = LayerId(0);

        scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        // "Module optics" is a new name, so it is appended, and the element must
        // follow it rather than keep pointing at local LayerId(0) = "Layer 0".
        assert_eq!(scene.layers.len(), 2);
        assert_eq!(scene.layers[1].name, "Module optics");
        assert_eq!(scene.elements[0].layer, LayerId(1));
    }

    #[test]
    fn insert_reuses_a_layer_of_the_same_name() {
        let mut scene = Scene::new();
        let fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);

        scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        assert_eq!(scene.layers.len(), 1, "matching name must not duplicate");
        assert_eq!(scene.elements[0].layer, LayerId(0));
    }

    // --- Placement ---------------------------------------------------------

    #[test]
    fn transformed_composes_onto_existing_object_transforms() {
        let mut fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        fragment.elements[0].transform = Isometry2::new(Vec2::new(2.0, 0.0), 0.25);
        fragment.sources.push(Source {
            kind: SourceKind::Collimated { half_width: 1.0 },
            transform: Isometry2::from_pos(Vec2::new(-5.0, 0.0)),
            spectrum: Spectrum::Monochromatic(0.5),
            flux: 1.0,
            rays: 3,
        });
        fragment.detectors.push(Detector::new(
            Vec2::new(10.0, -2.0),
            Vec2::new(10.0, 2.0),
            8,
        ));

        let t = Isometry2::new(Vec2::new(100.0, 50.0), FRAC_PI_2);
        let out = fragment.transformed(t).unwrap();

        // The element's own transform is preserved *through* the placement.
        assert_eq!(out.elements[0].transform, t.compose(Isometry2::new(Vec2::new(2.0, 0.0), 0.25)));
        assert_eq!(out.sources[0].transform, t.compose(Isometry2::from_pos(Vec2::new(-5.0, 0.0))));
        // Detectors carry no transform, so their endpoints move directly.
        assert!((out.detectors[0].a - t.transform_point(Vec2::new(10.0, -2.0))).length() < 1e-12);
        assert!((out.detectors[0].b - t.transform_point(Vec2::new(10.0, 2.0))).length() < 1e-12);
        // Boundary geometry is untouched: placement lives entirely in the transform.
        assert_eq!(out.elements[0].boundary, fragment.elements[0].boundary);
    }

    #[test]
    fn identity_placement_changes_nothing() {
        let mut fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        fragment.elements[0].transform = Isometry2::new(Vec2::new(2.0, -3.0), 0.4);
        let out = fragment.transformed(Transform2::IDENTITY).unwrap();
        assert_eq!(out, fragment);
    }

    #[test]
    fn from_scene_round_trips_through_insert() {
        let mut source_scene = Scene::new();
        source_scene.add_element(elem(Material::Catalog(GlassId(0))));
        source_scene.sources.push(Source {
            kind: SourceKind::Point { half_angle: 0.1 },
            transform: Isometry2::from_pos(Vec2::new(-4.0, 0.0)),
            spectrum: Spectrum::Monochromatic(0.55),
            flux: 2.0,
            rays: 4,
        });
        source_scene
            .detectors
            .push(Detector::new(Vec2::new(9.0, -1.0), Vec2::new(9.0, 1.0), 4));

        let fragment = SceneFragment::from_scene(&source_scene).unwrap();
        let mut dest = Scene::new();
        let inserted = dest
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        assert_eq!(inserted.elements.len(), 1);
        assert_eq!(inserted.sources.len(), 1);
        assert_eq!(inserted.detectors.len(), 1);
        assert_eq!(dest.elements[0].boundary, source_scene.elements[0].boundary);
        assert_eq!(dest.sources[0], source_scene.sources[0]);
        assert_eq!(dest.detectors[0], source_scene.detectors[0]);
    }

    #[test]
    fn inserted_ids_are_a_local_to_world_map() {
        let mut scene = Scene::new();
        scene.add_element(elem(Material::Absorber)); // occupy id 0
        let mut fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        fragment.elements.push(elem(Material::Mirror {
            reflectivity: 1.0,
        }));

        let inserted = scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        // Fragment order is preserved, so local index i -> inserted.elements[i].
        assert_eq!(inserted.elements, vec![ElementId(1), ElementId(2)]);
        // And the scene's own ids stay consistent with their positions.
        for (i, e) in scene.elements.iter().enumerate() {
            assert_eq!(e.id, ElementId(i as u32));
        }
    }

    #[test]
    fn two_fragments_do_not_collide() {
        let mut scene = Scene::new();
        let fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        let a = scene
            .insert_fragment(&fragment, Isometry2::from_pos(Vec2::new(0.0, 0.0)))
            .unwrap();
        let b = scene
            .insert_fragment(&fragment, Isometry2::from_pos(Vec2::new(50.0, 0.0)))
            .unwrap();
        assert_ne!(a.elements[0], b.elements[0]);
        assert_eq!(scene.elements.len(), 2);
    }

    // --- Rejections --------------------------------------------------------

    #[test]
    fn grin_is_refused_rather_than_left_behind() {
        // A GrinField's box is world-space and cannot follow a rotation, so
        // placing it must fail loudly instead of silently staying put.
        let mut scene = Scene::new();
        let fragment = fragment_of(
            Material::Grin(crate::material::GrinField {
                min: Vec2::new(-1.0, -1.0),
                max: Vec2::new(1.0, 1.0),
                base_index: 1.5,
            }),
            vec![],
            vec![Layer::default()],
        );

        assert_eq!(
            scene.insert_fragment(&fragment, Transform2::IDENTITY),
            Err(FragmentError::UnsupportedMaterial {
                element: 0,
                what: "Grin"
            })
        );
        assert!(scene.elements.is_empty());
    }

    #[test]
    fn non_finite_geometry_is_refused() {
        let mut scene = Scene::new();
        let mut fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        fragment.elements[0].boundary = vec![Edge::Line {
            a: Vec2::new(f64::NAN, 0.0),
            b: Vec2::new(1.0, 0.0),
        }];

        assert!(matches!(
            scene.insert_fragment(&fragment, Transform2::IDENTITY),
            Err(FragmentError::NonFinite { .. })
        ));
        assert!(scene.elements.is_empty());
    }

    #[test]
    fn non_finite_transform_is_refused() {
        let fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        assert!(matches!(
            fragment.transformed(Isometry2::new(Vec2::new(f64::INFINITY, 0.0), 0.0)),
            Err(FragmentError::NonFinite { .. })
        ));
    }

    #[test]
    fn a_dangling_glass_id_is_refused() {
        let mut scene = Scene::new();
        let fragment = fragment_of(Material::Catalog(GlassId(7)), vec![], vec![Layer::default()]);
        assert_eq!(
            scene.insert_fragment(&fragment, Transform2::IDENTITY),
            Err(FragmentError::UnknownGlass {
                element: 0,
                id: GlassId(7)
            })
        );
    }

    #[test]
    fn a_dangling_layer_id_is_refused() {
        let mut scene = Scene::new();
        let fragment = fragment_of(Material::Absorber, vec![], vec![]);
        assert_eq!(
            scene.insert_fragment(&fragment, Transform2::IDENTITY),
            Err(FragmentError::UnknownLayer {
                element: 0,
                id: LayerId(0)
            })
        );
    }

    // --- Removal -----------------------------------------------------------

    #[test]
    fn remove_renumbers_ids_and_reports_the_move() {
        let mut scene = Scene::new();
        for _ in 0..3 {
            scene.add_element(elem(Material::Absorber));
        }
        scene.elements[2].priority = 42; // tag the survivor

        let remap = scene
            .remove_objects(&ObjectIdSet::new().with_element(ElementId(0)))
            .unwrap();

        assert_eq!(scene.elements.len(), 2);
        assert_eq!(remap.element(ElementId(0)), None, "removed");
        assert_eq!(remap.element(ElementId(1)), Some(ElementId(0)));
        assert_eq!(remap.element(ElementId(2)), Some(ElementId(1)));
        // The survivor really is the object the remap points to.
        assert_eq!(scene.elements[1].priority, 42);
        // `Element::id` must track the new position.
        for (i, e) in scene.elements.iter().enumerate() {
            assert_eq!(e.id, ElementId(i as u32));
        }
    }

    #[test]
    fn remove_handles_every_object_kind() {
        let mut scene = Scene::new();
        scene.add_element(elem(Material::Absorber));
        scene.sources.push(Source {
            kind: SourceKind::Point { half_angle: 0.1 },
            transform: Isometry2::IDENTITY,
            spectrum: Spectrum::Monochromatic(0.5),
            flux: 1.0,
            rays: 2,
        });
        scene
            .detectors
            .push(Detector::new(Vec2::ZERO, Vec2::new(0.0, 1.0), 4));

        let all = ObjectIdSet::new()
            .with_element(ElementId(0))
            .with_source(SourceId(0))
            .with_detector(DetectorId(0));
        scene.remove_objects(&all).unwrap();

        assert!(scene.elements.is_empty());
        assert!(scene.sources.is_empty());
        assert!(scene.detectors.is_empty());
    }

    #[test]
    fn removing_an_inserted_fragment_leaves_no_orphans() {
        let mut scene = Scene::new();
        let mut fragment = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
        fragment.elements.push(elem(Material::Mirror {
            reflectivity: 0.9,
        }));
        fragment.detectors.push(Detector::new(
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            4,
        ));

        let inserted = scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();
        scene.remove_objects(&inserted.ids()).unwrap();

        assert!(scene.elements.is_empty());
        assert!(scene.detectors.is_empty());
    }

    #[test]
    fn remove_rejects_an_unknown_id_without_mutating() {
        let mut scene = Scene::new();
        scene.add_element(elem(Material::Absorber));
        assert_eq!(
            scene.remove_objects(&ObjectIdSet::new().with_element(ElementId(9))),
            Err(FragmentError::UnknownObject {
                what: "element",
                id: 9
            })
        );
        assert_eq!(scene.elements.len(), 1, "remove must be atomic");
    }

    #[test]
    fn remap_set_drops_removed_and_moves_survivors() {
        let mut scene = Scene::new();
        for _ in 0..3 {
            scene.add_element(elem(Material::Absorber));
        }
        let owned = ObjectIdSet::new()
            .with_element(ElementId(1))
            .with_element(ElementId(2));

        let remap = scene
            .remove_objects(&ObjectIdSet::new().with_element(ElementId(1)))
            .unwrap();

        // An ownership set survives the renumbering: 1 is gone, 2 slid down to 1.
        assert_eq!(
            remap.remap_set(&owned),
            ObjectIdSet::new().with_element(ElementId(1))
        );
    }

    // --- Batch transform ---------------------------------------------------

    #[test]
    fn transform_objects_moves_a_batch_together() {
        let mut scene = Scene::new();
        let fragment = {
            let mut f = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
            f.elements[0].transform = Isometry2::from_pos(Vec2::new(1.0, 0.0));
            f.detectors
                .push(Detector::new(Vec2::new(5.0, -1.0), Vec2::new(5.0, 1.0), 4));
            f
        };
        let inserted = scene
            .insert_fragment(&fragment, Transform2::IDENTITY)
            .unwrap();

        let delta = Isometry2::new(Vec2::new(10.0, 20.0), FRAC_PI_2);
        scene.transform_objects(&inserted.ids(), delta).unwrap();

        assert_eq!(
            scene.elements[0].transform,
            delta.compose(Isometry2::from_pos(Vec2::new(1.0, 0.0)))
        );
        assert!((scene.detectors[0].a - delta.transform_point(Vec2::new(5.0, -1.0))).length() < 1e-12);
    }

    #[test]
    fn transform_objects_matches_inserting_at_the_final_pose() {
        // Placing at A then nudging by B must equal placing at B∘A outright —
        // this is what lets a drag be applied incrementally.
        let fragment = {
            let mut f = fragment_of(Material::Absorber, vec![], vec![Layer::default()]);
            f.elements[0].transform = Isometry2::new(Vec2::new(2.0, 1.0), 0.3);
            f.detectors
                .push(Detector::new(Vec2::new(5.0, -1.0), Vec2::new(5.0, 1.0), 4));
            f
        };
        let a = Isometry2::new(Vec2::new(4.0, -2.0), 0.5);
        let b = Isometry2::new(Vec2::new(-7.0, 3.0), 1.1);

        let mut incremental = Scene::new();
        let ins = incremental.insert_fragment(&fragment, a).unwrap();
        incremental.transform_objects(&ins.ids(), b).unwrap();

        let mut direct = Scene::new();
        direct.insert_fragment(&fragment, b.compose(a)).unwrap();

        // Tolerance, not bit-equality: the two routes associate the same
        // rotations in a different order, which floating point rounds by ~1 ULP.
        assert!(
            (incremental.elements[0].transform.pos - direct.elements[0].transform.pos).length()
                < 1e-12
        );
        assert!(
            (incremental.elements[0].transform.rot - direct.elements[0].transform.rot).abs() < 1e-12
        );
        assert!((incremental.detectors[0].a - direct.detectors[0].a).length() < 1e-12);
    }

    #[test]
    fn transform_objects_rejects_an_unknown_id_without_mutating() {
        let mut scene = Scene::new();
        scene.add_element(elem(Material::Absorber));
        let before = scene.elements[0].transform;

        assert!(scene
            .transform_objects(
                &ObjectIdSet::new()
                    .with_element(ElementId(0))
                    .with_element(ElementId(9)),
                Isometry2::from_pos(Vec2::new(1.0, 1.0)),
            )
            .is_err());

        // Element 0 is valid and listed first, but the batch is atomic.
        assert_eq!(scene.elements[0].transform, before);
    }
}
