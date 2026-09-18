//! # oc-wasm
//!
//! The bridge the web shell talks to (spec §2: "JS holds UI state and an opaque
//! handle"). Scene state lives here in Rust; JS mutates it and asks for freshly
//! traced segments — no scene graph is mirrored across the boundary.
//!
//! Two tracers are exposed, per the spec's tiering (§4.1):
//! - [`Canvas::trace_colored`] — the **interactive** f32 path (`oc-gpu` CPU
//!   backend), for the picture. Rays carry wavelength → spectral colour.
//! - [`Canvas::detector_json`] — the **reference** f64 path (`oc-trace`), for
//!   any displayed *number*. Never read metrics off the f32 path.

use std::collections::{HashMap, HashSet};

use oc_core::{
    convex_hull, point_in_hull, reference_glasses, ConicParams, Detector, Edge, Element, ElementId,
    Isometry2, Layer, LayerId, Material, Scene, Source, SourceKind, Spectrum, Vec2,
};
use oc_fit::{fit_stroke, sample_edge, surface_rim, Fit, FitLadder, DEFAULT_TOL_UM};
use oc_gpu::{color::wavelength_um_to_linear, scene::GpuScene, trace_cpu, BuildOpts};
use oc_trace::{trace as ref_trace, TraceConfig};
use wasm_bindgen::prelude::*;

// --- 3D (Scene3) view surface (PR 3D-09) -----------------------------------
use oc_core::build3::{
    response_kind, singlet_face_sag, DichroicSpec, SingletAspheres, SingletGeom,
};
use oc_core::migrate_v1::{load_any, LoadReport};
use oc_core::planar_reduce::{preview_mode, PreviewMode};
use oc_core::profile2::ProfileConstruction3;
use oc_core::scene3::Scene3;
use oc_core::vec3::{Isometry3, Plane3, Vec3};
use oc_trace::{trace_scene3, TraceConfig3};

/// The openUC2 mode (§15.2). The freeform Canvas above does not know it exists.
pub mod openuc2;
// The WebGPU half of the interactive 3D preview.
pub mod gpu_preview;

/// Cross-section of the 3D scene with the view plane (§10.4), for drawing the
/// optical elements in the 2D layout.
mod xsection;

/// Render descriptors and display meshes for the GPU 3D view.
pub mod render_view;

/// State kept for an element drawn freehand: its fitted ladder (§5.3) so the
/// complexity slider can re-swap the front surface with no refit, plus the
/// extrusion depth used to close the body.
struct FittedSurface {
    ladder: FitLadder,
    rung: usize,
    thickness: f64,
}

/// One node of the non-linear undo history (§5.2: "Undo is a tree, not a
/// stack"). Scenes are kilobytes, so every commit snapshots the whole document
/// as JSON; branch points are where the variation gallery (§5.6) reads its
/// live variants from. Freehand fit-ladders are *not* snapshotted (they are not
/// serializable) — an undo past a freehand commit keeps the geometry but drops
/// that element's complexity slider.
struct HistoryNode {
    parent: Option<usize>,
    children: Vec<usize>,
    label: String,
    scene_json: String,
    /// Sorted `(element_id, param_code)` pins, so they survive undo (§5.2).
    pins: Vec<(u32, u8)>,
    active_layer: u32,
    /// Monotonic id shown in the sidebar; stable as the tree grows.
    seq: u32,
}

/// The undo tree: a flat arena of [`HistoryNode`]s plus the current position.
struct History {
    nodes: Vec<HistoryNode>,
    current: usize,
    next_seq: u32,
}

impl History {
    fn single(scene_json: String, pins: Vec<(u32, u8)>, active_layer: u32, label: &str) -> Self {
        History {
            nodes: vec![HistoryNode {
                parent: None,
                children: Vec::new(),
                label: label.to_string(),
                scene_json,
                pins,
                active_layer,
                seq: 0,
            }],
            current: 0,
            next_seq: 1,
        }
    }
}

/// Editable parameters that can be individually pinned (§5.1/§5.2). The returned
/// code is the discriminant stored in the pin set.
fn param_code(name: &str) -> Option<u8> {
    Some(match name {
        "x" => 0,
        "y" => 1,
        "rot" => 2,
        "thickness" => 4,
        "fit" => 7,
        // Per-surface (front = 1, rear = 2), so a doublet's two radii / apertures
        // / conic constants are pinned and edited independently (user request).
        "radius1" => 8,
        "radius2" => 9,
        "half_ap1" => 10,
        "half_ap2" => 11,
        "conic_k1" => 12,
        "conic_k2" => 13,
        _ => return None,
    })
}

/// The opaque handle JS holds. Owns the authoritative scene.
#[wasm_bindgen]
pub struct Canvas {
    scene: Scene,
    /// Ghost reflection-generation cap `G` (§4.2) — the only bounce knob the UI
    /// exposes. The primary transmitted path is *not* limited by a magic number:
    /// its depth is auto-sized to the scene's surface count (see
    /// [`auto_max_depth`]), so adding elements never truncates the rays.
    max_ghost: u32,
    ghosts: bool,
    /// Freehand-fitted elements, keyed by element id (§5.3).
    fitted: HashMap<u32, FittedSurface>,
    /// Pinned `(element_id, param_code)` pairs — locked against edits (§5.2).
    param_pins: HashSet<(u32, u8)>,
    /// Layer new elements are assigned to (§5.6).
    active_layer: u32,
    /// Non-linear undo history (§5.2). Lazily reset by JS after the demo scene
    /// is built, so undo does not walk back into an empty document.
    history: History,
    /// The canonical 3D scene, once a v2 (or migrated v1) document is loaded
    /// (§3.1, PR 3D-09). `None` until the browser opens one; the v1 `scene`
    /// above still drives the freeform demo. The 2D Canvas becomes a *view* of
    /// this: a slice on [`Canvas::view_plane`].
    scene3: Option<Scene3>,
    /// The plane the 2D Canvas slices/projects (default `Z = 0`, §10.1).
    view_plane: Plane3,
    /// Slice half-thickness in mm for distance-from-plane filtering (§10.4).
    slice_thickness: f64,
    /// Bumped on every mutation of `scene3`, so an async f64 result carrying an
    /// older revision is discarded rather than shown (§3.3, §13).
    revision: u64,
    /// Per-object geometry/transform/material revisions for the GPU 3D view.
    render_tracker: render_view::RenderTracker,
    /// The area detectors' flux maps from the most recent
    /// [`Canvas::trace_projected_3d`] (see [`Canvas::last_detector_maps_f32`]).
    detector_maps: std::cell::RefCell<Vec<f32>>,
}

#[wasm_bindgen]
impl Canvas {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Canvas {
        let scene = Scene::new();
        let history = History::single(scene.to_json(), Vec::new(), 0, "start");
        Canvas {
            scene,
            // Two ghost generations by default: enough for the classic
            // two-surface ghost image (light reflects twice, then reaches the
            // detector), still cheap. Raise it to hunt deeper stray light (§4.2).
            max_ghost: 2,
            ghosts: false,
            fitted: HashMap::new(),
            param_pins: HashSet::new(),
            active_layer: 0,
            history,
            scene3: None,
            view_plane: Plane3::XY,
            slice_thickness: 0.0,
            revision: 0,
            render_tracker: render_view::RenderTracker::default(),
            detector_maps: std::cell::RefCell::new(Vec::new()),
        }
    }

    // --- configuration -----------------------------------------------------

    /// Set the ghost reflection-generation cap `G` (§4.2): how many nested
    /// Fresnel back-reflections to follow. `0` = none, `1` = single ghosts, `2`
    /// = ghosts-of-ghosts, … This is the ray-explosion knob; the number of
    /// *elements* is unlimited (the primary path auto-sizes to the scene).
    #[wasm_bindgen(js_name = setMaxGhostBounces)]
    pub fn set_max_ghost_bounces(&mut self, g: u32) {
        self.max_ghost = g.min(8);
    }

    #[wasm_bindgen(js_name = maxGhostBounces)]
    pub fn max_ghost_bounces(&self) -> u32 {
        self.max_ghost
    }

    /// Toggle the ghost layer (Fresnel back-reflections, §5.5).
    #[wasm_bindgen(js_name = setGhosts)]
    pub fn set_ghosts(&mut self, on: bool) {
        self.ghosts = on;
    }

    // --- scene construction ------------------------------------------------

    /// Add a biconvex singlet centred at `(x, y)`. Returns its element id.
    #[wasm_bindgen(js_name = addSinglet)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_singlet(
        &mut self,
        x: f64,
        y: f64,
        radius: f64,
        thickness: f64,
        half_ap: f64,
        nd: f64,
        vd: f64,
    ) -> u32 {
        let c = 1.0 / radius;
        let front = ConicParams {
            vertex: Vec2::new(-thickness / 2.0, 0.0),
            axis: Vec2::new(1.0, 0.0),
            c,
            k: 0.0,
            half_ap,
        };
        let rear = ConicParams {
            vertex: Vec2::new(thickness / 2.0, 0.0),
            axis: Vec2::new(-1.0, 0.0),
            c,
            k: 0.0,
            half_ap,
        };
        let sag = c * half_ap * half_ap / (1.0 + (1.0 - c * c * half_ap * half_ap).sqrt());
        let (ftip, rtip) = (-thickness / 2.0 + sag, thickness / 2.0 - sag);
        let mut e = Element::new(
            ElementId(0),
            vec![
                Edge::Conic(front),
                Edge::Line {
                    a: Vec2::new(ftip, half_ap),
                    b: Vec2::new(rtip, half_ap),
                },
                Edge::Conic(rear),
                Edge::Line {
                    a: Vec2::new(rtip, -half_ap),
                    b: Vec2::new(ftip, -half_ap),
                },
            ],
            Material::AbbePoint { nd, vd },
        );
        e.transform = Isometry2::from_pos(Vec2::new(x, y));
        self.add_on_active(e).0
    }

    /// Add a plane mirror segment from `(x0,y0)` to `(x1,y1)`.
    #[wasm_bindgen(js_name = addMirror)]
    pub fn add_mirror(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) -> u32 {
        let a = Vec2::new(x0, y0);
        let b = Vec2::new(x1, y1);
        let e = Element::new(
            ElementId(0),
            vec![Edge::Line { a, b }, Edge::Line { a: b, b: a }],
            Material::Mirror { reflectivity: 1.0 },
        );
        self.add_on_active(e).0
    }

    /// Add a collimated monochromatic source. `aim_deg` is the beam direction.
    #[wasm_bindgen(js_name = addCollimatedSource)]
    pub fn add_collimated_source(
        &mut self,
        x: f64,
        y: f64,
        aim_deg: f64,
        half_width: f64,
        rays: u32,
        lambda_um: f64,
    ) {
        self.scene.sources.push(Source {
            kind: SourceKind::Collimated { half_width },
            transform: Isometry2::new(Vec2::new(x, y), aim_deg.to_radians()),
            spectrum: Spectrum::Monochromatic(lambda_um),
            flux: 1.0,
            rays,
        });
    }

    /// Add a collimated white beam. Sampled with a dense equal-energy spectrum
    /// (`lines` wavelengths across the visible) so the additive sum is neutral
    /// white and dispersion resolves as a smooth rainbow (§5.4/§5.5).
    #[wasm_bindgen(js_name = addWhiteBeam)]
    pub fn add_white_beam(
        &mut self,
        x: f64,
        y: f64,
        aim_deg: f64,
        half_width: f64,
        rays: u32,
        lines: u32,
    ) {
        let n = lines.max(2);
        let (lo, hi) = (0.420, 0.660); // µm
        let spectrum: Vec<(f64, f64)> = (0..n)
            .map(|i| (lo + (hi - lo) * i as f64 / (n - 1) as f64, 1.0))
            .collect();
        self.scene.sources.push(Source {
            kind: SourceKind::Collimated { half_width },
            transform: Isometry2::new(Vec2::new(x, y), aim_deg.to_radians()),
            spectrum: Spectrum::Lines(spectrum),
            flux: 1.0,
            rays,
        });
    }

    /// Add a detector segment. Returns its index.
    #[wasm_bindgen(js_name = addDetector)]
    pub fn add_detector(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, bins: usize) -> usize {
        let idx = self.scene.detectors.len();
        self.scene
            .detectors
            .push(Detector::new(Vec2::new(x0, y0), Vec2::new(x1, y1), bins));
        idx
    }

    // --- manipulation ------------------------------------------------------

    #[wasm_bindgen(js_name = moveElement)]
    pub fn move_element(&mut self, id: u32, x: f64, y: f64) {
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            e.transform.pos = Vec2::new(x, y);
        }
    }

    #[wasm_bindgen(js_name = rotateElement)]
    pub fn rotate_element(&mut self, id: u32, rot: f64) {
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            e.transform.rot = rot;
        }
    }

    /// Set the spatial ray count of a source — raise it to fill the beam into a
    /// continuous, homogeneous wavefront (brightness is flux-normalized, so the
    /// picture stays balanced as rays fill in).
    #[wasm_bindgen(js_name = setSourceRays)]
    pub fn set_source_rays(&mut self, idx: usize, rays: u32) {
        if let Some(s) = self.scene.sources.get_mut(idx) {
            s.rays = rays.max(1);
        }
    }

    /// Apply an Abbe-map point `(n_d, V_d)` to an element (the picker's live
    /// feedback, §5.4). Uses the closed-form Cauchy dispersion.
    #[wasm_bindgen(js_name = setElementAbbe)]
    pub fn set_element_abbe(&mut self, id: u32, nd: f64, vd: f64) {
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            e.material = Material::AbbePoint { nd, vd };
        }
    }

    /// Snap an element to a named catalog glass (full Sellmeier). Returns true
    /// if the name was found.
    #[wasm_bindgen(js_name = setElementGlass)]
    pub fn set_element_glass(&mut self, id: u32, name: &str) -> bool {
        if let (Some(gid), Some(e)) = (
            self.scene.catalog.find(name),
            self.scene.elements.get_mut(id as usize),
        ) {
            e.material = Material::Catalog(gid);
            true
        } else {
            false
        }
    }

    // --- the Abbe palette (§5.4) -------------------------------------------

    /// The glass scatter as JSON: `[{"name","nd","vd"}, ...]`.
    #[wasm_bindgen(js_name = abbeGlassesJson)]
    pub fn abbe_glasses_json(&self) -> String {
        let items: Vec<String> = reference_glasses()
            .iter()
            .map(|g| {
                format!(
                    "{{\"name\":\"{}\",\"nd\":{},\"vd\":{}}}",
                    g.name, g.nd, g.vd
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    /// The convex hull of the real-glass region as a flat `[vd, nd, ...]` list.
    /// Outside it, glass is "unobtainable" (hatch it in the UI).
    #[wasm_bindgen(js_name = abbeHull)]
    pub fn abbe_hull(&self) -> Vec<f64> {
        let pts: Vec<(f64, f64)> = reference_glasses().iter().map(|g| (g.vd, g.nd)).collect();
        let mut out = Vec::new();
        for (vd, nd) in convex_hull(&pts) {
            out.push(vd);
            out.push(nd);
        }
        out
    }

    /// Is `(n_d, V_d)` a physically obtainable glass (inside the hull)?
    #[wasm_bindgen(js_name = isObtainable)]
    pub fn is_obtainable(&self, nd: f64, vd: f64) -> bool {
        let pts: Vec<(f64, f64)> = reference_glasses().iter().map(|g| (g.vd, g.nd)).collect();
        point_in_hull(vd, nd, &convex_hull(&pts))
    }

    // --- tracing -----------------------------------------------------------

    /// Interactive spectral trace. Flat array, **9 floats per segment**:
    /// `[ax, ay, bx, by, r, g, b, flux, flags]`. Colour is CIE→sRGB of the
    /// ray's wavelength; `flags` bit1 = TIR, bit2 = ghost.
    #[wasm_bindgen(js_name = traceColored)]
    pub fn trace_colored(&self) -> Vec<f32> {
        colored_segments(&self.visible_scene(), self.ghosts, self.max_ghost)
    }

    /// Legacy M1 trace: flat `[ax, ay, bx, by, flux]` per segment.
    pub fn trace(&self) -> Vec<f32> {
        let vs = self.visible_scene();
        let d = auto_max_depth(&vs, 0, false);
        let gs = GpuScene::build_with(
            &vs,
            BuildOpts {
                max_segs: d + 8,
                ..BuildOpts::default()
            },
        );
        let per_ray = trace_cpu(&gs, d);
        let mut out = Vec::new();
        for segs in &per_ray {
            for s in segs {
                out.extend_from_slice(&[s.a.x, s.a.y, s.b.x, s.b.y, s.flux]);
            }
        }
        out
    }

    /// Reference (f64) detector readout as JSON — the honest numbers (§4.1).
    /// `{"hits","flux","centroid","opl","bins":[...]}`; `null` if no such
    /// detector or no hits.
    #[wasm_bindgen(js_name = detectorJson)]
    pub fn detector_json(&self, idx: usize) -> String {
        if idx >= self.scene.detectors.len() {
            return "null".to_string();
        }
        let vs = self.visible_scene();
        let cfg = TraceConfig {
            ghosts: self.ghosts,
            max_depth: auto_max_depth(&vs, self.max_ghost, self.ghosts),
            max_ghost_bounces: self.max_ghost,
            ..TraceConfig::default()
        };
        let res = ref_trace(&vs, &cfg);
        let d = &res.detectors[idx];
        let bins: Vec<String> = d.bins.iter().map(|v| format!("{v}")).collect();
        let centroid = d.centroid_u().map_or("null".into(), |v| format!("{v}"));
        let opl = d.mean_opl().map_or("null".into(), |v| format!("{v}"));
        format!(
            "{{\"hits\":{},\"flux\":{},\"centroid\":{},\"opl\":{},\"bins\":[{}]}}",
            d.hits,
            d.total_flux,
            centroid,
            opl,
            bins.join(",")
        )
    }

    // --- misc --------------------------------------------------------------

    #[wasm_bindgen(js_name = elementCount)]
    pub fn element_count(&self) -> usize {
        self.scene.elements.len()
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> String {
        self.scene.to_json()
    }

    /// Replace the whole scene from an `.ocanvas` document (the "Open…" action).
    /// The JSON is version-checked (a future-version file is rejected), then the
    /// scene, pins, active layer, and freehand-fit state are reset and the undo
    /// history is re-rooted so undo cannot walk back into the previous document.
    /// Throws with a readable message if the document is not loadable.
    #[wasm_bindgen(js_name = loadJSON)]
    pub fn load_json(&mut self, json: &str) -> Result<(), JsError> {
        let scene = Scene::from_json(json).map_err(|e| JsError::new(&e.to_string()))?;
        self.scene = scene;
        self.fitted.clear();
        self.param_pins.clear();
        self.active_layer = self
            .active_layer
            .min(self.scene.layers.len().saturating_sub(1) as u32);
        self.history = History::single(
            self.scene.to_json(),
            self.pins_sorted(),
            self.active_layer,
            "open",
        );
        Ok(())
    }

    // === M3: freehand → fit (§5.3) =========================================

    /// Fit a freehand stroke (`flat_xy` = `[x0,y0,x1,y1,...]`, world mm) against
    /// the whole complexity ladder **without committing** — the live preview
    /// while drawing. Returns JSON:
    /// `{chosen, tol_um, rungs:[{level,name,dof,residual_um,rms_um,curve:[x,y,…]}]}`.
    /// `curve` is a sampled polyline of that rung's fitted surface, so the
    /// complexity slider can redraw any rung for free (no refit).
    #[wasm_bindgen(js_name = fitStroke)]
    pub fn fit_stroke_json(&self, flat_xy: &[f64], tol_um: f64) -> String {
        let pts = flat_to_points(flat_xy);
        if pts.len() < 2 {
            return "null".to_string();
        }
        let tol = if tol_um > 0.0 { tol_um } else { DEFAULT_TOL_UM };
        let ladder = fit_stroke(&pts, tol);
        let rungs: Vec<String> = ladder
            .fits
            .iter()
            .map(|f| {
                let curve = sample_edge(&f.edge, 48);
                let flat: Vec<String> = curve
                    .iter()
                    .flat_map(|p| [format!("{:.4}", p.x), format!("{:.4}", p.y)])
                    .collect();
                format!("{},\"curve\":[{}]}}", rung_json_open(f), flat.join(","))
            })
            .collect();
        format!(
            "{{\"chosen\":{},\"tol_um\":{},\"rungs\":[{}]}}",
            ladder.chosen,
            tol,
            rungs.join(",")
        )
    }

    /// Commit a freehand stroke as an optical element: fit the stroke, extrude
    /// the selected rung's surface into a thin glass body, and add it. Pass
    /// `rung = -1` to use the auto-selected rung. Returns the new element id, or
    /// `-1` for a degenerate stroke.
    #[wasm_bindgen(js_name = addFittedElement)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_fitted_element(
        &mut self,
        flat_xy: &[f64],
        tol_um: f64,
        rung: i32,
        thickness: f64,
        nd: f64,
        vd: f64,
        mirror: bool,
    ) -> i32 {
        let pts = flat_to_points(flat_xy);
        if pts.len() < 2 {
            return -1;
        }
        let tol = if tol_um > 0.0 { tol_um } else { DEFAULT_TOL_UM };
        let ladder = fit_stroke(&pts, tol);
        let rung = clamp_rung(rung, &ladder);
        let thickness = thickness.max(0.5);
        let front = ladder.at_rung(rung).edge.clone();
        let body = plano_body(front, thickness);
        // A drawn surface becomes a reflective body (a curved front-surface
        // mirror) or a glass one, per the draw-tool material toggle.
        let material = if mirror {
            Material::Mirror { reflectivity: 1.0 }
        } else {
            Material::AbbePoint { nd, vd }
        };
        let e = Element::new(ElementId(0), body, material);
        let id = self.add_on_active(e).0;
        self.fitted.insert(
            id,
            FittedSurface {
                ladder,
                rung,
                thickness,
            },
        );
        id as i32
    }

    /// Move a fitted element to a different rung of its ladder (the complexity
    /// slider, §5.3) — re-extrudes the body in place. Returns the new max
    /// residual in microns, or `-1` if `id` is not a fitted element / is pinned.
    #[wasm_bindgen(js_name = setElementFitRung)]
    pub fn set_element_fit_rung(&mut self, id: u32, rung: i32) -> f64 {
        if self.is_pinned(id, 7) {
            return -1.0;
        }
        let (front, thickness, res_um, new_rung) = match self.fitted.get(&id) {
            Some(fs) => {
                let r = clamp_rung(rung, &fs.ladder);
                let fit = fs.ladder.at_rung(r);
                (fit.edge.clone(), fs.thickness, fit.max_residual_um(), r)
            }
            None => return -1.0,
        };
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            e.boundary = plano_body(front, thickness);
        }
        if let Some(fs) = self.fitted.get_mut(&id) {
            fs.rung = new_rung;
        }
        res_um
    }

    /// The fitted ladder for a committed element (§5.3), for the inspector's
    /// complexity slider: `{current, chosen, tol_um, rungs:[{level,name,dof,
    /// residual_um,rms_um}]}`. `null` if `id` is not a fitted element.
    #[wasm_bindgen(js_name = elementFitJson)]
    pub fn element_fit_json(&self, id: u32) -> String {
        let Some(fs) = self.fitted.get(&id) else {
            return "null".to_string();
        };
        let rungs: Vec<String> = fs
            .ladder
            .fits
            .iter()
            .map(|f| format!("{}}}", rung_json_open(f)))
            .collect();
        format!(
            "{{\"current\":{},\"chosen\":{},\"tol_um\":{},\"rungs\":[{}]}}",
            fs.rung,
            fs.ladder.chosen,
            fs.ladder.tol_um,
            rungs.join(",")
        )
    }

    // === M3: numeric inspector — bidirectional binding (§5.2) ==============

    /// The element's rigid transform: `{"x","y","rot_deg"}`, or `null`.
    #[wasm_bindgen(js_name = elementTransformJson)]
    pub fn element_transform_json(&self, id: u32) -> String {
        match self.scene.elements.get(id as usize) {
            Some(e) => format!(
                "{{\"x\":{},\"y\":{},\"rot_deg\":{}}}",
                e.transform.pos.x,
                e.transform.pos.y,
                e.transform.rot.to_degrees()
            ),
            None => "null".to_string(),
        }
    }

    /// Set position, honouring per-axis pins and the element lock (§5.2). A
    /// pinned coordinate keeps its current value. Returns true if anything moved.
    #[wasm_bindgen(js_name = setElementPos)]
    pub fn set_element_pos(&mut self, id: u32, x: f64, y: f64) -> bool {
        if self.locked(id) {
            return false;
        }
        let (px, py) = (
            self.param_pins.contains(&(id, 0)),
            self.param_pins.contains(&(id, 1)),
        );
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            if !px {
                e.transform.pos.x = x;
            }
            if !py {
                e.transform.pos.y = y;
            }
            true
        } else {
            false
        }
    }

    /// Set rotation in degrees, honouring the `rot` pin and lock.
    #[wasm_bindgen(js_name = setElementRotDeg)]
    pub fn set_element_rot_deg(&mut self, id: u32, deg: f64) -> bool {
        if self.is_pinned(id, 2) {
            return false;
        }
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            e.transform.rot = deg.to_radians();
            true
        } else {
            false
        }
    }

    /// A recognised singlet's **per-surface** parameters:
    /// `{"radius1","radius2","thickness","half_ap1","half_ap2","conic_k1",
    /// "conic_k2"}` (front = 1, rear = 2), or `null` if the element is not a
    /// two-conic singlet.
    #[wasm_bindgen(js_name = singletParamsJson)]
    pub fn singlet_params_json(&self, id: u32) -> String {
        match self.scene.elements.get(id as usize).and_then(read_singlet) {
            Some(s) => format!(
                "{{\"radius1\":{},\"radius2\":{},\"thickness\":{},\"half_ap1\":{},\"half_ap2\":{},\"conic_k1\":{},\"conic_k2\":{}}}",
                s.r1, s.r2, s.t, s.ha1, s.ha2, s.k1, s.k2
            ),
            None => "null".to_string(),
        }
    }

    /// Rebuild a singlet from **per-surface** numeric parameters (the inspector's
    /// numeric twin, §5.2). The two faces are independent: `radius1`/`radius2`,
    /// `half_ap1`/`half_ap2`, `conic_k1`/`conic_k2`. Each parameter honours its
    /// own pin; the whole edit honours the element lock. Returns true on success.
    #[wasm_bindgen(js_name = setSingletParams)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_singlet_params(
        &mut self,
        id: u32,
        radius1: f64,
        radius2: f64,
        thickness: f64,
        half_ap1: f64,
        half_ap2: f64,
        conic_k1: f64,
        conic_k2: f64,
    ) -> bool {
        if self.locked(id) {
            return false;
        }
        let cur = match self.scene.elements.get(id as usize).and_then(read_singlet) {
            Some(c) => c,
            None => return false,
        };
        let keep = |code: u8| self.param_pins.contains(&(id, code));
        let r1 = if keep(8) { cur.r1 } else { radius1 };
        let r2 = if keep(9) { cur.r2 } else { radius2 };
        let t = if keep(4) { cur.t } else { thickness };
        let ha1 = if keep(10) { cur.ha1 } else { half_ap1 };
        let ha2 = if keep(11) { cur.ha2 } else { half_ap2 };
        let k1 = if keep(12) { cur.k1 } else { conic_k1 };
        let k2 = if keep(13) { cur.k2 } else { conic_k2 };
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            e.boundary = singlet_boundary(r1, r2, t, ha1.max(0.1), ha2.max(0.1), k1, k2);
            true
        } else {
            false
        }
    }

    // === M3: snapping (§5.2) ===============================================

    /// Snap a proposed position for element `id` to nearby optical guides: the
    /// optical axis (`y = 0`), and vertex alignment (matching `x`/`y`) with the
    /// other elements, within `thresh` world units. Returns
    /// `{"x","y","snaps":[…],"guideX","guideY"}`; a `guide` is `null` when that
    /// axis did not snap. Enable it from the UI by holding a modifier (§5.2).
    #[wasm_bindgen(js_name = snapPointJson)]
    pub fn snap_point_json(&self, id: u32, x: f64, y: f64, thresh: f64) -> String {
        let (mut sx, mut sy) = (x, y);
        let mut snaps: Vec<&str> = Vec::new();
        let mut guide_x = f64::NAN;
        let mut guide_y = f64::NAN;

        // Optical axis wins for the y coordinate.
        if y.abs() <= thresh {
            sy = 0.0;
            guide_y = 0.0;
            snaps.push("axis");
        }
        // Vertex alignment with other elements.
        let (mut bx, mut by) = (thresh, thresh);
        let (mut gx, mut gy) = (f64::NAN, f64::NAN);
        for (i, e) in self.scene.elements.iter().enumerate() {
            if i as u32 == id {
                continue;
            }
            let p = e.transform.pos;
            if (p.x - x).abs() < bx {
                bx = (p.x - x).abs();
                gx = p.x;
            }
            if (p.y - y).abs() < by {
                by = (p.y - y).abs();
                gy = p.y;
            }
        }
        if gx.is_finite() {
            sx = gx;
            guide_x = gx;
            snaps.push("vertexX");
        }
        if gy.is_finite() && !snaps.contains(&"axis") {
            sy = gy;
            guide_y = gy;
            snaps.push("vertexY");
        }

        let names: Vec<String> = snaps.iter().map(|s| format!("\"{s}\"")).collect();
        let jx = if guide_x.is_finite() {
            format!("{guide_x}")
        } else {
            "null".into()
        };
        let jy = if guide_y.is_finite() {
            format!("{guide_y}")
        } else {
            "null".into()
        };
        format!(
            "{{\"x\":{},\"y\":{},\"snaps\":[{}],\"guideX\":{},\"guideY\":{}}}",
            sx,
            sy,
            names.join(","),
            jx,
            jy
        )
    }

    // === M3: pinning / locking (§5.1/§5.2) =================================

    /// Lock (or unlock) a whole element — it then rejects every edit and is
    /// drawn ghosted in the UI (§5.1). Returns the new locked state.
    #[wasm_bindgen(js_name = setElementLocked)]
    pub fn set_element_locked(&mut self, id: u32, locked: bool) -> bool {
        if let Some(e) = self.scene.elements.get_mut(id as usize) {
            e.locked = locked;
        }
        locked
    }

    #[wasm_bindgen(js_name = isElementLocked)]
    pub fn is_element_locked(&self, id: u32) -> bool {
        self.locked(id)
    }

    /// Pin (or unpin) a single named parameter of an element: one of
    /// `x`,`y`,`rot`,`radius`,`thickness`,`conic_k`,`half_ap`,`fit`. Returns the
    /// new pinned state.
    #[wasm_bindgen(js_name = setParamPinned)]
    pub fn set_param_pinned(&mut self, id: u32, name: &str, pinned: bool) -> bool {
        if let Some(code) = param_code(name) {
            if pinned {
                self.param_pins.insert((id, code));
            } else {
                self.param_pins.remove(&(id, code));
            }
            pinned
        } else {
            false
        }
    }

    #[wasm_bindgen(js_name = isParamPinned)]
    pub fn is_param_pinned(&self, id: u32, name: &str) -> bool {
        param_code(name)
            .map(|code| self.param_pins.contains(&(id, code)))
            .unwrap_or(false)
    }

    /// The pinned parameter names for an element, as a JSON string array.
    #[wasm_bindgen(js_name = pinnedParamsJson)]
    pub fn pinned_params_json(&self, id: u32) -> String {
        const NAMES: [&str; 11] = [
            "x",
            "y",
            "rot",
            "thickness",
            "fit",
            "radius1",
            "radius2",
            "half_ap1",
            "half_ap2",
            "conic_k1",
            "conic_k2",
        ];
        let names: Vec<String> = NAMES
            .iter()
            .filter(|n| {
                param_code(n)
                    .map(|c| self.param_pins.contains(&(id, c)))
                    .unwrap_or(false)
            })
            .map(|n| format!("\"{n}\""))
            .collect();
        format!("[{}]", names.join(","))
    }

    // === M3: delete / copy / paste (element lifecycle) =====================

    /// Delete an element and re-index the fitted-surface and pin tables so ids
    /// stay equal to positions (the scene uses index ids). Returns true if an
    /// element was removed. The caller must clear/relocate its own `selected`.
    #[wasm_bindgen(js_name = deleteElement)]
    pub fn delete_element(&mut self, id: u32) -> bool {
        let idx = id as usize;
        if idx >= self.scene.elements.len() {
            return false;
        }
        self.scene.elements.remove(idx);
        // ids are positions: renumber so every element.id == its index again.
        for (i, e) in self.scene.elements.iter_mut().enumerate() {
            e.id = ElementId(i as u32);
        }
        // Shift the side-tables keyed by id (drop `id`, decrement anything above).
        let remap = |k: u32| {
            if k > id {
                Some(k - 1)
            } else if k < id {
                Some(k)
            } else {
                None
            }
        };
        let fitted = std::mem::take(&mut self.fitted);
        self.fitted = fitted
            .into_iter()
            .filter_map(|(k, v)| remap(k).map(|nk| (nk, v)))
            .collect();
        let pins = std::mem::take(&mut self.param_pins);
        self.param_pins = pins
            .into_iter()
            .filter_map(|(k, code)| remap(k).map(|nk| (nk, code)))
            .collect();
        true
    }

    /// Serialize an element to a JSON string for the clipboard. Robust to later
    /// deletions/re-indexing (unlike an id), so copy → delete → paste is safe.
    /// Returns `"null"` if there is no such element.
    #[wasm_bindgen(js_name = cloneElementJson)]
    pub fn clone_element_json(&self, id: u32) -> String {
        match self.scene.elements.get(id as usize) {
            Some(e) => serde_json::to_string(e).unwrap_or_else(|_| "null".into()),
            None => "null".into(),
        }
    }

    /// Paste a previously [`clone_element_json`]'d element, offset by `(dx, dy)`
    /// world units. The paste is unlocked and editable. Returns the new element
    /// id, or `-1` if the JSON did not parse. (A pasted freehand element is a
    /// plain body — its fit ladder is not carried, so it has no complexity
    /// slider; its geometry is preserved exactly.)
    #[wasm_bindgen(js_name = pasteElement)]
    pub fn paste_element(&mut self, json: &str, dx: f64, dy: f64) -> i32 {
        match serde_json::from_str::<Element>(json) {
            Ok(mut e) => {
                e.transform.pos.x += dx;
                e.transform.pos.y += dy;
                e.locked = false;
                // Keep the copied element's own layer if it still exists,
                // otherwise fall back to the active layer (§5.6).
                if e.layer.index() >= self.scene.layers.len() {
                    e.layer = LayerId(self.active_layer);
                }
                self.scene.add_element(e).0 as i32
            }
            Err(_) => -1,
        }
    }

    // === M3: overlay data for the shell ====================================

    /// Per-element metadata for drawing handles / ghosting locked elements:
    /// `[{"id","x","y","rot_deg","locked","kind"}]`. `kind` is
    /// `singlet|fitted|mirror|other`.
    #[wasm_bindgen(js_name = elementsOverlayJson)]
    pub fn elements_overlay_json(&self) -> String {
        let items: Vec<String> = self
            .scene
            .elements
            .iter()
            .enumerate()
            .map(|(i, e)| {
                format!(
                    "{{\"id\":{},\"x\":{},\"y\":{},\"rot_deg\":{},\"locked\":{},\"kind\":\"{}\",\"layer\":{}}}",
                    i,
                    e.transform.pos.x,
                    e.transform.pos.y,
                    e.transform.rot.to_degrees(),
                    e.locked,
                    self.element_kind(i as u32, e),
                    e.layer.0
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    /// The element's boundary as a closed world-space polyline `[x,y,…]` — for
    /// drawing the actual optic (so a freehand-drawn surface is visible).
    #[wasm_bindgen(js_name = elementOutline)]
    pub fn element_outline(&self, id: u32) -> Vec<f32> {
        let Some(e) = self.scene.elements.get(id as usize) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for edge in &e.boundary {
            for p in sample_edge(edge, 24) {
                let w = e.transform.transform_point(p);
                out.push(w.x as f32);
                out.push(w.y as f32);
            }
        }
        // close the loop
        if out.len() >= 2 {
            out.push(out[0]);
            out.push(out[1]);
        }
        out
    }
}

// === M4: layers · undo tree · wavefront scrub · variation gallery ==========
// "It's a workflow." Scene state still lives in Rust (§2); the shell drives
// these through the opaque handle and mirrors nothing.

#[wasm_bindgen]
impl Canvas {
    // --- layers (§5.6) -----------------------------------------------------

    /// All layers: `[{id,name,visible,opacity,count}]`, `count` = elements on
    /// the layer.
    #[wasm_bindgen(js_name = layersJson)]
    pub fn layers_json(&self) -> String {
        let items: Vec<String> = self
            .scene
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let count = self
                    .scene
                    .elements
                    .iter()
                    .filter(|e| e.layer.index() == i)
                    .count();
                format!(
                    "{{\"id\":{},\"name\":{},\"visible\":{},\"opacity\":{},\"count\":{}}}",
                    i,
                    json_str(&l.name),
                    l.visible,
                    l.opacity,
                    count
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    /// Add a new (visible) layer; returns its id.
    #[wasm_bindgen(js_name = addLayer)]
    pub fn add_layer(&mut self, name: &str) -> u32 {
        let id = self.scene.layers.len() as u32;
        self.scene.layers.push(Layer {
            name: if name.is_empty() {
                format!("Layer {id}")
            } else {
                name.to_string()
            },
            visible: true,
            opacity: 1.0,
        });
        id
    }

    #[wasm_bindgen(js_name = renameLayer)]
    pub fn rename_layer(&mut self, id: u32, name: &str) -> bool {
        match self.scene.layers.get_mut(id as usize) {
            Some(l) => {
                l.name = name.to_string();
                true
            }
            None => false,
        }
    }

    #[wasm_bindgen(js_name = setLayerVisible)]
    pub fn set_layer_visible(&mut self, id: u32, visible: bool) -> bool {
        match self.scene.layers.get_mut(id as usize) {
            Some(l) => {
                l.visible = visible;
                true
            }
            None => false,
        }
    }

    #[wasm_bindgen(js_name = setLayerOpacity)]
    pub fn set_layer_opacity(&mut self, id: u32, opacity: f64) -> bool {
        match self.scene.layers.get_mut(id as usize) {
            Some(l) => {
                l.opacity = opacity.clamp(0.0, 1.0);
                true
            }
            None => false,
        }
    }

    /// Remove a layer, merging its elements onto layer 0 and re-indexing the
    /// `layer` field of every element above it. The last layer cannot be
    /// removed. Returns true on success.
    #[wasm_bindgen(js_name = removeLayer)]
    pub fn remove_layer(&mut self, id: u32) -> bool {
        let n = self.scene.layers.len();
        let idx = id as usize;
        if idx >= n || n <= 1 {
            return false;
        }
        self.scene.layers.remove(idx);
        for e in &mut self.scene.elements {
            let li = e.layer.index();
            if li == idx {
                e.layer = LayerId(0);
            } else if li > idx {
                e.layer = LayerId((li - 1) as u32);
            }
        }
        let a = self.active_layer as usize;
        if a == idx {
            self.active_layer = 0;
        } else if a > idx {
            self.active_layer -= 1;
        }
        true
    }

    #[wasm_bindgen(js_name = setActiveLayer)]
    pub fn set_active_layer(&mut self, id: u32) {
        if (id as usize) < self.scene.layers.len() {
            self.active_layer = id;
        }
    }

    #[wasm_bindgen(js_name = activeLayer)]
    pub fn active_layer(&self) -> u32 {
        self.active_layer
    }

    /// Assign an element to a layer (§5.6). Returns true on success.
    #[wasm_bindgen(js_name = setElementLayer)]
    pub fn set_element_layer(&mut self, elem_id: u32, layer_id: u32) -> bool {
        if (layer_id as usize) >= self.scene.layers.len() {
            return false;
        }
        match self.scene.elements.get_mut(elem_id as usize) {
            Some(e) => {
                e.layer = LayerId(layer_id);
                true
            }
            None => false,
        }
    }

    // --- undo tree (§5.2) --------------------------------------------------

    /// Reset the history to a single root node snapshotting the current state.
    /// The shell calls this once after building its demo scene, so undo does
    /// not walk back into the empty start document.
    #[wasm_bindgen(js_name = historyReset)]
    pub fn history_reset(&mut self, label: &str) {
        self.history = History::single(
            self.scene.to_json(),
            self.pins_sorted(),
            self.active_layer,
            label,
        );
    }

    /// Snapshot the current state as a new child of the current node and move
    /// there (§5.2). Returns the new node id, or -1 if nothing changed.
    #[wasm_bindgen]
    pub fn commit(&mut self, label: &str) -> i32 {
        let json = self.scene.to_json();
        let pins = self.pins_sorted();
        {
            let cur = &self.history.nodes[self.history.current];
            if cur.scene_json == json && cur.pins == pins && cur.active_layer == self.active_layer {
                return -1;
            }
        }
        let seq = self.history.next_seq;
        self.history.next_seq += 1;
        let parent = self.history.current;
        let new_idx = self.history.nodes.len();
        self.history.nodes.push(HistoryNode {
            parent: Some(parent),
            children: Vec::new(),
            label: label.to_string(),
            scene_json: json,
            pins,
            active_layer: self.active_layer,
            seq,
        });
        self.history.nodes[parent].children.push(new_idx);
        self.history.current = new_idx;
        new_idx as i32
    }

    #[wasm_bindgen(js_name = canUndo)]
    pub fn can_undo(&self) -> bool {
        self.history.nodes[self.history.current].parent.is_some()
    }

    #[wasm_bindgen(js_name = canRedo)]
    pub fn can_redo(&self) -> bool {
        !self.history.nodes[self.history.current].children.is_empty()
    }

    /// Move to the parent node, restoring its scene (§5.2). True if it moved.
    #[wasm_bindgen]
    pub fn undo(&mut self) -> bool {
        if let Some(p) = self.history.nodes[self.history.current].parent {
            self.restore_node(p);
            true
        } else {
            false
        }
    }

    /// Move to the most recently created child, restoring it. True if it moved.
    #[wasm_bindgen]
    pub fn redo(&mut self) -> bool {
        if let Some(&c) = self.history.nodes[self.history.current].children.last() {
            self.restore_node(c);
            true
        } else {
            false
        }
    }

    /// Jump to an arbitrary node — a branch click in the sidebar or a variant
    /// in the gallery (§5.6). Returns true if the id is valid.
    #[wasm_bindgen(js_name = historyJumpTo)]
    pub fn history_jump_to(&mut self, id: u32) -> bool {
        if (id as usize) < self.history.nodes.len() {
            self.restore_node(id as usize);
            true
        } else {
            false
        }
    }

    #[wasm_bindgen(js_name = currentHistoryNode)]
    pub fn current_history_node(&self) -> u32 {
        self.history.current as u32
    }

    /// The whole undo tree for the sidebar (§5.2) / variation gallery (§5.6):
    /// `{current, nodes:[{id,seq,parent,label,children:[…]}]}`.
    #[wasm_bindgen(js_name = historyJson)]
    pub fn history_json(&self) -> String {
        let nodes: Vec<String> = self
            .history
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let parent = n.parent.map_or(-1i64, |p| p as i64);
                let children: Vec<String> = n.children.iter().map(|c| c.to_string()).collect();
                format!(
                    "{{\"id\":{},\"seq\":{},\"parent\":{},\"label\":{},\"children\":[{}]}}",
                    i,
                    n.seq,
                    parent,
                    json_str(&n.label),
                    children.join(",")
                )
            })
            .collect();
        format!(
            "{{\"current\":{},\"nodes\":[{}]}}",
            self.history.current,
            nodes.join(",")
        )
    }

    // --- wavefront scrub (§5.5) --------------------------------------------

    /// Reference (f64) trace for the wavefront timeline. Flat array, **11 floats
    /// per segment**: `[ax, ay, bx, by, r, g, b, flux, flags, opl0, opl1]`.
    /// `opl0`/`opl1` are the cumulative optical path length (mm) at the segment
    /// ends — the propagation clock. The shell clips every segment at a common
    /// OPL `t` to draw the wavefront at that instant, and reads the largest
    /// `opl1` for the timeline's end. `flags` is 0 (layout parity with
    /// `traceColored`).
    #[wasm_bindgen(js_name = traceWavefront)]
    pub fn trace_wavefront(&self) -> Vec<f32> {
        let vs = self.visible_scene();
        let cfg = TraceConfig {
            ghosts: self.ghosts,
            max_depth: auto_max_depth(&vs, self.max_ghost, self.ghosts),
            max_ghost_bounces: self.max_ghost,
            ..TraceConfig::default()
        };
        let res = ref_trace(&vs, &cfg);
        let mut out = Vec::with_capacity(res.segments.len() * 11);
        for s in &res.segments {
            let [r, g, b] = wavelength_um_to_linear(s.lambda_um as f32);
            out.extend_from_slice(&[
                s.a.x as f32,
                s.a.y as f32,
                s.b.x as f32,
                s.b.y as f32,
                r,
                g,
                b,
                s.flux as f32,
                0.0,
                s.opl0 as f32,
                s.opl1 as f32,
            ]);
        }
        out
    }

    /// Per-launch-ray **primary path** with cumulative OPL, for drawing the
    /// wavefront as a *connected curve* — the surface of constant optical path
    /// (§5.5). Because a wavefront is the locus of equal OPL, joining each ray's
    /// OPL = t point across the fan is literally the wavefront at time t; it is
    /// curved wherever the rays converge or aberrate. Flat f32 encoding:
    /// `[nrays, len₀, (x, y, opl)×len₀, len₁, (x, y, opl)×len₁, …]`, rays in fan
    /// order. Ghosts are off here (the primary path defines the front); the shell
    /// links each ray's OPL = t point to its neighbour's.
    #[wasm_bindgen(js_name = wavefrontCurves)]
    pub fn wavefront_curves(&self) -> Vec<f32> {
        let mut scene = self.visible_scene().into_owned();
        let emitted: Vec<_> = scene.sources.iter().flat_map(|s| s.emit()).collect();
        let cfg = TraceConfig {
            ghosts: false, // the primary path defines the front
            max_depth: auto_max_depth(&scene, 0, false),
            ..TraceConfig::default()
        };
        let mut out: Vec<f32> = vec![emitted.len() as f32];
        for er in &emitted {
            let angle = er.ray.dir.y.atan2(er.ray.dir.x);
            // Trace just this one ray by swapping in a single-ray point source.
            scene.sources = vec![Source {
                kind: SourceKind::Point { half_angle: 0.0 },
                transform: Isometry2::new(er.ray.origin, angle),
                spectrum: Spectrum::Monochromatic(er.lambda_um),
                flux: er.flux,
                rays: 1,
            }];
            let mut segs = ref_trace(&scene, &cfg).segments;
            segs.sort_by(|a, b| {
                a.opl0
                    .partial_cmp(&b.opl0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Polyline: the first segment's start, then every segment's end.
            let n = if segs.is_empty() { 0 } else { segs.len() + 1 };
            out.push(n as f32);
            if let Some(s0) = segs.first() {
                out.extend_from_slice(&[s0.a.x as f32, s0.a.y as f32, s0.opl0 as f32]);
            }
            for s in &segs {
                out.extend_from_slice(&[s.b.x as f32, s.b.y as f32, s.opl1 as f32]);
            }
        }
        out
    }

    // --- variation gallery (§5.6) ------------------------------------------

    /// Interactive (f32) trace of a saved history node, for a live gallery
    /// thumbnail. Same 9-floats-per-segment layout as `traceColored`; empty if
    /// the node id is invalid.
    #[wasm_bindgen(js_name = traceNodeColored)]
    pub fn trace_node_colored(&self, id: u32) -> Vec<f32> {
        match self.node_scene(id) {
            Some(scene) => colored_segments(&scene, self.ghosts, self.max_ghost),
            None => Vec::new(),
        }
    }

    /// Reference (f64) detector readout for a saved node, so variants are
    /// compared on the *same* detector (§5.6). `{hits,flux,centroid,opl}`, or
    /// `null`.
    #[wasm_bindgen(js_name = nodeDetectorJson)]
    pub fn node_detector_json(&self, id: u32, idx: usize) -> String {
        let Some(scene) = self.node_scene(id) else {
            return "null".to_string();
        };
        if idx >= scene.detectors.len() {
            return "null".to_string();
        }
        let cfg = TraceConfig {
            ghosts: self.ghosts,
            max_depth: auto_max_depth(&scene, self.max_ghost, self.ghosts),
            max_ghost_bounces: self.max_ghost,
            ..TraceConfig::default()
        };
        let res = ref_trace(&scene, &cfg);
        let d = &res.detectors[idx];
        let centroid = d.centroid_u().map_or("null".into(), |v| format!("{v}"));
        let opl = d.mean_opl().map_or("null".into(), |v| format!("{v}"));
        format!(
            "{{\"hits\":{},\"flux\":{},\"centroid\":{},\"opl\":{}}}",
            d.hits, d.total_flux, centroid, opl
        )
    }

    /// A one-line summary of a node for the gallery card:
    /// `{label,seq,elements,sources,detectors}`, or `null`.
    #[wasm_bindgen(js_name = nodeSummaryJson)]
    pub fn node_summary_json(&self, id: u32) -> String {
        let Some(node) = self.history.nodes.get(id as usize) else {
            return "null".to_string();
        };
        let Some(scene) = self.node_scene(id) else {
            return "null".to_string();
        };
        format!(
            "{{\"label\":{},\"seq\":{},\"elements\":{},\"sources\":{},\"detectors\":{}}}",
            json_str(&node.label),
            node.seq,
            scene.elements.len(),
            scene.sources.len(),
            scene.detectors.len()
        )
    }
}

// --- 3D scene (Scene3): the canonical model the 2D Canvas is a view of -------
//
// PR 3D-09. These methods are additive: the v1 API above still drives the
// freeform demo. The browser opens a v2 (or migrated v1) `.ocanvas` into
// `scene3`, then reads its dimension, preview mode, and capability report, sets
// the slice plane, promotes bodies to real 3D constructions, and traces the
// authoritative f64 path projected onto the plane. Every scene mutation bumps
// `revision`, so a settled async result carrying a stale one is discarded (§3.3).
#[wasm_bindgen]
impl Canvas {
    /// Open an `.ocanvas` document of either version into the canonical 3D scene.
    /// v1 is migrated with **no third dimension invented** (bodies become
    /// `PlanarOnly`); v2 parses directly (§11.2). Returns a load-report JSON:
    /// `{dimension, migratedFromV1, unresolvedThirdDimension, planarOnlyBodies,
    /// lineProbeDetectors, legacySources, notes, previewMode}`. Throws with a
    /// readable message if the document is not loadable.
    #[wasm_bindgen(js_name = loadScene3JSON)]
    pub fn load_scene3_json(&mut self, json: &str) -> Result<String, JsError> {
        let (scene, report) = load_any(json).map_err(|e| JsError::new(&format!("{e:?}")))?;
        self.scene3 = Some(scene);
        self.revision += 1;
        Ok(self.load_report_json(&report))
    }

    /// The canonical 3D scene as v2 JSON, or `""` if none is loaded.
    #[wasm_bindgen(js_name = scene3JSON)]
    pub fn scene3_json(&self) -> String {
        self.scene3
            .as_ref()
            .map(|s| s.to_json())
            .unwrap_or_default()
    }

    /// The loaded scene's dimension (`3`), or `0` if no 3D scene is loaded.
    #[wasm_bindgen(js_name = sceneDimension)]
    pub fn scene_dimension(&self) -> u8 {
        self.scene3.as_ref().map_or(0, |s| s.dimension)
    }

    /// How the browser should preview the scene (§10.3): `"exact_planar"`,
    /// `"full_3d"`, `"legacy_planar"`, or `"none"` when nothing is loaded. This
    /// is the badge; it never labels a legacy result full 3D.
    #[wasm_bindgen(js_name = previewMode)]
    pub fn preview_mode(&self) -> String {
        match &self.scene3 {
            None => "none".into(),
            Some(s) => match preview_mode(s) {
                PreviewMode::ExactPlanarReduction => "exact_planar",
                PreviewMode::Full3dProjected => "full_3d",
                PreviewMode::LegacyPlanarCompatibility => "legacy_planar",
            }
            .into(),
        }
    }

    /// The current scene revision. A settled f64 result carries the revision it
    /// was traced at; the UI discards a result whose revision is not the latest.
    #[wasm_bindgen(js_name = sceneRevision)]
    pub fn scene_revision(&self) -> f64 {
        self.revision as f64
    }

    /// The capability report for the loaded scene (§3.5): which physics the f64
    /// reference supports, approximates, or rejects, each with the object. JSON
    /// `{supported,exact,approximated,rejected}`, or `null` if none is loaded.
    #[wasm_bindgen(js_name = capabilityReportJSON)]
    pub fn capability_report_json(&self) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        let res = trace_scene3(scene, &self.cfg3());
        let cap = &res.capability_report;
        let arr = |v: &[String]| format!("[{}]", v.join(","));
        let supported: Vec<String> = cap
            .supported
            .iter()
            .map(|f| json_str(&format!("{f:?}")))
            .collect();
        let exact: Vec<String> = cap
            .exact
            .iter()
            .map(|f| json_str(&format!("{f:?}")))
            .collect();
        let approx: Vec<String> = cap
            .approximated
            .iter()
            .map(|a| json_str(&format!("{a:?}")))
            .collect();
        let rejected: Vec<String> = cap
            .rejected
            .iter()
            .map(|r| json_str(&format!("{r:?}")))
            .collect();
        format!(
            "{{\"supported\":{},\"exact\":{},\"approximated\":{},\"rejected\":{}}}",
            arr(&supported),
            arr(&exact),
            arr(&approx),
            arr(&rejected)
        )
    }

    /// Extract the ordered surface chain the axial ray of `source_id` threads
    /// on its way to `detector_id` — the sequence prescription (`oc-seq3`
    /// JSON) a sequential consumer (an Optiland emitter, a paraxial ladder)
    /// reads instead of predicting the beam path statically. Throws when no 3D
    /// scene is loaded, when the ray never reaches the detector (the error
    /// names every branch's fate), or when it reaches it along more than one
    /// chain (the error names the chains).
    #[wasm_bindgen(js_name = sequenceJSON)]
    pub fn sequence_json(&self, source_id: f64, detector_id: f64) -> Result<String, JsError> {
        let Some(scene) = &self.scene3 else {
            return Err(JsError::new("no 3D scene loaded"));
        };
        oc_trace::sequence3::extract_sequence3(
            scene,
            oc_core::id::SourceId(source_id as u64),
            oc_core::id::DetectorId(detector_id as u64),
            self.revision,
        )
        .map(|p| p.to_json())
        .map_err(|e| JsError::new(&e.to_string()))
    }

    /// [`Canvas::sequence_json`] for a declared
    /// [`PrimaryPathIntent`](oc_core::scene3::PrimaryPathIntent): resolves the
    /// intent's source/detector pair and stamps the intent id into the
    /// prescription.
    #[wasm_bindgen(js_name = sequenceForIntentJSON)]
    pub fn sequence_for_intent_json(&self, intent_id: f64) -> Result<String, JsError> {
        let Some(scene) = &self.scene3 else {
            return Err(JsError::new("no 3D scene loaded"));
        };
        oc_trace::sequence3::extract_for_intent(
            scene,
            oc_core::id::PathIntentId(intent_id as u64),
            self.revision,
        )
        .map(|p| p.to_json())
        .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Set the plane the 2D Canvas slices/projects (§10.1). `u` and `v` are the
    /// in-plane axes; they must be perpendicular. Throws on a degenerate or
    /// sheared basis. The view is derived, so this does not bump the revision.
    #[wasm_bindgen(js_name = setViewPlane)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_view_plane(
        &mut self,
        ox: f64,
        oy: f64,
        oz: f64,
        ux: f64,
        uy: f64,
        uz: f64,
        vx: f64,
        vy: f64,
        vz: f64,
    ) -> Result<(), JsError> {
        let plane = Plane3::new(
            Vec3::new(ox, oy, oz),
            Vec3::new(ux, uy, uz),
            Vec3::new(vx, vy, vz),
        )
        .map_err(|e| JsError::new(&format!("{e:?}")))?;
        self.view_plane = plane;
        Ok(())
    }

    /// Slice half-thickness in mm for filtering projected paths by distance from
    /// the plane (§10.4). `0` shows everything projected.
    #[wasm_bindgen(js_name = setSliceThickness)]
    pub fn set_slice_thickness(&mut self, mm: f64) {
        self.slice_thickness = mm.max(0.0);
    }

    /// Trace the authoritative f64 path and project every segment onto the view
    /// plane. Eleven floats per segment:
    /// `[au, av, bu, bv, r, g, b, flux, flags, dist_a, dist_b]`, where
    /// `(au,av)`/`(bu,bv)` are in-plane coordinates, `(r,g,b)` is the segment's
    /// linear-light wavelength colour (the same convention as [`Canvas::trace_colored`],
    /// so the browser draws 3D rays through the identical HDR `RayLayer`),
    /// `flags` is bit1=TIR bit2=ghost, and `dist_*` is the signed distance of each
    /// endpoint from the plane so the UI can slab-filter (§10.4). Empty if no scene.
    ///
    /// The same trace also fills the detectors' flux maps, read with
    /// [`Canvas::last_detector_maps_f32`], so the 3D view can paint them without
    /// tracing again.
    #[wasm_bindgen(js_name = traceProjected3D)]
    pub fn trace_projected_3d(&self) -> Vec<f32> {
        let Some(scene) = &self.scene3 else {
            self.detector_maps.borrow_mut().clear();
            return Vec::new();
        };
        let res = trace_scene3(scene, &self.cfg3());
        *self.detector_maps.borrow_mut() = pack_detector_maps(scene, &res);
        let plane = &self.view_plane;
        let mut out = Vec::with_capacity(res.segments.len() * 11);
        for s in &res.segments {
            let a = plane.to_local(s.a);
            let b = plane.to_local(s.b);
            let flags = (s.flags.tir as u32) * 2 + (s.flags.ghost as u32) * 4;
            let [r, g, b_] = wavelength_um_to_linear(s.lambda_um as f32);
            out.extend_from_slice(&[
                a.x as f32,
                a.y as f32,
                b.x as f32,
                b.y as f32,
                r,
                g,
                b_,
                s.flux_end as f32,
                flags as f32,
                plane.signed_distance(s.a) as f32,
                plane.signed_distance(s.b) as f32,
            ]);
        }
        out
    }

    /// The incident-flux maps of the area detectors, from the most recent
    /// [`Canvas::trace_projected_3d`] call (the same f64 trace and the same cells
    /// as [`Canvas::detector_result_json`]'s `incident`). Flat f32 layout
    /// `[count, (id, nu, nv, halfW, halfH, cell × nu·nv) × count]`, cells
    /// row-major from local `(-halfW, -halfH)`. A picture of the numbers for
    /// painting a detector face; read the numbers themselves from the f64
    /// readout. Empty before the first trace or with no 3D scene.
    #[wasm_bindgen(js_name = lastDetectorMapsF32)]
    pub fn last_detector_maps_f32(&self) -> Vec<f32> {
        self.detector_maps.borrow().clone()
    }

    /// The **interactive** f32 projection (§4.1): the same 11-float layout as
    /// [`Canvas::trace_projected_3d`], traced on the fast CPU f32 path (`cpu3`) so
    /// it can update every pointer-move during a drag. The picture, not the
    /// numbers — settle with the f64 method for anything displayed.
    /// `GpuPreview3::traceProjected3D` draws the same picture on WebGPU.
    #[wasm_bindgen(js_name = traceProjected3DFast)]
    pub fn trace_projected_3d_fast(&self) -> Vec<f32> {
        let Some(job) = self.fast_scene() else {
            return Vec::new();
        };
        let per_ray = oc_gpu::cpu3::trace_segments(&job, FAST_DEPTH);
        pack_projected(&per_ray, &self.view_plane)
    }

    /// Trace the authoritative f64 path and return every segment in **world
    /// frame** (CV-B, the embed surface). Eleven floats per segment:
    /// `[ax, ay, az, bx, by, bz, r, g, b, flux, flags]` — endpoints in world mm,
    /// `(r,g,b)` the segment's linear-light wavelength colour, `flags` bit1=TIR
    /// bit2=ghost (the [`Canvas::trace_projected_3d`] convention). The trace runs
    /// in f64; this buffer is its downcast picture for Three.js/Konva line
    /// geometry. It is forbidden as the source of any displayed number — flux
    /// totals, centroids, and spot statistics come from the f64 readout APIs
    /// ([`Canvas::first_detector_result_json`] and companions). Empty if no scene.
    #[wasm_bindgen(js_name = traceWorld3D)]
    pub fn trace_world_3d(&self) -> Vec<f32> {
        let Some(scene) = &self.scene3 else {
            return Vec::new();
        };
        let res = trace_scene3(scene, &self.cfg3());
        let mut out = Vec::with_capacity(res.segments.len() * 11);
        for s in &res.segments {
            let flags = (s.flags.tir as u32) * 2 + (s.flags.ghost as u32) * 4;
            let [r, g, b] = wavelength_um_to_linear(s.lambda_um as f32);
            out.extend_from_slice(&[
                s.a.x as f32,
                s.a.y as f32,
                s.a.z as f32,
                s.b.x as f32,
                s.b.y as f32,
                s.b.z as f32,
                r,
                g,
                b,
                s.flux_end as f32,
                flags as f32,
            ]);
        }
        out
    }

    /// The **interactive** world-frame trace (CV-B): the same 11-float layout as
    /// [`Canvas::trace_world_3d`], traced on the fast CPU f32 path (`cpu3`) so it
    /// can drive drag-time preview frames. The picture, not the numbers — the
    /// settled frame and every displayed number come from the f64 method (§4.1).
    /// `GpuPreview3::traceWorld3D` draws the same picture on WebGPU.
    #[wasm_bindgen(js_name = traceWorld3DFast)]
    pub fn trace_world_3d_fast(&self) -> Vec<f32> {
        let Some(job) = self.fast_scene() else {
            return Vec::new();
        };
        pack_world(&oc_gpu::cpu3::trace_segments(&job, FAST_DEPTH))
    }

    /// CV-C: rigidly move a batch of loaded-scene objects by one delta —
    /// the wasm face of [`Scene3::transform_objects`]. `ids` are raw object
    /// ids (bodies, standalone surfaces, apertures, detectors, sources, in
    /// any mix; a dragged module passes all of its manifest object ids in one
    /// call so relative geometry is preserved exactly). `delta` is 7 floats
    /// `[px, py, pz, qx, qy, qz, qw]` — world-frame millimetres plus a unit
    /// quaternion. Bumps the revision; the next `traceWorld3DFast` call
    /// retraces the moved scene (the fast path rebuilds from live scene
    /// state). Returns `false` and mutates nothing on an unknown id, a
    /// malformed delta, or no loaded scene — the caller's tier-1 reload is
    /// the recovery.
    #[wasm_bindgen(js_name = transformObjects3)]
    pub fn transform_objects_3(&mut self, ids: &[f64], delta: &[f64]) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if delta.len() != 7 {
            return false;
        }
        let Ok(iso) = Isometry3::new(
            Vec3::new(delta[0], delta[1], delta[2]),
            [delta[3], delta[4], delta[5], delta[6]],
        ) else {
            return false;
        };
        let raw: Vec<u64> = ids.iter().map(|&v| v as u64).collect();
        if scene.transform_objects(&raw, iso).is_err() {
            return false;
        }
        self.revision += 1;
        true
    }

    /// Reconfigure the scene's first source (the light-source picker, §6.6).
    /// `kind` ∈ `collimated_rect` (p0 = ⌀Y, p1 = ⌀X mm) / `collimated_disc`
    /// (p0 = ⌀ mm) / `point` (p0 = cone half-angle °) / `lambertian` (p0 = ⌀ mm);
    /// `samples` is the ray count. Bumps the revision. `false` if no scene/source
    /// or an unknown kind.
    #[wasm_bindgen(js_name = setSourceConfig)]
    pub fn set_source_config(&mut self, kind: &str, p0: f64, p1: f64, samples: u32) -> bool {
        use oc_core::scene3::{AngularDistribution3 as A, SpatialDistribution3 as S};
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let n = samples.max(1);
        let ok = match kind {
            "collimated_rect" => scene.set_first_source(
                S::Rectangle {
                    half_width_mm: (p0 * 0.5).max(0.01),
                    half_height_mm: (p1 * 0.5).max(0.01),
                },
                A::Direction { local_dir: Vec3::X },
                n,
                1,
            ),
            "collimated_disc" => scene.set_first_source(
                S::Disc {
                    radius_mm: (p0 * 0.5).max(0.01),
                },
                A::Direction { local_dir: Vec3::X },
                n,
                1,
            ),
            "point" => scene.set_first_source(
                S::Point,
                A::UniformConeSolidAngle {
                    half_angle_rad: p0.to_radians().clamp(1e-4, 1.55),
                },
                1,
                n,
            ),
            "lambertian" => {
                // An extended diffuse disc: split the budget between area and angle.
                let m = (n as f64).sqrt().ceil() as u32;
                scene.set_first_source(
                    S::Disc {
                        radius_mm: (p0 * 0.5).max(0.01),
                    },
                    A::LambertianHemisphere,
                    m,
                    m,
                )
            }
            _ => false,
        };
        if ok {
            self.revision += 1;
        }
        ok
    }

    /// The first source's kind + params, for the picker UI to initialise:
    /// `{kind, p0, p1, samples}`, or `null` if there is no source.
    #[wasm_bindgen(js_name = firstSourceJSON)]
    pub fn first_source_json(&self) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        match scene.first_source_spec() {
            Some((kind, p0, p1, n)) => format!(
                "{{\"kind\":{},\"p0\":{},\"p1\":{},\"samples\":{}}}",
                json_str(kind),
                p0,
                p1,
                n
            ),
            None => "null".into(),
        }
    }

    /// Set the first source's wavelengths. `nm` are the wavelengths in nanometres
    /// and `weights` their relative weights (equal weights if `weights` is empty
    /// or a different length). At least one positive wavelength is required.
    /// Bumps the revision. `false` if there is no source or no usable line.
    #[wasm_bindgen(js_name = setSourceWavelengths)]
    pub fn set_source_wavelengths(&mut self, nm: &[f64], weights: &[f64]) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let use_weights = weights.len() == nm.len();
        let lines: Vec<(f64, f64)> = nm
            .iter()
            .enumerate()
            .map(|(i, w)| (w / 1000.0, if use_weights { weights[i] } else { 1.0 }))
            .collect();
        if scene.set_first_source_spectrum(&lines) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// The first source's wavelengths in nanometres (weights folded in as repeated
    /// samples are not — this returns the distinct lines), as a JSON array
    /// `[{nm, weight}]`, or `null` if there is no source.
    #[wasm_bindgen(js_name = firstSourceWavelengthsJSON)]
    pub fn first_source_wavelengths_json(&self) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        match scene.first_source_spectrum() {
            Some(lines) => {
                let items: Vec<String> = lines
                    .iter()
                    .map(|(um, wt)| format!("{{\"nm\":{},\"weight\":{}}}", um * 1000.0, wt))
                    .collect();
                format!("[{}]", items.join(","))
            }
            None => "null".into(),
        }
    }

    /// The display colour of a wavelength (nm) as a CSS `rgb(r,g,b)` string — the
    /// same wavelength→colour map the ray renderer uses, so a UI swatch matches
    /// the rays. Outside the visible band this is yellow.
    #[wasm_bindgen(js_name = wavelengthCssColor)]
    pub fn wavelength_css_color(&self, nm: f64) -> String {
        let [r, g, b] = oc_gpu::color::wavelength_um_to_srgb((nm / 1000.0) as f32);
        format!(
            "rgb({},{},{})",
            (r * 255.0).round() as u32,
            (g * 255.0).round() as u32,
            (b * 255.0).round() as u32
        )
    }

    /// Add a planar mirror spanning the meridional segment `(ax,ay)`→`(bx,by)`
    /// (view-plane mm) — the "draw a fold mirror" tool. Returns the new object id,
    /// or `-1` for a degenerate segment / no scene. Bumps the revision.
    #[wasm_bindgen(js_name = addPlanarMirror)]
    pub fn add_planar_mirror(&mut self, ax: f64, ay: f64, bx: f64, by: f64, reflectivity: f64) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_planar_mirror(
            Vec3::new(ax, ay, 0.0),
            Vec3::new(bx, by, 0.0),
            reflectivity,
        ) {
            Some(id) => {
                self.revision += 1;
                id as f64
            }
            None => -1.0,
        }
    }

    /// Add a round optical element (a flat plate) at `(x,y)`, radius `radius` mm,
    /// tilted `tilt_deg` in the meridional plane. Defaults to a front-surface
    /// mirror (front reflective, back clear); reconfigure with `setSurfaceResponse`.
    /// Returns the new object id. Bumps the revision.
    #[wasm_bindgen(js_name = addPlanarElement)]
    pub fn add_planar_element(&mut self, x: f64, y: f64, radius: f64, tilt_deg: f64) -> f64 {
        use oc_core::scene3::{SpectralScalar, SurfaceResponse};
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let front = SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(0.95),
        };
        let back = SurfaceResponse::ExplicitRt {
            reflectance: SpectralScalar::Constant(0.0),
            transmittance: SpectralScalar::Constant(1.0),
        };
        let id = scene.add_planar_element(Vec3::new(x, y, 0.0), radius.max(0.5), tilt_deg, front, back);
        self.revision += 1;
        id as f64
    }

    /// Add an ideal linear polarizer at `(x,y)`: a round plate of `radius` mm,
    /// tilted `tilt_deg`, passing the field along `axis_deg` from the plate's
    /// own local `+X` and absorbing the perpendicular component.
    ///
    /// `extinction` is the **power** fraction of the blocked component that
    /// still gets through: 0 is perfect, 1e-4 an ordinary sheet. Unpolarized
    /// light loses half its flux here, which is physics rather than a modelled
    /// loss. Returns the new object id, or `-1`.
    ///
    /// The preview cannot do this: it carries no polarization state, so it
    /// draws the plate as a 50% attenuator and a crossed pair stays bright
    /// there while the reference goes dark.
    #[wasm_bindgen(js_name = addPolarizer)]
    pub fn add_polarizer(
        &mut self,
        x: f64,
        y: f64,
        radius: f64,
        tilt_deg: f64,
        axis_deg: f64,
        extinction: f64,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_polarizer(
            Vec3::new(x, y, 0.0),
            radius.max(0.5),
            tilt_deg,
            axis_deg,
            extinction,
        ) {
            Some(id) => {
                self.revision += 1;
                id as f64
            }
            None => -1.0,
        }
    }

    /// Add an ideal linear retarder (a waveplate) at `(x,y)`: a round plate of
    /// `radius` mm, tilted `tilt_deg`, with its fast axis `axis_deg` from the
    /// plate's local `+X` and `retardance_deg` of phase on the slow axis.
    ///
    /// 90 is a quarter wave, 180 a half wave. Lossless and zero thickness: the
    /// polarization action of a waveplate and not the crystal, so no walk-off,
    /// no ordinary and extraordinary rays, and the retardance is the number
    /// given rather than one derived from a thickness at each wavelength.
    /// Returns the new object id, or `-1`.
    #[wasm_bindgen(js_name = addRetarder)]
    pub fn add_retarder(
        &mut self,
        x: f64,
        y: f64,
        radius: f64,
        tilt_deg: f64,
        axis_deg: f64,
        retardance_deg: f64,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_retarder(
            Vec3::new(x, y, 0.0),
            radius.max(0.5),
            tilt_deg,
            axis_deg,
            retardance_deg,
        ) {
            Some(id) => {
                self.revision += 1;
                id as f64
            }
            None => -1.0,
        }
    }

    /// Add a beam splitter at `(x,y)`: a round plate of `radius` mm, tilted
    /// `tilt_deg` in the meridional plane, reflecting `reflectance` and
    /// transmitting `transmittance` on both faces. Returns the new object id,
    /// or `-1` (nothing added) for fractions outside `[0, 1]` or summing past 1.
    /// Bumps the revision.
    #[wasm_bindgen(js_name = addBeamSplitter)]
    pub fn add_beam_splitter(
        &mut self,
        x: f64,
        y: f64,
        radius: f64,
        tilt_deg: f64,
        reflectance: f64,
        transmittance: f64,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_beam_splitter(
            Vec3::new(x, y, 0.0),
            radius.max(0.5),
            tilt_deg,
            reflectance,
            transmittance,
        ) {
            Some(id) => {
                self.revision += 1;
                id as f64
            }
            None => -1.0,
        }
    }

    /// Add a dichroic filter at `(x,y)`: a round plate of `radius` mm, tilted
    /// `tilt_deg`, whose coating reflects one side of `edge_nm` and transmits
    /// the other (`long_pass` transmits the long side) across a transition
    /// `width_nm` wide, with the default peak fractions (R 0.98, T 0.95).
    /// Returns the new object id, or `-1` for a non-finite spec. Bumps the
    /// revision.
    #[wasm_bindgen(js_name = addDichroic)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_dichroic(
        &mut self,
        x: f64,
        y: f64,
        radius: f64,
        tilt_deg: f64,
        edge_nm: f64,
        width_nm: f64,
        long_pass: bool,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let spec = DichroicSpec {
            edge_nm,
            width_nm,
            long_pass,
            ..DichroicSpec::default()
        };
        match scene.add_dichroic(Vec3::new(x, y, 0.0), radius.max(0.5), tilt_deg, spec) {
            Some(id) => {
                self.revision += 1;
                id as f64
            }
            None => -1.0,
        }
    }

    /// Make both faces of a surface object a beam splitter. Bumps the revision.
    /// `false` (scene unchanged) for invalid fractions or a non-surface id.
    #[wasm_bindgen(js_name = setSurfaceSplit)]
    pub fn set_surface_split(&mut self, id: f64, reflectance: f64, transmittance: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_surface_split(id as u64, reflectance, transmittance) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Give both faces of a surface object a dichroic coating. Out-of-range
    /// values are brought into range (see `DichroicSpec::sanitized`). Bumps the
    /// revision. `false` (scene unchanged) for a non-finite value or a
    /// non-surface id.
    #[wasm_bindgen(js_name = setSurfaceDichroic)]
    pub fn set_surface_dichroic(
        &mut self,
        id: f64,
        edge_nm: f64,
        width_nm: f64,
        long_pass: bool,
        r_max: f64,
        t_max: f64,
    ) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let spec = DichroicSpec {
            edge_nm,
            width_nm,
            long_pass,
            r_max,
            t_max,
        };
        if scene.set_surface_dichroic(id as u64, spec) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// A polarizing element's properties, or `null` for anything else.
    ///
    /// JSON `{kind, axisDeg, extinction, retardanceDeg}` where `kind` is
    /// `"polarizer"` or `"retarder"`. `extinction` is null on a retarder and
    /// `retardanceDeg` is null on a polarizer, because a property the element
    /// does not have should read as absent and not as zero.
    ///
    /// The axis is measured from the surface's **own local +X**, not from any
    /// world axis: it is a direction in the element, and it turns with the
    /// element.
    #[wasm_bindgen(js_name = surfacePolarizationJSON)]
    pub fn surface_polarization_json(&self, id: f64) -> String {
        use oc_core::scene3::SurfaceResponse;
        let Some((front, _back)) = self
            .scene3
            .as_ref()
            .and_then(|s| s.surface_responses(id as u64))
        else {
            return "null".into();
        };
        match front {
            SurfaceResponse::Polarizer {
                axis_deg,
                extinction,
                substrate_index,
            } => format!(
                "{{\"kind\":\"polarizer\",\"axisDeg\":{},\"extinction\":{},\"retardanceDeg\":null,\"substrateIndex\":{},\"crystal\":null}}",
                json_num(*axis_deg),
                json_num(*extinction),
                json_num(*substrate_index)
            ),
            SurfaceResponse::Retarder {
                axis_deg,
                retardance_deg,
                substrate_index,
                crystal,
            } => format!(
                "{{\"kind\":\"retarder\",\"axisDeg\":{},\"extinction\":null,\"retardanceDeg\":{},\"substrateIndex\":{},\"crystal\":{}}}",
                json_num(*axis_deg),
                json_num(*retardance_deg),
                json_num(*substrate_index),
                match crystal {
                    None => "null".to_string(),
                    Some(c) => format!(
                        "{{\"nO\":{},\"nE\":{},\"order\":{},\"designLambdaUm\":{},\"thicknessUm\":{}}}",
                        json_num(c.n_o),
                        json_num(c.n_e),
                        c.order,
                        json_num(c.design_lambda_um),
                        json_num(c.thickness_um(*retardance_deg))
                    ),
                }
            ),
            _ => "null".into(),
        }
    }

    /// Make a retarder out of a real crystal: two indices, how many whole
    /// waves sit under the nominal retardance, and the wavelength that nominal
    /// is quoted at.
    ///
    /// The retardance then follows from the plate rather than being a number:
    /// it moves as `1/λ` and as the square of the angle off axis, and a
    /// multi-order plate moves `order + 1` times as far as a zero-order one.
    /// False for anything that is not a retarder, or for a crystal that could
    /// not be one.
    #[wasm_bindgen(js_name = setRetarderCrystal)]
    pub fn set_retarder_crystal(
        &mut self,
        id: f64,
        n_o: f64,
        n_e: f64,
        order: u32,
        design_lambda_um: f64,
    ) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let c = oc_core::scene3::Waveplate {
            n_o,
            n_e,
            order,
            design_lambda_um,
        };
        if scene.set_retarder_crystal(id as u64, Some(c)) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Make a retarder ideal again: the nominal retardance at every wavelength
    /// and every angle.
    #[wasm_bindgen(js_name = clearRetarderCrystal)]
    pub fn clear_retarder_crystal(&mut self, id: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_retarder_crystal(id as u64, None) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Set the index of the glass a polarizing element is laminated in, or 1
    /// for a bare layer with no faces at all.
    ///
    /// This is the element's own glass, not the medium it stands in. It is
    /// what makes a real polarizer care about the angle it is used at: the two
    /// faces pass p better than s, and by 45 degrees that is a fifth more light
    /// for a polarizer passing p than for one passing s. False below 1.
    #[wasm_bindgen(js_name = setPolarizingSubstrate)]
    pub fn set_polarizing_substrate(&mut self, id: f64, index: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_polarizing_substrate(id as u64, index) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Turn a polarizing element's axis (degrees from the surface's local
    /// `+X`). False, and the scene unchanged, for anything that is not one.
    #[wasm_bindgen(js_name = setPolarizationAxis)]
    pub fn set_polarization_axis(&mut self, id: f64, axis_deg: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_polarization_axis(id as u64, axis_deg) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Set a polarizer's extinction: the power fraction of the blocked
    /// component that still gets through, clamped to `[0, 1]`.
    #[wasm_bindgen(js_name = setPolarizerExtinction)]
    pub fn set_polarizer_extinction(&mut self, id: f64, extinction: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_polarizer_extinction(id as u64, extinction) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Set a retarder's retardance in degrees: 90 a quarter wave, 180 a half
    /// wave.
    #[wasm_bindgen(js_name = setRetarderRetardance)]
    pub fn set_retarder_retardance(&mut self, id: f64, retardance_deg: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_retarder_retardance(id as u64, retardance_deg) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Each face's coating, for the component view: `{front, back}`, each
    /// `{kind, r, t, dichroic}` with `kind` as in [`Canvas::surface_response_json`],
    /// `r`/`t` the constant reflectance and transmittance (a mirror's
    /// reflectivity as `r`; `null` when tabulated or not defined), and
    /// `dichroic` the `{edgeNm, widthNm, longPass, rMax, tMax}` a dichroic
    /// coating was made from (else `null`). `null` if `id` is not a surface.
    #[wasm_bindgen(js_name = surfaceCoatingJSON)]
    pub fn surface_coating_json(&self, id: f64) -> String {
        use oc_core::scene3::{SpectralScalar, SurfaceResponse};
        let Some((front, back)) = self
            .scene3
            .as_ref()
            .and_then(|s| s.surface_responses(id as u64))
        else {
            return "null".into();
        };
        let constant = |v: &SpectralScalar| match v {
            SpectralScalar::Constant(x) => format!("{x}"),
            SpectralScalar::Tabulated { .. } => "null".into(),
        };
        let face = |r: &SurfaceResponse| {
            let (rr, tt) = match r {
                SurfaceResponse::Mirror { reflectivity } => (constant(reflectivity), "0".into()),
                SurfaceResponse::ExplicitRt {
                    reflectance,
                    transmittance,
                } => (constant(reflectance), constant(transmittance)),
                SurfaceResponse::Absorber => ("0".into(), "0".into()),
                SurfaceResponse::ThinLens { .. } => ("0".into(), "1".into()),
                SurfaceResponse::DielectricFresnel { .. } => ("null".into(), "null".into()),
                // A film has no constant R or T: both depend on the angle, and
                // reporting a number here would be the very thing this is meant
                // to stop. `surfaceFilmJSON` carries the stack instead.
                SurfaceResponse::Coating { .. } => ("null".into(), "null".into()),
                // Nor does a polarizing element: what it passes depends on the
                // state arriving, which is not a property of the surface. Half
                // is the answer only for unpolarized light.
                SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. } => {
                    ("null".into(), "null".into())
                }
            };
            let dichroic = DichroicSpec::from_response(r).map_or("null".into(), |d| {
                format!(
                    "{{\"edgeNm\":{},\"widthNm\":{},\"longPass\":{},\"rMax\":{},\"tMax\":{}}}",
                    d.edge_nm, d.width_nm, d.long_pass, d.r_max, d.t_max
                )
            });
            format!(
                "{{\"kind\":{},\"r\":{rr},\"t\":{tt},\"dichroic\":{dichroic}}}",
                json_str(response_kind(r))
            )
        };
        format!("{{\"front\":{},\"back\":{}}}", face(front), face(back))
    }

    /// A surface face's reflectance and transmittance at each wavelength in
    /// `nm`, as `[R, T, R, T, ...]`, **for light arriving at `aoi_deg`**.
    ///
    /// A real coating is a function of two variables and always was; the
    /// kernel just used to know only one of them. A thin-film face runs the
    /// transfer matrix at the angle asked for, so plotting this at the angle
    /// the part is actually used at is plotting what the rays see. The
    /// responses that carry a table in wavelength alone ignore `aoi_deg` and
    /// report the same curve at every angle, which is exactly what they do in
    /// the tracer.
    ///
    /// A mirror reports its reflectivity and no transmission; an uncoated
    /// dielectric face reports `NaN`, as does a table asked for a wavelength
    /// outside its range with no out-of-range value. Empty if `id` is not a
    /// surface.
    #[wasm_bindgen(js_name = surfaceRtF64)]
    pub fn surface_rt_f64(&self, id: f64, back: bool, nm: &[f64], aoi_deg: f64) -> Vec<f64> {
        use oc_core::scene3::SurfaceResponse;
        // A coated plate is a solid now, so its film is a face of a body rather
        // than a surface response. Evaluate it against the glass behind it,
        // which is what the tracer does and what makes the curve mean anything.
        if let Some(scene) = self.scene3.as_ref() {
            if scene.surface_responses(id as u64).is_none() {
                if let Some(coating) = scene.coating_for_object(id as u64) {
                    let cos_i = aoi_deg.to_radians().cos();
                    let mut out = Vec::with_capacity(nm.len() * 2);
                    for &w in nm {
                        let um = w / 1000.0;
                        let (n0, n_far) = scene.coated_face_media(id as u64, um);
                        let v = oc_core::thinfilm::coating_rt_at(coating, n0, n_far, um, cos_i);
                        out.push(v.map_or(f64::NAN, |x| x.r()));
                        out.push(v.map_or(f64::NAN, |x| x.t()));
                    }
                    return out;
                }
            }
        }
        let Some((front, back_r)) = self
            .scene3
            .as_ref()
            .and_then(|s| s.surface_responses(id as u64))
        else {
            return Vec::new();
        };
        let r = if back { back_r } else { front };
        let mut out = Vec::with_capacity(nm.len() * 2);
        for &w in nm {
            let um = w / 1000.0;
            let (rr, tt) = match r {
                SurfaceResponse::Mirror { reflectivity } => (reflectivity.sample(um), Some(0.0)),
                SurfaceResponse::ExplicitRt {
                    reflectance,
                    transmittance,
                } => (reflectance.sample(um), transmittance.sample(um)),
                SurfaceResponse::Absorber => (Some(0.0), Some(0.0)),
                SurfaceResponse::ThinLens { .. } => (Some(0.0), Some(1.0)),
                SurfaceResponse::DielectricFresnel { .. } => (None, None),
                SurfaceResponse::Coating { coating } => {
                    let cos_i = aoi_deg.to_radians().cos();
                    let ambient = self.scene3.as_ref().map_or(1.0, |s| s.env.ambient_index);
                    match oc_core::thinfilm::coating_rt(coating, ambient, um, cos_i) {
                        Some(v) => (Some(v.r()), Some(v.t())),
                        None => (None, None),
                    }
                }
                // What a polarizing element transmits depends on the state
                // arriving, not on the surface, so there is no curve in
                // wavelength to plot. Reporting the unpolarized half would read
                // as a property of the part.
                SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. } => {
                    (None, None)
                }
            };
            out.push(rr.unwrap_or(f64::NAN));
            out.push(tt.unwrap_or(f64::NAN));
        }
        out
    }

    /// The angle, in degrees, at which the scene's light meets this surface.
    ///
    /// The angle worth designing a coating for and worth plotting one at, since
    /// it is the one the part is being used at. `NaN` for a non-surface id or a
    /// scene with no light in it.
    #[wasm_bindgen(js_name = surfaceAoiDeg)]
    pub fn surface_aoi_deg(&self, id: f64) -> f64 {
        self.scene3
            .as_ref()
            .and_then(|s| s.object_axial_aoi_deg(id as u64))
            .unwrap_or(f64::NAN)
    }

    /// The thin-film stack on a surface's front face, or `null` if it carries
    /// none.
    ///
    /// `{aoiDeg, design: {kind, lambdaNm, aoiDeg, periods, nHigh, nLow} | null,
    /// substrateN, layers: [{name, n, k, thicknessNm}], edgeNm, peakR}`.
    ///
    /// `aoiDeg` is the angle the scene's light actually meets this surface at,
    /// and `edgeNm` is where the built stack really crosses 50 % at that angle,
    /// measured rather than assumed. When the design asked for an edge the stack
    /// cannot deliver, those two numbers disagree and the editor can say so.
    #[wasm_bindgen(js_name = surfaceFilmJSON)]
    pub fn surface_film_json(&self, id: f64) -> String {
        use oc_core::thinfilm::{crossing_nm, BandSide, CoatingBacking, DesignKind};
        let Some(scene) = self.scene3.as_ref() else {
            return "null".into();
        };
        let Some(coating) = scene.coating_for_object(id as u64) else {
            return "null".into();
        };
        let aoi = scene.object_axial_aoi_deg(id as u64).unwrap_or(0.0);
        let ambient = scene.env.ambient_index;

        let layers: Vec<String> = coating
            .layers
            .iter()
            .map(|l| {
                let n = l.material.n.sample(0.55).unwrap_or(f64::NAN);
                let k = l.material.k.sample(0.55).unwrap_or(0.0);
                format!(
                    "{{\"name\":{},\"n\":{},\"k\":{},\"thicknessNm\":{}}}",
                    serde_json::to_string(&l.material.name).unwrap_or_else(|_| "\"\"".into()),
                    json_num(n),
                    json_num(k),
                    json_num(l.thickness_nm)
                )
            })
            .collect();

        let design = match &coating.design {
            Some(d) => format!(
                "{{\"kind\":\"{}\",\"lambdaNm\":{},\"aoiDeg\":{},\"periods\":{},\"nHigh\":{},\"nLow\":{}}}",
                match d.kind {
                    DesignKind::QuarterWaveStack => "mirror",
                    DesignKind::LongPass => "long-pass",
                    DesignKind::ShortPass => "short-pass",
                    DesignKind::SingleLayerAr => "ar-1",
                    DesignKind::BroadbandAr => "ar-3",
                },
                json_num(d.lambda_nm),
                json_num(d.aoi_deg),
                d.periods,
                json_num(d.n_high),
                json_num(d.n_low)
            ),
            None => "null".into(),
        };
        let plate = scene.plate_geometry(id as u64);
        let substrate = match &coating.backing {
            CoatingBacking::Plate { substrate, .. } => {
                json_num(substrate.n.sample(0.55).unwrap_or(f64::NAN))
            }
            // A coating on a solid is backed by that solid's own glass, which is
            // the scene's to report rather than the coating's.
            CoatingBacking::Medium => {
                if plate.is_some() {
                    json_num(oc_core::build3::PLATE_ND)
                } else {
                    "null".into()
                }
            }
        };
        let long_pass = !matches!(
            coating.design.map(|d| d.kind),
            Some(DesignKind::ShortPass)
        );
        let edge = crossing_nm(
            coating,
            ambient,
            ambient,
            aoi,
            0.5,
            250.0,
            1600.0,
            BandSide::of(long_pass),
        );
        let peak = (250..=1600)
            .step_by(5)
            .filter_map(|nm| {
                oc_core::thinfilm::coating_rt(coating, ambient, nm as f64 / 1000.0, aoi.to_radians().cos())
            })
            .map(|v| v.r())
            .fold(0.0f64, f64::max);

        format!(
            "{{\"aoiDeg\":{},\"design\":{},\"substrateN\":{},\"layers\":[{}],\"edgeNm\":{},\"peakR\":{},\"thicknessMm\":{},\"radiusMm\":{}}}",
            json_num(aoi),
            design,
            substrate,
            layers.join(","),
            edge.map(json_num).unwrap_or_else(|| "null".into()),
            json_num(peak),
            plate.map(|(t, _)| json_num(t)).unwrap_or_else(|| "null".into()),
            plate.map(|(_, r)| json_num(r)).unwrap_or_else(|| "null".into())
        )
    }

    /// Rebuild a surface's coating from a design, as a real quarter-wave stack
    /// solved so its edge lands at `lambda_nm` **at `aoi_deg`**.
    ///
    /// `kind` is `long-pass`, `short-pass`, `mirror`, `ar-1` or `ar-3`. Bumps
    /// the revision. `false` for an unknown kind, a non-finite number, or a
    /// non-surface id.
    #[wasm_bindgen(js_name = setSurfaceCoatingDesign)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_surface_coating_design(
        &mut self,
        id: f64,
        kind: &str,
        lambda_nm: f64,
        aoi_deg: f64,
        periods: u32,
        n_high: f64,
        n_low: f64,
    ) -> bool {
        use oc_core::thinfilm::{CoatingDesign, DesignKind};
        let kind = match kind {
            "long-pass" => DesignKind::LongPass,
            "short-pass" => DesignKind::ShortPass,
            "mirror" => DesignKind::QuarterWaveStack,
            "ar-1" => DesignKind::SingleLayerAr,
            "ar-3" => DesignKind::BroadbandAr,
            _ => return false,
        };
        let design = CoatingDesign {
            kind,
            lambda_nm,
            aoi_deg,
            periods,
            n_high,
            n_low,
        };
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_object_coating_design(id as u64, design) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Set one film layer's physical thickness, in nanometres. Bumps the
    /// revision. `false` for a bad index, a non-finite thickness, or a surface
    /// with no film on it.
    #[wasm_bindgen(js_name = setCoatingLayerThickness)]
    pub fn set_coating_layer_thickness(&mut self, id: f64, index: usize, nm: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_object_layer_thickness(id as u64, index, nm) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Add a dichroic filter as a **real coated glass plate**: a disc of
    /// substrate `thickness_mm` thick with the stack on the face the light
    /// meets first, its edge solved for the angle the tilt implies.
    ///
    /// The plate is a solid, so the tracer crosses it the way light does: the
    /// transmitted beam leaves displaced sideways, carrying the glass in its
    /// optical path, and the far face returns its own reflection. Returns the
    /// new object id, or -1.
    #[wasm_bindgen(js_name = addPlateDichroic)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_plate_dichroic(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        radius_mm: f64,
        thickness_mm: f64,
        tilt_deg: f64,
        edge_nm: f64,
        long_pass: bool,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_plate_dichroic(
            oc_core::vec3::Vec3::new(x, y, z),
            radius_mm,
            thickness_mm,
            tilt_deg,
            edge_nm,
            long_pass,
        ) {
            Some(id) => {
                self.revision += 1;
                id.0 as f64
            }
            None => -1.0,
        }
    }

    /// Add a dichroic filter as a zero-thickness plate carrying a real stack.
    ///
    /// The older shape of the same component, kept for documents that hold one
    /// and for the tests that pin its behaviour. It gets the flux right and the
    /// geometry wrong: no glass to cross, so no walk-off and no far face.
    #[wasm_bindgen(js_name = addFilmDichroic)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_film_dichroic(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        radius_mm: f64,
        tilt_deg: f64,
        edge_nm: f64,
        long_pass: bool,
    ) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.add_film_dichroic(
            oc_core::vec3::Vec3::new(x, y, z),
            radius_mm,
            tilt_deg,
            edge_nm,
            long_pass,
        ) {
            Some(id) => {
                self.revision += 1;
                id as f64
            }
            None => -1.0,
        }
    }

    /// Set a surface object's front/back responses (`mirror`, `absorb`,
    /// `clear`, `split` (50:50), `dichroic` (the default long-pass coating), or
    /// `refract`). Bumps the revision. `false` for an unknown kind or a
    /// non-surface id.
    #[wasm_bindgen(js_name = setSurfaceResponse)]
    pub fn set_surface_response(&mut self, id: f64, front: &str, back: &str) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_surface_response(id as u64, front, back) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// A surface object's `{front, back}` response kinds (`mirror`, `absorb`,
    /// `clear`, `split`, `dichroic`, `coated`, `refract`, `thin-lens`), or
    /// `null` if `id` is not a surface.
    #[wasm_bindgen(js_name = surfaceResponseJSON)]
    pub fn surface_response_json(&self, id: f64) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        match scene.surface_response(id as u64) {
            Some((f, b)) => format!("{{\"front\":{},\"back\":{}}}", json_str(f), json_str(b)),
            None => "null".into(),
        }
    }

    /// Add a circular aperture stop at `(x,y)`, opening radius `opening` mm,
    /// tilted `tilt_deg`. Rays inside the opening pass; the rest are absorbed.
    /// Returns the new object id. Bumps the revision.
    #[wasm_bindgen(js_name = addAperture)]
    pub fn add_aperture(&mut self, x: f64, y: f64, opening: f64, tilt_deg: f64) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let id = scene.add_aperture(Vec3::new(x, y, 0.0), opening.max(0.2), tilt_deg);
        self.revision += 1;
        id as f64
    }

    /// Add a square area detector at `(x,y)`, half-size `half` mm, its face normal
    /// tilted `tilt_deg` in the meridional plane (same convention as surfaces and
    /// apertures). 64×64 bins. Returns the new object id. Bumps the revision.
    #[wasm_bindgen(js_name = addAreaDetector)]
    pub fn add_area_detector_3d(&mut self, x: f64, y: f64, half: f64, tilt_deg: f64) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let phi = tilt_deg.to_radians();
        let normal = Vec3::new(phi.cos(), phi.sin(), 0.0);
        let h = half.max(0.5);
        let id = scene.add_area_detector(Vec3::new(x, y, 0.0), normal, h, h, [64, 64]);
        self.revision += 1;
        id.0 as f64
    }

    /// An aperture's clear opening as `{"inner","outer"}` (min / max radius that
    /// passes, mm), or `null` if `id` is not an aperture. A plain disc opening
    /// reports `inner: 0`.
    #[wasm_bindgen(js_name = apertureOpeningJSON)]
    pub fn aperture_opening_json(&self, id: f64) -> String {
        match self
            .scene3
            .as_ref()
            .and_then(|s| s.aperture_opening(id as u64))
        {
            Some((inner, outer)) => format!("{{\"inner\":{inner},\"outer\":{outer}}}"),
            None => "null".into(),
        }
    }

    /// Set an aperture's clear opening to the ring `inner..outer` mm (min / max
    /// radius that passes). `inner = 0` is a plain hole; a positive inner radius
    /// adds a central obstruction. Bumps the revision. `false` (scene unchanged)
    /// if the radii are invalid (`0 <= inner < outer` required) or `id` is not an
    /// aperture.
    #[wasm_bindgen(js_name = setApertureOpening)]
    pub fn set_aperture_opening(&mut self, id: f64, inner: f64, outer: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_aperture_opening(id as u64, inner, outer) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Add a glass prism at `(x,y)`: an equilateral triangle of side `side` mm,
    /// extruded out of plane, glass `(nd,vd)`, tilted `tilt_deg`. Returns the new
    /// body id. Bumps the revision.
    #[wasm_bindgen(js_name = addPrism)]
    pub fn add_prism(&mut self, x: f64, y: f64, side: f64, tilt_deg: f64, nd: f64, vd: f64) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        let id = scene.add_prism(Vec3::new(x, y, 0.0), side, tilt_deg, nd, vd);
        self.revision += 1;
        id as f64
    }

    /// Set a surface object's meridional tilt (its normal's X–Y angle, degrees).
    /// Bumps the revision. `false` for a non-surface id.
    #[wasm_bindgen(js_name = setSurfaceTiltZDeg)]
    pub fn set_surface_tilt_z_deg(&mut self, id: f64, deg: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_surface_tilt_z_deg(id as u64, deg) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Set the spatial ray count of the scene's first source (the ray-density
    /// control). Bumps the revision. `false` if there is no 3D scene or source.
    #[wasm_bindgen(js_name = setSourceSamples)]
    pub fn set_source_samples(&mut self, samples: u32) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        match scene.sources.first_mut() {
            Some(src) => {
                src.samples.spatial_samples = samples.max(1);
                self.revision += 1;
                true
            }
            None => false,
        }
    }

    /// Move a body so its world position becomes `(x, y, z)` — the drag authoring
    /// action (§10.5: an in-plane move translates the body). Bumps the revision.
    /// Returns `false` if there is no such body.
    #[wasm_bindgen(js_name = setBodyPosition)]
    pub fn set_body_position(&mut self, body_id: f64, x: f64, y: f64, z: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let Some(body) = scene
            .bodies
            .iter_mut()
            .find(|b| b.common.id.0 == body_id as u64)
        else {
            return false;
        };
        body.common.transform.pos = Vec3::new(x, y, z);
        self.revision += 1;
        true
    }

    /// The f64 reference readout for a detector by id (§4.1: the honest numbers).
    /// JSON `{hits, totalIncident, totalSignal, cellArea, centroid, meanOpl,
    /// bins, incident, dop, dolp, azimuthDeg, stokes, maxTiltDeg}`, or `null`
    /// if there is no such detector.
    ///
    /// The polarization fields describe the summed state in the **detector's
    /// own frame**: every ray is rotated onto the projection of the detector's
    /// local `+X` before being added. `maxTiltDeg` is the worst angle between
    /// an arriving ray and the detector normal, and it is the caveat on that
    /// sum: rays from very different directions have genuinely different
    /// transverse planes, and adding their `S1` is then arithmetic rather than
    /// measurement. Total flux never has that problem.
    #[wasm_bindgen(js_name = detectorResultJSON)]
    pub fn detector_result_json(&self, detector_id: f64) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        let res = trace_scene3(scene, &self.cfg3());
        let Some(d) = res
            .detectors
            .iter()
            .find(|d| d.detector.0 == detector_id as u64)
        else {
            return "null".into();
        };
        let centroid = if d.total_incident > 0.0 {
            let c = d.flux_uv / d.total_incident;
            format!("[{},{}]", c.x, c.y)
        } else {
            "null".to_string()
        };
        let mean_opl = if d.total_incident > 0.0 {
            format!("{}", d.flux_opl / d.total_incident)
        } else {
            "null".to_string()
        };
        let cell_area = d
            .cell_area_mm2
            .map_or("null".to_string(), |a| format!("{a}"));
        let incident: Vec<String> = d.incident_flux.iter().map(|v| format!("{v}")).collect();
        let st = d.stokes_local;
        format!(
            "{{\"hits\":{},\"totalIncident\":{},\"totalSignal\":{},\"cellArea\":{},\"centroid\":{},\"meanOpl\":{},\"bins\":[{},{}],\"incident\":[{}],\"dop\":{},\"dolp\":{},\"azimuthDeg\":{},\"stokes\":[{},{},{},{}],\"maxTiltDeg\":{}}}",
            d.hits,
            d.total_incident,
            d.total_signal,
            cell_area,
            centroid,
            mean_opl,
            d.bins[0],
            d.bins[1],
            incident.join(","),
            json_num(st.dop()),
            json_num(st.dolp()),
            json_num(st.azimuth_rad().to_degrees()),
            json_num(st.s0),
            json_num(st.s1),
            json_num(st.s2),
            json_num(st.s3),
            json_num(d.max_tilt_rad.to_degrees())
        )
    }

    /// The f64 readout for the scene's first area detector (the common case: one
    /// detector), same JSON as [`Canvas::detector_result_json`], or `null` if
    /// there is no 3D scene or no detector. Saves the browser from tracking ids
    /// for the default single-detector scene.
    #[wasm_bindgen(js_name = firstDetectorResultJSON)]
    pub fn first_detector_result_json(&self) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        match scene.detectors.first() {
            Some(d) => self.detector_result_json(d.common.id.0 as f64),
            None => "null".into(),
        }
    }

    /// Per-hit records on the scene's first area detector, for a **spot diagram**
    /// (§6.9): the actual ray-detector intersection points, not a binned grid.
    /// One trace with hit-recording on (bounded); flat f32 layout
    /// `[halfW, halfH, n, (u, v, r, g, b, flux) × n]` where `(u,v)` are the
    /// detector-local mm coordinates and `(r,g,b)` the linear-light wavelength
    /// colour (same convention as the ray renderer). Empty if no 3D scene /
    /// detector. Records are capped at `max_records`, so a very dense beam shows a
    /// representative sample rather than every ray.
    #[wasm_bindgen(js_name = firstDetectorHitsF32)]
    pub fn first_detector_hits_f32(&self) -> Vec<f32> {
        let first = self
            .scene3
            .as_ref()
            .and_then(|s| s.detectors.first())
            .map(|d| d.common.id.0 as f64);
        match first {
            Some(id) => self.detector_hits_f32(id),
            None => Vec::new(),
        }
    }

    /// Per-hit spot-diagram records for the area detector with id `detector_id`,
    /// same flat f32 layout as [`Canvas::first_detector_hits_f32`]. Empty if there
    /// is no 3D scene or no such detector. Lets the browser show whichever
    /// detector is selected when a scene has several.
    #[wasm_bindgen(js_name = detectorHitsF32)]
    pub fn detector_hits_f32(&self, detector_id: f64) -> Vec<f32> {
        let Some(scene) = &self.scene3 else {
            return Vec::new();
        };
        let did = detector_id as u64;
        let Some(det) = scene.detectors.iter().find(|d| d.common.id.0 == did) else {
            return Vec::new();
        };
        let (hw, hh) = detector_half_extents(det);
        let cfg = TraceConfig3 {
            record_detector_hits: true,
            max_records: 8192,
            ..self.cfg3()
        };
        let res = trace_scene3(scene, &cfg);
        let Some(d) = res.detectors.iter().find(|d| d.detector.0 == did) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(3 + d.records.len() * 6);
        out.extend_from_slice(&[hw as f32, hh as f32, d.records.len() as f32]);
        for h in &d.records {
            let [r, g, b] = wavelength_um_to_linear(h.lambda_um as f32);
            out.extend_from_slice(&[
                h.uv_mm.x as f32,
                h.uv_mm.y as f32,
                r,
                g,
                b,
                h.incident_flux as f32,
            ]);
        }
        out
    }

    /// The optical elements' cross-section with the view plane (§10.4), for
    /// drawing the 2D layout — a lens as its meridional profile, a sphere as a
    /// circle, a mirror or detector as a line. A JSON array of polylines
    /// `[{kind, closed, pts:[u,v,...]}]` in view-plane coordinates; `[]` if no
    /// scene is loaded.
    #[wasm_bindgen(js_name = sceneCrossSectionJSON)]
    pub fn scene_cross_section_json(&self) -> String {
        match &self.scene3 {
            Some(scene) => xsection::cross_section_json(scene, &self.view_plane),
            None => "[]".into(),
        }
    }

    /// What the GPU 3D view draws: one descriptor per body, surface, aperture,
    /// and detector, with independent geometry, transform, and material
    /// revisions (`render_view`). JSON array; `[]` if no scene is loaded. Call it
    /// when the scene revision moves, not per frame.
    #[wasm_bindgen(js_name = renderObjectsJSON)]
    pub fn render_objects_json(&mut self) -> String {
        match &self.scene3 {
            Some(scene) => render_view::render_objects_json(scene, &mut self.render_tracker),
            None => "[]".into(),
        }
    }

    /// The display mesh of object `id` in its local frame, or `undefined`. A
    /// picture of the analytic geometry, never traced (`oc_core::display_mesh`).
    #[wasm_bindgen(js_name = renderMesh)]
    pub fn render_mesh(&self, id: f64) -> Option<render_view::RenderMesh> {
        render_view::render_mesh(self.scene3.as_ref()?, id as u64)
    }

    /// The 3D objects for the inspector: each body's id, geometry kind, 3D
    /// construction, whether its third dimension is resolved, and its world
    /// position. JSON array `[{id,kind,construction,is3d,pos:[x,y,z]}]`; empty if
    /// no scene is loaded. This is where "construction and missing depth are
    /// visible" (§13).
    #[wasm_bindgen(js_name = sceneObjectsJSON)]
    pub fn scene_objects_json(&self) -> String {
        let Some(scene) = &self.scene3 else {
            return "[]".into();
        };
        let mut items: Vec<String> = scene
            .bodies
            .iter()
            .map(|b| {
                let (kind, construction) = body_kind_construction(b);
                let p = b.common.transform.pos;
                let q = b.common.transform.rot_xyzw;
                // In-plane (about +Z) angle in degrees. App-authored bodies only ever
                // set a pure-Z rotation, so `2·atan2(qz, qw)` recovers it exactly;
                // imported scenes (optikit materializer) carry arbitrary quats, so
                // `rot` ships the full [x,y,z,w] for faithful 3D drawing.
                let rot_z = (2.0 * q[2].atan2(q[3])).to_degrees();
                format!(
                    "{{\"id\":{},\"kind\":{},\"construction\":{},\"is3d\":{},\"body\":true,\"pos\":[{},{},{}],\"radius\":{},\"rotZ\":{},\"rot\":[{},{},{},{}]}}",
                    b.common.id.0,
                    json_str(kind),
                    json_str(construction),
                    b.is_3d(),
                    p.x,
                    p.y,
                    p.z,
                    body_radius(b),
                    rot_z,
                    q[0],
                    q[1],
                    q[2],
                    q[3]
                )
            })
            .collect();
        // Surface objects (drawn mirrors, etc.) are selectable/movable/deletable
        // too, but not bodies: `body:false` so the UI omits the tilt/rotation
        // authoring (their base orientation is a facing, not a pure-Z rotation).
        for s in &scene.surfaces {
            let (kind, radius) = surface_kind_radius(s);
            let p = s.common.transform.pos;
            let q = s.common.transform.rot_xyzw;
            items.push(format!(
                "{{\"id\":{},\"kind\":{},\"construction\":{},\"is3d\":true,\"body\":false,\"pos\":[{},{},{}],\"radius\":{},\"rotZ\":{},\"rot\":[{},{},{},{}]}}",
                s.common.id.0,
                json_str(kind),
                json_str("planar"),
                p.x,
                p.y,
                p.z,
                radius,
                surface_tilt_deg(s),
                q[0],
                q[1],
                q[2],
                q[3]
            ));
        }
        // Apertures: selectable/movable/tiltable/deletable, drawn as an annulus.
        for a in &scene.apertures {
            let (outer, opening) = aperture_radius_opening(a);
            let p = a.common.transform.pos;
            let q = a.common.transform.rot_xyzw;
            items.push(format!(
                "{{\"id\":{},\"kind\":\"aperture\",\"construction\":\"planar\",\"is3d\":true,\"body\":false,\"pos\":[{},{},{}],\"radius\":{},\"opening\":{},\"rotZ\":{},\"rot\":[{},{},{},{}]}}",
                a.common.id.0,
                p.x,
                p.y,
                p.z,
                outer,
                opening,
                aperture_tilt_deg(a),
                q[0],
                q[1],
                q[2],
                q[3]
            ));
        }
        // Detectors: first-class movable/tiltable/deletable objects, like surfaces.
        for d in &scene.detectors {
            let (hw, hh) = detector_half_extents(d);
            let p = d.common.transform.pos;
            let q = d.common.transform.rot_xyzw;
            items.push(format!(
                "{{\"id\":{},\"kind\":\"detector\",\"construction\":\"planar\",\"is3d\":true,\"body\":false,\"pos\":[{},{},{}],\"radius\":{},\"rotZ\":{},\"rot\":[{},{},{},{}]}}",
                d.common.id.0,
                p.x,
                p.y,
                p.z,
                hw.hypot(hh),
                detector_tilt_deg(d),
                q[0],
                q[1],
                q[2],
                q[3]
            ));
        }
        format!("[{}]", items.join(","))
    }

    /// The scene's light sources in the [`Canvas::scene_objects_json`] item shape,
    /// with `kind: "source"` and `construction` naming the emitting aperture
    /// (`rectangle`, `disc`, `point`, `line`). Only the 3D model view lists
    /// them: it draws a source glyph, and the 2D cross-section does not. A source
    /// moves and rotates through the same setters as any object.
    #[wasm_bindgen(js_name = sourceObjectsJSON)]
    pub fn source_objects_json(&self) -> String {
        use oc_core::scene3::SpatialDistribution3 as S;
        let Some(scene) = &self.scene3 else {
            return "[]".into();
        };
        let items: Vec<String> = scene
            .sources
            .iter()
            .map(|s| {
                let construction = match s.spatial {
                    S::Rectangle { .. } => "rectangle",
                    S::Disc { .. } => "disc",
                    S::Point => "point",
                    S::LineLegacy { .. } => "line",
                };
                let p = s.transform.pos;
                let q = s.transform.rot_xyzw;
                let rot_z = (2.0 * q[2].atan2(q[3])).to_degrees();
                format!(
                    "{{\"id\":{},\"kind\":\"source\",\"construction\":{},\"is3d\":{},\"body\":false,\"pos\":[{},{},{}],\"radius\":{},\"rotZ\":{},\"rot\":[{},{},{},{}]}}",
                    s.id.0,
                    json_str(construction),
                    !s.spatial.is_legacy(),
                    p.x,
                    p.y,
                    p.z,
                    source_radius(s),
                    rot_z,
                    q[0],
                    q[1],
                    q[2],
                    q[3]
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    // --- Scene3 authoring + editing (the always-3D app) -------------------

    /// Initialize the canonical 3D scene to the default authoring scene: a
    /// biconvex singlet at the origin, a collimated beam from the left, and an
    /// area detector near the focus. Bumps the revision. Returns the singlet's
    /// body id, so the browser can select it.
    #[wasm_bindgen(js_name = newDefaultScene3)]
    pub fn new_default_scene3(&mut self) -> f64 {
        let scene = Scene3::default_scene();
        let body_id = scene.bodies.first().map_or(0, |b| b.common.id.0);
        self.scene3 = Some(scene);
        self.revision += 1;
        body_id as f64
    }

    /// Move any object or source by id in world space (§10.5). Bumps the
    /// revision. `false` if no object has that id or there is no 3D scene.
    #[wasm_bindgen(js_name = setObjectPosition)]
    pub fn set_object_position(&mut self, id: f64, x: f64, y: f64, z: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_object_position(id as u64, Vec3::new(x, y, z)) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Rotate an object in the meridional plane (right-handed about +Z), degrees.
    /// Bumps the revision. `false` if no object has that id.
    #[wasm_bindgen(js_name = setObjectRotationZDeg)]
    pub fn set_object_rotation_z_deg(&mut self, id: f64, deg: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_object_rotation_z_deg(id as u64, deg) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// A singlet body's geometry as JSON
    /// `{r1,r2,thickness,semiAperture,conic1,conic2}` (an infinite radius is
    /// `null`, i.e. a plane), or `null` if `id` is not a singlet.
    #[wasm_bindgen(js_name = singletGeomJSON)]
    pub fn singlet_geom_json(&self, id: f64) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        let Some(g) = scene.singlet_geom(id as u64) else {
            return "null".into();
        };
        let num = |v: f64| {
            if v.is_finite() {
                format!("{v}")
            } else {
                "null".into()
            }
        };
        format!(
            "{{\"r1\":{},\"r2\":{},\"thickness\":{},\"semiAperture\":{},\"conic1\":{},\"conic2\":{}}}",
            num(g.r1_mm),
            num(g.r2_mm),
            num(g.thickness_mm),
            num(g.semi_aperture_mm),
            num(g.conic1),
            num(g.conic2),
        )
    }

    /// A body's bulk glass as JSON `{nd,vd}` (an Abbe point), or `null` if `id` is
    /// not a body or its material is not an Abbe point (catalog glass, mirror, …).
    #[wasm_bindgen(js_name = bodyGlassJSON)]
    pub fn body_glass_json(&self, id: f64) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        match scene.body_glass_abbe(id as u64) {
            Some((nd, vd)) => format!("{{\"nd\":{nd},\"vd\":{vd}}}"),
            None => "null".into(),
        }
    }

    /// Replace a singlet body's geometry, keeping its aspheric terms. Refuses
    /// (returns `false`, scene unchanged) if `id` is not a singlet or the
    /// geometry is not revolvable — a bad card edit never corrupts the scene.
    /// Bumps the revision on success.
    #[wasm_bindgen(js_name = setSingletGeom)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_singlet_geom(
        &mut self,
        id: f64,
        r1: f64,
        r2: f64,
        thickness: f64,
        semi_aperture: f64,
        conic1: f64,
        conic2: f64,
    ) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let g = SingletGeom {
            r1_mm: r1,
            r2_mm: r2,
            thickness_mm: thickness,
            semi_aperture_mm: semi_aperture,
            conic1,
            conic2,
        };
        if scene.set_singlet_geom(id as u64, g) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// A singlet's whole shape, for the component view, or `null` if `id` is
    /// not a singlet:
    ///
    /// - the [`Canvas::singlet_geom_json`] fields (a plane's radius is `null`);
    /// - `asph1`, `asph2`: each face's aspheric terms A4..A18;
    /// - `nd`, `efl`, `bfl`: the paraxial d-line index, focal length, and back
    ///   focal distance from the rear vertex (`null` for an afocal lens or a
    ///   material without an index);
    /// - `edge`: the thickness at the rim, mm;
    /// - `faces`: for the front then the back face, `{vertex, pts}` with the
    ///   vertex X and `samples + 1` points `[x, y, ...]` along the upper half of
    ///   the face from the axis to the rim, in the lens's local frame (optical
    ///   axis +X, centred on the body origin).
    #[wasm_bindgen(js_name = singletShapeJSON)]
    pub fn singlet_shape_json(&self, id: f64, samples: u32) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        let Some((g, a)) = scene.singlet_shape(id as u64) else {
            return "null".into();
        };
        let num = |v: f64| {
            if v.is_finite() {
                format!("{v}")
            } else {
                "null".into()
            }
        };
        let list = |v: &[f64]| v.iter().map(|x| num(*x)).collect::<Vec<_>>().join(",");
        let n = samples.clamp(2, 400);
        let ha = g.semi_aperture_mm;
        let face = |back: bool| {
            let vertex = if back {
                g.thickness_mm / 2.0
            } else {
                -g.thickness_mm / 2.0
            };
            let mut pts = Vec::with_capacity(2 * n as usize + 2);
            for i in 0..=n {
                let y = ha * i as f64 / n as f64;
                pts.push(vertex + singlet_face_sag(&g, &a, back, y));
                pts.push(y);
            }
            format!("{{\"vertex\":{},\"pts\":[{}]}}", num(vertex), list(&pts))
        };
        let edge = g.thickness_mm + singlet_face_sag(&g, &a, true, ha)
            - singlet_face_sag(&g, &a, false, ha);
        let par = scene.singlet_paraxial(id as u64);
        format!(
            "{{\"r1\":{},\"r2\":{},\"thickness\":{},\"semiAperture\":{},\"conic1\":{},\"conic2\":{},\"asph1\":[{}],\"asph2\":[{}],\"nd\":{},\"efl\":{},\"bfl\":{},\"edge\":{},\"faces\":[{},{}]}}",
            num(g.r1_mm),
            num(g.r2_mm),
            num(g.thickness_mm),
            num(ha),
            num(g.conic1),
            num(g.conic2),
            list(&a.front),
            list(&a.back),
            par.map_or("null".into(), |p| num(p.nd)),
            par.map_or("null".into(), |p| num(p.efl_mm)),
            par.map_or("null".into(), |p| num(p.bfl_mm)),
            num(edge),
            face(false),
            face(true),
        )
    }

    /// The sag of a singlet face (`back` false for the front) at each of
    /// `heights`, measured from that face's vertex along the lens axis, or of
    /// its base conic alone with `conic_only`. `NaN` for a height outside the
    /// face; empty if `id` is not a singlet.
    #[wasm_bindgen(js_name = singletFaceSagF64)]
    pub fn singlet_face_sag_f64(
        &self,
        id: f64,
        back: bool,
        heights: &[f64],
        conic_only: bool,
    ) -> Vec<f64> {
        let Some((g, mut a)) = self
            .scene3
            .as_ref()
            .and_then(|s| s.singlet_shape(id as u64))
        else {
            return Vec::new();
        };
        if conic_only {
            a = SingletAspheres::default();
        }
        heights
            .iter()
            .map(|&h| {
                if h.is_finite() && h.abs() <= g.semi_aperture_mm * (1.0 + 1e-12) {
                    singlet_face_sag(&g, &a, back, h.abs())
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Replace a singlet's geometry and aspheric terms in one edit. `asph1` and
    /// `asph2` hold up to eight terms each (A4 first); missing ones are zero.
    /// Refuses (`false`, scene unchanged) more than eight terms, a non-finite
    /// value, or a shape whose faces cross. Bumps the revision on success.
    #[wasm_bindgen(js_name = setSingletShape)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_singlet_shape(
        &mut self,
        id: f64,
        r1: f64,
        r2: f64,
        thickness: f64,
        semi_aperture: f64,
        conic1: f64,
        conic2: f64,
        asph1: &[f64],
        asph2: &[f64],
    ) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let terms = |v: &[f64]| -> Option<[f64; 8]> {
            if v.len() > 8 {
                return None;
            }
            let mut out = [0.0; 8];
            out[..v.len()].copy_from_slice(v);
            Some(out)
        };
        let (Some(front), Some(back)) = (terms(asph1), terms(asph2)) else {
            return false;
        };
        let g = SingletGeom {
            r1_mm: r1,
            r2_mm: r2,
            thickness_mm: thickness,
            semi_aperture_mm: semi_aperture,
            conic1,
            conic2,
        };
        if scene.set_singlet_shape(id as u64, g, SingletAspheres { front, back }) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Bend one singlet face (`back` false for the front) so its rim sits
    /// `rim_sag` mm from its vertex, solving for the curvature and keeping the
    /// conic constant and aspheric terms (the component view's bend handle).
    /// Refuses (`false`, scene unchanged) a sag past the conic's reach or a
    /// shape whose faces cross. Bumps the revision on success.
    #[wasm_bindgen(js_name = bendSingletFace)]
    pub fn bend_singlet_face(&mut self, id: f64, back: bool, rim_sag: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.bend_singlet_face(id as u64, back, rim_sag) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Reshape one singlet face (`back` false for the front) to pass through
    /// the points `(heights[i], sags[i])`, sags measured from the face's vertex
    /// (the component view's control points): its curvature and conic stay,
    /// and its eight aspheric terms become the least-bending set through the
    /// points. Refuses (`false`, scene unchanged) mismatched lengths and
    /// anything [`oc_core::scene3::Scene3::fit_singlet_face`] refuses. Bumps
    /// the revision on success.
    #[wasm_bindgen(js_name = fitSingletFace)]
    pub fn fit_singlet_face(&mut self, id: f64, back: bool, heights: &[f64], sags: &[f64]) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if heights.len() != sags.len() {
            return false;
        }
        let pts: Vec<(f64, f64)> = heights.iter().copied().zip(sags.iter().copied()).collect();
        if scene.fit_singlet_face(id as u64, back, &pts) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Set a body's bulk glass to an Abbe point `(nd, vd)`. Bumps the revision.
    /// `false` if `id` is not a body.
    #[wasm_bindgen(js_name = setBodyGlassAbbe)]
    pub fn set_body_glass_abbe(&mut self, id: f64, nd: f64, vd: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.set_body_glass_abbe(id as u64, nd, vd) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Delete any object or source by id. Bumps the revision. `false` if nothing
    /// matched.
    #[wasm_bindgen(js_name = deleteObject3D)]
    pub fn delete_object_3d(&mut self, id: f64) -> bool {
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        if scene.delete_object(id as u64) {
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Duplicate a body offset by `(dx,dy,dz)`, with fresh ids. Returns the new
    /// body id, or `-1` if `id` is not a body. Bumps the revision.
    #[wasm_bindgen(js_name = cloneBody3D)]
    pub fn clone_body_3d(&mut self, id: f64, dx: f64, dy: f64, dz: f64) -> f64 {
        let Some(scene) = self.scene3.as_mut() else {
            return -1.0;
        };
        match scene.clone_body(id as u64, Vec3::new(dx, dy, dz)) {
            Some(new_id) => {
                self.revision += 1;
                new_id.0 as f64
            }
            None => -1.0,
        }
    }
}

/// Interaction depth of the interactive f32 trace, on the CPU mirror or the GPU
/// (§4.1). One constant, so the two pictures cannot drift apart.
pub(crate) const FAST_DEPTH: u32 = 32;

/// Pack a trace's area-detector maps into the layout of
/// [`Canvas::last_detector_maps_f32`]. A legacy line probe has no area and is
/// left out.
fn pack_detector_maps(scene: &Scene3, res: &oc_trace::TraceResult3) -> Vec<f32> {
    let mut out = vec![0.0];
    let mut count = 0;
    for d in &res.detectors {
        let Some(det) = scene
            .detectors
            .iter()
            .find(|x| x.common.id.0 == d.detector.0)
        else {
            continue;
        };
        let [nu, nv] = d.bins;
        if !det.geometry.has_area() || d.incident_flux.len() != nu * nv {
            continue;
        }
        let (hw, hh) = detector_half_extents(det);
        out.extend_from_slice(&[
            d.detector.0 as f32,
            nu as f32,
            nv as f32,
            hw as f32,
            hh as f32,
        ]);
        out.extend(d.incident_flux.iter().map(|&v| v as f32));
        count += 1;
    }
    out[0] = count as f32;
    out
}

/// Pack f32 segments into the projected 11-float layout of
/// [`Canvas::trace_projected_3d`].
pub(crate) fn pack_projected(per_ray: &[Vec<oc_gpu::cpu3::SegOut3>], plane: &Plane3) -> Vec<f32> {
    let mut out = Vec::new();
    for segs in per_ray {
        for s in segs {
            let a = plane.to_local(s.a);
            let b = plane.to_local(s.b);
            let [r, g, b_] = wavelength_um_to_linear(s.lambda_um);
            out.extend_from_slice(&[
                a.x as f32,
                a.y as f32,
                b.x as f32,
                b.y as f32,
                r,
                g,
                b_,
                s.flux,
                s.flags as f32,
                plane.signed_distance(s.a) as f32,
                plane.signed_distance(s.b) as f32,
            ]);
        }
    }
    out
}

/// Pack f32 segments into the world-frame 11-float layout of
/// [`Canvas::trace_world_3d`].
pub(crate) fn pack_world(per_ray: &[Vec<oc_gpu::cpu3::SegOut3>]) -> Vec<f32> {
    let mut out = Vec::new();
    for segs in per_ray {
        for s in segs {
            let [r, g, b] = wavelength_um_to_linear(s.lambda_um);
            out.extend_from_slice(&[
                s.a.x as f32,
                s.a.y as f32,
                s.a.z as f32,
                s.b.x as f32,
                s.b.y as f32,
                s.b.z as f32,
                r,
                g,
                b,
                s.flux,
                s.flags as f32,
            ]);
        }
    }
    out
}

/// A finite number for JSON, or `null`. `serde_json` refuses NaN and infinity,
/// and a silent `0` there would be a lie about a measurement that failed.
fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".to_string()
    }
}

impl Canvas {
    /// The compiled f32 scene the interactive preview traces, on the CPU mirror
    /// or the GPU. `None` until a 3D scene is loaded.
    pub(crate) fn fast_scene(&self) -> Option<oc_gpu::scene3::GpuScene3> {
        let scene = self.scene3.as_ref()?;
        Some(oc_gpu::scene3::GpuScene3::build(
            scene,
            oc_gpu::scene3::BuildOpts {
                ghosts: self.ghosts,
                flux_eps: 1e-4,
                max_ghost: self.max_ghost,
            },
        ))
    }

    /// The view plane, by value, for a trace that outlives this borrow.
    pub(crate) fn view_plane_copy(&self) -> Plane3 {
        self.view_plane
    }

    /// The f64 trace configuration, stamped with the current revision (§3.3).
    fn cfg3(&self) -> TraceConfig3 {
        TraceConfig3 {
            ghosts: self.ghosts,
            max_ghost_bounces: self.max_ghost,
            revision: self.revision,
            ..TraceConfig3::default()
        }
    }

    /// Build the load-report JSON from a [`LoadReport`].
    fn load_report_json(&self, report: &LoadReport) -> String {
        let ids = |v: &[u64]| {
            v.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let (migrated, planar, lines, sources, notes, unresolved) = match report {
            LoadReport::V2 => (false, Vec::new(), Vec::new(), Vec::new(), Vec::new(), false),
            LoadReport::MigratedFromV1(r) => (
                true,
                r.planar_only_bodies
                    .iter()
                    .map(|o| o.0)
                    .collect::<Vec<u64>>(),
                r.line_probe_detectors
                    .iter()
                    .map(|o| o.0)
                    .collect::<Vec<u64>>(),
                r.legacy_sources.iter().map(|s| s.0).collect::<Vec<u64>>(),
                r.notes.clone(),
                r.has_unresolved_dimension(),
            ),
        };
        let notes_json: Vec<String> = notes.iter().map(|n| json_str(n)).collect();
        format!(
            "{{\"dimension\":{},\"migratedFromV1\":{},\"unresolvedThirdDimension\":{},\"planarOnlyBodies\":[{}],\"lineProbeDetectors\":[{}],\"legacySources\":[{}],\"notes\":[{}],\"previewMode\":{}}}",
            self.scene_dimension(),
            migrated,
            unresolved,
            ids(&planar),
            ids(&lines),
            ids(&sources),
            notes_json.join(","),
            json_str(&self.preview_mode())
        )
    }
}

impl Canvas {
    /// The scene as it should be *traced*: elements on hidden layers are dropped
    /// so light ignores them (§5.6). When every layer is visible this borrows the
    /// live scene — no clone on the hot path. (Onion-skin ghosts of hidden layers
    /// are drawn by the shell from the overlay data, not traced.)
    fn visible_scene(&self) -> std::borrow::Cow<'_, Scene> {
        let layers = &self.scene.layers;
        let all_visible = layers.iter().all(|l| l.visible);
        if all_visible {
            std::borrow::Cow::Borrowed(&self.scene)
        } else {
            let mut s = self.scene.clone();
            s.elements
                .retain(|e| layers.get(e.layer.index()).map_or(true, |l| l.visible));
            std::borrow::Cow::Owned(s)
        }
    }

    /// Add an element onto the currently-active layer (§5.6).
    fn add_on_active(&mut self, mut e: Element) -> ElementId {
        let max = self.scene.layers.len().saturating_sub(1) as u32;
        e.layer = LayerId(self.active_layer.min(max));
        self.scene.add_element(e)
    }

    /// Pins as a sorted vec, for deterministic history snapshots/comparison.
    fn pins_sorted(&self) -> Vec<(u32, u8)> {
        let mut v: Vec<(u32, u8)> = self.param_pins.iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Parse a history node's snapshotted scene (variation-gallery trace, §5.6).
    fn node_scene(&self, id: u32) -> Option<Scene> {
        self.history
            .nodes
            .get(id as usize)
            .and_then(|n| Scene::from_json(&n.scene_json).ok())
    }

    /// Restore scene + pins + active layer from a history node and make it
    /// current (§5.2). Freehand fit ladders are not serialized, so the
    /// complexity slider resets for freehand elements after an undo.
    fn restore_node(&mut self, idx: usize) {
        let (json, pins, active) = {
            let n = &self.history.nodes[idx];
            (n.scene_json.clone(), n.pins.clone(), n.active_layer)
        };
        if let Ok(scene) = Scene::from_json(&json) {
            self.scene = scene;
            self.param_pins = pins.into_iter().collect();
            self.active_layer = active.min(self.scene.layers.len().saturating_sub(1) as u32);
            self.fitted.clear();
            self.history.current = idx;
        }
    }

    /// Is the element locked?
    fn locked(&self, id: u32) -> bool {
        self.scene
            .elements
            .get(id as usize)
            .map(|e| e.locked)
            .unwrap_or(false)
    }

    /// Locked element, or the specific parameter `code` pinned?
    fn is_pinned(&self, id: u32, code: u8) -> bool {
        self.locked(id) || self.param_pins.contains(&(id, code))
    }

    fn element_kind(&self, id: u32, e: &Element) -> &'static str {
        if self.fitted.contains_key(&id) {
            return "fitted";
        }
        match e.material {
            Material::Mirror { .. } => "mirror",
            _ if e.boundary.len() >= 4
                && matches!(e.boundary[0], Edge::Conic(_))
                && matches!(e.boundary[2], Edge::Conic(_)) =>
            {
                "singlet"
            }
            _ => "other",
        }
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

// --- free helpers ----------------------------------------------------------

/// The interaction depth cap `D`, auto-sized to the scene. A primary transmitted
/// ray crosses each surface once, so its path length is just the scene's surface
/// count — not a magic number. Ghost generations may re-traverse, so budget
/// `(G+1)×` the surfaces; a floor keeps simple folded scenes (a few mirrors)
/// usable and a ceiling bounds the per-ray buffer. This is why the number of
/// elements is unlimited while ghost bounces stay a small explicit knob (§4.2).
fn auto_max_depth(scene: &Scene, max_ghost: u32, ghosts: bool) -> u32 {
    let surfaces: u32 = scene.elements.iter().map(|e| e.boundary.len() as u32).sum();
    let g = if ghosts { max_ghost } else { 0 };
    surfaces
        .saturating_mul(g + 1)
        .saturating_add(16)
        .clamp(32, 1024)
}

/// Trace a scene on the interactive f32 path and pack colored segments — 9
/// floats each `[ax, ay, bx, by, r, g, b, flux, flags]`, in linear light (the
/// caller accumulates additively then tonemaps). Shared by the live view and
/// the variation-gallery thumbnails so both render identically. Depth and the
/// per-ray segment budget are auto-sized to the scene so element count never
/// truncates the rays.
fn colored_segments(scene: &Scene, ghosts: bool, max_ghost: u32) -> Vec<f32> {
    let max_depth = auto_max_depth(scene, max_ghost, ghosts);
    let opts = BuildOpts {
        ghosts,
        // The per-ray segment budget tracks the depth cap, or it becomes a
        // hidden second limit (a path is ~depth segments long).
        max_segs: max_depth + 8,
        flux_eps: 1e-4,
        max_ghost,
    };
    let gs = GpuScene::build_with(scene, opts);
    let per_ray = trace_cpu(&gs, max_depth);
    let mut out = Vec::new();
    for segs in &per_ray {
        for s in segs {
            let [r, g, b] = wavelength_um_to_linear(s.lambda_um);
            out.extend_from_slice(&[s.a.x, s.a.y, s.b.x, s.b.y, r, g, b, s.flux, s.flags as f32]);
        }
    }
    out
}

/// Escape a string as a JSON string literal (quotes included) — for names and
/// history labels that the shell round-trips through `JSON.parse`.
/// A coarse world bounding radius for a body, for pointer hit-testing in the 3D
/// view. Generous rather than tight: it only decides whether a click grabs it.
fn body_radius(b: &oc_core::scene3::Body3) -> f64 {
    use oc_core::profile2::Curve2;
    use oc_core::scene3::SolidGeometry3;
    match &b.geometry {
        SolidGeometry3::Sphere { radius_mm } => *radius_mm,
        SolidGeometry3::Box { half_extents_mm } => half_extents_mm.length(),
        SolidGeometry3::Cylinder {
            radius_mm,
            half_length_mm,
            ..
        } => radius_mm.hypot(*half_length_mm),
        SolidGeometry3::Profile(p) => {
            let r = p
                .profile
                .edges
                .iter()
                .flat_map(|e| match &e.curve {
                    Curve2::Line { a, b } => vec![a.length(), b.length()],
                    Curve2::Polyline { points } => points.iter().map(|q| q.length()).collect(),
                    Curve2::Conic(c) | Curve2::EvenAsphere { base: c, .. } => {
                        vec![c.vertex.length() + c.half_ap]
                    }
                    Curve2::Arc {
                        center, radius_mm, ..
                    } => vec![center.length() + radius_mm],
                })
                .fold(0.0_f64, f64::max);
            r.max(1.0)
        }
    }
}

/// A surface object's meridional tilt (its plane-normal angle in X–Y, degrees),
/// for the inspector's `rotZ`. The normal is `q · +Z`; only its X–Y angle matters
/// for a meridional surface.
fn surface_tilt_deg(s: &oc_core::scene3::SurfaceObject3) -> f64 {
    let [x, y, z, w] = s.common.transform.rot_xyzw;
    let nx = 2.0 * (z * x + w * y);
    let ny = 2.0 * (z * y - w * x);
    ny.atan2(nx).to_degrees()
}

/// An aperture's meridional tilt (its plane-normal X–Y angle, degrees).
fn aperture_tilt_deg(a: &oc_core::scene3::Aperture3) -> f64 {
    let [x, y, z, w] = a.common.transform.rot_xyzw;
    let nx = 2.0 * (z * x + w * y);
    let ny = 2.0 * (z * y - w * x);
    ny.atan2(nx).to_degrees()
}

/// A detector's meridional tilt (its face-normal X–Y angle, degrees).
fn detector_tilt_deg(d: &oc_core::scene3::Detector3) -> f64 {
    let [x, y, z, w] = d.common.transform.rot_xyzw;
    let nx = 2.0 * (z * x + w * y);
    let ny = 2.0 * (z * y - w * x);
    ny.atan2(nx).to_degrees()
}

/// An aperture's `(outer_mask_radius, clear_opening_radius)` for drawing + picking.
fn aperture_radius_opening(a: &oc_core::scene3::Aperture3) -> (f64, f64) {
    use oc_core::scene3::PlanarShape2;
    let radius_of = |s: &PlanarShape2| match s {
        PlanarShape2::Disc { radius_mm } => *radius_mm,
        PlanarShape2::Annulus { outer_mm, .. } => *outer_mm,
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => half_width_mm.hypot(*half_height_mm),
        PlanarShape2::Polygon { vertices } => {
            vertices.iter().fold(1.0_f64, |m, v| m.max(v.length()))
        }
    };
    let opening = radius_of(&a.shape.clear_opening);
    let outer = a
        .shape
        .outer_extent
        .as_ref()
        .map_or(opening * 2.5, radius_of);
    (outer.max(opening + 1.0), opening)
}

/// A surface object's kind label + a coarse hit-test radius, for the inspector
/// and pointer picking (mirrors the body helpers).
fn surface_kind_radius(s: &oc_core::scene3::SurfaceObject3) -> (&'static str, f64) {
    use oc_core::scene3::{PlanarShape2, SurfaceGeometry3};
    let kind = surface_object_kind(s);
    let radius = match &s.geometry {
        SurfaceGeometry3::Plane { shape } => match shape {
            PlanarShape2::Rectangle {
                half_width_mm,
                half_height_mm,
            } => half_width_mm.hypot(*half_height_mm),
            PlanarShape2::Disc { radius_mm } => *radius_mm,
            PlanarShape2::Annulus { outer_mm, .. } => *outer_mm,
            PlanarShape2::Polygon { vertices } => vertices
                .iter()
                .fold(1.0_f64, |m, v| m.max(v.length())),
        },
        SurfaceGeometry3::SurfaceOfRevolution { .. } => 5.0,
    };
    (kind, radius.max(1.0))
}

/// What a surface object is, by its coatings: `mirror` (a reflective front),
/// `dichroic` or `beamsplitter` (either face so coated), or `surface`.
pub(crate) fn surface_object_kind(s: &oc_core::scene3::SurfaceObject3) -> &'static str {
    let (f, b) = (
        response_kind(&s.response_front),
        response_kind(&s.response_back),
    );
    if f == "mirror" {
        "mirror"
    } else if f == "film" || b == "film" || f == "dichroic" || b == "dichroic" || f == "coated" || b == "coated"
    {
        // A film is a dichroic to the rest of the app: same role in a scene, same
        // colour treatment, same picking. What differs is that it knows the angle.
        "dichroic"
    } else if f == "split" || b == "split" {
        "beamsplitter"
    } else if f == "polarizer" || f == "retarder" {
        // Named, because "surface" tells a reader nothing and these two are
        // not interchangeable: one absorbs half the light and one absorbs none.
        f
    } else {
        "surface"
    }
}

/// A detector's in-plane half-extents (mm), for scaling a spot diagram into its
/// face. Disc/annulus report their bounding radius; a line probe has no area.
fn detector_half_extents(d: &oc_core::scene3::Detector3) -> (f64, f64) {
    use oc_core::scene3::DetectorGeometry3;
    match &d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => (*half_width_mm, *half_height_mm),
        DetectorGeometry3::Disc { radius_mm } => (*radius_mm, *radius_mm),
        DetectorGeometry3::Annulus { outer_mm, .. } => (*outer_mm, *outer_mm),
        DetectorGeometry3::LineProbeLegacy { .. } => (1.0, 1.0),
    }
}

/// A coarse radius for a source's glyph (proximity picking): its aperture plus
/// the housing margin, or a few mm for a point.
pub(crate) fn source_radius(s: &oc_core::scene3::Source3) -> f64 {
    use oc_core::scene3::SpatialDistribution3 as S;
    match s.spatial {
        S::Rectangle {
            half_width_mm,
            half_height_mm,
        } => half_width_mm.hypot(half_height_mm) + 1.0,
        S::Disc { radius_mm } => radius_mm + 1.0,
        S::LineLegacy { half_width_mm } => half_width_mm,
        S::Point => 2.0,
    }
}

/// A body's geometry kind and 3D construction, for the inspector.
fn body_kind_construction(b: &oc_core::scene3::Body3) -> (&'static str, &'static str) {
    use oc_core::scene3::SolidGeometry3;
    match &b.geometry {
        SolidGeometry3::Sphere { .. } => ("sphere", "analytic"),
        SolidGeometry3::Box { .. } => ("box", "analytic"),
        SolidGeometry3::Cylinder { .. } => ("cylinder", "analytic"),
        // A coated plate is a revolved solid like any lens, and calling it one
        // tells you nothing about what it is for. Its film is the component.
        SolidGeometry3::Profile(_)
            if b.surface.overrides.iter().any(|o| {
                matches!(
                    o.response.from_outside,
                    oc_core::scene3::SurfaceResponse::Coating { .. }
                )
            }) =>
        {
            ("dichroic", "coated plate")
        }
        SolidGeometry3::Profile(p) => (
            "profile",
            match p.construction {
                ProfileConstruction3::PlanarOnly { .. } => "planar-only",
                ProfileConstruction3::RevolveLocalX => "revolved",
                ProfileConstruction3::ExtrudeLocalZ { .. } => "extruded",
            },
        ),
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn flat_to_points(flat_xy: &[f64]) -> Vec<Vec2> {
    flat_xy
        .chunks_exact(2)
        .map(|c| Vec2::new(c[0], c[1]))
        .collect()
}

/// Clamp a UI rung index to the ladder; `rung < 0` means "use the auto-selected".
fn clamp_rung(rung: i32, ladder: &FitLadder) -> usize {
    if rung < 0 {
        ladder.chosen
    } else {
        (rung as usize).min(ladder.fits.len().saturating_sub(1))
    }
}

/// The common `{…}`-less prefix of a rung's JSON (caller appends `}` or more).
fn rung_json_open(f: &Fit) -> String {
    format!(
        "{{\"level\":{},\"name\":\"{}\",\"dof\":{},\"residual_um\":{:.5},\"rms_um\":{:.5}",
        f.level.rung(),
        f.level.name(),
        f.level.dof(),
        f.max_residual_um(),
        f.rms_residual_um()
    )
}

/// Extrude a front surface into a closed thin-body element: the surface, a flat
/// back offset `thickness` along the surface axis, and two rim lines (§5.3
/// "Closed strokes become elements"). Winding-agnostic (containment is by
/// crossing parity), so a valid loop is all that's required.
fn plano_body(front: Edge, thickness: f64) -> Vec<Edge> {
    let (p0, p1, axis) = surface_rim(&front);
    let back0 = p0 + thickness * axis;
    let back1 = p1 + thickness * axis;
    vec![
        front,
        Edge::Line { a: p1, b: back1 },
        Edge::Line { a: back1, b: back0 },
        Edge::Line { a: back0, b: p0 },
    ]
}

/// The two faces of a singlet, independent (front = 1, rear = 2).
struct Singlet {
    r1: f64,
    r2: f64,
    t: f64,
    ha1: f64,
    ha2: f64,
    k1: f64,
    k2: f64,
}

/// A singlet boundary with per-face radius / semi-aperture / conic constant
/// (front conic, top rim, rear conic, bottom rim), centred at the origin in the
/// element's local frame. The rim lines join the two apertures, so the faces may
/// differ in curvature *and* clear aperture.
#[allow(clippy::too_many_arguments)]
fn singlet_boundary(r1: f64, r2: f64, t: f64, ha1: f64, ha2: f64, k1: f64, k2: f64) -> Vec<Edge> {
    let c1 = if r1.abs() < 1e-9 { 0.0 } else { 1.0 / r1 };
    let c2 = if r2.abs() < 1e-9 { 0.0 } else { 1.0 / r2 };
    let front = ConicParams {
        vertex: Vec2::new(-t / 2.0, 0.0),
        axis: Vec2::new(1.0, 0.0),
        c: c1,
        k: k1,
        half_ap: ha1,
    };
    let rear = ConicParams {
        vertex: Vec2::new(t / 2.0, 0.0),
        axis: Vec2::new(-1.0, 0.0),
        c: c2,
        k: k2,
        half_ap: ha2,
    };
    let sag = |c: f64, k: f64, ha: f64| {
        let d = 1.0 - (1.0 + k) * c * c * ha * ha;
        c * ha * ha / (1.0 + d.max(0.0).sqrt())
    };
    let fx = -t / 2.0 + sag(c1, k1, ha1); // front rim x
    let rx = t / 2.0 - sag(c2, k2, ha2); // rear rim x
    vec![
        Edge::Conic(front),
        Edge::Line {
            a: Vec2::new(fx, ha1),
            b: Vec2::new(rx, ha2),
        },
        Edge::Conic(rear),
        Edge::Line {
            a: Vec2::new(rx, -ha2),
            b: Vec2::new(fx, -ha1),
        },
    ]
}

/// Read per-face parameters back from a two-conic singlet.
fn read_singlet(e: &Element) -> Option<Singlet> {
    if e.boundary.len() < 4 {
        return None;
    }
    let front = match e.boundary[0] {
        Edge::Conic(cp) => cp,
        _ => return None,
    };
    let rear = match e.boundary[2] {
        Edge::Conic(cp) => cp,
        _ => return None,
    };
    let radius = |c: f64| {
        if c.abs() < 1e-12 {
            f64::INFINITY
        } else {
            1.0 / c
        }
    };
    Some(Singlet {
        r1: radius(front.c),
        r2: radius(rear.c),
        t: (rear.vertex.x - front.vertex.x).abs(),
        ha1: front.half_ap,
        ha2: rear.half_ap,
        k1: front.k,
        k2: rear.k,
    })
}

#[cfg(test)]
mod m4_tests {
    use super::*;

    fn beam_and_lens() -> Canvas {
        let mut c = Canvas::new();
        c.add_collimated_source(-70.0, 0.0, 0.0, 10.0, 5, 0.5876);
        c
    }

    // --- layers (§5.6) -----------------------------------------------------

    #[test]
    fn hidden_layer_leaves_the_trace() {
        let mut c = beam_and_lens();
        let base = c.trace_colored().len(); // 5 straight rays, no optics
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17); // active layer 0
        let with_lens = c.trace_colored().len();
        assert!(with_lens > base, "a visible lens must add ray segments");
        // Hiding layer 0 removes the lens → light ignores it → straight rays again.
        assert!(c.set_layer_visible(0, false));
        assert_eq!(c.trace_colored().len(), base);
        assert!(c.set_layer_visible(0, true));
        assert_eq!(c.trace_colored().len(), with_lens);
    }

    #[test]
    fn active_layer_assignment_and_removal_reindex() {
        let mut c = Canvas::new();
        assert_eq!(c.add_layer(""), 1);
        assert_eq!(c.add_layer(""), 2);
        c.set_active_layer(2);
        assert_eq!(c.active_layer(), 2);
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        assert!(c.elements_overlay_json().contains("\"layer\":2"));
        // Removing layer 1 shifts layer 2 → 1; the element follows.
        assert!(c.remove_layer(1));
        assert!(c.elements_overlay_json().contains("\"layer\":1"));
        assert_eq!(c.active_layer(), 1);
        // The last layer cannot be removed.
        assert!(c.remove_layer(1));
        assert!(!c.remove_layer(0));
    }

    // --- undo tree (§5.2) --------------------------------------------------

    #[test]
    fn undo_redo_and_branching() {
        let mut c = beam_and_lens();
        c.history_reset("scene");
        assert!(!c.can_undo());
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        let lens_node = c.commit("add lens");
        assert!(lens_node > 0);
        assert_eq!(c.element_count(), 1);
        assert!(c.can_undo());
        // Undo removes the lens; redo restores it.
        assert!(c.undo());
        assert_eq!(c.element_count(), 0);
        assert!(c.can_redo());
        assert!(c.redo());
        assert_eq!(c.element_count(), 1);
        // Branch: from the root, take a different path (a mirror).
        assert!(c.undo());
        assert_eq!(c.element_count(), 0);
        c.add_mirror(-5.0, -5.0, 5.0, 5.0);
        c.commit("add mirror");
        assert_eq!(c.element_count(), 1);
        // Jump back to the lens branch — the tree is non-linear (§5.2).
        assert!(c.history_jump_to(lens_node as u32));
        assert_eq!(c.element_count(), 1);
        // The lens branch really is the singlet, not the mirror.
        assert!(c.elements_overlay_json().contains("\"kind\":\"singlet\""));
    }

    #[test]
    fn commit_dedups_no_ops() {
        let mut c = beam_and_lens();
        c.history_reset("scene");
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        assert!(c.commit("edit") >= 0);
        // No change since the last commit → nothing to snapshot.
        assert_eq!(c.commit("edit"), -1);
    }

    #[test]
    fn pins_survive_undo() {
        let mut c = beam_and_lens();
        let id = c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        c.history_reset("scene");
        c.set_param_pinned(id, "x", true);
        c.commit("pin x");
        c.set_param_pinned(id, "x", false);
        c.commit("unpin x");
        assert!(!c.is_param_pinned(id, "x"));
        assert!(c.undo()); // back to the pinned snapshot
        assert!(c.is_param_pinned(id, "x"));
    }

    // --- wavefront scrub (§5.5) --------------------------------------------

    #[test]
    fn wavefront_opl_is_monotonic_and_advances() {
        let mut c = beam_and_lens();
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        let wf = c.trace_wavefront();
        assert!(!wf.is_empty());
        assert_eq!(wf.len() % 11, 0);
        let mut advanced = false;
        for s in wf.chunks(11) {
            let (opl0, opl1) = (s[9], s[10]);
            assert!(opl0 <= opl1 + 1e-3, "opl must not run backwards");
            if opl1 > opl0 + 1e-6 {
                advanced = true;
            }
        }
        assert!(
            advanced,
            "at least one segment must carry the front forward"
        );
    }

    // --- variation gallery (§5.6) ------------------------------------------

    #[test]
    fn node_trace_and_summary() {
        let mut c = beam_and_lens();
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        c.add_detector(52.0, -7.0, 52.0, 7.0, 48);
        c.history_reset("scene");
        let id = c.current_history_node();
        let segs = c.trace_node_colored(id);
        assert!(!segs.is_empty() && segs.len() % 9 == 0);
        let summary = c.node_summary_json(id);
        assert!(summary.contains("\"elements\":1"));
        assert!(summary.contains("\"detectors\":1"));
        // The saved node's detector traces on the reference path.
        let det = c.node_detector_json(id, 0);
        assert!(det.contains("\"hits\""));
        assert_eq!(c.trace_node_colored(9999).len(), 0); // invalid id → empty
    }

    #[test]
    fn wavefront_curves_encode_monotonic_paths() {
        let mut c = beam_and_lens(); // collimated source, 5 rays
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        let a = c.wavefront_curves();
        assert!(!a.is_empty());
        let nrays = a[0] as usize;
        assert_eq!(nrays, 5, "one polyline per launch ray");
        let mut i = 1usize;
        for _ in 0..nrays {
            let len = a[i] as usize;
            i += 1;
            assert!(len >= 2, "each ray path has at least a start and an end");
            let mut prev = f32::NEG_INFINITY;
            for _ in 0..len {
                let opl = a[i + 2];
                assert!(opl >= prev - 1e-3, "OPL along a ray path must not decrease");
                prev = opl;
                i += 3;
            }
        }
        assert_eq!(i, a.len(), "the flat encoding is consumed exactly");
    }
}

/// The 3D scene (Scene3) view API (PR 3D-09).
#[cfg(test)]
mod scene3_tests {
    use super::*;

    /// A v1 Canvas with a collimated beam and a singlet, as the demo builds.
    fn v1_beam_and_lens() -> Canvas {
        let mut c = Canvas::new();
        c.add_collimated_source(-70.0, 0.0, 0.0, 10.0, 5, 0.5876);
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        c
    }

    #[test]
    fn the_default_scene3_boots_full_3d_and_focuses() {
        let mut c = Canvas::new();
        let body_id = c.new_default_scene3();

        // A genuine 3D scene: authored, not migrated, so it previews full 3D.
        assert_eq!(c.preview_mode(), "full_3d");
        assert!(body_id >= 0.0);

        // The authoritative f64 path traces rays through it.
        assert!(
            !c.trace_projected_3d().is_empty(),
            "the default scene must trace rays"
        );

        // Rays reach the detector near the focus: it registers flux.
        let det_id = c.scene3.as_ref().unwrap().detectors[0].common.id.0 as f64;
        let readout = c.detector_result_json(det_id);
        assert!(readout.contains("\"hits\":"), "{readout}");
        assert!(!readout.contains("\"hits\":0,"), "beam must reach the detector: {readout}");
    }

    #[test]
    fn the_projected_trace_leaves_the_detector_maps_of_the_readout() {
        let mut c = Canvas::new();
        assert!(c.last_detector_maps_f32().is_empty());
        c.new_default_scene3();
        assert!(!c.trace_projected_3d().is_empty());
        let maps = c.last_detector_maps_f32();
        assert_eq!(maps[0], 1.0, "one area detector");
        let det_id = c.scene3.as_ref().unwrap().detectors[0].common.id.0;
        assert_eq!(maps[1] as u64, det_id);
        let (nu, nv) = (maps[2] as usize, maps[3] as usize);
        let cells = &maps[6..];
        assert_eq!(cells.len(), nu * nv);

        // The same cells as the f64 readout, to f32 rounding.
        let readout: serde_json::Value =
            serde_json::from_str(&c.detector_result_json(det_id as f64)).unwrap();
        assert_eq!(readout["bins"][0].as_u64().unwrap() as usize, nu);
        let incident = readout["incident"].as_array().unwrap();
        assert_eq!(incident.len(), cells.len());
        for (a, b) in cells.iter().zip(incident) {
            let b = b.as_f64().unwrap();
            assert!((*a as f64 - b).abs() <= 1e-6 * b.abs().max(1e-12));
        }
        assert!(
            cells.iter().any(|v| *v > 0.0),
            "the beam reaches the detector"
        );

        // A scene with no detector leaves an empty map list.
        let id = det_id as f64;
        assert!(c.delete_object_3d(id));
        c.trace_projected_3d();
        assert_eq!(c.last_detector_maps_f32(), vec![0.0]);
    }

    #[test]
    fn sources_are_listed_for_the_3d_view_only() {
        let mut c = Canvas::new();
        assert_eq!(c.source_objects_json(), "[]");
        c.new_default_scene3();
        let v: serde_json::Value = serde_json::from_str(&c.source_objects_json()).unwrap();
        let items = v.as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["kind"], "source");
        assert_eq!(items[0]["construction"], "rectangle");
        assert_eq!(items[0]["is3d"], true);
        // The shared object list is unchanged: the 2D view has no source glyph.
        assert!(!c.scene_objects_json().contains("\"source\""));
        // A source moves and turns through the generic setters.
        let id = items[0]["id"].as_f64().unwrap();
        assert!(c.set_object_position(id, -40.0, 3.0, 0.0));
        assert!(c.set_object_rotation_z_deg(id, 10.0));
        let v: serde_json::Value = serde_json::from_str(&c.source_objects_json()).unwrap();
        assert_eq!(v[0]["pos"][1].as_f64().unwrap(), 3.0);
        assert!((v[0]["rotZ"].as_f64().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn the_component_view_reads_and_reshapes_a_singlet() {
        let mut c = Canvas::new();
        let lens = c.new_default_scene3();
        let v: serde_json::Value = serde_json::from_str(&c.singlet_shape_json(lens, 8)).unwrap();
        assert_eq!(v["r1"].as_f64(), Some(50.0));
        assert_eq!(v["asph1"].as_array().unwrap().len(), 8);
        // The default N-BK7 biconvex: f ≈ 49 mm, and its faces as polylines
        // from the vertex (x = -2 and x = +2) to the rim (y = 12).
        let efl = v["efl"].as_f64().unwrap();
        assert!((efl - 49.04).abs() < 0.05, "efl {efl}");
        let front = v["faces"][0]["pts"].as_array().unwrap();
        assert_eq!(front.len(), 18);
        assert_eq!(front[0].as_f64(), Some(-2.0));
        assert_eq!(front[17].as_f64(), Some(12.0));
        assert_eq!(v["faces"][1]["vertex"].as_f64(), Some(2.0));
        let edge = v["edge"].as_f64().unwrap();
        assert!(edge > 0.0 && edge < 4.0, "edge {edge}");

        // Control points: fit the front face through two dragged points.
        let base = c.singlet_face_sag_f64(lens, false, &[6.0, 12.0], false);
        let before = c.trace_projected_3d();
        let rev = c.scene_revision();
        assert!(c.fit_singlet_face(lens, false, &[6.0, 12.0], &[base[0] + 0.05, base[1] + 0.3]));
        assert!(c.scene_revision() > rev);
        let now = c.singlet_face_sag_f64(lens, false, &[6.0, 12.0], false);
        assert!((now[0] - base[0] - 0.05).abs() < 1e-9 && (now[1] - base[1] - 0.3).abs() < 1e-9);
        assert_ne!(c.trace_projected_3d(), before, "the asphere bends the rays");
        let v: serde_json::Value = serde_json::from_str(&c.singlet_shape_json(lens, 8)).unwrap();
        assert!(v["asph1"][0].as_f64().unwrap() != 0.0);
        // The base conic is still the sphere it was.
        let conic = c.singlet_face_sag_f64(lens, false, &[6.0, 12.0], true);
        assert!((conic[0] - base[0]).abs() < 1e-12 && (conic[1] - base[1]).abs() < 1e-12);
        // The bend handle: the rim moves, the aspheric terms stay.
        assert!(c.bend_singlet_face(lens, false, 1.0));
        assert!((c.singlet_face_sag_f64(lens, false, &[12.0], false)[0] - 1.0).abs() < 1e-9);
        assert!(!c.bend_singlet_face(lens, false, 20.0));
        assert!(c.singlet_face_sag_f64(lens, false, &[13.0], false)[0].is_nan());
        // Refusals leave the revision alone.
        let rev = c.scene_revision();
        assert!(!c.fit_singlet_face(lens, false, &[6.0], &[0.1, 0.2]));
        assert!(
            !c.fit_singlet_face(lens, false, &[12.0], &[30.0]),
            "faces would cross"
        );
        assert_eq!(c.scene_revision(), rev);

        // The card's setter keeps the terms; the shape setter replaces them.
        assert!(c.set_singlet_geom(lens, 45.0, -50.0, 4.0, 12.0, 0.0, 0.0));
        let v: serde_json::Value = serde_json::from_str(&c.singlet_shape_json(lens, 8)).unwrap();
        assert!(v["asph1"][0].as_f64().unwrap() != 0.0);
        assert!(c.set_singlet_shape(lens, 45.0, -50.0, 5.0, 12.0, -0.5, 0.0, &[], &[1e-6]));
        let v: serde_json::Value = serde_json::from_str(&c.singlet_shape_json(lens, 8)).unwrap();
        assert_eq!(v["asph1"][0].as_f64(), Some(0.0));
        assert_eq!(v["asph2"][0].as_f64(), Some(1e-6));
        assert_eq!(v["conic1"].as_f64(), Some(-0.5));
        assert!(!c.set_singlet_shape(lens, 45.0, -50.0, 5.0, 12.0, 0.0, 0.0, &[0.0; 9], &[]));
        // Not a singlet.
        assert_eq!(c.singlet_shape_json(999.0, 8), "null");
        assert!(c
            .singlet_face_sag_f64(999.0, false, &[1.0], false)
            .is_empty());
    }

    #[test]
    fn a_film_dichroic_is_added_read_back_and_edited() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        // The real component: a coated glass plate with thickness.
        let id = c.add_plate_dichroic(-35.0, 0.0, 0.0, 15.0, 3.0, 135.0, 550.0, true);
        assert!(id >= 0.0, "the plate is valid");

        let film: serde_json::Value =
            serde_json::from_str(&c.surface_film_json(id)).expect("film JSON");
        // The plate is hung at 135°, so the light meets it at 45°, and that is
        // the angle the design was solved for and the readout reports.
        assert!((film["aoiDeg"].as_f64().unwrap() - 45.0).abs() < 1e-9);
        assert_eq!(film["design"]["kind"], "long-pass");
        assert_eq!(film["design"]["lambdaNm"].as_f64(), Some(550.0));
        // A real stack, and one that really does cut where it was asked to.
        let layers = film["layers"].as_array().expect("layers");
        assert_eq!(layers.len(), 17, "eight symmetric periods, half layers merged");
        // It is a real plate: glass with a thickness and a clear aperture.
        assert_eq!(film["thicknessMm"].as_f64(), Some(3.0));
        assert_eq!(film["radiusMm"].as_f64(), Some(15.0));
        assert!((film["substrateN"].as_f64().unwrap() - 1.5168).abs() < 1e-9);
        assert!(layers[0]["thicknessNm"].as_f64().unwrap() > 10.0);
        let edge = film["edgeNm"].as_f64().expect("a measured edge");
        assert!((edge - 550.0).abs() < 4.0, "measured edge {edge}");
        assert!(film["peakR"].as_f64().unwrap() > 0.98);

        // R and T now depend on the angle, and the readout says so. The band
        // walks blue as the angle opens up, so on axis this stack's edge sits
        // some 40 nm HIGHER than the 550 nm it was solved for at 45°. A
        // wavelength between the two edges is reflected by one and passed by
        // the other, which is the whole behaviour that used to be missing.
        let at = |aoi: f64| c.surface_rt_f64(id, false, &[575.0], aoi)[0];
        assert!(
            at(45.0) < 0.25,
            "575 nm is past the edge at the design angle: {}",
            at(45.0)
        );
        assert!(
            at(0.0) > 0.8,
            "on axis the band has not reached 575 nm yet: {}",
            at(0.0)
        );

        // A layer edit reaches the stack the tracer reads.
        assert!(c.set_coating_layer_thickness(id, 0, 90.0));
        let after: serde_json::Value = serde_json::from_str(&c.surface_film_json(id)).unwrap();
        assert_eq!(after["layers"][0]["thicknessNm"].as_f64(), Some(90.0));
        assert!(!c.set_coating_layer_thickness(id, 99, 90.0), "no such layer");
        assert!(!c.set_coating_layer_thickness(id, 0, f64::NAN));

        // Re-designing replaces the stack.
        assert!(c.set_surface_coating_design(id, "short-pass", 600.0, 45.0, 10, 2.35, 1.46));
        let sp: serde_json::Value = serde_json::from_str(&c.surface_film_json(id)).unwrap();
        assert_eq!(sp["design"]["kind"], "short-pass");
        assert!((sp["edgeNm"].as_f64().unwrap() - 600.0).abs() < 5.0);
        assert!(!c.set_surface_coating_design(id, "nonsense", 600.0, 45.0, 10, 2.35, 1.46));

        // An object with no film says so rather than inventing one.
        let plain = c.add_beam_splitter(-60.0, 0.0, 15.0, 135.0, 0.5, 0.5);
        assert_eq!(c.surface_film_json(plain), "null");
    }

    #[test]
    fn splitters_and_dichroics_are_added_listed_and_read_back() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let bs = c.add_beam_splitter(-35.0, 0.0, 15.0, 135.0, 0.5, 0.5);
        let dc = c.add_dichroic(-50.0, 0.0, 15.0, 135.0, 550.0, 20.0, true);
        assert!(bs >= 0.0 && dc >= 0.0);
        assert_eq!(c.add_beam_splitter(0.0, 0.0, 5.0, 0.0, 0.8, 0.8), -1.0);

        let objs: serde_json::Value = serde_json::from_str(&c.scene_objects_json()).unwrap();
        let kind = |id: f64| {
            objs.as_array()
                .unwrap()
                .iter()
                .find(|o| o["id"].as_f64() == Some(id))
                .unwrap()["kind"]
                .clone()
        };
        assert_eq!(kind(bs), "beamsplitter");
        assert_eq!(kind(dc), "dichroic");
        assert_eq!(
            c.surface_response_json(bs),
            "{\"front\":\"split\",\"back\":\"split\"}"
        );

        let coat: serde_json::Value = serde_json::from_str(&c.surface_coating_json(dc)).unwrap();
        assert_eq!(coat["front"]["kind"], "dichroic");
        assert!(coat["front"]["r"].is_null());
        assert_eq!(coat["front"]["dichroic"]["edgeNm"].as_f64(), Some(550.0));
        assert_eq!(coat["back"]["dichroic"]["longPass"], true);
        let coat: serde_json::Value = serde_json::from_str(&c.surface_coating_json(bs)).unwrap();
        assert_eq!(
            (coat["front"]["r"].as_f64(), coat["front"]["t"].as_f64()),
            (Some(0.5), Some(0.5))
        );
        assert!(coat["front"]["dichroic"].is_null());
        assert_eq!(c.surface_coating_json(-1.0), "null");

        // The curves the tracer samples.
        // A table-driven dichroic ignores the angle, so 0 is as good as any.
        let rt = c.surface_rt_f64(dc, false, &[450.0, 550.0, 650.0], 0.0);
        assert_eq!(rt.len(), 6);
        assert_eq!((rt[0], rt[1]), (0.98, 0.0));
        assert!((rt[2] - 0.49).abs() < 1e-12);
        assert_eq!((rt[4], rt[5]), (0.0, 0.95));

        // Edits.
        assert!(c.set_surface_split(bs, 0.2, 0.8));
        assert_eq!(c.surface_rt_f64(bs, true, &[500.0], 0.0), vec![0.2, 0.8]);
        assert!(!c.set_surface_split(bs, 0.9, 0.9));
        assert!(c.set_surface_dichroic(dc, 600.0, 30.0, false, 0.9, 0.8));
        let coat: serde_json::Value = serde_json::from_str(&c.surface_coating_json(dc)).unwrap();
        assert_eq!(coat["front"]["dichroic"]["longPass"], false);
        assert_eq!(coat["front"]["dichroic"]["tMax"].as_f64(), Some(0.8));
        assert!(!c.set_surface_dichroic(dc, f64::NAN, 30.0, false, 0.9, 0.8));

        // The 2D section names both and tints the dichroic by what it reflects.
        let xs: serde_json::Value = serde_json::from_str(&c.scene_cross_section_json()).unwrap();
        let items = xs.as_array().unwrap();
        assert!(items
            .iter()
            .any(|p| p["kind"] == "splitter" && p.get("tint").is_none()));
        let tinted = items.iter().find(|p| p["kind"] == "dichroic").unwrap();
        // A short-pass edge at 600 nm reflects the red end.
        let tint = tinted["tint"].as_str().unwrap();
        assert!(tint.starts_with("rgb(255,"), "{tint}");
    }

    #[test]
    fn sequence_extraction_reads_the_default_scene() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let (src_id, det_id) = {
            let s = c.scene3.as_ref().unwrap();
            (s.sources[0].id.raw() as f64, s.detectors[0].common.id.0 as f64)
        };

        // The axial ray threads the singlet's two faces on its way to the
        // detector, and the prescription says so in beam order.
        let json = c.sequence_json(src_id, det_id).unwrap();
        let p = oc_core::seq3::SequencePrescription3::from_json(&json).unwrap();
        assert_eq!(p.hits.len(), 2, "front + back of the singlet: {json}");
        assert!(p.hits.iter().all(|h| h.kind == oc_core::seq3::SeqHitKind::Body));
        assert_eq!(p.hits[0].object, p.hits[1].object);
        assert_ne!(p.hits[0].patch, p.hits[1].patch);
        assert!(p.hits[0].cum_arc_mm < p.terminal.cum_arc_mm);
        // The unknown-id error path lives in oc-trace's sequence3 tests:
        // constructing a JsError panics on non-wasm targets.
    }

    #[test]
    fn source_reconfig_and_drawn_mirror_show_up() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        // The picker reads back the boot source (a collimated rectangle).
        assert!(c.first_source_json().contains("\"kind\":\"collimated_rect\""));

        // Switch to a point source; the trace still reaches the detector.
        let rev0 = c.scene_revision();
        assert!(c.set_source_config("point", 3.0, 0.0, 200));
        assert!(c.scene_revision() > rev0);
        assert!(c.first_source_json().contains("\"kind\":\"point\""));

        // Draw a fold mirror: it appears in the object list as a non-body surface.
        let before = c.scene_objects_json().matches("\"id\":").count();
        let id = c.add_planar_mirror(20.0, -6.0, 20.0, 6.0, 0.95);
        assert!(id >= 0.0);
        let objs = c.scene_objects_json();
        assert_eq!(objs.matches("\"id\":").count(), before + 1);
        assert!(objs.contains("\"kind\":\"mirror\""));
        assert!(objs.contains("\"body\":false"));
        // Every item carries its full quaternion (imported scenes hold
        // non-pure-Z rotations the scalar rotZ cannot represent).
        assert_eq!(objs.matches("\"rot\":[").count(), objs.matches("\"id\":").count());
        assert!(c.delete_object_3d(id));

        // A configurable element: front/back responses + tilt round-trip.
        let eid = c.add_planar_element(25.0, 0.0, 8.0, 45.0);
        assert!(c.surface_response_json(eid).contains("\"front\":\"mirror\""));
        assert!(c.set_surface_response(eid, "absorb", "clear"));
        assert!(c.surface_response_json(eid).contains("\"front\":\"absorb\""));
        assert!(c.set_surface_tilt_z_deg(eid, -30.0));
        // The object list reports the surface's (now negative) tilt as its rotZ.
        assert!(c.scene_objects_json().contains("\"rotZ\":-2") || c.scene_objects_json().contains("\"rotZ\":-3"));
    }

    #[test]
    fn apertures_and_prisms_show_up_and_trace() {
        let mut c = Canvas::new();
        c.new_default_scene3();

        // A prism: a glass-editable extruded body.
        let pid = c.add_prism(-40.0, 0.0, 16.0, 10.0, 1.62, 36.0);
        assert!(pid >= 0.0);
        assert!(c.scene_objects_json().contains("\"construction\":\"extruded\""));
        assert!(c.body_glass_json(pid).contains("\"nd\":1.62"));

        // An aperture: a non-body object, selectable with an opening + tilt.
        let aid = c.add_aperture(-20.0, 0.0, 4.0, 15.0);
        let objs = c.scene_objects_json();
        assert!(objs.contains("\"kind\":\"aperture\""));
        assert!(objs.contains("\"opening\":4"));
        assert!(c.set_surface_tilt_z_deg(aid, 25.0)); // apertures tilt too
        // A fresh aperture is a plain disc: inner 0, outer = the opening.
        assert_eq!(c.aperture_opening_json(aid), "{\"inner\":0,\"outer\":4}");
        // Widen it into an annular stop (min/max radius) — a central obstruction.
        assert!(c.set_aperture_opening(aid, 2.0, 6.0));
        assert_eq!(c.aperture_opening_json(aid), "{\"inner\":2,\"outer\":6}");
        // Back to a plain hole when the inner radius is cleared.
        assert!(c.set_aperture_opening(aid, 0.0, 5.0));
        assert_eq!(c.aperture_opening_json(aid), "{\"inner\":0,\"outer\":5}");
        // Invalid geometry (outer <= inner) is refused, scene unchanged.
        assert!(!c.set_aperture_opening(aid, 5.0, 5.0));
        assert_eq!(c.aperture_opening_json(aid), "{\"inner\":0,\"outer\":5}");
        // Non-apertures have no opening.
        assert_eq!(c.aperture_opening_json(pid), "null");
        // Both trace without error.
        assert!(!c.trace_projected_3d().is_empty());
    }

    /// End-to-end proof of the annular-stop card flow: `setApertureOpening` on a
    /// live scene makes the projected trace actually block the on-axis ray, the
    /// same call path the inspector card drives.
    #[test]
    fn an_annular_aperture_stops_the_on_axis_ray_in_the_projected_trace() {
        // Does any projected segment sit on the axis in the gap after the stop
        // (x ∈ (-40, -2), |y| < 2)? That region is only reached by a ray that got
        // through the centre of the stop at x = -40.
        let on_axis_after_stop = |c: &Canvas| {
            let segs = c.trace_projected_3d();
            segs.chunks(11).any(|s| {
                let (ax, ay, bx, by) = (s[0], s[1], s[2], s[3]);
                let mid_x = 0.5 * (ax + bx);
                let mid_y = 0.5 * (ay + by);
                mid_x > -40.0 && mid_x < -2.0 && mid_y.abs() < 2.0
            })
        };

        let mut c = Canvas::new();
        c.new_default_scene3(); // collimated beam +X, singlet at 0, detector at 48
        let aid = c.add_aperture(-40.0, 0.0, 5.0, 0.0); // a plain hole, radius 5

        // Plain hole: the on-axis ray sails through the centre and onto the lens.
        assert!(
            on_axis_after_stop(&c),
            "a plain aperture must pass the on-axis ray"
        );

        // Turn it into a 3..5 mm ring — the central obstruction must stop it.
        assert!(c.set_aperture_opening(aid, 3.0, 5.0));
        assert!(
            !on_axis_after_stop(&c),
            "the annulus centre must block the on-axis ray"
        );
    }

    #[test]
    fn detectors_are_first_class_movable_tiltable_multi_add_objects() {
        let mut c = Canvas::new();
        c.new_default_scene3(); // ships with one detector at x = 48

        // Add a second area detector: it shows up as a non-body "detector" object.
        let did = c.add_area_detector_3d(30.0, 0.0, 8.0, 0.0);
        assert!(did >= 0.0);
        let objs = c.scene_objects_json();
        let n_detectors = objs.matches("\"kind\":\"detector\"").count();
        assert_eq!(n_detectors, 2, "both detectors must be listed as objects");
        assert!(objs.contains("\"body\":false"));

        // Movable via the generic object-position setter (the drag path).
        assert!(c.set_object_position(did, 33.0, 4.0, 0.0));
        assert!(c.scene_objects_json().contains("[33,4,0]"));

        // Tiltable via the surface-tilt setter (the non-body rotation-handle path).
        assert!(c.set_surface_tilt_z_deg(did, 20.0));
        assert!(
            (c.scene3.as_ref().unwrap().surface_tilt_z_deg(did as u64).unwrap() - 20.0).abs()
                < 1e-6
        );

        // Its per-detector readout is reachable by id (dock follows selection).
        assert_ne!(c.detector_result_json(did), "null");
        // Deletable like any other object.
        assert!(c.delete_object_3d(did));
        assert_eq!(
            c.scene_objects_json().matches("\"kind\":\"detector\"").count(),
            1
        );
    }

    #[test]
    fn source_wavelengths_can_be_set_and_read_back() {
        let mut c = Canvas::new();
        c.new_default_scene3();

        // Set the Fraunhofer F/d/C lines (nm), equal weight (empty weights).
        assert!(c.set_source_wavelengths(&[486.13, 587.56, 656.27], &[]));
        let js = c.first_source_wavelengths_json();
        assert!(js.contains("\"nm\":486.13"), "{js}");
        assert!(js.contains("\"nm\":587.56"), "{js}");
        assert!(js.contains("\"nm\":656.27"), "{js}");

        // A single wavelength is fine; an empty list is refused.
        assert!(c.set_source_wavelengths(&[532.0], &[]));
        assert!(!c.set_source_wavelengths(&[], &[]));

        // Colour map: a visible line is coloured, an IR line falls back to yellow.
        assert_eq!(c.wavelength_css_color(900.0), "rgb(255,230,0)"); // IR → yellow
        assert_ne!(c.wavelength_css_color(530.0), "rgb(255,230,0)"); // green ≠ yellow
    }

    #[test]
    fn the_default_detector_reports_spot_hits() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let a = c.first_detector_hits_f32();
        assert!(a.len() >= 3, "header must be present");
        let (hw, hh, n) = (a[0], a[1], a[2] as usize);
        assert!(hw > 0.0 && hh > 0.0, "detector has a finite face");
        assert!(n > 0, "the focused beam must land some hits");
        assert_eq!(a.len(), 3 + n * 6, "flat layout is [hw,hh,n, 6×n]");
        // Every recorded hit lies within the detector face.
        for i in 0..n {
            let (u, v) = (a[3 + i * 6], a[3 + i * 6 + 1]);
            assert!(u.abs() <= hw + 1e-3 && v.abs() <= hh + 1e-3, "hit ({u},{v}) off-face");
        }
    }

    #[test]
    fn editing_the_default_singlet_re_traces_and_can_be_refused() {
        let mut c = Canvas::new();
        let id = c.new_default_scene3();
        let rev0 = c.scene_revision();

        // A valid radius edit lands and bumps the revision.
        assert!(c.set_singlet_geom(id, 40.0, -50.0, 4.0, 12.0, 0.0, 0.0));
        assert!(c.scene_revision() > rev0);
        let g = c.singlet_geom_json(id);
        assert!(g.contains("\"r1\":40"), "{g}");

        // An impossible geometry is refused; the scene and revision are unchanged.
        let rev1 = c.scene_revision();
        assert!(!c.set_singlet_geom(id, 5.0, -5.0, 4.0, 40.0, 0.0, 0.0));
        assert_eq!(c.scene_revision(), rev1);
        assert!(c.scene3.as_ref().unwrap().validate().is_ok());
    }

    #[test]
    fn loading_a_v1_document_migrates_without_inventing_a_third_dimension() {
        let mut c = v1_beam_and_lens();
        let v1_json = c.to_json();

        let rev0 = c.scene_revision();
        let report = c.load_scene3_json(&v1_json).unwrap();
        assert!(report.contains("\"migratedFromV1\":true"), "{report}");
        assert!(report.contains("\"dimension\":3"), "{report}");
        // The lens body is unresolved (PlanarOnly): no revolve/extrude invented.
        assert!(
            report.contains("\"unresolvedThirdDimension\":true"),
            "{report}"
        );
        assert!(report.contains("\"planarOnlyBodies\":["), "{report}");

        assert_eq!(c.scene_dimension(), 3);
        // Nothing has been promoted, so the badge is legacy-planar, not full 3D.
        assert_eq!(c.preview_mode(), "legacy_planar");
        assert!(c.scene_revision() > rev0, "load must bump the revision");
    }

    #[test]
    fn a_migrated_scene_round_trips_as_v2() {
        let mut c = v1_beam_and_lens();
        let v1_json = c.to_json();
        c.load_scene3_json(&v1_json).unwrap();

        let v2 = c.scene3_json();
        assert!(
            v2.contains("\"version\": 2"),
            "v2 header: {}",
            &v2[..80.min(v2.len())]
        );
        assert!(v2.contains("\"dimension\": 3"));

        // Reloading the v2 output is a parse, not a migration.
        let report = c.load_scene3_json(&v2).unwrap();
        assert!(report.contains("\"migratedFromV1\":false"), "{report}");
    }

    #[test]
    fn no_scene_loaded_reports_dimension_zero_and_no_preview() {
        let c = Canvas::new();
        assert_eq!(c.scene_dimension(), 0);
        assert_eq!(c.preview_mode(), "none");
        assert_eq!(c.scene3_json(), "");
        assert!(c.trace_projected_3d().is_empty());
        assert_eq!(c.capability_report_json(), "null");
        assert_eq!(c.detector_result_json(0.0), "null");
    }

    #[test]
    fn the_view_plane_accepts_an_orthonormal_basis() {
        // (Off-wasm, the error branch cannot be exercised: `JsError` construction
        // calls a wasm-bindgen import that panics on the host. `Plane3::new` owns
        // the rejection and is unit-tested in oc-core. Here the accept path sets
        // the slice plane the projector uses.)
        let mut c = Canvas::new();
        assert!(c
            .set_view_plane(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            .is_ok());
        c.set_slice_thickness(2.5);
    }

    /// Build a v2 demo scene — a glass sphere lit by a collimated beam onto a
    /// detector — and write it to `web/public/demo-3d.ocanvas` for the browser
    /// slice preview and its Playwright spec. Ignored: run explicitly with
    /// `cargo test -p oc-wasm --lib write_demo_3d_fixture -- --ignored`.
    #[test]
    #[ignore]
    fn write_demo_3d_fixture() {
        use oc_core::id::LayerId;
        use oc_core::material::{Material, LAMBDA_D};
        use oc_core::scene::Spectrum;
        use oc_core::scene3::{
            AngularDistribution3, Body3, Detector3, DetectorGeometry3, DetectorResponse,
            DetectorTermination, ObjectCommon, SamplingSequence, SolidGeometry3, Source3,
            SourceSampling, SpatialDistribution3, SurfaceProps3,
        };
        use oc_core::vec3::Isometry3;

        let mut scene = Scene3::new();

        // A glass sphere at the origin.
        let sphere = scene.alloc_object_id();
        let mut common = ObjectCommon::new(sphere, LayerId(0));
        common.transform = Isometry3::IDENTITY;
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Sphere { radius_mm: 8.0 },
            material: Material::AbbePoint { nd: 1.6, vd: 64.0 },
            absorption: Default::default(),
            surface: SurfaceProps3::default(),
            priority: 0,
        });

        // A collimated meridional beam from the left (spread in Y, thin in Z), so
        // it focuses in the Z=0 slice.
        let src = scene.alloc_source_id();
        scene.sources.push(Source3 {
            id: src,
            transform: Isometry3::from_pos(Vec3::new(-30.0, 0.0, 0.0)),
            spatial: SpatialDistribution3::Rectangle {
                half_width_mm: 6.0,
                half_height_mm: 0.25,
            },
            angular: AngularDistribution3::Direction { local_dir: Vec3::X },
            spectrum: Spectrum::Monochromatic(LAMBDA_D),
            flux: 1.0,
            samples: SourceSampling {
                spatial_samples: 21,
                angular_samples: 1,
                seed: 1,
                sequence: SamplingSequence::Grid,
            },
        });

        // A detector past the focus, facing back down the axis. Its local +Z
        // maps to world -X under a -90° rotation about Y: q = (0, sin(-45°), 0,
        // cos(-45°)).
        let det = scene.alloc_object_id();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut dcommon = ObjectCommon::new(det, LayerId(0));
        dcommon.transform = Isometry3::new(Vec3::new(24.0, 0.0, 0.0), [0.0, -s, 0.0, s]).unwrap();
        scene.detectors.push(Detector3 {
            common: dcommon,
            geometry: DetectorGeometry3::Rectangle {
                half_width_mm: 8.0,
                half_height_mm: 8.0,
            },
            bins: [48, 48],
            response: DetectorResponse::Unity,
            termination: DetectorTermination::Terminate,
        });

        scene.validate().expect("demo scene validates");

        // Sanity: it traces to non-empty projected segments and full-3D preview.
        let mut c = Canvas::new();
        c.scene3 = Some(scene.clone());
        assert_eq!(c.preview_mode(), "full_3d");
        assert!(!c.trace_projected_3d().is_empty(), "demo must trace rays");

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../web/public/demo-3d.ocanvas"
        );
        std::fs::write(path, scene.to_json()).expect("write demo fixture");
        eprintln!("wrote {path}");
    }

    /// Build a v2 scene that **folds the beam out of the Z=0 plane** — a mirror
    /// tilted so a collimated beam reflects toward +Y+Z — and write it to
    /// `web/public/demo-fold.ocanvas`. This is the out-of-plane-fold exit
    /// criterion (§20): the reflected path climbs off the slice, so the slice
    /// filter reveals it. Ignored (fixture generator).
    #[test]
    #[ignore]
    fn write_demo_fold_fixture() {
        use oc_core::id::LayerId;
        use oc_core::material::LAMBDA_D;
        use oc_core::scene::Spectrum;
        use oc_core::scene3::{
            AngularDistribution3, Detector3, DetectorGeometry3, DetectorResponse,
            DetectorTermination, ObjectCommon, PlanarShape2, SamplingSequence, Source3,
            SourceSampling, SpatialDistribution3, SpectralScalar, SurfaceGeometry3, SurfaceObject3,
            SurfaceResponse,
        };
        use oc_core::vec3::Isometry3;

        // A rotation carrying local +Z onto `normal`, without pulling in glam:
        // build it from the axis-angle of the shortest arc from +Z.
        fn facing(pos: Vec3, normal: Vec3) -> Isometry3 {
            let n = normal.normalize();
            let z = Vec3::Z;
            let dot = z.dot(n).clamp(-1.0, 1.0);
            if dot > 1.0 - 1e-12 {
                return Isometry3::from_pos(pos);
            }
            if dot < -1.0 + 1e-12 {
                // 180°: rotate about any axis ⟂ Z.
                return Isometry3::new(pos, [1.0, 0.0, 0.0, 0.0]).unwrap();
            }
            let axis = z.cross(n).normalize();
            let angle = dot.acos();
            let (s, cw) = ((angle * 0.5).sin(), (angle * 0.5).cos());
            Isometry3::new(pos, [axis.x * s, axis.y * s, axis.z * s, cw]).unwrap()
        }

        let mut scene = Scene3::new();

        // A collimated beam along +X in the Z=0 plane.
        let src = scene.alloc_source_id();
        scene.sources.push(Source3 {
            id: src,
            transform: Isometry3::from_pos(Vec3::new(-20.0, 0.0, 0.0)),
            spatial: SpatialDistribution3::Rectangle {
                half_width_mm: 3.0,
                half_height_mm: 0.25,
            },
            angular: AngularDistribution3::Direction { local_dir: Vec3::X },
            spectrum: Spectrum::Monochromatic(LAMBDA_D),
            flux: 1.0,
            samples: SourceSampling {
                spatial_samples: 9,
                angular_samples: 1,
                seed: 1,
                sequence: SamplingSequence::Grid,
            },
        });

        // A mirror at the origin whose normal folds +X to (0, 0.6, 0.8): out of
        // the plane. (Reflection is invariant to the normal's sign.)
        let mirror = scene.alloc_object_id();
        let mut mcommon = ObjectCommon::new(mirror, LayerId(0));
        mcommon.transform = facing(Vec3::ZERO, Vec3::new(-1.0, 0.6, 0.8));
        let resp = SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(0.95),
        };
        scene.surfaces.push(SurfaceObject3 {
            common: mcommon,
            geometry: SurfaceGeometry3::Plane {
                shape: PlanarShape2::Rectangle {
                    half_width_mm: 8.0,
                    half_height_mm: 8.0,
                },
            },
            response_front: resp.clone(),
            response_back: resp,
        });

        // A detector up off the plane, catching the folded beam.
        let det = scene.alloc_object_id();
        let mut dcommon = ObjectCommon::new(det, LayerId(0));
        dcommon.transform = facing(Vec3::new(0.0, 9.0, 12.0), Vec3::new(0.0, -0.6, -0.8));
        scene.detectors.push(Detector3 {
            common: dcommon,
            geometry: DetectorGeometry3::Rectangle {
                half_width_mm: 8.0,
                half_height_mm: 8.0,
            },
            bins: [48, 48],
            response: DetectorResponse::Unity,
            termination: DetectorTermination::Terminate,
        });

        scene.validate().expect("fold scene validates");
        let mut c = Canvas::new();
        c.scene3 = Some(scene.clone());
        assert_eq!(c.preview_mode(), "full_3d");
        let segs = c.trace_projected_3d();
        // Some segment endpoint must be well off the Z=0 plane (the fold).
        let off_plane = segs.chunks(9).any(|s| s[7].abs() > 2.0 || s[8].abs() > 2.0);
        assert!(off_plane, "the fold must leave the plane");

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../web/public/demo-fold.ocanvas"
        );
        std::fs::write(path, scene.to_json()).expect("write fold fixture");
        eprintln!("wrote {path}");
    }

    #[test]
    fn dragging_moves_a_body_and_bumps_the_revision() {
        use oc_core::id::LayerId;
        use oc_core::material::Material;
        use oc_core::scene3::{
            AbsorptionModel, Body3, ObjectCommon, SolidGeometry3, SurfaceProps3,
        };
        use oc_core::vec3::Isometry3;

        let mut scene = Scene3::new();
        let id = scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = Isometry3::IDENTITY;
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Sphere { radius_mm: 8.0 },
            material: Material::AbbePoint { nd: 1.5, vd: 64.0 },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });

        let mut c = Canvas::new();
        c.scene3 = Some(scene);

        // The inspector carries a hit-test radius.
        assert!(c.scene_objects_json().contains("\"radius\":8"));

        // An in-plane move updates the position and bumps the revision.
        let rev = c.scene_revision();
        assert!(c.set_body_position(id.0 as f64, 0.0, 5.0, 0.0));
        assert!(c.scene_revision() > rev, "a move must bump the revision");
        assert!(c.scene_objects_json().contains("\"pos\":[0,5,0]"));

        // An unknown body id changes nothing.
        assert!(!c.set_body_position(9999.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn the_cross_section_draws_the_elements() {
        use oc_core::id::LayerId;
        use oc_core::material::Material;
        use oc_core::scene3::{
            AbsorptionModel, Body3, Detector3, DetectorGeometry3, DetectorResponse,
            DetectorTermination, ObjectCommon, SolidGeometry3, SurfaceProps3,
        };
        use oc_core::vec3::Isometry3;

        let mut scene = Scene3::new();
        // A glass sphere at the origin: its Z=0 section is a circle.
        let sid = scene.alloc_object_id();
        let mut common = ObjectCommon::new(sid, LayerId(0));
        common.transform = Isometry3::IDENTITY;
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Sphere { radius_mm: 8.0 },
            material: Material::AbbePoint { nd: 1.5, vd: 64.0 },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        // A detector across the axis: its plane cuts Z=0 in a line.
        let did = scene.alloc_object_id();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut dcommon = ObjectCommon::new(did, LayerId(0));
        dcommon.transform = Isometry3::new(Vec3::new(20.0, 0.0, 0.0), [0.0, -s, 0.0, s]).unwrap();
        scene.detectors.push(Detector3 {
            common: dcommon,
            geometry: DetectorGeometry3::Rectangle {
                half_width_mm: 8.0,
                half_height_mm: 8.0,
            },
            bins: [16, 16],
            response: DetectorResponse::Unity,
            termination: DetectorTermination::Terminate,
        });

        let mut c = Canvas::new();
        c.scene3 = Some(scene);
        let xs = c.scene_cross_section_json();
        // The body (sphere circle) and the detector line are both present, with
        // actual points.
        assert!(xs.contains("\"kind\":\"body\""), "{xs}");
        assert!(xs.contains("\"kind\":\"detector\""), "{xs}");
        assert!(xs.contains("\"pts\":["), "{xs}");
        // No scene ⇒ empty.
        assert_eq!(Canvas::new().scene_cross_section_json(), "[]");
    }

    #[test]
    fn transform_objects_3_moves_the_scene_and_the_fast_trace_follows() {
        use oc_core::build3::SingletGeom;
        use oc_core::scene::Spectrum;

        let mut scene = Scene3::new();
        let body = scene.add_singlet(
            Vec3::new(50.0, 0.0, 0.0),
            SingletGeom::biconvex(),
            1.5168,
            64.17,
        );
        scene.add_collimated_beam(Vec3::ZERO, 1.0, 0.2, 8, Spectrum::Monochromatic(0.55));

        let mut c = Canvas::new();
        c.load_scene3_json(&serde_json::to_string(&scene).unwrap())
            .unwrap();
        let rev0 = c.scene_revision();
        let baseline = c.trace_world_3d_fast();
        assert!(!baseline.is_empty());

        // Unknown id: refused, nothing changes.
        assert!(!c.transform_objects_3(&[9999.0], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
        assert_eq!(c.scene_revision(), rev0);
        // Malformed delta: refused.
        assert!(!c.transform_objects_3(&[body.0 as f64], &[1.0, 0.0, 0.0]));

        // Identity delta: accepted, picture identical.
        assert!(c.transform_objects_3(&[body.0 as f64], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
        assert_eq!(c.trace_world_3d_fast(), baseline);

        // A real move: revision bumps and the fast retrace sees the new pose
        // (the 1 mm beam misses a lens shifted 20 mm sideways, so the picture
        // must change without any reload).
        assert!(c.transform_objects_3(&[body.0 as f64], &[0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
        assert!(c.scene_revision() > rev0);
        assert_ne!(c.trace_world_3d_fast(), baseline);
    }
}
