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

// --- 3D (Scene3) migration surface (PR 3D-09) ------------------------------
use oc_core::id::BodyId;
use oc_core::migrate_v1::{load_any, LoadReport};
use oc_core::planar_reduce::{preview_mode, PreviewMode};
use oc_core::profile2::ProfileConstruction3;
use oc_core::scene3::{
    can_promote_to_extruded, can_promote_to_revolved, promote_body, PromotionCheck, Scene3,
};
use oc_core::vec3::{Plane3, Vec3};
use oc_trace::{trace_scene3, TraceConfig3};

/// The openUC2 mode (§15.2). The freeform Canvas above does not know it exists.
pub mod openuc2;

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

    /// Promote a profile-authored body to a revolved solid (about local +X). The
    /// profile is validated (mirror-symmetric, closed, non-self-intersecting);
    /// the promotion is refused with a reason rather than guessed (§11.3).
    /// Returns a report JSON `{body, from, to}`; throws on refusal.
    #[wasm_bindgen(js_name = promoteBodyToRevolved)]
    pub fn promote_body_to_revolved(&mut self, body_id: f64) -> Result<String, JsError> {
        self.promote(body_id, ProfileConstruction3::RevolveLocalX)
    }

    /// Promote a profile-authored body to an extruded solid through local Z.
    /// Returns a report JSON `{body, from, to}`; throws on refusal.
    #[wasm_bindgen(js_name = promoteBodyToExtruded)]
    pub fn promote_body_to_extruded(
        &mut self,
        body_id: f64,
        z_min_mm: f64,
        z_max_mm: f64,
    ) -> Result<String, JsError> {
        self.promote(
            body_id,
            ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm },
        )
    }

    /// Whether a body can be promoted, without mutating it — for enabling the
    /// promotion controls. JSON `{revolve:{ok,reason},extrude:{ok,reason}}`, or
    /// `null` if the body or scene is absent.
    #[wasm_bindgen(js_name = promotionOptionsJSON)]
    pub fn promotion_options_json(&self, body_id: f64, z_min_mm: f64, z_max_mm: f64) -> String {
        let Some(scene) = &self.scene3 else {
            return "null".into();
        };
        let Some(body) = scene
            .bodies
            .iter()
            .find(|b| b.common.id.0 == body_id as u64)
        else {
            return "null".into();
        };
        let rev = check_json(&can_promote_to_revolved(body));
        let ext = check_json(&can_promote_to_extruded(body, z_min_mm, z_max_mm));
        format!("{{\"revolve\":{rev},\"extrude\":{ext}}}")
    }

    /// Trace the authoritative f64 path and project every segment onto the view
    /// plane. Nine floats per segment:
    /// `[au, av, bu, bv, flux, lambda_um, flags, dist_a, dist_b]`, where
    /// `(au,av)`/`(bu,bv)` are in-plane coordinates, `flags` is bit1=TIR bit2=ghost,
    /// and `dist_*` is the signed distance of each endpoint from the plane so the
    /// UI can filter by [`Canvas::set_slice_thickness`]. Empty if no scene.
    #[wasm_bindgen(js_name = traceProjected3D)]
    pub fn trace_projected_3d(&self) -> Vec<f32> {
        let Some(scene) = &self.scene3 else {
            return Vec::new();
        };
        let res = trace_scene3(scene, &self.cfg3());
        let plane = &self.view_plane;
        let mut out = Vec::with_capacity(res.segments.len() * 9);
        for s in &res.segments {
            let a = plane.to_local(s.a);
            let b = plane.to_local(s.b);
            let flags = (s.flags.tir as u32) * 2 + (s.flags.ghost as u32) * 4;
            out.extend_from_slice(&[
                a.x as f32,
                a.y as f32,
                b.x as f32,
                b.y as f32,
                s.flux_end as f32,
                s.lambda_um as f32,
                flags as f32,
                plane.signed_distance(s.a) as f32,
                plane.signed_distance(s.b) as f32,
            ]);
        }
        out
    }

    /// The f64 reference readout for a detector by id (§4.1: the honest numbers).
    /// JSON `{hits, totalIncident, totalSignal, cellArea, centroid, meanOpl,
    /// bins, incident}`, or `null` if there is no such detector.
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
        format!(
            "{{\"hits\":{},\"totalIncident\":{},\"totalSignal\":{},\"cellArea\":{},\"centroid\":{},\"meanOpl\":{},\"bins\":[{},{}],\"incident\":[{}]}}",
            d.hits,
            d.total_incident,
            d.total_signal,
            cell_area,
            centroid,
            mean_opl,
            d.bins[0],
            d.bins[1],
            incident.join(",")
        )
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
        let items: Vec<String> = scene
            .bodies
            .iter()
            .map(|b| {
                let (kind, construction) = body_kind_construction(b);
                let p = b.common.transform.pos;
                format!(
                    "{{\"id\":{},\"kind\":{},\"construction\":{},\"is3d\":{},\"pos\":[{},{},{}]}}",
                    b.common.id.0,
                    json_str(kind),
                    json_str(construction),
                    b.is_3d(),
                    p.x,
                    p.y,
                    p.z
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }
}

impl Canvas {
    /// Promote a body and bump the revision, mapping errors to a JS-readable form.
    fn promote(
        &mut self,
        body_id: f64,
        construction: ProfileConstruction3,
    ) -> Result<String, JsError> {
        let scene = self
            .scene3
            .as_mut()
            .ok_or_else(|| JsError::new("no 3D scene loaded"))?;
        let report = promote_body(scene, BodyId(body_id as u64), construction)
            .map_err(|e| JsError::new(&format!("{e:?}")))?;
        self.revision += 1;
        Ok(format!(
            "{{\"body\":{},\"from\":{},\"to\":{},\"undecided\":{}}}",
            report.body.0,
            json_str(&format!("{:?}", report.from)),
            json_str(&format!("{:?}", report.to)),
            report.undecided.len()
        ))
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
/// A body's geometry kind and 3D construction, for the inspector.
fn body_kind_construction(b: &oc_core::scene3::Body3) -> (&'static str, &'static str) {
    use oc_core::scene3::SolidGeometry3;
    match &b.geometry {
        SolidGeometry3::Sphere { .. } => ("sphere", "analytic"),
        SolidGeometry3::Box { .. } => ("box", "analytic"),
        SolidGeometry3::Cylinder { .. } => ("cylinder", "analytic"),
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

/// A [`PromotionCheck`] as `{ok, reason}` JSON.
fn check_json(check: &PromotionCheck) -> String {
    match check {
        PromotionCheck::Ok => "{\"ok\":true,\"reason\":null}".to_string(),
        PromotionCheck::No(reason) => format!("{{\"ok\":false,\"reason\":{}}}", json_str(reason)),
        PromotionCheck::Undecided(reason) => {
            format!("{{\"ok\":false,\"reason\":{}}}", json_str(reason))
        }
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

    #[test]
    fn promoting_a_migrated_lens_to_revolved_makes_it_full_3d() {
        let mut c = v1_beam_and_lens();
        let v1_json = c.to_json();
        let report = c.load_scene3_json(&v1_json).unwrap();

        // The migrated singlet is the one PlanarOnly body; pull its id out of the
        // report's list.
        let body_id = first_id(&report, "planarOnlyBodies").expect("a planar-only body");
        let rev = c.scene_revision();

        // The inspector shows it as an unresolved profile.
        let before = c.scene_objects_json();
        assert!(
            before.contains("\"construction\":\"planar-only\""),
            "{before}"
        );
        assert!(before.contains("\"is3d\":false"), "{before}");

        // A meridionally-symmetric singlet profile promotes to a revolved solid.
        let promo = c.promote_body_to_revolved(body_id as f64).unwrap();
        assert!(promo.contains("\"body\":"), "{promo}");
        assert!(c.scene_revision() > rev, "promotion must bump the revision");

        // Now a real 3D body exists, so the badge is no longer legacy-planar and
        // the inspector reports a resolved, revolved body.
        assert_ne!(c.preview_mode(), "legacy_planar");
        let after = c.scene_objects_json();
        assert!(after.contains("\"construction\":\"revolved\""), "{after}");
        assert!(after.contains("\"is3d\":true"), "{after}");
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

    /// Build a **v1** singlet scene and write it to `web/public/demo-v1-lens.ocanvas`.
    /// Loaded through the 3D preview it migrates to a legacy-planar scene whose
    /// lens body is `PlanarOnly`, so the promotion controls have something real to
    /// promote. Ignored (fixture generator).
    #[test]
    #[ignore]
    fn write_demo_v1_lens_fixture() {
        let mut c = Canvas::new();
        c.add_collimated_source(-30.0, 0.0, 0.0, 10.0, 9, 0.5876);
        c.add_singlet(0.0, 0.0, 50.0, 4.0, 12.0, 1.5168, 64.17);
        c.add_detector(30.0, -8.0, 30.0, 8.0, 48);
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../web/public/demo-v1-lens.ocanvas"
        );
        std::fs::write(path, c.to_json()).expect("write v1 lens fixture");
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

    /// Pull the first integer out of a `"key":[a,b,...]` array in a report JSON.
    fn first_id(json: &str, key: &str) -> Option<u64> {
        let start = json.find(&format!("\"{key}\":["))? + key.len() + 4;
        let rest = &json[start..];
        let end = rest.find(']')?;
        rest[..end].split(',').next()?.trim().parse().ok()
    }
}
