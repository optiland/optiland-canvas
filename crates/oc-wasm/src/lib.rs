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
    Isometry2, Material, Scene, Source, SourceKind, Spectrum, Vec2,
};
use oc_fit::{fit_stroke, sample_edge, surface_rim, Fit, FitLadder, DEFAULT_TOL_UM};
use oc_gpu::{color::wavelength_um_to_linear, scene::GpuScene, trace_cpu, BuildOpts};
use oc_trace::{trace as ref_trace, TraceConfig};
use wasm_bindgen::prelude::*;

/// State kept for an element drawn freehand: its fitted ladder (§5.3) so the
/// complexity slider can re-swap the front surface with no refit, plus the
/// extrusion depth used to close the body.
struct FittedSurface {
    ladder: FitLadder,
    rung: usize,
    thickness: f64,
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
    max_depth: u32,
    ghosts: bool,
    /// Freehand-fitted elements, keyed by element id (§5.3).
    fitted: HashMap<u32, FittedSurface>,
    /// Pinned `(element_id, param_code)` pairs — locked against edits (§5.2).
    param_pins: HashSet<(u32, u8)>,
}

#[wasm_bindgen]
impl Canvas {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Canvas {
        Canvas {
            scene: Scene::new(),
            max_depth: 10,
            ghosts: false,
            fitted: HashMap::new(),
            param_pins: HashSet::new(),
        }
    }

    // --- configuration -----------------------------------------------------

    #[wasm_bindgen(js_name = setMaxDepth)]
    pub fn set_max_depth(&mut self, depth: u32) {
        self.max_depth = depth.clamp(1, 32);
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
        self.scene.add_element(e).0
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
        self.scene.add_element(e).0
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
        let opts = BuildOpts {
            ghosts: self.ghosts,
            max_segs: if self.ghosts { 24 } else { 12 },
            flux_eps: 1e-4,
        };
        let gs = GpuScene::build_with(&self.scene, opts);
        let per_ray = trace_cpu(&gs, self.max_depth);
        let mut out = Vec::new();
        for segs in &per_ray {
            for s in segs {
                // Linear light — the caller accumulates additively then tonemaps.
                let [r, g, b] = wavelength_um_to_linear(s.lambda_um);
                out.extend_from_slice(&[
                    s.a.x,
                    s.a.y,
                    s.b.x,
                    s.b.y,
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

    /// Legacy M1 trace: flat `[ax, ay, bx, by, flux]` per segment.
    pub fn trace(&self) -> Vec<f32> {
        let gs = GpuScene::build_with(&self.scene, BuildOpts::default());
        let per_ray = trace_cpu(&gs, self.max_depth);
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
        let cfg = TraceConfig {
            ghosts: self.ghosts,
            ..TraceConfig::default()
        };
        let res = ref_trace(&self.scene, &cfg);
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
                format!(
                    "{},\"curve\":[{}]}}",
                    rung_json_open(f),
                    flat.join(",")
                )
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
        let id = self.scene.add_element(e).0;
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
        let (px, py) = (self.param_pins.contains(&(id, 0)), self.param_pins.contains(&(id, 1)));
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
        let jx = if guide_x.is_finite() { format!("{guide_x}") } else { "null".into() };
        let jy = if guide_y.is_finite() { format!("{guide_y}") } else { "null".into() };
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
            "x", "y", "rot", "thickness", "fit", "radius1", "radius2", "half_ap1", "half_ap2",
            "conic_k1", "conic_k2",
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
        let remap = |k: u32| if k > id { Some(k - 1) } else if k < id { Some(k) } else { None };
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
                    "{{\"id\":{},\"x\":{},\"y\":{},\"rot_deg\":{},\"locked\":{},\"kind\":\"{}\"}}",
                    i,
                    e.transform.pos.x,
                    e.transform.pos.y,
                    e.transform.rot.to_degrees(),
                    e.locked,
                    self.element_kind(i as u32, e)
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

impl Canvas {
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
    let radius = |c: f64| if c.abs() < 1e-12 { f64::INFINITY } else { 1.0 / c };
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
