//! `oc_core::Scene` → flat, f32, GPU-ready buffers.
//!
//! M2 makes the interactive tracer **spectral**: each element carries its
//! dispersion model (Cauchy for an Abbe point, Sellmeier for a catalog glass)
//! and each ray carries its wavelength, so `n(λ)` is evaluated in the tracer —
//! chromatic spread is a first-class result, not a post-process. Element
//! boundaries are baked into **world** coordinates at upload.
//!
//! These `#[repr(C)]` structs match the `std430` layout of the WGSL structs in
//! `shader.wgsl` field for field; keep them in lockstep. (The GPU-vs-CPU test
//! catches any drift loudly.)

use bytemuck::{Pod, Zeroable};
use oc_core::{Cauchy, ConicParams, Edge, Isometry2, Material, Scene, Vec2};

/// Primitive kinds understood by the interactive tracer (M1/M2 subset).
pub const KIND_LINE: u32 = 0;
pub const KIND_CONIC: u32 = 1;

/// Element material kinds.
pub const MAT_DIELECTRIC: u32 = 0;
pub const MAT_MIRROR: u32 = 1;
pub const MAT_ABSORBER: u32 = 2;

/// Dispersion models.
pub const DISP_CONST: u32 = 0;
pub const DISP_CAUCHY: u32 = 1;
pub const DISP_SELLMEIER: u32 = 2;

/// Segment flag bits.
pub const SEG_VALID: u32 = 1;
pub const SEG_TIR: u32 = 2;
pub const SEG_GHOST: u32 = 4;

/// A boundary primitive in world coordinates. `p` packs per-kind parameters:
/// - Line:  `[ax, ay, bx, by, _, _, _, _]`
/// - Conic: `[vx, vy, axis_x, axis_y, c, k, half_ap, _]`
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPrim {
    pub kind: u32,
    pub _pad: [u32; 3],
    pub p: [f32; 8],
}

/// Per-element metadata + dispersion. `edge_start`/`n_edges` index into prims.
///
/// Dispersion params `dp`:
/// - Cauchy:    `[a, b, 0, 0, 0, 0]`  → `n = a + b/λ²`
/// - Sellmeier: `[b0, b1, b2, c0, c1, c2]`
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuElem {
    pub kind: u32,
    pub disp: u32,
    pub priority: i32,
    pub edge_start: u32,
    pub n_edges: u32,
    pub reflectivity: f32,
    pub index_const: f32,
    pub _pad0: f32,
    pub dp: [f32; 6],
    pub _pad1: [f32; 2],
}

/// A launch ray (world coords, unit direction) with its wavelength.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRay {
    pub ox: f32,
    pub oy: f32,
    pub dx: f32,
    pub dy: f32,
    pub flux: f32,
    pub lambda: f32,
    pub _pad: [f32; 2],
}

/// An output ray segment. `flags`: bit0 valid, bit1 TIR, bit2 ghost.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct GpuSeg {
    pub ax: f32,
    pub ay: f32,
    pub bx: f32,
    pub by: f32,
    pub flux: f32,
    pub lambda: f32,
    pub flags: u32,
    pub _pad: u32,
}

/// Trace parameters (uniform buffer). Matches `Params` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParams {
    pub n_elems: u32,
    pub n_rays: u32,
    pub max_depth: u32,
    pub max_segs: u32,
    pub ambient: f32,
    pub bounds: f32,
    pub flux_eps: f32,
    pub ghosts: u32,
}

/// The full uploadable scene.
pub struct GpuScene {
    pub prims: Vec<GpuPrim>,
    pub elems: Vec<GpuElem>,
    pub rays: Vec<GpuRay>,
    pub ambient: f32,
    pub bounds: f32,
    /// Segment budget per launch ray (grows when ghosts multiply branches).
    pub max_segs: u32,
    pub ghosts: bool,
    pub flux_eps: f32,
}

fn line_prim(a: Vec2, b: Vec2) -> GpuPrim {
    GpuPrim {
        kind: KIND_LINE,
        _pad: [0; 3],
        p: [
            a.x as f32, a.y as f32, b.x as f32, b.y as f32, 0.0, 0.0, 0.0, 0.0,
        ],
    }
}

fn conic_prim(cp: &ConicParams, t: &Isometry2) -> GpuPrim {
    let v = t.transform_point(cp.vertex);
    let axis = t.rotate(cp.axis).normalize();
    GpuPrim {
        kind: KIND_CONIC,
        _pad: [0; 3],
        p: [
            v.x as f32,
            v.y as f32,
            axis.x as f32,
            axis.y as f32,
            cp.c as f32,
            cp.k as f32,
            cp.half_ap as f32,
            0.0,
        ],
    }
}

/// Options controlling scene upload.
#[derive(Debug, Clone, Copy)]
pub struct BuildOpts {
    pub ghosts: bool,
    /// Segment budget per launch ray.
    pub max_segs: u32,
    pub flux_eps: f32,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            ghosts: false,
            max_segs: 12,
            flux_eps: 1e-4,
        }
    }
}

impl GpuScene {
    /// Build with defaults (no ghosts). Kept for M1 call-sites/tests.
    pub fn build(scene: &Scene, _lambda_um: f64) -> Self {
        Self::build_with(scene, BuildOpts::default())
    }

    /// Build the uploadable scene. Rays for **all** spectral samples of every
    /// source are emitted, each carrying its own wavelength (spectral trace).
    pub fn build_with(scene: &Scene, opts: BuildOpts) -> Self {
        let mut prims = Vec::new();
        let mut elems = Vec::new();

        for e in &scene.elements {
            let edge_start = prims.len() as u32;
            for edge in &e.boundary {
                match edge {
                    Edge::Line { a, b } => prims.push(line_prim(
                        e.transform.transform_point(*a),
                        e.transform.transform_point(*b),
                    )),
                    Edge::Conic(cp) => prims.push(conic_prim(cp, &e.transform)),
                    Edge::Asphere { base, .. } => prims.push(conic_prim(base, &e.transform)),
                    Edge::Arc { center, r, a0, a1 } => {
                        let pa = *center + Vec2::new(a0.cos(), a0.sin()) * *r;
                        let pb = *center + Vec2::new(a1.cos(), a1.sin()) * *r;
                        prims.push(line_prim(
                            e.transform.transform_point(pa),
                            e.transform.transform_point(pb),
                        ));
                    }
                    Edge::Spline { ctrl, .. } => {
                        for w in ctrl.windows(2) {
                            prims.push(line_prim(
                                e.transform.transform_point(w[0]),
                                e.transform.transform_point(w[1]),
                            ));
                        }
                    }
                }
            }
            elems.push(element_meta(e, scene, edge_start, prims.len() as u32));
        }

        let mut rays = Vec::new();
        for s in &scene.sources {
            for er in s.emit() {
                rays.push(GpuRay {
                    ox: er.ray.origin.x as f32,
                    oy: er.ray.origin.y as f32,
                    dx: er.ray.dir.x as f32,
                    dy: er.ray.dir.y as f32,
                    flux: er.flux as f32,
                    lambda: er.lambda_um as f32,
                    _pad: [0.0; 2],
                });
            }
        }

        Self {
            prims,
            elems,
            rays,
            ambient: scene.env.ambient_index as f32,
            bounds: 1.0e4,
            max_segs: opts.max_segs.max(1),
            ghosts: opts.ghosts,
            flux_eps: opts.flux_eps,
        }
    }

    pub fn params(&self, max_depth: u32) -> GpuParams {
        GpuParams {
            n_elems: self.elems.len() as u32,
            n_rays: self.rays.len() as u32,
            max_depth,
            max_segs: self.max_segs,
            ambient: self.ambient,
            bounds: self.bounds,
            flux_eps: self.flux_eps,
            ghosts: self.ghosts as u32,
        }
    }
}

fn element_meta(e: &oc_core::Element, scene: &Scene, edge_start: u32, edge_end: u32) -> GpuElem {
    let n_edges = edge_end - edge_start;
    let mut g = GpuElem {
        kind: MAT_DIELECTRIC,
        disp: DISP_CONST,
        priority: e.priority,
        edge_start,
        n_edges,
        reflectivity: 0.0,
        index_const: 1.0,
        _pad0: 0.0,
        dp: [0.0; 6],
        _pad1: [0.0; 2],
    };
    match &e.material {
        Material::Mirror { reflectivity } => {
            g.kind = MAT_MIRROR;
            g.reflectivity = *reflectivity as f32;
        }
        Material::Absorber => {
            g.kind = MAT_ABSORBER;
        }
        Material::AbbePoint { nd, vd } => {
            let c = Cauchy::from_abbe(*nd, *vd);
            g.disp = DISP_CAUCHY;
            g.dp[0] = c.a as f32;
            g.dp[1] = c.b as f32;
        }
        Material::Catalog(id) => {
            if let Some(glass) = scene.catalog.get(*id) {
                g.disp = DISP_SELLMEIER;
                for i in 0..3 {
                    g.dp[i] = glass.sell.b[i] as f32;
                    g.dp[3 + i] = glass.sell.c[i] as f32;
                }
            } else {
                g.index_const = scene.env.ambient_index as f32;
            }
        }
        Material::Grin(f) => {
            g.index_const = f.base_index as f32;
        }
    }
    g
}
