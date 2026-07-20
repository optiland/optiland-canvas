//! The `repr(C)`/`std430` buffer layout for the 3D WebGPU preview (§12.2, PR
//! 3D-08).
//!
//! This is the wire format between the Rust compiler ([`crate::scene3`]) and the
//! WGSL shader (`shader3.wgsl`). It is deliberately dumb: every struct is a run of
//! 16-byte (`vec4`-sized) blocks, so `std430` alignment is trivial and there is no
//! `vec3`/`mat3` padding subtlety to get wrong between Rust and WGSL. A rotation
//! travels as its three columns in three `[f32; 4]`; the shader rebuilds the
//! `mat3x3` from them.
//!
//! Variable-length geometry (extruded polygons, revolved faces and contours, and
//! polygon face-shapes) lives in flat **arenas** — `verts` and `faces` — with each
//! object holding a `(start, count)` range into them. That is the one shape that
//! survives a GPU storage buffer: nested `Vec`s do not.
//!
//! `bytemuck::Pod`, so a `&[T]` uploads with `cast_slice`. Sizes are asserted at
//! compile time here and re-checked against the WGSL struct sizes at runtime in
//! `gpu3_vs_cpu3` — the two must never drift.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};

use crate::scene3::{
    DetShape, GpuAperture3, GpuBody3, GpuDetector3, GpuResponse, GpuScene3, GpuShape2, GpuSurface3,
    RevFace,
};

// Face-shape kinds (the finite region of a plane). Shared by surfaces, apertures,
// and detectors.
pub const SHAPE_RECT: u32 = 0;
pub const SHAPE_DISC: u32 = 1;
pub const SHAPE_ANNULUS: u32 = 2;
pub const SHAPE_POLYGON: u32 = 3;

// Surface response kinds.
pub const RESP_ABSORBER: u32 = 0;
pub const RESP_MIRROR: u32 = 1;
pub const RESP_DIELECTRIC: u32 = 2;
pub const RESP_EXPLICIT_RT: u32 = 3;

// Revolved-face kinds (Part 3).
pub const FACE_SAG: u32 = 0;
pub const FACE_FRUSTUM: u32 = 1;
pub const FACE_ARC: u32 = 2;

/// Trace parameters (uniform buffer). Mirrors `Params` in `shader3.wgsl`. A run of
/// scalars padded to a multiple of 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Params3 {
    pub n_bodies: u32,
    pub n_surfaces: u32,
    pub n_apertures: u32,
    pub n_detectors: u32,
    pub n_rays: u32,
    pub max_depth: u32,
    pub max_segs: u32,
    pub ghosts: u32,
    pub max_ghost: u32,
    pub ambient: f32,
    pub bounds: f32,
    pub flux_eps: f32,
    /// Scene scale in mm (the normalization divisor): converts a normalized path
    /// length back to millimetres for Beer–Lambert absorption.
    pub scale: f32,
    pub _pad: [u32; 3],
}

/// A launch ray in the normalized frame: `origin.xyz + flux`, `dir.xyz + lambda`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Ray3Buf {
    pub origin_flux: [f32; 4],
    pub dir_lambda: [f32; 4],
}

/// An output segment in the normalized frame. `a.xyz` start, `a.w` unused; `b.xyz`
/// end, `b.w` unused; plus flux, wavelength, and flags. Decoded back to world
/// millimetres on read-back.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct Seg3Buf {
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub flux: f32,
    pub lambda: f32,
    pub flags: u32,
    pub _pad: u32,
}

/// A compiled body. Analytic solids carry their parameters inline; extruded and
/// revolved bodies carry `(start, count)` ranges into the `verts`/`faces` arenas.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Body3Buf {
    /// Local(canonical)-to-world rotation, three columns (`.w` unused).
    pub rot0: [f32; 4],
    pub rot1: [f32; 4],
    pub rot2: [f32; 4],
    /// Normalized translation (`.w` unused).
    pub pos: [f32; 4],
    /// Sphere `(r,0,0)`; box half-extents; cylinder `(r,half_len,0)` (`.w` unused).
    pub params: [f32; 4],
    pub kind: u32,
    pub surface: u32,
    pub disp: u32,
    pub priority: i32,
    pub index_const: f32,
    pub alpha_per_mm: f32,
    /// Extruded profile: `verts[poly_start .. poly_start+poly_count]` and Z limits.
    pub poly_start: u32,
    pub poly_count: u32,
    /// Revolved profile: face range and containment-contour range.
    pub face_start: u32,
    pub face_count: u32,
    pub contour_start: u32,
    pub contour_count: u32,
    pub z_min: f32,
    pub z_max: f32,
    /// Sellmeier B/C or Cauchy a/b (6 used, 2 pad).
    pub dp: [f32; 8],
    /// Pad the struct to a 16-byte (std430 `vec4`) multiple so its array stride
    /// matches the WGSL side.
    pub _pad: [f32; 2],
}

/// A revolved-face record (`verts` arena is not used by these; params are inline).
/// `a`/`b`/`coeffs` are reinterpreted per `kind` — see `shader3.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Face3Buf {
    pub kind: u32,
    pub is_conic: u32,
    pub _p0: u32,
    pub _p1: u32,
    /// Sag: `(vertex_x, axis_sign, c, k)`; frustum: `(x0,y0,x1,y1)`; arc:
    /// `(center_x, r, a0, a1)`.
    pub a: [f32; 4],
    /// Sag: `(outer_r, 0, 0, 0)`.
    pub b: [f32; 4],
    /// Sag aspheric coefficients A4.. (unused by frustum/arc).
    pub coeffs: [f32; 8],
}

/// A zero-thickness surface: local `z=0` plane, a face shape, and a response per
/// side.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Surface3Buf {
    pub rot0: [f32; 4],
    pub rot1: [f32; 4],
    pub rot2: [f32; 4],
    pub pos: [f32; 4],
    pub shape_kind: u32,
    pub shape_a: f32,
    pub shape_b: f32,
    pub poly_start: u32,
    pub poly_count: u32,
    pub front_kind: u32,
    pub front_r: f32,
    pub front_t: f32,
    pub front_ar: f32,
    pub back_kind: u32,
    pub back_r: f32,
    pub back_t: f32,
    pub back_ar: f32,
    pub _pad: [f32; 3],
}

/// A finite aperture: an opening (`clear`) in an opaque mask (`outer`, optional),
/// in a placed plane.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Aperture3Buf {
    pub origin: [f32; 4],
    pub normal: [f32; 4],
    pub u: [f32; 4],
    pub v: [f32; 4],
    pub clear_kind: u32,
    pub clear_a: f32,
    pub clear_b: f32,
    pub clear_poly_start: u32,
    pub clear_poly_count: u32,
    /// `1` if an outer extent is present; `0` means the mask is unbounded.
    pub has_outer: u32,
    pub outer_kind: u32,
    pub outer_a: f32,
    pub outer_b: f32,
    pub outer_poly_start: u32,
    pub outer_poly_count: u32,
    pub _pad: u32,
}

/// An area detector: a placed face shape, with a terminate/probe flag.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Detector3Buf {
    pub rot0: [f32; 4],
    pub rot1: [f32; 4],
    pub rot2: [f32; 4],
    pub pos: [f32; 4],
    pub shape_kind: u32,
    pub shape_a: f32,
    pub shape_b: f32,
    pub terminate: u32,
}

// A vec2 arena element (`array<vec2<f32>>`, stride 8 in std430).
pub type Vert3Buf = [f32; 2];

// Compile-time layout guarantees. Every 16-byte multiple, so std430 places them
// with no implicit padding. If one of these fails, the WGSL struct that mirrors it
// must change in lockstep.
const _: () = {
    assert!(std::mem::size_of::<Params3>() == 64);
    assert!(std::mem::size_of::<Ray3Buf>() == 32);
    assert!(std::mem::size_of::<Seg3Buf>() == 48);
    assert!(std::mem::size_of::<Body3Buf>() == 176);
    assert!(std::mem::size_of::<Face3Buf>() == 80);
    assert!(std::mem::size_of::<Surface3Buf>() == 128);
    assert!(std::mem::size_of::<Aperture3Buf>() == 112);
    assert!(std::mem::size_of::<Detector3Buf>() == 80);
    assert!(std::mem::size_of::<Vert3Buf>() == 8);
    assert!(std::mem::align_of::<Body3Buf>() == 4);
};

/// The flat, uploadable form of a [`GpuScene3`]: packed struct arrays plus the two
/// shared arenas. This is what `gpu3.rs` writes into GPU buffers.
pub struct Gpu3Buffers {
    pub params: Params3,
    pub bodies: Vec<Body3Buf>,
    pub surfaces: Vec<Surface3Buf>,
    pub apertures: Vec<Aperture3Buf>,
    pub detectors: Vec<Detector3Buf>,
    pub rays: Vec<Ray3Buf>,
    pub verts: Vec<Vert3Buf>,
    pub faces: Vec<Face3Buf>,
}

/// Push a rotation's three columns as `[f32; 4]` blocks.
fn cols(rot: &Mat3) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let c = rot.to_cols_array();
    (
        [c[0], c[1], c[2], 0.0],
        [c[3], c[4], c[5], 0.0],
        [c[6], c[7], c[8], 0.0],
    )
}

fn v3(v: Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// Encode a face shape, appending any polygon vertices to the arena.
/// Returns `(kind, a, b, poly_start, poly_count)`.
fn encode_shape(shape: &GpuShape2, verts: &mut Vec<Vert3Buf>) -> (u32, f32, f32, u32, u32) {
    match shape {
        GpuShape2::Rectangle { hw, hh } => (SHAPE_RECT, *hw, *hh, 0, 0),
        GpuShape2::Disc { r } => (SHAPE_DISC, *r, 0.0, 0, 0),
        GpuShape2::Annulus { inner, outer } => (SHAPE_ANNULUS, *inner, *outer, 0, 0),
        GpuShape2::Polygon { verts: poly } => {
            let start = verts.len() as u32;
            verts.extend(poly.iter().map(|p| [p.x, p.y]));
            (SHAPE_POLYGON, 0.0, 0.0, start, poly.len() as u32)
        }
    }
}

fn encode_det_shape(shape: &DetShape) -> (u32, f32, f32) {
    match shape {
        DetShape::Rectangle { hw, hh } => (SHAPE_RECT, *hw, *hh),
        DetShape::Disc { r } => (SHAPE_DISC, *r, 0.0),
        DetShape::Annulus { inner, outer } => (SHAPE_ANNULUS, *inner, *outer),
    }
}

/// Encode a surface response as `(kind, r, t, ar)`.
fn encode_response(r: &GpuResponse) -> (u32, f32, f32, f32) {
    match r {
        GpuResponse::Absorber => (RESP_ABSORBER, 0.0, 0.0, 0.0),
        GpuResponse::Mirror { r } => (RESP_MIRROR, *r, 0.0, 0.0),
        GpuResponse::Dielectric { ar } => (RESP_DIELECTRIC, 0.0, 0.0, *ar),
        GpuResponse::ExplicitRt { r, t } => (RESP_EXPLICIT_RT, *r, *t, 0.0),
    }
}

fn pack_body(b: &GpuBody3, verts: &mut Vec<Vert3Buf>, faces: &mut Vec<Face3Buf>) -> Body3Buf {
    let (rot0, rot1, rot2) = cols(&b.rot);
    let (mut poly_start, mut poly_count) = (0u32, 0u32);
    let (mut z_min, mut z_max) = (0.0f32, 0.0f32);
    if let Some(e) = &b.extrude {
        poly_start = verts.len() as u32;
        verts.extend(e.verts.iter().map(|p| [p.x, p.y]));
        poly_count = e.verts.len() as u32;
        z_min = e.z_min;
        z_max = e.z_max;
    }
    let (mut face_start, mut face_count) = (0u32, 0u32);
    let (mut contour_start, mut contour_count) = (0u32, 0u32);
    if let Some(rev) = &b.revolve {
        face_start = faces.len() as u32;
        faces.extend(rev.faces.iter().map(pack_face));
        face_count = rev.faces.len() as u32;
        contour_start = verts.len() as u32;
        verts.extend(rev.contour.iter().map(|p| [p.x, p.y]));
        contour_count = rev.contour.len() as u32;
    }
    Body3Buf {
        rot0,
        rot1,
        rot2,
        pos: v3(b.pos),
        params: v3(b.params),
        kind: b.kind,
        surface: b.surface,
        disp: b.disp,
        priority: b.priority,
        index_const: b.index_const,
        alpha_per_mm: b.alpha_per_mm,
        poly_start,
        poly_count,
        face_start,
        face_count,
        contour_start,
        contour_count,
        z_min,
        z_max,
        dp: [
            b.dp[0], b.dp[1], b.dp[2], b.dp[3], b.dp[4], b.dp[5], 0.0, 0.0,
        ],
        _pad: [0.0; 2],
    }
}

fn pack_face(f: &RevFace) -> Face3Buf {
    match f {
        RevFace::Sag {
            vertex_x,
            axis_sign,
            c,
            k,
            coeffs,
            outer_r,
            is_conic,
        } => Face3Buf {
            kind: FACE_SAG,
            is_conic: *is_conic as u32,
            _p0: 0,
            _p1: 0,
            a: [*vertex_x, *axis_sign, *c, *k],
            b: [*outer_r, 0.0, 0.0, 0.0],
            coeffs: *coeffs,
        },
        RevFace::Frustum { x0, y0, x1, y1 } => Face3Buf {
            kind: FACE_FRUSTUM,
            is_conic: 0,
            _p0: 0,
            _p1: 0,
            a: [*x0, *y0, *x1, *y1],
            b: [0.0; 4],
            coeffs: [0.0; 8],
        },
        RevFace::Arc {
            center_x,
            r,
            a0,
            a1,
        } => Face3Buf {
            kind: FACE_ARC,
            is_conic: 0,
            _p0: 0,
            _p1: 0,
            a: [*center_x, *r, *a0, *a1],
            b: [0.0; 4],
            coeffs: [0.0; 8],
        },
    }
}

fn pack_surface(s: &GpuSurface3, verts: &mut Vec<Vert3Buf>) -> Surface3Buf {
    let (rot0, rot1, rot2) = cols(&s.rot);
    let (shape_kind, shape_a, shape_b, poly_start, poly_count) = encode_shape(&s.shape, verts);
    let (front_kind, front_r, front_t, front_ar) = encode_response(&s.front);
    let (back_kind, back_r, back_t, back_ar) = encode_response(&s.back);
    Surface3Buf {
        rot0,
        rot1,
        rot2,
        pos: v3(s.pos),
        shape_kind,
        shape_a,
        shape_b,
        poly_start,
        poly_count,
        front_kind,
        front_r,
        front_t,
        front_ar,
        back_kind,
        back_r,
        back_t,
        back_ar,
        _pad: [0.0; 3],
    }
}

fn pack_aperture(a: &GpuAperture3, verts: &mut Vec<Vert3Buf>) -> Aperture3Buf {
    let (clear_kind, clear_a, clear_b, clear_poly_start, clear_poly_count) =
        encode_shape(&a.clear, verts);
    let (has_outer, outer_kind, outer_a, outer_b, outer_poly_start, outer_poly_count) =
        match &a.outer {
            None => (0, 0, 0.0, 0.0, 0, 0),
            Some(s) => {
                let (k, oa, ob, ps, pc) = encode_shape(s, verts);
                (1, k, oa, ob, ps, pc)
            }
        };
    Aperture3Buf {
        origin: v3(a.origin),
        normal: v3(a.normal),
        u: v3(a.u),
        v: v3(a.v),
        clear_kind,
        clear_a,
        clear_b,
        clear_poly_start,
        clear_poly_count,
        has_outer,
        outer_kind,
        outer_a,
        outer_b,
        outer_poly_start,
        outer_poly_count,
        _pad: 0,
    }
}

fn pack_detector(d: &GpuDetector3) -> Detector3Buf {
    let (rot0, rot1, rot2) = cols(&d.rot);
    let (shape_kind, shape_a, shape_b) = encode_det_shape(&d.shape);
    Detector3Buf {
        rot0,
        rot1,
        rot2,
        pos: v3(d.pos),
        shape_kind,
        shape_a,
        shape_b,
        terminate: d.terminate as u32,
    }
}

fn pack_ray(r: &crate::scene3::GpuRay3) -> Ray3Buf {
    Ray3Buf {
        origin_flux: [r.origin.x, r.origin.y, r.origin.z, r.flux],
        dir_lambda: [r.dir.x, r.dir.y, r.dir.z, r.lambda_um],
    }
}

impl Gpu3Buffers {
    /// Flatten a compiled scene into upload buffers. `max_segs` is the per-ray
    /// segment budget; `max_depth` the interaction cap.
    pub fn from_scene(scene: &GpuScene3, max_depth: u32, max_segs: u32) -> Self {
        let mut verts: Vec<Vert3Buf> = Vec::new();
        let mut faces: Vec<Face3Buf> = Vec::new();

        let bodies = scene
            .bodies
            .iter()
            .map(|b| pack_body(b, &mut verts, &mut faces))
            .collect();
        let surfaces = scene
            .surfaces
            .iter()
            .map(|s| pack_surface(s, &mut verts))
            .collect();
        let apertures = scene
            .apertures
            .iter()
            .map(|a| pack_aperture(a, &mut verts))
            .collect();
        let detectors = scene.detectors.iter().map(pack_detector).collect();
        let rays = scene.rays.iter().map(pack_ray).collect();

        let params = Params3 {
            n_bodies: scene.bodies.len() as u32,
            n_surfaces: scene.surfaces.len() as u32,
            n_apertures: scene.apertures.len() as u32,
            n_detectors: scene.detectors.len() as u32,
            n_rays: scene.rays.len() as u32,
            max_depth,
            max_segs,
            ghosts: scene.ghosts as u32,
            max_ghost: scene.max_ghost,
            ambient: scene.ambient,
            bounds: scene.bounds_norm,
            flux_eps: scene.flux_eps,
            scale: scene.norm.scale as f32,
            _pad: [0; 3],
        };

        Self {
            params,
            bodies,
            surfaces,
            apertures,
            detectors,
            rays,
            verts,
            faces,
        }
    }
}
