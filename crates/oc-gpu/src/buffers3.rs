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
//! Variable-length data (extruded polygons, revolved faces and contours, polygon
//! face-shapes, and tabulated response curves) lives in flat **arenas** — `verts`
//! and `faces` — with each object holding a `(start, count)` range into them.
//! That is the one shape that survives a GPU storage buffer: nested `Vec`s do
//! not. A response table is stored in `verts` as one `(lambda_um, value)` pair
//! per sample.
//!
//! `bytemuck::Pod`, so a `&[T]` uploads with `cast_slice`. Sizes are asserted at
//! compile time here, and the `shader3_layout` test parses the WGSL with naga
//! and checks every struct's size and field offsets against these, so the two
//! cannot drift.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};
use oc_core::scene3::OutOfRangePolicy;

use crate::scene3::{
    DetShape, GpuAperture3, GpuBody3, GpuDetector3, GpuResponse, GpuScene3, GpuShape2, GpuSurface3,
    RevFace, Spectral32,
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
pub const RESP_THIN_LENS: u32 = 4;
/// A baked thin-film coating: `r.tab_start` points at
/// `COAT_ANGLES × COAT_LAMBDAS` `(R, T)` pairs in `verts`, row-major by angle.
pub const RESP_COATING: u32 = 5;
/// A diffraction grating: `r` its working order's efficiency, `t` the zero
/// order's, and `ar.value` the order times the groove density per µm.
pub const RESP_GRATING: u32 = 6;
/// `Response3Buf::flags`: the reflection is stray light, an AR coat's
/// residual, and so a ghost rather than a designed path. Must match the
/// shader's `RESP_FLAG_STRAY`.
pub const RESP_FLAG_STRAY: u32 = 1;
/// `Response3Buf::flags`: a transmission grating. Must match the shader's
/// `RESP_FLAG_TRANSMIT`.
pub const RESP_FLAG_TRANSMIT: u32 = 2;

// Out-of-range policies of a tabulated response scalar.
pub const POLICY_ERROR: u32 = 0;
pub const POLICY_CLAMP: u32 = 1;
pub const POLICY_ZERO: u32 = 2;

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
    /// 1 when `resp` carries this face's own response instead of the body's.
    ///
    /// A coated plate is one body whose two faces do different things, so the
    /// response has to travel with the face rather than with the body.
    pub has_resp: u32,
    pub _p1: u32,
    /// Sag: `(vertex_x, axis_sign, c, k)`; frustum: `(x0,y0,x1,y1)`; arc:
    /// `(center_x, r, a0, a1)`.
    pub a: [f32; 4],
    /// Sag: `(outer_r, inner_r, 0, 0)`.
    pub b: [f32; 4],
    /// Sag aspheric coefficients A4.. (unused by frustum/arc).
    pub coeffs: [f32; 8],
    /// This face's own response, when `has_resp` is 1.
    pub resp: Response3Buf,
}

/// A response scalar: the constant `value` when `tab_count == 0`, else
/// `tab_count` `(lambda_um, value)` samples at `verts[tab_start..]`, read with
/// the out-of-range `policy` (`POLICY_*`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct Spectral3Buf {
    pub value: f32,
    pub tab_start: u32,
    pub tab_count: u32,
    pub policy: u32,
}

/// One side of a zero-thickness surface. `r` is the mirror reflectivity or the
/// explicit reflectance (a thin lens carries its normalized focal length there,
/// always constant), `t` the explicit transmittance, `ar` the dielectric AR
/// residual. Scalars a kind does not use are zero constants.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct Response3Buf {
    pub kind: u32,
    /// `RESP_FLAG_*` bits. One so far: the reflection is stray light.
    pub flags: u32,
    pub _pad: [u32; 2],
    pub r: Spectral3Buf,
    pub t: Spectral3Buf,
    pub ar: Spectral3Buf,
}

/// A zero-thickness surface: local `z=0` plane, a face shape, and a response per
/// side.
///
/// Byte offsets: `rot0..pos` 0-63, `shape_kind` 64, `shape_a` 68, `shape_b` 72,
/// `poly_start` 76, `poly_count` 80, pad 84-95, `front` 96-159, `back` 160-223.
/// Within a side: `kind` +0, pad +4, `r` +16, `t` +32, `ar` +48, each scalar
/// `value` +0, `tab_start` +4, `tab_count` +8, `policy` +12.
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
    pub _pad: [u32; 3],
    pub front: Response3Buf,
    pub back: Response3Buf,
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
    assert!(std::mem::size_of::<Face3Buf>() == 144);
    assert!(std::mem::size_of::<Spectral3Buf>() == 16);
    assert!(std::mem::size_of::<Response3Buf>() == 64);
    assert!(std::mem::size_of::<Surface3Buf>() == 224);
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

fn constant(value: f32) -> Spectral3Buf {
    Spectral3Buf {
        value,
        tab_start: 0,
        tab_count: 0,
        policy: POLICY_ERROR,
    }
}

/// Encode a response scalar, appending a table's samples to the arena.
fn encode_spectral(s: &Spectral32, verts: &mut Vec<Vert3Buf>) -> Spectral3Buf {
    match s {
        Spectral32::Const(v) => constant(*v),
        Spectral32::Table {
            lambda,
            values,
            policy,
        } => {
            let tab_start = verts.len() as u32;
            verts.extend(lambda.iter().zip(values).map(|(l, v)| [*l, *v]));
            Spectral3Buf {
                value: 0.0,
                tab_start,
                tab_count: lambda.len() as u32,
                policy: match policy {
                    OutOfRangePolicy::Error => POLICY_ERROR,
                    OutOfRangePolicy::Clamp => POLICY_CLAMP,
                    OutOfRangePolicy::Zero => POLICY_ZERO,
                },
            }
        }
    }
}

thread_local! {
    /// The coating grids already written to `verts` during one pack, by the
    /// address of their shared grid. Cleared at the start of each pack.
    static GRIDS: std::cell::RefCell<Vec<(usize, Response3Buf)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Encode one side's response. A thin lens carries its normalized focal length
/// in the `r` slot.
fn encode_response(resp: &GpuResponse, verts: &mut Vec<Vert3Buf>) -> Response3Buf {
    let zero = constant(0.0);
    let flags = match resp {
        GpuResponse::CoatingGrid { stray: true, .. } => RESP_FLAG_STRAY,
        GpuResponse::Grating { transmit: true, .. } => RESP_FLAG_TRANSMIT,
        _ => 0,
    };
    let (kind, r, t, ar) = match resp {
        GpuResponse::Absorber => (RESP_ABSORBER, zero, zero, zero),
        GpuResponse::Mirror { r } => (RESP_MIRROR, encode_spectral(r, verts), zero, zero),
        GpuResponse::Dielectric { ar } => (RESP_DIELECTRIC, zero, zero, encode_spectral(ar, verts)),
        GpuResponse::ExplicitRt { r, t } => {
            let r = encode_spectral(r, verts);
            (RESP_EXPLICIT_RT, r, encode_spectral(t, verts), zero)
        }
        GpuResponse::ThinLens { f } => (RESP_THIN_LENS, constant(*f), zero, zero),
        GpuResponse::Grating { k, eta, zero: z0, .. } => {
            let r = encode_spectral(eta, verts);
            (RESP_GRATING, r, encode_spectral(z0, verts), constant(*k))
        }
        GpuResponse::CoatingGrid { lambdas, rt, rt_inside, .. } => {
            // Every half lives in `verts`, which is already a list of pairs:
            // the wavelength axis in the `t` slot, the grid itself in `r`, and
            // the grid for a ray from inside the body, when it has its own, in
            // `ar` (`tab_count` 0 when the outside grid serves both sides). A
            // grid is `COAT_ANGLES` rows of `t.tab_count` (R, T) pairs. Faces
            // that share a bake share its place in `verts`: a Fresnel lens's
            // grooves all point at one grid.
            let key = std::sync::Arc::as_ptr(rt) as usize;
            if let Some(&(_, cached)) = GRIDS.with(|g| g.borrow().iter().find(|(k, _)| *k == key).copied()).as_ref() {
                return Response3Buf { kind: RESP_COATING, flags, _pad: [0; 2], ..cached };
            }
            let lam_start = verts.len() as u32;
            verts.extend(lambdas.iter().map(|l| [*l, 0.0]));
            let axis = Spectral3Buf {
                value: 0.0,
                tab_start: lam_start,
                tab_count: lambdas.len() as u32,
                policy: POLICY_CLAMP,
            };
            let grid_start = verts.len() as u32;
            verts.extend(rt.iter().copied());
            let grid = Spectral3Buf {
                value: 0.0,
                tab_start: grid_start,
                tab_count: rt.len() as u32,
                policy: POLICY_CLAMP,
            };
            let inside = match rt_inside {
                Some(g) => {
                    let start = verts.len() as u32;
                    verts.extend(g.iter().copied());
                    Spectral3Buf { value: 0.0, tab_start: start, tab_count: g.len() as u32, policy: POLICY_CLAMP }
                }
                None => zero,
            };
            let packed = Response3Buf { kind: RESP_COATING, flags, _pad: [0; 2], r: grid, t: axis, ar: inside };
            GRIDS.with(|g| g.borrow_mut().push((key, packed)));
            (RESP_COATING, grid, axis, inside)
        }
    };
    Response3Buf {
        kind,
        flags,
        _pad: [0; 2],
        r,
        t,
        ar,
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
        // A face's own response travels with the face, because a coated plate is
        // one body whose two faces do different things.
        for (i, f) in rev.faces.iter().enumerate() {
            let mut packed = pack_face(f);
            if let Some(Some(resp)) = b.face_resp.get(i) {
                packed.has_resp = 1;
                packed.resp = encode_response(resp, verts);
            }
            faces.push(packed);
        }
        face_count = rev.faces.len() as u32;
        contour_start = verts.len() as u32;
        verts.extend(rev.contour.iter().map(|p| [p.x, p.y]));
        contour_count = rev.contour.len() as u32;
        // A many-edged body's contour index rides in the fields a revolved body
        // does not otherwise use: its tables at `verts[poly_start..]`, bucket
        // count in `poly_count`, and its `x` slab in `z_min`/`z_max`. Integers
        // go in as exact f32 values. See `indexed_poly_contains` in the shader.
        if let Some(ix) = &rev.index {
            poly_start = verts.len() as u32;
            poly_count = ix.buckets.len() as u32;
            verts.push([ix.y0, ix.inv_h]);
            verts.extend(ix.buckets.iter().map(|&(s, c)| [s as f32, c as f32]));
            verts.extend(ix.items.iter().map(|&i| [i as f32, 0.0]));
            z_min = ix.x_lo;
            z_max = ix.x_hi;
        }
    } else if !b.face_resp.is_empty() {
        // An analytic body with per-face responses (a tube): its faces need no
        // geometry here, only a slot for the response, indexed by the face
        // number its own hit routine reports.
        face_start = faces.len() as u32;
        for resp in &b.face_resp {
            let mut packed = Face3Buf::zeroed();
            if let Some(resp) = resp {
                packed.has_resp = 1;
                packed.resp = encode_response(resp, verts);
            }
            faces.push(packed);
        }
        face_count = b.face_resp.len() as u32;
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

/// A zeroed response, for a face that carries none of its own.
fn zero_response() -> Response3Buf {
    Response3Buf {
        kind: RESP_ABSORBER,
        flags: 0,
        _pad: [0; 2],
        r: constant(0.0),
        t: constant(0.0),
        ar: constant(0.0),
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
            inner_r,
            outer_r,
            is_conic,
        } => Face3Buf {
            kind: FACE_SAG,
            is_conic: *is_conic as u32,
            has_resp: 0,
            resp: zero_response(),
            _p1: 0,
            a: [*vertex_x, *axis_sign, *c, *k],
            b: [*outer_r, *inner_r, 0.0, 0.0],
            coeffs: *coeffs,
        },
        RevFace::Frustum { x0, y0, x1, y1 } => Face3Buf {
            kind: FACE_FRUSTUM,
            is_conic: 0,
            has_resp: 0,
            resp: zero_response(),
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
            has_resp: 0,
            resp: zero_response(),
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
    let front = encode_response(&s.front, verts);
    let back = encode_response(&s.back, verts);
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
        _pad: [0; 3],
        front,
        back,
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
        // A grid's address names it only within this pack.
        GRIDS.with(|g| g.borrow_mut().clear());
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
        GRIDS.with(|g| g.borrow_mut().clear());

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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn response_tables_land_in_the_verts_arena() {
        let surface = GpuSurface3 {
            rot: Mat3::IDENTITY,
            pos: Vec3::ZERO,
            shape: GpuShape2::Polygon {
                verts: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
            },
            front: GpuResponse::ExplicitRt {
                r: Spectral32::Table {
                    lambda: vec![0.4, 0.55, 0.7],
                    values: vec![1.0, 0.5, 0.0],
                    policy: OutOfRangePolicy::Clamp,
                },
                t: Spectral32::Const(0.25),
            },
            back: GpuResponse::Mirror {
                r: Spectral32::Table {
                    lambda: vec![0.5, 0.6],
                    values: vec![0.9, 0.8],
                    policy: OutOfRangePolicy::Zero,
                },
            },
            object: 7,
        };
        // Something is already in the arena: ranges are absolute.
        let mut verts: Vec<Vert3Buf> = vec![[9.0, 9.0]];
        let b = pack_surface(&surface, &mut verts);

        assert_eq!((b.poly_start, b.poly_count), (1, 3));
        assert_eq!(b.front.kind, RESP_EXPLICIT_RT);
        let front_r = Spectral3Buf {
            value: 0.0,
            tab_start: 4,
            tab_count: 3,
            policy: POLICY_CLAMP,
        };
        assert_eq!(b.front.r, front_r);
        assert_eq!(b.front.t, constant(0.25));
        assert_eq!(b.front.ar, constant(0.0));
        assert_eq!(b.back.kind, RESP_MIRROR);
        let back_r = Spectral3Buf {
            value: 0.0,
            tab_start: 7,
            tab_count: 2,
            policy: POLICY_ZERO,
        };
        assert_eq!(b.back.r, back_r);
        assert_eq!(
            verts[4..],
            [[0.4, 1.0], [0.55, 0.5], [0.7, 0.0], [0.5, 0.9], [0.6, 0.8]]
        );

        // The documented byte offsets.
        let bytes = bytemuck::bytes_of(&b);
        let u = |at: usize| u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
        let f = |at: usize| f32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(u(80), 3); // poly_count
        assert_eq!(u(96), RESP_EXPLICIT_RT); // front.kind
        assert_eq!((u(116), u(120), u(124)), (4, 3, POLICY_CLAMP)); // front.r
        assert_eq!(f(128), 0.25); // front.t.value
        assert_eq!(u(160), RESP_MIRROR); // back.kind
        assert_eq!((u(180), u(184), u(188)), (7, 2, POLICY_ZERO)); // back.r
    }

    #[test]
    fn a_thin_lens_keeps_its_focal_length_in_the_r_slot() {
        let mut verts = Vec::new();
        let b = encode_response(&GpuResponse::ThinLens { f: -0.75 }, &mut verts);
        assert_eq!(b.kind, RESP_THIN_LENS);
        assert_eq!(b.r, constant(-0.75));
        assert!(verts.is_empty());
    }
}
