//! The 3D shader checked without a device (PR 3D-08, §12.2).
//!
//! `gpu3_vs_cpu3` needs a working adapter, and a machine without one (or with a
//! backend that rejects the pipeline) never compiles `shader3.wgsl` at all. This
//! suite runs everywhere: naga, the WGSL front end wgpu uses, must parse and
//! validate the shader, and every struct the host uploads must have the size and
//! field offsets of its `buffers3` twin. A layout drift between Rust and WGSL
//! fails here by name instead of as a scrambled trace.
#![cfg(feature = "gpu3")]

use naga::valid::{Capabilities, ValidationFlags, Validator};
use oc_gpu::buffers3::{
    Aperture3Buf, Body3Buf, Detector3Buf, Face3Buf, Params3, Ray3Buf, Response3Buf, Seg3Buf,
    Spectral3Buf, Surface3Buf, Vert3Buf,
};

const SRC: &str = include_str!("../src/shader3.wgsl");

fn module() -> naga::Module {
    naga::front::wgsl::parse_str(SRC).unwrap_or_else(|e| panic!("{}", e.emit_to_string(SRC)))
}

/// A Rust struct's size and the byte offsets of the named fields.
macro_rules! rust_layout {
    ($t:ty { $($f:ident),* $(,)? }) => {{
        let v: $t = bytemuck::Zeroable::zeroed();
        let base = &v as *const $t as usize;
        let fields: Vec<(&str, usize)> = vec![
            $((stringify!($f), std::ptr::addr_of!(v.$f) as usize - base)),*
        ];
        (std::mem::size_of::<$t>(), fields)
    }};
}

/// Compare a WGSL struct's size and member offsets to its Rust twin. Every
/// listed Rust field must exist in WGSL under the same name.
fn assert_struct(m: &naga::Module, wgsl: &str, (size, fields): (usize, Vec<(&str, usize)>)) {
    let (members, span) = m
        .types
        .iter()
        .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(name), naga::TypeInner::Struct { members, span }) if name == wgsl => {
                Some((members, *span))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no WGSL struct `{wgsl}`"));
    assert_eq!(
        span as usize, size,
        "{wgsl}: WGSL size {span} vs Rust {size}"
    );
    for (name, offset) in fields {
        let member = members
            .iter()
            .find(|mb| mb.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("{wgsl}.{name} is missing in WGSL"));
        assert_eq!(
            member.offset as usize, offset,
            "{wgsl}.{name}: WGSL offset {} vs Rust {offset}",
            member.offset
        );
    }
}

/// The stride of a storage-buffer array global.
fn global_stride(m: &naga::Module, name: &str) -> usize {
    let g = m
        .global_variables
        .iter()
        .find(|(_, g)| g.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no WGSL global `{name}`"))
        .1;
    match m.types[g.ty].inner {
        naga::TypeInner::Array { stride, .. } => stride as usize,
        ref other => panic!("`{name}` is not an array: {other:?}"),
    }
}

#[test]
fn shader3_parses_and_validates() {
    let m = module();
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&m)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(SRC)));
}

#[test]
fn shader3_structs_match_buffers3() {
    let m = module();
    assert_struct(
        &m,
        "Params",
        rust_layout!(Params3 {
            n_bodies,
            n_surfaces,
            n_apertures,
            n_detectors,
            n_rays,
            max_depth,
            max_segs,
            ghosts,
            max_ghost,
            ambient,
            bounds,
            flux_eps,
            scale,
        }),
    );
    assert_struct(
        &m,
        "Body",
        rust_layout!(Body3Buf {
            rot0,
            rot1,
            rot2,
            pos,
            params,
            kind,
            surface,
            disp,
            priority,
            index_const,
            alpha_per_mm,
            poly_start,
            poly_count,
            face_start,
            face_count,
            contour_start,
            contour_count,
            z_min,
            z_max,
            dp,
            _pad,
        }),
    );
    assert_struct(
        &m,
        "RFace",
        rust_layout!(Face3Buf {
            kind,
            is_conic,
            has_resp,
            _p1,
            a,
            b,
            coeffs,
            resp,
        }),
    );
    assert_struct(
        &m,
        "Spectral",
        rust_layout!(Spectral3Buf {
            value,
            tab_start,
            tab_count,
            policy,
        }),
    );
    assert_struct(
        &m,
        "Response",
        rust_layout!(Response3Buf {
            kind,
            flags,
            _pad,
            r,
            t,
            ar,
        }),
    );
    assert_struct(
        &m,
        "Surface",
        rust_layout!(Surface3Buf {
            rot0,
            rot1,
            rot2,
            pos,
            shape_kind,
            shape_a,
            shape_b,
            poly_start,
            poly_count,
            _pad,
            front,
            back,
        }),
    );
    assert_struct(
        &m,
        "Aperture",
        rust_layout!(Aperture3Buf {
            origin,
            normal,
            u,
            v,
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
            _pad,
        }),
    );
    assert_struct(
        &m,
        "Detector",
        rust_layout!(Detector3Buf {
            rot0,
            rot1,
            rot2,
            pos,
            shape_kind,
            shape_a,
            shape_b,
            flags,
        }),
    );
    assert_struct(
        &m,
        "Seg",
        rust_layout!(Seg3Buf {
            a,
            b,
            flux,
            lambda,
            flags,
            _pad,
        }),
    );

    // The arrays the host fills element by element.
    assert_eq!(global_stride(&m, "bodies"), std::mem::size_of::<Body3Buf>());
    assert_eq!(
        global_stride(&m, "surfaces"),
        std::mem::size_of::<Surface3Buf>()
    );
    assert_eq!(
        global_stride(&m, "apertures"),
        std::mem::size_of::<Aperture3Buf>()
    );
    assert_eq!(
        global_stride(&m, "detectors"),
        std::mem::size_of::<Detector3Buf>()
    );
    assert_eq!(global_stride(&m, "faces"), std::mem::size_of::<Face3Buf>());
    assert_eq!(global_stride(&m, "segs"), std::mem::size_of::<Seg3Buf>());
    assert_eq!(global_stride(&m, "verts"), std::mem::size_of::<Vert3Buf>());
    // A launch ray travels as two vec4 (`[origin_flux, dir_lambda]`).
    assert_eq!(
        2 * global_stride(&m, "rays"),
        std::mem::size_of::<Ray3Buf>()
    );
}

/// The baked coating grid is addressed by arithmetic, not by a search: the host
/// writes `COAT_ANGLES` rows and the shader indexes straight into them. If the
/// two sides disagree about the shape of that grid, every coated surface reads
/// someone else's numbers and the trace is quietly wrong rather than loudly
/// broken. So the row count is pinned here, by value, on both sides.
#[test]
fn the_coating_grid_constants_agree() {
    let at = SRC
        .find("const COAT_ANGLES:")
        .expect("COAT_ANGLES is missing from the shader");
    let line = &SRC[at..SRC[at..].find('\n').map(|e| at + e).unwrap_or(SRC.len())];
    let rhs = line.split('=').nth(1).expect("a value").trim().trim_end_matches(';');
    assert_eq!(rhs.trim(), format!("{}u", oc_gpu::scene3::COAT_ANGLES));
    assert_eq!(oc_gpu::buffers3::RESP_COATING, 5);
    // The last baked angle, which both sides index the grid by.
    let at = SRC.find("const COAT_THETA_MAX:").expect("COAT_THETA_MAX is missing from the shader");
    let line = &SRC[at..SRC[at..].find('\n').map(|e| at + e).unwrap_or(SRC.len())];
    let rhs: f32 = line.split('=').nth(1).expect("a value").trim().trim_end_matches(';').parse().unwrap();
    assert_eq!(rhs, oc_gpu::scene3::COAT_THETA_MAX);
    assert!((f64::from(rhs) - 0.999f64.sqrt().asin()).abs() < 1e-6);
}

/// The value of a `const NAME: u32 = Nu;` line in the shader.
fn shader_u32(name: &str) -> u32 {
    let key = format!("const {name}:");
    let at = SRC.find(&key).unwrap_or_else(|| panic!("{name} is missing from the shader"));
    let line = &SRC[at..SRC[at..].find('\n').map(|e| at + e).unwrap_or(SRC.len())];
    let rhs = line.split('=').nth(1).expect("a value").trim().trim_end_matches(';');
    rhs.trim_end_matches('u').parse().unwrap_or_else(|_| panic!("{name}: {rhs}"))
}

/// The host tags each surface side with a response kind and the shader
/// branches on it, so a kind numbered differently on the two sides traces as
/// some other response. Each one is pinned by value.
#[test]
fn the_response_kinds_agree() {
    use oc_gpu::buffers3 as b;
    for (name, host) in [
        ("RESP_ABSORBER", b::RESP_ABSORBER),
        ("RESP_MIRROR", b::RESP_MIRROR),
        ("RESP_DIELECTRIC", b::RESP_DIELECTRIC),
        ("RESP_EXPLICIT_RT", b::RESP_EXPLICIT_RT),
        ("RESP_THIN_LENS", b::RESP_THIN_LENS),
        ("RESP_COATING", b::RESP_COATING),
        ("RESP_GRATING", b::RESP_GRATING),
        ("RESP_IDEAL_MIRROR", b::RESP_IDEAL_MIRROR),
        ("RESP_FLAG_STRAY", b::RESP_FLAG_STRAY),
        ("RESP_FLAG_TRANSMIT", b::RESP_FLAG_TRANSMIT),
    ] {
        assert_eq!(shader_u32(name), host, "{name}");
    }
}
