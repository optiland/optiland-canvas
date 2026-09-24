// Optiland Canvas — full-3D interactive tracer (PR 3D-08).
//
// One thread per launch ray, mirroring the f32 CPU oracle `cpu3.rs` line for
// line: nearest-hit transport in the normalized scene frame, media resolved by
// containment + priority, vector Snell with scalar Fresnel and exact TIR,
// deterministic reflected/transmitted branches on a per-thread stack. Its job is
// to agree with `cpu3.rs`, not with f64 — that split (shader bug vs f32 rounding)
// is what makes a disagreement diagnosable.
//
// Structs mirror `buffers3.rs` field for field; a rotation arrives as its three
// columns in three vec4 and is rebuilt here. Distances are normalized; the host
// maps segment endpoints back to millimetres on read-back.
//
// Part 1 traces the analytic bodies (sphere, box, cylinder), zero-thickness
// surfaces, finite apertures, and area detectors. Extruded and revolved bodies
// (the `verts`/`faces` arenas) land in later parts of this PR.

struct Params {
    n_bodies: u32,
    n_surfaces: u32,
    n_apertures: u32,
    n_detectors: u32,
    n_rays: u32,
    max_depth: u32,
    max_segs: u32,
    ghosts: u32,
    max_ghost: u32,
    ambient: f32,
    bounds: f32,
    flux_eps: f32,
    scale: f32,
    _p0: u32, _p1: u32, _p2: u32,
};

struct Body {
    rot0: vec4<f32>,
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    pos: vec4<f32>,
    params: vec4<f32>,
    kind: u32,
    surface: u32,
    disp: u32,
    priority: i32,
    index_const: f32,
    alpha_per_mm: f32,
    poly_start: u32,
    poly_count: u32,
    face_start: u32,
    face_count: u32,
    contour_start: u32,
    contour_count: u32,
    z_min: f32,
    z_max: f32,
    dp: array<f32, 8>,
    _pad: array<f32, 2>,
};

// A response scalar: the constant `value` when `tab_count == 0`, else
// `tab_count` `(lambda_um, value)` samples at `verts[tab_start..]`, read with the
// out-of-range `policy` (POLICY_*).
struct Spectral {
    value: f32,
    tab_start: u32,
    tab_count: u32,
    policy: u32,
};

// One side of a zero-thickness surface. `r` is the mirror reflectivity or the
// explicit reflectance (a thin lens keeps its normalized focal length there),
// `t` the explicit transmittance, `ar` the dielectric AR residual.
struct Response {
    kind: u32,
    flags: u32,
    _pad: array<u32, 2>,
    r: Spectral,
    t: Spectral,
    ar: Spectral,
};

struct Surface {
    rot0: vec4<f32>,
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    pos: vec4<f32>,
    shape_kind: u32,
    shape_a: f32,
    shape_b: f32,
    poly_start: u32,
    poly_count: u32,
    _pad: array<u32, 3>,
    front: Response,
    back: Response,
};

struct Aperture {
    origin: vec4<f32>,
    normal: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>,
    clear_kind: u32,
    clear_a: f32,
    clear_b: f32,
    clear_poly_start: u32,
    clear_poly_count: u32,
    has_outer: u32,
    outer_kind: u32,
    outer_a: f32,
    outer_b: f32,
    outer_poly_start: u32,
    outer_poly_count: u32,
    _pad: u32,
};

struct Detector {
    rot0: vec4<f32>,
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    pos: vec4<f32>,
    shape_kind: u32,
    shape_a: f32,
    shape_b: f32,
    terminate: u32,
};

struct Seg {
    a: vec4<f32>,
    b: vec4<f32>,
    flux: f32,
    lambda: f32,
    flags: u32,
    _pad: u32,
};

// A revolved-face record (`faces` arena). `a`/`b`/`coeffs` are read per `kind`.
struct RFace {
    kind: u32,
    is_conic: u32,
    // 1 when `resp` carries this face's own response instead of the body's: a
    // coated plate is one body whose two faces do different things.
    has_resp: u32,
    _p1: u32,
    a: vec4<f32>,
    b: vec4<f32>,
    coeffs: array<f32, 8>,
    resp: Response,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read> surfaces: array<Surface>;
@group(0) @binding(3) var<storage, read> apertures: array<Aperture>;
@group(0) @binding(4) var<storage, read> detectors: array<Detector>;
@group(0) @binding(5) var<storage, read> rays: array<vec4<f32>>; // pairs: [origin_flux, dir_lambda]
@group(0) @binding(6) var<storage, read> verts: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> faces: array<RFace>;
@group(0) @binding(8) var<storage, read_write> segs: array<Seg>;

const SEG_VALID: u32 = 1u;
const SEG_TIR: u32 = 2u;
const SEG_GHOST: u32 = 4u;
// GPU-only: the CPU mirror has neither limit. Set on a ray's first segment when
// its tree ran out of segment budget, or lost a branch to the full stack, so the
// host can widen the budget or draw the CPU picture instead.
const SEG_TRUNC_BUDGET: u32 = 8u;
const SEG_TRUNC_STACK: u32 = 16u;

// Face-shape kinds.
const SHAPE_RECT: u32 = 0u;
const SHAPE_DISC: u32 = 1u;
const SHAPE_ANNULUS: u32 = 2u;
const SHAPE_POLYGON: u32 = 3u;

// Response kinds.
const RESP_ABSORBER: u32 = 0u;
const RESP_MIRROR: u32 = 1u;
const RESP_DIELECTRIC: u32 = 2u;
const RESP_EXPLICIT_RT: u32 = 3u;
const RESP_THIN_LENS: u32 = 4u;
const RESP_COATING: u32 = 5u;
// Response.flags: the reflection is stray light (an AR coat's residual), so a
// ghost like bare glass's. Must match buffers3::RESP_FLAG_STRAY.
const RESP_FLAG_STRAY: u32 = 1u;

// A baked thin-film coating. `resp.t` holds the wavelength axis (tab_count
// entries at tab_start, x = lambda) and `resp.r` the grid itself: COAT_ANGLES
// rows, uniform in sin2(theta), of one (R, T) pair per wavelength. Must match
// scene3::COAT_ANGLES; `the_coating_grid_constants_agree` pins that.

// A baked thin-film coating grid: `COAT_ANGLES` rows of `COAT_LAMBDAS` (R, T)
// pairs, uniform in wavelength and in sin²θ. Must match scene3::COAT_* exactly;
// `the_coating_grid_constants_agree` pins that.
const COAT_ANGLES: u32 = 64u;

// Out-of-range policies of a tabulated response scalar.
const POLICY_ERROR: u32 = 0u;
const POLICY_CLAMP: u32 = 1u;
const POLICY_ZERO: u32 = 2u;

// Body geometry kinds.
const OBJ_SPHERE: u32 = 0u;
const OBJ_BOX: u32 = 1u;
const OBJ_CYLINDER: u32 = 2u;
const OBJ_EXTRUDE: u32 = 3u;
const OBJ_REVOLVE: u32 = 4u;
const OBJ_TUBE: u32 = 5u;

// Body surface behaviour kinds.
const SURF_DIELECTRIC: u32 = 0u;
const SURF_ABSORBER: u32 = 1u;

// Revolved-face kinds.
const FACE_SAG: u32 = 0u;
const FACE_FRUSTUM: u32 = 1u;
const FACE_ARC: u32 = 2u;
const TAU: f32 = 6.2831853;

// Self-hit offset and medium-probe distance, in the normalized frame. Matches
// cpu3::EPS exactly.
const EPS: f32 = 1e-4;
const MAX_BRANCHES: u32 = 4096u;
const STACK: u32 = 64u;

fn body_rot(b: Body) -> mat3x3<f32> {
    return mat3x3<f32>(b.rot0.xyz, b.rot1.xyz, b.rot2.xyz);
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

fn poly_contains(start: u32, count: u32, p: vec2<f32>) -> bool {
    var inside = false;
    var j = count - 1u;
    for (var i = 0u; i < count; i = i + 1u) {
        let a = verts[start + i];
        let b = verts[start + j];
        if ((a.y > p.y) != (b.y > p.y)) {
            let t = (p.y - a.y) / (b.y - a.y);
            if (p.x < a.x + t * (b.x - a.x)) {
                inside = !inside;
            }
        }
        j = i;
    }
    return inside;
}

fn shape_contains(kind: u32, a: f32, b: f32, poly_start: u32, poly_count: u32, p: vec2<f32>) -> bool {
    if (kind == SHAPE_RECT) {
        return abs(p.x) <= a && abs(p.y) <= b;
    } else if (kind == SHAPE_DISC) {
        return dot(p, p) <= a * a;
    } else if (kind == SHAPE_ANNULUS) {
        let r2 = dot(p, p);
        return r2 >= a * a && r2 <= b * b;
    }
    return poly_contains(poly_start, poly_count, p);
}

fn cross2(p: vec2<f32>, q: vec2<f32>) -> f32 { return p.x * q.y - p.y * q.x; }

fn extrude_contains(start: u32, count: u32, z_min: f32, z_max: f32, p: vec3<f32>) -> bool {
    return p.z >= z_min && p.z <= z_max && poly_contains(start, count, vec2<f32>(p.x, p.y));
}

// A revolved point is inside if `(x, r)` is inside the sampled meridian contour.
fn revolve_contains(cstart: u32, ccount: u32, p: vec3<f32>) -> bool {
    let r = sqrt(p.y * p.y + p.z * p.z);
    return poly_contains(cstart, ccount, vec2<f32>(p.x, r));
}

// ---------------------------------------------------------------------------
// Media (containment + priority)
// ---------------------------------------------------------------------------

fn body_contains(bi: u32, p: vec3<f32>) -> bool {
    let b = bodies[bi];
    let pl = transpose(body_rot(b)) * (p - b.pos.xyz);
    if (b.kind == OBJ_SPHERE) {
        return dot(pl, pl) <= b.params.x * b.params.x;
    } else if (b.kind == OBJ_BOX) {
        return abs(pl.x) <= b.params.x && abs(pl.y) <= b.params.y && abs(pl.z) <= b.params.z;
    } else if (b.kind == OBJ_CYLINDER) {
        return pl.x * pl.x + pl.y * pl.y <= b.params.x * b.params.x && abs(pl.z) <= b.params.y;
    } else if (b.kind == OBJ_EXTRUDE) {
        return extrude_contains(b.poly_start, b.poly_count, b.z_min, b.z_max, pl);
    } else if (b.kind == OBJ_REVOLVE) {
        return revolve_contains(b.contour_start, b.contour_count, pl);
    } else if (b.kind == OBJ_TUBE) {
        let rho2 = pl.y * pl.y + pl.z * pl.z;
        return abs(pl.x) <= b.params.z && rho2 >= b.params.x * b.params.x && rho2 <= b.params.y * b.params.y;
    }
    return false;
}

fn body_index(bi: u32, lambda: f32) -> f32 {
    let b = bodies[bi];
    if (b.disp == 2u) {
        let l2 = lambda * lambda;
        let n2 = 1.0
            + b.dp[0] * l2 / (l2 - b.dp[3])
            + b.dp[1] * l2 / (l2 - b.dp[4])
            + b.dp[2] * l2 / (l2 - b.dp[5]);
        return sqrt(max(n2, 1.0));
    } else if (b.disp == 1u) {
        return b.dp[0] + b.dp[1] / (lambda * lambda);
    }
    return b.index_const;
}

// Index and attenuation of the highest-priority body containing `p`, else ambient.
fn region_optics(p: vec3<f32>, lambda: f32) -> vec2<f32> {
    var best: i32 = -1;
    var best_pri: i32 = 0;
    for (var bi = 0u; bi < params.n_bodies; bi = bi + 1u) {
        if (body_contains(bi, p)) {
            if (best < 0 || bodies[bi].priority > best_pri) {
                best = i32(bi);
                best_pri = bodies[bi].priority;
            }
        }
    }
    if (best < 0) {
        return vec2<f32>(params.ambient, 0.0);
    }
    let b = bodies[u32(best)];
    if (b.surface == SURF_ABSORBER) {
        return vec2<f32>(params.ambient, 0.0);
    }
    return vec2<f32>(body_index(u32(best), lambda), b.alpha_per_mm);
}

// ---------------------------------------------------------------------------
// Body geometry (local frame)
// ---------------------------------------------------------------------------

struct Hit { hit: bool, t: f32, n: vec3<f32>, face: u32 };

fn no_hit() -> Hit { return Hit(false, 0.0, vec3<f32>(0.0), 0u); }

fn sphere_hit(o: vec3<f32>, d: vec3<f32>, r: f32) -> Hit {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) { return no_hit(); }
    let s = sqrt(disc);
    let t0 = -b - s;
    let t1 = -b + s;
    var t = t0;
    if (t0 <= EPS) { t = t1; }
    if (t <= EPS) { return no_hit(); }
    let p = o + d * t;
    return Hit(true, t, p / r, 0u);
}

fn box_hit(o: vec3<f32>, d: vec3<f32>, half: vec3<f32>) -> Hit {
    var t_enter = -1e30;
    var t_exit = 1e30;
    var enter_axis = 0;
    var exit_axis = 0;
    for (var i = 0; i < 3; i = i + 1) {
        let oi = o[i];
        let di = d[i];
        let hi = half[i];
        if (abs(di) < 1e-20) {
            if (oi < -hi || oi > hi) { return no_hit(); }
            continue;
        }
        let inv = 1.0 / di;
        var ta = (-hi - oi) * inv;
        var tb = (hi - oi) * inv;
        if (ta > tb) { let tmp = ta; ta = tb; tb = tmp; }
        if (ta > t_enter) { t_enter = ta; enter_axis = i; }
        if (tb < t_exit) { t_exit = tb; exit_axis = i; }
        if (t_enter > t_exit) { return no_hit(); }
    }
    var t = t_enter;
    var ax = enter_axis;
    if (t_enter <= EPS) { t = t_exit; ax = exit_axis; }
    if (t <= EPS) { return no_hit(); }
    let p = o + d * t;
    var nl = vec3<f32>(0.0);
    nl[ax] = sign(p[ax]);
    return Hit(true, t, nl, 0u);
}

fn cylinder_hit(o: vec3<f32>, d: vec3<f32>, r: f32, hl: f32) -> Hit {
    var best = no_hit();
    let a = d.x * d.x + d.y * d.y;
    if (a > 1e-20) {
        let b = 2.0 * (o.x * d.x + o.y * d.y);
        let c = o.x * o.x + o.y * o.y - r * r;
        let disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            let sq = sqrt(disc);
            var ts = array<f32, 2>((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a));
            for (var i = 0; i < 2; i = i + 1) {
                let t = ts[i];
                let z = o.z + t * d.z;
                if (t > EPS && abs(z) <= hl && (!best.hit || t < best.t)) {
                    let p = o + d * t;
                    best = Hit(true, t, vec3<f32>(p.x, p.y, 0.0) / r, 0u);
                }
            }
        }
    }
    if (abs(d.z) > 1e-20) {
        var zc = array<f32, 2>(hl, -hl);
        var nz = array<f32, 2>(1.0, -1.0);
        for (var i = 0; i < 2; i = i + 1) {
            let t = (zc[i] - o.z) / d.z;
            if (t > EPS && (!best.hit || t < best.t)) {
                let p = o + d * t;
                if (p.x * p.x + p.y * p.y <= r * r) {
                    best = Hit(true, t, vec3<f32>(0.0, 0.0, nz[i]), 0u);
                }
            }
        }
    }
    return best;
}

// A hollow tube about local +X: params (inner, outer, half_length). Faces in
// the kernel's tube_face order (outer 0, +x end 1, -x end 2, bore 3); the bore's
// outward normal faces the axis. Mirrors cpu3::tube_hit.
fn tube_hit(o: vec3<f32>, d: vec3<f32>, p: vec3<f32>) -> Hit {
    var best = no_hit();
    let ri = p.x;
    let ro = p.y;
    let hl = p.z;
    let a = d.y * d.y + d.z * d.z;
    if (a > 1e-20) {
        var rs = array<f32, 2>(ro, ri);
        var sg = array<f32, 2>(1.0, -1.0);
        var fc = array<u32, 2>(0u, 3u);
        for (var k = 0; k < 2; k = k + 1) {
            let r = rs[k];
            let b = 2.0 * (o.y * d.y + o.z * d.z);
            let c = o.y * o.y + o.z * o.z - r * r;
            let disc = b * b - 4.0 * a * c;
            if (disc >= 0.0) {
                let sq = sqrt(disc);
                var ts = array<f32, 2>((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a));
                for (var i = 0; i < 2; i = i + 1) {
                    let t = ts[i];
                    let q = o + d * t;
                    if (t > EPS && abs(q.x) <= hl && (!best.hit || t < best.t)) {
                        best = Hit(true, t, vec3<f32>(0.0, q.y, q.z) * (sg[k] / r), fc[k]);
                    }
                }
            }
        }
    }
    if (abs(d.x) > 1e-20) {
        var xc = array<f32, 2>(hl, -hl);
        var nx = array<f32, 2>(1.0, -1.0);
        var fe = array<u32, 2>(1u, 2u);
        for (var i = 0; i < 2; i = i + 1) {
            let t = (xc[i] - o.x) / d.x;
            if (t > EPS && (!best.hit || t < best.t)) {
                let q = o + d * t;
                let rho2 = q.y * q.y + q.z * q.z;
                if (rho2 >= ri * ri && rho2 <= ro * ro) {
                    best = Hit(true, t, vec3<f32>(nx[i], 0.0, 0.0), fe[i]);
                }
            }
        }
    }
    return best;
}

// 2D ray `o + t·d` (d need not be unit) against segment `a–b`; `w` is 1 on a hit
// with `t > EPS` on the segment, else 0. Returns `(t, unit edge normal, w)`.
fn segment_hit(a: vec2<f32>, b: vec2<f32>, o: vec2<f32>, d: vec2<f32>) -> vec4<f32> {
    let edge = b - a;
    let denom = cross2(d, edge);
    if (abs(denom) < 1e-20) { return vec4<f32>(0.0); }
    let ao = a - o;
    let t = cross2(ao, edge) / denom;
    let u = cross2(ao, d) / denom;
    if (t <= EPS || u < 0.0 || u > 1.0) { return vec4<f32>(0.0); }
    let n = normalize(vec2<f32>(edge.y, -edge.x));
    return vec4<f32>(t, n.x, n.y, 1.0);
}

// Nearest extruded-profile hit: ruled side faces plus the two end caps, with the
// outward local normal. Mirrors cpu3::extrude_hit.
fn extrude_hit(start: u32, count: u32, z_min: f32, z_max: f32, o: vec3<f32>, d: vec3<f32>) -> Hit {
    var best = no_hit();
    let o2 = vec2<f32>(o.x, o.y);
    let d2 = vec2<f32>(d.x, d.y);
    if (length(d2) > 1e-12) {
        for (var i = 0u; i < count; i = i + 1u) {
            let a = verts[start + i];
            let b = verts[start + (i + 1u) % count];
            let s = segment_hit(a, b, o2, d2);
            if (s.w > 0.5) {
                let t = s.x;
                let z = o.z + t * d.z;
                if (z >= z_min && z <= z_max && (!best.hit || t < best.t)) {
                    let p = o + d * t;
                    var n = vec3<f32>(s.y, s.z, 0.0);
                    if (extrude_contains(start, count, z_min, z_max, p + n * EPS)) { n = -n; }
                    best = Hit(true, t, n, 0u);
                }
            }
        }
    }
    if (abs(d.z) > 1e-12) {
        var zc = array<f32, 2>(z_min, z_max);
        var nz = array<f32, 2>(-1.0, 1.0);
        for (var i = 0; i < 2; i = i + 1) {
            let t = (zc[i] - o.z) / d.z;
            if (t > EPS && (!best.hit || t < best.t)) {
                let p = o + d * t;
                if (poly_contains(start, count, vec2<f32>(p.x, p.y))) {
                    best = Hit(true, t, vec3<f32>(0.0, 0.0, nz[i]), 0u);
                }
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Revolved profiles (Part 3): conic (exact quadratic, §7.3), even asphere
// (safeguarded Newton, §7.4), frustum rims, and axis-centred arcs.
// ---------------------------------------------------------------------------

fn rf_sag(c: f32, k: f32, coeffs: array<f32, 8>, r: f32) -> f32 {
    let r2 = r * r;
    let denom = 1.0 + sqrt(max(1.0 - (1.0 + k) * c * c * r2, 0.0));
    var sag = c * r2 / denom;
    var rp = r2 * r2;
    var cf = coeffs;
    for (var i = 0; i < 8; i = i + 1) {
        sag = sag + cf[i] * rp;
        rp = rp * r2;
    }
    return sag;
}

fn rf_dsag(c: f32, k: f32, coeffs: array<f32, 8>, r: f32) -> f32 {
    let r2 = r * r;
    let s = sqrt(max(1.0 - (1.0 + k) * c * c * r2, 0.0));
    var d = 0.0;
    if (s > 0.0) { d = c * r / s; }
    var rp = r2 * r;
    var cf = coeffs;
    for (var i = 0; i < 8; i = i + 1) {
        d = d + cf[i] * f32(2 * i + 4) * rp;
        rp = rp * r2;
    }
    return d;
}

// The outward normal of a conic/asphere of revolution at local point `p` (§7.4).
fn sag_normal(axis_sign: f32, c: f32, k: f32, coeffs: array<f32, 8>, p: vec3<f32>) -> vec3<f32> {
    let rad = vec3<f32>(0.0, p.y, p.z);
    let r = length(rad);
    let axis = vec3<f32>(axis_sign, 0.0, 0.0);
    if (r > 0.0) {
        return normalize(axis - rad / r * rf_dsag(c, k, coeffs, r));
    }
    return normalize(axis);
}

struct QRoots { n: u32, t0: f32, t1: f32 };

fn quad_roots(aa: f32, bb: f32, cc: f32) -> QRoots {
    if (abs(aa) < 1e-20) {
        if (abs(bb) > 1e-20) { return QRoots(1u, -cc / bb, 0.0); }
        return QRoots(0u, 0.0, 0.0);
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if (disc < 0.0) { return QRoots(0u, 0.0, 0.0); }
    let sq = sqrt(disc);
    let sgn = select(-1.0, 1.0, bb >= 0.0);
    let q = -0.5 * (bb + sgn * sq);
    let t0 = q / aa;
    if (abs(q) > 1e-20) { return QRoots(2u, t0, cc / q); }
    return QRoots(1u, t0, 0.0);
}

// The residual f(t) = axial(t) − sag(r(t)) and its derivative, for the asphere
// Newton solve (§7.4).
fn a_residual(t: f32, c: f32, k: f32, coeffs: array<f32, 8>, z0: f32, dz: f32,
              o_rad: vec3<f32>, d_rad: vec3<f32>) -> f32 {
    let r = length(o_rad + d_rad * t);
    return (z0 + t * dz) - rf_sag(c, k, coeffs, r);
}

fn a_dresidual(t: f32, c: f32, k: f32, coeffs: array<f32, 8>, dz: f32,
               o_rad: vec3<f32>, d_rad: vec3<f32>) -> f32 {
    let rad = o_rad + d_rad * t;
    let r = length(rad);
    var dr = 0.0;
    if (r > 0.0) { dr = dot(rad, d_rad) / r; }
    return dz - rf_dsag(c, k, coeffs, r) * dr;
}

// Safeguarded Newton for the asphere residual near `seed`; `.y` is 1 on success.
fn a_refine(seed: f32, tmin: f32, c: f32, k: f32, coeffs: array<f32, 8>, z0: f32, dz: f32,
            o_rad: vec3<f32>, d_rad: vec3<f32>) -> vec2<f32> {
    var t = seed;
    let span = 1e-2;
    var lo = seed - span;
    var hi = seed + span;
    var flo = a_residual(lo, c, k, coeffs, z0, dz, o_rad, d_rad);
    let fhi = a_residual(hi, c, k, coeffs, z0, dz, o_rad, d_rad);
    let bracketed = flo * fhi <= 0.0;

    for (var it = 0; it < 48; it = it + 1) {
        let ft = a_residual(t, c, k, coeffs, z0, dz, o_rad, d_rad);
        if (abs(ft) < 1e-7) { break; }
        let dft = a_dresidual(t, c, k, coeffs, dz, o_rad, d_rad);
        var nt = 1e30;
        if (abs(dft) > 1e-30) { nt = t - ft / dft; }
        let finite = nt == nt && nt < 1e30 && nt > -1e30;
        if (!finite || (bracketed && (nt < lo || nt > hi))) { nt = 0.5 * (lo + hi); }
        if (bracketed) {
            let fnt = a_residual(nt, c, k, coeffs, z0, dz, o_rad, d_rad);
            if (flo * fnt <= 0.0) { hi = nt; } else { lo = nt; flo = fnt; }
        }
        let step = abs(nt - t);
        t = nt;
        if (step < 1e-8 * (1.0 + abs(t))) { break; }
    }
    let ok = abs(a_residual(t, c, k, coeffs, z0, dz, o_rad, d_rad)) < 1e-5 && t > tmin;
    return vec2<f32>(t, select(0.0, 1.0, ok));
}

// Accept a sag-surface root: check the aperture/sag envelope, and keep it if it is
// the nearest so far. Returns the updated best.
fn sag_accept(best: Hit, t: f32, tmin: f32, o: vec3<f32>, d: vec3<f32>, vertex_x: f32,
              axis_sign: f32, c: f32, k: f32, coeffs: array<f32, 8>, outer_r: f32,
              z_lo: f32, z_hi: f32, z_tol: f32, r_tol: f32) -> Hit {
    if (t <= tmin) { return best; }
    if (best.hit && t >= best.t) { return best; }
    let p = o + d * t;
    let r = sqrt(p.y * p.y + p.z * p.z);
    let z = (p.x - vertex_x) * axis_sign;
    if (r <= outer_r + r_tol && z >= z_lo - z_tol && z <= z_hi + z_tol) {
        return Hit(true, t, sag_normal(axis_sign, c, k, coeffs, p), 0u);
    }
    return best;
}

fn sag_face_hit(f: RFace, o: vec3<f32>, d: vec3<f32>, tmin: f32) -> Hit {
    let vertex_x = f.a.x;
    let axis_sign = f.a.y;
    let c = f.a.z;
    let k = f.a.w;
    let outer_r = f.b.x;
    let coeffs = f.coeffs;

    let ov = o - vec3<f32>(vertex_x, 0.0, 0.0);
    let z0 = ov.x * axis_sign;
    let dz = d.x * axis_sign;
    let o_rad = vec3<f32>(0.0, ov.y, ov.z);
    let d_rad = vec3<f32>(0.0, d.y, d.z);

    let sag_edge = rf_sag(c, k, coeffs, outer_r);
    let z_lo = min(sag_edge, 0.0);
    let z_hi = max(sag_edge, 0.0);
    let z_tol = 1e-4 * (1.0 + abs(outer_r));
    let r_tol = 1e-5 * (1.0 + outer_r);

    let k1 = 1.0 + k;
    let aa = c * (dot(d_rad, d_rad) + k1 * dz * dz);
    let bb = c * (2.0 * dot(o_rad, d_rad) + 2.0 * k1 * z0 * dz) - 2.0 * dz;
    let cc = c * (dot(o_rad, o_rad) + k1 * z0 * z0) - 2.0 * z0;
    let qr = quad_roots(aa, bb, cc);

    var best = no_hit();
    if (f.is_conic == 1u) {
        if (qr.n >= 1u) { best = sag_accept(best, qr.t0, tmin, o, d, vertex_x, axis_sign, c, k, coeffs, outer_r, z_lo, z_hi, z_tol, r_tol); }
        if (qr.n >= 2u) { best = sag_accept(best, qr.t1, tmin, o, d, vertex_x, axis_sign, c, k, coeffs, outer_r, z_lo, z_hi, z_tol, r_tol); }
        return best;
    }

    // Even asphere: seed from the conic roots; if there are none, march across the
    // aperture span to bracket a sign change, then Newton-refine (§7.4).
    if (qr.n >= 1u) {
        let ref0 = a_refine(qr.t0, tmin, c, k, coeffs, z0, dz, o_rad, d_rad);
        if (ref0.y > 0.5) { best = sag_accept(best, ref0.x, tmin, o, d, vertex_x, axis_sign, c, k, coeffs, outer_r, z_lo, z_hi, z_tol, r_tol); }
    }
    if (qr.n >= 2u) {
        let ref1 = a_refine(qr.t1, tmin, c, k, coeffs, z0, dz, o_rad, d_rad);
        if (ref1.y > 0.5) { best = sag_accept(best, ref1.x, tmin, o, d, vertex_x, axis_sign, c, k, coeffs, outer_r, z_lo, z_hi, z_tol, r_tol); }
    }
    if (qr.n == 0u) {
        let start = max(tmin, 0.0);
        let step = max(outer_r, 0.05) * 0.1;
        var pt = start;
        var pv = a_residual(pt, c, k, coeffs, z0, dz, o_rad, d_rad);
        for (var i = 1; i <= 200; i = i + 1) {
            let t = start + step * f32(i);
            let v = a_residual(t, c, k, coeffs, z0, dz, o_rad, d_rad);
            if (pv * v < 0.0) {
                let rf = a_refine(0.5 * (pt + t), tmin, c, k, coeffs, z0, dz, o_rad, d_rad);
                if (rf.y > 0.5) { best = sag_accept(best, rf.x, tmin, o, d, vertex_x, axis_sign, c, k, coeffs, outer_r, z_lo, z_hi, z_tol, r_tol); }
            }
            pt = t;
            pv = v;
        }
    }
    return best;
}

// A line segment revolved about +X: cone, cylinder, or flat annular face.
fn frustum_face_hit(f: RFace, o: vec3<f32>, d: vec3<f32>, tmin: f32) -> Hit {
    let x0 = f.a.x;
    let y0 = f.a.y;
    let x1 = f.a.z;
    let y1 = f.a.w;
    let x_lo = min(x0, x1);
    let x_hi = max(x0, x1);
    let flat = abs(x1 - x0) <= 1e-9 * (1.0 + abs(x0) + abs(x1));

    if (flat) {
        if (abs(d.x) < 1e-12) { return no_hit(); }
        let t = (x0 - o.x) / d.x;
        if (t <= tmin) { return no_hit(); }
        let p = o + d * t;
        let r = sqrt(p.y * p.y + p.z * p.z);
        let r_lo = min(y0, y1);
        let r_hi = max(y0, y1);
        let pad = 1e-9 * (1.0 + r_hi);
        if (r >= r_lo - pad && r <= r_hi + pad) {
            return Hit(true, t, vec3<f32>(1.0, 0.0, 0.0), 0u);
        }
        return no_hit();
    }

    let m = (y1 - y0) / (x1 - x0);
    let b = y0 - m * x0;
    let g0 = m * o.x + b;
    let aa = d.y * d.y + d.z * d.z - m * m * d.x * d.x;
    let bb = 2.0 * (o.y * d.y + o.z * d.z) - 2.0 * m * d.x * g0;
    let cc = o.y * o.y + o.z * o.z - g0 * g0;
    let pad = 1e-6 * (1.0 + abs(x_hi) + abs(x_lo));
    let qr = quad_roots(aa, bb, cc);
    var best = no_hit();
    var roots = array<f32, 2>(qr.t0, qr.t1);
    for (var i = 0u; i < qr.n; i = i + 1u) {
        let t = roots[i];
        if (t <= tmin || (best.hit && t >= best.t)) { continue; }
        let p = o + d * t;
        let r_line = m * p.x + b;
        if (r_line >= -pad && p.x >= x_lo - pad && p.x <= x_hi + pad) {
            best = Hit(true, t, normalize(vec3<f32>(-m * r_line, p.y, p.z)), 0u);
        }
    }
    return best;
}

fn rem_euclid(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn angle_in_sweep(ang: f32, a0: f32, a1: f32) -> bool {
    let sweep = rem_euclid(a1 - a0, TAU);
    if (sweep >= TAU - 1e-6) { return true; }
    let rel = rem_euclid(ang - a0, TAU);
    return rel <= sweep + 1e-6;
}

// An axis-centred arc revolved about +X: a spherical zone.
fn arc_face_hit(f: RFace, o: vec3<f32>, d: vec3<f32>, tmin: f32) -> Hit {
    let center_x = f.a.x;
    let r = f.a.y;
    let a0 = f.a.z;
    let a1 = f.a.w;
    let c3 = vec3<f32>(center_x, 0.0, 0.0);
    let oc = o - c3;
    let b = dot(oc, d);
    let disc = b * b - (dot(oc, oc) - r * r);
    if (disc < 0.0) { return no_hit(); }
    let sq = sqrt(disc);
    var ts = array<f32, 2>(-b - sq, -b + sq);
    var best = no_hit();
    for (var i = 0; i < 2; i = i + 1) {
        let t = ts[i];
        if (t <= tmin || (best.hit && t >= best.t)) { continue; }
        let p = o + d * t;
        let rad = sqrt(p.y * p.y + p.z * p.z);
        let ang = atan2(rad, p.x - center_x);
        if (angle_in_sweep(ang, a0, a1)) {
            best = Hit(true, t, (p - c3) / r, 0u);
        }
    }
    return best;
}

// Nearest revolved-profile hit, with the outward local normal oriented by a
// containment probe (a single surface cannot say which side is glass).
fn revolve_hit(face_start: u32, face_count: u32, cstart: u32, ccount: u32,
               o: vec3<f32>, d: vec3<f32>, tmin: f32) -> Hit {
    var best = no_hit();
    for (var fi = 0u; fi < face_count; fi = fi + 1u) {
        let f = faces[face_start + fi];
        var h = no_hit();
        if (f.kind == FACE_SAG) {
            h = sag_face_hit(f, o, d, tmin);
        } else if (f.kind == FACE_FRUSTUM) {
            h = frustum_face_hit(f, o, d, tmin);
        } else {
            h = arc_face_hit(f, o, d, tmin);
        }
        if (h.hit && (!best.hit || h.t < best.t)) {
            best = h;
            best.face = fi;
        }
    }
    if (best.hit) {
        let p = o + d * best.t;
        if (revolve_contains(cstart, ccount, p + best.n * EPS)) {
            best.n = -best.n;
        }
    }
    return best;
}

// Nearest body hit, with the world-space outward normal.
fn body_hit(bi: u32, o: vec3<f32>, d: vec3<f32>) -> Hit {
    let b = bodies[bi];
    let rot = body_rot(b);
    let ol = transpose(rot) * (o - b.pos.xyz);
    let dl = transpose(rot) * d;
    var h = no_hit();
    if (b.kind == OBJ_SPHERE) {
        h = sphere_hit(ol, dl, b.params.x);
    } else if (b.kind == OBJ_BOX) {
        h = box_hit(ol, dl, b.params.xyz);
    } else if (b.kind == OBJ_CYLINDER) {
        h = cylinder_hit(ol, dl, b.params.x, b.params.y);
    } else if (b.kind == OBJ_EXTRUDE) {
        h = extrude_hit(b.poly_start, b.poly_count, b.z_min, b.z_max, ol, dl);
    } else if (b.kind == OBJ_REVOLVE) {
        h = revolve_hit(b.face_start, b.face_count, b.contour_start, b.contour_count, ol, dl, EPS);
    } else if (b.kind == OBJ_TUBE) {
        h = tube_hit(ol, dl, b.params.xyz);
    }
    if (h.hit) {
        h.n = normalize(rot * h.n);
    }
    return h;
}

// ---------------------------------------------------------------------------
// Event objects (planes)
// ---------------------------------------------------------------------------

struct Face { hit: bool, t: f32, uv: vec2<f32> };

fn plane_face(rot: mat3x3<f32>, pos: vec3<f32>, o: vec3<f32>, d: vec3<f32>) -> Face {
    let ol = transpose(rot) * (o - pos);
    let dl = transpose(rot) * d;
    if (abs(dl.z) < 1e-12) { return Face(false, 0.0, vec2<f32>(0.0)); }
    let t = -ol.z / dl.z;
    if (t <= EPS) { return Face(false, 0.0, vec2<f32>(0.0)); }
    let p = ol + dl * t;
    return Face(true, t, vec2<f32>(p.x, p.y));
}

fn surface_rot(s: Surface) -> mat3x3<f32> {
    return mat3x3<f32>(s.rot0.xyz, s.rot1.xyz, s.rot2.xyz);
}

fn detector_rot(dd: Detector) -> mat3x3<f32> {
    return mat3x3<f32>(dd.rot0.xyz, dd.rot1.xyz, dd.rot2.xyz);
}

// ---------------------------------------------------------------------------
// Optics
// ---------------------------------------------------------------------------

fn reflect3(d: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return normalize(d - 2.0 * dot(d, n) * n);
}

// Refracted direction; `w` is 1 on refraction, 0 on TIR. `n` opposes `d`.
fn refract3(d: vec3<f32>, n: vec3<f32>, n1: f32, n2: f32) -> vec4<f32> {
    let eta = n1 / n2;
    let cos_i = -dot(n, d);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if (k < 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    let t = normalize(eta * d + (eta * cos_i - sqrt(k)) * n);
    return vec4<f32>(t, 1.0);
}

fn reflectance(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let ci = clamp(abs(cos_i), 0.0, 1.0);
    let s2 = 1.0 - ci * ci;
    let st2 = (n1 / n2) * (n1 / n2) * s2;
    if (st2 >= 1.0) { return 1.0; }
    let ct = sqrt(1.0 - st2);
    let rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    let rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    return 0.5 * (rs * rs + rp * rp);
}

fn beer_lambert(alpha_per_mm: f32, t_norm: f32) -> f32 {
    if (alpha_per_mm <= 0.0) { return 1.0; }
    return exp(-alpha_per_mm * t_norm * params.scale);
}

// A response scalar at `lambda`. `.x` is the value; `.y` is 1 when there is
// one and 0 when a table under the Error policy has none there. Mirrors
// scene3::Spectral32::sample: piecewise linear over the interval that
// `partition_point(x <= lambda) - 1` picks, clamped to the last interval. The
// tables are short, so the interval is found by a linear scan.
fn sample_spectral(s: Spectral, lambda: f32) -> vec2<f32> {
    if (s.tab_count == 0u) { return vec2<f32>(s.value, 1.0); }
    let n = s.tab_count;
    let first = verts[s.tab_start];
    let last = verts[s.tab_start + n - 1u];
    if (lambda < first.x || lambda > last.x) {
        if (s.policy == POLICY_CLAMP) {
            return vec2<f32>(select(last.y, first.y, lambda < first.x), 1.0);
        } else if (s.policy == POLICY_ZERO) {
            return vec2<f32>(0.0, 1.0);
        }
        return vec2<f32>(0.0, 0.0);
    }
    // The first sample past `lambda` (n if none); the wavelengths increase.
    var pp = n;
    for (var k = 0u; k < n; k = k + 1u) {
        if (verts[s.tab_start + k].x > lambda) {
            pp = k;
            break;
        }
    }
    let i = min(max(pp, 1u) - 1u, n - 2u);
    let a = verts[s.tab_start + i];
    let b = verts[s.tab_start + i + 1u];
    var f = 0.0;
    if (b.x > a.x) { f = (lambda - a.x) / (b.x - a.x); }
    return vec2<f32>(a.y + f * (b.y - a.y), 1.0);
}

// Bilinear lookup into a baked coating grid: COAT_ANGLES rows, uniform in the
// ambient-medium sin^2(theta), of one (R, T) pair per baked wavelength.
//
// `cos_i` is measured in `n_medium`, the medium the ray is actually in. A coated
// face of a plate is crossed from both sides, and the quantity both crossings
// share is n*sin(theta), which Snell conserves, so that is what the grid is
// indexed on. Mirrors scene3::sample_coating_grid exactly.
fn sample_coating(resp: Response, lambda: f32, cos_i: f32, n_medium: f32) -> vec2<f32> {
    let n_l = resp.t.tab_count;
    if (n_l == 0u) { return vec2<f32>(0.0, 0.0); }
    var l0 = 0u;
    loop {
        if (l0 + 2u >= n_l) { break; }
        if (verts[resp.t.tab_start + l0 + 1u].x >= lambda) { break; }
        l0 = l0 + 1u;
    }
    let l1 = min(l0 + 1u, n_l - 1u);
    let lam_a = verts[resp.t.tab_start + l0].x;
    let lam_b = verts[resp.t.tab_start + l1].x;
    var tl = 0.0;
    if (lam_b > lam_a) { tl = clamp((lambda - lam_a) / (lam_b - lam_a), 0.0, 1.0); }

    let sin_local = sqrt(max(1.0 - cos_i * cos_i, 0.0));
    var sin_amb = sin_local;
    if (params.ambient > 0.0) {
        sin_amb = clamp(n_medium * sin_local / params.ambient, 0.0, 1.0);
    }
    let fa = clamp((sin_amb * sin_amb) / 0.999, 0.0, 1.0) * f32(COAT_ANGLES - 1u);
    let a0 = u32(floor(fa));
    let a1 = min(a0 + 1u, COAT_ANGLES - 1u);
    let ta = fa - floor(fa);

    let base = resp.r.tab_start;
    let lo = mix(verts[base + a0 * n_l + l0], verts[base + a0 * n_l + l1], tl);
    let hi = mix(verts[base + a1 * n_l + l0], verts[base + a1 * n_l + l1], tl);
    return mix(lo, hi, ta);
}

// ---------------------------------------------------------------------------
// Per-thread branch stack
// ---------------------------------------------------------------------------

struct Work {
    o: vec3<f32>,
    d: vec3<f32>,
    flux: f32,
    depth: u32,
    gdepth: u32,
    medium_n: f32,
    medium_alpha: f32,
    flags: u32,
};

var<private> stack: array<Work, 64>;
var<private> sp: u32;
// A branch was dropped because the stack was full (see SEG_TRUNC_STACK).
var<private> stack_dropped: bool;

fn push(w: Work, min_flux: f32) {
    if (w.depth > params.max_depth || w.flux < min_flux) { return; }
    if (sp < STACK) {
        stack[sp] = w;
        sp = sp + 1u;
    } else {
        stack_dropped = true;
    }
}

// ---------------------------------------------------------------------------
// Nearest event
// ---------------------------------------------------------------------------

// kind: 0 body, 1 surface, 2 aperture, 3 detector, 4 none
struct Event { t: f32, kind: u32, idx: u32, n: vec3<f32>, uv: vec2<f32>, face: u32 };

fn nearest_event(o: vec3<f32>, d: vec3<f32>) -> Event {
    var ev = Event(params.bounds, 4u, 0u, vec3<f32>(0.0), vec2<f32>(0.0), 0u);

    for (var bi = 0u; bi < params.n_bodies; bi = bi + 1u) {
        let h = body_hit(bi, o, d);
        if (h.hit && h.t < ev.t) {
            ev = Event(h.t, 0u, bi, h.n, vec2<f32>(0.0), h.face);
        }
    }
    for (var si = 0u; si < params.n_surfaces; si = si + 1u) {
        let s = surfaces[si];
        let f = plane_face(surface_rot(s), s.pos.xyz, o, d);
        if (f.hit && f.t < ev.t && shape_contains(s.shape_kind, s.shape_a, s.shape_b, s.poly_start, s.poly_count, f.uv)) {
            let n = normalize(surface_rot(s) * vec3<f32>(0.0, 0.0, 1.0));
            ev = Event(f.t, 1u, si, n, f.uv, 0u);
        }
    }
    for (var ai = 0u; ai < params.n_apertures; ai = ai + 1u) {
        let a = apertures[ai];
        let denom = dot(d, a.normal.xyz);
        if (abs(denom) < 1e-12) { continue; }
        let t = dot(a.origin.xyz - o, a.normal.xyz) / denom;
        if (t <= EPS || t >= ev.t) { continue; }
        let rel = o + d * t - a.origin.xyz;
        let uv = vec2<f32>(dot(rel, a.u.xyz), dot(rel, a.v.xyz));
        if (shape_contains(a.clear_kind, a.clear_a, a.clear_b, a.clear_poly_start, a.clear_poly_count, uv)) {
            continue; // passes through the opening
        }
        var masked = true;
        if (a.has_outer == 1u) {
            masked = shape_contains(a.outer_kind, a.outer_a, a.outer_b, a.outer_poly_start, a.outer_poly_count, uv);
        }
        if (masked) {
            ev = Event(t, 2u, ai, vec3<f32>(0.0), vec2<f32>(0.0), 0u);
        }
    }
    for (var di = 0u; di < params.n_detectors; di = di + 1u) {
        let dd = detectors[di];
        let f = plane_face(detector_rot(dd), dd.pos.xyz, o, d);
        if (f.hit && f.t < ev.t && shape_contains(dd.shape_kind, dd.shape_a, dd.shape_b, 0u, 0u, f.uv)) {
            ev = Event(f.t, 3u, di, vec3<f32>(0.0), f.uv, 0u);
        }
    }
    return ev;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rid = gid.x;
    if (rid >= params.n_rays) { return; }

    let origin_flux = rays[rid * 2u];
    let dir_lambda = rays[rid * 2u + 1u];
    let launch_o = origin_flux.xyz;
    let flux0 = origin_flux.w;
    let launch_d = normalize(dir_lambda.xyz);
    let lambda = dir_lambda.w;

    let base = rid * params.max_segs;
    let min_flux = params.flux_eps * flux0;

    let m0 = region_optics(launch_o, lambda);
    sp = 0u;
    stack_dropped = false;
    stack[0] = Work(launch_o, launch_d, flux0, 0u, 0u, m0.x, m0.y, 0u);
    sp = 1u;

    var seg_count = 0u;
    var branches = 0u;
    var budget_hit = false;
    loop {
        if (sp == 0u) { break; }
        if (seg_count >= params.max_segs) {
            budget_hit = true;
            break;
        }
        sp = sp - 1u;
        let w = stack[sp];
        branches = branches + 1u;
        if (branches > MAX_BRANCHES || w.depth >= params.max_depth || w.flux < min_flux) { continue; }

        let o = w.o;
        let d = w.d;
        let ev = nearest_event(o, d);

        if (ev.kind == 4u) {
            let end = o + d * params.bounds;
            segs[base + seg_count] = Seg(vec4<f32>(o, 0.0), vec4<f32>(end, 0.0), w.flux, lambda, w.flags | SEG_VALID, 0u);
            seg_count = seg_count + 1u;
            continue;
        }

        let point = o + d * ev.t;
        let flux_end = w.flux * beer_lambert(w.medium_alpha, ev.t);
        segs[base + seg_count] = Seg(vec4<f32>(o, 0.0), vec4<f32>(point, 0.0), flux_end, lambda, w.flags | SEG_VALID, 0u);
        seg_count = seg_count + 1u;

        if (ev.kind == 0u) {
            interact_body(w, ev, point, flux_end, lambda, min_flux);
        } else if (ev.kind == 1u) {
            interact_surface(w, ev, point, flux_end, lambda, min_flux);
        } else if (ev.kind == 2u) {
            // Aperture: the branch ends here (blocked).
        } else {
            // Detector.
            if (detectors[ev.idx].terminate == 0u) {
                var pw = w;
                pw.o = point + d * EPS;
                push(pw, min_flux);
            }
        }
    }

    // A ray whose tree did not fit is marked on its first segment; a truncated
    // ray always has one.
    if (seg_count > 0u && (budget_hit || stack_dropped)) {
        var mark = 0u;
        if (budget_hit) { mark = mark | SEG_TRUNC_BUDGET; }
        if (stack_dropped) { mark = mark | SEG_TRUNC_STACK; }
        segs[base].flags = segs[base].flags | mark;
    }
}

fn interact_body(w: Work, ev: Event, point: vec3<f32>, flux_end: f32, lambda: f32, min_flux: f32) {
    // An absorbing body swallows the ray at its surface, unless the face it hit
    // carries a response of its own: the silvered bore of a black tube.
    let bb = bodies[ev.idx];
    var own_face = false;
    if (ev.face < bb.face_count) {
        own_face = faces[bb.face_start + ev.face].has_resp == 1u;
    }
    if (bb.surface == SURF_ABSORBER && !own_face) { return; }
    var n = ev.n;
    if (dot(n, w.d) > 0.0) { n = -n; }
    let m1 = region_optics(point + n * EPS, lambda);
    let m2 = region_optics(point - n * EPS, lambda);
    let n1 = m1.x;
    let n2 = m2.x;

    // A coated face of a solid: a real plate carries its film on one face and
    // plain glass on the other, so the response belongs to the face. R and T
    // come from the baked grid at this ray's own angle; the transmitted branch
    // still refracts, because unlike a zero-thickness plate this interface
    // really does change the medium. Mirrors cpu3::interact_body.
    let b = bodies[ev.idx];
    if (ev.face < b.face_count) {
        let f = faces[b.face_start + ev.face];
        // A mirror face reflects what its reflectivity says and absorbs the
        // rest. Mirrors cpu3::interact_body.
        if (f.has_resp == 1u && f.resp.kind == RESP_MIRROR) {
            let rs = sample_spectral(f.resp.r, lambda);
            if (rs.y < 0.5) { return; }
            let dir = reflect3(w.d, n);
            push(Work(point + dir * EPS, dir, flux_end * clamp(rs.x, 0.0, 1.0), w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
            return;
        }
        if (f.has_resp == 1u && f.resp.kind == RESP_COATING) {
            // Past the critical angle nothing gets through whatever the stack,
            // and the grid saturates at grazing rather than saying so: total
            // internal reflection is the whole ray and a primary path, as at
            // bare glass below. Mirrors cpu3::interact_body.
            let resc = refract3(w.d, n, n1, n2);
            if (resc.w < 0.5) {
                let dir = reflect3(w.d, n);
                push(Work(point + dir * EPS, dir, flux_end, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags | SEG_TIR), min_flux);
                return;
            }
            let cos_ic = clamp(-dot(n, w.d), 0.0, 1.0);
            let rt = sample_coating(f.resp, lambda, cos_ic, w.medium_n);
            let rc = clamp(rt.x, 0.0, 1.0);
            let tc = clamp(rt.y, 0.0, 1.0);
            // An AR coat's reflection is a ghost, like bare glass's below; any
            // other stack's is the path it was designed for.
            let stray = (f.resp.flags & RESP_FLAG_STRAY) != 0u;
            if (rc > 0.0) {
                let dir = reflect3(w.d, n);
                if (!stray) {
                    push(Work(point + dir * EPS, dir, flux_end * rc, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
                } else if (params.ghosts == 1u && w.gdepth < params.max_ghost) {
                    push(Work(point + dir * EPS, dir, flux_end * rc, w.depth + 1u, w.gdepth + 1u, w.medium_n, w.medium_alpha, (w.flags & ~SEG_TIR) | SEG_GHOST), min_flux);
                }
            }
            if (tc > 0.0) {
                let td2 = resc.xyz;
                push(Work(point + td2 * EPS, td2, flux_end * tc, w.depth + 1u, w.gdepth, n2, m2.y, w.flags & ~SEG_TIR), min_flux);
            }
            return;
        }
    }
    if (b.surface == SURF_ABSORBER) { return; }

    let res = refract3(w.d, n, n1, n2);
    if (res.w < 0.5) {
        // Total internal reflection.
        let dir = reflect3(w.d, n);
        push(Work(point + dir * EPS, dir, flux_end, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags | SEG_TIR), min_flux);
    } else {
        let cos_i = clamp(-dot(n, w.d), 0.0, 1.0);
        let r = clamp(reflectance(cos_i, n1, n2), 0.0, 1.0);
        let td = res.xyz;
        push(Work(point + td * EPS, td, flux_end * (1.0 - r), w.depth + 1u, w.gdepth, n2, m2.y, w.flags & ~SEG_TIR), min_flux);
        if (params.ghosts == 1u && w.gdepth < params.max_ghost && r > 0.0) {
            let dir = reflect3(w.d, n);
            push(Work(point + dir * EPS, dir, flux_end * r, w.depth + 1u, w.gdepth + 1u, w.medium_n, w.medium_alpha, (w.flags & ~SEG_TIR) | SEG_GHOST), min_flux);
        }
    }
}

fn interact_surface(w: Work, ev: Event, point: vec3<f32>, flux_end: f32, lambda: f32, min_flux: f32) {
    let s = surfaces[ev.idx];
    let from_front = dot(w.d, ev.n) < 0.0;
    var resp = s.back;
    if (from_front) { resp = s.front; }
    let kind = resp.kind;
    var n_oppose = -ev.n;
    if (from_front) { n_oppose = ev.n; }

    if (kind == RESP_ABSORBER) {
        return;
    } else if (kind == RESP_THIN_LENS) {
        // Ideal-lens slope law u' = u - h/f in the surface's local frame
        // (`r` carries the normalized focal length). Full transmission.
        let fl = resp.r.value;
        let srot = surface_rot(s);
        let pl = transpose(srot) * (point - s.pos.xyz);
        let dl = transpose(srot) * w.d;
        var dir = w.d;
        if (abs(dl.z) > 1e-6 && fl != 0.0) {
            dir = srot * normalize(vec3<f32>(dl.x - abs(dl.z) * pl.x / fl, dl.y - abs(dl.z) * pl.y / fl, dl.z));
        }
        push(Work(point + dir * EPS, dir, flux_end, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
    } else if (kind == RESP_MIRROR) {
        // A table with no value at `lambda` ends the branch.
        let rs = sample_spectral(resp.r, lambda);
        if (rs.y < 0.5) { return; }
        let dir = reflect3(w.d, n_oppose);
        push(Work(point + dir * EPS, dir, flux_end * clamp(rs.x, 0.0, 1.0), w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
    } else if (kind == RESP_DIELECTRIC) {
        // A table with no value at `lambda` leaves the Fresnel reflection
        // unscaled (ar = 1).
        let ars = sample_spectral(resp.ar, lambda);
        let ar = clamp(select(1.0, ars.x, ars.y > 0.5), 0.0, 1.0);
        let cos_i = clamp(-dot(n_oppose, w.d), 0.0, 1.0);
        let r = clamp(reflectance(cos_i, w.medium_n, w.medium_n) * ar, 0.0, 1.0);
        push(Work(point + w.d * EPS, w.d, flux_end * (1.0 - r), w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
        if (params.ghosts == 1u && w.gdepth < params.max_ghost && r > 0.0) {
            let dir = reflect3(w.d, n_oppose);
            push(Work(point + dir * EPS, dir, flux_end * r, w.depth + 1u, w.gdepth + 1u, w.medium_n, w.medium_alpha, (w.flags & ~SEG_TIR) | SEG_GHOST), min_flux);
        }
    } else if (kind == RESP_COATING) {
        // A real thin-film stack, read off the grid baked from the transfer
        // matrix when the scene compiled. Both branches are primary paths, like
        // explicit R/T below, unless the stack is an AR coat, whose reflection
        // is a ghost.
        let cos_i = clamp(-dot(n_oppose, w.d), 0.0, 1.0);
        let rt = sample_coating(resp, lambda, cos_i, w.medium_n);
        let r = clamp(rt.x, 0.0, 1.0);
        let t = clamp(rt.y, 0.0, 1.0);
        let stray = (resp.flags & RESP_FLAG_STRAY) != 0u;
        if (r > 0.0) {
            let dir = reflect3(w.d, n_oppose);
            if (!stray) {
                push(Work(point + dir * EPS, dir, flux_end * r, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
            } else if (params.ghosts == 1u && w.gdepth < params.max_ghost) {
                push(Work(point + dir * EPS, dir, flux_end * r, w.depth + 1u, w.gdepth + 1u, w.medium_n, w.medium_alpha, (w.flags & ~SEG_TIR) | SEG_GHOST), min_flux);
            }
        }
        if (t > 0.0) {
            push(Work(point + w.d * EPS, w.d, flux_end * t, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
        }
    } else {
        // Explicit R/T (a beam splitter or dichroic). Both branches are primary
        // paths: each keeps the parent's ghost flag and ghost depth, and is not
        // gated by the ghost switch. Either scalar missing at `lambda` ends the
        // branch. Reflected first, as in cpu3 and tracer3, so both pop in the
        // same order.
        let rs = sample_spectral(resp.r, lambda);
        let ts = sample_spectral(resp.t, lambda);
        if (rs.y < 0.5 || ts.y < 0.5) { return; }
        let r = clamp(rs.x, 0.0, 1.0);
        let t = clamp(ts.x, 0.0, 1.0);
        if (r > 0.0) {
            let dir = reflect3(w.d, n_oppose);
            push(Work(point + dir * EPS, dir, flux_end * r, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
        }
        if (t > 0.0) {
            push(Work(point + w.d * EPS, w.d, flux_end * t, w.depth + 1u, w.gdepth, w.medium_n, w.medium_alpha, w.flags & ~SEG_TIR), min_flux);
        }
    }
}
