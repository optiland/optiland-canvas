// Optiland Canvas — interactive tracer (M2).
//
// One thread per launch ray, now with a per-thread ray stack so it is fully
// non-sequential: at a dielectric interface it transmits (primary) and — when
// ghosts are on — also spawns the Fresnel back-reflection (the ghost layer,
// §5.5). Media are resolved by containment + priority (§4.2). Dispersion n(λ)
// is evaluated per element per ray (Cauchy / Sellmeier), so chromatic spread is
// a first-class output. Still f32; still cross-validated against the f64
// reference (§10.1).

struct Prim { kind: u32, _p0: u32, _p1: u32, _p2: u32, p: array<f32, 8> };

struct Elem {
    kind: u32,          // 0 dielectric, 1 mirror, 2 absorber
    disp: u32,          // 0 const, 1 cauchy, 2 sellmeier
    priority: i32,
    edge_start: u32,
    n_edges: u32,
    reflectivity: f32,
    index_const: f32,
    _pad0: f32,
    dp: array<f32, 6>,
    _pad1: f32, _pad2: f32,
};

struct Ray {
    ox: f32, oy: f32, dx: f32, dy: f32,
    flux: f32, lambda: f32, _p0: f32, _p1: f32,
};

struct Seg {
    ax: f32, ay: f32, bx: f32, by: f32,
    flux: f32, lambda: f32, flags: u32, _p0: u32,
};

struct Params {
    n_elems: u32,
    n_rays: u32,
    max_depth: u32,
    max_segs: u32,
    ambient: f32,
    bounds: f32,
    flux_eps: f32,
    ghosts: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> prims: array<Prim>;
@group(0) @binding(2) var<storage, read> elems: array<Elem>;
@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<storage, read_write> segs: array<Seg>;

const SEG_VALID: u32 = 1u;
const SEG_TIR: u32 = 2u;
const SEG_GHOST: u32 = 4u;
const EPS_O: f32 = 1e-5;
const EPS_M: f32 = 1e-4;

fn perp(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(-v.y, v.x); }
fn perp_dot(a: vec2<f32>, b: vec2<f32>) -> f32 { return a.x * b.y - a.y * b.x; }

fn line_hit(a: vec2<f32>, b: vec2<f32>, o: vec2<f32>, d: vec2<f32>, tmin: f32) -> f32 {
    let e = b - a;
    let det = perp_dot(e, d);
    if (abs(det) < 1e-12) { return -1.0; }
    let rhs = a - o;
    let t = perp_dot(e, rhs) / det;
    let u = perp_dot(d, rhs) / det;
    if (t > tmin && u >= 0.0 && u <= 1.0) { return t; }
    return -1.0;
}

fn conic_accept(t: f32, tmin: f32, y0: f32, dy: f32, z0: f32, dz: f32,
                ha: f32, z_lo: f32, z_hi: f32, ztol: f32) -> bool {
    if (t <= tmin) { return false; }
    let y = y0 + t * dy;
    let z = z0 + t * dz;
    return abs(y) <= ha + 1e-5 && z >= z_lo - ztol && z <= z_hi + ztol;
}

fn conic_hit(vertex: vec2<f32>, axis_in: vec2<f32>, c: f32, k: f32, ha: f32,
             o: vec2<f32>, d: vec2<f32>, tmin: f32) -> f32 {
    let axis = normalize(axis_in);
    let pp = perp(axis);
    let oo = o - vertex;
    let y0 = dot(oo, pp);
    let z0 = dot(oo, axis);
    let dy = dot(d, pp);
    let dz = dot(d, axis);
    let k1 = 1.0 + k;
    let aa = c * dy * dy + c * k1 * dz * dz;
    let bb = 2.0 * c * y0 * dy + 2.0 * c * k1 * z0 * dz - 2.0 * dz;
    let cc = c * y0 * y0 + c * k1 * z0 * z0 - 2.0 * z0;
    let disc_e = 1.0 - k1 * c * c * ha * ha;
    let sag_edge = c * ha * ha / (1.0 + sqrt(max(disc_e, 0.0)));
    let z_lo = min(sag_edge, 0.0);
    let z_hi = max(sag_edge, 0.0);
    let ztol = 1e-4 * (1.0 + abs(ha));
    var best = -1.0;
    if (abs(aa) < 1e-12) {
        if (abs(bb) > 1e-12) {
            let t = -cc / bb;
            if (conic_accept(t, tmin, y0, dy, z0, dz, ha, z_lo, z_hi, ztol)) { best = t; }
        }
    } else {
        let disc = bb * bb - 4.0 * aa * cc;
        if (disc >= 0.0) {
            let sq = sqrt(disc);
            let sgn = select(-1.0, 1.0, bb >= 0.0);
            let q = -0.5 * (bb + sgn * sq);
            let t1 = q / aa;
            if (conic_accept(t1, tmin, y0, dy, z0, dz, ha, z_lo, z_hi, ztol)) { best = t1; }
            if (abs(q) > 1e-20) {
                let t2 = cc / q;
                if (conic_accept(t2, tmin, y0, dy, z0, dz, ha, z_lo, z_hi, ztol)) {
                    if (best < 0.0 || t2 < best) { best = t2; }
                }
            }
        }
    }
    return best;
}

fn prim_hit(pi: u32, o: vec2<f32>, d: vec2<f32>, tmin: f32) -> f32 {
    let pr = prims[pi];
    if (pr.kind == 0u) {
        return line_hit(vec2<f32>(pr.p[0], pr.p[1]), vec2<f32>(pr.p[2], pr.p[3]), o, d, tmin);
    }
    return conic_hit(vec2<f32>(pr.p[0], pr.p[1]), vec2<f32>(pr.p[2], pr.p[3]),
                     pr.p[4], pr.p[5], pr.p[6], o, d, tmin);
}

fn prim_normal(pi: u32, pt: vec2<f32>) -> vec2<f32> {
    let pr = prims[pi];
    if (pr.kind == 0u) {
        return normalize(perp(vec2<f32>(pr.p[2] - pr.p[0], pr.p[3] - pr.p[1])));
    }
    let vertex = vec2<f32>(pr.p[0], pr.p[1]);
    let axis = normalize(vec2<f32>(pr.p[2], pr.p[3]));
    let pp = perp(axis);
    let oo = pt - vertex;
    let n = (2.0 * pr.p[4] * dot(oo, pp)) * pp
          + (2.0 * pr.p[4] * (1.0 + pr.p[5]) * dot(oo, axis) - 2.0) * axis;
    return normalize(n);
}

fn conic_count(pi: u32, o: vec2<f32>, d: vec2<f32>) -> u32 {
    let pr = prims[pi];
    let axis = normalize(vec2<f32>(pr.p[2], pr.p[3]));
    let pp = perp(axis);
    let c = pr.p[4]; let k = pr.p[5]; let ha = pr.p[6];
    let oo = o - vec2<f32>(pr.p[0], pr.p[1]);
    let y0 = dot(oo, pp); let z0 = dot(oo, axis);
    let dy = dot(d, pp); let dz = dot(d, axis);
    let k1 = 1.0 + k;
    let aa = c * dy * dy + c * k1 * dz * dz;
    let bb = 2.0 * c * y0 * dy + 2.0 * c * k1 * z0 * dz - 2.0 * dz;
    let cc = c * y0 * y0 + c * k1 * z0 * z0 - 2.0 * z0;
    let disc_e = 1.0 - k1 * c * c * ha * ha;
    let sag_edge = c * ha * ha / (1.0 + sqrt(max(disc_e, 0.0)));
    let z_lo = min(sag_edge, 0.0); let z_hi = max(sag_edge, 0.0);
    let ztol = 1e-4 * (1.0 + abs(ha));
    var n = 0u;
    if (abs(aa) < 1e-12) {
        if (abs(bb) > 1e-12 && conic_accept(-cc / bb, 1e-7, y0, dy, z0, dz, ha, z_lo, z_hi, ztol)) { n = 1u; }
    } else {
        let disc = bb * bb - 4.0 * aa * cc;
        if (disc >= 0.0) {
            let sq = sqrt(disc);
            let sgn = select(-1.0, 1.0, bb >= 0.0);
            let q = -0.5 * (bb + sgn * sq);
            if (conic_accept(q / aa, 1e-7, y0, dy, z0, dz, ha, z_lo, z_hi, ztol)) { n = n + 1u; }
            if (abs(q) > 1e-20 && conic_accept(cc / q, 1e-7, y0, dy, z0, dz, ha, z_lo, z_hi, ztol)) { n = n + 1u; }
        }
    }
    return n;
}

fn cross_count(e: u32, pt: vec2<f32>) -> u32 {
    let dir = normalize(vec2<f32>(1.0, 0.31830987)); // (1, 1/π); matches cpu.rs
    let el = elems[e];
    var count = 0u;
    for (var pi = el.edge_start; pi < el.edge_start + el.n_edges; pi = pi + 1u) {
        let pr = prims[pi];
        if (pr.kind == 0u) {
            if (line_hit(vec2<f32>(pr.p[0], pr.p[1]), vec2<f32>(pr.p[2], pr.p[3]), pt, dir, 1e-7) > 0.0) {
                count = count + 1u;
            }
        } else {
            count = count + conic_count(pi, pt, dir);
        }
    }
    return count;
}

fn region_elem(pt: vec2<f32>) -> i32 {
    var best = -1;
    var best_pri = -2147483647;
    for (var e = 0u; e < params.n_elems; e = e + 1u) {
        if ((cross_count(e, pt) & 1u) == 1u) {
            if (best < 0 || elems[e].priority >= best_pri) {
                best = i32(e);
                best_pri = elems[e].priority;
            }
        }
    }
    return best;
}

fn element_index(e: i32, lambda: f32) -> f32 {
    if (e < 0) { return params.ambient; }
    let el = elems[u32(e)];
    if (el.kind != 0u) { return params.ambient; }
    if (el.disp == 2u) {
        let l2 = lambda * lambda;
        let n2 = 1.0
            + el.dp[0] * l2 / (l2 - el.dp[3])
            + el.dp[1] * l2 / (l2 - el.dp[4])
            + el.dp[2] * l2 / (l2 - el.dp[5]);
        return sqrt(n2);
    } else if (el.disp == 1u) {
        return el.dp[0] + el.dp[1] / (lambda * lambda);
    }
    return el.index_const;
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

fn refract(d: vec2<f32>, n: vec2<f32>, n1: f32, n2: f32) -> vec4<f32> {
    let eta = n1 / n2;
    let cosi = -dot(n, d);
    let kk = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (kk < 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    let t = eta * d + (eta * cosi - sqrt(kk)) * n;
    return vec4<f32>(t.x, t.y, 1.0, 0.0);
}

fn reflectv(d: vec2<f32>, n: vec2<f32>) -> vec2<f32> { return d - 2.0 * dot(d, n) * n; }

// Per-thread ray stack.
struct Work { o: vec2<f32>, d: vec2<f32>, flux: f32, depth: u32, ghost: u32 };
var<private> stack: array<Work, 48>;
var<private> sp: u32;

fn push(o: vec2<f32>, d: vec2<f32>, flux: f32, depth: u32, ghost: u32, min_flux: f32) {
    if (depth >= params.max_depth || flux < min_flux) { return; }
    if (sp < 48u) {
        stack[sp] = Work(o, d, flux, depth, ghost);
        sp = sp + 1u;
    }
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rid = gid.x;
    if (rid >= params.n_rays) { return; }

    let r = rays[rid];
    let lambda = r.lambda;
    let base = rid * params.max_segs;
    let min_flux = params.flux_eps * r.flux;

    sp = 0u;
    stack[0] = Work(vec2<f32>(r.ox, r.oy), normalize(vec2<f32>(r.dx, r.dy)), r.flux, 0u, 0u);
    sp = 1u;

    var seg_count = 0u;
    loop {
        if (sp == 0u || seg_count >= params.max_segs) { break; }
        sp = sp - 1u;
        let w = stack[sp];
        if (w.depth >= params.max_depth || w.flux < min_flux) { continue; }

        let o = w.o;
        let d = w.d;
        let n_here = element_index(region_elem(o + EPS_M * d), lambda);

        var best_t = params.bounds;
        var best_pi = 0xffffffffu;
        var best_elem = -1;
        for (var e = 0u; e < params.n_elems; e = e + 1u) {
            let el = elems[e];
            for (var pi = el.edge_start; pi < el.edge_start + el.n_edges; pi = pi + 1u) {
                let t = prim_hit(pi, o, d, EPS_O);
                if (t > 0.0 && t < best_t) {
                    best_t = t;
                    best_pi = pi;
                    best_elem = i32(e);
                }
            }
        }

        var gflag = 0u;
        if (w.ghost == 1u) { gflag = SEG_GHOST; }

        if (best_elem < 0) {
            let b = o + best_t * d;
            segs[base + seg_count] = Seg(o.x, o.y, b.x, b.y, w.flux, lambda, SEG_VALID | gflag, 0u);
            seg_count = seg_count + 1u;
            continue;
        }

        let hit = o + best_t * d;
        let si = base + seg_count;
        segs[si] = Seg(o.x, o.y, hit.x, hit.y, w.flux, lambda, SEG_VALID | gflag, 0u);
        seg_count = seg_count + 1u;

        var nrm = prim_normal(best_pi, hit);
        if (dot(nrm, d) > 0.0) { nrm = -nrm; }

        let far = region_elem(hit - EPS_M * nrm);
        let hit_el = elems[u32(best_elem)];

        var far_absorb = false;
        if (far >= 0) { far_absorb = elems[u32(far)].kind == 2u; }
        if (hit_el.kind == 2u || far_absorb) { continue; }

        var mirror = -1.0;
        if (hit_el.kind == 1u) { mirror = hit_el.reflectivity; }
        else if (far >= 0 && elems[u32(far)].kind == 1u) { mirror = elems[u32(far)].reflectivity; }

        if (mirror >= 0.0) {
            let dr = reflectv(d, nrm);
            push(hit + EPS_O * dr, dr, w.flux * mirror, w.depth + 1u, w.ghost, min_flux);
            continue;
        }

        let n1 = n_here;
        let n2 = element_index(far, lambda);
        let res = refract(d, nrm, n1, n2);
        let dr = reflectv(d, nrm);
        if (res.z < 0.5) {
            // Total internal reflection: primary path reflects; flag the surface.
            segs[si].flags = segs[si].flags | SEG_TIR;
            push(hit + EPS_O * dr, dr, w.flux, w.depth + 1u, w.ghost, min_flux);
        } else {
            let cosi = -dot(nrm, d);
            let rr = clamp(reflectance(cosi, n1, n2), 0.0, 1.0);
            let dt = vec2<f32>(res.x, res.y);
            push(hit + EPS_O * dt, dt, w.flux * (1.0 - rr), w.depth + 1u, w.ghost, min_flux);
            if (params.ghosts == 1u && rr > 0.0) {
                push(hit + EPS_O * dr, dr, w.flux * rr, w.depth + 1u, 1u, min_flux);
            }
        }
    }
}
