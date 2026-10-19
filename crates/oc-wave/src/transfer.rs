//! A field through a refractive body or a curved mirror, by geometric field
//! tracing (W6).
//!
//! Inside a lens the field does not diffract appreciably: a lens a few
//! millimetres thick is many Fresnel lengths short of changing a beam by
//! diffraction. What it does is delay each part of the field by the optical
//! path along its ray and move it across the beam where the ray goes. F.
//! Wyrowski and M. Kuhn, "Introduction to field tracing," *J. Mod. Opt.*
//! 58(5–6), 449–466 (2011), doi:10.1080/09500340.2010.532237, call this the
//! geometric field tracing of an element; it is the step between two planes
//! that the propagators then join by diffraction.
//!
//! The field arrives across the incoming chief ray at an *entry plane* in
//! front of the element and leaves across the outgoing chief ray at an *exit
//! plane* behind it. The f64 tracer carries one ray from each point of a grid
//! on the entry plane, parallel to the chief ray, through a copy of the scene
//! that holds the element alone. Each ray gives where it crosses the exit
//! plane, its optical path and its power transmittance. The field at an exit
//! point is then the field at the entry point whose ray lands there,
//!
//! ```text
//! U_out(x') = U_in(x) · √(T / |∂x'/∂x|) · exp(ik₀ OPL(x)),
//! ```
//!
//! so power is conserved ray tube by ray tube, less what the faces reflect,
//! and the field keeps its phase on the same clock as free space (the samples
//! carry the whole phase, `OPL` from the entry plane included).
//!
//! The map `x → x'` and the path are represented as a fitted affine map and
//! quadratic plus the residuals on the ray grid, interpolated bilinearly. A
//! lens's path is nearly quadratic and its map nearly affine, so the residuals
//! are small and smooth. Light whose ray misses the element's faces, or does
//! anything the chief ray did not, is taken as stopped by the mount. Where
//! that region's edge, the rim, is star-shaped about the chief ray, it is
//! found by bisection on traced rays and laid on the arriving field as a
//! polygon with exact cell coverage, so the rim diffracts at the field's own
//! sampling rather than the ray grid's; the map is carried a couple of rays
//! past it, where the field is already zero, so no ray-grid cell is lost at
//! the edge. Otherwise the rim is resolved to the ray grid, and the report
//! says so.
//!
//! The rays are traced along the arriving field's own wavefront: a sphere and
//! a tilt fitted to its phase gradient, weighted by its intensity, so a beam
//! that arrives converging or diverging is traced with rays that do. (Rays
//! traced parallel to the chief ray would carry the field's own curvature
//! through the element's map as if it were collimated light, and move the
//! focus.) What the fit leaves out, the field's aberrations, is not traced;
//! the report says how far its rays depart from the fitted sphere.

use crate::path::Beam;
use crate::{Cx, Field2, WaveError};
use oc_core::scene3::Scene3;
use oc_core::vec3::Ray3;
use oc_core::Vec3;
use oc_trace::{trace3, LaunchRay3, Segment3, TraceConfig3};

/// What a transfer delivered besides the field.
pub(crate) struct TransferStats {
    /// The chief ray's optical path from the entry plane to the exit plane, mm.
    pub opl_chief: f64,
    /// Rays traced, and how many passed.
    pub rays: usize,
    pub passed: usize,
    /// Power stopped outside the traced aperture, and reflected away by the
    /// faces (the scalar, unpolarized transmittance).
    pub stopped: f64,
    pub reflected: f64,
    /// Stopped by the rim laid on the arriving field, and diffracted by its
    /// edge beyond the grid's band; whether the rim was resolved that way.
    pub rim_stopped: f64,
    pub rim_beyond: f64,
    pub rim_resolved: bool,
    /// The arriving power, after the rim, that the map carries on.
    pub carried: f64,
    /// Power whose rays land outside the exit window.
    pub left_window: f64,
    /// The largest move across the beam between entry and exit, mm.
    pub walk_off_mm: f64,
    /// The arriving wavefront's fitted curvature, 1/mm (positive diverging).
    pub curvature: f64,
    /// The steepest departure of the arriving field's rays from the fitted
    /// sphere and tilt, rad.
    pub arriving_ray_angle: f64,
    /// The largest phase step between neighbouring samples of the field that
    /// leaves, rad.
    pub max_phase_step: f64,
}

/// One ray through the element: where it crosses the exit plane, in the exit
/// frame, its optical path from the entry plane, and its transmittance.
#[derive(Clone, Copy)]
struct RayOut {
    x: f64,
    y: f64,
    opl: f64,
    t: f64,
}

/// Where a ray, after exactly one interaction with the element, crosses the
/// exit plane (through `p` with normal `n`, the outgoing direction): the
/// crossing point, the optical path there, and the power left. Of several
/// branches (a splitting surface), the one that best continues along `n`.
fn crossing(segs: &[Segment3], p: Vec3, n: Vec3) -> Option<(Vec3, f64, f64)> {
    let depth = |s: &Segment3| {
        let mut d = 0;
        let mut cur = s.parent_path_id;
        while let Some(id) = cur {
            let Some(q) = segs.iter().find(|x| x.path_id == id) else { break };
            d += 1;
            cur = q.parent_path_id;
        }
        d
    };
    segs.iter()
        .filter(|s| depth(s) == 1)
        .filter_map(|s| {
            let len = (s.b - s.a).length();
            if len <= 0.0 {
                return None;
            }
            let d = (s.b - s.a) / len;
            let (ha, hb) = ((s.a - p).dot(n), (s.b - p).dot(n));
            if !(ha <= 0.0 && hb > 0.0) || d.dot(n) <= 0.0 {
                return None;
            }
            let x = s.a + d * (-ha / d.dot(n));
            let index = (s.opl1_mm - s.opl0_mm) / len;
            Some((d.dot(n), x, s.opl0_mm + index * (x - s.a).length(), s.flux_start))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, x, opl, t)| (x, opl, t))
}

/// Least squares of `y ≈ Σ c_k f_k`, by the normal equations (a handful of
/// terms on well-scaled coordinates).
fn lstsq<const K: usize>(rows: &[([f64; K], f64)]) -> Option<[f64; K]> {
    let mut a = [[0.0; K]; K];
    let mut b = [0.0; K];
    for (f, y) in rows {
        for i in 0..K {
            b[i] += f[i] * y;
            for j in 0..K {
                a[i][j] += f[i] * f[j];
            }
        }
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..K {
        let p = (c..K).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())?;
        if a[p][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        let pivot = a[c];
        for r in c + 1..K {
            let f = a[r][c] / pivot[c];
            for (x, y) in a[r][c..].iter_mut().zip(&pivot[c..]) {
                *x -= f * y;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; K];
    for c in (0..K).rev() {
        let s: f64 = (c + 1..K).map(|k| a[c][k] * x[k]).sum();
        x[c] = (b[c] - s) / a[c][c];
    }
    Some(x)
}

/// Solve the 3 × 3 system `m·x = v`, by Gaussian elimination with pivoting.
fn square3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())?;
        if a[p][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        let pivot = a[c];
        for r in c + 1..3 {
            let f = a[r][c] / pivot[c];
            for (x, y) in a[r][c..].iter_mut().zip(&pivot[c..]) {
                *x -= f * y;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 3];
    for c in (0..3).rev() {
        let s: f64 = (c + 1..3).map(|k| a[c][k] * x[k]).sum();
        x[c] = (b[c] - s) / a[c][c];
    }
    Some(x)
}

/// The traced map on its grid: fits plus residuals.
struct Map {
    g: usize,
    h: f64,
    /// `x' ≈ ax·(a, b, 1)`, `y' ≈ ay·(a, b, 1)`, `opl ≈ q·(1, a, b, a², ab, b²)`,
    /// in coordinates scaled by `s`.
    s: f64,
    ax: [f64; 3],
    ay: [f64; 3],
    q: [f64; 6],
    rays: Vec<Option<RayOut>>,
    rx: Vec<f64>,
    ry: Vec<f64>,
    rp: Vec<f64>,
}

impl Map {
    fn fit_x(&self, a: f64, b: f64) -> (f64, f64) {
        let (u, v) = (a / self.s, b / self.s);
        (self.ax[0] * u + self.ax[1] * v + self.ax[2], self.ay[0] * u + self.ay[1] * v + self.ay[2])
    }
    fn fit_p(&self, a: f64, b: f64) -> f64 {
        let (u, v) = (a / self.s, b / self.s);
        let q = &self.q;
        q[0] + q[1] * u + q[2] * v + q[3] * u * u + q[4] * u * v + q[5] * v * v
    }
    /// The fitted map's Jacobian, `∂x'/∂(a, b)`.
    fn fit_jac(&self) -> [[f64; 2]; 2] {
        [[self.ax[0] / self.s, self.ax[1] / self.s], [self.ay[0] / self.s, self.ay[1] / self.s]]
    }
    /// The residuals, transmittance and their derivatives at `(a, b)`, bilinear
    /// in its cell; `None` if a corner's ray failed or it is off the grid.
    #[allow(clippy::type_complexity)]
    fn local(&self, a: f64, b: f64) -> Option<([f64; 4], [[f64; 2]; 3])> {
        let g = self.g;
        let half = (g - 1) as f64 / 2.0;
        let (fi, fj) = (a / self.h + half, b / self.h + half);
        if !(fi >= 0.0 && fj >= 0.0 && fi <= (g - 1) as f64 && fj <= (g - 1) as f64) {
            return None;
        }
        let (i0, j0) = ((fi.floor() as usize).min(g - 2), (fj.floor() as usize).min(g - 2));
        let (s, t) = (fi - i0 as f64, fj - j0 as f64);
        let k = |i: usize, j: usize| j * g + i;
        let corners = [k(i0, j0), k(i0 + 1, j0), k(i0, j0 + 1), k(i0 + 1, j0 + 1)];
        let mut tt = [0.0; 4];
        for (n, &c) in corners.iter().enumerate() {
            tt[n] = self.rays[c]?.t;
        }
        let bil = |v: &[f64]| {
            let (v00, v10, v01, v11) = (v[corners[0]], v[corners[1]], v[corners[2]], v[corners[3]]);
            let val = v00 * (1.0 - s) * (1.0 - t) + v10 * s * (1.0 - t) + v01 * (1.0 - s) * t + v11 * s * t;
            let da = ((v10 - v00) * (1.0 - t) + (v11 - v01) * t) / self.h;
            let db = ((v01 - v00) * (1.0 - s) + (v11 - v10) * s) / self.h;
            (val, [da, db])
        };
        let (rx, dx) = bil(&self.rx);
        let (ry, dy) = bil(&self.ry);
        let (rp, dp) = bil(&self.rp);
        let tv = tt[0] * (1.0 - s) * (1.0 - t) + tt[1] * s * (1.0 - t) + tt[2] * (1.0 - s) * t + tt[3] * s * t;
        Some(([rx, ry, rp, tv], [dx, dy, dp]))
    }
    /// Where the ray from `(a, b)` lands, its path, transmittance and the
    /// map's Jacobian determinant there.
    fn at(&self, a: f64, b: f64) -> Option<(f64, f64, f64, f64, f64)> {
        let ([rx, ry, rp, t], [dx, dy, _]) = self.local(a, b)?;
        let (fx, fy) = self.fit_x(a, b);
        let j = self.fit_jac();
        let det = (j[0][0] + dx[0]) * (j[1][1] + dy[1]) - (j[0][1] + dx[1]) * (j[1][0] + dy[0]);
        Some((fx + rx, fy + ry, self.fit_p(a, b) + rp, t, det))
    }
    /// The entry point whose ray lands on `(x, y)`, by fixed-point iteration
    /// on the fitted map's inverse.
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let j = self.fit_jac();
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]];
        let c = (self.ax[2], self.ay[2]);
        let solve = |rx: f64, ry: f64| {
            let (u, v) = (x - c.0 - rx, y - c.1 - ry);
            (inv[0][0] * u + inv[0][1] * v, inv[1][0] * u + inv[1][1] * v)
        };
        let (mut a, mut b) = solve(0.0, 0.0);
        for _ in 0..12 {
            let ([rx, ry, ..], _) = self.local(a, b)?;
            let (na, nb) = solve(rx, ry);
            let step = ((na - a).powi(2) + (nb - b).powi(2)).sqrt();
            a = na;
            b = nb;
            if step < 1e-9 * self.h {
                break;
            }
        }
        let (xx, yy, ..) = self.at(a, b)?;
        (((xx - x).powi(2) + (yy - y).powi(2)).sqrt() < 1e-6 * self.h).then_some((a, b))
    }
}

/// A field's wavefront: direction cosines `t = (α + c·a, β + c·b)` across
/// the beam (carrier included), fitted by intensity-weighted least squares to
/// the phase gradient where the field is bright. Returns `[c, α, β]` and the
/// largest departure of a bright sample's direction from the fit; the
/// carrier alone when nothing is bright enough to fit.
pub(crate) fn fit_sphere(u: &Field2) -> ([f64; 3], f64) {
    let gu = u.grid;
    let k_in = u.k_mm();
    let mut top: f64 = 0.0;
    for c in &u.data {
        top = top.max(c.norm_sqr());
    }
    // (a, b, tx, ty, weight) at each bright sample.
    let mut grads: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for j in 1..gu.n[1].saturating_sub(1) {
        for i in 1..gu.n[0].saturating_sub(1) {
            let w = u.at(i, j).norm_sqr();
            if w < 1e-3 * top {
                continue;
            }
            let gx = (u.at(i + 1, j) * u.at(i - 1, j).conj()).arg() / (2.0 * gu.pitch_mm[0]);
            let gy = (u.at(i, j + 1) * u.at(i, j - 1).conj()).arg() / (2.0 * gu.pitch_mm[1]);
            let (a, b) = gu.xy(i, j);
            grads.push((a, b, gx / k_in + u.carrier[0], gy / k_in + u.carrier[1], w));
        }
    }
    // tx = c·a + α and ty = c·b + β, unknowns (c, α, β), weighted by intensity.
    let [curv, alpha, beta] = {
        let (mut m, mut v) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
        for &(a, b, tx, ty, w) in &grads {
            for (f, y) in [([a, 1.0, 0.0], tx), ([b, 0.0, 1.0], ty)] {
                for r in 0..3 {
                    v[r] += w * f[r] * y;
                    for q in 0..3 {
                        m[r][q] += w * f[r] * f[q];
                    }
                }
            }
        }
        square3(m, v).unwrap_or([0.0, u.carrier[0], u.carrier[1]])
    };
    let mut angle: f64 = 0.0;
    for &(a, b, tx, ty, _) in &grads {
        angle = angle.max(((tx - curv * a - alpha).powi(2) + (ty - curv * b - beta).powi(2)).sqrt());
    }
    ([curv, alpha, beta], angle)
}

/// A field read between its samples. Bilinear interpolation of a complex
/// field whose phase turns by `φ` from one sample to the next lowers its
/// amplitude between them: `|(1 − s) + s·e^{iφ}|² = 1 − 2s(1 − s)(1 − cos φ)`,
/// a power loss of `(1 − cos φ)/3 ≈ φ²/6` per axis averaged over a cell. A
/// beam converging at a few hundredths of a radian turns most of a radian
/// per sample at its edge, so the sphere and tilt the field is fitted to are
/// taken out of the samples first, and put back exactly at the point read:
/// what is interpolated is the slowly varying rest.
pub(crate) struct Smooth {
    d: Field2,
    k: f64,
    s: [f64; 3],
}

impl Smooth {
    /// `u` with its own fitted sphere taken out.
    pub(crate) fn new(u: &Field2) -> Self {
        let ([c, a, b], _) = fit_sphere(u);
        Self::with(u, [c, a - u.carrier[0], b - u.carrier[1]])
    }

    /// `u` with the phase `k·(α·a + β·b + c·(a² + b²)/2)` taken out, `s` =
    /// `[c, α, β]` relative to the carrier (the samples' own phase).
    pub(crate) fn with(u: &Field2, s: [f64; 3]) -> Self {
        let k = u.k_mm();
        let mut d = u.clone();
        let g = u.grid;
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (a, b) = g.xy(i, j);
                let idx = g.idx(i, j);
                d.data[idx] *= Cx::expi(-smooth_phase(k, s, a, b));
            }
        }
        Self { d, k, s }
    }

    /// The field's sample value at `(a, b)` (carrier excluded); zero outside.
    pub(crate) fn at(&self, a: f64, b: f64) -> Cx {
        crate::path::sample(&self.d, a, b) * Cx::expi(smooth_phase(self.k, self.s, a, b))
    }
}

fn smooth_phase(k: f64, s: [f64; 3], a: f64, b: f64) -> f64 {
    k * (s[1] * a + s[2] * b + 0.5 * s[0] * (a * a + b * b))
}

/// The edge of the region of the entry plane whose rays pass the element, as
/// a polygon of 720 vertices about the chief ray: along each azimuth, the
/// first ray of the grid that fails brackets the edge, which bisection on
/// traced rays then finds to a millionth of the ray spacing. `None` when the
/// region is not star-shaped about the chief ray (a ray passes again beyond
/// the edge), or the bracket does not hold.
fn rim_polygon(trace: &dyn Fn(f64, f64) -> Option<RayOut>, grid: &[Option<RayOut>], g: usize, h: f64) -> Option<Vec<[f64; 2]>> {
    let half = (g - 1) as f64 / 2.0;
    let r_max = half * h * std::f64::consts::SQRT_2;
    let on_grid = |a: f64, b: f64| -> Option<bool> {
        let (i, j) = ((a / h + half).round(), (b / h + half).round());
        if i < 0.0 || j < 0.0 || i > (g - 1) as f64 || j > (g - 1) as f64 {
            return None;
        }
        Some(grid[j as usize * g + i as usize].is_some())
    };
    const M: usize = 720;
    let mut poly = Vec::with_capacity(M);
    for k in 0..M {
        let th = 2.0 * std::f64::consts::PI * k as f64 / M as f64;
        let (c, s) = (th.cos(), th.sin());
        let mut r_fail = None;
        let mut r = 0.0;
        while r <= r_max {
            match on_grid(r * c, r * s) {
                None => break,
                Some(true) => {}
                Some(false) => {
                    r_fail = Some(r);
                    break;
                }
            }
            r += 0.5 * h;
        }
        let Some(rf) = r_fail else {
            // Rays pass out to the window's edge along this azimuth.
            poly.push([2.0 * r_max * c, 2.0 * r_max * s]);
            continue;
        };
        let mut q = rf;
        while q <= r_max {
            if on_grid(q * c, q * s) == Some(true) {
                return None;
            }
            q += 0.5 * h;
        }
        let mut lo = (rf - 1.5 * h).max(0.0);
        while lo > 0.0 && trace(lo * c, lo * s).is_none() {
            lo = (lo - h).max(0.0);
        }
        trace(lo * c, lo * s)?;
        let mut hi = rf + 0.5 * h;
        let mut tries = 0;
        while trace(hi * c, hi * s).is_some() {
            hi += h;
            tries += 1;
            if tries > 4 {
                return None;
            }
        }
        while hi - lo > 1e-6 * h {
            let mid = 0.5 * (lo + hi);
            if trace(mid * c, mid * s).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let re = 0.5 * (lo + hi);
        poly.push([re * c, re * s]);
    }
    Some(poly)
}

/// Carry `u` (across `inc`, at the entry plane) through the element `probe`
/// holds, to the exit plane across `out`, in a medium of index `n_out`, with
/// `rays` rays across the field's window.
pub(crate) fn transfer(
    probe: &Scene3,
    u: &Field2,
    inc: &Beam,
    out: &Beam,
    n_out: f64,
    rays: usize,
) -> Result<(Field2, TransferStats), WaveError> {
    let lam = u.lambda_um;
    let cfg = TraceConfig3 {
        ghosts: false,
        polarization: false,
        bounds_mm: oc_trace::bounds_for_scene3(probe),
        ..TraceConfig3::default()
    };
    // The arriving wavefront.
    let gu = u.grid;
    let k_in = u.k_mm();
    let ([curv, alpha, beta], angle) = fit_sphere(u);
    let dir_at = |a: f64, b: f64| -> Option<Vec3> {
        let (tx, ty) = (alpha + curv * a, beta + curv * b);
        let tz2 = 1.0 - tx * tx - ty * ty;
        (tz2 > 0.0).then(|| (inc.x * tx + inc.y * ty + inc.d * tz2.sqrt()).normalize())
    };
    let trace = |a: f64, b: f64| -> Option<RayOut> {
        let ray = Ray3::new(inc.point(a, b), dir_at(a, b)?).ok()?;
        let r = trace3(probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
        let (x, opl, t) = crossing(&r.segments, out.origin, out.d)?;
        let (xx, yy) = out.coords(x);
        Some(RayOut { x: xx, y: yy, opl, t })
    };
    let chief = trace(0.0, 0.0)
        .ok_or_else(|| WaveError::OutOfRange("the chief ray does not cross the surface to its exit plane".into()))?;

    // The ray grid: odd, so the chief ray is its centre, spanning the window.
    let g = (rays.max(9) | 1).min(1025);
    let w = u.grid.width_mm(0).max(u.grid.width_mm(1));
    let h = w / (g - 1) as f64;
    let half = (g - 1) as f64 / 2.0;
    let mut grid: Vec<Option<RayOut>> = Vec::with_capacity(g * g);
    for j in 0..g {
        for i in 0..g {
            let (a, b) = ((i as f64 - half) * h, (j as f64 - half) * h);
            grid.push(trace(a, b));
        }
    }
    let passed = grid.iter().filter(|r| r.is_some()).count();
    if passed < 9 {
        return Err(WaveError::OutOfRange("fewer than nine rays pass the element".into()));
    }
    // The rim at the field's own sampling, laid on the arriving field.
    let rim = if passed < g * g { rim_polygon(&trace, &grid, g, h) } else { None };
    let mut um = u.clone();
    let (rim_stopped, rim_beyond) = match &rim {
        Some(poly) => crate::elements::apply_coverage(&mut um, &crate::elements::polygon_coverage(&u.grid, poly))?,
        None => (0.0, 0.0),
    };
    let u = &um;
    // Fit the map and the path on coordinates scaled to the window.
    let s = 0.5 * w;
    let mut rows_x = Vec::with_capacity(passed);
    let mut rows_y = Vec::with_capacity(passed);
    let mut rows_p = Vec::with_capacity(passed);
    for j in 0..g {
        for i in 0..g {
            if let Some(r) = grid[j * g + i] {
                let (uu, vv) = ((i as f64 - half) * h / s, (j as f64 - half) * h / s);
                rows_x.push(([uu, vv, 1.0], r.x));
                rows_y.push(([uu, vv, 1.0], r.y));
                rows_p.push(([1.0, uu, vv, uu * uu, uu * vv, vv * vv], r.opl - chief.opl));
            }
        }
    }
    let fail = || WaveError::OutOfRange("the traced map could not be fitted".into());
    let ax = lstsq(&rows_x).ok_or_else(fail)?;
    let ay = lstsq(&rows_y).ok_or_else(fail)?;
    let q = lstsq(&rows_p).ok_or_else(fail)?;
    let mut map = Map { g, h, s, ax, ay, q, rays: grid, rx: vec![0.0; g * g], ry: vec![0.0; g * g], rp: vec![0.0; g * g] };
    let mut walk: f64 = 0.0;
    for j in 0..g {
        for i in 0..g {
            let k = j * g + i;
            if let Some(r) = map.rays[k] {
                let (a, b) = ((i as f64 - half) * h, (j as f64 - half) * h);
                let (fx, fy) = map.fit_x(a, b);
                map.rx[k] = r.x - fx;
                map.ry[k] = r.y - fy;
                map.rp[k] = (r.opl - chief.opl) - map.fit_p(a, b);
                walk = walk.max(((r.x - a).powi(2) + (r.y - b).powi(2)).sqrt());
            }
        }
    }
    // Past a resolved rim, the map carried two rays on: each failed ray next
    // to a passing one takes its neighbours' mean residuals and transmittance.
    if rim.is_some() {
        for _ in 0..2 {
            let live: Vec<bool> = map.rays.iter().map(|r| r.is_some()).collect();
            for j in 0..g {
                for i in 0..g {
                    let k = j * g + i;
                    if live[k] {
                        continue;
                    }
                    let (mut n, mut sx, mut sy, mut sp, mut st) = (0.0, 0.0, 0.0, 0.0, 0.0);
                    for dj in -1i64..=1 {
                        for di in -1i64..=1 {
                            let (ii, jj) = (i as i64 + di, j as i64 + dj);
                            if ii < 0 || jj < 0 || ii >= g as i64 || jj >= g as i64 {
                                continue;
                            }
                            let kk = jj as usize * g + ii as usize;
                            if let (true, Some(r)) = (live[kk], map.rays[kk]) {
                                n += 1.0;
                                sx += map.rx[kk];
                                sy += map.ry[kk];
                                sp += map.rp[kk];
                                st += r.t;
                            }
                        }
                    }
                    if n > 0.0 {
                        let (a, b) = ((i as f64 - half) * h, (j as f64 - half) * h);
                        let (fx, fy) = map.fit_x(a, b);
                        let fp = map.fit_p(a, b);
                        let (rx, ry, rp) = (sx / n, sy / n, sp / n);
                        map.rx[k] = rx;
                        map.ry[k] = ry;
                        map.rp[k] = rp;
                        map.rays[k] = Some(RayOut { x: fx + rx, y: fy + ry, opl: chief.opl + fp + rp, t: st / n });
                    }
                }
            }
        }
    }

    // The arriving field, carrier included, read between its samples with
    // the fitted sphere taken out (`Smooth`).
    let smooth = Smooth::with(u, [curv, alpha - u.carrier[0], beta - u.carrier[1]]);
    let full = |a: f64, b: f64| smooth.at(a, b) * Cx::expi(k_in * (u.carrier[0] * a + u.carrier[1] * b));

    // Where the arriving power goes: stopped, reflected, out of the window.
    let cell = gu.cell_area_mm2();
    let (mut stopped, mut reflected, mut left) = (0.0, 0.0, 0.0);
    let (wx, wy) = (0.5 * gu.width_mm(0), 0.5 * gu.width_mm(1));
    for j in 0..gu.n[1] {
        for i in 0..gu.n[0] {
            let p = u.at(i, j).norm_sqr() * cell;
            if p == 0.0 {
                continue;
            }
            let (a, b) = gu.xy(i, j);
            match map.at(a, b) {
                None => stopped += p,
                Some((x, y, _, t, _)) => {
                    reflected += p * (1.0 - t.clamp(0.0, 1.0));
                    if (x - gu.center_mm[0]).abs() > wx || (y - gu.center_mm[1]).abs() > wy {
                        left += p * t.clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    // The field that leaves, sample by sample across the exit plane.
    let k0 = 2.0 * std::f64::consts::PI / (lam / 1000.0);
    let mut v = u.clone();
    v.carrier = [0.0, 0.0];
    v.n_medium = n_out;
    v.frame = out.iso();
    v.opl_ref_mm = u.opl_ref_mm + chief.opl;
    let mut paths = vec![f64::NAN; gu.n[0] * gu.n[1]];
    for j in 0..gu.n[1] {
        for i in 0..gu.n[0] {
            let (x, y) = gu.xy(i, j);
            let idx = gu.idx(i, j);
            v.data[idx] = match map.inverse(x, y).and_then(|(a, b)| map.at(a, b).map(|r| (a, b, r))) {
                Some((a, b, (_, _, opl, t, det))) if det.abs() > 1e-12 => {
                    paths[idx] = opl;
                    full(a, b) * (t.clamp(0.0, 1.0) / det.abs()).sqrt() * Cx::expi(k0 * (chief.opl + opl))
                }
                _ => Cx::ZERO,
            };
        }
    }
    let mut step: f64 = 0.0;
    for j in 0..gu.n[1] {
        for i in 0..gu.n[0].saturating_sub(1) {
            let (p, q) = (paths[gu.idx(i, j)], paths[gu.idx(i + 1, j)]);
            if p.is_finite() && q.is_finite() {
                step = step.max(k0 * (q - p).abs());
            }
        }
    }
    for j in 0..gu.n[1].saturating_sub(1) {
        for i in 0..gu.n[0] {
            let (p, q) = (paths[gu.idx(i, j)], paths[gu.idx(i, j + 1)]);
            if p.is_finite() && q.is_finite() {
                step = step.max(k0 * (q - p).abs());
            }
        }
    }
    Ok((
        v,
        TransferStats {
            opl_chief: chief.opl,
            rays: g * g,
            passed,
            stopped,
            reflected,
            rim_stopped,
            rim_beyond,
            rim_resolved: rim.is_some(),
            carried: u.power(),
            left_window: left,
            walk_off_mm: walk,
            curvature: curv,
            arriving_ray_angle: angle,
            max_phase_step: step,
        },
    ))
}
