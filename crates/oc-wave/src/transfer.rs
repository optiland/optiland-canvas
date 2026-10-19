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
//! The rays are traced along a reference wavefront fixed by the geometry, not
//! by the field ([`Reference`]): the path gives the beam's paraxial wavefront
//! there, carried from the source through every element before (its Gaussian
//! beam parameter, through each element's paraxial matrices), about a ray of
//! its own, the chief ray or one of a modulator's orders. So a beam that
//! arrives converging or diverging is traced with rays that do (rays traced
//! parallel to the chief ray would carry the field's own curvature through
//! the element's map as if it were collimated light, and move the focus), and
//! the transfer is one linear operator of the field, the same whatever field
//! it is given: `T(u₁ + u₂) = T(u₁) + T(u₂)` (F. Wyrowski, M. Kuhn, above; a
//! reference fitted to the arriving field's own phase gradient would make
//! the operator depend on the field, and a field of several directions would
//! be traced along one that is none of them). What the reference leaves out,
//! the field's departure from it, is not traced; the report says how far the
//! field's directions depart from it, and what that departure moves across
//! the slab.
//!
//! The traced map ([`TracedMap`]) is the operator: traced once for a
//! reference and a window, it carries any field on that window
//! ([`apply`]).

use crate::path::Beam;
use crate::{Cx, Field2, Grid2, WaveError};
use oc_core::scene3::Scene3;
use oc_core::vec3::Ray3;
use oc_core::Vec3;
use oc_trace::{trace3, LaunchRay3, Segment3, TraceConfig3};

/// What a transfer delivered besides the field.
pub(crate) struct TransferStats {
    /// The beam axis's optical path from the entry plane to the exit plane, mm.
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
    /// The reference wavefront's curvature, 1/mm (positive diverging), the
    /// mean of its two axes'.
    pub curvature: f64,
    /// The steepest departure of the arriving field's directions, where it
    /// holds a hundredth of its peak, from the reference's, rad; and the
    /// power-weighted rms departure.
    pub arriving_ray_angle: f64,
    pub arriving_ray_rms: f64,
    /// The slab the map spans, entry plane to exit plane along the beam, mm.
    pub slab_mm: f64,
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
    /// The direction cosines across the exit frame.
    dx: f64,
    dy: f64,
}

/// Where a ray, after exactly one interaction with the element, crosses the
/// exit plane (through `p` with normal `n`, the outgoing direction): the
/// crossing point, the optical path there, and the power left. Of several
/// branches (a splitting surface), the one that best continues along `n`.
/// And the ray's direction there.
fn crossing(segs: &[Segment3], p: Vec3, n: Vec3) -> Option<(Vec3, f64, f64, Vec3)> {
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
            Some((d.dot(n), x, s.opl0_mm + index * (x - s.a).length(), s.flux_start, d))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, x, opl, t, d)| (x, opl, t, d))
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
    r: Reference,
    car: [f64; 2],
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
        Self::about(u, &Reference { curv: [s[0], s[0]], tilt: [s[1] + u.carrier[0], s[2] + u.carrier[1]], centre: [0.0, 0.0] })
    }

    /// `u` with a reference wavefront's phase taken out.
    pub(crate) fn about(u: &Field2, r: &Reference) -> Self {
        let k = u.k_mm();
        let mut d = u.clone();
        let g = u.grid;
        let car = u.carrier;
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (a, b) = g.xy(i, j);
                let idx = g.idx(i, j);
                d.data[idx] *= Cx::expi(-ref_phase(k, r, car, a, b));
            }
        }
        Self { d, k, r: *r, car }
    }

    /// The field's sample value at `(a, b)` (carrier excluded); zero outside.
    pub(crate) fn at(&self, a: f64, b: f64) -> Cx {
        crate::path::sample(&self.d, a, b) * Cx::expi(ref_phase(self.k, &self.r, self.car, a, b))
    }
}

/// A reference's phase at `(a, b)`, relative to a carrier: its gradient is
/// `k` times the reference's direction less the carrier.
fn ref_phase(k: f64, r: &Reference, car: [f64; 2], a: f64, b: f64) -> f64 {
    let (da, db) = (a - r.centre[0], b - r.centre[1]);
    k * ((r.tilt[0] - car[0]) * a + (r.tilt[1] - car[1]) * b + 0.5 * (r.curv[0] * da * da + r.curv[1] * db * db))
}

/// The wavefront an element's rays are traced along: direction cosines
/// across the incoming beam `t = tilt + curv ⊙ ((a, b) − centre)`, the
/// carrier included, curvature positive diverging. Fixed by the geometry, so
/// that the transfer is one linear operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Reference {
    pub curv: [f64; 2],
    pub tilt: [f64; 2],
    pub centre: [f64; 2],
}

impl Reference {
    pub(crate) fn dir(&self, a: f64, b: f64) -> [f64; 2] {
        [self.tilt[0] + self.curv[0] * (a - self.centre[0]), self.tilt[1] + self.curv[1] * (b - self.centre[1])]
    }
}

/// How far `u`'s directions depart from a reference's, rad: the steepest
/// where the field holds a hundredth of its peak, and the power-weighted rms
/// where it holds a ten-thousandth. The reference's phase is taken out first,
/// so that only the departure, not the field's whole tilt, has to be resolved
/// by the samples.
pub(crate) fn departure(u: &Field2, r: &Reference) -> (f64, f64) {
    let gu = u.grid;
    let k = u.k_mm();
    let top = u.data.iter().fold(0.0f64, |m, c| m.max(c.norm_sqr()));
    let mut worst: f64 = 0.0;
    let (mut sw, mut s2) = (0.0, 0.0);
    if top <= 0.0 || top.is_nan() {
        return (0.0, 0.0);
    }
    let rest = |i: usize, j: usize| {
        let (a, b) = gu.xy(i, j);
        u.at(i, j) * Cx::expi(-ref_phase(k, r, u.carrier, a, b))
    };
    for j in 1..gu.n[1].saturating_sub(1) {
        for i in 1..gu.n[0].saturating_sub(1) {
            let w = u.at(i, j).norm_sqr();
            if w < 1e-4 * top {
                continue;
            }
            let gx = (rest(i + 1, j) * rest(i - 1, j).conj()).arg() / (2.0 * gu.pitch_mm[0]);
            let gy = (rest(i, j + 1) * rest(i, j - 1).conj()).arg() / (2.0 * gu.pitch_mm[1]);
            let d = (gx / k).hypot(gy / k);
            sw += w;
            s2 += w * d * d;
            if w >= 1e-2 * top {
                worst = worst.max(d);
            }
        }
    }
    (worst, (s2 / sw).sqrt())
}

/// The edge of the region of the entry plane whose rays pass the element, as
/// a polygon of 720 vertices about the chief ray: along each azimuth, the
/// first ray of the grid that fails brackets the edge, which bisection on
/// traced rays then finds to a millionth of the ray spacing. `None` when the
/// region is not star-shaped about the chief ray (a ray passes again beyond
/// the edge), or the bracket does not hold.
pub(crate) fn rim_polygon(pass: &dyn Fn(f64, f64) -> bool, grid: &[bool], g: usize, h: f64) -> Option<Vec<[f64; 2]>> {
    let half = (g - 1) as f64 / 2.0;
    let r_max = half * h * std::f64::consts::SQRT_2;
    let on_grid = |a: f64, b: f64| -> Option<bool> {
        let (i, j) = ((a / h + half).round(), (b / h + half).round());
        if i < 0.0 || j < 0.0 || i > (g - 1) as f64 || j > (g - 1) as f64 {
            return None;
        }
        Some(grid[j as usize * g + i as usize])
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
        while lo > 0.0 && !pass(lo * c, lo * s) {
            lo = (lo - h).max(0.0);
        }
        if !pass(lo * c, lo * s) {
            return None;
        }
        let mut hi = rf + 0.5 * h;
        let mut tries = 0;
        while pass(hi * c, hi * s) {
            hi += h;
            tries += 1;
            if tries > 4 {
                return None;
            }
        }
        while hi - lo > 1e-6 * h {
            let mid = 0.5 * (lo + hi);
            if pass(mid * c, mid * s) {
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

/// One ray through the element `probe` holds: from `(a, b)` across `inc`
/// along the direction cosines `t`, to where it crosses the exit plane across
/// `out`: its point and direction cosines across `out`, and its optical path.
pub(crate) fn ray_through(probe: &Scene3, lam: f64, inc: &Beam, out: &Beam, a: f64, b: f64, t: [f64; 2]) -> Option<([f64; 2], [f64; 2], f64)> {
    let r = trace_one(probe, lam, inc, out, a, b, t)?;
    Some(([r.x, r.y], [r.dx, r.dy], r.opl))
}

fn trace_config(probe: &Scene3) -> TraceConfig3 {
    TraceConfig3 { ghosts: false, polarization: false, bounds_mm: oc_trace::bounds_for_scene3(probe), ..TraceConfig3::default() }
}

fn trace_one(probe: &Scene3, lam: f64, inc: &Beam, out: &Beam, a: f64, b: f64, t: [f64; 2]) -> Option<RayOut> {
    trace_with(probe, &trace_config(probe), lam, inc, out, a, b, t)
}

#[allow(clippy::too_many_arguments)]
fn trace_with(probe: &Scene3, cfg: &TraceConfig3, lam: f64, inc: &Beam, out: &Beam, a: f64, b: f64, t: [f64; 2]) -> Option<RayOut> {
    let tz2 = 1.0 - t[0] * t[0] - t[1] * t[1];
    if tz2 <= 0.0 {
        return None;
    }
    let dir = (inc.x * t[0] + inc.y * t[1] + inc.d * tz2.sqrt()).normalize();
    let ray = Ray3::new(inc.point(a, b), dir).ok()?;
    let r = trace3(probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], cfg);
    let (x, opl, tr, d) = crossing(&r.segments, out.origin, out.d)?;
    let (xx, yy) = out.coords(x);
    Some(RayOut { x: xx, y: yy, opl, t: tr, dx: d.dot(out.x), dy: d.dot(out.y) })
}

/// The element's paraxial matrices about the chief ray, `[A, B, C, D]` for
/// each axis of the beam frames, in reduced angles (`n·θ`): from rays moved
/// `ε` across the entry plane and tilted `δ`, differenced against the chief
/// ray. Cross terms between the axes are left out.
pub(crate) fn abcd(probe: &Scene3, lam: f64, inc: &Beam, out: &Beam, n_in: f64, n_out: f64) -> Option<[[f64; 4]; 2]> {
    let (eps, del) = (1e-3, 1e-4);
    let o = trace_one(probe, lam, inc, out, 0.0, 0.0, [0.0, 0.0])?;
    let mut m = [[0.0; 4]; 2];
    for (axis, row) in m.iter_mut().enumerate() {
        let (pa, pb) = if axis == 0 { (eps, 0.0) } else { (0.0, eps) };
        let tilt = if axis == 0 { [del, 0.0] } else { [0.0, del] };
        let rx = trace_one(probe, lam, inc, out, pa, pb, [0.0, 0.0])?;
        let rt = trace_one(probe, lam, inc, out, 0.0, 0.0, tilt)?;
        let pos = |r: &RayOut| if axis == 0 { r.x - o.x } else { r.y - o.y };
        let dir = |r: &RayOut| if axis == 0 { r.dx - o.dx } else { r.dy - o.dy };
        *row = [pos(&rx) / eps, pos(&rt) / (n_in * del), n_out * dir(&rx) / eps, n_out * dir(&rt) / (n_in * del)];
    }
    Some(m)
}

/// A Gaussian beam parameter `q` (in a medium of index `n_in`) through
/// reduced paraxial matrices into one of index `n_out`: `q̂ = q/n`,
/// `q̂' = (A q̂ + B)/(C q̂ + D)` (A. E. Siegman, *Lasers* (1986), ch. 15).
pub(crate) fn q_through(q: Cx, m: [f64; 4], n_in: f64, n_out: f64) -> Cx {
    let qh = q * (1.0 / n_in);
    (qh * m[0] + Cx::real(m[1])) / (qh * m[2] + Cx::real(m[3])) * n_out
}

/// The element `probe` holds, traced for a reference and a window: the
/// linear operator a transfer applies.
pub(crate) struct TracedMap {
    map: Map,
    /// The ray at the beam's origin along the reference, and the beam axis's
    /// optical path.
    chief: RayOut,
    opl_axis: f64,
    rim: Option<Vec<[f64; 2]>>,
    grid: Grid2,
    reference: Reference,
    passed: usize,
    walk: f64,
    slab: f64,
}

impl TracedMap {
    /// Whether it is the operator for this window and reference.
    pub(crate) fn fits(&self, grid: Grid2, r: &Reference) -> bool {
        self.grid == grid && self.reference == *r
    }
}

/// Trace the element `probe` holds across the window `grid` (across `inc`,
/// at the entry plane) to the exit plane across `out`, with `rays` rays
/// across the window, along the reference `r`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_map(probe: &Scene3, grid: Grid2, lam: f64, inc: &Beam, out: &Beam, rays: usize, r: &Reference) -> Result<TracedMap, WaveError> {
    let cfg = trace_config(probe);
    let trace = |a: f64, b: f64| -> Option<RayOut> { trace_with(probe, &cfg, lam, inc, out, a, b, r.dir(a, b)) };
    let chief = trace(0.0, 0.0)
        .ok_or_else(|| WaveError::OutOfRange("the chief ray does not cross the surface to its exit plane".into()))?;
    let opl_axis = if r.dir(0.0, 0.0) == [0.0, 0.0] { chief.opl } else { trace_with(probe, &cfg, lam, inc, out, 0.0, 0.0, [0.0, 0.0]).map_or(chief.opl, |x| x.opl) };

    // The ray grid: odd, so the chief ray is its centre, spanning the window.
    let g = (rays.max(9) | 1).min(1025);
    let w = grid.width_mm(0).max(grid.width_mm(1));
    let h = w / (g - 1) as f64;
    let half = (g - 1) as f64 / 2.0;
    let mut rays_out: Vec<Option<RayOut>> = Vec::with_capacity(g * g);
    for j in 0..g {
        for i in 0..g {
            let (a, b) = ((i as f64 - half) * h, (j as f64 - half) * h);
            rays_out.push(trace(a, b));
        }
    }
    let passed = rays_out.iter().filter(|r| r.is_some()).count();
    if passed < 9 {
        return Err(WaveError::OutOfRange("fewer than nine rays pass the element".into()));
    }
    // The rim at the field's own sampling, to be laid on the arriving field.
    let rim = if passed < g * g {
        let ok: Vec<bool> = rays_out.iter().map(|r| r.is_some()).collect();
        rim_polygon(&|a, b| trace(a, b).is_some(), &ok, g, h)
    } else {
        None
    };
    // Fit the map and the path on coordinates scaled to the window.
    let s = 0.5 * w;
    let mut rows_x = Vec::with_capacity(passed);
    let mut rows_y = Vec::with_capacity(passed);
    let mut rows_p = Vec::with_capacity(passed);
    for j in 0..g {
        for i in 0..g {
            if let Some(r) = rays_out[j * g + i] {
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
    let mut map = Map { g, h, s, ax, ay, q, rays: rays_out, rx: vec![0.0; g * g], ry: vec![0.0; g * g], rp: vec![0.0; g * g] };
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
                        map.rays[k] = Some(RayOut { x: fx + rx, y: fy + ry, opl: chief.opl + fp + rp, t: st / n, dx: 0.0, dy: 0.0 });
                    }
                }
            }
        }
    }
    Ok(TracedMap { map, chief, opl_axis, rim, grid, reference: *r, passed, walk, slab: (out.origin - inc.origin).length() })
}

/// Carry `u`, on the window `m` was traced for, through it into a medium of
/// index `n_out` across `out`: linear in `u`.
pub(crate) fn apply(m: &TracedMap, u: &Field2, out: &Beam, n_out: f64) -> Result<(Field2, TransferStats), WaveError> {
    if u.grid != m.grid {
        return Err(WaveError::OutOfRange("a traced map applied to a field on another window".into()));
    }
    let lam = u.lambda_um;
    let gu = u.grid;
    let k_in = u.k_mm();
    let map = &m.map;
    let mut um = u.clone();
    let (rim_stopped, rim_beyond) = match &m.rim {
        Some(poly) => crate::elements::apply_coverage(&mut um, &crate::elements::polygon_coverage(&u.grid, poly))?,
        None => (0.0, 0.0),
    };
    let u = &um;
    let (angle, rms) = departure(u, &m.reference);

    // The arriving field, carrier included, read between its samples with
    // the reference's phase taken out (`Smooth`).
    let smooth = Smooth::about(u, &m.reference);
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
    let chief = m.chief;
    let mut v = u.clone();
    v.carrier = [0.0, 0.0];
    v.n_medium = n_out;
    v.frame = out.iso();
    v.opl_ref_mm = u.opl_ref_mm + m.opl_axis;
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
            opl_chief: m.opl_axis,
            rays: map.g * map.g,
            passed: m.passed,
            stopped,
            reflected,
            rim_stopped,
            rim_beyond,
            rim_resolved: m.rim.is_some(),
            carried: u.power(),
            left_window: left,
            walk_off_mm: m.walk,
            curvature: 0.5 * (m.reference.curv[0] + m.reference.curv[1]),
            arriving_ray_angle: angle,
            arriving_ray_rms: rms,
            slab_mm: m.slab,
            max_phase_step: step,
        },
    ))
}

/// Carry `u` (across `inc`, at the entry plane) through the element `probe`
/// holds, to the exit plane across `out`, in a medium of index `n_out`, with
/// `rays` rays across the field's window, along the reference `r`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn transfer(probe: &Scene3, u: &Field2, inc: &Beam, out: &Beam, n_out: f64, rays: usize, r: &Reference) -> Result<(Field2, TransferStats), WaveError> {
    apply(&trace_map(probe, u.grid, u.lambda_um, inc, out, rays, r)?, u, out, n_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beams::GaussianBeam;
    use oc_core::build3::SingletGeom;

    /// The front face of an N-BK7 singlet, R ±20 mm, 4 mm thick, centred at
    /// x = 50: the scene with it alone, and the beam frames of the slab across
    /// it, the entry plane in air at x = 47.9 and the exit plane in the glass
    /// at x = 48.3.
    fn face() -> (Scene3, Beam, Beam) {
        let mut s = Scene3::new();
        let g = SingletGeom { r1_mm: 20.0, r2_mm: -20.0, thickness_mm: 4.0, semi_aperture_mm: 6.0, conic1: 0.0, conic2: 0.0 };
        s.add_singlet(Vec3::new(50.0, 0.0, 0.0), g, 1.5168, 64.17);
        let beam = |x: f64| Beam { origin: Vec3::new(x, 0.0, 0.0), x: Vec3::Y, y: Vec3::Z, d: Vec3::X };
        (s, beam(47.9), beam(48.3))
    }

    /// A 0.3 mm Gaussian on a 2 mm window of 256², tilted `t` along x.
    fn tilted(t: f64) -> Field2 {
        let g = Grid2::square(256, 2.0 / 256.0).unwrap();
        let mut u = GaussianBeam { w0_mm: 0.3, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
        let k = u.k_mm();
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (a, _) = g.xy(i, j);
                let idx = g.idx(i, j);
                u.data[idx] *= Cx::expi(k * t * a);
            }
        }
        u
    }

    fn sum(a: &Field2, b: &Field2) -> Field2 {
        let mut c = a.clone();
        for (x, y) in c.data.iter_mut().zip(&b.data) {
            *x += *y;
        }
        c
    }

    fn rel(a: &Field2, b: &Field2) -> f64 {
        let d: f64 = a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).norm_sqr()).sum();
        (d / a.data.iter().map(|x| x.norm_sqr()).sum::<f64>()).sqrt()
    }

    #[test]
    fn a_transfer_about_a_fixed_reference_is_linear_in_the_field() {
        // Two Gaussians tilted ±10 mrad through a lens face. About one
        // reference fixed by the geometry the transfer is one linear
        // operator, T(u₁ + u₂) = T(u₁) + T(u₂) to round-off; about a sphere
        // fitted to each field it is given, as before I6, it is not, and the
        // sum of the two lobes is traced along a direction that is neither's.
        let (probe, inc, out) = face();
        let (u1, u2) = (tilted(0.01), tilted(-0.01));
        let u12 = sum(&u1, &u2);
        let fixed = Reference { curv: [0.0, 0.0], tilt: [0.0, 0.0], centre: [0.0, 0.0] };
        let t = |u: &Field2, r: &Reference| transfer(&probe, u, &inc, &out, 1.5168, 129, r).unwrap().0;
        let lin = rel(&t(&u12, &fixed), &sum(&t(&u1, &fixed), &t(&u2, &fixed)));
        let fit = |u: &Field2| {
            let ([c, a, b], _) = fit_sphere(u);
            Reference { curv: [c, c], tilt: [a, b], centre: [0.0, 0.0] }
        };
        let old = rel(&t(&u12, &fit(&u12)), &sum(&t(&u1, &fit(&u1)), &t(&u2, &fit(&u2))));
        println!("measured: transfer about a fixed reference, |T(u1+u2) - T(u1) - T(u2)| / |T(u1+u2)| = {lin:.3e}");
        println!("measured: transfer about a sphere fitted to each field (before I6), the same = {old:.3e}");
        assert!(lin < 1e-12, "{lin}");
        assert!(old > 1e3 * lin.max(1e-15), "{old} vs {lin}");
    }

    #[test]
    fn a_tilted_beam_through_a_plane_face_matches_the_exact_interface_when_traced_about_its_own_direction() {
        // A plane interface keeps each plane wave's tangential wavevector, so
        // the field through it is exact in the angular spectrum: 0.1 mm of
        // air, the interface's amplitude (√(1 − R), R = ((n − 1)/(n + 1))²,
        // the tracer's unpolarized transmittance, constant to 1e-8 over these
        // angles), 0.3 mm of glass [Goodman, ch. 3; Born & Wolf, §1.5]. A
        // Gaussian tilted 10 mrad, traced about its own direction, lands on
        // it; traced about the chief ray, every one of its rays runs straight
        // across the slab, and the field gathers the phase of the straight
        // run where a wave tilted θ advances only `k cos θ` per unit length:
        // a piston of `k₀θ²/2·(0.1 + 0.3/n)` = 0.15 rad too much.
        use crate::propagate::propagate;
        let mut s = Scene3::new();
        let g = SingletGeom { r1_mm: f64::INFINITY, r2_mm: -20.0, thickness_mm: 4.0, semi_aperture_mm: 6.0, conic1: 0.0, conic2: 0.0 };
        s.add_singlet(Vec3::new(50.0, 0.0, 0.0), g, 1.5168, 64.17);
        let beam = |x: f64| Beam { origin: Vec3::new(x, 0.0, 0.0), x: Vec3::Y, y: Vec3::Z, d: Vec3::X };
        let (inc, out) = (beam(47.9), beam(48.3));
        // N-BK7's index at the beam's wavelength, as the tracer has it.
        let n = s.bodies[0].material.index_at(0.6328, &oc_core::material::GlassCatalog::builtin()).unwrap();
        let u = tilted(0.01);
        let (air, _) = propagate(&u, 0.1, crate::Propagator::AngularSpectrum(Default::default()), &mut crate::Fft2::new()).unwrap();
        let mut at = air.in_medium(n).unwrap();
        let t = (1.0 - ((n - 1.0) / (n + 1.0)).powi(2)).sqrt();
        for z in &mut at.data {
            *z *= t;
        }
        let (exact, _) = propagate(&at, 0.3, crate::Propagator::AngularSpectrum(Default::default()), &mut crate::Fft2::new()).unwrap();
        let own = Reference { curv: [0.0, 0.0], tilt: [0.01, 0.0], centre: [0.0, 0.0] };
        let chief = Reference { curv: [0.0, 0.0], tilt: [0.0, 0.0], centre: [0.0, 0.0] };
        let (a, _) = transfer(&s, &u, &inc, &out, n, 129, &own).unwrap();
        let (b, _) = transfer(&s, &u, &inc, &out, n, 129, &chief).unwrap();
        let (ea, eb) = (rel(&exact, &a), rel(&exact, &b));
        // The chief-ray trace's error is mostly that piston.
        let c = exact.grid.n[0] / 2;
        let piston = (b.at(c, c) / exact.at(c, c)).arg();
        let k0 = 2.0 * std::f64::consts::PI / 0.6328e-3;
        let want = k0 * 0.5 * 0.01f64.powi(2) * (0.1 + 0.3 / n);
        println!("measured: tilted Gaussian through a plane face, traced about its own direction, against the exact interface, relative = {ea:.3e}");
        println!("measured: the same traced about the chief ray, relative = {eb:.3e}; its piston {piston:.4} rad against {want:.4}");
        assert!(ea < 1e-3, "{ea}");
        assert!(eb > 0.1, "{eb}");
        assert!((piston - want).abs() < 0.02 * want.abs(), "{piston} vs {want}");
    }

    #[test]
    fn a_tilted_beam_crosses_a_curved_face_where_its_rays_do() {
        // Inside the slab the field is carried along its rays (geometric
        // field tracing [Wyrowski & Kuhn 2011]), so a Gaussian 0.3 mm off the
        // axis and tilted 30 mrad, traced about its own ray, leaves the face
        // with its centroid where its rays' centroid crosses the exit plane,
        // each ray weighted by the Gaussian's power where it starts and the
        // power the face lets through. The curved face spreads the bundle
        // unevenly, so that centroid is not the central ray's crossing, by
        // 11 nm here; a check of the map, its interpolation and its Jacobian,
        // against the rays it is fitted to.
        let mut s = Scene3::new();
        let g = SingletGeom { r1_mm: 8.0, r2_mm: -8.0, thickness_mm: 2.0, semi_aperture_mm: 2.5, conic1: 0.0, conic2: 0.0 };
        s.add_singlet(Vec3::new(50.0, 0.0, 0.0), g, 1.5168, 64.17);
        let beam = |x: f64| Beam { origin: Vec3::new(x, 0.0, 0.0), x: Vec3::Y, y: Vec3::Z, d: Vec3::X };
        let (inc, out) = (beam(48.9), beam(49.2));
        let n = s.bodies[0].material.index_at(0.6328, &oc_core::material::GlassCatalog::builtin()).unwrap();
        let gr = Grid2::square(512, 2.56 / 512.0).unwrap();
        let mut u = GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.3, 0.0] }.sample(gr, 0.0).unwrap();
        let k = u.k_mm();
        for j in 0..gr.n[1] {
            for i in 0..gr.n[0] {
                let (a, _) = gr.xy(i, j);
                let idx = gr.idx(i, j);
                u.data[idx] *= Cx::expi(k * 0.03 * a);
            }
        }
        let own = Reference { curv: [0.0, 0.0], tilt: [0.03, 0.0], centre: [0.3, 0.0] };
        let (v, st) = transfer(&s, &u, &inc, &out, n, 257, &own).unwrap();
        let (x, _, _) = ray_through(&s, 0.6328, &inc, &out, 0.3, 0.0, [0.03, 0.0]).unwrap();
        let (mut s0, mut sx) = (0.0, 0.0);
        for j in 0..gr.n[1] {
            for i in 0..gr.n[0] {
                let w = v.at(i, j).norm_sqr();
                s0 += w;
                sx += w * gr.xy(i, j).0;
            }
        }
        let wave = sx / s0;
        // The rays: a fine grid over the Gaussian, out to four radii.
        let cfg = trace_config(&s);
        let (mut r0, mut rx) = (0.0, 0.0);
        let h = 0.005;
        for jb in -80i32..=80 {
            for ia in -80i32..=80 {
                let (a, b) = (0.3 + ia as f64 * h, jb as f64 * h);
                let w = (-2.0 * ((a - 0.3).powi(2) + b * b) / 0.01).exp();
                if let Some(r) = trace_with(&s, &cfg, 0.6328, &inc, &out, a, b, [0.03, 0.0]) {
                    r0 += w * r.t;
                    rx += w * r.t * r.x;
                }
            }
        }
        let rays = rx / r0;
        let e = wave - rays;
        println!("measured: a tilted, off-axis Gaussian through a curved face: centroid against its rays', µm = {:.4e}; the rays' against the central ray, µm = {:.4e}; departure {:.3e} rad", 1e3 * e, 1e3 * (rays - x[0]), st.arriving_ray_angle);
        assert!(e.abs() < 3e-6 && e.abs() < 0.2 * (rays - x[0]).abs(), "{e}");
    }

    #[test]
    fn each_lobe_traced_about_its_own_direction_matches_it_traced_alone() {
        // A lobe tilted 10 mrad, traced about its own direction, against the
        // same lobe traced about the chief ray's: the difference is what a
        // field of several directions loses when one reference serves them
        // all, here across a slab 0.4 mm deep.
        let (probe, inc, out) = face();
        let u1 = tilted(0.01);
        let own = Reference { curv: [0.0, 0.0], tilt: [0.01, 0.0], centre: [0.0, 0.0] };
        let chief = Reference { curv: [0.0, 0.0], tilt: [0.0, 0.0], centre: [0.0, 0.0] };
        let (a, sa) = transfer(&probe, &u1, &inc, &out, 1.5168, 129, &own).unwrap();
        let (b, sb) = transfer(&probe, &u1, &inc, &out, 1.5168, 129, &chief).unwrap();
        let d = rel(&a, &b);
        println!("measured: a 10 mrad lobe traced about the chief ray against about its own direction, relative = {d:.3e}; departure {:.3e} against {:.3e} rad", sb.arriving_ray_angle, sa.arriving_ray_angle);
        assert!(sa.arriving_ray_angle < 1e-3 && sb.arriving_ray_angle > 9e-3, "{} {}", sa.arriving_ray_angle, sb.arriving_ray_angle);
    }
}
