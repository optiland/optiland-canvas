//! Thin elements: things that multiply a field in one plane.
//!
//! The thin-element approximation treats an element as a complex transmittance
//! `t(x, y)` applied in a single plane, `U → t·U` (Goodman, *Introduction to
//! Fourier Optics*, 4th ed. (2017), the thin lens and the amplitude and phase
//! transmittance of a thin object). It ignores what happens inside the
//! element's thickness; elements that need more say so in their report.
//!
//! **Apertures are anti-aliased by exact area.** A sample is the field over
//! its cell, so an aperture edge crossing a cell transmits that cell's
//! covered fraction, computed exactly for every shape here: a rectangle by
//! interval overlap, a disc by integrating the circle's chord analytically,
//! a polygon by clipping it to the cell and taking the shoelace area. An
//! aperture placed on a sampled field then passes exactly its geometric area
//! of a uniform field, which is what `a_disc_aperture_passes_its_area`
//! holds, and the edge error falls as `Δx²` rather than `Δx`.

use crate::report::{Method, SamplingWarning, StepReport};
use crate::{Cx, Field2, WaveError, WaveReport};
use std::f64::consts::PI;

/// A region of the plane, in the plane's local mm.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// Axis-aligned rectangle centred at `c` with half extents `h`.
    Rect { c: [f64; 2], h: [f64; 2] },
    Disc { c: [f64; 2], r: f64 },
    Annulus { c: [f64; 2], r_in: f64, r_out: f64 },
    /// A simple polygon (convex or not), vertices in order.
    Polygon(Vec<[f64; 2]>),
}

impl Shape {
    /// The area of `self ∩ [x0, x1] × [y0, y1]`, mm², exactly.
    pub fn cell_overlap(&self, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        match self {
            Shape::Rect { c, h } => {
                let ox = (x1.min(c[0] + h[0]) - x0.max(c[0] - h[0])).max(0.0);
                let oy = (y1.min(c[1] + h[1]) - y0.max(c[1] - h[1])).max(0.0);
                ox * oy
            }
            Shape::Disc { c, r } => disc_rect_area(*r, x0 - c[0], x1 - c[0], y0 - c[1], y1 - c[1]),
            Shape::Annulus { c, r_in, r_out } => {
                let (a, b, cc, d) = (x0 - c[0], x1 - c[0], y0 - c[1], y1 - c[1]);
                (disc_rect_area(*r_out, a, b, cc, d) - disc_rect_area(*r_in, a, b, cc, d)).max(0.0)
            }
            Shape::Polygon(v) => polygon_rect_area(v, x0, x1, y0, y1),
        }
    }

    /// The shape's own area, mm².
    pub fn area(&self) -> f64 {
        match self {
            Shape::Rect { h, .. } => 4.0 * h[0] * h[1],
            Shape::Disc { r, .. } => PI * r * r,
            Shape::Annulus { r_in, r_out, .. } => PI * (r_out * r_out - r_in * r_in),
            Shape::Polygon(v) => shoelace(v).abs(),
        }
    }

    fn validate(&self) -> Result<(), WaveError> {
        let ok = match self {
            Shape::Rect { c, h } => c.iter().chain(h).all(|v| v.is_finite()) && h[0] >= 0.0 && h[1] >= 0.0,
            Shape::Disc { c, r } => c.iter().all(|v| v.is_finite()) && r.is_finite() && *r >= 0.0,
            Shape::Annulus { c, r_in, r_out } => {
                c.iter().all(|v| v.is_finite()) && *r_in >= 0.0 && r_out.is_finite() && r_out >= r_in
            }
            Shape::Polygon(v) => v.len() >= 3 && v.iter().flatten().all(|x| x.is_finite()),
        };
        if ok {
            Ok(())
        } else {
            Err(WaveError::OutOfRange(format!("not a usable shape: {self:?}")))
        }
    }
}

/// `∫ √(r² − x²) dx`.
fn semicircle_integral(r: f64, x: f64) -> f64 {
    let x = x.clamp(-r, r);
    0.5 * (x * (r * r - x * x).max(0.0).sqrt() + r * r * (x / r).asin())
}

/// Area of the disc of radius `r` about the origin inside `[x0,x1]×[y0,y1]`.
///
/// The disc's vertical extent at `x` is `[−s(x), s(x)]`, `s = √(r² − x²)`, so
/// the area is `∫ max(0, min(y1, s) − max(y0, −s)) dx`. Between the
/// breakpoints where `s` crosses `y0` or `y1` the min and max pick a fixed
/// branch, and each branch integrates in closed form.
fn disc_rect_area(r: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    let a = x0.max(-r);
    let b = x1.min(r);
    if a >= b || y0 >= y1 {
        return 0.0;
    }
    let mut cuts = vec![a, b];
    for y in [y0, y1] {
        if y.abs() < r {
            let xs = (r * r - y * y).sqrt();
            for x in [-xs, xs] {
                if x > a && x < b {
                    cuts.push(x);
                }
            }
        }
    }
    cuts.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let s = |x: f64| (r * r - x * x).max(0.0).sqrt();
    let mut area = 0.0;
    for w in cuts.windows(2) {
        let (p, q) = (w[0], w[1]);
        if q <= p {
            continue;
        }
        let m = 0.5 * (p + q);
        let sm = s(m);
        let top_is_s = sm < y1;
        let bot_is_s = -sm > y0;
        let top = if top_is_s { sm } else { y1 };
        let bot = if bot_is_s { -sm } else { y0 };
        if top <= bot {
            continue;
        }
        // ∫ top dx − ∫ bot dx over [p, q].
        let int_s = semicircle_integral(r, q) - semicircle_integral(r, p);
        let int_top = if top_is_s { int_s } else { y1 * (q - p) };
        let int_bot = if bot_is_s { -int_s } else { y0 * (q - p) };
        area += int_top - int_bot;
    }
    area.max(0.0)
}

fn shoelace(v: &[[f64; 2]]) -> f64 {
    let n = v.len();
    let mut a = 0.0;
    for i in 0..n {
        let [x0, y0] = v[i];
        let [x1, y1] = v[(i + 1) % n];
        a += x0 * y1 - x1 * y0;
    }
    0.5 * a
}

/// Clip a polygon to an axis-aligned rectangle (Sutherland–Hodgman) and take
/// the shoelace area. The rectangle is convex, so the clip is exact for any
/// simple subject polygon; degenerate edges it may leave on the cell border
/// contribute zero area.
fn polygon_rect_area(v: &[[f64; 2]], x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let mut poly: Vec<[f64; 2]> = v.to_vec();
    // (axis, bound, keep_greater)
    for (axis, bound, keep_ge) in [(0, x0, true), (0, x1, false), (1, y0, true), (1, y1, false)] {
        if poly.is_empty() {
            break;
        }
        let inside = |p: &[f64; 2]| if keep_ge { p[axis] >= bound } else { p[axis] <= bound };
        let mut out = Vec::with_capacity(poly.len() + 4);
        for i in 0..poly.len() {
            let cur = poly[i];
            let prev = poly[(i + poly.len() - 1) % poly.len()];
            let (ci, pi) = (inside(&cur), inside(&prev));
            if ci != pi {
                let t = (bound - prev[axis]) / (cur[axis] - prev[axis]);
                let mut x = [prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])];
                x[axis] = bound;
                out.push(x);
            }
            if ci {
                out.push(cur);
            }
        }
        poly = out;
    }
    if poly.len() < 3 {
        0.0
    } else {
        shoelace(&poly).abs()
    }
}

/// The transmittance of an aperture on `u`'s grid: each sample's covered
/// fraction of its cell. `opaque_inside` inverts it (a stop, an obscuration).
pub fn aperture_mask(u: &Field2, shape: &Shape, opaque_inside: bool) -> Result<Vec<f64>, WaveError> {
    shape.validate()?;
    let g = u.grid;
    let [dx, dy] = g.pitch_mm;
    let cell = dx * dy;
    let mut t = Vec::with_capacity(g.len());
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, y) = g.xy(i, j);
            let f = (shape.cell_overlap(x - 0.5 * dx, x + 0.5 * dx, y - 0.5 * dy, y + 0.5 * dy) / cell)
                .clamp(0.0, 1.0);
            t.push(if opaque_inside { 1.0 - f } else { f });
        }
    }
    Ok(t)
}

/// Pass `u` through an aperture. The power the opaque part stops is booked as
/// `absorbed_aperture`.
pub fn apply_aperture(
    u: &mut Field2,
    shape: &Shape,
    opaque_inside: bool,
    report: &mut WaveReport,
) -> Result<(), WaveError> {
    let t = aperture_mask(u, shape, opaque_inside)?;
    // A cell passing a fraction t of its area transmits t·|U|²ΔA; the opaque
    // rest, (1 − t)·|U|²ΔA, is absorbed. The sample t·U carries only t²·|U|²ΔA:
    // the edge diffracted the difference beyond the grid's Nyquist band.
    let da = u.grid.cell_area_mm2();
    let before = u.power();
    let (mut absorbed, mut beyond) = (0.0, 0.0);
    for (z, ti) in u.data.iter_mut().zip(&t) {
        let i = z.norm_sqr() * da;
        absorbed += (1.0 - ti) * i;
        beyond += (ti - ti * ti) * i;
        *z *= *ti;
    }
    report.ledger.absorbed_aperture += absorbed;
    report.ledger.beyond_nyquist += beyond;
    close_element(report, before, u.power());
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!(
            "{} {shape:?}, edges weighted by exact cell coverage",
            if opaque_inside { "stop" } else { "aperture" }
        ),
    });
    Ok(())
}

/// An element's report standing alone: what arrived is what was launched,
/// what left is what was detected, and the ledger closes. A path merging it
/// takes the loss terms and keeps its own launched and detected.
fn close_element(report: &mut WaveReport, before: f64, after: f64) {
    if report.ledger.launched == 0.0 {
        report.ledger.launched = before;
    }
    report.ledger.detected = after;
    report.ledger.close();
}

/// An ideal thin lens of focal length `f_mm` centred at `c`:
/// `t = exp(−ik((x−cx)² + (y−cy)²)/2f)`, so a positive `f` converges
/// (Goodman's thin lens, in the `exp(−iωt)` convention). Lossless.
///
/// The lens is a chirp, and a chirp outruns its samples at the window edge
/// once `k r Δx / f > π`; the report warns there rather than letting the
/// aliased phase pass as a lens.
pub fn apply_thin_lens(
    u: &mut Field2,
    f_mm: f64,
    c: [f64; 2],
    report: &mut WaveReport,
) -> Result<(), WaveError> {
    if !(f_mm.is_finite() && f_mm != 0.0) {
        return Err(WaveError::OutOfRange(format!("focal length {f_mm} mm")));
    }
    let k = u.k_mm();
    let g = u.grid;
    let mut rmax: f64 = 0.0;
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, y) = g.xy(i, j);
            let (dx, dy) = (x - c[0], y - c[1]);
            rmax = rmax.max(dx.abs().max(dy.abs()));
            let idx = g.idx(i, j);
            u.data[idx] *= Cx::expi(-k * (dx * dx + dy * dy) / (2.0 * f_mm));
        }
    }
    let step = k * rmax * g.pitch_mm[0].max(g.pitch_mm[1]) / f_mm.abs();
    if step > PI {
        report.warn(SamplingWarning::Undersampled {
            what: format!("thin lens f = {f_mm} mm"),
            max_phase_step_rad: step,
        });
    }
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!("ideal thin lens, f = {f_mm} mm, centred at ({}, {}) mm", c[0], c[1]),
    });
    Ok(())
}

/// Multiply by `exp(iφ)` sample by sample. Lossless.
pub fn apply_phase(u: &mut Field2, phase_rad: &[f64], report: &mut WaveReport) -> Result<(), WaveError> {
    if phase_rad.len() != u.data.len() {
        return Err(WaveError::Mismatch("phase screen and field differ in size".into()));
    }
    if !phase_rad.iter().all(|p| p.is_finite()) {
        return Err(WaveError::InvalidField("phase screen has a non-finite sample".into()));
    }
    for (z, p) in u.data.iter_mut().zip(phase_rad) {
        *z *= Cx::expi(*p);
    }
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: "phase screen".into(),
    });
    Ok(())
}

/// Multiply by a complex transmittance sample by sample. Power it removes is
/// booked as `absorbed_element`; a transmittance above 1 in magnitude, which
/// would add power, is refused.
pub fn apply_transmittance(u: &mut Field2, t: &[Cx], report: &mut WaveReport) -> Result<(), WaveError> {
    if t.len() != u.data.len() {
        return Err(WaveError::Mismatch("mask and field differ in size".into()));
    }
    if !t.iter().all(|z| z.is_finite() && z.norm_sqr() <= 1.0 + 1e-12) {
        return Err(WaveError::OutOfRange(
            "a transmittance must be finite and at most 1 in magnitude".into(),
        ));
    }
    let before = u.power();
    for (z, ti) in u.data.iter_mut().zip(t) {
        *z *= *ti;
    }
    report.ledger.absorbed_element += before - u.power();
    close_element(report, before, u.power());
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: "complex transmittance".into(),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disc_overlap_matches_the_disc_and_its_quadrants() {
        let r = 1.3;
        let full = disc_rect_area(r, -5.0, 5.0, -5.0, 5.0);
        assert!((full - PI * r * r).abs() < 1e-14);
        let q = disc_rect_area(r, 0.0, 5.0, 0.0, 5.0);
        assert!((q - PI * r * r / 4.0).abs() < 1e-14);
        // A strip through the centre: 2∫₀^{w} √(r² − x²)... checked against
        // the half-strip by symmetry.
        let s = disc_rect_area(r, -0.4, 0.4, -5.0, 5.0);
        let half = disc_rect_area(r, 0.0, 0.4, -5.0, 5.0);
        assert!((s - 2.0 * half).abs() < 1e-14);
        // A cell straddling the rim, against brute-force midpoint sampling.
        let (x0, x1, y0, y1) = (0.9, 1.1, 0.7, 0.95);
        let n = 4000;
        let mut acc = 0.0;
        for a in 0..n {
            for b in 0..n {
                let x = x0 + (a as f64 + 0.5) * (x1 - x0) / n as f64;
                let y = y0 + (b as f64 + 0.5) * (y1 - y0) / n as f64;
                if x * x + y * y <= r * r {
                    acc += 1.0;
                }
            }
        }
        let brute = acc * (x1 - x0) * (y1 - y0) / (n * n) as f64;
        assert!((disc_rect_area(r, x0, x1, y0, y1) - brute).abs() < 2e-6);
    }

    #[test]
    fn polygon_overlap_of_a_triangle() {
        let tri = vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        assert!((polygon_rect_area(&tri, -1.0, 3.0, -1.0, 3.0) - 2.0).abs() < 1e-15);
        // The unit square's lower-left corner is fully inside.
        assert!((polygon_rect_area(&tri, 0.0, 1.0, 0.0, 1.0) - 1.0).abs() < 1e-15);
        // A concave L.
        let l = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
        assert!((polygon_rect_area(&l, -1.0, 3.0, -1.0, 3.0) - 3.0).abs() < 1e-15);
        assert!((polygon_rect_area(&l, 1.0, 2.0, 1.0, 2.0)).abs() < 1e-15);
    }
}
