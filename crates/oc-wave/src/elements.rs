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
use crate::{Cx, Field2, Grid2, WaveError, WaveReport};
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

/// The covered fraction of each cell of `grid` by `shape`, exactly: the closed
/// forms above for a rectangle, a disc and an annulus, and for a polygon
/// [`polygon_coverage`].
pub fn coverage(grid: &Grid2, shape: &Shape) -> Result<Vec<f64>, WaveError> {
    shape.validate()?;
    if let Shape::Polygon(v) = shape {
        return Ok(polygon_coverage(grid, v));
    }
    let [dx, dy] = grid.pitch_mm;
    let cell = dx * dy;
    let mut t = Vec::with_capacity(grid.len());
    for j in 0..grid.n[1] {
        for i in 0..grid.n[0] {
            let (x, y) = grid.xy(i, j);
            t.push((shape.cell_overlap(x - 0.5 * dx, x + 0.5 * dx, y - 0.5 * dy, y + 0.5 * dy) / cell).clamp(0.0, 1.0));
        }
    }
    Ok(t)
}

/// The covered fraction of each cell of `grid` by a simple polygon (convex or
/// not, either orientation), exactly and in time that grows with the
/// boundary rather than the area. The cells within a cell of the boundary
/// take the polygon's area clipped to them; every other cell lies wholly on
/// one side, which its centre's side says, found by crossing each row's
/// centre line with the edges.
pub fn polygon_coverage(grid: &Grid2, poly: &[[f64; 2]]) -> Vec<f64> {
    let [nx, ny] = grid.n;
    let [dx, dy] = grid.pitch_mm;
    let mut f = vec![0.0; nx * ny];
    let n = poly.len();
    if n < 3 {
        return f;
    }
    // Cell (i, j) spans [ex + i·dx, ex + (i + 1)·dx] × [ey + j·dy, ey + (j + 1)·dy].
    let ex = grid.coord(0, 0) - 0.5 * dx;
    let ey = grid.coord(1, 0) - 0.5 * dy;
    // The window, two cells wider each way: an edge is walked only there.
    let (wx0, wx1) = (ex - 2.0 * dx, ex + (nx as f64 + 2.0) * dx);
    let (wy0, wy1) = (ey - 2.0 * dy, ey + (ny as f64 + 2.0) * dy);
    let mut edge = vec![false; nx * ny];
    let step = 0.25 * dx.min(dy);
    for k in 0..n {
        let (p, q) = (poly[k], poly[(k + 1) % n]);
        // Liang–Barsky: the part of the edge inside the widened window.
        let (ddx, ddy) = (q[0] - p[0], q[1] - p[1]);
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let mut keep = true;
        for (pp, qq) in [(-ddx, p[0] - wx0), (ddx, wx1 - p[0]), (-ddy, p[1] - wy0), (ddy, wy1 - p[1])] {
            if pp == 0.0 {
                if qq < 0.0 {
                    keep = false;
                }
            } else {
                let r = qq / pp;
                if pp < 0.0 {
                    t0 = t0.max(r);
                } else {
                    t1 = t1.min(r);
                }
            }
        }
        if !keep || t0 > t1 {
            continue;
        }
        let len = (ddx * ddx + ddy * ddy).sqrt() * (t1 - t0);
        let m = (len / step).ceil().max(1.0) as usize;
        for s in 0..=m {
            let t = t0 + (t1 - t0) * s as f64 / m as f64;
            let (x, y) = (p[0] + t * ddx, p[1] + t * ddy);
            let (ci, cj) = (((x - ex) / dx).floor() as i64, ((y - ey) / dy).floor() as i64);
            for j in cj - 1..=cj + 1 {
                for i in ci - 1..=ci + 1 {
                    if i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny {
                        edge[j as usize * nx + i as usize] = true;
                    }
                }
            }
        }
    }
    let mut xs: Vec<f64> = Vec::new();
    for j in 0..ny {
        let yc = ey + (j as f64 + 0.5) * dy;
        xs.clear();
        for k in 0..n {
            let (a, b) = (poly[k], poly[(k + 1) % n]);
            if (a[1] > yc) != (b[1] > yc) {
                xs.push(a[0] + (yc - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks_exact(2) {
            // The cells whose centre ex + (i + ½)dx lies in [left, right).
            let i0 = ((pair[0] - ex) / dx - 0.5).ceil().clamp(0.0, nx as f64) as usize;
            let i1 = ((pair[1] - ex) / dx - 0.5).ceil().clamp(0.0, nx as f64) as usize;
            for v in &mut f[j * nx + i0..j * nx + i1.max(i0)] {
                *v = 1.0;
            }
        }
    }
    let cell = dx * dy;
    for j in 0..ny {
        for i in 0..nx {
            if edge[j * nx + i] {
                let (x0, y0) = (ex + i as f64 * dx, ey + j as f64 * dy);
                f[j * nx + i] = (polygon_rect_area(poly, x0, x0 + dx, y0, y0 + dy) / cell).clamp(0.0, 1.0);
            }
        }
    }
    f
}

/// A regular polygon of `n` sides about `c` with the area of the disc of
/// radius `r`: its circumradius is `r·√(θ / sin θ)`, `θ = 2π/n`, so its edges
/// cross the circle and depart from it by at most about `r·(π/n)²/2`.
pub fn disc_polygon(c: [f64; 2], r: f64, n: usize) -> Vec<[f64; 2]> {
    let n = n.max(8);
    let th = 2.0 * PI / n as f64;
    let rr = r * (th / th.sin()).sqrt();
    (0..n)
        .map(|k| {
            let a = k as f64 * th;
            [c[0] + rr * a.cos(), c[1] + rr * a.sin()]
        })
        .collect()
}

/// Sides for a disc of radius `r` drawn as a polygon on a grid of `pitch`:
/// enough that the polygon departs from the circle by under a five-hundredth
/// of a sample, at most 1024.
pub fn disc_sides(r: f64, pitch: f64) -> usize {
    let tol = 2e-3 * pitch;
    ((PI * (r.abs() / (2.0 * tol)).sqrt()).ceil() as usize).clamp(64, 1024)
}

/// The transmittance of an aperture on `u`'s grid: each sample's covered
/// fraction of its cell. `opaque_inside` inverts it (a stop, an obscuration).
pub fn aperture_mask(u: &Field2, shape: &Shape, opaque_inside: bool) -> Result<Vec<f64>, WaveError> {
    let mut t = coverage(&u.grid, shape)?;
    if opaque_inside {
        for v in &mut t {
            *v = 1.0 - *v;
        }
    }
    Ok(t)
}

/// Multiply `u` by the covered fractions `t`, one per sample, and book what
/// that does. A cell passing a fraction `t` of its area transmits `t·|U|²ΔA`;
/// the opaque rest, `(1 − t)·|U|²ΔA`, is returned as the first number. The
/// sample `t·U` carries only `t²·|U|²ΔA`: the difference, the second number,
/// is the sub-cell projection error of the edge (`subcell_projection`), the
/// norm the cell-average projection discards. That is not the same claim as
/// "beyond the grid's band"; what the band cannot hold is a separate
/// measurement.
pub fn apply_coverage(u: &mut Field2, t: &[f64]) -> Result<(f64, f64), WaveError> {
    if t.len() != u.data.len() {
        return Err(WaveError::Mismatch("mask and field differ in size".into()));
    }
    let da = u.grid.cell_area_mm2();
    let (mut stopped, mut beyond) = (0.0, 0.0);
    for (z, ti) in u.data.iter_mut().zip(t) {
        let i = z.norm_sqr() * da;
        stopped += (1.0 - ti) * i;
        beyond += (ti - ti * ti) * i;
        *z *= *ti;
    }
    Ok((stopped, beyond))
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
    let before = u.power();
    let (absorbed, beyond) = apply_coverage(u, &t)?;
    report.ledger.absorbed_aperture += absorbed;
    report.ledger.subcell_projection += beyond;
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

/// What a sharp-edged mask does to a uniform unit field on `grid`, read four
/// ways apart (Q3 and R19 of the expert review of the wave engine).
///
/// The samples carry the mask as each cell's covered fraction `f` times the
/// field, the coherent amplitude this crate lays on every edge. So the power
/// they carry, `Σ f² ΔA`, falls short of what the open area passes, `Σ f ΔA`,
/// by the sub-cell projection error `Σ (f − f²) ΔA` (`subcell_projection` on
/// a path), exactly. That error is not the power beyond the grid's band: a
/// cell average is not a band projection, and an edge on a cell boundary has
/// `f` of 0 or 1 everywhere and still a spectrum without end. The power the
/// true mask holds beyond the band is measured here apart: the mask laid `s`
/// times finer per axis (its cells tiling the grid's exactly), its spectrum
/// taken on the grid's own frequencies, and what the band holds of it
/// subtracted from the open area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeBudget {
    /// What the open area passes: `Σ f ΔA`, the area itself.
    pub transmitted: f64,
    /// What the grid's samples carry: `Σ f² ΔA`.
    pub carried: f64,
    /// The sub-cell projection error, `Σ (f − f²) ΔA`.
    pub subcell: f64,
    /// The power the mask holds beyond the grid's band: the open area less
    /// the finer mask's power on the grid's own frequencies.
    pub band_cut: f64,
    /// The coherent error in the band: the grid's spectrum against the finer
    /// mask's on the grid's frequencies, over the latter's norm there.
    pub coherent_in_band: f64,
}

/// [`EdgeBudget`] of `shape` on `grid`, the finer mask `s` times finer per
/// axis (`s ≥ 3`).
pub fn edge_budget(shape: &Shape, grid: &Grid2, s: usize, fft: &mut crate::Fft2) -> Result<EdgeBudget, WaveError> {
    if s < 3 {
        return Err(WaveError::OutOfRange("the finer mask must be at least three times finer".into()));
    }
    let [dx, dy] = grid.pitch_mm;
    let da = dx * dy;
    let t = coverage(grid, shape)?;
    let transmitted: f64 = t.iter().sum::<f64>() * da;
    let carried: f64 = t.iter().map(|f| f * f).sum::<f64>() * da;
    let subcell: f64 = t.iter().map(|f| f - f * f).sum::<f64>() * da;
    // The finer grid: the same window, its cells tiling the grid's, so its
    // centre sits (s − 1)/(2s) of a cell off the grid's.
    let sh = |d: f64| -(s as f64 - 1.0) / (2.0 * s as f64) * d;
    let fine = Grid2::new([grid.n[0] * s, grid.n[1] * s], [dx / s as f64, dy / s as f64], [grid.center_mm[0] + sh(dx), grid.center_mm[1] + sh(dy)])?;
    let tf = coverage(&fine, shape)?;
    let mut uc = Field2::zeros(*grid, 1.0)?;
    for (z, f) in uc.data.iter_mut().zip(&t) {
        *z = Cx::real(*f);
    }
    let mut uf = Field2::zeros(fine, 1.0)?;
    for (z, f) in uf.data.iter_mut().zip(&tf) {
        *z = Cx::real(*f);
    }
    // The grid's own frequencies, both spectra in absolute coordinates.
    let df = [grid.freq_pitch(0), grid.freq_pitch(1)];
    let f0 = [-((grid.n[0] / 2) as f64) * df[0], -((grid.n[1] / 2) as f64) * df[1]];
    let sc = crate::czt::czt2(&uc, f0, df, grid.n, fft);
    let sf = crate::czt::czt2(&uf, f0, df, grid.n, fft);
    let dfa = df[0] * df[1];
    let in_band: f64 = sf.iter().map(|z| z.norm_sqr()).sum::<f64>() * dfa;
    let diff: f64 = sc.iter().zip(&sf).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>() * dfa;
    Ok(EdgeBudget { transmitted, carried, subcell, band_cut: transmitted - in_band, coherent_in_band: (diff / in_band.max(f64::MIN_POSITIVE)).sqrt() })
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
    fn polygon_coverage_is_the_clipped_area_on_every_cell() {
        // The scanline fill and the band of clipped cells must give, cell by
        // cell, what clipping the polygon to that cell gives.
        let g = Grid2::new([41, 30], [0.1, 0.13], [0.2, -0.1]).unwrap();
        let polys = [
            // A concave L, wholly inside.
            vec![[-1.0, -1.0], [1.5, -1.0], [1.5, 0.2], [0.1, 0.2], [0.1, 1.4], [-1.0, 1.4]],
            // A triangle reaching far outside the window.
            vec![[-30.0, -0.7], [3.1, -0.2], [0.4, 40.0]],
            // A square enclosing the window: every cell covered.
            vec![[-50.0, -50.0], [50.0, -50.0], [50.0, 50.0], [-50.0, 50.0]],
            // A many-sided disc, clockwise.
            disc_polygon([0.31, -0.07], 1.13, 300).into_iter().rev().collect(),
        ];
        let [dx, dy] = g.pitch_mm;
        for poly in &polys {
            let f = polygon_coverage(&g, poly);
            let mut worst: f64 = 0.0;
            for j in 0..g.n[1] {
                for i in 0..g.n[0] {
                    let (x, y) = g.xy(i, j);
                    let want = (polygon_rect_area(poly, x - 0.5 * dx, x + 0.5 * dx, y - 0.5 * dy, y + 0.5 * dy) / (dx * dy)).clamp(0.0, 1.0);
                    worst = worst.max((f[g.idx(i, j)] - want).abs());
                }
            }
            assert!(worst < 1e-12, "{worst} for {poly:?}");
        }
    }

    #[test]
    fn a_disc_polygon_has_the_disc_s_area_and_hugs_its_circle() {
        let (r, n) = (1.7, 128);
        let p = disc_polygon([0.4, -0.2], r, n);
        assert!((shoelace(&p).abs() - PI * r * r).abs() < 1e-12 * PI * r * r);
        // Its edges depart from the circle by at most about r·(π/n)²/2.
        let mid = |a: [f64; 2], b: [f64; 2]| (((a[0] + b[0]) / 2.0 - 0.4).hypot((a[1] + b[1]) / 2.0 + 0.2) - r).abs();
        let worst = (0..n).map(|k| mid(p[k], p[(k + 1) % n])).fold(0.0, f64::max);
        assert!(worst < r * (PI / n as f64).powi(2) / 2.0, "{worst}");
        // Enough sides for a five-hundredth of a sample.
        let sides = disc_sides(0.5, 0.003);
        assert!(0.5 * (PI / sides as f64).powi(2) / 2.0 <= 2e-3 * 0.003 * (1.0 + 1e-9), "{sides}");
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
