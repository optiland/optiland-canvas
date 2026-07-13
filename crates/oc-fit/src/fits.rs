//! Least-squares fits of a point set against each rung of the complexity
//! ladder (step 3 of §5.3): `Line → Arc → Conic → Asphere(4) → Asphere(8)`.
//!
//! Every fit reports a **geometric** residual (perpendicular distance from the
//! original stroke points to the fitted curve), in world units. The caller
//! converts to microns for display. Higher rungs are seeded from lower ones so
//! adding complexity never makes the fit worse — which is what lets the
//! complexity slider read as a smooth "manufacturability dial" (§5.3).

use oc_core::{ConicParams, Edge, Vec2};

/// A rigid local frame for a surface of revolution: a vertex, the axis of
/// symmetry `axis` (unit, pointing into the curvature), and the transverse
/// tangent `tang = perp(axis)`.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub vertex: Vec2,
    pub axis: Vec2,
    pub tang: Vec2,
}

impl Frame {
    /// Transverse (`y`) and sag (`z`) of a world point in this frame.
    #[inline]
    pub fn yz(&self, p: Vec2) -> (f64, f64) {
        let d = p - self.vertex;
        (d.dot(self.tang), d.dot(self.axis))
    }
}

/// Result of one fit: the geometry plus its residual statistics (world units).
#[derive(Debug, Clone)]
pub struct FitResult {
    pub edge: Edge,
    pub max_residual: f64,
    pub rms_residual: f64,
}

// --- tiny linear algebra ---------------------------------------------------

/// Solve `A x = b` for a small dense system by Gaussian elimination with
/// partial pivoting. `a` is row-major `n×n`. Returns `None` if singular.
pub(crate) fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for r in (col + 1)..n {
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[i * n + c] * x[c];
        }
        x[i] = s / a[i * n + i];
    }
    Some(x)
}

// --- sag models ------------------------------------------------------------

/// Sag of a conic of revolution at transverse coordinate `y`.
pub(crate) fn conic_sag(c: f64, k: f64, y: f64) -> f64 {
    let r2 = y * y;
    let disc = 1.0 - (1.0 + k) * c * c * r2;
    c * r2 / (1.0 + disc.max(0.0).sqrt())
}

/// Sag of `conic + Σ coeffs[i]·y^(2i+4)` (matches [`Edge::Asphere`]).
pub(crate) fn asphere_sag(c: f64, k: f64, coeffs: &[f64; 8], y: f64) -> f64 {
    let mut sag = conic_sag(c, k, y);
    let r2 = y * y;
    let mut yp = r2 * r2; // y^4
    for &a in coeffs {
        sag += a * yp;
        yp *= r2;
    }
    sag
}

/// Perpendicular residual of `(y, z)` from a sag profile `f(y)`, to first order
/// (`|z − f(y)|` corrected by the local surface slope).
fn sag_residual<F: Fn(f64) -> f64>(y: f64, z: f64, f: &F) -> f64 {
    let h = 1e-6 * (1.0 + y.abs());
    let slope = (f(y + h) - f(y - h)) / (2.0 * h);
    (z - f(y)).abs() / (1.0 + slope * slope).sqrt()
}

fn stats<I: Iterator<Item = f64>>(residuals: I) -> (f64, f64) {
    let mut max = 0.0_f64;
    let mut sumsq = 0.0_f64;
    let mut n = 0usize;
    for r in residuals {
        max = max.max(r);
        sumsq += r * r;
        n += 1;
    }
    let rms = if n > 0 { (sumsq / n as f64).sqrt() } else { 0.0 };
    (max, rms)
}

// --- Line (total least squares) --------------------------------------------

/// Best-fit line by total least squares (minimises perpendicular distance).
/// Returns the fit and the unit line direction.
pub fn fit_line(pts: &[Vec2]) -> (FitResult, Vec2) {
    let n = pts.len().max(1) as f64;
    let centroid = pts.iter().copied().fold(Vec2::ZERO, |a, p| a + p) / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &p in pts {
        let d = p - centroid;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }
    // Major eigenvector of the 2×2 covariance = line direction.
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir = Vec2::new(theta.cos(), theta.sin());
    let normal = Vec2::new(-dir.y, dir.x);

    let (max, rms) = stats(pts.iter().map(|&p| (p - centroid).dot(normal).abs()));
    // Endpoints = extreme projections of the stroke onto the line.
    let ts: Vec<f64> = pts.iter().map(|&p| (p - centroid).dot(dir)).collect();
    let tmin = ts.iter().copied().fold(f64::INFINITY, f64::min);
    let tmax = ts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let a = centroid + tmin * dir;
    let b = centroid + tmax * dir;
    (
        FitResult {
            edge: Edge::Line { a, b },
            max_residual: max,
            rms_residual: rms,
        },
        dir,
    )
}

// --- Circle / Arc ----------------------------------------------------------

/// Geometric best-fit circle: Kåsa algebraic seed refined by Gauss–Newton on
/// the true geometric distance. Returns `(center, radius)`.
pub fn fit_circle(pts: &[Vec2]) -> Option<(Vec2, f64)> {
    if pts.len() < 3 {
        return None;
    }
    // Kåsa: minimise Σ (x²+y²+Dx+Ey+F)². Columns [x, y, 1], rhs −(x²+y²).
    let mut ata = vec![0.0; 9];
    let mut atb = vec![0.0; 3];
    for &p in pts {
        let row = [p.x, p.y, 1.0];
        let rhs = -(p.x * p.x + p.y * p.y);
        for i in 0..3 {
            for j in 0..3 {
                ata[i * 3 + j] += row[i] * row[j];
            }
            atb[i] += row[i] * rhs;
        }
    }
    let sol = solve_linear(ata, atb, 3)?;
    let mut cx = -0.5 * sol[0];
    let mut cy = -0.5 * sol[1];
    let mut r = (cx * cx + cy * cy - sol[2]).max(0.0).sqrt();
    if !r.is_finite() || r < 1e-9 {
        return None;
    }

    // Gauss–Newton on residual_i = |p_i − c| − r, unknowns (cx, cy, r).
    for _ in 0..24 {
        let mut ata = vec![0.0; 9];
        let mut atb = vec![0.0; 3];
        for &p in pts {
            let dx = p.x - cx;
            let dy = p.y - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d < 1e-12 {
                continue;
            }
            // Jacobian of residual wrt (cx, cy, r).
            let j = [-dx / d, -dy / d, -1.0];
            let res = d - r;
            for i in 0..3 {
                for m in 0..3 {
                    ata[i * 3 + m] += j[i] * j[m];
                }
                atb[i] += -j[i] * res;
            }
        }
        match solve_linear(ata, atb, 3) {
            Some(step) => {
                cx += step[0];
                cy += step[1];
                r += step[2];
                if step[0].hypot(step[1]).max(step[2].abs()) < 1e-12 {
                    break;
                }
            }
            None => break,
        }
    }
    if r.is_finite() && r > 1e-9 {
        Some((Vec2::new(cx, cy), r))
    } else {
        None
    }
}

/// Fit an `Arc` edge. The sweep is the angular span of the points about the
/// fitted centre, oriented CCW so [`Edge::Arc`] reproduces the stroke.
pub fn fit_arc(pts: &[Vec2]) -> Option<FitResult> {
    let (center, r) = fit_circle(pts)?;
    // Unwrap angles relative to the first point so a span crossing ±π is handled.
    let a0raw = {
        let d = pts[0] - center;
        d.y.atan2(d.x)
    };
    let mut lo = 0.0_f64;
    let mut hi = 0.0_f64;
    for &p in pts {
        let d = p - center;
        let mut a = d.y.atan2(d.x) - a0raw;
        while a > std::f64::consts::PI {
            a -= std::f64::consts::TAU;
        }
        while a < -std::f64::consts::PI {
            a += std::f64::consts::TAU;
        }
        lo = lo.min(a);
        hi = hi.max(a);
    }
    let (a0, a1) = (a0raw + lo, a0raw + hi); // CCW from a0 to a1
    let (max, rms) = stats(pts.iter().map(|&p| ((p - center).length() - r).abs()));
    Some(FitResult {
        edge: Edge::Arc {
            center,
            r,
            a0,
            a1,
        },
        max_residual: max,
        rms_residual: rms,
    })
}

// --- the surface-of-revolution frame ---------------------------------------

/// Establish the fitting frame (vertex + axis) from the circle fit, falling
/// back to the line fit for near-flat strokes. `half_ap` is `max |y|`.
pub fn surface_frame(pts: &[Vec2]) -> (Frame, f64) {
    let n = pts.len().max(1) as f64;
    let centroid = pts.iter().copied().fold(Vec2::ZERO, |a, p| a + p) / n;

    let (line, dir) = fit_line(pts);
    let make = |axis: Vec2, vertex: Vec2| {
        let tang = Vec2::new(-axis.y, axis.x);
        let half_ap = pts
            .iter()
            .map(|&p| (p - vertex).dot(tang).abs())
            .fold(0.0_f64, f64::max);
        (
            Frame {
                vertex,
                axis,
                tang,
            },
            half_ap.max(1e-6),
        )
    };

    match fit_circle(pts) {
        Some((o, r)) if r < 1.0e6 => {
            let n_hat = (centroid - o).normalize_or_zero();
            if n_hat == Vec2::ZERO {
                // centroid at the centre — degenerate; use the line normal.
                let axis = Vec2::new(-dir.y, dir.x);
                return make(axis, centroid);
            }
            let vertex = o + r * n_hat;
            let axis = (o - vertex).normalize(); // points from vertex toward centre
            make(axis, vertex)
        }
        _ => {
            // Near-flat: axis is the line normal, oriented toward the bulge.
            let mut axis = Vec2::new(-dir.y, dir.x);
            let bulge: f64 = pts.iter().map(|&p| (p - centroid).dot(axis)).sum();
            if bulge < 0.0 {
                axis = -axis;
            }
            let _ = line;
            make(axis, centroid)
        }
    }
}

// --- Conic -----------------------------------------------------------------

/// Recovered conic parameters in a frame: curvature, conic constant, and the
/// vertex shift `delta` along the axis (frame origin → true vertex).
struct ConicSolve {
    c: f64,
    k: f64,
    delta: f64,
}

/// Fit a conic of revolution by a **linear implicit** solve, given only the
/// axis *direction* (the frame's `axis`/`tang`). Solving the implicit form
/// `A y² + B z² + z + C = 0` (the `z`-coefficient normalised to 1, which is
/// always well-conditioned because the physical form's linear term is the
/// non-vanishing `−2Z`) recovers curvature, conic constant, **and** the vertex
/// offset along the axis in one shot — so a mis-placed frame origin no longer
/// biases `(c, k)`. Returns `None` when the data is too flat to be a conic.
fn solve_conic_implicit(yz: &[(f64, f64)]) -> Option<ConicSolve> {
    // Least squares: A·y² + B·z² + C = −z.
    let mut ata = vec![0.0; 9];
    let mut atb = vec![0.0; 3];
    for &(y, z) in yz {
        let row = [y * y, z * z, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                ata[i * 3 + j] += row[i] * row[j];
            }
            atb[i] += row[i] * (-z);
        }
    }
    let sol = solve_linear(ata, atb, 3)?;
    let (a, b, cc) = (sol[0], sol[1], sol[2]);
    if !a.is_finite() || a.abs() < 1e-12 {
        return None; // effectively flat: let the circle/line rungs handle it
    }
    let k = b / a - 1.0;
    // Vertex offset δ solves 3B·δ² + δ + C = 0 (root nearest the frame origin).
    let delta = if b.abs() < 1e-12 {
        -cc
    } else {
        let disc = 1.0 - 12.0 * b * cc;
        if disc < 0.0 {
            -cc // fall back to the linearised root
        } else {
            let sq = disc.sqrt();
            let d1 = (-1.0 + sq) / (6.0 * b);
            let d2 = (-1.0 - sq) / (6.0 * b);
            if d1.abs() <= d2.abs() {
                d1
            } else {
                d2
            }
        }
    };
    let scale = 1.0 + 2.0 * delta * b;
    if scale.abs() < 1e-12 {
        return None;
    }
    let c = -2.0 * a / scale;
    if !c.is_finite() || !k.is_finite() {
        return None;
    }
    Some(ConicSolve { c, k, delta })
}

/// Fit a `Conic`, correcting the frame's vertex along the axis as part of the
/// solve. Keeps the circle's curvature (a sphere) whenever the conic constant
/// is ill-determined — a shallow cap genuinely cannot tell a sphere from a
/// parabola, and the fitter shouldn't pretend otherwise. Returns the fit and
/// the **corrected** frame, so the asphere rungs build on the right vertex.
pub fn fit_conic(
    pts: &[Vec2],
    frame: Frame,
    half_ap: f64,
    circle_r: Option<f64>,
) -> (FitResult, Frame) {
    let yz: Vec<(f64, f64)> = pts.iter().map(|&p| frame.yz(p)).collect();
    let seed_c = circle_r.map(|r| 1.0 / r).unwrap_or(0.0);
    // Sign of curvature from the mean sag/y² (sag is +along +axis by convention).
    let mean_curv: f64 = yz
        .iter()
        .map(|&(y, z)| if y.abs() > 1e-9 { 2.0 * z / (y * y) } else { 0.0 })
        .sum::<f64>()
        / yz.len().max(1) as f64;
    let seed_c = if seed_c.abs() > 0.0 {
        seed_c.copysign(mean_curv)
    } else {
        mean_curv
    };

    // Implicit solve for (c, k, δ).
    let solved = solve_conic_implicit(&yz).filter(|s| {
        s.k.abs() < 40.0 && s.c.signum() == seed_c.signum() && s.delta.abs() < 10.0 * half_ap
    });
    let (c, k, corrected) = match solved {
        Some(s) => {
            let vertex = frame.vertex + s.delta * frame.axis;
            (
                s.c,
                s.k,
                Frame {
                    vertex,
                    ..frame
                },
            )
        }
        None => (seed_c, 0.0, frame),
    };

    let cyz: Vec<(f64, f64)> = pts.iter().map(|&p| corrected.yz(p)).collect();
    let f = |y: f64| conic_sag(c, k, y);
    let (max, rms) = stats(cyz.iter().map(|&(y, z)| sag_residual(y, z, &f)));

    // Never report worse than the sphere in the original frame: if the conic
    // constant hurt, drop back to it.
    let fs = |y: f64| conic_sag(seed_c, 0.0, y);
    let (smax, srms) = stats(yz.iter().map(|&(y, z)| sag_residual(y, z, &fs)));
    if smax <= max {
        (
            FitResult {
                edge: Edge::Conic(ConicParams {
                    vertex: frame.vertex,
                    axis: frame.axis,
                    c: seed_c,
                    k: 0.0,
                    half_ap,
                }),
                max_residual: smax,
                rms_residual: srms,
            },
            frame,
        )
    } else {
        (
            FitResult {
                edge: Edge::Conic(ConicParams {
                    vertex: corrected.vertex,
                    axis: corrected.axis,
                    c,
                    k,
                    half_ap,
                }),
                max_residual: max,
                rms_residual: rms,
            },
            corrected,
        )
    }
}

// --- Asphere ---------------------------------------------------------------

/// Fit `m` even aspheric terms on top of a fixed base conic `(c, k)`. Because
/// the base is held and only the linear polynomial coefficients are solved, the
/// fit can only improve on the conic. `y` is normalised by `half_ap` for
/// conditioning, then the coefficients are rescaled to world units.
pub fn fit_asphere(
    pts: &[Vec2],
    frame: Frame,
    half_ap: f64,
    c: f64,
    k: f64,
    m: usize,
) -> FitResult {
    let m = m.min(8);
    let yz: Vec<(f64, f64)> = pts.iter().map(|&p| frame.yz(p)).collect();

    // Residual sag to be explained by Σ a_j (y/half_ap)^(2j+4).
    let mut ata = vec![0.0; m * m];
    let mut atb = vec![0.0; m];
    for &(y, z) in &yz {
        let base = conic_sag(c, k, y);
        let target = z - base;
        let yn = y / half_ap;
        let yn2 = yn * yn;
        let mut basis = [0.0; 8];
        let mut yp = yn2 * yn2; // (y/ha)^4
        for b in basis.iter_mut().take(m) {
            *b = yp;
            yp *= yn2;
        }
        for i in 0..m {
            for j in 0..m {
                ata[i * m + j] += basis[i] * basis[j];
            }
            atb[i] += basis[i] * target;
        }
    }

    let mut coeffs = [0.0_f64; 8];
    if let Some(sol) = solve_linear(ata, atb, m) {
        // Rescale normalised coefficients back to world units.
        for (j, s) in sol.iter().enumerate() {
            let scale = half_ap.powi(2 * j as i32 + 4);
            coeffs[j] = s / scale;
        }
    }

    let f = |y: f64| asphere_sag(c, k, &coeffs, y);
    let (max, rms) = stats(yz.iter().map(|&(y, z)| sag_residual(y, z, &f)));

    FitResult {
        edge: Edge::Asphere {
            base: ConicParams {
                vertex: frame.vertex,
                axis: frame.axis,
                c,
                k,
                half_ap,
            },
            coeffs,
        },
        max_residual: max,
        rms_residual: rms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_arc(center: Vec2, r: f64, a0: f64, a1: f64, n: usize) -> Vec<Vec2> {
        (0..=n)
            .map(|i| {
                let a = a0 + (a1 - a0) * i as f64 / n as f64;
                center + Vec2::new(a.cos(), a.sin()) * r
            })
            .collect()
    }

    #[test]
    fn line_fit_recovers_a_line() {
        let pts: Vec<Vec2> = (0..=20)
            .map(|i| Vec2::new(i as f64 * 0.5, 3.0 + 0.0 * i as f64))
            .collect();
        let (fit, dir) = fit_line(&pts);
        assert!(fit.max_residual < 1e-12);
        assert!(dir.x.abs() > 0.999); // horizontal
    }

    #[test]
    fn circle_fit_recovers_radius() {
        // A 40° cap of a radius-50 circle centred at (50, 0): a realistic
        // shallow optical surface, where the algebraic seed alone is biased.
        let pts = sample_arc(Vec2::new(50.0, 0.0), 50.0, std::f64::consts::PI - 0.35, std::f64::consts::PI + 0.35, 40);
        let (c, r) = fit_circle(&pts).unwrap();
        assert!((r - 50.0).abs() < 1e-6, "radius {r}");
        assert!((c - Vec2::new(50.0, 0.0)).length() < 1e-6, "center {c:?}");
    }

    #[test]
    fn conic_fit_recovers_a_parabola() {
        // Sample z = c y²/2 style parabola (k = -1) with real depth, vertex at
        // origin, axis +x. Points (z_sag, y).
        let c = 1.0 / 20.0;
        let k = -1.0;
        let pts: Vec<Vec2> = (-30..=30)
            .map(|i| {
                let y = i as f64 * 0.3;
                let z = conic_sag(c, k, y);
                Vec2::new(z, y) // axis = +x, tang = +y
            })
            .collect();
        let (frame, half_ap) = surface_frame(&pts);
        let r = fit_circle(&pts).map(|(_, r)| r);
        let (fit, _frame) = fit_conic(&pts, frame, half_ap, r);
        if let Edge::Conic(cp) = fit.edge {
            assert!((cp.c.abs() - c).abs() < 1e-3, "c = {}", cp.c);
            assert!((cp.k - k).abs() < 0.05, "k = {}", cp.k);
        } else {
            panic!("expected a conic");
        }
        assert!(fit.max_residual < 1e-6, "residual {}", fit.max_residual);
    }

    #[test]
    fn asphere_improves_on_conic() {
        // A surface that is a sphere plus a real y^4 departure: the asphere rung
        // must drive the residual well below the conic rung.
        let c = 1.0 / 30.0;
        let a4 = 5e-6;
        let pts: Vec<Vec2> = (-40..=40)
            .map(|i| {
                let y = i as f64 * 0.2;
                let z = conic_sag(c, 0.0, y) + a4 * y.powi(4);
                Vec2::new(z, y)
            })
            .collect();
        let (frame, half_ap) = surface_frame(&pts);
        let r = fit_circle(&pts).map(|(_, r)| r);
        let (conic, cframe) = fit_conic(&pts, frame, half_ap, r);
        // Asphere builds on the *fitted* conic base in its corrected frame —
        // exactly how `fit_stroke` wires the rungs together.
        let (base_c, base_k) = match conic.edge {
            Edge::Conic(cp) => (cp.c, cp.k),
            _ => unreachable!(),
        };
        let asph = fit_asphere(&pts, cframe, half_ap, base_c, base_k, 4);
        assert!(
            asph.max_residual < conic.max_residual,
            "asphere {} should beat conic {}",
            asph.max_residual,
            conic.max_residual
        );
        // Sub-tenth-micron — a genuinely precise fit of a 20 µm departure.
        assert!(
            asph.max_residual < 1e-4,
            "asphere residual {} mm",
            asph.max_residual
        );
    }
}
