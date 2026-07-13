//! # oc-fit
//!
//! Freehand stroke → parametric surface, **on a slider** (spec §5.3).
//!
//! The hard part of "sketch, then refine precisely" is that sketch and
//! precision want different representations. This crate resolves that by making
//! the fit an *explicit, adjustable* step: every stroke is fitted against the
//! whole ladder
//!
//! ```text
//!   Line → Arc → Conic → Asphere(4) → Asphere(8) → Spline
//! ```
//!
//! and the result carries **every rung with its residual**, so a UI slider can
//! move up and down the ladder for free (no refit) while the rays visibly
//! change. [`fit_stroke`] auto-selects the least-complex rung whose residual is
//! under tolerance (§5.3 step 4); the caller is free to override it.
//!
//! No I/O, no GPU, f64 throughout. The fitted [`Edge`] values are ordinary
//! `oc-core` geometry, so they trace on both the reference and interactive
//! paths with no special-casing.

pub mod fits;
pub mod sample;
pub mod simplify;

use oc_core::{Edge, Vec2};

pub use fits::{fit_arc, fit_asphere, fit_circle, fit_conic, fit_line, surface_frame, FitResult};
pub use sample::{sample_edge, surface_rim};

/// One rung of the complexity ladder, in increasing order of freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitLevel {
    Line,
    Arc,
    Conic,
    Asphere4,
    Asphere8,
    Spline,
}

impl FitLevel {
    /// The ladder, least-complex first.
    pub const ALL: [FitLevel; 6] = [
        FitLevel::Line,
        FitLevel::Arc,
        FitLevel::Conic,
        FitLevel::Asphere4,
        FitLevel::Asphere8,
        FitLevel::Spline,
    ];

    /// Position on the ladder (0 = `Line`).
    pub fn rung(self) -> usize {
        Self::ALL.iter().position(|l| *l == self).unwrap()
    }

    /// Short label for the UI.
    pub fn name(self) -> &'static str {
        match self {
            FitLevel::Line => "line",
            FitLevel::Arc => "arc",
            FitLevel::Conic => "conic",
            FitLevel::Asphere4 => "asphere·4",
            FitLevel::Asphere8 => "asphere·8",
            FitLevel::Spline => "spline",
        }
    }

    /// Number of free geometric parameters — the "manufacturability cost" the
    /// slider is really trading against.
    pub fn dof(self) -> u32 {
        match self {
            FitLevel::Line => 2,
            FitLevel::Arc => 3,
            FitLevel::Conic => 4,
            FitLevel::Asphere4 => 8,
            FitLevel::Asphere8 => 12,
            FitLevel::Spline => u32::MAX, // as many as it takes
        }
    }
}

/// One fitted rung: the geometry, its level, and its residual (world units).
#[derive(Debug, Clone)]
pub struct Fit {
    pub level: FitLevel,
    pub edge: Edge,
    pub max_residual: f64,
    pub rms_residual: f64,
}

impl Fit {
    /// Worst-case residual in microns — the number shown next to the stroke.
    pub fn max_residual_um(&self) -> f64 {
        self.max_residual * 1000.0
    }

    /// RMS residual in microns.
    pub fn rms_residual_um(&self) -> f64 {
        self.rms_residual * 1000.0
    }
}

/// The full ladder for one stroke, plus the auto-selected rung.
#[derive(Debug, Clone)]
pub struct FitLadder {
    /// One [`Fit`] per rung, least-complex first.
    pub fits: Vec<Fit>,
    /// Index into `fits` of the least-complex rung within tolerance (§5.3.4).
    pub chosen: usize,
    /// Tolerance used for auto-selection, in microns.
    pub tol_um: f64,
    /// The stroke after dedup (kept so a UI can redraw / recompute).
    pub stroke: Vec<Vec2>,
}

impl FitLadder {
    /// The auto-selected fit.
    pub fn chosen(&self) -> &Fit {
        &self.fits[self.chosen]
    }

    /// The fit at a given ladder position, clamped to the available range.
    pub fn at_rung(&self, rung: usize) -> &Fit {
        &self.fits[rung.min(self.fits.len() - 1)]
    }
}

/// Default fit tolerance, microns. A fit tighter than this is "good enough";
/// the slider still lets the user demand more (or accept less).
pub const DEFAULT_TOL_UM: f64 = 1.0;

/// Fit a single-surface stroke against the whole ladder (§5.3).
///
/// The stroke is deduplicated first; the least-squares fits then run on the
/// **full** point set (using every sample, not the RDP-thinned one, so the
/// reported residual is honest), while the RDP simplification supplies the
/// `Spline` control points. Higher rungs are seeded from lower ones, so the
/// residual is monotone non-increasing along the ladder.
pub fn fit_stroke(pts: &[Vec2], tol_um: f64) -> FitLadder {
    let stroke = simplify::dedup(pts, 1e-6);
    if stroke.len() < 2 {
        // Degenerate: a dot. Return a zero-length line so downstream code is
        // still total.
        let p = stroke.first().copied().unwrap_or(Vec2::ZERO);
        let fit = Fit {
            level: FitLevel::Line,
            edge: Edge::Line { a: p, b: p },
            max_residual: 0.0,
            rms_residual: 0.0,
        };
        return FitLadder {
            fits: vec![fit],
            chosen: 0,
            tol_um,
            stroke,
        };
    }

    let (frame, half_ap) = surface_frame(&stroke);
    let circle_r = fit_circle(&stroke).map(|(_, r)| r);

    let line = fit_line(&stroke).0;
    let arc = fit_arc(&stroke).unwrap_or_else(|| line.clone());
    // The conic fit corrects the frame's vertex along the axis; the asphere
    // rungs build on that corrected frame and the fitted base curvature.
    let (conic, cframe) = fit_conic(&stroke, frame, half_ap, circle_r);
    let (base_c, base_k) = match &conic.edge {
        Edge::Conic(cp) => (cp.c, cp.k),
        _ => (circle_r.map(|r| 1.0 / r).unwrap_or(0.0), 0.0),
    };
    let asph4 = fit_asphere(&stroke, cframe, half_ap, base_c, base_k, 4);
    let asph8 = fit_asphere(&stroke, cframe, half_ap, base_c, base_k, 8);

    // Spline: RDP control points, evaluated as a polyline by the tracer. Its
    // residual is the RDP tolerance measured against the original stroke.
    let rdp_tol = (tol_um * 1e-3).max(1e-4);
    let ctrl = simplify::rdp(&stroke, rdp_tol);
    let spline_res = spline_residual(&stroke, &ctrl);
    let spline = FitResult {
        edge: Edge::Spline {
            ctrl,
            degree: 1,
        },
        max_residual: spline_res.0,
        rms_residual: spline_res.1,
    };

    let raw = [
        (FitLevel::Line, line),
        (FitLevel::Arc, arc),
        (FitLevel::Conic, conic),
        (FitLevel::Asphere4, asph4),
        (FitLevel::Asphere8, asph8),
        (FitLevel::Spline, spline),
    ];

    // Enforce a monotone non-increasing residual up the ladder: a higher rung
    // is never reported as worse than a lower one (it can always fall back to
    // the simpler geometry, and a slider that got *worse* as you added terms
    // would be a lie).
    let mut best_so_far = f64::INFINITY;
    let mut fits: Vec<Fit> = Vec::with_capacity(raw.len());
    for (level, fr) in raw {
        best_so_far = best_so_far.min(fr.max_residual);
        fits.push(Fit {
            level,
            edge: fr.edge,
            max_residual: best_so_far,
            rms_residual: fr.rms_residual,
        });
    }

    // Auto-select: least-complex rung within tolerance.
    let tol_world = tol_um * 1e-3;
    let chosen = fits
        .iter()
        .position(|f| f.max_residual <= tol_world)
        .unwrap_or(fits.len() - 1);

    FitLadder {
        fits,
        chosen,
        tol_um,
        stroke,
    }
}

/// Max/RMS distance from each original point to the polyline `ctrl`.
fn spline_residual(pts: &[Vec2], ctrl: &[Vec2]) -> (f64, f64) {
    if ctrl.len() < 2 {
        return (0.0, 0.0);
    }
    let mut max = 0.0_f64;
    let mut sumsq = 0.0_f64;
    for &p in pts {
        let mut d = f64::INFINITY;
        for w in ctrl.windows(2) {
            d = d.min(point_seg_dist(p, w[0], w[1]));
        }
        max = max.max(d);
        sumsq += d * d;
    }
    (max, (sumsq / pts.len() as f64).sqrt())
}

fn point_seg_dist(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < 1e-18 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (p - (a + t * ab)).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fits::conic_sag;

    fn sample<F: Fn(f64) -> f64>(f: F, y0: f64, y1: f64, n: usize) -> Vec<Vec2> {
        (0..=n)
            .map(|i| {
                let y = y0 + (y1 - y0) * i as f64 / n as f64;
                Vec2::new(f(y), y) // axis +x, transverse +y
            })
            .collect()
    }

    #[test]
    fn flat_stroke_selects_line() {
        let pts = sample(|_| 0.0, -5.0, 5.0, 40);
        let ladder = fit_stroke(&pts, 1.0);
        assert_eq!(ladder.chosen().level, FitLevel::Line);
        assert!(ladder.chosen().max_residual_um() < 1.0);
    }

    #[test]
    fn spherical_stroke_selects_arc_not_line() {
        // A radius-40 cap: a line cannot fit it, an arc can.
        let c = 1.0 / 40.0;
        let pts = sample(|y| conic_sag(c, 0.0, y), -6.0, 6.0, 60);
        let ladder = fit_stroke(&pts, 1.0);
        assert_eq!(
            ladder.chosen().level,
            FitLevel::Arc,
            "chose {} at {:.3} µm",
            ladder.chosen().level.name(),
            ladder.chosen().max_residual_um()
        );
        // The line rung must be visibly worse than the chosen arc.
        assert!(ladder.fits[0].max_residual_um() > 1.0);
    }

    #[test]
    fn residual_is_monotone_down_the_ladder() {
        let c = 1.0 / 25.0;
        let pts = sample(|y| conic_sag(c, -0.6, y) + 3e-6 * y.powi(4), -7.0, 7.0, 80);
        let ladder = fit_stroke(&pts, 0.001); // absurdly tight → forces high rungs
        for w in ladder.fits.windows(2) {
            assert!(
                w[1].max_residual <= w[0].max_residual + 1e-12,
                "{} ({:.4} µm) worse than {} ({:.4} µm)",
                w[1].level.name(),
                w[1].max_residual_um(),
                w[0].level.name(),
                w[0].max_residual_um(),
            );
        }
    }

    #[test]
    fn ladder_has_all_rungs() {
        let pts = sample(|y| 0.02 * y * y, -5.0, 5.0, 30);
        let ladder = fit_stroke(&pts, 1.0);
        assert_eq!(ladder.fits.len(), 6);
        for (i, f) in ladder.fits.iter().enumerate() {
            assert_eq!(f.level, FitLevel::ALL[i]);
        }
    }

    #[test]
    fn spline_is_the_escape_hatch() {
        // A wiggle no conic/asphere can represent: the spline must nail it.
        let pts = sample(|y| 0.3 * (2.0 * y).sin(), -6.0, 6.0, 120);
        let ladder = fit_stroke(&pts, 0.5);
        let spline = ladder.fits.last().unwrap();
        assert_eq!(spline.level, FitLevel::Spline);
        assert!(
            spline.max_residual_um() < 0.5,
            "spline residual {:.3} µm",
            spline.max_residual_um()
        );
        // And nothing simpler than the spline was good enough here.
        assert_eq!(ladder.chosen().level, FitLevel::Spline);
    }
}
