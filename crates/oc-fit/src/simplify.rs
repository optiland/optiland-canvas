//! Stroke pre-processing: Ramer–Douglas–Peucker simplification and corner
//! segmentation (steps 1–2 of the §5.3 pipeline).
//!
//! Simplification does two jobs: it thins a dense freehand stroke so the
//! least-squares fits are not dominated by redundant near-duplicate samples,
//! and it supplies the control points for the `Spline` escape-hatch rung.
//! Segmentation splits a stroke at curvature discontinuities (sharp corners) so
//! a closed outline becomes several surfaces rather than one wobbly one.

use oc_core::Vec2;

/// Perpendicular distance from `p` to the infinite line through `a`,`b`.
fn perp_dist(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let ab = b - a;
    let len = ab.length();
    if len < 1e-12 {
        return (p - a).length();
    }
    ((b.x - a.x) * (a.y - p.y) - (a.x - p.x) * (b.y - a.y)).abs() / len
}

/// Ramer–Douglas–Peucker: keep only the vertices needed to stay within
/// `tol` (world units) of the original polyline. Endpoints are always kept.
///
/// Returns the retained points in order. Runs iteratively (an explicit stack)
/// so a pathological stroke cannot blow the call stack.
pub fn rdp(pts: &[Vec2], tol: f64) -> Vec<Vec2> {
    let n = pts.len();
    if n <= 2 {
        return pts.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((lo, hi)) = stack.pop() {
        if hi <= lo + 1 {
            continue;
        }
        let (mut worst, mut worst_i) = (0.0_f64, lo);
        for i in (lo + 1)..hi {
            let d = perp_dist(pts[i], pts[lo], pts[hi]);
            if d > worst {
                worst = d;
                worst_i = i;
            }
        }
        if worst > tol {
            keep[worst_i] = true;
            stack.push((lo, worst_i));
            stack.push((worst_i, hi));
        }
    }
    pts.iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(*p))
        .collect()
}

/// Drop consecutive points closer than `eps` — freehand capture often stalls
/// and emits duplicates that would make direction estimates blow up.
pub fn dedup(pts: &[Vec2], eps: f64) -> Vec<Vec2> {
    let mut out: Vec<Vec2> = Vec::with_capacity(pts.len());
    for &p in pts {
        if out.last().map(|q| (p - *q).length() > eps).unwrap_or(true) {
            out.push(p);
        }
    }
    out
}

/// Split index ranges at corners: vertices whose turn angle exceeds
/// `angle_thresh` radians. Returns half-open `[start, end)` ranges into `pts`
/// that each cover one smooth segment (ranges share the corner vertex).
///
/// Robust to jitter by measuring the turn over a small window rather than
/// between immediate neighbours.
pub fn segment_corners(pts: &[Vec2], angle_thresh: f64) -> Vec<(usize, usize)> {
    let n = pts.len();
    if n < 3 {
        return vec![(0, n)];
    }
    let win = 2usize; // look this many samples either side to smooth jitter

    // Turn angle at each interior vertex.
    let turn = |i: usize| -> f64 {
        let a = pts[i.saturating_sub(win)];
        let b = pts[i];
        let c = pts[(i + win).min(n - 1)];
        let (u, v) = (b - a, c - b);
        let (lu, lv) = (u.length(), v.length());
        if lu < 1e-9 || lv < 1e-9 {
            return 0.0;
        }
        (u.dot(v) / (lu * lv)).clamp(-1.0, 1.0).acos()
    };

    // A corner is a strict local maximum of the turn angle above threshold —
    // non-maximum suppression collapses the smeared cluster a window creates
    // around a single sharp corner to one split.
    let mut corners: Vec<usize> = Vec::new();
    for i in 1..(n - 1) {
        let ti = turn(i);
        if ti <= angle_thresh {
            continue;
        }
        let lo = i.saturating_sub(win);
        let hi = (i + win).min(n - 2);
        let is_peak = (lo..=hi).all(|j| j == i || turn(j) < ti || (turn(j) == ti && j > i));
        if is_peak {
            corners.push(i);
        }
    }

    let mut ranges = Vec::new();
    let mut start = 0usize;
    for &c in &corners {
        if c > start {
            ranges.push((start, c + 1));
            start = c;
        }
    }
    ranges.push((start, n));
    ranges.retain(|(a, b)| b - a >= 2);
    if ranges.is_empty() {
        ranges.push((0, n));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdp_keeps_endpoints_and_drops_colinear() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0001),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0001),
            Vec2::new(4.0, 0.0),
        ];
        let out = rdp(&pts, 0.01);
        assert_eq!(out.first().unwrap(), &pts[0]);
        assert_eq!(out.last().unwrap(), &pts[4]);
        assert_eq!(out.len(), 2, "near-colinear run collapses to a segment");
    }

    #[test]
    fn rdp_keeps_a_real_bend() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 0.0),
        ];
        let out = rdp(&pts, 0.01);
        assert_eq!(out.len(), 3, "the apex is well above tolerance");
    }

    #[test]
    fn corner_split_finds_the_right_angle() {
        // An L: horizontal run then vertical run, sharp corner at the elbow.
        let mut pts = Vec::new();
        for i in 0..=10 {
            pts.push(Vec2::new(i as f64, 0.0));
        }
        for i in 1..=10 {
            pts.push(Vec2::new(10.0, i as f64));
        }
        let ranges = segment_corners(&pts, std::f64::consts::FRAC_PI_4);
        assert_eq!(ranges.len(), 2, "one corner -> two segments, got {ranges:?}");
    }

    #[test]
    fn smooth_arc_is_one_segment() {
        let mut pts = Vec::new();
        for i in 0..=20 {
            let t = std::f64::consts::PI * (i as f64 / 20.0);
            pts.push(Vec2::new(t.cos(), t.sin()));
        }
        let ranges = segment_corners(&pts, std::f64::consts::FRAC_PI_4);
        assert_eq!(ranges.len(), 1, "a smooth semicircle has no corners");
    }
}
