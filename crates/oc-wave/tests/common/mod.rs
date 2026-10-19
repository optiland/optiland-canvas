//! Helpers shared by the oc-wave integration tests.
#![allow(dead_code)]

use oc_wave::{Cx, Field2};

/// Largest `|a − b|` over two equal-length sample sets.
pub fn max_diff(a: &[Cx], b: &[Cx]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max)
}

/// The largest sample magnitude.
pub fn peak(a: &[Cx]) -> f64 {
    a.iter().map(|z| z.abs()).fold(0.0, f64::max)
}

/// `1/e²` radius from the second moment along x: for a Gaussian,
/// `⟨(x − x̄)²⟩ = w²/4`.
pub fn second_moment_radius_x(u: &Field2) -> f64 {
    let g = u.grid;
    let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, _) = g.xy(i, j);
            let w = u.at(i, j).norm_sqr();
            s0 += w;
            s1 += w * x;
            s2 += w * x * x;
        }
    }
    let m = s1 / s0;
    2.0 * (s2 / s0 - m * m).sqrt()
}

/// Print a measured number so the docs can record it (run with
/// `-- --nocapture`).
pub fn measured(what: &str, value: f64) {
    eprintln!("measured: {what} = {value:.3e}");
}
