//! The 2D transform: scale, sign and round trip.
//!
//! Sources:
//! - J. W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H.
//!   Freeman/Macmillan Learning (2017), ISBN 978-1-319-11916-4: Parseval's
//!   (Rayleigh's) theorem, and the Gaussian as its own transform,
//!   `F{exp(−πx²)} = exp(−πf²)`, with the shift theorem
//!   `F{g(x − a)} = G(f)·exp(−i2πfa)` under the kernel `exp(−i2πfx)`.
//!   Retrieved from the publisher's listing 2026-10-19; section numbers to be
//!   confirmed against the printed edition.

use oc_wave::{ifftshift2, fftshift2, Cx, Fft2, Grid2};
use std::f64::consts::PI;

/// A deterministic, structureless test field.
fn noise(n: usize, seed: u64) -> Vec<Cx> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    (0..n).map(|_| Cx::new(next(), next())).collect()
}

#[test]
fn fft2_is_unitary_under_parseval() {
    // Non-power-of-two on both axes, unequal pitches: Σ|U|²ΔxΔy = Σ|Û|²Δf_xΔf_y.
    let g = Grid2::new([48, 30], [0.013, 0.021], [0.0, 0.0]).unwrap();
    let u = noise(g.len(), 7);
    let p_space: f64 = u.iter().map(|z| z.norm_sqr()).sum::<f64>() * g.cell_area_mm2();
    let mut s = u.clone();
    let mut fft = Fft2::new();
    fft.forward(&mut s, g.n, g.cell_area_mm2());
    let df = g.freq_pitch(0) * g.freq_pitch(1);
    let p_freq: f64 = s.iter().map(|z| z.norm_sqr()).sum::<f64>() * df;
    assert!(
        ((p_space - p_freq) / p_space).abs() < 1e-13,
        "{p_space} vs {p_freq}"
    );
}

#[test]
fn forward_then_inverse_returns_the_field() {
    let g = Grid2::new([60, 45], [0.01, 0.01], [0.0, 0.0]).unwrap();
    let u = noise(g.len(), 11);
    let mut s = u.clone();
    let mut fft = Fft2::new();
    fft.forward(&mut s, g.n, g.cell_area_mm2());
    fft.inverse(&mut s, g.n, g.freq_pitch(0) * g.freq_pitch(1));
    let err = u
        .iter()
        .zip(&s)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-13, "round trip error {err}");
}

#[test]
fn the_centered_transform_of_a_shifted_gaussian_is_a_gaussian_with_the_shift_phase() {
    // g(x, y) = exp(−π((x − a)² + y²)) on a window far wider than the Gaussian.
    // Goodman: G(f_x, f_y) = exp(−π(f_x² + f_y²))·exp(−i2πf_x a). Both the
    // magnitude and the sign of the phase are pinned here, so a flipped
    // kernel or a lost centring shift cannot pass.
    let n = 128;
    let dx = 0.05;
    let a = 0.3;
    let g = Grid2::square(n, dx).unwrap();
    let mut s: Vec<Cx> = (0..g.len())
        .map(|k| {
            let (x, y) = g.xy(k % n, k / n);
            Cx::real((-PI * ((x - a).powi(2) + y * y)).exp())
        })
        .collect();
    let mut fft = Fft2::new();
    ifftshift2(&mut s, g.n);
    fft.forward(&mut s, g.n, g.cell_area_mm2());
    fftshift2(&mut s, g.n);
    let df = g.freq_pitch(0);
    let mut worst: f64 = 0.0;
    for j in 0..n {
        for i in 0..n {
            let fx = (i as f64 - (n / 2) as f64) * df;
            let fy = (j as f64 - (n / 2) as f64) * df;
            let want = Cx::from_polar((-PI * (fx * fx + fy * fy)).exp(), -2.0 * PI * fx * a);
            worst = worst.max((s[j * n + i] - want).abs());
        }
    }
    assert!(worst < 1e-12, "worst sample error {worst}");
}
