//! Phase-only hologram design.
//!
//! The Gerchberg–Saxton algorithm (R. W. Gerchberg, W. O. Saxton, "A practical
//! algorithm for the determination of phase from image and diffraction plane
//! pictures," *Optik* 35(2), 237–246 (1972)) finds a phase for a known
//! source amplitude whose Fourier transform has a wanted amplitude: transform,
//! keep the phase but impose the target amplitude, transform back, keep the
//! phase but impose the source amplitude, repeat. J. R. Fienup, "Phase
//! retrieval algorithms: a comparison," *Appl. Opt.* 21(15), 2758–2769
//! (1982), shows the error of this two-plane error-reduction cannot increase
//! from one iteration to the next; the test holds the code to that.
//!
//! The two planes are the source plane and its Fraunhofer far field, both on
//! the FFT's own grid, so a pixel of the hologram is a sample.

use crate::fft::{fftshift2, ifftshift2};
use crate::{Cx, Fft2, WaveError};

/// The result of a Gerchberg–Saxton run.
#[derive(Debug, Clone, PartialEq)]
pub struct GsResult {
    /// The phase in the source plane, radians, row-major, `x` fastest.
    pub phase: Vec<f64>,
    /// After each iteration, `Σ(|F| − target)² / Σ target²` in the far field.
    pub error: Vec<f64>,
}

/// Run `iterations` of Gerchberg–Saxton on an `n[0] × n[1]` grid: `source`
/// and `target` are amplitudes, centred, `x` fastest. The target is scaled to
/// the source's power first (Parseval), so the error measures shape. The
/// first phase is `start` (zero if `None`).
pub fn gerchberg_saxton(
    source: &[f64],
    target: &[f64],
    n: [usize; 2],
    iterations: usize,
    start: Option<&[f64]>,
) -> Result<GsResult, WaveError> {
    let len = n[0] * n[1];
    if source.len() != len || target.len() != len {
        return Err(WaveError::Mismatch("source, target and grid differ in size".into()));
    }
    if !source.iter().chain(target).all(|v| v.is_finite() && *v >= 0.0) {
        return Err(WaveError::OutOfRange("amplitudes must be finite and non-negative".into()));
    }
    let ps: f64 = source.iter().map(|a| a * a).sum();
    let pt: f64 = target.iter().map(|a| a * a).sum();
    if ps == 0.0 || pt == 0.0 {
        return Err(WaveError::OutOfRange("source and target need some power".into()));
    }
    // Unitary transforms (scale 1/√N each way), so powers compare directly.
    let unit = 1.0 / (len as f64).sqrt();
    let k = (ps / pt).sqrt();
    let tgt: Vec<f64> = target.iter().map(|a| a * k).collect();
    let mut phase: Vec<f64> = start.map(|s| s.to_vec()).unwrap_or_else(|| vec![0.0; len]);
    if phase.len() != len {
        return Err(WaveError::Mismatch("start phase and grid differ in size".into()));
    }
    let mut fft = Fft2::new();
    let mut error = Vec::with_capacity(iterations);
    let mut buf = vec![Cx::ZERO; len];
    for _ in 0..iterations {
        for ((b, a), p) in buf.iter_mut().zip(source).zip(&phase) {
            *b = Cx::from_polar(*a, *p);
        }
        ifftshift2(&mut buf, n);
        fft.forward(&mut buf, n, unit);
        fftshift2(&mut buf, n);
        let mut e = 0.0;
        for (b, t) in buf.iter_mut().zip(&tgt) {
            e += (b.abs() - t).powi(2);
            *b = Cx::from_polar(*t, b.arg());
        }
        error.push(e / ps);
        ifftshift2(&mut buf, n);
        fft.inverse(&mut buf, n, unit);
        fftshift2(&mut buf, n);
        for (p, b) in phase.iter_mut().zip(&buf) {
            *p = b.arg();
        }
    }
    Ok(GsResult { phase, error })
}
