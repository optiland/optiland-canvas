//! A regular sampling grid on a plane.

use crate::{WaveError, MAX_WAVE_SAMPLES};
use serde::{Deserialize, Serialize};

/// `n[0] × n[1]` samples at `pitch_mm`, centred on `center_mm` in the plane's
/// local `XY`.
///
/// Sample `i` along an axis sits at `center + (i − ⌊n/2⌋)·pitch`, so the
/// centre is a sample and the window runs from `−⌊n/2⌋` to `n − 1 − ⌊n/2⌋`
/// pitches about it. That is the layout an FFT expects after `ifftshift`, and
/// it puts zero spatial frequency at index `⌊n/2⌋` after `fftshift`.
///
/// Storage is row-major with `x` fastest: sample `(i, j)` is at `j·n[0] + i`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Grid2 {
    pub n: [usize; 2],
    pub pitch_mm: [f64; 2],
    pub center_mm: [f64; 2],
}

impl Grid2 {
    /// A validated grid. Refuses an empty axis, a non-positive or non-finite
    /// pitch, a non-finite centre, and more samples than this target allows.
    pub fn new(n: [usize; 2], pitch_mm: [f64; 2], center_mm: [f64; 2]) -> Result<Self, WaveError> {
        let g = Self {
            n,
            pitch_mm,
            center_mm,
        };
        g.validate()?;
        Ok(g)
    }

    /// A square grid centred on the origin.
    pub fn square(n: usize, pitch_mm: f64) -> Result<Self, WaveError> {
        Self::new([n, n], [pitch_mm, pitch_mm], [0.0, 0.0])
    }

    pub fn validate(&self) -> Result<(), WaveError> {
        if self.n[0] == 0 || self.n[1] == 0 {
            return Err(WaveError::InvalidGrid("an axis has no samples".into()));
        }
        if !self.pitch_mm.iter().all(|p| p.is_finite() && *p > 0.0) {
            return Err(WaveError::InvalidGrid(
                "the pitch must be finite and positive".into(),
            ));
        }
        if !self.center_mm.iter().all(|c| c.is_finite()) {
            return Err(WaveError::InvalidGrid("the centre is not finite".into()));
        }
        let samples = self.n[0]
            .checked_mul(self.n[1])
            .ok_or(WaveError::TooLarge {
                samples: usize::MAX,
                cap: MAX_WAVE_SAMPLES,
            })?;
        if samples > MAX_WAVE_SAMPLES {
            return Err(WaveError::TooLarge {
                samples,
                cap: MAX_WAVE_SAMPLES,
            });
        }
        Ok(())
    }

    /// Total number of samples.
    pub fn len(&self) -> usize {
        self.n[0] * self.n[1]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index of sample `(i, j)` in row-major storage.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.n[0] + i
    }

    /// Local coordinate, mm, of sample `i` along `axis`.
    #[inline]
    pub fn coord(&self, axis: usize, i: usize) -> f64 {
        self.center_mm[axis] + (i as f64 - (self.n[axis] / 2) as f64) * self.pitch_mm[axis]
    }

    /// Local `(x, y)` of sample `(i, j)`, mm.
    #[inline]
    pub fn xy(&self, i: usize, j: usize) -> (f64, f64) {
        (self.coord(0, i), self.coord(1, j))
    }

    /// The window's extent along `axis`: `n · pitch`, mm.
    pub fn width_mm(&self, axis: usize) -> f64 {
        self.n[axis] as f64 * self.pitch_mm[axis]
    }

    /// Area of one sample cell, mm².
    pub fn cell_area_mm2(&self) -> f64 {
        self.pitch_mm[0] * self.pitch_mm[1]
    }

    /// Spacing of the discrete spatial frequencies along `axis`: `1/(n·Δx)`,
    /// cycles per mm.
    pub fn freq_pitch(&self, axis: usize) -> f64 {
        1.0 / self.width_mm(axis)
    }

    /// Spatial frequency, cycles per mm, of FFT bin `k` along `axis`, in the
    /// order the transform returns them: `0, 1, …, ⌈n/2⌉−1, −⌊n/2⌋, …, −1`.
    #[inline]
    pub fn fft_freq(&self, axis: usize, k: usize) -> f64 {
        let n = self.n[axis];
        let signed = if k < n.div_ceil(2) {
            k as f64
        } else {
            k as f64 - n as f64
        };
        signed * self.freq_pitch(axis)
    }

    /// The same window with `factor` times the samples on each axis at the
    /// same pitch: the zero-padded grid a linear convolution needs.
    pub fn padded(&self, factor: usize) -> Result<Self, WaveError> {
        Self::new(
            [self.n[0] * factor, self.n[1] * factor],
            self.pitch_mm,
            self.center_mm,
        )
    }

    /// Is the window wholly inside `[−a, a]` about the centre? Used by tests
    /// and by the guard-band check.
    pub fn half_width_mm(&self, axis: usize) -> f64 {
        0.5 * self.width_mm(axis)
    }
}

#[cfg(test)]
mod tests {
    use super::Grid2;

    #[test]
    fn the_centre_is_a_sample_and_the_fft_frequencies_wrap() {
        let g = Grid2::new([8, 5], [0.5, 2.0], [1.0, -3.0]).unwrap();
        assert_eq!(g.coord(0, 4), 1.0);
        assert_eq!(g.coord(1, 2), -3.0);
        assert_eq!(g.coord(0, 0), 1.0 - 4.0 * 0.5);
        // 8 bins: 0..3 then −4..−1, spacing 1/(8·0.5) = 0.25.
        let f: Vec<f64> = (0..8).map(|k| g.fft_freq(0, k)).collect();
        assert_eq!(f, vec![0.0, 0.25, 0.5, 0.75, -1.0, -0.75, -0.5, -0.25]);
        // 5 bins: 0, 1, 2, −2, −1.
        let f: Vec<f64> = (0..5).map(|k| g.fft_freq(1, k) * g.width_mm(1)).collect();
        assert_eq!(f, vec![0.0, 1.0, 2.0, -2.0, -1.0]);
    }

    #[test]
    fn bad_grids_are_refused_by_name() {
        assert!(Grid2::new([0, 4], [1.0, 1.0], [0.0, 0.0]).is_err());
        assert!(Grid2::new([4, 4], [0.0, 1.0], [0.0, 0.0]).is_err());
        assert!(Grid2::new([4, 4], [1.0, f64::NAN], [0.0, 0.0]).is_err());
        assert!(Grid2::new([4, 4], [1.0, 1.0], [f64::INFINITY, 0.0]).is_err());
        assert!(matches!(
            Grid2::new([1 << 20, 1 << 20], [1.0, 1.0], [0.0, 0.0]),
            Err(crate::WaveError::TooLarge { .. })
        ));
    }
}
