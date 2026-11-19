//! The 2D discrete Fourier transform, scaled to approximate the continuous one.
//!
//! `rustfft` does the 1D transforms. It works on `num_complex::Complex<f64>`,
//! and the kernel's complex type is [`Cx`]; the workspace allows no `unsafe`,
//! so the two are bridged by one linear copy in and one out, into a buffer this
//! struct keeps between calls. ADR 0004 records the choice and its measured
//! cost.
//!
//! Sign and scale follow the crate's conventions (see the crate docs): the
//! forward kernel is `exp(−i2π k n / N)`, and the caller passes the scale that
//! makes the sum an integral, `ΔxΔy` forward and `Δf_x Δf_y` inverse.

use crate::Cx;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Which way to transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Kernel `exp(−i2πfx)`.
    Forward,
    /// Kernel `exp(+i2πfx)`.
    Inverse,
}

/// A reusable 2D transform: plans are cached by length and direction, and the
/// working buffers are kept, so a propagation loop allocates once.
pub struct Fft2 {
    planner: FftPlanner<f64>,
    a: Vec<Complex64>,
    b: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl Default for Fft2 {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Fft2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft2").finish_non_exhaustive()
    }
}

impl Fft2 {
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            a: Vec::new(),
            b: Vec::new(),
            scratch: Vec::new(),
        }
    }

    fn plan(&mut self, len: usize, dir: Direction) -> Arc<dyn Fft<f64>> {
        match dir {
            Direction::Forward => self.planner.plan_fft_forward(len),
            Direction::Inverse => self.planner.plan_fft_inverse(len),
        }
    }

    /// Transform `data` (row-major, `x` fastest, `n[0] × n[1]`) in place and
    /// multiply by `scale`.
    ///
    /// # Panics
    /// If `data.len() != n[0]·n[1]`. That is a programming error inside the
    /// crate, not a user input.
    pub fn transform(&mut self, data: &mut [Cx], n: [usize; 2], dir: Direction, scale: f64) {
        let [nx, ny] = n;
        assert_eq!(data.len(), nx * ny, "buffer does not match its grid");
        if data.is_empty() {
            return;
        }
        self.a.clear();
        self.a
            .extend(data.iter().map(|z| Complex64::new(z.re, z.im)));

        // Rows: `ny` transforms of length `nx`, contiguous.
        let row = self.plan(nx, dir);
        run(&row, &mut self.a, &mut self.scratch);

        // Columns: transpose so they are contiguous, transform, transpose back.
        if ny > 1 {
            transpose(&self.a, &mut self.b, nx, ny);
            let col = self.plan(ny, dir);
            run(&col, &mut self.b, &mut self.scratch);
            transpose(&self.b, &mut self.a, ny, nx);
        }

        for (z, c) in data.iter_mut().zip(self.a.iter()) {
            *z = Cx::new(c.re * scale, c.im * scale);
        }
    }

    /// Forward transform scaled by `scale` (normally `ΔxΔy`).
    pub fn forward(&mut self, data: &mut [Cx], n: [usize; 2], scale: f64) {
        self.transform(data, n, Direction::Forward, scale);
    }

    /// Inverse transform scaled by `scale` (normally `Δf_x Δf_y`).
    pub fn inverse(&mut self, data: &mut [Cx], n: [usize; 2], scale: f64) {
        self.transform(data, n, Direction::Inverse, scale);
    }
}

/// Run `fft` over every contiguous block of `buf`.
fn run(fft: &Arc<dyn Fft<f64>>, buf: &mut [Complex64], scratch: &mut Vec<Complex64>) {
    let need = fft.get_inplace_scratch_len();
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let len = fft.len();
        let rows = buf.len() / len;
        if rows >= 8 {
            // One scratch per task; a task takes a band of rows.
            let band = rows.div_ceil(rayon::current_num_threads().max(1)).max(1) * len;
            buf.par_chunks_mut(band).for_each(|chunk| {
                let mut s = vec![Complex64::new(0.0, 0.0); need];
                fft.process_with_scratch(chunk, &mut s);
            });
            return;
        }
    }
    if scratch.len() < need {
        scratch.resize(need, Complex64::new(0.0, 0.0));
    }
    fft.process_with_scratch(buf, &mut scratch[..need]);
}

/// `dst[i·rows + j] = src[j·cols + i]`: `src` is `rows × cols` row-major.
fn transpose(src: &[Complex64], dst: &mut Vec<Complex64>, cols: usize, rows: usize) {
    dst.clear();
    dst.resize(src.len(), Complex64::new(0.0, 0.0));
    // Blocked, so both sides walk memory in cache-sized pieces.
    const B: usize = 32;
    let mut j0 = 0;
    while j0 < rows {
        let j1 = (j0 + B).min(rows);
        let mut i0 = 0;
        while i0 < cols {
            let i1 = (i0 + B).min(cols);
            for j in j0..j1 {
                for i in i0..i1 {
                    dst[i * rows + j] = src[j * cols + i];
                }
            }
            i0 = i1;
        }
        j0 = j1;
    }
}

/// Move zero frequency from index 0 to index `⌊n/2⌋` on both axes.
pub fn fftshift2(data: &mut [Cx], n: [usize; 2]) {
    roll2(data, n, [n[0] / 2, n[1] / 2]);
}

/// The inverse of [`fftshift2`]: move index `⌊n/2⌋` to index 0. For a field on
/// a [`crate::Grid2`] this puts the window centre at the transform's origin.
pub fn ifftshift2(data: &mut [Cx], n: [usize; 2]) {
    roll2(data, n, [n[0] - n[0] / 2, n[1] - n[1] / 2]);
}

/// Cyclic shift by `s` along each axis: `out[(i + s) mod n] = in[i]`.
fn roll2(data: &mut [Cx], n: [usize; 2], s: [usize; 2]) {
    let [nx, ny] = n;
    assert_eq!(data.len(), nx * ny);
    if data.is_empty() {
        return;
    }
    let src = data.to_vec();
    for j in 0..ny {
        let jd = (j + s[1]) % ny;
        for i in 0..nx {
            let id = (i + s[0]) % nx;
            data[jd * nx + id] = src[j * nx + i];
        }
    }
}

/// The smallest length `≥ n` whose only prime factors are 2, 3 and 5: the
/// sizes a mixed-radix FFT handles at full speed. Padding to one of these
/// costs a few samples and saves the slow prime-length path.
pub fn next_fast_len(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut m = n;
    loop {
        let mut r = m;
        for p in [2, 3, 5] {
            while r % p == 0 {
                r /= p;
            }
        }
        if r == 1 {
            return m;
        }
        m += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_fast_len_has_only_small_prime_factors() {
        assert_eq!(next_fast_len(1), 1);
        assert_eq!(next_fast_len(7), 8);
        assert_eq!(next_fast_len(97), 100);
        assert_eq!(next_fast_len(1000), 1000);
        assert_eq!(next_fast_len(1025), 1080);
    }

    #[test]
    fn the_shifts_undo_each_other_on_odd_and_even_sizes() {
        for n in [[6, 5], [7, 4], [1, 9]] {
            let orig: Vec<Cx> = (0..n[0] * n[1]).map(|k| Cx::new(k as f64, -(k as f64))).collect();
            let mut d = orig.clone();
            fftshift2(&mut d, n);
            ifftshift2(&mut d, n);
            assert_eq!(d, orig);
        }
        // The window centre (index ⌊n/2⌋) lands on index 0.
        let mut d: Vec<Cx> = (0..5).map(|k| Cx::real(k as f64)).collect();
        ifftshift2(&mut d, [5, 1]);
        assert_eq!(d[0], Cx::real(2.0));
    }
}
