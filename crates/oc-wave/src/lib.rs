//! Scalar wave optics for Optiland Canvas.
//!
//! The rest of the kernel is a geometric ray tracer: a ray is a packet of flux,
//! never a field amplitude (docs/physics/conventions.md). This crate is the
//! other half. A [`Field2`] is a complex scalar field sampled on a plane, and
//! the propagators move it from one plane to another by diffraction theory.
//! It lives beside the tracer, not inside it, and the two never mix units: a
//! field does not feed ray flux, and a ray does not carry a phase it did not
//! earn from its optical path length (ADR 0004).
//!
//! # Conventions
//!
//! These follow docs/physics/conventions.md, which fixes the time dependence
//! as `exp(-iωt)`, and Goodman's *Introduction to Fourier Optics*, 4th ed.
//! (W. H. Freeman/Macmillan Learning, 2017, ISBN 978-1-319-11916-4), which uses
//! the same one.
//!
//! - A plane wave travelling toward `+z` is `exp(+ikz)`.
//! - The forward transform kernel is `exp(-i2π(f_x x + f_y y))`, scaled by the
//!   sample area `ΔxΔy` so that it approximates the continuous Fourier
//!   integral; the inverse is scaled by the frequency cell `Δf_x Δf_y`. With
//!   those two scales Parseval's theorem holds exactly between the sampled
//!   field and its sampled spectrum, so power is comparable on both sides.
//! - Lengths are millimetres, wavelengths micrometres, as everywhere in the
//!   kernel. A field converts between them through [`Field2::wavelength_mm`];
//!   the beams and the path do the same `λ_um / 1000` where no field is at
//!   hand yet.
//! - `|U|²` is flux per mm², so `Σ|U|²ΔA` is the flux the field carries. It is
//!   relative unless the source that launched it was calibrated, exactly like
//!   a ray's flux.

#![forbid(unsafe_code)]

pub mod beams;
pub mod czt;
pub mod design;
pub mod elements;
pub mod error;
pub mod fft;
pub mod field;
pub mod grid;
pub mod mie;
pub mod modulator;
#[cfg(feature = "scene")]
pub mod path;
pub mod plan;
#[cfg(feature = "scene")]
pub mod planar;
#[cfg(feature = "scene")]
mod transfer;
pub mod propagate;
pub mod report;
pub mod special;

pub use error::WaveError;
pub use fft::{fftshift2, ifftshift2, next_fast_len, Fft2};
pub use field::Field2;
pub use grid::Grid2;
pub use oc_core::complex::Cx;
pub use propagate::{asm_tilted, asm_zoom, propagate, AsmOptions, Propagator};
pub use report::{Method, SamplingWarning, ValidityFlag, WaveLedger, WaveReport};

/// The most samples a field may hold.
///
/// A sample is two `f64`, and a propagation keeps the field, its padded copy
/// and a scratch buffer of the same size. In the browser a 2²² field is about
/// 200 MB of working memory, which is where a tab starts to fail; natively the
/// cap is 2²⁶. A larger request is refused with a named reason, never quietly
/// truncated.
#[cfg(target_arch = "wasm32")]
pub const MAX_WAVE_SAMPLES: usize = 1 << 22;
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_WAVE_SAMPLES: usize = 1 << 26;

/// The widest padded grid a step pads a field to, samples a side: 4096, or
/// the widest power of two whose square [`MAX_WAVE_SAMPLES`] holds (2048 in
/// the browser). The readout and a re-gridded step cap their padding with it,
/// so that a target that holds the field is not refused its padded copy.
#[cfg(feature = "scene")]
pub(crate) fn max_padded_side() -> usize {
    padded_side_within(MAX_WAVE_SAMPLES)
}

/// The widest power of two, at most 4096, whose square is within `cap`.
#[cfg(feature = "scene")]
pub(crate) fn padded_side_within(cap: usize) -> usize {
    let mut side = 4096usize;
    while side > 1 && side * side > cap {
        side /= 2;
    }
    side
}

#[cfg(all(test, feature = "scene"))]
mod tests {
    #[test]
    fn a_padded_grid_fits_the_target_that_holds_its_field() {
        // The browser holds 2²² samples: a 2048² padded grid, not the
        // 4096² that refused its readout; natively 4096², the most a step
        // pads to.
        assert_eq!(super::padded_side_within(1 << 22), 2048);
        assert_eq!(super::padded_side_within(1 << 26), 4096);
        assert_eq!(super::padded_side_within((1 << 22) - 1), 1024);
        assert_eq!(super::padded_side_within(1 << 24), 4096);
    }
}
