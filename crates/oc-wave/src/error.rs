//! Why a wave computation refused to run.

use std::fmt;

/// A refusal. Every one names its reason; none is a silent fallback.
#[derive(Debug, Clone, PartialEq)]
pub enum WaveError {
    /// A grid with no samples, a non-positive pitch, or a non-finite number.
    InvalidGrid(String),
    /// A field, wavelength or medium that is not usable as given.
    InvalidField(String),
    /// More samples than [`crate::MAX_WAVE_SAMPLES`] allows on this target.
    TooLarge { samples: usize, cap: usize },
    /// Two fields that must share a grid and wavelength do not.
    Mismatch(String),
    /// A parameter outside the range a model supports.
    OutOfRange(String),
}

impl fmt::Display for WaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid(s) => write!(f, "invalid grid: {s}"),
            Self::InvalidField(s) => write!(f, "invalid field: {s}"),
            Self::TooLarge { samples, cap } => write!(
                f,
                "{samples} samples exceed the {cap} this target allows; \
                 use a smaller window, a coarser pitch, or run natively"
            ),
            Self::Mismatch(s) => write!(f, "mismatched fields: {s}"),
            Self::OutOfRange(s) => write!(f, "out of range: {s}"),
        }
    }
}

impl std::error::Error for WaveError {}
