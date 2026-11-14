//! Statistical numbers and their standard errors.
//!
//! A job traces its rays in independent passes. Each pass is the scene's
//! point sets shifted afresh ([`oc_core::scene3::Source3::emit_pass`]), so the
//! passes' readings are independent draws of the same estimate, and their
//! spread says how far the pooled reading can be from the one infinitely many
//! rays would give: the batch-means standard error, `s / √m` over the `m`
//! passes that saw anything. For a ratio, such as a centroid, that is its
//! first-order (delta-method) error.

use serde::Serialize;

/// A number estimated from rays, and its standard error. `stderr` is `None`
/// when fewer than two passes gave a reading, so there is no spread to say it
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Estimate {
    pub value: f64,
    pub stderr: Option<f64>,
}

impl Estimate {
    /// `value`, with the standard error of the mean of the per-pass readings
    /// `passes`.
    pub fn new(value: f64, passes: &[f64]) -> Self {
        Self {
            value,
            stderr: standard_error(passes),
        }
    }
}

/// The standard error of the mean of `x`: the sample standard deviation over
/// `√n`. `None` for fewer than two values.
pub fn standard_error(x: &[f64]) -> Option<f64> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let ss: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    Some((ss / (n - 1) as f64 / n as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_standard_error_of_a_known_sample() {
        // Sample sd of 1..=4 is sqrt(5/3); over sqrt(4).
        let se = standard_error(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((se - (5.0f64 / 3.0).sqrt() / 2.0).abs() < 1e-15);
        assert_eq!(standard_error(&[7.0, 7.0, 7.0]), Some(0.0));
        assert_eq!(standard_error(&[1.0]), None);
        assert_eq!(standard_error(&[]), None);
    }
}
