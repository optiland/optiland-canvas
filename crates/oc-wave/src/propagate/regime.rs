//! Which propagator a sampling supports.

use crate::report::Sampling;

/// The critical distance of Voelz & Roggemann, *Appl. Opt.* 48(32),
/// 6132–6142 (2009), doi:10.1364/AO.48.006132: the `z` at which
/// `Δx = λz/L`, i.e. `z_c = LΔx/λ` for a window of width `L`.
///
/// Below it the Fresnel transfer function's chirp is sampled at or above
/// Nyquist and that method is the right one; above it the impulse
/// response's chirp is, and that one is. At `z_c` they agree. The same `z_c`
/// is where the band limit of Matsushima & Shimobaba (2009) first bites: up
/// to it, the angular spectrum's transfer function is sampled finely enough
/// across the whole window and the band limit removes nothing.
pub fn critical_distance_mm(pitch_mm: f64, width_mm: f64, lambda_mm: f64) -> f64 {
    width_mm * pitch_mm / lambda_mm
}

/// Which side of the critical distance a Fresnel step is on.
pub fn voelz_sampling(pitch_mm: f64, width_mm: f64, lambda_mm: f64, z_mm: f64) -> Sampling {
    let zc = critical_distance_mm(pitch_mm, width_mm, lambda_mm);
    let r = z_mm.abs() / zc;
    if (r - 1.0).abs() < 1e-12 {
        Sampling::Critical
    } else if r < 1.0 {
        Sampling::Oversampled
    } else {
        Sampling::Undersampled
    }
}

/// The largest phase error of the Fresnel approximation, radians, for
/// lateral reach `ρ` (the largest `|r − r′|` between a source and an
/// observation sample) at distance `z`: the first neglected term of the
/// binomial expansion of the distance, `π ρ⁴ / (4 λ z³)` (Goodman,
/// *Introduction to Fourier Optics*, 4th ed. (2017), the Fresnel
/// approximation's sufficient condition).
pub fn fresnel_phase_error_rad(reach_mm: f64, z_mm: f64, lambda_mm: f64) -> f64 {
    std::f64::consts::PI * reach_mm.powi(4) / (4.0 * lambda_mm * z_mm.abs().powi(3))
}
