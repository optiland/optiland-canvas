//! What a wave computation did, what it approximated, and where the power went.
//!
//! The ray tracer's `CapabilityReport` and `EnergyLedger` are the model: every
//! method choice is recorded, every approximation is named, and power that
//! leaves the computation is booked to a cause rather than renormalized away.
//!
//! Each step books what it changed to a named cause, so the ledger sums to
//! what was launched by construction: that it does is bookkeeping, not
//! evidence. The evidence is in the entries themselves, above all
//! `resampling` (what reading a field between its samples cost) and
//! `numerical` (what no step accounted for, which should be round-off).

use serde::{Deserialize, Serialize};

/// The propagator that ran for one step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Method {
    /// Angular spectrum, exact scalar Rayleigh–Sommerfeld for a homogeneous
    /// medium; `band_limited` per Matsushima & Shimobaba, Opt. Express 17,
    /// 19662 (2009).
    AngularSpectrum { band_limited: bool },
    /// Fresnel transfer function (Voelz & Roggemann, Appl. Opt. 48, 6132 (2009)).
    FresnelTf,
    /// Fresnel impulse response (same source).
    FresnelIr,
    /// Single-transform Fraunhofer.
    Fraunhofer,
    /// Direct Rayleigh–Sommerfeld convolution (Shen & Wang, Appl. Opt. 45,
    /// 1102 (2006)). A slow reference.
    RsDirect,
    /// A thin element applied in one plane, no propagation.
    Element,
}

/// How the chirp of a Fresnel step was sampled, after Voelz & Roggemann.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sampling {
    /// `Δx = λz/L`: transfer function and impulse response agree.
    Critical,
    /// `Δx > λz/L`: the transfer function is the right choice.
    Oversampled,
    /// `Δx < λz/L`: the impulse response is the right choice.
    Undersampled,
}

/// Something about a step that a reader of its numbers should know.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplingWarning {
    /// The band limit removed this fraction of the step's input power.
    BandLimited { removed: f64 },
    /// This fraction of the output power sits in the outer sixteenth of the
    /// window, where a periodic transform wraps it around.
    WrapAround { edge_fraction: f64 },
    /// A method was forced outside the regime its source supports.
    OutsideRegime { method: Method, reason: String },
    /// The relative field error of a paraxial (Fresnel) step, exact for the
    /// field it ran on: `‖(U − U₂)g‖/‖g‖` (`crate::propagate::paraxial_error`).
    Paraxial { field_error: f64 },
    /// The step ran on a coarse preview grid; its numbers are not for display.
    CoarsePreview { n: [usize; 2] },
    /// A tilted element was evaluated on its projection; the depth across the
    /// beam is reported because the approximation degrades with it.
    TiltProjection { depth_mm: f64 },
    /// A phase that changes by more than π between neighbouring samples
    /// somewhere in the window: what it describes is aliased there.
    Undersampled { what: String, max_phase_step_rad: f64 },
}

/// A model limit that applies to the result as a whole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidityFlag {
    /// A feature smaller than 14 wavelengths: scalar theory's error exceeds
    /// ±5% there (Pommet, Moharam & Grann, JOSA A 11, 1827 (1994)).
    ScalarLimit { feature_um: f64, lambda_um: f64 },
    /// An LCoS phase scaled as `λ₀/λ`, ignoring the dispersion of its
    /// birefringence.
    PhaseDispersionApprox,
    /// A thin-element approximation, with what it leaves out.
    ThinElement { detail: String },
}

/// Where the power went, as a fraction of nothing in particular: the same
/// units as [`crate::Field2::power`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct WaveLedger {
    pub launched: f64,
    /// Still in the field at the end.
    pub detected: f64,
    pub absorbed_aperture: f64,
    pub absorbed_element: f64,
    /// Removed by a band limit.
    pub band_limit_removed: f64,
    /// Diffracted by a sharp edge into angles beyond the grid's Nyquist
    /// band: power the sampled field cannot carry. An aperture edge passing
    /// a fraction `f` of a cell transmits `f·|U|²ΔA` physically, but the
    /// sample, `f·U`, carries only `f²·|U|²ΔA`; the difference is here, not
    /// in `absorbed_aperture`.
    #[serde(default)]
    pub beyond_nyquist: f64,
    /// Carried by evanescent components that decayed.
    pub evanescent: f64,
    /// Left the sampled window.
    pub left_window: f64,
    /// Sent into a state family (a DMD's OFF mirrors, an LCoS's unmodulated
    /// polarization) that this computation did not follow.
    pub other_family: f64,
    /// Diffraction orders not followed.
    pub untracked_orders: f64,
    /// Sent by a splitter into its other arm, or reflected by a refracting
    /// face, on a path this computation does not follow. Not absorbed.
    #[serde(default)]
    pub other_arm: f64,
    /// Passed beside an element's outline, along a direction this
    /// computation does not follow (beside a fold mirror, a grating).
    #[serde(default)]
    pub passed_by: f64,
    /// Gained (negative) or lost carrying a field onto another grid or plane:
    /// interpolation between samples, a traced map's fit, a window's edge.
    #[serde(default)]
    pub resampling: f64,
    /// Left by ports a computation of several arms does not read: a
    /// recombining splitter's other output, and what misses the detector
    /// once the arms meet. It is the remainder after every arm's own losses
    /// are booked, so it is named, never a check.
    #[serde(default)]
    pub other_ports: f64,
    /// Round-off: whatever is left after every cause above.
    pub numerical: f64,
}

impl WaveLedger {
    /// Everything booked, `numerical` included.
    pub fn accounted(&self) -> f64 {
        self.detected
            + self.absorbed_aperture
            + self.absorbed_element
            + self.band_limit_removed
            + self.beyond_nyquist
            + self.evanescent
            + self.left_window
            + self.other_family
            + self.untracked_orders
            + self.other_arm
            + self.passed_by
            + self.resampling
            + self.other_ports
            + self.numerical
    }

    /// Book the unexplained remainder to `numerical` and return its size
    /// relative to what was launched.
    pub fn close(&mut self) -> f64 {
        self.numerical = 0.0;
        let gap = self.launched - self.accounted();
        self.numerical = gap;
        if self.launched > 0.0 {
            gap.abs() / self.launched
        } else {
            gap.abs()
        }
    }
}

/// One step of a computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepReport {
    pub method: Method,
    /// Propagation distance, mm (0 for an element).
    pub z_mm: f64,
    pub sampling: Option<Sampling>,
    /// What ran, in words, for a person reading the report.
    pub note: String,
}

/// The report a wave computation returns beside its field.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WaveReport {
    pub steps: Vec<StepReport>,
    pub warnings: Vec<SamplingWarning>,
    pub validity: Vec<ValidityFlag>,
    /// Approximations, named, in the style of `CapabilityReport::approximated`.
    pub approximated: Vec<String>,
    /// What was refused, and why.
    pub rejected: Vec<String>,
    pub ledger: WaveLedger,
}

impl WaveReport {
    pub fn warn(&mut self, w: SamplingWarning) {
        self.warnings.push(w);
    }

    pub fn approx(&mut self, what: impl Into<String>) {
        let s = what.into();
        if !self.approximated.contains(&s) {
            self.approximated.push(s);
        }
    }

    pub fn flag(&mut self, v: ValidityFlag) {
        if !self.validity.contains(&v) {
            self.validity.push(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ledger_closes_by_booking_the_gap_to_numerical() {
        let mut l = WaveLedger {
            launched: 1.0,
            detected: 0.7,
            absorbed_aperture: 0.2,
            band_limit_removed: 0.1 - 1e-15,
            ..Default::default()
        };
        let rel = l.close();
        assert!(rel < 1e-14);
        assert!((l.accounted() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn a_report_round_trips_through_json() {
        let mut r = WaveReport::default();
        r.warn(SamplingWarning::BandLimited { removed: 0.01 });
        r.flag(ValidityFlag::ScalarLimit {
            feature_um: 8.0,
            lambda_um: 1.55,
        });
        r.approx("thin element");
        r.approx("thin element");
        assert_eq!(r.approximated.len(), 1);
        let s = serde_json::to_string(&r).unwrap();
        let back: WaveReport = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
    }
}
