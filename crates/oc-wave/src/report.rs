//! What a wave computation did, what it approximated, and where the power went.
//!
//! The ray tracer's `CapabilityReport` and `EnergyLedger` are the model: every
//! method choice is recorded, every approximation is named, and power that
//! leaves the computation is booked to a cause rather than renormalized away.
//!
//! Each step books what it changed to a named cause, so the ledger sums to
//! what was launched by construction: that it does is bookkeeping, not
//! evidence. The evidence is in the entries themselves, in two groups. The
//! **physical** buckets say where the model sends the power (absorbed, into
//! another arm, order or family, past an outline, into evanescent waves);
//! none of them may be negative, and a passive model never detects more than
//! it launched ([`WaveLedger::check`]). The **numerical** buckets say what
//! the computation, not the model, took: a window's crop, a band cut, the
//! sub-cell projection of a sharp edge, interpolation, and the remainder
//! `numerical`, which should be round-off.

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
    /// A field read onto a face where the period of its padded spectrum
    /// (`period_mm`) is shorter than the span the field needs there
    /// (`span_mm`: its window on the face, with its walk to the face): the
    /// light beyond the period folds back into it. The face beyond the
    /// period reads zero; the field's periodic copies are never placed there.
    Periodic { period_mm: f64, span_mm: f64 },
}

/// A model limit that applies to the result as a whole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidityFlag {
    /// A heuristic, not a bound: a device's active feature (a mirror's side,
    /// an electrode's width) under 14 wavelengths. Pommet, Moharam & Grann
    /// (JOSA A 11, 1827 (1994)) found scalar theory's efficiencies off by
    /// more than ±5% against rigorous coupled-wave analysis for binary
    /// features below about 14λ, the error least near 50% fill (where the
    /// period sets it instead), for their 8-level element of index 1.5 at
    /// normal incidence below a period of 20λ, and growing with index, depth,
    /// level count and incidence (Q20 of the expert review). No pitch above
    /// the mark certifies a device; the flag records what sets the error.
    ScalarLimit {
        /// The lattice pitch, µm.
        #[serde(default)]
        pitch_um: f64,
        /// The active feature the mark is taken on, µm.
        feature_um: f64,
        /// The dead space between features, µm.
        #[serde(default)]
        gap_um: f64,
        /// The relief across a feature (a mirror's tilt over its side), µm,
        /// where the model knows it.
        #[serde(default)]
        depth_um: Option<f64>,
        /// The refractive-index contrast of the relief, where it is a
        /// dielectric the model knows (none for a mirror).
        #[serde(default)]
        index_contrast: Option<f64>,
        /// The light's incidence on the device, degrees.
        #[serde(default)]
        incidence_deg: f64,
        lambda_um: f64,
    },
    /// An LCoS phase scaled as `λ₀/λ`, ignoring the dispersion of its
    /// birefringence.
    PhaseDispersionApprox,
    /// A thin-element approximation, with what it leaves out.
    ThinElement { detail: String },
}

/// Where the power went, as a fraction of nothing in particular: the same
/// units as [`crate::Field2::power`].
///
/// The physical buckets ([`WaveLedger::physical`]) are `detected`,
/// `absorbed_aperture`, `absorbed_element`, `evanescent`, `other_family`,
/// `untracked_orders`, `other_arm`, `passed_by` and `other_ports`; the
/// numerical ones ([`WaveLedger::numerical_total`]) are `left_window`,
/// `band_limit_removed`, `spectral_projection`, `subcell_projection`,
/// `resampling` and `numerical`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct WaveLedger {
    pub launched: f64,
    /// Still in the field at the end.
    pub detected: f64,
    pub absorbed_aperture: f64,
    pub absorbed_element: f64,
    /// Removed by a propagator's band limit (the band-limited angular
    /// spectrum's).
    pub band_limit_removed: f64,
    /// Taken out by a band cut: the plane waves a coarser window, or a
    /// modulator's outgoing window, cannot hold, removed before the field is
    /// read onto it so that they do not fold back in at directions they do
    /// not have. A projection of the field's spectrum, not a loss of the
    /// model's.
    #[serde(default)]
    pub spectral_projection: f64,
    /// The sub-cell projection error of a sharp edge weighted by exact cell
    /// coverage. An edge passing a fraction `f` of a cell transmits
    /// `f·|U|²ΔA` physically, but the sample, `f·U`, carries only `f²·|U|²ΔA`;
    /// the difference, `(f − f²)|U|²ΔA`, is here, not in `absorbed_aperture`.
    /// It is the norm the cell-average projection discards, which is not the
    /// same as the norm beyond the grid's band: the two are separate claims,
    /// and this bucket makes only the first.
    #[serde(default, alias = "beyond_nyquist")]
    pub subcell_projection: f64,
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
    /// are booked, so it is named, never a check; [`WaveLedger::check`]
    /// reports it when it comes out negative.
    #[serde(default)]
    pub other_ports: f64,
    /// Round-off: whatever is left after every cause above.
    pub numerical: f64,
}

/// The relative tolerance [`WaveLedger::check`] allows a physical bucket
/// below zero, or `detected` above `launched`: the round-off of a long
/// chain and of the readout's band-limited interpolation onto the
/// detector's grid, whose sum can exceed the launched power by a few 1e-9;
/// never a model's error.
pub const LEDGER_TOLERANCE: f64 = 1e-7;

impl WaveLedger {
    /// The physical buckets together: where the model sends the power.
    pub fn physical(&self) -> f64 {
        self.detected
            + self.absorbed_aperture
            + self.absorbed_element
            + self.evanescent
            + self.other_family
            + self.untracked_orders
            + self.other_arm
            + self.passed_by
            + self.other_ports
    }

    /// The numerical buckets together: what the computation took.
    pub fn numerical_total(&self) -> f64 {
        self.left_window
            + self.band_limit_removed
            + self.spectral_projection
            + self.subcell_projection
            + self.resampling
            + self.numerical
    }

    /// Everything booked, `numerical` included.
    pub fn accounted(&self) -> f64 {
        self.physical() + self.numerical_total()
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

    /// What a passive model must never book, each named: a physical bucket
    /// below zero, or more detected than launched, by more than
    /// [`LEDGER_TOLERANCE`] of what was launched. A remainder bucket
    /// (`other_ports`) that comes out negative means an arm booked a loss
    /// another arm carried, or a field gained power on the way; neither is
    /// round-off.
    pub fn check(&self) -> Vec<String> {
        let scale = self.launched.abs().max(f64::MIN_POSITIVE);
        let tol = LEDGER_TOLERANCE * scale;
        let mut out = Vec::new();
        let buckets = [
            ("detected", self.detected),
            ("absorbed_aperture", self.absorbed_aperture),
            ("absorbed_element", self.absorbed_element),
            ("evanescent", self.evanescent),
            ("other_family", self.other_family),
            ("untracked_orders", self.untracked_orders),
            ("other_arm", self.other_arm),
            ("passed_by", self.passed_by),
            ("other_ports", self.other_ports),
        ];
        for (name, v) in buckets {
            if v < -tol {
                out.push(format!("the ledger's {name} is negative: {:.3e} of the launched power", v / scale));
            }
        }
        if self.detected > self.launched + tol {
            out.push(format!("more power detected than launched: {:.3e} of it", (self.detected - self.launched) / scale));
        }
        out
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
    /// What the ledger booked that a passive model cannot
    /// ([`WaveLedger::check`]), each named. Empty for a sound result.
    #[serde(default)]
    pub inconsistencies: Vec<String>,
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

    /// Run [`WaveLedger::check`] on the ledger and record what it finds.
    pub fn check_ledger(&mut self) {
        for s in self.ledger.check() {
            if !self.inconsistencies.contains(&s) {
                self.inconsistencies.push(s);
            }
        }
    }

    /// Close the ledger ([`WaveLedger::close`]) and check it.
    pub fn close_checked(&mut self) -> f64 {
        let rel = self.ledger.close();
        self.check_ledger();
        rel
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
        assert!((l.physical() + l.numerical_total() - l.accounted()).abs() < 1e-15);
        assert!(l.check().is_empty(), "{:?}", l.check());
    }

    #[test]
    fn the_check_names_a_negative_bucket_and_a_gain() {
        let l = WaveLedger { launched: 1.0, detected: 1.5, other_ports: -0.5, ..Default::default() };
        let found = l.check();
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].contains("other_ports") && found[0].contains("negative"), "{found:?}");
        assert!(found[1].contains("more power detected than launched"), "{found:?}");
        // Round-off below the tolerance is not an inconsistency.
        let l = WaveLedger { launched: 1.0, detected: 1.0 + 1e-13, other_arm: -1e-13, ..Default::default() };
        assert!(l.check().is_empty());
        let mut r = WaveReport { ledger: WaveLedger { launched: 1.0, detected: 0.4, absorbed_element: -0.1, ..Default::default() }, ..Default::default() };
        r.close_checked();
        assert_eq!(r.inconsistencies.len(), 1, "{:?}", r.inconsistencies);
        assert!((r.ledger.numerical - 0.7).abs() < 1e-15);
    }

    #[test]
    fn a_report_round_trips_through_json() {
        let mut r = WaveReport::default();
        r.warn(SamplingWarning::BandLimited { removed: 0.01 });
        r.flag(ValidityFlag::ScalarLimit {
            pitch_um: 8.0,
            feature_um: 8.0,
            gap_um: 0.0,
            depth_um: None,
            index_contrast: None,
            incidence_deg: 0.0,
            lambda_um: 1.55,
        });
        r.approx("thin element");
        r.approx("thin element");
        assert_eq!(r.approximated.len(), 1);
        let s = serde_json::to_string(&r).unwrap();
        let back: WaveReport = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn an_old_ledger_names_its_subcell_term_beyond_nyquist() {
        let old = r#"{"launched":1.0,"detected":0.9,"absorbed_aperture":0.0,"absorbed_element":0.0,"band_limit_removed":0.0,"beyond_nyquist":0.1,"evanescent":0.0,"left_window":0.0,"other_family":0.0,"untracked_orders":0.0,"numerical":0.0}"#;
        let l: WaveLedger = serde_json::from_str(old).unwrap();
        assert_eq!(l.subcell_projection, 0.1);
        assert_eq!(l.spectral_projection, 0.0);
        assert!(serde_json::to_string(&l).unwrap().contains("subcell_projection"));
    }
}
