//! Pixelated spatial light modulators: the digital micromirror device (DMD)
//! and the phase-only liquid-crystal-on-silicon (LCoS) SLM.
//!
//! This module describes a device and the pattern it displays, and computes
//! the diffraction orders of a periodic pattern **in closed form**, with no
//! FFT: what the ray tracer will need (W7), and what `oc-wave`'s sampled fields
//! are checked against. Sampled near fields and whole-device far fields live
//! in `oc-wave::modulator`.
//!
//! # The DMD
//!
//! An array of square aluminium mirrors on a square pitch, each tilting to one
//! of two angles, ON or OFF, about its diagonal (most Texas Instruments
//! devices) or about one side (the 5.4 µm pixels): Texas Instruments,
//! *DMD Optical Efficiency for Visible Wavelengths*, application note
//! DLPA083B (2019, rev. 2023), Table 2-1 and §4.2. Lit by a plane wave, the
//! array is a blazed two-dimensional grating: the orders sit where the
//! lattice puts them, `(α, β) = (α_in, β_in) + λ(m, n)/p` in direction cosines
//! (J. E. Harvey, C. L. Vernold, *Appl. Opt.* 37(34), 8158–8160 (1998),
//! doi:10.1364/AO.37.008158), and their strengths follow one mirror's
//! diffraction envelope, centred on the direction the tilted mirror reflects
//! specularly (S. M. Popoff et al., "A practical guide to digital
//! micro-mirror devices (DMDs) for wavefront shaping," *J. Phys. Photonics* 8,
//! 023002 (2026), doi:10.1088/2515-7647/ae60d5, §2 and App. A; the 2024
//! version, 6, 043001, was retracted for its blazed-number equation).
//!
//! In the thin-element approximation used here, a mirror reflects the
//! incident wave as the tilted plane would, restricted to the mirror's outline
//! projected onto the array plane:
//!
//! ```text
//! T(r) = √R · T_w · Σ_mirrors mask(r − r_ij) · exp(ik (t_spec − t_in)·(r − r_ij))
//! ```
//!
//! with `t` the in-plane (tangential) direction cosines, `t_spec` the specular
//! reflection of the incident direction off the mirror's own normal, and the
//! phase zero at each mirror's pivot. `T_w` is the window's single-pass power
//! transmittance (two passes make `T_w` in amplitude). Its Fourier series
//! over a pattern's unit cell is, exactly,
//!
//! ```text
//! c_mn = (√R T_w / A_cell) Σ_ij e^{−i2π G_mn·r_ij} · M̂(G_mn − (t_spec − t_in)/λ),
//! ```
//!
//! `M̂` the Fourier transform of the projected mirror outline: a square of
//! side `a = ρ·p` compressed by `cos θ` across the hinge, so
//! `M̂(f) = cos θ · a² · sinc(a (Af)ₓ) · sinc(a (Af)ᵧ)` with
//! `A = ââᵀ + cos θ ŵŵᵀ` (`â` the hinge, `ŵ` the in-plane direction across it).
//!
//! Not modelled: one mirror shading its neighbour at oblique incidence, the
//! mirror's edges and hinge, its curvature (Popoff et al. 2026, §3), and
//! anything rigorous electromagnetics would add for a pitch of a few
//! wavelengths (D. A. Pommet et al., *J. Opt. Soc. Am. A* 11, 1827 (1994):
//! scalar theory's error exceeds ±5% below about 14λ).
//!
//! # The LCoS SLM
//!
//! A phase-only modulator: each pixel retards the component of the field
//! polarized along the liquid crystal's director by a phase set by its grey
//! level; the orthogonal component sees no modulation (Z. Zhang, Z. You,
//! D. Chu, "Fundamentals of phase-only liquid crystal on silicon (LCOS)
//! devices," *Light Sci. Appl.* 3, e213 (2014), doi:10.1038/lsa.2014.94). The
//! dead space between pixels reflects unmodulated, which is a source of the
//! zero order (H. Zhang et al., *Appl. Opt.* 48(30), 5834 (2009),
//! doi:10.1364/AO.48.005834). With square pixels of side `ρ·p`,
//!
//! ```text
//! c_mn = g·δ_m0 δ_n0 + (ρ²/(PₓPᵧ)) sinc(ρm/Pₓ) sinc(ρn/Pᵧ) Σ_ij (√R e^{iφ_ij} − g) e^{−i2π(mi/Pₓ + nj/Pᵧ)},
//! ```
//!
//! `g` the dead space's complex reflectance. For an N-level blazed grating
//! with no dead space this is the classic `η₁ = sinc²(1/N)` (G. J. Swanson,
//! MIT Lincoln Laboratory TR-854 (1989)). Pixel crosstalk (the fringing field,
//! M. Persson, D. Engström, M. Goksör, *Opt. Express* 20, 22334 (2012)) has no
//! closed form here; `oc-wave` samples it.
//!
//! A phase set at the design wavelength `λ₀` is scaled as `λ₀/λ` at another
//! wavelength: first-order, ignoring the dispersion of the birefringence, and
//! reported as an approximation wherever it is used.
//!
//! # Frames and units
//!
//! The device lies in its local `XY` plane, facing `+Z`. Pixel `(i, j)` has its
//! centre at `((i − (Nₓ−1)/2)·p, (j − (Nᵧ−1)/2)·p)`. Light arrives travelling
//! toward `−Z`; reflected orders leave toward `+Z`. Pitch in µm, wavelength in
//! µm, both converted to mm once, where the kernel's lengths are needed.

use crate::complex::Cx;
use crate::vec3::Vec3;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// The axis a micromirror tilts about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TiltAxis {
    /// The pixel diagonal, `(1, 1)/√2`: 13.68, 10.8, 9.0, 7.6/7.56 µm devices.
    Diagonal,
    /// The pixel's `Y` side: the 5.4 µm devices (DLPA083B §4.2).
    Side,
}

/// A micromirror's state. Pattern value 0 is OFF, 1 is ON, 2 is flat (the
/// parked, unpowered position).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DmdState {
    Off,
    On,
    Flat,
}

impl DmdState {
    pub fn from_value(v: u16) -> Self {
        match v {
            0 => DmdState::Off,
            2 => DmdState::Flat,
            _ => DmdState::On,
        }
    }
    pub fn value(self) -> u16 {
        match self {
            DmdState::Off => 0,
            DmdState::On => 1,
            DmdState::Flat => 2,
        }
    }
    /// Signed tilt in units of the device's tilt angle.
    fn sign(self) -> f64 {
        match self {
            DmdState::Off => -1.0,
            DmdState::On => 1.0,
            DmdState::Flat => 0.0,
        }
    }
}

/// A DMD's optics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DmdSpec {
    /// Mirror tilt in the ON and OFF states, degrees (ON is `+`, OFF `−`).
    pub tilt_deg: f64,
    pub tilt_axis: TiltAxis,
    /// Mirror side over pitch, a linear fraction in `(0, 1]`.
    pub mirror_fill: f64,
    /// Mirror power reflectance.
    pub reflectance: f64,
    /// Window power transmittance, one pass (the light crosses it twice).
    pub window_transmittance: f64,
}

/// Grey level to phase, at the design wavelength.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhaseLut {
    /// `φ(g) = g · max_phase_rad / levels`: with `max_phase_rad = 2π` the top
    /// level is one step short of a full wave, as a calibrated 8-bit SLM is.
    Linear { levels: u16, max_phase_rad: f64 },
    /// One measured phase per grey level.
    Table { phase_rad: Vec<f64> },
}

impl PhaseLut {
    pub fn levels(&self) -> usize {
        match self {
            PhaseLut::Linear { levels, .. } => *levels as usize,
            PhaseLut::Table { phase_rad } => phase_rad.len(),
        }
    }
    pub fn phase(&self, g: u16) -> f64 {
        match self {
            PhaseLut::Linear { levels, max_phase_rad } => {
                (g as f64).min(*levels as f64 - 1.0) * max_phase_rad / *levels as f64
            }
            PhaseLut::Table { phase_rad } => phase_rad[(g as usize).min(phase_rad.len() - 1)],
        }
    }
    /// The grey level whose phase is nearest `φ` modulo 2π. For a linear table
    /// spanning a full wave this is `round(φ/step) mod levels`.
    pub fn level_for(&self, phi: f64) -> u16 {
        let w = phi.rem_euclid(2.0 * PI);
        match self {
            PhaseLut::Linear { levels, max_phase_rad } => {
                let step = max_phase_rad / *levels as f64;
                let g = (w / step).round() as i64;
                g.rem_euclid(*levels as i64).min(*levels as i64 - 1) as u16
            }
            PhaseLut::Table { phase_rad } => {
                let mut best = (f64::INFINITY, 0u16);
                for (g, p) in phase_rad.iter().enumerate() {
                    let d = (w - p.rem_euclid(2.0 * PI)).abs();
                    let d = d.min(2.0 * PI - d);
                    if d < best.0 {
                        best = (d, g as u16);
                    }
                }
                best.1
            }
        }
    }
}

/// An LCoS SLM's optics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LcosSpec {
    /// The liquid crystal's director, degrees from local `+X`.
    pub director_deg: f64,
    pub lut: PhaseLut,
    /// The wavelength the table holds for, µm.
    pub design_lambda_um: f64,
    /// Pixel electrode side over pitch, linear, `(0, 1]`.
    pub pixel_fill: f64,
    /// Dead space: amplitude reflectance and phase.
    pub gap_amplitude: f64,
    pub gap_phase_rad: f64,
    /// Width of the fringing-field broadening of the phase profile, µm
    /// (0: sharp pixels). Sampled only, in `oc-wave`.
    pub fringing_um: f64,
    /// Pixel power reflectance.
    pub reflectance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModulatorKind {
    Dmd(DmdSpec),
    Lcos(LcosSpec),
}

/// A target complex field for a hologram, in pixel units about the device
/// centre.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TargetField {
    Uniform,
    Gaussian { waist_px: f64 },
    /// Laguerre–Gauss `LG_p^l` (L. Allen et al., *Phys. Rev. A* 45, 8185
    /// (1992), doi:10.1103/PhysRevA.45.8185).
    LaguerreGauss { p: u32, l: i32, waist_px: f64 },
    HermiteGauss { m: u32, n: u32, waist_px: f64 },
}

impl TargetField {
    /// The unnormalized complex value at `(x, y)` pixels.
    pub fn value(&self, x: f64, y: f64) -> Cx {
        match self {
            TargetField::Uniform => Cx::ONE,
            TargetField::Gaussian { waist_px } => {
                Cx::real((-(x * x + y * y) / (waist_px * waist_px)).exp())
            }
            TargetField::LaguerreGauss { p, l, waist_px } => {
                let r2 = (x * x + y * y) / (waist_px * waist_px);
                let rho = (2.0 * r2).sqrt();
                let amp = rho.powi(l.unsigned_abs() as i32)
                    * laguerre(*p, l.unsigned_abs() as f64, 2.0 * r2)
                    * (-r2).exp();
                Cx::from_polar(amp, *l as f64 * y.atan2(x))
            }
            TargetField::HermiteGauss { m, n, waist_px } => {
                let (u, v) = (2f64.sqrt() * x / waist_px, 2f64.sqrt() * y / waist_px);
                let amp = hermite(*m, u) * hermite(*n, v) * (-(x * x + y * y) / (waist_px * waist_px)).exp();
                Cx::real(amp)
            }
        }
    }
}

/// Generalized Laguerre polynomial `L_p^α(x)` by its three-term recurrence.
fn laguerre(p: u32, a: f64, x: f64) -> f64 {
    let (mut l0, mut l1) = (1.0, 1.0 + a - x);
    if p == 0 {
        return l0;
    }
    for k in 1..p {
        let kf = k as f64;
        let l2 = ((2.0 * kf + 1.0 + a - x) * l1 - (kf + a) * l0) / (kf + 1.0);
        l0 = l1;
        l1 = l2;
    }
    l1
}

/// Physicists' Hermite polynomial `H_n(x)`.
fn hermite(n: u32, x: f64) -> f64 {
    let (mut h0, mut h1) = (1.0, 2.0 * x);
    if n == 0 {
        return h0;
    }
    for k in 1..n {
        let h2 = 2.0 * x * h1 - 2.0 * k as f64 * h0;
        h0 = h1;
        h1 = h2;
    }
    h1
}

/// A pattern described by how to make it: small, diffable, and exact at any
/// pixel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternGenerator {
    /// Binary (DMD): ON where `frac(u/period + phase) < duty`, `u` the pixel
    /// coordinate along `angle_deg`.
    BinaryGrating { period_px: f64, duty: f64, angle_deg: f64, phase: f64 },
    /// Phase (LCoS): a sawtooth `2π·frac(u/period)` along `angle_deg`.
    BlazedGrating { period_px: f64, angle_deg: f64 },
    /// Phase (LCoS): a thin lens `−πr²/(λ₀f)`, wrapped.
    FresnelLens { focal_mm: f64 },
    /// Phase (LCoS): a spiral `ℓ·atan2(y, x)`.
    Vortex { charge: i32 },
    /// Binary (DMD): Lee's hologram as Gutiérrez-Cuevas and Popoff write it,
    /// ON where `cos(2πν·r − φ) ≥ cos(πq)`, `A = sin(πq)`, with `A` the
    /// target's amplitude scaled to a peak of 1 and `φ` its phase: their
    /// Eq. (11) (R. Gutiérrez-Cuevas, S. M. Popoff, *J. Phys. Photonics* 6,
    /// 045022 (2024), doi:10.1088/2515-7647/ad8617; after W.-H. Lee,
    /// *Appl. Opt.* 18(21), 3661 (1979)). The −1 order then carries
    /// `(A/π)e^{iφ}` (their Eq. (15)).
    Lee { target: TargetField, period_px: f64, angle_deg: f64, peak: f64 },
    /// Phase (LCoS): the phases of the parts added, wrapped.
    Sum(Vec<PatternGenerator>),
}

/// What a device displays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    /// Every pixel the same value.
    Uniform { value: u16 },
    Generated(PatternGenerator),
    /// One value per pixel, row-major, `x` fastest: an imported bitmap or a
    /// designed hologram. Written to files packed (see [`packed_values`]):
    /// 1, 8 or 16 bits a pixel, PackBits, base64, and an FNV-1a checksum
    /// that a damaged file fails on load.
    Explicit(#[serde(with = "packed_values")] Vec<u16>),
}

/// The stored form of [`Pattern::Explicit`]: `{"bits", "count", "data",
/// "fnv1a"}`, `data` the base64 of the PackBits of the pixel bytes (1-bit
/// values packed eight to a byte, most significant first; 16-bit values
/// little-endian), `fnv1a` the 64-bit FNV-1a of those raw bytes as a hex
/// string. See [`crate::encoding`].
pub mod packed_values {
    use crate::encoding::{base64_decode, base64_encode, fnv1a64, packbits_decode, packbits_encode};
    use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Wire {
        bits: u8,
        count: usize,
        data: String,
        fnv1a: String,
    }

    fn raw(values: &[u16], bits: u8) -> Vec<u8> {
        match bits {
            1 => {
                let mut out = vec![0u8; values.len().div_ceil(8)];
                for (k, v) in values.iter().enumerate() {
                    if *v != 0 {
                        out[k / 8] |= 0x80 >> (k % 8);
                    }
                }
                out
            }
            8 => values.iter().map(|v| *v as u8).collect(),
            _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    pub fn serialize<S: Serializer>(values: &[u16], s: S) -> Result<S::Ok, S::Error> {
        let max = values.iter().copied().max().unwrap_or(0);
        let bits = if max <= 1 { 1 } else if max <= 255 { 8 } else { 16 };
        let bytes = raw(values, bits);
        Wire {
            bits,
            count: values.len(),
            data: base64_encode(&packbits_encode(&bytes)),
            fnv1a: format!("{:016x}", fnv1a64(&bytes)),
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u16>, D::Error> {
        let w = Wire::deserialize(d)?;
        let packed = base64_decode(&w.data).ok_or_else(|| D::Error::custom("pattern data is not base64"))?;
        let bytes = packbits_decode(&packed).ok_or_else(|| D::Error::custom("pattern data is a truncated PackBits stream"))?;
        if format!("{:016x}", fnv1a64(&bytes)) != w.fnv1a {
            return Err(D::Error::custom("pattern checksum does not match its data"));
        }
        let values: Vec<u16> = match w.bits {
            1 => (0..w.count)
                .map(|k| bytes.get(k / 8).map(|b| u16::from(b & (0x80 >> (k % 8)) != 0)))
                .collect::<Option<_>>()
                .ok_or_else(|| D::Error::custom("pattern data shorter than its count"))?,
            8 => bytes.iter().take(w.count).map(|b| *b as u16).collect(),
            16 => bytes.chunks_exact(2).take(w.count).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
            other => return Err(D::Error::custom(format!("{other} bits per pixel"))),
        };
        if values.len() != w.count {
            return Err(D::Error::custom("pattern data shorter than its count"));
        }
        Ok(values)
    }
}

/// A pixelated modulator and what it displays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialModulator {
    /// Columns along local `X`, rows along `Y`.
    pub pixels: [u32; 2],
    /// Pixel pitch, µm (square pixels).
    pub pitch_um: f64,
    pub kind: ModulatorKind,
    pub pattern: Pattern,
}

/// Why a device or a request was refused.
#[derive(Debug, Clone, PartialEq)]
pub struct ModulatorError(pub String);

impl std::fmt::Display for ModulatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModulatorError {}

fn err<T>(s: impl Into<String>) -> Result<T, ModulatorError> {
    Err(ModulatorError(s.into()))
}

impl SpatialModulator {
    pub fn validate(&self) -> Result<(), ModulatorError> {
        if self.pixels[0] == 0 || self.pixels[1] == 0 {
            return err("a modulator needs at least one pixel on each axis");
        }
        if !(self.pitch_um.is_finite() && self.pitch_um > 0.0) {
            return err(format!("pitch {} µm", self.pitch_um));
        }
        let frac = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        match &self.kind {
            ModulatorKind::Dmd(d) => {
                if !(d.tilt_deg.is_finite() && d.tilt_deg.abs() < 45.0) {
                    return err(format!("mirror tilt {}°", d.tilt_deg));
                }
                if !(frac(d.mirror_fill) && d.mirror_fill > 0.0) {
                    return err("mirror fill must be in (0, 1]");
                }
                if !frac(d.reflectance) || !frac(d.window_transmittance) {
                    return err("reflectance and window transmittance must be in [0, 1]");
                }
            }
            ModulatorKind::Lcos(l) => {
                if l.lut.levels() < 2 {
                    return err("a phase table needs at least two levels");
                }
                if let PhaseLut::Table { phase_rad } = &l.lut {
                    if !phase_rad.iter().all(|p| p.is_finite()) {
                        return err("a phase table entry is not finite");
                    }
                }
                if !(l.design_lambda_um.is_finite() && l.design_lambda_um > 0.0) {
                    return err("design wavelength");
                }
                if !(frac(l.pixel_fill) && l.pixel_fill > 0.0) || !frac(l.gap_amplitude) || !frac(l.reflectance) {
                    return err("fill, gap amplitude and reflectance must be in [0, 1]");
                }
                if !(l.fringing_um.is_finite() && l.fringing_um >= 0.0 && l.gap_phase_rad.is_finite()) {
                    return err("fringing width and gap phase must be finite, fringing ≥ 0");
                }
            }
        }
        if let Pattern::Explicit(v) = &self.pattern {
            if v.len() != self.pixels[0] as usize * self.pixels[1] as usize {
                return err("an explicit pattern must hold one value per pixel");
            }
        }
        Ok(())
    }

    pub fn pitch_mm(&self) -> f64 {
        self.pitch_um / 1000.0
    }

    /// Centre of pixel `(i, j)` in the device plane, mm.
    pub fn pixel_center_mm(&self, i: usize, j: usize) -> [f64; 2] {
        let p = self.pitch_mm();
        [
            (i as f64 - (self.pixels[0] as f64 - 1.0) / 2.0) * p,
            (j as f64 - (self.pixels[1] as f64 - 1.0) / 2.0) * p,
        ]
    }

    /// The value pixel `(i, j)` displays: a DMD state, or an LCoS grey level.
    pub fn value(&self, i: usize, j: usize) -> u16 {
        let x = i as f64 - (self.pixels[0] as f64 - 1.0) / 2.0;
        let y = j as f64 - (self.pixels[1] as f64 - 1.0) / 2.0;
        match &self.pattern {
            Pattern::Uniform { value } => *value,
            Pattern::Explicit(v) => v[j * self.pixels[0] as usize + i],
            Pattern::Generated(g) => match &self.kind {
                ModulatorKind::Dmd(_) => binary_value(g, x, y),
                ModulatorKind::Lcos(l) => l.lut.level_for(phase_value(g, x, y, self.pitch_mm(), l.design_lambda_um)),
            },
        }
    }
}

fn along(angle_deg: f64, x: f64, y: f64) -> f64 {
    let a = angle_deg.to_radians();
    x * a.cos() + y * a.sin()
}

fn binary_value(g: &PatternGenerator, x: f64, y: f64) -> u16 {
    match g {
        PatternGenerator::BinaryGrating { period_px, duty, angle_deg, phase } => {
            let f = (along(*angle_deg, x, y) / period_px + phase).rem_euclid(1.0);
            u16::from(f < *duty)
        }
        PatternGenerator::Lee { target, period_px, angle_deg, peak } => {
            let t = target.value(x, y);
            let a = (t.abs() / peak).clamp(0.0, 1.0);
            let q = a.asin() / PI;
            let carrier = 2.0 * PI * along(*angle_deg, x, y) / period_px;
            u16::from((carrier - t.arg()).cos() >= (PI * q).cos())
        }
        // A phase generator on a binary device: threshold its phase at π.
        other => u16::from(phase_value(other, x, y, 1e-3, 0.5).rem_euclid(2.0 * PI) < PI),
    }
}

fn phase_value(g: &PatternGenerator, x: f64, y: f64, pitch_mm: f64, lambda0_um: f64) -> f64 {
    match g {
        PatternGenerator::BlazedGrating { period_px, angle_deg } => {
            2.0 * PI * (along(*angle_deg, x, y) / period_px).rem_euclid(1.0)
        }
        PatternGenerator::FresnelLens { focal_mm } => {
            let r2 = (x * x + y * y) * pitch_mm * pitch_mm;
            -PI * r2 / (lambda0_um / 1000.0 * focal_mm)
        }
        PatternGenerator::Vortex { charge } => *charge as f64 * y.atan2(x),
        PatternGenerator::Sum(parts) => parts
            .iter()
            .map(|p| phase_value(p, x, y, pitch_mm, lambda0_um))
            .sum(),
        PatternGenerator::BinaryGrating { .. } | PatternGenerator::Lee { .. } => {
            PI * binary_value(g, x, y) as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry: the micromirror.
// ---------------------------------------------------------------------------

/// The hinge axis and the in-plane direction across it.
pub fn hinge(axis: TiltAxis) -> (Vec3, Vec3) {
    match axis {
        TiltAxis::Diagonal => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            (Vec3::new(s, s, 0.0), Vec3::new(-s, s, 0.0))
        }
        TiltAxis::Side => (Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
    }
}

/// A mirror's unit normal in `state`: `+Z` rotated by the signed tilt about
/// the hinge (Rodrigues; `ẑ ⊥ â`, so `R ẑ = cos θ ẑ + sin θ (â × ẑ)`).
pub fn facet_normal(d: &DmdSpec, state: DmdState) -> Vec3 {
    let th = state.sign() * d.tilt_deg.to_radians();
    let (a, _) = hinge(d.tilt_axis);
    let z = Vec3::Z;
    z * th.cos() + a.cross(z) * th.sin()
}

/// Mirror reflection of a direction about a unit normal.
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - n * (2.0 * d.dot(n))
}

/// The Fourier transform of a mirror's outline projected onto the array
/// plane, at spatial frequency `f` (cycles/mm): `cos θ · a² · sinc(a(Af)ₓ)
/// sinc(a(Af)ᵧ)`, `a` the mirror side in mm (see the module docs).
pub fn mirror_ft(d: &DmdSpec, pitch_mm: f64, state: DmdState, f: [f64; 2]) -> f64 {
    let th = state.sign() * d.tilt_deg.to_radians();
    let a = d.mirror_fill * pitch_mm;
    let (h, w) = hinge(d.tilt_axis);
    let c = th.cos();
    // A f = â(â·f) + cos θ ŵ(ŵ·f); A is symmetric.
    let fa = h.x * f[0] + h.y * f[1];
    let fw = w.x * f[0] + w.y * f[1];
    let af = [h.x * fa + c * w.x * fw, h.y * fa + c * w.y * fw];
    c * a * a * sinc(a * af[0]) * sinc(a * af[1])
}

/// The normalized sinc, `sin(πx)/(πx)`.
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        1.0 - (PI * x).powi(2) / 6.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// The blazed number `μ` of a diagonal-hinge DMD lit in the plane across its
/// hinge (S. M. Popoff et al., *J. Phys. Photonics* 8, 023002 (2026),
/// doi:10.1088/2515-7647/ae60d5, Eq. (5)): `θ_B` the mirror tilt and `α` the
/// incidence angle in that plane, both from the array normal, `d` the pitch,
///
/// ```text
/// μ = |(4 d / (√2 λ)) · sin θ_B · cos(θ_B − α)  mod 2  −  1|.
/// ```
///
/// The envelope's peak (the facets' specular direction) lies
/// `2 sin θ_B cos(θ_B − α)` from the incident direction in direction cosines
/// (their Eq. (4)), and the orders along the diagonal are `√2 λ/d` apart (the
/// lattice's period along it is `d/√2`): `μ` is one when the peak sits on an
/// order, zero when it lies halfway, where four orders share the light. The
/// 2024 version of this formula (*J. Phys. Photonics* 6, 043001, Eq. (2)),
/// which projected the angles by `tan/√2` and kept the pitch `d`, was
/// retracted (*J. Phys. Photonics* 7, 049701 (2025),
/// doi:10.1088/2515-7647/ae0605); `tests/dmd.rs` holds this one against the
/// 2026 paper's Fig. 6 and against this module's own facet kinematics.
pub fn blazed_number(pitch_um: f64, lambda_um: f64, tilt_deg: f64, incidence_deg: f64) -> f64 {
    let tb = tilt_deg.to_radians();
    let a = incidence_deg.to_radians();
    let x = 4.0 * pitch_um / (std::f64::consts::SQRT_2 * lambda_um) * tb.sin() * (tb - a).cos();
    (x.rem_euclid(2.0) - 1.0).abs()
}

// ---------------------------------------------------------------------------
// Closed-form diffraction orders of a periodic pattern.
// ---------------------------------------------------------------------------

/// One period of a pattern: `period[0] × period[1]` pixel values, row-major,
/// `x` fastest.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitCell {
    pub period: [usize; 2],
    pub values: Vec<u16>,
}

impl UnitCell {
    pub fn uniform(value: u16) -> Self {
        Self {
            period: [1, 1],
            values: vec![value],
        }
    }
}

/// How the ray tracer treats a DMD or an SLM.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum RayModel {
    /// The geometric limit: each mirror a tilted reflector, each electrode a
    /// phase-gradient deflector. No diffraction.
    #[default]
    Geometric,
    /// Each hit spawns the brightest diffraction orders of the pattern's unit
    /// cell where it hits ([`local_unit_cell`], [`unit_cell_orders`] with
    /// Harvey's renormalization): at most `max_orders`, none below
    /// `min_efficiency` of the arriving flux. The orders not followed are
    /// booked as untraced, never dropped.
    Orders { max_orders: u32, min_efficiency: f64 },
}

impl RayModel {
    pub fn is_geometric(&self) -> bool {
        matches!(self, RayModel::Geometric)
    }
    pub fn validate(&self) -> Result<(), ModulatorError> {
        match *self {
            RayModel::Geometric => Ok(()),
            RayModel::Orders { max_orders, min_efficiency } => {
                if max_orders == 0 || !(min_efficiency.is_finite() && (0.0..1.0).contains(&min_efficiency)) {
                    return err("orders: at least one, and a threshold in [0, 1)");
                }
                Ok(())
            }
        }
    }
}

/// The unit cell the pattern repeats with about pixel `(i, j)`, and whether it
/// is the pattern's own: one pixel of a uniform pattern, or the period of a
/// binary or blazed grating that repeats in whole pixels along a pixel axis.
/// Anything else is an 8 × 8 window from that pixel treated as one period,
/// an approximation (`false`).
pub fn local_unit_cell(d: &SpatialModulator, i: usize, j: usize) -> (UnitCell, bool) {
    let [nx, ny] = [d.pixels[0] as usize, d.pixels[1] as usize];
    let window = |px: usize, py: usize| {
        let (i0, j0) = (i.min(nx.saturating_sub(px)), j.min(ny.saturating_sub(py)));
        let values = (0..py)
            .flat_map(|b| (0..px).map(move |a| (a, b)))
            .map(|(a, b)| d.value((i0 + a).min(nx - 1), (j0 + b).min(ny - 1)))
            .collect();
        UnitCell { period: [px, py], values }
    };
    let period = match &d.pattern {
        Pattern::Uniform { value } => return (UnitCell::uniform(*value), true),
        Pattern::Generated(PatternGenerator::BinaryGrating { period_px, angle_deg, .. })
        | Pattern::Generated(PatternGenerator::BlazedGrating { period_px, angle_deg }) => {
            let p = period_px.round();
            let whole = (period_px - p).abs() < 1e-9 && p >= 1.0;
            let a = angle_deg.rem_euclid(180.0);
            match (whole, a) {
                (true, a) if a.abs() < 1e-9 && (p as usize) <= nx => Some([p as usize, 1]),
                (true, a) if (a - 90.0).abs() < 1e-9 && (p as usize) <= ny => Some([1, p as usize]),
                _ => None,
            }
        }
        _ => None,
    };
    match period {
        Some([px, py]) => (window(px, py), true),
        None => (window(8.min(nx), 8.min(ny)), false),
    }
}

/// How power is shared among the propagating orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderModel {
    /// `η = |c_mn|²`: the thin-element coefficients as they are. Power the
    /// evanescent orders would hold is lost.
    Tea,
    /// `η = |c_mn|² · Σ_all|c|² / Σ_propagating|c|²`: the propagating orders
    /// share the evanescent orders' power in proportion. This is the
    /// non-paraxial scalar treatment of J. E. Harvey et al., "Linear systems
    /// formulation of non-paraxial scalar diffraction theory," *Proc. SPIE*
    /// 8122, 812205 (2011), doi:10.1117/12.894562, which Texas Instruments'
    /// DLPA083B cites for its DMD efficiencies.
    HarveyRenormalized,
}

/// Which polarization component of the light an LCoS order table is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LcosBeam {
    /// Polarized along the director: modulated.
    Modulated,
    /// Orthogonal to it: reflected as by a uniform (pixelated) mirror.
    Unmodulated,
}

/// One diffraction order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Order {
    pub m: i32,
    pub n: i32,
    /// Unit direction of the reflected order, `+Z` going.
    pub dir: Vec3,
    /// The thin-element Fourier coefficient `c_mn`.
    pub amplitude: Cx,
    /// The power fraction under the table's [`OrderModel`].
    pub efficiency: f64,
}

/// The propagating orders of a periodic pattern under a plane wave.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderTable {
    pub orders: Vec<Order>,
    /// `Σ_all |c_mn|²`, by Parseval from the cell itself: the reflected power
    /// fraction before any order is dropped.
    pub total: f64,
    /// `Σ_propagating |c_mn|²`.
    pub propagating: f64,
    pub model: OrderModel,
}

impl OrderTable {
    /// The order carrying the most power.
    pub fn brightest(&self) -> Option<&Order> {
        self.orders
            .iter()
            .max_by(|a, b| a.efficiency.partial_cmp(&b.efficiency).unwrap())
    }
    /// Power fraction inside a cone of half-angle `asin(na)` about `axis`.
    pub fn within(&self, axis: Vec3, na: f64) -> f64 {
        let c = (1.0 - na * na).max(0.0).sqrt();
        self.orders
            .iter()
            .filter(|o| o.dir.dot(axis) >= c)
            .map(|o| o.efficiency)
            .sum()
    }
}

fn check_incidence(d_in: Vec3) -> Result<(), ModulatorError> {
    if !(d_in.is_finite() && (d_in.length() - 1.0).abs() < 1e-9 && d_in.z < 0.0) {
        return err("the incident direction must be a unit vector travelling toward −Z");
    }
    Ok(())
}

/// The propagating diffraction orders of `cell`, repeated over `device`, under
/// a unit plane wave travelling along `d_in`, at vacuum wavelength `lambda_um`
/// (the device sits in air). LCoS cells are for `beam`'s polarization; a DMD
/// ignores it. Refused for an LCoS with fringing, which has no closed form
/// here.
pub fn unit_cell_orders(
    device: &SpatialModulator,
    cell: &UnitCell,
    lambda_um: f64,
    d_in: Vec3,
    model: OrderModel,
    beam: LcosBeam,
) -> Result<OrderTable, ModulatorError> {
    device.validate()?;
    check_incidence(d_in)?;
    if cell.period[0] == 0 || cell.period[1] == 0 || cell.values.len() != cell.period[0] * cell.period[1] {
        return err("a unit cell must hold period[0] × period[1] values");
    }
    if !(lambda_um.is_finite() && lambda_um > 0.0) {
        return err("wavelength");
    }
    let lam = lambda_um / 1000.0;
    let p = device.pitch_mm();
    let [px, py] = cell.period;
    let (lx, ly) = (px as f64 * p, py as f64 * p);
    let t_in = [d_in.x, d_in.y];

    // Coefficient of order (m, n), and the analytic Σ|c|².
    let coeff: Box<dyn Fn(i32, i32) -> Cx>;
    let total: f64;
    match &device.kind {
        ModulatorKind::Dmd(d) => {
            let amp = d.reflectance.sqrt() * d.window_transmittance;
            let states: Vec<DmdState> = cell.values.iter().map(|v| DmdState::from_value(*v)).collect();
            // Specular tangential direction per state.
            let spec = |s: DmdState| {
                let r = reflect(d_in, facet_normal(d, s));
                [r.x, r.y]
            };
            let dd = d.clone();
            let specs: Vec<[f64; 2]> = states.iter().map(|s| spec(*s)).collect();
            let cell_area = lx * ly;
            total = amp * amp
                * states
                    .iter()
                    .map(|s| mirror_ft(&dd, p, *s, [0.0, 0.0]))
                    .sum::<f64>()
                / cell_area;
            coeff = Box::new(move |m, n| {
                let g = [m as f64 / lx, n as f64 / ly];
                let mut c = Cx::ZERO;
                for jj in 0..py {
                    for ii in 0..px {
                        let k = jj * px + ii;
                        let s = states[k];
                        let ts = specs[k];
                        let f = [g[0] - (ts[0] - t_in[0]) / lam, g[1] - (ts[1] - t_in[1]) / lam];
                        let r = [ii as f64 * p, jj as f64 * p];
                        c += Cx::expi(-2.0 * PI * (g[0] * r[0] + g[1] * r[1])) * mirror_ft(&dd, p, s, f);
                    }
                }
                c * (amp / cell_area)
            });
        }
        ModulatorKind::Lcos(l) => {
            if l.fringing_um > 0.0 {
                return err("an LCoS with fringing has no closed-form orders; sample it in oc-wave");
            }
            let rho = l.pixel_fill;
            let g = Cx::from_polar(l.gap_amplitude, l.gap_phase_rad);
            let scale = l.design_lambda_um / lambda_um;
            let pix: Vec<Cx> = cell
                .values
                .iter()
                .map(|v| {
                    let phi = match beam {
                        LcosBeam::Modulated => l.lut.phase(*v) * scale,
                        LcosBeam::Unmodulated => 0.0,
                    };
                    Cx::from_polar(l.reflectance.sqrt(), phi) - g
                })
                .collect();
            let nc = (px * py) as f64;
            total = rho * rho * l.reflectance + (1.0 - rho * rho) * l.gap_amplitude * l.gap_amplitude;
            coeff = Box::new(move |m, n| {
                let mut s = Cx::ZERO;
                for jj in 0..py {
                    for ii in 0..px {
                        let ph = -2.0 * PI * (m as f64 * ii as f64 / px as f64 + n as f64 * jj as f64 / py as f64);
                        s += pix[jj * px + ii] * Cx::expi(ph);
                    }
                }
                let mut c = s * (rho * rho * sinc(rho * m as f64 / px as f64) * sinc(rho * n as f64 / py as f64) / nc);
                if m == 0 && n == 0 {
                    c += g;
                }
                c
            });
        }
    }

    // Every order whose direction cosines stay inside the unit circle.
    let mmax = ((1.0 + t_in[0].abs()) * lx / lam).ceil() as i32 + 1;
    let nmax = ((1.0 + t_in[1].abs()) * ly / lam).ceil() as i32 + 1;
    let mut orders = Vec::new();
    let mut propagating = 0.0;
    for n in -nmax..=nmax {
        for m in -mmax..=mmax {
            let t = [t_in[0] + lam * m as f64 / lx, t_in[1] + lam * n as f64 / ly];
            let s2 = t[0] * t[0] + t[1] * t[1];
            if s2 >= 1.0 {
                continue;
            }
            let c = coeff(m, n);
            propagating += c.norm_sqr();
            orders.push(Order {
                m,
                n,
                dir: Vec3::new(t[0], t[1], (1.0 - s2).sqrt()),
                amplitude: c,
                efficiency: c.norm_sqr(),
            });
        }
    }
    if model == OrderModel::HarveyRenormalized && propagating > 0.0 {
        let k = total / propagating;
        for o in &mut orders {
            o.efficiency *= k;
        }
    }
    Ok(OrderTable {
        orders,
        total,
        propagating,
        model,
    })
}

/// The illumination direction that a DMD's ON mirrors send along `+Z`:
/// `ẑ` reflected off the ON facet and reversed into an incoming direction,
/// `ẑ − 2(ẑ·n)n`, which arrives at twice the tilt.
pub fn on_axis_illumination(d: &DmdSpec) -> Vec3 {
    let n = facet_normal(d, DmdState::On);
    reflect(Vec3::Z, n)
}

/// Directions filling a cone of numerical aperture `na` about `axis`
/// uniformly in direction cosines (equal étendue each), deterministically:
/// a Fibonacci spiral on the disk of radius `na` in the axis's own frame.
pub fn cone_directions(axis: Vec3, na: f64, count: usize) -> Vec<Vec3> {
    let a = axis.normalize();
    let helper = if a.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let u = a.cross(helper).normalize();
    let v = a.cross(u);
    let golden = PI * (3.0 - 5f64.sqrt());
    (0..count)
        .map(|i| {
            let r = na * ((i as f64 + 0.5) / count as f64).sqrt();
            let t = i as f64 * golden;
            let (s, c) = (r * t.cos(), r * t.sin());
            (a * (1.0 - s * s - c * c).sqrt() + u * s + v * c).normalize()
        })
        .collect()
}

/// The diffraction efficiency of an all-ON DMD in a projector, after Texas
/// Instruments DLPA083B §4.2: "the best way to approach broadband source
/// diffraction is to consider it as a combination of a large number of plane
/// waves varying in wavelength and direction ... combined incoherently";
/// the diffracted power is truncated by the projection lens's aperture.
///
/// Illumination fills a cone of f-number `illum_fno` about the direction the
/// ON mirrors send along `+Z`; the projection lens collects a cone of
/// f-number `proj_fno` about `+Z`. Each incident direction is a plane wave,
/// its orders come from [`unit_cell_orders`] under `model`, and the result is
/// the collected share of the reflected power, averaged over `samples`
/// directions: the window, the mirror reflectance and the fill factor are
/// TI's separate factors and are not in it.
pub fn dmd_projector_efficiency(
    device: &SpatialModulator,
    lambda_um: f64,
    illum_fno: f64,
    proj_fno: f64,
    samples: usize,
    model: OrderModel,
) -> Result<f64, ModulatorError> {
    let d = match &device.kind {
        ModulatorKind::Dmd(d) => d,
        _ => return err("a projector efficiency is for a DMD"),
    };
    if !(illum_fno > 0.5 && proj_fno > 0.5 && samples > 0) {
        return err("f-numbers above 0.5 and at least one sample");
    }
    let chief = on_axis_illumination(d);
    let na_ill = 1.0 / (2.0 * illum_fno);
    let na_proj = 1.0 / (2.0 * proj_fno);
    let cell = UnitCell::uniform(DmdState::On.value());
    let mut acc = 0.0;
    for dir in cone_directions(chief, na_ill, samples) {
        let t = unit_cell_orders(device, &cell, lambda_um, dir, model, LcosBeam::Modulated)?;
        acc += t.within(Vec3::Z, na_proj) / t.total;
    }
    Ok(acc / samples as f64)
}

/// The average of `f(λ)` over `[lo_nm, hi_nm]` for a flat power spectrum seen
/// by the eye: weighted by the photopic luminous efficiency `V(λ) = ȳ(λ)` of
/// the CIE 1931 standard observer ([`crate::colorimetry`]), trapezoidal in
/// steps of `step_nm`. That is the "photopic average" of TI DLPA083B Table 6-1
/// ("a source with a flat power spectrum with wavelengths of 420 nm–680 nm").
/// `f` takes the wavelength in µm.
pub fn photopic_average<E>(
    lo_nm: f64,
    hi_nm: f64,
    step_nm: f64,
    mut f: impl FnMut(f64) -> Result<f64, E>,
) -> Result<f64, E> {
    let n = ((hi_nm - lo_nm) / step_nm).round().max(1.0) as usize;
    let (mut num, mut den) = (0.0, 0.0);
    for k in 0..=n {
        let nm = lo_nm + (hi_nm - lo_nm) * k as f64 / n as f64;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 } * crate::colorimetry::cie_1931(nm / 1000.0)[1];
        num += w * f(nm / 1000.0)?;
        den += w;
    }
    Ok(num / den)
}

/// Device presets from Texas Instruments DLPA083B, Table 2-1 (pitch, nominal
/// tilt) and §4.2 (the tilt axis): name, pitch µm, tilt degrees, axis, the
/// illumination f-number TI pairs with that tilt, and TI's typical on-state
/// fill factor (an area fraction from a geometric ray trace at twice the tilt,
/// which includes shading; a mirror's own fill is not published there).
pub const TI_DMD_PRESETS: [(&str, f64, f64, TiltAxis, f64, f64); 6] = [
    ("DLP7000", 13.68, 12.0, TiltAxis::Diagonal, 2.4, 0.92),
    ("DLP650LE", 10.8, 12.0, TiltAxis::Diagonal, 2.4, 0.92),
    ("DLP780TE", 9.0, 14.5, TiltAxis::Diagonal, 2.0, 0.97),
    ("DLP9000", 7.56, 12.0, TiltAxis::Diagonal, 2.4, 0.94),
    ("DLP4501", 7.56, 12.0, TiltAxis::Diagonal, 2.4, 0.94),
    ("DLP3010", 5.4, 17.0, TiltAxis::Side, 1.7, 0.93),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn dmd(fill: f64) -> SpatialModulator {
        SpatialModulator {
            pixels: [16, 16],
            pitch_um: 7.56,
            kind: ModulatorKind::Dmd(DmdSpec {
                tilt_deg: 12.0,
                tilt_axis: TiltAxis::Diagonal,
                mirror_fill: fill,
                reflectance: 1.0,
                window_transmittance: 1.0,
            }),
            pattern: Pattern::Uniform { value: 1 },
        }
    }

    #[test]
    fn an_on_mirror_tilted_twelve_degrees_turns_a_normal_beam_through_twenty_four() {
        let d = match dmd(1.0).kind {
            ModulatorKind::Dmd(d) => d,
            _ => unreachable!(),
        };
        let n = facet_normal(&d, DmdState::On);
        assert!((n.length() - 1.0).abs() < 1e-15);
        assert!((n.z - 12f64.to_radians().cos()).abs() < 1e-15);
        let out = reflect(Vec3::new(0.0, 0.0, -1.0), n);
        assert!((out.z - 24f64.to_radians().cos()).abs() < 1e-14);
        // The tilt plane is across the diagonal hinge.
        assert!((out.x + out.y).abs() < 1e-15);
    }

    #[test]
    fn the_projected_mirror_transform_at_zero_is_its_projected_area() {
        let d = match dmd(0.9).kind {
            ModulatorKind::Dmd(d) => d,
            _ => unreachable!(),
        };
        let p = 7.56e-3;
        let a = 0.9 * p;
        let v = mirror_ft(&d, p, DmdState::On, [0.0, 0.0]);
        assert!((v - a * a * 12f64.to_radians().cos()).abs() < 1e-18);
        assert!((mirror_ft(&d, p, DmdState::Flat, [0.0, 0.0]) - a * a).abs() < 1e-18);
    }

    #[test]
    fn the_lut_round_trips_every_level() {
        let lut = PhaseLut::Linear { levels: 256, max_phase_rad: 2.0 * PI };
        for g in 0..256u16 {
            assert_eq!(lut.level_for(lut.phase(g)), g);
        }
        assert_eq!(lut.level_for(2.0 * PI), 0);
    }

    #[test]
    fn an_explicit_pattern_round_trips_packed_and_refuses_a_damaged_one() {
        for (values, bits) in [
            ((0..1000u16).map(|k| (k % 3 == 0) as u16).collect::<Vec<_>>(), 1),
            ((0..1000u16).map(|k| k % 256).collect(), 8),
            ((0..1000u16).map(|k| k * 37).collect(), 16),
        ] {
            let mut d = dmd(1.0);
            d.pixels = [40, 25];
            d.pattern = Pattern::Explicit(values.clone());
            let json = serde_json::to_string(&d).unwrap();
            assert!(json.contains(&format!("\"bits\":{bits}")), "{json}");
            let back: SpatialModulator = serde_json::from_str(&json).unwrap();
            assert_eq!(back, d);
            // One flipped character in the data is caught by the checksum.
            let i = json.find("\"data\":\"").unwrap() + 9;
            let mut bad = json.clone().into_bytes();
            bad[i] = if bad[i] == b'A' { b'B' } else { b'A' };
            assert!(serde_json::from_slice::<SpatialModulator>(&bad).is_err());
        }
        // A 1920 × 1080 binary hologram stays small.
        let mut d = dmd(1.0);
        d.pixels = [1920, 1080];
        d.pattern = Pattern::Explicit((0..1920 * 1080).map(|k| ((k % 1920) / 8 % 2) as u16).collect());
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.len() < 350_000, "{}", json.len());
    }

    #[test]
    fn laguerre_and_hermite_satisfy_their_low_orders() {
        // L_1^a(x) = 1 + a − x; H_2(x) = 4x² − 2.
        assert!((laguerre(1, 2.0, 0.7) - 2.3).abs() < 1e-15);
        assert!((laguerre(2, 0.0, 1.0) - (1.0 - 2.0 + 0.5)).abs() < 1e-15);
        assert!((hermite(2, 1.5) - 7.0).abs() < 1e-15);
    }
}
