//! Pixelated modulators in the wave engine: DMD and LCoS SLM.
//!
//! The device descriptions and their closed-form diffraction orders are in
//! `oc_core::modulator`, with the physics and its sources in that module's
//! documentation. This module puts a device into a sampled field, two ways:
//!
//! - **Near field, sampled.** [`dmd_reflect`] and [`lcos_reflect`] multiply a
//!   [`Field2`] lying on the device plane by the device's reflectance, sample
//!   by sample, with every pixel edge weighted by its exact area in the sample
//!   cell. A DMD reflects each mirror state into its own direction, so each
//!   state is its own *family*: its tilt is held as the field's analytic
//!   carrier `t_spec` and the samples keep only a piston phase per mirror,
//!   `exp(−ik(t_spec − t_in)·r_ij)`. Without the carrier, resolving the ramp
//!   across a 7.56 µm mirror tilted 12° at 0.4 µm would need about thirty
//!   samples per mirror; with it, four do. The piston step from mirror to
//!   mirror, modulo 2π, is exactly what Popoff et al.'s blaze condition
//!   measures. The power the families not followed carry is booked as
//!   `other_family`.
//! - **Far field, factorized.** [`dmd_far_field`] and [`lcos_far_field`] use the
//!   fact that every pixel of one state has the same response: the far field
//!   is the one-pixel transform times the lattice sum,
//!   `Û(f) = Σ_states M̂_s(f − t_s/λ) · Σ_{ij∈s} U_in(r_ij) e^{−i2πf·r_ij}`.
//!   The lattice sum is a chirp-z transform at one sample per pixel, so a
//!   whole 1920 × 1080 device costs one transform of that size, where a
//!   sampled field would need some 33 million samples. It assumes the
//!   incident field is a plane wave across each pixel (exact for a plane wave;
//!   flagged otherwise), and is exact within the thin-element model.
//!
//! The far field is returned on a grid of direction cosines `(α, β)`: the
//! angular spectrum `Û(α/λ, β/λ)`, whose `|Û|²/λ²` is power per unit `dα dβ`.
//! In direction cosines the orders are shift-invariant and the power density
//! carries no obliquity factor (J. E. Harvey, C. L. Vernold, *Appl. Opt.*
//! 37(34), 8158 (1998); J. E. Harvey et al., *Proc. SPIE* 8122, 812205 (2011)).

use crate::czt::czt2;
use crate::elements::Shape;
use crate::report::{Method, StepReport, ValidityFlag};
use crate::{Cx, Fft2, Field2, Grid2, WaveError, WaveReport};
use oc_core::modulator::{
    facet_normal, mirror_ft, reflect, sinc, DmdSpec, DmdState, LcosBeam, LcosSpec, ModulatorKind,
    SpatialModulator,
};
use oc_core::Vec3;
use std::f64::consts::PI;

fn me(e: oc_core::modulator::ModulatorError) -> WaveError {
    WaveError::OutOfRange(e.to_string())
}

/// The incident direction a field's carrier stands for: travelling toward
/// `−Z` with tangential direction cosines `carrier`.
pub fn incident_direction(carrier: [f64; 2]) -> Result<Vec3, WaveError> {
    let s2 = carrier[0] * carrier[0] + carrier[1] * carrier[1];
    if s2 >= 1.0 {
        return Err(WaveError::OutOfRange(
            "the incident carrier is not a propagating direction".into(),
        ));
    }
    Ok(Vec3::new(carrier[0], carrier[1], -(1.0 - s2).sqrt()))
}

fn dmd_spec(d: &SpatialModulator) -> Result<&DmdSpec, WaveError> {
    match &d.kind {
        ModulatorKind::Dmd(s) => Ok(s),
        _ => Err(WaveError::OutOfRange("not a DMD".into())),
    }
}

fn lcos_spec(d: &SpatialModulator) -> Result<&LcosSpec, WaveError> {
    match &d.kind {
        ModulatorKind::Lcos(s) => Ok(s),
        _ => Err(WaveError::OutOfRange("not an LCoS SLM".into())),
    }
}

/// The outline of a mirror in `state`, projected onto the array plane,
/// relative to its pixel centre, mm.
fn mirror_outline(d: &DmdSpec, pitch_mm: f64, state: DmdState) -> Shape {
    let a = 0.5 * d.mirror_fill * pitch_mm;
    let n = facet_normal(d, state);
    // Project the tilted square: rotate each corner as the mirror rotates and
    // drop z. The rotation maps ẑ to n about the hinge; corners on the hinge
    // stay put, corners across it shrink by cos θ.
    let th = (n.z).clamp(-1.0, 1.0).acos();
    let c = th.cos();
    let (h, w) = match d.tilt_axis {
        oc_core::modulator::TiltAxis::Diagonal => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            ([s, s], [-s, s])
        }
        oc_core::modulator::TiltAxis::Side => ([0.0, 1.0], [-1.0, 0.0]),
    };
    let corners = [[-a, -a], [a, -a], [a, a], [-a, a]];
    let v = corners
        .iter()
        .map(|p| {
            let ph = p[0] * h[0] + p[1] * h[1];
            let pw = p[0] * w[0] + p[1] * w[1];
            [h[0] * ph + c * w[0] * pw, h[1] * ph + c * w[1] * pw]
        })
        .collect();
    Shape::Polygon(v)
}

/// Pixel index range whose cells may overlap `[lo, hi]` along an axis.
fn pixel_range(lo: f64, hi: f64, n: usize, pitch: f64) -> std::ops::Range<usize> {
    let origin = -(n as f64) * pitch / 2.0; // left edge of pixel 0
    let a = ((lo - origin) / pitch).floor().max(0.0) as usize;
    let b = (((hi - origin) / pitch).floor() as i64 + 1).clamp(0, n as i64) as usize;
    a.min(n)..b
}

/// Reflect `u`, a field on the DMD plane whose carrier is the incident
/// direction, off the mirrors in `family`. The result carries that family's
/// specular direction as its carrier.
///
/// `u`'s frame is the device's: the device is centred on the field's plane
/// origin, its pixel axes on the plane's `X` and `Y`.
pub fn dmd_reflect(
    u: &Field2,
    device: &SpatialModulator,
    family: DmdState,
) -> Result<(Field2, WaveReport), WaveError> {
    device.validate().map_err(me)?;
    let d = dmd_spec(device)?;
    let d_in = incident_direction(u.carrier)?;
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    let amp = d.reflectance.sqrt() * d.window_transmittance;
    let refl = amp * amp;
    let k = u.k_mm();
    let t_in = [d_in.x, d_in.y];
    let spec = reflect(d_in, facet_normal(d, family));
    let t_s = [spec.x, spec.y];
    let outlines = [DmdState::Off, DmdState::On, DmdState::Flat].map(|s| (s, mirror_outline(d, p, s)));
    let outline = |s: DmdState| &outlines.iter().find(|(t, _)| *t == s).expect("all states").1;

    let g = u.grid;
    let [dx, dy] = g.pitch_mm;
    let da = dx * dy;
    let mut out = u.clone();
    let (mut absorbed, mut other, mut beyond) = (0.0, 0.0, 0.0);
    for j in 0..g.n[1] {
        let y = g.coord(1, j);
        let (y0, y1) = (y - 0.5 * dy, y + 0.5 * dy);
        for i in 0..g.n[0] {
            let x = g.coord(0, i);
            let (x0, x1) = (x - 0.5 * dx, x + 0.5 * dx);
            let idx = g.idx(i, j);
            let irr = u.data[idx].norm_sqr() * da;
            let mut acc = Cx::ZERO;
            let (mut cov_fam, mut cov_all) = (0.0, 0.0);
            for pj in pixel_range(y0, y1, ny, p) {
                for pi in pixel_range(x0, x1, nx, p) {
                    let c = device.pixel_center_mm(pi, pj);
                    let state = DmdState::from_value(device.value(pi, pj));
                    let cov = outline(state).cell_overlap(x0 - c[0], x1 - c[0], y0 - c[1], y1 - c[1]) / da;
                    if cov <= 0.0 {
                        continue;
                    }
                    cov_all += cov;
                    if state == family {
                        cov_fam += cov;
                        let piston = -k * ((t_s[0] - t_in[0]) * c[0] + (t_s[1] - t_in[1]) * c[1]);
                        acc += Cx::expi(piston) * cov;
                    }
                }
            }
            let v = u.data[idx] * acc * amp;
            out.data[idx] = v;
            absorbed += irr * ((1.0 - cov_all) + (1.0 - refl) * cov_all);
            other += irr * refl * (cov_all - cov_fam);
            beyond += irr * refl * cov_fam - v.norm_sqr() * da;
        }
    }
    out.carrier = t_s;
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();
    report.ledger.absorbed_element = absorbed;
    report.ledger.other_family = other;
    report.ledger.subcell_projection = beyond.max(0.0);
    report.ledger.detected = out.power();
    report.ledger.close();
    validity_flags(&mut report, p, u.lambda_um);
    report.approx("DMD as a thin element: mirror shading, edges, hinges and curvature not modelled");
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!(
            "DMD {}×{} at {} µm, {:?} family, carrier ({:.6}, {:.6})",
            nx, ny, device.pitch_um, family, t_s[0], t_s[1]
        ),
    });
    Ok((out, report))
}

fn validity_flags(report: &mut WaveReport, pitch_mm: f64, lambda_um: f64) {
    // Pommet, Moharam & Grann (1994): scalar error above ±5% below 14λ.
    let feature_um = pitch_mm * 1000.0;
    if feature_um < 14.0 * lambda_um {
        report.flag(ValidityFlag::ScalarLimit {
            feature_um,
            lambda_um,
        });
    }
}

/// The LCoS phase of pixel `(i, j)` at vacuum wavelength `lambda_um` for
/// `beam`: the table's phase scaled by `λ₀/λ`.
fn lcos_phase(device: &SpatialModulator, l: &LcosSpec, i: usize, j: usize, lambda_um: f64, beam: LcosBeam) -> f64 {
    match beam {
        LcosBeam::Modulated => l.lut.phase(device.value(i, j)) * l.design_lambda_um / lambda_um,
        LcosBeam::Unmodulated => 0.0,
    }
}

/// Reflect `u`, a field on the SLM plane, off an LCoS for one polarization
/// component.
///
/// The field is the scalar amplitude of that component: split an incident
/// linear polarization at angle `ψ` with [`lcos_split`], which sends
/// `cos(ψ − δ)` of the amplitude to the modulated beam and `sin(ψ − δ)` to the
/// unmodulated one (Zhang, You & Chu 2014). The two are orthogonally polarized
/// and add in intensity, never in amplitude.
///
/// Each sample takes the area-weighted mean of the reflectance over its cell:
/// `√R e^{iφ}` on the pixel electrodes, the dead space's `g` between them.
/// With `fringing_um > 0` the electrode phase at a sample is the pixel phases
/// averaged over a `w × w` window about it: a step between neighbouring
/// pixels becomes a linear ramp of width `w`, the linear broadening of the
/// fringing-field model ("Fringing-field effect in liquid-crystal
/// beam-steering devices: an approximate analytical model," *J. Opt. Soc. Am.
/// A* 21(10), 1996–2008 (2004); M. Persson, D. Engström, M. Goksör, *Opt.
/// Express* 20, 22334 (2012)). The window acts on the wrapped phases, so a 2π
/// reset gets a flyback region, as the liquid crystal's does.
pub fn lcos_reflect(u: &Field2, device: &SpatialModulator, beam: LcosBeam) -> Result<(Field2, WaveReport), WaveError> {
    device.validate().map_err(me)?;
    let l = lcos_spec(device)?;
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    let half = 0.5 * l.pixel_fill * p;
    let g_gap = Cx::from_polar(l.gap_amplitude, l.gap_phase_rad);
    let w = l.fringing_um / 1000.0;
    let lam = u.lambda_um;
    let grid = u.grid;
    let [dx, dy] = grid.pitch_mm;
    let da = dx * dy;
    let mut out = u.clone();
    let (mut absorbed, mut beyond) = (0.0, 0.0);

    // 1D overlap weights of [c − w/2, c + w/2] with each pixel's full cell.
    let weights = |c: f64, n: usize| -> Vec<(usize, f64)> {
        let mut v = Vec::new();
        for i in pixel_range(c - 0.5 * w, c + 0.5 * w, n, p) {
            let left = -(n as f64) * p / 2.0 + i as f64 * p;
            let o = ((c + 0.5 * w).min(left + p) - (c - 0.5 * w).max(left)).max(0.0) / w;
            if o > 0.0 {
                v.push((i, o));
            }
        }
        v
    };

    for j in 0..grid.n[1] {
        let y = grid.coord(1, j);
        let (y0, y1) = (y - 0.5 * dy, y + 0.5 * dy);
        let wy = if w > 0.0 { weights(y, ny) } else { vec![] };
        for i in 0..grid.n[0] {
            let x = grid.coord(0, i);
            let (x0, x1) = (x - 0.5 * dx, x + 0.5 * dx);
            let idx = grid.idx(i, j);
            let irr = u.data[idx].norm_sqr() * da;
            // Electrode coverage per pixel, and the cell inside the device.
            let mut acc = Cx::ZERO;
            let mut cov_el = 0.0;
            let smooth_phase = if w > 0.0 && beam == LcosBeam::Modulated {
                let wx = weights(x, nx);
                let mut ph = 0.0;
                let mut tot = 0.0;
                for &(pj, vy) in &wy {
                    for &(pi, vx) in &wx {
                        ph += vx * vy * lcos_phase(device, l, pi, pj, lam, beam);
                        tot += vx * vy;
                    }
                }
                if tot > 0.0 {
                    Some(ph / tot)
                } else {
                    None
                }
            } else {
                None
            };
            let mut cov_dev = 0.0;
            for pj in pixel_range(y0, y1, ny, p) {
                for pi in pixel_range(x0, x1, nx, p) {
                    let c = device.pixel_center_mm(pi, pj);
                    let cell = Shape::Rect { c, h: [0.5 * p, 0.5 * p] };
                    cov_dev += cell.cell_overlap(x0, x1, y0, y1) / da;
                    let el = Shape::Rect { c, h: [half, half] };
                    let cov = el.cell_overlap(x0, x1, y0, y1) / da;
                    if cov <= 0.0 {
                        continue;
                    }
                    cov_el += cov;
                    let phi = smooth_phase.unwrap_or_else(|| lcos_phase(device, l, pi, pj, lam, beam));
                    acc += Cx::from_polar(l.reflectance.sqrt() * cov, phi);
                }
            }
            let cov_gap = (cov_dev - cov_el).max(0.0);
            acc += g_gap * cov_gap;
            let v = u.data[idx] * acc;
            out.data[idx] = v;
            // Physically reflected: R on electrodes, |g|² on the dead space.
            let refl = l.reflectance * cov_el + l.gap_amplitude * l.gap_amplitude * cov_gap;
            absorbed += irr * (cov_el + cov_gap - refl) + irr * (1.0 - cov_dev).max(0.0);
            beyond += irr * refl - v.norm_sqr() * da;
        }
    }
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();
    report.ledger.absorbed_element = absorbed;
    // A sample spanning several phases carries less than the cell reflects:
    // the difference is the sub-cell projection error of the pixel edges.
    report.ledger.subcell_projection = beyond.max(0.0);
    report.ledger.detected = out.power();
    report.ledger.close();
    validity_flags(&mut report, p, lam);
    if beam == LcosBeam::Modulated && (lam - l.design_lambda_um).abs() > 1e-12 {
        report.flag(ValidityFlag::PhaseDispersionApprox);
    }
    report.approx("LCoS as a thin element: dead space unmodulated, fringing as linear broadening");
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!("LCoS {}×{} at {} µm, {:?} beam", nx, ny, device.pitch_um, beam),
    });
    Ok((out, report))
}

/// Split an incident linear polarization at `psi_deg` (from local `+X`)
/// into the amplitudes along and across the director: `(cos(ψ − δ),
/// sin(ψ − δ))`. Their squares are the powers each beam carries.
pub fn lcos_split(device: &SpatialModulator, psi_deg: f64) -> Result<(f64, f64), WaveError> {
    let l = lcos_spec(device)?;
    let a = (psi_deg - l.director_deg).to_radians();
    Ok((a.cos(), a.sin()))
}

/// A grid of direction cosines: `n` samples per axis from `start`, `step` apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngularGrid {
    pub start: [f64; 2],
    pub step: [f64; 2],
    pub n: [usize; 2],
}

/// A far field on a grid of direction cosines: `data` holds the angular
/// spectrum `Û(α/λ, β/λ)` (mm² of field amplitude), so `|Û|²/λ²` is power per
/// unit `dα dβ`.
#[derive(Debug, Clone, PartialEq)]
pub struct AngularField {
    /// First direction cosine and step, per axis.
    pub start: [f64; 2],
    pub step: [f64; 2],
    pub n: [usize; 2],
    /// Wavelength in the medium, mm.
    pub lambda_mm: f64,
    pub data: Vec<Cx>,
}

impl AngularField {
    pub fn alpha(&self, i: usize) -> f64 {
        self.start[0] + i as f64 * self.step[0]
    }
    pub fn beta(&self, j: usize) -> f64 {
        self.start[1] + j as f64 * self.step[1]
    }
    /// Power per unit `dα dβ` at sample `(i, j)`; zero where `α² + β² ≥ 1`.
    pub fn power_density(&self, i: usize, j: usize) -> f64 {
        let (a, b) = (self.alpha(i), self.beta(j));
        if a * a + b * b >= 1.0 {
            return 0.0;
        }
        self.data[j * self.n[0] + i].norm_sqr() / (self.lambda_mm * self.lambda_mm)
    }
    /// `Σ power_density · Δα Δβ`: the power the grid holds.
    pub fn power(&self) -> f64 {
        let mut s = 0.0;
        for j in 0..self.n[1] {
            for i in 0..self.n[0] {
                s += self.power_density(i, j);
            }
        }
        s * self.step[0] * self.step[1]
    }
}

/// The lattice sum `Σ_ij v_ij e^{−i2πf·r_ij}` over the device's pixel centres
/// on a frequency grid, by chirp-z.
fn lattice_sum(
    device: &SpatialModulator,
    values: Vec<Cx>,
    f0: [f64; 2],
    df: [f64; 2],
    n: [usize; 2],
    fft: &mut Fft2,
) -> Result<Vec<Cx>, WaveError> {
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    // A Grid2 whose samples sit on the pixel centres: for even counts the
    // grid's centre sample is half a pitch off the device centre.
    let cx = if nx % 2 == 0 { -0.5 * p } else { 0.0 };
    let cy = if ny % 2 == 0 { -0.5 * p } else { 0.0 };
    let g = Grid2::new([nx, ny], [p, p], [-cx, -cy])?;
    let mut lat = Field2::zeros(g, 1.0)?;
    debug_assert!((g.coord(0, 0) - device.pixel_center_mm(0, 0)[0]).abs() < 1e-12 * p.max(1.0));
    lat.data = values;
    let mut out = czt2(&lat, f0, df, n, fft);
    let inv = 1.0 / (p * p);
    for v in &mut out {
        *v *= inv;
    }
    Ok(out)
}

/// The far field of a whole DMD under a plane wave of unit amplitude
/// travelling along `d_in`, on a grid of direction cosines, by the factorized
/// transform (see the module docs). `states` limits which mirror families
/// contribute (all three for the whole field).
pub fn dmd_far_field(
    device: &SpatialModulator,
    lambda_um: f64,
    d_in: Vec3,
    grid: AngularGrid,
    states: &[DmdState],
    fft: &mut Fft2,
) -> Result<AngularField, WaveError> {
    let AngularGrid { start, step, n } = grid;
    device.validate().map_err(me)?;
    let d = dmd_spec(device)?;
    let lam = lambda_um / 1000.0;
    let k = 2.0 * PI / lam;
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    let amp = d.reflectance.sqrt() * d.window_transmittance;
    let f0 = [start[0] / lam, start[1] / lam];
    let df = [step[0] / lam, step[1] / lam];
    let mut total = vec![Cx::ZERO; n[0] * n[1]];
    for &s in states {
        let spec = reflect(d_in, facet_normal(d, s));
        let ts = [spec.x, spec.y];
        let mut vals = vec![Cx::ZERO; nx * ny];
        let mut any = false;
        for j in 0..ny {
            for i in 0..nx {
                if DmdState::from_value(device.value(i, j)) == s {
                    let c = device.pixel_center_mm(i, j);
                    vals[j * nx + i] = Cx::expi(k * (d_in.x * c[0] + d_in.y * c[1]));
                    any = true;
                }
            }
        }
        if !any {
            continue;
        }
        let b = lattice_sum(device, vals, f0, df, n, fft)?;
        for jj in 0..n[1] {
            for ii in 0..n[0] {
                let f = [f0[0] + ii as f64 * df[0], f0[1] + jj as f64 * df[1]];
                let m = mirror_ft(d, p, s, [f[0] - ts[0] / lam, f[1] - ts[1] / lam]);
                total[jj * n[0] + ii] += b[jj * n[0] + ii] * (amp * m);
            }
        }
    }
    Ok(AngularField {
        start,
        step,
        n,
        lambda_mm: lam,
        data: total,
    })
}

/// The far field of a whole LCoS under a plane wave of unit amplitude along
/// `d_in`, for one polarization beam, by the factorized transform: pixels
/// `(√R e^{iφ} − g)` times the electrode's transform, plus the dead space's
/// `g` over the whole device. No fringing (refused: it has no factorized
/// form; sample it with [`lcos_reflect`]).
pub fn lcos_far_field(
    device: &SpatialModulator,
    lambda_um: f64,
    d_in: Vec3,
    beam: LcosBeam,
    grid: AngularGrid,
    fft: &mut Fft2,
) -> Result<AngularField, WaveError> {
    let AngularGrid { start, step, n } = grid;
    device.validate().map_err(me)?;
    let l = lcos_spec(device)?;
    if l.fringing_um > 0.0 {
        return Err(WaveError::OutOfRange(
            "fringing has no factorized far field; sample the device with lcos_reflect".into(),
        ));
    }
    let lam = lambda_um / 1000.0;
    let k = 2.0 * PI / lam;
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    let g = Cx::from_polar(l.gap_amplitude, l.gap_phase_rad);
    let a = l.pixel_fill * p;
    let f0 = [start[0] / lam, start[1] / lam];
    let df = [step[0] / lam, step[1] / lam];
    let mut vals = vec![Cx::ZERO; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let c = device.pixel_center_mm(i, j);
            let inc = Cx::expi(k * (d_in.x * c[0] + d_in.y * c[1]));
            let r = Cx::from_polar(l.reflectance.sqrt(), lcos_phase(device, l, i, j, lambda_um, beam));
            vals[j * nx + i] = inc * (r - g);
        }
    }
    let b = lattice_sum(device, vals, f0, df, n, fft)?;
    let (wx, wy) = (nx as f64 * p, ny as f64 * p);
    let mut out = vec![Cx::ZERO; n[0] * n[1]];
    for jj in 0..n[1] {
        for ii in 0..n[0] {
            let f = [f0[0] + ii as f64 * df[0], f0[1] + jj as f64 * df[1]];
            // The incident tilt runs across each electrode too: its transform
            // is taken at f − t_in/λ, as the dead space's is over the device.
            let fr = [f[0] - d_in.x / lam, f[1] - d_in.y / lam];
            let el = a * a * sinc(a * fr[0]) * sinc(a * fr[1]);
            let dev = wx * wy * sinc(wx * fr[0]) * sinc(wy * fr[1]);
            out[jj * n[0] + ii] = b[jj * n[0] + ii] * el + g * dev;
        }
    }
    Ok(AngularField {
        start,
        step,
        n,
        lambda_mm: lam,
        data: out,
    })
}
