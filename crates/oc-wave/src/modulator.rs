//! Pixelated modulators in the wave engine: DMD and LCoS SLM.
//!
//! The device descriptions and their closed-form diffraction orders are in
//! `oc_core::modulator`, with the physics and its sources in that module's
//! documentation. This module puts a device into a sampled field, two ways:
//!
//! - **Near field, sampled.** [`dmd_reflect`] and [`lcos_reflect`] multiply a
//!   [`Field2`] lying on the device plane by the device's mask, sample by
//!   sample. The mask is band-limited exactly to the field's grid: its
//!   closed-form spectrum, the pixel's own transform (a mirror's projected
//!   outline, an electrode's square) times the lattice sum of the pixels'
//!   complex factors, taken on the grid's own frequencies and summed back
//!   onto its samples (Q17 of the expert review). A mask averaged over each
//!   sample's cell, as before, puts the cell's `sinc(f·Δx)` on every order: at
//!   8 samples a pixel an order 1.6 lattice steps out kept 87.5% of its power.
//!   Band-limited, every order the grid holds has its exact amplitude at any
//!   sampling, and what the grid cannot hold is booked as
//!   `spectral_projection`. The illumination multiplies the mask sample by
//!   sample, so a beam that varies across a pixel is carried as it varies.
//!   The power the mirrors intercept and reflect is still each sample's exact
//!   cell coverage. An LCoS whose fringing smooths its phases has no
//!   closed-form transform and keeps the cell average, which the report
//!   names. So does a device lit within a pixel of a window edge it reaches
//!   past: the band-limited mask is periodic over the window, and would fold
//!   the pixels the edge cuts onto the far side. A DMD reflects each mirror state into its own direction, so each
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

/// Above this share of the field's power within a pixel of a window edge the
/// device reaches past, the mask is averaged over each sample's cell instead
/// of band-limited ([`edge_share`]).
pub const FOLD_TOLERANCE: f64 = 1e-9;

/// The share of `u`'s power on its samples within one pitch of a window edge
/// that the device reaches past: there the band-limited mask, periodic over
/// the window, folds the pixels the edge cuts onto the far side.
fn edge_share(u: &Field2, device: &SpatialModulator) -> f64 {
    let g = u.grid;
    let p = device.pitch_mm();
    let total = u.power();
    if total <= 0.0 {
        return 0.0;
    }
    let mut strips: [Vec<(f64, f64)>; 2] = [Vec::new(), Vec::new()];
    for (axis, strip) in strips.iter_mut().enumerate() {
        let h = 0.5 * device.pixels[axis] as f64 * p;
        let half = 0.5 * g.pitch_mm[axis];
        let (lo, hi) = (g.coord(axis, 0) - half, g.coord(axis, g.n[axis] - 1) + half);
        if lo > -h + 1e-9 * p {
            strip.push((lo, lo + p));
        }
        if hi < h - 1e-9 * p {
            strip.push((hi - p, hi));
        }
    }
    let inside = |v: f64, s: &[(f64, f64)]| s.iter().any(|&(a, b)| v >= a && v <= b);
    let mut edge = 0.0;
    for j in 0..g.n[1] {
        let y = g.coord(1, j);
        for i in 0..g.n[0] {
            if inside(g.coord(0, i), &strips[0]) || inside(y, &strips[1]) {
                edge += u.data[g.idx(i, j)].norm_sqr();
            }
        }
    }
    edge * g.cell_area_mm2() / total
}

/// Pixel index range whose cells may overlap `[lo, hi]` along an axis.
fn pixel_range(lo: f64, hi: f64, n: usize, pitch: f64) -> std::ops::Range<usize> {
    let origin = -(n as f64) * pitch / 2.0; // left edge of pixel 0
    let a = ((lo - origin) / pitch).floor().max(0.0) as usize;
    let b = (((hi - origin) / pitch).floor() as i64 + 1).clamp(0, n as i64) as usize;
    a.min(n)..b
}

/// A device's mask on the grid `g`, band-limited exactly to the grid's band:
/// `Σ_ij factor(i, j)·P(r − c_ij)`, `P` the pixel's response, from its
/// spectrum `pixel_ft(f)·Σ_ij factor(i, j) e^{−i2πf·c_ij}` on the grid's own
/// frequencies `(k − n/2)/(nΔx)` and summed back onto the grid's samples by
/// chirp-z. The pixels are those whose cells may overlap the grid; the sum is
/// periodic over the grid's window, so a pixel straddling its edge folds
/// across (the illumination is negligible there on a wave path). `extra` adds
/// a spectrum of its own (an LCoS's dead space).
fn band_limited_mask(
    g: &Grid2,
    device: &SpatialModulator,
    pixel_ft: &dyn Fn([f64; 2]) -> Cx,
    factor: &dyn Fn(usize, usize) -> Cx,
    extra: Option<&dyn Fn([f64; 2]) -> Cx>,
    fft: &mut Fft2,
) -> Result<Vec<Cx>, WaveError> {
    let p = device.pitch_mm();
    let [nx, ny] = [device.pixels[0] as usize, device.pixels[1] as usize];
    let [dx, dy] = g.pitch_mm;
    let (x_lo, x_hi) = (g.coord(0, 0) - 0.5 * dx, g.coord(0, g.n[0] - 1) + 0.5 * dx);
    let (y_lo, y_hi) = (g.coord(1, 0) - 0.5 * dy, g.coord(1, g.n[1] - 1) + 0.5 * dy);
    let (ri, rj) = (pixel_range(x_lo, x_hi, nx, p), pixel_range(y_lo, y_hi, ny, p));
    let df = [1.0 / (g.n[0] as f64 * dx), 1.0 / (g.n[1] as f64 * dy)];
    let f0 = [-((g.n[0] / 2) as f64) * df[0], -((g.n[1] / 2) as f64) * df[1]];
    let f_at = |k: usize, axis: usize| f0[axis] + k as f64 * df[axis];
    let mut spec = vec![Cx::ZERO; g.len()];
    if !ri.is_empty() && !rj.is_empty() {
        let (lx, ly) = (ri.len(), rj.len());
        let c0 = device.pixel_center_mm(ri.start, rj.start);
        let lg = Grid2::new([lx, ly], [p, p], [c0[0] + (lx / 2) as f64 * p, c0[1] + (ly / 2) as f64 * p])?;
        let mut lat = Field2::zeros(lg, 1.0)?;
        for j in 0..ly {
            for i in 0..lx {
                lat.data[j * lx + i] = factor(ri.start + i, rj.start + j);
            }
        }
        // czt2 is the sampled continuous transform, Σ v e^{−i2πf·c} p².
        let lsum = czt2(&lat, f0, df, g.n, fft);
        let inv = 1.0 / (p * p);
        for ky in 0..g.n[1] {
            for kx in 0..g.n[0] {
                let f = [f_at(kx, 0), f_at(ky, 1)];
                spec[ky * g.n[0] + kx] = lsum[ky * g.n[0] + kx] * inv * pixel_ft(f);
            }
        }
    }
    if let Some(e) = extra {
        for ky in 0..g.n[1] {
            for kx in 0..g.n[0] {
                spec[ky * g.n[0] + kx] += e([f_at(kx, 0), f_at(ky, 1)]);
            }
        }
    }
    // Back onto the samples: Σ_k D̂(f_k) e^{+i2πf_k·x} Δf², one axis at a time.
    let (x0, y0) = (g.coord(0, 0), g.coord(1, 0));
    let mut rows = vec![Cx::ZERO; g.len()];
    for ky in 0..g.n[1] {
        let line = &spec[ky * g.n[0]..(ky + 1) * g.n[0]];
        let r = crate::czt::czt_line(line, f0[0], df[0], -x0, -dx, g.n[0], fft);
        rows[ky * g.n[0]..(ky + 1) * g.n[0]].copy_from_slice(&r);
    }
    let mut out = vec![Cx::ZERO; g.len()];
    let mut col = vec![Cx::ZERO; g.n[1]];
    for i in 0..g.n[0] {
        for ky in 0..g.n[1] {
            col[ky] = rows[ky * g.n[0] + i];
        }
        let c = crate::czt::czt_line(&col, f0[1], df[1], -y0, -dy, g.n[1], fft);
        for (j, v) in c.into_iter().enumerate() {
            out[j * g.n[0] + i] = v;
        }
    }
    Ok(out)
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
    // The family's mask, band-limited exactly to the grid: each mirror's
    // projected outline, its piston, the reflectance's amplitude. Averaged
    // over each sample's cell instead where lit pixels would fold across the
    // window's edge.
    let folded = edge_share(u, device) > FOLD_TOLERANCE;
    let mask = if folded {
        None
    } else {
        Some(band_limited_mask(
            &g,
            device,
            &|f| Cx::real(amp * mirror_ft(d, p, family, f)),
            &|i, j| {
                if DmdState::from_value(device.value(i, j)) == family {
                    let c = device.pixel_center_mm(i, j);
                    Cx::expi(-k * ((t_s[0] - t_in[0]) * c[0] + (t_s[1] - t_in[1]) * c[1]))
                } else {
                    Cx::ZERO
                }
            },
            None,
            &mut Fft2::new(),
        )?)
    };
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
                        if folded {
                            let piston = -k * ((t_s[0] - t_in[0]) * c[0] + (t_s[1] - t_in[1]) * c[1]);
                            acc += Cx::expi(piston) * cov;
                        }
                    }
                }
            }
            let v = u.data[idx] * mask.as_ref().map_or(acc * amp, |m| m[idx]);
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
    if folded {
        // A sample spanning a mirror's edge carries less than the cell
        // reflects: the sub-cell projection error of the edges.
        report.ledger.subcell_projection = beyond.max(0.0);
        report.approx("DMD lit within a pixel of a window edge it reaches past: its mask is averaged over each sample's cell, which weights an order m by sinc(m/s) at s samples a pixel, since the band-limited one would fold the cut mirrors across the window");
    } else {
        // What the family reflects beyond the grid's band (its mirrors' edges).
        report.ledger.spectral_projection = beyond;
    }
    report.ledger.detected = out.power();
    report.ledger.close();
    validity_flags(&mut report, device, u.lambda_um, t_in);
    let incidence =(t_in[0] * t_in[0] + t_in[1] * t_in[1]).sqrt().min(1.0).asin().to_degrees();
    if incidence > FACET_INCIDENCE_FLAG_DEG {
        report.flag(ValidityFlag::ThinElement {
            detail: format!(
                "DMD met at {incidence:.1}°: the array-normal mask reflects ρ² cos θ of each mirror's cell, where an isolated {family:?} facet would intercept {:.4} times that; neighbours' shading, which an actual array has, is not modelled either way",
                facet_interception_ratio(d, family, d_in)
            ),
        });
    }
    report.approx("DMD as a thin element: mirror shading, edges, hinges and curvature not modelled");
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!(
            "DMD {}×{} at {} µm, {:?} family, carrier ({:.6}, {:.6}); its mask {}, {:.3e} of what the family reflects {}",
            nx,
            ny,
            device.pitch_um,
            family,
            t_s[0],
            t_s[1],
            if folded { "averaged over each sample's cell (lit at a window edge)" } else { "band-limited exactly to the field's grid (each mirror's projected outline, its closed-form transform, times the lattice sum)" },
            beyond / (beyond + out.power()).max(f64::MIN_POSITIVE),
            if folded { "lost to the cells' averaging" } else { "beyond the grid's band" }
        ),
    });
    Ok((out, report))
}

/// The scalar-limit heuristic for `device` met along the in-plane direction
/// `t_in` ([`ValidityFlag::ScalarLimit`]): raised when the active feature is
/// under 14 wavelengths, with the pitch, the gap, the relief, the index
/// contrast and the incidence recorded.
fn validity_flags(report: &mut WaveReport, device: &SpatialModulator, lambda_um: f64, t_in: [f64; 2]) {
    let pitch_um = device.pitch_um;
    let (fill, depth_um) = match &device.kind {
        ModulatorKind::Dmd(d) => (d.mirror_fill, Some(d.mirror_fill * pitch_um * d.tilt_deg.to_radians().sin())),
        ModulatorKind::Lcos(l) => (l.pixel_fill, None),
    };
    let feature_um = fill * pitch_um;
    if feature_um < 14.0 * lambda_um {
        report.flag(ValidityFlag::ScalarLimit {
            pitch_um,
            feature_um,
            gap_um: pitch_um - feature_um,
            depth_um,
            index_contrast: None,
            incidence_deg: (t_in[0] * t_in[0] + t_in[1] * t_in[1]).sqrt().min(1.0).asin().to_degrees(),
            lambda_um,
        });
    }
}

/// Above this incidence on a DMD (degrees) the report raises
/// [`ValidityFlag::ThinElement`] with the interception the array-normal mask
/// and an isolated facet would differ by (`1/cos 15°`, 3.5%, there).
pub const FACET_INCIDENCE_FLAG_DEG: f64 = 15.0;

/// The far-field amplitude of one flat square facet of side `a` (mm) with
/// physical in-plane unit tangents `u_m`, `v_m`, centred at its pivot, under
/// a uniform plane wave of wavevector `k_in` and seen along `k_out` (rad/mm):
/// the surface integral `∫∫ e^{i(k_in − k_out)·r} dA` over the facet,
/// `a² sinc(a q·u_m/2π) sinc(a q·v_m/2π)`, `q = k_in − k_out`, normalized
/// sinc (Q4 of the expert review). Scalar: no obliquity, material or flux
/// factor, and no shading by neighbours. It peaks where `q` is along the
/// facet's normal, the facet's specular direction. A reference for the
/// thin element's array-normal mask and pivot ramp ([`dmd_reflect`]), which
/// is a defined approximation, not this integral.
pub fn facet_integral(a: f64, u_m: Vec3, v_m: Vec3, k_in: Vec3, k_out: Vec3) -> Cx {
    let q = k_in - k_out;
    let tau = 2.0 * PI;
    Cx::real(a * a * sinc(a * q.dot(u_m) / tau) * sinc(a * q.dot(v_m) / tau))
}

/// What an isolated facet would intercept of a mirror's cell over what the
/// array-normal mask reflects: `|d_in·n_f|/(|d_in·z| cos θ)`, the facet's
/// flux `ρ²|d_in·n_f|/|d_in·z|` over the mask's `ρ² cos θ` (Q4 of the expert
/// review; `1/cos 24°`, 1.095, at 24° incidence on 12° facets in the plane of
/// tilt). A diagnostic, not a correction: neighbours shade an actual array.
pub fn facet_interception_ratio(d: &oc_core::modulator::DmdSpec, state: DmdState, d_in: Vec3) -> f64 {
    let nf = oc_core::modulator::facet_normal(d, state);
    d_in.dot(nf).abs() / (d_in.z.abs() * nf.z)
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
    // Without fringing the mask has a closed-form transform: the dead
    // space's g over the device (within the grid), and each electrode's
    // square times its (√R e^{iφ} − g).
    let fringed = w > 0.0 && beam == LcosBeam::Modulated;
    let folded = edge_share(u, device) > FOLD_TOLERANCE;
    let smoothed = fringed || folded;
    let mask = if smoothed {
        None
    } else {
        let a = l.pixel_fill * p;
        let (gx_lo, gx_hi) = (grid.coord(0, 0) - 0.5 * dx, grid.coord(0, grid.n[0] - 1) + 0.5 * dx);
        let (gy_lo, gy_hi) = (grid.coord(1, 0) - 0.5 * dy, grid.coord(1, grid.n[1] - 1) + 0.5 * dy);
        let (hx, hy) = (0.5 * nx as f64 * p, 0.5 * ny as f64 * p);
        let (rx0, rx1, ry0, ry1) = (gx_lo.max(-hx), gx_hi.min(hx), gy_lo.max(-hy), gy_hi.min(hy));
        let rect = move |f: [f64; 2]| -> Cx {
            if rx1 <= rx0 || ry1 <= ry0 {
                return Cx::ZERO;
            }
            let (wx, wy) = (rx1 - rx0, ry1 - ry0);
            let (cx, cy) = (0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1));
            Cx::expi(-2.0 * PI * (f[0] * cx + f[1] * cy)) * (wx * wy * sinc(wx * f[0]) * sinc(wy * f[1]))
        };
        let gap = move |f: [f64; 2]| rect(f) * g_gap;
        Some(band_limited_mask(
            &grid,
            device,
            &|f| Cx::real(a * a * sinc(a * f[0]) * sinc(a * f[1])),
            &|i, j| Cx::from_polar(l.reflectance.sqrt(), lcos_phase(device, l, i, j, lam, beam)) - g_gap,
            Some(&gap),
            &mut Fft2::new(),
        )?)
    };

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
            let v = u.data[idx] * mask.as_ref().map_or(acc, |m| m[idx]);
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
    if smoothed {
        // A sample spanning several phases carries less than the cell
        // reflects: the sub-cell projection error of the pixel edges.
        report.ledger.subcell_projection = beyond.max(0.0);
    } else {
        // What the device reflects beyond the grid's band.
        report.ledger.spectral_projection = beyond;
    }
    report.ledger.detected = out.power();
    report.ledger.close();
    validity_flags(&mut report, device, lam, u.carrier);
    if beam == LcosBeam::Modulated && (lam - l.design_lambda_um).abs() > 1e-12 {
        report.flag(ValidityFlag::PhaseDispersionApprox);
    }
    report.approx("LCoS as a thin element: dead space unmodulated, fringing as linear broadening");
    if fringed {
        report.approx("LCoS with fringing: its smoothed phases have no closed-form transform, so the mask is averaged over each sample's cell, which weights an order m by sinc(m/s) at s samples a pixel");
    } else if folded {
        report.approx("LCoS lit within a pixel of a window edge it reaches past: its mask is averaged over each sample's cell, which weights an order m by sinc(m/s) at s samples a pixel, since the band-limited one would fold the cut electrodes across the window");
    }
    report.steps.push(StepReport {
        method: Method::Element,
        z_mm: 0.0,
        sampling: None,
        note: format!(
            "LCoS {}×{} at {} µm, {:?} beam; its mask {}",
            nx,
            ny,
            device.pitch_um,
            beam,
            if fringed { "averaged over each sample's cell (fringing)" } else if folded { "averaged over each sample's cell (lit at a window edge)" } else { "band-limited exactly to the field's grid (each electrode's square, its closed-form transform, times the lattice sum, the dead space's g over the device)" }
        ),
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
