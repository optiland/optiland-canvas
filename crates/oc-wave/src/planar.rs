//! Planar layers on the wave path, plane wave by plane wave (I3).
//!
//! A planar interface keeps every plane wave's tangential wavevector, so each
//! plane wave of a field meets it on its own, at its own angle, and leaves
//! with a complex amplitude for its polarization. In the field's angular
//! spectrum that is a multiplication, exact for an interface or a layer of
//! infinite extent: no projection, no ray, no paraxial step. Two things are
//! done this way.
//!
//! - **A plane-parallel plate** (a body whose two faces on the path are flat
//!   and parallel, bare or with a residual AR) is one filter: the closed-form
//!   slab, `t₀₁t₁₂e^{iδ}/(1 − r₁₀r₁₂e^{2iδ})` with `δ = k₀n_g L cos θ_g`, sums
//!   the round trips between the faces, so an etalon's multiple-beam fringes
//!   come out of it, and the lateral shift a tilted plate gives the beam
//!   comes out of its phase's slope across the spectrum. Relative to free
//!   space across the same planes it is that amplitude times
//!   `e^{−ik₀n₀L cos θ₀}`. A tilted plate's round trips leave the beam a
//!   round-trip shift apart; those that would land beyond the window are
//!   summed out of the series and booked, never wrapped back in.
//! - **A flat coated surface** (a coated splitter, a dichroic) multiplies each
//!   plane wave by its coating's complex reflection or transmission, from the
//!   transfer matrix in `oc_core::thinfilm`, before the field is handed to the
//!   outgoing beam.
//!
//! **Amplitudes.** In the tangential chart and power coordinates of a
//! scattering two-port: `r = (y₁ − y₂)/(y₁ + y₂)`, `t = 2√(y₁y₂)/(y₁ + y₂)`,
//! with admittances `y_s = n cos θ`, `y_p = n / cos θ`. The reflection from
//! the far side is `−r`, so the two-port is unitary, `|r|² + |t|² = 1`, and
//! at normal incidence `r = (n₁ − n₂)/(n₁ + n₂)` for both polarizations, the
//! Stokes signs the splitters already use. A coating's ray-frame amplitudes
//! (`oc_core::thinfilm::FilmAmp`, whose `p̂` follows the reflected ray, so its
//! `r_p` is `−r_s` at normal incidence) are put into this chart by negating
//! `r_p`, and its transmitted amplitudes into power coordinates by
//! `√t_power_scale`.
//!
//! **Polarization.** A field is scalar. A chain that meets an element telling
//! s from p obliquely carries one of two modes, s or p of the plane of
//! incidence that element and the chief ray share; an unpolarized source is
//! the two modes, mutually incoherent, half its power each. Every plane wave
//! of the field takes its own angle's amplitude for that mode; that the plane
//! waves off the plane of incidence see s and p slightly rotated is left out,
//! and the report names it.

use crate::fft::{fftshift2, ifftshift2, next_fast_len};
use crate::path::Beam;
use crate::propagate::{crop, embed};
use crate::{Cx, Fft2, Field2, Grid2, WaveError};
use oc_core::thinfilm::{coating_amp_at, Coating, CoatingBacking, ExitFace, Incidence};
use oc_core::Vec3;
use std::f64::consts::PI;

/// The polarization a chain's field stands for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pol {
    /// No element on the chain tells s from p: the field stands for
    /// unpolarized light.
    Unpolarized,
    /// Linear, along `n`, the normal of the plane of incidence: s.
    S(Vec3),
    /// Linear, in that plane: p.
    P(Vec3),
}

impl Pol {
    /// `Some(true)` for p, `Some(false)` for s.
    pub fn is_p(&self) -> Option<bool> {
        match self {
            Pol::Unpolarized => None,
            Pol::S(_) => Some(false),
            Pol::P(_) => Some(true),
        }
    }
}

/// `cos θ₂` from `cos θ₁` by Snell's law, on the decaying branch past the
/// critical angle (`exp(−iωt)`: `κ = +i|κ|`).
fn cos2(n1: f64, n2: f64, c1: f64) -> Cx {
    let s2 = (n1 / n2).powi(2) * (1.0 - c1 * c1).max(0.0);
    if s2 <= 1.0 {
        Cx::real((1.0 - s2).sqrt())
    } else {
        Cx::new(0.0, (s2 - 1.0).sqrt())
    }
}

/// A bare interface's two-port for a plane wave meeting it from index `n1`
/// at `cos θ₁ = c1`, in power coordinates: `(r, t)` for s or p. Past the
/// critical angle `|r| = 1` with its phase and `t = 0`.
pub fn interface(n1: f64, n2: f64, c1: f64, p: bool) -> (Cx, Cx) {
    let c1 = c1.abs().clamp(1e-12, 1.0);
    let c2 = cos2(n1, n2, c1);
    let (y1, y2) = if p { (Cx::real(n1 / c1), Cx::real(n2) / c2) } else { (Cx::real(n1 * c1), c2 * n2) };
    let r = (y1 - y2) / (y1 + y2);
    if c2.im != 0.0 {
        return (r, Cx::ZERO);
    }
    (r, Cx::real(2.0 * (y1.re * y2.re).sqrt() / (y1.re + y2.re)))
}

/// The same interface with its reflectance scaled by `residual` (a residual
/// AR), its transmission what the scaling leaves, keeping it unitary.
fn interface_ar(n1: f64, n2: f64, c1: f64, p: bool, residual: f64) -> (Cx, Cx) {
    let (r, t) = interface(n1, n2, c1, p);
    if t == Cx::ZERO {
        return (r, t);
    }
    let r = r * residual.clamp(0.0, 1.0).sqrt();
    (r, Cx::real((1.0 - r.norm_sqr()).max(0.0).sqrt()))
}

/// A plane-parallel plate on the path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plate {
    /// The faces' unit normal, and the plate's thickness between them, mm.
    pub normal: Vec3,
    pub thickness_mm: f64,
    pub n_glass: f64,
    /// The medium on both sides.
    pub n_out: f64,
    /// Each face's reflectance relative to bare glass (`ar_residual`).
    pub residual: [f64; 2],
    /// The chief ray's optical path from the entry plane to the exit plane.
    pub chief_opl_mm: f64,
}

/// What a plate did besides the field.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlateStats {
    /// Reflected back toward the source, by the round trips' sum.
    pub reflected: f64,
    /// Round trips summed out of the series because they land beyond the
    /// window, and power the crop to the window took.
    pub beyond: f64,
    pub left_window: f64,
    pub evanescent: f64,
    /// The round trips kept, `None` for all (a plate square to the beam), and
    /// the lateral shift of one round trip, mm.
    pub round_trips: Option<usize>,
    pub round_trip_shift_mm: f64,
    /// The power the plate received, and what the sum of the parts misses
    /// (the overlap of round trips kept and summed out, which spatially
    /// separated beams do not have, and round-off).
    pub received: f64,
    pub rest: f64,
}

/// The plate's transmission relative to free space for a plane wave at
/// `cos θ₀ = c0` outside, for s or p: `(full, kept, dropped)`, the whole
/// series, the series to `keep` round trips, and the power of the rest.
fn slab(plate: &Plate, k0: f64, c0: f64, p: bool, keep: Option<usize>) -> (Cx, Cx, f64) {
    let (n0, ng, l) = (plate.n_out, plate.n_glass, plate.thickness_mm);
    let (r01, t01) = interface_ar(n0, ng, c0, p, plate.residual[0]);
    let cg = cos2(n0, ng, c0).re;
    let (r12, t12) = interface_ar(ng, n0, cg, p, plate.residual[1]);
    // Face 1 from inside reflects −r₀₁ (unitary two-port).
    let r10 = -r01;
    let delta = k0 * ng * l * cg;
    let x = r10 * r12 * Cx::expi(2.0 * delta);
    let lead = t01 * t12 * Cx::expi(delta);
    let full = lead / (Cx::ONE - x);
    let (kept, dropped) = match keep {
        None => (full, 0.0),
        Some(m) => {
            let mut xm = Cx::ONE;
            for _ in 0..=m {
                xm *= x;
            }
            let ax = x.norm_sqr();
            (lead * (Cx::ONE - xm) / (Cx::ONE - x), lead.norm_sqr() * xm.norm_sqr() / (1.0 - ax).max(1e-300))
        }
    };
    let free = Cx::expi(-k0 * n0 * l * c0);
    (full * free, kept * free, dropped)
}

/// Carry `u`, across `inc` on a plane before the plate, through it to the
/// plane across `out` behind it, the plate's effect on each plane wave the
/// closed-form slab relative to free space, the free space between the two
/// planes the exact angular spectrum, and the field shifted onto `out`'s
/// origin (the chief ray's exit). Padded twice.
pub(crate) fn plate_step(u: &Field2, inc: &Beam, out: &Beam, plate: &Plate, pol: Pol, fft: &mut Fft2) -> Result<(Field2, PlateStats), WaveError> {
    let n = u.grid.n;
    let p = [next_fast_len(2 * n[0]), next_fast_len(2 * n[1])];
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let mut buf = embed(u, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    let lam0 = u.wavelength_mm();
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let k0 = 2.0 * PI / lam0;
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let df = du * dv;
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let dist = (out.origin - inc.origin).dot(inc.d);
    let lateral = out.origin - (inc.origin + inc.d * dist);
    let (sx, sy) = (lateral.dot(inc.x), lateral.dot(inc.y));
    // One round trip's shift across the beam, for the chief direction.
    let c_chief = inc.d.dot(plate.normal).abs().clamp(1e-12, 1.0);
    let cg_chief = cos2(plate.n_out, plate.n_glass, c_chief).re.max(1e-12);
    let sin0 = (1.0 - c_chief * c_chief).max(0.0).sqrt();
    let shift = 2.0 * plate.thickness_mm * (plate.n_out / plate.n_glass) * sin0 / cg_chief * c_chief;
    let window = u.grid.width_mm(0).min(u.grid.width_mm(1));
    let keep = (shift > 1e-9 * window).then(|| (0.4 * window / shift).floor() as usize);
    let mut st = PlateStats { round_trips: keep, round_trip_shift_mm: shift, received: u.power(), ..Default::default() };
    for ky in 0..p[1] {
        let fy = gp.fft_freq(1, ky) + v0;
        for kx in 0..p[0] {
            let fx = gp.fft_freq(0, kx) + u0;
            let a = &mut buf[ky * p[0] + kx];
            let w = inv_l2 - fx * fx - fy * fy;
            let e = a.norm_sqr() * df;
            if w <= 0.0 {
                st.evanescent += e;
                *a = Cx::ZERO;
                continue;
            }
            let dir = inc.x * (lam * fx) + inc.y * (lam * fy) + inc.d * (lam * w.sqrt());
            let c0 = dir.dot(plate.normal).abs();
            let (full, kept, dropped) = match pol.is_p() {
                Some(pp) => slab(plate, k0, c0, pp, keep),
                None => {
                    // Square to the beam s and p are one; off it an
                    // unpolarized field takes their mean (reported).
                    let (fs, ks, ds) = slab(plate, k0, c0, false, keep);
                    let (fp, kp, dp) = slab(plate, k0, c0, true, keep);
                    ((fs + fp) * 0.5, (ks + kp) * 0.5, 0.5 * (ds + dp))
                }
            };
            st.reflected += (1.0 - full.norm_sqr()).max(0.0) * e;
            st.beyond += dropped * e;
            *a *= kept * Cx::expi(2.0 * PI * (dist * w.sqrt() + fx * sx + fy * sy));
        }
    }
    fft.inverse(&mut buf, p, df);
    fftshift2(&mut buf, p);
    let whole: f64 = buf.iter().map(|z| z.norm_sqr()).sum::<f64>() * gp.cell_area_mm2();
    let data = crop(&buf, p, n);
    let mut v = u.clone();
    v.data = data;
    v.frame = out.iso();
    v.opl_ref_mm = u.opl_ref_mm + plate.chief_opl_mm;
    let kept_power = v.power();
    st.left_window = (whole - kept_power).max(0.0);
    st.rest = st.received - st.reflected - st.beyond - st.left_window - st.evanescent - kept_power;
    Ok((v, st))
}

/// Multiply each plane wave of `u` (across `inc`) by what an element in the
/// plane with normal `n_e` does to it: `f(cos θ)` gives the amplitude on the
/// chain's arm and the shares of the power that go into the other arm and
/// into the element. Returns the field and those two powers, and what left
/// the window. Padded twice; nothing propagates.
pub(crate) fn surface_filter(
    u: &Field2,
    inc: &Beam,
    n_e: Vec3,
    f: &dyn Fn(f64) -> (Cx, f64, f64),
    fft: &mut Fft2,
) -> Result<(Field2, f64, f64, f64), WaveError> {
    let n = u.grid.n;
    let p = [next_fast_len(2 * n[0]), next_fast_len(2 * n[1])];
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let mut buf = embed(u, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let df = du * dv;
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    let (mut other, mut absorbed) = (0.0, 0.0);
    for ky in 0..p[1] {
        let fy = gp.fft_freq(1, ky) + v0;
        for kx in 0..p[0] {
            let fx = gp.fft_freq(0, kx) + u0;
            let a = &mut buf[ky * p[0] + kx];
            let w = (inv_l2 - fx * fx - fy * fy).max(0.0);
            let dir = inc.x * (lam * fx) + inc.y * (lam * fy) + inc.d * (lam * w.sqrt());
            let (c, o, ab) = f(dir.dot(n_e).abs());
            let e = a.norm_sqr() * df;
            other += o * e;
            absorbed += ab * e;
            *a *= c;
        }
    }
    fft.inverse(&mut buf, p, df);
    fftshift2(&mut buf, p);
    let whole: f64 = buf.iter().map(|z| z.norm_sqr()).sum::<f64>() * gp.cell_area_mm2();
    let mut v = u.clone();
    v.data = crop(&buf, p, n);
    let left = (whole - v.power()).max(0.0);
    Ok((v, other, absorbed, left))
}

/// A flat coating's amplitude on one arm, for a plane wave meeting it from
/// index `n0` at `cos θ₀ = c0`, in the chart above: the amplitude, the share
/// of the power that goes into the other arm (and, for a plate-backed
/// coating, into the plate's own round trips), and the share absorbed.
///
/// A coating on a plate (`CoatingBacking::Plate`) is, as the scene declares
/// it, a plate far thicker than any coherence length the canvas models: its
/// stack is coherent, its two faces are not, so the transmitted amplitude is
/// the stack's into the glass times the back face's out of it, and the light
/// the faces send round the plate is booked, not added.
pub fn coating_arm(c: &Coating, reflect: bool, n0: f64, lambda_um: f64, c0: f64, p: bool) -> Result<(Cx, f64, f64), WaveError> {
    let fe = |e: oc_core::thinfilm::FilmError| WaveError::OutOfRange(format!("the coating cannot be evaluated: {e:?}"));
    let chart = |a: &oc_core::thinfilm::FilmAmp| -> (Cx, Cx) {
        let s = a.t_power_scale.max(0.0).sqrt();
        if p {
            (-a.r_p, a.t_p * s)
        } else {
            (a.r_s, a.t_s * s)
        }
    };
    match &c.backing {
        CoatingBacking::Medium | CoatingBacking::Opaque { .. } => {
            let a = coating_amp_at(c, n0, n0, lambda_um, c0).map_err(fe)?;
            let (r, t) = chart(&a);
            let (rr, tt) = (r.norm_sqr(), t.norm_sqr());
            let lost = (1.0 - rr - tt).max(0.0);
            Ok(if reflect { (r, tt, lost) } else { (t, rr, lost) })
        }
        CoatingBacking::Plate { substrate, exit, .. } => {
            let ns = substrate.index_at(lambda_um).map(|z| z.re).unwrap_or(1.5).max(1e-9);
            let film = Coating { backing: CoatingBacking::Medium, ..c.clone() };
            let residual = match exit {
                ExitFace::Bare => 1.0,
                ExitFace::AntiReflective { residual } => *residual,
            };
            match c.incidence {
                Incidence::Film => {
                    // The stack from outside into the glass, then the back face.
                    let a = coating_amp_at(&film, n0, ns, lambda_um, c0).map_err(fe)?;
                    let (rf, tf) = chart(&a);
                    let cs = cos2(n0, ns, c0).re.max(1e-12);
                    let (rb, tb) = interface_ar(ns, n0, cs, p, residual);
                    let (rr, tt) = (rf.norm_sqr(), tf.norm_sqr());
                    let lost = (1.0 - rr - tt).max(0.0);
                    if reflect {
                        Ok((rf, tt, lost))
                    } else {
                        Ok((tf * tb, rr + tt * rb.norm_sqr(), lost))
                    }
                }
                Incidence::Substrate => {
                    // The bare back face into the glass, then the stack from the
                    // glass side, then (reflected) the back face again.
                    let (rb, tb) = interface_ar(n0, ns, c0, p, residual);
                    let cs = cos2(n0, ns, c0).re.max(1e-12);
                    let a = coating_amp_at(&film, ns, n0, lambda_um, cs).map_err(fe)?;
                    let (rf, tf) = chart(&a);
                    let (rr, tt) = (rf.norm_sqr(), tf.norm_sqr());
                    let lost = (1.0 - rr - tt).max(0.0) * tb.norm_sqr();
                    let tbb = tb.norm_sqr();
                    if reflect {
                        Ok((tb * rf * tb, rb.norm_sqr() + tbb * tt + tbb * rr * (1.0 - tbb), lost))
                    } else {
                        Ok((tb * tf, rb.norm_sqr() + tbb * rr, lost))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_interface_is_a_unitary_two_port_with_the_textbook_numbers() {
        // Bundle B03: n 1 → 1.5 at normal incidence, r = −0.2, R + T = 1; at
        // Brewster's angle p is not reflected; the far side reflects −r.
        let (r, t) = interface(1.0, 1.5, 1.0, false);
        assert!((r - Cx::real(-0.2)).abs() < 1e-15 && (t.norm_sqr() - 0.96).abs() < 1e-15);
        let (rp, _) = interface(1.0, 1.5, 1.0, true);
        assert!((rp - r).abs() < 1e-15, "s and p are one at normal incidence");
        let b = 1.5f64.atan();
        let (rb, tb) = interface(1.0, 1.5, b.cos(), true);
        assert!(rb.abs() < 1e-15 && (tb.norm_sqr() - 1.0).abs() < 1e-14);
        for c in [1.0, 0.8, 0.5, 0.2] {
            for p in [false, true] {
                let (r1, t1) = interface(1.0, 1.5, c, p);
                let c2 = cos2(1.0, 1.5, c).re;
                let (r2, t2) = interface(1.5, 1.0, c2, p);
                assert!((r1.norm_sqr() + t1.norm_sqr() - 1.0).abs() < 1e-14);
                assert!((r2 + r1).abs() < 1e-14, "Stokes: r' = −r");
                assert!((t1 - t2).abs() < 1e-14, "in power coordinates t' = t");
            }
        }
        // Past the critical angle, glass to air: total, with a phase.
        let (r, t) = interface(1.5, 1.0, 50f64.to_radians().cos(), false);
        assert!((r.abs() - 1.0).abs() < 1e-14 && t == Cx::ZERO && r.arg().abs() > 0.1);
    }

    #[test]
    fn the_slab_matches_the_closed_form_and_its_series() {
        // Bundle M16: T = (1 − r²)²/((1 − r²)² + 4r² sin²δ), R + T = 1, and at
        // δ = 7π the slab is transparent with t = (−1)⁷.
        let k0 = 2.0 * PI / 0.6328e-3;
        let l7 = 7.0 * PI / (k0 * 1.5);
        let plate = Plate { normal: Vec3::Z, thickness_mm: l7, n_glass: 1.5, n_out: 1.0, residual: [1.0, 1.0], chief_opl_mm: 0.0 };
        let (full, kept, dropped) = slab(&plate, k0, 1.0, false, None);
        let t = full * Cx::expi(k0 * l7);
        assert!((t - Cx::real(-1.0)).abs() < 1e-12, "{t:?}");
        assert!(kept == full && dropped == 0.0);
        let plate = Plate { thickness_mm: 7.5 * PI / (k0 * 1.5), ..plate };
        let r2: f64 = 0.04;
        let (full, ..) = slab(&plate, k0, 1.0, false, None);
        let want = (1.0 - r2).powi(2) / ((1.0 - r2).powi(2) + 4.0 * r2);
        assert!((full.norm_sqr() - want).abs() < 1e-14, "{} vs {want}", full.norm_sqr());
        // The series to one round trip, at δ = 7.5π where e^{2iδ} = −1 and the
        // round-trip factor is x = −r²: t₀₁t₁₂(1 + x), and the rest of the
        // series carries |t₀₁t₁₂|²|x|⁴/(1 − |x|²) in power.
        let (_, kept, dropped) = slab(&plate, k0, 1.0, false, Some(1));
        assert!((kept.norm_sqr() - (1.0 - r2).powi(4)).abs() < 1e-14, "{}", kept.norm_sqr());
        let rest = (1.0 - r2).powi(2) * r2.powi(4) / (1.0 - r2 * r2);
        assert!((dropped - rest).abs() < 1e-16, "{dropped} vs {rest}");
    }
}
