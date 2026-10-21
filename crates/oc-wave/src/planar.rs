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
//!   summed out of the series and booked, never wrapped back in. What that
//!   takes from the observable is booked exactly, plane wave by plane wave,
//!   `|full|² − |kept|²`: the omitted round trips as a field of their own,
//!   `lead·x^{M+1}/(1 − x)`, whose power has `|1 − x|²` below (not
//!   `1 − |x|²`, which would add the round trips' intensities), and their
//!   interference with the round trips kept, at most twice the product of
//!   the two norms (Q11 of the expert review). A plate of lower index than
//!   its surroundings reflects whole what meets it past the critical angle;
//!   the evanescent wave across it would tunnel (frustrated total internal
//!   reflection), which is not modelled: the step refuses a gap thinner than
//!   five decay lengths, and names the leak it leaves out otherwise.
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
//! **Polarization.** The path hands these steps one polarization at a time,
//! s or p of the plane of incidence the element and the chief ray share: a
//! chain whose elements tell s from p carries two components, turned into
//! each element's s and p before it (`crate::path`). Every plane wave of the
//! field takes its own angle's amplitude for that polarization; that the
//! plane waves off the plane of incidence see s and p slightly rotated is
//! left out, and the report names it.

use crate::fft::{fftshift2, ifftshift2, next_fast_len};
use crate::path::Beam;
use crate::propagate::{crop, embed};
use crate::{Cx, Fft2, Field2, Grid2, WaveError};
use oc_core::thinfilm::{coating_amp_at, Coating, CoatingBacking, ExitFace, Incidence, PlateRoundTrips};
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
    /// The one component an element acts on, already projected out of a
    /// field of two (an LCoS's director component).
    Projected,
}

impl Pol {
    /// `Some(true)` for p, `Some(false)` for s.
    pub fn is_p(&self) -> Option<bool> {
        match self {
            Pol::Unpolarized | Pol::Projected => None,
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
    /// (round-off).
    pub received: f64,
    pub rest: f64,
    /// The round trips summed out as a field of their own (`‖tail‖²`), and
    /// their interference with those kept (`2 Re⟨kept, tail⟩`): `beyond` is
    /// their sum, what the truncation takes from the observable.
    pub tail: f64,
    pub cross: f64,
    /// The power that met the faces past the critical angle and was
    /// reflected whole, and the thinnest the gap was for it, in decay lengths
    /// of its evanescent wave.
    pub tir: f64,
    pub tir_decays: f64,
}

/// The plate's transmission relative to free space for a plane wave at
/// `cos θ₀ = c0` outside, for s or p: `(full, kept)`, the whole series and
/// the series to `keep` round trips; their difference is the omitted tail,
/// `lead·x^{M+1}/(1 − x)`.
fn slab(plate: &Plate, k0: f64, c0: f64, p: bool, keep: Option<usize>) -> (Cx, Cx) {
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
    let kept = match keep {
        None => full,
        Some(m) => {
            let mut xm = Cx::ONE;
            for _ in 0..=m {
                xm *= x;
            }
            lead * (Cx::ONE - xm) / (Cx::ONE - x)
        }
    };
    let free = Cx::expi(-k0 * n0 * l * c0);
    (full * free, kept * free)
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
    let mut st = PlateStats { round_trips: keep, round_trip_shift_mm: shift, received: u.power(), tir_decays: f64::INFINITY, ..Default::default() };
    let (n0, ng) = (plate.n_out, plate.n_glass);
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
            // Past the critical angle of a lower-index plate: reflected
            // whole, the tunnelling across the gap left out.
            let s2 = (n0 / ng).powi(2) * (1.0 - c0 * c0);
            if s2 > 1.0 && e > 0.0 {
                st.tir += e;
                if e > 1e-9 * st.received {
                    let kappa = k0 * ng * (s2 - 1.0).sqrt();
                    st.tir_decays = st.tir_decays.min(kappa * plate.thickness_mm);
                }
            }
            let (full, kept) = match pol.is_p() {
                Some(pp) => slab(plate, k0, c0, pp, keep),
                None => {
                    // Square to the beam s and p are one; off it an
                    // unpolarized field takes their mean (reported).
                    let (fs, ks) = slab(plate, k0, c0, false, keep);
                    let (fp, kp) = slab(plate, k0, c0, true, keep);
                    ((fs + fp) * 0.5, (ks + kp) * 0.5)
                }
            };
            let tail = full - kept;
            st.reflected += (1.0 - full.norm_sqr()).max(0.0) * e;
            st.beyond += (full.norm_sqr() - kept.norm_sqr()) * e;
            st.tail += tail.norm_sqr() * e;
            st.cross += 2.0 * (kept * tail.conj()).re * e;
            *a *= kept * Cx::expi(2.0 * PI * (dist * w.sqrt() + fx * sx + fy * sy));
        }
    }
    if st.tir > 1e-9 * st.received && st.tir_decays < 5.0 {
        return Err(WaveError::OutOfRange(format!(
            "a plate of index {ng} in a medium of {n0}, {:.4e} mm thick: {:.3e} of the field meets it past its critical angle, and the gap is {:.2} decay lengths of the evanescent wave thick, across which the light would tunnel; frustrated total internal reflection is not modelled",
            plate.thickness_mm,
            st.tir / st.received.max(f64::MIN_POSITIVE),
            st.tir_decays
        )));
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

/// A coated plate's two faces as a plane wave meets them, in the chart
/// above: the first face from outside (`r1`, `t1`) and from inside the glass
/// (`r1i`, `t1i`), the second from inside (`r2i`, and `t2` out of it), the
/// glass's index and thickness. Smooth in the angle, so it can be tabulated;
/// the plate's own phase is not.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlateFaces {
    pub r1: Cx,
    pub t1: Cx,
    pub r1i: Cx,
    pub t1i: Cx,
    pub r2i: Cx,
    pub t2: Cx,
    pub n_glass: f64,
    pub thickness_mm: f64,
}

impl PlateFaces {
    /// `a·(1 − f) + b·f`, field by field.
    pub(crate) fn lerp(a: &PlateFaces, b: &PlateFaces, f: f64) -> PlateFaces {
        let l = |x: Cx, y: Cx| x * (1.0 - f) + y * f;
        PlateFaces { r1: l(a.r1, b.r1), t1: l(a.t1, b.t1), r1i: l(a.r1i, b.r1i), t1i: l(a.t1i, b.t1i), r2i: l(a.r2i, b.r2i), t2: l(a.t2, b.t2), ..*a }
    }

    /// The round-trip factor `x = r₁′r₂′e^{2iδ}` and the plate's phase
    /// `δ = k₀n_g d cos θ_g` for a plane wave at `cos θ₀ = c0` in `n0`.
    fn trip(&self, k0: f64, n0: f64, c0: f64) -> (Cx, f64) {
        let cg = cos2(n0, self.n_glass, c0).re;
        let delta = k0 * self.n_glass * self.thickness_mm * cg;
        (self.r1i * self.r2i * Cx::expi(2.0 * delta), delta)
    }
}

/// The faces of `c`'s plate, `None` for a coating that is not plate-backed.
pub(crate) fn coating_plate_faces(c: &Coating, n0: f64, lambda_um: f64, c0: f64, p: bool) -> Result<Option<PlateFaces>, WaveError> {
    let fe = |e: oc_core::thinfilm::FilmError| WaveError::OutOfRange(format!("the coating cannot be evaluated: {e:?}"));
    let CoatingBacking::Plate { substrate, exit, thickness_mm, .. } = &c.backing else { return Ok(None) };
    let chart = |a: &oc_core::thinfilm::FilmAmp| -> (Cx, Cx) {
        let s = a.t_power_scale.max(0.0).sqrt();
        if p {
            (-a.r_p, a.t_p * s)
        } else {
            (a.r_s, a.t_s * s)
        }
    };
    let ns = substrate.index_at(lambda_um).map(|z| z.re).unwrap_or(1.5).max(1e-9);
    let residual = match exit {
        ExitFace::Bare => 1.0,
        ExitFace::AntiReflective { residual } => *residual,
    };
    let cs = cos2(n0, ns, c0).re.max(1e-12);
    // The stack lit from outside its film, and from the glass.
    let from_out = Coating { backing: CoatingBacking::Medium, incidence: Incidence::Film, ..c.clone() };
    let from_glass = Coating { backing: CoatingBacking::Medium, incidence: Incidence::Substrate, ..c.clone() };
    let faces = match c.incidence {
        Incidence::Film => {
            let (r1, t1) = chart(&coating_amp_at(&from_out, n0, ns, lambda_um, c0).map_err(fe)?);
            let (r1i, t1i) = chart(&coating_amp_at(&from_glass, ns, n0, lambda_um, cs).map_err(fe)?);
            let (r2i, t2) = interface_ar(ns, n0, cs, p, residual);
            PlateFaces { r1, t1, r1i, t1i, r2i, t2, n_glass: ns, thickness_mm: *thickness_mm }
        }
        Incidence::Substrate => {
            // The bare (or AR) face first; from inside it reflects −r.
            let (r1, t1) = interface_ar(n0, ns, c0, p, residual);
            let (r2i, t2) = chart(&coating_amp_at(&from_glass, ns, n0, lambda_um, cs).map_err(fe)?);
            PlateFaces { r1, t1, r1i: -r1, t1i: t1, r2i, t2, n_glass: ns, thickness_mm: *thickness_mm }
        }
    };
    Ok(Some(faces))
}

/// A coated plate's amplitude on one arm, with the shares of the power that
/// go into the other arm and into absorption, its round trips summed as
/// `mode` declares: as fields (`Coherent`), the transmission
/// `t₁t₂e^{iδ}/(1 − x)` relative to free space across the plate
/// (`e^{−ik₀n₀d cos θ₀}`) and the reflection `r₁ + t₁t₁′r₂′e^{2iδ}/(1 − x)`;
/// or in power (`Incoherent`), `T = T₁T₂/(1 − R₁′R₂′)` and
/// `R = R₁ + T₁T₁′R₂′/(1 − R₁′R₂′)`, the first pass's amplitude carrying
/// them (Katsidis & Siapkas 2002). `FirstPass` is [`coating_arm`]'s.
pub(crate) fn plate_series(f: &PlateFaces, mode: PlateRoundTrips, reflect: bool, k0: f64, n0: f64, c0: f64) -> (Cx, f64, f64) {
    let (r, t) = match mode {
        PlateRoundTrips::Coherent => {
            let (x, delta) = f.trip(k0, n0, c0);
            let e1 = Cx::expi(delta);
            let free = Cx::expi(-k0 * n0 * f.thickness_mm * c0);
            let t = f.t1 * f.t2 * e1 / (Cx::ONE - x) * free;
            let r = f.r1 + f.t1 * f.t1i * f.r2i * e1 * e1 / (Cx::ONE - x);
            (r, t)
        }
        PlateRoundTrips::Incoherent | PlateRoundTrips::FirstPass => {
            let den = (1.0 - f.r1i.norm_sqr() * f.r2i.norm_sqr()).max(1e-300);
            let tp = f.t1.norm_sqr() * f.t2.norm_sqr() / den;
            let rp = f.r1.norm_sqr() + f.t1.norm_sqr() * f.t1i.norm_sqr() * f.r2i.norm_sqr() / den;
            let scale = |a: Cx, power: f64| if a.abs() > 0.0 { a * (power.sqrt() / a.abs()) } else { Cx::real(power.sqrt()) };
            (scale(f.r1, rp), scale(f.t1 * f.t2, tp))
        }
    };
    let (rr, tt) = (r.norm_sqr(), t.norm_sqr());
    let lost = (1.0 - rr - tt).max(0.0);
    if reflect {
        (r, tt, lost)
    } else {
        (t, rr, lost)
    }
}

/// A flat coating's amplitude on one arm, for a plane wave meeting it from
/// index `n0` at `cos θ₀ = c0`, in the chart above: the amplitude, the share
/// of the power that goes into the other arm (and, for a plate-backed
/// coating, into the plate's own round trips), and the share absorbed.
///
/// A coating on a plate (`CoatingBacking::Plate`) takes its round trips as
/// the plate declares ([`PlateRoundTrips`]): with the first pass alone, the
/// default, the transmitted amplitude is the stack's into the glass times
/// the back face's out of it, and the light the faces send round the plate
/// is booked, not added; added in power or summed as fields, by
/// [`plate_series`].
pub fn coating_arm(c: &Coating, reflect: bool, n0: f64, lambda_um: f64, c0: f64, p: bool) -> Result<(Cx, f64, f64), WaveError> {
    if let CoatingBacking::Plate { round_trips, .. } = &c.backing {
        if *round_trips != PlateRoundTrips::FirstPass {
            let f = coating_plate_faces(c, n0, lambda_um, c0, p)?.expect("a plate-backed coating");
            return Ok(plate_series(&f, *round_trips, reflect, 2.0 * PI / (lambda_um / 1000.0), n0, c0));
        }
    }
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
        let (full, kept) = slab(&plate, k0, 1.0, false, None);
        let t = full * Cx::expi(k0 * l7);
        assert!((t - Cx::real(-1.0)).abs() < 1e-12, "{t:?}");
        assert!(kept == full);
        let plate = Plate { thickness_mm: 7.5 * PI / (k0 * 1.5), ..plate };
        let r2: f64 = 0.04;
        let (full, _) = slab(&plate, k0, 1.0, false, None);
        let want = (1.0 - r2).powi(2) / ((1.0 - r2).powi(2) + 4.0 * r2);
        assert!((full.norm_sqr() - want).abs() < 1e-14, "{} vs {want}", full.norm_sqr());
        // The series to one round trip, at δ = 7.5π where e^{2iδ} = −1 and the
        // round-trip factor is x = −r²: t₀₁t₁₂(1 + x), and the omitted tail is
        // t₀₁t₁₂x²/(1 − x), a field of power |t₀₁t₁₂|²|x|⁴/|1 − x|² (Q11 of
        // the expert review: not |x|⁴/(1 − |x|²), the round trips'
        // intensities added).
        let (full, kept) = slab(&plate, k0, 1.0, false, Some(1));
        assert!((kept.norm_sqr() - (1.0 - r2).powi(4)).abs() < 1e-14, "{}", kept.norm_sqr());
        let tail = (full - kept).norm_sqr();
        let want = (1.0 - r2).powi(2) * r2.powi(4) / (1.0 + r2).powi(2);
        assert!((tail - want).abs() < 1e-16, "{tail} vs {want}");
    }

    use crate::path::Beam;
    use oc_core::thinfilm::{coating_rt_at, FilmLayer, FilmMaterial};

    fn beam_z(z: f64) -> Beam {
        Beam { origin: Vec3::new(0.0, 0.0, z), x: Vec3::X, y: Vec3::Y, d: Vec3::Z }
    }

    #[test]
    fn a_plate_s_kept_round_trips_are_the_beams_summed_one_by_one() {
        // R20 of the expert review. A plate of index 4 (|x| = 0.36 a round
        // trip), 1 mm thick, tilted 30°: its round trips leave the beam a
        // round-trip shift apart, `2d tan θ_g cos θ₀` (Born & Wolf, §7.6).
        // The step keeps those within 0.4 of the window. Summed one by one,
        // each its own field `lead·x^m` per plane wave, they make the field
        // the step returns; the first lies the round-trip shift from the
        // direct beam; and what the truncation takes from the observable,
        // `|full|² − |kept|²` plane wave by plane wave, is the omitted tail's
        // own power and its interference with the kept round trips, the
        // latter bounded by twice the product of their norms.
        let k0 = 2.0 * PI / 0.6328e-3;
        let th = 30f64.to_radians();
        let plate = Plate { normal: Vec3::new(th.sin(), 0.0, th.cos()), thickness_mm: 1.0, n_glass: 4.0, n_out: 1.0, residual: [1.0, 1.0], chief_opl_mm: 0.0 };
        let g = Grid2::square(256, 4.0 / 256.0).unwrap();
        let u = crate::beams::GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
        let (inc, out) = (beam_z(0.0), beam_z(2.0));
        let mut fft = Fft2::new();
        let (v, st) = plate_step(&u, &inc, &out, &plate, Pol::S(Vec3::Y), &mut fft).unwrap();
        let m = st.round_trips.expect("a tilted plate keeps a few round trips");
        // The beams one by one: the series' terms, plane wave by plane wave.
        let n = g.n;
        let pp = [next_fast_len(2 * n[0]), next_fast_len(2 * n[1])];
        let gp = Grid2::new(pp, g.pitch_mm, g.center_mm).unwrap();
        let mut spec = embed(&u, pp);
        ifftshift2(&mut spec, pp);
        fft.forward(&mut spec, pp, gp.cell_area_mm2());
        let lam = u.wavelength_mm();
        let beam = |terms: &[usize], fft: &mut Fft2| -> Field2 {
            let mut b = spec.clone();
            for ky in 0..pp[1] {
                let fy = gp.fft_freq(1, ky);
                for kx in 0..pp[0] {
                    let fx = gp.fft_freq(0, kx);
                    let w = 1.0 / (lam * lam) - fx * fx - fy * fy;
                    let a = &mut b[ky * pp[0] + kx];
                    if w <= 0.0 {
                        *a = Cx::ZERO;
                        continue;
                    }
                    let dir = Vec3::new(lam * fx, lam * fy, lam * w.sqrt());
                    let c0 = dir.dot(plate.normal).abs();
                    let (r01, t01) = interface(1.0, 4.0, c0, false);
                    let cg = cos2(1.0, 4.0, c0).re;
                    let (r12, t12) = interface(4.0, 1.0, cg, false);
                    let delta = k0 * 4.0 * cg;
                    let x = -r01 * r12 * Cx::expi(2.0 * delta);
                    let lead = t01 * t12 * Cx::expi(delta) * Cx::expi(-k0 * c0);
                    let mut s = Cx::ZERO;
                    for &j in terms {
                        let mut xm = Cx::ONE;
                        for _ in 0..j {
                            xm *= x;
                        }
                        s += lead * xm;
                    }
                    *a *= s * Cx::expi(2.0 * PI * 2.0 * w.sqrt());
                }
            }
            fft.inverse(&mut b, pp, gp.freq_pitch(0) * gp.freq_pitch(1));
            fftshift2(&mut b, pp);
            let mut f = u.clone();
            f.data = crop(&b, pp, n);
            f
        };
        let all: Vec<usize> = (0..=m).collect();
        let summed = beam(&all, &mut fft);
        let d: f64 = summed.data.iter().zip(&v.data).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>();
        let rel = (d / v.data.iter().map(|z| z.norm_sqr()).sum::<f64>()).sqrt();
        let centroid = |f: &Field2| {
            let (mut s, mut sx) = (0.0, 0.0);
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let w = f.at(i, j).norm_sqr();
                    s += w;
                    sx += w * g.xy(i, j).0;
                }
            }
            sx / s
        };
        let (b0, b1) = (beam(&[0], &mut fft), beam(&[1], &mut fft));
        let walk = (centroid(&b1) - centroid(&b0)).abs();
        let want = 2.0 * 1.0 * (1.0 - (th.sin() / 4.0).powi(2)).sqrt().recip() * (th.sin() / 4.0) * th.cos();
        let bound = 2.0 * ((st.received - st.reflected - st.beyond).max(0.0) * st.tail).sqrt();
        println!("measured: plate of index 4 at 30°, {m} round trips kept: the step against the beams summed one by one, relative = {rel:.3e}; the first round trip {walk:.6} mm from the direct beam against 2d·tan θ_g·cos θ₀ = {want:.6}; tail {:.3e}, interference {:+.3e} (bound {bound:.3e}), rest {:.1e}, of the received", st.tail / st.received, st.cross / st.received, st.rest / st.received);
        assert!(rel < 1e-12, "{rel}");
        assert!((walk / want - 1.0).abs() < 1e-3, "{walk} vs {want}");
        assert!((st.beyond - st.tail - st.cross).abs() < 1e-12 * st.received, "{st:?}");
        assert!(st.cross.abs() <= bound + 1e-15, "{st:?}");
        assert!(st.rest.abs() < 1e-10 * st.received, "{st:?}");
        assert!((st.round_trip_shift_mm / want - 1.0).abs() < 1e-12, "{} vs {want}", st.round_trip_shift_mm);
    }

    #[test]
    fn a_thin_low_index_gap_met_past_its_critical_angle_is_refused() {
        // Frustrated total internal reflection (Born & Wolf, §1.5.4): a gap of
        // index 1 in glass of 1.5, met at 45°, past its critical angle of
        // 41.8°. The evanescent wave decays over 1/κ, κ = k₀√(n₀² sin²θ − n²),
        // 0.28 µm here; across 1 µm (3.5 decay lengths) it would tunnel, and
        // the step refuses by name; across 10 µm (35) it would tunnel e^{−70},
        // and the step names the leak it leaves out.
        let k0 = 2.0 * PI / 0.6328e-3;
        let th = 45f64.to_radians();
        let g = Grid2::square(128, 2.0 / 128.0).unwrap();
        let u = crate::beams::GaussianBeam { w0_mm: 0.3, lambda_um: 0.6328, n: 1.5, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
        let mut fft = Fft2::new();
        let kappa = k0 * (1.5f64.powi(2) * th.sin().powi(2) - 1.0).sqrt();
        for (d, refused) in [(1e-3, true), (1e-2, false)] {
            let plate = Plate { normal: Vec3::new(th.sin(), 0.0, th.cos()), thickness_mm: d, n_glass: 1.0, n_out: 1.5, residual: [1.0, 1.0], chief_opl_mm: 0.0 };
            let r = plate_step(&u, &beam_z(0.0), &beam_z(0.1), &plate, Pol::S(Vec3::Y), &mut fft);
            match r {
                Err(e) => {
                    assert!(refused, "{e}");
                    assert!(e.to_string().contains("frustrated total internal reflection is not modelled"), "{e}");
                }
                Ok((_, st)) => {
                    assert!(!refused, "a 1 µm gap must be refused");
                    println!("measured: a 10 µm gap past its critical angle: {:.3e} of the field reflected whole, the gap {:.2} decay lengths (κd = {:.2})", st.tir / st.received, st.tir_decays, kappa * d);
                    assert!(st.tir > 0.99 * st.received && st.tir_decays > 5.0 && st.tir_decays <= kappa * d * (1.0 + 1e-9), "{st:?}");
                }
            }
        }
    }

    /// A quarter-wave MgF₂ layer (1.38) on 0.5 mm of crown glass (1.52), lit
    /// from its film side or through the glass, its round trips as `mode`.
    fn coated_plate(incidence: Incidence, mode: PlateRoundTrips, exit: ExitFace) -> Coating {
        Coating {
            layers: vec![FilmLayer::new(FilmMaterial::constant("film", 1.38), 632.8 / (4.0 * 1.38))],
            backing: CoatingBacking::Plate { substrate: FilmMaterial::constant("glass", 1.52), exit, thickness_mm: 0.5, round_trips: mode },
            incidence,
            design: None,
        }
    }

    #[test]
    fn a_coated_plate_s_round_trips_meet_their_three_references() {
        // R21 of the expert review and finding 7. Summed as fields, the
        // plate's round trips are the transfer matrix of the whole sandwich,
        // the film and 0.5 mm of glass as layers between air and air (Byrnes,
        // arXiv:1603.02720; Katsidis & Siapkas 2002), relative to free space
        // across the plate; added in power, they are the ray side's
        // incoherent plate (`coating_rt_at`, the series `1 − R₁′R₂′` below);
        // the first pass alone is `T₁T₂`. At 0°, 30° and 60°, s and p, lit
        // from the film side and through the glass.
        let lam = 0.6328;
        let k0 = 2.0 * PI / (lam / 1000.0);
        let film = || FilmLayer::new(FilmMaterial::constant("film", 1.38), 632.8 / (4.0 * 1.38));
        let glass = || FilmLayer::new(FilmMaterial::constant("glass", 1.52), 0.5e6);
        let mut worst = [0.0f64; 3];
        for incidence in [Incidence::Film, Incidence::Substrate] {
            let layers = match incidence {
                Incidence::Film => vec![film(), glass()],
                Incidence::Substrate => vec![glass(), film()],
            };
            let whole = Coating { layers, backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None };
            for deg in [0.0f64, 30.0, 60.0] {
                let c0 = deg.to_radians().cos();
                for p in [false, true] {
                    // Coherent, against the whole sandwich's transfer matrix.
                    let a = coating_amp_at(&whole, 1.0, 1.0, lam, c0).unwrap();
                    let s = a.t_power_scale.sqrt();
                    let (r_ref, t_ref) = if p { (-a.r_p, a.t_p * s) } else { (a.r_s, a.t_s * s) };
                    let free = Cx::expi(k0 * 0.5 * c0);
                    let coh = coated_plate(incidence, PlateRoundTrips::Coherent, ExitFace::Bare);
                    let (t, ..) = coating_arm(&coh, false, 1.0, lam, c0, p).unwrap();
                    let (r, ..) = coating_arm(&coh, true, 1.0, lam, c0, p).unwrap();
                    worst[0] = worst[0].max((t * free - t_ref).abs()).max((r - r_ref).abs());
                    // Incoherent, against the ray side's plate.
                    let inc = coated_plate(incidence, PlateRoundTrips::Incoherent, ExitFace::Bare);
                    let rt = coating_rt_at(&inc, 1.0, 1.0, lam, c0).unwrap();
                    let (tw, ..) = coating_arm(&inc, false, 1.0, lam, c0, p).unwrap();
                    let (rw, ..) = coating_arm(&inc, true, 1.0, lam, c0, p).unwrap();
                    let (tr, rr) = if p { (rt.t_p, rt.r_p) } else { (rt.t_s, rt.r_s) };
                    worst[1] = worst[1].max((tw.norm_sqr() - tr).abs()).max((rw.norm_sqr() - rr).abs());
                    // The first pass, T₁T₂ from the faces.
                    let first = coated_plate(incidence, PlateRoundTrips::FirstPass, ExitFace::Bare);
                    let f = coating_plate_faces(&first, 1.0, lam, c0, p).unwrap().unwrap();
                    let (t1, ..) = coating_arm(&first, false, 1.0, lam, c0, p).unwrap();
                    worst[2] = worst[2].max((t1.norm_sqr() - f.t1.norm_sqr() * f.t2.norm_sqr()).abs());
                }
            }
        }
        println!("measured: coated plate, coherent against the sandwich's transfer matrix = {:.3e}; incoherent against the ray side's plate = {:.3e}; first pass against T₁T₂ = {:.3e}", worst[0], worst[1], worst[2]);
        assert!(worst[0] < 1e-10 && worst[1] < 1e-12 && worst[2] < 1e-14, "{worst:?}");
    }
}
