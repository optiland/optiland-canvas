//! The angular spectrum onto a tilted, shifted plane (W8).
//!
//! K. Matsushima, H. Schimmel, F. Wyrowski, "Fast calculation method for
//! optical diffraction on tilted planes by use of the angular spectrum of
//! plane waves," *J. Opt. Soc. Am. A* 20(9), 1755–1762 (2003),
//! doi:10.1364/JOSAA.20.001755 (resolved 2026-10-19): a field is a sum of plane waves, and a plane
//! wave is a plane wave in any frame. Its frequency in the target plane's
//! coordinates `(u′, v′, w′)` and in the source's `(u, v, w)` are the same
//! vector rotated, `(u, v, w) = Rᵀ (u′, v′, w′)`, so the target spectrum is
//! the source spectrum read at `(u, v)`, times the Jacobian of that change of
//! variables,
//!
//! ```text
//! A′(u′, v′) = A(u, v) · |∂(u, v)/∂(u′, v′)| · exp(i2π (u, v, w)·Δo),
//! ```
//!
//! `Δo` the target origin from the source origin, which carries the field the
//! distance between the planes and any sideways shift: an off-axis window is
//! the same sum (K. Matsushima, "Shifted angular spectrum method for off-axis
//! numerical propagation," *Opt. Express* 18(17), 18453–18463 (2010),
//! doi:10.1364/OE.18.018453, resolved 2026-10-19).
//!
//! The source spectrum is wanted between its FFT samples, at the rotated
//! frequencies. It is read there by gridding, the interpolation step of the
//! non-uniform FFT: the field is divided by the kernel's Fourier dual, its
//! spectrum taken on a grid at least twice finer than the window's (the
//! source zero-padded `pad ≥ 2` times), and each rotated frequency read as
//! the kernel-weighted sum of the [`GRID_KERNEL_WIDTH`] nearest samples per
//! axis (J. I. Jackson, C. H. Meyer, D. G. Nishimura, A. Macovski,
//! "Selection of a convolution function for Fourier inversion using
//! gridding," *IEEE Trans. Med. Imaging* 10(3), 473–478 (1991),
//! doi:10.1109/42.97598; L. Greengard, J.-Y. Lee, "Accelerating the
//! nonuniform fast Fourier transform," *SIAM Rev.* 46(3), 443–454 (2004),
//! doi:10.1137/S003614450343200X). The kernel is the Kaiser–Bessel window,
//! `I₀(β√(1 − (2z/w)²))`, whose Fourier dual is closed-form,
//! `w·sinh(√(β² − a²))/√(β² − a²)`, with `β` chosen for the width and the
//! oversampling after P. J. Beatty, D. G. Nishimura, J. M. Pauly, "Rapid
//! gridding reconstruction with a minimal oversampling ratio," *IEEE Trans.
//! Med. Imaging* 24(6), 799–808 (2005), doi:10.1109/TMI.2005.848376, Eq. (5).
//! The reading's error is the kernel's aliasing, about `e^{−β}`: with ten
//! samples a side and the source padded twice, below 1e-8 of the field (the
//! tilted-plane test prints it against a direct plane-wave sum). The target
//! field is then evaluated on the output grid by chirp-z, as
//! [`super::asm::asm_zoom`] does, so a small tilted window is not wrapped.
//! The target spectrum is sampled as finely as the padded source's, so its
//! sum repeats the field every padded period along the plane's axes; only
//! the period about where the carrier through the source window's centre
//! crosses the plane is the field, and output points beyond it are left at
//! zero rather than given the field's copies.
//! A plane wave travelling back across the source plane (`w ≤ 0`) is not part
//! of the forward field and is left out. A target frequency `(u′, v′)` is
//! met by two wavevectors, `w′ = ±√(1/λ² − u′² − v′²)`; each that is a
//! forward wave of the source is kept, with its own Jacobian, so a plane
//! steeply tilted (or facing back along the beam) holds every plane wave
//! that crosses it, whichever side it crosses from. The output carries the
//! source carrier's direction, seen in the target frame, as its own analytic
//! carrier; its samples are the envelope about it.
//!
//! What the plane cannot hold is booked by cause, from the source spectrum
//! (Parseval): its evanescent waves, which do not reach a distant plane, as
//! `evanescent`; plane waves whose frequencies on the plane fall beyond the
//! target spectrum's span as `spectral_projection`; the rest of what the
//! output window does not catch as `left_window`. The power through the
//! window is the irradiance times one obliquity, the carrier's: for a field
//! of directions spread about it, an approximation the step names. Plane
//! waves within about 6° of grazing the plane (`|w′|λ < 0.1`) are kept, and
//! warned of (`OutsideRegime`): the scalar field there is computed, the
//! flux it carries through the plane is not modelled.

use super::embed;
use crate::fft::{fftshift2, ifftshift2, next_fast_len};
use crate::report::{Method, StepReport};
use crate::{Cx, Fft2, Field2, Grid2, WaveError, WaveReport};
use oc_core::{Isometry3, Vec3};
use std::f64::consts::PI;

/// The width of the gridding kernel, in spectral samples per axis.
pub const GRID_KERNEL_WIDTH: usize = 10;

/// The modified Bessel function `I₀(x)` by its power series, every term
/// positive; held to round-off.
fn bessel_i0(x: f64) -> f64 {
    let q = 0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..400u32 {
        term *= q / ((k * k) as f64);
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

/// The Kaiser–Bessel kernel's shape parameter for a width of `w` samples and
/// an oversampling `p` (Beatty, Nishimura & Pauly 2005, Eq. (5)).
fn kaiser_beta(w: usize, p: f64) -> f64 {
    let w = w as f64;
    PI * ((w / p).powi(2) * (p - 0.5).powi(2) - 0.8).max(0.0).sqrt()
}

/// The kernel's Fourier dual (without its factor `w`),
/// `sinh(√(β² − a²))/√(β² − a²)`, at `a = πw·x/L` for a sample `x` from the
/// window's centre and the padded period `L`.
fn kernel_dual(beta: f64, a: f64) -> f64 {
    let s2 = beta * beta - a * a;
    if s2 > 0.0 {
        let s = s2.sqrt();
        s.sinh() / s
    } else if s2 < 0.0 {
        let s = (-s2).sqrt();
        s.sin() / s
    } else {
        1.0
    }
}

/// The kernel's weights for a read at the fractional sample `g`, over the
/// samples `i0..i0 + n` that fall within the kernel's width and the grid.
fn kernel_weights(beta: f64, w: usize, g: f64, len: usize, out: &mut Vec<f64>) -> usize {
    out.clear();
    let half = 0.5 * w as f64;
    let i0 = ((g - half).ceil().max(0.0)) as usize;
    let i1 = ((g + half).floor().min((len - 1) as f64)) as isize;
    if i1 < i0 as isize {
        return i0;
    }
    for i in i0..=(i1 as usize) {
        let z = (i as f64 - g) / half;
        let t = 1.0 - z * z;
        out.push(if t > 0.0 { bessel_i0(beta * t.sqrt()) } else { 0.0 });
    }
    i0
}

/// The field `u` on the plane `target` (a frame in world coordinates; its
/// local `XY` the plane), sampled on `out` in that frame, with the source
/// zero-padded `pad` times (at least twice, for the gridding kernel). The
/// output's analytic carrier is the source carrier's direction in the
/// target frame; its samples are the envelope about it.
pub fn asm_tilted(
    u: &Field2,
    target: Isometry3,
    out: Grid2,
    pad: usize,
    fft: &mut Fft2,
) -> Result<(Field2, WaveReport), WaveError> {
    out.validate()?;
    let mut report = WaveReport::default();
    report.ledger.launched = u.power();
    let n = u.grid.n;
    let pad = pad.max(2);
    let p = [next_fast_len(pad * n[0]), next_fast_len(pad * n[1])];
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let w = GRID_KERNEL_WIDTH;
    // The kernel for each axis's oversampling, and the field divided by its
    // dual (the samples' offsets from the window's centre, in the padded
    // period, as `embed` places them).
    let over = [p[0] as f64 / n[0] as f64, p[1] as f64 / n[1] as f64];
    let beta = [kaiser_beta(w, over[0]), kaiser_beta(w, over[1])];
    let dual = |axis: usize, i: usize| {
        let x = i as f64 - (n[axis] / 2) as f64;
        kernel_dual(beta[axis], PI * w as f64 * x / p[axis] as f64)
    };
    let dual_x: Vec<f64> = (0..n[0]).map(|i| dual(0, i)).collect();
    let dual_y: Vec<f64> = (0..n[1]).map(|j| dual(1, j)).collect();
    let mut pre = u.clone();
    for (j, dy) in dual_y.iter().enumerate() {
        for (i, dx) in dual_x.iter().enumerate() {
            pre.data[j * n[0] + i] *= 1.0 / (dx * dy);
        }
    }
    let mut buf = embed(&pre, p);
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    fftshift2(&mut buf, p);
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
    // Where the source spectrum is: reads beyond it (and the kernel's reach)
    // are skipped.
    let (lo, hi) = crate::plan::decide("reach of the rotated spectrum", || {
        let peak = buf.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
        let (mut lo, mut hi) = ([usize::MAX; 2], [0usize; 2]);
        for ky in 0..p[1] {
            for kx in 0..p[0] {
                if buf[ky * p[0] + kx].norm_sqr() > 1e-20 * peak {
                    lo = [lo[0].min(kx), lo[1].min(ky)];
                    hi = [hi[0].max(kx), hi[1].max(ky)];
                }
            }
        }
        (lo, hi)
    });
    let half = 0.5 * w as f64;
    let reach = [(lo[0] as f64 - half, hi[0] as f64 + half), (lo[1] as f64 - half, hi[1] as f64 + half)];

    // The target frame's axes and origin in the source frame.
    let src = u.frame;
    let ax = |v: Vec3| src.quat().inverse() * target.rotate(v);
    let (tx, ty, tz) = (ax(Vec3::X), ax(Vec3::Y), ax(Vec3::Z));
    // The source spectrum was taken about the source window's centre `c`
    // (`ifftshift2`), while the carrier is referred to the frame origin: the
    // plane waves about `c` carry the carrier's phase there, and the target
    // origin lies `Δo − c` from `c`.
    let c_win = Vec3::new(u.grid.center_mm[0], u.grid.center_mm[1], 0.0);
    let d = src.quat().inverse() * (target.pos - src.pos) - c_win;
    let at_c = Cx::expi(2.0 * PI * (u0 * c_win.x + v0 * c_win.y));
    // A source-frame frequency from a target-frame one: k = u′ tx + v′ ty + w′ tz.
    let to_src = |up: f64, vp: f64, wp: f64| tx * up + ty * vp + tz * wp;
    // The target spectrum on a grid as fine as the source's, over the band
    // the source holds, about the source's carrier seen in the target frame.
    let c = Vec3::new(u0, v0, (inv_l2 - u0 * u0 - v0 * v0).max(0.0).sqrt());
    let ct = Vec3::new(c.dot(tx), c.dot(ty), c.dot(tz));
    let (up0, vp0) = (ct.x, ct.y);
    let (mx, my) = (p[0], p[1]);
    let f_at = |k: usize, axis: usize| (k as f64 - (if axis == 0 { mx } else { my } / 2) as f64) * if axis == 0 { du } else { dv };
    let norm = 1.0 / (w * w) as f64;
    let mut spec = vec![Cx::ZERO; mx * my];
    let (mut wx, mut wy) = (Vec::with_capacity(w + 1), Vec::with_capacity(w + 1));
    // The power read at grazing waves, against all read.
    let (mut grazing, mut read_all) = (0.0, 0.0);
    for ky in 0..my {
        let vp = vp0 + f_at(ky, 1);
        for kx in 0..mx {
            let up = up0 + f_at(kx, 0);
            let w2 = inv_l2 - up * up - vp * vp;
            if w2 <= 0.0 {
                continue;
            }
            let mut acc = Cx::ZERO;
            // Both wavevectors with this frequency on the plane, each if it
            // is a forward wave of the source.
            for wp in [w2.sqrt(), -w2.sqrt()] {
                let k = to_src(up, vp, wp);
                if k.z <= 0.0 {
                    continue;
                }
                // The read, at the source's fractional samples.
                let (gx, gy) = ((k.x - u0) / du + (p[0] / 2) as f64, (k.y - v0) / dv + (p[1] / 2) as f64);
                if gx < reach[0].0 || gx > reach[0].1 || gy < reach[1].0 || gy > reach[1].1 {
                    continue;
                }
                let i0 = kernel_weights(beta[0], w, gx, p[0], &mut wx);
                let j0 = kernel_weights(beta[1], w, gy, p[1], &mut wy);
                if wx.is_empty() || wy.is_empty() {
                    continue;
                }
                let mut a = Cx::ZERO;
                for (dj, cy) in wy.iter().enumerate() {
                    let row = (j0 + dj) * p[0] + i0;
                    let mut line = Cx::ZERO;
                    for (di, cx) in wx.iter().enumerate() {
                        line += buf[row + di] * *cx;
                    }
                    a += line * *cy;
                }
                // ∂(u, v)/∂(u′, v′) with w′ = ±√(1/λ² − u′² − v′²).
                let (dux, duy) = (tx.x - tz.x * up / wp, ty.x - tz.x * vp / wp);
                let (dvx, dvy) = (tx.y - tz.y * up / wp, ty.y - tz.y * vp / wp);
                let jac = (dux * dvy - duy * dvx).abs();
                let phase = 2.0 * PI * (k.x * d.x + k.y * d.y + k.z * d.z);
                let term = a * Cx::expi(phase) * jac;
                let e = term.norm_sqr();
                read_all += e;
                if wp.abs() * lam < 0.1 {
                    grazing += e;
                }
                acc += term;
            }
            spec[ky * mx + kx] = acc * norm * at_c;
        }
    }
    // What the plane cannot hold, by cause, from the source spectrum.
    let (mut evan, mut beyond, mut tot) = (0.0, 0.0, 0.0);
    {
        let g = u.grid;
        let mut s = u.data.clone();
        ifftshift2(&mut s, n);
        fft.forward(&mut s, n, g.cell_area_mm2());
        let half = [(mx / 2) as f64, (my / 2) as f64];
        for ky in 0..n[1] {
            let fy = g.fft_freq(1, ky) + v0;
            for kx in 0..n[0] {
                let fx = g.fft_freq(0, kx) + u0;
                let e = s[ky * n[0] + kx].norm_sqr();
                if e == 0.0 {
                    continue;
                }
                tot += e;
                let w2 = inv_l2 - fx * fx - fy * fy;
                if w2 <= 0.0 {
                    evan += e;
                    continue;
                }
                let kv = Vec3::new(fx, fy, w2.sqrt());
                let (a, b) = ((kv.dot(tx) - up0) / du + half[0], (kv.dot(ty) - vp0) / dv + half[1]);
                if a < 0.0 || b < 0.0 || a > (mx - 1) as f64 || b > (my - 1) as f64 {
                    beyond += e;
                }
            }
        }
    }
    // Evaluate on the output points by chirp-z, one axis at a time: the
    // envelope about the target carrier `(u′₀, v′₀)`. The target spectrum is
    // sampled `du`, `dv` apart, so its sum repeats every `1/du`, `1/dv` along
    // the plane's own axes; only the period about where the field crosses the
    // plane is the field (centred where the carrier through the source
    // window's centre meets the plane). Points beyond it stay zero and are
    // not summed: there the sum would place the field's copies. A plane the
    // carrier grazes keeps every point.
    let (ox, oy) = (out.n[0], out.n[1]);
    let (dx, dy) = (out.pitch_mm[0], out.pitch_mm[1]);
    let chat = c.normalize();
    let along = chat.dot(tz);
    let (xc, yc) = if along.abs() > 1e-9 {
        let r = chat * (d.dot(tz) / along) - d;
        (r.dot(tx), r.dot(ty))
    } else {
        (f64::NAN, f64::NAN)
    };
    let (x0, y0) = (out.coord(0, 0), out.coord(1, 0));
    let ((i0, i1), (j0, j1)) = if xc.is_finite() && yc.is_finite() {
        (super::within_period(x0 - xc, dx, ox, 1.0 / du), super::within_period(y0 - yc, dy, oy, 1.0 / dv))
    } else {
        ((0, ox), (0, oy))
    };
    let (nx, ny) = (i1 - i0, j1 - j0);
    let first = |x0: f64, m0: usize, d: f64| if m0 == 0 { x0 } else { x0 + m0 as f64 * d };
    let (x0, y0) = (first(x0, i0, dx), first(y0, j0, dy));
    let mut rows = vec![Cx::ZERO; my.checked_mul(nx).ok_or(WaveError::TooLarge { samples: usize::MAX, cap: crate::MAX_WAVE_SAMPLES })?];
    for ky in 0..my {
        let line = &spec[ky * mx..(ky + 1) * mx];
        let r = crate::czt::czt_line(line, f_at(0, 0), du, -x0, -dx, nx, fft);
        rows[ky * nx..(ky + 1) * nx].copy_from_slice(&r);
    }
    let mut data = vec![Cx::ZERO; ox * oy];
    let mut col = vec![Cx::ZERO; my];
    for i in 0..nx {
        for ky in 0..my {
            col[ky] = rows[ky * nx + i];
        }
        let cz = crate::czt::czt_line(&col, f_at(0, 1), dv, -y0, -dy, ny, fft);
        for (m, v) in cz.into_iter().enumerate() {
            data[(j0 + m) * ox + i0 + i] = v;
        }
    }
    let zeroed = if nx * ny < ox * oy {
        format!("; the {} of its {} points beyond the field's period on the plane ({:.4e} × {:.4e} mm about where the carrier crosses it) left at zero", ox * oy - nx * ny, ox * oy, 1.0 / du, 1.0 / dv)
    } else {
        String::new()
    };
    let mut v = u.clone();
    v.grid = out;
    v.data = data;
    v.carrier = [up0 * lam, vp0 * lam];
    v.frame = target;
    v.opl_ref_mm = u.opl_ref_mm + u.n_medium * d.z;
    // |U|² over a tilted plane is not power: the power crossing it is the
    // irradiance times the obliquity, taken here as the carrier direction's.
    let obliquity = (ct.z * lam).abs().min(1.0);
    let launched = report.ledger.launched;
    let share = |x: f64| if tot > 0.0 { x / tot * launched } else { 0.0 };
    report.ledger.evanescent = share(evan);
    report.ledger.spectral_projection = share(beyond);
    report.ledger.detected = v.power() * obliquity;
    report.ledger.left_window = (launched - report.ledger.detected - report.ledger.evanescent - report.ledger.spectral_projection).max(0.0);
    if read_all > 0.0 && grazing > 1e-6 * read_all {
        report.warn(crate::report::SamplingWarning::OutsideRegime {
            method: Method::AngularSpectrum { band_limited: false },
            reason: format!(
                "{:.1e} of the field read onto the tilted plane is in plane waves within about 6° of grazing it (|w′|λ < 0.1): the scalar field there is computed, the flux it carries through the plane is not modelled",
                grazing / read_all
            ),
        });
    }
    report.steps.push(StepReport {
        method: Method::AngularSpectrum { band_limited: false },
        z_mm: d.z,
        sampling: None,
        note: format!(
            "angular spectrum of a {}×{} field (padded to {}×{}) onto a plane tilted {:.3}° and {:.4} mm along, by spectrum rotation (Matsushima, Schimmel & Wyrowski 2003), the spectrum read between its samples by a Kaiser–Bessel kernel {w} samples wide, and chirp-z onto {}×{} points; every forward plane wave that crosses the plane kept, from either side; power through the window projected by one obliquity, the carrier's{zeroed}",
            n[0], n[1], p[0], p[1], tz.z.abs().min(1.0).acos().to_degrees(), d.z, ox, oy
        ),
    });
    report.ledger.close();
    Ok((v, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bessel_series_matches_tabulated_values() {
        // Abramowitz & Stegun, Table 9.8: I₀(1) = 1.2660658778, I₀(2) =
        // 2.2795853023, I₀(5) = 27.239871823.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_8).abs() < 1e-9);
        assert!((bessel_i0(2.0) - 2.279_585_302_3).abs() < 1e-9);
        assert!((bessel_i0(5.0) - 27.239_871_823).abs() < 1e-7);
    }

    #[test]
    fn the_kernel_sums_to_its_dual_at_the_origin() {
        // Poisson: Σ_m ψ(m) over the integers equals w·S(0) up to the
        // kernel's aliasing, which is the reading's error.
        for p in [2.0, 4.0] {
            let w = GRID_KERNEL_WIDTH;
            let beta = kaiser_beta(w, p);
            let half = 0.5 * w as f64;
            let sum: f64 = (-(w as i64)..=(w as i64))
                .map(|m| {
                    let z = m as f64 / half;
                    let t = 1.0 - z * z;
                    if t > 0.0 {
                        bessel_i0(beta * t.sqrt())
                    } else {
                        0.0
                    }
                })
                .sum();
            let want = w as f64 * kernel_dual(beta, 0.0);
            let err = ((sum - want) / want).abs();
            eprintln!("measured: Kaiser–Bessel kernel of width {w} at oversampling {p}: Σψ(m) against w·S(0), relative = {err:.3e}");
            assert!(err < 1e-8, "{err}");
        }
    }
}
