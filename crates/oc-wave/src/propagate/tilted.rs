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
//! doi:10.1364/OE.18.018453, resolved 2026-10-19). The source spectrum is read between its FFT
//! samples by bilinear interpolation, which is the method's approximation:
//! padding the source (`pad`) samples the spectrum finer and makes it
//! smaller. The target field is then evaluated on the output grid by chirp-z,
//! as [`super::asm::asm_zoom`] does, so a small tilted window is not wrapped.
//! A plane wave travelling back across the source plane (`w ≤ 0`) is not part
//! of the forward field and is left out.

use super::embed;
use crate::fft::{fftshift2, ifftshift2, next_fast_len};
use crate::report::{Method, StepReport};
use crate::{Cx, Fft2, Field2, Grid2, WaveError, WaveReport};
use oc_core::{Isometry3, Vec3};
use std::f64::consts::PI;

/// The field `u` on the plane `target` (a frame in world coordinates; its
/// local `XY` the plane), sampled on `out` in that frame, with the source
/// zero-padded `pad` times. The output field carries no carrier.
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
    let p = if pad <= 1 { n } else { [next_fast_len(pad * n[0]), next_fast_len(pad * n[1])] };
    let gp = Grid2::new(p, u.grid.pitch_mm, u.grid.center_mm)?;
    let mut buf = if pad <= 1 { u.data.clone() } else { embed(u, p) };
    ifftshift2(&mut buf, p);
    fft.forward(&mut buf, p, gp.cell_area_mm2());
    fftshift2(&mut buf, p);
    let lam = u.wavelength_in_medium_mm();
    let inv_l2 = 1.0 / (lam * lam);
    let (du, dv) = (gp.freq_pitch(0), gp.freq_pitch(1));
    let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);

    // The target frame's axes and origin in the source frame.
    let src = u.frame;
    let ax = |v: Vec3| src.quat().inverse() * target.rotate(v);
    let (tx, ty, tz) = (ax(Vec3::X), ax(Vec3::Y), ax(Vec3::Z));
    let d = src.quat().inverse() * (target.pos - src.pos);
    // A source-frame frequency from a target-frame one: k = u′ tx + v′ ty + w′ tz.
    let to_src = |up: f64, vp: f64, wp: f64| tx * up + ty * vp + tz * wp;
    // The bilinear spectrum, at absolute source frequency (fx, fy).
    let read = |fx: f64, fy: f64| -> Cx {
        let (gx, gy) = ((fx - u0) / du + (p[0] / 2) as f64, (fy - v0) / dv + (p[1] / 2) as f64);
        if !(gx >= 0.0 && gy >= 0.0 && gx <= (p[0] - 1) as f64 && gy <= (p[1] - 1) as f64) {
            return Cx::ZERO;
        }
        let (i0, j0) = ((gx.floor() as usize).min(p[0] - 2), (gy.floor() as usize).min(p[1] - 2));
        let (s, t) = (gx - i0 as f64, gy - j0 as f64);
        let at = |i: usize, j: usize| buf[j * p[0] + i];
        at(i0, j0) * ((1.0 - s) * (1.0 - t)) + at(i0 + 1, j0) * (s * (1.0 - t)) + at(i0, j0 + 1) * ((1.0 - s) * t) + at(i0 + 1, j0 + 1) * (s * t)
    };
    // The target spectrum on a grid as fine as the source's, over the band
    // the source holds, about the source's carrier seen in the target frame.
    let c = Vec3::new(u0, v0, (inv_l2 - u0 * u0 - v0 * v0).max(0.0).sqrt());
    let ct = Vec3::new(c.dot(tx), c.dot(ty), c.dot(tz));
    let (up0, vp0) = (ct.x, ct.y);
    // A face may look back along the beam (a detector facing it): forward
    // plane waves then have w′ < 0 in its frame.
    let sgn = if tz.z >= 0.0 { 1.0 } else { -1.0 };
    let (mx, my) = (p[0], p[1]);
    let f_at = |k: usize, axis: usize| (k as f64 - (if axis == 0 { mx } else { my } / 2) as f64) * if axis == 0 { du } else { dv };
    let mut spec = vec![Cx::ZERO; mx * my];
    for ky in 0..my {
        let vp = vp0 + f_at(ky, 1);
        for kx in 0..mx {
            let up = up0 + f_at(kx, 0);
            let w2 = inv_l2 - up * up - vp * vp;
            if w2 <= 0.0 {
                continue;
            }
            let wp = sgn * w2.sqrt();
            let k = to_src(up, vp, wp);
            if k.z <= 0.0 {
                continue;
            }
            // ∂(u, v)/∂(u′, v′) with w′ = √(1/λ² − u′² − v′²).
            let (dux, duy) = (tx.x - tz.x * up / wp, ty.x - tz.x * vp / wp);
            let (dvx, dvy) = (tx.y - tz.y * up / wp, ty.y - tz.y * vp / wp);
            let jac = (dux * dvy - duy * dvx).abs();
            let phase = 2.0 * PI * (k.x * d.x + k.y * d.y + k.z * d.z);
            spec[ky * mx + kx] = read(k.x, k.y) * Cx::expi(phase) * jac;
        }
    }
    // Evaluate on the output points by chirp-z, one axis at a time. The
    // target carrier `(u′₀, v′₀)` is put back into the samples.
    let (ox, oy) = (out.n[0], out.n[1]);
    let (x0, y0) = (out.coord(0, 0), out.coord(1, 0));
    let (dx, dy) = (out.pitch_mm[0], out.pitch_mm[1]);
    let mut rows = vec![Cx::ZERO; my * ox];
    for ky in 0..my {
        let line = &spec[ky * mx..(ky + 1) * mx];
        let r = crate::czt::czt_line(line, f_at(0, 0), du, -x0, -dx, ox, fft);
        rows[ky * ox..(ky + 1) * ox].copy_from_slice(&r);
    }
    let mut data = vec![Cx::ZERO; ox * oy];
    let mut col = vec![Cx::ZERO; my];
    for i in 0..ox {
        for ky in 0..my {
            col[ky] = rows[ky * ox + i];
        }
        let cz = crate::czt::czt_line(&col, f_at(0, 1), dv, -y0, -dy, oy, fft);
        for (m, v) in cz.into_iter().enumerate() {
            let (x, y) = out.xy(i, m);
            data[m * ox + i] = v * Cx::expi(2.0 * PI * (up0 * x + vp0 * y));
        }
    }
    let mut v = u.clone();
    v.grid = out;
    v.data = data;
    v.carrier = [0.0, 0.0];
    v.frame = target;
    v.opl_ref_mm = u.opl_ref_mm + u.n_medium * d.z;
    // |U|² over a tilted plane is not power: the power crossing it is the
    // irradiance times the obliquity, taken here as the carrier direction's.
    let obliquity = (ct.z * lam).abs().min(1.0);
    report.ledger.detected = v.power() * obliquity;
    report.ledger.left_window = (report.ledger.launched - report.ledger.detected).max(0.0);
    report.steps.push(StepReport {
        method: Method::AngularSpectrum { band_limited: false },
        z_mm: d.z,
        sampling: None,
        note: format!(
            "angular spectrum of a {}×{} field (padded to {}×{}) onto a plane tilted {:.3}° and {:.4} mm along, by spectrum rotation (Matsushima, Schimmel & Wyrowski 2003) and chirp-z onto {}×{} points; power through the window projected by the carrier's obliquity",
            n[0], n[1], p[0], p[1], tz.z.abs().min(1.0).acos().to_degrees(), d.z, ox, oy
        ),
    });
    report.ledger.close();
    Ok((v, report))
}
