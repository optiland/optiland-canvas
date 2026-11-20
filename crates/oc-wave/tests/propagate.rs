//! The propagators as numerical methods: power, reversibility, regimes,
//! agreement with each other, and the ledger (acceptance cases C37, C40, C43).
//!
//! Sources, numbered as in the Sources list of docs/wave-optics.md:
//! - [1] J. W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H.
//!   Freeman/Macmillan Learning (2017), ISBN 978-1-319-11916-4: the angular
//!   spectrum's transfer function has unit magnitude for propagating waves,
//!   so power is conserved and a step back undoes a step forward.
//! - [2] K. Matsushima, T. Shimobaba, *Opt. Express* 17(22), 19662–19673
//!   (2009), doi:10.1364/OE.17.019662: the band limit, Eqs. (13), (18)–(20).
//! - [3] D. G. Voelz, M. C. Roggemann, *Appl. Opt.* 48(32), 6132–6142 (2009),
//!   doi:10.1364/AO.48.006132: the critical distance `z_c = LΔx/λ` and which
//!   Fresnel propagator each side of it favours.
//! - [4] F. Shen, A. Wang, *Appl. Opt.* 45(6), 1102–1110 (2006),
//!   doi:10.1364/AO.45.001102: direct Rayleigh–Sommerfeld integration.
//! - [6] K. Matsushima, H. Schimmel, F. Wyrowski, *J. Opt. Soc. Am. A* 20(9),
//!   1755–1762 (2003), doi:10.1364/JOSAA.20.001755: the spectrum rotated
//!   onto a tilted plane; [7] K. Matsushima, *Opt. Express* 18(17),
//!   18453–18463 (2010), doi:10.1364/OE.18.018453: the shifted window.
//! - [30] J. I. Jackson et al., *IEEE Trans. Med. Imaging* 10(3), 473–478
//!   (1991), doi:10.1109/42.97598, and [31] P. J. Beatty, D. G. Nishimura,
//!   J. M. Pauly, *IEEE Trans. Med. Imaging* 24(6), 799–808 (2005),
//!   doi:10.1109/TMI.2005.848376: gridding with the Kaiser–Bessel kernel,
//!   which reads the rotated spectrum between its samples.
//!
//! All retrieved 2026-10-19.

mod common;

use common::{max_diff, measured, peak};
use oc_wave::beams::gaussian;
use oc_wave::elements::{apply_aperture, Shape};
use oc_wave::propagate::{critical_distance_mm, propagate};
use oc_wave::{AsmOptions, Field2, Fft2, Grid2, Method, Propagator, SamplingWarning, WaveReport};

fn gaussian_beam() -> Field2 {
    // w₀ = 0.1 mm at 0.5 µm: z_R = 62.8 mm. 256 samples at 5 µm, 1.28 mm.
    gaussian(Grid2::square(256, 0.005).unwrap(), 0.5, 1.0, 0.1, 0.0, 1.0, [0.0, 0.0]).unwrap()
}

#[test]
fn every_propagator_conserves_power_when_nothing_is_band_limited() {
    let u = gaussian_beam();
    let p0 = u.power();
    let mut fft = Fft2::new();
    // Critical distance on the 2× padded window: 2.56 mm · 5 µm / 0.5 µm = 25.6 mm.
    let zc = critical_distance_mm(0.005, 2.56, 0.0005);
    assert!((zc - 25.6).abs() < 1e-12);

    // Unit-magnitude transfer functions: exact to round-off [1].
    for (how, z) in [
        (Propagator::AngularSpectrum(AsmOptions::default()), 10.0),
        (Propagator::FresnelTf, 10.0),
        (Propagator::Fraunhofer { pad: 2 }, 2000.0),
    ] {
        let (v, r) = propagate(&u, z, how, &mut fft).unwrap();
        let rel = ((v.power() - p0) / p0).abs();
        measured(&format!("power change, {how:?} at {z} mm"), rel);
        assert!(rel < 1e-12, "{how:?}: {rel}");
        assert!(r.ledger.numerical.abs() < 1e-12 * p0);
    }

    // The impulse response transforms a sampled, truncated chirp, which is
    // not exactly unitary even in its own regime [3]; what it loses or gains
    // is in the ledger as numerical, and it is small.
    let (v, r) = propagate(&u, 100.0, Propagator::FresnelIr, &mut fft).unwrap();
    let rel = ((v.power() - p0) / p0).abs();
    measured("power change, Fresnel impulse response at 100 mm", rel);
    assert!(rel < 1e-6, "{rel}");
    assert!(r.warnings.iter().all(|w| !matches!(w, SamplingWarning::OutsideRegime { .. })));
}

#[test]
fn asm_forward_then_back_returns_the_field() {
    let u = gaussian_beam();
    let mut fft = Fft2::new();
    let how = Propagator::AngularSpectrum(AsmOptions::default());
    let (v, _) = propagate(&u, 10.0, how, &mut fft).unwrap();
    let (w, _) = propagate(&v, -10.0, how, &mut fft).unwrap();
    let err = max_diff(&u.data, &w.data) / peak(&u.data);
    measured("forward then back, relative to peak", err);
    assert!(err < 1e-12, "{err}");
}

#[test]
fn voelz_regime_picks_the_transfer_function_below_the_critical_distance() {
    let u = gaussian_beam();
    let mut fft = Fft2::new();
    // Below z_c = 25.6 mm Auto runs the angular spectrum, which is exact there.
    let (_, r) = propagate(&u, 20.0, Propagator::Auto, &mut fft).unwrap();
    assert!(matches!(r.steps[0].method, Method::AngularSpectrum { band_limited: true }));
    // Forcing the impulse response below z_c, or the transfer function above
    // it, is allowed and reported [3].
    let (_, r) = propagate(&u, 20.0, Propagator::FresnelIr, &mut fft).unwrap();
    assert!(r.warnings.iter().any(|w| matches!(w, SamplingWarning::OutsideRegime { method: Method::FresnelIr, .. })));
    let (_, r) = propagate(&u, 40.0, Propagator::FresnelTf, &mut fft).unwrap();
    assert!(r.warnings.iter().any(|w| matches!(w, SamplingWarning::OutsideRegime { method: Method::FresnelTf, .. })));
    // In regime, no warning.
    let (_, r) = propagate(&u, 20.0, Propagator::FresnelTf, &mut fft).unwrap();
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    // Beyond z_c Auto takes the smaller of the two errors this field would
    // meet, both known from its spectrum: this Gaussian's spectrum lies
    // inside the band limit but for its far tails, so the angular spectrum
    // removes less than the paraxial step's small error. (Where the band
    // limit cuts,
    // a hard-edged aperture's far field, the impulse response wins: the
    // Heurtley test in analytic_fields.rs.)
    let (_, r) = propagate(&u, 200.0, Propagator::Auto, &mut fft).unwrap();
    assert_eq!(r.steps[0].method, Method::AngularSpectrum { band_limited: true });
    assert!(r.steps[0].note.contains("model-error indicators"), "{}", r.steps[0].note);
    let (eb, ep) = (oc_wave::propagate::band_limit_error(&u, 200.0, &mut fft), oc_wave::propagate::paraxial_error(&u, 200.0, &mut fft));
    measured("Gaussian at 200 mm: the band limit's error", eb);
    measured("Gaussian at 200 mm: the paraxial step's error", ep);
    assert!(eb < ep && ep < 1e-5, "{eb} {ep}");
}

#[test]
fn the_paraxial_step_s_error_is_the_identity_s_exactly() {
    // Bundle M13, Eq. M32 (B08): the difference between the exact and the
    // paraxial propagator on a field is Σ 4 sin²(z(κ − κ₂)/2)|ĝ|², and is
    // bounded by z²Σ|κ − κ₂|²|ĝ|². Both propagators run here on the same
    // twice-padded grid, so the identity holds to round-off on what stays in
    // the window. A Gaussian tilted 0.15 rad, whose spectrum sits far from
    // the axis, so the error is large enough to measure.
    let g = Grid2::square(256, 0.004).unwrap();
    let mut u = gaussian(g, 0.5, 1.0, 0.08, 0.0, 1.0, [0.0, 0.0]).unwrap();
    u.carrier = [0.15, 0.0];
    let mut fft = Fft2::new();
    let z = 2.0;
    let exact = Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 2 });
    let (a, _) = propagate(&u, z, exact, &mut fft).unwrap();
    let (b, _) = propagate(&u, z, Propagator::FresnelTf, &mut fft).unwrap();
    let diff: f64 = a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>().sqrt();
    let norm: f64 = u.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let want = oc_wave::propagate::paraxial_error(&u, z, &mut fft);
    measured("paraxial error: the propagators' difference against the identity, relative", diff / norm / want - 1.0);
    assert!(want > 1e-3, "{want}");
    assert!((diff / norm / want - 1.0).abs() < 1e-6, "{} vs {want}", diff / norm);
    // Forced, the paraxial step says how wrong it was.
    let (_, r) = propagate(&u, z, Propagator::FresnelTf, &mut fft).unwrap();
    assert!(r.warnings.iter().any(|w| matches!(w, SamplingWarning::Paraxial { field_error } if (field_error - want).abs() < 1e-12)), "{:?}", r.warnings);
}

#[test]
fn rayleigh_sommerfeld_direct_agrees_with_band_limited_asm() {
    // A beam narrow enough to diffract within a few mm: w₀ = 20 µm at 0.5 µm,
    // z_R = 2.51 mm. An odd sample count so the direct method uses
    // Simpson's rule as Shen & Wang do [4].
    let g = Grid2::square(255, 0.002).unwrap();
    let u = gaussian(g, 0.5, 1.0, 0.02, 0.0, 1.0, [0.0, 0.0]).unwrap();
    let mut fft = Fft2::new();
    // At 5 mm the kernel is sampled to the largest offset: Δx = 2 µm against
    // the λ√(z² + L²)/(2L) = 2.47 µm the direct method needs.
    let (a, _) = propagate(&u, 5.0, Propagator::AngularSpectrum(AsmOptions::default()), &mut fft).unwrap();
    let (b, rb) = propagate(&u, 5.0, Propagator::RsDirect, &mut fft).unwrap();
    assert!(rb.steps[0].note.contains("Simpson"));
    assert!(rb.warnings.is_empty(), "{:?}", rb.warnings);
    let err = max_diff(&a.data, &b.data) / peak(&a.data);
    measured("direct RS vs band-limited ASM at 5 mm, relative to peak", err);
    assert!(err < 1e-4, "{err}");
    // At 2 mm it is not, and the report says so instead of passing an
    // aliased kernel off as a result.
    let (_, rb) = propagate(&u, 2.0, Propagator::RsDirect, &mut fft).unwrap();
    assert!(rb
        .warnings
        .iter()
        .any(|w| matches!(w, SamplingWarning::Undersampled { .. })));
}

#[test]
fn the_band_limit_reports_the_power_it_removes() {
    // A sharp square aperture carried 2 m on a small window: the transfer
    // function is badly undersampled there, so the band limit must bite [2].
    let g = Grid2::square(128, 0.01).unwrap();
    let mut u = Field2::from_fn(g, 0.5, |_, _| oc_wave::Cx::ONE).unwrap();
    let mut r0 = WaveReport::default();
    apply_aperture(&mut u, &Shape::Rect { c: [0.0, 0.0], h: [0.2, 0.2] }, false, &mut r0).unwrap();
    let mut fft = Fft2::new();
    let (_, r) = propagate(&u, 2000.0, Propagator::AngularSpectrum(AsmOptions::default()), &mut fft).unwrap();
    let removed = r.ledger.band_limit_removed / r.ledger.launched;
    measured("band limit removed (fraction of launched)", removed);
    assert!(removed > 1e-3, "{removed}");
    assert!(r.warnings.iter().any(|w| matches!(w, SamplingWarning::BandLimited { .. })));
    // Detected + removed + evanescent + left the window = launched.
    assert!(r.ledger.numerical.abs() < 1e-12 * r.ledger.launched, "{:?}", r.ledger);
}

#[test]
fn an_aperture_passes_exactly_its_area_of_a_uniform_field() {
    // Exact cell coverage: the transmitted power of a unit field is the
    // shape's area, whatever the edges cut.
    let g = Grid2::square(200, 0.013).unwrap();
    let shapes = [
        Shape::Rect { c: [0.031, -0.2], h: [0.47, 0.333] },
        Shape::Disc { c: [0.1, 0.05], r: 0.777 },
        Shape::Annulus { c: [-0.05, 0.0], r_in: 0.3, r_out: 0.9 },
        Shape::Polygon(vec![[-0.8, -0.6], [0.9, -0.4], [0.2, 0.95], [0.0, 0.1]]),
    ];
    for s in shapes {
        let mut u = Field2::from_fn(g, 0.5, |_, _| oc_wave::Cx::ONE).unwrap();
        let mut r = WaveReport::default();
        apply_aperture(&mut u, &s, false, &mut r).unwrap();
        // Amplitude transmittance is the covered fraction, so power is Σ f²ΔA;
        // the area itself is Σ fΔA.
        let area: f64 = u.data.iter().map(|z| z.re).sum::<f64>() * g.cell_area_mm2();
        let rel = ((area - s.area()) / s.area()).abs();
        assert!(rel < 1e-12, "{s:?}: {rel}");
        assert!(r.ledger.absorbed_aperture > 0.0);
    }
}

#[test]
fn the_zoomed_angular_spectrum_lands_on_the_angular_spectrum_on_its_own_grid() {
    // asm_zoom evaluates the same plane-wave sum by chirp-z on any grid; on
    // the input's own grid, with nothing band-limited (z below the critical
    // distance), it must agree with the FFT angular spectrum (Goodman [1];
    // the chirp-z sum: Rabiner, Schafer & Rader [9]).
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_zoom, propagate, AsmOptions, Fft2, Grid2, Propagator};
    let grid = Grid2::square(256, 2.0 / 256.0).unwrap();
    let u = GaussianBeam { w0_mm: 0.15, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(grid, 0.0).unwrap();
    let mut fft = Fft2::new();
    let z = 5.0;
    let (a, _) = propagate(&u, z, Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 2 }), &mut fft).unwrap();
    let (b, rep) = asm_zoom(&u, z, grid, 2, &mut fft).unwrap();
    let peak = a.data.iter().map(|c| c.abs()).fold(0.0, f64::max);
    let worst = a.data.iter().zip(&b.data).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max);
    eprintln!("measured: zoomed ASM vs FFT ASM on its own grid, worst / peak = {:.3e}", worst / peak);
    assert!(worst / peak < 1e-10, "{worst}");
    assert!(rep.ledger.numerical.abs() < 1e-10, "{:?}", rep.ledger);
    // A window an eighth the size at four times the density holds the same
    // field where the two grids share points.
    let fine = Grid2::square(64, 2.0 / 1024.0).unwrap();
    let (c, _) = asm_zoom(&u, z, fine, 2, &mut fft).unwrap();
    let centre = (c.at(32, 32) - a.at(128, 128)).abs() / peak;
    eprintln!("measured: zoomed ASM on a 4× finer window, centre sample / peak = {centre:.3e}");
    assert!(centre < 1e-10);
}

#[test]
fn an_off_axis_window_holds_the_field_the_large_window_does() {
    // A tilted beam walks z·tan α sideways. The zoomed angular spectrum on a
    // window centred there (Matsushima 2010 [7]: the shifted window, here
    // evaluated directly rather than by a shifted FFT) must hold what the FFT
    // angular spectrum gives on a window four times as wide.
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_zoom, propagate, AsmOptions, Fft2, Grid2, Propagator};
    let pitch = 1.0 / 128.0;
    let small = Grid2::square(128, pitch).unwrap();
    let big = Grid2::square(512, pitch).unwrap();
    let mut a = GaussianBeam { w0_mm: 0.08, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(small, 0.0).unwrap();
    let mut b = GaussianBeam { w0_mm: 0.08, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(big, 0.0).unwrap();
    a.carrier = [0.03, 0.0];
    b.carrier = [0.03, 0.0];
    let z = 20.0;
    let mut fft = Fft2::new();
    let (wide, _) = propagate(&b, z, Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 2 }), &mut fft).unwrap();
    // 40 samples about x = 0.6 mm, on the wide grid's own points.
    let shift = (0.6 / pitch).round() * pitch;
    let win = Grid2::new([40, 40], [pitch, pitch], [shift, 0.0]).unwrap();
    let (zoom, _) = asm_zoom(&a, z, win, 2, &mut fft).unwrap();
    let peak = wide.data.iter().map(|c| c.abs()).fold(0.0, f64::max);
    let mut worst: f64 = 0.0;
    for j in 0..40 {
        for i in 0..40 {
            let (x, y) = win.xy(i, j);
            let (bi, bj) = ((x / pitch).round() as isize + 256, (y / pitch).round() as isize + 256);
            worst = worst.max((zoom.at(i, j) - wide.at(bi as usize, bj as usize)).abs());
        }
    }
    eprintln!("measured: off-axis zoomed window against the wide FFT angular spectrum, worst / peak = {:.3e}", worst / peak);
    assert!(worst / peak < 1e-6, "{worst}");
}

#[test]
fn a_tilted_plane_holds_the_field_its_plane_waves_sum_to() {
    // Matsushima, Schimmel & Wyrowski 2003 [6]: the spectrum rotated onto a
    // plane tilted 20°, 2 mm along, against the plane waves of the same
    // spectrum summed directly at that plane's points. The method reads the
    // source spectrum between its samples (by gridding, a Kaiser–Bessel
    // kernel ten samples wide on the spectrum of the field padded twice or
    // more [30], [31]), which is all that separates the two; with no tilt it
    // must agree with the zoomed angular spectrum to the kernel's own
    // accuracy; rolled about its normal, it reads the field's samples at the
    // rolled points, against the Gaussian itself there.
    use oc_core::{Isometry3, Vec3};
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_tilted, asm_zoom, Cx, Fft2, Grid2};
    let pitch = 1.0 / 128.0;
    let g = Grid2::square(128, pitch).unwrap();
    let u = GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
    let mut fft = Fft2::new();
    let out = Grid2::new([40, 40], [0.01, 0.01], [0.0, 0.0]).unwrap();
    // No tilt, shifted: exact.
    let shifted = Isometry3::from_pos(Vec3::new(0.05, 0.0, 2.0));
    let (t0, _) = asm_tilted(&u, shifted, out, 4, &mut fft).unwrap();
    let win = Grid2::new([40, 40], [0.01, 0.01], [0.05, 0.0]).unwrap();
    let (z0, _) = asm_zoom(&u, 2.0, win, 4, &mut fft).unwrap();
    let peak0 = z0.data.iter().map(|c| c.abs()).fold(0.0, f64::max);
    let worst0 = t0.data.iter().zip(&z0.data).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
    eprintln!("measured: shifted parallel plane, spectrum rotation against the zoomed angular spectrum, worst / peak = {:.3e}", worst0 / peak0);
    assert!(worst0 / peak0 < 1e-10);
    // Tilted 20° about the source's y axis.
    let th = 20f64.to_radians();
    let q = glam::DQuat::from_rotation_y(th);
    let tilted = Isometry3::new(Vec3::new(0.0, 0.0, 2.0), [q.x, q.y, q.z, q.w]).unwrap();
    let (t1, _) = asm_tilted(&u, tilted, out, 2, &mut fft).unwrap();
    let k1 = t1.k_mm();
    // The direct sum over the same padded spectrum.
    let p = 512usize;
    let mut buf = vec![Cx::ZERO; p * p];
    for j in 0..128 {
        for i in 0..128 {
            buf[(j + 192) * p + i + 192] = u.at(i, j);
        }
    }
    oc_wave::ifftshift2(&mut buf, [p, p]);
    fft.forward(&mut buf, [p, p], pitch * pitch);
    oc_wave::fftshift2(&mut buf, [p, p]);
    let df = 1.0 / (p as f64 * pitch);
    let lam = 0.6328e-3;
    let mut worst: f64 = 0.0;
    let mut peak: f64 = 0.0;
    for j in (0..40).step_by(3) {
        for i in (0..40).step_by(3) {
            let (x, y) = out.xy(i, j);
            let r = tilted.transform_point(Vec3::new(x, y, 0.0));
            let mut s = Cx::ZERO;
            for ky in 0..p {
                let fy = (ky as f64 - 256.0) * df;
                for kx in 0..p {
                    let a = buf[ky * p + kx];
                    if a.norm_sqr() < 1e-30 {
                        continue;
                    }
                    let fx = (kx as f64 - 256.0) * df;
                    let w2 = 1.0 / (lam * lam) - fx * fx - fy * fy;
                    if w2 <= 0.0 {
                        continue;
                    }
                    s += a * Cx::expi(2.0 * std::f64::consts::PI * (fx * r.x + fy * r.y + w2.sqrt() * r.z));
                }
            }
            let s = s * (df * df);
            peak = peak.max(s.abs());
            // The rotation keeps its carrier analytic: put it back.
            let full = t1.at(i, j) * Cx::expi(k1 * (t1.carrier[0] * x + t1.carrier[1] * y));
            worst = worst.max((full - s).abs());
        }
    }
    eprintln!("measured: tilted plane (20°), spectrum rotation against the direct plane-wave sum, worst / peak = {:.3e}", worst / peak);
    assert!(worst / peak < 1e-7, "{}", worst / peak);
    // Rolled 30° about its normal, on the source plane: the field at the
    // rolled points, which for the sampled Gaussian is the Gaussian itself
    // (its band well inside the grid's, its tails 1e-11 at the window's
    // edge).
    let qz = glam::DQuat::from_rotation_z(30f64.to_radians());
    let rolled = Isometry3::new(Vec3::ZERO, [qz.x, qz.y, qz.z, qz.w]).unwrap();
    let (t2, rep2) = asm_tilted(&u, rolled, out, 2, &mut fft).unwrap();
    let (mut worst2, mut peak2): (f64, f64) = (0.0, 0.0);
    for j in 0..40 {
        for i in 0..40 {
            let (x, y) = out.xy(i, j);
            let r = rolled.transform_point(Vec3::new(x, y, 0.0));
            let want = u.at(64, 64).abs() * (-(r.x * r.x + r.y * r.y) / (0.1f64 * 0.1)).exp();
            peak2 = peak2.max(want);
            worst2 = worst2.max((t2.at(i, j).abs() - want).abs());
        }
    }
    eprintln!("measured: plane rolled 30° about the normal, spectrum rotation against the Gaussian at the rolled points, worst / peak = {:.3e}", worst2 / peak2);
    assert!(worst2 / peak2 < 1e-9, "{}", worst2 / peak2);
    // The window holds all but the Gaussian's tails (1.3e-4 beyond 0.2 mm),
    // booked as left; the rest closes.
    assert!(t2.carrier == [0.0, 0.0] && rep2.ledger.detected > 0.9998 && rep2.ledger.numerical.abs() < 1e-12, "{:?}", rep2.ledger);
}

#[test]
fn an_off_centre_input_window_propagates_where_its_field_is() {
    // R03 of the expert review. The FFT takes a window's spectrum about the
    // window's centre `c`, so the plane-wave sum must be evaluated at `x − c`
    // (`asm_zoom`), and a target origin lies `Δo − c` from that centre
    // (`asm_tilted`). A window centred off the frame origin, as a re-grid
    // makes, must carry its field where it is: against the FFT angular
    // spectrum on a wide, centred window holding the same field [1], with
    // and without a carrier, onto output windows centred anywhere; and the
    // spectrum rotation onto a shifted parallel plane against the zoomed
    // angular spectrum, to round-off.
    use oc_core::{Isometry3, Vec3};
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_tilted, asm_zoom, propagate, AsmOptions, Cx, Fft2, Grid2, Propagator};
    let pitch = 1.0 / 128.0;
    let c = [0.2, -0.1];
    let small = Grid2::new([128, 128], [pitch, pitch], c).unwrap();
    let big = Grid2::square(512, pitch).unwrap();
    let mut fft = Fft2::new();
    let z = 20.0;
    for carrier in [[0.0, 0.0], [0.02, -0.01]] {
        let mut a = GaussianBeam { w0_mm: 0.08, lambda_um: 0.6328, n: 1.0, power: 1.0, center: c }.sample(small, 0.0).unwrap();
        let mut b = GaussianBeam { w0_mm: 0.08, lambda_um: 0.6328, n: 1.0, power: 1.0, center: c }.sample(big, 0.0).unwrap();
        a.carrier = carrier;
        b.carrier = carrier;
        let (wide, _) = propagate(&b, z, Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 2 }), &mut fft).unwrap();
        let peak = wide.data.iter().map(|q| q.abs()).fold(0.0, f64::max);
        // Forty samples about where the beam lands (it walks z·α), on the
        // wide grid's own points, and a window centred elsewhere.
        for shift in [[c[0] + z * carrier[0], c[1] + z * carrier[1]], [0.0, 0.0]] {
            let win = Grid2::new([40, 40], [pitch, pitch], [(shift[0] / pitch).round() * pitch, (shift[1] / pitch).round() * pitch]).unwrap();
            let (zoom, _) = asm_zoom(&a, z, win, 2, &mut fft).unwrap();
            let mut worst: f64 = 0.0;
            for j in 0..40 {
                for i in 0..40 {
                    let (x, y) = win.xy(i, j);
                    let (bi, bj) = ((x / pitch).round() as isize + 256, (y / pitch).round() as isize + 256);
                    worst = worst.max((zoom.at(i, j) - wide.at(bi as usize, bj as usize)).abs());
                }
            }
            measured(&format!("off-centre input window, carrier {carrier:?}, output about {shift:?}: zoomed against the wide FFT angular spectrum, worst / peak"), worst / peak);
            assert!(worst / peak < 1e-6, "{}", worst / peak);
        }
        // The spectrum rotation onto a parallel plane shifted sideways (to
        // where the beam lands, and a little beside it) is the same
        // plane-wave sum, so it must agree with the zoomed angular spectrum
        // to round-off, input centre or not.
        let land = [c[0] + z * carrier[0] + 0.05, c[1] + z * carrier[1]];
        let out = Grid2::new([40, 40], [0.01, 0.01], [0.0, 0.0]).unwrap();
        let shifted = Isometry3::from_pos(Vec3::new(land[0], land[1], z));
        let (t0, _) = asm_tilted(&a, shifted, out, 4, &mut fft).unwrap();
        let win = Grid2::new([40, 40], [0.01, 0.01], land).unwrap();
        let (z0, _) = asm_zoom(&a, z, win, 4, &mut fft).unwrap();
        // Both keep their carriers analytic, each about its own origin: put
        // them back on the samples.
        let k = z0.k_mm();
        let (mut worst0, mut peak0): (f64, f64) = (0.0, 0.0);
        for j in 0..40 {
            for i in 0..40 {
                let (x, y) = win.xy(i, j);
                let full = z0.at(i, j) * Cx::expi(k * (z0.carrier[0] * x + z0.carrier[1] * y));
                let (xt, yt) = out.xy(i, j);
                let full_t = t0.at(i, j) * Cx::expi(k * (t0.carrier[0] * xt + t0.carrier[1] * yt));
                peak0 = peak0.max(full.abs());
                worst0 = worst0.max((full_t - full).abs());
            }
        }
        measured(&format!("off-centre input window, carrier {carrier:?}: shifted parallel plane by spectrum rotation against the zoomed angular spectrum, worst / peak"), worst0 / peak0);
        assert!(worst0 / peak0 < 1e-9, "{}", worst0 / peak0);
    }
}

#[test]
fn a_translated_frame_and_a_translated_grid_hold_the_same_field() {
    // R03 and R06 of the expert review (gauge): one physical field, encoded
    // two ways, a grid centred at `c` in a frame at the origin and a centred
    // grid in a frame moved to `c`, propagated onto the same physical points
    // by the zoomed angular spectrum and by the spectrum rotation onto one
    // absolute plane, must agree to round-off. The encoding is not physics.
    use oc_core::{Isometry3, Vec3};
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_tilted, asm_zoom, Fft2, Grid2};
    let pitch = 1.0 / 128.0;
    let c = [0.3, 0.15];
    let z = 12.0;
    let mut fft = Fft2::new();
    let on_grid = GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: c }
        .sample(Grid2::new([128, 128], [pitch, pitch], c).unwrap(), 0.0)
        .unwrap();
    let mut in_frame = GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }
        .sample(Grid2::square(128, pitch).unwrap(), 0.0)
        .unwrap();
    in_frame.frame = Isometry3::from_pos(Vec3::new(c[0], c[1], 0.0));
    assert!(on_grid.data.iter().zip(&in_frame.data).all(|(a, b)| (*a - *b).abs() < 1e-12));
    let out_a = Grid2::new([48, 48], [0.005, 0.005], [c[0] + 0.02, c[1] - 0.03]).unwrap();
    let out_b = Grid2::new([48, 48], [0.005, 0.005], [0.02, -0.03]).unwrap();
    let (za, _) = asm_zoom(&on_grid, z, out_a, 2, &mut fft).unwrap();
    let (zb, _) = asm_zoom(&in_frame, z, out_b, 2, &mut fft).unwrap();
    let peak = za.data.iter().map(|q| q.abs()).fold(0.0, f64::max);
    let worst = za.data.iter().zip(&zb.data).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
    measured("translated grid against translated frame, zoomed angular spectrum, worst / peak", worst / peak);
    assert!(worst / peak < 1e-12, "{}", worst / peak);
    // One absolute target plane, tilted 15° about y.
    let q = glam::DQuat::from_rotation_y(15f64.to_radians());
    let target = Isometry3::new(Vec3::new(c[0] + 0.02, c[1], z), [q.x, q.y, q.z, q.w]).unwrap();
    let out = Grid2::new([40, 40], [0.006, 0.006], [0.0, 0.0]).unwrap();
    let (ta, _) = asm_tilted(&on_grid, target, out, 4, &mut fft).unwrap();
    let (tb, _) = asm_tilted(&in_frame, target, out, 4, &mut fft).unwrap();
    let peak = ta.data.iter().map(|q| q.abs()).fold(0.0, f64::max);
    let worst = ta.data.iter().zip(&tb.data).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
    measured("translated grid against translated frame, spectrum rotation onto one plane, worst / peak", worst / peak);
    assert!(worst / peak < 1e-12, "{}", worst / peak);
}

#[test]
fn the_zoomed_fresnel_and_fraunhofer_transforms_take_the_grid_where_it_is() {
    // Both chirp-z transforms run on the samples' absolute coordinates, so a
    // window centred off the frame origin needs no correction: the zoomed
    // Fresnel transform of a paraxial beam on such a window agrees with the
    // zoomed angular spectrum, and the far field of a shifted field differs
    // from the centred one's by the shift phase e^{−i2π f·c} alone [1].
    use oc_wave::beams::GaussianBeam;
    use oc_wave::czt::{fraunhofer_zoom, fresnel_zoom};
    use oc_wave::{asm_zoom, Cx, Fft2, Grid2};
    let pitch = 1.0 / 128.0;
    let c = [0.25, -0.2];
    let mut fft = Fft2::new();
    let u = GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: c }
        .sample(Grid2::new([128, 128], [pitch, pitch], c).unwrap(), 0.0)
        .unwrap();
    // 400 mm on: the beam is 0.8 mm wide there, so the angular spectrum is
    // padded eight times, for its periodic copies (a padded window apart)
    // to stay clear of the window read; the chirp-z Fresnel transform has
    // no copies.
    let z = 400.0;
    let out = Grid2::new([48, 48], [0.03, 0.03], [c[0], c[1]]).unwrap();
    let (f, _) = fresnel_zoom(&u, z, out, &mut fft).unwrap();
    let (a, _) = asm_zoom(&u, z, out, 8, &mut fft).unwrap();
    let peak = a.data.iter().map(|q| q.abs()).fold(0.0, f64::max);
    let worst = f.data.iter().zip(&a.data).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max);
    measured("zoomed Fresnel transform of an off-centre window against the zoomed angular spectrum, worst / peak", worst / peak);
    assert!(worst / peak < 1e-4, "{}", worst / peak);
    let u0 = GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }
        .sample(Grid2::square(128, pitch).unwrap(), 0.0)
        .unwrap();
    let zf = 2.0e4;
    let far = Grid2::new([32, 32], [0.5, 0.5], [0.0, 0.0]).unwrap();
    let (fc, _) = fraunhofer_zoom(&u, zf, far, &mut fft).unwrap();
    let (f0, _) = fraunhofer_zoom(&u0, zf, far, &mut fft).unwrap();
    let lam = 0.6328e-3;
    let peak = f0.data.iter().map(|q| q.abs()).fold(0.0, f64::max);
    let mut worst: f64 = 0.0;
    for j in 0..32 {
        for i in 0..32 {
            let (x, y) = far.xy(i, j);
            let shift = Cx::expi(-2.0 * std::f64::consts::PI * (x * c[0] + y * c[1]) / (lam * zf));
            worst = worst.max((fc.at(i, j) - f0.at(i, j) * shift).abs());
        }
    }
    measured("zoomed far field of a shifted field against the centred one's times the shift phase, worst / peak", worst / peak);
    assert!(worst / peak < 1e-10, "{}", worst / peak);
}

#[test]
fn a_baked_carrier_propagates_like_an_analytic_one() {
    // R06 of the expert review (gauge): the analytic carrier is a
    // representation, not physics. A field with its carrier analytic and the
    // same field with the carrier folded into its samples (`bake_carrier`)
    // must propagate to the same field, by `Auto` (the band-limited angular
    // spectrum here) and by the zoomed angular spectrum, to round-off, where
    // the tilt is well sampled.
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_zoom, propagate, Cx, Fft2, Grid2, Propagator};
    let pitch = 1.0 / 128.0;
    let g = Grid2::square(256, pitch).unwrap();
    let mut a = GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
    a.carrier = [0.02, -0.01];
    let mut b = a.clone();
    b.bake_carrier();
    let mut fft = Fft2::new();
    let z = 5.0;
    let (pa, _) = propagate(&a, z, Propagator::Auto, &mut fft).unwrap();
    let (pb, _) = propagate(&b, z, Propagator::Auto, &mut fft).unwrap();
    let mut qa = pa.clone();
    qa.bake_carrier();
    let peak = qa.data.iter().map(|q| q.abs()).fold(0.0, f64::max);
    let worst = qa.data.iter().zip(&pb.data).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max);
    measured("analytic against baked carrier, Auto, worst / peak", worst / peak);
    assert!(worst / peak < 1e-10, "{}", worst / peak);
    let out = Grid2::new([64, 64], [0.004, 0.004], [z * 0.02, -z * 0.01]).unwrap();
    let (za, _) = asm_zoom(&a, z, out, 2, &mut fft).unwrap();
    let (zb, _) = asm_zoom(&b, z, out, 2, &mut fft).unwrap();
    let k = za.k_mm();
    let mut worst: f64 = 0.0;
    for j in 0..64 {
        for i in 0..64 {
            let (x, y) = out.xy(i, j);
            let full = za.at(i, j) * Cx::expi(k * (za.carrier[0] * x + za.carrier[1] * y));
            worst = worst.max((full - zb.at(i, j)).abs());
        }
    }
    measured("analytic against baked carrier, zoomed angular spectrum, worst / peak", worst / peak);
    assert!(worst / peak < 1e-10, "{}", worst / peak);
}

#[test]
fn the_regime_indicators_read_the_spectrum_as_the_step_pads_it() {
    // Q7 of the expert review: on a padded axis of P samples of pitch Δx the
    // spectrum is sampled 1/(PΔx) apart, P the padded length itself
    // (next_fast_len(2n), here 540 for n = 257, not 514), axis by axis. The
    // band-limit indicator is then the field the band-limited step removes,
    // to round-off [2]: a 0.2 mm square aperture far past the critical
    // distance, on a 257 × 200 window.
    use oc_wave::elements::{apply_aperture, Shape};
    let g = Grid2::new([257, 200], [0.004, 0.005], [0.0, 0.0]).unwrap();
    let mut u = Field2::from_fn(g, 0.5, |_, _| oc_wave::Cx::ONE).unwrap();
    let mut r0 = WaveReport::default();
    apply_aperture(&mut u, &Shape::Rect { c: [0.0, 0.0], h: [0.1, 0.1] }, false, &mut r0).unwrap();
    let mut fft = Fft2::new();
    let z = 400.0;
    assert!(z > oc_wave::propagate::padded_critical_distance_mm(&g, 0.5e-3));
    let (_, r) = propagate(&u, z, Propagator::AngularSpectrum(AsmOptions::default()), &mut fft).unwrap();
    let removed = (r.ledger.band_limit_removed / r.ledger.launched).sqrt();
    let indicator = oc_wave::propagate::band_limit_error(&u, z, &mut fft);
    measured("band-limit indicator against the field the step removes, 257 × 200 window, relative", indicator / removed - 1.0);
    assert!(removed > 1e-2, "{removed}");
    assert!((indicator / removed - 1.0).abs() < 1e-9, "{indicator} vs {removed}");
}

#[test]
fn an_evanescent_wave_counts_by_the_difference_of_the_two_transfer_factors() {
    // Q6 of the expert review: the exact propagator damps an evanescent
    // plane wave by e^{−z√(|q|² − k²)}, the paraxial one carries it with
    // unit modulus, so the identity of bundle M13 holds with
    // |e^{−zκ} − e^{izκ₂}|² for it, not with the whole bin. A Gaussian with
    // a weak grating of period 0.8λ on it (evanescent), 3 µm wide on a
    // 25.6 µm window (1.5e-8 of its amplitude at the edge, so the window
    // holds the whole difference), both propagators on the same padded
    // grid, 2 µm apart: the indicator against their difference.
    let g = Grid2::square(256, 0.0001).unwrap();
    let lam_mm = 0.5e-3;
    let mut u = gaussian(g, 0.5, 1.0, 0.003, 0.0, 1.0, [0.0, 0.0]).unwrap();
    let period = 0.8 * lam_mm;
    let f = (1.0 / period / g.freq_pitch(0)).round() * g.freq_pitch(0);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, _) = g.xy(i, j);
            let idx = g.idx(i, j);
            u.data[idx] *= oc_wave::Cx::ONE + oc_wave::Cx::real(0.3) * oc_wave::Cx::expi(2.0 * std::f64::consts::PI * f * x);
        }
    }
    let mut fft = Fft2::new();
    let z = 0.002;
    let exact = Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 2 });
    let (a, _) = propagate(&u, z, exact, &mut fft).unwrap();
    let (b, _) = propagate(&u, z, Propagator::FresnelTf, &mut fft).unwrap();
    let diff: f64 = a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>().sqrt();
    let norm: f64 = u.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let want = oc_wave::propagate::paraxial_error(&u, z, &mut fft);
    measured("evanescent grating: the propagators' difference against the indicator, relative", diff / norm / want - 1.0);
    assert!(want > 0.1, "{want}");
    assert!((diff / norm / want - 1.0).abs() < 1e-6, "{} vs {want}", diff / norm);
}

#[test]
fn a_step_neither_candidate_holds_is_run_but_not_certified() {
    // Q6 of the expert review: the two indicators are model errors, not
    // certified totals; where both exceed the tolerance (1e-2 of the field)
    // the step runs with the smaller and the report says it is not
    // certified. A 3 µm slit at 0.5 µm diffracts over ±10°: 50 mm on, the
    // band limit cuts most of its spectrum and the paraxial step is wrong
    // across it.
    use oc_wave::elements::{apply_aperture, Shape};
    let g = Grid2::square(512, 0.0005).unwrap();
    let mut u = Field2::from_fn(g, 0.5, |_, _| oc_wave::Cx::ONE).unwrap();
    let mut r0 = WaveReport::default();
    apply_aperture(&mut u, &Shape::Rect { c: [0.0, 0.0], h: [0.0015, 0.1] }, false, &mut r0).unwrap();
    let mut fft = Fft2::new();
    let z = 50.0;
    let (ep, eb) = (oc_wave::propagate::paraxial_error(&u, z, &mut fft), oc_wave::propagate::band_limit_error(&u, z, &mut fft));
    measured("3 µm slit at 50 mm: the paraxial step's indicator", ep);
    measured("3 µm slit at 50 mm: the band limit's indicator", eb);
    assert!(ep > oc_wave::propagate::STEP_TOLERANCE && eb > oc_wave::propagate::STEP_TOLERANCE, "{ep} {eb}");
    let (_, r) = propagate(&u, z, Propagator::Auto, &mut fft).unwrap();
    assert!(
        r.warnings.iter().any(|w| matches!(w, SamplingWarning::OutsideRegime { reason, .. } if reason.contains("not certified"))),
        "{:?}",
        r.warnings
    );
}

#[test]
fn the_rotated_spectrum_converges_over_tilt_and_bandwidth() {
    // Q8 of the expert review: the rotated spectrum against the plane waves
    // of the same padded spectrum summed directly at the plane's points
    // [6], over tilts of 0° to 60° and two beam widths (the narrower beam's
    // spectrum four times as wide). The kernel reads the spectrum between
    // its samples; nothing else separates the two.
    use oc_core::{Isometry3, Vec3};
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_tilted, Cx, Fft2, Grid2};
    let pitch = 1.0 / 128.0;
    let g = Grid2::square(128, pitch).unwrap();
    let mut fft = Fft2::new();
    let out = Grid2::new([40, 40], [0.01, 0.01], [0.0, 0.0]).unwrap();
    let lam = 0.6328e-3;
    let p = 256usize;
    let df = 1.0 / (p as f64 * pitch);
    for w0 in [0.1, 0.025] {
        let u = GaussianBeam { w0_mm: w0, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
        let mut buf = vec![Cx::ZERO; p * p];
        for j in 0..128 {
            for i in 0..128 {
                buf[(j + 64) * p + i + 64] = u.at(i, j);
            }
        }
        oc_wave::ifftshift2(&mut buf, [p, p]);
        fft.forward(&mut buf, [p, p], pitch * pitch);
        oc_wave::fftshift2(&mut buf, [p, p]);
        for deg in [0.0f64, 20.0, 40.0, 60.0] {
            let q = glam::DQuat::from_rotation_y(deg.to_radians());
            let target = Isometry3::new(Vec3::new(0.0, 0.0, 1.0), [q.x, q.y, q.z, q.w]).unwrap();
            let (v, rep) = asm_tilted(&u, target, out, 2, &mut fft).unwrap();
            let k = v.k_mm();
            let (mut worst, mut peak): (f64, f64) = (0.0, 0.0);
            for j in (0..40).step_by(4) {
                for i in (0..40).step_by(4) {
                    let (x, y) = out.xy(i, j);
                    let r = target.transform_point(Vec3::new(x, y, 0.0));
                    let mut s = Cx::ZERO;
                    for ky in 0..p {
                        let fy = (ky as f64 - (p / 2) as f64) * df;
                        for kx in 0..p {
                            let a = buf[ky * p + kx];
                            if a.norm_sqr() < 1e-30 {
                                continue;
                            }
                            let fx = (kx as f64 - (p / 2) as f64) * df;
                            let w2 = 1.0 / (lam * lam) - fx * fx - fy * fy;
                            if w2 <= 0.0 {
                                continue;
                            }
                            s += a * Cx::expi(2.0 * std::f64::consts::PI * (fx * r.x + fy * r.y + w2.sqrt() * r.z));
                        }
                    }
                    let s = s * (df * df);
                    let full = v.at(i, j) * Cx::expi(k * (v.carrier[0] * x + v.carrier[1] * y));
                    peak = peak.max(s.abs());
                    worst = worst.max((full - s).abs());
                }
            }
            measured(&format!("rotated spectrum, w₀ {w0} mm, plane tilted {deg}°: against the direct plane-wave sum, worst / peak"), worst / peak);
            assert!(worst / peak < 1e-7, "{w0} {deg}: {}", worst / peak);
            assert!(rep.inconsistencies.is_empty(), "{:?}", rep.inconsistencies);
        }
    }
}

#[test]
fn the_rotated_spectrum_books_what_the_plane_cannot_hold_by_cause() {
    // Q8 of the expert review: missing reads are booked apart. A field whose
    // grating of period 0.8λ is evanescent (ε = 0.3 of the amplitude, so
    // ε²/(1 + ε²) of the power) cannot reach a plane 10 µm on: that share
    // is `evanescent`, not lost from the window. A plane tilted 87° from the
    // beam is crossed at grazing incidence, and the report warns that the
    // flux there is not modelled; at 20° it does not.
    use oc_core::{Isometry3, Vec3};
    use oc_wave::{asm_tilted, Cx, Fft2, Grid2};
    let g = Grid2::square(256, 0.0001).unwrap();
    let mut u = gaussian(g, 0.5, 1.0, 0.003, 0.0, 1.0, [0.0, 0.0]).unwrap();
    let f = (1.0 / (0.8 * 0.5e-3) / g.freq_pitch(0)).round() * g.freq_pitch(0);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, _) = g.xy(i, j);
            let idx = g.idx(i, j);
            u.data[idx] *= Cx::ONE + Cx::real(0.3) * Cx::expi(2.0 * std::f64::consts::PI * f * x);
        }
    }
    let mut fft = Fft2::new();
    let out = Grid2::new([128, 128], [0.0002, 0.0002], [0.0, 0.0]).unwrap();
    let q = glam::DQuat::from_rotation_y(20f64.to_radians());
    let target = Isometry3::new(Vec3::new(0.0, 0.0, 0.01), [q.x, q.y, q.z, q.w]).unwrap();
    let (_, rep) = asm_tilted(&u, target, out, 2, &mut fft).unwrap();
    let want = 0.09 / 1.09;
    let got = rep.ledger.evanescent / rep.ledger.launched;
    measured("rotated spectrum: the evanescent grating's share booked as evanescent, against ε²/(1 + ε²)", got - want);
    assert!((got - want).abs() < 1e-6, "{got} vs {want}: {:?}", rep.ledger);
    assert!(!rep.warnings.iter().any(|w| matches!(w, SamplingWarning::OutsideRegime { .. })), "{:?}", rep.warnings);
    let beam = oc_wave::beams::GaussianBeam { w0_mm: 0.1, lambda_um: 0.6328, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(Grid2::square(128, 1.0 / 128.0).unwrap(), 0.0).unwrap();
    let q = glam::DQuat::from_rotation_y(87f64.to_radians());
    let grazing = Isometry3::new(Vec3::new(0.0, 0.0, 0.5), [q.x, q.y, q.z, q.w]).unwrap();
    let (_, rep) = asm_tilted(&beam, grazing, Grid2::new([64, 64], [0.02, 0.02], [0.0, 0.0]).unwrap(), 2, &mut fft).unwrap();
    assert!(
        rep.warnings.iter().any(|w| matches!(w, SamplingWarning::OutsideRegime { reason, .. } if reason.contains("grazing"))),
        "{:?}",
        rep.warnings
    );
}

#[test]
fn a_small_field_read_onto_a_wide_window_is_there_once() {
    // The readout of a focal spot onto a detector many times its window. The
    // zoomed spectrum is sampled 1/P apart, P the padded window, so its sum
    // repeats the field every P (Goodman [1]: a sampled spectrum is a
    // periodic field). Only the period about the field's window is the field;
    // beyond it a window must read zero, not the field's copies, which put
    // 2300 times a line's power on the app's default detector. At z = 0 the
    // read is the field's band-limited interpolation, so a Gaussian 10 µm in
    // radius read at 2 µm points keeps its power; through a plane tilted 20°
    // the power crossing it (the irradiance times the carrier's obliquity)
    // is the field's too.
    use oc_core::{Isometry3, Vec3};
    use oc_wave::beams::GaussianBeam;
    use oc_wave::{asm_tilted, asm_zoom, Fft2, Grid2};
    let g = Grid2::square(64, 0.001).unwrap();
    let u = GaussianBeam { w0_mm: 0.01, lambda_um: 0.42, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
    let mut fft = Fft2::new();
    // 2 mm across: the field padded twice repeats every 0.128 mm.
    let wide = Grid2::square(1000, 0.002).unwrap();
    let half = 0.5 * 0.128 + 1e-9;
    let th = 20f64.to_radians();
    let q = glam::DQuat::from_rotation_y(th);
    let tilted = Isometry3::new(Vec3::new(0.0, 0.0, 0.05), [q.x, q.y, q.z, q.w]).unwrap();
    let (zoomed, zrep) = asm_zoom(&u, 0.0, wide, 2, &mut fft).unwrap();
    let (turned, trep) = asm_tilted(&u, tilted, wide, 2, &mut fft).unwrap();
    for (name, v, through) in [("zoomed", &zoomed, zrep.ledger.detected), ("tilted 20°", &turned, trep.ledger.detected)] {
        let mut beyond: f64 = 0.0;
        for j in 0..1000 {
            for i in 0..1000 {
                let (x, y) = wide.xy(i, j);
                if x.abs() > half || y.abs() > half {
                    beyond = beyond.max(v.at(i, j).abs());
                }
            }
        }
        let kept = through / u.power();
        eprintln!("measured: {name}: power through a window 15 periods wide, of the field's = {kept:.9}; largest sample beyond the period = {beyond:.1e}");
        assert_eq!(beyond, 0.0, "{name}");
        assert!((kept - 1.0).abs() < 1e-4, "{name}: {kept}");
    }
}
