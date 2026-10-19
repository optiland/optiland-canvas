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
    assert!(r.steps[0].note.contains("chosen on this field's errors"), "{}", r.steps[0].note);
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
    // source spectrum between its samples (bilinear), which is all that
    // separates the two; with no tilt it reads them on their samples and must
    // agree with the zoomed angular spectrum to round-off.
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
    let (t1, _) = asm_tilted(&u, tilted, out, 4, &mut fft).unwrap();
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
            worst = worst.max((t1.at(i, j) - s).abs());
        }
    }
    eprintln!("measured: tilted plane (20°), spectrum rotation against the direct plane-wave sum, worst / peak = {:.3e}", worst / peak);
    assert!(worst / peak < 2e-3, "{}", worst / peak);
}
