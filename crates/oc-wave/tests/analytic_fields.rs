//! Propagated fields against closed-form results from the literature
//! (acceptance cases C38–C42).
//!
//! Sources, numbered as in the Sources list of docs/wave-optics.md (each test
//! names the one it uses):
//! - [1] J. W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H.
//!   Freeman/Macmillan Learning (2017), ISBN 978-1-319-11916-4: Fraunhofer
//!   patterns of the rectangular and circular apertures; Fresnel diffraction
//!   of a square aperture through the Fresnel integrals; the thin lens.
//! - [5] W. H. Heurtley, "Scalar Rayleigh–Sommerfeld and Kirchhoff diffraction
//!   integrals: a comparison of exact evaluations for axial points," *J. Opt.
//!   Soc. Am.* 63(8), 1003–1008 (1973): the exact on-axis field of a circular
//!   aperture under the first Rayleigh–Sommerfeld solution.
//! - [13] H. Kogelnik, T. Li, "Laser beams and resonators," *Appl. Opt.* 5(10),
//!   1550–1567 (1966): Gaussian beam radius, curvature and Gouy phase.
//! - [14] S. A. Self, "Focusing of spherical Gaussian beams," *Appl. Opt.*
//!   22(5), 658–661 (1983): the thin-lens transformation of a Gaussian waist.
//! - [12] J. Wen, Y. Zhang, M. Xiao, "The Talbot effect: recent advances in
//!   classical optics, nonlinear optics, and quantum optics," *Adv. Opt.
//!   Photon.* 5, 83–130 (2013): self-imaging; for a field of three plane waves
//!   the revival distance is exactly `λ / (1 − √(1 − λ²/p²))`, paraxially
//!   `2p²/λ`.
//! - [15] NIST Digital Library of Mathematical Functions, ch. 7 (Fresnel
//!   integrals) and ch. 10 (Bessel functions), https://dlmf.nist.gov.
//! - [16] M. Born, E. Wolf, *Principles of Optics*, 7th ed., Cambridge
//!   University Press (1999), the Airy pattern: first zero at
//!   `v = 3.8317`, `1.21967 λz/D`; encircled energy `1 − J₀²(v) − J₁²(v)`
//!   (section number to be confirmed against the printed edition).
//!
//! Bibliographic data retrieved 2026-10-19.

mod common;

use common::{measured, peak, second_moment_radius_x};
use oc_wave::beams::{gaussian, gaussian_radius_mm, rayleigh_range_mm, GaussianBeam};
use oc_wave::elements::{apply_aperture, apply_thin_lens, Shape};
use oc_wave::special::{bessel_j, fresnel_cs};
use oc_wave::{propagate, AsmOptions, Cx, Field2, Fft2, Grid2, Propagator, WaveReport};
use std::f64::consts::PI;

fn uniform(g: Grid2, lambda_um: f64) -> Field2 {
    Field2::from_fn(g, lambda_um, |_, _| Cx::ONE).unwrap()
}

#[test]
fn fresnel_integrals_and_bessel_functions_match_the_tables() {
    // [15] DLMF ch. 7: C(1) = 0.7798934004, S(1) = 0.4382591474; both tend
    // to 1/2. Ch. 10: J₀(1) = 0.7651976866, J₁(1) = 0.4400505857, and the
    // first zero of J₁ is 3.8317059702.
    let (c, s) = fresnel_cs(1.0);
    assert!((c - 0.779_893_400_4).abs() < 1e-10, "{c}");
    assert!((s - 0.438_259_147_4).abs() < 1e-10, "{s}");
    let (c, s) = fresnel_cs(-1.0);
    assert!((c + 0.779_893_400_4).abs() < 1e-10 && (s + 0.438_259_147_4).abs() < 1e-10);
    // DLMF 7.5.3–7.5.4 with the auxiliary functions' leading terms (DLMF
    // 7.12.2–7.12.3), f ~ 1/(πx), g ~ 1/(π²x³), which leave an error of
    // order x⁻⁵: C = 1/2 + f sin(πx²/2) − g cos(πx²/2),
    // S = 1/2 − f cos(πx²/2) − g sin(πx²/2).
    let x = 40.0;
    let (c, s) = fresnel_cs(x);
    let (f, g) = (1.0 / (PI * x), 1.0 / (PI * PI * x * x * x));
    let ph = 0.5 * PI * x * x;
    assert!((c - (0.5 + f * ph.sin() - g * ph.cos())).abs() < 1e-8, "{c}");
    assert!((s - (0.5 - f * ph.cos() - g * ph.sin())).abs() < 1e-8, "{s}");
    assert!((bessel_j(0, 1.0) - 0.765_197_686_6).abs() < 1e-10);
    assert!((bessel_j(1, 1.0) - 0.440_050_585_7).abs() < 1e-10);
    assert!(bessel_j(1, 3.831_705_970_2).abs() < 1e-10);
}

#[test]
fn a_square_aperture_far_field_is_sinc_squared() {
    // [1] Fraunhofer: a w×w square, unit amplitude, gives
    // U₂ = e^{ikz} e^{ik r²/2z}/(iλz) · w² sinc(wx/λz) sinc(wy/λz).
    // 33 samples across, edges on cell boundaries, so the aperture is exact.
    let (lam, dx) = (0.5, 0.2 / 33.0);
    let w = 33.0 * dx;
    let g = Grid2::square(256, dx).unwrap();
    let mut u = uniform(g, lam);
    let mut r = WaveReport::default();
    apply_aperture(&mut u, &Shape::Rect { c: [0.0, 0.0], h: [0.5 * w, 0.5 * w] }, false, &mut r).unwrap();
    let z = 1000.0; // source Fresnel number (w/√2)²/(λz) = 0.04
    let (v, rep) = propagate(&u, z, Propagator::Fraunhofer { pad: 2 }, &mut Fft2::new()).unwrap();
    assert!(rep.warnings.is_empty(), "{:?}", rep.warnings);
    let lz = lam / 1000.0 * z;
    let k = 2.0 * PI / (lam / 1000.0);
    let sinc = |t: f64| if t == 0.0 { 1.0 } else { (PI * t).sin() / (PI * t) };
    let pk = w * w / lz;
    let (mut sq, mut cnt, mut worst) = (0.0, 0.0, 0.0f64);
    for j in 0..v.grid.n[1] {
        for i in 0..v.grid.n[0] {
            let (x, y) = v.grid.xy(i, j);
            if x.abs() > 5.0 * lz / w || y.abs() > 5.0 * lz / w {
                continue;
            }
            let want = Cx::expi(k * z + k * (x * x + y * y) / (2.0 * z)) / Cx::new(0.0, lz)
                * (w * w * sinc(w * x / lz) * sinc(w * y / lz));
            let e = (v.at(i, j) - want).abs() / pk;
            sq += e * e;
            cnt += 1.0;
            worst = worst.max(e);
        }
    }
    let rms = (sq / cnt).sqrt();
    measured("square aperture far field, rms complex error / peak", rms);
    measured("square aperture far field, worst complex error / peak", worst);
    assert!(rms < 1e-3, "{rms}");
}

#[test]
fn a_circular_aperture_makes_the_airy_pattern() {
    // [1], [16]: U₂ ∝ (πD²/4)·2J₁(v)/v, v = πDr/(λz); first zero at
    // 1.21967 λz/D; 83.78% of the power inside it.
    let (lam, d) = (0.5, 0.2);
    let dx = d / 64.0;
    let g = Grid2::square(1024, dx).unwrap();
    let mut u = uniform(g, lam);
    let mut r = WaveReport::default();
    apply_aperture(&mut u, &Shape::Disc { c: [0.0, 0.0], r: 0.5 * d }, false, &mut r).unwrap();
    let z = 1000.0;
    let (v, _) = propagate(&u, z, Propagator::Fraunhofer { pad: 2 }, &mut Fft2::new()).unwrap();
    let lz = lam / 1000.0 * z;
    let k = 2.0 * PI / (lam / 1000.0);
    let area = PI * d * d / 4.0;
    let pk = area / lz;
    let r1 = 1.219_67 * lz / d;
    // Profile along +x through the centre, out to four first-zero radii.
    let jc = v.grid.n[1] / 2;
    let mut worst: f64 = 0.0;
    for i in v.grid.n[0] / 2..v.grid.n[0] {
        let (x, _) = v.grid.xy(i, jc);
        if x > 4.0 * r1 {
            break;
        }
        let vv = PI * d * x / lz;
        let airy = if vv == 0.0 { 1.0 } else { 2.0 * bessel_j(1, vv) / vv };
        let want = Cx::expi(k * z + k * x * x / (2.0 * z)) / Cx::new(0.0, lz) * (area * airy);
        worst = worst.max((v.at(i, jc) - want).abs() / pk);
    }
    measured("Airy profile, worst complex error / peak", worst);
    assert!(worst < 2e-3, "{worst}");
    // Encircled energy inside the first dark ring, as a fraction of the power
    // the aperture transmits, πD²/4 for a unit field. (Not of the sampled
    // field's power: the sample at the aperture's edge carries less than the
    // cell transmits, which the ledger books as `subcell_projection`.)
    let total = area;
    assert!(r.ledger.subcell_projection > 0.0);
    let mut inside = 0.0;
    for j in 0..v.grid.n[1] {
        for i in 0..v.grid.n[0] {
            let (x, y) = v.grid.xy(i, j);
            if x * x + y * y < r1 * r1 {
                inside += v.at(i, j).norm_sqr();
            }
        }
    }
    let ee = inside * v.grid.cell_area_mm2() / total;
    let v1 = 3.831_705_970_2;
    let want = 1.0 - bessel_j(0, v1).powi(2) - bessel_j(1, v1).powi(2);
    measured("Airy encircled energy at the first dark ring", ee);
    assert!((want - 0.8378).abs() < 1e-4);
    assert!((ee - want).abs() < 1e-3, "{ee} vs {want}");
}

fn heurtley(lam_um: f64, a: f64, z: f64) -> Cx {
    // [5]: the first Rayleigh–Sommerfeld solution on the axis of a circular
    // aperture of radius a in a unit plane wave, exactly:
    // U(z) = e^{ikz} − (z/√(z² + a²)) e^{ik√(z² + a²)}.
    let k = 2.0 * PI / (lam_um / 1000.0);
    let rr = (z * z + a * a).sqrt();
    Cx::expi(k * z) - Cx::expi(k * rr) * (z / rr)
}

fn disc_field(n: usize, dx: f64, lam: f64, a: f64) -> Field2 {
    let mut u = uniform(Grid2::square(n, dx).unwrap(), lam);
    let mut r = WaveReport::default();
    apply_aperture(&mut u, &Shape::Disc { c: [0.0, 0.0], r: a }, false, &mut r).unwrap();
    u
}

#[test]
fn the_on_axis_field_of_a_circular_aperture_matches_heurtley() {
    // a = 0.1 mm at 0.5 µm, Fresnel numbers a²/(λz) from 10 (z = 2 mm) to
    // 0.25 (z = 80 mm).
    let (lam, a) = (0.5, 0.1);
    let lm = lam / 1000.0;
    let mut fft = Fft2::new();
    let fine = disc_field(1024, a / 256.0, lam, a);
    let coarse = disc_field(512, a / 128.0, lam, a);

    // Direct Rayleigh–Sommerfeld integration (Shen & Wang 2006) computes the
    // very integral Heurtley evaluated, over the sampled aperture, with no
    // band limit and no window to outgrow: on the axis its only error is the
    // aperture's area-weighted edge, which stands in for the true circle to
    // second order in Δx. The on-axis phase is sensitive to the radius in
    // proportion to the Fresnel number, so the near field is where that shows,
    // and halving Δx must cut it about fourfold.
    for nf in [10.0, 5.0, 2.5, 1.0, 0.5, 0.25] {
        let z = a * a / (lm * nf);
        let want = heurtley(lam, a, z);
        let (vf, rf) = propagate(&fine, z, Propagator::RsDirect, &mut fft).unwrap();
        assert!(rf.warnings.is_empty(), "N_F {nf}: {:?}", rf.warnings);
        let uf = vf.at(512, 512);
        let ef = (uf - want).abs();
        measured(&format!("Heurtley, direct RS, Fresnel number {nf}, a/Δx = 256: |ΔU|"), ef);
        assert!(ef < 5e-3, "N_F {nf}: {ef}");
        let (vc, _) = propagate(&coarse, z, Propagator::RsDirect, &mut fft).unwrap();
        let uc = vc.at(256, 256);
        // Richardson: with a second-order error, (4U(Δx/2) − U(Δx))/3 removes
        // the leading term, so the extrapolated value must sit far closer to
        // the exact one than either sampling does.
        let ur = (uf * 4.0 - uc) / 3.0;
        let er = (ur - want).abs();
        measured(&format!("Heurtley, direct RS, Fresnel number {nf}, Richardson-extrapolated |ΔU|"), er);
        if nf >= 2.5 {
            let ec = (uc - want).abs();
            measured(&format!("Heurtley, direct RS, Fresnel number {nf}, error ratio a/Δx 128 → 256"), ec / ef);
            assert!(ec / ef > 3.0, "N_F {nf}: not second order ({ec} → {ef})");
        }
        assert!(er < 3e-4, "N_F {nf}: extrapolated {er}");
    }

    // The band-limited angular spectrum, in the near field where its window
    // is ample: the same exact solution, sampled the other way.
    let asm = Propagator::AngularSpectrum(AsmOptions::default());
    let z = a * a / (lm * 10.0);
    let (v, _) = propagate(&fine, z, asm, &mut fft).unwrap();
    let e = (v.at(512, 512) - heurtley(lam, a, z)).abs();
    measured("Heurtley, band-limited ASM, Fresnel number 10, a/Δx = 256: |ΔU|", e);
    assert!(e < 3e-3, "{e}");

    // Toward the far field the angular spectrum needs a window far larger
    // than the aperture before its band limit stops cutting what reaches the
    // axis (Matsushima & Shimobaba 2009). Past the critical distance the
    // automatic choice is the Fresnel impulse response, in its own regime,
    // and that must land on the exact solution too.
    for nf in [1.0, 0.5, 0.25] {
        let z = a * a / (lm * nf);
        let (v, r) = propagate(&coarse, z, Propagator::Auto, &mut fft).unwrap();
        assert_eq!(r.steps[0].method, oc_wave::Method::FresnelIr, "N_F {nf}");
        let e = (v.at(256, 256) - heurtley(lam, a, z)).abs();
        measured(&format!("Heurtley, Auto (impulse response), Fresnel number {nf}: |ΔU|"), e);
        assert!(e < 3e-3, "N_F {nf}: {e}");
    }
}

#[test]
fn a_square_aperture_fresnel_pattern_matches_the_fresnel_integrals() {
    // [1]: U = e^{ikz}/i · I(x)·I(y), with
    // I(x) = [(C(α₂) − C(α₁)) + i(S(α₂) − S(α₁))]/√2,
    // α₁ = −√(2/λz)(w/2 + x), α₂ = √(2/λz)(w/2 − x).
    // w = 0.5 mm at 0.5 µm and z = 50 mm: Fresnel number 2.5, the aperture's
    // edges on cell boundaries, the window five apertures wide.
    let (lam, w, z) = (0.5, 0.5, 50.0);
    let lm = lam / 1000.0;
    let k = 2.0 * PI / lm;
    let s = (2.0 / (lm * z)).sqrt();
    let i1 = |x: f64| {
        let (c1, s1) = fresnel_cs(-s * (0.5 * w + x));
        let (c2, s2) = fresnel_cs(s * (0.5 * w - x));
        Cx::new(c2 - c1, s2 - s1) / 2f64.sqrt()
    };
    let mut fft = Fft2::new();
    let mut run = |m: usize| -> f64 {
        let dx = w / m as f64;
        let n = 5 * m + (5 * m) % 2;
        let g = Grid2::square(n, dx).unwrap();
        let mut u = uniform(g, lam);
        let mut r = WaveReport::default();
        apply_aperture(&mut u, &Shape::Rect { c: [0.0, 0.0], h: [0.5 * w, 0.5 * w] }, false, &mut r).unwrap();
        // The Fresnel method in regime (Voelz & Roggemann 2009): the transfer
        // function up to the padded window's critical distance, the impulse
        // response beyond it.
        let lp = oc_wave::next_fast_len(2 * n) as f64 * dx;
        let how = if z <= oc_wave::propagate::critical_distance_mm(dx, lp, lm) {
            Propagator::FresnelTf
        } else {
            Propagator::FresnelIr
        };
        let (v, rep) = propagate(&u, z, how, &mut fft).unwrap();
        // The sampled sharp edge diffracts up to the grid's Nyquist angle,
        // which can reach the padded window's edge; the report flags that as
        // `WrapAround`, and the comparison is what says whether it matters.
        assert!(
            !rep.warnings.iter().any(|w| matches!(w, oc_wave::SamplingWarning::OutsideRegime { .. })),
            "{how:?}: {:?}",
            rep.warnings
        );
        let ix: Vec<Cx> = (0..n).map(|i| i1(g.coord(0, i))).collect();
        let (mut worst, mut pk) = (0.0f64, 0.0f64);
        for j in 0..n {
            for i in 0..n {
                let want = Cx::expi(k * z) / Cx::I * ix[i] * ix[j];
                pk = pk.max(want.abs());
                worst = worst.max((v.at(i, j) - want).abs());
            }
        }
        measured(&format!("Fresnel square, {m} samples across, {how:?}: worst complex error / peak"), worst / pk);
        worst / pk
    };
    let coarse = run(101);
    let fine = run(201);
    measured("Fresnel square, error ratio 101 → 201 samples across", coarse / fine);
    assert!(fine < 1.5e-3, "{fine}");
    assert!(coarse / fine > 3.0, "not second order: {coarse} → {fine}");
}

#[test]
fn a_gaussian_beam_spreads_as_kogelnik_and_li_say() {
    // [13]: w(z) = w₀√(1 + (z/z_R)²), with the complex field following the
    // q-parameter, Gouy phase included. w₀ = 0.2 mm at 0.6328 µm, z_R = 198.6
    // mm, carried 300 mm by the exact angular spectrum. The Gaussian is a
    // paraxial solution, so the exact propagator may differ from it by the
    // order of its squared divergence, (λ/πw₀)² ≈ 1e-6.
    let (lam, w0, z) = (0.6328, 0.2, 300.0);
    let g = Grid2::square(512, 0.01).unwrap();
    let u = gaussian(g, lam, 1.0, w0, 0.0, 1.0, [0.0, 0.0]).unwrap();
    let (v, _) = propagate(&u, z, Propagator::AngularSpectrum(AsmOptions::default()), &mut Fft2::new()).unwrap();
    let w = second_moment_radius_x(&v);
    let want = gaussian_radius_mm(w0, lam, 1.0, z);
    let rel = ((w - want) / want).abs();
    measured("Gaussian radius after 300 mm, relative error", rel);
    assert!(rel < 1e-5, "{rel}");
    let mut worst: f64 = 0.0;
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, y) = g.xy(i, j);
            let want = GaussianBeam::new(w0, lam, 1.0, 1.0).value(x, y, z);
            worst = worst.max((v.at(i, j) - want).abs());
        }
    }
    let rel = worst / peak(&v.data);
    measured("Gaussian complex field after 300 mm, worst error / peak", rel);
    assert!(rel < 1e-5, "{rel}");
}

#[test]
fn a_thin_lens_focuses_a_gaussian_where_self_predicts() {
    // [14] Self (1983): a waist w₀ a distance s before a thin lens f is imaged
    // to a waist at s″ with 1/(s + z_R²/(s − f)) + 1/s″ = 1/f and
    // w₀″ = w₀ / √((1 − s/f)² + (z_R/f)²).
    let (lam, w0, f, s) = (0.6328, 0.5, 200.0, 100.0);
    let zr = rayleigh_range_mm(w0, lam, 1.0);
    let s2 = 1.0 / (1.0 / f - 1.0 / (s + zr * zr / (s - f)));
    let w2 = w0 / ((1.0 - s / f).powi(2) + (zr / f).powi(2)).sqrt();
    // The same transformation through the q-parameter, as a cross-check on
    // the transcription of Self's formulas: q = z − i z_R here, and a thin
    // lens maps 1/q to 1/q − 1/f.
    let q_lens = Cx::new(s, -zr);
    let q_after = Cx::ONE / (Cx::ONE / q_lens - Cx::real(1.0 / f));
    assert!(((-q_after.re) - s2).abs() < 1e-9 * s2);
    let zr2 = -q_after.im;
    let w2_q = (zr2 * (lam / 1000.0) / PI).sqrt();
    assert!((w2_q - w2).abs() < 1e-12 * w2);

    let g = Grid2::square(512, 0.008).unwrap();
    let mut u = gaussian(g, lam, 1.0, w0, 0.0, 1.0, [0.0, 0.0]).unwrap();
    let mut fft = Fft2::new();
    let how = Propagator::AngularSpectrum(AsmOptions::default());
    u = propagate(&u, s, how, &mut fft).unwrap().0;
    let mut r = WaveReport::default();
    apply_thin_lens(&mut u, f, [0.0, 0.0], &mut r).unwrap();
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    let v = propagate(&u, s2, how, &mut fft).unwrap().0;
    let w = second_moment_radius_x(&v);
    let rel = ((w - w2) / w2).abs();
    measured("Self waist radius, relative error", rel);
    assert!(rel < 1e-4, "{rel}");
    // At the predicted waist the wavefront is flat: the phase across the
    // central lobe matches the centre's.
    let c = g.n[0] / 2;
    let ph0 = v.at(c, c).arg();
    let i_edge = c + (w2 / g.pitch_mm[0]) as usize;
    let dph = (v.at(i_edge, c) * Cx::expi(-ph0)).arg().abs();
    measured("phase at r = w₀″ relative to the centre, rad", dph);
    assert!(dph < 1e-3, "{dph}");
}

#[test]
fn a_cosine_grating_reimages_at_the_exact_talbot_distance() {
    // [12]: t(x) = (1 + cos 2πx/p)/2 is three plane waves; they rephase at
    // z_T = λ/(1 − √(1 − λ²/p²)), and at z_T/2 the pattern has moved half a
    // period. The window holds 8 periods and the propagation is periodic
    // (no padding), so the angular spectrum is exact here.
    let (lam, p) = (0.5, 0.05);
    let lm = lam / 1000.0;
    let g = Grid2::new([256, 4], [p / 32.0, p / 32.0], [0.0, 0.0]).unwrap();
    let t = |x: f64| 0.5 * (1.0 + (2.0 * PI * x / p).cos());
    let u = Field2::from_fn(g, lam, |x, _| Cx::real(t(x))).unwrap();
    let zt = lm / (1.0 - (1.0 - lm * lm / (p * p)).sqrt());
    let paraxial = 2.0 * p * p / lm;
    assert!((zt - paraxial).abs() > 1e-5, "the exact distance differs from 2p²/λ");
    let k = 2.0 * PI / lm;
    let periodic = Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 1 });
    let mut fft = Fft2::new();
    let (v, _) = propagate(&u, zt, periodic, &mut fft).unwrap();
    let mut worst: f64 = 0.0;
    for (a, b) in u.data.iter().zip(&v.data) {
        worst = worst.max((*a - *b * Cx::expi(-k * zt)).abs());
    }
    // The field carries the absolute phase k·z_T ≈ 1.3e5 rad, so round-off in
    // that phase alone is of order 1e-11; the gate sits above it.
    measured("Talbot revival, worst error", worst);
    assert!(worst < 1e-10, "{worst}");
    let (h, _) = propagate(&u, 0.5 * zt, periodic, &mut fft).unwrap();
    let mut worst: f64 = 0.0;
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, _) = g.xy(i, j);
            let want = Cx::real(t(x - 0.5 * p)) * Cx::expi(k * 0.5 * zt);
            worst = worst.max((h.at(i, j) - want).abs());
        }
    }
    measured("half-Talbot shift, worst error", worst);
    assert!(worst < 1e-10, "{worst}");
}
