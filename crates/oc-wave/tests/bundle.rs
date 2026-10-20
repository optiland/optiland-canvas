//! Cases adopted from the wave-optics specification bundle v1.0.0
//! (`config/benchmark_cases.json`, frozen; `01_MODEL.md` M34–M35), in the
//! canvas's units: millimetres, micrometres for wavelengths.
//!
//! Sources:
//! - The bundle's B02 (finite periodic modes, exact homogeneous transport),
//!   B10 (aperture throughput and operator order) and M35 (two slits).
//! - [1] J. W. Goodman, *Introduction to Fourier Optics*, 4th ed. (2017): the
//!   angular spectrum, exact in a homogeneous medium; the Fraunhofer pattern
//!   of two slits.
//! - [15] NIST DLMF, §10.12 (the Jacobi–Anger expansion,
//!   `e^{iβ cos θ} = Σ i^m J_m(β) e^{imθ}`) and §10.2 (the series for `J_m`).
//!
//! Numbered as in docs/wave-optics.md; retrieved 2026-10-19.

mod common;

use common::measured;
use oc_wave::{AsmOptions, Cx, Fft2, Field2, Grid2, Propagator};

/// B02's six modes and their two-component raw coefficients, in the order
/// the fixture lists them.
const MODES: [(i64, i64); 6] = [(0, 0), (1, 0), (-2, 1), (4, -3), (7, 5), (-8, 6)];
const RAW: [[[f64; 2]; 2]; 6] = [
    [[1.0, 0.0], [0.0, 0.5]],
    [[0.4, -0.3], [0.2, 0.1]],
    [[-0.2, 0.6], [0.0, 0.0]],
    [[0.3, 0.0], [0.0, 0.4]],
    [[0.1, -0.1], [-0.2, 0.3]],
    [[0.0, 0.5], [0.1, 0.0]],
];

/// The coefficients scaled so that the modes carry 1 (the fixture's 1 mW, in
/// the canvas's units of power), per component.
fn coefficients() -> [[Cx; 2]; 6] {
    let total: f64 = RAW.iter().flat_map(|c| c.iter()).map(|z| z[0] * z[0] + z[1] * z[1]).sum();
    let s = 1.0 / total.sqrt();
    RAW.map(|c| c.map(|z| Cx::new(z[0] * s, z[1] * s)))
}

/// The field of one component on a cell of `lengths` (mm) sampled `shape`,
/// each mode the periodic basis function `e^{iq·ρ}/√A` with
/// `q = 2π(m/L_x, n/L_y)`, its coefficient `c[mode]`, and each propagated
/// `d` by the diagonal exact transport `e^{id√(k² − |q|²)}`.
fn synthesis(shape: [usize; 2], lengths: [f64; 2], lam_um: f64, c: &[Cx], d: f64) -> Field2 {
    let g = Grid2::new(shape, [lengths[0] / shape[0] as f64, lengths[1] / shape[1] as f64], [0.0, 0.0]).unwrap();
    let mut u = Field2::zeros(g, lam_um).unwrap();
    let area = lengths[0] * lengths[1];
    let k = 2.0 * std::f64::consts::PI / (lam_um / 1000.0);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, y) = g.xy(i, j);
            let mut v = Cx::ZERO;
            for (m, (mx, my)) in MODES.iter().enumerate() {
                let q = [2.0 * std::f64::consts::PI * *mx as f64 / lengths[0], 2.0 * std::f64::consts::PI * *my as f64 / lengths[1]];
                let kz = (k * k - q[0] * q[0] - q[1] * q[1]).sqrt();
                v += c[m] * Cx::expi(q[0] * x + q[1] * y + kz * d);
            }
            u.data[g.idx(i, j)] = v * (1.0 / area.sqrt());
        }
    }
    u
}

fn rel(a: &Field2, b: &Field2) -> f64 {
    let d: f64 = a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).norm_sqr()).sum();
    (d / b.data.iter().map(|x| x.norm_sqr()).sum::<f64>()).sqrt()
}

#[test]
fn b02_finite_modes_are_carried_by_the_exact_diagonal_transport() {
    // The bundle's B02: six periodic modes of a 128 × 96 µm cell on 32 × 24
    // samples, λ0 = 550 nm. The angular spectrum without padding or band
    // limit is exact for a field periodic in its window [1]: against the
    // modes multiplied by `e^{id√(k² − |q|²)}` one by one, at 3, 7 and 10 mm,
    // to the bundle's finite-matrix tolerance 2e-11. Power per cell, the
    // semigroup (3 + 7 against 10) and the adjoint (forward, then back) are
    // the propagator's own consistency.
    let (shape, lengths, lam) = ([32, 24], [0.128, 0.096], 0.55);
    let c = coefficients();
    let periodic = Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 1 });
    let mut fft = Fft2::new();
    for comp in 0..2 {
        let cc: Vec<Cx> = c.iter().map(|m| m[comp]).collect();
        let u0 = synthesis(shape, lengths, lam, &cc, 0.0);
        for d in [3.0, 7.0, 10.0] {
            let (u, _) = oc_wave::propagate(&u0, d, periodic, &mut fft).unwrap();
            let want = synthesis(shape, lengths, lam, &cc, d);
            let e = rel(&u, &want);
            measured(&format!("B02, component {comp}, {d} mm: against the diagonal exact transport"), e);
            assert!(e < 2e-11, "{d}: {e}");
            assert!((u.power() - u0.power()).abs() < 1e-13 * u0.power(), "power per cell");
        }
        let (a, _) = oc_wave::propagate(&u0, 3.0, periodic, &mut fft).unwrap();
        let (a, _) = oc_wave::propagate(&a, 7.0, periodic, &mut fft).unwrap();
        let (b, _) = oc_wave::propagate(&u0, 10.0, periodic, &mut fft).unwrap();
        measured(&format!("B02, component {comp}: 3 mm then 7 mm against 10 mm"), rel(&a, &b));
        assert!(rel(&a, &b) < 2e-11);
        let (back, _) = oc_wave::propagate(&b, -10.0, periodic, &mut fft).unwrap();
        measured(&format!("B02, component {comp}: 10 mm and back against the field"), rel(&back, &u0));
        assert!(rel(&back, &u0) < 2e-11);
    }
}

#[test]
fn b02_the_fft_spectrum_is_the_explicit_finite_dft() {
    // B02's last check: the FFT the propagators use against the explicit
    // finite DFT `Σ_j u_j e^{−i2π(f·ρ_j)} ΔA` at every frequency of the grid.
    let (shape, lengths, lam) = ([32, 24], [0.128, 0.096], 0.55);
    let c = coefficients();
    let cc: Vec<Cx> = c.iter().map(|m| m[0]).collect();
    let u = synthesis(shape, lengths, lam, &cc, 0.0);
    let g = u.grid;
    let mut buf = u.data.clone();
    oc_wave::ifftshift2(&mut buf, g.n);
    Fft2::new().forward(&mut buf, g.n, g.cell_area_mm2());
    let mut worst: f64 = 0.0;
    let mut top: f64 = 0.0;
    for ky in 0..g.n[1] {
        for kx in 0..g.n[0] {
            let (fx, fy) = (g.fft_freq(0, kx), g.fft_freq(1, ky));
            let mut s = Cx::ZERO;
            for j in 0..g.n[1] {
                for i in 0..g.n[0] {
                    let (x, y) = g.xy(i, j);
                    s += u.at(i, j) * Cx::expi(-2.0 * std::f64::consts::PI * (fx * x + fy * y));
                }
            }
            s *= g.cell_area_mm2();
            worst = worst.max((buf[ky * g.n[0] + kx] - s).abs());
            top = top.max(s.abs());
        }
    }
    measured("B02: FFT spectrum against the explicit DFT, worst / peak", worst / top);
    assert!(worst < 2e-11 * top, "{worst} vs {top}");
}

/// The periodic quadratic (Fresnel) transfer function of B10's ordering
/// case, by the canvas's FFT or by an explicit finite DFT: `e^{ikd}` times
/// `e^{−iπλd|f|²}` on each frequency of the cell.
fn fresnel_periodic(u: &Field2, d: f64, explicit: bool) -> Field2 {
    let g = u.grid;
    let n = g.n;
    let lam = u.lambda_um / 1000.0;
    let k = 2.0 * std::f64::consts::PI / lam;
    let h = |fx: f64, fy: f64| Cx::expi(k * d - std::f64::consts::PI * lam * d * (fx * fx + fy * fy));
    let mut v = u.clone();
    if explicit {
        let mut spec = vec![Cx::ZERO; g.len()];
        for ky in 0..n[1] {
            for kx in 0..n[0] {
                let (fx, fy) = (g.fft_freq(0, kx), g.fft_freq(1, ky));
                let mut s = Cx::ZERO;
                for j in 0..n[1] {
                    for i in 0..n[0] {
                        let (x, y) = g.xy(i, j);
                        s += u.at(i, j) * Cx::expi(-2.0 * std::f64::consts::PI * (fx * x + fy * y));
                    }
                }
                spec[ky * n[0] + kx] = s * h(fx, fy);
            }
        }
        for j in 0..n[1] {
            for i in 0..n[0] {
                let (x, y) = g.xy(i, j);
                let mut s = Cx::ZERO;
                for ky in 0..n[1] {
                    for kx in 0..n[0] {
                        let (fx, fy) = (g.fft_freq(0, kx), g.fft_freq(1, ky));
                        s += spec[ky * n[0] + kx] * Cx::expi(2.0 * std::f64::consts::PI * (fx * x + fy * y));
                    }
                }
                v.data[g.idx(i, j)] = s * (1.0 / (n[0] * n[1]) as f64);
            }
        }
    } else {
        let mut b = u.data.clone();
        oc_wave::ifftshift2(&mut b, n);
        let mut fft = Fft2::new();
        fft.forward(&mut b, n, g.cell_area_mm2());
        for ky in 0..n[1] {
            for kx in 0..n[0] {
                b[ky * n[0] + kx] *= h(g.fft_freq(0, kx), g.fft_freq(1, ky));
            }
        }
        fft.inverse(&mut b, n, g.freq_pitch(0) * g.freq_pitch(1));
        oc_wave::fftshift2(&mut b, n);
        v.data = b;
    }
    v
}

#[test]
fn b10_an_aperture_acts_at_its_plane_and_the_order_matters() {
    // The bundle's B10, operator order: B02's coefficients on a 240 µm cell of
    // 24², a circle of 6 grid steps about index (12, 12) by the integer rule
    // `(i − 12)² + (j − 12)² ≤ 36`, 7 mm before it and 11 mm after, by the
    // quadratic Fresnel transfer function. `U_after M U_before` by the FFT
    // against the explicit finite DFT to 2e-11; the misplaced
    // `M U_after U_before` differs by the fixture's frozen 0.4329834561 of
    // the input's root power. The wave path lays an aperture at its plane
    // between two propagations (`path.rs`), the first order.
    let (shape, lengths, lam) = ([24usize, 24usize], [0.24, 0.24], 0.55);
    let c = coefficients();
    let cc: Vec<Cx> = c.iter().map(|m| m[0]).collect();
    let u = synthesis(shape, lengths, lam, &cc, 0.0);
    // The bundle's grid has index 0 at minus half the length; the canvas's
    // has sample n/2 on the centre, the same points for an even count.
    let g = u.grid;
    assert!((g.xy(0, 0).0 + 0.12).abs() < 1e-15, "{:?}", g.xy(0, 0));
    // B10 reuses B02's coefficients with this cell's q: the synthesis above
    // takes q from these lengths. Both components carry power; the fixture
    // normalizes the two together, so the first is scaled to its share.
    let p_in: f64 = u.power() + synthesis(shape, lengths, lam, &c.iter().map(|m| m[1]).collect::<Vec<_>>(), 0.0).power();
    let mask = |f: &mut Field2| {
        for j in 0..24 {
            for i in 0..24 {
                if (i as i64 - 12).pow(2) + (j as i64 - 12).pow(2) > 36 {
                    let k = f.grid.idx(i, j);
                    f.data[k] = Cx::ZERO;
                }
            }
        }
    };
    let both = |explicit: bool, comp: usize| -> (Field2, Field2) {
        let cc: Vec<Cx> = c.iter().map(|m| m[comp]).collect();
        let v = synthesis(shape, lengths, lam, &cc, 0.0);
        let mut a = fresnel_periodic(&v, 7.0, explicit);
        mask(&mut a);
        let a = fresnel_periodic(&a, 11.0, explicit);
        let mut b = fresnel_periodic(&fresnel_periodic(&v, 7.0, explicit), 11.0, explicit);
        mask(&mut b);
        (a, b)
    };
    let mut diff2 = 0.0;
    for comp in 0..2 {
        let (fa, fb) = both(false, comp);
        let (ea, _) = both(true, comp);
        let e = rel(&fa, &ea);
        measured(&format!("B10, component {comp}: U_after M U_before by FFT against the explicit DFT"), e);
        assert!(e < 2e-11, "{e}");
        // Power after the mask is kept by the unitary propagation after it.
        let mut at = fresnel_periodic(&synthesis(shape, lengths, lam, &c.iter().map(|m| m[comp]).collect::<Vec<_>>(), 0.0), 7.0, false);
        mask(&mut at);
        assert!((fa.power() - at.power()).abs() < 1e-13 * p_in, "{} vs {}", fa.power(), at.power());
        let cell = fa.grid.cell_area_mm2();
        diff2 += fa.data.iter().zip(&fb.data).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>() * cell;
    }
    let order = (diff2 / p_in).sqrt();
    let want = 0.43298345607270855;
    measured("B10: the misplaced order's difference / √P_in against the frozen 0.4329834561", order - want);
    assert!((order - want).abs() < 2e-11, "{order} vs {want}");
}

#[test]
fn b10_a_gaussian_through_a_circular_hole_passes_one_minus_e_to_the_minus_half() {
    // B10, continuum throughput (M34): a Gaussian of radius w = 0.1 mm
    // through a centred hole of a = 0.05 mm passes `1 − e^{−2a²/w²}`
    // = 0.393469340287 [1]. The bundle's tolerance, 2e-10, is for a
    // quadrature of the continuum; a mask on a grid has its own boundary
    // error. The canvas's exact cell coverage `c` stops `1 − c` of each
    // cell's power, the area outside the hole, and of what it lets through,
    // `c − c²` is the sub-cell projection error of the sampled edge
    // (`subcell_projection`): what the mask does not stop is the
    // continuum's throughput, less the boundary quadrature of a sampled
    // field, the Gaussian read at the centres of the cells its edge crosses:
    // 4.4e-5 at 2 µm samples, a quarter of it at 1 µm (second order).
    let want = 1.0 - (-0.5f64).exp();
    assert!((want - 0.3934693402873666).abs() < 1e-15);
    let mut errs = Vec::new();
    for (n, dx) in [(512usize, 0.002), (1024, 0.001)] {
        let g = Grid2::square(n, dx).unwrap();
        let mut u = oc_wave::beams::GaussianBeam { w0_mm: 0.1, lambda_um: 0.55, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
        let p0 = u.power();
        let cover = oc_wave::elements::coverage(&g, &oc_wave::elements::Shape::Disc { c: [0.0, 0.0], r: 0.05 }).unwrap();
        let (stopped, beyond) = oc_wave::elements::apply_coverage(&mut u, &cover).unwrap();
        let got = 1.0 - stopped / p0;
        measured(&format!("B10: Gaussian through a circular hole, exact cell coverage at {} µm, against 1 − e^(−1/2)", 1e3 * dx), got - want);
        measured(&format!("B10: of it, the sub-cell projection error at the edge, at {} µm", 1e3 * dx), beyond / p0);
        assert!((u.power() + beyond + stopped - p0).abs() < 1e-11 * p0, "the ledger closes: {}", (u.power() + beyond + stopped - p0) / p0);
        errs.push((got - want).abs());
    }
    assert!(errs[0] < 1e-4 && errs[1] < 0.35 * errs[0], "{errs:?}");
}

/// `J_m(x)` for `m ≥ 0` by its power series (DLMF 10.2.2), enough terms for
/// `x` of a few.
fn bessel_j(m: u32, x: f64) -> f64 {
    let mut term = (0.5 * x).powi(m as i32) / (1..=m).map(|k| k as f64).product::<f64>();
    let mut sum = term;
    for k in 1..60 {
        term *= -(0.25 * x * x) / (k as f64 * (k + m) as f64);
        sum += term;
    }
    sum
}

#[test]
fn a_sinusoidal_phase_mask_puts_j_m_squared_into_its_orders() {
    // Bundle M14 (`01_MODEL.md`): `exp(iβ cos Gx)` has the Fourier
    // coefficients `i^m J_m(β)` (Jacobi–Anger, DLMF 10.12.1 [15]), so a beam
    // much wider than the period puts `J_m(β)²` of its power into order m,
    // in the paraxial scalar model, with phase `i^m`. β = 2, a period of 32
    // samples, a Gaussian 8 periods wide in a window of 8 of its radii, so
    // that neither its window nor its neighbours cut it; the orders read
    // from the field's spectrum, out to the twelfth.
    let (n, dx, period) = (2048usize, 0.001, 32usize);
    let g = Grid2::square(n, dx).unwrap();
    let mut u = oc_wave::beams::GaussianBeam { w0_mm: 8.0 * period as f64 * dx, lambda_um: 0.55, n: 1.0, power: 1.0, center: [0.0, 0.0] }.sample(g, 0.0).unwrap();
    let beta = 2.0;
    let gk = 2.0 * std::f64::consts::PI / (period as f64 * dx);
    for j in 0..n {
        for i in 0..n {
            let (x, _) = g.xy(i, j);
            u.data[g.idx(i, j)] *= Cx::expi(beta * (gk * x).cos());
        }
    }
    let mut b = u.data.clone();
    oc_wave::ifftshift2(&mut b, g.n);
    Fft2::new().forward(&mut b, g.n, g.cell_area_mm2());
    // Order m sits m·n/period spectral samples from zero along x; its power
    // is the cell of samples about it, its phase the peak sample's against
    // order 0's.
    let df = g.freq_pitch(0) * g.freq_pitch(1);
    let total: f64 = b.iter().map(|z| z.norm_sqr()).sum::<f64>() * df;
    let half = (n / period / 2) as i64;
    let at = |kx: i64, ky: i64| b[(ky.rem_euclid(n as i64) as usize) * n + kx.rem_euclid(n as i64) as usize];
    let mut sum = 0.0;
    for m in -12i64..=12 {
        let c = m * (n / period) as i64;
        let mut p = 0.0;
        for ky in -half..half {
            for kx in c - half..c + half {
                p += at(kx, ky).norm_sqr();
            }
        }
        let p = p * df / total;
        let j = bessel_j(m.unsigned_abs() as u32, beta) * if m < 0 && m % 2 != 0 { -1.0 } else { 1.0 };
        sum += p;
        measured(&format!("phase mask β = 2, order {m}: power against J_m(β)²"), p - j * j);
        assert!((p - j * j).abs() < 1e-10, "{m}: {p} vs {}", j * j);
        if j.abs() > 1e-3 {
            let phase = (at(c, 0) / at(0, 0)).arg();
            let want = (Cx::expi(std::f64::consts::FRAC_PI_2 * m as f64) * (j / bessel_j(0, beta))).arg();
            let dphi = (phase - want + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
            // The Gaussian cut at four radii, amplitude e^(−16), leaks
            // between orders at about 1e-8 of the weakest's phase.
            assert!(dphi.abs() < 1e-7, "{m}: phase {phase} vs {want}");
        }
    }
    assert!((sum - 1.0).abs() < 1e-10, "{sum}");
}

#[test]
fn two_slits_make_the_far_field_of_m35() {
    // Bundle M35 [1]: two slits of width a, centres d apart, have the far
    // field `2a sinc(a q/2) cos(d q/2)`, `q = 2πx/(λz)`. Slits 40 µm wide,
    // 160 µm apart, on a 1 µm grid, moved half a cell so that their edges lie
    // on cell boundaries and their exact coverage is binary (a common shift
    // turns the far field's phase, not its magnitude), read from the field's
    // spectrum at its own
    // frequencies `f = q/2π`; the sampled mask's spectrum is the continuum's
    // times `(πfΔx)/sin(πfΔx)`, the sum over cells of a constant, which the
    // reference carries.
    let (n, dx) = (2048usize, 0.001);
    let g = Grid2::new([n, 4], [dx, dx], [0.0, 0.0]).unwrap();
    let (a, d) = (0.04, 0.16);
    let mut u = Field2::zeros(g, 0.55).unwrap();
    let s = 0.5 * dx;
    let slits = [oc_wave::elements::Shape::Rect { c: [-0.5 * d + s, 0.0], h: [0.5 * a, 1.0] }, oc_wave::elements::Shape::Rect { c: [0.5 * d + s, 0.0], h: [0.5 * a, 1.0] }];
    let cover: Vec<f64> = {
        let c0 = oc_wave::elements::coverage(&g, &slits[0]).unwrap();
        let c1 = oc_wave::elements::coverage(&g, &slits[1]).unwrap();
        c0.iter().zip(&c1).map(|(x, y)| x + y).collect()
    };
    for (z, c) in u.data.iter_mut().zip(&cover) {
        *z = Cx::real(*c);
    }
    let mut b = u.data.clone();
    oc_wave::ifftshift2(&mut b, g.n);
    Fft2::new().forward(&mut b, g.n, g.cell_area_mm2());
    let mut worst: f64 = 0.0;
    let peak = b[0].abs();
    for kx in 0..n {
        let f = g.fft_freq(0, kx);
        let q = 2.0 * std::f64::consts::PI * f;
        let sinc = |x: f64| if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
        let cell = sinc(std::f64::consts::PI * f * dx);
        let want = 2.0 * a * sinc(a * q / 2.0) * (d * q / 2.0).cos() / cell * (b[0].abs() / (2.0 * a));
        worst = worst.max((b[kx].abs() - want.abs()).abs());
    }
    measured("M35: two slits' spectrum against 2a sinc(aq/2) cos(dq/2), worst / peak", worst / peak);
    assert!(worst < 1e-9 * peak, "{worst} vs {peak}");
}
