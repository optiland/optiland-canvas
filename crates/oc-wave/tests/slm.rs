//! The LCoS SLM model (acceptance cases C49–C51).
//!
//! Sources:
//! - [S1] Z. Zhang, Z. You, D. Chu, "Fundamentals of phase-only liquid crystal
//!   on silicon (LCOS) devices," *Light Sci. Appl.* 3, e213 (2014),
//!   doi:10.1038/lsa.2014.94: only the component along the director is
//!   modulated.
//! - [S2] G. J. Swanson, "Binary optics technology: the theory and design of
//!   multi-level diffractive optical elements," MIT Lincoln Laboratory TR-854
//!   (1989), DTIC ADA213404: an N-level phase staircase puts sinc²(1/N) into
//!   the first order (equation location to be confirmed against the report).
//! - [S3] H. Zhang et al., "Elimination of a zero-order beam induced by a
//!   pixelated spatial light modulator for holographic projection," *Appl.
//!   Opt.* 48(30), 5834 (2009), doi:10.1364/AO.48.005834: the dead space
//!   between pixels as a source of the zero order.
//! - [S4] M. Persson, D. Engström, M. Goksör, *Opt. Express* 20, 22334–22343
//!   (2012), doi:10.1364/OE.20.022334, and "Fringing-field effect in
//!   liquid-crystal beam-steering devices: an approximate analytical model,"
//!   *J. Opt. Soc. Am. A* 21(10), 1996–2008 (2004): crosstalk broadens a phase
//!   step into a linear ramp and costs a blazed grating efficiency.
//! - [S5] L. Allen, M. W. Beijersbergen, R. J. C. Spreeuw, J. P. Woerdman,
//!   *Phys. Rev. A* 45, 8185 (1992), doi:10.1103/PhysRevA.45.8185: a field
//!   carrying `exp(iℓφ)` has a dark core and its phase winds ℓ times.
//! - [S6] J. R. Fienup, "Phase retrieval algorithms: a comparison," *Appl.
//!   Opt.* 21(15), 2758–2769 (1982): the Gerchberg–Saxton error never rises.
//! - [S7] D. A. Pommet, M. G. Moharam, E. B. Grann, *J. Opt. Soc. Am. A*
//!   11(6), 1827 (1994), doi:10.1364/JOSAA.11.001827: scalar theory's error
//!   exceeds ±5% for features below 14λ.
//!
//! Retrieved 2026-10-19.

mod common;

use common::measured;
use oc_core::modulator::{
    sinc, unit_cell_orders, LcosBeam, LcosSpec, ModulatorKind, OrderModel, Pattern, PatternGenerator,
    PhaseLut, SpatialModulator, UnitCell,
};
use oc_core::Vec3;
use oc_wave::czt::czt2;
use oc_wave::design::gerchberg_saxton;
use oc_wave::modulator::{lcos_reflect, lcos_split};
use oc_wave::{propagate, Cx, Fft2, Field2, Grid2, Propagator, ValidityFlag};
use std::f64::consts::PI;

fn lcos(pixels: u32, pitch_um: f64, levels: u16, fill: f64, gap: (f64, f64), fringing_um: f64, pattern: Pattern) -> SpatialModulator {
    SpatialModulator {
        pixels: [pixels, pixels],
        pitch_um,
        kind: ModulatorKind::Lcos(LcosSpec {
            director_deg: 0.0,
            lut: PhaseLut::Linear { levels, max_phase_rad: 2.0 * PI },
            design_lambda_um: 0.633,
            pixel_fill: fill,
            gap_amplitude: gap.0,
            gap_phase_rad: gap.1,
            fringing_um,
            reflectance: 1.0,
        }),
        pattern,
    }
}

const NORMAL: Vec3 = Vec3::new(0.0, 0.0, -1.0);

fn staircase(n: usize) -> UnitCell {
    UnitCell { period: [n, 1], values: (0..n as u16).collect() }
}

fn order(t: &oc_core::modulator::OrderTable, m: i32, n: i32) -> f64 {
    t.orders.iter().find(|o| o.m == m && o.n == n).map(|o| o.efficiency).unwrap_or(0.0)
}

#[test]
fn an_n_level_blazed_grating_reaches_sinc_squared_one_over_n() {
    // [S2]: η₁ = sinc²(1/N): 0.40528, 0.81057, 0.94964, 0.98721 for N = 2, 4, 8, 16.
    for (n, published) in [(2usize, 0.40528), (4, 0.81057), (8, 0.94964), (16, 0.98721)] {
        let d = lcos(1, 8.0, n as u16, 1.0, (0.0, 0.0), 0.0, Pattern::Uniform { value: 0 });
        let t = unit_cell_orders(&d, &staircase(n), 0.633, NORMAL, OrderModel::Tea, LcosBeam::Modulated).unwrap();
        let eta = order(&t, 1, 0);
        measured(&format!("{n}-level blazed grating, first order"), eta);
        assert!((eta - sinc(1.0 / n as f64).powi(2)).abs() < 1e-12);
        assert!((eta - published).abs() < 5e-6);
    }
}

#[test]
fn fill_factor_and_dead_space_set_the_zero_order() {
    // Fourier series of the cell [S3]: with electrodes of linear fill ρ and
    // dead space of reflectance g, η₀₀ = |g|²(1 − ρ²)² and η₁₀ = ρ⁴ sinc²(ρ/N);
    // the dead space also diffracts at λ/p across the pixel rows (n ≠ 0).
    let (rho, gap, nlev) = (0.87, 0.6, 8usize);
    let d = lcos(1, 8.0, nlev as u16, rho, (gap, 0.4), 0.0, Pattern::Uniform { value: 0 });
    let t = unit_cell_orders(&d, &staircase(nlev), 0.633, NORMAL, OrderModel::Tea, LcosBeam::Modulated).unwrap();
    let z = order(&t, 0, 0);
    let f = order(&t, 1, 0);
    assert!((z - gap * gap * (1.0 - rho * rho).powi(2)).abs() < 1e-12, "{z}");
    assert!((f - rho.powi(4) * sinc(rho / nlev as f64).powi(2)).abs() < 1e-12, "{f}");
    assert!(order(&t, 0, 1) > 0.0, "the dead space diffracts across the rows");
    // Parseval: the total is ρ²R + (1 − ρ²)|g|².
    assert!((t.total - (rho * rho + (1.0 - rho * rho) * gap * gap)).abs() < 1e-15);
    measured("dead space zero order, ρ = 0.87, |g| = 0.6", z);
}

#[test]
fn a_kinoform_at_the_wrong_wavelength_loses_efficiency_as_sinc_squared() {
    // A 256-level blaze set at λ₀ used at λ: the phase scales as λ₀/λ (the
    // flagged first-order dispersion model), and the first order holds
    // sinc²(1 − λ₀/λ) in the continuous limit [S2]. The quantized value is the
    // geometric series (1/N)Σ e^{i2πj(s − 1)/N} times sinc(1/N), s = λ₀/λ.
    let n = 256usize;
    let d = lcos(1, 8.0, n as u16, 1.0, (0.0, 0.0), 0.0, Pattern::Uniform { value: 0 });
    for lam in [0.633 / 0.9, 0.633 / 1.1, 0.75] {
        let s = 0.633 / lam;
        let t = unit_cell_orders(&d, &staircase(n), lam, NORMAL, OrderModel::Tea, LcosBeam::Modulated).unwrap();
        let eta = order(&t, 1, 0);
        let x = 2.0 * PI * (s - 1.0) / n as f64;
        let series = Cx::new(1.0 - (n as f64 * x).cos(), -(n as f64 * x).sin())
            / Cx::new(1.0 - x.cos(), -x.sin())
            / n as f64;
        let quantized = series.norm_sqr() * sinc(1.0 / n as f64).powi(2);
        measured(&format!("kinoform at λ₀/λ = {s:.3}: η₁"), eta);
        assert!((eta - quantized).abs() < 1e-12, "{eta} vs {quantized}");
        assert!((eta - sinc(1.0 - s).powi(2)).abs() < 1e-4, "{eta} vs continuous");
    }
}

#[test]
fn the_lcos_modulates_only_the_director_component() {
    // [S1]: the director component carries cos²(ψ − δ) of the power and is
    // modulated; the rest, sin²(ψ − δ), reflects as from a plain mirror. The
    // two are orthogonally polarized and add in intensity. Each beam's power
    // is what it reflects onto the grid and beyond the grid's band (the
    // band-limited mask books the staircase's edges there).
    let d = lcos(16, 8.0, 256, 1.0, (0.0, 0.0), 0.0, Pattern::Generated(PatternGenerator::BlazedGrating { period_px: 8.0, angle_deg: 0.0 }));
    let (a, b) = lcos_split(&d, 30.0).unwrap();
    assert!((a * a - 30f64.to_radians().cos().powi(2)).abs() < 1e-15);
    assert!((a * a + b * b - 1.0).abs() < 1e-15);
    let g = Grid2::new([16 * 8, 16 * 8], [1e-3, 1e-3], [0.5e-3, 0.5e-3]).unwrap();
    let u = Field2::from_fn(g, 0.633, |_, _| Cx::ONE).unwrap();
    let mut um = u.clone();
    for z in &mut um.data {
        *z *= a;
    }
    let mut uo = u.clone();
    for z in &mut uo.data {
        *z *= b;
    }
    let (vm, rm) = lcos_reflect(&um, &d, LcosBeam::Modulated).unwrap();
    let (vo, ro) = lcos_reflect(&uo, &d, LcosBeam::Unmodulated).unwrap();
    let (pm, po) = (vm.power() + rm.ledger.spectral_projection, vo.power() + ro.ledger.spectral_projection);
    let total = pm + po;
    measured("director split: modulated share", pm / total);
    measured("director split: modulated beam beyond the grid's band, of its power", rm.ledger.spectral_projection / pm);
    assert!((pm / total - a * a).abs() < 1e-12, "{} vs {}", pm / total, a * a);
    assert!(rm.ledger.numerical.abs() < 1e-12 * rm.ledger.launched && ro.ledger.spectral_projection.abs() < 1e-12 * ro.ledger.launched);
    // The unmodulated beam is a plain mirror: its samples are all equal.
    let first = vo.data[0];
    assert!(vo.data.iter().all(|z| (*z - first).abs() < 1e-12));
}

#[test]
fn the_sampled_lcos_lands_on_the_closed_form_orders() {
    // An 8-level grating of 4 periods, electrodes of fill 0.75 with dead
    // space, sampled 8 per pixel. Over whole periods, the far field at an
    // order's direction is c_mn times the device's area. The mask is
    // band-limited exactly to the grid from its closed-form transform (Q17 of
    // the expert review), so at the window's own frequencies the near field's
    // transform is that closed form: on a window of 40 pixels the orders,
    // 1/(8p) apart, are among them, and the two must agree to round-off.
    let (np, per, nlev) = (32u32, 8usize, 8u16);
    let pitch_um = 8.0;
    let d = lcos(np, pitch_um, nlev, 0.75, (0.5, 0.3), 0.0, Pattern::Explicit(
        (0..np as usize * np as usize).map(|k| ((k % np as usize) % nlev as usize) as u16).collect(),
    ));
    let p = pitch_um / 1000.0;
    let dx = p / per as f64;
    // Even counts: pixel edges sit on integer multiples of Δx from the
    // centre, so the grid is offset half a sample to put its cell edges there.
    let ns = (np as usize + 8) * per;
    let g = Grid2::new([ns, ns], [dx, dx], [0.5 * dx, 0.5 * dx]).unwrap();
    let u = Field2::from_fn(g, 0.633, |_, _| Cx::ONE).unwrap();
    let (v, r) = lcos_reflect(&u, &d, LcosBeam::Modulated).unwrap();
    assert!(r.ledger.numerical.abs() < 1e-12 * r.ledger.launched && r.ledger.spectral_projection >= 0.0, "{:?}", r.ledger);
    measured("sampled LCoS, 8 samples a pixel: reflected beyond the grid's band, of the launched", r.ledger.spectral_projection / r.ledger.launched);
    let t = unit_cell_orders(&d, &UnitCell { period: [nlev as usize, 1], values: (0..nlev).collect() }, 0.633, NORMAL, OrderModel::Tea, LcosBeam::Modulated).unwrap();
    let area = (np as f64 * p).powi(2);
    let lm = 0.633 / 1000.0;
    let mut fft = Fft2::new();
    // The closed form puts its unit cell's origin at pixel 0; the device's
    // pixel 0 sits at r₀ = pixel_center(0, 0), which shifts every
    // coefficient's phase by e^{−i2πG·r₀}.
    let r0 = d.pixel_center_mm(0, 0);
    let mut worst: f64 = 0.0;
    for o in t.orders.iter().filter(|o| o.m.abs() <= 10 && o.n.abs() <= 2) {
        let f = [o.dir.x / lm, o.dir.y / lm];
        let s = czt2(&v, f, [1.0, 1.0], [1, 1], &mut fft)[0] / area * Cx::expi(2.0 * PI * (f[0] * r0[0] + f[1] * r0[1]));
        worst = worst.max((s - o.coefficient).abs());
    }
    measured("sampled LCoS far field vs closed-form c_mn", worst);
    assert!(worst < 1e-10, "{worst}");
}

/// c₁ of a blazed staircase of `n` pixels per period whose phase is averaged
/// over a window `w` (in pixel units), by direct quadrature of
/// `(1/Λ)∫ e^{iφ_w(x)} e^{−i2πx/Λ} dx`, `φ_w` the window average written out
/// piecewise: an independent path to the same number the sampled field gives.
fn fringed_c1(n: usize, w: f64) -> Cx {
    let lev = |j: i64| 2.0 * PI * (j.rem_euclid(n as i64)) as f64 / n as f64;
    let phi = |x: f64| -> f64 {
        if w == 0.0 {
            return lev(x.floor() as i64);
        }
        let (a, b) = (x - 0.5 * w, x + 0.5 * w);
        let mut s = 0.0;
        let mut k = a.floor() as i64;
        while (k as f64) < b {
            let lo = a.max(k as f64);
            let hi = b.min(k as f64 + 1.0);
            s += lev(k) * (hi - lo);
            k += 1;
        }
        s / w
    };
    let m = 200_000;
    let mut c = Cx::ZERO;
    for i in 0..m {
        let x = (i as f64 + 0.5) * n as f64 / m as f64;
        c += Cx::expi(phi(x) - 2.0 * PI * x / n as f64);
    }
    c / m as f64
}

#[test]
fn fringing_lowers_blazed_efficiency_and_matches_quadrature() {
    // [S4]: crosstalk broadens each phase step into a ramp of width w,
    // including the 2π flyback, and the first order loses power. The sampled
    // field (64 samples a pixel, the window average at each sample) against
    // direct quadrature of the same model. The 2π flyback is the steep part:
    // at w = 0.25 px it is 16 samples wide here.
    let (nlev, per) = (8u16, 32usize);
    // Six periods. The device's outermost pixels have no neighbour to fringe
    // with, so the comparison projects onto the four interior periods, where
    // the device is the periodic grating the quadrature describes.
    let np = 48u32;
    let pitch_um = 8.0;
    let mut last = 1.0;
    for w_px in [0.0, 0.25, 0.5, 1.0] {
        let d = lcos(np, pitch_um, nlev, 1.0, (0.0, 0.0), w_px * pitch_um, Pattern::Explicit(
            (0..np as usize * np as usize).map(|k| ((k % np as usize) % nlev as usize) as u16).collect(),
        ));
        let p = pitch_um / 1000.0;
        let dx = p / per as f64;
        let ns = np as usize * per;
        let g = Grid2::new([ns, 8], [dx, dx], [0.5 * dx, 0.5 * dx]).unwrap();
        let u = Field2::from_fn(g, 0.633, |_, _| Cx::ONE).unwrap();
        let (v, _) = lcos_reflect(&u, &d, LcosBeam::Modulated).unwrap();
        // The first order's coefficient over the interior: the field's
        // projection on e^{i2πx/Λ}, x from the device's left edge.
        let lam_g = nlev as f64 * p;
        let period = nlev as usize * per;
        let left = -(np as f64) * p / 2.0;
        let mut c = Cx::ZERO;
        let mut count = 0usize;
        for j in 0..g.n[1] {
            for i in period..ns - period {
                let (x, _) = g.xy(i, j);
                c += v.at(i, j) * Cx::expi(-2.0 * PI * (x - left) / lam_g);
                count += 1;
            }
        }
        let c = c / count as f64;
        let q = fringed_c1(nlev as usize, w_px);
        let eta = c.norm_sqr();
        measured(&format!("fringing w = {w_px} px: first-order efficiency (sampled)"), eta);
        measured(&format!("fringing w = {w_px} px: |sampled − quadrature| in η"), (eta - q.norm_sqr()).abs());
        assert!((eta - q.norm_sqr()).abs() < 1e-3, "w {w_px}: {eta} vs {}", q.norm_sqr());
        assert!(eta <= last + 1e-12, "efficiency must fall as the fringing widens");
        last = eta;
    }
}

#[test]
fn a_spiral_phase_leaves_a_dark_core_of_charge_one() {
    // [S5]: a Gaussian through an ℓ = 1 spiral phase, focused (Fraunhofer): no
    // light on the axis, and the phase winds once around it. 256 levels map
    // a quarter turn to 64 steps, so the pixelated spiral keeps its four-fold
    // symmetry and the null is exact to round-off.
    let np = 64u32;
    let d = lcos(np, 8.0, 256, 1.0, (0.0, 0.0), 0.0, Pattern::Generated(PatternGenerator::Vortex { charge: 1 }));
    let per = 4usize;
    let dx = 8e-3 / per as f64;
    let ns = np as usize * per;
    let g = Grid2::new([ns, ns], [dx, dx], [0.5 * dx, 0.5 * dx]).unwrap();
    let w0 = 0.12;
    let u = Field2::from_fn(g, 0.633, |x, y| Cx::real((-(x * x + y * y) / (w0 * w0)).exp())).unwrap();
    let (v, _) = lcos_reflect(&u, &d, LcosBeam::Modulated).unwrap();
    let (far, _) = propagate(&v, 500.0, Propagator::Fraunhofer { pad: 4 }, &mut Fft2::new()).unwrap();
    let peak = far.data.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
    // The axis: the far grid's centre is the axis of the source frame. The
    // source grid was offset half a sample, which the Fraunhofer step carries
    // as a linear phase, not a shift.
    let c = far.grid.n[0] / 2;
    let axis = far.at(c, c).norm_sqr() / peak;
    measured("vortex focus: on-axis intensity / peak", axis);
    assert!(axis < 1e-10, "{axis}");
    // Winding: the phase around a ring of radius 3 far-field samples.
    let mut wind = 0.0;
    let ring: Vec<Cx> = (0..64)
        .map(|k| {
            let a = 2.0 * PI * k as f64 / 64.0;
            let (i, j) = ((c as f64 + 3.0 * a.cos()).round() as usize, (c as f64 + 3.0 * a.sin()).round() as usize);
            far.at(i, j)
        })
        .collect();
    for k in 0..64 {
        let d = (ring[(k + 1) % 64] / ring[k]).arg();
        wind += d;
    }
    let winding = (wind / (2.0 * PI)).round() as i32;
    measured("vortex focus: phase winding number", winding as f64);
    assert_eq!(winding, 1);
}

#[test]
fn gerchberg_saxton_error_never_increases() {
    // [S6]: for the two-intensity problem the error-reduction (Gerchberg–
    // Saxton) error is non-increasing. A Gaussian source shaped into a flat
    // top.
    let n = [64usize, 64];
    let mut src = vec![0.0; 64 * 64];
    let mut tgt = vec![0.0; 64 * 64];
    for j in 0..64 {
        for i in 0..64 {
            let (x, y) = (i as f64 - 32.0, j as f64 - 32.0);
            src[j * 64 + i] = (-(x * x + y * y) / 200.0).exp();
            tgt[j * 64 + i] = if x.abs() < 6.0 && y.abs() < 6.0 { 1.0 } else { 0.0 };
        }
    }
    // A pseudo-random first phase, so the run has something to improve.
    let start: Vec<f64> = (0..64 * 64).map(|k| ((k as f64 * 0.618_034).fract() - 0.5) * 2.0 * PI).collect();
    let r = gerchberg_saxton(&src, &tgt, n, 60, Some(&start)).unwrap();
    for w in r.error.windows(2) {
        assert!(w[1] <= w[0] + 1e-12, "{} → {}", w[0], w[1]);
    }
    measured("Gerchberg–Saxton error, first iteration", r.error[0]);
    measured("Gerchberg–Saxton error, 60th iteration", *r.error.last().unwrap());
    assert!(r.error.last().unwrap() < &(0.5 * r.error[0]));
}

#[test]
fn pommet_flag_is_raised_below_14_wavelengths() {
    // [S7]: an 8 µm pixel at 1.55 µm (5.2λ) is flagged; at 0.532 µm (15λ) not.
    let d = lcos(4, 8.0, 256, 1.0, (0.0, 0.0), 0.0, Pattern::Uniform { value: 0 });
    let g = Grid2::square(16, 2e-3).unwrap();
    for (lam, flagged) in [(1.55, true), (0.532, false)] {
        let u = Field2::from_fn(g, lam, |_, _| Cx::ONE).unwrap();
        let (_, r) = lcos_reflect(&u, &d, LcosBeam::Modulated).unwrap();
        let has = r.validity.iter().any(|v| matches!(v, ValidityFlag::ScalarLimit { .. }));
        assert_eq!(has, flagged, "λ = {lam}");
    }
}
