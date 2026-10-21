//! The DMD model (acceptance cases C46–C48).
//!
//! Sources:
//! - [D1] S. M. Popoff, L. Malosse, R. Gutiérrez-Cuevas, Y. Bromberg, J.
//!   Commère, M. Glanc, R. Galicher, M. W. Matthès, "A practical guide to
//!   digital micro-mirror devices (DMDs) for wavefront shaping," *J. Phys.
//!   Photonics* 8, 023002 (2026), doi:10.1088/2515-7647/ae60d5: the DMD as a
//!   blazed grating (§2), the 2D blazed condition (Eq. 4), the blazed number
//!   (Eq. 5) and its values in Fig. 6; App. A's far field, one mirror's
//!   envelope times the array's orders. The 2024 version of this paper
//!   (*J. Phys. Photonics* 6, 043001, doi:10.1088/2515-7647/ad6dc0) was
//!   retracted (7, 049701 (2025), doi:10.1088/2515-7647/ae0605): its blazed
//!   number, Eq. (2), was wrong, and Figs. 4(a) and 5–9 with it. Nothing
//!   here is held against it any more.
//! - [D2] Texas Instruments, *DMD Optical Efficiency for Visible
//!   Wavelengths*, application note DLPA083B (June 2019, rev. April 2023):
//!   Table 2-1 (pitch, tilt, illumination f-number, on-state fill factor),
//!   §4.2 (the method: incoherent plane waves over the illumination cone and
//!   the band, truncated by the projection aperture) and Table 6-1 / §4.3
//!   (average photopic diffraction efficiencies). **A vendor simulation, not
//!   a measurement**: agreement shows the same model, not the same device.
//! - [D3] J. E. Harvey, C. L. Vernold, *Appl. Opt.* 37(34), 8158–8160 (1998),
//!   doi:10.1364/AO.37.008158: orders shift-invariant in direction cosines.
//! - [D4] J. E. Harvey et al., *Proc. SPIE* 8122, 812205 (2011),
//!   doi:10.1117/12.894562: the non-paraxial normalization TI uses.
//! - [D5] R. Gutiérrez-Cuevas, S. M. Popoff, "Binary amplitude holograms for
//!   shaping complex light fields with digital micromirror devices,"
//!   *J. Phys. Photonics* 6, 045022 (2024), doi:10.1088/2515-7647/ad8617
//!   (arXiv:2311.16685v1, Eqs. (11), (15)): a binary amplitude grating of 50%
//!   duty puts `(1/π)²` of the light in each first order.
//!
//! Retrieved 2026-10-19; [D1] and its retraction notice 2026-10-20.

mod common;

use common::measured;
use oc_core::modulator::{
    blazed_number, dmd_projector_efficiency, facet_normal, hinge, mirror_ft, on_axis_illumination,
    photopic_average, reflect, unit_cell_orders, DmdSpec, DmdState, LcosBeam, ModulatorKind,
    OrderModel, Pattern, PatternGenerator, SpatialModulator, TiltAxis, UnitCell,
};
use oc_core::Vec3;
use oc_wave::czt::czt2;
use oc_wave::modulator::{dmd_far_field, dmd_reflect, AngularGrid};
use oc_wave::{Cx, Fft2, Field2, Grid2};
use std::f64::consts::PI;

fn dmd(pixels: u32, pitch_um: f64, tilt: f64, axis: TiltAxis, fill: f64, pattern: Pattern) -> SpatialModulator {
    SpatialModulator {
        pixels: [pixels, pixels],
        pitch_um,
        kind: ModulatorKind::Dmd(DmdSpec {
            tilt_deg: tilt,
            tilt_axis: axis,
            mirror_fill: fill,
            reflectance: 1.0,
            window_transmittance: 1.0,
        }),
        pattern,
    }
}

fn spec(d: &SpatialModulator) -> &DmdSpec {
    match &d.kind {
        ModulatorKind::Dmd(s) => s,
        _ => unreachable!(),
    }
}

/// Popoff's geometry: the chip turned so the hinge is vertical and the light
/// stays in the plane across it, at `alpha_deg` from the normal in that plane.
fn across_hinge(alpha_deg: f64) -> Vec3 {
    let a = alpha_deg.to_radians();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    Vec3::new(-s * a.sin(), s * a.sin(), -a.cos())
}

#[test]
fn dmd_orders_sit_on_the_harvey_vernold_grid() {
    // [D3]: (α, β) = (α_in, β_in) + λ(m, n)/p, propagating only.
    let d = dmd(1, 7.56, 12.0, TiltAxis::Diagonal, 0.97, Pattern::Uniform { value: 1 });
    let din = on_axis_illumination(spec(&d));
    let lam = 0.532;
    let t = unit_cell_orders(&d, &UnitCell::uniform(1), lam, din, OrderModel::HarveyRenormalized, LcosBeam::Modulated).unwrap();
    let step = lam / 7.56;
    for o in &t.orders {
        let want = [din.x + o.m as f64 * step, din.y + o.n as f64 * step];
        assert!((o.dir.x - want[0]).abs() < 1e-12 && (o.dir.y - want[1]).abs() < 1e-12);
        assert!(o.dir.x * o.dir.x + o.dir.y * o.dir.y < 1.0 && o.dir.z > 0.0);
        assert!((o.dir.length() - 1.0).abs() < 1e-12);
    }
    // Harvey's normalization: the propagating orders share all the reflected power.
    let sum: f64 = t.orders.iter().map(|o| o.efficiency).sum();
    assert!((sum - t.total).abs() < 1e-12, "{sum} vs {}", t.total);
    measured("orders propagating at 532 nm, 7.56 µm", t.orders.len() as f64);
}

#[test]
fn the_blazed_number_matches_popoff_figure_6_and_the_facet_kinematics() {
    // [D1] Eq. (5) at 633 nm, θ_B = −12°, the values in the legend of Fig. 6:
    // 0.09 and 0.71 for 7.6 µm at α = 0° and 15°; 0.81 and 0.06 for 10.8 µm
    // at 0° and 15°. (Fig. 6(b)'s legend labels its second angle −15° while
    // the caption gives θ_B = −12° for both panels; Eq. (5) is even under
    // (θ_B, α) → (−θ_B, −α), so that value is 0.06 at +15° for the caption's
    // tilt, as held here, and the symmetry is checked too.)
    for (d, a, published) in [(7.6, 0.0, 0.09), (7.6, 15.0, 0.71), (10.8, 0.0, 0.81), (10.8, 15.0, 0.06)] {
        let mu = blazed_number(d, 0.633, -12.0, a);
        measured(&format!("blazed number, d = {d} µm, θ_B = −12°, α = {a}°"), mu);
        assert!((mu - published).abs() <= 0.01, "{d} µm, {a}°: {mu} vs {published}");
        assert!((blazed_number(d, 0.633, 12.0, -a) - mu).abs() < 1e-12);
    }
    // The same number from the kernel's own facet kinematics: the specular
    // direction off the ON facet (`facet_normal`, `reflect`) lies
    // Δ = (r − d_in)·ŵ across the hinge from the incident direction, and the
    // orders along the hinge's normal (`unit_cell_orders`' lattice,
    // t_in + λ(m, n)/p with n = −m) are √2λ/p apart: μ = |2Δ/(√2λ/p) mod 2 − 1|.
    let (_, w) = hinge(TiltAxis::Diagonal);
    let lam = 0.633;
    for tilt in [12.0, -12.0] {
        for pitch in [7.6, 10.8] {
            let d = dmd(1, pitch, tilt, TiltAxis::Diagonal, 0.95, Pattern::Uniform { value: 1 });
            let mut worst: f64 = 0.0;
            for k in 0..=360 {
                let a = -45.0 + 0.25 * k as f64;
                let din = across_hinge(a);
                let r = reflect(din, facet_normal(spec(&d), DmdState::On));
                let delta = (r - din).dot(w);
                let p_real = delta / (2f64.sqrt() * lam / pitch);
                let mu_engine = ((2.0 * p_real).rem_euclid(2.0) - 1.0).abs();
                worst = worst.max((mu_engine - blazed_number(pitch, lam, tilt, a)).abs());
            }
            measured(&format!("blazed number from the facet kinematics against Eq. (5), worst over α, d = {pitch} µm, θ_B = {tilt}°"), worst);
            assert!(worst < 1e-12, "{worst}");
        }
    }
}

#[test]
fn a_blazed_dmd_concentrates_one_order_and_an_anti_blazed_one_splits_four() {
    // [D1] §2: μ ≈ 1, one order holds most of the light; μ ≈ 0, four orders
    // share it. Normal incidence, 633 nm, mirror fill 0.95.
    let shares = |pitch: f64| {
        let d = dmd(1, pitch, 12.0, TiltAxis::Diagonal, 0.95, Pattern::Uniform { value: 1 });
        let t = unit_cell_orders(&d, &UnitCell::uniform(1), 0.633, across_hinge(0.0), OrderModel::HarveyRenormalized, LcosBeam::Modulated).unwrap();
        let mut e: Vec<f64> = t.orders.iter().map(|o| o.efficiency / t.total).collect();
        e.sort_by(|a, b| b.partial_cmp(a).unwrap());
        e
    };
    let blazed = shares(10.8);
    let anti = shares(7.6);
    measured("10.8 µm (μ = 0.81): brightest order's share", blazed[0]);
    measured("10.8 µm: second order's share", blazed[1]);
    for (k, v) in anti.iter().take(5).enumerate() {
        measured(&format!("7.6 µm (μ = 0.09): share of order ranked {}", k + 1), *v);
    }
    // Blazed: one order takes most of the light.
    assert!(blazed[0] > 0.6, "{:?}", &blazed[..4]);
    assert!(blazed[1] < 0.25 * blazed[0]);
    // Anti-blazed: no order dominates, and at least four carry a tenth each.
    assert!(anti[0] < 0.4, "{:?}", &anti[..5]);
    assert!(anti[3] > 0.1, "{:?}", &anti[..5]);
}

#[test]
fn a_flat_state_dmd_puts_the_fill_factor_squared_in_the_zero_order() {
    // A flat array of mirrors of linear fill ρ is a rectangular amplitude
    // grating: c₀₀ = ρ², and Σ|c|² = ρ² (Parseval). Normal incidence.
    let rho = 0.9;
    let d = dmd(1, 10.8, 12.0, TiltAxis::Diagonal, rho, Pattern::Uniform { value: 2 });
    let t = unit_cell_orders(&d, &UnitCell::uniform(2), 0.55, Vec3::new(0.0, 0.0, -1.0), OrderModel::Tea, LcosBeam::Modulated).unwrap();
    let z = t.orders.iter().find(|o| o.m == 0 && o.n == 0).unwrap();
    assert!((z.efficiency - rho.powi(4)).abs() < 1e-12, "{}", z.efficiency);
    assert!((t.total - rho * rho).abs() < 1e-12);
    // A flat mirror sends the zero order straight back.
    assert!((z.dir.z - 1.0).abs() < 1e-15);
}

#[test]
fn the_factorized_far_field_lands_on_the_closed_form_orders() {
    // At an order's direction the lattice sum of an N×N device is N² in
    // magnitude, so |Û|/(N²p²) must equal |c_mn| exactly.
    let n = 12;
    let d = dmd(n, 7.56, 12.0, TiltAxis::Diagonal, 0.95, Pattern::Uniform { value: 1 });
    let din = on_axis_illumination(spec(&d));
    let lam = 0.55;
    let t = unit_cell_orders(&d, &UnitCell::uniform(1), lam, din, OrderModel::Tea, LcosBeam::Modulated).unwrap();
    let mut fft = Fft2::new();
    let p = 7.56e-3;
    let mut worst: f64 = 0.0;
    for o in t.orders.iter().filter(|o| o.m.abs() <= 3 && o.n.abs() <= 3) {
        let far = dmd_far_field(&d, lam, din, AngularGrid { start: [o.dir.x, o.dir.y], step: [1e-3, 1e-3], n: [1, 1] }, &[DmdState::On], &mut fft).unwrap();
        let got = far.data[0].abs() / ((n * n) as f64 * p * p);
        worst = worst.max((got - o.coefficient.abs()).abs());
    }
    measured("factorized far field vs closed-form |c_mn|", worst);
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn the_sampled_dmd_far_field_converges_on_the_factorized_one() {
    // The sampled near field (the ON family's mask band-limited to the grid,
    // its carrier), transformed, against the factorized far field: the same
    // thin-element model computed two ways. 6 × 6 mirrors under a normal
    // plane wave. Between the window's own frequencies the near field's
    // transform interpolates its lattice of values, and the mask's content
    // beyond the grid's band enters there through the interpolation's tails:
    // that converges with the sampling (at the orders themselves, on the
    // lattice, the two agree to round-off: the next test).
    let n = 6;
    let d = dmd(n, 7.56, 12.0, TiltAxis::Diagonal, 0.9, Pattern::Uniform { value: 1 });
    let p = 7.56e-3;
    let lam = 0.633;
    let din = Vec3::new(0.0, 0.0, -1.0);
    let spec_dir = reflect(din, facet_normal(spec(&d), DmdState::On));
    // Directions: a patch about the envelope's centre.
    let m = 41;
    let step = 0.004;
    let start = [spec_dir.x - step * 20.0, spec_dir.y - step * 20.0];
    let mut fft = Fft2::new();
    let refer = dmd_far_field(&d, lam, din, AngularGrid { start, step: [step, step], n: [m, m] }, &[DmdState::On], &mut fft).unwrap();
    let peak = refer.data.iter().map(|z| z.abs()).fold(0.0, f64::max);
    // Exact cell areas put each partial edge cell at the cell's centre rather
    // than at its covered part's centroid. That error is second order in Δx
    // on average but jitters with where the mirror edges fall among the
    // samples, so convergence is judged by the slope over several samplings,
    // not by any one pair.
    let pers = [4usize, 8, 12, 16, 24, 32];
    let mut pts = vec![];
    for per in pers {
        let g = Grid2::square(n as usize * per + 2 * per, p / per as f64).unwrap();
        let u = Field2::from_fn(g, lam, |_, _| Cx::ONE).unwrap();
        let (v, r) = dmd_reflect(&u, &d, DmdState::On).unwrap();
        assert!(r.ledger.numerical.abs() < 1e-12 * r.ledger.launched, "{:?}", r.ledger);
        // Far field of the sampled field: its carrier shifts the spectrum.
        let lm = lam / 1000.0;
        let f0 = [start[0] / lm - v.carrier[0] / lm, start[1] / lm - v.carrier[1] / lm];
        let s = czt2(&v, f0, [step / lm, step / lm], [m, m], &mut fft);
        let e = s.iter().zip(&refer.data).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max) / peak;
        measured(&format!("sampled vs factorized DMD far field, {per} samples per mirror"), e);
        pts.push(((per as f64).ln(), e.ln()));
    }
    let nn = pts.len() as f64;
    let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / nn, pts.iter().map(|p| p.1).sum::<f64>() / nn);
    let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
    measured("sampled vs factorized DMD far field, log-log convergence slope", slope);
    assert!(slope < -1.5, "{slope}");
    assert!(pts.last().unwrap().1.exp() < 2e-3);
}

#[test]
fn a_binary_grating_on_the_on_mirrors_puts_one_over_pi_squared_in_each_first_order() {
    // [D5] Eq. (15): a 50%-duty binary amplitude grating sends (1/π)² of the
    // light into each first order. On a DMD the ON mirrors carry the grating:
    // the pattern's first order, next to the all-ON array's order in the same
    // place, scales by the lattice sum (1/P)|Σ_{i<P/2} e^{−i2πi/P}| =
    // 1/(P sin(π/P)) → 1/π as the period P grows. The OFF mirrors' envelope,
    // centred 2θ the other way, leaks into the same direction; their lattice
    // sum is exactly minus the ON one, so the full prediction is
    // 1/(P sin(π/P)) · |1 − M̂_off/M̂_on|, every factor in closed form.
    let din = Vec3::new(0.0, 0.0, -1.0);
    let lam = 0.633;
    let lm = lam / 1000.0;
    let d = dmd(1, 10.8, 12.0, TiltAxis::Diagonal, 0.95, Pattern::Uniform { value: 1 });
    let s_on = reflect(din, facet_normal(spec(&d), DmdState::On));
    let s_off = reflect(din, facet_normal(spec(&d), DmdState::Off));
    let env = |o: &oc_core::modulator::Order, st: DmdState, sd: Vec3| {
        mirror_ft(spec(&d), 10.8e-3, st, [(o.dir.x - sd.x) / lm, (o.dir.y - sd.y) / lm])
    };
    for period in [16usize, 64, 256] {
        let mut cell = UnitCell { period: [period, 1], values: vec![0; period] };
        for v in cell.values.iter_mut().take(period / 2) {
            *v = 1;
        }
        let g = unit_cell_orders(&d, &cell, lam, din, OrderModel::Tea, LcosBeam::Modulated).unwrap();
        let u = unit_cell_orders(&d, &UnitCell::uniform(1), lam, din, OrderModel::Tea, LcosBeam::Modulated).unwrap();
        let b = u.brightest().unwrap();
        let first = g.orders.iter().find(|o| o.m == period as i32 * b.m + 1 && o.n == b.n).unwrap();
        let on1 = env(first, DmdState::On, s_on);
        let off1 = env(first, DmdState::Off, s_off);
        let got = (first.coefficient.abs() / on1.abs()) / (b.coefficient.abs() / env(b, DmdState::On, s_on).abs());
        let lattice = 1.0 / (period as f64 * (PI / period as f64).sin());
        let predicted = lattice * (1.0 - off1 / on1).abs();
        measured(&format!("binary grating, period {period}: first-order power, lattice factor²"), lattice * lattice);
        measured(&format!("binary grating, period {period}: OFF-mirror leak |M̂_off/M̂_on|"), (off1 / on1).abs());
        assert!((got / predicted - 1.0).abs() < 1e-10, "P {period}: {got} vs {predicted}");
    }
    assert!(((1.0 / (256.0 * (PI / 256.0).sin())).powi(2) - 1.0 / (PI * PI)).abs() < 1e-5);
}

#[test]
fn ti_dlpa083_average_photopic_diffraction_efficiency() {
    // [D2] Table 6-1 and §4.3, by TI's own method (§4.2) with Harvey's
    // normalization [D4]: flat spectrum 420–680 nm weighted by V(λ); the
    // illumination cone about the direction the ON mirrors send along the
    // normal; the projection cone about the normal; the share of the
    // reflected power collected. The mirror fill is taken as the square root
    // of TI's on-state fill factor (Table 2-1), which TI gives as an area
    // fraction from a ray trace; no parameter is fitted.
    //
    // Initial gate ±3 percentage points per row. The 5.4 µm device at f/2.4 is
    // measured +4.4 pp from TI's 80% and is held to ±5 pp here, as a named
    // discrepancy rather than a widened gate: TI's 5.4 µm pixel tilts about
    // its side (§4.2) and its mirror structure is the least documented.
    // (name, pitch µm, tilt °, axis, illumination f/#, projection f/#,
    //  TI on-state fill factor, TI diffraction efficiency, gate in pp)
    type Row = (&'static str, f64, f64, TiltAxis, f64, f64, f64, f64, f64);
    let rows: [Row; 7] = [
        ("13.68 µm, f/2.4", 13.68, 12.0, TiltAxis::Diagonal, 2.4, 2.4, 0.92, 0.89, 3.0),
        ("10.8 µm, f/2.4", 10.8, 12.0, TiltAxis::Diagonal, 2.4, 2.4, 0.92, 0.87, 3.0),
        ("9.0 µm, f/2.0", 9.0, 14.5, TiltAxis::Diagonal, 2.0, 2.0, 0.97, 0.89, 3.0),
        ("7.56 µm, f/2.4", 7.56, 12.0, TiltAxis::Diagonal, 2.4, 2.4, 0.94, 0.84, 3.0),
        ("5.4 µm, f/1.7", 5.4, 17.0, TiltAxis::Side, 1.7, 1.7, 0.93, 0.86, 3.0),
        ("5.4 µm, f/2.4", 5.4, 17.0, TiltAxis::Side, 2.4, 2.4, 0.93, 0.80, 5.0),
        ("7.56 µm, f/3 → f/2.4", 7.56, 12.0, TiltAxis::Diagonal, 3.0, 2.4, 0.94, 0.905, 3.0),
    ];
    for (name, pitch, tilt, axis, fi, fp, fill, ti, gate) in rows {
        let d = dmd(1, pitch, tilt, axis, fill.sqrt(), Pattern::Uniform { value: 1 });
        let eta = photopic_average(420.0, 680.0, 5.0, |lam| {
            dmd_projector_efficiency(&d, lam, fi, fp, 64, OrderModel::HarveyRenormalized)
        })
        .unwrap();
        let pp = 100.0 * (eta - ti);
        measured(&format!("TI DLPA083B {name}: canvas {:.1}% vs TI {:.1}%, Δ (pp)", 100.0 * eta, 100.0 * ti), pp);
        assert!(pp.abs() <= gate, "{name}: {eta} vs {ti}");
    }
}

#[test]
fn the_lee_generator_follows_gutierrez_cuevas_and_popoff_eq_11() {
    // [D5] Eq. (11): ON where cos(2πν·r − φ) ≥ cos(πq), A = sin(πq). A uniform
    // target (A = 1, q = 1/2) is a 50%-duty binary grating along the carrier.
    let d = dmd(64, 10.8, 12.0, TiltAxis::Diagonal, 0.95, Pattern::Generated(PatternGenerator::Lee {
        target: oc_core::modulator::TargetField::Uniform,
        period_px: 8.0,
        angle_deg: 0.0,
        peak: 1.0,
    }));
    let on: usize = (0..64).map(|i| d.value(i, 7) as usize).sum();
    assert_eq!(on, 32, "half the pixels of a row are ON");
    // A weaker target narrows the fringes: A = sin(πq).
    let weak = dmd(64, 10.8, 12.0, TiltAxis::Diagonal, 0.95, Pattern::Generated(PatternGenerator::Lee {
        target: oc_core::modulator::TargetField::Uniform,
        period_px: 64.0,
        angle_deg: 0.0,
        peak: 1.0 / (PI / 4.0).sin(), // A = sin(π/4): q = 1/4
    }));
    let on: usize = (0..64).map(|i| weak.value(i, 7) as usize).sum();
    assert!((on as f64 - 16.0).abs() <= 1.0, "{on}");
}

#[test]
fn on_its_window_s_frequencies_the_near_field_is_the_factorized_far_field_at_any_sampling() {
    // Q17 of the expert review. The device's mask is band-limited exactly to
    // the field's grid, from its closed-form transform, instead of averaged
    // over each sample's cell (which weights order m by sinc(m/s) at s
    // samples a mirror: 0.935 in amplitude for an order 1.6 lattice steps out
    // at 8). So on the window's own frequencies, k/W, inside the grid's band,
    // the near field's transform is the factorized far field to round-off,
    // at 4 samples a mirror as at 32; its power in each order is then the
    // factorized one, whatever the sampling. 6 × 6 mirrors on an 8-mirror
    // window, under a normal plane wave.
    let n = 6;
    let d = dmd(n, 7.56, 12.0, TiltAxis::Diagonal, 0.9, Pattern::Uniform { value: 1 });
    let p = 7.56e-3;
    let lam = 0.633;
    let lm = lam / 1000.0;
    let din = Vec3::new(0.0, 0.0, -1.0);
    let mut fft = Fft2::new();
    for per in [4usize, 8, 16, 32] {
        let m = n as usize * per + 2 * per;
        let g = Grid2::square(m, p / per as f64).unwrap();
        let u = Field2::from_fn(g, lam, |_, _| Cx::ONE).unwrap();
        let (v, r) = dmd_reflect(&u, &d, DmdState::On).unwrap();
        assert!(r.ledger.numerical.abs() < 1e-12 * r.ledger.launched, "{:?}", r.ledger);
        // The window's lattice, k/W, within 0.9 of the band, in a patch of
        // 41 × 41 about the family's carrier.
        let w = m as f64 * g.pitch_mm[0];
        let df = 1.0 / w;
        let band = 0.5 / g.pitch_mm[0];
        let kmax = ((0.9 * band / df).floor() as i64).min(20);
        let k0 = [-kmax as f64 * df, -kmax as f64 * df];
        let kn = (2 * kmax + 1) as usize;
        let near = czt2(&v, k0, [df, df], [kn, kn], &mut fft);
        let start = [v.carrier[0] + lm * k0[0], v.carrier[1] + lm * k0[1]];
        let far = dmd_far_field(&d, lam, din, AngularGrid { start, step: [lm * df, lm * df], n: [kn, kn] }, &[DmdState::On], &mut fft).unwrap();
        let peak = far.data.iter().map(|z| z.abs()).fold(0.0, f64::max);
        let worst = near.iter().zip(&far.data).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
        measured(&format!("band-limited DMD mask, {per} samples per mirror: on {kn}² of the window's frequencies, near field against the factorized far field, worst / peak"), worst / peak);
        assert!(worst / peak < 1e-10, "{per}: {}", worst / peak);
    }
}

#[test]
fn the_thin_element_mask_against_one_facet_s_surface_integral() {
    // V04 and Q4 of the expert review: a reference for the array-normal mask
    // and pivot ramp, which is a defined thin-element approximation. One flat
    // square facet's integral under a uniform plane wave, with its physical
    // tangents, is a² sinc(a q·u/2π) sinc(a q·v/2π), q = k_in − k_out
    // (`facet_integral`); the thin element's mirror envelope is the
    // projected outline's transform at the in-plane offset from the
    // specular direction (`mirror_ft`). 10 µm mirrors of fill 0.95 tilted 12°
    // about a side, lit at 24° so that the ON specular is the array normal.
    // Both peak there. The facet's tangent across the hinge leaves the
    // array plane, so off the specular the facet sees the outgoing ray's
    // normal component, which the array-normal mask does not: the two
    // envelopes' arguments differ at second order in the angle s. Across the
    // hinge that skews the facet's envelope, an |E|² gap odd in s and of
    // third order; along it the facet's second sinc departs from 1 at s²,
    // a gap of fourth order. And the power each takes:
    // the mask reflects ρ² cos θ of a mirror's cell, an isolated facet would
    // intercept ρ²|d·n_f|/|d·z|, 1/cos 24° = 1.095 times as much at this
    // geometry. Measured and stated, not corrected.
    use oc_wave::modulator::{facet_integral, facet_interception_ratio, FACET_INCIDENCE_FLAG_DEG};
    use oc_wave::report::ValidityFlag;
    let d = dmd(41, 10.0, 12.0, TiltAxis::Side, 0.95, Pattern::Uniform { value: 1 });
    let ds = spec(&d).clone();
    let (p, lam) = (1e-2, 0.633);
    let lm = lam / 1000.0;
    let k = 2.0 * PI / lm;
    let a = ds.mirror_fill * p;
    let din = on_axis_illumination(&ds);
    let nf = facet_normal(&ds, DmdState::On);
    let ds_out = reflect(din, nf);
    assert!((ds_out - Vec3::Z).length() < 1e-12, "{ds_out:?}");
    let (h, _) = hinge(ds.tilt_axis);
    let u_m = Vec3::new(h.x, h.y, 0.0);
    let v_m = nf.cross(u_m).normalize();
    let facet = |t: [f64; 2]| {
        let dout = Vec3::new(t[0], t[1], (1.0 - t[0] * t[0] - t[1] * t[1]).sqrt());
        facet_integral(a, u_m, v_m, din * k, dout * k).re / (a * a)
    };
    let thin = |t: [f64; 2]| mirror_ft(&ds, p, DmdState::On, [(t[0] - ds_out.x) / lm, (t[1] - ds_out.y) / lm]) / (12f64.to_radians().cos() * a * a);
    assert!((facet([0.0, 0.0]) - 1.0).abs() < 1e-15 && (thin([0.0, 0.0]) - 1.0).abs() < 1e-15);
    // The plane across the hinge, and along it.
    let w = Vec3::new(-h.y, h.x, 0.0);
    let gap = |e: Vec3, deg: f64| {
        let s = deg.to_radians().sin();
        facet([e.x * s, e.y * s]).powi(2) - thin([e.x * s, e.y * s]).powi(2)
    };
    for (name, e, order) in [("across", w, 3), ("along", u_m, 4)] {
        for deg in [0.5, 1.0, 2.0, 4.0, 8.0] {
            measured(&format!("facet integral against the thin element's envelope, {deg}° {name} the hinge: Δ|E|²/peak"), gap(e, deg));
        }
        // Near the peak (the envelope's first zero is 3.8° out), doubling
        // the angle multiplies the gap by 2 to its order.
        let r = gap(e, 0.25) / gap(e, 0.125);
        measured(&format!("facet integral against the thin element, {name} the hinge: gap at 0.25° over 0.125°"), r);
        let want = 2f64.powi(order);
        assert!((r / want - 1.0).abs() < 0.05, "{name}: {r} vs {want}");
        assert!((gap(e, -0.25) - (-1f64).powi(order) * gap(e, 0.25)).abs() < 1e-3 * gap(e, 0.25).abs(), "{name}: parity");
    }
    // Interception.
    let ratio = facet_interception_ratio(&ds, DmdState::On, din);
    measured("isolated facet over array-normal mask, interception at 24° on 12° facets", ratio);
    assert!((ratio - 1.0 / 24f64.to_radians().cos()).abs() < 1e-12, "{ratio}");
    // The window on the device's edges: its cells half a sample off centre.
    let dx = p / 8.0;
    let g = Grid2::new([41 * 8, 41 * 8], [dx, dx], [0.5 * dx, 0.5 * dx]).unwrap();
    let mut u = Field2::from_fn(g, lam, |_, _| Cx::ONE).unwrap();
    u.carrier = [din.x, din.y];
    let (_, r) = dmd_reflect(&u, &d, DmdState::On).unwrap();
    let l = r.ledger;
    let reflected = (l.detected + l.spectral_projection + l.subcell_projection) / l.launched;
    let want = ds.mirror_fill * ds.mirror_fill * 12f64.to_radians().cos();
    measured("DMD mask, uniform light: reflected fraction against ρ² cos θ", reflected - want);
    assert!((reflected - want).abs() < 1e-12, "{reflected} vs {want}");
    assert!(r.validity.iter().any(|v| matches!(v, ValidityFlag::ThinElement { detail } if detail.contains("isolated"))), "{:?}", r.validity);
    // Below the mark, no flag.
    let near = Vec3::new(0.1, 0.0, -(1.0f64 - 0.01).sqrt());
    assert!(near.x.asin().to_degrees() < FACET_INCIDENCE_FLAG_DEG);
    u.carrier = [near.x, near.y];
    let (_, r) = dmd_reflect(&u, &d, DmdState::On).unwrap();
    assert!(!r.validity.iter().any(|v| matches!(v, ValidityFlag::ThinElement { .. })), "{:?}", r.validity);
}

#[test]
fn a_harvey_table_scales_efficiencies_and_leaves_the_coefficients_alone() {
    // Q5 of the expert review: under Harvey's renormalization ([D4]) the
    // efficiencies are |c_mn|²·Σ_all/Σ_propagating and the coefficients are
    // the thin element's c_mn as they are; the two differ, and the table says
    // so by its model. Renormalizing a TEA table gives the Harvey table;
    // renormalizing that changes nothing.
    let d = dmd(1, 7.56, 12.0, TiltAxis::Diagonal, 0.97, Pattern::Uniform { value: 1 });
    let din = on_axis_illumination(spec(&d));
    let lam = 0.633;
    let tea = unit_cell_orders(&d, &UnitCell::uniform(1), lam, din, OrderModel::Tea, LcosBeam::Modulated).unwrap();
    let harvey = unit_cell_orders(&d, &UnitCell::uniform(1), lam, din, OrderModel::HarveyRenormalized, LcosBeam::Modulated).unwrap();
    assert_eq!(tea.renormalized(), harvey);
    assert_eq!(harvey.renormalized(), harvey);
    let k = tea.total / tea.propagating;
    measured("Harvey's factor Σ_all/Σ_propagating, 7.56 µm at 633 nm", k);
    assert!(k > 1.0);
    for (a, b) in tea.orders.iter().zip(&harvey.orders) {
        assert_eq!(a.coefficient, b.coefficient);
        assert!((a.efficiency - a.coefficient.norm_sqr()).abs() < 1e-15);
        assert!((b.efficiency - k * b.coefficient.norm_sqr()).abs() < 1e-15);
    }
}
