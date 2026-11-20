//! The zoomed transform (acceptance case C44).
//!
//! Sources, numbered as in the Sources list of docs/wave-optics.md:
//! - [9] L. R. Rabiner, R. W. Schafer, C. M. Rader, "The chirp z-transform
//!   algorithm," *IEEE Trans. Audio Electroacoust.* 17(2), 86–92 (1969),
//!   doi:10.1109/TAU.1969.1162034: Bluestein's identity, which the zoom uses.
//! - [1] J. W. Goodman, *Introduction to Fourier Optics*, 4th ed. (2017):
//!   the Fraunhofer pattern of a circular aperture.
//! - [16] M. Born, E. Wolf, *Principles of Optics*, 7th ed. (1999): the Airy
//!   pattern's first zero at `v = 3.8317…`, i.e. `r = 1.21967 λz/D`.
//! - [15] NIST DLMF ch. 10: `j₁,₁ = 3.8317059702075123`.
//!
//! Retrieved 2026-10-19.

mod common;

use common::{max_diff, measured, peak};
use oc_wave::czt::{czt2, fraunhofer_zoom};
use oc_wave::elements::{apply_aperture, Shape};
use oc_wave::{propagate, Cx, Field2, Fft2, Grid2, Propagator, WaveReport};
use std::f64::consts::PI;

#[test]
fn chirp_z_zoom_matches_the_direct_dft() {
    // A structureless field on an off-centre window, transformed onto an
    // arbitrary frequency grid, against the sum written out [9].
    let g = Grid2::new([23, 17], [0.011, 0.007], [0.05, -0.02]).unwrap();
    let u = Field2::from_fn(g, 0.5, |x, y| Cx::new((31.0 * x).sin() + y, (17.0 * y).cos() - x)).unwrap();
    let (f0, df, m) = ([-20.0, 13.0], [0.83, 1.7], [29, 11]);
    let got = czt2(&u, f0, df, m, &mut Fft2::new());
    let mut want = vec![Cx::ZERO; m[0] * m[1]];
    for my in 0..m[1] {
        for mx in 0..m[0] {
            let (fx, fy) = (f0[0] + mx as f64 * df[0], f0[1] + my as f64 * df[1]);
            let mut s = Cx::ZERO;
            for j in 0..g.n[1] {
                for i in 0..g.n[0] {
                    let (x, y) = g.xy(i, j);
                    s += u.at(i, j) * Cx::expi(-2.0 * PI * (fx * x + fy * y));
                }
            }
            want[my * m[0] + mx] = s * g.cell_area_mm2();
        }
    }
    let err = max_diff(&got, &want) / peak(&want);
    measured("chirp-z against the direct 2D sum, relative to peak", err);
    assert!(err < 1e-11, "{err}");
}

fn airy_source(d: f64, dx: f64, n: usize) -> Field2 {
    let mut u = Field2::from_fn(Grid2::square(n, dx).unwrap(), 0.5, |_, _| Cx::ONE).unwrap();
    let mut r = WaveReport::default();
    apply_aperture(&mut u, &Shape::Disc { c: [0.0, 0.0], r: 0.5 * d }, false, &mut r).unwrap();
    u
}

#[test]
fn fraunhofer_zoom_lands_on_the_fft_fraunhofer_on_its_own_grid() {
    // Asked for the FFT's own output grid, the zoom must return the same field.
    let u = airy_source(0.2, 0.2 / 32.0, 128);
    let z = 1000.0;
    let mut fft = Fft2::new();
    let (a, _) = propagate(&u, z, Propagator::Fraunhofer { pad: 2 }, &mut fft).unwrap();
    let (b, rb) = fraunhofer_zoom(&u, z, a.grid, &mut fft).unwrap();
    let err = max_diff(&a.data, &b.data) / peak(&a.data);
    measured("zoom on the FFT grid against the FFT, relative to peak", err);
    assert!(err < 1e-11, "{err}");
    // The whole far field is in that window, so the ledger keeps it all.
    assert!(rb.ledger.left_window < 1e-9 * rb.ledger.launched, "{:?}", rb.ledger);
}

#[test]
fn the_airy_first_zero_lands_at_1_21967_lambda_z_over_d() {
    // [1], [16], [15]: the first zero of 2J₁(v)/v is v = j₁,₁, so
    // r₁ = j₁,₁ λz / (πD). The zoom samples |U| finely across it, a parabola
    // through the three samples about the minimum of |U|² locates it, and the
    // sampled aperture's error in it falls at second order in Δx/D.
    let (lam, d, z) = (0.5, 0.2, 1000.0);
    let lz = lam / 1000.0 * z;
    let r1 = 3.831_705_970_207_512_3 * lz / (PI * d);
    let mut errs = vec![];
    for per in [32.0, 64.0, 128.0] {
        let dx = d / per;
        let n = (2.0 * d / dx) as usize + 8;
        let u = airy_source(d, dx, n);
        let m = 801;
        let half = 0.02 * r1;
        let out = Grid2::new([m, 1], [2.0 * half / (m - 1) as f64, 1e-3], [r1, 0.0]).unwrap();
        let (v, _) = fraunhofer_zoom(&u, z, out, &mut Fft2::new()).unwrap();
        let i: Vec<f64> = v.data.iter().map(|c| c.norm_sqr()).collect();
        let k = (1..m - 1).min_by(|&a, &b| i[a].partial_cmp(&i[b]).unwrap()).unwrap();
        let (y0, y1, y2) = (i[k - 1], i[k], i[k + 1]);
        let off = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2);
        let r = out.coord(0, k) + off * out.pitch_mm[0];
        let rel = (r - r1).abs() / r1;
        measured(&format!("Airy first zero, D/Δx = {per}: relative error"), rel);
        errs.push(rel);
    }
    measured("Airy first zero, error ratio 64 → 128", errs[1] / errs[2]);
    assert!(errs[2] < 1e-4, "{errs:?}");
    assert!(errs[1] / errs[2] > 3.0, "not second order: {errs:?}");
}
