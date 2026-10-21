//! The Mie reference (B30 of the wave-optics bundle; V03 of the expert review
//! of the wave engine).
//!
//! Sources:
//! - [M1] C. F. Bohren, D. R. Huffman, *Absorption and Scattering of Light by
//!   Small Particles* (Wiley, 1983), ch. 4: the coefficients (Eq. 4.53), the
//!   fields (Eqs. 4.45, 4.50), the efficiencies (Eqs. 4.61, 4.62); for real
//!   m and x, Re a_n = |a_n|².
//! - [M2] H. Du, "Mie-scattering calculation," *Appl. Opt.* 43(9), 1951–1956
//!   (2004), doi:10.1364/AO.43.001951, Table 1: extinction and scattering
//!   efficiencies to seven digits for thirteen spheres (as reproduced in the
//!   scattnlay test suite, retrieved 2026-10-21).
//! - [M3] miepython 3.3.0 (S. Prahl), an independent implementation:
//!   efficiencies and the total exterior E and H at the bundle's points,
//!   frozen in `tests/fixtures/mie_b30.json` by `tools/fixtures/mie_reference.py`.
//! - [M4] W. J. Wiscombe, NCAR Technical Note TN-140+STR (1979): the
//!   number of terms, ⌈x + 4x^{1/3} + 2⌉.

mod common;

use common::measured;
use oc_wave::mie::{wiscombe_terms, MieSphere};
use oc_wave::Cx;

#[test]
fn the_efficiencies_match_du_s_table() {
    // [M2] Table 1, the absorbing cases included (m = n + iκ here, BH's
    // sign for exp(−iωt)).
    let table: [(f64, Cx, f64, f64); 13] = [
        (0.099, Cx::new(0.75, 0.0), 7.417859e-06, 7.417859e-06),
        (0.101, Cx::new(0.75, 0.0), 8.033538e-06, 8.033538e-06),
        (10.0, Cx::new(0.75, 0.0), 2.232265, 2.232265),
        (1000.0, Cx::new(0.75, 0.0), 1.997908, 1.997908),
        (100.0, Cx::new(1.33, 1e-5), 2.101321, 2.096594),
        (10000.0, Cx::new(1.33, 1e-5), 2.004089, 1.723857),
        (0.055, Cx::new(1.5, 1.0), 0.10149104, 1.131687e-05),
        (0.056, Cx::new(1.5, 1.0), 0.1033467, 1.216311e-05),
        (100.0, Cx::new(1.5, 1.0), 2.097502, 1.283697),
        (10000.0, Cx::new(1.5, 1.0), 2.004368, 1.236574),
        (1.0, Cx::new(10.0, 10.0), 2.532993, 2.049405),
        (100.0, Cx::new(10.0, 10.0), 2.071124, 1.836785),
        (10000.0, Cx::new(10.0, 10.0), 2.005914, 1.795393),
    ];
    let mut worst: f64 = 0.0;
    for (x, m, ext, sca) in table {
        let (qe, qs) = MieSphere::new(m, x, 1e-13).unwrap().efficiencies();
        // The table's seven digits: half a unit in the last.
        let digit = |q: f64| 0.5 * 10f64.powf(q.abs().log10().floor() - 6.0);
        let (de, ds) = ((qe - ext).abs() / digit(ext), (qs - sca).abs() / digit(sca));
        measured(&format!("Mie against Du's Table 1, x = {x}, m = {m:?}: Q_ext {qe:.7e}, Q_sca {qs:.7e}, in half units of the table's last digit"), de.max(ds));
        worst = worst.max(de.max(ds));
    }
    assert!(worst <= 1.0, "{worst}");
}

#[derive(serde::Deserialize)]
struct Fixture {
    source: String,
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
#[allow(non_snake_case)]
struct Case {
    n: f64,
    kR: f64,
    qext: f64,
    qsca: f64,
    points: Vec<Point>,
}

#[derive(serde::Deserialize)]
#[allow(non_snake_case)]
struct Point {
    kxyz: [f64; 3],
    E: [[f64; 2]; 3],
    H: [[f64; 2]; 3],
}

fn fixture() -> Fixture {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mie_b30.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn the_exterior_fields_match_an_independent_implementation_at_the_bundle_s_points() {
    // B30: a sphere of index 1 and 1.5 in vacuum, kR = 10, 100, 1000, the
    // total exterior E and H at r/R = 1.2, 2, 5, four polar and three
    // azimuthal angles, against [M3]; the efficiencies too.
    let f = fixture();
    println!("reference: {}", f.source);
    for c in &f.cases {
        let s = MieSphere::new(Cx::real(c.n), c.kR, 1e-14).unwrap();
        let (qe, qs) = s.efficiencies();
        let dq = (qe - c.qext).abs().max((qs - c.qsca).abs());
        let (mut de, mut dh): (f64, f64) = (0.0, 0.0);
        for p in &c.points {
            let (e, h) = s.exterior_field(p.kxyz).unwrap();
            for k in 0..3 {
                de = de.max((e[k] - Cx::new(p.E[k][0], p.E[k][1])).abs());
                dh = dh.max((h[k] - Cx::new(p.H[k][0], p.H[k][1])).abs());
            }
        }
        measured(&format!("Mie against miepython, n = {}, kR = {}: efficiencies, worst |ΔQ|", c.n, c.kR), dq);
        measured(&format!("Mie against miepython, n = {}, kR = {}: exterior E, worst |ΔE| / |E₀| over {} points", c.n, c.kR, c.points.len()), de);
        measured(&format!("Mie against miepython, n = {}, kR = {}: exterior H, worst |ΔH| / |H₀|", c.n, c.kR), dh);
        assert!(dq < 1e-10 * c.qext.max(1.0), "n {} kR {}: {dq}", c.n, c.kR);
        assert!(de < 1e-8 && dh < 1e-8, "n {} kR {}: E {de}, H {dh}", c.n, c.kR);
    }
}

#[test]
fn the_series_converges_and_the_fields_obey_maxwell_s_equations() {
    // B30's series refinement: from Wiscombe's start [M4] the terms grow by
    // 16 until the efficiencies hold; the sampled fields must hold too, at
    // the nearest of the bundle's radii, 1.2R. And the fields obey
    // Faraday's law, ∇ × E = iωμH, which in these units (lengths in 1/k, H
    // the incident's units) reads ∇ × E = iH: a central difference of step
    // 1e-4, second order, at points around the sphere.
    for x in [10.0, 100.0, 1000.0] {
        let n0 = wiscombe_terms(x);
        let a = MieSphere::with_terms(Cx::real(1.5), x, n0 + 16).unwrap();
        let b = MieSphere::with_terms(Cx::real(1.5), x, n0 + 32).unwrap();
        let mut worst: f64 = 0.0;
        for th in [0.3f64, 1.2, 2.5] {
            for ph in [0.0f64, 0.7, 1.9] {
                let r = 1.2 * x;
                let p = [r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), r * th.cos()];
                let ((ea, ha), (eb, hb)) = (a.exterior_field(p).unwrap(), b.exterior_field(p).unwrap());
                for k in 0..3 {
                    worst = worst.max((ea[k] - eb[k]).abs()).max((ha[k] - hb[k]).abs());
                }
            }
        }
        measured(&format!("Mie, m = 1.5, x = {x}: fields at 1.2R with {} against {} terms, worst |Δ|", n0 + 16, n0 + 32), worst);
        assert!(worst < 1e-12, "x {x}: {worst}");
        // Faraday's law.
        let h = 1e-4;
        let mut curl_err: f64 = 0.0;
        for p in [[0.7 * x, 0.4 * x, 1.1 * x], [-1.5 * x, 0.2 * x, -0.3 * x], [0.1 * x, -2.0 * x, 0.5 * x]] {
            let e_at = |dp: [f64; 3]| a.exterior_field([p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]]).unwrap().0;
            let d = |axis: usize, comp: usize| {
                let mut plus = [0.0; 3];
                plus[axis] = h;
                let minus = [-plus[0], -plus[1], -plus[2]];
                (e_at(plus)[comp] - e_at(minus)[comp]) / (2.0 * h)
            };
            let curl = [d(1, 2) - d(2, 1), d(2, 0) - d(0, 2), d(0, 1) - d(1, 0)];
            let (_, hh) = a.exterior_field(p).unwrap();
            for k in 0..3 {
                curl_err = curl_err.max((curl[k] - Cx::I * hh[k]).abs());
            }
        }
        measured(&format!("Mie, m = 1.5, x = {x}: |∇ × E − iH| by central differences of 1e-4"), curl_err);
        assert!(curl_err < 1e-6, "x {x}: {curl_err}");
    }
}
