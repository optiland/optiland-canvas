//! The sun a heliostat rig emits, checked ray by ray against the profiles it
//! claims: Neckel & Labs (1994) across the disc at each ray's own wavelength,
//! Buie et al. (2003) in the aureole, and the circumsolar ratio as a share of
//! the power the rays carry, not of their count.
//!
//! `cargo test -p oc-core --test sun_source_validation -- --nocapture` prints
//! the tables behind the asserts.

#![cfg(feature = "scene3")]
use oc_core::heliostat::{LocalTime, Site};
use oc_core::scene3::{AngularDistribution3, Scene3};
use oc_core::solar::sunshape::{buie_chi, buie_gamma, limb_coefficients, SunDisc};
use oc_core::vec3::Vec3;

const AUREOLE_EDGE: f64 = 43.6e-3;

/// The rig as the DOM study builds it: Jena, 20 March, solar noon, the full
/// continuum, limb-darkened disc, the aureole asked for.
fn sun_rays(csr: f64) -> (Vec<(f64, f64, f64)>, f64) {
    sun_rays_with(csr, false)
}

fn sun_rays_with(csr: f64, disc_only: bool) -> (Vec<(f64, f64, f64)>, f64) {
    let site = Site {
        latitude_deg: 50.92781,
        longitude_deg: 11.57928,
        elevation_m: 160.0,
        pressure_mbar: 1000.0,
        temperature_c: 10.0,
    };
    let time = LocalTime { year: 2026, month: 3, day: 20, hour: 12, minute: 21, second: 0.0, utc_offset_h: 1.0 };
    let mut s = Scene3::new();
    s.add_sun(site, time, 0.0);
    s.solar.as_mut().unwrap().csr = csr;
    s.solar.as_mut().unwrap().dni_disc_only = disc_only;
    s.resolve_solar().expect("the sun resolves");
    let src = &mut s.sources[0];
    src.samples.spatial_samples = 4096;
    src.samples.angular_samples = 128;
    let half = match src.angular {
        AngularDistribution3::Sun { half_angle_rad, disc, csr: c } => {
            assert_eq!(disc, SunDisc::LimbDarkened);
            assert_eq!(c, csr);
            half_angle_rad
        }
        ref other => panic!("{other:?}"),
    };
    let axis = src.transform.rotate(Vec3::X);
    let rays = src
        .emit()
        .into_iter()
        .map(|r| (r.ray.dir.dot(axis).clamp(-1.0, 1.0).acos(), r.lambda_um, r.flux))
        .collect();
    (rays, half)
}

/// Share of the disc's light inside fractional radius `r`, at `lambda_um`:
/// `I(μ) r dr = I(μ) μ dμ`, so the share is `Σ aₖ (1 − μᵏ⁺²)/(k+2)` over
/// `Σ aₖ/(k+2)` with `μ = √(1 − r²)`, written out here from the paper's form
/// rather than taken from the sampler.
fn disc_cdf(r: f64, lambda_um: f64) -> f64 {
    let a = limb_coefficients(lambda_um);
    let mu = (1.0 - r * r).max(0.0).sqrt();
    let part = |m: f64| (0..6).map(|k| a[k] * (1.0 - m.powi(k as i32 + 2)) / (k as f64 + 2.0)).sum::<f64>();
    part(mu) / part(0.0)
}

/// The aureole carries the circumsolar ratio in power, not just in rays.
#[test]
fn the_aureole_carries_the_ratio_in_power() {
    println!("CSR asked | power outside disc | ray count outside | max ray flux / min");
    for csr in [0.0, 0.01, 0.05, 0.10] {
        let (rays, half) = sun_rays(csr);
        let total: f64 = rays.iter().map(|r| r.2).sum();
        let out: f64 = rays.iter().filter(|r| r.0 > half * (1.0 + 1e-9)).map(|r| r.2).sum();
        let count = rays.iter().filter(|r| r.0 > half * (1.0 + 1e-9)).count() as f64 / rays.len() as f64;
        let (lo, hi) = rays.iter().fold((f64::MAX, 0.0_f64), |(a, b), r| (a.min(r.2), b.max(r.2)));
        println!("{csr:9.2} | {:18.6} | {count:17.6} | {:.12}", out / total, hi / lo);
        assert!((out / total - csr).abs() < 1e-4, "CSR {csr}: power share {}", out / total);
        assert!(rays.iter().all(|r| r.0 <= AUREOLE_EDGE * (1.0 + 1e-9)));
    }
}

/// The disc's rays, binned in radius and in wavelength, against Neckel and
/// Labs at each ray's own wavelength.
#[test]
fn the_disc_follows_neckel_and_labs_by_wavelength() {
    let (rays, half) = sun_rays(0.05);
    let disc: Vec<_> = rays.iter().filter(|r| r.0 <= half * (1.0 + 1e-9)).collect();
    let bands = [(0.30, 0.45), (0.45, 0.60), (0.60, 0.80), (0.80, 1.10), (1.10, 2.60)];
    let nb = 10;
    println!("band um     | rays  | max |obs-exp| share per bin | chi2/dof | r_rms/r_disc");
    for (lo, hi) in bands {
        let sel: Vec<_> = disc.iter().filter(|r| r.1 >= lo && r.1 < hi).collect();
        let w: f64 = sel.iter().map(|r| r.2).sum();
        let mut obs = vec![0.0; nb];
        let mut exp = vec![0.0; nb];
        for r in &sel {
            let x = (r.0 / half).min(1.0 - 1e-12);
            obs[(x * nb as f64) as usize] += r.2 / w;
            // Expected: each ray's own wavelength, so a broad band is exact.
            for (k, e) in exp.iter_mut().enumerate() {
                let (a, b) = (k as f64 / nb as f64, (k + 1) as f64 / nb as f64);
                *e += r.2 / w * (disc_cdf(b, r.1) - disc_cdf(a, r.1));
            }
        }
        let n = sel.len() as f64;
        let chi2: f64 = obs.iter().zip(&exp).map(|(o, e)| (o - e).powi(2) * n / e).sum::<f64>() / (nb - 1) as f64;
        let dev = obs.iter().zip(&exp).map(|(o, e)| (o - e).abs()).fold(0.0, f64::max);
        let rms = (sel.iter().map(|r| r.2 * (r.0 / half).powi(2)).sum::<f64>() / w).sqrt();
        println!("{lo:.2}-{hi:.2}   | {:5} | {dev:.5} | {chi2:.3} | {rms:.4}", sel.len());
        // Sobol is far better than random: 0.003 is a fraction of one random sigma.
        assert!(dev < 3e-3, "{lo}-{hi} um: {dev}");
    }
}

/// The aureole's rays against Buie's `θ^γ`, and Buie's own profile against
/// the CSR the Tonatiuh fit says it delivers.
#[test]
fn the_aureole_follows_buie() {
    let nb = 8;
    println!("CSR  | gamma  | max |obs-exp| share per log bin | CSR of Buie's profile (own disc)");
    for csr in [0.01, 0.05, 0.10] {
        let (rays, half) = sun_rays(csr);
        let aur: Vec<_> = rays.iter().filter(|r| r.0 > half * (1.0 + 1e-9)).collect();
        let w: f64 = aur.iter().map(|r| r.2).sum();
        let g = buie_gamma(csr) + 2.0;
        let cdf = |t: f64| (t.powf(g) - half.powf(g)) / (AUREOLE_EDGE.powf(g) - half.powf(g));
        let edge = |k: usize| half * (AUREOLE_EDGE / half).powf(k as f64 / nb as f64);
        let mut dev = 0.0_f64;
        for k in 0..nb {
            let (a, b) = (edge(k), edge(k + 1));
            let obs: f64 = aur.iter().filter(|r| r.0 > a && r.0 <= b).map(|r| r.2).sum::<f64>() / w;
            dev = dev.max((obs - (cdf(b) - cdf(a))).abs());
        }
        // Buie's whole profile, disc and aureole, in mrad: the share beyond
        // the disc it delivers with the χ the fit picks.
        let chi = buie_chi(csr);
        let kappa = 0.9 * (13.5 * chi).ln() * chi.powf(-0.3);
        let gamma = 2.2 * (0.52 * chi).ln() * chi.powf(0.43) - 0.1;
        let m = 200_000;
        let (mut d, mut a) = (0.0, 0.0);
        for i in 0..m {
            let t = (i as f64 + 0.5) / m as f64 * 43.6;
            if t <= 4.65 {
                d += ((0.326 * t).cos() / (0.308 * t).cos()) * t;
            } else {
                a += kappa.exp() * t.powf(gamma) * t;
            }
        }
        println!("{csr:.2} | {:.3} | {dev:.5} | {:.4}", buie_gamma(csr), a / (a + d));
        assert!(dev < 3e-3, "CSR {csr}: {dev}");
        assert!((a / (a + d) - csr).abs() < 0.1 * csr + 2e-3, "Buie's profile gives {}", a / (a + d));
    }
}

/// A disc-only DNI (CAMS McClear's) puts all of it on the disc and the
/// aureole on top: the disc carries exactly what a pyrheliometer-style DNI of
/// the same value would give a sun with no aureole, the whole source
/// `1 / (1 − CSR)` of that, and every ray leaves in the same direction.
#[test]
fn a_disc_only_dni_adds_the_aureole_on_top() {
    for csr in [0.01, 0.05, 0.10] {
        let (shared, half) = sun_rays_with(csr, false);
        let (on_top, _) = sun_rays_with(csr, true);
        let (bare, _) = sun_rays_with(0.0, false);
        let total = |r: &[(f64, f64, f64)]| r.iter().map(|x| x.2).sum::<f64>();
        let disc = |r: &[(f64, f64, f64)]| r.iter().filter(|x| x.0 <= half * (1.0 + 1e-9)).map(|x| x.2).sum::<f64>();
        let i = total(&bare);
        assert!((total(&shared) - i).abs() < 1e-9 * i);
        assert!((total(&on_top) - i / (1.0 - csr)).abs() < 1e-9 * i, "CSR {csr}");
        assert!((disc(&on_top) - i).abs() < 2e-4 * i, "CSR {csr}: disc {} of {i}", disc(&on_top));
        assert_eq!(shared.len(), on_top.len());
        for (a, b) in shared.iter().zip(&on_top) {
            assert_eq!((a.0, a.1), (b.0, b.1));
            assert!((b.2 * (1.0 - csr) - a.2).abs() < 1e-12 * a.2);
        }
        println!("CSR {csr}: disc {:.6} I, total {:.6} I", disc(&on_top) / i, total(&on_top) / i);
    }
}
