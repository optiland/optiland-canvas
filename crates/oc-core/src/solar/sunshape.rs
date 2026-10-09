//! The sun's brightness across the sky near it: the disc, darker toward its
//! limb, and the aureole of forward-scattered light around it.
//!
//! A sampler here turns two uniform numbers into a polar angle from the
//! sun's centre, so a source can draw each ray's direction from the sun as it
//! looks rather than from a flat disc.
//!
//! # The disc
//!
//! - [`SunDisc::Uniform`]: every point of the disc equally bright. What the
//!   app drew before there was a choice, and what a "pillbox" sun is in the
//!   concentrator literature.
//! - [`SunDisc::LimbDarkened`]: the measured centre-to-limb variation of Neckel
//!   & Labs, *On the wavelength dependency of solar limb darkening (λλ303 to
//!   1099 nm)*, Solar Physics 153, 91–114 (1994), Table 1: a fifth-order
//!   polynomial `I(μ)/I(1) = Σ Aₖ μᵏ` in `μ = cos ϑ`, the cosine of the angle
//!   between the line of sight and the solar surface normal, at 30 continuum
//!   wavelengths. Between them the coefficients are interpolated linearly in
//!   wavelength, as the Swedish Solar Telescope's reduction pipeline does
//!   (`red_neckel_coefficients.pro`), and renormalised to 1 at the centre.
//!   Outside 303–1099 nm the nearest row is held: the paper has no data
//!   there, so the infrared limb is taken as dark as at 1099 nm, where it is
//!   already the weakest of the table. Blue light is darker at the limb than
//!   red, so the sun a lens sees is a slightly different size in each colour.
//! - [`SunDisc::Buie`]: the broadband disc of Buie, Monger & Dey, *Sunshape
//!   distributions for terrestrial solar simulations*, Solar Energy 74,
//!   113–122 (2003): `φ(θ) = cos(0.326 θ) / cos(0.308 θ)`, `θ` in mrad over a
//!   4.65 mrad disc, scaled here to the disc's size on the date.
//!
//! # The aureole
//!
//! The circumsolar ratio `CSR` is the share of the light within 43.6 mrad
//! (2.5°, a pyrheliometer's field) that comes from outside the disc. The
//! aureole's shape is Buie's, `φ(θ) = e^κ θ^γ` out to 43.6 mrad with `κ, γ`
//! functions of a parameter `χ`. Buie's `χ` is not the CSR his profile
//! delivers; Tonatiuh (CENER's concentrator ray tracer, `SunshapeBuie.cpp`)
//! fits `χ` as a function of the CSR asked for, and that fit sets the shape
//! here. The energy split is then imposed exactly, a share `CSR` of the rays
//! in the aureole and the rest on the disc, whatever disc is chosen.
//!
//! # Angles
//!
//! The sun is small, so a point at fractional radius `r` on the disc is seen
//! at `θ = r · θ_disc`, and the aureole's `sin θ` is taken as `θ`. Both are
//! exact to `3·10⁻⁴` relative at the aureole's edge and to `10⁻⁵` on the disc.

use serde::{Deserialize, Serialize};

/// How bright the solar disc is across its face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SunDisc {
    /// Evenly bright out to its edge.
    #[default]
    Uniform,
    /// Neckel & Labs (1994), by wavelength.
    LimbDarkened,
    /// Buie et al. (2003), one profile for every wavelength.
    Buie,
}

impl SunDisc {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::LimbDarkened => "limb",
            Self::Buie => "buie",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "uniform" => Some(Self::Uniform),
            "limb" => Some(Self::LimbDarkened),
            "buie" => Some(Self::Buie),
            _ => None,
        }
    }
}

/// The outer edge of Buie's aureole, radians: 43.6 mrad.
pub const AUREOLE_EDGE_RAD: f64 = 0.0436;

/// The disc radius Buie's profile is written for, mrad.
const BUIE_DISC_MRAD: f64 = 4.65;

/// The largest circumsolar ratio Tonatiuh's `χ` fit covers.
pub const MAX_CSR: f64 = 0.849;

/// Neckel & Labs (1994), Table 1: wavelength (nm) and `A₀..A₅`, as the
/// Swedish Solar Telescope's pipeline transcribes it. Each row should sum to
/// 1; the 365.875 nm row sums to 0.998, which [`limb_coefficients`]'s
/// renormalisation absorbs.
const NECKEL_LABS_1994: [[f64; 7]; 30] = [
    [303.327, 0.08011, 0.70695, 0.49910, -0.31080, -0.02177, 0.04642],
    [310.843, 0.08160, 0.71609, 0.69685, -0.87703, 0.47008, -0.08760],
    [320.468, 0.08833, 0.77285, 0.65382, -1.04647, 0.72921, -0.19775],
    [329.897, 0.09188, 0.92459, 0.19604, -0.39546, 0.23599, -0.05303],
    [349.947, 0.11012, 1.02168, -0.10924, -0.00055, -0.08688, 0.06487],
    [365.875, 0.12828, 1.04969, 0.17482, -1.13371, 1.23882, -0.45990],
    [374.086, 0.12579, 0.85402, 0.54601, -1.15048, 0.88928, -0.26462],
    [390.915, 0.12995, 0.91836, -0.07566, 0.19149, -0.28712, 0.12298],
    [401.970, 0.12323, 1.08648, -0.43974, 0.45912, -0.32759, 0.09850],
    [416.319, 0.12814, 1.19947, -0.84407, 1.07201, -0.79537, 0.23982],
    [427.930, 0.14249, 1.28796, -1.19564, 1.68603, -1.36658, 0.44572],
    [443.885, 0.16220, 1.24893, -0.92165, 0.89978, -0.50148, 0.11220],
    [445.125, 0.15248, 1.38517, -1.49615, 1.99886, -1.48155, 0.44119],
    [457.345, 0.16604, 1.38544, -1.52275, 2.00232, -1.45969, 0.42864],
    [477.427, 0.19571, 1.30551, -1.25845, 1.50626, -1.05472, 0.30570],
    [492.905, 0.20924, 1.30798, -1.20411, 1.21505, -0.67196, 0.14381],
    [519.930, 0.23695, 1.29927, -1.28034, 1.37760, -0.85054, 0.21706],
    [541.760, 0.26073, 1.27428, -1.30352, 1.47085, -0.96618, 0.26384],
    [559.950, 0.26892, 1.34319, -1.58427, 1.91271, -1.31350, 0.37295],
    [579.880, 0.28392, 1.36896, -1.75998, 2.22154, -1.56074, 0.44630],
    [610.975, 0.30854, 1.36620, -1.83572, 2.33221, -1.63082, 0.45959],
    [640.970, 0.33644, 1.30590, -1.79238, 2.45040, -1.89979, 0.59943],
    [669.400, 0.34685, 1.37539, -2.04425, 2.70493, -1.94290, 0.55999],
    [700.875, 0.37885, 1.25553, -1.70908, 2.19647, -1.59554, 0.47378],
    [748.710, 0.40627, 1.22842, -1.67877, 2.05535, -1.39972, 0.38845],
    [811.760, 0.42977, 1.25182, -1.85164, 2.31949, -1.59101, 0.44155],
    [869.600, 0.45396, 1.25101, -2.02958, 2.75410, -2.02287, 0.59338],
    [948.850, 0.47855, 1.19813, -1.86296, 2.36939, -1.64367, 0.46056],
    [1046.60, 0.49870, 1.21429, -2.06976, 2.80703, -2.05247, 0.60221],
    [1098.95, 0.51149, 1.19354, -2.00174, 2.66936, -1.94981, 0.57715],
];

/// The limb-darkening coefficients `A₀..A₅` at `lambda_um`, interpolated
/// linearly between the table's wavelengths, held at its ends, and scaled so
/// `I(1) = Σ Aₖ = 1`.
pub fn limb_coefficients(lambda_um: f64) -> [f64; 6] {
    let nm = lambda_um * 1000.0;
    let t = &NECKEL_LABS_1994;
    let row = |r: &[f64; 7]| [r[1], r[2], r[3], r[4], r[5], r[6]];
    let mut a = if nm <= t[0][0] {
        row(&t[0])
    } else if nm >= t[t.len() - 1][0] {
        row(&t[t.len() - 1])
    } else {
        let k = t.iter().position(|r| r[0] > nm).expect("inside the table") - 1;
        let f = (nm - t[k][0]) / (t[k + 1][0] - t[k][0]);
        let (lo, hi) = (row(&t[k]), row(&t[k + 1]));
        std::array::from_fn(|i| lo[i] + f * (hi[i] - lo[i]))
    };
    let sum: f64 = a.iter().sum();
    for x in &mut a {
        *x /= sum;
    }
    a
}

/// `I(μ)/I(1)` for coefficients `a`.
pub fn limb_intensity(a: &[f64; 6], mu: f64) -> f64 {
    a.iter().rev().fold(0.0, |acc, c| acc * mu + c)
}

/// The disc's brightness at fractional radius `r` (0 at the centre, 1 at the
/// limb), relative to its centre.
pub fn disc_radiance(disc: SunDisc, r: f64, lambda_um: f64) -> f64 {
    if !(0.0..=1.0).contains(&r) {
        return 0.0;
    }
    match disc {
        SunDisc::Uniform => 1.0,
        SunDisc::LimbDarkened => {
            limb_intensity(&limb_coefficients(lambda_um), (1.0 - r * r).max(0.0).sqrt())
        }
        SunDisc::Buie => buie_disc(r),
    }
}

fn buie_disc(r: f64) -> f64 {
    let t = r * BUIE_DISC_MRAD;
    (0.326 * t).cos() / (0.308 * t).cos()
}

/// Fractional radius on the limb-darkened disc for a uniform `u` in `[0, 1)`.
///
/// Light from an annulus of the disc is its brightness times its area,
/// `I(μ) r dr`, and with `μ = √(1 − r²)` that is `I(μ) μ dμ`: a polynomial, so
/// the share inside `μ` is one too, `C(μ) = Σ Aₖ (1 − μᵏ⁺²)/(k+2)` over its
/// total `Σ Aₖ/(k+2)`. Inverted by bisection, which a monotone polynomial on
/// `[0, 1]` needs no more than.
fn limb_radius(a: &[f64; 6], u: f64) -> f64 {
    let total: f64 = a.iter().enumerate().map(|(k, c)| c / (k as f64 + 2.0)).sum();
    let inside = |mu: f64| -> f64 {
        a.iter()
            .enumerate()
            .map(|(k, c)| c * (1.0 - mu.powi(k as i32 + 2)) / (k as f64 + 2.0))
            .sum::<f64>()
            / total
    };
    // `inside` grows from 0 at μ = 1 (the centre) to 1 at μ = 0 (the limb).
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if inside(mid) > u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mu = 0.5 * (lo + hi);
    (1.0 - mu * mu).max(0.0).sqrt()
}

/// Fractional radius on Buie's disc for a uniform `u`, from its light inside
/// each radius tabulated once on a fine grid.
fn buie_radius(u: f64) -> f64 {
    use std::sync::OnceLock;
    const N: usize = 2048;
    static CDF: OnceLock<Vec<f64>> = OnceLock::new();
    let cdf = CDF.get_or_init(|| {
        // Simpson on each cell of `f(r) r`, which is smooth.
        let f = |r: f64| buie_disc(r) * r;
        let mut out = Vec::with_capacity(N + 1);
        let mut acc = 0.0;
        out.push(0.0);
        for i in 0..N {
            let (a, b) = (i as f64 / N as f64, (i + 1) as f64 / N as f64);
            acc += (b - a) / 6.0 * (f(a) + 4.0 * f(0.5 * (a + b)) + f(b));
            out.push(acc);
        }
        let total = acc;
        out.iter().map(|v| v / total).collect()
    });
    let k = cdf.partition_point(|c| *c <= u).clamp(1, N);
    let (c0, c1) = (cdf[k - 1], cdf[k]);
    let f = if c1 > c0 { (u - c0) / (c1 - c0) } else { 0.0 };
    // Within one cell the light grows as `r dr`; interpolating in `r²` keeps
    // the centre from clumping.
    let (r0, r1) = ((k - 1) as f64 / N as f64, k as f64 / N as f64);
    (r0 * r0 + f * (r1 * r1 - r0 * r0)).sqrt()
}

/// Tonatiuh's fit of Buie's `χ` to the circumsolar ratio it delivers.
pub fn buie_chi(csr: f64) -> f64 {
    if csr > 0.145 {
        -0.04419909985804843
            + csr
                * (1.401323894233574
                    + csr * (-0.3639746714505299 + csr * (-0.9579768560161194 + 1.1550475450828657 * csr)))
    } else if csr > 0.035 {
        0.022652077593662934 + csr * (0.5252380349996234 + (2.5484334534423887 - 0.8763755326550412 * csr) * csr)
    } else {
        0.004733749294807862
            + csr
                * (4.716738065192151
                    + csr
                        * (-463.506669149804
                            + csr * (24745.88727411664 + csr * (-606122.7511711778 + 5521693.445014727 * csr))))
    }
}

/// Buie's aureole exponent `γ` for a circumsolar ratio.
pub fn buie_gamma(csr: f64) -> f64 {
    let chi = buie_chi(csr);
    2.2 * (0.52 * chi).ln() * chi.powf(0.43) - 0.1
}

/// The sun as a direction distribution about its centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunShape {
    /// The disc's angular radius, radians.
    pub half_angle_rad: f64,
    pub disc: SunDisc,
    /// Share of the light in the aureole, `[0, MAX_CSR]`.
    pub csr: f64,
}

impl SunShape {
    /// The polar angle from the sun's centre, radians, for a uniform `u` in
    /// `[0, 1)`, at wavelength `lambda_um`. The first `csr` of `u` lands in
    /// the aureole and the rest on the disc, each stretched back to `[0, 1)`,
    /// so a stratified `u` stays stratified in both.
    pub fn polar_angle(&self, u: f64, lambda_um: f64) -> f64 {
        let ts = self.half_angle_rad;
        let csr = self.csr.clamp(0.0, MAX_CSR);
        if csr > 0.0 && u < csr && ts < AUREOLE_EDGE_RAD {
            // Light from a ring of the aureole goes as `θ^γ · θ dθ`, so the
            // share inside `θ` goes as `θ^(γ+2)`.
            let w = u / csr;
            let g = buie_gamma(csr) + 2.0;
            let (a, b) = (ts, AUREOLE_EDGE_RAD);
            return if g.abs() < 1e-9 {
                a * (b / a).powf(w)
            } else {
                (a.powf(g) + w * (b.powf(g) - a.powf(g))).powf(1.0 / g)
            };
        }
        let w = if csr > 0.0 { (u - csr) / (1.0 - csr) } else { u };
        let w = w.clamp(0.0, 1.0);
        let r = match self.disc {
            SunDisc::Uniform => {
                // Uniform in solid angle over the cone, exactly.
                let cos_max = ts.cos();
                return (1.0 - w * (1.0 - cos_max)).clamp(-1.0, 1.0).acos();
            }
            SunDisc::LimbDarkened => limb_radius(&limb_coefficients(lambda_um), w),
            SunDisc::Buie => buie_radius(w),
        };
        r * ts
    }

    /// The farthest a ray can leave the sun's centre, radians.
    pub fn max_angle_rad(&self) -> f64 {
        if self.csr > 0.0 {
            AUREOLE_EDGE_RAD.max(self.half_angle_rad)
        } else {
            self.half_angle_rad
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_of_the_table_is_one_at_the_centre() {
        // The paper normalises each row to 1 at the centre. Every row does, to
        // its printed digits, but one: 365.875 nm sums to 0.998 in the copy
        // this table comes from, a slip in a single digit somewhere in it.
        // `limb_coefficients` renormalises, which absorbs it.
        for r in &NECKEL_LABS_1994 {
            let sum: f64 = r[1..].iter().sum();
            let tol = if r[0] == 365.875 { 3e-3 } else { 5e-5 };
            assert!((sum - 1.0).abs() < tol, "{} nm sums to {sum}", r[0]);
        }
        for nm in [250.0, 303.327, 500.0, 519.93, 1098.95, 2000.0] {
            let a = limb_coefficients(nm / 1000.0);
            assert!((limb_intensity(&a, 1.0) - 1.0).abs() < 1e-12);
        }
    }

    /// The limb is darker in the blue than in the red: at μ = 0.1, the
    /// table's own numbers.
    #[test]
    fn blue_is_darker_at_the_limb_than_red() {
        let edge = |nm: f64| limb_intensity(&limb_coefficients(nm / 1000.0), 0.1);
        assert!(edge(400.0) < edge(550.0) && edge(550.0) < edge(1000.0));
        // At 519.93 nm the table row itself, renormalised.
        let r = NECKEL_LABS_1994[16];
        let sum: f64 = r[1..].iter().sum();
        let want: f64 = (0..6).map(|k| r[k + 1] * 0.1_f64.powi(k as i32)).sum::<f64>() / sum;
        assert!((edge(519.93) - want).abs() < 1e-12);
    }

    /// Sampled radii reproduce the disc's light: the share of rays inside half
    /// the radius is the integral of `I r dr` there.
    #[test]
    fn the_disc_samplers_reproduce_their_profiles() {
        for disc in [SunDisc::Uniform, SunDisc::LimbDarkened, SunDisc::Buie] {
            let shape = SunShape { half_angle_rad: 4.65e-3, disc, csr: 0.0 };
            let n = 20_000;
            let inside = (0..n)
                .filter(|i| shape.polar_angle((*i as f64 + 0.5) / n as f64, 0.5) < 0.5 * 4.65e-3)
                .count() as f64
                / n as f64;
            // The same share by quadrature of the profile.
            let m = 4000;
            let (mut half, mut all) = (0.0, 0.0);
            for i in 0..m {
                let r = (i as f64 + 0.5) / m as f64;
                let w = disc_radiance(disc, r, 0.5) * r;
                all += w;
                if r < 0.5 {
                    half += w;
                }
            }
            assert!((inside - half / all).abs() < 2e-3, "{disc:?}: {inside} vs {}", half / all);
        }
        // Limb darkening concentrates the light: more of it inside half the
        // radius than the uniform disc's quarter.
        let limb = SunShape { half_angle_rad: 1.0, disc: SunDisc::LimbDarkened, csr: 0.0 };
        let share = (0..1000).filter(|i| limb.polar_angle((*i as f64 + 0.5) / 1000.0, 0.5) < 0.5).count();
        assert!(share > 280, "{share} of 1000");
    }

    /// The aureole carries exactly the share of light asked for, and reaches
    /// 43.6 mrad and no farther.
    #[test]
    fn the_aureole_carries_the_circumsolar_ratio() {
        for csr in [0.01, 0.05, 0.2, 0.4] {
            let shape = SunShape { half_angle_rad: 4.65e-3, disc: SunDisc::Buie, csr };
            let n = 10_000;
            let angles: Vec<f64> = (0..n).map(|i| shape.polar_angle((i as f64 + 0.5) / n as f64, 0.5)).collect();
            let out = angles.iter().filter(|t| **t > 4.65e-3 + 1e-12).count() as f64 / n as f64;
            assert!((out - csr).abs() <= 1.0 / n as f64, "CSR {csr}: {out}");
            assert!(angles.iter().all(|t| *t <= AUREOLE_EDGE_RAD * (1.0 + 1e-12)));
            // Buie's aureole falls with angle: γ is negative over this range.
            assert!(buie_gamma(csr) < -1.0, "γ {}", buie_gamma(csr));
        }
    }

    /// Tonatiuh's fit is continuous where its pieces meet.
    #[test]
    fn the_chi_fit_is_continuous() {
        for at in [0.035, 0.145] {
            assert!((buie_chi(at - 1e-9) - buie_chi(at + 1e-9)).abs() < 2e-3, "at {at}");
        }
    }
}
