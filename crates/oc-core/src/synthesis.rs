//! Substrate-adapted coating design (coating-fidelity spec CF-08).
//!
//! The older presets (`SingleLayerAr` of constant MgF₂, the quarter-half-
//! quarter `BroadbandArLegacy`) ignore the substrate, the band and real
//! materials: the eyepiece study found the legacy three-layer AR reflecting
//! about 10% of the visible on N-SF66, worse than one MgF₂ layer. The designs
//! here look at all three:
//!
//! - `SingleLayerAr` (adapted): the catalogue film whose index at the design
//!   wavelength is nearest the ideal `√(n₀ n_s)`, a quarter wave thick;
//! - `VCoat`: two layers with an exact null at the design wavelength, from the
//!   closed-form solution of the admittance-locus construction (Macleod,
//!   *Thin-Film Optical Filters*, ch. 4); both solutions are computed and the
//!   thinner kept, and a pair that cannot null the substrate is refused by name;
//! - `Bbar`: quarter-half-quarter with the inner quarter matched to the
//!   substrate, `n_low √(n_s / n₀)`, from the catalogue; MgF₂ outside and a
//!   half wave of Ta₂O₅ by default (TiO₂ is so dispersive that its half wave
//!   loses to one MgF₂ layer on dense flint);
//! - `BandAr`: up to 8 layers, the thicknesses refined by a bounded,
//!   deterministic Nelder-Mead over a declared band and angle range;
//! - `EnhancedMetal`: a metal under quarter-wave high/low pairs.
//!
//! A synthesized broadband AR is never worse than the adapted single layer over
//! its own band and angles: if it would be, synthesis falls back to fewer
//! layers. Every material comes from the catalogue ([`crate::film_catalog`]),
//! at its index at the design wavelength. The default low-index film is MgF₂
//! at the bulk crystal's ordinary index, which a hot-deposited AR layer sits
//! near; the catalogue's 44 nm MgF₂ film reads 0.04 higher, enough to turn the
//! adapted single layer from better than the old constant one to worse.

use serde::{Deserialize, Serialize};

use crate::thinfilm::{coating_rt_at, Coating, CoatingBacking, CoatingDesign, DesignKind, FilmLayer, FilmMaterial, Incidence};

/// A built-in catalogue entry, by key: `Copy`, so a design can name its
/// materials and stay a plain value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilmKey {
    MagnesiumFluoride,
    MagnesiumFluorideBulk,
    Silica,
    Alumina,
    Titania,
    Tantala,
    Hafnia,
    Zirconia,
    Niobia,
    ZincSulfide,
    Silicon,
    Silver,
    SilverJohnson,
    Aluminium,
    Gold,
    Chromium,
    Nickel,
}

impl FilmKey {
    /// The dielectric films an AR can be built from.
    pub const DIELECTRICS: [FilmKey; 10] = [
        FilmKey::MagnesiumFluoride,
        FilmKey::MagnesiumFluorideBulk,
        FilmKey::Silica,
        FilmKey::Alumina,
        FilmKey::Titania,
        FilmKey::Tantala,
        FilmKey::Hafnia,
        FilmKey::Zirconia,
        FilmKey::Niobia,
        FilmKey::ZincSulfide,
    ];

    pub fn key(self) -> &'static str {
        match self {
            FilmKey::MagnesiumFluoride => "magnesium_fluoride",
            FilmKey::MagnesiumFluorideBulk => "magnesium_fluoride_bulk",
            FilmKey::Silica => "silica",
            FilmKey::Alumina => "alumina",
            FilmKey::Titania => "titania",
            FilmKey::Tantala => "tantala",
            FilmKey::Hafnia => "hafnia",
            FilmKey::Zirconia => "zirconia",
            FilmKey::Niobia => "niobia",
            FilmKey::ZincSulfide => "zinc_sulfide",
            FilmKey::Silicon => "silicon",
            FilmKey::Silver => "silver",
            FilmKey::SilverJohnson => "silver_johnson",
            FilmKey::Aluminium => "aluminium",
            FilmKey::Gold => "gold",
            FilmKey::Chromium => "chromium",
            FilmKey::Nickel => "nickel",
        }
    }

    pub fn material(self) -> FilmMaterial {
        crate::film_catalog::builtin_named(self.key()).expect("every key is a built-in entry")
    }
}

/// A metal for an enhanced mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetalKind {
    Silver,
    Aluminium,
    Gold,
}

impl MetalKind {
    pub fn key(self) -> FilmKey {
        match self {
            MetalKind::Silver => FilmKey::Silver,
            MetalKind::Aluminium => FilmKey::Aluminium,
            MetalKind::Gold => FilmKey::Gold,
        }
    }
}

/// How a band-averaged reflectance weighs its wavelengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DesignWeight {
    #[default]
    Uniform,
    /// The ASTM G173-03 direct-normal sun: what a solar concentrator's AR is
    /// designed for.
    SunDirect,
}

impl DesignWeight {
    fn at(self, lambda_um: f64) -> f64 {
        match self {
            DesignWeight::Uniform => 1.0,
            DesignWeight::SunDirect => {
                let t = crate::solar::g173_direct();
                let nm = lambda_um * 1000.0;
                if nm <= t[0][0] {
                    return t[0][1];
                }
                let i = t.partition_point(|r| r[0] <= nm).min(t.len() - 1);
                let (a, b) = (t[i - 1], t[i]);
                a[1] + (b[1] - a[1]) * (nm - a[0]) / (b[0] - a[0])
            }
        }
    }
}

/// What a substrate-adapted design is adapted to, and what it was designed
/// to: kept with the design so the editor can say so and re-synthesize.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DesignAdapt {
    /// The band the design is judged over, µm.
    pub band_um: [f64; 2],
    /// The angles of incidence it is judged over, degrees.
    pub aoi_range_deg: [f64; 2],
    /// The substrate's index at the design wavelength.
    pub substrate_n: f64,
    /// The materials, by role: low, high, and the matched inner quarter of a
    /// `Bbar` (unused by the other kinds). `None` takes the default for the
    /// role; for the matched quarter that is the catalogue film nearest the
    /// index the substrate calls for.
    #[serde(default)]
    pub materials: [Option<FilmKey>; 3],
    /// R and T at the design wavelength and angle, unpolarized, recorded when
    /// the design was synthesized. `coating_model: Nominal` answers with the
    /// layers' own response at the design point, which is this number until a
    /// layer is edited.
    #[serde(default)]
    pub design_point: Option<[f64; 2]>,
    #[serde(default)]
    pub weight: DesignWeight,
}

impl DesignAdapt {
    /// Adapted to `substrate_n`, judged over the visible at the design angle.
    pub fn visible(substrate_n: f64, aoi_deg: f64) -> Self {
        Self {
            band_um: [0.4, 0.7],
            aoi_range_deg: [aoi_deg, aoi_deg],
            substrate_n,
            materials: [None; 3],
            design_point: None,
            weight: DesignWeight::Uniform,
        }
    }
}

/// The band-averaged unpolarized reflectance of `c` between `n0` and
/// `n_far`: 41 wavelengths uniform in `band_um` by 5 angles uniform in
/// `aoi_range_deg`, weighted by `weight`, summed in a fixed order.
pub fn band_reflectance(c: &Coating, n0: f64, n_far: f64, band_um: [f64; 2], aoi_range_deg: [f64; 2], weight: DesignWeight) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..41 {
        let lam = band_um[0] + (band_um[1] - band_um[0]) * i as f64 / 40.0;
        let w = weight.at(lam);
        for j in 0..5 {
            let deg = aoi_range_deg[0] + (aoi_range_deg[1] - aoi_range_deg[0]) * j as f64 / 4.0;
            let r = coating_rt_at(c, n0, n_far, lam, deg.to_radians().cos()).map_or(1.0, |v| v.r());
            num += w * r;
            den += w;
        }
    }
    if den > 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

/// The real index of a catalogue entry at `lambda_um`.
fn n_at(key: FilmKey, lambda_um: f64) -> f64 {
    key.material().index_at(lambda_um).map_or(f64::NAN, |c| c.re)
}

/// The catalogue dielectric whose index at `lambda_um` is nearest `target`
/// (ties to the earlier entry).
fn nearest(target: f64, lambda_um: f64) -> FilmKey {
    let mut best = FilmKey::DIELECTRICS[0];
    let mut gap = f64::INFINITY;
    for k in FilmKey::DIELECTRICS {
        let g = (n_at(k, lambda_um) - target).abs();
        if g < gap {
            gap = g;
            best = k;
        }
    }
    best
}

fn quarter(key: FilmKey, d: &CoatingDesign, n0: f64) -> FilmLayer {
    FilmLayer::quarter_wave(key.material(), d.lambda_nm, n0, d.aoi_deg)
}

/// A layer of `key` with phase thickness `delta` at the design wavelength and
/// angle.
fn with_phase(key: FilmKey, delta: f64, d: &CoatingDesign, n0: f64) -> FilmLayer {
    let mut l = quarter(key, d, n0);
    l.thickness_nm *= delta / std::f64::consts::FRAC_PI_2;
    l
}

fn on(layers: Vec<FilmLayer>) -> Coating {
    Coating { layers, backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None }
}

/// Record in an adapted design the substrate it was built on and its R and T
/// at the design wavelength and angle: what the editor shows next to the curve.
/// A design that is not adapted is returned as it came.
pub(crate) fn record_design_point(mut c: Coating, n0: f64, n_far: f64) -> Coating {
    let Some(d) = c.design else {
        return c;
    };
    if d.adapt.is_none() {
        return c;
    }
    let point = coating_rt_at(&c, n0, n_far, d.lambda_nm / 1000.0, d.aoi_deg.to_radians().cos()).ok().map(|v| [v.r(), v.t()]);
    if let Some(a) = c.design.as_mut().and_then(|d| d.adapt.as_mut()) {
        a.design_point = point;
    }
    c
}

/// The layers a substrate-adapted design calls for, between `n0` and a
/// substrate of index `n_s`, or why there are none.
///
/// Every thickness is quantized to 1e-6 nm. A dispersion formula goes through
/// the platform's `powf`, which may differ by an ulp between the Windows and
/// Linux runners; on a grid a million times coarser than that ulp the layer
/// lists come out bit-identical, and a femtometre is no physics.
pub fn synthesize(d: &CoatingDesign, n0: f64, n_s: f64) -> Result<Vec<FilmLayer>, String> {
    let mut layers = synthesize_raw(d, n0, n_s)?;
    for l in &mut layers {
        l.thickness_nm = (l.thickness_nm * 1e6).round() / 1e6;
    }
    Ok(layers)
}

fn synthesize_raw(d: &CoatingDesign, n0: f64, n_s: f64) -> Result<Vec<FilmLayer>, String> {
    let adapt = d.adapt.unwrap_or_else(|| DesignAdapt::visible(n_s, d.aoi_deg));
    let lam = d.lambda_nm / 1000.0;
    let role = |i: usize, default: FilmKey| adapt.materials[i].unwrap_or(default);
    match d.kind {
        DesignKind::SingleLayerAr => Ok(vec![quarter(nearest((n0 * n_s).sqrt(), lam), d, n0)]),
        DesignKind::VCoat => vcoat(d, n0, n_s, role(0, FilmKey::Silica), role(1, FilmKey::Titania)),
        DesignKind::Bbar => {
            let full = bbar(d, n0, n_s, role(0, FilmKey::MagnesiumFluorideBulk), role(1, FilmKey::Tantala), adapt.materials[2]);
            let pair = vec![full[0].clone(), full[2].clone()];
            Ok(first_not_worse(vec![full, pair], d, &adapt, n0, n_s))
        }
        DesignKind::BandAr => band_ar(d, &adapt, n0, n_s),
        DesignKind::EnhancedMetal { metal, pairs } => {
            let (lo, hi) = (role(0, FilmKey::Silica), role(1, FilmKey::Tantala));
            let mut layers = Vec::new();
            for _ in 0..pairs.clamp(1, 20) {
                layers.push(quarter(hi, d, n0));
                layers.push(quarter(lo, d, n0));
            }
            layers.push(FilmLayer::new(metal.key().material(), 150.0));
            Ok(layers)
        }
        other => Err(format!("{other:?} is not a substrate-adapted design")),
    }
}

/// Two layers, `outer` next to the incident medium and `inner` next to the
/// substrate, with an exact null at the design wavelength. With `t₁`, `t₂`
/// the tangents of the two phase thicknesses, `r = 0` is two real equations
/// (normal incidence, the layers' real indices `n₁`, `n₂`):
///
/// ```text
/// t₁ t₂ = n₁ n₂ (n_s − n₀) / (n₁² n_s − n₀ n₂²)
/// t₂ = t₁ · n₂ (n₁² − n₀ n_s) / (n₁ (n₀ n_s − n₂²))
/// ```
///
/// so `t₁² = P / K` with `P` the first right-hand side and `K` the factor in
/// the second. A real solution needs `P / K > 0`; it comes as a pair, `±t₁`,
/// the two crossings of the admittance loci, and the thinner is kept. The
/// thicknesses are tilted for the design angle, where the null is exact for
/// s and very nearly so for p.
fn vcoat(d: &CoatingDesign, n0: f64, n_s: f64, outer: FilmKey, inner: FilmKey) -> Result<Vec<FilmLayer>, String> {
    let lam = d.lambda_nm / 1000.0;
    let (n1, n2) = (n_at(outer, lam), n_at(inner, lam));
    let p = n1 * n2 * (n_s - n0) / (n1 * n1 * n_s - n0 * n2 * n2);
    let k = n2 * (n1 * n1 - n0 * n_s) / (n1 * (n0 * n_s - n2 * n2));
    let t1_sq = p / k;
    if !(t1_sq.is_finite() && t1_sq > 0.0) {
        return Err(format!(
            "a V-coat of {} over {} cannot null a substrate of index {n_s:.4} at {:.0} nm: no real thickness solves it",
            outer.key(),
            inner.key(),
            d.lambda_nm
        ));
    }
    let wrap = |x: f64| if x < 0.0 { x + std::f64::consts::PI } else { x };
    let mut best: Option<(f64, Vec<FilmLayer>)> = None;
    for sign in [1.0, -1.0] {
        let t1 = sign * t1_sq.sqrt();
        let t2 = k * t1;
        let (d1, d2) = (wrap(t1.atan()), wrap(t2.atan()));
        let layers = vec![with_phase(outer, d1, d, n0), with_phase(inner, d2, d, n0)];
        let total: f64 = layers.iter().map(|l| l.thickness_nm).sum();
        if best.as_ref().map_or(true, |(b, _)| total < *b) {
            best = Some((total, layers));
        }
    }
    Ok(best.expect("two candidates").1)
}

/// Quarter-half-quarter: a quarter of `low` outside, a half of `high`, and
/// inside a quarter of the catalogue film nearest `n_low √(n_s / n₀)` (or
/// `matched`, when the design names it), which makes the two quarters a
/// matched pair for this substrate.
fn bbar(d: &CoatingDesign, n0: f64, n_s: f64, low: FilmKey, high: FilmKey, matched: Option<FilmKey>) -> Vec<FilmLayer> {
    let lam = d.lambda_nm / 1000.0;
    let matched = matched.unwrap_or_else(|| nearest(n_at(low, lam) * (n_s / n0).sqrt(), lam));
    let mut half = quarter(high, d, n0);
    half.thickness_nm *= 2.0;
    vec![quarter(low, d, n0), half, quarter(matched, d, n0)]
}

/// The adapted single layer for this substrate.
fn single_layer(d: &CoatingDesign, n0: f64, n_s: f64) -> Vec<FilmLayer> {
    vec![quarter(nearest((n0 * n_s).sqrt(), d.lambda_nm / 1000.0), d, n0)]
}

/// The first of `candidates` (most layers first) whose band-averaged
/// reflectance over the design's band and angles is no worse than the adapted
/// single layer's, or the single layer: the rule that a synthesized AR is never
/// worse than one layer.
/// Candidates are taken lazily, so a refinement after the first good one is
/// never run.
fn first_not_worse(candidates: impl IntoIterator<Item = Vec<FilmLayer>>, d: &CoatingDesign, adapt: &DesignAdapt, n0: f64, n_s: f64) -> Vec<FilmLayer> {
    let score = |layers: &[FilmLayer]| band_reflectance(&on(layers.to_vec()), n0, n_s, adapt.band_um, adapt.aoi_range_deg, adapt.weight);
    let single = single_layer(d, n0, n_s);
    let floor = score(&single);
    candidates.into_iter().find(|c| score(c) <= floor).unwrap_or(single)
}

/// A broadband AR refined over its band and angle range, never worse than
/// the adapted single layer there.
fn band_ar(d: &CoatingDesign, adapt: &DesignAdapt, n0: f64, n_s: f64) -> Result<Vec<FilmLayer>, String> {
    let score = |layers: &[FilmLayer]| band_reflectance(&on(layers.to_vec()), n0, n_s, adapt.band_um, adapt.aoi_range_deg, adapt.weight);
    let low = adapt.materials[0].unwrap_or(FilmKey::MagnesiumFluorideBulk);
    let high = adapt.materials[1].unwrap_or(FilmKey::Tantala);
    // Starting points, most layers first: the matched quarter-half-quarter,
    // the two quarters without the half wave, the single layer; each refined,
    // the first no worse than the single layer kept.
    let b = bbar(d, n0, n_s, low, high, adapt.materials[2]);
    let starts = [b.clone(), vec![b[0].clone(), b[2].clone()], single_layer(d, n0, n_s)];
    Ok(first_not_worse(starts.iter().map(|s| nelder_mead(s, &score)), d, adapt, n0, n_s))
}

/// Exactly 2000 objective evaluations of a Nelder-Mead search over the
/// layers' thicknesses, bounded to 5..600 nm, from a simplex of the start and
/// each thickness raised by 15%; reflection 1, expansion 2, contraction 0.5,
/// shrink 0.5; ties broken by vertex order. Deterministic: the same start and
/// the same arithmetic give the same layers. The result is rounded to 0.01 nm.
fn nelder_mead(start: &[FilmLayer], score: &dyn Fn(&[FilmLayer]) -> f64) -> Vec<FilmLayer> {
    const BUDGET: usize = 2000;
    let n = start.len();
    let clamp = |x: f64| x.clamp(5.0, 600.0);
    let build = |x: &[f64]| -> Vec<FilmLayer> {
        start.iter().zip(x).map(|(l, &t)| FilmLayer { thickness_nm: clamp(t), ..l.clone() }).collect()
    };
    let evals = std::cell::Cell::new(0usize);
    let f = |x: &[f64]| -> f64 {
        evals.set(evals.get() + 1);
        score(&build(x))
    };
    let x0: Vec<f64> = start.iter().map(|l| clamp(l.thickness_nm)).collect();
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    let v0 = f(&x0);
    simplex.push((x0.clone(), v0));
    for i in 0..n {
        let mut x = x0.clone();
        x[i] = clamp(x[i] * 1.15);
        let v = f(&x);
        simplex.push((x, v));
    }
    while evals.get() < BUDGET {
        // Stable sort: ties keep vertex order.
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let worst = simplex[n].clone();
        let centroid: Vec<f64> = (0..n).map(|j| simplex[..n].iter().map(|(x, _)| x[j]).sum::<f64>() / n as f64).collect();
        let along = |t: f64| -> Vec<f64> { (0..n).map(|j| clamp(centroid[j] + t * (worst.0[j] - centroid[j]))).collect() };
        let xr = along(-1.0);
        let fr = f(&xr);
        if fr < simplex[0].1 && evals.get() < BUDGET {
            let xe = along(-2.0);
            let fe = f(&xe);
            simplex[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (xr, fr);
        } else if evals.get() < BUDGET {
            let xc = if fr < worst.1 { along(-0.5) } else { along(0.5) };
            let fc = f(&xc);
            if fc < worst.1.min(fr) {
                simplex[n] = (xc, fc);
            } else {
                for k in 1..=n {
                    if evals.get() >= BUDGET {
                        break;
                    }
                    let x: Vec<f64> = (0..n).map(|j| clamp(simplex[0].0[j] + 0.5 * (simplex[k].0[j] - simplex[0].0[j]))).collect();
                    let v = f(&x);
                    simplex[k] = (x, v);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    let best: Vec<f64> = simplex[0].0.iter().map(|t| (t * 100.0).round() / 100.0).collect();
    build(&best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thinfilm::{film_coating, CoatingDesign};

    fn design(kind: DesignKind, n_s: f64) -> CoatingDesign {
        CoatingDesign {
            kind,
            lambda_nm: 550.0,
            aoi_deg: 0.0,
            periods: 1,
            n_high: 2.35,
            n_low: 1.38,
            adapt: Some(DesignAdapt::visible(n_s, 0.0)),
        }
    }

    fn bare(n_s: f64) -> f64 {
        ((n_s - 1.0) / (n_s + 1.0)).powi(2)
    }

    fn band(c: &Coating, n_s: f64) -> f64 {
        band_reflectance(c, 1.0, n_s, [0.4, 0.7], [0.0, 30.0], DesignWeight::Uniform)
    }

    /// C25: on every substrate the matched quarter-half-quarter is no worse
    /// than the adapted single layer over 400 to 700 nm and 0 to 30 degrees,
    /// and the single layer beats bare glass; the legacy preset on N-SF66
    /// reflects at least 8%, the regression the spec was written for. On
    /// crown the three layers are kept; on denser glass the matched quarter
    /// wants an index near 1.8 that no catalogue film has, and the rule falls
    /// back to fewer layers (1.72: one, 1.92: two).
    #[test]
    fn c25_the_adapted_ar_beats_the_single_layer_beats_bare_glass() {
        for n_s in [1.46, 1.52, 1.65, 1.72, 1.92, 2.00] {
            let mut bb = design(DesignKind::Bbar, n_s);
            bb.adapt.as_mut().unwrap().aoi_range_deg = [0.0, 30.0];
            let mut single = design(DesignKind::SingleLayerAr, n_s);
            single.adapt.as_mut().unwrap().aoi_range_deg = [0.0, 30.0];
            let bc = film_coating(bb, 1.0, n_s);
            let b = band(&bc, n_s);
            let s = band(&film_coating(single, 1.0, n_s), n_s);
            println!("C25 n_s {n_s}: Bbar {b:.5} ({} layers), single {s:.5}, bare {:.5}", bc.layers.len(), bare(n_s));
            assert!(b <= s + 1e-3, "n_s {n_s}: Bbar {b:.4} vs single {s:.4}");
            assert!(s <= bare(n_s), "n_s {n_s}: single {s:.4} vs bare {:.4}", bare(n_s));
            if n_s <= 1.52 {
                assert_eq!(bc.layers.len(), 3, "n_s {n_s}");
            }
        }
        // The eyepiece study's coating: the legacy preset at 650 nm on N-SF66
        // (n_d 1.9229). Per face it reflects 8.1% of 400 to 700 nm, against
        // 1.9% for one MgF2 layer at the same wavelength: about 12% of the
        // light through the two faces.
        let legacy = film_coating(
            CoatingDesign { kind: DesignKind::BroadbandArLegacy, lambda_nm: 650.0, aoi_deg: 0.0, periods: 1, n_high: 2.1, n_low: 1.38, adapt: None },
            1.0,
            1.9229,
        );
        let r = band_reflectance(&legacy, 1.0, 1.9229, [0.4, 0.7], [0.0, 0.0], DesignWeight::Uniform);
        println!("C25 legacy on N-SF66 at 650 nm: {r:.5}");
        assert!(r >= 0.08, "the legacy AR on N-SF66 reflects {r:.4}");
    }

    /// C26: a V-coat of silica over titania on N-BK7 nulls the design
    /// wavelength; a pair that cannot is refused by name.
    #[test]
    fn c26_a_v_coat_is_an_exact_null() {
        let d = design(DesignKind::VCoat, 1.5168);
        let layers = synthesize(&d, 1.0, 1.5168).unwrap();
        assert_eq!(layers.len(), 2);
        let r = coating_rt_at(&on(layers), 1.0, 1.5168, 0.55, 1.0).unwrap();
        assert!(r.r() <= 1e-10, "R at 550 nm: {:e}", r.r());
        // Two low-index films cannot make a V on glass.
        let mut bad = design(DesignKind::VCoat, 1.5168);
        bad.adapt.as_mut().unwrap().materials = [Some(FilmKey::MagnesiumFluoride), Some(FilmKey::Silica), None];
        let err = synthesize(&bad, 1.0, 1.5168).unwrap_err();
        assert!(err.contains("cannot null"), "{err}");
    }

    /// C27: a solar AR on low-iron glass, 0.4 to 2.5 um and 0 to 40 degrees,
    /// weighted by the direct sun: never worse than the single layer, and at
    /// most half of bare glass (spec Q6: measured 0.475 with dense films; MgF2
    /// at 1.38 outside is the floor, which porous silica would lower).
    #[test]
    fn c27_a_band_ar_is_never_worse_than_the_single_layer() {
        let n_s = 1.5231;
        let adapt = DesignAdapt {
            band_um: [0.4, 2.5],
            aoi_range_deg: [0.0, 40.0],
            substrate_n: n_s,
            materials: [None; 3],
            design_point: None,
            weight: DesignWeight::SunDirect,
        };
        let d = CoatingDesign { kind: DesignKind::BandAr, lambda_nm: 700.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38, adapt: Some(adapt) };
        let single = CoatingDesign { kind: DesignKind::SingleLayerAr, ..d };
        let score = |c: &Coating| band_reflectance(c, 1.0, n_s, adapt.band_um, adapt.aoi_range_deg, adapt.weight);
        let b = score(&film_coating(d, 1.0, n_s));
        let s = score(&film_coating(single, 1.0, n_s));
        let bare_c = on(Vec::new());
        let g = score(&bare_c);
        println!("C27: band AR {b:.5}, single {s:.5}, bare {g:.5}, ratio to bare {:.3}", b / g);
        assert!(b <= s + 1e-3, "band AR {b:.5} vs single {s:.5}");
        assert!(b <= 0.5 * g, "band AR {b:.5} vs bare {g:.5}");
    }

    /// Synthesis is deterministic: the same design gives the same layers.
    #[test]
    fn synthesis_is_deterministic() {
        let mut d = design(DesignKind::BandAr, 1.62);
        d.adapt.as_mut().unwrap().aoi_range_deg = [0.0, 20.0];
        let a = synthesize(&d, 1.0, 1.62).unwrap();
        let b = synthesize(&d, 1.0, 1.62).unwrap();
        assert_eq!(a, b);
        assert!(a.len() <= 8);
    }

    #[test]
    fn an_enhanced_metal_is_pairs_over_the_metal() {
        let mut d = design(DesignKind::EnhancedMetal { metal: MetalKind::Silver, pairs: 2 }, 1.52);
        d.lambda_nm = 520.0;
        let layers = synthesize(&d, 1.0, 1.52).unwrap();
        assert_eq!(layers.len(), 5);
        assert_eq!(layers[4].material.name, "Ag");
        let r = coating_rt_at(&on(layers), 1.0, 1.52, 0.52, 1.0).unwrap().r();
        let bare_silver = coating_rt_at(&on(vec![FilmLayer::new(FilmKey::Silver.material(), 150.0)]), 1.0, 1.52, 0.52, 1.0).unwrap().r();
        assert!(r > bare_silver, "{r} vs {bare_silver}");
    }

    /// The layer lists the CI runners must agree on to the bit (CF-08 exit
    /// criterion): recorded on Windows, checked on Linux.
    #[test]
    fn synthesized_layers_are_pinned() {
        let mut got = Vec::new();
        for (kind, n_s) in [
            (DesignKind::SingleLayerAr, 1.62),
            (DesignKind::VCoat, 1.5168),
            (DesignKind::Bbar, 1.52),
            (DesignKind::Bbar, 1.72),
            (DesignKind::Bbar, 1.92),
            (DesignKind::BandAr, 1.62),
            (DesignKind::EnhancedMetal { metal: MetalKind::Aluminium, pairs: 1 }, 1.52),
        ] {
            let mut d = design(kind, n_s);
            d.adapt.as_mut().unwrap().aoi_range_deg = [0.0, 20.0];
            let layers = synthesize(&d, 1.0, n_s).unwrap();
            got.push(layers.iter().map(|l| format!("{}:{:?}", l.material.name, l.thickness_nm)).collect::<Vec<_>>().join(" "));
        }
        let want = [
            "MgF2 (crystal, o):99.745687",
            "SiO2 (film):124.924205 TiO2 (film):14.176904",
            "MgF2 (crystal, o):99.745687 Ta2O5 (film):127.476403 Al2O3 (film):81.725326",
            "MgF2 (crystal, o):99.745687",
            "MgF2 (crystal, o):99.745687 HfO2 (film):72.288573",
            "MgF2 (crystal, o):93.48 Ta2O5 (film):117.11 Al2O3 (film):75.89",
            "Ta2O5 (film):63.738201 SiO2 (film):93.242338 Al:150.0",
        ];
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g, w);
        }
    }
}
