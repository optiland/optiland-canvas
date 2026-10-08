//! Materials and dispersion.
//!
//! Wavelengths are in **micrometres** everywhere in this crate.
//!
//! Two dispersion models:
//! - [`Material::AbbePoint`] uses a two-term Cauchy fit with a **closed form**
//!   from `(n_d, V_d)` (§3.1) — no solver, cheap enough to be a brush.
//! - [`Material::Catalog`] uses the glass's published data: the three-term
//!   Sellmeier equation, or, for a glass published in another form, that
//!   form ([`DispersionFormula`]), which the f64 reference evaluates exactly.

use serde::{Deserialize, Serialize};

/// d/F/C reference lines (Fraunhofer), in micrometres.
pub const LAMBDA_D: f64 = 0.587_56; // He d
pub const LAMBDA_F: f64 = 0.486_13; // H F
pub const LAMBDA_C: f64 = 0.656_27; // H C

/// Identifier into a [`GlassCatalog`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GlassId(pub u32);

/// Bulk material of an element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Material {
    /// A real catalog glass (full Sellmeier).
    Catalog(GlassId),
    /// An arbitrary point on the glass map; dispersion via closed-form Cauchy.
    AbbePoint { nd: f64, vd: f64 },
    /// A reflective body. `reflectivity` in `[0, 1]`.
    Mirror { reflectivity: f64 },
    /// A perfect absorber (beam block).
    Absorber,
    /// Gradient-index field (§8). The eikonal integrator lands in M7; for now a
    /// field carries a constant fallback index so the type is usable.
    Grin(GrinField),
}

/// Placeholder GRIN field (M7). Stores an axis-aligned world-space box and a
/// constant fallback index; the painted texture + eikonal march come later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrinField {
    pub min: crate::vec::Vec2,
    pub max: crate::vec::Vec2,
    pub base_index: f64,
}

/// Two-term Cauchy coefficients: `n(λ) = a + b / λ²`.
#[derive(Debug, Clone, Copy)]
pub struct Cauchy {
    pub a: f64,
    pub b: f64,
}

impl Cauchy {
    /// Closed-form fit from the Abbe pair `(n_d, V_d)` (§3.1).
    ///
    /// ```text
    /// B = (n_d − 1) / ( V_d · (1/λ_F² − 1/λ_C²) )
    /// A = n_d − B/λ_d²
    /// ```
    pub fn from_abbe(nd: f64, vd: f64) -> Self {
        let inv_f2 = 1.0 / (LAMBDA_F * LAMBDA_F);
        let inv_c2 = 1.0 / (LAMBDA_C * LAMBDA_C);
        let inv_d2 = 1.0 / (LAMBDA_D * LAMBDA_D);
        let b = (nd - 1.0) / (vd * (inv_f2 - inv_c2));
        let a = nd - b * inv_d2;
        Self { a, b }
    }

    #[inline]
    pub fn index(&self, lambda_um: f64) -> f64 {
        self.a + self.b / (lambda_um * lambda_um)
    }
}

/// Three-term Sellmeier coefficients. `c` terms are in µm².
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sellmeier {
    pub b: [f64; 3],
    pub c: [f64; 3],
}

impl Sellmeier {
    /// `n(λ) = sqrt(1 + Σ bᵢ λ² / (λ² − cᵢ))`, λ in µm.
    pub fn index(&self, lambda_um: f64) -> f64 {
        let l2 = lambda_um * lambda_um;
        let mut n2 = 1.0;
        for i in 0..3 {
            n2 += self.b[i] * l2 / (l2 - self.c[i]);
        }
        n2.sqrt()
    }
}

/// A dispersion formula as the refractiveindex.info database publishes it,
/// which is the form the catalogs behind Optiland's glasses come in. Numbered
/// as the database numbers them; wavelengths in µm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispersionFormula {
    /// Formula 1: `n² − 1 = c₀ + Σ cᵢ λ² / (λ² − cᵢ₊₁²)`.
    Sellmeier { coefficients: Vec<f64> },
    /// Formula 2: `n² − 1 = c₀ + Σ cᵢ λ² / (λ² − cᵢ₊₁)`.
    Sellmeier2 { coefficients: Vec<f64> },
    /// Formula 3: `n² = c₀ + Σ cᵢ λ^cᵢ₊₁`.
    Polynomial { coefficients: Vec<f64> },
    /// No dispersion: `n` at every wavelength (Optiland's `IdealMaterial`).
    Constant { n: f64 },
    /// Formula 5, the generalized Cauchy: `n = c₀ + Σ cᵢ λ^cᵢ₊₁`. What thin
    /// films are often published in (HfO₂, Al-Kuhaili).
    Cauchy { coefficients: Vec<f64> },
}

impl DispersionFormula {
    /// The refractive index at `lambda_um`, summed in the order the database's
    /// definition (and Optiland) sums it.
    pub fn index(&self, lambda_um: f64) -> f64 {
        let l2 = lambda_um * lambda_um;
        match self {
            Self::Sellmeier { coefficients: c } => {
                let mut n2 = 1.0 + c[0];
                for k in (1..c.len()).step_by(2) {
                    n2 += c[k] * l2 / (l2 - c[k + 1] * c[k + 1]);
                }
                n2.sqrt()
            }
            Self::Sellmeier2 { coefficients: c } => {
                let mut n2 = 1.0 + c[0];
                for k in (1..c.len()).step_by(2) {
                    n2 += c[k] * l2 / (l2 - c[k + 1]);
                }
                n2.sqrt()
            }
            Self::Polynomial { coefficients: c } => {
                let mut n2 = c[0];
                for k in (1..c.len()).step_by(2) {
                    n2 += c[k] * lambda_um.powf(c[k + 1]);
                }
                n2.sqrt()
            }
            Self::Constant { n } => *n,
            Self::Cauchy { coefficients: c } => {
                let mut n = c[0];
                for k in (1..c.len()).step_by(2) {
                    n += c[k] * lambda_um.powf(c[k + 1]);
                }
                n
            }
        }
    }

    /// Why the formula cannot be evaluated, if it cannot: a coefficient that
    /// is not finite, or a count that is not a constant plus whole pairs.
    pub fn check(&self) -> Result<(), String> {
        let c = match self {
            Self::Constant { n } => {
                return if n.is_finite() && *n > 0.0 {
                    Ok(())
                } else {
                    Err(format!("a constant index of {n}"))
                };
            }
            Self::Sellmeier { coefficients }
            | Self::Sellmeier2 { coefficients }
            | Self::Polynomial { coefficients }
            | Self::Cauchy { coefficients } => coefficients,
        };
        if c.iter().any(|v| !v.is_finite()) {
            return Err("a coefficient that is not finite".into());
        }
        if c.len() % 2 != 1 {
            return Err(format!(
                "{} coefficients, where a constant and whole pairs make an odd count",
                c.len()
            ));
        }
        Ok(())
    }

    /// The same curve as a three-term [`Sellmeier`], when it is one exactly.
    ///
    /// Formulas 1 and 2 are Sellmeier sums; one with at most three terms fits,
    /// and so does a nonzero constant `c₀` beside at most two, as a term with
    /// its pole at zero (`b λ² / (λ² − 0) = b`). A constant index is the
    /// constant term alone. A polynomial, or a longer sum, is not a Sellmeier
    /// curve, and a Sellmeier stand-in for it can only be fitted.
    pub fn as_sellmeier(&self) -> Option<Sellmeier> {
        let (c, squared) = match self {
            Self::Constant { n } => {
                return Some(Sellmeier {
                    b: [n * n - 1.0, 0.0, 0.0],
                    c: [0.0; 3],
                })
            }
            Self::Sellmeier { coefficients } => (coefficients, true),
            Self::Sellmeier2 { coefficients } => (coefficients, false),
            Self::Polynomial { .. } | Self::Cauchy { .. } => return None,
        };
        let mut terms: Vec<(f64, f64)> = c[1..]
            .chunks(2)
            .map(|p| (p[0], if squared { p[1] * p[1] } else { p[1] }))
            .collect();
        if c[0] != 0.0 {
            terms.push((c[0], 0.0));
        }
        if terms.len() > 3 {
            return None;
        }
        let mut s = Sellmeier {
            b: [0.0; 3],
            c: [0.0; 3],
        };
        for (i, (b, pole)) in terms.into_iter().enumerate() {
            s.b[i] = b;
            s.c[i] = pole;
        }
        Some(s)
    }
}

/// A catalog glass entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Glass {
    pub name: String,
    /// The three-term Sellmeier curve. The glass's own data when it is
    /// published in that form; otherwise a stand-in fitted to [`Glass::formula`],
    /// for the f32 preview, whose shader evaluates this form and no other.
    pub sell: Sellmeier,
    /// Cached refractive index at the d line.
    pub nd: f64,
    /// Cached Abbe number; 0 for a glass with no dispersion, whose Abbe
    /// number is infinite (the convention Zemax's model glass uses).
    pub vd: f64,
    /// The catalog the name is from ("schott", "hikari"): two makers sell
    /// different glasses under one name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// The published formula, when it is not [`Glass::sell`] itself. The f64
    /// reference evaluates this, so its indices are the catalog's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formula: Option<DispersionFormula>,
}

impl Glass {
    /// A glass published as a three-term Sellmeier curve.
    pub fn sellmeier(name: impl Into<String>, sell: Sellmeier) -> Self {
        let mut g = Self {
            name: name.into(),
            sell,
            nd: 0.0,
            vd: 0.0,
            reference: None,
            formula: None,
        };
        g.recache();
        g
    }

    /// A glass published in another form, with the Sellmeier curve the f32
    /// preview uses in its place: the formula's own when it is one
    /// ([`DispersionFormula::as_sellmeier`]), else `stand_in`. `None` when
    /// neither is given.
    pub fn with_formula(
        name: impl Into<String>,
        formula: DispersionFormula,
        stand_in: Option<Sellmeier>,
    ) -> Option<Self> {
        let sell = formula.as_sellmeier().or(stand_in)?;
        let mut g = Self {
            name: name.into(),
            sell,
            nd: 0.0,
            vd: 0.0,
            reference: None,
            formula: Some(formula),
        };
        g.recache();
        Some(g)
    }

    /// The refractive index at `lambda_um`: the published formula when there
    /// is one, else the Sellmeier curve.
    pub fn index(&self, lambda_um: f64) -> f64 {
        match &self.formula {
            Some(f) => f.index(lambda_um),
            None => self.sell.index(lambda_um),
        }
    }

    /// Does the index stay the same at every wavelength?
    pub fn is_dispersionless(&self) -> bool {
        matches!(self.formula, Some(DispersionFormula::Constant { .. }))
    }

    fn recache(&mut self) {
        self.nd = self.index(LAMBDA_D);
        let spread = self.index(LAMBDA_F) - self.index(LAMBDA_C);
        self.vd = if spread == 0.0 {
            0.0
        } else {
            (self.nd - 1.0) / spread
        };
    }
}

/// A glass catalog, indexed by [`GlassId`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlassCatalog {
    glasses: Vec<Glass>,
}

impl GlassCatalog {
    /// A small seed of public-domain Schott / Malitson data. Extend from
    /// refractiveindex.info + Schott's freely published catalog (§10.6).
    pub fn builtin() -> Self {
        // (name, B1, B2, B3, C1, C2, C3) — Sellmeier, C in µm².
        let raw: &[(&str, [f64; 3], [f64; 3])] = &[
            (
                "N-BK7",
                [1.039_612_12, 0.231_792_344, 1.010_469_45],
                [0.006_000_698_67, 0.020_017_914_4, 103.560_653],
            ),
            (
                "F2",
                [1.345_333_59, 0.209_073_176, 0.937_357_162],
                [0.009_977_438_71, 0.047_045_076_7, 111.886_764],
            ),
            (
                "N-SF11",
                [1.737_596_95, 0.313_747_346, 1.898_781_01],
                [0.013_188_707, 0.062_306_814_2, 155.236_29],
            ),
            (
                "N-LAK22",
                [1.142_291_76, 0.535_138_441, 1.040_830_92],
                [0.005_856_732_87, 0.019_808_638_0, 75.845_027_2],
            ),
        ];

        let mut glasses: Vec<Glass> = raw
            .iter()
            .map(|(name, b, c)| Glass::sellmeier(*name, Sellmeier { b: *b, c: *c }))
            .collect();

        // Fused silica (Malitson): the C constants are given un-squared.
        let fs_c_raw = [0.068_404_3, 0.116_241_4, 9.896_161];
        let fs = Sellmeier {
            b: [0.696_166_3, 0.407_942_6, 0.897_479_4],
            c: [
                fs_c_raw[0] * fs_c_raw[0],
                fs_c_raw[1] * fs_c_raw[1],
                fs_c_raw[2] * fs_c_raw[2],
            ],
        };
        glasses.push(Glass::sellmeier("FusedSilica", fs));

        Self { glasses }
    }

    pub fn get(&self, id: GlassId) -> Option<&Glass> {
        self.glasses.get(id.0 as usize)
    }

    pub fn find(&self, name: &str) -> Option<GlassId> {
        self.glasses
            .iter()
            .position(|g| g.name.eq_ignore_ascii_case(name))
            .map(|i| GlassId(i as u32))
    }

    /// Merge `glass` into this catalog, returning the id it resolves to.
    ///
    /// Glasses are identified by **name**, case-insensitively, to match
    /// [`GlassCatalog::find`]. A name already present with identical dispersion
    /// reuses its existing id; a new name is appended.
    ///
    /// Returns `None` when the name is already present with *different*
    /// dispersion data. Two different glasses sharing a name cannot be merged
    /// without silently changing the optics of one of them, so the caller must
    /// surface the conflict rather than pick a winner.
    ///
    /// This exists because [`Material::Catalog`] stores a positional [`GlassId`]:
    /// combining scenes that were authored against different catalogs *must*
    /// remap those ids, or `GlassId(2)` in one catalog silently becomes an
    /// unrelated glass in the other.
    pub fn intern(&mut self, glass: &Glass) -> Option<GlassId> {
        match self.find(&glass.name) {
            // `nd`/`vd` are caches derived from the dispersion data, so
            // identity is decided by the Sellmeier curve and the formula.
            Some(id) if same_dispersion(&self.glasses[id.0 as usize], glass) => Some(id),
            Some(_) => None,
            None => {
                let id = GlassId(self.glasses.len() as u32);
                self.glasses.push(glass.clone());
                Some(id)
            }
        }
    }

    /// Put `glass` in this catalog under its name, replacing a glass of that
    /// name with different data in place, and return its id.
    ///
    /// For a catalog no scene object refers to yet, such as a scene being
    /// built from a prescription whose glass data is the authority: the
    /// built-in "F2" gives way to the catalog's own F2 rather than the scene
    /// refusing to hold both. Replacing a glass a body already uses would
    /// change that body's optics, which is what [`GlassCatalog::intern`]
    /// refuses to do.
    pub fn put(&mut self, glass: &Glass) -> GlassId {
        match self.find(&glass.name) {
            Some(id) => {
                self.glasses[id.0 as usize] = glass.clone();
                id
            }
            None => {
                let id = GlassId(self.glasses.len() as u32);
                self.glasses.push(glass.clone());
                id
            }
        }
    }

    pub fn len(&self) -> usize {
        self.glasses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.glasses.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (GlassId, &Glass)> {
        self.glasses
            .iter()
            .enumerate()
            .map(|(i, g)| (GlassId(i as u32), g))
    }
}

fn same_dispersion(a: &Glass, b: &Glass) -> bool {
    a.sell == b.sell && a.formula == b.formula
}

/// Abbe number `V_d = (n_d − 1) / (n_F − n_C)` for a dispersion model.
pub fn abbe_number(sell: &Sellmeier) -> f64 {
    (sell.index(LAMBDA_D) - 1.0) / (sell.index(LAMBDA_F) - sell.index(LAMBDA_C))
}

/// A reference point on the n_d–V_d glass map (§5.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbbeGlass {
    pub name: String,
    pub nd: f64,
    pub vd: f64,
}

/// A curated scatter of real optical glasses `(name, n_d, V_d)` for the Abbe
/// picker. Published catalog values; the picker floats freely between them
/// (as `AbbePoint` materials) and can snap to them. "The physics is in the
/// palette" — the convex hull of this set bounds obtainable glass (§5.4).
pub fn reference_glasses() -> Vec<AbbeGlass> {
    const DATA: &[(&str, f64, f64)] = &[
        ("N-FK51A", 1.4866, 84.47),
        ("N-PK52A", 1.4970, 81.61),
        ("N-FK5", 1.4875, 70.41),
        ("FusedSilica", 1.4585, 67.82),
        ("N-BK7", 1.5168, 64.17),
        ("N-K5", 1.5224, 59.48),
        ("N-BAK4", 1.5688, 55.98),
        ("N-BALF4", 1.5796, 53.87),
        ("N-SK11", 1.5638, 60.80),
        ("N-SSK8", 1.6177, 49.83),
        ("N-BAF10", 1.6700, 47.11),
        ("F2", 1.6200, 36.37),
        ("N-SF2", 1.6477, 33.85),
        ("N-SF5", 1.6727, 32.21),
        ("N-LAK22", 1.6516, 55.89),
        ("N-LAK8", 1.7130, 53.83),
        ("N-SF11", 1.7847, 25.76),
        ("N-SF6", 1.8052, 25.36),
        ("N-LASF9", 1.8503, 32.17),
        ("N-SF57", 1.8467, 23.83),
        ("N-BK10", 1.4978, 66.95),
        ("N-KZFS4", 1.6134, 44.49),
    ];
    DATA.iter()
        .map(|(name, nd, vd)| AbbeGlass {
            name: (*name).to_string(),
            nd: *nd,
            vd: *vd,
        })
        .collect()
}

/// Convex hull (Andrew's monotone chain) of `pts`, returned CCW. Points are
/// `(x, y)`; for the Abbe map pass `(V_d, n_d)`.
pub fn convex_hull(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut p: Vec<(f64, f64)> = pts.to_vec();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower = Vec::new();
    for &pt in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], pt) <= 0.0 {
            lower.pop();
        }
        lower.push(pt);
    }
    let mut upper = Vec::new();
    for &pt in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], pt) <= 0.0 {
            upper.pop();
        }
        upper.push(pt);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Is point `(x, y)` inside (or on) the CCW convex polygon `hull`?
pub fn point_in_hull(x: f64, y: f64, hull: &[(f64, f64)]) -> bool {
    if hull.len() < 3 {
        return false;
    }
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let cross = (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0);
        if cross < -1e-9 {
            return false; // strictly right of a CCW edge ⇒ outside
        }
    }
    true
}

impl Material {
    /// Refractive index at `lambda_um` for transmissive materials.
    ///
    /// Returns `None` for materials that are not simple dielectrics
    /// (mirror/absorber), which the tracer handles by surface behaviour.
    pub fn index_at(&self, lambda_um: f64, catalog: &GlassCatalog) -> Option<f64> {
        match self {
            Material::Catalog(id) => catalog.get(*id).map(|g| g.index(lambda_um)),
            Material::AbbePoint { nd, vd } => Some(Cauchy::from_abbe(*nd, *vd).index(lambda_um)),
            Material::Grin(f) => Some(f.base_index),
            Material::Mirror { .. } | Material::Absorber => None,
        }
    }

    /// `(n_d, V_d)` for placement on the Abbe map, if defined. A glass with no
    /// dispersion has no finite Abbe number and no place on the map.
    pub fn abbe(&self, catalog: &GlassCatalog) -> Option<(f64, f64)> {
        match self {
            Material::Catalog(id) => catalog
                .get(*id)
                .filter(|g| !g.is_dispersionless())
                .map(|g| (g.nd, g.vd)),
            Material::AbbePoint { nd, vd } => Some((*nd, *vd)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glass(name: &str, b0: f64) -> Glass {
        Glass::sellmeier(
            name,
            Sellmeier {
                b: [b0, 0.2, 1.0],
                c: [0.006, 0.02, 103.0],
            },
        )
    }

    /// Schott N-BK7 as refractiveindex.info publishes it (formula 2), and the
    /// index Schott's datasheet gives at the d line.
    fn nbk7_formula2() -> DispersionFormula {
        DispersionFormula::Sellmeier2 {
            coefficients: vec![
                0.0,
                1.039_612_12,
                0.006_000_698_67,
                0.231_792_344,
                0.020_017_914_4,
                1.010_469_45,
                103.560_653,
            ],
        }
    }

    #[test]
    fn a_three_term_sellmeier_formula_is_the_sellmeier_curve() {
        let f = nbk7_formula2();
        let s = f.as_sellmeier().expect("three terms fit");
        for lambda in [0.36, 0.48613, 0.58756, 0.65627, 1.064, 2.3] {
            assert_eq!(f.index(lambda), s.index(lambda), "at {lambda} µm");
        }
        assert!((f.index(LAMBDA_D) - 1.516_8).abs() < 1e-4);
        // Formula 1 squares its poles; the same glass written that way agrees.
        let DispersionFormula::Sellmeier2 { coefficients: c } = &f else {
            unreachable!()
        };
        let mut squared = c.clone();
        for k in (2..squared.len()).step_by(2) {
            squared[k] = squared[k].sqrt();
        }
        let f1 = DispersionFormula::Sellmeier {
            coefficients: squared,
        };
        assert!((f1.index(0.5) - f.index(0.5)).abs() < 1e-15);
    }

    #[test]
    fn a_constant_term_rides_as_a_pole_at_zero() {
        // n² − 1 = 0.3 + 1.2 λ²/(λ² − 0.01): two terms and a constant.
        let f = DispersionFormula::Sellmeier2 {
            coefficients: vec![0.3, 1.2, 0.01, 0.0, 0.0],
        };
        let s = f.as_sellmeier().unwrap();
        for lambda in [0.4, 0.6, 1.5] {
            assert!((f.index(lambda) - s.index(lambda)).abs() < 1e-15);
        }
        // A fourth term does not fit, and neither does a polynomial.
        let four = DispersionFormula::Sellmeier2 {
            coefficients: vec![0.0, 1.0, 0.01, 0.2, 0.02, 1.0, 100.0, 0.5, 200.0],
        };
        assert!(four.as_sellmeier().is_none());
        let poly = DispersionFormula::Polynomial {
            coefficients: vec![2.3, -0.01, 2.0, 0.01, -2.0],
        };
        assert!(poly.as_sellmeier().is_none());
    }

    #[test]
    fn a_polynomial_glass_is_evaluated_as_published() {
        // n² = c0 + c1 λ^e1 + c2 λ^e2, at a point worked by hand.
        let f = DispersionFormula::Polynomial {
            coefficients: vec![2.5, -0.01, 2.0, 0.012, -2.0],
        };
        let lambda: f64 = 0.5;
        let by_hand = (2.5 - 0.01 * 0.25 + 0.012 * 4.0_f64).sqrt();
        assert_eq!(f.index(lambda), by_hand);
        // The glass uses the formula, and the Sellmeier only stands in for it.
        let stand_in = Sellmeier {
            b: [1.5, 0.0, 0.0],
            c: [0.0; 3],
        };
        let g = Glass::with_formula("POLY", f.clone(), Some(stand_in)).unwrap();
        assert_eq!(g.index(lambda), by_hand);
        assert_eq!(g.sell, stand_in);
        assert!(
            Glass::with_formula("POLY", f, None).is_none(),
            "no stand-in, no glass"
        );
    }

    #[test]
    fn a_constant_glass_has_one_index_and_no_place_on_the_abbe_map() {
        let g = Glass::with_formula("n = 4.002", DispersionFormula::Constant { n: 4.002 }, None)
            .unwrap();
        for lambda in [0.25, 0.5876, 4.2, 10.6] {
            assert_eq!(g.index(lambda), 4.002);
            // The preview's Sellmeier stand-in is exact too.
            assert!((g.sell.index(lambda) - 4.002).abs() < 1e-15);
        }
        assert_eq!((g.nd, g.vd), (4.002, 0.0));
        let mut cat = GlassCatalog::builtin();
        let id = cat.put(&g);
        assert_eq!(Material::Catalog(id).abbe(&cat), None);
        assert_eq!(Material::Catalog(id).index_at(10.6, &cat), Some(4.002));
    }

    #[test]
    fn put_replaces_a_glass_of_the_same_name_in_place() {
        let mut cat = GlassCatalog::builtin();
        let before = cat.len();
        let f2 = cat.find("F2").unwrap();
        let other = glass("F2", 1.3);
        assert_eq!(cat.intern(&other), None, "intern refuses the clash");
        assert_eq!(cat.put(&other), f2, "put keeps the id");
        assert_eq!(cat.get(f2).unwrap().sell, other.sell);
        assert_eq!(cat.len(), before);
    }

    #[test]
    fn a_formula_survives_the_document_and_old_glasses_read_as_before() {
        let g = Glass::with_formula("X", nbk7_formula2(), None).unwrap();
        let mut g = g;
        g.reference = Some("schott".into());
        let json = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<Glass>(&json).unwrap(), g);
        // A glass written before the two fields existed has neither.
        let old = r#"{"name":"OLD","sell":{"b":[1.0,0.2,1.0],"c":[0.006,0.02,103.0]},"nd":1.5,"vd":60.0}"#;
        let read: Glass = serde_json::from_str(old).unwrap();
        assert!(read.formula.is_none() && read.reference.is_none());
        assert!(!serde_json::to_string(&read).unwrap().contains("formula"));
    }

    #[test]
    fn intern_reuses_an_identical_named_glass() {
        let mut cat = GlassCatalog::builtin();
        let before = cat.len();
        let nbk7 = cat.get(cat.find("N-BK7").unwrap()).unwrap().clone();
        // Re-interning the same glass must not grow the catalog or move the id.
        assert_eq!(cat.intern(&nbk7), cat.find("N-BK7"));
        assert_eq!(cat.len(), before);
    }

    #[test]
    fn intern_matches_find_case_insensitively() {
        let mut cat = GlassCatalog::builtin();
        let before = cat.len();
        let mut lower = cat.get(cat.find("N-BK7").unwrap()).unwrap().clone();
        lower.name = "n-bk7".to_string();
        assert_eq!(cat.intern(&lower), cat.find("N-BK7"));
        assert_eq!(cat.len(), before, "case variant must not duplicate");
    }

    #[test]
    fn intern_appends_a_new_glass() {
        let mut cat = GlassCatalog::builtin();
        let before = cat.len();
        let id = cat.intern(&glass("MADE-UP", 1.1)).expect("new name interns");
        assert_eq!(id.0 as usize, before);
        assert_eq!(cat.len(), before + 1);
        assert_eq!(cat.get(id).unwrap().name, "MADE-UP");
    }

    #[test]
    fn intern_rejects_a_name_collision_with_different_dispersion() {
        let mut cat = GlassCatalog::builtin();
        cat.intern(&glass("HOUSE-CROWN", 1.1)).unwrap();
        let before = cat.len();
        // Same name, different Sellmeier data: merging either way would silently
        // change one scene's optics, so this must be reported, not resolved.
        assert_eq!(cat.intern(&glass("HOUSE-CROWN", 1.3)), None);
        assert_eq!(cat.len(), before, "a rejected intern must not mutate");
    }

    #[test]
    fn cauchy_reproduces_abbe_pair() {
        // Pick a plausible crown: nd = 1.5168, vd = 64.17 (≈ N-BK7).
        let (nd, vd) = (1.5168, 64.17);
        let c = Cauchy::from_abbe(nd, vd);
        // At the d line it must return nd exactly.
        assert!((c.index(LAMBDA_D) - nd).abs() < 1e-12);
        // And reproduce the Abbe number by construction.
        let nf = c.index(LAMBDA_F);
        let nc = c.index(LAMBDA_C);
        let vd_back = (nd - 1.0) / (nf - nc);
        assert!((vd_back - vd).abs() < 1e-9, "vd {vd} vs {vd_back}");
    }

    #[test]
    fn nbk7_index_is_physical() {
        let cat = GlassCatalog::builtin();
        let id = cat.find("N-BK7").unwrap();
        let g = cat.get(id).unwrap();
        // N-BK7: nd ≈ 1.5168, vd ≈ 64.17 (well-known).
        assert!((g.nd - 1.5168).abs() < 1e-3, "nd = {}", g.nd);
        assert!((g.vd - 64.17).abs() < 0.3, "vd = {}", g.vd);
        // Dispersion is normal: shorter λ -> higher index.
        assert!(g.sell.index(LAMBDA_F) > g.sell.index(LAMBDA_C));
    }

    #[test]
    fn abbe_hull_bounds_real_glass() {
        let glasses = reference_glasses();
        let pts: Vec<(f64, f64)> = glasses.iter().map(|g| (g.vd, g.nd)).collect();
        let hull = convex_hull(&pts);
        assert!(hull.len() >= 3);
        // Every catalog glass lies inside (or on) its own hull.
        for g in &glasses {
            assert!(point_in_hull(g.vd, g.nd, &hull), "{} outside hull", g.name);
        }
        // A physically absurd point (very high index, very high Abbe) is out.
        assert!(
            !point_in_hull(95.0, 2.4, &hull),
            "unobtainable point ruled in"
        );
    }

    #[test]
    fn fused_silica_present() {
        let cat = GlassCatalog::builtin();
        let id = cat.find("fusedsilica").unwrap(); // case-insensitive
        let g = cat.get(id).unwrap();
        // Fused silica nd ≈ 1.4585.
        assert!((g.nd - 1.4585).abs() < 2e-3, "nd = {}", g.nd);
    }
}
