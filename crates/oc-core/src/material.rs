//! Materials and dispersion.
//!
//! Wavelengths are in **micrometres** everywhere in this crate.
//!
//! Two dispersion models:
//! - [`Material::AbbePoint`] uses a two-term Cauchy fit with a **closed form**
//!   from `(n_d, V_d)` (§3.1) — no solver, cheap enough to be a brush.
//! - [`Material::Catalog`] uses the full three-term Sellmeier equation and is
//!   treated as exact.

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

/// A catalog glass entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Glass {
    pub name: String,
    pub sell: Sellmeier,
    /// Cached refractive index at the d line.
    pub nd: f64,
    /// Cached Abbe number.
    pub vd: f64,
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
            .map(|(name, b, c)| {
                let sell = Sellmeier { b: *b, c: *c };
                Glass {
                    name: (*name).to_string(),
                    nd: sell.index(LAMBDA_D),
                    vd: abbe_number(&sell),
                    sell,
                }
            })
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
        glasses.push(Glass {
            name: "FusedSilica".to_string(),
            nd: fs.index(LAMBDA_D),
            vd: abbe_number(&fs),
            sell: fs,
        });

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
            // `nd`/`vd` are caches derived from `sell`, so dispersion identity is
            // decided by the Sellmeier coefficients alone.
            Some(id) if self.glasses[id.0 as usize].sell == glass.sell => Some(id),
            Some(_) => None,
            None => {
                let id = GlassId(self.glasses.len() as u32);
                self.glasses.push(glass.clone());
                Some(id)
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
            Material::Catalog(id) => catalog.get(*id).map(|g| g.sell.index(lambda_um)),
            Material::AbbePoint { nd, vd } => Some(Cauchy::from_abbe(*nd, *vd).index(lambda_um)),
            Material::Grin(f) => Some(f.base_index),
            Material::Mirror { .. } | Material::Absorber => None,
        }
    }

    /// `(n_d, V_d)` for placement on the Abbe map, if defined.
    pub fn abbe(&self, catalog: &GlassCatalog) -> Option<(f64, f64)> {
        match self {
            Material::Catalog(id) => catalog.get(*id).map(|g| (g.nd, g.vd)),
            Material::AbbePoint { nd, vd } => Some((*nd, *vd)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glass(name: &str, b0: f64) -> Glass {
        let sell = Sellmeier {
            b: [b0, 0.2, 1.0],
            c: [0.006, 0.02, 103.0],
        };
        Glass {
            name: name.to_string(),
            nd: sell.index(LAMBDA_D),
            vd: abbe_number(&sell),
            sell,
        }
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
