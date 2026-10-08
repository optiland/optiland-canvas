//! Thin-film multilayer coatings: the transfer-matrix method.
//!
//! This is the one place in the kernel where a coating is a **stack of layers**
//! rather than a number. Everything a coating does — the reflectance band of a
//! dichroic, the way its edge walks toward the blue as the angle of incidence
//! opens up, the way s and p separate at 45°, the residual ripple of an AR
//! coating — comes out of the same calculation, with no fitted parameter.
//!
//! ## What it computes
//!
//! For one wavelength and one angle of incidence, the characteristic matrix of
//! each layer is
//!
//! ```text
//! M_j = [ cos δ_j          i sin δ_j / η_j ]
//!       [ i η_j sin δ_j    cos δ_j         ]
//! ```
//!
//! with phase thickness `δ_j = 2π/λ · n_j · d_j · cos θ_j` and tilted optical
//! admittance `η_j = n_j cos θ_j` (s) or `n̄_j² / (n_j cos θ_j)` (p). The stack
//! matrix is their ordered product from the incident side inward; with
//! `[B, C]ᵀ = M · [1, η_sub]ᵀ` the amplitude coefficients are
//!
//! ```text
//! r = (η_0 B − C) / (η_0 B + C)        t = 2 η_0 / (η_0 B + C)
//! R = |r|²                             T = |t|² · Re(η_sub)/Re(η_0)
//! ```
//!
//! and `A = 1 − R − T` is what the layers absorbed. This is Macleod, *Thin-Film
//! Optical Filters*, ch. 2.6, and it is the formulation Optiland's
//! `optiland.thin_film` uses, down to the branch choice in [`snell_cos`] and the
//! conjugated index in the p admittance. That is deliberate: it makes Optiland a
//! numerical oracle for this module rather than a rough second opinion, and
//! `crates/oc-py/tests/test_thinfilm_parity.py` holds the two to 1e-12.
//!
//! ## Angles matter, which is the point
//!
//! A quarter-wave stack designed for 550 nm at normal incidence does not have
//! its edge at 550 nm when you tilt it to 45°: the phase thickness carries a
//! `cos θ_j`, so every layer looks thinner and the whole band walks blue, by
//! roughly 8% for a typical dielectric pair. The kernel used to evaluate a
//! dichroic's reflectance from a table in wavelength alone, so a plate sitting
//! at 45° — which is how every beam splitter and dichroic in the app is placed —
//! reported its normal-incidence numbers. That is the bug this module exists to
//! kill.
//!
//! ## Units
//!
//! Wavelengths in micrometres, matching the rest of the kernel. Layer
//! thicknesses in **nanometres**, because that is how coatings are specified and
//! a quarter wave of anything visible is a three-digit number of nm.

use crate::scene3::SpectralScalar;
use serde::{Deserialize, Serialize};

/// One-line mistakes the test suite must catch (acceptance case C31).
///
/// Each variant is a slip that leaves every number plausible: a sign, a branch,
/// a unit, a missing factor. The solver carries a hook at the line each one
/// would sit on, compiled into test builds only, and `tests::mutations` turns
/// them on one at a time and asserts that the named check fails. A mutation
/// nothing catches is a test the suite is missing.
///
/// The preview's outside grid reused inside (C22) is the same idea in
/// `oc-gpu`, which holds its own switch.
#[cfg(test)]
pub(crate) mod mutation {
    use std::cell::Cell;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Mutation {
        /// `r_p` left in Macleod's basis instead of the ray frame.
        RpSignDropped,
        /// The exit medium's cosine on the growing evanescent branch.
        GrowingBranch,
        /// `T = |t|²`, without the admittance ratio.
        NoAdmittanceRatio,
        /// A layer thickness read as micrometres.
        ThicknessAsMicrons,
        /// A stack met from its substrate walked in the written order.
        NoReverseWalk,
        /// A layer's absorption read across the wrong pair of interfaces.
        AbsorptionOffByOne,
    }

    thread_local! {
        static ACTIVE: Cell<Option<Mutation>> = const { Cell::new(None) };
    }

    /// Is `m` switched on for this thread?
    pub(crate) fn on(m: Mutation) -> bool {
        ACTIVE.with(|a| a.get() == Some(m))
    }

    /// Run `f` with `m` switched on, and off again afterwards.
    pub(crate) fn with<T>(m: Mutation, f: impl FnOnce() -> T) -> T {
        ACTIVE.with(|a| a.set(Some(m)));
        let out = f();
        ACTIVE.with(|a| a.set(None));
        out
    }
}

#[cfg(test)]
use mutation::Mutation;

/// A complex number. Enough arithmetic for the transfer matrix and no more; a
/// dependency for this would be four functions of value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cx {
    pub re: f64,
    pub im: f64,
}

impl Cx {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    /// `|z|²`, which is what the power coefficients actually want.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.norm_sqr().sqrt()
    }
    /// The principal square root: the branch with `re ≥ 0`, and `+i` on the
    /// negative real axis. This is what `numpy.sqrt` returns for a complex
    /// argument, and matching it is what keeps an evanescent layer on the same
    /// branch as the reference implementation.
    pub fn sqrt(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let m = self.abs();
        // Near the real axis, but off it, one of the two half-angle formulas
        // below subtracts two nearly equal numbers: for a film with k = 6e-4
        // that lost 6e-9 in T (the `dispersive_catalogue_stack` fixture, against
        // `tmm`, Optiland and the 40-digit oracle). There the small part is
        // taken from the large one instead, `im / (2 re)`. Only there, so
        // every result that was well conditioned keeps its bits.
        if self.im != 0.0 && self.im.abs() < 1e-2 * self.re.abs() {
            let big = ((m + self.re.abs()) * 0.5).sqrt();
            let small = self.im.abs() / (2.0 * big);
            return if self.re > 0.0 {
                Self::new(big, if self.im < 0.0 { -small } else { small })
            } else {
                Self::new(small, if self.im < 0.0 { -big } else { big })
            };
        }
        let re = ((m + self.re) * 0.5).max(0.0).sqrt();
        let im = ((m - self.re) * 0.5).max(0.0).sqrt();
        // `im == 0` on the positive real axis; on the negative real axis the
        // real part vanishes instead and the sign below picks `+i`.
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }
    pub fn cos(self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }
    pub fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    /// `e^{iθ}`: a unit phasor, which is what a retardance is.
    pub fn expi(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self::new(c, s)
    }
    /// The phase, in radians, on `(-π, π]`.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl std::ops::Neg for Cx {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl std::ops::Add for Cx {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Cx {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Cx {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl std::ops::Mul<f64> for Cx {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        Self::new(self.re * k, self.im * k)
    }
}

impl std::ops::Div for Cx {
    type Output = Self;
    /// Division, with a division by zero going to infinity rather than `NaN` so
    /// a degenerate admittance shows up as total reflection instead of poisoning
    /// the whole trace.
    fn div(self, o: Self) -> Self {
        let d = o.norm_sqr();
        if d == 0.0 {
            return Self::new(f64::INFINITY, f64::INFINITY);
        }
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}

/// The optical constants of one film material: `n` and the extinction
/// coefficient `k`, each constant or tabulated in wavelength.
///
/// The complex index is `n + ik` as stored. Macleod writes it `n − ik`, and the
/// two places that distinction is visible ([`snell_cos`] and the p admittance)
/// conjugate explicitly rather than storing a sign the authoring UI would have
/// to explain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilmMaterial {
    /// What to call it in the editor. Not an identifier: two layers of the same
    /// material carry the same name and are still two layers.
    #[serde(default)]
    pub name: String,
    pub n: SpectralScalar,
    #[serde(default = "zero_scalar")]
    pub k: SpectralScalar,
    /// A dispersion formula for `n`, which replaces the `n` curve over the
    /// formula's own range. `n` then holds the formula sampled over that
    /// range, so a reader that knows no formulas (a v2 reader, or the lossy
    /// v2 export) still has the index. Absent from every v2 document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formula: Option<Box<FilmFormula>>,
    /// Whether the data describe a thin film, a bulk sample, or a crystal. A
    /// film's index differs from the bulk's by a few percent, and by process.
    #[serde(default, skip_serializing_if = "FilmForm::is_bulk")]
    pub form: FilmForm,
    /// Where the data came from. Absent from a v2 document; migration to v3
    /// writes `Nominal`, which is what an inline constant was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Box<Provenance>>,
    /// The constant index `film_materials: Nominal` uses. When absent, a
    /// constant material's own constant, and a tabulated one's value at
    /// 0.55 µm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nominal: Option<NominalIndex>,
}

/// A dispersion formula and the wavelengths it is valid over.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilmFormula {
    pub formula: crate::material::DispersionFormula,
    pub range_um: [f64; 2],
}

/// What a material's data describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FilmForm {
    Film,
    #[default]
    Bulk,
    Crystal,
}

impl FilmForm {
    fn is_bulk(&self) -> bool {
        *self == FilmForm::Bulk
    }
}

/// How a material's data were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    /// A published database entry, with the paper it cites.
    Database,
    /// Measured by the user on their own part.
    Measured,
    /// A round constant: a starting point, not a measurement.
    Nominal,
    /// Read off a plotted curve.
    Digitized,
    /// A model fitted to data.
    Fitted,
}

/// Where a material's data came from (coating-fidelity spec CF-03).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub source_kind: SourceKind,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reference: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub license: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub retrieved: String,
    /// The wavelengths the data cover, µm; `None` for a constant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_um: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

impl Provenance {
    /// What an inline constant was before the catalogue existed.
    pub fn nominal(note: &str) -> Self {
        Self {
            source_kind: SourceKind::Nominal,
            reference: String::new(),
            url: String::new(),
            license: String::new(),
            retrieved: String::new(),
            range_um: None,
            note: note.to_string(),
        }
    }
}

/// A constant complex index.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NominalIndex {
    pub n: f64,
    #[serde(default)]
    pub k: f64,
}

fn zero_scalar() -> SpectralScalar {
    SpectralScalar::Constant(0.0)
}

impl FilmMaterial {
    /// A material from its two curves, with no formula, provenance or
    /// nominal value: what a v2 document holds.
    pub fn scalars(name: &str, n: SpectralScalar, k: SpectralScalar) -> Self {
        Self {
            name: name.to_string(),
            n,
            k,
            formula: None,
            form: FilmForm::Bulk,
            provenance: None,
            nominal: None,
        }
    }

    /// A transparent material of constant index.
    pub fn constant(name: &str, n: f64) -> Self {
        Self::scalars(name, SpectralScalar::Constant(n), zero_scalar())
    }

    /// An absorbing material of constant complex index.
    pub fn absorbing(name: &str, n: f64, k: f64) -> Self {
        Self::scalars(name, SpectralScalar::Constant(n), SpectralScalar::Constant(k))
    }

    /// The complex index at `lambda_um`, or `None` when a table has no value
    /// there under [`crate::scene3::OutOfRangePolicy::Error`]. A formula
    /// answers inside its range; outside it the sampled `n` curve answers,
    /// with its own out-of-range policy.
    pub fn index_at(&self, lambda_um: f64) -> Option<Cx> {
        let n = match &self.formula {
            Some(f) if lambda_um >= f.range_um[0] && lambda_um <= f.range_um[1] => f.formula.index(lambda_um),
            _ => self.n.sample(lambda_um)?,
        };
        Some(Cx::new(n, self.k.sample(lambda_um)?))
    }

    /// Does the material have data at `lambda_um`, rather than an
    /// out-of-range policy's answer?
    pub fn covers(&self, lambda_um: f64) -> bool {
        let n = match &self.formula {
            Some(f) => lambda_um >= f.range_um[0] && lambda_um <= f.range_um[1],
            None => self.n.covers(lambda_um),
        };
        n && self.k.covers(lambda_um)
    }

    /// The constant index `film_materials: Nominal` uses: the declared
    /// nominal, else the constant itself, else the value at 0.55 µm (held at
    /// the table's end if it does not reach there).
    pub fn nominal_index(&self) -> Cx {
        if let Some(v) = self.nominal {
            return Cx::new(v.n, v.k);
        }
        let at = |s: &SpectralScalar| match s {
            SpectralScalar::Constant(v) => *v,
            SpectralScalar::Tabulated { lambda_um, values, .. } => {
                let x = 0.55f64.clamp(lambda_um[0], lambda_um[lambda_um.len() - 1]);
                let i = lambda_um.partition_point(|&v| v <= x).saturating_sub(1).min(lambda_um.len() - 2);
                let f = (x - lambda_um[i]) / (lambda_um[i + 1] - lambda_um[i]);
                values[i] + f * (values[i + 1] - values[i])
            }
        };
        let n = match &self.formula {
            Some(f) => f.formula.index(0.55f64.clamp(f.range_um[0], f.range_um[1])),
            None => at(&self.n),
        };
        Cx::new(n, at(&self.k))
    }

    /// Vacuum, and near enough to air that the difference is below anything the
    /// canvas claims.
    pub fn air() -> Self {
        Self::constant("air", 1.0)
    }
}

/// One layer: a material and a physical thickness in nanometres.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilmLayer {
    pub material: FilmMaterial,
    pub thickness_nm: f64,
    /// The scene film catalogue entry this layer's material is (v3). The
    /// material is still written inline beside it, so a reader that knows no
    /// catalogue reads the layer whole; on load the catalogue entry replaces
    /// the inline copy, which keeps every layer of one entry in step.
    #[serde(default, rename = "ref", skip_serializing_if = "Option::is_none")]
    pub catalog_ref: Option<u32>,
    /// A slab: an element far thicker than any coherence length the canvas
    /// models, whose faces add in intensity, not in amplitude (CF-05). A
    /// coating with slabs is evaluated by the incoherent transfer-matrix
    /// method; its films between slabs stay coherent groups. Absent, and
    /// false, in every older document.
    #[serde(default, skip_serializing_if = "is_false")]
    pub incoherent: bool,
}

fn is_false(v: &bool) -> bool {
    !*v
}

impl FilmLayer {
    pub fn new(material: FilmMaterial, thickness_nm: f64) -> Self {
        Self {
            material,
            thickness_nm,
            catalog_ref: None,
            incoherent: false,
        }
    }

    /// A slab of `material`, `thickness_mm` thick: an incoherent element.
    pub fn slab(material: FilmMaterial, thickness_mm: f64) -> Self {
        Self {
            material,
            thickness_nm: thickness_mm * 1e6,
            catalog_ref: None,
            incoherent: true,
        }
    }

    /// A quarter wave of this material at `lambda_nm`, for light arriving at
    /// `aoi_deg` **in the medium of index `n_incident`**.
    ///
    /// Quarter-wave means a quarter wave of the *tilted* optical thickness,
    /// `n d cos θ_layer = λ/4`, which is the thickness that puts the layer's
    /// phase at π/2 for that geometry. Designing at the angle the part will be
    /// used at is what a coating house does, and it is why a dichroic sold for
    /// 45° has its edge where the datasheet says at 45° and nowhere near it at
    /// normal incidence.
    pub fn quarter_wave(
        material: FilmMaterial,
        lambda_nm: f64,
        n_incident: f64,
        aoi_deg: f64,
    ) -> Self {
        let lambda_um = lambda_nm / 1000.0;
        let n = material
            .index_at(lambda_um)
            .map(|c| c.re)
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0);
        let sin_i = n_incident * aoi_deg.to_radians().sin();
        let cos_l = (1.0 - (sin_i / n) * (sin_i / n)).max(1e-12).sqrt();
        Self {
            material,
            thickness_nm: lambda_nm / (4.0 * n * cos_l),
            catalog_ref: None,
            incoherent: false,
        }
    }
}

/// What the stack is deposited on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoatingBacking {
    /// The medium on the far side of the interface, whatever the scene says it
    /// is: an AR coating on a lens face is backed by that lens's glass.
    Medium,
    /// A plate of `substrate` whose second face returns to the incident medium.
    ///
    /// This is what a beam splitter or a dichroic filter physically is, and
    /// leaving it out costs about 4% of the transmission: the light that gets
    /// through the coating still has to get out of the glass. The substrate is
    /// millimetres thick, so it is far beyond any coherence length the canvas
    /// models and the two faces add **incoherently** — no interference fringes
    /// across the plate, which is correct and is also why the plate stays
    /// zero-thickness geometrically.
    Plate {
        substrate: FilmMaterial,
        exit: ExitFace,
        /// The plate's thickness, mm, which only its absorption reads: each
        /// pass through the glass is attenuated by the substrate's own `k`
        /// along the slant path, `exp(-4π k d / (λ cos θ))`. A silvered sheet
        /// of float glass loses a few percent this way, most of it near 1 µm
        /// where the iron absorbs. `0`, the default, is a lossless plate.
        #[serde(default, skip_serializing_if = "is_zero")]
        thickness_mm: f64,
    },
    /// A semi-infinite absorbing substrate: a metal block, a mirror whose
    /// metal is far thicker than its skin depth. Nothing comes out the far
    /// side. What the stack transmits into the substrate is absorbed there,
    /// and the tracer books it as absorption and spawns no transmitted ray.
    ///
    /// Only the film side can be lit: light arriving through the substrate
    /// would arrive through an absorbing medium, which is
    /// [`FilmError::AbsorbingIncidentMedium`].
    Opaque { substrate: FilmMaterial },
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// The second face of a plate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExitFace {
    /// Bare glass: plain Fresnel at the internal angle.
    Bare,
    /// Anti-reflected, `residual` scaling the bare Fresnel reflectance. `0` is a
    /// perfect AR, `1` is bare. A real broadband AR sits near `0.05`.
    AntiReflective { residual: f64 },
}

/// Which way round the sandwich is entered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Incidence {
    /// The light meets the film stack first. This is the coated face.
    Film,
    /// The light arrives through what the stack is backed by and meets its
    /// layers last first: the back of a coated plate, or a lens's own coated
    /// face seen from inside the glass.
    Substrate,
}

/// A coating: an ordered stack of layers and what sits behind it.
///
/// Layers run from the incident side inward, which is how a coating is written
/// down and the order a deposition run would lay them in reverse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coating {
    pub layers: Vec<FilmLayer>,
    pub backing: CoatingBacking,
    #[serde(default = "incidence_film")]
    pub incidence: Incidence,
    /// What this stack was designed for, carried so the editor can say so and
    /// re-synthesize.
    ///
    /// Not used by the reflectance calculation: the layers are the truth, and
    /// an edited stack is simply a different coating. It does say what the
    /// part is *for*, and one thing reads that: whether the light the stack
    /// reflects is a path or a ghost ([`Coating::reflection_is_stray`]).
    #[serde(default)]
    pub design: Option<CoatingDesign>,
}

impl Coating {
    /// The same coating, met from the side it is backed by.
    ///
    /// The layers stay in the order they were written, so a layer edit
    /// addresses the same physical layer on both sides; only the side the
    /// light arrives from changes, and the evaluation walks the stack from
    /// there. See [`Incidence::Substrate`].
    pub fn seen_from_behind(&self) -> Coating {
        let mut c = self.clone();
        c.incidence = Incidence::Substrate;
        c
    }

    /// Where `coating_model: Nominal` evaluates this coating: `(lambda_um,
    /// cos_i, designed)`. The design's own wavelength and angle when it has a
    /// design record; 0.55 µm at normal incidence when it has none, which
    /// `designed = false` says so the result can say it too.
    pub fn nominal_point(&self) -> (f64, f64, bool) {
        match &self.design {
            Some(d) if d.lambda_nm.is_finite() && d.lambda_nm > 0.0 => {
                (d.lambda_nm / 1000.0, d.aoi_deg.to_radians().cos().abs(), true)
            }
            _ => (0.55, 1.0, false),
        }
    }

    /// Is what this stack reflects stray light, rather than a path?
    ///
    /// An AR coat exists to transmit. The little it still reflects is the
    /// residual it failed to suppress, which is a ghost for exactly the reason
    /// bare glass's reflection is one, so it is traced when ghosts are and
    /// named in the ledger when they are not. Every other design exists to
    /// reflect something (a band, a mirror's whole spectrum, a splitter's s),
    /// so its reflection is the part doing its job and is always followed.
    ///
    /// Read off the design because that is where the intent is written down.
    /// A layer edit keeps the design, so a hand-thinned AR coat is still an AR
    /// coat. A stack with no design at all reads as a path, which is what a
    /// hand-written dichroic or mirror wants.
    pub fn reflection_is_stray(&self) -> bool {
        matches!(
            self.design.map(|d| d.kind),
            Some(
                DesignKind::SingleLayerAr
                    | DesignKind::BroadbandArLegacy
                    | DesignKind::VCoat
                    | DesignKind::Bbar
                    | DesignKind::BandAr
            )
        )
    }
}

fn incidence_film() -> Incidence {
    Incidence::Film
}

/// The recipe a synthesized stack came from, so the editor can show the knobs
/// that produced it and rebuild when one moves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoatingDesign {
    pub kind: DesignKind,
    /// The wavelength the layers are a quarter wave at, nm.
    pub lambda_nm: f64,
    /// The angle of incidence the design is tilted for, degrees.
    pub aoi_deg: f64,
    /// Repeat count of the basic period.
    pub periods: u32,
    pub n_high: f64,
    pub n_low: f64,
    /// What a substrate-adapted design is adapted to (CF-08): the substrate,
    /// the band and angles it is judged over, its materials, and its design
    /// point. Absent from every older design, which builds as it always did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapt: Option<crate::synthesis::DesignAdapt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DesignKind {
    /// `(HL)^N H`: a quarter-wave mirror. Reflects a band centred on the design
    /// wavelength and passes both sides of it.
    QuarterWaveStack,
    /// A quarter-wave stack placed so the **long-wavelength** edge of its
    /// reflectance band falls at the requested edge: short wavelengths reflect,
    /// long ones pass.
    LongPass,
    /// The same, placed so the short-wavelength edge falls at the requested
    /// edge: long wavelengths reflect, short ones pass.
    ShortPass,
    /// `(HL)^N H` again, but used at the angle where the pair's own interfaces
    /// sit at Brewster: p reflects nothing anywhere in the stack, so the stack
    /// passes p and reflects s. This is the polarizing beam splitter, and it
    /// only works cemented in glass of the index the condition demands. See
    /// [`macneille_glass`].
    Polarizing,
    /// A single quarter wave of a low index: the classic single-layer AR.
    /// Without an `adapt` record it is constant MgF₂, as it always was; with
    /// one, the catalogue film nearest `√(n₀ n_s)` ([`crate::synthesis`]).
    SingleLayerAr,
    /// Quarter-half-quarter of constant MgF₂, TiO₂ and Al₂O₃ at one
    /// wavelength: the broadband preset before the catalogue existed.
    /// Nominal, not adapted: it ignores the substrate, and on N-SF66 it
    /// reflects about 10% of the visible. Written `BroadbandAr` in documents.
    #[serde(rename = "BroadbandAr")]
    BroadbandArLegacy,
    /// Two layers with an exact null at the design wavelength (a V-coat).
    VCoat,
    /// Quarter-half-quarter with the inner quarter matched to the substrate.
    Bbar,
    /// Up to 8 layers refined over a band and an angle range.
    BandAr,
    /// A metal under quarter-wave high/low pairs.
    EnhancedMetal { metal: crate::synthesis::MetalKind, pairs: u32 },
}

/// Reflectance and transmittance, resolved by polarization.
///
/// The kernel traces unpolarized light, so it consumes [`FilmRt::r`] and
/// [`FilmRt::t`]. The separated components are kept because they are where the
/// interesting behaviour of a tilted coating lives, and the component view plots
/// them: at 45° a dichroic's s and p edges can sit tens of nanometres apart, and
/// a mean that hides that would be worse than no number at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmRt {
    pub r_s: f64,
    pub r_p: f64,
    pub t_s: f64,
    pub t_p: f64,
}

impl FilmRt {
    /// Unpolarized reflectance: the mean of the two components, which is what
    /// unpolarized light sees.
    pub fn r(&self) -> f64 {
        0.5 * (self.r_s + self.r_p)
    }
    pub fn t(&self) -> f64 {
        0.5 * (self.t_s + self.t_p)
    }
    /// What the layers absorbed, never negative after rounding.
    pub fn a(&self) -> f64 {
        (1.0 - self.r() - self.t()).max(0.0)
    }
    /// How far apart the two polarizations are in reflectance. Zero at normal
    /// incidence by symmetry; a tilted dichroic is where it grows teeth.
    pub fn polarization_split(&self) -> f64 {
        (self.r_s - self.r_p).abs()
    }
}

/// Macleod's forward branch for the propagation cosine in a medium of complex
/// index `n`, given the conserved `q = n₀ sin θ₀`.
///
/// `sqrt(n̄² − q²) / n`, which reduces to the schoolbook `cos θ` for a
/// transparent medium and stays on the physically forward branch for an
/// absorbing one.
pub fn snell_cos(q: Cx, n: Cx) -> Cx {
    let n_bar_sq = Cx::new(n.re * n.re - n.im * n.im, -2.0 * n.re * n.im);
    (n_bar_sq - q * q).sqrt() / n
}

/// Put an exit medium's propagation cosine on the **decaying** branch.
///
/// [`snell_cos`] takes the principal square root, which is what numpy returns
/// and so what keeps this module bit-compatible with Optiland. On the negative
/// real axis — a transparent medium past its critical angle — the principal
/// branch is the *growing* evanescent wave in the `exp(i(ωt − kz))` convention
/// the characteristic matrix is written in, where decay means `Im(n cos θ) ≤ 0`.
///
/// For a lossless stack no power notices: past the critical angle `R = 1` and
/// `T = 0` on either branch, which is why this went unremarked while the module
/// only computed powers. The reflection **phase** notices, it is the only thing
/// a polarization calculation reads, and it is the whole of what a Fresnel
/// rhomb does. A stack that absorbs notices in power too: the evanescent field
/// reaches into the absorbing layer, and on the growing branch the attenuated
/// total reflection of a metal film comes out wrong by tens of percent (the
/// `asymmetric_absorbing_reversed` coating fixture, against `tmm`). A finite
/// layer is invariant under the flip (its matrix is even in `cos θ`), so the
/// exit medium is the only place this can matter.
fn decaying_cos(n: Cx, cos_t: Cx) -> Cx {
    #[cfg(test)]
    if mutation::on(Mutation::GrowingBranch) {
        return cos_t;
    }
    if (n * cos_t).im > 0.0 {
        -cos_t
    } else {
        cos_t
    }
}

/// Tilted optical admittance. `η_s = n cos θ`, `η_p = n̄² / (n cos θ)`.
///
/// The free-space admittance is dropped: it is a constant factor that cancels in
/// `r`, and in `T` it cancels between `Re(η_sub)` and `Re(η_0)`.
fn admittance(n: Cx, cos_t: Cx, p_pol: bool) -> Cx {
    let eta_s = n * cos_t;
    if !p_pol {
        eta_s
    } else {
        n.conj() * n.conj() / eta_s
    }
}

/// The imaginary unit, which the characteristic matrix is full of.
const I: Cx = Cx::new(0.0, 1.0);

/// A layer whose phase thickness has an imaginary part above this is opaque:
/// it passes `exp(-35)` of the field, `exp(-70)` of the power, and nothing
/// behind it can be seen from the front. The value is Byrnes' (`tmm`), so the
/// canvas and that oracle draw the line in the same place. The canvas
/// truncates the stack there, the opaque layer becoming the substrate, where
/// `tmm` clamps the phase; the two differ by less than `exp(-70)`.
pub const OPAQUE_PHASE: f64 = 35.0;

/// One polarization of one coherent stack, as Macleod's amplitudes and what
/// happened on the way to them.
#[derive(Debug, Clone, Copy)]
struct PolAmp {
    r: Cx,
    t: Cx,
    eta0: Cx,
    eta_s: Cx,
    /// The first opaque layer, in the order the light meets them.
    opaque_from: Option<usize>,
    /// The denominator vanished or overflowed: `r = 1`, `t = 0` by rule.
    degenerate: bool,
}

/// One coherent stack, as **complex amplitudes**.
///
/// `r` and `t` are Macleod's coefficients for the *tangential* field: for s
/// polarization the field is wholly tangential and `t` is the whole of it, but
/// for p the tangential part is `E cos θ`, so `t_p` here is
/// `t_full · cos θ_sub / cos θ₀`. [`coating_amp_at`] undoes that; the power
/// coefficients below never need it, because the same factor cancels against
/// the admittance ratio.
///
/// Returns the two admittances as well, since the transmitted power needs them.
/// `layers` run from the incident side inward; `q` is the conserved `n₀ sin θ₀`.
///
/// The phase convention is fixed by the sign of `δ` in the layer matrix and is
/// `exp(-iωt)`, as `a_single_layer_matches_the_airy_formula_and_fixes_the_sign`
/// establishes by comparing against an explicit two-interface sum. Everything
/// downstream that reads a phase depends on that test.
///
/// Two guards keep a thick absorbing stack finite, and neither touches a
/// stack that gave a finite answer before they existed. A layer past
/// [`OPAQUE_PHASE`] ends the stack: it becomes the substrate, and `t` is
/// what enters it. And the accumulated matrix is rescaled whenever an entry
/// passes `1e100`, which leaves `r` unchanged and divides `t` by the scale,
/// so many layers that each absorb a little underflow `t` to zero where they
/// used to overflow to NaN.
fn stack_amp_pol(
    layers: &[(Cx, f64)],
    n0: Cx,
    n_sub: Cx,
    lambda_um: f64,
    q: Cx,
    p_pol: bool,
) -> PolAmp {
    let cos0 = snell_cos(q, n0);
    let cos_s = decaying_cos(n_sub, snell_cos(q, n_sub));
    let eta0 = admittance(n0, cos0, p_pol);
    let eta_s = admittance(n_sub, cos_s, p_pol);

    // The stack matrix, accumulated from the incident side inward.
    let (mut a, mut b, mut c, mut d) = (Cx::real(1.0), Cx::real(0.0), Cx::real(0.0), Cx::real(1.0));
    // ln of what the matrix has been divided by.
    let mut log_scale = 0.0f64;
    for (k, &(n_l, thickness_nm)) in layers.iter().enumerate() {
        let cos_l = snell_cos(q, n_l);
        let eta_l = admittance(n_l, cos_l, p_pol);
        // δ = 2π/λ · n · d · cos θ, with d in µm to match λ.
        let k0 = std::f64::consts::TAU / lambda_um;
        let delta = n_l * cos_l * (k0 * thickness_nm / 1000.0);
        #[cfg(test)]
        let delta = if mutation::on(Mutation::ThicknessAsMicrons) {
            n_l * cos_l * (k0 * thickness_nm)
        } else {
            delta
        };
        if delta.im.abs() > OPAQUE_PHASE {
            // Opaque: the layer is the substrate, and nothing behind it counts.
            let mut out = stack_amp_pol(&layers[..k], n0, n_l, lambda_um, q, p_pol);
            out.opaque_from = Some(k);
            return out;
        }
        let (cs, sn) = (delta.cos(), delta.sin());
        let m_a = cs;
        let m_b = I * (sn / eta_l);
        let m_c = I * (eta_l * sn);
        let m_d = cs;
        let (na, nb, nc, nd) = (
            a * m_a + b * m_c,
            a * m_b + b * m_d,
            c * m_a + d * m_c,
            c * m_b + d * m_d,
        );
        a = na;
        b = nb;
        c = nc;
        d = nd;
        let big = a.norm_sqr().max(b.norm_sqr()).max(c.norm_sqr()).max(d.norm_sqr());
        if big > 1e200 {
            let k = 1.0 / big.sqrt();
            a = a * k;
            b = b * k;
            c = c * k;
            d = d * k;
            log_scale += 0.5 * big.ln();
        }
    }

    // [B, C]ᵀ = M · [1, η_sub]ᵀ.
    let big_b = a + eta_s * b;
    let big_c = c + eta_s * d;
    let denom = eta0 * big_b + big_c;
    if denom.norm_sqr() == 0.0 || !denom.is_finite() {
        // A degenerate stack reflects everything rather than returning NaN.
        return PolAmp { r: Cx::real(1.0), t: Cx::real(0.0), eta0, eta_s, opaque_from: None, degenerate: true };
    }
    let r = (eta0 * big_b - big_c) / denom;
    let t = eta0 * 2.0 / denom;
    let t = if log_scale > 0.0 { t * (-log_scale).exp() } else { t };
    PolAmp { r, t, eta0, eta_s, opaque_from: None, degenerate: false }
}

/// One polarization's powers, and what happened on the way to them.
#[derive(Debug, Clone, Copy)]
struct PolRt {
    r: f64,
    t: f64,
    opaque_from: Option<usize>,
    degenerate: bool,
    finite: bool,
}

/// One coherent stack: reflectance and transmittance for a single polarization.
fn stack_rt_pol(
    layers: &[(Cx, f64)],
    n0: Cx,
    n_sub: Cx,
    lambda_um: f64,
    q: Cx,
    p_pol: bool,
) -> PolRt {
    let PolAmp { r, t, eta0, eta_s, opaque_from, degenerate } =
        stack_amp_pol(layers, n0, n_sub, lambda_um, q, p_pol);
    let big_r = r.norm_sqr();
    // A substrate whose admittance has no real part carries no power: the wave
    // there is evanescent and everything that is not absorbed comes back.
    let big_t = if eta0.re.abs() > 0.0 {
        (t.norm_sqr() * eta_s.re / eta0.re).max(0.0)
    } else {
        0.0
    };
    #[cfg(test)]
    let big_t = if mutation::on(Mutation::NoAdmittanceRatio) {
        t.norm_sqr()
    } else {
        big_t
    };
    // Behind an opaque layer nothing is transmitted: what entered it was
    // absorbed there.
    let big_t = if opaque_from.is_some() { 0.0 } else { big_t };
    PolRt {
        r: big_r.clamp(0.0, 1.0),
        t: big_t.clamp(0.0, 1.0),
        opaque_from,
        degenerate,
        finite: big_r.is_finite() && big_t.is_finite(),
    }
}

/// Unpolarized Fresnel reflectance for one polarization at a bare interface,
/// used for the far face of a plate.
fn bare_interface_r(n1: Cx, n2: Cx, q: Cx, p_pol: bool) -> f64 {
    let (c1, c2) = (snell_cos(q, n1), snell_cos(q, n2));
    let (e1, e2) = (admittance(n1, c1, p_pol), admittance(n2, c2, p_pol));
    let denom = e1 + e2;
    // Exactly at the critical angle the p admittance of the far side is
    // infinite, the ratio is ∞/∞, and the limit is total reflection.
    if denom.norm_sqr() == 0.0 || !denom.is_finite() {
        return 1.0;
    }
    ((e1 - e2) / denom).norm_sqr().clamp(0.0, 1.0)
}

/// Why a coating has no answer at a wavelength and angle (spec CF-02).
///
/// Every refusal has a name and a sentence, so a result can say why a coated
/// face was not traced rather than reflecting everything or returning NaN.
/// Under `numerics: Permissive` only the errors that have no number at all are
/// raised (a table that refuses its wavelength, a non-physical index, an
/// overflow, an absorbing incident medium); under `Strict` a clamped table, a
/// negative extinction coefficient and a degenerate stack are refused too.
#[derive(Debug, Clone, PartialEq)]
pub enum FilmError {
    /// The stack's matrix is singular at this wavelength and angle.
    Degenerate,
    /// A layer's optical constants do not cover the wavelength. `layer` counts
    /// from 0 in the written order; [`FilmError::SUBSTRATE`] is the substrate.
    OutOfTable { layer: usize, lambda_um: f64 },
    /// A non-positive or non-finite index, a negative extinction coefficient
    /// (under `Strict`), or a thickness that is negative or not finite.
    NonPhysicalIndex { layer: usize },
    /// Light arriving through an absorbing medium.
    AbsorbingIncidentMedium,
    /// The arithmetic left the finite numbers.
    Overflow,
    /// A plate-backed coating adds its two faces incoherently and has no
    /// single complex amplitude to give.
    Incoherent,
    /// Slabs inside a plate-backed coating: the plate already is the slab.
    SlabOnPlate,
}

impl FilmError {
    /// The `layer` of an error about the substrate.
    pub const SUBSTRATE: usize = usize::MAX;
}

impl std::fmt::Display for FilmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let which = |layer: usize| {
            if layer == FilmError::SUBSTRATE {
                "the substrate".to_string()
            } else {
                format!("layer {}", layer + 1)
            }
        };
        match self {
            FilmError::Degenerate => write!(
                f,
                "the coating's transfer matrix is singular at this wavelength and angle, so it has no reflectance"
            ),
            FilmError::OutOfTable { layer, lambda_um } => write!(
                f,
                "{} has no optical constants at {:.4} um: its table does not reach that wavelength",
                which(*layer),
                lambda_um
            ),
            FilmError::NonPhysicalIndex { layer } => write!(
                f,
                "{} is not physical: the index must be positive and finite, the extinction coefficient not negative, and the thickness finite and not negative",
                which(*layer)
            ),
            FilmError::AbsorbingIncidentMedium => write!(
                f,
                "light cannot arrive through an absorbing substrate: an opaque-backed coating is lit from its film side only"
            ),
            FilmError::Overflow => write!(f, "the coating's arithmetic overflowed, so it has no finite answer"),
            FilmError::Incoherent => write!(
                f,
                "a plate-backed coating adds its two faces incoherently and has no single complex amplitude"
            ),
            FilmError::SlabOnPlate => write!(
                f,
                "a plate-backed coating cannot hold slabs: write it as its films and a slab on the far medium instead"
            ),
        }
    }
}

/// What a coating is evaluated under: the parts of the fidelity profile that
/// reach the solver.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FilmContext {
    pub numerics: crate::fidelity::Numerics,
    /// `Nominal`: every material at its nominal constant.
    pub materials: crate::fidelity::FilmMaterials,
    /// `Incoherent`: a plate is evaluated as its films, its slab and its far
    /// face by the incoherent transfer-matrix method, and has a Mueller
    /// matrix; `PhaseFree` keeps the plate's own series.
    pub plate: crate::fidelity::PlateModel,
}

impl FilmContext {
    /// The solver's part of a fidelity profile.
    pub fn from_profile(p: &crate::fidelity::FidelityProfile) -> Self {
        Self { numerics: p.numerics, materials: p.film_materials, plate: p.plate_model }
    }

    pub fn strict(&self) -> bool {
        self.numerics == crate::fidelity::Numerics::Strict
    }
}

/// What happened while a coating was evaluated, for the result to say.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FilmNotes {
    /// The first opaque layer, counted in the written order.
    pub opaque_from: Option<usize>,
    /// Layers (and [`FilmError::SUBSTRATE`]) whose tables were read outside
    /// their range and answered with their out-of-range policy.
    pub clamped: Vec<usize>,
    /// The stack was degenerate and reflected everything by rule.
    pub degenerate: bool,
}

/// Below this cosine the ray is grazing: it reflects entirely, by rule, and
/// the matrix is not asked.
pub const GRAZING_COS: f64 = 1e-9;

/// A material's complex index at `lambda_um` under `ctx`, for layer `layer`.
fn sample_index(
    m: &FilmMaterial,
    lambda_um: f64,
    ctx: &FilmContext,
    layer: usize,
    notes: &mut FilmNotes,
) -> Result<Cx, FilmError> {
    if ctx.materials == crate::fidelity::FilmMaterials::Nominal {
        let n = m.nominal_index();
        if !n.is_finite() || n.re <= 0.0 || (ctx.strict() && n.im < 0.0) {
            return Err(FilmError::NonPhysicalIndex { layer });
        }
        return Ok(n);
    }
    let covered = m.covers(lambda_um);
    if !covered && ctx.strict() {
        return Err(FilmError::OutOfTable { layer, lambda_um });
    }
    let n = m.index_at(lambda_um).ok_or(FilmError::OutOfTable { layer, lambda_um })?;
    if !covered {
        notes.clamped.push(layer);
    }
    if !n.is_finite() || n.re <= 0.0 || (ctx.strict() && n.im < 0.0) {
        return Err(FilmError::NonPhysicalIndex { layer });
    }
    Ok(n)
}

/// The layers' indices and thicknesses in the written order.
fn sample_layers(
    c: &Coating,
    lambda_um: f64,
    ctx: &FilmContext,
    notes: &mut FilmNotes,
) -> Result<Vec<(Cx, f64)>, FilmError> {
    let mut layers = Vec::with_capacity(c.layers.len());
    for (i, l) in c.layers.iter().enumerate() {
        let n = sample_index(&l.material, lambda_um, ctx, i, notes)?;
        if !l.thickness_nm.is_finite() || l.thickness_nm < 0.0 {
            return Err(FilmError::NonPhysicalIndex { layer: i });
        }
        layers.push((n, l.thickness_nm));
    }
    Ok(layers)
}

/// The conserved `n₀ sin θ₀` and the incident index, as the solver takes them.
fn incidence(n_incident: f64, cos_i: f64) -> (Cx, f64, Cx) {
    let n0 = Cx::real(n_incident.max(1e-9));
    let cos_i = cos_i.clamp(-1.0, 1.0).abs();
    let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    (n0, cos_i, n0 * sin_i)
}

/// An opaque layer index in the walked order, back in the written order.
fn written_index(c: &Coating, walked: usize) -> usize {
    if matches!(c.backing, CoatingBacking::Medium) && matches!(c.incidence, Incidence::Substrate) {
        c.layers.len() - 1 - walked
    } else {
        walked
    }
}

/// Fold one polarization's flags into the notes, or refuse.
fn take_pol(c: &Coating, p: &PolRt, ctx: &FilmContext, notes: &mut FilmNotes) -> Result<(), FilmError> {
    if !p.finite {
        return Err(FilmError::Overflow);
    }
    if p.degenerate {
        if ctx.strict() {
            return Err(FilmError::Degenerate);
        }
        notes.degenerate = true;
    }
    if let Some(k) = p.opaque_from {
        notes.opaque_from = Some(written_index(c, k));
    }
    Ok(())
}

/// [`coating_rt_at`] under a context, with what happened on the way.
pub fn coating_eval(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
    ctx: &FilmContext,
) -> Result<(FilmRt, FilmNotes), FilmError> {
    if uses_mixed(c, ctx) {
        return mixed_eval(c, n_incident, n_far, lambda_um, cos_i, ctx).map(|(p, notes)| (p.rt, notes));
    }
    let (n0, cos_i, q) = incidence(n_incident, cos_i);
    let n_far = Cx::real(n_far.max(1e-9));
    let mut notes = FilmNotes::default();
    if matches!(c.backing, CoatingBacking::Opaque { .. }) && matches!(c.incidence, Incidence::Substrate) {
        return Err(FilmError::AbsorbingIncidentMedium);
    }
    let mut layers = sample_layers(c, lambda_um, ctx, &mut notes)?;
    walk_from_the_incident_side(c, &mut layers);
    if cos_i < GRAZING_COS {
        return Ok((FilmRt { r_s: 1.0, r_p: 1.0, t_s: 0.0, t_p: 0.0 }, notes));
    }

    let rt = match &c.backing {
        // The stack sits straight on whatever the scene has on the far side: an
        // AR coating on a lens face is backed by that lens's own glass.
        CoatingBacking::Medium => {
            let s = stack_rt_pol(&layers, n0, n_far, lambda_um, q, false);
            let p = stack_rt_pol(&layers, n0, n_far, lambda_um, q, true);
            take_pol(c, &s, ctx, &mut notes)?;
            take_pol(c, &p, ctx, &mut notes)?;
            FilmRt { r_s: s.r, r_p: p.r, t_s: s.t, t_p: p.t }
        }
        CoatingBacking::Opaque { substrate } => {
            let n_sub = sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut notes)?;
            let s = stack_rt_pol(&layers, n0, n_sub, lambda_um, q, false);
            let p = stack_rt_pol(&layers, n0, n_sub, lambda_um, q, true);
            take_pol(c, &s, ctx, &mut notes)?;
            take_pol(c, &p, ctx, &mut notes)?;
            // Everything that enters the substrate stays there.
            FilmRt { r_s: s.r, r_p: p.r, t_s: 0.0, t_p: 0.0 }
        }
        CoatingBacking::Plate { substrate, exit, thickness_mm } => {
            let n_sub = sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut notes)?;
            let residual = match exit {
                ExitFace::Bare => 1.0,
                ExitFace::AntiReflective { residual } => residual.clamp(0.0, 1.0),
            };
            plate_rt(
                &layers,
                n0,
                n_sub,
                n_far,
                lambda_um,
                q,
                residual,
                c.incidence,
                thickness_mm.max(0.0) * 1e3,
                ctx,
                &mut notes,
            )?
        }
    };
    Ok((rt, notes))
}

/// Reflectance and transmittance of a coating at one wavelength and angle.
///
/// `n_incident` is the index of the medium the ray is arriving through, which
/// the scene owns: the same coating on the same plate behaves differently in air
/// and in oil, and the tracer knows which it is. `cos_i` is the cosine of the
/// angle of incidence, positive.
///
/// Evaluated under `numerics: Permissive`: a table answers outside its range
/// with its own out-of-range policy, and refuses only where that policy is
/// `Error`. [`coating_eval`] takes the context and says what happened.
pub fn coating_rt_at(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
) -> Result<FilmRt, FilmError> {
    coating_eval(c, n_incident, n_far, lambda_um, cos_i, &FilmContext::default()).map(|(rt, _)| rt)
}

/// [`coating_rt_at`] for a surface with the same medium on both sides, which is
/// every zero-thickness surface in the scene (§6.6) and so every plate.
pub fn coating_rt(c: &Coating, n_incident: f64, lambda_um: f64, cos_i: f64) -> Result<FilmRt, FilmError> {
    coating_rt_at(c, n_incident, n_incident, lambda_um, cos_i)
}

/// A coating's **complex amplitude** response, in the kernel's ray frame.
///
/// Powers say how much light goes each way; these say what happens to its
/// polarization, which needs the relative phase between s and p that
/// [`FilmRt`] throws away. A quarter-wave stack near its band edge can put tens
/// of degrees of retardance between the two, and a metal film or a total
/// internal reflection puts in a great deal more.
///
/// The frame is the one [`crate::polarization`] defines: `ŝ = d̂ × n̂` shared by
/// the incoming and outgoing rays, `p̂ = d̂ × ŝ` following each ray's own
/// direction. Two conversions from Macleod's algebra live here and nowhere
/// else:
///
/// - **`r_p` changes sign.** Macleod's p admittance makes `r_p = r_s` at normal
///   incidence; in a frame where `p̂` follows the reflected ray, `p̂` itself
///   flips, so the coefficient must too. Getting this wrong is invisible in any
///   power and inverts every polarization result.
/// - **`t_p` is rescaled** from the tangential field to the whole field, by
///   `cos θ₀ / cos θ_exit`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmAmp {
    pub r_s: Cx,
    pub r_p: Cx,
    /// Full-field transmitted amplitudes: `|t|²` is **not** the transmittance;
    /// multiply by [`FilmAmp::t_power_scale`].
    pub t_s: Cx,
    pub t_p: Cx,
    /// `Re(n_exit cos θ_exit) / Re(n₀ cos θ₀)`, the factor turning `|t|²` into a
    /// transmittance. Zero when the far side is evanescent.
    pub t_power_scale: f64,
}

impl FilmAmp {
    /// The powers these amplitudes imply, which must agree with
    /// [`coating_rt_at`] to the last bit; `the_amplitudes_and_the_powers_agree`
    /// holds them to it.
    pub fn powers(&self) -> FilmRt {
        FilmRt {
            r_s: self.r_s.norm_sqr().clamp(0.0, 1.0),
            r_p: self.r_p.norm_sqr().clamp(0.0, 1.0),
            t_s: (self.t_s.norm_sqr() * self.t_power_scale).clamp(0.0, 1.0),
            t_p: (self.t_p.norm_sqr() * self.t_power_scale).clamp(0.0, 1.0),
        }
    }

    /// Retardance the coating adds between p and s on reflection, radians.
    pub fn reflection_retardance(&self) -> f64 {
        self.r_p.arg() - self.r_s.arg()
    }

    /// Retardance the coating adds between p and s on transmission,
    /// `arg t_p − arg t_s`, radians. Both rays travel forward, so there is no
    /// basis flip in it: zero at normal incidence for any stack.
    pub fn transmission_retardance(&self) -> f64 {
        self.t_p.arg() - self.t_s.arg()
    }

    /// The ellipsometric angles `(ψ, Δ)`, radians: `tan ψ · e^{iΔ} = r_p / r_s`
    /// on the ray-frame amplitudes, `exp(-iωt)`. ψ in `[0, π/2]`, Δ in
    /// `(−π, π]`. In the ray frame `r_p = −r_s` at normal incidence, so a bare
    /// dielectric below Brewster reads Δ = π there, the Azzam and Bashara
    /// (Nebraska) convention's value. `tmm.ellips` defines Δ from `−r_p / r_s`
    /// on the same amplitudes, so `Δ_tmm = Δ − π` (mod 2π) and ψ is the same;
    /// `docs/physics/conventions.md` holds the table.
    pub fn ellipsometric(&self) -> (f64, f64) {
        let psi = self.r_p.abs().atan2(self.r_s.abs());
        let rho = self.r_p / self.r_s;
        (psi, rho.arg())
    }
}

/// [`coating_rt_at`]'s amplitudes: `r` and `t` for s and p, with their phases.
///
/// Refuses a coating whose backing is a [`CoatingBacking::Plate`] with
/// [`FilmError::Incoherent`]. That model adds two faces **incoherently**, which
/// is right for millimetres of glass and is not a single Jones matrix: the
/// relative phase between the two contributions is exactly what the
/// incoherent sum discards, so there is no honest amplitude to return. A
/// coating on a real glass body uses [`CoatingBacking::Medium`] and does have
/// one, and so does an [`CoatingBacking::Opaque`] one, whose transmitted
/// amplitudes are zero.
///
/// The supported domain is a real, non-absorbing incident medium; the layers
/// and an opaque substrate may absorb.
pub fn coating_amp_at(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
) -> Result<FilmAmp, FilmError> {
    coating_amp_eval(c, n_incident, n_far, lambda_um, cos_i, &FilmContext::default()).map(|(a, _)| a)
}

/// [`coating_amp_at`] under a context, with what happened on the way.
pub fn coating_amp_eval(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
    ctx: &FilmContext,
) -> Result<(FilmAmp, FilmNotes), FilmError> {
    if matches!(c.backing, CoatingBacking::Plate { .. }) || has_slab(c) {
        return Err(FilmError::Incoherent);
    }
    if matches!(c.backing, CoatingBacking::Opaque { .. }) && matches!(c.incidence, Incidence::Substrate) {
        return Err(FilmError::AbsorbingIncidentMedium);
    }
    let (n0, cos_i, q) = incidence(n_incident, cos_i);
    let mut notes = FilmNotes::default();
    let mut layers = sample_layers(c, lambda_um, ctx, &mut notes)?;
    walk_from_the_incident_side(c, &mut layers);
    if cos_i < GRAZING_COS {
        // Grazing: total reflection, `r = -1` in both polarizations in the ray
        // frame (Macleod's `r_p` is `+1` there, and the frame flips it).
        let minus_one = Cx::real(-1.0);
        let zero = Cx::real(0.0);
        let a = FilmAmp { r_s: minus_one, r_p: minus_one, t_s: zero, t_p: zero, t_power_scale: 0.0 };
        return Ok((a, notes));
    }
    let (n_sub, opaque) = match &c.backing {
        CoatingBacking::Opaque { substrate } => {
            (sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut notes)?, true)
        }
        _ => (Cx::real(n_far.max(1e-9)), false),
    };
    let (mut a, s, p) = stack_amp(&layers, n0, n_sub, lambda_um, q);
    for pol in [&s, &p] {
        if !pol.r.is_finite() || !pol.t.is_finite() {
            return Err(FilmError::Overflow);
        }
        if pol.degenerate {
            if ctx.strict() {
                return Err(FilmError::Degenerate);
            }
            notes.degenerate = true;
        }
        if let Some(k) = pol.opaque_from {
            notes.opaque_from = Some(written_index(c, k));
        }
    }
    if opaque || notes.opaque_from.is_some() {
        a.t_s = Cx::real(0.0);
        a.t_p = Cx::real(0.0);
        a.t_power_scale = 0.0;
    }
    Ok((a, notes))
}

/// Put a stack's layers in the order the light meets them.
///
/// They are written from the film side inward. A stack sitting straight on a
/// medium and met from that medium, which is a lens's coated face seen from
/// inside the glass, is met the other way round, and walking it in the
/// written order would put the layer meant to face the air against the glass.
/// A symmetric stack cannot tell the difference, which is how a single-layer
/// AR hid this; a three-layer broadband AR is not symmetric, and walked the
/// wrong way it reflects more than bare glass. A plate-backed stack handles
/// its own two sides in `plate_rt` and is left alone here.
fn walk_from_the_incident_side(c: &Coating, layers: &mut [(Cx, f64)]) {
    #[cfg(test)]
    if mutation::on(Mutation::NoReverseWalk) {
        return;
    }
    if matches!(c.backing, CoatingBacking::Medium) && matches!(c.incidence, Incidence::Substrate) {
        layers.reverse();
    }
}

/// The complex amplitudes of a **bare** dielectric interface, in the ray frame.
///
/// A stack with no layers, which is what a bare interface is, so it comes out
/// of the same matrix and needs no second implementation to keep in step. Past
/// the critical angle it returns the reflection with its phase, which is the
/// only thing an interface does there, and the transmitted amplitudes go to
/// zero along with the scale.
pub fn interface_amp(n1: f64, n2: f64, cos_i: f64) -> FilmAmp {
    let n0 = Cx::real(n1.max(1e-9));
    let ns = Cx::real(n2.max(1e-9));
    let cos_i = cos_i.clamp(-1.0, 1.0).abs();
    let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    // The wavelength cannot matter with no layer to put a phase thickness in.
    stack_amp(&[], n0, ns, 0.55, n0 * sin_i).0
}

/// Both polarizations of one coherent stack, converted into the ray frame.
///
/// Three conversions happen here and nowhere else.
///
/// **The direction of time.** The characteristic matrix above is Macleod's, and
/// Macleod writes a wave as `exp(i(ωt − kz))` with index `n − ik`; this module
/// stores `n + ik` and conjugates on the way in, so the amplitudes that come
/// out are his. The rest of the kernel, and every Stokes vector in
/// [`crate::polarization`], uses `exp(-iωt)`. Conjugating the amplitudes is the
/// whole of the conversion: conjugation turns his layer matrix into the other
/// convention's term by term, and `r` and `t` are rational functions of it with
/// real coefficients. Skipping it costs nothing in any power and inverts the
/// handedness of every circular state the kernel produces.
///
/// **The p basis**, and **the tangential-to-whole-field rescaling of `t_p`**,
/// both as [`FilmAmp`] describes.
fn stack_amp(layers: &[(Cx, f64)], n0: Cx, n_sub: Cx, lambda_um: f64, q: Cx) -> (FilmAmp, PolAmp, PolAmp) {
    let pol_s = stack_amp_pol(layers, n0, n_sub, lambda_um, q, false);
    let pol_p = stack_amp_pol(layers, n0, n_sub, lambda_um, q, true);
    let (r_s, t_s, eta0_s, eta_s_s) = (pol_s.r, pol_s.t, pol_s.eta0, pol_s.eta_s);
    let (r_p, t_p) = (pol_p.r, pol_p.t);
    let cos0 = snell_cos(q, n0);
    // The branch `stack_amp_pol` used for the exit admittance. Below the
    // critical angle it is the principal root unchanged; past it, the
    // principal root is the growing wave and would flip the evanescent `t_p`.
    let cos_sub = decaying_cos(n_sub, snell_cos(q, n_sub));
    // Tangential field to whole field for p. An evanescent exit carries no
    // transmitted ray at all, so the scale goes to zero and the amplitude with
    // it rather than through a division by nothing.
    let (t_p_full, scale) = if cos_sub.norm_sqr() > 1e-300 && eta0_s.re.abs() > 0.0 {
        (t_p * (cos0 / cos_sub), (eta_s_s.re / eta0_s.re).max(0.0))
    } else {
        (Cx::real(0.0), 0.0)
    };
    // The one sign that separates Macleod's p basis from the ray frame.
    let r_p = -r_p.conj();
    #[cfg(test)]
    let r_p = if mutation::on(Mutation::RpSignDropped) { -r_p } else { r_p };
    let amp = FilmAmp {
        r_s: r_s.conj(),
        r_p,
        t_s: t_s.conj(),
        t_p: t_p_full.conj(),
        t_power_scale: scale,
    };
    (amp, pol_s, pol_p)
}

/// A coated plate: the coherent stack, then the substrate, then the far face,
/// with the two faces added incoherently.
///
/// The substrate is thick, so its two surfaces do not interfere: intensities
/// add, and the multiple bounces between them sum as a geometric series. That
/// series is the `1 − R' R₂` denominator, and dropping it would lose the second
/// pass of a highly reflective coating entirely.
#[allow(clippy::too_many_arguments)]
fn plate_rt(
    layers: &[(Cx, f64)],
    n0: Cx,
    n_sub: Cx,
    n_far: Cx,
    lambda_um: f64,
    q: Cx,
    exit_residual: f64,
    incidence: Incidence,
    thickness_um: f64,
    ctx: &FilmContext,
    notes: &mut FilmNotes,
) -> Result<FilmRt, FilmError> {
    let reversed: Vec<(Cx, f64)> = layers.iter().rev().copied().collect();
    // One pass through the glass, by its extinction along the slant path.
    let tau = if thickness_um > 0.0 && n_sub.im > 0.0 {
        let cos_t = snell_cos(q, Cx::real(n_sub.re)).re.max(1e-6);
        (-4.0 * std::f64::consts::PI * n_sub.im * thickness_um / (lambda_um * cos_t)).exp()
    } else {
        1.0
    };
    let mut out = [(0.0, 0.0); 2];
    for (idx, p_pol) in [false, true].into_iter().enumerate() {
        // The coated face, lit from the air side and from inside the glass.
        let front = stack_rt_pol(layers, n0, n_sub, lambda_um, q, p_pol);
        let back = stack_rt_pol(&reversed, n_sub, n0, lambda_um, q, p_pol);
        for side in [&front, &back] {
            if !side.finite {
                return Err(FilmError::Overflow);
            }
            if side.degenerate {
                if ctx.strict() {
                    return Err(FilmError::Degenerate);
                }
                notes.degenerate = true;
            }
        }
        if let Some(k) = front.opaque_from {
            notes.opaque_from = Some(k);
        }
        let (r1, t1) = (front.r, front.t);
        let (r1p, t1p) = (back.r, back.t);
        // The bare far face, glass to ambient, at the internal angle.
        let r2 = (bare_interface_r(n_sub, n_far, q, p_pol) * exit_residual).clamp(0.0, 1.0);
        let t2 = 1.0 - r2;

        let (front_r, front_t, back_r, back_t) = match incidence {
            Incidence::Film => (r1, t1, r1p, t1p),
            // Entering through the glass: the far face is met first.
            Incidence::Substrate => (r2, t2, r2, t2),
        };
        let (second_r, second_t) = match incidence {
            Incidence::Film => (r2, t2),
            Incidence::Substrate => (r1p, t1p),
        };

        let tau2 = tau * tau;
        let den = 1.0 - back_r * second_r * tau2;
        let den = if den.abs() < 1e-12 { 1e-12 } else { den };
        let r = (front_r + front_t * back_t * second_r * tau2 / den).clamp(0.0, 1.0);
        let t = (front_t * second_t * tau / den).clamp(0.0, 1.0);
        // Absorption in the stack can only take energy away, never add it.
        let scale = if r + t > 1.0 { 1.0 / (r + t) } else { 1.0 };
        out[idx] = (r * scale, t * scale);
    }
    Ok(FilmRt {
        r_s: out[0].0,
        t_s: out[0].1,
        r_p: out[1].0,
        t_p: out[1].1,
    })
}

// ---------------------------------------------------------------------------
// Where the absorbed power goes: per layer, and into the substrate (CF-04)
// ---------------------------------------------------------------------------

/// A coating's powers with the absorption resolved by element (spec CF-04).
///
/// `a_layer_*[k]` is the fraction of the incident power absorbed in layer `k`,
/// counted in the **written** order; `a_substrate_*` is what the substrate
/// takes: the slab of a plate over all its passes, or what enters an opaque
/// substrate. For each polarization `R + T + Σ A_k + A_substrate = 1` to
/// rounding, and a lossless layer absorbs exactly nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct FilmPowers {
    pub rt: FilmRt,
    pub a_layer_s: Vec<f64>,
    pub a_layer_p: Vec<f64>,
    pub a_substrate_s: f64,
    pub a_substrate_p: f64,
}

impl FilmPowers {
    /// Unpolarized absorption per layer.
    pub fn a_layer(&self) -> Vec<f64> {
        self.a_layer_s.iter().zip(&self.a_layer_p).map(|(s, p)| 0.5 * (s + p)).collect()
    }

    /// Unpolarized absorption in the substrate.
    pub fn a_substrate(&self) -> f64 {
        0.5 * (self.a_substrate_s + self.a_substrate_p)
    }
}

/// The net flux through each interface of a coherent stack, relative to what
/// enters its front face, for one polarization: `F_0 = 1` at the front,
/// `F_N` into the substrate (Macleod's potential transmittance), from the
/// partial products `[B_k; C_k] = (Π_{j>k} M_j) [1; η_sub]` and
/// `F_k = Re(B_k C_k*) / Re(B_0 C_0*)`. The same characteristic matrices as
/// [`stack_amp_pol`], walked from the substrate side, with the same opacity
/// rule and the same rescaling, so a thick stack underflows rather than
/// overflowing.
///
/// Returns the flux at the `N + 1` interfaces of the stack that is actually
/// evaluated, and the first opaque layer if the stack was cut there; the
/// opaque layer is then the substrate, and the flux into it is `F_N`.
fn flux_profile(layers: &[(Cx, f64)], n_sub: Cx, lambda_um: f64, q: Cx, p_pol: bool) -> (Vec<f64>, Option<usize>) {
    let k0 = std::f64::consts::TAU / lambda_um;
    let phase = |n_l: Cx, d_nm: f64| {
        let cos_l = snell_cos(q, n_l);
        (n_l * cos_l * (k0 * d_nm / 1000.0), admittance(n_l, cos_l, p_pol))
    };
    let opaque = layers.iter().position(|&(n, d)| phase(n, d).0.im.abs() > OPAQUE_PHASE);
    let (layers, n_sub) = match opaque {
        Some(k) => (&layers[..k], layers[k].0),
        None => (layers, n_sub),
    };
    let cos_s = decaying_cos(n_sub, snell_cos(q, n_sub));
    let mut b = Cx::real(1.0);
    let mut c = admittance(n_sub, cos_s, p_pol);
    let n = layers.len();
    // `flux[k]` is Re(B_k C_k*) scaled by exp(-2 log_scale[k]).
    let mut flux = vec![0.0; n + 1];
    let mut log_scale = vec![0.0; n + 1];
    let mut ls = 0.0f64;
    flux[n] = (b * c.conj()).re;
    for k in (0..n).rev() {
        let (delta, eta) = phase(layers[k].0, layers[k].1);
        let (cs, sn) = (delta.cos(), delta.sin());
        let nb = cs * b + I * (sn / eta) * c;
        let nc = I * (eta * sn) * b + cs * c;
        b = nb;
        c = nc;
        let big = b.norm_sqr().max(c.norm_sqr());
        if big > 1e200 {
            let s = 1.0 / big.sqrt();
            b = b * s;
            c = c * s;
            ls += 0.5 * big.ln();
        }
        flux[k] = (b * c.conj()).re;
        log_scale[k] = ls;
    }
    let f0 = flux[0];
    let out = flux
        .iter()
        .zip(&log_scale)
        .map(|(f, l)| if f0 == 0.0 { 0.0 } else { f / f0 * (2.0 * (l - log_scale[0])).exp() })
        .collect();
    (out, opaque)
}

/// One coherent stack's absorption by layer for one polarization, in the
/// walked order, and what goes into the substrate: `(1 − R)(F_{k−1} − F_k)`
/// and `(1 − R) F_N`. `r` is the stack's reflectance as [`stack_rt_pol`]
/// computes it, so the per-layer split shares its R with every other path.
/// A lossless layer absorbs nothing, by rule rather than by cancellation.
fn absorbed_by_layer(layers: &[(Cx, f64)], n_sub: Cx, lambda_um: f64, q: Cx, p_pol: bool, r: f64) -> (Vec<f64>, f64) {
    absorbed_by_layer_from(layers, n_sub, lambda_um, q, p_pol, 1.0 - r)
}

/// [`absorbed_by_layer`] given the power that enters the front face, which
/// is `1 − R` from a transparent medium and [`power_entering`] from a lossy one.
fn absorbed_by_layer_from(layers: &[(Cx, f64)], n_sub: Cx, lambda_um: f64, q: Cx, p_pol: bool, into: f64) -> (Vec<f64>, f64) {
    let (f, opaque) = flux_profile(layers, n_sub, lambda_um, q, p_pol);
    let mut a = vec![0.0; layers.len()];
    let evaluated = f.len() - 1;
    for k in 0..evaluated {
        if layers[k].0.im != 0.0 {
            #[cfg(test)]
            let k = if mutation::on(Mutation::AbsorptionOffByOne) { (k + 1).min(evaluated - 1) } else { k };
            a[k.min(layers.len() - 1)] = into * (f[k] - f[k + 1]);
        }
    }
    let beyond = into * f[evaluated];
    match opaque {
        // The opaque layer takes everything that reached it.
        Some(k) => {
            a[k] = beyond;
            (a, 0.0)
        }
        None => (a, beyond),
    }
}

/// [`coating_eval`] with the absorption resolved by layer and substrate.
///
/// For a `Medium` backing `a_substrate` is what the far medium carries away,
/// which is the transmittance and is zero absorption; it is reported as zero
/// and the transmitted power stays in `rt`. For an `Opaque` backing it is the
/// power that enters the substrate and stays there. For a `Plate` it is the
/// slab's absorption over every pass of the incoherent series.
pub fn coating_power_flow_eval(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
    ctx: &FilmContext,
) -> Result<(FilmPowers, FilmNotes), FilmError> {
    if uses_mixed(c, ctx) {
        return mixed_eval(c, n_incident, n_far, lambda_um, cos_i, ctx);
    }
    let (rt, notes) = coating_eval(c, n_incident, n_far, lambda_um, cos_i, ctx)?;
    let (n0, cos_i, q) = incidence(n_incident, cos_i);
    let n_far = Cx::real(n_far.max(1e-9));
    let n = c.layers.len();
    let zero = || vec![0.0; n];
    if cos_i < GRAZING_COS {
        let p = FilmPowers { rt, a_layer_s: zero(), a_layer_p: zero(), a_substrate_s: 0.0, a_substrate_p: 0.0 };
        return Ok((p, notes));
    }
    let mut scratch = FilmNotes::default();
    let mut layers = sample_layers(c, lambda_um, ctx, &mut scratch)?;
    walk_from_the_incident_side(c, &mut layers);
    let written = |walked: Vec<f64>| -> Vec<f64> {
        if matches!(c.backing, CoatingBacking::Medium) && matches!(c.incidence, Incidence::Substrate) {
            walked.into_iter().rev().collect()
        } else {
            walked
        }
    };
    let powers = match &c.backing {
        CoatingBacking::Medium | CoatingBacking::Opaque { .. } => {
            let n_sub = match &c.backing {
                CoatingBacking::Opaque { substrate } => sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut scratch)?,
                _ => n_far,
            };
            let opaque = matches!(c.backing, CoatingBacking::Opaque { .. });
            let (a_s, sub_s) = absorbed_by_layer(&layers, n_sub, lambda_um, q, false, stack_rt_pol(&layers, n0, n_sub, lambda_um, q, false).r);
            let (a_p, sub_p) = absorbed_by_layer(&layers, n_sub, lambda_um, q, true, stack_rt_pol(&layers, n0, n_sub, lambda_um, q, true).r);
            FilmPowers {
                rt,
                a_layer_s: written(a_s),
                a_layer_p: written(a_p),
                a_substrate_s: if opaque { sub_s } else { 0.0 },
                a_substrate_p: if opaque { sub_p } else { 0.0 },
            }
        }
        CoatingBacking::Plate { substrate, exit, thickness_mm } => {
            let n_sub = sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut scratch)?;
            let residual = match exit {
                ExitFace::Bare => 1.0,
                ExitFace::AntiReflective { residual } => residual.clamp(0.0, 1.0),
            };
            let thickness_um = thickness_mm.max(0.0) * 1e3;
            let tau = plate_tau(n_sub, q, lambda_um, thickness_um);
            let reversed: Vec<(Cx, f64)> = layers.iter().rev().copied().collect();
            let mut a = [Vec::new(), Vec::new()];
            let mut sub = [0.0; 2];
            for (idx, p_pol) in [false, true].into_iter().enumerate() {
                let front = stack_rt_pol(&layers, n0, n_sub, lambda_um, q, p_pol);
                let back = stack_rt_pol(&reversed, n_sub, n0, lambda_um, q, p_pol);
                let (a_front, _) = absorbed_by_layer(&layers, n_sub, lambda_um, q, p_pol, front.r);
                let (a_back_rev, _) = absorbed_by_layer(&reversed, n0, lambda_um, q, p_pol, back.r);
                let a_back: Vec<f64> = a_back_rev.into_iter().rev().collect();
                let r2 = (bare_interface_r(n_sub, n_far, q, p_pol) * residual).clamp(0.0, 1.0);
                let t2 = 1.0 - r2;
                let tau2 = tau * tau;
                // Power that enters the slab heading in, and the power that
                // meets the coating from inside over every pass.
                let (enter, den) = match c.incidence {
                    Incidence::Film => (front.t, 1.0 - back.r * r2 * tau2),
                    Incidence::Substrate => (t2, 1.0 - r2 * back.r * tau2),
                };
                let den = if den.abs() < 1e-12 { 1e-12 } else { den };
                let from_inside = match c.incidence {
                    Incidence::Film => enter * tau2 * r2 / den,
                    Incidence::Substrate => enter * tau / den,
                };
                a[idx] = (0..n)
                    .map(|k| {
                        let first = if matches!(c.incidence, Incidence::Film) { a_front[k] } else { 0.0 };
                        first + from_inside * a_back[k]
                    })
                    .collect();
            }
            // The slab takes what the faces and the layers do not. Its
            // series, (1 − τ) over every pass, is the same number up to the
            // plate's own convention for a lossy medium on the inner side of
            // its interfaces (about 1e-8 for k = 2e-6), and the remainder is
            // the share that makes the books close exactly.
            for (idx, (r, t)) in [(rt.r_s, rt.t_s), (rt.r_p, rt.t_p)].into_iter().enumerate() {
                sub[idx] = (1.0 - r - t - a[idx].iter().sum::<f64>()).max(0.0);
            }
            let [a_s, a_p] = a;
            FilmPowers { rt, a_layer_s: a_s, a_layer_p: a_p, a_substrate_s: sub[0], a_substrate_p: sub[1] }
        }
    };
    Ok((powers, notes))
}

/// One pass through a plate's slab, by its extinction along the slant path.
fn plate_tau(n_sub: Cx, q: Cx, lambda_um: f64, thickness_um: f64) -> f64 {
    if thickness_um > 0.0 && n_sub.im > 0.0 {
        let cos_t = snell_cos(q, Cx::real(n_sub.re)).re.max(1e-6);
        (-4.0 * std::f64::consts::PI * n_sub.im * thickness_um / (lambda_um * cos_t)).exp()
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Coherent films and incoherent slabs together (CF-05)
// ---------------------------------------------------------------------------

/// One polarization of a stack with slabs: the powers and every element's
/// absorbed fraction, elements in the walked order.
struct MixedPol {
    r: f64,
    t: f64,
    /// Per walked element: films and slabs alike.
    a: Vec<f64>,
    /// What an opaque exit medium keeps; zero for a transparent one.
    a_exit: f64,
}

/// A stack of coherent films and incoherent slabs between `n0` and `n_exit`,
/// for one polarization, by the incoherent transfer-matrix method of Byrnes'
/// `inc_tmm` (Katsidis and Siapkas, Appl. Opt. 41, 3978 (2002)).
///
/// The films between two incoherent media form one coherent group, met from
/// either side with its own R and T ([`stack_rt_pol`]); each slab passes `P`
/// of the power on one crossing, `exp(-4π d |Im(n cos θ)| / λ)`. Intensities
/// chain through `L = diag(1/P, P) · [[1, −R_b], [R_f, T_b T_f − R_b R_f]] / T_f`,
/// and the forward and backward intensities at each slab say what each
/// element absorbs: a slab `V(1 − P) + W(1/P − 1)`, a film its share from
/// each side weighted by what arrives from that side.
fn mixed_pol(walked: &[(Cx, f64, bool)], n0: Cx, n_exit: Cx, exit_absorbs: bool, lambda_um: f64, q: Cx, p_pol: bool) -> MixedPol {
    // Incoherent media, and the coherent groups between them, as walked
    // indices: groups[i] lies between media[i] and media[i + 1].
    let mut media: Vec<(Cx, f64, Option<usize>)> = vec![(n0, 0.0, None)];
    let mut groups: Vec<Vec<usize>> = vec![Vec::new()];
    for (i, &(n, d, slab)) in walked.iter().enumerate() {
        if slab {
            media.push((n, d, Some(i)));
            groups.push(Vec::new());
        } else {
            groups.last_mut().expect("one group per gap").push(i);
        }
    }
    media.push((n_exit, 0.0, None));
    let m = media.len();
    let film = |g: &[usize]| -> Vec<(Cx, f64)> { g.iter().map(|&i| (walked[i].0, walked[i].1)).collect() };

    // Each group from the front and from the back: R, T, the power that
    // crosses its face, and its layers' absorbed fractions.
    struct Face {
        rf: f64,
        tf: f64,
        rb: f64,
        tb: f64,
        pef: f64,
        peb: f64,
        af: Vec<f64>,
        ab: Vec<f64>,
    }
    let faces: Vec<Face> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let (na, nb) = (media[i].0, media[i + 1].0);
            let fwd = film(g);
            let rev: Vec<(Cx, f64)> = fwd.iter().rev().copied().collect();
            let f = stack_rt_pol(&fwd, na, nb, lambda_um, q, p_pol);
            let b = stack_rt_pol(&rev, nb, na, lambda_um, q, p_pol);
            let fa = stack_amp_pol(&fwd, na, nb, lambda_um, q, p_pol);
            let ba = stack_amp_pol(&rev, nb, na, lambda_um, q, p_pol);
            let (pef, peb) = (power_entering(fa.r, fa.eta0), power_entering(ba.r, ba.eta0));
            let (af, _) = absorbed_by_layer_from(&fwd, nb, lambda_um, q, p_pol, pef);
            let (ab_rev, _) = absorbed_by_layer_from(&rev, na, lambda_um, q, p_pol, peb);
            Face { rf: f.r, tf: f.t, rb: b.r, tb: b.t, pef, peb, af, ab: ab_rev.into_iter().rev().collect() }
        })
        .collect();
    // One crossing of each slab.
    let pass: Vec<f64> = media
        .iter()
        .map(|&(n, d, slab)| match slab {
            Some(_) => {
                let nc = n * snell_cos(q, n);
                (-4.0 * std::f64::consts::PI * (d / 1000.0) * nc.im.abs() / lambda_um).exp().max(1e-30)
            }
            None => 1.0,
        })
        .collect();

    let mut a = vec![0.0; walked.len()];
    // A group that passes nothing ends the light's journey there.
    if let Some(blocked) = faces.iter().position(|f| f.tf <= 0.0) {
        if blocked == 0 {
            for (k, &i) in groups[0].iter().enumerate() {
                a[i] = faces[0].af[k];
            }
            let a_exit = if exit_absorbs && m == 2 { 1.0 - faces[0].rf - a.iter().sum::<f64>() } else { 0.0 };
            return MixedPol { r: faces[0].rf, t: 0.0, a, a_exit: a_exit.max(0.0) };
        }
    }
    let l_of = |f: &Face| -> [[f64; 2]; 2] {
        if f.tf <= 0.0 {
            return [[1e300, 0.0], [0.0, 0.0]];
        }
        [[1.0 / f.tf, -f.rb / f.tf], [f.rf / f.tf, (f.tb * f.tf - f.rb * f.rf) / f.tf]]
    };
    let mul = |x: [[f64; 2]; 2], y: [[f64; 2]; 2]| -> [[f64; 2]; 2] {
        [
            [x[0][0] * y[0][0] + x[0][1] * y[1][0], x[0][0] * y[0][1] + x[0][1] * y[1][1]],
            [x[1][0] * y[0][0] + x[1][1] * y[1][0], x[1][0] * y[0][1] + x[1][1] * y[1][1]],
        ]
    };
    let mut ls = vec![[[0.0; 2]; 2]; m - 1];
    let mut total = l_of(&faces[0]);
    for i in 1..m - 1 {
        let p = pass[i];
        let li = mul([[1.0 / p, 0.0], [0.0, p]], l_of(&faces[i]));
        ls[i] = li;
        total = mul(total, li);
    }
    let t = 1.0 / total[0][0];
    let r = total[1][0] / total[0][0];
    // Forward and backward intensity at the start of each incoherent medium.
    let mut vw = vec![[0.0f64; 2]; m];
    vw[m - 1] = [t, 0.0];
    for i in (1..m - 1).rev() {
        let next = vw[i + 1];
        vw[i] = [ls[i][0][0] * next[0] + ls[i][0][1] * next[1], ls[i][1][0] * next[0] + ls[i][1][1] * next[1]];
    }
    // What arrives at group j: forward from the medium before it, backward
    // from the medium after it.
    let forward = |j: usize| if j == 0 { 1.0 } else { vw[j][0] * pass[j] };
    let backward = |j: usize| vw[j + 1][1];
    // A slab keeps what crosses into it less what crosses out of it, each the
    // net Poynting flux at the face, as `tmm.inc_absorp_in_each_layer` books it.
    for i in 1..m - 1 {
        if let Some(idx) = media[i].2 {
            let entering = forward(i - 1) * faces[i - 1].tf - backward(i - 1) * faces[i - 1].peb;
            let leaving = forward(i) * faces[i].pef - backward(i) * faces[i].tb;
            a[idx] = entering - leaving;
        }
    }
    for (j, g) in groups.iter().enumerate() {
        let (forward, backward) = (forward(j), backward(j));
        for (k, &i) in g.iter().enumerate() {
            a[i] = forward * faces[j].af[k] + backward * faces[j].ab[k];
        }
    }
    let (t, a_exit) = if exit_absorbs { (0.0, t) } else { (t, 0.0) };
    MixedPol { r, t, a, a_exit }
}

/// The net power that crosses a stack's front face, as a fraction of the
/// incident power, from Macleod's `r` and the incident admittance:
/// `Re((1 + r)(1 − r̄) η̄₀) / Re(η₀)`. From a transparent medium that is
/// exactly `1 − |r|²`; from a lossy one (a slab of absorbing glass) the
/// incident and reflected waves overlap in a medium that absorbs, and the
/// two differ by a term of the order of its `k`. It is `tmm`'s
/// `power_entering`.
fn power_entering(r: Cx, eta0: Cx) -> f64 {
    let one = Cx::real(1.0);
    let num = ((one + r) * (one - r.conj()) * eta0.conj()).re;
    if eta0.re.abs() > 0.0 {
        num / eta0.re
    } else {
        0.0
    }
}

/// Does the coating hold a slab?
pub fn has_slab(c: &Coating) -> bool {
    c.layers.iter().any(|l| l.incoherent)
}

/// Is the coating evaluated by the incoherent transfer-matrix method: it
/// holds slabs, or it is a plate under `plate_model: Incoherent`?
fn uses_mixed(c: &Coating, ctx: &FilmContext) -> bool {
    has_slab(c)
        || (matches!(c.backing, CoatingBacking::Plate { .. }) && ctx.plate == crate::fidelity::PlateModel::Incoherent)
}

/// The Mueller matrices of a coating with one incoherent slab, reflected and
/// transmitted, in the ray frame's `(ŝ, p̂)` basis (spec CF-05): the two
/// faces' amplitudes as Mueller matrices `M(x)` of their diagonal Jones
/// elements, added incoherently over every pass of the slab,
///
/// ```text
/// M_R = M(r_E) + M(t_E') M(r_O) (I − M(r_E') M(r_O) P²)⁻¹ M(t_E) P²
/// M_T = M(t_O) (I − M(r_E') M(r_O) P²)⁻¹ M(t_E) P
/// ```
///
/// with `E` the face the light enters by (seen from outside, and `'` from
/// inside), `O` the other face from inside, and `P` the power one crossing of
/// the slab keeps. The matrices of diagonal Jones elements in one basis
/// commute, so the order in the loop is immaterial. The inverse exists
/// because every eigenvalue of `M(r_E') M(r_O) P²` is below 1 for a passive
/// slab; that is checked, and a plate that fails it is refused.
///
/// A plate's anti-reflected far face (a residual factor on the bare Fresnel
/// reflectance) scales the bare face's reflection matrix by the residual and
/// hands what it suppresses to its transmission matrix, unpolarized: the
/// residual is a single number and says nothing about polarization.
pub fn coating_mueller_eval(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
    ctx: &FilmContext,
) -> Result<(crate::polarization::Mueller, crate::polarization::Mueller), FilmError> {
    use crate::polarization::Mueller;
    let (n0, cos_i, q) = incidence(n_incident, cos_i);
    let mut notes = FilmNotes::default();
    let sampled = sample_layers(c, lambda_um, ctx, &mut notes)?;
    let mut elements: Vec<(Cx, f64, bool)> =
        sampled.iter().zip(&c.layers).map(|(&(nl, d), l)| (nl, d, l.incoherent)).collect();
    let n_far = Cx::real(n_far.max(1e-9));
    let mut residual = 1.0;
    match &c.backing {
        CoatingBacking::Plate { substrate, exit, thickness_mm } => {
            if has_slab(c) {
                return Err(FilmError::SlabOnPlate);
            }
            let n_sub = sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut notes)?;
            elements.push((n_sub, thickness_mm.max(0.0) * 1e6, true));
            if let ExitFace::AntiReflective { residual: r } = exit {
                residual = r.clamp(0.0, 1.0);
            }
        }
        CoatingBacking::Medium => {}
        CoatingBacking::Opaque { .. } => return Err(FilmError::Incoherent),
    }
    let slabs: Vec<usize> = elements.iter().enumerate().filter(|(_, e)| e.2).map(|(i, _)| i).collect();
    if slabs.len() != 1 {
        return Err(FilmError::Incoherent);
    }
    let at = slabs[0];
    let (n_sub, d_sub) = (elements[at].0, elements[at].1);
    let front: Vec<(Cx, f64)> = elements[..at].iter().map(|e| (e.0, e.1)).collect();
    let rear: Vec<(Cx, f64)> = elements[at + 1..].iter().map(|e| (e.0, e.1)).collect();
    let rev = |v: &[(Cx, f64)]| -> Vec<(Cx, f64)> { v.iter().rev().copied().collect() };
    // Entering face E and other face O, as stacks with their media.
    let substrate_side = matches!(c.incidence, Incidence::Substrate);
    let (e_out, e_in, o_in) = if substrate_side {
        (
            stack_amp(&rev(&rear), n_far, n_sub, lambda_um, q).0,
            stack_amp(&rear, n_sub, n_far, lambda_um, q).0,
            stack_amp(&rev(&front), n_sub, n0, lambda_um, q).0,
        )
    } else {
        (
            stack_amp(&front, n0, n_sub, lambda_um, q).0,
            stack_amp(&rev(&front), n_sub, n0, lambda_um, q).0,
            stack_amp(&rear, n_sub, n_far, lambda_um, q).0,
        )
    };
    let m_r = |a: &FilmAmp| Mueller::from_diagonal_jones(a.r_s, a.r_p);
    let m_t = |a: &FilmAmp| Mueller::from_diagonal_jones(a.t_s, a.t_p).scaled(a.t_power_scale);
    // The plate's far face, anti-reflected by a residual factor.
    let far_face = |a: &FilmAmp, reflect: bool| -> Mueller {
        if residual == 1.0 || !matches!(c.backing, CoatingBacking::Plate { .. }) {
            return if reflect { m_r(a) } else { m_t(a) };
        }
        if reflect {
            m_r(a).scaled(residual)
        } else {
            let p = a.powers();
            let gain = |r: f64, t: f64| if t > 0.0 { (t + (1.0 - residual) * r) / t } else { 1.0 };
            let g = 0.5 * (gain(p.r_s, p.t_s) + gain(p.r_p, p.t_p));
            m_t(a).scaled(g)
        }
    };
    let (r_e, t_e, r_e_in, t_e_in, r_o, t_o) = if substrate_side {
        (far_face(&e_out, true), far_face(&e_out, false), far_face(&e_in, true), far_face(&e_in, false), m_r(&o_in), m_t(&o_in))
    } else {
        (m_r(&e_out), m_t(&e_out), m_r(&e_in), m_t(&e_in), far_face(&o_in, true), far_face(&o_in, false))
    };
    let nc = n_sub * snell_cos(q, n_sub);
    let p = (-4.0 * std::f64::consts::PI * (d_sub / 1000.0) * nc.im.abs() / lambda_um).exp();
    if cos_i < GRAZING_COS {
        return Ok((Mueller::IDENTITY, Mueller::attenuator(0.0)));
    }
    let loop_m = r_e_in.times(&r_o).scaled(p * p);
    if loop_m.spectral_bound() >= 1.0 {
        return Err(FilmError::Degenerate);
    }
    let inv = Mueller::IDENTITY.minus(&loop_m).inverse().ok_or(FilmError::Degenerate)?;
    let reflected = r_e.plus(&t_e_in.times(&r_o).times(&inv).times(&t_e).scaled(p * p));
    let transmitted = t_o.times(&inv).times(&t_e).scaled(p);
    Ok((reflected, transmitted))
}

/// [`coating_power_flow_eval`] for a coating with slabs, or a plate under
/// `plate_model: Incoherent` written as its films, its slab and its far face.
fn mixed_eval(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
    ctx: &FilmContext,
) -> Result<(FilmPowers, FilmNotes), FilmError> {
    let (n0, cos_i, q) = incidence(n_incident, cos_i);
    let mut notes = FilmNotes::default();
    if matches!(c.backing, CoatingBacking::Opaque { .. }) && matches!(c.incidence, Incidence::Substrate) {
        return Err(FilmError::AbsorbingIncidentMedium);
    }
    let sampled = sample_layers(c, lambda_um, ctx, &mut notes)?;
    let n = c.layers.len();
    // Every element in the order written, with its slab flag.
    let mut elements: Vec<(Cx, f64, bool)> =
        sampled.iter().zip(&c.layers).map(|(&(nl, d), l)| (nl, d, l.incoherent)).collect();
    // A plate is its films, its slab, and a far face to the far medium.
    let (n_exit, exit_absorbs, plate_residual) = match &c.backing {
        CoatingBacking::Medium => (Cx::real(n_far.max(1e-9)), false, None),
        CoatingBacking::Opaque { substrate } => (sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut notes)?, true, None),
        CoatingBacking::Plate { substrate, exit, thickness_mm } => {
            if has_slab(c) {
                return Err(FilmError::SlabOnPlate);
            }
            let n_sub = sample_index(substrate, lambda_um, ctx, FilmError::SUBSTRATE, &mut notes)?;
            elements.push((n_sub, thickness_mm.max(0.0) * 1e6, true));
            let residual = match exit {
                ExitFace::Bare => 1.0,
                ExitFace::AntiReflective { residual } => residual.clamp(0.0, 1.0),
            };
            (Cx::real(n_far.max(1e-9)), false, Some(residual))
        }
    };
    let total = elements.len();
    // The plate's far face is bare and, when anti-reflected, scaled; a scaled
    // face is not a stack, so it is handled by the two-face series below.
    let walked: Vec<(Cx, f64, bool)> = match (&c.backing, c.incidence) {
        (CoatingBacking::Medium, Incidence::Substrate) => elements.iter().rev().copied().collect(),
        (CoatingBacking::Plate { .. }, Incidence::Substrate) => {
            // Entering through the far face: the slab, then the films reversed.
            let mut w = vec![elements[total - 1]];
            w.extend(elements[..total - 1].iter().rev().copied());
            w
        }
        _ => elements.clone(),
    };
    let zero_powers = || FilmPowers { rt: FilmRt { r_s: 1.0, r_p: 1.0, t_s: 0.0, t_p: 0.0 }, a_layer_s: vec![0.0; n], a_layer_p: vec![0.0; n], a_substrate_s: 0.0, a_substrate_p: 0.0 };
    if cos_i < GRAZING_COS {
        return Ok((zero_powers(), notes));
    }
    let mut out = [MixedPol { r: 0.0, t: 0.0, a: Vec::new(), a_exit: 0.0 }, MixedPol { r: 0.0, t: 0.0, a: Vec::new(), a_exit: 0.0 }];
    for (idx, p_pol) in [false, true].into_iter().enumerate() {
        out[idx] = match plate_residual {
            Some(residual) if residual != 1.0 => plate_with_scaled_face(&walked, n0, n_exit, residual, c.incidence, lambda_um, q, p_pol),
            _ => mixed_pol(&walked, n0, n_exit, exit_absorbs, lambda_um, q, p_pol),
        };
        if !out[idx].r.is_finite() || !out[idx].t.is_finite() {
            return Err(FilmError::Overflow);
        }
    }
    // Back to the written order: the films, and a plate's slab last.
    let unwalk = |a: &[f64]| -> Vec<f64> {
        match (&c.backing, c.incidence) {
            (CoatingBacking::Medium, Incidence::Substrate) => a.iter().rev().copied().collect(),
            (CoatingBacking::Plate { .. }, Incidence::Substrate) => {
                let mut w: Vec<f64> = a[1..].iter().rev().copied().collect();
                w.push(a[0]);
                w
            }
            _ => a.to_vec(),
        }
    };
    let (a_s, a_p) = (unwalk(&out[0].a), unwalk(&out[1].a));
    let plate = plate_residual.is_some();
    let split = |a: Vec<f64>, exit: f64| -> (Vec<f64>, f64) {
        if plate {
            (a[..n].to_vec(), a[n] + exit)
        } else {
            (a, exit)
        }
    };
    let (a_layer_s, a_substrate_s) = split(a_s, out[0].a_exit);
    let (a_layer_p, a_substrate_p) = split(a_p, out[1].a_exit);
    let rt = FilmRt { r_s: out[0].r.clamp(0.0, 1.0), r_p: out[1].r.clamp(0.0, 1.0), t_s: out[0].t.clamp(0.0, 1.0), t_p: out[1].t.clamp(0.0, 1.0) };
    Ok((FilmPowers { rt, a_layer_s, a_layer_p, a_substrate_s, a_substrate_p }, notes))
}

/// A plate whose far face is anti-reflected by a residual factor: the front
/// group, the slab, and a far face whose reflectance is the bare Fresnel
/// value scaled by `residual`, what it suppresses passing on.
#[allow(clippy::too_many_arguments)]
fn plate_with_scaled_face(
    walked: &[(Cx, f64, bool)],
    n0: Cx,
    n_exit: Cx,
    residual: f64,
    incidence: Incidence,
    lambda_um: f64,
    q: Cx,
    p_pol: bool,
) -> MixedPol {
    // The slab is the one incoherent element; the films are the rest.
    let slab_at = walked.iter().position(|e| e.2).expect("a plate has its slab");
    let (n_sub, d_sub) = (walked[slab_at].0, walked[slab_at].1);
    let nc = n_sub * snell_cos(q, n_sub);
    let p = (-4.0 * std::f64::consts::PI * (d_sub / 1000.0) * nc.im.abs() / lambda_um).exp().max(1e-30);
    let films: Vec<(Cx, f64)> = walked.iter().filter(|e| !e.2).map(|e| (e.0, e.1)).collect();
    let rev: Vec<(Cx, f64)> = films.iter().rev().copied().collect();
    let bare = |from: Cx, to: Cx| {
        let s = stack_rt_pol(&[], from, to, lambda_um, q, p_pol);
        let r = (s.r * residual).clamp(0.0, 1.0);
        (r, s.t + (s.r - r))
    };
    let mut a = vec![0.0; walked.len()];
    let (r, t) = match incidence {
        Incidence::Film => {
            // films (from n0 into the slab), slab, far face (slab to exit).
            let f = stack_rt_pol(&films, n0, n_sub, lambda_um, q, p_pol);
            let b = stack_rt_pol(&rev, n_sub, n0, lambda_um, q, p_pol);
            let (af, _) = absorbed_by_layer(&films, n_sub, lambda_um, q, p_pol, f.r);
            let (ab, _) = absorbed_by_layer(&rev, n0, lambda_um, q, p_pol, b.r);
            let (r2, t2) = bare(n_sub, n_exit);
            let (r2b, _) = bare(n_sub, n_exit);
            let den = 1.0 - b.r * r2b * p * p;
            let forward = f.t / den;
            let backward = forward * p * r2b;
            let at_films_from_inside = backward * p;
            for k in 0..films.len() {
                a[k] = af[k] + at_films_from_inside * ab[films.len() - 1 - k];
            }
            a[slab_at] = forward * (1.0 - p) + backward * (1.0 - p);
            (f.r + f.t * b.t * r2 * p * p / den, f.t * t2 * p / den)
        }
        Incidence::Substrate => {
            // far face (exit side into the slab), slab, films reversed out to n0.
            let (r2, t2) = bare(n_exit, n_sub);
            let (r2b, t2b) = bare(n_sub, n_exit);
            let f = stack_rt_pol(&films, n_sub, n0, lambda_um, q, p_pol);
            let (af, _) = absorbed_by_layer(&films, n0, lambda_um, q, p_pol, f.r);
            let den = 1.0 - f.r * r2b * p * p;
            let forward = t2 / den;
            let backward = forward * p * f.r;
            for k in 0..films.len() {
                a[slab_at + 1 + k] = forward * p * af[k];
            }
            a[slab_at] = forward * (1.0 - p) + backward * (1.0 - p);
            (r2 + t2 * t2b * f.r * p * p / den, t2 * f.t * p / den)
        }
    };
    MixedPol { r, t, a, a_exit: 0.0 }
}

// ---------------------------------------------------------------------------
// Synthesis: turning a design into layers
// ---------------------------------------------------------------------------

/// The standard film materials, enough to build every preset here: the
/// **nominal** set, constants that stay exactly as they were so every older
/// result stays bit for bit. Measured materials with their sources are in
/// [`materials::catalog`].
///
/// Indices are the usual room-temperature values near the middle of the visible
/// band, held constant across it. Real films disperse by a percent or two and
/// real deposition runs miss their targets by more than that, so a constant is
/// an honest starting point; the layer list is editable and `n` is a tabulated
/// [`SpectralScalar`] underneath, so a measured dispersion can replace any of
/// these without touching the calculation.
pub mod materials {
    use super::FilmMaterial;

    /// The film catalogue's measured entries ([`crate::film_catalog`]): real
    /// n(λ) and k(λ) with their sources. The constants beside this module are
    /// the nominal set; these are what design synthesis and the editor use.
    pub mod catalog {
        use crate::thinfilm::FilmMaterial;

        fn entry(key: &str) -> FilmMaterial {
            crate::film_catalog::builtin_named(key).expect("a built-in catalogue entry")
        }

        pub fn magnesium_fluoride() -> FilmMaterial {
            entry("magnesium_fluoride")
        }
        /// The bulk crystal's ordinary index: what a hot-deposited AR layer
        /// of MgF2 sits near, and the AR designs' default low-index film.
        pub fn magnesium_fluoride_bulk() -> FilmMaterial {
            entry("magnesium_fluoride_bulk")
        }
        pub fn silica() -> FilmMaterial {
            entry("silica")
        }
        pub fn alumina() -> FilmMaterial {
            entry("alumina")
        }
        pub fn titania() -> FilmMaterial {
            entry("titania")
        }
        pub fn tantala() -> FilmMaterial {
            entry("tantala")
        }
        pub fn hafnia() -> FilmMaterial {
            entry("hafnia")
        }
        pub fn zirconia() -> FilmMaterial {
            entry("zirconia")
        }
        pub fn niobia() -> FilmMaterial {
            entry("niobia")
        }
        pub fn zinc_sulfide() -> FilmMaterial {
            entry("zinc_sulfide")
        }
        pub fn silicon() -> FilmMaterial {
            entry("silicon")
        }
        pub fn silver() -> FilmMaterial {
            entry("silver")
        }
        pub fn silver_johnson() -> FilmMaterial {
            entry("silver_johnson")
        }
        pub fn aluminium() -> FilmMaterial {
            entry("aluminium")
        }
        pub fn gold() -> FilmMaterial {
            entry("gold")
        }
        pub fn chromium() -> FilmMaterial {
            entry("chromium")
        }
        pub fn nickel() -> FilmMaterial {
            entry("nickel")
        }
    }

    pub fn titania() -> FilmMaterial {
        FilmMaterial::constant("TiO₂", 2.35)
    }
    pub fn tantala() -> FilmMaterial {
        FilmMaterial::constant("Ta₂O₅", 2.10)
    }
    pub fn silica() -> FilmMaterial {
        FilmMaterial::constant("SiO₂", 1.46)
    }
    pub fn magnesium_fluoride() -> FilmMaterial {
        FilmMaterial::constant("MgF₂", 1.38)
    }
    pub fn alumina() -> FilmMaterial {
        FilmMaterial::constant("Al₂O₃", 1.65)
    }
    /// Aluminium near 550 nm. Absorbing, so a film thick enough to be opaque is
    /// a mirror rather than a filter.
    pub fn aluminium() -> FilmMaterial {
        FilmMaterial::absorbing("Al", 0.958, 6.69)
    }
    /// N-BK7, the default plate substrate.
    pub fn bk7() -> FilmMaterial {
        FilmMaterial::constant("N-BK7", 1.5168)
    }

    /// Silver, measured: Babar and Weaver, Appl. Opt. 54, 477 (2015), 0.21 to
    /// 3.54 µm, from the refractiveindex.info database (CC0).
    ///
    /// The constant [`aluminium`] above is good for one wavelength. A metal
    /// that has to reflect sunlight has to be right from the UV to the near
    /// infrared, and silver is not flat across that: it goes transparent near
    /// 320 nm, which a constant index cannot know.
    ///
    /// Clamped outside the table. Past 3.5 µm a metal only gets more
    /// reflective, so holding the last value slightly understates it, and the
    /// sun has less than half a percent of its direct power out there.
    pub fn silver() -> FilmMaterial {
        tabulated("Ag", &super::SILVER_BABAR_WEAVER)
    }

    /// Aluminium, measured: Rakić, Appl. Opt. 34, 4755 (1995), 0.21 to 3.9 µm,
    /// from the refractiveindex.info database (CC0). It carries the interband
    /// dip near 800 nm, where aluminium falls to about 87%. Clamped outside the
    /// table, as [`silver`] is.
    pub fn aluminium_measured() -> FilmMaterial {
        tabulated("Al", &super::ALUMINIUM_RAKIC)
    }

    fn tabulated(name: &str, rows: &[[f64; 3]]) -> FilmMaterial {
        use crate::scene3::{Interpolation, OutOfRangePolicy, SpectralScalar};
        let lambda_um: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let column = |i: usize| SpectralScalar::Tabulated {
            lambda_um: lambda_um.clone(),
            values: rows.iter().map(|r| r[i]).collect(),
            interpolation: Interpolation::Linear,
            out_of_range: OutOfRangePolicy::Clamp,
        };
        FilmMaterial::scalars(name, column(1), column(2))
    }
}

/// A metal mirror: an opaque metal film under an optional protective
/// overcoat, `(material, thickness_nm)`, the way a protected silver or
/// aluminium mirror is sold.
///
/// The film is 150 nm, which is what a coater lays down: opaque across the
/// visible and infrared, but not in silver's window near 320 nm, where a few
/// percent does get through a real one. What gets through is carried on into
/// whatever the film is backed by, as it would be.
///
/// No design recipe is attached, so what the stack reflects is the path.
pub fn metal_mirror(metal: FilmMaterial, overcoat: Option<(FilmMaterial, f64)>) -> Coating {
    let mut layers = Vec::new();
    if let Some((m, nm)) = overcoat {
        layers.push(FilmLayer::new(m, nm));
    }
    layers.push(FilmLayer::new(metal, 150.0));
    Coating {
        layers,
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: None,
    }
}

/// `[λ µm, n, k]`, Babar and Weaver (2015). See [`materials::silver`].
#[rustfmt::skip]
const SILVER_BABAR_WEAVER: [[f64; 3]; 64] = [
    [0.2066, 1.079, 1.247], [0.2101, 1.101, 1.258], [0.2138, 1.121, 1.267],
    [0.2175, 1.14, 1.272], [0.2214, 1.157, 1.28], [0.2254, 1.169, 1.283],
    [0.2296, 1.181, 1.287], [0.2339, 1.19, 1.294], [0.2384, 1.198, 1.306],
    [0.2431, 1.215, 1.317], [0.248, 1.233, 1.326], [0.253, 1.254, 1.328],
    [0.2583, 1.27, 1.327], [0.2638, 1.294, 1.329], [0.2695, 1.321, 1.31],
    [0.2755, 1.348, 1.291], [0.2818, 1.38, 1.247], [0.2883, 1.402, 1.181],
    [0.2952, 1.427, 1.079], [0.3024, 1.41, 0.894], [0.31, 1.265, 0.656],
    [0.3179, 0.859, 0.466], [0.3263, 0.307, 0.651], [0.3351, 0.187, 1.042],
    [0.3444, 0.142, 1.269], [0.3542, 0.106, 1.467], [0.3647, 0.076, 1.645],
    [0.3757, 0.061, 1.815], [0.3875, 0.05, 1.981], [0.3999, 0.054, 2.138],
    [0.4133, 0.05, 2.292], [0.4275, 0.051, 2.448], [0.4428, 0.052, 2.604],
    [0.4592, 0.052, 2.765], [0.4769, 0.053, 2.93], [0.4959, 0.052, 3.105],
    [0.5166, 0.052, 3.288], [0.5391, 0.05168, 3.483], [0.5636, 0.05009, 3.694],
    [0.5904, 0.04977, 3.918], [0.6199, 0.04803, 4.164], [0.6525, 0.04964, 4.432],
    [0.6888, 0.0508, 4.725], [0.7293, 0.05151, 5.048], [0.7749, 0.05269, 5.409],
    [0.8266, 0.05504, 5.814], [0.8856, 0.05896, 6.276], [0.9537, 0.06542, 6.802],
    [1.033, 0.07286, 7.412], [1.127, 0.08427, 8.128], [1.24, 0.09799, 8.981],
    [1.305, 0.1056, 9.472], [1.378, 0.1143, 10.018], [1.459, 0.1256, 10.63],
    [1.55, 0.1388, 11.31], [1.653, 0.1543, 12.08], [1.771, 0.1735, 12.97],
    [1.907, 0.1988, 13.98], [2.066, 0.2315, 15.16], [2.254, 0.2724, 16.56],
    [2.48, 0.3256, 18.23], [2.755, 0.397, 20.26], [3.1, 0.496, 22.8],
    [3.542, 0.6387, 26.05],
];

/// `[λ µm, n, k]`, Rakić (1995). See [`materials::aluminium_measured`].
#[rustfmt::skip]
const ALUMINIUM_RAKIC: [[f64; 3]; 37] = [
    [0.20664, 0.12677, 2.3563], [0.24797, 0.18137, 2.9029], [0.30996, 0.28003, 3.7081],
    [0.32628, 0.31474, 3.9165], [0.36466, 0.39877, 4.3957], [0.41328, 0.52135, 5.0008],
    [0.4428, 0.6079, 5.3676], [0.47687, 0.7278, 5.7781], [0.5166, 0.8734, 6.2418],
    [0.56357, 1.0728, 6.7839], [0.61993, 1.366, 7.4052], [0.65225, 1.5724, 7.7354],
    [0.68881, 1.8301, 8.0601], [0.72932, 2.1606, 8.3565], [0.77491, 2.6154, 8.4914],
    [0.79478, 2.7675, 8.3866], [0.81569, 2.7668, 8.2573], [0.83774, 2.6945, 8.1878],
    [0.88561, 2.2802, 8.1134], [0.91166, 1.9739, 8.3058], [0.93928, 1.6784, 8.597],
    [0.96863, 1.4867, 9.0655], [0.99988, 1.4359, 9.4939], [1.0332, 1.3998, 9.8914],
    [1.1271, 1.3281, 10.969], [1.2399, 1.3157, 12.245], [1.3776, 1.3899, 13.784],
    [1.5498, 1.5782, 15.656], [1.7712, 1.9205, 17.991], [2.0664, 2.4738, 20.982],
    [2.4797, 3.3372, 25.004], [2.7552, 3.938, 27.58], [3.0996, 4.7097, 30.737],
    [3.2628, 5.0735, 32.183], [3.444, 5.4903, 33.814], [3.6466, 5.9564, 35.608],
    [3.8745, 6.4808, 37.595],
];

impl CoatingDesign {
    /// A long-pass dichroic: short wavelengths reflect, long ones pass, with the
    /// 50% crossing at `edge_nm` **when used at `aoi_deg`**.
    pub fn long_pass(edge_nm: f64, aoi_deg: f64) -> Self {
        Self {
            kind: DesignKind::LongPass,
            lambda_nm: edge_nm,
            aoi_deg,
            periods: 8,
            n_high: 2.35,
            n_low: 1.46,
            adapt: None,
        }
    }

    /// A short-pass dichroic: long wavelengths reflect, short ones pass.
    pub fn short_pass(edge_nm: f64, aoi_deg: f64) -> Self {
        Self {
            kind: DesignKind::ShortPass,
            ..Self::long_pass(edge_nm, aoi_deg)
        }
    }

    /// Values brought into range, or `None` if one is not finite.
    pub fn sanitized(self) -> Option<Self> {
        if ![self.lambda_nm, self.aoi_deg, self.n_high, self.n_low]
            .iter()
            .all(|v| v.is_finite())
        {
            return None;
        }
        Some(Self {
            lambda_nm: self.lambda_nm.clamp(100.0, 20_000.0),
            aoi_deg: self.aoi_deg.clamp(0.0, 89.0),
            periods: self.periods.clamp(1, 40),
            // A pair needs a contrast to reflect at all, and an index below 1
            // is not a coating material.
            n_high: self.n_high.clamp(1.05, 5.0),
            n_low: self.n_low.clamp(1.01, 5.0),
            ..self
        })
    }

    fn high(&self) -> FilmMaterial {
        FilmMaterial::constant("H", self.n_high)
    }
    fn low(&self) -> FilmMaterial {
        FilmMaterial::constant("L", self.n_low)
    }
}

/// The index of the glass a polarizing beam splitter's stack has to be
/// cemented in, for a layer pair and a 45 degree cube.
///
/// A polarizing stack works by sitting at the Brewster angle **between its own
/// layers**: there the p reflectance of every interface in the stack is
/// exactly zero, so p passes and only s is reflected. Brewster fixes the angle
/// inside the high-index layer at `tan θ_H = n_L / n_H`, and Snell carries
/// that back out through the stack to the medium it is cemented in. At 45
/// degrees, which is what a cube is,
///
/// ```text
///   n₀ = √2 · n_H · n_L / √(n_H² + n_L²)
/// ```
///
/// For titania and silica that is 1.754. This is why a polarizing cube is made
/// of dense flint rather than crown: the glass is not a free choice, it is
/// what the coating pair demands, and getting it wrong is the difference
/// between a beam splitter that polarizes and one that does not.
pub fn macneille_glass(n_high: f64, n_low: f64) -> f64 {
    let (h, l) = (n_high.abs(), n_low.abs());
    std::f64::consts::SQRT_2 * h * l / (h * h + l * l).sqrt()
}

/// `(HL)^N H`, the quarter-wave mirror, tilted for `aoi_deg` in `n_incident`.
fn quarter_wave_pairs(
    d: &CoatingDesign,
    lambda0_nm: f64,
    n_incident: f64,
    periods: u32,
) -> Vec<FilmLayer> {
    let qw = |m: FilmMaterial| FilmLayer::quarter_wave(m, lambda0_nm, n_incident, d.aoi_deg);
    let mut layers = Vec::with_capacity(periods as usize * 2 + 1);
    for _ in 0..periods {
        layers.push(qw(d.high()));
        layers.push(qw(d.low()));
    }
    layers.push(qw(d.high()));
    layers
}

/// `(0.5H L 0.5H)^N`, the symmetric period, which is the edge filter.
///
/// A plain `(HL)^N H` mirror has a clean stopband and a violent passband: about
/// 20% mean reflectance with 37% peaks just past its edge, which makes "where
/// does it cut" almost meaningless. Splitting the high-index layer across the
/// period boundary (Herpin's equivalent layer) halves that, to about 10% mean
/// and 12% peak, and it is the classical construction for a reason.
///
/// Adjacent half layers merge, so `(0.5H L 0.5H)^N` is stored as
/// `0.5H (L H)^(N-1) L 0.5H`: the same film, fewer rows in the editor.
///
/// Residual passband ripple of about 10% is **real and stays**. Commercial edge
/// filters get under a percent with a computer-optimized matching section of
/// tens of extra layers; this does not attempt that, and the component view
/// plots the ripple rather than drawing the clean edge nobody's coating has.
fn symmetric_period(
    d: &CoatingDesign,
    lambda0_nm: f64,
    n_incident: f64,
    periods: u32,
) -> Vec<FilmLayer> {
    let qw = |m: FilmMaterial, frac: f64| {
        let mut l = FilmLayer::quarter_wave(m, lambda0_nm, n_incident, d.aoi_deg);
        l.thickness_nm *= frac;
        l
    };
    let n = periods.max(1);
    let mut layers = Vec::with_capacity(n as usize * 2 + 1);
    layers.push(qw(d.high(), 0.5));
    for i in 0..n {
        layers.push(qw(d.low(), 1.0));
        if i + 1 < n {
            layers.push(qw(d.high(), 1.0));
        }
    }
    layers.push(qw(d.high(), 0.5));
    layers
}

/// The layers a design asks for.
///
/// For the two edge filters the design wavelength is **solved for** rather than
/// taken from a textbook width formula: the stack is built, its reflectance at
/// the requested edge is measured with the same transfer matrix the tracer will
/// use, and the quarter-wave wavelength is bisected until that reflectance is
/// 50%. So "edge at 550 nm at 45°" means the real stack really does cross 50% at
/// 550 nm at 45°, not that an approximation says it should.
///
/// The transition width is *not* a free parameter, and no amount of wanting
/// makes it one: for a quarter-wave pair it is fixed by the index contrast, and
/// the period count buys depth and steepness. The editor reports the width the
/// design achieves rather than pretending to set it.
pub fn design_layers(design: &CoatingDesign, n_incident: f64, n_substrate: f64) -> Vec<FilmLayer> {
    design_layers_for(design, n_incident, &CoatingBacking::Medium, n_substrate)
}

/// [`design_layers`] against the backing the coating will really have, so the
/// solved edge accounts for a plate's second surface.
pub fn design_layers_for(
    design: &CoatingDesign,
    n_incident: f64,
    backing: &CoatingBacking,
    n_far: f64,
) -> Vec<FilmLayer> {
    let d = match design.sanitized() {
        Some(d) => d,
        None => return Vec::new(),
    };
    match d.kind {
        // A polarizing stack is the same `(HL)^N H`. What makes it polarizing
        // is not the layers but where it is used: at the Brewster angle
        // between them, which the glass it is cemented in has to be chosen to
        // deliver.
        DesignKind::QuarterWaveStack | DesignKind::Polarizing => {
            quarter_wave_pairs(&d, d.lambda_nm, n_incident, d.periods)
        }
        DesignKind::LongPass | DesignKind::ShortPass => {
            let side = BandSide::of(d.kind == DesignKind::LongPass);
            // Where the edge is asked to sit is not where a stack built at that
            // quarter-wave wavelength puts it: the band's edge is offset from
            // its centre by the index contrast. But a quarter-wave stack of
            // non-dispersive materials is **exactly scale-invariant** — its
            // whole curve is a function of λ₀/λ — so building it once, measuring
            // where the edge landed and rescaling by the ratio lands it on the
            // target in a single step. A few repeats absorb any dispersion, and
            // cost three spectral scans rather than the sixty a bisection would.
            let mut lambda0 = d.lambda_nm;
            for _ in 0..4 {
                let layers = symmetric_period(&d, lambda0, n_incident, d.periods);
                let trial = Coating {
                    layers,
                    backing: backing.clone(),
                    incidence: Incidence::Film,
                    design: None,
                };
                // Search only around the fundamental. A quarter-wave stack also
                // reflects at λ₀/3, λ₀/5 and so on, and those harmonics are just
                // as strong: a window wide enough to contain one would find it
                // as the peak and rescale the design onto a harmonic. The
                // fundamental's edges sit well inside ±40% of λ₀.
                let Some(measured) = crossing_nm(
                    &trial,
                    n_incident,
                    n_far,
                    d.aoi_deg,
                    0.5,
                    lambda0 * 0.6,
                    lambda0 * 1.7,
                    side,
                ) else {
                    break;
                };
                if !(measured.is_finite() && measured > 1.0) {
                    break;
                }
                let correction = d.lambda_nm / measured;
                lambda0 *= correction;
                if (correction - 1.0).abs() < 1e-6 {
                    break;
                }
            }
            symmetric_period(&d, lambda0, n_incident, d.periods)
        }
        DesignKind::SingleLayerAr if d.adapt.is_none() => vec![FilmLayer::quarter_wave(
            materials::magnesium_fluoride(),
            d.lambda_nm,
            n_incident,
            d.aoi_deg,
        )],
        // Substrate-adapted designs: the substrate is the far medium, or a
        // plate's own glass. A refused design builds no layers; the editor
        // reports why through `crate::synthesis::synthesize`.
        DesignKind::SingleLayerAr
        | DesignKind::VCoat
        | DesignKind::Bbar
        | DesignKind::BandAr
        | DesignKind::EnhancedMetal { .. } => {
            let n_s = match backing {
                CoatingBacking::Plate { substrate, .. } | CoatingBacking::Opaque { substrate } => {
                    substrate.index_at(d.lambda_nm / 1000.0).map_or(n_far, |c| c.re)
                }
                CoatingBacking::Medium => d.adapt.map(|a| a.substrate_n).filter(|n| n.is_finite() && *n > 0.0).unwrap_or(n_far),
            };
            crate::synthesis::synthesize(&d, n_incident, n_s).unwrap_or_default()
        }
        DesignKind::BroadbandArLegacy => {
            // Quarter-half-quarter, the classic three-layer broadband AR: a
            // half wave of a high index widens the band a single layer cannot,
            // and the middle index does the impedance match the substrate needs.
            let qw =
                |m: FilmMaterial| FilmLayer::quarter_wave(m, d.lambda_nm, n_incident, d.aoi_deg);
            let mut mid = qw(materials::titania());
            mid.thickness_nm *= 2.0;
            vec![
                qw(materials::magnesium_fluoride()),
                mid,
                qw(materials::alumina()),
            ]
        }
    }
}

/// A complete coating from a design, on a plate of `substrate`.
pub fn plate_coating(
    mut design: CoatingDesign,
    substrate: FilmMaterial,
    exit: ExitFace,
    n_incident: f64,
) -> Coating {
    if let Some(a) = design.adapt.as_mut() {
        a.substrate_n = substrate.index_at(design.lambda_nm / 1000.0).map_or(n_incident, |c| c.re);
    }
    let backing = CoatingBacking::Plate { substrate, exit, thickness_mm: 0.0 };
    let c = Coating {
        layers: design_layers_for(&design, n_incident, &backing, n_incident),
        backing,
        incidence: Incidence::Film,
        design: Some(design),
    };
    crate::synthesis::record_design_point(c, n_incident, n_incident)
}

/// Which side of the reflectance band an edge is measured on.
///
/// A reflectance band has two edges and they are different numbers. A long-pass
/// dichroic is sold on its long edge and a short-pass on its short one, so
/// asking for "the" crossing without saying which is how you end up quoting the
/// wrong one: a scan from the blue finds the band's rising edge first, which for
/// a long-pass is not the edge anybody means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandSide {
    /// The crossing below the reflectance peak: where a short-pass cuts.
    Short,
    /// The crossing above the reflectance peak: where a long-pass cuts.
    Long,
}

impl BandSide {
    /// The side a long-pass or short-pass filter is specified on.
    pub fn of(long_pass: bool) -> Self {
        if long_pass {
            Self::Long
        } else {
            Self::Short
        }
    }
}

/// A coating for a face of real glass: the stack, solved so its edge lands where
/// the design asks with `n_substrate` behind it.
///
/// This is the coating a **plate with thickness** carries, and the difference
/// from [`plate_coating`] is the whole point of having thickness. There, the
/// substrate and its far face were folded into one pair of numbers, because a
/// plane has no room for glass. Here the glass is a body the tracer crosses on
/// its own: it refracts at this face, travels, and meets the far face as a
/// second interface. Folding anything in would count it twice.
///
/// A substrate-adapted design (`design.adapt`) is solved for `n_substrate`,
/// which it records along with its R and T at the design point.
pub fn film_coating(mut design: CoatingDesign, n_incident: f64, n_substrate: f64) -> Coating {
    if let Some(a) = design.adapt.as_mut() {
        a.substrate_n = n_substrate;
    }
    let c = Coating {
        layers: design_layers(&design, n_incident, n_substrate),
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: Some(design),
    };
    crate::synthesis::record_design_point(c, n_incident, n_substrate)
}

/// The sideways shift a plane-parallel plate gives a beam crossing it:
/// `t sin(θ − θ′) / cos θ′`, measured perpendicular to the beam.
///
/// Nothing in the kernel uses this. It is the closed form the traced geometry is
/// checked against, which is the only honest way to know that refracting in one
/// face and out the other put the beam where a plate really puts it. A 3 mm
/// plate of N-BK7 at 45° moves a beam almost exactly a millimetre.
pub fn lateral_shift_mm(thickness_mm: f64, n_outside: f64, n_substrate: f64, aoi_deg: f64) -> f64 {
    let theta = aoi_deg.to_radians();
    let sin_t = n_outside * theta.sin() / n_substrate;
    if !(-1.0..=1.0).contains(&sin_t) {
        return 0.0;
    }
    let theta_p = sin_t.asin();
    thickness_mm * (theta - theta_p).sin() / theta_p.cos()
}

/// Where a coating's reflectance actually crosses `level`, walking outward from
/// the reflectance peak on `side`, or `None` if it never does inside
/// `[lo_nm, hi_nm]`.
///
/// This is a *measurement* of the built stack, which is what the editor should
/// show next to the number the user asked for: the two agree when the design can
/// deliver and visibly disagree when it cannot.
#[allow(clippy::too_many_arguments)]
pub fn crossing_nm(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    aoi_deg: f64,
    level: f64,
    lo_nm: f64,
    hi_nm: f64,
    side: BandSide,
) -> Option<f64> {
    let cos_i = aoi_deg.to_radians().cos();
    let steps = 600usize;
    let at = |i: usize| -> Option<(f64, f64)> {
        let nm = lo_nm + (hi_nm - lo_nm) * (i as f64) / (steps as f64);
        let r = coating_rt_at(c, n_incident, n_far, nm / 1000.0, cos_i).ok()?.r();
        Some((nm, r))
    };

    // The peak first: everything below is measured relative to it.
    let (mut peak_i, mut peak_r) = (0usize, f64::NEG_INFINITY);
    for i in 0..=steps {
        let (_, r) = at(i)?;
        if r > peak_r {
            peak_r = r;
            peak_i = i;
        }
    }
    if peak_r < level {
        // The band never gets as high as the level asked for, so it has no such
        // edge. Saying so beats returning a sidelobe.
        return None;
    }

    // Walk outward one sample at a time until the level is crossed.
    let mut prev = at(peak_i)?;
    let mut i = peak_i;
    loop {
        let stepped = match side {
            BandSide::Long => {
                if i >= steps {
                    return None;
                }
                i + 1
            }
            BandSide::Short => {
                if i == 0 {
                    return None;
                }
                i - 1
            }
        };
        let cur = at(stepped)?;
        if (prev.1 - level) * (cur.1 - level) <= 0.0 && prev.1 != cur.1 {
            let f = (level - prev.1) / (cur.1 - prev.1);
            return Some(prev.0 + f * (cur.0 - prev.0));
        }
        prev = cur;
        i = stepped;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_cube_glass_is_not_a_free_choice() {
        // Titania and silica, the usual pair.
        let (nh, nl) = (2.35f64, 1.46f64);
        let right = macneille_glass(nh, nl);
        assert!(
            (right - 1.7538).abs() < 1e-3,
            "a dense flint, not a crown: {right}"
        );

        let stack = |n0: f64| {
            let d = CoatingDesign {
                kind: DesignKind::Polarizing,
                lambda_nm: 550.0,
                aoi_deg: 45.0,
                periods: 12,
                n_high: nh,
                n_low: nl,
                adapt: None,
            };
            let c = film_coating(d, n0, n0);
            coating_rt_at(&c, n0, n0, 0.55, 45f64.to_radians().cos()).unwrap()
        };

        // In the glass the pair demands, the stack is a polarizer: s is
        // reflected entirely and p goes through almost untouched.
        let ok = stack(right);
        assert!(ok.r_s > 0.999_999, "s reflects: {}", ok.r_s);
        assert!(ok.t_p > 0.98, "p passes: {}", ok.t_p);
        assert!(ok.t_s < 1e-9, "and s does not: {:.3e}", ok.t_s);

        // Cement the same stack in ordinary crown and it is a mirror instead:
        // p is reflected too, and nothing useful gets through either way. The
        // glass is part of the design, and this is the whole reason a
        // polarizing cube is made of expensive glass.
        let wrong = stack(1.52);
        assert!(
            wrong.t_p < 0.2,
            "in the wrong glass p stops passing: {}",
            wrong.t_p
        );
    }

    use super::*;

    fn approx(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
    }

    /// [`approx`] for the checks a mutation test also runs: those must report
    /// a failure rather than panic on it.
    fn close(a: f64, b: f64, tol: f64, what: &str) -> Result<(), String> {
        if (a - b).abs() <= tol {
            Ok(())
        } else {
            Err(format!("{what}: {a} vs {b} (tol {tol})"))
        }
    }

    /// A bare interface with nothing on it, for the amplitude tests.
    fn bare() -> Coating {
        Coating {
            layers: vec![],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        }
    }

    /// Which way does time run? The characteristic matrix carries a sign in its
    /// `δ` that decides whether this module works in `exp(-iωt)` or its
    /// conjugate, and every phase downstream — the retardance of a coating, the
    /// handedness of the light a Fresnel rhomb makes — inverts with it. Powers
    /// cannot see it, so no existing test can.
    ///
    /// So: one layer, compared against the Airy sum written out by hand. A round
    /// trip inside the layer is `exp(+2iδ)` under `exp(-iωt)`, because a wave
    /// travelling `+z` is `exp(i(kz − ωt))`.
    #[test]
    fn a_single_layer_matches_the_airy_formula_and_fixes_the_sign() {
        let (n0, n1, n2) = (1.0, 1.72, 1.52);
        let (d_nm, lambda_um) = (137.0, 0.55);
        let c = Coating {
            layers: vec![FilmLayer::new(FilmMaterial::constant("test", n1), d_nm)],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        };

        for deg in [0.0f64, 35.0] {
            let th0 = deg.to_radians();
            let amp = coating_amp_at(&c, n0, n2, lambda_um, th0.cos()).unwrap();

            // Angles inside the layer and the substrate, by Snell.
            let s0 = th0.sin();
            let c1 = (1.0 - (n0 * s0 / n1).powi(2)).sqrt();
            let c2 = (1.0 - (n0 * s0 / n2).powi(2)).sqrt();
            let c0 = th0.cos();
            // Interface amplitudes in the ray frame (the p pair flips sign
            // relative to Macleod, exactly as the stack does).
            let rs = |na: f64, ca: f64, nb: f64, cb: f64| (na * ca - nb * cb) / (na * ca + nb * cb);
            let rp = |na: f64, ca: f64, nb: f64, cb: f64| (nb * ca - na * cb) / (nb * ca + na * cb);
            // One round trip through the layer: 2 · (2π/λ) · n₁ d cos θ₁.
            let delta = std::f64::consts::TAU / lambda_um * n1 * (d_nm / 1000.0) * c1;
            let phase = Cx::expi(2.0 * delta);

            for (r_stack, r1, r2) in [
                (amp.r_s, rs(n0, c0, n1, c1), rs(n1, c1, n2, c2)),
                (amp.r_p, rp(n0, c0, n1, c1), rp(n1, c1, n2, c2)),
            ] {
                let (r1, r2) = (Cx::real(r1), Cx::real(r2));
                let airy = (r1 + r2 * phase) / (Cx::real(1.0) + r1 * r2 * phase);
                approx(r_stack.re, airy.re, 1e-12, &format!("{deg}° Re r"));
                approx(r_stack.im, airy.im, 1e-12, &format!("{deg}° Im r"));
            }
        }
    }

    /// At normal incidence the two polarizations describe the same physical
    /// reflection, so in a frame whose `p̂` follows the reflected ray they must
    /// come out **equal and opposite**. This is the one test that catches the
    /// `r_p` sign, and it catches it in both directions: a missing flip makes
    /// them equal, a doubled one makes them equal again.
    #[test]
    fn at_normal_incidence_the_polarizations_are_equal_and_opposite() {
        check_rp_sign().unwrap();
    }

    fn check_rp_sign() -> Result<(), String> {
        // Bare interface, and a real stack, since the sign lives in the
        // conversion rather than in the layers.
        let stack = film_coating(
            CoatingDesign {
                kind: DesignKind::QuarterWaveStack,
                lambda_nm: 550.0,
                aoi_deg: 0.0,
                periods: 4,
                n_high: 2.35,
                n_low: 1.46,
                adapt: None,
            },
            1.0,
            1.5168,
        );
        for c in [bare(), stack] {
            // A hair off axis, where the frame is still defined but the physics
            // has not had room to separate the two.
            for cos_i in [1.0, (1e-6f64).cos()] {
                let a = coating_amp_at(&c, 1.0, 1.5168, 0.55, cos_i).unwrap();
                close(a.r_p.re, -a.r_s.re, 1e-9, "Re r_p = -Re r_s")?;
                close(a.r_p.im, -a.r_s.im, 1e-9, "Im r_p = -Im r_s")?;
                // Transmission has no basis flip: both rays travel forward.
                close(a.t_p.re, a.t_s.re, 1e-9, "Re t_p = Re t_s")?;
                close(a.t_p.im, a.t_s.im, 1e-9, "Im t_p = Im t_s")?;
            }
        }
        Ok(())
    }

    /// The branch correction in [`decaying_cos`] moves the exit medium's cosine,
    /// so it has to leave every power where it was: past the critical angle a
    /// stack reflects all of it whichever branch you stand on. This is the
    /// regression guard for that, in the one regime where the branch is live.
    #[test]
    fn past_the_critical_angle_nothing_gets_through() {
        let (n1, n2) = (1.5168f64, 1.0f64);
        let crit = (n2 / n1).asin().to_degrees();
        let ar = film_coating(
            CoatingDesign {
                kind: DesignKind::SingleLayerAr,
                lambda_nm: 550.0,
                aoi_deg: 0.0,
                periods: 1,
                n_high: 2.35,
                n_low: 1.38,
                adapt: None,
            },
            n1,
            n2,
        );
        for c in [bare(), ar] {
            for deg in [crit + 0.5, 50.0, 60.0, 80.0] {
                let rt = coating_rt_at(&c, n1, n2, 0.55, deg.to_radians().cos()).unwrap();
                approx(rt.r_s, 1.0, 1e-14, &format!("{deg}° R_s"));
                approx(rt.r_p, 1.0, 1e-14, &format!("{deg}° R_p"));
                approx(rt.t_s, 0.0, 1e-14, &format!("{deg}° T_s"));
                approx(rt.t_p, 0.0, 1e-14, &format!("{deg}° T_p"));
            }
        }
    }

    /// The amplitudes and the powers come from one matrix, so they must agree
    /// bit for bit with the power path that Optiland already validates.
    #[test]
    fn the_amplitudes_and_the_powers_agree() {
        let designs = [
            bare(),
            film_coating(
                CoatingDesign {
                    kind: DesignKind::LongPass,
                    lambda_nm: 550.0,
                    aoi_deg: 45.0,
                    periods: 8,
                    n_high: 2.35,
                    n_low: 1.46,
                    adapt: None,
                },
                1.0,
                1.5168,
            ),
            Coating {
                layers: vec![FilmLayer::new(materials::aluminium(), 12.0)],
                backing: CoatingBacking::Medium,
                incidence: Incidence::Film,
                design: None,
            },
        ];
        for c in &designs {
            for deg in [0.0f64, 22.0, 45.0, 68.0] {
                for nm in [450.0, 532.0, 610.0] {
                    let cos_i = deg.to_radians().cos();
                    let (lam, n0, n2) = (nm / 1000.0, 1.0, 1.5168);
                    let want = coating_rt_at(c, n0, n2, lam, cos_i).unwrap();
                    let got = coating_amp_at(c, n0, n2, lam, cos_i).unwrap().powers();
                    approx(got.r_s, want.r_s, 1e-15, "R_s");
                    approx(got.r_p, want.r_p, 1e-15, "R_p");
                    approx(got.t_s, want.t_s, 1e-15, "T_s");
                    approx(got.t_p, want.t_p, 1e-15, "T_p");
                }
            }
        }
    }

    /// Past the critical angle a bare interface reflects everything and the only
    /// thing left to get right is the phase it puts between s and p. The closed
    /// form is the one a Fresnel rhomb is designed from:
    ///
    /// ```text
    /// tan(Δ/2) = cos θ √(sin²θ − n²) / sin²θ,   n = n₂/n₁
    /// ```
    ///
    /// A rhomb of n = 1.51 hits Δ = 45° twice and turns linear light circular
    /// with no coating at all, which is the check at the end.
    #[test]
    fn total_internal_reflection_puts_the_right_phase_between_s_and_p() {
        check_tir_phase().unwrap();
    }

    fn check_tir_phase() -> Result<(), String> {
        let (n1, n2) = (1.51f64, 1.0f64);
        let n = n2 / n1;
        let crit = n.asin();
        for deg in [45.0f64, 50.0, 54.6231, 60.0, 70.0] {
            let th = deg.to_radians();
            assert!(th > crit, "{deg}° must be past critical");
            let a = coating_amp_at(&bare(), n1, n2, 0.55, th.cos()).unwrap();
            // Nothing gets out.
            close(a.r_s.norm_sqr(), 1.0, 1e-14, "|r_s|")?;
            close(a.r_p.norm_sqr(), 1.0, 1e-14, "|r_p|")?;
            close(a.powers().t_s, 0.0, 1e-14, "T_s")?;

            // Sign as well as size. Under `exp(-iωt)` the evanescent wave must
            // decay, which puts `cos θ_t` on the `+i` branch, and then
            // `δ_s = −2 arctan(β/cos θ)` and `δ_p = −2 arctan(β/(n² cos θ))`
            // with `β = √(sin²θ − n²)`. Since `n < 1` the p shift is the larger
            // of the two negatives, so the retardance is **negative**. Its size
            // is the rhomb formula.
            let want =
                -2.0 * (th.cos() * (th.sin().powi(2) - n * n).sqrt() / th.sin().powi(2)).atan();
            let got = a.reflection_retardance();
            close(got, want, 1e-12, &format!("{deg}° retardance"))?;
        }

        // Two bounces at 54.6231° make a quarter wave, the rhomb's whole point.
        let th = 54.6231f64.to_radians();
        let one = coating_amp_at(&bare(), n1, n2, 0.55, th.cos())
            .unwrap()
            .reflection_retardance();
        close(
            (2.0 * one).abs(),
            std::f64::consts::FRAC_PI_2,
            2e-5,
            "two bounces make a quarter wave",
        )
    }

    /// With no layers at all the machinery must reproduce plain Fresnel, at
    /// every angle and in both polarizations. This is the floor the whole module
    /// stands on: if a bare interface is wrong, nothing above it can be right.
    #[test]
    fn an_empty_stack_is_the_fresnel_interface() {
        let n1 = 1.0;
        let n2 = 1.5168;
        for deg in [0.0f64, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0] {
            let cos_i = deg.to_radians().cos();
            let c = Coating {
                layers: vec![],
                backing: CoatingBacking::Medium,
                incidence: Incidence::Film,
                design: None,
            };
            let got = coating_rt_at(&c, n1, n2, 0.5876, cos_i).unwrap();

            // Textbook Fresnel, written out independently of the admittances.
            let sin_t = n1 * deg.to_radians().sin() / n2;
            let cos_t = (1.0 - sin_t * sin_t).sqrt();
            let rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)).powi(2);
            let rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)).powi(2);
            approx(got.r_s, rs, 1e-13, "Rs");
            approx(got.r_p, rp, 1e-13, "Rp");
            approx(got.t_s, 1.0 - rs, 1e-13, "Ts");
            approx(got.t_p, 1.0 - rp, 1e-13, "Tp");
        }
    }

    /// Brewster: p reflects nothing off a bare dielectric interface.
    #[test]
    fn p_polarization_vanishes_at_brewster() {
        let (n1, n2) = (1.0f64, 1.5168f64);
        let brewster = (n2 / n1).atan();
        let c = Coating {
            layers: vec![],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        };
        let got = coating_rt_at(&c, n1, n2, 0.5876, brewster.cos()).unwrap();
        approx(got.r_p, 0.0, 1e-15, "Rp at Brewster");
        assert!(
            got.r_s > 0.07,
            "Rs should be substantial there: {}",
            got.r_s
        );
    }

    /// A single quarter wave of index `n1` on a substrate `n2` reflects
    /// `((n0 n2 − n1²)/(n0 n2 + n1²))²` at the design wavelength. The classic
    /// closed form, and the reason MgF2 on crown glass is the standard cheap AR.
    #[test]
    fn a_quarter_wave_layer_hits_its_closed_form() {
        let (n0, n1, n2) = (1.0, 1.38, 1.52);
        let lambda_nm = 550.0;
        let c = Coating {
            layers: vec![FilmLayer::quarter_wave(
                FilmMaterial::constant("MgF2", n1),
                lambda_nm,
                n0,
                0.0,
            )],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        };
        let got = coating_rt_at(&c, n0, n2, lambda_nm / 1000.0, 1.0).unwrap();
        let want = ((n0 * n2 - n1 * n1) / (n0 * n2 + n1 * n1)).powi(2);
        approx(got.r_s, want, 1e-12, "R of a quarter wave");
        // And it is an improvement on the bare surface, which is the point.
        let bare = ((n0 - n2) / (n0 + n2)).powi(2);
        assert!(got.r_s < bare * 0.5, "AR should more than halve R");
    }

    /// A half-wave layer is absentee: at the wavelength where its phase
    /// thickness is π it does nothing at all, whatever its index.
    #[test]
    fn a_half_wave_layer_is_absentee() {
        let (n0, n2) = (1.0, 1.52);
        let lambda_nm = 550.0;
        let mut layer =
            FilmLayer::quarter_wave(FilmMaterial::constant("H", 2.35), lambda_nm, n0, 0.0);
        layer.thickness_nm *= 2.0;
        let c = Coating {
            layers: vec![layer],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        };
        let got = coating_rt_at(&c, n0, n2, lambda_nm / 1000.0, 1.0).unwrap();
        let bare = ((n0 - n2) / (n0 + n2)).powi(2);
        approx(got.r_s, bare, 1e-12, "half-wave layer must be invisible");
    }

    /// Energy is conserved in a transparent stack at every angle: R + T = 1.
    #[test]
    fn a_transparent_stack_conserves_energy() {
        check_energy().unwrap();
    }

    fn check_energy() -> Result<(), String> {
        let c = quarter_wave_mirror(550.0, 0.0, 6);
        for deg in [0.0f64, 20.0, 45.0, 70.0] {
            for nm in [420.0, 500.0, 550.0, 620.0, 700.0] {
                let got =
                    coating_rt_at(&c, 1.0, 1.52, nm / 1000.0, deg.to_radians().cos()).unwrap();
                close(got.r_s + got.t_s, 1.0, 1e-12, "s energy")?;
                close(got.r_p + got.t_p, 1.0, 1e-12, "p energy")?;
            }
        }
        Ok(())
    }

    /// The whole reason this module exists: tilt a quarter-wave stack and its
    /// band walks toward the blue. Checked against the effective-index law,
    /// which is the standard approximation of exactly this calculation, so
    /// agreement to a few per cent is the right expectation and disagreement in
    /// *direction* or *order of magnitude* would be a bug.
    #[test]
    fn the_reflectance_band_shifts_blue_with_angle() {
        let design_nm = 550.0;
        let c = quarter_wave_mirror(design_nm, 0.0, 8);
        let peak_at = |deg: f64| -> f64 {
            let cos_i = deg.to_radians().cos();
            let (mut best_nm, mut best_r) = (0.0, -1.0);
            let mut nm = 380.0;
            while nm <= 750.0 {
                let r = coating_rt_at(&c, 1.0, 1.52, nm / 1000.0, cos_i)
                    .unwrap()
                    .r();
                if r > best_r {
                    best_r = r;
                    best_nm = nm;
                }
                nm += 0.25;
            }
            best_nm
        };
        let at0 = peak_at(0.0);
        let at45 = peak_at(45.0);
        approx(
            at0,
            design_nm,
            1.0,
            "the band centres on the design wavelength",
        );
        assert!(
            at45 < at0 - 25.0,
            "45° must move the band well toward the blue: {at0} -> {at45}"
        );
        // The effective-index law, λ(θ) = λ₀√(1 − (sin θ / n_eff)²), with n_eff
        // the stack's own effective index. For a quarter-wave pair it sits
        // between the two indices; 5% is a generous band around that estimate.
        let n_eff = (2.35f64 * 1.46).sqrt();
        let predicted = design_nm * (1.0 - (45f64.to_radians().sin() / n_eff).powi(2)).sqrt();
        approx(
            at45,
            predicted,
            0.05 * design_nm,
            "shift follows the edge law",
        );
    }

    /// At normal incidence s and p are the same thing by symmetry; tilting a
    /// dichroic pulls them apart, which is why a tilted dichroic polarizes.
    ///
    /// The split is smallest deep inside the reflectance band, where both
    /// components saturate near 1, and largest at the band **edge**, because s
    /// and p have different edges. A dichroic is used at its edge, so that is
    /// where this matters and where the test looks.
    #[test]
    fn polarization_splits_only_off_axis() {
        let c = quarter_wave_mirror(550.0, 45.0, 8);
        let sweep = |cos_i: f64| {
            let (mut worst, mut worst_nm) = (0.0f64, 0.0f64);
            let mut nm = 380.0;
            while nm <= 780.0 {
                let rt = coating_rt_at(&c, 1.0, 1.52, nm / 1000.0, cos_i).unwrap();
                if rt.polarization_split() > worst {
                    worst = rt.polarization_split();
                    worst_nm = nm;
                }
                nm += 0.5;
            }
            (worst, worst_nm)
        };

        let (on_axis, _) = sweep(1.0);
        approx(on_axis, 0.0, 1e-14, "no split on axis, at any wavelength");

        let (tilted, at_nm) = sweep(45f64.to_radians().cos());
        assert!(
            tilted > 0.25,
            "45° must separate s and p strongly somewhere in the band: \
             {tilted} at {at_nm} nm"
        );

        // Inside the reflectance band s holds up better than p, which is the
        // familiar statement and the reason a tilted dichroic polarizes what it
        // reflects. It is *only* true inside the band: out in the sidelobes the
        // two have different ripple, and p can cross above s at a particular
        // wavelength. Optiland's TMM agrees with this module to 1e-14 there, so
        // the crossing is the coating's behaviour and not an error here.
        let band = coating_rt_at(&c, 1.0, 1.52, 0.55, 45f64.to_radians().cos()).unwrap();
        assert!(
            band.r_s > band.r_p,
            "in the band s must reflect more than p: {band:?}"
        );
    }

    /// A thick plate absorbs by its own `k` on every pass: an uncoated one at
    /// normal incidence is the textbook incoherent slab, and a silvered one
    /// lit through its glass returns its front face's 4% plus the silver's
    /// share of what survives two passes.
    #[test]
    fn a_thick_plate_absorbs_on_every_pass() {
        let (n, k, d_mm, lam) = (1.52, 2e-6, 4.0, 1.0);
        let glass = FilmMaterial::absorbing("glass", n, k);
        let tau = (-4.0 * std::f64::consts::PI * k * d_mm * 1e3 / lam).exp();
        let r = ((n - 1.0) / (n + 1.0)).powi(2);
        let bare = Coating {
            layers: Vec::new(),
            backing: CoatingBacking::Plate { substrate: glass.clone(), exit: ExitFace::Bare, thickness_mm: d_mm },
            incidence: Incidence::Film,
            design: None,
        };
        let got = coating_rt(&bare, 1.0, lam, 1.0).unwrap();
        let t_slab = (1.0 - r) * (1.0 - r) * tau / (1.0 - r * r * tau * tau);
        assert!((got.t() - t_slab).abs() < 1e-5, "{} vs {t_slab}", got.t());
        assert!(tau < 0.92 && tau > 0.9);

        let silvered = Coating {
            layers: vec![FilmLayer::new(materials::silver(), 150.0)],
            backing: CoatingBacking::Plate { substrate: glass, exit: ExitFace::Bare, thickness_mm: d_mm },
            incidence: Incidence::Substrate,
            design: None,
        };
        let r_ag = coating_rt_at(
            &Coating { layers: vec![FilmLayer::new(materials::silver(), 150.0)], backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None },
            n,
            1.0,
            lam,
            1.0,
        )
        .unwrap()
        .r();
        let want = r + (1.0 - r) * (1.0 - r) * r_ag * tau * tau / (1.0 - r * r_ag * tau * tau);
        let got = coating_rt(&silvered, 1.0, lam, 1.0).unwrap().r();
        assert!((got - want).abs() < 2e-4, "{got} vs {want}");
    }

    /// A plate transmits less than its coating alone, because the light still
    /// has to leave the glass, and an AR on the back face recovers most of it.
    #[test]
    fn the_back_face_of_a_plate_costs_transmission() {
        let layers = quarter_wave_mirror(450.0, 45.0, 4).layers;
        let bk7 = FilmMaterial::constant("N-BK7", 1.5168);
        let bare = Coating {
            layers: layers.clone(),
            backing: CoatingBacking::Plate {
                substrate: bk7.clone(),
                exit: ExitFace::Bare,
                thickness_mm: 0.0,
            },
            incidence: Incidence::Film,
            design: None,
        };
        let coated = Coating {
            backing: CoatingBacking::Plate {
                substrate: bk7,
                exit: ExitFace::AntiReflective { residual: 0.05 },
                thickness_mm: 0.0,
            },
            ..bare.clone()
        };
        let cos45 = 45f64.to_radians().cos();
        // Well into the pass band of a 450 nm design.
        let t_bare = coating_rt(&bare, 1.0, 0.65, cos45).unwrap().t();
        let t_ar = coating_rt(&coated, 1.0, 0.65, cos45).unwrap().t();
        assert!(
            t_bare < 0.97,
            "a bare glass back face must cost a few percent: {t_bare}"
        );
        assert!(
            t_ar > t_bare + 0.01,
            "an AR back face should recover most of it: {t_bare} -> {t_ar}"
        );
    }

    /// A dichroic designed for 45° really does cross 50% at its stated edge when
    /// used at 45°. This is the contract the editor's number makes, and it is
    /// checked against the same transfer matrix the tracer runs.
    #[test]
    fn a_dichroic_designed_for_45_has_its_edge_at_45() {
        for (edge, long_pass) in [(500.0, true), (550.0, true), (620.0, false)] {
            let design = if long_pass {
                CoatingDesign::long_pass(edge, 45.0)
            } else {
                CoatingDesign::short_pass(edge, 45.0)
            };
            let c = plate_coating(design, materials::bk7(), ExitFace::Bare, 1.0);
            let got = crossing_nm(
                &c,
                1.0,
                1.0,
                45.0,
                0.5,
                380.0,
                900.0,
                BandSide::of(long_pass),
            )
            .expect("the band must cross 50% somewhere in the visible");
            approx(got, edge, 3.0, "edge at the design angle");
        }
    }

    /// The bug, stated as a test. A dichroic specified at normal incidence and
    /// then hung at 45°, which is how every splitter and dichroic in the app is
    /// placed, does not cut where its label says: the band walks about 8% blue.
    /// Reading the normal-incidence curve at 45° is what the kernel used to do.
    #[test]
    fn the_same_stack_cuts_somewhere_else_when_you_tilt_it() {
        let c = plate_coating(
            CoatingDesign::long_pass(550.0, 0.0),
            materials::bk7(),
            ExitFace::Bare,
            1.0,
        );
        let on_axis = crossing_nm(&c, 1.0, 1.0, 0.0, 0.5, 380.0, 900.0, BandSide::Long).unwrap();
        let at_45 = crossing_nm(&c, 1.0, 1.0, 45.0, 0.5, 380.0, 900.0, BandSide::Long).unwrap();
        approx(on_axis, 550.0, 3.0, "designed on axis, cuts on axis");
        assert!(
            at_45 < on_axis - 30.0,
            "tilting to 45° must move the edge well into the blue: \
             {on_axis} -> {at_45}"
        );
        // And the size of it: a shift this big is the difference between
        // passing and blocking a laser line.
        assert!(
            (on_axis - at_45) > 35.0 && (on_axis - at_45) < 60.0,
            "expected roughly a 40 to 50 nm shift, got {}",
            on_axis - at_45
        );
    }

    /// More periods make a deeper, steeper band. The transition width is set by
    /// the index contrast and cannot be dialled in, which is why the editor
    /// reports it rather than offering it as a handle.
    #[test]
    fn more_periods_steepen_the_edge() {
        let width = |periods: u32| {
            let mut d = CoatingDesign::long_pass(550.0, 45.0);
            d.periods = periods;
            let c = plate_coating(d, materials::bk7(), ExitFace::Bare, 1.0);
            let lo = crossing_nm(&c, 1.0, 1.0, 45.0, 0.9, 380.0, 900.0, BandSide::Long).unwrap();
            let hi = crossing_nm(&c, 1.0, 1.0, 45.0, 0.1, 380.0, 900.0, BandSide::Long).unwrap();
            (hi - lo).abs()
        };
        let few = width(4);
        let many = width(14);
        assert!(
            many < few,
            "a longer stack must give a steeper edge: {few} nm -> {many} nm"
        );
    }

    /// The single-layer AR really does beat bare glass across the visible, and
    /// the three-layer one beats the single layer. Sanity for the presets.
    #[test]
    fn the_ar_presets_actually_anti_reflect() {
        let bare = ((1.0 - 1.5168) / (1.0 + 1.5168f64)).powi(2);
        let mean_r = |kind: DesignKind| {
            let d = CoatingDesign {
                kind,
                lambda_nm: 550.0,
                aoi_deg: 0.0,
                periods: 1,
                n_high: 2.35,
                n_low: 1.46,
                adapt: None,
            };
            let c = Coating {
                layers: design_layers(&d, 1.0, 1.5168),
                backing: CoatingBacking::Medium,
                incidence: Incidence::Film,
                design: Some(d),
            };
            let mut sum = 0.0;
            let mut n = 0;
            for i in 0..=20 {
                let nm = 450.0 + 200.0 * (i as f64) / 20.0;
                sum += coating_rt_at(&c, 1.0, 1.5168, nm / 1000.0, 1.0)
                    .unwrap()
                    .r();
                n += 1;
            }
            sum / n as f64
        };
        let single = mean_r(DesignKind::SingleLayerAr);
        let broad = mean_r(DesignKind::BroadbandArLegacy);
        assert!(
            single < bare * 0.4,
            "single-layer AR: {single} vs bare {bare}"
        );
        assert!(
            broad < single,
            "three-layer AR should beat one: {broad} vs {single}"
        );
        assert!(broad < 0.01, "a broadband AR should hold under 1%: {broad}");
    }

    /// An opaque metal film transmits nothing and reflects most of what arrives,
    /// which is the one case where the absorbing branch of the admittance and
    /// the complex propagation cosine actually do work.
    #[test]
    fn an_opaque_aluminium_film_is_a_mirror() {
        let c = Coating {
            layers: vec![FilmLayer::new(materials::aluminium(), 120.0)],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        };
        let rt = coating_rt_at(&c, 1.0, 1.5168, 0.55, 1.0).unwrap();
        assert!(
            rt.t() < 1e-3,
            "120 nm of aluminium is opaque: T = {}",
            rt.t()
        );
        assert!(
            rt.r() > 0.88 && rt.r() < 0.95,
            "aluminium reflects about 91% in the visible, got {}",
            rt.r()
        );
        assert!(rt.a() > 0.05, "and absorbs the rest: A = {}", rt.a());
    }

    /// Helper mirroring the synthesis in `build3`, kept local so this module's
    /// tests do not depend on the authoring layer.
    /// A lossless stack reflects the same from either side, provided the side
    /// it is met from is the side it is walked from.
    ///
    /// The broadband AR is the stack that shows it: quarter, half, quarter of
    /// three different materials, so it is not its own mirror image. Met from
    /// inside the glass and walked in the order it was written, it is a
    /// different coating, and a worse one than none at all.
    #[test]
    fn a_stack_met_from_behind_is_walked_from_behind() {
        check_reverse_walk().unwrap();
    }

    fn check_reverse_walk() -> Result<(), String> {
        let n_glass = 1.5168;
        let design = CoatingDesign {
            kind: DesignKind::BroadbandArLegacy,
            lambda_nm: 550.0,
            aoi_deg: 0.0,
            periods: 1,
            n_high: 2.35,
            n_low: 1.38,
            adapt: None,
        };
        let front = film_coating(design, 1.0, n_glass);
        let back = front.seen_from_behind();
        assert!(
            !front.layers.is_empty()
                && front.layers.first().map(|l| &l.material) != front.layers.last().map(|l| &l.material),
            "the stack is not symmetric, or the test proves nothing"
        );

        for (lambda, aoi_air) in [(0.45, 0.0), (0.55, 0.0), (0.65, 0.0), (0.55, 30.0)] {
            // The same ray on either side of the interface: n sin theta is what
            // Snell keeps, so the angle inside the glass is the conjugate one.
            let cos_air = f64::to_radians(aoi_air).cos();
            let sin_glass = f64::to_radians(aoi_air).sin() / n_glass;
            let cos_glass = (1.0 - sin_glass * sin_glass).sqrt();

            let outside = coating_rt_at(&front, 1.0, n_glass, lambda, cos_air).unwrap();
            let inside = coating_rt_at(&back, n_glass, 1.0, lambda, cos_glass).unwrap();
            for (name, a, b) in [
                ("R_s", outside.r_s, inside.r_s),
                ("R_p", outside.r_p, inside.r_p),
                ("T_s", outside.t_s, inside.t_s),
                ("T_p", outside.t_p, inside.t_p),
            ] {
                close(a, b, 1e-12, &format!("{name} at {lambda} um, {aoi_air} deg, air against glass"))?;
            }

            // Walked the written way from inside, it is another coating.
            let wrong = coating_rt_at(&front, n_glass, 1.0, lambda, cos_glass).unwrap();
            if (wrong.r() - inside.r()).abs() <= 1e-3 {
                return Err(format!(
                    "at {lambda} um the wrong walk should show: {} vs {}",
                    wrong.r(),
                    inside.r()
                ));
            }
        }

        // And at the design wavelength the wrong walk reflects more than bare
        // glass does, which is how it made a coated lens pass less light.
        let bare = ((n_glass - 1.0) / (n_glass + 1.0)).powi(2);
        let wrong = coating_rt_at(&front, n_glass, 1.0, 0.55, 1.0).unwrap().r();
        let right = coating_rt_at(&back, n_glass, 1.0, 0.55, 1.0).unwrap().r();
        if right >= 0.005 {
            return Err(format!("the AR does its job from inside: {right}"));
        }
        if wrong <= bare {
            return Err(format!("the wrong walk is worse than bare glass: {wrong} vs {bare}"));
        }
        Ok(())
    }

    fn quarter_wave_mirror(lambda_nm: f64, aoi_deg: f64, periods: u32) -> Coating {
        let h = FilmMaterial::constant("TiO2", 2.35);
        let l = FilmMaterial::constant("SiO2", 1.46);
        let mut layers = Vec::new();
        for _ in 0..periods {
            layers.push(FilmLayer::quarter_wave(h.clone(), lambda_nm, 1.0, aoi_deg));
            layers.push(FilmLayer::quarter_wave(l.clone(), lambda_nm, 1.0, aoi_deg));
        }
        layers.push(FilmLayer::quarter_wave(h, lambda_nm, 1.0, aoi_deg));
        Coating {
            layers,
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        }
    }

    // -----------------------------------------------------------------------
    // CF-00: the cases the prior spec left open (C02, C03, C04, C08, C09), a
    // deterministic property sweep, and the mutation checks (C31).
    // -----------------------------------------------------------------------

    /// A coating on its own medium, layers given as `(n, k, thickness_nm)`.
    fn on_medium(layers: &[(f64, f64, f64)]) -> Coating {
        Coating {
            layers: layers
                .iter()
                .map(|&(n, k, d)| FilmLayer::new(FilmMaterial::absorbing("L", n, k), d))
                .collect(),
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        }
    }

    /// Every amplitude of `a` against `b`, to `tol`.
    fn same_amplitudes(a: &FilmAmp, b: &FilmAmp, tol: f64, what: &str) -> Result<(), String> {
        for (name, x, y) in [
            ("r_s", a.r_s, b.r_s),
            ("r_p", a.r_p, b.r_p),
            ("t_s", a.t_s, b.t_s),
            ("t_p", a.t_p, b.t_p),
        ] {
            close(x.re, y.re, tol, &format!("{what}: Re {name}"))?;
            close(x.im, y.im, tol, &format!("{what}: Im {name}"))?;
        }
        close(a.t_power_scale, b.t_power_scale, tol, &format!("{what}: scale"))
    }

    /// An asymmetric stack with an absorbing layer in it: silica, a thin
    /// aluminium film, titania. Not its own mirror image, and lossy, so
    /// reflection from the two sides differs while transmission must not.
    const ASYMMETRIC: [(f64, f64, f64); 3] = [(1.46, 0.0, 94.0), (0.958, 6.69, 8.0), (2.35, 0.0, 61.0)];

    /// C02: a zero-thickness layer is the same coating as no layer at all,
    /// amplitudes included, wherever it sits in the stack.
    #[test]
    fn c02_a_zero_thickness_layer_is_no_layer() {
        let base = [(1.46, 0.0, 94.0), (2.35, 0.0, 61.0)];
        let without = on_medium(&base);
        for at in 0..=base.len() {
            for ghost in [(1.9, 0.0, 0.0), (0.958, 6.69, 0.0)] {
                let mut with = base.to_vec();
                with.insert(at, ghost);
                let with = on_medium(&with);
                for deg in [0.0f64, 35.0, 70.0] {
                    for lam in [0.45, 0.55, 0.8] {
                        let cos_i = deg.to_radians().cos();
                        let a = coating_amp_at(&with, 1.0, 1.5168, lam, cos_i).unwrap();
                        let b = coating_amp_at(&without, 1.0, 1.5168, lam, cos_i).unwrap();
                        same_amplitudes(&a, &b, 1e-14, &format!("layer at {at}, {deg} deg, {lam} um"))
                            .unwrap();
                    }
                }
            }
        }
    }

    /// C03: one layer split in two of the same material is the same layer.
    #[test]
    fn c03_a_split_layer_is_one_layer() {
        for (n, k) in [(1.46, 0.0), (2.35, 0.0), (0.958, 6.69)] {
            let whole = on_medium(&[(1.38, 0.0, 100.0), (n, k, 70.0), (1.65, 0.0, 80.0)]);
            for cut in [0.5, 17.3, 35.0, 69.0] {
                let split =
                    on_medium(&[(1.38, 0.0, 100.0), (n, k, cut), (n, k, 70.0 - cut), (1.65, 0.0, 80.0)]);
                for deg in [0.0f64, 45.0, 80.0] {
                    for lam in [0.4, 0.55, 1.2] {
                        let cos_i = deg.to_radians().cos();
                        let a = coating_amp_at(&split, 1.0, 1.52, lam, cos_i).unwrap();
                        let b = coating_amp_at(&whole, 1.0, 1.52, lam, cos_i).unwrap();
                        same_amplitudes(&a, &b, 1e-14, &format!("n {n}, cut {cut}, {deg} deg, {lam} um"))
                            .unwrap();
                    }
                }
            }
        }
    }

    /// C04: the ideal single-layer AR, index √1.5 on n = 1.5, a quarter wave
    /// at 550 nm, reflects nothing at its design point. Exactly, up to f64:
    /// the layer is given to the twelfth decimal of a nanometre so that the
    /// residual is rounding and not a design error.
    #[test]
    fn c04_the_ideal_quarter_wave_ar_is_an_exact_null() {
        check_c04().unwrap();
    }

    fn check_c04() -> Result<(), String> {
        let c = on_medium(&[(1.5f64.sqrt(), 0.0, 112.268279877562)]);
        let rt = coating_rt_at(&c, 1.0, 1.5, 0.55, 1.0).unwrap();
        if rt.r_s > 1e-28 || rt.r_p > 1e-28 {
            return Err(format!("R_s {:.3e}, R_p {:.3e}: not a null", rt.r_s, rt.r_p));
        }
        close(rt.t_s, 1.0, 1e-15, "T_s at the null")
    }

    /// C08: an asymmetric absorbing stack met from the substrate is the same
    /// stack with its layers reversed and its media swapped, and nothing more.
    /// Transmission is reciprocal; reflection is not, and must not be assumed
    /// to be.
    #[test]
    fn c08_a_reversed_asymmetric_stack_is_its_mirror_image() {
        check_c08().unwrap();
    }

    fn check_c08() -> Result<(), String> {
        let (n0, ns) = (1.0, 1.5168);
        let front = on_medium(&ASYMMETRIC);
        let back = front.seen_from_behind();
        let mut flipped = ASYMMETRIC;
        flipped.reverse();
        let mirror = on_medium(&flipped);
        let mut biggest_r_gap = 0.0f64;
        for deg in [0.0f64, 30.0, 60.0] {
            for lam in [0.45, 0.55, 0.7] {
                let cos_air = deg.to_radians().cos();
                let sin_glass = deg.to_radians().sin() * n0 / ns;
                let cos_glass = (1.0 - sin_glass * sin_glass).sqrt();
                let what = format!("{deg} deg, {lam} um");

                // Met from behind is the mirror-image stack lit from the glass.
                let a = coating_amp_at(&back, ns, n0, lam, cos_glass).unwrap();
                let b = coating_amp_at(&mirror, ns, n0, lam, cos_glass).unwrap();
                same_amplitudes(&a, &b, 1e-15, &what)?;

                let fwd = coating_rt_at(&front, n0, ns, lam, cos_air).unwrap();
                let rev = coating_rt_at(&back, ns, n0, lam, cos_glass).unwrap();
                close(fwd.t_s, rev.t_s, 1e-12, &format!("{what}: T_s reciprocal"))?;
                close(fwd.t_p, rev.t_p, 1e-12, &format!("{what}: T_p reciprocal"))?;
                biggest_r_gap = biggest_r_gap.max((fwd.r() - rev.r()).abs());
            }
        }
        if biggest_r_gap <= 1e-3 {
            return Err(format!("R from the two sides differs by only {biggest_r_gap:.2e}"));
        }
        Ok(())
    }

    /// C09: no NaN or infinity anywhere on a grazing sweep, including exact
    /// grazing and the hair either side of a critical angle, for every kind of
    /// stack the canvas builds.
    #[test]
    fn c09_a_grazing_sweep_stays_finite() {
        let ar = |kind| {
            film_coating(
                CoatingDesign { kind, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38, adapt: None },
                1.0,
                1.5168,
            )
        };
        let stacks = [
            bare(),
            ar(DesignKind::SingleLayerAr),
            ar(DesignKind::BroadbandArLegacy),
            quarter_wave_mirror(550.0, 45.0, 10),
            on_medium(&ASYMMETRIC),
            on_medium(&[(0.958, 6.69, 150.0)]),
            plate_coating_for_test(),
        ];
        let mut cosines: Vec<f64> = (0..=500).map(|i| (85.0 + 0.01 * i as f64).to_radians().cos()).collect();
        cosines.extend([0.0, 1e-300, f64::MIN_POSITIVE, 1e-12, 1.0]);
        let crit = (1.0f64 / 1.5168).asin();
        let near_crit = [crit - 1e-9, crit, crit + 1e-9].map(f64::cos);

        let ok = |rt: FilmRt, what: &str| {
            for v in [rt.r_s, rt.r_p, rt.t_s, rt.t_p] {
                assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{what}: {rt:?}");
            }
            assert!(rt.r_s + rt.t_s <= 1.0 + 1e-12 && rt.r_p + rt.t_p <= 1.0 + 1e-12, "{what}: gain {rt:?}");
        };
        let finite = |a: FilmAmp| a.r_s.is_finite() && a.r_p.is_finite() && a.t_s.is_finite() && a.t_p.is_finite();
        for (i, c) in stacks.iter().enumerate() {
            for lam in [0.4, 0.55, 1.0] {
                for &cos_i in &cosines {
                    let what = format!("stack {i}, {lam} um, cos {cos_i:e}");
                    ok(coating_rt_at(c, 1.0, 1.5168, lam, cos_i).unwrap(), &what);
                    if let Ok(a) = coating_amp_at(c, 1.0, 1.5168, lam, cos_i) {
                        assert!(finite(a), "{what}: {a:?}");
                    }
                }
                // From inside the glass, through the critical angle.
                let back = c.seen_from_behind();
                for &cos_i in &near_crit {
                    let what = format!("stack {i}, {lam} um, critical, cos {cos_i}");
                    ok(coating_rt_at(&back, 1.5168, 1.0, lam, cos_i).unwrap(), &what);
                    if let Ok(a) = coating_amp_at(&back, 1.5168, 1.0, lam, cos_i) {
                        assert!(finite(a), "{what}: {a:?}");
                    }
                }
            }
        }
    }

    /// The power reflectance of a bare interface from `n0` (real) into the
    /// complex `n1`, s and p, by Fresnel's formulas written out with complex
    /// cosines and nothing from this module.
    fn bulk_fresnel(n0: f64, n1: Cx, deg: f64) -> (f64, f64) {
        // sin θ₀ from cos θ₀, as the kernel takes it: near grazing the
        // reflectance is steep enough in angle that the two roundings differ.
        let c0 = deg.to_radians().cos();
        let s0 = (1.0 - c0 * c0).max(0.0).sqrt();
        // cos θ₁ = √(1 − (n₀ sin θ₀ / n₁)²), the root with a positive real part.
        let ratio = Cx::real(n0 * s0) / n1;
        let c1 = (Cx::real(1.0) - ratio * ratio).sqrt();
        let n0c = Cx::real(n0);
        let rs = (n0c * c0 - n1 * c1) / (n0c * c0 + n1 * c1);
        let rp = (n1 * c0 - n0c * c1) / (n1 * c0 + n0c * c1);
        (rs.norm_sqr(), rp.norm_sqr())
    }

    /// C12: a metal film far past its skin depth is the bulk metal. 5 µm of
    /// aluminium used to overflow to NaN; the opacity rule makes it the
    /// semi-infinite metal, to rounding.
    #[test]
    fn c12_a_thick_metal_film_is_the_bulk_metal() {
        let al = Cx::new(0.958, 6.69);
        let c = on_medium(&[(al.re, al.im, 5000.0)]);
        for deg in [0.0f64, 30.0, 60.0, 85.0] {
            for lam in [0.4, 0.55, 1.0] {
                let (rt, notes) =
                    coating_eval(&c, 1.0, 1.5168, lam, deg.to_radians().cos(), &FilmContext::default()).unwrap();
                let (rs, rp) = bulk_fresnel(1.0, al, deg);
                approx(rt.r_s, rs, 1e-12, &format!("{deg} deg, {lam} um: R_s"));
                approx(rt.r_p, rp, 1e-12, &format!("{deg} deg, {lam} um: R_p"));
                assert_eq!((rt.t_s, rt.t_p), (0.0, 0.0));
                approx(rt.a(), 1.0 - rt.r(), 1e-15, "A = 1 - R");
                assert_eq!(notes.opaque_from, Some(0));
            }
        }
        // Behind the opaque layer nothing counts: a different substrate and a
        // layer under the metal change nothing.
        let behind = on_medium(&[(al.re, al.im, 5000.0), (2.35, 0.0, 61.0)]);
        let a = coating_rt_at(&c, 1.0, 1.5168, 0.55, 1.0).unwrap();
        let b = coating_rt_at(&behind, 1.0, 1.0, 0.55, 1.0).unwrap();
        assert_eq!(a, b);
        // And a film under the opacity limit still transmits.
        let thin = on_medium(&[(al.re, al.im, 12.0)]);
        let (rt, notes) = coating_eval(&thin, 1.0, 1.5168, 0.55, 1.0, &FilmContext::default()).unwrap();
        assert!(rt.t() > 0.05 && notes.opaque_from.is_none());
    }

    /// The rescaling guard: forty layers that each absorb short of the
    /// opacity limit add up to a matrix that used to overflow. Now the
    /// reflectance is finite and the transmittance underflows to nothing.
    #[test]
    fn many_absorbing_layers_short_of_opaque_stay_finite() {
        let layers: Vec<(f64, f64, f64)> =
            (0..40).map(|i| (1.5 + 0.02 * i as f64, 3.0, 600.0 + i as f64)).collect();
        let c = on_medium(&layers);
        for deg in [0.0f64, 45.0, 85.0] {
            let (rt, notes) =
                coating_eval(&c, 1.0, 1.5168, 0.4, deg.to_radians().cos(), &FilmContext::default()).unwrap();
            assert!(notes.opaque_from.is_none(), "no single layer is opaque");
            for v in [rt.r_s, rt.r_p, rt.t_s, rt.t_p] {
                assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{deg} deg: {rt:?}");
            }
            assert!(rt.t() < 1e-100, "{rt:?}");
            // The reflectance is the first few layers' and nothing else: the
            // stack truncated after them gives the same number.
            let front = on_medium(&layers[..3]);
            let r_front = coating_eval(&front, 1.0, 1.5168, 0.4, deg.to_radians().cos(), &FilmContext::default())
                .unwrap()
                .0;
            approx(rt.r_s, r_front.r_s, 1e-9, "R_s from the front layers");
        }
    }

    /// C13, the closed-form half: a bare opaque substrate is Fresnel into the
    /// complex medium, and what is not reflected is absorbed.
    #[test]
    fn c13_a_bare_opaque_substrate_is_fresnel_into_the_metal() {
        let al = FilmMaterial::absorbing("Al", 0.958, 6.69);
        let bare = Coating {
            layers: vec![],
            backing: CoatingBacking::Opaque { substrate: al.clone() },
            incidence: Incidence::Film,
            design: None,
        };
        for deg in [0.0f64, 20.0, 45.0, 70.0, 89.0] {
            let rt = coating_rt_at(&bare, 1.0, 1.0, 0.55, deg.to_radians().cos()).unwrap();
            let (rs, rp) = bulk_fresnel(1.0, Cx::new(0.958, 6.69), deg);
            approx(rt.r_s, rs, 1e-12, &format!("{deg} deg R_s"));
            approx(rt.r_p, rp, 1e-12, &format!("{deg} deg R_p"));
            assert_eq!((rt.t_s, rt.t_p), (0.0, 0.0));
            let amp = coating_amp_at(&bare, 1.0, 1.0, 0.55, deg.to_radians().cos()).unwrap();
            approx(amp.powers().r_s, rt.r_s, 1e-15, "amplitude R_s");
            assert_eq!(amp.t_power_scale, 0.0);
        }
        // Lit from the metal side, it refuses by name.
        let behind = bare.seen_from_behind();
        assert_eq!(coating_rt_at(&behind, 1.0, 1.0, 0.55, 1.0), Err(FilmError::AbsorbingIncidentMedium));
        assert_eq!(coating_amp_at(&behind, 1.0, 1.0, 0.55, 1.0), Err(FilmError::AbsorbingIncidentMedium));
        // An overcoat changes the reflectance and still transmits nothing.
        let over = Coating {
            layers: vec![FilmLayer::new(FilmMaterial::constant("SiO2", 1.46), 94.0)],
            ..bare
        };
        let rt = coating_rt_at(&over, 1.0, 1.0, 0.55, 1.0).unwrap();
        assert!(rt.t() == 0.0 && (rt.r() - bulk_fresnel(1.0, Cx::new(0.958, 6.69), 0.0).0).abs() > 1e-3);
    }

    /// C24, the kernel's half: Brewster, the last hundredth of a degree, and
    /// exact grazing.
    #[test]
    fn c24_the_edges_of_the_angle_range() {
        let (n1, n2) = (1.0f64, 1.5168f64);
        let c = bare();
        // Brewster.
        let rt = coating_rt_at(&c, n1, n2, 0.55, (n2 / n1).atan().cos()).unwrap();
        assert!(rt.r_p <= 1e-14, "R_p at Brewster: {:e}", rt.r_p);
        // 89.99 degrees: finite, and the Fresnel value, which is about
        // 1 - 6e-4 for glass rather than 1. To 1e-11, not 1e-12: the kernel
        // rebuilds cos θ₀ from n₀ sin θ₀, which near grazing keeps only about
        // 7e-9 of cos θ₀'s relative precision, and that is 1.3e-12 in R here
        // (the closed form, held to 50 digits, is exact to 2e-16). Using the
        // given cosine directly would move the last bit of every oblique
        // result, which C33 forbids. Up to 89.9 degrees the error is below
        // 1e-13, inside §7's 1e-12.
        let th = 89.99f64.to_radians();
        let rt = coating_rt_at(&c, n1, n2, 0.55, th.cos()).unwrap();
        let (rs, rp) = bulk_fresnel(n1, Cx::real(n2), 89.99);
        approx(rt.r_s, rs, 1e-11, "R_s at 89.99");
        approx(rt.r_p, rp, 1e-11, "R_p at 89.99");
        let th = 89.9f64.to_radians();
        let rt = coating_rt_at(&c, n1, n2, 0.55, th.cos()).unwrap();
        let (rs, rp) = bulk_fresnel(n1, Cx::real(n2), 89.9);
        approx(rt.r_s, rs, 1e-12, "R_s at 89.9");
        approx(rt.r_p, rp, 1e-12, "R_p at 89.9");
        assert!(rt.r() < 1.0 - 1e-4);
        // Exactly grazing: reflected entirely, by rule, for every kind of stack.
        for s in [bare(), on_medium(&ASYMMETRIC), plate_coating_for_test(), quarter_wave_mirror(550.0, 45.0, 6)] {
            let rt = coating_rt_at(&s, n1, n2, 0.55, 0.0).unwrap();
            assert_eq!(rt, FilmRt { r_s: 1.0, r_p: 1.0, t_s: 0.0, t_p: 0.0 });
        }
        let a = coating_amp_at(&on_medium(&ASYMMETRIC), n1, n2, 0.55, 0.0).unwrap();
        assert_eq!((a.r_s, a.r_p, a.t_power_scale), (Cx::real(-1.0), Cx::real(-1.0), 0.0));
        // And the limit approached from inside the range agrees.
        let near = coating_amp_at(&on_medium(&ASYMMETRIC), n1, n2, 0.55, 1e-7).unwrap();
        assert!((near.r_s - a.r_s).norm_sqr() < 1e-10 && (near.r_p - a.r_p).norm_sqr() < 1e-10, "{near:?}");
    }

    /// A table read past its end: `Permissive` answers with the table's own
    /// policy and says so; `Strict` refuses by name.
    #[test]
    fn strict_refuses_what_permissive_clamps_and_flags() {
        use crate::fidelity::Numerics;
        let silver = materials::silver(); // Babar and Weaver, 0.21 to 3.54 um, clamped outside
        let c = Coating {
            layers: vec![FilmLayer::new(silver, 150.0)],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        };
        let strict = FilmContext { numerics: Numerics::Strict, ..FilmContext::default() };
        let permissive = FilmContext::default();
        let (_, notes) = coating_eval(&c, 1.0, 1.52, 5.0, 1.0, &permissive).unwrap();
        assert_eq!(notes.clamped, vec![0]);
        let err = coating_eval(&c, 1.0, 1.52, 5.0, 1.0, &strict).unwrap_err();
        assert_eq!(err, FilmError::OutOfTable { layer: 0, lambda_um: 5.0 });
        assert!(err.to_string().starts_with("layer 1 has no optical constants at 5.0000 um"), "{err}");
        // Inside the table both agree, to the bit.
        let a = coating_eval(&c, 1.0, 1.52, 0.55, 0.8, &permissive).unwrap();
        let b = coating_eval(&c, 1.0, 1.52, 0.55, 0.8, &strict).unwrap();
        assert_eq!(a, b);
        assert!(a.1.clamped.is_empty());
        // A gain medium: accepted as before under Permissive, refused under Strict.
        let gain = on_medium(&[(1.5, -0.01, 100.0)]);
        assert!(coating_rt_at(&gain, 1.0, 1.52, 0.55, 1.0).is_ok());
        assert_eq!(
            coating_eval(&gain, 1.0, 1.52, 0.55, 1.0, &strict).unwrap_err(),
            FilmError::NonPhysicalIndex { layer: 0 }
        );
        // A negative thickness is refused whatever the numerics.
        assert_eq!(
            coating_rt_at(&on_medium(&[(1.5, 0.0, -1.0)]), 1.0, 1.52, 0.55, 1.0),
            Err(FilmError::NonPhysicalIndex { layer: 0 })
        );
        // A plate has no amplitude, and says so.
        assert_eq!(coating_amp_at(&plate_coating_for_test(), 1.0, 1.0, 0.55, 1.0), Err(FilmError::Incoherent));
    }

    /// Every reason a coating can be refused has a sentence, and
    /// `docs/coatings.md` carries each one (CF-02's exit criterion).
    #[test]
    fn every_film_error_has_its_sentence_in_the_docs() {
        let docs = include_str!("../../../docs/coatings.md");
        for e in [
            FilmError::Degenerate,
            FilmError::OutOfTable { layer: 2, lambda_um: 0.25 },
            FilmError::OutOfTable { layer: FilmError::SUBSTRATE, lambda_um: 0.25 },
            FilmError::NonPhysicalIndex { layer: 0 },
            FilmError::AbsorbingIncidentMedium,
            FilmError::Overflow,
            FilmError::Incoherent,
            FilmError::SlabOnPlate,
        ] {
            let text = e.to_string();
            assert!(!text.is_empty());
            assert!(docs.contains(&text), "docs/coatings.md does not quote: {text}");
        }
    }

    /// The deterministic sweep CF-02 asks for, on the kernel's side: 2000
    /// random 64-layer absorbing stacks, n in [1.3, 4], k in [0, 7], d in
    /// [1, 3000] nm, at 0, 30, 60 and 85 degrees, never NaN, never gain, and
    /// R + T + A = 1. Agreement with an oracle is the Python suite's half
    /// (`test_thinfilm_stress.py`).
    #[test]
    fn random_64_layer_absorbing_stacks_stay_physical() {
        let mut g = Sweep(0xCF02_0064);
        for case in 0..2000 {
            let layers: Vec<(f64, f64, f64)> =
                (0..64).map(|_| (g.range(1.3, 4.0), g.range(0.0, 7.0), g.range(1.0, 3000.0))).collect();
            let c = on_medium(&layers);
            for deg in [0.0f64, 30.0, 60.0, 85.0] {
                let rt = coating_rt_at(&c, 1.0, 1.52, 0.55, deg.to_radians().cos())
                    .unwrap_or_else(|e| panic!("case {case}, {deg} deg: {e}"));
                for v in [rt.r_s, rt.r_p, rt.t_s, rt.t_p] {
                    assert!(v.is_finite() && (0.0..=1.0).contains(&v), "case {case}, {deg} deg: {rt:?}");
                }
                assert!(rt.r_s + rt.t_s <= 1.0 + 1e-12 && rt.r_p + rt.t_p <= 1.0 + 1e-12);
            }
        }
    }

    // -----------------------------------------------------------------------
    // CF-04: absorption by layer
    // -----------------------------------------------------------------------

    fn powers(c: &Coating, n0: f64, nf: f64, lam: f64, deg: f64) -> FilmPowers {
        coating_power_flow_eval(c, n0, nf, lam, deg.to_radians().cos(), &FilmContext::default()).unwrap().0
    }

    /// `R + T + Σ A_k + A_substrate` for one polarization.
    fn closure(p: &FilmPowers, s_pol: bool) -> f64 {
        if s_pol {
            p.rt.r_s + p.rt.t_s + p.a_layer_s.iter().sum::<f64>() + p.a_substrate_s
        } else {
            p.rt.r_p + p.rt.t_p + p.a_layer_p.iter().sum::<f64>() + p.a_substrate_p
        }
    }

    /// C15: three layers with only the middle one absorbing. Everything the
    /// stack absorbs is in the middle layer, the lossless ones absorb nothing
    /// at all, and the books close.
    #[test]
    fn c15_the_absorption_is_where_the_absorber_is() {
        check_c15().unwrap();
    }

    fn check_c15() -> Result<(), String> {
        let c = on_medium(&[(1.46, 0.0, 94.0), (0.958, 6.69, 8.0), (2.35, 0.0, 61.0)]);
        for deg in [0.0f64, 30.0, 60.0] {
            for lam in [0.45, 0.55, 0.7] {
                let p = powers(&c, 1.0, 1.5168, lam, deg);
                let what = format!("{deg} deg, {lam} um");
                for (a, r, t) in [(&p.a_layer_s, p.rt.r_s, p.rt.t_s), (&p.a_layer_p, p.rt.r_p, p.rt.t_p)] {
                    if a[0] != 0.0 || a[2] != 0.0 {
                        return Err(format!("{what}: a lossless layer absorbed {a:?}"));
                    }
                    close(a[1], 1.0 - r - t, 1e-12, &format!("{what}: the metal takes 1 - R - T"))?;
                }
                close(closure(&p, true), 1.0, 1e-12, &format!("{what}: s closes"))?;
                close(closure(&p, false), 1.0, 1e-12, &format!("{what}: p closes"))?;
            }
        }
        // Met from the substrate the same layer still holds it all, and the
        // index is the written one.
        let back = c.seen_from_behind();
        let p = powers(&back, 1.5168, 1.0, 0.55, 20.0);
        if p.a_layer_s[0] != 0.0 || p.a_layer_s[2] != 0.0 || p.a_layer_s[1] <= 0.0 {
            return Err(format!("met from behind: {:?}", p.a_layer_s));
        }
        Ok(())
    }

    /// A dielectric AR deposits exactly nothing.
    #[test]
    fn a_lossless_coating_absorbs_exactly_nothing() {
        let ar = film_coating(
            CoatingDesign { kind: DesignKind::BroadbandArLegacy, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38, adapt: None },
            1.0,
            1.5168,
        );
        for deg in [0.0f64, 40.0, 80.0] {
            let p = powers(&ar, 1.0, 1.5168, 0.55, deg);
            assert!(p.a_layer_s.iter().chain(&p.a_layer_p).all(|a| *a == 0.0), "{p:?}");
            assert_eq!((p.a_substrate_s, p.a_substrate_p), (0.0, 0.0));
        }
    }

    /// The per-layer split closes on 2000 random absorbing stacks, both
    /// sides, and no layer absorbs a negative power.
    #[test]
    fn random_stacks_close_layer_by_layer() {
        let mut g = Sweep(0xCF04_0001);
        for case in 0..2000 {
            let layers = g.stack(24, 4.0);
            let mut c = on_medium(&layers);
            let ns = g.range(1.4, 2.0);
            let deg = g.range(0.0, 85.0);
            let lam = g.range(0.4, 1.6);
            let (n0, nf) = if case % 2 == 1 {
                c = c.seen_from_behind();
                (ns, 1.0)
            } else {
                (1.0, ns)
            };
            let p = powers(&c, n0, nf, lam, deg.min(if n0 > nf { 40.0 } else { 85.0 }));
            for s_pol in [true, false] {
                let total = closure(&p, s_pol);
                assert!((total - 1.0).abs() <= 1e-12, "case {case}: closes to {total}");
            }
            for a in p.a_layer_s.iter().chain(&p.a_layer_p) {
                assert!(*a >= -1e-15, "case {case}: negative absorption {a}");
            }
        }
    }

    /// An opaque substrate keeps what enters it; an opaque layer inside a
    /// stack takes everything that reaches it and the layers behind it none.
    #[test]
    fn what_an_opaque_element_takes_is_booked_to_it() {
        let c = Coating {
            layers: vec![FilmLayer::new(FilmMaterial::constant("SiO2", 1.46), 94.0)],
            backing: CoatingBacking::Opaque { substrate: FilmMaterial::absorbing("Al", 0.958, 6.69) },
            incidence: Incidence::Film,
            design: None,
        };
        let p = powers(&c, 1.0, 1.0, 0.55, 30.0);
        assert_eq!(p.a_layer_s[0], 0.0);
        approx(p.a_substrate_s, 1.0 - p.rt.r_s, 1e-12, "the substrate keeps the rest");
        approx(closure(&p, false), 1.0, 1e-12, "p closes");

        let buried = on_medium(&[(1.46, 0.0, 94.0), (0.958, 6.69, 5000.0), (2.35, 0.1, 61.0)]);
        let p = powers(&buried, 1.0, 1.5168, 0.55, 0.0);
        assert_eq!(p.a_layer_s[2], 0.0, "behind the opaque layer nothing is absorbed");
        approx(p.a_layer_s[1], 1.0 - p.rt.r_s, 1e-12, "the opaque layer takes the rest");
    }

    /// A coated plate: front stack, slab and far face, from either side. The
    /// slab's share grows with its extinction, and the books close.
    #[test]
    fn a_plate_books_its_layers_and_its_slab() {
        for incidence in [Incidence::Film, Incidence::Substrate] {
            let mut c = on_medium(&[(1.46, 0.0, 94.0), (0.958, 6.69, 6.0), (2.35, 0.0, 61.0)]);
            c.backing = CoatingBacking::Plate {
                substrate: FilmMaterial::absorbing("glass", 1.5168, 2e-6),
                exit: ExitFace::Bare,
                thickness_mm: 3.0,
            };
            c.incidence = incidence;
            for deg in [0.0f64, 45.0] {
                let p = powers(&c, 1.0, 1.0, 0.55, deg);
                approx(closure(&p, true), 1.0, 1e-12, &format!("{incidence:?} {deg}: s"));
                approx(closure(&p, false), 1.0, 1e-12, &format!("{incidence:?} {deg}: p"));
                assert!(p.a_substrate_s > 0.0 && p.a_layer_s[1] > 0.0);
                assert_eq!((p.a_layer_s[0], p.a_layer_s[2]), (0.0, 0.0));
            }
        }
    }

    // -----------------------------------------------------------------------
    // CF-05: slabs, and the plate's Mueller matrix
    // -----------------------------------------------------------------------

    fn incoherent_ctx() -> FilmContext {
        FilmContext { plate: crate::fidelity::PlateModel::Incoherent, ..FilmContext::default() }
    }

    /// A 2 mm absorbing slab with a front stack and a rear two-layer AR, on air.
    fn coated_slab() -> Coating {
        Coating {
            layers: vec![
                FilmLayer::new(FilmMaterial::constant("SiO2", 1.46), 94.0),
                FilmLayer::new(FilmMaterial::absorbing("Al", 0.958, 6.69), 5.0),
                FilmLayer::new(FilmMaterial::constant("TiO2", 2.35), 61.0),
                FilmLayer::slab(FilmMaterial::absorbing("glass", 1.5168, 3e-6), 2.0),
                FilmLayer::new(FilmMaterial::constant("Al2O3", 1.65), 83.0),
                FilmLayer::new(FilmMaterial::constant("MgF2", 1.38), 99.0),
            ],
            backing: CoatingBacking::Medium,
            incidence: Incidence::Film,
            design: None,
        }
    }

    /// A coated slab closes its books element by element, from either side,
    /// and its slab and its metal film are where the absorption is.
    #[test]
    fn a_stack_with_a_slab_closes_element_by_element() {
        for c in [coated_slab(), coated_slab().seen_from_behind()] {
            for deg in [0.0f64, 30.0, 60.0] {
                let (p, _) = coating_power_flow_eval(&c, 1.0, 1.0, 0.55, deg.to_radians().cos(), &FilmContext::default()).unwrap();
                approx(closure(&p, true), 1.0, 1e-12, "s");
                approx(closure(&p, false), 1.0, 1e-12, "p");
                assert!(p.a_layer_s[1] > 0.0 && p.a_layer_s[3] > 0.0, "{:?}", p.a_layer_s);
                for k in [0, 2, 4, 5] {
                    assert_eq!(p.a_layer_s[k], 0.0, "lossless layer {k}");
                }
                // The powers through the plain entry point are the same ones.
                let rt = coating_rt_at(&c, 1.0, 1.0, 0.55, deg.to_radians().cos()).unwrap();
                assert_eq!(rt, p.rt);
            }
        }
        // A slab has no amplitude.
        assert_eq!(coating_amp_at(&coated_slab(), 1.0, 1.0, 0.55, 1.0), Err(FilmError::Incoherent));
    }

    /// A plate under `Incoherent` is its films, its slab and its far face:
    /// the same powers as the same coating written with a slab.
    #[test]
    fn an_incoherent_plate_is_its_films_a_slab_and_a_face() {
        let films = vec![FilmLayer::new(FilmMaterial::constant("SiO2", 1.46), 94.0), FilmLayer::new(FilmMaterial::absorbing("Al", 0.958, 6.69), 5.0)];
        let glass = FilmMaterial::absorbing("glass", 1.5168, 3e-6);
        let plate = Coating {
            layers: films.clone(),
            backing: CoatingBacking::Plate { substrate: glass.clone(), exit: ExitFace::Bare, thickness_mm: 2.0 },
            incidence: Incidence::Film,
            design: None,
        };
        let mut layers = films;
        layers.push(FilmLayer::slab(glass, 2.0));
        let written = Coating { layers, backing: CoatingBacking::Medium, incidence: Incidence::Film, design: None };
        for deg in [0.0f64, 40.0] {
            let cos = deg.to_radians().cos();
            let a = coating_eval(&plate, 1.0, 1.0, 0.55, cos, &incoherent_ctx()).unwrap().0;
            let b = coating_eval(&written, 1.0, 1.0, 0.55, cos, &incoherent_ctx()).unwrap().0;
            for (x, y) in [(a.r_s, b.r_s), (a.r_p, b.r_p), (a.t_s, b.t_s), (a.t_p, b.t_p)] {
                approx(x, y, 1e-14, &format!("{deg} deg"));
            }
            // And PhaseFree is the plate's own series, a different convention
            // for the lossy far face, close but not the same.
            let old = coating_eval(&plate, 1.0, 1.0, 0.55, cos, &FilmContext::default()).unwrap().0;
            approx(old.r(), a.r(), 1e-6, "PhaseFree and Incoherent agree to the slab's loss");
        }
        // A slab inside a plate-backed coating is refused by name.
        let mut both = plate.clone();
        both.layers.push(FilmLayer::slab(FilmMaterial::constant("glass", 1.5), 1.0));
        assert_eq!(coating_eval(&both, 1.0, 1.0, 0.55, 1.0, &incoherent_ctx()).unwrap_err(), FilmError::SlabOnPlate);
    }

    /// C18: the plate's Mueller matrix is the incoherent series. Against the
    /// series summed term by term to 200 passes; its S0 is the incoherent
    /// powers; and the light it hands on is never more than fully polarized.
    #[test]
    fn c18_the_plate_mueller_matrix_is_the_incoherent_series() {
        use crate::polarization::{Mueller, Stokes};
        let glass = FilmMaterial::absorbing("glass", 1.5168, 2e-6);
        let films = vec![FilmLayer::new(FilmMaterial::constant("TiO2", 2.35), 58.0), FilmLayer::new(FilmMaterial::constant("SiO2", 1.46), 94.0)];
        let plate = Coating {
            layers: films.clone(),
            backing: CoatingBacking::Plate { substrate: glass.clone(), exit: ExitFace::Bare, thickness_mm: 3.0 },
            incidence: Incidence::Film,
            design: None,
        };
        let ctx = incoherent_ctx();
        for deg in [10.0f64, 45.0, 70.0] {
            for lam in [0.45, 0.6] {
                let cos = deg.to_radians().cos();
                let (mr, mt) = coating_mueller_eval(&plate, 1.0, 1.0, lam, cos, &ctx).unwrap();

                // The same series, term by term, from the faces' amplitudes
                // with the glass's complex index, as the plate's own faces.
                let n_sub = glass.index_at(lam).unwrap();
                let q = Cx::real(deg.to_radians().sin());
                let walked: Vec<(Cx, f64)> = films.iter().map(|l| (l.material.index_at(lam).unwrap(), l.thickness_nm)).collect();
                let rev: Vec<(Cx, f64)> = walked.iter().rev().copied().collect();
                let e = stack_amp(&walked, Cx::real(1.0), n_sub, lam, q).0;
                let e_in = stack_amp(&rev, n_sub, Cx::real(1.0), lam, q).0;
                let o = stack_amp(&[], n_sub, Cx::real(1.0), lam, q).0;
                let m_r = |a: &FilmAmp| Mueller::from_diagonal_jones(a.r_s, a.r_p);
                let m_t = |a: &FilmAmp| Mueller::from_diagonal_jones(a.t_s, a.t_p).scaled(a.t_power_scale);
                let nc = n_sub * snell_cos(q, n_sub);
                let p = (-4.0 * std::f64::consts::PI * 3000.0 * nc.im.abs() / lam).exp();
                let mut sum_r = m_r(&e);
                let mut sum_t = Mueller::attenuator(0.0);
                let mut loop_k = Mueller::IDENTITY;
                for _ in 0..200 {
                    sum_r = sum_r.plus(&m_t(&e_in).times(&m_r(&o)).times(&loop_k).times(&m_t(&e)).scaled(p * p));
                    sum_t = sum_t.plus(&m_t(&o).times(&loop_k).times(&m_t(&e)).scaled(p));
                    loop_k = loop_k.times(&m_r(&e_in).times(&m_r(&o)).scaled(p * p));
                }
                for i in 0..4 {
                    for j in 0..4 {
                        approx(mr.0[i][j], sum_r.0[i][j], 1e-12, &format!("{deg} deg {lam} um: M_R[{i}][{j}]"));
                        approx(mt.0[i][j], sum_t.0[i][j], 1e-12, &format!("{deg} deg {lam} um: M_T[{i}][{j}]"));
                    }
                }

                // S0 for unpolarized light is the incoherent powers.
                let rt = coating_eval(&plate, 1.0, 1.0, lam, cos, &ctx).unwrap().0;
                approx(mr.0[0][0], rt.r(), 1e-12, "S0 of M_R");
                approx(mt.0[0][0], rt.t(), 1e-12, "S0 of M_T");
                approx(mr.0[0][1], 0.5 * (rt.r_s - rt.r_p), 1e-12, "S1 row of M_R");

                // +45 degree linear light: never more than fully polarized.
                let plus45 = Stokes::linear(1.0, std::f64::consts::FRAC_PI_4);
                for m in [&mr, &mt] {
                    let out = m.apply(&plus45);
                    assert!(out.dop() <= 1.0 + 1e-12, "{deg} deg {lam} um: DOP {}", out.dop());
                }
            }
        }
    }

    /// A coated plate with a slightly absorbing substrate, for the sweeps.
    fn plate_coating_for_test() -> Coating {
        let mut c = quarter_wave_mirror(550.0, 45.0, 4);
        c.backing = CoatingBacking::Plate {
            substrate: FilmMaterial::absorbing("glass", 1.5168, 1e-6),
            exit: ExitFace::Bare,
            thickness_mm: 2.0,
        };
        c
    }

    /// A deterministic generator for the property sweeps (SplitMix64), so a
    /// failure reproduces from its case number and no random crate is needed.
    struct Sweep(u64);

    impl Sweep {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in `[lo, hi)`.
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
        }
        fn stack(&mut self, max_layers: u64, k_max: f64) -> Vec<(f64, f64, f64)> {
            let n = 1 + self.next() % max_layers;
            (0..n)
                .map(|_| {
                    let k = if k_max > 0.0 && self.next() % 3 == 0 { self.range(0.0, k_max) } else { 0.0 };
                    (self.range(1.3, 2.5), k, self.range(1.0, 400.0))
                })
                .collect()
        }
    }

    /// Lossless random stacks conserve energy in both polarizations; lossy
    /// ones never create it, and transmit the same from either side.
    #[test]
    fn random_stacks_conserve_or_lose_energy_and_transmit_reciprocally() {
        let mut g = Sweep(0x0C0A_7146);
        for case in 0..400 {
            let lossy = case % 2 == 1;
            let layers = g.stack(20, if lossy { 3.0 } else { 0.0 });
            let ns = g.range(1.4, 2.0);
            let deg = g.range(0.0, 80.0);
            let lam = g.range(0.4, 1.6);
            let front = on_medium(&layers);
            let back = front.seen_from_behind();
            let cos_air = deg.to_radians().cos();
            let sin_sub = deg.to_radians().sin() / ns;
            let cos_sub = (1.0 - sin_sub * sin_sub).sqrt();
            let fwd = coating_rt_at(&front, 1.0, ns, lam, cos_air).unwrap();
            let rev = coating_rt_at(&back, ns, 1.0, lam, cos_sub).unwrap();
            let what = format!("case {case}: {layers:?} on {ns}, {deg} deg, {lam} um");
            for (r, t) in [(fwd.r_s, fwd.t_s), (fwd.r_p, fwd.t_p), (rev.r_s, rev.t_s), (rev.r_p, rev.t_p)] {
                if lossy {
                    assert!(1.0 - r - t >= -1e-12, "{what}: gain");
                } else {
                    approx(r + t, 1.0, 1e-12, &what);
                }
            }
            approx(fwd.t_s, rev.t_s, 1e-12, &format!("{what}: T_s reciprocity"));
            approx(fwd.t_p, rev.t_p, 1e-12, &format!("{what}: T_p reciprocity"));
        }
    }

    /// C31: every one-line mistake in [`mutation::Mutation`] is caught by the
    /// check named next to it, and that check passes on the real code.
    #[test]
    fn every_listed_mistake_is_caught() {
        type Check = fn() -> Result<(), String>;
        let table: [(Mutation, &str, Check); 8] = [
            (Mutation::RpSignDropped, "check_rp_sign", check_rp_sign),
            (Mutation::GrowingBranch, "check_tir_phase", check_tir_phase),
            (Mutation::NoAdmittanceRatio, "check_energy", check_energy),
            (Mutation::NoAdmittanceRatio, "check_c08", check_c08),
            (Mutation::ThicknessAsMicrons, "check_c04", check_c04),
            (Mutation::NoReverseWalk, "check_c08", check_c08),
            (Mutation::NoReverseWalk, "check_reverse_walk", check_reverse_walk),
            (Mutation::AbsorptionOffByOne, "check_c15", check_c15),
        ];
        for (m, name, check) in table {
            check().unwrap_or_else(|e| panic!("{name} fails on the real code: {e}"));
            let caught = mutation::with(m, check);
            assert!(caught.is_err(), "{m:?} is not caught by {name}");
        }
    }
}
