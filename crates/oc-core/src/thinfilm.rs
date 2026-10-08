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
/// Pending, because the code they would mutate lands later: the preview's
/// outside grid reused inside (C22, `oc-gpu`) and an off-by-one interface in
/// the per-layer absorption (C15).
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
}

fn zero_scalar() -> SpectralScalar {
    SpectralScalar::Constant(0.0)
}

impl FilmMaterial {
    /// A transparent material of constant index.
    pub fn constant(name: &str, n: f64) -> Self {
        Self {
            name: name.to_string(),
            n: SpectralScalar::Constant(n),
            k: zero_scalar(),
        }
    }

    /// An absorbing material of constant complex index.
    pub fn absorbing(name: &str, n: f64, k: f64) -> Self {
        Self {
            name: name.to_string(),
            n: SpectralScalar::Constant(n),
            k: SpectralScalar::Constant(k),
        }
    }

    /// The complex index at `lambda_um`, or `None` when a table has no value
    /// there under [`crate::scene3::OutOfRangePolicy::Error`].
    pub fn index_at(&self, lambda_um: f64) -> Option<Cx> {
        Some(Cx::new(
            self.n.sample(lambda_um)?,
            self.k.sample(lambda_um)?,
        ))
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
}

impl FilmLayer {
    pub fn new(material: FilmMaterial, thickness_nm: f64) -> Self {
        Self {
            material,
            thickness_nm,
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

    pub fn reflection_is_stray(&self) -> bool {
        matches!(
            self.design.map(|d| d.kind),
            Some(DesignKind::SingleLayerAr | DesignKind::BroadbandAr)
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
    SingleLayerAr,
    /// Quarter-half-quarter, the broadband three-layer AR.
    BroadbandAr,
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
fn stack_amp_pol(
    layers: &[(Cx, f64)],
    n0: Cx,
    n_sub: Cx,
    lambda_um: f64,
    q: Cx,
    p_pol: bool,
) -> (Cx, Cx, Cx, Cx) {
    let cos0 = snell_cos(q, n0);
    let cos_s = decaying_cos(n_sub, snell_cos(q, n_sub));
    let eta0 = admittance(n0, cos0, p_pol);
    let eta_s = admittance(n_sub, cos_s, p_pol);

    // The stack matrix, accumulated from the incident side inward.
    let (mut a, mut b, mut c, mut d) = (Cx::real(1.0), Cx::real(0.0), Cx::real(0.0), Cx::real(1.0));
    for &(n_l, thickness_nm) in layers {
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
    }

    // [B, C]ᵀ = M · [1, η_sub]ᵀ.
    let big_b = a + eta_s * b;
    let big_c = c + eta_s * d;
    let denom = eta0 * big_b + big_c;
    if denom.norm_sqr() == 0.0 || !denom.is_finite() {
        // A degenerate stack reflects everything rather than returning NaN.
        return (Cx::real(1.0), Cx::real(0.0), eta0, eta_s);
    }
    let r = (eta0 * big_b - big_c) / denom;
    let t = eta0 * 2.0 / denom;
    (r, t, eta0, eta_s)
}

/// One coherent stack: reflectance and transmittance for a single polarization.
fn stack_rt_pol(
    layers: &[(Cx, f64)],
    n0: Cx,
    n_sub: Cx,
    lambda_um: f64,
    q: Cx,
    p_pol: bool,
) -> (f64, f64) {
    let (r, t, eta0, eta_s) = stack_amp_pol(layers, n0, n_sub, lambda_um, q, p_pol);
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
    (big_r.clamp(0.0, 1.0), big_t.clamp(0.0, 1.0))
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

/// Reflectance and transmittance of a coating at one wavelength and angle.
///
/// `n_incident` is the index of the medium the ray is arriving through, which
/// the scene owns: the same coating on the same plate behaves differently in air
/// and in oil, and the tracer knows which it is. `cos_i` is the cosine of the
/// angle of incidence, positive.
///
/// Returns `None` when a layer's optical constants are tabulated and the
/// wavelength falls outside the table under the error policy, which is the same
/// contract the other spectral responses honour.
pub fn coating_rt_at(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
) -> Option<FilmRt> {
    let n0 = Cx::real(n_incident.max(1e-9));
    let n_far = Cx::real(n_far.max(1e-9));
    let cos_i = cos_i.clamp(-1.0, 1.0).abs();
    let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    let q = n0 * sin_i;

    let mut layers = Vec::with_capacity(c.layers.len());
    for l in &c.layers {
        let n = l.material.index_at(lambda_um)?;
        if !n.is_finite() || n.re <= 0.0 || !l.thickness_nm.is_finite() || l.thickness_nm < 0.0 {
            return None;
        }
        layers.push((n, l.thickness_nm));
    }
    walk_from_the_incident_side(c, &mut layers);

    match &c.backing {
        // The stack sits straight on whatever the scene has on the far side: an
        // AR coating on a lens face is backed by that lens's own glass.
        CoatingBacking::Medium => Some(rt_both(&layers, n0, n_far, lambda_um, q)),
        CoatingBacking::Plate { substrate, exit, thickness_mm } => {
            let n_sub = substrate.index_at(lambda_um)?;
            if !n_sub.is_finite() || n_sub.re <= 0.0 {
                return None;
            }
            let residual = match exit {
                ExitFace::Bare => 1.0,
                ExitFace::AntiReflective { residual } => residual.clamp(0.0, 1.0),
            };
            Some(plate_rt(
                &layers,
                n0,
                n_sub,
                n_far,
                lambda_um,
                q,
                residual,
                c.incidence,
                thickness_mm.max(0.0) * 1e3,
            ))
        }
    }
}

/// [`coating_rt_at`] for a surface with the same medium on both sides, which is
/// every zero-thickness surface in the scene (§6.6) and so every plate.
pub fn coating_rt(c: &Coating, n_incident: f64, lambda_um: f64, cos_i: f64) -> Option<FilmRt> {
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
}

/// [`coating_rt_at`]'s amplitudes: `r` and `t` for s and p, with their phases.
///
/// Returns `None` for a coating whose backing is a [`CoatingBacking::Plate`].
/// That model adds two faces **incoherently**, which is right for millimetres of
/// glass and is not a single Jones matrix: the relative phase between the two
/// contributions is exactly what the incoherent sum discards, so there is no
/// honest amplitude to return. A coating on a real glass body uses
/// [`CoatingBacking::Medium`] and does have one.
///
/// The supported domain is a real, non-absorbing incident and exit medium; the
/// layers themselves may be absorbing. Outside that, `T` and the amplitude
/// normalisation stop meaning what this returns.
pub fn coating_amp_at(
    c: &Coating,
    n_incident: f64,
    n_far: f64,
    lambda_um: f64,
    cos_i: f64,
) -> Option<FilmAmp> {
    if matches!(c.backing, CoatingBacking::Plate { .. }) {
        return None;
    }
    let n0 = Cx::real(n_incident.max(1e-9));
    let n_far = Cx::real(n_far.max(1e-9));
    let cos_i = cos_i.clamp(-1.0, 1.0).abs();
    let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    let q = n0 * sin_i;

    let mut layers = Vec::with_capacity(c.layers.len());
    for l in &c.layers {
        let n = l.material.index_at(lambda_um)?;
        if !n.is_finite() || n.re <= 0.0 || !l.thickness_nm.is_finite() || l.thickness_nm < 0.0 {
            return None;
        }
        layers.push((n, l.thickness_nm));
    }
    walk_from_the_incident_side(c, &mut layers);
    Some(stack_amp(&layers, n0, n_far, lambda_um, q))
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
    stack_amp(&[], n0, ns, 0.55, n0 * sin_i)
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
fn stack_amp(layers: &[(Cx, f64)], n0: Cx, n_sub: Cx, lambda_um: f64, q: Cx) -> FilmAmp {
    let (r_s, t_s, eta0_s, eta_s_s) = stack_amp_pol(layers, n0, n_sub, lambda_um, q, false);
    let (r_p, t_p, _, _) = stack_amp_pol(layers, n0, n_sub, lambda_um, q, true);
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
    FilmAmp {
        r_s: r_s.conj(),
        r_p,
        t_s: t_s.conj(),
        t_p: t_p_full.conj(),
        t_power_scale: scale,
    }
}

/// Both polarizations of one coherent stack.
fn rt_both(layers: &[(Cx, f64)], n0: Cx, n_sub: Cx, lambda_um: f64, q: Cx) -> FilmRt {
    let (r_s, t_s) = stack_rt_pol(layers, n0, n_sub, lambda_um, q, false);
    let (r_p, t_p) = stack_rt_pol(layers, n0, n_sub, lambda_um, q, true);
    FilmRt { r_s, r_p, t_s, t_p }
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
) -> FilmRt {
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
        let (r1, t1) = stack_rt_pol(layers, n0, n_sub, lambda_um, q, p_pol);
        let (r1p, t1p) = stack_rt_pol(&reversed, n_sub, n0, lambda_um, q, p_pol);
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
    FilmRt {
        r_s: out[0].0,
        t_s: out[0].1,
        r_p: out[1].0,
        t_p: out[1].1,
    }
}

// ---------------------------------------------------------------------------
// Synthesis: turning a design into layers
// ---------------------------------------------------------------------------

/// The standard film materials, enough to build every preset here.
///
/// Indices are the usual room-temperature values near the middle of the visible
/// band, held constant across it. Real films disperse by a percent or two and
/// real deposition runs miss their targets by more than that, so a constant is
/// an honest starting point; the layer list is editable and `n` is a tabulated
/// [`SpectralScalar`] underneath, so a measured dispersion can replace any of
/// these without touching the calculation.
pub mod materials {
    use super::FilmMaterial;

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
        FilmMaterial {
            name: name.to_string(),
            n: column(1),
            k: column(2),
        }
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
        DesignKind::SingleLayerAr => vec![FilmLayer::quarter_wave(
            materials::magnesium_fluoride(),
            d.lambda_nm,
            n_incident,
            d.aoi_deg,
        )],
        DesignKind::BroadbandAr => {
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
    design: CoatingDesign,
    substrate: FilmMaterial,
    exit: ExitFace,
    n_incident: f64,
) -> Coating {
    let backing = CoatingBacking::Plate { substrate, exit, thickness_mm: 0.0 };
    Coating {
        layers: design_layers_for(&design, n_incident, &backing, n_incident),
        backing,
        incidence: Incidence::Film,
        design: Some(design),
    }
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
pub fn film_coating(design: CoatingDesign, n_incident: f64, n_substrate: f64) -> Coating {
    Coating {
        layers: design_layers(&design, n_incident, n_substrate),
        backing: CoatingBacking::Medium,
        incidence: Incidence::Film,
        design: Some(design),
    }
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
        let r = coating_rt_at(c, n_incident, n_far, nm / 1000.0, cos_i)?.r();
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
            layers: vec![FilmLayer {
                material: FilmMaterial::constant("test", n1),
                thickness_nm: d_nm,
            }],
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
                },
                1.0,
                1.5168,
            ),
            Coating {
                layers: vec![FilmLayer {
                    material: materials::aluminium(),
                    thickness_nm: 12.0,
                }],
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
        let broad = mean_r(DesignKind::BroadbandAr);
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
            kind: DesignKind::BroadbandAr,
            lambda_nm: 550.0,
            aoi_deg: 0.0,
            periods: 1,
            n_high: 2.35,
            n_low: 1.38,
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
                CoatingDesign { kind, lambda_nm: 550.0, aoi_deg: 0.0, periods: 1, n_high: 2.35, n_low: 1.38 },
                1.0,
                1.5168,
            )
        };
        let stacks = [
            bare(),
            ar(DesignKind::SingleLayerAr),
            ar(DesignKind::BroadbandAr),
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
                    if let Some(a) = coating_amp_at(c, 1.0, 1.5168, lam, cos_i) {
                        assert!(finite(a), "{what}: {a:?}");
                    }
                }
                // From inside the glass, through the critical angle.
                let back = c.seen_from_behind();
                for &cos_i in &near_crit {
                    let what = format!("stack {i}, {lam} um, critical, cos {cos_i}");
                    ok(coating_rt_at(&back, 1.5168, 1.0, lam, cos_i).unwrap(), &what);
                    if let Some(a) = coating_amp_at(&back, 1.5168, 1.0, lam, cos_i) {
                        assert!(finite(a), "{what}: {a:?}");
                    }
                }
            }
        }
    }

    /// Known defect at the baseline, closed by CF-02 (C12). A metal film thick
    /// enough that its layer phase passes Im δ ≈ 355 overflows |B|² inside the
    /// complex division, and R comes out NaN, not the bulk metal's reflectance;
    /// the degenerate-denominator guard never sees it because the denominator
    /// itself is still finite. CF-02's opacity rule removes the overflow, and
    /// this test must then be replaced by C12, which asserts the bulk value.
    #[test]
    fn known_defect_a_very_thick_metal_film_overflows() {
        let c = on_medium(&[(0.958, 6.69, 5000.0)]);
        let rt = coating_rt_at(&c, 1.0, 1.5168, 0.4, 1.0).unwrap();
        assert!(rt.r_s.is_nan(), "the overflow is gone ({rt:?}): replace this test with C12");
        assert_eq!(rt.t_s, 0.0);
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
        let table: [(Mutation, &str, Check); 7] = [
            (Mutation::RpSignDropped, "check_rp_sign", check_rp_sign),
            (Mutation::GrowingBranch, "check_tir_phase", check_tir_phase),
            (Mutation::NoAdmittanceRatio, "check_energy", check_energy),
            (Mutation::NoAdmittanceRatio, "check_c08", check_c08),
            (Mutation::ThicknessAsMicrons, "check_c04", check_c04),
            (Mutation::NoReverseWalk, "check_c08", check_c08),
            (Mutation::NoReverseWalk, "check_reverse_walk", check_reverse_walk),
        ];
        for (m, name, check) in table {
            check().unwrap_or_else(|e| panic!("{name} fails on the real code: {e}"));
            let caught = mutation::with(m, check);
            assert!(caught.is_err(), "{m:?} is not caught by {name}");
        }
    }
}
