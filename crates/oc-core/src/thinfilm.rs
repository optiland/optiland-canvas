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
    },
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
    /// The light meets the substrate first and the stack on the way out: the
    /// back of a coated plate.
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
    /// re-synthesize. Not used by the calculation: the layers are the truth.
    #[serde(default)]
    pub design: Option<CoatingDesign>,
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
/// No power notices: past the critical angle `R = 1` and `T = 0` on either
/// branch, which is why this went unremarked while the module only computed
/// powers. The reflection **phase** notices, it is the only thing a polarization
/// calculation reads, and it is the whole of what a Fresnel rhomb does. A finite
/// layer is invariant under the flip (its matrix is even in `cos θ`), so the
/// exit medium is the only place this can matter.
fn decaying_cos(n: Cx, cos_t: Cx) -> Cx {
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
    (big_r.clamp(0.0, 1.0), big_t.clamp(0.0, 1.0))
}

/// Unpolarized Fresnel reflectance for one polarization at a bare interface,
/// used for the far face of a plate.
fn bare_interface_r(n1: Cx, n2: Cx, q: Cx, p_pol: bool) -> f64 {
    let (c1, c2) = (snell_cos(q, n1), snell_cos(q, n2));
    let (e1, e2) = (admittance(n1, c1, p_pol), admittance(n2, c2, p_pol));
    let denom = e1 + e2;
    if denom.norm_sqr() == 0.0 {
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

    match &c.backing {
        // The stack sits straight on whatever the scene has on the far side: an
        // AR coating on a lens face is backed by that lens's own glass.
        CoatingBacking::Medium => Some(rt_both(&layers, n0, n_far, lambda_um, q)),
        CoatingBacking::Plate { substrate, exit } => {
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
    Some(stack_amp(&layers, n0, n_far, lambda_um, q))
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
    let cos_sub = snell_cos(q, n_sub);
    // Tangential field to whole field for p. An evanescent exit carries no
    // transmitted ray at all, so the scale goes to zero and the amplitude with
    // it rather than through a division by nothing.
    let (t_p_full, scale) = if cos_sub.norm_sqr() > 1e-300 && eta0_s.re.abs() > 0.0 {
        (t_p * (cos0 / cos_sub), (eta_s_s.re / eta0_s.re).max(0.0))
    } else {
        (Cx::real(0.0), 0.0)
    };
    FilmAmp {
        r_s: r_s.conj(),
        // The one sign that separates Macleod's p basis from the ray frame.
        r_p: -r_p.conj(),
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
) -> FilmRt {
    let reversed: Vec<(Cx, f64)> = layers.iter().rev().copied().collect();
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

        let den = 1.0 - back_r * second_r;
        let den = if den.abs() < 1e-12 { 1e-12 } else { den };
        let r = (front_r + front_t * back_t * second_r / den).clamp(0.0, 1.0);
        let t = (front_t * second_t / den).clamp(0.0, 1.0);
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
}

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
        DesignKind::QuarterWaveStack => quarter_wave_pairs(&d, d.lambda_nm, n_incident, d.periods),
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
    let backing = CoatingBacking::Plate { substrate, exit };
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
    use super::*;

    fn approx(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
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
                approx(a.r_p.re, -a.r_s.re, 1e-9, "Re r_p = -Re r_s");
                approx(a.r_p.im, -a.r_s.im, 1e-9, "Im r_p = -Im r_s");
                // Transmission has no basis flip: both rays travel forward.
                approx(a.t_p.re, a.t_s.re, 1e-9, "Re t_p = Re t_s");
                approx(a.t_p.im, a.t_s.im, 1e-9, "Im t_p = Im t_s");
            }
        }
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
        let (n1, n2) = (1.51f64, 1.0f64);
        let n = n2 / n1;
        let crit = n.asin();
        for deg in [45.0f64, 50.0, 54.6231, 60.0, 70.0] {
            let th = deg.to_radians();
            assert!(th > crit, "{deg}° must be past critical");
            let a = coating_amp_at(&bare(), n1, n2, 0.55, th.cos()).unwrap();
            // Nothing gets out.
            approx(a.r_s.norm_sqr(), 1.0, 1e-14, "|r_s|");
            approx(a.r_p.norm_sqr(), 1.0, 1e-14, "|r_p|");
            approx(a.powers().t_s, 0.0, 1e-14, "T_s");

            // Sign as well as size. Under `exp(-iωt)` the evanescent wave must
            // decay, which puts `cos θ_t` on the `+i` branch, and then
            // `δ_s = −2 arctan(β/cos θ)` and `δ_p = −2 arctan(β/(n² cos θ))`
            // with `β = √(sin²θ − n²)`. Since `n < 1` the p shift is the larger
            // of the two negatives, so the retardance is **negative**. Its size
            // is the rhomb formula.
            let want =
                -2.0 * (th.cos() * (th.sin().powi(2) - n * n).sqrt() / th.sin().powi(2)).atan();
            let got = a.reflection_retardance();
            approx(got, want, 1e-12, &format!("{deg}° retardance"));
        }

        // Two bounces at 54.6231° make a quarter wave, the rhomb's whole point.
        let th = 54.6231f64.to_radians();
        let one = coating_amp_at(&bare(), n1, n2, 0.55, th.cos())
            .unwrap()
            .reflection_retardance();
        approx(
            (2.0 * one).abs(),
            std::f64::consts::FRAC_PI_2,
            2e-5,
            "two bounces make a quarter wave",
        );
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
        let c = quarter_wave_mirror(550.0, 0.0, 6);
        for deg in [0.0f64, 20.0, 45.0, 70.0] {
            for nm in [420.0, 500.0, 550.0, 620.0, 700.0] {
                let got =
                    coating_rt_at(&c, 1.0, 1.52, nm / 1000.0, deg.to_radians().cos()).unwrap();
                approx(got.r_s + got.t_s, 1.0, 1e-12, "s energy");
                approx(got.r_p + got.t_p, 1.0, 1e-12, "p energy");
            }
        }
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
            },
            incidence: Incidence::Film,
            design: None,
        };
        let coated = Coating {
            backing: CoatingBacking::Plate {
                substrate: bk7,
                exit: ExitFace::AntiReflective { residual: 0.05 },
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
}
