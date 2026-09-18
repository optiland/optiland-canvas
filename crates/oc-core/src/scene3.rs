//! The canonical 3D scene: `.ocanvas` v2.
//!
//! The serialized [`Scene3`] *is* the v2 document, exactly as the serialized
//! [`Scene`](crate::scene::Scene) is the v1 one. v1 stays readable through
//! [`crate::migrate_v1`]; nothing here replaces it yet
//! (`docs/architecture/canonical-3d-kernel.md`).
//!
//! # Identity
//!
//! One id space, one counter. Every id a scene hands out is unique across every
//! kind of object in it, and none is ever reused while the scene lives. So
//! `DuplicateId` is a single flat check over the whole document, and
//! [`SurfaceRef`] cannot be ambiguous.
//!
//! [`ObjectId`] is the stored identity of anything with an [`ObjectCommon`].
//! [`BodyId`], [`SurfaceObjectId`], [`ApertureId`], and [`DetectorId`] are typed
//! *views* over that same number, so an API that means "a body" can say so
//! without a second id space to keep in step. They convert both ways and never
//! collide, because the number is allocated once.
//!
//! # Validation
//!
//! [`Scene3::validate`] runs on load and is the reason a v2 document cannot
//! carry a rotation that scales, a dangling reference, a duplicate id, or a
//! spectral table that is not a function. The rule throughout: **refuse, do not
//! repair.** A document that contradicts itself has an author whose intent is
//! unknown, and guessing it is how a scene starts lying.

use crate::id::{
    ApertureId, BodyId, DetectorId, LayerId, ObjectId, PatchId, PathIntentId, SourceId,
    SurfaceObjectId, SurfaceRef, ViewId,
};
use crate::material::{GlassCatalog, Material};
use crate::profile2::{ProfileConstruction3, ProfileSolid3};
use crate::scene::Layer;
use crate::vec::Vec2;
use crate::vec3::{Isometry3, Plane3, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// The magic string tagging a document as `.ocanvas`. Shared with v1.
pub const OCANVAS_FORMAT: &str = crate::scene::OCANVAS_FORMAT;
/// The v2 schema version.
pub const OCANVAS_V2_VERSION: u32 = 2;
/// The dimensionality v2 documents declare.
pub const OCANVAS_V2_DIMENSION: u8 = 3;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a v2 document is not loadable.
///
/// Every variant names the object, so an error reaches the inspector as
/// something a user can act on rather than "invalid scene" (§17).
#[derive(Debug, Clone, PartialEq)]
pub enum SceneError {
    WrongFormat {
        found: String,
    },
    /// A version this build cannot read. v1 goes through the migration layer;
    /// anything above v2 is a document from the future.
    UnsupportedVersion {
        found: u32,
    },
    /// v2 is 3D by definition. A document claiming otherwise is malformed.
    WrongDimension {
        found: u8,
    },
    DuplicateId {
        id: u64,
    },
    /// A reference to an object that is not in the document.
    DanglingRef {
        what: &'static str,
        id: u64,
    },
    NonFinite {
        object: Option<u64>,
        field: String,
    },
    InvalidQuaternion {
        object: u64,
    },
    /// An index, count, or physical quantity outside its permitted range.
    OutOfRange {
        object: Option<u64>,
        field: String,
        reason: String,
    },
    /// A spectral or absorption table that is not a function of wavelength.
    InvalidTable {
        field: String,
        reason: String,
    },
    Json(String),
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongFormat { found } => {
                write!(f, "not an .ocanvas document: format is {found:?}")
            }
            Self::UnsupportedVersion { found } => write!(
                f,
                "unsupported .ocanvas version {found}; this build reads 1 and {OCANVAS_V2_VERSION}"
            ),
            Self::WrongDimension { found } => {
                write!(f, "v2 is 3D; document declares dimension {found}")
            }
            Self::DuplicateId { id } => write!(f, "id {id} is used more than once"),
            Self::DanglingRef { what, id } => write!(f, "{what} {id} does not exist"),
            Self::NonFinite { object, field } => match object {
                Some(o) => write!(f, "object {o}: {field} is not finite"),
                None => write!(f, "{field} is not finite"),
            },
            Self::InvalidQuaternion { object } => {
                write!(f, "object {object}: transform rotation is not a rotation")
            }
            Self::OutOfRange {
                object,
                field,
                reason,
            } => match object {
                Some(o) => write!(f, "object {o}: {field} {reason}"),
                None => write!(f, "{field} {reason}"),
            },
            Self::InvalidTable { field, reason } => write!(f, "{field}: {reason}"),
            Self::Json(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SceneError {}

// ---------------------------------------------------------------------------
// Spectral and absorption models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Linear,
}

/// What to do when a sample falls outside a table's wavelength range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OutOfRangePolicy {
    /// Refuse. The default for authored scientific data: a table that does not
    /// cover the wavelength being traced has no answer, and inventing one is how
    /// a filter curve silently becomes a straight line.
    #[default]
    Error,
    /// Hold the endpoint value.
    Clamp,
    /// Zero outside the range.
    Zero,
}

/// Bulk attenuation, `mm⁻¹`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum AbsorptionModel {
    #[default]
    None,
    Constant {
        alpha_per_mm: f64,
    },
    Tabulated {
        lambda_um: Vec<f64>,
        alpha_per_mm: Vec<f64>,
        interpolation: Interpolation,
        out_of_range: OutOfRangePolicy,
    },
}

impl AbsorptionModel {
    pub(crate) fn validate(&self, field: &str) -> Result<(), SceneError> {
        match self {
            Self::None => Ok(()),
            Self::Constant { alpha_per_mm } => {
                if !alpha_per_mm.is_finite() {
                    return Err(SceneError::NonFinite {
                        object: None,
                        field: field.to_string(),
                    });
                }
                // Negative absorption is gain. Nothing here models a laser.
                if *alpha_per_mm < 0.0 {
                    return Err(SceneError::OutOfRange {
                        object: None,
                        field: field.to_string(),
                        reason: "absorption coefficient is negative".to_string(),
                    });
                }
                Ok(())
            }
            Self::Tabulated {
                lambda_um,
                alpha_per_mm,
                ..
            } => {
                validate_table(field, lambda_um, alpha_per_mm)?;
                if alpha_per_mm.iter().any(|a| *a < 0.0) {
                    return Err(SceneError::InvalidTable {
                        field: field.to_string(),
                        reason: "absorption coefficient is negative".to_string(),
                    });
                }
                Ok(())
            }
        }
    }
}

/// A dimensionless quantity that may vary with wavelength.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpectralScalar {
    Constant(f64),
    Tabulated {
        lambda_um: Vec<f64>,
        values: Vec<f64>,
        interpolation: Interpolation,
        out_of_range: OutOfRangePolicy,
    },
}

impl SpectralScalar {
    /// Every value, for range checks that apply to the whole curve.
    fn values(&self) -> &[f64] {
        match self {
            Self::Constant(v) => std::slice::from_ref(v),
            Self::Tabulated { values, .. } => values,
        }
    }

    pub(crate) fn validate(&self, field: &str) -> Result<(), SceneError> {
        match self {
            Self::Constant(v) => {
                if !v.is_finite() {
                    return Err(SceneError::NonFinite {
                        object: None,
                        field: field.to_string(),
                    });
                }
                Ok(())
            }
            Self::Tabulated {
                lambda_um, values, ..
            } => validate_table(field, lambda_um, values),
        }
    }

    /// Is every sample within `[0, 1]`? Used where the quantity is a fraction of
    /// incident flux.
    fn is_unit_fraction(&self) -> bool {
        self.values().iter().all(|v| (0.0..=1.0).contains(v))
    }

    /// The largest sample. Used to bound `R + T` conservatively.
    fn max(&self) -> f64 {
        self.values()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The smallest sample. Used where a quantity has a floor rather than a
    /// ceiling, such as a film index that has to be positive.
    pub(crate) fn min(&self) -> f64 {
        self.values().iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// A wavelength table is a function: strictly increasing, at least two samples,
/// matched lengths, all finite.
fn validate_table(field: &str, lambda_um: &[f64], values: &[f64]) -> Result<(), SceneError> {
    let bad = |reason: &str| SceneError::InvalidTable {
        field: field.to_string(),
        reason: reason.to_string(),
    };
    if lambda_um.len() != values.len() {
        return Err(bad("wavelength and value arrays have different lengths"));
    }
    if lambda_um.len() < 2 {
        return Err(bad("a table needs at least two samples to interpolate"));
    }
    if lambda_um.iter().chain(values).any(|v| !v.is_finite()) {
        return Err(bad("a sample is not finite"));
    }
    if lambda_um.iter().any(|l| *l <= 0.0) {
        return Err(bad("wavelength must be positive"));
    }
    // Strictly increasing, which rejects a duplicate sample as a side effect: two
    // values at one wavelength make the table a relation rather than a function,
    // and which one you get would depend on the search.
    if lambda_um.windows(2).any(|w| w[1] <= w[0]) {
        return Err(bad("wavelengths must be strictly increasing"));
    }
    Ok(())
}

/// Linear interpolation of a validated wavelength table, honouring the
/// out-of-range policy. `None` means the sample fell outside the range under
/// [`OutOfRangePolicy::Error`] — the caller (the tracer) turns that into a typed
/// capability issue rather than inventing a value (§6.7).
///
/// The table is assumed valid: strictly increasing, at least two samples, matched
/// lengths. That is the load-time contract [`validate_table`] enforces, so this
/// never has to re-check it.
fn interp_table(xs: &[f64], ys: &[f64], x: f64, out_of_range: OutOfRangePolicy) -> Option<f64> {
    let n = xs.len();
    if x < xs[0] {
        return match out_of_range {
            OutOfRangePolicy::Error => None,
            OutOfRangePolicy::Clamp => Some(ys[0]),
            OutOfRangePolicy::Zero => Some(0.0),
        };
    }
    if x > xs[n - 1] {
        return match out_of_range {
            OutOfRangePolicy::Error => None,
            OutOfRangePolicy::Clamp => Some(ys[n - 1]),
            OutOfRangePolicy::Zero => Some(0.0),
        };
    }
    // The interval `[xs[i], xs[i+1]]` containing `x`. `partition_point` is a
    // binary search, so this stays cheap for a long table.
    let i = xs.partition_point(|&v| v <= x).saturating_sub(1).min(n - 2);
    let (x0, x1) = (xs[i], xs[i + 1]);
    let f = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
    Some(ys[i] + f * (ys[i + 1] - ys[i]))
}

impl SpectralScalar {
    /// The value at `lambda_um`, or `None` if out of range under
    /// [`OutOfRangePolicy::Error`].
    pub fn sample(&self, lambda_um: f64) -> Option<f64> {
        match self {
            Self::Constant(v) => Some(*v),
            Self::Tabulated {
                lambda_um: xs,
                values,
                out_of_range,
                ..
            } => interp_table(xs, values, lambda_um, *out_of_range),
        }
    }
}

impl AbsorptionModel {
    /// The attenuation coefficient (`mm⁻¹`) at `lambda_um`, or `None` if out of
    /// range under [`OutOfRangePolicy::Error`]. Absorption-free models return `0`.
    pub fn sample(&self, lambda_um: f64) -> Option<f64> {
        match self {
            Self::None => Some(0.0),
            Self::Constant { alpha_per_mm } => Some(*alpha_per_mm),
            Self::Tabulated {
                lambda_um: xs,
                alpha_per_mm,
                out_of_range,
                ..
            } => interp_table(xs, alpha_per_mm, lambda_um, *out_of_range),
        }
    }
}

impl DetectorResponse {
    /// Quantum efficiency at `lambda_um`, or `None` if out of range under
    /// [`OutOfRangePolicy::Error`]. [`DetectorResponse::Unity`] detects everything.
    pub fn efficiency_at(&self, lambda_um: f64) -> Option<f64> {
        match self {
            Self::Unity => Some(1.0),
            Self::Tabulated {
                lambda_um: xs,
                efficiency,
                out_of_range,
                ..
            } => interp_table(xs, efficiency, lambda_um, *out_of_range),
        }
    }
}

/// What a surface does to a ray that reaches it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SurfaceResponse {
    /// An uncoated dielectric interface: the complex Fresnel amplitudes, with
    /// `ar_residual` scaling the reflection and handing what it suppresses to
    /// the transmitted branch. So it does polarize, and past the critical angle
    /// it retards. **Not a coating model**: `ar_residual` is one number and a
    /// coating is a stack.
    DielectricFresnel {
        ar_residual: SpectralScalar,
    },
    Mirror {
        reflectivity: SpectralScalar,
    },
    /// Explicit reflectance and transmittance. Angle-independent until a model
    /// says otherwise, and `1 - R - T` is absorbed at the surface.
    ExplicitRt {
        reflectance: SpectralScalar,
        transmittance: SpectralScalar,
    },
    Absorber,
    /// An ideal thin lens: the transmitted ray bends so the slope law
    /// `u' = u − h/f` holds exactly at every height `h` in the lens plane
    /// (direction-aware, so a backward ray converges toward the front focus).
    /// Full transmission, no reflection, no dispersion, no added path length —
    /// the stand-in for a designed multi-element assembly whose prescription
    /// is not modelled (an infinity objective, a camera lens). Only meaningful
    /// on a standalone planar surface; a body patch authoring it is rejected
    /// by the tracer. Positive `focal_length_mm` converges, negative diverges.
    ThinLens {
        focal_length_mm: f64,
    },
    /// A real thin-film coating: a stack of layers, evaluated by the transfer
    /// matrix at the wavelength **and the angle of incidence** of each ray.
    ///
    /// This is the only response that knows what angle the light arrived at.
    /// Every other one is a number, or a table in wavelength, and reports the
    /// same value to a ray at 45° as to one down the axis. For a dichroic at 45°
    /// that is wrong by tens of nanometres of edge position, which is the reason
    /// this variant exists. See [`crate::thinfilm`].
    Coating {
        coating: crate::thinfilm::Coating,
    },
    /// An ideal absorbing linear polarizer: it passes the field component along
    /// its transmission axis and absorbs the perpendicular one.
    ///
    /// Zero thickness, no reflection, no dispersion. It models the polarization
    /// action of a polarizer and not a sheet of anything: a real one reflects,
    /// has a contrast that varies with wavelength, and degrades as it is
    /// tilted. Unpolarized light loses half its flux here, which is a physical
    /// fact and not an assumption.
    ///
    /// Angle enters only through the geometry: the axis is a direction in the
    /// element, and a skew ray sees its projection. That is enough for a
    /// crossed pair to leak off axis and not enough for anything a datasheet
    /// calls an acceptance angle.
    ///
    /// `extinction` is the **power** fraction of the blocked component that
    /// still gets through, so 0 is perfect and 1e-4 is an ordinary sheet
    /// polarizer. Only meaningful on a standalone planar surface, whose local
    /// `+X` the axis is measured from; a body patch authoring one is rejected.
    Polarizer {
        /// Transmission axis, degrees from the surface's local `+X`.
        axis_deg: f64,
        extinction: f64,
    },
    /// An ideal linear retarder: a waveplate, lossless, adding
    /// `retardance_deg` of phase to the slow axis relative to the fast one.
    ///
    /// A quarter wave is 90 degrees and turns linear light at 45 degrees to its
    /// axis into circular; a half wave is 180 and rotates a linear state by
    /// twice the angle between them. Zero thickness, so this is the
    /// polarization action alone: no walk-off, no ordinary and extraordinary
    /// rays, no optical path through a crystal, and the retardance is the
    /// number given rather than one that follows from a thickness and a
    /// birefringence at each wavelength.
    ///
    /// The retardance is the same at every angle of incidence. A real plate's
    /// moves as the square of the obliquity, which a zero-order plate barely
    /// notices and a multi-order one does not survive.
    ///
    /// Only meaningful on a standalone planar surface, whose local `+X` the
    /// fast axis is measured from.
    Retarder {
        /// Fast axis, degrees from the surface's local `+X`.
        axis_deg: f64,
        retardance_deg: f64,
    },
}

impl Default for SurfaceResponse {
    fn default() -> Self {
        Self::DielectricFresnel {
            ar_residual: SpectralScalar::Constant(1.0),
        }
    }
}

impl SurfaceResponse {
    pub(crate) fn validate(&self, field: &str) -> Result<(), SceneError> {
        let range = |name: &str, reason: &str| SceneError::OutOfRange {
            object: None,
            field: format!("{field}.{name}"),
            reason: reason.to_string(),
        };
        match self {
            Self::Absorber => Ok(()),
            Self::DielectricFresnel { ar_residual } => {
                ar_residual.validate(&format!("{field}.ar_residual"))?;
                if !ar_residual.is_unit_fraction() {
                    return Err(range("ar_residual", "is outside [0, 1]"));
                }
                Ok(())
            }
            Self::Mirror { reflectivity } => {
                reflectivity.validate(&format!("{field}.reflectivity"))?;
                if !reflectivity.is_unit_fraction() {
                    return Err(range("reflectivity", "is outside [0, 1]"));
                }
                Ok(())
            }
            Self::ExplicitRt {
                reflectance,
                transmittance,
            } => {
                reflectance.validate(&format!("{field}.reflectance"))?;
                transmittance.validate(&format!("{field}.transmittance"))?;
                if !reflectance.is_unit_fraction() {
                    return Err(range("reflectance", "is outside [0, 1]"));
                }
                if !transmittance.is_unit_fraction() {
                    return Err(range("transmittance", "is outside [0, 1]"));
                }
                // R + T <= 1: the remainder is absorbed, and a surface that
                // returns more than it received is a source. Both scalars are
                // piecewise linear in λ (a constant is flat), so R(λ) + T(λ)
                // attains its maximum at a sample point of one of the tables —
                // checking the union grid is exact. The earlier per-curve-maxima
                // bound rejected physical dichroic pairs, whose complementary
                // step curves peak at different wavelengths.
                let mut grid: Vec<f64> = Vec::new();
                for scalar in [reflectance, transmittance] {
                    if let SpectralScalar::Tabulated { lambda_um, .. } = scalar {
                        grid.extend_from_slice(lambda_um);
                    }
                }
                let exceeds = if grid.is_empty() {
                    reflectance.max() + transmittance.max() > 1.0 + 1e-12
                } else {
                    grid.iter().any(|&lam| {
                        // Out-of-range under the Error policy has no value at
                        // λ; bound it by the curve's own maximum (conservative
                        // only there, exact everywhere else).
                        let r = reflectance.sample(lam).unwrap_or_else(|| reflectance.max());
                        let t = transmittance
                            .sample(lam)
                            .unwrap_or_else(|| transmittance.max());
                        r + t > 1.0 + 1e-12
                    })
                };
                if exceeds {
                    return Err(range("reflectance + transmittance", "exceeds 1"));
                }
                Ok(())
            }
            Self::ThinLens { focal_length_mm } => {
                if !focal_length_mm.is_finite() || *focal_length_mm == 0.0 {
                    return Err(range("focal_length_mm", "must be finite and nonzero"));
                }
                Ok(())
            }
            Self::Coating { coating } => {
                // A layer is a thickness and an index; both have to be real
                // numbers or the transfer matrix has nothing to multiply. There
                // is deliberately no bound on R or T here, unlike `ExplicitRt`:
                // they are not stored, they are computed, and the calculation
                // cannot return more than it was given.
                for (i, layer) in coating.layers.iter().enumerate() {
                    let at = |name: &str| format!("{field}.coating.layers[{i}].{name}");
                    if !layer.thickness_nm.is_finite() || layer.thickness_nm < 0.0 {
                        return Err(SceneError::OutOfRange {
                            object: None,
                            field: at("thickness_nm"),
                            reason: "must be finite and not negative".to_string(),
                        });
                    }
                    layer.material.n.validate(&at("material.n"))?;
                    layer.material.k.validate(&at("material.k"))?;
                    if layer.material.n.min() <= 0.0 {
                        return Err(range(
                            &format!("coating.layers[{i}].material.n"),
                            "must be positive",
                        ));
                    }
                    if layer.material.k.min() < 0.0 {
                        return Err(range(
                            &format!("coating.layers[{i}].material.k"),
                            "must not be negative",
                        ));
                    }
                }
                if let crate::thinfilm::CoatingBacking::Plate { substrate, exit } = &coating.backing
                {
                    substrate
                        .n
                        .validate(&format!("{field}.coating.substrate.n"))?;
                    substrate
                        .k
                        .validate(&format!("{field}.coating.substrate.k"))?;
                    if substrate.n.min() <= 0.0 {
                        return Err(range("coating.substrate.n", "must be positive"));
                    }
                    if let crate::thinfilm::ExitFace::AntiReflective { residual } = exit {
                        if !residual.is_finite() || !(0.0..=1.0).contains(residual) {
                            return Err(range("coating.exit.residual", "is outside [0, 1]"));
                        }
                    }
                }
                Ok(())
            }
            Self::Polarizer {
                axis_deg,
                extinction,
            } => {
                if !axis_deg.is_finite() {
                    return Err(range("axis_deg", "must be finite"));
                }
                if !extinction.is_finite() || !(0.0..=1.0).contains(extinction) {
                    return Err(range("extinction", "is outside [0, 1]"));
                }
                Ok(())
            }
            Self::Retarder {
                axis_deg,
                retardance_deg,
            } => {
                if !axis_deg.is_finite() {
                    return Err(range("axis_deg", "must be finite"));
                }
                if !retardance_deg.is_finite() {
                    return Err(range("retardance_deg", "must be finite"));
                }
                Ok(())
            }
        }
    }
}

/// A surface's behaviour from each side.
///
/// Two sides, always, because a real coating is asymmetric and a model that
/// cannot say so quietly averages them.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TwoSidedSurfaceResponse {
    /// The incident ray is in the material outside the closed body.
    pub from_outside: SurfaceResponse,
    /// The incident ray is in the body's own bulk material.
    pub from_inside: SurfaceResponse,
}

impl TwoSidedSurfaceResponse {
    pub(crate) fn validate(&self, field: &str) -> Result<(), SceneError> {
        self.from_outside
            .validate(&format!("{field}.from_outside"))?;
        self.from_inside.validate(&format!("{field}.from_inside"))
    }
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// A finite region of a plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlanarShape2 {
    Rectangle {
        half_width_mm: f64,
        half_height_mm: f64,
    },
    Disc {
        radius_mm: f64,
    },
    Annulus {
        inner_mm: f64,
        outer_mm: f64,
    },
    Polygon {
        vertices: Vec<Vec2>,
    },
}

impl PlanarShape2 {
    fn validate(&self, object: Option<u64>, field: &str) -> Result<(), SceneError> {
        let nf = || SceneError::NonFinite {
            object,
            field: field.to_string(),
        };
        let range = |reason: &str| SceneError::OutOfRange {
            object,
            field: field.to_string(),
            reason: reason.to_string(),
        };
        match self {
            Self::Rectangle {
                half_width_mm,
                half_height_mm,
            } => {
                if !half_width_mm.is_finite() || !half_height_mm.is_finite() {
                    return Err(nf());
                }
                if *half_width_mm <= 0.0 || *half_height_mm <= 0.0 {
                    return Err(range("has a non-positive extent"));
                }
                Ok(())
            }
            Self::Disc { radius_mm } => {
                if !radius_mm.is_finite() {
                    return Err(nf());
                }
                if *radius_mm <= 0.0 {
                    return Err(range("has a non-positive radius"));
                }
                Ok(())
            }
            Self::Annulus { inner_mm, outer_mm } => {
                if !inner_mm.is_finite() || !outer_mm.is_finite() {
                    return Err(nf());
                }
                if *inner_mm < 0.0 {
                    return Err(range("has a negative inner radius"));
                }
                if *outer_mm <= *inner_mm {
                    return Err(range("has outer radius <= inner radius"));
                }
                Ok(())
            }
            Self::Polygon { vertices } => {
                if vertices.iter().any(|v| !v.is_finite()) {
                    return Err(nf());
                }
                if vertices.len() < 3 {
                    return Err(range("has fewer than three vertices"));
                }
                Ok(())
            }
        }
    }
}

/// The sag of a surface of revolution about its local axis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SagModel {
    Plane,
    Conic {
        curvature_per_mm: f64,
        conic: f64,
    },
    EvenAsphere {
        curvature_per_mm: f64,
        conic: f64,
        /// A4, A6, ... in the Canvas convention.
        coeffs: [f64; 8],
    },
}

impl SagModel {
    fn validate(&self, object: Option<u64>, field: &str) -> Result<(), SceneError> {
        let ok = match self {
            Self::Plane => true,
            Self::Conic {
                curvature_per_mm,
                conic,
            } => curvature_per_mm.is_finite() && conic.is_finite(),
            Self::EvenAsphere {
                curvature_per_mm,
                conic,
                coeffs,
            } => {
                curvature_per_mm.is_finite()
                    && conic.is_finite()
                    && coeffs.iter().all(|c| c.is_finite())
            }
        };
        if ok {
            Ok(())
        } else {
            Err(SceneError::NonFinite {
                object,
                field: field.to_string(),
            })
        }
    }
}

/// The radial extent of a surface of revolution. `inner > 0` is an annulus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RadialAperture {
    pub inner_radius_mm: f64,
    pub outer_radius_mm: f64,
}

impl RadialAperture {
    fn validate(&self, object: Option<u64>, field: &str) -> Result<(), SceneError> {
        if !self.inner_radius_mm.is_finite() || !self.outer_radius_mm.is_finite() {
            return Err(SceneError::NonFinite {
                object,
                field: field.to_string(),
            });
        }
        if self.inner_radius_mm < 0.0 || self.outer_radius_mm <= self.inner_radius_mm {
            return Err(SceneError::OutOfRange {
                object,
                field: field.to_string(),
                reason: "radial aperture is empty or negative".to_string(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

/// What every placed object carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectCommon {
    pub id: ObjectId,
    pub transform: Isometry3,
    pub layer: LayerId,
    pub locked: bool,
    pub label: String,
    pub visible: bool,
}

impl ObjectCommon {
    pub fn new(id: ObjectId, layer: LayerId) -> Self {
        Self {
            id,
            transform: Isometry3::IDENTITY,
            layer,
            locked: false,
            label: String::new(),
            visible: true,
        }
    }
}

/// A closed solid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SolidGeometry3 {
    Profile(ProfileSolid3),
    Sphere {
        radius_mm: f64,
    },
    Box {
        half_extents_mm: Vec3,
    },
    Cylinder {
        radius_mm: f64,
        half_length_mm: f64,
        axis_local: Vec3,
    },
    // Triangle meshes are deliberately absent. Media resolve by containment plus
    // priority, so a leaky mesh is not a rendering artefact but a wrong
    // refractive index; they arrive with a watertightness specification or not at
    // all (§6.2).
}

/// A per-patch override of a body's default surface response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchResponseOverride {
    pub patch: PatchId,
    pub response: TwoSidedSurfaceResponse,
}

/// A body's surface behaviour: one default plus sparse per-patch overrides.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SurfaceProps3 {
    pub default: TwoSidedSurfaceResponse,
    pub overrides: Vec<PatchResponseOverride>,
}

impl SurfaceProps3 {
    /// The response for `patch`, falling back to the default.
    pub fn for_patch(&self, patch: PatchId) -> &TwoSidedSurfaceResponse {
        self.overrides
            .iter()
            .find(|o| o.patch == patch)
            .map(|o| &o.response)
            .unwrap_or(&self.default)
    }
}

/// A volumetric body with a bulk material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Body3 {
    pub common: ObjectCommon,
    pub geometry: SolidGeometry3,
    pub material: Material,
    pub absorption: AbsorptionModel,
    pub surface: SurfaceProps3,
    /// Nesting: the medium at a point is the material of the highest-priority
    /// body containing it.
    pub priority: i32,
}

impl Body3 {
    /// This body's id, as a body.
    pub fn id(&self) -> BodyId {
        BodyId(self.common.id.0)
    }

    /// The construction rule, if this body is profile-authored.
    pub fn construction(&self) -> Option<ProfileConstruction3> {
        match &self.geometry {
            SolidGeometry3::Profile(p) => Some(p.construction),
            _ => None,
        }
    }

    /// Can this body be traced in full 3D?
    ///
    /// False only for a profile body whose third dimension is unresolved. An
    /// analytic solid is 3D by construction.
    pub fn is_3d(&self) -> bool {
        match &self.geometry {
            SolidGeometry3::Profile(p) => p.construction.is_3d(),
            _ => true,
        }
    }
}

/// A zero-thickness surface.
///
/// Allowed for mirrors, scalar filters and beamsplitters, absorber sheets, and
/// probes. **Not** for a refractive sheet with different bulk media on its
/// sides: that needs a medium-region contract that does not exist, so ordinary
/// refractive optics are closed bodies (§6.6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceObject3 {
    pub common: ObjectCommon,
    pub geometry: SurfaceGeometry3,
    pub response_front: SurfaceResponse,
    pub response_back: SurfaceResponse,
}

impl SurfaceObject3 {
    pub fn id(&self) -> SurfaceObjectId {
        SurfaceObjectId(self.common.id.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SurfaceGeometry3 {
    Plane {
        shape: PlanarShape2,
    },
    SurfaceOfRevolution {
        sag: SagModel,
        radial_aperture: RadialAperture,
    },
}

/// A finite opening in an opaque mask.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Aperture3 {
    pub common: ObjectCommon,
    pub shape: ApertureShape3,
    pub interaction: ApertureInteraction,
    /// Stop designation is **metadata**. It selects which aperture the pupil,
    /// chief, and marginal diagnostics are computed about, and changes no ray.
    /// Physical clipping is [`ApertureShape3`], and the two are kept apart on
    /// purpose (§6.8).
    pub primary_stop_for: Vec<PathIntentId>,
}

impl Aperture3 {
    pub fn id(&self) -> ApertureId {
        ApertureId(self.common.id.0)
    }
}

/// The opening and the mask around it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApertureShape3 {
    pub plane_local: Plane3,
    /// A ray inside this passes with no event.
    pub clear_opening: PlanarShape2,
    /// A ray outside the opening but inside this is absorbed. `None` means the
    /// mask is unbounded in its plane.
    pub outer_extent: Option<PlanarShape2>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApertureInteraction {
    Absorb,
}

/// An area detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detector3 {
    pub common: ObjectCommon,
    pub geometry: DetectorGeometry3,
    /// Cell counts across the two local axes.
    pub bins: [usize; 2],
    pub response: DetectorResponse,
    pub termination: DetectorTermination,
}

impl Detector3 {
    pub fn id(&self) -> DetectorId {
        DetectorId(self.common.id.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DetectorGeometry3 {
    Rectangle {
        half_width_mm: f64,
        half_height_mm: f64,
    },
    Disc {
        radius_mm: f64,
    },
    Annulus {
        inner_mm: f64,
        outer_mm: f64,
    },

    /// **v1 compatibility only. Not an area detector.**
    ///
    /// A line has no area, so its bins have no `mm²` and its readings are not a
    /// flux density. v1 detectors migrate to this and are labelled it, rather
    /// than being given a height nobody authored (§3.2).
    LineProbeLegacy {
        a_local: Vec3,
        b_local: Vec3,
        bins: usize,
    },
}

impl DetectorGeometry3 {
    /// Does this geometry have a real area?
    pub fn has_area(&self) -> bool {
        !matches!(self, Self::LineProbeLegacy { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum DetectorResponse {
    /// Every wavelength detected equally.
    #[default]
    Unity,
    Tabulated {
        lambda_um: Vec<f64>,
        efficiency: Vec<f64>,
        interpolation: Interpolation,
        out_of_range: OutOfRangePolicy,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorTermination {
    /// The branch ends here.
    Terminate,
    /// Record and carry on, unchanged in direction and flux.
    ProbeAndContinue,
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source3 {
    pub id: SourceId,
    pub transform: Isometry3,
    pub spatial: SpatialDistribution3,
    pub angular: AngularDistribution3,
    pub spectrum: crate::scene::Spectrum,
    /// Total emitted flux, distributed across samples. Relative unless the
    /// source is explicitly calibrated, and nothing here calibrates one.
    pub flux: f64,
    pub samples: SourceSampling,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpatialDistribution3 {
    Point,
    Rectangle {
        half_width_mm: f64,
        half_height_mm: f64,
    },
    Disc {
        radius_mm: f64,
    },

    /// **v1 compatibility.** A line in the view plane: a degenerate 3D
    /// distribution with no area.
    LineLegacy {
        half_width_mm: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AngularDistribution3 {
    Direction {
        local_dir: Vec3,
    },
    /// Uniform in **solid angle** within a cone.
    UniformConeSolidAngle {
        half_angle_rad: f64,
    },
    LambertianHemisphere,

    /// **v1 compatibility.** An equal-angle fan in one plane: a degenerate 3D
    /// distribution whose measure exists only in that plane. It is not a cone,
    /// and it is not uniform in solid angle.
    PlanarFanLegacy {
        half_angle_rad: f64,
    },
}

impl SpatialDistribution3 {
    pub fn is_legacy(&self) -> bool {
        matches!(self, Self::LineLegacy { .. })
    }
}

impl AngularDistribution3 {
    pub fn is_legacy(&self) -> bool {
        matches!(self, Self::PlanarFanLegacy { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSampling {
    pub spatial_samples: u32,
    pub angular_samples: u32,
    /// Sampling is deterministic for a fixed seed and configuration.
    pub seed: u64,
    pub sequence: SamplingSequence,
}

impl Default for SourceSampling {
    fn default() -> Self {
        Self {
            spatial_samples: 1,
            angular_samples: 32,
            seed: 0,
            sequence: SamplingSequence::Grid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingSequence {
    Grid,
    Stratified,
    Sobol,
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct View3 {
    pub id: ViewId,
    pub name: String,
    pub mode: ViewMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ViewMode {
    Slice { plane: Plane3, thickness_mm: f64 },
    Orthographic { camera: Camera3 },
    Perspective { camera: Camera3 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Camera3 {
    pub transform: Isometry3,
    pub vertical_size_mm: f64,
    pub near_mm: f64,
    pub far_mm: f64,
    pub vertical_fov_rad: Option<f64>,
}

// ---------------------------------------------------------------------------
// Environment and intent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Environment3 {
    /// Ambient refractive index. A spectral ambient is a future additive change.
    pub ambient_index: f64,
    pub ambient_absorption: AbsorptionModel,
    /// Render/reference wavelength set `(lambda_um, weight)`.
    pub wavelengths: Vec<(f64, f64)>,
}

impl Default for Environment3 {
    fn default() -> Self {
        Self {
            ambient_index: 1.0,
            ambient_absorption: AbsorptionModel::None,
            wavelengths: vec![(crate::material::LAMBDA_D, 1.0)],
        }
    }
}

/// What the scene is *for*: which source is meant to reach which detector.
///
/// Diagnostics only. Nothing here reaches the tracer.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SceneIntent {
    pub primary_paths: Vec<PrimaryPathIntent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimaryPathIntent {
    pub id: PathIntentId,
    pub source: SourceId,
    pub detector: DetectorId,
    /// At most one nominal stop, and it is metadata (see [`Aperture3`]).
    pub stop: Option<ApertureId>,
    pub label: String,
}

// ---------------------------------------------------------------------------
// The scene
// ---------------------------------------------------------------------------

fn default_format() -> String {
    OCANVAS_FORMAT.to_string()
}
fn default_v2_version() -> u32 {
    OCANVAS_V2_VERSION
}
fn default_dimension() -> u8 {
    OCANVAS_V2_DIMENSION
}

/// The canonical 3D scene. The serialized form *is* the `.ocanvas` v2 document.
///
/// No `PartialEq`, matching v1's [`Scene`](crate::scene::Scene): [`GlassCatalog`]
/// has none, and a scene is compared through its JSON anyway. That is the
/// stronger check for a document format, since it tests what actually round
/// trips rather than what happens to be in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene3 {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_v2_version")]
    pub version: u32,
    #[serde(default = "default_dimension")]
    pub dimension: u8,
    pub bodies: Vec<Body3>,
    pub surfaces: Vec<SurfaceObject3>,
    pub apertures: Vec<Aperture3>,
    pub detectors: Vec<Detector3>,
    pub sources: Vec<Source3>,
    pub views: Vec<View3>,
    pub env: Environment3,
    pub layers: Vec<Layer>,
    pub catalog: GlassCatalog,
    pub intent: SceneIntent,
    /// The next id to hand out.
    ///
    /// Persisted so a save/load round trip cannot re-issue an id that a path
    /// record, an undo snapshot, or an external reference still names. Load
    /// recomputes it as `max(existing) + 1` when the field is absent or too low,
    /// which is the repair a hand-edited document needs and the one place this
    /// module repairs rather than refuses: a `next_id` below the high-water mark
    /// is not a statement of intent, it is bookkeeping.
    #[serde(default)]
    pub next_id: u64,
}

impl Default for Scene3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene3 {
    pub fn new() -> Self {
        Self {
            format: default_format(),
            version: OCANVAS_V2_VERSION,
            dimension: OCANVAS_V2_DIMENSION,
            bodies: Vec::new(),
            surfaces: Vec::new(),
            apertures: Vec::new(),
            detectors: Vec::new(),
            sources: Vec::new(),
            views: Vec::new(),
            env: Environment3::default(),
            layers: vec![Layer::default()],
            catalog: GlassCatalog::builtin(),
            intent: SceneIntent::default(),
            next_id: 0,
        }
    }

    /// Hand out a fresh id.
    ///
    /// The only way an id is minted. It is never a vector index, never reused,
    /// and monotonic for the scene's lifetime.
    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn alloc_object_id(&mut self) -> ObjectId {
        ObjectId(self.alloc_id())
    }
    pub fn alloc_patch_id(&mut self) -> PatchId {
        PatchId(self.alloc_id())
    }
    pub fn alloc_source_id(&mut self) -> SourceId {
        SourceId(self.alloc_id())
    }
    pub fn alloc_view_id(&mut self) -> ViewId {
        ViewId(self.alloc_id())
    }
    pub fn alloc_layer_id(&mut self) -> LayerId {
        LayerId(self.alloc_id())
    }
    pub fn alloc_path_intent_id(&mut self) -> PathIntentId {
        PathIntentId(self.alloc_id())
    }

    /// Every id the document uses, in a stable order.
    fn all_ids(&self) -> Vec<u64> {
        let mut v = Vec::new();
        for b in &self.bodies {
            v.push(b.common.id.0);
            if let SolidGeometry3::Profile(p) = &b.geometry {
                v.extend(p.profile.edges.iter().map(|e| e.patch.0));
            }
        }
        v.extend(self.surfaces.iter().map(|s| s.common.id.0));
        v.extend(self.apertures.iter().map(|a| a.common.id.0));
        v.extend(self.detectors.iter().map(|d| d.common.id.0));
        v.extend(self.sources.iter().map(|s| s.id.0));
        v.extend(self.views.iter().map(|x| x.id.0));
        v.extend(self.intent.primary_paths.iter().map(|p| p.id.0));
        v
    }

    /// Find a body by id.
    pub fn body(&self, id: BodyId) -> Option<&Body3> {
        self.bodies.iter().find(|b| b.common.id.0 == id.0)
    }

    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut Body3> {
        self.bodies.iter_mut().find(|b| b.common.id.0 == id.0)
    }

    pub fn detector(&self, id: DetectorId) -> Option<&Detector3> {
        self.detectors.iter().find(|d| d.common.id.0 == id.0)
    }

    pub fn source(&self, id: SourceId) -> Option<&Source3> {
        self.sources.iter().find(|s| s.id == id)
    }

    pub fn aperture(&self, id: ApertureId) -> Option<&Aperture3> {
        self.apertures.iter().find(|a| a.common.id.0 == id.0)
    }

    /// Resolve a surface reference to its body and patch.
    pub fn surface_ref(&self, r: SurfaceRef) -> Option<(&Body3, PatchId)> {
        let b = self.bodies.iter().find(|b| b.common.id == r.object)?;
        match &b.geometry {
            SolidGeometry3::Profile(p) if p.profile.edges.iter().any(|e| e.patch == r.patch) => {
                Some((b, r.patch))
            }
            _ => None,
        }
    }

    /// Serialize to pretty v2 JSON, stamping the header.
    pub fn to_json(&self) -> String {
        let mut doc = self.clone();
        doc.format = default_format();
        doc.version = OCANVAS_V2_VERSION;
        doc.dimension = OCANVAS_V2_DIMENSION;
        serde_json::to_string_pretty(&doc).expect("scene3 serializes")
    }

    /// Parse a v2 document. Use [`crate::migrate_v1::load_any`] to accept v1 too.
    pub fn from_json(s: &str) -> Result<Self, SceneError> {
        let scene: Scene3 = serde_json::from_str(s).map_err(|e| SceneError::Json(e.to_string()))?;
        scene.check_header()?;
        let mut scene = scene;
        scene.repair_next_id();
        scene.validate()?;
        Ok(scene)
    }

    pub fn check_header(&self) -> Result<(), SceneError> {
        if self.format != OCANVAS_FORMAT {
            return Err(SceneError::WrongFormat {
                found: self.format.clone(),
            });
        }
        if self.version != OCANVAS_V2_VERSION {
            return Err(SceneError::UnsupportedVersion {
                found: self.version,
            });
        }
        if self.dimension != OCANVAS_V2_DIMENSION {
            return Err(SceneError::WrongDimension {
                found: self.dimension,
            });
        }
        Ok(())
    }

    /// Lift `next_id` above every id in the document.
    ///
    /// A document written by this build already satisfies this. One that was
    /// hand-edited, or produced by a tool that did not track the counter, would
    /// otherwise re-issue a live id on the next edit.
    pub fn repair_next_id(&mut self) {
        let high = self.all_ids().into_iter().max();
        if let Some(h) = high {
            self.next_id = self.next_id.max(h + 1);
        }
    }

    /// Rigidly move a batch of objects by `delta` (CV-C).
    ///
    /// `ids` may name bodies, standalone surfaces, apertures, detectors, and
    /// sources in any mix — a dragged module transforms all of its objects in
    /// one call, and each object composes `delta` onto the transform it already
    /// carries, so the batch preserves its relative geometry exactly. This is
    /// the 3D sibling of the 2D document's `transform_objects`.
    ///
    /// Atomic: every id is resolved before anything moves; an unknown id is a
    /// [`SceneError::DanglingRef`] that mutates nothing. A duplicated id in
    /// `ids` applies once, not twice — the caller's list is a set of targets,
    /// not a multiplier.
    pub fn transform_objects(&mut self, ids: &[u64], delta: Isometry3) -> Result<(), SceneError> {
        #[derive(Clone, Copy)]
        enum Slot {
            Body(usize),
            Surface(usize),
            Aperture(usize),
            Detector(usize),
            Source(usize),
        }
        let mut slots: Vec<Slot> = Vec::with_capacity(ids.len());
        let mut seen = std::collections::BTreeSet::new();
        for &id in ids {
            if !seen.insert(id) {
                continue;
            }
            let slot = (self.bodies.iter())
                .position(|b| b.common.id.raw() == id)
                .map(Slot::Body)
                .or_else(|| {
                    (self.surfaces.iter())
                        .position(|s| s.common.id.raw() == id)
                        .map(Slot::Surface)
                })
                .or_else(|| {
                    (self.apertures.iter())
                        .position(|a| a.common.id.raw() == id)
                        .map(Slot::Aperture)
                })
                .or_else(|| {
                    (self.detectors.iter())
                        .position(|d| d.common.id.raw() == id)
                        .map(Slot::Detector)
                })
                .or_else(|| {
                    (self.sources.iter())
                        .position(|s| s.id.raw() == id)
                        .map(Slot::Source)
                });
            match slot {
                Some(s) => slots.push(s),
                None => {
                    return Err(SceneError::DanglingRef {
                        what: "transform target",
                        id,
                    })
                }
            }
        }
        for slot in slots {
            let t = match slot {
                Slot::Body(i) => &mut self.bodies[i].common.transform,
                Slot::Surface(i) => &mut self.surfaces[i].common.transform,
                Slot::Aperture(i) => &mut self.apertures[i].common.transform,
                Slot::Detector(i) => &mut self.detectors[i].common.transform,
                Slot::Source(i) => &mut self.sources[i].transform,
            };
            *t = delta.compose(*t);
        }
        Ok(())
    }

    /// Full structural and physical validation.
    pub fn validate(&self) -> Result<(), SceneError> {
        self.check_header()?;
        self.validate_ids()?;
        self.validate_env()?;
        self.validate_layers()?;
        self.validate_bodies()?;
        self.validate_surfaces()?;
        self.validate_apertures()?;
        self.validate_detectors()?;
        self.validate_sources()?;
        self.validate_views()?;
        self.validate_intent()?;
        Ok(())
    }

    fn validate_ids(&self) -> Result<(), SceneError> {
        let mut seen = HashSet::new();
        for id in self.all_ids() {
            if !seen.insert(id) {
                return Err(SceneError::DuplicateId { id });
            }
        }
        if let Some(h) = seen.iter().copied().max() {
            if self.next_id <= h {
                return Err(SceneError::OutOfRange {
                    object: None,
                    field: "next_id".to_string(),
                    reason: format!("is {} but id {h} is in use", self.next_id),
                });
            }
        }
        Ok(())
    }

    fn validate_env(&self) -> Result<(), SceneError> {
        if !self.env.ambient_index.is_finite() {
            return Err(SceneError::NonFinite {
                object: None,
                field: "env.ambient_index".to_string(),
            });
        }
        if self.env.ambient_index <= 0.0 {
            return Err(SceneError::OutOfRange {
                object: None,
                field: "env.ambient_index".to_string(),
                reason: "is not strictly positive".to_string(),
            });
        }
        self.env
            .ambient_absorption
            .validate("env.ambient_absorption")?;
        for (l, w) in &self.env.wavelengths {
            if !l.is_finite() || !w.is_finite() {
                return Err(SceneError::NonFinite {
                    object: None,
                    field: "env.wavelengths".to_string(),
                });
            }
            if *l <= 0.0 {
                return Err(SceneError::OutOfRange {
                    object: None,
                    field: "env.wavelengths".to_string(),
                    reason: "wavelength is not positive".to_string(),
                });
            }
            if *w < 0.0 {
                return Err(SceneError::OutOfRange {
                    object: None,
                    field: "env.wavelengths".to_string(),
                    reason: "weight is negative".to_string(),
                });
            }
        }
        Ok(())
    }

    fn validate_layers(&self) -> Result<(), SceneError> {
        if self.layers.is_empty() {
            return Err(SceneError::OutOfRange {
                object: None,
                field: "layers".to_string(),
                reason: "a scene needs at least one layer".to_string(),
            });
        }
        Ok(())
    }

    /// v1 `LayerId` is an index into `layers`, and v2 keeps that for layers
    /// alone: they are an ordered presentation stack, not placed objects.
    fn check_layer(&self, layer: LayerId, object: u64) -> Result<(), SceneError> {
        if layer.0 as usize >= self.layers.len() {
            return Err(SceneError::DanglingRef {
                what: "layer",
                id: layer.0,
            });
        }
        let _ = object;
        Ok(())
    }

    fn check_common(&self, c: &ObjectCommon) -> Result<(), SceneError> {
        if !c.transform.is_finite() {
            return Err(SceneError::NonFinite {
                object: Some(c.id.0),
                field: "transform".to_string(),
            });
        }
        // Isometry3's own deserializer already refuses a bad quaternion, so this
        // catches only a value built in memory and never serialized.
        let n: f64 = c
            .transform
            .rot_xyzw
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if (n - 1.0).abs() > 1e-9 {
            return Err(SceneError::InvalidQuaternion { object: c.id.0 });
        }
        self.check_layer(c.layer, c.id.0)
    }

    fn validate_bodies(&self) -> Result<(), SceneError> {
        for b in &self.bodies {
            self.check_common(&b.common)?;
            let oid = Some(b.common.id.0);
            match &b.geometry {
                SolidGeometry3::Profile(p) => {
                    if !p.profile.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry.profile".to_string(),
                        });
                    }
                    if !p.construction.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry.construction".to_string(),
                        });
                    }
                    if p.profile.edges.is_empty() {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "geometry.profile".to_string(),
                            reason: "has no edges".to_string(),
                        });
                    }
                    if let ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } =
                        p.construction
                    {
                        if z_min_mm >= z_max_mm {
                            return Err(SceneError::OutOfRange {
                                object: oid,
                                field: "geometry.construction".to_string(),
                                reason: "has z_min >= z_max, which bounds no volume".to_string(),
                            });
                        }
                    }
                }
                SolidGeometry3::Sphere { radius_mm } => {
                    if !radius_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry.radius_mm".to_string(),
                        });
                    }
                    if *radius_mm <= 0.0 {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "geometry.radius_mm".to_string(),
                            reason: "is not positive".to_string(),
                        });
                    }
                }
                SolidGeometry3::Box { half_extents_mm } => {
                    if !half_extents_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry.half_extents_mm".to_string(),
                        });
                    }
                    if half_extents_mm.min_element() <= 0.0 {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "geometry.half_extents_mm".to_string(),
                            reason: "has a non-positive extent".to_string(),
                        });
                    }
                }
                SolidGeometry3::Cylinder {
                    radius_mm,
                    half_length_mm,
                    axis_local,
                } => {
                    if !radius_mm.is_finite()
                        || !half_length_mm.is_finite()
                        || !axis_local.is_finite()
                    {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry".to_string(),
                        });
                    }
                    if *radius_mm <= 0.0 || *half_length_mm <= 0.0 {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "geometry".to_string(),
                            reason: "has a non-positive radius or length".to_string(),
                        });
                    }
                    if axis_local.length() < crate::vec3::MIN_NORM {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "geometry.axis_local".to_string(),
                            reason: "names no direction".to_string(),
                        });
                    }
                }
            }
            b.absorption.validate("absorption")?;
            b.surface.default.validate("surface.default")?;
            for o in &b.surface.overrides {
                o.response.validate("surface.override")?;
                // An override naming a patch this body does not have is a stale
                // reference, which is the whole failure class stable ids exist to
                // remove. It is a load error rather than a silently ignored entry.
                let known = match &b.geometry {
                    SolidGeometry3::Profile(p) => {
                        p.profile.edges.iter().any(|e| e.patch == o.patch)
                    }
                    _ => false,
                };
                if !known {
                    return Err(SceneError::DanglingRef {
                        what: "patch",
                        id: o.patch.0,
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_surfaces(&self) -> Result<(), SceneError> {
        for s in &self.surfaces {
            self.check_common(&s.common)?;
            let oid = Some(s.common.id.0);
            match &s.geometry {
                SurfaceGeometry3::Plane { shape } => shape.validate(oid, "geometry.shape")?,
                SurfaceGeometry3::SurfaceOfRevolution {
                    sag,
                    radial_aperture,
                } => {
                    sag.validate(oid, "geometry.sag")?;
                    radial_aperture.validate(oid, "geometry.radial_aperture")?;
                }
            }
            s.response_front.validate("response_front")?;
            s.response_back.validate("response_back")?;
        }
        Ok(())
    }

    fn validate_apertures(&self) -> Result<(), SceneError> {
        let intents: HashSet<u64> = self.intent.primary_paths.iter().map(|p| p.id.0).collect();
        for a in &self.apertures {
            self.check_common(&a.common)?;
            let oid = Some(a.common.id.0);
            a.shape.clear_opening.validate(oid, "shape.clear_opening")?;
            if let Some(o) = &a.shape.outer_extent {
                o.validate(oid, "shape.outer_extent")?;
            }
            for pi in &a.primary_stop_for {
                if !intents.contains(&pi.0) {
                    return Err(SceneError::DanglingRef {
                        what: "path intent",
                        id: pi.0,
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_detectors(&self) -> Result<(), SceneError> {
        for d in &self.detectors {
            self.check_common(&d.common)?;
            let oid = Some(d.common.id.0);
            let range = |field: &str, reason: &str| SceneError::OutOfRange {
                object: oid,
                field: field.to_string(),
                reason: reason.to_string(),
            };
            match &d.geometry {
                DetectorGeometry3::Rectangle {
                    half_width_mm,
                    half_height_mm,
                } => {
                    if !half_width_mm.is_finite() || !half_height_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry".to_string(),
                        });
                    }
                    if *half_width_mm <= 0.0 || *half_height_mm <= 0.0 {
                        return Err(range("geometry", "has a non-positive extent"));
                    }
                }
                DetectorGeometry3::Disc { radius_mm } => {
                    if !radius_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry.radius_mm".to_string(),
                        });
                    }
                    if *radius_mm <= 0.0 {
                        return Err(range("geometry.radius_mm", "is not positive"));
                    }
                }
                DetectorGeometry3::Annulus { inner_mm, outer_mm } => {
                    if !inner_mm.is_finite() || !outer_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry".to_string(),
                        });
                    }
                    if *inner_mm < 0.0 || *outer_mm <= *inner_mm {
                        return Err(range("geometry", "is an empty annulus"));
                    }
                }
                DetectorGeometry3::LineProbeLegacy {
                    a_local,
                    b_local,
                    bins,
                } => {
                    if !a_local.is_finite() || !b_local.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry".to_string(),
                        });
                    }
                    if (*b_local - *a_local).length() <= 0.0 {
                        return Err(range("geometry", "is a zero-length probe"));
                    }
                    if *bins == 0 {
                        return Err(range("geometry.bins", "is zero"));
                    }
                }
            }
            if d.geometry.has_area() && (d.bins[0] == 0 || d.bins[1] == 0) {
                return Err(range("bins", "is zero on an axis"));
            }
            if let DetectorResponse::Tabulated {
                lambda_um,
                efficiency,
                ..
            } = &d.response
            {
                validate_table("response", lambda_um, efficiency)?;
                if efficiency.iter().any(|e| !(0.0..=1.0).contains(e)) {
                    return Err(SceneError::InvalidTable {
                        field: "response".to_string(),
                        reason: "quantum efficiency is outside [0, 1]".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_sources(&self) -> Result<(), SceneError> {
        for s in &self.sources {
            let oid = Some(s.id.0);
            if !s.transform.is_finite() {
                return Err(SceneError::NonFinite {
                    object: oid,
                    field: "transform".to_string(),
                });
            }
            if !s.flux.is_finite() {
                return Err(SceneError::NonFinite {
                    object: oid,
                    field: "flux".to_string(),
                });
            }
            if s.flux < 0.0 {
                return Err(SceneError::OutOfRange {
                    object: oid,
                    field: "flux".to_string(),
                    reason: "is negative".to_string(),
                });
            }
            let range = |field: &str, reason: &str| SceneError::OutOfRange {
                object: oid,
                field: field.to_string(),
                reason: reason.to_string(),
            };
            match s.spatial {
                SpatialDistribution3::Point => {}
                SpatialDistribution3::Rectangle {
                    half_width_mm,
                    half_height_mm,
                } => {
                    if !half_width_mm.is_finite() || !half_height_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "spatial".to_string(),
                        });
                    }
                    if half_width_mm <= 0.0 || half_height_mm <= 0.0 {
                        return Err(range("spatial", "has a non-positive extent"));
                    }
                }
                SpatialDistribution3::Disc { radius_mm } => {
                    if !radius_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "spatial.radius_mm".to_string(),
                        });
                    }
                    if radius_mm <= 0.0 {
                        return Err(range("spatial.radius_mm", "is not positive"));
                    }
                }
                SpatialDistribution3::LineLegacy { half_width_mm } => {
                    if !half_width_mm.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "spatial.half_width_mm".to_string(),
                        });
                    }
                    if half_width_mm < 0.0 {
                        return Err(range("spatial.half_width_mm", "is negative"));
                    }
                }
            }
            match s.angular {
                AngularDistribution3::Direction { local_dir } => {
                    if !local_dir.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "angular.local_dir".to_string(),
                        });
                    }
                    if local_dir.length() < crate::vec3::MIN_NORM {
                        return Err(range("angular.local_dir", "names no direction"));
                    }
                }
                AngularDistribution3::UniformConeSolidAngle { half_angle_rad }
                | AngularDistribution3::PlanarFanLegacy { half_angle_rad } => {
                    if !half_angle_rad.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "angular.half_angle_rad".to_string(),
                        });
                    }
                    if !(0.0..=std::f64::consts::PI).contains(&half_angle_rad) {
                        return Err(range("angular.half_angle_rad", "is outside [0, pi]"));
                    }
                }
                AngularDistribution3::LambertianHemisphere => {}
            }
            for (l, w) in s.spectrum.samples() {
                if !l.is_finite() || !w.is_finite() || l <= 0.0 {
                    return Err(SceneError::OutOfRange {
                        object: oid,
                        field: "spectrum".to_string(),
                        reason: "has a non-finite or non-positive wavelength".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_views(&self) -> Result<(), SceneError> {
        for v in &self.views {
            let oid = Some(v.id.0);
            match v.mode {
                ViewMode::Slice { thickness_mm, .. } => {
                    if !thickness_mm.is_finite() || thickness_mm < 0.0 {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "mode.thickness_mm".to_string(),
                            reason: "is negative or not finite".to_string(),
                        });
                    }
                }
                ViewMode::Orthographic { camera } | ViewMode::Perspective { camera } => {
                    if !camera.transform.is_finite()
                        || !camera.vertical_size_mm.is_finite()
                        || !camera.near_mm.is_finite()
                        || !camera.far_mm.is_finite()
                    {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "mode.camera".to_string(),
                        });
                    }
                    if camera.near_mm >= camera.far_mm {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "mode.camera".to_string(),
                            reason: "has near >= far".to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_intent(&self) -> Result<(), SceneError> {
        let sources: HashSet<u64> = self.sources.iter().map(|s| s.id.0).collect();
        let detectors: HashSet<u64> = self.detectors.iter().map(|d| d.common.id.0).collect();
        let apertures: HashSet<u64> = self.apertures.iter().map(|a| a.common.id.0).collect();
        for p in &self.intent.primary_paths {
            if !sources.contains(&p.source.0) {
                return Err(SceneError::DanglingRef {
                    what: "source",
                    id: p.source.0,
                });
            }
            if !detectors.contains(&p.detector.0) {
                return Err(SceneError::DanglingRef {
                    what: "detector",
                    id: p.detector.0,
                });
            }
            if let Some(stop) = p.stop {
                if !apertures.contains(&stop.0) {
                    return Err(SceneError::DanglingRef {
                        what: "aperture",
                        id: stop.0,
                    });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Promotion (§6.4, §6.5)
// ---------------------------------------------------------------------------

/// Whether a body can take a construction rule, and why not if it cannot.
#[derive(Debug, Clone, PartialEq)]
pub enum PromotionCheck {
    /// Every check passed: the construction produces a closed body.
    Ok,
    /// Refused, with the reason.
    No(String),
    /// A check that could not be decided.
    ///
    /// The revolve and extrude validators run in full (`crate::geom3::profile`),
    /// so neither returns this today. It is kept for a future construction whose
    /// validation is genuinely partial, rather than letting such a case
    /// masquerade as [`Ok`](PromotionCheck::Ok).
    Undecided(String),
}

impl PromotionCheck {
    pub fn is_refusal(&self) -> bool {
        matches!(self, PromotionCheck::No(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PromotionError {
    UnknownBody(u64),
    /// The body is not profile-authored, so it has no profile to construct from.
    NotProfileAuthored(u64),
    Refused(String),
}

impl std::fmt::Display for PromotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownBody(id) => write!(f, "body {id} does not exist"),
            Self::NotProfileAuthored(id) => {
                write!(f, "body {id} is an analytic solid and needs no promotion")
            }
            Self::Refused(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for PromotionError {}

/// What a promotion did.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionReport {
    pub body: BodyId,
    pub from: ProfileConstruction3,
    pub to: ProfileConstruction3,
    /// Checks that could not be decided. Empty now that the revolve and extrude
    /// validators run in full.
    pub undecided: Vec<String>,
}

/// Can `body` be revolved about local `+X`? (§6.4)
///
/// The full geometric test now runs (`crate::geom3::profile::validate_revolvable`):
/// the loop closes, is mirror-symmetric about local `Y = 0`, keeps every conic on
/// the optical axis, and does not cross itself. A profile that fails any of these
/// is refused with the reason, not revolved into a body that does not close.
pub fn can_promote_to_revolved(body: &Body3) -> PromotionCheck {
    let p = match &body.geometry {
        SolidGeometry3::Profile(p) => p,
        _ => return PromotionCheck::No("body is an analytic solid".to_string()),
    };
    if p.profile.edges.is_empty() {
        return PromotionCheck::No("profile has no edges".to_string());
    }
    if !p.profile.is_finite() {
        return PromotionCheck::No("profile has a non-finite value".to_string());
    }
    match crate::geom3::profile::validate_revolvable(&p.profile) {
        Ok(()) => PromotionCheck::Ok,
        Err(why) => PromotionCheck::No(why),
    }
}

/// Can `body` be extruded between `z_min` and `z_max`? (§6.5)
///
/// The `XY` loop must close and not cross itself
/// (`crate::geom3::profile::validate_extrudable`); the limits must be finite and
/// ordered. No symmetry is required — an extrusion has no axis to be symmetric
/// about.
pub fn can_promote_to_extruded(body: &Body3, z_min: f64, z_max: f64) -> PromotionCheck {
    let p = match &body.geometry {
        SolidGeometry3::Profile(p) => p,
        _ => return PromotionCheck::No("body is an analytic solid".to_string()),
    };
    if p.profile.edges.is_empty() {
        return PromotionCheck::No("profile has no edges".to_string());
    }
    if !p.profile.is_finite() {
        return PromotionCheck::No("profile has a non-finite value".to_string());
    }
    if !z_min.is_finite() || !z_max.is_finite() {
        return PromotionCheck::No("extrusion limits are not finite".to_string());
    }
    if z_min >= z_max {
        return PromotionCheck::No("z_min >= z_max bounds no volume".to_string());
    }
    match crate::geom3::profile::validate_extrudable(&p.profile) {
        Ok(()) => PromotionCheck::Ok,
        Err(why) => PromotionCheck::No(why),
    }
}

/// Give a body an explicit 3D construction.
///
/// The only way a `PlanarOnly` body becomes 3D. There is no automatic path, and
/// that is the point (§3.2).
pub fn promote_body(
    scene: &mut Scene3,
    body: BodyId,
    construction: ProfileConstruction3,
) -> Result<PromotionReport, PromotionError> {
    let b = scene
        .body(body)
        .ok_or(PromotionError::UnknownBody(body.0))?;
    if !matches!(b.geometry, SolidGeometry3::Profile(_)) {
        return Err(PromotionError::NotProfileAuthored(body.0));
    }
    let check = match construction {
        ProfileConstruction3::RevolveLocalX => can_promote_to_revolved(b),
        ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
            can_promote_to_extruded(b, z_min_mm, z_max_mm)
        }
        // Demoting to PlanarOnly is always allowed: forgetting a claim needs no
        // evidence.
        ProfileConstruction3::PlanarOnly { .. } => PromotionCheck::Ok,
    };
    let undecided = match check {
        PromotionCheck::No(why) => return Err(PromotionError::Refused(why)),
        PromotionCheck::Undecided(why) => vec![why],
        PromotionCheck::Ok => Vec::new(),
    };

    let b = scene.body_mut(body).expect("body found above");
    let from = b.construction().expect("profile checked above");
    if let SolidGeometry3::Profile(p) = &mut b.geometry {
        p.construction = construction;
    }
    Ok(PromotionReport {
        body,
        from,
        to: construction,
        undecided,
    })
}
