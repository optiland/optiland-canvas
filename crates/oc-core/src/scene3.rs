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
/// The current document version: v2 plus the film catalogue, material
/// provenance, the opaque backing and the fidelity profile
/// (`crate::migrate_v2`). Every v3 addition is optional on read.
pub const OCANVAS_V3_VERSION: u32 = 3;
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
                "unsupported .ocanvas version {found}; this build reads 1, {OCANVAS_V2_VERSION} and {OCANVAS_V3_VERSION}"
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

    /// The largest sample, for a caller outside this crate: whether a film's
    /// extinction coefficient is ever non-zero, say.
    pub fn values_max(&self) -> f64 {
        self.max()
    }
}

/// A wavelength table is a function: strictly increasing, at least two samples,
/// matched lengths, all finite.
/// A quantity that varies with the angle a ray arrives at, in degrees from the
/// element's normal.
///
/// [`SpectralScalar`] in the other variable. A coating needs nothing like this:
/// its angle dependence falls out of its own stack, computed at each ray's own
/// angle. This is here for the one thing no stack can supply, an ideal
/// polarizing layer's contrast, which either comes off a datasheet or is not
/// known at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AngularScalar {
    Constant(f64),
    Tabulated {
        aoi_deg: Vec<f64>,
        values: Vec<f64>,
        interpolation: Interpolation,
        out_of_range: OutOfRangePolicy,
    },
}

impl AngularScalar {
    /// The value at an angle of incidence, or `None` where the table has no
    /// answer and its policy says to refuse rather than invent one.
    pub fn sample(&self, aoi_deg: f64) -> Option<f64> {
        match self {
            Self::Constant(v) => Some(*v),
            Self::Tabulated {
                aoi_deg: xs,
                values,
                out_of_range,
                ..
            } => interp_table(xs, values, aoi_deg, *out_of_range),
        }
    }

    /// The value on axis, which is the one number anything without an angle
    /// has to use: the f32 preview, and a datasheet headline.
    pub fn on_axis(&self) -> f64 {
        self.sample(0.0).unwrap_or_else(|| match self {
            Self::Constant(v) => *v,
            Self::Tabulated { values, .. } => values.first().copied().unwrap_or(0.0),
        })
    }

    fn values(&self) -> &[f64] {
        match self {
            Self::Constant(v) => std::slice::from_ref(v),
            Self::Tabulated { values, .. } => values,
        }
    }

    /// Is every sample within `[0, 1]`? Used where the quantity is a fraction.
    pub fn is_unit_fraction(&self) -> bool {
        self.values().iter().all(|v| (0.0..=1.0).contains(v))
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
                aoi_deg, values, ..
            } => validate_angle_table(field, aoi_deg, values),
        }
    }
}

/// The angle version of [`validate_table`]. Same rules, except that zero is the
/// most important sample rather than an illegal one, and nothing past grazing
/// means anything.
fn validate_angle_table(field: &str, aoi_deg: &[f64], values: &[f64]) -> Result<(), SceneError> {
    let bad = |reason: &str| SceneError::InvalidTable {
        field: field.to_string(),
        reason: reason.to_string(),
    };
    if aoi_deg.len() != values.len() {
        return Err(bad("angle and value arrays have different lengths"));
    }
    if aoi_deg.len() < 2 {
        return Err(bad("a table needs at least two samples to interpolate"));
    }
    if aoi_deg.iter().chain(values).any(|v| !v.is_finite()) {
        return Err(bad("a sample is not finite"));
    }
    if aoi_deg.iter().any(|a| !(0.0..90.0).contains(a)) {
        return Err(bad("angle of incidence must be in [0, 90)"));
    }
    if aoi_deg.windows(2).any(|w| w[1] <= w[0]) {
        return Err(bad("angles must be strictly increasing"));
    }
    Ok(())
}

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
    /// Does the curve have data at `lambda_um`, rather than an out-of-range
    /// policy's answer? A constant covers every wavelength.
    pub fn covers(&self, lambda_um: f64) -> bool {
        match self {
            Self::Constant(_) => true,
            Self::Tabulated { lambda_um: xs, .. } => {
                xs.first().is_some_and(|&a| lambda_um >= a) && xs.last().is_some_and(|&b| lambda_um <= b)
            }
        }
    }

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

/// A real waveplate: a slab of uniaxial crystal cut with its optic axis lying
/// in the plate's own plane, which is every ordinary waveplate.
///
/// Giving a retarder one of these replaces "the retardance is this number" with
/// "the retardance is what this plate gives", so it moves with the wavelength
/// and with the direction the ray arrives from. The nominal `retardance_deg`
/// becomes what the plate gives on axis at `design_lambda_um`, which is how a
/// waveplate is specified and sold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Waveplate {
    /// Ordinary index, taken as constant with wavelength.
    pub n_o: f64,
    /// Extraordinary index, taken as constant with wavelength. Larger than
    /// `n_o` in a positive crystal like quartz, where the optic axis is the
    /// slow one; smaller in a negative one like calcite, where it is the fast
    /// one. The element's `axis_deg` names the fast axis either way.
    pub n_e: f64,
    /// Whole waves of retardance on top of the nominal. 0 is a true zero-order
    /// plate; a quarter wave sold cheaply is often 5 or 10, and it is that many
    /// times more sensitive to both wavelength and angle.
    pub order: u32,
    /// The wavelength the nominal retardance is quoted at, microns.
    pub design_lambda_um: f64,
}

impl Waveplate {
    /// Crystal quartz at the sodium D line, which is what most waveplates are.
    pub fn quartz(order: u32, design_lambda_um: f64) -> Self {
        Self {
            n_o: 1.54424,
            n_e: 1.55335,
            order,
            design_lambda_um,
        }
    }

    /// The plate thickness the nominal retardance implies, in microns.
    pub fn thickness_um(&self, retardance_deg: f64) -> f64 {
        crate::polarization::a_plate_thickness(
            self.n_o,
            self.n_e,
            self.design_lambda_um,
            retardance_deg.to_radians(),
            self.order,
        )
    }
}

/// A polarizing element with nothing around it: the polarization action alone,
/// and no faces.
///
/// An index of 1 is not a convention for "off". It is air, where a face
/// reflects nothing and transmits everything, so the two faces drop out of the
/// arithmetic rather than being skipped by a branch. An ordinary sheet
/// polarizer is laminated in something near 1.5, and that glass is most of what
/// makes a real one care about the angle it is used at.
///
/// Below 1 is refused rather than clamped: the front face could then reflect a
/// ray totally before it ever reached the polarizing layer, which is a
/// different element than the one anybody meant to place.
fn bare_element() -> f64 {
    1.0
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
    /// An ideal mirror: it reflects as a flat mirror on its plane does, then
    /// bends the ray toward a focus on the reflected axis, at every angle of
    /// incidence, so it has no astigmatism and no coma: what it is asked to do
    /// and nothing more. A power is 1/f per mm, 0 for flat in that direction;
    /// `power_x` acts across the reflected ray along local `X`, `power_y` along
    /// local `Y`, and two equal powers make one point focus. Full reflection,
    /// no dispersion, no added path length. Only meaningful on a standalone
    /// planar surface; a body patch authoring it is rejected by the tracer.
    IdealMirror {
        power_x_per_mm: f64,
        power_y_per_mm: f64,
    },
    /// A plane diffraction grating: the grooves run along the surface's local
    /// `Y`. Every ray leaves in `order` by the grating equation, exactly, and
    /// in the zero order; the fractions going into each are declared, not
    /// computed, and the rest is absorbed (it would go into other orders).
    /// See [`crate::grating`]. Only meaningful on a standalone planar surface,
    /// whose frame sets the grooves; a body patch authoring it is rejected.
    Grating {
        lines_per_mm: f64,
        /// The diffraction order traced besides the zero order, never 0. Its
        /// sign says which way across the grooves light is sent: `+1` turns a
        /// ray toward local `+X`.
        order: i32,
        /// The power fraction sent into `order`.
        efficiency: SpectralScalar,
        /// The power fraction left in the zero order: the specular
        /// reflection off a reflection grating, the straight-through beam of
        /// a transmission one.
        zero_order: SpectralScalar,
        /// A transmission grating: light carries on through the surface.
        /// False for a reflection grating.
        #[serde(default)]
        transmissive: bool,
        /// The two fractions for light polarized across the grooves (TM, the
        /// electric field perpendicular to them), when the grating treats the
        /// polarizations differently, as a ruled metal grating does by tens of
        /// percent. `efficiency` and `zero_order` are then the fractions for
        /// light polarized along the grooves (TE). `None` treats both alike.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tm: Option<GratingTm>,
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
    /// Measured or supplied R and T over wavelength and angle, with what the
    /// numbers mean: resolution, side, scope, media, provenance and what to do
    /// outside the data (coating-fidelity spec CF-09). See
    /// [`crate::response_table`].
    ///
    /// `from` is the side of the part this copy answers for: a surface's front
    /// response is met from its front, a body face's outside response from
    /// outside. A table measured from one side refuses (or, permissively,
    /// flags) a ray from the other, because an absorbing stack does not
    /// reflect the same both ways.
    MeasuredTable {
        table: Box<crate::response_table::AngularTable>,
        from: crate::response_table::TableSide,
    },
    /// An ideal absorbing linear polarizer: it passes the field component along
    /// its transmission axis and absorbs the perpendicular one.
    ///
    /// Zero thickness and no dispersion. The polarizing layer itself is ideal:
    /// its contrast is one number at every wavelength and every angle, where a
    /// real sheet's degrades as it is tilted and a thin-film or prism polarizer
    /// has an acceptance cone of a few degrees. Unpolarized light loses half
    /// its flux here, which is a physical fact and not an assumption.
    ///
    /// Two things do depend on the angle. The axis is a direction in the
    /// element, so a skew ray sees its projection, which is enough for a
    /// crossed pair to leak off axis. And `substrate_index` puts the layer in
    /// glass, whose two faces reflect, more of s than of p and more of both as
    /// the element is tilted.
    ///
    /// `extinction` is the **power** fraction of the blocked component that
    /// still gets through, so 0 is perfect and 1e-4 is an ordinary sheet
    /// polarizer. One number is an ideal layer that holds its contrast at every
    /// angle; a table against angle of incidence is a real one, and is the only
    /// way to say where a wire grid, a thin-film polarizer or a Glan prism
    /// stops working, because no stack the kernel can compute will tell it.
    ///
    /// Only meaningful on a standalone planar surface, whose local `+X` the
    /// axis is measured from; a body patch authoring one is rejected.
    Polarizer {
        /// Transmission axis, degrees from the surface's local `+X`.
        axis_deg: f64,
        /// The power fraction of the blocked component that still gets
        /// through, either one number or a datasheet's curve against angle.
        extinction: AngularScalar,
        /// Index of the glass the polarizing layer is carried in. See
        /// [`bare_element`].
        #[serde(default = "bare_element")]
        substrate_index: f64,
    },
    /// An ideal linear retarder: a waveplate adding `retardance_deg` of phase
    /// to the slow axis relative to the fast one. Lossless unless it is given a
    /// substrate, whose faces then reflect.
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
    /// notices and a multi-order one does not survive. `substrate_index` adds
    /// the faces a real plate has; it does not make the retardance depend on
    /// anything.
    ///
    /// Only meaningful on a standalone planar surface, whose local `+X` the
    /// fast axis is measured from.
    Retarder {
        /// Fast axis, degrees from the surface's local `+X`.
        axis_deg: f64,
        retardance_deg: f64,
        /// Index of the glass the waveplate is carried in. See
        /// [`bare_element`].
        #[serde(default = "bare_element")]
        substrate_index: f64,
        /// What the plate is made of, if it is made of anything. `None` is an
        /// ideal retarder: `retardance_deg` at every wavelength and every
        /// angle.
        #[serde(default)]
        crystal: Option<Waveplate>,
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
            Self::IdealMirror { power_x_per_mm, power_y_per_mm } => {
                if !power_x_per_mm.is_finite() || !power_y_per_mm.is_finite() {
                    return Err(range("power", "must be finite"));
                }
                Ok(())
            }
            Self::Grating {
                lines_per_mm,
                order,
                efficiency,
                zero_order,
                tm,
                ..
            } => {
                if let Some(tm) = tm {
                    tm.efficiency.validate(&format!("{field}.tm.efficiency"))?;
                    tm.zero_order.validate(&format!("{field}.tm.zero_order"))?;
                    if !tm.efficiency.is_unit_fraction() || !tm.zero_order.is_unit_fraction() {
                        return Err(range("tm", "has an efficiency outside [0, 1]"));
                    }
                }
                if !(lines_per_mm.is_finite() && *lines_per_mm > 0.0 && *lines_per_mm <= 20_000.0) {
                    return Err(range("lines_per_mm", "must be in (0, 20000]"));
                }
                if *order == 0 || order.abs() > 50 {
                    return Err(range("order", "must be nonzero and at most 50 either way"));
                }
                efficiency.validate(&format!("{field}.efficiency"))?;
                zero_order.validate(&format!("{field}.zero_order"))?;
                if !efficiency.is_unit_fraction() {
                    return Err(range("efficiency", "is outside [0, 1]"));
                }
                if !zero_order.is_unit_fraction() {
                    return Err(range("zero_order", "is outside [0, 1]"));
                }
                if let (SpectralScalar::Constant(a), SpectralScalar::Constant(b)) = (efficiency, zero_order) {
                    if a + b > 1.0 + 1e-12 {
                        return Err(range("efficiency", "and zero_order add up to more than 1"));
                    }
                }
                Ok(())
            }
            Self::MeasuredTable { table, .. } => table.validate().map_err(|reason| SceneError::InvalidTable {
                field: format!("{field}.table"),
                reason,
            }),
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
                if let crate::thinfilm::CoatingBacking::Plate { substrate, exit, thickness_mm } = &coating.backing
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
                    if !thickness_mm.is_finite() || !(0.0..=1_000.0).contains(thickness_mm) {
                        return Err(range("coating.thickness_mm", "is outside [0, 1000]"));
                    }
                    if let crate::thinfilm::ExitFace::AntiReflective { residual } = exit {
                        if !residual.is_finite() || !(0.0..=1.0).contains(residual) {
                            return Err(range("coating.exit.residual", "is outside [0, 1]"));
                        }
                    }
                }
                if let crate::thinfilm::CoatingBacking::Opaque { substrate } = &coating.backing {
                    substrate.n.validate(&format!("{field}.coating.substrate.n"))?;
                    substrate.k.validate(&format!("{field}.coating.substrate.k"))?;
                    if substrate.n.min() <= 0.0 {
                        return Err(range("coating.substrate.n", "must be positive"));
                    }
                    if substrate.k.min() < 0.0 {
                        return Err(range("coating.substrate.k", "must not be negative"));
                    }
                }
                Ok(())
            }
            Self::Polarizer {
                axis_deg,
                extinction,
                substrate_index,
            } => {
                if !axis_deg.is_finite() {
                    return Err(range("axis_deg", "must be finite"));
                }
                extinction.validate(&format!("{field}.extinction"))?;
                if !extinction.is_unit_fraction() {
                    return Err(range("extinction", "is outside [0, 1]"));
                }
                if !substrate_index.is_finite() || *substrate_index < 1.0 {
                    return Err(range("substrate_index", "must be at least 1"));
                }
                Ok(())
            }
            Self::Retarder {
                axis_deg,
                retardance_deg,
                substrate_index,
                crystal,
            } => {
                if !axis_deg.is_finite() {
                    return Err(range("axis_deg", "must be finite"));
                }
                if !retardance_deg.is_finite() {
                    return Err(range("retardance_deg", "must be finite"));
                }
                if !substrate_index.is_finite() || *substrate_index < 1.0 {
                    return Err(range("substrate_index", "must be at least 1"));
                }
                if let Some(c) = crystal {
                    if !c.n_o.is_finite() || c.n_o < 1.0 {
                        return Err(range("crystal.n_o", "must be at least 1"));
                    }
                    if !c.n_e.is_finite() || c.n_e < 1.0 {
                        return Err(range("crystal.n_e", "must be at least 1"));
                    }
                    // An isotropic plate would silently retard by nothing
                    // however thick it was asked to be, which is worse than
                    // refusing it.
                    if (c.n_e - c.n_o).abs() < 1e-9 {
                        return Err(range("crystal", "has no birefringence to retard with"));
                    }
                    if !c.design_lambda_um.is_finite() || c.design_lambda_um <= 0.0 {
                        return Err(range("crystal.design_lambda_um", "must be positive"));
                    }
                    if c.order > 1000 {
                        return Err(range("crystal.order", "is beyond any real plate"));
                    }
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
    /// A hollow circular tube about local `+X`, centred on the origin: a light
    /// pipe, a lens barrel, a baffle. The three numbers define it completely.
    ///
    /// The one analytic solid whose faces can carry their own responses, by
    /// the patch ids in [`tube_face`](crate::geom3::primitives::tube_face). A
    /// reflective bore is a tube whose default response absorbs and whose bore
    /// overrides it with a mirror or a metal coating.
    Tube {
        inner_radius_mm: f64,
        outer_radius_mm: f64,
        half_length_mm: f64,
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
        /// The piece of the revolution that exists, in the local `XY` plane
        /// (seen down the axis): a strip cut from a large mirror, or an
        /// off-axis segment as a polygon. `None` is the whole annulus.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outline: Option<PlanarShape2>,
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

/// The most cells a detector's grid may hold, on its face or across its
/// directions: a 1024 × 1024 sensor.
///
/// A readout keeps two numbers per face cell and the tracer one readout per
/// worker, so a count nobody bounded is an allocation nobody asked for. And a
/// million pixels needs tens of millions of rays before any one pixel's
/// reading rests on more than a few of them.
pub const MAX_DETECTOR_CELLS: usize = 1 << 20;

/// Does a `[a, b]` grid stay within [`MAX_DETECTOR_CELLS`]? Checked so that a
/// count read from a file cannot overflow on the way.
pub fn grid_fits(bins: [usize; 2]) -> bool {
    bins[0]
        .checked_mul(bins[1])
        .is_some_and(|n| n <= MAX_DETECTOR_CELLS)
}

/// An area detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detector3 {
    pub common: ObjectCommon,
    pub geometry: DetectorGeometry3,
    /// Pixel counts across the two local axes, `[along u, along v]`. The grid
    /// spans the face's bounding box, so the pitch is that extent over these.
    pub bins: [usize; 2],
    pub response: DetectorResponse,
    pub termination: DetectorTermination,
    /// What the readings mean. A document written before there was a choice
    /// has no field here and reads power, which is what it always read.
    #[serde(default)]
    pub measure: DetectorMeasure,
}

impl Detector3 {
    pub fn id(&self) -> DetectorId {
        DetectorId(self.common.id.0)
    }

    /// The full extent the face grid spans, `[width, height]` in mm, or `None`
    /// for a line probe with no area.
    ///
    /// The grid is the shape's **bounding box**, which for a rectangle is the
    /// shape and for a disc or annulus is the square around it.
    pub fn grid_extent_mm(&self) -> Option<[f64; 2]> {
        match self.geometry {
            DetectorGeometry3::Rectangle {
                half_width_mm,
                half_height_mm,
            } => Some([2.0 * half_width_mm, 2.0 * half_height_mm]),
            DetectorGeometry3::Disc { radius_mm } => Some([2.0 * radius_mm; 2]),
            DetectorGeometry3::Annulus { outer_mm, .. } => Some([2.0 * outer_mm; 2]),
            DetectorGeometry3::LineProbeLegacy { .. } => None,
        }
    }

    /// The pixel pitch along the local `u` and `v` axes, mm: the grid's extent
    /// over its cell count. Pixels are square only when the two agree.
    pub fn pixel_pitch_mm(&self) -> Option<[f64; 2]> {
        let [w, h] = self.grid_extent_mm()?;
        Some([w / self.bins[0].max(1) as f64, h / self.bins[1].max(1) as f64])
    }

    /// The area of one whole pixel, `mm²`, or `None` for a line probe.
    ///
    /// A pixel is a square of the bounding box, so on a disc or annulus a
    /// pixel at the rim has part of itself off the face. What it collected
    /// came through the part on the face, which is
    /// [`Self::covered_cell_areas_mm2`]; this is the whole square, the
    /// camera's pixel. Dividing the disc's own area by the pixel count instead
    /// would put every reading 4/π high, which is what this did until the
    /// measure needed a unit.
    pub fn pixel_area_mm2(&self) -> Option<f64> {
        let [pu, pv] = self.pixel_pitch_mm()?;
        Some(pu * pv)
    }

    /// The sensitive area of the whole face, `mm²`, or `None` for a line
    /// probe.
    pub fn face_area_mm2(&self) -> Option<f64> {
        use std::f64::consts::PI;
        match self.geometry {
            DetectorGeometry3::Rectangle {
                half_width_mm,
                half_height_mm,
            } => Some(4.0 * half_width_mm * half_height_mm),
            DetectorGeometry3::Disc { radius_mm } => Some(PI * radius_mm * radius_mm),
            DetectorGeometry3::Annulus { inner_mm, outer_mm } => {
                Some(PI * (outer_mm * outer_mm - inner_mm * inner_mm))
            }
            DetectorGeometry3::LineProbeLegacy { .. } => None,
        }
    }

    /// How much of each pixel lies on the face, `mm²`, row-major over `bins`
    /// like the face histogram, or `None` for a line probe.
    ///
    /// Every pixel of a rectangle is whole. On a disc or annulus a pixel at
    /// the rim is only partly on the face, and one in a corner of the box or
    /// in an annulus's hole not at all. Flux reaches a pixel only through the
    /// part on the face, so that part is what irradiance divides by: a rim
    /// pixel under uniform light then reads the same as a whole one, and the
    /// readings times these areas sum back to the flux collected (A12). The
    /// areas are exact, from [`disc_rect_overlap_area`], and sum to
    /// [`Self::face_area_mm2`].
    ///
    /// [`disc_rect_overlap_area`]: crate::geom3::primitives::disc_rect_overlap_area
    pub fn covered_cell_areas_mm2(&self) -> Option<Vec<f64>> {
        self.geometry.covered_cell_areas_mm2(self.bins)
    }

    /// The direction cell an arriving ray falls in, or `None` when the
    /// measure bins no directions or the ray came in past the cone.
    ///
    /// Polar cells are row-major over the measure's own `[rings, sectors]` as
    /// `sector * rings + ring`. Grid cells are row-major over `[across, down]`
    /// with `across` fastest, laid out like the face's pixels, so a picture
    /// of either is drawn the same way.
    ///
    /// `dir_local` is the ray's direction in the detector's own frame. The
    /// polar angle is taken to the normal **line**, not the normal vector, so
    /// a detector reads the same whichever side light reaches it from, which
    /// is the convention `max_tilt_rad` already uses. The grid's two angles
    /// are taken the same way: `atan(x / |z|)` and `atan(y / |z|)`.
    pub fn direction_cell(&self, dir_local: Vec3) -> Option<usize> {
        if self.measure.direction_layout() == Some(DirectionLayout::Grid) {
            let (cone, nu, nv) = self.measure.direction_grid()?;
            let z = dir_local.z.abs();
            let (au, av) = (dir_local.x.atan2(z), dir_local.y.atan2(z));
            if au.abs() > cone || av.abs() > cone {
                return None;
            }
            let cell = |a: f64, n: usize| (((a + cone) / (2.0 * cone) * n as f64) as usize).min(n - 1);
            return Some(cell(av, nv) * nu + cell(au, nu));
        }
        let (cone, rings, sectors) = self.measure.direction_grid()?;
        let theta = dir_local.z.abs().clamp(0.0, 1.0).acos();
        if theta > cone {
            return None;
        }
        let ring = ((theta / cone * rings as f64) as usize).min(rings - 1);
        let phi = dir_local
            .y
            .atan2(dir_local.x)
            .rem_euclid(std::f64::consts::TAU);
        let sector = ((phi / std::f64::consts::TAU * sectors as f64) as usize).min(sectors - 1);
        Some(sector * rings + ring)
    }

    /// Every direction cell's solid angle, steradians, laid out like
    /// [`Self::direction_cell`], or `None` when the measure bins none.
    ///
    /// Closed form and exact: a ring between polar angles `θ₀` and `θ₁`, cut
    /// into azimuth sectors of width `Δφ`, gives each of its cells
    /// `Δφ(cos θ₀ − cos θ₁)`. Nothing here is quadrature and nothing here
    /// assumes a small cell, so a two-ring histogram is as exact as a
    /// hundred-ring one.
    pub fn direction_cell_solid_angles(&self) -> Option<Vec<f64>> {
        if self.measure.direction_layout() == Some(DirectionLayout::Grid) {
            return self
                .measure
                .direction_grid()
                .map(|(cone, nu, nv)| grid_solid_angles(cone, nu, nv));
        }
        let (cone, rings, sectors) = self.measure.direction_grid()?;
        let dphi = std::f64::consts::TAU / sectors as f64;
        let ring_cos: Vec<f64> = (0..=rings)
            .map(|i| (cone * i as f64 / rings as f64).cos())
            .collect();
        let mut out = vec![0.0; rings * sectors];
        for sector in 0..sectors {
            for ring in 0..rings {
                out[sector * rings + ring] = dphi * (ring_cos[ring] - ring_cos[ring + 1]);
            }
        }
        Some(out)
    }
}

/// Each cell's solid angle, steradians, for a grid `nu` by `nv` cells over
/// `±cone` in both angles, row-major with `u` fastest.
///
/// Closed form and exact. A direction `(p, q, 1)` in slope space covers
/// `dp dq / (1 + p² + q²)^{3/2}` of the sphere, and over the rectangle from
/// the origin to `(p, q)` that integrates to `F(p, q) = atan(pq / √(1 + p² +
/// q²))`, the solid angle a rectangle subtends from above its corner. A cell
/// between slopes `p₀..p₁` and `q₀..q₁` is then four values of `F`. At a
/// 90° cone the edge slope is `tan(π/2)`, which in f64 is a large finite
/// number, and the whole grid sums to `2π` as it should.
pub fn grid_solid_angles(cone: f64, nu: usize, nv: usize) -> Vec<f64> {
    let f = |p: f64, q: f64| (p * q / (1.0 + p * p + q * q).sqrt()).atan();
    let edges = |n: usize| -> Vec<f64> {
        (0..=n)
            .map(|i| (-cone + 2.0 * cone * i as f64 / n as f64).tan())
            .collect()
    };
    let (pu, pv) = (edges(nu), edges(nv));
    let mut out = Vec::with_capacity(nu * nv);
    for j in 0..nv {
        for i in 0..nu {
            out.push(
                f(pu[i + 1], pv[j + 1]) - f(pu[i], pv[j + 1]) - f(pu[i + 1], pv[j])
                    + f(pu[i], pv[j]),
            );
        }
    }
    out
}

impl DetectorGeometry3 {
    /// [`Detector3::covered_cell_areas_mm2`] for this face cut into `bins`,
    /// for a caller that holds the face and the counts but not the detector.
    pub fn covered_cell_areas_mm2(&self, bins: [usize; 2]) -> Option<Vec<f64>> {
        use crate::geom3::primitives::disc_rect_overlap_area as disc;
        let [w, h] = match *self {
            DetectorGeometry3::Rectangle {
                half_width_mm,
                half_height_mm,
            } => [2.0 * half_width_mm, 2.0 * half_height_mm],
            DetectorGeometry3::Disc { radius_mm } => [2.0 * radius_mm; 2],
            DetectorGeometry3::Annulus { outer_mm, .. } => [2.0 * outer_mm; 2],
            DetectorGeometry3::LineProbeLegacy { .. } => return None,
        };
        let (nu, nv) = (bins[0].max(1), bins[1].max(1));
        // The edges `detector_cell` bins by: `floor((u + w/2) / w * nu)`.
        let u_edge = |i: usize| -0.5 * w + w * i as f64 / nu as f64;
        let v_edge = |j: usize| -0.5 * h + h * j as f64 / nv as f64;
        let mut out = Vec::with_capacity(nu * nv);
        for j in 0..nv {
            let (y0, y1) = (v_edge(j), v_edge(j + 1));
            for i in 0..nu {
                let (x0, x1) = (u_edge(i), u_edge(i + 1));
                out.push(match *self {
                    DetectorGeometry3::Rectangle { .. } => (x1 - x0) * (y1 - y0),
                    DetectorGeometry3::Disc { radius_mm } => disc(radius_mm, x0, x1, y0, y1),
                    DetectorGeometry3::Annulus { inner_mm, outer_mm } => {
                        (disc(outer_mm, x0, x1, y0, y1) - disc(inner_mm, x0, x1, y0, y1))
                            .max(0.0)
                    }
                    DetectorGeometry3::LineProbeLegacy { .. } => return None,
                });
            }
        }
        Some(out)
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

/// What a detector's readings are, and so what unit they carry.
///
/// The bins hold flux whichever this is; the measure is the divisor, and the
/// divisor is what makes the number mean something. Flux stays **relative**
/// unless a source is calibrated and nothing here calibrates one, so
/// `Irradiance` reads per `mm²` of the source's own flux rather than W/mm²
/// (the units rule in `docs/architecture/canonical-3d-kernel.md`).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum DetectorMeasure {
    /// Flux landing in each cell of the face. What a detector read before
    /// there was anything to choose.
    #[default]
    Power,

    /// Flux over the area it was collected on, per `mm²`: a pixel's flux over
    /// the part of that pixel on the face, which is the whole pixel except at
    /// a round face's rim ([`Detector3::covered_cell_areas_mm2`]).
    ///
    /// The area is the face's, not the beam's cross-section. A beam arriving
    /// at a slant spreads across more cells and each one reads less, so the
    /// obliquity arrives on its own rather than being applied by hand.
    ///
    /// A line probe has no area and cannot ask for this (§3.2).
    Irradiance,

    /// Flux per steradian, binned by **the direction light arrived from**
    /// instead of by where it landed.
    ///
    /// Intensity is an angular density, so it needs an angular histogram, and
    /// this carries one of its own.
    ///
    /// The face histogram keeps filling as it always did, so a detector in
    /// this mode holds both pictures and the hit map still draws.
    Intensity {
        /// The cone around the face normal the histogram spans, degrees in
        /// `(0, 90]`. Flux arriving outside it still counts in the totals and
        /// is binned nowhere, which the result reports rather than hides.
        half_angle_deg: f64,
        /// Polar rings out to that cone and azimuth sectors around the normal,
        /// or, on a [`DirectionLayout::Grid`], cells across and down.
        /// `[24, 1]` polar is the `I(θ)` curve an LED datasheet plots.
        ///
        /// Its own resolution rather than the face's, because the two have
        /// nothing to say to each other: a detector can want a fine spot on
        /// its face and a coarse polar curve through it at the same time, and
        /// sharing one number would make every mode switch a resolution
        /// change as well.
        bins: [usize; 2],
        /// How the directions are cut up. Polar when a file says nothing,
        /// which is what every file written before there was a choice meant.
        #[serde(default, skip_serializing_if = "DirectionLayout::is_polar")]
        layout: DirectionLayout,
    },

    /// What the eye makes of the light in each cell: illuminance, the
    /// `683 lm/W · Σ Φ ȳ(λ)` a light meter reads, per `m²` of the cell on the
    /// face, with the cell's colour `(X, Y, Z)` under the CIE 1931 observer
    /// kept beside it so the face can be drawn as a person would see it
    /// ([`crate::colorimetry`]). Lux when the flux is watts; relative
    /// otherwise, like every other reading.
    ///
    /// A line probe has no area and cannot ask for this, as for irradiance.
    Illuminance,
}

/// How an intensity histogram cuts up the directions around the normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DirectionLayout {
    /// Rings of polar angle and sectors of azimuth: the goniometer's picture,
    /// and the `I(θ)` curve with one sector.
    #[default]
    Polar,
    /// A square grid in the two angles `atan(x/|z|)` and `atan(y/|z|)`, each
    /// over `±cone`: a far-field picture with straight axes, the one a camera
    /// in a lens's focal plane records, and one a cross-section can be cut
    /// through along either axis.
    Grid,
}

impl DirectionLayout {
    pub fn is_polar(&self) -> bool {
        *self == Self::Polar
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Polar => "polar",
            Self::Grid => "grid",
        }
    }
}

impl DetectorMeasure {
    /// How the intensity histogram is laid out, or `None` if this measure
    /// bins no directions.
    pub fn direction_layout(&self) -> Option<DirectionLayout> {
        match *self {
            Self::Intensity { layout, .. } => Some(layout),
            Self::Power | Self::Irradiance | Self::Illuminance => None,
        }
    }

    /// A cone in radians and the `[rings, sectors]` across it, or `None` if
    /// this measure bins no directions at all.
    pub fn direction_grid(&self) -> Option<(f64, usize, usize)> {
        match *self {
            Self::Intensity {
                half_angle_deg,
                bins,
                ..
            } => Some((half_angle_deg.to_radians(), bins[0].max(1), bins[1].max(1))),
            Self::Power | Self::Irradiance | Self::Illuminance => None,
        }
    }

    /// The cone this measure bins directions over, radians, or `None`.
    pub fn cone_rad(&self) -> Option<f64> {
        self.direction_grid().map(|(cone, _, _)| cone)
    }

    /// The unit one reading carries, with flux left as the relative thing it
    /// is.
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Power => "flux",
            Self::Irradiance => "flux/mm²",
            Self::Intensity { .. } => "flux/sr",
            Self::Illuminance => "lx",
        }
    }
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

    /// The sun as it looks: a disc of angular radius `half_angle_rad`,
    /// brighter at the centre than the limb as `disc` says, and a share `csr`
    /// of the light in the aureole around it out to 43.6 mrad
    /// ([`crate::solar::sunshape`]). A limb-darkened disc depends on the
    /// wavelength, so each ray's direction is drawn for its own.
    Sun {
        half_angle_rad: f64,
        disc: crate::solar::sunshape::SunDisc,
        csr: f64,
    },

    /// **v1 compatibility.** An equal-angle fan in one plane: a degenerate 3D
    /// distribution whose measure exists only in that plane. It is not a cone,
    /// and it is not uniform in solid angle.
    PlanarFanLegacy {
        half_angle_rad: f64,
    },

    /// A measured beam, the same all round its axis: radiance (per steradian,
    /// any scale) at [`RADIAL_NODES`] polar angles evenly spaced from 0 to
    /// `half_angle_rad`, linear between them and nothing beyond. What a long
    /// light pipe hands on: the sun's disc widened by every slope error and
    /// wall bounce on the way, which no analytic shape describes.
    RadialProfile {
        half_angle_rad: f64,
        radiance: [f64; RADIAL_NODES],
    },
}

/// The node count of [`AngularDistribution3::RadialProfile`].
pub const RADIAL_NODES: usize = 24;

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
    /// The sun and the heliostats that follow it, if the scene has them.
    ///
    /// Authored intent, like [`SceneIntent`]: what it decides is written into
    /// the scene's own sources and mirrors by
    /// [`resolve_solar`](Scene3::resolve_solar), and the tracer never reads
    /// it. Absent from a document that has no sun, so every older document
    /// reads unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solar: Option<Box<crate::heliostat::SolarRig>>,
    /// Elements locked to other elements: rigidly attached, or turned to
    /// face them. Authored intent like [`Scene3::solar`], resolved into the
    /// transforms by [`resolve_links`](Scene3::resolve_links). Absent from a
    /// document that has none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<crate::link::Link3>,
    /// Faces that are not quite the shape they are drawn as: a heliostat's
    /// waviness, a rolled sheet's ripple. See [`FigureError`]. Absent from a
    /// document that has none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub figure_errors: Vec<FigureError>,
    /// How much coating physics a trace of this scene pays for (coating
    /// fidelity spec §5). Absent from a document that keeps the default,
    /// `Standard`, so every older document reads, and writes, unchanged.
    #[serde(default, skip_serializing_if = "crate::fidelity::FidelityProfile::is_default")]
    pub fidelity: crate::fidelity::FidelityProfile,
    /// The scene's film materials, which coating layers name by id (v3).
    /// Absent from a v2 document; reading one fills it from the inline
    /// materials (`crate::migrate_v2`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub film_catalog: Vec<crate::film_catalog::FilmCatalogEntry>,
    /// The temperature the scene's coatings are at, °C, read only under
    /// `temperature: Declared` (coating-fidelity spec CF-10). Absent: every
    /// film at its reference temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_c: Option<f64>,
    /// Objects at a temperature of their own: an eyepiece warmed by the
    /// light it absorbs, while the rest of the bench stays at room
    /// temperature.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub object_temperatures: Vec<ObjectTemperature>,
}

/// One object's coatings at a temperature of their own, °C.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectTemperature {
    pub object: ObjectId,
    pub temperature_c: f64,
}

/// A grating's efficiencies for light polarized across its grooves (TM).
/// See [`SurfaceResponse::Grating`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GratingTm {
    pub efficiency: SpectralScalar,
    pub zero_order: SpectralScalar,
}

/// The figure error of one body's or surface's faces, as the slope error a
/// concentrator datasheet quotes: at every hit the local normal leans away
/// from the drawn one by a Gaussian angle, `slope_rms_mrad` RMS in each of
/// the two directions across it, drawn afresh for each hit. A reflected ray
/// turns by twice the lean, a refracted one by `(n1 - n2) / n2` of it.
///
/// This is the statistical model solar-concentrator tracers use. It does not
/// know where on the face the error sits, so it cannot stand for a
/// systematic figure (a mirror bent to the wrong curve), and the f32 preview
/// leaves it out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FigureError {
    pub object: ObjectId,
    pub slope_rms_mrad: f64,
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
            version: OCANVAS_V3_VERSION,
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
            solar: None,
            links: Vec::new(),
            figure_errors: Vec::new(),
            fidelity: crate::fidelity::FidelityProfile::default(),
            film_catalog: Vec::new(),
            temperature_c: None,
            object_temperatures: Vec::new(),
        }
    }

    /// The temperature object `id`'s coatings are at: its own, else the
    /// scene's, else none (every film at its reference temperature).
    pub fn film_temperature_c(&self, id: u64) -> Option<f64> {
        self.object_temperatures
            .iter()
            .find(|t| t.object.0 == id)
            .map(|t| t.temperature_c)
            .or(self.temperature_c)
    }

    /// Set object `id`'s own temperature, or clear it with `None`. False for
    /// an id that is not a body or a surface.
    pub fn set_object_temperature(&mut self, id: u64, temperature_c: Option<f64>) -> bool {
        let optical = self.bodies.iter().any(|b| b.common.id.0 == id) || self.surfaces.iter().any(|s| s.common.id.0 == id);
        if !optical {
            return false;
        }
        self.object_temperatures.retain(|t| t.object.0 != id);
        if let Some(t) = temperature_c {
            self.object_temperatures.push(ObjectTemperature { object: ObjectId(id), temperature_c: t });
        }
        true
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

    /// Serialize to pretty v3 JSON, stamping the header.
    pub fn to_json(&self) -> String {
        let mut doc = self.clone();
        doc.format = default_format();
        doc.version = OCANVAS_V3_VERSION;
        doc.dimension = OCANVAS_V2_DIMENSION;
        serde_json::to_string_pretty(&doc).expect("scene3 serializes")
    }

    /// Serialize as a v2 document for readers that know no v3, with what v2
    /// could not hold. Exact for a scene read from v2 and edited without v3
    /// features; see [`crate::migrate_v2::to_v2`].
    pub fn to_json_v2(&self) -> (String, Vec<String>) {
        let (mut doc, losses) = crate::migrate_v2::to_v2(self);
        doc.format = default_format();
        doc.version = OCANVAS_V2_VERSION;
        doc.dimension = OCANVAS_V2_DIMENSION;
        (serde_json::to_string_pretty(&doc).expect("scene3 serializes"), losses)
    }

    /// Parse a v2 or v3 document, migrating a v2 one to v3 in memory. Use
    /// [`crate::migrate_v1::load_any`] to accept v1 too.
    pub fn from_json(s: &str) -> Result<Self, SceneError> {
        let scene: Scene3 = serde_json::from_str(s).map_err(|e| SceneError::Json(e.to_string()))?;
        scene.check_header()?;
        let mut scene = scene;
        if scene.version == OCANVAS_V2_VERSION {
            crate::migrate_v2::migrate_v2_to_v3(&mut scene);
        }
        crate::migrate_v2::resolve_film_refs(&mut scene)?;
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
        if self.version != OCANVAS_V2_VERSION && self.version != OCANVAS_V3_VERSION {
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
        if let Some(rig) = &self.solar {
            rig.validate(self)?;
        }
        self.validate_links()?;
        self.validate_figure_errors()?;
        self.validate_film_catalog()?;
        self.validate_body_coatings()?;
        self.validate_temperatures()?;
        Ok(())
    }

    /// Temperatures are finite and above absolute zero, each override names
    /// a body or surface, and an object has at most one.
    fn validate_temperatures(&self) -> Result<(), SceneError> {
        let check = |object: Option<u64>, field: &str, t: f64| {
            if !t.is_finite() {
                return Err(SceneError::NonFinite { object, field: field.into() });
            }
            if t <= -273.15 {
                return Err(SceneError::OutOfRange { object, field: field.into(), reason: "is at or below absolute zero".into() });
            }
            Ok(())
        };
        if let Some(t) = self.temperature_c {
            check(None, "temperature_c", t)?;
        }
        let mut seen = HashSet::new();
        for o in &self.object_temperatures {
            let id = o.object.0;
            let optical = self.bodies.iter().any(|b| b.common.id.0 == id) || self.surfaces.iter().any(|s| s.common.id.0 == id);
            if !optical {
                return Err(SceneError::DanglingRef { what: "object temperature", id });
            }
            if !seen.insert(id) {
                return Err(SceneError::OutOfRange {
                    object: Some(id),
                    field: "object_temperatures".into(),
                    reason: "an object has one temperature".into(),
                });
            }
            check(Some(id), "object_temperatures.temperature_c", o.temperature_c)?;
        }
        Ok(())
    }

    /// A body is its own substrate: a coating on one of its faces sits on
    /// the body's glass, so a plate-backed coating (which brings a substrate
    /// of its own) or one with slabs is refused there by name (CF-05), and so
    /// is a measured table of a whole part, which already holds the substrate
    /// and its rear face (CF-09).
    fn validate_body_coatings(&self) -> Result<(), SceneError> {
        let wrong = |r: &SurfaceResponse| match r {
            SurfaceResponse::Coating { coating } => {
                (matches!(coating.backing, crate::thinfilm::CoatingBacking::Plate { .. })
                    || crate::thinfilm::has_slab(coating))
                .then_some(("surface.coating", "a plate-backed or slab coating cannot coat a body face: the body is already its substrate"))
            }
            SurfaceResponse::MeasuredTable { table, .. } => (table.scope == crate::response_table::TableScope::FiniteAssembly).then_some((
                "surface.table",
                "a measured table of a whole part cannot sit on a body face: it already holds the substrate and its rear face, and the body would count them again",
            )),
            _ => None,
        };
        for b in &self.bodies {
            let s = &b.surface;
            let faces = std::iter::once(&s.default).chain(s.overrides.iter().map(|o| &o.response));
            for f in faces {
                if let Some((field, reason)) = wrong(&f.from_outside).or_else(|| wrong(&f.from_inside)) {
                    return Err(SceneError::OutOfRange {
                        object: Some(b.common.id.0),
                        field: field.to_string(),
                        reason: reason.to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Catalogue ids are unique and every layer names one that exists.
    fn validate_film_catalog(&self) -> Result<(), SceneError> {
        let mut ids: Vec<u32> = self.film_catalog.iter().map(|e| e.id).collect();
        ids.sort_unstable();
        if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
            return Err(SceneError::DuplicateId { id: u64::from(w[0]) });
        }
        for c in crate::migrate_v2::coatings(self) {
            for l in &c.layers {
                if let Some(id) = l.catalog_ref {
                    if ids.binary_search(&id).is_err() {
                        return Err(SceneError::DanglingRef { what: "film catalogue entry", id: u64::from(id) });
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_figure_errors(&self) -> Result<(), SceneError> {
        let mut seen = HashSet::new();
        for f in &self.figure_errors {
            let id = f.object.0;
            let optical = self.bodies.iter().any(|b| b.common.id.0 == id)
                || self.surfaces.iter().any(|x| x.common.id.0 == id);
            if !optical {
                return Err(SceneError::DanglingRef { what: "figure error", id });
            }
            if !f.slope_rms_mrad.is_finite() {
                return Err(SceneError::NonFinite {
                    object: Some(id),
                    field: "figure_errors.slope_rms_mrad".into(),
                });
            }
            let bad = |reason: &str| SceneError::OutOfRange {
                object: Some(id),
                field: "figure_errors".into(),
                reason: reason.into(),
            };
            // Past 100 mrad a face is no longer a wavy version of its drawn
            // shape, and the small-lean model below stops meaning anything.
            if !(0.0..=100.0).contains(&f.slope_rms_mrad) {
                return Err(bad("slope_rms_mrad must be in [0, 100]"));
            }
            if !seen.insert(id) {
                return Err(bad("an object has one figure error"));
            }
        }
        Ok(())
    }

    /// The RMS slope error of object `id`'s faces, in radians per axis; 0 for
    /// one drawn true.
    pub fn slope_error_rad(&self, id: ObjectId) -> f64 {
        self.figure_errors
            .iter()
            .find(|f| f.object == id)
            .map_or(0.0, |f| f.slope_rms_mrad * 1e-3)
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
                SolidGeometry3::Tube {
                    inner_radius_mm,
                    outer_radius_mm,
                    half_length_mm,
                } => {
                    if !inner_radius_mm.is_finite()
                        || !outer_radius_mm.is_finite()
                        || !half_length_mm.is_finite()
                    {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "geometry".to_string(),
                        });
                    }
                    // A zero bore is a cylinder, which has its own variant; a
                    // tube claiming one would carry a bore patch with no surface.
                    if *inner_radius_mm <= 0.0
                        || *outer_radius_mm <= *inner_radius_mm
                        || *half_length_mm <= 0.0
                    {
                        return Err(SceneError::OutOfRange {
                            object: oid,
                            field: "geometry".to_string(),
                            reason: "needs 0 < inner radius < outer radius and a positive length"
                                .to_string(),
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
                    SolidGeometry3::Tube { .. } => {
                        o.patch.0 <= u64::from(crate::geom3::primitives::tube_face::INNER)
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
                    outline,
                } => {
                    sag.validate(oid, "geometry.sag")?;
                    radial_aperture.validate(oid, "geometry.radial_aperture")?;
                    if let Some(o) = outline {
                        o.validate(oid, "geometry.outline")?;
                    }
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
            if d.geometry.has_area() && !grid_fits(d.bins) {
                return Err(range("bins", "holds more cells than a detector may"));
            }
            match d.measure {
                DetectorMeasure::Power => {}
                DetectorMeasure::Irradiance => {
                    if !d.geometry.has_area() {
                        return Err(range(
                            "measure",
                            "asks a line probe for irradiance, and a line has no area",
                        ));
                    }
                }
                DetectorMeasure::Illuminance => {
                    if !d.geometry.has_area() {
                        return Err(range(
                            "measure",
                            "asks a line probe for illuminance, and a line has no area",
                        ));
                    }
                }
                DetectorMeasure::Intensity {
                    half_angle_deg,
                    bins,
                    ..
                } => {
                    if !half_angle_deg.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "measure.half_angle_deg".to_string(),
                        });
                    }
                    if half_angle_deg <= 0.0 || half_angle_deg > 90.0 {
                        return Err(range("measure.half_angle_deg", "is outside (0, 90] degrees"));
                    }
                    if bins[0] == 0 || bins[1] == 0 {
                        return Err(range("measure.bins", "has no rings or no sectors"));
                    }
                    if !grid_fits(bins) {
                        return Err(range(
                            "measure.bins",
                            "holds more cells than a detector may",
                        ));
                    }
                }
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
                AngularDistribution3::RadialProfile { half_angle_rad, radiance } => {
                    if !half_angle_rad.is_finite() || radiance.iter().any(|v| !v.is_finite()) {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "angular.radial_profile".to_string(),
                        });
                    }
                    if !(half_angle_rad > 0.0 && half_angle_rad <= std::f64::consts::FRAC_PI_2) {
                        return Err(range("angular.half_angle_rad", "is outside (0, pi/2]"));
                    }
                    if radiance.iter().any(|v| *v < 0.0) || radiance.iter().all(|v| *v == 0.0) {
                        return Err(range("angular.radiance", "must be non-negative and not all zero"));
                    }
                }
                AngularDistribution3::Sun {
                    half_angle_rad,
                    csr,
                    ..
                } => {
                    if !half_angle_rad.is_finite() || !csr.is_finite() {
                        return Err(SceneError::NonFinite {
                            object: oid,
                            field: "angular.sun".to_string(),
                        });
                    }
                    if !(0.0..=0.1).contains(&half_angle_rad) {
                        return Err(range("angular.half_angle_rad", "is not a sun's disc"));
                    }
                    if !(0.0..=crate::solar::sunshape::MAX_CSR).contains(&csr) {
                        return Err(range("angular.csr", "is outside [0, 0.849]"));
                    }
                }
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
