//! Typed failures for the adapter.

use crate::coordinates::InvalidQuadrant;
use oc_core::FragmentError;
use serde::Serialize;
use std::fmt;

/// Anything that can go wrong loading, placing, or exporting.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The catalogue or setup JSON did not parse, or is not the expected format.
    Format(String),
    /// The setup names a grid profile the catalogue does not define.
    UnknownGridProfile(String),
    /// The setup names an instance the project does not have.
    UnknownInstance(String),
    /// A rotation that is not a whole quarter turn.
    Rotation(InvalidQuadrant),
    /// A module's Canvas fragment is unusable.
    Fragment(FragmentError),
    /// A placement was refused.
    Placement(PlacementError),
    /// A descriptor names a generator this build does not have (§9.5).
    ///
    /// Never a fallback: the set of builders is closed and reviewed, and quietly
    /// substituting a different one would build somebody else's lens.
    UnknownGenerator(String),
    /// A generated module's parameters do not describe the thing they claim to.
    Parameter(String),
    /// A parameter names a glass the builtin catalogue does not have.
    UnknownGlass(String),
}

/// Why a module could not go where it was asked to go.
#[derive(Debug, Clone, PartialEq)]
pub enum PlacementError {
    /// Cells are already taken by another enabled instance on the same layer.
    Collision {
        /// The instance already holding the cell.
        occupied_by: String,
        cell: [i32; 3],
    },
    /// The grid profile does not permit this rotation.
    RotationNotAllowed { degrees: i32 },
    /// The catalogue has no module with this identity.
    UnknownModule { uuid: String, version: String },
    /// The module resolved, but its fragment could not be placed.
    Fragment(FragmentError),
    /// The scene could not be rebuilt after the placement changed.
    Rebuild(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Format(m) => write!(f, "{m}"),
            Error::UnknownGridProfile(id) => {
                write!(f, "the catalogue defines no grid profile {id:?}")
            }
            Error::UnknownInstance(id) => write!(f, "no such instance: {id:?}"),
            Error::Rotation(e) => write!(f, "{e}"),
            Error::Fragment(e) => write!(f, "{e}"),
            Error::Placement(e) => write!(f, "{e}"),
            Error::UnknownGenerator(name) => write!(
                f,
                "this build has no {name:?} generator; a descriptor may only name a builder \
                 that exists (§9.5)"
            ),
            Error::Parameter(m) => write!(f, "{m}"),
            Error::UnknownGlass(name) => {
                write!(f, "the glass catalogue has no {name:?}")
            }
        }
    }
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementError::Collision { occupied_by, cell } => write!(
                f,
                "cell [{}, {}, {}] is already occupied by instance {occupied_by:?}",
                cell[0], cell[1], cell[2]
            ),
            PlacementError::RotationNotAllowed { degrees } => {
                write!(f, "this grid does not allow a {degrees}° rotation")
            }
            PlacementError::UnknownModule { uuid, version } => write!(
                f,
                "the catalogue has no module {uuid} at version {version}"
            ),
            PlacementError::Fragment(e) => write!(f, "{e}"),
            PlacementError::Rebuild(m) => write!(f, "the scene could not be rebuilt: {m}"),
        }
    }
}

impl From<Error> for PlacementError {
    fn from(e: Error) -> Self {
        match e {
            Error::Placement(p) => p,
            Error::Fragment(f) => PlacementError::Fragment(f),
            other => PlacementError::Rebuild(other.to_string()),
        }
    }
}

impl std::error::Error for Error {}
impl std::error::Error for PlacementError {}

impl From<FragmentError> for Error {
    fn from(e: FragmentError) -> Self {
        Error::Fragment(e)
    }
}

impl From<PlacementError> for Error {
    fn from(e: PlacementError) -> Self {
        Error::Placement(e)
    }
}

impl From<InvalidQuadrant> for Error {
    fn from(e: InvalidQuadrant) -> Self {
        Error::Rotation(e)
    }
}

/// Something the user should know that is not a failure.
///
/// Warnings are how the adapter reports reduced fidelity instead of silently
/// approximating it: a module that is layout-only, an instance stranded on
/// another layer, a version the catalogue no longer has. They never block a
/// trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectWarning {
    /// The catalogue has no such `(uuid, version)`; the instance is a
    /// placeholder built from the setup's stored fallback data.
    UnresolvedModule {
        instance: String,
        uuid: String,
        version: String,
        /// The module id the setup remembered, for a human to search on.
        fallback_id: String,
    },
    /// The module resolved but carries no Canvas optics, so it occupies space
    /// and appears in the BOM while contributing nothing to the trace.
    LayoutOnly {
        instance: String,
        module_id: String,
        /// The declared optical capability: `absent`, `placeholder`, …
        optical: String,
    },
    /// The module's optics are declared approximate.
    IdealizedOptics {
        instance: String,
        module_id: String,
        fidelity: String,
    },
    /// The instance sits on a layer the active 2D trace does not cover, so it is
    /// mechanical context only.
    InactiveLayer {
        instance: String,
        module_id: String,
        layer: i32,
        active_layer: i32,
    },
    /// The module has a Canvas fragment declared, but its asset was never
    /// attached to the catalogue, so there is nothing to place.
    MissingFragmentAsset {
        instance: String,
        module_id: String,
        asset: String,
    },
    /// This instance's optics were rebuilt from parameters the user changed, so
    /// the module's `validated` badge no longer describes what is in the scene.
    ///
    /// A validation compares a *specific configuration* against a reference
    /// (§9.5). Change a radius and that comparison is about a different lens. The
    /// geometry is still built by reviewed code and still traces correctly — what
    /// is gone is the evidence that it matches anything.
    ReconfiguredOptics {
        instance: String,
        module_id: String,
        /// The overridden parameter names, sorted.
        keys: Vec<String>,
    },
}

impl fmt::Display for ProjectWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProjectWarning::UnresolvedModule {
                instance,
                uuid,
                version,
                fallback_id,
            } => write!(
                f,
                "{instance}: the catalogue has no {fallback_id} ({uuid}) at version {version}; \
                 kept as an unresolved placeholder rather than substituting another version"
            ),
            ProjectWarning::LayoutOnly {
                instance,
                module_id,
                optical,
            } => write!(
                f,
                "{instance}: {module_id} has no Canvas optics (optical: {optical}); \
                 it occupies grid space but does not affect the rays"
            ),
            ProjectWarning::IdealizedOptics {
                instance,
                module_id,
                fidelity,
            } => write!(
                f,
                "{instance}: {module_id} optics are {fidelity}, not validated against a reference"
            ),
            ProjectWarning::InactiveLayer {
                instance,
                module_id,
                layer,
                active_layer,
            } => write!(
                f,
                "{instance}: {module_id} is on layer {layer} and the trace covers layer \
                 {active_layer}; Canvas traces one layer at a time, so it is context only"
            ),
            ProjectWarning::ReconfiguredOptics {
                instance,
                module_id,
                keys,
            } => write!(
                f,
                "{instance}: {module_id} optics were rebuilt from changed {}; the module's \
                 validation compared a different configuration, so it no longer covers this one",
                keys.join(", ")
            ),
            ProjectWarning::MissingFragmentAsset {
                instance,
                module_id,
                asset,
            } => write!(
                f,
                "{instance}: {module_id} declares the Canvas fragment {asset:?}, but that asset \
                 was not attached to the catalogue"
            ),
        }
    }
}
