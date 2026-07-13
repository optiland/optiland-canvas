//! # oc-core
//!
//! Scene model, geometry, and materials for Optiland Canvas. **No I/O, no GPU.**
//! This crate is the library the app consumes; it must be worth using on its own
//! (see the spec's repository-layout note).
//!
//! Everything is f64. Wavelengths are in micrometres. The serialized [`Scene`]
//! *is* the `.ocanvas` document format.

pub mod geom;
pub mod material;
pub mod scene;
pub mod vec;

pub use geom::{Aabb, ConicParams, Edge, EdgeHit, Ray};
pub use material::{
    abbe_number, convex_hull, point_in_hull, reference_glasses, AbbeGlass, Cauchy, Glass,
    GlassCatalog, GlassId, GrinField, Material, Sellmeier, LAMBDA_C, LAMBDA_D, LAMBDA_F,
};
pub use scene::{
    spherical_edge, Detector, DetectorId, Element, ElementHit, ElementId, EmittedRay, Environment,
    Layer, LayerId, Region, Scene, Source, SourceId, SourceKind, Spectrum, SurfaceProps,
    SurfaceSpec,
};
pub use vec::{perp, Isometry2, Vec2};
