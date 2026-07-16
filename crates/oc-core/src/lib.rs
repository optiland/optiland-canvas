//! # oc-core
//!
//! Scene model, geometry, and materials for Optiland Canvas. **No I/O, no GPU.**
//! This crate is the library the app consumes; it must be worth using on its own
//! (see the spec's repository-layout note).
//!
//! Everything is f64. Wavelengths are in micrometres. The serialized [`Scene`]
//! *is* the `.ocanvas` document format.

//! ## The 3D migration
//!
//! The canonical scene is becoming 3D, and the 2D Canvas a view over it
//! (`docs/architecture/canonical-3d-kernel.md`). That work lands behind the
//! off-by-default **`scene3`** feature, so the default build is exactly the 2D
//! engine it was. Nothing below `scene3` is wired into the browser yet.

#[cfg(feature = "scene3")]
pub mod coordinates;
pub mod fragment;
pub mod geom;
pub mod material;
pub mod scene;
pub mod seq;
pub mod vec;

pub use fragment::{
    FragmentError, IdRemap, InsertedFragment, ObjectIdSet, SceneFragment, Transform2,
};
pub use geom::{Aabb, ConicParams, Edge, EdgeHit, Ray};
pub use material::{
    abbe_number, convex_hull, point_in_hull, reference_glasses, AbbeGlass, Cauchy, Glass,
    GlassCatalog, GlassId, GrinField, Material, Sellmeier, LAMBDA_C, LAMBDA_D, LAMBDA_F,
};
pub use scene::{
    spherical_edge, Detector, DetectorId, Element, ElementHit, ElementId, EmittedRay, Environment,
    Layer, LayerId, Region, Scene, Source, SourceId, SourceKind, Spectrum, SurfaceProps,
    SurfaceSpec, OCANVAS_FORMAT, OCANVAS_VERSION,
};
pub use seq::{FieldType, SeqAperture, SeqMaterial, SeqSurface, SeqSystem};
pub use vec::{perp, Isometry2, Vec2};
