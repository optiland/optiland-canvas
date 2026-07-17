//! # oc-core
//!
//! Scene model, geometry, and materials for Optiland Canvas. **No I/O, no GPU.**
//! This crate is the library the app consumes; it must be worth using on its own
//! (see the spec's repository-layout note).
//!
//! Everything is f64. Wavelengths are in micrometres. The serialized [`Scene`]
//! *is* the `.ocanvas` document format.
//!
//! ## The 3D migration
//!
//! The canonical scene is becoming 3D, and the 2D Canvas a view over it
//! (`docs/architecture/canonical-3d-kernel.md`).
//!
//! The **math** layer ([`vec3`], [`id`], [`coordinates`]) is always compiled. It
//! is additive and complete: no existing type changed, no existing behaviour
//! moved, and nothing in the 2D engine calls it except `oc-openuc2`, which uses
//! [`coordinates`] because §3.7 allows exactly one conversion module and a
//! second copy of the same axis mapping is the drift that rule exists to
//! prevent.
//!
//! The **model** layer (`Scene3`, 3D geometry, the 3D tracer) lands behind the
//! off-by-default **`scene3`** feature, so the shipped engine cannot reach a
//! half-built scene model while the migration is in progress.

pub mod coordinates;
pub mod fragment;
pub mod geom;
pub mod id;
pub mod material;
#[cfg(feature = "scene3")]
pub mod migrate_v1;
#[cfg(feature = "scene3")]
pub mod profile2;
pub mod scene;
#[cfg(feature = "scene3")]
pub mod scene3;
pub mod seq;
pub mod vec;
pub mod vec3;

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
pub use vec3::{
    Aabb3, Isometry3, Isometry3Error, Plane3, Plane3Error, Ray3, Ray3Error, Vec3, MIN_NORM,
};

// `id` is deliberately not re-exported: its `SourceId`, `DetectorId`, and
// `LayerId` collide with the v1 types above, and the two are not
// interchangeable. Spell them `oc_core::id::SourceId`. See the module docs.
