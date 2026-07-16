//! Stable identity for the canonical scene.
//!
//! A v1 id **is** a vector index ([`crate::scene`]: "IDs are indices into the
//! respective vectors", and `add_element` hands out `ElementId(len)`). Every
//! persistent reference is therefore a position, and every insertion or removal
//! moves it. `remove_objects` returning an `IdRemap` exists to chase that: each
//! structural mutator has to renumber every reference in the scene, and any
//! mutator that forgets leaves a reference resolving to a *different object*
//! rather than to nothing. That failure is silent, and the picture still looks
//! right.
//!
//! These ids are not positions. They are allocated, never reused within a scene
//! lifetime, and unaffected by vector order, which stays presentation and
//! serialization order only. Removing an object leaves every survivor's id
//! alone, so there is nothing to remap and no mutator that can forget to.
//!
//! **There is deliberately no `index()` here.** The v1 macro has one, and it is
//! what makes `scene.elements[id.index()]` spellable. A v2 id is resolved by
//! looking it up.
//!
//! Ids are allocated by `Scene3` (PR 3D-02), which also persists `next_id` or
//! recomputes it as `max(existing) + 1` on load. Nothing else mints one.
//!
//! # Naming
//!
//! [`SourceId`], [`DetectorId`], and [`LayerId`] collide with the v1 types of
//! the same name in [`crate::scene`], so this module is **not** re-exported from
//! the crate root during the migration. Spell them `oc_core::id::SourceId`. The
//! collision is the point: the two are not interchangeable, and an import that
//! has to say which one it means cannot silently pick the wrong one. PR 3D-11
//! removes the v1 types and the qualification with them.

use serde::{Deserialize, Serialize};

/// Declare a stable id.
///
/// `u64` because ids are allocated monotonically and never reused: the count
/// rises with edits over a scene's lifetime rather than with the number of live
/// objects, so a `u32` is a wraparound waiting for a long session. Wrapping
/// would hand out an id that is already in use, which is the one thing this
/// module exists to prevent.
macro_rules! id_type {
    ($name:ident, $repr:ty) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub $repr);

        impl $name {
            /// The raw value. For serialization, diagnostics, and allocation
            /// arithmetic. **Not** a vector index.
            #[inline]
            pub fn raw(self) -> $repr {
                self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }
    };
}

id_type!(ObjectId, u64);
id_type!(BodyId, u64);
id_type!(SurfaceObjectId, u64);
id_type!(SourceId, u64);
id_type!(DetectorId, u64);
id_type!(ApertureId, u64);
id_type!(LayerId, u64);
id_type!(PatchId, u64);
id_type!(ViewId, u64);
id_type!(PathIntentId, u64);
id_type!(PathId, u64);

/// One surface of one object.
///
/// A patch is only meaningful inside its object, so a reference to a surface is
/// both halves. This is what a stop designation, a per-patch response override,
/// and a path record hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SurfaceRef {
    pub object: ObjectId,
    pub patch: PatchId,
}

impl SurfaceRef {
    pub fn new(object: ObjectId, patch: PatchId) -> Self {
        Self { object, patch }
    }
}

impl std::fmt::Display for SurfaceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.object, self.patch)
    }
}
