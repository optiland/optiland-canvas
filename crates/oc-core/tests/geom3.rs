//! 3D analytic geometry: intersection, normals, containment, bounds.
//!
//! **Scaffolding.** Empty until PR 3D-03 fills it in; gated so that
//! `cargo test -p oc-core --test geom3` is a stable command from today and
//! reports zero tests until then.
//!
//! What this suite owes (contract §15.1, §15.6):
//!
//! - plane, sphere, cylinder, box intersection;
//! - conic-of-revolution intersection and **analytic** normal;
//! - even-asphere root and **analytic** normal (no finite difference: that is
//!   the known ~1e-9 floor on the current 2D asphere, and 3D must not inherit
//!   it);
//! - extruded profile sides and caps;
//! - revolved profile containment;
//! - event deduplication at tangency and at patch seams;
//! - conservative AABBs: too large is allowed, missing a hit is not;
//! - transform invariance of intersections;
//! - property/fuzz invariants: no NaN in an accepted hit, `t > t_min`, unit
//!   normals, scale invariance from 1e-3 mm to 1e4 mm.
#![cfg(feature = "scene3")]
