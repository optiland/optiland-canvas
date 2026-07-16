//! The 2D tracer is an exact reduction of the 3D one, or it is not used.
//!
//! **Scaffolding.** Empty until PR 3D-05 fills it in; gated so that
//! `cargo test -p oc-trace --test planar_equivalence` is a stable command from
//! today and reports zero tests until then.
//!
//! This is the suite that earns the right to keep the fast planar path. Without
//! it, "the 2D result is the same as the 3D one" is an assertion rather than a
//! measurement, and the reducer becomes exactly the silent approximation the
//! contract exists to forbid (§3.5).
//!
//! What this suite owes (contract §10.2, §15.3):
//!
//! For every existing golden 2D scene that promotes exactly: build the v2 3D
//! scene, trace the original 2D f64 scene, trace the 3D f64 scene with rays in
//! `Z = 0`, and compare every event position, direction, flux, and OPL.
//!
//! | Quantity | Lines/conics | Even asphere |
//! |---|---:|---:|
//! | position | `<= 1e-10 mm` | `<= 1e-8 mm` |
//! | direction | `<= 1e-12` | `<= 1e-12` |
//! | OPL | `<= 1e-10 mm` | `<= 1e-8 mm` |
//!
//! And the negative cases, which matter more than the positive ones: the
//! reducer must **refuse** an out-of-plane mirror and a non-planar source
//! rather than reduce them anyway.
#![cfg(feature = "trace3")]
