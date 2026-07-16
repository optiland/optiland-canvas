//! 3D optical analytic fixtures: the proof the engine is not merely meridional.
//!
//! **Scaffolding.** Empty until PR 3D-04 fills it in; gated so that
//! `cargo test -p oc-trace --test analytic3` is a stable command from today and
//! reports zero tests until then.
//!
//! What this suite owes (contract §15.2):
//!
//! - arbitrary-azimuth Snell: `n1 sin θi = n2 sin θt` to 1e-12, with the
//!   transmitted ray in the plane of incidence and unit length;
//! - arbitrary 3D reflection: equal angles, tangential component preserved to
//!   1e-13;
//! - TIR just below and just above the critical angle, with a skew component;
//! - **spherical lens skew ray**: non-zero out-of-plane field and pupil
//!   coordinates through a centered singlet, against Optiland's sequential
//!   real-ray tracer;
//! - **tangential and sagittal fans distinct** for an off-axis field. This is
//!   the release proof (§20);
//! - out-of-plane fold: a mirror that puts light on a non-coplanar detector;
//! - a skew ray through a finite extruded prism against hand vector-Snell;
//! - aperture clipping, circular/annular/rectangular/polygonal, from both
//!   propagation directions;
//! - detector area normalization: uniform illumination gives equal flux density
//!   across equal-area cells.
#![cfg(feature = "trace3")]
