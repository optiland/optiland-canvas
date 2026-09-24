//! [`Intersect3`] for the closed solids: sphere, box, cylinder, and the two
//! profile constructions.
//!
//! Each impl pulls the world ray into the geometry's local frame, calls a
//! transform-free primitive, and reports the hit in local coordinates. A
//! `PlanarOnly` profile has no 3D body, so it intersects nothing and contains
//! nothing — it answers [`Containment::Unsupported`], which is the signal a
//! full-3D trace refuses on rather than inventing a depth (§3.2).

use super::primitives::{
    box_contains, box_intersect, cylinder_contains, cylinder_intersect, sphere_contains,
    sphere_intersect, tube_contains, tube_intersect, RawHit3,
};
use super::profile::{
    extrude_aabb, extrude_contains, extrude_intersect, revolve_aabb, revolve_contains,
    revolve_intersect, revolve_intersect_with, revolve_tolerance,
};
use super::{ray_to_local, Containment, Intersect3, LocalHit3, ToleranceContext};
use crate::id::PatchId;
use crate::profile2::{ProfileConstruction3, ProfileSolid3};
use crate::scene3::SolidGeometry3;
use crate::vec3::{Aabb3, Isometry3, Ray3, Vec3};

/// The unit axis of a cylinder, defaulting to `+X` if the stored axis is
/// unusable. Validation refuses a zero axis on load, so the fallback only guards
/// a value built in memory.
fn cylinder_axis(axis_local: Vec3) -> Vec3 {
    if axis_local.length() >= crate::vec3::MIN_NORM {
        axis_local.normalize()
    } else {
        Vec3::X
    }
}

/// Transform a local box to a conservative world box by bounding its eight
/// transformed corners. A rigid transform of a box is not axis-aligned, so the
/// smallest enclosing axis-aligned box is what a BVH can use — larger than the
/// oriented box, never smaller (§7.1).
fn aabb_local_to_world(local: Aabb3, transform: &Isometry3) -> Aabb3 {
    let mut bb = Aabb3::EMPTY;
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { local.min.x } else { local.max.x },
            if i & 2 == 0 { local.min.y } else { local.max.y },
            if i & 4 == 0 { local.min.z } else { local.max.z },
        );
        bb.expand(transform.transform_point(corner));
    }
    bb
}

/// Local bounding box of a solid, before placement.
fn local_aabb(geom: &SolidGeometry3) -> Aabb3 {
    match geom {
        SolidGeometry3::Sphere { radius_mm } => Aabb3 {
            min: Vec3::splat(-radius_mm),
            max: Vec3::splat(*radius_mm),
        },
        SolidGeometry3::Box { half_extents_mm } => Aabb3 {
            min: -*half_extents_mm,
            max: *half_extents_mm,
        },
        SolidGeometry3::Cylinder {
            radius_mm,
            half_length_mm,
            axis_local,
        } => {
            // The tight box of a finite cylinder about an arbitrary axis: along
            // each local axis, the length projects as `L·|aᵢ|` and the radius as
            // `r·√(1−aᵢ²)`.
            let a = cylinder_axis(*axis_local);
            let ext = Vec3::new(
                half_length_mm * a.x.abs() + radius_mm * (1.0 - a.x * a.x).max(0.0).sqrt(),
                half_length_mm * a.y.abs() + radius_mm * (1.0 - a.y * a.y).max(0.0).sqrt(),
                half_length_mm * a.z.abs() + radius_mm * (1.0 - a.z * a.z).max(0.0).sqrt(),
            );
            Aabb3 {
                min: -ext,
                max: ext,
            }
        }
        SolidGeometry3::Tube {
            outer_radius_mm,
            half_length_mm,
            ..
        } => Aabb3 {
            min: Vec3::new(-half_length_mm, -outer_radius_mm, -outer_radius_mm),
            max: Vec3::new(*half_length_mm, *outer_radius_mm, *outer_radius_mm),
        },
        SolidGeometry3::Profile(p) => profile_aabb(p),
    }
}

fn profile_aabb(p: &ProfileSolid3) -> Aabb3 {
    match p.construction {
        ProfileConstruction3::RevolveLocalX => revolve_aabb(&p.profile),
        ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
            extrude_aabb(&p.profile, z_min_mm, z_max_mm)
        }
        // A planar-only body has no 3D extent to bound. A thin box in its plane is
        // conservative for the compatibility path and keeps the value finite.
        ProfileConstruction3::PlanarOnly { plane_z_mm } => {
            let flat = extrude_aabb(&p.profile, plane_z_mm, plane_z_mm);
            flat.padded(0.0)
        }
    }
}

/// Wrap an analytic primitive's [`RawHit3`] as a [`LocalHit3`], turning the face
/// index into a stable synthetic patch. An analytic solid carries no authored
/// patches, and validation refuses a per-patch override on every one but the
/// tube, whose faces are fixed by construction and so are stable ids already
/// ([`tube_face`](super::primitives::tube_face)).
fn raw_to_local(raw: RawHit3) -> LocalHit3 {
    LocalHit3 {
        t_mm: raw.t_mm,
        point_local: raw.point,
        normal_local: raw.normal,
        patch: PatchId(raw.face as u64),
        uv: raw.uv,
    }
}

/// What intersecting a solid needs that depends on its shape alone: found once
/// for a trace ([`SolidGeometry3::prepare`]) instead of at every step.
#[derive(Debug, Clone, Copy, Default)]
pub struct SolidPrep {
    /// A revolved profile's own tolerances ([`revolve_tolerance`]).
    revolve_tol: Option<ToleranceContext>,
}

impl SolidGeometry3 {
    /// What [`SolidGeometry3::intersect_prepared`] can reuse from one step to
    /// the next.
    pub fn prepare(&self) -> SolidPrep {
        match self {
            SolidGeometry3::Profile(p) if p.construction == ProfileConstruction3::RevolveLocalX => {
                SolidPrep {
                    revolve_tol: Some(revolve_tolerance(&p.profile)),
                }
            }
            _ => SolidPrep::default(),
        }
    }

    /// [`Intersect3::intersect`], with the work that depends on the shape
    /// alone taken from `prep` rather than done again. The same hits, to the
    /// bit, for a `prep` this geometry prepared.
    pub fn intersect_prepared(
        &self,
        prep: &SolidPrep,
        ray_world: &Ray3,
        transform: &Isometry3,
        t_min_mm: f64,
        t_max_mm: f64,
        out: &mut dyn FnMut(LocalHit3),
    ) {
        match (self, &prep.revolve_tol) {
            (SolidGeometry3::Profile(p), Some(tol))
                if p.construction == ProfileConstruction3::RevolveLocalX =>
            {
                let ray = ray_to_local(transform, ray_world);
                revolve_intersect_with(&p.profile, tol, &ray, t_min_mm, t_max_mm, out)
            }
            _ => self.intersect(ray_world, transform, t_min_mm, t_max_mm, out),
        }
    }
}

impl Intersect3 for SolidGeometry3 {
    fn aabb_world(&self, transform: &Isometry3) -> Aabb3 {
        aabb_local_to_world(local_aabb(self), transform)
    }

    fn intersect(
        &self,
        ray_world: &Ray3,
        transform: &Isometry3,
        t_min_mm: f64,
        t_max_mm: f64,
        out: &mut dyn FnMut(LocalHit3),
    ) {
        let ray = ray_to_local(transform, ray_world);
        match self {
            SolidGeometry3::Sphere { radius_mm } => {
                sphere_intersect(*radius_mm, &ray, t_min_mm, t_max_mm, &mut |h| {
                    out(raw_to_local(h))
                })
            }
            SolidGeometry3::Box { half_extents_mm } => {
                box_intersect(*half_extents_mm, &ray, t_min_mm, t_max_mm, &mut |h| {
                    out(raw_to_local(h))
                })
            }
            SolidGeometry3::Cylinder {
                radius_mm,
                half_length_mm,
                axis_local,
            } => cylinder_intersect(
                *radius_mm,
                *half_length_mm,
                cylinder_axis(*axis_local),
                &ray,
                t_min_mm,
                t_max_mm,
                &mut |h| out(raw_to_local(h)),
            ),
            SolidGeometry3::Tube {
                inner_radius_mm,
                outer_radius_mm,
                half_length_mm,
            } => tube_intersect(
                *inner_radius_mm,
                *outer_radius_mm,
                *half_length_mm,
                &ray,
                t_min_mm,
                t_max_mm,
                &mut |h| out(raw_to_local(h)),
            ),
            SolidGeometry3::Profile(p) => match p.construction {
                ProfileConstruction3::RevolveLocalX => {
                    revolve_intersect(&p.profile, &ray, t_min_mm, t_max_mm, out)
                }
                ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => extrude_intersect(
                    &p.profile, z_min_mm, z_max_mm, &ray, t_min_mm, t_max_mm, out,
                ),
                // No 3D body exists yet; refuse to trace rather than invent one.
                ProfileConstruction3::PlanarOnly { .. } => {}
            },
        }
    }

    fn contains(
        &self,
        point_world: Vec3,
        transform: &Isometry3,
        tol: &ToleranceContext,
    ) -> Containment {
        let p = transform.inverse_transform_point(point_world);
        match self {
            SolidGeometry3::Sphere { radius_mm } => sphere_contains(*radius_mm, p, tol),
            SolidGeometry3::Box { half_extents_mm } => box_contains(*half_extents_mm, p, tol),
            SolidGeometry3::Cylinder {
                radius_mm,
                half_length_mm,
                axis_local,
            } => cylinder_contains(
                *radius_mm,
                *half_length_mm,
                cylinder_axis(*axis_local),
                p,
                tol,
            ),
            SolidGeometry3::Tube {
                inner_radius_mm,
                outer_radius_mm,
                half_length_mm,
            } => tube_contains(*inner_radius_mm, *outer_radius_mm, *half_length_mm, p, tol),
            SolidGeometry3::Profile(prof) => match prof.construction {
                ProfileConstruction3::RevolveLocalX => revolve_contains(&prof.profile, p, tol),
                ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
                    extrude_contains(&prof.profile, z_min_mm, z_max_mm, p, tol)
                }
                ProfileConstruction3::PlanarOnly { .. } => Containment::Unsupported,
            },
        }
    }
}
