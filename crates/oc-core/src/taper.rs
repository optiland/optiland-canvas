//! Tapers and light pipes: solids of revolution whose wall runs from a flat
//! input face to a flat exit face.
//!
//! A taper is a four-edge revolved profile, the same layout a singlet has
//! ([`BodyFace`](crate::build3::BodyFace) reads its faces at edges 0 and 2):
//!
//! 1. the input face, a line across the axis at the first wall point's `x`;
//! 2. the wall, above the axis: a line, or a polyline through every wall point;
//! 3. the exit face, a line across the axis at the last wall point's `x`;
//! 4. the wall's mirror below the axis.
//!
//! The wall is `(x, r)` points in the body's local frame, input first, `x`
//! strictly increasing and `r` positive. Two points are a straight taper (or a
//! rod, when the radii agree); more bend the wall in straight steps, each
//! revolved to a cone frustum. Light that meets the wall from inside reflects
//! by TIR or leaves by Fresnel, as at any other face.

use crate::geom3::profile::validate_revolvable;
use crate::profile2::{ClosedProfile2, Curve2, ProfileConstruction3};
use crate::scene3::{Scene3, SolidGeometry3};
use crate::vec::Vec2;

/// A taper's wall, `(x, r)` in mm in the body's local frame, input first.
#[derive(Debug, Clone, PartialEq)]
pub struct TaperGeom {
    pub wall: Vec<(f64, f64)>,
}

impl TaperGeom {
    /// The wall's axial length.
    pub fn length_mm(&self) -> f64 {
        self.wall.last().map_or(0.0, |l| l.0) - self.wall.first().map_or(0.0, |f| f.0)
    }

    /// Why this wall cannot be a taper, if it cannot.
    fn check(&self) -> Result<(), String> {
        if self.wall.len() < 2 {
            return Err("a taper's wall needs at least two points".into());
        }
        for (i, &(x, r)) in self.wall.iter().enumerate() {
            if !(x.is_finite() && r.is_finite() && r > 0.0) {
                return Err(format!("wall point {i} is not a finite point above the axis"));
            }
        }
        if self.wall.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err("the wall points must run forward along the axis".into());
        }
        Ok(())
    }
}

/// The four edges of a taper with this wall, in order.
fn taper_curves(g: &TaperGeom) -> [Curve2; 4] {
    let p = |x: f64, y: f64| Vec2::new(x, y);
    let (x0, r0) = g.wall[0];
    let (x1, r1) = *g.wall.last().unwrap();
    let side = |sign: f64, reversed: bool| -> Curve2 {
        let mut pts: Vec<Vec2> = g.wall.iter().map(|&(x, r)| p(x, sign * r)).collect();
        if reversed {
            pts.reverse();
        }
        match pts.len() {
            2 => Curve2::Line { a: pts[0], b: pts[1] },
            _ => Curve2::Polyline { points: pts },
        }
    };
    [
        Curve2::Line { a: p(x0, -r0), b: p(x0, r0) },
        side(1.0, false),
        Curve2::Line { a: p(x1, r1), b: p(x1, -r1) },
        side(-1.0, true),
    ]
}

/// The points of a line or polyline, in its own order.
fn points(c: &Curve2) -> Option<Vec<Vec2>> {
    match c {
        Curve2::Line { a, b } => Some(vec![*a, *b]),
        Curve2::Polyline { points } => Some(points.clone()),
        _ => None,
    }
}

/// Read a taper's wall out of a revolved profile, if the profile is one.
fn taper_from_profile(profile: &ClosedProfile2) -> Option<TaperGeom> {
    let e = &profile.edges;
    if e.len() != 4 {
        return None;
    }
    // A face: a line across the axis, symmetric about it, at one x.
    let face = |c: &Curve2| -> Option<(f64, f64)> {
        let Curve2::Line { a, b } = c else { return None };
        let tol = 1e-9 * (1.0 + a.x.abs() + a.y.abs());
        ((a.x - b.x).abs() <= tol && (a.y + b.y).abs() <= tol && a.y.abs() > 0.0).then_some((a.x, a.y.abs()))
    };
    let (x_in, r_in) = face(&e[0].curve)?;
    let (x_out, r_out) = face(&e[2].curve)?;
    let mut wall = points(&e[1].curve)?;
    if wall.first()?.x > wall.last()?.x {
        wall.reverse();
    }
    let g = TaperGeom { wall: wall.iter().map(|p| (p.x, p.y)).collect() };
    g.check().ok()?;
    let close = |a: (f64, f64), b: (f64, f64)| {
        (a.0 - b.0).abs() <= 1e-9 * (1.0 + a.0.abs()) && (a.1 - b.1).abs() <= 1e-9 * (1.0 + a.1.abs())
    };
    if !close(g.wall[0], (x_in, r_in)) || !close(*g.wall.last()?, (x_out, r_out)) {
        return None;
    }
    // The lower wall is the upper one's mirror.
    let mut lower = points(&e[3].curve)?;
    if lower.first()?.x > lower.last()?.x {
        lower.reverse();
    }
    let mirrored = lower.len() == g.wall.len()
        && lower.iter().zip(&g.wall).all(|(p, &(x, r))| close((p.x, -p.y), (x, r)));
    mirrored.then_some(g)
}

impl Scene3 {
    /// The wall of body `id`, if it is a taper (or a light pipe).
    pub fn taper_geom(&self, id: u64) -> Option<TaperGeom> {
        let body = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        match &body.geometry {
            SolidGeometry3::Profile(p) if p.construction == ProfileConstruction3::RevolveLocalX => {
                taper_from_profile(&p.profile)
            }
            _ => None,
        }
    }

    /// Give taper `id` a new wall. Its faces keep their patches, and with them
    /// any coating; the wall keeps its own. Refuses (scene unchanged) when `id`
    /// is not a taper, or the wall is not one: fewer than two points, a point
    /// on or below the axis, points that do not run forward along it.
    pub fn set_taper_geom(&mut self, id: u64, g: &TaperGeom) -> Result<(), String> {
        g.check()?;
        if self.taper_geom(id).is_none() {
            return Err(format!("#{id} is not a taper"));
        }
        let body = self.bodies.iter_mut().find(|b| b.common.id.0 == id).unwrap();
        let SolidGeometry3::Profile(solid) = &mut body.geometry else { unreachable!() };
        let mut rebuilt = solid.profile.clone();
        for (edge, curve) in rebuilt.edges.iter_mut().zip(taper_curves(g)) {
            edge.curve = curve;
        }
        if !rebuilt.is_finite() {
            return Err("the wall is not finite".into());
        }
        validate_revolvable(&rebuilt)?;
        solid.profile = rebuilt;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{LayerId, PatchId};
    use crate::material::Material;
    use crate::profile2::{ProfileEdge2, ProfileSolid3};
    use crate::scene3::{Body3, ObjectCommon, SurfaceProps3};

    fn scene_with(g: &TaperGeom) -> (Scene3, u64) {
        let mut s = Scene3::new();
        let id = s.alloc_object_id();
        let edges = taper_curves(g)
            .into_iter()
            .map(|curve| ProfileEdge2 { patch: s.alloc_patch_id(), curve })
            .collect();
        s.bodies.push(Body3 {
            common: ObjectCommon::new(id, LayerId(0)),
            geometry: SolidGeometry3::Profile(ProfileSolid3 {
                profile: ClosedProfile2 { edges },
                construction: ProfileConstruction3::RevolveLocalX,
            }),
            material: Material::AbbePoint { nd: 1.49, vd: 57.0 },
            absorption: Default::default(),
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        (s, id.0)
    }

    fn straight() -> TaperGeom {
        TaperGeom { wall: vec![(0.0, 2.0), (40.0, 6.0)] }
    }

    #[test]
    fn a_straight_taper_reads_back_its_wall() {
        let (s, id) = scene_with(&straight());
        assert_eq!(s.taper_geom(id), Some(straight()));
        assert_eq!(s.taper_geom(id).unwrap().length_mm(), 40.0);
    }

    #[test]
    fn a_bent_wall_is_a_polyline_and_keeps_the_faces_patches() {
        let (mut s, id) = scene_with(&straight());
        let patches = |s: &Scene3| -> Vec<PatchId> {
            let SolidGeometry3::Profile(p) = &s.bodies[0].geometry else { unreachable!() };
            p.profile.edges.iter().map(|e| e.patch).collect()
        };
        let before = patches(&s);
        let bent = TaperGeom { wall: vec![(0.0, 2.0), (10.0, 4.5), (25.0, 5.6), (40.0, 6.0)] };
        s.set_taper_geom(id, &bent).unwrap();
        assert_eq!(s.taper_geom(id), Some(bent.clone()));
        assert_eq!(patches(&s), before);
        let SolidGeometry3::Profile(p) = &s.bodies[0].geometry else { unreachable!() };
        assert!(matches!(p.profile.edges[1].curve, Curve2::Polyline { .. }));
        // And back to two points: a line again.
        s.set_taper_geom(id, &straight()).unwrap();
        let SolidGeometry3::Profile(p) = &s.bodies[0].geometry else { unreachable!() };
        assert!(matches!(p.profile.edges[1].curve, Curve2::Line { .. }));
    }

    #[test]
    fn a_wall_that_is_not_a_taper_is_refused_and_changes_nothing() {
        let (mut s, id) = scene_with(&straight());
        let before = s.bodies[0].clone();
        for wall in [
            vec![(0.0, 2.0)],
            vec![(0.0, 2.0), (0.0, 6.0)],
            vec![(0.0, 2.0), (20.0, 4.0), (15.0, 5.0), (40.0, 6.0)],
            vec![(0.0, 2.0), (40.0, 0.0)],
            vec![(0.0, 2.0), (40.0, f64::NAN)],
        ] {
            assert!(s.set_taper_geom(id, &TaperGeom { wall }).is_err());
            assert_eq!(s.bodies[0], before);
        }
    }

    #[test]
    fn a_catalog_glass_shows_its_index_and_comes_back_exactly() {
        use crate::material::{Glass, GlassId, Sellmeier};
        let (mut s, id) = scene_with(&straight());
        let pmma = Glass::sellmeier("PMMA", Sellmeier { b: [1.1819, 0.0, 0.0], c: [0.011313, 0.0, 0.0] });
        let g = s.catalog.intern(&pmma).unwrap();
        s.set_body_material(id, Material::Catalog(g)).unwrap();
        let (nd, vd, name) = s.body_index(id).unwrap();
        assert_eq!(name.as_deref(), Some("PMMA"));
        assert!((nd - pmma.nd).abs() < 1e-12 && (vd - pmma.vd).abs() < 1e-12);
        // An Abbe point replaces it; the saved material puts it back.
        let saved = s.body_material(id).unwrap().clone();
        assert!(s.set_body_glass_abbe(id, 1.585, 30.0));
        assert_eq!(s.body_index(id), Some((1.585, 30.0, None)));
        s.set_body_material(id, saved).unwrap();
        assert_eq!(s.bodies[0].material, Material::Catalog(g));
        // A catalog glass the scene does not hold is refused.
        assert!(s.set_body_material(id, Material::Catalog(GlassId(999))).is_err());
        assert_eq!(s.bodies[0].material, Material::Catalog(g));
    }

    #[test]
    fn a_singlet_is_not_a_taper() {
        let mut s = Scene3::new();
        let id = s.add_singlet(crate::vec3::Vec3::ZERO, crate::build3::SingletGeom::biconvex(), 1.5, 60.0);
        assert!(s.taper_geom(id.0).is_none());
    }
}
