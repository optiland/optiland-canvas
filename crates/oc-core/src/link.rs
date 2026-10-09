//! One element locked to another.
//!
//! A heliostat is the first element whose pose the kernel decides rather than
//! the author: the sun and its target turn it. A [`Link3`] is the general form
//! of that for any two elements, in one of three ways:
//!
//! - [`LinkMode::Follow`]: the element is rigidly attached to its parent. When
//!   the parent moves or turns, the element moves and turns with it, keeping
//!   the pose it had relative to the parent. An aperture fixed in front of a
//!   heliostat mirror follows the mirror as it tracks.
//! - [`LinkMode::Front`]: the element sits on the parent's axis, `gap_mm` in
//!   front of it and parallel to it, and stays there as the parent moves: a
//!   baffle in front of a heliostat, a filter on a lens. The parent's axis is
//!   its face normal for a mirror, aperture or detector (local `+Z`) and its
//!   optical axis for a lens, tube or other body (local `+X`); the element
//!   is turned so its own axis lies along it.
//! - [`LinkMode::Face`]: the element keeps its position and turns to point at
//!   the parent, its own axis toward the parent's origin.
//!
//! Like the solar rig this is authored intent, resolved into the ordinary
//! transforms the tracer reads by [`Scene3::resolve_links`]. The tracer never
//! reads a link.
//!
//! A followed element can still be moved by hand. The link remembers the pose
//! it last gave the element, and an element found somewhere else has been
//! moved by its author: its new pose relative to the parent becomes the
//! offset. So dragging it, or typing a position into its card, adjusts where
//! it sits on the parent rather than fighting the link.

use crate::id::ObjectId;
use crate::scene3::{Scene3, SceneError};
use crate::vec3::{Isometry3, Vec3};
use serde::{Deserialize, Serialize};

/// How a linked element follows its parent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LinkMode {
    /// Rigidly attached: moves and turns with the parent, keeping the pose it
    /// had on it.
    Follow,
    /// Parallel to the parent, on its axis, `gap_mm` in front of it (negative:
    /// behind).
    Front { gap_mm: f64 },
    /// Stays where it is and turns to point at the parent.
    Face,
}

impl LinkMode {
    pub fn name(self) -> &'static str {
        match self {
            LinkMode::Follow => "follow",
            LinkMode::Front { .. } => "front",
            LinkMode::Face => "face",
        }
    }

    /// A mode from its name; `gap_mm` is used by `front`.
    pub fn parse(s: &str, gap_mm: f64) -> Option<Self> {
        match s {
            "follow" => Some(LinkMode::Follow),
            "front" => gap_mm.is_finite().then_some(LinkMode::Front { gap_mm }),
            "face" => Some(LinkMode::Face),
            _ => None,
        }
    }
}

/// One element locked to another. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Link3 {
    pub child: ObjectId,
    pub parent: ObjectId,
    pub mode: LinkMode,
    /// `Follow`: the child's pose in the parent's frame.
    pub offset: Isometry3,
    /// The pose the link last gave the child. A child found elsewhere was
    /// moved by hand, and its new pose sets the offset.
    pub placed: Isometry3,
}

/// Poses equal to well below anything a card shows or a drag makes.
fn same_pose(a: &Isometry3, b: &Isometry3) -> bool {
    let q = a.quat().dot(b.quat()).abs();
    (a.pos - b.pos).length() < 1e-7 && q > 1.0 - 1e-12
}

impl Scene3 {
    fn transform_of(&self, id: ObjectId) -> Option<Isometry3> {
        let n = id.0;
        self.bodies
            .iter()
            .map(|b| &b.common)
            .chain(self.surfaces.iter().map(|s| &s.common))
            .chain(self.apertures.iter().map(|a| &a.common))
            .chain(self.detectors.iter().map(|d| &d.common))
            .find(|c| c.id.0 == n)
            .map(|c| c.transform)
    }

    /// Is `id` a body (whose axis is local `+X`) rather than a planar element
    /// (whose face normal is local `+Z`)?
    fn is_body(&self, id: ObjectId) -> bool {
        self.bodies.iter().any(|b| b.common.id == id)
    }

    /// Is `id` a heliostat mirror? The sun turns those; a link may not.
    fn is_heliostat(&self, id: ObjectId) -> bool {
        self.solar
            .as_ref()
            .is_some_and(|r| r.heliostats.iter().any(|h| h.mirror == id))
    }

    /// Lock `child` to `parent`. Replaces any link `child` already has. The
    /// offset of a `Follow` link is the pose the child has now, so linking
    /// moves nothing; `Front` and `Face` place or turn the child at once.
    ///
    /// Refused, with the scene unchanged, for an element linked to itself, an
    /// id that is not a placed element (a body, mirror, aperture or detector),
    /// a heliostat mirror as the child (the sun already turns it), or a link
    /// that would close a loop.
    pub fn link_object(&mut self, child: u64, parent: u64, mode: LinkMode) -> Result<(), String> {
        let (c, p) = (ObjectId(child), ObjectId(parent));
        if c == p {
            return Err("an element cannot be locked to itself".into());
        }
        let Some(ct) = self.transform_of(c) else {
            return Err(format!("#{child} is not an element that can be locked"));
        };
        let Some(pt) = self.transform_of(p) else {
            return Err(format!("#{parent} is not an element to lock to"));
        };
        if self.is_heliostat(c) {
            return Err(format!("#{child} is a heliostat: the sun already turns it"));
        }
        // Walk up from the parent: meeting the child means a loop.
        let mut at = p;
        for _ in 0..=self.links.len() {
            match self.links.iter().find(|l| l.child == at) {
                Some(l) if l.parent == c => {
                    return Err(format!(
                        "#{parent} already follows #{child}: that would be a loop"
                    ))
                }
                Some(l) => at = l.parent,
                None => break,
            }
        }
        let was = self.links.clone();
        self.links.retain(|l| l.child != c);
        self.links.push(Link3 {
            child: c,
            parent: p,
            mode,
            offset: pt.inverse().compose(ct),
            placed: ct,
        });
        self.resolve_links();
        if let Err(e) = self.validate() {
            self.links = was;
            if let Some(t) = self.object_transform_mut(child) {
                *t = ct;
            }
            return Err(e.to_string());
        }
        Ok(())
    }

    /// Release `child` from its link, leaving it where it is. False if it had
    /// none.
    pub fn unlink_object(&mut self, child: u64) -> bool {
        let before = self.links.len();
        self.links.retain(|l| l.child.0 != child);
        self.links.len() != before
    }

    /// The link `child` has, if any.
    pub fn link_of(&self, child: u64) -> Option<&Link3> {
        self.links.iter().find(|l| l.child.0 == child)
    }

    /// Write every link into the transforms it governs, parents before
    /// children. Links whose child or parent no longer exists are dropped
    /// first, which is the repair a delete needs.
    pub fn resolve_links(&mut self) {
        let live = |s: &Scene3, id: ObjectId| s.transform_of(id).is_some();
        let links: Vec<Link3> = self.links.clone();
        let kept: Vec<Link3> = links
            .into_iter()
            .filter(|l| live(self, l.child) && live(self, l.parent))
            .collect();
        self.links = kept;
        // Depth in the link tree: a parent that follows something resolves
        // first. Loops are refused when made, but a hand-edited file could
        // hold one, so the walk is bounded.
        let depth = |s: &Scene3, l: &Link3| {
            let mut d = 0;
            let mut at = l.parent;
            while let Some(up) = s.links.iter().find(|x| x.child == at) {
                d += 1;
                at = up.parent;
                if d > s.links.len() {
                    break;
                }
            }
            d
        };
        let mut order: Vec<(usize, usize)> = (0..self.links.len())
            .map(|i| (depth(self, &self.links[i]), i))
            .collect();
        order.sort();
        for (_, i) in order {
            let l = self.links[i].clone();
            let (Some(pt), Some(ct)) = (self.transform_of(l.parent), self.transform_of(l.child))
            else {
                continue;
            };
            let next = match l.mode {
                LinkMode::Follow => {
                    // Moved by hand since the last resolve: keep where it is
                    // now, relative to the parent.
                    let offset = if same_pose(&ct, &l.placed) {
                        l.offset
                    } else {
                        pt.inverse().compose(ct)
                    };
                    self.links[i].offset = offset;
                    pt.compose(offset)
                }
                LinkMode::Front { gap_mm } => {
                    let (pb, cb) = (self.is_body(l.parent), self.is_body(l.child));
                    let axis = if pb { Vec3::X } else { Vec3::Z };
                    let own = if cb { Vec3::X } else { Vec3::Z };
                    let q = glam::DQuat::from_rotation_arc(own, axis);
                    let offset =
                        Isometry3::new(axis * gap_mm, [q.x, q.y, q.z, q.w]).unwrap_or(l.offset);
                    self.links[i].offset = offset;
                    pt.compose(offset)
                }
                LinkMode::Face => match facing_frame(pt.pos - ct.pos, self.is_body(l.child)) {
                    Some(rot) => Isometry3::new(ct.pos, rot).unwrap_or(ct),
                    None => ct,
                },
            };
            if let Some(t) = self.object_transform_mut(l.child.0) {
                *t = next;
            }
            self.links[i].placed = next;
        }
    }

    /// Links and the sun together, in the order they depend on each other:
    /// links first (a heliostat may aim at a linked element), the sun, then
    /// links again (an element may follow a heliostat the sun just turned).
    pub fn resolve_constraints(&mut self) {
        self.resolve_links();
        self.resolve_solar();
        self.resolve_links();
    }

    pub(crate) fn validate_links(&self) -> Result<(), SceneError> {
        let mut children = std::collections::HashSet::new();
        for l in &self.links {
            for (what, id) in [("link child", l.child), ("link parent", l.parent)] {
                if self.transform_of(id).is_none() {
                    return Err(SceneError::DanglingRef { what, id: id.0 });
                }
            }
            let bad = |reason: &str| SceneError::OutOfRange {
                object: Some(l.child.0),
                field: "links".into(),
                reason: reason.into(),
            };
            if l.child == l.parent {
                return Err(bad("an element cannot be locked to itself"));
            }
            if !children.insert(l.child) {
                return Err(bad("an element can be locked to one other only"));
            }
            if self.is_heliostat(l.child) {
                return Err(bad("a heliostat mirror is turned by the sun, not a link"));
            }
            if !(l.offset.is_finite() && l.placed.is_finite()) {
                return Err(SceneError::NonFinite {
                    object: Some(l.child.0),
                    field: "links".into(),
                });
            }
        }
        // No loops: every chain of parents ends.
        for l in &self.links {
            let mut at = l.parent;
            for step in 0..=self.links.len() {
                match self.links.iter().find(|x| x.child == at) {
                    Some(up) => at = up.parent,
                    None => break,
                }
                if step == self.links.len() {
                    return Err(SceneError::OutOfRange {
                        object: Some(l.child.0),
                        field: "links".into(),
                        reason: "the links form a loop".into(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// The rotation that points an element along `dir`: a planar element's face
/// normal (local `+Z`) or a body's axis (local `+X`). The other axes are kept
/// level where they can be, the way a heliostat's width stays horizontal.
/// `None` for a zero direction, or one straight up or down for a planar
/// element, where the previous orientation is kept.
fn facing_frame(dir: Vec3, body: bool) -> Option<[f64; 4]> {
    let d = dir.normalize_or_zero();
    if d == Vec3::ZERO {
        return None;
    }
    let (x, y, z) = if body {
        // Axis along d; local Z horizontal where possible, so an in-plane
        // direction gives the unrotated in-plane frame.
        let z = d.cross(Vec3::Y);
        let z = if z.length() < 1e-9 {
            Vec3::Z
        } else {
            z.normalize()
        };
        let y = z.cross(d);
        (d, y, z)
    } else {
        let x = Vec3::Y.cross(d);
        if x.length() < 1e-9 {
            return None;
        }
        let x = x.normalize();
        (x, d.cross(x), d)
    };
    let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(x, y, z)).normalize();
    Some([q.x, q.y, q.z, q.w])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heliostat::{HelioTarget, LocalTime, Site};

    fn aperture_and_mirror() -> (Scene3, u64, u64) {
        let mut s = Scene3::new();
        let m = s
            .add_planar_mirror(Vec3::new(0.0, -50.0, 0.0), Vec3::new(0.0, 50.0, 0.0), 1.0)
            .unwrap();
        let a = s.add_aperture(Vec3::new(20.0, 0.0, 0.0), 30.0, 0.0);
        (s, m, a)
    }

    #[test]
    fn a_following_element_moves_and_turns_with_its_parent() {
        let (mut s, m, a) = aperture_and_mirror();
        s.link_object(a, m, LinkMode::Follow).unwrap();
        let rel = s
            .transform_of(ObjectId(m))
            .unwrap()
            .inverse()
            .compose(s.transform_of(ObjectId(a)).unwrap());
        // Move and turn the mirror; the aperture keeps its pose on it.
        s.set_object_position(m, Vec3::new(100.0, 40.0, -5.0));
        s.set_surface_tilt_z_deg(m, 30.0);
        s.resolve_links();
        let now = s
            .transform_of(ObjectId(m))
            .unwrap()
            .inverse()
            .compose(s.transform_of(ObjectId(a)).unwrap());
        assert!(same_pose(&rel, &now), "{rel:?} vs {now:?}");
        assert!(
            (s.transform_of(ObjectId(a)).unwrap().pos - Vec3::new(100.0, 40.0, -5.0)).length()
                > 1.0
        );
    }

    #[test]
    fn moving_a_follower_by_hand_sets_a_new_offset() {
        let (mut s, m, a) = aperture_and_mirror();
        s.link_object(a, m, LinkMode::Follow).unwrap();
        s.set_object_position(a, Vec3::new(50.0, 0.0, 0.0));
        s.resolve_links();
        assert!(
            (s.transform_of(ObjectId(a)).unwrap().pos - Vec3::new(50.0, 0.0, 0.0)).length() < 1e-9
        );
        s.set_object_position(m, Vec3::new(0.0, 10.0, 0.0));
        s.resolve_links();
        assert!(
            (s.transform_of(ObjectId(a)).unwrap().pos - Vec3::new(50.0, 10.0, 0.0)).length() < 1e-9
        );
    }

    #[test]
    fn a_facing_element_points_at_its_parent() {
        let (mut s, m, a) = aperture_and_mirror();
        s.set_object_position(m, Vec3::new(20.0, 300.0, 400.0));
        s.link_object(a, m, LinkMode::Face).unwrap();
        let t = s.transform_of(ObjectId(a)).unwrap();
        let want = (Vec3::new(20.0, 300.0, 400.0) - t.pos).normalize();
        assert!((t.rotate(Vec3::Z) - want).length() < 1e-12);
        assert!(
            (t.pos - Vec3::new(20.0, 0.0, 0.0)).length() < 1e-12,
            "it does not move"
        );
        // A body points its axis instead.
        let lens = s.add_singlet(
            Vec3::new(-100.0, 0.0, 0.0),
            crate::build3::SingletGeom::biconvex(),
            1.5,
            60.0,
        );
        s.link_object(lens.0, m, LinkMode::Face).unwrap();
        let t = s.transform_of(ObjectId(lens.0)).unwrap();
        let want = (Vec3::new(20.0, 300.0, 400.0) - t.pos).normalize();
        assert!((t.rotate(Vec3::X) - want).length() < 1e-12);
    }

    #[test]
    fn loops_self_links_and_heliostat_children_are_refused() {
        let (mut s, m, a) = aperture_and_mirror();
        assert!(s.link_object(a, a, LinkMode::Follow).is_err());
        s.link_object(a, m, LinkMode::Follow).unwrap();
        assert!(s.link_object(m, a, LinkMode::Follow).is_err(), "loop");
        assert_eq!(s.links.len(), 1);
        let site = Site {
            latitude_deg: 37.0,
            longitude_deg: -2.4,
            elevation_m: 500.0,
            pressure_mbar: 950.0,
            temperature_c: 20.0,
        };
        let time = LocalTime {
            year: 2026,
            month: 3,
            day: 20,
            hour: 12,
            minute: 0,
            second: 0.0,
            utc_offset_h: 1.0,
        };
        s.add_sun(site, time, 0.0);
        let h = s
            .add_heliostat(
                Vec3::new(0.0, -2000.0, 0.0),
                500.0,
                500.0,
                0.9,
                HelioTarget::Object(ObjectId(m)),
            )
            .unwrap();
        assert!(
            s.link_object(h.0, m, LinkMode::Face).is_err(),
            "the sun turns a heliostat"
        );
    }

    #[test]
    fn an_aperture_follows_a_tracking_heliostat_through_the_day() {
        let mut s = Scene3::new();
        let site = Site {
            latitude_deg: 37.0,
            longitude_deg: -2.4,
            elevation_m: 500.0,
            pressure_mbar: 950.0,
            temperature_c: 20.0,
        };
        let at = |h: u32| LocalTime {
            year: 2026,
            month: 3,
            day: 20,
            hour: h,
            minute: 0,
            second: 0.0,
            utc_offset_h: 1.0,
        };
        s.add_sun(site, at(10), 0.0);
        let target = s.add_aperture(Vec3::new(0.0, 5000.0, 0.0), 100.0, -90.0);
        let h = s
            .add_heliostat(
                Vec3::ZERO,
                1000.0,
                1000.0,
                0.9,
                HelioTarget::Object(ObjectId(target)),
            )
            .unwrap();
        s.resolve_constraints();
        // An aperture 50 mm in front of the mirror, parallel to it.
        let mt = s.transform_of(h).unwrap();
        let ap = s.add_aperture(mt.transform_point(Vec3::new(0.0, 0.0, 50.0)), 450.0, 0.0);
        if let Some(t) = s.object_transform_mut(ap) {
            *t = mt.compose(Isometry3::from_pos(Vec3::new(0.0, 0.0, 50.0)));
        }
        s.link_object(ap, h.0, LinkMode::Follow).unwrap();
        for hour in [8, 12, 16] {
            s.solar.as_mut().unwrap().time = at(hour);
            s.resolve_constraints();
            let (m, a) = (
                s.transform_of(h).unwrap(),
                s.transform_of(ObjectId(ap)).unwrap(),
            );
            let rel = m.inverse().compose(a);
            assert!(
                (rel.pos - Vec3::new(0.0, 0.0, 50.0)).length() < 1e-9,
                "{hour}h: {rel:?}"
            );
            assert!(
                (a.rotate(Vec3::Z) - m.rotate(Vec3::Z)).length() < 1e-12,
                "{hour}h: parallel"
            );
        }
    }

    #[test]
    fn an_element_in_front_sits_parallel_at_its_gap_as_the_parent_turns() {
        let (mut s, m, a) = aperture_and_mirror();
        s.link_object(a, m, LinkMode::Front { gap_mm: 25.0 })
            .unwrap();
        for tilt in [0.0, 30.0, -70.0] {
            s.set_surface_tilt_z_deg(m, tilt);
            s.set_object_position(a, Vec3::new(500.0, 500.0, 0.0)); // a stray drag
            s.resolve_links();
            let (mt, at) = (
                s.transform_of(ObjectId(m)).unwrap(),
                s.transform_of(ObjectId(a)).unwrap(),
            );
            let n = mt.rotate(Vec3::Z);
            assert!(
                (at.pos - (mt.pos + 25.0 * n)).length() < 1e-9,
                "{tilt}: on the axis"
            );
            assert!(
                (at.rotate(Vec3::Z) - n).length() < 1e-12,
                "{tilt}: parallel"
            );
        }
        // In front of a lens: on its optical axis, face normal along it.
        let lens = s.add_singlet(
            Vec3::new(-100.0, 20.0, 0.0),
            crate::build3::SingletGeom::biconvex(),
            1.5,
            60.0,
        );
        s.link_object(a, lens.0, LinkMode::Front { gap_mm: -30.0 })
            .unwrap();
        let at = s.transform_of(ObjectId(a)).unwrap();
        assert!((at.pos - Vec3::new(-130.0, 20.0, 0.0)).length() < 1e-9);
        assert!((at.rotate(Vec3::Z) - Vec3::X).length() < 1e-12);
    }

    #[test]
    fn links_round_trip_and_deleting_a_parent_drops_the_link() {
        let (mut s, m, a) = aperture_and_mirror();
        s.link_object(a, m, LinkMode::Face).unwrap();
        let back = Scene3::from_json(&s.to_json()).unwrap();
        assert_eq!(back.links, s.links);
        s.surfaces.retain(|x| x.common.id.0 != m);
        s.resolve_links();
        assert!(s.links.is_empty());
    }
}
