//! The 3D models of the parts around the optics, as things that stop light.
//!
//! A scene holds optics only. The app knows what surrounds them (a cube's walls, a holder, a camera's body) from its mesh
//! files, and hands the triangles here: each mesh once, then where each one stands. They are not part of the scene, and no
//! document saves them. A ray that meets one ends there, booked as blocked, as at an aperture with no opening.
//!
//! The rules, for what the models and the optics' records do not say alike:
//! - a solid within [`SEAT_MM`] of the ray's start or of its next event does not count: a record's surface can sit a
//!   little inside its model's glass or mount;
//! - a ray straight from a source leaves the material it starts inside, and only that: past it, and everywhere else, the
//!   source's own holder and cube stop its light like any other solid;
//! - a solid marked [`YIELDS_TO_DETECTOR`] (a camera's body, around its sensor) does not stop a ray on its way to a detector.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use oc_core::vec3::{Aabb3, Isometry3, Ray3, Vec3};

use crate::bvh3::Bvh3;

/// How near the ends of a step a solid is not believed, mm.
pub const SEAT_MM: f64 = 1.5;
/// In an instance's flags: this solid lets through a ray bound for a detector.
pub const YIELDS_TO_DETECTOR: u32 = 1;
/// Floats per instance: mesh id, flags, position (3), quaternion `xyzw` (4).
pub const INSTANCE_FLOATS: usize = 9;
/// Further than any model reaches, mm.
const FAR_MM: f64 = 1e7;

/// One mesh in its own frame, mm.
pub struct SolidMesh {
    triangles: Vec<[Vec3; 3]>,
    bvh: Bvh3,
    bound: Aabb3,
}

impl SolidMesh {
    /// From flat positions, 9 floats a triangle.
    pub fn new(positions: &[f32]) -> Self {
        let triangles: Vec<[Vec3; 3]> = positions
            .chunks_exact(9)
            .map(|t| {
                let p = |i: usize| Vec3::new(t[i] as f64, t[i + 1] as f64, t[i + 2] as f64);
                [p(0), p(3), p(6)]
            })
            .collect();
        let boxes: Vec<Aabb3> = triangles
            .iter()
            .map(|[a, b, c]| {
                let mut bb = Aabb3::from_point(*a);
                bb.expand(*b);
                bb.expand(*c);
                bb.padded(1e-6)
            })
            .collect();
        let bound = boxes.iter().fold(Aabb3::EMPTY, |all, b| all.union(b));
        let bvh = Bvh3::build(&boxes);
        Self { triangles, bvh, bound }
    }

    /// The nearest triangle between `tmin` and `tmax` along a ray in this mesh's frame.
    fn nearest(&self, ray: &Ray3, tmin: f64, tmax: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        let mut limit = tmax;
        self.bvh.nearest(ray, tmin, tmax, &mut |i, _| {
            if let Some(t) = hit_triangle(&self.triangles[i], ray) {
                if t >= tmin && t < limit {
                    limit = t;
                    best = Some(t);
                }
            }
            limit
        });
        best
    }

    /// Whether a point is in this mesh's material: a ray from it crosses the surface an odd number of times. Three rays and
    /// the majority, in directions no model lines up with: a ray can meet an edge, and a mesh from another CAD tool need not
    /// be watertight.
    fn holds(&self, point: Vec3) -> bool {
        let inside = |dir: Vec3| {
            let Ok(ray) = Ray3::new(point, dir) else { return false };
            let mut crossings = 0_u32;
            self.bvh.nearest(&ray, 0.0, FAR_MM, &mut |i, _| {
                if hit_triangle(&self.triangles[i], &ray).is_some_and(|t| t >= 0.0) {
                    crossings += 1;
                }
                FAR_MM
            });
            crossings % 2 == 1
        };
        #[allow(clippy::approx_constant)] // fixed, unrelated directions; 0.7071 is not 1/√2 here
        let votes = [Vec3::new(0.5377, 0.2813, 0.7948), Vec3::new(-0.3119, 0.8347, 0.4537), Vec3::new(0.7071, -0.6113, -0.3559)]
            .into_iter()
            .filter(|d| inside(*d))
            .count();
        votes >= 2
    }
}

/// Moeller-Trumbore, both faces.
fn hit_triangle([a, b, c]: &[Vec3; 3], ray: &Ray3) -> Option<f64> {
    let (e1, e2) = (*b - *a, *c - *a);
    let p = ray.dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = ray.origin - *a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = ray.dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(e2.dot(q) * inv)
}

struct Instance {
    mesh: Arc<SolidMesh>,
    pose: Isometry3,
    flags: u32,
    world: Aabb3,
}

/// Every solid where it stands.
pub struct Solids {
    instances: Vec<Instance>,
    bvh: Bvh3,
}

impl Solids {
    /// What a step from a ray's origin meets first, before `ahead` (the distance of its next event, if any). `from_source`:
    /// the ray has just left its source. `to_detector`: its next event is a detector.
    pub fn first_hit(&self, ray: &Ray3, ahead: Option<f64>, bounds: f64, from_source: bool, to_detector: bool) -> Option<f64> {
        let tmax = ahead.map_or(bounds, |t| t - SEAT_MM);
        if tmax <= SEAT_MM {
            return None;
        }
        let mut best: Option<f64> = None;
        let mut limit = tmax;
        self.bvh.nearest(ray, SEAT_MM, tmax, &mut |i, _| {
            let solid = &self.instances[i];
            let yields = to_detector && solid.flags & YIELDS_TO_DETECTOR != 0;
            if !yields {
                if let Ok(local) = Ray3::new(solid.pose.inverse_transform_point(ray.origin), solid.pose.inv_rotate(ray.dir)) {
                    // Starting in this solid's material, a ray from a source first has to get out of it.
                    let out = if from_source && contains(&solid.world, ray.origin) && solid.mesh.holds(local.origin) {
                        solid.mesh.nearest(&local, 0.0, FAR_MM).map_or(FAR_MM, |t| t + 1e-6)
                    } else {
                        0.0
                    };
                    if let Some(t) = solid.mesh.nearest(&local, out.max(SEAT_MM), limit) {
                        limit = t;
                        best = Some(t);
                    }
                }
            }
            limit
        });
        best
    }
}

fn contains(b: &Aabb3, p: Vec3) -> bool {
    p.x >= b.min.x && p.x <= b.max.x && p.y >= b.min.y && p.y <= b.max.y && p.z >= b.min.z && p.z <= b.max.z
}

/// The mesh's box where it stands.
fn world_box(bound: &Aabb3, pose: &Isometry3) -> Aabb3 {
    let mut out = Aabb3::EMPTY;
    for i in 0..8 {
        let p = Vec3::new(
            if i & 1 == 0 { bound.min.x } else { bound.max.x },
            if i & 2 == 0 { bound.min.y } else { bound.max.y },
            if i & 4 == 0 { bound.min.z } else { bound.max.z },
        );
        out.expand(pose.transform_point(p));
    }
    out.padded(1e-6)
}

static MESHES: RwLock<Option<HashMap<u32, Arc<SolidMesh>>>> = RwLock::new(None);
static CURRENT: RwLock<Option<Arc<Solids>>> = RwLock::new(None);

/// Keep a mesh under an id. A mesh does not change: sent once per session.
pub fn define_mesh(id: u32, positions: &[f32]) {
    let mut meshes = MESHES.write().unwrap();
    meshes.get_or_insert_with(HashMap::new).insert(id, Arc::new(SolidMesh::new(positions)));
}

/// Where the solids stand now, [`INSTANCE_FLOATS`] each; the next trace meets them. Returns false, and changes nothing, on a
/// mesh not defined or a pose that is not one. Empty: no solids.
pub fn set_instances(flat: &[f64]) -> bool {
    if flat.len() % INSTANCE_FLOATS != 0 {
        return false;
    }
    let meshes = MESHES.read().unwrap();
    let mut instances = Vec::with_capacity(flat.len() / INSTANCE_FLOATS);
    for s in flat.chunks_exact(INSTANCE_FLOATS) {
        let Some(mesh) = meshes.as_ref().and_then(|m| m.get(&(s[0] as u32))) else {
            return false;
        };
        let Ok(pose) = Isometry3::new(Vec3::new(s[2], s[3], s[4]), [s[5], s[6], s[7], s[8]]) else {
            return false;
        };
        instances.push(Instance { world: world_box(&mesh.bound, &pose), mesh: Arc::clone(mesh), pose, flags: s[1] as u32 });
    }
    let next = (!instances.is_empty()).then(|| {
        let boxes: Vec<Aabb3> = instances.iter().map(|i| i.world).collect();
        Arc::new(Solids { bvh: Bvh3::build(&boxes), instances })
    });
    *CURRENT.write().unwrap() = next;
    true
}

/// The solids a trace about to start meets.
pub fn current() -> Option<Arc<Solids>> {
    CURRENT.read().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles across y = 0, 40 mm square, about the origin.
    fn wall() -> Vec<f32> {
        vec![
            -20.0, 0.0, -20.0, 20.0, 0.0, -20.0, 20.0, 0.0, 20.0, //
            -20.0, 0.0, -20.0, 20.0, 0.0, 20.0, -20.0, 0.0, 20.0,
        ]
    }

    fn solids(flags: f64, at_y: f64) -> Solids {
        let mesh = Arc::new(SolidMesh::new(&wall()));
        let pose = Isometry3::new(Vec3::new(0.0, at_y, 0.0), [0.0, 0.0, 0.0, 1.0]).unwrap();
        let world = world_box(&mesh.bound, &pose);
        let instances = vec![Instance { mesh, pose, flags: flags as u32, world }];
        let boxes: Vec<Aabb3> = instances.iter().map(|i| i.world).collect();
        Solids { bvh: Bvh3::build(&boxes), instances }
    }

    fn along_y(y: f64) -> Ray3 {
        Ray3::new(Vec3::new(0.0, y, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap()
    }

    #[test]
    fn a_ray_stops_at_the_wall() {
        let t = solids(0.0, 0.0).first_hit(&along_y(-100.0), None, 1e6, false, false).unwrap();
        assert!((t - 100.0).abs() < 1e-9);
    }

    #[test]
    fn a_ray_past_it_goes_on() {
        let ray = Ray3::new(Vec3::new(30.0, -100.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap();
        assert!(solids(0.0, 0.0).first_hit(&ray, None, 1e6, false, false).is_none());
    }

    #[test]
    fn a_surface_in_its_mount_is_reached() {
        // The next event 1 mm behind the wall: the wall is within the seat, and does not count.
        assert!(solids(0.0, 0.0).first_hit(&along_y(-100.0), Some(101.0), 1e6, false, false).is_none());
        // The next event well behind it: it does.
        assert!(solids(0.0, 0.0).first_hit(&along_y(-100.0), Some(120.0), 1e6, false, false).is_some());
    }

    #[test]
    fn nothing_right_at_the_start() {
        assert!(solids(0.0, 0.0).first_hit(&along_y(-1.0), None, 1e6, false, false).is_none());
    }

    /// The twelve triangles of a closed block, `half` mm from the origin along each axis.
    fn block(half: f32) -> Vec<f32> {
        let p = |x: f32, y: f32, z: f32| [x * half, y * half, z * half];
        let quad = |a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]| [a, b, c, a, c, d].concat();
        [
            quad(p(-1., -1., -1.), p(1., -1., -1.), p(1., 1., -1.), p(-1., 1., -1.)),
            quad(p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.)),
            quad(p(-1., -1., -1.), p(1., -1., -1.), p(1., -1., 1.), p(-1., -1., 1.)),
            quad(p(-1., 1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(-1., 1., 1.)),
            quad(p(-1., -1., -1.), p(-1., 1., -1.), p(-1., 1., 1.), p(-1., -1., 1.)),
            quad(p(1., -1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(1., -1., 1.)),
        ]
        .concat()
    }

    fn solid_of(positions: Vec<f32>) -> Solids {
        let mesh = Arc::new(SolidMesh::new(&positions));
        let pose = Isometry3::new(Vec3::new(0.0, 0.0, 0.0), [0.0, 0.0, 0.0, 1.0]).unwrap();
        let world = world_box(&mesh.bound, &pose);
        let instances = vec![Instance { mesh, pose, flags: 0, world }];
        let boxes: Vec<Aabb3> = instances.iter().map(|i| i.world).collect();
        Solids { bvh: Bvh3::build(&boxes), instances }
    }

    /// A cube's walls: a block with a block's room inside it.
    fn walls() -> Solids {
        solid_of([block(20.0), block(15.0)].concat())
    }

    #[test]
    fn a_source_leaves_the_material_it_starts_in() {
        let ray = Ray3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap();
        assert!(solid_of(block(20.0)).first_hit(&ray, None, 1e6, true, false).is_none());
        // Coming from anywhere else, it stops.
        assert!(solid_of(block(20.0)).first_hit(&ray, None, 1e6, false, false).is_some());
    }

    #[test]
    fn a_source_in_a_room_stops_at_its_walls() {
        // The room is inside the part's box, and in no material: its own cube stops its light like any other.
        let ray = Ray3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap();
        let t = walls().first_hit(&ray, None, 1e6, true, false).unwrap();
        assert!((t - 15.0).abs() < 1e-9);
    }

    #[test]
    fn past_the_material_it_starts_in_a_source_stops_again() {
        // In the wall at y = 17, going across the room: out at 15, then the far wall at -15.
        let ray = Ray3::new(Vec3::new(0.0, 17.0, 0.0), Vec3::new(0.0, -1.0, 0.0)).unwrap();
        let t = walls().first_hit(&ray, None, 1e6, true, false).unwrap();
        assert!((t - 32.0).abs() < 1e-9);
    }

    #[test]
    fn a_camera_body_lets_the_sensor_see() {
        let body = solids(YIELDS_TO_DETECTOR as f64, 0.0);
        assert!(body.first_hit(&along_y(-100.0), Some(120.0), 1e6, false, true).is_none());
        assert!(body.first_hit(&along_y(-100.0), Some(120.0), 1e6, false, false).is_some());
    }
}
