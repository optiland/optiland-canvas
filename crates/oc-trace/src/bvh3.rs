//! A bounding-volume hierarchy over world-space [`Aabb3`]s (§9.2 step 1, §12.4).
//!
//! The 3D counterpart of [`crate::bvh`]. It indexes whatever the tracer wants a
//! nearest-hit over — bodies, surfaces, apertures, detectors — by their input
//! index; the boxes are the objects' conservative world AABBs, and traversal
//! yields the candidates whose box a ray crosses within `tmax`. The tracer then
//! refines each candidate to its exact hits.

use oc_core::vec3::{Aabb3, Ray3};

struct Node {
    aabb: Aabb3,
    /// Child node index for internal nodes; `-1` for leaves.
    left: i32,
    right: i32,
    /// Leaf range into `order`.
    start: u32,
    count: u32,
}

/// A BVH over a set of primitive AABBs, identified by their input index.
pub struct Bvh3 {
    nodes: Vec<Node>,
    order: Vec<u32>,
    /// Each primitive's own box, by input index, so a leaf can pass over a
    /// primitive the ray does not reach before the nearest hit found so far.
    boxes: Vec<Aabb3>,
}

impl Bvh3 {
    /// Build over the given per-primitive world boxes.
    pub fn build(boxes: &[Aabb3]) -> Self {
        let mut order: Vec<u32> = (0..boxes.len() as u32).collect();
        let mut nodes: Vec<Node> = Vec::new();
        if !boxes.is_empty() {
            build_recursive(boxes, &mut order, 0, boxes.len(), &mut nodes);
        }
        Bvh3 {
            nodes,
            order,
            boxes: boxes.to_vec(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Invoke `f` with each primitive index whose box the ray crosses within
    /// `(tmin, tmax)`.
    pub fn candidates(&self, ray: &Ray3, tmin: f64, tmax: f64, f: &mut impl FnMut(usize)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = [0usize; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let ni = stack[sp];
            let node = &self.nodes[ni];
            if !node.aabb.intersects_ray(ray, tmin, tmax) {
                continue;
            }
            if node.left < 0 {
                for k in node.start..node.start + node.count {
                    f(self.order[k as usize] as usize);
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.left as usize;
                sp += 1;
                stack[sp] = node.right as usize;
                sp += 1;
            } else {
                // Depth guard: report everything under here rather than overflow.
                for k in node.start..node.start + node.count {
                    f(self.order[k as usize] as usize);
                }
            }
        }
    }

    /// Invoke `f` with each primitive whose box the ray reaches before the
    /// nearest hit found so far, nearest boxes first. `f` is given the current
    /// limit and returns the new one: the distance to the nearest hit it has
    /// found, or the limit it was given.
    ///
    /// This is the nearest-hit walk. A box that the ray enters beyond the
    /// nearest hit cannot hold a nearer one, since every box encloses its
    /// primitive, so it is passed over, and the walk goes nearest first so the
    /// limit tightens early. In a coaxial lens every element's box straddles
    /// the axis, and a walk that kept its first limit asked every element for
    /// its hits at every step, which made a ray's cost grow with the square
    /// of the element count.
    ///
    /// A box is passed over only when the ray enters it clearly beyond the
    /// nearest hit: by more than a part in 1e9 of that distance. Two bodies
    /// cemented together share a face, their two copies of it are hit a
    /// rounding apart, and a box entry is itself a rounding; without the margin
    /// the walk could skip one twin, record the other, and move a ghost's
    /// path by a unit in the last place. With it the nearest hit is the one
    /// asking every box finds, to the bit.
    pub fn nearest(&self, ray: &Ray3, tmin: f64, tmax: f64, f: &mut impl FnMut(usize, f64) -> f64) {
        if self.nodes.is_empty() {
            return;
        }
        let mut limit = tmax;
        let Some(root) = ray_entry(&self.nodes[0].aabb, ray, tmin, limit) else {
            return;
        };
        let mut stack = [(0usize, 0.0f64); 64];
        let mut sp = 0usize;
        stack[sp] = (0, root);
        sp += 1;
        let slack = |limit: f64| limit + ROUNDING * limit.abs();
        while sp > 0 {
            sp -= 1;
            let (ni, entry) = stack[sp];
            if entry > slack(limit) {
                continue;
            }
            let node = &self.nodes[ni];
            if node.left < 0 || sp + 2 > stack.len() {
                // A leaf (or the depth guard): its primitives, in the order
                // the build left them.
                for k in node.start..node.start + node.count {
                    let idx = self.order[k as usize] as usize;
                    if ray_entry(&self.boxes[idx], ray, tmin, slack(limit)).is_some() {
                        limit = f(idx, limit).min(limit);
                    }
                }
                continue;
            }
            let (l, r) = (node.left as usize, node.right as usize);
            let el = ray_entry(&self.nodes[l].aabb, ray, tmin, slack(limit));
            let er = ray_entry(&self.nodes[r].aabb, ray, tmin, slack(limit));
            // Push the farther child first, so the nearer one is walked first.
            let mut push = |child: usize, e: f64| {
                stack[sp] = (child, e);
                sp += 1;
            };
            match (el, er) {
                (Some(a), Some(b)) if a <= b => {
                    push(r, b);
                    push(l, a);
                }
                (Some(a), Some(b)) => {
                    push(l, a);
                    push(r, b);
                }
                (Some(a), None) => push(l, a),
                (None, Some(b)) => push(r, b),
                (None, None) => {}
            }
        }
    }
}

/// Where the ray enters `bb` within `(tmin, tmax)`, or `None` when it does not
/// reach it there: the slab test of [`Aabb3::intersects_ray`], step for step,
/// returning its entry distance.
fn ray_entry(bb: &Aabb3, ray: &Ray3, tmin: f64, tmax: f64) -> Option<f64> {
    let (mut enter, mut exit) = (tmin, tmax);
    for axis in 0..3 {
        let inv = 1.0 / ray.dir[axis];
        let mut near = (bb.min[axis] - ray.origin[axis]) * inv;
        let mut far = (bb.max[axis] - ray.origin[axis]) * inv;
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        if near > enter {
            enter = near;
        }
        if far < exit {
            exit = far;
        }
        if enter > exit {
            return None;
        }
    }
    Some(enter)
}

const LEAF_SIZE: usize = 4;

/// How far past the nearest hit, as a fraction of its distance, a box still
/// counts as reached. See [`Bvh3::nearest`].
const ROUNDING: f64 = 1e-9;

fn bounds_of(boxes: &[Aabb3], order: &[u32], start: usize, end: usize) -> Aabb3 {
    let mut bb = Aabb3::EMPTY;
    for &idx in &order[start..end] {
        bb = bb.union(&boxes[idx as usize]);
    }
    bb
}

fn build_recursive(
    boxes: &[Aabb3],
    order: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> usize {
    let bb = bounds_of(boxes, order, start, end);
    let count = end - start;
    let node_index = nodes.len();
    nodes.push(Node {
        aabb: bb,
        left: -1,
        right: -1,
        start: start as u32,
        count: count as u32,
    });

    if count <= LEAF_SIZE {
        return node_index;
    }

    // Split along the longest centroid extent, at the median.
    let extent = bb.max - bb.min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let key = |idx: u32| boxes[idx as usize].center()[axis];
    order[start..end].sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap());
    let mid = start + count / 2;

    let left = build_recursive(boxes, order, start, mid, nodes);
    let right = build_recursive(boxes, order, mid, end, nodes);
    nodes[node_index].left = left as i32;
    nodes[node_index].right = right as i32;
    node_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::vec3::Vec3;

    fn unit_boxes_along_x() -> Vec<Aabb3> {
        (0..10)
            .map(|i| {
                let x = 2.0 * i as f64;
                Aabb3 {
                    min: Vec3::new(x - 0.5, -0.5, -0.5),
                    max: Vec3::new(x + 0.5, 0.5, 0.5),
                }
            })
            .collect()
    }

    #[test]
    fn candidates_cover_a_hit_row() {
        let bvh = Bvh3::build(&unit_boxes_along_x());
        let ray = Ray3::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap();
        let mut hits = Vec::new();
        bvh.candidates(&ray, 1e-9, f64::INFINITY, &mut |i| hits.push(i));
        hits.sort_unstable();
        assert_eq!(hits, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn candidates_prune_a_ray_that_misses() {
        let bvh = Bvh3::build(&unit_boxes_along_x());
        let ray = Ray3::new(Vec3::new(-5.0, 100.0, 0.0), Vec3::X).unwrap();
        let mut n = 0;
        bvh.candidates(&ray, 1e-9, f64::INFINITY, &mut |_| n += 1);
        assert_eq!(n, 0);
    }
}
