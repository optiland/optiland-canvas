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
}

impl Bvh3 {
    /// Build over the given per-primitive world boxes.
    pub fn build(boxes: &[Aabb3]) -> Self {
        let mut order: Vec<u32> = (0..boxes.len() as u32).collect();
        let mut nodes: Vec<Node> = Vec::new();
        if !boxes.is_empty() {
            build_recursive(boxes, &mut order, 0, boxes.len(), &mut nodes);
        }
        Bvh3 { nodes, order }
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
}

const LEAF_SIZE: usize = 4;

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
