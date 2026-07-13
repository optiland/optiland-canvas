//! A small bounding-volume hierarchy over element AABBs (§4.2 step 1).
//!
//! The spec calls for a BVH over edges; an element-level BVH is a valid coarser
//! structure that is ample for M0 scene sizes and keeps leaves cheap. Traversal
//! yields candidate element indices whose box the ray crosses within `tmax`; the
//! tracer refines to the exact nearest edge hit.

use oc_core::Aabb;
use oc_core::Ray;

struct Node {
    aabb: Aabb,
    /// Child node index for internal nodes; `-1` for leaves.
    left: i32,
    right: i32,
    /// Leaf range into `order`.
    start: u32,
    count: u32,
}

/// BVH over a set of primitive AABBs, identified by their input index.
pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
}

impl Bvh {
    /// Build over the given per-primitive boxes.
    pub fn build(boxes: &[Aabb]) -> Self {
        let mut order: Vec<u32> = (0..boxes.len() as u32).collect();
        let mut nodes: Vec<Node> = Vec::new();
        if !boxes.is_empty() {
            build_recursive(boxes, &mut order, 0, boxes.len(), &mut nodes);
        }
        Bvh { nodes, order }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Invoke `f` with each primitive index whose box the ray crosses within
    /// `(tmin, tmax)`.
    pub fn candidates(&self, ray: &Ray, tmin: f64, tmax: f64, f: &mut impl FnMut(usize)) {
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
                // Depth guard: fall back to reporting all prims under here.
                // (Should not happen for balanced trees at M0 scale.)
                for k in node.start..node.start + node.count {
                    f(self.order[k as usize] as usize);
                }
            }
        }
    }
}

const LEAF_SIZE: usize = 4;

fn bounds_of(boxes: &[Aabb], order: &[u32], start: usize, end: usize) -> Aabb {
    let mut bb = Aabb::EMPTY;
    for &idx in &order[start..end] {
        bb = bb.union(&boxes[idx as usize]);
    }
    bb
}

fn build_recursive(
    boxes: &[Aabb],
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

    // Split along the longer centroid extent at the median.
    let extent = bb.max - bb.min;
    let axis = if extent.x >= extent.y { 0 } else { 1 };
    let key = |idx: u32| {
        let c = boxes[idx as usize].center();
        if axis == 0 {
            c.x
        } else {
            c.y
        }
    };
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
    use oc_core::Vec2;

    #[test]
    fn candidates_cover_a_hit_box() {
        // A row of unit boxes along x at 0,2,4,...,18.
        let boxes: Vec<Aabb> = (0..10)
            .map(|i| {
                let x = 2.0 * i as f64;
                Aabb {
                    min: Vec2::new(x - 0.5, -0.5),
                    max: Vec2::new(x + 0.5, 0.5),
                }
            })
            .collect();
        let bvh = Bvh::build(&boxes);
        // Ray along +x at y=0 should encounter every box.
        let ray = Ray::new(Vec2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));
        let mut hits = Vec::new();
        bvh.candidates(&ray, 1e-9, f64::INFINITY, &mut |i| hits.push(i));
        hits.sort_unstable();
        assert_eq!(hits, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn candidates_prune_off_axis() {
        let boxes: Vec<Aabb> = (0..10)
            .map(|i| {
                let x = 2.0 * i as f64;
                Aabb {
                    min: Vec2::new(x - 0.5, -0.5),
                    max: Vec2::new(x + 0.5, 0.5),
                }
            })
            .collect();
        let bvh = Bvh::build(&boxes);
        // Ray well above all boxes hits none.
        let ray = Ray::new(Vec2::new(-5.0, 100.0), Vec2::new(1.0, 0.0));
        let mut n = 0;
        bvh.candidates(&ray, 1e-9, f64::INFINITY, &mut |_| n += 1);
        assert_eq!(n, 0);
    }
}
