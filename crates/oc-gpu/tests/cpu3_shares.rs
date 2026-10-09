//! The f32 preview traced in shares of launch rays, as a pool of browser
//! workers traces a heavy frame, is the frame traced whole: the same segments,
//! bit for bit, in the same order.
#![cfg(feature = "gpu3")]

use oc_core::scene3::Scene3;
use oc_gpu::cpu3::{self, SegOut3};
use oc_gpu::scene3::{BuildOpts, GpuScene3};
use oc_trace::depth_for_scene3;

const GALLERY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/public/gallery");

fn bits(per_ray: &[Vec<SegOut3>]) -> Vec<[u64; 9]> {
    per_ray
        .iter()
        .flatten()
        .map(|s| {
            [
                s.a.x.to_bits(),
                s.a.y.to_bits(),
                s.a.z.to_bits(),
                s.b.x.to_bits(),
                s.b.y.to_bits(),
                s.b.z.to_bits(),
                s.flux.to_bits() as u64,
                s.lambda_um.to_bits() as u64,
                s.flags as u64,
            ]
        })
        .collect()
}

#[test]
fn shares_one_after_another_are_the_whole_frame() {
    for (id, ghosts) in [("cooke-triplet", true), ("double-gauss", false)] {
        let text = std::fs::read_to_string(format!("{GALLERY}/{id}.ocanvas")).unwrap();
        let scene = Scene3::from_json(&text).unwrap();
        let job = GpuScene3::build(
            &scene,
            BuildOpts {
                ghosts,
                flux_eps: 1e-4,
                max_ghost: 2,
            },
        );
        let depth = depth_for_scene3(&scene, ghosts, 2);
        let whole = bits(&cpu3::trace_segments(&job, depth));
        let n = job.rays.len();
        for shares in [1, 3, 7] {
            let mut joined = Vec::new();
            for i in 0..shares {
                let range = i * n / shares..(i + 1) * n / shares;
                joined.extend(bits(&cpu3::trace_segments_range(&job, depth, range)));
            }
            assert_eq!(joined, whole, "{id}, {shares} shares");
        }
    }
}
