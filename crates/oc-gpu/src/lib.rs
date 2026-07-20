//! # oc-gpu
//!
//! The **interactive** tracer (spec §4.1): f32, shallow, approximate — for
//! *looking at*, not for numbers. Two interchangeable backends over one shared
//! scene representation and one shared algorithm:
//!
//! - [`gpu::GpuTracer`] — wgpu compute (`shader.wgsl`), the real interactive path.
//! - [`cpu::trace`] — an f32 CPU mirror; the no-WebGPU fallback and the oracle
//!   the GPU path is validated against.
//!
//! M2 makes it spectral (per-ray `n(λ)`, dispersion in the tracer), splits
//! ghosts (the ghost layer), and flags TIR surfaces. Both backends emit the
//! same `n_rays × max_segs` segment layout and are cross-validated against the
//! f64 reference tracer (`oc-trace`) — the project's top safeguard (§10.1).

#[cfg(feature = "gpu3")]
pub mod buffers3;
pub mod color;
pub mod cpu;
#[cfg(feature = "gpu3")]
pub mod cpu3;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(all(feature = "gpu", feature = "gpu3"))]
pub mod gpu3;
pub mod scene;
#[cfg(feature = "gpu3")]
pub mod scene3;

pub use scene::{BuildOpts, GpuScene, GpuSeg, SEG_GHOST, SEG_TIR, SEG_VALID};

/// A world-space segment with remaining flux, wavelength, and flags.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub a: glam::Vec2,
    pub b: glam::Vec2,
    pub flux: f32,
    pub lambda_um: f32,
    pub flags: u32,
}

impl Segment {
    pub fn is_ghost(&self) -> bool {
        self.flags & SEG_GHOST != 0
    }
    pub fn is_tir(&self) -> bool {
        self.flags & SEG_TIR != 0
    }
}

/// Decode a raw `n_rays × max_segs` buffer into per-ray valid segment lists.
pub fn segments_per_ray(raw: &[GpuSeg], n_rays: usize, max_segs: usize) -> Vec<Vec<Segment>> {
    let mut out = Vec::with_capacity(n_rays);
    for r in 0..n_rays {
        let mut v = Vec::new();
        for i in 0..max_segs {
            let s = raw[r * max_segs + i];
            if s.flags & SEG_VALID != 0 {
                v.push(Segment {
                    a: glam::Vec2::new(s.ax, s.ay),
                    b: glam::Vec2::new(s.bx, s.by),
                    flux: s.flux,
                    lambda_um: s.lambda,
                    flags: s.flags,
                });
            }
        }
        out.push(v);
    }
    out
}

/// Convenience: trace on the CPU f32 backend and decode to per-ray segments.
pub fn trace_cpu(scene: &GpuScene, max_depth: u32) -> Vec<Vec<Segment>> {
    let raw = cpu::trace(scene, max_depth);
    segments_per_ray(raw.as_slice(), scene.rays.len(), scene.max_segs as usize)
}
