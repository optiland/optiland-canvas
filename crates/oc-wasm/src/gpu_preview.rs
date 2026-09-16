//! The WebGPU half of the interactive 3D preview (§4.1).
//!
//! It draws the picture `Canvas::traceProjected3DFast` and
//! `Canvas::traceWorld3DFast` draw, traced by `shader3.wgsl` instead of the CPU
//! f32 mirror. Every displayed number still comes from the f64 reference; this
//! path only draws.
//!
//! A trace resolves to a `Float32Array` in the CPU methods' layout, or to one of
//! two strings. `"busy"` means a trace is already in flight on this device.
//! `"truncated"` means some ray's branch tree does not fit on the GPU even at the
//! largest budget, so the caller should draw the CPU picture rather than a
//! thinner one. A device failure rejects the promise, and the tracer refuses
//! every later trace.

use std::cell::Cell;
use std::rc::Rc;

use js_sys::{Float32Array, Promise};
use oc_gpu::buffers3::Seg3Buf;
use oc_gpu::cpu3::SegOut3;
use oc_gpu::gpu3::{Gpu3Error, Gpu3InitError, Gpu3Tracer};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use crate::{pack_projected, pack_world, Canvas, FAST_DEPTH};

/// The starting per-ray segment budget. A ghost-free ray is a single path of at
/// most `FAST_DEPTH + 1` segments, so it fits with room to spare.
const MIN_SEGS: u32 = 64;
/// The shader stops a ray after 4096 branches, so no budget needs more.
const MAX_SEGS: u32 = 4096;
/// The largest segment buffer one trace allocates and reads back: 64 MB, half of
/// WebGPU's default storage-binding limit. Read-back time grows with it.
const MAX_SEG_BYTES: u64 = 64 << 20;

/// An interactive 3D tracer on the browser's WebGPU device.
#[wasm_bindgen]
pub struct GpuPreview3 {
    tracer: Rc<Gpu3Tracer>,
    budget: Rc<Cell<Budget>>,
}

/// The per-ray segment budget that last traced without truncation, keyed by the
/// ghost settings that decide how bushy a ray tree gets.
#[derive(Clone, Copy)]
struct Budget {
    segs: u32,
    ghosts: (bool, u32),
}

/// Create the WebGPU preview. Rejects when the browser has no usable device or
/// the device refuses the shader; the caller keeps the CPU mirror.
#[wasm_bindgen(js_name = createGpuPreview3)]
pub async fn create_gpu_preview3() -> Result<GpuPreview3, JsError> {
    match Gpu3Tracer::new_async().await {
        Ok(tracer) => Ok(GpuPreview3 {
            tracer: Rc::new(tracer),
            budget: Rc::new(Cell::new(Budget {
                segs: MIN_SEGS,
                ghosts: (false, 0),
            })),
        }),
        Err(Gpu3InitError::NoAdapter) => Err(JsError::new("no WebGPU adapter")),
        Err(Gpu3InitError::Device(msg)) => Err(JsError::new(&format!(
            "the WebGPU adapter refused the device: {msg}"
        ))),
        Err(Gpu3InitError::Pipeline(msg)) => Err(JsError::new(&format!(
            "WebGPU rejected the 3D preview shader: {msg}"
        ))),
    }
}

#[wasm_bindgen]
impl GpuPreview3 {
    /// The adapter's name, for the preview status line.
    #[wasm_bindgen(js_name = adapterName)]
    pub fn adapter_name(&self) -> String {
        self.tracer.adapter_info.name.clone()
    }

    /// The per-ray segment budget the next trace starts from.
    #[wasm_bindgen(js_name = segmentBudget)]
    pub fn segment_budget(&self) -> u32 {
        self.budget.get().segs
    }

    /// The GPU counterpart of `Canvas::traceProjected3DFast`: the same 11-float
    /// projected layout, on `canvas`'s current view plane.
    #[wasm_bindgen(js_name = traceProjected3D)]
    pub fn trace_projected_3d(&self, canvas: &Canvas) -> Promise {
        let plane = canvas.view_plane_copy();
        self.run(canvas, move |segs| pack_projected(segs, &plane))
    }

    /// The GPU counterpart of `Canvas::traceWorld3DFast`: the same 11-float
    /// world-frame layout.
    #[wasm_bindgen(js_name = traceWorld3D)]
    pub fn trace_world_3d(&self, canvas: &Canvas) -> Promise {
        self.run(canvas, pack_world)
    }
}

impl GpuPreview3 {
    /// Compile the scene now, so the result belongs to the scene revision the
    /// caller saw, then trace it on the device.
    fn run(
        &self,
        canvas: &Canvas,
        pack: impl Fn(&[Vec<SegOut3>]) -> Vec<f32> + 'static,
    ) -> Promise {
        let Some(job) = canvas.fast_scene() else {
            let empty: JsValue = Float32Array::new_with_length(0).into();
            return Promise::resolve(&empty);
        };
        let ghosts = (job.ghosts, job.max_ghost);
        let cap = budget_cap(job.rays.len());
        let tracer = Rc::clone(&self.tracer);
        let budget = Rc::clone(&self.budget);
        future_to_promise(async move {
            let known = budget.get();
            let start = if known.ghosts == ghosts {
                known.segs
            } else {
                MIN_SEGS
            };
            let mut segs = start.min(cap);
            loop {
                let trace = match tracer.trace_async(&job, FAST_DEPTH, segs).await {
                    Ok(trace) => trace,
                    Err(Gpu3Error::Busy) => return Ok(JsValue::from_str("busy")),
                    Err(e) => {
                        return Err(JsError::new(&format!("3D GPU preview failed: {e:?}")).into())
                    }
                };
                if trace.is_complete() {
                    budget.set(Budget { segs, ghosts });
                    return Ok(Float32Array::from(&pack(&trace.segments)[..]).into());
                }
                let wider = segs.saturating_mul(2).min(cap);
                if trace.stack_truncated > 0 || wider <= segs {
                    budget.set(Budget { segs, ghosts });
                    return Ok(JsValue::from_str("truncated"));
                }
                segs = wider;
            }
        })
    }
}

/// The largest per-ray budget whose segment buffer stays under
/// [`MAX_SEG_BYTES`].
fn budget_cap(n_rays: usize) -> u32 {
    let per_ray = std::mem::size_of::<Seg3Buf>() as u64 * n_rays.max(1) as u64;
    (MAX_SEG_BYTES / per_ray).clamp(1, MAX_SEGS as u64) as u32
}
