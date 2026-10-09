//! wgpu compute backend for the full-3D preview (PR 3D-08).
//!
//! The 3D counterpart of [`crate::gpu`]: it uploads a [`Gpu3Buffers`], dispatches
//! `shader3.wgsl`, and reads back the segment buffer, decoding it to the same
//! world-millimetre [`SegOut3`] the CPU oracle ([`crate::cpu3`]) returns — so the
//! two are compared directly. Buffers are persistent and grow-only, as in the 2D
//! path: after the first frame a steady scene is pure `write_buffer` + dispatch +
//! read-back.
//!
//! The shader is validated against the CPU f32 tracer, never against f64: that
//! isolates a WGSL bug from an f32-precision gap (`gpu3_vs_cpu3`, §15.5).
//!
//! One code path serves native callers and the browser.
//! [`Gpu3Tracer::trace_async`] submits the work and awaits the read-back.
//! Natively, `device.poll(Wait)` finishes the work before the await and
//! [`Gpu3Tracer::trace`] blocks on it, so the parity and performance suites run
//! the code a browser runs. In a browser the poll returns at once and the event
//! loop resolves the mapping.
//!
//! The shader has two limits the CPU mirror lacks: a 64-deep branch stack and a
//! per-ray segment budget. It marks a ray that outgrew either, and
//! [`Gpu3Trace`] counts those rays, so a caller can widen the budget or draw the
//! CPU picture instead of one with branches missing.

use crate::buffers3::{
    Aperture3Buf, Body3Buf, Detector3Buf, Face3Buf, Gpu3Buffers, Params3, Ray3Buf, Seg3Buf,
    Surface3Buf, Vert3Buf,
};
use crate::cpu3::{SegOut3, SEG_GHOST, SEG_TIR};
use crate::scene3::GpuScene3;
use bytemuck::cast_slice;
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

const SEG_VALID: u32 = 1;
/// Set by the shader on a ray's first segment when the ray used its whole
/// segment budget with branches still waiting.
const SEG_TRUNC_BUDGET: u32 = 8;
/// Set by the shader on a ray's first segment when a branch was dropped because
/// the 64-deep stack was full.
const SEG_TRUNC_STACK: u32 = 16;

/// Why a tracer could not be created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gpu3InitError {
    /// No adapter matched.
    NoAdapter,
    /// The adapter refused the device, with the backend's message.
    Device(String),
    /// The device rejected the shader or the pipeline, with the backend's message.
    Pipeline(String),
}

/// Why a trace returned no picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gpu3Error {
    /// A trace on this tracer is still in flight. The two would share one
    /// staging buffer.
    Busy,
    /// The segment buffer could not be mapped for read-back.
    Readback,
    /// The device reported an error (a lost device, a validation failure). The
    /// tracer refuses every later trace.
    Device(String),
}

/// One GPU trace: per-ray segments, plus the rays the shader cut short.
#[derive(Debug, Clone)]
pub struct Gpu3Trace {
    /// Per-ray world-millimetre segments, the shape of
    /// [`crate::cpu3::trace_segments`].
    pub segments: Vec<Vec<SegOut3>>,
    /// Rays that used their whole segment budget with branches left. A larger
    /// budget fixes these.
    pub budget_truncated: usize,
    /// Rays that lost a branch to the full stack. No budget fixes these.
    pub stack_truncated: usize,
}

impl Gpu3Trace {
    /// Every ray's branch tree fit, so the picture is the CPU mirror's.
    pub fn is_complete(&self) -> bool {
        self.budget_truncated == 0 && self.stack_truncated == 0
    }
}

/// A ready-to-use compute device and pipeline for the 3D preview.
pub struct Gpu3Tracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buffers: RefCell<Option<Buffers>>,
    /// A trace is between submit and read-back.
    busy: Cell<bool>,
    /// The first device error reported in a browser, or a failed read-back.
    device_error: Arc<Mutex<Option<String>>>,
    pub adapter_info: wgpu::AdapterInfo,
}

/// Cached GPU buffers. `cap_*` are current allocated element counts; a buffer is
/// reallocated only when a scene needs more than it holds.
struct Buffers {
    params: wgpu::Buffer,
    bodies: wgpu::Buffer,
    surfaces: wgpu::Buffer,
    apertures: wgpu::Buffer,
    detectors: wgpu::Buffer,
    rays: wgpu::Buffer,
    verts: wgpu::Buffer,
    faces: wgpu::Buffer,
    segs: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cap_bodies: usize,
    cap_surfaces: usize,
    cap_apertures: usize,
    cap_detectors: usize,
    cap_rays: usize,
    cap_verts: usize,
    cap_faces: usize,
    cap_segs: usize,
    cap_stage_bytes: usize,
}

impl Gpu3Tracer {
    /// Create a tracer, preferring a real high-performance adapter and falling
    /// back to the software adapter. Returns `None` if neither is available.
    ///
    /// Panics when a device accepts the adapter but rejects the shader, so a
    /// broken backend fails a test loudly instead of skipping it. Set
    /// `WGPU_BACKEND` (for example `vulkan`) to steer around such a backend.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Option<Self> {
        init_or_panic(pollster::block_on(Self::new_async()))
    }

    /// Build a tracer on a specific adapter (per-device benchmarking / gates).
    /// Same failure rules as [`Gpu3Tracer::new`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Option<Self> {
        init_or_panic(pollster::block_on(Self::from_adapter_async(adapter)))
    }

    /// Create a tracer without blocking. In a browser only WebGPU is tried.
    pub async fn new_async() -> Result<Self, Gpu3InitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backends(),
            ..Default::default()
        });
        let preferred = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await;
        let adapter = match preferred {
            Some(adapter) => adapter,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                })
                .await
                .ok_or(Gpu3InitError::NoAdapter)?,
        };
        Self::from_adapter_async(&adapter).await
    }

    /// Build a tracer on a specific adapter without blocking.
    pub async fn from_adapter_async(adapter: &wgpu::Adapter) -> Result<Self, Gpu3InitError> {
        let adapter_info = adapter.get_info();
        // The 3D path binds eight storage buffers (the object arrays plus the two
        // arenas and the segment output) — the WebGPU default ceiling, but above
        // the downlevel default of four, so lift just that one limit.
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("oc-gpu3 device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| Gpu3InitError::Device(e.to_string()))?;

        let device_error = Arc::new(Mutex::new(None));
        // wgpu's default handler panics. In a browser a panic poisons the whole
        // wasm module, the user's scene with it, so record the error and let the
        // next trace report it. Native callers keep the panic.
        #[cfg(target_arch = "wasm32")]
        {
            let sink = Arc::clone(&device_error);
            device.on_uncaptured_error(Box::new(move |e| {
                record(&sink, &e.to_string());
            }));
        }

        // Shader and pipeline errors are caught on every platform and returned,
        // so the caller decides what they mean: `new` panics, the browser keeps
        // the CPU mirror.
        device.push_error_scope(wgpu::ErrorFilter::Internal);
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace3"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader3.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace3 bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trace3 pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace3 pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "trace",
            compilation_options: Default::default(),
            cache: None,
        });

        let validation = device.pop_error_scope().await;
        let internal = device.pop_error_scope().await;
        if let Some(e) = validation.or(internal) {
            return Err(Gpu3InitError::Pipeline(e.to_string()));
        }

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            buffers: RefCell::new(None),
            busy: Cell::new(false),
            device_error,
            adapter_info,
        })
    }

    /// Trace `scene` for up to `max_depth` interactions, with a per-ray segment
    /// budget of `max_segs`. Returns per-ray world-millimetre segment lists (the
    /// same shape as [`crate::cpu3::trace_segments`]). Blocks.
    ///
    /// It does not report truncation; use [`Gpu3Tracer::trace_checked`] when a
    /// ray tree may outgrow `max_segs`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn trace(&self, scene: &GpuScene3, max_depth: u32, max_segs: u32) -> Vec<Vec<SegOut3>> {
        self.trace_checked(scene, max_depth, max_segs).segments
    }

    /// [`Gpu3Tracer::trace`] with the truncation counts. Blocks; panics if the
    /// device fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn trace_checked(&self, scene: &GpuScene3, max_depth: u32, max_segs: u32) -> Gpu3Trace {
        match pollster::block_on(self.trace_async(scene, max_depth, max_segs)) {
            Ok(trace) => trace,
            Err(e) => panic!("3D GPU trace failed: {e:?}"),
        }
    }

    /// Trace `scene` and await the read-back.
    ///
    /// Drive the future to completion. Dropping it between submit and read-back
    /// leaves the staging buffer mapped, and wgpu refuses to map it again.
    pub async fn trace_async(
        &self,
        scene: &GpuScene3,
        max_depth: u32,
        max_segs: u32,
    ) -> Result<Gpu3Trace, Gpu3Error> {
        self.trace_async_with(scene, max_depth, max_segs, |raw| raw.decode())
            .await
    }

    /// [`Gpu3Tracer::trace_async`], handing `read` the read-back as it lies
    /// in the staging buffer. A caller that packs the segments into its own
    /// layout reads them once from there, instead of from per-ray lists
    /// built first: for a ghost-lit lens that is millions of segments less
    /// to allocate and copy on the thread that waits.
    pub async fn trace_async_with<T>(
        &self,
        scene: &GpuScene3,
        max_depth: u32,
        max_segs: u32,
        read: impl FnOnce(RawTrace<'_>) -> T,
    ) -> Result<T, Gpu3Error> {
        self.check_device()?;
        if self.busy.replace(true) {
            return Err(Gpu3Error::Busy);
        }
        let _busy = BusyGuard(&self.busy);

        let bufs = Gpu3Buffers::from_scene(scene, max_depth, max_segs);
        let n_rays = bufs.rays.len();
        let n_segs = n_rays * max_segs as usize;
        if n_segs == 0 {
            return Ok(read(RawTrace {
                raw: &[],
                scene,
                n_rays,
                max_segs: 0,
            }));
        }
        let seg_bytes = (n_segs * std::mem::size_of::<Seg3Buf>()) as wgpu::BufferAddress;
        let signal = MapSignal::default();

        {
            // wgpu forbids zero-sized buffers, so clamp every capacity to >= 1.
            let need_bodies = bufs.bodies.len().max(1);
            let need_surfaces = bufs.surfaces.len().max(1);
            let need_apertures = bufs.apertures.len().max(1);
            let need_detectors = bufs.detectors.len().max(1);
            let need_rays = bufs.rays.len().max(1);
            let need_verts = bufs.verts.len().max(1);
            let need_faces = bufs.faces.len().max(1);

            let mut slot = self.buffers.borrow_mut();
            if slot.is_none() {
                let params = uniform_buf(&self.device, std::mem::size_of::<Params3>());
                let bodies = storage_buf(&self.device, "bodies3", need_bodies * S_BODY);
                let surfaces = storage_buf(&self.device, "surfaces3", need_surfaces * S_SURF);
                let apertures = storage_buf(&self.device, "apertures3", need_apertures * S_APER);
                let detectors = storage_buf(&self.device, "detectors3", need_detectors * S_DET);
                let rays = storage_buf(&self.device, "rays3", need_rays * S_RAY);
                let verts = storage_buf(&self.device, "verts3", need_verts * S_VERT);
                let faces = storage_buf(&self.device, "faces3", need_faces * S_FACE);
                let segs = seg_buf(&self.device, seg_bytes);
                let staging = staging_buf(&self.device, seg_bytes);
                let bind_group = self.make_bind_group(
                    &params, &bodies, &surfaces, &apertures, &detectors, &rays, &verts, &faces,
                    &segs,
                );
                *slot = Some(Buffers {
                    params,
                    bodies,
                    surfaces,
                    apertures,
                    detectors,
                    rays,
                    verts,
                    faces,
                    segs,
                    staging,
                    bind_group,
                    cap_bodies: need_bodies,
                    cap_surfaces: need_surfaces,
                    cap_apertures: need_apertures,
                    cap_detectors: need_detectors,
                    cap_rays: need_rays,
                    cap_verts: need_verts,
                    cap_faces: need_faces,
                    cap_segs: n_segs,
                    cap_stage_bytes: seg_bytes as usize,
                });
            }
            let b = slot.as_mut().unwrap();

            let mut rebuild = false;
            macro_rules! grow {
                ($field:ident, $cap:ident, $need:expr, $label:literal, $stride:expr) => {
                    if $need > b.$cap {
                        b.$field = storage_buf(&self.device, $label, $need * $stride);
                        b.$cap = $need;
                        rebuild = true;
                    }
                };
            }
            grow!(bodies, cap_bodies, need_bodies, "bodies3", S_BODY);
            grow!(surfaces, cap_surfaces, need_surfaces, "surfaces3", S_SURF);
            grow!(
                apertures,
                cap_apertures,
                need_apertures,
                "apertures3",
                S_APER
            );
            grow!(
                detectors,
                cap_detectors,
                need_detectors,
                "detectors3",
                S_DET
            );
            grow!(rays, cap_rays, need_rays, "rays3", S_RAY);
            grow!(verts, cap_verts, need_verts, "verts3", S_VERT);
            grow!(faces, cap_faces, need_faces, "faces3", S_FACE);
            if n_segs > b.cap_segs {
                b.segs = seg_buf(&self.device, seg_bytes);
                b.cap_segs = n_segs;
                rebuild = true;
            }
            if seg_bytes as usize > b.cap_stage_bytes {
                b.staging = staging_buf(&self.device, seg_bytes);
                b.cap_stage_bytes = seg_bytes as usize;
            }
            if rebuild {
                b.bind_group = self.make_bind_group(
                    &b.params,
                    &b.bodies,
                    &b.surfaces,
                    &b.apertures,
                    &b.detectors,
                    &b.rays,
                    &b.verts,
                    &b.faces,
                    &b.segs,
                );
            }

            self.queue
                .write_buffer(&b.params, 0, cast_slice(&[bufs.params]));
            write_if(&self.queue, &b.bodies, &bufs.bodies);
            write_if(&self.queue, &b.surfaces, &bufs.surfaces);
            write_if(&self.queue, &b.apertures, &bufs.apertures);
            write_if(&self.queue, &b.detectors, &bufs.detectors);
            write_if(&self.queue, &b.rays, &bufs.rays);
            write_if(&self.queue, &b.verts, &bufs.verts);
            write_if(&self.queue, &b.faces, &bufs.faces);

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.clear_buffer(&b.segs, 0, Some(seg_bytes));
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("trace3 pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &b.bind_group, &[]);
                pass.dispatch_workgroups((n_rays as u32).div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&b.segs, 0, &b.staging, 0, seg_bytes);
            self.queue.submit(Some(encoder.finish()));

            let callback = signal.clone();
            b.staging
                .slice(0..seg_bytes)
                .map_async(wgpu::MapMode::Read, move |r| callback.complete(r));
        }

        // Natively this finishes the submission and runs the map callback. In a
        // browser it returns at once, and the event loop resolves the mapping.
        self.device.poll(wgpu::Maintain::Wait);
        let mapped = signal.wait().await;

        let slot = self.buffers.borrow();
        let b = slot
            .as_ref()
            .expect("buffers exist once a trace was submitted");
        if mapped.is_err() {
            // The staging buffer's map state is unknown now; refuse later traces
            // rather than risk mapping it twice.
            record(&self.device_error, "segment read-back could not be mapped");
            return Err(Gpu3Error::Readback);
        }
        let data = b.staging.slice(0..seg_bytes).get_mapped_range();
        let raw: &[Seg3Buf] = &cast_slice::<u8, Seg3Buf>(&data)[..n_segs];
        let out = read(RawTrace {
            raw,
            scene,
            n_rays,
            max_segs: max_segs as usize,
        });
        drop(data);
        b.staging.unmap();
        self.check_device()?;
        Ok(out)
    }

    fn check_device(&self) -> Result<(), Gpu3Error> {
        let slot = self
            .device_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match slot.as_ref() {
            Some(msg) => Err(Gpu3Error::Device(msg.clone())),
            None => Ok(()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_bind_group(
        &self,
        params: &wgpu::Buffer,
        bodies: &wgpu::Buffer,
        surfaces: &wgpu::Buffer,
        apertures: &wgpu::Buffer,
        detectors: &wgpu::Buffer,
        rays: &wgpu::Buffer,
        verts: &wgpu::Buffer,
        faces: &wgpu::Buffer,
        segs: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace3 bg"),
            layout: &self.layout,
            entries: &[
                bind(0, params),
                bind(1, bodies),
                bind(2, surfaces),
                bind(3, apertures),
                bind(4, detectors),
                bind(5, rays),
                bind(6, verts),
                bind(7, faces),
                bind(8, segs),
            ],
        })
    }
}

/// Which backends to try. Natively `WGPU_BACKEND` narrows the choice; a browser
/// offers only WebGPU.
#[cfg(not(target_arch = "wasm32"))]
fn backends() -> wgpu::Backends {
    wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all())
}

#[cfg(target_arch = "wasm32")]
fn backends() -> wgpu::Backends {
    wgpu::Backends::BROWSER_WEBGPU
}

#[cfg(not(target_arch = "wasm32"))]
fn init_or_panic(result: Result<Gpu3Tracer, Gpu3InitError>) -> Option<Gpu3Tracer> {
    match result {
        Ok(tracer) => Some(tracer),
        Err(Gpu3InitError::NoAdapter) => None,
        Err(Gpu3InitError::Device(msg)) => {
            eprintln!("3D GPU: the adapter refused the device: {msg}");
            None
        }
        Err(Gpu3InitError::Pipeline(msg)) => panic!("3D GPU pipeline rejected: {msg}"),
    }
}

/// Keep the first error; later ones are usually its consequences.
fn record(slot: &Mutex<Option<String>>, msg: &str) {
    let mut slot = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if slot.is_none() {
        *slot = Some(msg.to_string());
    }
}

/// A one-shot for the staging buffer's map callback. It resolves under
/// `pollster` natively and under the browser's executor, and it is `Send`,
/// which the native callback bound requires.
#[derive(Clone, Default)]
struct MapSignal(Arc<Mutex<MapState>>);

#[derive(Default)]
struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

impl MapSignal {
    fn complete(&self, result: Result<(), wgpu::BufferAsyncError>) {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.result = Some(result);
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }

    fn wait(&self) -> MapWait {
        MapWait(self.clone())
    }
}

struct MapWait(MapSignal);

impl Future for MapWait {
    type Output = Result<(), wgpu::BufferAsyncError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self
            .0
             .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match state.result.take() {
            Some(result) => Poll::Ready(result),
            None => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Clears the busy flag however the trace ends.
struct BusyGuard<'a>(&'a Cell<bool>);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

/// Decode the raw segment buffer into per-ray world-millimetre segment lists,
/// counting the rays the shader marked as cut short.
/// A GPU trace's read-back as it lies in the staging buffer: each ray's
/// segment slots, `max_segs` of them, the used ones marked valid.
pub struct RawTrace<'a> {
    raw: &'a [Seg3Buf],
    scene: &'a GpuScene3,
    n_rays: usize,
    max_segs: usize,
}

impl<'a> RawTrace<'a> {
    /// Rays that used their whole segment budget with branches left, and
    /// rays that lost a branch to the full stack (see [`Gpu3Trace`]).
    pub fn truncated(&self) -> (usize, usize) {
        let (mut budget, mut stack) = (0, 0);
        if self.max_segs == 0 {
            return (0, 0);
        }
        for r in 0..self.n_rays {
            let first = self.raw[r * self.max_segs].flags;
            if first & SEG_VALID != 0 {
                if first & SEG_TRUNC_BUDGET != 0 {
                    budget += 1;
                }
                if first & SEG_TRUNC_STACK != 0 {
                    stack += 1;
                }
            }
        }
        (budget, stack)
    }

    /// Every segment, ray after ray, in world millimetres: the per-ray lists
    /// of [`RawTrace::decode`] one after another.
    pub fn segments(&self) -> RawSegments<'a> {
        let valid = self.raw.iter().filter(|s| s.flags & SEG_VALID != 0).count();
        RawSegments {
            slots: self.raw.iter(),
            scene: self.scene,
            left: valid,
        }
    }

    /// The per-ray segment lists and the truncation counts.
    pub fn decode(&self) -> Gpu3Trace {
        let (budget_truncated, stack_truncated) = self.truncated();
        let segments = (0..self.n_rays)
            .map(|r| {
                let slots = &self.raw[r * self.max_segs..(r + 1) * self.max_segs];
                RawSegments {
                    slots: slots.iter(),
                    scene: self.scene,
                    left: 0,
                }
                .collect()
            })
            .collect();
        Gpu3Trace {
            segments,
            budget_truncated,
            stack_truncated,
        }
    }
}

/// The valid segment slots of a read-back, decoded one at a time.
pub struct RawSegments<'a> {
    slots: std::slice::Iter<'a, Seg3Buf>,
    scene: &'a GpuScene3,
    /// Segments still to come, when counted (0 when not).
    left: usize,
}

impl Iterator for RawSegments<'_> {
    type Item = SegOut3;

    fn next(&mut self) -> Option<SegOut3> {
        let s = self.slots.by_ref().find(|s| s.flags & SEG_VALID != 0)?;
        self.left = self.left.saturating_sub(1);
        let a = glam::Vec3::new(s.a[0], s.a[1], s.a[2]);
        let b = glam::Vec3::new(s.b[0], s.b[1], s.b[2]);
        Some(SegOut3 {
            a: self.scene.norm.world_point(a),
            b: self.scene.norm.world_point(b),
            flux: s.flux,
            lambda_um: s.lambda,
            flags: s.flags & (SEG_TIR | SEG_GHOST),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, None)
    }
}

// Struct strides, one place so the buffer allocations cannot disagree with the
// packed types.
const S_BODY: usize = std::mem::size_of::<Body3Buf>();
const S_SURF: usize = std::mem::size_of::<Surface3Buf>();
const S_APER: usize = std::mem::size_of::<Aperture3Buf>();
const S_DET: usize = std::mem::size_of::<Detector3Buf>();
const S_RAY: usize = std::mem::size_of::<Ray3Buf>();
const S_VERT: usize = std::mem::size_of::<Vert3Buf>();
const S_FACE: usize = std::mem::size_of::<Face3Buf>();

fn write_if<T: bytemuck::Pod>(queue: &wgpu::Queue, buf: &wgpu::Buffer, data: &[T]) {
    if !data.is_empty() {
        queue.write_buffer(buf, 0, cast_slice(data));
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn uniform_buf(device: &wgpu::Device, bytes: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params3"),
        size: bytes as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn storage_buf(device: &wgpu::Device, label: &str, bytes: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(1) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn seg_buf(device: &wgpu::Device, bytes: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("segs3"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn staging_buf(device: &wgpu::Device, bytes: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging3"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
