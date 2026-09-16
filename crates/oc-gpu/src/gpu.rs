//! wgpu compute backend: uploads a [`GpuScene`], dispatches the trace shader,
//! and reads back the segment buffer. Works headless (no surface) on a real GPU
//! or the software fallback adapter.
//!
//! Buffers are **persistent**: the GPU-side scene/ray/segment/staging buffers and
//! the bind group are cached on the tracer and reused across `trace()` calls,
//! growing only when the scene outgrows them. In the interactive drag loop the
//! scene and ray count are stable, so after the first frame every trace is pure
//! `write_buffer` + dispatch + read-back with no per-frame allocation. (This is
//! the persistent-buffer perf task the M1 code left as a TODO.)

use crate::scene::{GpuElem, GpuParams, GpuPrim, GpuRay, GpuScene, GpuSeg};
use bytemuck::cast_slice;
use std::cell::RefCell;

/// A ready-to-use compute device and pipeline.
pub struct GpuTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Persistent GPU buffers + bind group, reused (and grown) across traces.
    buffers: RefCell<Option<Buffers>>,
    /// Info about the adapter that was selected (for diagnostics/tests).
    pub adapter_info: wgpu::AdapterInfo,
}

/// Cached GPU buffers. `cap_*` are the current allocated capacities (in element
/// counts, except `cap_stage_bytes`); a buffer is reallocated only when the
/// scene needs more than it holds.
struct Buffers {
    params: wgpu::Buffer,
    prims: wgpu::Buffer,
    elems: wgpu::Buffer,
    rays: wgpu::Buffer,
    segs: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cap_prims: usize,
    cap_elems: usize,
    cap_rays: usize,
    cap_segs: usize,
    cap_stage_bytes: usize,
}

impl GpuTracer {
    /// Create a tracer, preferring a real high-performance adapter and falling
    /// back to the software adapter. Returns `None` if neither is available.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: true,
                compatible_surface: None,
            }))
        })?;
        Self::from_adapter(&adapter)
    }

    /// Build a tracer on a *specific* adapter. This is what per-device
    /// benchmarking and the performance-regression gate use to exercise every
    /// adapter `wgpu` enumerates (discrete GPU, integrated, software fallback),
    /// not just the one `new()` would auto-select. Returns `None` if the adapter
    /// cannot provide a compute device.
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Option<Self> {
        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("oc-gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trace pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "trace",
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            layout,
            buffers: RefCell::new(None),
            adapter_info,
        })
    }

    /// Trace `scene` for up to `max_depth` bounces. Returns an
    /// `n_rays * max_segs` segment array (same layout as the CPU path).
    ///
    /// Reuses cached GPU buffers across calls; only a scene that outgrows a
    /// buffer triggers a reallocation (and a bind-group rebuild).
    pub fn trace(&self, scene: &GpuScene, max_depth: u32) -> Vec<GpuSeg> {
        let params = scene.params(max_depth);
        let n_segs = (params.n_rays * params.max_segs) as usize;
        if n_segs == 0 {
            return Vec::new();
        }
        let seg_bytes = (n_segs * std::mem::size_of::<GpuSeg>()) as wgpu::BufferAddress;

        // Needed capacities (wgpu forbids zero-sized buffers, so clamp to >= 1).
        let need_prims = scene.prims.len().max(1);
        let need_elems = scene.elems.len().max(1);
        let need_rays = scene.rays.len().max(1);

        let mut slot = self.buffers.borrow_mut();
        if slot.is_none() {
            let params_buf = uniform_buf(&self.device, std::mem::size_of::<GpuParams>());
            let prims = storage_buf(
                &self.device,
                "prims",
                need_prims * std::mem::size_of::<GpuPrim>(),
            );
            let elems = storage_buf(
                &self.device,
                "elems",
                need_elems * std::mem::size_of::<GpuElem>(),
            );
            let rays = storage_buf(
                &self.device,
                "rays",
                need_rays * std::mem::size_of::<GpuRay>(),
            );
            let segs = seg_buf(&self.device, seg_bytes);
            let staging = staging_buf(&self.device, seg_bytes);
            let bind_group = self.make_bind_group(&params_buf, &prims, &elems, &rays, &segs);
            *slot = Some(Buffers {
                params: params_buf,
                prims,
                elems,
                rays,
                segs,
                staging,
                bind_group,
                cap_prims: need_prims,
                cap_elems: need_elems,
                cap_rays: need_rays,
                cap_segs: n_segs,
                cap_stage_bytes: seg_bytes as usize,
            });
        }
        let b = slot.as_mut().unwrap();

        // Grow any buffer the scene has outgrown; rebuild the bind group only if
        // a storage buffer identity actually changed.
        let mut rebuild_bg = false;
        if need_prims > b.cap_prims {
            b.prims = storage_buf(
                &self.device,
                "prims",
                need_prims * std::mem::size_of::<GpuPrim>(),
            );
            b.cap_prims = need_prims;
            rebuild_bg = true;
        }
        if need_elems > b.cap_elems {
            b.elems = storage_buf(
                &self.device,
                "elems",
                need_elems * std::mem::size_of::<GpuElem>(),
            );
            b.cap_elems = need_elems;
            rebuild_bg = true;
        }
        if need_rays > b.cap_rays {
            b.rays = storage_buf(
                &self.device,
                "rays",
                need_rays * std::mem::size_of::<GpuRay>(),
            );
            b.cap_rays = need_rays;
            rebuild_bg = true;
        }
        if n_segs > b.cap_segs {
            b.segs = seg_buf(&self.device, seg_bytes);
            b.cap_segs = n_segs;
            rebuild_bg = true;
        }
        if seg_bytes as usize > b.cap_stage_bytes {
            b.staging = staging_buf(&self.device, seg_bytes);
            b.cap_stage_bytes = seg_bytes as usize;
        }
        if rebuild_bg {
            b.bind_group = self.make_bind_group(&b.params, &b.prims, &b.elems, &b.rays, &b.segs);
        }

        // Upload the current scene into the (reused) buffers.
        self.queue.write_buffer(&b.params, 0, cast_slice(&[params]));
        if !scene.prims.is_empty() {
            self.queue
                .write_buffer(&b.prims, 0, cast_slice(&scene.prims));
        }
        if !scene.elems.is_empty() {
            self.queue
                .write_buffer(&b.elems, 0, cast_slice(&scene.elems));
        }
        if !scene.rays.is_empty() {
            self.queue.write_buffer(&b.rays, 0, cast_slice(&scene.rays));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Clear only the range we read back, so unused slots read as invalid.
        encoder.clear_buffer(&b.segs, 0, Some(seg_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &b.bind_group, &[]);
            pass.dispatch_workgroups(params.n_rays.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&b.segs, 0, &b.staging, 0, seg_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Read back only the used range (the staging buffer may be larger).
        let slice = b.staging.slice(0..seg_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let out: Vec<GpuSeg> = cast_slice::<u8, GpuSeg>(&data)[..n_segs].to_vec();
        drop(data);
        b.staging.unmap();
        out
    }

    fn make_bind_group(
        &self,
        params: &wgpu::Buffer,
        prims: &wgpu::Buffer,
        elems: &wgpu::Buffer,
        rays: &wgpu::Buffer,
        segs: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace bg"),
            layout: &self.layout,
            entries: &[
                bind(0, params),
                bind(1, prims),
                bind(2, elems),
                bind(3, rays),
                bind(4, segs),
            ],
        })
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

/// A uniform buffer of `bytes`, writable via `queue.write_buffer`.
fn uniform_buf(device: &wgpu::Device, bytes: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: bytes as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A read-only storage buffer, writable via `queue.write_buffer`.
fn storage_buf(device: &wgpu::Device, label: &str, bytes: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(1) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The read-write segment output buffer.
fn seg_buf(device: &wgpu::Device, bytes: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("segs"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The CPU-mappable read-back buffer.
fn staging_buf(device: &wgpu::Device, bytes: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
