//! wgpu compute backend: uploads a [`GpuScene`], dispatches the trace shader,
//! and reads back the segment buffer. Works headless (no surface) on a real GPU
//! or the software fallback adapter.
//!
//! M1 rebuilds buffers per trace for simplicity; a persistent-buffer, per-frame
//! dispatch (the actual interactive loop) is a perf task for M6.

use crate::scene::{GpuElem, GpuPrim, GpuRay, GpuScene, GpuSeg};
use bytemuck::{cast_slice, Zeroable};
use wgpu::util::DeviceExt;

/// A ready-to-use compute device and pipeline.
pub struct GpuTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Info about the adapter that was selected (for diagnostics/tests).
    pub adapter_info: wgpu::AdapterInfo,
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
            adapter_info,
        })
    }

    /// Trace `scene` for up to `max_depth` bounces. Returns an
    /// `n_rays * (max_depth + 1)` segment array (same layout as the CPU path).
    pub fn trace(&self, scene: &GpuScene, max_depth: u32) -> Vec<GpuSeg> {
        let params = scene.params(max_depth);
        let n_segs = (params.n_rays * params.max_segs) as usize;
        if n_segs == 0 {
            return Vec::new();
        }

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let prims_buf = storage_init(&self.device, "prims", &scene.prims, GpuPrim::zeroed());
        let elems_buf = storage_init(&self.device, "elems", &scene.elems, GpuElem::zeroed());
        let rays_buf = storage_init(&self.device, "rays", &scene.rays, GpuRay::zeroed());

        let seg_bytes = (n_segs * std::mem::size_of::<GpuSeg>()) as wgpu::BufferAddress;
        let segs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segs"),
            size: seg_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: seg_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace bg"),
            layout: &self.layout,
            entries: &[
                bind(0, &params_buf),
                bind(1, &prims_buf),
                bind(2, &elems_buf),
                bind(3, &rays_buf),
                bind(4, &segs_buf),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Clear so unused segment slots read back as invalid (flags = 0).
        encoder.clear_buffer(&segs_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.n_rays.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&segs_buf, 0, &staging, 0, seg_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Read back.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let out: Vec<GpuSeg> = cast_slice::<u8, GpuSeg>(&data).to_vec();
        drop(data);
        staging.unmap();
        out
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

/// Create a read-only storage buffer, padding empty vectors to one element
/// (wgpu forbids zero-sized buffers).
fn storage_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    pad: T,
) -> wgpu::Buffer {
    let one = [pad];
    let bytes: &[u8] = if data.is_empty() {
        cast_slice(&one)
    } else {
        cast_slice(data)
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}
