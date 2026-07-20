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

use crate::buffers3::{
    Aperture3Buf, Body3Buf, Detector3Buf, Face3Buf, Gpu3Buffers, Params3, Ray3Buf, Seg3Buf,
    Surface3Buf, Vert3Buf,
};
use crate::cpu3::{SegOut3, SEG_GHOST, SEG_TIR};
use crate::scene3::GpuScene3;
use bytemuck::cast_slice;
use std::cell::RefCell;

const SEG_VALID: u32 = 1;

/// A ready-to-use compute device and pipeline for the 3D preview.
pub struct Gpu3Tracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buffers: RefCell<Option<Buffers>>,
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

    /// Build a tracer on a specific adapter (per-device benchmarking / gates).
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Option<Self> {
        let adapter_info = adapter.get_info();
        // The 3D path binds eight storage buffers (the object arrays plus the two
        // arenas and the segment output) — the WebGPU default ceiling, but above
        // the downlevel default of four, so lift just that one limit.
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("oc-gpu3 device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

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

        Some(Self {
            device,
            queue,
            pipeline,
            layout,
            buffers: RefCell::new(None),
            adapter_info,
        })
    }

    /// Trace `scene` for up to `max_depth` interactions, with a per-ray segment
    /// budget of `max_segs`. Returns per-ray world-millimetre segment lists (the
    /// same shape as [`crate::cpu3::trace_segments`]).
    pub fn trace(&self, scene: &GpuScene3, max_depth: u32, max_segs: u32) -> Vec<Vec<SegOut3>> {
        let bufs = Gpu3Buffers::from_scene(scene, max_depth, max_segs);
        let n_rays = bufs.rays.len();
        let n_segs = n_rays * max_segs as usize;
        if n_segs == 0 {
            return vec![Vec::new(); n_rays];
        }
        let seg_bytes = (n_segs * std::mem::size_of::<Seg3Buf>()) as wgpu::BufferAddress;

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
                &params, &bodies, &surfaces, &apertures, &detectors, &rays, &verts, &faces, &segs,
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

        let slice = b.staging.slice(0..seg_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let raw: &[Seg3Buf] = &cast_slice::<u8, Seg3Buf>(&data)[..n_segs];
        let out = decode(raw, scene, n_rays, max_segs as usize);
        drop(data);
        b.staging.unmap();
        out
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

/// Decode the raw segment buffer into per-ray world-millimetre segment lists.
fn decode(raw: &[Seg3Buf], scene: &GpuScene3, n_rays: usize, max_segs: usize) -> Vec<Vec<SegOut3>> {
    let mut out = Vec::with_capacity(n_rays);
    for r in 0..n_rays {
        let mut v = Vec::new();
        for i in 0..max_segs {
            let s = raw[r * max_segs + i];
            if s.flags & SEG_VALID == 0 {
                continue;
            }
            let a = glam::Vec3::new(s.a[0], s.a[1], s.a[2]);
            let b = glam::Vec3::new(s.b[0], s.b[1], s.b[2]);
            v.push(SegOut3 {
                a: scene.norm.world_point(a),
                b: scene.norm.world_point(b),
                flux: s.flux,
                lambda_um: s.lambda,
                flags: s.flags & (SEG_TIR | SEG_GHOST),
            });
        }
        out.push(v);
    }
    out
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
