//! Probe: can this machine give wgpu a compute-capable device? Tries a normal
//! adapter first, then forces the software fallback (DX12 WARP on Windows).

fn try_adapter(instance: &wgpu::Instance, fallback: bool) -> Option<wgpu::Adapter> {
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: fallback,
        compatible_surface: None,
    }))
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    println!("Enumerating adapters:");
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = a.get_info();
        println!(
            "  - {:?} {} ({:?}, {:?})",
            info.backend, info.name, info.device_type, info.driver
        );
    }

    for fallback in [false, true] {
        match try_adapter(&instance, fallback) {
            Some(adapter) => {
                let info = adapter.get_info();
                println!(
                    "\nfallback={fallback}: got {:?} '{}' type={:?}",
                    info.backend, info.name, info.device_type
                );
                let dev = pollster::block_on(adapter.request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("probe"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_defaults(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                ));
                match dev {
                    Ok(_) => println!("           device OK ✅"),
                    Err(e) => println!("           device FAILED: {e}"),
                }
            }
            None => println!("\nfallback={fallback}: no adapter"),
        }
    }
}
