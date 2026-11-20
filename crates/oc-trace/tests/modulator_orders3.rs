//! DMDs and LCoS SLMs traced as diffraction orders (acceptance case C56).
//!
//! In the orders ray model each hit spawns the brightest orders of the
//! pattern's local unit cell, from the closed-form Fourier series
//! (`oc_core::modulator::unit_cell_orders`) with Harvey's renormalization:
//! J. E. Harvey et al., "Linear systems formulation of non-paraxial scalar
//! diffraction theory," *Proc. SPIE* 8122, 812205 (2011),
//! doi:10.1117/12.894562, which Texas Instruments' DLPA083B (2019, rev. 2023)
//! cites for its DMD efficiencies. Orders sit on the direction-cosine grid of
//! J. E. Harvey, C. L. Vernold, *Appl. Opt.* 37(34), 8158 (1998),
//! doi:10.1364/AO.37.008158. What is not followed is booked as untraced.

#![cfg(feature = "trace3")]

use oc_core::modulator::{
    cone_directions, dmd_projector_efficiency, on_axis_illumination, unit_cell_orders, DmdSpec, LcosBeam,
    ModulatorKind, OrderModel, Pattern, RayModel, SpatialModulator, TiltAxis, UnitCell,
};
use oc_core::scene3::{Scene3, SurfaceResponse};
use oc_core::vec3::{Ray3, Vec3};
use oc_trace::{trace3, LaunchRay3, TraceConfig3};

const LAM: f64 = 0.55;

fn device(pixels: u32) -> SpatialModulator {
    SpatialModulator {
        pixels: [pixels, pixels],
        pitch_um: 7.56,
        kind: ModulatorKind::Dmd(DmdSpec {
            tilt_deg: 12.0,
            tilt_axis: TiltAxis::Diagonal,
            mirror_fill: 0.94f64.sqrt(),
            reflectance: 1.0,
            window_transmittance: 1.0,
        }),
        pattern: Pattern::Uniform { value: 1 },
    }
}

/// A DMD facing −X at x = 100 mm, in the orders model.
fn scene(pixels: u32, model: RayModel) -> (Scene3, u64) {
    let mut s = Scene3::new();
    let id = s.add_modulator(Vec3::new(100.0, 0.0, 0.0), 180.0, device(pixels)).unwrap();
    if let SurfaceResponse::Modulator { ray_model, .. } = &mut s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap().response_front {
        *ray_model = model;
    }
    s.validate().unwrap();
    (s, id)
}

/// The device's frame, and a launch ray along `local` (a device-frame
/// direction toward −Z) aimed at its centre.
fn aimed(s: &Scene3, id: u64, local: Vec3) -> LaunchRay3 {
    let t = s.surfaces.iter().find(|x| x.common.id.0 == id).unwrap().common.transform;
    let d = (t.quat() * local).normalize();
    LaunchRay3 { ray: Ray3::new(t.pos - d * 50.0, d).unwrap(), lambda_um: LAM, flux: 1.0, source: None, pol: None }
}

fn cfg() -> TraceConfig3 {
    TraceConfig3 { ghosts: false, ..TraceConfig3::default() }
}

#[test]
fn dmd_orders_traced_match_the_unit_cell_order_for_order() {
    let (s, id) = scene(64, RayModel::Orders { max_orders: 12, min_efficiency: 0.0 });
    let d = device(64);
    let ModulatorKind::Dmd(spec) = &d.kind else { unreachable!() };
    let local = on_axis_illumination(spec);
    let res = trace3(&s, &[aimed(&s, id, local)], &cfg());
    let t = s.surfaces[0].common.transform;
    let mut want = unit_cell_orders(&d, &UnitCell::uniform(1), LAM, local, OrderModel::HarveyRenormalized, LcosBeam::Modulated)
        .unwrap()
        .orders;
    want.sort_by(|a, b| b.efficiency.partial_cmp(&a.efficiency).unwrap());
    want.truncate(12);
    // The children: every segment that starts at the device.
    let kids: Vec<_> = res.segments.iter().filter(|g| g.parent_path_id.is_some()).collect();
    assert_eq!(kids.len(), 12, "{kids:?}");
    for o in &want {
        let dir = (t.quat() * o.dir).normalize();
        let k = kids.iter().find(|g| ((g.b - g.a).normalize() - dir).length() < 1e-9);
        let k = k.unwrap_or_else(|| panic!("no child along order ({}, {})", o.m, o.n));
        assert!((k.flux_start - o.efficiency).abs() < 1e-12, "order ({}, {}): {} vs {}", o.m, o.n, k.flux_start, o.efficiency);
    }
    assert!(res.capability_report.approximated.iter().any(|a| format!("{a:?}").contains("Harvey renormalized")));
}

#[test]
fn untraced_orders_are_in_the_ledger() {
    let (s, id) = scene(64, RayModel::Orders { max_orders: 4, min_efficiency: 0.0 });
    let d = device(64);
    let ModulatorKind::Dmd(spec) = &d.kind else { unreachable!() };
    let local = on_axis_illumination(spec);
    let res = trace3(&s, &[aimed(&s, id, local)], &cfg());
    let table = unit_cell_orders(&d, &UnitCell::uniform(1), LAM, local, OrderModel::HarveyRenormalized, LcosBeam::Modulated).unwrap();
    let mut eff: Vec<f64> = table.orders.iter().map(|o| o.efficiency).collect();
    eff.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let rest: f64 = eff[4..].iter().sum();
    let l = res.ledger;
    eprintln!("measured: untraced orders {:.6}, the unit cell's {rest:.6}; dead space {:.6}; residual {:.3e}", l.untraced_orders, l.absorbed_surface, l.residual());
    assert!((l.untraced_orders - rest).abs() < 1e-12, "{l:?}");
    assert!((l.absorbed_surface - (1.0 - table.total)).abs() < 1e-12, "{l:?}");
    assert!(l.residual() < 1e-10, "{l:?}");
}

#[test]
fn ti_dlpa083_through_the_tracer_matches_direct_integration() {
    // TI DLPA083B §4.2's method, through the tracer: an f/2.4 cone of plane
    // waves along the direction the ON mirrors send onto the normal, a stop
    // admitting the f/2.4 projection cone 200 mm away, a detector behind it.
    // The share of the reflected light the detector sees must be what the
    // closed-form integration (W3) gives for the same 64 directions.
    let (fno, n): (f64, usize) = (2.4, 64);
    let (mut s, id) = scene(64, RayModel::Orders { max_orders: 4096, min_efficiency: 1e-9 });
    let na = 1.0 / (2.0 * fno);
    let radius = 200.0 * (na / (1.0 - na * na).sqrt());
    s.add_aperture(Vec3::new(-100.0, 0.0, 0.0), radius, 0.0);
    let det = s.add_area_detector(Vec3::new(-101.0, 0.0, 0.0), Vec3::X, 50.0, 50.0, [8, 8]);
    s.validate().unwrap();
    let d = device(64);
    let ModulatorKind::Dmd(spec) = &d.kind else { unreachable!() };
    let launches: Vec<_> = cone_directions(on_axis_illumination(spec), na, n).into_iter().map(|l| aimed(&s, id, l)).collect();
    let res = trace3(&s, &launches, &cfg());
    let collected = res.detectors.iter().find(|x| x.detector == det).unwrap().total_incident;
    let reflected = res.ledger.launched - res.ledger.absorbed_surface;
    let tracer = collected / reflected;
    let direct = dmd_projector_efficiency(&d, LAM, fno, fno, n, OrderModel::HarveyRenormalized).unwrap();
    eprintln!("measured: TI method through the tracer {tracer:.6}, direct {direct:.6}; untraced {:.3e}", res.ledger.untraced_orders / reflected);
    assert!((tracer - direct).abs() < 5e-3, "{tracer} vs {direct}");
    assert!(res.ledger.residual() < 1e-10, "{:?}", res.ledger);
}

#[test]
fn the_ray_model_round_trips_and_defaults_to_the_geometric_limit() {
    let (s, _) = scene(16, RayModel::Orders { max_orders: 8, min_efficiency: 1e-4 });
    let json = s.to_json();
    assert!(json.contains("Orders"));
    let back = Scene3::from_json(&json).unwrap();
    assert_eq!(back.surfaces, s.surfaces);
    let (g, _) = scene(16, RayModel::Geometric);
    assert!(!g.to_json().contains("ray_model"), "the default is left out on save");
    let (mut bad, id) = scene(16, RayModel::Geometric);
    if let SurfaceResponse::Modulator { ray_model, .. } = &mut bad.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap().response_front {
        *ray_model = RayModel::Orders { max_orders: 0, min_efficiency: 0.0 };
    }
    assert!(bad.validate().is_err());
}

#[test]
fn an_lcos_blazed_grating_sends_the_director_half_into_its_first_order() {
    // An 8-level staircase of period 8 pixels puts sinc²(1/8) = 0.94964 of the
    // modulated light into its first order (G. J. Swanson, MIT Lincoln
    // Laboratory TR-854 (1989)), at sin θ = λ/(8p) (Harvey & Vernold). Light
    // across the director is not modulated: a full-fill, lossless device
    // reflects it specularly. Unpolarized light splits evenly between them.
    use oc_core::modulator::{LcosSpec, PatternGenerator, PhaseLut};
    let mut s = Scene3::new();
    // Odd, so the generator's pixel coordinates are whole numbers and its
    // phases land exactly on the table's eight levels.
    let dev = SpatialModulator {
        pixels: [65, 65],
        pitch_um: 8.0,
        kind: ModulatorKind::Lcos(LcosSpec {
            director_deg: 0.0,
            lut: PhaseLut::Linear { levels: 8, max_phase_rad: 2.0 * std::f64::consts::PI },
            design_lambda_um: LAM,
            pixel_fill: 1.0,
            gap_amplitude: 0.0,
            gap_phase_rad: 0.0,
            fringing_um: 0.0,
            reflectance: 1.0,
        }),
        pattern: Pattern::Generated(PatternGenerator::BlazedGrating { period_px: 8.0, angle_deg: 0.0 }),
    };
    let id = s.add_modulator(Vec3::new(100.0, 0.0, 0.0), 180.0, dev).unwrap();
    if let SurfaceResponse::Modulator { ray_model, .. } = &mut s.surfaces[0].response_front {
        *ray_model = RayModel::Orders { max_orders: 2, min_efficiency: 0.0 };
    }
    s.validate().unwrap();
    let res = trace3(&s, &[aimed(&s, id, Vec3::new(0.0, 0.0, -1.0))], &TraceConfig3 { ghosts: false, polarization: false, ..TraceConfig3::default() });
    let mut kids: Vec<_> = res.segments.iter().filter(|g| g.parent_path_id.is_some()).collect();
    kids.sort_by(|a, b| b.flux_start.partial_cmp(&a.flux_start).unwrap());
    // The thin-element value is Swanson's closed form; the tracer uses
    // Harvey's renormalization, which shares the evanescent orders' power
    // (0.2% here, past |m| ≈ 116) among the propagating ones.
    let sinc = |x: f64| (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x);
    let dev = match &s.surfaces[0].response_front { SurfaceResponse::Modulator { device, .. } => (**device).clone(), _ => unreachable!() };
    let cell = oc_core::modulator::local_unit_cell(&dev, 32, 32).0;
    let tea = unit_cell_orders(&dev, &cell, LAM, Vec3::new(0.0, 0.0, -1.0), OrderModel::Tea, LcosBeam::Modulated).unwrap();
    let first = tea.brightest().unwrap();
    assert!((first.efficiency - sinc(1.0 / 8.0).powi(2)).abs() < 1e-12, "{} vs sinc²(1/8)", first.efficiency);
    let want = 0.5 * first.efficiency * tea.total / tea.propagating;
    assert!((kids[0].flux_start - 0.5).abs() < 1e-12, "{kids:?}");
    assert!((kids[1].flux_start - want).abs() < 1e-12, "{} vs {want}", kids[1].flux_start);
    let t = s.surfaces[0].common.transform;
    let local = t.quat().inverse() * (kids[1].b - kids[1].a).normalize();
    let sin_theta = local.x.hypot(local.y);
    eprintln!("measured: first order flux {:.9} (want {want:.9}), sin θ {sin_theta:.9} (want {:.9})", kids[1].flux_start, LAM / (8.0 * 8.0));
    assert!((sin_theta - LAM / 64.0).abs() < 1e-12);
}
