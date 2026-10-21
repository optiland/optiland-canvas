//! The coherent wave path through a scene (acceptance case C53).
//!
//! Sources:
//! - [13] H. Kogelnik, T. Li, *Appl. Opt.* 5(10), 1550–1567 (1966): Gaussian
//!   beam propagation; the power a Gaussian puts through a centred disc of
//!   radius `a`, `1 − exp(−2a²/w²)`; the waist `λf/(πw₀)` at a lens's focus.
//! - [14] S. A. Self, *Appl. Opt.* 22(5), 658–661 (1983): the thin-lens
//!   transformation of a Gaussian waist.
//! - [1] J. W. Goodman, *Introduction to Fourier Optics*, 4th ed. (2017): the
//!   4f system images with magnification `−f₂/f₁`, its Fourier plane holding
//!   the object's spectrum at `λf₁` per unit spatial frequency; a flat mirror's
//!   reflected field is the incident field's mirror image.
//! - [10] J. E. Harvey, C. L. Vernold, *Appl. Opt.* 37(34), 8158 (1998): a
//!   grating's orders, here a DMD's, are `λ/p` apart in direction cosine, so a
//!   lens of focal length `f` focuses them `fλ/p` apart.
//! - [20] Z. Zhang, Z. You, D. Chu, *Light Sci. Appl.* 3, e213 (2014): an LCoS
//!   modulates the field along its director only.
//! - [21] G. J. Swanson, MIT Lincoln Laboratory TR-854 (1989): an N-level
//!   staircase puts `sinc²(1/N)` into the first order.
//! - [22] E. Wolf, *J. Opt. Soc. Am.* 72, 343–351 (1982): mutually incoherent
//!   modes add in power.
//! - [23] G. G. Stokes (1849), and [24] A. Zeilinger, *Am. J. Phys.* 49, 882
//!   (1981): a lossless splitter is a unitary two-port, its reflections from
//!   the two sides of opposite sign.
//! - [26] R. C. Jones, *J. Opt. Soc. Am.* 31(7), 488–493 (1941): the Jones
//!   calculus, a polarizer and a retarder as 2×2 matrices on the field's two
//!   components.
//! - [27] E. Hecht, *Optics*, 5th ed. (2017), ch. 8: Malus's law; a retarder
//!   between crossed polarizers; the Fresnel–Arago laws, orthogonally
//!   polarized beams do not interfere.
//!
//! Numbered as in docs/wave-optics.md; retrieved 2026-10-19.

#![cfg(feature = "scene")]

mod common;

use common::{measured, second_moment_radius_x};
use oc_core::id::{DetectorId, SourceId};
use oc_core::scene::Spectrum;
use oc_core::scene3::{Scene3, SurfaceResponse};
use oc_core::vec3::Vec3;
use oc_wave::beams::{rayleigh_range_mm, GaussianBeam};
use oc_wave::path::{BeamProfile, Readout, WavePathResult, WavePathSpec};
use oc_wave::SamplingWarning;

const LAM: f64 = 0.6328;

/// The path, with the ledger's own checks asserted on every result: a
/// passive model books no negative physical bucket and detects no more than
/// it launched (`WaveLedger::check`).
fn wave_path(s: &Scene3, spec: &WavePathSpec, revision: u64) -> Result<WavePathResult, oc_wave::WaveError> {
    let r = oc_wave::path::wave_path(s, spec, revision)?;
    assert!(r.report.inconsistencies.is_empty(), "ledger inconsistencies: {:?}\n{:?}", r.report.inconsistencies, r.report.ledger);
    Ok(r)
}

fn source(s: &mut Scene3) -> SourceId {
    s.add_collimated_beam(Vec3::ZERO, 1.0, 1.0, 1, Spectrum::Monochromatic(LAM))
}

fn spec(source: SourceId, detector: DetectorId, w0: f64, grid_n: usize, window_mm: f64) -> WavePathSpec {
    WavePathSpec {
        source,
        detector,
        beam: BeamProfile::Gaussian { waist_radius_mm: w0, waist_offset_mm: 0.0 },
        power: 1.0,
        grid_n,
        window_mm,
        device_samples_per_pixel: 4,
        readout: Readout::Projected,
        detector_model: oc_wave::path::DetectorModel::NormalFlux,
    }
}

fn lens(f: f64) -> SurfaceResponse {
    SurfaceResponse::ThinLens { focal_length_mm: f }
}

/// Give every body's faces a perfect anti-reflection: bare Fresnel with its
/// reflectance scaled to nothing.
fn perfect_ar(s: &mut Scene3) {
    let ar = SurfaceResponse::DielectricFresnel { ar_residual: oc_core::scene3::SpectralScalar::Constant(0.0) };
    for b in &mut s.bodies {
        b.surface.default.from_outside = ar.clone();
        b.surface.default.from_inside = ar.clone();
        b.surface.overrides.clear();
    }
}

#[test]
fn a_gaussian_through_empty_space_spreads_as_kogelnik_and_li_say() {
    // [13], through the scene's own source, sequence and detector.
    let mut s = Scene3::new();
    let src = source(&mut s);
    let det = s.add_area_detector(Vec3::new(300.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [64, 64]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, 0.2, 512, 6.0), 1).unwrap();
    let w = second_moment_radius_x(&r.field);
    let want = 0.2 * (1.0 + (300.0 / rayleigh_range_mm(0.2, LAM, 1.0)).powi(2)).sqrt();
    measured("scene wave path, Gaussian radius after 300 mm, relative error", ((w - want) / want).abs());
    assert!(((w - want) / want).abs() < 1e-5);
    let b = GaussianBeam::new(0.2, LAM, 1.0, 1.0);
    let mut worst: f64 = 0.0;
    for j in 0..512 {
        for i in 0..512 {
            let (x, y) = r.field.grid.xy(i, j);
            worst = worst.max((r.field.at(i, j) - b.value(x, y, 300.0)).abs());
        }
    }
    let peak = b.value(0.0, 0.0, 300.0).abs();
    measured("scene wave path, Gaussian field after 300 mm, worst / peak", worst / peak);
    assert!(worst / peak < 1e-5);
    let on: f64 = r.power.iter().sum();
    assert!((on - 1.0).abs() < 1e-9, "the detector catches the whole beam: {on}");
    assert!(r.report.ledger.numerical.abs() < 1e-9, "{:?}", r.report.ledger);
}

#[test]
fn a_thin_lens_in_the_scene_focuses_like_self() {
    // [14]: waist 0.5 mm at the source, lens f = 200 mm at s = 100 mm.
    let (w0, f, s_lens) = (0.5, 200.0, 100.0);
    let zr = rayleigh_range_mm(w0, LAM, 1.0);
    let s2 = 1.0 / (1.0 / f - 1.0 / (s_lens + zr * zr / (s_lens - f)));
    let w2 = w0 / ((1.0 - s_lens / f).powi(2) + (zr / f).powi(2)).sqrt();
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_planar_element(Vec3::new(s_lens, 0.0, 0.0), 10.0, 0.0, lens(f), lens(f));
    let det = s.add_area_detector(Vec3::new(s_lens + s2, 0.0, 0.0), -Vec3::X, 1.0, 1.0, [32, 32]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, w0, 512, 4.096), 1).unwrap();
    assert!(r.report.steps.iter().any(|st| st.note.contains("thin lens")));
    let w = second_moment_radius_x(&r.field);
    measured("scene thin lens, Self waist radius, relative error", ((w - w2) / w2).abs());
    assert!(((w - w2) / w2).abs() < 1e-4, "{w} vs {w2}");
}

#[test]
fn a_fold_mirror_hands_on_the_mirror_image_exactly() {
    // [1]: a flat mirror's reflected field is the incident field's mirror
    // image. Folded through 90° or carried straight, the beam must arrive the
    // same, and the image-theory handover reports no tilt projection.
    let mut folded = Scene3::new();
    let src = source(&mut folded);
    folded.add_planar_mirror(Vec3::new(97.0, -3.0, 0.0), Vec3::new(103.0, 3.0, 0.0), 1.0).unwrap();
    let det = folded.add_area_detector(Vec3::new(100.0, 200.0, 0.0), -Vec3::Y, 3.0, 3.0, [32, 32]);
    folded.validate().unwrap();
    let a = wave_path(&folded, &spec(src, det, 0.2, 256, 3.0), 1).unwrap();
    // The straight reference takes the same two steps, 100 mm and 200 mm, so
    // the propagators match and only the handover differs.
    let g = oc_wave::Grid2::square(256, 3.0 / 256.0).unwrap();
    let u0 = GaussianBeam::new(0.2, LAM, 1.0, 1.0).sample(g, 0.0).unwrap();
    let mut fft = oc_wave::Fft2::new();
    let u1 = oc_wave::propagate(&u0, 100.0, oc_wave::Propagator::Auto, &mut fft).unwrap().0;
    let b = oc_wave::propagate(&u1, 200.0, oc_wave::Propagator::Auto, &mut fft).unwrap().0;
    assert!(!a.report.warnings.iter().any(|w| matches!(w, SamplingWarning::TiltProjection { .. })));
    let peak = b.data.iter().map(|z| z.abs()).fold(0.0, f64::max);
    let mut worst: f64 = 0.0;
    for (x, y) in a.field.data.iter().zip(&b.data) {
        worst = worst.max((x.abs() - y.abs()).abs());
    }
    measured("folded vs straight wave path, |U| worst / peak", worst / peak);
    assert!(worst / peak < 1e-9, "{worst}");
}

#[test]
fn an_aperture_on_the_path_is_found_although_the_trace_logs_no_event() {
    // [13]: a centred disc of radius a passes 1 − exp(−2a²/w²) of a Gaussian.
    let (w0, a) = (0.5, 0.4);
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_aperture(Vec3::new(50.0, 0.0, 0.0), a, 0.0);
    let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [16, 16]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, w0, 512, 4.0), 1).unwrap();
    assert!(r.report.steps.iter().any(|st| st.note.contains("aperture")));
    let w = w0 * (1.0 + (50.0 / rayleigh_range_mm(w0, LAM, 1.0)).powi(2)).sqrt();
    let want = (-2.0 * a * a / (w * w)).exp();
    let blocked = r.report.ledger.absorbed_aperture / r.report.ledger.launched;
    measured("aperture on the path, blocked fraction, |Δ|", (blocked - want).abs());
    assert!((blocked - want).abs() < 1e-4, "{blocked} vs {want}");
}

/// A 4f system: lenses f₁ and f₂, the object at the front focal plane.
fn four_f(iris: Option<f64>) -> (Scene3, SourceId, DetectorId) {
    let (f1, f2) = (100.0, 50.0);
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_planar_element(Vec3::new(f1, 0.0, 0.0), 10.0, 0.0, lens(f1), lens(f1));
    if let Some(r) = iris {
        // The builder keeps openings of at least 0.2 mm for drawing; this iris
        // is set directly.
        s.add_aperture(Vec3::new(2.0 * f1, 0.0, 0.0), r, 0.0);
        s.apertures.last_mut().unwrap().shape.clear_opening = oc_core::scene3::PlanarShape2::Disc { radius_mm: r };
    }
    s.add_planar_element(Vec3::new(2.0 * f1 + f2, 0.0, 0.0), 10.0, 0.0, lens(f2), lens(f2));
    let det = s.add_area_detector(Vec3::new(2.0 * f1 + 2.0 * f2, 0.0, 0.0), -Vec3::X, 2.0, 2.0, [64, 64]);
    s.validate().unwrap();
    (s, src, det)
}

#[test]
fn a_4f_system_images_at_minus_f2_over_f1_and_its_iris_cuts_the_spectrum() {
    // [1]: the image is the object scaled by −f₂/f₁ = −1/2. A Gaussian object
    // of waist w₀ images to a waist w₀/2 at the back focal plane.
    let w0 = 0.3;
    let (s, src, det) = four_f(None);
    let r = wave_path(&s, &spec(src, det, w0, 512, 4.0), 1).unwrap();
    let w = second_moment_radius_x(&r.field);
    measured("4f image waist / (w₀ f₂/f₁)", w / (w0 * 0.5));
    assert!((w / (w0 * 0.5) - 1.0).abs() < 1e-4, "{w}");
    // The Fourier plane holds the spectrum: a waist λf₁/(πw₀) [13]. An iris
    // of radius r there passes 1 − exp(−2r²/w_F²): the coherent cutoff at
    // spatial frequency r/(λf₁).
    let wf = LAM / 1000.0 * 100.0 / (std::f64::consts::PI * w0);
    let iris = 0.8 * wf;
    let (s, src, det) = four_f(Some(iris));
    let r = wave_path(&s, &spec(src, det, w0, 512, 4.0), 1).unwrap();
    let blocked = r.report.ledger.absorbed_aperture / r.report.ledger.launched;
    let want = (-2.0 * iris * iris / (wf * wf)).exp();
    measured("4f Fourier-plane iris, blocked fraction, |Δ|", (blocked - want).abs());
    assert!((blocked - want).abs() < 2e-3, "{blocked} vs {want}");
    // The ledger closes along the path.
    let l = r.report.ledger;
    measured("4f with iris, ledger numerical / launched", l.numerical.abs() / l.launched);
    assert!(l.numerical.abs() < 1e-6 * l.launched, "{l:?}");
}

#[test]
fn wave_and_ray_agree_on_where_the_beam_lands() {
    // A mirror tilted off 45° sends the beam off the detector's centre; the
    // wave's power centroid must land where the chief ray does.
    let mut s = Scene3::new();
    let src = source(&mut s);
    let a = 40f64.to_radians();
    let half = Vec3::new(a.cos(), a.sin(), 0.0) * 3.0;
    let c = Vec3::new(100.0, 0.0, 0.0);
    s.add_planar_mirror(c - half, c + half, 1.0).unwrap();
    let det = s.add_area_detector(Vec3::new(100.0, 150.0, 0.0), -Vec3::Y, 30.0, 3.0, [600, 60]);
    s.validate().unwrap();
    let r: WavePathResult = wave_path(&s, &spec(src, det, 0.3, 256, 4.0), 1).unwrap();
    let seq = oc_trace::extract_sequence3(&s, src, det, 1).unwrap();
    let (nu, nv) = (r.bins[0], r.bins[1]);
    let (du, dv) = (60.0 / nu as f64, 6.0 / nv as f64);
    let (mut s0, mut su, mut sv) = (0.0, 0.0, 0.0);
    for j in 0..nv {
        for i in 0..nu {
            let p = r.power[j * nu + i];
            s0 += p;
            su += p * (-30.0 + (i as f64 + 0.5) * du);
            sv += p * (-3.0 + (j as f64 + 0.5) * dv);
        }
    }
    let (cu, cv) = (su / s0, sv / s0);
    let ray = seq.terminal.uv_mm;
    let e = ((cu - ray.x).powi(2) + (cv - ray.y).powi(2)).sqrt();
    measured("wave centroid vs chief ray on the detector, mm", e);
    assert!(e < 0.05 * du, "{cu}, {cv} vs {ray:?}");
}

#[test]
fn a_dmd_in_the_scene_focuses_its_orders_f_lambda_over_p_apart() {
    // [10]: a 10 µm, 12° side-tilt DMD lit at 24°, its ON mirrors sending the
    // beam along the device normal; a lens f = 15 mm across that beam focuses
    // the orders f·λ/p = 0.949 mm apart on the detector.
    use oc_core::modulator::{facet_normal, DmdSpec, DmdState, ModulatorKind, Pattern, SpatialModulator, TiltAxis};
    let p_um = 10.0;
    let mut s = Scene3::new();
    let src = source(&mut s);
    let phi = 180.0 - 24.0;
    let mk = |value: u16| SpatialModulator {
        // Odd, so the chief ray meets a mirror's centre rather than a gap.
        pixels: [401, 401],
        pitch_um: p_um,
        kind: ModulatorKind::Dmd(DmdSpec {
            tilt_deg: 12.0,
            tilt_axis: TiltAxis::Side,
            mirror_fill: 0.95,
            reflectance: 1.0,
            window_transmittance: 1.0,
        }),
        pattern: Pattern::Uniform { value },
    };
    let dmd_pos = Vec3::new(100.0, 0.0, 0.0);
    let id = s.add_modulator(dmd_pos, phi, mk(1)).unwrap();
    // Pick the state whose facet sends the beam back along the device normal.
    let t = s.surfaces.iter().find(|x| x.common.id.0 == id).unwrap().common.transform;
    let n_dev = t.rotate(Vec3::Z);
    let bis = (n_dev - Vec3::X).normalize();
    let ModulatorKind::Dmd(spec_d) = &mk(1).kind else { unreachable!() };
    let on = t.rotate(facet_normal(spec_d, DmdState::On));
    let value = if on.dot(bis) > t.rotate(facet_normal(spec_d, DmdState::Off)).dot(bis) { 1 } else { 0 };
    if let SurfaceResponse::Modulator { device, .. } = &mut s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap().response_front {
        device.pattern = Pattern::Uniform { value };
    }
    let f = 15.0;
    // The lens sits 5 mm along the outgoing ray, small enough (1 mm) that the
    // incoming chief ray passes beside it; its rim clips only the outgoing
    // beam's far tail.
    let lens_pos = dmd_pos + n_dev * 5.0;
    let lens_tilt = n_dev.y.atan2(n_dev.x).to_degrees();
    s.add_planar_element(lens_pos, 1.0, lens_tilt, lens(f), lens(f));
    let det = s.add_area_detector(lens_pos + n_dev * f, -n_dev, 1.25, 1.25, [16, 16]);
    s.validate().unwrap();
    let mut sp = spec(src, det, 0.3, 1024, 2.56);
    sp.device_samples_per_pixel = 4;
    let r = wave_path(&s, &sp, 1).unwrap();
    assert!(r.report.steps.iter().any(|st| st.note.contains("dmd")), "{:?}", r.report.steps);
    // The field at the lens's focus, across the chief ray. Every direction on
    // the path lies in the XY plane, so the beam's x axis stays in it and the
    // orders (m, 0) lie along the row through the centre.
    let g = r.field.grid;
    let jc = g.n[1] / 2;
    let row: Vec<f64> = (0..g.n[0]).map(|i| r.field.at(i, jc).norm_sqr()).collect();
    let max = row.iter().cloned().fold(0.0, f64::max);
    let mut peaks = vec![];
    for i in 2..g.n[0] - 2 {
        if row[i] > 0.01 * max && row[i] >= row[i - 1] && row[i] >= row[i + 1] && row[i] > row[i - 2] && row[i] > row[i + 2] {
            let off = 0.5 * (row[i - 1] - row[i + 1]) / (row[i - 1] - 2.0 * row[i] + row[i + 1]);
            peaks.push(g.coord(0, i) + off * g.pitch_mm[0]);
        }
    }
    let want = f * LAM / 1000.0 / (p_um / 1000.0);
    measured("DMD orders focused: number of peaks above 1%", peaks.len() as f64);
    assert!(peaks.len() >= 2, "{peaks:?}");
    for w in peaks.windows(2) {
        let gap = w[1] - w[0];
        measured("DMD orders focused: spacing / (fλ/p)", gap / want);
        assert!((gap / want - 1.0).abs() < 0.01, "{gap} vs {want}");
    }
    let l = r.report.ledger;
    measured("DMD wave path: detected / launched", l.detected / l.launched);
    measured("DMD wave path: numerical / launched", l.numerical / l.launched);
    eprintln!("{l:?}");
    assert!(l.numerical.abs() < 1e-3 * l.launched, "{l:?}");
}

#[test]
fn slicing_a_wave_job_changes_no_bit() {
    // A worker runs the path an event at a time so that it can drop a stale
    // job between events; that must not change the answer.
    use oc_wave::path::WaveJob;
    let wf = LAM / 1000.0 * 100.0 / (std::f64::consts::PI * 0.3);
    let (s, src, det) = four_f(Some(0.8 * wf));
    let sp = spec(src, det, 0.3, 256, 4.0);
    let whole = wave_path(&s, &sp, 7).unwrap();
    let mut job = WaveJob::new(&s, &sp, 7).unwrap();
    assert_eq!(job.revision(), 7);
    let events = job.events();
    assert!(events >= 4, "lens, iris, lens, detector: {events}");
    let mut steps = 0;
    while !job.step(1).unwrap() {
        steps += 1;
        assert_eq!(job.done(), steps);
    }
    let sliced = job.finish().unwrap();
    assert_eq!(whole, sliced, "the sliced job differs from the whole one");
    // The detector is not read before the chain is done.
    let mut early = WaveJob::new(&s, &sp, 7).unwrap();
    early.step(1).unwrap();
    assert!(early.finish().is_err());
}

#[test]
fn wave_power_agrees_with_rays_when_the_fresnel_number_is_large() {
    // A uniform beam of radius R through a centred opening of radius a: at a
    // Fresnel number a²/(λz) above 1000 the power reaching the detector is
    // the geometric share a²/R² to well within 1% (Goodman [1]: the edge's
    // diffraction is a fringe of width ~√(λz) around a large bright area),
    // and the f64 ray tracer, which knows no diffraction, must say the same.
    use oc_trace::{trace3, LaunchRay3, TraceConfig3};
    let (r_beam, a, z) = (3.0, 2.0, 5.0);
    let nf = a * a / (LAM / 1000.0 * z);
    assert!(nf > 1000.0, "{nf}");
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_aperture(Vec3::new(10.0, 0.0, 0.0), a, 0.0);
    let det = s.add_area_detector(Vec3::new(10.0 + z, 0.0, 0.0), -Vec3::X, 8.0, 8.0, [32, 32]);
    s.validate().unwrap();

    let mut sp = spec(src, det, 0.3, 1024, 8.0);
    sp.beam = BeamProfile::TopHat { radius_mm: r_beam };
    let r = wave_path(&s, &sp, 1).unwrap();
    let wave = r.report.ledger.detected / r.report.ledger.launched;

    // Rays on a square grid filling the same disc, equal flux each.
    let n = 600;
    let h = 2.0 * r_beam / n as f64;
    let mut launch = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let (y, zz) = (-r_beam + (i as f64 + 0.5) * h, -r_beam + (j as f64 + 0.5) * h);
            if y * y + zz * zz <= r_beam * r_beam {
                launch.push(LaunchRay3 {
                    ray: oc_core::vec3::Ray3::new(Vec3::new(0.0, y, zz), Vec3::X).unwrap(),
                    lambda_um: LAM,
                    flux: 1.0,
                    source: None,
                    pol: None,
                });
            }
        }
    }
    let count = launch.len() as f64;
    let res = trace3(&s, &launch, &TraceConfig3::default());
    let rays = res.detectors.iter().find(|d| d.detector == det).unwrap().total_incident / count;
    let geometric = (a / r_beam).powi(2);
    measured("wave vs rays, power share on the detector, relative difference", (wave - rays).abs() / rays);
    measured("wave share vs a²/R², relative", (wave - geometric).abs() / geometric);
    measured("ray share vs a²/R², relative", (rays - geometric).abs() / geometric);
    assert!((wave - rays).abs() < 0.01 * rays, "wave {wave} vs rays {rays}");
}

#[test]
fn a_source_carries_its_coherent_beam_through_the_file_and_into_its_wave_path() {
    // The beam is the scene's: saved with the source, launched with the
    // source's flux as its power, dropped by name when written as v2.
    use oc_core::scene3::CoherentBeam;
    let mut s = Scene3::new();
    let src = source(&mut s);
    let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [16, 16]);
    s.validate().unwrap();
    // No beam declared: refused by name.
    let e = WavePathSpec::for_source(&s, src, det, 256, 4.0).unwrap_err();
    assert!(e.to_string().contains("no coherent beam"), "{e}");

    s.sources[0].coherence = Some(CoherentBeam::Gaussian { waist_radius_mm: 0.3, waist_offset_mm: -20.0 });
    s.sources[0].flux = 2.5;
    s.validate().unwrap();
    let back = Scene3::from_json(&s.to_json()).unwrap();
    assert_eq!(back.sources, s.sources);
    assert!(s.to_json().contains("coherence"));
    let sp = WavePathSpec::for_source(&back, src, det, 256, 4.0).unwrap();
    assert_eq!(sp.beam, BeamProfile::Gaussian { waist_radius_mm: 0.3, waist_offset_mm: -20.0 });
    let r = wave_path(&back, &sp, 1).unwrap();
    assert!((r.report.ledger.launched - 2.5).abs() < 1e-9 * 2.5, "{:?}", r.report.ledger);

    // A source traced only as rays writes no key at all.
    let mut plain = s.clone();
    plain.sources[0].coherence = None;
    assert!(!plain.to_json().contains("coherence"));

    let (v2, losses) = s.to_json_v2();
    assert!(!v2.contains("coherence"));
    assert!(losses.iter().any(|l| l.contains("coherent beam")), "{losses:?}");

    s.sources[0].coherence = Some(CoherentBeam::TopHat { radius_mm: -1.0 });
    assert!(s.validate().is_err());
}

#[test]
fn a_continuous_source_is_refused_and_lines_add_in_power() {
    // A coherent field has one wavelength: a source of several lines is a set
    // of mutually incoherent modes (Wolf 1982 [22]), each run coherently with
    // its share of the power and added in power. A continuum is refused by
    // name rather than sampled into lines the source did not declare.
    let run = |spectrum: Spectrum| {
        let mut s = Scene3::new();
        let src = s.add_collimated_beam(Vec3::ZERO, 1.0, 1.0, 1, spectrum);
        let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::X, 1.0, 1.0, [16, 16]);
        s.validate().unwrap();
        wave_path(&s, &spec(src, det, 0.3, 64, 4.0), 1)
    };
    let e = run(Spectrum::Continuum(vec![(0.45, 1.0), (0.55, 1.0), (0.65, 1.0)])).unwrap_err().to_string();
    assert!(e.contains("continuous spectrum") && e.contains("single line"), "{e}");
    let (a, b) = (0.486, 0.656);
    let both = run(Spectrum::Lines(vec![(a, 3.0), (b, 1.0)])).unwrap();
    let (ra, rb) = (run(Spectrum::Monochromatic(a)).unwrap(), run(Spectrum::Monochromatic(b)).unwrap());
    let worst = both.power.iter().zip(ra.power.iter().zip(&rb.power)).map(|(p, (x, y))| (p - (0.75 * x + 0.25 * y)).abs()).fold(0.0, f64::max);
    let peak = both.power.iter().cloned().fold(0.0, f64::max);
    measured("two lines: power per pixel against the lines' own, weighted, worst / peak", worst / peak);
    assert!(worst / peak < 1e-12, "{}", worst / peak);
    assert!((both.report.ledger.launched - 1.0).abs() < 1e-12);
    assert!(both.report.approximated.iter().any(|a| a.contains("2 spectral lines")), "{:?}", both.report.approximated);
}

/// A Mach–Zehnder interferometer in the XY plane: 50/50 splitters at
/// (50, 0) and (150, 100), mirrors at (150, 0) and (50, 100), all at 45°, the
/// second mirror moved `delta` mm along its normal. Its two outputs leave
/// along +X at y = 100 and along +Y at x = 150.
fn mach_zehnder(delta: f64) -> (Scene3, SourceId, DetectorId, DetectorId) {
    let mut s = Scene3::new();
    let src = source(&mut s);
    let m = SurfaceResponse::Mirror { reflectivity: oc_core::scene3::SpectralScalar::Constant(1.0) };
    let n = Vec3::new(-1.0, 1.0, 0.0).normalize();
    s.add_beam_splitter(Vec3::new(50.0, 0.0, 0.0), 10.0, 135.0, 0.5, 0.5).unwrap();
    s.add_planar_element(Vec3::new(150.0, 0.0, 0.0), 10.0, 135.0, m.clone(), m.clone());
    s.add_planar_element(Vec3::new(50.0, 100.0, 0.0) + n * delta, 10.0, 135.0, m.clone(), m);
    s.add_beam_splitter(Vec3::new(150.0, 100.0, 0.0), 10.0, 135.0, 0.5, 0.5).unwrap();
    let dx = s.add_area_detector(Vec3::new(250.0, 100.0, 0.0), -Vec3::X, 3.0, 3.0, [64, 64]);
    let dy = s.add_area_detector(Vec3::new(150.0, 200.0, 0.0), -Vec3::Y, 3.0, 3.0, [64, 64]);
    s.validate().unwrap();
    (s, src, dx, dy)
}

#[test]
fn a_mach_zehnder_sends_all_the_power_out_of_its_two_ports_whatever_the_path_difference() {
    // Bundle M09: a network of lossless parts is unitary, so the two output
    // ports of a Mach–Zehnder carry all the power between them at every path
    // difference, and trade it as (1 ∓ cos Δφ)/2. That holds only if each
    // splitter is a unitary two-port, which a real √R, √T on both arms is not
    // (its outputs would sum to 1 + cos Δφ); the splitter takes the Stokes
    // relations' signs [23], [24]. Moving the second mirror δ along its normal
    // lengthens its arm by √2·δ.
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    for j in 0..4 {
        let dphi = j as f64 * std::f64::consts::FRAC_PI_2;
        let delta = dphi / (k0 * std::f64::consts::SQRT_2);
        let (s, src, dx, dy) = mach_zehnder(delta);
        let rx = wave_path(&s, &spec(src, dx, 0.5, 256, 4.0), 1).unwrap();
        let ry = wave_path(&s, &spec(src, dy, 0.5, 256, 4.0), 1).unwrap();
        let (px, py) = (rx.report.ledger.detected, ry.report.ledger.detected);
        let share = px / (px + py);
        let want = 0.5 * (1.0 - dphi.cos());
        measured(&format!("Mach–Zehnder at Δφ = {j}·π/2: both ports' power, of the launched"), px + py);
        measured(&format!("Mach–Zehnder at Δφ = {j}·π/2: the X port's share against (1 − cos Δφ)/2"), share - want);
        assert!((px + py - 1.0).abs() < 1e-3, "{px} + {py}");
        assert!((share - want).abs() < 2e-3, "{share} vs {want}");
        assert!(rx.report.steps.iter().any(|st| st.note.contains("arm 2 joins arm 1")), "{:?}", rx.report.steps);
        // The arms' readout onto the detector's grid books what it takes (or
        // gains, reading the spectrum between its samples) in the arm's own
        // ledger, so the network's remainder is the readouts': small, named.
        measured(&format!("Mach–Zehnder at Δφ = {j}·π/2: the network ledger's remainder, of the launched"), rx.report.ledger.numerical);
        assert!(rx.report.ledger.numerical.abs() < 1e-8, "{:?}", rx.report.ledger);
    }
}

/// A Michelson interferometer in the XY plane: a 50/50 splitter at (50, 0)
/// at 45°, mirror 1 at (150 + d1, 0) facing back along −X, mirror 2 at
/// (50, 100 + d2) facing back along −Y and tilted `alpha` rad about Z; the
/// output leaves along −Y onto a detector at (50, −100).
fn michelson(d1: f64, d2: f64, alpha: f64, spectrum: Spectrum, half: f64, bins: usize) -> (Scene3, SourceId, DetectorId) {
    let mut s = Scene3::new();
    let src = s.add_collimated_beam(Vec3::ZERO, 1.0, 1.0, 1, spectrum);
    let m = SurfaceResponse::Mirror { reflectivity: oc_core::scene3::SpectralScalar::Constant(1.0) };
    s.add_beam_splitter(Vec3::new(50.0, 0.0, 0.0), 10.0, 135.0, 0.5, 0.5).unwrap();
    s.add_planar_element(Vec3::new(150.0 + d1, 0.0, 0.0), 10.0, 180.0, m.clone(), m.clone());
    s.add_planar_element(Vec3::new(50.0, 100.0 + d2, 0.0), 10.0, -90.0 + alpha.to_degrees(), m.clone(), m);
    let det = s.add_area_detector(Vec3::new(50.0, -100.0, 0.0), Vec3::Y, half, half, [bins, bins]);
    s.validate().unwrap();
    (s, src, det)
}

#[test]
fn a_michelson_with_a_tilted_mirror_makes_fringes_of_period_lambda_over_sin_2_alpha() {
    // Two plane waves crossing at an angle 2α make fringes of period
    // λ/sin 2α across the plane they cross on [1]. A mirror tilted α turns its
    // arm by 2α. The period is read from the power's spectrum along the face.
    // The beams are nearly plane. A beam's wavefront is a sphere of radius R,
    // and the tilted arm's is shifted 2α(R − D) across the face, D = 200 mm
    // from the mirror, which lowers the fringes' frequency by D/R: for a 1 mm
    // beam (R ≈ 71 m here) −2.8e-3, where the path reads −3.5e-3; for this
    // 3 mm beam −3.5e-5, where it reads −7.1e-5. The rest is of the order of
    // the tilted-plane readout's own error (its spectrum read between
    // samples).
    let alpha = 5e-4;
    let (s, src, det) = michelson(0.0, 0.0, alpha, Spectrum::Monochromatic(LAM), 2.0, 256);
    let r = wave_path(&s, &spec(src, det, 3.0, 512, 16.0), 1).unwrap();
    let d = s.detectors.iter().find(|x| x.common.id.0 == det.0).unwrap();
    let t = d.common.transform;
    let (nb, du) = (256, 4.0 / 256.0);
    // |Σ w (P − P̄) e^{−i2πfx}| over the pixels, x their world X, w a Hann
    // window over the face: its peak is the fringes' frequency. The window and
    // the mean taken out keep the face's edges and the mean power from
    // leaking into the peak (a hard-edged face's sidelobes bias it by 3e-3).
    let px: Vec<(f64, f64)> = (0..nb * nb)
        .map(|k| {
            let (i, j) = (k % nb, k / nb);
            let (u, v) = (-2.0 + (i as f64 + 0.5) * du, -2.0 + (j as f64 + 0.5) * du);
            let w = (std::f64::consts::PI * u / 4.0).cos().powi(2) * (std::f64::consts::PI * v / 4.0).cos().powi(2);
            (t.transform_point(Vec3::new(u, v, 0.0)).x - 50.0, w)
        })
        .collect();
    let wsum: f64 = px.iter().map(|p| p.1).sum();
    let mean = px.iter().zip(&r.power).map(|(p, q)| p.1 * q).sum::<f64>() / wsum;
    let spectrum_at = |f: f64| {
        let mut acc = oc_wave::Cx::ZERO;
        for (k, (x, w)) in px.iter().enumerate() {
            acc += oc_wave::Cx::expi(-2.0 * std::f64::consts::PI * f * x) * (w * (r.power[k] - mean));
        }
        acc.abs()
    };
    let want = (2.0 * alpha).sin() / (LAM / 1000.0);
    let (mut best, mut f_best) = (0.0, 0.0);
    for k in 0..=200 {
        let f = want * (0.8 + 0.4 * k as f64 / 200.0);
        let v = spectrum_at(f);
        if v > best {
            best = v;
            f_best = f;
        }
    }
    // A parabola through the peak and its neighbours.
    let h = want * 0.4 / 200.0;
    let (a, b, c) = (spectrum_at(f_best - h), best, spectrum_at(f_best + h));
    let f_peak = f_best + 0.5 * h * (a - c) / (a - 2.0 * b + c);
    measured("Michelson, mirror tilted 0.5 mrad: fringe frequency against sin 2α / λ, relative", f_peak / want - 1.0);
    assert!((f_peak / want - 1.0).abs() < 2e-3, "{f_peak} vs {want} /mm");
}

#[test]
fn moving_a_michelson_mirror_by_d_shifts_its_fringes_by_two_k_d() {
    // The light crosses the arm twice, so moving its mirror by d changes the
    // phase between the arms by 2k₀d [1]: four steps of λ/8 are four steps of
    // π/2, and four powers a quarter period apart satisfy P₀ + P₂ = P₁ + P₃.
    // A single pass, k₀d, would break it by the fringes' amplitude. The
    // visibility of two equal, overlapping arms is 1.
    let p: Vec<f64> = (0..4)
        .map(|j| {
            let (s, src, det) = michelson(j as f64 * LAM / 1000.0 / 8.0, 0.0, 0.0, Spectrum::Monochromatic(LAM), 3.0, 32);
            wave_path(&s, &spec(src, det, 1.0, 256, 5.0), 1).unwrap().report.ledger.detected
        })
        .collect();
    let mean = p.iter().sum::<f64>() / 4.0;
    let vis = ((p[0] - p[2]).powi(2) + (p[1] - p[3]).powi(2)).sqrt() / (2.0 * mean);
    measured("Michelson, mirror stepped by λ/8: (P₀ + P₂ − P₁ − P₃) / mean", (p[0] + p[2] - p[1] - p[3]) / mean);
    measured("Michelson, mirror stepped by λ/8: visibility", vis);
    assert!(((p[0] + p[2] - p[1] - p[3]) / mean).abs() < 1e-3, "{p:?}");
    assert!(vis > 0.995, "{vis}");
    // Equal arms through a dielectric splitter: the output is dark.
    assert!(p[0] < 1e-3, "{p:?}");
}

#[test]
fn two_lines_wash_a_michelson_s_fringes_out_where_their_beats_cancel() {
    // Two equal lines are two incoherent modes. Each makes its own fringes,
    // P_l ∝ 1 − cos(2k_l Δ); at the arm difference where 2Δ(1/λ₁ − 1/λ₂) = ½
    // they are in antiphase and the sum stands still as a mirror moves: the
    // visibility of the beat, |cos(2πΔ(1/λ₁ − 1/λ₂))|, is zero there [1]. At no
    // difference both are dark together.
    let (l1, l2) = (LAM, LAM * 1.001);
    let beat = 1.0 / (l1 / 1000.0) - 1.0 / (l2 / 1000.0);
    let walk = 0.25 / beat;
    let run = |dd: f64, j: usize| {
        let (s, src, det) = michelson(j as f64 * l1 / 1000.0 / 8.0, dd, 0.0, Spectrum::Lines(vec![(l1, 1.0), (l2, 1.0)]), 3.0, 32);
        wave_path(&s, &spec(src, det, 1.0, 256, 5.0), 1).unwrap().report.ledger.detected
    };
    let at = |dd: f64| -> f64 {
        let p: Vec<f64> = (0..4).map(|j| run(dd, j)).collect();
        let mean = p.iter().sum::<f64>() / 4.0;
        ((p[0] - p[2]).powi(2) + (p[1] - p[3]).powi(2)).sqrt() / (2.0 * mean)
    };
    let (v0, v_half) = (at(0.0), at(walk));
    measured("two lines in a Michelson: visibility at equal arms", v0);
    measured("two lines in a Michelson: visibility where the beats cancel", v_half);
    assert!(v0 > 0.99, "{v0}");
    assert!(v_half < 2e-2, "{v_half}");
}

/// The singlet's paraxial focal length and principal planes, traced by the
/// f64 tracer with rays 1 µm off the axis: `(efl, x_H, x_H')`.
fn traced_cardinal_points(s: &Scene3, lens_x: f64) -> (f64, f64, f64) {
    use oc_trace::{trace3, LaunchRay3, TraceConfig3};
    let cfg = TraceConfig3 { ghosts: false, polarization: false, ..TraceConfig3::default() };
    let leaf = |origin: Vec3, dir: Vec3| {
        let ray = oc_core::vec3::Ray3::new(origin, dir).unwrap();
        let r = trace3(s, &[LaunchRay3 { ray, lambda_um: LAM, flux: 1.0, source: None, pol: None }], &cfg);
        let last = *r.segments.iter().max_by(|a, b| a.opl0_mm.partial_cmp(&b.opl0_mm).unwrap()).unwrap();
        (last.a, (last.b - last.a).normalize())
    };
    let h = 1e-3;
    let (a, d) = leaf(Vec3::new(lens_x - 20.0, h, 0.0), Vec3::X);
    let x_f2 = a.x - a.y / d.y * d.x;
    let efl = -h * d.x / d.y;
    let (a, d) = leaf(Vec3::new(lens_x + 20.0, h, 0.0), -Vec3::X);
    let x_f1 = a.x - a.y / d.y * d.x;
    (efl, x_f1 + efl, x_f2 - efl)
}

#[test]
fn a_gaussian_through_a_singlet_waists_where_self_predicts() {
    // [14], with the singlet's own focal length and principal planes from the
    // f64 tracer: a real N-BK7 biconvex lens (R ±50 mm, 4 mm thick), not an
    // ideal one, carried by geometric field tracing [17].
    use oc_core::build3::SingletGeom;
    let (w0, lens_x) = (0.2, 100.0);
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_singlet(Vec3::new(lens_x, 0.0, 0.0), SingletGeom::biconvex(), 1.5168, 64.17);
    let mut probe = s.clone();
    probe.sources.clear();
    let (f, x_h, x_h2) = traced_cardinal_points(&probe, lens_x);
    let zr = rayleigh_range_mm(w0, LAM, 1.0);
    let s1 = x_h; // the input waist sits at the source, x = 0
    let s2 = 1.0 / (1.0 / f - 1.0 / (s1 + zr * zr / (s1 - f)));
    let w2 = w0 / ((1.0 - s1 / f).powi(2) + (zr / f).powi(2)).sqrt();
    let zr2 = std::f64::consts::PI * w2 * w2 / (LAM / 1000.0);
    eprintln!("traced f = {f:.6} mm, H at {x_h:.6}, H' at {x_h2:.6}; Self: waist {w2:.6} mm at {:.6}", x_h2 + s2);
    // Radii either side of the predicted waist; a parabola in w² finds the
    // simulated waist.
    let mut rows = Vec::new();
    let mut last_report = None;
    for k in -2..=2 {
        let z = x_h2 + s2 + 0.5 * zr2 * k as f64;
        let mut sc = s.clone();
        let det = sc.add_area_detector(Vec3::new(z, 0.0, 0.0), -Vec3::X, 1.0, 1.0, [16, 16]);
        sc.validate().unwrap();
        let r = wave_path(&sc, &spec(src, det, w0, 512, 2.048), 1).unwrap();
        let w = second_moment_radius_x(&r.field);
        rows.push((z, w * w));
        last_report = Some(r.report);
    }
    let rep = last_report.unwrap();
    assert!(rep.steps.iter().any(|st| st.note.contains("field traced through it")), "{:?}", rep.steps);
    // Least-squares parabola w² = c0 + c1 z + c2 z² about the predicted waist.
    let zc = x_h2 + s2;
    let (mut m, mut v) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
    for (z, y) in &rows {
        let t = z - zc;
        let f = [1.0, t, t * t];
        for i in 0..3 {
            v[i] += f[i] * y;
            for j in 0..3 {
                m[i][j] += f[i] * f[j];
            }
        }
    }
    // Cramer's rule on the 3 × 3 normal equations.
    let det3 = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det3(m);
    let col = |k: usize| {
        let mut a = m;
        for i in 0..3 {
            a[i][k] = v[i];
        }
        det3(a) / d
    };
    let (c0, c1, c2) = (col(0), col(1), col(2));
    let dz = -c1 / (2.0 * c2);
    let w_min = (c0 - c1 * c1 / (4.0 * c2)).sqrt();
    measured("singlet: simulated waist offset from Self's, / Rayleigh range (the gate is on its ratio to s'')", dz / zr2);
    measured("singlet: simulated waist offset from Self's, / s''", dz / s2);
    measured("singlet: simulated waist radius vs Self's, relative", (w_min - w2) / w2);
    assert!((dz / s2).abs() < 1e-3, "waist {dz} mm from Self's s'' = {s2}");
    assert!(((w_min - w2) / w2).abs() < 1e-3, "{w_min} vs {w2}");
    // The faces' reflection booked as light the chain does not follow, not
    // absorbed; the traced maps' resampling measured, and nothing left over.
    let l = rep.ledger;
    measured("singlet: power reflected by the faces", l.other_arm);
    measured("singlet: resampling across the two faces, of the launched power", l.resampling / l.launched);
    assert!(l.other_arm > 0.07 && l.other_arm < 0.10, "two N-BK7 faces reflect about 8%: {l:?}");
    assert_eq!(l.absorbed_element, 0.0, "{l:?}");
    assert!(l.resampling.abs() < 1e-5 * l.launched, "{l:?}");
    assert!(l.numerical.abs() < 1e-10 * l.launched, "{l:?}");
}

#[test]
fn a_glass_plate_shifts_a_gaussian_as_free_space_of_t_over_n() {
    // [13] in ABCD form: a plate of thickness t and index n is, to the beam,
    // free space of length t/n. A Gaussian through a 10 mm N-BK7 plate must
    // spread as one carried 160 − t + t/n mm through air. Transporting it by
    // rays through the whole plate would not (inside its Rayleigh range a
    // beam's curvature does not grow by the distance travelled); the path
    // diffracts it inside the glass and traces only the faces.
    use oc_core::build3::SingletGeom;
    let (w0, t, l) = (0.2, 10.0, 160.0);
    let mut s = Scene3::new();
    let src = source(&mut s);
    let g = SingletGeom { r1_mm: f64::INFINITY, r2_mm: f64::INFINITY, thickness_mm: t, semi_aperture_mm: 12.0, conic1: 0.0, conic2: 0.0 };
    s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
    // Perfectly anti-reflected faces: one pass, no round trips (those are the
    // etalon test's).
    perfect_ar(&mut s);
    let n = s.bodies[0].material.index_at(LAM, &oc_core::material::GlassCatalog::builtin()).unwrap();
    let det = s.add_area_detector(Vec3::new(l, 0.0, 0.0), -Vec3::X, 1.0, 1.0, [16, 16]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, w0, 512, 2.048), 1).unwrap();
    let w = second_moment_radius_x(&r.field);
    let zr = rayleigh_range_mm(w0, LAM, 1.0);
    let want = w0 * (1.0 + ((l - t + t / n) / zr).powi(2)).sqrt();
    let implied_n = t / (zr * ((w / w0).powi(2) - 1.0).sqrt() - l + t);
    measured("plate: radius against free space of t/n, relative", (w - want) / want);
    measured("plate: index implied by the radius, minus the glass's", implied_n - n);
    assert!(((w - want) / want).abs() < 1e-5, "{w} vs {want}");
    // Two faces reflect, the rest arrives; the ledger closes.
    let l = r.report.ledger;
    assert!(l.numerical.abs() < 1e-6 * l.launched, "{l:?}");
}

#[test]
fn a_tilted_plate_displaces_the_beam_where_the_rays_say() {
    // A plate tilted θ displaces a beam by t·sin θ·(1 − cos θ/√(n² − sin² θ))
    // (geometric optics); the field crosses each face obliquely, so the
    // face's transfer has to move it across the beam. The wave's centroid on
    // the detector must land where the f64 chief ray does.
    use oc_core::build3::SingletGeom;
    let (t, theta) = (10.0, 30f64.to_radians());
    let mut s = Scene3::new();
    let src = source(&mut s);
    let g = SingletGeom { r1_mm: f64::INFINITY, r2_mm: f64::INFINITY, thickness_mm: t, semi_aperture_mm: 12.0, conic1: 0.0, conic2: 0.0 };
    let id = s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
    assert!(s.set_object_rotation_z_deg(id.0, theta.to_degrees()));
    let n = s.bodies[0].material.index_at(LAM, &oc_core::material::GlassCatalog::builtin()).unwrap();
    let det = s.add_area_detector(Vec3::new(160.0, 0.0, 0.0), -Vec3::X, 4.0, 4.0, [400, 400]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, 0.3, 512, 3.0), 1).unwrap();
    let seq = oc_trace::extract_sequence3(&s, src, det, 1).unwrap();
    let (nu, nv) = (r.bins[0], r.bins[1]);
    let (du, dv) = (8.0 / nu as f64, 8.0 / nv as f64);
    let (mut s0, mut su, mut sv) = (0.0, 0.0, 0.0);
    for j in 0..nv {
        for i in 0..nu {
            let p = r.power[j * nu + i];
            s0 += p;
            su += p * (-4.0 + (i as f64 + 0.5) * du);
            sv += p * (-4.0 + (j as f64 + 0.5) * dv);
        }
    }
    let (cu, cv) = (su / s0, sv / s0);
    let ray = seq.terminal.uv_mm;
    let shift = t * theta.sin() * (1.0 - theta.cos() / (n * n - theta.sin().powi(2)).sqrt());
    measured("tilted plate: displacement of the chief ray, mm (against the formula)", (ray.x.hypot(ray.y) - shift).abs());
    let e = ((cu - ray.x).powi(2) + (cv - ray.y).powi(2)).sqrt();
    measured("tilted plate: wave centroid vs chief ray, mm", e);
    assert!((ray.x.hypot(ray.y) - shift).abs() < 1e-9, "the rays shift {} vs {shift}", ray.x.hypot(ray.y));
    assert!(e < 0.05 * du, "{cu}, {cv} vs {ray:?}");
    assert!(r.report.steps.iter().any(|st| st.note.contains("plate #") && st.note.contains("round trip")), "{:?}", r.report.steps);
}

#[test]
fn a_singlet_psf_matches_optiland_huygens_psf() {
    // The oracle (verification, not validation; docs/validation.md): Optiland's
    // HuygensPSF of the same singlet, from the clone of
    // github.com/optiland/optiland recorded in the fixture, generated by
    // tools/fixtures/huygens_singlet.py. An equiconvex lens, R ±50 mm, 4 mm,
    // index 1.5168, at f/10, stop at its first surface, at its paraxial focus:
    // Strehl 0.864 from its spherical aberration. Here the canvas carries a
    // uniform beam through an iris at that surface and the real lens (a
    // geometric field trace of each face, diffraction between, [17]) and reads
    // the focus with the zoomed angular spectrum, no paraxial step anywhere.
    use oc_core::build3::SingletGeom;
    use oc_trace::{trace3, LaunchRay3, TraceConfig3};
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/huygens_singlet.json")).unwrap();
    let lam = fx["lambda_um"].as_f64().unwrap();
    let epd = fx["epd_mm"].as_f64().unwrap();
    let image = fx["image_distance_from_back_vertex_mm"].as_f64().unwrap();
    let grid: Vec<f64> = fx["grid_mm"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let psf: Vec<Vec<f64>> = fx["psf_rows_y_then_x"].as_array().unwrap().iter()
        .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect();
    let n_img = grid.len();
    let spacing = grid[1] - grid[0];
    eprintln!("oracle: {}", fx["oracle"]);

    let mut s = Scene3::new();
    let src = s.add_collimated_beam(Vec3::new(97.9, 0.0, 0.0), 3.0, 3.0, 1, Spectrum::Monochromatic(lam));
    let g = SingletGeom { r1_mm: 50.0, r2_mm: -50.0, thickness_mm: 4.0, semi_aperture_mm: 12.0, conic1: 0.0, conic2: 0.0 };
    s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
    s.add_aperture(Vec3::new(97.999, 0.0, 0.0), 0.5 * epd, 0.0);
    s.apertures.last_mut().unwrap().shape.clear_opening = oc_core::scene3::PlanarShape2::Disc { radius_mm: 0.5 * epd };
    let width = n_img as f64 * spacing;
    let det = s.add_area_detector(Vec3::new(102.0 + image, 0.0, 0.0), -Vec3::X, 0.5 * width, 0.5 * width, [n_img, n_img]);
    s.validate().unwrap();
    let mut sp = spec(src, det, 0.3, 2048, 6.0);
    sp.beam = BeamProfile::TopHat { radius_mm: 2.8 };
    // One point a pixel, at its centre: the oracle's values are point values.
    sp.readout = Readout::Diffracted { samples_per_pixel: 1 };
    let r = wave_path(&s, &sp, 1).unwrap();
    for st in &r.report.steps {
        eprintln!("  {}", st.note);
    }

    // The ideal peak for this pupil: P·π·sin²α/λ², α the half-angle the
    // exit pupil subtends at the point read (the paraxial focus): the peak of
    // a perfect spherical wave filling that pupil. Not the marginal ray's own
    // angle, which the spherical aberration tilts by its ray error (here by
    // 0.38%, 0.76% in the peak). Both rays traced: the exit pupil is where a
    // ray through the stop's centre crosses the axis in image space, and the
    // pupil's edge is the marginal ray's height there.
    let mut lens = s.clone();
    lens.sources.clear();
    lens.apertures.clear();
    lens.detectors.clear();
    let last = |o: Vec3, d: Vec3| {
        let ray = oc_core::vec3::Ray3::new(o, d.normalize()).unwrap();
        let tr = trace3(&lens, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }],
            &TraceConfig3 { ghosts: false, polarization: false, ..TraceConfig3::default() });
        let l = *tr.segments.iter().max_by(|a, b| a.opl0_mm.partial_cmp(&b.opl0_mm).unwrap()).unwrap();
        (l.a, (l.b - l.a).normalize())
    };
    let (pc, dc) = last(Vec3::new(97.999, 0.0, 0.0), Vec3::new(1.0, 1e-3, 0.0));
    let x_xp = pc.x - pc.y / dc.y * dc.x;
    let (pm, dm) = last(Vec3::new(90.0, 0.5 * epd, 0.0), Vec3::X);
    let h_xp = (pm.y + (x_xp - pm.x) * dm.y / dm.x).abs();
    let na = h_xp / h_xp.hypot(102.0 + image - x_xp);
    eprintln!("exit pupil {:.6} mm before the back vertex, radius {h_xp:.6} mm; sin α {na:.6}, the marginal ray's sine {:.6}", 102.0 - x_xp, dm.y.abs());
    {
        // The canvas lens's own paraxial focus at this wavelength, against the oracle's.
        let ray = oc_core::vec3::Ray3::new(Vec3::new(90.0, 1e-3, 0.0), Vec3::X).unwrap();
        let tr = trace3(&lens, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }],
            &TraceConfig3 { ghosts: false, polarization: false, ..TraceConfig3::default() });
        let l = tr.segments.iter().max_by(|a, b| a.opl0_mm.partial_cmp(&b.opl0_mm).unwrap()).unwrap();
        let d = (l.b - l.a).normalize();
        let xf = l.a.x - l.a.y / d.y * d.x;
        let n_glass = lens.bodies[0].material.index_at(lam, &oc_core::material::GlassCatalog::builtin()).unwrap();
        eprintln!("canvas: n({lam}) = {n_glass:.8}; paraxial focus {:.6} mm after the back vertex (oracle {image:.6}); EFL {:.6}", xf - 102.0, -1e-3 * d.x / d.y);
    }
    let lam_mm = lam / 1000.0;
    let i0 = r.arriving * std::f64::consts::PI * na * na / (lam_mm * lam_mm);
    let pix = spacing * spacing;
    // The canvas's PSF on the oracle's points, normalized like the oracle's.
    let canvas: Vec<f64> = r.power.iter().map(|p| 100.0 * p / pix / i0).collect();
    let c = n_img / 2;
    let strehl = canvas[c * n_img + c] / 100.0;
    let strehl_oracle = fx["strehl"].as_f64().unwrap();
    let mut sq = 0.0;
    for j in 0..n_img {
        for i in 0..n_img {
            sq += (canvas[j * n_img + i] - psf[j][i]).powi(2);
        }
    }
    let rms = (sq / (n_img * n_img) as f64).sqrt() / 100.0;
    measured("singlet PSF: Strehl, canvas minus Optiland HuygensPSF", strehl - strehl_oracle);
    measured("singlet PSF: rms difference over the image grid, / ideal peak", rms);
    eprintln!("canvas Strehl {strehl:.6}, oracle {strehl_oracle:.6}, NA {na:.6}");
    let (sc, so): (f64, f64) = (canvas.iter().sum(), psf.iter().flatten().sum());
    eprintln!("sums over the grid: canvas {sc:.4} oracle {so:.4} ratio {:.6}", sc / so);
    // Measured: Strehl 0.8653 against 0.8642, rms 1.2e-4 of the ideal peak,
    // the sums over the grid 0.08% apart. What is left is of the order of the
    // canvas's sampling (0.06% finer than 2.9 µm) and of the reference
    // (a uniform spherical cap peaks 0.06% above π·sin²α/λ² at f/10).
    assert!((strehl - strehl_oracle).abs() < 0.01, "{strehl} vs {strehl_oracle}");
    assert!(rms < 1e-2, "{rms}");
}

#[test]
fn a_tilted_detector_reads_the_beam_on_its_own_plane() {
    // The readout of a face tilted 30°: the spectrum rotated onto its plane
    // (Matsushima, Schimmel & Wyrowski 2003 [6]) rather than the field
    // projected along the beam. Each pixel's power must be Kogelnik and
    // Li's irradiance [13] over that pixel of the tilted plane (its mean
    // over an 8 × 8 quadrature), times the obliquity cos 30°, and the face
    // must catch the whole beam.
    let w0 = 0.2;
    let mut s = Scene3::new();
    let src = source(&mut s);
    let th = 30f64.to_radians();
    let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::new(th.cos(), th.sin(), 0.0), 0.6, 0.6, [48, 48]);
    s.validate().unwrap();
    let sp = spec(src, det, w0, 256, 2.048);
    let r = wave_path(&s, &sp, 1).unwrap();
    assert!(r.report.steps.iter().any(|st| st.note.contains("tilted")), "{:?}", r.report.steps);
    assert!(!r.report.warnings.iter().any(|w| matches!(w, SamplingWarning::TiltProjection { .. })));
    let d = s.detectors.iter().find(|x| x.common.id.0 == det.0).unwrap();
    let t = d.common.transform;
    let (du, dv) = (1.2 / 48.0, 1.2 / 48.0);
    let zr = rayleigh_range_mm(w0, LAM, 1.0);
    let irradiance = |u: f64, v: f64| {
        let p = t.transform_point(Vec3::new(u, v, 0.0));
        let w = w0 * (1.0 + (p.x / zr).powi(2)).sqrt();
        2.0 / (std::f64::consts::PI * w * w) * (-2.0 * (p.y * p.y + p.z * p.z) / (w * w)).exp() * th.cos()
    };
    let (mut worst, mut peak, mut on_face): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for j in 0..48 {
        for i in 0..48 {
            let m = 8;
            let mut want = 0.0;
            for b in 0..m {
                for a in 0..m {
                    want += irradiance(-0.6 + (i as f64 + (a as f64 + 0.5) / m as f64) * du, -0.6 + (j as f64 + (b as f64 + 0.5) / m as f64) * dv);
                }
            }
            let want = want / (m * m) as f64;
            let got = r.power[j * 48 + i] / (du * dv);
            peak = peak.max(want);
            on_face += want * du * dv;
            worst = worst.max((got - want).abs());
        }
    }
    // The rotated spectrum is read between its samples (bilinear), which
    // costs a few 1e-3 of the power on a 30° face; its convergence is the
    // propagation tests' business.
    let caught = r.report.ledger.detected / r.report.ledger.launched;
    measured("tilted detector: irradiance against Kogelnik–Li on the tilted plane, worst / peak", worst / peak);
    measured("tilted detector: power caught against Kogelnik–Li's on the face, relative", (caught - on_face) / on_face);
    assert!(worst / peak < 2e-3, "{}", worst / peak);
    assert!(((caught - on_face) / on_face).abs() < 3e-3, "{caught} vs {on_face}");
}

#[test]
fn an_aperture_whose_outer_extent_the_beam_passes_beyond_leaves_it_alone() {
    // As the tracer has an aperture: light inside the opening passes, light
    // between the opening and the outer extent is stopped, light beyond the
    // outer extent passes. A small aperture whose plane the beam crosses 5 mm
    // from its centre, outside its 2 mm outer extent, must leave the field
    // exactly as free space does.
    // The reference is an opening the beam passes through at the same plane,
    // so that both take the same propagation steps.
    let run = |beyond: bool| {
        let mut s = Scene3::new();
        let src = source(&mut s);
        if beyond {
            s.add_aperture(Vec3::new(50.0, 0.0, 5.0), 0.2, 0.0);
            s.apertures.last_mut().unwrap().shape.outer_extent = Some(oc_core::scene3::PlanarShape2::Disc { radius_mm: 2.0 });
        } else {
            s.add_aperture(Vec3::new(50.0, 0.0, 0.0), 10.0, 0.0);
        }
        let det = s.add_area_detector(Vec3::new(100.0, 0.0, 0.0), -Vec3::X, 1.0, 1.0, [32, 32]);
        s.validate().unwrap();
        wave_path(&s, &spec(src, det, 0.2, 256, 2.048), 1).unwrap()
    };
    let (free, past) = (run(false), run(true));
    assert!(past.report.steps.iter().any(|st| st.note.contains("clear of the beam")), "{:?}", past.report.steps);
    assert_eq!(past.report.ledger.absorbed_aperture, 0.0);
    assert_eq!(past.power, free.power, "the field is the one through the open aperture, bit for bit");
}

/// `∫∫` over the ellipse `(y/a)² + (z/b)² ≤ 1` of `f(y, z)`, by the midpoint
/// rule in elliptic polar coordinates.
fn over_ellipse(a: f64, b: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    let (nr, nt) = (800, 1600);
    let mut acc = 0.0;
    for i in 0..nr {
        let rho = (i as f64 + 0.5) / nr as f64;
        for k in 0..nt {
            let t = 2.0 * std::f64::consts::PI * (k as f64 + 0.5) / nt as f64;
            acc += f(a * rho * t.cos(), b * rho * t.sin()) * rho;
        }
    }
    acc * a * b * 2.0 * std::f64::consts::PI / (nr * nt) as f64
}

#[test]
fn a_fold_mirror_smaller_than_the_beam_reflects_only_what_meets_it() {
    // A 45° mirror, a disc of radius 0.6 mm, in a Gaussian of 1 mm radius
    // [13]: it reflects the power over its outline seen along the beam, an
    // ellipse of semi-axes 0.6 cos 45° and 0.6 mm; the rest passes beside it,
    // off the chain.
    let (w0, r_m) = (1.0, 0.6);
    let mut s = Scene3::new();
    let src = source(&mut s);
    let m = SurfaceResponse::Mirror { reflectivity: oc_core::scene3::SpectralScalar::Constant(1.0) };
    s.add_planar_element(Vec3::new(50.0, 0.0, 0.0), r_m, 135.0, m.clone(), m);
    let det = s.add_area_detector(Vec3::new(50.0, 60.0, 0.0), -Vec3::Y, 3.0, 3.0, [64, 64]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, w0, 512, 6.0), 1).unwrap();
    let w = w0 * (1.0 + (50.0 / rayleigh_range_mm(w0, LAM, 1.0)).powi(2)).sqrt();
    let irr = |y: f64, z: f64| 2.0 / (std::f64::consts::PI * w * w) * (-2.0 * (y * y + z * z) / (w * w)).exp();
    let on_mirror = over_ellipse(r_m * std::f64::consts::FRAC_1_SQRT_2, r_m, irr);
    let l = r.report.ledger;
    measured("small fold mirror: power passed beside it, against the Gaussian off its outline", l.passed_by / l.launched - (1.0 - on_mirror));
    assert!((l.passed_by / l.launched - (1.0 - on_mirror)).abs() < 1e-4, "{} vs {}: {l:?}", l.passed_by, 1.0 - on_mirror);
    assert!(l.subcell_projection > 0.0 && l.subcell_projection < 1e-2 * on_mirror, "{l:?}");
    assert!(l.numerical.abs() < 1e-10, "{l:?}");
    assert!(r.report.steps.iter().any(|st| st.note.contains("off the chain")), "{:?}", r.report.steps);
}

#[test]
fn a_thin_lens_smaller_than_the_beam_lets_the_rest_pass_beside_it() {
    // A thin lens, a disc of radius 0.5 mm, in a Gaussian of 1 mm radius:
    // inside its outline the field takes the lens's phase exp(−ikr²/2f),
    // outside it passes unchanged and stays on the chain. Just behind the lens
    // the field must be free space's times the lens's phase inside and times
    // 1 outside, away from the edge's own diffraction.
    let (w0, a, f) = (1.0, 0.5, 100.0);
    let run = |with: bool| {
        let mut s = Scene3::new();
        let src = source(&mut s);
        if with {
            s.add_planar_element(Vec3::new(50.0, 0.0, 0.0), a, 0.0, lens(f), lens(f));
        }
        let det = s.add_area_detector(Vec3::new(50.01, 0.0, 0.0), -Vec3::X, 2.0, 2.0, [32, 32]);
        s.validate().unwrap();
        wave_path(&s, &spec(src, det, w0, 512, 4.0), 1).unwrap()
    };
    let (free, lensed) = (run(false), run(true));
    let g = lensed.field.grid;
    let k = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let (mut worst_in, mut worst_out): (f64, f64) = (0.0, 0.0);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, y) = g.xy(i, j);
            let rr = x.hypot(y);
            let ratio = lensed.field.at(i, j) / free.field.at(i, j);
            if rr < 0.45 {
                let want = oc_wave::Cx::expi(-k * rr * rr / (2.0 * f));
                worst_in = worst_in.max((ratio - want).abs());
            } else if rr > 0.56 && rr < 1.8 {
                worst_out = worst_out.max((ratio - oc_wave::Cx::ONE).abs());
            }
        }
    }
    measured("small thin lens: field inside its outline against the lens's phase, worst", worst_in);
    measured("small thin lens: field outside its outline against free space, worst", worst_out);
    assert!(worst_in < 2e-3 && worst_out < 2e-3, "{worst_in} {worst_out}");
    let l = lensed.report.ledger;
    assert_eq!(l.passed_by, 0.0, "the light beside a lens stays on the chain: {l:?}");
    assert!(lensed.report.steps.iter().any(|st| st.note.contains("on the chain")), "{:?}", lensed.report.steps);
}

#[test]
fn a_lens_rim_smaller_than_the_beam_stops_and_diffracts_as_an_iris_against_it() {
    // A singlet whose clear aperture (1 mm radius) is smaller than the beam
    // (a 2.8 mm top-hat) must act as an iris of the same radius just in front
    // of a lens too large to clip: the same power stopped, and the same focal
    // spot. The rim is laid on the field with exact cell coverage. Resolved
    // only to the ray grid, four samples a ray here, it stops 3.9e-3 more of
    // the power and moves the spot by 6.8e-2 of its peak (measured).
    //
    // The lens is plano-convex, curved side first, so that the beam converges
    // inside the glass and the light the front rim's edge diffracts outward
    // stays mostly inside the 1 mm back face; what it still stops there (9e-4
    // of what reaches it, here 1.1e-4 of the launched power) is real, and the
    // gate on the power allows for it.
    use oc_core::build3::SingletGeom;
    let lam = 0.5876;
    let image = 46.0;
    let run = |rim: f64, iris: Option<f64>| {
        let mut s = Scene3::new();
        let src = s.add_collimated_beam(Vec3::new(97.9, 0.0, 0.0), 3.0, 3.0, 1, Spectrum::Monochromatic(lam));
        let g = SingletGeom { r1_mm: 25.0, r2_mm: f64::INFINITY, thickness_mm: 4.0, semi_aperture_mm: rim, conic1: 0.0, conic2: 0.0 };
        s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
        if let Some(a) = iris {
            s.add_aperture(Vec3::new(97.9985, 0.0, 0.0), a, 0.0);
            s.apertures.last_mut().unwrap().shape.clear_opening = oc_core::scene3::PlanarShape2::Disc { radius_mm: a };
        }
        let det = s.add_area_detector(Vec3::new(102.0 + image, 0.0, 0.0), -Vec3::X, 0.1, 0.1, [64, 64]);
        s.validate().unwrap();
        let mut sp = spec(src, det, 0.3, 1024, 6.0);
        sp.beam = BeamProfile::TopHat { radius_mm: 2.8 };
        sp.readout = Readout::Diffracted { samples_per_pixel: 1 };
        wave_path(&s, &sp, 1).unwrap()
    };
    let (rim, iris) = (run(1.0, None), run(12.0, Some(1.0)));
    assert!(rim.report.steps.iter().any(|st| st.note.contains("its rim laid on the field")), "{:?}", rim.report.steps);
    let (lr, li) = (rim.report.ledger, iris.report.ledger);
    let stopped = |l: oc_wave::WaveLedger| l.absorbed_aperture / l.launched;
    measured("lens rim against an iris: power stopped, difference", stopped(lr) - stopped(li));
    let peak = iris.power.iter().cloned().fold(0.0, f64::max);
    let worst = rim.power.iter().zip(&iris.power).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    measured("lens rim against an iris: focal spot, worst difference / peak", worst / peak);
    assert!((stopped(lr) - stopped(li)).abs() < 3e-4, "{lr:?} vs {li:?}");
    assert!(worst / peak < 2e-3, "{}", worst / peak);
    assert!(lr.numerical.abs() < 1e-10 && lr.resampling.abs() < 1e-4, "{lr:?}");
}

#[test]
fn an_lcos_on_the_path_sends_half_of_unpolarized_light_into_each_beam() {
    // [20] Zhang, You & Chu: an LCoS modulates the field along its director.
    // The wave path carries unpolarized light, so of what the device reflects
    // half is modulated and half is not; the chain follows the modulated beam,
    // steered by a 16-pixel blazed grating, and books the other half. A
    // 16-level staircase puts sinc²(1/16) of the modulated beam into its first
    // order [21] (Swanson), which the detector catches.
    use oc_core::modulator::{LcosSpec, ModulatorKind, Pattern, PatternGenerator, PhaseLut, SpatialModulator};
    let dev = SpatialModulator {
        pixels: [128, 128],
        pitch_um: 8.0,
        kind: ModulatorKind::Lcos(LcosSpec {
            director_deg: 0.0,
            lut: PhaseLut::Linear { levels: 256, max_phase_rad: 2.0 * std::f64::consts::PI },
            design_lambda_um: LAM,
            pixel_fill: 1.0,
            gap_amplitude: 0.0,
            gap_phase_rad: 0.0,
            fringing_um: 0.0,
            reflectance: 1.0,
        }),
        pattern: Pattern::Generated(PatternGenerator::BlazedGrating { period_px: 16.0, angle_deg: 0.0 }),
    };
    let mut s = Scene3::new();
    let src = source(&mut s);
    let pos = Vec3::new(100.0, 0.0, 0.0);
    let id = s.add_modulator(pos, 180.0, dev).unwrap();
    let t = s.surfaces.iter().find(|x| x.common.id.0 == id).unwrap().common.transform;
    let tl = LAM / 1000.0 / (16.0 * 0.008);
    let d_mod = t.rotate(Vec3::new(tl, 0.0, (1.0 - tl * tl).sqrt()));
    // 400 mm on, the unmodulated beam's centre is 2 mm away, off the face.
    let det = s.add_area_detector(pos + d_mod * 400.0, -d_mod, 0.9, 0.9, [32, 32]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, 0.3, 512, 2.56), 1).unwrap();
    let l = r.report.ledger;
    // The device, 1.024 mm square, meets erf(√2·0.512/w)² of the beam.
    let w = 0.3 * (1.0 + (100.0 / rayleigh_range_mm(0.3, LAM, 1.0)).powi(2)).sqrt();
    let on_dev = common::erf(std::f64::consts::SQRT_2 * 0.512 / w).powi(2);
    let eta1 = (std::f64::consts::PI / 16.0).sin().powi(2) / (std::f64::consts::PI / 16.0).powi(2);
    measured("LCoS on the path: unmodulated share against half of what the device meets", l.other_family / l.launched - 0.5 * on_dev);
    measured("LCoS on the path: detected against half the device's share in the first order", l.detected / l.launched - 0.5 * on_dev * eta1);
    assert!((l.other_family / l.launched - 0.5 * on_dev).abs() < 2e-3, "{l:?}");
    assert!((l.detected / l.launched - 0.5 * on_dev * eta1).abs() < 3e-3, "{l:?}");
    assert!(l.numerical.abs() < 1e-10 * l.launched, "{l:?}");
}

#[test]
fn a_glass_plate_delays_the_beam_by_its_optical_path() {
    // The phase the field carries is the whole phase, a traced face's optical
    // path included, so that two arms can be added: through a plate of
    // thickness t and index n the beam's centre must lag free space's by
    // k₀(n − 1)t, less the Gouy phase its shorter effective length (t/n in
    // place of t, [13] in ABCD form) does not accumulate. The window holds
    // the beam to four radii: cut at two, the cut's own diffraction moves the
    // centre's phase by a few milliradians.
    use oc_core::build3::SingletGeom;
    let (w0, t, l) = (0.5, 10.0, 160.0);
    let run = |plate: bool| {
        let mut s = Scene3::new();
        let src = source(&mut s);
        if plate {
            let g = SingletGeom { r1_mm: f64::INFINITY, r2_mm: f64::INFINITY, thickness_mm: t, semi_aperture_mm: 12.0, conic1: 0.0, conic2: 0.0 };
            s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
            perfect_ar(&mut s);
        }
        let det = s.add_area_detector(Vec3::new(l, 0.0, 0.0), -Vec3::X, 1.0, 1.0, [16, 16]);
        s.validate().unwrap();
        (wave_path(&s, &spec(src, det, w0, 512, 4.096), 1).unwrap(), s)
    };
    let ((free, _), (thru, s)) = (run(false), run(true));
    let n = s.bodies[0].material.index_at(LAM, &oc_core::material::GlassCatalog::builtin()).unwrap();
    let zr = rayleigh_range_mm(w0, LAM, 1.0);
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let want = k0 * (n - 1.0) * t - (l - t + t / n).atan2(zr) + l.atan2(zr);
    let c = thru.field.grid.n[0] / 2;
    let got = (thru.field.at(c, c) / free.field.at(c, c)).arg();
    let wrap = |x: f64| (x + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
    measured("plate: phase at the beam's centre against k0(n−1)t less the Gouy change, rad", wrap(got - want));
    assert!(wrap(got - want).abs() < 1e-4, "{got} vs {want} (mod 2π)");
}

#[test]
fn an_lcos_whose_two_polarization_beams_both_reach_the_detector_adds_them_in_power() {
    // [20]: an LCoS's modulated and unmodulated beams are orthogonally
    // polarized, the modulated one along the director; meeting on a detector
    // they add in power [27], whatever phase the device puts on one of them.
    // A uniform device sends both back the same way: the detector reads the
    // same for any uniform level, and each beam carries half of unpolarized
    // light.
    use oc_core::modulator::{LcosSpec, ModulatorKind, Pattern, PhaseLut, SpatialModulator};
    let run = |value: u16| {
        let dev = SpatialModulator {
            pixels: [64, 64],
            pitch_um: 8.0,
            kind: ModulatorKind::Lcos(LcosSpec {
                director_deg: 0.0,
                lut: PhaseLut::Linear { levels: 256, max_phase_rad: 2.0 * std::f64::consts::PI },
                design_lambda_um: LAM,
                pixel_fill: 1.0,
                gap_amplitude: 0.0,
                gap_phase_rad: 0.0,
                fringing_um: 0.0,
                reflectance: 1.0,
            }),
            pattern: Pattern::Uniform { value },
        };
        let mut s = Scene3::new();
        let src = source(&mut s);
        s.add_modulator(Vec3::new(100.0, 0.0, 0.0), 180.0, dev).unwrap();
        let det = s.add_area_detector(Vec3::new(-50.0, 0.0, 0.0), Vec3::X, 1.0, 1.0, [16, 16]);
        s.validate().unwrap();
        wave_path(&s, &spec(src, det, 0.3, 128, 2.0), 1).unwrap()
    };
    let (a, b) = (run(40), run(200));
    let (la, lb) = (a.report.ledger, b.report.ledger);
    measured("LCoS, both beams on one detector: uniform level 40 against 200, detected, relative", la.detected / lb.detected - 1.0);
    assert!((la.detected / lb.detected - 1.0).abs() < 1e-9, "{la:?} vs {lb:?}");
    assert!(a.report.steps.iter().any(|st| st.note.contains("2 arms added")), "{:?}", a.report.steps);
    assert!(a.field_y.is_some(), "two components");
    // Neither beam is lost to the other: both reach the detector. The
    // device, 0.512 mm square, meets erf(√2·0.256/w)² of the beam, and what
    // misses it is absorbed.
    assert!(la.other_family.abs() < 1e-12 * la.launched, "{la:?}");
    let w = 0.3 * (1.0 + (100.0 / rayleigh_range_mm(0.3, LAM, 1.0)).powi(2)).sqrt();
    let on_dev = common::erf(std::f64::consts::SQRT_2 * 0.256 / w).powi(2);
    let met = 1.0 - la.absorbed_element / la.launched;
    measured("LCoS, both beams: what the device meets against erf(√2a/w)²", met - on_dev);
    assert!((met - on_dev).abs() < 1e-3, "{met} vs {on_dev}: {la:?}");
    assert!(la.numerical.abs() < 1e-10, "{la:?}");
}

/// A bare plane-parallel plate of N-BK7, `t` mm thick, at x = 100, its faces
/// tilted `tilt_deg` about Z; a 0.5 mm Gaussian, held by the window to four
/// radii; a detector at x = 200.
fn etalon(t: f64, tilt_deg: f64, with_plate: bool) -> (WavePathResult, f64) {
    use oc_core::build3::SingletGeom;
    let mut s = Scene3::new();
    let src = source(&mut s);
    let mut n = 1.0;
    if with_plate {
        let g = SingletGeom { r1_mm: f64::INFINITY, r2_mm: f64::INFINITY, thickness_mm: t, semi_aperture_mm: 12.0, conic1: 0.0, conic2: 0.0 };
        let id = s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
        if tilt_deg != 0.0 {
            assert!(s.set_object_rotation_z_deg(id.0, tilt_deg));
        }
        n = s.bodies[0].material.index_at(LAM, &oc_core::material::GlassCatalog::builtin()).unwrap();
    }
    let det = s.add_area_detector(Vec3::new(200.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [32, 32]);
    s.validate().unwrap();
    (wave_path(&s, &spec(src, det, 0.5, 256, 4.0), 1).unwrap(), n)
}

/// The closed-form slab (bundle M16): a lossless slab of index `n` in air,
/// met at `theta0`, its round-trip phase `delta`: `(T_s, T_p)`, from
/// `r = (y₀ − y₁)/(y₀ + y₁)` with `y_s = n cos θ`, `y_p = n / cos θ`.
fn slab_t(n: f64, theta0: f64, delta: f64) -> (f64, f64) {
    let c0 = theta0.cos();
    let cg = (1.0 - (theta0.sin() / n).powi(2)).sqrt();
    let t = |r: f64| {
        let r2 = r * r;
        (1.0 - r2).powi(2) / ((1.0 - r2).powi(2) + 4.0 * r2 * delta.sin().powi(2))
    };
    (t((c0 - n * cg) / (c0 + n * cg)), t((1.0 / c0 - n / cg) / (1.0 / c0 + n / cg)))
}

#[test]
fn a_thin_plate_is_an_etalon_transparent_at_seven_pi_and_dimmest_at_seven_and_a_half() {
    // Bundle B05 [1]: a bare slab's round trips sum to
    // t = (1 − r²)e^{iδ}/(1 − r²e^{2iδ}): at δ = 7π it is transparent with
    // t = (−1)⁷, at δ = 7.5π it transmits (1 − r²)²/(1 + r²)². Both against
    // free space across the same planes, whose phase k₀L the plate replaces.
    let (free, _) = etalon(0.0, 0.0, false);
    let n = etalon(1.0, 0.0, true).1;
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    for (m, want_t) in [(7.0, None), (7.5, Some(()))] {
        let l = m * std::f64::consts::PI / (k0 * n);
        let (r, _) = etalon(l, 0.0, true);
        let (ts, _) = slab_t(n, 0.0, m * std::f64::consts::PI);
        let got = r.report.ledger.detected / free.report.ledger.detected;
        measured(&format!("etalon at δ = {m}π, normal incidence: transmission against the closed form"), got - ts);
        assert!((got - ts).abs() < 1e-6, "{got} vs {ts}");
        if want_t.is_none() {
            let c = r.field.grid.n[0] / 2;
            let ratio = r.field.at(c, c) / free.field.at(c, c);
            let want = -oc_wave::Cx::expi(-k0 * l);
            measured("etalon at δ = 7π: the field at the centre against −e^{−ik₀L} times free space's", (ratio - want).abs());
            assert!((ratio - want).abs() < 1e-6, "{ratio:?} vs {want:?}");
        }
        assert!(r.report.steps.iter().any(|st| st.note.contains("every round trip")), "{:?}", r.report.steps);
    }
}

#[test]
fn an_oblique_etalon_transmits_s_and_p_as_their_own_closed_forms() {
    // Bundle B05 at 31°: s and p have their own interface reflections, so
    // their transmissions differ at δ = 7.5π; unpolarized light, the two
    // modes half each, transmits their mean.
    let (free, _) = etalon(0.0, 0.0, false);
    let n = etalon(1.0, 0.0, true).1;
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let th = 31f64.to_radians();
    let l = 7.5 * std::f64::consts::PI / (k0 * (n * n - th.sin().powi(2)).sqrt());
    let (r, _) = etalon(l, 31.0, true);
    let (ts, tp) = slab_t(n, th, 7.5 * std::f64::consts::PI);
    let got = r.report.ledger.detected / free.report.ledger.detected;
    measured("etalon at 31°, δ = 7.5π: unpolarized transmission against (T_s + T_p)/2", got - 0.5 * (ts + tp));
    assert!((ts - tp).abs() > 1e-3, "the test needs s and p apart: {ts} {tp}");
    assert!((got - 0.5 * (ts + tp)).abs() < 1e-5, "{got} vs {}", 0.5 * (ts + tp));
    assert!(r.report.approximated.iter().any(|a| a.contains("two mutually incoherent modes")), "{:?}", r.report.approximated);
}

#[test]
fn a_millimetre_plate_s_transmission_follows_its_round_trip_phase() {
    // The same closed form at δ = k₀nL ≈ 15 000 rad: the round trips are
    // summed at the field's own phase, not wrapped or lost.
    let (free, _) = etalon(0.0, 0.0, false);
    let (r, n) = etalon(1.0, 0.0, true);
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let (ts, _) = slab_t(n, 0.0, k0 * n * 1.0);
    let got = r.report.ledger.detected / free.report.ledger.detected;
    measured("1 mm plate: transmission against the closed form at its round-trip phase", got - ts);
    assert!((got - ts).abs() < 1e-4, "{got} vs {ts}");
}

/// A single quarter-wave layer of MgF₂ (1.38, quarter-wave at normal
/// incidence) on a crown-glass plate (1.52, bare back face): the front and the
/// back response of a flat surface.
fn quarter_wave_on_a_plate() -> (SurfaceResponse, SurfaceResponse) {
    use oc_core::thinfilm::{Coating, CoatingBacking, ExitFace, FilmLayer, FilmMaterial, Incidence};
    let (n1, ns) = (1.38, 1.52);
    let layer = FilmLayer::new(FilmMaterial::constant("film", n1), LAM * 1000.0 / (4.0 * n1));
    let c = Coating {
        layers: vec![layer],
        backing: CoatingBacking::Plate { substrate: FilmMaterial::constant("glass", ns), exit: ExitFace::Bare, thickness_mm: 0.0 },
        incidence: Incidence::Film,
        design: None,
    };
    (SurfaceResponse::Coating { coating: c.clone() }, SurfaceResponse::Coating { coating: c.seen_from_behind() })
}

/// That coating on a flat surface tilted `tilt_deg` in a 0.5 mm Gaussian; the
/// detector catches `reflect`ed or transmitted light.
fn coated(tilt_deg: f64, reflect: bool) -> WavePathResult {
    let (front, back) = quarter_wave_on_a_plate();
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_planar_element(Vec3::new(50.0, 0.0, 0.0), 10.0, tilt_deg, front, back);
    let det = if reflect {
        s.add_area_detector(Vec3::new(50.0, 60.0, 0.0), -Vec3::Y, 3.0, 3.0, [32, 32])
    } else {
        s.add_area_detector(Vec3::new(150.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [32, 32])
    };
    s.validate().unwrap();
    wave_path(&s, &spec(src, det, 0.5, 256, 4.0), 1).unwrap()
}

#[test]
fn a_coated_surface_reflects_and_transmits_as_its_single_layer_s_closed_form() {
    // A single layer: r = (r₀₁ + r₁₂e^{2iβ})/(1 + r₀₁r₁₂e^{2iβ}), β = k₀n₁d cos θ₁,
    // each interface's Fresnel amplitude for its polarization (Born & Wolf
    // [16], the theory of stratified media), computed here on its own, apart
    // from the canvas's transfer matrix. At 45° reflection, unpolarized light
    // reflects (R_s + R_p)/2; at normal incidence the plate transmits
    // (1 − R_front)(1 − R_back), its two faces in series, as the coating
    // declares its plate incoherent.
    let (n0, n1, ns) = (1.0f64, 1.38f64, 1.52f64);
    let lam = LAM / 1000.0;
    let d = lam / (4.0 * n1);
    let single = |theta0: f64, p: bool| -> f64 {
        let s0 = theta0.sin();
        let (c0, c1, cs) = (theta0.cos(), (1.0 - (s0 / n1).powi(2)).sqrt(), (1.0 - (s0 / ns).powi(2)).sqrt());
        let fr = |na: f64, ca: f64, nb: f64, cb: f64| if p { (nb * ca - na * cb) / (nb * ca + na * cb) } else { (na * ca - nb * cb) / (na * ca + nb * cb) };
        let (r01, r12) = (fr(n0, c0, n1, c1), fr(n1, c1, ns, cs));
        let e = oc_wave::Cx::expi(2.0 * 2.0 * std::f64::consts::PI / lam * n1 * d * c1);
        let r = (oc_wave::Cx::real(r01) + e * r12) / (oc_wave::Cx::ONE + e * (r01 * r12));
        r.norm_sqr()
    };
    let launched_through = |r: &WavePathResult| r.report.ledger.detected / r.report.ledger.launched;
    let refl = coated(135.0, true);
    let want_r = 0.5 * (single(std::f64::consts::FRAC_PI_4, false) + single(std::f64::consts::FRAC_PI_4, true));
    measured("single-layer coating at 45°: reflected power against (R_s + R_p)/2", launched_through(&refl) - want_r);
    assert!((launched_through(&refl) - want_r).abs() < 1e-6, "{} vs {want_r}", launched_through(&refl));
    let thru = coated(180.0, false);
    let rf = single(0.0, false);
    let rb = ((ns - 1.0) / (ns + 1.0)).powi(2);
    let want_t = (1.0 - rf) * (1.0 - rb);
    measured("single-layer coating on a plate, normal incidence: transmitted against (1 − R_f)(1 − R_b)", launched_through(&thru) - want_t);
    assert!((launched_through(&thru) - want_t).abs() < 1e-6, "{} vs {want_t}", launched_through(&thru));
    assert!(rf < 0.02 && rf > 0.01, "a quarter-wave MgF₂ on crown glass reflects 1.3%: {rf}");
}

/// The single layer's reflectance at `theta0`, for s or `p`: Born & Wolf
/// [16], r = (r₀₁ + r₁₂e^{2iβ})/(1 + r₀₁r₁₂e^{2iβ}), β = k₀n₁d cos θ₁, apart
/// from the canvas's transfer matrix; the layer of `quarter_wave_on_a_plate`.
fn single_layer_r(theta0: f64, p: bool) -> f64 {
    let (n0, n1, ns) = (1.0f64, 1.38f64, 1.52f64);
    let lam = LAM / 1000.0;
    let d = lam / (4.0 * n1);
    let s0 = theta0.sin();
    let (c0, c1, cs) = (theta0.cos(), (1.0 - (s0 / n1).powi(2)).sqrt(), (1.0 - (s0 / ns).powi(2)).sqrt());
    let fr = |na: f64, ca: f64, nb: f64, cb: f64| if p { (nb * ca - na * cb) / (nb * ca + na * cb) } else { (na * ca - nb * cb) / (na * ca + nb * cb) };
    let (r01, r12) = (fr(n0, c0, n1, c1), fr(n1, c1, ns, cs));
    let e = oc_wave::Cx::expi(2.0 * 2.0 * std::f64::consts::PI / lam * n1 * d * c1);
    let r = (oc_wave::Cx::real(r01) + e * r12) / (oc_wave::Cx::ONE + e * (r01 * r12));
    r.norm_sqr()
}

#[test]
fn coatings_whose_planes_of_incidence_differ_reflect_s_of_one_as_p_of_the_other() {
    // A coating that folds the beam in the XY plane, then one that folds it
    // out of it, both at 45°: the first's s (along Z) lies in the second's
    // plane of incidence, so it is the second's p. Unpolarized light reflects
    // (R₁ₛR₂ₚ + R₁ₚR₂ₛ)/2 [26], not the (R₁ₛR₂ₛ + R₁ₚR₂ₚ)/2 of a shared
    // plane; the field carries both components through both.
    let (front, back) = quarter_wave_on_a_plate();
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_planar_element(Vec3::new(50.0, 0.0, 0.0), 10.0, 135.0, front.clone(), back.clone());
    s.add_planar_element(Vec3::new(50.0, 50.0, 0.0), 10.0, 0.0, front, back);
    let n = Vec3::new(0.0, -1.0, 1.0).normalize();
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, n);
    s.surfaces.last_mut().unwrap().common.transform = oc_core::Isometry3::new(Vec3::new(50.0, 50.0, 0.0), [q.x, q.y, q.z, q.w]).unwrap();
    let det = s.add_area_detector(Vec3::new(50.0, 50.0, 60.0), -Vec3::Z, 3.0, 3.0, [16, 16]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, 0.5, 128, 4.0), 1).unwrap();
    let got = r.report.ledger.detected / r.report.ledger.launched;
    let (rs, rp) = (single_layer_r(std::f64::consts::FRAC_PI_4, false), single_layer_r(std::f64::consts::FRAC_PI_4, true));
    let want = rs * rp;
    let shared = 0.5 * (rs * rs + rp * rp);
    measured("two coatings in crossed planes: reflected power against (R1s R2p + R1p R2s)/2, relative", got / want - 1.0);
    assert!((shared / want - 1.0).abs() > 0.1, "the test needs the planes to matter: {shared} vs {want}");
    assert!((got / want - 1.0).abs() < 1e-5, "{got} vs {want}");
    assert!(r.field_y.is_some());
}

#[test]
fn a_pinhole_at_a_lens_s_focus_passes_what_the_focused_gaussian_puts_through_it() {
    // [13] and bundle M30: a Gaussian whose waist w₀ sits on a lens of focal
    // length f has radius w(f) = λf/(πw₀) at the focal plane, so a pinhole of
    // that radius there stops exp(−2) of it. The focus, 20 µm here, is under
    // two samples of the 6 mm window the beam arrives on (which holds the
    // beam to three radii); the path carries the field onto a window fitted
    // to the focus first, then back out to the declared one.
    let (w0, f) = (1.0, 100.0);
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_planar_element(Vec3::new(50.0, 0.0, 0.0), 10.0, 0.0, lens(f), lens(f));
    let wf = LAM / 1000.0 * f / (std::f64::consts::PI * w0);
    s.add_aperture(Vec3::new(50.0 + f, 0.0, 0.0), 0.2, 0.0);
    s.apertures.last_mut().unwrap().shape.clear_opening = oc_core::scene3::PlanarShape2::Disc { radius_mm: wf };
    let det = s.add_area_detector(Vec3::new(250.0, 0.0, 0.0), -Vec3::X, 2.0, 2.0, [32, 32]);
    s.validate().unwrap();
    let mut sp = spec(src, det, w0, 512, 6.0);
    sp.beam = BeamProfile::Gaussian { waist_radius_mm: w0, waist_offset_mm: 50.0 };
    let r = wave_path(&s, &sp, 1).unwrap();
    let l = r.report.ledger;
    let stopped = l.absorbed_aperture / l.launched;
    let want = (-2.0f64).exp();
    measured("pinhole of radius λf/πw₀ at a lens's focus: power stopped against exp(−2)", stopped - want);
    assert!((stopped - want).abs() < 1e-5, "{stopped} vs {want}");
    assert!(r.report.steps.iter().any(|st| st.note.contains("onto a new window")), "{:?}", r.report.steps.iter().map(|s| &s.note).collect::<Vec<_>>());
    assert!(l.numerical.abs() < 1e-10, "{l:?}");
}

#[test]
fn a_ten_times_beam_expander_widens_the_beam_as_the_abcd_law_says() {
    // [13] in ABCD form: a waist w₀ on a lens f₁, a lens f₂ at f₁ + f₂, the
    // beam's radius at the second lens from the q parameter carried through:
    // 1/q₁ = 1/q₀ − 1/f₁, q₂ = q₁ + f₁ + f₂, w² = −λ/(π Im(1/q₂)). The beam
    // goes from 0.2 mm through a 50 µm focus to 2 mm: the path narrows its
    // window onto the first lens's beam, then widens it again.
    let (w0, f1, f2) = (0.2, 50.0, 500.0);
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_planar_element(Vec3::new(50.0, 0.0, 0.0), 10.0, 0.0, lens(f1), lens(f1));
    s.add_planar_element(Vec3::new(50.0 + f1 + f2, 0.0, 0.0), 20.0, 0.0, lens(f2), lens(f2));
    let det = s.add_area_detector(Vec3::new(50.0 + f1 + f2 + 0.01, 0.0, 0.0), -Vec3::X, 4.0, 4.0, [32, 32]);
    s.validate().unwrap();
    // A window that holds the 2 mm beam to three radii.
    let mut sp = spec(src, det, w0, 512, 12.0);
    sp.beam = BeamProfile::Gaussian { waist_radius_mm: w0, waist_offset_mm: 50.0 };
    let r = wave_path(&s, &sp, 1).unwrap();
    let lam = LAM / 1000.0;
    let q0 = oc_wave::Cx::new(0.0, std::f64::consts::PI * w0 * w0 / lam);
    let q1 = oc_wave::Cx::ONE / (oc_wave::Cx::ONE / q0 - oc_wave::Cx::real(1.0 / f1));
    let q2 = q1 + oc_wave::Cx::real(f1 + f2);
    let want = (-lam / (std::f64::consts::PI * (oc_wave::Cx::ONE / q2).im)).sqrt();
    let w = second_moment_radius_x(&r.field);
    measured("10× expander: radius at the second lens against the ABCD law, relative", w / want - 1.0);
    assert!((w / want - 1.0).abs() < 1e-5, "{w} vs {want}");
    assert!(want / w0 > 9.5 && want / w0 < 11.0, "an expander of about ten: {}", want / w0);
    let narrowed = r.report.steps.iter().filter(|st| st.note.contains("onto a new window")).count();
    assert!(narrowed >= 2, "narrowed onto the small beam and widened again: {narrowed}");
    let l = r.report.ledger;
    assert!(l.detected / l.launched > 0.999, "{l:?}");
}

/// Elements in a row along +X from a source at the origin, each square to the
/// beam, and a detector at x = 200: what of a 0.5 mm Gaussian, launched
/// unpolarized, reaches it.
fn through(elements: &[(f64, SurfaceResponse)]) -> f64 {
    let mut s = Scene3::new();
    let src = source(&mut s);
    for (x, r) in elements {
        s.add_planar_element(Vec3::new(*x, 0.0, 0.0), 10.0, 0.0, r.clone(), r.clone());
    }
    let det = s.add_area_detector(Vec3::new(200.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [16, 16]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, 0.5, 128, 4.0), 1).unwrap();
    assert!(r.report.ledger.numerical.abs() < 1e-10, "{:?}", r.report.ledger);
    r.report.ledger.detected / r.report.ledger.launched
}

fn polarizer(axis_deg: f64, extinction: f64) -> SurfaceResponse {
    SurfaceResponse::Polarizer { axis_deg, extinction: oc_core::scene3::AngularScalar::Constant(extinction), substrate_index: 1.0 }
}

fn retarder(axis_deg: f64, retardance_deg: f64) -> SurfaceResponse {
    SurfaceResponse::Retarder { axis_deg, retardance_deg, substrate_index: 1.0, crystal: None }
}

#[test]
fn two_polarizers_pass_unpolarized_light_as_malus_s_law_says() {
    // [27] Malus: of unpolarized light the first polarizer passes half, the
    // second cos²θ of that. Crossed sheets that leak ε each pass ε exactly
    // [26]: each of the two modes is blocked once.
    for th in [0.0f64, 30.0, 60.0, 90.0] {
        let got = through(&[(50.0, polarizer(0.0, 0.0)), (100.0, polarizer(th, 0.0))]);
        let want = 0.5 * th.to_radians().cos().powi(2);
        measured(&format!("Malus's law at {th}°: transmitted against cos²θ/2"), got - want);
        assert!((got - want).abs() < 1e-9, "{th}: {got} vs {want}");
    }
    let eps = 1e-4;
    let got = through(&[(50.0, polarizer(0.0, eps)), (100.0, polarizer(90.0, eps))]);
    measured("crossed polarizers of extinction 1e-4: transmitted against ε, relative", got / eps - 1.0);
    assert!((got / eps - 1.0).abs() < 1e-6, "{got}");
}

#[test]
fn a_retarder_between_crossed_polarizers_passes_half_sin2_2phi_sin2_half_delta() {
    // [26], [27]: a retarder of retardance δ, its fast axis φ from the first
    // of two crossed polarizers, passes sin²2φ·sin²(δ/2) of what the first
    // polarizer passes, half of unpolarized light. A quarter-wave plate at
    // 45° makes circular light, which any analyzer halves.
    for (phi, delta) in [(45.0f64, 180.0f64), (22.5, 180.0), (45.0, 90.0), (30.0, 60.0), (10.0, 120.0)] {
        let got = through(&[(50.0, polarizer(0.0, 0.0)), (75.0, retarder(phi, delta)), (100.0, polarizer(90.0, 0.0))]);
        let want = 0.5 * (2.0 * phi.to_radians()).sin().powi(2) * (0.5 * delta.to_radians()).sin().powi(2);
        measured(&format!("retarder φ = {phi}°, δ = {delta}° between crossed polarizers: against sin²2φ sin²(δ/2)/2"), got - want);
        assert!((got - want).abs() < 1e-9, "{phi} {delta}: {got} vs {want}");
    }
    for th in [0.0, 45.0, 90.0, 120.0] {
        let got = through(&[(50.0, polarizer(0.0, 0.0)), (75.0, retarder(45.0, 90.0)), (100.0, polarizer(th, 0.0))]);
        measured(&format!("circular light through an analyzer at {th}°: against 1/4"), got - 0.25);
        assert!((got - 0.25).abs() < 1e-9, "{th}: {got}");
    }
}

#[test]
fn tilted_crossed_polarizers_leak_as_the_ray_tracer_says() {
    // Crossed polarizers at ±45° to a plane of incidence, both met at 40°
    // (the plane's normal, world Z, lies at 40° in these elements' local
    // frame, so their axes are at 85° and 175°):
    // their axes, projected across the beam, are no longer at right angles,
    // so the pair leaks, which is why a display has a viewing angle. The
    // wave path applies the tracer's Jones matrix at the chief ray, the
    // element's faces in glass of 1.5 included in the second case: a check
    // of one model in two implementations, not of the physics. Bare, the
    // axes projected across the ray meet at cos γ = sin²θ/(1 + cos²θ), so
    // the pair passes (cos γ)²/2 of unpolarized light; in glass the ray is
    // straighter and the pair leaks less.
    use oc_trace::{trace3, LaunchRay3, TraceConfig3};
    let mut leaks = Vec::new();
    for n_sub in [1.0, 1.5] {
        let mut s = Scene3::new();
        let src = source(&mut s);
        for (x, axis) in [(50.0, 85.0), (100.0, 175.0)] {
            let r = SurfaceResponse::Polarizer { axis_deg: axis, extinction: oc_core::scene3::AngularScalar::Constant(0.0), substrate_index: n_sub };
            s.add_planar_element(Vec3::new(x, 0.0, 0.0), 10.0, 40.0, r.clone(), r);
        }
        let det = s.add_area_detector(Vec3::new(200.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [16, 16]);
        s.validate().unwrap();
        let r = wave_path(&s, &spec(src, det, 0.5, 128, 4.0), 1).unwrap();
        let wave = r.report.ledger.detected / r.report.ledger.launched;
        let launch = [LaunchRay3 { ray: oc_core::vec3::Ray3::new(Vec3::ZERO, Vec3::X).unwrap(), lambda_um: LAM, flux: 1.0, source: None, pol: None }];
        let res = trace3(&s, &launch, &TraceConfig3::default());
        let rays = res.detectors.iter().find(|d| d.detector == det).unwrap().total_incident;
        measured(&format!("tilted crossed polarizers in glass of {n_sub}: wave against the ray tracer, relative"), wave / rays - 1.0);
        assert!((wave / rays - 1.0).abs() < 1e-6, "{wave} vs {rays}");
        assert!(r.report.ledger.numerical.abs() < 1e-10, "{:?}", r.report.ledger);
        leaks.push(wave);
    }
    let th = 40f64.to_radians();
    let want = 0.5 * (th.sin().powi(2) / (1.0 + th.cos().powi(2))).powi(2);
    measured("tilted crossed polarizers, bare: leak against (sin²θ/(1 + cos²θ))²/2, relative", leaks[0] / want - 1.0);
    assert!((leaks[0] / want - 1.0).abs() < 1e-9, "{} vs {want}", leaks[0]);
    assert!(leaks[1] > 1e-3 && leaks[1] < 0.2 * leaks[0], "{leaks:?}");
}

#[test]
fn a_half_wave_plate_in_one_arm_of_a_mach_zehnder_takes_its_fringes_away_and_a_polarizer_brings_them_back() {
    // [27] Fresnel–Arago: orthogonally polarized beams do not interfere. A
    // half-wave plate in one arm of a Mach–Zehnder, its fast axis at θ, turns
    // each polarization into its mirror image about θ; unpolarized light then
    // adds in power at the output, half the launched in the X port at every
    // path difference. An analyzer at β behind the port brings the fringes
    // back with visibility cos 2(θ − β) [26]: the port passes
    // 1/4 − cos Δφ·cos 2(θ − β)/4, where without the plate it would carry
    // (1 − cos Δφ)/2.
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let theta = 20.0f64;
    for j in 0..4 {
        let dphi = j as f64 * std::f64::consts::FRAC_PI_2;
        let delta = dphi / (k0 * std::f64::consts::SQRT_2);
        for analyzer in [None, Some(theta), Some(theta + 45.0), Some(theta + 30.0)] {
            let (mut s, src, dx, _) = mach_zehnder(delta);
            let hwp = retarder(theta, 180.0);
            s.add_planar_element(Vec3::new(100.0, 0.0, 0.0), 10.0, 0.0, hwp.clone(), hwp);
            if let Some(beta) = analyzer {
                let a = polarizer(beta, 0.0);
                s.add_planar_element(Vec3::new(200.0, 100.0, 0.0), 10.0, 0.0, a.clone(), a);
            }
            s.validate().unwrap();
            let r = wave_path(&s, &spec(src, dx, 0.5, 256, 4.0), 1).unwrap();
            let got = r.report.ledger.detected / r.report.ledger.launched;
            let want = match analyzer {
                None => 0.5,
                Some(beta) => 0.25 - 0.25 * dphi.cos() * (2.0 * (theta - beta).to_radians()).cos(),
            };
            measured(&format!("Mach–Zehnder, half-wave plate in one arm, Δφ = {j}·π/2, analyzer {analyzer:?}: X port against the Jones calculus"), got - want);
            assert!((got - want).abs() < 1e-6, "{j} {analyzer:?}: {got} vs {want}");
            assert!(r.report.steps.iter().any(|st| st.note.contains("arm 2 joins arm 1")));
            assert!(!r.report.approximated.iter().any(|a| a.contains("scalar field")), "{:?}", r.report.approximated);
        }
    }
}

#[test]
fn crossed_polarizers_before_a_lens_leave_it_a_dark_field_to_carry() {
    // The chain is geometry: crossed polarizers pass nothing, and the lens
    // and the detector behind them still have a field to carry, all zeros.
    use oc_core::build3::SingletGeom;
    let mut s = Scene3::new();
    let src = source(&mut s);
    s.add_planar_element(Vec3::new(20.0, 0.0, 0.0), 10.0, 0.0, polarizer(0.0, 0.0), polarizer(0.0, 0.0));
    s.add_planar_element(Vec3::new(30.0, 0.0, 0.0), 10.0, 0.0, polarizer(90.0, 0.0), polarizer(90.0, 0.0));
    let g = SingletGeom { r1_mm: 50.0, r2_mm: -50.0, thickness_mm: 4.0, semi_aperture_mm: 10.0, conic1: 0.0, conic2: 0.0 };
    s.add_singlet(Vec3::new(50.0, 0.0, 0.0), g, 1.5168, 64.17);
    let det = s.add_area_detector(Vec3::new(120.0, 0.0, 0.0), -Vec3::X, 3.0, 3.0, [16, 16]);
    s.validate().unwrap();
    let r = wave_path(&s, &spec(src, det, 0.5, 128, 4.0), 1).unwrap();
    let l = r.report.ledger;
    assert!(l.detected < 1e-30, "{l:?}");
    assert!([l.numerical, l.resampling, l.left_window, l.other_ports].iter().all(|v| v.is_finite() && v.abs() < 1e-12), "{l:?}");
    assert!(r.field.power() + r.field_y.as_ref().unwrap().power() < 1e-30);
    assert!((l.absorbed_element - l.launched).abs() < 1e-12, "{l:?}");
}

#[test]
fn a_dmd_s_orders_cross_a_singlet_each_about_its_own_direction() {
    // [10] and [17]: the DMD of the test above, its orders λ/p = 63 mrad
    // apart and its blaze between two of them, through an N-BK7 singlet
    // (R ±8 mm, 2 mm thick) at twice its focal length; the orders overlap on
    // the lens, so a field of several directions crosses each face. The
    // chief ray leaves along the blaze, which no order follows: traced about
    // it, the brightest order would depart from its reference by its angle to
    // it, and gather the piston k₀θ²L/2 per slab that the plane-face test of
    // `transfer` measures. The path carries the orders apart, so each crosses
    // the faces about its own direction, departing from it only by its own
    // spread.
    use oc_core::modulator::{facet_normal, DmdSpec, DmdState, ModulatorKind, Pattern, SpatialModulator, TiltAxis};
    let p_um = 10.0;
    let mut s = Scene3::new();
    let src = source(&mut s);
    let mk = |value: u16| SpatialModulator {
        pixels: [401, 401],
        pitch_um: p_um,
        kind: ModulatorKind::Dmd(DmdSpec { tilt_deg: 12.0, tilt_axis: TiltAxis::Side, mirror_fill: 0.95, reflectance: 1.0, window_transmittance: 1.0 }),
        pattern: Pattern::Uniform { value },
    };
    let dmd_pos = Vec3::new(100.0, 0.0, 0.0);
    let id = s.add_modulator(dmd_pos, 180.0 - 24.0, mk(1)).unwrap();
    let t = s.surfaces.iter().find(|x| x.common.id.0 == id).unwrap().common.transform;
    let n_dev = t.rotate(Vec3::Z);
    let bis = (n_dev - Vec3::X).normalize();
    let ModulatorKind::Dmd(spec_d) = &mk(1).kind else { unreachable!() };
    let on = t.rotate(facet_normal(spec_d, DmdState::On));
    let value = if on.dot(bis) > t.rotate(facet_normal(spec_d, DmdState::Off)).dot(bis) { 1 } else { 0 };
    if let SurfaceResponse::Modulator { device, .. } = &mut s.surfaces.iter_mut().find(|x| x.common.id.0 == id).unwrap().response_front {
        device.pattern = Pattern::Uniform { value };
    }
    let lens_pos = dmd_pos + n_dev * 16.2;
    let g = oc_core::build3::SingletGeom { r1_mm: 8.0, r2_mm: -8.0, thickness_mm: 2.0, semi_aperture_mm: 2.5, conic1: 0.0, conic2: 0.0 };
    s.add_singlet(lens_pos, g, 1.5168, 64.17);
    let q = glam::DQuat::from_rotation_arc(Vec3::X, n_dev);
    s.bodies.last_mut().unwrap().common.transform = oc_core::Isometry3::new(lens_pos, [q.x, q.y, q.z, q.w]).unwrap();
    perfect_ar(&mut s);
    let det = s.add_area_detector(lens_pos + n_dev * 16.2, -n_dev, 1.5, 1.5, [16, 16]);
    s.validate().unwrap();
    // A window of 5.12 mm at 1024 samples holds directions to 63 mrad: the
    // two orders either side of the blaze, the others booked beyond its band.
    let mut sp = spec(src, det, 0.1, 1024, 5.12);
    sp.beam = BeamProfile::Gaussian { waist_radius_mm: 0.1, waist_offset_mm: 100.0 };
    sp.device_samples_per_pixel = 4;
    let r = wave_path(&s, &sp, 1).unwrap();
    let apart = r.report.approximated.iter().find(|a| a.contains("of its orders apart")).cloned();
    assert!(apart.is_some(), "{:?}", r.report.approximated);
    // The orders nearest the blaze, and their angle to it.
    let spec_dir = Vec3::X - n_dev * (2.0 * Vec3::X.dot(n_dev));
    let (ex, ey) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
    let step = LAM / 1000.0 / (p_um / 1000.0);
    let nearest = (-12i64..=12)
        .filter_map(|m| {
            let (a, b) = (spec_dir.dot(ex) + m as f64 * step, spec_dir.dot(ey));
            let z2 = 1.0 - a * a - b * b;
            (z2 > 0.0).then(|| (ex * a + ey * b + n_dev * z2.sqrt()).dot(n_dev).clamp(-1.0, 1.0).acos())
        })
        .fold(f64::INFINITY, f64::min);
    // Each face's report: how far the field's directions depart from the
    // references they were traced along, power-weighted over the orders. A
    // weak order at the window's band edge turns its phase by nearly π a
    // sample and departs by its fold: the power-weighted rms, which counts
    // the field down to 1e-4 of its peak, reads it (3.3 mrad on the first
    // face; the difference over two samples, used before R08, folded that
    // step to a small one and read 0.36), and so does the spectral metric
    // (3.8 mrad), while where the field holds a hundredth of its peak the
    // orders follow their own references to 0.3 mrad.
    let read = |key: &str| -> Vec<f64> {
        r.report
            .approximated
            .iter()
            .filter(|a| a.contains("geometric field tracing"))
            .filter_map(|a| {
                let at = a.find(key)? + key.len();
                a[at..].split(' ').next()?.parse::<f64>().ok()
            })
            .collect()
    };
    let (worst, rms, spectral) = (read("depart from it by up to "), read("(rms "), read("spreads "));
    let rms_max = rms.iter().cloned().fold(0.0, f64::max);
    let worst_max = worst.iter().cloned().fold(0.0, f64::max);
    let spectral_max = spectral.iter().cloned().fold(0.0, f64::max);
    measured("DMD orders through a singlet: the nearest order's angle to the chief ray, mrad", 1e3 * nearest);
    measured("DMD orders through a singlet: the orders' rms departure from their own references, the larger face's, mrad", rms_max);
    measured("DMD orders through a singlet: the steepest departure, the larger face's, mrad", worst_max);
    measured("DMD orders through a singlet: the spectral departure, the larger face's, mrad", spectral_max);
    assert!(nearest > 0.02, "the blaze must lie between orders: {nearest}");
    assert_eq!(rms.len(), 2, "one note per face: {:?}", r.report.approximated);
    assert_eq!(spectral.len(), 2, "one note per face: {:?}", r.report.approximated);
    assert!(worst_max < 0.02 * 1e3 * nearest, "{worst:?} against {nearest}");
    assert!(rms_max < 0.2 * 1e3 * nearest && spectral_max < 0.2 * 1e3 * nearest, "{rms:?}, {spectral:?} against {nearest}");
    // The orders, carried apart, each through an operator of its own, add
    // back with a cross term their ledgers do not hold; the DMD's take-up
    // onto the narrow beam's fine window resamples a little of the power.
    let l = r.report.ledger;
    measured("DMD orders through a singlet: numerical, of the launched", l.numerical / l.launched);
    measured("DMD orders through a singlet: resampling, of the launched", l.resampling / l.launched);
    assert!(l.numerical.abs() < 1e-6 && l.resampling.abs() < 2e-2, "{l:?}");
}

/// A transmission grating 50 mm from the source, met at `incidence_deg` in
/// the XY plane, `lines_per_mm` grooves along Z, with the given order and
/// zero-order fractions, and a detector 200 mm behind it facing back along
/// the beam, `half` mm wide on `bins` pixels a side. Returns the grating's id.
fn grating_scene(lines_per_mm: f64, incidence_deg: f64, eta: f64, zero: f64, half: f64, bins: usize) -> (Scene3, SourceId, DetectorId, u64) {
    let mut s = Scene3::new();
    let src = source(&mut s);
    let g = s.add_grating(Vec3::new(50.0, 0.0, 0.0), 10.0, 10.0, 180.0 + incidence_deg, lines_per_mm, true);
    assert!(s.set_grating(g, lines_per_mm, 1, true, Some(eta), Some(zero)));
    let det = s.add_area_detector(Vec3::new(250.0, 0.0, 0.0), -Vec3::X, half, half, [bins, bins]);
    s.validate().unwrap();
    (s, src, det, g)
}

#[test]
fn a_grating_order_leaves_with_its_momentum_and_its_phase() {
    // R01 of the expert review. A grating's order carries the grating's own
    // phase, exp(i m K·(r − r_g)) with K = 2π/d across the grooves: its
    // tangential wavevector is the incident one plus mK (the grating
    // equation [1]), so in a frame turned to the order's direction the field
    // stays centred on the chief ray instead of drifting off at the angle
    // the frame turned through; and moving the grating by Δ across its
    // grooves moves the order's phase by −mKΔ (by 2π for Δ = d: the same
    // field). 20 l/mm (d = 50 µm) met at 10°, the zero order set to 0 so that
    // one chain runs, η = 0.65.
    let bins = 320;
    let (s, src, det, g) = grating_scene(20.0, 10.0, 0.65, 0.0, 8.0, bins);
    let r = wave_path(&s, &spec(src, det, 0.5, 256, 4.0), 1).unwrap();
    let l = r.report.ledger;
    measured("grating order: detected, of the launched", l.detected);
    measured("grating order: untracked orders, of the launched", l.untracked_orders);
    assert!((l.detected - 0.65).abs() < 1e-6, "{l:?}");
    assert!((l.untracked_orders - 0.35).abs() < 1e-6, "{l:?}");
    // The field's centroid in the beam frame, which follows the chief ray.
    let f = &r.field;
    let (mut p, mut mx, mut my) = (0.0, 0.0, 0.0);
    for j in 0..f.grid.n[1] {
        for i in 0..f.grid.n[0] {
            let w = f.at(i, j).norm_sqr();
            let (x, y) = f.grid.xy(i, j);
            p += w;
            mx += w * x;
            my += w * y;
        }
    }
    let (mx, my) = (mx / p, my / p);
    measured("grating order: the field's centroid off the chief ray, mm", mx.hypot(my));
    assert!(mx.hypot(my) < 1e-4, "({mx}, {my}): the order drifted off its chief ray");
    // The order lands on the detector where the traced chief ray does, off
    // the axis.
    let seqs = oc_trace::extract_sequences3(&s, src, det, Some(LAM), 1).unwrap();
    assert_eq!(seqs.len(), 1, "one chain: the zero order carries nothing");
    let dt = s.detectors.iter().find(|d| d.common.id.0 == det.0).unwrap().common.transform;
    let land = dt.inverse_transform_point(seqs[0].terminal.point);
    let du = 16.0 / bins as f64;
    let (mut pw, mut cu, mut cv) = (0.0, 0.0, 0.0);
    for j in 0..bins {
        for i in 0..bins {
            let w = r.power[j * bins + i];
            pw += w;
            cu += w * (-8.0 + (i as f64 + 0.5) * du);
            cv += w * (-8.0 + (j as f64 + 0.5) * du);
        }
    }
    let (cu, cv) = (cu / pw, cv / pw);
    // The projected readout bins each field cell's power into the pixel its
    // sub-cells land in, so a centroid is known to a sub-cell (a third of a
    // 15.6 µm field cell here).
    measured("grating order: power centroid on the face against the chief ray's landing point, mm", (cu - land.x).hypot(cv - land.y));
    assert!((cu - land.x).hypot(cv - land.y) < 6e-3, "({cu}, {cv}) vs ({}, {})", land.x, land.y);
    assert!(land.x.abs() + land.y.abs() > 1.0, "the order must land off the axis: {land:?}");
    // The grating moved Δ along its own X: the same field times e^{−i2πΔ/d}.
    let gt = s.surfaces.iter().find(|o| o.common.id.0 == g).unwrap().common.transform;
    let gx = gt.rotate(Vec3::X);
    let d_mm = 1.0 / 20.0;
    let peak = r.field.data.iter().map(|z| z.abs()).fold(0.0, f64::max);
    for (frac, note) in [(0.3, "0.3 d"), (1.0, "d")] {
        let mut moved = s.clone();
        let o = moved.surfaces.iter_mut().find(|o| o.common.id.0 == g).unwrap();
        o.common.transform = oc_core::Isometry3::from_pos(gx * (frac * d_mm)).compose(o.common.transform);
        let r2 = wave_path(&moved, &spec(src, det, 0.5, 256, 4.0), 1).unwrap();
        let want = oc_wave::Cx::expi(-2.0 * std::f64::consts::PI * frac);
        let worst = r
            .field
            .data
            .iter()
            .zip(&r2.field.data)
            .filter(|(a, _)| a.abs() > 1e-3 * peak)
            .map(|(a, b)| (*b - *a * want).abs() / a.abs())
            .fold(0.0, f64::max);
        measured(&format!("grating moved {note} across its grooves: the order's field against the same field times e^(−i2πΔ/d), worst relative"), worst);
        assert!(worst < 1e-8, "{worst}");
    }
}

#[test]
fn both_orders_of_a_grating_book_what_neither_carries_once() {
    // R02 of the expert review. With the zero order and the first order both
    // followed to one detector, the grating's untracked orders are
    // 1 − η − η₀ of what met it, booked once, not (1 − η) + (1 − η₀), one per
    // arm; nothing is left to the remainder. η = 0.65, η₀ = 0.10; the
    // detector holds both spots, 2.6 mm apart.
    let (s, src, det, _) = grating_scene(20.0, 10.0, 0.65, 0.10, 8.0, 160);
    let r = wave_path(&s, &spec(src, det, 0.5, 256, 4.0), 1).unwrap();
    let l = r.report.ledger;
    measured("grating, both orders: detected, of the launched", l.detected);
    measured("grating, both orders: untracked orders, of the launched", l.untracked_orders);
    measured("grating, both orders: other ports, of the launched", l.other_ports);
    assert!((l.untracked_orders - 0.25).abs() < 1e-6, "{l:?}");
    assert!((l.detected - 0.75).abs() < 2e-3, "{l:?}");
    assert!(l.other_ports.abs() < 2e-3, "{l:?}");
    assert!(r.report.steps.iter().any(|st| st.note.contains("2 arms added")), "{:?}", r.report.steps);
}

/// The power centroid of a detector's pixels, in its own frame, mm.
fn power_centroid(r: &WavePathResult, half: f64) -> (f64, f64) {
    let [nu, nv] = r.bins;
    let (du, dv) = (2.0 * half / nu as f64, 2.0 * half / nv as f64);
    let (mut p, mut cu, mut cv) = (0.0, 0.0, 0.0);
    for j in 0..nv {
        for i in 0..nu {
            let w = r.power[j * nu + i];
            p += w;
            cu += w * (-half + (i as f64 + 0.5) * du);
            cv += w * (-half + (j as f64 + 0.5) * dv);
        }
    }
    (cu / p, cv / p)
}

#[test]
fn a_window_fitted_off_the_chief_ray_carries_the_field_where_it_is() {
    // R03 of the expert review. A re-grid centres the window on the beam's
    // predicted centroid, which a half-blocked beam moves off the chief ray;
    // every step after it (the zoomed propagators, a traced face's ray grid,
    // the readout) must then take the window where it is. A 0.1 mm Gaussian
    // has half of it stopped 10 mm on; the fitted run's window is centred
    // off the chief ray from there on, which its re-grid note records, while
    // on a 2 mm window the path never re-grids. The observable that an
    // offset would move: the beam's centroid on a detector 140 mm further.
    // The two runs must agree on it to a fraction of a sample (the offset
    // window is 55 µm off the chief ray; carried where the frame origin is,
    // it would read 55 µm off). The second-moment law gives the centroid of
    // the whole field exactly ([25]; it walks with the beam's mean direction,
    // the wavefront's slope at the centroid), ⟨y⟩ = w/√(2π) at the aperture
    // times 1 + L/R; the detector holds 96.7% of the light, the rest
    // diffracted by the edge beyond 7 mrad leaves the window, which moves
    // what it reads by a few µm. With a singlet in the way (its ray grid
    // must span the offset window) the two runs must still read the same
    // power and centroid.
    use oc_core::build3::SingletGeom;
    use oc_core::scene3::PlanarShape2;
    use oc_core::Vec2;
    let w0 = 0.1;
    let z_ap = 10.0;
    let build = |singlet: bool| {
        let mut s = Scene3::new();
        let src = source(&mut s);
        let a = s.add_aperture(Vec3::new(z_ap, 0.0, 0.0), 1.0, 180.0);
        let ap = s.apertures.iter_mut().find(|o| o.common.id.0 == a).unwrap();
        ap.shape.clear_opening =
            PlanarShape2::Polygon { vertices: vec![Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(-1.0, 1.0)] };
        ap.shape.outer_extent = None;
        if singlet {
            let g = SingletGeom { r1_mm: 50.0, r2_mm: -50.0, thickness_mm: 4.0, semi_aperture_mm: 12.0, conic1: 0.0, conic2: 0.0 };
            s.add_singlet(Vec3::new(100.0, 0.0, 0.0), g, 1.5168, 64.17);
        }
        let det = s.add_area_detector(Vec3::new(150.0, 0.0, 0.0), -Vec3::X, 1.0, 1.0, [100, 100]);
        s.validate().unwrap();
        (s, src, det)
    };
    // Free space: the centroid against the second-moment law.
    let (s, src, det) = build(false);
    let fitted = wave_path(&s, &spec(src, det, w0, 512, 4.0), 1).unwrap();
    let fixed = wave_path(&s, &spec(src, det, w0, 512, 2.0), 1).unwrap();
    let off: Vec<&str> = fitted.report.steps.iter().map(|st| st.note.as_str()).filter(|n| n.contains("onto a new window")).collect();
    assert!(off.iter().any(|n| n.contains("centred (") && !n.contains("centred (+0.0000e0, +0.0000e0)")), "a window off the chief ray: {off:?}");
    assert!(!fixed.report.steps.iter().any(|st| st.note.contains("onto a new window")), "the reference must not re-grid");
    let zr = rayleigh_range_mm(w0, LAM, 1.0);
    let w_ap = w0 * (1.0 + (z_ap / zr).powi(2)).sqrt();
    let radius = (z_ap * z_ap + zr * zr) / z_ap;
    let want = w_ap / (2.0 * std::f64::consts::PI).sqrt() * (1.0 + (150.0 - z_ap) / radius);
    let (fu, fv) = power_centroid(&fitted, 1.0);
    let (xu, xv) = power_centroid(&fixed, 1.0);
    measured("window fitted off the chief ray: the beam's centroid on the detector against the second-moment law, mm", fu.hypot(fv) - want);
    measured("the same on a fixed window, mm", xu.hypot(xv) - want);
    measured("window fitted off the chief ray against a fixed window: centroid difference, mm", (fu - xu).hypot(fv - xv));
    assert!((fu.hypot(fv) - want).abs() < 1e-2, "{} vs {want}", fu.hypot(fv));
    assert!((xu.hypot(xv) - want).abs() < 1e-2, "{} vs {want}", xu.hypot(xv));
    assert!((fu - xu).hypot(fv - xv) < 2e-4, "{}", (fu - xu).hypot(fv - xv));
    // Through a singlet: the same power and centroid either way.
    let (s, src, det) = build(true);
    let fitted = wave_path(&s, &spec(src, det, w0, 512, 4.0), 1).unwrap();
    let fixed = wave_path(&s, &spec(src, det, w0, 512, 2.0), 1).unwrap();
    assert!(fitted.report.steps.iter().any(|st| st.note.contains("onto a new window") && st.note.contains("centred (") && !st.note.contains("centred (+0.0000e0, +0.0000e0)")));
    let (fu, fv) = power_centroid(&fitted, 1.0);
    let (xu, xv) = power_centroid(&fixed, 1.0);
    // The windows differ in what they hold: the fixed 2 mm one loses 1.5% of
    // the edge's wide-angle light to its edges and 0.8% to the coarser
    // sampling of the edge, where the fitted one, 4 mm wide at the lens,
    // loses 0.6% and 0.2% (their ledgers say so).
    measured("window fitted off the chief ray, through a singlet: detected power against a fixed window", fitted.report.ledger.detected - fixed.report.ledger.detected);
    measured("window fitted off the chief ray, through a singlet: centroid difference against a fixed window, mm", (fu - xu).hypot(fv - xv));
    assert!((fitted.report.ledger.detected - fixed.report.ledger.detected).abs() < 2e-2, "{:?} vs {:?}", fitted.report.ledger, fixed.report.ledger);
    assert!((fu - xu).hypot(fv - xv) < 2e-3);
}

/// Every object of `s` carried by the rigid motion `t`.
fn moved_scene(s: &Scene3, t: oc_core::Isometry3) -> Scene3 {
    let mut m = s.clone();
    for o in &mut m.surfaces {
        o.common.transform = t.compose(o.common.transform);
    }
    for o in &mut m.bodies {
        o.common.transform = t.compose(o.common.transform);
    }
    for o in &mut m.apertures {
        o.common.transform = t.compose(o.common.transform);
    }
    for o in &mut m.detectors {
        o.common.transform = t.compose(o.common.transform);
    }
    for o in &mut m.sources {
        o.transform = t.compose(o.transform);
    }
    m
}

#[test]
fn a_rigidly_moved_scene_reads_the_same_detector() {
    // R06 of the expert review (gauge): a rotation and a translation of the
    // whole scene change no observable. A folded path, a Mach–Zehnder and a
    // Michelson with a tilted mirror (two retro-reflections, whose frames are
    // carried by a half-turn about the beam's own axis), moved by 0.7 rad
    // about (1, 2, 3) and by (3, −2, 5) mm, must read the same pixels.
    let q = glam::DQuat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    let t = oc_core::Isometry3::new(Vec3::new(3.0, -2.0, 5.0), [q.x, q.y, q.z, q.w]).unwrap();
    // The projected readout bins whole field cells: a cell whose centre
    // falls on a pixel edge goes either way by round-off, so the pixels are
    // sized not to put edges on cell centres (3 mm over 256 cells, 6 mm over
    // 30 pixels).
    let mut folded = Scene3::new();
    let src_f = source(&mut folded);
    folded.add_planar_mirror(Vec3::new(97.0, -3.0, 0.0), Vec3::new(103.0, 3.0, 0.0), 1.0).unwrap();
    let det_f = folded.add_area_detector(Vec3::new(100.0, 200.0, 0.0), -Vec3::Y, 3.0, 3.0, [30, 30]);
    folded.validate().unwrap();
    let (mz, src_m, det_m, _) = mach_zehnder(0.3e-3);
    // Arms 0.1 mm apart, so neither port is dark.
    let (mi, src_i, det_i) = michelson(0.1, 0.0, 5e-4, Spectrum::Monochromatic(LAM), 3.0, 64);
    let (mi0, src_0, det_0) = michelson(0.1, 0.0, 0.0, Spectrum::Monochromatic(LAM), 3.0, 64);
    // An interferometer reads the phase between its arms, k·OPL ≈ 4e6 rad
    // over these paths: the tracer's hit points, rotated, move by round-off
    // (1e-13 of 250 mm), which is 1e-7 rad, so its pixels can differ by
    // about that much of the launched power, and an arm read through the
    // rotated spectrum (the tilted Michelson's) by its interpolation's
    // round-off too; a single arm's cannot.
    for (name, s, src, det, w0, n, win, gate) in [
        ("folded path", &folded, src_f, det_f, 0.2, 256, 3.0, 1e-9),
        ("Mach–Zehnder", &mz, src_m, det_m, 0.5, 256, 4.0, 1e-5),
        ("Michelson, square", &mi0, src_0, det_0, 1.0, 256, 5.0, 1e-5),
        ("Michelson, tilted mirror", &mi, src_i, det_i, 1.0, 256, 5.0, 1e-5),
    ] {
        let run = |sc: &Scene3| wave_path(sc, &spec(src, det, w0, n, win), 1).unwrap();
        let a = run(s);
        let b = run(&moved_scene(s, t));
        let launched = a.report.ledger.launched;
        let worst = a.power.iter().zip(&b.power).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        measured(&format!("{name}, moved rigidly: pixel power difference / launched"), worst / launched);
        assert!(worst / launched < gate, "{name}: {}", worst / launched);
    }
}

/// The Mach–Zehnder of `mach_zehnder`, its first splitter reflecting `r1`
/// (transmitting the rest), so that the reflected arm can be made as faint
/// as one likes.
fn mach_zehnder_split(delta: f64, r1: f64) -> (Scene3, SourceId, DetectorId, DetectorId) {
    let mut s = Scene3::new();
    let src = source(&mut s);
    let m = SurfaceResponse::Mirror { reflectivity: oc_core::scene3::SpectralScalar::Constant(1.0) };
    let n = Vec3::new(-1.0, 1.0, 0.0).normalize();
    s.add_beam_splitter(Vec3::new(50.0, 0.0, 0.0), 10.0, 135.0, r1, 1.0 - r1).unwrap();
    s.add_planar_element(Vec3::new(150.0, 0.0, 0.0), 10.0, 135.0, m.clone(), m.clone());
    s.add_planar_element(Vec3::new(50.0, 100.0, 0.0) + n * delta, 10.0, 135.0, m.clone(), m);
    s.add_beam_splitter(Vec3::new(150.0, 100.0, 0.0), 10.0, 135.0, 0.5, 0.5).unwrap();
    let dx = s.add_area_detector(Vec3::new(250.0, 100.0, 0.0), -Vec3::X, 3.0, 3.0, [64, 64]);
    let dy = s.add_area_detector(Vec3::new(150.0, 200.0, 0.0), -Vec3::Y, 3.0, 3.0, [64, 64]);
    s.validate().unwrap();
    (s, src, dx, dy)
}

#[test]
fn a_vanishing_second_arm_changes_the_bright_arm_continuously() {
    // R12 of the expert review. One chain and several are read by one
    // method, so a second arm carrying ε of the power must change the bright
    // arm's pixels by the cross term alone, of order √ε, and nothing at
    // ε = 0 (the splitter then sends no child along the dark arm, and one
    // chain runs). The first splitter reflects ε into the second arm.
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let delta = 0.37 / (k0 * std::f64::consts::SQRT_2);
    let (s0, src, dx, _) = mach_zehnder_split(delta, 0.0);
    let base = wave_path(&s0, &spec(src, dx, 0.5, 256, 4.0), 1).unwrap();
    let arms0 = oc_trace::extract_sequences3(&s0, src, dx, Some(LAM), 1).unwrap().len();
    assert_eq!(arms0, 1, "at ε = 0 one chain reaches the detector");
    let peak = base.power.iter().cloned().fold(0.0, f64::max);
    for eps in [1e-2, 1e-4, 1e-8] {
        let (s, src, dx, _) = mach_zehnder_split(delta, eps);
        let r = wave_path(&s, &spec(src, dx, 0.5, 256, 4.0), 1).unwrap();
        assert!(r.report.steps.iter().any(|st| st.note.contains("arm 2 joins arm 1")), "{:?}", r.report.steps);
        let worst = base.power.iter().zip(&r.power).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        measured(&format!("Mach–Zehnder with ε = {eps:.0e} in the second arm: the bright arm's pixels against ε = 0, worst / peak"), worst / peak);
        measured(&format!("Mach–Zehnder with ε = {eps:.0e} in the second arm: detected power against ε = 0"), r.report.ledger.detected - base.report.ledger.detected);
        assert!(worst / peak < 3.0 * eps.sqrt(), "{}", worst / peak);
        assert!((r.report.ledger.detected - base.report.ledger.detected).abs() < 3.0 * eps.sqrt(), "{:?}", r.report.ledger);
    }
}

#[test]
fn a_rolled_detector_reads_the_same_polarized_fringes() {
    // R14 of the expert review, the readout's basis: each arm's two
    // components are turned onto the face's own axes before the arms are
    // added. A detector rolled 30° about its normal, and one tilted 30°
    // about its own in-plane axis (the arms then oblique to it, their
    // components turned onto the face by the rotation of their direction
    // onto its normal), read the half-wave-plate Mach–Zehnder as the square
    // one does: half the launched power in the X port at every phase, and
    // behind an analyzer at β the Fresnel–Arago law
    // `1/4 − cos Δφ · cos 2(θ − β)/4` [26], [27] (the pixels move with the
    // face; the totals cannot).
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let theta = 20.0;
    for (dphi, beta) in [(0.0, None), (std::f64::consts::FRAC_PI_2, Some(50.0)), (std::f64::consts::PI, Some(20.0))] {
        let delta = dphi / (k0 * std::f64::consts::SQRT_2);
        let mut totals = Vec::new();
        // The face turned about its own axis: Z its normal (a roll), X an
        // in-plane axis (a tilt).
        for (name, axis, deg) in [("square", Vec3::Z, 0.0f64), ("rolled 30°", Vec3::Z, 30.0), ("tilted 30°", Vec3::X, 30.0)] {
            let (mut s, src, dx, _) = mach_zehnder(delta);
            // The half-wave plate in the transmitted arm, square to it.
            s.add_planar_element(Vec3::new(100.0, 0.0, 0.0), 10.0, 0.0, retarder(theta, 180.0), retarder(theta, 180.0));
            if let Some(b) = beta {
                s.add_planar_element(Vec3::new(200.0, 100.0, 0.0), 10.0, 0.0, polarizer(b, 0.0), polarizer(b, 0.0));
            }
            let d = s.detectors.iter_mut().find(|d| d.common.id.0 == dx.0).unwrap();
            let turn = oc_core::Isometry3::from_axis_angle(axis, deg.to_radians()).unwrap();
            d.common.transform = d.common.transform.compose(turn);
            s.validate().unwrap();
            let r = wave_path(&s, &spec(src, dx, 0.5, 256, 4.0), 1).unwrap();
            if deg != 0.0 {
                assert!(r.report.steps.iter().any(|st| st.note.contains("spectrum rotation")), "{name}: {:?}", r.report.steps);
            }
            totals.push((name, r.report.ledger.detected));
        }
        let want = match beta {
            None => 0.5,
            Some(b) => 0.25 - 0.25 * dphi.cos() * (2.0 * (theta - b).to_radians()).cos(),
        };
        measured(&format!("half-wave plate Mach–Zehnder, Δφ = {dphi:.3}, analyzer {beta:?}: square detector against the law"), totals[0].1 - want);
        assert!((totals[0].1 - want).abs() < 1e-6, "{} vs {want}", totals[0].1);
        for (name, total) in &totals[1..] {
            measured(&format!("half-wave plate Mach–Zehnder, Δφ = {dphi:.3}, analyzer {beta:?}: {name} detector against the square one"), total - totals[0].1);
            assert!((total - totals[0].1).abs() < 1e-6, "{name}: {total} vs {}", totals[0].1);
        }
    }
}

/// A Mach–Zehnder whose arms leave the plane: from a splitter at the origin,
/// arm A goes +X, is folded up +Z, then along +Y to the second splitter at
/// (100, 100, 100); arm B goes +Y, is folded along +X, then up +Z to the same
/// splitter, which sends both along +Z to a detector at z = 200 (and the
/// other port along +Y to a second detector).
fn noncoplanar_mach_zehnder() -> (Scene3, SourceId, DetectorId, DetectorId) {
    use oc_core::scene3::SpectralScalar;
    let mut s = Scene3::new();
    let src = source(&mut s);
    let m = SurfaceResponse::Mirror { reflectivity: SpectralScalar::Constant(1.0) };
    let mirror = |s: &mut Scene3, pos: Vec3, normal: Vec3| {
        let id = s.add_planar_element(pos, 10.0, 0.0, m.clone(), m.clone());
        let o = s.surfaces.iter_mut().find(|o| o.common.id.0 == id).unwrap();
        o.common.transform = facing_iso(pos, normal);
        id
    };
    let split = |s: &mut Scene3, pos: Vec3, normal: Vec3| {
        let id = s.add_beam_splitter(pos, 10.0, 0.0, 0.5, 0.5).unwrap();
        let o = s.surfaces.iter_mut().find(|o| o.common.id.0 == id).unwrap();
        o.common.transform = facing_iso(pos, normal);
        id
    };
    // Splitter 1 at the origin+50: +X transmits, reflects to +Y.
    split(&mut s, Vec3::new(50.0, 0.0, 0.0), Vec3::new(-1.0, 1.0, 0.0).normalize());
    // Arm A: +X → +Z at (150, 0, 0); +Z → +Y at (150, 0, 100).
    mirror(&mut s, Vec3::new(150.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 1.0).normalize());
    mirror(&mut s, Vec3::new(150.0, 0.0, 100.0), Vec3::new(0.0, 1.0, -1.0).normalize());
    // Arm B: +Y → +X at (50, 100, 0); +X → +Z at (150, 100, 0).
    mirror(&mut s, Vec3::new(50.0, 100.0, 0.0), Vec3::new(1.0, -1.0, 0.0).normalize());
    mirror(&mut s, Vec3::new(150.0, 100.0, 0.0), Vec3::new(-1.0, 0.0, 1.0).normalize());
    // Splitter 2 at (150, 100, 100): arm A along +Y reflects to +Z, arm B
    // along +Z transmits; the other port leaves along +Y.
    split(&mut s, Vec3::new(150.0, 100.0, 100.0), Vec3::new(0.0, -1.0, 1.0).normalize());
    let dz = s.add_area_detector(Vec3::new(150.0, 100.0, 200.0), -Vec3::Z, 3.0, 3.0, [64, 64]);
    let dy = s.add_area_detector(Vec3::new(150.0, 200.0, 100.0), -Vec3::Y, 3.0, 3.0, [64, 64]);
    s.validate().unwrap();
    (s, src, dz, dy)
}

/// An isometry whose local +Z is `normal`, at `pos`.
fn facing_iso(pos: Vec3, normal: Vec3) -> oc_core::Isometry3 {
    let z = normal.normalize();
    let seed = if z.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let x = (seed - z * z.dot(seed)).normalize();
    let y = z.cross(x);
    let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(x, y, z));
    oc_core::Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap()
}

#[test]
fn a_noncoplanar_mach_zehnder_conserves_power_across_its_ports() {
    // R14 of the expert review: arms that leave the plane reach the second
    // splitter with transverse frames turned differently about the beam.
    // Added on the detector's own axes, the two output ports must still
    // carry the launched power between them, at whatever phase the arms
    // meet, and a rigid roll of the whole scene must change neither port
    // (the frames are carried by the geometry alone).
    let (s, src, dz, dy) = noncoplanar_mach_zehnder();
    assert_eq!(oc_trace::extract_sequences3(&s, src, dz, Some(LAM), 1).unwrap().len(), 2, "two arms reach the Z port");
    let rz = wave_path(&s, &spec(src, dz, 0.5, 256, 4.0), 1).unwrap();
    let ry = wave_path(&s, &spec(src, dy, 0.5, 256, 4.0), 1).unwrap();
    let (pz, py) = (rz.report.ledger.detected, ry.report.ledger.detected);
    measured("non-coplanar Mach–Zehnder: both ports' power, of the launched", pz + py);
    measured("non-coplanar Mach–Zehnder: the Z port's share", pz / (pz + py));
    assert!((pz + py - 1.0).abs() < 2e-3, "{pz} + {py}");
    assert!(rz.report.steps.iter().any(|st| st.note.contains("arm 2 joins arm 1")), "{:?}", rz.report.steps);
    let q = glam::DQuat::from_axis_angle(Vec3::new(2.0, -1.0, 3.0).normalize(), 1.1);
    let t = oc_core::Isometry3::new(Vec3::new(-4.0, 7.0, 2.0), [q.x, q.y, q.z, q.w]).unwrap();
    let moved = moved_scene(&s, t);
    let rz2 = wave_path(&moved, &spec(src, dz, 0.5, 256, 4.0), 1).unwrap();
    let worst = rz.power.iter().zip(&rz2.power).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
    measured("non-coplanar Mach–Zehnder, moved rigidly: the Z port's pixels against the unmoved, / launched", worst);
    assert!(worst < 1e-5, "{worst}");
}

#[test]
fn an_aperture_behind_a_recombining_splitter_clips_the_sum_of_the_arms() {
    // R04 of the expert review. The Mach–Zehnder's arms meet at the second
    // splitter and leave it as one beam, so the arm that reflects there
    // joins the one that goes through, and an aperture behind it clips
    // their sum: a disc of radius a in the X port takes the same share of
    // what reaches it at every phase, exp(−2a²/w²) [13] (w the beam's
    // radius there, 300 mm from its waist; the disc's edge cells on this
    // 15.6 µm grid differ from the closed form by 9e-5, as the single-beam
    // disc test's do by 3.2e-5 at 512 samples), and nothing when the port
    // is dark. Run arm by arm, the disc would take its share of each arm's
    // power whatever the phase.
    // The Y port, read on its own detector, carries what the X port's run
    // books to the port it leaves unread. Sliced into single events, the
    // job joins the arms at the same event and gives the same bits.
    use oc_wave::path::WaveJob;
    let k0 = 2.0 * std::f64::consts::PI / (LAM / 1000.0);
    let a = 0.4;
    let w0: f64 = 0.5;
    let z_r = std::f64::consts::PI * w0 * w0 / (LAM / 1000.0);
    let w = w0 * (1.0 + (300.0 / z_r).powi(2)).sqrt();
    let share_law = (-2.0 * a * a / (w * w)).exp();
    let mut shares = Vec::new();
    for j in 0..3 {
        let dphi = j as f64 * std::f64::consts::FRAC_PI_2;
        let delta = dphi / (k0 * std::f64::consts::SQRT_2);
        let (mut s, src, dx, dy) = mach_zehnder(delta);
        s.add_aperture(Vec3::new(200.0, 100.0, 0.0), a, 0.0);
        s.validate().unwrap();
        let rx = wave_path(&s, &spec(src, dx, w0, 256, 4.0), 1).unwrap();
        let ry = wave_path(&s, &spec(src, dy, w0, 256, 4.0), 1).unwrap();
        let (lx, ly) = (rx.report.ledger, ry.report.ledger);
        assert!(rx.report.steps.iter().any(|st| st.note.contains("arm 2 joins arm 1")), "{:?}", rx.report.steps);
        // What reaches the disc: the X port's share of the launched,
        // (1 − cos Δφ)/2 [23], [24].
        let port = 0.5 * (1.0 - dphi.cos());
        measured(&format!("Mach–Zehnder at Δφ = {j}·π/2, a disc behind the second splitter: absorbed, of the launched"), lx.absorbed_aperture);
        if port > 0.1 {
            let share = lx.absorbed_aperture / port;
            measured(&format!("Mach–Zehnder at Δφ = {j}·π/2: the disc's share of the port against exp(−2a²/w²)"), share - share_law);
            assert!((share - share_law).abs() < 2e-4, "{share} vs {share_law}");
            shares.push(share);
        } else {
            assert!(lx.absorbed_aperture < 1e-6, "{lx:?}");
        }
        measured(&format!("Mach–Zehnder at Δφ = {j}·π/2: the X run's unread port against the Y port's run"), lx.other_arm - ly.detected);
        assert!((lx.other_arm - ly.detected).abs() < 1e-6, "{lx:?} vs {ly:?}");
        measured(&format!("Mach–Zehnder at Δφ = {j}·π/2: the X run's remainder, of the launched"), lx.numerical + lx.other_ports);
        if j == 1 {
            let sp = spec(src, dx, w0, 256, 4.0);
            let mut job = WaveJob::new(&s, &sp, 1).unwrap();
            while !job.step(1).unwrap() {}
            let sliced = job.finish().unwrap();
            assert_eq!(rx, sliced, "the sliced job differs from the whole one");
        }
    }
    measured("Mach–Zehnder, a disc behind the second splitter: its share of the port at π/2 against π", shares[0] - shares[1]);
    assert!((shares[0] - shares[1]).abs() < 1e-6, "{shares:?}");
}
