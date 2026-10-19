//! The coherent wave path: a laser beam's field carried along the chain of
//! surfaces the f64 tracer extracts (ADR 0002), element by element, to a
//! detector.
//!
//! The chief ray's sequence (`oc_trace::extract_sequence3`) says where the
//! beam goes. The field rides in a frame across that ray: its origin on the
//! ray, its `+Z` along it. Between two events the field propagates the arc
//! length between them with [`crate::propagate`]'s automatic choice. At an
//! event it is handed from the plane across the incoming ray to the plane
//! across the outgoing one:
//!
//! - **Flat mirror, splitter arm, grating order.** The element maps the
//!   incident field onto the outgoing one point for point. For a flat mirror
//!   that is image theory and exact in scalar optics: the reflected field is
//!   the incident field's mirror image. A grating adds its momentum along the
//!   grooves, which the new direction absorbs. The amplitude is `√R`, `√T`
//!   or `√η`.
//! - **Ideal thin lens.** `exp(−ik|r − c|²/2f)` in the lens plane, about the
//!   lens's centre.
//! - **Aperture.** Its clear opening, sampled 3 × 3 per sample on the
//!   aperture's own plane. The tracer logs no event when a ray passes an
//!   opening, so the path finds every aperture by intersecting the chief ray
//!   with each aperture's plane.
//! - **DMD or LCoS SLM.** The field is laid onto the device plane at a few
//!   samples per pixel, reflected by [`crate::modulator`], and taken up across
//!   the outgoing ray, which the tracer has already pointed along the hit
//!   pixel's facet reflection or phase gradient.
//!
//! When an element's plane is not square to the beam, the handover projects
//! the field along the rays rather than diffracting it across the depth of
//! the plane. That is exact for a flat mirror, and an approximation
//! (`TiltProjection`, with the depth reported) for a lens or a modulator
//! seen obliquely: the textbook physical-optics practice, and the case W8
//! replaces with the tilted-plane angular spectrum (Matsushima, Schimmel &
//! Wyrowski, *J. Opt. Soc. Am. A* 20, 1755 (2003)).
//!
//! Refused, by name: refractive bodies and curved mirrors (W6), polarizing
//! elements (the field is scalar), and anything else the field has no model
//! for. Nothing is skipped silently.

use crate::beams::GaussianBeam;
use crate::modulator::{dmd_reflect, lcos_reflect};
use crate::propagate::propagate;
use crate::report::{SamplingWarning, StepReport, WaveLedger};
use crate::transfer::{fit_sphere, Smooth};
use crate::{Cx, Fft2, Field2, Grid2, Method, Propagator, WaveError, WaveReport};
use oc_core::id::{DetectorId, SourceId};
use oc_core::modulator::{DmdState, LcosBeam, ModulatorKind};
use oc_core::scene3::{DetectorGeometry3, PlanarShape2, Scene3, SurfaceGeometry3, SurfaceResponse};
use oc_core::seq3::{SeqAction, SeqHitKind};
use oc_core::{Isometry3, Vec3};

/// The coherent beam a wave path launches: the scene's own type, which a
/// source carries as its `coherence` and the document saves.
pub use oc_core::scene3::CoherentBeam as BeamProfile;

/// How the detector is read.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Readout {
    /// The field carried to the detector's centre on its own grid, each
    /// cell's power binned into the pixel its projection lands in.
    #[default]
    Projected,
    /// The last free-space step taken straight onto the detector by the
    /// zoomed angular spectrum ([`crate::asm_zoom`]), on `samples_per_pixel`²
    /// points a pixel and then binned: a focal spot far smaller than the
    /// field's own samples is resolved, with no paraxial approximation.
    Diffracted { samples_per_pixel: usize },
}

/// What to compute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavePathSpec {
    pub source: SourceId,
    pub detector: DetectorId,
    pub beam: BeamProfile,
    /// The beam's power (the same relative units as a ray's flux).
    pub power: f64,
    /// Samples across the field, and the field's width, mm.
    pub grid_n: usize,
    pub window_mm: f64,
    /// Samples per pixel when the field is laid onto a DMD or SLM.
    pub device_samples_per_pixel: usize,
    pub readout: Readout,
}

impl WavePathSpec {
    /// A spec for the beam `source` declares (its `coherence`), with the
    /// source's flux as the beam's power, `device_samples_per_pixel` 4.
    /// Refused, by name, for a source with no coherent beam.
    pub fn for_source(
        scene: &Scene3,
        source: SourceId,
        detector: DetectorId,
        grid_n: usize,
        window_mm: f64,
    ) -> Result<Self, WaveError> {
        let src = scene
            .sources
            .iter()
            .find(|s| s.id == source)
            .ok_or_else(|| WaveError::OutOfRange("no such source".into()))?;
        let beam = src.coherence.ok_or_else(|| {
            WaveError::OutOfRange(format!("source #{} launches no coherent beam; give it one to run a wave path", source.0))
        })?;
        beam.check().map_err(WaveError::OutOfRange)?;
        Ok(WavePathSpec {
            source,
            detector,
            beam,
            power: src.flux,
            grid_n,
            window_mm,
            device_samples_per_pixel: 4,
            readout: Readout::Projected,
        })
    }
}

/// What the path delivered.
#[derive(Debug, Clone, PartialEq)]
pub struct WavePathResult {
    pub revision: u64,
    pub detector: DetectorId,
    /// The detector's pixel counts and the power in each pixel, `u` fastest.
    pub bins: [usize; 2],
    pub power: Vec<f64>,
    /// The field arriving at the detector, across the chief ray.
    pub field: Field2,
    /// The power arriving at the detector's plane, before its face (or the
    /// diffracted readout's window) takes its share.
    pub arriving: f64,
    pub report: WaveReport,
}

/// A frame across the chief ray: origin on it, `d` along it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Beam {
    pub(crate) origin: Vec3,
    pub(crate) x: Vec3,
    pub(crate) y: Vec3,
    pub(crate) d: Vec3,
}

impl Beam {
    pub(crate) fn iso(&self) -> Isometry3 {
        let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols(self.x, self.y, self.d));
        Isometry3::new(self.origin, [q.x, q.y, q.z, q.w]).unwrap_or(Isometry3::from_pos(self.origin))
    }
    pub(crate) fn coords(&self, r: Vec3) -> (f64, f64) {
        let v = r - self.origin;
        (v.dot(self.x), v.dot(self.y))
    }
    pub(crate) fn point(&self, a: f64, b: f64) -> Vec3 {
        self.origin + self.x * a + self.y * b
    }
    /// The same beam going along `d2` from `origin`: its transverse axes
    /// carried by the shortest rotation, which keeps the frame right-handed.
    fn turned(&self, origin: Vec3, d2: Vec3) -> Beam {
        let q = glam::DQuat::from_rotation_arc(self.d, d2);
        let x = (q * self.x - d2 * d2.dot(q * self.x)).normalize();
        Beam { origin, x, y: d2.cross(x), d: d2 }
    }
}

/// Bilinear sample of a field's data at local `(a, b)`; zero outside.
pub(crate) fn sample(u: &Field2, a: f64, b: f64) -> Cx {
    let g = u.grid;
    let fi = (a - g.center_mm[0]) / g.pitch_mm[0] + (g.n[0] / 2) as f64;
    let fj = (b - g.center_mm[1]) / g.pitch_mm[1] + (g.n[1] / 2) as f64;
    if !(fi >= 0.0 && fj >= 0.0 && fi <= (g.n[0] - 1) as f64 && fj <= (g.n[1] - 1) as f64) {
        return Cx::ZERO;
    }
    let (i0, j0) = (fi.floor() as usize, fj.floor() as usize);
    let (i1, j1) = ((i0 + 1).min(g.n[0] - 1), (j0 + 1).min(g.n[1] - 1));
    let (s, t) = (fi - i0 as f64, fj - j0 as f64);
    u.at(i0, j0) * ((1.0 - s) * (1.0 - t))
        + u.at(i1, j0) * (s * (1.0 - t))
        + u.at(i0, j1) * ((1.0 - s) * t)
        + u.at(i1, j1) * (s * t)
}

/// Where the line through `r` along `dir` meets the plane through `p` with
/// normal `n`. `None` when it runs along the plane.
fn onto_plane(r: Vec3, dir: Vec3, p: Vec3, n: Vec3) -> Option<Vec3> {
    let den = dir.dot(n);
    if den.abs() < 1e-12 {
        return None;
    }
    Some(r + dir * ((p - r).dot(n) / den))
}

/// Hand the field `u` (across `inc`) through an element in the plane through
/// `inc.origin` with normal `n_e`, to the plane across `out`. `factor` is the
/// element's complex factor at a point of its plane, including any phase the
/// change of direction does not already carry. Returns the new field and the
/// largest depth the projection spanned, mm.
fn hand_over(u: &Field2, inc: &Beam, n_e: Vec3, out: &Beam, factor: &dyn Fn(Vec3) -> Cx) -> (Field2, f64) {
    let mut v = u.clone();
    let g = u.grid;
    let k = u.k_mm();
    let p = inc.origin;
    let mut depth: f64 = 0.0;
    let obliquity = obliquity(inc.d, out.d, n_e);
    let smooth = Smooth::new(u);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (a, b) = g.xy(i, j);
            let r_out = out.point(a, b);
            let val = match onto_plane(r_out, out.d, p, n_e) {
                Some(r_e) => {
                    let r_in = r_e - inc.d * (r_e - p).dot(inc.d);
                    depth = depth.max((r_out - r_e).length() + (r_e - r_in).length());
                    let (ai, bi) = inc.coords(r_in);
                    // The reference wave changes from exp(ik d·r) to
                    // exp(ik d'·r): what the new direction does not carry of
                    // the element's own phase is put on the samples.
                    smooth.at(ai, bi) * factor(r_e) * Cx::expi(k * (inc.d - out.d).dot(r_e - p)) * obliquity
                }
                None => Cx::ZERO,
            };
            v.data[g.idx(i, j)] = val;
        }
    }
    v.frame = out.iso();
    (v, depth)
}

/// The amplitude factor that conserves the power crossing an element's
/// plane when the beam leaves at another angle than it came:
/// `√(cos θ_in / cos θ_out)`. A beam's cross-section changes by
/// `cos θ_out / cos θ_in` on the way through (a grating's anamorphic factor),
/// while projection keeps the amplitude; this puts the power right. It is 1
/// for a mirror, and it is the obliquity the non-paraxial scalar treatment
/// carries by working in direction cosines (J. E. Harvey et al., *Proc. SPIE*
/// 8122, 812205 (2011)).
fn obliquity(d_in: Vec3, d_out: Vec3, n_e: Vec3) -> f64 {
    let (ci, co) = (d_in.dot(n_e).abs(), d_out.dot(n_e).abs());
    if co < 1e-12 {
        0.0
    } else {
        (ci / co).sqrt()
    }
}

fn shape_contains(s: &PlanarShape2, x: f64, y: f64) -> bool {
    match s {
        PlanarShape2::Rectangle { half_width_mm, half_height_mm } => x.abs() <= *half_width_mm && y.abs() <= *half_height_mm,
        PlanarShape2::Disc { radius_mm } => x * x + y * y <= radius_mm * radius_mm,
        PlanarShape2::Annulus { inner_mm, outer_mm } => {
            let r2 = x * x + y * y;
            r2 >= inner_mm * inner_mm && r2 <= outer_mm * outer_mm
        }
        PlanarShape2::Polygon { vertices } => {
            let mut inside = false;
            let n = vertices.len();
            for i in 0..n {
                let (a, b) = (vertices[i], vertices[(i + 1) % n]);
                if (a.y > y) != (b.y > y) && x < a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y) {
                    inside = !inside;
                }
            }
            inside
        }
    }
}

fn merge(total: &mut WaveReport, step: WaveReport) {
    let l = step.ledger;
    let t = &mut total.ledger;
    t.absorbed_aperture += l.absorbed_aperture;
    t.absorbed_element += l.absorbed_element;
    t.band_limit_removed += l.band_limit_removed;
    t.beyond_nyquist += l.beyond_nyquist;
    t.evanescent += l.evanescent;
    t.left_window += l.left_window;
    t.other_family += l.other_family;
    t.untracked_orders += l.untracked_orders;
    t.numerical += l.numerical;
    total.steps.extend(step.steps);
    for w in step.warnings {
        total.warn(w);
    }
    for v in step.validity {
        total.flag(v);
    }
    for a in step.approximated {
        total.approx(a);
    }
    total.rejected.extend(step.rejected);
}

/// The grid's samples moved onto the centres of the cells it divides its
/// width into: `Grid2` puts sample `n/2` on the centre, which on an even axis
/// is half a pitch off every cell's centre.
fn centred(mut g: Grid2) -> Grid2 {
    for a in 0..2 {
        if g.n[a] % 2 == 0 {
            g.center_mm[a] += 0.5 * g.pitch_mm[a];
        }
    }
    g
}

/// The scene with one element alone: nothing else can meet its rays.
fn probe_scene(scene: &Scene3, object: oc_core::id::ObjectId) -> Scene3 {
    let mut probe = scene.clone();
    probe.bodies.retain(|b| b.common.id == object);
    probe.surfaces.retain(|s| s.common.id == object);
    probe.apertures.clear();
    probe.detectors.clear();
    probe.sources.clear();
    probe
}

/// How far, before the chief ray's hit along `d_in` and after it along
/// `d_out`, the surface reaches where rays over the beam window meet it:
/// rays from `from` along `d_in`, on rings out to the window's corners, and
/// where each first meets the element.
fn slab(probe: &Scene3, from: Vec3, d_in: Vec3, hit: Vec3, d_out: Vec3, half: f64, lam: f64) -> (f64, f64) {
    use oc_trace::{trace3, LaunchRay3, TraceConfig3};
    let cfg = TraceConfig3 { ghosts: false, polarization: false, bounds_mm: oc_trace::bounds_for_scene3(probe), ..TraceConfig3::default() };
    let x = if d_in.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let x = (x - d_in * d_in.dot(x)).normalize();
    let y = d_in.cross(x);
    let (mut s_min, mut s_max) = (0.0f64, 0.0f64);
    let r_max = half * std::f64::consts::SQRT_2;
    for ring in 1..=4 {
        let r = r_max * ring as f64 / 4.0;
        for k in 0..24 {
            let t = 2.0 * std::f64::consts::PI * k as f64 / 24.0;
            let o = from + x * (r * t.cos()) + y * (r * t.sin());
            let Ok(ray) = oc_core::vec3::Ray3::new(o, d_in) else { continue };
            let res = trace3(probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
            let Some(root) = res.segments.iter().find(|s| s.parent_path_id.is_none()) else { continue };
            // A root with a child met the element at its end.
            if !res.segments.iter().any(|s| s.parent_path_id == Some(root.path_id)) {
                continue;
            }
            s_min = s_min.min((root.b - hit).dot(d_in));
            s_max = s_max.max((root.b - hit).dot(d_out));
        }
    }
    (s_min, s_max)
}

fn element_step(z: f64, note: String) -> StepReport {
    StepReport { method: Method::Element, z_mm: z, sampling: None, note }
}

/// One point of the chain: a surface the trace logged, an aperture plane the
/// chief ray crosses (which the trace does not log), a refractive body or a
/// curved surface the field is traced through, or the detector.
#[derive(Debug, Clone)]
enum Ev {
    Hit(oc_core::seq3::SeqHit3),
    Aperture(usize),
    /// The field reaches the entry plane at the event's point and leaves
    /// from `exit` along `d_out`; `hit` is the chief ray's hit between.
    Transfer { hit: oc_core::seq3::SeqHit3, exit: Vec3, d_out: Vec3 },
    End,
}

/// A wave path in resumable steps, for a worker that must be able to drop a
/// stale job between steps: [`WaveJob::new`] plans the chain and launches
/// the beam, each [`WaveJob::step`] carries the field through a number of
/// the chain's events (the free space up to an event and the element there),
/// and [`WaveJob::finish`] reads the detector. However the work is sliced,
/// the result is the same to the bit: [`wave_path`] is this job run in one
/// step.
///
/// The job keeps its own copy of the scene, so it answers for the revision
/// it was started at whatever happens to the scene meanwhile.
pub struct WaveJob {
    scene: Scene3,
    spec: WavePathSpec,
    revision: u64,
    lam: f64,
    /// Each event: where the field reaches it, where it leaves from (the
    /// same point except after a transfer), and what it is.
    points: Vec<(Vec3, Vec3, Ev)>,
    next: usize,
    beam: Beam,
    u: Field2,
    report: WaveReport,
    fft: Fft2,
}

impl std::fmt::Debug for WaveJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveJob")
            .field("revision", &self.revision)
            .field("events", &self.points.len())
            .field("done", &self.next)
            .finish_non_exhaustive()
    }
}

impl WaveJob {
    /// Plan the chain from `spec.source` to `spec.detector` and launch the
    /// beam at the source.
    pub fn new(scene: &Scene3, spec: &WavePathSpec, revision: u64) -> Result<Self, WaveError> {
        if !(spec.power.is_finite() && spec.power > 0.0 && spec.window_mm > 0.0 && spec.grid_n >= 8) {
            return Err(WaveError::OutOfRange("a wave path needs power, a window and at least 8 samples".into()));
        }
        let seq = oc_trace::extract_sequence3(scene, spec.source, spec.detector, revision)
            .map_err(|e| WaveError::OutOfRange(format!("no chain from the source to the detector: {e:?}")))?;
        let src = scene
            .sources
            .iter()
            .find(|s| s.id == spec.source)
            .ok_or_else(|| WaveError::OutOfRange("no such source".into()))?;
        // A coherent field has one wavelength. Lines summed incoherently, or
        // a continuum sampled into lines, are not done yet: refused by name
        // rather than run at whichever wavelength the chief ray drew.
        match &src.spectrum {
            oc_core::scene::Spectrum::Monochromatic(_) => {}
            oc_core::scene::Spectrum::Lines(v) if v.len() == 1 => {}
            oc_core::scene::Spectrum::Lines(v) => {
                return Err(WaveError::OutOfRange(format!(
                    "source #{} emits {} lines; the wave path is coherent at one wavelength (summing lines is not done yet): give it a single line",
                    spec.source.0,
                    v.len()
                )))
            }
            oc_core::scene::Spectrum::Continuum(_) => {
                return Err(WaveError::OutOfRange(format!(
                    "source #{} emits a continuous spectrum; the wave path is coherent at one wavelength: give it a single line",
                    spec.source.0
                )))
            }
        }
        let lam = seq.launch.lambda_um;
        let d0 = seq.launch.dir.normalize();
        let x0 = src.transform.rotate(Vec3::Y);
        let x0 = (x0 - d0 * d0.dot(x0)).normalize();
        let beam = Beam { origin: seq.launch.point, x: x0, y: d0.cross(x0), d: d0 };
        let n0 = seq.hits.first().map_or(scene.env.ambient_index, |h| h.n_before);
        let grid = Grid2::square(spec.grid_n, spec.window_mm / spec.grid_n as f64)?;
        let mut u = match spec.beam {
            BeamProfile::Gaussian { waist_radius_mm, waist_offset_mm } => {
                GaussianBeam { w0_mm: waist_radius_mm, lambda_um: lam, n: n0, power: spec.power, center: [0.0, 0.0] }
                    .sample(grid, -waist_offset_mm)?
            }
            BeamProfile::TopHat { radius_mm } => {
                let a = (spec.power / (std::f64::consts::PI * radius_mm * radius_mm)).sqrt();
                Field2::from_fn(grid, lam, |x, y| if x * x + y * y <= radius_mm * radius_mm { Cx::real(a) } else { Cx::ZERO })?
                    .in_medium(n0)?
            }
        };
        u.frame = beam.iso();
        let report = WaveReport {
            ledger: WaveLedger { launched: u.power(), ..Default::default() },
            ..Default::default()
        };

        // The chain's events. Each face of a body, and each curved surface, is
        // a transfer across the thin slab that holds the surface where the
        // beam meets it; between faces the field diffracts inside the glass.
        let mut chain: Vec<(Vec3, Vec3, Ev)> = Vec::new();
        let mut resume = seq.launch.point;
        for (k, h) in seq.hits.iter().enumerate() {
            let curved = h.kind == SeqHitKind::Surface
                && scene.surfaces.iter().any(|s| s.common.id == h.object && !matches!(s.geometry, SurfaceGeometry3::Plane { .. }));
            if h.kind != SeqHitKind::Body && !curved {
                chain.push((h.point, h.point, Ev::Hit(h.clone())));
                resume = h.point;
                continue;
            }
            let next_point = seq.hits.get(k + 1).map_or(seq.terminal.point, |n| n.point);
            let d_in = (h.point - resume).normalize();
            let d_out = (next_point - h.point).normalize();
            let probe = probe_scene(scene, h.object);
            let half = 0.5 * spec.window_mm;
            let (s_min, s_max) = slab(&probe, resume, d_in, h.point, d_out, half, lam);
            // A margin of a micrometre, so neither plane touches the surface.
            let entry = h.point + d_in * (s_min - 1e-3);
            let exit = h.point + d_out * (s_max + 1e-3);
            if (entry - resume).dot(d_in) <= 0.0 || (next_point - exit).dot(d_out) <= 0.0 {
                return Err(WaveError::OutOfRange(format!(
                    "#{} ({}): the surface where the beam meets it is too close to the next one on the path for a plane to fit between them",
                    h.object.0, h.label
                )));
            }
            chain.push((entry, exit, Ev::Transfer { hit: h.clone(), exit, d_out }));
            resume = exit;
        }
        chain.push((seq.terminal.point, seq.terminal.point, Ev::End));

        // Insert the aperture planes the chief ray crosses between events.
        let mut points: Vec<(Vec3, Vec3, Ev)> = Vec::new();
        let mut prev = seq.launch.point;
        for (p, after, ev) in chain {
            let seg = p - prev;
            let len = seg.length();
            if len > 0.0 {
                let dir = seg / len;
                let mut crossings: Vec<(f64, usize)> = Vec::new();
                for (ai, a) in scene.apertures.iter().enumerate() {
                    let t = &a.common.transform;
                    let o = t.transform_point(a.shape.plane_local.origin);
                    let n = t.rotate(a.shape.plane_local.normal);
                    if let Some(x) = onto_plane(prev, dir, o, n) {
                        let s = (x - prev).dot(dir);
                        // One at the event's own plane is applied just before it.
                        if s > 1e-9 && s <= len + 1e-9 {
                            crossings.push((s, ai));
                        }
                    }
                }
                crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (s, ai) in crossings {
                    let q = prev + dir * s;
                    points.push((q, q, Ev::Aperture(ai)));
                }
            }
            // An aperture inside a face's slab has no place in the field's
            // steps: refused by name rather than skipped.
            if let Ev::Transfer { hit, .. } = &ev {
                for (from, to) in [(p, hit.point), (hit.point, after)] {
                    let len = (to - from).length();
                    if len <= 0.0 {
                        continue;
                    }
                    let dir = (to - from) / len;
                    for a in &scene.apertures {
                        let t = &a.common.transform;
                        let o = t.transform_point(a.shape.plane_local.origin);
                        let n = t.rotate(a.shape.plane_local.normal);
                        if let Some(x) = onto_plane(from, dir, o, n) {
                            let s = (x - from).dot(dir);
                            if s > 1e-9 && s < len - 1e-9 {
                                return Err(WaveError::OutOfRange(format!(
                                    "aperture #{} lies inside the slab the field is traced across at #{} ({}): move it clear of the surface",
                                    a.common.id.0, hit.object.0, hit.label
                                )));
                            }
                        }
                    }
                }
            }
            points.push((p, after, ev));
            prev = after;
        }
        Ok(WaveJob {
            scene: scene.clone(),
            spec: *spec,
            revision,
            lam,
            points,
            next: 0,
            beam,
            u,
            report,
            fft: Fft2::new(),
        })
    }

    /// The scene revision the job answers for.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The chain's events (surfaces, aperture planes, the detector), and how
    /// many the field has been carried through.
    pub fn events(&self) -> usize {
        self.points.len()
    }
    pub fn done(&self) -> usize {
        self.next
    }

    /// Carry the field through up to `events` more events. `Ok(true)` once
    /// the whole chain is done; an element the field has no model for is an
    /// error naming it, and the job is not resumable after one.
    pub fn step(&mut self, events: usize) -> Result<bool, WaveError> {
        let end = self.next.saturating_add(events).min(self.points.len());
        while self.next < end {
            self.event(self.next)?;
            self.next += 1;
        }
        Ok(self.next == self.points.len())
    }

    fn next_dir(&self, i: usize, here: Vec3, current: Vec3) -> Vec3 {
        self.points.get(i + 1).map_or(current, |(p, _, _)| (*p - here).normalize())
    }

    fn tilt_warn(&mut self, depth: f64) {
        let dx = self.u.grid.pitch_mm[0];
        if depth > dx * dx / self.u.wavelength_in_medium_mm() {
            self.report.warn(SamplingWarning::TiltProjection { depth_mm: depth });
        }
    }

    fn event(&mut self, idx: usize) -> Result<(), WaveError> {
        let (p, _, ev) = self.points[idx].clone();
        let lam = self.lam;
        // A diffracted readout takes the last step itself, onto the detector.
        if matches!(ev, Ev::End) && matches!(self.spec.readout, Readout::Diffracted { .. }) {
            return Ok(());
        }
        // Free space up to this event.
        let z = (p - self.beam.origin).dot(self.beam.d);
        if z > 1e-12 {
            let (v, r) = propagate(&self.u, z, Propagator::Auto, &mut self.fft)?;
            merge(&mut self.report, r);
            self.u = v;
            self.beam.origin = p;
            self.u.frame = self.beam.iso();
        }
        let beam = self.beam;
        match ev {
            Ev::End => {}
            Ev::Transfer { hit, exit, d_out } => {
                let first = &hit;
                let last = &hit;
                // The scene with the element alone: nothing else can meet its rays.
                let probe = probe_scene(&self.scene, hit.object);
                let out = beam.turned(exit, d_out);
                let before = self.u.power();
                let rays = self.spec.grid_n.clamp(65, 257);
                let (v, st) = crate::transfer::transfer(&probe, &self.u, &beam, &out, last.n_after, rays)?;
                self.report.ledger.absorbed_element += st.stopped + st.reflected;
                self.report.ledger.left_window += st.left_window;
                let what = if first.kind == SeqHitKind::Body { "face of body" } else { "curved surface" };
                self.report.steps.push(element_step(
                    0.0,
                    format!(
                        "{what} #{} ({}): field traced through it along {} rays ({} passed), chief path {:.6} mm; rays move up to {:.4} mm across the beam; {:.3e} of the power stopped outside the traced aperture, {:.3e} reflected by the faces",
                        first.object.0,
                        first.label,
                        st.rays,
                        st.passed,
                        st.opl_chief,
                        st.walk_off_mm,
                        st.stopped / before.max(f64::MIN_POSITIVE),
                        st.reflected / before.max(f64::MIN_POSITIVE),
                    ),
                ));
                self.report.approx(format!(
                    "{what} #{}: geometric field tracing (no diffraction inside the element; rays traced along the arriving wavefront's fitted sphere, curvature {:.4e} /mm, which its rays depart from by up to {:.3} mrad; unpolarized transmittance of the faces)",
                    first.object.0,
                    st.curvature,
                    1e3 * st.arriving_ray_angle
                ));
                if st.max_phase_step > std::f64::consts::PI {
                    self.report.warn(SamplingWarning::Undersampled {
                        what: format!("the field after {what} #{}", first.object.0),
                        max_phase_step_rad: st.max_phase_step,
                    });
                }
                self.u = v;
                self.beam = out;
                return Ok(());
            }
            Ev::Aperture(ai) => {
                let a = &self.scene.apertures[ai];
                let t = &a.common.transform;
                let plane = &a.shape.plane_local;
                let n_e = t.rotate(plane.normal);
                // Square to the beam, the opening is a shape in the beam's own
                // plane: exact cell areas, as for any aperture (elements.rs).
                if n_e.dot(beam.d).abs() > 1.0 - 1e-12 {
                    let to_beam = |pu: f64, pv: f64| {
                        let w = t.transform_point(plane.origin + plane.u * pu + plane.v * pv);
                        let (x, y) = beam.coords(w);
                        [x, y]
                    };
                    let c = to_beam(0.0, 0.0);
                    let shape = match &a.shape.clear_opening {
                        PlanarShape2::Disc { radius_mm } => crate::elements::Shape::Disc { c, r: *radius_mm },
                        PlanarShape2::Annulus { inner_mm, outer_mm } => {
                            crate::elements::Shape::Annulus { c, r_in: *inner_mm, r_out: *outer_mm }
                        }
                        PlanarShape2::Rectangle { half_width_mm, half_height_mm } => {
                            let (w, h) = (*half_width_mm, *half_height_mm);
                            crate::elements::Shape::Polygon(vec![to_beam(-w, -h), to_beam(w, -h), to_beam(w, h), to_beam(-w, h)])
                        }
                        PlanarShape2::Polygon { vertices } => {
                            crate::elements::Shape::Polygon(vertices.iter().map(|v| to_beam(v.x, v.y)).collect())
                        }
                    };
                    let mut step = WaveReport::default();
                    crate::elements::apply_aperture(&mut self.u, &shape, false, &mut step)?;
                    merge(&mut self.report, step);
                    return Ok(());
                }
                let before = self.u.power();
                let dx = self.u.grid.pitch_mm[0];
                let factor = |r: Vec3| -> Cx {
                    // 3 × 3 sub-samples across the sample cell, in the beam plane.
                    let mut inside = 0.0;
                    for sj in [-1.0, 0.0, 1.0] {
                        for si in [-1.0, 0.0, 1.0] {
                            let q = r + beam.x * (si * dx / 3.0) + beam.y * (sj * dx / 3.0);
                            let local = t.inverse_transform_point(q) - plane.origin;
                            let (pu, pv) = (local.dot(plane.u), local.dot(plane.v));
                            if shape_contains(&a.shape.clear_opening, pu, pv) {
                                inside += 1.0;
                            }
                        }
                    }
                    Cx::real(inside / 9.0)
                };
                let (v, depth) = hand_over(&self.u, &beam, n_e, &beam, &factor);
                let note = element_step(0.0, format!("aperture #{} at its plane", a.common.id.0));
                self.tilt_warn(depth);
                self.report.ledger.absorbed_aperture += (before - v.power()).max(0.0);
                self.report.steps.push(note);
                self.u = v;
            }
            Ev::Hit(h) => {
                let si = self
                    .scene
                    .surfaces
                    .iter()
                    .position(|s| s.common.id == h.object)
                    .ok_or_else(|| WaveError::OutOfRange("a chain surface is missing".into()))?;
                let s = &self.scene.surfaces[si];
                let n_s = s.common.transform.rotate(Vec3::Z);
                let from_front = beam.d.dot(n_s) < 0.0;
                let response = if from_front { &s.response_front } else { &s.response_back };
                let d2 = self.next_dir(idx, p, beam.d);
                let out = beam.turned(p, d2);
                let before = self.u.power();
                let amp: Option<f64> = match (response, h.action) {
                    (SurfaceResponse::Mirror { reflectivity }, SeqAction::Reflect) => {
                        reflectivity.sample(lam).map(|r| r.clamp(0.0, 1.0).sqrt())
                    }
                    (SurfaceResponse::ExplicitRt { reflectance, .. }, SeqAction::Reflect) => {
                        reflectance.sample(lam).map(|r| r.clamp(0.0, 1.0).sqrt())
                    }
                    (SurfaceResponse::ExplicitRt { transmittance, .. }, SeqAction::Transmit) => {
                        transmittance.sample(lam).map(|t| t.clamp(0.0, 1.0).sqrt())
                    }
                    (SurfaceResponse::Grating { efficiency, .. }, SeqAction::Diffract) => {
                        efficiency.sample(lam).map(|e| e.clamp(0.0, 1.0).sqrt())
                    }
                    (SurfaceResponse::Grating { zero_order, .. }, _) => {
                        zero_order.sample(lam).map(|e| e.clamp(0.0, 1.0).sqrt())
                    }
                    _ => None,
                };
                if let Some(a) = amp {
                    let (v, depth) = hand_over(&self.u, &beam, n_s, &out, &|_| Cx::real(a));
                    // A flat mirror's handover is image theory, exact at any
                    // tilt; anything else seen obliquely is a projection.
                    let exact = matches!(response, SurfaceResponse::Mirror { .. }) && h.action == SeqAction::Reflect;
                    let note = element_step(0.0, format!("{:?} at #{} ({}), amplitude {a:.6}", h.action, h.object.0, h.label));
                    if !exact {
                        self.tilt_warn(depth);
                    }
                    self.report.ledger.absorbed_element += before * (1.0 - a * a);
                    self.report.steps.push(note);
                    self.u = v;
                } else {
                    match response {
                        SurfaceResponse::ThinLens { focal_length_mm } => {
                            let f = *focal_length_mm;
                            let c = s.common.transform.pos;
                            let k = self.u.k_mm();
                            let lens = move |r: Vec3| {
                                // Distance from the lens's centre within its plane.
                                let v = r - c;
                                let v = v - n_s * v.dot(n_s);
                                Cx::expi(-k * v.length_squared() / (2.0 * f))
                            };
                            let (v, depth) = hand_over(&self.u, &beam, n_s, &out, &lens);
                            self.tilt_warn(depth);
                            let dx = self.u.grid.pitch_mm[0];
                            let edge = 0.5 * self.u.grid.width_mm(0) * 2f64.sqrt();
                            if k * edge * dx / f.abs() > std::f64::consts::PI {
                                self.report.warn(SamplingWarning::Undersampled {
                                    what: format!("thin lens #{} f = {f} mm", h.object.0),
                                    max_phase_step_rad: k * edge * dx / f.abs(),
                                });
                            }
                            self.report
                                .steps
                                .push(element_step(0.0, format!("thin lens #{} ({}), f = {f} mm", h.object.0, h.label)));
                            self.u = v;
                        }
                        SurfaceResponse::Modulator { device, .. } => {
                            self.u = modulator_step(&self.u, &beam, &out, s, device, &h, &self.spec, &mut self.report)?;
                        }
                        other => {
                            let msg = format!(
                                "{} at #{} ({}): no wave model for this response on the path (the field is scalar)",
                                oc_core::build3::response_kind(other),
                                h.object.0,
                                h.label
                            );
                            self.report.rejected.push(msg.clone());
                            return Err(WaveError::OutOfRange(msg));
                        }
                    }
                }
                self.beam = out;
                self.u.frame = out.iso();
                self.u.n_medium = h.n_after;
            }
        }
        Ok(())
    }

    /// Read the detector: each pixel's power, from the field projected onto
    /// its face. Refused before the chain is done.
    pub fn finish(mut self) -> Result<WavePathResult, WaveError> {
        if self.next < self.points.len() {
            return Err(WaveError::OutOfRange(format!(
                "the wave job has {} of {} events still to run",
                self.points.len() - self.next,
                self.points.len()
            )));
        }
        let spec = self.spec;
        let beam = self.beam;
        let det = self
            .scene
            .detectors
            .iter()
            .find(|d| d.common.id.0 == spec.detector.0)
            .ok_or_else(|| WaveError::OutOfRange("no such detector".into()))?;
        let (hw, hh) = match det.geometry {
            DetectorGeometry3::Rectangle { half_width_mm, half_height_mm } => (half_width_mm, half_height_mm),
            _ => return Err(WaveError::OutOfRange("the wave readout needs a rectangular detector".into())),
        };
        let [nu, nv] = det.bins;
        let t = &det.common.transform;
        let n_det = t.rotate(Vec3::Z);
        let cos = beam.d.dot(n_det).abs();
        let (du, dv) = (2.0 * hw / nu as f64, 2.0 * hh / nv as f64);
        if let Readout::Diffracted { samples_per_pixel } = spec.readout {
            return self.finish_diffracted(samples_per_pixel, (hw, hh), [nu, nv], (du, dv));
        }
        // A detector integrates power, so the readout bins it: each field
        // cell's power |U|²ΔA goes to the pixel its projection along the beam
        // lands in, as `s × s` sub-cells so that each, stretched by 1/cos on
        // a tilted face, spans at most an eighth of a pixel (a cell snapped
        // whole to a pixel would move the light by up to half a cell). Power
        // is conserved to round-off; what misses the face left the window.
        let g = self.u.grid;
        let o = t.pos;
        let cell = g.pitch_mm[0].max(g.pitch_mm[1]) / cos.max(1e-6);
        let s = ((8.0 * cell / du.min(dv)).ceil() as usize).clamp(1, 16);
        let share = g.cell_area_mm2() / (s * s) as f64;
        let mut power = vec![0.0; nu * nv];
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let p = self.u.data[g.idx(i, j)].norm_sqr() * share;
                if p == 0.0 {
                    continue;
                }
                let (x, y) = g.xy(i, j);
                for sj in 0..s {
                    for si in 0..s {
                        let a = x + ((si as f64 + 0.5) / s as f64 - 0.5) * g.pitch_mm[0];
                        let b = y + ((sj as f64 + 0.5) / s as f64 - 0.5) * g.pitch_mm[1];
                        let Some(r) = onto_plane(beam.point(a, b), beam.d, o, n_det) else {
                            continue;
                        };
                        let l = t.inverse_transform_point(r);
                        if l.x.abs() > hw || l.y.abs() > hh {
                            continue;
                        }
                        let iu = (((l.x + hw) / du) as usize).min(nu - 1);
                        let iv = (((l.y + hh) / dv) as usize).min(nv - 1);
                        power[iv * nu + iu] += p;
                    }
                }
            }
        }
        let bins = det.bins;
        let on_detector: f64 = power.iter().sum();
        let in_face = on_detector;
        let arriving = self.u.power();
        self.report.ledger.left_window += (arriving - in_face).max(0.0);
        let arriving_power = arriving;
        self.report.ledger.detected = on_detector;
        self.report.ledger.close();
        Ok(WavePathResult {
            revision: self.revision,
            detector: spec.detector,
            bins,
            power,
            field: self.u,
            arriving: arriving_power,
            report: self.report,
        })
    }
}

impl WaveJob {
    /// The diffracted readout: the field carried from where it is straight
    /// onto a window of the plane across the beam at the detector's centre,
    /// at `s × s` points a pixel, then binned into the pixels.
    fn finish_diffracted(
        mut self,
        s: usize,
        (hw, hh): (f64, f64),
        [nu, nv]: [usize; 2],
        (du, dv): (f64, f64),
    ) -> Result<WavePathResult, WaveError> {
        let spec = self.spec;
        let beam = self.beam;
        let det = self
            .scene
            .detectors
            .iter()
            .find(|d| d.common.id.0 == spec.detector.0)
            .ok_or_else(|| WaveError::OutOfRange("no such detector".into()))?;
        let t = &det.common.transform;
        let n_det = t.rotate(Vec3::Z);
        let z = (t.pos - beam.origin).dot(beam.d);
        if z <= 0.0 {
            return Err(WaveError::OutOfRange("the detector is behind the field's last plane".into()));
        }
        // A face that is not square to the beam: the spectrum rotated onto the
        // face's own plane (Matsushima, Schimmel & Wyrowski 2003), on its own
        // subpixel grid; each pixel its mean irradiance over its area.
        if n_det.dot(beam.d).abs() <= 1.0 - 1e-12 {
            let s = s.max(1);
            let grid = centred(Grid2::new([nu * s, nv * s], [du / s as f64, dv / s as f64], [0.0, 0.0])?);
            // The rotation reads the spectrum between its samples: pad finer
            // where the field is small enough to afford it.
            let n_max = self.u.grid.n[0].max(self.u.grid.n[1]);
            let pad = if n_max <= 512 { 4 } else if n_max <= 1024 { 2 } else { 1 };
            let arriving = self.u.power();
            let (zf, mut rep) = crate::asm_tilted(&self.u, *t, grid, pad, &mut self.fft)?;
            let cos = n_det.dot(beam.d).abs();
            let mut power = vec![0.0; nu * nv];
            for j in 0..nv * s {
                for i in 0..nu * s {
                    power[(j / s) * nu + i / s] += zf.data[grid.idx(i, j)].norm_sqr();
                }
            }
            for v in &mut power {
                *v *= du * dv * cos / (s * s) as f64;
            }
            let on_detector: f64 = power.iter().sum();
            rep.ledger.left_window = (arriving - on_detector).max(0.0);
            rep.ledger.detected = on_detector;
            merge(&mut self.report, rep);
            self.report.ledger.detected = on_detector;
            self.report.ledger.close();
            self.u = zf;
            return Ok(WavePathResult { revision: self.revision, detector: spec.detector, bins: [nu, nv], power, field: self.u, arriving, report: self.report });
        }
        // The face's corners, projected along the beam onto the plane across
        // it at the detector's centre, give the window.
        let plane = beam.origin + beam.d * z;
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        let mut depth: f64 = 0.0;
        for (u, v) in [(-hw, -hh), (hw, -hh), (-hw, hh), (hw, hh)] {
            let r = t.transform_point(Vec3::new(u, v, 0.0));
            depth = depth.max((r - plane).dot(beam.d).abs());
            let (a, b) = beam.coords(r - beam.d * (r - plane).dot(beam.d));
            lo = [lo[0].min(a), lo[1].min(b)];
            hi = [hi[0].max(a), hi[1].max(b)];
        }
        // A face square to the beam with its axes along the beam's is read
        // on its own subpixel grid, `s × s` points at the centres of equal
        // parts of each pixel (one point: the pixel's centre). Any other face
        // is read on a grid along the beam's axes, each pixel taking the mean
        // of the points that fall in it.
        let s = s.max(1);
        let (eu, ev) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
        let square = n_det.dot(beam.d).abs() > 1.0 - 1e-12;
        let along = |a: Vec3, b: Vec3| a.dot(b).abs() > 1.0 - 1e-12;
        // The face's axes along the beam's, either way round.
        let aligned = if square && along(eu, beam.x) && along(ev, beam.y) {
            Some(([nu * s, nv * s], [du / s as f64, dv / s as f64]))
        } else if square && along(eu, beam.y) && along(ev, beam.x) {
            Some(([nv * s, nu * s], [dv / s as f64, du / s as f64]))
        } else {
            None
        };
        let (out, pitch_a, pitch_b) = if let Some((n, pitch)) = aligned {
            let c = beam.coords(t.pos - beam.d * (t.pos - plane).dot(beam.d));
            (centred(Grid2::new(n, pitch, [c.0, c.1])?), pitch[0], pitch[1])
        } else {
            let pitch = du.min(dv) / s as f64;
            let m = [
                (((hi[0] - lo[0]) / pitch).ceil() as usize).clamp(1, 4096),
                (((hi[1] - lo[1]) / pitch).ceil() as usize).clamp(1, 4096),
            ];
            (Grid2::new(m, [pitch, pitch], [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])])?, pitch, pitch)
        };
        let pitch = pitch_a.min(pitch_b);
        let m = out.n;
        // Without padding the field repeats a window away, which a beam
        // converging onto the detector never reaches; small fields are padded.
        let pad = if self.u.grid.n[0].max(self.u.grid.n[1]) >= 1024 { 1 } else { 2 };
        let arriving = self.u.power();
        let (zf, mut rep) = crate::asm_zoom(&self.u, z, out, pad, &mut self.fft)?;
        if depth > pitch * pitch / zf.wavelength_in_medium_mm() {
            self.report.warn(SamplingWarning::TiltProjection { depth_mm: depth });
        }
        // Each pixel: the mean irradiance of its points, over its area seen
        // across the beam.
        let mut sum = vec![0.0; nu * nv];
        let mut count = vec![0usize; nu * nv];
        for j in 0..m[1] {
            for i in 0..m[0] {
                let (a, b) = out.xy(i, j);
                let r = plane + beam.x * a + beam.y * b;
                let Some(q) = onto_plane(r, beam.d, t.pos, n_det) else { continue };
                let l = t.inverse_transform_point(q);
                if l.x.abs() > hw || l.y.abs() > hh {
                    continue;
                }
                let iu = (((l.x + hw) / du) as usize).min(nu - 1);
                let iv = (((l.y + hh) / dv) as usize).min(nv - 1);
                sum[iv * nu + iu] += zf.data[out.idx(i, j)].norm_sqr();
                count[iv * nu + iu] += 1;
            }
        }
        let cos = n_det.dot(beam.d).abs();
        if count.contains(&0) {
            self.report.warn(SamplingWarning::Undersampled {
                what: format!("the diffracted readout of detector #{}: a pixel got no sample", spec.detector.0),
                max_phase_step_rad: 0.0,
            });
        }
        let power: Vec<f64> =
            sum.iter().zip(&count).map(|(s, &c)| if c > 0 { s / c as f64 * du * dv * cos } else { 0.0 }).collect();
        let on_detector: f64 = power.iter().sum();
        // What the window held but the face did not left the window too.
        rep.ledger.left_window += (zf.power() - on_detector).max(0.0);
        merge(&mut self.report, rep);
        self.report.ledger.detected = on_detector;
        self.report.ledger.close();
        self.u = zf;
        self.u.frame = beam.turned(plane, beam.d).iso();
        Ok(WavePathResult {
            revision: self.revision,
            detector: spec.detector,
            bins: [nu, nv],
            power,
            field: self.u,
            arriving,
            report: self.report,
        })
    }
}

/// Carry a coherent beam from `spec.source` to `spec.detector` along the
/// chief ray's chain: a [`WaveJob`] run in one step.
pub fn wave_path(scene: &Scene3, spec: &WavePathSpec, revision: u64) -> Result<WavePathResult, WaveError> {
    let mut job = WaveJob::new(scene, spec, revision)?;
    job.step(usize::MAX)?;
    job.finish()
}

/// Lay the beam onto a DMD or SLM, reflect it, and take it up across the
/// outgoing ray.
#[allow(clippy::too_many_arguments)]
fn modulator_step(
    u: &Field2,
    beam: &Beam,
    out: &Beam,
    s: &oc_core::scene3::SurfaceObject3,
    device: &oc_core::modulator::SpatialModulator,
    h: &oc_core::seq3::SeqHit3,
    spec: &WavePathSpec,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    let t = &s.common.transform;
    let n_s = t.rotate(Vec3::Z);
    // The device-plane grid: the beam window's footprint, at a few samples a
    // pixel.
    let p = device.pitch_mm();
    let per = spec.device_samples_per_pixel.max(2);
    let dxd = p / per as f64;
    let half = 0.5 * u.grid.width_mm(0);
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for (a, b) in [(-half, -half), (half, -half), (-half, half), (half, half)] {
        if let Some(r) = onto_plane(beam.point(a, b), beam.d, beam.origin, n_s) {
            let l = t.inverse_transform_point(r);
            lo = [lo[0].min(l.x), lo[1].min(l.y)];
            hi = [hi[0].max(l.x), hi[1].max(l.y)];
        }
    }
    let n = [((hi[0] - lo[0]) / dxd).ceil() as usize + 2, ((hi[1] - lo[1]) / dxd).ceil() as usize + 2];
    let center = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let g = Grid2::new(n, [dxd, dxd], center)?;
    // The incident field on the device: data from the beam plane, carrier the
    // beam's direction seen in the device frame.
    let d_local = t.quat().inverse() * beam.d;
    let mut on_dev = Field2::zeros(g, u.lambda_um)?.in_medium(u.n_medium)?;
    on_dev.carrier = [d_local.x, d_local.y];
    let k = u.k_mm();
    let smooth = Smooth::new(u);
    for j in 0..n[1] {
        for i in 0..n[0] {
            let (x, y) = g.xy(i, j);
            let r = t.transform_point(Vec3::new(x, y, 0.0));
            let r_in = r - beam.d * (r - beam.origin).dot(beam.d);
            let (a, b) = beam.coords(r_in);
            // The beam's reference wave, exp(ik d·(r − o)), is the carrier
            // plus a constant on the device plane; take the constant out.
            let c = Cx::expi(k * (beam.d.dot(r - beam.origin) - (d_local.x * x + d_local.y * y)));
            on_dev.data[g.idx(i, j)] = smooth.at(a, b) * c;
        }
    }
    let before = u.power();
    let (reflected, rep) = match &device.kind {
        ModulatorKind::Dmd(_) => {
            // The chain followed the family of the pixel the chief ray hit.
            let l = t.inverse_transform_point(h.point);
            let p = device.pitch_mm();
            let fi = ((l.x / p + device.pixels[0] as f64 / 2.0).floor().max(0.0) as usize).min(device.pixels[0] as usize - 1);
            let fj = ((l.y / p + device.pixels[1] as f64 / 2.0).floor().max(0.0) as usize).min(device.pixels[1] as usize - 1);
            dmd_reflect(&on_dev, device, DmdState::from_value(device.value(fi, fj)))?
        }
        ModulatorKind::Lcos(_) => {
            let beam_kind = if h.action == SeqAction::Diffract { LcosBeam::Modulated } else { LcosBeam::Unmodulated };
            report.approx("LCoS on the wave path: the beam's polarization split is not applied (the field is scalar); the chosen arm takes the full amplitude");
            lcos_reflect(&on_dev, device, beam_kind)?
        }
    };
    // Take the reflected field up across the outgoing ray. Its samples are
    // the incident ones times a piston per pixel, so the incident field's
    // sphere is the one to take out of them.
    let ([c_dev, a_dev, b_dev], _) = fit_sphere(&on_dev);
    let smooth_out = Smooth::with(&reflected, [c_dev, a_dev - on_dev.carrier[0], b_dev - on_dev.carrier[1]]);
    let mut v = u.clone();
    let gb = u.grid;
    let mut depth: f64 = 0.0;
    let ob = obliquity(beam.d, out.d, n_s);
    for j in 0..gb.n[1] {
        for i in 0..gb.n[0] {
            let (a, b) = gb.xy(i, j);
            let r_out = out.point(a, b);
            let val = match onto_plane(r_out, out.d, out.origin, n_s) {
                Some(r_e) => {
                    depth = depth.max((r_out - r_e).length());
                    let l = t.inverse_transform_point(r_e);
                    // Device-plane data carries the outgoing family's
                    // carrier; across the outgoing ray that carrier is the
                    // reference wave, and the constant is put back.
                    let c = Cx::expi(-k * (out.d.dot(r_e - out.origin) - (reflected.carrier[0] * l.x + reflected.carrier[1] * l.y)));
                    smooth_out.at(l.x, l.y) * c * ob
                }
                None => Cx::ZERO,
            };
            v.data[gb.idx(i, j)] = val;
        }
    }
    if depth > gb.pitch_mm[0] * gb.pitch_mm[0] / u.wavelength_in_medium_mm() {
        report.warn(SamplingWarning::TiltProjection { depth_mm: depth });
    }
    // The device's own ledger, scaled to this beam; resampling closes the rest.
    let scale = if rep.ledger.launched > 0.0 { before / rep.ledger.launched } else { 0.0 };
    report.ledger.absorbed_element += rep.ledger.absorbed_element * scale;
    report.ledger.other_family += rep.ledger.other_family * scale;
    report.ledger.beyond_nyquist += rep.ledger.beyond_nyquist * scale;
    for f in rep.validity {
        report.flag(f);
    }
    for a in rep.approximated {
        report.approx(a);
    }
    report.steps.push(element_step(
        0.0,
        format!("{} #{} ({}): laid on {}×{} device samples, {} per pixel", oc_core::build3::response_kind(&s.response_front), h.object.0, h.label, n[0], n[1], per),
    ));
    Ok(v)
}
