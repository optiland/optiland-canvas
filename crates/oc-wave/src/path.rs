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
//! - **An element's outline.** A flat element's own shape clips the field,
//!   its edge weighted by exact cell coverage. Light outside it passes the
//!   element by: still on the chain where the chain goes straight on, off it
//!   where the chain turns.
//! - **Aperture.** Its opening and outer extent, as the tracer has them: light
//!   passes inside the opening, is stopped between it and the outer extent,
//!   and passes beyond. Each edge is weighted by exact cell coverage, of the
//!   shape itself square to the beam and of its projection along the beam
//!   otherwise. The tracer logs no event when a ray passes an opening, so the
//!   path finds every aperture by intersecting the chief ray with each
//!   aperture's plane.
//! - **Refractive body or curved surface.** Traced through face by face
//!   ([`crate::transfer`]), its rim laid on the field the same way.
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
//! - **Plane-parallel plate, flat coating.** Plane wave by plane wave in the
//!   field's angular spectrum ([`crate::planar`]), s or p.
//! - **Polarizer, retarder.** The tracer's Jones matrix at the chief ray: the
//!   element's faces, its layer on its own axis projected across the ray in
//!   its glass, the faces again.
//!
//! **Polarization.** A chain that meets an element telling polarizations
//! apart (a polarizer, a retarder, an LCoS, a flat coating or a plate met
//! obliquely) carries two fields, the components along its beam frame's `x`
//! and `y` (R. C. Jones, *J. Opt. Soc. Am.* 31, 488 (1941), sample by
//! sample). At an element the pair is turned into the element's s and p, each
//! carried through as a field of that polarization alone, and turned into the
//! outgoing frame; an unpolarized source is two such launches, along `x` and
//! along `y`, mutually incoherent and half its power each. Arms add component
//! by component, so orthogonally polarized arms do not interfere. Other
//! chains carry one field standing for unpolarized light.
//!
//! Refused, by name: measured tables (no phase), and anything else the field
//! has no model for. Nothing is skipped silently.

use crate::beams::GaussianBeam;
use crate::modulator::{dmd_reflect, lcos_reflect};
use crate::propagate::propagate;
use crate::report::{SamplingWarning, StepReport, WaveLedger};
use crate::planar::{Plate, Pol};
use crate::transfer::{fit_sphere, Reference, Smooth, TracedMap};
use crate::{Cx, Fft2, Field2, Grid2, Method, Propagator, WaveError, WaveReport};
use oc_core::id::{DetectorId, SourceId};
use oc_core::modulator::{DmdState, LcosBeam, ModulatorKind};
use oc_core::scene3::{DetectorGeometry3, PlanarShape2, Scene3, SurfaceGeometry3, SurfaceResponse};
use oc_core::seq3::{SeqAction, SeqHitKind};
use oc_core::{Isometry3, Vec3};
use std::sync::Arc;

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
    /// The field arriving at the detector, across the chief ray: of a field of
    /// two polarization components (a path with an element that tells them
    /// apart), the one along the beam frame's `x`, and `field_y` the other.
    pub field: Field2,
    pub field_y: Option<Field2>,
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

/// The covered fraction of each of `grid`'s cells by `shape`, a shape in an
/// element's plane, seen across the beam: `to_beam` takes a point of that
/// plane (its local mm) along the beam to the beam's plane. Square to the
/// beam, a disc or an annulus stays one and is covered by its closed form;
/// otherwise every shape is a polygon there (a disc one of
/// [`crate::elements::disc_sides`] sides with the disc's own area), covered
/// exactly. Either way the edge is resolved to the field's own sample.
fn shape_coverage(grid: &Grid2, shape: &PlanarShape2, square: bool, to_beam: &dyn Fn(f64, f64) -> [f64; 2]) -> Result<Vec<f64>, WaveError> {
    use crate::elements::{coverage, disc_polygon, disc_sides, Shape};
    let pitch = grid.pitch_mm[0].min(grid.pitch_mm[1]);
    let ring = |r: f64| -> Vec<[f64; 2]> { disc_polygon([0.0, 0.0], r, disc_sides(r, pitch)).iter().map(|p| to_beam(p[0], p[1])).collect() };
    let c = to_beam(0.0, 0.0);
    match shape {
        PlanarShape2::Disc { radius_mm } if square => coverage(grid, &Shape::Disc { c, r: *radius_mm }),
        PlanarShape2::Annulus { inner_mm, outer_mm } if square => {
            coverage(grid, &Shape::Annulus { c, r_in: *inner_mm, r_out: *outer_mm })
        }
        PlanarShape2::Disc { radius_mm } => coverage(grid, &Shape::Polygon(ring(*radius_mm))),
        PlanarShape2::Annulus { inner_mm, outer_mm } => {
            let outer = coverage(grid, &Shape::Polygon(ring(*outer_mm)))?;
            let inner = if *inner_mm > 0.0 { coverage(grid, &Shape::Polygon(ring(*inner_mm)))? } else { vec![0.0; outer.len()] };
            Ok(outer.iter().zip(&inner).map(|(o, i)| (o - i).clamp(0.0, 1.0)).collect())
        }
        PlanarShape2::Rectangle { half_width_mm, half_height_mm } => {
            let (w, h) = (*half_width_mm, *half_height_mm);
            coverage(grid, &Shape::Polygon(vec![to_beam(-w, -h), to_beam(w, -h), to_beam(w, h), to_beam(-w, h)]))
        }
        PlanarShape2::Polygon { vertices } => coverage(grid, &Shape::Polygon(vertices.iter().map(|v| to_beam(v.x, v.y)).collect())),
    }
}

/// Points around the edge of `shape`, in its plane: what must lie inside an
/// aperture's outer extent for the opening to be within it.
fn shape_rim(shape: &PlanarShape2) -> Vec<oc_core::Vec2> {
    use oc_core::Vec2;
    let circle = |r: f64| {
        (0..256).map(move |k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / 256.0;
            Vec2::new(r * a.cos(), r * a.sin())
        })
    };
    match shape {
        PlanarShape2::Disc { radius_mm } => circle(*radius_mm).collect(),
        PlanarShape2::Annulus { outer_mm, .. } => circle(*outer_mm).collect(),
        PlanarShape2::Rectangle { half_width_mm: w, half_height_mm: h } => {
            vec![Vec2::new(-w, -h), Vec2::new(*w, -h), Vec2::new(*w, *h), Vec2::new(-w, *h)]
        }
        PlanarShape2::Polygon { vertices } => vertices.clone(),
    }
}

/// The plane of an element seen across `beam`: where the line along the beam
/// through each corner of the window meets it, the largest depth from the
/// beam's plane, mm.
fn plane_depth(beam: &Beam, grid: &Grid2, o: Vec3, n: Vec3) -> f64 {
    let (hx, hy) = (0.5 * grid.width_mm(0), 0.5 * grid.width_mm(1));
    let mut depth: f64 = 0.0;
    for (a, b) in [(-hx, -hy), (hx, -hy), (-hx, hy), (hx, hy)] {
        let r = beam.point(a + grid.center_mm[0], b + grid.center_mm[1]);
        if let Some(q) = onto_plane(r, beam.d, o, n) {
            depth = depth.max((q - r).length());
        }
    }
    depth
}

fn merge(total: &mut WaveReport, step: WaveReport) {
    let l = step.ledger;
    let t = &mut total.ledger;
    t.absorbed_aperture += l.absorbed_aperture;
    t.absorbed_element += l.absorbed_element;
    t.band_limit_removed += l.band_limit_removed;
    t.spectral_projection += l.spectral_projection;
    t.subcell_projection += l.subcell_projection;
    t.evanescent += l.evanescent;
    t.left_window += l.left_window;
    t.other_family += l.other_family;
    t.untracked_orders += l.untracked_orders;
    t.other_arm += l.other_arm;
    t.passed_by += l.passed_by;
    t.resampling += l.resampling;
    t.other_ports += l.other_ports;
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
    for s in step.inconsistencies {
        if !total.inconsistencies.contains(&s) {
            total.inconsistencies.push(s);
        }
    }
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

/// A field's second moments, across its plane and in angle, per axis: its
/// centroid `x̄` and variance `σ²`, its mean direction `θ̄` (a direction
/// cosine, the carrier included) and angular variance `σ_θ²`, and the cross
/// moment `C = ⟨(x − x̄)θ⟩`, `θ = ∂φ/∂x / k` the local direction. Under
/// paraxial propagation a distance `z` they give the beam's size exactly,
/// `σ²(z) = σ² + 2zC + z²σ_θ²` (the second-moment law of the Wigner
/// distribution; A. E. Siegman, *Proc. SPIE* 1224, 2 (1990)), whatever the
/// beam's shape: what re-gridding needs to choose the window at the next
/// plane before the field is carried there.
struct Moments {
    power: f64,
    mean: [f64; 2],
    var: [f64; 2],
    tilt: [f64; 2],
    spread: [f64; 2],
    cross: [f64; 2],
}

impl Moments {
    fn of(u: &Field2, fft: &mut Fft2) -> Option<Moments> {
        let g = u.grid;
        let k = u.k_mm();
        let (mut p, mut mx, mut my) = (0.0, 0.0, 0.0);
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (x, y) = g.xy(i, j);
                let w = u.at(i, j).norm_sqr();
                p += w;
                mx += w * x;
                my += w * y;
            }
        }
        if p <= 0.0 || p.is_nan() {
            return None;
        }
        let (mx, my) = (mx / p, my / p);
        // The local direction θ = α + c·x + the rest: the fitted sphere and
        // tilt analytically, what they leave by the phase difference of
        // neighbouring samples (which a sphere turning most of a radian a
        // sample would make a finite difference understate).
        let ([c, al, be], _) = fit_sphere(u);
        let sm = Smooth::with(u, [c, al - u.carrier[0], be - u.carrier[1]]);
        let rest = |i: usize, j: usize| {
            let (x, y) = g.xy(i, j);
            sm.at(x, y) * Cx::expi(-k * (0.5 * c * (x * x + y * y) + (al - u.carrier[0]) * x + (be - u.carrier[1]) * y))
        };
        let (mut vx, mut vy, mut cx, mut cy) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let (x, y) = g.xy(i, j);
                let w = u.at(i, j).norm_sqr();
                if w == 0.0 {
                    continue;
                }
                vx += w * (x - mx).powi(2);
                vy += w * (y - my).powi(2);
                let tx = if i > 0 && i + 1 < g.n[0] { (rest(i + 1, j) * rest(i - 1, j).conj()).arg() / (2.0 * g.pitch_mm[0] * k) } else { 0.0 };
                let ty = if j > 0 && j + 1 < g.n[1] { (rest(i, j + 1) * rest(i, j - 1).conj()).arg() / (2.0 * g.pitch_mm[1] * k) } else { 0.0 };
                cx += (x - mx) * w * (al + c * x + tx);
                cy += (y - my) * w * (be + c * y + ty);
            }
        }
        // The angular moments from the field's spectrum.
        let n = g.n;
        let mut buf = u.data.clone();
        crate::fft::ifftshift2(&mut buf, n);
        fft.forward(&mut buf, n, g.cell_area_mm2());
        let lam = u.wavelength_in_medium_mm();
        let (u0, v0) = (u.carrier[0] / lam, u.carrier[1] / lam);
        let (mut q, mut tx, mut ty, mut sx, mut sy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for ky in 0..n[1] {
            let fy = g.fft_freq(1, ky) + v0;
            for kx in 0..n[0] {
                let fx = g.fft_freq(0, kx) + u0;
                let w = buf[ky * n[0] + kx].norm_sqr();
                q += w;
                tx += w * lam * fx;
                ty += w * lam * fy;
                sx += w * (lam * fx).powi(2);
                sy += w * (lam * fy).powi(2);
            }
        }
        let (tx, ty) = (tx / q, ty / q);
        Some(Moments {
            power: p,
            mean: [mx, my],
            var: [vx / p, vy / p],
            tilt: [tx, ty],
            spread: [(sx / q - tx * tx).max(0.0), (sy / q - ty * ty).max(0.0)],
            cross: [cx / p, cy / p],
        })
    }

    /// The moments of two orthogonal components together: power-weighted,
    /// the variances about the common centroid.
    fn with(&self, o: &Moments) -> Moments {
        let (a, b) = (self.power, o.power);
        let t = (a + b).max(f64::MIN_POSITIVE);
        let w = |x: f64, y: f64| (a * x + b * y) / t;
        let mean = [w(self.mean[0], o.mean[0]), w(self.mean[1], o.mean[1])];
        let tilt = [w(self.tilt[0], o.tilt[0]), w(self.tilt[1], o.tilt[1])];
        let var = [0, 1].map(|k| w(self.var[k] + (self.mean[k] - mean[k]).powi(2), o.var[k] + (o.mean[k] - mean[k]).powi(2)));
        let spread = [0, 1].map(|k| w(self.spread[k] + (self.tilt[k] - tilt[k]).powi(2), o.spread[k] + (o.tilt[k] - tilt[k]).powi(2)));
        let cross = [0, 1].map(|k| {
            w(self.cross[k] + (self.mean[k] - mean[k]) * (self.tilt[k] - tilt[k]), o.cross[k] + (o.mean[k] - mean[k]) * (o.tilt[k] - tilt[k]))
        });
        Moments { power: t, mean, var, tilt, spread, cross }
    }

    /// The rms radius along `axis`, `z` on.
    fn sigma_at(&self, axis: usize, z: f64) -> f64 {
        (self.var[axis] + 2.0 * z * self.cross[axis] + z * z * self.spread[axis]).max(0.0).sqrt()
    }
}

/// Where a chain's arm parts from others: the element, how the arms share
/// it, what met it on this arm and the fraction this arm carries on, and
/// what the chain booked there for the other arms' shares, which those arms
/// book themselves when they too reach the detector.
#[derive(Clone, Copy, Debug)]
struct Junction {
    object: u64,
    kind: JunctionKind,
    /// The bucket this arm books the shares it does not carry to.
    bucket: Bucket,
    /// The power that met the element on this arm.
    input: f64,
    /// The fraction of `input` this arm carries on (a splitter's R or T, a
    /// grating's η, a coating's R or T averaged over the field).
    share: f64,
    /// What this arm booked at the element for the other arms' shares
    /// (`PerBeam`).
    foreign_absorbed: f64,
    foreign_family: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JunctionKind {
    /// The arms meet the element with one field and part there: a
    /// splitter's two ports, a grating's orders.
    Split,
    /// Each arm meets the element with its own share of the field: an
    /// LCoS's two polarization beams.
    PerBeam,
}

/// The ledger bucket a junction's unfollowed shares go to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bucket {
    OtherArm,
    UntrackedOrders,
    OtherFamily,
}

fn bucket_mut(l: &mut WaveLedger, b: Bucket) -> &mut f64 {
    match b {
        Bucket::OtherArm => &mut l.other_arm,
        Bucket::UntrackedOrders => &mut l.untracked_orders,
        Bucket::OtherFamily => &mut l.other_family,
    }
}

/// One event of a chain's history: its identity, the junction met there,
/// and what the event booked (losses only).
#[derive(Clone, Debug)]
struct HistoryEntry {
    key: String,
    junction: Option<Junction>,
    delta: WaveLedger,
}

/// `u` with the plane waves a window `out` cannot hold taken out, those
/// beyond its band about the field's carrier, and the power they carried;
/// `u` as it is when `out` is no coarser.
fn band_to(u: &Field2, out: &Grid2, fft: &mut Fft2) -> (Field2, f64) {
    let g = u.grid;
    let lim = [0.5 / out.pitch_mm[0], 0.5 / out.pitch_mm[1]];
    if out.pitch_mm[0] <= g.pitch_mm[0] * (1.0 + 1e-12) && out.pitch_mm[1] <= g.pitch_mm[1] * (1.0 + 1e-12) {
        return (u.clone(), 0.0);
    }
    let n = g.n;
    let mut b = u.data.clone();
    crate::fft::ifftshift2(&mut b, n);
    fft.forward(&mut b, n, g.cell_area_mm2());
    let df = g.freq_pitch(0) * g.freq_pitch(1);
    let mut removed = 0.0;
    for ky in 0..n[1] {
        let fy = g.fft_freq(1, ky);
        for kx in 0..n[0] {
            let fx = g.fft_freq(0, kx);
            if fx.abs() > lim[0] || fy.abs() > lim[1] {
                let a = &mut b[ky * n[0] + kx];
                removed += a.norm_sqr() * df;
                *a = Cx::ZERO;
            }
        }
    }
    if removed == 0.0 {
        return (u.clone(), 0.0);
    }
    fft.inverse(&mut b, n, df);
    crate::fft::fftshift2(&mut b, n);
    let mut v = u.clone();
    v.data = b;
    (v, removed)
}

/// A window a field is to be carried onto, and how.
struct Regrid {
    out: Grid2,
    how: RegridHow,
    half: f64,
    needed: f64,
}

#[derive(Clone, Copy)]
enum RegridHow {
    /// The zoomed angular spectrum, padded so many times.
    Zoom(usize),
    /// The zoomed Fresnel transform.
    Fresnel,
}

/// The polarization basis an element works in, met along `d_in` with normal
/// `n` and left along `d_out`: s and p of its plane of incidence, `ŝ` normal
/// to the plane and `p̂ = ŝ × d` in it, on either side; square to the beam,
/// the beam's own axes, whose field the element leaves as it is. Across the
/// element the s and p components keep their direction (`p̂` turns with the
/// beam, its part along the surface kept, as the tangential chart has it),
/// and an element's scalar amplitudes, which carry its signs, apply to each.
fn element_basis(inc: &Beam, n: Vec3, d_out: Vec3) -> ([Vec3; 2], [Vec3; 2]) {
    let n = n.normalize();
    let sn = inc.d.cross(n);
    if sn.length() < 1e-9 {
        return ([inc.x, inc.y], [inc.x, inc.y]);
    }
    let s_in = sn.normalize();
    let p_in = s_in.cross(inc.d);
    let s_out = (s_in - d_out * d_out.dot(s_in)).normalize();
    let mut p_out = s_out.cross(d_out);
    let along = |v: Vec3| v - n * n.dot(v);
    if along(p_out).dot(along(p_in)) < 0.0 {
        p_out = -p_out;
    }
    ([s_in, p_in], [s_out, p_out])
}

/// `a·u + b·v`, sample by sample, on `u`'s grid.
fn mix(u: &Field2, v: &Field2, a: f64, b: f64) -> Field2 {
    let mut w = u.clone();
    for (z, (x, y)) in w.data.iter_mut().zip(u.data.iter().zip(&v.data)) {
        *z = *x * a + *y * b;
    }
    w
}

/// `a·u + b·v` with complex weights.
fn mix_c(u: &Field2, v: &Field2, a: Cx, b: Cx) -> Field2 {
    let mut w = u.clone();
    for (z, (x, y)) in w.data.iter_mut().zip(u.data.iter().zip(&v.data)) {
        *z = *x * a + *y * b;
    }
    w
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
    // Where the ray from `from`, `r` off the chief ray at azimuth `t`, first
    // meets the element: before the hit along `d_in`, after it along `d_out`.
    let meet = |r: f64, t: f64| -> Option<(f64, f64)> {
        let o = from + x * (r * t.cos()) + y * (r * t.sin());
        let ray = oc_core::vec3::Ray3::new(o, d_in).ok()?;
        let res = trace3(probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
        let root = res.segments.iter().find(|s| s.parent_path_id.is_none())?;
        // A root with a child met the element at its end.
        res.segments.iter().any(|s| s.parent_path_id == Some(root.path_id)).then(|| ((root.b - hit).dot(d_in), (root.b - hit).dot(d_out)))
    };
    // Rings out to the window's corners, and along each azimuth the edge of
    // the element where a ring first misses it, found by bisection: an
    // element smaller than the window reaches its greatest depth at its rim,
    // which no ring need meet.
    for k in 0..24 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / 24.0;
        let mut last = 0.0;
        for ring in 1..=8 {
            let r = r_max * ring as f64 / 8.0;
            if let Some((a, b)) = meet(r, t) {
                s_min = s_min.min(a);
                s_max = s_max.max(b);
                last = r;
                continue;
            }
            if r - last > r_max / 8.0 + 1e-12 {
                continue;
            }
            let (mut lo, mut hi) = (last, r);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if meet(mid, t).is_some() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            if let Some((a, b)) = meet(lo, t) {
                s_min = s_min.min(a);
                s_max = s_max.max(b);
            }
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
    /// A plane-parallel plate: the field reaches the plane before its first
    /// face at the event's point and leaves from the plane behind its second.
    Plate { first: oc_core::seq3::SeqHit3, second: oc_core::seq3::SeqHit3, exit: Vec3, plate: Plate },
    End,
}

/// A plane-parallel plate where the chain enters a body by `h` and leaves it
/// by `h2`: both faces flat (every probe ray meets them along one normal),
/// parallel, bare or with a residual AR, the medium the same on both sides.
/// Returns the plate and the planes before and after it, or `None` for
/// anything else, which is then traced face by face.
fn plate_of(
    scene: &Scene3,
    h: &oc_core::seq3::SeqHit3,
    h2: &oc_core::seq3::SeqHit3,
    resume: Vec3,
    next: Vec3,
    half: f64,
    lam: f64,
) -> Option<(Plate, Vec3, Vec3)> {
    use oc_trace::{trace3, LaunchRay3, TraceConfig3, TraceEventKind};
    if h2.object != h.object || h2.kind != SeqHitKind::Body || h.action != SeqAction::Transmit || h2.action != SeqAction::Transmit {
        return None;
    }
    let n = h.normal.normalize();
    if n.dot(h2.normal.normalize()).abs() < 1.0 - 1e-12 {
        return None;
    }
    let d = (h.point - resume).normalize();
    let d_out = (next - h2.point).normalize();
    if d.dot(d_out) < 1.0 - 1e-12 || (h.n_before - h2.n_after).abs() > 1e-12 || (h.n_after - h2.n_before).abs() > 1e-12 {
        return None;
    }
    let body = scene.bodies.iter().find(|b| b.common.id == h.object)?;
    let residual = |r: &SurfaceResponse| match r {
        SurfaceResponse::DielectricFresnel { ar_residual } => ar_residual.sample(lam),
        _ => None,
    };
    let r1 = residual(&body.surface.for_patch(h.patch).from_outside)?;
    let r2 = residual(&body.surface.for_patch(h2.patch).from_inside)?;
    // The planes before the first face and behind the second, over the window.
    let x = if d.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let x = (x - d * d.dot(x)).normalize();
    let y = d.cross(x);
    let corners = [(-half, -half), (half, -half), (-half, half), (half, half)];
    let depth = |a: f64, b: f64| -(x * a + y * b).dot(n) / d.dot(n);
    let s_min = corners.iter().map(|&(a, b)| depth(a, b)).fold(0.0f64, f64::min);
    let s_max = corners.iter().map(|&(a, b)| depth(a, b)).fold(0.0f64, f64::max);
    let entry = h.point + d * (s_min - 1e-3);
    let exit = h2.point + d * (s_max + 1e-3);
    if (entry - resume).dot(d) <= 0.0 || (next - exit).dot(d) <= 0.0 {
        return None;
    }
    // Flat: along each of eight directions from the chief ray, out to where
    // the faces end (found by bisection, so a small element is not missed),
    // every ray meets the two faces square to n.
    let probe = probe_scene(scene, h.object);
    let cfg = TraceConfig3 { ghosts: false, polarization: false, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(&probe), ..TraceConfig3::default() };
    // The normals at the two faces where the ray from (a, b) meets them.
    let faces = |a: f64, b: f64| -> Option<(Vec3, Vec3)> {
        let ray = oc_core::vec3::Ray3::new(entry + x * a + y * b, d).ok()?;
        let res = trace3(&probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
        let hits: Vec<(oc_core::id::PatchId, Vec3)> = res
            .events
            .iter()
            .filter_map(|e| match &e.kind {
                TraceEventKind::Interact { surface, normal, .. } => Some((surface.patch, *normal)),
                _ => None,
            })
            .take(2)
            .collect();
        (hits.len() == 2 && hits[0].0 == h.patch && hits[1].0 == h2.patch).then(|| (hits[0].1, hits[1].1))
    };
    let r_max = half * std::f64::consts::SQRT_2;
    for k in 0..8 {
        let t = std::f64::consts::PI * k as f64 / 4.0;
        let (c, sn) = (t.cos(), t.sin());
        let (mut lo, mut hi) = (0.0, r_max);
        if faces(hi * c, hi * sn).is_none() {
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if faces(mid * c, mid * sn).is_some() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        } else {
            lo = hi;
        }
        for f in [0.3, 0.6, 0.9, 0.999] {
            let r = f * lo;
            let (n1, n2) = faces(r * c, r * sn)?;
            if n1.normalize().dot(n).abs() < 1.0 - 1e-9 || n2.normalize().dot(n).abs() < 1.0 - 1e-9 {
                return None;
            }
        }
    }
    let thickness_mm = (h2.point - h.point).dot(n).abs();
    let chief_opl_mm = h.n_before * (h.point - entry).length() + h.n_after * (h2.point - h.point).length() + h2.n_after * (exit - h2.point).length();
    let plate = Plate { normal: n, thickness_mm, n_glass: h.n_after, n_out: h.n_before, residual: [r1, r2], chief_opl_mm };
    Some((plate, entry, exit))
}

/// One chain of a wave path: the beam carried along one branch lineage of
/// the axial ray, at one wavelength.
struct ChainJob {
    scene: Arc<Scene3>,
    spec: WavePathSpec,
    lam: f64,
    /// The beam's power on this chain, and the index it is launched into.
    power: f64,
    n0: f64,
    /// The field is laid out only when the chain first runs, so a job of
    /// many chains holds one full field at a time.
    launched: bool,
    /// Read out onto the detector's own grid, to be added to other arms.
    face: bool,
    /// The polarization the field (or, while an element runs on one of two
    /// components, that component) stands for.
    pol: Pol,
    /// A field of two polarization components, `u` along the beam frame's
    /// `x` and this along its `y`; `None` for a chain whose elements cannot
    /// tell polarizations apart, whose one field stands for unpolarized
    /// light.
    uy: Option<Field2>,
    /// Whether the chain carries two components, and the launch's Jones
    /// vector in the beam frame.
    polarized: bool,
    jones: [f64; 2],
    /// Each event: where the field reaches it, where it leaves from (the
    /// same point except after a transfer), and what it is.
    points: Vec<(Vec3, Vec3, Ev)>,
    /// An identity per event, for finding the events chains share.
    keys: Vec<String>,
    /// The launch's ledger, and each event run so far: its key, the
    /// junction met there and what it booked, for the network's ledger.
    launch: WaveLedger,
    history: Vec<HistoryEntry>,
    /// The junction the event now running met, taken into its history entry.
    junction: Option<Junction>,
    next: usize,
    beam: Beam,
    u: Field2,
    /// The beam's Gaussian parameter across each axis of its frame, in the
    /// medium it is in (`q = z + i z_R`, its curvature `Re 1/q`), carried from
    /// the source's declared beam through each element's paraxial matrices:
    /// the wavefront a transfer's rays are traced along, which the field
    /// itself does not choose.
    q: [Cx; 2],
    /// The ray that reference is about, across the beam: its point and
    /// direction cosines `[a, b, tx, ty]`, the chief ray's zeros unless the
    /// field is one of a modulator's orders.
    at: [f64; 4],
    /// The map traced at an event, kept for the other component.
    cache: Option<(usize, Arc<TracedMap>)>,
    /// How far an event's fields departed from the references their
    /// transfer traced them along, gathered over its orders and components.
    departures: Option<Departures>,
    /// A modulator's other orders, each carried apart with its own reference
    /// ray while a traced element lies ahead; `u`, `uy` and `at` are the
    /// strongest one. Their sum is the field.
    rest: Vec<Lobe>,
    report: WaveReport,
    fft: Fft2,
}

/// What a transfer's fields departed from their references by, over every
/// order and component it carried.
#[derive(Default)]
struct Departures {
    what: &'static str,
    id: u64,
    curvature: f64,
    slab: f64,
    k: f64,
    orders: bool,
    power: f64,
    worst: f64,
    rms2: f64,
}

impl Departures {
    /// A field of power `p`: its steepest departure counts where it holds a
    /// millionth of what the transfer carried so far, its rms by its power.
    fn add(&mut self, p: f64, worst: f64, rms: f64) {
        self.power += p;
        self.rms2 += p * rms * rms;
        if p >= 1e-6 * self.power {
            self.worst = self.worst.max(worst);
        }
    }

    fn note(&self) -> String {
        let about = if self.orders { "about each order's own ray" } else { "about the chief ray" };
        let rms = (self.rms2 / self.power.max(f64::MIN_POSITIVE)).sqrt();
        format!(
            "{} #{}: geometric field tracing (no diffraction inside the element; rays traced along a reference fixed by the geometry, the beam's paraxial wavefront, curvature {:.4e} /mm, {about}; the field's directions depart from it by up to {:.3} mrad where it holds a hundredth of its peak (rms {:.3} mrad), which across the {:.4} mm slab moves its light by up to {:.2e} mm and its phase by up to {:.2e} rad; unpolarized transmittance of the faces)",
            self.what,
            self.id,
            self.curvature,
            1e3 * self.worst,
            1e3 * rms,
            self.slab,
            self.worst * self.slab,
            0.5 * self.k * self.worst * self.worst * self.slab
        )
    }
}

/// One of a modulator's orders, carried apart from the others.
struct Lobe {
    u: Field2,
    uy: Option<Field2>,
    at: [f64; 4],
}

/// The most orders of a modulator carried apart, each holding at least 1e-4
/// of the field's power; the weaker ones join their nearest.
const MAX_LOBES: usize = 9;

/// The Gaussian parameter of the declared beam at the launch plane, in a
/// medium of index `n0`: a Gaussian's own, a top-hat's as the Gaussian of its
/// radius with its waist there (H. Kogelnik, T. Li, *Appl. Opt.* 5, 1550
/// (1966)).
fn launch_q(spec: &WavePathSpec, lam_um: f64, n0: f64) -> Cx {
    let lam = lam_um / 1000.0 / n0;
    match spec.beam {
        BeamProfile::Gaussian { waist_radius_mm, waist_offset_mm } => Cx::new(-waist_offset_mm, std::f64::consts::PI * waist_radius_mm * waist_radius_mm / lam),
        BeamProfile::TopHat { radius_mm } => Cx::new(0.0, std::f64::consts::PI * radius_mm * radius_mm / lam),
    }
}

/// The field a source launches on `grid`, with the ledger of the launch.
fn launch_field(spec: &WavePathSpec, grid: Grid2, lam: f64, n0: f64, power: f64, beam: &Beam) -> Result<(Field2, WaveLedger), WaveError> {
    let mut u = match spec.beam {
        BeamProfile::Gaussian { waist_radius_mm, waist_offset_mm } => {
            GaussianBeam { w0_mm: waist_radius_mm, lambda_um: lam, n: n0, power, center: [0.0, 0.0] }.sample(grid, -waist_offset_mm)?
        }
        BeamProfile::TopHat { radius_mm } => {
            // The disc's edge weighted by exact cell coverage, as an
            // aperture's is.
            let a = (power / (std::f64::consts::PI * radius_mm * radius_mm)).sqrt();
            let cover = crate::elements::coverage(&grid, &crate::elements::Shape::Disc { c: [0.0, 0.0], r: radius_mm })?;
            let mut u = Field2::zeros(grid, lam)?.in_medium(n0)?;
            for (z, c) in u.data.iter_mut().zip(&cover) {
                *z = Cx::real(a * c);
            }
            u
        }
    };
    u.frame = beam.iso();
    // The source launches its power; what the samples do not hold is the
    // Gaussian's tail beyond the window, or the top-hat's edge diffracted
    // beyond the grid's band.
    let mut ledger = WaveLedger { launched: power, ..Default::default() };
    let short = power - u.power();
    match spec.beam {
        BeamProfile::TopHat { .. } => ledger.subcell_projection += short,
        BeamProfile::Gaussian { .. } if short >= 0.0 => ledger.left_window += short,
        BeamProfile::Gaussian { .. } => ledger.resampling += short,
    }
    Ok((u, ledger))
}

impl ChainJob {
    /// Plan the events along `seq` and frame the beam at the source; the
    /// field is launched when the chain first steps.
    fn new(scene: &Arc<Scene3>, spec: &WavePathSpec, seq: oc_core::seq3::SequencePrescription3, power: f64, face: bool, jones: Option<[f64; 2]>) -> Result<Self, WaveError> {
        let src = scene
            .sources
            .iter()
            .find(|s| s.id == spec.source)
            .ok_or_else(|| WaveError::OutOfRange("no such source".into()))?;
        let lam = seq.launch.lambda_um;
        let d0 = seq.launch.dir.normalize();
        let x0 = src.transform.rotate(Vec3::Y);
        let x0 = (x0 - d0 * d0.dot(x0)).normalize();
        let beam = Beam { origin: seq.launch.point, x: x0, y: d0.cross(x0), d: d0 };
        let n0 = seq.hits.first().map_or(scene.env.ambient_index, |h| h.n_before);
        // A placeholder until the chain runs.
        let u = Field2::zeros(Grid2::square(1, 1.0)?, lam)?.in_medium(n0)?;
        let report = WaveReport { ledger: WaveLedger { launched: power, ..Default::default() }, ..Default::default() };

        // The chain's events. Each face of a body, and each curved surface, is
        // a transfer across the thin slab that holds the surface where the
        // beam meets it; between faces the field diffracts inside the glass.
        let mut chain: Vec<(Vec3, Vec3, Ev)> = Vec::new();
        let mut resume = seq.launch.point;
        let mut k = 0;
        while k < seq.hits.len() {
            let h = &seq.hits[k];
            k += 1;
            let curved = h.kind == SeqHitKind::Surface
                && scene.surfaces.iter().any(|s| s.common.id == h.object && !matches!(s.geometry, SurfaceGeometry3::Plane { .. }));
            if h.kind != SeqHitKind::Body && !curved {
                chain.push((h.point, h.point, Ev::Hit(h.clone())));
                resume = h.point;
                continue;
            }
            // A plane-parallel plate is one closed-form step, its round trips
            // included (crate::planar).
            if h.kind == SeqHitKind::Body {
                if let Some(h2) = seq.hits.get(k) {
                    let next = seq.hits.get(k + 1).map_or(seq.terminal.point, |n| n.point);
                    if let Some((plate, entry, exit)) = plate_of(scene, h, h2, resume, next, 0.5 * spec.window_mm, lam) {
                        chain.push((entry, exit, Ev::Plate { first: h.clone(), second: h2.clone(), exit, plate }));
                        resume = exit;
                        k += 1;
                        continue;
                    }
                }
            }
            let next_point = seq.hits.get(k).map_or(seq.terminal.point, |n| n.point);
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
            // An aperture inside a face's slab, or a plate, has no place in
            // the field's steps: refused by name rather than skipped.
            let inside: Vec<(Vec3, Vec3, &oc_core::seq3::SeqHit3)> = match &ev {
                Ev::Transfer { hit, .. } => vec![(p, hit.point, hit), (hit.point, after, hit)],
                Ev::Plate { first, .. } => vec![(p, after, first)],
                _ => Vec::new(),
            };
            for (from, to, hit) in inside {
                {
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
        let keys = points
            .iter()
            .map(|(p, _, ev)| {
                let at = format!("{:.6},{:.6},{:.6}", p.x, p.y, p.z);
                match ev {
                    Ev::Hit(h) => format!("hit {} {:?} {at}", h.object.0, h.action),
                    Ev::Aperture(ai) => format!("aperture {ai} {at}"),
                    Ev::Transfer { hit, .. } => format!("transfer {} {:?} {at}", hit.object.0, hit.action),
                    Ev::Plate { first, .. } => format!("plate {} {at}", first.object.0),
                    Ev::End => format!("end {at}"),
                }
            })
            .collect();
        Ok(ChainJob {
            scene: Arc::clone(scene),
            spec: *spec,
            lam,
            power,
            n0,
            launched: false,
            face,
            pol: Pol::Unpolarized,
            uy: None,
            polarized: jones.is_some(),
            jones: jones.unwrap_or([1.0, 0.0]),
            points,
            keys,
            launch: WaveLedger::default(),
            history: Vec::new(),
            junction: None,
            next: 0,
            beam,
            u,
            q: [launch_q(spec, lam, n0); 2],
            at: [0.0; 4],
            cache: None,
            departures: None,
            rest: Vec::new(),
            report,
            fft: Fft2::new(),
        })
    }

    /// Carry the field through up to `events` more events. `Ok(true)` once
    /// the whole chain is done; an element the field has no model for is an
    /// error naming it, and the job is not resumable after one.
    fn step(&mut self, events: usize) -> Result<bool, WaveError> {
        if !self.launched {
            let grid = Grid2::square(self.spec.grid_n, self.spec.window_mm / self.spec.grid_n as f64)?;
            let (u, ledger) = launch_field(&self.spec, grid, self.lam, self.n0, self.power, &self.beam)?;
            if self.polarized {
                let (mut a, mut b) = (u.clone(), u);
                for z in &mut a.data {
                    *z *= self.jones[0];
                }
                for z in &mut b.data {
                    *z *= self.jones[1];
                }
                self.u = a;
                self.uy = Some(b);
            } else {
                self.u = u;
            }
            self.report.ledger = ledger;
            self.launch = ledger;
            self.history.clear();
            self.launched = true;
        }
        let end = self.next.saturating_add(events).min(self.points.len());
        while self.next < end {
            let before = self.report.ledger;
            self.junction = None;
            self.event(self.next)?;
            self.history.push(HistoryEntry {
                key: self.keys[self.next].clone(),
                junction: self.junction.take(),
                delta: losses_between(&self.report.ledger, &before),
            });
            self.next += 1;
        }
        Ok(self.next == self.points.len())
    }

    /// The junction the running event met. A second pass at the same element
    /// (the field's other component, another of its orders) adds its input,
    /// the share power-weighted, so the junction speaks for the whole field.
    fn note_junction(&mut self, j: Junction) {
        self.junction = Some(match self.junction {
            Some(prev) if prev.object == j.object && prev.kind == j.kind => {
                let input = prev.input + j.input;
                Junction {
                    input,
                    share: if input > 0.0 { (prev.input * prev.share + j.input * j.share) / input } else { 0.0 },
                    foreign_absorbed: prev.foreign_absorbed + j.foreign_absorbed,
                    foreign_family: prev.foreign_family + j.foreign_family,
                    ..j
                }
            }
            _ => j,
        });
    }

    fn next_dir(&self, i: usize, here: Vec3, current: Vec3) -> Vec3 {
        self.points.get(i + 1).map_or(current, |(p, _, _)| (*p - here).normalize())
    }

    /// A plane-parallel plate: its rim laid on the field at the field's own
    /// sampling, then the slab's closed form plane wave by plane wave.
    fn plate_event(&mut self, first: &oc_core::seq3::SeqHit3, second: &oc_core::seq3::SeqHit3, exit: Vec3, plate: &Plate) -> Result<(), WaveError> {
        use oc_trace::{trace3, LaunchRay3, TraceConfig3, TraceEventKind};
        let beam = self.beam;
        let lam = self.lam;
        let before = self.u.power();
        // The rim: where a ray from the plane before the plate meets the two
        // faces in turn.
        let probe = probe_scene(&self.scene, first.object);
        let cfg = TraceConfig3 { ghosts: false, polarization: false, record_events: true, bounds_mm: oc_trace::bounds_for_scene3(&probe), ..TraceConfig3::default() };
        let pass = |a: f64, b: f64| -> bool {
            let Ok(ray) = oc_core::vec3::Ray3::new(beam.point(a, b), beam.d) else { return false };
            let res = trace3(&probe, &[LaunchRay3 { ray, lambda_um: lam, flux: 1.0, source: None, pol: None }], &cfg);
            let hits: Vec<_> = res
                .events
                .iter()
                .filter_map(|e| match &e.kind {
                    TraceEventKind::Interact { surface, .. } => Some(surface.patch),
                    _ => None,
                })
                .take(2)
                .collect();
            hits == [first.patch, second.patch]
        };
        let g = 65usize;
        let w = self.u.grid.width_mm(0).max(self.u.grid.width_mm(1));
        let h = w / (g - 1) as f64;
        let half = (g - 1) as f64 / 2.0;
        let grid: Vec<bool> = (0..g * g).map(|k| pass(((k % g) as f64 - half) * h, ((k / g) as f64 - half) * h)).collect();
        let rim = if grid.iter().all(|&p| p) {
            "no rim within the window".to_string()
        } else {
            let poly = crate::transfer::rim_polygon(&pass, &grid, g, h)
                .ok_or_else(|| WaveError::OutOfRange(format!("plate #{} ({}): the edge of what passes it is not star-shaped about the chief ray", first.object.0, first.label)))?;
            let cov = crate::elements::polygon_coverage(&self.u.grid, &poly);
            let (stopped, beyond) = crate::elements::apply_coverage(&mut self.u, &cov)?;
            self.report.ledger.absorbed_aperture += stopped;
            self.report.ledger.subcell_projection += beyond;
            format!("its rim laid on the field with exact cell coverage, {:.3e} of the power stopped there", stopped / before.max(f64::MIN_POSITIVE))
        };
        let out = beam.turned(exit, beam.d);
        let (v, st) = crate::planar::plate_step(&self.u, &beam, &out, plate, self.pol, &mut self.fft)?;
        let l = &mut self.report.ledger;
        l.other_arm += st.reflected;
        l.left_window += st.beyond + st.left_window;
        l.evanescent += st.evanescent;
        l.resampling += st.rest;
        let oblique = beam.d.dot(plate.normal).abs() < 1.0 - 1e-12;
        let pol = match self.pol {
            Pol::S(_) => "s",
            Pol::P(_) => "p",
            Pol::Unpolarized | Pol::Projected => "s = p",
        };
        let trips = match st.round_trips {
            None => "every round trip".to_string(),
            Some(m) => format!(
                "{m} round trip(s), each {:.4} mm across the beam; the rest, beyond the window, {:.3e} of the power",
                st.round_trip_shift_mm,
                st.beyond / before.max(f64::MIN_POSITIVE)
            ),
        };
        self.report.steps.push(element_step(
            0.0,
            format!(
                "plate #{} ({}): {:.6} mm of index {:.6}, the slab's closed form plane wave by plane wave ({pol}), {trips}; {rim}; {:.3e} reflected",
                first.object.0,
                first.label,
                plate.thickness_mm,
                plate.n_glass,
                st.reflected / before.max(f64::MIN_POSITIVE)
            ),
        ));
        if oblique && self.pol == Pol::Unpolarized {
            self.report.approx(format!("plate #{}: met obliquely by a field standing for unpolarized light, its s and p amplitudes averaged", first.object.0));
        }
        if oblique {
            self.report.approx(
                "planar layers: every plane wave takes its own angle's amplitude for the chain's s or p; plane waves off the plane of incidence see s and p slightly rotated, which is left out",
            );
        }
        self.u = v;
        self.beam = out;
        self.u.n_medium = plate.n_out;
        Ok(())
    }

    /// Carry the field `z` on onto a window fitted to the beam there, when
    /// the one it has would no longer suit it: when the beam, six of its rms
    /// radii across (`Moments`), would fill less than a quarter of the window
    /// (a focus, a pinhole's plane), or, on a window already narrowed, more
    /// than four fifths of it. The window is centred on the beam's predicted
    /// centroid, keeps its sample count, and never grows past the one the
    /// spec declares. `None` to stay on the window the field has.
    ///
    /// The zoomed angular spectrum carries it, exactly, padded enough that
    /// the plane waves' walk over `z` stays inside; where that walk exceeds
    /// the window and `z` is past the window's critical distance (a narrow
    /// window widening again, typically), the zoomed Fresnel transform, whose
    /// paraxial error on the field is reported.
    fn regrid_plan(&mut self, z: f64) -> Result<Option<Regrid>, WaveError> {
        let mut fields: Vec<&Field2> = vec![&self.u];
        fields.extend(self.uy.as_ref());
        for l in &self.rest {
            fields.push(&l.u);
            fields.extend(l.uy.as_ref());
        }
        let mut m: Option<Moments> = None;
        for f in fields {
            if let Some(a) = Moments::of(f, &mut self.fft) {
                m = Some(match m {
                    Some(b) => b.with(&a),
                    None => a,
                });
            }
        }
        let Some(m) = m else { return Ok(None) };
        let g = self.u.grid;
        let n = g.n[0].max(g.n[1]);
        let half_now = 0.5 * g.width_mm(0).max(g.width_mm(1));
        let half_max = 0.5 * self.spec.window_mm;
        let needed = 6.0 * m.sigma_at(0, z).max(m.sigma_at(1, z));
        let lam = self.u.wavelength_in_medium_mm();
        // A window no narrower than a few dozen wavelengths.
        let target = needed.min(half_max).max(16.0 * lam);
        let shrink = target < 0.25 * half_now;
        let grow = half_now < half_max * (1.0 - 1e-9) && needed > 0.8 * half_now;
        if !(shrink || grow) {
            return Ok(None);
        }
        let half = if grow { (2.0 * needed).min(half_max) } else { target };
        let centre = [m.mean[0] + z * m.tilt[0], m.mean[1] + z * m.tilt[1]];
        let pitch = 2.0 * half / n as f64;
        let out = Grid2::new([n, n], [pitch, pitch], centre)?;
        // How far the field's plane waves walk across the beam over z.
        let walk = z * (m.tilt[0].abs().max(m.tilt[1].abs()) + 4.0 * m.spread[0].max(m.spread[1]).sqrt());
        let width_in = 2.0 * half_now;
        let zc = g.width_mm(0) * g.pitch_mm[0] / lam;
        let how = if walk <= width_in {
            RegridHow::Zoom(2)
        } else if z > zc {
            RegridHow::Fresnel
        } else {
            RegridHow::Zoom(((2.0 * walk / width_in).ceil() as usize).clamp(2, 8))
        };
        Ok(Some(Regrid { out, how, half, needed }))
    }

    /// Carry `u` `z` on onto the window a plan chose. A coarser window holds
    /// fewer directions: the plane waves beyond its band are taken out first
    /// and booked (`spectral_projection`), since free space keeps every plane wave
    /// and the new samples, read at their points, would fold them back in at
    /// directions they do not have.
    fn regrid_apply(&mut self, u: &Field2, z: f64, pl: &Regrid) -> Result<(Field2, WaveReport), WaveError> {
        let (banded, beyond) = band_to(u, &pl.out, &mut self.fft);
        let u = &banded;
        let ((v, mut r), how) = match pl.how {
            RegridHow::Zoom(2) => (crate::asm_zoom(u, z, pl.out, 2, &mut self.fft)?, "the zoomed angular spectrum"),
            RegridHow::Zoom(pad) => (crate::asm_zoom(u, z, pl.out, pad, &mut self.fft)?, "the zoomed angular spectrum, padded"),
            RegridHow::Fresnel => {
                let e = crate::propagate::paraxial_error(u, z, &mut self.fft);
                let mut step = crate::czt::fresnel_zoom(u, z, pl.out, &mut self.fft)?;
                if e > 1e-4 {
                    step.1.warn(SamplingWarning::Paraxial { field_error: e });
                }
                (step, "the zoomed Fresnel transform")
            }
        };
        let n = pl.out.n[0];
        r.ledger.spectral_projection += beyond;
        let cut = if beyond > 0.0 { format!("; {:.3e} of its power in directions beyond the new window's band taken out first", beyond / (beyond + u.power()).max(f64::MIN_POSITIVE)) } else { String::new() };
        r.steps.push(element_step(
            z,
            format!(
                "the field carried {z:.4} mm onto a new window {:.4e} mm across ({n}², pitch {:.4e} mm), where the beam is predicted {:.4e} mm across (six rms radii), by {how}{cut}",
                2.0 * pl.half,
                pl.out.pitch_mm[0],
                pl.needed
            ),
        ));
        Ok((v, r))
    }

    /// The element at event `idx` on a field of two components: the pair
    /// turned into the element's own basis (`element_basis`), each component
    /// carried through the element as the field alone would be, with its s or
    /// p, and the pair turned into the outgoing beam's frame. A polarizer or a
    /// retarder then applies its Jones matrix there (`layer`); an LCoS works
    /// on the component along its director or across it, whichever beam the
    /// chain follows.
    fn element_polarized(&mut self, idx: usize, ev: &Ev, p: Vec3) -> Result<(), WaveError> {
        let beam0 = self.beam;
        let ux = self.u.clone();
        let uy = self.uy.take().expect("two components");
        // The element's normal and the way out, and what it does to the pair.
        let mut layer: Option<usize> = None;
        let mut lcos: Option<(usize, bool)> = None;
        let (n_e, d_out) = match ev {
            Ev::Transfer { hit, d_out, .. } => (hit.normal, *d_out),
            Ev::Plate { plate, .. } => (plate.normal, beam0.d),
            Ev::Hit(h) => {
                let si = self
                    .scene
                    .surfaces
                    .iter()
                    .position(|s| s.common.id == h.object)
                    .ok_or_else(|| WaveError::OutOfRange("a chain surface is missing".into()))?;
                let s = &self.scene.surfaces[si];
                let n_s = s.common.transform.rotate(Vec3::Z);
                let response = if beam0.d.dot(n_s) < 0.0 { &s.response_front } else { &s.response_back };
                match response {
                    SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. } => layer = Some(si),
                    SurfaceResponse::Modulator { device, .. } if matches!(device.kind, ModulatorKind::Lcos(_)) => {
                        lcos = Some((si, h.action == SeqAction::Diffract));
                    }
                    _ => {}
                }
                (n_s, self.next_dir(idx, p, beam0.d))
            }
            _ => (beam0.d, beam0.d),
        };
        if let Some((si, modulated)) = lcos {
            // The director across the beam, and the component the beam the
            // chain follows takes: along it modulated, across it not (Zhang,
            // You & Chu 2014).
            let s = &self.scene.surfaces[si];
            let h_id = s.common.id.0;
            let t = s.common.transform;
            let n = t.rotate(Vec3::Z);
            let ModulatorKind::Lcos(l) = (match if beam0.d.dot(n) < 0.0 { &s.response_front } else { &s.response_back } {
                SurfaceResponse::Modulator { device, .. } => &device.kind,
                _ => unreachable!("an LCoS is noted"),
            }) else {
                unreachable!("an LCoS is noted")
            };
            let dl = l.director_deg.to_radians();
            let dir = t.rotate(Vec3::new(dl.cos(), dl.sin(), 0.0));
            let dir = (dir - beam0.d * beam0.d.dot(dir)).normalize();
            let keep = if modulated { dir } else { beam0.d.cross(dir) };
            let c = mix(&ux, &uy, beam0.x.dot(keep), beam0.y.dot(keep));
            let total = ux.power() + uy.power();
            let kept = c.power();
            let dropped = (total - kept).max(0.0);
            // The device takes the same share of either component (its fill
            // and reflectance do not tell the beams apart), measured on the
            // kept one, or on the other standing in for it when the kept one
            // is dark; the rest of the other component goes into the other
            // beam, which the chain does not follow.
            let before = self.report.ledger;
            let stand_in = kept <= 0.0 && dropped > 0.0;
            let ran = if stand_in { dropped } else { kept };
            self.u = if stand_in { mix(&ux, &uy, beam0.x.dot(beam0.d.cross(keep)), beam0.y.dot(beam0.d.cross(keep))) } else { c };
            self.pol = Pol::Projected;
            self.element(idx, ev.clone(), p)?;
            self.pol = Pol::Unpolarized;
            let rho = if ran > 0.0 { ((self.report.ledger.absorbed_element - before.absorbed_element) / ran).clamp(0.0, 1.0) } else { 0.0 };
            if stand_in {
                self.report.ledger = before;
                self.u.data.iter_mut().for_each(|v| *v = Cx::ZERO);
            }
            self.report.ledger.absorbed_element += dropped * rho;
            self.report.ledger.other_family += dropped * (1.0 - rho);
            self.note_junction(Junction {
                object: h_id,
                kind: JunctionKind::PerBeam,
                bucket: Bucket::OtherFamily,
                input: total,
                share: if total > 0.0 { ran / total } else { 0.0 },
                foreign_absorbed: dropped * rho,
                foreign_family: dropped * (1.0 - rho),
            });
            // The component leaves along the device's image of its direction:
            // its part along the surface kept, as the tangential chart has it.
            let out = self.beam;
            let kt = keep - n * n.dot(keep);
            let kn = -kt.dot(out.d) / n.dot(out.d);
            let e = (kt + n * kn).normalize();
            let o = self.u.clone();
            self.u = mix(&o, &o, e.dot(out.x), 0.0);
            self.uy = Some(mix(&o, &o, e.dot(out.y), 0.0));
            return Ok(());
        }
        let (bin, bout) = element_basis(&beam0, n_e, d_out);
        let comps = [mix(&ux, &uy, beam0.x.dot(bin[0]), beam0.y.dot(bin[0])), mix(&ux, &uy, beam0.x.dot(bin[1]), beam0.y.dot(bin[1]))];
        let oblique = bin[0] != beam0.x || bin[1] != beam0.y;
        let pols = if oblique { [Pol::S(bin[0]), Pol::P(bin[0])] } else { [Pol::Unpolarized, Pol::Unpolarized] };
        // A component without power is carried as zeros, on the other's
        // outgoing window: an element fitted to a field has nothing to fit.
        let lit: Vec<bool> = comps.iter().map(|c| c.power() > 0.0).collect();
        let mut outs: [Option<Field2>; 2] = [None, None];
        let mut beam1 = beam0;
        let mut ran = false;
        for k in 0..2 {
            if !lit[k] && (ran || lit[1 - k]) {
                continue;
            }
            self.u = comps[k].clone();
            self.pol = pols[k];
            self.beam = beam0;
            let n = self.report.steps.len();
            self.element(idx, ev.clone(), p)?;
            if ran {
                self.report.steps.truncate(n);
            }
            ran = true;
            beam1 = self.beam;
            outs[k] = Some(self.u.clone());
        }
        self.pol = Pol::Unpolarized;
        let zeros = |f: &Field2| {
            let mut z = f.clone();
            z.data.iter_mut().for_each(|v| *v = Cx::ZERO);
            z
        };
        let [o1, o2] = outs;
        let (o1, o2) = match (o1, o2) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => {
                let b = zeros(&a);
                (a, b)
            }
            (None, Some(b)) => (zeros(&b), b),
            (None, None) => unreachable!("one component is always run"),
        };
        let out = beam1;
        let mut fx = mix(&o1, &o2, bout[0].dot(out.x), bout[1].dot(out.x));
        let mut fy = mix(&o1, &o2, bout[0].dot(out.y), bout[1].dot(out.y));
        if let Some(si) = layer {
            let (gx, gy) = self.layer(si, &out, &fx, &fy)?;
            fx = gx;
            fy = gy;
        }
        self.u = fx;
        self.uy = Some(fy);
        Ok(())
    }

    /// A polarizer's or a retarder's Jones matrix on the pair `fx`, `fy`
    /// across `out`, as the ray tracer applies it to a ray along the chief
    /// ray: the element's front face, its layer on its own axis (the element's
    /// local `+X` turned `axis_deg` towards `+Y`, projected across the
    /// direction inside its glass), its back face; the faces diagonal in s and
    /// p, their reflection booked as the other arm and the rest of what the
    /// element takes as absorbed. One matrix, at the chief ray's angle, for
    /// the whole field.
    fn layer(&mut self, si: usize, out: &Beam, fx: &Field2, fy: &Field2) -> Result<(Field2, Field2), WaveError> {
        let s = &self.scene.surfaces[si];
        let t = s.common.transform;
        let nrm = t.rotate(Vec3::Z);
        let d = out.d;
        let response = if d.dot(nrm) < 0.0 { &s.response_front } else { &s.response_back };
        let (axis_deg, n_sub) = match response {
            SurfaceResponse::Polarizer { axis_deg, substrate_index, .. } | SurfaceResponse::Retarder { axis_deg, substrate_index, .. } => (*axis_deg, substrate_index.max(1.0)),
            _ => unreachable!("only polarizers and retarders are layers"),
        };
        let id = s.common.id.0;
        let n_here = fx.n_medium;
        let n_opp = if d.dot(nrm) < 0.0 { nrm } else { -nrm };
        let cos_i = (-n_opp.dot(d)).clamp(0.0, 1.0);
        let s_hat = oc_core::polarization::incidence_axis(d, n_opp).unwrap_or(out.x);
        let p_hat = s_hat.cross(d);
        // Into the element's glass by Snell's law.
        let eta = n_here / n_sub;
        let kk = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
        if kk < 0.0 {
            return Err(WaveError::OutOfRange(format!("#{id}: the field is past the critical angle of the element's own glass and never reaches its layer")));
        }
        let cos_t = kk.sqrt();
        let d_in = (d * eta + n_opp * (eta * cos_i - cos_t)).normalize();
        let p_glass = s_hat.cross(d_in);
        let front = oc_core::thinfilm::interface_amp(n_here, n_sub, cos_i);
        let back = oc_core::thinfilm::interface_amp(n_sub, n_here, cos_t);
        let (sa, ca) = axis_deg.to_radians().sin_cos();
        let axis = t.rotate(Vec3::X) * ca + t.rotate(Vec3::Y) * sa;
        let projected = axis - d_in * d_in.dot(axis);
        if projected.length() < 1e-9 {
            return Err(WaveError::OutOfRange(format!("#{id}: the beam runs along the element's own axis, which has no direction across it")));
        }
        let e1 = projected.normalize();
        let e2 = d_in.cross(e1);
        let (a, b, what) = match response {
            SurfaceResponse::Polarizer { extinction, .. } => {
                let aoi = cos_i.acos().to_degrees();
                let leak = extinction
                    .sample(aoi)
                    .ok_or_else(|| WaveError::OutOfRange(format!("polarizer #{id}: its extinction curve has no value at {aoi:.3}°")))?;
                (Cx::ONE, Cx::real(leak.clamp(0.0, 1.0).sqrt()), format!("polarizer, axis {axis_deg}°, extinction {leak:.3e}"))
            }
            SurfaceResponse::Retarder { retardance_deg, crystal, .. } => {
                let rad = match crystal {
                    None => retardance_deg.to_radians(),
                    Some(c) => {
                        // As the tracer has it: the optic axis is the slow one
                        // in a positive crystal, the fast one in a negative.
                        let optic = if c.n_e >= c.n_o { axis_deg + 90.0 } else { axis_deg };
                        let (so, co) = optic.to_radians().sin_cos();
                        let c_hat = t.rotate(Vec3::X) * co + t.rotate(Vec3::Y) * so;
                        let lean = d - n_opp * d.dot(n_opp);
                        let cos_axis = if lean.length() > 1e-12 { lean.normalize().dot(c_hat) } else { 0.0 };
                        let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
                        oc_core::polarization::a_plate_retardance(c.n_o, c.n_e, c.thickness_um(*retardance_deg), self.lam, n_here * sin_i, cos_axis).abs()
                    }
                };
                (Cx::ONE, Cx::expi(rad), format!("retarder, fast axis {axis_deg}°, retardance {:.4}°", rad.to_degrees()))
            }
            _ => unreachable!("only polarizers and retarders are layers"),
        };
        // The layer in the glass's (ŝ, p̂) basis, between the faces'.
        let c1 = [e1.dot(s_hat), e1.dot(p_glass)];
        let c2 = [e2.dot(s_hat), e2.dot(p_glass)];
        let lay = |i: usize, j: usize| a * (c1[i] * c1[j]) + b * (c2[i] * c2[j]);
        let tf = [front.t_s * front.t_power_scale.max(0.0).sqrt(), front.t_p * front.t_power_scale.max(0.0).sqrt()];
        let tb = [back.t_s * back.t_power_scale.max(0.0).sqrt(), back.t_p * back.t_power_scale.max(0.0).sqrt()];
        let j = |i: usize, k: usize| tb[i] * lay(i, k) * tf[k];
        let cs = mix(fx, fy, out.x.dot(s_hat), out.y.dot(s_hat));
        let cp = mix(fx, fy, out.x.dot(p_hat), out.y.dot(p_hat));
        let before = cs.power() + cp.power();
        let reflected = front.r_s.norm_sqr() * cs.power() + front.r_p.norm_sqr() * cp.power();
        let ns = mix_c(&cs, &cp, j(0, 0), j(0, 1));
        let np = mix_c(&cs, &cp, j(1, 0), j(1, 1));
        let gx = mix(&ns, &np, s_hat.dot(out.x), p_hat.dot(out.x));
        let gy = mix(&ns, &np, s_hat.dot(out.y), p_hat.dot(out.y));
        let after = gx.power() + gy.power();
        let l = &mut self.report.ledger;
        l.other_arm += reflected;
        l.absorbed_element += (before - after - reflected).max(0.0);
        if let Some(st) = self.report.steps.last_mut() {
            st.note += &format!("; its Jones matrix on the field's two components ({what}), at the chief ray's {:.3}°", cos_i.acos().to_degrees());
        }
        self.report.approx("polarizers and retarders on the wave path: one Jones matrix for the whole field, the chief ray's");
        if n_sub > 1.0 {
            self.report.approx(format!("#{id}: the element's faces added singly, at the chief ray's angle: no bounce between them"));
        }
        if matches!(response, SurfaceResponse::Retarder { crystal: Some(_), .. }) {
            self.report.approx(format!("#{id}: one wave through the crystal, no walk-off between its ordinary and extraordinary waves"));
        }
        Ok((gx, gy))
    }

    /// The chain's twin for its second component: the same chain, done, with
    /// `u` as its field and a report of its own.
    fn twin(&self, u: Field2) -> ChainJob {
        let launched = u.power();
        ChainJob {
            scene: Arc::clone(&self.scene),
            spec: self.spec,
            lam: self.lam,
            power: 0.0,
            n0: self.n0,
            launched: true,
            face: self.face,
            pol: Pol::Unpolarized,
            uy: None,
            polarized: false,
            jones: [1.0, 0.0],
            points: self.points.clone(),
            keys: Vec::new(),
            launch: WaveLedger::default(),
            history: Vec::new(),
            junction: None,
            next: self.next,
            beam: self.beam,
            u,
            q: self.q,
            at: self.at,
            cache: None,
            departures: None,
            rest: Vec::new(),
            report: WaveReport { ledger: WaveLedger { launched, ..Default::default() }, ..Default::default() },
            fft: Fft2::new(),
        }
    }

    fn tilt_warn(&mut self, depth: f64) {
        let dx = self.u.grid.pitch_mm[0];
        if depth > dx * dx / self.u.wavelength_in_medium_mm() {
            self.report.warn(SamplingWarning::TiltProjection { depth_mm: depth });
        }
    }

    /// A flat standalone surface on the chain: a mirror, a splitter's arm, a
    /// grating's order or a thin lens, met at `p` (the field's plane is across
    /// the beam there) and left along `out`.
    ///
    /// The surface's outline clips the field, its edge weighted by exact cell
    /// coverage of its projection along the beam. Light outside it passes the
    /// element by: where the chain goes on through (a transmitted arm, a thin
    /// lens) it is still on the chain, carried unchanged beside the element;
    /// where the chain turns (a reflection, a diffracted order) it leaves it,
    /// booked as `passed_by`. What the element itself does not pass on is
    /// booked by cause: a mirror's absorption, a splitter's other arm
    /// (`other_arm`) and what it absorbs, a grating's other orders
    /// (`untracked_orders`).
    #[allow(clippy::too_many_arguments)]
    fn flat_element(
        &mut self,
        s: &oc_core::scene3::SurfaceObject3,
        response: &SurfaceResponse,
        h: &oc_core::seq3::SeqHit3,
        p: Vec3,
        n_s: Vec3,
        out: &Beam,
        from_front: bool,
    ) -> Result<(), WaveError> {
        let lam = self.lam;
        let beam = self.beam;
        #[derive(Clone, Copy)]
        enum Loss {
            Absorbed,
            /// The other arm's power fraction; the rest of the loss is absorbed.
            Arm(f64),
            Orders,
        }
        let refuse = |kind: &str| {
            WaveError::OutOfRange(format!(
                "{kind} at #{} ({}): no wave model for this response on the path",
                h.object.0, h.label
            ))
        };
        let kind = oc_core::build3::response_kind(response);
        let unit = |v: Option<f64>| v.map(|x| x.clamp(0.0, 1.0));
        // A coating's junction: what met it and the share it carries on.
        let mut coat: Option<(f64, f64)> = None;
        // The amplitude inside the outline, a thin lens's focal length and
        // centre, and where the rest goes.
        //
        // The amplitudes' signs make every lossless element unitary, so that
        // arms recombined after it conserve power. A splitter is the two-port
        // of a dielectric interface (bundle M14; the Stokes relations):
        // reflection −√R from its front, +√R from its back, transmission √T
        // both ways; a perfect conductor is that interface's limit, so a
        // mirror reflects −√R. One phase per element is a convention a real
        // coating fixes otherwise, and it moves fringes, never the power.
        let (amp, lens, loss): (f64, Option<(f64, Vec3)>, Loss) = match (response, h.action) {
            (SurfaceResponse::Mirror { reflectivity }, SeqAction::Reflect) => {
                (-unit(reflectivity.sample(lam)).ok_or_else(|| refuse(kind))?.sqrt(), None, Loss::Absorbed)
            }
            (SurfaceResponse::ExplicitRt { reflectance, transmittance, .. }, SeqAction::Reflect | SeqAction::Transmit) => {
                let r = unit(reflectance.sample(lam)).ok_or_else(|| refuse(kind))?;
                let t = unit(transmittance.sample(lam)).ok_or_else(|| refuse(kind))?;
                if h.action == SeqAction::Reflect {
                    (if from_front { -r.sqrt() } else { r.sqrt() }, None, Loss::Arm(t.min(1.0 - r)))
                } else {
                    (t.sqrt(), None, Loss::Arm(r.min(1.0 - t)))
                }
            }
            (SurfaceResponse::Grating { efficiency, .. }, SeqAction::Diffract) => {
                (unit(efficiency.sample(lam)).ok_or_else(|| refuse(kind))?.sqrt(), None, Loss::Orders)
            }
            (SurfaceResponse::Grating { zero_order, .. }, _) => {
                (unit(zero_order.sample(lam)).ok_or_else(|| refuse(kind))?.sqrt(), None, Loss::Orders)
            }
            (SurfaceResponse::ThinLens { focal_length_mm }, _) => (1.0, Some((*focal_length_mm, s.common.transform.pos)), Loss::Absorbed),
            // A zero-thickness refracting surface has the same medium on both
            // sides: nothing to reflect, nothing to bend.
            (SurfaceResponse::DielectricFresnel { .. }, SeqAction::Transmit) if (h.n_before - h.n_after).abs() < 1e-12 => (1.0, None, Loss::Absorbed),
            // A coating: each plane wave of the field takes its own angle's
            // amplitude (crate::planar), applied here; the handover then
            // carries the field on unchanged.
            (SurfaceResponse::Coating { coating }, SeqAction::Reflect | SeqAction::Transmit) => {
                let reflect = h.action == SeqAction::Reflect;
                let n0 = h.n_before;
                let pol = self.pol;
                // The amplitude on a table of cosines, read between its
                // entries: it is smooth in the angle, and a transfer matrix
                // per plane wave would cost the field's size in stacks.
                let table: Vec<(Cx, f64, f64)> = (0..=1024)
                    .map(|i| {
                        let c = (i as f64 / 1024.0).max(1e-6);
                        let at = |pp: bool| crate::planar::coating_arm(coating, reflect, n0, lam, c, pp);
                        match pol.is_p() {
                            Some(pp) => at(pp),
                            None => at(false).and_then(|a| at(true).map(|b| ((a.0 + b.0) * 0.5, 0.5 * (a.1 + b.1), 0.5 * (a.2 + b.2)))),
                        }
                    })
                    .collect::<Result<_, _>>()?;
                let read = |c: f64| -> (Cx, f64, f64) {
                    let x = (c.clamp(0.0, 1.0) * 1024.0).min(1023.999999);
                    let (i, f) = (x.floor() as usize, x - x.floor());
                    let (a, b) = (table[i], table[i + 1]);
                    (a.0 * (1.0 - f) + b.0 * f, a.1 * (1.0 - f) + b.1 * f, a.2 * (1.0 - f) + b.2 * f)
                };
                let input = self.u.power();
                let (v, other, absorbed, left) = crate::planar::surface_filter(&self.u, &beam, n_s, &read, &mut self.fft)?;
                coat = Some((input, if input > 0.0 { 1.0 - (other + absorbed) / input } else { 0.0 }));
                let l = &mut self.report.ledger;
                l.other_arm += other;
                l.absorbed_element += absorbed;
                l.left_window += left;
                self.u = v;
                if beam.d.dot(n_s).abs() < 1.0 - 1e-12 && self.pol == Pol::Unpolarized {
                    self.report.approx(format!("coating #{}: met obliquely by a field standing for unpolarized light, its s and p amplitudes averaged", h.object.0));
                }
                self.report.approx("coatings on the path: each plane wave takes its own angle's amplitude for the chain's s or p, from the transfer matrix; a plate-backed coating's plate is incoherent, as it declares");
                (1.0, None, Loss::Absorbed)
            }
            // A polarizer or a retarder acts on the pair of components
            // (`element_polarized`); in each component's own pass it hands
            // the field over unchanged.
            (SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. }, _) if self.polarized => (1.0, None, Loss::Absorbed),
            (SurfaceResponse::MeasuredTable { .. }, _) => {
                let e = WaveError::OutOfRange(format!(
                    "table at #{} ({}): a measured table gives R and T without their phases, so the field has no amplitude to take from it",
                    h.object.0, h.label
                ));
                self.report.rejected.push(e.to_string());
                return Err(e);
            }
            _ => {
                let e = refuse(kind);
                self.report.rejected.push(e.to_string());
                return Err(e);
            }
        };
        let k = self.u.k_mm();
        let fin = move |r: Vec3| -> Cx {
            match lens {
                // Distance from the lens's centre within its plane.
                Some((f, c)) => {
                    let v = r - c;
                    let v = v - n_s * v.dot(n_s);
                    Cx::expi(-k * v.length_squared() / (2.0 * f))
                }
                None => Cx::real(amp),
            }
        };
        let pass_by = matches!(h.action, SeqAction::Transmit | SeqAction::ThinLens);
        let pb = if pass_by { 1.0 } else { 0.0 };

        // The outline across the beam.
        let g = self.u.grid;
        let square = n_s.dot(beam.d).abs() > 1.0 - 1e-12;
        let tf = s.common.transform;
        let to_beam = |x: f64, y: f64| {
            let w = tf.transform_point(Vec3::new(x, y, 0.0));
            let (a, b) = beam.coords(w - beam.d * (w - beam.origin).dot(beam.d));
            [a, b]
        };
        let cov = match &s.geometry {
            SurfaceGeometry3::Plane { shape } => shape_coverage(&g, shape, square, &to_beam)?,
            SurfaceGeometry3::SurfaceOfRevolution { .. } => vec![1.0; g.len()],
        };

        // Each sample's share: inside, where the element's factor acts; outside,
        // passed by. Physically a cell sends on f·|t|² + (1 − f)·pb of its
        // power; the sample (f·t + (1 − f)·pb)·U carries less where an edge
        // crosses it, and the difference is diffracted beyond the grid's band.
        let da = g.cell_area_mm2();
        let (mut inside, mut lost_in, mut beside, mut beyond, mut sent, mut total) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut clipped = false;
        for j in 0..g.n[1] {
            for i in 0..g.n[0] {
                let ix = g.idx(i, j);
                let pw = self.u.data[ix].norm_sqr() * da;
                if pw == 0.0 {
                    continue;
                }
                let f = cov[ix];
                clipped |= f < 1.0;
                let (a, b) = g.xy(i, j);
                let r_b = beam.point(a, b);
                let t = fin(onto_plane(r_b, beam.d, p, n_s).unwrap_or(r_b));
                let carried = (t * f + Cx::real((1.0 - f) * pb)).norm_sqr();
                inside += f * pw;
                lost_in += f * pw * (1.0 - t.norm_sqr());
                beside += (1.0 - f) * (1.0 - pb) * pw;
                beyond += (f * t.norm_sqr() + (1.0 - f) * pb - carried) * pw;
                sent += carried * pw;
                total += pw;
            }
        }
        let (v, depth) = if !clipped {
            hand_over(&self.u, &beam, n_s, out, &fin)
        } else {
            let mut u_in = self.u.clone();
            for (z, f) in u_in.data.iter_mut().zip(&cov) {
                *z *= *f;
            }
            let (mut v, depth) = hand_over(&u_in, &beam, n_s, out, &fin);
            if pass_by {
                let mut u_by = self.u.clone();
                for (z, f) in u_by.data.iter_mut().zip(&cov) {
                    *z *= 1.0 - *f;
                }
                let (v_by, _) = hand_over(&u_by, &beam, n_s, out, &|_| Cx::ONE);
                for (a, b) in v.data.iter_mut().zip(&v_by.data) {
                    *a += *b;
                }
                // The light beside the element keeps its direction, which the
                // outgoing plane holds as a phase ramp across its samples.
                let ramp = k * (beam.d - out.d).length() * g.pitch_mm[0].max(g.pitch_mm[1]);
                if ramp > std::f64::consts::PI {
                    self.report.warn(SamplingWarning::Undersampled {
                        what: format!("the light passing beside #{}", h.object.0),
                        max_phase_step_rad: ramp,
                    });
                }
            }
            (v, depth)
        };
        let l = &mut self.report.ledger;
        match loss {
            Loss::Absorbed => l.absorbed_element += lost_in,
            Loss::Arm(other) => {
                l.other_arm += inside * other;
                l.absorbed_element += (lost_in - inside * other).max(0.0);
            }
            Loss::Orders => l.untracked_orders += lost_in,
        }
        l.passed_by += beside;
        l.subcell_projection += beyond;
        l.resampling += sent - v.power();
        // Where arms part: the element, what met it, and what this arm
        // carries on of it, for the network's ledger (`book_history`).
        let junction = match response {
            SurfaceResponse::ExplicitRt { .. } => Some((Bucket::OtherArm, inside, amp * amp)),
            SurfaceResponse::Grating { .. } => Some((Bucket::UntrackedOrders, inside, amp * amp)),
            SurfaceResponse::Coating { .. } => coat.map(|(input, share)| (Bucket::OtherArm, input, share)),
            _ => None,
        };
        if let Some((bucket, input, share)) = junction {
            self.note_junction(Junction { object: h.object.0, kind: JunctionKind::Split, bucket, input, share, foreign_absorbed: 0.0, foreign_family: 0.0 });
        }
        // A flat mirror's handover is image theory, exact at any tilt;
        // anything else seen obliquely is a projection.
        let exact = matches!(response, SurfaceResponse::Mirror { .. }) && h.action == SeqAction::Reflect;
        if !exact {
            self.tilt_warn(depth);
        }
        let mut note = match lens {
            Some((f, _)) => format!("thin lens #{} ({}), f = {f} mm", h.object.0, h.label),
            None => format!("{:?} at #{} ({}), amplitude {amp:+.6}", h.action, h.object.0, h.label),
        };
        if clipped {
            note += &format!(
                "; its outline clips the beam, edges weighted by exact cell coverage, {:.3e} of the power inside it{}",
                inside / f64::max(total, f64::MIN_POSITIVE),
                if pass_by { ", the rest passing beside it on the chain" } else { ", the rest passing beside it off the chain" }
            );
        }
        self.report.steps.push(element_step(0.0, note));
        if let Some((f, _)) = lens {
            let dx = g.pitch_mm[0];
            let edge = 0.5 * g.width_mm(0) * 2f64.sqrt();
            if k * edge * dx / f.abs() > std::f64::consts::PI {
                self.report.warn(SamplingWarning::Undersampled {
                    what: format!("thin lens #{} f = {f} mm", h.object.0),
                    max_phase_step_rad: k * edge * dx / f.abs(),
                });
            }
        }
        self.u = v;
        Ok(())
    }

    fn event(&mut self, idx: usize) -> Result<(), WaveError> {
        let (p, _, ev) = self.points[idx].clone();
        // A diffracted readout takes the last step itself, onto the detector,
        // and so does an arm read out to be added to others.
        if matches!(ev, Ev::End) && (self.face || matches!(self.spec.readout, Readout::Diffracted { .. })) {
            self.merge_lobes();
            return Ok(());
        }
        // Free space up to this event, onto a window fitted to the beam there
        // when the one it has would no longer suit it; both components, and
        // every order carried apart, onto the same one.
        let z = (p - self.beam.origin).dot(self.beam.d);
        if z > 1e-12 {
            self.free_space(z, p)?;
        }
        let (beam_in, n_in) = (self.beam, self.u.n_medium);
        self.element_any(idx, &ev, p)?;
        if !self.rest.is_empty() {
            // The other orders through the same element, each about its own
            // reference ray; the report keeps the first one's step.
            let beam_out = self.beam;
            let n = self.report.steps.len();
            let mut lobes = std::mem::take(&mut self.rest);
            for l in &mut lobes {
                std::mem::swap(&mut self.u, &mut l.u);
                std::mem::swap(&mut self.uy, &mut l.uy);
                std::mem::swap(&mut self.at, &mut l.at);
                self.beam = beam_in;
                let r = self.element_any(idx, &ev, p);
                std::mem::swap(&mut self.u, &mut l.u);
                std::mem::swap(&mut self.uy, &mut l.uy);
                std::mem::swap(&mut self.at, &mut l.at);
                r?;
                self.report.steps.truncate(n);
            }
            self.rest = lobes;
            self.beam = beam_out;
        }
        if let Some(d) = self.departures.take() {
            self.report.approx(d.note());
        }
        self.paraxial(&ev, p, &beam_in, n_in)?;
        let transfer_ahead = self.points[idx + 1..].iter().any(|(_, _, e)| matches!(e, Ev::Transfer { .. }));
        if !transfer_ahead {
            self.merge_lobes();
        } else if let Ev::Hit(h) = &ev {
            self.split_orders(h, &beam_in)?;
        }
        if matches!(ev, Ev::End) {
            self.merge_lobes();
        }
        Ok(())
    }

    /// The orders carried apart added back into one field.
    fn merge_lobes(&mut self) {
        if self.rest.is_empty() {
            return;
        }
        self.at = [0.0; 4];
        for l in std::mem::take(&mut self.rest) {
            for (a, b) in self.u.data.iter_mut().zip(&l.u.data) {
                *a += *b;
            }
            if let (Some(uy), Some(ly)) = (&mut self.uy, &l.uy) {
                for (a, b) in uy.data.iter_mut().zip(&ly.data) {
                    *a += *b;
                }
            }
        }
    }

    /// Where a modulator's field meets a traced element further on, carry
    /// its orders apart: the field's angular spectrum cut into the cells of
    /// the device's lattice of orders (directions `λ/p` apart in the device's
    /// plane [Harvey & Vernold 1998], about the specular reflection for a DMD,
    /// whose facets only weight the orders, and about the beam the chain
    /// follows for an LCoS), each order a field of its own with its own
    /// reference ray, its direction. A transfer then traces each order along
    /// its own direction rather than all of them along one; the fields add
    /// back to the one field at the detector.
    fn split_orders(&mut self, h: &oc_core::seq3::SeqHit3, beam_in: &Beam) -> Result<(), WaveError> {
        let scene = Arc::clone(&self.scene);
        let Some(s) = scene.surfaces.iter().find(|s| s.common.id == h.object) else { return Ok(()) };
        let t = s.common.transform;
        let nrm = t.rotate(Vec3::Z);
        let response = if beam_in.d.dot(nrm) < 0.0 { &s.response_front } else { &s.response_back };
        let SurfaceResponse::Modulator { device, .. } = response else { return Ok(()) };
        self.merge_lobes();
        self.at = [0.0; 4];
        let (ex, ey) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
        let out = self.beam;
        let centre = match device.kind {
            ModulatorKind::Dmd(_) => beam_in.d - nrm * (2.0 * beam_in.d.dot(nrm)),
            ModulatorKind::Lcos(_) => out.d,
        };
        let side = if out.d.dot(nrm) >= 0.0 { 1.0 } else { -1.0 };
        let lam = self.u.wavelength_in_medium_mm();
        let step = lam / device.pitch_mm();
        let (cx, cy) = (centre.dot(ex), centre.dot(ey));
        let g = self.u.grid;
        let n = g.n;
        let spectrum = |f: &Field2, fft: &mut Fft2| {
            let mut b = f.data.clone();
            crate::fft::ifftshift2(&mut b, n);
            fft.forward(&mut b, n, g.cell_area_mm2());
            b
        };
        let sx = spectrum(&self.u, &mut self.fft);
        let sy = self.uy.as_ref().map(|u| spectrum(u, &mut self.fft));
        // Each plane wave's order: the lattice point nearest its direction's
        // part along the device's plane.
        let (u0, v0) = (self.u.carrier[0] / lam, self.u.carrier[1] / lam);
        let mut order = vec![(0i64, 0i64); n[0] * n[1]];
        let mut power: std::collections::HashMap<(i64, i64), f64> = std::collections::HashMap::new();
        for ky in 0..n[1] {
            let fy = g.fft_freq(1, ky) + v0;
            for kx in 0..n[0] {
                let fx = g.fft_freq(0, kx) + u0;
                let (tx, ty) = (lam * fx, lam * fy);
                let tz = (1.0 - tx * tx - ty * ty).max(0.0).sqrt();
                let d = out.x * tx + out.y * ty + out.d * tz;
                let o = (((d.dot(ex) - cx) / step).round() as i64, ((d.dot(ey) - cy) / step).round() as i64);
                let k = ky * n[0] + kx;
                order[k] = o;
                *power.entry(o).or_default() += sx[k].norm_sqr() + sy.as_ref().map_or(0.0, |y| y[k].norm_sqr());
            }
        }
        let total: f64 = power.values().sum();
        let mut ranked: Vec<((i64, i64), f64)> = power.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        let kept: Vec<(i64, i64)> = ranked.iter().filter(|(_, p)| *p >= 1e-4 * total).take(MAX_LOBES).map(|x| x.0).collect();
        // An order's direction across the outgoing beam.
        let dir = |o: (i64, i64)| -> Option<[f64; 2]> {
            let (a, b) = (cx + o.0 as f64 * step, cy + o.1 as f64 * step);
            let z2 = 1.0 - a * a - b * b;
            (z2 > 0.0).then(|| {
                let d = ex * a + ey * b + nrm * (side * z2.sqrt());
                [d.dot(out.x), d.dot(out.y)]
            })
        };
        let kept: Vec<((i64, i64), [f64; 2])> = kept.into_iter().filter_map(|o| dir(o).map(|d| (o, d))).collect();
        if kept.len() <= 1 {
            // One order: the field is traced about its direction.
            if let Some((_, d)) = kept.first() {
                self.at = [0.0, 0.0, d[0], d[1]];
            }
            return Ok(());
        }
        // Every order joins its nearest kept one.
        let mut join: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();
        let nearest = |o: (i64, i64)| {
            kept.iter()
                .enumerate()
                .min_by_key(|(_, (k, _))| (k.0 - o.0).pow(2) + (k.1 - o.1).pow(2))
                .map_or(0, |(i, _)| i)
        };
        let df = g.freq_pitch(0) * g.freq_pitch(1);
        let mut lobe = |spec: &[Cx], which: usize, like: &Field2, fft: &mut Fft2| -> Field2 {
            let mut b: Vec<Cx> = spec
                .iter()
                .zip(&order)
                .map(|(z, o)| if *join.entry(*o).or_insert_with(|| nearest(*o)) == which { *z } else { Cx::ZERO })
                .collect();
            fft.inverse(&mut b, n, df);
            crate::fft::fftshift2(&mut b, n);
            let mut f = like.clone();
            f.data = b;
            f
        };
        let mut lobes = Vec::with_capacity(kept.len());
        for (i, (_, d)) in kept.iter().enumerate() {
            let u = lobe(&sx, i, &self.u, &mut self.fft);
            let uy = match (&sy, &self.uy) {
                (Some(y), Some(like)) => Some(lobe(y, i, like, &mut self.fft)),
                _ => None,
            };
            lobes.push(Lobe { u, uy, at: [0.0, 0.0, d[0], d[1]] });
        }
        let first = lobes.remove(0);
        self.u = first.u;
        self.uy = first.uy;
        self.at = first.at;
        self.rest = lobes;
        let count = self.rest.len() + 1;
        if let Some(st) = self.report.steps.last_mut() {
            st.note += &format!("; its field carried on as {count} of the device's orders apart, each traced about its own direction through the elements ahead");
        }
        self.report.approx(format!(
            "the field leaving modulator #{} carried as {count} of its orders apart (the device's lattice, λ/p apart; at most {MAX_LOBES}, each of at least 1e-4 of the power, a weaker order joining its nearest), each traced through the lenses ahead about its own direction; their sum is the field",
            h.object.0
        ));
        Ok(())
    }

    /// The field `z` on to `p`, both components onto one window; the
    /// reference's wavefront and ray with it.
    fn free_space(&mut self, z: f64, p: Vec3) -> Result<(), WaveError> {
        let plan = self.regrid_plan(z)?;
        let (v, r) = match &plan {
            Some(pl) => self.regrid_apply(&self.u.clone(), z, pl)?,
            None => propagate(&self.u, z, Propagator::Auto, &mut self.fft)?,
        };
        merge(&mut self.report, r);
        self.u = v;
        if let Some(uy) = self.uy.take() {
            let (vy, ry) = match &plan {
                Some(pl) => self.regrid_apply(&uy, z, pl)?,
                None => propagate(&uy, z, Propagator::Auto, &mut self.fft)?,
            };
            let mut ry = ry;
            ry.steps.clear();
            merge(&mut self.report, ry);
            self.uy = Some(vy);
        }
        let mut lobes = std::mem::take(&mut self.rest);
        for l in &mut lobes {
            for f in std::iter::once(&mut l.u).chain(l.uy.as_mut()) {
                let (v, mut r) = match &plan {
                    Some(pl) => self.regrid_apply(f, z, pl)?,
                    None => propagate(f, z, Propagator::Auto, &mut self.fft)?,
                };
                r.steps.clear();
                merge(&mut self.report, r);
                *f = v;
            }
        }
        // An order free space has emptied, its power booked as it left,
        // rejoins the strongest.
        let pw = |u: &Field2, uy: &Option<Field2>| u.power() + uy.as_ref().map_or(0.0, |f| f.power());
        let total: f64 = pw(&self.u, &self.uy) + lobes.iter().map(|l| pw(&l.u, &l.uy)).sum::<f64>();
        let (spent, kept): (Vec<Lobe>, Vec<Lobe>) = lobes.into_iter().partition(|l| pw(&l.u, &l.uy) < 1e-9 * total);
        for l in spent {
            for (a, b) in self.u.data.iter_mut().zip(&l.u.data) {
                *a += *b;
            }
            if let (Some(uy), Some(ly)) = (&mut self.uy, &l.uy) {
                for (a, b) in uy.data.iter_mut().zip(&ly.data) {
                    *a += *b;
                }
            }
        }
        self.rest = kept;
        self.beam.origin = p;
        let iso = self.beam.iso();
        self.u.frame = iso;
        if let Some(uy) = &mut self.uy {
            uy.frame = iso;
        }
        for q in &mut self.q {
            *q += Cx::real(z);
        }
        let walk = |at: &mut [f64; 4]| {
            let tz = (1.0 - at[2] * at[2] - at[3] * at[3]).max(1e-12).sqrt();
            at[0] += z * at[2] / tz;
            at[1] += z * at[3] / tz;
        };
        walk(&mut self.at);
        for l in &mut self.rest {
            l.u.frame = iso;
            if let Some(uy) = &mut l.uy {
                uy.frame = iso;
            }
            walk(&mut l.at);
        }
        Ok(())
    }

    /// The element at event `idx`, on the field of one component or two.
    fn element_any(&mut self, idx: usize, ev: &Ev, p: Vec3) -> Result<(), WaveError> {
        if self.uy.is_some() && !matches!(ev, Ev::End | Ev::Aperture(_)) {
            return self.element_polarized(idx, ev, p);
        }
        if matches!(ev, Ev::Aperture(_)) {
            if let Some(uy) = self.uy.take() {
                // An aperture acts on both components alike.
                self.element(idx, ev.clone(), p)?;
                let ux = std::mem::replace(&mut self.u, uy);
                let n = self.report.steps.len();
                self.element(idx, ev.clone(), p)?;
                self.report.steps.truncate(n);
                self.uy = Some(std::mem::replace(&mut self.u, ux));
                return Ok(());
            }
        }
        self.element(idx, ev.clone(), p)
    }

    /// The reference the next transfer's rays follow: the beam's paraxial
    /// wavefront about the reference ray.
    fn reference(&self) -> Reference {
        let c = |q: Cx| (Cx::ONE / q).re;
        Reference { curv: [c(self.q[0]), c(self.q[1])], tilt: [self.at[2], self.at[3]], centre: [self.at[0], self.at[1]] }
    }

    /// The reference's wavefront and ray through the element at an event,
    /// from the geometry alone: a transfer's paraxial matrices from rays
    /// traced through it, a thin lens's power, a plate as the free space its
    /// glass is worth (`t/n`); a flat element hands the beam's frame on as it
    /// is. `beam_in` and `n_in` are the beam's frame and medium before it.
    fn paraxial(&mut self, ev: &Ev, p: Vec3, beam_in: &Beam, n_in: f64) -> Result<(), WaveError> {
        match ev {
            Ev::Transfer { hit, .. } => {
                let probe = probe_scene(&self.scene, hit.object);
                let out = self.beam;
                let n_out = hit.n_after;
                if let Some(m) = crate::transfer::abcd(&probe, self.lam, beam_in, &out, n_in, n_out) {
                    for (q, m) in self.q.iter_mut().zip(m) {
                        *q = crate::transfer::q_through(*q, m, n_in, n_out);
                    }
                } else {
                    self.report.approx(format!("#{}: its paraxial matrices could not be traced, so the reference wavefront is carried through it unchanged", hit.object.0));
                }
                let mut ats: Vec<[f64; 4]> = vec![self.at];
                ats.extend(self.rest.iter().map(|l| l.at));
                for a in &mut ats {
                    if *a == [0.0; 4] {
                        continue;
                    }
                    match crate::transfer::ray_through(&probe, self.lam, beam_in, &out, a[0], a[1], [a[2], a[3]]) {
                        Some((x, t, _)) => *a = [x[0], x[1], t[0], t[1]],
                        None => self.report.approx(format!("#{}: the reference ray of one of the field's orders misses it, and is carried on as it was", hit.object.0)),
                    }
                }
                self.at = ats[0];
                for (l, a) in self.rest.iter_mut().zip(&ats[1..]) {
                    l.at = *a;
                }
            }
            Ev::Hit(h) => {
                let s = self.scene.surfaces.iter().find(|s| s.common.id == h.object);
                let lens = s.and_then(|s| {
                    let r = if beam_in.d.dot(s.common.transform.rotate(Vec3::Z)) < 0.0 { &s.response_front } else { &s.response_back };
                    match r {
                        SurfaceResponse::ThinLens { focal_length_mm } => Some(*focal_length_mm),
                        _ => None,
                    }
                });
                if let Some(f) = lens {
                    for q in &mut self.q {
                        *q = Cx::ONE / (Cx::ONE / *q - Cx::real(1.0 / f));
                    }
                    // The frame turns with the chief ray: a reference ray
                    // turns by its own height on the lens less the chief's.
                    for a in std::iter::once(&mut self.at).chain(self.rest.iter_mut().map(|l| &mut l.at)) {
                        a[2] -= a[0] / f;
                        a[3] -= a[1] / f;
                    }
                }
            }
            Ev::Plate { first, second, exit, plate } => {
                let (n_g, n_out) = (plate.n_glass, plate.n_out);
                let glass = (second.point - first.point).length();
                let worth = (first.point - p).length() / n_in + glass / n_g + (*exit - second.point).length() / n_out;
                for q in &mut self.q {
                    *q = (*q * (1.0 / n_in) + Cx::real(worth)) * n_out;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The element at event `idx`, on the field `u` (one component of two,
    /// while `element_polarized` runs it).
    fn element(&mut self, idx: usize, ev: Ev, p: Vec3) -> Result<(), WaveError> {
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
                let r = self.reference();
                let map = match &self.cache {
                    Some((i, m)) if *i == idx && m.fits(self.u.grid, &r) => Arc::clone(m),
                    _ => {
                        let m = Arc::new(crate::transfer::trace_map(&probe, self.u.grid, self.u.lambda_um, &beam, &out, rays, &r)?);
                        self.cache = Some((idx, Arc::clone(&m)));
                        m
                    }
                };
                let (v, st) = crate::transfer::apply(&map, &self.u, &out, last.n_after)?;
                let l = &mut self.report.ledger;
                // The rim and the mount stop light; the faces reflect it into
                // a path the chain does not follow.
                l.absorbed_aperture += st.rim_stopped + st.stopped;
                l.subcell_projection += st.rim_beyond;
                l.other_arm += st.reflected;
                l.left_window += st.left_window;
                l.resampling += st.carried - st.stopped - st.reflected - st.left_window - v.power();
                let what = if first.kind == SeqHitKind::Body { "face of body" } else { "curved surface" };
                let rim = if st.rim_resolved {
                    format!("its rim laid on the field with exact cell coverage, {:.3e} of the power stopped there", st.rim_stopped / before.max(f64::MIN_POSITIVE))
                } else if st.stopped > 1e-12 * before {
                    "its rim resolved only to the ray grid".to_string()
                } else {
                    "no rim within the window".to_string()
                };
                self.report.steps.push(element_step(
                    0.0,
                    format!(
                        "{what} #{} ({}): field traced through it along {} rays ({} passed), chief path {:.6} mm; rays move up to {:.4} mm across the beam; {rim}; {:.3e} of the power stopped elsewhere, {:.3e} reflected by the faces",
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
                if !st.rim_resolved && st.stopped > 1e-12 * before {
                    self.report.approx(format!(
                        "{what} #{}: the edge of what passes is not star-shaped about the chief ray, so it is resolved only to the ray grid",
                        first.object.0
                    ));
                }
                let k_in = self.u.k_mm();
                let orders = !self.rest.is_empty() || self.at != [0.0; 4];
                let note = self.departures.get_or_insert(Departures { what, id: first.object.0, curvature: st.curvature, slab: st.slab_mm, k: k_in, orders, ..Default::default() });
                note.add(before, st.arriving_ray_angle, st.arriving_ray_rms);
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
            Ev::Plate { first, second, exit, plate } => {
                self.plate_event(&first, &second, exit, &plate)?;
                return Ok(());
            }
            Ev::Aperture(ai) => {
                let a = &self.scene.apertures[ai];
                let t = &a.common.transform;
                let plane = &a.shape.plane_local;
                let n_e = t.rotate(plane.normal);
                let o_e = t.transform_point(plane.origin);
                let square = n_e.dot(beam.d).abs() > 1.0 - 1e-12;
                // A point of the aperture's plane, along the beam onto the
                // plane across it here.
                let to_beam = |pu: f64, pv: f64| {
                    let w = t.transform_point(plane.origin + plane.u * pu + plane.v * pv);
                    let (x, y) = beam.coords(w - beam.d * (w - beam.origin).dot(beam.d));
                    [x, y]
                };
                // As the tracer has it: light passes inside the opening, is
                // stopped outside it within the outer extent, and passes
                // beyond the outer extent; with none, the mask is unbounded.
                let g = self.u.grid;
                let open = shape_coverage(&g, &a.shape.clear_opening, square, &to_beam)?;
                let pass: Vec<f64> = match &a.shape.outer_extent {
                    None => open,
                    Some(ext) => {
                        let outer = shape_coverage(&g, ext, square, &to_beam)?;
                        let inside = shape_rim(&a.shape.clear_opening)
                            .into_iter()
                            .all(|p| oc_core::geom3::primitives::planar_shape_contains(ext, p));
                        if !inside {
                            self.report.approx(format!(
                                "aperture #{}: its opening reaches past its outer extent, where the two covers are combined as if it did not",
                                a.common.id.0
                            ));
                        }
                        outer.iter().zip(&open).map(|(o, c)| (1.0 - o + c).clamp(0.0, 1.0)).collect()
                    }
                };
                let (stopped, beyond) = crate::elements::apply_coverage(&mut self.u, &pass)?;
                self.report.ledger.absorbed_aperture += stopped;
                self.report.ledger.subcell_projection += beyond;
                let id = a.common.id.0;
                if !square {
                    // Not square to the beam, the opening is its projection
                    // along the beam: an approximation by the depth it spans.
                    self.tilt_warn(plane_depth(&beam, &g, o_e, n_e));
                }
                let note = if stopped > 0.0 {
                    format!(
                        "aperture #{} at its plane, edges weighted by exact cell coverage{}: {:.3e} of the power stopped",
                        id,
                        if square { "" } else { " of their projection along the beam" },
                        stopped / (stopped + self.u.power() + beyond).max(f64::MIN_POSITIVE)
                    )
                } else {
                    format!("aperture #{id} at its plane, clear of the beam")
                };
                self.report.steps.push(element_step(0.0, note));
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
                if let SurfaceResponse::Modulator { device, .. } = response {
                    self.u = modulator_step(&self.u, &beam, &out, s, device, &h, &self.spec, self.pol, &mut self.report)?;
                } else {
                    let (s, response) = (s.clone(), response.clone());
                    self.flat_element(&s, &response, &h, p, n_s, &out, from_front)?;
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
    fn finish(mut self, revision: u64) -> Result<WavePathResult, WaveError> {
        if let Some(uy) = self.uy.take() {
            // The two components are read alike and add in power.
            let twin = self.twin(uy);
            let mut rx = self.finish(revision)?;
            let ry = twin.finish(revision)?;
            for (a, b) in rx.power.iter_mut().zip(&ry.power) {
                *a += *b;
            }
            rx.arriving += ry.arriving;
            let l = &mut rx.report.ledger;
            l.detected += ry.report.ledger.detected;
            let mut extra = ry.report.ledger;
            extra.numerical = 0.0;
            add_losses(l, &extra);
            l.numerical = 0.0;
            l.close();
            rx.report.inconsistencies.clear();
            rx.report.check_ledger();
            for w in ry.report.warnings {
                rx.report.warn(w);
            }
            rx.field_y = Some(ry.field);
            return Ok(rx);
        }
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
            return self.finish_diffracted(samples_per_pixel, (hw, hh), [nu, nv], (du, dv), revision);
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
        self.report.close_checked();
        Ok(WavePathResult {
            revision,
            detector: spec.detector,
            bins,
            power,
            field: self.u,
            field_y: None,
            arriving: arriving_power,
            report: self.report,
        })
    }
}

impl ChainJob {
    /// The diffracted readout: the field carried from where it is straight
    /// onto a window of the plane across the beam at the detector's centre,
    /// at `s × s` points a pixel, then binned into the pixels.
    fn finish_diffracted(
        mut self,
        s: usize,
        (hw, hh): (f64, f64),
        [nu, nv]: [usize; 2],
        (du, dv): (f64, f64),
        revision: u64,
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
            self.report.close_checked();
            self.u = zf;
            return Ok(WavePathResult { revision, detector: spec.detector, bins: [nu, nv], power, field: self.u, field_y: None, arriving, report: self.report });
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
        self.report.close_checked();
        self.u = zf;
        self.u.frame = beam.turned(plane, beam.d).iso();
        Ok(WavePathResult {
            revision,
            detector: spec.detector,
            bins: [nu, nv],
            power,
            field: self.u,
            field_y: None,
            arriving,
            report: self.report,
        })
    }
}

/// One arm read out onto its detector's own grid, ready to be added to the
/// other arms.
struct FaceField {
    /// The detector's sub-pixel grid, in its own frame, and the arm's whole
    /// field there (carrier and phase included).
    grid: Grid2,
    data: Vec<Cx>,
    /// The obliquity of the arm's beam on the face, and its direction.
    cos: f64,
    dir: Vec3,
    frame: Isometry3,
    lambda_um: f64,
    n_medium: f64,
    /// The power the arm brought to the detector's plane.
    arriving: f64,
    /// The second component's field on the same points, if the arm has one.
    data_y: Option<Vec<Cx>>,
    report: WaveReport,
    /// The launch's ledger and the arm's history, for the network's ledger.
    launch: WaveLedger,
    history: Vec<HistoryEntry>,
    /// What reading the arm onto the detector's grid took (evanescent waves,
    /// what missed the face): in `report` and in `history` already.
    readout: WaveLedger,
    events: usize,
}

impl ChainJob {
    /// The arm's field on the detector's own sub-pixel grid, `s × s` points a
    /// pixel at the centres of equal parts of it: the zoomed angular
    /// spectrum onto the face when it is square to the beam with its axes
    /// along the beam's, the spectrum rotated onto it otherwise
    /// ([`crate::asm_tilted`]). Every arm is read on the same points, so the
    /// fields can be added.
    fn face(mut self, s: usize) -> Result<FaceField, WaveError> {
        if self.next < self.points.len() {
            return Err(WaveError::OutOfRange("an arm read out before its chain is done".into()));
        }
        if let Some(uy) = self.uy.take() {
            let twin = self.twin(uy);
            let mut fx = self.face(s)?;
            let fy = twin.face(s)?;
            fx.arriving += fy.arriving;
            fx.data_y = Some(fy.data);
            // The second component's readout took its own share.
            add_losses(&mut fx.report.ledger, &fy.readout);
            add_losses(&mut fx.readout, &fy.readout);
            if let Some(last) = fx.history.last_mut() {
                add_losses(&mut last.delta, &fy.readout);
            }
            return Ok(fx);
        }
        let beam = self.beam;
        let det = self
            .scene
            .detectors
            .iter()
            .find(|d| d.common.id.0 == self.spec.detector.0)
            .ok_or_else(|| WaveError::OutOfRange("no such detector".into()))?;
        let (hw, hh) = match det.geometry {
            DetectorGeometry3::Rectangle { half_width_mm, half_height_mm } => (half_width_mm, half_height_mm),
            _ => return Err(WaveError::OutOfRange("the wave readout needs a rectangular detector".into())),
        };
        let [nu, nv] = det.bins;
        let t = det.common.transform;
        let n_det = t.rotate(Vec3::Z);
        let (du, dv) = (2.0 * hw / nu as f64, 2.0 * hh / nv as f64);
        let z = (t.pos - beam.origin).dot(beam.d);
        if z <= 0.0 {
            return Err(WaveError::OutOfRange("the detector is behind the field's last plane".into()));
        }
        let s = s.max(1);
        let (ps, qs) = (du / s as f64, dv / s as f64);
        let fg = centred(Grid2::new([nu * s, nv * s], [ps, qs], [0.0, 0.0])?);
        let arriving = self.u.power();
        let n_max = self.u.grid.n[0].max(self.u.grid.n[1]);
        let square = n_det.dot(beam.d).abs() > 1.0 - 1e-12;
        let (eu, ev) = (t.rotate(Vec3::X), t.rotate(Vec3::Y));
        let along = |a: Vec3, b: Vec3| a.dot(b).abs() > 1.0 - 1e-12;
        let aligned = if square && along(eu, beam.x) && along(ev, beam.y) {
            Some(([nu * s, nv * s], [ps, qs]))
        } else if square && along(eu, beam.y) && along(ev, beam.x) {
            Some(([nv * s, nu * s], [qs, ps]))
        } else {
            None
        };
        let mut data = vec![Cx::ZERO; fg.len()];
        let rep = if let Some((n, pitch)) = aligned {
            let plane = beam.origin + beam.d * z;
            let c = beam.coords(t.pos - beam.d * (t.pos - plane).dot(beam.d));
            let out = centred(Grid2::new(n, pitch, [c.0, c.1])?);
            let pad = if n_max >= 1024 { 1 } else { 2 };
            let (zf, rep) = crate::asm_zoom(&self.u, z, out, pad, &mut self.fft)?;
            let k = zf.k_mm();
            for j in 0..out.n[1] {
                for i in 0..out.n[0] {
                    let (a, b) = out.xy(i, j);
                    let l = t.inverse_transform_point(plane + beam.x * a + beam.y * b);
                    let (iu, iv) = (((l.x + hw) / ps).floor(), ((l.y + hh) / qs).floor());
                    if iu < 0.0 || iv < 0.0 || iu >= (nu * s) as f64 || iv >= (nv * s) as f64 {
                        continue;
                    }
                    data[fg.idx(iu as usize, iv as usize)] = zf.data[out.idx(i, j)] * Cx::expi(k * (zf.carrier[0] * a + zf.carrier[1] * b));
                }
            }
            rep
        } else {
            let pad = if n_max <= 512 { 4 } else if n_max <= 1024 { 2 } else { 1 };
            let (zt, rep) = crate::asm_tilted(&self.u, t, fg, pad, &mut self.fft)?;
            data = zt.data;
            rep
        };
        // The readout's own losses (evanescent waves, what missed the face)
        // are the arm's, booked in its history as one more event.
        let mut report = self.report;
        let before = report.ledger;
        merge(&mut report, rep);
        let readout = losses_between(&report.ledger, &before);
        let mut history = self.history;
        history.push(HistoryEntry { key: "readout".to_string(), junction: None, delta: readout });
        Ok(FaceField {
            grid: fg,
            data,
            cos: n_det.dot(beam.d).abs(),
            dir: beam.d,
            frame: t,
            lambda_um: self.lam,
            n_medium: self.u.n_medium,
            arriving,
            data_y: None,
            report,
            launch: self.launch,
            history,
            readout,
            events: self.points.len(),
        })
    }
}

/// Add `d`'s losses (everything but what was launched and detected) into `l`.
fn add_losses(l: &mut WaveLedger, d: &WaveLedger) {
    l.absorbed_aperture += d.absorbed_aperture;
    l.absorbed_element += d.absorbed_element;
    l.band_limit_removed += d.band_limit_removed;
    l.spectral_projection += d.spectral_projection;
    l.subcell_projection += d.subcell_projection;
    l.evanescent += d.evanescent;
    l.left_window += d.left_window;
    l.other_family += d.other_family;
    l.untracked_orders += d.untracked_orders;
    l.other_arm += d.other_arm;
    l.passed_by += d.passed_by;
    l.resampling += d.resampling;
    l.other_ports += d.other_ports;
    l.numerical += d.numerical;
}

/// `after − before`, losses only.
fn losses_between(after: &WaveLedger, before: &WaveLedger) -> WaveLedger {
    WaveLedger {
        launched: 0.0,
        detected: 0.0,
        absorbed_aperture: after.absorbed_aperture - before.absorbed_aperture,
        absorbed_element: after.absorbed_element - before.absorbed_element,
        band_limit_removed: after.band_limit_removed - before.band_limit_removed,
        spectral_projection: after.spectral_projection - before.spectral_projection,
        subcell_projection: after.subcell_projection - before.subcell_projection,
        evanescent: after.evanescent - before.evanescent,
        left_window: after.left_window - before.left_window,
        other_family: after.other_family - before.other_family,
        untracked_orders: after.untracked_orders - before.untracked_orders,
        other_arm: after.other_arm - before.other_arm,
        passed_by: after.passed_by - before.passed_by,
        resampling: after.resampling - before.resampling,
        other_ports: after.other_ports - before.other_ports,
        numerical: after.numerical - before.numerical,
    }
}

/// The network's ledger from its arms' histories. Each event is booked once
/// per distinct history (arms carry one field up to where they part). At a
/// junction where several arms part with one field (`Split`), the element's
/// absorption and what passes beside it are booked with the first arm, each
/// arm's own resampling with that arm, and the shares no arm carries on,
/// `input·(1 − Σ shares)` less the absorption, once, to the junction's
/// bucket: with both orders of a grating followed, its untracked orders are
/// `1 − η − η₀` of what met it, not `(1 − η) + (1 − η₀)`. An element met
/// again by arms of other histories (arms recombining) books no other port
/// there: that light reached the detector by the other arm, and what the
/// element sends by ports no arm reads is the remainder, `other_ports`. At a
/// `PerBeam` junction each arm books its own share of the device's loss and
/// takes back what it booked for the other arm's.
fn book_history(l: &mut WaveLedger, arms: &[&[HistoryEntry]]) {
    use std::collections::{HashMap, HashSet};
    let mut met: HashMap<u64, usize> = HashMap::new();
    for a in arms {
        let mut objs: Vec<u64> = a.iter().filter_map(|e| e.junction.map(|j| j.object)).collect();
        objs.sort_unstable();
        objs.dedup();
        for o in objs {
            *met.entry(o).or_default() += 1;
        }
    }
    // The shares carried on by the arms parting at one event: by the history
    // before it and the element, each distinct event once.
    let mut shares: HashMap<String, f64> = HashMap::new();
    let mut counted: HashSet<String> = HashSet::new();
    for a in arms {
        let mut history = String::new();
        for e in *a {
            let before = history.clone();
            history.push_str(&e.key);
            history.push('|');
            if let Some(j) = e.junction {
                if j.kind == JunctionKind::Split && counted.insert(history.clone()) {
                    *shares.entry(format!("{before}#{}", j.object)).or_default() += j.share;
                }
            }
        }
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut parted: HashSet<String> = HashSet::new();
    for a in arms {
        let mut history = String::new();
        for e in *a {
            let before = history.clone();
            history.push_str(&e.key);
            history.push('|');
            if !seen.insert(history.clone()) {
                continue;
            }
            let mut d = e.delta;
            if let Some(j) = e.junction {
                let n_met = met.get(&j.object).copied().unwrap_or(0);
                match j.kind {
                    JunctionKind::Split => {
                        let tag = format!("{before}#{}", j.object);
                        let taken = shares.get(&tag).copied().unwrap_or(j.share);
                        if parted.insert(tag) {
                            if taken > j.share * (1.0 + 1e-12) {
                                // Other arms part here and carry their shares on.
                                *bucket_mut(&mut d, j.bucket) = j.input * (1.0 - taken) - d.absorbed_element;
                            } else if n_met >= 2 {
                                // Arms of other histories meet the element: what
                                // went by its other port reached the detector.
                                *bucket_mut(&mut d, j.bucket) = 0.0;
                            }
                        } else {
                            // The arms met the element with one field: what it
                            // absorbs or passes by was booked with the first.
                            d = WaveLedger { resampling: d.resampling, numerical: d.numerical, ..Default::default() };
                        }
                    }
                    JunctionKind::PerBeam => {
                        if n_met >= 2 {
                            d.absorbed_element -= j.foreign_absorbed;
                            d.other_family -= j.foreign_family;
                        }
                    }
                }
            }
            add_losses(l, &d);
        }
    }
}

/// Several arms of one coherent mode, added on the detector's grid.
///
/// The power a pixel catches is `Σ |Σ_c U_c √cos θ_c|² ΔA` over its points:
/// each arm's field weighted by the root of its obliquity on the face, so
/// that one arm alone reads as it would by itself. Two arms crossing at an
/// angle make fringes of period `λ/|t₁ − t₂|`, `t` the directions' components
/// along the face; a period under two of the readout's points is flagged.
///
/// The ledger counts each arm's losses, an event the arms share once (they
/// carry the same field up to where they part). What a splitter sends into
/// another arm is not lost, so a splitter met by several arms books no
/// `other_arm`; the light that leaves by the ports no arm reads (a
/// recombining splitter's other output, what misses the detector once the
/// arms meet) is then the remainder, `other_ports`.
fn combine_arms(scene: &Scene3, spec: &WavePathSpec, faces: Vec<FaceField>, revision: u64) -> Result<WavePathResult, WaveError> {
    let det = scene
        .detectors
        .iter()
        .find(|d| d.common.id.0 == spec.detector.0)
        .ok_or_else(|| WaveError::OutOfRange("no such detector".into()))?;
    let [nu, nv] = det.bins;
    let g = faces[0].grid;
    let s = g.n[0] / nu;
    let (ps, qs) = (g.pitch_mm[0], g.pitch_mm[1]);
    let mut data = vec![Cx::ZERO; g.len()];
    let two = faces.iter().any(|f| f.data_y.is_some());
    let mut data_y = vec![Cx::ZERO; if two { g.len() } else { 0 }];
    for f in &faces {
        let w = f.cos.sqrt();
        for (a, b) in data.iter_mut().zip(&f.data) {
            *a += *b * w;
        }
        if let Some(dy) = &f.data_y {
            for (a, b) in data_y.iter_mut().zip(dy) {
                *a += *b * w;
            }
        }
    }
    let mut power = vec![0.0; nu * nv];
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let k = g.idx(i, j);
            let y = if two { data_y[k].norm_sqr() } else { 0.0 };
            power[(j / s) * nu + i / s] += (data[k].norm_sqr() + y) * ps * qs;
        }
    }
    let mut report = WaveReport::default();
    for (c, f) in faces.iter().enumerate() {
        for st in &f.report.steps {
            let mut st = st.clone();
            st.note = format!("arm {}: {}", c + 1, st.note);
            report.steps.push(st);
        }
        for w in &f.report.warnings {
            report.warn(w.clone());
        }
        for v in &f.report.validity {
            report.flag(v.clone());
        }
        for a in &f.report.approximated {
            report.approx(a.clone());
        }
        for s in &f.report.inconsistencies {
            if !report.inconsistencies.contains(s) {
                report.inconsistencies.push(s.clone());
            }
        }
    }
    // The fringes between each pair of arms, against the readout's points.
    let n_det = det.common.transform.rotate(Vec3::Z);
    let lam_mm = faces[0].lambda_um / 1000.0 / faces[0].n_medium;
    for a in 0..faces.len() {
        for b in a + 1..faces.len() {
            let (da, db) = (faces[a].dir, faces[b].dir);
            let ta = da - n_det * da.dot(n_det);
            let tb = db - n_det * db.dot(n_det);
            let gap = (ta - tb).length();
            if gap > 0.0 {
                let period = lam_mm / gap;
                let h = ps.min(qs);
                if period < 2.0 * h {
                    report.warn(SamplingWarning::Undersampled {
                        what: format!(
                            "the fringes of arms {} and {} on detector #{}, {period:.4e} mm apart, against readout points {h:.4e} mm apart",
                            a + 1,
                            b + 1,
                            spec.detector.0
                        ),
                        max_phase_step_rad: 2.0 * std::f64::consts::PI * h / period,
                    });
                }
            }
        }
    }
    // The network's ledger: the launch once, then every arm's history.
    let mut l = WaveLedger { launched: faces[0].launch.launched, ..Default::default() };
    add_losses(&mut l, &faces[0].launch);
    let histories: Vec<&[HistoryEntry]> = faces.iter().map(|f| f.history.as_slice()).collect();
    book_history(&mut l, &histories);
    l.detected = power.iter().sum();
    l.other_ports = l.launched - l.accounted();
    let lam = faces[0].lambda_um;
    let n = faces[0].n_medium;
    let arms = faces.len();
    report.steps.push(element_step(
        0.0,
        format!(
            "{arms} arms added as fields on detector #{}'s own grid, {s}×{s} points a pixel, each weighted by the root of its obliquity on the face",
            spec.detector.0
        ),
    ));
    if !two {
        report.approx("arms added in a scalar field: their polarizations taken as the same, so the fringes' visibility is an upper bound");
    }
    report.ledger = l;
    report.check_ledger();
    let mut field = Field2::zeros(g, lam)?.in_medium(n)?;
    field.frame = faces[0].frame;
    let field_y = two.then(|| {
        let mut f = field.clone();
        f.data = data_y;
        f
    });
    field.data = data;
    Ok(WavePathResult {
        revision,
        detector: spec.detector,
        bins: [nu, nv],
        power,
        field,
        field_y,
        arriving: faces.iter().map(|f| f.arriving).sum(),
        report,
    })
}

/// The most spectral lines a wave path runs, each a coherent computation of
/// its own.
pub const MAX_WAVE_LINES: usize = 16;

/// A wave path in resumable steps, for a worker that must be able to drop a
/// stale job between steps: [`WaveJob::new`] plans every chain from the
/// source to the detector, each [`WaveJob::step`] carries the fields
/// through a number of the chains' events (the free space up to an event
/// and the element there), and [`WaveJob::finish`] reads the detector.
/// However the work is sliced, the result is the same to the bit:
/// [`wave_path`] is this job run in one step.
///
/// **Arms.** Where splitter arms recombine on the detector (an
/// interferometer) the axial ray reaches it by several chains
/// (`oc_trace::extract_sequences3`). Each carries the same launched field
/// along its own arm, and the arms' fields are added on the detector's own
/// grid before the power is formed ([`combine_arms`]): the coherent sum of
/// paths the scattering network of bundle M00 asks for, for networks without
/// feedback.
///
/// **Lines.** A source of several spectral lines is a set of mutually
/// incoherent modes (E. Wolf, *J. Opt. Soc. Am.* 72, 343 (1982); bundle M15):
/// each line runs coherently through every chain at its own wavelength, with
/// its share of the power, and the lines add in power. A continuous spectrum
/// is refused: sampled into lines, it would be a choice the path should not
/// make for the source.
///
/// The job keeps one shared copy of the scene, so it answers for the revision
/// it was started at whatever happens to the scene meanwhile.
pub struct WaveJob {
    scene: Arc<Scene3>,
    spec: WavePathSpec,
    revision: u64,
    /// One per line, strongest first.
    modes: Vec<Mode>,
}

struct Mode {
    lambda_um: f64,
    /// What the mode is, for the report: its line, and its polarization.
    label: String,
    chains: Vec<Chain>,
}

/// Whether the chains' light must be carried as two polarization components:
/// when one of them meets a polarizer, a retarder or an LCoS, or a flat
/// coating or a plate obliquely, where s and p differ.
fn polarization_matters(scene: &Scene3, seqs: &[oc_core::seq3::SequencePrescription3], half: f64, lam: f64) -> bool {
    seqs.iter().any(|seq| {
        let mut prev = seq.launch.point;
        seq.hits.iter().enumerate().any(|(k, h)| {
            let d = (h.point - prev).normalize();
            prev = h.point;
            let n = h.normal.normalize();
            let oblique = d.dot(n).abs() < 1.0 - 1e-9;
            let surface = scene.surfaces.iter().find(|s| s.common.id == h.object);
            let response = surface.map(|s| if d.dot(s.common.transform.rotate(Vec3::Z)) < 0.0 { &s.response_front } else { &s.response_back });
            let telling = h.kind == SeqHitKind::Surface
                && response.is_some_and(|r| match r {
                    SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. } => true,
                    SurfaceResponse::Modulator { device, .. } => matches!(device.kind, ModulatorKind::Lcos(_)),
                    SurfaceResponse::Coating { .. } => oblique && surface.is_some_and(|s| matches!(s.geometry, SurfaceGeometry3::Plane { .. })),
                    _ => false,
                });
            let plate = h.kind == SeqHitKind::Body
                && oblique
                && seq.hits.get(k + 1).is_some_and(|h2| {
                    let next = seq.hits.get(k + 2).map_or(seq.terminal.point, |x| x.point);
                    let from = if k == 0 { seq.launch.point } else { seq.hits[k - 1].point };
                    plate_of(scene, h, h2, from, next, half, lam).is_some()
                });
            telling || plate
        })
    })
}

enum Chain {
    Running(Box<ChainJob>),
    /// An arm done and read out onto the detector's grid.
    Face(Box<FaceField>),
    /// A lone chain of one line among several, finished.
    Done(Box<WavePathResult>, usize),
}

impl Chain {
    fn events(&self) -> usize {
        match self {
            Chain::Running(j) => j.points.len(),
            Chain::Face(f) => f.events,
            Chain::Done(_, n) => *n,
        }
    }
    fn done(&self) -> usize {
        match self {
            Chain::Running(j) => j.next,
            Chain::Face(f) => f.events,
            Chain::Done(_, n) => *n,
        }
    }
}

impl std::fmt::Debug for WaveJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveJob")
            .field("revision", &self.revision)
            .field("lines", &self.modes.len())
            .field("chains", &self.modes.iter().map(|m| m.chains.len()).sum::<usize>())
            .field("events", &self.events())
            .field("done", &self.done())
            .finish_non_exhaustive()
    }
}

impl WaveJob {
    /// Plan every chain from `spec.source` to `spec.detector`, at each of the
    /// source's lines.
    pub fn new(scene: &Scene3, spec: &WavePathSpec, revision: u64) -> Result<Self, WaveError> {
        if !(spec.power.is_finite() && spec.power > 0.0 && spec.window_mm > 0.0 && spec.grid_n >= 8) {
            return Err(WaveError::OutOfRange("a wave path needs power, a window and at least 8 samples".into()));
        }
        let src = scene
            .sources
            .iter()
            .find(|s| s.id == spec.source)
            .ok_or_else(|| WaveError::OutOfRange("no such source".into()))?;
        let mut lines: Vec<(f64, f64)> = match &src.spectrum {
            oc_core::scene::Spectrum::Monochromatic(l) => vec![(*l, 1.0)],
            oc_core::scene::Spectrum::Lines(v) => v.iter().copied().filter(|l| l.1 > 0.0).collect(),
            oc_core::scene::Spectrum::Continuum(_) => {
                return Err(WaveError::OutOfRange(format!(
                    "source #{} emits a continuous spectrum; the wave path runs each spectral line coherently and adds the lines in power: give it a single line or a few lines",
                    spec.source.0
                )))
            }
        };
        if lines.is_empty() {
            return Err(WaveError::OutOfRange(format!("source #{} emits no line of positive weight", spec.source.0)));
        }
        if lines.len() > MAX_WAVE_LINES {
            return Err(WaveError::OutOfRange(format!(
                "source #{} emits {} lines; the wave path runs at most {MAX_WAVE_LINES}, each a coherent computation: give it fewer",
                spec.source.0,
                lines.len()
            )));
        }
        lines.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = lines.iter().map(|l| l.1).sum();
        let shared = Arc::new(scene.clone());
        let mut modes = Vec::with_capacity(lines.len());
        for (lam, w) in lines {
            let seqs = oc_trace::extract_sequences3(scene, spec.source, spec.detector, Some(lam), revision)
                .map_err(|e| WaveError::OutOfRange(format!("no chain from the source to the detector: {e}")))?;
            let face = seqs.len() > 1;
            let pols: Vec<Option<[f64; 2]>> =
                if polarization_matters(scene, &seqs, 0.5 * spec.window_mm, lam) { vec![Some([1.0, 0.0]), Some([0.0, 1.0])] } else { vec![None] };
            let share = 1.0 / pols.len() as f64;
            for jones in pols {
                let chains = seqs
                    .iter()
                    .cloned()
                    .map(|seq| ChainJob::new(&shared, spec, seq, spec.power * w / total * share, face, jones).map(|j| Chain::Running(Box::new(j))))
                    .collect::<Result<Vec<_>, _>>()?;
                let label = match jones {
                    None => format!("λ {lam} µm"),
                    Some(j) => format!("λ {lam} µm, launched along {}", if j[1] == 0.0 { "x" } else { "y" }),
                };
                modes.push(Mode { lambda_um: lam, label, chains });
            }
        }
        Ok(WaveJob { scene: shared, spec: *spec, revision, modes })
    }

    /// The scene revision the job answers for.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The chains' events (surfaces, aperture planes, the detector), and how
    /// many the fields have been carried through.
    pub fn events(&self) -> usize {
        self.modes.iter().flat_map(|m| &m.chains).map(Chain::events).sum()
    }
    pub fn done(&self) -> usize {
        self.modes.iter().flat_map(|m| &m.chains).map(Chain::done).sum()
    }

    /// How many coherent chains (arms) and incoherent lines the job holds.
    pub fn arms(&self) -> usize {
        self.modes.iter().map(|m| m.chains.len()).max().unwrap_or(0)
    }
    pub fn lines(&self) -> usize {
        let mut l: Vec<f64> = self.modes.iter().map(|m| m.lambda_um).collect();
        l.dedup();
        l.len()
    }
    /// The mutually incoherent modes the job adds in power: its lines, each
    /// once, or twice (s and p) where an element tells them apart.
    pub fn modes(&self) -> usize {
        self.modes.len()
    }

    /// Carry the fields through up to `events` more events, chain after
    /// chain. `Ok(true)` once every chain is done; an element a field has no
    /// model for is an error naming it, and the job is not resumable after
    /// one. A chain that finishes among several is read out at once, so the
    /// job holds one full field at a time.
    pub fn step(&mut self, events: usize) -> Result<bool, WaveError> {
        let lone = self.modes.len() == 1 && self.modes[0].chains.len() == 1;
        let s = match self.spec.readout {
            Readout::Diffracted { samples_per_pixel } => samples_per_pixel.max(1),
            Readout::Projected => 2,
        };
        let mut left = events;
        for m in &mut self.modes {
            let several = m.chains.len() > 1;
            for c in &mut m.chains {
                let Chain::Running(job) = c else { continue };
                if left == 0 {
                    return Ok(false);
                }
                let k = (job.points.len() - job.next).min(left);
                let finished = job.step(k)?;
                left -= k;
                if finished && !lone {
                    let Chain::Running(job) = std::mem::replace(c, Chain::Done(Box::new(placeholder_result(self.revision, self.spec.detector)?), 0)) else {
                        unreachable!()
                    };
                    let n = job.points.len();
                    *c = if several {
                        Chain::Face(Box::new(job.face(s)?))
                    } else {
                        Chain::Done(Box::new(job.finish(self.revision)?), n)
                    };
                }
            }
        }
        Ok(self.done() == self.events())
    }

    /// Read the detector: each pixel's power. Refused before every chain is
    /// done.
    pub fn finish(self) -> Result<WavePathResult, WaveError> {
        if self.done() < self.events() {
            return Err(WaveError::OutOfRange(format!(
                "the wave job has {} of {} events still to run",
                self.events() - self.done(),
                self.events()
            )));
        }
        let WaveJob { scene, spec, revision, mut modes } = self;
        if modes.len() == 1 && modes[0].chains.len() == 1 {
            if let Some(Chain::Running(job)) = modes[0].chains.pop() {
                return job.finish(revision);
            }
            unreachable!("a lone chain stays running until it is finished");
        }
        let count = modes.len();
        let mut lines: Vec<f64> = modes.iter().map(|m| m.lambda_um).collect();
        lines.dedup();
        let (lines, pols) = (lines.len(), count / lines.len().max(1));
        let mut results = Vec::with_capacity(count);
        for m in modes {
            let lam = m.label.clone();
            let r = if m.chains.len() > 1 {
                let faces = m
                    .chains
                    .into_iter()
                    .map(|c| match c {
                        Chain::Face(f) => Ok(*f),
                        _ => Err(WaveError::OutOfRange("an arm was not read out".into())),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                combine_arms(&scene, &spec, faces, revision)?
            } else {
                match m.chains.into_iter().next() {
                    Some(Chain::Done(r, _)) => *r,
                    _ => return Err(WaveError::OutOfRange("a line's chain was not finished".into())),
                }
            };
            results.push((lam, r));
        }
        if results.len() == 1 {
            return Ok(results.pop().expect("one").1);
        }
        // Lines add in power: mutually incoherent modes.
        let mut iter = results.into_iter();
        let (lam0, mut total) = iter.next().expect("several");
        for st in &mut total.report.steps {
            st.note = format!("{lam0}: {}", st.note);
        }
        for (lam, r) in iter {
            if r.bins != total.bins {
                return Err(WaveError::Mismatch("the lines were read on different detector grids".into()));
            }
            for (a, b) in total.power.iter_mut().zip(&r.power) {
                *a += *b;
            }
            total.arriving += r.arriving;
            let l = &mut total.report.ledger;
            l.launched += r.report.ledger.launched;
            l.detected += r.report.ledger.detected;
            add_losses(l, &r.report.ledger);
            for st in r.report.steps {
                let mut st = st;
                st.note = format!("{lam}: {}", st.note);
                total.report.steps.push(st);
            }
            for w in r.report.warnings {
                total.report.warn(w);
            }
            for v in r.report.validity {
                total.report.flag(v);
            }
            for a in r.report.approximated {
                total.report.approx(a);
            }
            for s in r.report.inconsistencies {
                if !total.report.inconsistencies.contains(&s) {
                    total.report.inconsistencies.push(s);
                }
            }
        }
        if lines > 1 {
            total.report.approx(format!("{lines} spectral lines added in power, as mutually incoherent modes; the field shown is the first mode's, {lam0}"));
        }
        if pols > 1 {
            total.report.approx(format!(
                "unpolarized light as two mutually incoherent modes, launched along the beam frame's x and y, each a field of two polarization components, added in power; the field shown is the first mode's, {lam0}"
            ));
        }
        Ok(total)
    }
}

/// A result holding nothing, for the moment a chain is moved out of its slot.
fn placeholder_result(revision: u64, detector: DetectorId) -> Result<WavePathResult, WaveError> {
    Ok(WavePathResult {
        revision,
        detector,
        bins: [0, 0],
        power: Vec::new(),
        field: Field2::zeros(Grid2::square(1, 1.0)?, 1.0)?,
        field_y: None,
        arriving: 0.0,
        report: WaveReport::default(),
    })
}

/// Carry a coherent beam from `spec.source` to `spec.detector` along every
/// chain the chief ray takes: a [`WaveJob`] run in one step.
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
    pol: Pol,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    let t = &s.common.transform;
    let n_s = t.rotate(Vec3::Z);
    // The device-plane grid: the beam window's footprint, at a few samples a
    // pixel, and no coarser than the beam's own samples (a window fitted to a
    // narrow beam is finer than a few samples a pixel, and the take-up would
    // read the device's facets between samples it does not have), up to 32.
    let p = device.pitch_mm();
    let per = spec.device_samples_per_pixel.max(2).max(((2.0 * p / u.grid.pitch_mm[0].min(u.grid.pitch_mm[1])).ceil() as usize).min(32));
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
    let dev_in = on_dev.power();
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
            // Of what the device reflects, the part along its director is
            // modulated and the part across it is not (Zhang, You & Chu 2014):
            // the chain brings the one component its beam takes
            // (`element_polarized`), the other booked there.
            if pol != Pol::Projected {
                return Err(WaveError::OutOfRange(format!(
                    "LCoS #{}: met by a field of one component, where it needs the two its beams split",
                    s.common.id.0
                )));
            }
            report.approx("LCoS on the wave path: the field's two polarization components split on the director, each into its own beam; the two never interfere");
            lcos_reflect(&on_dev, device, beam_kind)?
        }
    };
    // The device grid, finer than the outgoing window, holds plane waves that
    // window cannot: read between device samples, they would fold back in at
    // directions they do not have, and weight the orders wrongly. Take them
    // out first, about the reflected family's carrier, as a re-grid does,
    // and book them. The window's samples, `p` apart across the outgoing
    // ray, are `p/√(1 − (e·d)²)` apart along a device axis `e` on its plane.
    let scale = if dev_in > 0.0 { before / dev_in } else { 0.0 };
    let pb = u.grid.pitch_mm[0].max(u.grid.pitch_mm[1]);
    let along = |e: Vec3| pb / (1.0 - e.dot(out.d).powi(2)).max(1e-12).sqrt();
    let band = Grid2::new(reflected.grid.n, [along(t.rotate(Vec3::X)), along(t.rotate(Vec3::Y))], [0.0, 0.0])?;
    let (reflected, cut) = band_to(&reflected, &band, &mut Fft2::new());
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
    // The device's own ledger, in this beam's units: the device plane sees the
    // beam's power spread over its oblique footprint, so its samples sum to
    // more than the beam's, by the scale below. What the take-up across the
    // outgoing ray does not carry is resampling.
    report.ledger.absorbed_element += rep.ledger.absorbed_element * scale;
    report.ledger.other_family += rep.ledger.other_family * scale;
    report.ledger.subcell_projection += rep.ledger.subcell_projection * scale;
    report.ledger.spectral_projection += cut * scale;
    report.ledger.numerical += rep.ledger.numerical * scale;
    report.ledger.resampling += (rep.ledger.detected - cut) * scale - v.power();
    for f in rep.validity {
        report.flag(f);
    }
    for a in rep.approximated {
        report.approx(a);
    }
    report.steps.push(element_step(
        0.0,
        format!(
            "{} #{} ({}): laid on {}×{} device samples, {} per pixel; {:.3e} of the power, beyond the outgoing window's band, taken out before the take-up",
            oc_core::build3::response_kind(&s.response_front),
            h.object.0,
            h.label,
            n[0],
            n[1],
            per,
            cut * scale / before.max(f64::MIN_POSITIVE)
        ),
    ));
    Ok(v)
}
