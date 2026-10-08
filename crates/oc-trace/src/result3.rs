//! The 3D trace result: segments, detector readouts, the energy ledger, and the
//! capability report (§9.8, §9.9, §17).
//!
//! Two rules shape these types. First, **every launched photon is accounted
//! for**: the [`EnergyLedger`] balances launched flux against everything that
//! could remove it, and flux killed by a threshold or a cap lands in
//! `truncated_flux` rather than vanishing. Second, **fidelity is reported, never
//! faked**: a scene the tracer cannot handle exactly produces a
//! [`CapabilityReport`] with a typed [`CapabilityIssue`], not a quietly
//! substituted geometry (§3.5).

use oc_core::id::{
    ApertureId, BodyId, DetectorId, ObjectId, PatchId, PathId, SourceId, SurfaceRef,
};
use oc_core::scene3::{Detector3, DetectorMeasure};
use oc_core::vec::Vec2;
use oc_core::vec3::{Ray3, Vec3};

/// One photon bundle launched into the scene.
///
/// The tracer takes launch rays explicitly; turning a [`Source3`](oc_core::scene3::Source3)
/// into a deterministic set of these is PR 3D-06's job, so the transport engine
/// can be validated on hand-built rays without a sampler in the way.
#[derive(Debug, Clone, Copy)]
pub struct LaunchRay3 {
    pub ray: Ray3,
    pub lambda_um: f64,
    pub flux: f64,
    /// The source this ray came from, for diagnostics. Not required for transport.
    pub source: Option<SourceId>,
    /// The polarization state the ray starts in, on its own transverse frame.
    ///
    /// `None` is unpolarized, and is not the same as "unset": an unpolarized
    /// state looks identical in every frame, so a source that declares nothing
    /// needs no frame chosen for it. Where one is given, `flux` still governs
    /// the total: the state is rescaled to it, so the two cannot disagree.
    pub pol: Option<oc_core::polarization::PolRay>,
}

/// Flags on a traced segment, for rendering and diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentFlags {
    /// A Fresnel back-reflection (or its descendants): the non-primary path.
    pub ghost: bool,
    /// Born of total internal reflection.
    pub tir: bool,
}

/// A traced ray segment (§9.9). `opl0`/`opl1` are the cumulative optical path
/// length in mm at the segment's ends, the propagation clock a wavefront view
/// reads.
#[derive(Debug, Clone, Copy)]
pub struct Segment3 {
    pub a: Vec3,
    pub b: Vec3,
    pub flux_start: f64,
    pub flux_end: f64,
    pub lambda_um: f64,
    pub opl0_mm: f64,
    pub opl1_mm: f64,
    pub path_id: PathId,
    pub parent_path_id: Option<PathId>,
    pub flags: SegmentFlags,
}

/// One recorded detector hit (§9.7).
#[derive(Debug, Clone, Copy)]
pub struct DetectorHit3 {
    pub point_world: Vec3,
    /// Position on the detector face, local `(u, v)` in mm.
    pub uv_mm: Vec2,
    /// Incident direction in the detector's local frame.
    pub incident_dir_local: Vec3,
    pub lambda_um: f64,
    pub incident_flux: f64,
    /// Flux after the detector's quantum efficiency.
    pub signal: f64,
    pub opl_mm: f64,
    pub path_id: PathId,
    pub parent_path_id: Option<PathId>,
    pub depth: u32,
    pub ghost_depth: u32,
    /// The arriving polarization state, written on this ray's own frame.
    /// `s0` is `incident_flux`.
    pub stokes: oc_core::polarization::Stokes,
    /// That frame's transverse axis, in world coordinates. Without it the
    /// `s1`/`s2` above name nothing: they are components on an axis, and a
    /// reader that does not know which axis cannot use them.
    pub frame_x: Vec3,
    /// How many hits this branch made, since launch, on the objects
    /// [`crate::TraceConfig3::count_bodies`] names: the reflections in a
    /// light pipe, say. 0 when it names none.
    pub counted: u32,
}

/// A scene object, by kind and by its index in the scene's list of that kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ObjectRef3 {
    /// The launch itself: what a branch carries before its first interaction.
    Source,
    Surface(usize),
    /// A closed body, and the patch of it that was hit (a tube's bore, its
    /// outer wall, an end ring).
    Body(usize, u64),
    Aperture(usize),
    Detector(usize),
}

/// What one object did to the light of one band
/// ([`crate::TraceConfig3::object_ledger`]).
///
/// Every watt the ledger removes is booked here once, against the object
/// where it left the light: a mirror's absorption at the mirror, a stop's
/// block at the stop, the bulk absorption on the way out of an object against
/// that object, an escape or a culled branch against the object that sent the
/// branch on its way. Summed over every object and band, the removals are the
/// [`EnergyLedger`]'s, so an object table that closes is the ledger's closure
/// told object by object. `hits` and `incident` count what arrived, which
/// includes light that passed a probe detector and went on.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ObjectLedger3 {
    pub hits: u64,
    pub incident: f64,
    pub absorbed_surface: f64,
    pub absorbed_volume: f64,
    pub blocked: f64,
    pub detected: f64,
    pub escaped: f64,
    pub truncated: f64,
    pub numerical: f64,
}

impl ObjectLedger3 {
    /// What this object took out of the light, every way.
    pub fn removed(&self) -> f64 {
        self.absorbed_surface + self.absorbed_volume + self.blocked + self.detected + self.escaped + self.truncated + self.numerical
    }

    pub fn add(&mut self, o: &ObjectLedger3) {
        self.hits += o.hits;
        self.incident += o.incident;
        self.absorbed_surface += o.absorbed_surface;
        self.absorbed_volume += o.absorbed_volume;
        self.blocked += o.blocked;
        self.detected += o.detected;
        self.escaped += o.escaped;
        self.truncated += o.truncated;
        self.numerical += o.numerical;
    }
}

/// Where light was absorbed or blocked: one entry of [`TraceResult3::deposits`]
/// ([`crate::TraceConfig3::deposit_bodies`], [`crate::TraceConfig3::deposit_apertures`]).
/// A segment's bulk absorption is split over four equal steps of its length,
/// each placed at its step's middle with the power the step took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deposit3 {
    pub object: ObjectRef3,
    /// World mm.
    pub point: Vec3,
    pub lambda_um: f64,
    /// W.
    pub flux: f64,
}

/// One row of [`TraceResult3::objects`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectEntry3 {
    pub object: ObjectRef3,
    /// Inside [`crate::TraceConfig3::ledger_band_um`] or not.
    pub in_band: bool,
    pub ledger: ObjectLedger3,
}

/// The role a spawned child branch plays at an interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildRole {
    /// Continued through the interface (Snell across a real transition, or
    /// undeviated where both sides share a medium).
    Transmit,
    /// Specular reflection: a mirror, a splitter's reflected arm, or a Fresnel
    /// ghost.
    Reflect,
    /// Total internal reflection (R = 1, the primary path).
    Tir,
    /// The ideal thin-lens bend of a standalone surface.
    ThinLens,
    /// A diffraction grating's working order (its zero order is `Reflect` or
    /// `Transmit`).
    Diffract,
}

/// What an event-log entry records (see [`TraceEvent3`]).
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEventKind {
    /// An interaction with a body patch or a standalone surface. `children`
    /// lists the branches actually spawned, in spawn order; an entry that
    /// terminated the branch (absorber, unsupported response, every child
    /// culled) has an empty list.
    Interact {
        surface: SurfaceRef,
        /// True for a closed body's patch, false for a standalone surface.
        body: bool,
        /// Unit normal at the hit, oriented against the incident ray.
        normal: Vec3,
        /// Refractive index of the medium the ray arrived through.
        n1: f64,
        /// Refractive index on the far side (equal to `n1` at a zero-thickness
        /// surface).
        n2: f64,
        children: Vec<(ChildRole, PathId)>,
    },
    /// The branch met a detector face. `terminal` follows the detector's
    /// termination policy; a probe records the hit and the branch continues
    /// under the same path id.
    DetectorHit {
        detector: DetectorId,
        uv: Vec2,
        terminal: bool,
    },
    /// The branch was absorbed by an opaque aperture mask.
    ApertureBlocked { aperture: ApertureId },
    /// Nothing ahead: the branch left the scene bounds.
    Escaped,
}

/// One entry of the opt-in event log (`TraceConfig3::record_events`).
///
/// Segments record where flux *went*; events record what *happened* and to
/// which scene object, which is what sequence extraction and interaction-level
/// diagnostics need. Within one path id, entries appear in propagation order;
/// entries of different branches interleave. With the parallel feature, path
/// ids are only unique per launch ray — single-launch traces (the sequence
/// extractor's case) are unambiguous.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceEvent3 {
    pub path_id: PathId,
    pub parent: Option<PathId>,
    pub depth: u32,
    pub ghost: bool,
    /// World-space event point in mm.
    pub point: Vec3,
    /// Cumulative optical path length at the event, in mm.
    pub opl_mm: f64,
    pub lambda_um: f64,
    pub kind: TraceEventKind,
}

/// An area detector's accumulated readout (§6.9, §9.7).
///
/// Flux density is `incident / area`, and the area is a real `mm²` — the thing a
/// line probe never had, and the reason [`DetectorGeometry3::LineProbeLegacy`](oc_core::scene3::DetectorGeometry3)
/// reports no density (§3.2).
#[derive(Debug, Clone)]
pub struct DetectorResult3 {
    pub detector: DetectorId,
    pub bins: [usize; 2],
    /// Area of one whole pixel in `mm²`, or `None` for a line probe with no
    /// area. See [`Detector3::pixel_area_mm2`].
    pub pixel_area_mm2: Option<f64>,
    /// The part of each pixel on the face, `mm²`, row-major over `bins`, and
    /// empty for a line probe. What irradiance divides by. See
    /// [`Detector3::covered_cell_areas_mm2`].
    pub covered_area_mm2: Vec<f64>,
    /// The whole face's sensitive area, `mm²`, or `None` for a line probe.
    pub face_area_mm2: Option<f64>,
    /// Incident optical flux per cell, row-major over `bins`.
    pub incident_flux: Vec<f64>,
    /// Detected signal per cell, after quantum efficiency.
    pub detected_signal: Vec<f64>,
    pub hits: u64,
    pub total_incident: f64,
    pub total_signal: f64,
    /// Flux-weighted sum of `(u, v)`; divide by `total_incident` for the centroid.
    pub flux_uv: Vec2,
    /// Flux-weighted sum of optical path length.
    pub flux_opl: f64,
    /// Bounded per-hit records, if requested.
    pub records: Vec<DetectorHit3>,
    /// Every hit's state, summed in **the detector's own frame**: each ray's
    /// state is rotated onto the projection of the detector's local `+X` axis
    /// before being added.
    ///
    /// `s0` equals `total_incident` and is always meaningful. The other three
    /// are only as meaningful as that common frame is, which is why
    /// [`DetectorResult3::max_tilt_rad`] sits next to them: rays arriving from
    /// very different directions have genuinely different transverse planes,
    /// and adding their `s1` is then an operation on numbers rather than on
    /// light. Read the sum when the cone is narrow; read an analyzer response
    /// when it is not.
    pub stokes_local: oc_core::polarization::Stokes,
    /// The largest angle between an arriving ray and the detector's normal,
    /// radians. The caveat on `stokes_local`, as a number.
    pub max_tilt_rad: f64,

    /// What this detector was asked to measure, copied from the scene so a
    /// reader of the result knows which of the fields below hold anything.
    pub measure: DetectorMeasure,
    /// Flux per **direction** cell, row-major over `bins` as
    /// `sector * rings + ring`. Filled only under
    /// [`DetectorMeasure::Intensity`], because an angular histogram nobody
    /// asked for is a second grid per detector paid for in every trace.
    pub direction_flux: Vec<f64>,
    /// Each of those cells' solid angle in steradians, alongside it.
    pub direction_solid_angle_sr: Vec<f64>,
    /// Flux that arrived past the intensity cone and so landed in no direction
    /// cell. Named rather than dropped: `direction_flux` summing to less than
    /// `total_incident` is then a fact to read instead of a discrepancy to
    /// hunt.
    pub outside_cone_flux: f64,

    /// Each cell's colour, `Σ Φ · (x̄, ȳ, z̄)(λ)` under the CIE 1931
    /// observer, row-major over `bins`. Filled only under
    /// [`DetectorMeasure::Illuminance`], for the same reason as
    /// `direction_flux`.
    pub colour_xyz: Vec<[f64; 3]>,

    /// The hits split by wavelength, one entry per wavelength bin
    /// ([`spectral_bin`]) in order of wavelength: what the chromatic
    /// diagnostics are read from. Sums rather than records, so every hit
    /// counts however many there are, and a pool adds them up like the rest
    /// ([`SpectralSums3`]).
    pub spectral: Vec<SpectralSums3>,
    /// Flux in bins past the first [`MAX_SPECTRAL_LINES`]: counted in every
    /// total, and broken out by wavelength nowhere.
    pub spectral_untracked_flux: f64,
}

/// How many wavelength bins a detector breaks its hits out by: enough for the
/// whole of [`spectral_bin`]'s grid from 0.2 to 4 µm.
pub const MAX_SPECTRAL_LINES: usize = 256;

/// The bin a wavelength's hits are summed in, named by its centre, µm: 5 nm
/// wide below 1 µm, 20 nm to 2.5 µm, 100 nm beyond. A continuous source gives
/// every ray its own wavelength, and a sum per wavelength would be a sum per
/// ray; bins this narrow still resolve a glass's dispersion, which changes by
/// a few parts in 10⁵ across 5 nm in the visible. Each bin also keeps its
/// flux-weighted wavelength, so a source of a few lines reads each line's own
/// wavelength back exactly.
pub fn spectral_bin(lambda_um: f64) -> f64 {
    let nm = lambda_um * 1000.0;
    let width = if nm < 1000.0 {
        5.0
    } else if nm < 2500.0 {
        20.0
    } else {
        100.0
    };
    ((nm / width).floor() + 0.5) * width / 1000.0
}

/// One wavelength's hits on a detector, as flux-weighted sums.
///
/// Positions are the face's local `(u, v)` in mm. Slopes are the arriving
/// direction's `(x, y) / |z|` in the detector's frame: the tangent of the
/// angle to the normal in each plane, and the rate at which a ray's `(u, v)`
/// moves per mm travelled along the normal. From these come a line's
/// centroid, its spot size, its mean direction and spread, and the plane
/// along the normal where its spot is smallest, all exactly, from one pass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpectralSums3 {
    /// The bin's centre, µm ([`spectral_bin`]).
    pub lambda_um: f64,
    /// `Σ w·λ`: over `flux`, the bin's mean wavelength.
    pub lw: f64,
    pub hits: u64,
    /// `Σw`, and the weighted sums of `u`, `v`, `u²`, `v²`.
    pub flux: f64,
    pub u: f64,
    pub v: f64,
    pub uu: f64,
    pub vv: f64,
    /// Weighted sums of the slopes `a = x/|z|`, `b = y/|z|`, their squares,
    /// and the cross terms `u·a` and `v·b` the focus needs.
    pub a: f64,
    pub b: f64,
    pub aa: f64,
    pub bb: f64,
    pub ua: f64,
    pub vb: f64,
}

/// The fields of a [`SpectralSums3`] as f64s, in [`SpectralSums3::to_array`]'s order.
pub const SPECTRAL_FIELDS: usize = 14;

impl SpectralSums3 {
    /// The flux-weighted wavelength of the hits in the bin, µm.
    pub fn mean_lambda_um(&self) -> f64 {
        if self.flux > 0.0 {
            self.lw / self.flux
        } else {
            self.lambda_um
        }
    }

    fn add(&mut self, flux: f64, lambda_um: f64, uv: Vec2, a: f64, b: f64) {
        self.lw += flux * lambda_um;
        self.hits += 1;
        self.flux += flux;
        self.u += flux * uv.x;
        self.v += flux * uv.y;
        self.uu += flux * uv.x * uv.x;
        self.vv += flux * uv.y * uv.y;
        self.a += flux * a;
        self.b += flux * b;
        self.aa += flux * a * a;
        self.bb += flux * b * b;
        self.ua += flux * uv.x * a;
        self.vb += flux * uv.y * b;
    }

    fn plus(&mut self, o: &SpectralSums3) {
        self.lw += o.lw;
        self.hits += o.hits;
        self.flux += o.flux;
        self.u += o.u;
        self.v += o.v;
        self.uu += o.uu;
        self.vv += o.vv;
        self.a += o.a;
        self.b += o.b;
        self.aa += o.aa;
        self.bb += o.bb;
        self.ua += o.ua;
        self.vb += o.vb;
    }

    pub fn to_array(&self) -> [f64; SPECTRAL_FIELDS] {
        [
            self.lambda_um,
            self.lw,
            self.hits as f64,
            self.flux,
            self.u,
            self.v,
            self.uu,
            self.vv,
            self.a,
            self.b,
            self.aa,
            self.bb,
            self.ua,
            self.vb,
        ]
    }

    pub fn from_array(x: &[f64; SPECTRAL_FIELDS]) -> Self {
        Self {
            lambda_um: x[0],
            lw: x[1],
            hits: x[2] as u64,
            flux: x[3],
            u: x[4],
            v: x[5],
            uu: x[6],
            vv: x[7],
            a: x[8],
            b: x[9],
            aa: x[10],
            bb: x[11],
            ua: x[12],
            vb: x[13],
        }
    }

    /// The flux-weighted centroid on the face, mm.
    pub fn centroid(&self) -> Option<Vec2> {
        (self.flux > 0.0).then(|| Vec2::new(self.u / self.flux, self.v / self.flux))
    }

    /// The flux-weighted mean slope `(x/|z|, y/|z|)`.
    pub fn mean_slope(&self) -> Option<Vec2> {
        (self.flux > 0.0).then(|| Vec2::new(self.a / self.flux, self.b / self.flux))
    }

    /// The variances of `u`, `v`, `a`, `b` and the covariances of `u` with `a`
    /// and `v` with `b`, each floored at zero where rounding makes a variance
    /// of a tight spot come out a hair negative.
    fn moments(&self) -> Option<[f64; 6]> {
        if self.flux <= 0.0 {
            return None;
        }
        let w = self.flux;
        let (mu, mv, ma, mb) = (self.u / w, self.v / w, self.a / w, self.b / w);
        Some([
            (self.uu / w - mu * mu).max(0.0),
            (self.vv / w - mv * mv).max(0.0),
            (self.aa / w - ma * ma).max(0.0),
            (self.bb / w - mb * mb).max(0.0),
            self.ua / w - mu * ma,
            self.vb / w - mv * mb,
        ])
    }

    /// The RMS radius of the spot about its centroid, mm.
    pub fn rms_radius(&self) -> Option<f64> {
        self.moments().map(|m| (m[0] + m[1]).sqrt())
    }

    /// The RMS spread of the slopes about their mean: for small angles, the
    /// RMS angular spread in radians. What is left of a collimated beam's
    /// divergence at this line.
    pub fn rms_slope(&self) -> Option<f64> {
        self.moments().map(|m| (m[2] + m[3]).sqrt())
    }

    /// How far along the normal, mm, the spot of this line is smallest:
    /// positive in the direction the light travels, so a beam still
    /// converging on the face focuses past it. Each ray's position there is
    /// `u + s·a`, and the spot's variance in `s` is a parabola whose least
    /// point is `-(cov(u,a) + cov(v,b)) / (var a + var b)`. `None` for a
    /// beam whose rays are all parallel, which focuses nowhere.
    pub fn best_focus_mm(&self) -> Option<f64> {
        let m = self.moments()?;
        let spread = m[2] + m[3];
        (spread > 1e-18).then(|| -(m[4] + m[5]) / spread)
    }
}

impl DetectorResult3 {
    /// An empty readout sized and configured by the detector it belongs to.
    ///
    /// This computes every rim pixel's covered area exactly, so a trace builds
    /// one per detector and clones it into each worker rather than calling
    /// this per worker.
    pub fn new(d: &Detector3) -> Self {
        let bins = d.bins;
        let cells = (bins[0].max(1)) * (bins[1].max(1));
        let direction_solid_angle_sr = d.direction_cell_solid_angles().unwrap_or_default();
        let direction_flux = vec![0.0; direction_solid_angle_sr.len()];
        Self {
            detector: d.id(),
            bins,
            pixel_area_mm2: d.pixel_area_mm2(),
            covered_area_mm2: d.covered_cell_areas_mm2().unwrap_or_default(),
            face_area_mm2: d.face_area_mm2(),
            measure: d.measure,
            direction_flux,
            direction_solid_angle_sr,
            outside_cone_flux: 0.0,
            colour_xyz: if matches!(d.measure, DetectorMeasure::Illuminance) {
                vec![[0.0; 3]; cells]
            } else {
                Vec::new()
            },
            spectral: Vec::new(),
            spectral_untracked_flux: 0.0,
            incident_flux: vec![0.0; cells],
            detected_signal: vec![0.0; cells],
            stokes_local: oc_core::polarization::Stokes::ZERO,
            max_tilt_rad: 0.0,
            hits: 0,
            total_incident: 0.0,
            total_signal: 0.0,
            flux_uv: Vec2::ZERO,
            flux_opl: 0.0,
            records: Vec::new(),
        }
    }

    /// Flux-weighted centroid in local `(u, v)` mm, or `None` with no flux.
    pub fn centroid_uv(&self) -> Option<Vec2> {
        (self.total_incident > 0.0).then(|| self.flux_uv / self.total_incident)
    }

    /// Flux-weighted mean optical path length, or `None` with no flux.
    pub fn mean_opl_mm(&self) -> Option<f64> {
        (self.total_incident > 0.0).then(|| self.flux_opl / self.total_incident)
    }

    /// Relative flux density per pixel, or `None` for a line probe: the flux a
    /// pixel collected over the part of it on the face.
    ///
    /// A pixel wholly off the face (a corner of a disc's box, an annulus's
    /// hole) collected nothing and reads 0. So `Σ reading × covered area` is
    /// the flux the grid holds, whatever the shape (A12).
    pub fn flux_density(&self) -> Option<Vec<f64>> {
        self.pixel_area_mm2?;
        Some(
            self.incident_flux
                .iter()
                .zip(&self.covered_area_mm2)
                .map(|(f, a)| if *a > 0.0 { f / a } else { 0.0 })
                .collect(),
        )
    }

    /// Irradiance per pixel, flux per `mm²`, or `None` for a line probe.
    ///
    /// The same numbers as [`Self::flux_density`], under the name the unit
    /// belongs to.
    pub fn irradiance(&self) -> Option<Vec<f64>> {
        self.flux_density()
    }

    /// The flux collected over the whole face's area, flux per `mm²`, or
    /// `None` for a line probe or an empty face.
    ///
    /// It does not depend on the pixel count, which no pixel's reading can
    /// say. It equals the irradiance only where the light covers the face
    /// evenly; a small spot on a large face reads low here, and high in its
    /// pixels.
    pub fn mean_irradiance(&self) -> Option<f64> {
        let a = self.face_area_mm2?;
        (a > 0.0).then(|| self.total_incident / a)
    }

    /// Each cell's colour as a light meter scales it: `683 lm/W · (X, Y, Z)`
    /// per `m²` of the cell on the face, so `Y` is the illuminance (lux when
    /// the flux is watts). `None` unless the detector measures illuminance.
    pub fn colour_per_m2(&self) -> Option<Vec<[f64; 3]>> {
        (!self.colour_xyz.is_empty()).then(|| {
            self.colour_xyz
                .iter()
                .zip(&self.covered_area_mm2)
                .map(|(c, a)| {
                    let k = if *a > 0.0 { oc_core::colorimetry::LM_PER_W / (a * 1e-6) } else { 0.0 };
                    [c[0] * k, c[1] * k, c[2] * k]
                })
                .collect()
        })
    }

    /// Radiant intensity per direction cell, flux per steradian, or `None`
    /// unless the detector was asked to measure it.
    pub fn radiant_intensity(&self) -> Option<Vec<f64>> {
        (!self.direction_flux.is_empty()).then(|| {
            self.direction_flux
                .iter()
                .zip(&self.direction_solid_angle_sr)
                .map(|(f, w)| if *w > 0.0 { f / w } else { 0.0 })
                .collect()
        })
    }

    /// The cells in whatever this detector measures, and the unit they are in,
    /// for a reader that would rather not branch.
    pub fn readings(&self) -> (Vec<f64>, &'static str) {
        let cells = match self.measure {
            DetectorMeasure::Power => Some(self.incident_flux.clone()),
            DetectorMeasure::Irradiance => self.irradiance(),
            DetectorMeasure::Intensity { .. } => self.radiant_intensity(),
            DetectorMeasure::Illuminance => self.colour_per_m2().map(|c| c.iter().map(|x| x[1]).collect()),
        };
        (cells.unwrap_or_default(), self.measure.unit())
    }

    /// The largest reading, in the detector's own measure. `None` when there
    /// is nothing to read.
    ///
    /// For irradiance the peak skips rim pixels with less than half the
    /// covered area of the fullest pixel. Such a pixel's reading is right on
    /// average, but it rests on few rays, and dividing by a sliver of area
    /// makes one ray there a spike the maximum would pick out: a pixel 1%
    /// on the face reads a single ray a hundred times higher than a whole
    /// pixel would. The readings themselves are left as they are.
    pub fn peak_reading(&self) -> Option<f64> {
        let floor = match self.measure {
            DetectorMeasure::Irradiance | DetectorMeasure::Illuminance => {
                Some(0.5 * self.covered_area_mm2.iter().copied().fold(0.0, f64::max))
            }
            DetectorMeasure::Power | DetectorMeasure::Intensity { .. } => None,
        };
        self.readings()
            .0
            .into_iter()
            .enumerate()
            .filter(|(i, _)| match floor {
                None => true,
                Some(f) => self.covered_area_mm2.get(*i).is_some_and(|a| *a >= f),
            })
            .fold(None, |m: Option<f64>, (_, v)| Some(m.map_or(v, |m| m.max(v))))
    }

    /// Add one hit to its wavelength's sums. `dir_local` is the direction it
    /// arrived in, in the detector's frame.
    pub(crate) fn add_spectral(&mut self, lambda_um: f64, flux: f64, uv: Vec2, dir_local: Vec3) {
        let z = dir_local.z.abs().max(1e-12);
        let (a, b) = (dir_local.x / z, dir_local.y / z);
        match self.spectral_line(spectral_bin(lambda_um)) {
            Some(i) => self.spectral[i].add(flux, lambda_um, uv, a, b),
            None => self.spectral_untracked_flux += flux,
        }
    }

    /// The index of `lambda_um`'s sums, made if there is room for it. The list
    /// stays in order of wavelength, so two traces that met the same lines in
    /// a different order still lay them out, and add them up, the same way.
    fn spectral_line(&mut self, lambda_um: f64) -> Option<usize> {
        match self
            .spectral
            .binary_search_by(|s| s.lambda_um.total_cmp(&lambda_um))
        {
            Ok(i) => Some(i),
            Err(i) if self.spectral.len() < MAX_SPECTRAL_LINES => {
                self.spectral.insert(
                    i,
                    SpectralSums3 {
                        lambda_um,
                        ..Default::default()
                    },
                );
                Some(i)
            }
            Err(_) => None,
        }
    }

    pub(crate) fn merge(&mut self, other: &DetectorResult3) {
        for (a, b) in self.incident_flux.iter_mut().zip(&other.incident_flux) {
            *a += *b;
        }
        for (a, b) in self.detected_signal.iter_mut().zip(&other.detected_signal) {
            *a += *b;
        }
        self.hits += other.hits;
        self.total_incident += other.total_incident;
        self.total_signal += other.total_signal;
        self.flux_uv += other.flux_uv;
        self.flux_opl += other.flux_opl;
        self.records.extend_from_slice(&other.records);
        for (a, b) in self
            .direction_flux
            .iter_mut()
            .zip(&other.direction_flux)
        {
            *a += *b;
        }
        self.outside_cone_flux += other.outside_cone_flux;
        for (a, b) in self.colour_xyz.iter_mut().zip(&other.colour_xyz) {
            for k in 0..3 {
                a[k] += b[k];
            }
        }
        for line in &other.spectral {
            match self.spectral_line(line.lambda_um) {
                Some(i) => self.spectral[i].plus(line),
                None => self.spectral_untracked_flux += line.flux,
            }
        }
        self.spectral_untracked_flux += other.spectral_untracked_flux;
        // Both sides already rotated into the same detector frame, so this is a
        // sum of comparable numbers rather than of different bases.
        self.stokes_local = self.stokes_local.plus(&other.stokes_local);
        self.max_tilt_rad = self.max_tilt_rad.max(other.max_tilt_rad);
    }
}

/// Where a branch's flux ended up (§9.8). For an uncapped deterministic scene,
/// `launched` balances the rest to `1e-10` relative on analytic fixtures.
///
/// The invariant is not "energy is conserved" — a scene absorbs, a detector
/// swallows, an aperture blocks. It is that *no flux leaves the accounting*: what
/// a threshold or a cap removes is named in `truncated_flux`, so a ledger that
/// balances is a real statement and not an artefact of quietly dropped light.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EnergyLedger {
    pub launched: f64,
    pub detected_incident: f64,
    pub detected_signal: f64,
    pub escaped: f64,
    pub absorbed_volume: f64,
    pub absorbed_surface: f64,
    pub blocked_aperture: f64,
    pub truncated_flux: f64,
    pub numerical_loss: f64,
}

impl EnergyLedger {
    /// Everything the ledger accounts for besides what was launched. A
    /// terminating detector's incident flux is a sink like any other.
    pub fn accounted(&self) -> f64 {
        self.detected_incident
            + self.escaped
            + self.absorbed_volume
            + self.absorbed_surface
            + self.blocked_aperture
            + self.truncated_flux
            + self.numerical_loss
    }

    /// Relative imbalance `|launched − accounted| / launched`, or the absolute
    /// residual when nothing was launched.
    pub fn residual(&self) -> f64 {
        let r = (self.launched - self.accounted()).abs();
        if self.launched > 0.0 {
            r / self.launched
        } else {
            r
        }
    }

    pub(crate) fn merge(&mut self, o: &EnergyLedger) {
        self.launched += o.launched;
        self.detected_incident += o.detected_incident;
        self.detected_signal += o.detected_signal;
        self.escaped += o.escaped;
        self.absorbed_volume += o.absorbed_volume;
        self.absorbed_surface += o.absorbed_surface;
        self.blocked_aperture += o.blocked_aperture;
        self.truncated_flux += o.truncated_flux;
        self.numerical_loss += o.numerical_loss;
    }
}

/// Why a branch stopped (§9.8). Every branch ends with exactly one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerminationReason {
    Detector,
    AbsorbedVolume,
    AbsorbedSurface,
    ApertureBlocked,
    EscapedBounds,
    FluxThreshold,
    MaxDepth,
    MaxGhostDepth,
    MaxBranches,
    NumericalFailure,
    UnsupportedPhysics,
}

/// A count of branch terminations by reason.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerminationSummary {
    pub detector: u64,
    pub absorbed_volume: u64,
    pub absorbed_surface: u64,
    pub aperture_blocked: u64,
    pub escaped_bounds: u64,
    pub flux_threshold: u64,
    pub max_depth: u64,
    pub max_ghost_depth: u64,
    pub max_branches: u64,
    pub numerical_failure: u64,
    pub unsupported_physics: u64,
}

impl TerminationSummary {
    pub(crate) fn record(&mut self, reason: TerminationReason) {
        use TerminationReason::*;
        match reason {
            Detector => self.detector += 1,
            AbsorbedVolume => self.absorbed_volume += 1,
            AbsorbedSurface => self.absorbed_surface += 1,
            ApertureBlocked => self.aperture_blocked += 1,
            EscapedBounds => self.escaped_bounds += 1,
            FluxThreshold => self.flux_threshold += 1,
            MaxDepth => self.max_depth += 1,
            MaxGhostDepth => self.max_ghost_depth += 1,
            MaxBranches => self.max_branches += 1,
            NumericalFailure => self.numerical_failure += 1,
            UnsupportedPhysics => self.unsupported_physics += 1,
        }
    }

    pub(crate) fn merge(&mut self, o: &TerminationSummary) {
        self.detector += o.detector;
        self.absorbed_volume += o.absorbed_volume;
        self.absorbed_surface += o.absorbed_surface;
        self.aperture_blocked += o.aperture_blocked;
        self.escaped_bounds += o.escaped_bounds;
        self.flux_threshold += o.flux_threshold;
        self.max_depth += o.max_depth;
        self.max_ghost_depth += o.max_ghost_depth;
        self.max_branches += o.max_branches;
        self.numerical_failure += o.numerical_failure;
        self.unsupported_physics += o.unsupported_physics;
    }
}

/// A physics feature a backend can run (§3.5). Coarse on purpose: the report
/// exists to say what was and was not handled, not to enumerate every constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PhysicsFeature {
    AnalyticSolid,
    ProfileBody,
    ZeroThicknessSurface,
    Aperture,
    AreaDetector,
    DielectricFresnel,
    Mirror,
    ExplicitRt,
    ThinFilmCoating,
    ThinLens,
    Grating,
    Polarizer,
    Retarder,
    VolumeAbsorption,
    SpectralResponse,
}

/// A feature that ran, but not exactly (§3.5). None are emitted yet — the
/// reference path is exact or it refuses — but the field exists so a later
/// approximate backend has somewhere honest to record itself.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Approximation {
    pub feature: PhysicsFeature,
    pub detail: String,
}

/// A scene, or part of one, a backend cannot handle exactly (§17).
///
/// The tracer emits these instead of substituting a superficially similar
/// geometry or surface. A backend that cannot trace a required feature rejects
/// the scene; it never traces it wrong and stays quiet.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CapabilityIssue {
    /// A profile body whose third dimension is unresolved (§3.2).
    PlanarOnlyObject {
        object: ObjectId,
    },
    UnsupportedGeometry {
        object: ObjectId,
        kind: String,
    },
    UnsupportedSurfaceResponse {
        surface: SurfaceRef,
    },
    UnsupportedSourceDistribution {
        source: SourceId,
    },
    InvalidClosedBody {
        body: BodyId,
        reason: String,
    },
    /// Coincident physical interfaces the event policy cannot order (§9.3).
    AmbiguousEvent {
        point: Vec3,
        objects: Vec<ObjectId>,
    },
    /// A spectral or absorption table asked for a wavelength it does not cover,
    /// under the error policy (§6.7).
    WavelengthOutOfRange {
        object: ObjectId,
        lambda_um: f64,
    },
    NonFiniteInput {
        object: Option<ObjectId>,
        field: String,
    },
    NumericalNonConvergence {
        object: ObjectId,
        patch: PatchId,
    },
    BlockedByStop {
        aperture: ApertureId,
    },
}

/// A backend's structured statement of what it can and cannot do for a scene
/// (§3.5). For the reference tracer, `rejected` is the load-bearing field: a
/// non-empty `rejected` means some flux could not be traced honestly.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapabilityReport {
    pub supported: Vec<PhysicsFeature>,
    pub exact: Vec<PhysicsFeature>,
    pub approximated: Vec<Approximation>,
    pub rejected: Vec<CapabilityIssue>,
    /// The profile the trace ran under. Set when a trace finishes; a report
    /// assembled by hand may leave it out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fidelity: Option<oc_core::fidelity::FidelityStamp>,
}

impl CapabilityReport {
    /// Did every required feature trace exactly?
    pub fn is_clean(&self) -> bool {
        self.rejected.is_empty() && self.approximated.is_empty()
    }

    pub(crate) fn note_exact(&mut self, f: PhysicsFeature) {
        if !self.exact.contains(&f) {
            self.exact.push(f);
        }
        if !self.supported.contains(&f) {
            self.supported.push(f);
        }
    }

    /// Record that a feature ran under a stated model assumption rather than
    /// exactly. Deduplicated on the pair, so a scene with a thousand mirrors
    /// says it once.
    pub(crate) fn note_approx(&mut self, f: PhysicsFeature, detail: &str) {
        if !self
            .approximated
            .iter()
            .any(|a| a.feature == f && a.detail == detail)
        {
            self.approximated.push(Approximation {
                feature: f,
                detail: detail.to_string(),
            });
        }
        if !self.supported.contains(&f) {
            self.supported.push(f);
        }
    }

    pub(crate) fn reject(&mut self, issue: CapabilityIssue) {
        if !self.rejected.contains(&issue) {
            self.rejected.push(issue);
        }
    }

    pub(crate) fn merge(&mut self, o: &CapabilityReport) {
        for f in &o.exact {
            self.note_exact(*f);
        }
        for f in &o.supported {
            if !self.supported.contains(f) {
                self.supported.push(*f);
            }
        }
        for a in &o.approximated {
            if !self.approximated.contains(a) {
                self.approximated.push(a.clone());
            }
        }
        for i in &o.rejected {
            self.reject(i.clone());
        }
    }
}

/// The result of a full 3D trace (§9.9).
#[derive(Debug, Clone)]
pub struct TraceResult3 {
    /// The profile these numbers were computed under (coating-fidelity spec
    /// §5.3). Results of two profiles are never merged without saying so.
    pub fidelity: oc_core::fidelity::FidelityStamp,
    /// The scene revision that produced these numbers, so a stale result can be
    /// discarded rather than shown (§3.3). The tracer copies it from the config;
    /// the caller that owns the scene sets it.
    pub revision: u64,
    pub segments: Vec<Segment3>,
    /// Segments not recorded for being dimmer than the configured floor
    /// ([`crate::TraceConfig3::segment_floor`]). Their flux is in the ledger and
    /// the detectors all the same.
    pub segments_omitted: u64,
    pub detectors: Vec<DetectorResult3>,
    pub ledger: EnergyLedger,
    pub terminations: TerminationSummary,
    pub capability_report: CapabilityReport,
    /// The event log, empty unless `TraceConfig3::record_events` was set.
    pub events: Vec<TraceEvent3>,
    /// The ledger object by object and band, empty unless
    /// `TraceConfig3::object_ledger` was set. A pooled trace's encoded parts do
    /// not carry it.
    pub objects: Vec<ObjectEntry3>,
    /// Where the light was absorbed in the bodies, and blocked on the stops,
    /// the config asks for; empty by default. A pooled trace's encoded parts
    /// do not carry it.
    pub deposits: Vec<Deposit3>,
}
