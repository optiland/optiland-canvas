//! Scene3 authoring — build and edit the canonical 3D scene from high-level
//! optical primitives.
//!
//! The migration layer ([`crate::migrate_v1`]) deliberately makes *planar-only*
//! bodies: a v1 document never said whether its outline was revolved or extruded,
//! so it refuses to guess. Authoring is the opposite situation — the tool *does*
//! know it is placing a lens — so these constructors build genuine 3D bodies
//! (`RevolveLocalX` lenses, disc/rectangle sources, area detectors) that trace in
//! full 3D, and the edit setters are what the inspector cards drive.
//!
//! Every constructor mints ids through [`Scene3::alloc_id`] (never a vector
//! index, never reused) and produces geometry that satisfies [`Scene3::validate`]
//! and [`validate_revolvable`]. The optical axis is local **+X** (§4.1); a lens
//! is a meridional profile mirror-symmetric about `y = 0`, revolved about it.
//!
//! Beam splitters and dichroic filters are zero-thickness plates with an
//! explicit reflectance and transmittance ([`SurfaceResponse::ExplicitRt`]):
//! constant for a splitter, tabulated over wavelength for a dichroic
//! ([`DichroicSpec`]). The tracer splits both into a reflected and a
//! transmitted branch, weighted per wavelength.

use crate::geom3::primitives::even_asphere_sag;
use crate::geom3::profile::validate_revolvable;
use crate::id::{BodyId, DetectorId, LayerId, SourceId};
use crate::material::Material;
use crate::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use crate::scene::Spectrum;
use crate::thinfilm::{
    film_coating, plate_coating, Coating, CoatingDesign, DesignKind, ExitFace, Incidence,
};
use crate::scene3::{
    DetectorMeasure,
    AbsorptionModel, AngularDistribution3, AngularScalar, Aperture3, ApertureInteraction,
    ApertureShape3, Body3,
    Detector3, DetectorGeometry3, DetectorResponse, DetectorTermination, Interpolation,
    ObjectCommon, OutOfRangePolicy, PatchResponseOverride, PlanarShape2, SamplingSequence, Scene3,
    SolidGeometry3, Source3, SourceSampling, SpatialDistribution3, SpectralScalar,
    SurfaceGeometry3, SurfaceObject3, SurfaceProps3, SurfaceResponse, TwoSidedSurfaceResponse,
    Waveplate,
};
use crate::vec::Vec2;
use crate::vec3::{Isometry3, Plane3, Vec3};

/// Which face of a revolved body a coating goes on.
///
/// The profile `singlet_curves` builds runs [front face, rim, back face, rim],
/// with the front at local `-X`, and every transform that places one of these
/// carries local `-X` onto the direction an axial beam arrives from. So "front"
/// is the face the light reaches first, for a lens and for a plate alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyFace {
    /// The face an axial beam reaches first.
    Front,
    /// The one it leaves by.
    Back,
}

impl BodyFace {
    fn edge(self) -> usize {
        match self {
            Self::Front => 0,
            Self::Back => 2,
        }
    }
}

/// The rotationally-symmetric geometry of a singlet lens, in its local frame
/// (optical axis local +X, centred on its own body origin). Material is separate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingletGeom {
    /// Front (−X face) radius of curvature, mm. `+inf` is a plane; sign follows
    /// the optical convention (R > 0 ⇒ centre of curvature toward +X).
    pub r1_mm: f64,
    /// Rear (+X face) radius of curvature, mm.
    pub r2_mm: f64,
    /// Axial thickness, mm.
    pub thickness_mm: f64,
    /// Semi-aperture (radial half-height), mm.
    pub semi_aperture_mm: f64,
    /// Front conic constant.
    pub conic1: f64,
    /// Rear conic constant.
    pub conic2: f64,
}

impl SingletGeom {
    /// A symmetric biconvex N-BK7-ish singlet, a sensible authoring default.
    pub fn biconvex() -> Self {
        Self {
            r1_mm: 50.0,
            r2_mm: -50.0,
            thickness_mm: 4.0,
            semi_aperture_mm: 12.0,
            conic1: 0.0,
            conic2: 0.0,
        }
    }
}

/// The even-asphere terms of a singlet's two faces: `sag += Σ coeffs[i]·r^(2i+4)`
/// on top of each face's conic, so `coeffs[0]` is A4 (mm⁻³), `coeffs[1]` A6,
/// and so on to A18. A face whose terms are all zero is stored as a plain conic,
/// which keeps a conic singlet's document and its planar reduction unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SingletAspheres {
    pub front: [f64; 8],
    pub back: [f64; 8],
}

/// The sag of one singlet face at height `h_mm`, measured from that face's
/// vertex along +X: the conic plus its aspheric terms, the same function the
/// geometry traces.
pub fn singlet_face_sag(g: &SingletGeom, a: &SingletAspheres, back: bool, h_mm: f64) -> f64 {
    if back {
        even_asphere_sag(curvature(g.r2_mm), g.conic2, &a.back, h_mm)
    } else {
        even_asphere_sag(curvature(g.r1_mm), g.conic1, &a.front, h_mm)
    }
}

/// A singlet's paraxial first-order properties at the d line, in the ambient
/// medium: effective focal length, and the back focal distance measured from
/// the rear vertex along the lens axis. Thick-lens formulas; the aspheric terms
/// have no paraxial effect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingletParaxial {
    pub nd: f64,
    pub efl_mm: f64,
    pub bfl_mm: f64,
}

/// A dichroic coating: wavelengths on one side of `edge_nm` reflect, those on
/// the other side transmit. Long-pass transmits above the edge.
///
/// The curves are a raised-cosine transition `width_nm` wide, sampled at
/// [`DICHROIC_SAMPLES`] wavelengths and held flat beyond them
/// ([`OutOfRangePolicy::Clamp`]): reflectance falls from `r_max` to 0 while
/// transmittance rises from 0 to `t_max` (the other way round for short-pass).
/// Both tables share one grid, so `R + T <= max(r_max, t_max)` at every
/// wavelength and the remainder is absorbed. A scalar model: no angle of
/// incidence and no polarization, like every response in the kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DichroicSpec {
    pub edge_nm: f64,
    pub width_nm: f64,
    pub long_pass: bool,
    pub r_max: f64,
    pub t_max: f64,
}

/// Samples in a dichroic's tables.
pub const DICHROIC_SAMPLES: usize = 9;

impl Default for DichroicSpec {
    /// A long-pass edge at 550 nm: blue and green reflect, yellow and red pass.
    fn default() -> Self {
        Self {
            edge_nm: 550.0,
            width_nm: 20.0,
            long_pass: true,
            r_max: 0.98,
            t_max: 0.95,
        }
    }
}

impl DichroicSpec {
    /// The spec with its values brought into range (a positive edge, a
    /// transition at least 0.5 nm wide that stays at positive wavelengths,
    /// fractions in `[0, 1]`), or `None` if a value is not finite.
    pub fn sanitized(self) -> Option<Self> {
        let finite = [self.edge_nm, self.width_nm, self.r_max, self.t_max]
            .iter()
            .all(|v| v.is_finite());
        if !finite {
            return None;
        }
        let edge_nm = self.edge_nm.max(1.0);
        Some(Self {
            edge_nm,
            width_nm: self.width_nm.clamp(0.5, 2.0 * edge_nm - 1.0),
            long_pass: self.long_pass,
            r_max: self.r_max.clamp(0.0, 1.0),
            t_max: self.t_max.clamp(0.0, 1.0),
        })
    }

    /// The shared wavelength grid (µm) and the reflectance and transmittance
    /// tables on it.
    fn tables(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = DICHROIC_SAMPLES;
        let lo = self.edge_nm - self.width_nm / 2.0;
        let mut lambda = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f64 / (n - 1) as f64;
            let rise = 0.5 - 0.5 * (std::f64::consts::PI * f).cos();
            let pass = if self.long_pass { rise } else { 1.0 - rise };
            lambda.push((lo + self.width_nm * f) / 1000.0);
            r.push(self.r_max * (1.0 - pass));
            t.push(self.t_max * pass);
        }
        (lambda, r, t)
    }

    /// The surface response this spec stands for.
    pub fn response(&self) -> SurfaceResponse {
        let (lambda_um, r, t) = self.tables();
        let table = |values: Vec<f64>| SpectralScalar::Tabulated {
            lambda_um: lambda_um.clone(),
            values,
            interpolation: Interpolation::Linear,
            out_of_range: OutOfRangePolicy::Clamp,
        };
        SurfaceResponse::ExplicitRt {
            reflectance: table(r),
            transmittance: table(t),
        }
    }

    /// The spec a response was made from, if [`DichroicSpec::response`] made
    /// it. Any other table, however similar, is `None`: its parameters would
    /// be a guess.
    pub fn from_response(r: &SurfaceResponse) -> Option<Self> {
        let SurfaceResponse::ExplicitRt {
            reflectance:
                SpectralScalar::Tabulated {
                    lambda_um: lr,
                    values: rv,
                    out_of_range: OutOfRangePolicy::Clamp,
                    ..
                },
            transmittance:
                SpectralScalar::Tabulated {
                    lambda_um: lt,
                    values: tv,
                    out_of_range: OutOfRangePolicy::Clamp,
                    ..
                },
        } = r
        else {
            return None;
        };
        let n = DICHROIC_SAMPLES;
        if lr.len() != n || lr != lt || rv.len() != n || tv.len() != n {
            return None;
        }
        let (lo, hi) = (lr[0] * 1000.0, lr[n - 1] * 1000.0);
        let max = |v: &[f64]| v.iter().copied().fold(0.0_f64, f64::max);
        let (r_max, t_max) = (max(rv), max(tv));
        let long_pass = if t_max > 0.0 {
            tv[n - 1] > tv[0]
        } else {
            rv[0] >= rv[n - 1]
        };
        let spec = Self {
            edge_nm: (lo + hi) / 2.0,
            width_nm: hi - lo,
            long_pass,
            r_max,
            t_max,
        };
        let (l2, r2, t2) = spec.tables();
        let close = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-9 * (1.0 + x.abs()))
        };
        (close(lr, &l2) && close(rv, &r2) && close(tv, &t2)).then_some(spec)
    }
}

/// A constant reflectance/transmittance pair, if it is a valid one.
fn split_response(reflectance: f64, transmittance: f64) -> Option<SurfaceResponse> {
    let ok = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
    if !ok(reflectance) || !ok(transmittance) || reflectance + transmittance > 1.0 + 1e-12 {
        return None;
    }
    Some(SurfaceResponse::ExplicitRt {
        reflectance: SpectralScalar::Constant(reflectance),
        transmittance: SpectralScalar::Constant(transmittance),
    })
}

/// A white beam: nine equal-weight lines across the visible band, so a full
/// spectrum sums to white and chromatic dispersion shows as coloured fringes
/// (the v1 default beam's behaviour). Wavelengths in µm.
fn white_spectrum() -> Spectrum {
    let lines = (0..9)
        .map(|i| (0.42 + (0.66 - 0.42) * (i as f64) / 8.0, 1.0))
        .collect();
    Spectrum::Lines(lines)
}

/// Curvature `c = 1/R`; an infinite (or effectively infinite) radius is a plane.
fn curvature(r_mm: f64) -> f64 {
    if r_mm.is_finite() && r_mm.abs() > 1e-12 {
        1.0 / r_mm
    } else {
        0.0
    }
}

/// The four meridional edges of a singlet, in closed-loop order: front face,
/// top rim line, back face, bottom rim line. Mirror-symmetric about `y = 0`.
///
/// The rim X of each face is its sag at the semi-aperture, computed with the same
/// [`even_asphere_sag`] the geometry and [`validate_revolvable`] use, so the two
/// rim lines meet the faces exactly and the loop closes.
fn singlet_curves(g: SingletGeom, a: SingletAspheres) -> [Curve2; 4] {
    let ha = g.semi_aperture_mm;
    let (zf, zr) = (-g.thickness_mm / 2.0, g.thickness_mm / 2.0);
    let (c1, c2) = (curvature(g.r1_mm), curvature(g.r2_mm));
    let ftip = zf + even_asphere_sag(c1, g.conic1, &a.front, ha);
    let rtip = zr + even_asphere_sag(c2, g.conic2, &a.back, ha);
    let face = |x: f64, c: f64, k: f64, coeffs: [f64; 8]| {
        let base = ConicParams2 {
            vertex: Vec2::new(x, 0.0),
            axis: Vec2::X,
            c,
            k,
            half_ap: ha,
        };
        if coeffs.iter().all(|v| *v == 0.0) {
            Curve2::Conic(base)
        } else {
            Curve2::EvenAsphere { base, coeffs }
        }
    };
    [
        face(zf, c1, g.conic1, a.front),
        Curve2::Line {
            a: Vec2::new(ftip, ha),
            b: Vec2::new(rtip, ha),
        },
        face(zr, c2, g.conic2, a.back),
        Curve2::Line {
            a: Vec2::new(rtip, -ha),
            b: Vec2::new(ftip, -ha),
        },
    ]
}

/// Read a singlet body's geometry and aspheric terms back out of its profile
/// (for the inspector).
fn shape_from_profile(profile: &ClosedProfile2) -> Option<(SingletGeom, SingletAspheres)> {
    let edges = &profile.edges;
    if edges.len() != 4 {
        return None;
    }
    let face = |c: &Curve2| match c {
        Curve2::Conic(p) => Some((*p, [0.0; 8])),
        Curve2::EvenAsphere { base, coeffs } => Some((*base, *coeffs)),
        _ => None,
    };
    let (front, a1) = face(&edges[0].curve)?;
    let (back, a2) = face(&edges[2].curve)?;
    let radius = |c: f64| {
        if c.abs() > 1e-12 {
            1.0 / c
        } else {
            f64::INFINITY
        }
    };
    Some((
        SingletGeom {
            r1_mm: radius(front.c),
            r2_mm: radius(back.c),
            thickness_mm: back.vertex.x - front.vertex.x,
            semi_aperture_mm: front.half_ap,
            conic1: front.k,
            conic2: back.k,
        },
        SingletAspheres {
            front: a1,
            back: a2,
        },
    ))
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting. `None` when
/// a pivot vanishes against the matrix's largest entry.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let scale = a
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-15 * scale {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let (upper, lower) = a.split_at_mut(col + 1);
        let top = &upper[col];
        for (r, row) in lower.iter_mut().enumerate() {
            let f = row[col] / top[col];
            for (x, t) in row[col..n].iter_mut().zip(&top[col..n]) {
                *x -= f * t;
            }
            b[col + 1 + r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}


/// Both sides of a coated face of a solid.
///
/// The film sits on the outside of the glass, so light from outside meets it
/// first and light from inside meets it last, layers in the other order. The
/// inside copy says so ([`Coating::seen_from_behind`]); a copy that did not
/// would walk the stack the wrong way round for every ray leaving the glass.
fn coated_face(film: SurfaceResponse) -> TwoSidedSurfaceResponse {
    let inside = match &film {
        SurfaceResponse::Coating { coating } => SurfaceResponse::Coating {
            coating: coating.seen_from_behind(),
        },
        other => other.clone(),
    };
    TwoSidedSurfaceResponse {
        from_outside: film,
        from_inside: inside,
    }
}

impl Scene3 {
    /// Add a rotationally-symmetric singlet on the optical axis, its body centred
    /// at `center`, with glass `(nd, vd)`. Returns the new body id.
    pub fn add_singlet(&mut self, center: Vec3, g: SingletGeom, nd: f64, vd: f64) -> BodyId {
        let curves = singlet_curves(g, SingletAspheres::default());
        let edges = curves
            .into_iter()
            .map(|curve| ProfileEdge2 {
                patch: self.alloc_patch_id(),
                curve,
            })
            .collect();
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = Isometry3::from_pos(center);
        self.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Profile(ProfileSolid3 {
                profile: ClosedProfile2 { edges },
                construction: ProfileConstruction3::RevolveLocalX,
            }),
            material: Material::AbbePoint { nd, vd },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        BodyId(id.0)
    }

    /// Add a collimated beam aimed along +X, its aperture a rectangle (spread in
    /// Y, thin in Z for a clean meridional fan). `spatial_samples` rays fill it.
    pub fn add_collimated_beam(
        &mut self,
        pos: Vec3,
        half_width_mm: f64,
        half_height_mm: f64,
        spatial_samples: u32,
        spectrum: Spectrum,
    ) -> SourceId {
        let id = self.alloc_source_id();
        self.sources.push(Source3 {
            id,
            transform: Isometry3::from_pos(pos),
            spatial: SpatialDistribution3::Rectangle {
                half_width_mm,
                half_height_mm,
            },
            angular: AngularDistribution3::Direction { local_dir: Vec3::X },
            spectrum,
            flux: 1.0,
            samples: SourceSampling {
                spatial_samples: spatial_samples.max(1),
                angular_samples: 1,
                seed: 1,
                sequence: SamplingSequence::Grid,
            },
        });
        id
    }

    /// Reconfigure the first source in place — the light-source picker (§6.6).
    /// Keeps the source's position, spectrum, seed, and sequence; replaces its
    /// spatial/angular distribution and per-dimension sample counts. Returns false
    /// if there is no source. The total ray count is `spatial × angular`.
    pub fn set_first_source(
        &mut self,
        spatial: SpatialDistribution3,
        angular: AngularDistribution3,
        spatial_samples: u32,
        angular_samples: u32,
    ) -> bool {
        let Some(src) = self.sources.first_mut() else {
            return false;
        };
        src.spatial = spatial;
        src.angular = angular;
        src.samples.spatial_samples = spatial_samples.max(1);
        src.samples.angular_samples = angular_samples.max(1);
        true
    }

    /// Set the first source's emission spectrum to the given `(wavelength_µm,
    /// weight)` lines — the wavelength picker (a list, or a Fraunhofer preset).
    /// Non-finite or non-positive wavelengths and non-positive weights are
    /// dropped; a line is kept only if its weight is positive. Returns false if
    /// there is no source or no usable line survived (spectrum left unchanged).
    pub fn set_first_source_spectrum(&mut self, lines: &[(f64, f64)]) -> bool {
        let clean: Vec<(f64, f64)> = lines
            .iter()
            .filter(|(w, wt)| w.is_finite() && *w > 0.0 && wt.is_finite() && *wt > 0.0)
            .copied()
            .collect();
        if clean.is_empty() {
            return false;
        }
        let Some(src) = self.sources.first_mut() else {
            return false;
        };
        src.spectrum = Spectrum::Lines(clean);
        true
    }

    /// The first source's spectrum as `(wavelength_µm, weight)` lines, for the
    /// picker UI to initialise. `None` if there is no source.
    pub fn first_source_spectrum(&self) -> Option<Vec<(f64, f64)>> {
        Some(self.sources.first()?.spectrum.samples())
    }

    /// The first source's kind + primary parameters, for the picker UI to
    /// initialise: `("collimated_rect"|"collimated_disc"|"point"|"lambertian",
    /// p0, p1, ray_count)`. `None` if there is no source.
    pub fn first_source_spec(&self) -> Option<(&'static str, f64, f64, u32)> {
        let s = self.sources.first()?;
        let (spatial_n, angular_n) = (s.samples.spatial_samples, s.samples.angular_samples);
        Some(match (&s.spatial, &s.angular) {
            (SpatialDistribution3::Point, AngularDistribution3::UniformConeSolidAngle { half_angle_rad }) => {
                ("point", half_angle_rad.to_degrees(), 0.0, angular_n)
            }
            (SpatialDistribution3::Disc { radius_mm }, AngularDistribution3::LambertianHemisphere) => {
                ("lambertian", 2.0 * radius_mm, 0.0, spatial_n * angular_n)
            }
            (SpatialDistribution3::Disc { radius_mm }, _) => ("collimated_disc", 2.0 * radius_mm, 0.0, spatial_n),
            (SpatialDistribution3::Rectangle { half_width_mm, half_height_mm }, _) => {
                ("collimated_rect", 2.0 * half_width_mm, 2.0 * half_height_mm, spatial_n)
            }
            _ => ("collimated_rect", 20.0, 0.5, spatial_n),
        })
    }

    /// A round planar surface (§6): a disc of `radius_mm` whose normal is
    /// `normal`, with the given front/back responses. Rotationally symmetric on
    /// purpose — the building block for mirrors, beam blocks, and beamsplitters.
    fn planar_surface(
        &mut self,
        pos: Vec3,
        normal: Vec3,
        radius_mm: f64,
        front: SurfaceResponse,
        back: SurfaceResponse,
    ) -> u64 {
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = facing(pos, normal);
        self.surfaces.push(SurfaceObject3 {
            common,
            geometry: SurfaceGeometry3::Plane {
                shape: PlanarShape2::Disc {
                    radius_mm: radius_mm.max(0.5),
                },
            },
            response_front: front,
            response_back: back,
        });
        id.0
    }

    /// Add a round mirror spanning the meridional segment `a`→`b` (world, view
    /// plane): a disc whose normal is perpendicular to the segment, so it reads as
    /// the drawn line in the 2D section and a circle in 3D. The "draw a fold
    /// mirror" tool (§5.3). `None` for a degenerate (zero-length) segment.
    pub fn add_planar_mirror(&mut self, a: Vec3, b: Vec3, reflectivity: f64) -> Option<u64> {
        let along = b - a;
        let len = along.length();
        if len < 1e-6 {
            return None;
        }
        let dir = along / len;
        let normal = Vec3::new(-dir.y, dir.x, 0.0);
        let m = SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(reflectivity.clamp(0.0, 1.0)),
        };
        Some(self.planar_surface((a + b) * 0.5, normal, len * 0.5, m.clone(), m))
    }

    /// Add a beam splitter: a round plate at `pos`, tilted `tilt_deg` in the
    /// meridional plane, that reflects `reflectance` and transmits
    /// `transmittance` of the light on either face (the rest is absorbed).
    /// `None`, and nothing added, for fractions outside `[0, 1]` or summing past 1.
    pub fn add_beam_splitter(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        tilt_deg: f64,
        reflectance: f64,
        transmittance: f64,
    ) -> Option<u64> {
        let r = split_response(reflectance, transmittance)?;
        Some(self.add_planar_element(pos, radius_mm, tilt_deg, r.clone(), r))
    }

    /// Add a polarizing beam splitter: a cube of glass with a coated diagonal,
    /// which is what a real one is.
    ///
    /// Everything else in the canvas that polarizes does it by absorbing. This
    /// does it by reflecting, and it is the one polarizing part here whose
    /// behaviour is derived rather than declared: the stack is a plain
    /// quarter-wave mirror, and what makes it polarizing is that it is used at
    /// the Brewster angle between its own layers, where every p reflection
    /// inside it vanishes. So its angle dependence, its wavelength dependence
    /// and its contrast all come out of the transfer matrix at each ray's own
    /// angle, with nothing declared and nothing tabulated.
    ///
    /// The cube glass is **not a free choice**: it is
    /// [`macneille_glass`] of the coating pair, 1.754 for titania on silica,
    /// which is why a polarizing cube is dense flint. Its outer faces carry
    /// the scalar AR stand-in, which is not an approximation where they are
    /// used: at normal incidence s and p are the same thing and one number is
    /// the whole truth.
    ///
    /// Returns `(cube, diagonal)`. They are two objects, because a solid with
    /// an interface inside it is not something the scene model can say yet;
    /// they share a label so the object list at least calls them one part, and
    /// moving one moves only that one.
    ///
    /// `None` for a cube smaller than a millimetre across or a design that
    /// does not sanitize.
    #[allow(clippy::too_many_arguments)]
    pub fn add_polarizing_beam_splitter(
        &mut self,
        pos: Vec3,
        half_mm: f64,
        tilt_deg: f64,
        lambda_nm: f64,
        periods: u32,
        n_high: f64,
        n_low: f64,
    ) -> Option<(u64, u64)> {
        if !half_mm.is_finite() || half_mm < 0.5 || !tilt_deg.is_finite() {
            return None;
        }
        let n0 = crate::thinfilm::macneille_glass(n_high, n_low);
        if !n0.is_finite() || n0 < 1.05 {
            return None;
        }
        let design = CoatingDesign {
            kind: DesignKind::Polarizing,
            lambda_nm,
            aoi_deg: 45.0,
            periods,
            n_high,
            n_low,
        }
        .sanitized()?;
        let coating = film_coating(design, n0, n0);

        // The whole part turns together: at the standard 135 degrees the cube
        // is axis aligned and an axial beam meets its face square on, and any
        // other tilt is that rigid assembly rotated.
        let spin = glam::DQuat::from_rotation_z((tilt_deg - 135.0).to_radians());
        let cube = self.alloc_object_id();
        let mut common = ObjectCommon::new(cube, LayerId(0));
        common.transform =
            Isometry3::new(pos, [spin.x, spin.y, spin.z, spin.w]).unwrap_or_else(|_| Isometry3::from_pos(pos));
        common.label = PBS_LABEL.to_string();
        self.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Box {
                half_extents_mm: Vec3::new(half_mm, half_mm, half_mm),
            },
            material: Material::AbbePoint {
                nd: n0,
                vd: PBS_VD,
            },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3 {
                default: TwoSidedSurfaceResponse {
                    from_outside: SurfaceResponse::DielectricFresnel {
                        ar_residual: SpectralScalar::Constant(PBS_FACE_AR),
                    },
                    from_inside: SurfaceResponse::DielectricFresnel {
                        ar_residual: SpectralScalar::Constant(PBS_FACE_AR),
                    },
                },
                overrides: Vec::new(),
            },
            priority: 0,
        });

        // The diagonal, built with its own frame rather than by `facing`: the
        // rectangle has to lie along the cube's diagonal and its edge, and the
        // shortest rotation onto the normal puts neither there.
        let (sin_t, cos_t) = tilt_deg.to_radians().sin_cos();
        let cols = glam::DMat3::from_cols(
            glam::DVec3::new(-sin_t, cos_t, 0.0),
            glam::DVec3::Z,
            glam::DVec3::new(cos_t, sin_t, 0.0),
        );
        let dq = glam::DQuat::from_mat3(&cols);
        let diagonal = self.alloc_object_id();
        let mut dc = ObjectCommon::new(diagonal, LayerId(0));
        dc.transform =
            Isometry3::new(pos, [dq.x, dq.y, dq.z, dq.w]).unwrap_or_else(|_| Isometry3::from_pos(pos));
        dc.label = PBS_LABEL.to_string();
        let film = SurfaceResponse::Coating { coating };
        self.surfaces.push(SurfaceObject3 {
            common: dc,
            geometry: SurfaceGeometry3::Plane {
                shape: PlanarShape2::Rectangle {
                    half_width_mm: half_mm * std::f64::consts::SQRT_2,
                    half_height_mm: half_mm,
                },
            },
            response_front: film.clone(),
            response_back: film,
        });
        Some((cube.0, diagonal.0))
    }

    /// Add an ideal linear polarizer: a round plate at `pos`, tilted
    /// `tilt_deg`, passing the field along `axis_deg` from the plate's own
    /// local `+X` and absorbing the rest.
    ///
    /// `extinction` is the power fraction of the blocked component that still
    /// gets through: 0 is perfect, 1e-4 is an ordinary sheet. `None` for a
    /// non-finite axis or an extinction outside `[0, 1]`.
    ///
    /// Unpolarized light loses half its flux here. That is the physics, not a
    /// loss the model invented, and it is why the preview draws this as a 50%
    /// attenuator: with no state to act on, half is all it can say.
    pub fn add_polarizer(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        tilt_deg: f64,
        axis_deg: f64,
        extinction: f64,
    ) -> Option<u64> {
        if !axis_deg.is_finite() || !extinction.is_finite() || !(0.0..=1.0).contains(&extinction) {
            return None;
        }
        let r = SurfaceResponse::Polarizer {
            axis_deg,
            extinction: AngularScalar::Constant(extinction),
            substrate_index: 1.0,
        };
        Some(self.add_planar_element(pos, radius_mm, tilt_deg, r.clone(), r))
    }

    /// Add an ideal linear retarder (a waveplate): a round plate at `pos`,
    /// tilted `tilt_deg`, with its fast axis `axis_deg` from the plate's local
    /// `+X` and `retardance_deg` of phase added to the slow axis.
    ///
    /// 90 degrees is a quarter wave, 180 a half wave. Lossless and zero
    /// thickness: the polarization action of a waveplate without the crystal,
    /// so no walk-off, no ordinary and extraordinary rays, and a retardance
    /// that is the number given rather than one that follows from a thickness
    /// and a birefringence at each wavelength. `None` for non-finite arguments.
    pub fn add_retarder(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        tilt_deg: f64,
        axis_deg: f64,
        retardance_deg: f64,
    ) -> Option<u64> {
        if !axis_deg.is_finite() || !retardance_deg.is_finite() {
            return None;
        }
        let r = SurfaceResponse::Retarder {
            axis_deg,
            retardance_deg,
            substrate_index: 1.0,
            crystal: None,
        };
        Some(self.add_planar_element(pos, radius_mm, tilt_deg, r.clone(), r))
    }

    /// Add a dichroic filter: a round plate at `pos`, tilted `tilt_deg`, with the
    /// coating `spec` on both faces. `None` for a non-finite spec.
    pub fn add_dichroic(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        tilt_deg: f64,
        spec: DichroicSpec,
    ) -> Option<u64> {
        let r = spec.sanitized()?.response();
        Some(self.add_planar_element(pos, radius_mm, tilt_deg, r.clone(), r))
    }

    /// Make both faces of a surface object a beam splitter with these fractions.
    /// False (scene unchanged) for invalid fractions or a non-surface id.
    pub fn set_surface_split(&mut self, id: u64, reflectance: f64, transmittance: f64) -> bool {
        let Some(r) = split_response(reflectance, transmittance) else {
            return false;
        };
        self.set_surface_responses(id, r.clone(), r)
    }

    /// Turn a polarizing element's axis: a polarizer's transmission axis or a
    /// retarder's fast axis, degrees from the surface's local `+X`.
    ///
    /// False (scene unchanged) for a non-finite angle, or an id that is not a
    /// surface carrying one of those two responses. The angle is stored as
    /// given rather than wrapped, because 370 and 10 are the same element and a
    /// field the user is typing into should not jump under them.
    pub fn set_polarization_axis(&mut self, id: u64, axis_deg: f64) -> bool {
        if !axis_deg.is_finite() {
            return false;
        }
        self.map_polarizing(id, &mut |r| match r {
            SurfaceResponse::Polarizer { axis_deg: a, .. } => *a = axis_deg,
            SurfaceResponse::Retarder { axis_deg: a, .. } => *a = axis_deg,
            _ => {}
        })
    }

    /// Set a polarizer's extinction: the **power** fraction of the blocked
    /// component that still gets through. Clamped to `[0, 1]`. False for a
    /// non-finite value or an element that is not a polarizer.
    pub fn set_polarizer_extinction(&mut self, id: u64, extinction: f64) -> bool {
        if !extinction.is_finite() {
            return false;
        }
        let v = extinction.clamp(0.0, 1.0);
        let mut touched = false;
        let ok = self.map_polarizing(id, &mut |r| {
            if let SurfaceResponse::Polarizer { extinction: e, .. } = r {
                *e = AngularScalar::Constant(v);
                touched = true;
            }
        });
        ok && touched
    }

    /// Set a polarizer's extinction from a datasheet: the power fraction that
    /// leaks past the blocked axis, against angle of incidence in degrees.
    ///
    /// This is the only way to say where a polarizer stops working. A wire
    /// grid loses contrast off axis, a thin-film polarizer or a MacNeille cube
    /// is built around a Brewster match at one angle and gives up a couple of
    /// degrees away from it, and a Glan prism has a hard acceptance cone. None
    /// of that follows from anything the kernel can compute about an ideal
    /// layer, so it comes from measurement or not at all.
    ///
    /// Outside the angles given, the table holds its endpoint. A real
    /// polarizer is worse out there, not the same, so the range it was
    /// measured over is the range to trust.
    ///
    /// False for an element that is not a polarizer, or for a table that is
    /// not two or more strictly increasing angles in `[0, 90)` with values in
    /// `[0, 1]`.
    pub fn set_polarizer_extinction_table(
        &mut self,
        id: u64,
        aoi_deg: &[f64],
        extinction: &[f64],
    ) -> bool {
        let table = AngularScalar::Tabulated {
            aoi_deg: aoi_deg.to_vec(),
            values: extinction.to_vec(),
            interpolation: Interpolation::Linear,
            out_of_range: OutOfRangePolicy::Clamp,
        };
        if table.validate("extinction").is_err() || !table.is_unit_fraction() {
            return false;
        }
        let mut touched = false;
        let ok = self.map_polarizing(id, &mut |r| {
            if let SurfaceResponse::Polarizer { extinction: e, .. } = r {
                *e = table.clone();
                touched = true;
            }
        });
        ok && touched
    }

    /// A polarizer's extinction curve, if `id` is one.
    pub fn polarizer_extinction(&self, id: u64) -> Option<&AngularScalar> {
        let s = self.surfaces.iter().find(|s| s.common.id.0 == id)?;
        match &s.response_front {
            SurfaceResponse::Polarizer { extinction, .. } => Some(extinction),
            _ => None,
        }
    }

    /// Set a retarder's retardance in degrees: 90 is a quarter wave, 180 a half
    /// wave. False for a non-finite value or an element that is not a retarder.
    pub fn set_retarder_retardance(&mut self, id: u64, retardance_deg: f64) -> bool {
        if !retardance_deg.is_finite() {
            return false;
        }
        let mut touched = false;
        let ok = self.map_polarizing(id, &mut |r| {
            if let SurfaceResponse::Retarder { retardance_deg: d, .. } = r {
                *d = retardance_deg;
                touched = true;
            }
        });
        ok && touched
    }

    /// Make a retarder out of a real crystal, or `None` to make it ideal
    /// again. False for anything that is not a retarder, and for a crystal
    /// that could not be one: an index below 1, no birefringence to retard
    /// with, or a design wavelength that is not positive.
    ///
    /// Its `retardance_deg` becomes the nominal: what the plate gives on axis
    /// at the design wavelength. Everywhere else the retardance follows from
    /// the plate, so it moves as `1/λ` and as the square of the obliquity.
    pub fn set_retarder_crystal(&mut self, id: u64, crystal: Option<Waveplate>) -> bool {
        if let Some(c) = &crystal {
            let ok = c.n_o.is_finite()
                && c.n_e.is_finite()
                && c.n_o >= 1.0
                && c.n_e >= 1.0
                && (c.n_e - c.n_o).abs() >= 1e-9
                && c.design_lambda_um.is_finite()
                && c.design_lambda_um > 0.0
                && c.order <= 1000;
            if !ok {
                return false;
            }
        }
        let mut touched = false;
        let ok = self.map_polarizing(id, &mut |r| {
            if let SurfaceResponse::Retarder { crystal: c, .. } = r {
                *c = crystal.clone();
                touched = true;
            }
        });
        ok && touched
    }

    /// Set the index of the glass a polarizing element is laminated in, or 1
    /// for a bare layer with no faces. False below 1, for a non-finite value,
    /// or for an element that is not a polarizer or a retarder.
    ///
    /// This is the element's own glass and not the medium it sits in. It is
    /// what gives a real sheet polarizer most of its dependence on the angle
    /// it is used at, and an ideal layer has none.
    pub fn set_polarizing_substrate(&mut self, id: u64, index: f64) -> bool {
        if !index.is_finite() || index < 1.0 {
            return false;
        }
        self.map_polarizing(id, &mut |r| match r {
            SurfaceResponse::Polarizer {
                substrate_index, ..
            }
            | SurfaceResponse::Retarder {
                substrate_index, ..
            } => *substrate_index = index,
            _ => {}
        })
    }

    /// Apply `f` to both faces of a surface that carries a polarizing response.
    /// False for any other id, so an edit cannot silently land on the wrong
    /// kind of element.
    fn map_polarizing(&mut self, id: u64, f: &mut dyn FnMut(&mut SurfaceResponse)) -> bool {
        let Some(s) = self.surfaces.iter_mut().find(|s| s.common.id.0 == id) else {
            return false;
        };
        let polarizing = |r: &SurfaceResponse| {
            matches!(
                r,
                SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. }
            )
        };
        if !polarizing(&s.response_front) && !polarizing(&s.response_back) {
            return false;
        }
        f(&mut s.response_front);
        f(&mut s.response_back);
        true
    }

    /// Give both faces of a surface object the dichroic coating `spec`. False
    /// (scene unchanged) for a non-finite spec or a non-surface id.
    pub fn set_surface_dichroic(&mut self, id: u64, spec: DichroicSpec) -> bool {
        let Some(spec) = spec.sanitized() else {
            return false;
        };
        let r = spec.response();
        self.set_surface_responses(id, r.clone(), r)
    }

    /// Add a dichroic filter as a **real coated glass plate**: a disc of
    /// substrate with the thin-film stack on the face the light meets first.
    ///
    /// The plate is a body, not a plane, and that is the difference. The tracer
    /// crosses it the way light does: reflect at the coated face, refract into
    /// the glass, cross at the internal angle, refract back out at the far face.
    /// So the transmitted beam leaves parallel to where it started but displaced
    /// sideways, about a millimetre for a 3 mm plate at 45°, carrying the glass
    /// path in its optical path length, and the far face returns its own
    /// reflection as a second beam. None of that is special-cased here; it is
    /// what tracing a solid already does.
    ///
    /// The coating goes on one patch and plain glass on the other, which is why
    /// this needs per-face responses rather than one response for the body.
    pub fn add_plate_dichroic(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        thickness_mm: f64,
        tilt_deg: f64,
        edge_nm: f64,
        long_pass: bool,
    ) -> Option<BodyId> {
        let design = CoatingDesign {
            kind: if long_pass {
                DesignKind::LongPass
            } else {
                DesignKind::ShortPass
            },
            ..CoatingDesign::long_pass(edge_nm, axial_aoi_deg(tilt_deg))
        }
        .sanitized()?;
        let thickness_mm = if thickness_mm.is_finite() && thickness_mm > 0.0 {
            thickness_mm
        } else {
            DEFAULT_PLATE_MM
        };
        let radius_mm = if radius_mm.is_finite() && radius_mm > 0.5 {
            radius_mm
        } else {
            return None;
        };
        let coating = film_coating(design, self.env.ambient_index, PLATE_ND);
        Some(self.add_coated_plate(pos, radius_mm, thickness_mm, tilt_deg, coating))
    }

    /// A glass disc with `coating` on the face that meets an axial beam, and
    /// bare glass on the other.
    pub fn add_coated_plate(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        thickness_mm: f64,
        tilt_deg: f64,
        coating: Coating,
    ) -> BodyId {
        // A plate is a singlet with both faces flat.
        let g = SingletGeom {
            r1_mm: f64::INFINITY,
            r2_mm: f64::INFINITY,
            thickness_mm,
            semi_aperture_mm: radius_mm,
            conic1: 0.0,
            conic2: 0.0,
        };
        let curves = singlet_curves(g, SingletAspheres::default());
        let edges: Vec<ProfileEdge2> = curves
            .into_iter()
            .map(|curve| ProfileEdge2 {
                patch: self.alloc_patch_id(),
                curve,
            })
            .collect();
        // `singlet_curves` orders the loop [front face, rim, back face, rim],
        // with the front face at local −X. The transform below carries local −X
        // onto the plate normal, so the front face is the one an axial beam
        // reaches first, and that is where the coating belongs.
        let coated = edges[0].patch;
        let phi = tilt_deg.to_radians();
        let normal = Vec3::new(phi.cos(), phi.sin(), 0.0);
        let q = glam::DQuat::from_rotation_arc(-Vec3::X, normal.normalize());
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform =
            Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap_or_else(|_| Isometry3::from_pos(pos));
        let film = coated_face(SurfaceResponse::Coating { coating });
        self.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Profile(ProfileSolid3 {
                profile: ClosedProfile2 { edges },
                construction: ProfileConstruction3::RevolveLocalX,
            }),
            material: Material::AbbePoint {
                nd: PLATE_ND,
                vd: PLATE_VD,
            },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3 {
                // Everything else on the body is plain glass: the far face and
                // the rim get ordinary Fresnel, which is what they are.
                default: TwoSidedSurfaceResponse::default(),
                overrides: vec![PatchResponseOverride {
                    patch: coated,
                    response: film,
                }],
            },
            priority: 0,
        });
        BodyId(id.0)
    }

    /// The coating on a body's coated face, if it carries one.
    pub fn body_coating(&self, id: u64) -> Option<&Coating> {
        let b = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        b.surface.overrides.iter().find_map(|o| {
            match &o.response.from_outside {
                SurfaceResponse::Coating { coating } => Some(coating),
                _ => None,
            }
        })
    }

    /// The plate thickness and semi-aperture of a coated plate body, in mm.
    pub fn plate_geometry(&self, id: u64) -> Option<(f64, f64)> {
        let b = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        let SolidGeometry3::Profile(p) = &b.geometry else {
            return None;
        };
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut ha: f64 = 0.0;
        for e in &p.profile.edges {
            for v in curve_extent(&e.curve) {
                lo = lo.min(v.x);
                hi = hi.max(v.x);
                ha = ha.max(v.y.abs());
            }
        }
        (hi > lo).then_some((hi - lo, ha))
    }

    /// Add a dichroic filter as a **real thin-film stack** on a glass plate.
    ///
    /// The design is solved so the edge lands where asked *at the angle the
    /// plate is being placed at*, which for the usual 45° fold is nowhere near
    /// where the same stack would cut on axis.
    ///
    /// `tilt_deg` sets the plate normal to `(cos φ, sin φ, 0)`, so a beam down
    /// the axis meets it at `arccos|cos φ|`: 0° for a plate square to the beam
    /// at 180°, 45° for the usual 135° fold, and grazing at 90° where the plate
    /// lies edge-on and the beam misses it entirely. Taking the design angle
    /// from the placement is the whole point — a coating is designed for the
    /// geometry it will be used in.
    pub fn add_film_dichroic(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        tilt_deg: f64,
        edge_nm: f64,
        long_pass: bool,
    ) -> Option<u64> {
        let design = CoatingDesign {
            kind: if long_pass {
                DesignKind::LongPass
            } else {
                DesignKind::ShortPass
            },
                ..CoatingDesign::long_pass(edge_nm, axial_aoi_deg(tilt_deg))
        }
        .sanitized()?;
        let (front, back) = self.coating_pair(design);
        Some(self.add_planar_element(pos, radius_mm, tilt_deg, front, back))
    }

    /// Both faces of one coated plate: the film side, and the side that is
    /// entered through the glass.
    ///
    /// A plate is one object with two faces and they are not the same surface.
    /// Light arriving on the coated face meets the stack first; light arriving
    /// from behind crosses the substrate and its far face before reaching the
    /// stack. For a lossless stack reciprocity makes the two agree, and for one
    /// with any absorption in it they do not.
    fn coating_pair(&self, design: CoatingDesign) -> (SurfaceResponse, SurfaceResponse) {
        let ambient = self.env.ambient_index;
        let film = plate_coating(
            design,
            crate::thinfilm::materials::bk7(),
            ExitFace::Bare,
            ambient,
        );
        let mut back = film.clone();
        back.incidence = Incidence::Substrate;
        (
            SurfaceResponse::Coating { coating: film },
            SurfaceResponse::Coating { coating: back },
        )
    }

    /// Rebuild the coating on a surface from a design. False (scene unchanged)
    /// for a non-finite design or a non-surface id.
    pub fn set_surface_coating_design(&mut self, id: u64, design: CoatingDesign) -> bool {
        let Some(design) = design.sanitized() else {
            return false;
        };
        let (front, back) = self.coating_pair(design);
        self.set_surface_responses(id, front, back)
    }

    /// Set one layer's physical thickness, in nanometres, on both faces.
    ///
    /// This is the handle that makes the stack real rather than a recipe: a
    /// deposition run misses its targets, and so can you. The design is kept so
    /// the editor can still show what the stack started as, but the layers are
    /// what the transfer matrix reads, so an edited stack is simply a different
    /// coating. False for a bad index, a non-finite or negative thickness, a
    /// non-surface id, or a surface that carries no film.
    pub fn set_coating_layer_thickness(&mut self, id: u64, index: usize, nm: f64) -> bool {
        if !nm.is_finite() || nm < 0.0 {
            return false;
        }
        let Some(s) = self.surfaces.iter_mut().find(|s| s.common.id.0 == id) else {
            return false;
        };
        let mut touched = false;
        for side in [&mut s.response_front, &mut s.response_back] {
            if let SurfaceResponse::Coating { coating } = side {
                match coating.layers.get_mut(index) {
                    Some(l) => {
                        l.thickness_nm = nm;
                        touched = true;
                    }
                    None => return false,
                }
            }
        }
        touched
    }

    /// The angle, in degrees, at which the scene's light actually meets this
    /// surface: the angle between its normal and the first source's rays.
    ///
    /// A coating is a function of wavelength *and* angle, so a plot of one is
    /// meaningless until you say which angle. This is the angle worth plotting
    /// at by default, because it is the one the part is being used at. `None`
    /// for a non-surface id or a scene with no light in it.
    pub fn surface_axial_aoi_deg(&self, id: u64) -> Option<f64> {
        let s = self.surfaces.iter().find(|s| s.common.id.0 == id)?;
        let normal = s.common.transform.quat() * Vec3::Z;
        let dir = self.sources.first()?.emit().first()?.ray.dir;
        Some(dir.dot(normal).abs().clamp(0.0, 1.0).acos().to_degrees())
    }

    /// Coat one face of a lens, a plate, or anything else built on the
    /// four-edge revolved profile, or strip it back to bare glass with `None`.
    ///
    /// [`Scene3::set_object_coating_design`] rewrites films that are already
    /// on a body, which is what editing a dichroic needs. This puts one where
    /// there was none, which is what coating a bare lens needs, and it says
    /// which face.
    ///
    /// The stack is solved against the body's **own glass**, because what a
    /// coating reflects depends on what it is sitting on: the same three
    /// layers on crown and on flint are not the same coating. False for an
    /// unknown id, a body that is not a four-edge profile, or a design that
    /// does not sanitize.
    pub fn set_body_face_coating(
        &mut self,
        id: u64,
        face: BodyFace,
        design: Option<CoatingDesign>,
    ) -> bool {
        let (ambient, glass) = self.coated_face_media(id, crate::material::LAMBDA_D);
        let film = match design {
            None => None,
            Some(d) => match d.sanitized() {
                None => return false,
                Some(d) => Some(SurfaceResponse::Coating {
                    coating: film_coating(d, ambient, glass),
                }),
            },
        };
        let Some(b) = self.bodies.iter_mut().find(|b| b.common.id.0 == id) else {
            return false;
        };
        let patch = {
            let SolidGeometry3::Profile(solid) = &b.geometry else {
                return false;
            };
            if solid.profile.edges.len() != 4 {
                return false;
            }
            solid.profile.edges[face.edge()].patch
        };
        b.surface.overrides.retain(|o| o.patch != patch);
        if let Some(f) = film {
            b.surface.overrides.push(PatchResponseOverride {
                patch,
                response: coated_face(f),
            });
        }
        true
    }

    /// The coating on one named face of a body, if that face carries one.
    pub fn body_face_coating(&self, id: u64, face: BodyFace) -> Option<&Coating> {
        let b = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        let SolidGeometry3::Profile(solid) = &b.geometry else {
            return None;
        };
        let patch = solid.profile.edges.get(face.edge())?.patch;
        b.surface.overrides.iter().find_map(|o| match &o.response.from_outside {
            SurfaceResponse::Coating { coating } if o.patch == patch => Some(coating),
            _ => None,
        })
    }

    /// The media either side of an object's coated face at `lambda_um`:
    /// `(incident, behind)`.
    ///
    /// For a coated solid the film sits on that solid's own glass, and what a
    /// stack reflects depends on what it is sitting on, so a plot of one has to
    /// be told. For a zero-thickness plate both sides are the ambient medium and
    /// the coating carries its own substrate.
    pub fn coated_face_media(&self, id: u64, lambda_um: f64) -> (f64, f64) {
        let ambient = self.env.ambient_index;
        let behind = self
            .bodies
            .iter()
            .find(|b| b.common.id.0 == id)
            .and_then(|b| match &b.material {
                Material::AbbePoint { nd, vd } => {
                    Some(crate::material::Cauchy::from_abbe(*nd, *vd).index(lambda_um))
                }
                Material::Catalog(gid) => {
                    self.catalog.get(*gid).map(|g| g.sell.index(lambda_um))
                }
                _ => None,
            })
            .unwrap_or(ambient);
        (ambient, behind)
    }

    /// The coating on an object, whether it is a zero-thickness plate or a
    /// coated face of a solid.
    ///
    /// The editor should not have to know which, and after plates grew
    /// thickness most of them are solids.
    pub fn coating_for_object(&self, id: u64) -> Option<&Coating> {
        self.surface_coating(id).or_else(|| self.body_coating(id))
    }

    /// The angle the scene's light meets an object's coated face at, degrees.
    ///
    /// A coating is a function of wavelength *and* angle, so a plot of one means
    /// nothing until you say which angle. This is the angle worth plotting at,
    /// because it is the one the part is being used at. `None` for an object
    /// with no coating, or a scene with no light in it.
    pub fn object_axial_aoi_deg(&self, id: u64) -> Option<f64> {
        let normal = if let Some(s) = self.surfaces.iter().find(|s| s.common.id.0 == id) {
            s.common.transform.quat() * Vec3::Z
        } else {
            // A coated plate is built with its local −X carried onto the plate
            // normal, so that is where its coated face looks.
            let b = self.bodies.iter().find(|b| b.common.id.0 == id)?;
            b.common.transform.quat() * -Vec3::X
        };
        let dir = self.sources.first()?.emit().first()?.ray.dir;
        Some(dir.dot(normal).abs().clamp(0.0, 1.0).acos().to_degrees())
    }

    /// Rebuild the coating on an object from a design, whichever kind it is.
    pub fn set_object_coating_design(&mut self, id: u64, design: CoatingDesign) -> bool {
        let Some(design) = design.sanitized() else {
            return false;
        };
        if self.surfaces.iter().any(|s| s.common.id.0 == id) {
            return self.set_surface_coating_design(id, design);
        }
        let ambient = self.env.ambient_index;
        let Some(b) = self.bodies.iter_mut().find(|b| b.common.id.0 == id) else {
            return false;
        };
        let coating = film_coating(design, ambient, PLATE_ND);
        let mut touched = false;
        for o in &mut b.surface.overrides {
            if matches!(o.response.from_outside, SurfaceResponse::Coating { .. }) {
                o.response = coated_face(SurfaceResponse::Coating {
                    coating: coating.clone(),
                });
                touched = true;
            }
        }
        touched
    }

    /// Set one film layer's thickness on an object, whichever kind it is.
    pub fn set_object_layer_thickness(&mut self, id: u64, index: usize, nm: f64) -> bool {
        if !nm.is_finite() || nm < 0.0 {
            return false;
        }
        if self.surfaces.iter().any(|s| s.common.id.0 == id) {
            return self.set_coating_layer_thickness(id, index, nm);
        }
        let Some(b) = self.bodies.iter_mut().find(|b| b.common.id.0 == id) else {
            return false;
        };
        let mut touched = false;
        for o in &mut b.surface.overrides {
            for side in [&mut o.response.from_outside, &mut o.response.from_inside] {
                if let SurfaceResponse::Coating { coating } = side {
                    match coating.layers.get_mut(index) {
                        Some(l) => {
                            l.thickness_nm = nm;
                            touched = true;
                        }
                        None => return false,
                    }
                }
            }
        }
        touched
    }

    /// The coating on a surface's front face, if it carries one.
    pub fn surface_coating(&self, id: u64) -> Option<&Coating> {
        let s = self.surfaces.iter().find(|s| s.common.id.0 == id)?;
        match &s.response_front {
            SurfaceResponse::Coating { coating } => Some(coating),
            _ => None,
        }
    }

    fn set_surface_responses(
        &mut self,
        id: u64,
        front: SurfaceResponse,
        back: SurfaceResponse,
    ) -> bool {
        match self.surfaces.iter_mut().find(|s| s.common.id.0 == id) {
            Some(s) => {
                s.response_front = front;
                s.response_back = back;
                true
            }
            None => false,
        }
    }

    /// A surface object's `(front, back)` responses, if `id` is one.
    pub fn surface_responses(&self, id: u64) -> Option<(&SurfaceResponse, &SurfaceResponse)> {
        let s = self.surfaces.iter().find(|s| s.common.id.0 == id)?;
        Some((&s.response_front, &s.response_back))
    }

    /// Add a round optical element (a flat plate) at `pos`, tilted `tilt_deg` in
    /// the meridional plane, radius `radius_mm`, with configurable front/back
    /// responses — make either face reflective, absorbing, clear, or refractive.
    /// The angle an axial beam meets a plate tilted by `tilt_deg`, in degrees.
    ///
    /// [`Scene3::add_planar_element`] points the normal along `(cos φ, sin φ, 0)`
    /// and the default beam runs along `+X`, so the angle of incidence is
    /// `arccos|cos φ|`. Worth its own function because getting it wrong is
    /// invisible at 135°, where the two plausible formulas happen to agree.
    pub fn add_planar_element(
        &mut self,
        pos: Vec3,
        radius_mm: f64,
        tilt_deg: f64,
        front: SurfaceResponse,
        back: SurfaceResponse,
    ) -> u64 {
        let phi = tilt_deg.to_radians();
        let normal = Vec3::new(phi.cos(), phi.sin(), 0.0);
        self.planar_surface(pos, normal, radius_mm, front, back)
    }

    /// Set a surface object's front/back responses from kind strings
    /// (`mirror`/`absorb`/`clear`/`refract`). False for an unknown kind or a
    /// non-surface id.
    pub fn set_surface_response(&mut self, id: u64, front: &str, back: &str) -> bool {
        let (Some(f), Some(b)) = (response_from_kind(front), response_from_kind(back)) else {
            return false;
        };
        match self.surfaces.iter_mut().find(|s| s.common.id.0 == id) {
            Some(s) => {
                s.response_front = f;
                s.response_back = b;
                true
            }
            None => false,
        }
    }

    /// A surface object's `(front, back)` response kinds, if `id` is one.
    pub fn surface_response(&self, id: u64) -> Option<(&'static str, &'static str)> {
        let s = self.surfaces.iter().find(|s| s.common.id.0 == id)?;
        Some((response_kind(&s.response_front), response_kind(&s.response_back)))
    }

    /// Set a surface's meridional tilt — its plane-normal angle in the X–Y plane,
    /// degrees. (A surface's base orientation is a facing, not a body's pure-Z
    /// rotation, so it gets its own setter.) Keeps position. False for a non-surface.
    pub fn set_surface_tilt_z_deg(&mut self, id: u64, deg: f64) -> bool {
        if !deg.is_finite() {
            return false;
        }
        let phi = deg.to_radians();
        let normal = Vec3::new(phi.cos(), phi.sin(), 0.0);
        if let Some(s) = self.surfaces.iter_mut().find(|s| s.common.id.0 == id) {
            s.common.transform = facing(s.common.transform.pos, normal);
            return true;
        }
        if let Some(a) = self.apertures.iter_mut().find(|a| a.common.id.0 == id) {
            a.common.transform = facing(a.common.transform.pos, normal);
            return true;
        }
        if let Some(d) = self.detectors.iter_mut().find(|d| d.common.id.0 == id) {
            d.common.transform = facing(d.common.transform.pos, normal);
            return true;
        }
        false
    }

    /// Read a surface's (or aperture's, or detector's) meridional tilt — its
    /// normal's X–Y angle.
    pub fn surface_tilt_z_deg(&self, id: u64) -> Option<f64> {
        let t = self
            .surfaces
            .iter()
            .find(|s| s.common.id.0 == id)
            .map(|s| &s.common.transform)
            .or_else(|| {
                self.apertures
                    .iter()
                    .find(|a| a.common.id.0 == id)
                    .map(|a| &a.common.transform)
            })
            .or_else(|| {
                self.detectors
                    .iter()
                    .find(|d| d.common.id.0 == id)
                    .map(|d| &d.common.transform)
            })?;
        let n = surface_normal(t);
        Some(n.y.atan2(n.x).to_degrees())
    }

    /// Add a circular aperture stop at `pos`, tilted `tilt_deg` in the meridional
    /// plane: rays inside `opening_mm` pass, the rest hit an absorbing mask. The
    /// mask is a finite disc (drawable), large enough to block the working beam.
    pub fn add_aperture(&mut self, pos: Vec3, opening_mm: f64, tilt_deg: f64) -> u64 {
        let phi = tilt_deg.to_radians();
        let normal = Vec3::new(phi.cos(), phi.sin(), 0.0);
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = facing(pos, normal);
        let opening = opening_mm.max(0.2);
        self.apertures.push(Aperture3 {
            common,
            shape: ApertureShape3 {
                plane_local: Plane3::XY,
                clear_opening: PlanarShape2::Disc { radius_mm: opening },
                outer_extent: Some(PlanarShape2::Disc {
                    radius_mm: (opening * 2.5).max(opening + 6.0),
                }),
            },
            interaction: ApertureInteraction::Absorb,
            primary_stop_for: Vec::new(),
        });
        id.0
    }

    /// Add a glass prism at `pos`: an equilateral triangle (apex up) of side
    /// `side_mm`, extruded out of the meridional plane, glass `(nd, vd)`, tilted
    /// `tilt_deg` about +Z. A dispersing element — the collimated white beam fans
    /// into a spectrum through it. Returns the new body id.
    pub fn add_prism(&mut self, pos: Vec3, side_mm: f64, tilt_deg: f64, nd: f64, vd: f64) -> u64 {
        let s = side_mm.max(2.0);
        let h = s * 3.0_f64.sqrt() / 2.0;
        let verts = [
            Vec2::new(0.0, 2.0 * h / 3.0),
            Vec2::new(-s / 2.0, -h / 3.0),
            Vec2::new(s / 2.0, -h / 3.0),
        ];
        let edges: Vec<_> = (0..3)
            .map(|i| ProfileEdge2 {
                patch: self.alloc_patch_id(),
                curve: Curve2::Line {
                    a: verts[i],
                    b: verts[(i + 1) % 3],
                },
            })
            .collect();
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = Isometry3::from_pos(pos);
        let depth = (s / 3.0).max(3.0);
        self.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Profile(ProfileSolid3 {
                profile: ClosedProfile2 { edges },
                construction: ProfileConstruction3::ExtrudeLocalZ {
                    z_min_mm: -depth,
                    z_max_mm: depth,
                },
            }),
            material: crate::material::Material::AbbePoint { nd, vd },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        if tilt_deg != 0.0 {
            self.set_object_rotation_z_deg(id.0, tilt_deg);
        }
        id.0
    }

    /// Add a rectangular area detector at `pos`, its face pointing along
    /// `normal` (local +Z is the face normal). Returns the new detector id.
    pub fn add_area_detector(
        &mut self,
        pos: Vec3,
        normal: Vec3,
        half_width_mm: f64,
        half_height_mm: f64,
        bins: [usize; 2],
    ) -> DetectorId {
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = facing(pos, normal);
        self.detectors.push(Detector3 {
            common,
            geometry: DetectorGeometry3::Rectangle {
                half_width_mm,
                half_height_mm,
            },
            bins,
            response: DetectorResponse::Unity,
            termination: DetectorTermination::Terminate,
            measure: DetectorMeasure::Power,
        });
        DetectorId(id.0)
    }

    /// The default authoring scene: one biconvex singlet at the origin, a
    /// collimated beam from the left, and an area detector near the focus. This
    /// is what the app boots into (§4.1: a singlet traced in the Y–Z plane).
    pub fn default_scene() -> Self {
        let mut s = Scene3::new();
        s.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
        s.add_collimated_beam(
            Vec3::new(-70.0, 0.0, 0.0),
            10.0,
            0.25,
            21,
            white_spectrum(),
        );
        // A symmetric biconvex R=50 N-BK7 singlet has f ≈ 48 mm; the detector sits
        // near that focus, facing back down the axis.
        s.add_area_detector(
            Vec3::new(48.0, 0.0, 0.0),
            -Vec3::X,
            12.0,
            12.0,
            [64, 64],
        );
        s
    }

    // --- edits the inspector cards drive ----------------------------------

    /// Move any placed object (body / detector / surface / aperture) or source by
    /// id, keeping its orientation. Returns false if no object has that id, or the
    /// position is non-finite.
    pub fn set_object_position(&mut self, id: u64, pos: Vec3) -> bool {
        if !pos.is_finite() {
            return false;
        }
        match self.object_transform_mut(id) {
            Some(t) => {
                t.pos = pos;
                true
            }
            None => false,
        }
    }

    /// Rotate an object in the meridional plane: a right-handed angle about local
    /// +Z (the view-plane normal), matching the v1 in-plane rotation. Returns
    /// false if no object has that id.
    pub fn set_object_rotation_z_deg(&mut self, id: u64, deg: f64) -> bool {
        if !deg.is_finite() {
            return false;
        }
        let h = 0.5 * deg.to_radians();
        let rot = [0.0, 0.0, h.sin(), h.cos()];
        match self.object_transform_mut(id) {
            Some(t) => match Isometry3::new(t.pos, rot) {
                Ok(next) => {
                    *t = next;
                    true
                }
                Err(_) => false,
            },
            None => false,
        }
    }

    /// The clear opening of an aperture as `(inner_mm, outer_mm)` — the pass
    /// region's minimum and maximum radius. A plain disc opening reports an inner
    /// radius of 0. `None` if `id` is not an aperture (or its opening is not a
    /// disc/annulus).
    pub fn aperture_opening(&self, id: u64) -> Option<(f64, f64)> {
        let a = self.apertures.iter().find(|a| a.common.id.0 == id)?;
        match &a.shape.clear_opening {
            PlanarShape2::Disc { radius_mm } => Some((0.0, *radius_mm)),
            PlanarShape2::Annulus { inner_mm, outer_mm } => Some((*inner_mm, *outer_mm)),
            _ => None,
        }
    }

    /// Set an aperture's clear opening to the ring `inner_mm..outer_mm` (the min
    /// and max radius that pass). An inner radius of 0 (or less) is a plain disc
    /// opening; a positive inner radius adds a central obstruction. The outer
    /// mask is grown if needed so it still surrounds the opening. Rejects the
    /// edit (returns false, scene unchanged) if the radii are non-finite or do
    /// not satisfy `0 <= inner < outer`, or `id` is not an aperture.
    pub fn set_aperture_opening(&mut self, id: u64, inner_mm: f64, outer_mm: f64) -> bool {
        if !inner_mm.is_finite() || !outer_mm.is_finite() {
            return false;
        }
        let inner = inner_mm.max(0.0);
        if outer_mm <= inner {
            return false;
        }
        let Some(a) = self.apertures.iter_mut().find(|a| a.common.id.0 == id) else {
            return false;
        };
        a.shape.clear_opening = if inner > 0.0 {
            PlanarShape2::Annulus {
                inner_mm: inner,
                outer_mm,
            }
        } else {
            PlanarShape2::Disc { radius_mm: outer_mm }
        };
        // Keep the absorbing mask surrounding the opening (a plate needs a rim).
        if let Some(PlanarShape2::Disc { radius_mm }) = &mut a.shape.outer_extent {
            if *radius_mm <= outer_mm {
                *radius_mm = (outer_mm * 2.5).max(outer_mm + 6.0);
            }
        }
        true
    }

    /// Replace a singlet body's geometry, keeping its aspheric terms. Rejects
    /// the edit (returns false, scene unchanged) if the id is not a singlet or
    /// the new geometry is not revolvable — a bad card edit never corrupts the
    /// scene.
    pub fn set_singlet_geom(&mut self, id: u64, g: SingletGeom) -> bool {
        match self.singlet_shape(id) {
            Some((_, a)) => self.set_singlet_shape(id, g, a),
            None => false,
        }
    }

    /// Replace a singlet body's aspheric terms, keeping its geometry. Same
    /// refusal rules as [`Scene3::set_singlet_shape`].
    pub fn set_singlet_aspheres(&mut self, id: u64, a: SingletAspheres) -> bool {
        match self.singlet_shape(id) {
            Some((g, _)) => self.set_singlet_shape(id, g, a),
            None => false,
        }
    }

    /// Reshape one face of a singlet so it passes through the given points,
    /// each `(height_mm, sag_mm)` with the sag measured from that face's vertex
    /// along +X. The face keeps its curvature and conic constant. Its eight
    /// aspheric terms (A4 to A18) become the set that passes through every
    /// point and bends least doing so: the departure `d(u)` from the conic,
    /// with `u` the height over the semi-aperture, minimises `∫₀¹ d''(u)² du`.
    /// Interpolating with only as many terms as points would also pass
    /// through them, but it swings between them by several times the edit.
    ///
    /// False (scene unchanged) for no points or more than eight, a height
    /// outside `(0, semi-aperture]`, two points at one height, a base conic
    /// that is undefined out to a point's height, a solve that misses a point
    /// by more than 1e-9 of its sag, or a shape the singlet setters refuse.
    pub fn fit_singlet_face(&mut self, id: u64, back: bool, points: &[(f64, f64)]) -> bool {
        let Some((g, mut a)) = self.singlet_shape(id) else {
            return false;
        };
        let n = points.len();
        let ha = g.semi_aperture_mm;
        if n == 0 || n > 8 || ha.is_nan() || ha <= 0.0 {
            return false;
        }
        let (r, k) = if back {
            (g.r2_mm, g.conic2)
        } else {
            (g.r1_mm, g.conic1)
        };
        let c = curvature(r);
        // Heights scaled by the semi-aperture keep the system well
        // conditioned; each coefficient is scaled back at the end.
        let mut us = Vec::with_capacity(n);
        let mut rhs = Vec::with_capacity(n);
        for &(h, sag) in points {
            if !h.is_finite() || !sag.is_finite() || h <= 0.0 || h > ha * (1.0 + 1e-12) {
                return false;
            }
            if (1.0 + k) * c * c * h * h >= 1.0 {
                return false;
            }
            let u = h / ha;
            if us.iter().any(|v: &f64| (v - u).abs() < 1e-9) {
                return false;
            }
            us.push(u);
            rhs.push(sag - even_asphere_sag(c, k, &[0.0; 8], h));
        }
        // Term j is u^p with p = 2j + 4.
        const TERMS: usize = 8;
        let power = |j: usize| 2 * j as i32 + 4;
        let row = |u: f64| (0..TERMS).map(|j| u.powi(power(j))).collect::<Vec<f64>>();
        // Minimise aᵀHa subject to A a = d, as one linear system:
        //   [H Aᵀ] [a]   [0]
        //   [A 0 ] [λ] = [d]
        // with H[i][j] = ∫₀¹ (u^pi)''(u^pj)'' du = pi(pi-1)pj(pj-1)/(pi+pj-3).
        let size = TERMS + n;
        let mut kkt = vec![vec![0.0; size]; size];
        for (i, row) in kkt.iter_mut().enumerate().take(TERMS) {
            for (j, cell) in row.iter_mut().enumerate().take(TERMS) {
                let (pi, pj) = (power(i) as f64, power(j) as f64);
                *cell = pi * (pi - 1.0) * pj * (pj - 1.0) / (pi + pj - 3.0);
            }
        }
        for (m, &u) in us.iter().enumerate() {
            for (j, v) in row(u).into_iter().enumerate() {
                kkt[TERMS + m][j] = v;
                kkt[j][TERMS + m] = v;
            }
        }
        let mut b = vec![0.0; TERMS];
        b.extend_from_slice(&rhs);
        let Some(x) = solve_dense(kkt, b) else {
            return false;
        };
        let scaled = &x[..TERMS];
        for (&u, d) in us.iter().zip(&rhs) {
            let got: f64 = row(u).iter().zip(scaled).map(|(p, a)| p * a).sum();
            if (got - d).abs() > 1e-9 * (1.0 + d.abs()) {
                return false;
            }
        }
        let mut coeffs = [0.0; 8];
        for (j, v) in scaled.iter().enumerate() {
            coeffs[j] = v / ha.powi(power(j));
        }
        if back {
            a.back = coeffs;
        } else {
            a.front = coeffs;
        }
        self.set_singlet_shape(id, g, a)
    }

    /// Bend one face of a singlet so its rim sits `rim_sag_mm` from its vertex
    /// (along +X): the curvature is solved for, and the conic constant and the
    /// aspheric terms stay. A rim sag equal to the aspheric part alone makes
    /// the base a plane.
    ///
    /// False (scene unchanged) for a non-finite sag, a sag no curvature
    /// reaches on the conic's principal branch (past a hemisphere, for a
    /// sphere), or a shape the singlet setters refuse.
    pub fn bend_singlet_face(&mut self, id: u64, back: bool, rim_sag_mm: f64) -> bool {
        let Some((mut g, a)) = self.singlet_shape(id) else {
            return false;
        };
        let ha = g.semi_aperture_mm;
        if !rim_sag_mm.is_finite() || ha.is_nan() || ha <= 0.0 {
            return false;
        }
        let (k, coeffs) = if back {
            (g.conic2, &a.back)
        } else {
            (g.conic1, &a.front)
        };
        // The conic's share of the rim sag: the target less the aspheric terms.
        let s = rim_sag_mm - even_asphere_sag(0.0, k, coeffs, ha);
        let r = if s.abs() < 1e-12 {
            f64::INFINITY
        } else {
            // s = c h² / (1 + √(1 − (1+k) c² h²)) inverts to
            // c = 2 s / (h² + (1+k) s²) while (1+k) s² <= h².
            if (1.0 + k) * s * s > ha * ha {
                return false;
            }
            (ha * ha + (1.0 + k) * s * s) / (2.0 * s)
        };
        if back {
            g.r2_mm = r;
        } else {
            g.r1_mm = r;
        }
        self.set_singlet_shape(id, g, a)
    }

    /// Replace a singlet body's geometry and aspheric terms together. Rejects
    /// the edit (returns false, scene unchanged) if the id is not a 4-edge
    /// revolved profile, the centre thickness or semi-aperture is not a
    /// positive number, a radius is NaN (an infinite radius is a plane), any
    /// other value is not finite, or the new shape is not revolvable (its
    /// faces cross).
    ///
    /// A thickness of zero or less is refused on its own: with the faces
    /// swapped the outline still closes without crossing, so the profile
    /// check alone would accept a lens turned inside out.
    pub fn set_singlet_shape(&mut self, id: u64, g: SingletGeom, a: SingletAspheres) -> bool {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(g.thickness_mm) || !positive(g.semi_aperture_mm) {
            return false;
        }
        if g.r1_mm.is_nan() || g.r2_mm.is_nan() || !g.conic1.is_finite() || !g.conic2.is_finite() {
            return false;
        }
        if a.front.iter().chain(&a.back).any(|v| !v.is_finite()) {
            return false;
        }
        let curves = singlet_curves(g, a);
        let Some(body) = self.bodies.iter_mut().find(|b| b.common.id.0 == id) else {
            return false;
        };
        let SolidGeometry3::Profile(solid) = &mut body.geometry else {
            return false;
        };
        if solid.construction != ProfileConstruction3::RevolveLocalX
            || solid.profile.edges.len() != 4
        {
            return false;
        }
        // Rebuild the curves, preserving each edge's patch id so a surface
        // override or stop designation keeps pointing at the right face (§3.8).
        let rebuilt = ClosedProfile2 {
            edges: solid
                .profile
                .edges
                .iter()
                .zip(curves)
                .map(|(e, curve)| ProfileEdge2 {
                    patch: e.patch,
                    curve,
                })
                .collect(),
        };
        if !rebuilt.is_finite() || validate_revolvable(&rebuilt).is_err() {
            return false;
        }
        solid.profile = rebuilt;
        true
    }

    /// Read a singlet body's geometry, if `id` is one.
    pub fn singlet_geom(&self, id: u64) -> Option<SingletGeom> {
        self.singlet_shape(id).map(|(g, _)| g)
    }

    /// Read a singlet body's geometry and aspheric terms, if `id` is one.
    pub fn singlet_shape(&self, id: u64) -> Option<(SingletGeom, SingletAspheres)> {
        let body = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        match &body.geometry {
            SolidGeometry3::Profile(p) if p.construction == ProfileConstruction3::RevolveLocalX => {
                shape_from_profile(&p.profile)
            }
            _ => None,
        }
    }

    /// A singlet's paraxial focal lengths at the d line, if `id` is a singlet
    /// with a d-line index. `None` for an afocal lens as well.
    pub fn singlet_paraxial(&self, id: u64) -> Option<SingletParaxial> {
        let (g, _) = self.singlet_shape(id)?;
        let body = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        let nd = match body.material.abbe(&self.catalog) {
            Some((nd, _)) => nd,
            None => body
                .material
                .index_at(crate::material::LAMBDA_D, &self.catalog)?,
        };
        let n0 = self.env.ambient_index;
        let t = g.thickness_mm;
        let p1 = (nd - n0) * curvature(g.r1_mm);
        let p2 = (n0 - nd) * curvature(g.r2_mm);
        let power = p1 + p2 - p1 * p2 * t / nd;
        if !power.is_finite() || power.abs() < 1e-12 {
            return None;
        }
        let efl_mm = n0 / power;
        Some(SingletParaxial {
            nd,
            efl_mm,
            bfl_mm: efl_mm * (1.0 - p1 * t / nd),
        })
    }

    /// A body's bulk glass as an Abbe point `(nd, vd)`, if it has one (an
    /// `AbbePoint` material). None for a catalog glass, mirror, or non-body id.
    pub fn body_glass_abbe(&self, id: u64) -> Option<(f64, f64)> {
        match &self.bodies.iter().find(|b| b.common.id.0 == id)?.material {
            Material::AbbePoint { nd, vd } => Some((*nd, *vd)),
            _ => None,
        }
    }

    /// Set a body's bulk glass to an Abbe point `(nd, vd)`. Returns false if `id`
    /// is not a body.
    pub fn set_body_glass_abbe(&mut self, id: u64, nd: f64, vd: f64) -> bool {
        match self.bodies.iter_mut().find(|b| b.common.id.0 == id) {
            Some(b) => {
                b.material = Material::AbbePoint { nd, vd };
                true
            }
            None => false,
        }
    }

    /// Delete any object or source by id. Returns false if nothing matched.
    pub fn delete_object(&mut self, id: u64) -> bool {
        let before = self.object_count();
        self.bodies.retain(|b| b.common.id.0 != id);
        self.surfaces.retain(|s| s.common.id.0 != id);
        self.apertures.retain(|a| a.common.id.0 != id);
        self.detectors.retain(|d| d.common.id.0 != id);
        self.sources.retain(|s| s.id.0 != id);
        self.object_count() != before
    }

    /// Duplicate a body, offset by `delta`, with fresh ids. Returns the new id, or
    /// None if `id` is not a body.
    pub fn clone_body(&mut self, id: u64, delta: Vec3) -> Option<BodyId> {
        let src = self.bodies.iter().find(|b| b.common.id.0 == id)?.clone();
        let new_id = self.alloc_object_id();
        let mut body = src;
        body.common.id = new_id;
        body.common.transform.pos += delta;
        if let SolidGeometry3::Profile(p) = &mut body.geometry {
            for e in &mut p.profile.edges {
                e.patch = self.alloc_patch_id();
            }
        }
        // Per-patch overrides referenced old patch ids; a fresh clone drops them
        // rather than dangle. Default response is preserved.
        body.surface.overrides.clear();
        self.bodies.push(body);
        Some(BodyId(new_id.0))
    }

    /// Count of all placed objects and sources — used to detect a real deletion.
    fn object_count(&self) -> usize {
        self.bodies.len()
            + self.surfaces.len()
            + self.apertures.len()
            + self.detectors.len()
            + self.sources.len()
    }

    /// The transform of whichever object or source carries `id`.
    fn object_transform_mut(&mut self, id: u64) -> Option<&mut Isometry3> {
        if let Some(b) = self.bodies.iter_mut().find(|b| b.common.id.0 == id) {
            return Some(&mut b.common.transform);
        }
        if let Some(d) = self.detectors.iter_mut().find(|d| d.common.id.0 == id) {
            return Some(&mut d.common.transform);
        }
        if let Some(s) = self.surfaces.iter_mut().find(|s| s.common.id.0 == id) {
            return Some(&mut s.common.transform);
        }
        if let Some(a) = self.apertures.iter_mut().find(|a| a.common.id.0 == id) {
            return Some(&mut a.common.transform);
        }
        if let Some(s) = self.sources.iter_mut().find(|s| s.id.0 == id) {
            return Some(&mut s.transform);
        }
        None
    }
}

/// A transform placing an object at `pos` with its local +Z aimed along `normal`
/// (the shortest-arc rotation from +Z). Falls back to a pure translation if
/// `normal` is degenerate.
fn facing(pos: Vec3, normal: Vec3) -> Isometry3 {
    let n = normal.normalize_or_zero();
    if n.length_squared() < 0.5 {
        return Isometry3::from_pos(pos);
    }
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, n);
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap_or_else(|_| Isometry3::from_pos(pos))
}

/// A surface transform's world normal (its local +Z rotated into world).
fn surface_normal(t: &Isometry3) -> Vec3 {
    let r = t.rot_xyzw;
    let q = glam::DQuat::from_xyzw(r[0], r[1], r[2], r[3]);
    let n = q * glam::DVec3::Z;
    Vec3::new(n.x, n.y, n.z)
}

/// A surface response from a short kind string, or `None` for an unknown kind.
fn response_from_kind(kind: &str) -> Option<SurfaceResponse> {
    Some(match kind {
        "mirror" => SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(0.95),
        },
        "absorb" => SurfaceResponse::Absorber,
        "clear" => SurfaceResponse::ExplicitRt {
            reflectance: SpectralScalar::Constant(0.0),
            transmittance: SpectralScalar::Constant(1.0),
        },
        "split" => split_response(0.5, 0.5)?,
        "dichroic" => DichroicSpec::default().response(),
        "refract" => SurfaceResponse::DielectricFresnel {
            ar_residual: SpectralScalar::Constant(1.0),
        },
        _ => return None,
    })
}

/// What the object list calls both halves of a polarizing beam splitter, so it
/// reads as one part even though the scene holds it as two objects.
const PBS_LABEL: &str = "polarizing beam splitter";

/// A dense flint of index 1.754 has an Abbe number around here. The cube's
/// glass index is fixed by the coating pair; its dispersion is not, and this
/// is a plausible glass rather than a measured one.
const PBS_VD: f64 = 27.5;

/// What is left of a cube face's Fresnel reflection after a good broadband AR:
/// about half a percent a face, against 7.5% bare. `ar_residual` is one number
/// and not a coating model, and at normal incidence that costs nothing,
/// because there s and p are the same thing.
const PBS_FACE_AR: f64 = 0.07;

/// The substrate a plate is made of: N-BK7, the ordinary choice.
pub const PLATE_ND: f64 = 1.5168;
pub const PLATE_VD: f64 = 64.17;

/// The substrate thickness a new plate gets, in mm.
///
/// Three millimetres is an ordinary catalogue beamsplitter: thin enough to drop
/// into a beam without hunting for space, thick enough that the walk-off it
/// gives a transmitted beam is a millimetre, which you can see rather than have
/// to be told about.
pub const DEFAULT_PLATE_MM: f64 = 3.0;

/// The two endpoints of a profile curve, for measuring a plate.
fn curve_extent(c: &Curve2) -> Vec<Vec2> {
    match c {
        Curve2::Line { a, b } => vec![*a, *b],
        Curve2::Conic(p) => vec![
            Vec2::new(p.vertex.x, p.half_ap),
            Vec2::new(p.vertex.x, -p.half_ap),
        ],
        Curve2::EvenAsphere { base, .. } => vec![
            Vec2::new(base.vertex.x, base.half_ap),
            Vec2::new(base.vertex.x, -base.half_ap),
        ],
        _ => Vec::new(),
    }
}

/// The angle an axial beam (along `+X`) meets a plate whose tilt is `tilt_deg`.
///
/// [`Scene3::add_planar_element`] points the plate normal along
/// `(cos φ, sin φ, 0)`, so the angle of incidence is `arccos|cos φ|`: 0° square
/// to the beam at 180°, 45° at the usual 135° fold, 90° edge-on at 90°. Worth
/// naming because the wrong formula is invisible at exactly 135°, where two
/// plausible ones agree.
pub fn axial_aoi_deg(tilt_deg: f64) -> f64 {
    tilt_deg.to_radians().cos().abs().clamp(0.0, 1.0).acos().to_degrees()
}

/// The short kind string for a surface response: `mirror`, `absorb`, `clear`
/// (full transmission), `split` (constant reflectance and transmittance),
/// `dichroic` (a [`DichroicSpec`] coating), `coated` (any other tabulated
/// reflectance/transmittance), `film` (a real thin-film stack), `polarizer`,
/// `retarder`, `refract`, or `thin-lens`.
pub fn response_kind(r: &SurfaceResponse) -> &'static str {
    match r {
        SurfaceResponse::Mirror { .. } => "mirror",
        SurfaceResponse::Absorber => "absorb",
        SurfaceResponse::ExplicitRt {
            reflectance: SpectralScalar::Constant(rr),
            transmittance: SpectralScalar::Constant(tt),
        } => {
            if *rr == 0.0 && *tt == 1.0 {
                "clear"
            } else {
                "split"
            }
        }
        SurfaceResponse::ExplicitRt { .. } => {
            if DichroicSpec::from_response(r).is_some() {
                "dichroic"
            } else {
                "coated"
            }
        }
        SurfaceResponse::Coating { .. } => "film",
        SurfaceResponse::Polarizer { .. } => "polarizer",
        SurfaceResponse::Retarder { .. } => "retarder",
        SurfaceResponse::DielectricFresnel { .. } => "refract",
        SurfaceResponse::ThinLens { .. } => "thin-lens",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_scene_is_valid_and_full_3d() {
        let s = Scene3::default_scene();
        s.validate().expect("default scene validates");
        assert_eq!(s.bodies.len(), 1);
        assert_eq!(s.sources.len(), 1);
        assert_eq!(s.detectors.len(), 1);
        // The lens is a real 3D body, not a planar-only migration artefact.
        assert!(s.bodies[0].is_3d());
    }

    #[test]
    fn a_singlet_round_trips_its_geometry() {
        let mut s = Scene3::new();
        let g = SingletGeom {
            r1_mm: 40.0,
            r2_mm: -60.0,
            thickness_mm: 5.0,
            semi_aperture_mm: 10.0,
            conic1: 0.0,
            conic2: -0.5,
        };
        let id = s.add_singlet(Vec3::ZERO, g, 1.5, 64.0).0;
        let read = s.singlet_geom(id).expect("is a singlet");
        assert!((read.r1_mm - 40.0).abs() < 1e-9);
        assert!((read.r2_mm - (-60.0)).abs() < 1e-9);
        assert!((read.thickness_mm - 5.0).abs() < 1e-9);
        assert!((read.semi_aperture_mm - 10.0).abs() < 1e-9);
        assert!((read.conic2 - (-0.5)).abs() < 1e-9);
        s.validate().expect("authored singlet validates");
    }

    #[test]
    fn a_plano_face_is_a_valid_singlet() {
        let mut s = Scene3::new();
        let g = SingletGeom {
            r1_mm: f64::INFINITY, // plano front
            r2_mm: -30.0,
            thickness_mm: 3.0,
            semi_aperture_mm: 8.0,
            conic1: 0.0,
            conic2: 0.0,
        };
        let id = s.add_singlet(Vec3::ZERO, g, 1.5, 64.0).0;
        s.validate().expect("plano-convex validates");
        let read = s.singlet_geom(id).unwrap();
        assert!(read.r1_mm.is_infinite());
    }

    #[test]
    fn set_singlet_geom_edits_and_preserves_patch_ids() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        let patches: Vec<u64> = match &s.bodies[0].geometry {
            SolidGeometry3::Profile(p) => p.profile.edges.iter().map(|e| e.patch.0).collect(),
            _ => unreachable!(),
        };
        let mut g = s.singlet_geom(id).unwrap();
        g.r1_mm = 30.0;
        assert!(s.set_singlet_geom(id, g));
        let after: Vec<u64> = match &s.bodies[0].geometry {
            SolidGeometry3::Profile(p) => p.profile.edges.iter().map(|e| e.patch.0).collect(),
            _ => unreachable!(),
        };
        assert_eq!(patches, after, "patch ids are stable across an edit");
        assert!((s.singlet_geom(id).unwrap().r1_mm - 30.0).abs() < 1e-9);
        s.validate().expect("edited scene validates");
    }

    #[test]
    fn a_lens_turned_inside_out_is_refused() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        let json = s.to_json();
        let g = s.singlet_geom(id).unwrap();
        for bad in [
            SingletGeom {
                thickness_mm: -4.0,
                ..g
            },
            SingletGeom {
                thickness_mm: 0.0,
                ..g
            },
            SingletGeom {
                semi_aperture_mm: 0.0,
                ..g
            },
            SingletGeom {
                r1_mm: f64::NAN,
                ..g
            },
            SingletGeom {
                conic2: f64::INFINITY,
                ..g
            },
        ] {
            assert!(!s.set_singlet_geom(id, bad), "{bad:?}");
        }
        assert_eq!(s.to_json(), json);
        // A flat face is an infinite radius, and stays allowed.
        assert!(s.set_singlet_geom(
            id,
            SingletGeom {
                r2_mm: f64::INFINITY,
                ..g
            }
        ));
    }

    #[test]
    fn a_self_crossing_edit_is_refused() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        let mut g = s.singlet_geom(id).unwrap();
        // A semi-aperture larger than |R| makes the conic sag imaginary / the
        // faces cross: the edit must be refused, the scene left untouched.
        g.r1_mm = 5.0;
        g.semi_aperture_mm = 40.0;
        let ok = s.set_singlet_geom(id, g);
        assert!(!ok, "an invalid geometry is refused");
        s.validate().expect("scene is unchanged and still valid");
    }

    #[test]
    fn delete_and_clone() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        let clone = s.clone_body(id, Vec3::new(0.0, 20.0, 0.0)).unwrap();
        assert_eq!(s.bodies.len(), 2);
        assert_ne!(clone.0, id);
        s.validate().expect("clone keeps the scene valid");
        assert!(s.delete_object(clone.0));
        assert_eq!(s.bodies.len(), 1);
        assert!(!s.delete_object(999_999), "deleting nothing returns false");
    }

    #[test]
    fn reconfigure_the_source_to_a_point_cone() {
        let mut s = Scene3::default_scene();
        assert_eq!(s.first_source_spec().unwrap().0, "collimated_rect");
        assert!(s.set_first_source(
            SpatialDistribution3::Point,
            AngularDistribution3::UniformConeSolidAngle { half_angle_rad: 0.1 },
            1,
            64,
        ));
        s.validate().expect("point source validates");
        let (kind, p0, _p1, n) = s.first_source_spec().unwrap();
        assert_eq!(kind, "point");
        assert!((p0 - 0.1_f64.to_degrees()).abs() < 1e-9);
        assert_eq!(n, 64);
    }

    #[test]
    fn draw_a_planar_mirror() {
        let mut s = Scene3::default_scene();
        let n0 = s.surfaces.len();
        let id = s
            .add_planar_mirror(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 5.0, 0.0), 0.95)
            .expect("non-degenerate segment");
        assert_eq!(s.surfaces.len(), n0 + 1);
        s.validate().expect("mirror scene validates");
        // It is a real object: movable and deletable by id.
        assert!(s.set_object_position(id, Vec3::new(2.0, 0.0, 0.0)));
        assert!(s.delete_object(id));
        assert_eq!(s.surfaces.len(), n0);
        // A zero-length segment is refused.
        assert!(s.add_planar_mirror(Vec3::ZERO, Vec3::ZERO, 1.0).is_none());
    }

    #[test]
    fn an_element_carries_per_face_responses_and_tilts() {
        let mut s = Scene3::default_scene();
        let mirror = SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(0.95),
        };
        let clear = SurfaceResponse::ExplicitRt {
            reflectance: SpectralScalar::Constant(0.0),
            transmittance: SpectralScalar::Constant(1.0),
        };
        let id = s.add_planar_element(Vec3::new(25.0, 0.0, 0.0), 8.0, 45.0, mirror, clear);
        s.validate().expect("element validates");
        assert_eq!(s.surface_response(id), Some(("mirror", "clear")));
        // The tilt round-trips through the normal.
        assert!((s.surface_tilt_z_deg(id).unwrap() - 45.0).abs() < 1e-6);
        assert!(s.set_surface_response(id, "absorb", "mirror"));
        assert_eq!(s.surface_response(id), Some(("absorb", "mirror")));
        assert!(s.set_surface_tilt_z_deg(id, -30.0));
        assert!((s.surface_tilt_z_deg(id).unwrap() - (-30.0)).abs() < 1e-6);
        // Moving keeps the tilt.
        assert!(s.set_object_position(id, Vec3::new(30.0, 4.0, 0.0)));
        assert!((s.surface_tilt_z_deg(id).unwrap() - (-30.0)).abs() < 1e-6);
        s.validate().expect("still valid after edits");
    }

    #[test]
    fn an_aperture_stops_the_beam_and_tilts() {
        let mut s = Scene3::default_scene();
        let id = s.add_aperture(Vec3::new(-20.0, 0.0, 0.0), 4.0, 0.0);
        s.validate().expect("aperture scene validates");
        assert_eq!(s.apertures.len(), 1);
        // The aperture tilts through the shared surface-tilt path and moves.
        assert!(s.set_surface_tilt_z_deg(id, 20.0));
        assert!((s.surface_tilt_z_deg(id).unwrap() - 20.0).abs() < 1e-6);
        assert!(s.set_object_position(id, Vec3::new(-18.0, 1.0, 0.0)));
        assert!(s.delete_object(id));
        assert_eq!(s.apertures.len(), 0);
    }

    #[test]
    fn a_prism_is_a_valid_extruded_glass_body() {
        let mut s = Scene3::default_scene();
        let n0 = s.bodies.len();
        let id = s.add_prism(Vec3::new(-40.0, 0.0, 0.0), 16.0, 10.0, 1.62, 36.0);
        s.validate().expect("prism scene validates");
        assert_eq!(s.bodies.len(), n0 + 1);
        // It is a real body: glass-editable, and NOT a singlet (no r1/r2).
        assert_eq!(s.body_glass_abbe(id), Some((1.62, 36.0)));
        assert!(s.singlet_geom(id).is_none());
        assert!(s.delete_object(id));
    }

    #[test]
    fn move_and_rotate() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        assert!(s.set_object_position(id, Vec3::new(1.0, 2.0, 0.0)));
        assert_eq!(s.body(BodyId(id)).unwrap().common.transform.pos, Vec3::new(1.0, 2.0, 0.0));
        assert!(s.set_object_rotation_z_deg(id, 10.0));
        s.validate().expect("moved/rotated scene validates");
        assert!(!s.set_object_position(999, Vec3::ZERO));
    }

    fn front_curve(s: &Scene3, id: u64) -> Curve2 {
        match &s
            .bodies
            .iter()
            .find(|b| b.common.id.0 == id)
            .unwrap()
            .geometry
        {
            SolidGeometry3::Profile(p) => p.profile.edges[0].curve.clone(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn aspheric_terms_round_trip_and_survive_geometry_edits() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        // A conic singlet stays a conic: no asphere variant appears unasked.
        assert!(matches!(front_curve(&s, id), Curve2::Conic(_)));
        assert_eq!(s.singlet_shape(id).unwrap().1, SingletAspheres::default());

        let mut a = SingletAspheres::default();
        a.front[0] = 2e-5;
        a.front[1] = -3e-8;
        assert!(s.set_singlet_aspheres(id, a));
        s.validate().expect("aspheric singlet validates");
        assert!(matches!(front_curve(&s, id), Curve2::EvenAsphere { .. }));
        assert_eq!(s.singlet_shape(id).unwrap().1, a);

        // A geometry edit keeps the aspheric terms and the patch ids.
        let patches = |s: &Scene3| match &s.bodies[0].geometry {
            SolidGeometry3::Profile(p) => p
                .profile
                .edges
                .iter()
                .map(|e| e.patch.0)
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        let before = patches(&s);
        let mut g = s.singlet_geom(id).unwrap();
        g.r1_mm = 40.0;
        assert!(s.set_singlet_geom(id, g));
        assert_eq!(s.singlet_shape(id).unwrap().1, a);
        assert_eq!(patches(&s), before);

        // Clearing the terms turns the face back into a plain conic.
        assert!(s.set_singlet_aspheres(id, SingletAspheres::default()));
        assert!(matches!(front_curve(&s, id), Curve2::Conic(_)));
        s.validate().expect("still valid");
    }

    #[test]
    fn a_fitted_face_passes_through_its_points() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        let (g, _) = s.singlet_shape(id).unwrap();
        let c = 1.0 / g.r1_mm;
        let base = |h: f64| even_asphere_sag(c, g.conic1, &[0.0; 8], h);
        let pts: Vec<(f64, f64)> = [3.0, 6.0, 9.0, 12.0]
            .iter()
            .zip([0.01, -0.02, 0.05, 0.12])
            .map(|(&h, d)| (h, base(h) + d))
            .collect();
        assert!(s.fit_singlet_face(id, false, &pts));
        s.validate().expect("fitted singlet validates");
        let (g2, a) = s.singlet_shape(id).unwrap();
        assert_eq!(g2, g, "the base surface is kept");
        for &(h, sag) in &pts {
            let got = even_asphere_sag(c, g.conic1, &a.front, h);
            assert!((got - sag).abs() < 1e-9, "at {h}: {got} vs {sag}");
        }
        assert_eq!(a.back, [0.0; 8], "the other face is untouched");
        // Between and beyond the points the departure stays within the
        // points' own range: no swing of several times the edit.
        let departure = |h: f64| even_asphere_sag(c, g.conic1, &a.front, h) - base(h);
        for i in 0..=120 {
            let d = departure(12.0 * i as f64 / 120.0);
            assert!(
                (-0.03..=0.125).contains(&d),
                "departure {d} at {}",
                12.0 * i as f64 / 120.0
            );
        }

        // The back face is measured from its own vertex, and fits the same way.
        assert!(s.fit_singlet_face(id, true, &[(12.0, -1.0)]));
        let (g3, a3) = s.singlet_shape(id).unwrap();
        let rim = even_asphere_sag(1.0 / g3.r2_mm, g3.conic2, &a3.back, 12.0);
        assert!((rim + 1.0).abs() < 1e-9);

        // Refusals leave the scene as it was.
        let json = s.to_json();
        assert!(
            !s.fit_singlet_face(id, false, &[(6.0, 0.4), (6.0, 0.5)]),
            "one height twice"
        );
        assert!(
            !s.fit_singlet_face(id, false, &[(12.5, 0.4)]),
            "beyond the semi-aperture"
        );
        assert!(!s.fit_singlet_face(id, false, &[(0.0, 0.0)]), "on the axis");
        assert!(!s.fit_singlet_face(id, false, &[]), "no points");
        assert!(!s.fit_singlet_face(id, false, &[(f64::NAN, 0.0)]));
        assert_eq!(s.to_json(), json);
    }

    #[test]
    fn bending_a_face_puts_its_rim_where_asked() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        let rim = |s: &Scene3, back: bool| {
            let (g, a) = s.singlet_shape(id).unwrap();
            singlet_face_sag(&g, &a, back, g.semi_aperture_mm)
        };
        // Flatter, then steeper, then a conic face with an aspheric term.
        for (back, target) in [(true, -0.5), (false, 0.8), (false, 2.0)] {
            assert!(s.bend_singlet_face(id, back, target));
            assert!((rim(&s, back) - target).abs() < 1e-9, "{back} {target}");
        }
        let (mut g, mut a) = s.singlet_shape(id).unwrap();
        g.conic1 = -0.6;
        a.front[0] = 1e-5;
        assert!(s.set_singlet_shape(id, g, a));
        assert!(s.bend_singlet_face(id, false, 1.2));
        assert!((rim(&s, false) - 1.2).abs() < 1e-9);
        let (g2, a2) = s.singlet_shape(id).unwrap();
        assert_eq!(
            (g2.conic1, a2.front),
            (-0.6, a.front),
            "conic and terms stay"
        );

        // The aspheric part alone: the base becomes a plane.
        let poly = even_asphere_sag(0.0, -0.6, &a.front, 12.0);
        assert!(s.bend_singlet_face(id, false, poly));
        assert!(s.singlet_geom(id).unwrap().r1_mm.is_infinite());

        // Past a hemisphere (12 mm rim on a sphere) and NaN are refused.
        let json = s.to_json();
        assert!(!s.bend_singlet_face(id, true, -13.0));
        assert!(!s.bend_singlet_face(id, true, f64::NAN));
        assert_eq!(s.to_json(), json);
    }

    #[test]
    fn an_asphere_that_crosses_the_other_face_is_refused() {
        let mut s = Scene3::default_scene();
        let id = s.bodies[0].common.id.0;
        let json = s.to_json();
        // Front rim pushed 20 mm towards +X: far past the back face.
        assert!(!s.fit_singlet_face(id, false, &[(12.0, 20.0)]));
        let mut a = SingletAspheres::default();
        a.back[0] = f64::INFINITY;
        assert!(!s.set_singlet_aspheres(id, a), "a non-finite term");
        assert_eq!(s.to_json(), json);
    }

    #[test]
    fn paraxial_focal_lengths_match_a_ray_transfer_matrix() {
        let mut s = Scene3::new();
        let g = SingletGeom {
            r1_mm: 40.0,
            r2_mm: -90.0,
            thickness_mm: 6.0,
            semi_aperture_mm: 10.0,
            conic1: 0.0,
            conic2: 0.0,
        };
        let id = s.add_singlet(Vec3::ZERO, g, 1.6, 40.0).0;
        let p = s.singlet_paraxial(id).expect("a focusing lens");
        // Refraction, translation, refraction on (height, n·angle).
        let n = 1.6;
        let refr = |r: f64, n1: f64, n2: f64| [[1.0, 0.0], [-(n2 - n1) / r, 1.0]];
        let tran = |t: f64, n: f64| [[1.0, t / n], [0.0, 1.0]];
        let mul = |a: [[f64; 2]; 2], b: [[f64; 2]; 2]| {
            [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                ],
            ]
        };
        let m = mul(refr(-90.0, n, 1.0), mul(tran(6.0, n), refr(40.0, 1.0, n)));
        let efl = -1.0 / m[1][0];
        let bfl = -m[0][0] / m[1][0];
        assert!((p.efl_mm - efl).abs() < 1e-9, "{} vs {efl}", p.efl_mm);
        assert!((p.bfl_mm - bfl).abs() < 1e-9, "{} vs {bfl}", p.bfl_mm);
        assert_eq!(p.nd, 1.6);

        // A flat plate has no power.
        let flat = SingletGeom {
            r1_mm: f64::INFINITY,
            r2_mm: f64::INFINITY,
            ..g
        };
        let plate = s.add_singlet(Vec3::new(30.0, 0.0, 0.0), flat, 1.5, 60.0).0;
        assert!(s.singlet_paraxial(plate).is_none());
        assert!(s.singlet_paraxial(999).is_none());
    }

    #[test]
    fn beam_splitters_and_dichroics_are_valid_surfaces() {
        let mut s = Scene3::default_scene();
        let bs = s
            .add_beam_splitter(Vec3::new(-35.0, 0.0, 0.0), 15.0, 135.0, 0.5, 0.5)
            .expect("a valid split");
        let dc = s
            .add_dichroic(
                Vec3::new(-50.0, 0.0, 0.0),
                15.0,
                135.0,
                DichroicSpec::default(),
            )
            .expect("a valid coating");
        s.validate().expect("the scene validates");
        assert_eq!(s.surface_response(bs), Some(("split", "split")));
        assert_eq!(s.surface_response(dc), Some(("dichroic", "dichroic")));
        assert!((s.surface_tilt_z_deg(bs).unwrap() - 135.0).abs() < 1e-9);

        let (front, _) = s.surface_responses(dc).unwrap();
        assert_eq!(
            DichroicSpec::from_response(front),
            Some(DichroicSpec::default())
        );

        // Refused splits add or change nothing.
        let json = s.to_json();
        assert!(s
            .add_beam_splitter(Vec3::ZERO, 5.0, 0.0, 0.7, 0.4)
            .is_none());
        assert!(s
            .add_beam_splitter(Vec3::ZERO, 5.0, 0.0, -0.1, 0.4)
            .is_none());
        assert!(!s.set_surface_split(bs, f64::NAN, 0.5));
        assert!(
            !s.set_surface_split(s.bodies[0].common.id.0, 0.5, 0.5),
            "not a surface"
        );
        let bad = DichroicSpec {
            edge_nm: f64::NAN,
            ..DichroicSpec::default()
        };
        assert!(!s.set_surface_dichroic(dc, bad));
        assert_eq!(s.to_json(), json);

        // Edits: a 30:70 split, a short-pass edge at 600 nm.
        assert!(s.set_surface_split(bs, 0.3, 0.7));
        assert_eq!(s.surface_response(bs), Some(("split", "split")));
        let sp = DichroicSpec {
            edge_nm: 600.0,
            width_nm: 40.0,
            long_pass: false,
            r_max: 0.9,
            t_max: 0.8,
        };
        assert!(s.set_surface_dichroic(dc, sp));
        let (front, back) = s.surface_responses(dc).unwrap();
        assert_eq!(DichroicSpec::from_response(front), Some(sp));
        assert_eq!(DichroicSpec::from_response(back), Some(sp));
        s.validate().expect("still valid");

        // The card's kind strings reach the new responses too.
        assert!(s.set_surface_response(bs, "dichroic", "split"));
        assert_eq!(s.surface_response(bs), Some(("dichroic", "split")));
    }

    #[test]
    fn a_dichroic_reflects_one_side_of_its_edge_and_passes_the_other() {
        let sample = |r: &SurfaceResponse, nm: f64| match r {
            SurfaceResponse::ExplicitRt {
                reflectance,
                transmittance,
            } => (
                reflectance.sample(nm / 1000.0).unwrap(),
                transmittance.sample(nm / 1000.0).unwrap(),
            ),
            _ => unreachable!(),
        };
        let lp = DichroicSpec::default().response();
        assert_eq!(sample(&lp, 420.0), (0.98, 0.0));
        assert_eq!(sample(&lp, 700.0), (0.0, 0.95));
        let (r, t) = sample(&lp, 550.0);
        assert!(
            (r - 0.49).abs() < 1e-12 && (t - 0.475).abs() < 1e-12,
            "{r} {t}"
        );
        let sp = DichroicSpec {
            long_pass: false,
            ..DichroicSpec::default()
        }
        .response();
        assert_eq!(sample(&sp, 420.0), (0.0, 0.95));
        assert_eq!(sample(&sp, 700.0), (0.98, 0.0));
        // Never more out than in, across the transition.
        for i in 0..=200 {
            let nm = 530.0 + 40.0 * i as f64 / 200.0;
            for resp in [&lp, &sp] {
                let (r, t) = sample(resp, nm);
                assert!(r + t <= 1.0 + 1e-12, "{nm}: {r} + {t}");
            }
        }
        // A table this spec did not make has no spec.
        let SurfaceResponse::ExplicitRt {
            reflectance,
            transmittance:
                SpectralScalar::Tabulated {
                    lambda_um, values, ..
                },
        } = lp
        else {
            unreachable!()
        };
        let mut nudged = values.clone();
        nudged[4] += 0.01;
        let other = SurfaceResponse::ExplicitRt {
            reflectance,
            transmittance: SpectralScalar::Tabulated {
                lambda_um,
                values: nudged,
                interpolation: Interpolation::Linear,
                out_of_range: OutOfRangePolicy::Clamp,
            },
        };
        assert_eq!(DichroicSpec::from_response(&other), None);
        assert_eq!(response_kind(&other), "coated");
    }

    #[test]
    fn a_dichroic_spec_is_brought_into_range() {
        let s = DichroicSpec {
            edge_nm: 3.0,
            width_nm: 50.0,
            long_pass: true,
            r_max: 1.5,
            t_max: -0.2,
        }
        .sanitized()
        .unwrap();
        assert_eq!(
            (s.edge_nm, s.width_nm, s.r_max, s.t_max),
            (3.0, 5.0, 1.0, 0.0)
        );
        let tiny = DichroicSpec {
            width_nm: 0.0,
            ..DichroicSpec::default()
        };
        assert_eq!(tiny.sanitized().unwrap().width_nm, 0.5);
        // Every sanitized spec makes a table the scene accepts.
        let mut scene = Scene3::default_scene();
        scene.add_dichroic(Vec3::ZERO, 5.0, 0.0, s).unwrap();
        scene.validate().expect("a sanitized coating validates");
    }
}
