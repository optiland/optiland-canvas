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

use crate::geom3::primitives::even_asphere_sag;
use crate::geom3::profile::validate_revolvable;
use crate::id::{BodyId, DetectorId, LayerId, SourceId};
use crate::material::Material;
use crate::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use crate::scene::Spectrum;
use crate::scene3::{
    AbsorptionModel, AngularDistribution3, Aperture3, ApertureInteraction, ApertureShape3, Body3,
    Detector3, DetectorGeometry3, DetectorResponse, DetectorTermination, ObjectCommon, PlanarShape2,
    SamplingSequence, Scene3, SolidGeometry3, Source3, SourceSampling, SpatialDistribution3,
    SpectralScalar, SurfaceGeometry3, SurfaceObject3, SurfaceProps3, SurfaceResponse,
};
use crate::vec::Vec2;
use crate::vec3::{Isometry3, Plane3, Vec3};

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

/// The four meridional edges of a singlet, in closed-loop order: front conic,
/// top rim line, back conic, bottom rim line. Mirror-symmetric about `y = 0`.
///
/// The rim X of each face is its sag at the semi-aperture, computed with the same
/// [`even_asphere_sag`] the geometry and [`validate_revolvable`] use, so the two
/// rim lines meet the conics exactly and the loop closes.
fn singlet_curves(g: SingletGeom) -> [Curve2; 4] {
    let ha = g.semi_aperture_mm;
    let (zf, zr) = (-g.thickness_mm / 2.0, g.thickness_mm / 2.0);
    let (c1, c2) = (curvature(g.r1_mm), curvature(g.r2_mm));
    let z8 = [0.0f64; 8];
    let ftip = zf + even_asphere_sag(c1, g.conic1, &z8, ha);
    let rtip = zr + even_asphere_sag(c2, g.conic2, &z8, ha);
    [
        Curve2::Conic(ConicParams2 {
            vertex: Vec2::new(zf, 0.0),
            axis: Vec2::X,
            c: c1,
            k: g.conic1,
            half_ap: ha,
        }),
        Curve2::Line {
            a: Vec2::new(ftip, ha),
            b: Vec2::new(rtip, ha),
        },
        Curve2::Conic(ConicParams2 {
            vertex: Vec2::new(zr, 0.0),
            axis: Vec2::X,
            c: c2,
            k: g.conic2,
            half_ap: ha,
        }),
        Curve2::Line {
            a: Vec2::new(rtip, -ha),
            b: Vec2::new(ftip, -ha),
        },
    ]
}

/// Read a singlet body's geometry back out of its profile (for the inspector).
fn geom_from_profile(profile: &ClosedProfile2) -> Option<SingletGeom> {
    let edges = &profile.edges;
    if edges.len() != 4 {
        return None;
    }
    let (front, back) = match (&edges[0].curve, &edges[2].curve) {
        (Curve2::Conic(a), Curve2::Conic(b)) => (a, b),
        _ => return None,
    };
    let radius = |c: f64| if c.abs() > 1e-12 { 1.0 / c } else { f64::INFINITY };
    Some(SingletGeom {
        r1_mm: radius(front.c),
        r2_mm: radius(back.c),
        thickness_mm: back.vertex.x - front.vertex.x,
        semi_aperture_mm: front.half_ap,
        conic1: front.k,
        conic2: back.k,
    })
}

impl Scene3 {
    /// Add a rotationally-symmetric singlet on the optical axis, its body centred
    /// at `center`, with glass `(nd, vd)`. Returns the new body id.
    pub fn add_singlet(&mut self, center: Vec3, g: SingletGeom, nd: f64, vd: f64) -> BodyId {
        let curves = singlet_curves(g);
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

    /// Add a round optical element (a flat plate) at `pos`, tilted `tilt_deg` in
    /// the meridional plane, radius `radius_mm`, with configurable front/back
    /// responses — make either face reflective, absorbing, clear, or refractive.
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

    /// Replace a singlet body's geometry. Rejects the edit (returns false, scene
    /// unchanged) if the id is not a 4-edge revolved profile or the new geometry
    /// is not revolvable — a bad card edit never corrupts the scene.
    pub fn set_singlet_geom(&mut self, id: u64, g: SingletGeom) -> bool {
        let curves = singlet_curves(g);
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
        if validate_revolvable(&rebuilt).is_err() {
            return false;
        }
        solid.profile = rebuilt;
        true
    }

    /// Read a singlet body's geometry, if `id` is one.
    pub fn singlet_geom(&self, id: u64) -> Option<SingletGeom> {
        let body = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        match &body.geometry {
            SolidGeometry3::Profile(p) if p.construction == ProfileConstruction3::RevolveLocalX => {
                geom_from_profile(&p.profile)
            }
            _ => None,
        }
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
        "refract" => SurfaceResponse::DielectricFresnel {
            ar_residual: SpectralScalar::Constant(1.0),
        },
        _ => return None,
    })
}

/// The short kind string for a surface response.
fn response_kind(r: &SurfaceResponse) -> &'static str {
    match r {
        SurfaceResponse::Mirror { .. } => "mirror",
        SurfaceResponse::Absorber => "absorb",
        SurfaceResponse::ExplicitRt { .. } => "clear",
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
}
