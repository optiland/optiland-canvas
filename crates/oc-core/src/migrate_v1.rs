//! `.ocanvas` v1 → v2.
//!
//! Every v1 document loads, and **nothing gains a dimension it did not have**.
//! That is the whole design, and it is why this module is boring: the migration
//! is a translation, not an interpretation.
//!
//! A v1 element is a closed 2D loop. It does not record whether it is a revolved
//! lens or an extruded prism, the two trace differently, and no amount of
//! reading the file will say which the author meant. So every migrated body
//! becomes [`ProfileConstruction3::PlanarOnly`], every migrated detector a
//! [`DetectorGeometry3::LineProbeLegacy`], every migrated source a degenerate
//! planar distribution, and the report says so. They display and trace in the
//! planar compatibility path exactly as before. A full-3D trace refuses them
//! until somebody promotes them on purpose (§3.2, §11.2).
//!
//! The alternative, guessing, is worse than the cliff. A lens revolved when it
//! should have been extruded is not a slightly wrong answer; it is a different
//! object that still renders beautifully.
//!
//! # What is exact
//!
//! Materials, wavelengths, layers, and catalogue values are copied, not
//! converted, so no f64 moves. Geometry is copied edge for edge. The one v1
//! field that does not survive is `SurfaceSpec::roughness`, which the tracer
//! never read (see [`MigrationReport`]).

use crate::id::{LayerId, ObjectId, SourceId};
use crate::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use crate::scene::{Detector, Element, Scene, Source, SourceKind, SurfaceSpec};
use crate::scene3::{
    DetectorMeasure,
    AbsorptionModel, AngularDistribution3, Body3, Detector3, DetectorGeometry3, DetectorResponse,
    DetectorTermination, Environment3, ObjectCommon, PatchResponseOverride, Scene3, SceneError,
    SceneIntent, SolidGeometry3, Source3, SourceSampling, SpatialDistribution3, SpectralScalar,
    SurfaceProps3, SurfaceResponse, TwoSidedSurfaceResponse,
};
use crate::vec::{Isometry2, Vec2};
use crate::vec3::{Isometry3, Vec3};

/// What a migration could not carry, and what it deliberately refused to invent.
///
/// Not a warning list to be dismissed. It is the record of what a v1 document
/// never said, and every entry is a question only the author can answer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MigrationReport {
    /// Bodies whose third dimension is unresolved. They are `PlanarOnly`.
    pub planar_only_bodies: Vec<ObjectId>,
    /// Detectors that became line probes. A line has no area, so these report no
    /// flux density.
    pub line_probe_detectors: Vec<ObjectId>,
    /// Sources with a degenerate 3D distribution: a fan with in-plane measure, or
    /// a line with no area.
    pub legacy_sources: Vec<SourceId>,
    /// Fields that were dropped because nothing read them.
    pub dropped: Vec<String>,
    /// Free-text notes for the UI.
    pub notes: Vec<String>,
}

impl MigrationReport {
    /// Did anything land in a state that blocks a full-3D trace?
    pub fn has_unresolved_dimension(&self) -> bool {
        !self.planar_only_bodies.is_empty()
            || !self.line_probe_detectors.is_empty()
            || !self.legacy_sources.is_empty()
    }
}

/// A v1 `Isometry2` as a 3D transform: translate in `Z = 0`, rotate about `+Z`.
///
/// Exact. The v1 plane is the `Z = 0` slice of the canonical frame, and a v1
/// in-plane rotation is a rotation about the axis normal to it, so this adds a
/// coordinate rather than moving one.
pub fn isometry2_to_3(t: Isometry2) -> Isometry3 {
    // Built from the half-angle directly rather than through a conversion, so a
    // zero rotation stays exactly identity.
    let h = 0.5 * t.rot;
    Isometry3::new(
        Vec3::new(t.pos.x, t.pos.y, 0.0),
        [0.0, 0.0, h.sin(), h.cos()],
    )
    .expect("a finite v1 transform is a valid 3D one")
}

fn edge_to_curve(e: &crate::geom::Edge) -> Curve2 {
    use crate::geom::Edge;
    let conic = |c: &crate::geom::ConicParams| ConicParams2 {
        vertex: c.vertex,
        axis: c.axis,
        c: c.c,
        k: c.k,
        half_ap: c.half_ap,
    };
    match e {
        Edge::Line { a, b } => Curve2::Line { a: *a, b: *b },
        Edge::Arc { center, r, a0, a1 } => Curve2::Arc {
            center: *center,
            radius_mm: *r,
            a0: *a0,
            a1: *a1,
        },
        Edge::Conic(c) => Curve2::Conic(conic(c)),
        Edge::Asphere { base, coeffs } => Curve2::EvenAsphere {
            base: conic(base),
            coeffs: *coeffs,
        },
        // v1's `Spline` is documented as "evaluated as a polyline through the
        // control points", so this changes no ray. `degree` is dropped because
        // nothing ever read it, and carrying it would keep implying a curve the
        // tracer does not compute (§6.3).
        Edge::Spline { ctrl, .. } => Curve2::Polyline {
            points: ctrl.clone(),
        },
    }
}

/// A v1 `SurfaceSpec` as a v2 response (§6.7's table).
///
/// v1 has one spec per edge and no notion of side, so both sides get the same
/// response. That is not an assumption about coatings: it is what the v1
/// document said, restated without adding to it.
fn spec_to_response(s: &SurfaceSpec) -> SurfaceResponse {
    if s.absorb {
        SurfaceResponse::Absorber
    } else if s.mirror {
        SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(s.reflectivity),
        }
    } else {
        SurfaceResponse::DielectricFresnel {
            ar_residual: SpectralScalar::Constant(s.ar_residual),
        }
    }
}

fn spec_to_two_sided(s: &SurfaceSpec) -> TwoSidedSurfaceResponse {
    let r = spec_to_response(s);
    TwoSidedSurfaceResponse {
        from_outside: r.clone(),
        from_inside: r,
    }
}

fn migrate_element(
    e: &Element,
    scene3: &mut Scene3,
    report: &mut MigrationReport,
) -> Result<Body3, SceneError> {
    let id = scene3.alloc_object_id();

    // One patch id per v1 edge, allocated in edge order. Deterministic, so the
    // same v1 file always migrates to the same v2 ids (§5.3).
    let edges: Vec<ProfileEdge2> = e
        .boundary
        .iter()
        .map(|edge| ProfileEdge2 {
            patch: scene3.alloc_patch_id(),
            curve: edge_to_curve(edge),
        })
        .collect();

    // v1 overrides are keyed by edge *index*; v2 by patch id. This is the
    // translation that stable ids exist to make once and never again.
    let mut overrides = Vec::new();
    for (idx, spec) in &e.surface.overrides {
        match edges.get(*idx) {
            Some(pe) => overrides.push(PatchResponseOverride {
                patch: pe.patch,
                response: spec_to_two_sided(spec),
            }),
            // A v1 override pointing past the end of the boundary was already
            // dead: `SurfaceProps::for_edge` would never have found it. Dropping
            // it is the migration recording that, not losing anything.
            None => report.dropped.push(format!(
                "element {}: surface override for edge {idx}, which the boundary does not have",
                e.id.0
            )),
        }
        if spec.roughness != 0.0 {
            report.dropped.push(format!(
                "element {}: edge {idx} roughness {} (v1 reserved it for scatter and never read it)",
                e.id.0, spec.roughness
            ));
        }
    }
    if e.surface.default.roughness != 0.0 {
        report.dropped.push(format!(
            "element {}: default roughness {} (v1 reserved it for scatter and never read it)",
            e.id.0, e.surface.default.roughness
        ));
    }

    report.planar_only_bodies.push(id);

    Ok(Body3 {
        common: ObjectCommon {
            id,
            transform: isometry2_to_3(e.transform),
            layer: LayerId(e.layer.0 as u64),
            locked: e.locked,
            label: String::new(),
            visible: true,
        },
        geometry: SolidGeometry3::Profile(ProfileSolid3 {
            profile: ClosedProfile2 { edges },
            // The whole point. A v1 element is a cross-section with no recorded
            // construction, so it says so.
            construction: ProfileConstruction3::PlanarOnly { plane_z_mm: 0.0 },
        }),
        material: e.material.clone(),
        // v1 wired absorption but resolved it to zero, so glass did not
        // attenuate. `None` is that behaviour stated honestly, not a loss.
        absorption: AbsorptionModel::None,
        surface: SurfaceProps3 {
            default: spec_to_two_sided(&e.surface.default),
            overrides,
        },
        priority: e.priority,
    })
}

fn migrate_source(s: &Source, scene3: &mut Scene3, report: &mut MigrationReport) -> Source3 {
    let id = scene3.alloc_source_id();
    report.legacy_sources.push(id);
    let (spatial, angular) = match s.kind {
        // A v1 point fan is an *equal-angle* fan in one plane. It is not a cone
        // and it is not uniform in solid angle, so calling it either would change
        // the radiometry of every existing scene.
        SourceKind::Point { half_angle } => (
            SpatialDistribution3::Point,
            AngularDistribution3::PlanarFanLegacy {
                half_angle_rad: half_angle,
            },
        ),
        // A v1 collimated source is a line, not a rectangle: it has no height,
        // so it has no area.
        SourceKind::Collimated { half_width } => (
            SpatialDistribution3::LineLegacy {
                half_width_mm: half_width,
            },
            AngularDistribution3::Direction { local_dir: Vec3::X },
        ),
    };
    Source3 {
        id,
        transform: isometry2_to_3(s.transform),
        spatial,
        angular,
        spectrum: s.spectrum.clone(),
        flux: s.flux,
        samples: SourceSampling {
            // v1 spreads `rays` across the fan or the aperture, which is the
            // angular axis for a point and the spatial axis for a collimated
            // source. Preserved on whichever axis v1 used.
            spatial_samples: match s.kind {
                SourceKind::Point { .. } => 1,
                SourceKind::Collimated { .. } => s.rays.max(1),
            },
            angular_samples: match s.kind {
                SourceKind::Point { .. } => s.rays.max(1),
                SourceKind::Collimated { .. } => 1,
            },
            seed: 0,
            sequence: crate::scene3::SamplingSequence::Grid,
        },
    }
}

fn migrate_detector(d: &Detector, scene3: &mut Scene3, report: &mut MigrationReport) -> Detector3 {
    let id = scene3.alloc_object_id();
    report.line_probe_detectors.push(id);
    // The endpoints are world-space in v1 and there is no transform to fold, so
    // the object sits at the origin and the segment stays where it was, exactly.
    Detector3 {
        common: ObjectCommon {
            id,
            transform: Isometry3::IDENTITY,
            layer: LayerId(0),
            locked: false,
            label: String::new(),
            visible: true,
        },
        geometry: DetectorGeometry3::LineProbeLegacy {
            a_local: Vec3::new(d.a.x, d.a.y, 0.0),
            b_local: Vec3::new(d.b.x, d.b.y, 0.0),
            bins: d.bins.max(1),
        },
        // A line probe bins along itself; the second axis does not exist. Both
        // are set so the field is never read as an area, and `has_area()` is what
        // decides that.
        bins: [d.bins.max(1), 1],
        response: DetectorResponse::Unity,
        termination: DetectorTermination::Terminate,
        // A v1 detector measured flux and nothing else, so it keeps measuring
        // flux. A line has no area to divide by in any case.
        measure: DetectorMeasure::Power,
    }
}

/// Migrate a parsed v1 scene to v2.
pub fn migrate_v1_to_v2(v1: &Scene) -> Result<(Scene3, MigrationReport), SceneError> {
    let mut report = MigrationReport::default();
    let mut s3 = Scene3::new();

    // Copied wholesale: no value is recomputed, so nothing moves.
    s3.layers = v1.layers.clone();
    s3.catalog = v1.catalog.clone();
    s3.env = Environment3 {
        ambient_index: v1.env.ambient_index,
        ambient_absorption: AbsorptionModel::None,
        wavelengths: v1.env.wavelengths.clone(),
    };
    s3.intent = SceneIntent::default();

    // Order matters and is fixed: elements, then sources, then detectors, each in
    // v1 vector order. That is what makes the id assignment deterministic, so the
    // same v1 file migrates byte-for-byte the same way every time (§11.5).
    for e in &v1.elements {
        let b = migrate_element(e, &mut s3, &mut report)?;
        s3.bodies.push(b);
    }
    for s in &v1.sources {
        let src = migrate_source(s, &mut s3, &mut report);
        s3.sources.push(src);
    }
    for d in &v1.detectors {
        let det = migrate_detector(d, &mut s3, &mut report);
        s3.detectors.push(det);
    }

    if report.has_unresolved_dimension() {
        report.notes.push(
            "The third dimension is unresolved. A v1 document does not record whether a \
             profile is revolved or extruded, so every body loaded as PlanarOnly, every \
             detector as a line probe with no area, and every source as a degenerate planar \
             distribution. They trace exactly as they did in v1. Promote each body \
             explicitly to trace in full 3D."
                .to_string(),
        );
    }

    s3.repair_next_id();
    s3.validate()?;
    Ok((s3, report))
}

/// What a document turned out to be.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadReport {
    /// It was v2, read and migrated to v3 in memory.
    V2,
    /// It was v3.
    V3,
    /// It was v1 and has been migrated.
    MigratedFromV1(MigrationReport),
}

/// Load a `.ocanvas` document of either version.
///
/// The header decides: v1 goes through the migration layer, v2 parses directly,
/// and anything else is refused rather than guessed at (§11.2).
pub fn load_any(json: &str) -> Result<(Scene3, LoadReport), SceneError> {
    #[derive(serde::Deserialize)]
    struct Header {
        #[serde(default = "v1_format")]
        format: String,
        #[serde(default = "v1_version")]
        version: u32,
    }
    fn v1_format() -> String {
        crate::scene::OCANVAS_FORMAT.to_string()
    }
    fn v1_version() -> u32 {
        1
    }

    let h: Header = serde_json::from_str(json).map_err(|e| SceneError::Json(e.to_string()))?;
    if h.format != crate::scene::OCANVAS_FORMAT {
        return Err(SceneError::WrongFormat { found: h.format });
    }
    match h.version {
        1 => {
            let v1 = Scene::from_json(json).map_err(|e| SceneError::Json(e.to_string()))?;
            let (s3, report) = migrate_v1_to_v2(&v1)?;
            Ok((s3, LoadReport::MigratedFromV1(report)))
        }
        2 => Ok((Scene3::from_json(json)?, LoadReport::V2)),
        3 => Ok((Scene3::from_json(json)?, LoadReport::V3)),
        other => Err(SceneError::UnsupportedVersion { found: other }),
    }
}

/// The v1 detector segment a legacy probe came from, for the compatibility path.
pub fn line_probe_segment(d: &Detector3) -> Option<(Vec2, Vec2, usize)> {
    match &d.geometry {
        DetectorGeometry3::LineProbeLegacy {
            a_local,
            b_local,
            bins,
        } => {
            let a = d.common.transform.transform_point(*a_local);
            let b = d.common.transform.transform_point(*b_local);
            Some((Vec2::new(a.x, a.y), Vec2::new(b.x, b.y), *bins))
        }
        _ => None,
    }
}
