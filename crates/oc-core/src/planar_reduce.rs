//! Exact planar reduction: the 3D scene, projected back onto the Canvas plane
//! when — and only when — that projection changes no ray (§10.2).
//!
//! The 2D tracer is fast, and it is worth keeping. But a 2D result is only worth
//! showing if it is the *same* result the 3D reference would give in the plane.
//! This module is what earns that: it turns a [`Scene3`] into the v1
//! [`Scene`](crate::scene::Scene) the fast path consumes, and it does so only
//! after checking every condition under which the two agree. Where a condition
//! fails, it returns a typed [`ReductionIssue`] rather than a plausible-looking
//! plane — the silent approximation the whole contract exists to forbid (§3.5).
//!
//! # What reduces, and why
//!
//! The Canvas plane is `Z = 0` (`u = +X`, `v = +Y`, `normal = +Z`), and this
//! reducer only targets that plane. Reduction onto an arbitrary slice is a real
//! feature, but it is not this one: an off-axis plane needs a general in-plane
//! frame for every element, and getting that subtly wrong is exactly the kind of
//! quiet error the reducer is supposed to prevent. So a non-canonical plane is
//! refused, not approximated.
//!
//! An object reduces when its 3D behaviour, restricted to rays in the plane, is
//! identical to a v1 element's:
//!
//! - a **profile body** (revolved lens, extruded prism, or an unresolved
//!   `PlanarOnly` cross-section) whose local `XY` plane is the Canvas plane —
//!   its meridian *is* the v1 boundary, edge for edge;
//! - a **sphere** centred on the plane — its great circle;
//! - a **box** whose local `XY` plane is the Canvas plane — its rectangular
//!   cross-section.
//!
//! Everything else refuses. A zero-thickness [`SurfaceObject3`] has no closed-loop
//! v1 form; an [`Aperture3`] and an area [`Detector3`] are v2-native; a cone or a
//! tilted source launches out of the plane; volume absorption and a spectral
//! surface response have nowhere to live in the v1 tracer. Each of these is a
//! distinct, named refusal, because "cannot reduce" is a statement the UI acts on
//! (§10.3), not an error to swallow.
//!
//! # Round trip
//!
//! For a v1 document, migration ([`crate::migrate_v1`]) and reduction are
//! inverses: `reduce(migrate(v1))` reproduces the v1 scene the fast path already
//! traced, which is what makes the legacy-compatibility mode exact rather than
//! merely close (§10.3, `LegacyPlanarCompatibility`).

use crate::geom::{ConicParams, Edge};
use crate::id::{BodyId, DetectorId, SourceId};
use crate::profile2::{ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2};
use crate::scene::{
    Detector, Element, ElementId, Environment, Scene as Scene2, Source, SourceKind, SurfaceProps,
    SurfaceSpec,
};
use crate::scene3::{
    AbsorptionModel, AngularDistribution3, Body3, Detector3, DetectorGeometry3, Scene3,
    SolidGeometry3, Source3, SpatialDistribution3, SpectralScalar, SurfaceObject3, SurfaceResponse,
    TwoSidedSurfaceResponse,
};
use crate::vec::{Isometry2, Vec2};
use crate::vec3::{Isometry3, Plane3, Vec3};

/// The v1 scene the fast planar path traces. Named to match the contract (§10.2).
pub type LegacyScene2 = Scene2;

/// How the browser should preview a scene (§10.3).
///
/// The mode is chosen once per scene, shown as a non-blocking badge, and never
/// lies: a legacy planar result is never labelled full 3D.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMode {
    /// The scene reduced exactly; the fast 2D tracer is showing the true answer.
    ExactPlanarReduction,
    /// The 3D preview path is drawing the scene as it is.
    Full3dProjected,
    /// An unresolved v1 scene, traced in the plane exactly as v1 did.
    LegacyPlanarCompatibility,
}

/// A check the reducer ran and passed, recorded so a `ReductionProof` states what
/// it verified rather than merely asserting exactness (§10.2).
#[derive(Debug, Clone, PartialEq)]
pub enum ReductionCheck {
    /// The target plane is the canonical Canvas `Z = 0`.
    PlaneIsCanvasXy,
    /// A body reduced to a v1 element. `kind` names the construction.
    BodyReduced { body: BodyId, kind: &'static str },
    /// A source reduced to a v1 planar source.
    SourcePlanar { source: SourceId },
    /// A detector reduced to a v1 line probe.
    DetectorLineProbe { detector: DetectorId },
    /// No out-of-plane process (scatter, cone, tilt, volume absorption) is active.
    NoOutOfPlaneProcess,
}

/// Evidence that a 2D scene is the exact planar reduction of a 3D one (§10.2).
///
/// Not a certificate to be trusted on faith: it lists the checks that were
/// actually run. The `revision` is the source scene's, filled by the caller that
/// owns it (the reducer in `oc-core` has no revision of its own), so a proof and
/// the numbers drawn from it can be tied to one scene state (§3.3).
#[derive(Debug, Clone, PartialEq)]
pub struct ReductionProof {
    pub plane: Plane3,
    pub checks: Vec<ReductionCheck>,
    /// The source scene revision, if the caller set one. `None` from the bare
    /// [`reduce_to_planar`]; the WASM Canvas fills it in.
    pub source_revision: Option<u64>,
}

/// Why a scene does not reduce exactly onto the plane (§10.2, §17).
///
/// Every variant names the object, so a refusal reaches the inspector as
/// something a user can act on. "Cannot reduce" is a UI state, not a failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ReductionIssue {
    /// The plane is not the canonical Canvas `Z = 0`. Reduction onto an arbitrary
    /// slice is future work, not a silent best effort.
    UnsupportedPlane { reason: String },
    /// A global process the v1 tracer does not model (e.g. ambient absorption).
    UnsupportedEnvironment { reason: String },
    /// The object's placement takes it out of the plane, so an in-plane ray no
    /// longer sees the v1 cross-section (a tilted body, an extrusion that does not
    /// cross the plane, a probe with an out-of-plane endpoint).
    OutOfPlaneObject { object: u64, reason: String },
    /// A geometry with no exact v1 form on this plane.
    UnsupportedGeometry { object: u64, kind: String },
    /// A `PlanarOnly` body sitting at a plane other than the target.
    PlanarOnlyOffPlane { object: u64, plane_z_mm: f64 },
    /// A two-sided response whose sides differ; v1 has one spec per edge.
    AsymmetricSurfaceResponse { object: u64 },
    /// A surface response the v1 model cannot express (explicit R/T, a spectral
    /// table).
    UnsupportedSurfaceResponse { object: u64, kind: String },
    /// A zero-thickness surface. It has no closed-loop v1 element form, and an
    /// out-of-plane one would fold light off the plane entirely.
    ZeroThicknessSurface { object: u64, out_of_plane: bool },
    /// A finite aperture. v2-native; v1 has no aperture object.
    Aperture { object: u64 },
    /// An area detector. It has no unambiguous v1 line form without a projection
    /// choice the reducer will not make on its own.
    AreaDetector { object: u64 },
    /// A source whose distribution launches out of the plane (a cone, a
    /// Lambertian hemisphere, a tilted direction).
    NonPlanarSource { source: u64, reason: String },
}

impl std::fmt::Display for ReductionIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlane { reason } => write!(f, "unsupported reduction plane: {reason}"),
            Self::UnsupportedEnvironment { reason } => {
                write!(f, "environment does not reduce: {reason}")
            }
            Self::OutOfPlaneObject { object, reason } => {
                write!(f, "object {object} is out of the plane: {reason}")
            }
            Self::UnsupportedGeometry { object, kind } => {
                write!(f, "object {object} geometry does not reduce: {kind}")
            }
            Self::PlanarOnlyOffPlane { object, plane_z_mm } => write!(
                f,
                "object {object} is a planar-only body at z = {plane_z_mm} mm, not on the plane"
            ),
            Self::AsymmetricSurfaceResponse { object } => write!(
                f,
                "object {object} has different front/back responses; v1 has one spec per edge"
            ),
            Self::UnsupportedSurfaceResponse { object, kind } => {
                write!(
                    f,
                    "object {object} surface response does not reduce: {kind}"
                )
            }
            Self::ZeroThicknessSurface {
                object,
                out_of_plane,
            } => {
                if *out_of_plane {
                    write!(
                        f,
                        "object {object} is a zero-thickness surface whose normal folds light out of the plane"
                    )
                } else {
                    write!(
                        f,
                        "object {object} is a zero-thickness surface with no v1 element form"
                    )
                }
            }
            Self::Aperture { object } => {
                write!(f, "object {object} is a finite aperture, which v1 lacks")
            }
            Self::AreaDetector { object } => write!(
                f,
                "object {object} is an area detector with no exact v1 line form"
            ),
            Self::NonPlanarSource { source, reason } => {
                write!(f, "source {source} is not planar: {reason}")
            }
        }
    }
}

impl std::error::Error for ReductionIssue {}

/// The tolerance for calling a value on the plane, or an axis aligned. Absolute
/// in the quaternion components (which are unit-scale) and relative to magnitude
/// for positions.
const TOL: f64 = 1e-9;

/// Is `plane` the canonical Canvas `Z = 0`?
fn is_canvas_xy(plane: &Plane3) -> bool {
    plane.origin.length() <= TOL
        && (plane.u - Vec3::X).length() <= TOL
        && (plane.v - Vec3::Y).length() <= TOL
        && (plane.normal - Vec3::Z).length() <= TOL
}

/// A 3D transform as an in-plane v1 isometry, or `None` if it moves the local
/// `XY` plane off `Z = 0`.
///
/// Exact only when the rotation is about `+Z` (so `x = y = 0` in the quaternion)
/// and the translation stays in the plane. This is the inverse of
/// [`crate::migrate_v1::isometry2_to_3`], so a migrated body reduces back to the
/// isometry it came from.
fn reduce_isometry_xy(t: &Isometry3) -> Option<Isometry2> {
    let [x, y, z, w] = t.rot_xyzw;
    if t.pos.z.abs() > TOL * t.pos.length().max(1.0) {
        return None;
    }
    // A rotation about any axis other than +Z tips the local plane. The x and y
    // quaternion components are exactly the off-Z tilt.
    if x.abs() > TOL || y.abs() > TOL {
        return None;
    }
    let angle = 2.0 * z.atan2(w);
    Some(Isometry2::new(Vec2::new(t.pos.x, t.pos.y), angle))
}

fn conic2_to_1(c: &ConicParams2) -> ConicParams {
    ConicParams {
        vertex: c.vertex,
        axis: c.axis,
        c: c.c,
        k: c.k,
        half_ap: c.half_ap,
    }
}

/// A profile curve as a v1 edge. The inverse of
/// [`crate::migrate_v1`]'s `edge_to_curve`. `Polyline` becomes the degree-1
/// `Spline` it always was (§6.3); no ray changes. A [`Curve2::SagSegment`] (a
/// Fresnel facet) has no v1 form, since a v1 conic always spans its axis, and
/// comes back `None`.
fn curve2_to_edge(c: &Curve2) -> Option<Edge> {
    Some(match c {
        Curve2::Line { a, b } => Edge::Line { a: *a, b: *b },
        Curve2::Arc {
            center,
            radius_mm,
            a0,
            a1,
        } => Edge::Arc {
            center: *center,
            r: *radius_mm,
            a0: *a0,
            a1: *a1,
        },
        Curve2::Conic(p) => Edge::Conic(conic2_to_1(p)),
        Curve2::EvenAsphere { base, coeffs } => Edge::Asphere {
            base: conic2_to_1(base),
            coeffs: *coeffs,
        },
        Curve2::Polyline { points } => Edge::Spline {
            ctrl: points.clone(),
            degree: 1,
        },
        Curve2::SagSegment { .. } => return None,
    })
}

/// A constant spectral scalar's value, or `None` if it is tabulated (which v1
/// cannot express).
fn constant_scalar(s: &SpectralScalar) -> Option<f64> {
    match s {
        SpectralScalar::Constant(v) => Some(*v),
        SpectralScalar::Tabulated { .. } => None,
    }
}

/// A v2 surface response as a v1 spec. The inverse of `spec_to_response`.
fn response_to_spec(object: u64, r: &SurfaceResponse) -> Result<SurfaceSpec, ReductionIssue> {
    let unsupported = |kind: &str| ReductionIssue::UnsupportedSurfaceResponse {
        object,
        kind: kind.to_string(),
    };
    match r {
        SurfaceResponse::DielectricFresnel { ar_residual } => {
            let ar =
                constant_scalar(ar_residual).ok_or_else(|| unsupported("tabulated ar_residual"))?;
            Ok(SurfaceSpec {
                ar_residual: ar,
                ..SurfaceSpec::default()
            })
        }
        SurfaceResponse::Mirror { reflectivity } => {
            let refl = constant_scalar(reflectivity)
                .ok_or_else(|| unsupported("tabulated reflectivity"))?;
            Ok(SurfaceSpec {
                mirror: true,
                reflectivity: refl,
                ..SurfaceSpec::default()
            })
        }
        SurfaceResponse::Absorber => Ok(SurfaceSpec {
            absorb: true,
            ..SurfaceSpec::default()
        }),
        // A thin-film stack is angle-dependent by construction, and v1's
        // `SurfaceSpec` has nowhere to put an angle. Reducing one would have to
        // pick an angle and bake it in, which is exactly the mistake the stack
        // was added to stop making. Refuse instead.
        SurfaceResponse::Coating { .. } => Err(unsupported("thin-film coating")),
        // The same for a measured table: its angle axis is its meaning.
        SurfaceResponse::MeasuredTable { .. } => Err(unsupported("measured coating table")),
        // A v1 surface has no polarization state to act on, so there is
        // nothing for these to reduce to. A polarizer would become "half
        // the light", which is the right number and the wrong component.
        SurfaceResponse::Polarizer { .. } => Err(unsupported("polarizer")),
        SurfaceResponse::Retarder { .. } => Err(unsupported("retarder")),
        // v1 has no explicit reflectance/transmittance surface; the closest it
        // could do is a scaled Fresnel, which is a different physics. Refuse.
        SurfaceResponse::ExplicitRt { .. } => {
            Err(unsupported("explicit reflectance/transmittance"))
        }
        // v1 has no ideal-lens surface either; a thin lens reduced to a plain
        // plane would silently stop focusing. Refuse.
        SurfaceResponse::ThinLens { .. } => Err(unsupported("thin lens")),
        // v1 has no diffraction at all.
        SurfaceResponse::Grating { .. } => Err(unsupported("grating")),
    }
}

/// A two-sided response as one v1 spec, requiring the sides to agree.
fn two_sided_to_spec(
    object: u64,
    r: &TwoSidedSurfaceResponse,
) -> Result<SurfaceSpec, ReductionIssue> {
    // The kind first, so a response the plane cannot hold at all is refused
    // for what it is. A coated face's two sides differ only in which way the
    // stack is walked, and "asymmetric" would hide that it is a coating.
    let spec = response_to_spec(object, &r.from_outside)?;
    if r.from_outside != r.from_inside {
        return Err(ReductionIssue::AsymmetricSurfaceResponse { object });
    }
    Ok(spec)
}

/// A body's surface properties (default plus per-patch overrides) as v1
/// [`SurfaceProps`], keyed by edge index rather than patch id.
fn body_surface_props(
    body: &Body3,
    edges: &[ProfileEdge2],
) -> Result<SurfaceProps, ReductionIssue> {
    let object = body.common.id.0;
    let default = two_sided_to_spec(object, &body.surface.default)?;
    let mut overrides = Vec::new();
    for o in &body.surface.overrides {
        // Map patch id back to the edge index the v1 boundary uses.
        let idx = edges
            .iter()
            .position(|e| e.patch == o.patch)
            .ok_or_else(|| ReductionIssue::UnsupportedSurfaceResponse {
                object,
                kind: "override on a non-profile patch".to_string(),
            })?;
        overrides.push((idx, two_sided_to_spec(object, &o.response)?));
    }
    Ok(SurfaceProps { default, overrides })
}

/// The v1 boundary and construction label for a body's geometry, in local
/// coordinates. The element transform is applied by the caller.
fn body_boundary(body: &Body3) -> Result<(Vec<Edge>, &'static str), ReductionIssue> {
    let object = body.common.id.0;
    match &body.geometry {
        SolidGeometry3::Profile(p) => {
            let kind = match p.construction {
                ProfileConstruction3::PlanarOnly { plane_z_mm } => {
                    if plane_z_mm.abs() > TOL {
                        return Err(ReductionIssue::PlanarOnlyOffPlane { object, plane_z_mm });
                    }
                    "planar-only"
                }
                ProfileConstruction3::RevolveLocalX => "revolved",
                ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
                    // The plane sits at local Z = 0. An extrusion reduces only if
                    // it crosses that plane; otherwise an in-plane ray misses it.
                    if !(z_min_mm < 0.0 && z_max_mm > 0.0) {
                        return Err(ReductionIssue::OutOfPlaneObject {
                            object,
                            reason: format!(
                                "extrusion spans z ∈ [{z_min_mm}, {z_max_mm}] mm and does not cross the plane"
                            ),
                        });
                    }
                    "extruded"
                }
            };
            let boundary = p
                .profile
                .edges
                .iter()
                .map(|e| curve2_to_edge(&e.curve))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| ReductionIssue::UnsupportedGeometry {
                    object,
                    kind: "a Fresnel facet (an annular piece of a sag curve)".to_string(),
                })?;
            Ok((boundary, kind))
        }
        SolidGeometry3::Sphere { radius_mm } => {
            // A ray in the plane meets the sphere on the great circle of the
            // plane-through-centre; the transform check has already put the
            // centre on the plane. The circle is one full-turn arc at the local
            // origin (§ one-edge circle is a valid closed boundary).
            let boundary = vec![Edge::Arc {
                center: Vec2::ZERO,
                r: *radius_mm,
                a0: 0.0,
                a1: std::f64::consts::TAU,
            }];
            Ok((boundary, "sphere"))
        }
        SolidGeometry3::Box { half_extents_mm } => {
            let hx = half_extents_mm.x;
            let hy = half_extents_mm.y;
            // Local Z = 0 cuts the box (it is centred), so the cross-section is
            // the XY rectangle. CCW.
            let boundary = vec![
                Edge::Line {
                    a: Vec2::new(-hx, -hy),
                    b: Vec2::new(hx, -hy),
                },
                Edge::Line {
                    a: Vec2::new(hx, -hy),
                    b: Vec2::new(hx, hy),
                },
                Edge::Line {
                    a: Vec2::new(hx, hy),
                    b: Vec2::new(-hx, hy),
                },
                Edge::Line {
                    a: Vec2::new(-hx, hy),
                    b: Vec2::new(-hx, -hy),
                },
            ];
            Ok((boundary, "box"))
        }
        SolidGeometry3::Cylinder { .. } => Err(ReductionIssue::UnsupportedGeometry {
            object,
            kind: "cylinder (its cross-section depends on axis orientation; not yet reduced)"
                .to_string(),
        }),
        // Its section through the axis is two rectangles, and a v1 element is
        // one closed boundary.
        SolidGeometry3::Tube { .. } => Err(ReductionIssue::UnsupportedGeometry {
            object,
            kind: "tube (its section is two disjoint walls; a v1 element has one boundary)"
                .to_string(),
        }),
    }
}

fn reduce_body(body: &Body3) -> Result<(Element, &'static str), ReductionIssue> {
    let object = body.common.id.0;
    // The v1 tracer does not attenuate in glass, so a body that does cannot be
    // shown through it without dropping physics.
    if !matches!(body.absorption, AbsorptionModel::None) {
        return Err(ReductionIssue::UnsupportedGeometry {
            object,
            kind: "volume absorption".to_string(),
        });
    }
    let transform = reduce_isometry_xy(&body.common.transform).ok_or_else(|| {
        ReductionIssue::OutOfPlaneObject {
            object,
            reason: "placement tilts the body off the plane".to_string(),
        }
    })?;
    let (boundary, kind) = body_boundary(body)?;
    // Analytic solids carry no authored patches, so a per-patch override on one
    // has nothing to map onto; only profile bodies can carry overrides.
    let edges: Vec<ProfileEdge2> = match &body.geometry {
        SolidGeometry3::Profile(p) => p.profile.edges.clone(),
        _ => Vec::new(),
    };
    let surface = if edges.is_empty() && !body.surface.overrides.is_empty() {
        return Err(ReductionIssue::UnsupportedSurfaceResponse {
            object,
            kind: "per-patch override on an analytic solid".to_string(),
        });
    } else {
        body_surface_props(body, &edges)?
    };
    let element = Element {
        id: ElementId(0), // reassigned by add_element
        boundary,
        material: body.material.clone(),
        surface,
        transform,
        priority: body.priority,
        locked: body.common.locked,
        layer: crate::scene::LayerId(body.common.layer.0 as u32),
    };
    Ok((element, kind))
}

fn reduce_source(source: &Source3) -> Result<Source, ReductionIssue> {
    let sid = source.id.0;
    let transform =
        reduce_isometry_xy(&source.transform).ok_or_else(|| ReductionIssue::NonPlanarSource {
            source: sid,
            reason: "placement tilts the source off the plane".to_string(),
        })?;
    // Only the two combinations a v1 source produces reduce exactly. Everything
    // else — a cone, a Lambertian hemisphere, an area emitter, a tilted aim — is
    // a genuinely 3D distribution and is refused (this is the inverse of
    // `migrate_v1::migrate_source`).
    let (kind, rays) = match (&source.spatial, &source.angular) {
        (SpatialDistribution3::Point, AngularDistribution3::PlanarFanLegacy { half_angle_rad }) => {
            (
                SourceKind::Point {
                    half_angle: *half_angle_rad,
                },
                source.samples.angular_samples,
            )
        }
        (
            SpatialDistribution3::LineLegacy { half_width_mm },
            AngularDistribution3::Direction { local_dir },
        ) => {
            // v1 collimated light aims along the transform's +X; a different local
            // aim would not survive as a v1 Collimated.
            if local_dir.z.abs() > TOL || (local_dir.normalize_or_zero() - Vec3::X).length() > TOL {
                return Err(ReductionIssue::NonPlanarSource {
                    source: sid,
                    reason: "collimated aim is not the in-plane +X direction".to_string(),
                });
            }
            (
                SourceKind::Collimated {
                    half_width: *half_width_mm,
                },
                source.samples.spatial_samples,
            )
        }
        (SpatialDistribution3::Point, AngularDistribution3::Direction { local_dir }) => {
            if local_dir.z.abs() > TOL || (local_dir.normalize_or_zero() - Vec3::X).length() > TOL {
                return Err(ReductionIssue::NonPlanarSource {
                    source: sid,
                    reason: "point aim is not the in-plane +X direction".to_string(),
                });
            }
            (SourceKind::Point { half_angle: 0.0 }, 1)
        }
        _ => {
            return Err(ReductionIssue::NonPlanarSource {
                source: sid,
                reason: "distribution has measure out of the plane (cone, hemisphere, or area)"
                    .to_string(),
            });
        }
    };
    Ok(Source {
        kind,
        transform,
        spectrum: source.spectrum.clone(),
        flux: source.flux,
        rays: rays.max(1),
    })
}

fn reduce_detector(detector: &Detector3) -> Result<Detector, ReductionIssue> {
    let object = detector.common.id.0;
    match &detector.geometry {
        DetectorGeometry3::LineProbeLegacy {
            a_local,
            b_local,
            bins,
        } => {
            let a = detector.common.transform.transform_point(*a_local);
            let b = detector.common.transform.transform_point(*b_local);
            if a.z.abs() > TOL * a.length().max(1.0) || b.z.abs() > TOL * b.length().max(1.0) {
                return Err(ReductionIssue::OutOfPlaneObject {
                    object,
                    reason: "probe endpoints are off the plane".to_string(),
                });
            }
            Ok(Detector {
                a: Vec2::new(a.x, a.y),
                b: Vec2::new(b.x, b.y),
                bins: *bins,
            })
        }
        _ => Err(ReductionIssue::AreaDetector { object }),
    }
}

/// Reduce a 3D scene to the exact v1 planar scene the fast tracer consumes, or
/// explain precisely why it cannot (§10.2).
///
/// `plane` must be the canonical Canvas `Z = 0`. The returned [`ReductionProof`]
/// lists the checks that passed; its `source_revision` is `None`, for the caller
/// that owns the scene to fill in.
pub fn reduce_to_planar(
    scene: &Scene3,
    plane: &Plane3,
) -> Result<(LegacyScene2, ReductionProof), ReductionIssue> {
    let mut checks = vec![ReductionCheck::PlaneIsCanvasXy];

    if !is_canvas_xy(plane) {
        return Err(ReductionIssue::UnsupportedPlane {
            reason: "only the canonical Canvas Z = 0 plane is reduced".to_string(),
        });
    }

    // The v1 tracer models no ambient attenuation; a scene that does cannot be
    // shown through the fast path without dropping it.
    if !matches!(scene.env.ambient_absorption, AbsorptionModel::None) {
        return Err(ReductionIssue::UnsupportedEnvironment {
            reason: "ambient absorption has no v1 form".to_string(),
        });
    }

    // Nor any slope error: the fast path would draw the faces true.
    if !scene.figure_errors.is_empty() {
        return Err(ReductionIssue::UnsupportedEnvironment {
            reason: "slope error has no v1 form".to_string(),
        });
    }

    // v2-native object kinds have no v1 element: refuse each with its own reason.
    if let Some(s) = scene.surfaces.first() {
        return Err(zero_thickness_issue(s));
    }
    if let Some(a) = scene.apertures.first() {
        return Err(ReductionIssue::Aperture {
            object: a.common.id.0,
        });
    }

    let mut out = Scene2::new();
    out.env = Environment {
        ambient_index: scene.env.ambient_index,
        wavelengths: scene.env.wavelengths.clone(),
    };
    out.layers = scene.layers.clone();
    out.catalog = scene.catalog.clone();
    out.elements.clear();
    out.sources.clear();
    out.detectors.clear();

    for body in &scene.bodies {
        let (element, kind) = reduce_body(body)?;
        out.add_element(element);
        checks.push(ReductionCheck::BodyReduced {
            body: BodyId(body.common.id.0),
            kind,
        });
    }
    for source in &scene.sources {
        out.sources.push(reduce_source(source)?);
        checks.push(ReductionCheck::SourcePlanar { source: source.id });
    }
    for detector in &scene.detectors {
        out.detectors.push(reduce_detector(detector)?);
        checks.push(ReductionCheck::DetectorLineProbe {
            detector: DetectorId(detector.common.id.0),
        });
    }

    checks.push(ReductionCheck::NoOutOfPlaneProcess);

    Ok((
        out,
        ReductionProof {
            plane: *plane,
            checks,
            source_revision: None,
        },
    ))
}

/// A zero-thickness surface's refusal, flagged as out-of-plane when its normal
/// leaves the plane (which is the case the contract calls out, §10.2).
fn zero_thickness_issue(s: &SurfaceObject3) -> ReductionIssue {
    let object = s.common.id.0;
    // Local +Z is the surface normal (§5.4); the transform rotates it to world.
    let world_normal = s.common.transform.rotate(Vec3::Z);
    // "In plane" for a surface means its normal lies in the plane (so the sheet
    // appears edge-on as a line). A normal with a Z component folds light off it.
    let out_of_plane = world_normal.z.abs() > TOL;
    ReductionIssue::ZeroThicknessSurface {
        object,
        out_of_plane,
    }
}

/// Would this scene reduce onto the canonical plane? A convenience for the UI to
/// pick a [`PreviewMode`] without building the whole v1 scene.
pub fn reduces_exactly(scene: &Scene3) -> bool {
    reduce_to_planar(scene, &Plane3::XY).is_ok()
}

/// Pick the preview mode for a scene (§10.3).
///
/// An exact reduction wins. Otherwise, a scene that is only unresolved v1
/// geometry (`PlanarOnly` bodies, line probes, legacy sources) gets the legacy
/// compatibility badge; anything with real 3D geometry gets the projected 3D
/// path. This never labels a legacy result "full 3D".
pub fn preview_mode(scene: &Scene3) -> PreviewMode {
    if reduces_exactly(scene) {
        // A scene whose bodies are all still `PlanarOnly` reduces, but it reduces
        // because nothing has been promoted — that is the legacy path, not a
        // proof of 3D equivalence.
        let all_planar_only = !scene.bodies.is_empty() && scene.bodies.iter().all(|b| !b.is_3d());
        if all_planar_only {
            PreviewMode::LegacyPlanarCompatibility
        } else {
            PreviewMode::ExactPlanarReduction
        }
    } else {
        PreviewMode::Full3dProjected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::scene3::{
        Body3, ObjectCommon, SolidGeometry3, SurfaceProps3, TwoSidedSurfaceResponse,
    };
    use crate::vec3::Vec3;
    use std::f64::consts::FRAC_PI_2;

    fn sphere_body(scene: &mut Scene3, center: Vec3) -> &mut Body3 {
        let id = scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, crate::id::LayerId(0));
        common.transform = Isometry3::from_pos(center);
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Sphere { radius_mm: 4.0 },
            material: Material::AbbePoint { nd: 1.5, vd: 64.0 },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        scene.bodies.last_mut().unwrap()
    }

    #[test]
    fn a_z_rotation_reduces_but_an_off_axis_tilt_does_not() {
        let flat = reduce_isometry_xy(&Isometry3::from_pos(Vec3::new(3.0, -2.0, 0.0))).unwrap();
        assert!((flat.pos - Vec2::new(3.0, -2.0)).length() < 1e-12);
        assert!(flat.rot.abs() < 1e-12);

        let spun =
            reduce_isometry_xy(&Isometry3::from_axis_angle(Vec3::Z, FRAC_PI_2).unwrap()).unwrap();
        assert!((spun.rot - FRAC_PI_2).abs() < 1e-9);

        // Off the plane, or tilted about an in-plane axis: no reduction.
        assert!(reduce_isometry_xy(&Isometry3::from_pos(Vec3::new(0.0, 0.0, 1.0))).is_none());
        assert!(reduce_isometry_xy(&Isometry3::from_axis_angle(Vec3::Y, 0.3).unwrap()).is_none());
    }

    #[test]
    fn a_polyline_reduces_to_a_degree_one_spline() {
        let e = curve2_to_edge(&Curve2::Polyline {
            points: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
        })
        .expect("a polyline has a v1 form");
        match e {
            Edge::Spline { ctrl, degree } => {
                assert_eq!(degree, 1);
                assert_eq!(ctrl.len(), 3);
            }
            other => panic!("expected a degree-1 spline, got {other:?}"),
        }
    }

    #[test]
    fn each_surface_response_maps_to_its_v1_spec() {
        let d = response_to_spec(
            0,
            &SurfaceResponse::DielectricFresnel {
                ar_residual: SpectralScalar::Constant(0.02),
            },
        )
        .unwrap();
        assert!(!d.mirror && !d.absorb && (d.ar_residual - 0.02).abs() < 1e-15);

        let m = response_to_spec(
            0,
            &SurfaceResponse::Mirror {
                reflectivity: SpectralScalar::Constant(0.9),
            },
        )
        .unwrap();
        assert!(m.mirror && (m.reflectivity - 0.9).abs() < 1e-15);

        let a = response_to_spec(0, &SurfaceResponse::Absorber).unwrap();
        assert!(a.absorb);
    }

    #[test]
    fn explicit_rt_and_tabulated_responses_are_refused() {
        assert!(matches!(
            response_to_spec(
                7,
                &SurfaceResponse::ExplicitRt {
                    reflectance: SpectralScalar::Constant(0.3),
                    transmittance: SpectralScalar::Constant(0.6),
                }
            ),
            Err(ReductionIssue::UnsupportedSurfaceResponse { object: 7, .. })
        ));
        assert!(matches!(
            response_to_spec(
                7,
                &SurfaceResponse::Mirror {
                    reflectivity: SpectralScalar::Tabulated {
                        lambda_um: vec![0.4, 0.7],
                        values: vec![0.8, 0.9],
                        interpolation: crate::scene3::Interpolation::Linear,
                        out_of_range: crate::scene3::OutOfRangePolicy::Error,
                    },
                }
            ),
            Err(ReductionIssue::UnsupportedSurfaceResponse { object: 7, .. })
        ));
    }

    #[test]
    fn an_asymmetric_two_sided_response_is_refused() {
        let asymmetric = TwoSidedSurfaceResponse {
            from_outside: SurfaceResponse::Mirror {
                reflectivity: SpectralScalar::Constant(1.0),
            },
            from_inside: SurfaceResponse::Absorber,
        };
        assert!(matches!(
            two_sided_to_spec(3, &asymmetric),
            Err(ReductionIssue::AsymmetricSurfaceResponse { object: 3 })
        ));
    }

    #[test]
    fn volume_absorption_blocks_a_body() {
        let mut scene = Scene3::new();
        {
            let b = sphere_body(&mut scene, Vec3::ZERO);
            b.absorption = AbsorptionModel::Constant { alpha_per_mm: 0.1 };
        }
        assert!(matches!(
            reduce_to_planar(&scene, &Plane3::XY),
            Err(ReductionIssue::UnsupportedGeometry { .. })
        ));
    }

    #[test]
    fn a_sphere_off_the_plane_will_not_reduce() {
        let mut scene = Scene3::new();
        sphere_body(&mut scene, Vec3::new(0.0, 0.0, 2.0));
        assert!(matches!(
            reduce_to_planar(&scene, &Plane3::XY),
            Err(ReductionIssue::OutOfPlaneObject { .. })
        ));
    }

    #[test]
    fn the_proof_records_every_check_it_ran() {
        let mut scene = Scene3::new();
        sphere_body(&mut scene, Vec3::ZERO);
        scene.repair_next_id();
        scene.validate().unwrap();
        let (out, proof) = reduce_to_planar(&scene, &Plane3::XY).unwrap();
        assert_eq!(out.elements.len(), 1);
        assert_eq!(proof.source_revision, None);
        assert!(proof.checks.contains(&ReductionCheck::PlaneIsCanvasXy));
        assert!(proof.checks.contains(&ReductionCheck::NoOutOfPlaneProcess));
        assert!(proof
            .checks
            .iter()
            .any(|c| matches!(c, ReductionCheck::BodyReduced { kind: "sphere", .. })));
        assert!(reduces_exactly(&scene));
        assert_eq!(preview_mode(&scene), PreviewMode::ExactPlanarReduction);
    }

    #[test]
    fn ambient_absorption_blocks_the_whole_scene() {
        let mut scene = Scene3::new();
        scene.env.ambient_absorption = AbsorptionModel::Constant { alpha_per_mm: 0.01 };
        assert!(matches!(
            reduce_to_planar(&scene, &Plane3::XY),
            Err(ReductionIssue::UnsupportedEnvironment { .. })
        ));
    }
}
