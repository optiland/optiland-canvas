//! The sequential export of a 3D scene: one traced path, written as the
//! rotationally symmetric prescription ([`SeqSystem`]) an Optiland `Optic` is
//! built from.
//!
//! **The trace decides the order, the scene decides the geometry.** The
//! source's axial ray is traced to the detector ([`extract_sequence3`]), and
//! the surfaces it meets, in the order it meets them, are the sequence. Each
//! hit is then resolved back to the scene object it landed on for its radius,
//! conic, aspheric terms, clear aperture and the medium behind it. Nothing
//! here sorts objects by their position along some world axis, which would be
//! wrong the moment a path folds.
//!
//! **Stops are metadata, and apertures are places.** A ray down the axis goes
//! through an aperture's opening without an event, so the trace never records
//! the aperture a design is stopped by. Every aperture the path crosses is
//! placed where the path crosses its plane, as a flat surface with the
//! opening's radius, and the one the scene declares as this path's stop is
//! marked as the stop. A scene that declares none gets `stop: None`, and the
//! caller decides whether to ask for one or to infer it out loud
//! (`optiland_canvas.to_optic`); nothing here picks surface zero.
//!
//! **What this covers.** A coaxial chain of revolved singlets and circular
//! apertures, lit by a collimated beam down the axis or a point source's cone.
//! Everything else on the path is refused with the object it concerns and the
//! reason: a fold, a zero-thickness surface, a tilted or decentred element, a
//! central obstruction, a non-circular opening, an off-axis beam, a Lambertian
//! emitter. Every reason found is reported, not only the first. What the
//! prescription can hold only approximately (a rectangular beam's pupil, a
//! coating the prescription has no field for) goes in the notes, which
//! `strict` export refuses and default export warns about.

use oc_core::geom3::{Containment, Intersect3, ToleranceContext};
use oc_core::id::{ApertureId, DetectorId, ObjectId, SourceId};
use oc_core::profile2::{Curve2, ProfileConstruction3};
use oc_core::scene3::{
    AbsorptionModel, AngularDistribution3, Body3, PlanarShape2, Scene3, SolidGeometry3,
    SpatialDistribution3, SpectralScalar, SurfaceResponse,
};
use oc_core::seq::{FieldType, SeqAperture, SeqMaterial, SeqSurface, SeqSystem};
use oc_core::seq3::{SeqAction, SeqHitKind};
use oc_core::vec3::Vec3;

use crate::sequence3::{extract_sequence3, SequenceError};

/// How far off the path's axis a point may sit and still be on it, mm.
const ON_AXIS_MM: f64 = 1e-6;
/// How far from the path's axis a direction may turn and still be along it,
/// as `|a × d|` (the sine of the angle).
const ALONG_AXIS: f64 = 1e-9;
/// The d line, which breaks a tie between equally weighted spectral lines for
/// the primary wavelength.
const LAMBDA_D_UM: f64 = oc_core::material::LAMBDA_D;

/// Why a scene has no sequential form.
#[derive(Debug, Clone, PartialEq)]
pub enum ExportError {
    /// Which path to export is not settled: the scene has more than one
    /// source or detector and no single declared path.
    WhichPath(String),
    /// The path itself could not be found.
    Path(SequenceError),
    /// Content on the path the prescription cannot hold, one reason per item.
    Unsupported(Vec<String>),
    /// The scene's stop declarations contradict each other or the path.
    Stop(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WhichPath(why) => write!(f, "which path to export is not settled: {why}"),
            Self::Path(e) => write!(f, "the path has no sequence: {e}"),
            Self::Unsupported(reasons) => {
                write!(f, "the path has content a sequential system cannot hold:")?;
                for r in reasons {
                    write!(f, "\n  - {r}")?;
                }
                Ok(())
            }
            Self::Stop(why) => write!(f, "the aperture stop is not settled: {why}"),
        }
    }
}

impl std::error::Error for ExportError {}

/// The path a scene exports when the caller names none: its one declared
/// primary path, or else its one source and one area detector.
pub fn default_path(scene: &Scene3) -> Result<(SourceId, DetectorId), ExportError> {
    if let [only] = scene.intent.primary_paths.as_slice() {
        return Ok((only.source, only.detector));
    }
    if scene.intent.primary_paths.len() > 1 {
        return Err(ExportError::WhichPath(format!(
            "it declares {} primary paths; name the source and detector",
            scene.intent.primary_paths.len()
        )));
    }
    let areas: Vec<_> = scene
        .detectors
        .iter()
        .filter(|d| d.geometry.has_area())
        .collect();
    match (scene.sources.as_slice(), areas.as_slice()) {
        ([s], [d]) => Ok((s.id, d.id())),
        (sources, detectors) => Err(ExportError::WhichPath(format!(
            "it has {} sources and {} area detectors and declares no primary path; \
             name the source and detector",
            sources.len(),
            detectors.len()
        ))),
    }
}

/// One place on the axis where the prescription has a surface.
struct Station {
    /// Distance along the axis from the source, mm.
    s: f64,
    surface: SeqSurface,
    /// The aperture this station stands for, if it is one. Its medium is the
    /// one it sits in, filled in once the stations are in order.
    aperture: Option<ApertureId>,
}

/// Export the path from `source` to `detector`. See the module docs.
pub fn export_seq(
    scene: &Scene3,
    source: SourceId,
    detector: DetectorId,
) -> Result<SeqSystem, ExportError> {
    let chain = extract_sequence3(scene, source, detector, 0).map_err(ExportError::Path)?;
    let o = chain.launch.point;
    let d = chain.launch.dir;
    let axial = |p: Vec3| (p - o).dot(d);
    let off_axis = |p: Vec3| ((p - o) - d * axial(p)).length();
    let lambda = chain.launch.lambda_um;

    let mut reasons: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut stations: Vec<Station> = Vec::new();
    // Media are read the way the tracer reads them, by what contains a point
    // just past each surface (§9.4), so the step is the tracer's.
    let tol = ToleranceContext::for_scene_scale(axial(chain.terminal.point).abs());
    let step = 8.0 * tol.medium_abs_mm;
    let mut noted_absorption: Vec<ObjectId> = Vec::new();
    let ambient = ambient_material(scene);

    for hit in &chain.hits {
        let who = describe(scene, hit.object);
        if hit.kind == SeqHitKind::Surface {
            reasons.push(format!(
                "{who}, {}, is on the path; this export takes lens bodies only",
                surface_kind(scene, hit.object)
            ));
            continue;
        }
        match hit.action {
            SeqAction::Transmit => {}
            SeqAction::Reflect | SeqAction::Tir => {
                reasons.push(format!(
                    "the path reflects at {who}; a fold needs coordinate breaks, which this export \
                     does not write yet"
                ));
                continue;
            }
            SeqAction::ThinLens => {
                reasons.push(format!(
                    "{who} is an ideal thin lens, which has no surfaces to export"
                ));
                continue;
            }
        }
        if hit.split {
            reasons.push(format!("the path divides at {who}"));
        }
        let Some(body) = scene.bodies.iter().find(|b| b.common.id.0 == hit.object.0) else {
            reasons.push(format!("{who} is not a body in the scene"));
            continue;
        };
        if has_sag_segments(body) {
            reasons.push(format!(
                "{who} is a Fresnel lens: its facets are annular pieces of a surface, and a sequential system has one surface per face"
            ));
            continue;
        }
        let Some(face) = revolved_face(body, hit.patch.0) else {
            reasons.push(format!(
                "the path meets {who} on a patch that is not a revolved optical face"
            ));
            continue;
        };

        // Coaxial: the body's own axis along the path's, and its origin on it.
        let axis_w = body.common.transform.rotate(Vec3::X);
        if axis_w.cross(d).length() > ALONG_AXIS {
            let deg = axis_w.dot(d).abs().clamp(0.0, 1.0).acos().to_degrees();
            reasons.push(format!("{who} is tilted {deg:.6}° from the path's axis"));
            continue;
        }
        let origin_w = body.common.transform.transform_point(Vec3::ZERO);
        if off_axis(origin_w) > ON_AXIS_MM {
            reasons.push(format!(
                "{who} is decentred {:.6} mm from the path's axis",
                off_axis(origin_w)
            ));
            continue;
        }
        let vertex_w = body
            .common
            .transform
            .transform_point(Vec3::new(face.vertex_x, 0.0, 0.0));
        if (hit.point - vertex_w).length() > ON_AXIS_MM {
            reasons.push(format!(
                "the axial ray meets {who} away from the face's vertex"
            ));
            continue;
        }

        // The face in the path's forward frame: the body may face either way.
        let sigma = axis_w.dot(d).signum();
        let c = sigma * face.c;
        let coefficients = trim(face.coeffs.iter().map(|a| sigma * a).collect());

        // The medium behind the surface: the body that holds the path just
        // past it. A cemented face is one surface between two bodies, and the
        // trace meets it once, on either body's face; what lies past it is
        // the other body either way.
        let past = hit.point + d * step;
        let entering =
            body.geometry.contains(past, &body.common.transform, &tol) == Containment::Inside;
        let behind = medium_body(scene, past, &tol);
        if let Some(b) = behind {
            if !matches!(b.absorption, AbsorptionModel::None)
                && !noted_absorption.contains(&b.common.id)
            {
                noted_absorption.push(b.common.id);
                notes.push(format!(
                    "{}: its bulk absorption is not carried; the sequential glass is clear",
                    describe(scene, b.common.id)
                ));
            }
        }
        let (material, n_behind) = match behind {
            None => ambient.clone(),
            Some(b) => match SeqMaterial::from_material(&b.material, &scene.catalog) {
                Some(m) => (m, b.material.index_at(lambda, &scene.catalog)),
                None => {
                    reasons.push(format!(
                        "{} is a material with no sequential form",
                        describe(scene, b.common.id)
                    ));
                    continue;
                }
            },
        };
        match n_behind {
            Some(n) if (n - hit.n_after).abs() <= 1e-9 => {}
            _ => {
                reasons.push(format!(
                    "the medium behind {who} could not be resolved: the trace saw index {:.9}",
                    hit.n_after
                ));
                continue;
            }
        }

        let response = body.surface.for_patch(hit.patch);
        let seen = if entering {
            &response.from_outside
        } else {
            &response.from_inside
        };
        match face_fidelity(seen) {
            FaceFidelity::Exact => {}
            FaceFidelity::Dropped(what) => notes.push(format!(
                "{who}, {} face: {what} is not carried; the sequential surface is bare",
                if hit.patch.0 == 0 { "front" } else { "back" }
            )),
            FaceFidelity::Refused(what) => {
                reasons.push(format!("{who} has {what} on a face the path crosses"));
                continue;
            }
        }

        stations.push(Station {
            s: axial(vertex_w),
            surface: SeqSurface {
                radius: (c.abs() > 1e-15).then(|| 1.0 / c),
                conic: face.k,
                thickness: 0.0,
                semi_aperture: face.half_ap,
                coefficients,
                material,
                reflective: false,
            },
            aperture: None,
        });
    }

    // The image plane, square to the axis.
    let s_image = axial(chain.terminal.point);
    if let Some(det) = scene.detectors.iter().find(|x| x.id() == detector) {
        let normal = det.common.transform.rotate(Vec3::Z);
        if normal.cross(d).length() > ALONG_AXIS {
            let deg = normal.dot(d).abs().clamp(0.0, 1.0).acos().to_degrees();
            reasons.push(format!(
                "the detector is tilted {deg:.6}° from square to the axis; a sequential image \
                 plane is square to it"
            ));
        }
    }

    // Every aperture the path crosses between the source and the image.
    for a in &scene.apertures {
        let plane = &a.shape.plane_local;
        let origin_w = a.common.transform.transform_point(plane.origin);
        let normal_w = a.common.transform.rotate(plane.normal);
        let denom = normal_w.dot(d);
        if denom.abs() < 1e-12 {
            continue;
        }
        let t = (origin_w - o).dot(normal_w) / denom;
        if t <= 0.0 || t >= s_image {
            continue;
        }
        let at = o + d * t;
        let local = a.common.transform.inverse_transform_point(at);
        let uv = plane.to_local(local);
        let contains =
            |shape: &PlanarShape2| oc_core::geom3::primitives::planar_shape_contains(shape, uv);
        if !contains(&a.shape.clear_opening) {
            // Outside the opening the path either met the mask, in which case
            // it never reached the detector, or passed the plate entirely.
            continue;
        }
        let who = describe(scene, a.common.id);
        if normal_w.cross(d).length() > ALONG_AXIS {
            reasons.push(format!("aperture {who} is tilted to the path's axis"));
            continue;
        }
        if off_axis(origin_w) > ON_AXIS_MM {
            reasons.push(format!(
                "aperture {who} is decentred {:.6} mm from the path's axis",
                off_axis(origin_w)
            ));
            continue;
        }
        let semi = match a.shape.clear_opening {
            PlanarShape2::Disc { radius_mm } => radius_mm,
            PlanarShape2::Annulus { .. } => {
                reasons.push(format!(
                    "aperture {who} has a central obstruction, which a sequential stop cannot hold"
                ));
                continue;
            }
            PlanarShape2::Rectangle { .. } | PlanarShape2::Polygon { .. } => {
                reasons.push(format!("aperture {who} is not circular"));
                continue;
            }
        };
        stations.push(Station {
            s: t,
            surface: SeqSurface {
                radius: None,
                conic: 0.0,
                thickness: 0.0,
                semi_aperture: semi,
                coefficients: Vec::new(),
                material: SeqMaterial::Air,
                reflective: false,
            },
            aperture: Some(a.id()),
        });
    }

    // In order along the axis. The lens faces already are, being the path's
    // own; the apertures fall between them. An aperture sits in whatever the
    // surface before it left the path in.
    stations.sort_by(|x, y| x.s.total_cmp(&y.s));
    // A stop on a lens face's vertex ties with the face, and goes on the
    // face's air side: in front of a front face, behind a back face, where a
    // stop drawn on a lens is. The sort left it behind the face.
    let tie = 1e-9 * s_image.abs().max(1.0);
    for i in 1..stations.len() {
        let (face, stop) = (&stations[i - 1], &stations[i]);
        if stop.aperture.is_none() || face.aperture.is_some() || (stop.s - face.s).abs() > tie {
            continue;
        }
        let before = stations[..i - 1]
            .iter()
            .rev()
            .find(|st| st.aperture.is_none())
            .map_or(&ambient.0, |st| &st.surface.material);
        if *before == ambient.0 {
            stations.swap(i - 1, i);
        }
    }
    let mut current = ambient.0.clone();
    for st in &mut stations {
        if st.aperture.is_some() {
            st.surface.material = current.clone();
        } else {
            current = st.surface.material.clone();
        }
    }
    for i in 0..stations.len() {
        let next = stations.get(i + 1).map_or(s_image, |n| n.s);
        stations[i].surface.thickness = next - stations[i].s;
    }

    let stop = match declared_stop(scene, source, detector) {
        Err(e) => return Err(e),
        Ok(None) => None,
        Ok(Some(ap)) => match stations.iter().position(|st| st.aperture == Some(ap)) {
            Some(i) => Some(i),
            None => {
                return Err(ExportError::Stop(format!(
                    "the declared stop, aperture {}, is not on the path from source {} to \
                     detector {}",
                    describe(scene, ObjectId(ap.0)),
                    source.0,
                    detector.0
                )))
            }
        },
    };

    let illumination = illumination(scene, source, d, stations.first().map(|s| s.s), &mut notes);
    let (aperture, fields, field_type, object_distance) = match illumination {
        Ok(i) => i,
        Err(why) => {
            reasons.push(why);
            (
                SeqAperture::Epd { value: 0.0 },
                vec![0.0],
                FieldType::Angle,
                None,
            )
        }
    };

    if stations.is_empty() {
        reasons.push("the path meets no surface between the source and the detector".into());
    }
    if !reasons.is_empty() {
        return Err(ExportError::Unsupported(reasons));
    }

    Ok(SeqSystem {
        surfaces: stations.into_iter().map(|st| st.surface).collect(),
        stop,
        aperture,
        fields,
        field_type,
        object_distance,
        object_space_telecentric: false,
        image_semi_aperture: None,
        aimed_beams: Vec::new(),
        wavelengths: wavelengths(scene, source),
        notes,
    })
}

/// The body whose material fills `point`: the highest priority containing it,
/// ties to the highest id, the rule the tracer resolves media by (§9.4).
fn medium_body<'a>(scene: &'a Scene3, point: Vec3, tol: &ToleranceContext) -> Option<&'a Body3> {
    scene
        .bodies
        .iter()
        .filter(|b| b.is_3d())
        .filter(|b| b.geometry.contains(point, &b.common.transform, tol) == Containment::Inside)
        .max_by_key(|b| (b.priority, b.common.id.0))
}

/// The aperture the scene declares as the stop of this path, if it declares
/// one. A stop can be named from either end, by the path intent or by the
/// aperture, and the two have to agree.
fn declared_stop(
    scene: &Scene3,
    source: SourceId,
    detector: DetectorId,
) -> Result<Option<ApertureId>, ExportError> {
    let mut named: Vec<ApertureId> = Vec::new();
    for intent in scene
        .intent
        .primary_paths
        .iter()
        .filter(|p| p.source == source && p.detector == detector)
    {
        named.extend(intent.stop);
        for a in &scene.apertures {
            if a.primary_stop_for.contains(&intent.id) {
                named.push(a.id());
            }
        }
    }
    named.sort_by_key(|a| a.0);
    named.dedup();
    match named.as_slice() {
        [] => Ok(None),
        [one] => Ok(Some(*one)),
        several => Err(ExportError::Stop(format!(
            "the scene declares {} different stops for this path: {}",
            several.len(),
            several
                .iter()
                .map(|a| describe(scene, ObjectId(a.0)))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// One optical face of a revolved body, in the body's local frame, oriented
/// so a positive curvature has its centre toward local `+X`.
struct Face {
    vertex_x: f64,
    c: f64,
    k: f64,
    half_ap: f64,
    coeffs: Vec<f64>,
}

/// Is `body` built from annular sag pieces (a Fresnel lens)?
fn has_sag_segments(body: &Body3) -> bool {
    match &body.geometry {
        SolidGeometry3::Profile(p) => p
            .profile
            .edges
            .iter()
            .any(|e| matches!(e.curve, Curve2::SagSegment { .. })),
        _ => false,
    }
}

fn revolved_face(body: &Body3, patch: u64) -> Option<Face> {
    let SolidGeometry3::Profile(p) = &body.geometry else {
        return None;
    };
    if p.construction != ProfileConstruction3::RevolveLocalX {
        return None;
    }
    let edge = p.profile.edges.iter().find(|e| e.patch.0 == patch)?;
    let (base, coeffs) = match &edge.curve {
        Curve2::Conic(c) => (*c, Vec::new()),
        Curve2::EvenAsphere { base, coeffs } => (*base, coeffs.to_vec()),
        _ => return None,
    };
    // A face whose axis runs along local `-X` describes the same surface with
    // its curvature measured the other way.
    let along = base.axis.x.signum();
    if base.axis.y.abs() > ALONG_AXIS || base.vertex.y.abs() > ON_AXIS_MM {
        return None;
    }
    Some(Face {
        vertex_x: base.vertex.x,
        c: along * base.c,
        k: base.k,
        half_ap: base.half_ap,
        coeffs: coeffs.into_iter().map(|a| along * a).collect(),
    })
}

enum FaceFidelity {
    Exact,
    Dropped(&'static str),
    Refused(&'static str),
}

fn face_fidelity(r: &SurfaceResponse) -> FaceFidelity {
    match r {
        SurfaceResponse::DielectricFresnel {
            ar_residual: SpectralScalar::Constant(v),
        } if *v == 1.0 => FaceFidelity::Exact,
        SurfaceResponse::DielectricFresnel { .. } => FaceFidelity::Dropped("its AR residual"),
        SurfaceResponse::Coating { .. } => FaceFidelity::Dropped("its thin-film coating"),
        _ => FaceFidelity::Refused("a surface response other than a glass interface"),
    }
}

/// The ambient medium as a prescription material, with its index.
fn ambient_material(scene: &Scene3) -> (SeqMaterial, Option<f64>) {
    let n = scene.env.ambient_index;
    if n == 1.0 {
        (SeqMaterial::Air, Some(1.0))
    } else {
        (SeqMaterial::Ideal { n }, Some(n))
    }
}

/// The system aperture, the fields, their kind and the object distance, from
/// the source that lights the path.
fn illumination(
    scene: &Scene3,
    source: SourceId,
    d: Vec3,
    first_surface: Option<f64>,
    notes: &mut Vec<String>,
) -> Result<(SeqAperture, Vec<f64>, FieldType, Option<f64>), String> {
    let src = scene
        .sources
        .iter()
        .find(|s| s.id == source)
        .ok_or_else(|| format!("source {} is not in the scene", source.0))?;
    match src.angular {
        AngularDistribution3::Direction { local_dir } => {
            let beam = src.transform.rotate(local_dir.normalize());
            if beam.cross(d).length() > ALONG_AXIS || beam.dot(d) < 0.0 {
                let deg = beam.dot(d).clamp(-1.0, 1.0).acos().to_degrees();
                return Err(format!(
                    "the beam runs {deg:.6}° from the path's axis; off-axis fields are not \
                     exported yet"
                ));
            }
            let epd = match src.spatial {
                SpatialDistribution3::Disc { radius_mm } => 2.0 * radius_mm,
                SpatialDistribution3::Rectangle {
                    half_width_mm,
                    half_height_mm,
                } => {
                    let epd = 2.0 * half_width_mm.max(half_height_mm);
                    notes.push(format!(
                        "the beam is a {} x {} mm rectangle and a sequential pupil is round: the \
                         entrance pupil is taken as {epd} mm, across its longer side",
                        2.0 * half_width_mm,
                        2.0 * half_height_mm
                    ));
                    epd
                }
                SpatialDistribution3::LineLegacy { half_width_mm } => {
                    notes.push(format!(
                        "the beam is a v1 line {} mm long: the entrance pupil is taken as its \
                         length",
                        2.0 * half_width_mm
                    ));
                    2.0 * half_width_mm
                }
                SpatialDistribution3::Point => {
                    return Err("the collimated source is a single ray; give it a width".into())
                }
            };
            Ok((
                SeqAperture::Epd { value: epd },
                vec![0.0],
                FieldType::Angle,
                None,
            ))
        }
        AngularDistribution3::UniformConeSolidAngle { half_angle_rad } => {
            if !matches!(src.spatial, SpatialDistribution3::Point) {
                return Err(
                    "an extended source's field points are not exported yet; a point source is"
                        .into(),
                );
            }
            let distance = first_surface
                .ok_or_else(|| "the point source has no surface to be an object for".to_string())?;
            let na = scene.env.ambient_index * half_angle_rad.sin();
            Ok((
                SeqAperture::ObjectNa { value: na },
                vec![0.0],
                FieldType::ObjectHeight,
                Some(distance),
            ))
        }
        AngularDistribution3::LambertianHemisphere => Err(
            "the source is a Lambertian emitter, which fills its hemisphere; a sequential system \
             needs its aperture from a cone or a beam"
                .into(),
        ),
        AngularDistribution3::PlanarFanLegacy { .. } => {
            Err("the source is a v1 fan, which has no aperture in 3D".into())
        }
    }
}

/// The source's spectral lines, primary first: the highest weight, and among
/// equal weights the line nearest the d line, then the rest as authored.
fn wavelengths(scene: &Scene3, source: SourceId) -> Vec<f64> {
    let Some(src) = scene.sources.iter().find(|s| s.id == source) else {
        return vec![LAMBDA_D_UM];
    };
    let lines = src.spectrum.samples();
    let top = lines
        .iter()
        .map(|(_, w)| *w)
        .fold(f64::NEG_INFINITY, f64::max);
    let primary = lines
        .iter()
        .enumerate()
        .filter(|(_, (_, w))| *w == top)
        .min_by(|(_, (a, _)), (_, (b, _))| {
            (a - LAMBDA_D_UM).abs().total_cmp(&(b - LAMBDA_D_UM).abs())
        })
        .map(|(i, _)| i);
    let mut out = Vec::with_capacity(lines.len());
    if let Some(p) = primary {
        out.push(lines[p].0);
    }
    out.extend(
        lines
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != primary)
            .map(|(_, (l, _))| *l),
    );
    if out.is_empty() {
        out.push(LAMBDA_D_UM);
    }
    out
}

/// What a zero-thickness surface is, in the words its component would use.
fn surface_kind(scene: &Scene3, id: ObjectId) -> &'static str {
    let Some(s) = scene.surfaces.iter().find(|s| s.common.id == id) else {
        return "a surface";
    };
    match &s.response_front {
        SurfaceResponse::Mirror { .. } => "a mirror",
        SurfaceResponse::ExplicitRt { .. } => "a beam splitter or filter",
        SurfaceResponse::Coating { .. } => "a coated plate (a dichroic, filter or mirror)",
        SurfaceResponse::Polarizer { .. } => "a polarizer",
        SurfaceResponse::Retarder { .. } => "a waveplate",
        SurfaceResponse::Absorber => "an absorber",
        SurfaceResponse::ThinLens { .. } => "an ideal thin lens",
        SurfaceResponse::DielectricFresnel { .. } => "a bare zero-thickness interface",
    }
}

/// `#id`, with the object's label when it has one.
fn describe(scene: &Scene3, id: ObjectId) -> String {
    let label = scene
        .bodies
        .iter()
        .map(|b| &b.common)
        .chain(scene.surfaces.iter().map(|s| &s.common))
        .chain(scene.apertures.iter().map(|a| &a.common))
        .chain(scene.detectors.iter().map(|d| &d.common))
        .find(|c| c.id == id)
        .map(|c| c.label.trim())
        .unwrap_or("");
    if label.is_empty() {
        format!("#{}", id.0)
    } else {
        format!("#{} ({label})", id.0)
    }
}

/// Drop trailing zero aspheric terms, so a plain conic reports none.
fn trim(mut coeffs: Vec<f64>) -> Vec<f64> {
    while coeffs.last() == Some(&0.0) {
        coeffs.pop();
    }
    coeffs
}
