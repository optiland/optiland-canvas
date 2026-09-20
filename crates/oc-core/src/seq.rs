//! Sequential prescription — the rotationally-symmetric handoff to Optiland (§6).
//!
//! A canvas [`Scene`] is a *non-sequential* 2D pile of closed elements; an
//! Optiland `Optic` is an *ordered* list of surfaces along an axis. This module
//! is the exact, lossless-where-it-can-be bridge between the two, expressed
//! entirely in plain data so the Python (`oc-py`) side only has to build/read an
//! Optiland object — none of the optical geometry lives in Python.
//!
//! The neutral form is [`SeqSystem`]: surfaces front-to-back along the canvas
//! `+x` axis (≡ Optiland `+z`), each carrying the medium that follows it. The
//! two conversions are:
//!
//! - [`Scene::to_seq`] — extract the prescription. Exact for the rotationally
//!   symmetric refractive subset; anything it cannot represent (splines, GRIN,
//!   tilted/off-axis elements, …) is *reported* in [`SeqSystem::notes`] rather
//!   than silently approximated. "A tool that lies about fidelity is worthless."
//! - [`SeqSystem::to_scene`] — rebuild a scene. A run of surfaces bounding one
//!   glass becomes one [`Element`] (two curved faces + a ground rim), exactly as
//!   the part is drawn.
//!
//! Round-tripping a prescription through a scene (or an Optiland `Optic`) is
//! parameter-stable; the tests at the bottom and in `oc-py` lock that down.

use crate::geom::{ConicParams, Edge};
use crate::material::{GlassCatalog, Material};
use crate::scene::{
    Detector, Element, Environment, Region, Scene, Source, SourceKind, Spectrum, SurfaceSpec,
};
use crate::vec::{Isometry2, Vec2};
use serde::{Deserialize, Serialize};

/// Where the sequential axis lives in a canvas scene: the `+x` direction.
const AXIS: Vec2 = Vec2::new(1.0, 0.0);
/// A surface counts as rotationally symmetric only if its axis is within this
/// of `±x̂` (dot product) — a tilted face cannot map to an Optiland surface.
const AXIS_ALIGN_TOL: f64 = 1e-6;
/// Two curved faces at the same axial position with the same curvature are the
/// same physical surface (a cemented interface, drawn once per touching body).
const COINCIDE_TOL: f64 = 1e-9;
/// A default object-space gap when rebuilding a scene's source.
const DEFAULT_OBJECT_GAP: f64 = 25.0;

/// The bulk medium after a surface, in the neutral vocabulary shared with
/// Optiland. Each variant maps 1:1 in both directions (see `oc-py`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SeqMaterial {
    /// Ambient (Optiland `"air"` / `IdealMaterial(1)`).
    Air,
    /// Fixed index, no dispersion (Optiland `IdealMaterial(n)`).
    Ideal { n: f64 },
    /// A point on the Abbe map (canvas `AbbePoint`, Optiland `AbbeMaterial`).
    Abbe { nd: f64, vd: f64 },
    /// A named catalog glass (canvas `Catalog`, Optiland `Material(name)`).
    Glass { name: String },
    /// A reflective body (canvas `Mirror`, Optiland `"mirror"`).
    Mirror,
}

impl SeqMaterial {
    /// The neutral material for a canvas bulk [`Material`] (needs the catalog to
    /// resolve a [`Material::Catalog`] id back to its glass name). Absorber/GRIN
    /// have no sequential analogue and return `None` (the caller notes it).
    pub fn from_material(m: &Material, catalog: &GlassCatalog) -> Option<Self> {
        match m {
            Material::AbbePoint { nd, vd } => Some(SeqMaterial::Abbe { nd: *nd, vd: *vd }),
            Material::Catalog(id) => catalog
                .get(*id)
                .map(|g| SeqMaterial::Glass { name: g.name.clone() }),
            Material::Mirror { .. } => Some(SeqMaterial::Mirror),
            Material::Absorber | Material::Grin(_) => None,
        }
    }

    /// The canvas bulk [`Material`] for this medium, resolving a glass name via
    /// the catalog. Returns `None` for [`SeqMaterial::Air`] (no element bulk).
    fn to_material(&self, catalog: &GlassCatalog) -> Option<Material> {
        match self {
            SeqMaterial::Air => None,
            // Non-dispersive: a huge Abbe number flattens the Cauchy fit to a
            // constant `n = nd` (canvas has no dedicated ideal-glass variant).
            SeqMaterial::Ideal { n } => Some(Material::AbbePoint {
                nd: *n,
                vd: 1.0e9,
            }),
            SeqMaterial::Abbe { nd, vd } => Some(Material::AbbePoint { nd: *nd, vd: *vd }),
            SeqMaterial::Glass { name } => catalog
                .find(name)
                .map(Material::Catalog)
                .or(Some(Material::AbbePoint { nd: 1.5, vd: 50.0 })),
            SeqMaterial::Mirror => Some(Material::Mirror { reflectivity: 1.0 }),
        }
    }

    fn is_glassy(&self) -> bool {
        !matches!(self, SeqMaterial::Air)
    }
}

/// One sequential surface: a rotationally-symmetric conic (optionally with even
/// aspheric terms) plus the medium that fills the space *after* it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeqSurface {
    /// Signed radius of curvature (mm). `None` is a plane (infinite radius) —
    /// kept as an explicit null so the JSON never has to encode an infinity.
    pub radius: Option<f64>,
    /// Conic constant `k` (0 sphere, −1 parabola, …).
    #[serde(default)]
    pub conic: f64,
    /// Axial gap to the next surface's vertex (mm). On the last surface this is
    /// the distance to the image plane.
    pub thickness: f64,
    /// Clear semi-aperture (mm).
    pub semi_aperture: f64,
    /// Even aspheric coefficients for `r⁴, r⁶, …` (empty ⇒ a plain conic).
    #[serde(default)]
    pub coefficients: Vec<f64>,
    /// Medium filling the space after this surface.
    pub material: SeqMaterial,
    /// This surface reflects (the medium after equals the medium before).
    #[serde(default)]
    pub reflective: bool,
}

impl SeqSurface {
    fn curvature(&self) -> f64 {
        match self.radius {
            Some(r) if r.abs() > f64::EPSILON => 1.0 / r,
            _ => 0.0,
        }
    }

    /// Sag (displacement along `+x`) at the aperture edge, for placing the rim.
    fn edge_sag(&self) -> f64 {
        let c = self.curvature();
        let h = self.semi_aperture;
        let r2 = h * h;
        let disc = 1.0 - (1.0 + self.conic) * c * c * r2;
        let mut sag = c * r2 / (1.0 + disc.max(0.0).sqrt());
        let mut yp = r2 * r2; // y⁴
        for a in &self.coefficients {
            sag += a * yp;
            yp *= r2;
        }
        sag
    }

    /// Build the local-frame boundary edge for this surface at axial position
    /// `x`, with the axis pointing along `+x` (the standard, unified sign
    /// convention: `radius > 0` ⇒ centre of curvature toward `+x`).
    fn to_edge(&self, x: f64) -> Edge {
        let base = ConicParams {
            vertex: Vec2::new(x, 0.0),
            axis: AXIS,
            c: self.curvature(),
            k: self.conic,
            half_ap: self.semi_aperture,
        };
        if self.coefficients.is_empty() {
            Edge::Conic(base)
        } else {
            let mut coeffs = [0.0; 8];
            for (dst, src) in coeffs.iter_mut().zip(self.coefficients.iter()) {
                *dst = *src;
            }
            Edge::Asphere { base, coeffs }
        }
    }
}

/// The system aperture, mirroring Optiland's three aperture kinds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SeqAperture {
    /// Entrance pupil diameter (mm).
    Epd { value: f64 },
    /// Image-space f-number.
    ImageFno { value: f64 },
    /// Object-space numerical aperture.
    ObjectNa { value: f64 },
}

impl SeqAperture {
    fn epd_or(&self, fallback: f64) -> f64 {
        match self {
            SeqAperture::Epd { value } => *value,
            _ => fallback,
        }
    }
}

/// How field points are specified (Optiland's field types we can represent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    /// Field angle in degrees (collimated input, object at infinity).
    Angle,
    /// Object height (mm).
    ObjectHeight,
}

/// A rotationally-symmetric sequential system — the neutral prescription that
/// maps 1:1 onto an Optiland `Optic`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeqSystem {
    /// Optical surfaces, front (object side) to back (image side).
    pub surfaces: Vec<SeqSurface>,
    /// Index of the aperture-stop surface, when the design declares one.
    ///
    /// `None` means nothing named a stop: a v1 scene has no apertures to name
    /// one with, and a v2 scene may simply not have declared it. Choosing one
    /// then is the caller's decision to make out loud (`to_optic`'s `stop=`,
    /// or its labelled inference), never a silent surface zero. Stop semantics
    /// and clear-aperture clipping are different things, so a surface that
    /// vignettes is not thereby the stop.
    #[serde(default)]
    pub stop: Option<usize>,
    /// System aperture.
    pub aperture: SeqAperture,
    /// Field points (interpreted per [`SeqSystem::field_type`]).
    #[serde(default)]
    pub fields: Vec<f64>,
    /// Interpretation of [`SeqSystem::fields`].
    pub field_type: FieldType,
    /// Distance from the object to the first surface (mm). `None` is an object
    /// at **infinity** (collimated input) — the usual case for angle fields, and
    /// what Optiland writes as an infinite object thickness. Kept as an explicit
    /// null so the JSON never has to encode an infinity.
    #[serde(default)]
    pub object_distance: Option<f64>,
    /// Wavelengths (µm); the first is primary.
    pub wavelengths: Vec<f64>,
    /// Human-readable fidelity report: what could not be represented exactly.
    #[serde(default)]
    pub notes: Vec<String>,
}

impl SeqSystem {
    /// Distance from the last optical surface to the image plane (mm).
    pub fn image_distance(&self) -> f64 {
        self.surfaces.last().map(|s| s.thickness).unwrap_or(0.0)
    }

    /// Serialize to pretty JSON (the wire format shared with `oc-py`).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("seq serializes")
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }

    /// Rebuild a canvas [`Scene`]: each glass gap becomes one lens element, a
    /// collimated source is placed per field, and a detector marks the image
    /// plane. Uses the built-in glass catalog to resolve named glasses.
    pub fn to_scene(&self) -> Scene {
        let mut scene = Scene::new();
        let catalog = scene.catalog.clone();

        // Absolute axial position of each surface vertex (first at x = 0).
        let mut xs = Vec::with_capacity(self.surfaces.len());
        let mut x = 0.0;
        for s in &self.surfaces {
            xs.push(x);
            x += s.thickness;
        }
        let image_x = x;

        // A glass gap (medium after surface i is glassy) → one element bounded
        // by surface i (front) and surface i+1 (rear).
        let mut max_ap: f64 = 1.0;
        for (i, surf) in self.surfaces.iter().enumerate() {
            max_ap = max_ap.max(surf.semi_aperture);
            if !surf.material.is_glassy() || i + 1 >= self.surfaces.len() {
                continue;
            }
            let Some(bulk) = surf.material.to_material(&catalog) else {
                continue;
            };
            let front = &self.surfaces[i];
            let rear = &self.surfaces[i + 1];
            let boundary = lens_boundary(front, xs[i], rear, xs[i + 1]);
            let mut elem = Element::new(crate::scene::ElementId(0), boundary, bulk);
            // A reflective surface makes its element a mirror face.
            if surf.reflective || matches!(surf.material, SeqMaterial::Mirror) {
                elem.surface.default = SurfaceSpec {
                    mirror: true,
                    ..SurfaceSpec::default()
                };
            }
            scene.add_element(elem);
        }

        // Wavelengths drive both the source spectrum and the render set.
        let wls: Vec<f64> = if self.wavelengths.is_empty() {
            vec![crate::material::LAMBDA_D]
        } else {
            self.wavelengths.clone()
        };
        scene.env = Environment {
            ambient_index: 1.0,
            wavelengths: wls.iter().map(|l| (*l, 1.0)).collect(),
        };
        let spectrum = if wls.len() == 1 {
            Spectrum::Monochromatic(wls[0])
        } else {
            Spectrum::Lines(wls.iter().map(|l| (*l, 1.0)).collect())
        };

        // One collimated source per field angle, filling the entrance pupil. A
        // collimated source's axial position does not affect the optics (the
        // rays are parallel), so an object at infinity just gets a sensible
        // drawing gap; a finite conjugate is placed at its true distance.
        let half_width = 0.5 * self.aperture.epd_or(2.0 * max_ap);
        let object_gap = self.object_distance.unwrap_or(DEFAULT_OBJECT_GAP);
        let source_x = xs.first().copied().unwrap_or(0.0) - object_gap;
        let fields = if self.fields.is_empty() {
            vec![0.0]
        } else {
            self.fields.clone()
        };
        for f in &fields {
            let rot = match self.field_type {
                FieldType::Angle => f.to_radians(),
                FieldType::ObjectHeight => 0.0,
            };
            scene.sources.push(Source {
                kind: SourceKind::Collimated { half_width },
                transform: Isometry2::new(Vec2::new(source_x, 0.0), rot),
                spectrum: spectrum.clone(),
                flux: 1.0,
                rays: 15,
            });
        }

        // A detector segment across the axis at the image plane.
        let det_h = max_ap.max(1.0);
        scene.detectors.push(Detector::new(
            Vec2::new(image_x, -det_h),
            Vec2::new(image_x, det_h),
            64,
        ));

        scene
    }
}

/// The closed boundary of a lens element bounded by `front` (at `xf`) and
/// `rear` (at `xr`): front face, top rim, rear face, bottom rim.
fn lens_boundary(front: &SeqSurface, xf: f64, rear: &SeqSurface, xr: f64) -> Vec<Edge> {
    let (hf, hr) = (front.semi_aperture, rear.semi_aperture);
    let ftip_x = xf + front.edge_sag();
    let rtip_x = xr + rear.edge_sag();
    vec![
        front.to_edge(xf),
        Edge::Line {
            a: Vec2::new(ftip_x, hf),
            b: Vec2::new(rtip_x, hr),
        },
        rear.to_edge(xr),
        Edge::Line {
            a: Vec2::new(rtip_x, -hr),
            b: Vec2::new(ftip_x, -hf),
        },
    ]
}

/// A curved optical face pulled out of an element, in world coordinates.
struct RawSurface {
    vertex_x: f64,
    /// Signed curvature in the `+x`-forward frame (`c > 0` ⇒ centre toward `+x`).
    c: f64,
    k: f64,
    half_ap: f64,
    coeffs: Vec<f64>,
}

impl Scene {
    /// Extract the rotationally-symmetric sequential prescription (§6).
    ///
    /// Exact for the axis-aligned refractive subset. Every feature that has no
    /// sequential analogue — splines, arcs, GRIN, absorbers, tilted or
    /// off-axis elements, extra sources — is recorded in [`SeqSystem::notes`]
    /// instead of being quietly dropped.
    pub fn to_seq(&self) -> SeqSystem {
        let mut notes = Vec::new();

        // 1. Pull every curved face out of every element into world space.
        let mut raw: Vec<RawSurface> = Vec::new();
        for elem in &self.elements {
            if matches!(elem.material, Material::Grin(_)) {
                notes.push("GRIN element dropped: no sequential analogue".into());
            }
            for edge in &elem.boundary {
                match edge {
                    Edge::Line { .. } => {} // ground rim, not an optical surface
                    Edge::Conic(cp) => push_raw(&mut raw, &mut notes, elem, cp, &[]),
                    Edge::Asphere { base, coeffs } => {
                        push_raw(&mut raw, &mut notes, elem, base, coeffs)
                    }
                    Edge::Arc { .. } => notes
                        .push("arc face dropped: not a rotationally-symmetric conic".into()),
                    Edge::Spline { .. } => {
                        notes.push("spline (freeform) face dropped: not sequential".into())
                    }
                }
            }
        }

        // 2. Sort front-to-back and merge coincident faces (cemented interfaces
        //    are drawn once per touching body).
        raw.sort_by(|a, b| a.vertex_x.partial_cmp(&b.vertex_x).unwrap());
        let mut merged: Vec<RawSurface> = Vec::new();
        for s in raw {
            if let Some(last) = merged.last() {
                if (last.vertex_x - s.vertex_x).abs() < COINCIDE_TOL
                    && (last.c - s.c).abs() < COINCIDE_TOL
                {
                    continue;
                }
            }
            merged.push(s);
        }

        // 3. Resolve the medium in each gap by sampling the scene on-axis, and
        //    assemble the sequential surfaces.
        let n = merged.len();
        let mut surfaces = Vec::with_capacity(n);
        for (i, s) in merged.iter().enumerate() {
            // Sample the medium after this surface: the on-axis midpoint to the
            // next surface, or just past the last one (image space).
            let sample_x = if i + 1 < n {
                0.5 * (s.vertex_x + merged[i + 1].vertex_x)
            } else {
                s.vertex_x + 1.0
            };
            let region = self.region_at(Vec2::new(sample_x, 0.0));
            let (material, reflective) = self.sample_material(region, &mut notes);

            let thickness = if i + 1 < n {
                merged[i + 1].vertex_x - s.vertex_x
            } else {
                self.image_gap(s.vertex_x)
            };
            surfaces.push(SeqSurface {
                radius: if s.c.abs() > f64::EPSILON {
                    Some(1.0 / s.c)
                } else {
                    None
                },
                conic: s.k,
                thickness,
                semi_aperture: s.half_ap,
                coefficients: trim_coeffs(&s.coeffs),
                material,
                reflective,
            });
        }

        // 4. Aperture / fields / wavelengths from the sources and environment.
        let first_x = merged.first().map(|s| s.vertex_x).unwrap_or(0.0);
        let (aperture, fields, field_type, object_distance) =
            self.extract_illumination(first_x, &mut notes);
        let wavelengths: Vec<f64> = if self.env.wavelengths.is_empty() {
            vec![crate::material::LAMBDA_D]
        } else {
            self.env.wavelengths.iter().map(|(l, _)| *l).collect()
        };

        SeqSystem {
            surfaces,
            // A v1 scene has no aperture objects, so nothing in it can name a
            // stop. Reporting none is the honest answer; the export decides.
            stop: None,
            aperture,
            fields,
            field_type,
            object_distance,
            wavelengths,
            notes,
        }
    }

    /// Neutral material for a sampled region, plus whether the surface into it
    /// is reflective.
    fn sample_material(&self, region: Region<'_>, notes: &mut Vec<String>) -> (SeqMaterial, bool) {
        match region {
            Region::Ambient => (SeqMaterial::Air, false),
            Region::In(e) => match SeqMaterial::from_material(&e.material, &self.catalog) {
                Some(SeqMaterial::Mirror) => (SeqMaterial::Air, true),
                Some(m) => (m, false),
                None => {
                    notes.push(format!(
                        "element material {:?} has no sequential analogue; treated as air",
                        e.material
                    ));
                    (SeqMaterial::Air, false)
                }
            },
        }
    }

    /// Distance from `vertex_x` (last surface) to the image plane, taken from
    /// the first on-axis detector if there is one.
    fn image_gap(&self, vertex_x: f64) -> f64 {
        for d in &self.detectors {
            // A near-vertical detector segment marks an image plane at its x.
            if (d.a.x - d.b.x).abs() < 1e-6 && d.a.x > vertex_x {
                return d.a.x - vertex_x;
            }
        }
        0.0
    }

    /// Aperture, fields, field type, and object distance inferred from the
    /// scene's sources. `first_x` is the leading surface's axial position, which
    /// a finite-conjugate (point) source is measured back from.
    fn extract_illumination(
        &self,
        first_x: f64,
        notes: &mut Vec<String>,
    ) -> (SeqAperture, Vec<f64>, FieldType, Option<f64>) {
        let mut epd = 0.0_f64;
        let mut fields = Vec::new();
        let mut field_type = FieldType::Angle;
        let mut saw_point = false;
        // `None` ⇒ object at infinity, which is what a collimated source means.
        let mut object_distance: Option<f64> = None;
        for src in &self.sources {
            match src.kind {
                SourceKind::Collimated { half_width } => {
                    epd = epd.max(2.0 * half_width);
                    let aim = src.transform.rotate(Vec2::X);
                    fields.push(aim.y.atan2(aim.x).to_degrees());
                }
                SourceKind::Point { .. } => {
                    saw_point = true;
                    field_type = FieldType::ObjectHeight;
                    fields.push(src.transform.pos.y);
                    object_distance = Some(first_x - src.transform.pos.x);
                }
            }
        }
        if saw_point {
            notes.push(
                "point source mapped to an object-height field; entrance pupil is a guess".into(),
            );
        }
        if epd <= 0.0 {
            epd = 2.0
                * self
                    .elements
                    .iter()
                    .flat_map(|e| e.boundary.iter())
                    .filter_map(|edge| match edge {
                        Edge::Conic(cp) => Some(cp.half_ap),
                        Edge::Asphere { base, .. } => Some(base.half_ap),
                        _ => None,
                    })
                    .fold(1.0_f64, f64::max);
        }
        fields.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        if fields.is_empty() {
            fields.push(0.0);
        }
        (
            SeqAperture::Epd { value: epd },
            fields,
            field_type,
            object_distance,
        )
    }
}

/// Push an element's conic face into `raw`, in world space, noting anything
/// that breaks rotational symmetry about the `x` axis.
fn push_raw(
    raw: &mut Vec<RawSurface>,
    notes: &mut Vec<String>,
    elem: &Element,
    cp: &ConicParams,
    coeffs: &[f64],
) {
    let world_axis = elem.transform.rotate(cp.axis).normalize();
    let along = world_axis.dot(AXIS);
    if along.abs() < 1.0 - AXIS_ALIGN_TOL {
        notes.push(format!(
            "tilted surface dropped: axis·x̂ = {along:.4} (not rotationally symmetric)"
        ));
        return;
    }
    let vertex = elem.transform.transform_point(cp.vertex);
    if vertex.y.abs() > 1e-6 {
        notes.push(format!(
            "off-axis surface dropped: vertex y = {:.4} mm",
            vertex.y
        ));
        return;
    }
    // Flip the local curvature into the +x-forward frame if the face's axis
    // points along −x (the two descriptions denote the same physical surface).
    let c = cp.c * along.signum();
    raw.push(RawSurface {
        vertex_x: vertex.x,
        c,
        k: cp.k,
        half_ap: cp.half_ap,
        coeffs: coeffs.to_vec(),
    });
}

/// Drop trailing-zero aspheric coefficients so a plain conic reports none.
fn trim_coeffs(coeffs: &[f64]) -> Vec<f64> {
    let last = coeffs.iter().rposition(|c| *c != 0.0);
    match last {
        Some(i) => coeffs[..=i].to_vec(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn abbe(nd: f64, vd: f64) -> SeqMaterial {
        SeqMaterial::Abbe { nd, vd }
    }

    /// A biconvex singlet, a plano-convex, a meniscus, a parabolic-front conic,
    /// an even asphere, an air-spaced doublet, and a cemented doublet.
    fn corpus() -> Vec<(&'static str, SeqSystem)> {
        let base = |surfaces, fields| SeqSystem {
            surfaces,
            stop: Some(0),
            aperture: SeqAperture::Epd { value: 20.0 },
            fields,
            field_type: FieldType::Angle,
            object_distance: None, // collimated ⇒ object at infinity
            wavelengths: vec![0.58756],
            notes: vec![],
        };
        let air = || SeqSurface {
            radius: None,
            conic: 0.0,
            thickness: 0.0,
            semi_aperture: 14.0,
            coefficients: vec![],
            material: SeqMaterial::Air,
            reflective: false,
        };
        vec![
            (
                "biconvex",
                base(
                    vec![
                        SeqSurface {
                            radius: Some(50.0),
                            thickness: 6.0,
                            material: abbe(1.5, 60.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(-50.0),
                            thickness: 44.0,
                            ..air()
                        },
                    ],
                    vec![0.0, 5.0],
                ),
            ),
            (
                "plano_convex",
                base(
                    vec![
                        SeqSurface {
                            radius: Some(30.0),
                            thickness: 5.0,
                            material: abbe(1.5168, 64.17),
                            ..air()
                        },
                        SeqSurface {
                            radius: None,
                            thickness: 60.0,
                            ..air()
                        },
                    ],
                    vec![0.0],
                ),
            ),
            (
                "meniscus",
                base(
                    vec![
                        SeqSurface {
                            radius: Some(25.0),
                            thickness: 4.0,
                            material: abbe(1.6, 50.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(60.0),
                            thickness: 80.0,
                            ..air()
                        },
                    ],
                    vec![0.0],
                ),
            ),
            (
                "parabolic_front",
                base(
                    vec![
                        SeqSurface {
                            radius: Some(50.0),
                            conic: -1.0,
                            thickness: 6.0,
                            material: abbe(1.5, 60.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(-50.0),
                            thickness: 54.0,
                            ..air()
                        },
                    ],
                    vec![0.0],
                ),
            ),
            (
                "even_asphere",
                base(
                    vec![
                        SeqSurface {
                            radius: Some(50.0),
                            thickness: 6.0,
                            coefficients: vec![1e-6, -2e-9],
                            material: abbe(1.5, 60.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(-50.0),
                            thickness: 54.0,
                            ..air()
                        },
                    ],
                    vec![0.0],
                ),
            ),
            (
                "air_spaced_doublet",
                base(
                    vec![
                        SeqSurface {
                            radius: Some(60.0),
                            thickness: 5.0,
                            material: abbe(1.6, 50.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(-60.0),
                            thickness: 3.0,
                            material: SeqMaterial::Air,
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(40.0),
                            thickness: 4.0,
                            material: abbe(1.5, 60.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(-200.0),
                            thickness: 48.0,
                            ..air()
                        },
                    ],
                    vec![0.0],
                ),
            ),
            (
                "cemented_doublet",
                base(
                    vec![
                        SeqSurface {
                            radius: Some(60.0),
                            thickness: 5.0,
                            material: abbe(1.62, 60.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(-45.0),
                            thickness: 3.0,
                            material: abbe(1.52, 55.0),
                            ..air()
                        },
                        SeqSurface {
                            radius: Some(-150.0),
                            thickness: 82.0,
                            ..air()
                        },
                    ],
                    vec![0.0],
                ),
            ),
        ]
    }

    fn assert_surface_eq(name: &str, a: &SeqSurface, b: &SeqSurface) {
        let ra = a.radius.unwrap_or(f64::INFINITY);
        let rb = b.radius.unwrap_or(f64::INFINITY);
        assert!(
            (ra - rb).abs() < 1e-6 || (ra.is_infinite() && rb.is_infinite()),
            "{name}: radius {ra} vs {rb}"
        );
        assert!((a.conic - b.conic).abs() < 1e-9, "{name}: conic");
        assert!(
            (a.thickness - b.thickness).abs() < 1e-6,
            "{name}: thickness {} vs {}",
            a.thickness,
            b.thickness
        );
        assert!(
            (a.semi_aperture - b.semi_aperture).abs() < 1e-6,
            "{name}: semi_aperture"
        );
        assert_eq!(a.coefficients.len(), b.coefficients.len(), "{name}: n coeffs");
        for (x, y) in a.coefficients.iter().zip(&b.coefficients) {
            assert!((x - y).abs() < 1e-15, "{name}: coeff");
        }
        assert_eq!(a.material, b.material, "{name}: material");
    }

    #[test]
    fn scene_roundtrip_is_parameter_stable() {
        for (name, sys) in corpus() {
            let scene = sys.to_scene();
            let back = scene.to_seq();
            assert!(
                back.notes.is_empty(),
                "{name}: unexpected fidelity notes: {:?}",
                back.notes
            );
            assert_eq!(
                back.surfaces.len(),
                sys.surfaces.len(),
                "{name}: surface count {} vs {}",
                back.surfaces.len(),
                sys.surfaces.len()
            );
            for (i, (a, b)) in sys.surfaces.iter().zip(&back.surfaces).enumerate() {
                assert_surface_eq(&format!("{name}[{i}]"), a, b);
            }
            assert!(
                (sys.aperture.epd_or(0.0) - back.aperture.epd_or(0.0)).abs() < 1e-6,
                "{name}: EPD"
            );
            assert_eq!(sys.wavelengths, back.wavelengths, "{name}: wavelengths");
            assert_eq!(sys.field_type, back.field_type, "{name}: field_type");
            assert_eq!(sys.fields.len(), back.fields.len(), "{name}: field count");
            for (a, b) in sys.fields.iter().zip(&back.fields) {
                assert!((a - b).abs() < 1e-9, "{name}: field angle {a} vs {b}");
            }
        }
    }

    #[test]
    fn seq_json_roundtrips() {
        for (name, sys) in corpus() {
            let json = sys.to_json();
            let back = SeqSystem::from_json(&json).unwrap();
            assert_eq!(sys, back, "{name}: JSON round trip");
        }
    }

    #[test]
    fn catalog_glass_survives_roundtrip() {
        let mut sys = corpus().into_iter().next().unwrap().1;
        sys.surfaces[0].material = SeqMaterial::Glass {
            name: "N-BK7".into(),
        };
        let back = sys.to_scene().to_seq();
        assert_eq!(
            back.surfaces[0].material,
            SeqMaterial::Glass {
                name: "N-BK7".into()
            }
        );
    }

    #[test]
    fn reconstructed_lens_actually_focuses_light() {
        // A quick sanity check that the rebuilt element is a real, traceable
        // lens: its glass region contains the axis between the two vertices.
        let sys = corpus().into_iter().next().unwrap().1; // biconvex
        let scene = sys.to_scene();
        assert_eq!(scene.elements.len(), 1);
        let e = &scene.elements[0];
        assert!(e.contains(Vec2::new(3.0, 0.0)), "axis point inside glass");
        assert!(!e.contains(Vec2::new(-20.0, 0.0)), "object space is air");
        assert!(!e.contains(Vec2::new(40.0, 0.0)), "image space is air");
    }

    #[test]
    fn spline_face_is_reported_not_dropped_silently() {
        let mut scene = Scene::new();
        scene.add_element(Element::new(
            crate::scene::ElementId(0),
            vec![Edge::Spline {
                ctrl: vec![Vec2::new(0.0, -5.0), Vec2::new(0.5, 0.0), Vec2::new(0.0, 5.0)],
                degree: 2,
            }],
            Material::AbbePoint { nd: 1.5, vd: 60.0 },
        ));
        let seq = scene.to_seq();
        assert!(
            seq.notes.iter().any(|n| n.contains("spline")),
            "spline should be reported: {:?}",
            seq.notes
        );
    }
}
