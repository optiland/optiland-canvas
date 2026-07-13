//! The scene model — the `.ocanvas` document (§3, §6).
//!
//! Scene state is the single source of truth. IDs are indices into the
//! respective vectors; the serialized form is plain, diffable JSON.

use crate::geom::{Aabb, ConicParams, Edge, EdgeHit, Ray};
use crate::material::{GlassCatalog, Material};
use crate::vec::{Isometry2, Vec2};
use serde::{Deserialize, Serialize};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(pub u32);
        impl $name {
            #[inline]
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(ElementId);
id_type!(SourceId);
id_type!(DetectorId);
id_type!(LayerId);

/// Per-surface optical behaviour. Defaults describe an uncoated dielectric
/// interface (full Fresnel, no forced mirror/absorb).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceSpec {
    /// Force the surface to reflect (a coated mirror on this edge).
    pub mirror: bool,
    /// Reflectivity used when `mirror` (or the element is `Material::Mirror`).
    pub reflectivity: f64,
    /// Multiplies the Fresnel reflectance. `1.0` = uncoated; a good AR coat is
    /// ~`5e-3`. This is the scalar-R stand-in for coatings (§1 "Out").
    pub ar_residual: f64,
    /// The surface fully absorbs incident flux (a painted-black edge).
    pub absorb: bool,
    /// Surface roughness. Reserved for scatter (M2+); ignored by M0.
    pub roughness: f64,
}

impl Default for SurfaceSpec {
    fn default() -> Self {
        Self {
            mirror: false,
            reflectivity: 1.0,
            ar_residual: 1.0,
            absorb: false,
            roughness: 0.0,
        }
    }
}

/// Surface properties for an element: a default plus sparse per-edge overrides.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SurfaceProps {
    pub default: SurfaceSpec,
    /// `(edge_index, spec)` overrides.
    pub overrides: Vec<(usize, SurfaceSpec)>,
}

impl SurfaceProps {
    pub fn for_edge(&self, i: usize) -> SurfaceSpec {
        self.overrides
            .iter()
            .find(|(idx, _)| *idx == i)
            .map(|(_, s)| *s)
            .unwrap_or(self.default)
    }
}

/// An optical element: a closed CCW boundary loop with a bulk material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub id: ElementId,
    /// Closed loop, CCW, in the element's local frame.
    pub boundary: Vec<Edge>,
    pub material: Material,
    pub surface: SurfaceProps,
    pub transform: Isometry2,
    /// Nesting resolution: the medium at a point is the material of the
    /// highest-priority element containing it (§4.2).
    pub priority: i32,
    pub locked: bool,
    pub layer: LayerId,
}

/// A single edge intersection in world space, tagged with the edge index.
#[derive(Debug, Clone, Copy)]
pub struct ElementHit {
    pub t: f64,
    pub point: Vec2,
    /// Unit geometric normal in world space (orientation unspecified).
    pub normal: Vec2,
    pub edge_index: usize,
}

impl Element {
    pub fn new(id: ElementId, boundary: Vec<Edge>, material: Material) -> Self {
        Self {
            id,
            boundary,
            material,
            surface: SurfaceProps::default(),
            transform: Isometry2::IDENTITY,
            priority: 0,
            locked: false,
            layer: LayerId(0),
        }
    }

    #[inline]
    fn to_local(&self, ray: &Ray) -> Ray {
        Ray {
            origin: self.transform.inverse_transform_point(ray.origin),
            dir: self.transform.inv_rotate(ray.dir),
        }
    }

    /// Enumerate world-space intersections of `ray` with this element's
    /// boundary at `t > tmin`.
    pub fn intersect<F: FnMut(ElementHit)>(&self, ray: &Ray, tmin: f64, push: &mut F) {
        let local = self.to_local(ray);
        for (ei, edge) in self.boundary.iter().enumerate() {
            edge.intersect(&local, tmin, &mut |h: EdgeHit| {
                push(ElementHit {
                    t: h.t, // rotation preserves distance, so world t == local t
                    point: self.transform.transform_point(h.point),
                    normal: self.transform.rotate(h.normal),
                    edge_index: ei,
                });
            });
        }
    }

    /// Point-in-element test via boundary crossing count. The test direction is
    /// deliberately non-axis-aligned to dodge tangency/vertex degeneracies (a
    /// known robustness risk, §10.3, to be fuzzed later).
    pub fn contains(&self, world_point: Vec2) -> bool {
        let local = self.transform.inverse_transform_point(world_point);
        let dir = Vec2::new(1.0, std::f64::consts::FRAC_1_PI).normalize();
        let ray = Ray { origin: local, dir };
        let mut count = 0u32;
        for edge in &self.boundary {
            edge.intersect(&ray, 1e-9, &mut |_| count += 1);
        }
        count % 2 == 1
    }

    /// World-space bounding box (rotated local box, re-bounded).
    pub fn aabb(&self) -> Aabb {
        let mut local = Aabb::EMPTY;
        for e in &self.boundary {
            local = local.union(&e.local_aabb());
        }
        let corners = [
            Vec2::new(local.min.x, local.min.y),
            Vec2::new(local.max.x, local.min.y),
            Vec2::new(local.min.x, local.max.y),
            Vec2::new(local.max.x, local.max.y),
        ];
        let mut world = Aabb::EMPTY;
        for c in corners {
            world.expand(self.transform.transform_point(c));
        }
        world
    }
}

/// A light source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub kind: SourceKind,
    /// Position (`pos`) and aim (`rot`, the +x direction after rotation).
    pub transform: Isometry2,
    pub spectrum: Spectrum,
    /// Total emitted flux, distributed across emitted rays.
    pub flux: f64,
    /// Number of spatial samples per wavelength.
    pub rays: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceKind {
    /// A point emitting a fan of half-angle `half_angle` radians about the aim.
    Point { half_angle: f64 },
    /// Parallel rays across a transverse aperture of half-width `half_width`.
    Collimated { half_width: f64 },
}

/// A wavelength distribution, in micrometres.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Spectrum {
    Monochromatic(f64),
    /// Discrete lines: `(λ_µm, weight)`.
    Lines(Vec<(f64, f64)>),
}

impl Spectrum {
    /// Normalized `(λ, weight)` samples; weights sum to 1.
    pub fn samples(&self) -> Vec<(f64, f64)> {
        match self {
            Spectrum::Monochromatic(l) => vec![(*l, 1.0)],
            Spectrum::Lines(v) => {
                let total: f64 = v.iter().map(|(_, w)| *w).sum();
                let total = if total > 0.0 { total } else { 1.0 };
                v.iter().map(|(l, w)| (*l, w / total)).collect()
            }
        }
    }
}

/// One ray emitted by a source, ready to trace.
#[derive(Debug, Clone, Copy)]
pub struct EmittedRay {
    pub ray: Ray,
    pub lambda_um: f64,
    pub flux: f64,
}

impl Source {
    /// Generate the rays for this source across its spatial and spectral samples.
    pub fn emit(&self) -> Vec<EmittedRay> {
        let aim = self.transform.rotate(Vec2::X);
        let normal = crate::vec::perp(aim);
        let origin = self.transform.pos;
        let n = self.rays.max(1);
        let spec = self.spectrum.samples();
        let mut out = Vec::with_capacity(n as usize * spec.len());

        for (lambda, sw) in &spec {
            let flux_each = self.flux * sw / n as f64;
            for i in 0..n {
                // s in [-1, 1] (or 0 for a single ray).
                let s = if n == 1 {
                    0.0
                } else {
                    -1.0 + 2.0 * i as f64 / (n - 1) as f64
                };
                let (origin_i, dir_i) = match self.kind {
                    SourceKind::Point { half_angle } => {
                        let ang = s * half_angle;
                        let (sn, cs) = ang.sin_cos();
                        // Rotate aim by `ang`.
                        let d = Vec2::new(cs * aim.x - sn * aim.y, sn * aim.x + cs * aim.y);
                        (origin, d)
                    }
                    SourceKind::Collimated { half_width } => {
                        (origin + s * half_width * normal, aim)
                    }
                };
                out.push(EmittedRay {
                    ray: Ray::new(origin_i, dir_i),
                    lambda_um: *lambda,
                    flux: flux_each,
                });
            }
        }
        out
    }
}

/// A detector: a line segment that accumulates ray hits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detector {
    pub a: Vec2,
    pub b: Vec2,
    /// Number of irradiance bins along the segment.
    pub bins: usize,
}

impl Detector {
    pub fn new(a: Vec2, b: Vec2, bins: usize) -> Self {
        Self { a, b, bins }
    }

    /// Intersect `ray` with the detector segment (`t > tmin`). Returns the hit
    /// parameter `t` and the fractional position `u ∈ [0, 1]` along `a → b`.
    pub fn intersect(&self, ray: &Ray, tmin: f64) -> Option<(f64, f64)> {
        let e = self.b - self.a;
        let det = e.perp_dot(ray.dir);
        if det.abs() < 1e-12 {
            return None;
        }
        let rhs = self.a - ray.origin;
        let t = e.perp_dot(rhs) / det;
        let u = ray.dir.perp_dot(rhs) / det;
        if t > tmin && (0.0..=1.0).contains(&u) {
            Some((t, u))
        } else {
            None
        }
    }
}

/// A drawing layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f64,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            name: "Layer 0".to_string(),
            visible: true,
            opacity: 1.0,
        }
    }
}

/// Global environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    /// Ambient (surrounding) refractive index, e.g. 1.0 for vacuum/air.
    pub ambient_index: f64,
    /// Default wavelength set `(λ_µm, weight)` for rendering/metrics.
    pub wavelengths: Vec<(f64, f64)>,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            ambient_index: 1.0,
            wavelengths: vec![(crate::material::LAMBDA_D, 1.0)],
        }
    }
}

/// The region a point falls in: ambient, or inside an element.
#[derive(Debug, Clone, Copy)]
pub enum Region<'a> {
    Ambient,
    In(&'a Element),
}

/// The full scene document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub elements: Vec<Element>,
    pub sources: Vec<Source>,
    pub detectors: Vec<Detector>,
    pub env: Environment,
    pub layers: Vec<Layer>,
    pub catalog: GlassCatalog,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            sources: Vec::new(),
            detectors: Vec::new(),
            env: Environment::default(),
            layers: vec![Layer::default()],
            catalog: GlassCatalog::builtin(),
        }
    }

    /// Add an element, assigning it a fresh id equal to its index.
    pub fn add_element(&mut self, mut e: Element) -> ElementId {
        let id = ElementId(self.elements.len() as u32);
        e.id = id;
        self.elements.push(e);
        id
    }

    /// The highest-priority element containing `point`, else ambient (§4.2).
    /// Ties in priority are broken toward the later (higher-index) element.
    pub fn region_at(&self, point: Vec2) -> Region<'_> {
        let mut best: Option<&Element> = None;
        for e in &self.elements {
            if e.contains(point) {
                match best {
                    Some(b) if b.priority >= e.priority => {}
                    _ => best = Some(e),
                }
            }
        }
        match best {
            Some(e) => Region::In(e),
            None => Region::Ambient,
        }
    }

    /// Refractive index of the material occupying `region` at `lambda_um`.
    /// Ambient and non-dielectric regions fall back to the ambient index.
    pub fn region_index(&self, region: Region<'_>, lambda_um: f64) -> f64 {
        match region {
            Region::Ambient => self.env.ambient_index,
            Region::In(e) => e
                .material
                .index_at(lambda_um, &self.catalog)
                .unwrap_or(self.env.ambient_index),
        }
    }

    /// Serialize to pretty `.ocanvas` JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("scene serializes")
    }

    /// Deserialize from `.ocanvas` JSON.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
}

/// Build a spherical `Conic` edge from an optical radius `R` (0 → plane).
pub fn spherical_edge(vertex: Vec2, axis: Vec2, radius: f64, half_ap: f64) -> Edge {
    let c = if radius.abs() < f64::EPSILON {
        0.0
    } else {
        1.0 / radius
    };
    Edge::Conic(ConicParams {
        vertex,
        axis,
        c,
        k: 0.0,
        half_ap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_block(half: f64) -> Element {
        // A square glass block centered at origin, side 2*half, CCW.
        let boundary = vec![
            Edge::Line {
                a: Vec2::new(-half, -half),
                b: Vec2::new(half, -half),
            },
            Edge::Line {
                a: Vec2::new(half, -half),
                b: Vec2::new(half, half),
            },
            Edge::Line {
                a: Vec2::new(half, half),
                b: Vec2::new(-half, half),
            },
            Edge::Line {
                a: Vec2::new(-half, half),
                b: Vec2::new(-half, -half),
            },
        ];
        Element::new(
            ElementId(0),
            boundary,
            Material::AbbePoint { nd: 1.5, vd: 60.0 },
        )
    }

    #[test]
    fn contains_inside_and_outside() {
        let e = unit_block(2.0);
        assert!(e.contains(Vec2::new(0.0, 0.0)));
        assert!(e.contains(Vec2::new(1.9, -1.9)));
        assert!(!e.contains(Vec2::new(3.0, 0.0)));
        assert!(!e.contains(Vec2::new(0.0, 5.0)));
    }

    #[test]
    fn region_priority_wins() {
        let mut scene = Scene::new();
        let mut outer = unit_block(3.0);
        outer.priority = 0;
        outer.material = Material::AbbePoint { nd: 1.5, vd: 60.0 };
        let mut inner = unit_block(1.0);
        inner.priority = 10;
        inner.material = Material::AbbePoint { nd: 1.8, vd: 40.0 };
        scene.add_element(outer);
        scene.add_element(inner);

        // Inside both -> inner (higher priority) wins.
        let n = scene.region_index(scene.region_at(Vec2::ZERO), crate::material::LAMBDA_D);
        assert!((n - 1.8).abs() < 1e-9, "expected inner index 1.8, got {n}");
        // Between the two -> outer.
        let n2 = scene.region_index(
            scene.region_at(Vec2::new(2.0, 0.0)),
            crate::material::LAMBDA_D,
        );
        assert!(
            (n2 - 1.5).abs() < 1e-9,
            "expected outer index 1.5, got {n2}"
        );
        // Outside both -> ambient.
        let n3 = scene.region_index(
            scene.region_at(Vec2::new(9.0, 0.0)),
            crate::material::LAMBDA_D,
        );
        assert!((n3 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn json_round_trip() {
        let mut scene = Scene::new();
        scene.add_element(unit_block(2.0));
        scene.sources.push(Source {
            kind: SourceKind::Collimated { half_width: 1.0 },
            transform: Isometry2::from_pos(Vec2::new(-5.0, 0.0)),
            spectrum: Spectrum::Monochromatic(crate::material::LAMBDA_D),
            flux: 1.0,
            rays: 8,
        });
        let json = scene.to_json();
        let back = Scene::from_json(&json).unwrap();
        assert_eq!(back.elements.len(), 1);
        assert_eq!(back.sources.len(), 1);
    }

    #[test]
    fn collimated_emit_is_parallel() {
        let s = Source {
            kind: SourceKind::Collimated { half_width: 2.0 },
            transform: Isometry2::from_pos(Vec2::new(-5.0, 0.0)),
            spectrum: Spectrum::Monochromatic(0.5),
            flux: 4.0,
            rays: 5,
        };
        let rays = s.emit();
        assert_eq!(rays.len(), 5);
        for r in &rays {
            assert!((r.ray.dir - Vec2::X).length() < 1e-12);
            assert!((r.flux - 4.0 / 5.0).abs() < 1e-12);
        }
        // Transverse spread spans the aperture.
        assert!((rays[0].ray.origin.y + 2.0).abs() < 1e-12);
        assert!((rays[4].ray.origin.y - 2.0).abs() < 1e-12);
    }
}
