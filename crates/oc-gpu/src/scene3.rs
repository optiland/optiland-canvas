//! `Scene3` → flat, f32, normalized buffers for the 3D preview (§12, PR 3D-07).
//!
//! This is the 3D counterpart of [`crate::scene`]. It compiles the canonical f64
//! [`Scene3`] into an f32 form the interactive CPU tracer ([`crate::cpu3`]) and,
//! later, the WGSL shader consume. Two rules shape it:
//!
//! **Normalize before narrowing.** f32 has ~7 significant digits, so a scene a
//! metre from the origin loses millimetre detail the moment its coordinates are
//! cast. Every point is first mapped into a unit-ish frame around the scene
//! centre — `p_norm = (p_world − origin) / scale` (§12.5) — and only then cast to
//! f32. Results map back to millimetres at the end. A rigid transform survives
//! this unchanged in rotation; only lengths scale.
//!
//! **Reject, never fake.** A geometry the preview cannot trace exactly is recorded
//! as a [`Reject3`], not swapped for a lookalike. No arc becomes a chord, no
//! asphere its base conic (§12.3). Supported: the analytic bodies (sphere, box,
//! cylinder), extruded line profiles (prisms), zero-thickness surfaces (mirror,
//! absorber, scalar dielectric, explicit R/T), finite aperture masks, and area
//! detectors. Rejected, each with a named issue: revolved profiles and extrusions
//! with curved edges (the surface-of-revolution solver is the remaining piece),
//! and tabulated responses.

use glam::{DQuat, DVec3, Mat3, Vec2, Vec3};
use oc_core::material::{Cauchy, Material};
use oc_core::profile2::ProfileConstruction3;
use oc_core::scene3::{
    AbsorptionModel, Aperture3, Body3, Detector3, DetectorGeometry3, DetectorTermination,
    PlanarShape2, Scene3, SolidGeometry3, SpectralScalar, SurfaceGeometry3, SurfaceObject3,
    SurfaceResponse, TwoSidedSurfaceResponse,
};
use oc_core::source3::EmittedRay3;
use oc_core::vec3::Isometry3;

/// Body geometry kinds the preview understands.
pub const OBJ_SPHERE: u32 = 0;
pub const OBJ_BOX: u32 = 1;
/// A finite cylinder along the object's canonical `+Z` (the axis is folded into
/// [`GpuBody3::rot`]). `params = (radius, half_length, 0)`.
pub const OBJ_CYLINDER: u32 = 2;
/// A closed line-profile extruded through local Z. Its polygon and Z limits are
/// in [`GpuBody3::extrude`].
pub const OBJ_EXTRUDE: u32 = 3;

/// Surface behaviour kinds.
pub const SURF_DIELECTRIC: u32 = 0;
pub const SURF_ABSORBER: u32 = 1;

/// Dispersion models (shared with the 2D compiler's meaning).
pub const DISP_CONST: u32 = 0;
pub const DISP_CAUCHY: u32 = 1;
pub const DISP_SELLMEIER: u32 = 2;

/// The scene-normalization frame (§12.5). `scale` is a length in mm; `origin` is
/// the scene centre in mm.
#[derive(Debug, Clone, Copy)]
pub struct Norm {
    pub origin: DVec3,
    pub scale: f64,
}

impl Norm {
    /// A world point (mm) in the normalized frame, as f32.
    #[inline]
    pub fn point(&self, p: DVec3) -> Vec3 {
        ((p - self.origin) / self.scale).as_vec3()
    }
    /// A world point (mm) back from a normalized f32 point.
    #[inline]
    pub fn world_point(&self, p: Vec3) -> DVec3 {
        p.as_dvec3() * self.scale + self.origin
    }
    /// A normalized length (dimensionless) back to millimetres.
    #[inline]
    pub fn world_len(&self, t: f32) -> f64 {
        t as f64 * self.scale
    }
}

/// A closed line-profile extruded through local Z, normalized.
#[derive(Debug, Clone)]
pub struct GpuExtrude {
    /// The closed polygon in the local `XY` plane, in loop order.
    pub verts: Vec<Vec2>,
    pub z_min: f32,
    pub z_max: f32,
}

/// A body compiled for the preview: geometry, surface, and a rigid placement in
/// the *normalized* frame. Plain `f32` — the GPU `repr(C)` buffer layout lands
/// with the shader (PR 3D-08); this is the CPU oracle's working form.
#[derive(Debug, Clone)]
pub struct GpuBody3 {
    pub kind: u32,
    pub surface: u32,
    pub disp: u32,
    pub priority: i32,
    /// Local(canonical)-to-world rotation. For a cylinder this folds in the axis
    /// alignment, so the primitive is always along canonical `+Z`.
    pub rot: Mat3,
    /// Normalized translation `(t_world − origin) / scale`.
    pub pos: Vec3,
    /// Sphere: `(radius, 0, 0)`; box: half-extents; cylinder: `(radius,
    /// half_length, 0)`. Normalized.
    pub params: Vec3,
    /// The extruded polygon, when `kind == OBJ_EXTRUDE`.
    pub extrude: Option<GpuExtrude>,
    pub index_const: f32,
    pub dp: [f32; 6],
    /// Constant bulk attenuation, `mm⁻¹` (0 = none). Applied per world length.
    pub alpha_per_mm: f32,
    /// The source object id, for diagnostics and capability parity.
    pub object: u64,
}

impl GpuBody3 {
    /// The refractive index at `lambda_um`. Mirrors the 2D compiler's dispersion.
    pub fn index(&self, lambda_um: f32) -> f32 {
        match self.disp {
            DISP_SELLMEIER => {
                let l2 = lambda_um * lambda_um;
                let n2 = 1.0
                    + self.dp[0] * l2 / (l2 - self.dp[3])
                    + self.dp[1] * l2 / (l2 - self.dp[4])
                    + self.dp[2] * l2 / (l2 - self.dp[5]);
                n2.max(1.0).sqrt()
            }
            DISP_CAUCHY => self.dp[0] + self.dp[1] / (lambda_um * lambda_um),
            _ => self.index_const,
        }
    }
}

/// A normalized launch ray.
#[derive(Debug, Clone, Copy)]
pub struct GpuRay3 {
    pub origin: Vec3,
    pub dir: Vec3,
    pub flux: f32,
    pub lambda_um: f32,
}

/// An object the preview cannot trace exactly, and why (§3.5, §12.3).
#[derive(Debug, Clone, PartialEq)]
pub struct Reject3 {
    pub object: u64,
    pub kind: String,
}

/// Build options.
#[derive(Debug, Clone, Copy)]
pub struct BuildOpts {
    pub ghosts: bool,
    pub flux_eps: f32,
    pub max_ghost: u32,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            ghosts: false,
            flux_eps: 1e-4,
            max_ghost: 2,
        }
    }
}

/// The compiled, uploadable 3D scene.
pub struct GpuScene3 {
    pub bodies: Vec<GpuBody3>,
    pub surfaces: Vec<GpuSurface3>,
    pub apertures: Vec<GpuAperture3>,
    pub detectors: Vec<GpuDetector3>,
    pub rays: Vec<GpuRay3>,
    pub ambient: f32,
    pub norm: Norm,
    /// The scene-escape distance, normalized.
    pub bounds_norm: f32,
    pub ghosts: bool,
    pub flux_eps: f32,
    pub max_ghost: u32,
    /// Objects left out of the trace, each with its reason. A non-empty list
    /// means the preview is not showing the whole scene.
    pub rejected: Vec<Reject3>,
}

impl GpuScene3 {
    /// Compile a scene, emitting launch rays from its sources.
    pub fn build(scene: &Scene3, opts: BuildOpts) -> Self {
        let rays: Vec<EmittedRay3> = scene.sources.iter().flat_map(|s| s.emit()).collect();
        Self::build_with_rays(scene, &rays, opts)
    }

    /// Compile a scene with an explicit launch set — the form the parity tests
    /// use so the f32 and f64 paths see identical rays.
    pub fn build_with_rays(scene: &Scene3, launch: &[EmittedRay3], opts: BuildOpts) -> Self {
        let norm = scene_norm(scene);
        let mut bodies = Vec::new();
        let mut surfaces = Vec::new();
        let mut apertures = Vec::new();
        let mut detectors = Vec::new();
        let mut rejected = Vec::new();
        // A geometry the preview cannot trace exactly is named, not dropped: the
        // caller (the preview badge) must be able to say the scene is not fully
        // shown (§3.5, §12.3).
        let mut reject = |object: u64, kind: String| rejected.push(Reject3 { object, kind });

        for b in &scene.bodies {
            match compile_body(b, &norm, &scene.catalog) {
                Ok(gb) => bodies.push(gb),
                Err(kind) => reject(b.common.id.0, kind),
            }
        }
        for s in &scene.surfaces {
            match compile_surface(s, &norm) {
                Ok(gs) => surfaces.push(gs),
                Err(kind) => reject(s.common.id.0, kind),
            }
        }
        for a in &scene.apertures {
            match compile_aperture(a, &norm) {
                Ok(ga) => apertures.push(ga),
                Err(kind) => reject(a.common.id.0, kind),
            }
        }
        for d in &scene.detectors {
            match compile_detector(d, &norm) {
                Ok(gd) => detectors.push(gd),
                Err(kind) => reject(d.common.id.0, kind),
            }
        }

        let rays = launch
            .iter()
            .map(|e| GpuRay3 {
                origin: norm.point(e.ray.origin),
                dir: e.ray.dir.as_vec3().normalize(),
                flux: e.flux as f32,
                lambda_um: e.lambda_um as f32,
            })
            .collect();

        Self {
            bodies,
            surfaces,
            apertures,
            detectors,
            rays,
            ambient: scene.env.ambient_index as f32,
            norm,
            bounds_norm: (1.0e4 / norm.scale) as f32,
            ghosts: opts.ghosts,
            flux_eps: opts.flux_eps,
            max_ghost: opts.max_ghost,
            rejected,
        }
    }
}

/// The normalization frame: centre on the supported bodies' world bounds, scale
/// by the diagonal (floored at 1 mm so an empty or single-point scene is sane).
fn scene_norm(scene: &Scene3) -> Norm {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    let mut grow = |p: DVec3| {
        min = min.min(p);
        max = max.max(p);
    };
    for b in &scene.bodies {
        if let Some((c, r)) = body_world_bound(b) {
            grow(c - DVec3::splat(r));
            grow(c + DVec3::splat(r));
        }
    }
    // The event objects have extent too; a mirror or detector far from the bodies
    // must not fall outside the frame and lose its f32 detail.
    for s in &scene.surfaces {
        grow(s.common.transform.pos);
    }
    for a in &scene.apertures {
        grow(a.common.transform.pos);
    }
    for d in &scene.detectors {
        grow(d.common.transform.pos);
    }
    if !min.is_finite() || !max.is_finite() {
        return Norm {
            origin: DVec3::ZERO,
            scale: 1.0,
        };
    }
    let origin = 0.5 * (min + max);
    let scale = (max - min).length().max(1.0);
    Norm { origin, scale }
}

/// A crude world bounding sphere `(centre, radius)` for scene scale. Only the
/// supported analytic solids contribute; a rejected body is not being traced.
fn body_world_bound(b: &Body3) -> Option<(DVec3, f64)> {
    let t = &b.common.transform;
    match &b.geometry {
        SolidGeometry3::Sphere { radius_mm } => Some((t.pos, *radius_mm)),
        SolidGeometry3::Box { half_extents_mm } => Some((t.pos, half_extents_mm.length())),
        SolidGeometry3::Cylinder {
            radius_mm,
            half_length_mm,
            ..
        } => Some((t.pos, radius_mm.hypot(*half_length_mm))),
        SolidGeometry3::Profile(p) => {
            // The profile's XY reach plus its Z span, roughly.
            let r = p
                .profile
                .edges
                .iter()
                .flat_map(profile_edge_points)
                .fold(0.0_f64, |m, v| m.max(v.length()));
            let z = match p.construction {
                ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
                    z_min_mm.abs().max(z_max_mm.abs())
                }
                _ => 0.0,
            };
            Some((t.pos, r.hypot(z).max(1.0)))
        }
    }
}

/// A profile edge's representative points, for a crude bound.
fn profile_edge_points(e: &oc_core::profile2::ProfileEdge2) -> Vec<oc_core::vec::Vec2> {
    use oc_core::profile2::Curve2;
    match &e.curve {
        Curve2::Line { a, b } => vec![*a, *b],
        Curve2::Polyline { points } => points.clone(),
        Curve2::Conic(c) | Curve2::EvenAsphere { base: c, .. } => {
            vec![c.vertex, c.vertex + oc_core::vec::Vec2::new(0.0, c.half_ap)]
        }
        Curve2::Arc {
            center, radius_mm, ..
        } => {
            vec![*center + oc_core::vec::Vec2::splat(*radius_mm)]
        }
    }
}

/// Compile one body, or return the reason it cannot be traced exactly.
fn compile_body(
    b: &Body3,
    norm: &Norm,
    catalog: &oc_core::material::GlassCatalog,
) -> Result<GpuBody3, String> {
    let s = |v: f64| (v / norm.scale) as f32;
    // Body rotation from the transform. A cylinder composes an axis alignment on
    // top so the primitive is always along canonical +Z.
    let q = DQuat::from_xyzw(
        b.common.transform.rot_xyzw[0],
        b.common.transform.rot_xyzw[1],
        b.common.transform.rot_xyzw[2],
        b.common.transform.rot_xyzw[3],
    );
    let body_rot = Mat3::from_quat(q.as_quat());
    let pos = norm.point(b.common.transform.pos);

    let (kind, params, rot, extrude) = match &b.geometry {
        SolidGeometry3::Sphere { radius_mm } => (
            OBJ_SPHERE,
            Vec3::new(s(*radius_mm), 0.0, 0.0),
            body_rot,
            None,
        ),
        SolidGeometry3::Box { half_extents_mm } => (
            OBJ_BOX,
            (*half_extents_mm / norm.scale).as_vec3(),
            body_rot,
            None,
        ),
        SolidGeometry3::Cylinder {
            radius_mm,
            half_length_mm,
            axis_local,
        } => {
            let axis = axis_local.normalize().as_vec3();
            let align = Mat3::from_quat(glam::Quat::from_rotation_arc(Vec3::Z, axis));
            (
                OBJ_CYLINDER,
                Vec3::new(s(*radius_mm), s(*half_length_mm), 0.0),
                body_rot * align,
                None,
            )
        }
        SolidGeometry3::Profile(p) => {
            let (z_min, z_max) = match p.construction {
                ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => (z_min_mm, z_max_mm),
                ProfileConstruction3::RevolveLocalX => {
                    return Err("revolved profile (not yet in the f32 preview)".into())
                }
                ProfileConstruction3::PlanarOnly { .. } => {
                    return Err("planar-only body has no 3D form to preview".into())
                }
            };
            let verts = line_polygon(&p.profile, norm.scale)?;
            (
                OBJ_EXTRUDE,
                Vec3::ZERO,
                body_rot,
                Some(GpuExtrude {
                    verts,
                    z_min: s(z_min),
                    z_max: s(z_max),
                }),
            )
        }
    };

    let surface = classify_surface(&b.surface.default)?;
    let alpha_per_mm = match &b.absorption {
        AbsorptionModel::None => 0.0,
        AbsorptionModel::Constant { alpha_per_mm } => *alpha_per_mm as f32,
        AbsorptionModel::Tabulated { .. } => {
            return Err("tabulated volume absorption (not yet in the f32 preview)".into())
        }
    };

    let mut gb = GpuBody3 {
        kind,
        surface,
        disp: DISP_CONST,
        priority: b.priority,
        rot,
        pos,
        params,
        extrude,
        index_const: 1.0,
        dp: [0.0; 6],
        alpha_per_mm,
        object: b.common.id.0,
    };
    set_dispersion(&mut gb, &b.material, catalog)?;
    Ok(gb)
}

/// The closed polygon of a line-only profile, normalized and in loop order, or
/// the reason it is not extrudable in the f32 preview (a curved edge).
fn line_polygon(
    profile: &oc_core::profile2::ClosedProfile2,
    scale: f64,
) -> Result<Vec<Vec2>, String> {
    use oc_core::profile2::Curve2;
    let mut verts = Vec::new();
    for e in &profile.edges {
        match &e.curve {
            Curve2::Line { a, .. } => {
                verts.push(Vec2::new((a.x / scale) as f32, (a.y / scale) as f32))
            }
            Curve2::Polyline { points } => {
                for p in points.iter().take(points.len().saturating_sub(1)) {
                    verts.push(Vec2::new((p.x / scale) as f32, (p.y / scale) as f32));
                }
            }
            Curve2::Conic(_) | Curve2::EvenAsphere { .. } | Curve2::Arc { .. } => {
                return Err(
                    "extruded profile has a curved edge (not yet in the f32 preview)".into(),
                )
            }
        }
    }
    if verts.len() < 3 {
        return Err("extruded profile has fewer than three vertices".into());
    }
    Ok(verts)
}

/// A two-sided response the oracle supports, or the reason it does not. This
/// increment takes symmetric dielectric or absorber surfaces; a mirror body, an
/// explicit R/T body, or an asymmetric pair is deferred with a named issue.
fn classify_surface(r: &TwoSidedSurfaceResponse) -> Result<u32, String> {
    if r.from_outside != r.from_inside {
        return Err("asymmetric surface response (not yet in the f32 preview)".into());
    }
    match &r.from_outside {
        SurfaceResponse::DielectricFresnel { .. } => Ok(SURF_DIELECTRIC),
        SurfaceResponse::Absorber => Ok(SURF_ABSORBER),
        SurfaceResponse::Mirror { .. } => {
            Err("mirror body surface (not yet in the f32 preview)".into())
        }
        SurfaceResponse::ExplicitRt { .. } => {
            Err("explicit R/T body surface (not yet in the f32 preview)".into())
        }
    }
}

fn set_dispersion(
    gb: &mut GpuBody3,
    material: &Material,
    catalog: &oc_core::material::GlassCatalog,
) -> Result<(), String> {
    match material {
        Material::AbbePoint { nd, vd } => {
            let c = Cauchy::from_abbe(*nd, *vd);
            gb.disp = DISP_CAUCHY;
            gb.dp[0] = c.a as f32;
            gb.dp[1] = c.b as f32;
        }
        Material::Catalog(id) => {
            let glass = catalog
                .get(*id)
                .ok_or_else(|| "catalog glass not in the scene catalogue".to_string())?;
            gb.disp = DISP_SELLMEIER;
            for i in 0..3 {
                gb.dp[i] = glass.sell.b[i] as f32;
                gb.dp[3 + i] = glass.sell.c[i] as f32;
            }
        }
        Material::Grin(f) => gb.index_const = f.base_index as f32,
        Material::Mirror { .. } | Material::Absorber => gb.index_const = 1.0,
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shapes, responses, and the event objects
// ---------------------------------------------------------------------------

/// A finite region of a plane, normalized. The f32 counterpart of
/// [`PlanarShape2`], with the containment test the tracer needs.
#[derive(Debug, Clone)]
pub enum GpuShape2 {
    Rectangle { hw: f32, hh: f32 },
    Disc { r: f32 },
    Annulus { inner: f32, outer: f32 },
    Polygon { verts: Vec<Vec2> },
}

impl GpuShape2 {
    fn from_planar(shape: &PlanarShape2, scale: f64) -> Self {
        let s = |v: f64| (v / scale) as f32;
        match shape {
            PlanarShape2::Rectangle {
                half_width_mm,
                half_height_mm,
            } => GpuShape2::Rectangle {
                hw: s(*half_width_mm),
                hh: s(*half_height_mm),
            },
            PlanarShape2::Disc { radius_mm } => GpuShape2::Disc { r: s(*radius_mm) },
            PlanarShape2::Annulus { inner_mm, outer_mm } => GpuShape2::Annulus {
                inner: s(*inner_mm),
                outer: s(*outer_mm),
            },
            PlanarShape2::Polygon { vertices } => GpuShape2::Polygon {
                verts: vertices.iter().map(|p| Vec2::new(s(p.x), s(p.y))).collect(),
            },
        }
    }

    /// Is the in-plane point `p` inside the shape?
    pub fn contains(&self, p: Vec2) -> bool {
        match self {
            GpuShape2::Rectangle { hw, hh } => p.x.abs() <= *hw && p.y.abs() <= *hh,
            GpuShape2::Disc { r } => p.length_squared() <= r * r,
            GpuShape2::Annulus { inner, outer } => {
                let r2 = p.length_squared();
                r2 >= inner * inner && r2 <= outer * outer
            }
            GpuShape2::Polygon { verts } => point_in_polygon(verts, p),
        }
    }
}

/// Even-odd crossing test, matching `oc_core::geom3::primitives`.
fn point_in_polygon(verts: &[Vec2], p: Vec2) -> bool {
    let mut inside = false;
    let n = verts.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (verts[i], verts[j]);
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if p.x < a.x + t * (b.x - a.x) {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// A constant surface response the preview supports. A tabulated one is rejected.
#[derive(Debug, Clone, Copy)]
pub enum GpuResponse {
    Absorber,
    Mirror { r: f32 },
    Dielectric { ar: f32 },
    ExplicitRt { r: f32, t: f32 },
}

fn constant(s: &SpectralScalar) -> Result<f32, String> {
    match s {
        SpectralScalar::Constant(v) => Ok(*v as f32),
        SpectralScalar::Tabulated { .. } => {
            Err("tabulated surface response (not yet in the f32 preview)".into())
        }
    }
}

fn compile_response(r: &SurfaceResponse) -> Result<GpuResponse, String> {
    Ok(match r {
        SurfaceResponse::Absorber => GpuResponse::Absorber,
        SurfaceResponse::Mirror { reflectivity } => GpuResponse::Mirror {
            r: constant(reflectivity)?,
        },
        SurfaceResponse::DielectricFresnel { ar_residual } => GpuResponse::Dielectric {
            ar: constant(ar_residual)?,
        },
        SurfaceResponse::ExplicitRt {
            reflectance,
            transmittance,
        } => GpuResponse::ExplicitRt {
            r: constant(reflectance)?,
            t: constant(transmittance)?,
        },
    })
}

/// The rigid placement of an object in the normalized frame.
fn rigid(t: &Isometry3, norm: &Norm) -> (Mat3, Vec3) {
    let q = DQuat::from_xyzw(t.rot_xyzw[0], t.rot_xyzw[1], t.rot_xyzw[2], t.rot_xyzw[3]);
    (Mat3::from_quat(q.as_quat()), norm.point(t.pos))
}

/// A compiled zero-thickness surface: a finite region of its local `XY` plane,
/// with a response on each side.
#[derive(Debug, Clone)]
pub struct GpuSurface3 {
    pub rot: Mat3,
    pub pos: Vec3,
    pub shape: GpuShape2,
    pub front: GpuResponse,
    pub back: GpuResponse,
    pub object: u64,
}

fn compile_surface(s: &SurfaceObject3, norm: &Norm) -> Result<GpuSurface3, String> {
    let shape = match &s.geometry {
        SurfaceGeometry3::Plane { shape } => GpuShape2::from_planar(shape, norm.scale),
        SurfaceGeometry3::SurfaceOfRevolution { .. } => {
            return Err("surface of revolution (not yet in the f32 preview)".into())
        }
    };
    let (rot, pos) = rigid(&s.common.transform, norm);
    Ok(GpuSurface3 {
        rot,
        pos,
        shape,
        front: compile_response(&s.response_front)?,
        back: compile_response(&s.response_back)?,
        object: s.common.id.0,
    })
}

/// A compiled finite aperture: an opening in an opaque mask, in a plane placed in
/// the normalized frame.
#[derive(Debug, Clone)]
pub struct GpuAperture3 {
    pub origin: Vec3,
    pub normal: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub clear: GpuShape2,
    pub outer: Option<GpuShape2>,
    pub object: u64,
}

fn compile_aperture(a: &Aperture3, norm: &Norm) -> Result<GpuAperture3, String> {
    let t = &a.common.transform;
    let pl = &a.shape.plane_local;
    let (rot, _) = rigid(t, norm);
    let origin = norm.point(t.transform_point(pl.origin));
    Ok(GpuAperture3 {
        origin,
        normal: (rot * pl.normal.as_vec3()).normalize(),
        u: (rot * pl.u.as_vec3()).normalize(),
        v: (rot * pl.v.as_vec3()).normalize(),
        clear: GpuShape2::from_planar(&a.shape.clear_opening, norm.scale),
        outer: a
            .shape
            .outer_extent
            .as_ref()
            .map(|s| GpuShape2::from_planar(s, norm.scale)),
        object: a.common.id.0,
    })
}

/// A compiled area detector's face shape.
#[derive(Debug, Clone, Copy)]
pub enum DetShape {
    Rectangle { hw: f32, hh: f32 },
    Disc { r: f32 },
    Annulus { inner: f32, outer: f32 },
}

/// A compiled area detector: a finite region of its local `XY` plane, binned.
#[derive(Debug, Clone)]
pub struct GpuDetector3 {
    pub rot: Mat3,
    pub pos: Vec3,
    pub shape: DetShape,
    pub bins: [usize; 2],
    pub terminate: bool,
    /// Cell area in `mm²` (physical, not normalized), for flux density.
    pub cell_area_mm2: f32,
    pub object: u64,
}

fn compile_detector(d: &Detector3, norm: &Norm) -> Result<GpuDetector3, String> {
    let s = |v: f64| (v / norm.scale) as f32;
    let (shape, total_mm2) = match &d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => (
            DetShape::Rectangle {
                hw: s(*half_width_mm),
                hh: s(*half_height_mm),
            },
            4.0 * half_width_mm * half_height_mm,
        ),
        DetectorGeometry3::Disc { radius_mm } => (
            DetShape::Disc { r: s(*radius_mm) },
            std::f64::consts::PI * radius_mm * radius_mm,
        ),
        DetectorGeometry3::Annulus { inner_mm, outer_mm } => (
            DetShape::Annulus {
                inner: s(*inner_mm),
                outer: s(*outer_mm),
            },
            std::f64::consts::PI * (outer_mm * outer_mm - inner_mm * inner_mm),
        ),
        DetectorGeometry3::LineProbeLegacy { .. } => {
            return Err("line probe has no area in 3D (not previewed)".into())
        }
    };
    let (rot, pos) = rigid(&d.common.transform, norm);
    let cells = (d.bins[0].max(1) * d.bins[1].max(1)) as f64;
    Ok(GpuDetector3 {
        rot,
        pos,
        shape,
        bins: d.bins,
        terminate: matches!(d.termination, DetectorTermination::Terminate),
        cell_area_mm2: (total_mm2 / cells) as f32,
        object: d.common.id.0,
    })
}

impl DetShape {
    /// Is the local `(u, v)` point on the face?
    pub fn contains(&self, p: Vec2) -> bool {
        match self {
            DetShape::Rectangle { hw, hh } => p.x.abs() <= *hw && p.y.abs() <= *hh,
            DetShape::Disc { r } => p.length_squared() <= r * r,
            DetShape::Annulus { inner, outer } => {
                let r2 = p.length_squared();
                r2 >= inner * inner && r2 <= outer * outer
            }
        }
    }

    /// The cell index for a local hit, row-major over `bins`, or `None` off the
    /// bounding grid. Mirrors `oc_trace`'s `detector_cell`.
    pub fn cell(&self, p: Vec2, bins: [usize; 2]) -> Option<usize> {
        let (nu, nv) = (bins[0].max(1), bins[1].max(1));
        let (fx, fy) = match self {
            DetShape::Rectangle { hw, hh } => ((p.x + hw) / (2.0 * hw), (p.y + hh) / (2.0 * hh)),
            DetShape::Disc { r } => ((p.x + r) / (2.0 * r), (p.y + r) / (2.0 * r)),
            DetShape::Annulus { outer, .. } => {
                ((p.x + outer) / (2.0 * outer), (p.y + outer) / (2.0 * outer))
            }
        };
        if !(0.0..=1.0).contains(&fx) || !(0.0..=1.0).contains(&fy) {
            return None;
        }
        let iu = ((fx * nu as f32) as usize).min(nu - 1);
        let iv = ((fy * nv as f32) as usize).min(nv - 1);
        Some(iv * nu + iu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::scene3::{ObjectCommon, SolidGeometry3, SurfaceProps3};

    fn sphere_scene(center: DVec3, radius: f64) -> Scene3 {
        let mut scene = Scene3::new();
        let id = scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, oc_core::id::LayerId(0));
        common.transform = oc_core::vec3::Isometry3::from_pos(center);
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Sphere { radius_mm: radius },
            material: Material::AbbePoint { nd: 1.5, vd: 64.0 },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        scene
    }

    #[test]
    fn normalization_round_trips_a_point() {
        let norm = Norm {
            origin: DVec3::new(100.0, -50.0, 25.0),
            scale: 40.0,
        };
        let p = DVec3::new(112.0, -37.5, 60.0);
        let back = norm.world_point(norm.point(p));
        // f32 in the middle, so a few ULP at scene scale, not exact.
        assert!(
            (back - p).length() < 1e-3,
            "round trip off by {}",
            (back - p).length()
        );
    }

    #[test]
    fn the_frame_centres_on_the_scene_and_scales_by_its_diagonal() {
        // A sphere far from the origin: the frame must move to it, or f32 would
        // lose the detail (§12.5).
        let scene = sphere_scene(DVec3::new(1000.0, 0.0, 0.0), 5.0);
        let gpu = GpuScene3::build(&scene, BuildOpts::default());
        assert!((gpu.norm.origin - DVec3::new(1000.0, 0.0, 0.0)).length() < 1e-9);
        // Diagonal of a 10 mm cube bound = 10·√3.
        assert!((gpu.norm.scale - 10.0 * 3.0_f64.sqrt()).abs() < 1e-9);
        // The body sits at the normalized origin, radius scaled down.
        assert!(gpu.bodies[0].pos.length() < 1e-6);
        assert!((gpu.bodies[0].params.x as f64 - 5.0 / gpu.norm.scale).abs() < 1e-6);
    }

    #[test]
    fn a_revolved_profile_is_rejected_by_name() {
        // The curved profile bodies are the one geometry family still to land; the
        // preview must reject them, not draw a chord.
        use oc_core::profile2::{
            ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
        };
        let mut scene = Scene3::new();
        let id = scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, oc_core::id::LayerId(0));
        common.transform = oc_core::vec3::Isometry3::IDENTITY;
        let edges = vec![
            ProfileEdge2 {
                patch: scene.alloc_patch_id(),
                curve: Curve2::Conic(ConicParams2 {
                    vertex: oc_core::vec::Vec2::new(-1.0, 0.0),
                    axis: oc_core::vec::Vec2::X,
                    c: 0.05,
                    k: 0.0,
                    half_ap: 4.0,
                }),
            },
            ProfileEdge2 {
                patch: scene.alloc_patch_id(),
                curve: Curve2::Line {
                    a: oc_core::vec::Vec2::new(-1.0, 4.0),
                    b: oc_core::vec::Vec2::new(1.0, 4.0),
                },
            },
            ProfileEdge2 {
                patch: scene.alloc_patch_id(),
                curve: Curve2::Line {
                    a: oc_core::vec::Vec2::new(1.0, 4.0),
                    b: oc_core::vec::Vec2::new(1.0, 0.0),
                },
            },
        ];
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Profile(ProfileSolid3 {
                profile: ClosedProfile2 { edges },
                construction: ProfileConstruction3::RevolveLocalX,
            }),
            material: Material::AbbePoint { nd: 1.5, vd: 64.0 },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        let gpu = GpuScene3::build(&scene, BuildOpts::default());
        assert!(gpu.bodies.is_empty());
        assert_eq!(gpu.rejected.len(), 1);
        assert_eq!(gpu.rejected[0].object, id.0);
        assert!(gpu.rejected[0].kind.contains("revolved"));
    }

    #[test]
    fn a_catalog_glass_body_resolves_its_dispersion() {
        let mut scene = sphere_scene(DVec3::ZERO, 5.0);
        let bk7 = scene.catalog.find("N-BK7").expect("N-BK7 in catalogue");
        scene.bodies[0].material = Material::Catalog(bk7);
        let gpu = GpuScene3::build(&scene, BuildOpts::default());
        assert_eq!(gpu.bodies.len(), 1);
        assert_eq!(gpu.bodies[0].disp, DISP_SELLMEIER);
        // N-BK7 near the d line is ~1.5168.
        let n = gpu.bodies[0].index(oc_core::material::LAMBDA_D as f32);
        assert!((n - 1.5168).abs() < 2e-3, "N-BK7 index {n}");
    }
}
