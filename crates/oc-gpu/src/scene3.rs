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
//! cylinder), extruded line profiles (prisms), revolved profiles (conic and
//! even-asphere surfaces of revolution with their frustum rims), zero-thickness
//! surfaces (mirror, absorber, scalar dielectric, explicit R/T, thin lens),
//! finite aperture masks, and area detectors. A zero-thickness surface's
//! response scalars may be constant or tabulated in wavelength ([`Spectral32`],
//! sampled exactly as the f64 kernel samples them), which is what beam
//! splitters and dichroic filters need. Rejected, each with a named issue: an
//! off-axis arc revolve, a curved-edge extrusion, tabulated volume absorption,
//! and any body surface other than a symmetric dielectric or absorber.

use glam::{DQuat, DVec3, Mat3, Vec2, Vec3};
use oc_core::geom3::Intersect3;
use oc_core::material::{Cauchy, Material};
use oc_core::profile2::{ClosedProfile2, ProfileConstruction3};
use oc_core::scene3::{
    DetectorMeasure,
    AbsorptionModel, Aperture3, Body3, Detector3, DetectorGeometry3, DetectorTermination,
    Interpolation, OutOfRangePolicy, PlanarShape2, Scene3, SolidGeometry3, SpectralScalar,
    SurfaceGeometry3, SurfaceObject3, SurfaceResponse, TwoSidedSurfaceResponse,
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
/// A profile revolved about local `+X`. Its surfaces and containment contour are
/// in [`GpuBody3::revolve`].
pub const OBJ_REVOLVE: u32 = 4;
/// A hollow tube about local `+X`. `params = (inner_radius, outer_radius,
/// half_length)`, and its four faces carry their own responses through
/// [`GpuBody3::face_resp`] in the kernel's
/// [`tube_face`](oc_core::geom3::primitives::tube_face) order.
pub const OBJ_TUBE: u32 = 5;

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
    /// The revolved profile, when `kind == OBJ_REVOLVE`.
    pub revolve: Option<GpuRevolve>,
    /// One entry per revolved face, in face order: the response that face
    /// carries instead of the body's default, if any.
    ///
    /// This is how a coated plate is one object rather than two. The stack lives
    /// on the face the light meets first and plain glass on the other, and the
    /// difference between those two faces is the whole component.
    pub face_resp: Vec<Option<GpuResponse>>,
    pub index_const: f32,
    pub dp: [f32; 6],
    /// Constant bulk attenuation, `mm⁻¹` (0 = none). Applied per world length.
    pub alpha_per_mm: f32,
    /// The source object id, for diagnostics and capability parity.
    pub object: u64,
    /// The body's world box in the normalized frame, padded past f32
    /// rounding: a ray that does not enter it before a nearer hit cannot hit
    /// the body. The CPU mirror passes over such a body without solving its
    /// faces; the shader does not read it.
    pub bounds: (Vec3, Vec3),
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

/// What the f32 preview does with a fidelity profile (coating-fidelity spec
/// §5.3 rule 2).
///
/// The preview carries no Stokes vector, records no deposits, has no
/// temperature and clamps rather than refuses, so it runs the nearest profile
/// it can and says, by name, every switch value it could not honour.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewFidelity {
    /// The profile the preview actually runs.
    pub runs: oc_core::fidelity::FidelityProfile,
    /// One phrase per switch value refused, for example "polarized coatings".
    pub refused: Vec<String>,
    /// The sentence the preview badge shows, `None` when nothing was refused.
    pub message: Option<String>,
}

/// The preview's answer to a profile.
pub fn preview_fidelity(asked: &oc_core::fidelity::FidelityProfile) -> PreviewFidelity {
    use oc_core::fidelity::{CoatingModel, LayerAbsorption, Numerics, PlateModel, Temperature};
    let mut refused = Vec::new();
    let mut runs = *asked;
    if asked.coating_model == CoatingModel::Polarized {
        refused.push("polarized coatings".to_string());
        runs.coating_model = CoatingModel::Unpolarized;
    }
    if asked.layer_absorption != LayerAbsorption::Off {
        if asked.layer_absorption == LayerAbsorption::PerLayer {
            refused.push("per-layer absorption".to_string());
        }
        runs.layer_absorption = LayerAbsorption::Off;
    }
    if asked.plate_model == PlateModel::Incoherent {
        refused.push("the incoherent plate Mueller matrix".to_string());
        runs.plate_model = PlateModel::PhaseFree;
    }
    if asked.temperature == Temperature::Declared {
        refused.push("the declared temperature".to_string());
        runs.temperature = Temperature::Off;
    }
    runs.numerics = Numerics::Permissive;
    let message = (!refused.is_empty()).then(|| {
        let coatings = match runs.coating_model {
            CoatingModel::Nominal => "nominal coatings",
            _ => "unpolarized coatings",
        };
        let asked_name = asked.preset().map_or("a custom profile", oc_core::fidelity::Preset::name);
        format!(
            "Preview: {coatings}, absorption not recorded; it cannot run {}. The f64 trace uses {asked_name}.",
            refused.join(", ")
        )
    });
    PreviewFidelity { runs, refused, message }
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
        // Coatings bake at the wavelengths this launch set actually carries.
        let coat_lambdas = launch_lambdas(launch);
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
            match compile_body(
                b,
                &norm,
                &scene.catalog,
                scene.env.ambient_index,
                &coat_lambdas,
            ) {
                Ok(gb) => bodies.push(gb),
                Err(kind) => reject(b.common.id.0, kind),
            }
        }
        for s in &scene.surfaces {
            match compile_surface(s, &norm, scene.env.ambient_index, &coat_lambdas) {
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
        SolidGeometry3::Tube {
            outer_radius_mm,
            half_length_mm,
            ..
        } => Some((t.pos, outer_radius_mm.hypot(*half_length_mm))),
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
        Curve2::SagSegment {
            base,
            coeffs,
            y0,
            y1,
        } => vec![
            oc_core::geom3::profile::sag_segment_point(base, coeffs, *y0),
            oc_core::geom3::profile::sag_segment_point(base, coeffs, *y1),
        ],
    }
}

/// Compile one body, or return the reason it cannot be traced exactly.
/// A body's refractive index at one wavelength, in f64, for baking a coating
/// against the glass it actually sits on.
fn body_index_at(
    m: &Material,
    catalog: &oc_core::material::GlassCatalog,
    lambda_um: f64,
) -> Option<f64> {
    match m {
        Material::AbbePoint { nd, vd } => Some(Cauchy::from_abbe(*nd, *vd).index(lambda_um)),
        Material::Catalog(id) => catalog.get(*id).map(|g| g.index(lambda_um)),
        Material::Grin(f) => Some(f.base_index),
        Material::Mirror { .. } | Material::Absorber => Some(1.0),
    }
}

fn compile_body(
    b: &Body3,
    norm: &Norm,
    catalog: &oc_core::material::GlassCatalog,
    ambient: f64,
    coat_lambdas: &[f32],
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

    let mut extrude = None;
    let mut revolve = None;
    let (kind, params, rot) = match &b.geometry {
        SolidGeometry3::Sphere { radius_mm } => {
            (OBJ_SPHERE, Vec3::new(s(*radius_mm), 0.0, 0.0), body_rot)
        }
        SolidGeometry3::Box { half_extents_mm } => {
            (OBJ_BOX, (*half_extents_mm / norm.scale).as_vec3(), body_rot)
        }
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
            )
        }
        SolidGeometry3::Tube {
            inner_radius_mm,
            outer_radius_mm,
            half_length_mm,
        } => (
            OBJ_TUBE,
            Vec3::new(s(*inner_radius_mm), s(*outer_radius_mm), s(*half_length_mm)),
            body_rot,
        ),
        SolidGeometry3::Profile(p) => match p.construction {
            ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
                extrude = Some(GpuExtrude {
                    verts: line_polygon(&p.profile, norm.scale)?,
                    z_min: s(z_min_mm),
                    z_max: s(z_max_mm),
                });
                (OBJ_EXTRUDE, Vec3::ZERO, body_rot)
            }
            ProfileConstruction3::RevolveLocalX => {
                revolve = Some(compile_revolve(&p.profile, norm.scale)?);
                (OBJ_REVOLVE, Vec3::ZERO, body_rot)
            }
            ProfileConstruction3::PlanarOnly { .. } => {
                return Err("planar-only body has no 3D form to preview".into())
            }
        },
    };

    let surface = classify_surface(&b.surface.default)?;
    // Per-face responses, in the same order the revolved faces were built. A
    // film on a face of a solid is backed by that solid's own glass, so it is
    // baked against the body's index at each wavelength, not against air.
    let mut face_resp: Vec<Option<GpuResponse>> = Vec::new();
    let body_indices: Vec<f64> = if b.surface.overrides.is_empty() {
        Vec::new()
    } else {
        coat_lambdas
            .iter()
            .map(|&l| body_index_at(&b.material, catalog, l as f64).unwrap_or(1.0))
            .collect()
    };
    // The patches a per-face response can name, in the face order the preview
    // reports hits in: a revolved profile's edges, or a tube's four faces.
    let patches: Vec<oc_core::id::PatchId> = match &b.geometry {
        SolidGeometry3::Profile(_) if !b.surface.overrides.is_empty() => match &revolve {
            Some(rev) => rev.face_patch.clone(),
            // An extruded body reports every hit as face 0, so no response
            // could reach the face it was authored on.
            None => {
                return Err(
                    "per-face response on an extruded body (not yet in the f32 preview)".into(),
                )
            }
        },
        SolidGeometry3::Tube { .. } if !b.surface.overrides.is_empty() => (0
            ..=oc_core::geom3::primitives::tube_face::INNER)
            .map(|f| oc_core::id::PatchId(f as u64))
            .collect(),
        _ => Vec::new(),
    };
    for &patch in &patches {
        let two = b.surface.for_patch(patch);
        if std::ptr::eq(two, &b.surface.default) {
            face_resp.push(None);
            continue;
        }
        // A coated face's inside copy is the same stack met from behind.
        // The grid is baked from outside and serves a ray from inside as
        // well, because a lossless stack reflects and transmits the same
        // from either side at conjugate angles; the inside copy exists to
        // tell the reference which way round to walk the layers.
        let mirrored = match (&two.from_outside, &two.from_inside) {
            (SurfaceResponse::Coating { coating: a }, SurfaceResponse::Coating { coating: b }) => {
                a.seen_from_behind() == *b
            }
            _ => false,
        };
        if two.from_outside != two.from_inside && !mirrored {
            return Err("asymmetric per-face response (not yet in the f32 preview)".into());
        }
        match &two.from_outside {
            SurfaceResponse::Coating { coating } => {
                face_resp.push(Some(bake_coating(
                    coating,
                    ambient,
                    coat_lambdas,
                    &body_indices,
                )?));
            }
            SurfaceResponse::DielectricFresnel { .. } => face_resp.push(None),
            // A mirror face on a solid: the bore of a light pipe.
            SurfaceResponse::Mirror { .. } => face_resp.push(Some(compile_response(
                &two.from_outside,
                norm.scale,
                ambient,
                coat_lambdas,
            )?)),
            other => {
                return Err(format!(
                    "per-face {} response (not yet in the f32 preview)",
                    oc_core::build3::response_kind(other)
                ));
            }
        }
    }
    // Per-face responses are honoured for revolved bodies, which is what a
    // coated plate and a coated lens are, and for a tube, whose bore is the
    // reason it exists. Anything else carrying an override
    // is rejected by name rather than traced as if the override were not there:
    // a dichroic silently reduced to plain glass is worse than a dichroic that
    // says it cannot be previewed.
    if !b.surface.overrides.is_empty()
        && !matches!(
            b.geometry,
            SolidGeometry3::Profile(_) | SolidGeometry3::Tube { .. }
        )
    {
        return Err(
            "per-face response on a body other than a profile or tube (not yet in the f32 preview)"
                .into(),
        );
    }
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
        revolve,
        face_resp,
        index_const: 1.0,
        dp: [0.0; 6],
        alpha_per_mm,
        object: b.common.id.0,
        bounds: normalized_bounds(&b.geometry.aabb_world(&b.common.transform), norm),
    };
    set_dispersion(&mut gb, &b.material, catalog)?;
    Ok(gb)
}

/// A world box in the normalized frame, padded by 1e-4 of the scene's scale
/// (f32 rounding is 1e-7): large enough that no hit inside the true box falls
/// outside it, which is all a conservative bound owes.
fn normalized_bounds(bb: &oc_core::vec3::Aabb3, norm: &Norm) -> (Vec3, Vec3) {
    const PAD: f32 = 1e-4;
    (
        norm.point(bb.min) - Vec3::splat(PAD),
        norm.point(bb.max) + Vec3::splat(PAD),
    )
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
            Curve2::Conic(_)
            | Curve2::EvenAsphere { .. }
            | Curve2::Arc { .. }
            | Curve2::SagSegment { .. } => {
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

// ---------------------------------------------------------------------------
// Revolved profiles (§7.3–7.5)
// ---------------------------------------------------------------------------

/// One surface of a revolved profile, in the normalized frame. Its axis is local
/// `+X`, matching the profile authoring convention.
#[derive(Debug, Clone)]
pub enum RevFace {
    /// A conic or even-asphere surface of revolution. `axis_sign` is the sign of
    /// the profile conic's axis (`±X`). `is_conic` selects the exact quadratic
    /// over the safeguarded Newton solver. `inner_r` is `0` for a face that
    /// spans the axis and the facet's inner radius for an annular one (a
    /// Fresnel facet, [`Curve2::SagSegment`](oc_core::profile2::Curve2)).
    Sag {
        vertex_x: f32,
        axis_sign: f32,
        c: f32,
        k: f32,
        coeffs: [f32; 8],
        inner_r: f32,
        outer_r: f32,
        is_conic: bool,
    },
    /// A line segment revolved about `+X`: a cone, cylinder, or flat annulus. The
    /// endpoints are clipped to `r >= 0`.
    Frustum { x0: f32, y0: f32, x1: f32, y1: f32 },
    /// An axis-centred arc revolved about `+X`: a spherical zone.
    Arc {
        center_x: f32,
        r: f32,
        a0: f32,
        a1: f32,
    },
}

/// A profile revolved about local `+X`: its surfaces plus a sampled `(x, r)`
/// contour for the containment (media) test.
#[derive(Debug, Clone)]
pub struct GpuRevolve {
    pub faces: Vec<RevFace>,
    /// The profile edge each face came from, face for face. Not one to one
    /// with the edges: an edge below the axis makes no face (its mirror edge
    /// already did), and a polyline makes one per segment, so a per-face
    /// response is found through this and never by edge index.
    pub face_patch: Vec<oc_core::id::PatchId>,
    /// The meridian, sampled and closed, in `(x, r)` normalized coordinates. Used
    /// by point-in-polygon to classify a point's medium.
    pub contour: Vec<Vec2>,
    /// A many-edged profile's contour bucketed by height, with its local `x`
    /// range; `None` for an ordinary lens, which is scanned whole.
    pub index: Option<ContourIndex>,
}

/// The contour's segments bucketed by the heights they span, for a profile with
/// hundreds of edges (a Fresnel lens), plus the body's local `x` range.
///
/// Point-in-polygon counts the segments a horizontal probe at the point's
/// height crosses, and only a segment whose `y` range holds that height can be
/// crossed, so the one bucket holding it has every segment that counts: the
/// same parity as the whole scan, from a handful of segments. The `x` range
/// lets the face loop pass over a face whose radii the ray never reaches
/// inside the body's slab. The shader reads the same tables from the `verts`
/// arena (`buffers3`), so the two stay in lockstep.
#[derive(Debug, Clone, Default)]
pub struct ContourIndex {
    pub y0: f32,
    pub inv_h: f32,
    /// `(start, count)` into `items`, one per bucket.
    pub buckets: Vec<(u32, u32)>,
    /// Segment `i` joins contour points `i - 1` (wrapping) and `i`, the pair
    /// point-in-polygon visits at `i`.
    pub items: Vec<u32>,
    pub x_lo: f32,
    pub x_hi: f32,
}

/// Contours at least this long are indexed; a lens's hundred points are
/// scanned, exactly as before.
pub const CONTOUR_INDEX_MIN: usize = 256;

impl ContourIndex {
    /// Index a closed contour, or `None` for one short enough to scan.
    pub fn build(contour: &[Vec2]) -> Option<Self> {
        let n = contour.len();
        if n < CONTOUR_INDEX_MIN {
            return None;
        }
        let (mut y_lo, mut y_hi) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut x_lo, mut x_hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in contour {
            y_lo = y_lo.min(p.y);
            y_hi = y_hi.max(p.y);
            x_lo = x_lo.min(p.x);
            x_hi = x_hi.max(p.x);
        }
        let nb = (n / 4).clamp(1, 1024);
        let span = if y_hi > y_lo { y_hi - y_lo } else { 1.0 };
        let mut ix = Self {
            y0: y_lo,
            inv_h: nb as f32 / span,
            buckets: vec![(0, 0); nb],
            items: Vec::new(),
            x_lo,
            x_hi,
        };
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nb];
        for i in 0..n {
            let (a, b) = (contour[i], contour[if i == 0 { n - 1 } else { i - 1 }]);
            let (k0, k1) = (ix.bucket(a.y.min(b.y)), ix.bucket(a.y.max(b.y)));
            for list in &mut lists[k0..=k1] {
                list.push(i as u32);
            }
        }
        for (k, list) in lists.into_iter().enumerate() {
            ix.buckets[k] = (ix.items.len() as u32, list.len() as u32);
            ix.items.extend(list);
        }
        Some(ix)
    }

    /// The bucket holding height `y`. Monotone in `y`, which is what makes a
    /// segment's bucket range hold every height it spans.
    pub fn bucket(&self, y: f32) -> usize {
        let last = (self.buckets.len() - 1) as f32;
        ((y - self.y0) * self.inv_h).floor().clamp(0.0, last) as usize
    }
}

/// Even-asphere sag at radial coordinate `r`, in f32 (§7.4). Shared by the
/// contour sampler here and the intersector in `cpu3`.
pub fn even_asphere_sag(c: f32, k: f32, coeffs: &[f32; 8], r: f32) -> f32 {
    let r2 = r * r;
    let denom = 1.0 + (1.0 - (1.0 + k) * c * c * r2).max(0.0).sqrt();
    let mut sag = c * r2 / denom;
    let mut rp = r2 * r2;
    for &a in coeffs {
        sag += a * rp;
        rp *= r2;
    }
    sag
}

/// `d(sag)/dr`, analytic — the point of §7.4, so the asphere normal is exact.
pub fn even_asphere_dsag(c: f32, k: f32, coeffs: &[f32; 8], r: f32) -> f32 {
    let r2 = r * r;
    let s = (1.0 - (1.0 + k) * c * c * r2).max(0.0).sqrt();
    let mut d = if s > 0.0 { c * r / s } else { 0.0 };
    let mut rp = r2 * r;
    for (i, &a) in coeffs.iter().enumerate() {
        d += a * (2 * i + 4) as f32 * rp;
        rp *= r2;
    }
    d
}

fn compile_revolve(profile: &ClosedProfile2, scale: f64) -> Result<GpuRevolve, String> {
    use oc_core::profile2::Curve2;
    let mut faces = Vec::new();
    let mut face_patch = Vec::new();
    for e in &profile.edges {
        match &e.curve {
            Curve2::Line { a, b } => {
                if let Some(f) = revolve_line_face(*a, *b, scale) {
                    faces.push(f);
                }
            }
            Curve2::Polyline { points } => {
                for w in points.windows(2) {
                    if let Some(f) = revolve_line_face(w[0], w[1], scale) {
                        faces.push(f);
                    }
                }
            }
            Curve2::Conic(cp) => faces.push(sag_face(cp, &[0.0; 8], scale, true)?),
            Curve2::EvenAsphere { base, coeffs } => {
                faces.push(sag_face(base, coeffs, scale, false)?)
            }
            Curve2::SagSegment {
                base,
                coeffs,
                y0,
                y1,
            } => {
                // Only the part at r >= 0 is a surface; one entirely below the
                // axis is its mirror edge's surface already.
                let (lo, hi) = (y0.min(*y1), y0.max(*y1));
                if !(hi <= 0.0 && lo < 0.0) {
                    let is_conic = coeffs.iter().all(|a| *a == 0.0);
                    let mut face = sag_face(base, coeffs, scale, is_conic)?;
                    if let RevFace::Sag {
                        inner_r, outer_r, ..
                    } = &mut face
                    {
                        *inner_r = (lo.max(0.0) / scale) as f32;
                        *outer_r = (hi / scale) as f32;
                    }
                    faces.push(face);
                }
            }
            Curve2::Arc {
                center,
                radius_mm,
                a0,
                a1,
            } => {
                if center.y.abs() > 1e-9 * (1.0 + center.x.abs()) {
                    return Err("an off-axis arc revolves to a torus (not supported)".into());
                }
                faces.push(RevFace::Arc {
                    center_x: (center.x / scale) as f32,
                    r: (radius_mm / scale) as f32,
                    a0: *a0 as f32,
                    a1: *a1 as f32,
                });
            }
        }
        face_patch.resize(faces.len(), e.patch);
    }
    let contour = revolve_contour(profile, scale);
    Ok(GpuRevolve {
        faces,
        face_patch,
        index: ContourIndex::build(&contour),
        contour,
    })
}

fn sag_face(
    cp: &oc_core::profile2::ConicParams2,
    coeffs: &[f64; 8],
    scale: f64,
    is_conic: bool,
) -> Result<RevFace, String> {
    let axis = cp.axis.normalize();
    if axis.y.abs() > 1e-6 {
        return Err("an off-axis conic cannot revolve about +X".into());
    }
    let axis_sign = if axis.x >= 0.0 { 1.0 } else { -1.0 };
    let mut cn = [0.0f32; 8];
    for (i, cc) in coeffs.iter().enumerate() {
        cn[i] = (cc * scale.powi(2 * i as i32 + 3)) as f32;
    }
    Ok(RevFace::Sag {
        vertex_x: (cp.vertex.x / scale) as f32,
        axis_sign,
        // A curvature `1/R` scales like inverse length.
        c: (cp.c * scale) as f32,
        k: cp.k as f32,
        coeffs: cn,
        inner_r: 0.0,
        outer_r: (cp.half_ap / scale) as f32,
        is_conic,
    })
}

/// A line revolved about `+X`, clipped to `r >= 0`. A segment entirely below the
/// axis is skipped: its mirror above covers the same surface.
fn revolve_line_face(
    mut a: oc_core::vec::Vec2,
    mut b: oc_core::vec::Vec2,
    scale: f64,
) -> Option<RevFace> {
    if a.y < 0.0 && b.y < 0.0 {
        return None;
    }
    if a.y < 0.0 {
        let u = a.y / (a.y - b.y);
        a = oc_core::vec::Vec2::new(a.x + u * (b.x - a.x), 0.0);
    } else if b.y < 0.0 {
        let u = b.y / (b.y - a.y);
        b = oc_core::vec::Vec2::new(b.x + u * (a.x - b.x), 0.0);
    }
    Some(RevFace::Frustum {
        x0: (a.x / scale) as f32,
        y0: (a.y / scale) as f32,
        x1: (b.x / scale) as f32,
        y1: (b.y / scale) as f32,
    })
}

/// Sample the profile into a closed `(x, r)` contour, chained into loop order so
/// point-in-polygon can classify a revolved point's medium. Dense enough that the
/// chord error is far below the medium-probe epsilon.
fn revolve_contour(profile: &ClosedProfile2, scale: f64) -> Vec<Vec2> {
    use oc_core::profile2::Curve2;
    const N: usize = 48;
    let s = |v: f64| (v / scale) as f32;
    // Each edge as an ordered polyline in normalized (x, y).
    let mut segs: Vec<Vec<Vec2>> = Vec::new();
    for e in &profile.edges {
        let mut pts = Vec::new();
        match &e.curve {
            Curve2::Line { a, b } => {
                pts.push(Vec2::new(s(a.x), s(a.y)));
                pts.push(Vec2::new(s(b.x), s(b.y)));
            }
            Curve2::Polyline { points } => {
                for p in points {
                    pts.push(Vec2::new(s(p.x), s(p.y)));
                }
            }
            Curve2::Conic(cp) => sample_sag(cp, &[0.0; 8], scale, N, &mut pts),
            Curve2::EvenAsphere { base, coeffs } => sample_sag(base, coeffs, scale, N, &mut pts),
            Curve2::SagSegment {
                base,
                coeffs,
                y0,
                y1,
            } => {
                // A facet is one annular piece, so fewer samples hold the same
                // chord error a whole face gets from `N`.
                const N_SEG: usize = 12;
                for i in 0..=N_SEG {
                    let t = y0 + (y1 - y0) * i as f64 / N_SEG as f64;
                    let q = oc_core::geom3::profile::sag_segment_point(base, coeffs, t);
                    pts.push(Vec2::new(s(q.x), s(q.y)));
                }
            }
            Curve2::Arc {
                center,
                radius_mm,
                a0,
                a1,
            } => {
                for i in 0..=N {
                    let ang = a0 + (a1 - a0) * i as f64 / N as f64;
                    pts.push(Vec2::new(
                        s(center.x + radius_mm * ang.cos()),
                        s(center.y + radius_mm * ang.sin()),
                    ));
                }
            }
        }
        if pts.len() >= 2 {
            segs.push(pts);
        }
    }
    chain_loop(segs)
}

/// Sample a conic/asphere profile edge across its aperture, `x = vertex_x +
/// axis_sign·sag(|y|)`, appending normalized `(x, y)` points.
fn sample_sag(
    cp: &oc_core::profile2::ConicParams2,
    coeffs: &[f64; 8],
    scale: f64,
    n: usize,
    out: &mut Vec<Vec2>,
) {
    let axis_sign = if cp.axis.normalize().x >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let cn = (cp.c * scale) as f32;
    let mut cf = [0.0f32; 8];
    for (i, c) in coeffs.iter().enumerate() {
        cf[i] = (c * scale.powi(2 * i as i32 + 3)) as f32;
    }
    let vx = (cp.vertex.x / scale) as f32;
    let ha = (cp.half_ap / scale) as f32;
    for i in 0..=n {
        let y = -ha + 2.0 * ha * i as f32 / n as f32;
        let x = vx + axis_sign as f32 * even_asphere_sag(cn, cp.k as f32, &cf, y.abs());
        out.push(Vec2::new(x, y));
    }
}

/// Chain edge polylines into one closed loop by matching nearest endpoints.
fn chain_loop(mut segs: Vec<Vec<Vec2>>) -> Vec<Vec2> {
    if segs.is_empty() {
        return Vec::new();
    }
    let mut loop_pts = segs.swap_remove(0);
    while !segs.is_empty() {
        let tail = *loop_pts.last().unwrap();
        // The unused segment whose nearer endpoint is closest to the tail.
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        let mut best_rev = false;
        for (i, seg) in segs.iter().enumerate() {
            let df = (seg[0] - tail).length();
            let db = (*seg.last().unwrap() - tail).length();
            if df < best_d {
                best_d = df;
                best = i;
                best_rev = false;
            }
            if db < best_d {
                best_d = db;
                best = i;
                best_rev = true;
            }
        }
        let mut seg = segs.swap_remove(best);
        if best_rev {
            seg.reverse();
        }
        // Skip the duplicate join point.
        loop_pts.extend(seg.into_iter().skip(1));
    }
    loop_pts
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
        SurfaceResponse::ThinLens { .. } => {
            Err("thin lens on a body patch (only standalone surfaces carry it)".into())
        }
        SurfaceResponse::Grating { .. } => {
            Err("grating on a body patch (only standalone surfaces carry one)".into())
        }
        SurfaceResponse::Polarizer { .. } | SurfaceResponse::Retarder { .. } => {
            Err("polarizing element on a body patch (only standalone surfaces carry one)".into())
        }
        SurfaceResponse::Coating { .. } => {
            Err("thin-film coating on a body patch (not yet in the f32 preview)".into())
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

/// A dimensionless response scalar in f32: a constant, or a wavelength table.
/// The f32 twin of [`SpectralScalar`], sampled by the same rule.
#[derive(Debug, Clone, PartialEq)]
pub enum Spectral32 {
    Const(f32),
    /// `(lambda[i], values[i])` samples, wavelengths in µm and increasing, at
    /// least two of them (the compiler checks).
    Table {
        lambda: Vec<f32>,
        values: Vec<f32>,
        policy: OutOfRangePolicy,
    },
}

impl Spectral32 {
    /// The value at `lambda_um`, or `None` outside the table under
    /// [`OutOfRangePolicy::Error`]. Mirrors [`SpectralScalar::sample`] step for
    /// step: piecewise linear, the interval found by `partition_point`, the
    /// endpoint held (`Clamp`) or zero (`Zero`) outside the range.
    pub fn sample(&self, lambda_um: f32) -> Option<f32> {
        let (xs, ys, policy) = match self {
            Spectral32::Const(v) => return Some(*v),
            Spectral32::Table {
                lambda,
                values,
                policy,
            } => (lambda, values, *policy),
        };
        let n = xs.len();
        let x = lambda_um;
        if x < xs[0] {
            return match policy {
                OutOfRangePolicy::Error => None,
                OutOfRangePolicy::Clamp => Some(ys[0]),
                OutOfRangePolicy::Zero => Some(0.0),
            };
        }
        if x > xs[n - 1] {
            return match policy {
                OutOfRangePolicy::Error => None,
                OutOfRangePolicy::Clamp => Some(ys[n - 1]),
                OutOfRangePolicy::Zero => Some(0.0),
            };
        }
        let i = xs.partition_point(|&v| v <= x).saturating_sub(1).min(n - 2);
        let (x0, x1) = (xs[i], xs[i + 1]);
        // Two samples closer than one f32 step round to the same value; the
        // f64 guard against a zero-width interval covers that case too.
        let f = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
        Some(ys[i] + f * (ys[i + 1] - ys[i]))
    }
}

/// A surface response the preview supports. Its scalars are sampled at the
/// ray's wavelength when the ray arrives.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuResponse {
    Absorber,
    Mirror {
        r: Spectral32,
    },
    Dielectric {
        ar: Spectral32,
    },
    ExplicitRt {
        r: Spectral32,
        t: Spectral32,
    },
    /// Ideal thin lens; `f` is the focal length in NORMALIZED units
    /// (`focal_length_mm / norm.scale`), so the slope law `u' = u − h/f`
    /// works directly on normalized hit heights.
    ThinLens {
        f: f32,
    },
    /// A diffraction grating, grooves along the surface's local Y: `k` is
    /// the order times the groove density per micrometre, `eta` and `zero`
    /// the fractions sent into that order and the zero order.
    Grating {
        k: f32,
        eta: Spectral32,
        zero: Spectral32,
        transmit: bool,
    },
    /// A thin-film stack evaluated onto a grid: `COAT_ANGLES` rows, uniform in
    /// `sin2 theta`, of one `(R, T)` pair per entry in `lambdas`.
    CoatingGrid {
        lambdas: Vec<f32>,
        rt: Vec<[f32; 2]>,
        /// The reflection is stray light (an AR coat's residual) and so a
        /// ghost, rather than a designed path. See
        /// `oc_core::thinfilm::Coating::reflection_is_stray`.
        stray: bool,
    },
}

/// Bilinear lookup into a baked coating grid. The CPU mirror and the shader do
/// exactly this, on exactly these numbers.
///
/// `cos_i` is measured in `n_medium`, the medium the ray is actually in, and is
/// converted to the ambient-medium angle the grid is indexed on by the conserved
/// `n sin θ`. A ray's wavelength is normally one of the baked columns exactly, so
/// the wavelength weight comes out 0 or 1 and only the angle interpolates.
pub fn sample_coating_grid(
    lambdas: &[f32],
    rt: &[[f32; 2]],
    lambda_um: f32,
    cos_i: f32,
    n_medium: f32,
    n_ambient: f32,
) -> (f32, f32) {
    let n_l = lambdas.len();
    if n_l == 0 || rt.len() < n_l * COAT_ANGLES {
        return (0.0, 0.0);
    }
    // The interval containing `lambda_um`, held at either end.
    let mut l0 = 0usize;
    while l0 + 2 < n_l && lambdas[l0 + 1] < lambda_um {
        l0 += 1;
    }
    let l1 = (l0 + 1).min(n_l - 1);
    let span = lambdas[l1] - lambdas[l0];
    let tl = if span > 0.0 {
        ((lambda_um - lambdas[l0]) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // n sin(theta) is conserved, so carry it into the ambient medium the grid
    // is indexed on. Past grazing there it saturates, which is where a ray is
    // beyond total internal reflection at the outer face anyway.
    let sin_local = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    let sin_amb = if n_ambient > 0.0 {
        (n_medium * sin_local / n_ambient).clamp(0.0, 1.0)
    } else {
        sin_local
    };
    let fa = ((sin_amb * sin_amb) / 0.999).clamp(0.0, 1.0) * (COAT_ANGLES as f32 - 1.0);
    let a0 = fa.floor() as usize;
    let a1 = (a0 + 1).min(COAT_ANGLES - 1);
    let ta = fa - a0 as f32;

    let at = |a: usize, l: usize| rt[a * n_l + l];
    let mix =
        |x: [f32; 2], y: [f32; 2], t: f32| [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t];
    let lo = mix(at(a0, l0), at(a0, l1), tl);
    let hi = mix(at(a1, l0), at(a1, l1), tl);
    let v = mix(lo, hi, ta);
    (v[0].clamp(0.0, 1.0), v[1].clamp(0.0, 1.0))
}

/// A response scalar in f32, or the reason it cannot be previewed. A table
/// must satisfy the load-time contract (`Scene3::validate`): matched lengths,
/// at least two samples, finite, strictly increasing in wavelength. A scene
/// that skipped validation gets a named reject here, not a guess.
fn spectral(s: &SpectralScalar) -> Result<Spectral32, String> {
    let (lambda_um, values, interpolation, policy) = match s {
        SpectralScalar::Constant(v) => return Ok(Spectral32::Const(*v as f32)),
        SpectralScalar::Tabulated {
            lambda_um,
            values,
            interpolation,
            out_of_range,
        } => (lambda_um, values, *interpolation, *out_of_range),
    };
    // Linear is the only rule today. A new one must be mirrored in
    // `Spectral32::sample` and the shader before it can be previewed.
    match interpolation {
        Interpolation::Linear => {}
    }
    let bad = || "tabulated surface response with an invalid table (not previewed)".to_string();
    if lambda_um.len() != values.len()
        || lambda_um.len() < 2
        || lambda_um.windows(2).any(|w| w[1] <= w[0])
    {
        return Err(bad());
    }
    let lambda: Vec<f32> = lambda_um.iter().map(|&v| v as f32).collect();
    let values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    // Checked after the cast: a finite f64 past f32::MAX is not finite here.
    if lambda.iter().chain(&values).any(|v| !v.is_finite()) {
        return Err(bad());
    }
    Ok(Spectral32::Table {
        lambda,
        values,
        policy,
    })
}

fn compile_response(
    r: &SurfaceResponse,
    scale: f64,
    ambient: f64,
    coat_lambdas: &[f32],
) -> Result<GpuResponse, String> {
    Ok(match r {
        SurfaceResponse::Absorber => GpuResponse::Absorber,
        SurfaceResponse::ThinLens { focal_length_mm } => GpuResponse::ThinLens {
            f: (focal_length_mm / scale) as f32,
        },
        // Directions do not depend on the scene's scale: the shift in a ray's
        // direction cosine is per micrometre of wavelength.
        // The preview traces no polarization, so a grating that splits it
        // gives each ray the mean of its two curves.
        SurfaceResponse::Grating {
            lines_per_mm,
            order,
            efficiency,
            zero_order,
            transmissive,
            tm,
        } => GpuResponse::Grating {
            k: oc_core::grating::k_per_um(*lines_per_mm, *order) as f32,
            eta: match tm {
                Some(t) => spectral(&oc_core::grating::unpolarized(efficiency, &t.efficiency))?,
                None => spectral(efficiency)?,
            },
            zero: match tm {
                Some(t) => spectral(&oc_core::grating::unpolarized(zero_order, &t.zero_order))?,
                None => spectral(zero_order)?,
            },
            transmit: *transmissive,
        },
        SurfaceResponse::Mirror { reflectivity } => GpuResponse::Mirror {
            r: spectral(reflectivity)?,
        },
        SurfaceResponse::DielectricFresnel { ar_residual } => GpuResponse::Dielectric {
            ar: spectral(ar_residual)?,
        },
        SurfaceResponse::ExplicitRt {
            reflectance,
            transmittance,
        } => GpuResponse::ExplicitRt {
            r: spectral(reflectance)?,
            t: spectral(transmittance)?,
        },
        SurfaceResponse::Coating { coating } => {
            let far: Vec<f64> = coat_lambdas.iter().map(|_| ambient).collect();
            bake_coating(coating, ambient, coat_lambdas, &far)?
        }
        // The preview carries no polarization state, so all these elements can
        // be there is what they do to unpolarized light: a polarizer passes half
        // of it plus whatever leaks past the blocked axis, and a lossless
        // retarder passes all of it. Those are the right *numbers* under the
        // preview's declared model, and they are the reason a crossed pair looks
        // bright in the preview and goes dark in the reference. The component
        // view says which model produced a number.
        //
        // An element laminated in glass also has two faces, and the preview
        // takes them at normal incidence because it has no angle here. That is
        // the same shape of approximation as the rest of this arm: the right
        // number on axis and increasingly wrong as the part is tilted, where
        // the reference splits s from p and this cannot.
        SurfaceResponse::Polarizer {
            extinction,
            substrate_index,
            ..
        } => GpuResponse::ExplicitRt {
            r: spectral(&SpectralScalar::Constant(face_reflectance(*substrate_index)))?,
            t: spectral(&SpectralScalar::Constant(
                two_faces(*substrate_index) * 0.5 * (1.0 + extinction.on_axis().clamp(0.0, 1.0)),
            ))?,
        },
        SurfaceResponse::Retarder {
            substrate_index, ..
        } => GpuResponse::ExplicitRt {
            r: spectral(&SpectralScalar::Constant(face_reflectance(*substrate_index)))?,
            t: spectral(&SpectralScalar::Constant(two_faces(*substrate_index)))?,
        },
    })
}

/// One air-to-glass face's reflectance on axis, which is all the preview has
/// room for: a polarizing element carries no baked angle grid the way a coating
/// does.
fn face_reflectance(n: f64) -> f64 {
    let n = n.max(1.0);
    ((n - 1.0) / (n + 1.0)).powi(2)
}

/// What the pair of faces passes on axis. Both faces reflect the same, so this
/// is the square of one.
fn two_faces(n: f64) -> f64 {
    (1.0 - face_reflectance(n)).powi(2)
}

/// Angle samples in a baked coating grid, uniform in `sin2 theta` from 0 (on
/// axis) to 1 (grazing).
pub const COAT_ANGLES: usize = 64;
/// The most wavelengths a coating grid is baked at. Scenes carry a handful of
/// spectral lines; this only bounds a pathological one.
pub const COAT_MAX_LAMBDAS: usize = 64;

/// Evaluate a thin-film stack onto a grid: the wavelengths actually in flight,
/// crossed with `COAT_ANGLES` samples of the angle.
///
/// The transfer matrix is too much to run per ray in a shader, and baking is not
/// a compromise: the grid is the *same* table for the CPU mirror and for the
/// GPU, so those two stay bit-identical, which is the invariant the parity suite
/// rests on. The exact tracer never reads it, it runs the matrix per hit, so the
/// grid only ever has to be good enough to draw.
///
/// Baking at the scene's own wavelengths rather than over a fixed spectral range
/// is what makes this affordable: the preview recompiles the scene on every
/// frame of a drag, and a source emits a few lines, not a spectrum. A three-line
/// scene bakes 192 points instead of tens of thousands, and every ray then lands
/// exactly on a sample in wavelength, leaving angle as the only axis that
/// interpolates at all.
///
/// `n_far` is the medium behind the coating at each wavelength: ambient for a
/// standalone plate, and the body's own glass for a coated face of a solid.
/// Baking a film with air behind it and then using it on glass is wrong by tens
/// of percent, because what a stack reflects depends on what it is sitting on.
///
/// The angle axis is `sin²θ` measured **in the ambient medium**, not in whatever
/// medium the ray happens to be in. A coated face of a plate is crossed from
/// both sides, and the quantity both crossings share is `n sin θ`, which Snell
/// conserves; indexing on it lets one grid serve a ray arriving through air and
/// the same ray coming back out through the glass. It is also very nearly what
/// the band shift is linear in.
fn bake_coating(
    c: &oc_core::thinfilm::Coating,
    ambient: f64,
    lambdas: &[f32],
    n_far: &[f64],
) -> Result<GpuResponse, String> {
    if lambdas.is_empty() {
        return Err("a coating in a scene that emits no light".to_string());
    }
    let mut rt = Vec::with_capacity(lambdas.len() * COAT_ANGLES);
    for a in 0..COAT_ANGLES {
        let sin2 = a as f64 / (COAT_ANGLES - 1) as f64;
        // Stop just short of grazing: at exactly 90 degrees the admittances
        // degenerate and every coating reflects everything, which is true but
        // useless to interpolate toward.
        let cos_i = (1.0 - sin2 * 0.999).max(0.0).sqrt();
        for (i, &lam) in lambdas.iter().enumerate() {
            let far = n_far.get(i).copied().unwrap_or(ambient);
            let v = oc_core::thinfilm::coating_rt_at(c, ambient, far, lam as f64, cos_i)
                .ok_or_else(|| {
                    "coating whose optical constants are tabulated away from this scene's wavelengths"
                        .to_string()
                })?;
            rt.push([v.r() as f32, v.t() as f32]);
        }
    }
    Ok(GpuResponse::CoatingGrid {
        lambdas: lambdas.to_vec(),
        rt,
        stray: c.reflection_is_stray(),
    })
}

/// The distinct wavelengths a launch set carries, sorted, for baking coatings.
///
/// Deduplicated at 1e-9 um, far below any real line spacing and far above f32
/// noise, so two rays of "the same" wavelength share one column.
pub fn launch_lambdas(launch: &[oc_core::source3::EmittedRay3]) -> Vec<f32> {
    let mut v: Vec<f64> = launch.iter().map(|r| r.lambda_um).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    v.truncate(COAT_MAX_LAMBDAS);
    v.into_iter().map(|x| x as f32).collect()
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

fn compile_surface(
    s: &SurfaceObject3,
    norm: &Norm,
    ambient: f64,
    coat_lambdas: &[f32],
) -> Result<GpuSurface3, String> {
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
        front: compile_response(&s.response_front, norm.scale, ambient, coat_lambdas)?,
        back: compile_response(&s.response_back, norm.scale, ambient, coat_lambdas)?,
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
    /// The face in physical millimetres, as the scene has it. What a flux
    /// density divides by is computed from this on demand, not on every
    /// compile: the preview recompiles every frame of a drag, and a round
    /// face's exact rim areas are a cost nothing in that loop reads.
    pub geometry: DetectorGeometry3,
    /// What the readout means, mirrored so the preview reports the quantity
    /// the reference reports.
    pub measure: DetectorMeasure,
    /// Each direction cell's solid angle in steradians, empty unless the
    /// measure bins directions. Computed in f64 by the scene model, so the two
    /// tiers cannot disagree about the geometry, only about the flux.
    pub solid_angle_sr: Vec<f32>,
    pub object: u64,
}

fn compile_detector(d: &Detector3, norm: &Norm) -> Result<GpuDetector3, String> {
    let s = |v: f64| (v / norm.scale) as f32;
    let (shape, _total_mm2) = match &d.geometry {
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
    Ok(GpuDetector3 {
        rot,
        pos,
        shape,
        bins: d.bins,
        terminate: matches!(d.termination, DetectorTermination::Terminate),
        geometry: d.geometry.clone(),
        measure: d.measure,
        solid_angle_sr: d
            .direction_cell_solid_angles()
            .map(|v| v.into_iter().map(|w| w as f32).collect())
            .unwrap_or_default(),
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
    fn the_preview_names_every_switch_it_cannot_honour() {
        use oc_core::fidelity::{CoatingModel, Preset};
        let reference = preview_fidelity(&Preset::Reference.profile(None, true));
        assert_eq!(reference.refused.len(), 4, "{:?}", reference.refused);
        let msg = reference.message.unwrap();
        assert!(msg.starts_with("Preview: unpolarized coatings"), "{msg}");
        assert!(msg.contains("The f64 trace uses Reference"), "{msg}");
        assert_eq!(reference.runs.coating_model, CoatingModel::Unpolarized);
        // Standard is polarized in f64, so the preview says it runs unpolarized.
        let standard = preview_fidelity(&Preset::Standard.profile(None, false));
        assert_eq!(standard.refused, vec!["polarized coatings".to_string()]);
        // Fast asks for nothing the preview lacks.
        let fast = preview_fidelity(&Preset::Fast.profile(Some(0), false));
        assert!(fast.refused.is_empty() && fast.message.is_none());
        assert_eq!(fast.runs.coating_model, CoatingModel::Nominal);
    }

    #[test]
    fn an_extrusion_with_a_curved_edge_is_rejected_by_name() {
        // Extrusions revolve only line profiles; a curved edge needs the ruled
        // side-surface solver that has not landed, so it is rejected, not a chord.
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
                construction: ProfileConstruction3::ExtrudeLocalZ {
                    z_min_mm: -2.0,
                    z_max_mm: 2.0,
                },
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
        assert!(gpu.rejected[0].kind.contains("curved edge"));
    }

    fn plane_surface(scene: &mut Scene3, front: SurfaceResponse, back: SurfaceResponse) -> u64 {
        let id = scene.alloc_object_id();
        scene.surfaces.push(SurfaceObject3 {
            common: ObjectCommon::new(id, oc_core::id::LayerId(0)),
            geometry: SurfaceGeometry3::Plane {
                shape: PlanarShape2::Disc { radius_mm: 5.0 },
            },
            response_front: front,
            response_back: back,
        });
        id.0
    }

    fn table(lambda_um: &[f64], values: &[f64], out_of_range: OutOfRangePolicy) -> SpectralScalar {
        SpectralScalar::Tabulated {
            lambda_um: lambda_um.to_vec(),
            values: values.to_vec(),
            interpolation: Interpolation::Linear,
            out_of_range,
        }
    }

    #[test]
    fn splitters_dichroics_and_tabulated_responses_are_previewed() {
        // These used to be rejected, which dropped the object from the drag
        // preview. Every response scalar may now be a table.
        use OutOfRangePolicy::{Clamp, Error, Zero};
        let mut scene = Scene3::new();
        let splitter = SurfaceResponse::ExplicitRt {
            reflectance: SpectralScalar::Constant(0.5),
            transmittance: SpectralScalar::Constant(0.5),
        };
        plane_surface(&mut scene, splitter.clone(), splitter);
        let edge = [0.40, 0.53, 0.57, 0.80];
        let dichroic = SurfaceResponse::ExplicitRt {
            reflectance: table(&edge, &[1.0, 1.0, 0.0, 0.0], Clamp),
            transmittance: table(&edge, &[0.0, 0.0, 1.0, 1.0], Clamp),
        };
        plane_surface(&mut scene, dichroic.clone(), dichroic);
        let mirror = SurfaceResponse::Mirror {
            reflectivity: table(&[0.4, 0.7], &[0.9, 0.96], Error),
        };
        plane_surface(&mut scene, mirror, SurfaceResponse::Absorber);
        let window = SurfaceResponse::DielectricFresnel {
            ar_residual: table(&[0.4, 0.7], &[0.01, 0.03], Zero),
        };
        plane_surface(&mut scene, window.clone(), window);
        scene.validate().unwrap();

        let gpu = GpuScene3::build(&scene, BuildOpts::default());
        assert!(gpu.rejected.is_empty(), "{:?}", gpu.rejected);
        assert_eq!(gpu.surfaces.len(), 4);

        let GpuResponse::ExplicitRt { r, t } = &gpu.surfaces[0].back else {
            panic!("splitter: {:?}", gpu.surfaces[0].back);
        };
        assert_eq!((r.sample(0.55), t.sample(0.55)), (Some(0.5), Some(0.5)));

        let GpuResponse::ExplicitRt { r, t } = &gpu.surfaces[1].front else {
            panic!("dichroic: {:?}", gpu.surfaces[1].front);
        };
        assert_eq!((r.sample(0.45), t.sample(0.45)), (Some(1.0), Some(0.0)));
        assert_eq!((r.sample(0.95), t.sample(0.95)), (Some(0.0), Some(1.0)));
        let (r55, t55) = (r.sample(0.55).unwrap(), t.sample(0.55).unwrap());
        assert!((r55 - 0.5).abs() < 1e-6 && (t55 - 0.5).abs() < 1e-6);

        let GpuResponse::Mirror { r } = &gpu.surfaces[2].front else {
            panic!("mirror: {:?}", gpu.surfaces[2].front);
        };
        assert_eq!(r.sample(0.8), None);
        assert!(matches!(gpu.surfaces[2].back, GpuResponse::Absorber));

        let GpuResponse::Dielectric { ar } = &gpu.surfaces[3].front else {
            panic!("window: {:?}", gpu.surfaces[3].front);
        };
        assert_eq!(ar.sample(0.8), Some(0.0));
    }

    #[test]
    fn a_spectral_table_samples_like_the_f64_kernel() {
        let xs = [0.42, 0.5, 0.55, 0.61, 0.9];
        let ys = [0.1, 0.7, 0.2, 0.95, 0.3];
        for policy in [
            OutOfRangePolicy::Error,
            OutOfRangePolicy::Clamp,
            OutOfRangePolicy::Zero,
        ] {
            let s64 = table(&xs, &ys, policy);
            let s32 = spectral(&s64).unwrap();
            // A sweep across and beyond the table, plus every sample point.
            let sweep = (0..=280).map(|i| 0.3 + 0.7 * i as f64 / 280.0);
            for lam in sweep.chain(xs) {
                match (s64.sample(lam), s32.sample(lam as f32)) {
                    (None, None) => {}
                    (Some(a), Some(b)) => assert!(
                        (a as f32 - b).abs() < 1e-5,
                        "{policy:?} at {lam}: f64 {a} vs f32 {b}"
                    ),
                    (a, b) => panic!("{policy:?} at {lam}: f64 {a:?} vs f32 {b:?}"),
                }
            }
        }
    }

    #[test]
    fn an_invalid_response_table_is_rejected_by_name() {
        // `Scene3::validate` refuses these; a scene built without it must get a
        // named reject, not a panic or a made-up curve.
        let cases = [
            (vec![0.5], vec![0.3]),
            (vec![0.5, 0.6], vec![0.3]),
            (vec![0.6, 0.5], vec![0.3, 0.4]),
            (vec![0.5, f64::NAN], vec![0.3, 0.4]),
        ];
        for (lambda_um, values) in cases {
            let mut scene = Scene3::new();
            let bad = SurfaceResponse::Mirror {
                reflectivity: table(&lambda_um, &values, OutOfRangePolicy::Clamp),
            };
            let id = plane_surface(&mut scene, SurfaceResponse::Absorber, bad);
            let gpu = GpuScene3::build(&scene, BuildOpts::default());
            assert!(gpu.surfaces.is_empty(), "{lambda_um:?}");
            assert_eq!(gpu.rejected.len(), 1, "{lambda_um:?}");
            assert_eq!(gpu.rejected[0].object, id);
            assert!(gpu.rejected[0].kind.contains("invalid table"));
        }
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
