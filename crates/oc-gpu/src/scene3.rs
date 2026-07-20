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
//! asphere its base conic (§12.3). This first increment supports the analytic
//! bodies — sphere and box — with dielectric or absorbing surfaces; profile
//! bodies, cylinders, surfaces-of-revolution, zero-thickness surfaces, apertures,
//! and detectors are rejected here and arrive in later increments, each with a
//! named issue so the caller knows exactly what was left out.

use glam::{DQuat, DVec3, Mat3, Vec3};
use oc_core::material::{Cauchy, Material};
use oc_core::scene3::{
    AbsorptionModel, Body3, Scene3, SolidGeometry3, SurfaceResponse, TwoSidedSurfaceResponse,
};
use oc_core::source3::EmittedRay3;

/// Body geometry kinds the preview understands.
pub const OBJ_SPHERE: u32 = 0;
pub const OBJ_BOX: u32 = 1;

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

/// A body compiled for the preview: geometry, surface, and a rigid placement in
/// the *normalized* frame. Plain `f32` — the GPU `repr(C)` buffer layout lands
/// with the shader (PR 3D-08); this is the CPU oracle's working form.
#[derive(Debug, Clone, Copy)]
pub struct GpuBody3 {
    pub kind: u32,
    pub surface: u32,
    pub disp: u32,
    pub priority: i32,
    /// Local-to-world rotation (scale-free, so identical normalized).
    pub rot: Mat3,
    /// Normalized translation `(t_world − origin) / scale`.
    pub pos: Vec3,
    /// Sphere: `(radius, 0, 0)`; box: half-extents. Normalized.
    pub params: Vec3,
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
        let mut rejected = Vec::new();

        for b in &scene.bodies {
            match compile_body(b, &norm, &scene.catalog) {
                Ok(gb) => bodies.push(gb),
                Err(kind) => rejected.push(Reject3 {
                    object: b.common.id.0,
                    kind,
                }),
            }
        }

        // v2-native object kinds the CPU oracle does not yet trace. Named so the
        // preview badge can say the scene is not fully shown, rather than quietly
        // dropping them (§3.5).
        for s in &scene.surfaces {
            rejected.push(Reject3 {
                object: s.common.id.0,
                kind: "zero-thickness surface (not yet in the f32 preview)".into(),
            });
        }
        for a in &scene.apertures {
            rejected.push(Reject3 {
                object: a.common.id.0,
                kind: "aperture (not yet in the f32 preview)".into(),
            });
        }
        for d in &scene.detectors {
            rejected.push(Reject3 {
                object: d.common.id.0,
                kind: "detector (not yet in the f32 preview)".into(),
            });
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
        _ => None,
    }
}

/// Compile one body, or return the reason it cannot be traced exactly.
fn compile_body(
    b: &Body3,
    norm: &Norm,
    catalog: &oc_core::material::GlassCatalog,
) -> Result<GpuBody3, String> {
    let (kind, params_mm) = match &b.geometry {
        SolidGeometry3::Sphere { radius_mm } => (OBJ_SPHERE, DVec3::new(*radius_mm, 0.0, 0.0)),
        SolidGeometry3::Box { half_extents_mm } => (OBJ_BOX, *half_extents_mm),
        SolidGeometry3::Cylinder { .. } => {
            return Err("cylinder (not yet in the f32 preview)".into())
        }
        SolidGeometry3::Profile(_) => {
            return Err("profile body (not yet in the f32 preview)".into())
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

    let q = DQuat::from_xyzw(
        b.common.transform.rot_xyzw[0],
        b.common.transform.rot_xyzw[1],
        b.common.transform.rot_xyzw[2],
        b.common.transform.rot_xyzw[3],
    );
    let rot = Mat3::from_quat(q.as_quat());
    let pos = norm.point(b.common.transform.pos);
    let params = (params_mm / norm.scale).as_vec3();

    let mut gb = GpuBody3 {
        kind,
        surface,
        disp: DISP_CONST,
        priority: b.priority,
        rot,
        pos,
        params,
        index_const: 1.0,
        dp: [0.0; 6],
        alpha_per_mm,
        object: b.common.id.0,
    };
    set_dispersion(&mut gb, &b.material, catalog)?;
    Ok(gb)
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
    fn a_cylinder_is_rejected_by_name() {
        let mut scene = Scene3::new();
        let id = scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, oc_core::id::LayerId(0));
        common.transform = oc_core::vec3::Isometry3::IDENTITY;
        scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Cylinder {
                radius_mm: 2.0,
                half_length_mm: 5.0,
                axis_local: Vec3::X.as_dvec3(),
            },
            material: Material::AbbePoint { nd: 1.5, vd: 64.0 },
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        let gpu = GpuScene3::build(&scene, BuildOpts::default());
        assert!(gpu.bodies.is_empty());
        assert_eq!(gpu.rejected.len(), 1);
        assert_eq!(gpu.rejected[0].object, id.0);
        assert!(gpu.rejected[0].kind.contains("cylinder"));
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
