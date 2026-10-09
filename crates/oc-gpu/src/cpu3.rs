//! The f32 CPU 3D tracer: the oracle the WGSL shader is validated against (§12,
//! PR 3D-07).
//!
//! It mirrors the f64 reference ([`oc_trace::trace3`]) in miniature — nearest-hit
//! transport, media by containment plus priority, vector Snell with scalar
//! Fresnel and exact TIR, deterministic reflected/transmitted branches — but in
//! f32 and in the normalized frame of [`GpuScene3`], so it does exactly what the
//! shader will do and can stand in for it wherever no GPU is present.
//!
//! It traces the objects the compiler admits: the analytic bodies (sphere, box,
//! cylinder), extruded line profiles (prisms), revolved profiles (conic and
//! even-asphere surfaces of revolution, plus their frustum rims), zero-thickness
//! surfaces (mirror, absorber, scalar dielectric, explicit R/T, thin lens, with
//! constant or tabulated response scalars), finite aperture masks, and area
//! detectors. The conic of revolution is an exact quadratic and the asphere a
//! safeguarded Newton with an analytic derivative (§7.3–7.4). What the compiler
//! still cannot trace exactly it rejects by name — an off-axis arc, a
//! curved-edge extrusion, tabulated volume absorption — rather than guessing.

use glam::{DVec3, Vec2, Vec3};

use crate::scene3::{
    even_asphere_dsag, even_asphere_sag, GpuAperture3, GpuBody3, GpuDetector3, GpuExtrude,
    GpuResponse, GpuRevolve, GpuScene3, GpuSurface3, RevFace, Spectral32, OBJ_BOX, OBJ_CYLINDER,
    OBJ_EXTRUDE, OBJ_REVOLVE, OBJ_SPHERE, OBJ_TUBE, SURF_ABSORBER,
};

/// Segment flag bits (shared meaning with the 2D path).
pub const SEG_TIR: u32 = 2;
pub const SEG_GHOST: u32 = 4;

/// A traced segment in world millimetres.
#[derive(Debug, Clone, Copy)]
pub struct SegOut3 {
    pub a: DVec3,
    pub b: DVec3,
    pub flux: f32,
    pub lambda_um: f32,
    pub flags: u32,
}

impl SegOut3 {
    pub fn is_tir(&self) -> bool {
        self.flags & SEG_TIR != 0
    }
    pub fn is_ghost(&self) -> bool {
        self.flags & SEG_GHOST != 0
    }
}

/// An area detector's accumulated readout across every launch ray.
#[derive(Debug, Clone)]
pub struct DetReadout3 {
    pub object: u64,
    pub bins: [usize; 2],
    /// The face in physical millimetres, for [`DetReadout3::density`].
    pub geometry: oc_core::scene3::DetectorGeometry3,
    /// Incident flux per cell, row-major over `bins`.
    pub incident: Vec<f32>,
    pub hits: u64,
    pub total: f32,
    /// Flux per **direction** cell, row-major over `bins` as
    /// `sector * rings + ring`, filled only under
    /// [`DetectorMeasure::Intensity`].
    pub direction: Vec<f32>,
    /// Those cells' solid angles in steradians, alongside it.
    pub solid_angle_sr: Vec<f32>,
    /// Flux that arrived past the intensity cone and landed in no cell.
    pub outside: f32,
}

impl DetReadout3 {
    /// Relative flux density per pixel: its flux over the part of it on the
    /// face, and 0 for a pixel wholly off it. The areas are the scene model's
    /// own, so the preview and the reference cannot drift on what a pixel is.
    pub fn density(&self) -> Vec<f32> {
        let covered = self
            .geometry
            .covered_cell_areas_mm2(self.bins)
            .unwrap_or_default();
        self.incident
            .iter()
            .zip(&covered)
            .map(|(f, a)| if *a > 0.0 { (*f as f64 / a) as f32 } else { 0.0 })
            .collect()
    }

    /// Radiant intensity per direction cell, flux per steradian, or `None`
    /// unless the detector was asked to measure it.
    pub fn radiant_intensity(&self) -> Option<Vec<f32>> {
        (!self.direction.is_empty()).then(|| {
            self.direction
                .iter()
                .zip(&self.solid_angle_sr)
                .map(|(f, w)| if *w > 0.0 { f / w } else { 0.0 })
                .collect()
        })
    }
}

/// The result of an f32 preview trace.
#[derive(Debug, Clone)]
pub struct Cpu3Result {
    /// Segments per launch ray, in world millimetres.
    pub segments: Vec<Vec<SegOut3>>,
    /// One readout per compiled detector, in scene order.
    pub detectors: Vec<DetReadout3>,
}

impl Cpu3Result {
    /// A detector readout by object id.
    pub fn detector(&self, object: u64) -> Option<&DetReadout3> {
        self.detectors.iter().find(|d| d.object == object)
    }
}

/// Self-hit offset and medium-probe distance, in the normalized frame. f32 across
/// a ~unit-scale scene resolves this comfortably.
const EPS: f32 = 1e-4;
/// Branch-count safety cap per launch ray.
const MAX_BRANCHES: usize = 4096;

#[derive(Clone, Copy)]
struct Work {
    o: Vec3,
    d: Vec3,
    flux: f32,
    depth: u32,
    gdepth: u32,
    /// Index and attenuation of the medium this branch travels in.
    medium_n: f32,
    medium_alpha: f32,
    flags: u32,
}

/// Trace every launch ray, returning segments and detector readouts.
pub fn trace(scene: &GpuScene3, max_depth: u32) -> Cpu3Result {
    let mut detectors = blank_readouts(scene);
    let segments = scene
        .rays
        .iter()
        .map(|r| trace_one(scene, *r, max_depth, &mut detectors))
        .collect();

    Cpu3Result {
        segments,
        detectors,
    }
}

/// Trace segments only, for the many call sites that do not read detectors.
pub fn trace_segments(scene: &GpuScene3, max_depth: u32) -> Vec<Vec<SegOut3>> {
    trace(scene, max_depth).segments
}

/// The segments of the launch rays in `rays` only, per ray, in order. A ray's
/// segments do not depend on any other ray, so the ranges of a split, one
/// after another, are [`trace_segments`] of the whole scene.
pub fn trace_segments_range(
    scene: &GpuScene3,
    max_depth: u32,
    rays: std::ops::Range<usize>,
) -> Vec<Vec<SegOut3>> {
    let mut detectors = blank_readouts(scene);
    scene.rays[rays]
        .iter()
        .map(|r| trace_one(scene, *r, max_depth, &mut detectors))
        .collect()
}

fn blank_readouts(scene: &GpuScene3) -> Vec<DetReadout3> {
    scene
        .detectors
        .iter()
        .map(|d| DetReadout3 {
            object: d.object,
            bins: d.bins,
            geometry: d.geometry.clone(),
            incident: vec![0.0; d.bins[0].max(1) * d.bins[1].max(1)],
            hits: 0,
            total: 0.0,
            direction: vec![0.0; d.solid_angle_sr.len()],
            solid_angle_sr: d.solid_angle_sr.clone(),
            outside: 0.0,
        })
        .collect()
}

fn trace_one(
    scene: &GpuScene3,
    ray: crate::scene3::GpuRay3,
    max_depth: u32,
    detectors: &mut [DetReadout3],
) -> Vec<SegOut3> {
    let (o, d, flux, lambda) = (ray.origin, ray.dir, ray.flux, ray.lambda_um);
    let min_flux = scene.flux_eps * flux;
    let mut out = Vec::new();

    let (n0, a0) = region_optics(scene, o, lambda);
    let mut stack = vec![Work {
        o,
        d,
        flux,
        depth: 0,
        gdepth: 0,
        medium_n: n0,
        medium_alpha: a0,
        flags: 0,
    }];

    let mut branches = 0usize;
    while let Some(w) = stack.pop() {
        branches += 1;
        if branches > MAX_BRANCHES || w.depth >= max_depth || w.flux < min_flux {
            continue;
        }

        let Some(ev) = nearest_event(scene, w.o, w.d) else {
            let end = w.o + w.d * scene.bounds_norm;
            out.push(seg(scene, &w, end, w.flux, lambda));
            continue;
        };

        let point = w.o + w.d * ev.t;
        let flux_end = w.flux * beer_lambert(w.medium_alpha, scene.norm.world_len(ev.t));
        out.push(seg(scene, &w, point, flux_end, lambda));

        match ev.kind {
            EvKind::Body { body, normal, face } => interact_body(
                scene, &mut stack, &w, body, face, point, normal, flux_end, lambda, max_depth,
                min_flux,
            ),
            EvKind::Surface { surf, normal } => interact_surface(
                scene, &mut stack, &w, surf, point, normal, flux_end, lambda, max_depth, min_flux,
            ),
            EvKind::Aperture => { /* blocked: the branch ends here */ }
            EvKind::Detector { det, uv } => {
                record_detector(&mut detectors[det], &scene.detectors[det], uv, w.d, flux_end);
                if !scene.detectors[det].terminate {
                    // Probe: continue undeviated, just past the face.
                    push(
                        &mut stack,
                        Work {
                            o: point + w.d * EPS,
                            ..w
                        },
                        max_depth,
                        min_flux,
                    );
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Interactions
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn interact_body(
    scene: &GpuScene3,
    stack: &mut Vec<Work>,
    w: &Work,
    body: usize,
    face: usize,
    point: Vec3,
    hit_normal: Vec3,
    flux_end: f32,
    lambda: f32,
    max_depth: u32,
    min_flux: f32,
) {
    // An absorbing body swallows the ray at its surface, unless the face it hit
    // carries a response of its own: the silvered bore of a black tube.
    let own_face = matches!(scene.bodies[body].face_resp.get(face), Some(Some(_)));
    if scene.bodies[body].surface == SURF_ABSORBER && !own_face {
        return;
    }
    let n = if hit_normal.dot(w.d) > 0.0 {
        -hit_normal
    } else {
        hit_normal
    };
    // A mirror face reflects what its reflectivity says and absorbs the rest.
    // A table with no value at this wavelength ends the branch, as it does on a
    // standalone mirror.
    if let Some(Some(GpuResponse::Mirror { r })) = scene.bodies[body].face_resp.get(face) {
        let Some(r) = r.sample(lambda) else {
            return;
        };
        let dir = reflect(w.d, n);
        push(
            stack,
            Work {
                o: point + dir * EPS,
                d: dir,
                flux: flux_end * r.clamp(0.0, 1.0),
                depth: w.depth + 1,
                gdepth: w.gdepth,
                medium_n: w.medium_n,
                medium_alpha: w.medium_alpha,
                flags: w.flags & !SEG_TIR,
            },
            max_depth,
            min_flux,
        );
        return;
    }
    let (n1, _a1) = region_optics(scene, point + n * EPS, lambda);
    let (n2, a2) = region_optics(scene, point - n * EPS, lambda);

    // A coated face of a solid: a real plate's film sits on one face and plain
    // glass on the other, so the response is per face rather than per body. R
    // and T come from the baked grid at this ray's own angle, and the
    // transmitted branch still refracts, because unlike a zero-thickness plate
    // this interface really does change the medium.
    if let Some(Some(GpuResponse::CoatingGrid { lambdas, rt, rt_inside, stray })) =
        scene.bodies[body].face_resp.get(face)
    {
        let cos_i = (-n.dot(w.d)).clamp(0.0, 1.0);
        // A ray from inside the body reads the grid baked from inside, when
        // the face has one (an absorbing stack does).
        let from_inside = hit_normal.dot(w.d) > 0.0;
        let grid = match rt_inside {
            Some(g) if from_inside => g,
            _ => rt,
        };
        let (r, t) = crate::scene3::sample_coating_grid(
            lambdas,
            grid,
            lambda,
            cos_i,
            w.medium_n,
            scene.ambient,
        );
        // Past the critical angle nothing gets through whatever the stack, and
        // the grid cannot say so: it is baked from outside and saturates at
        // grazing. So total internal reflection is taken the way bare glass
        // takes it below, the whole ray and a primary path, which is what the
        // reference's transfer matrix returns for a lossless coating.
        let Some(td) = refract(w.d, n, n1, n2) else {
            let dir = reflect(w.d, n);
            push(
                stack,
                Work {
                    o: point + dir * EPS,
                    d: dir,
                    flux: flux_end,
                    depth: w.depth + 1,
                    gdepth: w.gdepth,
                    medium_n: w.medium_n,
                    medium_alpha: w.medium_alpha,
                    flags: w.flags | SEG_TIR,
                },
                max_depth,
                min_flux,
            );
            return;
        };
        // An AR coat's reflection is a ghost, exactly as bare glass's is below:
        // followed only with ghosts on, and within the ghost budget. Any other
        // stack's reflection is the path it was designed for.
        let follow = !*stray || (scene.ghosts && w.gdepth < scene.max_ghost);
        if r > 0.0 && follow {
            let dir = reflect(w.d, n);
            let (gdepth, flags) = if *stray {
                (w.gdepth + 1, (w.flags & !SEG_TIR) | SEG_GHOST)
            } else {
                (w.gdepth, w.flags & !SEG_TIR)
            };
            push(
                stack,
                Work {
                    o: point + dir * EPS,
                    d: dir,
                    flux: flux_end * r,
                    depth: w.depth + 1,
                    gdepth,
                    medium_n: w.medium_n,
                    medium_alpha: w.medium_alpha,
                    flags,
                },
                max_depth,
                min_flux,
            );
        }
        if t > 0.0 {
            // Snell across the boundary, already solved above.
            push(
                stack,
                Work {
                    o: point + td * EPS,
                    d: td,
                    flux: flux_end * t,
                    depth: w.depth + 1,
                    gdepth: w.gdepth,
                    medium_n: n2,
                    medium_alpha: a2,
                    flags: w.flags & !SEG_TIR,
                },
                max_depth,
                min_flux,
            );
        }
        return;
    }
    if scene.bodies[body].surface == SURF_ABSORBER {
        return;
    }

    match refract(w.d, n, n1, n2) {
        None => {
            let dir = reflect(w.d, n);
            push(
                stack,
                Work {
                    o: point + dir * EPS,
                    d: dir,
                    flux: flux_end,
                    depth: w.depth + 1,
                    gdepth: w.gdepth,
                    medium_n: w.medium_n,
                    medium_alpha: w.medium_alpha,
                    flags: w.flags | SEG_TIR,
                },
                max_depth,
                min_flux,
            );
        }
        Some(td) => {
            let cos_i = (-n.dot(w.d)).clamp(0.0, 1.0);
            let r = reflectance(cos_i, n1, n2).clamp(0.0, 1.0);
            push(
                stack,
                Work {
                    o: point + td * EPS,
                    d: td,
                    flux: flux_end * (1.0 - r),
                    depth: w.depth + 1,
                    gdepth: w.gdepth,
                    medium_n: n2,
                    medium_alpha: a2,
                    flags: w.flags & !SEG_TIR, // transmitted: clear the TIR mark
                },
                max_depth,
                min_flux,
            );
            if scene.ghosts && w.gdepth < scene.max_ghost && r > 0.0 {
                let dir = reflect(w.d, n);
                push(
                    stack,
                    Work {
                        o: point + dir * EPS,
                        d: dir,
                        flux: flux_end * r,
                        depth: w.depth + 1,
                        gdepth: w.gdepth + 1,
                        medium_n: w.medium_n,
                        medium_alpha: w.medium_alpha,
                        flags: (w.flags & !SEG_TIR) | SEG_GHOST, // Fresnel ghost, not TIR
                    },
                    max_depth,
                    min_flux,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn interact_surface(
    scene: &GpuScene3,
    stack: &mut Vec<Work>,
    w: &Work,
    surf: usize,
    point: Vec3,
    normal: Vec3,
    flux_end: f32,
    lambda: f32,
    max_depth: u32,
    min_flux: f32,
) {
    let s: &GpuSurface3 = &scene.surfaces[surf];
    // Front is the local +Z side; incident on the front when running against it.
    let from_front = w.d.dot(normal) < 0.0;
    let resp = if from_front { &s.front } else { &s.back };
    let n_oppose = if from_front { normal } else { -normal };

    // A zero-thickness surface has the same medium on both sides (§6.6): no
    // refraction, transmission continues undeviated.
    let transmit = |stack: &mut Vec<Work>, flux: f32| {
        push(
            stack,
            Work {
                o: point + w.d * EPS,
                d: w.d,
                flux,
                depth: w.depth + 1,
                gdepth: w.gdepth,
                medium_n: w.medium_n,
                medium_alpha: w.medium_alpha,
                flags: w.flags & !SEG_TIR, // TIR is per-interaction, not inherited
            },
            max_depth,
            min_flux,
        );
    };
    let reflect_branch = |stack: &mut Vec<Work>, flux: f32, ghost: bool| {
        let dir = reflect(w.d, n_oppose);
        let mut flags = w.flags & !SEG_TIR; // specular reflection is not TIR
        let mut gdepth = w.gdepth;
        if ghost {
            flags |= SEG_GHOST;
            gdepth += 1;
        }
        push(
            stack,
            Work {
                o: point + dir * EPS,
                d: dir,
                flux,
                depth: w.depth + 1,
                gdepth,
                medium_n: w.medium_n,
                medium_alpha: w.medium_alpha,
                flags,
            },
            max_depth,
            min_flux,
        );
    };

    match resp {
        GpuResponse::Absorber => { /* terminate */ }
        GpuResponse::Mirror { r } => {
            // A table with no value at this wavelength ends the branch (tracer3
            // names it as unsupported physics).
            let Some(r) = r.sample(lambda) else {
                return;
            };
            reflect_branch(stack, flux_end * r.clamp(0.0, 1.0), false);
        }
        GpuResponse::Dielectric { ar } => {
            // Same medium both sides: Fresnel at n1 = n2 is ~0, so the transmitted
            // ray is the whole beam undeviated; the AR-scaled sliver reflects.
            let cos_i = (-n_oppose.dot(w.d)).clamp(0.0, 1.0);
            let ar = ar_at(ar, lambda);
            let r = (reflectance(cos_i, w.medium_n, w.medium_n) * ar).clamp(0.0, 1.0);
            transmit(stack, flux_end * (1.0 - r));
            if scene.ghosts && w.gdepth < scene.max_ghost && r > 0.0 {
                reflect_branch(stack, flux_end * r, true);
            }
        }
        GpuResponse::ExplicitRt { r, t } => {
            // A beam splitter or dichroic: both branches are primary paths, as in
            // tracer3. Each keeps the parent's ghost state and ghost depth and
            // is followed whether or not ghosts are on. Either scalar missing at
            // this wavelength ends the branch.
            let (Some(r), Some(t)) = (r.sample(lambda), t.sample(lambda)) else {
                return;
            };
            let (r, t) = (r.clamp(0.0, 1.0), t.clamp(0.0, 1.0));
            // Reflected first, like tracer3, so both pop in the same order.
            if r > 0.0 {
                reflect_branch(stack, flux_end * r, false);
            }
            if t > 0.0 {
                transmit(stack, flux_end * t);
            }
        }
        GpuResponse::CoatingGrid { lambdas, rt, stray, .. } => {
            // A real thin-film stack, read off the grid that was baked from the
            // transfer matrix when the scene compiled. Both branches are primary
            // paths, exactly as for `ExplicitRt`, unless the stack is an AR coat,
            // whose reflection is a ghost like bare glass's. `R` and `T` depend
            // on the angle the ray arrived at.
            let cos_i = (-n_oppose.dot(w.d)).clamp(0.0, 1.0);
            let (r, t) = crate::scene3::sample_coating_grid(
                lambdas,
                rt,
                lambda,
                cos_i,
                w.medium_n,
                scene.ambient,
            );
            let follow = !*stray || (scene.ghosts && w.gdepth < scene.max_ghost);
            if r > 0.0 && follow {
                reflect_branch(stack, flux_end * r, *stray);
            }
            if t > 0.0 {
                transmit(stack, flux_end * t);
            }
        }
        GpuResponse::Grating {
            k,
            eta,
            zero,
            transmit,
        } => {
            // The grating equation in the surface's frame (mirrors tracer3):
            // the working order shifts the direction cosine across the
            // grooves by k·λ; the zero order is the mirror or the straight
            // beam; the rest is absorbed.
            let em = eta.sample(lambda).unwrap_or(0.0).clamp(0.0, 1.0);
            let e0 = zero.sample(lambda).unwrap_or(0.0).clamp(0.0, 1.0);
            let over = (em + e0).max(1.0);
            let rt = s.rot.transpose();
            let dl = rt * w.d;
            let tx = dl.x + k * lambda;
            let t2 = tx * tx + dl.y * dl.y;
            let go = |stack: &mut Vec<Work>, dir: Vec3, flux: f32| {
                push(
                    stack,
                    Work {
                        o: point + dir * EPS,
                        d: dir,
                        flux,
                        depth: w.depth + 1,
                        gdepth: w.gdepth,
                        medium_n: w.medium_n,
                        medium_alpha: w.medium_alpha,
                        flags: w.flags & !SEG_TIR,
                    },
                    max_depth,
                    min_flux,
                );
            };
            if em > 0.0 && t2 < 1.0 {
                let ahead = if dl.z >= 0.0 { 1.0 } else { -1.0 };
                let nz = (1.0 - t2).sqrt();
                let z = if *transmit { ahead * nz } else { -ahead * nz };
                let dir = (s.rot * Vec3::new(tx, dl.y, z)).normalize();
                go(stack, dir, flux_end * em / over);
            }
            if e0 > 0.0 {
                let dir = if *transmit { w.d } else { reflect(w.d, n_oppose) };
                go(stack, dir, flux_end * e0 / over);
            }
        }
        GpuResponse::ThinLens { f } => {
            // Ideal-lens slope law u' = u − h/f in the surface's local frame
            // (direction-aware; mirrors tracer3's f64 branch). Full
            // transmission, no reflection, no absorption.
            let f = *f;
            let rt = s.rot.transpose();
            let pl = rt * (point - s.pos);
            let dl = rt * w.d;
            let dir = if dl.z.abs() < 1e-6 || f == 0.0 {
                w.d
            } else {
                s.rot
                    * Vec3::new(
                        dl.x - dl.z.abs() * pl.x / f,
                        dl.y - dl.z.abs() * pl.y / f,
                        dl.z,
                    )
                    .normalize()
            };
            push(
                stack,
                Work {
                    o: point + dir * EPS,
                    d: dir,
                    flux: flux_end,
                    depth: w.depth + 1,
                    gdepth: w.gdepth,
                    medium_n: w.medium_n,
                    medium_alpha: w.medium_alpha,
                    flags: w.flags & !SEG_TIR,
                },
                max_depth,
                min_flux,
            );
        }
    }
}

/// The AR residual at `lambda`, clamped to `[0, 1]`. A table with no value there
/// leaves the Fresnel reflection unscaled (`1`), as tracer3 does.
fn ar_at(ar: &Spectral32, lambda: f32) -> f32 {
    ar.sample(lambda).unwrap_or(1.0).clamp(0.0, 1.0)
}

fn record_detector(out: &mut DetReadout3, d: &GpuDetector3, uv: Vec2, dir: Vec3, flux: f32) {
    out.hits += 1;
    out.total += flux;
    if let Some(cell) = d.shape.cell(uv, d.bins) {
        out.incident[cell] += flux;
    }
    // The angular histogram, when the detector asked for one. `dir` is the
    // world direction; `rot` is orthonormal, so its transpose takes it to the
    // face's own frame.
    if d.measure.cone_rad().is_some() {
        let local = (d.rot.transpose() * dir).normalize_or_zero();
        match direction_cell(d, local) {
            Some(cell) => out.direction[cell] += flux,
            None => out.outside += flux,
        }
    }
}

/// The direction cell an arrival falls in, mirroring `Detector3::direction_cell`
/// in f32. The ring layout and the solid angles belong to the scene model, so
/// only the arithmetic is repeated here, never the geometry.
fn direction_cell(d: &GpuDetector3, dir_local: Vec3) -> Option<usize> {
    let (cone, rings, sectors) = d.measure.direction_grid()?;
    let cone = cone as f32;
    if d.measure.direction_layout() == Some(oc_core::scene3::DirectionLayout::Grid) {
        let (nu, nv) = (rings, sectors);
        let z = dir_local.z.abs();
        let (au, av) = (dir_local.x.atan2(z), dir_local.y.atan2(z));
        if au.abs() > cone || av.abs() > cone {
            return None;
        }
        let cell = |a: f32, n: usize| (((a + cone) / (2.0 * cone) * n as f32) as usize).min(n - 1);
        return Some(cell(av, nv) * nu + cell(au, nu));
    }
    let theta = dir_local.z.abs().clamp(0.0, 1.0).acos();
    if theta > cone {
        return None;
    }
    let ring = ((theta / cone * rings as f32) as usize).min(rings - 1);
    let tau = std::f32::consts::TAU;
    let phi = dir_local.y.atan2(dir_local.x).rem_euclid(tau);
    let sector = ((phi / tau * sectors as f32) as usize).min(sectors - 1);
    Some(sector * rings + ring)
}

fn push(stack: &mut Vec<Work>, w: Work, max_depth: u32, min_flux: f32) {
    if w.depth <= max_depth && w.flux >= min_flux {
        stack.push(w);
    }
}

fn seg(scene: &GpuScene3, w: &Work, end_norm: Vec3, flux_end: f32, lambda: f32) -> SegOut3 {
    SegOut3 {
        a: scene.norm.world_point(w.o),
        b: scene.norm.world_point(end_norm),
        flux: flux_end,
        lambda_um: lambda,
        flags: w.flags,
    }
}

// ---------------------------------------------------------------------------
// Nearest event
// ---------------------------------------------------------------------------

enum EvKind {
    Body {
        body: usize,
        normal: Vec3,
        /// Index into the body's revolved faces, for a per-face response.
        face: usize,
    },
    Surface { surf: usize, normal: Vec3 },
    Aperture,
    Detector { det: usize, uv: Vec2 },
}

struct Event {
    t: f32,
    kind: EvKind,
}

fn nearest_event(scene: &GpuScene3, o: Vec3, d: Vec3) -> Option<Event> {
    let mut best: Option<Event> = None;
    fn consider(best: &mut Option<Event>, t: f32, kind: EvKind) {
        if best.as_ref().map(|e| t < e.t).unwrap_or(true) {
            *best = Some(Event { t, kind });
        }
    }

    for (bi, b) in scene.bodies.iter().enumerate() {
        // A body the ray does not reach before the nearest hit so far cannot
        // be the nearest: pass over it without solving its faces.
        let limit = best.as_ref().map_or(f32::INFINITY, |e| e.t);
        if !reaches(b.bounds, o, d, limit) {
            continue;
        }
        if let Some((t, nl, face)) = body_hit(b, o, d) {
            consider(
                &mut best,
                t,
                EvKind::Body {
                    body: bi,
                    normal: (b.rot * nl).normalize(),
                    face,
                },
            );
        }
    }
    for (si, s) in scene.surfaces.iter().enumerate() {
        if let Some((t, normal)) = surface_hit(s, o, d) {
            consider(&mut best, t, EvKind::Surface { surf: si, normal });
        }
    }
    for a in &scene.apertures {
        if let Some(t) = aperture_hit(a, o, d) {
            consider(&mut best, t, EvKind::Aperture);
        }
    }
    for (di, det) in scene.detectors.iter().enumerate() {
        if let Some((t, uv)) = detector_hit(det, o, d) {
            consider(&mut best, t, EvKind::Detector { det: di, uv });
        }
    }
    best
}

/// Does the ray enter the box `(lo, hi)` before `limit`? Permissive, like the
/// reference's slab test: an axis it runs along, or a `NaN` from starting on a
/// face, never rules the box out.
fn reaches((lo, hi): (Vec3, Vec3), o: Vec3, d: Vec3, limit: f32) -> bool {
    let (mut enter, mut exit) = (0.0f32, limit);
    for axis in 0..3 {
        let inv = 1.0 / d[axis];
        let mut near = (lo[axis] - o[axis]) * inv;
        let mut far = (hi[axis] - o[axis]) * inv;
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        if near > enter {
            enter = near;
        }
        if far < exit {
            exit = far;
        }
        if enter > exit {
            return false;
        }
    }
    true
}

/// A body's `(t, world normal, face index)`. Only a revolved body distinguishes
/// its faces; everything else reports face 0 and uses the body's one response.
fn body_hit(b: &GpuBody3, o: Vec3, d: Vec3) -> Option<(f32, Vec3, usize)> {
    let ol = b.rot.transpose() * (o - b.pos);
    let dl = b.rot.transpose() * d;
    let plain = |r: Option<(f32, Vec3)>| r.map(|(t, n)| (t, n, 0usize));
    match b.kind {
        OBJ_SPHERE => plain(sphere_hit(ol, dl, b.params.x, EPS)),
        OBJ_BOX => plain(box_hit(ol, dl, b.params, EPS)),
        OBJ_CYLINDER => plain(cylinder_hit(ol, dl, b.params.x, b.params.y, EPS)),
        OBJ_EXTRUDE => plain(extrude_hit(b.extrude.as_ref()?, ol, dl, EPS)),
        OBJ_REVOLVE => revolve_hit(b.revolve.as_ref()?, ol, dl, EPS),
        OBJ_TUBE => tube_hit(ol, dl, b.params, EPS),
        _ => None,
    }
}

/// A zero-thickness surface's `(t, world normal)` — its local `z = 0` plane,
/// gated by the face shape.
fn surface_hit(s: &GpuSurface3, o: Vec3, d: Vec3) -> Option<(f32, Vec3)> {
    let (t, uv) = plane_face(s.rot, s.pos, o, d)?;
    s.shape
        .contains(uv)
        .then(|| (t, (s.rot * Vec3::Z).normalize()))
}

/// A detector face hit: `(t, local uv)`.
fn detector_hit(det: &GpuDetector3, o: Vec3, d: Vec3) -> Option<(f32, Vec2)> {
    let (t, uv) = plane_face(det.rot, det.pos, o, d)?;
    det.shape.contains(uv).then_some((t, uv))
}

/// The `(t, local uv)` of a ray crossing an object's local `z = 0` plane.
fn plane_face(rot: glam::Mat3, pos: Vec3, o: Vec3, d: Vec3) -> Option<(f32, Vec2)> {
    let ol = rot.transpose() * (o - pos);
    let dl = rot.transpose() * d;
    if dl.z.abs() < 1e-12 {
        return None;
    }
    let t = -ol.z / dl.z;
    if t <= EPS {
        return None;
    }
    let p = ol + dl * t;
    Some((t, Vec2::new(p.x, p.y)))
}

/// If a ray is blocked by an aperture (outside the opening, inside the mask),
/// its `t`. A ray through the opening has no event.
fn aperture_hit(a: &GpuAperture3, o: Vec3, d: Vec3) -> Option<f32> {
    let denom = d.dot(a.normal);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = (a.origin - o).dot(a.normal) / denom;
    if t <= EPS {
        return None;
    }
    let rel = o + d * t - a.origin;
    let uv = Vec2::new(rel.dot(a.u), rel.dot(a.v));
    if a.clear.contains(uv) {
        return None; // passes through
    }
    let masked = a.outer.as_ref().map(|s| s.contains(uv)).unwrap_or(true);
    masked.then_some(t)
}

// ---------------------------------------------------------------------------
// Body geometry
// ---------------------------------------------------------------------------

/// Nearest sphere intersection at `t > tmin`, with the outward local normal.
fn sphere_hit(o: Vec3, d: Vec3, r: f32, tmin: f32) -> Option<(f32, Vec3)> {
    let b = o.dot(d);
    let c = o.dot(o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let (t0, t1) = (-b - s, -b + s);
    let t = if t0 > tmin {
        t0
    } else if t1 > tmin {
        t1
    } else {
        return None;
    };
    let p = o + d * t;
    Some((t, p / r))
}

/// Nearest box intersection at `t > tmin`, with the outward local face normal.
fn box_hit(o: Vec3, d: Vec3, half: Vec3, tmin: f32) -> Option<(f32, Vec3)> {
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    let mut enter_axis = 0usize;
    let mut exit_axis = 0usize;
    for i in 0..3 {
        let (oi, di, hi) = (o[i], d[i], half[i]);
        if di.abs() < 1e-20 {
            if oi < -hi || oi > hi {
                return None;
            }
            continue;
        }
        let inv = 1.0 / di;
        let mut t0 = (-hi - oi) * inv;
        let mut t1 = (hi - oi) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        if t0 > t_enter {
            t_enter = t0;
            enter_axis = i;
        }
        if t1 < t_exit {
            t_exit = t1;
            exit_axis = i;
        }
        if t_enter > t_exit {
            return None;
        }
    }
    let (t, ax) = if t_enter > tmin {
        (t_enter, enter_axis)
    } else if t_exit > tmin {
        (t_exit, exit_axis)
    } else {
        return None;
    };
    let p = o + d * t;
    let mut nl = Vec3::ZERO;
    nl[ax] = p[ax].signum();
    Some((t, nl))
}

/// A point's medium optics: index and attenuation of the highest-priority body
/// containing it, else the ambient (§9.4).
fn region_optics(scene: &GpuScene3, p: Vec3, lambda: f32) -> (f32, f32) {
    let mut best: Option<&GpuBody3> = None;
    for b in &scene.bodies {
        // A body lies inside its padded box, so a point outside the box is
        // outside the body: its shape need not be asked.
        let (lo, hi) = b.bounds;
        if !(p.cmpge(lo).all() && p.cmple(hi).all()) {
            continue;
        }
        if contains(b, p) {
            match best {
                Some(cur) if cur.priority >= b.priority => {}
                _ => best = Some(b),
            }
        }
    }
    match best {
        Some(b) if b.surface != SURF_ABSORBER => (b.index(lambda), b.alpha_per_mm),
        _ => (scene.ambient, 0.0),
    }
}

fn contains(b: &GpuBody3, p: Vec3) -> bool {
    let pl = b.rot.transpose() * (p - b.pos);
    match b.kind {
        OBJ_SPHERE => pl.length_squared() <= b.params.x * b.params.x,
        OBJ_BOX => pl.x.abs() <= b.params.x && pl.y.abs() <= b.params.y && pl.z.abs() <= b.params.z,
        OBJ_CYLINDER => {
            pl.x * pl.x + pl.y * pl.y <= b.params.x * b.params.x && pl.z.abs() <= b.params.y
        }
        OBJ_EXTRUDE => b
            .extrude
            .as_ref()
            .map(|e| extrude_contains(e, pl))
            .unwrap_or(false),
        OBJ_REVOLVE => b
            .revolve
            .as_ref()
            .map(|r| revolve_contains(r, pl))
            .unwrap_or(false),
        OBJ_TUBE => {
            let rho2 = pl.y * pl.y + pl.z * pl.z;
            pl.x.abs() <= b.params.z
                && rho2 >= b.params.x * b.params.x
                && rho2 <= b.params.y * b.params.y
        }
        _ => false,
    }
}

/// Is a local point inside the revolved body? Map to `(x, r)` and test the
/// sampled meridian contour.
fn revolve_contains(rev: &GpuRevolve, p: Vec3) -> bool {
    let r = (p.y * p.y + p.z * p.z).sqrt();
    let q = Vec2::new(p.x, r);
    match &rev.index {
        Some(ix) => indexed_point_in_polygon(&rev.contour, ix, q),
        None => point_in_polygon(&rev.contour, q),
    }
}

/// [`point_in_polygon`] over the one bucket holding `p.y`: the segments a
/// horizontal probe at that height can cross, which are all that change the
/// parity. The same test on each, so the same answer.
fn indexed_point_in_polygon(verts: &[Vec2], ix: &crate::scene3::ContourIndex, p: Vec2) -> bool {
    let n = verts.len();
    let (start, count) = ix.buckets[ix.bucket(p.y)];
    let mut inside = false;
    for &i in &ix.items[start as usize..(start + count) as usize] {
        let i = i as usize;
        let (a, b) = (verts[i], verts[if i == 0 { n - 1 } else { i - 1 }]);
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if p.x < a.x + t * (b.x - a.x) {
                inside = !inside;
            }
        }
    }
    inside
}

/// The radii a ray passes through inside an indexed body's `x` slab at
/// `t > tmin`, or `None` if it never enters the slab. Mirrors `ray_radii` in
/// the shader.
fn ray_radii(ix: &crate::scene3::ContourIndex, o: Vec3, d: Vec3, tmin: f32) -> Option<(f32, f32)> {
    const PAD: f32 = 1e-3;
    let (mut ta, mut tb) = (tmin, 1e30_f32);
    if d.x.abs() > 1e-12 {
        let u = (ix.x_lo - PAD - o.x) / d.x;
        let v = (ix.x_hi + PAD - o.x) / d.x;
        ta = ta.max(u.min(v));
        tb = tb.min(u.max(v));
    } else if o.x < ix.x_lo - PAD || o.x > ix.x_hi + PAD {
        return None;
    }
    if ta > tb {
        return None;
    }
    let oo = o.y * o.y + o.z * o.z;
    let od = o.y * d.y + o.z * d.z;
    let dd = d.y * d.y + d.z * d.z;
    let r2 = |t: f32| (oo + t * (2.0 * od + t * dd)).max(0.0);
    let ts = if dd > 0.0 {
        (-od / dd).clamp(ta, tb)
    } else {
        ta
    };
    let lo = r2(ts).min(r2(ta)).min(r2(tb)).sqrt();
    let hi = r2(ta).max(r2(tb)).sqrt();
    Some((lo, hi))
}

/// The radii a revolved face spans, where that is read straight off it.
fn face_radii(face: &RevFace) -> Option<(f32, f32)> {
    match face {
        RevFace::Sag {
            inner_r, outer_r, ..
        } => Some((*inner_r, *outer_r)),
        RevFace::Frustum { y0, y1, .. } => Some((y0.min(*y1), y0.max(*y1))),
        RevFace::Arc { .. } => None,
    }
}

/// Nearest revolved-profile intersection at `t > tmin`, with the outward local
/// normal. Each face is exact; the normal is oriented by a containment probe,
/// because a single surface cannot say which side is glass.
fn revolve_hit(rev: &GpuRevolve, o: Vec3, d: Vec3, tmin: f32) -> Option<(f32, Vec3, usize)> {
    let mut best: Option<(f32, Vec3, usize)> = None;
    let keep = |best: &mut Option<(f32, Vec3, usize)>, t: f32, n: Vec3, i: usize| {
        if t > tmin && best.map(|(bt, _, _)| t < bt).unwrap_or(true) {
            *best = Some((t, orient_rev(rev, n, o + d * t), i));
        }
    };
    // An indexed (many-faced) body passes over the faces whose radii the ray
    // never reaches inside its slab.
    let reach = match &rev.index {
        Some(ix) => Some(ray_radii(ix, o, d, tmin)?),
        None => None,
    };
    for (fi, face) in rev.faces.iter().enumerate() {
        if let (Some((lo, hi)), Some((f_lo, f_hi))) = (reach, face_radii(face)) {
            let pad = 1e-3 * (1.0 + hi.min(1e3));
            if f_hi < lo - pad || f_lo > hi + pad {
                continue;
            }
        }
        match face {
            RevFace::Sag {
                vertex_x,
                axis_sign,
                c,
                k,
                coeffs,
                inner_r,
                outer_r,
                is_conic,
            } => sag_hit(
                *vertex_x,
                *axis_sign,
                *c,
                *k,
                coeffs,
                (*inner_r, *outer_r),
                *is_conic,
                o,
                d,
                tmin,
                &mut |t, n| keep(&mut best, t, n, fi),
            ),
            RevFace::Frustum { x0, y0, x1, y1 } => {
                frustum_hit(*x0, *y0, *x1, *y1, o, d, tmin, &mut |t, n| {
                    keep(&mut best, t, n, fi)
                })
            }
            RevFace::Arc {
                center_x,
                r,
                a0,
                a1,
            } => arc_hit(*center_x, *r, *a0, *a1, o, d, tmin, &mut |t, n| {
                keep(&mut best, t, n, fi)
            }),
        }
    }
    best
}

fn orient_rev(rev: &GpuRevolve, n: Vec3, p: Vec3) -> Vec3 {
    if revolve_contains(rev, p + n * EPS) {
        -n
    } else {
        n
    }
}

/// The outward normal of a conic/asphere of revolution at `p` (analytic, §7.4).
/// `vertex` is on the axis; radial is the component perpendicular to `+X`.
fn sag_normal(vertex_x: f32, axis_sign: f32, c: f32, k: f32, coeffs: &[f32; 8], p: Vec3) -> Vec3 {
    let rad = Vec3::new(0.0, p.y, p.z);
    let r = rad.length();
    let axis = Vec3::new(axis_sign, 0.0, 0.0);
    let n = if r > 0.0 {
        axis - rad / r * even_asphere_dsag(c, k, coeffs, r)
    } else {
        axis
    };
    let _ = vertex_x;
    n.normalize()
}

/// Intersect a conic (exact quadratic) or even-asphere (safeguarded Newton)
/// surface of revolution about `+X`, reporting each aperture-clipped hit.
#[allow(clippy::too_many_arguments)]
fn sag_hit(
    vertex_x: f32,
    axis_sign: f32,
    c: f32,
    k: f32,
    coeffs: &[f32; 8],
    (inner_r, outer_r): (f32, f32),
    is_conic: bool,
    o: Vec3,
    d: Vec3,
    tmin: f32,
    out: &mut dyn FnMut(f32, Vec3),
) {
    let ov = o - Vec3::new(vertex_x, 0.0, 0.0);
    let z0 = ov.x * axis_sign;
    let dz = d.x * axis_sign;
    let o_rad = Vec3::new(0.0, ov.y, ov.z);
    let d_rad = Vec3::new(0.0, d.y, d.z);

    // The sag envelope: keep the near nappe out to the aperture; reject the far
    // side of the whole revolution.
    let sag_edge = even_asphere_sag(c, k, coeffs, outer_r);
    let z_lo = sag_edge.min(0.0);
    let z_hi = sag_edge.max(0.0);
    let z_tol = 1e-4 * (1.0 + outer_r.abs());
    let r_tol = 1e-5 * (1.0 + outer_r);

    let accept = |t: f32, out: &mut dyn FnMut(f32, Vec3)| {
        if t <= tmin {
            return;
        }
        let p = o + d * t;
        let r = (p.y * p.y + p.z * p.z).sqrt();
        let z = (p - Vec3::new(vertex_x, 0.0, 0.0)).x * axis_sign;
        if r >= inner_r - r_tol && r <= outer_r + r_tol && z >= z_lo - z_tol && z <= z_hi + z_tol {
            out(t, sag_normal(vertex_x, axis_sign, c, k, coeffs, p));
        }
    };

    if is_conic {
        let k1 = 1.0 + k;
        let aa = c * (d_rad.length_squared() + k1 * dz * dz);
        let bb = c * (2.0 * o_rad.dot(d_rad) + 2.0 * k1 * z0 * dz) - 2.0 * dz;
        let cc = c * (o_rad.length_squared() + k1 * z0 * z0) - 2.0 * z0;
        for t in quadratic_roots(aa, bb, cc) {
            accept(t, out);
        }
        return;
    }

    // Even asphere: seed from the base conic, then Newton-refine the residual
    // axial(t) − sag(r(t)) with the analytic derivative (§7.4).
    let r_of = |t: f32| (o_rad + d_rad * t).length();
    let residual = |t: f32| (z0 + t * dz) - even_asphere_sag(c, k, coeffs, r_of(t));
    let dresidual = |t: f32| {
        let rad = o_rad + d_rad * t;
        let r = rad.length();
        let dr = if r > 0.0 { rad.dot(d_rad) / r } else { 0.0 };
        dz - even_asphere_dsag(c, k, coeffs, r) * dr
    };

    let k1 = 1.0 + k;
    let aa = c * (d_rad.length_squared() + k1 * dz * dz);
    let bb = c * (2.0 * o_rad.dot(d_rad) + 2.0 * k1 * z0 * dz) - 2.0 * dz;
    let cc = c * (o_rad.length_squared() + k1 * z0 * z0) - 2.0 * z0;
    let mut seeds: Vec<f32> = quadratic_roots(aa, bb, cc).into_iter().collect();
    if seeds.is_empty() {
        // A coarse march over the aperture span, so a strongly aspheric surface
        // with no nearby conic root is still bracketed.
        let start = tmin.max(0.0);
        let step = outer_r.max(0.05) * 0.1;
        let mut pt = start;
        let mut pv = residual(pt);
        for i in 1..=200 {
            let t = start + step * i as f32;
            let v = residual(t);
            if pv * v < 0.0 {
                seeds.push(0.5 * (pt + t));
            }
            pt = t;
            pv = v;
        }
    }
    for seed in seeds {
        if let Some(t) = refine_root(seed, tmin, &residual, &dresidual) {
            accept(t, out);
        }
    }
}

/// Real roots of `aa·t² + bb·t + cc`, numerically stable. Handles the degenerate
/// linear case (a plane, or a ray along the axis).
fn quadratic_roots(aa: f32, bb: f32, cc: f32) -> Vec<f32> {
    if aa.abs() < 1e-20 {
        return if bb.abs() > 1e-20 {
            vec![-cc / bb]
        } else {
            Vec::new()
        };
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return Vec::new();
    }
    let sq = disc.sqrt();
    let q = -0.5 * (bb + bb.signum() * sq);
    let mut roots = vec![q / aa];
    if q.abs() > 1e-20 {
        roots.push(cc / q);
    }
    roots
}

/// Safeguarded Newton for `f(t) = 0` near `seed`, bisecting when a step leaves a
/// sign-change bracket (§7.4). f32 tolerances.
fn refine_root(
    seed: f32,
    tmin: f32,
    f: &dyn Fn(f32) -> f32,
    df: &dyn Fn(f32) -> f32,
) -> Option<f32> {
    let mut t = seed;
    let span = 1e-2_f32;
    let (mut lo, mut hi) = (seed - span, seed + span);
    let (flo, fhi) = (f(lo), f(hi));
    let bracketed = flo * fhi <= 0.0;
    let mut flo = flo;

    for _ in 0..48 {
        let ft = f(t);
        if ft.abs() < 1e-7 {
            break;
        }
        let dft = df(t);
        let mut nt = if dft.abs() > 1e-30 {
            t - ft / dft
        } else {
            f32::INFINITY
        };
        if !nt.is_finite() || (bracketed && (nt < lo || nt > hi)) {
            nt = 0.5 * (lo + hi);
        }
        if bracketed {
            let fnt = f(nt);
            if flo * fnt <= 0.0 {
                hi = nt;
            } else {
                lo = nt;
                flo = fnt;
            }
        }
        let step = (nt - t).abs();
        t = nt;
        if step < 1e-8 * (1.0 + t.abs()) {
            break;
        }
    }
    let _ = fhi;
    (f(t).abs() < 1e-5 && t > tmin).then_some(t)
}

/// A line segment `(x0,y0)–(x1,y1)` revolved about `+X`: a cone, cylinder, or flat
/// annular face. Ported from `oc_core::geom3::primitives::frustum_intersect`.
#[allow(clippy::too_many_arguments)]
fn frustum_hit(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    o: Vec3,
    d: Vec3,
    tmin: f32,
    out: &mut dyn FnMut(f32, Vec3),
) {
    let (x_lo, x_hi) = (x0.min(x1), x0.max(x1));
    let flat = (x1 - x0).abs() <= 1e-9 * (1.0 + x0.abs() + x1.abs());

    if flat {
        if d.x.abs() < 1e-12 {
            return;
        }
        let t = (x0 - o.x) / d.x;
        if t <= tmin {
            return;
        }
        let p = o + d * t;
        let r = (p.y * p.y + p.z * p.z).sqrt();
        let (r_lo, r_hi) = (y0.min(y1), y0.max(y1));
        let pad = 1e-9 * (1.0 + r_hi);
        if r >= r_lo - pad && r <= r_hi + pad {
            out(t, Vec3::X);
        }
        return;
    }

    let m = (y1 - y0) / (x1 - x0);
    let b = y0 - m * x0;
    let g0 = m * o.x + b;
    let aa = d.y * d.y + d.z * d.z - m * m * d.x * d.x;
    let bb = 2.0 * (o.y * d.y + o.z * d.z) - 2.0 * m * d.x * g0;
    let cc = o.y * o.y + o.z * o.z - g0 * g0;
    let pad = 1e-6 * (1.0 + x_hi.abs() + x_lo.abs());
    for t in quadratic_roots(aa, bb, cc) {
        if t <= tmin {
            continue;
        }
        let p = o + d * t;
        let r_line = m * p.x + b;
        if r_line >= -pad && p.x >= x_lo - pad && p.x <= x_hi + pad {
            out(t, Vec3::new(-m * r_line, p.y, p.z).normalize());
        }
    }
}

/// An axis-centred arc revolved about `+X`: a spherical zone.
#[allow(clippy::too_many_arguments)]
fn arc_hit(
    center_x: f32,
    r: f32,
    a0: f32,
    a1: f32,
    o: Vec3,
    d: Vec3,
    tmin: f32,
    out: &mut dyn FnMut(f32, Vec3),
) {
    let c3 = Vec3::new(center_x, 0.0, 0.0);
    let oc = o - c3;
    let b = oc.dot(d);
    let disc = b * b - (oc.length_squared() - r * r);
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    for t in [-b - sq, -b + sq] {
        if t <= tmin {
            continue;
        }
        let p = o + d * t;
        let rad = (p.y * p.y + p.z * p.z).sqrt();
        let ang = rad.atan2(p.x - center_x);
        if angle_in_sweep(ang, a0, a1) {
            out(t, (p - c3) / r);
        }
    }
}

/// Is `ang` within the CCW sweep from `a0` to `a1`? Mirrors the profile arc's
/// unwrapped-difference convention.
fn angle_in_sweep(ang: f32, a0: f32, a1: f32) -> bool {
    use std::f32::consts::TAU;
    let sweep = (a1 - a0).rem_euclid(TAU);
    if sweep >= TAU - 1e-6 {
        return true;
    }
    let rel = (ang - a0).rem_euclid(TAU);
    rel <= sweep + 1e-6
}

/// Is a local point inside the extruded body?
fn extrude_contains(e: &GpuExtrude, p: Vec3) -> bool {
    p.z >= e.z_min && p.z <= e.z_max && point_in_polygon(&e.verts, Vec2::new(p.x, p.y))
}

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

/// Nearest finite-cylinder intersection at `t > tmin` (canonical `+Z` axis), with
/// the outward local normal. Side wall plus the two end caps.
fn cylinder_hit(o: Vec3, d: Vec3, r: f32, hl: f32, tmin: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    let keep = |best: &mut Option<(f32, Vec3)>, t: f32, n: Vec3| {
        if t > tmin && best.map(|(bt, _)| t < bt).unwrap_or(true) {
            *best = Some((t, n));
        }
    };

    let a = d.x * d.x + d.y * d.y;
    if a > 1e-20 {
        let b = 2.0 * (o.x * d.x + o.y * d.y);
        let c = o.x * o.x + o.y * o.y - r * r;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            for t in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                let z = o.z + t * d.z;
                if z.abs() <= hl {
                    let p = o + d * t;
                    keep(&mut best, t, Vec3::new(p.x, p.y, 0.0) / r);
                }
            }
        }
    }
    if d.z.abs() > 1e-20 {
        for (zc, nz) in [(hl, 1.0_f32), (-hl, -1.0_f32)] {
            let t = (zc - o.z) / d.z;
            let p = o + d * t;
            if p.x * p.x + p.y * p.y <= r * r {
                keep(&mut best, t, Vec3::new(0.0, 0.0, nz));
            }
        }
    }
    best
}

/// Nearest tube intersection at `t > tmin`: `(t, outward local normal, face)`,
/// faces in the kernel's `tube_face` order. `p = (inner, outer, half_length)`,
/// axis local `+X`; the bore's outward normal faces the axis.
fn tube_hit(o: Vec3, d: Vec3, p: Vec3, tmin: f32) -> Option<(f32, Vec3, usize)> {
    let (ri, ro, hl) = (p.x, p.y, p.z);
    let mut best: Option<(f32, Vec3, usize)> = None;
    let mut keep = |t: f32, n: Vec3, face: usize| {
        if t > tmin && best.map(|(bt, _, _)| t < bt).unwrap_or(true) {
            best = Some((t, n, face));
        }
    };
    let a = d.y * d.y + d.z * d.z;
    if a > 1e-20 {
        for (r, face, sign) in [(ro, 0usize, 1.0f32), (ri, 3usize, -1.0f32)] {
            let b = 2.0 * (o.y * d.y + o.z * d.z);
            let c = o.y * o.y + o.z * o.z - r * r;
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                for t in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                    let q = o + d * t;
                    if q.x.abs() <= hl {
                        keep(t, Vec3::new(0.0, q.y, q.z) * (sign / r), face);
                    }
                }
            }
        }
    }
    if d.x.abs() > 1e-20 {
        for (xc, nx, face) in [(hl, 1.0f32, 1usize), (-hl, -1.0f32, 2usize)] {
            let t = (xc - o.x) / d.x;
            let q = o + d * t;
            let rho2 = q.y * q.y + q.z * q.z;
            if rho2 >= ri * ri && rho2 <= ro * ro {
                keep(t, Vec3::new(nx, 0.0, 0.0), face);
            }
        }
    }
    best
}

/// Nearest extruded-profile intersection at `t > tmin`, with the outward local
/// normal: ruled side faces plus the two end caps.
fn extrude_hit(e: &GpuExtrude, o: Vec3, d: Vec3, tmin: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    let keep = |best: &mut Option<(f32, Vec3)>, t: f32, n: Vec3| {
        if t > tmin && best.map(|(bt, _)| t < bt).unwrap_or(true) {
            *best = Some((t, n));
        }
    };

    let n = e.verts.len();
    let o2 = Vec2::new(o.x, o.y);
    let d2 = Vec2::new(d.x, d.y);
    if d2.length() > 1e-12 {
        for i in 0..n {
            let (a, b) = (e.verts[i], e.verts[(i + 1) % n]);
            if let Some((t, n2)) = segment_hit(a, b, o2, d2, tmin) {
                let z = o.z + t * d.z;
                if z >= e.z_min && z <= e.z_max {
                    let p = o + d * t;
                    let nrm = orient_out(e, Vec3::new(n2.x, n2.y, 0.0), p);
                    keep(&mut best, t, nrm);
                }
            }
        }
    }
    if d.z.abs() > 1e-12 {
        for (z, nz) in [(e.z_min, -1.0_f32), (e.z_max, 1.0_f32)] {
            let t = (z - o.z) / d.z;
            if t > tmin {
                let p = o + d * t;
                if point_in_polygon(&e.verts, Vec2::new(p.x, p.y)) {
                    keep(&mut best, t, Vec3::new(0.0, 0.0, nz));
                }
            }
        }
    }
    best
}

/// Orient a side-face normal outward: away from the body's interior.
fn orient_out(e: &GpuExtrude, n: Vec3, p: Vec3) -> Vec3 {
    if extrude_contains(e, p + n * EPS) {
        -n
    } else {
        n
    }
}

/// Intersect the 2D ray `o + t·d` (`d` need not be unit) with segment `a–b`.
/// Returns `(t, unit edge normal)` when `t > tmin` and the hit is on the segment.
fn segment_hit(a: Vec2, b: Vec2, o: Vec2, d: Vec2, tmin: f32) -> Option<(f32, Vec2)> {
    let edge = b - a;
    let denom = cross2(d, edge);
    if denom.abs() < 1e-20 {
        return None;
    }
    let ao = a - o;
    let t = cross2(ao, edge) / denom;
    let u = cross2(ao, d) / denom;
    if t <= tmin || !(0.0..=1.0).contains(&u) {
        return None;
    }
    Some((t, Vec2::new(edge.y, -edge.x).normalize()))
}

#[inline]
fn cross2(p: Vec2, q: Vec2) -> f32 {
    p.x * q.y - p.y * q.x
}

// ---------------------------------------------------------------------------
// Optics (f32, unpolarized)
// ---------------------------------------------------------------------------

fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    (d - 2.0 * d.dot(n) * n).normalize()
}

/// Refracted direction, or `None` at total internal reflection. `n` opposes `d`.
fn refract(d: Vec3, n: Vec3, n1: f32, n2: f32) -> Option<Vec3> {
    let eta = n1 / n2;
    let cos_i = -n.dot(d);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        None
    } else {
        Some((eta * d + (eta * cos_i - k.sqrt()) * n).normalize())
    }
}

fn reflectance(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let ci = cos_i.abs().clamp(0.0, 1.0);
    let s2 = 1.0 - ci * ci;
    let st2 = (n1 / n2) * (n1 / n2) * s2;
    if st2 >= 1.0 {
        return 1.0;
    }
    let ct = (1.0 - st2).sqrt();
    let rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    let rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    0.5 * (rs * rs + rp * rp)
}

fn beer_lambert(alpha_per_mm: f32, len_mm: f64) -> f32 {
    if alpha_per_mm <= 0.0 {
        1.0
    } else {
        (-alpha_per_mm as f64 * len_mm).exp() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::scene3::OutOfRangePolicy;

    #[test]
    fn a_missing_ar_residual_leaves_fresnel_unscaled() {
        // tracer3: `sample(ar_residual).unwrap_or(1.0).clamp(0.0, 1.0)`.
        let table = |policy| Spectral32::Table {
            lambda: vec![0.5, 0.6],
            values: vec![0.2, 0.4],
            policy,
        };
        assert_eq!(ar_at(&table(OutOfRangePolicy::Error), 0.7), 1.0);
        assert_eq!(ar_at(&table(OutOfRangePolicy::Zero), 0.7), 0.0);
        assert_eq!(ar_at(&table(OutOfRangePolicy::Clamp), 0.7), 0.4);
        assert!((ar_at(&table(OutOfRangePolicy::Error), 0.55) - 0.3).abs() < 1e-6);
        assert_eq!(ar_at(&Spectral32::Const(1.5), 0.55), 1.0);
    }
}
