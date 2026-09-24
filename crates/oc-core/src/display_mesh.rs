//! Display tessellation of the canonical 3D scene.
//!
//! Triangle meshes for a renderer to **look at**. Nothing here is traced, and no
//! mesh produced here is ever fed back into the scene or written to a file: the
//! tracer keeps intersecting the analytic geometry in [`crate::geom3`]. A mesh is
//! a picture of that geometry, sampled from the same definitions (the same sag
//! polynomial, the same profile curves), so it cannot describe a different lens
//! than the one the tracer sees.
//!
//! # Coordinates
//!
//! Every mesh is in its object's **local** frame, in millimetres. The renderer
//! applies the object's transform, so moving or tilting an object never needs a
//! new mesh.
//!
//! # Accuracy
//!
//! Vertices lie on the analytic surface (to `f32` rounding). Between vertices a
//! triangle departs from the surface by at most [`Tessellation::chord_tol_mm`]
//! along a meridian, and neighbouring vertex normals differ by at most
//! [`Tessellation::max_normal_step_rad`]. Normals are analytic wherever the
//! kernel has a closed form (conics and even aspheres use the kernel's own
//! `dsag/dr`), and flat for planes.
//!
//! # Display-only thickness
//!
//! Zero-thickness objects (mirrors and plates, apertures, detectors) are drawn as
//! thin slabs centred on (or, for a detector, behind) the physical plane so they
//! have a visible edge. That thickness exists only in the picture. The tracer
//! still treats each of these as the zero-thickness surface it is, and the
//! thickness is never exported.
//!
//! # Feature edges
//!
//! Each mesh also lists its creases as line pairs ([`DisplayMesh::edges`]): a
//! lens's rim rings, a prism's corners and cap outlines, a slab's outlines. They
//! come from how the mesh was built, not from comparing face normals, so a
//! renderer can outline a body without analysing its triangles.
//!
//! # Light sources
//!
//! A source is not an object the tracer can hit, but it has a place, an
//! emitting aperture, and a direction. Its mesh is a glyph: the aperture as a
//! thin plate whose front face is the emission plane, and a housing behind it.
//! The housing is a display choice with no size in the scene.
//!
//! # What cannot be drawn
//!
//! A 2D-only body (a v1 migration whose third dimension was never authored) has
//! no solid to draw. Its mesh is the profile outline as line segments, with a
//! [`DisplayMesh::note`] that says why, rather than an invented depth (§3.2).

use std::collections::HashSet;
use std::f64::consts::TAU;

use crate::geom3::primitives::{even_asphere_dsag, even_asphere_sag};
use crate::geom3::profile::{profile_contains_2d, validate_extrudable, validate_revolvable};
use crate::profile2::{ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3};
use crate::scene3::{
    AngularDistribution3, Aperture3, Body3, Detector3, DetectorGeometry3, PlanarShape2,
    RadialAperture, SagModel, Scene3, SolidGeometry3, Source3, SpatialDistribution3,
    SurfaceGeometry3, SurfaceObject3,
};
use crate::vec::{perp, Vec2};
use crate::vec3::{Plane3, Vec3};

/// How finely curved geometry is sampled for display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tessellation {
    /// Largest distance between a curve and the chord that stands in for it, mm.
    pub chord_tol_mm: f64,
    /// Largest angle between the normals at the two ends of one step, radians.
    pub max_normal_step_rad: f64,
    /// Fewest segments around a circle.
    pub min_segments: usize,
    /// Most segments around a circle.
    pub max_segments: usize,
    /// Most samples along one profile curve.
    pub max_curve_samples: usize,
}

impl Default for Tessellation {
    /// 0.02 mm chord error and about 6 degrees between neighbouring normals. For
    /// a 25 mm lens this is 56 segments around and 10 to 30 samples across a
    /// face, a few thousand triangles. Smooth normals carry the look; the chord
    /// tolerance keeps the silhouette honest at the rim.
    fn default() -> Self {
        Self {
            chord_tol_mm: 0.02,
            max_normal_step_rad: 0.105,
            min_segments: 24,
            max_segments: 128,
            max_curve_samples: 257,
        }
    }
}

/// What a run of triangles is, so a renderer can shade it differently. The
/// numbers are part of the browser contract (`web/src/render3d/types.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FaceGroup {
    /// A body's optical surfaces: lens faces, prism walls and caps, solids.
    Optical = 0,
    /// A lens's ground rim (a line in the profile).
    Rim = 1,
    /// The front (local `+Z`) face of a zero-thickness object.
    Front = 2,
    /// The back (local `-Z`) face.
    Back = 3,
    /// The edge band of a display-only slab.
    Edge = 4,
    /// A detector's sensitive faces.
    DetectorActive = 5,
    /// A detector's edge band.
    DetectorHousing = 6,
    /// A light source's emitting aperture.
    SourceEmitter = 7,
    /// A light source's display housing.
    SourceHousing = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Triangles,
    Lines,
}

/// A contiguous run of indices that shares one [`FaceGroup`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshGroup {
    pub start: u32,
    pub count: u32,
    pub group: FaceGroup,
}

/// A mesh in its object's local frame.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayMesh {
    pub topology: Topology,
    /// `x y z` per vertex, mm.
    pub positions: Vec<f32>,
    /// Unit `x y z` per vertex. Zero for line meshes.
    pub normals: Vec<f32>,
    /// Triangles (counter-clockwise seen from outside) or line pairs.
    pub indices: Vec<u32>,
    pub groups: Vec<MeshGroup>,
    /// Crease lines as vertex-index pairs, each drawn once.
    pub edges: Vec<u32>,
    /// Local bounds of the vertices; zero for an empty mesh.
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    /// Why this is not a solid, when it is not.
    pub note: Option<String>,
}

impl DisplayMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    pub fn triangle_count(&self) -> usize {
        match self.topology {
            Topology::Triangles => self.indices.len() / 3,
            Topology::Lines => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene entry points
// ---------------------------------------------------------------------------

/// The display mesh of the object with `id`, whatever kind it is, or `None` if
/// no body, surface, aperture, detector, or source has that id.
pub fn object_mesh(scene: &Scene3, id: u64, tess: &Tessellation) -> Option<DisplayMesh> {
    if let Some(b) = scene.bodies.iter().find(|b| b.common.id.0 == id) {
        return Some(body_mesh(b, tess));
    }
    if let Some(s) = scene.surfaces.iter().find(|s| s.common.id.0 == id) {
        return Some(surface_mesh(s, tess));
    }
    if let Some(a) = scene.apertures.iter().find(|a| a.common.id.0 == id) {
        return Some(aperture_mesh(a, tess));
    }
    if let Some(d) = scene.detectors.iter().find(|d| d.common.id.0 == id) {
        return Some(detector_mesh(d, tess));
    }
    if let Some(s) = scene.sources.iter().find(|s| s.id.0 == id) {
        return Some(source_mesh(s, tess));
    }
    None
}

/// A body's display mesh. A body with no drawable solid (2D-only, or a profile
/// the kernel refuses) comes back as its outline with a note.
pub fn body_mesh(b: &Body3, tess: &Tessellation) -> DisplayMesh {
    match &b.geometry {
        SolidGeometry3::Profile(p) => match p.construction {
            ProfileConstruction3::RevolveLocalX => match validate_revolvable(&p.profile) {
                Ok(()) => revolve(&p.profile, tess),
                Err(why) => outline(&p.profile, 0.0, tess, format!("cannot revolve: {why}")),
            },
            ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } => {
                match validate_extrudable(&p.profile) {
                    Ok(()) if z_max_mm > z_min_mm => extrude(&p.profile, z_min_mm, z_max_mm, tess),
                    Ok(()) => outline(
                        &p.profile,
                        0.5 * (z_min_mm + z_max_mm),
                        tess,
                        "cannot extrude: empty Z range".to_string(),
                    ),
                    Err(why) => {
                        outline(&p.profile, z_min_mm, tess, format!("cannot extrude: {why}"))
                    }
                }
            }
            ProfileConstruction3::PlanarOnly { plane_z_mm } => outline(
                &p.profile,
                plane_z_mm,
                tess,
                "2D-only body: its third dimension was never authored".to_string(),
            ),
        },
        SolidGeometry3::Sphere { radius_mm } => sphere(*radius_mm, tess),
        SolidGeometry3::Box { half_extents_mm } => cuboid(*half_extents_mm),
        SolidGeometry3::Cylinder {
            radius_mm,
            half_length_mm,
            axis_local,
        } => cylinder(*radius_mm, *half_length_mm, *axis_local, tess),
        SolidGeometry3::Tube {
            inner_radius_mm,
            outer_radius_mm,
            half_length_mm,
        } => tube(*inner_radius_mm, *outer_radius_mm, *half_length_mm, tess),
    }
}

/// A zero-thickness surface, drawn as a display-only slab (see the module doc).
pub fn surface_mesh(s: &SurfaceObject3, tess: &Tessellation) -> DisplayMesh {
    let mut m = Builder::new(Topology::Triangles);
    match &s.geometry {
        SurfaceGeometry3::Plane { shape } => {
            let half = slab_half_thickness(shape_radius(shape));
            let frame = PlaneFrame::local_xy();
            planar_slab(&mut m, shape, half, &frame, tess, SlabGroups::surface());
        }
        SurfaceGeometry3::SurfaceOfRevolution {
            sag,
            radial_aperture,
        } => {
            let half = slab_half_thickness(radial_aperture.outer_radius_mm);
            revolved_sheet(&mut m, sag, radial_aperture, half, tess);
        }
    }
    m.finish(None)
}

/// An aperture's opaque mask (outside the clear opening), as a display-only
/// slab in its plane. A mask with no outer extent is unbounded in the model; it
/// is drawn out to a finite radius, which is a display choice.
pub fn aperture_mesh(a: &Aperture3, tess: &Tessellation) -> DisplayMesh {
    let mut m = Builder::new(Topology::Triangles);
    let frame = PlaneFrame::from_plane(&a.shape.plane_local);
    let clear = &a.shape.clear_opening;
    let clear_r = shape_radius(clear);
    let (mask_outer, mask_inner): (Option<&PlanarShape2>, f64) = match &a.shape.outer_extent {
        Some(PlanarShape2::Annulus { inner_mm, .. }) => (a.shape.outer_extent.as_ref(), *inner_mm),
        Some(s) => (Some(s), 0.0),
        None => (None, 0.0),
    };
    let unbounded_r = (clear_r * 2.5).max(clear_r + 6.0);
    let outer_r = mask_outer.map_or(unbounded_r, shape_radius);
    let half = slab_half_thickness(outer_r);
    let groups = SlabGroups::surface();

    // The opaque ring: from the opening's outer boundary to the mask's.
    let inner = RadialShape::of(clear, true);
    let outer = match mask_outer {
        Some(s) => RadialShape::of(s, true),
        None => RadialShape::Circle(unbounded_r),
    };
    let inner = if mask_inner > 0.0 {
        inner.at_least(mask_inner)
    } else {
        inner
    };
    ring_slab(&mut m, &inner, &outer, half, &frame, tess, groups);

    // A ring-shaped opening leaves an opaque centre: the central obstruction.
    if let PlanarShape2::Annulus { inner_mm, .. } = clear {
        if *inner_mm > 0.0 {
            planar_slab(
                &mut m,
                &PlanarShape2::Disc {
                    radius_mm: *inner_mm,
                },
                half,
                &frame,
                tess,
                groups,
            );
        }
    }
    m.finish(None)
}

/// A detector: both faces are sensitive (the tracer records a crossing from
/// either side), with a dark edge band. The slab is centred on the detector
/// plane and is display-only.
pub fn detector_mesh(d: &Detector3, tess: &Tessellation) -> DisplayMesh {
    let shape = match &d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => PlanarShape2::Rectangle {
            half_width_mm: *half_width_mm,
            half_height_mm: *half_height_mm,
        },
        DetectorGeometry3::Disc { radius_mm } => PlanarShape2::Disc {
            radius_mm: *radius_mm,
        },
        DetectorGeometry3::Annulus { inner_mm, outer_mm } => PlanarShape2::Annulus {
            inner_mm: *inner_mm,
            outer_mm: *outer_mm,
        },
        DetectorGeometry3::LineProbeLegacy {
            a_local, b_local, ..
        } => {
            let mut m = Builder::new(Topology::Lines);
            let a = m.vertex(*a_local, Vec3::ZERO);
            let b = m.vertex(*b_local, Vec3::ZERO);
            m.group(FaceGroup::DetectorActive);
            m.line(a, b);
            return m.finish(Some(
                "legacy line probe: it has no area, so it is drawn as a line".to_string(),
            ));
        }
    };
    let mut m = Builder::new(Topology::Triangles);
    let half = slab_half_thickness(shape_radius(&shape)).max(0.25);
    planar_slab(
        &mut m,
        &shape,
        half,
        &PlaneFrame::local_xy(),
        tess,
        SlabGroups {
            front: FaceGroup::DetectorActive,
            back: FaceGroup::DetectorActive,
            edge: FaceGroup::DetectorHousing,
        },
    );
    m.finish(None)
}

/// A light source's glyph (see the module doc). The emitting aperture lies in
/// the local YZ plane and light leaves along local `+X`, or `-X` for a
/// collimated source aimed backwards; the plate's front face is the emission
/// plane and the housing sits behind it. A point source is a small ball on a
/// short stem. A legacy line source has no area and is drawn as its line.
pub fn source_mesh(s: &Source3, tess: &Tessellation) -> DisplayMesh {
    let forward = match s.angular {
        AngularDistribution3::Direction { local_dir } if local_dir.x < 0.0 => -1.0,
        _ => 1.0,
    };
    let axis = Vec3::X * forward;
    let aperture = match s.spatial {
        SpatialDistribution3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => Some(PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        }),
        SpatialDistribution3::Disc { radius_mm } => Some(PlanarShape2::Disc { radius_mm }),
        SpatialDistribution3::Point => None,
        SpatialDistribution3::LineLegacy { half_width_mm } => {
            let mut m = Builder::new(Topology::Lines);
            let a = m.vertex(Vec3::new(0.0, -half_width_mm, 0.0), Vec3::ZERO);
            let b = m.vertex(Vec3::new(0.0, half_width_mm, 0.0), Vec3::ZERO);
            m.group(FaceGroup::SourceEmitter);
            m.line(a, b);
            return m.finish(Some(
                "legacy line source: it has no area, so it is drawn as a line".to_string(),
            ));
        }
    };
    let mut m = Builder::new(Topology::Triangles);
    let Some(shape) = aperture else {
        // A point: the ball is the emitter, the stem behind it the housing.
        let r = 1.0;
        m.group(FaceGroup::SourceEmitter);
        ball_into(&mut m, Vec3::ZERO, r, tess);
        m.group(FaceGroup::SourceHousing);
        let stem = 3.0;
        cylinder_into(&mut m, -axis * (0.5 * r + stem), 0.6 * r, stem, axis, tess);
        return m.finish(None);
    };
    let r = shape_radius(&shape);
    let half = slab_half_thickness(r);
    // The plate spans `-2·half ..= 0` along the axis, so its front face is the
    // emission plane and no drawn ray starts inside it.
    let frame = PlaneFrame {
        origin: -axis * half,
        u: Vec3::Y,
        v: Vec3::Z,
        n: axis,
    };
    let groups = SlabGroups {
        front: FaceGroup::SourceEmitter,
        back: FaceGroup::SourceHousing,
        edge: FaceGroup::SourceHousing,
    };
    planar_slab(&mut m, &shape, half, &frame, tess, groups);
    // The housing: a bezel a little wider than the aperture, reaching back.
    let margin = (0.12 * r).clamp(0.5, 2.0);
    let depth = (0.6 * r).clamp(3.0, 12.0);
    m.group(FaceGroup::SourceHousing);
    match shape {
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => {
            let (x0, x1) = (-2.0 * half - depth, -2.0 * half);
            let (x0, x1) = if forward > 0.0 { (x0, x1) } else { (-x1, -x0) };
            let hy = half_width_mm + margin;
            let hz = half_height_mm + margin;
            box_into(&mut m, Vec3::new(x0, -hy, -hz), Vec3::new(x1, hy, hz));
        }
        _ => {
            let centre = -axis * (2.0 * half + 0.5 * depth);
            cylinder_into(&mut m, centre, r + margin, 0.5 * depth, axis, tess);
        }
    }
    m.finish(None)
}

/// Half of the display thickness given to a zero-thickness object of radius
/// `r`: 3% of the radius, between 0.15 and 1 mm.
fn slab_half_thickness(r: f64) -> f64 {
    (0.03 * r).clamp(0.15, 1.0)
}

// ---------------------------------------------------------------------------
// Mesh builder
// ---------------------------------------------------------------------------

struct Builder {
    topology: Topology,
    pos: Vec<Vec3>,
    nrm: Vec<Vec3>,
    idx: Vec<u32>,
    groups: Vec<MeshGroup>,
    open: Option<(FaceGroup, usize)>,
    edges: Vec<(u32, u32)>,
}

impl Builder {
    fn new(topology: Topology) -> Self {
        Self {
            topology,
            pos: Vec::new(),
            nrm: Vec::new(),
            idx: Vec::new(),
            groups: Vec::new(),
            open: None,
            edges: Vec::new(),
        }
    }

    /// A crease line between two vertices.
    fn edge(&mut self, a: u32, b: u32) {
        self.edges.push((a, b));
    }

    /// A closed crease loop through `ids`.
    fn edge_loop(&mut self, ids: &[u32]) {
        for (i, &a) in ids.iter().enumerate() {
            self.edge(a, ids[(i + 1) % ids.len()]);
        }
    }

    fn vertex(&mut self, p: Vec3, n: Vec3) -> u32 {
        self.pos.push(p);
        self.nrm.push(n);
        (self.pos.len() - 1) as u32
    }

    /// Start a run of indices in `g`.
    fn group(&mut self, g: FaceGroup) {
        self.close_group();
        self.open = Some((g, self.idx.len()));
    }

    fn close_group(&mut self) {
        if let Some((g, start)) = self.open.take() {
            let count = self.idx.len() - start;
            if count == 0 {
                return;
            }
            // Merge with the previous run when it is the same group and adjacent.
            if let Some(last) = self.groups.last_mut() {
                if last.group == g && (last.start + last.count) as usize == start {
                    last.count += count as u32;
                    return;
                }
            }
            self.groups.push(MeshGroup {
                start: start as u32,
                count: count as u32,
                group: g,
            });
        }
    }

    /// A triangle, wound so its face normal agrees with its vertex normals.
    /// Zero-area triangles (the pole of a revolved face) are dropped.
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        let (pa, pb, pc) = (
            self.pos[a as usize],
            self.pos[b as usize],
            self.pos[c as usize],
        );
        let face = (pb - pa).cross(pc - pa);
        let scale = (pb - pa).length() + (pc - pa).length();
        if face.length() <= 1e-12 * (1.0 + scale * scale) {
            return;
        }
        let n = self.nrm[a as usize] + self.nrm[b as usize] + self.nrm[c as usize];
        if face.dot(n) < 0.0 {
            self.idx.extend_from_slice(&[a, c, b]);
        } else {
            self.idx.extend_from_slice(&[a, b, c]);
        }
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.tri(a, b, c);
        self.tri(a, c, d);
    }

    fn line(&mut self, a: u32, b: u32) {
        self.idx.extend_from_slice(&[a, b]);
    }

    fn finish(mut self, note: Option<String>) -> DisplayMesh {
        self.close_group();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut positions = Vec::with_capacity(self.pos.len() * 3);
        let mut normals = Vec::with_capacity(self.nrm.len() * 3);
        for (p, n) in self.pos.iter().zip(&self.nrm) {
            for (k, v) in [p.x, p.y, p.z].into_iter().enumerate() {
                lo[k] = lo[k].min(v);
                hi[k] = hi[k].max(v);
            }
            positions.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
            let n = n.normalize_or_zero();
            normals.extend_from_slice(&[n.x as f32, n.y as f32, n.z as f32]);
        }
        if self.pos.is_empty() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        // Several pieces meet at the same crease (a lens face and its rim both
        // end on the rim ring); keep each line once, by welded position.
        let key = |p: Vec3| {
            (
                (p.x * 1e6).round() as i64,
                (p.y * 1e6).round() as i64,
                (p.z * 1e6).round() as i64,
            )
        };
        let mut seen = HashSet::new();
        let mut edges = Vec::with_capacity(self.edges.len() * 2);
        for (a, b) in self.edges {
            let (ka, kb) = (key(self.pos[a as usize]), key(self.pos[b as usize]));
            if ka == kb {
                continue;
            }
            if seen.insert(if ka <= kb { (ka, kb) } else { (kb, ka) }) {
                edges.extend_from_slice(&[a, b]);
            }
        }
        DisplayMesh {
            topology: self.topology,
            positions,
            normals,
            indices: self.idx,
            groups: self.groups,
            edges,
            bounds_min: lo,
            bounds_max: hi,
            note,
        }
    }
}

// ---------------------------------------------------------------------------
// Profile sampling
// ---------------------------------------------------------------------------

/// A point on a profile curve and the curve's unit normal there.
#[derive(Debug, Clone, Copy)]
struct Sample {
    p: Vec2,
    n: Vec2,
}

/// Refinement depth limit for one initial interval.
const MAX_DEPTH: u32 = 10;

/// Sample `f` over `[t0, t1]`: `initial` equal steps, each split until the
/// chord and normal-step tolerances hold.
fn adaptive(
    f: &dyn Fn(f64) -> Sample,
    t0: f64,
    t1: f64,
    initial: usize,
    tess: &Tessellation,
) -> Vec<Sample> {
    let mut out = vec![f(t0)];
    let n = initial.max(1);
    for i in 0..n {
        let a = t0 + (t1 - t0) * i as f64 / n as f64;
        let b = t0 + (t1 - t0) * (i + 1) as f64 / n as f64;
        let (sa, sb) = (*out.last().unwrap(), f(b));
        refine(f, a, sa, b, sb, 0, tess, &mut out);
    }
    out
}

#[allow(clippy::too_many_arguments)] // interval, its end samples, depth, limits, sink
fn refine(
    f: &dyn Fn(f64) -> Sample,
    a: f64,
    sa: Sample,
    b: f64,
    sb: Sample,
    depth: u32,
    tess: &Tessellation,
    out: &mut Vec<Sample>,
) {
    if depth < MAX_DEPTH && out.len() < tess.max_curve_samples {
        let m = 0.5 * (a + b);
        let sm = f(m);
        let chord = point_segment_distance(sm.p, sa.p, sb.p);
        let turn = sa.n.dot(sb.n).clamp(-1.0, 1.0).acos();
        if chord > tess.chord_tol_mm || turn > tess.max_normal_step_rad {
            refine(f, a, sa, m, sm, depth + 1, tess, out);
            refine(f, m, sm, b, sb, depth + 1, tess, out);
            return;
        }
    }
    out.push(sb);
}

fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= 0.0 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// Segments for a full circle of radius `r` under the tolerances, rounded up to
/// a multiple of four so the four principal half-planes (the meridional plane
/// among them) are sampled exactly.
fn circle_segments(r: f64, tess: &Tessellation) -> usize {
    let by_chord = if r > tess.chord_tol_mm {
        let step = 2.0 * (1.0 - tess.chord_tol_mm / r).clamp(-1.0, 1.0).acos();
        (TAU / step.max(1e-6)).ceil() as usize
    } else {
        tess.min_segments
    };
    let by_normal = (TAU / tess.max_normal_step_rad).ceil() as usize;
    let n = by_chord
        .max(by_normal)
        .clamp(tess.min_segments, tess.max_segments);
    n.div_ceil(4) * 4
}

/// Sample one profile curve into pieces (a polyline is one piece per segment,
/// so its corners stay sharp).
fn sample_curve(curve: &Curve2, tess: &Tessellation) -> Vec<Vec<Sample>> {
    match curve {
        Curve2::Line { a, b } => line_piece(*a, *b).into_iter().collect(),
        Curve2::Polyline { points } => points
            .windows(2)
            .filter_map(|w| line_piece(w[0], w[1]))
            .collect(),
        Curve2::Arc {
            center,
            radius_mm,
            a0,
            a1,
        } => {
            let (c, r, a0, a1) = (*center, *radius_mm, *a0, *a1);
            if r <= 0.0 || (a1 - a0).abs() < 1e-12 {
                return Vec::new();
            }
            let f = move |t: f64| {
                let d = Vec2::new(t.cos(), t.sin());
                Sample { p: c + d * r, n: d }
            };
            let steps = ((a1 - a0).abs() / TAU * circle_segments(r, tess) as f64).ceil() as usize;
            vec![adaptive(&f, a0, a1, steps.max(2), tess)]
        }
        Curve2::Conic(cp) => vec![sag_curve(cp, &[0.0; 8], tess)],
        Curve2::EvenAsphere { base, coeffs } => vec![sag_curve(base, coeffs, tess)],
        Curve2::SagSegment {
            base,
            coeffs,
            y0,
            y1,
        } => vec![sag_segment_curve(base, coeffs, *y0, *y1, tess)],
    }
}

/// One annular piece of a sag curve, `t` from `t0` to `t1` along `perp(axis)`,
/// sampled the way [`sag_curve`] samples a whole face (a Fresnel facet).
fn sag_segment_curve(
    cp: &ConicParams2,
    coeffs: &[f64; 8],
    t0: f64,
    t1: f64,
    tess: &Tessellation,
) -> Vec<Sample> {
    let axis = cp.axis.normalize();
    let p = perp(axis);
    let (c, k) = (cp.c, cp.k);
    let coeffs = *coeffs;
    let vertex = cp.vertex;
    let f = move |t: f64| {
        let z = even_asphere_sag(c, k, &coeffs, t);
        let n = (-even_asphere_dsag(c, k, &coeffs, t)) * p + axis;
        Sample {
            p: vertex + axis * z + p * t,
            n: n.normalize_or_zero(),
        }
    };
    if t0 == t1 {
        return vec![f(t0)];
    }
    adaptive(&f, t0, t1, 2, tess)
}

fn line_piece(a: Vec2, b: Vec2) -> Option<Vec<Sample>> {
    let e = b - a;
    if e.length_squared() <= 1e-24 {
        return None;
    }
    let n = perp(e).normalize();
    Some(vec![Sample { p: a, n }, Sample { p: b, n }])
}

/// A conic or even-asphere curve across its full aperture, `t` in
/// `[-half_ap, half_ap]` along `perp(axis)`, with the vertex (`t = 0`) always a
/// sample. The point is `vertex + axis·sag(t) + perp(axis)·t`; the normal is the
/// kernel's analytic one.
fn sag_curve(cp: &ConicParams2, coeffs: &[f64; 8], tess: &Tessellation) -> Vec<Sample> {
    let axis = cp.axis.normalize();
    let p = perp(axis);
    let (c, k, ha) = (cp.c, cp.k, cp.half_ap);
    let coeffs = *coeffs;
    let pure = coeffs.iter().all(|a| *a == 0.0);
    let vertex = cp.vertex;
    let f = move |t: f64| {
        let z = even_asphere_sag(c, k, &coeffs, t);
        let n = if pure {
            // The implicit conic's gradient, finite at any aperture.
            (2.0 * c * t) * p + (2.0 * c * (1.0 + k) * z - 2.0) * axis
        } else {
            (-even_asphere_dsag(c, k, &coeffs, t)) * p + axis
        };
        Sample {
            p: vertex + axis * z + p * t,
            n: n.normalize_or_zero(),
        }
    };
    if ha <= 0.0 {
        return vec![f(0.0)];
    }
    let mut left = adaptive(&f, -ha, 0.0, 4, tess);
    let right = adaptive(&f, 0.0, ha, 4, tess);
    left.extend_from_slice(&right[1..]);
    left
}

/// The part of a sampled piece with `y >= 0`, cut exactly at `y = 0`.
fn clip_upper(piece: &[Sample]) -> Vec<Vec<Sample>> {
    let mut out = Vec::new();
    let mut cur: Vec<Sample> = Vec::new();
    for (i, s) in piece.iter().enumerate() {
        let inside = s.p.y >= 0.0;
        if i > 0 {
            let prev = piece[i - 1];
            let was_inside = prev.p.y >= 0.0;
            if inside != was_inside {
                let t = prev.p.y / (prev.p.y - s.p.y);
                let mut cut = Sample {
                    p: prev.p + (s.p - prev.p) * t,
                    n: (prev.n + (s.n - prev.n) * t).normalize_or_zero(),
                };
                cut.p.y = 0.0;
                cur.push(cut);
                if was_inside {
                    out.push(std::mem::take(&mut cur));
                }
            }
        }
        if inside {
            cur.push(*s);
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out.retain(|p| p.len() >= 2 && p.iter().any(|s| s.p.y > 0.0));
    out
}

/// Flip a piece's normals if they point into the profile. The probe is a point
/// on the curve itself (an interior sample, or a line's midpoint), nudged along
/// the normal by a hair.
fn orient_outward(piece: &mut [Sample], profile: &ClosedProfile2, scale: f64) {
    if piece.len() < 2 {
        return;
    }
    let (p, n) = if piece.len() >= 3 {
        let s = piece[piece.len() / 2];
        (s.p, s.n)
    } else {
        (
            (piece[0].p + piece[1].p) * 0.5,
            (piece[0].n + piece[1].n).normalize_or_zero(),
        )
    };
    let eps = 1e-5 * scale.max(1.0);
    if profile_contains_2d(profile, p + n * eps) {
        for s in piece.iter_mut() {
            s.n = -s.n;
        }
    }
}

fn profile_scale(profile: &ClosedProfile2, pieces: &[Vec<Sample>]) -> f64 {
    let _ = profile;
    let mut m: f64 = 1.0;
    for s in pieces.iter().flatten() {
        m = m.max(s.p.x.abs()).max(s.p.y.abs());
    }
    m
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

/// Revolve the upper meridian (`y >= 0`) of a profile about local `+X`.
fn revolve(profile: &ClosedProfile2, tess: &Tessellation) -> DisplayMesh {
    let mut pieces: Vec<(FaceGroup, Vec<Sample>)> = Vec::new();
    for e in &profile.edges {
        let group = match e.curve {
            Curve2::Line { .. } | Curve2::Polyline { .. } => FaceGroup::Rim,
            _ => FaceGroup::Optical,
        };
        for piece in sample_curve(&e.curve, tess) {
            for upper in clip_upper(&piece) {
                pieces.push((group, upper));
            }
        }
    }
    let all: Vec<Vec<Sample>> = pieces.iter().map(|(_, p)| p.clone()).collect();
    let scale = profile_scale(profile, &all);
    let r_max = all.iter().flatten().fold(0.0_f64, |m, s| m.max(s.p.y));
    let segs = circle_segments(r_max, tess);
    let trig: Vec<(f64, f64)> = (0..segs)
        .map(|j| {
            let th = TAU * j as f64 / segs as f64;
            (th.cos(), th.sin())
        })
        .collect();

    let mut m = Builder::new(Topology::Triangles);
    for (group, mut piece) in pieces {
        orient_outward(&mut piece, profile, scale);
        m.group(group);
        let rings: Vec<Vec<u32>> = piece
            .iter()
            .map(|s| {
                trig.iter()
                    .map(|&(c, sn)| {
                        let r = s.p.y.max(0.0);
                        m.vertex(
                            Vec3::new(s.p.x, r * c, r * sn),
                            Vec3::new(s.n.x, s.n.y * c, s.n.y * sn),
                        )
                    })
                    .collect()
            })
            .collect();
        for w in rings.windows(2) {
            for j in 0..segs {
                let j1 = (j + 1) % segs;
                m.quad(w[0][j], w[0][j1], w[1][j1], w[1][j]);
            }
        }
        // Where one profile curve ends and the next begins is a crease ring,
        // unless it is the pole on the axis.
        for (s, ring) in [
            (piece[0], &rings[0]),
            (piece[piece.len() - 1], &rings[rings.len() - 1]),
        ] {
            if s.p.y > 1e-9 * scale {
                m.edge_loop(ring);
            }
        }
    }
    m.finish(None)
}

/// Extrude a closed profile between `z_min` and `z_max`: side walls from the
/// edges, caps from the loop.
fn extrude(profile: &ClosedProfile2, z_min: f64, z_max: f64, tess: &Tessellation) -> DisplayMesh {
    let mut pieces: Vec<Vec<Sample>> = profile
        .edges
        .iter()
        .flat_map(|e| sample_curve(&e.curve, tess))
        .collect();
    let scale = profile_scale(profile, &pieces);
    for p in pieces.iter_mut() {
        orient_outward(p, profile, scale);
    }

    let mut m = Builder::new(Topology::Triangles);
    m.group(FaceGroup::Optical);
    for piece in &pieces {
        let verts: Vec<(u32, u32)> = piece
            .iter()
            .map(|s| {
                let n = Vec3::new(s.n.x, s.n.y, 0.0);
                (
                    m.vertex(Vec3::new(s.p.x, s.p.y, z_min), n),
                    m.vertex(Vec3::new(s.p.x, s.p.y, z_max), n),
                )
            })
            .collect();
        for w in verts.windows(2) {
            m.quad(w[0].0, w[1].0, w[1].1, w[0].1);
        }
        // A curve's two ends are the walls' corners.
        let (first, last) = (verts[0], verts[verts.len() - 1]);
        m.edge(first.0, first.1);
        m.edge(last.0, last.1);
    }

    let outline = chain_loop(&pieces, 1e-6 * scale.max(1.0));
    let tris = triangulate(&outline);
    for (z, nz) in [(z_max, 1.0), (z_min, -1.0)] {
        let n = Vec3::new(0.0, 0.0, nz);
        let ids: Vec<u32> = outline
            .iter()
            .map(|p| m.vertex(Vec3::new(p.x, p.y, z), n))
            .collect();
        for t in &tris {
            m.tri(ids[t[0]], ids[t[1]], ids[t[2]]);
        }
        m.edge_loop(&ids);
    }
    m.finish(None)
}

/// Join sampled pieces end to end into one closed loop, reversing a piece when
/// it runs the other way. Best effort: an unjoinable piece is skipped.
fn chain_loop(pieces: &[Vec<Sample>], eps: f64) -> Vec<Vec2> {
    let usable: Vec<Vec<Vec2>> = pieces
        .iter()
        .filter(|p| p.len() >= 2)
        .map(|p| p.iter().map(|s| s.p).collect())
        .collect();
    if usable.is_empty() {
        return Vec::new();
    }
    let near = |a: Vec2, b: Vec2| (a - b).length() <= eps;
    let mut used = vec![false; usable.len()];
    let mut out = usable[0].clone();
    used[0] = true;
    loop {
        let end = *out.last().unwrap();
        let mut next = None;
        for (i, p) in usable.iter().enumerate() {
            if used[i] {
                continue;
            }
            if near(p[0], end) {
                next = Some((i, false));
                break;
            }
            if near(*p.last().unwrap(), end) {
                next = Some((i, true));
                break;
            }
        }
        let Some((i, rev)) = next else { break };
        used[i] = true;
        if rev {
            out.extend(usable[i].iter().rev().skip(1));
        } else {
            out.extend(usable[i].iter().skip(1));
        }
    }
    if out.len() > 2 && near(out[0], *out.last().unwrap()) {
        out.pop();
    }
    out.dedup_by(|a, b| near(*a, *b));
    out
}

fn signed_area(poly: &[Vec2]) -> f64 {
    let n = poly.len();
    (0..n)
        .map(|i| poly[i].perp_dot(poly[(i + 1) % n]))
        .sum::<f64>()
        * 0.5
}

/// Ear-clipping triangulation of a simple polygon. Returns index triples into
/// `poly`. Falls back to a fan if no ear can be found (a degenerate outline).
fn triangulate(poly: &[Vec2]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let ccw = signed_area(poly) >= 0.0;
    let mut v: Vec<usize> = if ccw {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    };
    let mut out = Vec::with_capacity(n - 2);
    let mut guard = 0;
    while v.len() > 3 && guard < 4 * n * n {
        guard += 1;
        let m = v.len();
        let mut clipped = false;
        for i in 0..m {
            let (a, b, c) = (v[(i + m - 1) % m], v[i], v[(i + 1) % m]);
            let (pa, pb, pc) = (poly[a], poly[b], poly[c]);
            if (pb - pa).perp_dot(pc - pb) <= 0.0 {
                continue; // reflex or straight
            }
            let blocked = v
                .iter()
                .any(|&q| q != a && q != b && q != c && point_in_triangle(poly[q], pa, pb, pc));
            if blocked {
                continue;
            }
            out.push([a, b, c]);
            v.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if v.len() == 3 {
        out.push([v[0], v[1], v[2]]);
    } else if v.len() > 3 {
        for i in 1..v.len() - 1 {
            out.push([v[0], v[i], v[i + 1]]);
        }
    }
    out
}

fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = (b - a).perp_dot(p - a);
    let d2 = (c - b).perp_dot(p - b);
    let d3 = (a - c).perp_dot(p - c);
    d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0
}

fn sphere(r: f64, tess: &Tessellation) -> DisplayMesh {
    let mut m = Builder::new(Topology::Triangles);
    m.group(FaceGroup::Optical);
    ball_into(&mut m, Vec3::ZERO, r, tess);
    m.finish(None)
}

fn cuboid(h: Vec3) -> DisplayMesh {
    let mut m = Builder::new(Topology::Triangles);
    m.group(FaceGroup::Optical);
    box_into(&mut m, -h, h);
    m.finish(None)
}

fn cylinder(r: f64, half_len: f64, axis_local: Vec3, tess: &Tessellation) -> DisplayMesh {
    let mut m = Builder::new(Topology::Triangles);
    m.group(FaceGroup::Optical);
    cylinder_into(&mut m, Vec3::ZERO, r, half_len, axis_local, tess);
    m.finish(None)
}

/// A hollow tube about `+X`: two walls and two end rings, the four rims as
/// creases. The bore's normals face the axis, as the tracer's do.
fn tube(ri: f64, ro: f64, half_len: f64, tess: &Tessellation) -> DisplayMesh {
    let mut m = Builder::new(Topology::Triangles);
    m.group(FaceGroup::Optical);
    let segs = circle_segments(ro, tess);
    let rim = |j: usize| {
        let th = TAU * j as f64 / segs as f64;
        Vec3::new(0.0, th.cos(), th.sin())
    };
    for (r, sign) in [(ro, 1.0), (ri, -1.0)] {
        let wall: Vec<(u32, u32)> = (0..segs)
            .map(|j| {
                let d = rim(j);
                (
                    m.vertex(d * r - Vec3::X * half_len, d * sign),
                    m.vertex(d * r + Vec3::X * half_len, d * sign),
                )
            })
            .collect();
        for j in 0..segs {
            let j1 = (j + 1) % segs;
            m.quad(wall[j].0, wall[j1].0, wall[j1].1, wall[j].1);
        }
    }
    for sign in [-1.0, 1.0] {
        let n = Vec3::X * sign;
        let ring = |r: f64, m: &mut Builder| -> Vec<u32> {
            (0..segs)
                .map(|j| m.vertex(rim(j) * r + n * half_len, n))
                .collect()
        };
        let outer = ring(ro, &mut m);
        let inner = ring(ri, &mut m);
        for j in 0..segs {
            let j1 = (j + 1) % segs;
            m.quad(inner[j], outer[j], outer[j1], inner[j1]);
        }
        m.edge_loop(&outer);
        m.edge_loop(&inner);
    }
    m.finish(None)
}

/// A sphere of radius `r` about `centre`, in the builder's open group.
fn ball_into(m: &mut Builder, centre: Vec3, r: f64, tess: &Tessellation) {
    let segs = circle_segments(r, tess);
    let rows = segs / 2;
    let mut grid = Vec::with_capacity(rows + 1);
    for i in 0..=rows {
        let phi = std::f64::consts::PI * i as f64 / rows as f64;
        let row: Vec<u32> = (0..segs)
            .map(|j| {
                let th = TAU * j as f64 / segs as f64;
                let n = Vec3::new(phi.cos(), phi.sin() * th.cos(), phi.sin() * th.sin());
                m.vertex(centre + n * r, n)
            })
            .collect();
        grid.push(row);
    }
    for w in grid.windows(2) {
        for j in 0..segs {
            let j1 = (j + 1) % segs;
            m.quad(w[0][j], w[0][j1], w[1][j1], w[1][j]);
        }
    }
}

/// The axis-aligned box `lo..hi`, with its twelve edges as creases.
fn box_into(m: &mut Builder, lo: Vec3, hi: Vec3) {
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut n = Vec3::ZERO;
            n[axis] = sign;
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let pick = |s: f64, k: usize| if s < 0.0 { lo[k] } else { hi[k] };
            let corner = |su: f64, sv: f64| {
                let mut p = Vec3::ZERO;
                p[axis] = pick(sign, axis);
                p[u] = pick(su, u);
                p[v] = pick(sv, v);
                p
            };
            let ids = [
                m.vertex(corner(-1.0, -1.0), n),
                m.vertex(corner(1.0, -1.0), n),
                m.vertex(corner(1.0, 1.0), n),
                m.vertex(corner(-1.0, 1.0), n),
            ];
            m.quad(ids[0], ids[1], ids[2], ids[3]);
            m.edge_loop(&ids);
        }
    }
}

/// A capped cylinder about `centre` along `axis_local`, with its rims as
/// creases.
fn cylinder_into(
    m: &mut Builder,
    centre: Vec3,
    r: f64,
    half_len: f64,
    axis_local: Vec3,
    tess: &Tessellation,
) {
    let w = axis_local.normalize_or_zero();
    let w = if w.length_squared() < 0.5 { Vec3::X } else { w };
    let u = w.any_orthonormal_vector();
    let v = w.cross(u);
    let segs = circle_segments(r, tess);
    let rim = |j: usize| {
        let th = TAU * j as f64 / segs as f64;
        u * th.cos() + v * th.sin()
    };
    let side: Vec<(u32, u32)> = (0..segs)
        .map(|j| {
            let d = rim(j);
            (
                m.vertex(centre + d * r - w * half_len, d),
                m.vertex(centre + d * r + w * half_len, d),
            )
        })
        .collect();
    for j in 0..segs {
        let j1 = (j + 1) % segs;
        m.quad(side[j].0, side[j1].0, side[j1].1, side[j].1);
    }
    for sign in [-1.0, 1.0] {
        let n = w * sign;
        let c = m.vertex(centre + n * half_len, n);
        let ring: Vec<u32> = (0..segs)
            .map(|j| m.vertex(centre + rim(j) * r + n * half_len, n))
            .collect();
        for j in 0..segs {
            m.tri(c, ring[j], ring[(j + 1) % segs]);
        }
        m.edge_loop(&ring);
    }
}

/// A profile drawn as line segments in the plane `z = plane_z`, with a note:
/// the stand-in for a body that has no solid to show.
fn outline(
    profile: &ClosedProfile2,
    plane_z: f64,
    tess: &Tessellation,
    note: String,
) -> DisplayMesh {
    let mut m = Builder::new(Topology::Lines);
    m.group(FaceGroup::Optical);
    for e in &profile.edges {
        for piece in sample_curve(&e.curve, tess) {
            let ids: Vec<u32> = piece
                .iter()
                .map(|s| m.vertex(Vec3::new(s.p.x, s.p.y, plane_z), Vec3::ZERO))
                .collect();
            for w in ids.windows(2) {
                m.line(w[0], w[1]);
            }
        }
    }
    m.finish(Some(note))
}

// ---------------------------------------------------------------------------
// Zero-thickness objects
// ---------------------------------------------------------------------------

/// Where a planar object's local `(u, v, w)` lands in its object frame: `w` is
/// along the plane normal.
struct PlaneFrame {
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    n: Vec3,
}

impl PlaneFrame {
    fn local_xy() -> Self {
        Self::from_plane(&Plane3::XY)
    }

    fn from_plane(p: &Plane3) -> Self {
        Self {
            origin: p.origin,
            u: p.u,
            v: p.v,
            n: p.normal,
        }
    }

    fn at(&self, uv: Vec2, w: f64) -> Vec3 {
        self.origin + self.u * uv.x + self.v * uv.y + self.n * w
    }

    fn dir(&self, d: Vec2, w: f64) -> Vec3 {
        self.u * d.x + self.v * d.y + self.n * w
    }
}

#[derive(Clone, Copy)]
struct SlabGroups {
    front: FaceGroup,
    back: FaceGroup,
    edge: FaceGroup,
}

impl SlabGroups {
    fn surface() -> Self {
        Self {
            front: FaceGroup::Front,
            back: FaceGroup::Back,
            edge: FaceGroup::Edge,
        }
    }
}

/// The radius that encloses a planar shape.
fn shape_radius(shape: &PlanarShape2) -> f64 {
    match shape {
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => half_width_mm.hypot(*half_height_mm),
        PlanarShape2::Disc { radius_mm } => *radius_mm,
        PlanarShape2::Annulus { outer_mm, .. } => *outer_mm,
        PlanarShape2::Polygon { vertices } => vertices.iter().fold(0.0, |m, p| m.max(p.length())),
    }
}

/// A planar shape as a slab from `-half` to `+half` along its normal.
fn planar_slab(
    m: &mut Builder,
    shape: &PlanarShape2,
    half: f64,
    frame: &PlaneFrame,
    tess: &Tessellation,
    groups: SlabGroups,
) {
    match shape {
        PlanarShape2::Annulus { inner_mm, outer_mm } => {
            ring_slab(
                m,
                &RadialShape::Circle(*inner_mm),
                &RadialShape::Circle(*outer_mm),
                half,
                frame,
                tess,
                groups,
            );
        }
        PlanarShape2::Disc { radius_mm } => {
            let segs = circle_segments(*radius_mm, tess);
            let outline: Vec<(Vec2, Vec2)> = (0..segs)
                .map(|j| {
                    let th = TAU * j as f64 / segs as f64;
                    let d = Vec2::new(th.cos(), th.sin());
                    (d * *radius_mm, d)
                })
                .collect();
            fan_slab(m, &outline, half, frame, groups, true);
        }
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => {
            let (w, h) = (*half_width_mm, *half_height_mm);
            let corners = [
                Vec2::new(-w, -h),
                Vec2::new(w, -h),
                Vec2::new(w, h),
                Vec2::new(-w, h),
            ];
            polygon_slab(m, &corners, half, frame, groups);
        }
        PlanarShape2::Polygon { vertices } => polygon_slab(m, vertices, half, frame, groups),
    }
}

/// A convex-around-its-centre outline (a circle), faces as fans from the centre.
/// `smooth` gives the edge band radial normals.
fn fan_slab(
    m: &mut Builder,
    outline: &[(Vec2, Vec2)],
    half: f64,
    frame: &PlaneFrame,
    groups: SlabGroups,
    smooth: bool,
) {
    let n = outline.len();
    for (w, g) in [(half, groups.front), (-half, groups.back)] {
        m.group(g);
        let normal = frame.dir(Vec2::ZERO, w.signum());
        let c = m.vertex(frame.at(Vec2::ZERO, w), normal);
        let ring: Vec<u32> = outline
            .iter()
            .map(|(p, _)| m.vertex(frame.at(*p, w), normal))
            .collect();
        for j in 0..n {
            m.tri(c, ring[j], ring[(j + 1) % n]);
        }
        m.edge_loop(&ring);
    }
    m.group(groups.edge);
    let _ = smooth;
    let band: Vec<(u32, u32)> = outline
        .iter()
        .map(|(p, d)| {
            let nn = frame.dir(*d, 0.0);
            (
                m.vertex(frame.at(*p, -half), nn),
                m.vertex(frame.at(*p, half), nn),
            )
        })
        .collect();
    for j in 0..n {
        let j1 = (j + 1) % n;
        m.quad(band[j].0, band[j1].0, band[j1].1, band[j].1);
    }
}

/// A polygon slab: ear-clipped faces and flat edge walls.
fn polygon_slab(
    m: &mut Builder,
    verts: &[Vec2],
    half: f64,
    frame: &PlaneFrame,
    groups: SlabGroups,
) {
    if verts.len() < 3 {
        return;
    }
    let tris = triangulate(verts);
    for (w, g) in [(half, groups.front), (-half, groups.back)] {
        m.group(g);
        let normal = frame.dir(Vec2::ZERO, w.signum());
        let ids: Vec<u32> = verts
            .iter()
            .map(|p| m.vertex(frame.at(*p, w), normal))
            .collect();
        for t in &tris {
            m.tri(ids[t[0]], ids[t[1]], ids[t[2]]);
        }
        m.edge_loop(&ids);
    }
    m.group(groups.edge);
    let ccw = signed_area(verts) >= 0.0;
    let n = verts.len();
    for i in 0..n {
        let (a, b) = (verts[i], verts[(i + 1) % n]);
        let e = b - a;
        if e.length_squared() <= 1e-24 {
            continue;
        }
        // Outward is to the right of a counter-clockwise edge.
        let out = if ccw {
            Vec2::new(e.y, -e.x)
        } else {
            Vec2::new(-e.y, e.x)
        }
        .normalize();
        let nn = frame.dir(out, 0.0);
        let q = [
            m.vertex(frame.at(a, -half), nn),
            m.vertex(frame.at(b, -half), nn),
            m.vertex(frame.at(b, half), nn),
            m.vertex(frame.at(a, half), nn),
        ];
        m.quad(q[0], q[1], q[2], q[3]);
        m.edge(q[0], q[3]);
    }
}

/// A boundary described by its distance from the centre at each angle. Exact
/// for circles; rectangles and polygons add their corner angles to the sampling
/// so corners are hit exactly. Valid for shapes star-shaped about the centre,
/// which every authored opening and mask is.
#[derive(Clone)]
enum RadialShape {
    Circle(f64),
    Rect {
        hw: f64,
        hh: f64,
    },
    Polygon(Vec<Vec2>),
    /// Never closer to the centre than `min`.
    AtLeast(Box<RadialShape>, f64),
}

impl RadialShape {
    /// The outer boundary of a shape (an annulus contributes its outer circle).
    fn of(shape: &PlanarShape2, _outer: bool) -> Self {
        match shape {
            PlanarShape2::Rectangle {
                half_width_mm,
                half_height_mm,
            } => Self::Rect {
                hw: *half_width_mm,
                hh: *half_height_mm,
            },
            PlanarShape2::Disc { radius_mm } => Self::Circle(*radius_mm),
            PlanarShape2::Annulus { outer_mm, .. } => Self::Circle(*outer_mm),
            PlanarShape2::Polygon { vertices } => Self::Polygon(vertices.clone()),
        }
    }

    fn at_least(self, min: f64) -> Self {
        Self::AtLeast(Box::new(self), min)
    }

    fn rho(&self, th: f64) -> f64 {
        let (c, s) = (th.cos(), th.sin());
        match self {
            Self::Circle(r) => *r,
            Self::Rect { hw, hh } => {
                let tx = if c.abs() > 1e-15 {
                    hw / c.abs()
                } else {
                    f64::INFINITY
                };
                let ty = if s.abs() > 1e-15 {
                    hh / s.abs()
                } else {
                    f64::INFINITY
                };
                tx.min(ty)
            }
            Self::Polygon(vs) => {
                let d = Vec2::new(c, s);
                let mut best: f64 = 0.0;
                for i in 0..vs.len() {
                    let (a, b) = (vs[i], vs[(i + 1) % vs.len()]);
                    let e = b - a;
                    let det = d.perp_dot(e);
                    if det.abs() < 1e-15 {
                        continue;
                    }
                    let t = a.perp_dot(e) / det;
                    let u = a.perp_dot(d) / det;
                    if t > 0.0 && (-1e-12..=1.0 + 1e-12).contains(&u) {
                        best = best.max(t);
                    }
                }
                best
            }
            Self::AtLeast(inner, min) => inner.rho(th).max(*min),
        }
    }

    fn radius(&self) -> f64 {
        match self {
            Self::Circle(r) => *r,
            Self::Rect { hw, hh } => hw.hypot(*hh),
            Self::Polygon(vs) => vs.iter().fold(0.0, |m, p| m.max(p.length())),
            Self::AtLeast(inner, min) => inner.radius().max(*min),
        }
    }

    fn corners(&self, out: &mut Vec<f64>) {
        match self {
            Self::Circle(_) => {}
            Self::Rect { hw, hh } => {
                for (x, y) in [(*hw, *hh), (-*hw, *hh), (-*hw, -*hh), (*hw, -*hh)] {
                    out.push(y.atan2(x));
                }
            }
            Self::Polygon(vs) => out.extend(vs.iter().map(|p| p.y.atan2(p.x))),
            Self::AtLeast(inner, _) => inner.corners(out),
        }
    }
}

/// The region between two star-shaped boundaries, as a slab.
fn ring_slab(
    m: &mut Builder,
    inner: &RadialShape,
    outer: &RadialShape,
    half: f64,
    frame: &PlaneFrame,
    tess: &Tessellation,
    groups: SlabGroups,
) {
    let segs = circle_segments(outer.radius(), tess);
    let mut angles: Vec<f64> = (0..segs).map(|j| TAU * j as f64 / segs as f64).collect();
    let mut extra = Vec::new();
    inner.corners(&mut extra);
    outer.corners(&mut extra);
    angles.extend(extra.into_iter().map(|a| a.rem_euclid(TAU)));
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    angles.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let pts = |shape: &RadialShape| -> Vec<Vec2> {
        angles
            .iter()
            .map(|&a| Vec2::new(a.cos(), a.sin()) * shape.rho(a))
            .collect()
    };
    let (pin, pout) = (pts(inner), pts(outer));
    let n = angles.len();

    for (w, g) in [(half, groups.front), (-half, groups.back)] {
        m.group(g);
        let normal = frame.dir(Vec2::ZERO, w.signum());
        let a: Vec<u32> = pin
            .iter()
            .map(|p| m.vertex(frame.at(*p, w), normal))
            .collect();
        let b: Vec<u32> = pout
            .iter()
            .map(|p| m.vertex(frame.at(*p, w), normal))
            .collect();
        for j in 0..n {
            let j1 = (j + 1) % n;
            m.quad(a[j], a[j1], b[j1], b[j]);
        }
        m.edge_loop(&b);
        if pin.iter().any(|p| p.length_squared() > 1e-24) {
            m.edge_loop(&a);
        }
    }

    // Edge walls: the outer one faces away from the centre, the inner one toward
    // it. Normals come from the boundary's tangent, so straight sides are flat.
    m.group(groups.edge);
    for (ring, outward) in [(&pout, 1.0), (&pin, -1.0)] {
        if ring.iter().all(|p| p.length_squared() < 1e-24) {
            continue;
        }
        let band: Vec<(u32, u32)> = (0..n)
            .map(|j| {
                let t = ring[(j + 1) % n] - ring[(j + n - 1) % n];
                let mut d = Vec2::new(t.y, -t.x).normalize_or_zero();
                if d.dot(ring[j]) * outward < 0.0 {
                    d = -d;
                }
                let nn = frame.dir(d, 0.0);
                (
                    m.vertex(frame.at(ring[j], -half), nn),
                    m.vertex(frame.at(ring[j], half), nn),
                )
            })
            .collect();
        for j in 0..n {
            let j1 = (j + 1) % n;
            m.quad(band[j].0, band[j1].0, band[j1].1, band[j].1);
        }
    }
}

/// A zero-thickness surface of revolution about local `+Z` (sag along `+Z`),
/// as two sheets `half` apart along the axis, joined at the rim and at any
/// central hole.
fn revolved_sheet(
    m: &mut Builder,
    sag: &SagModel,
    ap: &RadialAperture,
    half: f64,
    tess: &Tessellation,
) {
    let (c, k, coeffs) = match sag {
        SagModel::Plane => (0.0, 0.0, [0.0; 8]),
        SagModel::Conic {
            curvature_per_mm,
            conic,
        } => (*curvature_per_mm, *conic, [0.0; 8]),
        SagModel::EvenAsphere {
            curvature_per_mm,
            conic,
            coeffs,
        } => (*curvature_per_mm, *conic, *coeffs),
    };
    // The meridian in (r, z): r along local +X at angle 0, z the sag.
    let f = move |r: f64| {
        let z = even_asphere_sag(c, k, &coeffs, r);
        let dz = even_asphere_dsag(c, k, &coeffs, r);
        Sample {
            p: Vec2::new(r, z),
            n: Vec2::new(-dz, 1.0).normalize(),
        }
    };
    let (r0, r1) = (ap.inner_radius_mm.max(0.0), ap.outer_radius_mm);
    let meridian = adaptive(&f, r0, r1, 4, tess);
    let segs = circle_segments(r1, tess);
    let trig: Vec<(f64, f64)> = (0..segs)
        .map(|j| {
            let th = TAU * j as f64 / segs as f64;
            (th.cos(), th.sin())
        })
        .collect();
    for (off, g, flip) in [
        (half, FaceGroup::Front, 1.0),
        (-half, FaceGroup::Back, -1.0),
    ] {
        m.group(g);
        let rings: Vec<Vec<u32>> = meridian
            .iter()
            .map(|s| {
                trig.iter()
                    .map(|&(cs, sn)| {
                        let (r, z) = (s.p.x, s.p.y + off);
                        let n = Vec3::new(s.n.x * cs, s.n.x * sn, s.n.y) * flip;
                        m.vertex(Vec3::new(r * cs, r * sn, z), n)
                    })
                    .collect()
            })
            .collect();
        for w in rings.windows(2) {
            for j in 0..segs {
                let j1 = (j + 1) % segs;
                m.quad(w[0][j], w[0][j1], w[1][j1], w[1][j]);
            }
        }
        m.edge_loop(&rings[rings.len() - 1]);
        if r0 > 0.0 {
            m.edge_loop(&rings[0]);
        }
    }
    m.group(FaceGroup::Edge);
    let walls = [(*meridian.last().unwrap(), 1.0), (meridian[0], -1.0)];
    for (s, out) in walls {
        if s.p.x <= 0.0 {
            continue; // a solid centre has no inner wall
        }
        let band: Vec<(u32, u32)> = trig
            .iter()
            .map(|&(cs, sn)| {
                let n = Vec3::new(cs, sn, 0.0) * out;
                let (r, z) = (s.p.x, s.p.y);
                (
                    m.vertex(Vec3::new(r * cs, r * sn, z - half), n),
                    m.vertex(Vec3::new(r * cs, r * sn, z + half), n),
                )
            })
            .collect();
        for j in 0..segs {
            let j1 = (j + 1) % segs;
            m.quad(band[j].0, band[j1].0, band[j1].1, band[j].1);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build3::SingletGeom;
    use crate::profile2::ProfileSolid3;
    use crate::vec3::Isometry3;
    use std::collections::HashMap;

    fn tess() -> Tessellation {
        Tessellation::default()
    }

    fn singlet(g: SingletGeom) -> (Scene3, u64) {
        let mut s = Scene3::new();
        let id = s.add_singlet(Vec3::ZERO, g, 1.5168, 64.17);
        (s, id.0)
    }

    fn vertex(m: &DisplayMesh, i: usize) -> Vec3 {
        Vec3::new(
            m.positions[3 * i] as f64,
            m.positions[3 * i + 1] as f64,
            m.positions[3 * i + 2] as f64,
        )
    }

    fn normal(m: &DisplayMesh, i: usize) -> Vec3 {
        Vec3::new(
            m.normals[3 * i] as f64,
            m.normals[3 * i + 1] as f64,
            m.normals[3 * i + 2] as f64,
        )
    }

    fn group_vertices(m: &DisplayMesh, g: FaceGroup) -> Vec<usize> {
        let mut v: Vec<usize> = m
            .groups
            .iter()
            .filter(|r| r.group == g)
            .flat_map(|r| {
                m.indices[r.start as usize..(r.start + r.count) as usize]
                    .iter()
                    .map(|&i| i as usize)
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Every undirected edge, after welding coincident vertices, is used by
    /// exactly two triangles: the surface is closed.
    fn assert_watertight(m: &DisplayMesh) {
        assert_eq!(m.topology, Topology::Triangles);
        let key = |i: u32| {
            let p = vertex(m, i as usize);
            (
                (p.x * 1e4).round() as i64,
                (p.y * 1e4).round() as i64,
                (p.z * 1e4).round() as i64,
            )
        };
        let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut id = |i: u32| {
            let k = key(i);
            let n = weld.len();
            *weld.entry(k).or_insert(n)
        };
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for t in m.indices.chunks(3) {
            let v = [id(t[0]), id(t[1]), id(t[2])];
            for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let open: Vec<_> = edges.iter().filter(|(_, &c)| c != 2).collect();
        assert!(
            open.is_empty(),
            "{} edges not shared by two triangles",
            open.len()
        );
    }

    /// The divergence-theorem volume; positive when the triangles face outward.
    fn signed_volume(m: &DisplayMesh) -> f64 {
        m.indices
            .chunks(3)
            .map(|t| {
                let (a, b, c) = (
                    vertex(m, t[0] as usize),
                    vertex(m, t[1] as usize),
                    vertex(m, t[2] as usize),
                );
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// Each triangle's winding agrees with its vertex normals.
    fn assert_faces_outward(m: &DisplayMesh) {
        for t in m.indices.chunks(3) {
            let (a, b, c) = (
                vertex(m, t[0] as usize),
                vertex(m, t[1] as usize),
                vertex(m, t[2] as usize),
            );
            let face = (b - a).cross(c - a);
            let n = normal(m, t[0] as usize) + normal(m, t[1] as usize) + normal(m, t[2] as usize);
            assert!(face.dot(n) > 0.0, "a triangle is wound against its normals");
        }
    }

    fn assert_bounds(m: &DisplayMesh) {
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        for i in 0..m.vertex_count() {
            lo = lo.min(vertex(m, i));
            hi = hi.max(vertex(m, i));
        }
        for k in 0..3 {
            assert!((lo[k] - m.bounds_min[k]).abs() < 1e-5, "bounds min {k}");
            assert!((hi[k] - m.bounds_max[k]).abs() < 1e-5, "bounds max {k}");
        }
    }

    /// The analytic sag of a singlet face at radius `r`, and which face `x` is on.
    fn face_x(g: &SingletGeom, front: bool, r: f64) -> f64 {
        let (z0, radius, k) = if front {
            (-g.thickness_mm / 2.0, g.r1_mm, g.conic1)
        } else {
            (g.thickness_mm / 2.0, g.r2_mm, g.conic2)
        };
        let c = if radius.is_finite() {
            1.0 / radius
        } else {
            0.0
        };
        z0 + even_asphere_sag(c, k, &[0.0; 8], r)
    }

    fn check_singlet(g: SingletGeom) -> DisplayMesh {
        let (s, id) = singlet(g);
        let m = object_mesh(&s, id, &tess()).unwrap();
        assert!(m.note.is_none());
        assert_watertight(&m);
        assert_faces_outward(&m);
        assert!(signed_volume(&m) > 0.0);
        assert_bounds(&m);
        // Optical vertices lie on the analytic faces, and their normals are the
        // analytic ones.
        for i in group_vertices(&m, FaceGroup::Optical) {
            let p = vertex(&m, i);
            let r = p.y.hypot(p.z);
            let front = p.x < 0.5 * (face_x(&g, true, r) + face_x(&g, false, r));
            let x = face_x(&g, front, r);
            assert!(
                (p.x - x).abs() < 2e-5,
                "vertex off its face by {}",
                (p.x - x).abs()
            );
            let radius = if front { g.r1_mm } else { g.r2_mm };
            let k = if front { g.conic1 } else { g.conic2 };
            let c = if radius.is_finite() {
                1.0 / radius
            } else {
                0.0
            };
            let ds = even_asphere_dsag(c, k, &[0.0; 8], r);
            let mut want =
                Vec3::new(1.0, -ds * p.y / r.max(1e-30), -ds * p.z / r.max(1e-30)).normalize();
            if front {
                want = -want;
            }
            if r < 1e-9 {
                want = Vec3::new(if front { -1.0 } else { 1.0 }, 0.0, 0.0);
            }
            let got = normal(&m, i);
            assert!(
                got.dot(want) > 1.0 - 1e-6,
                "normal off by {} rad",
                got.dot(want).clamp(-1.0, 1.0).acos()
            );
        }
        // The rim is a cylinder at the semi-aperture.
        for i in group_vertices(&m, FaceGroup::Rim) {
            let p = vertex(&m, i);
            assert!((p.y.hypot(p.z) - g.semi_aperture_mm).abs() < 1e-4);
        }
        m
    }

    #[test]
    fn a_biconvex_singlet_is_a_closed_solid_on_its_analytic_faces() {
        let g = SingletGeom::biconvex();
        let m = check_singlet(g);
        // Volume against a fine numerical integral of the same faces.
        let n = 4000;
        let mut v = 0.0;
        for i in 0..n {
            let r = g.semi_aperture_mm * (i as f64 + 0.5) / n as f64;
            let len = face_x(&g, false, r) - face_x(&g, true, r);
            v += TAU * r * len * g.semi_aperture_mm / n as f64;
        }
        let rel = (signed_volume(&m) - v).abs() / v;
        assert!(rel < 5e-3, "volume off by {:.3}%", rel * 100.0);
    }

    #[test]
    fn meniscus_plano_and_steep_singlets_tessellate() {
        for g in [
            SingletGeom {
                r1_mm: 30.0,
                r2_mm: 60.0,
                thickness_mm: 5.0,
                semi_aperture_mm: 10.0,
                conic1: 0.0,
                conic2: 0.0,
            },
            SingletGeom {
                r1_mm: f64::INFINITY,
                r2_mm: -40.0,
                thickness_mm: 4.0,
                semi_aperture_mm: 12.0,
                conic1: 0.0,
                conic2: 0.0,
            },
            SingletGeom {
                r1_mm: 12.5,
                r2_mm: -12.5,
                thickness_mm: 20.0,
                semi_aperture_mm: 12.0,
                conic1: 0.0,
                conic2: 0.0,
            },
            SingletGeom {
                r1_mm: -80.0,
                r2_mm: 80.0,
                thickness_mm: 6.0,
                semi_aperture_mm: 10.0,
                conic1: 0.0,
                conic2: 0.0,
            },
            SingletGeom {
                r1_mm: 50.0,
                r2_mm: -50.0,
                thickness_mm: 4.0,
                semi_aperture_mm: 12.0,
                conic1: -1.0,
                conic2: -0.5,
            },
            SingletGeom {
                r1_mm: 1e6,
                r2_mm: -1e6,
                thickness_mm: 0.3,
                semi_aperture_mm: 0.05,
                conic1: 0.0,
                conic2: 0.0,
            },
        ] {
            check_singlet(g);
        }
    }

    #[test]
    fn the_chord_tolerance_holds_along_the_meridian() {
        // Steep faces (sag 6.8 mm at the 12 mm rim), thick enough not to cross.
        let g = SingletGeom {
            r1_mm: 14.0,
            r2_mm: -14.0,
            thickness_mm: 16.0,
            semi_aperture_mm: 12.0,
            conic1: 0.0,
            conic2: 0.0,
        };
        let (s, id) = singlet(g);
        let t = tess();
        let m = object_mesh(&s, id, &t).unwrap();
        assert!(m.note.is_none(), "{:?}", m.note);
        // The theta = 0 meridian: vertices with z == 0 and y >= 0 on the front
        // face. This lens is steep enough that its front face crosses x = 0, so
        // the face is told apart by comparing against both analytic faces.
        let on_front = |p: &Vec3| p.x < 0.5 * (face_x(&g, true, p.y) + face_x(&g, false, p.y));
        let mut pts: Vec<Vec3> = group_vertices(&m, FaceGroup::Optical)
            .into_iter()
            .map(|i| vertex(&m, i))
            .filter(|p| p.z.abs() < 1e-9 && p.y >= 0.0 && on_front(p))
            .collect();
        pts.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
        assert!(pts.len() >= 8);
        for w in pts.windows(2) {
            let r = 0.5 * (w[0].y + w[1].y);
            let mid = Vec2::new(face_x(&g, true, r), r);
            let d =
                point_segment_distance(mid, Vec2::new(w[0].x, w[0].y), Vec2::new(w[1].x, w[1].y));
            assert!(
                d <= t.chord_tol_mm * 1.05,
                "chord error {d} between {:?} and {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn an_even_asphere_face_follows_its_polynomial() {
        let coeffs = [2e-5, -1e-7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (ha, c) = (10.0, 1.0 / 40.0);
        let ftip = -2.0 + even_asphere_sag(c, 0.0, &coeffs, ha);
        let rtip = 2.0;
        let profile = ClosedProfile2 {
            edges: [
                Curve2::EvenAsphere {
                    base: ConicParams2 {
                        vertex: Vec2::new(-2.0, 0.0),
                        axis: Vec2::X,
                        c,
                        k: 0.0,
                        half_ap: ha,
                    },
                    coeffs,
                },
                Curve2::Line {
                    a: Vec2::new(ftip, ha),
                    b: Vec2::new(rtip, ha),
                },
                Curve2::Line {
                    a: Vec2::new(rtip, ha),
                    b: Vec2::new(rtip, -ha),
                },
                Curve2::Line {
                    a: Vec2::new(rtip, -ha),
                    b: Vec2::new(ftip, -ha),
                },
            ]
            .into_iter()
            .enumerate()
            .map(|(i, curve)| crate::profile2::ProfileEdge2 {
                patch: crate::id::PatchId(i as u64 + 1),
                curve,
            })
            .collect(),
        };
        let mut s = Scene3::new();
        let id = s
            .add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5, 60.0)
            .0;
        s.bodies[0].geometry = SolidGeometry3::Profile(ProfileSolid3 {
            profile,
            construction: ProfileConstruction3::RevolveLocalX,
        });
        let m = object_mesh(&s, id, &tess()).unwrap();
        assert!(m.note.is_none(), "{:?}", m.note);
        assert_watertight(&m);
        assert_faces_outward(&m);
        for i in group_vertices(&m, FaceGroup::Optical) {
            let p = vertex(&m, i);
            let r = p.y.hypot(p.z);
            let want = -2.0 + even_asphere_sag(c, 0.0, &coeffs, r);
            assert!((p.x - want).abs() < 2e-5);
            let ds = even_asphere_dsag(c, 0.0, &coeffs, r);
            let mut want_n = Vec3::new(-1.0, 0.0, 0.0);
            if r > 1e-9 {
                want_n = -Vec3::new(1.0, -ds * p.y / r, -ds * p.z / r).normalize();
            }
            assert!(normal(&m, i).dot(want_n) > 1.0 - 1e-6);
        }
    }

    #[test]
    fn a_prism_is_a_closed_extrusion() {
        let mut s = Scene3::new();
        let id = s.add_prism(Vec3::ZERO, 16.0, 0.0, 1.62, 36.0);
        let m = object_mesh(&s, id, &tess()).unwrap();
        assert!(m.note.is_none(), "{:?}", m.note);
        assert_watertight(&m);
        assert_faces_outward(&m);
        assert!(signed_volume(&m) > 0.0);
        assert_bounds(&m);
        // An equilateral triangle of side 16 extruded over its Z range.
        let SolidGeometry3::Profile(p) = &s.bodies[0].geometry else {
            panic!()
        };
        let ProfileConstruction3::ExtrudeLocalZ { z_min_mm, z_max_mm } = p.construction else {
            panic!()
        };
        let area = 3f64.sqrt() / 4.0 * 16.0 * 16.0;
        let want = area * (z_max_mm - z_min_mm);
        assert!((signed_volume(&m) - want).abs() / want < 1e-4);
    }

    #[test]
    fn the_mesh_does_not_depend_on_placement() {
        let (mut s, id) = singlet(SingletGeom::biconvex());
        let before = object_mesh(&s, id, &tess()).unwrap();
        s.bodies[0].common.transform =
            Isometry3::new(Vec3::new(10.0, -3.0, 2.0), [0.0, 0.0, 0.3826834, 0.9238795]).unwrap();
        assert_eq!(before, object_mesh(&s, id, &tess()).unwrap());
    }

    #[test]
    fn a_2d_only_body_is_an_outline_with_a_note() {
        let (mut s, id) = singlet(SingletGeom::biconvex());
        if let SolidGeometry3::Profile(p) = &mut s.bodies[0].geometry {
            p.construction = ProfileConstruction3::PlanarOnly { plane_z_mm: 0.0 };
        }
        let m = object_mesh(&s, id, &tess()).unwrap();
        assert_eq!(m.topology, Topology::Lines);
        assert!(m.note.as_deref().unwrap().contains("2D-only"));
        assert!(m.positions.chunks(3).all(|p| p[2] == 0.0));
    }

    #[test]
    fn surfaces_detectors_and_apertures_are_closed_slabs() {
        let mut s = Scene3::new();
        let mirror = s
            .add_planar_mirror(Vec3::new(0.0, -5.0, 0.0), Vec3::new(5.0, 5.0, 0.0), 0.95)
            .unwrap();
        let det = s
            .add_area_detector(Vec3::X * 40.0, -Vec3::X, 6.0, 4.0, [8, 8])
            .0;
        let ap = s.add_aperture(Vec3::X * -10.0, 3.0, 0.0);
        for id in [mirror, det, ap] {
            let m = object_mesh(&s, id, &tess()).unwrap();
            assert_watertight(&m);
            assert_faces_outward(&m);
            assert!(signed_volume(&m) > 0.0);
            assert_bounds(&m);
            // Display-only thickness: thin, and centred on the plane.
            assert!(m.bounds_max[2] <= 1.0 + 1e-9 && m.bounds_min[2] >= -1.0 - 1e-9);
            assert!((m.bounds_max[2] + m.bounds_min[2]).abs() < 1e-9);
        }
        // The detector's faces are sensitive, its edge is housing.
        let m = object_mesh(&s, det, &tess()).unwrap();
        let groups: Vec<FaceGroup> = m.groups.iter().map(|g| g.group).collect();
        assert!(groups.contains(&FaceGroup::DetectorActive));
        assert!(groups.contains(&FaceGroup::DetectorHousing));
        assert!((m.bounds_max[0] - 6.0).abs() < 1e-6 && (m.bounds_max[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn an_aperture_opening_matches_its_radius() {
        let mut s = Scene3::new();
        let id = s.add_aperture(Vec3::ZERO, 3.0, 0.0);
        assert!(s.set_aperture_opening(id, 1.0, 3.0));
        let m = object_mesh(&s, id, &tess()).unwrap();
        assert_watertight(&m);
        let radii: Vec<f64> = (0..m.vertex_count())
            .map(|i| {
                let p = vertex(&m, i);
                p.x.hypot(p.y)
            })
            .collect();
        // Nothing opaque inside the clear ring (1 mm .. 3 mm).
        assert!(radii.iter().all(|r| *r <= 1.0 + 1e-5 || *r >= 3.0 - 1e-5));
        // Both boundaries are drawn exactly.
        assert!(radii.iter().any(|r| (r - 3.0).abs() < 1e-5));
        assert!(radii.iter().any(|r| (r - 1.0).abs() < 1e-5));
    }

    #[test]
    fn a_rectangular_opening_keeps_its_corners() {
        let mut s = Scene3::new();
        let id = s.add_aperture(Vec3::ZERO, 3.0, 0.0);
        s.apertures[0].shape.clear_opening = PlanarShape2::Rectangle {
            half_width_mm: 2.0,
            half_height_mm: 1.0,
        };
        let m = object_mesh(&s, id, &tess()).unwrap();
        assert_watertight(&m);
        let has = |x: f64, y: f64| {
            (0..m.vertex_count()).any(|i| {
                let p = vertex(&m, i);
                (p.x - x).abs() < 1e-5 && (p.y - y).abs() < 1e-5
            })
        };
        assert!(has(2.0, 1.0) && has(-2.0, 1.0) && has(-2.0, -1.0) && has(2.0, -1.0));
    }

    #[test]
    fn analytic_solids_are_closed() {
        let t = tess();
        for m in [
            sphere(5.0, &t),
            cuboid(Vec3::new(1.0, 2.0, 3.0)),
            cylinder(2.0, 4.0, Vec3::new(0.0, 1.0, 1.0), &t),
        ] {
            assert_watertight(&m);
            assert_faces_outward(&m);
            assert!(signed_volume(&m) > 0.0);
        }
        assert!((signed_volume(&cuboid(Vec3::new(1.0, 2.0, 3.0))) - 48.0).abs() < 1e-4);
    }

    #[test]
    fn a_curved_surface_object_is_a_closed_sheet() {
        let rev = SurfaceGeometry3::SurfaceOfRevolution {
            sag: SagModel::Conic {
                curvature_per_mm: 1.0 / 30.0,
                conic: -1.0,
            },
            radial_aperture: RadialAperture {
                inner_radius_mm: 2.0,
                outer_radius_mm: 10.0,
            },
        };
        let mut s = Scene3::new();
        let id = s
            .add_planar_mirror(Vec3::ZERO, Vec3::Y * 10.0, 0.9)
            .unwrap();
        s.surfaces[0].geometry = rev;
        let m = object_mesh(&s, id, &tess()).unwrap();
        assert_watertight(&m);
        assert_faces_outward(&m);
        assert!(signed_volume(&m) > 0.0);
    }

    /// The crease lines, as welded position pairs.
    fn edge_lines(m: &DisplayMesh) -> Vec<(Vec3, Vec3)> {
        m.edges
            .chunks(2)
            .map(|e| (vertex(m, e[0] as usize), vertex(m, e[1] as usize)))
            .collect()
    }

    #[test]
    fn creases_are_the_rims_corners_and_outlines() {
        let t = tess();
        // A lens: the two rim rings, nothing on its smooth faces.
        let (s, id) = singlet(SingletGeom::biconvex());
        let lens = object_mesh(&s, id, &t).unwrap();
        let rings = edge_lines(&lens);
        let segs = circle_segments(12.0, &t);
        assert_eq!(rings.len(), 2 * segs);
        for (a, b) in &rings {
            assert!((a.y.hypot(a.z) - 12.0).abs() < 1e-4 && (b.y.hypot(b.z) - 12.0).abs() < 1e-4);
        }
        // A prism: three vertical corners and two triangular outlines.
        let mut s = Scene3::new();
        let prism = s.add_prism(Vec3::ZERO, 16.0, 0.0, 1.62, 36.0);
        assert_eq!(object_mesh(&s, prism, &t).unwrap().edges.len() / 2, 9);
        // A box has twelve edges; a rectangle slab has twelve too.
        assert_eq!(cuboid(Vec3::new(1.0, 2.0, 3.0)).edges.len() / 2, 12);
        let det = s.add_area_detector(Vec3::ZERO, Vec3::X, 4.0, 3.0, [4, 4]).0;
        assert_eq!(object_mesh(&s, det, &t).unwrap().edges.len() / 2, 12);
        // A sphere has none.
        assert!(sphere(3.0, &t).edges.is_empty());
        // No line is listed twice.
        let mut seen = HashSet::new();
        for (a, b) in rings {
            let k = |p: Vec3| ((p.x * 1e4) as i64, (p.y * 1e4) as i64, (p.z * 1e4) as i64);
            let (ka, kb) = (k(a), k(b));
            assert!(seen.insert(if ka <= kb { (ka, kb) } else { (kb, ka) }));
        }
    }

    #[test]
    fn a_source_is_a_glyph_behind_its_emission_plane() {
        let t = tess();
        let mut s = Scene3::default_scene();
        let id = s.sources[0].id.0;
        let shapes = [
            SpatialDistribution3::Rectangle {
                half_width_mm: 10.0,
                half_height_mm: 0.25,
            },
            SpatialDistribution3::Disc { radius_mm: 4.0 },
        ];
        for spatial in shapes {
            for dir in [Vec3::X, -Vec3::X] {
                s.sources[0].spatial = spatial;
                s.sources[0].angular = AngularDistribution3::Direction { local_dir: dir };
                let m = object_mesh(&s, id, &t).unwrap();
                assert_watertight(&m);
                assert_faces_outward(&m);
                assert!(signed_volume(&m) > 0.0);
                assert_bounds(&m);
                // Nothing is drawn on the side the light leaves toward.
                let ahead = (0..m.vertex_count()).map(|i| vertex(&m, i).dot(dir));
                assert!(ahead.fold(f64::NEG_INFINITY, f64::max).abs() < 1e-6);
                // The emitting face lies in the emission plane and faces the light.
                let front: Vec<usize> = group_vertices(&m, FaceGroup::SourceEmitter);
                assert!(!front.is_empty());
                for i in front {
                    assert!(vertex(&m, i).x.abs() < 1e-6);
                    assert!((normal(&m, i) - dir).length() < 1e-6);
                }
                let groups: Vec<FaceGroup> = m.groups.iter().map(|g| g.group).collect();
                assert!(groups.contains(&FaceGroup::SourceHousing));
                assert!(!m.edges.is_empty());
            }
        }
        // The aperture is drawn at its size: a 20 mm wide slit spans local Y.
        s.sources[0].spatial = shapes[0];
        s.sources[0].angular = AngularDistribution3::Direction { local_dir: Vec3::X };
        let m = object_mesh(&s, id, &t).unwrap();
        let emit = group_vertices(&m, FaceGroup::SourceEmitter);
        let top = emit
            .iter()
            .map(|&i| vertex(&m, i).y)
            .fold(f64::NEG_INFINITY, f64::max);
        let deep = emit
            .iter()
            .map(|&i| vertex(&m, i).z)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((top - 10.0).abs() < 1e-6 && (deep - 0.25).abs() < 1e-6);

        // A point source: a closed ball and stem, the ball centred on the point.
        s.sources[0].spatial = SpatialDistribution3::Point;
        s.sources[0].angular = AngularDistribution3::UniformConeSolidAngle {
            half_angle_rad: 0.2,
        };
        let m = object_mesh(&s, id, &t).unwrap();
        assert_watertight(&m);
        assert_faces_outward(&m);
        let ball = group_vertices(&m, FaceGroup::SourceEmitter);
        assert!(ball
            .iter()
            .all(|&i| (vertex(&m, i).length() - 1.0).abs() < 1e-6));

        // A legacy line source has no area: a line and a note.
        s.sources[0].spatial = SpatialDistribution3::LineLegacy { half_width_mm: 5.0 };
        let m = object_mesh(&s, id, &t).unwrap();
        assert_eq!(m.topology, Topology::Lines);
        assert!(m.note.as_deref().unwrap().contains("legacy line source"));
    }

    #[test]
    fn segment_counts_are_multiples_of_four_within_limits() {
        let t = tess();
        for r in [0.01, 1.0, 12.0, 100.0, 1e4] {
            let n = circle_segments(r, &t);
            assert_eq!(n % 4, 0);
            assert!(n >= t.min_segments && n <= t.max_segments.div_ceil(4) * 4);
        }
    }
}
