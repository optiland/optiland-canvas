//! Shared builders for the 3D trace suites (`analytic3`, `energy3`).
//!
//! Small, explicit scene constructors so each test reads as the optical setup it
//! is rather than as struct boilerplate. Not every test uses every builder, so
//! dead code is allowed here.
#![allow(dead_code)]

pub use oc_core::scene3::Scene3;

use oc_core::id::{LayerId, ObjectId};
use oc_core::material::{Material, LAMBDA_D};
use oc_core::profile2::{
    ClosedProfile2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use oc_core::scene3::{
    Aperture3, ApertureInteraction, ApertureShape3, Body3, Detector3, DetectorGeometry3,
    DetectorResponse, DetectorTermination, ObjectCommon, PlanarShape2, SolidGeometry3,
    SpectralScalar, SurfaceGeometry3, SurfaceObject3, SurfaceProps3, SurfaceResponse,
};
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Plane3, Ray3, Vec3};
use oc_trace::{DetectorResult3, LaunchRay3, Segment3, TraceConfig3, TraceResult3};

// ---------------------------------------------------------------------------
// Launch and config
// ---------------------------------------------------------------------------

pub fn launch(ray: Ray3, lambda_um: f64, flux: f64) -> LaunchRay3 {
    LaunchRay3 {
        ray,
        lambda_um,
        flux,
        source: None,
    }
}

/// Ghosts off: the primary refractive chain only. Used where the test reasons
/// about a single transmitted path.
pub fn no_ghosts() -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        ..TraceConfig3::default()
    }
}

/// A rotation carrying local `+Z` onto `normal`, so a surface authored in its
/// local `XY` plane faces the way the test wants.
fn facing(pos: Vec3, normal: Vec3) -> Isometry3 {
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, normal.normalize());
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap()
}

fn next_id(scene: &mut Scene3) -> ObjectId {
    scene.alloc_object_id()
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

fn push_body(
    scene: &mut Scene3,
    geometry: SolidGeometry3,
    material: Material,
    transform: Isometry3,
) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = transform;
    scene.bodies.push(Body3 {
        common,
        geometry,
        material,
        absorption: Default::default(),
        surface: SurfaceProps3::default(),
        priority: 0,
    });
    id
}

/// A glass sphere. Returns the refractive index the tracer will use at the d
/// line, so a test can check Snell against the exact value.
pub fn add_glass_sphere(scene: &mut Scene3, center: Vec3, radius_mm: f64, nd: f64) -> f64 {
    let material = Material::AbbePoint { nd, vd: 64.0 };
    let n = material.index_at(LAMBDA_D, &scene.catalog).unwrap();
    push_body(
        scene,
        SolidGeometry3::Sphere { radius_mm },
        material,
        Isometry3::from_pos(center),
    );
    n
}

/// A glass body with a bulk absorption coefficient (`mm⁻¹`), as a box.
pub fn add_absorbing_box(scene: &mut Scene3, half: Vec3, nd: f64, alpha_per_mm: f64) -> ObjectId {
    let id = push_body(
        scene,
        SolidGeometry3::Box {
            half_extents_mm: half,
        },
        Material::AbbePoint { nd, vd: 64.0 },
        Isometry3::IDENTITY,
    );
    let b = scene.bodies.last_mut().unwrap();
    b.absorption = oc_core::scene3::AbsorptionModel::Constant { alpha_per_mm };
    id
}

/// A glass slab (box) spanning `x ∈ [x0, x1]`, wide in y and z.
pub fn add_glass_slab(scene: &mut Scene3, x0: f64, x1: f64, nd: f64) -> f64 {
    let material = Material::AbbePoint { nd, vd: 64.0 };
    let n = material.index_at(LAMBDA_D, &scene.catalog).unwrap();
    let half = Vec3::new((x1 - x0) / 2.0, 20.0, 20.0);
    push_body(
        scene,
        SolidGeometry3::Box {
            half_extents_mm: half,
        },
        material,
        Isometry3::from_pos(Vec3::new((x0 + x1) / 2.0, 0.0, 0.0)),
    );
    n
}

/// A pure absorber body (a black box).
pub fn add_absorber_box(scene: &mut Scene3, center: Vec3, half: Vec3) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = Isometry3::from_pos(center);
    let mut surface = SurfaceProps3::default();
    surface.default.from_outside = SurfaceResponse::Absorber;
    surface.default.from_inside = SurfaceResponse::Absorber;
    scene.bodies.push(Body3 {
        common,
        geometry: SolidGeometry3::Box {
            half_extents_mm: half,
        },
        material: Material::Absorber,
        absorption: Default::default(),
        surface,
        priority: 0,
    });
    id
}

/// A right-angle glass prism, extruded along Z, with a flat front face at
/// `x = -2` (normal ±X). Returns its index at the d line.
pub fn add_glass_prism(scene: &mut Scene3, nd: f64) -> f64 {
    let material = Material::AbbePoint { nd, vd: 64.0 };
    let n = material.index_at(LAMBDA_D, &scene.catalog).unwrap();
    let mut edges = Vec::new();
    let verts = [
        Vec2::new(-2.0, -2.0),
        Vec2::new(-2.0, 2.0),
        Vec2::new(2.0, -2.0),
    ];
    for i in 0..3 {
        edges.push(ProfileEdge2 {
            patch: scene.alloc_patch_id(),
            curve: Curve2::Line {
                a: verts[i],
                b: verts[(i + 1) % 3],
            },
        });
    }
    let geometry = SolidGeometry3::Profile(ProfileSolid3 {
        profile: ClosedProfile2 { edges },
        construction: ProfileConstruction3::ExtrudeLocalZ {
            z_min_mm: -5.0,
            z_max_mm: 5.0,
        },
    });
    push_body(scene, geometry, material, Isometry3::IDENTITY);
    n
}

// ---------------------------------------------------------------------------
// Zero-thickness surfaces
// ---------------------------------------------------------------------------

fn push_surface(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    half: f64,
    response: SurfaceResponse,
) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.surfaces.push(SurfaceObject3 {
        common,
        geometry: SurfaceGeometry3::Plane {
            shape: PlanarShape2::Rectangle {
                half_width_mm: half,
                half_height_mm: half,
            },
        },
        response_front: response.clone(),
        response_back: response,
    });
    id
}

pub fn add_mirror(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    half: f64,
    reflectivity: f64,
) -> ObjectId {
    push_surface(
        scene,
        pos,
        normal,
        half,
        SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(reflectivity),
        },
    )
}

pub fn add_beamsplitter(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    half: f64,
    r: f64,
    t: f64,
) -> ObjectId {
    push_surface(
        scene,
        pos,
        normal,
        half,
        SurfaceResponse::ExplicitRt {
            reflectance: SpectralScalar::Constant(r),
            transmittance: SpectralScalar::Constant(t),
        },
    )
}

// ---------------------------------------------------------------------------
// Apertures and detectors
// ---------------------------------------------------------------------------

pub fn add_circular_aperture(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    opening: f64,
    mask: f64,
) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Disc { radius_mm: opening },
            outer_extent: Some(PlanarShape2::Disc { radius_mm: mask }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });
    id
}

pub fn add_annular_aperture(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    inner: f64,
    outer: f64,
    mask: f64,
) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Annulus {
                inner_mm: inner,
                outer_mm: outer,
            },
            outer_extent: Some(PlanarShape2::Disc { radius_mm: mask }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });
    id
}

pub fn add_rect_aperture(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    hw: f64,
    hh: f64,
    mask: f64,
) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Rectangle {
                half_width_mm: hw,
                half_height_mm: hh,
            },
            outer_extent: Some(PlanarShape2::Disc { radius_mm: mask }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });
    id
}

pub fn add_rect_detector(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    hw: f64,
    hh: f64,
    bins: [usize; 2],
) -> ObjectId {
    add_detector(
        scene,
        pos,
        normal,
        DetectorGeometry3::Rectangle {
            half_width_mm: hw,
            half_height_mm: hh,
        },
        bins,
    )
}

pub fn add_disc_detector(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    radius_mm: f64,
    bins: [usize; 2],
) -> ObjectId {
    add_detector(
        scene,
        pos,
        normal,
        DetectorGeometry3::Disc { radius_mm },
        bins,
    )
}

pub fn add_annulus_detector(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    inner_mm: f64,
    outer_mm: f64,
    bins: [usize; 2],
) -> ObjectId {
    add_detector(
        scene,
        pos,
        normal,
        DetectorGeometry3::Annulus { inner_mm, outer_mm },
        bins,
    )
}

fn add_detector(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    geometry: DetectorGeometry3,
    bins: [usize; 2],
) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.detectors.push(Detector3 {
        common,
        geometry,
        bins,
        response: DetectorResponse::Unity,
        termination: DetectorTermination::Terminate,
    });
    id
}

/// An aperture with a polygon clear opening inside a disc mask.
pub fn add_polygon_aperture(
    scene: &mut Scene3,
    pos: Vec3,
    normal: Vec3,
    opening: Vec<Vec2>,
    mask: f64,
) -> ObjectId {
    let id = next_id(scene);
    let mut common = ObjectCommon::new(id, LayerId(0));
    common.transform = facing(pos, normal);
    scene.apertures.push(Aperture3 {
        common,
        shape: ApertureShape3 {
            plane_local: Plane3::XY,
            clear_opening: PlanarShape2::Polygon { vertices: opening },
            outer_extent: Some(PlanarShape2::Disc { radius_mm: mask }),
        },
        interaction: ApertureInteraction::Absorb,
        primary_stop_for: Vec::new(),
    });
    id
}

// ---------------------------------------------------------------------------
// Reading a result
// ---------------------------------------------------------------------------

/// The detector result for a given object id.
pub fn detector_named(res: &TraceResult3, id: ObjectId) -> &DetectorResult3 {
    res.detectors
        .iter()
        .find(|d| d.detector.0 == id.0)
        .expect("detector not in result")
}

/// The primary (non-ghost) segment chain, ordered along the path.
pub fn primary_segments(res: &TraceResult3) -> Vec<Segment3> {
    let mut segs: Vec<Segment3> = res
        .segments
        .iter()
        .filter(|s| !s.flags.ghost)
        .copied()
        .collect();
    segs.sort_by(|a, b| a.opl0_mm.partial_cmp(&b.opl0_mm).unwrap());
    segs
}

/// The unit direction of each primary segment, in path order.
pub fn primary_dirs(res: &TraceResult3) -> Vec<Vec3> {
    primary_segments(res)
        .iter()
        .map(|s| (s.b - s.a).normalize())
        .collect()
}

/// The vertices of the primary path: launch origin, then each segment end.
pub fn primary_points(res: &TraceResult3) -> Vec<Vec3> {
    let segs = primary_segments(res);
    let mut pts = Vec::new();
    if let Some(first) = segs.first() {
        pts.push(first.a);
    }
    for s in &segs {
        pts.push(s.b);
    }
    pts
}
