//! Render views of the canonical 3D scene, for the browser's GPU 3D view.
//!
//! Two things cross the boundary. A **descriptor** per object says what to draw
//! and at which revision (small JSON, re-read when the scene changes). A
//! **mesh** is fetched only when an object's geometry revision moves, as flat
//! typed arrays (`display_mesh`, local frame).
//!
//! Revisions are split three ways so the renderer can do the least work:
//! geometry (a new mesh), transform (a new matrix), material (a new shading
//! role). Each is decided by a fingerprint of exactly the fields that affect it,
//! so a revision moves only when its content changes, whatever setter changed
//! it. Revision numbers come from one counter for the whole tracker, so an object
//! that reuses a deleted object's id can never match the old object's cache.
//!
//! Nothing here is optical data. A display role is a shading choice, and the
//! display index is only ever used to shade glass.

use std::collections::HashMap;
use std::fmt::Debug;

use oc_core::display_mesh::{object_mesh, DisplayMesh, Tessellation, Topology};
use oc_core::profile2::ProfileConstruction3;
use oc_core::scene3::{Scene3, SolidGeometry3, SurfaceResponse};
use oc_core::Material;
use oc_gpu::color::wavelength_um_to_linear;
use wasm_bindgen::prelude::*;

/// FNV-1a over the `Debug` text of `parts`. `Debug` prints an `f64` in its
/// shortest round-trip form, so any change of value changes the text.
fn fingerprint(parts: &[&dyn Debug]) -> u64 {
    struct Fnv(u64);
    impl std::fmt::Write for Fnv {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            for b in s.bytes() {
                self.0 ^= b as u64;
                self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
            }
            Ok(())
        }
    }
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    for p in parts {
        let _ = std::fmt::write(&mut h, format_args!("{p:?}|"));
    }
    h.0
}

#[derive(Clone, Copy)]
struct Tracked {
    fp: u64,
    rev: u32,
}

#[derive(Clone, Copy)]
struct Entry {
    geometry: Tracked,
    transform: Tracked,
    material: Tracked,
}

/// Per-object revision bookkeeping, kept on the `Canvas` handle.
#[derive(Default)]
pub struct RenderTracker {
    entries: HashMap<u64, Entry>,
    counter: u32,
}

impl RenderTracker {
    fn bump(&mut self, old: Option<Tracked>, fp: u64) -> Tracked {
        match old {
            Some(t) if t.fp == fp => t,
            _ => {
                self.counter += 1;
                Tracked {
                    fp,
                    rev: self.counter,
                }
            }
        }
    }

    fn track(&mut self, id: u64, geometry: u64, transform: u64, material: u64) -> Entry {
        let old = self.entries.get(&id).copied();
        let e = Entry {
            geometry: self.bump(old.map(|e| e.geometry), geometry),
            transform: self.bump(old.map(|e| e.transform), transform),
            material: self.bump(old.map(|e| e.material), material),
        };
        self.entries.insert(id, e);
        e
    }
}

/// A display role for one side of a zero-thickness surface.
fn response_role(r: &SurfaceResponse) -> &'static str {
    match r {
        SurfaceResponse::Mirror { .. } => "mirror",
        SurfaceResponse::Absorber => "absorber",
        SurfaceResponse::ExplicitRt { .. }
        | SurfaceResponse::DielectricFresnel { .. }
        | SurfaceResponse::ThinLens { .. } => "generic_dielectric",
    }
}

fn opt_str(s: Option<&str>) -> String {
    s.map_or("null".to_string(), crate::json_str)
}

fn opt_num(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x}"),
        _ => "null".to_string(),
    }
}

/// The display colour of a source's light: its spectral lines' colours summed
/// by weight, the brightest channel scaled to 1. The same wavelength colours as
/// the rays, so a white source looks white and a laser line its own colour.
fn emission_color(s: &oc_core::scene3::Source3) -> [f32; 3] {
    let mut c = [0.0f32; 3];
    for (um, w) in s.spectrum.samples() {
        let rgb = wavelength_um_to_linear(um as f32);
        for k in 0..3 {
            c[k] += rgb[k] * w as f32;
        }
    }
    let peak = c[0].max(c[1]).max(c[2]);
    if peak > 0.0 && peak.is_finite() {
        c.map(|v| v / peak)
    } else {
        [1.0, 1.0, 1.0]
    }
}

#[allow(clippy::too_many_arguments)] // one descriptor's fields, written once
fn descriptor(
    id: u64,
    kind: &str,
    e: Entry,
    visible: bool,
    t: &oc_core::Isometry3,
    role: &str,
    front: Option<&str>,
    back: Option<&str>,
    nd: Option<f64>,
    radius: f64,
    error: Option<&str>,
    emit: Option<[f32; 3]>,
) -> String {
    let p = t.pos;
    let q = t.rot_xyzw;
    let emit = emit.map_or("null".to_string(), |[r, g, b]| format!("[{r},{g},{b}]"));
    format!(
        "{{\"id\":{id},\"kind\":\"{kind}\",\"geometryRevision\":{},\"transformRevision\":{},\"materialRevision\":{},\"visible\":{visible},\"pos\":[{},{},{}],\"rot\":[{},{},{},{}],\"role\":\"{role}\",\"front\":{},\"back\":{},\"nd\":{},\"radius\":{},\"error\":{},\"emit\":{emit}}}",
        e.geometry.rev,
        e.transform.rev,
        e.material.rev,
        p.x,
        p.y,
        p.z,
        q[0],
        q[1],
        q[2],
        q[3],
        opt_str(front),
        opt_str(back),
        opt_num(nd),
        if radius.is_finite() { radius } else { 1.0 },
        opt_str(error),
    )
}

/// The descriptors for every drawable object, as a JSON array. Updates the
/// tracker's revisions as a side effect.
pub fn render_objects_json(scene: &Scene3, tracker: &mut RenderTracker) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut live: Vec<u64> = Vec::new();

    for b in &scene.bodies {
        let id = b.common.id.0;
        live.push(id);
        let (kind, error) = match &b.geometry {
            SolidGeometry3::Profile(p) => match p.construction {
                ProfileConstruction3::RevolveLocalX => ("lens", None),
                ProfileConstruction3::ExtrudeLocalZ { .. } => ("prism", None),
                ProfileConstruction3::PlanarOnly { .. } => (
                    "planar_only",
                    Some("2D-only body: its third dimension was never authored"),
                ),
            },
            _ => ("solid", None),
        };
        let (role, nd) = match &b.material {
            Material::Mirror { .. } => ("mirror", None),
            Material::Absorber => ("absorber", None),
            Material::Grin(f) => ("generic_dielectric", Some(f.base_index)),
            m => match m.abbe(&scene.catalog) {
                Some((nd, _)) => ("optical_glass", Some(nd)),
                None => ("generic_dielectric", None),
            },
        };
        // A dielectric body whose default surface is a mirror or an absorber
        // reads as that surface.
        let role = match &b.surface.default.from_outside {
            SurfaceResponse::Mirror { .. } => "mirror",
            SurfaceResponse::Absorber => "absorber",
            _ => role,
        };
        let e = tracker.track(
            id,
            fingerprint(&[&"body", &b.geometry]),
            fingerprint(&[&b.common.transform]),
            fingerprint(&[&b.material, &b.surface]),
        );
        out.push(descriptor(
            id,
            kind,
            e,
            b.common.visible,
            &b.common.transform,
            role,
            None,
            None,
            nd,
            crate::body_radius(b),
            error,
            None,
        ));
    }

    for s in &scene.surfaces {
        let id = s.common.id.0;
        live.push(id);
        let (front, back) = (
            response_role(&s.response_front),
            response_role(&s.response_back),
        );
        let e = tracker.track(
            id,
            fingerprint(&[&"surface", &s.geometry]),
            fingerprint(&[&s.common.transform]),
            fingerprint(&[&s.response_front, &s.response_back]),
        );
        out.push(descriptor(
            id,
            "surface",
            e,
            s.common.visible,
            &s.common.transform,
            front,
            Some(front),
            Some(back),
            None,
            crate::surface_kind_radius(s).1,
            None,
            None,
        ));
    }

    for a in &scene.apertures {
        let id = a.common.id.0;
        live.push(id);
        let e = tracker.track(
            id,
            fingerprint(&[&"aperture", &a.shape]),
            fingerprint(&[&a.common.transform]),
            fingerprint(&[&a.interaction]),
        );
        out.push(descriptor(
            id,
            "aperture",
            e,
            a.common.visible,
            &a.common.transform,
            "aperture",
            None,
            None,
            None,
            crate::aperture_radius_opening(a).0,
            None,
            None,
        ));
    }

    for d in &scene.detectors {
        let id = d.common.id.0;
        live.push(id);
        let e = tracker.track(
            id,
            fingerprint(&[&"detector", &d.geometry]),
            fingerprint(&[&d.common.transform]),
            fingerprint(&[&d.response, &d.termination]),
        );
        let (hw, hh) = crate::detector_half_extents(d);
        out.push(descriptor(
            id,
            "detector",
            e,
            d.common.visible,
            &d.common.transform,
            "detector_active",
            None,
            None,
            None,
            hw.hypot(hh),
            None,
            None,
        ));
    }

    // Sources: a glyph at the emitting aperture (display_mesh::source_mesh).
    for s in &scene.sources {
        let id = s.id.0;
        live.push(id);
        let e = tracker.track(
            id,
            fingerprint(&[&"source", &s.spatial, &s.angular]),
            fingerprint(&[&s.transform]),
            fingerprint(&[&s.spectrum]),
        );
        out.push(descriptor(
            id,
            "source",
            e,
            true,
            &s.transform,
            "source_emitter",
            None,
            None,
            None,
            crate::source_radius(s),
            None,
            Some(emission_color(s)),
        ));
    }

    tracker.entries.retain(|id, _| live.contains(id));
    format!("[{}]", out.join(","))
}

/// One object's display mesh, handed to JavaScript as typed arrays.
#[wasm_bindgen]
pub struct RenderMesh {
    mesh: DisplayMesh,
}

#[wasm_bindgen]
impl RenderMesh {
    /// `"triangles"` or `"lines"`.
    pub fn topology(&self) -> String {
        match self.mesh.topology {
            Topology::Triangles => "triangles".into(),
            Topology::Lines => "lines".into(),
        }
    }

    /// `x y z` per vertex, mm, local frame.
    pub fn positions(&self) -> Vec<f32> {
        self.mesh.positions.clone()
    }

    pub fn normals(&self) -> Vec<f32> {
        self.mesh.normals.clone()
    }

    pub fn indices(&self) -> Vec<u32> {
        self.mesh.indices.clone()
    }

    /// Crease lines as vertex-index pairs (rims, corners, outlines).
    pub fn edges(&self) -> Vec<u32> {
        self.mesh.edges.clone()
    }

    /// Triples `[start, count, group]`, the group numbered as `FaceGroup`.
    pub fn groups(&self) -> Vec<u32> {
        self.mesh
            .groups
            .iter()
            .flat_map(|g| [g.start, g.count, g.group as u32])
            .collect()
    }

    /// `[minx, miny, minz, maxx, maxy, maxz]`, local frame.
    pub fn bounds(&self) -> Vec<f32> {
        let (a, b) = (self.mesh.bounds_min, self.mesh.bounds_max);
        vec![
            a[0] as f32,
            a[1] as f32,
            a[2] as f32,
            b[0] as f32,
            b[1] as f32,
            b[2] as f32,
        ]
    }

    /// Why this is not a solid, when it is not.
    pub fn note(&self) -> Option<String> {
        self.mesh.note.clone()
    }
}

/// The display mesh for `id`, or `None` if there is no such object.
pub fn render_mesh(scene: &Scene3, id: u64) -> Option<RenderMesh> {
    object_mesh(scene, id, &Tessellation::default()).map(|mesh| RenderMesh { mesh })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::vec3::Vec3;

    fn revs(json: &str, id: u64) -> (u64, u64, u64) {
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        let o = v
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["id"].as_u64() == Some(id))
            .unwrap();
        (
            o["geometryRevision"].as_u64().unwrap(),
            o["transformRevision"].as_u64().unwrap(),
            o["materialRevision"].as_u64().unwrap(),
        )
    }

    #[test]
    fn each_revision_moves_only_with_its_own_content() {
        let mut scene = Scene3::default_scene();
        let id = scene.bodies[0].common.id.0;
        let mut t = RenderTracker::default();
        let r0 = revs(&render_objects_json(&scene, &mut t), id);
        // No change: nothing moves.
        assert_eq!(r0, revs(&render_objects_json(&scene, &mut t), id));

        // A move is a transform change only.
        assert!(scene.set_object_position(id, Vec3::new(5.0, 1.0, 0.0)));
        let r1 = revs(&render_objects_json(&scene, &mut t), id);
        assert_eq!((r1.0, r1.2), (r0.0, r0.2));
        assert_ne!(r1.1, r0.1);

        // A tilt too.
        assert!(scene.set_object_rotation_z_deg(id, 12.0));
        let r2 = revs(&render_objects_json(&scene, &mut t), id);
        assert_eq!((r2.0, r2.2), (r1.0, r1.2));
        assert_ne!(r2.1, r1.1);

        // Glass is a material change only.
        assert!(scene.set_body_glass_abbe(id, 1.7, 40.0));
        let r3 = revs(&render_objects_json(&scene, &mut t), id);
        assert_eq!((r3.0, r3.1), (r2.0, r2.1));
        assert_ne!(r3.2, r2.2);

        // A radius is a geometry change only.
        let mut g = scene.singlet_geom(id).unwrap();
        g.r1_mm = 40.0;
        assert!(scene.set_singlet_geom(id, g));
        let r4 = revs(&render_objects_json(&scene, &mut t), id);
        assert_eq!((r4.1, r4.2), (r3.1, r3.2));
        assert_ne!(r4.0, r3.0);
    }

    #[test]
    fn a_reused_id_never_matches_the_old_cache() {
        let mut t = RenderTracker::default();
        let a = Scene3::default_scene();
        let id = a.bodies[0].common.id.0;
        let first = revs(&render_objects_json(&a, &mut t), id);
        // A different scene whose first body has the same id but other geometry.
        let mut b = Scene3::default_scene();
        let mut g = b.singlet_geom(id).unwrap();
        g.thickness_mm = 9.0;
        assert!(b.set_singlet_geom(id, g));
        let second = revs(&render_objects_json(&b, &mut t), id);
        assert_ne!(first.0, second.0);
    }

    #[test]
    fn every_object_kind_has_a_descriptor_and_a_mesh() {
        let mut scene = Scene3::default_scene();
        let prism = scene.add_prism(Vec3::new(-40.0, 0.0, 0.0), 16.0, 0.0, 1.62, 36.0);
        let ap = scene.add_aperture(Vec3::new(-20.0, 0.0, 0.0), 5.0, 0.0);
        let mirror = scene
            .add_planar_mirror(Vec3::new(20.0, -5.0, 0.0), Vec3::new(25.0, 5.0, 0.0), 0.9)
            .unwrap();
        let mut t = RenderTracker::default();
        let v: serde_json::Value =
            serde_json::from_str(&render_objects_json(&scene, &mut t)).unwrap();
        let kinds: Vec<(u64, String, String)> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|o| {
                (
                    o["id"].as_u64().unwrap(),
                    o["kind"].as_str().unwrap().to_string(),
                    o["role"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let body = scene.bodies[0].common.id.0;
        let det = scene.detectors[0].common.id.0;
        let src = scene.sources[0].id.0;
        for (id, kind, role) in [
            (body, "lens", "optical_glass"),
            (prism, "prism", "optical_glass"),
            (ap, "aperture", "aperture"),
            (mirror, "surface", "mirror"),
            (det, "detector", "detector_active"),
            (src, "source", "source_emitter"),
        ] {
            assert!(
                kinds.contains(&(id, kind.to_string(), role.to_string())),
                "{id} {kind} {role} missing from {kinds:?}"
            );
            let m = render_mesh(&scene, id).unwrap();
            assert_eq!(m.topology(), "triangles");
            assert!(!m.indices().is_empty());
            assert_eq!(m.positions().len(), m.normals().len());
            assert!(m.note().is_none());
        }
        let nd = v[0]["nd"].as_f64().unwrap();
        assert!((nd - 1.5168).abs() < 1e-12);
        assert!(render_mesh(&scene, 999_999).is_none());
        // Only a source carries a light colour; the default white beam's is white.
        for o in v.as_array().unwrap() {
            let emit = &o["emit"];
            if o["kind"] == "source" {
                let c: Vec<f64> = emit
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                assert!(
                    c.iter().all(|x| *x > 0.8 && *x <= 1.0),
                    "white source drawn as {c:?}"
                );
            } else {
                assert!(emit.is_null());
            }
        }
    }

    #[test]
    fn a_source_change_moves_only_its_own_revision() {
        let mut scene = Scene3::default_scene();
        let src = scene.sources[0].id.0;
        let mut t = RenderTracker::default();
        let r0 = revs(&render_objects_json(&scene, &mut t), src);
        // A new wavelength is a material change: the glyph takes the line's colour.
        assert!(scene.set_first_source_spectrum(&[(0.6328, 1.0)]));
        let json = render_objects_json(&scene, &mut t);
        let r1 = revs(&json, src);
        assert_eq!((r1.0, r1.1), (r0.0, r0.1));
        assert_ne!(r1.2, r0.2);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let o = v
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["id"].as_u64() == Some(src))
            .unwrap();
        let c: Vec<f64> = o["emit"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert!(
            c[0] == 1.0 && c[2] < 0.2,
            "a HeNe line should draw red, got {c:?}"
        );
        // A move is a transform change; a new aperture is a geometry change.
        assert!(scene.set_object_position(src, Vec3::new(-50.0, 2.0, 0.0)));
        let r2 = revs(&render_objects_json(&scene, &mut t), src);
        assert_eq!((r2.0, r2.2), (r1.0, r1.2));
        assert_ne!(r2.1, r1.1);
        scene.sources[0].spatial = oc_core::scene3::SpatialDistribution3::Disc { radius_mm: 3.0 };
        let r3 = revs(&render_objects_json(&scene, &mut t), src);
        assert_eq!((r3.1, r3.2), (r2.1, r2.2));
        assert_ne!(r3.0, r2.0);
    }
}
