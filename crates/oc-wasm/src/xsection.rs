//! Cross-section of the 3D scene with the view plane, for the 2D layout (PR
//! 3D-09, §10.4: "the exact intersection curve with the view plane").
//!
//! The 2D Canvas is a slice of the canonical 3D scene, so the optical elements
//! must be drawn as they cut that plane — a lens shows its meridional profile, a
//! sphere a circle, a mirror or detector a line. Everything is returned in the
//! view plane's `(u, v)` coordinates, ready for the renderer's `toScreen`.
//!
//! Output is a JSON array of polylines, each `{kind, closed, pts:[u,v,...]}`, so
//! the renderer can style a body, a mirror, a splitter, a dichroic, a detector,
//! and an aperture differently. A dichroic also carries `tint`, the CSS colour
//! of what its front face reflects. A profile body is stroked edge by edge (robust against edge
//! ordering); analytic solids and planar events are emitted as closed or open
//! curves.

use oc_core::profile2::{ClosedProfile2, ConicParams2, Curve2};
use oc_core::scene3::{
    Aperture3, Body3, Detector3, DetectorGeometry3, PlanarShape2, Scene3, SolidGeometry3,
    SurfaceGeometry3, SurfaceObject3, SurfaceResponse,
};
use oc_core::vec::Vec2;
use oc_core::vec3::{Isometry3, Plane3, Vec3};

/// Samples per curved span.
const N: usize = 40;

/// The whole scene's cross-section with `plane`, as a JSON array of polylines.
pub fn cross_section_json(scene: &Scene3, plane: &Plane3) -> String {
    let mut items: Vec<String> = Vec::new();
    for b in &scene.bodies {
        body(b, plane, &mut items);
    }
    for s in &scene.surfaces {
        surface(s, plane, &mut items);
    }
    for d in &scene.detectors {
        detector(d, plane, &mut items);
    }
    for a in &scene.apertures {
        aperture(a, plane, &mut items);
    }
    format!("[{}]", items.join(","))
}

/// One polyline as JSON. `pts` are view-plane `(u, v)` pairs.
fn poly(kind: &str, closed: bool, pts: &[Vec2]) -> String {
    let mut s = String::new();
    for (i, p) in pts.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{},{}", p.x, p.y));
    }
    format!("{{\"kind\":\"{kind}\",\"closed\":{closed},\"pts\":[{s}]}}")
}

/// Project a body-local 2D profile point `(x, y)` (z = 0) to view `(u, v)`.
fn proj_local(iso: &Isometry3, plane: &Plane3, x: f64, y: f64) -> Vec2 {
    plane.to_local(iso.transform_point(Vec3::new(x, y, 0.0)))
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

fn body(b: &Body3, plane: &Plane3, out: &mut Vec<String>) {
    let iso = &b.common.transform;
    match &b.geometry {
        SolidGeometry3::Profile(p) => profile(&p.profile, iso, plane, &b.surface, out),
        SolidGeometry3::Sphere { radius_mm } => sphere(iso, plane, *radius_mm, out),
        SolidGeometry3::Box { half_extents_mm } => box_xsec(iso, plane, *half_extents_mm, out),
        SolidGeometry3::Cylinder {
            radius_mm,
            half_length_mm,
            axis_local,
        } => cylinder(iso, plane, *radius_mm, *half_length_mm, *axis_local, out),
        SolidGeometry3::Tube {
            inner_radius_mm,
            outer_radius_mm,
            half_length_mm,
        } => tube(
            iso,
            plane,
            [*inner_radius_mm, *outer_radius_mm, *half_length_mm],
            &b.surface,
            out,
        ),
    }
}

/// A profile-authored body: each edge stroked in the plane it was drawn in
/// (local `z = 0`), which is the meridian for a revolve and the section for an
/// extrude — the classic 2D lens/prism outline.
fn profile(
    profile: &ClosedProfile2,
    iso: &Isometry3,
    plane: &Plane3,
    props: &oc_core::scene3::SurfaceProps3,
    out: &mut Vec<String>,
) {
    for e in &profile.edges {
        let mut pts = Vec::new();
        sample_curve(&e.curve, &mut pts);
        let uv: Vec<Vec2> = pts
            .iter()
            .map(|p| proj_local(iso, plane, p.x, p.y))
            .collect();
        if uv.len() < 2 {
            continue;
        }
        // A coated face of a solid draws as the coating it carries, tinted by
        // what it reflects: on a plate that face is the whole component, and
        // drawing it as one more edge of a glass disc would hide it.
        let two = props.for_patch(e.patch);
        let coated = matches!(two.from_outside, SurfaceResponse::Coating { .. });
        if coated {
            let line = poly("dichroic", false, &uv);
            out.push(match coat_css(&two.from_outside) {
                Some(css) => format!("{},\"tint\":\"{css}\"}}", &line[..line.len() - 1]),
                None => line,
            });
        } else {
            out.push(poly("body", false, &uv));
        }
    }
}

/// Sample one profile curve into local `(x, y)` points.
fn sample_curve(c: &Curve2, out: &mut Vec<Vec2>) {
    match c {
        Curve2::Line { a, b } => {
            out.push(*a);
            out.push(*b);
        }
        Curve2::Polyline { points } => out.extend_from_slice(points),
        Curve2::Arc {
            center,
            radius_mm,
            a0,
            a1,
        } => {
            for i in 0..=N {
                let t = a0 + (a1 - a0) * i as f64 / N as f64;
                out.push(Vec2::new(
                    center.x + radius_mm * t.cos(),
                    center.y + radius_mm * t.sin(),
                ));
            }
        }
        Curve2::Conic(cp) => sample_sag(cp, &[0.0; 8], out),
        Curve2::EvenAsphere { base, coeffs } => sample_sag(base, coeffs, out),
        Curve2::SagSegment {
            base,
            coeffs,
            y0,
            y1,
        } => {
            // One facet: a short piece of a face, so a fraction of a face's
            // samples draws it as smoothly.
            const N_SEG: usize = 8;
            for i in 0..=N_SEG {
                let t = y0 + (y1 - y0) * i as f64 / N_SEG as f64;
                out.push(oc_core::geom3::profile::sag_segment_point(base, coeffs, t));
            }
        }
    }
}

/// Sample a conic/asphere surface curve across its aperture: `x = vertex.x +
/// axis_sign · sag(|y|)`, `y ∈ [-half_ap, half_ap]`.
fn sample_sag(cp: &ConicParams2, coeffs: &[f64; 8], out: &mut Vec<Vec2>) {
    let axis_sign = if cp.axis.normalize().x >= 0.0 {
        1.0
    } else {
        -1.0
    };
    for i in 0..=N {
        let y = -cp.half_ap + 2.0 * cp.half_ap * i as f64 / N as f64;
        let x = cp.vertex.x + axis_sign * sag(cp.c, cp.k, coeffs, y.abs());
        out.push(Vec2::new(x, y));
    }
}

fn sag(c: f64, k: f64, coeffs: &[f64; 8], r: f64) -> f64 {
    let r2 = r * r;
    let denom = 1.0 + (1.0 - (1.0 + k) * c * c * r2).max(0.0).sqrt();
    let mut s = c * r2 / denom;
    let mut rp = r2 * r2;
    for &a in coeffs {
        s += a * rp;
        rp *= r2;
    }
    s
}

/// A sphere cut by the plane: a circle where `|centre distance| < r`.
fn sphere(iso: &Isometry3, plane: &Plane3, r: f64, out: &mut Vec<String>) {
    let c = iso.pos;
    let d = plane.signed_distance(c);
    if d.abs() >= r {
        return;
    }
    let rho = (r * r - d * d).sqrt();
    let center = plane.to_local(c);
    out.push(poly("body", true, &circle(center, rho)));
}

fn circle(center: Vec2, r: f64) -> Vec<Vec2> {
    (0..N)
        .map(|i| {
            let t = std::f64::consts::TAU * i as f64 / N as f64;
            Vec2::new(center.x + r * t.cos(), center.y + r * t.sin())
        })
        .collect()
}

/// A box cut by the plane: the convex polygon where the plane crosses its 12
/// edges.
fn box_xsec(iso: &Isometry3, plane: &Plane3, h: Vec3, out: &mut Vec<String>) {
    // Box corners in world.
    let mut corners = [Vec3::ZERO; 8];
    for (i, corner) in corners.iter_mut().enumerate() {
        let sx = if i & 1 == 0 { -h.x } else { h.x };
        let sy = if i & 2 == 0 { -h.y } else { h.y };
        let sz = if i & 4 == 0 { -h.z } else { h.z };
        *corner = iso.transform_point(Vec3::new(sx, sy, sz));
    }
    // The 12 edges as corner-index pairs.
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7), // x
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7), // y
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // z
    ];
    let mut pts: Vec<Vec2> = Vec::new();
    for (a, b) in EDGES {
        let (da, db) = (
            plane.signed_distance(corners[a]),
            plane.signed_distance(corners[b]),
        );
        if (da > 0.0) != (db > 0.0) && (da - db).abs() > 1e-12 {
            let t = da / (da - db);
            let p = corners[a] + (corners[b] - corners[a]) * t;
            pts.push(plane.to_local(p));
        }
    }
    if pts.len() >= 3 {
        sort_ccw(&mut pts);
        out.push(poly("body", true, &pts));
    }
}

/// Order polygon points counter-clockwise about their centroid.
fn sort_ccw(pts: &mut [Vec2]) {
    let c = pts.iter().fold(Vec2::ZERO, |a, p| a + *p) / pts.len() as f64;
    pts.sort_by(|a, b| {
        (a.y - c.y)
            .atan2(a.x - c.x)
            .partial_cmp(&(b.y - c.y).atan2(b.x - c.x))
            .unwrap()
    });
}

/// A finite cylinder cut by the plane. The two clean cases — the plane square to
/// the axis (a circle) or containing it (a rectangle). An oblique cut is skipped
/// rather than approximated.
fn cylinder(
    iso: &Isometry3,
    plane: &Plane3,
    r: f64,
    hl: f64,
    axis_local: Vec3,
    out: &mut Vec<String>,
) {
    let axis = (iso.rotate(axis_local.normalize())).normalize();
    let dot = axis.dot(plane.normal).abs();
    if dot > 0.98 {
        // Square to the axis: a circle at the centre (if the plane crosses it).
        if plane.signed_distance(iso.pos).abs() < hl {
            out.push(poly("body", true, &circle(plane.to_local(iso.pos), r)));
        }
    } else if dot < 0.02 {
        // Contains the axis: a rectangle 2r × 2·hl.
        let c = iso.pos;
        let rad = plane.normal.cross(axis).normalize(); // radial, in the plane
        let rect = [
            plane.to_local(c + axis * hl + rad * r),
            plane.to_local(c - axis * hl + rad * r),
            plane.to_local(c - axis * hl - rad * r),
            plane.to_local(c + axis * hl - rad * r),
        ];
        out.push(poly("body", true, &rect));
    }
}

/// A tube's section: its two walls when the plane holds the axis, two rings
/// when the plane is square to it. A bore that reflects draws as a mirror, so
/// a light pipe reads as one on the canvas.
fn tube(
    iso: &Isometry3,
    plane: &Plane3,
    [ri, ro, hl]: [f64; 3],
    props: &oc_core::scene3::SurfaceProps3,
    out: &mut Vec<String>,
) {
    let axis = iso.rotate(Vec3::X).normalize();
    let dot = axis.dot(plane.normal).abs();
    let c = iso.pos;
    if dot > 0.98 {
        if plane.signed_distance(c).abs() < hl {
            let centre = plane.to_local(c);
            out.push(poly("body", true, &circle(centre, ro)));
            out.push(poly("body", true, &circle(centre, ri)));
        }
    } else if dot < 0.02 {
        let rad = plane.normal.cross(axis).normalize();
        for side in [1.0, -1.0] {
            let at = |x: f64, r: f64| plane.to_local(c + axis * x + rad * (side * r));
            out.push(poly(
                "body",
                true,
                &[at(hl, ri), at(hl, ro), at(-hl, ro), at(-hl, ri)],
            ));
        }
        let bore = &props
            .for_patch(oc_core::id::PatchId(u64::from(
                oc_core::geom3::primitives::tube_face::INNER,
            )))
            .from_outside;
        if matches!(
            bore,
            SurfaceResponse::Mirror { .. } | SurfaceResponse::Coating { .. }
        ) {
            for side in [1.0, -1.0] {
                let at = |x: f64| plane.to_local(c + axis * x + rad * (side * ri));
                out.push(poly("mirror", false, &[at(hl), at(-hl)]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Planar events: mirrors/surfaces, detectors, apertures
// ---------------------------------------------------------------------------

/// The line where an object's own plane (local `z = 0` at `iso`) meets the view
/// plane, clipped to the in-plane region `keep`. `bound` is a radius covering the
/// shape. Returns the contiguous kept spans as `(u,v)` endpoint pairs.
fn plane_line(
    iso: &Isometry3,
    plane: &Plane3,
    bound: f64,
    keep: &dyn Fn(Vec2) -> bool,
) -> Vec<[Vec2; 2]> {
    let o = iso.pos;
    let n = iso.rotate(Vec3::Z).normalize();
    let u = iso.rotate(Vec3::X).normalize();
    let v = iso.rotate(Vec3::Y).normalize();
    let dir = n.cross(plane.normal);
    if dir.length() < 1e-9 {
        return Vec::new(); // parallel planes (coplanar handled elsewhere)
    }
    let dir = dir.normalize();
    // The point on the intersection line closest to `o` (which is on the object
    // plane): move within the object plane onto the view plane.
    let d0 = plane.signed_distance(o);
    let (pu, pv) = (u.dot(plane.normal), v.dot(plane.normal));
    let denom = pu * pu + pv * pv;
    let p0 = if denom > 1e-18 {
        o - u * (d0 * pu / denom) - v * (d0 * pv / denom)
    } else {
        o
    };

    // March the line over the shape's reach and keep the in-shape spans.
    let steps = 256;
    let mut spans: Vec<[Vec2; 2]> = Vec::new();
    let mut run_start: Option<Vec3> = None;
    let mut last_in = Vec3::ZERO;
    for i in 0..=steps {
        let t = -bound + 2.0 * bound * i as f64 / steps as f64;
        let p = p0 + dir * t;
        let luv = Vec2::new((p - o).dot(u), (p - o).dot(v));
        let inside = keep(luv);
        if inside {
            if run_start.is_none() {
                run_start = Some(p);
            }
            last_in = p;
        } else if let Some(s) = run_start.take() {
            spans.push([plane.to_local(s), plane.to_local(last_in)]);
        }
    }
    if let Some(s) = run_start {
        spans.push([plane.to_local(s), plane.to_local(last_in)]);
    }
    spans
}

fn shape_bound(shape: &PlanarShape2) -> f64 {
    match shape {
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => half_width_mm.hypot(*half_height_mm),
        PlanarShape2::Disc { radius_mm } => *radius_mm,
        PlanarShape2::Annulus { outer_mm, .. } => *outer_mm,
        PlanarShape2::Polygon { vertices } => {
            vertices.iter().fold(1.0_f64, |m, p| m.max(p.length()))
        }
    }
}

fn shape_contains(shape: &PlanarShape2, p: Vec2) -> bool {
    match shape {
        PlanarShape2::Rectangle {
            half_width_mm,
            half_height_mm,
        } => p.x.abs() <= *half_width_mm && p.y.abs() <= *half_height_mm,
        PlanarShape2::Disc { radius_mm } => p.length() <= *radius_mm,
        PlanarShape2::Annulus { inner_mm, outer_mm } => {
            let r = p.length();
            r >= *inner_mm && r <= *outer_mm
        }
        PlanarShape2::Polygon { vertices } => point_in_polygon(vertices, p),
    }
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

fn surface(s: &SurfaceObject3, plane: &Plane3, out: &mut Vec<String>) {
    let SurfaceGeometry3::Plane { shape } = &s.geometry else {
        return; // a surface of revolution has no simple planar section here
    };
    let kind = match crate::surface_object_kind(s) {
        "mirror" => "mirror",
        "beamsplitter" => "splitter",
        "dichroic" => "dichroic",
        "polarizer" | "retarder" => "polarizing",
        _ => "surface",
    };
    let tint = match kind {
        "dichroic" => coat_css(&s.response_front).or_else(|| coat_css(&s.response_back)),
        _ => None,
    };
    for seg in plane_line(&s.common.transform, plane, shape_bound(shape), &|uv| {
        shape_contains(shape, uv)
    }) {
        let line = poly(kind, false, &seg);
        out.push(match &tint {
            // Splice the tint into the object before its closing brace.
            Some(css) => format!("{},\"tint\":\"{css}\"}}", &line[..line.len() - 1]),
            None => line,
        });
    }
}

/// The reflected colour of a coated face as a CSS `rgb()`, brightest channel
/// at full scale and gamma-encoded, or `None` for an uncoated face or one that
/// reflects nothing visible.
fn coat_css(r: &SurfaceResponse) -> Option<String> {
    let (rc, _) = crate::render_view::coat_colors(r)?;
    let peak = rc[0].max(rc[1]).max(rc[2]);
    if peak.is_nan() || peak <= 1e-6 {
        return None;
    }
    let enc = |v: f32| {
        let l = (v / peak).clamp(0.0, 1.0);
        let s = if l <= 0.003_130_8 {
            12.92 * l
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    Some(format!("rgb({},{},{})", enc(rc[0]), enc(rc[1]), enc(rc[2])))
}

fn detector(d: &Detector3, plane: &Plane3, out: &mut Vec<String>) {
    let (bound, keep): (f64, Box<dyn Fn(Vec2) -> bool>) = match &d.geometry {
        DetectorGeometry3::Rectangle {
            half_width_mm,
            half_height_mm,
        } => {
            let (hw, hh) = (*half_width_mm, *half_height_mm);
            (
                hw.hypot(hh),
                Box::new(move |p: Vec2| p.x.abs() <= hw && p.y.abs() <= hh),
            )
        }
        DetectorGeometry3::Disc { radius_mm } => {
            let r = *radius_mm;
            (r, Box::new(move |p: Vec2| p.length() <= r))
        }
        DetectorGeometry3::Annulus { inner_mm, outer_mm } => {
            let (i, o) = (*inner_mm, *outer_mm);
            (
                o,
                Box::new(move |p: Vec2| {
                    let r = p.length();
                    r >= i && r <= o
                }),
            )
        }
        DetectorGeometry3::LineProbeLegacy { .. } => return,
    };
    for seg in plane_line(&d.common.transform, plane, bound, &keep) {
        out.push(poly("detector", false, &seg));
    }
}

/// An aperture's opaque mask, with the clear opening left as a gap.
fn aperture(a: &Aperture3, plane: &Plane3, out: &mut Vec<String>) {
    // The mask plane is placed by the object transform composed with the local
    // aperture plane origin; for the built-in XY aperture plane this is the object
    // transform. Bound by the outer extent if present, else a generous span.
    let bound = a
        .shape
        .outer_extent
        .as_ref()
        .map(shape_bound)
        .unwrap_or(50.0);
    let clear = &a.shape.clear_opening;
    let outer = a.shape.outer_extent.as_ref();
    for seg in plane_line(&a.common.transform, plane, bound, &|uv| {
        // Opaque where inside the mask but outside the clear opening.
        let in_mask = outer.map(|s| shape_contains(s, uv)).unwrap_or(true);
        in_mask && !shape_contains(clear, uv)
    }) {
        out.push(poly("aperture", false, &seg));
    }
}
