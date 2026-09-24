//! A grooved, rotationally symmetric Fresnel lens: OpticStudio's non-sequential
//! "Fresnel 1" object, built as an ordinary revolved profile.
//!
//! A Fresnel lens is a smooth lens with the glass it does not need taken out.
//! The aperture is cut into annular zones, and in each zone the curved face is
//! shifted along the axis until the zone's thinnest point sits on a common plane,
//! the groove valleys. Every facet is therefore an **exact** piece of the base
//! surface, not a chord of it: a ray leaving a facet at height `h` sees the same
//! normal the smooth lens has at `h`, and leaves in the same direction. What
//! changes is the thickness it crossed, and the risers between the facets, which
//! are new surfaces the smooth lens never had.
//!
//! The body is a closed meridional loop (profile frame, `+X` the optical axis)
//! revolved about `+X` like any lens: one flat back face, two rim lines, and a
//! sawtooth of [`Curve2::SagSegment`] facets and straight risers. The central
//! facet spans the axis, so it is a plain [`Curve2::Conic`] or
//! [`Curve2::EvenAsphere`]. Nothing here is new geometry for the tracer: a facet
//! revolves to an annular surface of revolution, a riser to a cone or cylinder.
//!
//! # Conventions
//!
//! - `radius_mm` is the base surface's radius of curvature, **positive when the
//!   grooved face is convex**, bulging away from the substrate the way the curved
//!   face of a plano-convex lens does. A positive radius converges whichever way
//!   the grooves face. The aspheric terms carry the same sign: a positive `A4`
//!   adds to the bulge. In OpticStudio's sign convention this is the radius as
//!   seen from a ray arriving at the grooved face.
//! - The flat back face is at local `x = -f·substrate/2` and the groove valleys
//!   at `x = +f·substrate/2`, with `f = +1` when the grooves face `+X` and `-1`
//!   when they face `-X`. `substrate_mm` is the distance between them: from the
//!   flat back face to the plane the facet bases sit on. The teeth stand on that
//!   plane. With a draft angle a riser meets its lower neighbour slightly above
//!   the plane, because the neighbour facet is cut short where the riser leans
//!   into it.
//! - The draft angle tilts each riser away from the axis direction so the tooth
//!   is wider at its base (what a mould needs to release). `0` is a riser
//!   parallel to the optical axis.
//!
//! The recipe is not stored separately on the body: [`fresnel_from_profile`]
//! reads it back out of the profile, the way the singlet inspector reads a
//! singlet's radii back out of its four edges. One source of truth, and a
//! document written before this module existed loads unchanged.

use crate::geom3::primitives::even_asphere_sag;
use crate::geom3::profile::validate_revolvable;
use crate::id::{LayerId, PatchId};
use crate::material::Material;
use crate::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use crate::scene3::{AbsorptionModel, Body3, ObjectCommon, Scene3, SolidGeometry3, SurfaceProps3};
use crate::vec::Vec2;
use crate::vec3::{Isometry3, Vec3};
use serde::{Deserialize, Serialize};

/// How the aperture is cut into zones.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FresnelGrooves {
    /// Zones of equal radial width. The last zone takes what is left of the
    /// clear radius.
    ConstantPitch { pitch_mm: f64 },
    /// Zones of equal groove depth: a zone ends where the base surface has
    /// moved `depth_mm` along the axis since the zone began. Zones narrow
    /// toward the rim of a converging lens. The last zone is whatever depth is
    /// left at the clear radius.
    ConstantDepth { depth_mm: f64 },
}

/// Which way the grooved face points. The flat face is the other one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FresnelFacing {
    PlusX,
    MinusX,
}

impl FresnelFacing {
    fn sign(self) -> f64 {
        match self {
            FresnelFacing::PlusX => 1.0,
            FresnelFacing::MinusX => -1.0,
        }
    }
}

/// The recipe of a rotationally symmetric Fresnel lens (OpticStudio's
/// "Fresnel 1"). All lengths in mm. See the module docs for the conventions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FresnelLens1 {
    /// Base radius of curvature, positive for a convex grooved face. Infinite
    /// (or zero) is a flat base, which needs aspheric terms to be a lens.
    pub radius_mm: f64,
    /// Base conic constant.
    pub conic: f64,
    /// Even aspheric terms `A4, A6, …, A18` of the base sag.
    pub aspheric: [f64; 8],
    /// Radius of the grooved region, which is also the body's radius.
    pub clear_radius_mm: f64,
    pub grooves: FresnelGrooves,
    /// Flat back face to the groove valleys.
    pub substrate_mm: f64,
    pub facing: FresnelFacing,
    /// Riser tilt from the optical axis, degrees. `0` is a riser parallel to
    /// the axis.
    pub draft_deg: f64,
}

impl Default for FresnelLens1 {
    /// A 50 mm diameter, 1 mm pitch lens on an R 50 base with its grooves
    /// facing `+X`: in PMMA, about a 100 mm focal length.
    fn default() -> Self {
        Self {
            radius_mm: 50.0,
            conic: 0.0,
            aspheric: [0.0; 8],
            clear_radius_mm: 25.0,
            grooves: FresnelGrooves::ConstantPitch { pitch_mm: 1.0 },
            substrate_mm: 2.0,
            facing: FresnelFacing::PlusX,
            draft_deg: 0.0,
        }
    }
}

/// PMMA, the usual Fresnel lens material, as an Abbe point `(n_d, V_d)`.
pub const PMMA_ND_VD: (f64, f64) = (1.4917, 57.4);

/// The most zones a lens may have. A 100 mm lens at 0.01 mm pitch is 5000;
/// anything past this is a mistake in the units, not a design.
pub const MAX_ZONES: usize = 20_000;

impl FresnelLens1 {
    /// Base curvature `1/R`; an infinite or zero radius is flat.
    fn curvature(&self) -> f64 {
        if self.radius_mm.is_finite() && self.radius_mm.abs() > 1e-12 {
            1.0 / self.radius_mm
        } else {
            0.0
        }
    }

    /// The base surface's bulge at radius `r`: how far the smooth convex face
    /// has fallen back toward the substrate from its vertex.
    fn bulge(&self, r: f64) -> f64 {
        even_asphere_sag(self.curvature(), self.conic, &self.aspheric, r)
    }

    /// Relative thickness of the smooth lens at `r`, `0` on the axis. A facet's
    /// height above the valleys is this, shifted.
    fn thickness(&self, r: f64) -> f64 {
        -self.bulge(r)
    }

    /// Paraxial focal length of the thin lens this approximates, for glass of
    /// index `n`: `R / (n - 1)` for a spherical base (the aspheric terms add no
    /// paraxial power). `None` for a flat base.
    pub fn paraxial_efl_mm(&self, n: f64) -> Option<f64> {
        let c = self.curvature();
        (c != 0.0 && n > 1.0).then(|| 1.0 / (c * (n - 1.0)))
    }

    fn check(&self) -> Result<(), String> {
        let finite = self.conic.is_finite()
            && self.aspheric.iter().all(|a| a.is_finite())
            && self.clear_radius_mm.is_finite()
            && self.substrate_mm.is_finite()
            && self.draft_deg.is_finite()
            && !self.radius_mm.is_nan();
        if !finite {
            return Err("a Fresnel lens parameter is not a finite number".to_string());
        }
        if self.clear_radius_mm <= 0.0 {
            return Err("the clear radius must be positive".to_string());
        }
        if self.substrate_mm <= 0.0 {
            return Err("the substrate thickness must be positive".to_string());
        }
        if !(0.0..60.0).contains(&self.draft_deg) {
            return Err("the draft angle must be at least 0 and below 60 degrees".to_string());
        }
        match self.grooves {
            FresnelGrooves::ConstantPitch { pitch_mm } => {
                if !(pitch_mm.is_finite() && pitch_mm > 0.0) {
                    return Err("the groove pitch must be positive".to_string());
                }
                if self.clear_radius_mm / pitch_mm > MAX_ZONES as f64 {
                    return Err(format!(
                        "{:.0} grooves is more than the {MAX_ZONES} a lens may have",
                        self.clear_radius_mm / pitch_mm
                    ));
                }
            }
            FresnelGrooves::ConstantDepth { depth_mm } => {
                if !(depth_mm.is_finite() && depth_mm > 0.0) {
                    return Err("the groove depth must be positive".to_string());
                }
            }
        }
        let c = self.curvature();
        if (1.0 + self.conic) * c * c * self.clear_radius_mm * self.clear_radius_mm >= 1.0 {
            return Err("the base surface turns over before the clear radius".to_string());
        }
        if c == 0.0 && self.aspheric.iter().all(|a| *a == 0.0) {
            return Err("a flat base surface has no grooves to cut".to_string());
        }
        Ok(())
    }

    /// The zone boundaries `0 = r_0 < r_1 < … < r_n = clear radius`.
    fn zone_radii(&self) -> Result<Vec<f64>, String> {
        let rc = self.clear_radius_mm;
        let mut radii = vec![0.0];
        match self.grooves {
            FresnelGrooves::ConstantPitch { pitch_mm } => {
                let mut j = 1usize;
                // A sliver of a last zone narrower than a millionth of the pitch
                // is rounding, not a groove.
                while (j as f64) * pitch_mm < rc - 1e-6 * pitch_mm {
                    radii.push(j as f64 * pitch_mm);
                    j += 1;
                }
            }
            FresnelGrooves::ConstantDepth { depth_mm } => {
                // March out from each boundary until the base has moved one
                // depth, then bisect the crossing. The march step resolves a
                // zone far narrower than any a real lens has.
                let step = rc / 4096.0;
                let mut r0 = 0.0;
                loop {
                    let h0 = self.thickness(r0);
                    let moved = |r: f64| (self.thickness(r) - h0).abs() - depth_mm;
                    let mut lo = r0;
                    let mut found = None;
                    while lo < rc {
                        let hi = (lo + step).min(rc);
                        if moved(hi) >= 0.0 {
                            found = Some((lo, hi));
                            break;
                        }
                        lo = hi;
                    }
                    let Some((mut a, mut b)) = found else { break };
                    for _ in 0..200 {
                        let m = 0.5 * (a + b);
                        if moved(m) >= 0.0 {
                            b = m;
                        } else {
                            a = m;
                        }
                        if b - a <= 1e-15 * rc.max(1.0) {
                            break;
                        }
                    }
                    if b >= rc - 1e-9 * rc {
                        break;
                    }
                    radii.push(b);
                    r0 = b;
                    if radii.len() > MAX_ZONES {
                        return Err(format!(
                            "more than {MAX_ZONES} grooves at this depth; use a deeper groove"
                        ));
                    }
                }
            }
        }
        radii.push(rc);
        Ok(radii)
    }
}

/// One zone of the built lens: the base shifted by `-m` so its thinnest end
/// sits on the valleys, used over `[start, end]` (its boundaries, less what the
/// drafted risers cut off).
#[derive(Debug, Clone, Copy)]
struct Zone {
    /// The lower of the smooth thickness at the zone's two boundaries.
    m: f64,
    start: f64,
    end: f64,
}

/// The built lens: zones plus the frame numbers every edge is placed with.
struct Layout {
    zones: Vec<Zone>,
    f: f64,
    /// Valley plane and flat back face, local `x`.
    x_v: f64,
    x_b: f64,
    /// Canvas-frame curvature and aspheric terms of every facet: the base
    /// bulge, pointed along the grooved face's outward direction and negated,
    /// since the face falls back toward the substrate as `r` grows.
    c: f64,
    k: f64,
    coeffs: [f64; 8],
}

impl Layout {
    /// A facet's height above the valleys at `r`.
    fn height(&self, lens: &FresnelLens1, j: usize, r: f64) -> f64 {
        lens.thickness(r) - self.zones[j].m
    }

    /// The profile point of zone `j`'s facet at signed height `y`.
    fn point(&self, lens: &FresnelLens1, j: usize, y: f64) -> Vec2 {
        Vec2::new(self.x_v + self.f * self.height(lens, j, y.abs()), y)
    }

    /// Zone `j`'s facet curve's base: the canvas conic, vertex on the axis.
    fn base(&self, j: usize, half_ap: f64) -> ConicParams2 {
        ConicParams2 {
            vertex: Vec2::new(self.x_v - self.f * self.zones[j].m, 0.0),
            axis: Vec2::X,
            c: self.c,
            k: self.k,
            half_ap,
        }
    }
}

fn layout(lens: &FresnelLens1) -> Result<Layout, String> {
    lens.check()?;
    let radii = lens.zone_radii()?;
    let n = radii.len() - 1;
    if n < 2 {
        return Err(
            "the clear radius ends inside the first groove, so this is a plain lens".to_string(),
        );
    }
    let scale = lens.clear_radius_mm.max(1.0);
    let mut zones: Vec<Zone> = (0..n)
        .map(|j| {
            let (a, b) = (radii[j], radii[j + 1]);
            Zone {
                m: lens.thickness(a).min(lens.thickness(b)),
                start: a,
                end: b,
            }
        })
        .collect();

    // A facet must stand on the valleys, not dip through them: a base whose
    // sag turns back inside a zone would cut the valley plane mid-facet.
    for (j, z) in zones.iter().enumerate() {
        for i in 1..16 {
            let r = z.start + (z.end - z.start) * i as f64 / 16.0;
            if lens.thickness(r) - z.m < -1e-9 * scale {
                return Err(format!(
                    "the base surface turns back inside groove {} (r {:.4} to {:.4} mm); use a finer pitch",
                    j + 1,
                    z.start,
                    z.end
                ));
            }
        }
    }

    // Drafted risers: each leans into its lower neighbour, which is cut short
    // where the riser meets it.
    let tan = lens.draft_deg.to_radians().tan();
    if tan > 0.0 {
        for b in 1..n {
            let rb = radii[b];
            let inner = lens.thickness(rb) - zones[b - 1].m;
            let outer = lens.thickness(rb) - zones[b].m;
            let (top, low) = (inner.max(outer), inner.min(outer));
            if top - low <= 1e-12 * scale {
                continue;
            }
            // `+1`: the riser leans outward into the outer zone; `-1`: inward.
            let (lower, dir) = if inner < outer {
                (b - 1, -1.0)
            } else {
                (b, 1.0)
            };
            let width = zones[lower].end - zones[lower].start;
            let g = |delta: f64| lens.thickness(rb + dir * delta) - zones[lower].m;
            // Riser height at `delta` from the boundary, measured along it.
            let gap = |delta: f64| g(delta) - (top - delta / tan);
            if gap(width) < 0.0 {
                return Err(format!(
                    "a {:.2} degree draft is too steep for the groove at r {:.4} mm",
                    lens.draft_deg, rb
                ));
            }
            let (mut lo, mut hi) = (0.0, width);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if gap(mid) >= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
                if hi - lo <= 1e-15 * scale {
                    break;
                }
            }
            if dir < 0.0 {
                zones[lower].end = rb - hi;
            } else {
                zones[lower].start = rb + hi;
            }
        }
        if zones.iter().any(|z| z.end - z.start <= 1e-9 * scale) {
            return Err(format!(
                "a {:.2} degree draft leaves no facet in a groove; use a smaller draft or a wider pitch",
                lens.draft_deg
            ));
        }
    }

    let f = lens.facing.sign();
    let mut coeffs = lens.aspheric;
    for a in &mut coeffs {
        *a *= -f;
    }
    Ok(Layout {
        zones,
        f,
        x_v: 0.5 * f * lens.substrate_mm,
        x_b: -0.5 * f * lens.substrate_mm,
        c: -f * lens.curvature(),
        k: lens.conic,
        coeffs,
    })
}

/// The closed meridional loop of a Fresnel lens, ready to revolve about `+X`.
///
/// Edge order, which [`Scene3::set_fresnel_lens`] leans on to keep patch ids:
/// the flat back face first, the upper rim line second, the lower rim line
/// last, and in between the upper facets from the rim in, the central facet,
/// and the lower facets from the axis out, each riser between its two facets.
/// Patch ids are `0, 1, 2, …` in that order; a scene assigns its own.
pub fn fresnel_profile(lens: &FresnelLens1) -> Result<ClosedProfile2, String> {
    let lay = layout(lens)?;
    let n = lay.zones.len();
    let rc = lens.clear_radius_mm;
    let pure = lay.coeffs.iter().all(|a| *a == 0.0);
    let segment = |j: usize, y0: f64, y1: f64| Curve2::SagSegment {
        base: lay.base(j, y0.abs().max(y1.abs())),
        coeffs: lay.coeffs,
        y0,
        y1,
    };
    let scale = rc.max(1.0);
    let mut curves = Vec::with_capacity(4 * n);
    let back_top = Vec2::new(lay.x_b, rc);
    let back_bottom = Vec2::new(lay.x_b, -rc);
    curves.push(Curve2::Line {
        a: back_bottom,
        b: back_top,
    });
    curves.push(Curve2::Line {
        a: back_top,
        b: lay.point(lens, n - 1, rc),
    });
    // A riser between zone `j - 1` and zone `j`, from `a` to `b`. Where the two
    // facets meet at one point (no step, no draft) there is no riser.
    let riser = |curves: &mut Vec<Curve2>, a: Vec2, b: Vec2| {
        if (a - b).length() > 1e-12 * scale {
            curves.push(Curve2::Line { a, b });
        }
    };
    for j in (1..n).rev() {
        let z = lay.zones[j];
        curves.push(segment(j, z.end, z.start));
        let prev = lay.zones[j - 1];
        riser(
            &mut curves,
            lay.point(lens, j, z.start),
            lay.point(lens, j - 1, prev.end),
        );
    }
    let e0 = lay.zones[0].end;
    let center = lay.base(0, e0);
    curves.push(if pure {
        Curve2::Conic(center)
    } else {
        Curve2::EvenAsphere {
            base: center,
            coeffs: lay.coeffs,
        }
    });
    for j in 1..n {
        let z = lay.zones[j];
        let prev = lay.zones[j - 1];
        riser(
            &mut curves,
            lay.point(lens, j - 1, -prev.end),
            lay.point(lens, j, -z.start),
        );
        curves.push(segment(j, -z.start, -z.end));
    }
    curves.push(Curve2::Line {
        a: lay.point(lens, n - 1, -rc),
        b: back_bottom,
    });
    let profile = ClosedProfile2 {
        edges: curves
            .into_iter()
            .enumerate()
            .map(|(i, curve)| ProfileEdge2 {
                patch: PatchId(i as u64),
                curve,
            })
            .collect(),
    };
    validate_revolvable(&profile)
        .map_err(|why| format!("the Fresnel profile is invalid: {why}"))?;
    Ok(profile)
}

/// The Fresnel recipe a profile was built from, read back out of its edges, or
/// `None` if the profile is not a Fresnel lens this module built.
///
/// The read is checked by rebuilding: whatever recipe comes out must reproduce
/// the profile it was read from, edge for edge, or the answer is `None`. A lens
/// someone edited by hand is then not mistaken for a recipe it no longer is.
pub fn fresnel_from_profile(profile: &ClosedProfile2) -> Option<FresnelLens1> {
    let edges = &profile.edges;
    if edges.len() < 4 {
        return None;
    }
    let Curve2::Line { a: b0, b: b1 } = &edges[0].curve else {
        return None;
    };
    let Curve2::Line { b: rim_end, .. } = &edges[1].curve else {
        return None;
    };
    // The grooved region needs at least one facet segment; a plain lens is not
    // a Fresnel lens.
    if !edges
        .iter()
        .any(|e| matches!(e.curve, Curve2::SagSegment { .. }))
    {
        return None;
    }
    let (center, coeffs) = edges.iter().find_map(|e| match &e.curve {
        Curve2::Conic(cp) => Some((*cp, [0.0; 8])),
        Curve2::EvenAsphere { base, coeffs } => Some((*base, *coeffs)),
        _ => None,
    })?;
    let x_b = b0.x;
    if (b1.x - x_b).abs() > 1e-9 || b1.y <= 0.0 {
        return None;
    }
    let rc = b1.y;
    let f = if rim_end.x > x_b { 1.0 } else { -1.0 };
    let facing = if f > 0.0 {
        FresnelFacing::PlusX
    } else {
        FresnelFacing::MinusX
    };
    let substrate = 2.0 * x_b.abs();
    let radius_mm = if center.c != 0.0 {
        -f / center.c
    } else {
        f64::INFINITY
    };
    let mut aspheric = coeffs;
    for a in &mut aspheric {
        *a *= -f;
    }

    // Zone boundaries and the draft, from the upper half: a riser's top is on
    // its boundary, and a facet meeting the next without a riser shares it.
    let upper: Vec<&Curve2> = edges[2..edges.len() - 1]
        .iter()
        .map(|e| &e.curve)
        .take_while(|c| !matches!(c, Curve2::Conic(_) | Curve2::EvenAsphere { .. }))
        .collect();
    let mut bounds: Vec<f64> = Vec::new();
    let mut draft_deg: f64 = 0.0;
    let mut i = 0;
    while i < upper.len() {
        match upper[i] {
            Curve2::SagSegment { y1, .. } => {
                // The facet's inner end; the boundary is here unless a riser
                // follows and says otherwise.
                match upper.get(i + 1) {
                    Some(Curve2::Line { a, b }) => {
                        let away = |p: &Vec2| (p.x - x_b).abs();
                        let top = if away(a) >= away(b) { a } else { b };
                        bounds.push(top.y);
                        let (dx, dy) = ((a.x - b.x).abs(), (a.y - b.y).abs());
                        if dx > 0.0 {
                            draft_deg = draft_deg.max(dy.atan2(dx).to_degrees());
                        }
                        i += 2;
                    }
                    _ => {
                        bounds.push(*y1);
                        i += 1;
                    }
                }
            }
            _ => return None,
        }
    }
    bounds.reverse();
    let r1 = *bounds.first()?;
    let mut lens = FresnelLens1 {
        radius_mm,
        conic: center.k,
        aspheric,
        clear_radius_mm: rc,
        grooves: FresnelGrooves::ConstantPitch { pitch_mm: r1 },
        substrate_mm: substrate,
        facing,
        draft_deg: (draft_deg * 1e9).round() / 1e9,
    };
    let depth = (lens.thickness(r1) - lens.thickness(0.0)).abs();
    for grooves in [
        FresnelGrooves::ConstantPitch { pitch_mm: r1 },
        FresnelGrooves::ConstantDepth { depth_mm: depth },
    ] {
        lens.grooves = grooves;
        if let Ok(rebuilt) = fresnel_profile(&lens) {
            if same_curves(&rebuilt, profile) {
                return Some(lens);
            }
        }
    }
    None
}

/// Do two profiles have the same curves, to within a relative `1e-9`?
fn same_curves(a: &ClosedProfile2, b: &ClosedProfile2) -> bool {
    if a.edges.len() != b.edges.len() {
        return false;
    }
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * (1.0 + x.abs().max(y.abs()));
    let close2 = |p: Vec2, q: Vec2| close(p.x, q.x) && close(p.y, q.y);
    let conic = |p: &ConicParams2, q: &ConicParams2| {
        close2(p.vertex, q.vertex) && close(p.c, q.c) && close(p.k, q.k)
    };
    let terms = |p: &[f64; 8], q: &[f64; 8]| p.iter().zip(q).all(|(x, y)| close(*x, *y));
    a.edges
        .iter()
        .zip(&b.edges)
        .all(|(e, g)| match (&e.curve, &g.curve) {
            (Curve2::Line { a, b }, Curve2::Line { a: c, b: d }) => {
                close2(*a, *c) && close2(*b, *d)
            }
            (Curve2::Conic(p), Curve2::Conic(q)) => conic(p, q) && close(p.half_ap, q.half_ap),
            (
                Curve2::EvenAsphere { base: p, coeffs: s },
                Curve2::EvenAsphere { base: q, coeffs: t },
            ) => conic(p, q) && close(p.half_ap, q.half_ap) && terms(s, t),
            (
                Curve2::SagSegment {
                    base: p,
                    coeffs: s,
                    y0,
                    y1,
                },
                Curve2::SagSegment {
                    base: q,
                    coeffs: t,
                    y0: u0,
                    y1: u1,
                },
            ) => conic(p, q) && terms(s, t) && close(*y0, *u0) && close(*y1, *u1),
            _ => false,
        })
}

impl Scene3 {
    /// Add a Fresnel lens centred at `pos` (the middle of its substrate), its
    /// axis along world `+X` turned by `tilt_deg` about `Z`, in `material`.
    /// Returns the new body id, or why the recipe does not build.
    pub fn add_fresnel_lens(
        &mut self,
        pos: Vec3,
        tilt_deg: f64,
        lens: &FresnelLens1,
        material: Material,
    ) -> Result<u64, String> {
        let built = fresnel_profile(lens)?;
        let edges = built
            .edges
            .into_iter()
            .map(|e| ProfileEdge2 {
                patch: self.alloc_patch_id(),
                curve: e.curve,
            })
            .collect();
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = Isometry3::from_pos(pos);
        self.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Profile(ProfileSolid3 {
                profile: ClosedProfile2 { edges },
                construction: ProfileConstruction3::RevolveLocalX,
            }),
            material,
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        if tilt_deg != 0.0 {
            self.set_object_rotation_z_deg(id.0, tilt_deg);
        }
        Ok(id.0)
    }

    /// The Fresnel recipe of body `id`, if it is a Fresnel lens.
    pub fn fresnel_lens(&self, id: u64) -> Option<FresnelLens1> {
        let b = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        match &b.geometry {
            SolidGeometry3::Profile(p) if p.construction == ProfileConstruction3::RevolveLocalX => {
                fresnel_from_profile(&p.profile)
            }
            _ => None,
        }
    }

    /// Rebuild body `id` from a new recipe. The back face and the two rim lines
    /// keep their patch ids, and the grooved edges reuse the old ids in order as
    /// far as they go, so an override on the back face (a coating on the flat
    /// side) survives a change of pitch. Fails, changing nothing, if `id` is not
    /// a Fresnel lens or the recipe does not build.
    pub fn set_fresnel_lens(&mut self, id: u64, lens: &FresnelLens1) -> Result<(), String> {
        if self.fresnel_lens(id).is_none() {
            return Err(format!("object {id} is not a Fresnel lens"));
        }
        let built = fresnel_profile(lens)?;
        let old: Vec<PatchId> = match &self
            .bodies
            .iter()
            .find(|b| b.common.id.0 == id)
            .expect("found above")
            .geometry
        {
            SolidGeometry3::Profile(p) => p.profile.edges.iter().map(|e| e.patch).collect(),
            _ => unreachable!("checked above"),
        };
        let n = built.edges.len();
        let mut patches = Vec::with_capacity(n);
        // Back face, upper rim, then the grooved edges in order, then the lower
        // rim: the old ids by role first, fresh ones where the new lens has more.
        let old_inner = &old[2..old.len() - 1];
        for i in 0..n {
            let p = if i == 0 || i == 1 {
                old[i]
            } else if i == n - 1 {
                old[old.len() - 1]
            } else if let Some(p) = old_inner.get(i - 2) {
                *p
            } else {
                self.alloc_patch_id()
            };
            patches.push(p);
        }
        let b = self
            .bodies
            .iter_mut()
            .find(|b| b.common.id.0 == id)
            .expect("found above");
        if let SolidGeometry3::Profile(p) = &mut b.geometry {
            p.profile.edges = built
                .edges
                .into_iter()
                .zip(patches.iter())
                .map(|(e, patch)| ProfileEdge2 {
                    patch: *patch,
                    curve: e.curve,
                })
                .collect();
        }
        // An override naming a patch the new lens no longer has would dangle.
        b.surface.overrides.retain(|o| patches.contains(&o.patch));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upper_facets(p: &ClosedProfile2) -> usize {
        p.edges
            .iter()
            .filter(|e| matches!(e.curve, Curve2::SagSegment { y0, y1, .. } if y0 + y1 > 0.0))
            .count()
    }

    #[test]
    fn the_default_lens_builds_a_valid_revolvable_profile() {
        let lens = FresnelLens1::default();
        let p = fresnel_profile(&lens).expect("builds");
        // 25 zones at 1 mm pitch: 24 annular facets above the axis.
        assert_eq!(upper_facets(&p), 24);
        assert!(validate_revolvable(&p).is_ok());
    }

    #[test]
    fn every_facet_is_a_piece_of_the_base_shifted_along_the_axis() {
        let lens = FresnelLens1 {
            conic: -0.6,
            aspheric: [2e-6, -1e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..FresnelLens1::default()
        };
        let p = fresnel_profile(&lens).unwrap();
        for e in &p.edges {
            if let Curve2::SagSegment {
                base,
                coeffs,
                y0,
                y1,
            } = &e.curve
            {
                // Same curvature and terms as the base, pointed back toward the
                // substrate from a grooved face that faces +X.
                assert_eq!(base.c, -1.0 / 50.0);
                assert_eq!(base.k, lens.conic);
                assert_eq!(coeffs[0], -2e-6);
                // Its thinnest end sits on the valleys.
                let ends = [
                    crate::geom3::profile::sag_segment_point(base, coeffs, *y0).x,
                    crate::geom3::profile::sag_segment_point(base, coeffs, *y1).x,
                ];
                let low = ends[0].min(ends[1]);
                assert!((low - 1.0).abs() < 1e-12, "valley at {low}");
            }
        }
    }

    #[test]
    fn constant_depth_zones_end_where_the_base_has_moved_one_depth() {
        let lens = FresnelLens1 {
            grooves: FresnelGrooves::ConstantDepth { depth_mm: 0.25 },
            ..FresnelLens1::default()
        };
        let radii = lens.zone_radii().unwrap();
        for w in radii.windows(2).take(radii.len() - 2) {
            let moved = lens.thickness(w[0]) - lens.thickness(w[1]);
            assert!((moved - 0.25).abs() < 1e-12, "moved {moved}");
        }
        // Equal depth means narrowing zones toward the rim of a converging lens.
        assert!(radii[2] - radii[1] > radii[radii.len() - 2] - radii[radii.len() - 3]);
    }

    #[test]
    fn the_recipe_reads_back_out_of_the_profile() {
        for lens in [
            FresnelLens1::default(),
            FresnelLens1 {
                grooves: FresnelGrooves::ConstantDepth { depth_mm: 0.3 },
                facing: FresnelFacing::MinusX,
                draft_deg: 3.0,
                substrate_mm: 1.5,
                ..FresnelLens1::default()
            },
            FresnelLens1 {
                radius_mm: -80.0,
                conic: -1.0,
                aspheric: [1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                clear_radius_mm: 20.3,
                grooves: FresnelGrooves::ConstantPitch { pitch_mm: 0.7 },
                draft_deg: 2.0,
                ..FresnelLens1::default()
            },
        ] {
            let p = fresnel_profile(&lens).unwrap();
            let back = fresnel_from_profile(&p).expect("reads back");
            let q = fresnel_profile(&back).unwrap();
            assert!(same_curves(&p, &q), "{lens:?} read back as {back:?}");
            assert_eq!(back.facing, lens.facing);
            assert!((back.substrate_mm - lens.substrate_mm).abs() < 1e-12);
            assert!((back.draft_deg - lens.draft_deg).abs() < 1e-6);
        }
    }

    #[test]
    fn it_draws_as_a_solid_and_refuses_the_planar_shortcut() {
        let mut s = Scene3::new();
        let (nd, vd) = PMMA_ND_VD;
        s.add_fresnel_lens(
            Vec3::ZERO,
            0.0,
            &FresnelLens1::default(),
            Material::AbbePoint { nd, vd },
        )
        .unwrap();
        let mesh = crate::display_mesh::body_mesh(
            &s.bodies[0],
            &crate::display_mesh::Tessellation::default(),
        );
        assert!(mesh.note.is_none(), "{:?}", mesh.note);
        assert!(!mesh.indices.is_empty());
        // A v1 edge cannot say "this annulus of that conic", so the 2D fast
        // path is not exact for it and must not be offered.
        assert!(!crate::planar_reduce::reduces_exactly(&s));
    }

    #[test]
    fn a_plain_singlet_is_not_a_fresnel_lens() {
        let mut s = Scene3::new();
        let id = s.add_singlet(
            Vec3::ZERO,
            crate::build3::SingletGeom::biconvex(),
            1.5168,
            64.17,
        );
        assert!(s.fresnel_lens(id.0).is_none());
    }

    #[test]
    fn a_drafted_riser_leans_into_its_lower_neighbour() {
        let lens = FresnelLens1 {
            draft_deg: 10.0,
            ..FresnelLens1::default()
        };
        let lay = layout(&lens).unwrap();
        // Converging: each zone's outer end is its low end, so the riser at
        // that boundary eats into it from outside.
        for w in lay.zones.windows(2) {
            assert!(w[0].end < w[1].start);
        }
        // The riser meets the neighbour it leans into where that facet has
        // risen to meet it: `delta = H / (slope + cot(draft))`, less than the
        // groove is wide, so even a steep draft leaves every facet a width.
        let steep = FresnelLens1 {
            radius_mm: 30.0,
            draft_deg: 55.0,
            ..FresnelLens1::default()
        };
        let lay = layout(&steep).unwrap();
        assert!(lay.zones.iter().all(|z| z.end > z.start));
        assert!(fresnel_profile(&steep).is_ok());
    }

    #[test]
    fn a_rebuild_keeps_the_flat_face_patch_and_the_scene_valid() {
        let mut s = Scene3::new();
        let (nd, vd) = PMMA_ND_VD;
        let id = s
            .add_fresnel_lens(
                Vec3::ZERO,
                0.0,
                &FresnelLens1::default(),
                Material::AbbePoint { nd, vd },
            )
            .unwrap();
        s.validate().expect("valid");
        let back_patch = |s: &Scene3| match &s.bodies[0].geometry {
            SolidGeometry3::Profile(p) => p.profile.edges[0].patch,
            _ => unreachable!(),
        };
        let before = back_patch(&s);
        let finer = FresnelLens1 {
            grooves: FresnelGrooves::ConstantPitch { pitch_mm: 0.5 },
            ..FresnelLens1::default()
        };
        s.set_fresnel_lens(id, &finer).unwrap();
        assert_eq!(back_patch(&s), before);
        assert_eq!(s.fresnel_lens(id), Some(finer));
        s.validate().expect("still valid");
        assert!(s
            .set_fresnel_lens(
                id,
                &FresnelLens1 {
                    clear_radius_mm: -1.0,
                    ..finer
                }
            )
            .is_err());
        assert_eq!(
            s.fresnel_lens(id),
            Some(finer),
            "a refused edit changes nothing"
        );
    }
}
