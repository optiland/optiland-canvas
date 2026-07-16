//! Typed parametric generators (§9.5 `builtin-generator`).
//!
//! A module's optics normally come from a stored `.ocanvas` fragment. A generated
//! module instead names a **typed generator** and a set of numbers, and its
//! geometry is built when it is placed — so changing a number changes the glass
//! and the rays follow.
//!
//! # Typed, not scripted
//!
//! §9.5 forbids executing anything a descriptor carries: no scripts, no
//! expression evaluation. So a descriptor cannot describe *how* to build its
//! optics, only *which* builder to use and with what numbers. The set of builders
//! is closed, lives here, and is code someone reviewed. A descriptor naming a
//! generator this build does not have is an error, never an attempt.
//!
//! # The sign convention is the whole ballgame
//!
//! `radius_1_mm` and `radius_2_mm` follow the **standard optical convention**,
//! the one a lens prescription and Optiland both use: a radius is positive when
//! the centre of curvature lies on the `+X` side of the vertex. So a biconvex
//! singlet is `R1 > 0, R2 < 0`, and a plano-convex is `R1 > 0, R2 = inf`.
//!
//! This is worth stating loudly because `oc-wasm`'s freehand singlet inspector
//! uses a *different* convention: it builds its rear surface on a reversed axis,
//! so there a positive `radius2` means a convex rear. Both are internally
//! consistent; they are not interchangeable, and a descriptor is a prescription,
//! so it gets the prescription convention.
//!
//! # What the numbers mean
//!
//! Everything is millimetres in the module's own frame, with the lens centred on
//! the anchor and its axis along `+X` (§6.1). An infinite radius is a plane.

use oc_core::{spherical_edge, Edge, Element, ElementId, Material, Scene, SceneFragment, Vec2};
use std::collections::BTreeMap;

use crate::errors::Error;

/// The design wavelength for a reported index, micrometres.
///
/// The d-line. Paraxial focal length depends on the index, which depends on
/// wavelength, so a single reported number has to name its wavelength or it means
/// nothing. The trace itself is fully dispersive; this is the label on the ruler.
pub const DESIGN_LAMBDA_UM: f64 = 0.587_561_8;

/// A builder this build knows how to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generator {
    /// A two-surface lens: [`SingletParams`].
    Singlet,
}

impl Generator {
    /// Resolve a descriptor's generator name.
    pub fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "singlet" => Ok(Generator::Singlet),
            other => Err(Error::UnknownGenerator(other.to_string())),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Generator::Singlet => "singlet",
        }
    }

    /// Build a module fragment from resolved parameters.
    pub fn build(
        self,
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<SceneFragment, Error> {
        match self {
            Generator::Singlet => SingletParams::from_json(params)?.fragment(),
        }
    }

    /// The parameters this generator reads, in the order a UI should show them.
    ///
    /// A UI must only offer what is actually bound to geometry (§15.7): an
    /// editable field that changes nothing is worse than no field.
    pub fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Generator::Singlet => &[
                "material",
                "diameter_mm",
                "center_thickness_mm",
                "radius_1_mm",
                "radius_2_mm",
            ],
        }
    }
}

/// A two-surface lens, in the standard prescription convention.
#[derive(Debug, Clone, PartialEq)]
pub struct SingletParams {
    /// A glass name in the builtin catalogue, e.g. `N-BK7`.
    pub material: String,
    pub diameter_mm: f64,
    pub center_thickness_mm: f64,
    /// Front radius. Positive = centre of curvature on `+X`. Infinite = plane.
    pub radius_1_mm: f64,
    /// Rear radius, same convention: **negative** is a convex rear.
    pub radius_2_mm: f64,
}

/// What the lensmaker's equation says this lens should do.
///
/// The analytic prediction, computed independently of the tracer. Two ways of
/// answering the same question is the only reason either can be checked: if the
/// rays cross the axis where this says they should, both are probably right.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Paraxial {
    /// Effective focal length, millimetres. Negative for a diverging lens.
    pub efl_mm: f64,
    /// Back focal distance from the **rear vertex**, millimetres.
    pub bfd_mm: f64,
    /// Where the rear focus lands in the module's own frame, along `+X`. The rear
    /// vertex sits at `+t/2`, so this is what to compare a traced crossing to.
    pub rear_focus_x_mm: f64,
    /// The index used, and the wavelength it belongs to.
    pub nd: f64,
    pub lambda_um: f64,
}

impl SingletParams {
    /// Read the resolved parameter map.
    ///
    /// Missing keys are an error rather than a default: a lens quietly built with
    /// somebody's guess at a radius would trace perfectly and mean nothing.
    pub fn from_json(p: &BTreeMap<String, serde_json::Value>) -> Result<Self, Error> {
        let out = SingletParams {
            material: string_param(p, "material")?,
            diameter_mm: number_param(p, "diameter_mm")?,
            center_thickness_mm: number_param(p, "center_thickness_mm")?,
            radius_1_mm: number_param(p, "radius_1_mm")?,
            radius_2_mm: number_param(p, "radius_2_mm")?,
        };
        out.validate()?;
        Ok(out)
    }

    fn half_aperture(&self) -> f64 {
        self.diameter_mm / 2.0
    }

    /// Curvature, with an infinite radius meaning a plane.
    fn c1(&self) -> f64 {
        curvature(self.radius_1_mm)
    }
    fn c2(&self) -> f64 {
        curvature(self.radius_2_mm)
    }

    /// Refuse anything that is not a lens.
    ///
    /// This matters more than it looks. A radius shorter than the semi-diameter
    /// makes the sag's square root imaginary; the geometry clamps it, produces a
    /// shape that is not the surface anyone asked for, and traces happily. That is
    /// the exact failure this project exists to refuse: plausible and wrong.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |m: String| Err(Error::Parameter(m));
        for (k, v) in [
            ("diameter_mm", self.diameter_mm),
            ("center_thickness_mm", self.center_thickness_mm),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return bad(format!("{k} must be a positive, finite number (got {v})"));
            }
        }
        let ha = self.half_aperture();
        for (k, r) in [
            ("radius_1_mm", self.radius_1_mm),
            ("radius_2_mm", self.radius_2_mm),
        ] {
            if r.is_nan() {
                return bad(format!("{k} is not a number"));
            }
            if r == 0.0 {
                return bad(format!(
                    "{k} is 0: a zero radius is a point, not a surface — use inf for a plane"
                ));
            }
            if r.is_finite() && r.abs() < ha {
                return bad(format!(
                    "{k} = {r} is shorter than the {ha} mm semi-diameter: the surface would be \
                     deeper than a hemisphere, and the sag is imaginary past |R|"
                ));
            }
        }
        if self.rim_thickness() < 0.0 {
            return bad(format!(
                "the surfaces cross before the rim: {} mm centre thickness is too thin for \
                 R1 = {} and R2 = {} at a {} mm diameter",
                self.center_thickness_mm, self.radius_1_mm, self.radius_2_mm, self.diameter_mm
            ));
        }
        Ok(())
    }

    /// Thickness at the rim. Negative means the two surfaces have crossed.
    fn rim_thickness(&self) -> f64 {
        let ha = self.half_aperture();
        let (zf, zr) = (-self.center_thickness_mm / 2.0, self.center_thickness_mm / 2.0);
        (zr + sag(self.c2(), ha)) - (zf + sag(self.c1(), ha))
    }

    /// The lens, as a module fragment about its own anchor.
    pub fn fragment(&self) -> Result<SceneFragment, Error> {
        self.validate()?;
        let mut scene = Scene::new();
        let glass = scene
            .catalog
            .find(&self.material)
            .ok_or_else(|| Error::UnknownGlass(self.material.clone()))?;

        let ha = self.half_aperture();
        let (zf, zr) = (-self.center_thickness_mm / 2.0, self.center_thickness_mm / 2.0);
        // Both surfaces are measured along +X. That is what makes R2 negative for
        // a convex rear, which is the convention a prescription is written in.
        let front = spherical_edge(Vec2::new(zf, 0.0), Vec2::X, self.radius_1_mm, ha);
        let rear = spherical_edge(Vec2::new(zr, 0.0), Vec2::X, self.radius_2_mm, ha);
        let ftip = zf + sag(self.c1(), ha);
        let rtip = zr + sag(self.c2(), ha);

        scene.add_element(Element::new(
            ElementId(0),
            vec![
                front,
                Edge::Line {
                    a: Vec2::new(ftip, ha),
                    b: Vec2::new(rtip, ha),
                },
                rear,
                Edge::Line {
                    a: Vec2::new(rtip, -ha),
                    b: Vec2::new(ftip, -ha),
                },
            ],
            Material::Catalog(glass),
        ));
        Ok(SceneFragment::from_scene(&scene)?)
    }

    /// The lensmaker's equation for this lens, at [`DESIGN_LAMBDA_UM`].
    ///
    /// Thick-lens form, so the centre thickness is not quietly ignored:
    ///
    /// ```text
    /// 1/f = (n-1) [ c1 - c2 + (n-1) t c1 c2 / n ]
    /// BFD = f [ 1 - (n-1) t c1 / n ]
    /// ```
    ///
    /// Curvatures rather than radii throughout, so an infinite radius is `c = 0`
    /// and needs no special case.
    pub fn paraxial(&self) -> Result<Paraxial, Error> {
        let scene = Scene::new();
        let glass = scene
            .catalog
            .find(&self.material)
            .and_then(|id| scene.catalog.get(id))
            .ok_or_else(|| Error::UnknownGlass(self.material.clone()))?;
        // From the Sellmeier coefficients, which are the authoritative dispersion
        // (§10.3): `nd` is a cache, and the tracer refracts with the same formula.
        let n = glass.sell.index(DESIGN_LAMBDA_UM);

        let (c1, c2, t) = (self.c1(), self.c2(), self.center_thickness_mm);
        let inv_f = (n - 1.0) * (c1 - c2 + (n - 1.0) * t * c1 * c2 / n);
        // A flat plate has no power: report an infinite focal length rather than
        // dividing by zero and reporting something that looks like a number.
        let efl = if inv_f == 0.0 { f64::INFINITY } else { 1.0 / inv_f };
        let bfd = efl * (1.0 - (n - 1.0) * t * c1 / n);

        Ok(Paraxial {
            efl_mm: efl,
            bfd_mm: bfd,
            rear_focus_x_mm: t / 2.0 + bfd,
            nd: n,
            lambda_um: DESIGN_LAMBDA_UM,
        })
    }
}

/// `1/R`, with an infinite radius meaning a plane.
fn curvature(r: f64) -> f64 {
    if r.is_finite() && r != 0.0 {
        1.0 / r
    } else {
        0.0
    }
}

/// Sag of a spherical surface at height `y`, signed by the curvature.
fn sag(c: f64, y: f64) -> f64 {
    let d = 1.0 - c * c * y * y;
    c * y * y / (1.0 + d.max(0.0).sqrt())
}

/// Read a number, or the one number JSON cannot hold.
///
/// **A plane radius has to be a string.** JSON has no infinity: `json!(f64::INFINITY)`
/// serializes to `null`, and Python's `json.dump(float("inf"))` writes a bare
/// `Infinity` that is not JSON at all and that `serde_json` refuses. A plane is a
/// perfectly ordinary surface — §9.6's own pilot descriptor is plano-convex with
/// `radius_2_mm: .inf` — so the catalogue compiler must emit `"inf"`, and this
/// reads it back.
fn number_param(p: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<f64, Error> {
    let v = p
        .get(key)
        .ok_or_else(|| Error::Parameter(format!("missing parameter {key:?}")))?;
    if let Some(s) = v.as_str() {
        return match s.trim().to_ascii_lowercase().as_str() {
            "inf" | "+inf" | "infinity" => Ok(f64::INFINITY),
            "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
            other => Err(Error::Parameter(format!(
                "parameter {key:?} is {other:?}, which is not a number"
            ))),
        };
    }
    if v.is_null() {
        // Almost always a writer that tried to put an infinity through JSON and
        // got this instead. Say the fix rather than "not a number".
        return Err(Error::Parameter(format!(
            "parameter {key:?} is null; JSON has no infinite number, so write a plane \
             surface as the string \"inf\""
        )));
    }
    v.as_f64()
        .ok_or_else(|| Error::Parameter(format!("parameter {key:?} is not a number: {v}")))
}

fn string_param(p: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<String, Error> {
    p.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| Error::Parameter(format!("missing or non-string parameter {key:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A radius as the catalogue can actually carry it.
    ///
    /// JSON has no infinity — `json!(f64::INFINITY)` is `null` — so a plane is
    /// the string `"inf"`, and the fixtures have to go through the same door a
    /// real descriptor does.
    fn radius(r: f64) -> serde_json::Value {
        if r.is_infinite() {
            serde_json::json!(if r > 0.0 { "inf" } else { "-inf" })
        } else {
            serde_json::json!(r)
        }
    }

    fn params(r1: f64, r2: f64) -> BTreeMap<String, serde_json::Value> {
        let mut p = BTreeMap::new();
        p.insert("material".into(), serde_json::json!("N-BK7"));
        p.insert("diameter_mm".into(), serde_json::json!(24.0));
        p.insert("center_thickness_mm".into(), serde_json::json!(5.0));
        p.insert("radius_1_mm".into(), radius(r1));
        p.insert("radius_2_mm".into(), radius(r2));
        p
    }

    fn biconvex() -> SingletParams {
        SingletParams::from_json(&params(100.0, -100.0)).unwrap()
    }

    #[test]
    fn a_singlet_builds_one_element_of_catalogue_glass() {
        let f = Generator::Singlet.build(&params(100.0, -100.0)).unwrap();
        assert_eq!(f.elements.len(), 1);
        assert_eq!(f.elements[0].boundary.len(), 4, "two surfaces and two rims");

        // The fragment carries the glass table its material id points into
        // (§10.1). It may carry the whole catalogue — insertion merges only what
        // is referenced — so what matters is that the id resolves to N-BK7 here,
        // not how many glasses came along.
        let id = match f.elements[0].material {
            Material::Catalog(id) => id,
            ref m => panic!("expected catalogue glass, got {m:?}"),
        };
        let glass = f.glasses.get(id.0 as usize).expect("the id resolves");
        assert!(glass.name.eq_ignore_ascii_case("N-BK7"), "{}", glass.name);
    }

    #[test]
    fn a_plane_cannot_be_a_json_number_and_says_so() {
        // The trap this cost an hour of: `json!(f64::INFINITY)` is `null`, and
        // Python writes a bare `Infinity` that is not JSON. A descriptor's plane
        // has to be the string, and the error has to say which.
        assert!(serde_json::json!(f64::INFINITY).is_null(), "JSON has no infinity");

        let mut p = params(50.0, 0.0);
        p.insert("radius_2_mm".into(), serde_json::json!(f64::INFINITY));
        let err = SingletParams::from_json(&p).unwrap_err();
        assert!(format!("{err}").contains("\"inf\""), "{err}");
    }

    #[test]
    fn an_unknown_generator_is_an_error_not_an_attempt() {
        let err = Generator::from_name("frobnicate").unwrap_err();
        assert!(format!("{err}").contains("frobnicate"), "{err}");
    }

    #[test]
    fn the_convention_is_the_prescription_one() {
        // R1 > 0 and R2 < 0 is biconvex: thicker in the middle than at the rim.
        // If the rear surface were built on a reversed axis — as oc-wasm's
        // freehand inspector does — this same prescription would come out as a
        // meniscus, and every number downstream would be wrong while every file
        // still parsed.
        let lens = biconvex();
        assert!(lens.rim_thickness() < lens.center_thickness_mm, "convex");
        assert!(lens.rim_thickness() > 0.0);

        // And the mirror image is a biconcave: thinner in the middle.
        let concave = SingletParams::from_json(&params(-100.0, 100.0)).unwrap();
        assert!(concave.rim_thickness() > concave.center_thickness_mm);
    }

    #[test]
    fn a_biconvex_lens_converges_and_a_biconcave_one_does_not() {
        assert!(biconvex().paraxial().unwrap().efl_mm > 0.0);
        let concave = SingletParams::from_json(&params(-100.0, 100.0)).unwrap();
        assert!(concave.paraxial().unwrap().efl_mm < 0.0, "diverging");
    }

    #[test]
    fn the_lensmakers_equation_matches_a_hand_calculation() {
        // R1 = 100, R2 = -100, t = 5, N-BK7 at the d-line (n = 1.5168):
        //   1/f = 0.5168 * [0.01 + 0.01 + 0.5168*5*0.01*(-0.01)/1.5168]
        //       = 0.5168 * 0.0198296 = 0.0102481  ->  f = 97.58 mm
        //   BFD = f * (1 - 0.5168*5*0.01/1.5168) = 97.58 * 0.982964 = 95.92 mm
        let p = biconvex().paraxial().unwrap();
        assert!((p.nd - 1.5168).abs() < 1e-3, "N-BK7 nd = {}", p.nd);
        assert!((p.efl_mm - 97.58).abs() < 0.05, "efl = {}", p.efl_mm);
        assert!((p.bfd_mm - 95.92).abs() < 0.05, "bfd = {}", p.bfd_mm);
        // The rear vertex is at +t/2, so the focus sits that much further along.
        assert!((p.rear_focus_x_mm - (2.5 + p.bfd_mm)).abs() < 1e-12);
    }

    #[test]
    fn a_plano_convex_lens_focuses_at_its_own_focal_length() {
        // With a flat rear, BFD = f * (1 - (n-1) t c1 / n): the front surface does
        // all the work and the thickness still shifts the focus.
        let p = SingletParams::from_json(&params(50.0, f64::INFINITY)).unwrap();
        let par = p.paraxial().unwrap();
        // 1/f = (n-1)/R1 exactly, because c2 = 0 kills the thickness term too.
        let expected = 50.0 / (par.nd - 1.0);
        assert!((par.efl_mm - expected).abs() < 1e-9, "{par:?}");
        assert!(par.bfd_mm < par.efl_mm, "the rear vertex is inside the lens");
    }

    #[test]
    fn a_flat_plate_has_no_power() {
        let p = SingletParams::from_json(&params(f64::INFINITY, f64::INFINITY)).unwrap();
        assert!(p.paraxial().unwrap().efl_mm.is_infinite());
    }

    #[test]
    fn an_infinite_radius_may_be_written_as_a_string() {
        // YAML `.inf` survives some writers as the string "inf". A plane is a real
        // surface, and refusing to read one back would be a silly way to fail.
        let mut p = params(50.0, 0.0);
        p.insert("radius_2_mm".into(), serde_json::json!("inf"));
        assert!(SingletParams::from_json(&p).unwrap().radius_2_mm.is_infinite());
    }

    #[test]
    fn a_radius_shorter_than_the_semi_diameter_is_refused() {
        // The one that would otherwise trace happily: past |R| the sag's square
        // root is imaginary, the geometry clamps, and the shape is not a sphere.
        let err = SingletParams::from_json(&params(5.0, -100.0)).unwrap_err();
        assert!(format!("{err}").contains("semi-diameter"), "{err}");
    }

    #[test]
    fn a_zero_radius_is_refused_rather_than_read_as_a_plane() {
        // `spherical_edge` maps a zero radius to a plane, which would make R = 0
        // silently mean "flat" — the opposite of what anyone typing 0 expects.
        let err = SingletParams::from_json(&params(0.0, -100.0)).unwrap_err();
        assert!(format!("{err}").contains("zero radius"), "{err}");
    }

    #[test]
    fn a_lens_whose_surfaces_cross_is_refused() {
        // 1 mm of centre thickness cannot hold two R = 20 surfaces apart over a
        // 24 mm diameter: the rim thickness goes negative and the "lens" is a
        // self-intersecting boundary.
        let mut p = params(20.0, -20.0);
        p.insert("center_thickness_mm".into(), serde_json::json!(1.0));
        let err = SingletParams::from_json(&p).unwrap_err();
        assert!(format!("{err}").contains("cross"), "{err}");
    }

    #[test]
    fn a_missing_parameter_is_an_error_rather_than_a_guess() {
        let mut p = params(100.0, -100.0);
        p.remove("radius_1_mm");
        let err = SingletParams::from_json(&p).unwrap_err();
        assert!(format!("{err}").contains("radius_1_mm"), "{err}");
    }

    #[test]
    fn an_unknown_glass_is_an_error() {
        let mut p = params(100.0, -100.0);
        p.insert("material".into(), serde_json::json!("UNOBTAINIUM"));
        let err = SingletParams::from_json(&p).unwrap().fragment().unwrap_err();
        assert!(format!("{err}").contains("UNOBTAINIUM"), "{err}");
    }

    #[test]
    fn a_shorter_radius_makes_a_stronger_lens() {
        // The property the whole feature exists for: turn the radius down and the
        // lens bends light harder.
        let long = SingletParams::from_json(&params(200.0, -200.0)).unwrap();
        let short = SingletParams::from_json(&params(50.0, -50.0)).unwrap();
        assert!(short.paraxial().unwrap().efl_mm < long.paraxial().unwrap().efl_mm);
    }
}
