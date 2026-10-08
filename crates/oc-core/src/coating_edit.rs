//! Editing a coating's stack in place, and saying what is wrong with one
//! (coating-fidelity spec CF-11).
//!
//! An object's coating lives in more than one copy: a surface carries it on
//! its front and, seen from behind, on its back; a coated body face carries it
//! from outside and from inside. An edit is applied to every copy the same
//! way, and the scene is validated afterwards: an edit the scene refuses is
//! undone and the reason returned.
//!
//! A hand edit makes the stack the user's own. The design record keeps its
//! kind (an edited AR is still an AR, and its reflection still a ghost) but
//! loses its substrate adaptation, so changing the glass afterwards does not
//! re-solve the layers over the edit.

use crate::scene3::{Scene3, SurfaceResponse};
use crate::thinfilm::{Coating, CoatingBacking, CoatingDesign, DesignKind, FilmContext, FilmLayer, FilmMaterial};

/// One change to a stack. Indices count layers in the written order, from the
/// incident medium inward.
#[derive(Debug, Clone, PartialEq)]
pub enum StackEdit {
    Insert { index: usize, layer: FilmLayer },
    Remove { index: usize },
    Move { from: usize, to: usize },
    Material { index: usize, material: FilmMaterial },
    Thickness { index: usize, nm: f64 },
    Backing(CoatingBacking),
}

impl StackEdit {
    fn apply(&self, c: &mut Coating) -> Result<(), String> {
        let n = c.layers.len();
        let bad = |i: usize| format!("there is no layer {} in a stack of {n}", i + 1);
        match self {
            StackEdit::Insert { index, layer } => {
                if *index > n {
                    return Err(bad(*index));
                }
                c.layers.insert(*index, layer.clone());
            }
            StackEdit::Remove { index } => {
                if *index >= n {
                    return Err(bad(*index));
                }
                c.layers.remove(*index);
            }
            StackEdit::Move { from, to } => {
                if *from >= n || *to >= n {
                    return Err(bad((*from).max(*to)));
                }
                let l = c.layers.remove(*from);
                c.layers.insert(*to, l);
            }
            StackEdit::Material { index, material } => {
                let l = c.layers.get_mut(*index).ok_or_else(|| bad(*index))?;
                l.material = material.clone();
                l.catalog_ref = None;
            }
            StackEdit::Thickness { index, nm } => {
                if !nm.is_finite() || *nm < 0.0 {
                    return Err("a thickness must be finite and not negative".into());
                }
                c.layers.get_mut(*index).ok_or_else(|| bad(*index))?.thickness_nm = *nm;
            }
            StackEdit::Backing(b) => c.backing = b.clone(),
        }
        if let Some(d) = c.design.as_mut() {
            d.adapt = None;
        }
        Ok(())
    }
}

/// Visit every coating response object `id` carries, in a fixed order.
fn for_coatings(scene: &mut Scene3, id: u64, f: &mut dyn FnMut(&mut Coating)) {
    let mut visit = |r: &mut SurfaceResponse| {
        if let SurfaceResponse::Coating { coating } = r {
            f(coating);
        }
    };
    if let Some(s) = scene.surfaces.iter_mut().find(|s| s.common.id.0 == id) {
        visit(&mut s.response_front);
        visit(&mut s.response_back);
    } else if let Some(b) = scene.bodies.iter_mut().find(|b| b.common.id.0 == id) {
        visit(&mut b.surface.default.from_outside);
        visit(&mut b.surface.default.from_inside);
        for o in &mut b.surface.overrides {
            visit(&mut o.response.from_outside);
            visit(&mut o.response.from_inside);
        }
    }
}

/// The name a design kind goes by in a sentence.
fn kind_name(k: DesignKind) -> String {
    match k {
        DesignKind::BroadbandArLegacy => "BroadbandAr (nominal, not adapted)".into(),
        DesignKind::EnhancedMetal { metal, pairs } => {
            format!("EnhancedMetal ({metal:?}, {pairs} pair{})", if pairs == 1 { "" } else { "s" })
        }
        other => format!("{other:?}"),
    }
}

/// The design record in a sentence: "Bbar adapted to n 1.920, band 400 to
/// 700 nm, at 0 degrees".
pub fn describe_design(d: &CoatingDesign) -> String {
    let deg = |a: f64| format!("{}", (a * 10.0).round() / 10.0);
    match &d.adapt {
        Some(a) => {
            let angles = if a.aoi_range_deg[0] == a.aoi_range_deg[1] {
                format!("at {} degrees", deg(a.aoi_range_deg[0]))
            } else {
                format!("{} to {} degrees", deg(a.aoi_range_deg[0]), deg(a.aoi_range_deg[1]))
            };
            format!(
                "{} adapted to n {:.3}, band {} to {} nm, {angles}",
                kind_name(d.kind),
                a.substrate_n,
                (a.band_um[0] * 1000.0).round(),
                (a.band_um[1] * 1000.0).round()
            )
        }
        None => format!("{} at {} nm, {} degrees", kind_name(d.kind), d.lambda_nm.round(), deg(d.aoi_deg)),
    }
}

impl Scene3 {
    /// Apply `edit` to every copy of object `id`'s coating, then validate the
    /// scene; a refused edit is undone and the reason returned.
    pub fn edit_coating(&mut self, id: u64, edit: &StackEdit) -> Result<(), String> {
        let mut before: Vec<Coating> = Vec::new();
        for_coatings(self, id, &mut |c| before.push(c.clone()));
        if before.is_empty() {
            return Err("this object carries no thin-film coating".into());
        }
        let mut result = Ok(());
        for_coatings(self, id, &mut |c| {
            if result.is_ok() {
                result = edit.apply(c);
            }
        });
        if result.is_ok() {
            result = self.validate().map_err(|e| e.to_string());
        }
        if result.is_err() {
            let mut old = before.into_iter();
            for_coatings(self, id, &mut |c| {
                if let Some(o) = old.next() {
                    *c = o;
                }
            });
        }
        result
    }

    /// The distinct wavelengths the scene's sources launch, µm, sorted.
    pub fn launch_wavelengths_um(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self.sources.iter().flat_map(|s| s.spectrum.samples().into_iter().map(|(l, _)| l)).collect();
        v.sort_by(f64::total_cmp);
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        v
    }

    /// What is wrong with object `id`'s coating, in sentences: its design
    /// record first, then every material without data at a wavelength the
    /// scene launches, then every refusal a `Strict` trace would meet at those
    /// wavelengths at the angle the light arrives at. `None` for an object with
    /// no coating.
    pub fn coating_report(&self, id: u64) -> Option<Vec<String>> {
        let c = self.coating_for_object(id)?;
        let mut out = vec![match &c.design {
            Some(d) => format!("design: {}", describe_design(d)),
            None => "design: none recorded (a hand-written stack)".into(),
        }];
        let lambdas = self.launch_wavelengths_um();
        let range = |m: &FilmMaterial| m.provenance.as_ref().and_then(|p| p.range_um);
        let mut check = |what: String, m: &FilmMaterial| {
            let missing: Vec<f64> = lambdas.iter().copied().filter(|&l| !m.covers(l)).collect();
            if let Some(&l) = missing.first() {
                let data = range(m).map_or(String::new(), |[a, b]| {
                    format!(" (its data run from {} to {} nm)", (a * 1000.0).round(), (b * 1000.0).round())
                });
                let more = if missing.len() > 1 { format!(" and {} more lines", missing.len() - 1) } else { String::new() };
                out.push(format!(
                    "{what} has no data at {} nm{more}{data}: a Strict trace refuses it there, a Permissive one holds the table's end and says so",
                    (l * 1000.0 * 10.0).round() / 10.0
                ));
            }
        };
        for (i, l) in c.layers.iter().enumerate() {
            check(format!("layer {} ({})", i + 1, l.material.name), &l.material);
        }
        if let CoatingBacking::Plate { substrate, .. } | CoatingBacking::Opaque { substrate } = &c.backing {
            check(format!("the substrate ({})", substrate.name), substrate);
        }
        let aoi = self.object_axial_aoi_deg(id).unwrap_or(0.0);
        let strict = FilmContext { numerics: crate::fidelity::Numerics::Strict, ..FilmContext::from_profile(&self.fidelity) };
        let mut seen = Vec::new();
        for &l in &lambdas {
            let (n0, n_far) = if self.surfaces.iter().any(|s| s.common.id.0 == id) {
                (self.env.ambient_index, self.env.ambient_index)
            } else {
                self.coated_face_media(id, l)
            };
            if let Err(e) = crate::thinfilm::coating_eval(c, n0, n_far, l, aoi.to_radians().cos(), &strict) {
                let s = e.to_string();
                if !seen.contains(&s) {
                    out.push(format!("refused at {} nm: {s}", (l * 1000.0 * 10.0).round() / 10.0));
                    seen.push(s);
                }
            }
        }
        Some(out)
    }
}
