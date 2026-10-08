//! How much physics a computation pays for: the fidelity profile.
//!
//! Every coating effect the kernel can model costs time, and not every
//! question needs all of them. A user sketching a layout does not need
//! per-layer heat and full Stokes transport; a user signing off a design does.
//! The profile is the one object that says which effects a run includes. It is
//! the user's choice, never the kernel's: nothing here picks a profile, and
//! nothing downstream may raise or lower one silently.
//!
//! Each switch has one value that reproduces the kernel as it was before the
//! switch existed, bit for bit, and [`FidelityProfile::default`] is made of
//! those values ([`Preset::Standard`]). Every result carries the profile it was
//! computed under as a [`FidelityStamp`], so two numbers computed under two
//! profiles cannot be mistaken for each other.
//!
//! The spec is `SPEC-OPTILAND-CANVAS-COATING-FIDELITY.md`, §5.

use serde::{Deserialize, Serialize};

/// How a coating answers a ray.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CoatingModel {
    /// The design's R and T at its design wavelength and angle, whatever the
    /// ray: no wavelength or angle dependence, unpolarized. Phenomenological,
    /// and labelled so.
    Nominal,
    /// The full transfer matrix, s and p powers averaged: unpolarized light in,
    /// unpolarized light out. What the f32 preview does.
    Unpolarized,
    /// The full transfer matrix with its phases, acting on the ray's Stokes
    /// vector. The f64 kernel's behaviour before the profile existed.
    #[default]
    Polarized,
}

/// Which optical constants a film material contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FilmMaterials {
    /// Every film at its catalogue entry's nominal constant.
    Nominal,
    /// Every film at its entry's n(λ), k(λ). A constant entry is its constant
    /// either way, which is why the pre-profile corpora do not move.
    #[default]
    Dispersive,
}

/// What is recorded of the power a coating absorbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LayerAbsorption {
    /// The absorbed power still leaves the ray and the ledger, but no surface
    /// deposit is recorded.
    Off,
    /// One deposit per face: the whole stack's absorption.
    #[default]
    PerFace,
    /// One deposit per absorbing layer, from the Poynting flux at each
    /// interface, and the substrate's loss in its own.
    PerLayer,
}

/// How a plate-backed coating acts on polarized light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PlateModel {
    /// Each power's square root as a phase-free Jones element: exact for
    /// unpolarized light, an approximation for polarized light, and flagged.
    #[default]
    PhaseFree,
    /// The incoherent Mueller series of the two faces and the slab.
    Incoherent,
}

/// Whether films are evaluated at the scene's declared temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Temperature {
    #[default]
    Off,
    /// First-order thermo-optic shift at the declared temperature: declared,
    /// not validated.
    Declared,
}

/// What happens at the edge of a model's domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Numerics {
    /// Out-of-range tables clamp and the result is flagged.
    #[default]
    Permissive,
    /// Out-of-range and degenerate inputs are refused with a named reason.
    Strict,
}

/// One run's physics, switch by switch (spec §5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(from = "ProfileDoc", into = "ProfileDoc")]
pub struct FidelityProfile {
    pub coating_model: CoatingModel,
    pub film_materials: FilmMaterials,
    pub layer_absorption: LayerAbsorption,
    pub plate_model: PlateModel,
    pub temperature: Temperature,
    pub numerics: Numerics,
    /// Ghost generations. `None` leaves the trace's own ghost setting alone;
    /// `Some(0)` turns ghosts off; `Some(n)` follows `n` generations.
    pub ghosts: Option<u32>,
}

/// A named profile. Any other combination of switches is custom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Preset {
    /// For sketching: nominal coatings, nothing recorded, no ghosts.
    Fast,
    /// The kernel as it was before the profile existed, bit for bit.
    Standard,
    /// Everything the kernel can model, for a final check.
    Reference,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Fast, Preset::Standard, Preset::Reference];

    /// The profile this preset names. `ghosts` is what "as set" means for the
    /// presets that leave ghosts alone; `has_temperature` says whether the
    /// scene declares one, which `Reference` then uses.
    pub fn profile(self, ghosts: Option<u32>, has_temperature: bool) -> FidelityProfile {
        match self {
            Preset::Fast => FidelityProfile {
                coating_model: CoatingModel::Nominal,
                film_materials: FilmMaterials::Nominal,
                layer_absorption: LayerAbsorption::Off,
                plate_model: PlateModel::PhaseFree,
                temperature: Temperature::Off,
                numerics: Numerics::Permissive,
                ghosts: Some(0),
            },
            Preset::Standard => FidelityProfile { ghosts, ..FidelityProfile::default() },
            Preset::Reference => FidelityProfile {
                coating_model: CoatingModel::Polarized,
                film_materials: FilmMaterials::Dispersive,
                layer_absorption: LayerAbsorption::PerLayer,
                plate_model: PlateModel::Incoherent,
                temperature: if has_temperature { Temperature::Declared } else { Temperature::Off },
                numerics: Numerics::Strict,
                ghosts,
            },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Fast => "Fast",
            Preset::Standard => "Standard",
            Preset::Reference => "Reference",
        }
    }
}

impl FidelityProfile {
    /// The named preset these switches match, or `None` for a custom profile.
    ///
    /// `Fast` turns ghosts off, so it matches only with ghosts off; the other
    /// two leave ghosts as set and match whatever the ghost setting is.
    /// `Reference` declares the temperature only when the scene has one, so it
    /// matches either temperature value.
    pub fn preset(&self) -> Option<Preset> {
        let fast = Preset::Fast.profile(Some(0), false);
        if *self == fast {
            return Some(Preset::Fast);
        }
        let any_ghosts = FidelityProfile { ghosts: None, ..*self };
        if any_ghosts == Preset::Standard.profile(None, false) {
            return Some(Preset::Standard);
        }
        let any_temperature = FidelityProfile { temperature: Temperature::Off, ..any_ghosts };
        if any_temperature == Preset::Reference.profile(None, false) {
            return Some(Preset::Reference);
        }
        None
    }

    /// Is every switch at the value that reproduces the pre-profile kernel?
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Is the trace polarized: Stokes vectors carried and coating phases
    /// applied?
    pub fn polarized(&self) -> bool {
        self.coating_model == CoatingModel::Polarized
    }

    /// The approximations this profile makes, one short sentence each, for
    /// the stamp. A profile at `Reference` makes none of its own.
    pub fn approximations(&self) -> Vec<String> {
        let mut out = Vec::new();
        match self.coating_model {
            CoatingModel::Nominal => out.push(
                "coatings nominal: each answers with its design-point R and T at every wavelength and angle; nominal, not verified"
                    .to_string(),
            ),
            CoatingModel::Unpolarized => out.push(
                "coatings unpolarized: s and p powers averaged, no phase, unpolarized light handed on".to_string(),
            ),
            CoatingModel::Polarized => {}
        }
        if self.film_materials == FilmMaterials::Nominal {
            out.push("film materials at their nominal constants, no dispersion".to_string());
        }
        match self.layer_absorption {
            LayerAbsorption::Off => out.push("coating absorption not deposited".to_string()),
            LayerAbsorption::PerFace => out.push("coating absorption deposited per face, not per layer".to_string()),
            LayerAbsorption::PerLayer => {}
        }
        if self.plate_model == PlateModel::PhaseFree {
            out.push("coated plates phase-free: polarized light through a plate is approximated".to_string());
        }
        if self.numerics == Numerics::Permissive {
            out.push("out-of-range tables clamp; flagged where it happens".to_string());
        }
        if self.ghosts == Some(0) {
            out.push("ghosts not followed".to_string());
        }
        out
    }

    /// The profile in words, for a person reading a result: "computed with
    /// Standard: coatings polarized, materials dispersive, absorption per
    /// face, ...".
    pub fn describe(&self) -> String {
        let name = self.preset().map_or("a custom profile", Preset::name);
        let coatings = match self.coating_model {
            CoatingModel::Nominal => "coatings nominal",
            CoatingModel::Unpolarized => "coatings unpolarized",
            CoatingModel::Polarized => "coatings polarized",
        };
        let materials = match self.film_materials {
            FilmMaterials::Nominal => "materials nominal",
            FilmMaterials::Dispersive => "materials dispersive",
        };
        let absorption = match self.layer_absorption {
            LayerAbsorption::Off => "absorption not deposited",
            LayerAbsorption::PerFace => "absorption per face",
            LayerAbsorption::PerLayer => "absorption per layer",
        };
        let plates = match self.plate_model {
            PlateModel::PhaseFree => "plates phase-free",
            PlateModel::Incoherent => "plates incoherent",
        };
        let temperature = match self.temperature {
            Temperature::Off => "no temperature",
            Temperature::Declared => "temperature declared, not validated",
        };
        let numerics = match self.numerics {
            Numerics::Permissive => "numerics permissive",
            Numerics::Strict => "numerics strict",
        };
        let ghosts = match self.ghosts {
            None => String::new(),
            Some(0) => ", no ghosts".to_string(),
            Some(n) => format!(", ghosts to depth {n}"),
        };
        let fast = if self.coating_model == CoatingModel::Nominal { " (nominal, not verified)" } else { "" };
        format!(
            "computed with {name}{fast}: {coatings}, {materials}, {absorption}, {plates}, {temperature}, {numerics}{ghosts}"
        )
    }

    /// A short hash of the switches, for cache keys that must never serve a
    /// result computed under one profile to another.
    pub fn key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

/// The record a result carries of the profile it was computed under (spec
/// §5.3). Structured, so a report can refuse to merge results of two profiles
/// by reading it; the text a person reads is rendered from it and nothing
/// parses the text back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidelityStamp {
    pub profile: FidelityProfile,
    /// The preset the switches match, `None` for custom.
    pub preset: Option<Preset>,
    pub approximations: Vec<String>,
    /// Which solver produced the numbers.
    pub solver_version: String,
}

/// The thin-film solver revision, bumped with any change that moves a number.
pub const SOLVER_VERSION: &str = concat!("oc-core ", env!("CARGO_PKG_VERSION"), ", thin film: Macleod admittance TMM r2");

impl FidelityStamp {
    pub fn new(profile: FidelityProfile) -> Self {
        Self {
            profile,
            preset: profile.preset(),
            approximations: profile.approximations(),
            solver_version: SOLVER_VERSION.to_string(),
        }
    }

    pub fn text(&self) -> String {
        self.profile.describe()
    }
}

impl Default for FidelityStamp {
    fn default() -> Self {
        Self::new(FidelityProfile::default())
    }
}

/// The profile as a document stores it: the switches, and the preset they
/// match written alongside for a person reading the file. The preset is
/// informational; on read the switches are the truth.
#[derive(Serialize, Deserialize)]
struct ProfileDoc {
    #[serde(default, skip_deserializing)]
    preset: Option<String>,
    #[serde(default)]
    coating_model: CoatingModel,
    #[serde(default)]
    film_materials: FilmMaterials,
    #[serde(default)]
    layer_absorption: LayerAbsorption,
    #[serde(default)]
    plate_model: PlateModel,
    #[serde(default)]
    temperature: Temperature,
    #[serde(default)]
    numerics: Numerics,
    #[serde(default)]
    ghosts: Option<u32>,
}

impl From<ProfileDoc> for FidelityProfile {
    fn from(d: ProfileDoc) -> Self {
        Self {
            coating_model: d.coating_model,
            film_materials: d.film_materials,
            layer_absorption: d.layer_absorption,
            plate_model: d.plate_model,
            temperature: d.temperature,
            numerics: d.numerics,
            ghosts: d.ghosts,
        }
    }
}

impl From<FidelityProfile> for ProfileDoc {
    fn from(p: FidelityProfile) -> Self {
        Self {
            preset: Some(p.preset().map_or("Custom", Preset::name).to_string()),
            coating_model: p.coating_model,
            film_materials: p.film_materials,
            layer_absorption: p.layer_absorption,
            plate_model: p.plate_model,
            temperature: p.temperature,
            numerics: p.numerics,
            ghosts: p.ghosts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_standard_and_standard_is_the_pre_profile_kernel() {
        let p = FidelityProfile::default();
        assert_eq!(p.preset(), Some(Preset::Standard));
        assert!(p.polarized(), "the f64 kernel was polarized before the profile existed");
        assert_eq!(p.layer_absorption, LayerAbsorption::PerFace);
        assert_eq!(p.plate_model, PlateModel::PhaseFree);
    }

    #[test]
    fn each_preset_is_recognized_and_one_change_makes_it_custom() {
        for preset in Preset::ALL {
            for ghosts in [None, Some(3)] {
                let p = preset.profile(ghosts, true);
                assert_eq!(p.preset(), Some(preset), "{preset:?} with ghosts {ghosts:?}");
            }
        }
        // Fast turns ghosts off; turning them back on is custom.
        let fast_with_ghosts = FidelityProfile { ghosts: Some(2), ..Preset::Fast.profile(Some(0), false) };
        assert_eq!(fast_with_ghosts.preset(), None);
        let unpolarized = FidelityProfile { coating_model: CoatingModel::Unpolarized, ..Default::default() };
        assert_eq!(unpolarized.preset(), None);
    }

    #[test]
    fn the_document_form_writes_the_preset_and_reads_the_switches() {
        let p = Preset::Reference.profile(Some(2), false);
        let text = serde_json::to_string(&p).unwrap();
        assert!(text.contains("\"preset\":\"Reference\""), "{text}");
        assert_eq!(serde_json::from_str::<FidelityProfile>(&text).unwrap(), p);
        // A preset name that does not match the switches is ignored on read:
        // the switches are the truth, and they say custom.
        let lie = text.replace("\"Strict\"", "\"Permissive\"");
        let read: FidelityProfile = serde_json::from_str(&lie).unwrap();
        assert_eq!(read.numerics, Numerics::Permissive);
        assert_eq!(read.preset(), None);
        // The spec's example (§10) reads.
        let spec = r#"{"preset": "Standard", "coating_model": "Unpolarized", "film_materials": "Dispersive",
            "layer_absorption": "PerFace", "plate_model": "PhaseFree", "temperature": "Off",
            "numerics": "Permissive", "ghosts": 2}"#;
        let read: FidelityProfile = serde_json::from_str(spec).unwrap();
        assert_eq!(read.ghosts, Some(2));
        // An empty object is the default.
        assert_eq!(serde_json::from_str::<FidelityProfile>("{}").unwrap(), FidelityProfile::default());
    }

    #[test]
    fn fast_says_it_is_not_verified_and_reference_makes_no_approximation_of_its_own() {
        let fast = FidelityStamp::new(Preset::Fast.profile(Some(0), false));
        assert!(fast.text().contains("nominal, not verified"), "{}", fast.text());
        assert!(fast.approximations.iter().any(|a| a.contains("not verified")));
        let reference = FidelityStamp::new(Preset::Reference.profile(None, false));
        assert!(reference.approximations.is_empty(), "{:?}", reference.approximations);
        assert_eq!(reference.preset, Some(Preset::Reference));
        assert!(Preset::Standard.profile(None, false).describe().starts_with("computed with Standard:"));
    }

    #[test]
    fn profiles_that_differ_have_different_keys() {
        let mut keys = std::collections::HashSet::new();
        for preset in Preset::ALL {
            keys.insert(preset.profile(None, false).key());
        }
        keys.insert(FidelityProfile { coating_model: CoatingModel::Unpolarized, ..Default::default() }.key());
        assert_eq!(keys.len(), 4);
    }
}
