//! A 3D scene built from a sequential prescription: the inverse of
//! `oc_trace::seq_export`, and the way a lens designed elsewhere (an Optiland
//! `Optic`, read by `optiland_canvas`) comes into the canvas.
//!
//! **Surfaces become solids.** Two surfaces with glass between them bound one
//! revolved body. Its rim is cut at one diameter where both faces reach it,
//! and stepped the way Optiland draws a lens where one face stops short. Two
//! bodies that meet at a cemented surface share that face exactly. A surface
//! with the same medium on both sides changes nothing, so it becomes nothing,
//! unless it is the stop.
//!
//! **The stop is a place and a declaration.** It becomes an aperture at its
//! surface's vertex, the opening its clear radius, and the scene's path
//! declares it. Exporting the scene again finds the same stop, and with it the
//! same pupils.
//!
//! **Fields become sources.** Optiland launches the rays of a field at an
//! object at infinity across the paraxial entrance pupil, so each such field
//! is a collimated beam whose footprint on a plane square to the axis is that
//! pupil. A field at a finite distance is a point on the object plane, its
//! cone filling the pupil: about the axis when object space is telecentric,
//! about the line to the pupil's centre otherwise.
//!
//! **What this does not build.** Mirrors, and a virtual image (one a negative
//! last thickness puts before the last surface), are refused, every reason
//! listed. What it builds only approximately is said in the notes.

use crate::geom3::primitives::even_asphere_sag;
use crate::geom3::profile::validate_revolvable;
use crate::id::{ApertureId, LayerId, SourceId};
use crate::material::{Glass, Material};
use crate::profile2::{
    ClosedProfile2, ConicParams2, Curve2, ProfileConstruction3, ProfileEdge2, ProfileSolid3,
};
use crate::scene::Spectrum;
use crate::scene3::{
    AbsorptionModel, AngularDistribution3, Aperture3, ApertureInteraction, ApertureShape3, Body3,
    Detector3, DetectorGeometry3, DetectorMeasure, DetectorResponse, DetectorTermination,
    ObjectCommon, PlanarShape2, PrimaryPathIntent, SamplingSequence, Scene3, SolidGeometry3,
    Source3, SourceSampling, SpatialDistribution3, SurfaceProps3,
};
use crate::seq::{FieldType, SeqAperture, SeqMaterial, SeqSystem};
use crate::vec::Vec2;
use crate::vec3::{Isometry3, Plane3, Vec3};

/// How much wider than the stop's opening its mask is drawn.
const STOP_MASK: f64 = 1.3;
/// How far past the image plane a body holding the image reaches, as a
/// fraction of the system's length.
const IMAGE_MEDIUM_TAIL: f64 = 0.05;
/// How far before the system the collimated beams start, as a fraction of its
/// length.
const SOURCE_GAP: f64 = 0.1;
/// Two faces are cut to one diameter when their clear apertures are this close;
/// further apart, the smaller face ends at its own and a flat shoulder takes
/// the element out to the larger. Carrying a steep face far past the aperture
/// it was designed for makes a shape no designer meant: an eye's paraboloid
/// lens face, 3.7 mm across, carried out to the 26 mm its vitreous body needs,
/// curls 57 mm back and swallows the front of the eye.
const ONE_DIAMETER: f64 = 1.1;

/// What the built scene samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImportOptions {
    /// Rays per field (and per wavelength): the beam's spatial samples, or
    /// the cone's angular ones.
    pub rays_per_field: u32,
    /// The image detector's pixels along each side.
    pub pixels: usize,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            rays_per_field: 64,
            pixels: 64,
        }
    }
}

/// A built scene, and what in it is approximate.
#[derive(Debug, Clone)]
pub struct Imported {
    pub scene: Scene3,
    pub notes: Vec<String>,
}

/// Why a prescription has no scene: every reason found, not only the first.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportError(pub Vec<String>);

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the prescription cannot be built as a scene:")?;
        for r in &self.0 {
            write!(f, "\n  - {r}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ImportError {}

/// One face of a body, in the body's local frame.
#[derive(Debug, Clone, Copy)]
struct Face {
    x: f64,
    c: f64,
    k: f64,
    coeffs: [f64; 8],
    semi: f64,
}

impl Face {
    fn sag(&self, r: f64) -> f64 {
        even_asphere_sag(self.c, self.k, &self.coeffs, r)
    }

    /// Is the face's conic defined out to `r`? Past `(1 + k) c² r² = 1` it
    /// turns back on itself.
    fn reaches(&self, r: f64) -> bool {
        (1.0 + self.k) * self.c * self.c * r * r < 1.0 - 1e-12
    }

    fn curve(&self, half_ap: f64) -> Curve2 {
        let base = ConicParams2 {
            vertex: Vec2::new(self.x, 0.0),
            axis: Vec2::X,
            c: self.c,
            k: self.k,
            half_ap,
        };
        if self.coeffs.iter().all(|a| *a == 0.0) {
            Curve2::Conic(base)
        } else {
            Curve2::EvenAsphere {
                base,
                coeffs: self.coeffs,
            }
        }
    }
}

/// Build the scene for `seq`. See the module docs.
pub fn import_seq(seq: &SeqSystem, opts: &ImportOptions) -> Result<Imported, ImportError> {
    let mut b = Builder {
        scene: Scene3::new(),
        reasons: Vec::new(),
        notes: Vec::new(),
        put: Vec::new(),
    };
    b.build(seq, opts)?;
    Ok(Imported {
        scene: b.scene,
        notes: b.notes,
    })
}

struct Builder {
    scene: Scene3,
    reasons: Vec<String>,
    notes: Vec<String>,
    /// The glasses this build put in the catalog, to tell two different
    /// glasses under one name from one glass met twice.
    put: Vec<Glass>,
}

/// The medium a prescription names, with air and a unit ideal index read as
/// the ambient.
fn ambient(m: &SeqMaterial) -> bool {
    matches!(m, SeqMaterial::Air) || matches!(m, SeqMaterial::Ideal { n } if *n == 1.0)
}

fn same_medium(a: &SeqMaterial, b: &SeqMaterial) -> bool {
    (ambient(a) && ambient(b)) || a == b
}

fn medium_name(m: &SeqMaterial) -> String {
    match m {
        SeqMaterial::Air => "air".into(),
        SeqMaterial::Ideal { n } => format!("n = {n}"),
        SeqMaterial::Abbe { nd, vd } => format!("nd {nd} / Vd {vd}"),
        SeqMaterial::Glass { name, .. } => name.clone(),
        SeqMaterial::Mirror => "mirror".into(),
    }
}

impl Builder {
    fn build(&mut self, seq: &SeqSystem, opts: &ImportOptions) -> Result<(), ImportError> {
        let n = seq.surfaces.len();
        if n == 0 {
            return Err(ImportError(vec!["the prescription has no surfaces".into()]));
        }
        let wavelengths: Vec<f64> = seq.wavelengths.clone();
        if wavelengths.is_empty() || wavelengths.iter().any(|l| !l.is_finite() || *l <= 0.0) {
            self.reasons
                .push("the wavelengths are missing or not positive".into());
        }
        let lambda0 = wavelengths
            .first()
            .copied()
            .unwrap_or(crate::material::LAMBDA_D);

        // --- what each surface is -------------------------------------------
        for (i, s) in seq.surfaces.iter().enumerate() {
            let name = format!("S{}", i + 1);
            if s.reflective || matches!(s.material, SeqMaterial::Mirror) {
                self.reasons.push(format!(
                    "{name} is a mirror; a folded path is not built as a scene yet"
                ));
            }
            if !s.thickness.is_finite() {
                self.reasons
                    .push(format!("{name} has a thickness that is not finite"));
            } else if s.thickness < 0.0 && i + 1 < n {
                self.reasons.push(format!(
                    "{name} has a negative thickness, which only a mirror can make"
                ));
            }
            if s.coefficients.len() > 8 {
                self.reasons.push(format!(
                    "{name} has {} aspheric terms, and a canvas face holds up to 8 (A4 to A18)",
                    s.coefficients.len()
                ));
            }
            let finite = s.radius.map_or(true, f64::is_finite)
                && s.conic.is_finite()
                && s.semi_aperture.is_finite()
                && s.coefficients.iter().all(|a| a.is_finite());
            if !finite {
                self.reasons
                    .push(format!("{name} has a value that is not finite"));
            }
        }
        let last = &seq.surfaces[n - 1];
        if last.thickness.is_finite() && last.thickness <= 0.0 {
            self.reasons.push(format!(
                "the image lies {:.6} mm before the last surface: a virtual image, which a \
                 detector cannot sit at",
                -last.thickness
            ));
        }
        if !self.reasons.is_empty() {
            return Err(ImportError(std::mem::take(&mut self.reasons)));
        }

        // Axial positions, the first vertex at x = 0.
        let mut xs = Vec::with_capacity(n);
        let mut x = 0.0;
        for s in &seq.surfaces {
            xs.push(x);
            x += s.thickness;
        }
        let x_image = x;

        let air = SeqMaterial::Air;
        let before = |i: usize| {
            if i == 0 {
                &air
            } else {
                &seq.surfaces[i - 1].material
            }
        };
        let real: Vec<usize> = (0..n)
            .filter(|&i| !same_medium(before(i), &seq.surfaces[i].material))
            .collect();
        for i in 0..n {
            let s = &seq.surfaces[i];
            let curved = s.radius.is_some() || s.coefficients.iter().any(|a| *a != 0.0);
            if !real.contains(&i) && curved && seq.stop != Some(i) {
                self.notes.push(format!(
                    "S{} has {} on both sides, so it bends no ray and is left out",
                    i + 1,
                    medium_name(before(i))
                ));
            }
        }
        if real.is_empty() {
            return Err(ImportError(vec![
                "no surface changes the medium, so there is nothing to build".into(),
            ]));
        }
        for &i in &real {
            let s = &seq.surfaces[i];
            if s.semi_aperture <= 0.0 {
                self.reasons.push(format!(
                    "S{} has no clear aperture to cut its element to",
                    i + 1
                ));
            }
        }

        // --- media, and the index each has at the primary wavelength ---------
        let mut bulk: Vec<Option<Material>> = Vec::with_capacity(n);
        for s in &seq.surfaces {
            bulk.push(self.bulk(&s.material));
        }
        let index = |this: &Builder, m: &Option<Material>| -> f64 {
            m.as_ref()
                .and_then(|m| m.index_at(lambda0, &this.scene.catalog))
                .unwrap_or(1.0)
        };
        let n_after: Vec<f64> = bulk.iter().map(|m| index(self, m)).collect();

        // --- the image, and how big it is ------------------------------------
        let para = Paraxial {
            seq,
            n_after: &n_after,
        };
        let stop = seq.stop;
        if let Some(s) = stop {
            if s >= n {
                self.reasons.push(format!(
                    "the stop is surface {s}, and the prescription has {n}"
                ));
            } else if seq.surfaces[s].semi_aperture <= 0.0 {
                self.reasons.push(format!(
                    "the stop, S{}, has no clear radius to open to",
                    s + 1
                ));
            }
        }
        if !self.reasons.is_empty() {
            return Err(ImportError(std::mem::take(&mut self.reasons)));
        }
        let pupil = self.pupil(seq, &para);

        // --- bodies ------------------------------------------------------------
        let span_guess = (x_image - xs[0]).abs().max(1e-3);
        let mut lens = 0;
        let mut body_extent: Vec<(f64, f64, f64)> = Vec::new(); // (x_min, x_max, radius)
        for (w, &a) in real.iter().enumerate() {
            let medium = &seq.surfaces[a].material;
            if ambient(medium) {
                continue;
            }
            let Some(material) = bulk[a].clone() else {
                continue; // the reason is already recorded
            };
            lens += 1;
            let front = face(seq, a, 0.0);
            let (back, back_x, to) = match real.get(w + 1) {
                Some(&b) => (face(seq, b, 0.0), xs[b], format!("S{}", b + 1)),
                None => {
                    // The image lies inside this medium: carry the body past
                    // it, so the detector sits in the medium the prescription
                    // puts the image in.
                    let tail = IMAGE_MEDIUM_TAIL * span_guess;
                    self.notes.push(format!(
                        "the image lies inside {}; its body reaches {tail:.4} mm past the image \
                         plane so the detector sits in it",
                        medium_name(medium)
                    ));
                    let semi = front.semi.max(pupil.image_semi * 1.1);
                    (
                        Face {
                            x: 0.0,
                            c: 0.0,
                            k: 0.0,
                            coeffs: [0.0; 8],
                            semi,
                        },
                        x_image + tail,
                        "image".to_string(),
                    )
                }
            };
            let label = format!("L{lens} · S{}–{to} · {}", a + 1, medium_name(medium));
            match self.body(front, back, xs[a], back_x, material, &label) {
                Ok(extent) => body_extent.push(extent),
                Err(why) => self.reasons.push(format!("{label}: {why}")),
            }
        }

        // --- the stop ------------------------------------------------------------
        // With rays aimed through the real stop, the beams overfill their
        // pupils and the stop is what trims them, so its mask reaches the
        // barrel: nothing that misses the opening goes round it.
        let aimed = !seq.aimed_beams.is_empty();
        let widest = body_extent.iter().map(|e| e.2).fold(0.0, f64::max);
        let aperture = stop.map(|s| {
            let opening = seq.surfaces[s].semi_aperture;
            let mask = if aimed {
                (STOP_MASK * opening).max(1.05 * widest)
            } else {
                STOP_MASK * opening
            };
            self.aperture(
                Vec3::new(xs[s], 0.0, 0.0),
                opening,
                mask,
                &format!("stop · S{}", s + 1),
            )
        });
        if aimed {
            if stop.is_none() {
                self.reasons.push(
                    "the rays are aimed through the stop, and the prescription declares none"
                        .into(),
                );
            }
            if seq.object_distance.is_some() {
                self.reasons.push(
                    "aimed beams are built for an object at infinity, and this one is at a \
                     finite distance"
                        .into(),
                );
            }
            if seq.aimed_beams.len() != seq.fields.len().max(1) {
                self.reasons.push(format!(
                    "{} aimed beams for {} fields",
                    seq.aimed_beams.len(),
                    seq.fields.len()
                ));
            }
            if let Some(on_axis) = seq
                .fields
                .iter()
                .position(|f| *f == 0.0)
                .and_then(|i| seq.aimed_beams.get(i))
            {
                let wider = on_axis.half_width_mm.max(on_axis.half_height_mm) / (0.5 * pupil.epd);
                self.notes.push(format!(
                    "the design aims its rays through the real stop: each field's beam covers \
                     its real pupil and the stop trims it there, so an export of this scene \
                     reads an entrance pupil {:.2}% wider than the design's",
                    100.0 * (wider - 1.0)
                ));
            }
        }
        if stop.is_none() {
            self.notes.push(
                "no stop is declared: the entrance pupil is taken at the first surface, and the \
                 scene declares no stop"
                    .into(),
            );
        }

        if !self.reasons.is_empty() {
            return Err(ImportError(std::mem::take(&mut self.reasons)));
        }

        // --- sources ---------------------------------------------------------------
        let x_min = body_extent
            .iter()
            .map(|e| e.0)
            .chain(stop.map(|s| xs[s]))
            .fold(xs[0], f64::min);
        let x_max = body_extent.iter().map(|e| e.1).fold(x_image, f64::max);
        let span = (x_max - x_min).max(1e-3);
        let spectrum = if wavelengths.len() == 1 {
            Spectrum::Monochromatic(wavelengths[0])
        } else {
            Spectrum::Lines(wavelengths.iter().map(|l| (*l, 1.0)).collect())
        };
        let fields = if seq.fields.is_empty() {
            vec![0.0]
        } else {
            seq.fields.clone()
        };
        let mut sources: Vec<(f64, SourceId)> = Vec::new();
        for (i, &f) in fields.iter().enumerate() {
            let at = SourcePlace {
                field: f,
                beam: seq.aimed_beams.get(i).copied(),
                x_first: xs[0],
                x_start: x_min - SOURCE_GAP * span,
            };
            match self.field_source(seq, &pupil, at, opts, &spectrum) {
                Ok(id) => sources.push((f, id)),
                Err(why) => self.reasons.push(why),
            }
        }

        // --- the image detector ---------------------------------------------------
        let radius = (pupil.image_semi * 1.1).max(0.25 * widest).max(1e-3);
        let detector = self.detector(Vec3::new(x_image, 0.0, 0.0), radius, opts.pixels);

        // --- the path the scene is for -------------------------------------------
        if let Some(&(_, on_axis)) = sources
            .iter()
            .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs()))
        {
            let id = self.scene.alloc_path_intent_id();
            self.scene.intent.primary_paths.push(PrimaryPathIntent {
                id,
                source: on_axis,
                detector,
                stop: aperture,
                label: "the design's path".into(),
            });
            if let Some(ap) = aperture {
                if let Some(a) = self.scene.apertures.iter_mut().find(|a| a.id() == ap) {
                    a.primary_stop_for.push(id);
                }
            }
        }
        self.scene.env.wavelengths = wavelengths.iter().map(|l| (*l, 1.0)).collect();

        if !self.reasons.is_empty() {
            return Err(ImportError(std::mem::take(&mut self.reasons)));
        }
        if let Err(e) = self.scene.validate() {
            return Err(ImportError(vec![format!(
                "the built scene does not validate: {e}"
            )]));
        }
        Ok(())
    }

    /// The canvas material for a medium, its glass put in the scene's
    /// catalog. `None` for the ambient, and for a medium that cannot be built
    /// (the reason recorded).
    fn bulk(&mut self, m: &SeqMaterial) -> Option<Material> {
        if ambient(m) {
            return None;
        }
        match m {
            SeqMaterial::Abbe { nd, vd } => Some(Material::AbbePoint { nd: *nd, vd: *vd }),
            SeqMaterial::Mirror | SeqMaterial::Air => None,
            SeqMaterial::Ideal { .. } | SeqMaterial::Glass { .. } => match m.to_glass() {
                Err(why) => {
                    self.reasons.push(why);
                    None
                }
                Ok(Some(g)) => {
                    if let Some(had) = self
                        .put
                        .iter()
                        .find(|p| p.name.eq_ignore_ascii_case(&g.name))
                    {
                        if had.sell != g.sell || had.formula != g.formula {
                            self.reasons
                                .push(format!("two different glasses are both called {}", g.name));
                            return None;
                        }
                    } else {
                        self.put.push(g.clone());
                    }
                    Some(Material::Catalog(self.scene.catalog.put(&g)))
                }
                Ok(None) => {
                    let SeqMaterial::Glass { name, .. } = m else {
                        return None;
                    };
                    match self.scene.catalog.find(name) {
                        Some(id) => Some(Material::Catalog(id)),
                        None => {
                            self.reasons.push(format!(
                                "glass {name} is not in the canvas catalog, and the prescription \
                                 carries no data for it"
                            ));
                            None
                        }
                    }
                }
            },
        }
    }

    /// A revolved body between `front` (at `x_front`) and `back` (at `x_back`),
    /// placed on the axis. Returns its axial extent and radius.
    fn body(
        &mut self,
        front: Face,
        back: Face,
        x_front: f64,
        x_back: f64,
        material: Material,
        label: &str,
    ) -> Result<(f64, f64, f64), String> {
        let mid = 0.5 * (x_front + x_back);
        let front = Face {
            x: x_front - mid,
            ..front
        };
        let back = Face {
            x: x_back - mid,
            ..back
        };
        let (curves, radius) = match one_diameter(front, back) {
            Some(c) => c,
            None => stepped(front, back)?,
        };
        let edges: Vec<ProfileEdge2> = curves
            .into_iter()
            .map(|curve| ProfileEdge2 {
                patch: self.scene.alloc_patch_id(),
                curve,
            })
            .collect();
        let profile = ClosedProfile2 { edges };
        let lo = mid + (front.x + front.sag(front.semi)).min(front.x);
        let hi = mid + (back.x + back.sag(back.semi)).max(back.x);
        let id = self.scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = Isometry3::from_pos(Vec3::new(mid, 0.0, 0.0));
        common.label = label.to_string();
        self.scene.bodies.push(Body3 {
            common,
            geometry: SolidGeometry3::Profile(ProfileSolid3 {
                profile,
                construction: ProfileConstruction3::RevolveLocalX,
            }),
            material,
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3::default(),
            priority: 0,
        });
        Ok((lo, hi, radius))
    }

    fn aperture(&mut self, pos: Vec3, opening: f64, mask: f64, label: &str) -> ApertureId {
        let id = self.scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = facing(pos, Vec3::X);
        common.label = label.to_string();
        self.scene.apertures.push(Aperture3 {
            common,
            shape: ApertureShape3 {
                plane_local: Plane3::XY,
                clear_opening: PlanarShape2::Disc { radius_mm: opening },
                outer_extent: Some(PlanarShape2::Disc { radius_mm: mask }),
            },
            interaction: ApertureInteraction::Absorb,
            primary_stop_for: Vec::new(),
        });
        ApertureId(id.0)
    }

    fn detector(&mut self, pos: Vec3, radius: f64, pixels: usize) -> crate::id::DetectorId {
        let id = self.scene.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = facing(pos, -Vec3::X);
        common.label = "image".into();
        self.scene.detectors.push(Detector3 {
            common,
            geometry: DetectorGeometry3::Disc { radius_mm: radius },
            bins: [pixels.max(1); 2],
            response: DetectorResponse::Unity,
            termination: DetectorTermination::Terminate,
            measure: DetectorMeasure::Power,
            sides: Default::default(),
        });
        crate::id::DetectorId(id.0)
    }

    /// The entrance pupil and the image's size, from the paraxial trace.
    fn pupil(&mut self, seq: &SeqSystem, para: &Paraxial) -> Pupil {
        // Where the pupil is: the stop imaged back into object space.
        let (a, bb) = match seq.stop {
            Some(s) => {
                let m = para.matrix(s);
                (m[0][0], m[0][1])
            }
            None => (1.0, 0.0),
        };
        let telecentric = seq.object_space_telecentric || a.abs() < 1e-12 * bb.abs().max(1.0);
        let epl = if telecentric { None } else { Some(bb / a) };
        let f2 = para.f2();
        let finite = seq.object_distance;
        let epd = match (&seq.aperture, finite) {
            (SeqAperture::Epd { value }, _) => Some(*value),
            (SeqAperture::ImageFno { value }, None) => Some(f2 / value),
            (SeqAperture::ImageFno { .. }, Some(_)) => {
                self.reasons.push(
                    "an image-space f-number with an object at a finite distance is not built yet; \
                     give the aperture as an entrance pupil diameter or an object-space NA"
                        .into(),
                );
                None
            }
            (SeqAperture::ObjectNa { .. }, None) => {
                self.reasons.push(
                    "an object-space NA needs an object at a finite distance, and this one is at \
                     infinity"
                        .into(),
                );
                None
            }
            (SeqAperture::ObjectNa { value }, Some(d)) => {
                let u = value.clamp(-1.0, 1.0).asin();
                epl.map(|l| 2.0 * (d + l) * u.tan())
            }
        };
        let na = match (&seq.aperture, finite) {
            (SeqAperture::ObjectNa { value }, Some(_)) => Some(*value),
            (_, Some(d)) => match (epd, epl) {
                (Some(e), Some(l)) => Some((0.5 * e / (d + l)).atan().sin()),
                _ => None,
            },
            _ => None,
        };
        if let Some(e) = epd {
            if !(e.is_finite() && e > 0.0) {
                self.reasons
                    .push(format!("the entrance pupil diameter comes out as {e}"));
            }
        }

        // The image's size: Optiland's rule for a clear aperture, the
        // marginal and chief ray heights added, at the largest field.
        let field = seq.fields.iter().fold(0.0_f64, |m, f| m.max(f.abs()));
        let image_semi = seq.image_semi_aperture.unwrap_or_else(|| {
            let (ym, yc) = para.image_heights(seq, epd.unwrap_or(0.0), epl, na, field);
            ym.abs() + yc.abs()
        });
        Pupil {
            epl,
            epd: epd.unwrap_or(0.0),
            na,
            image_semi: if image_semi.is_finite() {
                image_semi
            } else {
                0.0
            },
        }
    }

    /// The source standing for one field.
    fn field_source(
        &mut self,
        seq: &SeqSystem,
        pupil: &Pupil,
        at: SourcePlace,
        opts: &ImportOptions,
        spectrum: &Spectrum,
    ) -> Result<SourceId, String> {
        let SourcePlace {
            field: f,
            beam,
            x_first,
            x_start,
        } = at;
        let unit = match seq.field_type {
            FieldType::Angle => "°",
            FieldType::ObjectHeight => " mm",
        };
        let label = format!("field {f}{unit}");
        let id = self.scene.alloc_source_id();
        let (transform, spatial, angular, samples) = match seq.object_distance {
            None => {
                if seq.field_type != FieldType::Angle {
                    return Err(format!(
                        "{label}: an object at infinity has field angles, not heights"
                    ));
                }
                if f.abs() >= 89.9 {
                    return Err(format!(
                        "{label}: a field this close to 90° never reaches the lens"
                    ));
                }
                let Some(epl) = pupil.epl else {
                    return Err("an object at infinity with its entrance pupil at infinity \
                                has no beam to build"
                        .into());
                };
                let theta = f.to_radians();
                let dir = Vec3::new(theta.cos(), theta.sin(), 0.0);
                let sampling = SourceSampling {
                    spatial_samples: opts.rays_per_field.max(1),
                    angular_samples: 1,
                    seed: 1,
                    sequence: SamplingSequence::Grid,
                };
                match beam {
                    // The beam that covers the real pupil, square to its own
                    // direction: turned so its local X runs along the field.
                    Some(b) => {
                        let finite = b.centre_mm.iter().all(|v| v.is_finite())
                            && b.half_width_mm.is_finite()
                            && b.half_height_mm.is_finite()
                            && b.half_width_mm > 0.0
                            && b.half_height_mm > 0.0;
                        if !finite {
                            return Err(format!("{label}: its aimed beam has no extent"));
                        }
                        let crossing = Vec3::new(x_first, b.centre_mm[0], b.centre_mm[1]);
                        let pos = crossing + dir * ((x_start - x_first) / theta.cos());
                        let q = glam::DQuat::from_rotation_arc(Vec3::X, dir);
                        let transform = Isometry3::new(pos, [q.x, q.y, q.z, q.w])
                            .map_err(|e| format!("{label}: {e:?}"))?;
                        let spatial = if f == 0.0 {
                            SpatialDistribution3::Disc {
                                radius_mm: b.half_width_mm.max(b.half_height_mm),
                            }
                        } else {
                            SpatialDistribution3::Rectangle {
                                half_width_mm: b.half_width_mm,
                                half_height_mm: b.half_height_mm,
                            }
                        };
                        (
                            transform,
                            spatial,
                            AngularDistribution3::Direction { local_dir: Vec3::X },
                            sampling,
                        )
                    }
                    // The paraxial pupil: its footprint on a plane square to
                    // the axis, the chief ray crossing the axis at its centre.
                    None => {
                        let x_ep = x_first + epl;
                        let y = (x_start - x_ep) * theta.tan();
                        (
                            Isometry3::from_pos(Vec3::new(x_start, y, 0.0)),
                            SpatialDistribution3::Disc {
                                radius_mm: 0.5 * pupil.epd,
                            },
                            AngularDistribution3::Direction { local_dir: dir },
                            sampling,
                        )
                    }
                }
            }
            Some(d) => {
                if !(d.is_finite() && d > 0.0) {
                    return Err(format!("the object distance is {d} mm"));
                }
                let x_obj = x_first - d;
                let telecentric = pupil.epl.is_none();
                let h = match seq.field_type {
                    FieldType::ObjectHeight => f,
                    FieldType::Angle => match pupil.epl {
                        Some(l) => -(d + l) * f.to_radians().tan(),
                        None => {
                            return Err(format!(
                                "{label}: a telecentric object space has no field angles"
                            ))
                        }
                    },
                };
                let p = Vec3::new(x_obj, h, 0.0);
                let Some(na) = pupil.na else {
                    return Err("the object-space NA could not be settled".into());
                };
                let (axis, half) = if telecentric {
                    (Vec3::X, na.clamp(-1.0, 1.0).asin())
                } else {
                    let l = pupil.epl.unwrap_or(0.0);
                    let centre = Vec3::new(x_first + l, 0.0, 0.0);
                    let axis = (centre - p).normalize();
                    // The widest angle from that line to the pupil's rim.
                    let r = 0.5 * pupil.epd;
                    let widest = (0..720)
                        .map(|k| {
                            let phi = k as f64 * std::f64::consts::TAU / 720.0;
                            let rim = centre + Vec3::new(0.0, r * phi.cos(), r * phi.sin());
                            (rim - p).normalize().dot(axis).clamp(-1.0, 1.0).acos()
                        })
                        .fold(0.0_f64, f64::max);
                    (axis, widest)
                };
                let q = glam::DQuat::from_rotation_arc(Vec3::X, axis);
                let transform = Isometry3::new(p, [q.x, q.y, q.z, q.w])
                    .map_err(|e| format!("{label}: {e:?}"))?;
                (
                    transform,
                    SpatialDistribution3::Point,
                    AngularDistribution3::UniformConeSolidAngle {
                        half_angle_rad: half,
                    },
                    SourceSampling {
                        spatial_samples: 1,
                        angular_samples: opts.rays_per_field.max(1),
                        seed: 1,
                        sequence: SamplingSequence::Grid,
                    },
                )
            }
        };
        self.scene.sources.push(Source3 {
            window: Default::default(),
            id,
            transform,
            spatial,
            angular,
            spectrum: spectrum.clone(),
            flux: 1.0,
            samples,
        });
        Ok(id)
    }
}

/// The face surface `i` makes, its vertex at local `x`.
fn face(seq: &SeqSystem, i: usize, x: f64) -> Face {
    let s = &seq.surfaces[i];
    let mut coeffs = [0.0; 8];
    for (dst, a) in coeffs.iter_mut().zip(&s.coefficients) {
        *dst = *a;
    }
    Face {
        x,
        c: s.radius
            .map_or(0.0, |r| if r.abs() > 1e-300 { 1.0 / r } else { 0.0 }),
        k: s.conic,
        coeffs,
        semi: s.semi_aperture,
    }
}

/// A body cut to one diameter, both faces out to the larger clear radius:
/// the loop [front, top rim, back, bottom rim], the layout the inspector reads
/// as a singlet. `None` when the clear radii differ by more than
/// [`ONE_DIAMETER`], a face does not reach that far, or the faces cross before
/// it.
fn one_diameter(front: Face, back: Face) -> Option<(Vec<Curve2>, f64)> {
    let h = front.semi.max(back.semi);
    if h > ONE_DIAMETER * front.semi.min(back.semi) {
        return None;
    }
    if !front.reaches(h) || !back.reaches(h) {
        return None;
    }
    let ftip = front.x + front.sag(h);
    let rtip = back.x + back.sag(h);
    if rtip - ftip <= 1e-9 * (back.x - front.x).abs().max(1.0) {
        return None;
    }
    let curves = vec![
        front.curve(h),
        Curve2::Line {
            a: Vec2::new(ftip, h),
            b: Vec2::new(rtip, h),
        },
        back.curve(h),
        Curve2::Line {
            a: Vec2::new(rtip, -h),
            b: Vec2::new(ftip, -h),
        },
    ];
    let profile = ClosedProfile2 {
        edges: curves
            .iter()
            .enumerate()
            .map(|(i, c)| ProfileEdge2 {
                patch: crate::id::PatchId(i as u64),
                curve: c.clone(),
            })
            .collect(),
    };
    validate_revolvable(&profile).ok()?;
    Some((curves, h))
}

/// A body whose faces stop at their own clear radii, joined to the rim by
/// flat steps square to the axis, as Optiland draws a lens.
fn stepped(front: Face, back: Face) -> Result<(Vec<Curve2>, f64), String> {
    for (which, f) in [("front", front), ("back", back)] {
        if !f.reaches(f.semi) {
            return Err(format!(
                "its {which} face turns back on itself before its clear radius {}",
                f.semi
            ));
        }
    }
    let h = front.semi.max(back.semi);
    let ftip = front.x + front.sag(front.semi);
    let rtip = back.x + back.sag(back.semi);
    if rtip - ftip <= 1e-9 * (back.x - front.x).abs().max(1.0) {
        return Err(format!(
            "its faces cross inside the clear aperture (the rim would be {:.6} mm long)",
            rtip - ftip
        ));
    }
    // The upper chain from the front face's rim to the back face's.
    let mut top = vec![Vec2::new(ftip, front.semi)];
    if front.semi < h {
        top.push(Vec2::new(ftip, h));
    }
    if back.semi < h {
        top.push(Vec2::new(rtip, h));
    }
    top.push(Vec2::new(rtip, back.semi));
    let mut curves = vec![front.curve(front.semi)];
    for w in top.windows(2) {
        curves.push(Curve2::Line { a: w[0], b: w[1] });
    }
    curves.push(back.curve(back.semi));
    for w in top.windows(2).rev() {
        let flip = |p: Vec2| Vec2::new(p.x, -p.y);
        curves.push(Curve2::Line {
            a: flip(w[1]),
            b: flip(w[0]),
        });
    }
    let profile = ClosedProfile2 {
        edges: curves
            .iter()
            .enumerate()
            .map(|(i, c)| ProfileEdge2 {
                patch: crate::id::PatchId(i as u64),
                curve: c.clone(),
            })
            .collect(),
    };
    validate_revolvable(&profile)?;
    Ok((curves, h))
}

/// A transform placing an object at `pos` with its local `+Z` along `normal`.
fn facing(pos: Vec3, normal: Vec3) -> Isometry3 {
    let q = glam::DQuat::from_rotation_arc(Vec3::Z, normal.normalize());
    Isometry3::new(pos, [q.x, q.y, q.z, q.w]).unwrap_or_else(|_| Isometry3::from_pos(pos))
}

/// Where one field's source goes.
#[derive(Debug, Clone, Copy)]
struct SourcePlace {
    field: f64,
    beam: Option<crate::seq::AimedBeam>,
    /// The first surface's vertex, on the axis.
    x_first: f64,
    /// The plane a collimated beam starts on.
    x_start: f64,
}

/// Where the entrance pupil is and how big, and the image's radius.
struct Pupil {
    /// From the first surface's vertex; `None` at infinity.
    epl: Option<f64>,
    epd: f64,
    /// Object-space NA, for an object at a finite distance.
    na: Option<f64>,
    image_semi: f64,
}

/// The paraxial (y, n·u) trace of a prescription, in air on the object side.
struct Paraxial<'a> {
    seq: &'a SeqSystem,
    n_after: &'a [f64],
}

type M2 = [[f64; 2]; 2];

fn mul(a: M2, b: M2) -> M2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

impl Paraxial<'_> {
    fn n_before(&self, i: usize) -> f64 {
        if i == 0 {
            1.0
        } else {
            self.n_after[i - 1]
        }
    }

    fn power(&self, i: usize) -> f64 {
        let c =
            self.seq.surfaces[i]
                .radius
                .map_or(0.0, |r| if r.abs() > 1e-300 { 1.0 / r } else { 0.0 });
        (self.n_after[i] - self.n_before(i)) * c
    }

    /// From just before the first surface to the vertex plane of surface
    /// `upto`, before it refracts.
    fn matrix(&self, upto: usize) -> M2 {
        let mut m = [[1.0, 0.0], [0.0, 1.0]];
        for i in 0..upto {
            m = mul([[1.0, 0.0], [-self.power(i), 1.0]], m);
            m = mul(
                [
                    [1.0, self.seq.surfaces[i].thickness / self.n_after[i]],
                    [0.0, 1.0],
                ],
                m,
            );
        }
        m
    }

    /// The rear focal length, `−n′/C`, as Optiland's `f2` measures it.
    fn f2(&self) -> f64 {
        let n = self.seq.surfaces.len();
        let mut m = self.matrix(n - 1);
        m = mul([[1.0, 0.0], [-self.power(n - 1), 1.0]], m);
        -self.n_after[n - 1] / m[1][0]
    }

    /// The marginal ray's and the chief ray's heights on the image plane, for
    /// the field `field` (degrees or mm, as the prescription says).
    fn image_heights(
        &self,
        seq: &SeqSystem,
        epd: f64,
        epl: Option<f64>,
        na: Option<f64>,
        field: f64,
    ) -> (f64, f64) {
        let n = seq.surfaces.len();
        let whole = self.matrix(n); // includes the last transfer, to the image
        let at = |y: f64, nu: f64| whole[0][0] * y + whole[0][1] * nu;
        match seq.object_distance {
            None => {
                let marginal = at(0.5 * epd, 0.0);
                let u = field.to_radians().tan();
                let chief = match epl {
                    Some(l) => at(-l * u, u),
                    None => 0.0,
                };
                (marginal, chief)
            }
            Some(d) => {
                let u = na.map_or(0.0, |v| v.clamp(-1.0, 1.0).asin().tan());
                let marginal = at(d * u, u);
                let h = match seq.field_type {
                    FieldType::ObjectHeight => field,
                    FieldType::Angle => epl.map_or(0.0, |l| (d + l) * field.to_radians().tan()),
                };
                let chief = match epl {
                    Some(l) => {
                        let uc = -h / (d + l);
                        at(h + d * uc, uc)
                    }
                    None => at(h, 0.0),
                };
                (marginal, chief)
            }
        }
    }
}
