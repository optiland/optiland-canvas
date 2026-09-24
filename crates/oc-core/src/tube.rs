//! Tubes: a light pipe, a lens barrel, a baffle.
//!
//! A tube is three numbers, inner and outer diameter and length, and one
//! choice: what its bore does to light. The outside and the end rings are
//! black. The bore is black, an ideal mirror of one reflectance, or a real
//! metal film, and the metal is the choice that matters for a light pipe:
//! light that travels a long pipe meets the wall at grazing angles, dozens of
//! times, and a metal's reflectance at grazing incidence is not its
//! reflectance at normal incidence. One number held for every angle and every
//! wavelength is the error a light-pipe calculation exists to avoid.
//!
//! The geometry is [`SolidGeometry3::Tube`]. What this module adds is the
//! vocabulary the editor speaks: diameters rather than radii, and a bore
//! finish rather than a patch override.

use crate::geom3::primitives::tube_face;
use crate::id::{LayerId, PatchId};
use crate::material::Material;
use crate::scene3::{
    AbsorptionModel, Body3, ObjectCommon, PatchResponseOverride, Scene3, SolidGeometry3,
    SpectralScalar, SurfaceProps3, SurfaceResponse, TwoSidedSurfaceResponse,
};
use crate::thinfilm::{materials, metal_mirror, Coating, FilmMaterial};
use crate::vec3::{Isometry3, Vec3};

/// A metal a bore can be coated with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoreMetal {
    Silver,
    Aluminium,
}

impl BoreMetal {
    fn material(self) -> FilmMaterial {
        match self {
            BoreMetal::Silver => materials::silver(),
            BoreMetal::Aluminium => materials::aluminium_measured(),
        }
    }

    /// The editor's name for it, and the one [`Self::parse`] reads back.
    pub fn name(self) -> &'static str {
        match self {
            BoreMetal::Silver => "silver",
            BoreMetal::Aluminium => "aluminium",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "silver" => Some(BoreMetal::Silver),
            "aluminium" | "aluminum" => Some(BoreMetal::Aluminium),
            _ => None,
        }
    }
}

/// The half-wave silica overcoat of a protected metal mirror, at 550 nm.
///
/// Half a wave is optically absent at its design wavelength, which is why a
/// coater uses it: the film protects the metal without changing what it
/// reflects there. Away from 550 nm, and at a slant, it is no longer absent,
/// and a protected mirror reflects a little differently from a bare one. The
/// transfer matrix reports that difference; nothing here approximates it.
const OVERCOAT_NM: f64 = 550.0 / (2.0 * 1.46);

/// What a tube's bore does to light.
#[derive(Debug, Clone, PartialEq)]
pub enum BoreFinish {
    /// Absorbs, like the rest of the tube: a baffle.
    Black,
    /// Reflects `reflectivity` at every angle and wavelength. The ideal
    /// light pipe, and a useful bound, but not a real one.
    Mirror { reflectivity: f64 },
    /// A metal film, bare or under a protective overcoat, evaluated by the
    /// transfer matrix at each ray's own angle and wavelength.
    Metal { metal: BoreMetal, protected: bool },
}

impl BoreFinish {
    fn response(&self) -> Option<TwoSidedSurfaceResponse> {
        let coated = |c: Coating| TwoSidedSurfaceResponse {
            from_inside: SurfaceResponse::Coating {
                coating: c.seen_from_behind(),
            },
            from_outside: SurfaceResponse::Coating { coating: c },
        };
        match *self {
            BoreFinish::Black => None,
            BoreFinish::Mirror { reflectivity } => {
                let r = SurfaceResponse::Mirror {
                    reflectivity: SpectralScalar::Constant(reflectivity.clamp(0.0, 1.0)),
                };
                Some(TwoSidedSurfaceResponse {
                    from_outside: r.clone(),
                    from_inside: r,
                })
            }
            BoreFinish::Metal { metal, protected } => {
                let overcoat = protected.then(|| (materials::silica(), OVERCOAT_NM));
                Some(coated(metal_mirror(metal.material(), overcoat)))
            }
        }
    }

    /// Read a bore's finish back from its response, or `None` for a response
    /// this vocabulary did not write (a hand-edited coating, say), which the
    /// editor then shows as it is rather than renaming.
    fn of(response: Option<&TwoSidedSurfaceResponse>) -> Option<BoreFinish> {
        let Some(r) = response else {
            return Some(BoreFinish::Black);
        };
        match &r.from_outside {
            SurfaceResponse::Absorber => Some(BoreFinish::Black),
            SurfaceResponse::Mirror {
                reflectivity: SpectralScalar::Constant(v),
            } => Some(BoreFinish::Mirror { reflectivity: *v }),
            SurfaceResponse::Coating { coating } => {
                let metal_layer = coating.layers.last()?;
                let metal = [BoreMetal::Silver, BoreMetal::Aluminium]
                    .into_iter()
                    .find(|m| m.material() == metal_layer.material)?;
                Some(BoreFinish::Metal {
                    metal,
                    protected: coating.layers.len() > 1,
                })
            }
            _ => None,
        }
    }
}

/// A tube's dimensions as the editor shows them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TubeDims {
    pub inner_diameter_mm: f64,
    pub outer_diameter_mm: f64,
    pub length_mm: f64,
}

impl TubeDims {
    fn geometry(&self) -> Option<SolidGeometry3> {
        let (ri, ro, hl) = (
            0.5 * self.inner_diameter_mm,
            0.5 * self.outer_diameter_mm,
            0.5 * self.length_mm,
        );
        let ok = [ri, ro, hl].iter().all(|v| v.is_finite()) && ri > 0.0 && ro > ri && hl > 0.0;
        ok.then_some(SolidGeometry3::Tube {
            inner_radius_mm: ri,
            outer_radius_mm: ro,
            half_length_mm: hl,
        })
    }
}

const BORE: PatchId = PatchId(tube_face::INNER as u64);

impl Scene3 {
    /// Add a tube centred on `pos`, its axis along world `+X` turned by
    /// `tilt_deg` about `+Z`, the way a lens is placed. Black outside, with
    /// `bore` inside. `None` if the dimensions do not make a tube.
    pub fn add_tube(
        &mut self,
        pos: Vec3,
        tilt_deg: f64,
        dims: TubeDims,
        bore: BoreFinish,
    ) -> Option<u64> {
        let geometry = dims.geometry()?;
        if !pos.is_finite() {
            return None;
        }
        let id = self.alloc_object_id();
        let mut common = ObjectCommon::new(id, LayerId(0));
        common.transform = Isometry3::from_pos(pos);
        let black = TwoSidedSurfaceResponse {
            from_outside: SurfaceResponse::Absorber,
            from_inside: SurfaceResponse::Absorber,
        };
        self.bodies.push(Body3 {
            common,
            geometry,
            material: Material::Absorber,
            absorption: AbsorptionModel::None,
            surface: SurfaceProps3 {
                default: black,
                overrides: Vec::new(),
            },
            priority: 0,
        });
        self.set_tube_bore(id.0, bore);
        if tilt_deg != 0.0 {
            self.set_object_rotation_z_deg(id.0, tilt_deg);
        }
        Some(id.0)
    }

    fn tube_mut(&mut self, id: u64) -> Option<&mut Body3> {
        self.bodies
            .iter_mut()
            .find(|b| b.common.id.0 == id && matches!(b.geometry, SolidGeometry3::Tube { .. }))
    }

    /// Resize a tube in place, keeping its placement and bore. False for an
    /// unknown id, a body that is not a tube, or dimensions that make none.
    pub fn set_tube_dims(&mut self, id: u64, dims: TubeDims) -> bool {
        let Some(geometry) = dims.geometry() else {
            return false;
        };
        match self.tube_mut(id) {
            Some(b) => {
                b.geometry = geometry;
                true
            }
            None => false,
        }
    }

    /// Refinish a tube's bore. False for an unknown id or a body that is not a
    /// tube.
    pub fn set_tube_bore(&mut self, id: u64, bore: BoreFinish) -> bool {
        let Some(b) = self.tube_mut(id) else {
            return false;
        };
        b.surface.overrides.retain(|o| o.patch != BORE);
        if let Some(response) = bore.response() {
            b.surface.overrides.push(PatchResponseOverride {
                patch: BORE,
                response,
            });
        }
        true
    }

    /// A tube's dimensions and bore finish, `None` if `id` is not a tube. The
    /// finish is `None` inside when the bore carries a response this module
    /// did not write.
    pub fn tube_info(&self, id: u64) -> Option<(TubeDims, Option<BoreFinish>)> {
        let b = self.bodies.iter().find(|b| b.common.id.0 == id)?;
        let SolidGeometry3::Tube {
            inner_radius_mm,
            outer_radius_mm,
            half_length_mm,
        } = b.geometry
        else {
            return None;
        };
        let dims = TubeDims {
            inner_diameter_mm: 2.0 * inner_radius_mm,
            outer_diameter_mm: 2.0 * outer_radius_mm,
            length_mm: 2.0 * half_length_mm,
        };
        let bore = b
            .surface
            .overrides
            .iter()
            .find(|o| o.patch == BORE)
            .map(|o| &o.response);
        Some((dims, BoreFinish::of(bore)))
    }
}
