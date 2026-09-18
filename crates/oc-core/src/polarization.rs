//! Polarization: Stokes states, Mueller matrices, and the frames they live in.
//!
//! The kernel computes s and p separately wherever a coating or a Fresnel
//! interface is involved, and until this module existed it threw the difference
//! away: every ray carried one number for flux, so a beam left a tilted dichroic
//! as unpolarized as it arrived. That is exactly wrong at a band edge, where a
//! dichroic at 45° can reflect 78% of s and 2% of p and hand on a beam that is
//! 95% polarized. The error does not show up at that element (for unpolarized
//! input the mean is the right flux) but at the **next** polarization-sensitive
//! one, which no longer sees unpolarized light.
//!
//! ## What a ray carries
//!
//! A [`Stokes`] vector, power-weighted: `S0` **is** the ray's flux, in whatever
//! units the source declared, and `S1..S3` are the polarized part of it. An
//! unpolarized packet is `(Φ, 0, 0, 0)`. A single Jones vector cannot represent
//! that, which is why the state here is Stokes and not a complex 2-vector.
//!
//! Alongside it a ray carries its **frame**: a unit vector `x̂` perpendicular to
//! the direction of travel. The second axis is implied, `ŷ = d̂ × x̂`, so that
//!
//! ```text
//! x̂ × ŷ = d̂
//! ```
//!
//! Between interactions the frame is constant in world space: a straight ray in
//! an isotropic medium has nothing to rotate it.
//!
//! ## Conventions, stated so they can be checked
//!
//! Fields vary as `exp(-iωt)`, so a passive medium has index `n + ik` with
//! `k ≥ 0`. For a transverse field `E = Re{(Ex, Ey) exp(-iωt)}` written in the
//! frame above,
//!
//! ```text
//! S0 = |Ex|² + |Ey|²        S1 = |Ex|² − |Ey|²
//! S2 = 2 Re(Ex* Ey)         S3 = 2 Im(Ex* Ey)
//! ```
//!
//! so `S1 > 0` is linear along `x̂`, `S2 > 0` is linear at +45° from `x̂` toward
//! `ŷ`, and `S3 > 0` is the state whose field vector turns from `x̂` toward `ŷ`
//! as time advances. Because `x̂ × ŷ = d̂`, an observer the beam is travelling
//! **toward** sees that rotation as clockwise, which is right-handed in the
//! traditional (Born and Wolf) optical sense. [`Stokes::circular`] and
//! `a_right_circular_state_turns_the_way_the_convention_says` pin that sentence
//! to arithmetic rather than to the word "right".
//!
//! ## The plane of incidence
//!
//! At an interface the natural frame is `ŝ`, perpendicular to the plane of
//! incidence, with `p̂ = d̂ × ŝ` completing it:
//!
//! ```text
//! ŝ = normalize(d̂ × n̂)        p̂ = d̂ × ŝ
//! ```
//!
//! for `n̂` the surface normal oriented against the ray. **Both** the incoming
//! and the outgoing ray use this `ŝ`; it is the same vector for reflection and
//! transmission, since the plane of incidence is. That choice fixes the sign of
//! every p coefficient in the kernel, because at normal incidence
//! `p̂_reflected = −p̂_incident`: the basis flips even though the physical field
//! does not. It is the convention in which the bare-interface coefficients are
//!
//! ```text
//! r_p = (n₂cos θᵢ − n₁cos θₜ) / (n₂cos θᵢ + n₁cos θₜ)
//! ```
//!
//! which is **minus** the one Macleod's admittance algebra produces, the algebra
//! [`crate::thinfilm`] uses. [`crate::thinfilm::coating_amp_at`] does that
//! conversion in one named place rather than leaving a stray sign for each
//! caller to rediscover.
//!
//! At exactly normal incidence the plane of incidence does not exist. The
//! degenerate case keeps the ray's own frame, which is continuous in the tilt
//! going to zero and does not depend on which azimuth the limit is taken from:
//! at normal incidence `r_s = r_p` in the physical (lab) basis, so every azimuth
//! gives the same answer once the basis flip is accounted for.

use crate::thinfilm::Cx;
use crate::vec3::Vec3;
use serde::{Deserialize, Serialize};

/// A power-weighted Stokes vector. `s0` is the ray's flux.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stokes {
    pub s0: f64,
    pub s1: f64,
    pub s2: f64,
    pub s3: f64,
}

impl Stokes {
    pub const ZERO: Self = Self {
        s0: 0.0,
        s1: 0.0,
        s2: 0.0,
        s3: 0.0,
    };

    /// Unpolarized light of the given flux: `(Φ, 0, 0, 0)`.
    pub const fn unpolarized(flux: f64) -> Self {
        Self {
            s0: flux,
            s1: 0.0,
            s2: 0.0,
            s3: 0.0,
        }
    }

    /// Fully linearly polarized at `angle_rad` from the frame's `x̂`, measured
    /// toward `ŷ`.
    pub fn linear(flux: f64, angle_rad: f64) -> Self {
        let (s, c) = (2.0 * angle_rad).sin_cos();
        Self {
            s0: flux,
            s1: flux * c,
            s2: flux * s,
            s3: 0.0,
        }
    }

    /// Fully circularly polarized. `right` is the handedness defined in the
    /// module header: the field turns from `x̂` toward `ŷ` with time.
    pub fn circular(flux: f64, right: bool) -> Self {
        Self {
            s0: flux,
            s1: 0.0,
            s2: 0.0,
            s3: if right { flux } else { -flux },
        }
    }

    /// The state of a coherent field `(Ex, Ey)` in this frame.
    pub fn from_jones(ex: Cx, ey: Cx) -> Self {
        let cross = ex.conj() * ey;
        Self {
            s0: ex.norm_sqr() + ey.norm_sqr(),
            s1: ex.norm_sqr() - ey.norm_sqr(),
            s2: 2.0 * cross.re,
            s3: 2.0 * cross.im,
        }
    }

    /// Polarized flux: `√(S1² + S2² + S3²)`, never larger than `s0` for a
    /// physical state.
    pub fn polarized_flux(&self) -> f64 {
        (self.s1 * self.s1 + self.s2 * self.s2 + self.s3 * self.s3).sqrt()
    }

    /// Degree of polarization, 0 for unpolarized and 1 for fully polarized.
    /// Zero flux has no degree of polarization; this reports 0 for it.
    pub fn dop(&self) -> f64 {
        if self.s0 <= 0.0 {
            return 0.0;
        }
        (self.polarized_flux() / self.s0).min(1.0)
    }

    /// Degree of **linear** polarization, which is what an analyzer on a
    /// rotating mount measures.
    pub fn dolp(&self) -> f64 {
        if self.s0 <= 0.0 {
            return 0.0;
        }
        ((self.s1 * self.s1 + self.s2 * self.s2).sqrt() / self.s0).min(1.0)
    }

    /// Orientation of the polarization ellipse's major axis, radians from `x̂`.
    pub fn azimuth_rad(&self) -> f64 {
        0.5 * self.s2.atan2(self.s1)
    }

    /// Whether this is a state light could actually be in: non-negative flux
    /// and a polarized part no larger than the whole.
    ///
    /// `tol` is absolute, in flux units; f64 arithmetic through a long chain of
    /// Mueller matrices lands a hair outside.
    pub fn is_realizable(&self, tol: f64) -> bool {
        self.s0 >= -tol
            && self.s0.is_finite()
            && self.s1.is_finite()
            && self.s2.is_finite()
            && self.s3.is_finite()
            && self.polarized_flux() <= self.s0 + tol
    }

    /// Scale the flux, leaving the polarization state alone.
    pub fn scaled(&self, k: f64) -> Self {
        Self {
            s0: self.s0 * k,
            s1: self.s1 * k,
            s2: self.s2 * k,
            s3: self.s3 * k,
        }
    }

    /// Add two states **that share a frame and a direction**. Incoherent
    /// addition: this is a sum of powers, and it is only meaningful for beams an
    /// instrument would actually add.
    pub fn plus(&self, o: &Self) -> Self {
        Self {
            s0: self.s0 + o.s0,
            s1: self.s1 + o.s1,
            s2: self.s2 + o.s2,
            s3: self.s3 + o.s3,
        }
    }

    /// The same state written in a frame whose `x̂` is rotated by `theta` from
    /// this one, positively about the direction of travel.
    pub fn rotated(&self, theta: f64) -> Self {
        let (s2t, c2t) = (2.0 * theta).sin_cos();
        Self {
            s0: self.s0,
            s1: c2t * self.s1 + s2t * self.s2,
            s2: -s2t * self.s1 + c2t * self.s2,
            s3: self.s3,
        }
    }

    /// The flux an ideal linear analyzer at `angle_rad` from `x̂` would pass.
    ///
    /// Malus' law when the input is fully linear, half the flux when it is
    /// unpolarized, and the right thing in between. Useful as a measurement
    /// model at a detector, where a raw Stokes sum over rays with different
    /// directions would be basis-dependent.
    pub fn through_analyzer(&self, angle_rad: f64) -> f64 {
        let (s, c) = (2.0 * angle_rad).sin_cos();
        0.5 * (self.s0 + c * self.s1 + s * self.s2)
    }
}

/// A 4×4 Mueller matrix, row-major: what an interaction does to a Stokes vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mueller(pub [[f64; 4]; 4]);

impl Mueller {
    pub const IDENTITY: Self = Self([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    /// Every element scaled: an ideal grey attenuator passing a fraction `k`.
    pub fn attenuator(k: f64) -> Self {
        let mut m = Self::IDENTITY;
        for row in m.0.iter_mut() {
            for v in row.iter_mut() {
                *v *= k;
            }
        }
        m
    }

    /// The matrix of a **diagonal** Jones element: amplitude `a` on `x̂` and `b`
    /// on `ŷ`, with no cross-coupling. Every isotropic interface and every
    /// layered coating is one of these in its own `(ŝ, p̂)` frame, and so is an
    /// ideal polarizer or retarder in its own axis frame.
    ///
    /// Derived directly from `S' = f(J S)` with `J = diag(a, b)`:
    /// the upper block mixes flux and `S1` by the two powers, and the lower
    /// block rotates `(S2, S3)` by the relative phase `arg(a* b)` while scaling
    /// it by `|a||b|`.
    pub fn from_diagonal_jones(a: Cx, b: Cx) -> Self {
        let (pa, pb) = (a.norm_sqr(), b.norm_sqr());
        let cross = a.conj() * b;
        let (u, v) = (cross.re, cross.im);
        Self([
            [0.5 * (pa + pb), 0.5 * (pa - pb), 0.0, 0.0],
            [0.5 * (pa - pb), 0.5 * (pa + pb), 0.0, 0.0],
            [0.0, 0.0, u, -v],
            [0.0, 0.0, v, u],
        ])
    }

    /// The matrix of a general Jones element `[[j00, j01], [j10, j11]]`.
    ///
    /// Built by pushing the four basis states through the coherency form
    /// `C' = J C J†`, which is shorter than writing out sixteen products and
    /// is the definition the closed forms above are checked against.
    pub fn from_jones(j: [[Cx; 2]; 2]) -> Self {
        let basis = [
            Stokes {
                s0: 1.0,
                s1: 0.0,
                s2: 0.0,
                s3: 0.0,
            },
            Stokes {
                s0: 0.0,
                s1: 1.0,
                s2: 0.0,
                s3: 0.0,
            },
            Stokes {
                s0: 0.0,
                s1: 0.0,
                s2: 1.0,
                s3: 0.0,
            },
            Stokes {
                s0: 0.0,
                s1: 0.0,
                s2: 0.0,
                s3: 1.0,
            },
        ];
        let mut m = [[0.0; 4]; 4];
        for (col, s) in basis.iter().enumerate() {
            let out = jones_on_coherency(j, *s);
            let v = [out.s0, out.s1, out.s2, out.s3];
            for (row, item) in v.iter().enumerate() {
                m[row][col] = *item;
            }
        }
        Self(m)
    }

    /// An ideal absorbing linear polarizer whose transmission axis is along the
    /// frame's `x̂`. Half of unpolarized light survives; Malus' law follows.
    pub fn linear_polarizer() -> Self {
        Self::from_diagonal_jones(Cx::real(1.0), Cx::real(0.0))
    }

    /// An ideal lossless linear retarder: `retardance_rad` of phase added to the
    /// `ŷ` (slow) component relative to `x̂` (fast).
    ///
    /// A quarter wave is `π/2`, a half wave `π`. Lossless and phase-only, so it
    /// is a model of a waveplate's polarization action, not of a real crystal:
    /// no walk-off, no ordinary/extraordinary splitting, no dispersion unless a
    /// caller supplies the retardance per wavelength.
    pub fn linear_retarder(retardance_rad: f64) -> Self {
        Self::from_diagonal_jones(Cx::real(1.0), Cx::expi(retardance_rad))
    }

    /// The Stokes rotation taking a state into a frame whose `x̂` is rotated by
    /// `theta` about the direction of travel.
    pub fn rotation(theta: f64) -> Self {
        let (s2t, c2t) = (2.0 * theta).sin_cos();
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c2t, s2t, 0.0],
            [0.0, -s2t, c2t, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// This element with its own axes turned by `theta` from the frame it is
    /// applied in: `R(−θ) M R(θ)`.
    pub fn rotated(&self, theta: f64) -> Self {
        Mueller::rotation(-theta).times(&self.times(&Mueller::rotation(theta)))
    }

    /// Matrix product, this one applied **after** `rhs`.
    pub fn times(&self, rhs: &Self) -> Self {
        let mut m = [[0.0; 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..4).map(|k| self.0[i][k] * rhs.0[k][j]).sum();
            }
        }
        Self(m)
    }

    /// Apply to a state.
    pub fn apply(&self, s: &Stokes) -> Stokes {
        let v = [s.s0, s.s1, s.s2, s.s3];
        let out: Vec<f64> = (0..4)
            .map(|i| (0..4).map(|k| self.0[i][k] * v[k]).sum())
            .collect();
        Stokes {
            s0: out[0],
            s1: out[1],
            s2: out[2],
            s3: out[3],
        }
    }

    /// Every element scaled by `k`, for the power factor that accompanies a
    /// transmitted amplitude.
    pub fn scaled(&self, k: f64) -> Self {
        let mut m = *self;
        for row in m.0.iter_mut() {
            for v in row.iter_mut() {
                *v *= k;
            }
        }
        m
    }

    /// The flux this element passes for unpolarized input: its `M00`.
    pub fn unpolarized_gain(&self) -> f64 {
        self.0[0][0]
    }
}

/// `J C J†` expressed on Stokes vectors, the definition [`Mueller::from_jones`]
/// is built from.
fn jones_on_coherency(j: [[Cx; 2]; 2], s: Stokes) -> Stokes {
    // C = [[ (S0+S1)/2, (S2 - i S3)/2 ], [ (S2 + i S3)/2, (S0-S1)/2 ]]
    let c = [
        [
            Cx::real(0.5 * (s.s0 + s.s1)),
            Cx::new(0.5 * s.s2, -0.5 * s.s3),
        ],
        [
            Cx::new(0.5 * s.s2, 0.5 * s.s3),
            Cx::real(0.5 * (s.s0 - s.s1)),
        ],
    ];
    // (J C) J†
    let mut jc = [[Cx::real(0.0); 2]; 2];
    for (i, row) in jc.iter_mut().enumerate() {
        for (k, cell) in row.iter_mut().enumerate() {
            *cell = j[i][0] * c[0][k] + j[i][1] * c[1][k];
        }
    }
    let mut out = [[Cx::real(0.0); 2]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        for (k, cell) in row.iter_mut().enumerate() {
            // (J†)[m][k] = conj(J[k][m])
            *cell = jc[i][0] * j[k][0].conj() + jc[i][1] * j[k][1].conj();
        }
    }
    Stokes {
        s0: out[0][0].re + out[1][1].re,
        s1: out[0][0].re - out[1][1].re,
        s2: 2.0 * out[0][1].re,
        s3: -2.0 * out[0][1].im,
    }
}

// ---------------------------------------------------------------------------
// Frames in space
// ---------------------------------------------------------------------------

/// A ray's polarization state together with the transverse frame it is written
/// in. `x` is a unit vector perpendicular to the ray; the other axis is
/// `d̂ × x̂`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolRay {
    pub stokes: Stokes,
    pub x: Vec3,
}

impl PolRay {
    /// Unpolarized light of `flux`, in an arbitrary but valid frame for `dir`.
    pub fn unpolarized(dir: Vec3, flux: f64) -> Self {
        Self {
            stokes: Stokes::unpolarized(flux),
            x: any_perpendicular(dir),
        }
    }

    /// The same state expressed on a different transverse axis `new_x` (which
    /// must also be perpendicular to `dir`).
    pub fn in_frame(&self, dir: Vec3, new_x: Vec3) -> Stokes {
        self.stokes.rotated(frame_angle(dir, self.x, new_x))
    }
}

/// Any unit vector perpendicular to `dir`, chosen deterministically.
///
/// Used for a source that declares no polarization: an unpolarized state looks
/// the same in every frame, so the choice cannot change a result, only the
/// numbers `S1` and `S2` would carry if the state later becomes polarized in
/// this frame. It still has to be deterministic, or two runs of the same scene
/// would disagree on those numbers.
pub fn any_perpendicular(dir: Vec3) -> Vec3 {
    let d = dir.normalize();
    // Pick the axis the direction leans on least; the cross product with it is
    // then far from degenerate.
    let a = if d.x.abs() <= d.y.abs() && d.x.abs() <= d.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if d.y.abs() <= d.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    d.cross(a).normalize()
}

/// The signed angle about `dir` from frame axis `from` to frame axis `to`, both
/// unit and perpendicular to `dir`.
///
/// Positive is the sense in which [`Stokes::rotated`] turns, so
/// `state.rotated(frame_angle(dir, old_x, new_x))` is the state written on
/// `new_x`.
pub fn frame_angle(dir: Vec3, from: Vec3, to: Vec3) -> f64 {
    let d = dir.normalize();
    let y = d.cross(from);
    to.dot(y).atan2(to.dot(from))
}

/// Carry a transverse frame across a change of direction, by the shortest
/// rotation that takes one direction onto the other.
///
/// Reflection and refraction need none of this: both outgoing rays lie in the
/// plane of incidence, so the `ŝ` axis already serves them. This is for a bend
/// with no plane of incidence to work in, which in this kernel means the ideal
/// thin lens. Rotating by the smallest angle adds no twist the element does not
/// physically apply.
pub fn transport_frame(x: Vec3, from: Vec3, to: Vec3) -> Vec3 {
    let (a, b) = (from.normalize(), to.normalize());
    let axis = a.cross(b);
    let len = axis.length();
    if len < 1e-12 {
        // Unchanged direction, or exactly reversed, where no rotation is
        // preferred over any other and the frame may as well stay put.
        return x;
    }
    let axis = axis * (1.0 / len);
    let angle = a.dot(b).clamp(-1.0, 1.0).acos();
    let (s, c) = angle.sin_cos();
    // Rodrigues.
    (x * c + axis.cross(x) * s + axis * (axis.dot(x) * (1.0 - c))).normalize()
}

/// The `ŝ` axis of the plane of incidence: perpendicular to both the ray and
/// the surface normal.
///
/// `normal` must oppose `dir` (`n̂ · d̂ < 0`), the orientation the tracer already
/// hands to every response. Returns `None` at normal incidence, where the plane
/// of incidence does not exist; the caller keeps the ray's own frame there,
/// which is the continuous limit.
pub fn incidence_axis(dir: Vec3, normal: Vec3) -> Option<Vec3> {
    let c = dir.normalize().cross(normal.normalize());
    let len = c.length();
    // sin θᵢ below this is a normal-incidence hit for any purpose here: the s/p
    // difference goes as sin²θ, so at 1e-9 rad the two coefficients agree to
    // some 1e-18, far under f64 noise in the coefficients themselves.
    if len < 1e-9 {
        return None;
    }
    Some(c * (1.0 / len))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn a_right_circular_state_turns_the_way_the_convention_says() {
        // Ex = 1, Ey = i under exp(-iωt) gives E(t) = (cos ωt, sin ωt): the
        // field turns from x̂ toward ŷ as time advances, which the module header
        // calls right-handed. The Stokes vector must agree.
        let s = Stokes::from_jones(Cx::real(1.0), Cx::new(0.0, 1.0));
        assert!(close(s.s0, 2.0));
        assert!(close(s.s3, 2.0), "S3 = {}", s.s3);
        assert!(s.s3 > 0.0);
        assert!(close(s.s1, 0.0) && close(s.s2, 0.0));

        // And the constructor agrees with the field it claims to describe.
        let c = Stokes::circular(2.0, true);
        assert!(close(c.s3, s.s3));

        // Reconstruct the field at two instants and check the turn direction in
        // the x-y plane: the cross product x̂-component-to-ŷ-component must be
        // positive going forward in time.
        let (ex, ey) = (Cx::real(1.0), Cx::new(0.0, 1.0));
        let at = |t: f64| {
            let ph = Cx::expi(-t);
            ((ex * ph).re, (ey * ph).re)
        };
        let (x0, y0) = at(0.0);
        let (x1, y1) = at(0.1);
        assert!(x0 * y1 - y0 * x1 > 0.0, "field turns x̂ → ŷ with time");
    }

    #[test]
    fn linear_states_sit_where_their_angle_says() {
        for deg in [0.0f64, 22.5, 45.0, 90.0, 135.0] {
            let a = deg.to_radians();
            let s = Stokes::linear(1.0, a);
            assert!(close(s.dop(), 1.0));
            let az = s.azimuth_rad();
            // Azimuth is defined modulo π, which is all a linear state has.
            let diff = (az - a).rem_euclid(std::f64::consts::PI);
            assert!(
                close(diff, 0.0) || close(diff, std::f64::consts::PI),
                "{deg}°: azimuth {az}"
            );
        }
    }

    #[test]
    fn malus_law_comes_out_of_the_analyzer() {
        // Fully linear input at 0°, analyzer swept: Φ cos²θ.
        let s = Stokes::linear(1.0, 0.0);
        for deg in [0.0f64, 15.0, 30.0, 45.0, 60.0, 90.0] {
            let a = deg.to_radians();
            assert!(
                close(s.through_analyzer(a), a.cos().powi(2)),
                "{deg}°: {} vs {}",
                s.through_analyzer(a),
                a.cos().powi(2)
            );
        }
        // Unpolarized input: half, at every angle.
        let u = Stokes::unpolarized(1.0);
        for deg in [0.0f64, 37.0, 90.0] {
            assert!(close(u.through_analyzer(deg.to_radians()), 0.5));
        }
    }

    #[test]
    fn one_ideal_polarizer_halves_unpolarized_light_and_two_obey_malus() {
        let pol = Mueller::linear_polarizer();
        let out = pol.apply(&Stokes::unpolarized(1.0));
        assert!(close(out.s0, 0.5), "one polarizer passes {}", out.s0);
        assert!(close(out.dop(), 1.0), "and fully polarizes what it passes");

        for deg in [0.0f64, 30.0, 45.0, 60.0, 90.0] {
            let a = deg.to_radians();
            let second = pol.rotated(a);
            let two = second.apply(&out);
            assert!(
                close(two.s0, 0.5 * a.cos().powi(2)),
                "crossed at {deg}°: {}",
                two.s0
            );
        }
        // Crossed polarizers pass nothing at all, not merely a small number.
        let crossed = pol.rotated(std::f64::consts::FRAC_PI_2).apply(&out).s0;
        assert!(crossed.abs() < 1e-15, "crossed: {crossed}");
    }

    #[test]
    fn a_quarter_wave_at_45_degrees_makes_circular_light() {
        // Linear at 45° into a quarter-wave plate with its fast axis on x̂:
        // equal projections, a quarter wave apart, which is circular.
        let qwp = Mueller::linear_retarder(std::f64::consts::FRAC_PI_2);
        let input = Stokes::linear(1.0, std::f64::consts::FRAC_PI_4);
        let out = qwp.apply(&input);
        assert!(close(out.s0, 1.0), "lossless: {}", out.s0);
        assert!(close(out.dop(), 1.0));
        assert!(close(out.s1, 0.0) && close(out.s2, 0.0));
        // Sign, not magnitude, is the part a convention can get wrong: retarding
        // ŷ by +π/2 turns Ey = 1 into e^{iπ/2} = i, which is the right-handed
        // state by the module's own definition.
        assert!(close(out.s3, 1.0), "S3 = {}", out.s3);

        // Two quarter waves make a half wave, which mirrors a linear state about
        // the fast axis: 45° goes to −45°.
        let hwp = qwp.times(&qwp);
        let flipped = hwp.apply(&input);
        assert!(close(flipped.s2, -input.s2), "S2 = {}", flipped.s2);
        assert!(close(flipped.s0, 1.0) && close(flipped.s1, 0.0));
    }

    #[test]
    fn a_half_wave_rotates_a_linear_state_by_twice_its_angle() {
        let input = Stokes::linear(1.0, 0.0);
        for deg in [10.0f64, 22.5, 45.0, 70.0] {
            let a = deg.to_radians();
            let hwp = Mueller::linear_retarder(std::f64::consts::PI).rotated(a);
            let out = hwp.apply(&input);
            let want = Stokes::linear(1.0, 2.0 * a);
            assert!(
                close(out.s1, want.s1) && close(out.s2, want.s2),
                "{deg}°: ({}, {}) vs ({}, {})",
                out.s1,
                out.s2,
                want.s1,
                want.s2
            );
        }
    }

    #[test]
    fn the_closed_form_diagonal_matrix_matches_the_general_one() {
        let cases = [
            (Cx::new(0.6, -0.2), Cx::new(-0.3, 0.5)),
            (Cx::real(1.0), Cx::real(0.0)),
            (Cx::real(1.0), Cx::expi(0.7)),
            (Cx::new(0.0, 0.9), Cx::new(0.4, 0.4)),
        ];
        for (a, b) in cases {
            let closed = Mueller::from_diagonal_jones(a, b);
            let general = Mueller::from_jones([[a, Cx::real(0.0)], [Cx::real(0.0), b]]);
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        close(closed.0[i][j], general.0[i][j]),
                        "[{i}][{j}]: {} vs {}",
                        closed.0[i][j],
                        general.0[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn rotating_the_frame_leaves_the_physics_alone() {
        let s = Stokes {
            s0: 1.0,
            s1: 0.3,
            s2: -0.5,
            s3: 0.2,
        };
        for deg in [0.0f64, 17.0, 90.0, 213.0] {
            let a = deg.to_radians();
            let r = s.rotated(a);
            assert!(close(r.s0, s.s0), "flux is frame-independent");
            assert!(close(r.s3, s.s3), "handedness is frame-independent");
            assert!(close(r.dop(), s.dop()));
            // And rotating back returns the original.
            let back = r.rotated(-a);
            assert!(close(back.s1, s.s1) && close(back.s2, s.s2));
        }
    }

    #[test]
    fn a_frame_rotation_is_what_the_matrix_says_it_is() {
        let s = Stokes {
            s0: 1.0,
            s1: 0.3,
            s2: -0.5,
            s3: 0.2,
        };
        for deg in [11.0f64, 45.0, 120.0] {
            let a = deg.to_radians();
            let m = Mueller::rotation(a).apply(&s);
            let d = s.rotated(a);
            assert!(close(m.s1, d.s1) && close(m.s2, d.s2) && close(m.s3, d.s3));
        }
    }

    #[test]
    fn frames_in_space_agree_with_frames_in_algebra() {
        let dir = Vec3::new(0.3, -0.4, 1.0).normalize();
        let x = any_perpendicular(dir);
        assert!(close(x.length(), 1.0));
        assert!(x.dot(dir).abs() < TOL, "perpendicular: {}", x.dot(dir));

        // Turn the axis by a known angle about the ray and recover it.
        let y = dir.cross(x);
        for deg in [0.0f64, 30.0, -80.0, 179.0] {
            let a = deg.to_radians();
            let turned = x * a.cos() + y * a.sin();
            let got = frame_angle(dir, x, turned);
            assert!(close(got, a), "{deg}°: recovered {}", got.to_degrees());
        }
    }

    #[test]
    fn the_incidence_axis_is_perpendicular_to_the_plane_it_names() {
        let dir = Vec3::new(1.0, 0.0, -1.0).normalize();
        let normal = Vec3::new(0.0, 0.0, 1.0); // opposes dir
        let s = incidence_axis(dir, normal).expect("45° is not normal incidence");
        assert!(s.dot(dir).abs() < TOL && s.dot(normal).abs() < TOL);
        assert!(close(s.length(), 1.0));

        // Straight in: no plane of incidence to speak of.
        assert!(incidence_axis(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0)).is_none());
    }

    #[test]
    fn realizability_catches_states_light_cannot_be_in() {
        assert!(Stokes::unpolarized(1.0).is_realizable(0.0));
        assert!(Stokes::linear(1.0, 0.3).is_realizable(0.0));
        assert!(Stokes::circular(1.0, false).is_realizable(0.0));
        // Polarized part larger than the whole.
        assert!(!Stokes {
            s0: 1.0,
            s1: 0.8,
            s2: 0.8,
            s3: 0.0
        }
        .is_realizable(1e-9));
        // Negative flux.
        assert!(!Stokes::unpolarized(-1.0).is_realizable(1e-9));
    }

    #[test]
    fn every_ideal_element_keeps_its_output_realizable() {
        let inputs = [
            Stokes::unpolarized(1.0),
            Stokes::linear(1.0, 0.4),
            Stokes::circular(1.0, true),
            Stokes {
                s0: 1.0,
                s1: 0.2,
                s2: 0.1,
                s3: -0.3,
            },
        ];
        let elements = [
            Mueller::linear_polarizer(),
            Mueller::linear_polarizer().rotated(0.6),
            Mueller::linear_retarder(std::f64::consts::FRAC_PI_2).rotated(0.9),
            Mueller::from_diagonal_jones(Cx::new(0.7, 0.1), Cx::new(0.2, -0.4)),
            Mueller::attenuator(0.3),
        ];
        for s in inputs {
            for m in elements {
                let out = m.apply(&s);
                assert!(out.is_realizable(1e-12), "{out:?} from {s:?}");
                assert!(out.s0 <= s.s0 + 1e-12, "no element here amplifies");
            }
        }
    }
}
