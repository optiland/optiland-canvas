//! Complex numbers, kept in the kernel and free of features.
//!
//! The thin-film transfer matrix needed a complex type first, and
//! `thinfilm.rs` used to hold it. Scalar wave optics (`oc-wave`) needs the
//! same type for every sample of a field, and polarization uses it for Jones
//! elements. So it lives here, outside the `scene3` gate, and `thinfilm`
//! re-exports it under its old path. Every method the transfer matrix used
//! is unchanged here, so a coating computes the same bits it always did
//! (acceptance case C33).
//!
//! The additions for wave optics are `exp`, `from_polar`, the assigning
//! operators, `Sum`, and a real scalar on the left. They are new functions;
//! no existing one changed.

/// A complex number. Enough arithmetic for the transfer matrix and the wave
/// kernels, and no more; a dependency for this would be a handful of functions
/// of value.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cx {
    pub re: f64,
    pub im: f64,
}

impl Cx {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);
    pub const I: Self = Self::new(0.0, 1.0);

    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    /// `|z|²`, which is what the power coefficients actually want.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.norm_sqr().sqrt()
    }
    /// The principal square root: the branch with `re ≥ 0`, and `+i` on the
    /// negative real axis. This is what `numpy.sqrt` returns for a complex
    /// argument, and matching it is what keeps an evanescent layer on the same
    /// branch as the reference implementation.
    pub fn sqrt(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let m = self.abs();
        // Near the real axis, but off it, one of the two half-angle formulas
        // below subtracts two nearly equal numbers: for a film with k = 6e-4
        // that lost 6e-9 in T (the `dispersive_catalogue_stack` fixture, against
        // `tmm`, Optiland and the 40-digit oracle). There the small part is
        // taken from the large one instead, `im / (2 re)`. Only there, so
        // every result that was well conditioned keeps its bits.
        if self.im != 0.0 && self.im.abs() < 1e-2 * self.re.abs() {
            let big = ((m + self.re.abs()) * 0.5).sqrt();
            let small = self.im.abs() / (2.0 * big);
            return if self.re > 0.0 {
                Self::new(big, if self.im < 0.0 { -small } else { small })
            } else {
                Self::new(small, if self.im < 0.0 { -big } else { big })
            };
        }
        let re = ((m + self.re) * 0.5).max(0.0).sqrt();
        let im = ((m - self.re) * 0.5).max(0.0).sqrt();
        // `im == 0` on the positive real axis; on the negative real axis the
        // real part vanishes instead and the sign below picks `+i`.
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }
    pub fn cos(self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }
    pub fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    /// `e^{iθ}`: a unit phasor, which is what a retardance is.
    pub fn expi(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self::new(c, s)
    }
    /// The phase, in radians, on `(-π, π]`.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
    /// `e^z = e^re (cos im + i sin im)`.
    pub fn exp(self) -> Self {
        Self::expi(self.im) * self.re.exp()
    }
    /// `r e^{iθ}`.
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::expi(theta) * r
    }
}

impl std::ops::Neg for Cx {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl std::ops::Add for Cx {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Cx {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Cx {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl std::ops::Mul<f64> for Cx {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        Self::new(self.re * k, self.im * k)
    }
}

impl std::ops::Mul<Cx> for f64 {
    type Output = Cx;
    fn mul(self, z: Cx) -> Cx {
        z * self
    }
}

impl std::ops::Div for Cx {
    type Output = Self;
    /// Division, with a division by zero going to infinity rather than `NaN` so
    /// a degenerate admittance shows up as total reflection instead of poisoning
    /// the whole trace.
    fn div(self, o: Self) -> Self {
        let d = o.norm_sqr();
        if d == 0.0 {
            return Self::new(f64::INFINITY, f64::INFINITY);
        }
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}

impl std::ops::Div<f64> for Cx {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        Self::new(self.re / k, self.im / k)
    }
}

impl std::ops::AddAssign for Cx {
    fn add_assign(&mut self, o: Self) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl std::ops::SubAssign for Cx {
    fn sub_assign(&mut self, o: Self) {
        self.re -= o.re;
        self.im -= o.im;
    }
}

impl std::ops::MulAssign for Cx {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl std::ops::MulAssign<f64> for Cx {
    fn mul_assign(&mut self, k: f64) {
        self.re *= k;
        self.im *= k;
    }
}

impl std::iter::Sum for Cx {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::Cx;
    use std::f64::consts::PI;

    #[test]
    fn cx_exp_obeys_euler() {
        // e^{a+ib} = e^a (cos b + i sin b), and e^{iπ} = −1.
        for &(a, b) in &[(0.0, PI), (0.3, -1.2), (-2.0, 7.5), (1.0, 0.0)] {
            let z = Cx::new(a, b).exp();
            let want = Cx::new(a.exp() * b.cos(), a.exp() * b.sin());
            assert!((z - want).abs() <= 1e-15 * want.abs().max(1.0), "{z:?} vs {want:?}");
        }
        let e = Cx::new(0.0, PI).exp() + Cx::ONE;
        assert!(e.abs() < 1e-15);
    }

    #[test]
    fn from_polar_round_trips_through_abs_and_arg() {
        let z = Cx::from_polar(2.5, -0.7);
        assert!((z.abs() - 2.5).abs() < 1e-15);
        assert!((z.arg() + 0.7).abs() < 1e-15);
    }

    #[test]
    fn the_assigning_operators_match_the_plain_ones() {
        let a = Cx::new(1.5, -2.0);
        let b = Cx::new(-0.25, 3.0);
        let mut c = a;
        c += b;
        assert_eq!(c, a + b);
        let mut c = a;
        c -= b;
        assert_eq!(c, a - b);
        let mut c = a;
        c *= b;
        assert_eq!(c, a * b);
        let mut c = a;
        c *= 3.0;
        assert_eq!(c, a * 3.0);
        assert_eq!(3.0 * a, a * 3.0);
        assert_eq!([a, b].into_iter().sum::<Cx>(), a + b);
        assert_eq!(a / 2.0, Cx::new(0.75, -1.0));
    }
}
