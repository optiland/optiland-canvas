//! A complex scalar field sampled on a plane.

use crate::{Cx, Grid2, WaveError};
use oc_core::Isometry3;
use std::f64::consts::PI;

/// `U(x, y)` on a [`Grid2`], at one vacuum wavelength, in one medium.
///
/// The true field is `data(x, y) · exp(ik(α₀x + β₀y))`, where `(α₀, β₀)` is
/// [`Field2::carrier`]: a plane-wave tilt held analytically instead of in the
/// samples. A beam crossing its own plane at 24° would otherwise need a phase
/// ramp of many radians per sample. Propagators account for the carrier;
/// everything that reads `data` directly sees the field with the tilt removed.
///
/// `frame` places the plane in the world: its local `+Z` is the normal the
/// field propagates along, and the samples lie over its local `X` and `Y`. That
/// is the surface convention `oc_core::grating` already uses.
#[derive(Debug, Clone, PartialEq)]
pub struct Field2 {
    pub grid: Grid2,
    /// Row-major, `x` fastest. `|U|²` is flux per mm².
    pub data: Vec<Cx>,
    /// Vacuum wavelength, µm.
    pub lambda_um: f64,
    /// Refractive index of the homogeneous medium the field propagates in.
    pub n_medium: f64,
    /// Direction cosines `(α₀, β₀)` of the analytic carrier, in the plane's
    /// local frame.
    pub carrier: [f64; 2],
    /// The plane in the world.
    pub frame: Isometry3,
    /// Optical path length at the frame origin, mm. It ties the field's phase
    /// to the optical path a ray reports, so a field and a trace can be
    /// compared on the same clock.
    pub opl_ref_mm: f64,
}

impl Field2 {
    /// A zero field.
    pub fn zeros(grid: Grid2, lambda_um: f64) -> Result<Self, WaveError> {
        grid.validate()?;
        check_wavelength(lambda_um)?;
        Ok(Self {
            grid,
            data: vec![Cx::ZERO; grid.len()],
            lambda_um,
            n_medium: 1.0,
            carrier: [0.0, 0.0],
            frame: Isometry3::IDENTITY,
            opl_ref_mm: 0.0,
        })
    }

    /// A field whose sample at `(x, y)` is `f(x, y)`, coordinates in mm.
    pub fn from_fn(
        grid: Grid2,
        lambda_um: f64,
        f: impl Fn(f64, f64) -> Cx,
    ) -> Result<Self, WaveError> {
        let mut u = Self::zeros(grid, lambda_um)?;
        for j in 0..grid.n[1] {
            for i in 0..grid.n[0] {
                let (x, y) = grid.xy(i, j);
                u.data[grid.idx(i, j)] = f(x, y);
            }
        }
        if !u.data.iter().all(|z| z.is_finite()) {
            return Err(WaveError::InvalidField(
                "the field has a non-finite sample".into(),
            ));
        }
        Ok(u)
    }

    /// The same field in another medium.
    pub fn in_medium(mut self, n: f64) -> Result<Self, WaveError> {
        if !(n.is_finite() && n > 0.0) {
            return Err(WaveError::InvalidField(format!(
                "medium index {n} is not a positive finite number"
            )));
        }
        self.n_medium = n;
        Ok(self)
    }

    /// Vacuum wavelength in mm: the one place µm become mm.
    #[inline]
    pub fn wavelength_mm(&self) -> f64 {
        self.lambda_um / 1000.0
    }

    /// Wavelength in the medium, mm.
    #[inline]
    pub fn wavelength_in_medium_mm(&self) -> f64 {
        self.wavelength_mm() / self.n_medium
    }

    /// Wavenumber in the medium, `2πn/λ`, rad/mm.
    #[inline]
    pub fn k_mm(&self) -> f64 {
        2.0 * PI / self.wavelength_in_medium_mm()
    }

    /// `Σ|U|²ΔA`: the flux the field carries.
    pub fn power(&self) -> f64 {
        self.data.iter().map(|z| z.norm_sqr()).sum::<f64>() * self.grid.cell_area_mm2()
    }

    /// The sample at `(i, j)`.
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> Cx {
        self.data[self.grid.idx(i, j)]
    }

    /// `|U|²` per sample, flux per mm².
    pub fn irradiance(&self) -> Vec<f64> {
        self.data.iter().map(|z| z.norm_sqr()).collect()
    }

    /// Fold the analytic carrier into the samples, leaving `carrier = 0`.
    ///
    /// Only sensible when the tilt is well sampled; a caller that needs it for
    /// a large tilt should not be holding the field on this grid.
    pub fn bake_carrier(&mut self) {
        if self.carrier == [0.0, 0.0] {
            return;
        }
        let k = self.k_mm();
        let [a, b] = self.carrier;
        for j in 0..self.grid.n[1] {
            for i in 0..self.grid.n[0] {
                let (x, y) = self.grid.xy(i, j);
                let idx = self.grid.idx(i, j);
                self.data[idx] *= Cx::expi(k * (a * x + b * y));
            }
        }
        self.carrier = [0.0, 0.0];
    }

    /// Do two fields share a grid, wavelength and medium? Element-wise
    /// operations require it.
    pub fn same_sampling(&self, other: &Field2) -> bool {
        self.grid == other.grid
            && self.lambda_um == other.lambda_um
            && self.n_medium == other.n_medium
    }
}

fn check_wavelength(lambda_um: f64) -> Result<(), WaveError> {
    if lambda_um.is_finite() && lambda_um > 0.0 {
        Ok(())
    } else {
        Err(WaveError::InvalidField(format!(
            "wavelength {lambda_um} µm is not a positive finite number"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_is_the_irradiance_summed_over_the_cell_area() {
        let g = Grid2::new([4, 2], [0.5, 0.25], [0.0, 0.0]).unwrap();
        let u = Field2::from_fn(g, 0.5, |_, _| Cx::new(3.0, 4.0)).unwrap();
        // |U|² = 25 on 8 samples of 0.125 mm².
        assert!((u.power() - 25.0 * 8.0 * 0.125).abs() < 1e-12);
    }

    #[test]
    fn the_wavelength_crosses_into_millimetres_once() {
        let g = Grid2::square(2, 1.0).unwrap();
        let u = Field2::zeros(g, 0.5).unwrap().in_medium(1.5).unwrap();
        assert_eq!(u.wavelength_mm(), 0.0005);
        assert!((u.k_mm() - 2.0 * PI * 1.5 / 0.0005).abs() < 1e-9);
    }

    #[test]
    fn a_baked_carrier_is_a_phase_ramp() {
        let g = Grid2::square(8, 0.001).unwrap();
        let mut u = Field2::from_fn(g, 0.5, |_, _| Cx::ONE).unwrap();
        u.carrier = [0.1, 0.0];
        u.bake_carrier();
        let k = u.k_mm();
        let (x, _) = g.xy(7, 0);
        assert!((u.at(7, 0) - Cx::expi(k * 0.1 * x)).abs() < 1e-12);
        assert_eq!(u.carrier, [0.0, 0.0]);
    }

    #[test]
    fn bad_wavelengths_and_media_are_refused() {
        let g = Grid2::square(2, 1.0).unwrap();
        assert!(Field2::zeros(g, 0.0).is_err());
        assert!(Field2::zeros(g, f64::NAN).is_err());
        assert!(Field2::zeros(g, 0.5).unwrap().in_medium(-1.0).is_err());
        assert!(Field2::from_fn(g, 0.5, |_, _| Cx::new(f64::NAN, 0.0)).is_err());
    }
}
