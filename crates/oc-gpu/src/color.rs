//! Wavelength → sRGB for spectral rendering (§5.5: "dispersion looks like
//! dispersion").
//!
//! Uses the analytic multi-lobe Gaussian fit of the CIE 1931 colour-matching
//! functions from Wyman, Sloan & Shirley (2013), then XYZ → linear sRGB (D65)
//! → gamma. Out-of-gamut results are clamped and rescaled to preserve hue.

/// One piecewise Gaussian lobe: sharper on one side of the mean.
fn lobe(x: f32, mu: f32, s1: f32, s2: f32) -> f32 {
    let s = if x < mu { s1 } else { s2 };
    let t = (x - mu) * s;
    (-0.5 * t * t).exp()
}

/// CIE 1931 colour-matching functions at wavelength `nm`.
fn cie_xyz(nm: f32) -> [f32; 3] {
    let x = 1.056 * lobe(nm, 599.8, 0.0264, 0.0323) + 0.362 * lobe(nm, 442.0, 0.0624, 0.0374)
        - 0.065 * lobe(nm, 501.1, 0.0490, 0.0382);
    let y = 0.821 * lobe(nm, 568.8, 0.0213, 0.0247) + 0.286 * lobe(nm, 530.9, 0.0613, 0.0322);
    let z = 1.217 * lobe(nm, 437.0, 0.0845, 0.0278) + 0.681 * lobe(nm, 459.0, 0.0385, 0.0725);
    [x, y, z]
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Raw (unbalanced) linear sRGB of a single wavelength (nm), negatives clamped.
fn raw_linear(nm: f32) -> [f32; 3] {
    let [x, y, z] = cie_xyz(nm);
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    [r.max(0.0), g.max(0.0), b.max(0.0)]
}

/// Per-channel white balance so that a full equal-energy spectrum integrates to
/// neutral. An equal-energy spectrum is Illuminant E, which is warm on a D65
/// display; balancing to it is the standard "white light looks white" choice.
fn white_balance() -> [f32; 3] {
    use std::sync::OnceLock;
    static WB: OnceLock<[f32; 3]> = OnceLock::new();
    *WB.get_or_init(|| {
        let mut s = [0.0f32; 3];
        let mut nm = 380.0;
        while nm <= 780.0 {
            let c = raw_linear(nm);
            for k in 0..3 {
                s[k] += c[k];
            }
            nm += 1.0;
        }
        let mean = (s[0] + s[1] + s[2]) / 3.0;
        [mean / s[0], mean / s[1], mean / s[2]]
    })
}

/// Map a wavelength in micrometres to a **linear**, white-balanced sRGB triple
/// (photometric, negatives clamped). This is the form to **accumulate
/// additively**: summing a full equal-energy spectrum yields neutral white, and
/// each wavelength keeps its physical relative luminance (green brightest).
/// Tonemap the accumulated result before display.
pub fn wavelength_um_to_linear(um: f32) -> [f32; 3] {
    let nm = (um * 1000.0).clamp(380.0, 780.0);
    let c = raw_linear(nm);
    let wb = white_balance();
    [c[0] * wb[0], c[1] * wb[1], c[2] * wb[2]]
}

/// Map a wavelength in **micrometres** to a display sRGB triple in `[0, 1]`.
pub fn wavelength_um_to_srgb(um: f32) -> [f32; 3] {
    let nm = (um * 1000.0).clamp(380.0, 780.0);
    let [x, y, z] = cie_xyz(nm);

    // XYZ → linear sRGB (D65).
    let mut r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let mut g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let mut b = 0.0557 * x - 0.2040 * y + 1.0570 * z;

    // Clamp negatives, then rescale so the brightest channel is 1 (keep hue).
    r = r.max(0.0);
    g = g.max(0.0);
    b = b.max(0.0);
    let m = r.max(g).max(b);
    if m > 0.0 {
        r /= m;
        g /= m;
        b /= m;
    }
    [linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dominant(rgb: [f32; 3]) -> usize {
        let mut best = 0;
        for i in 1..3 {
            if rgb[i] > rgb[best] {
                best = i;
            }
        }
        best
    }

    #[test]
    fn primaries_land_in_the_right_channel() {
        // ~450 nm blue, ~530 nm green, ~640 nm red.
        assert_eq!(dominant(wavelength_um_to_srgb(0.450)), 2, "blue");
        assert_eq!(dominant(wavelength_um_to_srgb(0.530)), 1, "green");
        assert_eq!(dominant(wavelength_um_to_srgb(0.640)), 0, "red");
    }

    #[test]
    fn equal_energy_spectrum_sums_neutral() {
        // Additively summing a dense equal-energy spectrum in LINEAR space must
        // land near neutral (R≈G≈B) — that is what makes white light white.
        let mut s = [0.0f32; 3];
        let n = 300;
        for i in 0..n {
            let um = 0.40 + 0.30 * i as f32 / (n - 1) as f32; // 400–700 nm
            let c = wavelength_um_to_linear(um);
            for k in 0..3 {
                s[k] += c[k];
            }
        }
        let mean = (s[0] + s[1] + s[2]) / 3.0;
        for k in 0..3 {
            let dev = (s[k] - mean).abs() / mean;
            assert!(
                dev < 0.15,
                "channel {k} off neutral by {dev:.3} (sum {s:?})"
            );
        }
    }

    #[test]
    fn output_is_bounded() {
        for i in 0..=40 {
            let um = 0.38 + 0.4 * i as f32 / 40.0;
            for c in wavelength_um_to_srgb(um) {
                assert!((0.0..=1.0).contains(&c), "channel {c} out of range at {um}");
            }
        }
    }
}
