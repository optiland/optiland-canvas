//! Direct Rayleigh–Sommerfeld integration: the slow reference.
//!
//! The first Rayleigh–Sommerfeld solution, in the `exp(−iωt)` convention,
//!
//! ```text
//! U(x, y, z) = ∬ U(ξ, η, 0) · (1/2π) (z/r) (1/r − ik) e^{ikr}/r dξ dη,
//! r = √((x−ξ)² + (y−η)² + z²),
//! ```
//!
//! evaluated as a linear convolution by FFT on a grid of at least `2n − 1`
//! samples with Simpson weights on the source, after F. Shen and A. Wang,
//! "Fast-Fourier-transform based numerical integration method for the
//! Rayleigh–Sommerfeld diffraction formula," *Appl. Opt.* 45(6), 1102–1110
//! (2006), doi:10.1364/AO.45.001102. Simpson's rule needs an odd sample count;
//! on an even axis the trapezoidal rule is used and the report says so.
//!
//! The kernel makes no approximation in angle, so this checks the angular
//! spectrum method from a different direction: one samples the impulse
//! response, the other the transfer function. Agreement is verification of
//! the code, not new evidence about the physics.

use crate::fft::next_fast_len;
use crate::propagate::asm::advance;
use crate::report::{Method, StepReport};
use crate::{Cx, Fft2, Field2, WaveError, WaveReport};
use std::f64::consts::PI;

fn weights(n: usize) -> (Vec<f64>, &'static str) {
    if n == 1 {
        return (vec![1.0], "single sample");
    }
    if n % 2 == 1 {
        let w = (0..n)
            .map(|i| {
                if i == 0 || i == n - 1 {
                    1.0 / 3.0
                } else if i % 2 == 1 {
                    4.0 / 3.0
                } else {
                    2.0 / 3.0
                }
            })
            .collect();
        (w, "Simpson")
    } else {
        let w = (0..n)
            .map(|i| if i == 0 || i == n - 1 { 0.5 } else { 1.0 })
            .collect();
        (w, "trapezoidal")
    }
}

pub(super) fn run(
    u: &Field2,
    z: f64,
    fft: &mut Fft2,
    report: &mut WaveReport,
) -> Result<Field2, WaveError> {
    if z <= 0.0 {
        return Err(WaveError::OutOfRange(
            "direct Rayleigh–Sommerfeld integration needs z > 0".into(),
        ));
    }
    let mut src = u.clone();
    src.bake_carrier();
    let [nx, ny] = u.grid.n;
    // The kernel is sampled out to the largest source–observation offset,
    // `L ≈ (n − 1)Δx`. Its local frequency there is `L / (λ√(z² + L²))`
    // (the sine of the angle over λ), and it must stay below Nyquist,
    // `1/(2Δx)`: the impulse-response sampling condition of Voelz &
    // Roggemann (2009), extended by them to Rayleigh–Sommerfeld, for a
    // linear convolution.
    {
        let lam = u.wavelength_in_medium_mm();
        for axis in 0..2 {
            let dx = u.grid.pitch_mm[axis];
            let l = (u.grid.n[axis].max(2) - 1) as f64 * dx;
            let local = l / (lam * (z * z + l * l).sqrt());
            let step = 2.0 * PI * local * dx;
            if step > PI {
                report.warn(crate::report::SamplingWarning::Undersampled {
                    what: format!(
                        "Rayleigh–Sommerfeld kernel along {} at the largest offset (needs Δx ≤ {:.4e} mm)",
                        if axis == 0 { "x" } else { "y" },
                        lam * (z * z + l * l).sqrt() / (2.0 * l)
                    ),
                    max_phase_step_rad: step,
                });
            }
        }
    }
    let q = [next_fast_len(2 * nx - 1), next_fast_len(2 * ny - 1)];
    let (wx, rule_x) = weights(nx);
    let (wy, rule_y) = weights(ny);
    let [dx, dy] = u.grid.pitch_mm;
    let k = u.k_mm();

    let mut a = vec![Cx::ZERO; q[0] * q[1]];
    for j in 0..ny {
        for i in 0..nx {
            a[j * q[0] + i] = src.at(i, j) * (wx[i] * wy[j]);
        }
    }
    let mut h = vec![Cx::ZERO; q[0] * q[1]];
    let kernel = |x: f64, y: f64| {
        let r2 = x * x + y * y + z * z;
        let r = r2.sqrt();
        Cx::new(1.0 / r, -k) * Cx::expi(k * r) * (z / (2.0 * PI * r2))
    };
    for mj in -(ny as isize - 1)..=(ny as isize - 1) {
        let jj = mj.rem_euclid(q[1] as isize) as usize;
        for mi in -(nx as isize - 1)..=(nx as isize - 1) {
            let ii = mi.rem_euclid(q[0] as isize) as usize;
            h[jj * q[0] + ii] = kernel(mi as f64 * dx, mj as f64 * dy);
        }
    }
    fft.forward(&mut a, q, 1.0);
    fft.forward(&mut h, q, 1.0);
    for (x, y) in a.iter_mut().zip(h.iter()) {
        *x *= *y;
    }
    fft.inverse(&mut a, q, dx * dy / (q[0] * q[1]) as f64);

    let mut out = src;
    for j in 0..ny {
        for i in 0..nx {
            out.data[j * nx + i] = a[j * q[0] + i];
        }
    }
    out.frame = advance(u, z);
    out.opl_ref_mm = u.opl_ref_mm + u.n_medium * z;
    // The output window is the input window. This method cannot tell power
    // that diffracted out of it from quadrature error, so it books neither:
    // the ledger's close puts the whole gap under `numerical`, and the note
    // says why.
    report.steps.push(StepReport {
        method: Method::RsDirect,
        z_mm: z,
        sampling: None,
        note: format!(
            "direct Rayleigh–Sommerfeld, {rule_x}×{rule_y} weights, convolution grid {}×{}; \
             power leaving the window and quadrature error are not separated (both booked as numerical)",
            q[0], q[1]
        ),
    });
    Ok(out)
}
