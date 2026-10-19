# Phase-only LCoS spatial light modulators (SLM)

A liquid-crystal-on-silicon SLM is a pixelated phase screen. Each pixel
retards the light polarized along the liquid crystal's director by a phase set
by its grey level; the orthogonal polarization is not modulated [1], [2].

The model is implemented in the kernel and from Python, and validated against
closed-form and published results. An SLM is a scene element: the ray tracer
and the preview trace it in the geometric limit, and the wave path carries its
diffraction ([in the scene](#in-the-scene)). In the app, ＋ SLM adds one and
the component view edits it.

## The model

**Pixels:**
- Square electrodes of side `ρ·p` on pitch `p`.
- Grey level `g` maps to phase `φ(g)` through a table at the design
  wavelength `λ₀`. `PhaseLut::Linear` spreads `levels` steps over one wave, so
  the top level is one step short of 2π, as on a calibrated 8-bit SLM.
- At another wavelength the phase scales as `λ₀/λ`. This ignores the
  dispersion of the liquid crystal's birefringence and is reported as
  `PhaseDispersionApprox`.

**Dead space.** The gaps between electrodes reflect with their own complex
amplitude `g`, unmodulated. This is a source of the zero order [3].

**Polarization.** The field is scalar. `lcos_split` sends `cos(ψ − δ)` of an
incident linear polarization's amplitude, at angle `ψ` against director `δ`,
to the modulated beam and `sin(ψ − δ)` to the unmodulated one [1]. The two are
orthogonally polarized: propagate them separately and add their intensities.

**Orders, in closed form** (`unit_cell_orders`, for a periodic cell of
`Pₓ × Pᵧ` pixels):

```text
c_mn = g·δ_m0 δ_n0 + (ρ²/(PₓPᵧ)) sinc(ρm/Pₓ) sinc(ρn/Pᵧ) Σ_ij (√R e^{iφ_ij} − g) e^{−i2π(mi/Pₓ + nj/Pᵧ)}
```

- With no dead space this gives the classic N-level staircase efficiency
  `sinc²(1/N)` [4].
- With dead space the zero order holds `|g|²(1 − ρ²)²`, and the rows diffract
  at `λ/p` too.

**Sampled near field** (`lcos_reflect`). Each sample takes the area-weighted
mean of the reflectance over its cell. With `fringing_um = w > 0`, the
electrode phase at a sample is the pixel phases averaged over a `w × w` window:
a step between pixels becomes a linear ramp of width `w`, and a 2π reset gets
a flyback. This is the linear broadening of the fringing-field model [5], [6].
It has no closed form, so it is sampled only.

**Far field of a whole device, factorized** (`lcos_far_field`). This is the
electrode transform times the lattice sum, plus the dead space over the
device. It is exact within the thin-element model and refuses fringing.

**Patterns** (`PatternGenerator`):
- blazed gratings;
- Fresnel lenses;
- spiral phases;
- sums of these, wrapped;
- any explicit grey map.

`oc_wave::design::gerchberg_saxton` designs a phase between a source amplitude
and a far-field target [7].

## In the scene

`Scene3::add_modulator(pos, tilt_deg, device)` places a device on a standalone
rectangle the size of its pixel array, its rows along world `Z`. The front is
the device (`SurfaceResponse::Modulator`); the back absorbs.

**The ray tracer traces the geometric limit.** On an electrode the ray's state
splits on the director [1]. The component along it is steered by the local
phase gradient, its tangential direction cosines changing by `(λ/2π)∇φ` (the
grating equation for a phase profile [11]), and gains the optical path
`φλ/2π`. The orthogonal component reflects specularly; unpolarized light splits
evenly. The gradient is a central difference of the displayed phases, wrapped
into `(−π, π]`, one-sided at the array's edge. Away from the design wavelength
the phase scales as `λ₀/λ`, and the report says so. The dead space reflects
specularly with `|g|²`. Diffraction orders are not traced: the report lists
`PhaseModulator` as approximated ("geometric limit").

**Or it traces the diffraction orders,** as for a DMD ([dmd.md](dmd.md#in-the-scene)):
each hit spawns the brightest orders of the local unit cell, the components
along and across the director each with their own table (the dead space in
both), the rest booked as untraced orders. A device with fringing has no
closed-form orders here and is traced in the geometric limit, which the report
names.

**The preview draws the geometric limit, pixel for pixel,** in f32 and unpolarized, so
each electrode sends half the flux each way. Each pixel's phase travels to the
GPU at four bytes a pixel, up to 2²² pixels; a uniform pattern needs none. A
larger pattern is named on the preview's badge and traced by the reference
alone.

**On the wave path** the field follows the arm the chief ray took: the
modulated one where it was steered, the unmodulated one where it reflected
([wave path](wave-optics.md#the-wave-path)). The path carries unpolarized
light, so the arm takes half of what the device reflects and the other half
is booked as the family not followed; the two never interfere. The report
names that approximation.

**In a file** the device is part of the version 3 document. Written as version
2 it becomes an absorber of the same footprint, and the loss is named
([format](ocanvas-format-v3.md#spatial-light-modulators)).

**In the app** the tools dock's ＋ SLM adds the generic device of
`modulator_ui.rs`, every parameter in plain view (8 µm pixels, a linear 8-bit
table spanning a wave at 633 nm, fill 0.9, dead-space amplitude 0.5,
reflectance 0.9); no vendor's numbers ship. The editor sets the array, the
optics and the pattern: flat, a blazed grating, a Fresnel lens, a vortex, or
an image of your own mapped onto the grey levels. It draws the phase beside
the modulated beam's orders.

| Test | Checks | Gate | Measured |
|---|---|---|---|
| `an_slm_steers_the_director_component_by_its_phase_gradient` (`oc-trace/tests/modulator3.rs`) | a blazed grating of period `Λ` steers by `λ/Λ`; the orthogonal half reflects; unpolarized light splits 1:1 | 1e-12 | passes |
| `an_slm_matches_f64_electrode_for_electrode` (`oc-gpu/tests/cpu3_vs_f64.rs`) | the preview against the f64 reference, ray for ray: a blazed grating, a Fresnel lens, a vortex, a uniform pattern; the dead space, the array's edge, three wavelengths, skew rays | the suite's landing and flux gates | passes |
| `an_slm_matches_cpu` (`oc-gpu/tests/gpu3_vs_cpu3.rs`) | the shader against its CPU mirror | the suite's gates | passes (Vulkan) |
| `an_lcos_blazed_grating_sends_the_director_half_into_its_first_order` (`oc-trace/tests/modulator_orders3.rs`) | an 8-level staircase: the thin-element first order sinc²(1/8) [4], renormalized as Harvey does [12], half the unpolarized light; `sin θ = λ/(8p)` [11] | 1e-12 | exact |

## What is not modelled

- Liquid-crystal dynamics and flicker, the dispersion of the birefringence,
  and the phase's dependence on the angle of incidence.
- The dead space's own liquid-crystal phase; it is unmodulated here.
- Rigorous electromagnetics. An 8 µm pixel at 1.55 µm is 5λ, well inside the
  region where scalar theory's error exceeds ±5% [8]. The sampled near field
  raises a `ScalarLimit` flag below 14λ.
- No vendor device presets ship until a datasheet has been checked and cited.

## Validation

The tests are in `crates/oc-wave/tests/slm.rs` and cover acceptance cases
C49–C51.

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `an_n_level_blazed_grating_reaches_sinc_squared_one_over_n` | `sinc²(1/N)`: 0.40528, 0.81057, 0.94964, 0.98721 [4] | 1e-12 | exact |
| `fill_factor_and_dead_space_set_the_zero_order` | `η₀₀ = \|g\|²(1 − ρ²)²`, `η₁₀ = ρ⁴ sinc²(ρ/N)` [3]; Parseval | 1e-12 | exact |
| `a_kinoform_at_the_wrong_wavelength_loses_efficiency_as_sinc_squared` | the quantized geometric series; `sinc²(1 − λ₀/λ)` in the limit [4] | 1e-12; 1e-4 | exact; 0.9675 at λ₀/λ = 0.9 and 1.1 |
| `the_lcos_modulates_only_the_director_component` | `cos²(ψ − δ)` modulated, the rest a plain mirror [1] | 1e-12 | 0.750 at 30° |
| `the_sampled_lcos_lands_on_the_closed_form_orders` | the closed form, edges on sample boundaries | 1e-10 | 2.7e-15 |
| `fringing_lowers_blazed_efficiency_and_matches_quadrature` | direct quadrature of the same model [5], [6] | 1e-3 | first order 0.950, 0.906, 0.860, 0.766 at w = 0, ¼, ½, 1 pixel; agreement 4.8e-5 or better |
| `a_spiral_phase_leaves_a_dark_core_of_charge_one` | `exp(iℓφ)` focuses to a dark core, its phase winding ℓ times [9] | 1e-10; ℓ = 1 | 8e-33; 1 |
| `gerchberg_saxton_error_never_increases` | the error-reduction property [10] | non-increasing | 2.00 → 0.096 in 60 iterations |
| `pommet_flag_is_raised_below_14_wavelengths` | [8] | exact | passes |

The Python bridge repeats the staircase and Gerchberg–Saxton checks
(`crates/oc-py/tests/test_wave_modulators.py`).

## Sources

Bibliographic data checked 2026-10-19; items marked † still need a detail
confirmed against the printed text.

1. Z. Zhang, Z. You, D. Chu, "Fundamentals of phase-only liquid crystal on silicon (LCOS) devices," *Light Sci. Appl.* 3, e213 (2014), doi:10.1038/lsa.2014.94.
2. G. Lazarev, P.-J. Chen, J. Strauß, N. Fontaine, A. Forbes, "Beyond the display: phase-only liquid crystal on Silicon devices and their applications in photonics [Invited]," *Opt. Express* 27(11), 16206–16249 (2019), doi:10.1364/OE.27.016206.
3. H. Zhang et al., "Elimination of a zero-order beam induced by a pixelated spatial light modulator for holographic projection," *Appl. Opt.* 48(30), 5834 (2009), doi:10.1364/AO.48.005834.
4. G. J. Swanson, "Binary optics technology: the theory and design of multi-level diffractive optical elements," MIT Lincoln Laboratory Technical Report 854 (1989), DTIC ADA213404. † equation location
5. "Fringing-field effect in liquid-crystal beam-steering devices: an approximate analytical model," *J. Opt. Soc. Am. A* 21(10), 1996–2008 (2004). † author list
6. M. Persson, D. Engström, M. Goksör, "Reducing the effect of pixel crosstalk in phase only spatial light modulators," *Opt. Express* 20, 22334–22343 (2012), doi:10.1364/OE.20.022334.
7. R. W. Gerchberg, W. O. Saxton, "A practical algorithm for the determination of phase from image and diffraction plane pictures," *Optik* 35(2), 237–246 (1972).
8. D. A. Pommet, M. G. Moharam, E. B. Grann, "Limits of scalar diffraction theory for diffractive phase elements," *J. Opt. Soc. Am. A* 11(6), 1827–1834 (1994), doi:10.1364/JOSAA.11.001827.
9. L. Allen, M. W. Beijersbergen, R. J. C. Spreeuw, J. P. Woerdman, "Orbital angular momentum of light and the transformation of Laguerre-Gaussian laser modes," *Phys. Rev. A* 45, 8185–8189 (1992), doi:10.1103/PhysRevA.45.8185.
10. J. R. Fienup, "Phase retrieval algorithms: a comparison," *Appl. Opt.* 21(15), 2758–2769 (1982).
11. J. E. Harvey, C. L. Vernold, "Description of diffraction grating behavior in direction cosine space," *Appl. Opt.* 37(34), 8158–8160 (1998), doi:10.1364/AO.37.008158.
12. J. E. Harvey et al., "Linear systems formulation of non-paraxial scalar diffraction theory," *Proc. SPIE* 8122, 812205 (2011), doi:10.1117/12.894562.
