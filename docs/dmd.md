# Digital micromirror devices (DMD)

A DMD is an array of square mirrors on a square pitch. Each mirror tilts to
one of two angles, ON or OFF. Lit by a plane wave it acts as a blazed
two-dimensional grating, so where its light goes depends on the wavelength and
the angle of incidence.

The model is implemented in the kernel and from Python, and validated against
two published analyses [1], [2]. A DMD is a scene element: the ray tracer and
the preview trace it in the geometric limit, and the wave path carries its
diffraction ([in the scene](#in-the-scene)). In the app, ＋ DMD adds one and
the component view edits it.

## The model

**Geometry.** Each pixel `(i, j)` holds a mirror of side `ρ·p`, centred on its
pixel, `p` being the pitch.
- ON tilts the mirror by `+θ` about its hinge, OFF by `−θ`; flat (the parked
  state) leaves it level.
- The hinge is the pixel diagonal on the 13.68, 10.8, 9.0 and 7.6/7.56 µm
  devices, and a pixel side on the 5.4 µm devices ([2] §4.2).
- The device lies in its local `XY` plane facing `+Z`. Light arrives toward
  `−Z`.

**Reflection, in the thin-element approximation.** A mirror reflects the
incident wave as its tilted plane would, restricted to the mirror's outline
projected onto the array plane, with the phase zero at the mirror's pivot:

```text
T(r) = √R · T_w · Σ_mirrors mask(r − r_ij) · exp(ik (t_spec − t_in)·(r − r_ij))
```

- `t` is the in-plane (tangential) part of a direction.
- `t_spec` is the incident direction reflected off that mirror's own normal.
- `R` is the mirror reflectance and `T_w` the window's single-pass power
  transmittance (two passes give `T_w` in amplitude).

This is the 2D, any-incidence form of Popoff et al.'s pixel response
([1] App. A). The envelope of the far field is centred on the tilted mirror's
specular direction, and the orders sit on the lattice.

**Orders, in closed form** (`oc_core::modulator::unit_cell_orders`). For a
periodic pattern, the Fourier series over its unit cell is exact:

```text
c_mn = (√R T_w / A_cell) Σ_ij e^{−i2π G_mn·r_ij} · M̂(G_mn − (t_spec − t_in)/λ)
```

- Order `(m, n)` leaves along `t_in + λ(m, n)/p` in direction cosines [3].
- `M̂` is the projected mirror's transform: a square compressed by `cos θ`
  across the hinge, `M̂(f) = cos θ · a² · sinc(a(Af)ₓ) sinc(a(Af)ᵧ)` with
  `A = ââᵀ + cos θ ŵŵᵀ`.

**Two order models:**
- `Tea` takes `|c_mn|²` as it is. The share the evanescent orders hold is lost.
- `HarveyRenormalized` (the default) lets the propagating orders share the
  evanescent power in proportion. This is Harvey's non-paraxial scalar
  treatment [4], which TI cites for its DMD efficiencies [2].

Each `Order` holds two things apart: its `coefficient`, the thin element's
`c_mn`, which a coherent calculation adds as a field, and its `efficiency`,
the power fraction under the table's model. Under `Tea` the efficiency is
`|coefficient|²`; under `HarveyRenormalized` it is that times
`Σ_all|c|²/Σ_propagating|c|²` and the coefficient is left alone, so
`√efficiency` is not an amplitude (Q5 of the expert review of the wave
engine). `OrderTable::renormalized()` turns a `Tea` table into Harvey's. The
renormalization is a declared scalar redistribution of the evanescent orders'
share, not a field the element produces.

**Sampled near field** (`oc_wave::modulator::dmd_reflect`). A field on the DMD
plane is multiplied by the family's mask, sample by sample. The mask is
band-limited exactly to the field's grid: its spectrum in closed form, `M̂`
times the lattice sum of the mirrors' pistons, taken on the grid's own
frequencies and summed back onto its samples. So every order the grid holds
has its exact amplitude whatever the sampling, and what the mirrors reflect
beyond the grid's band is booked as `spectral_projection`. A mask averaged
over each sample's cell, as before, weighted an order `m` by `sinc(m/s)` at `s`
samples a mirror: 0.935 in amplitude for an order 1.6 lattice steps out at 8
(Q17 of the expert review). The power the mirrors intercept and reflect is
each sample's exact cell coverage. A device lit within a pixel of a window
edge it reaches past keeps the cell average, and the report says so: the
band-limited mask is periodic over the window and would fold the mirrors the
edge cuts onto the far side.

Each mirror state reflects into its own direction, so a state is a *family*:
its tilt is held as the field's analytic carrier, and the samples keep only a
piston phase per mirror. Without the carrier, a 7.56 µm mirror tilted 12° at
0.4 µm would need about thirty samples to resolve its ramp; with it, four
suffice. The families not followed are booked as `other_family`.

**The facet against the mask** (`facet_integral`, `facet_interception_ratio`).
The array-normal mask with its pivot ramp is a defined thin-element
approximation, not one facet's scattering (Q4 of the expert review). One flat
square facet of side `a` under a uniform plane wave, with its physical
tangents `u`, `v`, has the surface integral `a² sinc(a q·u/2π) sinc(a q·v/2π)`,
`q = k_in − k_out`. Both peak at the facet's specular direction. Off it the
facet sees the outgoing ray's normal component through its tilted tangent,
which the mask does not, so the two envelopes part: at third order in the
angle across the hinge, at fourth along it (below). The power differs too: the
mask reflects `ρ² cos θ` of each mirror's cell, where an isolated facet would
intercept `ρ²|d_in·n_f|/|d_in·z|`, `1/cos 24°` = 1.095 times as much at 24°
incidence on 12° facets. Neighbours shade an actual array, so neither is the
device, and no correction is applied. Above 15° incidence the report raises a
`ThinElement` flag with that ratio.

**Far field of a whole device, factorized** (`dmd_far_field`). Every mirror of
one state has the same response, so the far field is one mirror's transform
times the lattice sum:

```text
Û(f) = Σ_states M̂_s(f − t_s/λ) · Σ_{ij∈s} U_in(r_ij) e^{−i2πf·r_ij}
```

The lattice sum is a chirp-z transform at one sample per mirror. A 1920 × 1080
device costs one transform of that size, where a sampled field would need some
33 million samples. The result is on a grid of direction cosines; `|Û|²/λ²` is
power per unit `dα dβ`.

**Projector efficiency** (`dmd_projector_efficiency`). This follows TI's
method ([2] §4.2):
- an incoherent sum of plane waves over the illumination cone (an f-number
  about the direction the ON mirrors send along the normal);
- each plane wave's orders collected inside the projection lens's cone;
- the result as a share of the reflected power.

`photopic_average` weights a flat 420–680 nm spectrum by `V(λ)` from the CIE
1931 observer, as [2] Table 6-1 does.

**Patterns** (`PatternGenerator`):
- binary gratings;
- Lee holograms, encoded as Gutiérrez-Cuevas and Popoff write them [5], Eq. (11):
  ON where `cos(2πν·r − φ) ≥ cos(πq)`, with `A = sin(πq)`.

## In the scene

`Scene3::add_modulator(pos, tilt_deg, device)` places a device on a standalone
rectangle the size of its pixel array, its rows along world `Z`. The front is
the device (`SurfaceResponse::Modulator`); the back absorbs.

**The ray tracer traces the geometric limit.** A ray finds the pixel it hits.
If it lands inside that mirror's outline, projected onto the array plane, it
reflects off the pixel's facet, tilted to the pixel's state, as an ideal mirror
of power reflectance `R·T_w²` (`T_w` per pass of the window). In the dead space
between mirrors it is absorbed. Diffraction orders are not traced: the report
lists `Micromirror` as approximated ("geometric limit"), and the
[wave path](wave-optics.md#the-wave-path) carries the diffraction.

**Or it traces the diffraction orders.** With the surface's `ray_model` set to
`Orders { max_orders, min_efficiency }` (the editor's *rays: orders*), each hit
spawns the brightest diffraction orders of the pattern's unit cell where it
hits, from the closed form above with Harvey's renormalization [4], each a
child ray along its direction on the grid `t_in + λ(m, n)/p` [3] carrying
`η` of the arriving flux: at most `max_orders`, none below `min_efficiency`
of it. The dead space's share is absorbed. The orders not followed are booked
in the energy ledger as `untraced_orders`, never dropped. The unit cell is one
pixel of a uniform pattern, or the period of a binary grating that repeats in
whole pixels along a pixel axis; anything else is the 8 × 8 pixels about the
hit taken as one period, and the report says so. The order tables assume the
device in air; elsewhere the geometric limit is traced, and named. This is the
model for projector work, where a lamp's incoherent cone of plane waves meets
the device. The preview always draws the geometric limit.

**The preview draws the geometric limit, pixel for pixel,** in f32. Each pixel's state
travels to the GPU at two bits a pixel, up to 2²⁴ pixels; a uniform pattern
needs none. A larger pattern is named on the preview's badge and traced by the
reference alone.

**In a file** the device is part of the version 3 document. Written as version
2 it becomes an absorber of the same footprint, and the loss is named
([format](ocanvas-format-v3.md#spatial-light-modulators)).

**In the app** the tools dock's ＋ DMD adds a device facing the beam with TI
DLP9000's pitch and tilt, all mirrors ON, about 10 mm across. The component
view's editor sets the array, the optics and a TI preset ([2] Table 2-1), and
the pattern: all ON, all OFF, a binary grating, or an image of your own,
thresholded at half its luminance. It draws the pattern beside its
propagating orders in direction cosines, at a wavelength you choose, and says
when the pitch is below 14λ. The 3D view draws the device as one dark,
glossy face; its pattern is not drawn there yet.

**On the wave path** the device's chains come from the wave path's own
extraction (`extract_wave_sequences3`). Under the geometric limit it traces
once for each mirror family the pattern holds, every mirror taken in that
family, so each family the field lights has its chain whatever the one pixel
(or gap) the axial ray meets (R17 of the expert review); the chain carries the
family whose direction it leaves along, and the report says when the axial
ray's own pixel is another. Under the orders model each order the tracer
follows is a chain, and a chain carries its own order only: of every family's
reflected field it keeps the cell of its order on the lattice of the
pattern's unit cell (directions `λ/P` apart about the incident one), added as
fields, and the cells no chain follows are booked once as `untracked_orders`
(R16 and finding 2). Before, each order's chain carried the whole spectrum.

**Not yet:** a tested wave path through a Fourier-plane filter that picks one
order, a Lee hologram's −1 (C56). The orders model gives each order its own
chain, which such a filter needs; the case itself is not in the suite.

| Test | Checks | Gate | Measured |
|---|---|---|---|
| `an_on_mirror_reflects_a_ray_through_twice_its_tilt` (`oc-trace/tests/modulator3.rs`) | facet reflection and `R·T_w²`; OFF mirrors send the ray `2θ` the other way | 1e-12 | passes |
| `a_dmd_in_the_ray_limit_absorbs_its_gaps` (same) | mirrors of side `ρp` tilted `θ` about a side cover `ρ²cos θ` of the array | 2e-3 | 0.792000 against 0.792300; ledger closes to 1e-10 |
| `a_modulator_document_round_trips_and_writing_v2_notes_the_loss` (same) | an explicit bitmap through the file; the v2 absorber and its loss line | exact | passes |
| `a_dmd_matches_f64_mirror_for_mirror` (`oc-gpu/tests/cpu3_vs_f64.rs`) | the preview against the f64 reference, ray for ray: ON, OFF and flat mirrors, both hinge layouts, the dead space, skew rays | the suite's landing and flux gates | passes |
| `a_dmd_matches_cpu` (`oc-gpu/tests/gpu3_vs_cpu3.rs`) | the shader against its CPU mirror | the suite's gates | passes (Vulkan) |
| `a_dmd_in_the_scene_focuses_its_orders_f_lambda_over_p_apart` (`oc-wave/tests/path.rs`) | orders `λ/p` apart [3], so `fλ/p` apart at a lens's focus | 1% | 1.000 |
| `a_dmd_scene_reads_the_same_whatever_the_device_sampling` (same) | V06: the focusing scene with the device laid at 4, 8 and 16 samples a pixel, nothing else changed | 1e-3 of the peak pixel | 2.7e-5 at 4, 5.5e-6 at 8 |
| `a_dmd_s_orders_each_carry_their_own_cell_and_add_up_to_the_full_field` (same) | R16: the orders model against one full-field chain, 2 mm behind the device where four orders overlap, at most 1, 2, 4 orders and thresholds 0.1, 0.03; detected plus unfollowed constant over the sweep | four orders 1e-3 of the peak, 1e-4 of the launched; the threshold picks the count's orders exactly | 1.9e-4; detected 0.9017 both; one order 0.518, two 0.817; the sum constant to 6e-5 |
| `a_dmd_chain_follows_the_mirrors_the_field_lights_not_the_axial_ray_s_pixel` (same) | R17: the axial ray on the gap at four mirrors' corner (a 400 × 400 device), and on an OFF mirror among ON ones, against the plain device | 1e-3 of the peak; three times the mirror's share | gap 5.6e-6 (the tracer's own chain is absorbed there); OFF mirror 4.3e-3, the mirror's share 2.6e-3 |
| `dmd_orders_traced_match_the_unit_cell_order_for_order` (`oc-trace/tests/modulator_orders3.rs`) | the tracer's children against the unit cell's twelve brightest orders, direction and flux | 1e-12 | passes |
| `untraced_orders_are_in_the_ledger` (same) | the orders past the count booked as untraced; the dead space absorbed; the ledger closes | 1e-12; 1e-10 | 1.3e-15 residual |
| `ti_dlpa083_through_the_tracer_matches_direct_integration` (same) | TI's method [2] §4.2 through the tracer: 64 plane waves of an f/2.4 cone onto an all-ON 7.56 µm device, an f/2.4 stop 200 mm away and a detector, against the closed-form integration | 5e-3 | 0.882030 against 0.882032 |

## What is not modelled

- One mirror shading its neighbour at oblique incidence, the mirror's edges,
  hinges and curvature ([1] §3), and thermal drift. TI's on-state fill factor
  includes shading, which this model does not have; see the presets below.
- Rigorous electromagnetics. Pommet, Moharam and Grann [6] found scalar
  theory's efficiencies off by more than ±5% against rigorous coupled-wave
  analysis for binary features below about 14λ, for their 8-level element of
  index 1.5 at normal incidence below a period of 20λ, and the error growing
  with index, depth, level count and incidence. The sampled near field raises
  a `ScalarLimit` flag when the mirror is under 14λ, recording the pitch, the
  mirror, the gap, the relief across the mirror (`ρp sin θ`) and the
  incidence. It is a heuristic, not a bound: a 7.56 µm pitch is 14λ at
  0.54 µm, and no pitch above the mark certifies a device. A gap far below a
  wavelength, as every DMD has, is recorded but does not raise it.
- The superpixel method [7] is not implemented yet.
- Light outside the device's active area is treated as absorbed by its frame.

## Presets

`oc_core::modulator::TI_DMD_PRESETS` holds TI's published pitches, tilts, hinge
axes, matched illumination f-numbers and on-state fill factors ([2] Table 2-1).

TI gives the fill factor as an area fraction from a geometric ray trace at twice
the tilt, so it includes shading. The validation below takes the mirror's
linear fill as its square root. That is a modelling choice: TI does not publish
the mirror gap.

## Validation

The tests are in `crates/oc-wave/tests/dmd.rs` and cover acceptance cases
C46–C48.

| Test | Held against | Gate | Measured |
|---|---|---|---|
| `dmd_orders_sit_on_the_harvey_vernold_grid` | `t_in + λ(m, n)/p` [3]; Harvey normalization sums to the reflected power | 1e-12 | exact; 634 orders propagate at 532 nm, 7.56 µm |
| `the_blazed_number_matches_popoff_figure_6_and_the_facet_kinematics` | [1] Eq. (5) at θ_B = −12°, the values in the legend of Fig. 6: 0.09, 0.71 (7.6 µm at 0° and 15°), 0.81, 0.06 (10.8 µm); the same number from `facet_normal`, `reflect` and the lattice's `√2λ/d` step along the diagonal, over α = −45° to 45° | ±0.01; 1e-12 | 0.094, 0.709, 0.814, 0.060; 5.3e-15 |
| `a_blazed_dmd_concentrates_one_order_and_an_anti_blazed_one_splits_four` | [1] §2: one order dominates where μ is near 1 (10.8 µm, μ = 0.81 at normal incidence); several share where it is near 0 (7.6 µm, μ = 0.09) | shares | 10.8 µm: 85% in one order; 7.6 µm: 26%, 18%, 18%, 14% |
| `a_flat_state_dmd_puts_the_fill_factor_squared_in_the_zero_order` | the rectangular-grating Fourier series: `η₀₀ = ρ⁴`, `Σ = ρ²` | 1e-12 | exact |
| `the_factorized_far_field_lands_on_the_closed_form_orders` | `\|Û\|/(N²p²) = \|c_mn\|` | 1e-12 | 8.0e-17 |
| `the_sampled_dmd_far_field_converges_on_the_factorized_one` | the same model computed two ways, between the window's own frequencies | log-log slope < −1.5; 2e-3 at 32 samples per mirror | slope −1.92; 2.3e-5 (the cell-averaged mask: −1.87; 8.4e-4) |
| `on_its_window_s_frequencies_the_near_field_is_the_factorized_far_field_at_any_sampling` | Q17: the band-limited mask's near field against the factorized far field on the window's frequencies inside the grid's band, at 4, 8, 16 and 32 samples a mirror | 1e-10 of the peak | 3.0e-14 to 4.8e-13 |
| `the_thin_element_mask_against_one_facet_s_surface_integral` | V04, Q4: one facet's surface integral against the mask's envelope at 24° on 12° facets, both peaking at the specular; the gap's order in the angle; the interception ratio; the mask's reflected share `ρ² cos θ`; the `ThinElement` flag above 15° | order to 5%; parity; 1e-12 | the gap 6.7e-4 of the peak 1° across the hinge, 3.0e-3 at 2°, third order (×7.93 on doubling); along it 5.9e-7 at 1°, fourth order (×15.8); ratio 1.0946 = 1/cos 24°; `ρ² cos θ` to 5e-14 |
| `a_harvey_table_scales_efficiencies_and_leaves_the_coefficients_alone` | Q5: Harvey's table is `renormalized()` TEA's, coefficients equal, efficiencies `k|c|²` | exact | `k` 1.016 at 7.56 µm and 633 nm |
| `a_binary_grating_on_the_on_mirrors_puts_one_over_pi_squared_in_each_first_order` | [5] Eq. (15) in the limit; the exact prediction `1/(P sin(π/P))·\|1 − M̂_off/M̂_on\|` | 1e-10 | lattice factor² 0.1026, 0.1014, 0.1013 for P = 16, 64, 256 (1/π² = 0.1013); the OFF mirrors leak 5e-4 into the ON direction |
| `ti_dlpa083_average_photopic_diffraction_efficiency` | [2] Table 6-1 and §4.3, a **vendor simulation** | ±3 pp (one row ±5 pp, below) | see the table below |
| `the_lee_generator_follows_gutierrez_cuevas_and_popoff_eq_11` | [5] Eq. (11) | exact | passes |

**Against TI DLPA083B.** No parameter was fitted. The mirror fill is the square
root of TI's on-state fill factor.

| Device, illumination → projection | TI | This model | Δ |
|---|---|---|---|
| 13.68 µm, 12°, f/2.4 → f/2.4 | 89% | 88.8% | −0.2 pp |
| 10.8 µm, 12°, f/2.4 → f/2.4 | 87% | 86.7% | −0.3 pp |
| 9.0 µm, 14.5°, f/2.0 → f/2.0 | 89% | 87.2% | −1.8 pp |
| 7.56 µm, 12°, f/2.4 → f/2.4 | 84% | 82.9% | −1.1 pp |
| 5.4 µm, 17° side tilt, f/1.7 → f/1.7 | 86% | 87.9% | +1.9 pp |
| 5.4 µm, 17° side tilt, f/2.4 → f/2.4 | 80% | 84.4% | **+4.4 pp** |
| 7.56 µm, 12°, f/3 → f/2.4 (mismatched, §4.3) | 90.5% | 90.5% | 0.0 pp |

The 5.4 µm device at f/2.4 is outside the ±3 pp gate. It is held to ±5 pp and
named here rather than hidden by a wider gate. TI's 5.4 µm pixel tilts about
its side, its mirror structure is the least documented of the set, and the
model's loss from f/1.7 to f/2.4 (3.5 pp) is smaller than TI's (6 pp).

Agreeing with TI shows the canvas computes TI's model. It is not evidence about
a real device: no measured DMD data is in the suite yet.

## Sources

Bibliographic data checked 2026-10-19.

1. S. M. Popoff, L. Malosse, R. Gutiérrez-Cuevas, Y. Bromberg, J. Commère, M. Glanc, R. Galicher, M. W. Matthès, "A practical guide to digital micro-mirror devices (DMDs) for wavefront shaping," *J. Phys. Photonics* 8, 023002 (2026), doi:10.1088/2515-7647/ae60d5. Eqs. (1)–(5), Figs. 3 and 6, §2 and Apps. A–B read from the published text (2026-10-20). The 2024 version, *J. Phys. Photonics* 6, 043001, doi:10.1088/2515-7647/ad6dc0 (arXiv:2311.17496), was retracted (*J. Phys. Photonics* 7, 049701 (2025), doi:10.1088/2515-7647/ae0605) for its blazed number, Eq. (2), on which its Figs. 4(a) and 5–9 rested; nothing here is held against the 2024 text any more.
2. Texas Instruments, *DMD Optical Efficiency for Visible Wavelengths*, application note DLPA083B (June 2019, rev. April 2023). Read in full: Tables 2-1 and 6-1, §§4.2–4.3, Eqs. 1–3, Figs. 4-1 to 4-8.
3. J. E. Harvey, C. L. Vernold, "Description of diffraction grating behavior in direction cosine space," *Appl. Opt.* 37(34), 8158–8160 (1998), doi:10.1364/AO.37.008158.
4. J. E. Harvey et al., "Linear systems formulation of non-paraxial scalar diffraction theory," *Proc. SPIE* 8122, 812205 (2011), doi:10.1117/12.894562.
5. R. Gutiérrez-Cuevas, S. M. Popoff, "Binary amplitude holograms for shaping complex light fields with digital micromirror devices," *J. Phys. Photonics* 6, 045022 (2024), doi:10.1088/2515-7647/ad8617 (arXiv:2311.16685v1: Eqs. (11), (15), (22), (28)). After W.-H. Lee, "Binary computer-generated holograms," *Appl. Opt.* 18(21), 3661–3669 (1979) (DOI to confirm).
6. D. A. Pommet, M. G. Moharam, E. B. Grann, "Limits of scalar diffraction theory for diffractive phase elements," *J. Opt. Soc. Am. A* 11(6), 1827–1834 (1994), doi:10.1364/JOSAA.11.001827.
7. S. A. Goorden, J. Bertolotti, A. P. Mosk, "Superpixel-based spatial amplitude and phase modulation using a digital micromirror device," *Opt. Express* 22(15), 17999–18009 (2014). Planned.
8. CIE 1931 2° standard observer, as tabulated in `crates/oc-core/src/colorimetry.rs`: `V(λ) = ȳ(λ)` for the photopic average.
