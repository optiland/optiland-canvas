# The demonstrator's projection spectrograph

The light pipe at the Deutsches Optisches Museum feeds two experiments. The
metal-melting experiment at 37 m takes nearly all of it. At 32 m a
spectrograph takes a strip of the light and projects spectra onto a table for
visitors. There are three, side by side, each 375 to 690 nm over 2.25 m and
180 mm across, 50 mm apart:

- the sun's, in the middle, sharp enough that the sodium D lines (588.995 and
  589.592 nm) are two separate dark lines seen from 1 m;
- a halogen lamp's on one side: a continuous spectrum with no lines;
- a sodium lamp's on the other: the two D lines bright where the sun's are
  dark.

The scene is built by `crates/oc-trace/examples/heliostat_lightpipe.rs`, the
optics in `examples/shared/spectrograph.rs`, the materials and coatings in
`examples/shared/realism.rs`. The locked design (round 4) is a set of fitted
values passed through the environment
(`D:/tmp/spectrograph-bright/current-best/states/round4.sh`):

```sh
. round4.sh
cargo run --release -p oc-trace --features trace3 --example heliostat_lightpipe -- heliostat-lightpipe.ocanvas
```

## The room

The tube's axis is 5.0 m up in the model. The tube is 355 mm across inside
and 360 mm outside. The ceiling is taken 250 mm above the tube's axis, 70 mm
above its top, the floor 2.97 m below the ceiling, the table 0.55 m high.
Every optic above the table stays at least 2.15 m above it, so visitors can
walk under it; the nearest, the projection lens, is 2.26 m up. With
`TABLE_UP` the model is the room upside down (world `-Y` up).

## The chain

The source is a strip of the tube's light, not a slit cut in it: the tube's
beam is 355 mm across and about 1 degree wide, and a slit in a sheet would pass
a few millimetres of it. So a condenser images a strip of the whole bore onto
the slit.

1. **Condenser pick-off**: an off-axis paraboloid strip on the tube's axis in
   a 200 mm gap between two lengths of tube, a few millimetres wide and as
   long as the bore, ground on a 6 mm glass bar whose back shades the light
   running on to the melting experiment.
2. **Condenser mirror C1**, an off-axis ellipsoid used 6 degrees off its axis,
   and the **slit**, an off-axis paraboloid strip 0.45 mm wide on the arc the
   pick-off's image lies on. Together they image the pick-off onto the slit
   at f/8.4.
3. **UV/IR cut filter**, 375 to 690 nm, 30 mm past the slit.
4. **Periscope P1/P2** drops the light to the collimator's height without
   turning the slit's image (a drop inside the condenser would tilt the strip).
5. **Collimator M1**: an off-axis paraboloid, f 1.8 m, its return beam 18
   degrees off the incoming one; then **fold F2**.
6. **Grating**: ruled, 1800 lines/mm, the light arriving 50 degrees off its
   normal. The zero order goes into a dump 600 mm out; a black mount behind
   the grating takes what the lamps send past it.
7. **Camera M2**, a weakly aspheric mirror with the grating near its centre of
   curvature, leaning about 8.7 degrees so its light passes the grating, and
   **M3**, a weakly convex fold that takes out the bow the lean puts in.
8. **Fold F4** and the **field lens** at the intermediate spectrum: two PMMA
   Fresnels face to face, which send the camera's pupil into the projection
   lens.
9. **Fold F5** and the **projection lens**: a Cooke triplet cut to a slab,
   N-SK16 and N-F2 with every face an even asphere, designed in Optiland,
   f 832 mm, throwing the spectrum 2.3 times larger onto the table, tilted
   8 degrees across the band to take out the keystone.

`IDEAL_PROJ` and `IDEAL_FIELD` swap the projection and field lenses for thin
lenses, for comparison only.

## The lamps

| | halogen | sodium |
|---|---|---|
| lamp | Osram HLX 64625, 100 W 12 V | Osram Na/10 spectral lamp |
| emitter | coiled filament 4.2 × 2.3 mm, 3600 lm, colour 3450 K | discharge, 1.2 × 10⁵ cd/m² |
| spectrum | tungsten at that colour: 3369 K true, de Vos's emissivity | the two D lines, D2 twice D1 |
| optics | ellipsoidal reflector (enhanced aluminium) throwing the coil onto the slit at f/8.4 | condenser, N-BK7 plano-hyperbolic |
| slit | 23 mm (a continuum shows no lines to blur) | 1.35 mm |

Each lamp's slit sits in the sun slit's focal surface, one either side of it
along its length, curved so its lines land where the sun's do, and folded into
the shared periscope. The datasheet fluxes are measured outside the bulbs, so
the bulbs' losses are in them.

## What the parts are made of

Every part is modelled as a vendor would sell it (`examples/shared/realism.rs`); `IDEAL=1`
puts back the constants the design was laid out with, `REAL_<PART>=0` one of
them, so each one's cost can be read off.

| part | as built | was |
|---|---|---|
| mirrors | enhanced silver (Ta₂O₅/SiO₂ quarter-wave pair over silver, transfer matrix): 99.1% of the light at normal incidence, 98.7% at 45°, 97.6% at 380 nm | 0.99 at every angle and wavelength |
| grating | ruled aluminium, 1800 lines/mm, 500 nm Littrow blaze (MKS/Richardson 53-*-290R), TE and TM by rigorous coupled-wave analysis in this mount, the groove fitted to the vendor's measured curves | one declared curve, 80% at its peak, polarization-blind |
| field Fresnels | PLEXIGLAS GS solar clear 0Z023 (absorption from the measured sheet), Beadie's index, risers drafted 3° | Sultanova's index, lossless, vertical risers |
| projection lens | N-SK16 and N-F2 absorbing as Schott's internal transmittance says | lossless |
| lamp optics | N-BK7 absorbing; halogen reflector enhanced aluminium | lossless; 0.95 |
| feed | the angular profile of the tube's light at 32 m and its spectral gain, both measured from the whole run | the sun's disc through the telescope, 0.8°; a straight line through the gain |

The grating is the change that matters. At 50 degrees it turns the light 37
degrees, far from Littrow, and there light polarized along the grooves goes
mostly into the zero order: 48% into the first order at 550 nm, against 81%
for the other polarization. Weighted by the eye that is 57% of the light where
the declared curve said 79%. Below 490.6 nm the second order opens and takes
a fifth to two thirds of the TM light, so the violet end loses most. A
grating ruled for this mount, or used nearer Littrow, would recover part of
it; the catalogue part is what is modelled.

The filter stays a table: in the f/8.4 cone the edges of a dielectric filter
move by about 0.4 nm, which no line notices.

## The sun's spectrum

ASTM G173-03 is tabulated in 1 nm steps across the visible, too coarse for
lines 0.6 nm apart. Inside 380 to 720 nm the sun uses the TSIS-1 Hybrid Solar
Reference Spectrum (Coddington et al., 2021) at 0.005 nm, moved to air
wavelengths. G173's own atmosphere is carried over at its 1 nm, with its
telluric bands divided out and a line-by-line transmission put in: HITRAN
2024 O₂ and H₂O lines through a US Standard Atmosphere from 160 m, 1.42 cm of
water as G173 has, air mass 1.5. The O₂ B band at 687 nm is resolved: its
strongest lines go black at their cores, and after the spectrograph's 0.09 nm
the band head reaches 0.40 of the continuum, where G173's 1 nm says 0.70.
`examples/heliostat-lightpipe/site_spectrum.py` builds it.

## The bench

Tracing the whole demonstrator puts a handful of rays through a 0.45 mm slit.
The bench is the spectrograph alone, fed with what the tube carries at 32 m,
both measured from a whole run:

- **the spectral gain**, the tube's irradiance over the sun's in 20 nm bands,
  smoothed by a median of three bands. The sun's sampler ties a ray's
  wavelength to where on the mirror it starts, which leaves the odd band
  10-15% off its neighbours; a median removes it and keeps the violet edge
  the eyepiece's glass puts in (0 at 370 nm, 2.5 at 410, 5.9 at 550).
- **the angular profile** of the light crossing the bore, as a
  `RadialProfile` source: the sun's disc magnified 2.97 times, widened by the
  mirrors' slope errors and folded back by the wall's bounces. 98.5% of the
  light lies within 2.5 degrees.

`BENCH_ONLY=1` with `TUBE_GAIN` and `TUBE_PROFILE` (files a whole run wrote
through `TUBE_GAIN_OUT` and `TUBE_PROFILE_OUT`) runs the bench alone.

## What it gives

March equinox at solar noon, 900 W/m² of direct sun in G173's shape; the whole
demonstrator traced with 262 thousand sun rays, the bench with 103 million.

| | ideal parts (`IDEAL=1`) | as built |
|---|---|---|
| sun band, 2.25 m × 186 mm | 599 lux | 252 lux |
| halogen band, 2.25 m × 180 mm | 560 lux | 406 lux |
| sodium D2 / D1, line peaks | 114 / 57 lux | 85 / 42 lux |

The lamps lose only to the grating and the optics. The sun's band also loses to
its feed: slope errors on the heliostat (1.2 mrad) and the pick-up mirror
(1 mrad) turn the light by twice their size, and the telescope multiplies that
by 2.97. The tube's beam then has a tail out past 1.5 degrees, where the ideal
optics give the sun's disc, cut at 0.8 degrees, and the condenser and
collimator, sized for the disc, pass less of it.

What each change costs the sun band, one put back at a time (a quarter of the
rays; as built 257 lux): perfect mirror figures 386, the declared grating curve
352, lossless glass 301, everything 602. Together, the heliostat and pick-up
mirror flat to 0.5 mrad and a grating that meets the declared curve give 470
lux; every mirror at 0.3 mrad, 508. The catalogue grating does not meet that
curve in this mount; one chosen or made for 50 degrees, or a mount nearer
Littrow, would.

The violet end goes dark. At 382 nm less than a thousandth of the light that
reaches 486 nm arrives: the eyepiece's dense flint absorbs below 420 nm, the
PMMA's UV absorber below 375 nm, and the grating's TM efficiency falls toward
the second order's onset at 490.6 nm.

The sun's lines, each row registered on the traced monochromatic line and
averaged over the band (as built):

| line | nm | depth | width (mono FWHM) |
|---|---|---|---|
| L | 382.0 | 35% | 1.3 mm |
| K | 393.4 | 80% | 1.1 mm |
| Hβ | 486.1 | 48% | 0.8 mm |
| Mg b1 | 518.4 | 45% | 0.8 mm |
| Na D2 | 589.0 | 38% | 0.8 mm |
| Hα | 656.3 | 49% | 1.3 mm |
| B | 686.7 | 46% | 2.0 mm |

The sodium D lines land 4.36 mm apart (7.29 mm/nm times their 0.597 nm). The
slope errors dim the band but leave the slit's image as sharp: the depths are
those of the ideal parts within 2 points, except B, now the resolved telluric
band (25% with G173's 1 nm atmosphere), and L, measured on almost no light.

The table is an illuminance detector, so it reads lux and draws the spectra in
the colours a visitor sees; `examples/heliostat-lightpipe/table_image.py`
turns its export into a PNG.

## What it leaves out

- The sodium lamp's D2/D1 ratio is the optically thin 2:1; a spectral lamp
  reabsorbs D2 more, and no measured ratio was found.
- The mounts are the grating's, the pick-off's glass bar and the barrel stops;
  frames outside the beams are left out.
- Ruled-grating ghosts and scattered light are not modelled; the second order
  is absorbed.
- The curved mirrors are traced by the f64 reference but not by the fast
  preview.
- The ceiling's height above the tube is assumed, and the model's table sits
  further below the tube than the room's: the clearances need checking
  against the floor plans.
