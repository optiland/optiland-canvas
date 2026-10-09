# Diffraction gratings

A plane grating on a flat standalone surface, reflecting or transmitting. The
reference tracer, the f32 preview on the CPU and the GPU shader all trace it,
and the three agree ray for ray.

## What it does to a ray

The grooves run along the surface's local `Y`, so the grating vector lies
along local `X`. A ray arriving with direction cosines `(dx, dy, dz)` in the
surface's frame leaves in order `m` with

```text
dx' = dx + m λ / d,   dy' = dy,   dz' = ±√(1 − dx'² − dy'²)
```

where `d` is the groove spacing. In the plane across the grooves this is the
grating equation `sin β = sin α + m λ / d`; out of that plane it is conical
diffraction, which the same three lines already cover. A reflection grating
sends `dz'` back to the side the ray came from; a transmission grating carries
it on. When `dx'² + dy'² > 1` the order is past grazing and carries no light.

Both sides of a zero-thickness surface are in the same medium, so `λ` is the
wavelength in that medium, which in air is the air wavelength line tables
quote. A positive order turns light toward local `+X`.

## How much light goes where

The directions are exact. The fractions are not computed: a grating's
efficiency depends on its groove profile, the wavelength and the polarization,
and takes a rigorous electromagnetic calculation this kernel does not do. So
they are declared:

- `efficiency`: the power fraction sent into `order`, one number or a table in
  wavelength (a vendor's curve).
- `zero_order`: the fraction left in the zero order, the specular reflection
  off a reflection grating or the straight-through beam of a transmission one.
  It is traced, because in an instrument it is a bright white beam that has to
  go into a dump.
- The rest, which a real grating sends into its other orders, is absorbed at
  the surface and counted as absorbed in the ledger.

## Polarization

A ruled metal grating treats the two polarizations differently, by tens of
percent near its blaze and with sharp anomalies where an order opens. `tm`,
when present, gives `{efficiency, zero_order}` for light polarized across the
grooves (TM, the electric field perpendicular to them); `efficiency` and
`zero_order` are then the fractions along them (TE). The f64 tracer splits
each ray's Stokes state on the grooves, as a diattenuator with amplitudes
`√η_TE` and `√η_TM` and no retardance, so the light leaving is partly
polarized and the mirrors after it see that. The f32 preview takes the mean of
the two curves. Without `tm` the state passes through scaled, and the
capability report records the grating as polarization-blind.

Vendors publish TE and TM curves measured near Littrow. An instrument far
from Littrow behaves differently, and the curve does not transfer by
wavelength alone. The demonstrator's grating table was computed by rigorous
coupled-wave analysis in its own mount, for the groove whose facet angle
reproduces the vendor's measured curves (D:/tmp/realism/grating).

The second order is not traced. A spectrograph that needs to be free of it
blocks the wavelengths that would land there, as the demonstrator's does with
a UV cut filter.

## In the scene file

```json
"Grating": {
  "lines_per_mm": 1800.0,
  "order": 1,
  "efficiency": { "Tabulated": { "lambda_um": [0.40, 0.55, 0.70], "values": [0.55, 0.86, 0.62], "interpolation": "Linear", "out_of_range": "Clamp" } },
  "zero_order": { "Constant": 0.05 },
  "transmissive": true
}
```

A reflection grating's back face is an `Absorber`; a transmission grating
carries the same response on both faces. A body patch cannot be a grating: the
grooves need the flat surface's frame, and the tracer refuses one.

## In the editor

**＋ grating** in the tools card adds a 25 × 25 mm reflection grating of 600
lines/mm facing the beam, its grooves out of the section plane so it disperses
in it. The component view draws it edge on with the fan it makes: an arriving
ray at an angle you drag (◆), every 25 nm from 400 to 700 nm leaving in its own
colour, the zero order dashed, and a warning for wavelengths past grazing. The
readout gives the angles, the fan's width and the angular dispersion at
550 nm. The arriving angle is the drawing's alone; in the scene every ray
meets the grating at its own. Typing an efficiency replaces a tabulated curve
with that number.

## What the tests hold it to

- **The grating equation**: Littrow sends a ray back along itself, normal
  incidence on 600 lines/mm at 0.5 µm leaves at sin β = 0.3, and an order past
  grazing is gone (`oc_core::grating`).
- **The reference tracer**: a beam onto a reflection grating lands on a
  detector placed where the equation says, to 1e-9 mm, with 65% of the light
  there, 10% in the zero order and 25% absorbed; skew rays keep their
  component along the grooves, reflecting and transmitting
  (`oc-trace/tests/grating3.rs`).
- **The preview**: the f32 CPU mirror follows the f64 reference branch for
  branch, reflecting and transmitting, both signs of order, skew rays, and a
  wavelength whose order is evanescent (`cpu3_vs_f64`); the GPU shader follows
  the CPU mirror (`gpu3_vs_cpu3`).
- **The editor**: the palette's grating opens its editor, its fields reach
  the scene, and an efficiency that would make more light than arrives is
  refused (`web/tests/grating.spec.ts`).
