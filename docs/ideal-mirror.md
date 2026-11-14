# The ideal mirror

A flat standalone surface that reflects as a mirror of given power: a
collimated beam comes to a focus on the reflected axis, at any angle of
incidence, with no aberration of its own. It is the reflecting twin of the
ideal thin lens, for a design that needs a focusing mirror's first-order job
done before its figure is chosen: the parabola, the off-axis section or the
toroid that will do it in glass comes later. The reference tracer, the f32
preview on the CPU and the GPU shader all trace it, and the three agree ray
for ray.

## What it does to a ray

Two steps, in the surface's own frame. First the ray reflects as off a flat
mirror on the surface's plane: the same reflection, about the same normal, a
`Mirror` of reflectivity 1 gives it. Then it is bent across that reflection,
toward a focus on the **reflected axis**, the line through the surface's
origin along the flat reflection, at `f = 1 / power` down it.

A ray reflected at `p` runs parallel to that axis, a height `h` off it and
already a distance `s` down it, so the line from `p` to the focus has the
slope

```text
h / (f − s)  =  h · power / (1 − s · power)
```

across the axis, and that is the slope it leaves with. Every ray of a
parallel bundle is aimed at the one point, however steeply the bundle meets
the surface: there is no astigmatism and no coma, by construction, and a
diverging mirror (a negative power) sends each ray out along the line from
the virtual focus `|f|` behind the surface. A real curved mirror cannot do
this off axis. A sphere used at 45° focuses its two planes at `f cos θ` and
`f / cos θ`, 70.7 and 141 mm for `f = 100`.

The two powers act separately. `power_x_per_mm` bends along the surface's
local `X`, carried onto the plane square to the reflected axis, and
`power_y_per_mm` along local `Y`, square to both. Two equal powers make one
point focus. Two different ones make two line foci, each at its own focal
length down the axis: the `X` power brings the beam to a line running along
`Y`, and the other way round. A zero power is flat in that direction, so one
power and one zero is a cylindrical mirror, and both zero is a flat mirror,
traced bit for bit as the `Mirror` response of reflectivity 1.

The focus belongs to a parallel bundle. A beam that arrives at a different
angle has a different reflected axis and focuses on that one, `f` down it.
Light from a point at a finite distance arrives at a different angle at every
point of the surface, and each ray is aimed at the focus of its own axis: near
the axis that images the point where the mirror equation `1/s + 1/s' = 1/f`
puts it, and away from it the rays fall a little short. On axis, at normal
incidence, an `f = 100` mm mirror images a point 300 mm away at 150 mm, and
the ray 10 mm off the axis crosses it at 149.875 mm. The ideal thin lens,
whose slope law is written for any conjugate, images every axial point
exactly; this mirror is exact for the collimated beam only.

## How much light goes where

All of it, along the one reflected ray: no transmission, no absorption, no
ghost, and no dispersion, so every wavelength focuses at the same point. No
path length is added at the surface, so the optical path to the focus differs
from ray to ray by the geometry of the flat reflection and the bend; the
mirror is ideal in where it sends light, not in the wavefront it leaves.

## Polarization

The flat reflection is the ideal reflector's: equal coefficients for `s` and
`p` and no phase between them, the same model `Mirror` uses, and the
capability report says so when polarized light meets it. The bend after it is
not a reflection and has no plane of incidence; the polarization frame is
carried across it by the shortest rotation, as through an ideal thin lens.

## Figure error

A scene's slope error leans the normal of the flat reflection, as it does a
`Mirror`'s. The axis then follows the leaned reflection, so each ray focuses
`f` down its own axis and the spot grows by about `2σ f`, the blur a real
mirror of that figure shows.

## In the scene file

```json
"IdealMirror": {
  "power_x_per_mm": 0.01,
  "power_y_per_mm": 0.025
}
```

A power is `1 / f` in `mm⁻¹`, positive converging and negative diverging, and
must be finite. The response is named `ideal-mirror` where a surface's kind is
reported. A one-sided mirror carries it on the front face and an `Absorber` on
the back; on both faces it reflects from either side.

It belongs on a flat standalone surface, whose frame gives it an origin, an
axis and the two directions its powers act along. A body patch has none of
these, so a patch that asks for one is refused, by the reference tracer with
a capability issue and by the preview at compile time. The planar 2D model
and the sequential export to Optiland have no surface like it and refuse it
too rather than tracing it as flat.

The editor has no palette entry for it. A scene that carries one, written by
hand or by another tool, opens and traces in the canvas with the mirror drawn
as its flat surface.

## What the tests hold it to

- **The reference tracer** (`oc-trace/tests/ideal_mirror3.rs`): a 9 × 9
  collimated beam onto equal powers at 0°, 30°, 45° and 60° runs every
  reflected ray through the focus to 1e-9 mm (the worst measured, 8e-14 mm,
  is f64 rounding), the detector's per-axis best focus reads `f` on both axes,
  and at the focus the spot's RMS radius is under 1e-9 mm, on the axis. A
  diverging mirror's rays come from the virtual focus the same way. Two
  powers put the two line foci at `f_x` and `f_y` along the right axes, with
  no width across each line at its own focus; swapping the powers swaps them.
  Both powers zero traces the same segments, detector readouts, polarization
  states, ledger and report as a `Mirror` of reflectivity 1, bit for bit. All
  the light reaches the detector, in one branch per ray with ghosts on, and
  polarized light keeps its degree of polarization. A body patch is refused.
- **The scene format** (`oc-core/tests/scene3.rs`): the response serializes
  as above, and a power that is not finite fails validation.
- **The preview**: the f32 CPU mirror lands where the f64 reference does, for
  two powers, equal powers, a diverging one and none, rays off the axis in
  both directions and a skew ray (`cpu3_vs_f64`); the GPU shader follows the
  CPU mirror (`gpu3_vs_cpu3`). At 4 mm off the axis of a 100 mm mirror the
  bend is 0.04 rad, forty times the direction tolerance, so a preview that
  drew the mirror flat fails both.
