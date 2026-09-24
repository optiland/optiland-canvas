# The sun, heliostats and tubes

Three palette buttons cover a daylighting or concentrator layout: **＋ sun**,
**＋ heliostat** and **＋ tube**. This page says what each one models, what it
leaves out, and how the results were checked.

## The sun

The sun is a source the kernel places for you. You give it a site (latitude,
longitude, elevation), a date, a local clock time with its UTC offset, and the
compass bearing of the canvas's `+X`. The canvas is a vertical section, so
`+Y` is up; `+Z` points 90° clockwise of `+X`. The default bearing of 180°
puts south to the right.

- **Position:** NREL's Solar Position Algorithm (Reda and Andreas 2004, 2008
  revision), with atmospheric refraction. It matches the report's worked
  example to 5e-6° and pvlib's `spa_python` to 5e-13° on nine dates and sites.
- **Size:** a uniform disc of angular radius 959.63″ over the Earth-Sun
  distance, about 4.65 mrad. No limb darkening and no circumsolar aureole.
- **Spectrum:** the ASTM G173-03 direct plus circumsolar spectrum, split into
  equal-width bands over 0.3 to 2.5 µm (six lines by default). That band holds
  99.13% of the spectrum's 900.1 W/m².
- **Power:** the sun is calibrated. Its beam carries DNI × beam area × band
  share, in watts. When the sun is the only source, detector readings are in
  W and W/mm². Add another source and they go back to relative flux, because
  the other source has no unit.
- **Night:** below the horizon the beam carries 0 W.

The beam is a disc square to the sun, a standoff up-sun of what it lights. With
heliostats in the scene it centres on them and grows to cover every mirror at
any orientation. Without heliostats you set its diameter in the sun panel.

The time slider sweeps the day in 5-minute steps, and the heliostats follow.

## Heliostats

A heliostat is a flat rectangular mirror (silvered front, black back) that the
kernel turns on every edit. Its normal bisects the direction to the sun and the
direction to its target, so reflected sunlight leaves along the line to the
target. The mount is azimuth-elevation, which keeps the mirror's width
horizontal. The target is an object (the heliostat follows it when you drag
it) or a fixed point. The card shows the angle of incidence and its cosine,
the cosine efficiency.

The mirror's tilt field is overwritten by tracking. To stop a mirror tracking,
release it (`releaseHeliostat`); it keeps its last orientation.

Checked against closed forms: the normal makes equal angles with sun and
target to 1e-12, the width stays level to 1e-12, the spot centroid lands
within 5 mm of a target 21.7 m away from 08:30 to 18:45, and the power
delivered is DNI × R × area × cos θ to within 1% (one mirror) and 2% (two).

Not modelled yet: mirror slope error, tracking error, focused or canted
facets, and atmospheric attenuation between mirror and target.

## Tubes

A tube is inner diameter, outer diameter and length, its axis along the
element's local `+X`. The outside and both end rings absorb. The bore is one
of:

- **silver** or **aluminium**, bare or protected by a half-wave silica
  overcoat: a 150 nm film of measured metal (silver from Babar and Weaver
  2015, aluminium from Rakić 1995, 0.2 to 3.5 µm), evaluated by the transfer
  matrix at each ray's own angle and wavelength;
- **ideal mirror** of one reflectance at every angle and wavelength;
- **black**, for a baffle.

The metal is the choice that matters for a light pipe. Light travelling down a
long pipe meets the wall dozens of times at grazing angles, where a metal's
reflectance differs from its value at normal incidence: bare silver at 550 nm
goes from 0.98 at normal incidence toward 1 at grazing.

The tracer budgets interaction depth for a tube by its length over its bore:
a ray at 80° to the axis meets the wall about six times per bore diameter.

Checked by counting bounces: a meridional ray and a skew ray through a mirror
bore come out at exactly `R^N`, with `N` from the geometry, and a 3 m pipe
carries a ray through 60 bounces. One bounce off a metal bore matches the
transfer matrix to 1e-9. The GPU preview matches the f64 reference: exact
paths, and the silver coating grid drifts at most 2.5e-4 of a ray's flux.

## Scale

A heliostat field is tens of metres. The reference tracer's longest free
flight grows with the scene (four scene diagonals, at least 10 m), so a
target 50 m from its mirror is reached rather than written off as an escape.
