# The sun, heliostats, tubes and Fresnel lenses

Four palette buttons cover a daylighting or concentrator layout: **＋ sun**,
**＋ heliostat**, **＋ tube** and **＋ Fresnel**. This page says what each one models, what it
leaves out, and how the results were checked, and ends with the two tools a
layout like that leans on: sizing an aperture, and locking one element to
another.

## The sun

The sun is a source the kernel places for you. You give it a site (latitude,
longitude, elevation), a date, a local clock time with its UTC offset, and the
compass bearing of the canvas's `+X`. The canvas is a vertical section, so
`+Y` is up; `+Z` points 90° clockwise of `+X`. The default bearing of 180°
puts south to the right.

- **Position:** NREL's Solar Position Algorithm (Reda and Andreas 2004, 2008
  revision), with atmospheric refraction. It matches the report's worked
  example to 5e-6° and pvlib's `spa_python` to 5e-13° on nine dates and sites.
- **Size:** a disc of angular radius 959.63″ over the Earth-Sun distance,
  about 4.65 mrad.
- **Shape:** how bright the disc is across its face, and the aureole around
  it (see [the shape of the sun](#the-shape-of-the-sun)). A new sun is
  limb-darkened as measured, with 5% of its light in the aureole.
- **Spectrum:** the ASTM G173-03 direct plus circumsolar spectrum over 0.3 to
  2.5 µm, which holds 99.13% of its 900.1 W/m², or a table you import for your
  site and day. A new sun samples it continuously: every ray draws its own
  wavelength at the table's resolution, 1661 intervals for G173, half a
  nanometre apart in the ultraviolet and one in the visible. The ray count
  does not grow with the table. **lines** instead sends every ray out once per
  line, each standing for an equal-width band, which is what files from before
  this choice use.
- **Power:** the sun is calibrated. Its beam carries DNI × beam area × band
  share, in watts. When the sun is the only source, detector readings are in
  W and W/mm². Add another source and they go back to relative flux, because
  the other source has no unit.
- **Night:** below the horizon the beam carries 0 W.

The beam is a disc square to the sun, a standoff up-sun of what it lights. With
heliostats in the scene it centres on them and grows to cover every mirror at
any orientation. Without heliostats you set its diameter in the sun panel.

The time slider sweeps the day in 5-minute steps, and the heliostats follow.

A new sun stands over Jena's Deutsches Optisches Museum (Carl-Zeiss-Platz 12,
50°55'40.1″ N, 11°34'45.4″ E, about 160 m), with the clock at solar noon on
the browser's own time zone, until you set a site.

### The shape of the sun

- **limb-darkened:** Neckel and Labs, *On the wavelength dependency of solar
  limb darkening (λλ303 to 1099 nm)*, Solar Physics 153, 91–114 (1994),
  Table 1: a fifth-order polynomial in the cosine of the angle on the solar
  surface, measured at 30 wavelengths, interpolated between them and held at
  the ends. The limb is darker in the blue than in the red, so each ray's
  direction is drawn for its own wavelength, and a lens sees a slightly
  smaller sun in blue light. The coefficients come from the Swedish Solar
  Telescope's reduction pipeline, whose copy of the 365.875 nm row sums to
  0.998 instead of 1; every row is renormalised to 1 at the centre, which
  absorbs it. Past 1099 nm there is no data in the paper, and the 1099 nm
  profile is held.
- **Buie:** the broadband disc of Buie, Monger and Dey, *Sunshape
  distributions for terrestrial solar simulations*, Solar Energy 74, 113–122
  (2003), one profile for every wavelength.
- **uniform:** evenly bright, the flat "pillbox" sun.
- **aureole (CSR):** the circumsolar ratio, the share of the light within
  43.6 mrad (2.5°, a pyrheliometer's field) that comes from outside the disc.
  The aureole has Buie's shape, falling as a power of the angle out to
  43.6 mrad. Buie's shape parameter is not the CSR his profile delivers, so
  the parameter comes from the fit in Tonatiuh (CENER's concentrator ray
  tracer), and the split between disc and aureole is then imposed exactly.
  Clear days run from 1 to 10%; 5% is a new sun's. The G173 spectrum includes
  the circumsolar light, so the aureole is where part of the DNI already
  comes from; it does not add any.

Each shape moves light around and makes none: the beam carries the same watts
whichever you pick.

### Rays: every one its own direction

A source that spreads both its origins and its directions, like the sun, gives
every ray its own origin and its own direction: two low-discrepancy sets of
`spatial × angular` points, paired by a fixed shuffle. Before, every origin
went out along the same `angular` directions, so a sun of 2 angular samples
was 2 points in the sky. A light pipe keeps each ray's angle to its axis and
spins it around the axis, so those 2 points came out of a tube as 2 rings.
With every ray its own direction the far field at a tube's exit is the filled
disc the sun's size and the optics' magnification make.

### A spectrum for your site

G173 is a reference: air mass 1.5, a sun 41.8° up, a standard atmosphere. For a
particular site, day and sky, import a table: **import…** in the sun panel
reads a SMARTS output file, a spectroradiometer export, or two typed columns.
It takes wavelength in nm (or µm, when every value is below 30) and
irradiance in W/m²/nm (or per µm, when the column says so). With several
columns it picks the direct beam with its circumsolar part by the header's
names, and the panel says which column it read. The DNI becomes the table's
total; **G173** goes back to the standard.

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

Slope error is modelled ([below](#slope-error)). Not modelled yet: tracking
error as a time series (the melting example turns the mirror by a fixed
offset, `HELIO_ERR_MRAD`), focused or canted facets, and atmospheric
attenuation between mirror and target.

A real heliostat facet is silver behind float glass. The demonstrator models
it as a coating on the mirror's own surface: a plate of glass with silver on
its far face, met through the glass (`CoatingBacking::Plate` with
`Incidence::Substrate`), its thickness absorbing by the glass's `k`
([coatings](coatings.md#a-plate-is-a-piece-of-glass)). That sums the front
face's 4% and the bounces inside the glass exactly, which a body would only
do with ghost tracing on.

## Slope error

No mirror is the shape it is drawn as. `figure_errors` lists objects whose
faces lean, at every hit, away from the drawn normal by a Gaussian angle,
`slope_rms_mrad` RMS in each of the two directions across it; a reflected ray
turns by twice that. This is the statistical model concentrator tracers use
(SolTrace's slope error): it knows nothing of where on the face the error
sits, so it cannot stand for a mirror bent into the wrong curve.

The draw is hashed from the hit's own point and direction, so a trace
repeats exactly and no two rays share one. A lean that would turn the face
away from the ray, or send its mirror image into the face, is drawn again: at
grazing incidence a wavy face hides those slopes behind its crests.

Published values the demonstrator uses: 1.22 mrad measured on a glass-faceted
heliostat at Sandia (Mitchell and Zhu, Solar Energy 2020), 0.8 mrad for
dish glass facets, 1.45 mrad radially and 2.32 around on an electroformed
nickel dish (NASA TN D-4072).

The f32 preview draws faces true; the reduction to the 2D tracer refuses a
scene with a figure error, and the sequential export notes it and drops it.
`figure_error3.rs` checks that a 2 mrad mirror spreads a parallel beam by
4 mrad per axis, the same on every run.

### Feeding an instrument from a measured beam

A source can follow a measured angular profile:
`AngularDistribution3::RadialProfile { half_angle_rad, radiance }`, radiance
(any scale) at 24 polar angles evenly spaced out to `half_angle_rad`, linear
between them, the same all round the axis. The demonstrator measures the
light pipe's beam at 32 m from the whole run and feeds the spectrograph's
bench with it: the sun's disc as the telescope magnified it, widened by every
slope error and wall bounce.

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

## Fresnel lenses

A grooved, rotationally symmetric Fresnel lens, like OpticStudio's
non-sequential Fresnel 1. The recipe is a base surface (radius, conic, even
aspheric A4 to A18), a clear radius, the grooves (constant pitch or constant
depth), the substrate under the groove valleys, which way the grooves face,
and a draft angle for the risers. Positive R converges whichever way the
grooves face. The component view edits the recipe and rebuilds the lens.

Each facet is an exact piece of the base surface shifted along the axis, not a
chord, so the lens bends a ray exactly as the smooth lens does at the same
exit height: 0.0 difference in exit direction over 240 heights on three
lenses. Risers are real surfaces. With a 5° draft they catch 2.96% of a
collimated beam against a 2.97% area estimate; with no draft, collimated light
from the flat side never reaches one.

The default pitch is 0.73 mm. A grid source puts its rays at whole
millimetres, and a zero-draft riser at a whole millimetre then has a ray
running along its surface into a corner where the normal has no defined side.
At 0.73 mm none of the default lens's risers sits on a whole millimetre. A
real beam meets that corner with probability zero.

Sequential export to Optiland refuses a Fresnel lens by name: Optiland has no
grooved surface to receive it.

## Apertures

An aperture is a hole in a plate. The hole is a circle, a ring (a circle
with a central stop) or a rectangle; the plate is a circle, a rectangle, or
unbounded, blocking everything in its plane outside the hole. The card and
the component view size each on its own, in diameters and full widths.
Making the hole larger than its plate grows the plate to keep a rim; making
the plate smaller than its hole is refused.

## Locking one element to another

Any mirror, aperture, detector, lens or tube can be locked to another
element from its card ("lock to"), in one of three ways:

- **in front of it:** parallel to the other element and on its axis, a set
  gap in front of it (a negative gap is behind). The axis is the face normal
  of a mirror, aperture or detector and the optical axis of a lens or tube.
  A baffle in front of a heliostat, or a filter on a lens, stays put however
  the parent moves or turns.
- **attached where it is:** rigidly attached in the pose it has when you
  lock it, and moved and turned with the parent from then on. Dragging it,
  or typing a position, moves it on the parent.
- **turned to face it:** stays where it is and turns its axis toward the
  other element, the way a heliostat turns toward its target.

The kernel writes every link into the scene's transforms on each edit,
before and after it turns the heliostats, so an element locked to a
heliostat follows the mirror through the day. A heliostat mirror cannot be
locked itself (the sun turns it), and a link that would close a loop is
refused. Deleting either element drops the link.

## Scale

A heliostat field is tens of metres. The reference tracer's longest free
flight grows with the scene (four scene diagonals, at least 10 m), so a
target 50 m from its mirror is reached rather than written off as an escape.
