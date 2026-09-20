# The gallery

The `▦ gallery` button in the top bar opens a set of ready-made designs. Pick
one and it opens as the scene, framed in whichever view is showing. From then
on it is an ordinary scene: drag the lenses, change the glass, save it.

Today the gallery holds the sample designs that ship with
[Optiland](https://github.com/optiland/optiland) (`optiland.samples`): 26 of
its 29, from a single aspheric lens to a 21-element lithography lens. A link
opens one directly: `?gallery=cooke-triplet`. The ids are the file names in
`web/public/gallery`.

## How a design comes in

`optiland_canvas.from_optic(optic, version=2)` reads the `Optic` into a
sequential prescription (`seq_from_optic`), and `oc_core::seq_import` builds
the scene from it. The gallery is that, run over every sample:

```sh
python -m optiland_canvas.gallery web/public/gallery
```

What the builder does with each part of a design:

| design | scene |
|---|---|
| two surfaces with glass between them | one revolved body, its faces the two surfaces |
| a cemented surface | one face shared by the two bodies either side of it |
| a surface with the same medium on both sides | nothing, unless it is the stop (a curved one is noted) |
| the aperture stop | an aperture at its vertex, opening its clear radius, declared as the stop of the scene's path |
| a field, object at infinity | a collimated beam whose footprint on a plane square to the axis is the paraxial entrance pupil, as Optiland launches it |
| a field, object at a finite distance | a point on the object plane with a cone that fills the pupil, about the axis when object space is telecentric |
| the image | an area detector on the image plane |
| a catalog glass | the glass with its published dispersion formula |
| an ideal material | a glass with that one index at every wavelength |

**Clear apertures.** Optiland's samples give most surfaces no aperture; it
draws each to the widest ray that reaches it. The reader does the same, over
every field, wavelength and pupil edge, and adds 1% so the edge rays do not
graze the rim they define.

**Lens outlines.** A body's two faces are cut to one diameter when their clear
apertures are within 10% of each other. Further apart, the smaller face ends
at its own aperture and a flat shoulder takes the element out to the larger,
the way Optiland draws it. Carrying a steep face far past the aperture it was
designed for makes a shape nobody designed. The eye model's paraboloid lens
face is 3.7 mm across. Carried out to the 26 mm its vitreous body needs, it
curls 57 mm back toward the light and swallows the front of the eye.

**Glasses.** Most catalog glasses are published as three-term Sellmeier
curves, and the canvas stores them as they are. Hikari's and Hoya's are
published as polynomials, and one CaF2 dataset has four terms. The glass then
carries its formula, and the exact reference tracer evaluates that, so its
indices are Optiland's. The fast preview evaluates only Sellmeier curves on
the GPU, so for those glasses it draws with a fitted curve. `index.json`
records how far each fit strays over 0.36 to 1.0 µm, and the worst is
9e-5 in index.

**Rays aimed through the stop.** Four wide-angle samples switch on Optiland's
ray aiming, so their rays fill the real stop rather than the paraxial pupil.
At 85° the difference is not small. The real pupil is a sliver of the
paraxial one, well off its centre. For these designs each field's beam covers
its real pupil, measured from Optiland's aimed rays, square to the beam's own
direction. The stop aperture is set to the stop's exact size and trims each
beam there, which is how a real lens works. Exporting such a scene reads the
on-axis beam as the entrance pupil, 2% wider than the design's, and the scene
says so.

## How it is checked

**Ray for ray against Optiland.** The generator records, for every design,
rays Optiland traced at every field and wavelength: the chief ray, both
meridional marginal rays and a sagittal one, 816 in all, with where each
lands (`checks.json`). `oc-trace/tests/gallery3.rs` checks two things for
each:

- the ray is one the scene's source for that field emits: it runs in the
  beam's direction and crosses the beam at the pupil radius Optiland sampled
  (inside the beam, for an aimed design), or leaves the object point at the
  angle that pupil radius makes;
- the canvas's reference tracer lands it on the image detector where
  Optiland lands it.

Every design agrees to about 1e-15 of its length. The aspheric singlet agrees
to 5e-12 (see below).

**Through the bridge and back.** `crates/oc-py/tests/test_gallery.py` builds
each sample from the installed Optiland, exports it with `to_optic`, and
checks that every surface that bends light comes back at the same place with
the same shape. It also checks that the stop is the same surface, and that
the focal length and entrance pupil are the design's. It then checks that the
committed gallery is what the generator writes, when the installed Optiland
is the version the gallery was written with.

**In the browser.** `web/tests/gallery.spec.ts` opens every design through the
app and checks that its light reaches its image in the app's own exact trace.

## What is approximate

A card with approximations says so, and so does the message when it opens.

- **Aspheric singlet.** Optiland's even asphere has an `r²` term; a canvas
  face starts at `r⁴`. The face is refitted: the vertex curvature takes the
  `r²` term exactly, and `A4` to `A18` the rest, within 1e-6 nm of the design's
  sag. Optiland's own paraxial trace reads the base radius and leaves the
  `r²` term out, while its real rays feel it. So the design's paraxial focal
  length (25.48 mm) is not its surface's (25.72 mm), and the real rays agree
  with the canvas.
- **Navarro wide-angle eye.** Its image surface, the retina, is curved
  (R = -12 mm). The canvas detector is flat, at the retina's vertex, inside
  the vitreous body, which runs on past it. Rays land where they cross that
  plane.
- **Uncoated glass.** A sample names no coatings, so every face is bare glass,
  and the reference trace takes its Fresnel reflections. Uncoated germanium
  reflects 36% per face, so the infrared triplets deliver about 10% of their
  light. Optiland's own trace does not attenuate.

## What is not in yet

The gallery lists these, with the reason:

- **Hubble Space Telescope** and the **53x UV reflecting objective** are
  mirror systems whose mirrors have central holes. That needs a curved
  zero-thickness surface with a hole. The reference tracer has one; the fast
  preview does not yet, so the picture would show light through the mirrors.
- **20x microscope objective.** Its image lies 0.196 mm before its last
  surface, a virtual image no detector can sit at.

## Fixed on the way

Porting 26 real designs found faults the hand-built scenes had not:

- **Trace depth.** The app's exact 3D trace stopped every ray after 24
  interactions, and the fast preview after 32. The lithography lens needs 43,
  so its rays died halfway through it. Both are now sized to the scene's face
  count (`oc_trace::depth_for_scene3`), the way the 2D path already was.
- **Meniscus outlines.** The check that a lens outline does not cross itself
  sampled every face from `-h` to `+h`, whichever way the outline ran through
  it. For a bent meniscus it joined the back face by a chord straight across
  the body, and refused valid lenses.
- **Cemented faces on export.** The exporter tracked which glass the ray is in
  by toggling bodies as the ray met them. A cemented face is met once, on
  either body, so the toggle lost track. It now reads the medium the way the
  tracer does: whatever contains the point just past the surface.
- **Reading an `Optic`.** `seq_from_optic` read every field type as an angle
  (Optiland 0.6 names the type by class), never saw a mirror (it asked the
  surface, and Optiland 0.6 keeps that on the interaction model), took the
  first wavelength as primary, gave every surface without a physical aperture
  a clear radius of 0, and dropped a nonzero `r²` aspheric term without a
  word.
- **Writing an `Optic`.** `optic_from_seq` put the image in air behind the
  last surface. For an image inside a medium, as in the eye, that added a
  refraction the design does not have, and its focal length read short by the
  vitreous index.
