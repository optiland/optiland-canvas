# optiland-canvas (Python bindings)

`optiland_canvas` is the seam between [Optiland Canvas](https://github.com/optiland/optiland-canvas)
— the browser-based optical sketchpad — and the [Optiland](https://github.com/optiland/optiland)
engine it hands designs off to (spec §6). It round-trips a `.ocanvas` scene to an
Optiland `Optic` and back.

The exact geometry translation lives in Rust (`oc-core`'s `SeqSystem`); this
package is a thin PyO3 layer (`optiland_canvas._native`) plus a little Python that
builds/reads the Optiland object.

## Install / build

Requires a Rust toolchain and [maturin](https://www.maturin.rs).

```sh
cd crates/oc-py
maturin develop --release          # build + install into the active environment
# or build a wheel:
maturin build --release            # -> target/wheels/optiland_canvas-*.whl
```

The Optiland bridge needs Optiland itself:

```sh
pip install optiland            # or: pip install -e '.[optiland]'
```

## Use

```python
import optiland_canvas as oc

# Export: .ocanvas scene  ->  Optiland Optic
optic = oc.to_optic("design.ocanvas")
optic.draw()

# Import: Optiland Optic  ->  .ocanvas scene (the 3D scene the browser edits)
open("out.ocanvas", "w").write(oc.from_optic(optic, version=2))
```

### Fidelity is reported, never faked

Export is **exact** for the rotationally-symmetric refractive subset. Anything the
sequential `Optic` cannot represent — splines, GRIN, tilted/off-axis surfaces,
extra sources — is reported, not silently approximated:

```python
optic = oc.to_optic(scene)                 # emits FidelityWarning(s) if lossy
optic = oc.to_optic(scene, strict=True)    # raises NotExportableError instead
notes = oc.scene_to_seq(scene)["notes"]    # the raw report
```

### Optiland-free helpers

These work without Optiland installed (pure `.ocanvas` ⇄ prescription):

| Function | Does |
|---|---|
| `scene_to_seq(scene)` | scene → sequential-prescription dict (with `notes`) |
| `seq_to_scene(seq)` | prescription → `.ocanvas` JSON (version 1) |
| `seq_to_scene3(seq)` | prescription → `.ocanvas` JSON (version 2), and its notes |
| `validate(scene)` | read and validate a document of any version, return the version |
| `load(path)` / `save(scene, path)` | read / normalize-and-write a `.ocanvas` file, keeping its version |

`scene` may be a path, a JSON string, or a `dict`.

### Versions, and the ways a document fails to read

The browser saves version 2; older files and `seq_to_scene` write version 1.
Every function that takes a scene reads both (`oc.READABLE_VERSIONS`), and
normalizing a document keeps it in the version it was written in.

A document that cannot be read raises one of three errors, so a script can
tell a wrong file from a newer tool from a broken scene:

| Error | Means |
|---|---|
| `NotADocumentError` | not JSON, or JSON that is not an `.ocanvas` document |
| `UnsupportedVersionError` | an `.ocanvas` version this build does not read |
| `InvalidSceneError` | a readable version whose scene does not hold up (a field of the wrong type, a detector with no pixels, a dangling reference) |

A scene that reads fine but has no sequential form raises `NotExportableError`
from the export functions, and a design with no scene form (a mirror, a
virtual image) raises `NotImportableError` from the import functions. All five
are `OcanvasError`, which is a `ValueError`, so code written against the old
`ValueError` still catches them.

### Exporting a version 2 scene

A version 2 scene exports one path: from a source to a detector. Leave out
`source_id` and `detector_id` for a scene with one declared path, or one source
and one area detector; otherwise name them.

```python
seq = oc.scene_to_seq("scene.ocanvas", source_id=5, detector_id=6)
optic = oc.to_optic("scene.ocanvas", source_id=5, detector_id=6)
```

The source's axial ray is traced to the detector, and the surfaces it meets, in
the order it meets them, are the sequence; each is then read from the scene for
its radius, conic, aspheric terms, clear aperture and the glass behind it. This
first export takes a coaxial chain of revolved singlets and circular apertures,
lit by a collimated beam down the axis or a point source's cone. A fold, a
zero-thickness surface (mirror, splitter, filter, polarizer), a tilted or
decentred element, a central obstruction, a non-circular opening, an off-axis
beam or a Lambertian emitter raises `NotExportableError`, listing every one it
found with the object it concerns.

### The aperture stop

The stop is the one the scene declares for the path, on the aperture
(`primary_stop_for`) or on the path (`intent.primary_paths[].stop`); the two
must agree. Every aperture the path crosses becomes a flat surface where the
path crosses it, and the declared one is the stop, so an interior stop lands
between the lenses it sits between and the pupils follow from it.

When nothing declares a stop (and a version 1 scene cannot declare one, having
no apertures), `scene_to_seq` returns `"stop": None`, and `to_optic` decides:

```python
optic = oc.to_optic(scene, stop=3)          # name the stop surface
optic = oc.to_optic(scene)                  # first surface, and a FidelityWarning saying so
optic = oc.to_optic(scene, strict=True)     # NotExportableError: declare a stop, or name one
```

An inferred stop moves the entrance and exit pupils, and with them every
off-axis ray, which is why it is announced or refused rather than assumed.

### Importing an Optic as a 3D scene

`from_optic(optic, version=2)` builds the scene the browser edits: each lens
element a revolved body, the stop an aperture the scene declares, one source
per field filling the entrance pupil the way Optiland launches its rays, and a
detector on the image plane. The default, `version=1`, writes the older 2D
document for existing callers; it has no apertures and so cannot hold the
stop.

```python
scene = oc.from_optic(optic, version=2)              # warns about anything approximate
scene = oc.from_optic(optic, version=2, strict=True) # NotImportableError instead
scene, notes = oc.seq_to_scene3(oc.seq_from_optic(optic))
```

Reading the `Optic` (`seq_from_optic`) keeps what the canvas needs to trace
the same design:

- each surface's clear aperture: its physical aperture, or else the widest
  real ray over every field, wavelength and pupil edge, 1% wider;
- each glass's catalog and published dispersion formula, so the exact
  tracer's indices are Optiland's; a glass whose formula is not a three-term
  Sellmeier curve also carries a fitted one for the GPU preview;
- the primary wavelength first, object-height fields, mirrors, object-space
  telecentricity, and, for a design that aims its rays through the real stop,
  each field's real pupil.

What it cannot carry is reported in `notes`: an `r²` aspheric term (refitted
as `A4` to `A18`, with the largest sag difference), a curved image surface, a
tilted or decentred surface, a surface type other than a conic or an even
asphere.

### The gallery

`python -m optiland_canvas.gallery web/public/gallery` builds every design in
`optiland.samples` this way and writes the browser's gallery: one `.ocanvas`
per design, `index.json` for the cards, and `checks.json`, rays Optiland
traced through each design with where they land. `oc-trace/tests/gallery3.rs`
traces the committed scenes with the canvas's reference tracer and holds them
to those rays; `tests/test_gallery.py` round-trips every sample through
`to_optic`. See [the gallery](../../docs/gallery.md).

The tests in `tests/test_documents.py` read files the browser actually saved
(`tests/data/browser-*.ocanvas`). `web/tests/save.spec.ts` saves the same
scenes again and fails when the app's output drifts from them; regenerate with
`UPDATE_BRIDGE_FIXTURES=1 npx playwright test tests/save.spec.ts` in `web/`.

## Tests

```sh
pip install -e '.[test]'
pytest tests/
```

The round-trip corpus (`tests/test_roundtrip.py`) asserts that a set of designs
survives `prescription → Optic → prescription` with parameter-level equality, and
that a rebuilt scene traces the same as the Optiland `Optic` it came from.
