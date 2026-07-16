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

# Import: Optiland Optic  ->  .ocanvas scene
open("out.ocanvas", "w").write(oc.from_optic(optic))
```

### Fidelity is reported, never faked

Export is **exact** for the rotationally-symmetric refractive subset. Anything the
sequential `Optic` cannot represent — splines, GRIN, tilted/off-axis surfaces,
extra sources — is reported, not silently approximated:

```python
optic = oc.to_optic(scene)                 # emits FidelityWarning(s) if lossy
optic = oc.to_optic(scene, strict=True)    # raises ValueError instead
notes = oc.scene_to_seq(scene)["notes"]    # the raw report
```

### Optiland-free helpers

These work without Optiland installed (pure `.ocanvas` ⇄ prescription):

| Function | Does |
|---|---|
| `scene_to_seq(scene)` | scene → sequential-prescription dict (with `notes`) |
| `seq_to_scene(seq)` | prescription → `.ocanvas` JSON |
| `validate(scene)` | parse + version-check, returns the version |
| `load(path)` / `save(scene, path)` | read / normalize-and-write a `.ocanvas` file |

`scene` may be a path, a JSON string, or a `dict`.

## Tests

```sh
pip install -e '.[test]'
pytest tests/
```

The round-trip corpus (`tests/test_roundtrip.py`) asserts that a set of designs
survives `prescription → Optic → prescription` with parameter-level equality, and
that a rebuilt scene traces the same as the Optiland `Optic` it came from.
