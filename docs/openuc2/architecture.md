# openUC2 mode: architecture

The openUC2 mode lets you assemble [openUC2](https://openuc2.com) modules on a
50 mm grid and trace the light through them. It is reached at `#openuc2`; the
freeform Canvas is untouched and remains the default route.

The one-line summary of the whole design:

> openUC2 does not need its own ray tracer, and Optiland Canvas does not need to
> become a product database.

## What is true of this engine

Read these first. Everything below assumes them.

- **Canvas is 2D.** It traces a single horizontal plane of meridional rays. It is
  not a 3D ray tracer, and no amount of module metadata makes it one.
- **One layer is traced at a time.** Modules on other layers are mechanical
  context: they are drawn as onion-skin footprints and contribute nothing to the
  trace. A stacked openUC2 system is *displayed*, not simulated.
- **Displayed numbers come from the f64 reference tracer.** The interactive f32
  path draws the picture while you drag; every number you can read off a panel is
  produced by [`oc-trace`](../../crates/oc-trace). This rule is not new to
  openUC2 mode and is not relaxed by it.
- **Fidelity is reported, never assumed.** A module with no optics is called
  layout-only. A module whose optics are approximate says `idealized`. See
  [fidelity.md](fidelity.md).
- **Custom objects are not buildable.** You can draw your own lenses and mirrors
  alongside catalogue modules; they are labelled `custom`, excluded from the BOM,
  and correspond to no openUC2 product.

## Three models, three owners

| Model | Answers | Lives in |
|---|---|---|
| Module descriptor | what a reusable openUC2 object *is* | `openUC2-optikit-modules` |
| Setup (`.uc2canvas`) | which objects are assembled, and where | the file you save |
| Scene (`.ocanvas`) | what the light does | derived, regenerated |

The scene is **derived**. It is materialized from the setup plus the catalogue and
is never maintained in parallel with them, so the two cannot disagree about where
a lens is. Undo snapshots the setup and regenerates the scene from it, which is
why undo can never move the optics without moving the module.

## Crates

```text
oc-core      scene model, geometry, materials, .ocanvas, scene fragments
oc-trace     f64 reference tracer — the authoritative numbers
oc-gpu       interactive f32 tracer — the picture
oc-openuc2   THE ONLY CRATE THAT KNOWS WHAT AN openUC2 CUBE IS
oc-wasm      browser bindings
```

`oc-core` and the `.ocanvas` format stay domain-free. They have no idea what a
cube, a SKU, an ImSwitch manager, or a grid cell is. Everything openUC2 lives in
[`oc-openuc2`](../../crates/oc-openuc2), which owns the catalogue, the setup, the
module instances, materialization, the BOM, and the port graph — and owns no ray
physics whatsoever.

What openUC2 needed from the core was *generic*, and was added generically: scene
fragments, rigid batch transforms, scene-table merging, and removal remaps. See
[module-fragments.md](module-fragments.md).

## The data path

```text
module repository (module.yaml + assets + optics/canvas.ocanvas)
        |  build_canvas_catalog.py  (pinned, deterministic, offline)
        v
catalog.json + assets  ──fetch once──>  browser
                                          |
                                          v
                                     oc-wasm  ──>  oc-openuc2
                                                   catalogue + setup + instances
                                                   ownership map
                                                        |  materialize
                                                        v
                                                     oc-core Scene
                                                    /            \
                                              oc-gpu f32      oc-trace f64
                                              (picture)       (numbers)
                                                                  |
                                                                  v
                                                            flat .ocanvas → Optiland
```

The browser never talks to GitHub. The catalogue is a generated, pinned artifact
served as a static file; see [catalogue.md](catalogue.md).

## Where state lives

In Rust. The browser holds a search string, which filters and overlays are on, the
viewport, and which pointer gesture is in flight. It does not hold the instance
list, the placements, the selection, or the undo stack — it asks for a view
whenever it draws one. A mirrored copy is a copy that goes stale.

## Files

- [coordinates.md](coordinates.md) — the canonical frame, and the axis mappings.
- [catalogue.md](catalogue.md) — how the module catalogue is built and pinned.
- [setup-format.md](setup-format.md) — `.uc2canvas`, and the flat `.ocanvas` export.
- [module-fragments.md](module-fragments.md) — how a module's optics get into a
  scene, and the id hazards that come with it.
- [fidelity.md](fidelity.md) — what the engine will and will not claim.
- [../../examples/openuc2/README.md](../../examples/openuc2/README.md) — the demo.
