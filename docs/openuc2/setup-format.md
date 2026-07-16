# `.uc2canvas`, and the flat `.ocanvas` export

Two files come out of openUC2 mode, and they are not interchangeable.

| File | Carries | Use it to |
|---|---|---|
| `.uc2canvas` | product identity + placement | reopen your setup, generate a BOM |
| `.ocanvas` | optical geometry only | hand the optics to Optiland or Canvas |

**Save `.uc2canvas`.** It is the one that remembers what you built. The `.ocanvas`
export is derived and cannot reconstruct it: arbitrary scene elements carry no
product identity, so flattening is one-way by construction.

## `.uc2canvas`

Plain JSON, serialized deterministically (every map ordered, no timestamp in the
scene-bearing parts) so a diff shows only what you changed.

```json
{
  "format": "openuc2-canvas-setup",
  "version": "1.0.0",
  "metadata": { "name": "Folded singlet demo", "...": "" },
  "catalog": {
    "repository": "openUC2/openUC2-optikit-modules",
    "commit": "<full-commit-sha>",
    "grid_profile": "openuc2-core-50"
  },
  "view": { "active_layer": 0, "trace_plane": "XY" },
  "instances": [
    {
      "instance_id": "m-0001",
      "module_uuid": "…",
      "module_version": "1.0.0",
      "fallback": { "module_id": "lens-pcx-nbk7-1x1", "name": "…", "footprint_cells": [[0,0,0]] },
      "placement": { "anchor_grid": [1, 0, 0], "rotation_quadrants": 0 },
      "parameter_overrides": {},
      "enabled": true
    }
  ],
  "custom_canvas": { "ocanvas": {} }
}
```

Rules worth knowing:

- **Modules resolve by `(uuid, version)`**, never by display name.
- **A missing version is never silently replaced by a newer one.** It becomes an
  unresolved placeholder built from `fallback`, which still occupies its cells and
  still says what it was. Substituting a different version of a product under you
  is not a convenience.
- **Module-derived geometry is never serialized here.** It comes from the
  catalogue; storing it too is how the two drift apart. Only *custom* drawing goes
  in `custom_canvas`.
- The catalogue commit is pinned, so a setup describes a specific revision of the
  module repository.
- Dates are metadata and never affect scene generation.
- A newer **major** version is refused rather than mis-read. Minor is additive and
  still opens; view state is additive on purpose.

## The flat `.ocanvas` export

Materializes the enabled modules on the active layer, merges any custom geometry,
and emits an ordinary `.ocanvas` v1 scene — the same format the freeform Canvas
reads and [`oc-py`](../../crates/oc-py) hands to Optiland. It contains no UUIDs,
no prices, no grid cells. That is the point: the generic scene format stays
generic ([architecture.md](architecture.md)).

**The fidelity caveats travel beside the file, not inside it.** `.ocanvas` v1 has
nowhere to put them, and inventing a field would couple the format to openUC2. So
the UI shows the report when you export, and you should read it: a flattened scene
whose source and detector are idealized looks exactly like one that was measured.

## Custom geometry

Drawn objects live in `custom_canvas`, are inserted before module geometry (so
their ids are stable while modules come and go), are excluded from the BOM, and
are exported. They are `custom / unverified` and correspond to no product; see
[fidelity.md](fidelity.md).

## The BOM

Enabled instances aggregate by `(module_uuid, module_version, parameter_overrides)`.
Two versions of one product are never merged, and two instances configured
differently are different lines. Exports as JSON or CSV; the CSV is generated in
Rust because escaping a module name containing a comma is not a thing to get wrong
in a file someone orders parts from. Custom objects are counted separately and
priced never.

No quantity discounts, no quotation, no checkout.
