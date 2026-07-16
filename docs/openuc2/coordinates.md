# openUC2 coordinates

Coordinate drift is the failure mode where the icon, the CAD, the ports, the
optical geometry, and the rays all disagree about where a module is, and every
individual file looks fine. This page is the single convention that prevents it.

Implemented in [`coordinates.rs`](../../crates/oc-openuc2/src/coordinates.rs) as
named functions. **Do not repeat axis swaps inline** — that is how the fifth
convention gets invented.

## The canonical frame

Right-handed, millimetres:

| Axis | Meaning |
|---|---|
| `+X_uc2` | nominal forward optical direction |
| `+Y_uc2` | lateral, in the module/grid plane |
| `+Z_uc2` | vertical; layer stacking |

> **This is also the canonical Canvas world frame.** The 3D migration adopts
> these exact axes as Canvas `X/Y/Z`
> ([architecture/coordinates.md](../architecture/coordinates.md)), so there is no
> openUC2-to-Canvas rotation and never was. `uc2_to_canvas` drops `z` because the
> tracer is 2D, not because the frames disagree. That drop is what PR 3D-01
> removes, at which point `oc-openuc2` calls the shared conversions in `oc-core`
> instead of keeping its own copy. Until then, everything on this page is current.

## Mappings

Canvas, for the horizontal trace plane:

```text
canvas.x = X_uc2
canvas.y = Y_uc2
```

`Z_uc2` has no Canvas counterpart. It is carried by the selected layer and by
explicit port heights, and the 2D tracer never sees it.

Three.js, for the mechanical view:

```text
X_three = Y_uc2
Y_three = Z_uc2
Z_three = X_uc2
```

This keeps openUC2/Canvas on a `+X` optical axis and Three.js on its usual `+Z`,
with vertical mapped to Three.js `+Y`.

The canonical point `(100, 150, 55)` maps to Canvas `(100, 150)` and Three.js
`(150, 55, 100)`. Both round trips are tested to `1e-12 mm`.

## The grid

The default profile is **data, not a constant**:

```yaml
id: openuc2-core-50
pitch_mm: [50.0, 50.0, 55.0]   # X, Y, Z/layer
cell_size_mm: [50.0, 50.0, 50.0]
mounting_plane_offset_mm: 5.0
allowed_rotations_deg: [0, 90, 180, 270]
```

Note the in-plane pitch (50) is not the layer pitch (55).

**A cell origin is the cell's centre**, on the mounting plane — not a corner. A
module's frame origin is the centre of its anchor cell. Anything drawing a grid
must offset by half a pitch, or the lines run through the modules instead of
between them. `GridProfile::cell_at` is the tested inverse of `cell_origin` and is
what a pointer snaps through; nothing in the browser divides by its own 50.

Optical height is **never inferred from cube dimensions**. Ports carry an explicit
`z`.

## Rotation

Only whole quarter turns, and only those the profile allows. A rotation is about
the **anchor cell's centre**, and it moves everything the module owns together:
optical geometry, source origins and directions, detector endpoints, port poses,
and occupancy cells.

Placement is one rigid `Isometry2` composed onto the fragment's existing
transforms — boundary geometry is never rewritten. See
[module-fragments.md](module-fragments.md).

## Footprints

`cells` are absolute cells; `Footprint::offsets()` subtracts `anchor_cell` to get
the offsets rotation acts on.

**A footprint is not required to contain its own anchor cell.** An L-shaped module
can have its anchor on a corner it does not occupy. Never address a module by its
anchor cell — use `select_instance(id)`. Occupancy uses cells, never CAD bounding
boxes.

## Ports

`pose.position_mm` is in the canonical module frame and `pose.direction` is the
**outward** connector normal, unit length.

Because `+X_uc2` is forward, an in-line optic takes light in through `-X`
(direction `[-1, 0, 0]`) and emits through `+X` (direction `[1, 0, 0]`). On a
50 mm grid this makes neighbours mate by construction: a `+X` port at `x = +25`
meets the next cube's `-X` port at `x = -25` on the boundary between them.

Two optical ports connect when, after placement, their positions coincide
(0.5 mm), their heights agree (0.5 mm), and their outward normals oppose
(`dot <= -0.999`). This is advisory: it produces diagnostics, never optical
topology, and free-space tracing does not consult it.
