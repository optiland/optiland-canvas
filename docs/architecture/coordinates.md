# Coordinates: the canonical Canvas world frame

Coordinate drift is the failure mode where the sketch, the 3D preview, the
Optiland handoff, and the rays all disagree about where something is, and every
individual file looks fine. A handedness flip is the worst of them: it mirrors
every chiral layout, produces a picture that looks entirely plausible, and is
invisible until a prism sends light the wrong way.

So there is **one** conversion module, every mapping in it is a named function,
and **no code anywhere else reorders an axis inline**. That rule is
[contract §3.7](canonical-3d-kernel.md#right-handed-coordinates-only), and it is
the same rule `oc-openuc2` has followed since it was written.

Implemented in
[`crates/oc-core/src/coordinates.rs`](../../crates/oc-core/src/coordinates.rs)
(behind `scene3`; bodies land in PR 3D-01).

## The canonical frame

Right-handed, millimetres, `X × Y = Z`:

| Axis | Meaning |
|---|---|
| `+X` | nominal forward optical direction; the Canvas horizontal axis |
| `+Y` | the Canvas vertical/lateral axis |
| `+Z` | out of the Canvas plane; the openUC2 vertical/layer direction |

The default Canvas view is the plane `Z = 0`, with screen horizontal `+X` and
screen vertical `+Y`. So the 2D scene you see today is the `Z = 0` slice of the
canonical frame, and the migration adds an axis rather than moving any existing
one.

**This is the frame openUC2 already calls canonical.** `+X_uc2`, `+Y_uc2`, and
`+Z_uc2` (`docs/openuc2/coordinates.md`) are these axes under different names.
There is no openUC2-to-Canvas rotation, and there never was: `uc2_to_canvas`
drops `z` because the tracer is 2D, not because the frames disagree. Once
`Scene3` lands, that drop is what goes away.

## Optiland

Optiland's nominal optical axis is `+Z` where Canvas uses `+X`:

```text
optiland.x = -canvas.z
optiland.y =  canvas.y
optiland.z =  canvas.x
```

```text
canvas.x =  optiland.z
canvas.y =  optiland.y
canvas.z = -optiland.x
```

Two things about this mapping are load-bearing.

**The negation is not cosmetic.** This is a signed axis permutation with
determinant `+1`, so it is a rotation. Mapping `canvas.z` to `optiland.x`
unsigned would be a bare swap of two axes, determinant `-1`, a reflection: every
design would arrive in Optiland mirrored, and the meridional cases that the
project validates today would all still pass, because a mirror is invisible in
the plane it reflects through. The sign is what keeps that from being a trap
waiting for the first skew ray.

**It preserves what already exists.** `canvas +x ↔ optiland +z`, transverse
`y ↔ y` is the convention [`seq.rs`](../../crates/oc-core/src/seq.rs) and the
whole [`validation/`](../../validation/) corpus already assume, and the README
publishes agreement numbers against. The 3D extension had to keep it, or every
committed cross-validation figure would move for a reason that had nothing to do
with physics.

The map is linear and orthogonal, so it applies unchanged to **points,
directions, and normals** alike. Both sides are in millimetres, so there is no
scale factor.

## Three.js

Three.js keeps its own convention (`+Y` up, `+Z` nominally forward), which is a
relabelling of the same right-handed frame:

```text
three.x = canvas.y
three.y = canvas.z
three.z = canvas.x
```

Unchanged from the mapping `oc-openuc2` already ships as `uc2_to_three`. It is a
cyclic permutation, so its determinant is `+1` too. PR 3D-01 makes `oc-openuc2`
call through the shared module rather than keep its own copy.

The canonical point `(100, 150, 55)` maps to Three.js `(150, 55, 100)`, which is
the value the openUC2 suite already pins.

## What is tested

In [`coordinates.rs`](../../crates/oc-core/src/coordinates.rs), and red until
PR 3D-01:

- each mapping sends the basis vectors exactly where the tables above say;
- `canvas +x ↔ optiland +z` and `y ↔ y`, so the meridional relationship holds;
- every round trip is **exact**, not within a tolerance. Both maps are signed
  axis permutations, so in f64 the only arithmetic is a sign flip and exactness
  is achievable. A tolerance here would hide an implementation that had quietly
  stopped being a permutation;
- **determinant is `+1`** for all four directions. Handedness is checked, never
  assumed;
- length and dot product are preserved, so a unit normal arrives a unit normal
  and an incidence angle survives the trip. This one is a tolerance, because
  permuting components reorders the sums inside `dot` and f64 addition is not
  associative. The claim under test is orthogonality, not a bit pattern.
