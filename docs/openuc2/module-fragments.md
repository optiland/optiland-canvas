# Module fragments

A module's optics are an ordinary `.ocanvas` scene, stored at `optics/canvas.ocanvas`
in the module repository and interpreted as a **local fragment**: geometry in the
module's own frame, with the anchor at the origin, and nothing about where the
module ends up baked into it.

This page is about how that fragment gets into a scene, and the two id hazards
that come with it. Both are hazards of the same kind: **`oc-core` ids are vector
positions, not identities.**

## What a fragment carries

```rust
pub struct SceneFragment {
    pub elements: Vec<Element>,
    pub sources: Vec<Source>,
    pub detectors: Vec<Detector>,
    pub glasses: Vec<Glass>,   // the tables its ids point into
    pub layers: Vec<Layer>,
}
```

A fragment carries its **own glass and layer tables**, because its
`Material::Catalog(GlassId)` and `Element::layer(LayerId)` are positions in
*those* tables. A full `.ocanvas` is read as a fragment by discarding its
document-only state: viewport, zoom, selection, history, and ambient medium. The
destination scene's ambient is authoritative.

Local identity is vector order: `elements[i]`, `sources[i]`, `detectors[i]`.

## Hazard 1: scene-table ids are positional

Fragment `GlassId(2)` may mean N-SF11 while destination `GlassId(2)` means N-BK7.
Copy the id across unchanged and the lens still parses, still draws, still traces
— **as the wrong glass**. Nothing about the file looks broken.

So insertion never copies a table id. It:

1. validates every referenced `GlassId` and `LayerId`;
2. merges **only referenced** glasses and layers into the destination;
3. remaps every inserted element to the merged destination ids;
4. commits only once all fallible work has succeeded.

Glasses merge by case-insensitive name. A name that matches with **identical
authoritative Sellmeier coefficients** reuses the destination id; a name that
matches with *different* coefficients is a typed `GlassConflict` that changes
nothing. The merge never picks a winner. Derived caches like `nd`/`vd` do not
decide identity when they are computed from the coefficients.

Layers merge by exact name and are presentation only; on a collision the
destination's visibility and opacity win.

## Hazard 2: removal renumbers survivors

Removing objects from vector-backed storage slides every later id down. An owner
holding a stale id now points at **another module's geometry** — so deleting one
module makes the next one move the wrong lens.

```rust
pub fn remove_objects(&mut self, ids: &ObjectIdSet) -> Result<IdRemap, FragmentError>;
```

`IdRemap` maps every pre-removal id to its post-removal id, or `None` if it was
removed. **A caller that discards it is incorrect.** `oc-openuc2` routes every
destructive scene operation through one helper that pushes the remap into every
retained id: each surviving instance's `InsertedFragment`, the custom batch, the
reverse ownership map, and the selection.

Insertion is the mirror image, and returns the local-to-scene map in fragment
order:

```text
fragment.elements[i] -> inserted.elements[i]
```

Semantic keys (`<instance-id>::element::<local-index>`) are built and owned by
`oc-openuc2`. They are never passed into `oc-core`, which takes no string
namespace: numeric ids are the core's business and product identity is the
adapter's.

## Placement is transform composition

```rust
pub type Transform2 = Isometry2;   // outer.compose(inner) = outer after inner
```

- `Element` and `Source` already carry an `Isometry2`: placement composes onto it.
- **Boundary edges are never rewritten, re-fit, or resampled.** One path covers
  line, arc, conic, asphere, and spline geometry exactly, because none of it is
  re-derived.
- `Detector` stores bare endpoints — the one geometry mapped directly.
- Only rigid translation and rotation. Scale and shear are unrepresentable.
- `Material::Grin` is **rejected** with a typed error: its world-space
  axis-aligned box cannot represent a rotated field, and silently placing it
  would be a lie.

Batch insert, remove, and transform are atomic: everything is checked before
anything mutates, and on error the scene is unchanged in every observable field.

## Full-turn arcs

`Edge::Arc { a0, a1 }` reduces its sweep modulo `TAU`, so a naive full turn
(`a0 = 0`, `a1 = TAU`) once collapsed to a **zero sweep** — a circular aperture
that admitted hits near one angle and looked plausible while tracing wrongly.

Since the C1b fix:

- `abs(a1 - a0) >= TAU - ARC_ANGLE_EPS` is a full circle for intersection and
  containment;
- `a1 == a0` remains a zero sweep and is **not** implicitly a circle;
- ordinary wrapped CCW arcs are unchanged.

**Minimum compatible Canvas version for one-edge circular fragments: the build
containing C1b.** Against an older build a one-edge full circle traces as a point,
so the catalogue validator rejects that representation rather than silently
rewriting it. Authoring two semicircular arcs instead is legitimate and always
was; a compiler that rewrote your circle would be hiding a core semantics
mismatch, so it does not.

No part of any of this changes `.ocanvas` v1. `SceneFragment`, `InsertedFragment`,
`IdRemap`, and `Transform2` are API abstractions, not file fields.
