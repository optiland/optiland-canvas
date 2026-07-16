# Module optics

A module's optics come from one of two places, never both:

- a **stored fragment** — an ordinary `.ocanvas` scene at `optics/canvas.ocanvas`
  in the module repository;
- a **typed generator** — the module names a builder and some numbers, and its
  geometry is built when it is placed.

Either way the result is a **local fragment**: geometry in the module's own frame,
with the anchor at the origin, and nothing about where the module ends up baked
into it.

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

## Generated optics

A module can build its optics from numbers instead of shipping a scene:

```yaml
generator: singlet
parameters:
  static:
    material: N-BK7
    diameter_mm: 24.0
    center_thickness_mm: 5.0
    radius_1_mm: 100.0
    radius_2_mm: -100.0
  variable:
    radius_1_mm: { min: -1000.0, max: 1000.0 }
```

Change `radius_1_mm` on a placed instance and its glass is rebuilt; the rays
follow. `variable` is what a UI may offer; only parameters the generator actually
reads are editable, because a field that changes nothing is worse than no field
(§15.7).

**Typed, not scripted.** §9.5 forbids executing anything a descriptor carries: no
scripts, no expressions. A descriptor cannot say *how* to build its optics, only
*which* builder and with what numbers. The set of builders is closed, lives in
[`generator.rs`](../../crates/oc-openuc2/src/generator.rs), and is reviewed code.
A descriptor naming a builder this build does not have is an error, never an
attempt.

**A generated instance stops being `validated`.** A validation compares a
*specific configuration* against a reference. Change a radius and that comparison
is about a different lens, so the instance reports `ReconfiguredOptics` instead of
the module's badge. The geometry is still built by reviewed code and still traces
correctly — what is gone is the evidence that it matches anything.

### The singlet's sign convention

`radius_1_mm` and `radius_2_mm` are the **standard prescription convention**, the
one Optiland and a lens datasheet use: a radius is positive when the centre of
curvature lies on the `+X` side of the vertex. A biconvex singlet is `R1 > 0,
R2 < 0`; a plano-convex is `R1 > 0, R2 = inf`.

Worth stating because `oc-wasm`'s freehand singlet inspector uses a *different*
one — it builds its rear surface on a reversed axis, so there a positive `radius2`
is a convex rear. Both are internally consistent. They are not interchangeable,
and a descriptor is a prescription, so it gets the prescription convention.

The generator refuses what is not a lens: a zero radius (a point, not a surface —
use `inf`), a radius shorter than the semi-diameter (the sag's square root goes
imaginary, and a clamped shape traces happily while being no sphere at all), and a
centre thickness too thin to keep the surfaces from crossing.

### A plane radius has to be a string

**JSON has no infinity.** `json!(f64::INFINITY)` serializes to `null`, and
Python's `json.dump(float("inf"))` writes a bare `Infinity` that is not JSON and
that `serde_json` refuses. A plane is an ordinary surface — §9.6's own pilot
descriptor is plano-convex — so a catalogue compiler must emit `"inf"`, and the
generator reads that back.

### Checking it

Each singlet reports the lensmaker's equation for its current numbers:

```text
1/f = (n-1) [ c1 - c2 + (n-1) t c1 c2 / n ]
BFD = f [ 1 - (n-1) t c1 / n ]
```

computed **without the tracer**, at the d-line, from the same Sellmeier
coefficients the tracer refracts with. The rays are then an independent second
answer:
[`singlet_physics.rs`](../../crates/oc-openuc2/tests/singlet_physics.rs) traces
generated lenses at several radii and asserts the beam crosses the axis within
0.05 mm of the prediction — and asserts that a rim ray misses it by *more*, which
is spherical aberration being real rather than the prediction being fudged.

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
