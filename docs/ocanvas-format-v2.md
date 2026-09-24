# The `.ocanvas` v2 scene format

v2 is the on-disk form of a [`Scene3`](../crates/oc-core/src/scene3.rs): the
canonical **3D** scene. It is plain JSON, human-readable, diffable, and
git-friendly, exactly as [v1](ocanvas-format.md) is.

**v2 is a breaking change**, and the first one since the format was frozen at M5.
Why it had to break is [ADR 0001](architecture/adr/0001-scene3-is-canonical-2d-is-a-view.md);
what it costs is below.

**Status: live.** A `Scene3` traces, and the browser holds one. It opens both v1
and v2, and it saves v2, falling back to the v1 writer only when no 3D scene is
loaded. v1 stays readable, and the v1 export stays for a consumer that needs it.

The type is still spelled `Scene3` below. Renaming it to `Scene`, with the 2D
types moving under `legacy_planar`, is the rest of PR 3D-11 and has not happened.

## The header

```json
{
  "format": "ocanvas",
  "version": 2,
  "dimension": 3
}
```

`dimension` is new and always `3`. A v2 document claiming `2` is malformed rather
than interesting: v2 *is* the 3D schema, and a reader that accepted a 2D one
would have to guess what the rest of the document meant.

The v1 rules still hold. A reader accepts any `version` up to its own and refuses
one above it, so a v1 reader meeting a v2 file says so instead of mis-parsing it.
That is the existing contract working as designed, not a regression.

## Why it breaks

Because a drawn outline does not say what it is.

The same closed curve is a revolved lens or an extruded prism depending only on
what the author meant, and the two trace differently. v1 never recorded the
difference because a 2D tracer never had to ask. A 3D one cannot avoid it, and no
amount of reading a v1 file will answer it. So the construction rule has to be
stored, which means the schema has to change, and doing that implicitly would
hide the break rather than avoid it.

## Loading a v1 document

```text
parse the header
  version 1  -> parse as v1, migrate, return a MigrationReport
  version 2  -> parse as v2
  otherwise  -> refuse
```

[`load_any`](../crates/oc-core/src/migrate_v1.rs) does this. Every v1 document
loads, and **nothing gains a dimension it did not have**:

| v1 | v2 | Why not more |
|---|---|---|
| element | body with `PlanarOnly { plane_z_mm: 0 }` | the file never said revolved or extruded |
| detector (line segment) | `LineProbeLegacy` | a line has no area, so its bins have no `mm²` |
| `Collimated { half_width }` | `LineLegacy` spatial + `Direction` angular | a v1 collimated source is a line, not a rectangle |
| `Point { half_angle }` | `PlanarFanLegacy` angular | a v1 fan is equal-angle in one plane, not uniform in solid angle |
| `Isometry2` | `Isometry3`, translation `(x, y, 0)`, rotation about `+Z` | exact: v1 is the `Z = 0` slice |
| materials, wavelengths, layers, catalogue | copied | not recomputed, so no `f64` moves |

These trace exactly as they did in v1. A **full-3D** trace refuses them with a
typed capability issue until each body is promoted on purpose. That cliff is
deliberate. Guessing is not a slightly wrong answer; it is a different object
that still renders beautifully.

The migration is deterministic: ids are allocated in a fixed traversal order, so
the same v1 file always produces the same v2 file, byte for byte.

### Promotion

```rust
promote_body(&mut scene, body, ProfileConstruction3::RevolveLocalX)?;
```

The only way a `PlanarOnly` body becomes 3D. There is no automatic path.

## Identity

**Ids are not vector indices.** In v1 they are: `scene.rs` says so, and
`add_element` hands out `ElementId(len)`. Every persistent reference is therefore
a position, every insertion or removal moves it, and a mutator that forgets to
renumber leaves a reference resolving to a *different object*, silently, with the
picture still looking right.

v2 ids are allocated from one `u64` counter, are never reused while the scene
lives, and are unaffected by vector order. Vector order is presentation and
serialization order only. Removing an object leaves every survivor's id alone, so
there is nothing to remap.

One id space covers everything: bodies, patches, surfaces, apertures, detectors,
sources, views, path intents. So no two objects in a document share a number, and
`DuplicateId` is a single flat check.

`next_id` is persisted. On load it is lifted above the highest id in use, which
is the one thing this format repairs rather than refuses: a `next_id` below the
high-water mark is not a statement of intent, it is bookkeeping, and refusing a
document whose objects are all fine would help nobody.

## Enums are externally tagged

```json
{ "Mirror": { "reflectivity": { "Constant": 0.96 } } }
```

Same as v1 (`{"Collimated": {"half_width": 6.0}}`), and not a style choice.
serde's internally-tagged form cannot serialize a variant whose payload is a bare
float, which `SpectralScalar::Constant(f64)` is, so an internally-tagged v2 would
have failed on the first save of a scene with a constant reflectivity.

## Floats round-trip exactly

A number that goes out comes back as the *same* `f64`, not a near one. This is
inherited from v1 and asserted the same way, over subnormals, `MIN_POSITIVE`, the
largest normal, and values whose shortest decimal form is not their exact one. It
needs `serde_json`'s `float_roundtrip` feature, which the workspace enables, and
it is not free: without it a value can come back one ULP off.

## Fields added after v2 shipped

A field added to v2 carries a serde default that reads as what documents
written before it meant, so no file is invalidated by the kernel growing. The
default is the old behaviour, never a new one:

| field | on | absent means | added |
|---|---|---|---|
| `measure` | detector | `Power`, flux per cell, which is what a detector read before there was a choice | [what a detector measures](detectors.md) |
| `solar` | scene | no sun, and the key is left out on save | [the sun and heliostats](solar.md) |

A variant added to an enum cannot default, so it only reads in a kernel that
knows it. `Tube` (a hollow tube about local `+X`: `inner_radius_mm`,
`outer_radius_mm`, `half_length_mm`) is one: a document containing a tube
does not open in a kernel older than the tube. A tube's faces carry per-patch
overrides by fixed patch ids: 0 outer wall, 1 the `+X` end ring, 2 the `-X`
end ring, 3 the bore.

A new variant is the other way the schema grows, and an old document never
contains one. `Curve2::SagSegment { base, coeffs, y0, y1 }` is an annular piece of
an on-axis conic or even asphere, between signed profile heights `y0` and `y1`;
`base.half_ap` is not read. It is what a Fresnel lens's facets are
(`oc_core::fresnel_lens`), and a Fresnel lens stores no recipe beside its
profile: the recipe is read back out of the edges. A reader that predates the
variant refuses such a document rather than dropping the facets.

## What a document may not contain

Validation runs on load and refuses, rather than repairs:

- a rotation that is not a rotation (a zero or non-finite quaternion). An
  unnormalized but finite one is normalized, since its length carries no meaning
  a document could have intended, and leaving it would scale the body by `|q|²`;
- a duplicate id, anywhere in the one id space;
- a dangling reference: a layer, patch, source, detector, aperture, or path
  intent that is not in the document, or a sun rig naming a source, mirror or
  target that is not in it;
- a non-finite geometric value;
- an ambient index that is not strictly positive;
- a wavelength table that is not a function: fewer than two samples, mismatched
  lengths, duplicate or decreasing wavelengths, or a non-finite sample. Two
  values at one wavelength make the table a relation, and which one you get would
  depend on how the search happened to land;
- a reflectance, transmittance, AR residual, or quantum efficiency outside
  `[0, 1]`;
- `R + T > 1`: the remainder is absorbed, and a surface returning more than it
  received is a source;
- negative absorption, which is gain.

The rule is one line: **a document that contradicts itself has an author whose
intent is unknown, and guessing it is how a scene starts lying.**

## What v1 loses

One field: `SurfaceSpec::roughness`. v1 reserved it for scatter and never read
it, so carrying it into v2 would let the schema imply a model that does not
exist. It is dropped **and reported**, because silently losing an authored number
is the other half of the same mistake.

`Spline { ctrl, degree }` becomes `Polyline { points }`. Not a loss: v1 evaluated
it as a polyline through its control points, so `degree` was read by nothing and
the name promised curvature the tracer never delivered. The rename changes no ray
and stops the schema making a claim the physics does not. A real B-spline arrives
with the PR that implements B-spline intersection, or not at all.

## Fixtures

`crates/oc-core/tests/data/ocanvas-v2/` holds the committed schema fixture. It is
**generated**, not hand-written:

```sh
cargo run -p oc-core --features scene3 --example emit_v2_fixture \
  > crates/oc-core/tests/data/ocanvas-v2/mirror-migrated.ocanvas
```

A test re-derives it and compares. A fixture nobody regenerates drifts away from
the schema it exists to pin, and keeps passing while describing something the
writer no longer emits.
