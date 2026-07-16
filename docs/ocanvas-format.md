# The `.ocanvas` scene format

`.ocanvas` is the on-disk form of an Optiland Canvas [`Scene`](../crates/oc-core/src/scene.rs).
It is **plain JSON — human-readable, diffable, and git-friendly** (spec §6). It is
the contract other tools build on, so from M5 onward it is **frozen**: existing
fields keep their meaning, and any breaking change bumps the version.

## Stability policy

Every document begins with a two-field header:

```json
{
  "format": "ocanvas",
  "version": 1,
  ...
}
```

- **`format`** is always `"ocanvas"`. A reader rejects anything else.
- **`version`** is the schema version — currently **1**
  ([`OCANVAS_VERSION`](../crates/oc-core/src/scene.rs)).

The rules, enforced by [`Scene::check_version`](../crates/oc-core/src/scene.rs):

1. A reader accepts any document whose `version` is **≤** its own.
2. A document from the **future** (a higher `version`) is refused with a clear
   error rather than silently mis-parsed.
3. Documents written **before** the header existed have no `format`/`version`
   fields; they load as v1 (the fields are `#[serde(default)]`).
4. **Additive changes stay at the same version.** New optional fields carry a
   `serde` default, so an old file still loads and a new file still round-trips.
   Only a change that would break an existing reader increments `version`.

Writers always emit the header (`Scene::to_json` stamps it on every save).

## Top-level shape

| Field | Type | Meaning |
|---|---|---|
| `format` | string | `"ocanvas"` |
| `version` | integer | schema version (1) |
| `elements` | array | optical bodies (see below) |
| `sources` | array | light sources |
| `detectors` | array | line-segment detectors |
| `env` | object | ambient index + render wavelength set |
| `layers` | array | drawing layers (name / visible / opacity) |
| `catalog` | object | the glass catalog referenced by `Catalog` materials |

Units: **millimetres** for geometry, **micrometres** for wavelengths, radians for
angles. Everything is `f64`. The optical axis convention is canvas `+x` ↔ Optiland
`+z`, transverse `y` ↔ `y`.

### Elements

An element is a closed CCW loop of `boundary` edges with a bulk `material`, a
rigid `transform` (`{rot, pos}`), a `priority` (nesting resolution, §4.2), a
`locked` flag, and a `layer`. Edge variants:

- `Line { a, b }` — a straight segment (e.g. a ground lens rim).
- `Arc { center, r, a0, a1 }` — a circular arc swept CCW.
- `Conic { vertex, axis, c, k, half_ap }` — a conic-of-revolution meridian:
  curvature `c = 1/R`, conic constant `k`, clear semi-aperture `half_ap`.
- `Asphere { base, coeffs }` — a `Conic` plus even terms `Σ coeffs[i]·y^(2i+4)`
  (i.e. `r⁴, r⁶, …`).
- `Spline { ctrl, degree }` — the freeform escape hatch.

A lens is exactly what it looks like: a biconvex singlet is **one** element whose
boundary is `[front Conic, top Line, rear Conic, bottom Line]`.

### Materials

Tagged union: `Catalog(GlassId)`, `AbbePoint { nd, vd }`, `Mirror { reflectivity }`,
`Absorber`, or `Grin(field)`. `AbbePoint` disperses via the closed-form two-term
Cauchy fit from `(n_d, V_d)` (§3.1); `Catalog` glasses use full Sellmeier.

## The Optiland handoff

The rotationally-symmetric **sequential prescription** is a separate neutral form,
[`SeqSystem`](../crates/oc-core/src/seq.rs), used only at the Optiland seam:

- [`Scene::to_seq`] extracts it from a scene. It is **exact for the axis-aligned
  refractive subset**; every feature with no sequential analogue (splines, arcs,
  GRIN, absorbers, tilted or off-axis surfaces, point sources) is recorded in
  `SeqSystem.notes` — the fidelity report — instead of being silently dropped.
- [`SeqSystem::to_scene`] rebuilds a scene (one element per glass gap).

`SeqSystem` is JSON too, and it is what the [`oc-py`](../crates/oc-py) bindings
exchange with the Optiland `Optic` builder/reader. Round-tripping a prescription
through a scene — or through an Optiland `Optic` — is parameter-stable; that is
asserted in `oc-core`'s `seq` tests and in `oc-py`'s round-trip corpus.
