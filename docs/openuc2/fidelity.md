# Fidelity: what this engine will and will not claim

The point of this page, and of every badge in the UI, is that **an optical layout
that looks right is not evidence that it is right**. Canvas will tell you what it
did not model rather than quietly approximate it.

## Readiness is four independent things

A module advertises four axes, and they never collapse into one status:

```yaml
capabilities:
  layout: ready        # absent | ready
  optical: validated   # absent | placeholder | idealized | validated
  cad: preview         # absent | reference | preview | manufacturing
  control: absent      # absent | configured | validated
```

A module can be perfectly good for layout and contribute nothing to the trace.
That is a normal, useful state, not a defect — an empty structural cube is exactly
that. The catalogue ingests every valid descriptor and makes the gaps visible;
incomplete modules are never hidden by default.

### What `optical` means

| Value | Means |
|---|---|
| `absent` | no Canvas optics. Placing it takes up grid space and changes no ray. |
| `placeholder` | geometry exists but stands in for something else. |
| `idealized` | it traces, but it is an idealization, not a measured part. |
| `validated` | compared against a reference model within a recorded tolerance. |

`validated` refers to the module **as the descriptor configures it**. Override a
parameter and that comparison no longer applies to your instance.

## Where the numbers come from

Every number the UI displays comes from the **f64 reference tracer**. The
interactive f32 path draws while you drag and settles to f64 the moment you let
go. This is a hard rule, and openUC2 mode does not relax it.

The detector histogram is **relative** flux per bin. It is not absolute
irradiance, and is not called that, because no source power calibration exists to
make it one.

## What Canvas does not model

Not "not yet configured" — **not implemented**. If your question depends on one of
these, this engine cannot answer it:

- **3D.** Canvas is a 2D meridional tracer. Skew rays do not exist here.
- **Multiple layers at once.** One horizontal plane is traced; other layers are
  drawn as context. A stacked system is displayed, not simulated.
- **Volume absorption.** Wired but inert: material absorption resolves to zero, so
  glass does not attenuate.
- **Wavelength-dependent surfaces.** There is no dichroic or filter model.
  Coatings are a scalar reflectance modifier.
- **Detector quantum efficiency.** Every wavelength is detected equally.
- **Polarization, diffraction, thin films, GRIN, tolerancing, thermal.**
- **Extended sources.** Point-fan and collimated only; no Lambertian emitters.
- Splines trace as polylines; asphere normals are finite-differenced.

### Filters, dichroics, and fluorescence

**Do not use this engine to make fluorescence or filter-efficiency claims.**

A dichroic needs wavelength-dependent reflectance and transmittance; an emission
filter needs both that and volume absorption; a fluorescence budget needs both
plus detector response. None of the three exists yet. Until the spectral-response
milestone lands, such modules can only be `layout-only` or `idealized`, and a
picture of coloured rays passing through something labelled "dichroic" would be
showing you geometry, not physics.

## Custom objects

You can draw your own optics beside catalogue modules. They are stored separately,
labelled `custom / unverified`, excluded from the BOM, included in the `.ocanvas`
export, and they never mutate the module descriptor. **A custom object corresponds
to no openUC2 product** — nothing in the catalogue will build it for you.

## Ports are advisory

The port graph reports what your *layout* says, not what the light does. Nothing
in it reaches the tracer: two modules whose ports disagree still exchange light
exactly as the geometry dictates. It exists to catch the layout mistake that the
picture and the numbers both hide — a cube one cell out, or mounted facing away.

It does not judge roles. An `output` meeting an `output` is reported as connected,
because Canvas light is reversible and a singlet run backwards is a singlet.

## The development fixture

Until the module repository's compiler (PR M1) is wired in, the browser ships a
**development fixture** catalogue: invented UUIDs, `Example Vendor`, `EXAMPLE-*`
SKUs, invented prices, and an all-zero source commit that reads as unpinned
everywhere a real catalogue carries a sha. The UI says so in red. Nothing in it is
an openUC2 product you can order.
