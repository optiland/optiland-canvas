# ADR 0002: Sequential prescriptions are extracted from the trace, not predicted

**Status:** Implemented, 2026-08-07
**Baseline:** `3d-09-slice-ui` at `eeeaa2f`

## Context

Sequential consumers — an Optiland emitter, a paraxial ladder, any tool that
wants "the ordered list of surfaces the beam meets" — have historically gotten
that list by *predicting* it: walk a declared component chain, assume the beam
follows it, unfold folds onto a straight axis, and police the assumption with
alignment tolerances and unsupported-case errors. The optikit `compile_path`
pipeline is the canonical example: flat folds are emitted as non-reflective
planes, pose residuals are squeezed into per-surface decenter/tilt, a curved
mirror in a folded path is `E_UNSUPPORTED`, and misalignment beyond 20°/25 mm
is a hard error. Meanwhile the reference tracer (§9) already knows exactly
where the beam goes, because it traces it — and a scene previewed with the
kernel could disagree structurally with the sequential model compiled from the
same document.

## Decision

The ordered surface chain is **derived from the trace**. Three pieces:

1. **An opt-in event log in the reference tracer**
   (`TraceConfig3::record_events`): each interaction records the
   `SurfaceRef` it happened on, the incident-opposing normal, the media on
   both sides, and the branches it spawned with their roles
   (transmit/reflect/TIR/thin-lens). Detector hits, aperture blocks, and
   escapes are logged too. Off by default; the render path is untouched.

2. **Sequence extraction** (`oc-trace::sequence3`): trace the source's
   **axial ray** (frame origin along local `+X`, highest-weight spectral
   line) with ghosts off, then walk the branch lineage that ends on the
   requested detector. Splitters follow both arms; the detector
   disambiguates. `SceneIntent::primary_paths` names source/detector pairs,
   so an intent is the natural query handle.

3. **A neutral wire format** (`oc-core::seq3::SequencePrescription3`,
   `oc-seq3` v1): hits in beam order with object/patch identity, world
   point and normal, action, media, exact arc lengths and OPL, plus a
   revision stamp and `SeqSystem`-style `notes` for everything the
   sequential idealization cannot express (arms not taken, probe passages,
   the ghost policy). Exposed as `sequenceJSON`/`sequenceForIntentJSON` on
   the wasm `Canvas` and `scene3_sequence(_for_intent)` in `oc-py`.

Failure is physical and typed, never heuristic: `NotReached` carries every
branch's actual fate ("lens → fold_mirror → escaped"), `Ambiguous` names the
recombining chains. There are no alignment tolerances and no unfoldable
cases — a folded path, a curved mirror inside it, and a beamsplitter arm are
simply what the ray did.

## Consequences

- A downstream sequential emitter reduces to a mechanical stamp: prescription
  × poses × surface data. The prediction layer (chain-walking, unfolding,
  tolerance policing) has no reason to exist, and preview/analysis divergence
  becomes structurally impossible — both views come from the same scene and
  the same tracer.
- The prescription is a cache keyed by `revision` (§3.3's stale-result rule):
  consumers holding a newer scene re-extract rather than trust.
- Extraction traces one ray with ghosts off, so its cost is microseconds; it
  can run on every scene edit.
- What this does *not* solve: a paraxial consumer still needs per-surface
  power and arc-length spacing (both present in the prescription), and
  field/pupil machinery downstream of a fold remains the consumer's problem.
- `flux_eps` during extraction is 1e-9, far below the render default: a
  legitimate chain through k splitters carries ~0.5^k of the launch flux, and
  culling it would fake a `NotReached`.
