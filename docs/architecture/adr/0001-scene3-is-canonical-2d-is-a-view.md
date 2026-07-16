# ADR 0001: `Scene3` is canonical; 2D is a view over it

**Status:** Accepted, 2026-07-16
**Baseline:** `main` at `bc44b8b`
**Supersedes:** the README roadmap line listing 3D as out of scope

## Context

Canvas is not a 3D tracer with a 2D front end. At the baseline it is
**physically two-dimensional all the way down**, and that is visible in the
types rather than hidden behind them:

| Area | Baseline |
|---|---|
| core vector | `Vec2 = glam::DVec2` |
| transform | `Isometry2 { rot: f64, pos: Vec2 }` |
| geometry | closed loops of 2D `Edge` values |
| region resolution | 2D point-in-element crossing count |
| detector | a line segment |
| source | a fan whose measure exists only in a plane |
| GPU ray | `ox, oy, dx, dy` |
| openUC2 | defines XYZ, then drops Z at the Canvas boundary |

This was a good decision and it bought what it promised. For a rotationally
symmetric system the meridional section really is the design, 2D really is where
sub-16 ms interactivity is tractable, and the engine really is validated against
Optiland to machine precision on the meridional subset.

What it cannot do is represent skew rays, sagittal behaviour, out-of-plane
folds, finite 3D apertures, or area detectors. Those are not missing features
around the edges. They are most of what "non-sequential" means once a layout
stops being a single plane, and openUC2 layouts stop being a single plane as
soon as there is a second layer.

Two forces made this a decision that had to be taken now rather than later:

1. **openUC2 already has a third dimension.** `oc-openuc2` carries real XYZ port
   poses and a 55 mm layer pitch, and then throws `z` away at
   `uc2_to_canvas`. A stacked system is displayed, not simulated
   (`docs/openuc2/fidelity.md`). The adapter is holding data the engine cannot
   accept.
2. **A second production tracer was forming.** Optiland has a draft Python
   non-sequential tracer. Two independent production implementations of
   nearest-hit transport, branching, absorption, and surface interaction will
   diverge, and they will diverge silently, because nobody diffs two engines
   they both trust.

## Decision

**The canonical physical model is three-dimensional. The 2D Canvas is a view
over it.**

1. The canonical persisted scene is 3D (`Scene3`, `.ocanvas` v2,
   `dimension: 3`). Every 2D view is derived from it, never the reverse.
2. The current 2D tracer stops being the definition of the optical system. It
   survives as exactly two things, and is labelled as whichever it is:
   - an **exact planar reduction** of a 3D scene, carrying a `ReductionProof`
     that records the checks that made the reduction legal; or
   - a **legacy compatibility path** for `.ocanvas` v1 documents whose missing
     third dimension has not been resolved yet.
3. **One production non-sequential kernel**, in Rust. Optiland may provide scene
   builders, analyses, reports, optimization, and plotting around it. It does
   not maintain a second production implementation of transport.
4. **No invented geometry.** Loading a legacy 2D body does not assume it is
   revolved, extruded, infinitely deep, or one millimetre thick. It loads as
   `PlanarOnly` and a full-3D trace returns a typed capability error until
   somebody promotes it on purpose.
5. **Fidelity is reported, never faked.** A backend that cannot do something
   rejects the scene with a capability issue. It does not substitute a
   superficially similar surface and stay quiet.
6. **f64 3D is authoritative.** f32 CPU and GPU never supply a displayed number.
   This is the existing two-tier rule, unchanged.

## Consequences

**What this buys.** Skew rays, tangential/sagittal fans, out-of-plane folds,
finite 3D apertures, and area detectors with real `mm²` cells become
representable. openUC2 stops discarding `z`. There is one kernel to validate
rather than two to reconcile, and Optiland's independent *sequential* tracer
stays available as a genuine oracle, which is worth more than a second
non-sequential implementation would have been.

**What it costs, stated plainly:**

- `.ocanvas` v2 is a **breaking schema change**, on a format frozen since M5. v1
  stays readable through an explicit migration layer; v1 *writing* stops being
  the default.
- Every crate is touched. This is a migration, not a feature.
- The fast planar path now has to **earn its place** with a proof. Where the
  proof fails, the fast path is not available, and the honest answer is a
  slower one.
- Legacy v1 files load, display, and trace in compatibility mode, but **cannot
  be traced in full 3D** until each body is promoted. That is a deliberate
  cliff. The alternative is inventing a depth nobody authored.
- The migration is 11 further PRs. No intermediate PR may remove the current
  browser experience before its replacement is green (§3.6).

**Knock-on decisions this forces:**

- The README's "Out of scope: 3D" is reversed. The rest of the README's 2D
  claims stay accurate until the 3D path actually lands, and are rewritten in
  PR 3D-11, not before.
- Spectral surface R/T and volume absorption (the openUC2 spec's P1, the gate on
  any honest filter or dichroic claim) now **target the 3D path first**. A 2D
  implementation alone is no longer sufficient (§19).
- Stable 64-bit ids replace vector-index identity (§3.8). This structurally
  removes the stale-`SurfaceRef` hazard that `optiland-canvas-handoff-specs`
  PR1 exists to patch.

## Alternatives considered

**Keep Canvas 2D and let Optiland's Python tracer own 3D non-sequential.**
Rejected. It produces two production implementations of the same physics, and
the one users actually watch is the fast one. Divergence would be discovered by
a user, not by CI. It also leaves openUC2 permanently unable to simulate a
stacked system.

**Trace 3D under the existing 2D API and infer the third dimension.** Rejected.
Inferring depth from a 2D profile is invented geometry by definition: a drawn
outline does not say whether it is a revolved lens or an extruded prism, and the
two trace differently. The construction rule has to be authored, so it has to be
in the schema, so the schema breaks. Doing it implicitly buys nothing and hides
the break.

**Big-bang rewrite to 3D.** Rejected. The planar performance gate and the
"no Run button" contract are the product. They must stay green on every
intermediate commit, which means the 2D path and the 3D path coexist behind
feature flags until the 3D one passes both the acceptance suite and the
performance gate (§12.1).

**Triangle meshes as the universal geometry.** Rejected for the first reference
release. Watertightness and containment for meshes is its own specification, and
containment is load-bearing here: media are resolved by containment plus
priority, so a leaky mesh is not a rendering artefact but a wrong refractive
index (§6.2).
