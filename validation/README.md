# Forward-trace & wavefront validation against Optiland

Cross-checks the `oc-trace` f64 reference tracer against the main
[Optiland](https://github.com/optiland/optiland) engine (the 3D sequential ray
tracer this canvas hands designs off to). Matched rotationally-symmetric systems
are traced in both engines and the final ray **position**, **direction**, and
**optical path length** are compared meridional-ray by meridional-ray.

The OPL check is what validates the M4 wavefront scrub: a wavefront is a surface
of constant optical path, so the wavefront aberration is the **OPD** — each ray's
OPL to the image plane relative to the axial ray. The canvas accumulates
`Σ nᵢ·dᵢ` (`Segment::opl0/opl1`); Optiland accumulates the same in
`RealRays.opd`. Taking OPL relative to the axial ray cancels the reference
tracer's deliberate 1e-7 mm ray-spawn offset (a wavefront is defined only up to a
constant piston), so the `d_OPD` column agrees to ~1e-14 mm on spheres and
~1e-8 mm on the even asphere (whose normal is finite-differenced). The
self-contained `cargo test -p oc-trace --test wavefront` bakes two of these
Optiland OPD sets as golden values.

Axis convention: canvas `+x` ↔ Optiland `+z`, transverse `y` ↔ `y`. Geometry and
interface physics are isolated from glass-catalog differences by tracing at the
d line, where a canvas `AbbePoint{nd, vd}` returns exactly `nd`; this is paired
with Optiland's `IdealMaterial(nd)`. Dispersion is checked separately with real
N-BK7.

## Files

- `systems.json` — the shared spec (systems + launch rays), read by both sides.
- `optiland_trace.py` — traces the systems through the real Optiland engine
  (injects `RealRays` into a built `Optic`'s `surface_group`), emits JSON.
- `compare.py` — ray-by-ray diff of the two JSON outputs.

The canvas side is the `audit_compare` example in `crates/oc-trace`.

## Run

```sh
# canvas (Rust) side — from the repo root
cargo run -p oc-trace --example audit_compare -- validation/systems.json > canvas_out.json

# Optiland (Python) side — requires optiland installed (e.g. `pip install optiland`
# or `uv run` from an Optiland checkout)
python validation/optiland_trace.py validation/systems.json > optiland_out.json

# ray-by-ray diff
python validation/compare.py canvas_out.json optiland_out.json
```

## Expected result

Direction agrees to machine precision (~1e-16) on spheres, planes, conics,
doublets, the mirror, and N-BK7 dispersion; position to ~1e-15 after removing
the reference tracer's deliberate 1e-7 mm ray-spawn offset. Even aspheres agree
to ~1e-9 because both engines use an independent Newton solver seeded by the
analytic conic (the reference finds the intersection point to machine precision
but takes its surface normal by finite difference). The N-BK7 Sellmeier index is
bit-identical between the two.
