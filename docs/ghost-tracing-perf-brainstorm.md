# Ghost-tracing performance — brainstorm

Status: **analysis only, nothing implemented.** Captured 2026-07-21 after the user
observed severe slowdown when enabling ghosts with ~7 lenses.

## Why it explodes

A ghost is a Fresnel back-reflection **branch** spawned at *every* surface a ray
crosses. So the number of ray paths grows roughly **exponentially** in
(surfaces × G), where G = `max_ghost` generations:

- 7 lenses ≈ 14 refracting surfaces (2 conic faces each).
- Depth cap `auto_max_depth = surfaces·(G+1) + 16`, clamped `[32, 1024]`.
- Each ray step does a **brute-force O(surfaces) nearest-hit** — there is no BVH
  (noted as a deliberate future optimization in 3D-08).
- The wasm **f64 reference tracer is serial** (no rayon in wasm) and re-traces the
  whole exploded tree **synchronously on the main thread** on every edit → the UI
  jank you feel.
- `flux_eps = 1e-4` culls dim branches, but a 2-surface ghost is ~0.04² ≈ 1.6e-3,
  which survives — and with many surfaces, many such ghosts survive.

## Levers, ranked by leverage

### 1. Pairwise ghost enumeration for the common G=2 (the root-cause fix)
Real stray-light analysis is a **ghost image**: light reflecting off exactly two
surfaces before reaching the detector. Instead of a recursive
reflect-at-every-surface tree, **enumerate surface pairs (i, j) and trace only
those specific double-reflection paths → O(S²)** instead of exponential. This
matches how stray-light tools actually work and collapses the cost for the case
users care about. Highest-value structural change.

### 2. Wire the WebGPU tracer that already exists
`oc-gpu`'s `gpu3` / `shader3.wgsl` (from PR 3D-08) is written and parity-tested —
it hit ~0.38 ms for 2k rays × 8 depth × 32 objects on an RTX 5070 — but it is
**not connected to the browser preview** (the browser still uses the CPU path).
Wiring it in (with a CPU fallback) is the **biggest speedup for the least new
algorithm work**: rays are embarrassingly parallel and the GPU eats the exploded
tree.

### 3. Progressive / deferred ghosts, off the main thread
- Trace **primaries first**, paint them instantly; compute ghosts in a **Web
  Worker** or `requestIdleCallback` and composite in a follow-up frame.
- **Debounce** the full trace during rapid edits: primaries-only while dragging,
  full ghost tree on settle. (The f32 `traceProjected3DFast` path already
  half-does the fast-during-drag / f64-on-settle split.)
- Kills perceived latency and the freeze even before reducing total work.

### 4. Contribution-based ghost culling
The picture is HDR + Reinhard-tonemapped, so ghosts below a visibility floor never
show. Cull by **contribution-to-pixel**, not just raw `flux_eps` — expose it as a
"ghost quality" knob for the interactive preview while the settled reference keeps
the full set.

### 5. BVH over surfaces
O(log S) nearest-hit instead of O(S). Modest at 14 surfaces, but it compounds with
the exploded path count and scales as designs grow.

## Recommended sequence

1. **Measure first**: read the stat-line `trace ms` for the 7-lens + ghost case to
   confirm branch-count (not the per-step surface loop) dominates.
2. **#1 pairwise ghosts** + **#2 wire WebGPU** as the structural fixes.
3. **#3 progressive/worker** for responsiveness.
4. #4 culling as a cheap interactive win; #5 BVH later as scenes scale.
