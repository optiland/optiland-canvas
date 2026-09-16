# Ghost-tracing performance — brainstorm

Status: lever 2 (the WebGPU preview) is implemented and measured; the other
levers are still analysis. Captured 2026-07-21 after the user observed severe
slowdown when enabling ghosts with ~7 lenses; measurements added 2026-09-16.

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

## Measured

The interactive (f32) projected trace in Chromium on an RTX 5070, median of five
runs, from `web/tests/webgpu.spec.ts`:

| Scene | Segments | CPU mirror | GPU | GPU segment budget |
|---|---:|---:|---:|---:|
| default singlet | 567 | 0.2 ms | 2.9 ms | 64 |
| 1 lens, ghosts | 1,755 | 0.7 ms | 3.6 ms | 64 |
| 2 lenses, ghosts | 6,534 | 3.8 ms | 3.2 ms | 64 |
| 3 lenses, ghosts | 16,632 | 14.1 ms | 8.5 ms | 128 |
| 4 lenses, ghosts | 31,968 | 35.3 ms | 17.4 ms | 256 |
| 5 lenses, ghosts | 51,618 | 70.4 ms | 36.5 ms | 512 |
| 6 lenses, ghosts | 77,670 | 143.6 ms | 70.2 ms | 1024 |
| 7 lenses, ghosts | 117,826 | 262.8 ms | 123.0 ms | 1024 |
| 7 lenses, no ghosts, 240 samples | 32,400 | 72.8 ms | 6.0 ms | 64 |

From one lens to seven with ghosts, segments grow 67 times and CPU time 375
times, so the cost per segment rises about 5.6 times (the one-lens timing sits
near the browser's timer resolution). That matches the O(surfaces) nearest-hit
loop running over 7 times the surfaces. Branch count and the per-step loop both
matter, and lever 5 is worth more than this note first assumed.

## Levers, ranked by leverage

### 1. Pairwise ghost enumeration for the common G=2 (the root-cause fix)
Real stray-light analysis is a **ghost image**: light reflecting off exactly two
surfaces before reaching the detector. Instead of a recursive
reflect-at-every-surface tree, **enumerate surface pairs (i, j) and trace only
those specific double-reflection paths → O(S²)** instead of exponential. This
matches how stray-light tools actually work and collapses the cost for the case
users care about. Highest-value structural change.

### 2. Wire the WebGPU tracer that already exists (done)
`oc-gpu`'s `gpu3` / `shader3.wgsl` (from PR 3D-08) now draws the browser's
interactive picture through `GpuPreview3` and `web/src/fastpreview.ts`, when the
browser has a WebGPU device. The shader marks a ray whose branch tree outgrew
its segment budget; the preview doubles the budget up to a 64 MB read-back, and
draws a tree that still does not fit with the CPU mirror rather than as a
thinner picture.

What the table says about it:

- A GPU frame costs about 3 ms before any work (upload, dispatch, read-back),
  so the preview stays on the CPU below 3,000 segments and moves to the GPU from
  5,000.
- Without ghosts the GPU wins by 12 times.
- With ghosts it wins by about 2 times, and 123 ms is still not interactive.
  The GPU time follows the budget column: every ray gets the bushiest ray's
  segment slots, and all of them are read back. Compacting the output on the
  GPU before the read-back is the next step for this lever.

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
O(log S) nearest-hit instead of O(S). The measurements above show the per-step
loop already costs about 5.6 times more per segment at 14 surfaces than at 2, so
this pays off sooner than expected, on both the CPU mirror and the shader.

## Recommended sequence

1. **Measured** (table above): both branch count and the per-step surface loop
   drive the cost.
2. **#2 WebGPU** is in. **#1 pairwise ghosts** remains the structural fix; GPU
   output compaction is what lets #2 keep its lead with ghosts on.
3. **#5 BVH**, sooner than first planned.
4. **#3 progressive/worker** for responsiveness; #4 culling as a cheap
   interactive win.
