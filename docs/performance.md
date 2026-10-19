# Performance

A gallery lens with twenty elements and the ghost layer on traces millions of
ray segments. Every one of them is physics: each surface reflects a few percent
of what reaches it, and with two ghost generations every pair of surfaces makes
a ghost path through the whole lens. The app used to trace all of that on the
page's one thread, so the page froze for seconds after every edit, and for the
largest designs it never came back.

None of the changes below alter a number. The exact trace still follows every
branch, and every speedup is checked bit for bit against the code it replaced.
What changed is where the work runs, how much of it is repeated, and how much
of the picture is drawn at once.

## The reference runs in a pool of workers

The f64 reference is the trace every displayed number comes from. It runs in a
pool of Web Workers ([`web/src/reference.ts`](../web/src/reference.ts),
[`web/src/reference-worker.ts`](../web/src/reference-worker.ts)): one fewer
than the machine's cores, one fewer than its memory in GB
(`navigator.deviceMemory`), and at most eight. Each worker loads its own copy
of the scene.

The trace is cut into 64 blocks of launch rays (`oc_trace::TRACE_BLOCKS`).
The page hands blocks to whichever worker is free. A worker traces its block a
few milliseconds at a time and returns to its event loop between batches, so
an edit reaches it within one batch and the stale trace is dropped. Each block
comes back as two things: its share of the picture, and its numbers (detector
cells, hit records, the energy ledger, the capability report) encoded as a flat
f64 buffer (`TracePart3::encode`). One worker adds the numbers up in block
order (`TraceMerge3`) and puts the picture together.

The blocks do not depend on how many workers there are, and neither does the
order they are added in. The reference is therefore the same, to the bit, on a
machine with one core or sixteen, and the same one worker tracing it whole
gives (`TraceJob3`). `crates/oc-trace/tests/trace_job3.rs` holds the tracer to
that: three workers trace the blocks out of order, the parts go through the
encoding, and the merged result matches the whole trace in every sum, cell,
hit record and report entry, and in the order of its segments.

While a reference is traced, the numbers on screen are the last reference's,
dimmed and marked stale, and the status line says how far the new one has got.

## The picture while the reference is traced

The picture that stands in until the reference lands is the fast f32 trace.
Three backends can draw it ([`web/src/fastpreview.ts`](../web/src/fastpreview.ts)):
WebGPU, the CPU mirror on the page's thread, and the workers, each tracing a
share of the launch rays (`Canvas.traceProjected3DFastShare`; a ray's segments
do not depend on any other ray, so the shares one after another are the whole
frame, `crates/oc-gpu/tests/cpu3_shares.rs`).

The GPU is not always the fastest. A ghost-lit lens's ray trees are deep and
uneven, and every ray gets the bushiest ray's segment budget. Measured in
Chromium, one frame of the preview with ghosts on:

| Design | Segments | CPU, one thread | RTX 5070 | Radeon iGPU (RDNA 2) |
|---|---:|---:|---:|---:|
| Double Gauss | 129k | 84 ms | 47 ms | 100 ms |
| Objective 60x | 334k | 286 ms | 244 ms | 540 ms |
| UV projection lens | 786k | 610 ms | 1.8 s, then gives up | 2.4 s, then gives up |
| US objective | 1.5M | 1.3 s | 0.64 s, then gives up | 2.5 s, then gives up |

A GPU frame also holds the GPU for as long as it runs. So the page never
traces a frame that would take it more than 50 ms: the workers do. A ghost
setting whose GPU frames take over 100 ms twice running, whose ray trees the
GPU cannot hold, or whose CPU frame costs over 400 ms, is drawn by the workers
until another design loads. The workers report what their shares cost, so the
page learns what a CPU frame of the scene would cost without tracing one.

One stand-in is traced at a time. While a lens is dragged, the next position
waits for the one in flight rather than superseding it, so the rays follow
the drag a few times a second instead of freezing until the pointer stops.

## Drawing large pictures

- **2D view.** Every picture is rasterized about 12 ms per animation frame.
  Its tonemap, which lights up to every pixel on screen, runs a band of rows
  per frame too. The last complete picture stays on screen, moved to the
  current view, until the next is whole (or for a quarter of a second, then
  the new one shows as it develops). The ray layer reads the projected picture
  in place and clips each segment to the slab as it draws it, and draws one TIR
  ring per spot, once.
- **3D view.** The ray picture is held in chunks of half a million segments,
  one GPU buffer each: WebGPU refuses a buffer over 256 MB unless the device
  asks for more, and before the chunks the largest ghost pictures drew no rays
  at all. A picture of more than one chunk is written and uploaded two chunks
  per frame, through two arrays the chunks share on the page's side, so the
  page holds 64 MB of vertices rather than all of them. The buffers use
  three.js's static usage: with the dynamic usage the view had before,
  three.js 0.186 wrote every ray buffer whole to the GPU on every frame, even
  while the camera merely turned.
- **WebGPU preview read-back.** The GPU's segments are packed into the
  picture's layout straight from the read-back, not through per-ray lists
  first.

## A picture budget, not a numbers budget

Neither view draws a segment dimmer than 10⁻⁵ of the brightest launch ray, so
the reference does not store one (`TraceConfig3::segment_floor`). A Double
Gauss with ghosts traces 2.0 million segments and keeps 543 thousand. The
ledger, the detectors and every other number still count every branch, and
`TraceResult3::segments_omitted` says how many segments were left out.

## Faster tracers, same bits

- **Nearest event.** The f64 tracer walks its bounding-volume hierarchy
  nearest box first and stops at boxes beyond the nearest hit found so far,
  with a rounding margin of 10⁻⁹. Two exactly tied hits (cemented twins) go to
  the lower object id, so the order the walk visits objects in cannot move a
  ray. A test runs every gallery design both ways, against a brute-force
  search that asks every object, and requires the same bits.
- **Medium lookup.** Both tracers skip a body whose bounding box does not
  contain the point before asking the body itself.
- **Per-body preparation.** The f64 tracer computes each revolved profile's
  tolerances once per trace instead of once per intersection.

Measured with `bench_gallery` on one thread, the f64 reference traces 3.5 to
4.5 times faster than before these changes. In the ghost-lit f32 preview the
event search's box test made it 1.6 to 2.8 times faster, and the medium
lookup's another 1.6 to 3.5 times. Both tracers give the fingerprints they
gave before, bit for bit.

## Measured

The reference with ghosts on, on a 24-thread desktop: one thread native (as
the app traced it before the pool, where the heaviest never finished in the
browser), and eight workers in headless Chromium, from
`web/tests/reference-bench.spec.ts`. Timings move by about 15% with the
machine's state; both columns come from one session.

| Design, ghosts on | Segments kept | One thread | Eight workers |
|---|---:|---:|---:|
| Cooke triplet | 122k | 0.3 s | 0.08 s |
| Double Gauss | 543k | 2.3 s | 0.49 s |
| Objective 60x | 2.0M | 9.5 s | 1.8 s |
| UV projection lens | 5.5M | 28.5 s | 4.9 s |
| US objective (US 8,879,901) | 9.8M | 84 s | 15.1 s |

In none of these cases does the page's thread run a task over 50 ms while the
reference is traced. Switching ghosts on for the US objective used to freeze
the page (a 3.4 s CPU frame, then a stream of 50 to 160 ms frames while the
picture was drawn, and in the 3D view no rays at all). Measured with the
Chrome profiler on both GPUs and without one, it now runs no task over 50 ms
in the 2D view and at most two of 50 to 57 ms in the 3D view, around the
moment the stand-in lands. Dragging a lens of the heaviest designs runs no
task over 70 ms, and the rays update four to twenty-six times in a two-second
drag.

## Without a dedicated GPU

Measured on the Radeon iGPU of a Ryzen 7000 CPU (RDNA 2, two compute units:
Chrome pointed at it with `--use-adapter-luid`), and with no GPU at all:

- **The 2D view** needs no GPU. It rasterizes on the CPU and behaves the same
  everywhere.
- **The preview** of a ghost-lit lens goes to the workers on an iGPU, which
  is slower than the CPU for these trees (table above). Without WebGPU it
  goes to the workers or the CPU mirror by cost, as on any machine.
- **The 3D view** runs on the iGPU through WebGPU, and through three.js's
  WebGL 2 fallback where there is no WebGPU. On the iGPU the heaviest design
  with ghosts runs without long tasks. Where the browser has no GPU
  acceleration at all and rasterizes WebGL in software, the 3D view of a
  picture of millions of segments takes seconds per frame; the 2D view is
  the one to use there.
- **Memory** is the limit that matters on a laptop, since an iGPU draws its
  buffers from system RAM. For the US objective with ghosts on, the browser
  holds about 0.8 GB with ghosts off, 1.6 GB in the 2D view with ghosts on,
  and 2.4 GB in the 3D view, of which about 1.1 GB are the GPU process's
  buffers. The US objective and the 120° projection lens are the two
  heaviest gallery designs; most others need far less. The worker count
  follows `navigator.deviceMemory`, so a 4 GB machine runs three workers.

To measure:

```text
cargo build --release -p oc-gpu --example bench_gallery --features gpu3
RAYON_NUM_THREADS=1 target/release/examples/bench_gallery web/public/gallery [design...]
# JOB=<seconds>  the app's ghost reference, stepped as a worker steps it
# F32HASH=1      the f32 preview's fingerprint and time
# HASH=1         the exact trace's fingerprint and time
cd web && OC_BENCH=1 OC_BENCH_CHANNEL=chromium npx playwright test --project=bench tests/reference-bench.spec.ts
```

## Coating grids in the preview

A coated face is baked onto a grid when the scene compiles (CF-07): 128 angle
rows by the scene's own wavelengths, per side when the stack absorbs. At 64
wavelengths a grid is 64 kB, the spec's budget per face per side, and faces
that share a response share a grid. Baking protected silver from both sides at
64 wavelengths takes 5.0 ms in release on this machine (budget 20 ms):

```text
cargo test --release -p oc-gpu --features gpu3 --test coating_preview3 bake_cost -- --ignored --nocapture
```

The preview recompiles on every frame of a drag, so this is a per-frame cost
for a coated face that is being moved; a scene's coatings that are not edited
cost it once per compile.

## Wave optics (`oc-wave`)

The scalar wave kernel ([ADR 0003](architecture/adr/0003-scalar-wave-optics-engine.md),
[wave-optics.md](wave-optics.md)) runs in the app in a worker of its own (W5),
off the page thread. Its native timings, single thread, release, AMD Ryzen 9
9900X, measured 2026-10-19:

```text
cargo run -p oc-wave --release --example bench_wave
```

| n² | 2D FFT | bridging `Cx` to rustfft's type | share |
|---|---|---|---|
| 128² | 0.07 ms | 0.01 ms | 7.0% |
| 256² | 0.61 ms | 0.03 ms | 4.7% |
| 512² | 2.49 ms | 0.13 ms | 5.1% |
| 1024² | 11.8 ms | 0.82 ms | 6.9% |
| 2048² | 53.8 ms | 6.0 ms | 11.2% |

| Field (padded 2×) | One band-limited angular-spectrum step |
|---|---|
| 128² (256²) | 2.9 ms |
| 256² (512²) | 11.7 ms |
| 512² (1024²) | 52 ms |
| 1024² (2048²) | 209 ms |

ADR 0003 set a 15% ceiling on the bridge before the internals would switch
type; it is under that everywhere. The plan's initial gate for a settled
1024² step natively is 200 ms. At 209 ms it is just over: the cost is the
per-sample transfer function (a square root and a sine-cosine pair over 4.2 M
samples) as much as the transforms. The gate is recorded here rather than
moved.

In the browser (Playwright 1.61's bundled Chromium, same machine,
2026-10-19; `tests/wave-bench.spec.ts`, opt-in): the whole wave path of the
default scene with its singlet taken out, one band-limited angular-spectrum
step over 118 mm and the detector readout, timed in the worker from the
scene's load to the result:

| Field | Wave path in the worker | Longest page task meanwhile |
|---|---|---|
| 128² | 10 ms (6 ms warm) | 0 ms |
| 256² | 22 ms | 0 ms |
| 512² | 87 ms | 0 ms |
| 1024² | 298 ms | 0 ms |

The page thread ran no task over 50 ms during any of them: the work is the
worker's. One earlier run's first measurement showed a single 1.7 s page task
that did not recur on a repeat; it is recorded here, not explained.

The wave tests transform million-sample fields, so `rustfft` and `oc-wave` get
an optimized dev profile in the workspace `Cargo.toml`. Without it the analytic
suite takes 37 s in a debug build; with it, under 3 s.

## Diffraction orders in the ray tracer

A DMD or SLM in the orders ray model (W7) computes its unit cell's order
table at every hit: for a 7.56 µm pitch at 550 nm about 750 propagating
orders, each a product of sincs, a few tens of microseconds. Each hit then
spawns up to `max_orders` children, so a trace's ray count grows by that
factor at every modulator a ray meets; the editor's setting is 16 above
0.1%. The TI test (64 rays, every order above 1e-9 followed) runs in 10 ms.

## What each fidelity switch costs

Relative trace time of each switch value against the value named, about 200000 launch rays per scene, best of 3, ghosts off, deposits recorded on every body. Measured on one Windows desktop (release build, all cores), regenerated with
`cargo run --release -p oc-trace --features trace3 --example bench_throughput3 -- --fidelity-table --write-costs`,
which also writes `web/public/fidelity-costs.json`, the numbers the fidelity panel shows. The ceiling is the coating-fidelity spec's §11 budget; every value is inside it. Values below 1 are timing noise of a switch that costs nothing on that scene: the singlet has no coating for `PerLayer` to split, and no scene declares a temperature.

| Switch value | against | heliostat-light-pipe | two-dichroics | singlet | worst | ceiling |
|---|---|---|---|---|---|---|
| `coating_model: Nominal` | Fast | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `coating_model: Unpolarized` | Nominal | 1.047 | 1.027 | 1.002 | 1.047 |  |
| `coating_model: Polarized` | Unpolarized | 1.000 | 1.004 | 1.026 | 1.026 | 1.5x |
| `film_materials: Nominal` | Fast | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `film_materials: Dispersive` | Nominal | 1.010 | 0.989 | 1.006 | 1.010 | 1.15x |
| `layer_absorption: Off` | Fast | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `layer_absorption: PerFace` | Off | 1.021 | 1.067 | 1.095 | 1.095 |  |
| `layer_absorption: PerLayer` | PerFace | 1.053 | 1.275 | 0.950 | 1.275 | 1.3x |
| `plate_model: PhaseFree` | Fast | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `plate_model: Incoherent` | PhaseFree | 1.027 | 0.998 | 0.994 | 1.027 | 1.05x |
| `temperature: Off` | Fast | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `temperature: Declared` | Off | 0.991 | 0.986 | 0.986 | 0.991 | 1.05x |
| `numerics: Permissive` | Fast | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `numerics: Strict` | Permissive | 1.012 | 0.989 | 0.972 | 1.012 | 1.02x |

## Still open

- Software-rasterized WebGL (no GPU acceleration) cannot draw a ghost-lit
  twenty-element lens in 3D at an interactive rate.
- The heaviest ghost pictures need gigabytes. A picture budget that follows
  the screen rather than the flux (what a pixel can show) would cut that, and
  would change what is drawn, so it is a decision rather than a fix.
- Pairwise ghost enumeration, the structural fix in
  [the ghost-tracing note](ghost-tracing-perf-brainstorm.md), would change
  what is traced and is not part of this work.
