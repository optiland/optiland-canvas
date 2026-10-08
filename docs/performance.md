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

## Still open

- Software-rasterized WebGL (no GPU acceleration) cannot draw a ghost-lit
  twenty-element lens in 3D at an interactive rate.
- The heaviest ghost pictures need gigabytes. A picture budget that follows
  the screen rather than the flux (what a pixel can show) would cut that, and
  would change what is drawn, so it is a decision rather than a fix.
- Pairwise ghost enumeration, the structural fix in
  [the ghost-tracing note](ghost-tracing-perf-brainstorm.md), would change
  what is traced and is not part of this work.
