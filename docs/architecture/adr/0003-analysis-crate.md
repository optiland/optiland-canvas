# ADR 0003: Analyses live in `oc-analysis`, on top of the one kernel

**Status:** Accepted, 2026-11-14
**Baseline:** `main` at `5bef5af`
**Amends:** the README's "Out of scope" line, and ADR 0001's point 3
("Optiland may provide … analyses, reports, optimization"), for the analyses
named below

## Context

The reference tracer already sums more than any panel shows: per detector and
wavelength it keeps the flux, the centroid, the spot size per axis and the
plane of best focus, and its ledger knows where every watt went. A designer's
questions are one step further on. How much of each channel reaches its
camera, and how much of the others leaks in? How even is the light across a
field? Where did the rest of it go? What are the focal length and the
magnification of a folded system? How thick is a light sheet? Where must the
stage go for focus?

Each of those is a figure computed from traces of the scene, and each needs
answering in the browser, where the scene is, without a second engine. Three
things make that a decision rather than a feature:

1. **The written scope excludes it.** The README sends optimization (and
   diffraction) to Optiland, and ADR 0001 lets Optiland provide analyses
   around the kernel. A browser application that wants these figures from the
   kernel it already ships would otherwise compute them in TypeScript, or
   route the scene through Python: a second source of numbers either way.
2. **The kernel must stay one thing.** `oc-trace` is the transport engine
   every number comes from, cross-validated against Optiland and against the
   f32 preview. An analysis that reached into transport, a "fast path for
   channels" say, would be a second engine in the same crate.
3. **Numbers from rays are noisy, and today none says how noisy.** A
   detector's flux from a hundred thousand rays is an estimate. A panel that
   quotes it to six figures and a crosstalk of 0.1 % that is really 0.1 ± 0.3 %
   are wrong in the way that matters, and nothing in a trace result says so.

## Decision

**Analyses are a new crate, `crates/oc-analysis`, that drives `oc-trace`
through its public API.**

1. **In scope** for `oc-analysis`:
   - **ray statistics**: what each detector received per source, channel and
     wavelength, crosstalk and leak between channels, evenness over a region,
     and the light budget (detected, absorbed, blocked, escaped, per part and
     along a path), from the existing sums and ledgers;
   - **first-order figures from rays**: focal lengths, magnification, stop and
     numerical aperture, from ray-transfer matrices of rays close to the axial
     ray, through folds;
   - **Gaussian beam size** along a path in both axes, propagated through
     those matrices;
   - **stepped optimisation**: one or a few scene values moved until a figure
     is zero or least (focus the stage), each step a trace like any other.
2. **Out of scope**, still: wave optics (wavefront error, PSF, MTF,
   interference, diffraction beyond what a grating's orders already do), which
   needs its own ADR; tolerancing; and any change to transport. Birefringence
   and thermal effects stay where the README puts them.
3. **The kernel stays the one transport engine.** `oc-analysis` chooses which
   rays to launch and reads the sums a trace keeps (`TraceBlocks3`,
   `TracePart3`, `TraceMerge3`, `launch_from_source_pass`, `trace3`). It never
   changes how a ray travels. When an analysis needs a sum the kernel does not
   keep, the sum is added to `oc-trace` as a kernel change of its own, tested
   there, and the analysis reads it.
4. **One job interface for every analysis.** A request names its `kind`, how
   many launch `rays` and independent `passes`, and per-kind `params`. A job
   is stepped a batch of launch rays at a time, so a browser worker returns to
   its event loop between batches; it is dropped when the scene, its profile
   or its ghost setting changes, as the reference trace is
   (`referenceBegin` / `referenceStep`); and stepping it in any batches gives
   the same result, to the bit. The wasm handle exposes it as `analysisBegin`,
   `analysisStep`, `analysisProgressJSON`, `analysisResultJSON` and
   `analysisCancel` (`crates/oc-wasm/src/analysis_ui.rs`).
5. **Every result says what it rests on**, in one envelope: the scene
   revision it was traced at, the fidelity stamp, the tracer's approximations
   and rejections from the capability report, and the rays and passes behind
   it. **Every statistical number carries a standard error**, from the spread
   between independent passes; a number with no noise says so with a zero,
   and one that cannot say has `null`, never a bare value.

## How the first analysis meets it

`detectorStats` is the first kind, the smallest that exercises every part of
the interface: per detector its hits, flux and signal, and per wavelength its
hits, flux, centroid and RMS spot size along each axis.

- **Passes.** The job traces `passes` passes of the scene's rays
  (`TraceBlocks3::with_passes`): each pass the sources' point sets shifted
  afresh, so the passes are independent estimates of the same numbers. The
  kernel cuts the trace into 64 blocks; a job asks for a number of passes that
  divides them, so every pass is whole blocks.
- **Pooled values** come from the kernel's merge of every block, so they are
  exactly the reference readout of the same rays. **Standard errors** come
  from the per-pass readings, read off each block before it is merged (the
  accessors `TracePart3::detectors` and `TracePart3::ledger`, the one addition
  to `oc-trace` this ADR makes: they read, they do not trace). The standard
  error is the batch-means one, `s/√m` over the `m` passes that saw anything;
  for a ratio such as a centroid it is the first-order error.
- **Tests** (`crates/oc-analysis/tests/job.rs`,
  `crates/oc-wasm/src/analysis_ui.rs`): stepping one ray at a time, in odd
  batches and all at once give the same JSON; pooled values equal the kernel's
  readout to the bit; a beam of known spot reads its RMS within a few standard
  errors, and a beam that lands whole states a zero error on its flux; over
  forty independent runs the actual error over the stated one has a mean
  square near one; cancelling, or changing the scene or a trace setting,
  drops the job and its result.

## Consequences

- The application holds adapters only: it sends a request, steps the job,
  and shows the numbers that come back with their errors. No figure is
  computed in TypeScript, and the service is not in the loop.
- A new analysis is a new `kind`: its params, its per-pass readings, and its
  result body, behind the same five calls and the same envelope.
- Pass 0 is the scene's unshifted point set and passes from 1 on are random
  shifts of it. For a grid or a Sobol' source, pass 0 is one fixed draw among
  the others; with sixteen passes its weight in the mean is a sixteenth, and
  the error it adds is of the size the standard error already states.
- A pass holds at most 524 288 launch rays, as a reference pass does; the
  rays a job traces are therefore capped at that times its passes, and the
  result states the count it traced.
- The figures this ADR admits that are not statistics (first-order, beam
  size, optimisation) still come from traces, and still carry the envelope.
  How each states its accuracy is decided when it is built.

## Alternatives considered

**Compute the figures in the application.** Rejected. It is a second source
of numbers, in a language the kernel's tests do not reach, and it has nothing
to compute standard errors from: the per-pass sums are inside the trace.

**Put the analyses in `oc-trace`.** Rejected. The kernel crate would grow a
second purpose, and every analysis change would be a kernel change to review
as one. Keeping them apart keeps "transport did not change" a fact a diff
shows.

**Ask Optiland.** Rejected for these figures. Optiland's analyses are
sequential: they answer for the path they are given, not for the non-sequential scene the
browser traces, and it is not in the browser. It stays the oracle for
straight systems in tests.

**Standard errors from per-hit records.** Rejected. Records are capped (8192
per detector), so they cover a fraction of a large trace, and a hit is not an
independent sample: the hits of one launch ray are correlated through its
branches. Passes are independent by construction.
