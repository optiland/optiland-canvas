# Analyses

An analysis is a question the tracer answers from many rays: what each
detector received, with how much noise; later, which channel a source's light
reached, how even a field is, where the light went. The answers come from the
crate `oc-analysis`, on top of the one transport engine `oc-trace`; nothing
in it traces a ray any other way. What belongs there, and what does not, is
[ADR 0003](architecture/adr/0003-analysis-crate.md).

Every analysis runs as one job, stepped a batch of launch rays at a time, so
a page can show progress and stop it. However it is stepped, the result is
the same to the bit.

## Running a job

```text
analysisBegin(requestJSON)  -> launch rays in all
analysisStep(rays)          -> fraction traced; 1 once the result is ready
analysisProgressJSON()      -> where the job is
analysisResultJSON()        -> the result, or null
analysisCancel()            -> drop the job
```

A job belongs to the scene it was begun on: its revision, its fidelity
profile and the ghost setting. The first call after any of those changes
drops the job, and `analysisProgressJSON` says why (`sceneChanged`, or
`cancelled` after `analysisCancel`). A dropped job's result is never served.
This is the reference trace's rule (`referenceBegin` / `referenceStep`).

`analysisBegin` throws, with the reason, for a request this build cannot run
or this scene cannot answer: an unknown kind, an unknown field, a detector
or a source the scene does not have, a pass count that does not divide the
kernel's 64 blocks.

## The request

```json
{"kind": "detectorStats", "rays": 20000, "passes": 16, "params": {"detectors": [6]}}
```

| Field | Meaning |
|---|---|
| `kind` | Which analysis. |
| `rays` | Launch rays in all (optional; the scene's own count if absent). At most 524,288 per pass. |
| `passes` | Independent passes the rays are split into, 2 to 64 and dividing 64 (optional; 16). The noise comes from how the passes differ. |
| `params` | What that kind needs. |

## The result

Every result carries, beside its own numbers:

- `revision` and `fidelity`: the scene and the profile it was traced with,
  and the profile's approximations;
- `approximations` and `rejected`: what the capability report says this
  scene's features were traced with, or could not be;
- `sampling`: the rays and passes traced, rays per pass, and how the noise
  was estimated. A job that stops on a noise target states where it stopped.

A number with noise is `{"value": ..., "stderr": ...}`. `stderr` is the
standard error from the passes (batch means, `s / √m`): 0 for a number with
no noise, `null` when fewer than two passes saw anything. Hit counts carry
none.

Each pass shifts the sources' sample points afresh, so the passes are
independent estimates; pass 0 is the unshifted set. The pooled values are
the kernel's own merge, so they equal the reference readout to the bit.

## detectorStats

What each detector received, per wavelength line. `params.detectors` lists
the detector ids (all detectors if absent).

Per detector: `hits`, `flux` and `signal`. Per line: `lambdaUm`,
`meanLambdaUm`, `hits`, `flux`, `centroidUv` and `rmsUv`, each axis with its
standard error. `u` and `v` are the detector's local X and Y, as in the
[detector readout](detectors.md).

## What is checked

- Stepping one ray, 37, 1000 or all at once gives the same JSON.
- The pooled flux, hits and spot sizes equal the reference trace's.
- A square beam of half-width 5 mm reads an rms of 5/√3 mm and a centroid
  of 0, each within 5 standard errors; a beam that lands whole reads flux 1
  with no noise.
- Over 40 seeds the standard error is calibrated: the mean of
  (error / stderr)² stays near the t-distribution's.
- Cancelling, a scene edit and a ghost toggle drop the job; a bad request
  throws and keeps nothing.

## channels

What each channel passes, and how much of the other sources' light leaks
into it. For a scene of several sources, such as the DAPI, EGFP and mCherry
lines of a fluorescence microscope, and a detector per channel.

```json
{"kind": "channels", "rays": 60000, "passes": 16, "params": {"channels": [
  {"name": "EGFP", "detector": 6, "sources": [2], "bandUm": [0.5, 0.55]}]}}
```

A channel has:

| Field | Meaning |
|---|---|
| `name` | Its name, unique in the request (optional; the detector's label, or `detector <id>`). |
| `detector` | The detector it reads, by id. |
| `sources` | The sources it is meant to see, by id (optional; every source). |
| `bandUm` | `[low, high]` in µm: only light detected in this band counts (optional; every wavelength). |

With no `params`, there is one channel per detector, seeing every source,
with no band.

The job traces each source alone, one after the other, so the light on a
detector is known by the source it came from. Each trace is the scene with
that one source, under the whole scene's settings. `rays` is split across
the sources as their own counts are: every source is sampled by the same
factor, so a source that emits four times the rays of another still does.
Each source traces the same `passes`.

A band is read off the detector's wavelength bins (5 nm wide below 1 µm). A
bin counts when its flux-weighted mean wavelength is in the band, ends
included, so a line counts exactly by its own wavelength, and a continuous
spectrum is cut to the nearest bin. Light past a detector's 256th bin is in
no band.

The result has:

- `sources`: per source its id, the launch `rays` it traced and the flux it
  `launched`.
- `channels`: per channel its `name`, `detector`, `label`, own `sources` and
  `bandUm`, and
  - `signal`: the flux from its own sources, in band;
  - `leak`: the flux from every other source, in band;
  - `ratio`: leak over signal, `null` when there is no signal. Its standard
    error is the first-order one.
- `crosstalk`: the table of `sources` by `channels`. `flux[s][c]` is the flux
  source `s` puts on channel `c`'s detector, in its band.

Every flux is the incident flux (W), not the detector's response-weighted
signal. Every number is `{"value", "stderr"}`. The sources are traced
independently, so a pass's signal is the sum of its sources' readings in
that pass, and the spread of those sums is the signal's error.

```json
"result": {
  "sources": [{"source": 1, "rays": 1024, "launched": 1.0},
              {"source": 2, "rays": 1024, "launched": 1.0}],
  "channels": [{"name": "red", "detector": 7, "label": "", "sources": [2],
                "bandUm": null,
                "signal": {"value": 0.5, "stderr": 0.0},
                "leak":   {"value": 0.5, "stderr": 0.0},
                "ratio":  {"value": 1.0, "stderr": 0.0}}],
  "crosstalk": {"sources": [1, 2], "channels": ["red"],
                "flux": [[{"value": 0.5, "stderr": 0.0}],
                         [{"value": 0.5, "stderr": 0.0}]]}
}
```

`analysisBegin` also throws for a channel with no sources, a source listed
twice, a band that is not `0 < low < high`, two channels of one name, or an
empty channel list.

### What is checked

- Stepping one ray, 37, 1000 or all at once gives the same JSON.
- Each table entry equals the reference trace of that source alone, to the
  bit; a signal of every source is the column's sum, to the bit.
- Two beams side by side, each onto its own detector, read a signal of 1 and
  a leak of 0, both with no noise.
- A blue and a red beam split 50/50: behind an ideal short-pass the blue
  channel reads no leak; behind a long-pass that passes both, the red
  channel leaks the blue half, a ratio of 1. A band on the red line takes
  that leak out again.
- Two beams onto a detector smaller than both: over 40 seeds the standard
  errors of the signal, the leak and the ratio are calibrated, as for
  `detectorStats`.
- `rays` splits across sources as their own counts do.
- Each bad request throws and keeps nothing.

## uniformity

How even the light is over a region of one detector, read on a grid of
cells, with a noise target to trace down to.

```json
{"kind": "uniformity", "rays": 2000000, "passes": 64, "params": {
  "detector": 6, "cells": [20, 20], "regionMm": [[-5, 5], [-5, 5]],
  "target": {"metric": "cv", "stderr": 0.005}}}
```

| Field | Meaning |
|---|---|
| `detector` | The detector read, by id. |
| `cells` | `[across u, across v]`, at most 1024 along an axis and 65,536 in all. |
| `regionMm` | `[[u low, u high], [v low, v high]]` in the detector's local mm (optional; the face's whole grid). |
| `target` | `{"metric": "cv" or "minMax", "stderr": ...}`: stop once that metric's standard error is this small (optional; trace every ray). |

`passes` must be at least 4. With a target, `rays` is the ray limit; ask for
64 passes, so the job has checks to stop at.

**Cells are the kernel's pixels.** The job traces the scene with the
detector's pixel grid set so each cell is a whole number of pixels: the
cells themselves for the whole face, and for a region a finer grid, up to
1024 pixels along an axis, whose edges land on the region's to within a
thousandth of a cell, or as near as 1024 pixels allow. `regionMm` in the
result is the region read; `regionAskedMm` the one asked for. A pixel grid
bins hits and moves no ray. A cell's irradiance is its flux over its area on
the face (flux per mm², relative unless the source is calibrated). A cell
off a round face is `null` and counts in no metric.

**Noise in, noise out.** Noise makes cells look less even than the light
is: their spread is wider, and the least and most of them lie further out.
So the passes are split into the even and the odd ones, whose noise is
independent:

- `cv`, the coefficient of variation across cells: its variance is the mean
  product of a cell's deviation in one half with its deviation in the other,
  over the product of the halves' means. Noise does not bias it. The CV is
  the root, 0 if noise leaves the variance at or below 0; its `stderr` is
  the distance up to the root of the variance plus one standard error, so an
  even field reads 0 with the noise it could hide.
- `minMax`: the least and most lit cells picked in one half and their ratio
  read in the other, both ways round, the mean of the two. Its error is
  conditional on the cells picked. When noise swamps the contrast the cells
  picked are not the true extremes, and it reads nearer 1 than the light.

The other numbers are the pooled ones from every pass traced, with
batch-means errors:

- `mean`: the mean irradiance over the cells;
- `falloff.edge` and `falloff.corner`: the mean irradiance of the border
  cells and of the four corners over that of the centre cells (the middle
  one, or the two or four of an even count). `null` under 3 × 3 cells.

**A cell with few hits says so.** The kernel counts hits per detector, not
per cell. `irradiance` holds, row-major with `u` fastest, each cell's
`value`, `stderr` and `litPasses`, the passes that put any light in it. A
cell lit in some passes and not all has too few hits per pass for its error
to be trusted; `cellsSparse` counts those, `cellsDark` the cells no pass
lit, and `medianCellRelStderr` is the typical cell's relative error.

**Stopping on the target.** A job with a target checks it after 8, 16 and
32 passes (powers of two, short of the last): if the metric's standard error
is at or below the target, the job stops there and its result is that of
the passes traced. A check reads whole passes only, so a job stops at the
same pass however it is stepped, and gives the same JSON. Few checks keep it
from stopping on a lucky low reading. `analysisStep` returns 1 when it
stops; `sampling` states the rays and passes traced.

A beam 24 mm square over a 20 mm square face, `rays` 256,000 in 64
passes, a target of 0.05 on the CV: met at the first check.

```json
"sampling": {"rays": 32000, "passes": 8, "raysPerPass": 4000, "stderr": "batchMeans"},
"result": {
  "detector": 1, "label": "", "hits": 22265,
  "cells": [8, 8], "regionMm": [[-10.0, 10.0], [-10.0, 10.0]],
  "regionAskedMm": null, "cellSizeMm": [2.5, 2.5],
  "irradiance": {"value": [0.001775, 0.00171, 0.00174, ...],
                 "stderr": [0.0000501, 0.0000344, 0.0000414, ...],
                 "litPasses": [8, 8, 8, ...]},
  "mean":   {"value": 0.0017395, "stderr": 0.0000035},
  "minMax": {"value": 0.9972, "stderr": 0.0374},
  "cv":     {"value": 0.0054, "stderr": 0.0118},
  "falloff": {"edge":   {"value": 1.0232, "stderr": 0.0145},
              "corner": {"value": 1.0448, "stderr": 0.0225}},
  "minCell": [2, 7], "maxCell": [7, 7],
  "cellsOnFace": 64, "cellsDark": 0, "cellsSparse": 0,
  "medianCellRelStderr": 0.0189,
  "stop": {"metric": "cv", "target": 0.05, "stderr": 0.0118,
           "reached": true, "passes": 8, "rays": 32000, "rayLimit": 256000}
}
```

`stop` is `null` without a target; `reached` is false when the ray limit
came first, and `stderr` is then the noise at the limit.

`analysisBegin` also throws for fewer than 4 passes, a cell count of 0 or
past the limits, a region off the face's grid or not `low < high`, a region
with no cell on the face, a target on another metric, or a target that is
not above 0.

### What is checked

- Stepping one ray, 7, 37, 1000 or all at once gives the same JSON, with a
  target or without, stopped early or at the limit.
- Over the whole face, each cell's irradiance is the kernel's pixel flux
  over its area, to the bit.
- A beam wider than the face reads a CV of 0 and a min/max of 1, and edge
  and corner falloffs of 1, within 3 standard errors; four times the rays
  bring the errors down by half or more (the sampling is stratified).
- A beam over half the region reads a CV of 1, a min/max of 0 with no
  noise, and half the beam's irradiance.
- A beam edge through a cell's middle: over 30 seeds the CV's standard
  error is calibrated, as for `detectorStats`.
- A loose target stops at 8 of 64 passes with `reached: true`; an
  impossible one runs every ray with `reached: false`.
- A tile of the face is read on a grid snapped to its edges.
- A sparse field counts the cells some pass missed.
- Each bad request throws and keeps nothing.

## budget

Where the light went: every launched watt by the sink it ended in, each
detector's catch, and the parts that lost it.

```json
{"kind": "budget", "rays": 16384, "passes": 8, "params": {"bySource": false}}
```

| Field | Meaning |
|---|---|
| `bySource` | Trace each source alone, and give each its own budget beside the scene's (optional; false). Costs no extra rays: `rays` is split across the sources as for `channels`. |

The numbers are the kernel's ledgers, read as they are. The job turns on the
object ledger, which books each removal against the part where it happened;
it changes no other number, so the totals equal the reference trace's
ledger to the bit.

The result has:

- `launched`: the flux launched (W).
- `totals`: the seven sinks, each `{"value", "stderr"}`. They do not
  overlap, and together they are everything that left the light:

| Sink | What ended there |
|---|---|
| `detected` | Light a stopping detector caught. A probe detector lets light on and is no sink. |
| `absorbedSurface` | Absorbed at a face: a filter, a coating, a black face, a beam splitter's loss. |
| `absorbedVolume` | Absorbed in a body's bulk, or in an absorbing medium. |
| `blocked` | Stopped by an aperture's mask. |
| `escaped` | Left the scene bounds. |
| `truncated` | Culled by a limit: depth, ghost depth, branch count or the flux threshold. `branches` says which by count. |
| `numerical` | Lost to a numerical failure. |

- `shares`: each sink over `launched`.
- `closure`: `accounted`, the sum of the sinks; `residual`, launched less
  that; `relative`, its size over launched; `parts` and `partsResidual`, the
  same for the parts' removals. The kernel books every removal once, so both
  close to rounding (under 1e-12 of launched in the tests).
- `detectors`: per detector its id, `label`, `terminates`, `hits`, the
  incident `flux` with its error, and its `share`. A probe's flux is light
  that passed through.
- `parts`: per part `object` (its id), `kind` (`surface`, `body`,
  `aperture`, `detector`, or `source` for the launch), `label`, and what it
  did to the light: `hits` (a count), `arrived` (the flux arriving at it),
  `lost` (absorbed or blocked there) and its `lostShare`, and the light it
  removed every way: `absorbedSurface`, `absorbedVolume`, `blocked`,
  `detected`, `escaped`, `truncated`, `numerical`. Each flux is
  `{"value", "stderr"}`; `hits` and `lostShare` are plain numbers. Sorted
  by `lost`'s value, largest first.
- `branches`: the branches that ended in each way, as counts.
- `sources`, with `bySource` only: per source its `source` id and `rays`,
  and the same budget for that source alone.

**Who a loss is booked to.** Absorption at a face is booked to that face,
a block to its stop, a catch to its detector. Light that leaves the scene,
is culled, or is absorbed in a bulk on the way is booked to the part it last
left, so the bulk of a lens is the lens's, and a ghost that leaves the scene
is the face that reflected it. Light that meets nothing is booked to the
launch, the `source` row: its source's id when it was traced alone or is the
only one, else `null`.

**A body's faces.** A body's row adds its faces up, and `faces` gives each
by its `patch`. `arrived` and `hits` for a body are summed over its faces,
so light through a lens arrives twice: once at each face.

**Errors.** The totals, the detectors and the parts, faces included, carry
batch-means errors. A part's error comes from each block's object ledger,
added up pass by pass the same way the totals come from each block's
ledger; a pass that never met the part reads 0 for it. The values are the
pooled ones from the merged trace, the same to the bit as without the
errors.

**Along a path.** `arrived` is the flux that reached each part, from any
direction and by any route, ghosts included. The kernel does not book the
order the light met the parts in, so the budget gives no sequence. The
event log has it, ray by ray, but a job does not keep it.

A beam through an absorbing lens, a stop of 2 mm radius, a filter that
transmits 30% and onto a detector, ghosts on:

```json
"result": {
  "launched": 1.0,
  "totals": {
    "detected":        {"value": 0.22789,  "stderr": 0.00019},
    "absorbedSurface": {"value": 0.53174,  "stderr": 0.00044},
    "absorbedVolume":  {"value": 0.07452,  "stderr": 0.0000063},
    "blocked":         {"value": 0.09015,  "stderr": 0.00064},
    "escaped":         {"value": 0.07570,  "stderr": 0.00000082},
    "truncated":       {"value": 1.32e-7,  "stderr": 2.5e-10},
    "numerical":       {"value": 0.0,      "stderr": 0.0}},
  "shares": {"detected": 0.22789, "absorbedSurface": 0.53174, ...},
  "closure": {"accounted": 0.9999999999999997, "residual": 3.3e-16,
              "relative": 3.3e-16, "parts": 0.9999999999999997,
              "partsResidual": 3.3e-16},
  "detectors": [{"detector": 8, "label": "", "terminates": true, "hits": 16535,
                 "flux": {"value": 0.22789, "stderr": 0.00019}, "share": 0.22789}],
  "parts": [
    {"object": 7, "kind": "surface", "label": "", "hits": 17161,
     "arrived":         {"value": 0.75962, "stderr": 0.00063},
     "lost":            {"value": 0.53174, "stderr": 0.00044},
     "lostShare": 0.53174,
     "absorbedSurface": {"value": 0.53174, "stderr": 0.00044},
     "absorbedVolume":  {"value": 0.0,     "stderr": 0.0},
     "blocked":         {"value": 0.0,     "stderr": 0.0},
     "detected":        {"value": 0.0,     "stderr": 0.0},
     "escaped":         {"value": 0.0,     "stderr": 0.0},
     "truncated":       {"value": 2.3e-8,  "stderr": 3.3e-10},
     "numerical":       {"value": 0.0,     "stderr": 0.0}},
    {"object": 6, "kind": "aperture", "lost": {"value": 0.09015, "stderr": 0.00064}, ...},
    {"object": 5, "kind": "body", "arrived": {"value": 1.92219, "stderr": 0.0000066},
     "lost": {"value": 0.07452, "stderr": 0.0000063}, ...,
     "faces": [{"patch": 1, "arrived": {"value": 1.03478, "stderr": 0.00000041}, ...},
               {"patch": 3, "arrived": {"value": 0.88741, "stderr": 0.0000062}, ...}]},
    {"object": 8, "kind": "detector", "detected": {"value": 0.22789, "stderr": 0.00019}, ...}],
  "branches": {"detector": 16535, "blocked": 24496, "escaped": 56647,
               "fluxThreshold": 17010, "maxDepth": 0, ...}
}
```

`analysisBegin` throws for an unknown field in `params` or a `bySource`
that is not true or false, besides what it throws for every kind.

### What is checked

- Stepping one ray, 37, 1000 or all at once gives the same JSON, whole and
  by source.
- With ghosts on and a depth limit of 3, so every sink but `numerical` is in
  play, the sinks and the parts close to 1e-12 of launched, the stopping
  detectors add up to `detected`, and each total equals the reference
  trace's ledger to the bit. Each part's values, and the lens's faces',
  equal the reference trace's object ledger to the bit, and the lens's and
  the stop's losses have a non-zero error.
- An ideal filter that transmits 30% reads 0.7 absorbed at it and 0.3
  detected, with no noise, in the totals and part by part; it heads the
  parts.
- A square beam of half-width 4 mm through a stop of radius 2 mm reads
  `blocked` 1 − π/16 within 5 standard errors, all of it at the stop,
  whose error is the total's.
- A beam that misses everything reads `escaped` 1, booked to the launch.
- By source, each source closes on its own, and the scene's budget is
  theirs added up.
- Each bad request throws and keeps nothing.
