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
- `sampling`: rays, passes, rays per pass, and how the noise was estimated.

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
