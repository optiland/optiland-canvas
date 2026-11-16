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
the scene does not have, a pass count that does not divide the kernel's 64
blocks.

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
