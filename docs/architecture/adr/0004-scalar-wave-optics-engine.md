# ADR 0004: Scalar wave optics lives beside the ray tracer, in its own crate

**Status:** Accepted, 2026-10-19
**Baseline:** `main` at `b2ea503`

## Context

The kernel is a non-sequential geometric ray tracer. A ray is a packet of
flux, never a field amplitude (`docs/physics/conventions.md`), and
interference and diffraction were declared out of scope (README;
`canonical-3d-kernel.md` lists "coherent wave propagation, diffraction PSF/MTF"
under *Not in this epic*). That boundary is what blocks two devices people
put on an optical table every day:

- the **digital micromirror device** (DMD), an array of mirrors a few
  microns across, which acts as a 2D blazed grating, so where its light goes
  depends on the wavelength (Popoff et al., *J. Phys. Photonics* 8, 023002
  (2026), doi:10.1088/2515-7647/ae60d5, first published as 6, 043001 (2024)
  and since retracted for its blazed-number equation; Texas Instruments
  application note DLPA083B);
- the **phase-only LCoS spatial light modulator** (SLM), a pixelated phase
  screen whose purpose is diffraction (Zhang, You & Chu, *Light Sci. Appl.* 3,
  e213 (2014), doi:10.1038/lsa.2014.94).

Neither can be modelled honestly without a field. The question this ADR settles
is where the field lives, and what it may touch.

## Decision

1. **A new crate, `oc-wave`, holds the wave optics.** It owns a complex
   scalar field sampled on a plane (`Field2`), the 2D transform, the
   propagators, thin elements, and the report that says what was computed and
   what was approximated. It is f64 throughout: displayed numbers come from
   f64, as everywhere else.

2. **The FFT dependency is confined to `oc-wave`.** It is `rustfft` 6.4.1
   (MIT OR Apache-2.0, MSRV 1.61, pure Rust, builds for
   `wasm32-unknown-unknown`). `oc-core` and `oc-trace` keep their dependency
   policy. The tracer never needs a transform: order efficiencies it will
   need for DMDs and SLMs come from analytic or direct-DFT unit-cell
   coefficients in `oc-core`, and `oc-wave`'s FFT checks them independently.

3. **The complex type is the kernel's own.** `Cx` moved from `thinfilm.rs`
   to an ungated `oc-core/src/complex.rs`, re-exported under the old path, so
   every import kept working and every coating kept its bits (C33). rustfft
   works on `num_complex::Complex<f64>`; the workspace has no `unsafe`, so
   `Fft2` bridges the two types with one copy in and one copy out, into
   buffers it keeps. The bridge was to be measured, and the type switched
   internally if it cost more than 15% of a transform. Measured on the
   baseline machine (release, single thread, `examples/bench_wave.rs`):

   | n² | 2D FFT | bridge copy | share |
   |---|---|---|---|
   | 256² | 0.59 ms | 0.03 ms | 4.7% |
   | 1024² | 11.9 ms | 0.77 ms | 6.5% |
   | 2048² | 51.8 ms | 5.9 ms | 11.3% |

   Under the threshold everywhere, so `Cx` stays.

4. **Fields and rays do not mix units.** A field never feeds a ray's flux,
   and a ray never carries a phase other than `k · OPL`. They meet only
   through optical path length (`Field2::opl_ref_mm`), which both already
   use, and through a planned wave path that propagates a field along a
   sequence extracted from the trace (ADR 0002).

5. **Fields are scalar.** An LCoS modulates only the polarization component
   along its liquid-crystal director; the orthogonal component is carried as
   a second, unmodulated beam and reported, never dropped. Jones fields are
   later work.

   *Amended 2026-10-19 (the wave-optics review, I5).* A chain that meets an
   element telling polarizations apart carries two components, a Jones
   vector at every sample; other chains carry one scalar field standing for
   unpolarized light (`docs/wave-optics.md`, "Two polarization components").

6. **Fidelity is reported the same way as in the tracer.** Every step records
   its method; sampling problems, band limits and model limits are warnings
   or validity flags; power leaving the computation is booked to a cause in a
   `WaveLedger` that must close to `1e-12` of what was launched.

   *Amended 2026-10-20 (the expert review, E1).* The ledger closes by
   construction, so its closure is not the check. Its buckets are in two
   groups, physical and numerical; what is checked on every result is that
   no physical bucket is negative and nothing is detected beyond what was
   launched (`WaveLedger::check`, named in the report's `inconsistencies`),
   and the numerical buckets are reported term by term.

## Consequences

- The README's out-of-scope list can lose "diffraction" and "interference"
  only in scalar form, and only as the propagators land with their
  validation. Rigorous vector diffraction stays out.
- `oc-wasm` and `oc-py` will depend on `oc-wave`; the browser runs it in a
  worker, sliced, keyed by scene revision, like the reference tracer.
- A sampled field has a size. The crate refuses more than `MAX_WAVE_SAMPLES`
  (2²² in the browser, 2²⁶ natively) rather than truncating.
- The work is scheduled as milestones W0–W8 (README roadmap, "M8: It
  diffracts"); the acceptance cases are C35 onward in
  `acceptance-manifest.md`.

## Alternatives

- **`num-complex` everywhere.** Replacing `Cx` would touch the thin-film
  solver and polarization for no physical gain, and risk the bit-identity the
  coating corpus holds.
- **An FFT inside `oc-core`.** Puts a numerical dependency in the scene-model
  crate that every consumer pays for, including ones that never diffract.
- **Field tracing inside the non-sequential tracer** (Wyrowski & Kuhn,
  *J. Mod. Opt.* 58, 449 (2011), doi:10.1080/09500340.2010.532237). The
  general solution, and the wrong first step: it rewrites the tracer's
  contract before there is a validated propagator to trust.
- **Gaussian beamlet decomposition.** Good for coherent fields through
  aberrated optics, but it does not represent a pixelated modulator's sharp
  edges well, and those are the reason this work exists.
