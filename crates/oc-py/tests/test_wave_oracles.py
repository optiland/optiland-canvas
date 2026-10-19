"""The wave kernel's transforms against Optiland's PSF code (acceptance case C45).

These are oracle checks: they verify, they do not validate (docs/validation.md).
Both engines take the Fourier transform of the same sampled pupil, so agreement
shows that the canvas's Fraunhofer step and its chirp-z zoom index, centre and
scale the transform as Optiland's ``ScalarFFTPSF`` and ``MMDFTPSF`` do. It is no
evidence about the physics of a lens. The physics is held by the closed-form
tests in ``crates/oc-wave/tests``.

**Oracle.** Optiland, cloned from https://github.com/optiland/optiland on
2026-10-19 at ``master`` = ``1e0513902c68948a727394c6226f3580c7261c65``
(committed 2026-10-14), installed editable in its own virtual environment. That
is newer than the ``optiland==0.6.1`` the coating fixtures pin. Each test
prints the checkout it ran against, so a rerun on another commit says so.

**Conventions.** Optiland writes the pupil as ``A·exp(−i2π·OPD)`` with OPD in
waves, the opposite sign to the canvas's ``exp(+ik·OPL)``. Here that does not
matter: the pupil array is handed over as Optiland made it, so both engines
transform the same numbers. For a PSF built by the canvas from its own phase
the image would come out point-reflected relative to Optiland's, and a later
comparison of two independently built pupils has to account for that.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

optiland = pytest.importorskip("optiland")
# Both classes are in today's master; a release without them skips this module
# with the reason, rather than failing on an import.
ScalarFFTPSF = pytest.importorskip("optiland.psf.fft").ScalarFFTPSF
MMDFTPSF = pytest.importorskip("optiland.psf.mmdft").MMDFTPSF
from optiland.samples.objectives import CookeTriplet  # noqa: E402

from optiland_canvas import wave  # noqa: E402


def _oracle_checkout() -> str:
    """Where the Optiland under test came from, and its commit if it is a clone."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(optiland.__file__)))
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "not a git checkout"
    return f"{root} @ {sha}"


# An off-axis field of a Cooke triplet: an aberrated pupil, so the comparison
# exercises the phase and not only an aperture.
FIELD = (0.0, 1.0)
WAVELENGTH_UM = 0.55
NUM_RAYS = 64
# Arbitrary physical sampling for the canvas side: the comparison is of the
# transform, and these keep the source Fresnel number far below 0.1.
DX_MM = 1e-3
Z_MM = 1e4


def _canvas_scale(lam_um: float) -> float:
    """|U₂|²·(λz)²/Δx⁴ = |Σ pupil·e^{−i2πfx}|², the transform Optiland squares."""
    return (lam_um / 1000.0 * Z_MM) ** 2 / DX_MM**4


def test_fraunhofer_matches_optiland_scalar_fft_psf_on_its_own_pupil(capsys):
    lens = CookeTriplet()
    psf = ScalarFFTPSF(lens, field=FIELD, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS, grid_size=256)
    pupil = np.asarray(psf.pupils[0])
    count = np.sum(np.abs(pupil) > 0)
    ref = np.asarray(psf.psf)
    assert np.ptp(np.angle(pupil[np.abs(pupil) > 0])) > 0.1, "the pupil should be aberrated"

    # Padding 64 → 256 places the pupil exactly where Optiland's symmetric pad
    # does, and both transforms put zero frequency at index 128.
    res = wave.propagate(pupil, DX_MM, WAVELENGTH_UM, Z_MM, "fraunhofer", pad=4)
    assert res.field.shape == ref.shape
    assert not [w for w in res.report["warnings"] if "OutsideRegime" in w]
    ours = np.abs(res.field) ** 2 * _canvas_scale(WAVELENGTH_UM) * 100.0 / count**2
    err = np.max(np.abs(ours - ref)) / np.max(ref)
    with capsys.disabled():
        print(f"\noracle {_oracle_checkout()}")
        print(f"measured: Fraunhofer vs Optiland ScalarFFTPSF, worst / peak = {err:.3e}")
    assert err < 1e-10, err


def test_zoomed_psf_matches_optiland_mmdft(capsys):
    lens = CookeTriplet()
    image_size = 101
    probe = MMDFTPSF(lens, field=FIELD, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS, image_size=image_size)
    fno = probe._get_working_FNO()
    lam = probe.wavelengths[0].value
    # A pixel pitch that puts the image samples off the FFT grid: Optiland's
    # pad size, λ·F#·(n − 1)/pitch, becomes 373.7, not an integer.
    pad_size = 3.7 * image_size
    pixel_pitch = lam * fno * (NUM_RAYS - 1) / pad_size
    psf = MMDFTPSF(
        lens, field=FIELD, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS,
        image_size=image_size, pixel_pitch=pixel_pitch,
    )
    pupil = np.asarray(psf.pupil)
    count = np.sum(np.abs(pupil) > 0)
    ref = np.asarray(psf.psf)

    # Optiland's kernel is exp(−i2π (m − M//2)(p − n//2) / pad_size): the canvas
    # zoom at frequency step 1/(pad_size·Δx), sample m at (m − M//2)·step.
    lz = lam / 1000.0 * Z_MM
    out_pitch = lz / (pad_size * DX_MM)
    res = wave.fraunhofer_zoom(pupil, DX_MM, lam, Z_MM, image_size, out_pitch)
    ours = np.abs(res.field) ** 2 * _canvas_scale(lam) * 100.0 / count**2
    err = np.max(np.abs(ours - ref)) / np.max(ref)
    with capsys.disabled():
        print(f"\noracle {_oracle_checkout()}")
        print(f"measured: chirp-z zoom vs Optiland MMDFTPSF (pad size {pad_size}), worst / peak = {err:.3e}")
    assert err < 1e-10, err


def test_a_field_round_trips_through_the_bytes_boundary():
    # complex128 in C order is the wire format: nothing reordered or rescaled.
    rng = np.random.default_rng(3)
    u = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    res = wave.propagate(u, 0.01, 0.5, 0.0)
    assert np.array_equal(res.field, u)
    assert res.report["ledger"]["launched"] == pytest.approx(np.sum(np.abs(u) ** 2) * 1e-4, rel=1e-12)
