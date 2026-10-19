"""Scalar wave optics from Python: the ``oc-wave`` crate (ADR 0003).

A field is a 2D ``numpy.complex128`` array indexed ``[y, x]``, sampled at
``pitch_mm`` about ``center_mm`` in its plane, with sample ``i`` at
``center + (i - n//2) * pitch``. ``|U|**2`` is flux per mm². Lengths are mm,
the wavelength is in micrometres, as everywhere in the kernel.

Every call returns the kernel's report beside the field: the method that ran,
its sampling warnings, and a power ledger that accounts for what was launched.
See ``docs/wave-optics.md`` for the methods and the sources they follow.

Needs numpy (the ``test`` extra installs it); the rest of the package does not,
so this module is imported on its own: ``from optiland_canvas import wave``.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np

from . import _native

__all__ = [
    "WaveResult", "propagate", "fraunhofer_zoom", "METHODS",
    "dmd", "lcos", "orders", "projector_efficiency", "gerchberg_saxton",
]

#: The propagators :func:`propagate` accepts.
METHODS = ("auto", "asm", "fresnel_tf", "fresnel_ir", "fraunhofer", "rs_direct")


class WaveResult(NamedTuple):
    """A propagated field, its grid, and the kernel's report."""

    field: np.ndarray
    pitch_mm: tuple
    center_mm: tuple
    report: dict


def _pair(v) -> tuple:
    if np.isscalar(v):
        return (float(v), float(v))
    a, b = v
    return (float(a), float(b))


def _to_bytes(u) -> tuple:
    u = np.ascontiguousarray(u, dtype="<c16")
    if u.ndim != 2:
        raise ValueError(f"a field is a 2D array indexed [y, x], not shape {u.shape}")
    ny, nx = u.shape
    return u.tobytes(), (nx, ny)


def _from_bytes(data: bytes, n) -> np.ndarray:
    return np.frombuffer(data, dtype="<c16").reshape(n[1], n[0]).copy()


def propagate(
    u,
    pitch_mm,
    wavelength_um: float,
    z_mm: float,
    method: str = "auto",
    *,
    n_medium: float = 1.0,
    center_mm=(0.0, 0.0),
    carrier=(0.0, 0.0),
    pad: int = 2,
    band_limit: bool = True,
) -> WaveResult:
    """Propagate the field ``u`` a distance ``z_mm`` along its plane's normal.

    ``method`` is one of :data:`METHODS`:

    - ``"asm"``: band-limited angular spectrum;
    - ``"fresnel_tf"`` and ``"fresnel_ir"``: the two Fresnel propagators;
    - ``"fraunhofer"``: returns the far field on its own grid, so check the
      result's ``pitch_mm``;
    - ``"rs_direct"``: direct Rayleigh–Sommerfeld;
    - ``"auto"``: the kernel's choice, recorded in the report.
    """
    data, n = _to_bytes(u)
    out, n2, pitch2, center2, rep = _native.wave_propagate(
        data,
        n,
        _pair(pitch_mm),
        float(wavelength_um),
        float(z_mm),
        method,
        float(n_medium),
        _pair(center_mm),
        _pair(carrier),
        int(pad),
        bool(band_limit),
    )
    return WaveResult(_from_bytes(out, n2), tuple(pitch2), tuple(center2), json.loads(rep))


def fraunhofer_zoom(
    u,
    pitch_mm,
    wavelength_um: float,
    z_mm: float,
    out_n,
    out_pitch_mm,
    out_center_mm=(0.0, 0.0),
    *,
    n_medium: float = 1.0,
    center_mm=(0.0, 0.0),
    carrier=(0.0, 0.0),
) -> WaveResult:
    """The Fraunhofer field ``z_mm`` ahead on any output grid, by the chirp-z
    transform: ``out_n`` samples (``(nx, ny)`` or one number) at
    ``out_pitch_mm`` about ``out_center_mm``."""
    data, n = _to_bytes(u)
    out_n = (int(out_n), int(out_n)) if np.isscalar(out_n) else (int(out_n[0]), int(out_n[1]))
    out, rep = _native.wave_fraunhofer_zoom(
        data,
        n,
        _pair(pitch_mm),
        float(wavelength_um),
        float(z_mm),
        out_n,
        _pair(out_pitch_mm),
        _pair(out_center_mm),
        float(n_medium),
        _pair(center_mm),
        _pair(carrier),
    )
    return WaveResult(_from_bytes(out, out_n), _pair(out_pitch_mm), _pair(out_center_mm), json.loads(rep))


# --------------------------------------------------------------------------- #
# Pixelated modulators (oc_core::modulator, oc_wave::design).


def dmd(pixels, pitch_um, tilt_deg=12.0, tilt_axis="Diagonal", mirror_fill=0.97,
        reflectance=1.0, window_transmittance=1.0, pattern=None) -> dict:
    """A DMD description, the ``SpatialModulator`` the kernel reads.

    ``tilt_axis`` is ``"Diagonal"`` or ``"Side"``. ``pattern`` defaults to all
    mirrors ON (``{"Uniform": {"value": 1}}``; 0 is OFF, 2 is flat).
    """
    px = (int(pixels), int(pixels)) if np.isscalar(pixels) else (int(pixels[0]), int(pixels[1]))
    return {
        "pixels": list(px),
        "pitch_um": float(pitch_um),
        "kind": {"Dmd": {
            "tilt_deg": float(tilt_deg), "tilt_axis": tilt_axis, "mirror_fill": float(mirror_fill),
            "reflectance": float(reflectance), "window_transmittance": float(window_transmittance),
        }},
        "pattern": pattern or {"Uniform": {"value": 1}},
    }


def lcos(pixels, pitch_um, levels=256, design_lambda_um=0.633, pixel_fill=0.93,
         gap_amplitude=0.0, gap_phase_rad=0.0, fringing_um=0.0, reflectance=1.0,
         director_deg=0.0, pattern=None) -> dict:
    """An LCoS SLM description: a linear phase table of ``levels`` grey levels
    over one wave at ``design_lambda_um``. No vendor's numbers are implied by
    the defaults; set them from a datasheet you cite."""
    px = (int(pixels), int(pixels)) if np.isscalar(pixels) else (int(pixels[0]), int(pixels[1]))
    return {
        "pixels": list(px),
        "pitch_um": float(pitch_um),
        "kind": {"Lcos": {
            "director_deg": float(director_deg),
            "lut": {"Linear": {"levels": int(levels), "max_phase_rad": 2.0 * np.pi}},
            "design_lambda_um": float(design_lambda_um), "pixel_fill": float(pixel_fill),
            "gap_amplitude": float(gap_amplitude), "gap_phase_rad": float(gap_phase_rad),
            "fringing_um": float(fringing_um), "reflectance": float(reflectance),
        }},
        "pattern": pattern or {"Uniform": {"value": 0}},
    }


def orders(device: dict, cell, wavelength_um: float, d_in, model: str = "harvey",
           beam: str = "modulated") -> dict:
    """The closed-form diffraction orders of one period ``cell`` (a 2D integer
    array ``[y, x]`` of pixel values) under a unit plane wave along ``d_in``
    (a unit vector travelling toward −Z). ``model`` is ``"harvey"`` or
    ``"tea"``. Returns ``{"total", "propagating", "orders": [...]}``."""
    cell = np.atleast_2d(np.asarray(cell, dtype=np.uint16))
    py, px = cell.shape
    out = _native.modulator_orders(
        json.dumps(device), (px, py), [int(v) for v in cell.ravel()], float(wavelength_um),
        tuple(float(c) for c in d_in), model, beam,
    )
    return json.loads(out)


def projector_efficiency(device: dict, wavelength_um: float, illum_fno: float, proj_fno: float,
                         samples: int = 64, model: str = "harvey") -> float:
    """An all-ON DMD's diffraction efficiency into a projection lens, by the
    method of Texas Instruments DLPA083B §4.2 (see docs/dmd.md)."""
    return _native.dmd_projector_efficiency(
        json.dumps(device), float(wavelength_um), float(illum_fno), float(proj_fno), int(samples), model
    )


def gerchberg_saxton(source, target, iterations: int, start=None):
    """Gerchberg–Saxton between a source-plane amplitude and a far-field
    amplitude on the same ``[y, x]`` grid. Returns ``(phase, errors)``."""
    source = np.ascontiguousarray(source, dtype=float)
    target = np.ascontiguousarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be 2D arrays of the same shape")
    ny, nx = source.shape
    st = None if start is None else [float(v) for v in np.ravel(start)]
    phase, err = _native.gerchberg_saxton(list(source.ravel()), list(target.ravel()), (nx, ny), int(iterations), st)
    return np.asarray(phase).reshape(ny, nx), np.asarray(err)
