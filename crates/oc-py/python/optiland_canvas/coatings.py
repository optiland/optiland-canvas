"""The canvas's thin-film kernel, and the same stack handed to two oracles.

The canvas evaluates coatings in Rust. This module exposes that evaluation to
Python and translates a canvas stack into the inputs of two independent
libraries, so that a test can run all three and subtract:

- `Byrnes' tmm <https://pypi.org/project/tmm/>`_ (``to_tmm``), an independent
  transfer-matrix code in the same ``exp(-iwt)`` convention as the canvas;
- Optiland's ``thin_film`` (``to_optiland``), Macleod's admittance form, which
  the canvas mirrors on purpose.

Every oracle is fed the complex indices **the canvas sampled**
(:func:`indices`), never its own reading of the material table. A disagreement
is then the solver's and not a second implementation of the lookup.

A stack is the JSON of ``oc_core::thinfilm::Coating``, as a ``dict`` or a
string. Wavelengths are vacuum micrometres, angles degrees in the incident
medium, thicknesses nanometres inside the stack.

Conventions of the returned amplitudes (the kernel's ray frame, see
``docs/physics/conventions.md``): ``exp(-iwt)``; ``r_p = -r_s`` at normal
incidence; ``t_p`` the whole field, not its tangential part. ``tmm`` uses the
same three, so its amplitudes compare without a map; Optiland's need the three
steps written out in ``tests/test_thinfilm_parity.py``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Union

from . import _native

__all__ = ["rt", "amplitudes", "indices", "substrate_index", "layer_order", "to_tmm", "to_optiland"]

Stack = Union[dict, str]


def _json(stack: Stack) -> str:
    return stack if isinstance(stack, str) else json.dumps(stack)


def _dict(stack: Stack) -> dict:
    return json.loads(stack) if isinstance(stack, str) else stack


def _grid(flat: list, n_lambda: int, n_aoi: int) -> list[list[float]]:
    return [flat[i * n_aoi : (i + 1) * n_aoi] for i in range(n_lambda)]


def rt(
    stack: Stack,
    lambdas_um: list[float],
    aois_deg: list[float],
    n_incident: float = 1.0,
    n_far: float | None = None,
) -> dict[str, list[list[float]]]:
    """Powers ``Rs, Rp, Ts, Tp``, each indexed ``[wavelength][angle]``.

    ``n_far`` is the medium behind a ``Medium``-backed stack; it defaults to
    ``n_incident``, which is what a zero-thickness surface in one medium (every
    plate in a scene) sees.
    """
    lambdas_um, aois_deg = list(map(float, lambdas_um)), list(map(float, aois_deg))
    got = json.loads(_native.thinfilm_rt(_json(stack), lambdas_um, aois_deg, float(n_incident), n_far))
    shape = (len(lambdas_um), len(aois_deg))
    return {
        "Rs": _grid(got["rs"], *shape),
        "Rp": _grid(got["rp"], *shape),
        "Ts": _grid(got["ts"], *shape),
        "Tp": _grid(got["tp"], *shape),
    }


def amplitudes(
    stack: Stack,
    lambdas_um: list[float],
    aois_deg: list[float],
    n_incident: float = 1.0,
    n_exit: float = 1.0,
) -> dict[str, list[list[Any]]]:
    """Complex ``rs, rp, ts, tp`` and the power ``scale``, ``[wavelength][angle]``.

    ``|t|^2 * scale`` is the transmittance. Only a ``Medium``-backed stack has
    amplitudes: a plate adds its two faces incoherently, which no single
    Jones matrix describes, and raises ``ValueError``.
    """
    lambdas_um, aois_deg = list(map(float, lambdas_um)), list(map(float, aois_deg))
    got = json.loads(
        _native.thinfilm_amplitudes(_json(stack), lambdas_um, aois_deg, float(n_incident), float(n_exit))
    )
    shape = (len(lambdas_um), len(aois_deg))
    out: dict[str, list[list[Any]]] = {}
    for key in ("rs", "rp", "ts", "tp"):
        flat = [complex(a, b) for a, b in zip(got[f"{key}_re"], got[f"{key}_im"])]
        out[key] = _grid(flat, *shape)
    out["scale"] = _grid(got["scale"], *shape)
    return out


def indices(stack: Stack, lambda_um: float) -> list[complex]:
    """Each layer's ``n + ik`` at ``lambda_um``, in the written order, as the
    kernel samples it."""
    got = json.loads(_native.thinfilm_indices(_json(stack), [float(lambda_um)]))
    return [complex(a, b) for a, b in zip(got["n_re"][0], got["n_im"][0])]


def substrate_index(stack: Stack, lambda_um: float) -> complex | None:
    """The substrate's ``n + ik`` for a ``Plate`` or ``Opaque`` backing, as the
    kernel samples it; ``None`` for ``Medium``."""
    got = json.loads(_native.thinfilm_indices(_json(stack), [float(lambda_um)]))
    return None if got["sub"] is None else complex(*got["sub"][0])


def _far_index(c: dict, lambda_um: float, n_far: float) -> complex:
    """The medium behind the stack: the opaque substrate, or the scene's."""
    if isinstance(c.get("backing"), dict) and "Opaque" in c["backing"]:
        return substrate_index(c, lambda_um)
    return complex(n_far)


def layer_order(stack: Stack) -> list[int]:
    """The written layer indices in the order the light meets them.

    A ``Medium``-backed stack met from its substrate (``incidence:
    "Substrate"``) is walked last layer first. A plate-backed stack is not a
    single coherent stack and has no order to give here.
    """
    c = _dict(stack)
    opaque = isinstance(c.get("backing"), dict) and "Opaque" in c["backing"]
    if c.get("backing") != "Medium" and not opaque:
        raise ValueError("only a Medium- or Opaque-backed stack is one coherent stack")
    order = list(range(len(c["layers"])))
    if c.get("incidence", "Film") == "Substrate":
        order.reverse()
    return order


def to_tmm(
    stack: Stack, lambda_um: float, n_incident: float = 1.0, n_far: float | None = None
) -> tuple[list[complex], list[float]]:
    """``(n_list, d_list)`` for ``tmm.coh_tmm`` at ``lambda_um``.

    Thicknesses come out in micrometres, the unit the wavelength is given in,
    which is all ``tmm`` asks. The two semi-infinite media carry ``inf``.
    """
    c = _dict(stack)
    n_far = n_incident if n_far is None else n_far
    ns = indices(c, lambda_um)
    order = layer_order(c)
    n_list = [complex(n_incident)] + [ns[i] for i in order] + [_far_index(c, lambda_um, n_far)]
    d_list = [math.inf] + [c["layers"][i]["thickness_nm"] / 1000.0 for i in order] + [math.inf]
    return n_list, d_list


def to_optiland(
    stack: Stack, lambda_um: float, n_incident: float = 1.0, n_far: float | None = None
):
    """An ``optiland.thin_film.ThinFilmStack`` of the same stack at one
    wavelength.

    Every layer becomes an ``IdealMaterial`` at the index the canvas sampled
    at ``lambda_um``, so the stack is only valid at that wavelength. That is
    deliberate: it compares solvers, not material models.
    """
    from optiland.materials import IdealMaterial
    from optiland.thin_film import Layer, ThinFilmStack

    c = _dict(stack)
    n_far = n_incident if n_far is None else n_far
    ns = indices(c, lambda_um)
    layers = [
        Layer(IdealMaterial(ns[i].real, k=ns[i].imag), thickness_um=c["layers"][i]["thickness_nm"] / 1000.0)
        for i in layer_order(c)
    ]
    far = _far_index(c, lambda_um, n_far)
    return ThinFilmStack(
        incident_material=IdealMaterial(n_incident),
        substrate_material=IdealMaterial(far.real, k=far.imag),
        layers=layers,
    )
