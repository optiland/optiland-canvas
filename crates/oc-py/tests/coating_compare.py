"""Hold the canvas's answers for one case against one oracle's.

Shared by the frozen-fixture test and the live ``tmm`` test so both apply the
same convention maps and the same tolerances.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from coating_cases import critical_deg
from optiland_canvas import coatings

TOLERANCES = json.loads((Path(__file__).resolve().parent / "data" / "coatings" / "tolerances.json").read_text())


def canvas(doc: dict) -> dict:
    """The canvas's powers and amplitudes on the case's grid."""
    args = (doc["stack"], doc["lambda_um"], doc["aoi_deg"], doc["n_incident"])
    return {
        "power": coatings.rt(*args, n_far=doc["n_far"]),
        "amp": coatings.amplitudes(*args, n_exit=doc["n_far"]),
    }


def _to_canvas_frame(oracle: str, key: str, z: complex, deg: float, n0: float, nf: float) -> complex:
    """An oracle amplitude in the kernel's ray frame.

    ``tmm`` already is in it (measured: ``exp(-iwt)``, ``r_p = -r_s`` at normal
    incidence, full-field ``t_p``). Optiland is Macleod's: conjugate ``r``,
    flip the sign of ``r_p``, and rescale ``t_p`` from the tangential field to
    the whole field. ``test_thinfilm_parity.py`` explains each step.
    """
    if oracle == "tmm":
        return z
    sin_i = n0 * math.sin(math.radians(deg))
    cos_0 = math.cos(math.radians(deg))
    cos_f = math.sqrt(1.0 - (sin_i / nf) ** 2)
    return {
        "rs": z.conjugate(),
        "rp": -z.conjugate(),
        "ts": z,
        "tp": z * (cos_0 / cos_f),
    }[key]


def worst(doc: dict, oracle: str, got: dict | None = None) -> dict[str, tuple[float, object]]:
    """The largest disagreement per quantity class, with where it happened."""
    got = got or canvas(doc)
    ref = doc["oracles"][oracle]
    n0, nf = doc["n_incident"], doc["n_far"]
    crit = critical_deg(n0, nf)
    out = {"power": (0.0, None), "amplitude": (0.0, None)}
    for i, lam in enumerate(doc["lambda_um"]):
        for j, deg in enumerate(doc["aoi_deg"]):
            if oracle == "optiland" and crit is not None and deg >= crit:
                continue  # Optiland's exit sits on the growing branch here (spec section 9)
            for key in ("Rs", "Rp", "Ts", "Tp"):
                d = abs(got["power"][key][i][j] - ref[key][i][j])
                if d > out["power"][0]:
                    out["power"] = (d, (lam, deg, key, got["power"][key][i][j], ref[key][i][j]))
            for key in ("rs", "rp", "ts", "tp"):
                pair = ref[key][i][j]
                want = _to_canvas_frame(oracle, key, complex(*pair), deg, n0, nf)
                mine = got["amp"][key][i][j]
                d = abs(mine - want)
                if d > out["amplitude"][0]:
                    out["amplitude"] = (d, (lam, deg, key, mine, want))
    return out


def assert_agrees(doc: dict, oracle: str, got: dict | None = None) -> None:
    tol = TOLERANCES[oracle]
    for kind, (d, at) in worst(doc, oracle, got).items():
        assert d <= tol[kind], f"{doc['case']} against {oracle}, {kind}: worst {d:.3e} at {at}"
