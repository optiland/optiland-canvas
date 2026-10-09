"""Coherent films and incoherent slabs, against Byrnes' ``tmm.inc_tmm``.

C17: a 2 mm coated plate with an absorbing substrate and a real rear AR, from
both sides; R, T and every element's absorbed fraction against
``tmm.inc_tmm`` and ``tmm.inc_absorp_in_each_layer`` to 1e-9. And a plate
written the canvas's short way (``Plate`` backing) under ``plate_model:
Incoherent``, which is the same physics: its films, its slab, its bare far
face.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

tmm = pytest.importorskip("tmm")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import coating_cases as cc  # noqa: E402

from optiland_canvas import coatings  # noqa: E402

ANGLES = [0.0, 20.0, 45.0, 70.0]
LAMBDAS = [0.42, 0.55, 0.8, 1.3]


def _slab(n: float, k: float, mm: float) -> dict:
    return {"material": cc.material(n, k, "glass"), "thickness_nm": mm * 1e6, "incoherent": True}


def coated_slab() -> dict:
    return cc.stack(
        [
            cc.layer(1.46, 94.0),
            cc.layer(0.958, 5.0, k=6.69),
            cc.layer(2.35, 61.0),
            _slab(1.5168, 3e-6, 2.0),
            cc.layer(1.65, 83.0),
            cc.layer(1.38, 99.0),
        ]
    )


def _tmm_lists(st: dict, lam: float, n0: float, nf: float):
    layers = st["layers"]
    order = list(range(len(layers)))
    if st.get("incidence") == "Substrate":
        order.reverse()
    ns = coatings.indices(st, lam)
    n_list = [complex(n0)] + [ns[i] for i in order] + [complex(nf)]
    d_list = [math.inf] + [layers[i]["thickness_nm"] / 1000.0 for i in order] + [math.inf]
    c_list = ["i"] + ["i" if layers[i].get("incoherent") else "c" for i in order] + ["i"]
    return n_list, d_list, c_list, order


def _compare(st: dict, n0: float, nf: float, plate_incoherent: bool = False, lists=None):
    got = coatings.absorption(st, LAMBDAS, ANGLES, n0, nf, plate_incoherent=plate_incoherent)
    worst = 0.0
    for i, lam in enumerate(LAMBDAS):
        n_list, d_list, c_list, order = lists(lam) if lists else _tmm_lists(st, lam, n0, nf)
        for j, deg in enumerate(ANGLES):
            for pol in "sp":
                data = tmm.inc_tmm(pol, n_list, d_list, c_list, math.radians(deg), lam)
                absorbed = tmm.inc_absorp_in_each_layer(data)[1:-1]
                worst = max(worst, abs(got[f"R{pol}"][i][j] - data["R"]), abs(got[f"T{pol}"][i][j] - data["T"]))
                mine = list(got[f"A_{pol}"][i][j])
                if plate_incoherent:
                    mine = mine + [got[f"sub_{pol}"][i][j]]
                want = [0.0] * len(order)
                for walked, written in enumerate(order):
                    want[written] = absorbed[walked]
                for a, b in zip(mine, want):
                    worst = max(worst, abs(a - b))
    return worst


@pytest.mark.parametrize("side", ["Film", "Substrate"])
def test_c17_a_coated_slab_with_a_rear_ar_agrees_with_inc_tmm(side):
    st = coated_slab()
    st["incidence"] = side
    worst = _compare(st, 1.0, 1.0)
    assert worst < 1e-9, f"{side}: worst {worst:.3e}"


def test_c17_a_plate_under_incoherent_is_the_same_physics():
    films = [cc.layer(1.46, 94.0), cc.layer(0.958, 5.0, k=6.69)]
    plate = {
        "layers": films,
        "backing": {"Plate": {"substrate": cc.material(1.5168, 3e-6, "glass"), "exit": "Bare", "thickness_mm": 2.0}},
        "incidence": "Film",
        "design": None,
    }

    def lists(lam):
        ns = coatings.indices(plate, lam) + [coatings.substrate_index(plate, lam)]
        n_list = [1.0 + 0j] + ns + [1.0 + 0j]
        d_list = [math.inf, 0.094, 0.005, 2000.0, math.inf]
        return n_list, d_list, ["i", "c", "c", "i", "i"], [0, 1, 2]

    worst = _compare(plate, 1.0, 1.0, plate_incoherent=True, lists=lists)
    assert worst < 1e-9, f"worst {worst:.3e}"
