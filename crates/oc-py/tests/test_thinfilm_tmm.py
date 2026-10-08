"""The canvas thin-film kernel against Byrnes' ``tmm``, run live.

``tmm`` is an independent transfer-matrix code: not Macleod's admittance form,
which the canvas and Optiland share, so the two agreeing is evidence rather
than one algorithm checked against its own transcription. It also sits on the
decaying evanescent branch past the critical angle, which Optiland does not.

The frozen version of the same comparison is ``test_coating_fixtures.py``; this
one runs the oracle itself, so it catches a fixture that was regenerated
wrong, and adds a seeded sweep of random absorbing stacks.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

tmm = pytest.importorskip("tmm")
np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coating_cases  # noqa: E402
import coating_compare  # noqa: E402
import gen_coating_fixtures  # noqa: E402

from optiland_canvas import coatings  # noqa: E402


def test_tmm_works_in_the_canvas_ray_frame():
    """The convention map, measured rather than remembered (spec section 9).

    One layer, n = 1.72, 137 nm on n = 1.52 at 550 nm, the case
    ``a_single_layer_matches_the_airy_formula_and_fixes_the_sign`` pins in
    Rust. ``tmm`` returns the canvas's amplitudes with no conversion at all:
    the same time convention (``exp(-iwt)``), the same p basis (``r_p = -r_s``
    at normal incidence) and the whole field in ``t_p``.
    """
    st = coating_cases.stack([coating_cases.layer(1.72, 137.0)])
    got = coatings.amplitudes(st, [0.55], [0.0, 35.0], 1.0, 1.52)
    n_list, d_list = coatings.to_tmm(st, 0.55, 1.0, 1.52)
    for j, deg in enumerate((0.0, 35.0)):
        for pol in "sp":
            o = tmm.coh_tmm(pol, n_list, d_list, math.radians(deg), 0.55)
            assert abs(got[f"r{pol}"][0][j] - o["r"]) < 1e-14
            assert abs(got[f"t{pol}"][0][j] - o["t"]) < 1e-14
    # At normal incidence the p basis flips on reflection in both.
    o_s = tmm.coh_tmm("s", n_list, d_list, 0.0, 0.55)
    o_p = tmm.coh_tmm("p", n_list, d_list, 0.0, 0.55)
    assert abs(o_p["r"] + o_s["r"]) < 1e-15


@pytest.mark.parametrize("name", list(coating_cases.CASES))
def test_every_case_agrees_with_tmm(name):
    stack, n0, nf, description = coating_cases.CASES[name]
    lambdas, angles = coating_cases.lambdas_for(name), coating_cases.angles_for(name)
    doc = {
        "case": name,
        "stack": stack,
        "n_incident": n0,
        "n_far": nf,
        "lambda_um": lambdas,
        "aoi_deg": angles,
        "oracles": {"tmm": gen_coating_fixtures._tmm(stack, lambdas, angles, n0, nf)},
    }
    coating_compare.assert_agrees(doc, "tmm")


def _random_stack(rng: random.Random) -> list[dict]:
    """Up to 20 layers, n in [1.3, 4], one in three absorbing with k up to 1.

    The total absorption is kept below the overflow the baseline cannot handle
    (``known_defect_a_very_thick_metal_film_overflows`` in Rust); CF-02 widens
    this to 64 layers with k up to 7.
    """
    return [
        coating_cases.layer(
            rng.uniform(1.3, 4.0),
            rng.uniform(1.0, 400.0),
            k=rng.uniform(0.0, 1.0) if rng.random() < 1 / 3 else 0.0,
        )
        for _ in range(rng.randint(1, 20))
    ]


@pytest.mark.parametrize("seed", range(10))
def test_random_absorbing_stacks_agree_with_tmm(seed):
    """Twenty seeded stacks per seed, both sides, through the critical angle."""
    rng = random.Random(seed)
    angles = [0.0, 30.0, 60.0, 85.0]
    for case in range(20):
        layers = _random_stack(rng)
        ns = rng.uniform(1.4, 2.0)
        lam = rng.uniform(0.4, 1.6)
        incidence = "Substrate" if case % 2 else "Film"
        n0, nf = (ns, 1.0) if incidence == "Substrate" else (1.0, ns)
        doc = {
            "case": f"seed {seed} case {case}",
            "stack": coating_cases.stack(layers, incidence),
            "n_incident": n0,
            "n_far": nf,
            "lambda_um": [lam],
            "aoi_deg": angles,
        }
        doc["oracles"] = {"tmm": gen_coating_fixtures._tmm(doc["stack"], [lam], angles, n0, nf)}
        coating_compare.assert_agrees(doc, "tmm")


def _tmm_layer_absorption(pol, n_list, d_list, deg, lam):
    o = tmm.coh_tmm(pol, n_list, d_list, math.radians(deg), lam)
    return [float(x) for x in tmm.absorp_in_each_layer(o)][1:-1]


@pytest.mark.parametrize("name", ["aluminium_film", "aluminium_under_dielectric", "asymmetric_absorbing_forward",
                                  "asymmetric_absorbing_reversed", "dispersive_catalogue_stack"])
def test_c15_absorption_by_layer_agrees_with_tmm(name):
    """Each layer's absorbed fraction against ``tmm.absorp_in_each_layer``,
    to 1e-9, for every absorbing case, from either side."""
    stack, n0, nf, _ = coating_cases.CASES[name]
    lambdas, angles = coating_cases.lambdas_for(name), coating_cases.angles_for(name)
    got = coatings.absorption(stack, lambdas, angles, n0, nf)
    order = coatings.layer_order(stack)
    worst = 0.0
    for i, lam in enumerate(lambdas):
        n_list, d_list = coatings.to_tmm(stack, lam, n0, nf)
        for j, deg in enumerate(angles):
            for pol in "sp":
                want_walked = _tmm_layer_absorption(pol, n_list, d_list, deg, lam)
                want = [0.0] * len(order)
                for walked, written in enumerate(order):
                    want[written] = want_walked[walked]
                mine = got[f"A_{pol}"][i][j]
                for a, b in zip(mine, want):
                    worst = max(worst, abs(a - b))
                closes = got[f"R{pol}"][i][j] + got[f"T{pol}"][i][j] + sum(mine)
                assert abs(closes - 1.0) < 1e-12, (name, lam, deg, pol, closes)
    assert worst < 1e-9, f"{name}: worst {worst:.3e}"


def test_c15_three_layers_with_an_absorbing_middle():
    """C15 itself: only the absorbing layer absorbs, as tmm says it does."""
    st = coating_cases.stack(
        [coating_cases.layer(1.46, 94.0), coating_cases.layer(0.958, 8.0, k=6.69), coating_cases.layer(2.35, 61.0)]
    )
    got = coatings.absorption(st, [0.45, 0.55, 0.7], [0.0, 30.0, 60.0], 1.0, 1.5168)
    for i, lam in enumerate([0.45, 0.55, 0.7]):
        n_list, d_list = coatings.to_tmm(st, lam, 1.0, 1.5168)
        for j, deg in enumerate([0.0, 30.0, 60.0]):
            for pol in "sp":
                want = _tmm_layer_absorption(pol, n_list, d_list, deg, lam)
                mine = got[f"A_{pol}"][i][j]
                assert mine[0] == 0.0 and mine[2] == 0.0
                assert abs(mine[1] - want[1]) < 1e-9
                assert min(mine) >= -1e-15
