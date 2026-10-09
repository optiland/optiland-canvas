"""Thick absorbing stacks: the canvas against an arbitrary-precision oracle.

CF-02 asks that 2000 random 64-layer absorbing stacks (n in [1.3, 4], k in
[0, 7], d in [1, 3000] nm) agree with an oracle to 1e-9 at 0, 30, 60 and 85
degrees. ``tmm`` 0.2.0 cannot be that oracle: its double-precision matrix
product overflows on every such stack (measured: 1200 of 1200 NaN). So the
oracle is ``mp_tmm``, Byrnes' formulation in 40-digit arithmetic, with no
opacity rule and no rescaling, which is held to ``tmm`` first on stacks
``tmm`` can do.

The canvas truncates a stack at its first opaque layer (Im delta > 35) and
rescales its matrix; the exact answer differs from the truncated one by less
than exp(-70), which is what this measures.
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

import pytest

mp = pytest.importorskip("mpmath")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import coating_cases  # noqa: E402
import mp_tmm  # noqa: E402

from optiland_canvas import coatings  # noqa: E402

ANGLES = [0.0, 30.0, 60.0, 85.0]
TOL = 1e-9


def test_the_oracle_agrees_with_tmm_where_tmm_can_answer():
    tmm = pytest.importorskip("tmm")
    rng = random.Random(7)
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(60):
            ns = [
                complex(rng.uniform(1.3, 4.0), rng.uniform(0.0, 1.0) if rng.random() < 0.4 else 0.0)
                for _ in range(rng.randint(1, 12))
            ]
            ds = [rng.uniform(1.0, 400.0) / 1000.0 for _ in ns]
            n_list, d_list = [1.0] + ns + [1.52], [math.inf] + ds + [math.inf]
            for deg in ANGLES:
                for pol in "sp":
                    a = tmm.coh_tmm(pol, n_list, d_list, math.radians(deg), 0.55)
                    b = mp_tmm.coh(pol, n_list, d_list, math.radians(deg), 0.55)
                    for k in ("r", "t"):
                        worst = max(worst, abs(complex(a[k]) - complex(b[k])))
                    for k in ("R", "T"):
                        worst = max(worst, abs(float(a[k]) - float(b[k])))
    assert worst < 1e-12, worst


def _stack(rng: random.Random) -> list[dict]:
    return [
        coating_cases.layer(rng.uniform(1.3, 4.0), rng.uniform(1.0, 3000.0), k=rng.uniform(0.0, 7.0))
        for _ in range(64)
    ]


@pytest.mark.parametrize("chunk", range(20))
def test_random_64_layer_absorbing_stacks_agree_with_the_oracle(chunk):
    """Twenty chunks of a hundred stacks, seeded, so a failure names its seed."""
    worst, where = 0.0, None
    for case in range(100):
        seed = chunk * 100 + case
        rng = random.Random(seed)
        layers = _stack(rng)
        st = coating_cases.stack(layers)
        got = coatings.rt(st, [0.55], ANGLES, 1.0, 1.52)
        ns = coatings.indices(st, 0.55)
        n_list = [1.0] + ns + [1.52]
        d_list = [math.inf] + [l["thickness_nm"] / 1000.0 for l in layers] + [math.inf]
        for j, deg in enumerate(ANGLES):
            for pol in "sp":
                ref = mp_tmm.coh(pol, n_list, d_list, math.radians(deg), 0.55)
                for k, want in (("R", float(ref["R"])), ("T", float(ref["T"]))):
                    mine = got[f"{k}{pol}"][0][j]
                    assert math.isfinite(mine), f"seed {seed}, {deg} deg, {k}{pol}"
                    d = abs(mine - want)
                    if d > worst:
                        worst, where = d, (seed, deg, f"{k}{pol}", mine, want)
    assert worst <= TOL, f"worst {worst:.3e} at {where}"
