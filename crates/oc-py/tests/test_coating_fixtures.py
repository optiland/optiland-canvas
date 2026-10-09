"""The canvas against the frozen ``tmm`` and Optiland answers.

``gen_coating_fixtures.py`` wrote what both oracles say about every case in
``coating_cases``; this holds the canvas to those files without either oracle
installed. A failure here is the kernel moving away from two independent
references, not a library release moving under it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import coating_cases  # noqa: E402
import coating_compare  # noqa: E402
import gen_coating_fixtures  # noqa: E402

FIXTURES = HERE / "data" / "coatings"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_every_case_has_a_fixture_and_every_fixture_a_case():
    on_disk = {p.stem for p in FIXTURES.glob("*.json")} - {"tolerances"}
    assert on_disk == set(coating_cases.CASES)


def test_the_fixtures_were_made_by_the_scripts_in_the_tree():
    """Edit a case or the generator and the fixtures must be regenerated."""
    want = gen_coating_fixtures.script_hash()
    stale = [n for n in coating_cases.CASES if _load(n)["generator"]["script_sha256"] != want]
    assert not stale, f"regenerate with `python tests/gen_coating_fixtures.py`: {stale}"


def _same(a, b) -> bool:
    """Equal, floats to a few ulp: a quarter wave at 45 degrees goes through
    `math.cos`, which differs in the last bit between platforms; the frozen
    stack is the one the oracles were fed, and the one compared."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-15, abs_tol=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("name", list(coating_cases.CASES))
def test_the_fixture_describes_the_case(name):
    doc = _load(name)
    stack, n0, nf, _ = coating_cases.CASES[name]
    assert _same(doc["stack"], stack) and doc["n_incident"] == n0 and doc["n_far"] == nf
    assert doc["lambda_um"] == coating_cases.lambdas_for(name) and doc["aoi_deg"] == coating_cases.angles_for(name)


@pytest.mark.parametrize("name", list(coating_cases.CASES))
def test_the_canvas_samples_the_indices_the_oracles_were_fed(name):
    """The material lookup, separately from the solver."""
    from optiland_canvas import coatings

    doc = _load(name)
    for lam, row in zip(doc["lambda_um"], doc["indices_fed"]):
        assert [[z.real, z.imag] for z in coatings.indices(doc["stack"], lam)] == row


@pytest.mark.parametrize("oracle", ["tmm", "optiland"])
@pytest.mark.parametrize("name", list(coating_cases.CASES))
def test_the_canvas_agrees_with_the_frozen_oracle(name, oracle):
    coating_compare.assert_agrees(_load(name), oracle)


def test_optiland_diverges_past_the_critical_angle_when_the_stack_absorbs():
    """A known divergence, frozen with the Optiland version that shows it.

    Past the critical angle Optiland's ``thin_film`` puts the exit medium on
    the growing evanescent branch. For a lossless stack only the phase notices;
    for one that absorbs, the reflected **power** is wrong as well (attenuated
    total reflection), here by tens of percent, while the canvas and ``tmm``
    agree to rounding. The canvas keeps the decaying branch; the upstream
    report is the spec's R9. If a regenerated fixture from a fixed Optiland
    release removes the gap, this test must be revisited, not loosened.
    """
    doc = _load("asymmetric_absorbing_reversed")
    assert doc["oracles"]["optiland"]["version"] == "0.6.1"
    crit = coating_cases.critical_deg(doc["n_incident"], doc["n_far"])
    gap = max(
        abs(doc["oracles"]["optiland"][k][i][j] - doc["oracles"]["tmm"][k][i][j])
        for k in ("Rs", "Rp")
        for i in range(len(doc["lambda_um"]))
        for j, deg in enumerate(doc["aoi_deg"])
        if deg >= crit
    )
    assert gap > 0.1, f"Optiland now agrees past the critical angle (gap {gap:.2e}): revisit R9 and this test"
