"""The built-in film catalogue against the files it was imported from.

CF-03's exit criteria: every imported table reproduces its source file at the
knots to 1e-12 in n and k, and every formula entry agrees with Optiland's own
evaluation of the same file to 1e-12 over its declared range. The source is
the refractiveindex.info database as the installed Optiland packages it,
which is what ``tools/ri_import.py`` read; the canvas's numbers are read back
through the kernel (``coatings.indices``), so the test covers the lookup and
not a second copy of it.
"""

from __future__ import annotations

import urllib.parse
import warnings
from pathlib import Path

import pytest

optiland = pytest.importorskip("optiland")
yaml = pytest.importorskip("yaml")

from optiland_canvas import coatings  # noqa: E402

DB = Path(optiland.__file__).resolve().parent / "database" / "data-nk"
CATALOG = coatings.catalog()


def _source(material: dict) -> Path:
    q = urllib.parse.parse_qs(urllib.parse.urlparse(material["provenance"]["url"]).query)
    return DB / q["shelf"][0] / q["book"][0] / f"{q['page'][0]}.yml"


def _one_layer(material: dict) -> dict:
    return {"layers": [{"material": material, "thickness_nm": 100.0}], "backing": "Medium", "incidence": "Film", "design": None}


def _rows(block: dict) -> list[list[float]]:
    return [[float(x) for x in r.split()] for r in block["data"].strip().splitlines()]


def test_the_catalogue_has_what_the_spec_lists():
    want = {
        "magnesium_fluoride", "magnesium_fluoride_bulk", "silica", "alumina", "titania", "tantala", "hafnia", "zirconia", "niobia",
        "zinc_sulfide", "silicon", "silver", "silver_johnson", "aluminium", "gold", "chromium", "nickel",
    }
    assert set(CATALOG) == want
    for key, m in CATALOG.items():
        p = m["provenance"]
        assert p["source_kind"] == "Database" and p["license"] == "CC0-1.0", key
        assert _source(m).exists(), f"{key}: {_source(m)}"


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_each_table_reproduces_its_source_at_the_knots(key):
    m = CATALOG[key]
    doc = yaml.safe_load(_source(m).read_text(encoding="utf-8"))
    lo, hi = m["provenance"]["range_um"]
    st = _one_layer(m)
    checked = 0
    for block in doc["DATA"]:
        t = block["type"]
        if not t.startswith("tabulated"):
            continue
        cols = {"tabulated nk": ("n", "k"), "tabulated n": ("n",), "tabulated k": ("k",)}[t]
        for row in _rows(block):
            lam = row[0]
            if not (lo <= lam <= hi) and t != "tabulated k":
                continue
            got = coatings.indices(st, lam)[0]
            for col, want in zip(cols, row[1:]):
                mine = got.real if col == "n" else got.imag
                assert abs(mine - want) <= 1e-12, f"{key} {col} at {lam} um: {mine} vs {want}"
                checked += 1
    if m.get("formula") is None:
        assert checked > 10, f"{key}: only {checked} knots"


@pytest.mark.parametrize("key", sorted(k for k, m in CATALOG.items() if m.get("formula")))
def test_each_formula_agrees_with_optiland_on_the_same_file(key):
    from optiland.materials import MaterialFile

    m = CATALOG[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = MaterialFile(str(_source(m)))
    lo, hi = m["formula"]["range_um"]
    st = _one_layer(m)
    worst = 0.0
    for i in range(201):
        lam = lo + (hi - lo) * i / 200
        mine = coatings.indices(st, lam)[0].real
        want = float(ref.n(lam)[0]) if hasattr(ref.n(lam), "__len__") else float(ref.n(lam))
        worst = max(worst, abs(mine - want))
    assert worst <= 1e-12, f"{key}: worst {worst:.3e}"


def test_a_film_says_it_is_a_film_and_the_crystal_says_why_it_is_there():
    assert CATALOG["magnesium_fluoride"]["form"] == "Film"
    z = CATALOG["zirconia"]
    assert z["form"] == "Crystal"
    assert "no ZrO2 film data" in z["provenance"]["note"]
