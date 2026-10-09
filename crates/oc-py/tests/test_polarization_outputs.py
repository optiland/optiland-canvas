"""Polarization outputs and the conventions they rest on (coating-fidelity
spec CF-06): ellipsometric angles (C19), transmission retardance (C20), and
the Fresnel rhomb's phase (C14), against ``tmm`` and closed forms, and
Optiland's known divergence past the critical angle, pinned to its version.

The map between the canvas and ``tmm`` is measured, not remembered
(``test_thinfilm_tmm.py``): the amplitudes are the same, and so is ψ; ``tmm``
defines Δ from ``−r_p / r_s`` where the canvas uses ``r_p / r_s``, so
``Δ_tmm = Δ − π`` (mod 2π).
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

PINNED_OPTILAND = "0.6.1"


def _wrap(x: float) -> float:
    """An angle difference folded into (-pi, pi]."""
    return math.atan2(math.sin(x), math.cos(x))


def test_c19_silica_on_silicon_reads_the_ellipsometer_tmm_reads():
    cat = coatings.catalog()
    st = {
        "layers": [{"material": cat["silica"], "thickness_nm": 100.0}],
        "backing": {"Opaque": {"substrate": cat["silicon"]}},
        "incidence": "Film",
        "design": None,
    }
    lambdas = [0.30 + 0.025 * i for i in range(21)]
    got = coatings.amplitudes(st, lambdas, [70.0], 1.0, 1.0)
    worst = 0.0
    for i, lam in enumerate(lambdas):
        n_list, d_list = coatings.to_tmm(st, lam, 1.0, 1.0)
        e = tmm.ellips(n_list, d_list, math.radians(70.0), lam)
        psi, delta = got["psi"][i][0], got["delta"][i][0]
        worst = max(worst, abs(psi - e["psi"]), abs(_wrap(delta - math.pi - e["Delta"])))
    assert worst < 1e-10, f"worst {worst:.3e} rad"


def test_ellipsometric_angles_are_the_textbook_ones_on_bare_glass():
    st = cc.stack([])
    got = coatings.amplitudes(st, [0.55], [0.0, math.degrees(math.atan(1.5168))], 1.0, 1.5168)
    # Normal incidence: |r_p| = |r_s|, and r_p = -r_s in the ray frame.
    assert abs(got["psi"][0][0] - math.pi / 4) < 1e-14
    assert abs(_wrap(got["delta"][0][0] - math.pi)) < 1e-14
    # Brewster: r_p vanishes.
    assert got["psi"][0][1] < 1e-7


def _mirror() -> dict:
    hi, lo = 2.35, 1.46
    layers = []
    for _ in range(8):
        layers += [cc.layer(hi, cc.quarter_wave_nm(hi, 550.0, 45.0)), cc.layer(lo, cc.quarter_wave_nm(lo, 550.0, 45.0))]
    layers.append(cc.layer(hi, cc.quarter_wave_nm(hi, 550.0, 45.0)))
    return cc.stack(layers)


def test_c20_transmission_retardance_at_the_band_edges_agrees_with_tmm():
    st = _mirror()
    lambdas = [0.46 + 0.004 * i for i in range(16)] + [0.60 + 0.004 * i for i in range(16)]
    got = coatings.amplitudes(st, lambdas, [45.0], 1.0, 1.5168)
    worst = 0.0
    for i, lam in enumerate(lambdas):
        n_list, d_list = coatings.to_tmm(st, lam, 1.0, 1.5168)
        ts = tmm.coh_tmm("s", n_list, d_list, math.radians(45.0), lam)["t"]
        tp = tmm.coh_tmm("p", n_list, d_list, math.radians(45.0), lam)["t"]
        want = math.atan2(tp.imag, tp.real) - math.atan2(ts.imag, ts.real)
        worst = max(worst, abs(_wrap(got["t_retardance"][i][0] - want)))
    assert worst < 1e-10, f"worst {worst:.3e} rad"
    # And it is not small there, or the test would prove nothing.
    assert max(abs(_wrap(v[0])) for v in got["t_retardance"]) > 0.3


def _rhomb_closed_form(deg: float, n: float) -> float:
    th = math.radians(deg)
    return -2.0 * math.atan(math.cos(th) * math.sqrt(math.sin(th) ** 2 - n * n) / math.sin(th) ** 2)


def test_c14_the_fresnel_rhomb_phase_against_the_closed_form_and_tmm():
    n_glass = 1.5
    angles = [45.0, 50.0, 55.0, 60.0, 70.0, 80.0]
    got = coatings.amplitudes(cc.stack([]), [0.55], angles, n_glass, 1.0)
    for j, deg in enumerate(angles):
        rs, rp = got["rs"][0][j], got["rp"][0][j]
        mine = math.atan2(rp.imag, rp.real) - math.atan2(rs.imag, rs.real)
        assert abs(_wrap(mine - _rhomb_closed_form(deg, 1.0 / n_glass))) < 1e-12, deg
        s = tmm.coh_tmm("s", [n_glass, 1.0], [math.inf, math.inf], math.radians(deg), 0.55)["r"]
        p = tmm.coh_tmm("p", [n_glass, 1.0], [math.inf, math.inf], math.radians(deg), 0.55)["r"]
        theirs = math.atan2(p.imag, p.real) - math.atan2(s.imag, s.real)
        assert abs(_wrap(mine - theirs)) < 1e-12, deg


def test_optiland_puts_the_rhomb_phase_on_the_growing_branch():
    """The known divergence of spec section 9, live, pinned to the Optiland
    version the fixtures were generated with. On any other version it is
    skipped with a message: an upstream fix must be noticed by a person, not
    turned silently into a failure or a pass."""
    optiland = pytest.importorskip("optiland")
    if optiland.__version__ != PINNED_OPTILAND:
        pytest.skip(
            f"Optiland {optiland.__version__} is not the pinned {PINNED_OPTILAND}: revisit this "
            "known divergence (spec R9) before re-pinning"
        )
    from optiland.materials import IdealMaterial
    from optiland.thin_film import ThinFilmStack

    ref = ThinFilmStack(incident_material=IdealMaterial(1.5), substrate_material=IdealMaterial(1.0), layers=[])
    for deg in (50.0, 60.0, 70.0):
        th = [math.radians(deg)]
        rs = complex(ref.compute_rtRTA([0.55], th, polarization="s")["r"][0][0])
        rp = complex(ref.compute_rtRTA([0.55], th, polarization="p")["r"][0][0])
        # Optiland's amplitudes in the canvas's frame (conjugate, flip r_p).
        rs_c, rp_c = rs.conjugate(), -rp.conjugate()
        theirs = math.atan2(rp_c.imag, rp_c.real) - math.atan2(rs_c.imag, rs_c.real)
        right = _rhomb_closed_form(deg, 1.0 / 1.5)
        assert abs(_wrap(theirs - right)) > 1e-3, f"Optiland now agrees at {deg} deg: revisit R9"
        assert abs(_wrap(theirs + right)) < 1e-9, "the divergence is the sign of the retardance"
