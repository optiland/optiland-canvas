"""The canvas thin-film calculation against Optiland's.

Both sides run the transfer-matrix method over the same stack, and the canvas
side follows Macleod's formulation deliberately closely: the same forward branch
for the propagation cosine, the same conjugated index in the p admittance. That
is what makes this a *check* rather than two independent guesses that happen to
be near each other, and it is why the tolerance below is machine precision
rather than a few percent.

A disagreement here means the kernel is reporting coatings that Optiland would
not, which for a design tool whose whole pitch is "ideate here, finish there" is
the worst kind of quiet wrong.
"""

from __future__ import annotations

import json
import math

import pytest

optiland = pytest.importorskip("optiland")

from optiland.materials import IdealMaterial  # noqa: E402
from optiland.thin_film import Layer, ThinFilmStack  # noqa: E402

from optiland_canvas import _native  # noqa: E402

# Machine precision. Both sides are f64 and the algorithms are the same one, so
# anything above this is a real divergence and not rounding.
TOL = 1e-11

LAMBDAS = [0.40, 0.45, 0.50, 0.53, 0.55, 0.58, 0.62, 0.70, 0.90]
ANGLES = [0.0, 10.0, 30.0, 45.0, 60.0, 75.0]


def _canvas_material(n: float, k: float = 0.0, name: str = "L") -> dict:
    return {"name": name, "n": {"Constant": n}, "k": {"Constant": k}}


def _canvas_layer(n: float, thickness_nm: float, k: float = 0.0) -> dict:
    return {"material": _canvas_material(n, k), "thickness_nm": thickness_nm}


def _quarter_wave_nm(n: float, lambda_nm: float, aoi_deg: float, n0: float = 1.0) -> float:
    """A quarter wave of tilted optical thickness, the same rule the kernel uses."""
    sin_i = n0 * math.sin(math.radians(aoi_deg))
    cos_l = math.sqrt(max(1e-12, 1.0 - (sin_i / n) ** 2))
    return lambda_nm / (4.0 * n * cos_l)


def _compare(layers: list[dict], substrate_n: float, n_incident: float = 1.0) -> None:
    """Evaluate one stack both ways and assert every number agrees."""
    stack_json = json.dumps(
        {
            "layers": layers,
            # `Medium` backing: the bare stack on a semi-infinite substrate,
            # which is exactly what Optiland's ThinFilmStack models. The plate
            # sandwich the canvas adds on top has no Optiland counterpart and is
            # checked in Rust instead.
            "backing": {"Plate": {"substrate": _canvas_material(substrate_n, name="sub"),
                                  "exit": {"AntiReflective": {"residual": 0.0}}}},
            "incidence": "Film",
            "design": None,
        }
    )
    # A perfect AR on the far face and no absorption makes the plate's second
    # surface invisible, so the sandwich reduces to the single coherent stack
    # Optiland computes. That is the only way to compare like with like.
    got = json.loads(_native.thinfilm_rt(stack_json, LAMBDAS, ANGLES, n_incident))

    ref = ThinFilmStack(
        incident_material=IdealMaterial(n_incident),
        substrate_material=IdealMaterial(substrate_n),
        layers=[
            Layer(
                IdealMaterial(l["material"]["n"]["Constant"], k=l["material"]["k"]["Constant"]),
                thickness_um=l["thickness_nm"] / 1000.0,
            )
            for l in layers
        ],
    )

    worst = 0.0
    worst_at = None
    for i, nm in enumerate(LAMBDAS):
        for j, deg in enumerate(ANGLES):
            flat = i * len(ANGLES) + j
            th = [math.radians(deg)]
            s = ref.compute_rtRTA([nm], th, polarization="s")
            p = ref.compute_rtRTA([nm], th, polarization="p")
            for key, want in (
                ("rs", float(s["R"][0][0])),
                ("rp", float(p["R"][0][0])),
                ("ts", float(s["T"][0][0])),
                ("tp", float(p["T"][0][0])),
            ):
                d = abs(got[key][flat] - want)
                if d > worst:
                    worst, worst_at = d, (nm, deg, key, got[key][flat], want)
    assert worst < TOL, f"worst {worst:.3e} at {worst_at}"


def test_a_bare_interface_agrees():
    """No layers at all: both sides must reduce to plain Fresnel."""
    _compare([], substrate_n=1.5168)


def test_a_single_quarter_wave_agrees():
    n = 1.38
    _compare([_canvas_layer(n, _quarter_wave_nm(n, 550.0, 0.0))], substrate_n=1.52)


def test_a_quarter_wave_mirror_agrees():
    """(HL)^8 H, the stack a dichroic is built from."""
    hi, lo = 2.35, 1.46
    layers = []
    for _ in range(8):
        layers.append(_canvas_layer(hi, _quarter_wave_nm(hi, 550.0, 45.0)))
        layers.append(_canvas_layer(lo, _quarter_wave_nm(lo, 550.0, 45.0)))
    layers.append(_canvas_layer(hi, _quarter_wave_nm(hi, 550.0, 45.0)))
    _compare(layers, substrate_n=1.5168)


def test_a_symmetric_period_edge_filter_agrees():
    """0.5H (L H)^N L 0.5H, which is what the canvas actually synthesizes."""
    hi, lo = 2.35, 1.46
    qh = _quarter_wave_nm(hi, 500.0, 45.0)
    ql = _quarter_wave_nm(lo, 500.0, 45.0)
    layers = [_canvas_layer(hi, qh * 0.5)]
    for i in range(8):
        layers.append(_canvas_layer(lo, ql))
        if i < 7:
            layers.append(_canvas_layer(hi, qh))
    layers.append(_canvas_layer(hi, qh * 0.5))
    _compare(layers, substrate_n=1.5168)


def test_an_absorbing_metal_film_agrees():
    """The one case where the complex branches actually do work.

    A dielectric stack never exercises the complex propagation cosine or the
    conjugated index in the p admittance, because k is zero everywhere. Aluminium
    does, and it is where a sign convention copied slightly wrong would show up.
    """
    _compare([_canvas_layer(0.958, 12.0, k=6.69)], substrate_n=1.5168)


def test_an_absorbing_film_under_a_dielectric_agrees():
    """A metal under a quarter wave: absorption and interference together."""
    layers = [
        _canvas_layer(1.46, _quarter_wave_nm(1.46, 550.0, 0.0)),
        _canvas_layer(0.958, 30.0, k=6.69),
    ]
    _compare(layers, substrate_n=1.5168)


def test_in_a_denser_incident_medium():
    """Lit through glass rather than air, where the angles run the other way."""
    hi, lo = 2.35, 1.46
    layers = [
        _canvas_layer(hi, _quarter_wave_nm(hi, 550.0, 0.0, n0=1.52)),
        _canvas_layer(lo, _quarter_wave_nm(lo, 550.0, 0.0, n0=1.52)),
    ]
    _compare(layers, substrate_n=1.0, n_incident=1.52)
