"""The coating cases every oracle comparison runs, defined once.

``gen_coating_fixtures.py`` evaluates these with ``tmm`` and Optiland and
freezes the answers under ``data/coatings/``; ``test_coating_fixtures.py``
holds the canvas to the frozen answers on every pull request, with no oracle
installed; ``test_thinfilm_tmm.py`` runs ``tmm`` live.

A case is a ``Medium``-backed stack (one coherent stack between two
semi-infinite media), the grid it is evaluated on, and the two media. Plates
are not here: they add two faces incoherently, which is CF-05's comparison
against ``tmm.inc_tmm``.
"""

from __future__ import annotations

import math

LAMBDAS = [0.40, 0.45, 0.50, 0.53, 0.55, 0.58, 0.62, 0.70, 0.90]
ANGLES = [0.0, 10.0, 30.0, 45.0, 60.0, 75.0]


def material(n: float, k: float = 0.0, name: str = "L") -> dict:
    return {"name": name, "n": {"Constant": n}, "k": {"Constant": k}}


def layer(n: float, thickness_nm: float, k: float = 0.0) -> dict:
    return {"material": material(n, k), "thickness_nm": thickness_nm}


def quarter_wave_nm(n: float, lambda_nm: float, aoi_deg: float, n0: float = 1.0) -> float:
    """A quarter wave of tilted optical thickness, the kernel's own rule."""
    sin_i = n0 * math.sin(math.radians(aoi_deg))
    cos_l = math.sqrt(max(1e-12, 1.0 - (sin_i / n) ** 2))
    return lambda_nm / (4.0 * n * cos_l)


def stack(layers: list[dict], incidence: str = "Film") -> dict:
    return {"layers": layers, "backing": "Medium", "incidence": incidence, "design": None}


def _mirror() -> list[dict]:
    hi, lo = 2.35, 1.46
    out = []
    for _ in range(8):
        out.append(layer(hi, quarter_wave_nm(hi, 550.0, 45.0)))
        out.append(layer(lo, quarter_wave_nm(lo, 550.0, 45.0)))
    out.append(layer(hi, quarter_wave_nm(hi, 550.0, 45.0)))
    return out


def _edge_filter() -> list[dict]:
    """0.5H (L H)^7 L 0.5H, the symmetric period the canvas synthesizes."""
    hi, lo = 2.35, 1.46
    qh, ql = quarter_wave_nm(hi, 500.0, 45.0), quarter_wave_nm(lo, 500.0, 45.0)
    out = [layer(hi, qh * 0.5)]
    for i in range(8):
        out.append(layer(lo, ql))
        if i < 7:
            out.append(layer(hi, qh))
    out.append(layer(hi, qh * 0.5))
    return out


# Silica, a thin aluminium film, titania: asymmetric and lossy (C08).
_ASYMMETRIC = [layer(1.46, 94.0), layer(0.958, 8.0, k=6.69), layer(2.35, 61.0)]

# name -> (stack, n_incident, n_far, description). Each one is compared in
# power and in amplitude.
CASES: dict[str, tuple[dict, float, float, str]] = {
    "bare": (stack([]), 1.0, 1.5168, "C01: no layers, plain Fresnel"),
    "single_quarter_wave": (
        stack([layer(1.38, quarter_wave_nm(1.38, 550.0, 0.0))]),
        1.0,
        1.52,
        "C05: one lossless layer swept in wavelength and angle",
    ),
    "quarter_wave_mirror": (stack(_mirror()), 1.0, 1.5168, "C06: (HL)^8 H tilted for 45 degrees"),
    "edge_filter": (stack(_edge_filter()), 1.0, 1.5168, "C06: symmetric-period edge filter"),
    "aluminium_film": (stack([layer(0.958, 12.0, k=6.69)]), 1.0, 1.5168, "C07: thin absorbing film"),
    "aluminium_under_dielectric": (
        stack([layer(1.46, quarter_wave_nm(1.46, 550.0, 0.0)), layer(0.958, 30.0, k=6.69)]),
        1.0,
        1.5168,
        "C07: absorption and interference together",
    ),
    "dense_incident_medium": (
        stack(
            [
                layer(2.35, quarter_wave_nm(2.35, 550.0, 0.0, n0=1.52)),
                layer(1.46, quarter_wave_nm(1.46, 550.0, 0.0, n0=1.52)),
            ]
        ),
        1.52,
        1.0,
        "lit through glass into air; past 41 degrees the exit is evanescent",
    ),
    "asymmetric_absorbing_forward": (stack(_ASYMMETRIC), 1.0, 1.5168, "C08: asymmetric lossy stack, film side"),
    "asymmetric_absorbing_reversed": (
        stack(_ASYMMETRIC, incidence="Substrate"),
        1.5168,
        1.0,
        "C08: the same stack met from its substrate, walked last layer first",
    ),
    "thick_aluminium": (
        stack([layer(0.958, 5000.0, k=6.69)]),
        1.0,
        1.5168,
        "C12: 5 um of aluminium, opaque: the bulk metal, which used to overflow to NaN",
    ),
    "silica_on_opaque_aluminium": (
        {
            "layers": [layer(1.46, 94.0)],
            "backing": {"Opaque": {"substrate": material(0.958, 6.69, "Al")}},
            "incidence": "Film",
            "design": None,
        },
        1.0,
        1.0,
        "C13: a silica overcoat on a semi-infinite aluminium substrate",
    ),
    "total_internal_reflection": (
        stack([]),
        1.5,
        1.0,
        "glass to air through the critical angle; the phase is the Fresnel rhomb's",
    ),
}


def _catalogue_stack() -> dict:
    """C11: MgF2 over TiO2 over SiO2 from the measured catalogue, each a
    quarter wave at 550 nm of its own index there."""
    from optiland_canvas import coatings

    cat = coatings.catalog()
    layers = []
    for key in ("magnesium_fluoride", "titania", "silica"):
        m = cat[key]
        n = m["nominal"]["n"]
        layers.append({"material": m, "thickness_nm": 550.0 / (4.0 * n)})
    return stack(layers)


CASES["dispersive_catalogue_stack"] = (
    _catalogue_stack(),
    1.0,
    1.5168,
    "C11: MgF2, TiO2 and SiO2 from the catalogue, dispersive, 0.4 to 2.0 um",
)

C11_LAMBDAS = [round(0.4 + 1.6 * i / 19, 12) for i in range(20)]


def lambdas_for(name: str) -> list[float]:
    """The catalogue stack is swept across its materials' dispersion."""
    return C11_LAMBDAS if name == "dispersive_catalogue_stack" else LAMBDAS


def critical_deg(n_incident: float, n_far: float) -> float | None:
    """The critical angle in the incident medium, or ``None`` if there is none."""
    return math.degrees(math.asin(n_far / n_incident)) if n_far < n_incident else None


def angles_for(name: str) -> list[float]:
    """The total-internal-reflection case samples densely through 41.8 deg."""
    if name == "total_internal_reflection":
        return [0.0, 20.0, 40.0, 41.0, 42.0, 45.0, 50.0, 54.6231, 60.0, 70.0, 80.0, 89.0]
    return ANGLES
