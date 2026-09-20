"""Round-trip corpus for the Optiland handoff (spec §6).

Two guarantees are locked in here:

1. A design survives ``prescription -> Optiland Optic -> prescription`` with
   parameter-level equality (the CI round-trip the spec asks for), and the same
   holds for the full ``.ocanvas -> Optic -> .ocanvas`` path.
2. Fidelity is reported, never faked: a scene with a non-sequential feature
   raises/warns instead of silently approximating.

The pure-format checks run without Optiland; the bridge checks skip if it is
absent.
"""

from __future__ import annotations

import json
import math

import pytest

import optiland_canvas as oc

optiland = pytest.importorskip("optiland")


# --------------------------------------------------------------------------- #
# Corpus                                                                      #
# --------------------------------------------------------------------------- #
def _surf(radius, thickness, material, semi=14.0, conic=0.0, coeffs=None, reflective=False):
    return {
        "radius": radius,
        "conic": conic,
        "thickness": thickness,
        "semi_aperture": semi,
        "coefficients": coeffs or [],
        "material": material,
        "reflective": reflective,
    }


def _abbe(nd, vd):
    return {"kind": "abbe", "nd": nd, "vd": vd}


def _glass(name):
    return {"kind": "glass", "name": name}


_AIR = {"kind": "air"}


def _system(surfaces, fields=(0.0,), epd=20.0, wavelengths=(0.58756,), stop=0):
    return {
        "surfaces": surfaces,
        "stop": stop,
        "aperture": {"kind": "epd", "value": epd},
        "fields": list(fields),
        "field_type": "angle",
        "wavelengths": list(wavelengths),
        "notes": [],
    }


CORPUS = {
    "biconvex": _system(
        [_surf(50.0, 6.0, _abbe(1.5, 60.0)), _surf(-50.0, 44.0, _AIR)],
        fields=(0.0, 5.0),
    ),
    "plano_convex": _system(
        [_surf(30.0, 5.0, _abbe(1.5168, 64.17)), _surf(None, 60.0, _AIR)],
    ),
    "meniscus": _system(
        [_surf(25.0, 4.0, _abbe(1.6, 50.0)), _surf(60.0, 80.0, _AIR)],
    ),
    "parabolic_front": _system(
        [_surf(50.0, 6.0, _abbe(1.5, 60.0), conic=-1.0), _surf(-50.0, 54.0, _AIR)],
    ),
    "even_asphere": _system(
        [
            _surf(50.0, 6.0, _abbe(1.5, 60.0), coeffs=[1e-6, -2e-9]),
            _surf(-50.0, 54.0, _AIR),
        ],
    ),
    "catalog_nbk7": _system(
        [_surf(50.0, 6.0, _glass("N-BK7"), semi=18.0), _surf(-50.0, 54.0, _AIR, semi=18.0)],
        wavelengths=(0.48613, 0.58756, 0.65627),
    ),
    "air_spaced_doublet": _system(
        [
            _surf(60.0, 5.0, _abbe(1.6, 50.0)),
            _surf(-60.0, 3.0, _AIR),
            _surf(40.0, 4.0, _abbe(1.5, 60.0)),
            _surf(-200.0, 48.0, _AIR),
        ],
    ),
    "cemented_doublet": _system(
        [
            _surf(60.0, 5.0, _abbe(1.62, 60.0)),
            _surf(-45.0, 3.0, _abbe(1.52, 55.0)),
            _surf(-150.0, 82.0, _AIR),
        ],
    ),
}


# --------------------------------------------------------------------------- #
# Comparison helpers                                                          #
# --------------------------------------------------------------------------- #
def _close(a, b, tol=1e-6):
    if a is None or b is None:
        return a == b
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol + tol * abs(b)


def _assert_systems_equal(name, expect, got):
    assert len(got["surfaces"]) == len(expect["surfaces"]), f"{name}: surface count"
    for i, (a, b) in enumerate(zip(expect["surfaces"], got["surfaces"])):
        where = f"{name}[{i}]"
        assert _close(a["radius"], b["radius"]), f"{where}: radius {a['radius']} vs {b['radius']}"
        assert _close(a["conic"], b["conic"]), f"{where}: conic"
        assert _close(a["thickness"], b["thickness"]), f"{where}: thickness"
        assert _close(a["semi_aperture"], b["semi_aperture"]), f"{where}: semi_aperture"
        assert a["material"] == b["material"], f"{where}: material {a['material']} vs {b['material']}"
        ca, cb = a["coefficients"], b["coefficients"]
        assert len(ca) == len(cb), f"{where}: n coeffs {ca} vs {cb}"
        for x, y in zip(ca, cb):
            assert _close(x, y, 1e-14), f"{where}: coeff {x} vs {y}"
    assert _close(expect["aperture"]["value"], got["aperture"]["value"]), f"{name}: EPD"
    assert len(expect["wavelengths"]) == len(got["wavelengths"]), f"{name}: n wavelengths"
    for x, y in zip(expect["wavelengths"], got["wavelengths"]):
        assert _close(x, y), f"{name}: wavelength"
    assert len(expect["fields"]) == len(got["fields"]), f"{name}: n fields"
    for x, y in zip(expect["fields"], got["fields"]):
        assert _close(x, y), f"{name}: field"


# --------------------------------------------------------------------------- #
# The round trips                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CORPUS))
def test_prescription_optic_roundtrip(name):
    """prescription -> Optiland Optic -> prescription is parameter-stable."""
    seq = CORPUS[name]
    optic = oc.optic_from_seq(seq)
    back = oc.seq_from_optic(optic)
    _assert_systems_equal(name, seq, back)
    assert back["stop"] == seq["stop"], f"{name}: the Optic's stop came back as {back['stop']}"


@pytest.mark.parametrize("name", list(CORPUS))
def test_full_ocanvas_optic_roundtrip(name):
    """.ocanvas -> Optic -> .ocanvas preserves the whole prescription, losslessly."""
    seq = CORPUS[name]
    ocanvas = oc.seq_to_scene(seq)          # prescription -> .ocanvas
    assert oc.validate(ocanvas) == oc.OCANVAS_VERSION

    # A version 1 scene has no apertures, so the stop cannot travel through
    # it; the corpus names it, and the export is told.
    optic = oc.to_optic(ocanvas, stop=seq["stop"])  # must not warn: exact subset
    back_ocanvas = oc.from_optic(optic)     # import
    final = oc.scene_to_seq(back_ocanvas)   # read the prescription back out

    assert final["notes"] == [], f"{name}: unexpected fidelity notes {final['notes']}"
    _assert_systems_equal(name, seq, final)


@pytest.mark.parametrize("name", list(CORPUS))
def test_exported_optic_is_traceable(name):
    """The built Optic is a real, traceable system (finite, positive-ish EFL)."""
    optic = oc.optic_from_seq(CORPUS[name])
    efl = float(optic.paraxial.f2())
    assert math.isfinite(efl) and abs(efl) > 1.0, f"{name}: implausible EFL {efl}"


def test_to_optic_does_not_warn_on_exact_subset(recwarn):
    oc.to_optic(oc.seq_to_scene(CORPUS["cemented_doublet"]), stop=0)
    assert not [w for w in recwarn if issubclass(w.category, oc.FidelityWarning)]


def test_a_stop_nothing_declared_is_inferred_out_loud_or_refused():
    """A version 1 scene cannot declare a stop, and the export says so.

    It used to write surface zero without a word. An inferred stop moves the
    pupils and changes every off-axis ray, so default export names the
    inference in its warning and strict export refuses it.
    """
    scene = oc.seq_to_scene(CORPUS["cemented_doublet"])
    assert oc.scene_to_seq(scene)["stop"] is None
    with pytest.warns(oc.FidelityWarning, match="inferred"):
        optic = oc.to_optic(scene)
    assert optic.surfaces.stop_index == 1
    with pytest.raises(oc.NotExportableError, match="no aperture stop is declared"):
        oc.to_optic(scene, strict=True)
    # Naming it satisfies strict, and a name outside the system does not.
    assert oc.to_optic(scene, strict=True, stop=1).surfaces.stop_index == 2
    with pytest.raises(ValueError, match="not a surface index"):
        oc.to_optic(scene, stop=99)
    # The builder will not guess either.
    with pytest.raises(oc.NotExportableError):
        oc.optic_from_seq(oc.scene_to_seq(scene))


@pytest.mark.parametrize("name", list(CORPUS))
def test_object_at_infinity_is_truly_infinite(name):
    """An angle-field system must put the object at a real infinity.

    A large-but-finite object distance is numerically fine for the paraxial
    maths but silently ruins `optic.draw()`, whose axes then span the object
    distance and squash the actual lenses to a speck. Optiland's own samples use
    an infinite object thickness; so must we.
    """
    optic = oc.optic_from_seq(CORPUS[name])
    z_obj = float(_np().atleast_1d(optic.surfaces.surfaces[0].geometry.cs.z).ravel()[0])
    assert math.isinf(z_obj), f"{name}: object at finite z={z_obj}"
    assert oc.seq_from_optic(optic)["object_distance"] is None


def test_finite_object_distance_round_trips():
    """A finite conjugate survives prescription -> Optic -> prescription."""
    seq = dict(CORPUS["biconvex"], field_type="object_height", fields=[0.0], object_distance=150.0)
    back = oc.seq_from_optic(oc.optic_from_seq(seq))
    assert _close(back["object_distance"], 150.0)


def _np():
    import numpy

    return numpy


# --------------------------------------------------------------------------- #
# Fidelity is reported, not faked                                             #
# --------------------------------------------------------------------------- #
def _spline_scene():
    """A valid .ocanvas with an extra freeform (spline) element."""
    scene = json.loads(oc.seq_to_scene(CORPUS["biconvex"]))
    scene["elements"].append(
        {
            "id": 1,
            "boundary": [
                {
                    "Spline": {
                        "ctrl": [[70.0, -6.0], [71.0, 0.0], [70.0, 6.0]],
                        "degree": 2,
                    }
                }
            ],
            "material": {"AbbePoint": {"nd": 1.5, "vd": 60.0}},
            "surface": json.loads(oc.seq_to_scene(CORPUS["biconvex"]))["elements"][0]["surface"],
            "transform": {"rot": 0.0, "pos": [0.0, 0.0]},
            "priority": 0,
            "locked": False,
            "layer": 0,
        }
    )
    return json.dumps(scene)


def test_fidelity_note_reports_spline():
    seq = oc.scene_to_seq(_spline_scene())
    assert any("spline" in n.lower() for n in seq["notes"]), seq["notes"]


def test_to_optic_warns_then_strict_raises_on_lossy_scene():
    scene = _spline_scene()
    with pytest.warns(oc.FidelityWarning):
        oc.to_optic(scene)
    with pytest.raises(ValueError):
        oc.to_optic(scene, strict=True)


# --------------------------------------------------------------------------- #
# Format freeze                                                               #
# --------------------------------------------------------------------------- #
def test_validate_and_version():
    ocanvas = oc.seq_to_scene(CORPUS["biconvex"])
    assert oc.validate(ocanvas) == 1
    assert oc.OCANVAS_FORMAT == "ocanvas"


def test_future_version_is_rejected():
    scene = json.loads(oc.seq_to_scene(CORPUS["biconvex"]))
    scene["version"] = oc.OCANVAS_VERSION + 1
    with pytest.raises(ValueError):
        oc.validate(json.dumps(scene))
