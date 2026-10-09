"""Exporting a version 2 scene to Optiland: the geometry, and the stop.

The export traces the scene's path for the surface order and reads the scene
for the geometry (``oc_trace::seq_export``). These tests hold what Optiland
builds from it to numbers computed here independently from the scene file:
a thick lens's focal length, and the entrance pupil an interior stop implies.
The glass index is Optiland's own, so what is under test is the geometry and
the stop, not the dispersion model.

The scenes are a file the browser saved (``data/browser-*.ocanvas``) and a
two-lens bench with a declared interior stop (``data/interior-stop.ocanvas``,
kept equal to the Rust suite's bench by ``oc-trace/tests/seq_export3.rs``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import optiland_canvas as oc

optiland = pytest.importorskip("optiland")

DATA = Path(__file__).parent / "data"
DEFAULT = DATA / "browser-default.ocanvas"
SPLITTER = DATA / "browser-splitter.ocanvas"
INTERIOR = DATA / "interior-stop.ocanvas"


def _doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _index(optic, surface: int, wavelength_um: float) -> float:
    """Optiland's index behind a surface, at a wavelength."""
    import numpy

    return float(numpy.ravel(optic.surfaces.surfaces[surface].material_post.n(wavelength_um))[0])


# --------------------------------------------------------------------------- #
# The browser's own file                                                      #
# --------------------------------------------------------------------------- #
def test_the_browser_default_scene_exports_its_singlet():
    seq = oc.scene_to_seq(str(DEFAULT))
    lens = _doc(DEFAULT)["bodies"][0]["geometry"]["Profile"]["profile"]["edges"]
    front, back = lens[0]["curve"]["Conic"], lens[2]["curve"]["Conic"]
    s1, s2 = seq["surfaces"]
    assert s1["radius"] == pytest.approx(1.0 / front["c"], abs=1e-12)
    assert s2["radius"] == pytest.approx(1.0 / back["c"], abs=1e-12)
    assert s1["thickness"] == pytest.approx(back["vertex"][0] - front["vertex"][0], abs=1e-12)
    # Back vertex to the detector, which the browser puts at x = 48.
    detector_x = _doc(DEFAULT)["detectors"][0]["common"]["transform"]["pos"][0]
    assert s2["thickness"] == pytest.approx(detector_x - back["vertex"][0], abs=1e-9)
    assert seq["stop"] is None


def test_the_singlet_s_focal_length_is_the_thick_lens_s():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", oc.FidelityWarning)
        optic = oc.to_optic(str(DEFAULT))
    seq = oc.scene_to_seq(str(DEFAULT))
    lam = seq["wavelengths"][0]
    n = _index(optic, 1, lam)
    r1, r2 = seq["surfaces"][0]["radius"], seq["surfaces"][1]["radius"]
    t = seq["surfaces"][0]["thickness"]
    power = (n - 1.0) * (1.0 / r1 - 1.0 / r2 + (n - 1.0) * t / (n * r1 * r2))
    assert float(optic.paraxial.f2()) == pytest.approx(1.0 / power, rel=1e-9)


def test_the_browser_default_scene_warns_about_what_it_approximates():
    with pytest.warns(oc.FidelityWarning) as caught:
        oc.to_optic(str(DEFAULT))
    said = " ".join(str(w.message) for w in caught)
    # Its beam is a 20 x 0.5 mm slit, and nothing in it declares a stop.
    assert "rectangle" in said and "inferred" in said
    with pytest.raises(oc.NotExportableError):
        oc.to_optic(str(DEFAULT), strict=True)


def test_a_splitter_on_the_path_is_refused_by_name():
    plate = _doc(SPLITTER)["surfaces"][0]["common"]["id"]
    with pytest.raises(oc.NotExportableError, match=f"#{plate}.*beam splitter"):
        oc.scene_to_seq(str(SPLITTER))


# --------------------------------------------------------------------------- #
# An interior stop                                                            #
# --------------------------------------------------------------------------- #
def test_the_declared_interior_stop_is_optiland_s_stop():
    seq = oc.scene_to_seq(str(INTERIOR))
    assert seq["stop"] == 3, "front aperture, two lens faces, then the stop"
    # Declared stop, round beam, bare glass: nothing approximated, so strict
    # export takes it.
    optic = oc.to_optic(str(INTERIOR), strict=True)
    assert optic.surfaces.stop_index == seq["stop"] + 1, "Optiland counts the object surface"


def test_the_entrance_pupil_is_the_image_of_the_interior_stop():
    """Where the stop sits decides where the entrance pupil is.

    The chief ray crosses the axis at the stop. Traced paraxially from the
    first surface to the stop, a ray (y, nu) arrives at height
    ``A y + B nu``; the chief ray has that zero, and in object space it
    crosses the axis at ``B / A`` behind the first surface. Optiland's EPL is
    measured from the same surface. A stop exported at surface zero, as it
    used to be, would put the pupil at 0.
    """
    optic = oc.to_optic(str(INTERIOR), strict=True)
    seq = oc.scene_to_seq(str(INTERIOR))
    lam = seq["wavelengths"][0]
    surfaces = seq["surfaces"]
    stop = seq["stop"]

    # Transfer and refraction in (y, n u), from surface 0 to the stop.
    m = [[1.0, 0.0], [0.0, 1.0]]

    def apply(k):
        nonlocal m
        m = [
            [k[0][0] * m[0][0] + k[0][1] * m[1][0], k[0][0] * m[0][1] + k[0][1] * m[1][1]],
            [k[1][0] * m[0][0] + k[1][1] * m[1][0], k[1][0] * m[0][1] + k[1][1] * m[1][1]],
        ]

    n_before = 1.0
    for i in range(stop):
        s = surfaces[i]
        n_after = _index(optic, i + 1, lam)
        if s["radius"] is not None:
            apply([[1.0, 0.0], [-(n_after - n_before) / s["radius"], 1.0]])
        apply([[1.0, s["thickness"] / n_after], [0.0, 1.0]])
        n_before = n_after
    a, b = m[0][0], m[0][1]
    expected = b / a
    assert expected > 0.0, "the pupil lies behind the first surface"
    assert float(optic.paraxial.EPL()) == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# Which path                                                                  #
# --------------------------------------------------------------------------- #
def test_a_scene_with_two_detectors_needs_the_path_named():
    doc = _doc(DEFAULT)
    extra = json.loads(json.dumps(doc["detectors"][0]))
    extra["common"]["id"] = 99
    # Off the path: the first detector ends the axial ray, so nothing behind
    # it on the axis could be reached anyway.
    extra["common"]["transform"]["pos"] = [60.0, 50.0, 0.0]
    doc["detectors"].append(extra)
    doc["next_id"] = max(doc.get("next_id", 0), 100)
    with pytest.raises(oc.NotExportableError, match="name the source and detector"):
        oc.scene_to_seq(doc)
    source = doc["sources"][0]["id"]
    first = doc["detectors"][0]["common"]["id"]
    seq = oc.scene_to_seq(doc, source_id=source, detector_id=first)
    assert seq["surfaces"][-1]["thickness"] == pytest.approx(48.0 - 2.0, abs=1e-9)
    with pytest.raises(oc.NotExportableError, match="never reached"):
        oc.scene_to_seq(doc, source_id=source, detector_id=99)


def test_a_version_1_scene_has_no_path_to_name():
    v1 = oc.seq_to_scene(oc.scene_to_seq(str(DEFAULT)) | {"stop": 0})
    with pytest.raises(ValueError, match="version 1"):
        oc.scene_to_seq(v1, source_id=1, detector_id=2)
