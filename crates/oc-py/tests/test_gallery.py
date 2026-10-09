"""Optiland's sample designs as canvas scenes (``optiland_canvas.gallery``).

A design goes into the canvas through :func:`from_optic` (``version=2``) and
comes back out through :func:`to_optic`. These tests hold the round trip to
the design it started from, for every sample Optiland ships:

- every surface that bends light comes back at the same place with the same
  shape, and the stop is the same surface;
- the focal length is the design's, and so is the entrance pupil, except
  where the design aims its rays through the real stop and the scene's beams
  overfill the pupil on purpose (the scene says by how much);
- a design the canvas cannot build is refused, and says why.

The rays themselves are ``oc-trace/tests/gallery3.rs``'s: it traces the
committed gallery scenes with the canvas's reference tracer and holds them to
the rays Optiland traced, recorded in ``web/public/gallery/checks.json``. The
last test here checks those committed files are what the generator writes,
when the installed Optiland is the one they were written with.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest

import optiland_canvas as oc

optiland = pytest.importorskip("optiland")
gallery = pytest.importorskip("optiland_canvas.gallery")

GALLERY = Path(__file__).resolve().parents[3] / "web" / "public" / "gallery"

#: The samples the canvas cannot build yet, and the word that says why.
REFUSED = {
    "Microscope20x": "virtual image",
    "UVReflectingMicroscope": "mirror",
    "HubbleTelescope": "mirror",
}

SAMPLES = [(m, cls) for m, cls in gallery.samples()]


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _scalar(x) -> float:
    import numpy

    return float(numpy.ravel(numpy.asarray(x, dtype=float))[0])


def _stations(seq: dict) -> list:
    """Each surface with its axial position, the first at 0."""
    z = 0.0
    out = []
    for i, s in enumerate(seq["surfaces"]):
        out.append((z, i, s))
        z += s["thickness"]
    return out


def _bends_light(seq: dict, i: int) -> bool:
    before = {"kind": "air"} if i == 0 else seq["surfaces"][i - 1]["material"]
    after = seq["surfaces"][i]["material"]

    def key(m):
        return ("air",) if m["kind"] == "air" else (m["kind"], m.get("name"), m.get("n"), m.get("nd"))

    return key(before) != key(after)


def _close(a, b, rel=1e-9, abs_=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= abs_ + rel * abs(b)


@pytest.mark.parametrize("module,cls", SAMPLES, ids=[c.__name__ for _, c in SAMPLES])
def test_a_sample_comes_back_as_itself(module, cls):
    design = _quiet(cls)
    if cls.__name__ in REFUSED:
        with pytest.raises(oc.NotImportableError, match=REFUSED[cls.__name__]):
            _quiet(oc.from_optic, design, version=2)
        return

    seq = oc.seq_from_optic(design)
    scene, notes = oc.seq_to_scene3(seq)
    back = oc.scene_to_seq(scene)
    optic = _quiet(oc.to_optic, scene)

    # Every surface that bends light is there, where it was, with its shape.
    # (The first surface of both sits at z = 0: the scene keeps the design's
    # coordinates, and the export measures from the first surface it meets.)
    theirs = _stations(back)
    refitted = {n.split(":")[0] for n in notes if "refitted" in n}
    for z, i, s in _stations(seq):
        if not _bends_light(seq, i):
            continue
        # A stop on a face's vertex exports as a plane at the same place, so
        # the face is the curved one there, when the design's face is curved.
        here = [t for zt, _, t in theirs if abs(zt - z) < 1e-9]
        assert here, f"S{i + 1} at z={z} is missing"
        got = next((t for t in here if (t["radius"] is None) == (s["radius"] is None)), here[0])
        if f"S{i + 1}" not in refitted:
            assert _close(got["radius"], s["radius"]), f"S{i + 1} radius"
            assert _close(got["conic"], s["conic"]), f"S{i + 1} conic"
            assert got["coefficients"] == pytest.approx(s["coefficients"], rel=1e-12, abs=0)

    # The stop is the same surface: at the same place.
    if seq["stop"] is not None:
        z_stop = _stations(seq)[seq["stop"]][0]
        z_back = _stations(back)[back["stop"]][0]
        assert abs(z_back - z_stop) < 1e-9, "the stop moved"

    # The focal length is the design's. Not for a refitted face: Optiland's
    # paraxial trace reads a face's base radius and leaves out its r^2 term,
    # which its real rays do feel, so the design's own paraxial focal length
    # is not its surface's. The real rays are gallery3.rs's to compare.
    if not refitted:
        efl = _scalar(design.paraxial.f2())
        assert _scalar(optic.paraxial.f2()) == pytest.approx(efl, rel=1e-9)

    # And so is the entrance pupil, where the scene's beam is the design's.
    if seq.get("aimed_beams"):
        assert any("wider than the design's" in n for n in notes)
    elif seq["object_distance"] is None:
        assert _scalar(optic.paraxial.EPD()) == pytest.approx(
            _scalar(design.paraxial.EPD()), rel=1e-9
        )
    else:
        assert back["aperture"]["kind"] == "object_na"
        assert back["aperture"]["value"] == pytest.approx(seq["aperture"]["value"], rel=1e-12)


def test_the_reader_takes_what_optiland_0_6_says():
    from optiland.samples import CookeTriplet, HubbleTelescope, UVProjectionLens

    # Object heights are object heights (Optiland 0.6 names the kind by class).
    assert oc.seq_from_optic(_quiet(UVProjectionLens))["field_type"] == "object_height"
    # Mirrors are mirrors (the interaction model says so, not the material).
    hubble = _quiet(oc.seq_from_optic, _quiet(HubbleTelescope))
    assert [s["reflective"] for s in hubble["surfaces"]] == [False, True, True]
    # A glass carries its catalog and its published formula.
    cooke = oc.seq_from_optic(_quiet(CookeTriplet))
    f2 = cooke["surfaces"][2]["material"]
    assert (f2["name"], f2["reference"], f2["formula"]["kind"]) == ("F2", "schott", "sellmeier2")
    # A glass published as a polynomial also gets the fitted Sellmeier curve
    # the preview draws with, close to it over the band it covers.
    sk16 = cooke["surfaces"][0]["material"]
    assert sk16["formula"]["kind"] == "polynomial" and "sellmeier" in sk16
    glass = _quiet(CookeTriplet).surfaces.surfaces[1].material_post
    assert oc.sellmeier_fit_error(glass, sk16["sellmeier"], cooke["wavelengths"]) < 5e-5


def test_the_primary_wavelength_comes_first():
    from optiland.samples import CookeTriplet

    design = _quiet(CookeTriplet)  # 0.48, 0.55 (primary), 0.65
    assert oc.seq_from_optic(design)["wavelengths"] == [0.55, 0.48, 0.65]


def test_an_r2_aspheric_term_is_refitted_out_loud():
    from optiland.samples import AsphericSinglet

    design = _quiet(AsphericSinglet)
    seq = oc.seq_from_optic(design)
    (note,) = seq["notes"]
    assert "r^2" in note and "refitted" in note
    # The vertex curvature takes the r^2 term: 1/R + 2 a2.
    g = design.surfaces.surfaces[1].geometry
    a2 = _scalar(list(g.coefficients)[0])
    assert 1.0 / seq["surfaces"][0]["radius"] == pytest.approx(1.0 / _scalar(g.radius) + 2 * a2)
    # Strict import refuses what is not exact.
    with pytest.raises(oc.NotImportableError, match="refitted"):
        oc.from_optic(design, version=2, strict=True)


def test_aimed_designs_get_a_beam_per_field_and_an_exact_stop():
    from optiland.samples.objectives import WideAngle170FOV

    design = _quiet(WideAngle170FOV)
    seq = oc.seq_from_optic(design)
    assert len(seq["aimed_beams"]) == len(seq["fields"])
    # The stop's opening is where the aimed rays reach, not a margin past it.
    stop = seq["surfaces"][seq["stop"]]["semi_aperture"]
    reach = oc._ray_reach(design)[seq["stop"] + 1]
    assert stop == pytest.approx(reach, rel=1e-8)
    # At 85° the real pupil is a sliver of the paraxial one, off its centre.
    wide = seq["aimed_beams"][-1]
    assert wide["half_width_mm"] < 0.5 * wide["half_height_mm"]


def test_the_committed_gallery_is_what_the_generator_writes(tmp_path):
    index = json.loads((GALLERY / "index.json").read_text(encoding="utf-8"))
    if index["optiland"] != optiland.__version__:
        pytest.skip(
            f"the gallery was written with Optiland {index['optiland']}, and this is "
            f"{optiland.__version__}; regenerate it with python -m optiland_canvas.gallery"
        )
    _quiet(gallery.build, str(tmp_path))
    for name in ["index.json", "checks.json"] + [
        d["file"] for d in index["designs"] if d["status"] == "ported"
    ]:
        fresh = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        kept = json.loads((GALLERY / name).read_text(encoding="utf-8"))
        _assert_same(fresh, kept, name)


# A glass published in a form the f32 preview cannot evaluate carries a
# three-term Sellmeier curve fitted to it with scipy's least_squares, which
# lands in a slightly different place on another platform or another scipy.
# The coefficients are therefore not compared: the curve they draw is, to
# CURVE_TOL in index, and the error the fit reports is held to the bound the
# gallery claims rather than to the digit (below 1e-8 it is noise, and its
# third digit moves between platforms). Nothing here is a number the app
# reports: every index comes from the glass's own published formula.
CURVE_TOL = 1e-6
FIT_BOUND = 1e-4


def _assert_same(a, b, where, tol=1e-9):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), where
        for k in a:
            if k == "sell" and _is_fitted(a):
                _assert_same_curve(a[k], b[k], f"{where}.{k}")
            elif k == "max_index_error":
                assert a[k] <= FIT_BOUND and b[k] <= FIT_BOUND, f"{where}.{k}: {a[k]} vs {b[k]}"
            else:
                _assert_same(a[k], b[k], f"{where}.{k}", tol)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{where}[{i}]", tol)
    elif isinstance(a, float) and isinstance(b, (int, float)):
        assert math.isclose(a, b, rel_tol=tol, abs_tol=1e-12), f"{where}: {a} vs {b}"
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


def _assert_same_curve(a: dict, b: dict, where):
    """Two fitted Sellmeier curves draw the same index over the canvas's
    band."""
    import numpy as np

    lam = np.linspace(*oc._FIT_BAND_UM, 400)
    worst = float(np.max(np.abs(oc._sellmeier_n(a, lam) - oc._sellmeier_n(b, lam))))
    assert worst <= CURVE_TOL, f"{where}: the fitted curves differ by {worst:.2g} in index"


def _is_fitted(glass: dict) -> bool:
    """Is this glass's ``sell`` a fit, rather than the curve it is published
    as? A glass with no formula, or with one the preview evaluates itself,
    carries the published curve and is compared to the bit."""
    formula = glass.get("formula")
    if not isinstance(formula, dict) or "coefficients" not in formula:
        # No formula, or a constant index, whose curve is not fitted either.
        return False
    return not oc._is_three_term(formula["kind"], formula["coefficients"])
