"""Optiland Canvas ⇄ Optiland.

The seam between the browser sketchpad and the Optiland engine (spec §6). A
canvas ``.ocanvas`` scene is a non-sequential 2D pile of elements; an Optiland
``Optic`` is an ordered list of surfaces. The exact geometry translation lives in
Rust (``oc-core``'s ``SeqSystem``); this module only builds/reads the Optiland
object from the neutral *sequential prescription* the Rust core hands it.

Typical use::

    import optiland_canvas as oc

    optic = oc.to_optic("design.ocanvas")   # export: scene -> Optiland Optic
    optic.draw()

    scene_json = oc.from_optic(optic)        # import: Optiland Optic -> scene
    open("out.ocanvas", "w").write(scene_json)

The pure-format helpers (``scene_to_seq``, ``seq_to_scene``, ``validate``) work
without Optiland installed; ``to_optic`` / ``from_optic`` require it.

Every function that takes a scene reads version 1 and version 2 documents (the
browser saves version 2). A document that cannot be read raises one of three
errors, all ``ValueError`` subclasses under :class:`OcanvasError`:

- :class:`NotADocumentError`: not JSON, or not an ``.ocanvas`` document;
- :class:`UnsupportedVersionError`: a version this build does not read;
- :class:`InvalidSceneError`: a readable version whose scene does not hold up.

A scene that reads fine but has no sequential form raises
:class:`NotExportableError` from the export functions.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any

from . import _native

__all__ = [
    "to_optic",
    "from_optic",
    "optic_from_seq",
    "seq_from_optic",
    "scene_to_seq",
    "seq_to_scene",
    "scene3_sequence",
    "scene3_sequence_for_intent",
    "validate",
    "load",
    "save",
    "OCANVAS_VERSION",
    "OCANVAS_FORMAT",
    "READABLE_VERSIONS",
    "FidelityWarning",
    "OcanvasError",
    "NotADocumentError",
    "UnsupportedVersionError",
    "InvalidSceneError",
    "NotExportableError",
]

#: The version :func:`seq_to_scene` writes. The browser writes version 2.
OCANVAS_VERSION: int = _native.OCANVAS_VERSION
OCANVAS_FORMAT: str = _native.OCANVAS_FORMAT
#: Every document version this build reads.
READABLE_VERSIONS: tuple = tuple(_native.READABLE_VERSIONS)

OcanvasError = _native.OcanvasError
NotADocumentError = _native.NotADocumentError
UnsupportedVersionError = _native.UnsupportedVersionError
InvalidSceneError = _native.InvalidSceneError
NotExportableError = _native.NotExportableError


class FidelityWarning(UserWarning):
    """A canvas feature could not be represented exactly in the Optiland handoff."""


# --------------------------------------------------------------------------- #
# Native, Optiland-free helpers                                               #
# --------------------------------------------------------------------------- #
def scene_to_seq(
    scene: Any, *, source_id: int | None = None, detector_id: int | None = None
) -> dict:
    """Sequential prescription (a dict) extracted from an ``.ocanvas`` scene.

    ``scene`` may be a path to a ``.ocanvas`` file, a JSON string, or a scene
    ``dict``. The returned dict's ``notes`` list is the fidelity report, and its
    ``stop`` is the declared aperture stop's surface index, or ``None`` when
    nothing declares one.

    A version 2 scene exports the path from ``source_id`` to ``detector_id``,
    in the order its axial ray meets the surfaces. Leave both out for a scene
    with one declared path, or one source and one area detector. Raises
    :class:`NotExportableError` with every reason when the path has content a
    sequential system cannot hold.
    """
    return json.loads(
        _native.scene_to_seq(_as_ocanvas_json(scene), source_id, detector_id)
    )


def seq_to_scene(seq: dict | str) -> str:
    """``.ocanvas`` JSON rebuilt from a sequential-prescription dict/JSON."""
    seq_json = seq if isinstance(seq, str) else json.dumps(seq)
    return _native.seq_to_scene(seq_json)


def scene3_sequence(scene: Any, source_id: int, detector_id: int, revision: int = 0) -> dict:
    """The ``oc-seq3`` axial-ray sequence prescription of a 3D scene.

    Traces the axial ray of ``source_id`` through the scene (v1 documents
    migrate on load) and returns the ordered surface chain that reaches
    ``detector_id``, as a dict. Raises ``ValueError`` when the ray never
    reaches the detector (the message lists every branch's fate) or reaches it
    along more than one chain.
    """
    return json.loads(
        _native.scene3_sequence(_as_ocanvas_json(scene), source_id, detector_id, revision)
    )


def scene3_sequence_for_intent(scene: Any, intent_id: int, revision: int = 0) -> dict:
    """:func:`scene3_sequence` for a declared primary-path intent."""
    return json.loads(
        _native.scene3_sequence_for_intent(_as_ocanvas_json(scene), intent_id, revision)
    )


def validate(scene: Any) -> int:
    """Read and validate a scene of any readable version, and return that version.

    Raises :class:`NotADocumentError`, :class:`UnsupportedVersionError` or
    :class:`InvalidSceneError` when it cannot be read.
    """
    return _native.validate(_as_ocanvas_json(scene))


def load(path: str) -> str:
    """Read, validate, and return the ``.ocanvas`` JSON at ``path``, any version."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    validate(text)
    return text


def save(scene: Any, path: str) -> None:
    """Normalize a scene and write it to ``path``, in the version it was read in."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_native.normalize(_as_ocanvas_json(scene)))


# --------------------------------------------------------------------------- #
# Export: scene -> Optiland Optic                                             #
# --------------------------------------------------------------------------- #
def to_optic(
    scene: Any,
    *,
    strict: bool = False,
    stop: int | None = None,
    source_id: int | None = None,
    detector_id: int | None = None,
):
    """Build an Optiland ``Optic`` from an ``.ocanvas`` scene (spec §6 export).

    Exact for the rotationally-symmetric refractive subset. Anything that cannot
    be represented is reported: as a :class:`FidelityWarning` by default, or,
    with ``strict=True``, as a :class:`NotExportableError`.

    The aperture stop is the one the scene declares. ``stop`` names a surface
    index for a scene that declares none (a version 1 scene never can). With
    neither, the first surface is used and the warning says so; ``strict``
    refuses instead, because an inferred stop moves the pupils.
    ``source_id``/``detector_id`` pick the path, as for :func:`scene_to_seq`.
    """
    seq = scene_to_seq(scene, source_id=source_id, detector_id=detector_id)
    seq = _settle_stop(seq, stop, strict)
    _report(seq.get("notes", []), strict)
    return optic_from_seq(seq)


def _settle_stop(seq: dict, stop: int | None, strict: bool) -> dict:
    """The prescription with its stop decided: declared, named, or inferred aloud."""
    declared = seq.get("stop")
    n = len(seq["surfaces"])
    if stop is not None:
        if not 0 <= stop < n:
            raise ValueError(f"stop={stop} is not a surface index (0 to {n - 1})")
        if declared is not None and declared != stop:
            raise ValueError(
                f"the scene declares surface {declared} as the stop, and stop={stop} contradicts it"
            )
        return dict(seq, stop=stop)
    if declared is not None:
        return seq
    if strict:
        raise NotExportableError(
            "no aperture stop is declared: declare one on an aperture in the scene, "
            "or pass stop=<surface index>"
        )
    inferred = (
        "no aperture stop is declared, so the first surface is used as the stop (inferred); "
        "declare one on an aperture, or pass stop=<surface index>"
    )
    return dict(seq, stop=0, notes=[*seq.get("notes", []), inferred])


def optic_from_seq(seq: dict):
    """Build an Optiland ``Optic`` from a sequential-prescription dict."""
    from optiland.optic import Optic

    optic = Optic()
    surfaces = seq["surfaces"]
    n = len(surfaces)
    if seq.get("stop") is None:
        raise NotExportableError(
            "the prescription names no aperture stop; to_optic decides one, or set seq['stop']"
        )
    stop = int(seq["stop"])

    # Object surface. `object_distance = None` means an object at infinity, which
    # Optiland spells as a genuinely infinite thickness (as its own sample systems
    # do) — NOT a large finite number, which would place the object kilometres
    # away and wreck `optic.draw()`'s axis scaling.
    field_type = seq.get("field_type", "angle")
    obj_distance = seq.get("object_distance")
    obj_thickness = math.inf if obj_distance is None else float(obj_distance)
    optic.surfaces.add(index=0, thickness=obj_thickness, radius=math.inf, material="air")

    for i, s in enumerate(surfaces):
        radius = s.get("radius")
        radius = math.inf if radius is None else float(radius)
        coeffs = list(s.get("coefficients") or [])
        kwargs = dict(
            index=i + 1,
            thickness=float(s["thickness"]),
            radius=radius,
            conic=float(s.get("conic", 0.0)),
            is_stop=(i == stop),
            material=_material_for_optiland(s["material"], reflective=s.get("reflective", False)),
            aperture=2.0 * float(s["semi_aperture"]),
        )
        if s.get("reflective") or s["material"].get("kind") == "mirror":
            kwargs["material"] = "mirror"
        if coeffs:
            # Optiland even-asphere coefficients lead with the r^2 term (0 here).
            kwargs.update(surface_type="even_asphere", coefficients=[0.0, *coeffs])
        optic.surfaces.add(**kwargs)

    # Image surface.
    optic.surfaces.add(index=n + 1)

    # Aperture.
    ap = seq["aperture"]
    optic.set_aperture(_APERTURE_KIND[ap["kind"]], float(ap["value"]))

    # Fields.
    optic.fields.set_type("angle" if field_type == "angle" else "object_height")
    fields = seq.get("fields") or [0.0]
    for f in fields:
        optic.fields.add(y=float(f))

    # Wavelengths (first is primary).
    for j, w in enumerate(seq["wavelengths"]):
        optic.wavelengths.add(value=float(w), is_primary=(j == 0), unit="um")

    return optic


# --------------------------------------------------------------------------- #
# Import: Optiland Optic -> scene                                             #
# --------------------------------------------------------------------------- #
def from_optic(optic) -> str:
    """Return the ``.ocanvas`` JSON for an Optiland ``Optic`` (spec §6 import)."""
    return seq_to_scene(seq_from_optic(optic))


def seq_from_optic(optic) -> dict:
    """Read a sequential-prescription dict out of an Optiland ``Optic``."""
    surfaces = list(optic.surfaces.surfaces)
    n = len(surfaces)
    if n < 3:
        raise ValueError("Optic has no optical surfaces between object and image")

    zs = [_scalar(s.geometry.cs.z) for s in surfaces]
    optical = surfaces[1 : n - 1]  # drop object (0) and image (last)

    seq_surfaces = []
    stop = 0
    for i, s in enumerate(optical):
        gi = i + 1
        g = s.geometry
        radius = _scalar(getattr(g, "radius", math.inf))
        radius = None if (radius is None or math.isinf(radius)) else radius
        coeffs_raw = getattr(g, "coefficients", None)
        if coeffs_raw is not None:
            coeffs = [_scalar(c) for c in list(coeffs_raw)[1:]]  # drop leading r^2 term
            coeffs = _trim(coeffs)
        else:
            coeffs = []
        semi = _scalar(s.aperture.r_max) if getattr(s, "aperture", None) is not None else 0.0
        if getattr(s, "is_stop", False):
            stop = i
        seq_surfaces.append(
            {
                "radius": radius,
                "conic": _scalar(getattr(g, "k", 0.0)),
                "thickness": zs[gi + 1] - zs[gi],
                "semi_aperture": semi,
                "coefficients": coeffs,
                "material": _material_to_seq(getattr(s, "material_post", None)),
                "reflective": bool(getattr(s, "is_reflective", False)),
            }
        )

    # Object distance: an infinite object thickness (z = -inf) is an object at
    # infinity, which the prescription records as null.
    obj_gap = zs[1] - zs[0]
    object_distance = None if math.isinf(obj_gap) else obj_gap

    return {
        "surfaces": seq_surfaces,
        "stop": stop,
        "aperture": _aperture_to_seq(optic.aperture),
        "fields": _fields_of(optic),
        "field_type": _field_type_of(optic),
        "object_distance": object_distance,
        "wavelengths": _wavelengths_of(optic),
        "notes": [],
    }


# --------------------------------------------------------------------------- #
# Material mapping                                                            #
# --------------------------------------------------------------------------- #
def _material_for_optiland(m: dict, *, reflective: bool):
    kind = m["kind"]
    if kind == "air":
        return "air"
    if kind == "mirror":
        return "mirror"
    if kind == "ideal":
        from optiland.materials import IdealMaterial

        return IdealMaterial(float(m["n"]))
    if kind == "abbe":
        from optiland.materials import AbbeMaterial

        return AbbeMaterial(float(m["nd"]), float(m["vd"]), model="polynomial")
    if kind == "glass":
        from optiland.materials import Material

        return Material(m["name"])
    raise ValueError(f"unknown material kind: {kind!r}")


def _material_to_seq(mat) -> dict:
    if mat is None:
        return {"kind": "air"}
    tn = type(mat).__name__
    if tn == "IdealMaterial":
        n = _scalar(mat.index)
        return {"kind": "air"} if abs(n - 1.0) < 1e-12 else {"kind": "ideal", "n": n}
    if tn in ("AbbeMaterial", "AbbeMaterialE"):
        return {"kind": "abbe", "nd": _scalar(mat.index), "vd": _scalar(mat.abbe)}
    if tn == "Material":
        return {"kind": "glass", "name": mat.name}
    if tn == "Mirror":
        return {"kind": "mirror"}
    # Fall back to a constant index read at the d line.
    try:
        return {"kind": "ideal", "n": _scalar(mat.n(0.58756))}
    except Exception:  # pragma: no cover - defensive
        return {"kind": "air"}


_APERTURE_KIND = {"epd": "EPD", "image_fno": "imageFNO", "object_na": "objectNA"}
_APERTURE_KIND_INV = {"EPD": "epd", "imageFNO": "image_fno", "objectNA": "object_na"}


def _aperture_to_seq(aperture) -> dict:
    d = aperture.to_dict()
    ap_type = d.get("type") or d.get("ap_type") or "EPD"
    value = d.get("value")
    if value is None:  # some versions nest the value differently
        value = next((v for k, v in d.items() if isinstance(v, (int, float))), 0.0)
    return {"kind": _APERTURE_KIND_INV.get(ap_type, "epd"), "value": float(value)}


# --------------------------------------------------------------------------- #
# Field / wavelength read-back (tolerant of Optiland version differences)     #
# --------------------------------------------------------------------------- #
def _fields_of(optic) -> list:
    fg = optic.fields
    fields = getattr(fg, "fields", None)
    if fields is None:
        return [0.0]
    return [_scalar(f.y) for f in fields]


def _field_type_of(optic) -> str:
    fg = optic.fields
    ft = getattr(fg, "field_type", None)
    if ft is None:
        fd = getattr(fg, "field_definition", None)
        ft = getattr(fd, "field_type", None) if fd is not None else None
    return "object_height" if (ft and "height" in str(ft).lower()) else "angle"


def _wavelengths_of(optic) -> list:
    wg = optic.wavelengths
    wls = getattr(wg, "wavelengths", None)
    if wls is None:
        return [0.58756]
    return [_scalar(w.value) for w in wls]


# --------------------------------------------------------------------------- #
# Small utilities                                                            #
# --------------------------------------------------------------------------- #
def _scalar(x) -> float:
    """Coerce a backend array / numpy scalar / python number to ``float``."""
    if x is None:
        return math.inf
    try:
        import numpy as np

        return float(np.atleast_1d(np.asarray(x, dtype=float)).ravel()[0])
    except Exception:
        return float(x)


def _trim(coeffs: list) -> list:
    last = -1
    for i, c in enumerate(coeffs):
        if c != 0.0:
            last = i
    return coeffs[: last + 1]


def _as_ocanvas_json(scene: Any) -> str:
    if isinstance(scene, dict):
        return json.dumps(scene)
    if isinstance(scene, str):
        # A path to an existing file, otherwise treat the string as JSON.
        if os.path.exists(scene) and scene.strip()[:1] not in "{[":
            with open(scene, "r", encoding="utf-8") as fh:
                return fh.read()
        return scene
    raise TypeError(f"expected a path, JSON string, or dict; got {type(scene).__name__}")


def _report(notes: list, strict: bool) -> None:
    if not notes:
        return
    msg = "; ".join(notes)
    if strict:
        raise NotExportableError(f"scene cannot be represented exactly: {msg}")
    warnings.warn(msg, FidelityWarning, stacklevel=3)
