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

Every function that takes a scene reads version 1, 2 and 3 documents (the
browser saves version 3, which adds the film catalogue and the fidelity
profile to version 2). A document that cannot be read raises one of three
errors, all ``ValueError`` subclasses under :class:`OcanvasError`:

- :class:`NotADocumentError`: not JSON, or not an ``.ocanvas`` document;
- :class:`UnsupportedVersionError`: a version this build does not read;
- :class:`InvalidSceneError`: a readable version whose scene does not hold up.

A scene that reads fine but has no sequential form raises
:class:`NotExportableError` from the export functions, and a design with no
scene form (a mirror, a virtual image) raises :class:`NotImportableError` from
the import functions.
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
    "NotImportableError",
    "seq_to_scene3",
]

#: The version :func:`seq_to_scene` writes. The browser writes version 3.
OCANVAS_VERSION: int = _native.OCANVAS_VERSION
OCANVAS_FORMAT: str = _native.OCANVAS_FORMAT
#: Every document version this build reads.
READABLE_VERSIONS: tuple = tuple(_native.READABLE_VERSIONS)

OcanvasError = _native.OcanvasError
NotADocumentError = _native.NotADocumentError
UnsupportedVersionError = _native.UnsupportedVersionError
InvalidSceneError = _native.InvalidSceneError
NotExportableError = _native.NotExportableError
NotImportableError = _native.NotImportableError


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

    A version 2 or 3 scene exports the path from ``source_id`` to ``detector_id``,
    in the order its axial ray meets the surfaces. Leave both out for a scene
    with one declared path, or one source and one area detector. Raises
    :class:`NotExportableError` with every reason when the path has content a
    sequential system cannot hold.
    """
    return json.loads(
        _native.scene_to_seq(_as_ocanvas_json(scene), source_id, detector_id)
    )


def seq_to_scene(seq: dict | str) -> str:
    """Version 1 ``.ocanvas`` JSON rebuilt from a sequential-prescription
    dict/JSON. :func:`seq_to_scene3` builds the version 3 scene."""
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

    # Image surface, in the medium the last surface leaves the light in: an
    # image inside a medium (an eye's retina in its vitreous) is not behind a
    # refraction into air that the design does not have.
    last = surfaces[-1]["material"] if surfaces else {"kind": "air"}
    if last.get("kind") in ("air", "mirror") or surfaces[-1].get("reflective"):
        optic.surfaces.add(index=n + 1)
    else:
        optic.surfaces.add(index=n + 1, material=_material_for_optiland(last, reflective=False))

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
#: How far past its rays' own reach a surface's clear aperture extends when
#: Optiland gives the surface none, so the edge rays do not graze the rim they
#: define.
CLEAR_APERTURE_MARGIN = 0.01

#: How far an aimed field's beam reaches past its real pupil, so the pupil's
#: rim between the rays that trace it lies inside the beam.
AIMED_BEAM_MARGIN = 0.02


def from_optic(
    optic,
    *,
    version: int = 1,
    strict: bool = False,
    rays_per_field: int = 64,
    pixels: int = 64,
) -> str:
    """Return the ``.ocanvas`` JSON for an Optiland ``Optic`` (spec §6 import).

    ``version=2`` builds the canvas's 3D scene: each lens element a revolved
    body, the aperture stop an aperture the scene declares, each field a
    source filling the entrance pupil the way Optiland launches its rays, and
    a detector on the image plane. ``version=1`` (the default, for existing
    callers) writes the older 2D document, which has no apertures and so
    cannot hold the stop.

    What the scene holds only approximately is reported as a
    :class:`FidelityWarning`, or raised as a :class:`NotImportableError` with
    ``strict=True``. A design the scene cannot hold at all (a mirror, a
    virtual image) raises :class:`NotImportableError`.
    """
    seq = seq_from_optic(optic)
    if version == 1:
        _report(seq.get("notes", []), strict, error=NotImportableError)
        return seq_to_scene(seq)
    if version != 2:
        raise ValueError(f"version must be 1 or 2, not {version!r}")
    scene, notes = seq_to_scene3(seq, rays_per_field=rays_per_field, pixels=pixels)
    _report(notes, strict, error=NotImportableError)
    return scene


def seq_to_scene3(seq: dict | str, *, rays_per_field: int = 64, pixels: int = 64):
    """A version 3 ``.ocanvas`` scene built from a prescription, and its notes.

    Returns ``(scene_json, notes)``: the notes are the prescription's own and
    the builder's, what the scene holds only approximately. Raises
    :class:`NotImportableError` for a prescription it cannot build.
    """
    seq_dict = json.loads(seq) if isinstance(seq, str) else seq
    scene, built = _native.seq_to_scene3(json.dumps(seq_dict), rays_per_field, pixels)
    return scene, [*seq_dict.get("notes", []), *built]


def seq_from_optic(optic) -> dict:
    """Read a sequential-prescription dict out of an Optiland ``Optic``.

    Each surface's clear aperture is its physical aperture when it has one, and
    otherwise the widest real ray over every field, wavelength and pupil edge,
    :data:`CLEAR_APERTURE_MARGIN` wider: what Optiland draws the lens to. A
    catalog glass carries its published dispersion formula and the catalog it
    is from, so the canvas evaluates the same indices Optiland does. What the
    prescription cannot hold goes in ``notes``.
    """
    surfaces = list(optic.surfaces.surfaces)
    n = len(surfaces)
    if n < 3:
        raise ValueError("Optic has no optical surfaces between object and image")

    notes: list = []
    zs = [_scalar(s.geometry.cs.z) for s in surfaces]
    wavelengths = _wavelengths_of(optic)
    reach = _ray_reach(optic)
    optical = surfaces[1 : n - 1]  # drop object (0) and image (last)
    stop_index = getattr(optic.surfaces, "stop_index", None)
    stop = (
        stop_index - 1
        if isinstance(stop_index, int) and 1 <= stop_index <= len(optical)
        else None
    )
    aimed = _is_aimed(optic)

    seq_surfaces = []
    for i, s in enumerate(optical):
        gi = i + 1
        g = s.geometry
        kind = type(g).__name__
        if kind not in ("Plane", "StandardGeometry", "EvenAsphere"):
            notes.append(f"S{gi}: a {kind} surface is read as its base conic")
        _note_placement(s, f"S{gi}", notes)
        radius = _scalar(getattr(g, "radius", math.inf))
        radius = None if (radius is None or math.isinf(radius)) else radius
        conic = _scalar(getattr(g, "k", 0.0))
        semi = _clear_semi(s, reach[gi])
        if aimed and i == stop:
            # Aimed rays fill the stop exactly, and the stop is what trims the
            # scene's beams to them: its opening is the rays' reach, with only
            # the room rounding needs.
            semi = _clear_semi(s, reach[gi] / (1.0 + CLEAR_APERTURE_MARGIN) * (1.0 + 1e-9))
        coeffs = []
        raw = getattr(g, "coefficients", None)
        if raw is not None and kind == "EvenAsphere":
            raw = [_scalar(c) for c in list(raw)]
            if raw and raw[0] != 0.0:
                # Optiland's first term multiplies r^2; a canvas face starts at
                # r^4. The same shape is refitted: the vertex curvature takes
                # the r^2 term, and A4..A18 the rest.
                radius, coeffs, err = _refit_r2(radius, conic, raw, semi)
                notes.append(
                    f"S{gi}: its r^2 aspheric term has no canvas form, so the face is refitted as "
                    f"a conic with A4 to A18, within {err * 1e6:.3g} nm of the design's sag over "
                    f"its clear aperture"
                )
            else:
                coeffs = _trim(raw[1:])
        seq_surfaces.append(
            {
                "radius": radius,
                "conic": conic,
                "thickness": zs[gi + 1] - zs[gi],
                "semi_aperture": semi,
                "coefficients": coeffs,
                "material": _material_to_seq(
                    getattr(s, "material_post", None), wavelengths, f"S{gi}", notes
                ),
                "reflective": _is_reflective(s),
            }
        )

    image = surfaces[-1]
    image_radius = _scalar(getattr(image.geometry, "radius", math.inf))
    if not math.isinf(image_radius):
        notes.append(
            f"the image surface is curved (R = {image_radius:g} mm); the image plane is flat at "
            f"its vertex"
        )
    _note_placement(image, "the image surface", notes)
    obj_medium = getattr(surfaces[0], "material_post", None)
    if obj_medium is not None and _material_to_seq(obj_medium, wavelengths, "", []) != {
        "kind": "air"
    }:
        notes.append("object space is not air, and a prescription's object space is")

    # Object distance: an infinite object thickness (z = -inf) is an object at
    # infinity, which the prescription records as null.
    obj_gap = zs[1] - zs[0]
    object_distance = None if math.isinf(obj_gap) else obj_gap

    seq = {
        "surfaces": seq_surfaces,
        "stop": stop,
        "aperture": _aperture_to_seq(optic, notes),
        "fields": _fields_of(optic, notes),
        "field_type": _field_type_of(optic),
        "object_distance": object_distance,
        "object_space_telecentric": bool(getattr(optic, "obj_space_telecentric", False)),
        "image_semi_aperture": reach[n - 1] if reach[n - 1] > 0.0 else None,
        "wavelengths": wavelengths,
        "notes": notes,
    }
    if aimed and object_distance is None:
        seq["aimed_beams"] = _aimed_beams(optic)
    return seq


def _is_aimed(optic) -> bool:
    """Does the design aim its rays through the real stop (Optiland's
    ``set_aiming``) rather than launch them from the paraxial pupil?"""
    config = getattr(getattr(optic, "ray_tracer", None), "ray_aiming_config", None) or {}
    return config.get("mode") not in (None, "paraxial")


def _aimed_beams(optic) -> list:
    """For each field, the rectangle square to its beam that covers its real
    entrance pupil: the rim rays Optiland aims through the stop's edge, at
    every wavelength, crossing a plane square to the beam, boxed and widened by
    :data:`AIMED_BEAM_MARGIN`. In canvas coordinates: Optiland's ``z`` is the
    canvas's ``x``, its ``y`` the canvas's ``y`` and its ``x`` the canvas's
    ``-z``."""
    import numpy as np

    rim = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    px = np.concatenate([[0.0], np.cos(rim)])
    py = np.concatenate([[0.0], np.sin(rim)])
    beams = []
    for hx, hy in optic.fields.get_field_coords():
        points = []
        direction = None
        for w in _wavelengths_of(optic):
            optic.trace_generic(Hx=float(hx), Hy=float(hy), Px=px, Py=py, wavelength=w)
            sg = optic.surfaces
            x, y, z, L, M, N = (
                np.asarray(_to_numpy(getattr(sg, k))[0], dtype=float)
                for k in ("x", "y", "z", "L", "M", "N")
            )
            # Canvas coordinates, each ray taken where it crosses the first
            # vertex plane, x = 0.
            t = -z / N
            points.append(np.stack([np.zeros_like(t), y + t * M, -(x + t * L)], axis=1))
            if direction is None:
                direction = np.array([N[0], M[0], -L[0]])
        pts = np.concatenate(points)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        d = direction / np.linalg.norm(direction)
        # Axes square to the beam: meridional, and out of the meridian.
        across = np.array([-d[1], d[0], 0.0])
        across /= np.linalg.norm(across)
        out = np.array([0.0, 0.0, 1.0])
        u = pts @ across
        v = pts @ out
        cu, cv = 0.5 * (u.max() + u.min()), 0.5 * (v.max() + v.min())
        hu, hv = 0.5 * (u.max() - u.min()), 0.5 * (v.max() - v.min())
        # The box's centre line, where it crosses the first vertex plane.
        centre = cu * across + cv * out
        centre = centre - d * (centre[0] / d[0])
        beams.append(
            {
                "centre_mm": [float(centre[1]), float(centre[2])],
                "half_width_mm": float(hu * (1.0 + AIMED_BEAM_MARGIN)),
                "half_height_mm": float(hv * (1.0 + AIMED_BEAM_MARGIN)),
            }
        )
    return beams


def _ray_reach(optic) -> list:
    """The widest real ray on each surface, over every field and wavelength,
    for a pupil sampled on its rim, its centre and its meridian."""
    import numpy as np

    rim = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    line = np.linspace(-1.0, 1.0, 21)
    px = np.concatenate([np.cos(rim), np.zeros_like(line)])
    py = np.concatenate([np.sin(rim), line])
    surfaces = optic.surfaces.surfaces
    reach = np.zeros(len(surfaces))
    z_image = _scalar(surfaces[-1].geometry.cs.z)
    for hx, hy in optic.fields.get_field_coords():
        for w in _wavelengths_of(optic):
            optic.trace_generic(Hx=float(hx), Hy=float(hy), Px=px, Py=py, wavelength=w)
            sg = optic.surfaces
            x = np.asarray(_to_numpy(sg.x), dtype=float)
            y = np.asarray(_to_numpy(sg.y), dtype=float)
            # The image is read on the plane square to the axis at the image
            # surface's vertex, where the scene's detector is, so a curved
            # image surface is measured there too: each ray carried on from
            # the last surface to that plane.
            last = {k: np.asarray(_to_numpy(getattr(sg, k))[-2], dtype=float) for k in "xyzLMN"}
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (z_image - last["z"]) / last["N"]
            x[-1] = last["x"] + t * last["L"]
            y[-1] = last["y"] + t * last["M"]
            r = np.hypot(x, y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                widest = np.nanmax(np.where(np.isfinite(r), r, np.nan), axis=1)
            reach = np.fmax(reach, np.nan_to_num(widest, nan=0.0))
    return [float(v) for v in reach]


def _clear_semi(surface, reach: float) -> float:
    """A surface's clear semi-aperture: its physical aperture's outer radius
    when finite, else its rays' reach with the margin."""
    ap = getattr(surface, "aperture", None)
    r_max = _scalar(getattr(ap, "r_max", math.inf)) if ap is not None else math.inf
    if math.isfinite(r_max) and r_max > 0.0:
        return r_max
    return reach * (1.0 + CLEAR_APERTURE_MARGIN)


def _refit_r2(radius, conic: float, raw: list, semi: float):
    """Refit an even asphere with an r^2 term as a conic plus A4..A18: the
    vertex curvature absorbs the r^2 term exactly (it is the paraxial
    curvature), and a least-squares fit in r/semi the remainder. Returns
    ``(radius, coefficients, largest sag error in mm)``."""
    import numpy as np

    c = 0.0 if radius is None else 1.0 / radius
    c_new = c + 2.0 * raw[0]
    scale = max(semi, 1e-9)

    def conic_sag(cv, r):
        return cv * r**2 / (1.0 + np.sqrt(np.maximum(1.0 - (1.0 + conic) * cv**2 * r**2, 0.0)))

    r = np.linspace(0.0, scale, 801)
    target = conic_sag(c, r) + sum(a * r ** (2 * (i + 1)) for i, a in enumerate(raw))
    rest = target - conic_sag(c_new, r)
    u = r / scale
    basis = np.stack([u ** (2 * j + 4) for j in range(8)], axis=1)
    sol, *_ = np.linalg.lstsq(basis, rest, rcond=None)
    err = float(np.max(np.abs(basis @ sol - rest)))
    coeffs = [float(a / scale ** (2 * j + 4)) for j, a in enumerate(sol)]
    new_radius = None if c_new == 0.0 else 1.0 / c_new
    return new_radius, _trim(coeffs), err


def _note_placement(surface, who: str, notes: list) -> None:
    cs = surface.geometry.cs
    moved = [a for a in ("x", "y", "rx", "ry", "rz") if abs(_scalar(getattr(cs, a, 0.0))) > 1e-12]
    if moved:
        notes.append(f"{who} is decentred or tilted ({', '.join(moved)}); it is read as coaxial")


def _is_reflective(surface) -> bool:
    model = getattr(surface, "interaction_model", None)
    if model is not None and hasattr(model, "is_reflective"):
        return bool(model.is_reflective)
    return bool(getattr(surface, "is_reflective", False))


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

        return Material(m["name"], reference=m.get("reference"))
    raise ValueError(f"unknown material kind: {kind!r}")


#: refractiveindex.info formula names, as the prescription spells them.
_FORMULA_KIND = {"formula 1": "sellmeier", "formula 2": "sellmeier2", "formula 3": "polynomial"}

#: Where a three-term Sellmeier stand-in is fitted when a design's own
#: wavelengths do not reach further: the band the canvas renders.
_FIT_BAND_UM = (0.36, 1.0)


def _material_to_seq(mat, wavelengths: list, who: str, notes: list) -> dict:
    if mat is None:
        return {"kind": "air"}
    tn = type(mat).__name__
    if tn == "IdealMaterial":
        n = _scalar(mat.index)
        return {"kind": "air"} if abs(n - 1.0) < 1e-12 else {"kind": "ideal", "n": n}
    if tn in ("AbbeMaterial", "AbbeMaterialE"):
        return {"kind": "abbe", "nd": _scalar(mat.index), "vd": _scalar(mat.abbe)}
    if tn == "Mirror":
        return {"kind": "mirror"}
    if hasattr(mat, "_n_formula") and hasattr(mat, "coefficients"):
        out = {"kind": "glass", "name": mat.name}
        reference = _catalog_of(mat)
        if reference:
            out["reference"] = reference
        coefficients = [float(c) for c in _to_numpy(mat.coefficients).ravel()]
        kind = _FORMULA_KIND.get(mat._n_formula)
        if kind is not None:
            out["formula"] = {"kind": kind, "coefficients": coefficients}
            if not _is_three_term(kind, coefficients):
                out["sellmeier"] = _fit_sellmeier(mat, wavelengths)[0]
            return out
        fit, err = _fit_sellmeier(mat, wavelengths)
        out["sellmeier"] = fit
        notes.append(
            f"{who}: glass {mat.name} is published as {mat._n_formula}, which the scene does not "
            f"evaluate; it is a three-term Sellmeier fit within {err:.2g} in index"
        )
        return out
    # Fall back to a constant index read at the primary wavelength.
    try:
        n = _scalar(mat.n(wavelengths[0]))
    except Exception:  # pragma: no cover - defensive
        n = 1.0
    notes.append(f"{who}: a {tn} is read as a constant index of {n:.6f}")
    return {"kind": "ideal", "n": n}


def _catalog_of(mat) -> str | None:
    """What Optiland needs beside a glass's name to find its data again: the
    catalog of a catalog glass ("schott" for ``glass/schott/F2.yml``), or the
    dataset of a material measured by several authors ("Daimon-20" for
    ``main/CaF2/Daimon-20.yml``). Two CaF2 datasets differ in the fourth
    decimal, which moves a focal length."""
    parts = str(getattr(mat, "filename", "") or "").replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[-3] == "glass":
        return parts[-2]
    if len(parts) >= 3 and parts[-3] == "main":
        return parts[-1].rsplit(".", 1)[0]
    return None


def _is_three_term(kind: str, c: list) -> bool:
    """Is a formula a three-term Sellmeier curve (what the f32 preview
    evaluates), a constant term counting as one?"""
    if kind not in ("sellmeier", "sellmeier2"):
        return False
    return (len(c) - 1) // 2 + (1 if c[0] != 0.0 else 0) <= 3


def sellmeier_fit_error(mat, sellmeier: dict, wavelengths: list) -> float:
    """The largest index difference between an Optiland glass and a Sellmeier
    stand-in for it, over the band stand-ins are fitted on."""
    import numpy as np

    lam = np.linspace(*_fit_band(wavelengths), 400)
    return float(np.max(np.abs(_sellmeier_n(sellmeier, lam) - _index_of(mat, lam))))


def _fit_band(wavelengths: list) -> tuple:
    lo = min(_FIT_BAND_UM[0], 0.95 * min(wavelengths))
    hi = max(_FIT_BAND_UM[1], 1.05 * max(wavelengths))
    return lo, hi


def _index_of(mat, lam):
    import numpy as np

    return np.asarray(_to_numpy(mat.n(lam)), dtype=float).ravel()


def _sellmeier_n(s: dict, lam):
    import numpy as np

    l2 = lam**2
    n2 = 1.0 + sum(b * l2 / (l2 - c) for b, c in zip(s["b"], s["c"]))
    return np.sqrt(n2)


def _fit_sellmeier(mat, wavelengths: list):
    """A three-term Sellmeier curve fitted to a glass over the band the
    canvas renders (and the design's wavelengths), its poles kept outside the
    band. Returns ``(sellmeier, largest index error)``."""
    import numpy as np
    from scipy.optimize import least_squares

    lo, hi = _fit_band(wavelengths)
    lam = np.linspace(lo, hi, 400)
    target = _index_of(mat, lam)

    def model(p):
        return _sellmeier_n({"b": p[:3], "c": p[3:]}, lam)

    uv = (0.9 * lo) ** 2
    ir = (1.1 * hi) ** 2
    start = [1.04, 0.23, 1.01, min(0.006, 0.5 * uv), min(0.02, 0.9 * uv), max(103.56, 2.0 * ir)]
    fit = least_squares(
        lambda p: model(p) - target,
        start,
        bounds=([-10.0, -10.0, -10.0, 0.0, 0.0, ir], [10.0, 10.0, 10.0, uv, uv, 1e5]),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
        max_nfev=20000,
    )
    s = {"b": [float(v) for v in fit.x[:3]], "c": [float(v) for v in fit.x[3:]]}
    return s, float(np.max(np.abs(model(fit.x) - target)))


_APERTURE_KIND = {"epd": "EPD", "image_fno": "imageFNO", "object_na": "objectNA"}
_APERTURE_KIND_INV = {"EPD": "epd", "imageFNO": "image_fno", "objectNA": "object_na"}


def _aperture_to_seq(optic, notes: list | None = None) -> dict:
    aperture = optic.aperture if hasattr(optic, "aperture") else optic
    d = aperture.to_dict()
    ap_type = d.get("type") or d.get("ap_type") or getattr(aperture, "ap_type", None) or "EPD"
    value = d.get("value")
    if value is None:  # some versions nest the value differently
        value = next((v for k, v in d.items() if isinstance(v, (int, float))), 0.0)
    kind = _APERTURE_KIND_INV.get(ap_type)
    if kind is not None:
        return {"kind": kind, "value": float(value)}
    # Any other kind (a stop size, a working f-number) fixes an entrance pupil
    # diameter, which Optiland computes and the prescription records.
    epd = _scalar(optic.paraxial.EPD())
    if ap_type != "float_by_stop_size" and notes is not None:
        notes.append(
            f"the {ap_type} aperture is recorded as its entrance pupil diameter, {epd:g} mm"
        )
    return {"kind": "epd", "value": epd}


# --------------------------------------------------------------------------- #
# Field / wavelength read-back (tolerant of Optiland version differences)     #
# --------------------------------------------------------------------------- #
def _fields_of(optic, notes: list | None = None) -> list:
    fg = optic.fields
    fields = getattr(fg, "fields", None)
    if fields is None:
        return [0.0]
    if notes is not None and any(_scalar(f.x) != 0.0 for f in fields):
        notes.append("x fields are not carried; the fields are read along y")
    return [_scalar(f.y) for f in fields]


def _field_type_of(optic) -> str:
    fg = optic.fields
    ft = getattr(fg, "field_type", None)
    if ft is None:
        # Optiland 0.6 keeps the kind as a class: AngleField, ObjectHeightField.
        fd = getattr(fg, "field_definition", None)
        ft = getattr(fd, "field_type", None) or (type(fd).__name__ if fd is not None else None)
    return "object_height" if (ft and "height" in str(ft).lower()) else "angle"


def _wavelengths_of(optic) -> list:
    """The wavelengths, the primary first (the prescription's rule), then the
    rest in order."""
    wg = optic.wavelengths
    wls = getattr(wg, "wavelengths", None)
    if not wls:
        return [0.58756]
    values = [_scalar(w.value) for w in wls]
    primary = getattr(wg, "primary_index", 0)
    primary = primary if isinstance(primary, int) and 0 <= primary < len(values) else 0
    return [values[primary], *(v for j, v in enumerate(values) if j != primary)]


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


def _report(notes: list, strict: bool, error=None) -> None:
    if not notes:
        return
    msg = "; ".join(notes)
    if strict:
        raise (error or NotExportableError)(f"scene cannot be represented exactly: {msg}")
    warnings.warn(msg, FidelityWarning, stacklevel=3)


def _to_numpy(x):
    """A backend array (numpy or torch) as numpy."""
    try:
        from optiland import backend

        return backend.to_numpy(x)
    except Exception:  # pragma: no cover - Optiland-free callers pass numpy
        import numpy as np

        return np.asarray(x)
