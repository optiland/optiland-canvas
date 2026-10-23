"""The canvas gallery: Optiland's sample designs as canvas scenes.

Every design in ``optiland.samples`` is read the way :func:`from_optic` reads
any ``Optic`` and built as a version 2 scene. The browser lists them and opens
the one you pick. Run it from the repository root::

    python -m optiland_canvas.gallery web/public/gallery

It writes one ``.ocanvas`` per design the canvas can build, and two JSON
files:

- ``index.json``, what the gallery lists: each design's name, what it is, its
  first-order numbers, what the scene holds only approximately, and, for a
  design the canvas cannot build, why not;
- ``checks.json``, the evidence: rays Optiland traced through each design,
  meridional and sagittal, at every field and wavelength, with where Optiland
  lands them. ``oc-trace/tests/gallery3.rs`` checks that each is a ray the
  scene's source for that field emits, traces it through the scene with the
  canvas's reference tracer, and holds the landing point to Optiland's.

A design's glasses keep their catalog formulas, so the reference tracer's
indices are Optiland's. The f32 preview evaluates only three-term Sellmeier
curves; for a glass published in another form, ``index.json`` records how far
the preview's fitted curve strays from it.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import re
import sys
import warnings

from . import (
    NotImportableError,
    _fit_band,
    _is_three_term,
    _scalar,
    _to_numpy,
    _wavelengths_of,
    seq_from_optic,
    seq_to_scene3,
    sellmeier_fit_error,
)

#: The sample modules, in the order the gallery lists them.
SAMPLE_MODULES = (
    "simple",
    "objectives",
    "eyepieces",
    "microscopes",
    "infrared",
    "lithography",
    "miscellaneous",
    "telescopes",
)

CATEGORIES = {
    "simple": "Singlets and doublets",
    "objectives": "Objectives",
    "eyepieces": "Eyepieces",
    "microscopes": "Microscope objectives",
    "infrared": "Infrared",
    "lithography": "Lithography",
    "miscellaneous": "Eye models",
    "telescopes": "Telescopes",
}

#: Names for the cards, where the class name does not read as one.
NAMES = {
    "AsphericSinglet": "Aspheric singlet",
    "CementedAchromat": "Cemented achromat",
    "Edmund_49_847": "Edmund Optics 49-847",
    "SingletStopSurf2": "Singlet, stop on the back face",
    "TelescopeDoublet": "Telescope doublet",
    "CookeTriplet": "Cooke triplet",
    "DoubleGauss": "Double Gauss",
    "HeliarLens": "Heliar",
    "LensWithFieldCorrector": "Lens with field corrector",
    "ObjectiveUS008879901": "Objective, US 8,879,901",
    "PetzvalLens": "Petzval",
    "ProjectionLens120FOV": "Projection lens, 120° field",
    "ProjectionLens160FOV": "Projection lens, 160° field",
    "ReverseTelephoto": "Reverse telephoto",
    "Telephoto": "Telephoto",
    "TelescopeObjective48Inch": "48-inch telescope objective",
    "TessarLens": "Tessar",
    "TripletTelescopeObjective": "Triplet telescope objective",
    "WideAngle100FOV": "Wide angle, 100° field",
    "WideAngle170FOV": "Wide angle, 170° field",
    "EyepieceErfle": "Erfle eyepiece",
    "Microscope20x": "20x microscope objective",
    "Objective60x": "60x microscope objective",
    "UVReflectingMicroscope": "53x UV reflecting objective",
    "InfraredTriplet": "Infrared triplet, 10.6 µm",
    "InfraredTripletF4": "Infrared triplet f/4, 4.2 µm",
    "UVProjectionLens": "UV projection lens, 248 nm",
    "NavarroWideAngleEye": "Navarro wide-angle eye",
    "HubbleTelescope": "Hubble Space Telescope",
}

#: The pupil points each check ray samples: the chief ray, the meridional
#: marginal rays on both sides, and a sagittal marginal ray.
CHECK_PUPIL = ((0.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0))


def optiland_version() -> str:
    """The installed Optiland's version, from its package metadata. Not
    ``optiland.__version__``: Optiland 0.6.3 has none, and 0.6.2's said 0.6.1."""
    from importlib.metadata import version

    return version("optiland")


def samples():
    """Every sample design as ``(module, class)``, modules in gallery order
    and classes in the order their module defines them."""
    from optiland.optic import Optic

    for module_name in SAMPLE_MODULES:
        module = importlib.import_module(f"optiland.samples.{module_name}")
        found = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, Optic) and cls is not Optic and cls.__module__ == module.__name__
        ]
        found.sort(key=lambda cls: inspect.getsourcelines(cls)[1])
        for cls in found:
            yield module_name, cls


def build(out_dir: str, *, rays_per_field: int = 64, pixels: int = 64) -> dict:
    """Write the gallery to ``out_dir`` and return its index."""
    os.makedirs(out_dir, exist_ok=True)
    designs = []
    checks = []
    for module_name, cls in samples():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optic = cls()
        entry = {
            "id": _slug(cls.__name__),
            "name": NAMES.get(cls.__name__, _words(cls.__name__)),
            "category": CATEGORIES.get(module_name, module_name),
            "source": f"optiland.samples.{module_name}.{cls.__name__}",
            **_describe(cls),
        }
        try:
            seq = seq_from_optic(optic)
            scene_json, notes = seq_to_scene3(seq, rays_per_field=rays_per_field, pixels=pixels)
        except NotImportableError as err:
            entry["status"] = "not_ported"
            entry["reasons"] = _reasons(str(err))
            designs.append(entry)
            continue
        scene = json.loads(scene_json)
        file = f"{entry['id']}.ocanvas"
        with open(os.path.join(out_dir, file), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(scene_json)
            fh.write("\n")
        entry["status"] = "ported"
        entry["file"] = file
        entry["specs"] = _specs(optic, seq, scene)
        entry["notes"] = notes
        entry["preview"] = _preview_fits(optic, seq)
        entry["outline"] = outline(scene)
        designs.append(entry)
        checks.append(
            {
                "id": entry["id"],
                "file": file,
                # Aimed rays are the stop's, and the scene's beams overfill
                # them: a check ray lies inside its beam rather than at the
                # beam's own pupil radius.
                "aimed": bool(seq.get("aimed_beams")),
                "rays": _check_rays(optic, scene),
            }
        )

    index = {
        "generator": "optiland_canvas.gallery",
        "optiland": optiland_version(),
        "designs": designs,
    }
    _write_json(os.path.join(out_dir, "index.json"), index)
    _write_json(
        os.path.join(out_dir, "checks.json"),
        {"optiland": optiland_version(), "designs": checks},
    )
    return index


# --------------------------------------------------------------------------- #
# What a card says                                                            #
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    """``UVProjectionLens`` -> ``uv-projection-lens``, ``Objective60x`` ->
    ``objective-60x``, ``ProjectionLens120FOV`` -> ``projection-lens-120-fov``."""
    words = re.sub(
        r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-z]{2})(?=[0-9])|(?<=[0-9])(?=[A-Z])",
        "-",
        name.replace("_", "-"),
    )
    return re.sub(r"-+", "-", words).strip("-").lower()


def _words(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.replace("_", " ")).capitalize()


def _describe(cls) -> dict:
    """The class docstring's summary line, and its reference when it cites one."""
    doc = inspect.getdoc(cls) or ""
    paragraphs = [p.strip() for p in doc.split("\n\n") if p.strip()]
    summary = paragraphs[0].replace("\n", " ") if paragraphs else ""
    reference = next(
        (
            re.sub(r"^Reference:\s*", "", p.replace("\n", " "))
            for p in paragraphs[1:]
            if p.startswith("Reference")
        ),
        None,
    )
    out = {"description": summary}
    if reference:
        out["reference"] = reference
    return out


def _reasons(message: str) -> list:
    lines = [line.strip() for line in message.splitlines()]
    return [line[2:] for line in lines if line.startswith("- ")] or [message]


def _round(x: float, digits: int = 6) -> float:
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def _specs(optic, seq: dict, scene: dict) -> dict:
    p = optic.paraxial
    fields = seq["fields"] or [0.0]
    angle = seq["field_type"] == "angle"
    specs = {
        "efl_mm": _round(_scalar(p.f2())),
        "epd_mm": _round(_scalar(p.EPD())),
        "field": _round(max(abs(f) for f in fields)),
        "field_unit": "deg" if angle else "mm",
        "fields": len(fields),
        "wavelengths_um": seq["wavelengths"],
        "elements": len(scene["bodies"]),
        "surfaces": len(seq["surfaces"]),
    }
    if seq["object_distance"] is None:
        specs["fno"] = _round(_scalar(p.FNO()))
    else:
        specs["object_distance_mm"] = _round(seq["object_distance"])
        if seq["aperture"]["kind"] == "object_na":
            specs["object_na"] = seq["aperture"]["value"]
    return specs


def _preview_fits(optic, seq: dict) -> list:
    """Each glass the preview draws with a fitted Sellmeier curve, and how far
    the fit strays from the catalog's formula over the band it covers."""
    out = []
    seen = set()
    wavelengths = _wavelengths_of(optic)
    for i, s in enumerate(seq["surfaces"]):
        m = s["material"]
        if m.get("kind") != "glass" or "sellmeier" not in m or m["name"] in seen:
            continue
        formula = m.get("formula")
        if formula is not None and _is_three_term(formula["kind"], formula["coefficients"]):
            continue
        seen.add(m["name"])
        glass = optic.surfaces.surfaces[i + 1].material_post
        lo, hi = _fit_band(wavelengths)
        out.append(
            {
                "glass": m["name"],
                "max_index_error": float(f"{sellmeier_fit_error(glass, m['sellmeier'], wavelengths):.3g}"),
                "band_um": [_round(lo, 4), _round(hi, 4)],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# The card's drawing                                                          #
# --------------------------------------------------------------------------- #
def outline(scene: dict) -> dict:
    """The design's meridional section for its card, as SVG path data.

    Each body's profile, the stop's mask on both sides of its opening, and
    the image plane, in a ``viewBox`` with ``y`` flipped to point down as SVG
    does. Drawn from the scene itself, so the card shows what opens.
    """
    glass = []
    points = []
    for body in scene["bodies"]:
        profile = body["geometry"].get("Profile")
        if profile is None:
            continue
        ox, oy = body["common"]["transform"]["pos"][:2]
        loop = []
        for edge in profile["profile"]["edges"]:
            pts = _sample_curve(edge["curve"])
            if loop and _dist(loop[-1], pts[-1]) < _dist(loop[-1], pts[0]):
                pts.reverse()
            loop.extend(pts)
        loop = [(x + ox, y + oy) for x, y in loop]
        points.extend(loop)
        glass.append(loop)
    marks = []
    for ap in scene["apertures"]:
        x = ap["common"]["transform"]["pos"][0]
        inner = ap["shape"]["clear_opening"].get("Disc", {}).get("radius_mm", 0.0)
        outer = (ap["shape"].get("outer_extent") or {}).get("Disc", {}).get("radius_mm", inner)
        marks.append([(x, inner), (x, outer)])
        marks.append([(x, -inner), (x, -outer)])
    image = []
    for det in scene["detectors"]:
        x = det["common"]["transform"]["pos"][0]
        r = det["geometry"].get("Disc", {}).get("radius_mm", 0.0)
        image.append([(x, -r), (x, r)])
    points += [p for seg in marks + image for p in seg]
    xs = [p[0] for p in points] or [0.0]
    ys = [p[1] for p in points] or [0.0]
    width = max(max(xs) - min(xs), 1e-9)
    height = max(2.0 * max(abs(y) for y in ys), 1e-9)
    pad = 0.04 * max(width, height)
    digits = max(0, 3 - int(math.floor(math.log10(max(width, height)))))

    def fmt(v: float) -> str:
        text = f"{v:.{digits}f}".rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text

    def path(loops: list, closed: bool) -> str:
        parts = []
        for loop in loops:
            # Curves meet end to end, so each seam is a point written twice.
            kept = [loop[0]]
            for p in loop[1:]:
                if (fmt(p[0]), fmt(p[1])) != (fmt(kept[-1][0]), fmt(kept[-1][1])):
                    kept.append(p)
            head, *rest = kept
            parts.append(
                "M" + fmt(head[0]) + " " + fmt(-head[1])
                + "".join("L" + fmt(x) + " " + fmt(-y) for x, y in rest)
                + ("Z" if closed else "")
            )
        return "".join(parts)

    return {
        "view_box": " ".join(
            fmt(v) for v in (min(xs) - pad, -height / 2 - pad, width + 2 * pad, height + 2 * pad)
        ),
        "glass": path(glass, True),
        "stop": path(marks, False),
        "image": path(image, False),
    }


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sample_curve(curve: dict, n: int = 24) -> list:
    """A profile curve as points, a face sampled across its aperture with the
    same sag the scene's surface has."""
    if "Line" in curve:
        return [tuple(curve["Line"]["a"]), tuple(curve["Line"]["b"])]
    if "Conic" in curve:
        base, coeffs = curve["Conic"], [0.0] * 8
    elif "EvenAsphere" in curve:
        base, coeffs = curve["EvenAsphere"]["base"], curve["EvenAsphere"]["coeffs"]
    else:
        return []
    c, k, h = base["c"], base["k"], base["half_ap"]
    vx, vy = base["vertex"]
    pts = []
    for i in range(n + 1):
        y = -h + 2.0 * h * i / n
        r2 = y * y
        sag = c * r2 / (1.0 + math.sqrt(max(1.0 - (1.0 + k) * c * c * r2, 0.0)))
        sag += sum(a * r2 ** (j + 2) for j, a in enumerate(coeffs))
        pts.append((vx + sag, vy + y))
    return pts


# --------------------------------------------------------------------------- #
# The evidence                                                                #
# --------------------------------------------------------------------------- #
def _check_rays(optic, scene: dict) -> list:
    """Rays Optiland traced through the design, in canvas coordinates, with
    where each crosses the image plane.

    Optiland's axis ``z`` is the canvas's ``x``, its ``y`` the canvas's ``y``,
    and its ``x`` the canvas's ``-z`` (a rotation, so handedness is kept). The
    landing point is where the ray, leaving the last surface, crosses the
    plane square to the axis at the image surface's vertex: Optiland's image
    point for a flat image, and the flat detector's reading for a curved one.
    """
    import numpy as np

    rays = []
    fields = optic.fields.get_field_coords()
    values = [float(_to_numpy(f.y).ravel()[0]) for f in optic.fields.fields] or [0.0]
    n = len(optic.surfaces.surfaces)
    z_image = float(_to_numpy(optic.surfaces.surfaces[-1].geometry.cs.z).ravel()[0])
    for j, (hx, hy) in enumerate(fields):
        for w in _wavelengths_of(optic):
            px = np.array([p[0] for p in CHECK_PUPIL])
            py = np.array([p[1] for p in CHECK_PUPIL])
            optic.trace_generic(Hx=float(hx), Hy=float(hy), Px=px, Py=py, wavelength=w)
            sg = optic.surfaces
            cols = {
                k: np.asarray(_to_numpy(getattr(sg, k)), dtype=float)
                for k in ("x", "y", "z", "L", "M", "N")
            }
            for r, (qx, qy) in enumerate(CHECK_PUPIL):
                start = [cols[k][0, r] for k in ("x", "y", "z", "L", "M", "N")]
                last = [cols[k][n - 2, r] for k in ("x", "y", "z", "L", "M", "N")]
                if not all(math.isfinite(v) for v in start + last) or last[5] == 0.0:
                    continue
                t = (z_image - last[2]) / last[5]
                land = (last[0] + t * last[3], last[1] + t * last[4], z_image)
                rays.append(
                    {
                        "field": values[j] if j < len(values) else 0.0,
                        "source": j,
                        "wavelength_um": w,
                        "pupil": [qx, qy],
                        "origin": _canvas(start[0], start[1], start[2]),
                        "dir": _canvas(start[3], start[4], start[5]),
                        "image": _canvas(*land),
                    }
                )
    return rays


def _canvas(x: float, y: float, z: float) -> list:
    """An Optiland point or direction in canvas coordinates."""
    return [float(z), float(y), float(-x) + 0.0]


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def main(argv: list | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m optiland_canvas.gallery <out_dir>", file=sys.stderr)
        return 2
    index = build(args[0])
    ported = sum(d["status"] == "ported" for d in index["designs"])
    print(f"{ported} of {len(index['designs'])} designs written to {args[0]}")
    for d in index["designs"]:
        if d["status"] != "ported":
            print(f"  not ported: {d['name']}: {'; '.join(d['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
