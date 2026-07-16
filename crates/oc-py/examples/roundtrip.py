"""Demo: the Optiland handoff, both directions (spec §6).

    python examples/roundtrip.py

Builds a cemented-doublet prescription, ships it to an Optiland ``Optic``, reads
it back, and checks the prescription survived byte-for-parameter. Requires
``optiland_canvas`` (``maturin develop``) and ``optiland``.
"""

from __future__ import annotations

import optiland_canvas as oc


def main() -> None:
    # A cemented doublet, written as an .ocanvas scene by the Rust core.
    prescription = {
        "surfaces": [
            {"radius": 60.0, "conic": 0.0, "thickness": 5.0, "semi_aperture": 14.0,
             "coefficients": [], "material": {"kind": "abbe", "nd": 1.62, "vd": 60.0},
             "reflective": False},
            {"radius": -45.0, "conic": 0.0, "thickness": 3.0, "semi_aperture": 14.0,
             "coefficients": [], "material": {"kind": "abbe", "nd": 1.52, "vd": 55.0},
             "reflective": False},
            {"radius": -150.0, "conic": 0.0, "thickness": 82.0, "semi_aperture": 14.0,
             "coefficients": [], "material": {"kind": "air"}, "reflective": False},
        ],
        "stop": 0,
        "aperture": {"kind": "epd", "value": 24.0},
        "fields": [0.0],
        "field_type": "angle",
        "wavelengths": [0.58756],
        "notes": [],
    }
    ocanvas = oc.seq_to_scene(prescription)
    print(f".ocanvas document: {len(ocanvas)} bytes, format v{oc.validate(ocanvas)}")

    # Export: scene -> Optiland Optic.
    optic = oc.to_optic(ocanvas)
    print(f"built Optiland Optic: EFL = {float(optic.paraxial.f2()):+.3f} mm")

    # Import: Optiland Optic -> scene -> prescription.
    back = oc.scene_to_seq(oc.from_optic(optic))

    print("\nsurface   radius   thickness   material")
    for s in back["surfaces"]:
        r = "  plane" if s["radius"] is None else f"{s['radius']:7.2f}"
        print(f"  {r}   {s['thickness']:7.2f}    {s['material']}")

    ok = back["surfaces"] == prescription["surfaces"]
    print(f"\nround-trip prescription identical: {ok}")


if __name__ == "__main__":
    main()
