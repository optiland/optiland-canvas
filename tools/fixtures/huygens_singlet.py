"""Fixture for oc-wave's W6 oracle: a singlet's PSF from Optiland's HuygensPSF.

Run with the Optiland clone's own interpreter (the oracle is Optiland's current
development line, not a release; see docs/wave-optics.md, Oracles):

    <clone>/.venv/Scripts/python tools/fixtures/huygens_singlet.py [--probe]

writes crates/oc-wave/tests/fixtures/huygens_singlet.json: the prescription,
the paraxial image distance, the image grid, the PSF normalized so that the
same pupil with no aberration would peak at 100, and the Strehl ratio. The
Rust test (crates/oc-wave/tests/path.rs) builds the same lens in a canvas
scene, carries a uniform beam through it with the wave path, and compares.

The lens: an equiconvex singlet, R = ±50 mm, 4 mm thick, of a constant index
1.5168 (N-BK7's at the d line, which the canvas's Abbe-point glass gives
exactly at 587.6 nm), the stop at its first surface, an object at infinity
on axis, the image at the paraxial focus. This is an oracle: it verifies the
canvas's wave path against an independent implementation of the Huygens
integral over the exit pupil (docs/validation.md); it is not a measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import optiland
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.psf import HuygensPSF

LAMBDA_UM = 0.5876
N_GLASS = 1.5168
R1, R2, T = 50.0, -50.0, 4.0


def singlet(epd_mm: float) -> Optic:
    lens = Optic()
    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=R1, thickness=T, material=IdealMaterial(n=N_GLASS), is_stop=True)
    lens.add_surface(index=2, radius=R2, thickness=45.0)
    lens.add_surface(index=3)
    lens.set_aperture(aperture_type="EPD", value=epd_mm)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0.0)
    lens.add_wavelength(value=LAMBDA_UM, is_primary=True)
    lens.image_solve()
    return lens


def checkout() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(optiland.__file__)))
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "not a git checkout"
    return f"{root} @ {sha}"


def main() -> int:
    if "--probe" in sys.argv:
        for fno in (10.0, 7.0, 5.0, 4.0):
            efl = singlet(5.0).paraxial.f2()
            lens = singlet(efl / fno)
            psf = HuygensPSF(lens, field=(0, 0), wavelength=LAMBDA_UM, num_rays=128, image_size=64)
            print(f"f/{fno}: EPD {efl / fno:.4f} mm, EFL {efl:.6f}, Strehl {psf.strehl_ratio():.5f}")
        return 0
    fno = 10.0
    efl = singlet(5.0).paraxial.f2()
    epd = efl / fno
    lens = singlet(epd)
    # An odd grid, so a sample sits on the axis: HuygensPSF's Strehl is the
    # largest sample, and an even grid straddles the peak (0.8503 against
    # 0.8642 here). 512 pupil rays: the value is converged to 2e-5 (0.85039,
    # 0.85029, 0.85027 at 256, 384, 512 on the even grid).
    image_size = 65
    airy = 1.22 * LAMBDA_UM * 1e-3 * fno
    pitch = 2 * 3.0 * airy / (image_size - 1)
    psf = HuygensPSF(lens, field=(0, 0), wavelength=LAMBDA_UM, num_rays=512, image_size=image_size, pixel_pitch=pitch)
    data = np.asarray(psf.psf, dtype=float)
    x = np.linspace(-0.5 * image_size * pitch, 0.5 * image_size * pitch, image_size)
    out = {
        "oracle": checkout(),
        "what": "Optiland HuygensPSF of an equiconvex singlet, stop at its first surface, object at infinity on axis, image at the paraxial focus",
        "lambda_um": LAMBDA_UM,
        "n_glass": N_GLASS,
        "r1_mm": R1,
        "r2_mm": R2,
        "thickness_mm": T,
        "epd_mm": epd,
        "f_number": fno,
        "efl_mm": float(efl),
        "image_distance_from_back_vertex_mm": float(lens.surfaces[3].geometry.cs.z - lens.surfaces[2].geometry.cs.z),
        "num_rays": 512,
        "grid_note": "the image points are linspace(-image_size*pitch/2, +image_size*pitch/2, image_size), as HuygensPSF lays them",
        "pixel_pitch_mm": pitch,
        "image_size": image_size,
        "grid_mm": x.tolist(),
        "psf_rows_y_then_x": data.tolist(),
        "strehl": psf.strehl_ratio(),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "oc-wave", "tests", "fixtures", "huygens_singlet.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}: Strehl {out['strehl']:.6f}, EFL {efl:.6f}, image at {out['image_distance_from_back_vertex_mm']:.6f} mm, pitch {pitch * 1e3:.4f} um")
    return 0


if __name__ == "__main__":
    sys.exit(main())
