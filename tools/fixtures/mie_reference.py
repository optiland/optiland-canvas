"""Fixture for oc-wave's Mie reference (B30 of the wave-optics bundle, V03 of
the expert review): the same sphere computed by an independent implementation.

Run with an interpreter that has miepython (S. Prahl, https://github.com/scottprahl/miepython;
version recorded in the output) installed:

    python tools/fixtures/mie_reference.py

writes crates/oc-wave/tests/fixtures/mie_b30.json. The cases are the bundle's:
a sphere of index 1 (no contrast) and 1.5 in vacuum, at kR = 10, 100 and 1000
for a vacuum wavelength of 550 nm; the incident plane wave along +z, E along
+x, unit amplitude, time dependence exp(-i omega t). For each: the extinction
and scattering efficiencies, and the total exterior E and H (H in units where
the incident H is 1 along +y) at r/R = 1.2, 2 and 5, theta = 15, 45, 90 and
135 degrees and phi = 0, 45 and 90 degrees, in Cartesian components. The Rust
test (crates/oc-wave/tests/mie.rs) evaluates the same fields with
oc_wave::mie, an implementation written apart from this one (its own
recurrences and series), and compares. This verifies the reference, not the
wave path: the wave path is compared with the Mie fields separately.
"""

from __future__ import annotations

import json
import math
import os

import miepython
import miepython.field as field
import numpy as np
import scipy

LAMBDA_UM = 0.55
KR = [10.0, 100.0, 1000.0]
INDICES = [1.0, 1.5]
R_OVER_R = [1.2, 2.0, 5.0]
THETA = [math.pi / 12, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
PHI = [0.0, math.pi / 4, math.pi / 2]


def main() -> None:
    k = 2 * math.pi / LAMBDA_UM
    cases = []
    for n in INDICES:
        for kr in KR:
            radius = kr / k
            qext, qsca, _, _ = miepython.efficiencies_mx(complex(n, 0.0), kr)
            points = []
            for rr in R_OVER_R:
                for th in THETA:
                    for ph in PHI:
                        r = rr * radius
                        x, y, z = r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)
                        e, h = field.eh_near_cartesian(LAMBDA_UM, 2 * radius, complex(n, 0.0), 1.0, x, y, z)
                        e = np.asarray(e).reshape(3)
                        h = np.asarray(h).reshape(3)
                        points.append(
                            {
                                "r_over_R": rr,
                                "theta": th,
                                "phi": ph,
                                "kxyz": [k * x, k * y, k * z],
                                "E": [[float(c.real), float(c.imag)] for c in e],
                                "H": [[float(c.real), float(c.imag)] for c in h],
                            }
                        )
            cases.append({"n": n, "kR": kr, "qext": float(qext), "qsca": float(qsca), "points": points})
    out = {
        "source": "miepython " + miepython.__version__ + " (numpy " + np.__version__ + ", scipy " + scipy.__version__ + "): efficiencies_mx and field.eh_near_cartesian",
        "lambda_um": LAMBDA_UM,
        "convention": "exp(-i omega t); incident along +z, E along +x, |E0| = 1; H in units where the incident H is 1 along +y; total exterior field",
        "cases": cases,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "crates", "oc-wave", "tests", "fixtures", "mie_b30.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
