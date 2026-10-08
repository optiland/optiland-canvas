"""Design the Galilean afocal of the heliostat light-pipe demonstrator
(crates/oc-trace/examples/heliostat_lightpipe.rs carries the numbers this
prints): a 1 m PMMA Fresnel lens, flat face to the sun, and a dense-flint
negative lens that sends the beam into a tube of 355 mm bore and 360 mm
outside, run through a 360 mm hole in a wall.

Colour: in a Galilean of magnification M the eyepiece cancels the
objective's primary colour when V_eyepiece = V_objective / M, because a
lens's contribution to axial colour goes as (ray height)^2 * power / V. Here
M ~ 3 and PMMA's V ~ 53, so the eyepiece wants V ~ 18; N-SF66 (20.9) is the
nearest real glass.

Frame: light travels +x, meridional height y. A facet's normal is (1, t),
t its slope; with the grooves facing +x the recipe's base bulge has slope t.
Every zone aims at one focus F1 past the valley plane at the design
wavelength, and the slopes are fitted as a paraboloid plus A4..A18. Facet
heights (under 0.4 mm) are neglected.

The eyepiece sits where the whole converging beam, sun disc included, is
355 mm across, the tube's bore. Its concave face is the Cartesian
surface (a hyperboloid) that makes the cone parallel.

Refractive indices: PMMA from Sultanova et al. 2009 (one-term Sellmeier,
0.44 to 1.05 um, V 53.2); N-SF66 from Schott.

    python design.py [F1_mm] [design_wavelength_um]      # defaults 3000 0.6
"""
import sys
import json
import numpy as np
from scipy.optimize import brentq, least_squares


def sell(B, C):
    return lambda l: np.sqrt(1 + sum(b * l * l / (l * l - c) for b, c in zip(B, C)))


n_pmma = sell([1.1819], [0.011313])
n_sf66 = sell([2.0245976, 0.470187196, 2.59970433], [0.0147053225, 0.0692998276, 161.817601])
for name, f in [("PMMA", n_pmma), ("N-SF66", n_sf66)]:
    nd, nF, nC = f(0.5875618), f(0.4861327), f(0.6562725)
    print(f"{name}: nd {nd:.5f} vd {(nd - 1) / (nF - nC):.2f}")


def refract(d, nrm, n1, n2):
    c = np.dot(d, nrm)
    if c < 0:
        nrm, c = -nrm, -c
    eta = n1 / n2
    k = 1 - eta**2 * (1 - c * c)
    return None if k < 0 else eta * d + (np.sqrt(k) - eta * c) * nrm


S = 3.0
F1 = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0   # valleys to focus
L0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6      # design wavelength
BEAM = 355.0      # whole beam, sun disc included, at the eyepiece vertex
SUN = 0.00467
X = np.array([1.0, 0.0])


def fresnel_out(h, t, lam):
    """Collimated ray at height h inside the PMMA hits a facet of slope t."""
    return refract(X, np.array([1.0, t]) / np.hypot(1, t), n_pmma(lam), 1.0)


# Zone slopes: every zone aims at the focus F1 past the valley plane at L0.
hs = np.linspace(0, 505, 506)
ts = [0.0]
for h in hs[1:]:
    def miss(t):
        d = fresnel_out(h, t, L0)
        return np.arctan2(d[1], d[0]) + np.arctan2(h, F1)
    ts.append(brentq(miss, 0.0, 0.8))
ts = np.array(ts)


def fit(h, t, rc=500.0):
    u = h / rc
    def slope(p):
        s = p[0] / rc * h
        for i, a in enumerate(p[1:]):
            e = 2 * (i + 2)
            s = s + e * a * u ** (e - 1)
        return s
    p0 = np.zeros(9); p0[0] = np.polyfit(h[:50], t[:50], 1)[0] * rc
    p = least_squares(lambda p: slope(p) - t, p0, xtol=1e-15, ftol=1e-15, gtol=1e-15).x
    return rc / p[0], [p[1 + i] / rc ** (2 * (i + 2) - 1) for i in range(8)], np.max(np.abs(slope(p) - t))


R1, A1, e1 = fit(hs, ts)
print(f"PMMA Fresnel: F1 {F1:.0f}, R {R1:.3f}, fit err {e1:.1e}, edge slope {np.degrees(np.arctan(ts[-6])):.1f} deg")

# Eyepiece where the whole beam is BEAM across: 500(1 - z/F1) + z*SUN = BEAM/2.
z_e = (500 - BEAM / 2) / (500 / F1 - SUN)
d_f = F1 - z_e
M = 500 / (500 * d_f / F1)
print(f"eyepiece {z_e:.1f} mm past the valleys, {d_f:.1f} mm before focus, M {M:.3f}, "
      f"exit half-angle {np.degrees(M * SUN):.2f} deg, need V {53.18 / M:.1f}")
ne = n_sf66(L0)
TE = 10.0


def eye(p, y0, d, lam):
    """Ray (height y0 at the vertex plane, direction d) through the concave
    conic front (R, k) and a flat back."""
    R, k = p
    c = 1 / R
    sag = lambda y: c * y * y / (1 + np.sqrt(1 - (1 + k) * c * c * y * y))
    dsag = lambda y: c * y / np.sqrt(1 - (1 + k) * c * c * y * y)
    u = d[1] / d[0]
    s = 0.0
    for _ in range(60):
        y = y0 + u * s
        s -= (s - sag(y)) / (1 - dsag(y) * u)
    y = y0 + u * s
    nr = np.array([1.0, -dsag(y)]); nr /= np.linalg.norm(nr)
    d = refract(d / np.linalg.norm(d), nr, 1.0, n_sf66(lam))
    return refract(d, X, n_sf66(lam), 1.0)


def res(p):
    return np.array([eye(p, y, np.array([1.0, -y / d_f]), L0)[1] for y in np.linspace(-175, 175, 21)])


pe = least_squares(res, [-(ne - 1) * d_f, -ne * ne], x_scale=[100, 1]).x
print(f"eyepiece N-SF66: R {pe[0]:.3f} k {pe[1]:.4f}, residual {np.max(np.abs(res(pe))):.1e} rad")


def slope_at(h):
    s = h / R1
    for i, a in enumerate(A1):
        s += 2 * (i + 2) * a * h ** (2 * (i + 2) - 1)
    return s


# Whole system, on axis, across the band: exit angle of rays at several heights.
print("exit angle (mrad) at h = 100/250/400/500 mm; sun alone would give +-%.1f mrad" % (M * SUN * 1e3))
worst = 0
for lam in [0.45, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.3]:
    row = []
    for h in [100, 250, 400, 500]:
        d = fresnel_out(h, slope_at(h), lam)
        y = h + z_e * d[1] / d[0]
        o = eye(pe, y, d, lam)
        row.append(1e3 * np.arctan2(o[1], o[0]))
    worst = max(worst, max(abs(np.array(row))))
    print(f"  {lam:.2f} um: {np.round(row, 2)}")
print(f"worst {worst:.2f} mrad")

json.dump({"F1": F1, "L0": L0, "S": S, "z_e": z_e, "d_f": d_f, "M": M, "TE": TE,
           "pmma": {"R": R1, "A": A1}, "eye": {"R": pe[0], "k": pe[1]}},
          open("design.json", "w"), indent=1)
