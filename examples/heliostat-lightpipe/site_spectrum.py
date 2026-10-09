"""The direct solar spectrum the demonstrator's sun emits.

ASTM G173-03 (direct + circumsolar) is tabulated in 1 nm steps across the
visible, too coarse to hold the solar Fraunhofer lines: the sodium D lines are
0.6 nm apart, and each is about 0.03 nm wide at its core. So inside 380-720 nm
this takes the TSIS-1 Hybrid Solar Reference Spectrum (Coddington et al.,
Geophys. Res. Lett. 48, e2020GL091709, 2021) at its 0.005 nm resolution,
moves it from vacuum to air wavelengths (Edlen 1966, the air every line table
quotes), and carries it through the atmosphere G173 assumes: G173 divided by
the HSRS averaged to G173's own 1 nm. Outside 380-720 nm G173 is kept as it
is. The total differs from G173's by 0.02 W/m2.

G173's atmosphere is smooth at 1 nm, which smears the telluric lines: the O2
B band at 687 nm, the O2 gamma band near 630 nm, water near 590 and 650 nm.
So its own telluric bands are divided out (G173 averaged over each 1 nm) and
a line-by-line transmission put in instead: HITRAN 2024 O2 and H2O lines
through a US Standard Atmosphere from 160 m, 1.42 cm of water as G173 has,
air mass 1.5, at 0.001 nm (telluric_am1p5.csv and g173_telluric_divisor.csv
in ./.cache, written by D:/tmp/realism/atmosphere/scripts/telluric.py).
G173's continuum stays. With REAL_TELLURIC=0, or without those files, the
atmosphere stays G173's at 1 nm.

    python site_spectrum.py   # writes jena-direct-g173-hsrs.csv here

The two HSRS products are fetched from LASP's LISIRD server the first time and
kept in ./.cache.
"""
import os
import re
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
G173_RS = os.path.join(HERE, "..", "..", "crates", "oc-core", "src", "solar", "astm_g173.rs")
OUT = os.path.join(HERE, "jena-direct-g173-hsrs.csv")
LISIRD = "https://lasp.colorado.edu/lisird/latis/dap/"
FILES = {
    "hi": ("tsis1_hsrs_p005nm.csv?wavelength>=380&wavelength<=720", "hsrs_p005_380_720.csv"),
    "lo": ("tsis1_hsrs_1nm.csv?wavelength>=280&wavelength<=2600", "hsrs_1nm.csv"),
}


def fetch(key):
    query, name = FILES[key]
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(LISIRD + query, path)
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1))


def vac_to_air(nm):
    """Edlen (1966) standard air."""
    s2 = (1e3 / nm) ** 2
    n = 1 + 1e-8 * (8342.54 + 2406147 / (130 - s2) + 15998 / (38.9 - s2))
    return nm / n


def main():
    g = np.array(
        [[float(a), float(b)] for a, b in re.findall(r"\[([0-9.e+-]+), ([0-9.e+-]+)\]", open(G173_RS).read())]
    )
    hi, lo = fetch("hi"), fetch("lo")

    # The atmosphere G173 assumes, at its own resolution.
    t_atm = g[:, 1] / np.interp(g[:, 0], vac_to_air(lo[:, 0]), lo[:, 1])
    tel_path = os.path.join(CACHE, "telluric_am1p5.csv")
    div_path = os.path.join(CACHE, "g173_telluric_divisor.csv")
    telluric = os.environ.get("REAL_TELLURIC", "1") != "0" and os.path.exists(tel_path) and os.path.exists(div_path)
    if telluric:
        # G173's 1 nm telluric bands out of its atmosphere.
        div = np.loadtxt(div_path, delimiter=",", skiprows=1, usecols=(0, 1))
        t_atm = t_atm / np.interp(g[:, 0], div[:, 0], div[:, 1], left=1.0, right=1.0)
        tel = np.loadtxt(tel_path, delimiter=",", skiprows=1, usecols=(0, 2))

    # The 0.005 nm product is sampled every 0.001 nm: every fifth row is one
    # resolution element.
    hi = hi[::5]
    hi_air = vac_to_air(hi[:, 0])
    fine = hi[:, 1] * np.interp(hi_air, g[:, 0], t_atm)
    if telluric:
        # The line-by-line telluric lines, on the same air wavelengths.
        fine = fine * np.interp(hi_air, tel[:, 0], tel[:, 1], left=1.0, right=1.0)

    rows = np.vstack([g[g[:, 0] < hi_air[0]], np.column_stack([hi_air, fine]), g[g[:, 0] > hi_air[-1]]])
    assert np.all(np.diff(rows[:, 0]) > 0)
    with open(OUT, "w", newline="") as f:
        f.write("wavelength_nm,direct_W/m2/nm\n")
        for lam, e in rows:
            f.write(f"{lam:.4f},{e:.6g}\n")

    total, g_total = np.trapezoid(rows[:, 1], rows[:, 0]), np.trapezoid(g[:, 1], g[:, 0])
    print(f"{len(rows)} rows, {total:.2f} W/m2 (G173 {g_total:.2f}) -> {os.path.basename(OUT)}")
    near_b = (rows[:, 0] > 686.0) & (rows[:, 0] < 688.0)
    cont_b = np.percentile(rows[(rows[:, 0] > 684) & (rows[:, 0] < 686), 1], 95)
    print(f"telluric lines {'line by line' if telluric else 'G173 1 nm'}; B band minimum {rows[near_b, 1].min() / cont_b:.3f} of the continuum")
    for name, lam in [("D2", 588.995), ("D1", 589.592)]:
        near = (rows[:, 0] > lam - 0.3) & (rows[:, 0] < lam + 0.3)
        i = np.argmin(np.where(near, rows[:, 1], np.inf))
        cont = np.percentile(rows[(rows[:, 0] > 588) & (rows[:, 0] < 591), 1], 95)
        print(f"{name}: core at {rows[i, 0]:.3f} nm, {rows[i, 1] / cont:.3f} of the continuum")


if __name__ == "__main__":
    main()
