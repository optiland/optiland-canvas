"""Bulk absorption tables (lambda um, alpha 1/mm) from the refractiveindex.info
database (CC0) shipped with Optiland, and a three-term Sellmeier fit to Beadie
et al. 2015 for PMMA. Writes one CSV per material into ./tables."""
import io, os, yaml, numpy as np
from scipy.optimize import least_squares
DB = "D:/_mos/workspace/dev/optiland/optiland/optiland/database/data-nk/"
os.makedirs("tables", exist_ok=True)

def k_table(path):
    d = yaml.safe_load(open(DB + path, encoding="utf-8"))["DATA"]
    for e in d:
        if e["type"].strip() in ("tabulated k", "tabulated nk"):
            a = np.loadtxt(io.StringIO(e["data"]))
            return a[:, 0], a[:, -1]
    raise ValueError(path)

def write(name, lam, alpha, note):
    with open(f"tables/{name}.csv", "w") as f:
        f.write(f"# {note}\nlambda_um,alpha_per_mm\n")
        for l, a in zip(lam, alpha):
            f.write(f"{l:.4f},{a:.6e}\n")

def alpha_of(lam, k):
    return 4 * np.pi * k / (lam * 1e-3)

def extend_blue(lam, k, to=0.30):
    """Below the catalogue's first point the absorption edge keeps rising: carry
    the last two points' log slope down to `to`, capped at k = 1e-3."""
    if lam[0] <= to + 1e-9:
        return lam, k
    s = np.log(k[1] / k[0]) / (lam[1] - lam[0])
    ext = np.arange(to, lam[0] - 1e-9, 0.01)
    kx = np.minimum(k[0] * np.exp(s * (ext - lam[0])), 1e-3)
    return np.concatenate([ext, lam]), np.concatenate([kx, k])

for glass in ["N-SF66", "N-SK16", "N-F2", "N-BK7"]:
    lam, k = k_table(f"glass/schott/{glass}.yml")
    lam, k = extend_blue(lam, k)
    write(glass, lam, alpha_of(lam, k), f"{glass}: SCHOTT catalogue 2017-01-20b internal transmittance via refractiveindex.info (CC0); below the first catalogue point the edge's log slope is carried down to 0.30 um")
    print(glass, " ".join(f"{l:.2f}:{np.exp(-10*alpha_of(l, kk)):.3f}" for l, kk in zip(lam, k) if l in (0.3, 0.35, 0.37, 0.38, 0.39, 0.4, 0.42, 0.46, 0.5, 0.546, 0.7, 1.06, 1.97, 2.5)))

lam, k = k_table("glass/misc/soda-lime/Rubin-lowiron.yml")
sel = lam <= 2.6
write("soda-lime-low-iron", lam[sel], alpha_of(lam[sel], k[sel]), "Low-iron soda-lime silica glass, Rubin, Sol. Energy Mater. 12, 275 (1985), via refractiveindex.info (CC0)")
print("low-iron T(8 mm):", " ".join(f"{l:.2f}:{np.exp(-8*alpha_of(l, kk)):.3f}" for l, kk in zip(lam, k) if 0.3 <= l <= 2.5 and round(l*100) % 10 == 0))

# PMMA index: three-term Sellmeier fitted to Beadie et al. 2015 (0.42-1.62 um).
c = [2.1778, 6.1209e-3, -1.5004e-3, 2.3678e-2, -4.2137e-3, 7.3417e-4, -4.5042e-5]
beadie = lambda l: np.sqrt(c[0] + c[1]*l**2 + c[2]*l**4 + c[3]*l**-2 + c[4]*l**-4 + c[5]*l**-6 + c[6]*l**-8)
L = np.linspace(0.42, 1.62, 121)
def sell(p, l):
    b1, c1, b2, c2 = p
    return np.sqrt(1 + b1*l**2/(l**2 - c1) + b2*l**2/(l**2 - c2))
fit = least_squares(lambda p: sell(p, L) - beadie(L), [1.18, 0.0113, 0.5, 100.0], bounds=([0, 0, 0, 1], [3, 0.1, 50, 1e4]))
p = fit.x
print("PMMA Sellmeier b1 c1 b2 c2:", p, "max err", np.abs(fit.fun).max())
for l in [0.3, 0.35, 0.4, 0.5876, 1.0, 1.62, 2.0, 2.5]:
    print(f"  {l}: {sell(p, l):.5f}")
nd, nF, nC = sell(p, 0.5876), sell(p, 0.4861), sell(p, 0.6563)
print("  Vd", (nd - 1) / (nF - nC))
with open("tables/pmma-sellmeier.txt", "w") as f:
    f.write("# PMMA, three-term Sellmeier (b1 c1 b2 c2, c in um^2) fitted to Beadie et al., Appl. Opt. 54, F139 (2015), 0.42-1.62 um\n")
    f.write(" ".join(f"{v:.10g}" for v in p) + "\n")
