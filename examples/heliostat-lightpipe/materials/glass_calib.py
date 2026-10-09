"""Fit the iron content of a solar mirror's glass to DLR's measured 4 mm Flabeg
mirror: alpha = alpha(10 ppm) + c (alpha(703 ppm) - alpha(10 ppm)), Vogt 2016,
silver behind it (Babar & Weaver), 8 deg, incoherent, Rubin's index."""
import io, yaml, numpy as np
from scipy.optimize import minimize_scalar
DB = "D:/_mos/workspace/dev/optiland/optiland/optiland/database/data-nk/"
def table(path):
    for e in yaml.safe_load(open(DB + path, encoding="utf-8"))["DATA"]:
        if e["type"].strip() in ("tabulated nk", "tabulated k"):
            return np.loadtxt(io.StringIO(e["data"]))
ag = table("main/Ag/Babar.yml")
v10, v703 = table("glass/misc/soda-lime/Vogt-10ppm.yml"), table("glass/misc/soda-lime/Vogt-703ppm.yml")
meas = np.loadtxt("D:/tmp/realism/data/mirrors/silvered_glass_Flabeg_2_4_5mm_hemispherical_8deg_DLR_digitized.csv", delimiter=",", skiprows=1)
lam = meas[:, 0] / 1e3; R4 = meas[:, 2] / 100
ng = 1.5130 - 0.003169 * lam**2 + 0.003962 / lam**2
k10 = np.interp(lam, v10[:, 0], v10[:, -1]); k703 = np.interp(lam, v703[:, 0], v703[:, -1])
nAg = np.interp(lam, ag[:, 0], ag[:, 1]) + 1j * np.interp(lam, ag[:, 0], ag[:, 2])
th = np.radians(8.0); st = np.sin(th) / ng; ct = np.sqrt(1 - st**2)
def fres(n1, n2, c1):
    s1 = np.sqrt(1 - c1**2) * n1 / n2
    c2 = np.sqrt(1 - s1**2 + 0j)
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2); rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    return 0.5 * (abs(rs)**2 + abs(rp)**2)
rf = fres(1.0, ng, np.cos(th)); rm = fres(ng + 0j, nAg, ct)
def model(c, d=4.0):
    k = k10 + c * (k703 - k10)
    tau = np.exp(-4 * np.pi * k * d * 1e3 / (lam * ct))
    return rf + (1 - rf)**2 * rm * tau**2 / (1 - rf * rm * tau**2)
g = np.loadtxt("D:/tmp/deck/ASTMG173.csv", delimiter=",", skiprows=2)
sun = np.interp(lam, g[:, 0] / 1e3, g[:, 3])
sel = lam >= 0.5
fit = minimize_scalar(lambda c: np.sum((model(c)[sel] - R4[sel])**2), bounds=(0, 1), method="bounded")
c = fit.x
print(f"iron fraction c = {c:.3f} -> about {10 + c*693:.0f} ppm Fe2O3")
for name, r in [("measured", R4), ("model", model(c)), ("10 ppm", model(0)), ("Rubin-like 703", model(1))]:
    print(f"{name:15s} solar {np.sum(r*sun)/np.sum(sun):.4f}  " + " ".join(f"{l:.2f}:{v:.3f}" for l, v in zip(lam[::40], r[::40])))
print("rms residual 500-2475:", np.sqrt(np.mean((model(c)[sel] - R4[sel])**2)))
# write the absorption table for the fitted glass, 0.30-2.50 um
L = np.round(np.arange(0.30, 2.501, 0.01), 3)
k = np.interp(L, v10[:, 0], v10[:, -1]) + c * (np.interp(L, v703[:, 0], v703[:, -1]) - np.interp(L, v10[:, 0], v10[:, -1]))
with open("tables/solar-glass.csv", "w") as f:
    f.write(f"# Solar mirror float glass: Vogt et al. 2016 soda-lime k at 10 and 703 ppm Fe2O3 (refractiveindex.info, CC0), mixed {c:.3f} of the way to 703 ppm (about {10 + c*693:.0f} ppm) so a 4 mm silvered mirror matches DLR's measured Flabeg curve (Sutter et al. 2018)\nlambda_um,alpha_per_mm\n")
    for l, kk in zip(L, k):
        f.write(f"{l:.3f},{4*np.pi*kk/(l*1e-3):.6e}\n")

# Invert instead: the glass transmission per pass that gives the measured mirror.
# R = rf + (1-rf)^2 rm x / (1 - rf rm x), x = tau^2  ->  x = (R - rf) / ((1-rf)^2 rm + rf rm (R - rf))
x = (R4 - rf) / ((1 - rf)**2 * rm + rf * rm * (R4 - rf))
x = np.clip(x, 1e-6, 1.0)
alpha = -np.log(x) / 2 * ct / 4.0   # per mm along the normal-equivalent path: tau = exp(-alpha d / ct)
alpha = np.clip(alpha, 0, None)
print("inverted alpha (1/mm):", " ".join(f"{l:.2f}:{a:.4f}" for l, a in zip(lam[::30], alpha[::30])))
with open("tables/solar-glass.csv", "w") as f:
    f.write("# Solar mirror float glass, 4 mm: the absorption that makes a silvered 4 mm sheet (silver Babar & Weaver behind Rubin's index) reproduce DLR's measured Flabeg mirror at 8 deg (Sutter et al., Solar Energy 2018, Fig. 6, digitized); it carries the glass's iron and water bands and the silver's real shortfall\nlambda_um,alpha_per_mm\n")
    for l, a in zip(lam, alpha):
        f.write(f"{l:.4f},{a:.6e}\n")
def back(a, d=4.0, deg=8.0):
    th = np.radians(deg); st = np.sin(th)/ng; c2 = np.sqrt(1-st**2)
    rf_ = fres(1.0, ng, np.cos(th)); rm_ = fres(ng+0j, nAg, c2)
    t = np.exp(-a*d/c2)
    return rf_ + (1-rf_)**2*rm_*t**2/(1-rf_*rm_*t**2)
for deg in (8, 25, 45):
    r = back(alpha, deg=deg)
    print(f"{deg} deg: solar {np.sum(r*sun)/np.sum(sun):.4f}  lum {np.sum(r*sun*np.exp(-285.4*(lam-0.559)**2))/np.sum(sun*np.exp(-285.4*(lam-0.559)**2)):.4f}")
