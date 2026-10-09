"""PMMA internal absorption from measured total transmittance of 3 mm sheet:
PLEXIGLAS GS solar clear 0Z023 (Aphalo photobiologyFilters, measured, 250-1100 nm)
and Arkema's colourless PLEXIGLAS at 3.0 mm (1100-2800 nm, the same for all
colourless grades). T = (1-R)^2 tau / (1 - R^2 tau^2), R from Beadie's index."""
import numpy as np
a = np.genfromtxt("D:/tmp/realism/data/pmma/plexiglas_GS_XT_3mm_total_T_measured_Aphalo_photobiologyFilters.csv", delimiter=",", names=True)
ir = np.genfromtxt("D:/tmp/realism/data/pmma/plexiglas_colorless_IR_3.0mm_Arkema_fig6_vector.csv", delimiter=",", names=True)
lam = np.concatenate([a["wavelength_nm"][a["wavelength_nm"] <= 1100], ir["wavelength_nm"][ir["wavelength_nm"] > 1100]]) / 1e3
T = np.concatenate([a["T_GS_solar_clear_0Z023_pct"][a["wavelength_nm"] <= 1100], ir["T_total_pct"][ir["wavelength_nm"] > 1100]]) / 100
b1, c1, b2, c2 = 1.19195585, 0.0101775815, 0.0262462388, 18.195663
n = np.sqrt(1 + b1 * lam**2 / (lam**2 - c1) + b2 * lam**2 / (lam**2 - c2))
R = ((n - 1) / (n + 1))**2
Tc = np.clip(T, 1e-4, None)
tau = (-(1 - R)**2 + np.sqrt((1 - R)**4 + 4 * R**2 * Tc**2)) / (2 * R**2 * Tc)
alpha = np.clip(-np.log(np.clip(tau, 1e-12, 1.0)) / 3.0, 0, None)
sel = lam <= 2.6
with open("tables/pmma.csv", "w") as f:
    f.write("# PMMA, PLEXIGLAS GS solar clear 0Z023 (3 mm total transmittance, measured; Aphalo, R package photobiologyFilters, 250-1100 nm) and Arkema colourless PLEXIGLAS 3.0 mm (1100-2600 nm, vector-extracted from the optical-properties brochure), turned into internal absorption with Beadie's index\nlambda_um,alpha_per_mm\n")
    for l, al in zip(lam[sel], alpha[sel]):
        f.write(f"{l:.4f},{al:.6e}\n")
for l in [0.3, 0.36, 0.375, 0.382, 0.393, 0.45, 0.55, 0.7, 0.9, 1.2, 1.4, 1.7, 2.0, 2.3]:
    i = np.argmin(abs(lam - l)); print(f"{lam[i]:.3f} T={T[i]:.3f} alpha={alpha[i]:.4f}/mm  tau(3mm)={np.exp(-3*alpha[i]):.3f}")
