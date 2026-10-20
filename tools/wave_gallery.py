#!/usr/bin/env python3
"""Validation figures for oc-wave, from the datasets of
crates/oc-wave/examples/gallery.rs.

    cargo run -p oc-wave --release --example gallery -- <dir>
    python tools/wave_gallery.py <dir>

Reads ``<dir>/<dataset>.json`` and the raw little-endian f64 arrays it
describes, and writes ``<dir>/figures/F*.png`` (200 dpi), ``F*.svg`` and
``<dir>/figures/manifest.json`` (title, caption, sources and the measured
numbers of every figure).

Simulation is drawn as markers or an image; closed-form references as solid
lines. Every number quoted in a caption is read from the dataset's JSON. The
closed forms were evaluated in Rust with the crate's own special functions;
this script recomputes them with SciPy as an independent check and records
the agreement in the manifest.

Style: the light-mode reference palette of the dataviz method (categorical
blue, orange, aqua in fixed order, validated for colour-vision deficiency on
all pairs), solid hairline axes, one y-scale per axes, a perceptually uniform
monotone-lightness colormap (inferno) with a colorbar on every image.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter  # noqa: E402
from scipy import special  # noqa: E402

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CMAP = "inferno"
WIDTH_IN = 7.2
DPI = 200

SOURCES = {
    "Goodman": "J. W. Goodman, Introduction to Fourier Optics, 4th ed., W. H. Freeman/Macmillan Learning (2017), ISBN 978-1-319-11916-4.",
    "Matsushima 2009": "K. Matsushima, T. Shimobaba, Opt. Express 17(22), 19662–19673 (2009), doi:10.1364/OE.17.019662.",
    "Voelz 2009": "D. G. Voelz, M. C. Roggemann, Appl. Opt. 48(32), 6132–6142 (2009), doi:10.1364/AO.48.006132.",
    "Shen 2006": "F. Shen, A. Wang, Appl. Opt. 45(6), 1102–1110 (2006), doi:10.1364/AO.45.001102.",
    "Heurtley 1973": "W. H. Heurtley, J. Opt. Soc. Am. 63(8), 1003–1008 (1973).",
    "Kogelnik 1966": "H. Kogelnik, T. Li, Appl. Opt. 5(10), 1550–1567 (1966).",
    "Self 1983": "S. A. Self, Appl. Opt. 22(5), 658–661 (1983).",
    "Wen 2013": "J. Wen, Y. Zhang, M. Xiao, Adv. Opt. Photon. 5, 83–130 (2013).",
    "DLMF": "NIST Digital Library of Mathematical Functions, ch. 7 and 10.",
    "Born & Wolf": "M. Born, E. Wolf, Principles of Optics, 7th ed., CUP (1999).",
    "Popoff 2024": "S. M. Popoff et al., J. Phys. Photonics 6, 043001 (2024), doi:10.1088/2515-7647/ad6dc0.",
    "TI DLPA083B": "Texas Instruments, DMD Optical Efficiency for Visible Wavelengths, application note DLPA083B (2019, rev. 2023).",
    "Harvey 1998": "J. E. Harvey, C. L. Vernold, Appl. Opt. 37, 8158 (1998).",
    "Harvey 2011": "J. E. Harvey et al., Proc. SPIE 8122, 812205 (2011).",
    "Swanson 1989": "G. J. Swanson, MIT Lincoln Laboratory Technical Report TR-854 (1989).",
    "Zhang 2014": "Z. Zhang, Z. You, D. Chu, Light Sci. Appl. 3, e213 (2014).",
    "Persson 2012": "M. Persson, D. Engström, M. Goksör, Opt. Express 20, 22334 (2012).",
    "JOSA A 2004": "Fringing-field effect in liquid-crystal beam-steering devices: an approximate analytical model, J. Opt. Soc. Am. A 21(10), 1996 (2004).",
    "Optiland": "Optiland, https://github.com/optiland/optiland, master at 1e0513902c68 (2026-10-14), cloned 2026-10-19: HuygensPSF.",
    "Wyrowski 2011": "F. Wyrowski, M. Kuhn, Introduction to field tracing, J. Mod. Opt. 58(5-6), 449-466 (2011), doi:10.1080/09500340.2010.532237.",
    "Rabiner 1969": "L. R. Rabiner, R. W. Schafer, C. M. Rader, The chirp z-transform algorithm, IEEE Trans. Audio Electroacoust. 17(2), 86-92 (1969), doi:10.1109/TAU.1969.1162034.",
    "Allen 1992": "L. Allen et al., Phys. Rev. A 45, 8185 (1992).",
}

MAGENTA = "#e87ba4"
YELLOW = "#eda100"
# Sequential blue ramp of the reference palette, light to dark, for magnitudes on marks.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.3,
            "lines.linewidth": 1.1,
            "svg.fonttype": "none",
            "image.interpolation": "nearest",
        }
    )


def sim_style(color=BLUE, marker="o", ms=3.0, filled=True):
    return dict(
        ls="none",
        marker=marker,
        ms=ms,
        mfc=color if filled else SURFACE,
        mec=SURFACE if filled else color,
        mew=0.5 if filled else 0.8,
    )


REF = dict(color=INK, lw=1.0, solid_capstyle="round")


def reciprocal(c: float):
    """x -> c/x for a secondary axis, quiet at 0 (matplotlib probes it)."""

    def f(v):
        with np.errstate(divide="ignore", invalid="ignore"):
            return c / np.asarray(v, float)

    return f


def label(ax, letter: str, dx: float = -30, dy: float = 4, y_fig: float | None = None) -> None:
    """Panel letter `dx` points left of the axes. With `y_fig`, the letter
    sits at that figure height, so a row of panels shares one baseline."""
    fig = ax.figure
    if y_fig is None:
        ax.annotate(
            f"({letter})",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=INK,
        )
        return
    x0 = ax.get_position().x0 + dx / 72.0 / fig.get_size_inches()[0]
    fig.text(x0, y_fig, f"({letter})", fontsize=9, fontweight="bold", ha="left", va="bottom", color=INK)


def frame(ax) -> None:
    """Images get a full frame."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(AXIS)


def grid(ax, which="major", axis="both") -> None:
    ax.grid(True, which=which, axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def colorbar(fig, ax, im, text: str, extend: str = "neither", gap: float = 0.035, width: float = 0.05):
    cax = ax.inset_axes([1.0 + gap, 0.0, width, 1.0])
    cb = fig.colorbar(im, cax=cax, extend=extend)
    cb.set_label(text, fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, color=INK2)
    cb.outline.set_edgecolor(AXIS)
    cb.outline.set_linewidth(0.6)
    return cb


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 2) -> str:
    """2.8 × 10⁻⁴ for captions."""
    if x is None or not math.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    if round(abs(m), digits - 1) >= 10:
        m /= 10
        e += 1
    if -2 <= e <= 2:
        return f"{x:.{max(digits - 1 - e, 0)}f}"
    return f"{m:.{digits - 1}f} × 10{str(e).translate(_SUP)}"


def tex(x: float, digits: int = 2) -> str:
    """$2.8\\times10^{-4}$ for figure text."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    if round(abs(m), digits - 1) >= 10:
        m /= 10
        e += 1
    if -2 <= e <= 2:
        return f"{x:.{max(digits - 1 - e, 0)}f}"
    return rf"${m:.{digits - 1}f}\times10^{{{e}}}$"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Data:
    def __init__(self, root: Path):
        self.root = root

    def load(self, name: str):
        meta = json.loads((self.root / f"{name}.json").read_text(encoding="utf-8"))
        arrays = {}
        for key, spec in meta["arrays"].items():
            a = np.fromfile(self.root / spec["file"], dtype=np.dtype(spec["dtype"]))
            arrays[key] = a.reshape(spec["shape"])
        return meta, arrays


def save(fig, outdir: Path, stem: str) -> dict:
    png = outdir / f"{stem}.png"
    svg = outdir / f"{stem}.svg"
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg)
    w, h = fig.get_size_inches()
    plt.close(fig)
    return {"png": str(png.resolve()), "svg": str(svg.resolve()), "size_in": [round(w, 3), round(h, 3)], "dpi": DPI}


# ---------------------------------------------------------------------------
# Independent checks of the closed forms (SciPy)
# ---------------------------------------------------------------------------


def scipy_slit(w, lam_mm, z, x):
    s = np.sqrt(2.0 / (lam_mm * z))
    s1, c1 = special.fresnel(-s * (0.5 * w + x))  # scipy returns (S, C)
    s2, c2 = special.fresnel(s * (0.5 * w - x))
    return ((c2 - c1) + 1j * (s2 - s1)) / np.sqrt(2.0)


def crosschecks(D: Data) -> dict:
    out = {}
    _, a = D.load("f1_airy")
    v = a["dense_v"]
    with np.errstate(invalid="ignore", divide="ignore"):
        airy = np.where(v == 0, 1.0, (2 * special.j1(v) / v) ** 2)
    out["airy_intensity_max_abs_diff"] = float(np.max(np.abs(airy - a["dense_airy"])))
    ee = 1 - special.j0(a["ee_v"]) ** 2 - special.j1(a["ee_v"]) ** 2
    out["encircled_energy_max_abs_diff"] = float(np.max(np.abs(ee - a["ee_analytic"])))

    m, a = D.load("f2_heurtley")
    lam, rad = 0.5e-3, m["parameters"]["a_mm"]
    z = rad**2 / (lam * a["dense_nf"])
    k = 2 * np.pi / lam
    r = np.hypot(z, rad)
    h = np.exp(1j * k * z) - z / r * np.exp(1j * k * r)
    out["heurtley_intensity_max_abs_diff"] = float(np.max(np.abs(np.abs(h) ** 2 - a["dense_heurtley_int"])))

    m, a = D.load("f3_fresnel_square")
    p = m["parameters"]
    x = a["m201_line_x_mm"]
    ix = scipy_slit(p["w_mm"], p["lambda_um"] * 1e-3, p["z_mm"], x)
    i0 = scipy_slit(p["w_mm"], p["lambda_um"] * 1e-3, p["z_mm"], 0.0)
    out["fresnel_square_line_intensity_max_abs_diff"] = float(np.max(np.abs(np.abs(ix * i0) ** 2 - a["m201_line_analytic"])))

    m, a = D.load("f4_fresnel_to_fraunhofer")
    worst = 0.0
    for c in m["errors"]["cases"]:
        t = f"c{c['index']}"
        x = a[f"{t}_x_mm"]
        ix = scipy_slit(0.5, 0.5e-3, c["z_mm"], x) * scipy_slit(0.5, 0.5e-3, c["z_mm"], 0.0)
        ii = np.abs(ix) ** 2
        # The dataset normalizes by the analytic maximum on the displayed line.
        worst = max(worst, float(np.max(np.abs(ii / ii.max() - a[f"{t}_analytic"]))))
    out["fresnel_to_fraunhofer_normalized_intensity_max_abs_diff"] = worst
    out.update(talbot_check(D))
    out.update(vortex_check(D))
    return out


def talbot_check(D: Data) -> dict:
    """The binary grating's orders propagated exactly, and paraxially, by an
    independent NumPy order sum on the crate's own sampled grating."""
    m, a = D.load("f6_talbot")
    p = m["parameters"]
    x, t2 = a["x_mm"], a["profile_input"]
    t = np.sqrt(t2)
    lam = p["lambda_um"] * 1e-3
    f = np.fft.fftfreq(len(x), x[1] - x[0])
    T = np.fft.fft(t)
    exact = np.fft.ifft(T * np.exp(2j * np.pi * p["zT_exact_mm"] * np.sqrt(1 / lam**2 - f**2 + 0j)))
    z = p["zT_paraxial_mm"]
    parax = np.fft.ifft(T * np.exp(2j * np.pi * z / lam) * np.exp(-1j * np.pi * lam * z * f**2))
    return {
        "talbot_numpy_exact_vs_crate_max_intensity_diff": float(np.max(np.abs(np.abs(exact) ** 2 - a["profile_zt_exact"]))),
        "talbot_paraxial_model_revival_worst_intensity_err": float(np.max(np.abs(np.abs(parax) ** 2 - t2))),
    }


# ---------------------------------------------------------------------------
# F1: the Airy pattern
# ---------------------------------------------------------------------------


def fig1(D: Data, outdir: Path) -> dict:
    m, a = D.load("f1_airy")
    p, e = m["parameters"], m["errors"]
    img, c = a["image_intensity"], a["image_coord_mm"]
    hp = 0.5 * (c[1] - c[0])
    fig = plt.figure(figsize=(WIDTH_IN, 2.6))
    gs = fig.add_gridspec(1, 3, left=0.068, right=0.985, bottom=0.175, top=0.86, wspace=0.85, width_ratios=[1, 1.08, 1.08])

    ax = fig.add_subplot(gs[0])
    frame(ax)
    im = ax.imshow(
        np.log10(np.clip(img, 1e-12, None)),
        origin="lower",
        extent=[c[0] - hp, c[-1] + hp, c[0] - hp, c[-1] + hp],
        cmap=CMAP,
        vmin=-6,
        vmax=0,
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    colorbar(fig, ax, im, r"$\log_{10}\,I/I_0$", extend="min")
    ax.set_title(f"far field at z = {p['z_mm']/1000:g} m", fontsize=7.5, color=INK2)
    label(ax, "a", dx=-32, y_fig=0.93)

    ax = fig.add_subplot(gs[1])
    v, s = a["profile_v"], a["profile_sim"]
    dv, di = a["dense_v"], a["dense_airy"]
    ax.semilogy(dv, np.clip(di, 1e-14, None), **REF, label=r"$[2J_1(v)/v]^2$", zorder=2)
    ax.semilogy(v[::2], s[::2], **sim_style(ms=3.0), label="simulation", zorder=3)
    zeros = [3.8317, 7.0156, 10.1735]
    for vz in zeros:
        ax.axvline(vz, color=MUTED, lw=0.6, zorder=1)
    ax.set_ylim(1e-8, 2.5)
    ax.set_xlim(0, v[-1])
    for vz, txt in zip(zeros, ["3.832", "7.016", "10.17"]):
        ax.text(vz, 2.7, txt, rotation=0, ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlabel(r"$v = \pi D r/(\lambda z)$")
    ax.set_ylabel(r"$I/I_0$ along $y = 0$")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower left", fontsize=6.5, handletextpad=0.4, borderaxespad=0.2)
    grid(ax, axis="y")
    label(ax, "b", dx=-40, y_fig=0.93)

    ax = fig.add_subplot(gs[2])
    ev, es, ea = a["ee_v"], a["ee_sim"], a["ee_analytic"]
    ax.plot(ev, ea, **REF, label=r"$1 - J_0^2(v) - J_1^2(v)$", zorder=2)
    ax.plot(ev[::5], es[::5], **sim_style(ms=3.0), label="simulation", zorder=3)
    v1 = 3.8317
    ax.axvline(v1, color=MUTED, lw=0.6, zorder=1)
    ax.axhline(e["ee_first_ring_analytic"], color=MUTED, lw=0.6, zorder=1)
    ax.annotate(
        f"first dark ring\nanalytic {e['ee_first_ring_analytic']:.4f}\nsimulated {e['ee_first_ring_sim']:.4f}",
        xy=(v1, e["ee_first_ring_analytic"]),
        xytext=(7.0, 0.47),
        fontsize=6.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=2),
    )
    ax.set_xlim(0, ev[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$v = \pi D r/(\lambda z)$")
    ax.set_ylabel(r"encircled energy / $(\pi D^2/4)$")
    ax.legend(loc="lower right", fontsize=6.5, handletextpad=0.4)
    grid(ax)
    label(ax, "c", dx=-36, y_fig=0.93)

    files = save(fig, outdir, "F1_airy")
    nyq = 1 - p["sampled_field_power"] / p["aperture_power_area_mm2"]
    caption = (
        f"Far field of a circular aperture, D = {p['D_mm']} mm, λ = {p['lambda_um']} µm, z = {p['z_mm']:g} mm "
        f"(source Fresnel number (D/2)²/(λz) = {p['source_fresnel_number_r2_over_lambda_z']:.2f}), sampled with D/Δx = 64 on a "
        f"{p['n']}² grid (exact area-weighted edge cells) and carried to the far field by the chirp-z Fraunhofer zoom "
        f"(fraunhofer_zoom; 401² samples over ±4 first-zero radii, r₁ = {p['first_zero_radius_mm']:.3f} mm; 801² for the encircled energy). "
        f"(a) Simulated log₁₀ I/I₀, I₀ = (πD²/4λz)² the analytic peak. (b) Simulated profile along +x (markers, every second sample) "
        f"against [2J₁(v)/v]² (line); the first three zeros v = 3.8317, 7.0156, 10.1735 are marked. Worst complex error along the "
        f"profile {sci(e['profile_worst_complex_err_rel_peak'])} of the peak amplitude ({sci(e['image_worst_complex_err_rel_peak'])} over the whole 401² window); "
        f"simulated intensity at the first zero {sci(e['intensity_at_dark_rings'][0]['sim_intensity'])} I₀. "
        f"(c) Encircled energy, the simulated |U|²ΔA summed inside r (cells cut by the circle weighted by their exact covered area) "
        f"divided by the aperture area πD²/4, against 1 − J₀²(v) − J₁²(v): {e['ee_first_ring_sim']:.5f} vs {e['ee_first_ring_analytic']:.5f} at the first "
        f"dark ring (|Δ| = {sci(e['ee_first_ring_abs_err'])}), worst {sci(e['ee_worst_abs_err_0_to_4r1'])} out to v = 15.3. The sampled field carries "
        f"{100*nyq:.2f}% less power than πD²/4; that is the sub-cell projection error of the sampled edge, which the aperture's ledger books as subcell_projection. "
        f"Sources: [Goodman], [Born & Wolf], [DLMF] (ch. 10)."
    )
    return {
        "id": "F1",
        "title": "Airy pattern of a circular aperture",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Goodman", "Born & Wolf", "DLMF")],
        "datasets": ["f1_airy.json"],
        "numbers": e,
    }


# ---------------------------------------------------------------------------
# F2: Heurtley
# ---------------------------------------------------------------------------


def fig2(D: Data, outdir: Path) -> dict:
    m, a = D.load("f2_heurtley")
    p, e = m["parameters"], m["errors"]
    lam_mm, rad = 0.5e-3, p["a_mm"]
    fig = plt.figure(figsize=(WIDTH_IN, 3.35))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.125, top=0.855, wspace=0.3, hspace=0.12, width_ratios=[1.45, 1], height_ratios=[2.4, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    nf, sim = a["nf"], a["sim_re"] ** 2 + a["sim_im"] ** 2
    ax.plot(a["dense_nf"], a["dense_heurtley_int"], **REF, label="Heurtley, exact Rayleigh–Sommerfeld", zorder=2)
    ax.plot(a["dense_nf"], a["dense_paraxial_int"], color=ORANGE, lw=0.7, ls=(0, (3, 2)), label=r"paraxial (Fresnel) $4\sin^2(\pi N_F/2)$", zorder=3)
    ax.plot(nf, sim, **sim_style(ms=4.0), label=r"simulation: direct RS, $a/\Delta x = 256$", zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(23, 0.18)
    ax.set_ylim(-0.1, 6.15)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel(r"$|U(0,z)|^2 / |U_\mathrm{inc}|^2$")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", ncol=2, fontsize=6.3, borderaxespad=0.15, handletextpad=0.5, columnspacing=1.2)
    grid(ax, axis="y")
    top = ax.secondary_xaxis("top", functions=(reciprocal(rad**2 / lam_mm), reciprocal(rad**2 / lam_mm)))
    top.set_xlabel("z (mm)", fontsize=7.5)
    top.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    top.xaxis.set_major_formatter(ScalarFormatter())
    top.xaxis.set_minor_formatter(NullFormatter())
    top.tick_params(labelsize=7)
    top.spines["top"].set_visible(True)
    top.spines["top"].set_color(AXIS)
    label(ax, "a", dx=-36, y_fig=0.935)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.semilogy(a["dense_nf"], a["dense_paraxial_abs_err"], color=ORANGE, lw=0.8, ls=(0, (3, 2)), label="paraxial", zorder=2)
    ax2.semilogy(nf, a["abs_err"], **sim_style(ms=3.6), label="simulation", zorder=3)
    ax2.set_ylabel(r"$|U - U_\mathrm{H}|$")
    ax2.set_xlabel(r"Fresnel number $N_F = a^2/(\lambda z)$ (decreasing: $z$ increases to the right)")
    ax2.set_ylim(1e-7, 1.0)
    ax2.yaxis.set_major_locator(FixedLocator([1e-6, 1e-4, 1e-2]))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.set_xticks([20, 10, 5, 2, 1, 0.5, 0.2])
    ax2.set_xticklabels(["20", "10", "5", "2", "1", "0.5", "0.2"])
    ax2.legend(loc="upper right", fontsize=6.2, ncol=2, borderaxespad=0.1, handletextpad=0.4, columnspacing=1.0)
    grid(ax2, axis="y")
    label(ax2, "b", dx=-36, dy=-2)

    ax3 = fig.add_subplot(gs[:, 1])
    lv = a["conv_a_over_dx"]
    conv, rich = a["conv_abs_err"], a["rich_abs_err"]
    colors, markers = [BLUE, ORANGE, AQUA], ["o", "s", "^"]
    for i, n in enumerate(a["conv_nf"]):
        ax3.loglog(lv, conv[i], color=colors[i], lw=0.8, zorder=2)
        ax3.loglog(lv, conv[i], **sim_style(colors[i], markers[i], ms=4.6), zorder=3)
        ax3.loglog(lv[1:], rich[i], color=colors[i], lw=0.6, ls=(0, (2, 1.5)), zorder=2)
        ax3.loglog(lv[1:], rich[i], **sim_style(colors[i], markers[i], ms=4.6, filled=False), zorder=3)
    xg = np.array([60.0, 280.0])
    yg = 0.17 * (xg / 64.0) ** -2
    ax3.loglog(xg, yg, color=MUTED, lw=0.9, zorder=1)
    ax3.text(150, 0.17 * (150 / 64.0) ** -2 * 1.35, "slope −2", fontsize=6.5, color=INK2, rotation=-24, ha="center", va="bottom", rotation_mode="anchor")
    ax3.set_xticks(lv)
    ax3.set_xticklabels([f"{int(x)}" for x in lv])
    ax3.xaxis.set_minor_formatter(NullFormatter())
    ax3.set_xlim(26, 310)
    ax3.set_ylim(1e-6, 0.5)
    ax3.set_xlabel(r"samples per radius $a/\Delta x$")
    ax3.set_ylabel(r"on-axis $|U - U_\mathrm{H}|$")
    handles = [Line2D([], [], color=colors[i], lw=0.8, **{k: v for k, v in sim_style(colors[i], markers[i], ms=4.6).items() if k != "ls"}) for i in range(3)]
    labels = [rf"$N_F = {n:g}$" for n in a["conv_nf"]]
    handles += [
        Line2D([], [], color=INK2, lw=0.8, marker="o", ms=4.2, mfc=INK2, mec=SURFACE, mew=0.5),
        Line2D([], [], color=INK2, lw=0.6, ls=(0, (2, 1.5)), marker="o", ms=4.2, mfc=SURFACE, mec=INK2, mew=0.8),
    ]
    labels += ["direct RS", "Richardson (4U(Δx/2) − U(Δx))/3"]
    ax3.legend(handles, labels, loc="lower left", fontsize=6.3, ncol=1, borderaxespad=0.2, handletextpad=0.5)
    grid(ax3)
    label(ax3, "c", dx=-40, y_fig=0.935)

    files = save(fig, outdir, "F2_heurtley")
    orders = e["convergence"]
    order_txt = "; ".join(f"N_F = {o['nf']:g}: {min(o['error_ratio_per_halving']):.2f}–{max(o['error_ratio_per_halving']):.2f}" for o in orders)
    rich = np.array(e["richardson_abs_err"]).reshape(3, 3)
    n_under = sum(1 for w in m["warnings"] if "Undersampled" in w["warning"])
    caption = (
        f"On-axis field behind a circular aperture of radius a = {rad} mm in a unit plane wave, λ = 0.5 µm. Markers: direct Rayleigh–Sommerfeld "
        f"integration (RsDirect, FFT convolution of the first RS kernel [Shen 2006]) of the area-weighted sampled aperture; solid line: Heurtley's exact "
        f"on-axis solution U = e^(ikz) − z/√(z²+a²)·e^(ik√(z²+a²)) [Heurtley 1973]; dashed: the paraxial (Fresnel) result 4 sin²(πN_F/2) [Goodman]. "
        f"(a) |U(0,z)|² for 20 Fresnel numbers from 20 to 0.2 (z = 1–100 mm), a/Δx = 256 "
        f"(1024² window of 4a). (b) Complex error against Heurtley: the simulation's is {sci(e['points_worst_abs_err_a_dx_256'])} at N_F = 20, falling roughly as N_F² to "
        f"{sci(min(a['abs_err']))} at N_F = 0.2; the paraxial field e^(ikz)(1 − e^(iπN_F)) is off by {sci(a['dense_paraxial_abs_err'][0])} at N_F = 20 "
        f"(a phase error ≈ πλ²N_F³/(4a²)), so above N_F ≈ 1 the sampled exact integral lands closer to the exact solution than the Fresnel approximation does. (c) Convergence of the on-axis error with a/Δx ∈ {{32, 64, 128, 256}} (window 4a): the error falls by "
        f"{order_txt} per halving of Δx, i.e. second order, from the aperture's edge representation; Richardson extrapolation (open markers) reaches "
        f"{sci(rich[0, 2])}, {sci(rich[1, 2])} and {sci(rich[2, 2])} at N_F = 10, 5, 2.5. The {n_under} Undersampled warnings at a/Δx = 32 and 64 "
        f"(N_F = 10, 5) concern the kernel at the window's largest offset, i.e. off-axis points; the on-axis value uses offsets ≤ a, where the kernel "
        f"is resolved. RsDirect books power leaving its window as numerical (up to {100*e['ledger_worst_numerical_fraction']:.0f}% here), by design. "
        f"Sources: [Heurtley 1973], [Shen 2006], [Goodman]."
    )
    return {
        "id": "F2",
        "title": "Heurtley's exact on-axis field of a circular aperture",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Heurtley 1973", "Shen 2006", "Goodman")],
        "datasets": ["f2_heurtley.json"],
        "numbers": e,
    }


# ---------------------------------------------------------------------------
# F3: Fresnel square
# ---------------------------------------------------------------------------


def fig3(D: Data, outdir: Path) -> dict:
    m, a = D.load("f3_fresnel_square")
    p, e = m["parameters"], m["errors"]
    runs = {r["samples_across"]: r for r in e["runs"]}
    w = p["w_mm"]
    fig = plt.figure(figsize=(WIDTH_IN, 2.6))
    gs = fig.add_gridspec(1, 3, left=0.068, right=0.985, bottom=0.175, top=0.86, wspace=0.85, width_ratios=[1, 1.08, 1.08])

    ax = fig.add_subplot(gs[0])
    frame(ax)
    img, c = a["image_intensity"], a["image_coord_mm"]
    hp = 0.5 * (c[1] - c[0])
    im = ax.imshow(img, origin="lower", extent=[c[0] - hp, c[-1] + hp, c[0] - hp, c[-1] + hp], cmap=CMAP, vmin=0, vmax=float(img.max()))
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * 0.5 * w
    ax.plot(sq[:, 0], sq[:, 1], color="white", lw=0.6, alpha=0.75)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels(["−0.5", "", "0", "", "0.5"])
    ax.set_yticklabels(["−0.5", "", "0", "", "0.5"])
    colorbar(fig, ax, im, r"$I/I_\mathrm{inc}$")
    ax.set_title(f"z = {p['z_mm']:g} mm, $N_F$ = {p['fresnel_number_half_width']:g}", fontsize=7.5, color=INK2)
    label(ax, "a", dx=-32, y_fig=0.93)

    ax = fig.add_subplot(gs[1])
    x, s, an = a["m201_line_x_mm"], a["m201_line_sim"], a["m201_line_analytic"]
    sel = np.abs(x) <= 0.5 + 1e-9
    ax.plot(x[sel], an[sel], **REF, label="Fresnel integrals", zorder=2)
    idx = np.where(sel)[0][::3]
    ax.plot(x[idx], s[idx], **sim_style(ms=2.8), label="simulation (Fresnel IR)", zorder=3)
    for xe in (-0.5 * w, 0.5 * w):
        ax.axvline(xe, color=MUTED, lw=0.6, zorder=1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 2.75)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("x (mm), y = 0")
    ax.set_ylabel(r"$I/I_\mathrm{inc}$")
    ax.legend(loc="upper left", fontsize=6.5, ncol=1, borderaxespad=0.1, handletextpad=0.4)
    grid(ax, axis="y")
    label(ax, "b", dx=-36, y_fig=0.93)

    ax = fig.add_subplot(gs[2])
    for mm, col, mk in ((201, BLUE, "IR"), (101, ORANGE, "TF")):
        xs, er = a[f"m{mm}_line_x_mm"], a[f"m{mm}_line_complex_err"]
        meth = "IR" if runs[mm]["method"] == "FresnelIr" else "TF"
        ax.semilogy(xs, np.clip(er, 1e-12, None), color=col, lw=0.7, label=f"w/Δx = {mm}, Fresnel {meth}")
    for xe in (-0.5 * w, 0.5 * w):
        ax.axvline(xe, color=MUTED, lw=0.6, zorder=1)
    ax.set_xlim(a["m201_line_x_mm"][0], a["m201_line_x_mm"][-1])
    ax.set_ylim(3e-6, 4e-2)
    ax.set_xlabel("x (mm), y = 0 (full window)")
    ax.set_ylabel(r"$|U - U_\mathrm{Fresnel}| \,/\, \max|U_\mathrm{Fresnel}|$")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper center", fontsize=6.3, borderaxespad=0.1, handletextpad=0.5)
    grid(ax, axis="y")
    label(ax, "c", dx=-40, y_fig=0.93)

    files = save(fig, outdir, "F3_fresnel_square")
    r201, r101 = runs[201], runs[101]
    # The reports carry a Paraxial warning beside each WrapAround (I4).
    wraps = [w_ for w_ in m["warnings"] if "WrapAround" in w_["warning"]]
    n_wrap = len(wraps)
    caption = (
        f"Fresnel diffraction of a square aperture, w = {w} mm, λ = {p['lambda_um']} µm, z = {p['z_mm']:g} mm (N_F = (w/2)²/(λz) = {p['fresnel_number_half_width']:g}). "
        f"Simulation: unit plane wave through the area-weighted aperture, 201 samples across (Δx = w/201, n = {r201['n']}, window {r201['window_mm']:.2f} mm, "
        f"padded {r201['padded_window_mm']:.2f} mm), propagated by the Fresnel impulse response, the in-regime method beyond z_c = LΔx/λ = "
        f"{r201['critical_distance_mm']:.1f} mm [Voelz 2009]. Reference: U = e^(ikz)/i·I(x)I(y) with I(x) = [(C(α₂) − C(α₁)) + i(S(α₂) − S(α₁))]/√2 "
        f"[Goodman; DLMF ch. 7]. (a) Simulated irradiance (white: the geometric aperture). (b) Profile along y = 0, markers every third sample. "
        f"(c) Complex error along y = 0 over the full window: worst {sci(r201['worst_complex_err_rel_peak_line'])} of the peak on this line and "
        f"{sci(r201['worst_complex_err_rel_peak_2d'])} over the plane, reached at the window edge, where the convolution sum's phase step per cell is largest. "
        f"With 101 samples across, z_c = {r101['critical_distance_mm']:.1f} mm lies beyond z, so the transfer function is the in-regime method; it gives "
        f"{sci(r101['worst_complex_err_rel_peak_2d'])}, a ratio of {e['error_ratio_101_to_201_2d']:.2f} on halving Δx (second order). Both reports carry WrapAround "
        f"(edge fractions {sci(wraps[0]['warning']['WrapAround']['edge_fraction'])} and {sci(wraps[1]['warning']['WrapAround']['edge_fraction'])}); "
        f"the sharp edge diffracts up to the grid's Nyquist angle. Sources: [Goodman], [Voelz 2009], [DLMF]."
    )
    return {
        "id": "F3",
        "title": "Fresnel diffraction of a square aperture",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Goodman", "Voelz 2009", "DLMF")],
        "datasets": ["f3_fresnel_square.json"],
        "numbers": e,
        "warnings_count": n_wrap,
    }


# ---------------------------------------------------------------------------
# F4: from Fresnel to Fraunhofer
# ---------------------------------------------------------------------------


def fig4(D: Data, outdir: Path) -> dict:
    m, a = D.load("f4_fresnel_to_fraunhofer")
    cases = m["errors"]["cases"]
    w = m["parameters"]["w_mm"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, 4.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.83, wspace=0.28, hspace=0.62)
    for ax, c in zip(axes.flat, cases):
        t = f"c{c['index']}"
        x, s, an = a[f"{t}_x_mm"], a[f"{t}_sim"], a[f"{t}_analytic"]
        ax.plot(x, an, **REF, label="Fresnel integrals", zorder=2)
        step = max(1, len(x) // 85)
        ax.plot(x[::step], s[::step], **sim_style(ms=2.6), label="simulation", zorder=3)
        if c["index"] == len(cases) - 1:
            ax.plot(x, a[f"{t}_fraunhofer"], color=ORANGE, lw=0.9, ls=(0, (3, 1.6)), label=r"Fraunhofer sinc$^2$", zorder=4)
        for xe in (-0.5 * w, 0.5 * w):
            ax.axvline(xe, color=MUTED, lw=0.5, zorder=1)
        xd = c["display_half_range_mm"]
        ax.set_xlim(-xd, xd)
        ax.set_ylim(0, 1.12)
        meth = "IR" if c["method"] == "FresnelIr" else "TF"
        ax.set_title(f"$N_F$ = {c['nf']:g},  z = {c['z_mm']:g} mm", fontsize=7.8, pad=12)
        ax.text(
            0.5,
            1.015,
            f"Fresnel {meth} · w/Δx = {c['samples_across']} · err {tex(c['worst_complex_err_rel_peak_line'])}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=INK2,
        )
        ax.set_xlabel("x (mm)", labelpad=1.5)
        grid(ax, axis="y")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$I \,/\, \max I_\mathrm{Fresnel}$")
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    for i, ax in enumerate(axes.flat):
        label(ax, "abcdef"[i], dx=-12 if i % 3 else -30, dy=18)
    handles = [
        Line2D([], [], **REF),
        Line2D([], [], **sim_style(ms=3.4)),
        Line2D([], [], color=ORANGE, lw=0.9, ls=(0, (3, 1.6))),
        Line2D([], [], color=MUTED, lw=0.6),
    ]
    labels = ["Fresnel integrals (analytic)", "simulation", r"Fraunhofer sinc$^2$ (panel f)", r"geometric edges $\pm w/2$"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=4, fontsize=6.8, handlelength=2.0, columnspacing=1.4, handletextpad=0.5)
    files = save(fig, outdir, "F4_fresnel_to_fraunhofer")
    last = cases[-1]
    errs = ", ".join(f"{sci(c['worst_complex_err_rel_peak_line'])}" for c in cases)
    n_wrap = sum(1 for w_ in m["warnings"] if "WrapAround" in w_["warning"])
    lw = [c["ledger"]["left_window"] / c["ledger"]["launched"] for c in cases]
    caption = (
        f"The square aperture of F3 (w = {w} mm, λ = 0.5 µm) at Fresnel numbers N_F = (w/2)²/(λz) = 10, 2.5, 1, 0.4, 0.1 and 0.02 (z = 12.5 mm to 6.25 m). "
        f"Each case uses a 2048² grid whose window covers the displayed range (geometric half-width plus 2.5 λz/w, at least 0.4 mm, with a 10% margin) "
        f"and as many samples across the aperture as that allows (w/Δx = {', '.join(str(c['samples_across']) for c in cases)}); with these windows every z lies "
        f"beyond the critical distance (z/z_c = {', '.join(f'{c['z_over_zc']:.0f}' for c in cases)}), so the Fresnel impulse response is the in-regime method in all six "
        f"[Voelz 2009]. Markers: simulated irradiance along y = 0 (about 85 of the line's samples per panel); solid lines: Goodman's Fresnel-integral solution, both normalized by the analytic maximum on the line; "
        f"grey lines mark the geometric edges ±w/2. Worst complex error relative to the analytic peak on each line: {errs}. The largest, at N_F = 0.02 with only "
        f"{last['samples_across']} samples across, is the discrete transform of the sampled aperture (a Dirichlet kernel) departing from the continuous sinc toward "
        f"the side lobes. Panel (f) also shows the Fraunhofer limit (w²/λz)² sinc²(wx/λz) (dashed), which differs from the Fresnel solution by at most "
        f"{sci(last['fraunhofer_vs_fresnel_max_intensity_diff_rel_peak'])} of the peak here. All six reports carry WrapAround ({n_wrap} warnings); for the impulse response "
        f"the cropped window is a linear convolution and unaffected, as the errors show. The ledger books {sci(min(lw))} to {sci(max(lw))} of the launched power "
        f"as having left the window: the cropped margin plus what the truncated kernel never carries, a share that grows as its reach (L/2)/(λz) in spatial "
        f"frequency shrinks. The impulse response books all it does not carry that way, so its numerical term is 0 by construction "
        f"(here {sci(max(abs(c['ledger_numerical_fraction']) for c in cases))}). "
        f"Sources: [Goodman], [Voelz 2009], [DLMF]."
    )
    return {
        "id": "F4",
        "title": "From Fresnel to Fraunhofer: a square aperture",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Goodman", "Voelz 2009", "DLMF")],
        "datasets": ["f4_fresnel_to_fraunhofer.json"],
        "numbers": m["errors"],
    }


# ---------------------------------------------------------------------------
# F5: Gaussian beam through a thin lens
# ---------------------------------------------------------------------------


def fig5(D: Data, outdir: Path) -> dict:
    m, a = D.load("f5_gaussian_lens")
    p, e = m["parameters"], m["errors"]
    z, x = a["z_mm"], a["x_mm"]
    sl = a["slice_intensity"]
    norm = sl / sl.max(axis=1, keepdims=True)
    s, zw = p["lens_z_mm"], p["waist_z_mm"]
    fig = plt.figure(figsize=(WIDTH_IN, 5.3))
    gs = fig.add_gridspec(3, 1, left=0.1, right=0.895, bottom=0.08, top=0.955, hspace=0.12, height_ratios=[2.1, 1.45, 1.05])
    ax = fig.add_subplot(gs[0])
    frame(ax)
    sel = np.abs(x) <= 1.0 + 1e-9
    im = ax.pcolormesh(z, x[sel], norm[:, sel].T, cmap=CMAP, vmin=0, vmax=1, shading="nearest", rasterized=True)
    dz, dw = a["dense_z_mm"], a["dense_w_mm"]
    ax.plot(dz, dw, color="white", lw=0.8, label=r"analytic $\pm w(z)$")
    ax.plot(dz, -dw, color="white", lw=0.8)
    ax.axvline(s, color="white", lw=0.6, alpha=0.8)
    ax.text(s + 3, 0.86, f"thin lens, f = {p['f_mm']:g} mm", color="white", fontsize=6.8, va="center")
    ax.annotate(
        f"Self waist, z = {zw:.2f} mm\n$w_0''$ = {1000*p['w0pp_mm']:.1f} µm",
        xy=(zw, 0.0),
        xytext=(zw + 20, -0.72),
        color="white",
        fontsize=6.8,
        arrowprops=dict(arrowstyle="-", color="white", lw=0.6, shrinkA=0, shrinkB=1),
    )
    ax.plot([zw], [0], marker="+", color="white", ms=6, mew=0.9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim(0, z[-1])
    ax.set_ylabel("x (mm), y = 0")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=6.5, labelcolor="white", borderaxespad=0.4)
    colorbar(fig, ax, im, r"$I(x,z)\,/\,\max_x I$", gap=0.012, width=0.016)
    label(ax, "a", dx=-38)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.plot(dz, dw, **REF, label="Kogelnik–Li, then Self's transformed beam", zorder=2)
    k = slice(None, None, 3)
    ax2.plot(z[k], a["w_sim_mm"][k], **sim_style(ms=3.2), label=r"simulation, $2\sqrt{\langle x^2\rangle}$", zorder=3)
    for zz in (s, zw):
        ax2.axvline(zz, color=MUTED, lw=0.6, zorder=1)
    ax2.set_ylabel("beam radius w (mm)")
    ax2.set_ylim(0, 0.56)
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc="lower left", fontsize=6.5, borderaxespad=0.2)
    grid(ax2, axis="y")
    label(ax2, "b", dx=-38)

    ax3 = fig.add_subplot(gs[2], sharex=ax)
    ax3.plot(z, 1e6 * a["w_rel_residual"], **sim_style(ms=2.8), label="angular spectrum (exact), band-limited", zorder=3)
    ax3.plot(z, 1e6 * a["w_rel_residual_paraxial"], **sim_style(ORANGE, "s", ms=2.6), label="Fresnel TF/IR (paraxial control)", zorder=2)
    ax3.axhline(0, color=AXIS, lw=0.6, zorder=1)
    for zz in (s, zw):
        ax3.axvline(zz, color=MUTED, lw=0.6, zorder=1)
    ax3.set_ylabel(r"$(w - w_\mathrm{an})/w_\mathrm{an}$" + "\n" + r"($10^{-6}$)")
    ax3.set_xlabel("z (mm) from the input waist")
    ax3.set_ylim(-14, 14)
    ax3.legend(loc="lower left", fontsize=6.3, ncol=2, borderaxespad=0.15)
    grid(ax3, axis="y")
    label(ax3, "c", dx=-38)

    files = save(fig, outdir, "F5_gaussian_lens")
    caption = (
        f"A Gaussian beam, w₀ = {p['w0_mm']} mm at its waist (z_R = {p['zR_mm']:.1f} mm), λ = {p['lambda_um']} µm, travels s = {p['s_mm']:g} mm to an ideal thin lens "
        f"t = exp(−ikr²/2f), f = {p['f_mm']:g} mm (apply_thin_lens), and on past the new waist. Simulation: 512² samples at 8 µm, band-limited angular spectrum "
        f"[Matsushima 2009], each plane a single step from the input waist or from the lens plane. (a) Irradiance along y = 0, each column normalized to its peak, "
        f"with the analytic ±w(z) (white): Kogelnik–Li w(z) = w₀√(1 + (z/z_R)²) before the lens [Kogelnik 1966]; after it the beam Self's formulas predict, "
        f"1/(s + z_R²/(s − f)) + 1/s″ = 1/f and w₀″ = w₀/√((1 − s/f)² + (z_R/f)²), giving s″ = {p['s2_mm']:.3f} mm and w₀″ = {1000*p['w0pp_mm']:.2f} µm [Self 1983] "
        f"(identical to the q-parameter transformation to {sci(e['self_vs_q_parameter_s2_rel'])}). (b) Second-moment radius 2√⟨x²⟩ of the full 2D irradiance "
        f"(markers, every third plane) against w(z). (c) Relative residual: {sci(e['radius_worst_rel_residual_before_lens'])} before the lens, "
        f"{sci(e['radius_rel_residual_at_predicted_waist'])} at the predicted waist, up to {sci(e['radius_worst_rel_residual_after_lens'])} after it. "
        f"A parabola fitted to w²(z) after the lens puts the simulated waist {1000*abs(e['fitted_waist_position_minus_self_mm']):.2f} µm "
        f"{'before' if e['fitted_waist_position_minus_self_mm'] < 0 else 'beyond'} Self's s″, radius within "
        f"{sci(abs(e['fitted_waist_radius_rel_err']))}. The same planes propagated with the paraxial Fresnel methods in their regimes (transfer function to "
        f"z_c = {e['paraxial_control_critical_distance_mm']:.1f} mm, impulse response beyond) agree with Self to {sci(e['paraxial_control_worst_rel_residual'])} "
        f"(waist offset {1e6*e['paraxial_control_fitted_waist_position_minus_self_mm']:.2f} nm). The angular spectrum's residual is therefore the non-paraxial "
        f"behaviour of a parabolic lens phase propagated exactly, which focuses slightly early; the shift is comparable to w_L²/(2f) = "
        f"{1000*e['lens_plane_w2_over_2f_mm']:.2f} µm (w_L the radius at the lens), the scale of that focal shift. The band limit removed at most {sci(e['band_limit_removed_max_fraction'])} of the power; the ledger closes to "
        f"{sci(e['ledger_worst_numerical_fraction'])}. Sources: [Kogelnik 1966], [Self 1983], [Matsushima 2009], [Goodman] (thin lens), [Voelz 2009] (control)."
    )
    return {
        "id": "F5",
        "title": "Gaussian beam through a thin lens",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Kogelnik 1966", "Self 1983", "Matsushima 2009", "Goodman", "Voelz 2009")],
        "datasets": ["f5_gaussian_lens.json"],
        "numbers": e,
    }


# ---------------------------------------------------------------------------
# F6: Talbot carpet
# ---------------------------------------------------------------------------


def fig6(D: Data, outdir: Path) -> dict:
    m, a = D.load("f6_talbot")
    p, e = m["parameters"], m["errors"]
    z, x, cp = a["z_mm"], a["x_mm"], a["carpet"]
    pum = p["p_mm"] * 1000
    xu = x * 1000
    sel = (xu >= -pum - 1e-9) & (xu < pum - 1e-9)
    zt, ztp = p["zT_exact_mm"], p["zT_paraxial_mm"]
    fig = plt.figure(figsize=(WIDTH_IN, 4.0))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.9, bottom=0.11, top=0.9, hspace=0.55, wspace=0.18, height_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0, :])
    frame(ax)
    dz = z[1] - z[0]
    dxu = xu[1] - xu[0]
    im = ax.imshow(
        cp[:, sel].T,
        origin="lower",
        aspect="auto",
        extent=[z[0] - dz / 2, z[-1] + dz / 2, xu[sel][0] - dxu / 2, xu[sel][-1] + dxu / 2],
        cmap=CMAP,
        vmin=0,
        vmax=float(cp.max()),
        interpolation="antialiased",
        interpolation_stage="data",
    )
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (µm)")
    ax.set_yticks([-50, -25, 0, 25, 50])
    top = ax.secondary_xaxis("top")
    top.set_xticks([ztp / 4, ztp / 2, ztp])
    top.set_xticklabels([r"$z_T/4$", r"$z_T/2$", r"$z_T$"])
    top.tick_params(length=4, width=0.8, color=INK2, labelsize=7.5)
    top.spines["top"].set_visible(True)
    top.spines["top"].set_color(AXIS)
    for zz in (ztp / 4, ztp / 2, ztp):
        ax.plot([zz, zz], [pum * 0.93, pum * 1.0], color="white", lw=1.0, clip_on=True)
        ax.plot([zz, zz], [-pum * 1.0, -pum * 0.93], color="white", lw=1.0, clip_on=True)
    colorbar(fig, ax, im, r"$I/I_\mathrm{inc}$", gap=0.012, width=0.016)
    label(ax, "a", dx=-38, dy=12)

    period = (xu >= -pum - 1e-9) & (xu < pum - 1e-9)
    xs = xu[period]
    i_in = a["profile_input"][period]
    nx = len(a["x_mm"])
    shift = np.roll(a["profile_input"], int(p["samples_per_period"]) // 2)[period]
    for j, (key, ref, title, lab) in enumerate(
        [
            ("profile_zt_exact", i_in, r"at the exact $z_T$", "grating |t|², the paraxial revival"),
            ("profile_zt_half", shift, r"at $z_T/2$", "grating |t|² shifted by p/2"),
        ]
    ):
        axp = fig.add_subplot(gs[1, j])
        axp.plot(xs, ref, color=INK, lw=0.9, drawstyle="steps-mid", label=lab, zorder=2)
        axp.plot(xs, a[key][period], color=BLUE, lw=0.7, zorder=3)
        axp.plot(xs, a[key][period], **sim_style(ms=2.4), label="simulation", zorder=4)
        axp.set_xlim(-pum, pum)
        axp.set_ylim(0, 2.1)
        axp.set_xticks([-50, -25, 0, 25, 50])
        axp.set_xlabel("x (µm)")
        if j == 0:
            axp.set_ylabel(r"$I/I_\mathrm{inc}$")
        else:
            axp.tick_params(labelleft=False)
        axp.set_title(title, fontsize=7.5, color=INK2)
        axp.legend(loc="upper center", fontsize=6.2, ncol=2, borderaxespad=0.1, handletextpad=0.4, columnspacing=1.0)
        grid(axp, axis="y")
        label(axp, "bc"[j], dx=-30 if j == 0 else -12, dy=4)
    del nx
    files = save(fig, outdir, "F6_talbot")
    tc = talbot_check(D)
    caption = (
        f"Talbot self-imaging behind a 50%-duty binary amplitude grating, period p = {pum:g} µm, λ = {p['lambda_um']} µm (p/λ = {p['p_mm']/(p['lambda_um']/1000):g}), "
        f"{p['samples_per_period']} samples per period (edge cells weighted by their open fraction), {p['periods_in_window']} periods in a periodic window, carried "
        f"by the angular spectrum without band limit or padding (AsmOptions {{band_limit: false, pad: 1}}), which is exact for a field periodic in its window "
        f"[Goodman]. (a) Simulated irradiance from z = 0 to 1.1 z_T, z_T = 2p²/λ = {ztp:g} mm (paraxial; ticks at z_T/4, z_T/2, z_T), two of the four periods shown. "
        f"(b) The profile at the exact z_T = λ/(1 − √(1 − λ²/p²)) = {zt:.6f} mm against the grating, and (c) at z_T/2 against the grating shifted by p/2 [Wen 2013]. "
        f"For a cosine grating (three plane waves) the revival at the exact z_T is exact: worst complex error {sci(e['cosine_revival_worst_complex_err'])} "
        f"(half-period shift at z_T/2: {sci(e['cosine_half_talbot_shift_worst_complex_err'])}), round-off on an absolute phase kz_T ≈ 1.3 × 10⁵ rad. "
        f"The binary grating's revival is not perfect: its orders m rephase at z_T only paraxially. Each order's residual phase at the exact z_T is "
        f"about π(λ/p)²(m⁴ − m²)/2, 0.011 rad for m = 3, 0.37 rad for m = 7 and above 1 rad from m = 9 (the sampling carries |m| ≤ {p['highest_order_sampled']}). "
        f"The edges therefore ring: the worst irradiance error at z_T is {100*e['binary_revival_exact_zt_worst_intensity_err_rel_max']:.0f}% of the maximum "
        f"(rms {100*e['binary_revival_exact_zt_rms_intensity_err_rel_max']:.0f}%), and the carpet is not perfectly periodic. An independent NumPy sum of the same orders reproduces "
        f"the z_T profile to {sci(tc['talbot_numpy_exact_vs_crate_max_intensity_diff'])}, and the paraxial transfer function at 2p²/λ revives it to "
        f"{sci(tc['talbot_paraxial_model_revival_worst_intensity_err'])}, so the ringing is wholly non-paraxial. These steps run unpadded on purpose, so they raise no WrapAround "
        f"({len(m['warnings'])} warnings): power at the window's edge is the next period, which is the physics being modelled. "
        f"Sources: [Wen 2013], [Goodman]."
    )
    return {
        "id": "F6",
        "title": "Talbot carpet of a binary amplitude grating",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Wen 2013", "Goodman")],
        "datasets": ["f6_talbot.json"],
        "numbers": e,
    }


# ---------------------------------------------------------------------------
# F7: sampling regimes and the ledger
# ---------------------------------------------------------------------------


def fig7(D: Data, outdir: Path) -> dict:
    m, a = D.load("f7_regimes_ledger")
    p, e = m["parameters"], m["errors"]
    fig = plt.figure(figsize=(WIDTH_IN, 3.25))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.135, top=0.83, wspace=0.3, hspace=0.12, width_ratios=[1, 1.05], height_ratios=[2.3, 1])
    ax = fig.add_subplot(gs[:, 0])
    r = a["a_z_over_zc"]
    ax.loglog(r, a["a_err_tf"], color=BLUE, lw=0.8, zorder=2)
    ax.loglog(r, a["a_err_tf"], **sim_style(BLUE, "o", ms=3.4), label="Fresnel transfer function", zorder=3)
    ax.loglog(r, a["a_err_ir"], color=ORANGE, lw=0.8, zorder=2)
    ax.loglog(r, a["a_err_ir"], **sim_style(ORANGE, "s", ms=3.2), label="Fresnel impulse response", zorder=3)
    ax.axvline(1.0, color=MUTED, lw=0.7, zorder=1)
    ax.text(1.08, 3e-7, rf"$z_c = L\Delta x/\lambda$" + f"\n= {p['a']['critical_distance_mm']:g} mm", fontsize=6.5, color=INK2, va="center")
    ax.text(0.93, 1e-11, "← TF in regime", fontsize=6.5, color=INK2, ha="right")
    ax.text(1.08, 1e-11, "IR in regime →", fontsize=6.5, color=INK2, ha="left")
    ax.set_xlim(0.04, 25)
    ax.set_ylim(1e-17, 30)
    ax.set_xlabel(r"$z / z_c$")
    ax.set_ylabel(r"$\max|U - U_\mathrm{Gauss}| \,/\, |U_\mathrm{Gauss}(0,0,z)|$")
    ax.yaxis.set_major_locator(FixedLocator([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1.0]))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=40))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper center", bbox_to_anchor=(0.6, 1.0), fontsize=6.5, borderaxespad=0.2)
    grid(ax, axis="y")
    zc = p["a"]["critical_distance_mm"]
    top = ax.secondary_xaxis("top", functions=(lambda q: np.asarray(q, float) * zc, lambda zz: np.asarray(zz, float) / zc))
    top.set_xlabel("z (mm)", fontsize=7.5)
    top.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    top.xaxis.set_major_formatter(ScalarFormatter())
    top.xaxis.set_minor_formatter(NullFormatter())
    top.tick_params(labelsize=7)
    top.spines["top"].set_visible(True)
    top.spines["top"].set_color(AXIS)
    label(ax, "a", dx=-40, dy=14)

    ax2 = fig.add_subplot(gs[0, 1])
    zb = a["b_z_mm"]
    parts = [a["b_detected"], a["b_band_limit_removed"], a["b_left_window"]]
    names = ["detected", "band-limit removed", "left the window"]
    ax2.stackplot(zb, *parts, colors=[BLUE, ORANGE, AQUA], labels=names, edgecolor=SURFACE, linewidth=0.8, zorder=2)
    ax2.set_xscale("log")
    ax2.set_xlim(zb[0], zb[-1])
    ax2.set_ylim(0, 1.0)
    ax2.axvline(p["b"]["critical_distance_mm"], color=INK2, lw=0.6, zorder=3)
    ax2.text(p["b"]["critical_distance_mm"] * 0.88, 0.08, r"$z_c$", fontsize=6.8, color="white", ha="right")
    ax2.set_ylabel("fraction of launched power")
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.2, borderaxespad=0.2, handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
    label(ax2, "b", dx=-38, dy=16)

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.plot(zb, 1e14 * a["b_numerical"], **sim_style(INK2, "o", ms=2.8), zorder=3)
    ax3.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax3.axvline(p["b"]["critical_distance_mm"], color=MUTED, lw=0.6, zorder=1)
    ax3.set_ylim(-5, 5)
    ax3.set_ylabel("numerical\n" + r"($10^{-14}$)")
    ax3.set_xlabel("z (mm)")
    grid(ax3, axis="y")
    label(ax3, "c", dx=-38, dy=-2)

    files = save(fig, outdir, "F7_regimes_ledger")
    regime_txt = (
        "Every OutsideRegime warning the reports carry falls on a method's wrong side of z_c, and none on its right side. "
        if e["a_outside_regime_flags_agree_with_zc"]
        else "WARNING: the OutsideRegime flags do not match z_c. "
    )
    caption = (
        f"(a) The two Fresnel propagators [Voelz 2009] against the analytic Gaussian beam (w₀ = {p['a']['w0_mm']} mm, λ = {p['a']['lambda_um']} µm, "
        f"z_R = {p['a']['zR_mm']:.1f} mm), an exact solution of the Fresnel equation [Kogelnik 1966], on a {p['a']['n']}² grid at {1000*p['a']['dx_mm']:g} µm padded to "
        f"L = {p['a']['padded_window_mm']:g} mm; markers are the worst complex error over the window relative to the analytic on-axis amplitude, for z/z_c from "
        f"0.05 to 20 with z_c = LΔx/λ = {e['a_critical_distance_mm']:g} mm. Each method is accurate on its own side: the transfer function to "
        f"{sci(e['a_tf_worst_for_z_le_zc'])} for z ≤ z_c and the impulse response to {sci(e['a_ir_worst_for_z_ge_zc'])} for z ≥ z_c (round-off; both ≤ "
        f"{sci(max(e['a_tf_at_zc'], e['a_ir_at_zc']))} at z_c). Out of regime the transfer function's error grows to {sci(e['a_tf_at_20zc'])} at 20 z_c "
        f"(the spreading beam wraps around the periodic window) and the impulse response's reaches {sci(e['a_ir_at_0p05zc'])} at 0.05 z_c (its sampled "
        f"chirp aliases). For this beam, which fills only part of the window, both methods reach round-off over about 0.5–7 z_c; z_c is the sufficient "
        f"condition for a field that fills the window. "
        f"{regime_txt}"
        f"(b) The power ledger of a sharp 0.4 mm square aperture (128² at 10 µm, padded to {p['b']['padded_window_mm']:g} mm) carried 1 mm to 3 m by the "
        f"band-limited angular spectrum [Matsushima 2009]: fractions of the launched power detected in the window, removed by the band limit (from just "
        f"beyond z_c = {e['b_critical_distance_mm']:g} mm; {e['b_band_limited_warnings']} of 49 steps carry BandLimited) and cropped as having left the window. "
        f"At 3 m: {100*e['b_detected_at_3000mm']:.1f}% detected, {100*e['b_band_limit_removed_at_3000mm']:.1f}% band-limited, {100*e['b_left_window_at_3000mm']:.1f}% "
        f"left; evanescent power is {sci(e['b_max_evanescent_fraction'])}, since the grid's Nyquist frequency is below 1/λ. (c) The unexplained remainder the ledger "
        f"books as numerical: at most {sci(e['b_max_abs_numerical_fraction'])} of the launched power, so the ledger closes to round-off. "
        f"Sources: [Voelz 2009], [Matsushima 2009], [Kogelnik 1966], [Goodman]."
    )
    return {
        "id": "F7",
        "title": "Sampling regimes and the power ledger",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Voelz 2009", "Matsushima 2009", "Kogelnik 1966", "Goodman")],
        "datasets": ["f7_regimes_ledger.json"],
        "numbers": e,
    }


# ---------------------------------------------------------------------------
# F8: the DMD
# ---------------------------------------------------------------------------


def fig8(D: Data, outdir: Path) -> dict:
    from matplotlib.colors import LinearSegmentedColormap

    m, a = D.load("f8_dmd")
    p, e = m["parameters"], m["errors"]
    fig = plt.figure(figsize=(WIDTH_IN, 6.0))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.975, bottom=0.07, top=0.875, hspace=0.42, wspace=0.42, width_ratios=[1, 1.1], height_ratios=[1.2, 1])

    # (a) the order map
    ax = fig.add_subplot(gs[0, 0])
    frame(ax)
    share = a["a_share"]
    floor = 1e-6
    keep = share >= floor
    lg = np.log10(share[keep])
    order = np.argsort(lg)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    sizes = 2.0 + 30.0 * ((lg - np.log10(floor)) / -np.log10(floor)) ** 2
    sc = ax.scatter(a["a_alpha"][keep][order], a["a_beta"][keep][order], c=lg[order], s=sizes[order], cmap=cmap, vmin=-6, vmax=0,
                    edgecolors=SURFACE, linewidths=0.3, zorder=3)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=INK2, lw=0.7, zorder=2)
    na = e["a_projection_na"]
    ax.plot(na * np.cos(th), na * np.sin(th), color=ORANGE, lw=1.0, zorder=4, label=f"projection cone, f/{p['a']['projection_f_number']:g}")
    ec = e["a_envelope_centre"]
    ax.plot([ec[0]], [ec[1]], marker="+", color=INK, ms=9, mew=1.2, ls="none", zorder=5, label="mirror-envelope centre")
    zo = e["a_zero_order_direction"]
    ax.plot([zo[0]], [zo[1]], marker="x", color=INK, ms=6, mew=1.0, ls="none", zorder=5, label="order (0,0): flat-array specular")
    br = e["a_brightest"]
    ax.annotate(f"brightest ({br['m']},{br['n']})\n{100*br['share']:.0f}% of reflected", xy=(br["alpha"], br["beta"]), xytext=(0.28, -0.62),
                fontsize=6.3, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=3))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel(r"direction cosine $\alpha$")
    ax.set_ylabel(r"direction cosine $\beta$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), fontsize=6.0, ncol=2, borderaxespad=0.0, handletextpad=0.4, markerscale=0.8, columnspacing=1.0)
    cb = colorbar(fig, ax, sc, r"$\log_{10}$ share of reflected power", extend="min")
    label(ax, "a", dx=-34, y_fig=0.95)

    # (b) blazed number
    axc = fig.add_subplot(gs[0, 1])
    ac = a["c_alpha_deg"]
    axc.plot(ac, a["c_mu_7p6"], color=BLUE, lw=1.1, label=r"$\mu$, 7.6 µm", zorder=2)
    axc.plot(ac, a["c_mu_10p8"], color=ORANGE, lw=1.1, label=r"$\mu$, 10.8 µm", zorder=2)
    aco = a["c_alpha_coarse_deg"]
    sh = a["c_brightest_share"]
    axc.plot(aco, sh[0], **sim_style(BLUE, "s", ms=3.0), label="brightest-order share, 7.6 µm", zorder=3)
    axc.plot(aco, sh[1], **sim_style(ORANGE, "s", ms=3.0), label="brightest-order share, 10.8 µm", zorder=3)
    pubs = e["c_published_vs_model"]
    axc.plot([q["alpha_deg"] for q in pubs], [q["published"] for q in pubs], ls="none", marker="D", ms=5.5, mfc="none", mec=INK, mew=1.0,
             label="Popoff et al. 2024, Fig. 5", zorder=4)
    axc.set_xlim(-45, 45)
    axc.set_ylim(-0.02, 1.08)
    axc.set_xticks([-45, -30, -15, 0, 15, 30, 45])
    axc.set_xlabel(r"incidence $\alpha'$ across the hinge (deg)")
    axc.set_ylabel(r"blazed number $\mu$ / share")
    axc.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=2, fontsize=6.0, borderaxespad=0.0, handletextpad=0.4, columnspacing=1.0)
    grid(axc)
    label(axc, "b", dx=-34, y_fig=0.95)

    # (c) projector efficiency
    axb = fig.add_subplot(gs[1, :])
    lam = a["b_lambda_nm"]
    eff = a["b_efficiency"]
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    rows = e["b_rows"]
    # Photopic averages: one slot per device in each block, so equal values
    # (TI gives 89% for two devices) never hide each other.
    for i, r in enumerate(rows):
        axb.plot(lam, 100 * eff[i], color=cols[i], lw=1.3, label=r["device"], zorder=3)
        x0 = 690 + 7.0 * i
        axb.plot([x0, x0 + 5.6], [100 * r["ti_photopic"]] * 2, color=cols[i], lw=1.6, ls=(0, (1.8, 0.9)), solid_capstyle="butt", zorder=3)
        x1 = 727 + 7.0 * i
        axb.plot([x1, x1 + 5.6], [100 * r["model_photopic"]] * 2, color=cols[i], lw=2.0, solid_capstyle="butt", zorder=3)
    axb.axvline(685, color=AXIS, lw=0.6)
    axb.axvline(724.3, color=GRID, lw=0.6)
    axb.text(707, 95.6, "TI (dashed)", ha="center", fontsize=6.3, color=INK2)
    axb.text(744, 95.6, "this model", ha="center", fontsize=6.3, color=INK2)
    axb.text(724.3, 64.6, "photopic averages", ha="center", fontsize=6.3, color=INK2)
    axb.set_xlim(415, 764)
    axb.set_ylim(63.5, 97)
    axb.set_xticks([420, 460, 500, 540, 580, 620, 660, 680])
    axb.set_xlabel("wavelength (nm)")
    axb.set_ylabel("collected share of\nreflected power (%)")
    axb.legend(loc="lower left", ncol=3, fontsize=6.3, borderaxespad=0.2, handlelength=1.8, columnspacing=1.2)
    grid(axb, axis="y")
    label(axb, "c", dx=-40, dy=4)
    del cb

    files = save(fig, outdir, "F8_dmd")
    deltas = ", ".join(f"{r['device'].split(',')[0]} {r['delta_pp']:+.1f}" for r in rows)
    ours = ", ".join(f"{100*r['model_photopic']:.1f}" for r in rows)
    pubtxt = "; ".join(f"{q['pitch_um']:g} µm at {q['alpha_deg']:g}°: {q['model']:.3f} (published {q['published']:.2f})" for q in pubs)
    caption = (
        f"(a) The propagating diffraction orders of an all-ON DMD (7.56 µm pitch, 12° tilt about the diagonal, linear mirror fill √0.94 = "
        f"{p['a']['mirror_fill']:.3f}) at 633 nm, lit along the direction its ON mirrors send onto the normal (on_axis_illumination, "
        f"{e['a_incidence_deg']:.1f}° incidence), from the closed-form unit-cell Fourier series (unit_cell_orders) with Harvey's non-paraxial "
        f"renormalization [Harvey 2011]. Each of the {e['a_orders']} orders sits on the grid t_in + λ(m, n)/p in direction cosines [Harvey 1998]; "
        f"marker colour and size give its share of the reflected power (log scale; shares below 10⁻⁶ omitted). The mirror envelope is centred on "
        f"the normal (+); the lattice is offset from it by the incident direction (×, the flat-array specular order). At this wavelength the device "
        f"is far from blazed: the brightest order ({e['a_brightest']['m']},{e['a_brightest']['n']}) carries {100*e['a_brightest']['share']:.0f}% and "
        f"the f/2.4 projection cone (orange) collects {100*e['a_share_in_projection_cone']:.1f}% of this single plane wave's light. "
        f"(b) Popoff's blazed number μ (Eq. 2) against incidence across the hinge, "
        f"633 nm, 12° tilt (lines), with the four values read from Popoff et al. 2024 Fig. 5 (diamonds): {pubtxt}; worst |Δ| "
        f"{e['c_worst_abs_diff_published']:.3f}. Squares: the brightest order's share of the reflected power from the order model (fill 0.95), "
        f"an independent check that tracks μ: near 0.9 where μ ≈ 1, near 0.2 where μ ≈ 0. "
        f"(c) Collected share of the reflected power against wavelength for five TI devices (illumination and projection cones matched, 64 plane "
        f"waves per cone, combined incoherently: TI's §4.2 method), mirror fill √(TI on-state fill). Right: photopic averages over a flat 420–680 nm "
        f"spectrum weighted by V(λ), TI DLPA083B Table 6-1 dashed, this model solid ({ours}%). TI's values are a vendor simulation, not a measurement; "
        f"agreement shows the same model, not the same device. Δ (model − TI, percentage points): {deltas}; the 5.4 µm device at f/2.4 is "
        f"{e['b_5p4um_f2p4']['delta_pp']:+.1f} pp, the one row outside ±3 pp. Scalar thin-element model; at 633 nm a 7.56 µm pitch is "
        f"{7.56/0.633:.1f}λ, below the 14λ at which the crate's ScalarLimit validity flag warns that scalar theory may be off by more than ±5%. "
        f"Sources: [Popoff 2024], [TI DLPA083B], [Harvey 1998], [Harvey 2011]."
    )
    return {
        "id": "F8",
        "title": "The DMD as a blazed grating, and its projector efficiency",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Popoff 2024", "TI DLPA083B", "Harvey 1998", "Harvey 2011")],
        "datasets": ["f8_dmd.json"],
        "numbers": e,
    }


# ---------------------------------------------------------------------------
# F9: the LCoS SLM
# ---------------------------------------------------------------------------


_VORTEX_CACHE: dict = {}


def vortex_check(D: Data) -> dict:
    """The focused ring radius of a Gaussian times exp(iφ) by a direct
    Fourier sum along five azimuths (independent of the crate): untruncated
    with a continuous spiral, with the device's pixelated 256-level spiral,
    and also truncated by the device's square aperture."""
    if _VORTEX_CACHE:
        return _VORTEX_CACHE
    m, _ = D.load("f9_slm")
    c = m["parameters"]["c"]
    lam, z, w0 = c["lambda_um"] * 1e-3, c["focal_mm"], c["w0_mm"]
    pitch = c["pitch_um"] * 1e-3
    half_dev = 0.5 * c["pixels"] * pitch
    dx = 2e-3

    def ring(trunc: bool, pixel: bool) -> float:
        half = half_dev if trunc else 0.6
        n = int(round(2 * half / dx))
        x = (np.arange(n) - n / 2 + 0.5) * dx
        X, Y = np.meshgrid(x, x)
        if pixel:
            ci, cj = np.floor(X / pitch) + 0.5, np.floor(Y / pitch) + 0.5
            step = 2 * np.pi / c["lut_levels"]
            phi = np.round(np.arctan2(cj, ci) / step) * step
        else:
            phi = np.arctan2(Y, X)
        U = (np.exp(-(X**2 + Y**2) / w0**2) * np.exp(1j * phi)).ravel()
        Xr, Yr = X.ravel(), Y.ravel()
        R = np.linspace(0.62, 0.80, 181)
        acc = np.zeros_like(R)
        for az in np.deg2rad([0, 11.25, 22.5, 33.75, 45]):
            f = R / (lam * z)
            P = np.exp(-2j * np.pi * (np.outer(f * np.cos(az), Xr) + np.outer(f * np.sin(az), Yr))) @ U
            acc += np.abs(P) ** 2
        i = int(np.argmax(acc))
        a_, b_, c_ = acc[i - 1], acc[i], acc[i + 1]
        return float(R[i] + 0.5 * (R[1] - R[0]) * (a_ - c_) / (a_ - 2 * b_ + c_))

    _VORTEX_CACHE.update(
        {
            "vortex_ring_untruncated_continuous_um": 1000 * ring(False, False),
            "vortex_ring_untruncated_pixelated_um": 1000 * ring(False, True),
            "vortex_ring_truncated_pixelated_um": 1000 * ring(True, True),
        }
    )
    return _VORTEX_CACHE


def fig9(D: Data, outdir: Path) -> dict:
    m, a = D.load("f9_slm")
    p, e = m["parameters"], m["errors"]
    fig = plt.figure(figsize=(WIDTH_IN, 5.0))
    top = fig.add_gridspec(1, 2, left=0.085, right=0.975, bottom=0.6, top=0.92, wspace=0.28)
    bot = fig.add_gridspec(1, 3, left=0.06, right=0.975, bottom=0.085, top=0.46, wspace=0.9, width_ratios=[1, 1, 1.0])

    ax = fig.add_subplot(top[0])
    n = a["a_levels"]
    dn = a["a_dense_n"]
    ax.plot(dn, a["a_dense_sinc2"], **REF, zorder=2)
    ax.plot(dn, a["a_dense_dead"], color=INK, lw=0.8, zorder=2)
    ax.axhline(e["a_dead_space_zero_order"], color=INK, lw=0.8, zorder=1)
    ax.plot(n, a["a_eta1"], **sim_style(BLUE, "o", ms=4.0), label="first order, fill 1", zorder=3)
    ax.plot(n, a["a_eta1_dead"], **sim_style(ORANGE, "s", ms=3.6), label="first order, fill 0.9, |g| = 0.5", zorder=3)
    ax.plot(n, a["a_eta0_dead"], **sim_style(AQUA, "^", ms=4.0), label="zero order, fill 0.9, |g| = 0.5", zorder=3)
    ax.text(11, 1.025, r"sinc$^2(1/N)$", fontsize=6.5, color=INK, va="bottom", ha="center")
    ax.text(11, 0.675, r"$\rho^4$ sinc$^2(\rho/N)$", fontsize=6.5, color=INK, va="bottom", ha="center")
    ax.text(40, 0.03, r"$|g|^2(1-\rho^2)^2$", fontsize=6.5, color=INK, va="bottom", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.set_xlim(1.8, 72)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("phase levels per period N")
    ax.set_ylabel("efficiency")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=6.2, borderaxespad=0.2, handletextpad=0.4)
    grid(ax, axis="y")
    label(ax, "a", dx=-34, y_fig=0.94)

    ax = fig.add_subplot(top[1])
    ax.plot(a["b_w_quad"], a["b_eta_quadrature"], **REF, label="direct quadrature", zorder=2)
    ax.plot(a["b_w_px"], a["b_eta_sampled"], **sim_style(ms=3.8), label="sampled field (lcos_reflect)", zorder=3)
    ax.set_xlim(-0.03, 1.53)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("fringing width w / pixel pitch")
    ax.set_ylabel("first-order efficiency, N = 8")
    ax.legend(loc="upper right", fontsize=6.3, borderaxespad=0.2)
    grid(ax)
    label(ax, "b", dx=-34, y_fig=0.94)

    c = a["c_coord_mm"]
    hp = 0.5 * (c[1] - c[0])
    ext = [c[0] - hp, c[-1] + hp, c[0] - hp, c[-1] + hp]
    I = a["c_intensity"]
    ax = fig.add_subplot(bot[0])
    frame(ax)
    im = ax.imshow(I, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    colorbar(fig, ax, im, r"$I/I_\mathrm{peak}$", gap=0.03, width=0.06)
    label(ax, "c", dx=-28, y_fig=0.5)

    ax = fig.add_subplot(bot[1])
    frame(ax)
    ph = np.ma.masked_where(I < 1e-3, a["c_phase"])
    cm = plt.get_cmap("twilight").copy()
    cm.set_bad("#d9d8d2")
    im2 = ax.imshow(ph, origin="lower", extent=ext, cmap=cm, vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("x (mm)")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelleft=False)
    cb2 = colorbar(fig, ax, im2, "phase (rad)", gap=0.03, width=0.06)
    cb2.set_ticks([-np.pi, 0, np.pi])
    cb2.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    label(ax, "d", dx=-10, y_fig=0.5)

    ax = fig.add_subplot(bot[2])
    ax.plot(a["c_dense_r_mm"], a["c_profile_analytic"], **REF, label="closed form", zorder=2)
    ax.plot(a["c_radius_mm"], a["c_profile_sim"], **sim_style(ms=3.4), label="simulation", zorder=3)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("radius (mm)")
    ax.set_ylabel(r"azimuthal mean $I/I_\max$")
    ax.legend(loc="upper right", fontsize=6.2, borderaxespad=0.1)
    grid(ax)
    label(ax, "e", dx=-30, y_fig=0.5)

    files = save(fig, outdir, "F9_slm")
    vc = vortex_check(D)
    bw = {round(q["w_px"], 3): q["eta"] for q in e["b_eta_at_w"]}
    cp = p["c"]
    caption = (
        f"Phase-only LCoS SLM, 8 µm pixels, 633 nm (the table's design wavelength). (a) First-order efficiency of an N-level blazed staircase "
        f"(period N pixels) from the closed-form unit-cell Fourier series (unit_cell_orders, thin-element amplitudes) as markers, on the analytic "
        f"curves (lines): sinc²(1/N) with no dead space [Swanson 1989], within {sci(e['a_worst_abs_err_sinc2'])}; with electrodes of linear fill "
        f"ρ = 0.9 and unmodulated dead space of amplitude |g| = 0.5, the first order falls to ρ⁴ sinc²(ρ/N) (within "
        f"{sci(e['a_worst_abs_err_dead_space_first_order'])}) and the dead space feeds a zero order |g|²(1 − ρ²)² = {e['a_dead_space_zero_order']:.4f} "
        f"at every N (within {sci(e['a_worst_abs_err_dead_space_zero_order'])}). Only the director's polarization component is modulated "
        f"[Zhang 2014]; these are its efficiencies. (b) The fringing field modelled as linear broadening of each phase step over a width w, "
        f"the 2π flyback included [JOSA A 2004; Persson 2012]: 8-level grating, first-order efficiency of the sampled field (lcos_reflect, 48 "
        f"pixels at 32 samples each, projected onto the first order over the four interior periods) against direct quadrature of the same "
        f"window-averaged phase (line); agreement within {sci(e['b_worst_abs_diff_sampled_vs_quadrature'])}. The efficiency falls from "
        f"{bw[0.0]:.4f} (= sinc²(1/8)) to {bw[0.5]:.3f} at w = p/2, {bw[1.0]:.3f} at w = p and {bw[1.5]:.3f} at w = 1.5p. (c–e) A Gaussian "
        f"(w₀ = {cp['w0_mm']} mm) through an ℓ = 1 spiral phase (Vortex, 256-level table) on a {cp['pixels']}×{cp['pixels']} device sampled "
        f"{cp['samples_per_pixel']} per pixel, carried to the far field by propagate(Fraunhofer, pad 16) at z = {cp['focal_mm']:g} mm with the "
        f"quadratic phase e^(ikr²/2z) and e^(ikz)/i removed, i.e. the back focal plane of an f = {cp['focal_mm']:g} mm lens with the SLM in its front "
        f"focal plane. (c) Irradiance: dark core, on-axis I/I_peak = {sci(e['c_on_axis_over_peak'])}. (d) Phase (cyclic colormap; grey where "
        f"I < 10⁻³ I_peak): it winds once around the axis (measured winding number {e['c_winding_number']:.0f}) [Allen 1992]. (e) Azimuthal mean "
        f"of (c) (markers, one far-field sample of {1000*e['c_far_pitch_mm']:.0f} µm per annulus) against the closed-form focus of a Gaussian times "
        f"e^(iφ), ∝ b² e^(−2q)[I₀(q) − I₁(q)]², q = b²w₀²/8, b = 2πR/(λf) (the order-1 Hankel transform of a Gaussian, checked against numerical "
        f"quadrature): ring radius {1000*e['c_ring_radius_sim_mm']:.1f} µm vs {1000*e['c_ring_radius_analytic_mm']:.1f} µm ({100*e['c_ring_radius_rel_err']:+.2f}%), "
        f"worst normalized profile difference {e['c_profile_worst_abs_diff_normalized']:.3f}. An independent direct Fourier sum (NumPy, 2 µm "
        f"sampling, five azimuths) puts the ring at {vc['vortex_ring_untruncated_continuous_um']:.1f} µm for the untruncated Gaussian with a continuous "
        f"spiral (the closed form's value), {vc['vortex_ring_untruncated_pixelated_um']:.1f} µm with the pixelated 256-level spiral, and "
        f"{vc['vortex_ring_truncated_pixelated_um']:.1f} µm when the device's {cp['pixels'] * 8 / 1000:.3f} mm aperture also truncates the beam at "
        f"{cp['pixels'] * 4e-3 / cp['w0_mm']:.1f} w₀. The simulated ring lies between: the aperture explains the offset, and the annulus binning of the "
        f"{1000*e['c_far_pitch_mm']:.0f} µm far-field grid pulls it part of the way back. "
        f"The Fraunhofer step reports OutsideRegime (source Fresnel number 0.41 > 0.1); that criterion is for free-space propagation and does not "
        f"apply when the step stands in for a lens's Fourier transform. The reflection report flags ScalarLimit: an 8 µm pixel is "
        f"{8/0.633:.1f}λ at 633 nm, below 14λ. Sources: [Swanson 1989], [Zhang 2014], [JOSA A 2004], [Persson 2012], [Allen 1992]."
    )
    return {
        "id": "F9",
        "title": "Phase-only LCoS SLM",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Swanson 1989", "Zhang 2014", "JOSA A 2004", "Persson 2012", "Allen 1992")],
        "datasets": ["f9_slm.json"],
        "numbers": e,
    }



# ---------------------------------------------------------------------------
# F10: a real singlet's PSF through the wave path, against Optiland
# ---------------------------------------------------------------------------


def fig10(D: Data, outdir: Path) -> dict:
    m, a = D.load("f10_singlet")
    p, e = m["parameters"], m["errors"]
    x = 1e3 * a["grid_mm"]
    n = x.size
    c = n // 2
    canvas, oracle, airy = a["canvas"], a["oracle"], a["airy"]
    fig = plt.figure(figsize=(WIDTH_IN, 2.75))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.885, bottom=0.17, top=0.86, wspace=0.8, width_ratios=[1, 1.25, 1])
    ext = [x[0], x[-1], x[0], x[-1]]
    # (a) the canvas PSF
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.log10(np.maximum(canvas / 100.0, 1e-5)), origin="lower", extent=ext, cmap="inferno", vmin=-4, vmax=0)
    frame(ax)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("canvas, at the paraxial focus", fontsize=7, color=INK2)
    colorbar(fig, ax, im, r"$\log_{10}\, I / I_{\rm ideal}$", extend="min")
    label(ax, "a", dx=-34, y_fig=0.92)
    # (b) the profile through the centre, against Optiland and the Airy pattern
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(x, airy[c] / 100.0, color=INK2, lw=0.9, ls=(0, (3, 2)), label="Airy, same NA")
    ax2.semilogy(x, oracle[c] / 100.0, color=INK, lw=1.2, label="Optiland HuygensPSF")
    ax2.semilogy(x, canvas[c] / 100.0, **sim_style(BLUE, "o", ms=2.6), label="canvas wave path")
    ax2.set_ylim(1e-5, 1.6)
    ax2.set_xlim(x[0], x[-1])
    ax2.set_xlabel("x (µm), y = 0")
    ax2.set_ylabel(r"$I / I_{\rm ideal}$")
    grid(ax2)
    ax2.legend(loc="lower center", fontsize=5.8, borderaxespad=0.3, handlelength=1.8, framealpha=0.9)
    label(ax2, "b", dx=-38, y_fig=0.92)
    # (c) the difference map
    ax3 = fig.add_subplot(gs[0, 2])
    diff = canvas - oracle
    lim = float(np.max(np.abs(diff)))
    im3 = ax3.imshow(diff, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    frame(ax3)
    ax3.set_xlabel("x (µm)")
    ax3.set_title("canvas − Optiland", fontsize=7, color=INK2)
    colorbar(fig, ax3, im3, r"% of $I_{\rm ideal}$")
    label(ax3, "c", dx=-26, y_fig=0.92)
    files = save(fig, outdir, "F10_singlet")
    caption = (
        f"A real singlet's point-spread function through the canvas's wave path, against Optiland's HuygensPSF of the same lens (an oracle: it "
        f"verifies, it does not validate). The lens: equiconvex, R ±50 mm, 4 mm thick, index 1.5168 (N-BK7 at the d line), at f/{p['f_number']:g} "
        f"(entrance pupil {p['epd_mm']:.4f} mm), stop at its first surface, object at infinity on axis, λ = {p['lambda_um']} µm, the image at the paraxial "
        f"focus {p['image_distance_mm']:.6f} mm behind the lens. Canvas: a uniform beam through an iris at that surface, 2048² samples over 6 mm, each "
        f"face traced across its sag by geometric field tracing (257 × 257 f64 rays per face) with the field diffracting between, and the focus read by "
        f"the zoomed angular spectrum on Optiland's own 65 × 65 points (no paraxial step anywhere). Oracle: Optiland's Huygens sum over its exit "
        f"pupil, 512 rays across, from the clone {m['oracle']} (fixture crates/oc-wave/tests/fixtures/huygens_singlet.json, generated by "
        f"tools/fixtures/huygens_singlet.py). Both are normalized so the diffraction-limited peak of the same pupil is 1: the canvas by the peak of "
        f"a perfect spherical wave filling the exit pupil, P·π·sin²α/λ², α the half-angle the exit pupil subtends at the focus (traced: the pupil "
        f"{p['exit_pupil_before_back_vertex_mm']:.3f} mm before the back vertex, radius {p['exit_pupil_radius_mm']:.4f} mm, sin α = {p['na']:.6f}). "
        f"The marginal ray's own sine, {p['na_marginal_ray']:.6f}, is tilted by the spherical aberration's ray error and would read the whole PSF "
        f"{100 * ((p['na_marginal_ray'] / p['na']) ** 2 - 1):.2f}% low. (a) The canvas PSF, log scale. (b) The profile through the centre: the spherical "
        f"aberration fills the first dark ring of the diffraction-limited Airy pattern (dashed). Strehl ratio {e['strehl_canvas']:.4f} (canvas) "
        f"against {float(e['strehl_oracle']):.4f} (Optiland). (c) The difference: {e['rms_rel_ideal_peak']:.1e} rms of the ideal peak, at most "
        f"{np.max(np.abs(diff)):.2f}% of it; the sums over the grid agree to {100 * abs(canvas.sum() / oracle.sum() - 1):.2f}%. A residual of this "
        f"order is left by the canvas's sampling (its Strehl ratio drops 0.06% at half the 2.9 µm pitch) and by the reference itself (a uniform "
        f"spherical cap peaks 0.06% above π·sin²α/λ² at f/10). Found "
        f"along the way: Optiland's strehl_ratio() is the largest sample of its image grid, so an even grid, which has no sample on the axis, reads "
        f"0.8503 here against 0.8642 on an odd one. Sources: [Optiland], [Wyrowski & Kuhn 2011], [Matsushima 2009], [Rabiner 1969]."
    )
    return {
        "id": "F10",
        "title": "A real singlet's PSF against Optiland's HuygensPSF",
        **files,
        "caption": caption,
        "sources": [SOURCES[k] for k in ("Optiland", "Wyrowski 2011", "Matsushima 2009", "Rabiner 1969")],
        "datasets": ["f10_singlet.json"],
        "numbers": e,
    }

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = Path(argv[1])
    outdir = root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    style()
    D = Data(root)
    only = argv[2] if len(argv) > 2 else None
    figs = []
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
        if only and not fn.__name__.endswith(only):
            continue
        if fn is fig10 and not (root / "f10_singlet.json").exists():
            print("skipping F10: run the gallery_scene example (feature scene) first")
            continue
        figs.append(fn(D, outdir))
        print(f"wrote {figs[-1]['png']}")
    if only:
        return 0
    run = json.loads((root / "run.json").read_text(encoding="utf-8")) if (root / "run.json").exists() else {}
    manifest = {
        "generator": "tools/wave_gallery.py",
        "data_generator": "crates/oc-wave/examples/gallery.rs",
        "data_dir": str(root.resolve()),
        "example_run": run,
        "conventions": "Simulation: markers or image. Closed-form reference: solid line. Time dependence exp(-i omega t); lengths mm, wavelengths um.",
        "special_function_crosscheck_vs_scipy": crosschecks(D),
        "sources": SOURCES,
        "figures": figs,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {outdir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
