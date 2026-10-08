"""The spectrograph's table as a visitor would see it, as a PNG.

Reads the pixel colours the demonstrator's bench writes
(`spectrograph-table-xyz.csv`: CIE 1931 X, Y, Z per pixel in lux, u fastest,
v upward) and draws them the way the app's detector view does
(web/src/true-colour.ts): spectral colours moved toward the grey of their
own luminance until sRGB can show them, exposed so the brightest part of the
table (the 99th percentile of each pixel's largest channel) is full scale
times 2^EV, sRGB encoded.

    python table_image.py spectrograph-table-xyz.csv table.png [--ev 0] [--mm-per-px 1 2]

The pixels are 1 x 2 mm on the table by default; rows are repeated so the
picture keeps the table's proportions.
"""

import argparse

import numpy as np
from PIL import Image

XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("png")
    ap.add_argument("--ev", type=float, default=0.0)
    ap.add_argument("--mm-per-px", type=float, nargs=2, default=[1.0, 2.0])
    a = ap.parse_args()

    with open(a.csv) as f:
        head = f.readline()
    nu, nv = (int(t) for t in head.split()[1:4:2])
    xyz = np.loadtxt(a.csv, delimiter=",", comments="#").reshape(nv, nu, 3)

    rgb = xyz @ XYZ_TO_SRGB.T
    lum = xyz[..., 1:2]
    lo = rgb.min(axis=-1, keepdims=True)
    t = np.where((lo < 0) & (lum > 0), -lo / np.maximum(lum - lo, 1e-30), 0.0)
    rgb = rgb + t * (lum - rgb)
    peak = rgb.max(axis=-1)
    lit = peak[xyz[..., 1] > 0]
    white = np.percentile(lit, 99) if lit.size else 1.0
    rgb = rgb / white * 2.0**a.ev
    hi = rgb.max(axis=-1, keepdims=True)
    rgb = np.where(hi > 1, rgb / np.maximum(hi, 1e-30), rgb)
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    img = (srgb * 255 + 0.5).astype(np.uint8)[::-1]  # v upward; image rows down
    rows = max(1, round(a.mm_per_px[1] / a.mm_per_px[0]))
    img = np.repeat(img, rows, axis=0)
    Image.fromarray(img, "RGB").save(a.png)
    print(f"{nu} x {nv} pixels, full scale at {white:.1f} lux, wrote {a.png}")


if __name__ == "__main__":
    main()
