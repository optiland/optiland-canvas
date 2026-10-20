"""The modulator bindings reproduce the kernel's published-value checks.

These repeat, through Python, two results held in Rust by
``crates/oc-wave/tests/dmd.rs`` and ``slm.rs``, so a broken bridge cannot hide
behind passing Rust tests:

- an N-level blazed phase grating puts sinc²(1/N) in its first order (G. J.
  Swanson, MIT Lincoln Laboratory TR-854 (1989));
- a 7.56 µm, 12° DMD at matched f/2.4 cones is within 8 percentage points of
  Texas Instruments DLPA083B Table 6-1's 84% photopic diffraction efficiency,
  a vendor simulation, here at 550 nm only as a smoke check (the photopic
  average is the Rust test's).
"""

import numpy as np
import pytest

from optiland_canvas import wave


@pytest.mark.parametrize("n,published", [(2, 0.40528), (4, 0.81057), (8, 0.94964), (16, 0.98721)])
def test_blazed_staircase_first_order(n, published):
    dev = wave.lcos(1, 8.0, levels=n, pixel_fill=1.0)
    t = wave.orders(dev, [list(range(n))], 0.633, (0.0, 0.0, -1.0), model="tea")
    first = next(o for o in t["orders"] if o["m"] == 1 and o["n"] == 0)
    assert first["efficiency"] == pytest.approx(np.sinc(1.0 / n) ** 2, abs=1e-12)
    assert first["efficiency"] == pytest.approx(published, abs=5e-6)


def test_dmd_projector_efficiency_is_near_ti_at_550_nm():
    dev = wave.dmd(1, 7.56, 12.0, "Diagonal", mirror_fill=np.sqrt(0.94))
    eta = wave.projector_efficiency(dev, 0.55, 2.4, 2.4)
    assert abs(eta - 0.84) < 0.08, eta


def test_gerchberg_saxton_error_never_rises():
    y, x = np.mgrid[-16:16, -16:16]
    src = np.exp(-(x**2 + y**2) / 60.0)
    tgt = ((abs(x) < 4) & (abs(y) < 4)).astype(float)
    start = np.random.default_rng(1).uniform(-np.pi, np.pi, src.shape)
    phase, err = wave.gerchberg_saxton(src, tgt, 30, start=start)
    assert phase.shape == src.shape
    assert np.all(np.diff(err) <= 1e-12)
