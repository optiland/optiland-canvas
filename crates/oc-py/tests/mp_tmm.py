"""A coherent transfer-matrix reference in arbitrary precision.

Byrnes' formulation (interface matrices and propagation matrices, as in
``tmm.coh_tmm``), written out again here with ``mpmath`` numbers so that no
stack can overflow it: a 64-layer stack of thick absorbing films has a matrix
product of size exp(thousands), which ``tmm`` in double precision turns into
NaN (every one of 1200 such stacks did, measured on tmm 0.2.0) and the canvas
handles with its opacity rule and rescaling. This computes the same stack
exactly, with no rule at all, so it can judge both.

It is checked against ``tmm`` itself on stacks ``tmm`` can do
(``test_thinfilm_stress.py``), so it is an oracle in its own right and not a
transcription of the code it tests: it shares neither the canvas's admittance
algebra nor its floating point.

Convention: ``exp(-iwt)``, index ``n + ik``, angles in the incident medium,
thicknesses in the wavelength's unit, ``tmm``'s. Its amplitudes are therefore
the canvas's ray-frame amplitudes with no conversion (``r_p = -r_s`` at normal
incidence, whole-field ``t_p``).
"""

from __future__ import annotations

import mpmath as mp

DPS = 40


def _forward_cos(n, sin_t):
    """cos theta in a medium of index n, on the physically forward branch: the
    wave decays (Im(n cos) > 0) when it can, else travels forward."""
    c = mp.sqrt(1 - sin_t * sin_t)
    nc = n * c
    if abs(mp.im(nc)) > mp.mpf(10) ** (-DPS + 10) * (1 + abs(nc)):
        return c if mp.im(nc) > 0 else -c
    return c if mp.re(nc) > 0 else -c


def coh(pol: str, n_list, d_list, th0_rad: float, lam: float) -> dict:
    """``{"r", "t", "R", "T"}`` for one polarization, as mpmath numbers.

    ``n_list`` and ``d_list`` run from the incident medium to the exit medium;
    the two outer thicknesses are ignored (semi-infinite).
    """
    with mp.workdps(DPS):
        n = [mp.mpc(complex(x)) for x in n_list]
        d = [mp.mpf(0)] + [mp.mpf(x) for x in d_list[1:-1]] + [mp.mpf(0)]
        lam = mp.mpf(lam)
        sin0 = mp.sin(mp.mpf(th0_rad))
        cos = [_forward_cos(ni, n[0] * sin0 / ni) for ni in n]
        kz = [2 * mp.pi * n[i] * cos[i] / lam for i in range(len(n))]

        def interface(i, j):
            ni, nj, ci, cj = n[i], n[j], cos[i], cos[j]
            if pol == "s":
                den = ni * ci + nj * cj
                return (ni * ci - nj * cj) / den, 2 * ni * ci / den
            den = nj * ci + ni * cj
            return (nj * ci - ni * cj) / den, 2 * ni * ci / den

        def mul(a, b):
            return [
                [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
            ]

        r01, t01 = interface(0, 1)
        m = [[1 / t01, r01 / t01], [r01 / t01, 1 / t01]]
        for i in range(1, len(n) - 1):
            delta = kz[i] * d[i]
            r, t = interface(i, i + 1)
            e_minus, e_plus = mp.exp(-1j * delta), mp.exp(1j * delta)
            step = [[e_minus / t, e_minus * r / t], [e_plus * r / t, e_plus / t]]
            m = mul(m, step)
        r = m[1][0] / m[0][0]
        t = 1 / m[0][0]
        nf, cf, n0, c0 = n[-1], cos[-1], n[0], cos[0]
        if pol == "s":
            big_t = abs(t) ** 2 * mp.re(nf * cf) / mp.re(n0 * c0)
        else:
            big_t = abs(t) ** 2 * mp.re(nf * mp.conj(cf)) / mp.re(n0 * mp.conj(c0))
        return {"r": r, "t": t, "R": abs(r) ** 2, "T": big_t}
