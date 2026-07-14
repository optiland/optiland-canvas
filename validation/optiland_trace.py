"""Optiland harness: trace the shared systems.json sample systems through the
REAL Optiland engine (build an Optic, inject custom RealRays into its
surface_group, read recorded per-surface state). Emits per-ray exit state as
JSON in the same 'canvas frame' as the Rust harness:

    exit_x = z (surface 1 sits at z=0),  exit_y = y,
    dir_x  = N,  dir_y = M,  img_y = ray y at the image plane.

Coefficient mapping: canvas asphere coeffs[i]*y^(2i+4) (r^4, r^6, ...) map to
optiland EvenAsphere coefficients [0.0, c0, c1, ...] (the leading 0.0 is the r^2
term canvas does not have).

Mirror: canvas builds a concave conic via axis=-x, c=+1/R; the same physical
surface z=-y^2/(2R) is optiland radius=-R. So we negate the spec radius.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

import optiland.backend as be
from optiland.optic import Optic
from optiland.materials import IdealMaterial, Material
from optiland.rays.real_rays import RealRays

INF = float("inf")


def a1(x):
    return np.atleast_1d(be.to_numpy(be.array(x))).astype(float)


def build_refractive_optic(surfaces, image_z, material_maker):
    """Build an N-surface sequential refractive system. material_maker(idx) ->
    the material for the medium AFTER surface idx. Returns (optic, exit_idx,
    image_idx). Surface 1 sits at global z=0; the image plane at z=image_z."""
    def rad(r):
        return INF if abs(r) >= 1e30 else r

    opt = Optic()
    opt.add_surface(index=0, thickness=25.0, radius=INF, material="air")

    n = len(surfaces)
    z_cum = 0.0
    for i, s in enumerate(surfaces):
        thickness = s["thickness"] if i < n - 1 else (image_z - z_cum)
        coeffs = s.get("coeffs", []) or []
        kwargs = dict(index=i + 1, thickness=thickness, radius=rad(s["radius"]),
                      conic=s["conic"], material=material_maker(i))
        if coeffs:
            kwargs.update(surface_type="even_asphere", coefficients=[0.0, *coeffs])
        opt.add_surface(**kwargs)
        z_cum += s["thickness"]
    opt.add_surface(index=n + 1, thickness=0.0, radius=INF, material="air")
    return opt, n, n + 1  # exit = last optical surface, image = next


def build_mirror_optic(radius, conic, image_z):
    opt = Optic()
    opt.add_surface(index=0, thickness=25.0, radius=INF, material="air")
    # Negate radius: canvas axis=-x, c=+1/R  <=>  optiland radius=-R (same sag).
    opt.add_surface(
        index=1, thickness=image_z, radius=-radius, conic=conic, material="mirror"
    )
    opt.add_surface(index=2, thickness=0.0, radius=INF, material="air")
    return opt, 1, 2


def trace_rays(opt, exit_idx, image_idx, z_start, rays_spec, wavelength):
    """Inject rays, trace through surface_group (skip object), return list of
    per-ray dicts in the canvas frame."""
    ys = [r["y"] for r in rays_spec]
    angs = [math.radians(r["angle_deg"]) for r in rays_spec]
    n = len(ys)
    rays = RealRays(
        x=[0.0] * n,
        y=list(ys),
        z=[z_start] * n,
        L=[0.0] * n,
        M=[math.sin(a) for a in angs],
        N=[math.cos(a) for a in angs],
        intensity=[1.0] * n,
        wavelength=[wavelength] * n,
    )
    opt.surface_group.trace(rays, skip=1)

    ex = opt.surface_group.surfaces[exit_idx]
    im = opt.surface_group.surfaces[image_idx]
    exit_z, exit_y = a1(ex.z), a1(ex.y)
    exit_N, exit_M = a1(ex.N), a1(ex.M)
    img_y = a1(im.y)

    out = []
    for i, rspec in enumerate(rays_spec):
        out.append(
            {
                "y_in": rspec["y"],
                "angle_deg": rspec["angle_deg"],
                "exit_x": exit_z[i],
                "exit_y": exit_y[i],
                "dir_x": exit_N[i],
                "dir_y": exit_M[i],
                "img_y": img_y[i],
            }
        )
    return out


def main():
    spec = json.load(open(sys.argv[1]))
    z_start = spec["z_start"]

    out_systems = []
    for sysv in spec["systems"]:
        name = sysv["name"]
        kind = sysv["kind"]
        image_z = sysv["image_z"]

        if kind == "mirror":
            opt, exit_idx, image_idx = build_mirror_optic(
                sysv["radius"], sysv["conic"], image_z
            )
            wavelength = 0.58756
        else:
            surfaces = sysv["surfaces"]
            n_after = [s["n_after"] for s in surfaces]

            def maker(idx, n_after=n_after):
                return "air" if abs(n_after[idx] - 1.0) < 1e-12 else IdealMaterial(n_after[idx])

            opt, exit_idx, image_idx = build_refractive_optic(surfaces, image_z, maker)
            wavelength = 0.58756

        rays_out = trace_rays(opt, exit_idx, image_idx, z_start, sysv["rays"], wavelength)
        out_systems.append({"name": name, "rays": rays_out})

    out_disp = []
    for sysv in spec.get("dispersion_systems", []):
        name = sysv["name"]
        glass = sysv["glass"]
        image_z = sysv["image_z"]
        surfaces = sysv["surfaces"]

        index_report = []
        rays_out = []
        for lam in sysv["wavelengths_um"]:
            glass_mat = Material(glass)
            n_here = float(a1(glass_mat.n(lam))[0])
            index_report.append({"lambda_um": lam, "n": n_here})

            def maker(idx, glass=glass, n=len(surfaces)):
                return "air" if idx == n - 1 else Material(glass)

            opt, exit_idx, image_idx = build_refractive_optic(surfaces, image_z, maker)
            traced = trace_rays(opt, exit_idx, image_idx, z_start, sysv["rays"], lam)
            for t in traced:
                t["lambda_um"] = lam
            rays_out.extend(traced)
        out_disp.append({"name": name, "index": index_report, "rays": rays_out})

    print(json.dumps({"engine": "optiland", "systems": out_systems, "dispersion": out_disp}, indent=2))


if __name__ == "__main__":
    main()
