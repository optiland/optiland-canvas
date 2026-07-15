"""Compare the canvas and optiland harness outputs ray-by-ray."""
from __future__ import annotations
import json, math, sys

canvas = json.load(open(sys.argv[1]))
opti = json.load(open(sys.argv[2]))


def key(r):
    return (round(r["y_in"], 6), round(r["angle_deg"], 6), round(r.get("lambda_um", 0.0), 6))


def index_rays(sysobj):
    return {key(r): r for r in sysobj["rays"]}


def compare_group(cv_list, op_list, label):
    cv = {s["name"]: s for s in cv_list}
    op = {s["name"]: s for s in op_list}
    print(f"\n{'='*92}\n{label}\n{'='*92}")
    header = (f"{'system':<34} {'d_exit_x':>10} {'d_exit_y':>10} "
              f"{'d_dir':>11} {'d_img_y':>11}  n")
    grand = {"exit_x": 0.0, "exit_y": 0.0, "dir": 0.0, "img_y": 0.0}
    for name in cv:
        if name not in op:
            print(f"  !! {name} missing from optiland output")
            continue
        cr = index_rays(cv[name])
        orr = index_rays(op[name])
        d_ex = d_ey = d_dir = d_img = 0.0
        cnt = 0
        details = []
        for k, c in cr.items():
            if k not in orr:
                details.append(f"    missing optiland ray {k}")
                continue
            o = orr[k]
            e_x = abs(c["exit_x"] - o["exit_x"])
            e_y = abs(c["exit_y"] - o["exit_y"])
            # direction difference as chord length between unit vectors
            dd = math.hypot(c["dir_x"] - o["dir_x"], c["dir_y"] - o["dir_y"])
            if c.get("img_y") is None or o.get("img_y") is None:
                e_img = 0.0
            else:
                e_img = abs(c["img_y"] - o["img_y"])
            d_ex = max(d_ex, e_x); d_ey = max(d_ey, e_y)
            d_dir = max(d_dir, dd); d_img = max(d_img, e_img)
            cnt += 1
        if cnt and (cnt == len(cr)):
            pass
        print(header if name == list(cv)[0] else "", end="")
        print(f"{name:<34} {d_ex:>10.2e} {d_ey:>10.2e} {d_dir:>11.2e} {d_img:>11.2e}  {cnt}")
        for dline in details:
            print(dline)
        for kk, vv in [("exit_x", d_ex), ("exit_y", d_ey), ("dir", d_dir), ("img_y", d_img)]:
            grand[kk] = max(grand[kk], vv)
    print(f"{'-'*92}")
    print(f"{'WORST OVER ALL SYSTEMS':<34} {grand['exit_x']:>10.2e} {grand['exit_y']:>10.2e} "
          f"{grand['dir']:>11.2e} {grand['img_y']:>11.2e}")


compare_group(canvas["systems"], opti["systems"], "CONSTANT-INDEX / MIRROR SYSTEMS (isolates geometry + interface physics)")
compare_group(canvas["dispersion"], opti["dispersion"], "DISPERSION SYSTEMS (N-BK7, F/d/C)")

# dispersion index cross-check
print(f"\n{'-'*60}\nRefractive index model cross-check (canvas vs optiland)")
cv_disp = {s["name"]: s for s in canvas["dispersion"]}
op_disp = {s["name"]: s for s in opti["dispersion"]}
for name in cv_disp:
    ci = {round(x["lambda_um"], 6): x["n"] for x in cv_disp[name]["index"]}
    oi = {round(x["lambda_um"], 6): x["n"] for x in op_disp[name]["index"]}
    for lam in sorted(ci):
        print(f"  {name}  lam={lam:.5f}  canvas n={ci[lam]:.12f}  optiland n={oi[lam]:.12f}  delta={abs(ci[lam]-oi[lam]):.2e}")
