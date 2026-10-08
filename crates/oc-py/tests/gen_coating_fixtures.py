"""Freeze what ``tmm`` and Optiland say about every case in ``coating_cases``.

    python gen_coating_fixtures.py            # (re)write data/coatings/*.json
    python gen_coating_fixtures.py --check    # regenerate and compare, write nothing

One JSON file per case: the stack, the grid, the complex indices the canvas
sampled (and fed to both oracles), each oracle's answers with its name and
version, and the hash of the scripts that produced them. The canvas is held to
these files on every pull request by ``test_coating_fixtures.py``, so CI does
not need the oracles installed and an oracle release cannot move the goalposts
silently.

``--check`` is the scheduled job's half. It regenerates every file twice and
requires the two runs to be byte-identical, then compares the result with the
committed files: the metadata exactly, the numbers to ``CHECK_ATOL``. Not byte
for byte, because the oracles run on numpy's complex ``libm``, which is allowed
to differ in the last place between platforms, and a fixture written on one
machine is checked on another.

The oracle versions are pinned in ``requirements-oracles.txt``; a version bump
is a deliberate regeneration, reviewed like any other change to a reference.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "coatings"
CHECK_ATOL = 1e-13

sys.path.insert(0, str(HERE))
import coating_cases  # noqa: E402

from optiland_canvas import coatings  # noqa: E402


def script_hash() -> str:
    """SHA-256 of the scripts that define a fixture, line endings normalised so
    a Windows checkout hashes the same as a Linux one."""
    h = hashlib.sha256()
    for name in ("gen_coating_fixtures.py", "coating_cases.py"):
        h.update((HERE / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _c(z: complex) -> list[float]:
    return [float(z.real), float(z.imag)]


def _tmm(case_stack, lambdas, angles, n0, nf) -> dict:
    import numpy as np
    import tmm

    out = {k: [] for k in ("Rs", "Rp", "Ts", "Tp", "rs", "rp", "ts", "tp")}
    for lam in lambdas:
        n_list, d_list = coatings.to_tmm(case_stack, lam, n0, nf)
        rows = {k: [] for k in out}
        for deg in angles:
            for pol in "sp":
                o = tmm.coh_tmm(pol, n_list, d_list, np.radians(deg), lam)
                rows[f"R{pol}"].append(float(o["R"]))
                rows[f"T{pol}"].append(float(o["T"]))
                rows[f"r{pol}"].append(_c(complex(o["r"])))
                rows[f"t{pol}"].append(_c(complex(o["t"])))
        for k in out:
            out[k].append(rows[k])
    return {"name": "tmm", "version": tmm.__version__ if hasattr(tmm, "__version__") else _dist_version("tmm"), **out}


def _optiland(case_stack, lambdas, angles, n0, nf) -> dict:
    """Optiland's own numbers, in its own convention (Macleod's). Amplitudes
    past the critical angle are ``None``: Optiland's exit medium sits on the
    growing branch there (spec §9), and the reconciliation is CF-06's."""
    import optiland

    crit = coating_cases.critical_deg(n0, nf)
    out = {k: [] for k in ("Rs", "Rp", "Ts", "Tp", "rs", "rp", "ts", "tp")}
    for lam in lambdas:
        ref = coatings.to_optiland(case_stack, lam, n0, nf)
        rows = {k: [] for k in out}
        for deg in angles:
            past = crit is not None and deg >= crit
            for pol in "sp":
                o = ref.compute_rtRTA([lam], [math.radians(deg)], polarization=pol)
                rows[f"R{pol}"].append(float(o["R"][0][0]))
                rows[f"T{pol}"].append(float(o["T"][0][0]))
                rows[f"r{pol}"].append(None if past else _c(complex(o["r"][0][0])))
                rows[f"t{pol}"].append(None if past else _c(complex(o["t"][0][0])))
        for k in out:
            out[k].append(rows[k])
    return {"name": "optiland.thin_film", "version": optiland.__version__, **out}


def _dist_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


def build(name: str, date: str) -> dict:
    case_stack, n0, nf, description = coating_cases.CASES[name]
    lambdas = coating_cases.LAMBDAS
    angles = coating_cases.angles_for(name)
    sampled = [coatings.indices(case_stack, lam) for lam in lambdas]
    return {
        "case": name,
        "description": description,
        "stack": case_stack,
        "n_incident": n0,
        "n_far": nf,
        "lambda_um": lambdas,
        "aoi_deg": angles,
        "indices_fed": [[_c(z) for z in row] for row in sampled],
        "oracles": {
            "tmm": _tmm(case_stack, lambdas, angles, n0, nf),
            "optiland": _optiland(case_stack, lambdas, angles, n0, nf),
        },
        "generator": {"script_sha256": script_hash(), "date": date},
    }


def dumps(doc) -> str:
    """JSON with every list of scalars or of ``[re, im]`` pairs on one line, so
    a grid reads as rows and a regeneration diffs as rows."""

    def flat(v) -> bool:
        return isinstance(v, list) and all(
            not isinstance(x, (dict, list)) or (isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x))
            for x in v
        )

    def emit(v, depth: int) -> str:
        pad, inner = " " * depth, " " * (depth + 1)
        if isinstance(v, dict):
            if not v:
                return "{}"
            items = [f"{inner}{json.dumps(k)}: {emit(x, depth + 1)}" for k, x in v.items()]
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        if isinstance(v, list) and not flat(v):
            return "[\n" + ",\n".join(inner + emit(x, depth + 1) for x in v) + "\n" + pad + "]"
        return json.dumps(v, separators=(", ", ": "))

    return emit(doc, 0) + "\n"


def existing_date(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["generator"]["date"]
    except (OSError, KeyError, ValueError):
        return None


def regenerate() -> dict[str, str]:
    """Every case as the text it would be written as. A file whose content is
    unchanged keeps its date, so regenerating an up-to-date tree is a no-op."""
    today = datetime.date.today().isoformat()
    texts = {}
    for name in coating_cases.CASES:
        path = OUT / f"{name}.json"
        old = existing_date(path)
        text = dumps(build(name, old or today))
        if old is not None and path.read_text(encoding="utf-8") != text:
            text = dumps(build(name, today))
        texts[name] = text
    return texts


def _numbers_close(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a)} vs {sorted(b)}"]
        return [m for k in a for m in _numbers_close(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in _numbers_close(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return [] if abs(a - b) <= CHECK_ATOL else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def check() -> int:
    first, second = regenerate(), regenerate()
    problems = [f"{n}: two runs differ" for n in first if first[n] != second[n]]
    on_disk = {p.stem for p in OUT.glob("*.json") if p.stem != "tolerances"}
    problems += [f"{n}: committed but no longer a case" for n in sorted(on_disk - set(first))]
    for name, text in first.items():
        path = OUT / f"{name}.json"
        if not path.exists():
            problems.append(f"{name}: not committed")
            continue
        new, old = json.loads(text), json.loads(path.read_text(encoding="utf-8"))
        problems += [f"{name}{m}" for m in _numbers_close(new, old)][:5]
    for p in problems:
        print(p)
    print("fixtures up to date" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="regenerate and compare; write nothing")
    args = ap.parse_args()
    if args.check:
        return check()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in regenerate().items():
        (OUT / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote data/coatings/{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
