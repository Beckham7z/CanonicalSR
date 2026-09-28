"""Generalization scan: CanonicalSR on a large set of formulas never used in design.

For each gen-able MF-bench formula outside every touched/seen set, run
solve_union (400 pts, noiseless, seed 0) and record strict structural SE.

Output: results/generalization.json (written incrementally)
Usage: python experiments/run_generalization.py
"""
from __future__ import annotations

import json, time, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scesr import gen, solve_union
from common import structural_equivalent

DATA = "/home/zyx/A_project/SR_Works/MF-bench-dataset/kept_formulas_with_background.json"
OUT = HERE.parent / "results" / "generalization.json"

FIRST20 = set(range(1, 21))
CUSTOM_SEEN = {9, 284, 49, 183, 200, 280, 257, 296, 8, 52, 21, 94, 106, 71,
               111, 187, 199, 285, 69, 3, 286, 292, 206, 25, 228, 234, 283, 291,
               305, 28, 29}
HELDOUT = {33, 26, 211, 59, 22, 98, 247, 191, 104, 230, 248, 251, 289, 254,
           240, 158, 47, 268, 279, 270}
EXCLUDE = FIRST20 | CUSTOM_SEEN | HELDOUT


def load_done():
    if OUT.exists():
        d = json.load(open(OUT))
        return {r["idx"]: r for r in d["rows"]}
    return {}


def main():
    meta = {x["new_idx"]: x for x in json.load(open(DATA))}
    done = load_done()
    rows = list(done.values())

    candidates = [i for i in sorted(meta) if i not in EXCLUDE and i not in done]
    t0 = time.time()
    for n, idx in enumerate(candidates):
        try:
            g = gen(idx, n=400, sigma=0.0, seed=0)
        except Exception:
            g = None
        if g is None:
            continue
        X = np.asarray(g["X"], float); y = np.asarray(g["yc"], float)
        nv = X.shape[0]
        if not np.all(np.isfinite(y)) or y.std() < 1e-12:
            continue
        ts = time.time()
        try:
            r = solve_union(X.T, y)
            expr = r.get("expression", "")
            se = structural_equivalent(expr, g["true"], nv)
        except Exception:
            expr, se = "", False
        rec = {"idx": idx, "nv": nv, "se": bool(se), "expression": expr,
               "sec": round(time.time() - ts, 2)}
        rows.append(rec); done[idx] = rec
        if (n + 1) % 10 == 0:
            ok = sum(x["se"] for x in rows)
            print(f"{n+1}/{len(candidates)} tested, recovered {ok}/{len(rows)}",
                  flush=True)
            json.dump({"rows": rows}, open(OUT, "w"), ensure_ascii=False)

    json.dump({"rows": rows}, open(OUT, "w"), ensure_ascii=False)
    ok = sum(x["se"] for x in rows)
    print(f"DONE {ok}/{len(rows)} over {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
