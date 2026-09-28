"""复现并固定核心结果：MF-bench first-20（new_idx 1..20），干净数据。

协议（见 REPORT.md §2）：
    n=400, sigma=0, 训练 seed=0；独立复核 seed=7；dtype=float64；
    选解 = 完整多路线（误差门 NRMSE<1e-4 内取算子最少）。
输出：results/core_first20.json
用法：python experiments/reproduce_core.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scesr import gen
from common import (N_SAMPLES, SIGMA, SEED_TRAIN, structural_equivalent,
                    fresh_relerr)
from pool import build_pool, pick_full_multi

OUT = HERE.parent / "results" / "core_first20.json"


def main():
    rows = []
    t0 = time.time()
    for idx in range(1, 21):
        g = gen(idx, n=N_SAMPLES, sigma=SIGMA, seed=SEED_TRAIN)
        X = np.asarray(g["X"]).T.astype(np.float64)
        y = np.asarray(g["yc"], np.float64)
        d = X.shape[1]

        ts = time.time()
        pool = build_pool(X, y, seed=SEED_TRAIN)
        best = pick_full_multi(pool)
        dt = time.time() - ts

        se = structural_equivalent(best["expression"], g["true"], d)
        fr = fresh_relerr(best["expression"], gen, idx)

        rows.append({
            "idx": idx, "true": g["true"], "expression": best["expression"],
            "arm": best["arm"], "nrmse": round(best["nrmse"], 10),
            "complexity": best["complexity"],
            "se": bool(se), "fresh_relerr": fr,
            "n_pool": len(pool), "sec": round(dt, 2),
        })
        print(f"idx{idx:>2} SE={int(se)} fresh={fr:.2e} "
              f"arm={best['arm']:<16} t={dt:5.1f}s", flush=True)

    total = time.time() - t0
    summary = {
        "protocol": {"n": N_SAMPLES, "sigma": SIGMA, "seed_train": SEED_TRAIN,
                     "seed_fresh": 7, "dtype": "float64", "exact_tol": 1e-4,
                     "set": "MF-bench new_idx 1..20"},
        "SE": sum(r["se"] for r in rows),
        "fresh_strict": sum(r["fresh_relerr"] < 1e-3 for r in rows),
        "mean_nrmse": float(np.mean([r["nrmse"] for r in rows])),
        "total_sec": round(total, 1),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1))
    print(f"\nSE={summary['SE']}/20  fresh-strict={summary['fresh_strict']}/20"
          f"  total={total:.1f}s  -> {OUT}")


if __name__ == "__main__":
    main()
