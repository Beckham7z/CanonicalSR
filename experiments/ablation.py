"""最小消融：同一份数据、同一个候选池，只逐级加机制。

三档（每档候选同源，差异可严格归因）：
  A_error_only   一次通用稀疏搜索(narrow seed0)，纯 argmin NRMSE
                 ——raw OLS 高精度脏系数被选中
  B_simplicity   同候选，有理吸附 + NRMSE<1e-4 门内取算子最少
  C_full_multi   候选扩到全部路线族(logspace/logdiff/linear_denom/recursive)
                 + narrow 全部起点，同误差门取最简

提升归因：
  A→B：有理系数吸附 + 简洁性 tie-break（几乎不花 NRMSE，换严格等价）
  B→C：候选覆盖（多起点 + 多模型族）
输出：results/ablation_first20.json
用法：python experiments/ablation.py
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
from pool import (build_pool, pick_error_only, pick_simplicity,
                  pick_full_multi)

OUT = HERE.parent / "results" / "ablation_first20.json"
LEVELS = [("A_error_only", pick_error_only),
          ("B_simplicity", pick_simplicity),
          ("C_full_multi", pick_full_multi)]


def main():
    per_idx = []
    t0 = time.time()
    for idx in range(1, 21):
        g = gen(idx, n=N_SAMPLES, sigma=SIGMA, seed=SEED_TRAIN)
        X = np.asarray(g["X"]).T.astype(np.float64)
        y = np.asarray(g["yc"], np.float64)
        d = X.shape[1]

        pool = build_pool(X, y, seed=SEED_TRAIN)
        cell = {}
        for name, fn in LEVELS:
            ts = time.time()
            c = fn(pool)
            cell[name] = {
                "expression": c["expression"], "arm": c["arm"],
                "nrmse": round(c["nrmse"], 10), "complexity": c["complexity"],
                "se": structural_equivalent(c["expression"], g["true"], d),
                "fresh_relerr": fresh_relerr(c["expression"], gen, idx),
            }
        per_idx.append({"idx": idx, "true": g["true"], "levels": cell})
        a, b, c = (cell[k] for k, _ in LEVELS)
        print(f"idx{idx:>2}: A_se={int(a['se'])}({a['complexity']}) "
              f"B_se={int(b['se'])}({b['complexity']}) "
              f"C_se={int(c['se'])}({c['complexity']})", flush=True)

    total = time.time() - t0
    summary = {}
    for name, _ in LEVELS:
        cells = [r["levels"][name] for r in per_idx]
        summary[name] = {
            "SE": sum(c["se"] for c in cells),
            "fresh_strict": sum(c["fresh_relerr"] < 1e-3 for c in cells),
            "mean_nrmse": float(np.mean([c["nrmse"] for c in cells])),
            "mean_complexity": float(np.mean([c["complexity"] for c in cells])),
        }
    # 翻转计数：每一步新救回/丢失多少
    def flips(lo, hi):
        gained = lost = 0
        for r in per_idx:
            a, b = r["levels"][lo]["se"], r["levels"][hi]["se"]
            gained += (not a) and b
            lost += a and (not b)
        return {"gained": gained, "lost": lost}

    summary["A_to_B"] = flips("A_error_only", "B_simplicity")
    summary["B_to_C"] = flips("B_simplicity", "C_full_multi")

    meta = {"protocol": {"n": N_SAMPLES, "sigma": SIGMA,
                         "seed_train": SEED_TRAIN, "seed_fresh": 7,
                         "dtype": "float64", "exact_tol": 1e-4},
            "total_sec": round(total, 1)}
    OUT.write_text(json.dumps({"meta": meta, "summary": summary,
                               "rows": per_idx}, ensure_ascii=False, indent=1))

    print("\n" + "=" * 56)
    for name, _ in LEVELS:
        s = summary[name]
        print(f"{name:<14} SE={s['SE']:>2}/20  fresh={s['fresh_strict']:>2}/20"
              f"  meanNRMSE={s['mean_nrmse']:.2e}"
              f"  meanCx={s['mean_complexity']:.1f}")
    print(f"A→B {summary['A_to_B']}    B→C {summary['B_to_C']}")
    print(f"total {total:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
