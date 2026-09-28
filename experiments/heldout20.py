"""跨公式泛化测试：方法设计期完全未接触的 20 条 MF-bench 公式。

与"同一公式换种子"的严格区分：
  - 主测试 HELD-OUT 20：20 条【不同公式】，每条只跑 1 个采样种子
    => 测的是跨公式泛化。
  - 稳定性小测 STABILITY 5：另取 5 条公式，每条跑 3 个采样种子
    => 测的是采样稳定性，不充当泛化证据。

选样（可复现）：
  候选 = 307 条中 gen 可成功生成者，排除所有方法设计/评测期接触过的
  （first-20、PAPER20、custom 10 条、Feynman 非 MF 无关）；
  按算子族（纯乘除 / 加减分式 / 含根号 / 含超越 / 含平方）×变量数分层，
  PRNG(固定种子) 分层抽取，全程无人工挑选。
输出：results/heldout20.json
用法：python experiments/heldout20.py
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

DATASET = "/home/zyx/A_project/SR_Works/MF-bench-dataset/kept_formulas_with_background.json"
OUT = HERE.parent / "results" / "heldout20.json"

# 方法设计/评测期接触过的全部 new_idx（first-20 + PAPER20 + custom 其余）
TOUCHED = (set(range(1, 21))
           | {8, 9, 21, 49, 52, 71, 94, 106, 111, 183,
              187, 198, 199, 200, 239, 257, 280, 284, 293, 296}
           | {41})

SELECT_SEED = 20260927
STABILITY_SEEDS = [0, 11, 23]


def op_family(ops, has_log_exp_trig):
    s = set(ops)
    if has_log_exp_trig:
        return "transcend"
    if "\\sqrt" in s:
        return "sqrt"
    if "^{2}" in s:
        return "square"
    if "+" in s or "-" in s or "\\frac" in s:
        return "addfrac"
    return "puremul"


def nvar_bin(nv):
    return "d<=3" if nv <= 3 else ("d4-5" if nv <= 5 else "d>=6")


def main():
    meta = {x["new_idx"]: x for x in json.load(open(DATASET))}

    # 1) 候选：可生成 + 未接触
    candidates = []
    for idx in sorted(meta):
        if idx in TOUCHED:
            continue
        g = None
        try:
            g = gen(idx, n=80, sigma=0, seed=0)
        except Exception:
            g = None
        if g is None or not np.all(np.isfinite(np.asarray(g["y"]))):
            continue
        m = meta[idx]
        nv = len(m["var_ranges"])
        trig = any(o in m["operators"] for o in ("\\ln", "\\log", "\\exp",
                                                "\\sin", "\\cos", "\\tan"))
        candidates.append({"idx": idx,
                           "stratum": (op_family(m["operators"], trig),
                                       nvar_bin(nv)),
                           "nv": nv})

    # 2) 分层抽 20（固定种子，均匀铺到算子族；族内按变量数分层随机）
    rng = np.random.default_rng(SELECT_SEED)
    by_fam = {}
    for c in candidates:
        by_fam.setdefault(c["stratum"], []).append(c)
    for k in by_fam:
        rng.shuffle(by_fam[k])

    fams = sorted(by_fam)
    quota = {f: 20 // len(fams) for f in fams}
    for i in range(20 - sum(quota.values())):
        quota[fams[i % len(fams)]] += 1

    held, used = [], set()
    for f in fams:
        take = quota[f]
        for c in by_fam[f]:
            if take == 0:
                break
            if c["idx"] not in used:
                held.append(c); used.add(c["idx"]); take -= 1

    # 3) 稳定性小测：从未入选的候选里另抽 5 条
    rest = [c for f in fams for c in by_fam[f] if c["idx"] not in used]
    rng.shuffle(rest)
    stab = rest[:5]

    def solve_one(idx, seed):
        g = gen(idx, n=N_SAMPLES, sigma=SIGMA, seed=seed)
        X = np.asarray(g["X"]).T.astype(np.float64)
        y = np.asarray(g["yc"], np.float64)
        pool = build_pool(X, y, seed=seed)
        best = pick_full_multi(pool)
        return {
            "true": g["true"], "expression": best["expression"],
            "arm": best["arm"], "nrmse": round(best["nrmse"], 10),
            "complexity": best["complexity"],
            "se": structural_equivalent(best["expression"], g["true"], X.shape[1]),
            "fresh_relerr": fresh_relerr(best["expression"], gen, idx),
        }

    # 4) 主测试
    rows, t0 = [], time.time()
    for c in held:
        ts = time.time()
        r = solve_one(c["idx"], SEED_TRAIN)
        r.update({"idx": c["idx"], "stratum": list(c["stratum"]),
                  "nv": c["nv"], "sec": round(time.time() - ts, 2)})
        rows.append(r)
        print(f"held idx{c['idx']:>3} {'/'.join(c['stratum']):<16} "
              f"SE={int(r['se'])} fresh={r['fresh_relerr']:.1e}", flush=True)

    # 5) 稳定性小测（5 公式 × 3 种子）
    stab_rows = []
    for c in stab:
        cell = []
        for sd in STABILITY_SEEDS:
            r = solve_one(c["idx"], sd)
            cell.append({"seed": sd, "expression": r["expression"],
                         "arm": r["arm"], "se": r["se"],
                         "fresh_relerr": r["fresh_relerr"]})
        stab_rows.append({"idx": c["idx"], "runs": cell})
        same = len({x["expression"] for x in cell}) == 1
        print(f"stab idx{c['idx']:>3} expr-identical-across-3-seeds={same} "
              f"SEs={[int(x['se']) for x in cell]}", flush=True)

    total = time.time() - t0
    summary = {
        "n_held": len(rows),
        "SE": sum(r["se"] for r in rows),
        "fresh_strict": sum(r["fresh_relerr"] < 1e-3 for r in rows),
        "stability": {
            "n_formulas": len(stab_rows), "seeds": STABILITY_SEEDS,
            "expr_identical_all_seeds": sum(
                len({x["expression"] for x in s["runs"]}) == 1
                for s in stab_rows),
        },
        "total_sec": round(total, 1),
    }
    meta_out = {
        "protocol": {"n": N_SAMPLES, "sigma": SIGMA, "seed_train": SEED_TRAIN,
                     "seed_fresh": 7, "dtype": "float64", "exact_tol": 1e-4,
                     "select_seed": SELECT_SEED},
        "exclusion": "剔除全部设计/评测期接触过的 new_idx",
    }
    OUT.write_text(json.dumps({"meta": meta_out, "summary": summary,
                              "heldout": rows, "stability_check": stab_rows},
                             ensure_ascii=False, indent=1))
    print(f"\nHELD-OUT 跨公式 SE={summary['SE']}/{len(rows)} "
          f"fresh={summary['fresh_strict']}/{len(rows)}")
    print(f"STABILITY 表达式跨3种子完全一致 "
          f"{summary['stability']['expr_identical_all_seeds']}/{len(stab_rows)}")
    print(f"total {total:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
