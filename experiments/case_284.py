"""有材料意义的详细案例：MF-bench idx284 线弹性断裂韧度。

真值：K_Ic = Y · σ · sqrt(π · a)
变量与定义域（取自 MF-bench var_ranges）：
    Y     几何修正因子，无量纲，0.8–1.5
    σ     远场拉应力，MPa，10–1500
    a     裂纹半长/长度，m，1e-6–1e-2
输出 K_Ic 单位 MPa·m^1/2。

本脚本展示：
  1) 训练域内恢复（无真解输入，只给 (X,y)）；
  2) 训练域内 NRMSE；
  3) 外推：各变量单独放大到定义域 2 倍/5 倍，对比恢复式与真式；
  4) 与一个"纯多项式伪装"候选对照，展示范围内打平、外推分化。
输出：results/case284.json
用法：python experiments/case_284.py
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
from scesr.util import nrmse_score
from common import structural_equivalent
from pool import build_pool, pick_full_multi

OUT = HERE.parent / "results" / "case284.json"

DOMAIN = {
    "formula": "K_Ic = Y*sigma*sqrt(pi*a)",
    "vars": [
        {"sym": "Y (x0)", "meaning": "几何修正因子（裂纹形状/构件几何）",
         "unit": "1", "range": [0.8, 1.5]},
        {"sym": "sigma (x1)", "meaning": "远场施加拉应力",
         "unit": "MPa", "range": [10.0, 1500.0]},
        {"sym": "a (x2)", "meaning": "裂纹长度",
         "unit": "m", "range": [1e-6, 1e-2]},
    ],
    "output": {"sym": "K_Ic", "unit": "MPa*m^1/2"},
    "source": "Mechanical behavior of materials; LEFM I 型应力强度因子",
}


def true_value(X):
    return X[:, 0] * X[:, 1] * np.sqrt(np.pi * X[:, 2])


def sample_grid(n, ranges, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(lo, hi, n) for lo, hi in ranges], axis=1)


def relmax(pred, true):
    return float(np.max(np.abs(pred - true)) / (np.max(np.abs(true)) + 1e-12))


def main():
    ranges = [v["range"] for v in DOMAIN["vars"]]
    t0 = time.time()

    # 1) 训练数据（走 scesr.gen，n=400, seed=0）
    g = gen(284, n=400, sigma=0.0, seed=0)
    X = np.asarray(g["X"]).T.astype(np.float64)
    y = np.asarray(g["yc"], np.float64)

    pool = build_pool(X, y, seed=0)
    best = pick_full_multi(pool)
    from common import eval_expr
    pred_in = eval_expr(best["expression"], X)

    recovery = {
        "expression": best["expression"], "arm": best["arm"],
        "nrmse": round(best["nrmse"], 10),
        "complexity": best["complexity"],
        "se": structural_equivalent(best["expression"], g["true"], X.shape[1]),
        "in_sample_relerr": relmax(pred_in, y),
    }

    # 2) 外推：独立大样本，逐变量放大到 2x / 5x 域
    extrap = []
    base_hi = np.array([r[1] for r in ranges])
    base_lo = np.array([r[0] for r in ranges])
    Xbig = sample_grid(2000, ranges, seed=77)
    for k, var in enumerate(DOMAIN["vars"]):
        for mult in (2.0, 5.0):
            Xe = Xbig.copy()
            Xe[:, k] = base_hi[k] * mult * (Xe[:, k] - base_lo[k]) \
                / (base_hi[k] - base_lo[k])
            # 保持其余变量在域内，目标变量在 [hi, hi*mult]
            tval = true_value(Xe)
            pval = eval_expr(best["expression"], Xe)
            extrap.append({
                "variable": var["sym"], "factor": mult,
                "range": [base_hi[k], base_hi[k] * mult],
                "relerr": relmax(pval, tval),
            })

    # 3) 多项式伪装对照：用 a 的高次多项式在范围内拟合 sqrt(pi*a) 的行为
    #    y = Y*sigma * g(a)，g 为 a 的 5 次最小二乘（仅用训练数据）。
    a_col = X[:, 2]
    scale = X[:, 0] * X[:, 1]
    target_g = np.sqrt(np.pi * a_col)
    A = np.stack([a_col ** p for p in range(1, 6)] + [np.ones_like(a_col)], 1)
    c, *_ = np.linalg.lstsq(A, target_g, rcond=None)

    def poly_g(a):
        return sum(c[p - 1] * a ** p for p in range(1, 6)) + c[5]

    decoy_in = scale * poly_g(a_col)
    # 外推 a 到 5x
    Xe = Xbig.copy()
    Xe[:, 2] = base_hi[2] * 5.0 * (Xe[:, 2] - base_lo[2]) \
        / (base_hi[2] - base_lo[2])
    decoy_out = Xe[:, 0] * Xe[:, 1] * poly_g(Xe[:, 2])
    decoy = {
        "description": "范围内 5 次多项式伪装 sqrt(pi*a)（经典装饰式）",
        "in_sample_relerr": relmax(decoy_in, y),
        "extrap_a_5x_relerr": relmax(decoy_out, true_value(Xe)),
        "recovered_extrap_a_5x_relerr": next(
            e["relerr"] for e in extrap
            if e["variable"] == "a (x2)" and e["factor"] == 5.0),
    }

    result = {"domain": DOMAIN, "recovery": recovery,
              "extrapolation": extrap, "decoy": decoy,
              "sec": round(time.time() - t0, 2)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    print(f"true       : {DOMAIN['formula']}")
    print(f"recovered  : {recovery['expression']}  (arm={recovery['arm']})")
    print(f"SE={recovery['se']}  in-sample relerr={recovery['in_sample_relerr']:.2e}")
    for e in extrap:
        print(f"  extrap {e['variable']:<10} x{e['factor']:g}: relerr={e['relerr']:.2e}")
    print(f"decoy in-sample={decoy['in_sample_relerr']:.2e} "
          f"a-5x={decoy['extrap_a_5x_relerr']:.2e} "
          f"(recovered a-5x={decoy['recovered_extrap_a_5x_relerr']:.2e})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
