"""统一入口 solve_union：无真解，自动选择合适的恢复路线。

候选路线（全部只用 (X,y)，不看真解、不用公式库、不训练网络）：
  - logspace      纯乘除单项式        c·Π x_k^{p_k}
  - logdiff       对数空间+差分原子    c·Π x^p·Π(x_i-x_j)^q
  - linear_denom  线性分母×单项式      monomial/(1+a·x_i)
  - narrow        窄字典稀疏加法       稀疏多项式（含比值/对数比值）
  - recursive     残差逐层剥离 + exp/log 外壳

择优：在"近似精确拟合（NRMSE<1e-4）"的候选里取【最简】（算子最少）；
      若都不精确，取 NRMSE 最小。干净结构因此压过数值等价但更繁的装饰式。
SE/OSS/TSS/PSC 由外部 metrics 在评测期对照真式计算，不进入求解。
"""
from __future__ import annotations

import numpy as np

from .util import nrmse_score, complexity
from . import logspace as _ls
from . import sparse as _sp
from . import recursive as _rc


def solve_union(X, y, seed=0, n_restarts=None):
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, d = X.shape

    def _nr(e):
        return nrmse_score(e, X, y)

    # 1) 便宜路线：纯单项式 / 差分根式 / 线性分母
    cheap = []
    for route, fn in (("logspace", _ls.fit_logspace),
                      ("logdiff", _ls.fit_logspace_diff),
                      ("linear_denom", _ls.solve_linear_denom)):
        try:
            r = fn(X, y)
            if r.get("applicable") and r.get("within_tol"):
                e = r["expression"]
                cheap.append({"route": route, "expression": e,
                              "nrmse": round(float(_nr(e)), 8),
                              "complexity": complexity(e)})
        except Exception:
            pass
    exact_cheap = [c for c in cheap if c["nrmse"] < 1e-6]
    if exact_cheap:
        best = min(exact_cheap, key=lambda s: (s["complexity"], s["nrmse"]))
        best["all_routes"] = [c["route"] for c in cheap]
        return best

    cands = list(cheap)

    # 2) 窄字典稀疏（开销随维度上升 → 自适应起点数）
    if n_restarts is None:
        n_restarts = 6 if d <= 4 else (3 if d <= 6 else 1)
    try:
        r = _sp.solve_sparse_narrow(X, y, seed=seed, n_restarts=n_restarts)
        if r.get("applicable"):
            e = r["expression"]
            cands.append({"route": "narrow", "expression": e,
                          "nrmse": round(float(_nr(e)), 8),
                          "complexity": complexity(e)})
    except Exception:
        pass

    # 3) 残差递归兜底
    try:
        r = _rc.solve(X, y)
        if r.get("applicable"):
            e = r["expression"]
            cands.append({"route": r.get("route", "recursive"),
                          "expression": e, "nrmse": round(float(_nr(e)), 8),
                          "complexity": complexity(e)})
    except Exception:
        pass

    if not cands:
        return {"route": "unsolved", "expression": ""}

    exact = [s for s in cands if s["nrmse"] < 1e-4]
    if exact:
        best = min(exact, key=lambda s: (s["complexity"], s["nrmse"]))
    else:
        best = min(cands, key=lambda s: s["nrmse"])
    best["all_routes"] = [s["route"] for s in cands]
    return best
