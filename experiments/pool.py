"""候选池构造：复现与消融共用，保证三档选择器在同一候选池上比较。

池内候选（全部只用 (X,y)，无真解）：
  - logspace / logdiff / linear_denom 路线输出（系数已吸附）
  - narrow 每个起点的两种版本：
      * narrow_raw_*  选中特征上的原始 OLS（高精度脏系数，未吸附）
      * narrow_snap_* 有理吸附后的干净式（= 路线实际输出）
  - recursive 路线输出（系数已吸附）
"""
from __future__ import annotations

import numpy as np

from scesr import logspace as ls
from scesr import recursive as rc
from scesr.sparse import _narrow_feats, _sparse_narrow_once
from scesr.util import nrmse_score, complexity


def _raw_ols_expr(F, names, feats, y):
    cols = [names.index(nm) for nm in feats]
    A = np.concatenate([F[:, cols], np.ones((len(y), 1))], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    body = ""
    for i, (u, nm) in enumerate(zip(c[:-1], feats)):
        body += ("-" if u < 0 else ("+" if i else "")) + f"{abs(u):.15g}*{nm}"
    if abs(c[-1]) > 1e-12:
        body += ("+" if c[-1] > 0 else "-") + f"{abs(c[-1]):.15g}"
    return body or "0"


def build_pool(X, y, seed=0):
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, d = X.shape
    P = []

    def add(arm, expr):
        nr = nrmse_score(expr, X, y)
        if np.isfinite(nr):
            P.append({"arm": arm, "expression": expr,
                      "nrmse": float(nr), "complexity": complexity(expr)})

    for arm, fn in (("logspace", ls.fit_logspace),
                    ("logdiff", ls.fit_logspace_diff),
                    ("linear_denom", ls.solve_linear_denom)):
        try:
            r = fn(X, y)
            if r.get("applicable"):
                add(arm, r["expression"])
        except Exception:
            pass

    # 忠实复刻 solve_union 短路：便宜路线已有 NRMSE<1e-6 的精确候选时，
    # 不再运行 narrow / recursive（避免脏常数的 expwrap 版以更小 NRMSE 混入）。
    if any(c["nrmse"] < 1e-6 for c in P):
        return P

    n_restarts = 6 if d <= 4 else (3 if d <= 6 else 1)
    try:
        F, names = _narrow_feats(X)
        # 忠实复刻 solve_sparse_narrow：多起点只保留全数据 NRMSE 最低的一个，
        # 不把每个起点都塞进池（否则复杂度平局时错误起点会以极小 NRMSE 胜出）。
        best_r = None
        for r in range(n_restarts):
            res = _sparse_narrow_once(F, names, X, y, seed + r)
            if res and (best_r is None or res["fit_nrmse"] < best_r[1]["fit_nrmse"]):
                best_r = (r, res)
        # seed0 始终保留（A/B 的"单起点通用搜索"），含 raw 与 snap
        r0 = _sparse_narrow_once(F, names, X, y, seed)
        if r0 is not None:
            add("narrow_snap_r0", r0["expression"])
            add("narrow_raw_r0", _raw_ols_expr(F, names, r0["features"], y))
        # 多起点最优 snap：与 seed0 不同才加入（C 档多路线成员）
        if best_r is not None and best_r[0] != 0 and r0 is not None \
                and best_r[1]["expression"] != r0["expression"]:
            add(f"narrow_multistart_r{best_r[0]}", best_r[1]["expression"])
    except Exception:
        pass

    try:
        r = rc.solve(X, y)
        if r.get("applicable"):
            add(r.get("route", "recursive"), r["expression"])
    except Exception:
        pass

    return P


# --- 三档选择器（候选同源，逐级加机制）---

def baseline_cands(cands):
    """A/B 的候选集：一次"通用稀疏搜索" = narrow 单一起点(seed=0)。

    池内 narrow 只保留多起点最优的那个；A/B 要模拟"不做多起点"，故只在
    seed=0 时才认 narrow，否则退回各便宜路线与 recursive 的单路线候选
    ——后一类公式（纯乘除/根式直接命中）三档结果相同（天花板公式）。
    """
    s0 = [c for c in cands if c["arm"] in ("narrow_raw_r0", "narrow_snap_r0")]
    if s0:
        return s0
    # narrow seed0 无候选（纯乘除/根式或仅多起点才命中）：
    # 用全部"已构造干净"候选，但不含多起点最优（那是 C 的机制）。
    return [c for c in cands
            if not c["arm"].startswith("narrow_raw")
            and not c["arm"].startswith("narrow_multistart")]


def pick_error_only(cands):
    """A：仅按误差——在基础候选上 argmin NRMSE。raw OLS 天然占优。"""
    return min(baseline_cands(cands), key=lambda c: c["nrmse"])


def pick_simplicity(cands, tol=1e-4):
    """B：加入简洁性——同基础候选，有理吸附后在误差门(NRMSE<tol)内取算子最少。"""
    base = [c for c in baseline_cands(cands)
            if not c["arm"].startswith("narrow_raw")]
    base = base or baseline_cands(cands)
    exact = [c for c in base if c["nrmse"] < tol]
    if exact:
        return min(exact, key=lambda c: (c["complexity"], c["nrmse"]))
    return min(base, key=lambda c: c["nrmse"])


def pick_full_multi(cands, tol=1e-4):
    """C：完整多路线——全部"已构造干净"候选（多起点 snap、各路线族），误差门取最简。

    raw OLS 仅用于 A 档展示"纯误差驱动"的坏味道，不进入正式求解；
    这与 src solve_union 的行为完全一致。
    与 B 的唯一差别：B 只在 narrow 单起点 snap 上取简，
    C 额外汇聚多起点 snap 与全部路线族，覆盖单起点路径依赖与模型族缺口。
    """
    clean = [c for c in cands if not c["arm"].startswith("narrow_raw")]
    exact = [c for c in clean if c["nrmse"] < tol]
    if exact:
        return min(exact, key=lambda c: (c["complexity"], c["nrmse"]))
    return min(clean, key=lambda c: c["nrmse"])
