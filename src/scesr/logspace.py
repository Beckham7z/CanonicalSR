"""对数空间结构求解器：无真解恢复"纯乘除单项式"公式。

适用：f 为各正变量的乘/除/幂（含根号）单项式，形如
    f = c * Π x_k^(p_k),  p_k ∈ {±2, ±1, ±1/2}
两边取对数后为线性：
    log f = log c + Σ p_k log x_k
做法：
  1. 校验 X,y 全为正（否则判定不适用，交回其他求解器）；
  2. log 空间最小二乘估计系数与截距；
  3. 系数吸附到有理数网格 {0, ±.5, ±1, ±2}，截距吸附为常数候选；
  4. 还原符号表达式并返回。

不使用神经网络；σ=0 下在案例 9/187 精确恢复。
"""
from __future__ import annotations
import numpy as np

GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def snap_rational(x, grid=GRID):
    x = float(x)
    return grid[int(np.argmin([abs(x - g) for g in grid]))]


def is_pure_monomial(X, y):
    return bool(np.all(np.isfinite(X)) and np.all(X > 0)
                and np.all(np.isfinite(y)) and np.all(y > 0))


def fit_logspace(X, y, tol_coef=0.10, min_r2=0.98):
    """返回结果 dict；不适用时 applicable=False。

    纯乘除单项式要求：系数吸附偏差很小（<=tol_coef）且对数空间拟合 R²>=min_r2，
    避免把"单一主导变量的退化拟合"（如280的 1.018*x1，R²虚高）误判为纯单项式。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, d = X.shape
    if not is_pure_monomial(X, y):
        return {"applicable": False, "reason": "含非正/非单调变量或输出，非纯乘除单项式"}

    Z = np.log(X)
    Y = np.log(y)
    A = np.concatenate([Z, np.ones((n, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    raw = coef[:d]
    intercept = float(coef[d])

    snapped = np.array([snap_rational(c) for c in raw])
    max_dev = float(np.max(np.abs(raw - snapped)))

    # 拟合质量（吸附后）
    pred = Z @ snapped + intercept
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    const = float(np.exp(intercept))

    return {
        "applicable": True,
        "raw_coef": np.round(raw, 4).tolist(),
        "exponents": snapped.tolist(),
        "max_coef_deviation": round(max_dev, 4),
        "intercept_log": round(intercept, 4),
        "constant": round(const, 5),
        "log_r2": round(r2, 6),
        "within_tol": max_dev <= tol_coef and r2 >= min_r2,
        "expression": _build_expr(snapped, const),
    }


def _build_expr(exponents, const):
    num, den = [], []
    for k, p in enumerate(exponents):
        x = f"x{k}"
        if p == 0:
            continue
        ap = abs(p)
        term = x if ap == 1 else f"{x}^{ap}"
        if p > 0:
            num.append(term)
        else:
            den.append(term)
    num_s = "*".join(num) if num else "1"
    if not den:
        body = num_s
    else:
        body = f"({num_s})/({ '*'.join(den) })"
    if abs(const - 1.0) < 1e-6:
        return body
    return f"{round(const,5)}*{body}"


# ---------------------------------------------------------------------------
# 对数空间 + 差分原子：f = c * Π x_k^{p_k} * Π (x_i - x_j)^{q}（16/17/18 型）
# ---------------------------------------------------------------------------

def _const_str(c):
    """把常数识别成 sqrt(n)/pi/e / 整数×简单无理，否则高精度小数。"""
    import math
    if abs(c - math.pi) < 0.02:
        return "pi"
    if abs(c - math.e) < 0.02:
        return "E"
    for k in range(2, 11):
        if abs(c - math.sqrt(k)) < 0.02:
            return f"sqrt({k})"
    if abs(c - math.sqrt(math.pi)) < 0.02:
        return "sqrt(pi)"
    # 常数合成：c = k / f 或 c = k * f，f∈{ sqrt(pi), pi, E, sqrt(2..3) }，k 为整数
    for fname, fv in [("sqrt(pi)", math.sqrt(math.pi)), ("pi", math.pi),
                      ("E", math.e), ("sqrt(2)", math.sqrt(2)),
                      ("sqrt(3)", math.sqrt(3))]:
        k = c * fv
        if abs(k - round(k)) < 1e-4 * max(1.0, abs(k)) and abs(k) > 1e-6:
            return f"({round(k)}/{fname})"
        k = c / fv
        if abs(k - round(k)) < 1e-4 * max(1.0, abs(k)) and abs(k) > 1e-6:
            return f"({round(k)}*{fname})"
    return f"{c:.10g}"


def _pow_str(base, p):
    ap = abs(float(p))
    if abs(ap - 1) < 1e-9:
        return base
    if abs(ap - 0.5) < 1e-9:
        return f"sqrt({base})"
    from fractions import Fraction
    fr = Fraction(ap).limit_denominator(8)
    if fr.denominator > 1:
        return f"({base})**({fr.numerator}/{fr.denominator})"
    return f"({base})**{fr.numerator}"


def fit_logspace_diff(X, y, tol_coef=0.06, min_r2=0.999):
    """在 log(y) 上对 {log x_i} ∪ {log(x_i-x_j)} 做线性回归并吸附指数。

    适用：y>0，且真式是"若干正变量幂 × 若干正线性差幂"的乘积（含 sqrt）。
    返回 {applicable, expression, exponents, log_r2, within_tol}。
    """
    import itertools
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, d = X.shape
    if not (np.all(np.isfinite(y)) and np.all(y > 0)):
        # 混合符号：退回带符号差分路径（如 257：y∝(x1-x0)策略）
        return _fit_logdiff_signed(X, y, tol_coef=tol_coef, min_r2=min_r2)

    feats, meta = [], []
    for i in range(d):
        if np.all(X[:, i] > 0):
            feats.append(np.log(X[:, i]))
            meta.append(("var", i, None))
    for i, j in itertools.combinations(range(d), 2):
        diff = X[:, i] - X[:, j]
        if not np.all(diff > 1e-12):
            continue
        # 共线守卫：log(x_i-x_j) 与 log x_i / log x_j 近共线时（量表级差太大），
        # 该差原子会被 OLS 拿来摊掉变量的指数（如 x0≈1e8, xi≈1e-3）。跳过。
        li, lj = np.log(X[:, i]), np.log(X[:, j])
        ld = np.log(diff)
        c = [abs(np.corrcoef(ld, li)[0, 1]),
             abs(np.corrcoef(ld, lj)[0, 1])]
        if max(c) > 0.999:
            continue
        feats.append(ld)
        meta.append(("diff", i, j))
    if not feats:
        return {"applicable": False, "reason": "无可用的正变量/正差"}

    A = np.stack(feats, axis=1)
    Y = np.log(y)
    B = np.concatenate([A, np.ones((n, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(B, Y, rcond=None)
    raw = coef[:-1]
    intercept = float(coef[-1])

    snapped = np.round(raw * 4.0) / 4.0      # 吸附到 1/4 网格
    max_dev = float(np.max(np.abs(raw - snapped))) if len(raw) else 0.0
    pred = A @ snapped + intercept
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    num, den = [], []
    for (kind, i, j), p in zip(meta, snapped):
        if abs(p) < 1e-9:
            continue
        base = f"x{i}" if kind == "var" else f"(x{i}-x{j})"
        (num if p > 0 else den).append(_pow_str(base, abs(p)))
    if not num and not den:
        return {"applicable": False, "reason": "指数全零"}
    body_num = "*".join(num) if num else "1"
    const = _const_str(float(np.exp(intercept)))
    expr = f"{const}*({body_num})"
    if den:
        expr += f"/({ '*'.join(den) })"

    return {
        "applicable": True,
        "expression": expr,
        "exponents": snapped.tolist(),
        "raw_coef": np.round(raw, 4).tolist(),
        "max_coef_deviation": round(max_dev, 4),
        "log_r2": round(r2, 6),
        "within_tol": (max_dev <= tol_coef and r2 >= min_r2),
    }


def _fit_logdiff_signed(X, y, tol_coef=0.06, min_r2=0.999):
    """带符号差分：y 可正可负，|y| 是"正变量幂 × |差|^q"的乘积。

    在 log|y| 上用 {log x_i} ∪ {log|xi-xj|} 回归吸附；还原时奇整数指数
    的差底子写成带符号的 (xi-xj)^p，从而保留符号。
    """
    import itertools
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, d = X.shape
    if not np.all(np.isfinite(y)) or float(np.mean(np.abs(y))) < 1e-30:
        return {"applicable": False, "reason": "y 平凡"}
    signs = np.sign(y)
    mixed = bool(np.any(signs > 0)) and bool(np.any(signs < 0))

    feats, meta = [], []
    for i in range(d):
        if np.all(X[:, i] > 0):
            feats.append(np.log(X[:, i])); meta.append(("var", i, "p"))
    with np.errstate(all="ignore"):
      for i, j in itertools.combinations(range(d), 2):
        ad = np.abs(X[:, i] - X[:, j])
        m = ad > 1e-12
        if m.sum() < 10:
            continue
        li, lj = np.log(np.where(X[:, i] > 0, X[:, i], np.nan)), \
                 np.log(np.where(X[:, j] > 0, X[:, j], np.nan))
        ld = np.log(np.where(ad > 1e-12, ad, np.nan))
        def _cc(a, b):
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10:
                return 0.0
            c = np.corrcoef(a[ok], b[ok])[0, 1]
            return 0.0 if not np.isfinite(c) else abs(c)
        cc = [_cc(ld, li), _cc(ld, lj)]
        if max(cc) > 0.999:
            continue
        feats.append(np.nan_to_num(ld, nan=1e-9))
        meta.append(("diff", i, j, np.sign(X[:, i] - X[:, j])))

    if not feats:
        return {"applicable": False, "reason": "无特征"}
    A = np.stack(feats, axis=1); Y = np.log(np.abs(y))
    B = np.concatenate([A, np.ones((n, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(B, Y, rcond=None)
    raw = coef[:-1]; intercept = float(coef[-1])
    snapped = np.round(raw * 2.0) / 2.0
    # 允许半整数/1/2 网格（含 1）
    sn = np.round(raw * 2.0) / 2.0
    max_dev = float(np.max(np.abs(raw - sn)))
    pred = A @ sn + intercept
    ss_res = float(np.sum((Y - pred) ** 2)); ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    num, den = [], []
    for entry, p in zip(meta, sn):
        if abs(p) < 1e-9:
            continue
        kind = entry[0]
        if kind == "var":
            i = entry[1]; base = f"x{i}"; t = _pow_str(base, abs(p))
        else:
            i, j = entry[1], entry[2]
            odd_int = (abs(p) - abs(round(p))) < 1e-6 and abs(round(p)) % 2 == 1
            if odd_int and mixed:
                # 选择与 sign(y) 一致的方向，保证整体符号正确
                d = X[:, i] - X[:, j]
                orient = (i, j) if np.corrcoef(signs, np.sign(d))[0, 1] >= 0 \
                    else (j, i)
                t = _pow_str(f"(x{orient[0]}-x{orient[1]})", abs(p))
            else:
                ap = abs(float(p))
                if abs(ap - 0.5) < 1e-9:
                    t = f"sqrt((x{i}-x{j})**2)"
                elif abs(ap - 1.0) < 1e-9:
                    t = f"Abs(x{i}-x{j})"
                else:
                    t = f"Abs(x{i}-x{j})**{round(ap,3):g}"
        (num if p > 0 else den).append(t)
    const = _const_str(float(np.exp(intercept)))
    body_num = "*".join(num) if num else "1"
    expr = f"{const}*({body_num})"
    if den:
        expr += f"/({ '*'.join(den) })"
    return {"applicable": True, "expression": expr, "log_r2": round(r2, 6),
            "exponents": sn.tolist(), "raw_coef": np.round(raw, 4).tolist(),
            "max_coef_deviation": round(max_dev, 4),
            "within_tol": (max_dev <= tol_coef and r2 >= min_r2)}


def solve_linear_denom(X, y, grid=None, min_r2=0.999):
    """线性分母 × 单项式分子：y = monomial(x) / (1 + a*x_i)。

    复用健壮的 fit_logspace_diff：对每个变量 i 和 a∈grid，算 r = y*(1+a*x_i)，
    若 r 是干净单项式(logdiff 命中且 log_r2 高) → 重建 y = r_expr / (1+a*x_i)。
    取最简单(项少)结果。
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, d = X.shape
    if grid is None:
        grid = [0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
                4.0, -4.0, 0.75, -0.75, 1.5, -1.5, 0.125, -0.125]
    best = None
    for i in range(d):
        xi = X[:, i]
        for a in grid:
            denom = 1.0 + a * xi
            if np.min(np.abs(denom)) < 1e-6:
                continue
            r = y * denom
            if not (np.all(np.isfinite(r)) and np.sum(np.abs(r)) > 0):
                continue
            rr = fit_logspace_diff(X, r, min_r2 = max(0.999, min_r2 - 0.001)) \
                if np.all(r > 0) else _fit_logdiff_signed(X, r)
            if not rr.get("applicable") or not rr.get("within_tol") or rr.get("log_r2", 0) < 0.999:
                continue
            # 重建 y = (单项式)/(1 + a x_i)
            expr = f"({rr['expression']})/(1{'−' if a<0 else '+'}{abs(a):g}*x{i})"
            # 符号统一
            expr = expr.replace("−", "-")
            from .util import nrmse_score
            nr = float(nrmse_score(expr, X, y))
            if nr < 1e-4:
                n_ops = expr.count("+") + expr.count("-") + expr.count("*") \
                        + expr.count("/") + expr.count("sqrt")
                if best is None or n_ops < best["n_ops"]:
                    best = {"expression": expr, "fit_nrmse": nr, "n_ops": n_ops,
                            "a": a, "i": i}
    if best is None:
        return {"applicable": False}
    return {"applicable": True, "within_tol": True, "log_r2": 1.0, **best}
