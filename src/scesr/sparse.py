"""通用多项式结构求解器：无真解恢复 线性/加减/乘方/双线性 闭式结构。

稳健流程：
  1. 构造多项式特征（变量、平方、两两乘积，可扩到更高次/比值型）；
  2. 标准化特征与 y；
  3. 用 Lasso（L1）稀疏地选出真正起作用的少数特征（强相关特征下比 OLS 稳）；
  4. 在被选特征上做普通最小二乘精修系数；
  5. 系数吸附到有理网格 {0,±.5,±1,±2}；吸附后若仍高拟合、且最简，则接受；
  6. 还原成原始变量表达式（截距按尺度吸附为0）。

这是主入口在"非纯乘除单项式"时的通用兜底路线。
"""
from __future__ import annotations
import itertools
import numpy as np

GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
RATIONAL_GRID = np.array(GRID)


def snap_value(x):
    x = float(x)
    return float(RATIONAL_GRID[int(np.argmin(np.abs(RATIONAL_GRID - x)))])


def poly_features(X, degree=3):
    n, d = X.shape
    cols, names = [], []
    # degree 1
    for i in range(d):
        cols.append(X[:, i]); names.append(f"x{i}")
    if degree >= 2:
        for i in range(d):
            cols.append(X[:, i] ** 2); names.append(f"x{i}**2")
        for i, j in itertools.combinations(range(d), 2):
            cols.append(X[:, i] * X[:, j]); names.append(f"x{i}*x{j}")
    if degree >= 3:
        for i in range(d):
            cols.append(X[:, i] ** 3); names.append(f"x{i}**3")
        for i in range(d):
            for j in range(d):
                if j != i:
                    cols.append(X[:, i] ** 2 * X[:, j]); names.append(f"x{i}**2*x{j}")
        for i, j, k in itertools.combinations(range(d), 3):
            cols.append(X[:, i] * X[:, j] * X[:, k]); names.append(f"x{i}*x{j}*x{k}")
    return np.stack(cols, axis=1), names


def _zscore(F):
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (F - mu) / sd, mu, sd


def lasso_select(Fz, yz, alpha):
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=alpha, max_iter=20000, tol=1e-5)
    model.fit(Fz, yz)
    return np.abs(model.coef_) > 1e-6


def solve_poly(X, y, degree=3, min_r2=0.99):
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, d = X.shape

    F, names = poly_features(X, degree=degree)
    Fz, f_mu, f_sd = _zscore(F)
    y_mu, y_sd = y.mean(), y.std()
    if y_sd < 1e-12:
        y_sd = 1.0
    yz = (y - y_mu) / y_sd

    # L1（标准化空间）只负责"选哪些特征"；
    # 选中后，在【原始空间】做 OLS，并把原始系数吸附到有理网格。
    candidates = []
    for alpha in (0.10, 0.05, 0.02, 0.01, 0.005):
        mask = lasso_select(Fz, yz, alpha)
        if mask.sum() == 0:
            continue
        idx_cols = np.where(mask)[0]
        Fsub = F[:, idx_cols]  # 原始空间特征
        A = np.concatenate([Fsub, np.ones((Fsub.shape[0], 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        raw_sub = np.array(coef[:-1])
        ic = float(coef[-1])
        snapped = np.array([snap_value(v) for v in raw_sub])
        pred = Fsub @ snapped + ic
        sse = float(np.sum((pred - y) ** 2))
        r2 = 1 - sse / float(np.sum((y - y.mean()) ** 2) or 1e-12)
        candidates.append((r2, mask, snapped, ic, idx_cols))
        if r2 >= min_r2:
            break  # L1 由稀到密，首个达标即最简

    if not candidates:
        return {"applicable": False, "reason": "L1路径无候选"}

    ok = [c for c in candidates if c[0] >= min_r2]
    chosen = min(ok, key=lambda c: int(c[1].sum())) if ok else max(candidates, key=lambda c: c[0])
    r2, mask, snapped, ic, idx_cols = chosen
    if r2 < min_r2:
        return {"applicable": False, "reason": f"多项式R2={r2:.4f}<.99", "r2": round(r2, 4)}

    raw = np.zeros(F.shape[1])
    for local, gi in enumerate(idx_cols):
        raw[gi] = snapped[local]

    # 截距吸附：相对 y 尺度近似0则归零
    if abs(ic) < 0.02 * (np.mean(np.abs(y)) + 1e-12):
        ic = 0.0

    expression = _build_expr(raw, names, ic)
    return {"applicable": True, "r2": round(r2, 6),
            "coef": raw.round(4).tolist(),
            "intercept": round(ic, 6),
            "expression": expression}


def _build_expr(raw, names, intercept):
    """把各特征按【加法】线性组合（矩阵乘法语义），而非相乘。"""
    terms = []
    for k in range(len(raw)):
        c = float(raw[k])
        if abs(c) < 1e-9:
            continue
        a = snap_value(c)
        # 仅当吸附值仍接近原系数才使用有理系数，否则保留3位
        use = a if abs(a - c) < 5e-3 else round(c, 3)
        terms.append((use, names[k]))

    def piece(u, n):
        return n if abs(abs(u) - 1) < 1e-9 else f"{_fmt(abs(u))}*{n}"

    if not terms:
        body = "0"
    else:
        u0, n0 = terms[0]
        body = ("-" if u0 < 0 else "") + piece(u0, n0)
        for u, n in terms[1:]:
            body += ("+" if u > 0 else "-") + piece(u, n)

    if abs(intercept) > 1e-9:
        body += ("+" if intercept > 0 else "-") + _fmt(abs(intercept))
    return body


def _fmt(u):
    if abs(u - round(u)) < 1e-9:
        return str(int(round(u)))
    return str(round(u, 3))


# ---------------------------------------------------------------------------
# 窄字典稀疏加法求解（无 sqrt 和式等强装饰特征，标准化贪心 + 验证早停）
# ---------------------------------------------------------------------------

def _frac_str(u):
    """近似有理数就直接写成 (p/q)，否则保留高精度小数。"""
    from fractions import Fraction
    fr = Fraction(float(u)).limit_denominator(64)
    if abs(float(fr) - float(u)) < 1e-6 and fr.denominator > 1:
        return f"({fr.numerator}/{fr.denominator})"
    return f"{round(float(u), 9):g}"


def _narrow_feats(X):
    """窄特征字典：单项式(1–4次选择性) + 两两比 + 平方比 + 双因子分式 + 对数比值。"""
    n, d = X.shape
    F = {}
    for i in range(d):
        F[f"x{i}"] = X[:, i]
        F[f"x{i}**2"] = X[:, i] ** 2
        F[f"x{i}**3"] = X[:, i] ** 3
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            F[f"x{i}*x{j}"] = X[:, i] * X[:, j]
            F[f"x{i}**2*x{j}"] = X[:, i] ** 2 * X[:, j]
            F[f"x{i}*x{j}**2"] = X[:, i] * X[:, j] ** 2
            F[f"x{i}/x{j}"] = X[:, i] / (X[:, j] + 1e-12)
            F[f"x{i}**2/x{j}"] = X[:, i] ** 2 / (X[:, j] + 1e-12)
            for k in range(d):
                if k in (i, j):
                    continue
                F[f"x{i}*x{j}*x{k}"] = X[:, i] * X[:, j] * X[:, k]
                F[f"x{i}*x{j}**2*x{k}"] = X[:, i] * X[:, j] ** 2 * X[:, k]
                F[f"x{i}*x{j}/x{k}"] = X[:, i] * X[:, j] / (X[:, k] + 1e-12)
    if np.all(X > 0):
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                for k in range(d):
                    if k in (i, j):
                        continue
                    F[f"x{i}*log(x{j}/x{k})"] = X[:, i] * np.log(X[:, j] / X[:, k])
    names = list(F)
    M = np.stack([F[nm] for nm in names], axis=1)
    keep = np.all(np.isfinite(M), 0) & (M.std(0) > 1e-12)
    return M[:, keep], [names[k] for k in np.where(keep)[0]]


def _sparse_narrow_once(F, names, X, y, seed, max_terms=8, val_frac=0.3, val_tol=1e-6):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int((1.0 - val_frac) * n)
    tr, va = perm[:ntr], perm[ntr:]

    mu = F.mean(0)
    sd = F.std(0)
    sd[sd < 1e-12] = 1.0
    ym, ys = float(y.mean()), float(y.std()) + 1e-30
    Fz = (F - mu) / sd
    yz = (y - ym) / ys

    def fit(cols):
        A = np.concatenate([Fz[tr][:, cols], np.ones((len(tr), 1))], axis=1)
        c, _, _, _ = np.linalg.lstsq(A, yz[tr], rcond=None)
        return c

    def val_err(cols, c):
        A = np.concatenate([Fz[va][:, cols], np.ones((len(va), 1))], axis=1)
        return float(np.sqrt(np.mean((yz[va] - A @ c) ** 2)))

    chosen = []
    best_val, best_state = 1e18, None
    for _ in range(max_terms):
        curval = val_err(chosen, fit(chosen)) if chosen else 1e18
        bestk, bestv, bestc = None, curval, None
        for k in range(Fz.shape[1]):
            if k in chosen:
                continue
            c = fit(chosen + [k])
            v = val_err(chosen + [k], c)
            if v < bestv - 1e-12:
                bestv, bestk, bestc = v, k, c
        if bestk is None:
            break
        chosen.append(bestk)
        if bestv < best_val:
            best_val, best_state = bestv, (list(chosen), bestc)
        if bestv < val_tol:
            break

    if best_state is None:
        return None
    chosen, c = best_state

    # 后向剔除：删掉"删了反而让验证误差更小"的装饰项（反复直到不能再降）
    while len(chosen) > 1:
        base = val_err(chosen, c)
        cand, candv = None, base
        for pos in range(len(chosen)):
            trial = chosen[:pos] + chosen[pos + 1:]
            c2 = fit(trial)
            v = val_err(trial, c2)
            if v < candv - 1e-12:
                candv, cand = v, (trial, c2)
        if cand is None:
            break
        chosen, c = cand
        best_val = candv

    raw = c[:-1] * ys / sd[chosen]
    ic = ym - ys * float((mu[chosen] / sd[chosen]) @ c[:-1])
    sub_names = [names[k] for k in chosen]

    mx = max((abs(u) for u in raw), default=1.0)
    body = ""
    for u, nm in zip(raw, sub_names):
        if abs(u) < 1e-6 * mx:
            continue
        body += ("-" if u < 0 else ("+" if body else "")) + _frac_str(abs(u)) + "*" + nm
    if abs(ic) > 1e-6 * (np.mean(np.abs(y)) + 1e-12):
        body += ("+" if ic > 0 else "-") + f"{abs(ic):.6g}"
    if not body:
        return None

    from .util import nrmse_score
    return {"expression": body, "features": sub_names,
            "coef": np.round(raw, 6).tolist(), "intercept": round(ic, 8),
            "val_nrmse": round(best_val, 8),
            "fit_nrmse": round(float(nrmse_score(body, X, y)), 8),
            "n_terms": len(chosen)}


def solve_sparse_narrow(X, y, seed=0, max_terms=8, val_frac=0.3, val_tol=1e-6,
                        n_restarts=6):
    """窄字典 + 标准化空间残差贪心 + 留出验证早停 + 多起点重采样。

    多起点：换若干随机划分各跑一次，取"全数据 NRMSE 最低"的候选，
    缓解单次划分导致的贪心路径依赖（如 idx8）。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    F, names = _narrow_feats(X)
    if F.shape[1] == 0:
        return {"applicable": False, "reason": "无有效特征"}

    best = None
    for r in range(max(1, n_restarts)):
        res = _sparse_narrow_once(F, names, X, y, seed + r,
                                  max_terms=max_terms, val_frac=val_frac,
                                  val_tol=val_tol)
        if res is None:
            continue
        if best is None or res["fit_nrmse"] < best["fit_nrmse"]:
            best = res
    if best is None:
        return {"applicable": False, "reason": "所有起点均无候选"}
    return {"applicable": True, **best}


# ---------------------------------------------------------------------------
# 有理函数路线：y = P(x) / Q(x)，P 分子、Q=1-Σ b·ψ 分母，联合稀疏求解
# 原理：y·Q = P  =>  y = P + y·(Q-1) = P + y·Σ b·ψ
#       广增特征 G = [分子字典 D | y·ψ]，对 y 做稀疏线性拟合得 [a; b]。
#       选中 b 非零 → Q=1-Σ b·ψ，重构 y=P/(1-Σ b·ψ)，有理化系数。
# ---------------------------------------------------------------------------

def _linear_psi(X):
    """分母基 ψ：线性 x_i + 双线性 x_i*x_j（用于 Q=1-Σ b·ψ）。"""
    n, d = X.shape
    cols, names = [], []
    for i in range(d):
        cols.append(X[:, i]); names.append(f"x{i}")
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j]); names.append(f"x{i}x{j}")
    return np.column_stack(cols) if cols else np.zeros((n, 0)), names


def solve_rational(X, y, seed=0, max_terms=8, val_frac=0.3, val_tol=1e-6,
                   n_restarts=4):
    """有理函数拟合。返回 {expression, features, coef, ...} 或 applicable=False。"""
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, d = X.shape
    D, dnames = _narrow_feats(X)                       # 分子字典 [n,m]
    PSI, psnames = _linear_psi(X)                      # 分母基 [n,k]
    if PSI.size == 0:
        return {"applicable": False, "reason": "无分母基"}
    G = np.hstack([D, y[:, None] * PSI])               # [n, m+k]
    names = dnames + [f"y*{p}" for p in psnames]

    mu = G.mean(0); sd = G.std(0); sd[sd < 1e-12] = 1.0
    ym, ys = float(y.mean()), float(y.std()) + 1e-30
    Gz = (G - mu) / sd; yz = (y - ym) / ys

    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    ntr = int((1.0 - val_frac) * n); tr, va = perm[:ntr], perm[ntr:]

    def fit(cols):
        A = np.concatenate([Gz[tr][:, cols], np.ones((len(tr), 1))], axis=1)
        c, *_ = np.linalg.lstsq(A, yz[tr], rcond=None)
        return c

    def val_err(cols, c):
        A = np.concatenate([Gz[va][:, cols], np.ones((len(va), 1))], axis=1)
        return float(np.sqrt(np.mean((yz[va] - A @ c) ** 2)))

    best = None
    for r in range(n_restarts):
        chosen, c = _rational_greedy(Gz, yz, tr, va, names, seed + r,
                                     max_terms, val_tol)
        if chosen is None:
            continue
        # 需至少一个分母项(b非零)才成"有理式"
        m = D.shape[1]
        b_idx = [k for k in chosen if k >= m]
        if not b_idx:
            continue
        raw = c[:-1] * ys / sd[chosen]
        ic_raw = c[-1]
        ic = ym - ys * float(c[-1]) if False else ym + (c[-1] * ys if False else 0.0)
        # 系数截距处理：标准化 OLS 的截距 c[-1]（yz 空间）= (ym_hat-ym)/ys → 原 ic
        ic = ym - ys * float(c[-1])
        coef = dict(zip(chosen, raw))
        Pc = {k: v for k, v in coef.items() if k < m}
        Bc = {k: v for k, v in coef.items() if k >= m}
        expr = _rational_build(Pc, Bc, names, dnames, psnames, ic, m)
        if expr is None:
            continue
        from .util import nrmse_score
        nr = float(nrmse_score(expr, X, y))
        cand = {"expression": expr, "fit_nrmse": nr,
                "n_terms": len(chosen), "b_terms": len(Bc)}
        if best is None or cand["fit_nrmse"] < best["fit_nrmse"]:
            best = cand
    if best is None:
        return {"applicable": False, "reason": "无有效有理候选"}
    return {"applicable": True, **best}


def _rational_greedy(Gz, yz, tr, va, names, seed, max_terms, val_tol):
    """标准化空间残差贪心 + 验证早停（返回 chosen, coef）。"""
    rng = np.random.default_rng(seed)

    def fit(cols):
        A = np.concatenate([Gz[tr][:, cols], np.ones((len(tr), 1))], axis=1)
        c, *_ = np.linalg.lstsq(A, yz[tr], rcond=None)
        return c

    def val_err(cols, c):
        A = np.concatenate([Gz[va][:, cols], np.ones((len(va), 1))], axis=1)
        return float(np.sqrt(np.mean((yz[va] - A @ c) ** 2)))

    chosen = []; best_val, best_state = 1e18, None
    for _ in range(max_terms):
        curval = val_err(chosen, fit(chosen)) if chosen else 1e18
        bestk, bestv, bestc = None, curval, None
        for k in range(Gz.shape[1]):
            if k in chosen:
                continue
            c = fit(chosen + [k]); v = val_err(chosen + [k], c)
            if v < bestv - 1e-12:
                bestv, bestk, bestc = v, k, c
        if bestk is None:
            break
        chosen.append(bestk)
        if bestv < best_val:
            best_val, best_state = bestv, (list(chosen), bestc)
        if bestv < val_tol:
            break
    if best_state is None:
        return None, None
    chosen, c = best_state
    # 后向剔除：删掉"删了验证误差更小"的项
    while len(chosen) > 1:
        base = val_err(chosen, c)
        cand2, candv = None, base
        for pos in range(len(chosen)):
            trial = chosen[:pos] + chosen[pos + 1:]
            c2 = fit(trial); v = val_err(trial, c2)
            if v < candv - 1e-12:
                candv, cand2 = v, (trial, c2)
        if cand2 is None:
            break
        chosen, c = cand2
    return chosen, c


def _rational_build(Pc, Bc, names, dnames, psnames, ic, m):
    """由分子字典系数 Pc、分母基系数 Bc 重构 P/Q。"""
    from pathlib import Path
    # 分子 P
    num_terms = []
    for k, v in Pc.items():
        if abs(v) < 1e-6:
            continue
        num_terms.append((v, dnames[k]))
    # 分母：Q = 1 - Σ b·ψ
    den_terms = []
    for k, v in Bc.items():
        if abs(v) < 1e-6:
            continue
        den_terms.append((v, psnames[k - m]))
    if not den_terms:
        return None

    def piece(terms):
        body = ""
        for u, nm in terms:
            body += ("-" if u < 0 else ("+" if body else "")) + _frac_str(abs(u)) + "*" + nm
        return body

    num = piece(num_terms)
    if not num:
        num = "0"
    den = "1"
    for u, nm in den_terms:
        # Q = 1 - Σ b·ψ
        sign = "-" if u > 0 else "+"
        den += sign + _frac_str(abs(u)) + "*" + nm
    if abs(ic) > 1e-6:
        num += ("+" if ic > 0 else "-") + f"{abs(ic):.6g}"
    return f"({num})/({den})"
