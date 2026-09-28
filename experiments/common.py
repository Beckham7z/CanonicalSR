"""实验公共件：协议常量、SE 判据、独立新数据 relerr、表达式求值。

所有实验脚本统一从这里取协议，保证口径一致、可复现。
不修改 src/ 求解器；求解器只吃 (X, y)。
"""
from __future__ import annotations

import numpy as np
import sympy as sp

N_SAMPLES = 400
SIGMA = 0.0
SEED_TRAIN = 0
SEED_FRESH = 7
EXACT_TOL = 1e-4

_NUM_NS = {"sqrt": np.sqrt, "log": np.log, "exp": np.exp, "sin": np.sin,
           "cos": np.cos, "tan": np.tan, "abs": np.abs, "pi": np.pi, "E": np.e,
           "max": np.maximum, "min": np.minimum, "tanh": np.tanh,
           "arcsin": np.arcsin, "asin": np.arcsin}


def structural_equivalent(pred: str, true: str, d: int, tol: float = 1e-6) -> bool:
    """与 12th metrics.structural_equivalent 同口径：先 simplify，再数值兜底。"""
    try:
        loc = {f"x{i}": sp.Symbol(f"x{i}", real=True) for i in range(d)}
        p, t = sp.sympify(pred, locals=loc), sp.sympify(true, locals=loc)
    except Exception:
        return False
    try:
        if sp.simplify(p - t) == 0:
            return True
    except Exception:
        pass
    try:
        syms = sp.symbols([f"x{i}" for i in range(d)], real=True)
        f = sp.lambdify(syms, sp.simplify(p - t), "numpy")
        rng = np.random.default_rng(123)
        pts = rng.uniform(0.1, 1.5, (30, d))
        vals = np.array([f(*pts[i]) for i in range(30)], dtype=complex)
        vals = vals[np.isfinite(vals)]
        return len(vals) > 0 and float(np.max(np.abs(vals))) < tol
    except Exception:
        return False


def eval_expr(pred: str, X: np.ndarray):
    """在 X[n,d] 上求值，失败返回 None。"""
    ns = {f"x{i}": X[:, i] for i in range(X.shape[1])}
    ns.update(_NUM_NS)
    try:
        with np.errstate(all="ignore"):
            v = np.asarray(eval(pred, {"__builtins__": {}}, ns), float)
        if v.ndim == 0:
            v = np.full(X.shape[0], float(v))
        if v.shape[0] != X.shape[0] or not np.all(np.isfinite(v)):
            return None
        return v
    except Exception:
        return None


def fresh_relerr(pred: str, gen_fn, idx: int, n: int = N_SAMPLES,
                 seed: int = SEED_FRESH) -> float:
    """独立新数据（不同种子）上的最大相对误差。"""
    g = gen_fn(idx, n=n, sigma=0.0, seed=seed)
    Xf = np.asarray(g["X"]).T.astype(float)
    yf = np.asarray(g["yc"] if "yc" in g else g["y"], float)
    yp = eval_expr(pred, Xf)
    if yp is None:
        return float("inf")
    return float(np.max(np.abs(yp - yf)) / (np.max(np.abs(yf)) + 1e-12))
