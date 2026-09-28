"""共享小工具：表达式安全求值、NRMSE、复杂度（供各路线与统一求解器使用）。"""
from __future__ import annotations

import numpy as np

_EVAL_NS = {"sqrt": np.sqrt, "log": np.log, "exp": np.exp,
            "sin": np.sin, "cos": np.sin and __import__("numpy").cos,
            "abs": np.abs, "pi": np.pi, "max": np.maximum,
            "min": np.minimum, "tan": np.tan,
            "arcsin": np.arcsin, "tanh": np.tanh}
# 上面 cos 的写法不直观，显式重建
_EVAL_NS["cos"] = np.cos
_EVAL_NS["asin"] = np.arcsin


def eval_on_X(expression, X):
    """在原始变量矩阵 X[n,d] 上安全计算表达式，返回预测 [n]；失败返回 None。"""
    if not expression:
        return None
    d = X.shape[1]
    ns = {f"x{k}": X[:, k] for k in range(d)}
    ns.update(_EVAL_NS)
    try:
        with np.errstate(all="ignore"):
            v = eval(expression, {"__builtins__": {}}, ns)
        v = np.asarray(v, float)
        if v.ndim == 0:
            v = np.full(X.shape[0], float(v))
        if v.shape[0] != X.shape[0] or not np.all(np.isfinite(v)):
            return None
        return v
    except Exception:
        return None


def nrmse_score(expression, X, y):
    """候选的真实拟合 NRMSE（越小越好）；不可用返回 inf。"""
    pred = eval_on_X(expression, X)
    if pred is None:
        return float("inf")
    return float(np.sqrt(np.mean((pred - y) ** 2)) / (np.std(y) + 1e-12))


def complexity(expr):
    """表达式算子/节点规模（越小越简）；解析失败给大值。"""
    try:
        import sympy as sp
        e = sp.sympify(expr, locals={"sqrt": sp.sqrt, "log": sp.log,
                                     "exp": sp.exp, "sin": sp.sin,
                                     "cos": sp.cos, "tan": sp.tan,
                                     "Abs": sp.Abs, "pi": sp.pi,
                                     "E": sp.E})
        return int(sp.count_ops(e, visual=False)) + 1
    except Exception:
        return 10 ** 6
