"""SCE-SR：无真解的结构化符号回归求解器。

公共 API：
    solve_union(X, y, seed=0) -> dict
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        返回 {"route": str, "expression": str, "nrmse": float, ...}。

    gen_problem(idx, n=400, sigma=0.0, seed=0) -> dict
        从 MF-bench 的公式生成一个基准问题（仅用于评测/演示）。
"""
from .solver import solve_union
from .data import gen

__all__ = ["solve_union", "gen"]
__version__ = "0.1.0"
