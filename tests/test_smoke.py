"""冒烟测试：solve_union 在一条简单 MF-bench 公式上可运行并给出表达式。"""
import numpy as np

from scesr import gen, solve_union


def test_smoke_idx52():
    d = gen(52, n=200, sigma=0.0, seed=0)
    assert d is not None
    X = d["X"].T.astype(float)
    y = np.asarray(d["y"], float)
    r = solve_union(X, y)
    assert r.get("expression")
    assert r.get("nrmse", 1.0) < 1e-3
