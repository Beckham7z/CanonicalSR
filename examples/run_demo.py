"""最小演示：对一条 MF-bench 公式，生成数据 -> solve_union 恢复 -> 打印结果。

用法（在仓库根目录）：
    pip install -e .
    python examples/run_demo.py            # 默认 idx=52（线性差）
    python examples/run_demo.py 9           # 指定 MF-bench 公式编号
"""
import sys

import numpy as np
from scesr import gen, solve_union


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 52
    d = gen(idx, n=400, sigma=0.0, seed=0)
    if d is None:
        print(f"idx{idx}: 该公式当前生成器不支持")
        return
    X = d["X"].T.astype(float)          # [n, d]
    y = np.asarray(d["y"], float)
    r = solve_union(X, y)
    print(f"problem idx={idx}, n={X.shape[0]}, d={X.shape[1]}")
    print(f"  true      : {d['true']}")
    print(f"  expression: {r.get('expression')}")
    print(f"  route     : {r.get('route')}   nrmse={r.get('nrmse'):.3e}")


if __name__ == "__main__":
    main()
