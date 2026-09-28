# SCE-SR — Structure-first Symbolic Regression

一个**只用数据 `(X, y)`** 的符号回归求解器：不看真解、不依赖公式库、不训练神经网络。
目标是恢复**结构正确、系数干净（有理/常用常数）**的闭式表达式，而不只是数值拟合好。

在干净数据上，SCE-SR 对 MF-bench 前 20 条公式的**严格结构恢复率为 17/20**，
对照基线 iMCTS 为 6/20；二者训练 R² 均≈0.98。

---

## 为什么需要它

经典 SR（遗传规划 / 强化学习 / MCTS）的奖励几乎只看拟合误差。结果是：
很多式子**训练 R²≈1，但结构是"装饰式"**——在小范围内数值对，换成独立数据就显著偏离。
SCE-SR 在多个**规范化的模型族**上求解，并用"**精确拟合优先、同样精确取最简**"来选择，
从而把数值等价但更繁的装饰式压下去。

## 五条恢复路线

| 路线 | 可表示的模型 | 方法 |
|---|---|---|
| `logspace` | 纯乘除幂 `c·Π x_k^{p_k}` | 取对数→线性回归→指数吸附到有理网格 |
| `logdiff` | 差分根式 `c·Πx^p·Π(x_i−x_j)^q` | 对数 + `log(x_i−x_j)` 原子 + 共线守卫 |
| `linear_denom` | 线性分母 `monomial/(1+a·x_i)` | 扫描有理 `a`，分子命中单项式即重建 |
| `narrow` | 稀疏加法（含比值/对数比值） | 窄字典 + 标准化贪心 + 验证早停 + 后向剔除 + 多起点 |
| `recursive` | 残差逐层剥离 + `exp/log` 外壳 | 原子库贪心 + 系数有理吸附 |

**统一选择规则**：
1. 先收集各路线候选，算其在数据上的真实 NRMSE；
2. 在"近似精确（NRMSE < 1e-4）"的候选里取**算子最少**的；
3. 若都不精确，取 NRMSE 最小的。

> 形式化地：SCE-SR 能精确恢复目标，**当且仅当**目标落在上述模型族的并集（且指数在网格内）。

## 复现实验 / 墙报结果

完整协议（代码版本、公式编号、种子、噪声定义、计算预算、选解规则）与
消融/跨公式泛化/详细案例见 **[REPORT.md](REPORT.md)**。

```bash
python experiments/reproduce_core.py   # 核心：first-20 SE 17/20
python experiments/ablation.py         # 消融：9 → 15 → 17
python experiments/heldout20.py        # 跨公式留出（未参与设计）：8/20
python experiments/case_284.py         # 断裂韧度详细案例
```

> 协议注意：精确路线需用 **float64 的干净 `yc`**；直接喂 float32 的 `y` 会因
> 舍入残差产生垃圾补偿项（详见 REPORT.md §2.2）。

## 安装

```bash
pip install -e .
```
依赖：`numpy`、`sympy`。

## 快速使用

```python
import numpy as np
from scesr import solve_union

# X: (n_samples, n_features)   y: (n_samples,)
r = solve_union(X, y)
print(r["expression"], r["route"], r["nrmse"])
```

命令行演示（需要 MF-bench 数据集用于生成示例）：

```bash
python examples/run_demo.py 52     # problem idx
```

用内置数据生成器造一个基准问题：

```python
from scesr import gen
d = gen(idx=52, n=400, sigma=0.0, seed=0)
X, y = d["X"].T, d["y"]
```

## 评测结果（干净数据 σ=0，n=400，MF-bench first-20）

| 指标 | iMCTS | **SCE-SR** |
|---|---|---|
| R² | 0.975 | **0.992** |
| OSS（算子相似度） | 0.819 | **0.887** |
| TSS（树结构相似度） | 0.739 | **0.784** |
| PSC（物理结构共现） | 0.841 | **0.905** |
| **SE（严格等价率）** | 0.300（6/20） | **0.850（17/20）** |
| 独立新数据严格等价 | 7/20 | **18/20** |

未恢复的问题集中在**深嵌套分式**（first-20 中为 idx 11、19）。

## 当前边界（诚实）

- **含噪鲁棒性是已知缺口**：σ=0.05 时严格恢复率明显下降；
  精确路线的"精确性门"在含噪下不触发。噪声鲁棒的稳健结构识别是下一步的主攻方向。
- 评测使用的 OSS/TSS/PSC/SE 在共享 `metrics` 中对照真式计算，**不进入求解过程**。

## 仓库结构

```
src/scesr/
  solver.py      # solve_union 统一入口
  logspace.py    # logspace / logdiff / linear_denom
  sparse.py      # narrow 稀疏加法
  recursive.py   # 残差递归
  data.py        # MF-bench 数据生成（custom 规格 + generic）
  gen_formula.py # generic 公式 AST 规范化
  util.py        # 安全求值 / NRMSE / 复杂度
examples/
  run_demo.py
```

## 许可

MIT
