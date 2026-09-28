# CanonicalSR 算法伪代码

> 对应源码：`src/scesr/solver.py`（统一入口）、`logspace.py`、`sparse.py`、
> `recursive.py`、`util.py`。
> 记号：输入 `X ∈ R^{n×d}`（n 个样本、d 个变量），输出 `y ∈ R^n`；
> 全程不访问真式、OSS、TSS、PSC。

---

## 主算法　`CanonicalSR(X, y)`

```
输入：数据矩阵 X[n,d]，目标 y[n]；种子 seed
输出：候选 e*（表达式）、命中路线、NRMSE

C ← ∅                              # 候选池

# ---------- 便宜路线（确定性，解析解）----------
for route ∈ {logspace, logdiff, linear_denom}:
    r ← route(X, y)
    if r.applicable:
        C ← C ∪ {(r.expression, route)}

# 短路：便宜路线已给出机器精度解 → 直接在其中取算子最少者
if ∃ c ∈ C with NRMSE(c) < 1e-6:
    e* ← argmin_{c ∈ C, NRMSE(c)<1e-6}  ops(c)
    return e*

# ---------- 窄字典稀疏路线（多起点）----------
n_restarts ← 6  if d ≤ 4
           ← 3  if 5 ≤ d ≤ 6
           ← 1  otherwise
b* ← ∅
for k = 0 .. n_restarts-1:
    b ← NarrowOnce(X, y, seed + k)
    if b ≠ ∅ and (b* = ∅ or NRMSE(b) < NRMSE(b*)):
        b* ← b                    # 只保留全数据 NRMSE 最低的起点
C ← C ∪ {b*}

# ---------- 残差递归兜底 ----------
r ← Recursive(X, y)
if r.applicable:
    C ← C ∪ {(r.expression, r.route)}

# ---------- 统一裁决 ----------
E ← { c ∈ C : NRMSE(c) < 1e-4 }    # 近似精确候选
if E ≠ ∅:
    e* ← argmin_{c ∈ E}  ( ops(c), NRMSE(c) )     # 先算子最少，误差次之
else:
    e* ← argmin_{c ∈ C}  NRMSE(c)                # 都不精确 → 误差最小
return e*
```

其中

```
NRMSE(e) = sqrt( (1/n) Σ_i ( e(x_i) − y_i )² ) / std(y)
ops(e)   = count_operators(e) + 1                 # 算子计数（复杂度）
```

---

## 路线 1　`Logspace(X, y)`　纯乘除幂

```
要求：X > 0 且 y > 0，否则 return not-applicable
Z ← log X ;  v ← log y
(p_1..p_d, b) ← OLS 解  v = b + Σ_k p_k Z[:,k]
p_k ← Snap(p_k) 到网格 G = {-2,-1,-1/2,0,1/2,1,2}
c   ← exp(b)
e   ← c · Π_k x_k^{p_k}
return applicable if (吸附偏差 ≤ 0.10 and log 空间 R² ≥ 0.98)
```

## 路线 2　`Logdiff(X, y)`　乘幂 × 差分幂

```
构造列：  log x_k        （对正变量）
         log(x_i − x_j)  （对恒正差；与 log x_i/log x_j 相关 > 0.999 则跳过：共线守卫）
(p, b) ← 上述列上 OLS 解 log y
p ← 吸附到 1/4 网格
c ← 识别常数 exp(b) ∈ {π, e, sqrt(2), k/√π, …}
e ← c · Π x_k^{p_k} · Π (x_i−x_j)^{q}
y 含正负号时：在 log|y| 上求解，奇整数幂差底按与 sign(y) 的相关定向（带符号版本）
return applicable if (吸附偏差 ≤ 0.06 and R² ≥ 0.999)
```

## 路线 3　`LinearDenom(X, y)`　线性分母

```
for i = 0 .. d-1:
  for a ∈ 有理网格 (18 个值, 如 ±1/8,±1/4,±1/2,±3/4,±1,±3/2,±2,±4):
      r ← y · (1 + a x_i)
      s ← Logdiff(X, r)
      if s.applicable and s 为干净单项式:
          e ← s.expression / (1 + a x_i)
          记录 e
return 其中（通过数据 NRMSE 自检的）算子最少者；无则 not-applicable
```

## 路线 4　`NarrowOnce(X, y, seed)`　窄字典稀疏加法

```
Φ ← 窄字典：{x_i, x_i², x_i³, x_i x_j, x_i/x_j, x_i²/x_j,
            x_i x_j/x_k }；X>0 时加入 x_i·log(x_j/x_k)
标准化：Φz ← (Φ−mean)/std ;  yz ← (y−mean)/std
随机划分（由 seed）：训练 70% / 验证 30%
S ← ∅
repeat:
    k* ← argmax 下降 验证误差 的字典列          # 前向贪心
    S ← S ∪ {k*}
    记录验证误差最低状态
until 验证误差 < 1e-6 或 |S| = 8
# 后向剔除
repeat:
    若存在 j ∈ S 使 删除 j 后验证误差更小：S ← S \\ {j}
until 不能再降
系数还原到原始尺度并吸附为有理数
e ← Σ_k c_k Φ_k（+ 截距，仅当显著）
return e
```

## 路线 5　`Recursive(X, y)`　残差剥层 + exp/log 外壳

```
A ← 原子库：{x_i, log x_i, 1/x_i, x_i x_j, x_i/x_j, x_i², x_i²x_j, 1±x_i/x_j}
（direct）
e ← 在 A 上前向贪心：每步加入“吸附系数仍非零且 SSE 显著下降”的原子
若 R²(e) ≥ 0.9999：return e, route = direct
（exp 外壳：y 全正时）
g ← 在 log y 上重跑上述贪心
若成功：return exp(g), route = expwrap
否则 not-applicable
```

---

## 关键性质（伪代码可直接读出）

1. **label-blind**：除 `(X, y)` 外无任何输入；误差与算子计数是仅有的决策量。
2. **精确优先，同级取简**：简洁性只在 NRMSE 同处一个精度带时起作用，
   绝不以牺牲拟合换简洁。
3. **可恢复性边界**：目标落在五条路线模型族的并集（指数在网格上）内
   ⇔ 可严格恢复；族外问题显式落入“无精确候选”。
