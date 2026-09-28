# Results

这些是从第 13 期实验迁移过来的原始测试结果（只读快照）。当前 `src/scesr` 求解器
只用 `(X,y)`；这些 JSON 是评测期对照真式得到的指标记录。

| 文件 | 协议 | 内容 |
|---|---|---|
| `first20_sig0.json` | first-20（idx 1..20）, n=400, σ=0, seed=0 训练 + seed=7 独立复核 | iMCTS vs SCE-SR 的逐条 R²/OSS/TSS/PSC/SE 与表达式 |
| `first20_sig0.05.json` | 同上，σ=0.05 | 含噪结果（暴露噪声鲁棒缺口） |
| `feynman15_sig0.jsonl` | Feynman15 子集, n=400, σ=0 | 跨物理域评测 |
| `paper20_radar5_sig0.json` | PAPER20（分层抽样的 20 条）, n=400, σ=0, 5 seed | 雷达五轴 iMCTS/ours/兜底逐条记录 |
| `core_first20.json` | first-20, n=400, σ=0, float64, seed0/7 | 第14期复现：SE 17/20、fresh 18/20 |
| `ablation_first20.json` | 同池三档：仅误差/+简洁性/完整多路线 | 消融：SE 9 → 15 → 17 |
| `heldout20.json` | 20 条**未参与设计**公式 + 5 公式×3 种子稳定性 | 跨公式泛化 8/20（40%） |
| `case284.json` | idx284 断裂韧度 | 定义域/恢复/5 倍外推（~1e-16） |

协议与结论见上级目录 `REPORT.md`。

墙报配套（伪代码见 `experiments/PSEUDOCODE.md`）：

| 路径 | 内容 |
|---|---|
| `figures/painpoint.png/.pdf` | 痛点图：窗内同拟合、窗外结构分化 |
| `tables/core_results.md/.csv` | first-20 核心结果（SE 17/20） |
| `tables/ablation.md/.csv` | 消融 9→15→17 |
| `tables/heldout20.md/.csv` | 未接触公式（8/20） |

由 `experiments/make_painpoint.py`、`experiments/make_tables.py` 重新生成。

指标含义：
- `r2` 数值拟合；`oss` 算子相似度；`tss` 树结构相似度；
- `psc` 物理结构共现；`se` 严格代数等价（struct）。

## 头条结论（σ=0, n=400, first-20）

| 指标 | iMCTS | SCE-SR |
|---|---|---|
| R² | 0.975 | 0.992 |
| OSS | 0.819 | 0.887 |
| TSS | 0.739 | 0.784 |
| PSC | 0.841 | 0.905 |
| SE | 0.300（6/20） | **0.850（17/20）** |
