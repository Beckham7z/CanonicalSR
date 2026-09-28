"""真实公式对照图：数据点+真实曲线（含两种方法淡色曲线）| 错误方法 | 我们的方法。

真实测试公式：MF-bench idx2   f = x0 * exp(x3*(x2-x1))
一维观测截面（固定 x0=1, x1=0, x2=2）：f(x3) = exp(2 x3)
局部观测窗 [0, 0.6]，18 点，0.2% 微噪声。

错误方法（具体）：无约束的 10 次最小二乘多项式回归——只最小化窗内 SSE，
没有结构模型；其回归式由本次数据实算：
  -1.094e4 x^10 + 2.952e4 x^9 - 3.128e4 x^8 + 1.516e4 x^7 - 1827 x^6
  - 1473 x^5 + 749.5 x^4 - 147.2 x^3 + 15.64 x^2 + 1.504 x + 1.003
窗内近完美，窗外立即偏离。

我们的方法：CanonicalSR（label-blind 多路线规范恢复），solve_union 真实输出
  exp(log x0 + x2 x3 - x1 x3)，截面 = exp(2 x3)，与真式全局重合。

布局（方法信息放在绘图区外的文字带，不遮挡曲线）：
  左（跨全部）：真实曲线 + 数据点 + 两种方法的淡色曲线
  右上：错误方法信息带 + 错误拟合放大
  右下：我们的方法信息带 + 正确拟合放大

输出：results/figures/compare_panel.png / .pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from scesr import gen, solve_union
from common import eval_expr

OUT = HERE.parent / "results" / "figures"

A, B = 0.0, 0.6
N, DEG, NOISE_FRAC, SEED = 18, 10, 2e-3, 3
XG_LO, XG_HI = -0.4, 1.0
GLOB_LO, GLOB_HI = -1.4, 1.6

WRONG_POLY = (r"$\hat f=-1.094{\times}10^{4}x^{10}+2.952{\times}10^{4}x^{9}"
              r"-3.128{\times}10^{4}x^{8}+1.516{\times}10^{4}x^{7}"
              r"-1827x^{6}-1473x^{5}+749.5x^{4}-147.2x^{3}"
              r"+15.64x^{2}+1.504x+1.003$")
OURS_POLY = r"$\hat f=e^{\log x_0\,+\,x_2x_3\,-\,x_1x_3}\ \Rightarrow\ \mathrm{slice}\ \hat f=e^{2x_3}$"


def main():
    g = gen(2, n=400, sigma=0.0, seed=0)
    X4 = np.asarray(g["X"]).T.astype(float)
    y4 = np.asarray(g["yc"], float)
    ours_expr = solve_union(X4, y4)["expression"]

    rng = np.random.default_rng(SEED)
    x = np.linspace(A, B, N)
    y0 = np.exp(2 * x)
    y = y0 + rng.normal(0, NOISE_FRAC * y0.std(), N)
    coef = np.polyfit(x, y, DEG)

    xg = np.linspace(XG_LO, XG_HI, 600)
    true_g = np.exp(2 * xg)
    wrong_g = np.polyval(coef, xg)
    M = np.column_stack([np.full_like(xg, 1.0), np.full_like(xg, 0.0),
                        np.full_like(xg, 2.0), xg])
    ours_g = eval_expr(ours_expr, M)

    xall = np.linspace(GLOB_LO, GLOB_HI, 800)
    true_all = np.exp(2 * xall)
    wrong_all = np.clip(np.polyval(coef, xall), -60, 60)
    Mall = np.column_stack([np.full_like(xall, 1.0), np.full_like(xall, 0.0),
                           np.full_like(xall, 2.0), xall])
    ours_all = eval_expr(ours_expr, Mall)

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1.7],
                          height_ratios=[0.20, 1.0, 0.20, 1.0],
                          hspace=0.16, wspace=0.18)

    # ============ 左：真实曲线 + 数据点 + 两方法淡色曲线 ============
    axL = fig.add_subplot(gs[:, 0])
    axL.plot(xall, true_all, color="black", lw=2.2, label="True formula")
    axL.plot(xall, wrong_all, "--", color="#e74c3c", lw=1.6, alpha=0.4,
             label="Fit-only degree-10 polynomial")
    axL.plot(xall, ours_all, ":", color="#27ae60", lw=2.0, alpha=0.55,
             label="CanonicalSR recovery")
    axL.scatter(x, y, s=24, color="#2c3e50", zorder=6, label="Observed data")
    axL.axvspan(A, B, color="#3498db", alpha=0.07)
    axL.text((A + B) / 2, 21.3, "local observation window", ha="center",
             fontsize=9, color="#7f8c8d")
    axL.set_xlim(GLOB_LO, GLOB_HI)
    axL.set_ylim(-2, 24)
    axL.set_xlabel("$x_3$  (other variables fixed)")
    axL.set_ylabel("$y$")
    axL.set_title("True formula, observed data, and both methods")
    axL.legend(loc="upper left", fontsize=9, frameon=False)

    # ============ 右上信息带（绘图区外）============
    axBW = fig.add_subplot(gs[0, 1]); axBW.axis("off")
    axBW.text(0.0, 0.82,
              "FIT-ONLY METHOD — unconstrained degree-10 least-squares polynomial\n"
              "(minimizes in-window SSE only; no structural model)",
              fontsize=9.5, color="#922b21", va="top", ha="left", weight="bold")
    axBW.text(0.0, 0.02, WRONG_POLY, fontsize=8.6, color="#922b21",
              va="bottom", ha="left")
    axBW.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axBW.transAxes,
                                 fc="#fdedec", ec="#c0392b", lw=1.2,
                                 zorder=-1))

    # ============ 右上：错误拟合放大 ============
    axW = fig.add_subplot(gs[1, 1])
    axW.plot(xg, true_g, color="black", lw=1.8, label="true")
    axW.plot(xg, wrong_g, color="#c0392b", lw=2, label="fit-only polynomial")
    axW.scatter(x, y, s=14, color="black", zorder=6)
    axW.axvspan(A, B, color="#3498db", alpha=0.07)
    axW.set_xlim(XG_LO, XG_HI)
    axW.set_ylim(-3, 9)
    axW.legend(loc="upper left", fontsize=8.5, frameon=False)
    plt.setp(axW.get_xticklabels(), visible=False)

    # ============ 右下信息带 ============
    axBO = fig.add_subplot(gs[2, 1]); axBO.axis("off")
    axBO.text(0.0, 0.82,
              "CANONICALSR — label-blind multi-route canonical recovery\n"
              "(near-exact fit first; among equally exact, fewest operators)",
              fontsize=9.5, color="#1e8449", va="top", ha="left", weight="bold")
    axBO.text(0.0, 0.02, OURS_POLY, fontsize=9, color="#1e8449",
              va="bottom", ha="left")
    axBO.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axBO.transAxes,
                                 fc="#eafaf1", ec="#1e8449", lw=1.2,
                                 zorder=-1))

    # ============ 右下：正确拟合放大 ============
    axO = fig.add_subplot(gs[3, 1], sharex=axW)
    axO.plot(xg, true_g, color="black", lw=1.8, label="true")
    axO.plot(xg, ours_g, ":", color="#1e8449", lw=2.6,
             label="CanonicalSR recovery")
    axO.scatter(x, y, s=14, color="black", zorder=6)
    axO.axvspan(A, B, color="#3498db", alpha=0.07)
    axO.set_ylim(-3, 9)
    axO.set_xlabel("$x_3$")
    axO.legend(loc="upper left", fontsize=8.5, frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"compare_panel.{ext}", dpi=200, bbox_inches="tight")
    print("->", OUT / "compare_panel.png")
    print("->", OUT / "compare_panel.pdf")


if __name__ == "__main__":
    main()
