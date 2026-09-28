"""痛点对照图（真实 iMCTS）：拟合满分但结构错误 vs CanonicalSR 结构正确。

真实测试公式：MF-bench idx3
  真式在数据定义域（恒有 x2>x1）内化简为：f = x0*(x2 - 2*x1)
iMCTS 真实输出（400 点，60000 evals，seed0）：
  g = x0*x2 + cos(x0)*x0 + x2 - 2*x0*x1
  归一化 reward≈1.000（NRMSE=3.4e-6），但含装饰项，SE=False。
CanonicalSR 真实输出：h = x0*(x2-2*x1)，结构正确。

左图用真实数据尺度（y~1e11）：装饰项~1e6 被归一化吞掉，三曲线肉眼重合。
右上/右下：用放大（residual）尺度，结构差异才显形。

输出：results/figures/compare_imcts.png/.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scesr import gen

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "figures"


def main():
    g = gen(3, n=400, sigma=0.0, seed=0)
    X = np.asarray(g["X"]).T.astype(float)
    y = np.asarray(g["yc"], float)

    im = X[:, 0] * X[:, 2] + np.cos(X[:, 0]) * X[:, 0] + X[:, 2] \
        - 2 * X[:, 0] * X[:, 1]
    ours = X[:, 0] * (X[:, 2] - 2 * X[:, 1])

    order = np.argsort(y)
    idx = np.arange(len(y))

    fig = plt.figure(figsize=(19, 9.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1.7],
                          height_ratios=[0.30, 1.0, 0.30, 1.0],
                          hspace=0.55, wspace=0.24)

    # ============ 左：真实尺度（肉眼重合）============
    axL = fig.add_subplot(gs[:, 0])
    axL.plot(idx, y[order], color="black", lw=2.2, label="True formula")
    axL.plot(idx, im[order], "--", color="#e74c3c", lw=1.7, alpha=0.5,
             label="iMCTS")
    axL.plot(idx, ours[order], ":", color="#27ae60", lw=2.0, alpha=0.6,
             label="CanonicalSR")
    axL.scatter(idx, y[order], s=8, color="#2c3e50", alpha=0.5,
                label="Observed data")
    axL.set_xlabel("samples (ordered by $y$)", fontsize=11)
    axL.set_ylabel("$y$", fontsize=11)
    axL.tick_params(labelsize=10)
    axL.set_title("On the real data scale ($y\\sim10^{11}$): all three overlap",
                  fontsize=12)
    axL.text(0.03, 0.05,
             "max |iMCTS - true| $\\approx1.9\\times10^{6}$\n"
             "relative to $y\\sim10^{11}$: NRMSE$=3.4\\times10^{-6}$\n"
             "reward$\\approx1.000$ — but iMCTS structure is wrong",
             transform=axL.transAxes, fontsize=11, color="#34495e", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7",
                       alpha=0.9))
    axL.legend(loc="upper left", fontsize=10.5, frameon=False)

    # ============ 右上文字带 ============
    axBW = fig.add_subplot(gs[0, 1]); axBW.axis("off")
    axBW.text(0.0, 0.92,
              "iMCTS — normalized reward ~1.000 (perfect fit, wrong structure)",
              fontsize=11.5, color="#922b21", va="top", weight="bold")
    axBW.text(0.0, 0.08,
              r"$\hat g=x_0x_2+\cos(x_0)x_0+x_2-2x_0x_1$" "\n"
              "decoration terms: $\\cos(x_0)\\,x_0+x_2$   $\\Rightarrow$ SE=False",
              fontsize=11, color="#922b21", va="bottom")
    axBW.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axBW.transAxes,
                                 fc="#fdedec", ec="#c0392b", lw=1.2, zorder=-1))

    # ============ 右上：放大尺度（错误）============
    axW = fig.add_subplot(gs[1, 1])
    d_im = im[order] - y[order]
    axW.plot(idx, np.zeros_like(idx), color="black", lw=1.9, label="true (zero residual)")
    axW.plot(idx, d_im, color="#c0392b", lw=1.6, label="iMCTS residual")
    axW.set_ylabel("residual", fontsize=11)
    axW.tick_params(labelsize=10)
    axW.legend(loc="upper right", fontsize=10, frameon=False)
    plt.setp(axW.get_xticklabels(), visible=False)
    axW.set_title("Zoomed scale: iMCTS carries a nonzero structural offset",
                  color="#922b21", fontsize=12)

    # ============ 右下文字带 ============
    axBO = fig.add_subplot(gs[2, 1]); axBO.axis("off")
    axBO.text(0.0, 0.92,
              "CanonicalSR — label-blind multi-route canonical recovery",
              fontsize=11.5, color="#1e8449", va="top", weight="bold")
    axBO.text(0.0, 0.08,
              r"$\hat h=x_0(x_2-2x_1)$" "\n"
              "no decoration terms   $\\Rightarrow$ structure exact (SE)",
              fontsize=11, color="#1e8449", va="bottom")
    axBO.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axBO.transAxes,
                                 fc="#eafaf1", ec="#1e8449", lw=1.2, zorder=-1))

    # ============ 右下：放大尺度（正确）============
    axO = fig.add_subplot(gs[3, 1], sharex=axW)
    d_ou = ours[order] - y[order]
    axO.plot(idx, np.zeros_like(idx), color="black", lw=1.9, label="true")
    axO.plot(idx, d_ou, ":", color="#1e8449", lw=2.6,
             label="CanonicalSR residual")
    axO.set_xlabel("samples (ordered by $y$)", fontsize=11)
    axO.set_ylabel("residual", fontsize=11)
    axO.tick_params(labelsize=10)
    axO.legend(loc="upper right", fontsize=10, frameon=False)
    axO.set_title("Zoomed scale: CanonicalSR residual is zero",
                  color="#1e8449", fontsize=12)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"compare_imcts.{ext}", dpi=200, bbox_inches="tight")
    print("->", OUT / "compare_imcts.png")
    print("->", OUT / "compare_imcts.pdf")


if __name__ == "__main__":
    main()
