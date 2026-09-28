"""痛点图：拟合导向候选 vs CanonicalSR 规范候选。

MF-Bench 揭示的痛点：R² 很高的候选可能结构错误。
用 MF-bench 指数族（idx2 x0*exp(x3(x2-x1)) 的一维形式）展示：
  - 拟合导向候选（高次多项式，纯按训练误差）：窗内几乎完美，窗外发散；
  - CanonicalSR（恢复指数规范式 e^{2x}）：全局与真式重合。
左面板：训练窗（结构不可辨识）；右面板：外推（断轴显示多项式发散）。

输出：results/figures/painpoint.png / .pdf
用法：python experiments/make_painpoint.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "figures"

A, B = 0.0, 0.6
N, DEG, K = 18, 10, 2.0
NOISE_FRAC = 2e-3
SEED = 3


def main():
    rng = np.random.default_rng(SEED)
    x = np.linspace(A, B, N)
    y0 = np.exp(K * x)
    y = y0 + rng.normal(0, NOISE_FRAC * y0.std(), N)
    coef = np.polyfit(x, y, DEG)

    xe = np.linspace(-1.5, 1.8, 700)
    true = np.exp(K * xe)
    fit = np.polyval(coef, xe)
    canon = true.copy()

    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2])

    # ---- 左：训练窗 ----
    m = (xe >= A) & (xe <= B)
    ax0.plot(xe[m], true[m], color="black", lw=2.2,
             label="True $f(x)=e^{2x}$")
    ax0.plot(xe[m], fit[m], "--", color="#c0392b", lw=2,
             label="Fit-oriented: deg-10 poly")
    ax0.scatter(x, y, s=22, color="black", zorder=5)
    ax0.axvspan(A, B, color="#3498db", alpha=0.07)
    ax0.set_xlim(A - 0.02, B + 0.02)
    ax0.set_ylim(-0.2, 3.4)
    ax0.set_title("In-sample: both fit nearly\nperfectly (unidentifiable)")
    ax0.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax0.set_xlabel("x"); ax0.set_ylabel("y")

    # ---- 中：外推近轴（真式 + CanonicalSR 重合）----
    ax1.plot(xe, true, color="black", lw=2.2, label="True")
    ax1.plot(xe, canon, ":", color="#27ae60", lw=2.6,
             label="CanonicalSR: $e^{2x}$")
    ax1.plot(xe, fit, "--", color="#c0392b", lw=1.6,
             label="Fit-oriented poly (leaves axis)")
    ax1.axvspan(A, B, color="#3498db", alpha=0.07)
    ax1.set_xlim(-1.5, 1.8)
    ax1.set_ylim(-0.3, 3.3)
    ax1.set_title("Extrapolation (zoom on true scale)")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax1.set_xlabel("x")
    plt.setp(ax1.get_yticklabels(), visible=False)

    # ---- 右：发散全貌（log|fit|）----
    ax2.plot(xe, np.abs(fit), "--", color="#c0392b", lw=2,
             label="|fit-oriented poly|")
    ax2.plot(xe, true, color="black", lw=2, label="|true| = $e^{2x}$")
    ax2.axvspan(A, B, color="#3498db", alpha=0.07)
    ax2.set_yscale("log")
    ax2.set_xlim(-1.5, 1.8)
    ax2.set_ylim(1e-2, 1e9)
    ax2.set_title("Extrapolation (log scale): decoration blows up")
    ax2.legend(loc="lower center", fontsize=8.5, frameon=False)
    ax2.set_xlabel("x")

    fig.suptitle("Nearly identical in-sample fit, fundamentally different scientific expressions",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"painpoint.{ext}", dpi=200, bbox_inches="tight")
    print("->", OUT / "painpoint.png")
    print("->", OUT / "painpoint.pdf")


if __name__ == "__main__":
    main()
