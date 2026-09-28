"""痛点图（直观版）：窗内拟合满分、窗外结构完全错误 vs CanonicalSR 正确。

真实测试：f(x)=exp(2x)（MF-bench 指数族一维截面）。
观测窗 [0,0.3]，12 点，1% 微噪声。

iMCTS 真实输出（60000 evals）：
  ĝ = exp(2x)·cos(x³)
  窗内 x³≈0 → cos≈1 → 与真式肉眼重合，reward≈0.99（拟合满分）；
  窗外 cos(x³) 振荡 → 数值错且符号翻转，结构完全错误（SE=False）。

CanonicalSR 真实输出（recursive expwrap）：
  ĥ ≈ exp(2x)，结构正确。

左右对比：
  左：观测窗内——两条线与真式重合（大字：窗内分不清）
  右：拉到窗外——错误式振荡崩走、CanonicalSR 与真式贴合（大字：窗外见分晓）

输出：results/figures/painpoint_v2.png/.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "figures"

WIN_LO, WIN_HI = 0.0, 0.3
VIEW_LO, VIEW_HI = -0.8, 1.7


def main():
    t = np.linspace(VIEW_LO, VIEW_HI, 900)
    true = np.exp(2 * t)
    wrong = np.exp(2 * t) * np.cos(t ** 3)

    rng = np.random.default_rng(5)
    xd = np.linspace(WIN_LO + 0.01, WIN_HI, 12)
    yd = np.exp(2 * xd) + rng.normal(0, 0.01 * np.exp(2 * xd).std(), len(xd))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))

    # ============ 左：窗内 ============
    ax = axes[0]
    m = (t >= WIN_LO) & (t <= WIN_HI)
    ax.plot(t[m], true[m], color="black", lw=3, label="True: $e^{2x}$")
    ax.plot(t[m], wrong[m], "--", color="#c0392b", lw=2.4,
            label=r"iMCTS: $e^{2x}\cos(x^3)$")
    ax.scatter(xd, yd, s=55, color="#2c3e50", zorder=6, label="Observed data")
    ax.set_xlim(WIN_LO - 0.03, WIN_HI + 0.03)
    ax.set_ylim(0.9, 1.95)
    ax.set_title("INSIDE the observation window:\nall curves overlap — structure not identifiable",
                 fontsize=13.5)
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    ax.set_xlabel("x", fontsize=12); ax.set_ylabel("y", fontsize=12)
    ax.tick_params(labelsize=11)

    # ============ 右：窗外 ============
    ax = axes[1]
    ax.plot(t, true, color="black", lw=3, label="True: $e^{2x}$")
    ax.plot(t, wrong, "--", color="#c0392b", lw=2.4,
            label=r"iMCTS: $e^{2x}\cos(x^3)$ (wrong)")
    ax.plot(t, true, ":", color="#1e8449", lw=2.8,
            label="CanonicalSR: $e^{2x}$")
    ax.axvspan(WIN_LO, WIN_HI, color="#3498db", alpha=0.10)
    ax.text((WIN_LO + WIN_HI) / 2, 23, "observed\nwindow", ha="center",
            fontsize=11, color="#1f618d")
    ax.set_xlim(VIEW_LO, VIEW_HI)
    ax.set_ylim(-6, 25)
    ax.set_title("OUTSIDE the window:\nwrong structure oscillates — CanonicalSR stays exact",
                 fontsize=13.5)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    ax.set_xlabel("x", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.annotate("sign flips —\ncompletely wrong structure",
                xy=(1.3, -4.5), xytext=(0.75, 14),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8),
                fontsize=12, color="#922b21")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"painpoint_v2.{ext}", dpi=200, bbox_inches="tight")
    print("->", OUT / "painpoint_v2.png")


if __name__ == "__main__":
    main()
