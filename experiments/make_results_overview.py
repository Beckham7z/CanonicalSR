"""结果总览图（左右布局）：baseline 恢复成功率对比 + 外推效果。

左：first-20（每条 400 点，无噪声）上的严格结构恢复率
    iMCTS            6/20 (30%)
    error-only       9/20 (45%)   消融 A：纯按数值误差
    + simplicity     15/20 (75%)  消融 B：有理吸附 + 简洁选择
    CanonicalSR      17/20 (85%)  消融 C：完整多路线
右：外推效果（真实公式 idx284 一维截面 K=sqrt(pi*a)）
    拟合导向多项式在训练域 a∈[1,5] 与真式重合，外推即弯曲、翻号、发散；
    CanonicalSR = sqrt(pi*a) 在全域与真式重合。

输出：results/figures/results_overview.png/.pdf
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "figures"


def main():
    # ---------- 左：恢复成功率柱状 ----------
    methods = ["iMCTS", "error-only", "+ simplicity", "CanonicalSR"]
    counts = [6, 9, 15, 17]
    rates = [c / 20 * 100 for c in counts]
    colors = ["#95a5a6", "#e67e22", "#2980b9", "#27ae60"]

    # ---------- 右：外推曲线 ----------
    rng = np.random.default_rng(1)
    ad = np.linspace(1, 5, 16)
    Kd = np.sqrt(np.pi * ad)
    Kn = Kd + rng.normal(0, 0.004 * Kd.std(), len(ad))
    coef = np.polyfit(ad, Kn, 6)
    a = np.linspace(0.5, 13, 800)
    true = np.sqrt(np.pi * a)
    fit = np.polyval(coef, a)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6),
                              gridspec_kw={"width_ratios": [1.0, 1.25]})

    # 左
    ax = axes[0]
    bars = ax.bar(methods, rates, color=colors, edgecolor="black", width=0.62)
    for b_, c_, r_ in zip(bars, counts, rates):
        ax.text(b_.get_x() + b_.get_width() / 2, r_ + 1.5,
                f"{c_}/20\n{r_:.0f}%", ha="center", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("structural exact recovery rate (%)", fontsize=12)
    ax.set_title("Recovery rate on first-20 (400 samples, noiseless)",
                 fontsize=13)
    ax.tick_params(axis="x", labelsize=10.5, rotation=10)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # 右
    ax = axes[1]
    ax.plot(a, true, color="black", lw=2.8, label="True: $\\sqrt{\\pi a}$")
    ax.plot(a, fit, "--", color="#c0392b", lw=2.1,
            label="Fit-only baseline")
    ax.plot(a, true, ":", color="#27ae60", lw=2.6,
            label="CanonicalSR")
    ax.axvspan(1, 5, color="#3498db", alpha=0.08)
    ax.text(3, 6.7, "training domain", ha="center", fontsize=10,
            color="#1f618d")
    ax.annotate("bends, flips sign\non extrapolation",
                xy=(12.2, -50), xytext=(8.3, 4.5),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8),
                fontsize=10.5, color="#922b21")
    ax.set_xlim(0.5, 13); ax.set_ylim(-60, 8)
    ax.set_xlabel("crack length $a$", fontsize=12)
    ax.set_ylabel("$K$", fontsize=12)
    ax.set_title("Extrapolation: fit-only diverges, CanonicalSR stays exact",
                 fontsize=13)
    ax.legend(loc="lower left", fontsize=10.5, frameon=False)
    ax.tick_params(labelsize=10.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"results_overview.{ext}", dpi=200, bbox_inches="tight")
    print("->", OUT / "results_overview.png")


if __name__ == "__main__":
    main()
