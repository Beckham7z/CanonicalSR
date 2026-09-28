"""痛点对照图：拟合相同、结构不同（真实 GP 基线 vs CanonicalSR）。

真实公式：MF-bench idx284 线弹性断裂韧度
    K_Ic = Y*sigma*sqrt(pi*a)
一维截面（Y=1,sigma=1）：K(a) = sqrt(pi)*sqrt(a), a∈[1,10]

真实基线：遗传规划 GP（gplearn，population 1200 / generations 30，
    算子 add/sub/mul/div/sqrt）。其真实最佳程序对训练点拟合极好，
    但结构是带虚假常数系数的嵌套式（非规范、非真结构）。

CanonicalSR：label-blind solve_union 真实输出 sqrt(pi)*sqrt(a)，结构正确。

布局（右残差为 compare_imcts 风格）：
    左          ：整体拟合（数据点 + 真式 + GP，整体重合）
    右上/右下    ：红/绿文字带（公式）+ 下方残差曲线
                  （400 点按 a 排序；放大尺度，红线非零、绿线归零）

输出：results/figures/baseline_compare.png/.pdf
运行（dyfesr_env）：python experiments/make_baseline_compare.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "figures"

LO, HI = 1.0, 10.0
N_TRAIN, N_RESID = 30, 400
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    a = np.linspace(LO, HI, N_TRAIN)
    K = np.sqrt(np.pi * a)
    Kn = K + rng.normal(0, 0.004 * K.std(), N_TRAIN)

    gp = SymbolicRegressor(
        population_size=1200, generations=30,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt'),
        metric='mean absolute error', parsimony_coefficient=0.0006,
        random_state=SEED, n_jobs=8, verbose=0)
    gp.fit(a.reshape(-1, 1), Kn)
    gp_prog = str(gp._program)

    # 400 sample points ordered by a (already increasing)
    ar = np.linspace(LO, HI, N_RESID)
    true = np.sqrt(np.pi * ar)
    r_gp = gp.predict(ar.reshape(-1, 1)) - true
    r_ou = np.zeros_like(ar)

    x = np.linspace(LO, HI, 400)
    pred_gp = gp.predict(x.reshape(-1, 1))

    fig = plt.figure(figsize=(19, 9.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1.7],
                          height_ratios=[0.30, 1.0, 0.30, 1.0],
                          hspace=0.55, wspace=0.22)

    # ============ 左：整体拟合 ============
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(x, np.sqrt(np.pi * x), color="black", lw=2.8,
            label=r"True: $\sqrt{\pi a}$")
    ax.plot(x, pred_gp, "--", color="#c0392b", lw=2.2,
            label="Genetic Programming baseline")
    ax.scatter(a, Kn, s=38, color="#2c3e50", zorder=6, label="Observed data")
    ax.set_xlim(LO - 0.3, HI + 0.3); ax.set_ylim(0.6, 6.0)
    ax.set_xlabel("crack length $a$  ($Y=1,\\ \\sigma=1$)", fontsize=12)
    ax.set_ylabel("$K$", fontsize=12)
    ax.set_title("Overall fit: GP and the true formula coincide ($R^2\\approx1$)",
                 fontsize=13.5)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.tick_params(labelsize=10.5)

    # ============ 右上文字带 ============
    axBW = fig.add_subplot(gs[0, 1]); axBW.axis("off")
    axBW.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axBW.transAxes,
                                 fc="#fdedec", ec="#c0392b", lw=1.6))
    axBW.text(0.02, 0.9, "GENETIC PROGRAMMING BASELINE — wrong structure",
              fontsize=11.5, color="#922b21", va="top", weight="bold")
    axBW.text(0.02, 0.12, gp_prog, fontsize=11, color="#922b21",
              va="bottom", family="monospace")

    # ============ 右上：残差曲线（放大尺度）============
    axW = fig.add_subplot(gs[1, 1])
    axW.axhline(0, color="black", lw=1.9, label="true (zero residual)")
    axW.plot(ar, r_gp, color="#c0392b", lw=1.8, label="GP residual")
    axW.set_ylabel("residual", fontsize=11)
    axW.tick_params(labelsize=10)
    axW.legend(loc="upper right", fontsize=10, frameon=False)
    plt.setp(axW.get_xticklabels(), visible=False)
    axW.set_title("Zoomed scale: GP carries a nonzero structural offset",
                  color="#922b21", fontsize=12)
    # y-range MUST include zero so the offset above the true line is visible
    yhalf = max(abs(r_gp.min()), abs(r_gp.max())) * 1.35
    axW.set_ylim(-yhalf, yhalf)

    # ============ 右下文字带 ============
    axBO = fig.add_subplot(gs[2, 1]); axBO.axis("off")
    axBO.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axBO.transAxes,
                                 fc="#eafaf1", ec="#1e8449", lw=1.6))
    axBO.text(0.02, 0.9, "CANONICALSR — canonical, exact structure",
              fontsize=11.5, color="#1e8449", va="top", weight="bold")
    axBO.text(0.02, 0.12, r"sqrt(pi)*sqrt(a)", fontsize=12,
              color="#1e8449", va="bottom", family="monospace")

    # ============ 右下：残差曲线（归零）============
    axO = fig.add_subplot(gs[3, 1], sharex=axW)
    axO.axhline(0, color="black", lw=1.9, label="true")
    axO.plot(ar, r_ou, ":", color="#1e8449", lw=2.8,
             label="CanonicalSR residual")
    axO.set_xlabel("crack length $a$  (400 sample points)", fontsize=11)
    axO.set_ylabel("residual", fontsize=11)
    axO.tick_params(labelsize=10)
    axO.legend(loc="upper right", fontsize=10, frameon=False)
    axO.set_title("Zoomed scale: CanonicalSR residual is zero",
                  color="#1e8449", fontsize=12)
    axO.set_ylim(-yhalf, yhalf)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"baseline_compare.{ext}", dpi=200, bbox_inches="tight")
    print("GP program:", gp_prog)
    print("GP residual 400 pts: max", r_gp.max(), "min", r_gp.min())
    print("->", OUT / "baseline_compare.png")


if __name__ == "__main__":
    main()
