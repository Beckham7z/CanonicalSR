"""Method comparison figure (left-right).

Left : strict structural recovery rate on first-20 (400 samples, noiseless)
        iMCTS 5/20, PySR 9/20, LLM-SR 0/20, CanonicalSR 17/20
Right: generalization beyond the 20 — CanonicalSR on 103 formulas never used
        in design (43/103). Tiles colored recovered (green) / not (gray),
        ordered by number of variables; summary rate annotated.

Output: results/figures/method_comparison.png/.pdf
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
    import json
    base = json.load(open(HERE.parent / "results/baselines_first20.json"))
    gen = json.load(open(HERE.parent / "results/generalization.json"))
    grows = gen["rows"]

    methods = ["iMCTS", "PySR", "LLM-SR", "CanonicalSR"]
    key = {"iMCTS": "imcts", "PySR": "pysr", "LLM-SR": "llmsr"}
    counts = []
    for m in methods:
        if m == "CanonicalSR":
            counts.append(17)
        else:
            counts.append(sum(r[key[m]]["se"] for r in base["rows"]))
    rates = [c / 20 * 100 for c in counts]
    colors = ["#95a5a6", "#e67e22", "#8e44ad", "#27ae60"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.0),
                              gridspec_kw={"width_ratios": [1.0, 1.35]})

    # ---------------- 左：恢复率柱状 ----------------
    ax = axes[0]
    bars = ax.bar(methods, rates, color=colors, edgecolor="black", width=0.62)
    for b_, c_, r_ in zip(bars, counts, rates):
        ax.text(b_.get_x() + b_.get_width() / 2, r_ + 1.8,
                f"{c_}/20\n{r_:.0f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("structural exact recovery rate (%)", fontsize=12.5)
    ax.set_title("First-20 (400 samples, noiseless)", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # ---------------- 右：103 未接触公式瓦片 ----------------
    ax = axes[1]
    rows = sorted(grows, key=lambda r: (r["nv"], r["idx"]))
    n = len(rows)
    ncol = 20
    nrow = int(np.ceil(n / ncol))
    for i, r in enumerate(rows):
        rr = i // ncol
        cc = i % ncol
        c = "#27ae60" if r["se"] else "#d5dbdb"
        ec = "#1e8449" if r["se"] else "#aeb6bf"
        ax.add_patch(plt.Rectangle((cc, nrow - 1 - rr), 0.92, 0.92,
                                   fc=c, ec=ec, lw=0.8))
    ok = sum(r["se"] for r in rows)
    ax.set_xlim(-0.5, ncol)
    ax.set_ylim(-0.8, nrow + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Beyond the 20: 103 formulas never used in design",
                 fontsize=14)
    ax.text(0.0, nrow + 0.55,
            f"CanonicalSR recovers {ok}/103 ({ok/n*100:.0f}%) unseen formulas"
            "  —  green = recovered,  gray = not",
            fontsize=12, color="#1e8449", va="bottom")

    # legend swatches
    ax.add_patch(plt.Rectangle((13.5, nrow + 0.2), 0.9, 0.9, fc="#27ae60",
                               ec="#1e8449"))
    ax.text(14.6, nrow + 0.65, "recovered", fontsize=10.5, va="center")
    ax.add_patch(plt.Rectangle((16.6, nrow + 0.2), 0.9, 0.9, fc="#d5dbdb",
                               ec="#aeb6bf"))
    ax.text(17.7, nrow + 0.65, "not", fontsize=10.5, va="center")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"method_comparison.{ext}", dpi=200, bbox_inches="tight")
    print("->", OUT / "method_comparison.png")


if __name__ == "__main__":
    main()
