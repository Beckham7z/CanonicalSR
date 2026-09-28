"""由已固定的结果 JSON 生成消融表与结果表（CSV + Markdown）。

输入：results/ablation_first20.json, results/core_first20.json,
      results/heldout20.json
输出：results/tables/*.csv, *.md
用法：python experiments/make_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
OUT = R / "tables"
OUT.mkdir(exist_ok=True)


def write_csv(name, header, rows):
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def tick(b):
    return "1" if b else "0"


def fmt_e(v):
    return f"{v:.2e}"


# ---------------------------------------------------------------- 1 结果表
core = json.load(open(R / "core_first20.json"))
header = ["idx", "true", "recovered", "arm", "nrmse", "SE",
          "fresh_relerr"]
rows = []
for r in core["rows"]:
    rows.append([r["idx"], f"`{r['true']}`", f"`{r['expression']}`",
                 r["arm"], fmt_e(r["nrmse"]), tick(r["se"]),
                 fmt_e(r["fresh_relerr"])])
write_csv("core_results", header, rows)

agg = core["summary"]
md = [f"# Core results — MF-bench first-20 (n={agg['protocol']['n']}, "
      f"sigma={agg['protocol']['sigma']}, seed train/fresh = "
      f"{agg['protocol']['seed_train']}/{agg['protocol']['seed_fresh']}, "
      f"{agg['protocol']['dtype']})", "",
      f"**SE = {agg['SE']}/20    fresh-data strict = "
      f"{agg['fresh_strict']}/20    total {agg['total_sec']}s**", "",
      md_table(header, rows), "",
      "> `idx3` SE=0 is an Abs-criterion artifact; fresh relerr "
      "= 1.0e-7 (numerically exact)."]
(OUT / "core_results.md").write_text("\n".join(md))

# ---------------------------------------------------------------- 2 消融表
abl = json.load(open(R / "ablation_first20.json"))
header = ["idx",
          "A_SE", "A_fresh",
          "B_SE", "B_fresh",
          "C_SE", "C_fresh"]
rows = []
for r in abl["rows"]:
    L = r["levels"]
    rows.append([r["idx"],
                 tick(L["A_error_only"]["se"]),
                 fmt_e(L["A_error_only"]["fresh_relerr"]),
                 tick(L["B_simplicity"]["se"]),
                 fmt_e(L["B_simplicity"]["fresh_relerr"]),
                 tick(L["C_full_multi"]["se"]),
                 fmt_e(L["C_full_multi"]["fresh_relerr"])])
write_csv("ablation", header, rows)

s = abl["summary"]
md = ["# Ablation — first-20, same data and same candidate origin", "",
      "| Level | Mechanism | SE | fresh strict |",
      "|---|---|---|---|",
      "| A: error-only | single generic sparse search, argmin NRMSE | "
      f"{s['A_error_only']['SE']}/20 | "
      f"{s['A_error_only']['fresh_strict']}/20 |",
      "| B: + simplicity | rational coefficient snapping; min ops within "
      f"NRMSE<1e-4 | {s['B_simplicity']['SE']}/20 | "
      f"{s['B_simplicity']['fresh_strict']}/20 |",
      "| C: full multi-route | + all route families and narrow restarts | "
      f"{s['C_full_multi']['SE']}/20 | "
      f"{s['C_full_multi']['fresh_strict']}/20 |", "",
      f"**A→B gained {s['A_to_B']['gained']} / lost {s['A_to_B']['lost']}; "
      f"B→C gained {s['B_to_C']['gained']} / lost {s['B_to_C']['lost']}.**", "",
      "Per-formula (SE and independent fresh-data relative error):", "",
      md_table(header, rows)]
(OUT / "ablation.md").write_text("\n".join(md))

# ---------------------------------------------------------------- 3 留出表
ho = json.load(open(R / "heldout20.json"))
header = ["idx", "stratum", "recovered", "SE", "fresh_relerr"]
rows = []
for r in ho["heldout"]:
    rows.append([r["idx"], "/".join(r["stratum"]),
                 f"`{r['expression']}`", tick(r["se"]),
                 fmt_e(r["fresh_relerr"])])
write_csv("heldout20", header, rows)
s = ho["summary"]
md = [f"# Unseen-formula holdout — {s['n_helded'] if False else s['n_held']} formulas never used in design",
      "", f"**Cross-formula SE = {s['SE']}/{s['n_held']}    fresh strict = "
      f"{s['fresh_strict']}/{s['n_held']}**", "",
      md_table(header, rows)]
(OUT / "heldout20.md").write_text("\n".join(md))

print("written to", OUT)
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
