"""Run iMCTS, PySR, and LLM-SR on first-20 under one protocol; compare SE.

Protocol (identical to CanonicalSR core run):
  n=400, sigma=0, seed=0; strict structural-equivalence vs the true expression.

Methods:
  imcts  : T2 engine (dyfesr_env), full ops, 60000 evals / 8 s
  pysr   : PySR (pysr_env), serial, bounded time
  llmsr  : lean DeepSeek proposer + numeric coefficient refinement

Output: results/baselines_first20.json (incremental)
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from scesr import gen
from common import structural_equivalent

OUT = HERE.parent / "results" / "baselines_first20.json"

DYF_PY = "/home/zyx/anaconda3/envs/dyfesr_env/bin/python"
PYSR_PY = "/home/zyx/anaconda3/envs/pysr_env/bin/python"
T2_BUILD = "/home/zyx/A_project/SR_Works/MCTS-4-SR-T2/build_t2"


def load():
    if OUT.exists():
        return {r["idx"]: r for r in json.load(OUT)["rows"]}
    return {}


# ---------------- iMCTS ----------------
def run_imcts(idx, g, X, y):
    import subprocess
    worker = HERE / "_imcts_worker.py"
    env = {**os.environ, "OMP_NUM_THREADS": "8"}
    try:
        out = subprocess.run(
            [DYF_PY, str(worker), str(idx)],
            capture_output=True, text=True, timeout=90, env=env)
        line = [l for l in out.stdout.splitlines() if l.startswith("RES ")]
        expr = line[0][4:] if line else ""
    except Exception:
        expr = ""
    nv = X.shape[1]
    return structural_equivalent(expr, g["true"], nv), expr


# ---------------- PySR ----------------
def run_pysr(idx, g, X, y):
    import subprocess
    worker = HERE / "_pysr_worker.py"
    env = {**os.environ, "JULIA_NUM_THREADS": "4", "OMP_NUM_THREADS": "8"}
    try:
        out = subprocess.run([PYSR_PY, str(worker), str(idx)],
                             capture_output=True, text=True, timeout=400, env=env)
        line = [l for l in out.stdout.splitlines() if l.startswith("RES ")]
        expr = line[0][4:] if line else ""
    except Exception:
        expr = ""
    nv = X.shape[1]
    return structural_equivalent(expr, g["true"], nv), expr


# ---------------- LLM-SR (lean) ----------------
def run_llmsr(idx, g, X, y):
    from llm_proposer import llm_propose, refine
    nv = X.shape[1]
    info = [f"input variable x{i}" for i in range(nv)]
    best = (None, np.inf, "")
    try:
        props = llm_propose(X, y, info, n_tries=8)
    except Exception:
        props = []
    for f, code in props:
        try:
            f2, rmse, baked = refine(f, code, X, y)
            if rmse < best[1]:
                best = (f2, rmse, baked)
        except Exception:
            continue
    f2, rmse, baked = best
    if f2 is None:
        return False, ""
    return structural_equivalent(baked, g["true"], nv), baked


def main():
    done = load()
    rows = list(done.values())
    for idx in range(1, 21):
        if idx in done:
            continue
        g = gen(idx, n=400, sigma=0.0, seed=0)
        X = np.asarray(g["X"], float).T
        y = np.asarray(g["yc"], float)
        t0 = time.time()
        se_i, e_i = run_imcts(idx, g, X, y)
        se_p, e_p = run_pysr(idx, g, X, y)
        se_l, e_l = run_llmsr(idx, g, X, y)
        rec = {"idx": idx, "true": g["true"],
               "imcts": {"se": bool(se_i), "expr": e_i},
               "pysr": {"se": bool(se_p), "expr": e_p},
               "llmsr": {"se": bool(se_l), "expr": e_l},
               "sec": round(time.time() - t0, 1)}
        rows.append(rec); done[idx] = rec
        json.dump({"rows": rows}, open(OUT, "w"), ensure_ascii=False, indent=1)
        print(f"idx{idx}: imcts={int(se_i)} pysr={int(se_p)} llmsr={int(se_l)} "
              f"({rec['sec']}s)", flush=True)

    s = {"imcts": sum(r["imcts"]["se"] for r in rows),
         "pysr": sum(r["pysr"]["se"] for r in rows),
         "llmsr": sum(r["llmsr"]["se"] for r in rows),
         "CanonicalSR": 17}
    print("TOTALS:", s)


if __name__ == "__main__":
    main()
