"""快速对照：SISSO 风格基线（SIS 相关筛选 + LASSO 稀疏算子）vs CanonicalSR。

诚实声明：这不是官方 SISSO 实现，而是按其核心思想的有界复现——
  * 递归构造特征（+,-,*,/ 与 sqrt/log/square/cube/inverse 单目），
    有复杂度上限；
  * SIS：按与目标的绝对相关保留 top-k 子空间；
  * SO ：LASSO 稀疏选择，OLS 重拟合。
SISSO 输出系数为连续小数，故另报"系数吸附后"的 SE，检验其特征空间
是否含正确结构——与 CanonicalSR 的对比才公平。

协议同主实验：first-20, n=400, sigma=0, seed 0, float64, fresh seed 7。
输出：results/compare_sisso.json
用法：python experiments/baseline_sisso.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scesr import gen
from common import (N_SAMPLES, SIGMA, SEED_TRAIN, structural_equivalent,
                    fresh_relerr, eval_expr)

OUT = HERE.parent / "results" / "compare_sisso.json"

MAX_FEATURES = 4000
MAX_DEPTH = 2
SIS_TOPK = 300
PAIR_SAMPLE = 2500
SEED_FEAT = 0


def fingerprint(v):
    idx = np.linspace(0, len(v) - 1, 48).astype(int)
    z = (v - v.mean()) / (v.std() + 1e-12)
    return np.round(z[idx], 4).tobytes()


def guarded(v):
    if not np.all(np.isfinite(v)):
        return None
    if v.std() < 1e-10:
        return None
    return v


def unary_transforms(name, v):
    out = []
    sc = float(np.max(np.abs(v))) + 1.0

    def add(expr, val):
        g = guarded(np.asarray(val, float))
        if g is not None:
            out.append((expr, g))

    add(f"({name})**2", v ** 2)
    add(f"({name})**3", v ** 3)
    add(f"1/({name})", 1.0 / (v + 1e-10 * sc))
    if np.all(v > 1e-12 * sc):
        add(f"sqrt({name})", np.sqrt(np.clip(v, 0, None)))
        add(f"log({name})", np.log(np.clip(v, 1e-300, None)))
    elif np.all(v < -1e-12 * sc):
        add(f"sqrt(-{name})", np.sqrt(np.clip(-v, 0, None)))
    return out


def build_features(X):
    n, d = X.shape
    feats = {f"x{i}": X[:, i] for i in range(d)}
    seen = {fingerprint(v) for v in feats.values()}
    current = list(feats.items())

    def register(expr, val):
        g = guarded(val)
        if g is None or len(expr) > 160:
            return False
        fp = fingerprint(g)
        if fp in seen or len(feats) >= MAX_FEATURES:
            return False
        seen.add(fp); feats[expr] = g
        return True

    for depth in range(MAX_DEPTH):
        added = []
        # 单目
        for name, v in current:
            for expr, gv in unary_transforms(name, v):
                if register(expr, gv):
                    added.append((expr, gv))
        # 二元
        pairs = list(combinations(range(len(current)), 2))
        if len(pairs) > PAIR_SAMPLE:
            rng = np.random.default_rng(SEED_FEAT + depth)
            sel = rng.choice(len(pairs), PAIR_SAMPLE, replace=False)
            pairs = [pairs[i] for i in sorted(sel)]
        for ai, bi in pairs:
            (na, va), (nb, vb) = current[ai], current[bi]
            if register(f"({na})+({nb})", va + vb):
                added.append((f"({na})+({nb})", va + vb))
            if register(f"({na})-({nb})", va - vb):
                added.append((f"({na})-({nb})", va - vb))
            if register(f"({na})*({nb})", va * vb):
                added.append((f"({na})*({nb})", va * vb))
            sc = max(float(np.max(np.abs(vb))), 1.0)
            if np.min(np.abs(vb)) > 1e-10 * sc:
                if register(f"({na})/({nb})", va / vb):
                    added.append((f"({na})/({nb})", va / vb))
        if len(feats) >= MAX_FEATURES:
            break
        current = added

    names = list(feats)
    M = np.stack([feats[k] for k in names], axis=1)
    return M, names


def sisso_solve(M, names, y):
    mu, sd = M.mean(0), M.std(0)
    sd[sd < 1e-12] = 1.0
    Mz = (M - mu) / sd
    ym, ys = float(y.mean()), float(y.std())
    yz = (y - ym) / ys

    corr = np.array([abs(np.corrcoef(Mz[:, k], yz)[0, 1])
                     if np.isfinite(np.corrcoef(Mz[:, k], yz)[0, 1]) else 0.0
                     for k in range(Mz.shape[1])])
    top = np.argsort(-corr)[:SIS_TOPK]

    from sklearn.linear_model import LassoCV, Lasso
    model = LassoCV(cv=5, max_iter=50000)
    model.fit(Mz[:, top], yz)
    coefz = model.coef_
    nz = np.where(np.abs(coefz) > 1e-8)[0]
    if len(nz) == 0:
        model = Lasso(alpha=1e-4, max_iter=50000)
        model.fit(Mz[:, top], yz)
        nz = np.where(np.abs(model.coef_) > 1e-8)[0]

    cols = top[nz]
    # 原始空间 OLS 重拟合
    A = np.concatenate([M[:, cols], np.ones((len(y), 1))], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)

    body = ""
    for i, (u, nm) in enumerate(zip(c[:-1], [names[k] for k in cols])):
        body += ("-" if u < 0 else ("+" if i else "")) + f"{abs(u):.10g}*{nm}"
    if abs(c[-1]) > 1e-9 * (np.mean(np.abs(y)) + 1e-12):
        body += ("+" if c[-1] > 0 else "-") + f"{abs(c[-1]):.6g}"
    return body or "0"


def snap_linear(expr):
    """把线性组合里的小数系数吸附到有理网格（仅系数，结构不动）。"""
    from fractions import Fraction
    def repl(m):
        u = float(m.group(0))
        fr = Fraction(abs(u)).limit_denominator(12)
        if abs(float(fr) - abs(u)) < 0.02:
            s = f"({fr.numerator}/{fr.denominator})" if fr.denominator > 1 \
                else str(fr.numerator)
            return ("-" if u < 0 else "") + s
        return m.group(0)
    return __import__("re").sub(r"-?\d+\.\d+(?:e-?\d+)?", repl, expr)


def main():
    rows, t0 = [], time.time()
    for idx in range(1, 21):
        g = gen(idx, n=N_SAMPLES, sigma=SIGMA, seed=SEED_TRAIN)
        X = np.asarray(g["X"]).T.astype(np.float64)
        y = np.asarray(g["yc"], np.float64)
        d = X.shape[1]

        ts = time.time()
        M, names = build_features(X)
        raw = sisso_solve(M, names, y)
        snapped = snap_linear(raw)
        dt = time.time() - ts

        cell = {
            "raw_expression": raw,
            "raw_se": structural_equivalent(raw, g["true"], d),
            "raw_fresh": fresh_relerr(raw, gen, idx),
            "snapped_expression": snapped,
            "snapped_se": structural_equivalent(snapped, g["true"], d),
            "snapped_fresh": fresh_relerr(snapped, gen, idx),
            "n_features": int(M.shape[1]), "sec": round(dt, 2),
        }
        rows.append({"idx": idx, "true": g["true"], **cell})
        print(f"idx{idx:>2} nf={M.shape[1]:>4} rawSE={int(cell['raw_se'])} "
              f"snapSE={int(cell['snapped_se'])} snapfresh={cell['snapped_fresh']:.1e}"
            f" t={dt:5.1f}s", flush=True)

    total = time.time() - t0
    summary = {
        "raw_SE": sum(r["raw_se"] for r in rows),
        "snapped_SE": sum(r["snapped_se"] for r in rows),
        "snapped_fresh_strict": sum(r["snapped_fresh"] < 1e-3 for r in rows),
        "CanonicalSR_SE": 17, "CanonicalSR_fresh": 18,
        "total_sec": round(total, 1),
    }
    meta = {"note": "SISSO-style bounded reimplementation (SIS+LASSO)",
            "max_features": MAX_FEATURES, "max_depth": MAX_DEPTH,
            "sis_topk": SIS_TOPK}
    OUT.write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows},
                             ensure_ascii=False, indent=1))
    print("\n" + "=" * 56)
    print(f"SISSO-style raw SE      : {summary['raw_SE']}/20")
    print(f"SISSO-style + snap SE   : {summary['snapped_se'] if False else summary['snapped_SE']}/20"
          f"  fresh {summary['snapped_fresh_strict']}/20")
    print(f"CanonicalSR SE          : 17/20  fresh 18/20")
    print(f"total {total:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
