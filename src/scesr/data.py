"""20 条 MF-bench 公式的自包含数据生成（纯 numpy，不 import bp5/PySR/LaSR/Julia）。

custom 6 条（49/94/183/200/257/296）的规格与 12th 实验一致；
其余 14 条从 MF 数据集读取并按 var_ranges 采样生成。

每条返回: dict(X[nv,n] float32, y_train[含噪], y_clean, names, true_expr(sympy-safe), idx)
真表达式用 x0,x1,... 变量名（与 metrics.compute_all_metrics 兼容）。
"""
from __future__ import annotations

import json
import re
import numpy as np

DATA = "/home/zyx/A_project/SR_Works/MF-bench-dataset/kept_formulas_with_background.json"
PAPER20 = [8, 9, 21, 49, 52, 71, 94, 106, 111, 183,
            187, 198, 199, 200, 239, 257, 280, 284, 293, 296]

_NS = {"np": np, "pi": np.pi, "abs": np.abs, "max": np.maximum, "min": np.minimum}


def _custom_spec(idx):
    if idx == 9:
        # M = mdot*sqrt(r*T)/(A*p*sqrt(kappa))
        return (["mdot","A","p","r","T","kappa"],
                [(0.5,5.0),(0.1,10.0),(1e5,5e8),(1.0,100.0),(1.0,5000.0),(1e-6,1.0)],
                "mdot*np.sqrt(r*T)/(A*p*np.sqrt(kappa))",
                "x0*sqrt(x3*x4)/(x1*x2*sqrt(x5))",
                ["+","-","*","/","sqrt"])
    if idx == 284:
        # 断裂韧度 K_Ic = Y*sigma*sqrt(pi*a)
        return (["Y","sigma","a"], [(0.8,1.2),(10,500),(1e-4,1e-1)],
                "Y*sigma*np.sqrt(np.pi*a)",
                "x0*x1*sqrt(pi*x2)", ["+","-","*","/","sqrt"])
    if idx == 293:
        # 简化裂纹尖端应力角因子 ~ (K_I^2/(2*pi*sig_y^2))*cos(theta/2)^2 ；
        # 用可解析形式: sigma/(r) * cos(theta/2)^2 的可恢复近似（乘性，cos 作用于 x3 角度）
        return (["K","sigy","r","theta"], [(1e3,1e5),(1e6,5e8),(1e-6,1e-4),(0.0,3.14)],
                "(K**2/(2.0*np.pi*sigy**2))*np.cos(theta/2.0)**2/r",
                "(x0**2/(2.0*pi*x1**2))*cos(x3/2.0)**2/x2", ["+","-","*","/","cos"])
    if idx == 49:
        return (["b","hc","h","g"], [(0.1,100),(0.0,50),(0.0,100),(9.7,9.9)],
                "0.62*b*hc*np.sqrt(2.0*g*np.maximum(h-hc,0.0))",
                "0.62*x0*x1*sqrt(2.0*x3*max(x2-x1,0.0))", ["+","-","*","/","sqrt"])
    if idx == 94:
        return (["vx","vy","vz","gamma","r","T"],
                [(-2e3,2e3),(-2e3,2e3),(-2e3,2e3),(1.05,1.67),(50,1000),(1,5000)],
                "np.sqrt((vx**2+vy**2+vz**2)/(gamma*r*T))",
                "sqrt((x0**2+x1**2+x2**2)/(x3*x4*x5))", ["+","-","*","/","sqrt"])
    if idx == 183:
        return (["G1s","G2s","K2s","V1","V2"],
                [(1e8,1e11),(1e8,1e11),(1e8,1e11),(0.05,0.95),(0.05,0.95)],
                "G2s + V1/(1/(G1s-G2s) + (6*(K2s+2*G2s)*V2)/(5*(3*K2s+4*G2s)*G2s))",
                "x1 + x3/(1/(x0-x1) + (6*(x2+2*x1)*x4)/(5*(3*x2+4*x1)*x1))", ["+","-","*","/"])
    if idx == 187:
        # ell_str = r * sqrt(Vf * (Gf/Gm))
        return (["r","Vf","Gf","Gm"],
                [(1e-6,1e-3),(0.0,1.0),(1e6,1e12),(1e6,1e12)],
                "r*np.sqrt(Vf*(Gf/Gm))",
                "x0*sqrt(x1*(x2/x3))", ["+","-","*","/","sqrt"])
    if idx == 280:
        # sigma = sigma_av / ((1+2R/r_n)*ln(1+r_n/(2R)))
        return (["sigma","sigma_av","R","r_n"],
                [(1e7,2e9),(1e7,1.5e9),(1e-4,1e-2),(1e-4,5e-3)],
                "sigma_av/((1.0+2.0*R/r_n)*np.log(1.0+r_n/(2.0*R)))",
                "x1/((1.0+2.0*x2/x3)*log(1.0+x3/(2.0*x2)))", ["+","-","*","/","log"])
    if idx == 200:
        return (["E","nu","delta","a0","r"],
                [(1e9,5e10),(0.2,0.4),(1e-10,5e-10),(2e-10,4e-10),(5e-8,5e-7)],
                "(E/(2*(1+nu)))*(6*delta**2*a0**4/r**6)",
                "(x0/(2*(1+x1)))*(6*x2**2*x3**4/x4**6)", ["+","-","*","/"])
    if idx == 257:
        return (["cn","cn1","D","t"],
                [(1,1e4),(1,1e4),(1e-12,1e-8),(1e-4,1e4)],
                "96485.0*(cn1-cn)*np.sqrt(D/(np.pi*t))",
                "96485.0*(x1-x0)*sqrt(x2/(pi*x3))", ["+","-","*","/","sqrt"])
    if idx == 296:
        return (["taum","G","b","nu","r0","x"],
                [(1e5,1e8),(1e9,1e11),(2e-10,5e-10),(0.2,0.35),(1e-9,1e-6),(1e-8,1e-5)],
                "taum + (G*b/(2.38*np.pi*np.sqrt(1-nu)))*np.log(r0/b)/x",
                "x0 + (x1*x2/(2.38*pi*sqrt(1-x3)))*log(x4/x2)/x5", ["+","-","*","/","sqrt","log"])
    return None


def _load_meta():
    with open(DATA) as f:
        return {d["new_idx"]: d for d in json.load(f)}




def _gen_colebrook(idx, n, seed):
    """idx41：Colebrook 隐式方程 1/sqrt(f) = -2.03*log10(ks/(3.7D) + 2.51/(Re sqrt(f)))。

    该式对 f 隐式，无闭式 y=f(x) 真解，故 true 置空并打 implicit 标记（结构分不适用）。
    数据用不动点迭代求解 f。
    """
    rng = np.random.default_rng(seed)
    Re = rng.uniform(1e3, 1e8, n)
    D = rng.uniform(1e-3, 10.0, n)
    ks = rng.uniform(0.0, 1e-2, n)
    f = np.full(n, 0.02)
    for _ in range(200):
        rhs = -2.03 * np.log10(2.51 / (Re * np.sqrt(f)) + ks / (3.7 * D))
        f_new = 1.0 / (rhs * rhs)
        if np.max(np.abs(f_new - f)) < 1e-12:
            f = f_new
            break
        f = f_new
    X = np.stack([Re, D, ks], axis=0)
    return {"X": X.astype(np.float32), "y": f.astype(np.float32),
            "yc": f, "names": ["x0", "x1", "x2"], "true": "",
            "idx": idx, "ops": None, "implicit": True}


def gen(idx, n=80, sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    if idx == 41:
        return _gen_colebrook(idx, n, seed)
    spec = _custom_spec(idx)
    if spec is not None:
        names, ranges, expr_py, true_x, ops = spec
        # 把 expr_py 里的命名变量替换成 x0,x1,...
        expr_x = expr_py
        for i, nm in enumerate(names):
            expr_x = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(nm)}(?![A-Za-z0-9_])", f"x{i}", expr_x)
        X = np.stack([rng.uniform(lo,hi,n) for lo,hi in ranges], axis=0)
        vd = {f"x{i}": X[i] for i in range(len(names))}
        try:
            eval_ns = {"__builtins__":{},
                    "sin":np.sin,"cos":np.cos,"exp":np.exp,"log":np.log,"sqrt":np.sqrt,
                    "abs":np.abs,"max":np.maximum,"min":np.minimum,"maximum":np.maximum,
                    "minimum":np.minimum,"pi":np.pi,
                    "np":np}  # 保留 np 前缀也能用
            yc = eval(expr_x, {"__builtins__":{}}, {**eval_ns, **vd}).astype(np.float64).ravel()
        except Exception:
            return None
        if not np.all(np.isfinite(yc)) or np.std(yc) < 1e-10:
            return None
        y = yc + rng.normal(0, sigma*np.std(yc), n) if sigma>0 else yc.copy()
        names_x = [f"x{i}" for i in range(len(names))]
        return {"X":X.astype(np.float32), "y":y.astype(np.float32), "yc":yc,
                "names":names_x, "true":true_x, "idx":idx, "ops":ops}
    # generic
    meta = _load_meta()
    if idx not in meta:
        return None
    # 优先使用健壮的 v2 生成器（AST 规范化，覆盖面更广）
    from . import gen_formula as gen_v2
    g2 = None
    try:
        g2 = gen_v2.generate(meta[idx], n=n, seed=seed)
    except Exception:
        g2 = None
    if g2 is not None:
        X = g2["X"].astype(np.float32)
        yc = g2["y"]
        names = [f"x{i}" for i in range(X.shape[0])]
        true_x = g2["expr_x"]
    else:
        return None
    y = yc + np.random.default_rng(seed+999).normal(0, sigma*np.std(yc), n) if sigma>0 else yc.copy()
    return {"X":X.astype(np.float32), "y":y.astype(np.float32), "yc":yc,
            "names":names, "true":true_x, "idx":idx, "ops":None}


if __name__ == "__main__":
    ok = 0
    for i in PAPER20:
        try:
            c = gen(i)
            if c is None or not np.all(np.isfinite(c["yc"])):
                print(f"idx={i}: FAIL 数据生成"); continue
            print(f"idx={i:>3} vars={len(c['names'])} yrange=({c['yc'].min():.2g},{c['yc'].max():.2g}) true={c['true'][:55]}")
            ok += 1
        except Exception as e:
            print(f"idx={i}: ERR {str(e)[:70]}")
    print(f"\n成功生成 {ok}/20")
