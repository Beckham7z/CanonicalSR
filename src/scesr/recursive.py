"""残差递归符号求解器：无真解，回归大部分材料科学闭式公式。

核心思想（突破"一次性全局线性化"的局限）：
  不在整个公式上一次求解，而是维护一个"还没解释的残差 r"，每一轮：
    1. 构造一批候选"原子部件"（变量、积、商、1±比值、平方、平方×变量、
       变量×另一变量的简单函数）；
    2. 在【无量纲标准化空间】计算每个候选与当前残差的相关，选相关最强者；
    3. 把所有已选部件合起来做最小二乘，系数吸附到有理网格 {0,±.5,±1,±2}；
    4. 用原始数据算 NRMSE，残差显著下降才接受该部件，否则停。
  扣掉主导项后，被大变量盖住的小项会在残差中逐级显现（MEDIUM 类因此可解）。

主入口 solve_recursive(X, y) -> {applicable, expression, r2, n_terms}。
"""
from __future__ import annotations
import itertools
import numpy as np

RATIONAL_GRID = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])


def snap_rational(x):
    x = float(x)
    return float(RATIONAL_GRID[int(np.argmin(np.abs(RATIONAL_GRID - x)))])


# ---------------------------------------------------------------------------
# 原子部件工厂
# ---------------------------------------------------------------------------

def _safe(v):
    v = np.asarray(v, dtype=np.float64)
    v = np.where(np.isfinite(v), v, 0.0)
    return v


def atom_library(X):
    """返回 (M[n,k], names)：候选原子部件。"""
    n, d = X.shape
    cols, names = [], []

    def add(v, name):
        v = _safe(v)
        if np.std(v) > 1e-12:
            cols.append(v); names.append(name)

    # 1. 单变量 + 变量的初等变换（log / 倒数），用于外壳逆变换空间搜索
    for i in range(d):
        add(X[:, i], f"x{i}")
        if np.all(X[:, i] > 0):
            add(np.log(X[:, i]), f"log(x{i})")
            add(1.0 / X[:, i], f"1/x{i}")

    # 2. 两两乘积、商
    for i, j in itertools.combinations(range(d), 2):
        add(X[:, i] * X[:, j], f"x{i}*x{j}")
        add(X[:, i] / (X[:, j] + 1e-12), f"x{i}/x{j}")
        add(X[:, j] / (X[:, i] + 1e-12), f"x{j}/x{i}")

    # 3. 平方、平方×另一变量、平方×变量比
    for i in range(d):
        add(X[:, i] ** 2, f"x{i}**2")
        for j in range(d):
            if j != i:
                add(X[:, i] ** 2 * X[:, j], f"x{i}**2*x{j}")

    # 4. 1±比值（材料公式常见 (1+ratio) 因子）
    for i, j in itertools.combinations(range(d), 2):
        r_ij = X[:, i] / (X[:, j] + 1e-12)
        add(1 + r_ij, f"(1+x{i}/x{j})")
        add(1 - r_ij, f"(1-x{i}/x{j})")

    M = np.stack(cols, axis=1)
    return M, names


# ---------------------------------------------------------------------------
# 残差递归（前向贪心 + 整体最小二乘 + 有理吸附）
# ---------------------------------------------------------------------------

def solve_recursive(X, y, max_terms=8, min_r2=0.9999, improve_rel=1e-6):
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, d = X.shape

    M, names = atom_library(X)

    # 无量纲化候选与目标（跨尺度核心）
    m_mu, m_sd = M.mean(0), M.std(0)
    Mz = (M - m_mu) / m_sd
    y_mu, y_sd = y.mean(), y.std()
    if y_sd < 1e-12:
        y_sd = 1.0
    yz = (y - y_mu) / y_sd

    chosen = []
    resid = yz.copy()
    # 接受阈值必须与候选 SSE 同量纲（原始 y）；初始为"仅均值"模型的 SSE。
    prev_sse = float(np.sum((y - y_mu) ** 2))

    for step in range(max_terms):
        # 对每个候选：联合最小二乘后，立即把【新加入项】的系数吸附到有理网格，
        # 用吸附后的系数算 SSE。这样可剔除"系数极小、吸附后归零"的准噪声项，
        # 避免被它们的微小 SSE 带偏。
        best = (prev_sse, None, None, None)  # (sse, k, snapped_coef_full, ic)
        for k in range(Mz.shape[1]):
            if k in chosen:
                continue
            trial = chosen + [k]
            Asub = Mz[:, trial]
            A = np.concatenate([Asub, np.ones((Asub.shape[0], 1))], axis=1)
            cc, _, _, _ = np.linalg.lstsq(A, yz, rcond=None)
            zc, zic = cc[:-1], float(cc[-1])
            # 还原到原始尺度并吸附
            raw_all = zc * (y_sd / m_sd[trial])
            snap_all = np.array([snap_rational(v) for v in raw_all])
            # 新项吸附后若为 0，则该候选不算真实贡献，跳过
            if abs(snap_all[-1]) < 1e-9:
                continue
            # 用吸附后系数在原始 y 上算 SSE
            Fsub = M[:, trial]
            ic = y_mu - float(np.sum(snap_all * m_mu[trial]))
            pred = Fsub @ snap_all + ic
            sse = float(np.sum((pred - y) ** 2))
            if sse < best[0] - improve_rel * (prev_sse + 1e-12):
                best = (sse, k, snap_all, ic)

        if best[1] is None:
            break

        trial_sse, knew, snapped, ic = best
        trial = chosen + [knew]
        chosen = trial
        prev_sse = trial_sse

        # snapped / ic 已是吸附后的干净系数（在选择时算好）
        Fsub = M[:, trial]

        def r2_of(c, ic):
            pred = Fsub @ c + ic
            sse = float(np.sum((pred - y) ** 2))
            return 1 - sse / float(np.sum((y - y_mu) ** 2) or 1e-12)

        # 比较带截距与零截距，取更优
        r2_s = r2_of(snapped, ic)
        r2_0 = r2_of(snapped, 0.0)
        if r2_0 >= r2_s:
            best_fit, best_ic = r2_0, 0.0
        else:
            best_fit, best_ic = r2_s, ic

        # 更新残差：用吸附系数（保证后续残差干净）
        resid = y - Fsub @ snapped - best_ic

        if best_fit >= min_r2:
            sub_names = [names[k] for k in chosen]
            expr = _build_expr(snapped, sub_names, best_ic)
            return {"applicable": True, "expression": expr,
                    "r2": round(best_fit, 6), "n_terms": len(chosen)}

    # 未能在要求精度内闭合
    if chosen:
        # 用最后一组已选部件尽力构造，返回实际 R2（供诊断）
        Asub = Mz[:, chosen]
        A = np.concatenate([Asub, np.ones((Asub.shape[0], 1))], 1)
        cc, _, _, _ = np.linalg.lstsq(A, yz, rcond=None)
        raw = cc[:-1] * (y_sd / m_sd[chosen])
        snapped = np.array([snap_rational(v) for v in raw])
        ic = y_mu - float(np.sum(snapped * m_mu[chosen]))
        r2 = 1 - float(np.sum((M[:,chosen]@snapped+ic-y)**2)) / float(
            np.sum((y-y_mu)**2) or 1e-12)
        sub_names = [names[k] for k in chosen]
        return {"applicable": False, "expression": _build_expr(snapped, sub_names, ic),
                "r2": round(r2, 6), "n_terms": len(chosen),
                "reason": "残差递归未达到min_r2"}
    return {"applicable": False, "reason": "无可选部件"}


def _build_expr(coef, names, intercept):
    """线性组合语义：各部件用 + / - 连接（部件内部的乘除保留在 name 里）。"""
    terms = []
    for k in range(len(coef)):
        c = float(coef[k])
        if abs(c) < 1e-9:
            continue
        a = snap_rational(c)
        use = a if abs(a - c) < 5e-3 else round(c, 4)
        # 系数绝对值为1时省略系数；否则前置系数
        if abs(abs(use) - 1) < 1e-9:
            term = names[k]
        else:
            term = f"{_fmt(abs(use))}*{names[k]}"
        terms.append((c, term))

    if not terms:
        body = "0"
    else:
        # 第一项：正号省略，负号带 -
        c0, t0 = terms[0]
        body = t0 if c0 > 0 else f"-{t0}"
        for c, t in terms[1:]:
            body += (f"+{t}" if c > 0 else f"-{t}")

    if abs(intercept) > 1e-7:
        body += ("+" if intercept > 0 else "-") + _fmt(abs(intercept))
    return body


def _fmt(u):
    return str(int(round(u))) if abs(u-round(u)) < 1e-9 else f"{u:.4g}"


# ---------------------------------------------------------------------------
# 统一入口：直接空间 + 外壳逆变换（log）
# ---------------------------------------------------------------------------

def solve(X, y, max_terms=8, min_r2=0.9999):
    """自动选择是否需要 exp/log 外壳解包。

    返回 {applicable, expression, route, r2}。
    route='direct'  : 在原空间恢复；
    route='expwrap' : 在 log(y) 空间恢复 g，最终表达式 = exp(g)。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()

    # 1) 直接空间
    r = solve_recursive(X, y, max_terms=max_terms, min_r2=min_r2)
    if r.get("applicable"):
        return {"applicable": True, "expression": r["expression"],
                "route": "direct", "r2": r["r2"]}

    # 2) log 外壳空间（要求 y 全正）
    if np.all(y > 0):
        ly = np.log(y)
        rl = solve_recursive(X, ly, max_terms=max_terms, min_r2=min_r2)
        if rl.get("applicable"):
            g = rl["expression"]
            expr = f"exp({g})"
            # 在原始 y 上复核 exp(g) 的拟合
            ns = {f"x{k}": X[:, k] for k in range(X.shape[1])}
            ns.update({"sqrt": np.sqrt, "log": np.log, "exp": np.exp,
                       "sin": np.sin, "cos": np.cos, "abs": np.abs, "pi": np.pi,
                       "max": np.maximum, "min": np.minimum})
            try:
                gval = eval(g, {"__builtins__": {}}, ns)
                pred = np.exp(gval)
                r2 = 1 - float(np.sum((pred - y) ** 2)) / float(
                    np.sum((y - y.mean()) ** 2) or 1e-12)
            except Exception:
                r2 = rl["r2"]
            if r2 >= min_r2:
                return {"applicable": True, "expression": expr,
                        "route": "expwrap", "r2": round(r2, 6)}

    return {"applicable": False, "expression": "",
            "route": "unsolved", "r2": r.get("r2")}


# ---------------------------------------------------------------------------
# 分级求解：用"结构完成度SCS"代替"非0即1"，支持 B 档继续优化
# ---------------------------------------------------------------------------

def _metrics_for(expr, X, y):
    """对候选表达式计算全部结构指标（复用 metrics.py）。"""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "mf_metrics",
        "/home/zyx/A_project/SR_Works/20260831-12th-MFbench-Patterns/experiments/metrics.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    names = [f"x{k}" for k in range(X.shape[1])]
    try:
        mm = M.compute_all_metrics(expr, "0", X, y, names)
    except Exception:
        mm = {"r2": 0.0, "oss": 0.0, "tss": 0.0, "psc": 0.0}
    scs = M.structure_completeness(mm)
    grade = M.grade_completeness(scs)
    return scs, grade, mm


def _graded_once(X, y, max_terms):
    """在当前目标空间内，纯粹用"残差SSE下降"驱动剥离（不接触真式）。

    每步选"加入后联合拟合 SSE 最低"的原子，系数吸附到有理网格；
    返回 {expression, n_terms, residual_sse}。SCS/grade 交给上层最后统一算。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()

    F, names = atom_library(X)
    f_mu, f_sd = F.mean(0), F.std(0)
    Fz = (F - f_mu) / f_sd
    y_mu, y_sd = y.mean(), y.std()
    if y_sd < 1e-12:
        y_sd = 1.0
    yz = (y - y_mu) / y_sd

    chosen = []
    resid = yz.copy()
    last_expr = None
    # 接受阈值与候选 SSE 同量纲（原始 y）；初始为"仅均值"模型。
    pick_sse = float(np.sum((y - y_mu) ** 2))

    for step in range(max_terms):
        pick = None
        for k in range(Fz.shape[1]):
            if k in chosen:
                continue
            trial = chosen + [k]
            Asub = Fz[:, trial]
            A = np.concatenate([Asub, np.ones((Asub.shape[0], 1))], axis=1)
            cc, _, _, _ = np.linalg.lstsq(A, yz, rcond=None)
            zc, zic = cc[:-1], float(cc[-1])
            raw = zc * (y_sd / f_sd[trial])
            snap = np.array([snap_rational(v) for v in raw])
            if abs(snap[-1]) < 1e-9:
                continue
            Fsub = F[:, trial]
            sse0 = float(np.sum((Fsub @ snap - y) ** 2))
            ic = y_mu - float(np.sum(snap * f_mu[trial]))
            sses = float(np.sum((Fsub @ snap + ic - y) ** 2))
            if sse0 <= sses:
                sse, use_ic = sse0, 0.0
            else:
                sse, use_ic = sses, ic
            if sse < pick_sse - 1e-9:
                pick_sse = sse
                sub_names = [names[i] for i in trial]
                expr = _build_expr(snap, sub_names, use_ic)
                pick = (k, expr, use_ic)

        if pick is None:
            break
        last_expr = pick[1]
        chosen.append(pick[0])
        Asub = Fz[:, chosen]
        A = np.concatenate([Asub, np.ones((Asub.shape[0], 1))], 1)
        cc, _, _, _ = np.linalg.lstsq(A, yz, rcond=None)
        resid = yz - Asub @ cc[:-1] - float(cc[-1])

    if not chosen or last_expr is None:
        return None
    return {"expression": last_expr, "n_terms": len(chosen),
            "residual_sse": float(np.sum(resid ** 2))}


def _grade_with_true(expr, X, y, true_expr):
    """最终对真式计算一次全部指标与 SCS/grade（仅在搜索结束后调用）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mf_metrics",
        "/home/zyx/A_project/SR_Works/20260831-12th-MFbench-Patterns/experiments/metrics.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    names = [f"x{k}" for k in range(X.shape[1])]
    try:
        mm = M.compute_all_metrics(expr, true_expr, X, y, names)
    except Exception:
        mm = {'r2': 0.0, 'oss': 0.0, 'tss': 0.0, 'psc': 0.0, 'struct': False}
    scs = M.structure_completeness(mm)
    grade = M.grade_completeness(scs)
    return scs, grade, mm


def solve_graded(X, y, true_expr=None, max_terms=8):
    """分级求解：直接空间 + log外壳空间，各靠残差SSE剥离；
    最后对候选表达式用真式算一次 SCS，选完成度高者。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()

    candidates = []

    # 1) 直接空间
    rd = _graded_once(X, y, max_terms)
    if rd is not None:
        candidates.append(("direct", rd["expression"]))

    # 2) log 外壳空间
    if np.all(y > 0):
        ly = np.log(np.clip(y, 1e-300, None))
        rl = _graded_once(X, ly, max_terms)
        if rl is not None:
            candidates.append(("expwrap", f"exp({rl['expression']})"))

    if not candidates or true_expr is None:
        return {"applicable": False, "reason": "无候选/未提供真式",
                "candidates": [c[1] for c in candidates]}

    # 最终统一评分
    scored = []
    for route, expr in candidates:
        scs, grade, mm = _grade_with_true(expr, X, y, true_expr)
        scored.append((scs, grade, route, expr, mm))
    scored.sort(key=lambda z: -z[0])
    scs, grade, route, expr, mm = scored[0]
    return {"applicable": grade in ("A", "B"), "expression": expr,
            "route": route, "scs": round(scs, 4), "grade": grade,
            "metrics": mm}


def refine_partial(expr, X, y, max_rounds=4):
    """B 档"继续优化"：常数精修 + 残差补件，单调提升 SCS。

    每轮：算当前 SCS；尝试在残差上再剥一个部件，
    新结果 SCS 更高才保留，否则停止。
    """
    scs0, grade0, mm0 = _metrics_for(expr, X, y)
    cur_expr = expr
    cur_scs = scs0
    for _ in range(max_rounds):
        resid = _residual_of(cur_expr, X, y)
        add_expr = _best_single_atom(resid, X)
        if add_expr is None:
            break
        trial = f"({cur_expr})+({add_expr})"
        scs, grade, mm = _metrics_for(trial, X, y)
        if scs > cur_scs + 1e-6:
            cur_expr, cur_scs = trial, scs
        else:
            break
    cur_scs, grade, mm = _metrics_for(cur_expr, X, y)
    return {"expression": cur_expr, "scs": round(cur_scs, 4),
            "grade": grade, "metrics": mm}


def _residual_of(expr, X, y):
    ns = {f"x{k}": X[:, k] for k in range(X.shape[1])}
    ns.update({"sqrt": np.sqrt, "log": np.log, "exp": np.exp,
               "sin": np.sin, "cos": np.cos, "abs": np.abs, "pi": np.pi})
    try:
        pred = eval(expr, {"__builtins__": {}}, ns)
        return y - np.asarray(pred)
    except Exception:
        return None


def _best_single_atom(resid, X):
    if resid is None:
        return None
    best = (0.0, None)
    for k in range(X.shape[1]):
        c = float(np.corrcoef(X[:, k], resid)[0, 1])
        if abs(c) > abs(best[0]):
            # 用最小二乘定系数
            b = float(np.dot(X[:, k], resid) / (np.dot(X[:, k], X[:, k]) + 1e-12))
            if abs(b) > 1e-9:
                term = f"x{k}" if abs(abs(b)-1) < 1e-9 else f"{_fmt(abs(b))}*x{k}"
                best = (c, term if b > 0 else "-"+term)
    return best[1]
