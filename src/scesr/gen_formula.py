"""健壮的通用 MF-bench 数据生成器 v2。

核心：用 AST 把 python_expr 中对 var_ranges 变量的引用（包括 alpha_bar(T) 这种
函数形式的"伪函数变量"，以及 \\lambda / p_{t1} 这类 LaTeX 键）精确替换成
x0,x1,...，再安全 exec。AST 替换比正则可靠：能识别 Name 节点和"函数调用名等于
某变量键"的 Call 节点。

相较 v1，本版修复：
- LaTeX 变量键（\\lambda、p_{t1}、\\mathrm{Re}…）经 latex_to_plain/coerce 匹配，
  并支持唯一前缀别名（数据集里 \\lambda 常写成 lam）；
- 过滤派生键（p1-p2、D1+D2、s/t、k_s/D 等含算符的键）与非输入键
  （denominator/radicand/…）、以及作为输出（lhs）的键，避免把输出当输入造成泄漏；
- 真表达式（标准答案）做中间赋值内联：把 num/den/rad/head/n 等中间量展开为
  只含 x_i 的闭式，并数值裁决 max/min 守卫、把 norm 转 Abs，保证指标可解析。

主入口 generate(item, n=400, seed=0) -> dict(X[nv,n], y, expr_x, var_keys) 或 None。
"""
from __future__ import annotations
import ast
import re
import numpy as np


# ---------------------------------------------------------------------------
# LaTeX 变量名规范化（逻辑与 AAA_SR_DEMO/test_mfbench_30.py 保持一致，
# 此处内联以避免 import 该模块时连带拉起重型 imcts 依赖）
# ---------------------------------------------------------------------------

_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta",
    "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho",
    "sigma": "sigma", "tau": "tau", "upsilon": "upsilon", "phi": "phi",
    "chi": "chi", "psi": "psi", "omega": "omega", "Gamma": "Gamma",
    "Delta": "Delta", "Theta": "Theta", "Lambda": "Lambda", "Xi": "Xi",
    "Pi": "Pi", "Sigma": "Sigma", "Phi": "Phi", "Psi": "Psi", "Omega": "Omega",
}


def latex_to_plain(name):
    """把 var_ranges 的 LaTeX 键转成普通标识符（与 test_mfbench_30 同款）。"""
    s = str(name)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\dot\{([^}]*)\}", r"\1dot", s)
    s = re.sub(r"\\bar\{([^}]*)\}", r"\1bar", s)
    s = re.sub(r"\\overline\{([^}]*)\}", r"\1bar", s)
    s = re.sub(r"\\hat\{([^}]*)\}", r"\1hat", s)
    s = re.sub(r"\\vec\{([^}]*)\}", r"\1vec", s)
    s = re.sub(r"\\tilde\{([^}]*)\}", r"\1tilde", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    for gk, gv in _GREEK.items():
        s = re.sub(r"\\" + gk + r"\b", gv, s)
    for acc in ("dot", "bar", "hat", "vec", "tilde"):
        s = re.sub(r"\\" + acc + r"\{([^}]*)\}", r"\1" + acc, s)
        s = re.sub(r"\\" + acc + r"\s*([A-Za-z])\b", r"\1" + acc, s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    return s.strip()


def coerce_py_name(name):
    """把（已去 LaTeX 的）名字转成合法 Python 标识符（与 test_mfbench_30 一致）。"""
    if name is None:
        return name
    s = str(name).replace("{", "").replace("}", "")
    s = s.replace("\\", "").replace(" ", "")
    s = s.replace("+", "").replace("-", "").replace("_", "")
    if s and s[0].isdigit():
        s = "v" + s
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    return s or "v"


# ---------------------------------------------------------------------------
# 区间解析（与 test_mfbench_30 兼容，增强健壮性）
# ---------------------------------------------------------------------------

def parse_range(s):
    s = str(s).strip()
    m = re.match(r"[\[(]\s*(.+?)\s*,\s*(.+?)\s*[\])]", s)
    if not m:
        return None
    lo_s, hi_s = m.group(1), m.group(2)
    def one(t):
        t = t.strip()
        if t in ("+inf", "inf"):
            return 1e6
        if t == "-inf":
            return -1e6
        try:
            return float(t)
        except ValueError:
            return None
    lo, hi = one(lo_s), one(hi_s)
    if lo is None or hi is None:
        return None
    return lo, hi


# ---------------------------------------------------------------------------
# 键过滤 / 键->列映射
# ---------------------------------------------------------------------------

# 明确不是独立输入的键（是派生量或说明字段）
_NON_INPUT_KEYS = {"denominator", "numerator", "radicand", "lhs", "rhs"}


def _strip_braces(s):
    return re.sub(r"\{[^}]*\}", "", str(s))


def _is_derived_key(k):
    """含算术算符的键是派生量（p1-p2、D1+D2、s/t、p_{t1}/p_{t2}、k_s/D）。"""
    s = _strip_braces(k)
    return bool(re.search(r"[+\-*/]", s))


def _lhs_and_free(expr):
    """返回 (输出名集合, 自由变量集合)。自由变量=被使用但未在代码里定义的标识符。"""
    tree = ast.parse(expr)
    lhs, defined = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    lhs.add(t.id)
                    defined.add(t.id)
        elif isinstance(n, ast.For):
            if isinstance(n.target, ast.Name):
                defined.add(n.target.id)
        elif isinstance(n, ast.FunctionDef):
            defined.add(n.name)
            for a in n.args.args:
                defined.add(a.arg)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return lhs, used - defined


def _final_target(expr):
    """模块顶层最后一个赋值的目标名（即公式输出）。"""
    tree = ast.parse(expr)
    tgt = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    tgt = t.id
    return tgt


def _subtractive_orders(var_keys):
    """从形如 'p1-p2' 的派生键推断输入间的序约束。

    若键 'A-B' 对应区间下界>=0，则数据应满足 A>=B；上界<=0 则 A<=B。
    返回 [(A_key, B_key, 'ge'|'le')]。
    """
    orders = []
    keyset = set(var_keys)
    for k in var_keys:
        s = _strip_braces(k)
        m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*", s)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if a in keyset and b in keyset and a != b:
            orders.append((a, b))
    return orders


def _spellings(key):
    """一个键可能出现在表达式里的各种合法写法（含 alpha_bar(T) 的函数基名）。"""
    plain = latex_to_plain(key)
    cands = {str(key), plain, coerce_py_name(plain), plain.replace("_", "")}
    for form in (str(key), plain):
        fb = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", form)
        if fb:
            cands.add(fb.group(1))
    return {c for c in cands if c and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c)}


def _build_key_map(expr, var_keys):
    """构建 {表达式中的拼写 -> 列号} 以及按序的实际输入键列表。

    仅保留**表达式真正引用到的键**（含函数基名 alpha_bar(T)->alpha_bar、唯一前缀
    别名 lam->lambda），并排除最终输出键与派生键，避免把噪声列/输出列当输入。
    """
    lhs, free = _lhs_and_free(expr)
    final = _final_target(expr)
    cand = [k for k in var_keys
            if k not in _NON_INPUT_KEYS and not _is_derived_key(k)]

    name_to_key = {}
    for k in cand:
        for sp in _spellings(k):
            name_to_key.setdefault(sp, k)

    # 唯一前缀别名：处理 lam/lambda 这类数据集命名不一致
    for nm in (free | lhs):
        if nm in name_to_key:
            continue
        hits = {name_to_key[sp] for sp in name_to_key
                if len(nm) >= 3 and len(sp) >= 3
                and (sp.startswith(nm) or nm.startswith(sp))}
        if len(hits) == 1:
            name_to_key[nm] = hits.pop()

    referenced = {name_to_key[nm] for nm in (free | lhs) if nm in name_to_key}

    inputs, key_to_col = [], {}
    for k in cand:
        if k not in referenced:
            continue
        if final is not None and final in _spellings(k):
            continue
        key_to_col[k] = len(inputs)
        inputs.append(k)

    spell_to_col = {}
    for k, col in key_to_col.items():
        for sp in _spellings(k):
            spell_to_col.setdefault(sp, col)
    for nm, kk in name_to_key.items():
        if kk in key_to_col:
            spell_to_col.setdefault(nm, key_to_col[kk])
    return spell_to_col, inputs


# ---------------------------------------------------------------------------
# AST 变量规范化
# ---------------------------------------------------------------------------

class _VarNormalizer(ast.NodeTransformer):
    def __init__(self, key_to_idx):
        self.key_to_idx = key_to_idx

    def _replace(self, key):
        if key in self.key_to_idx:
            return ast.Name(id=f"x{self.key_to_idx[key]}", ctx=ast.Load())
        return None

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Name(self, node):
        repl = self._replace(node.id)
        return repl if repl is not None else self.generic_visit(node)

    def visit_Attribute(self, node):
        # np.linalg.norm -> norm
        if (isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in ("np", "numpy")
                and node.value.attr == "linalg"
                and node.attr == "norm"):
            return ast.Name(id="norm", ctx=node.ctx)
        # np.<ufunc> / math.<ufunc> -> <ufunc>
        if isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy", "math", "cmath"):
            return ast.Name(id=node.attr, ctx=node.ctx)
        return self.generic_visit(node)

    def visit_Call(self, node):
        # 形如 alpha_bar(T)：函数名是变量键，整个叶子代表一个输入变量
        if isinstance(node.func, ast.Name):
            repl = self._replace(node.func.id)
            if repl is not None:
                return repl
        node.func = self.visit(node.func)
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [ast.keyword(arg=k.arg, value=self.visit(k.value))
                         for k in node.keywords]
        return node


def _normalize_code(expr, key_to_idx):
    tree = ast.parse(expr)
    norm = _VarNormalizer(key_to_idx)
    tree = norm.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# 安全执行
# ---------------------------------------------------------------------------

def _norm(a, axis=None):
    """逐样本范数；样本按列存放时 1-D 输入等价于 Abs(a)。"""
    a = np.asarray(a)
    if a.ndim <= 1:
        return np.abs(a)
    return np.sqrt(np.sum(a ** 2, axis=0))


def _scalar_max(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.where(a >= b, a, b)


def _scalar_min(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.where(a <= b, a, b)


def _safe_namespace(X):
    """X: [nv,n]。返回含 x0.. 与 numpy 函数的命名空间。"""
    nv, n = X.shape
    ns = {f"x{i}": X[i] for i in range(nv)}
    ns.update({
        "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "ln": np.log,
        "log10": np.log10, "log2": np.log2,
        "sin": np.sin, "cos": np.cos, "tan": np.tan, "abs": np.abs,
        "Abs": np.abs, "pi": np.pi, "E": np.e,
        "max": _scalar_max, "min": _scalar_min,
        "maximum": np.maximum, "minimum": np.minimum, "norm": _norm,
    })
    ns["np"] = np
    ns["__builtins__"] = {}
    return ns


def _exec_target(code, ns):
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()
             and not ln.strip().startswith(("import", "from"))]
    target = None
    for ln in reversed(lines):
        if "=" in ln and not ln.lstrip().startswith(("if ", "for ", "while ", "def ", "return")):
            target = ln.split("=", 1)[0].strip()
            break
    exec(code, ns)
    if target is None or target not in ns:
        return None
    y = np.asarray(ns[target], dtype=np.float64).ravel()
    return y, target


# ---------------------------------------------------------------------------
# 真表达式内联（把中间赋值展开为只含 x_i 的闭式）
# ---------------------------------------------------------------------------

def _build_true_expr(code, target, X):
    """用 sympy 符号执行 code，取 target 的闭式（天然按语句顺序处理变量覆盖），
    再把 Max/Min 守卫生成指标可解析的形式。"""
    import sympy as sp

    nv = X.shape[0]
    syms = sp.symbols([f"x{i}" for i in range(nv)], real=True)
    ns = {f"x{i}": syms[i] for i in range(nv)}
    ns.update({
        "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "log10": lambda a: sp.log(a) / sp.log(10),
        "log2": lambda a: sp.log(a) / sp.log(2),
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "abs": sp.Abs, "Abs": sp.Abs, "pi": sp.pi, "E": sp.E,
        "norm": sp.Abs,
        "max": sp.Max, "min": sp.Min,
        "maximum": sp.Max, "minimum": sp.Min,
    })
    exec(code, ns)
    if target not in ns:
        return ""
    expr = ns[target]
    return _resolve_guards(str(expr), X).strip()


def _split_args(s):
    """按顶层逗号切分（参数里无嵌套括号时足够）。"""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return [a.strip() for a in out]


def _match_paren(s, open_idx):
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _resolve_guards(expr, X):
    """消去 max/min/Max/Min，使真式只含 metrics 支持的函数。

    - 若某分支在全部样本上恒占优 -> 取该分支；
    - 否则用恒等式 max(a,b)=((a+b)+Abs(a-b))/2、min(a,b)=((a+b)-Abs(a-b))/2，
      保持数值等价且可被 metrics 解析。
    """
    nv = X.shape[0]
    ones = np.ones(len(X[0]))
    ns = {f"x{i}": X[i] for i in range(nv)}
    ns.update({"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "Abs": np.abs,
               "abs": np.abs, "sin": np.sin, "cos": np.cos, "tan": np.tan,
               "pi": np.pi, "log10": np.log10, "log2": np.log2,
               "Max": np.maximum, "Min": np.minimum,
               "max": np.maximum, "min": np.minimum,
               "maximum": np.maximum, "minimum": np.minimum,
               "__builtins__": {}})
    for _ in range(64):
        found = None
        for m in re.finditer(r"\b(Max|Min|max|min)\(", expr):
            open_idx = m.end() - 1
            close = _match_paren(expr, open_idx)
            if close < 0:
                continue
            args = _split_args(expr[open_idx + 1:close])
            if len(args) != 2:
                continue
            found = (m.start(), close, m.group(1), args)
            break
        if found is None:
            break
        start, close, fn, args = found
        try:
            va = np.asarray(eval(args[0], {"__builtins__": {}}, ns), float) * ones
            vb = np.asarray(eval(args[1], {"__builtins__": {}}, ns), float) * ones
        except Exception:
            break
        is_max = fn.lower() == "max"
        keep_first = (va >= vb) if is_max else (va <= vb)
        keep_second = (va <= vb) if is_max else (va >= vb)
        if np.all(keep_first):
            repl = args[0]
        elif np.all(keep_second):
            repl = args[1]
        else:
            # 恒等式：用 Abs 表示，保证 metrics 可解析
            sign = "+" if is_max else "-"
            repl = f"((({args[0]})+({args[1]})){sign}Abs(({args[0]})-({args[1]})))/2"
        expr = expr[:start] + "(" + repl + ")" + expr[close + 1:]
    return expr


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def generate(item, n=400, seed=0):
    vr = item["var_ranges"]
    var_keys = list(vr.keys())
    expr_raw = item["python_expr"]
    rng = np.random.default_rng(seed)

    try:
        key_to_idx, inputs = _build_key_map(expr_raw, var_keys)
    except Exception:
        return None
    if not inputs:
        return None

    # 采样
    X = np.zeros((len(inputs), n), dtype=np.float64)
    for i, key in enumerate(inputs):
        info = vr[key]
        bounds = parse_range(info.get("range")) if isinstance(info, dict) else None
        if bounds is None:
            bounds = (0.1, 10.0)
        lo, hi = bounds
        X[i] = rng.uniform(lo, hi, n)

    # 派生差键（p1-p2 等）给出输入间的物理序约束，按需排序保证 sqrt 定义域
    idx_of = {k: i for i, k in enumerate(inputs)}
    for a, b in _subtractive_orders(list(vr.keys())):
        if a not in idx_of or b not in idx_of:
            continue
        ia, ib = idx_of[a], idx_of[b]
        move = X[ib] > X[ia]
        X[ia], X[ib] = (np.where(move, X[ib], X[ia]),
                        np.where(move, X[ia], X[ib]))

    try:
        code = _normalize_code(expr_raw, key_to_idx)
    except Exception:
        return None

    ns = _safe_namespace(X)
    try:
        out = _exec_target(code, ns)
    except Exception:
        return None
    if out is None:
        return None
    y, target = out
    if not np.all(np.isfinite(y)) or np.std(y) < 1e-10:
        return None

    try:
        expr_x = _build_true_expr(code, target, X)
    except Exception:
        return None
    if not expr_x:
        return None

    return {
        "X": X, "y": y, "expr_x": expr_x, "var_keys": inputs, "target": target,
    }
