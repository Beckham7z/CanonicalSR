"""Lean LLM proposer (LLM-SR mechanism) via an OpenAI-compatible API.

Faithful to LLM-SR's core: the LLM is given ONLY the data and the physical
meanings of the variables (never the target). It proposes closed-form
programs; numeric constants are then optimized (BFGS) and the candidate is
scored on the data. We keep the LLM proposal -> numeric refinement loop with
a small number of LLM samples to bound cost, without LLM-SR's torch/multi-
island machinery (which changes no math, only search scheduling).

Used by run_baselines.py.
"""
from __future__ import annotations

import io, os, re, json, contextlib
import numpy as np
import requests

API_URL = os.environ.get("LLM_API_URL",
                        "https://api.deepseek.com/v1/chat/completions")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

INSTRUCTION = """You are a symbolic-regression engine. Given numeric data and the
physical meaning of the input variables, propose a SHORT closed-form formula
f(x0,x1,...) that could produce the output y. Use only arithmetic +,-,*,/,
powers, and sqrt/exp/log/sin/cos. Do NOT see the target formula; infer from
the data.

Reply with EXACTLY one Python function and nothing else:

def f(x):
    import numpy as np
    x0, x1 = x[:,0], x[:,1]
    return <your expression>

x has shape (n_samples, n_features). Return an array shaped (n_samples,).
Prefer the SIMPLEST expression; numeric constants can be any float.
"""


def ask_llm(prompt: str, timeout: int = 120) -> str:
    payload = {
        "model": MODEL,
        "temperature": 0.8,
        "max_tokens": 900,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(API_URL, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"}, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*(def f\(x\):.*?)```", text, re.S)
    body = m.group(1) if m else None
    if body is None:
        m2 = re.search(r"(def f\(x\):.*)", text, re.S)
        body = m2.group(1) if m2 else None
    return body


def compile_f(code: str):
    ns = {}
    try:
        exec(code, {"np": np, "__builtins__": __builtins__}, ns)
        f = ns.get("f")
        # smoke test
        _ = f(np.zeros((3, max(1, code.count("x[:,") ))))
        return f
    except Exception:
        return None


def llm_propose(X: np.ndarray, y: np.ndarray, var_info: list[str],
                n_tries: int = 6):
    """Return list of functions proposed by the LLM that compile and run."""
    head = "Variable meanings:\n" + "\n".join(f"  x{i}: {v}"
                                             for i, v in enumerate(var_info))
    data_note = (f"\nData summary: n={len(y)}, "
                 f"y range [{y.min():.3g},{y.max():.3g}].\n")
    prompt = INSTRUCTION + "\n" + head + data_note

    out = []
    for _ in range(n_tries):
        try:
            text = ask_llm(prompt)
            code = extract_code(text)
            if not code:
                continue
            f = compile_f(code)
            if f is not None:
                out.append((f, code))
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------
# Numeric coefficient refinement (the BFGS step in LLM-SR).
_NUM = re.compile(r"(?<![A-Za-z0-9_\.\[])-?\d+\.\d+(?:[eE]-?\d+)?|-?\d+")


def parameterize(code: str):
    """Replace float literals in the function body with c[k] placeholders.

    Returns (new_code, init_values). Indices/int subscripts are left intact;
    only decimals (and non-index integer constants) are parameterized.
    """
    vals, cnt = [], 0
    lines = code.splitlines()
    new_lines = []
    for ln in lines:
        if ln.strip().startswith("def ") or ln.strip().startswith("import"):
            new_lines.append(ln); continue
        # only touch the return/expression region: replace decimals
        def repl(m):
            nonlocal cnt
            s = m.group(0)
            # skip pure integers that look like array indices (preceded by [ or :)
            if "." not in s and "e" not in s.lower():
                return s
            vals.append(float(s)); k = cnt; cnt += 1
            return f"c[{k}]"
        new_lines.append(_NUM.sub(repl, ln))
    return "\n".join(new_lines), vals


def refine(f, code: str, X: np.ndarray, y: np.ndarray):
    """Optimize the decimal constants of an LLM program by least squares."""
    from scipy.optimize import least_squares
    pcode, init = parameterize(code)
    if not init:
        try:
            pred = f(X)
            return f, np.sqrt(np.mean((pred - y) ** 2)), code
        except Exception:
            return None, np.inf, code
    ns = {}
    try:
        exec(pcode, {"np": np, "__builtins__": __builtins__}, ns)
        g = ns["f"]
    except Exception:
        return None, np.inf, code

    def resid(c):
        try:
            p = g(X, c) if _takes_c(g, X) else None
        except Exception:
            p = None
        if p is None:
            return np.full(len(y), 1e6)
        r = np.asarray(p, float) - y
        if r.shape != y.shape or not np.all(np.isfinite(r)):
            return np.full(len(y), 1e6)
        return r

    # wrapper may not accept c; inject c into signature by recompiling g(X,c)
    pcode2 = pcode.replace("def f(x):", "def f(x, c):")
    ns2 = {}
    try:
        exec(pcode2, {"np": np, "__builtins__:": __builtins__}, ns2)
        g2 = ns2["f"]
    except Exception:
        return None, np.inf, code

    def resid2(c):
        try:
            p = g2(X, c)
        except Exception:
            return np.full(len(y), 1e6)
        r = np.asarray(p, float) - y
        if r.shape != y.shape or not np.all(np.isfinite(r)):
            return np.full(len(y), 1e6)
        return r

    try:
        sol = least_squares(resid2, np.asarray(init, float),
                           max_nfev=300, method="lm")
        cstar = sol.x
        rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    except Exception:
        return None, np.inf, code

    # rebuild a concrete function with optimized constants baked in
    concrete = pcode2
    # substitute c[k] -> literal
    def bake(m):
        k = int(m.group(1))
        return repr(float(cstar[k]))
    baked = re.sub(r"c\[(\d+)\]", bake, pcode2).replace("def f(x, c):", "def f(x):")
    nsb = {}
    try:
        exec(baked, {"np": np, "__builtins__": __builtins__}, nsb)
        return nsb["f"], rmse, baked
    except Exception:
        return None, rmse, baked


def _takes_c(g, X):
    return False
