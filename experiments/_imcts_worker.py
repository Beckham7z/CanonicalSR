import sys
import numpy as np
sys.path.insert(0, "/home/zyx/A_project/SR_Works/MCTS-4-SR-T2/build_t2")
sys.path.insert(0, "/home/zyx/A_project/SR_Works/20260927-14th-sr/src")
from scesr import gen
import imcts_py_t2 as imcts

idx = int(sys.argv[1])
g = gen(idx, n=400, sigma=0.0, seed=0)
X = np.asarray(g["X"], np.float64)
y = np.asarray(g["yc"], np.float64)
cfg = imcts.RegressorConfig()
cfg.ops = ["+", "-", "*", "/", "sqrt", "log", "exp", "sin", "cos", "R"]
cfg.max_evals = 60000
cfg.max_time_sec = 8.0
cfg.max_depth = 12
try:
    cfg.complexity_penalty = 0.002
except Exception:
    pass
res = imcts.Regressor(X.astype(np.float32), y.astype(np.float32), cfg).fit(seed=0)
print("RES " + (res.expression or ""))
