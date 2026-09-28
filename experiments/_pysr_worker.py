import sys
import numpy as np
sys.path.insert(0, "/home/zyx/A_project/SR_Works/20260927-14th-sr/src")
from scesr import gen
from pysr import PySRRegressor

idx = int(sys.argv[1])
g = gen(idx, n=400, sigma=0.0, seed=0)
X = np.asarray(g["X"], np.float64).T
y = np.asarray(g["yc"], np.float64)

model = PySRRegressor(
    niterations=60, populations=15, population_size=35,
    maxsize=20, binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "exp", "log"],
    verbosity=0, progress=False, model_selection="best",
    random_state=0, parallelism="serial", timeout_in_seconds=300)
model.fit(X, y)
print("RES " + str(model.sympy()))
