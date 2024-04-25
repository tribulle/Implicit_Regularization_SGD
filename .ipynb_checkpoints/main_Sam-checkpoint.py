import torch
import matplotlib.pyplot as plt
import numpy as np

from utils_Sam import *

### Global variables
np.random.seed(42)


n,d = 10,2
lambda_ = 1.


A = np.random.rand(n,d)
Ws = [np.random.rand(d,d) for _ in range(2)] + [A]
b = np.random.rand(n)

x = ridge(A,b,lambda_)
obj = objective(A,b,x)

x_nonlin = solve_nonlinear_ridge(Ws,b,lambda_)
obj_nonlin = objective_nonlinear(Ws,b,x_nonlin)

print(f'Solution: {x}')
print(f'Objective: {obj:.3e}')

print(f'Solution non-linear: {x_nonlin}')
print(f'Objective non-linear: {obj_nonlin:.3e}')