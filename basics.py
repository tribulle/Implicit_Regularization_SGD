import numpy as np
import torch
import matplotlib.pyplot as plt
import numbers

### Global variables
np.random.seed(42)

n,d = 10,2
lambda_ = 1.

### Define objective function
def objective(A,b,x):
    return np.sum((A@x-b)**2, axis=-1)

### Ridge regression (L2 penalization)
def ridge(A,b,lambda_):
    '''
    Returns the solution of min ||Ax-b||^2+||lambda_*x||^2 where
    lambda_ is 0D, 1D (A.shape[0],) or 2D (A.shape)
    '''
    if isinstance(lambda_, numbers.Number): # real number
        inv = np.linalg.inv(A.T@A+lambda_**2*np.eye(A.shape[1]))
    else: # multi-dim
        if lambda_.ndim == 1: # assumes regularization per dimension
            inv = np.linalg.inv(A.T@A+np.eye(lambda_**2))
        elif lambda_.ndim == 2: # assumes matrix regularization
            inv = np.linalg.inv(A.T@A+lambda_.T@lambda_)
    res = inv@(A.T)@b
    return res

if __name__=='__main__':
    A = np.random.rand(n,d)
    b = np.random.rand(n)

    x = ridge(A,b,lambda_)
    obj = objective(A,b,x)
    print(f'Solution: {x}')
    print(f'Objective: {obj:.3e}')