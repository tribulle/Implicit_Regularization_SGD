import numpy as np
import numbers

### Define objective function
def objective(A,b,x):
    return np.sum((A@x-b)**2)/(2*A.shape[0])

def objective_nonlinear(Ws,b,x):
    A = Ws[0]
    for k in range(1,len(Ws)):
        A = Ws[k]@A
    return np.sum((A@x-b)**2)/(2*A.shape[0])

### Ridge regression (L2 penalization)
def ridge(A,b,lambda_):
    '''
    Returns the solution of min 1/(2*n_samples)*||Ax-b||^2+||lambda_*x||^2 where
    lambda_ is 0D, 1D (A.shape[0]=n_samples,) or 2D (A.shape)
    '''
    if isinstance(lambda_, numbers.Number): # real number
        inv = np.linalg.inv(A.T@A+lambda_**2*np.eye(A.shape[1]))
    else: # multi-dim
        if lambda_.ndim == 1: # assumes regularization per dimension
            inv = np.linalg.inv(A.T@A+np.eye(lambda_**2))
        elif lambda_.ndim == 2: # assumes matrix regularization
            inv = np.linalg.inv(A.T@A+lambda_.T@lambda_)
    res = 1/A.shape[0]*inv@(A.T)@b
    return res

def solve_nonlinear_ridge(Ws, b, lambda_):
    '''
    Solves min ||Ws[-1]@...@W[0]*x - b||^2 + ||lambda_*x||^2
    '''
    A = Ws[0]
    for k in range(1,len(Ws)):
        A = Ws[k]@A
    x = ridge(A,b,lambda_)
    return x

