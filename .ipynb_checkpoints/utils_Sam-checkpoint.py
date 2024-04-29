import numpy as np
import numbers
from collections import OrderedDict
import torch
import torch.nn as nn
import math

DIRPATH = 'models/'

### Define objective function
def objective(A,b,x):
    return ((A@x-b)**2).sum()/(2*A.shape[0])

def objective_nonlinear(A,b,Xs):
    x = Xs[0]
    for k in range(1,len(Xs)):
        x = Xs[k]@x
    x = np.squeeze(x)
    return ((A@x-b)**2).sum()/(2*A.shape[0])

### Ridge regression (L2 penalization)
def ridge(A,b,lambda_):
    '''
    Solves min 1/(2*n)*||Ax-b||^2+lambda_*||x||^2

    Parameters
    ----------
    A: (n,d) array
        Design matrix
    
    b: (n,) array
        Observations
    
    lambda_: number
        L2 penalization parameter
    
    Returns
    -------
    x: (d,) array
        The solution of the ridge regression
    '''
    n,d = A.shape
    if isinstance(lambda_, numbers.Number): # real number
        if d<=n: # underparametrized
            inv = np.linalg.inv(1/n*A.T@A+2*lambda_*np.eye(d))
            res = inv@(A.T)@b/n
        else: # overparametrized
            inv = np.linalg.inv(1/n*A@(A.T)+2*lambda_*np.eye(n))
            res = A.T@inv@b/n
    else: # multi-dim
        raise NotImplementedError('Penalization per dimension not yet implemented')
    return res

def solve_nonlinear_ridge(Ws, b, lambda_):
    '''
    Solves min ||Ws[-1]@...@W[0]*x - b||^2 + ||lambda_*x||^2

    Parameters
    ----------
    Ws: list of (n,d) arrays
        Weight matrices
    
    b: (n,) array
        Observations
    
    lambda_: number, (n,) or (n,d) array
        L2 penalization parameter
    
    Returns
    -------
    x: (d,) array
        The solution of the non-linear ridge regression (see above)
    '''
    A = Ws[0]
    for k in range(1,len(Ws)):
        A = Ws[k]@A
    x = ridge(A,b,lambda_)
    return x

### CustomLoss
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, inputs, targets):
        loss = torch.nn.functional.pairwise_distance(inputs,targets).square().mean()/2
        return loss
        
### MLP
class Multi_Layer_Perceptron(nn.Sequential):
    def __init__(self, input_dim, intern_dim, output_dim, depth = 2, isBiased = False):
        
        dict = OrderedDict([("input",nn.Linear(input_dim,intern_dim, bias=isBiased))])
        for i in range(depth):
            dict.update({str(i) : nn.Linear(intern_dim,intern_dim,bias=isBiased)})
        dict.update({"output" : nn.Linear(intern_dim,output_dim,bias=isBiased)})

        super().__init__(dict)

        self.reset_init_weights_biases() # so that we do not use a default initialization

    def reset_init_weights_biases(self, norm = None):
        for layer in self.children():
            if norm == None:
                stdv = 1. / math.sqrt(layer.weight.size(1))
            else :
                stdv = norm
            
            layer.weight.data.uniform_(-stdv, stdv)
            if layer.bias is not None:
                layer.biases.data.uniform_(-stdv, stdv)

class GD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super(GD, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data
                    p.data -= group['lr'] * grad

def train(model, input_data, output_data, lossFct = nn.MSELoss(), optimizer = 'SGD', lr=0.001, epochs = 20, return_vals = False, init_norm = None, save = True, debug = False, savename='model.pt'):

    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
    elif optimizer == 'GD':
        optimizer = GD(model.parameters(), lr=lr)
    
    if init_norm is not None:
        model.reset_init_weights_biases(init_norm)
    
    if return_vals:
        errors = np.zeros(epochs)

    for i in range(epochs):
        y_pred = model(input_data).squeeze_()
        loss = lossFct(y_pred, output_data)

        if return_vals:
            errors[i] = loss.item()

        if math.isnan(loss.item()):
            print(f"Epoch: {i+1}   Loss: {loss.item()}")
            break
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if debug:
            if (i+1)%(epochs/debug) == 0:
                print(f"Epoch: {i+1}   Loss: {loss.item():.3e}")
        
    if save:
        torch.save(model.state_dict(), DIRPATH+savename)
    
    if return_vals:
        return errors

### Comparison of models
def compare(input, output, w1, w2):
    res1 = objective(input, output, w1)
    res2 = objective(input, output, w2)
    print('Model 1:')
    print(f'   - objective: {res1:.3e}')
    print(f'   - weights norm: {np.linalg.norm(w1):.2f}')

    print('Model 2:')
    print(f'   - objective: {res2:.3e}')
    print(f'   - weights norm: {np.linalg.norm(w2):.2f}')