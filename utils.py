import numpy as np
import numbers
from collections import OrderedDict
import torch
import torch.nn as nn
import math
from tqdm import tqdm

DIRPATH = 'models/'

### Define objective function
def objective(A,b,x):
    return ((A@x-b)**2).sum()/(A.shape[0])

def objective_nonlinear(A,b,Xs):
    x = Xs[0]
    for k in range(1,len(Xs)):
        x = Xs[k]@x
    x = np.squeeze(x)
    return ((A@x-b)**2).sum()/(A.shape[0])


### Ridge regression (L2 penalization)
def ridge(A,b,lambda_):
    '''
    Solves min 1/n*||Ax-b||^2+lambda_*||x||^2

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
            inv = np.linalg.inv(2/n*A.T@A+2*lambda_*np.eye(d))
            res = inv@(A.T)@b*2/n
        else: # overparametrized
            inv = np.linalg.inv(2/n*A@(A.T)+2*lambda_*np.eye(n))
            res = A.T@inv@b*2/n
    else: # multi-dim
        raise NotImplementedError('Penalization per dimension not yet implemented')
    return res

def solve_nonlinear_ridge(Ws, b, lambda_):
    '''
    Solves min 1/n*||Ws[-1]@...@W[0]*x - b||^2 + ||lambda_*x||^2

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
        
### MLP Can be a SLN with depth = -1
class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, input_dim, intern_dim, output_dim, depth = 2, isBiased = False, init='uniform'):
        
        self.depth = depth
        self.intern_dim = intern_dim
        
        if depth ==-1:
            super(MultiLayerPerceptron, self).__init__()
            self.layer = nn.Linear(input_dim, output_dim, bias=isBiased)
            if init == 'uniform':
                self.layer.weight.data.uniform_(0.0, 1.0)
            if init == 'zero':
                self.layer.weight.data.fill_(0)
        else:
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
                layer.bias.data.uniform_(-stdv, stdv)

class NonLinearMLP(nn.Sequential):
    def __init__(self, input_dim, intern_dim, output_dim, depth=2, isBiased=False, init='uniform'):
        
        self.depth = depth
        if depth == -1:
            super(MultiLayerPerceptron, self).__init__()
            self.layer = nn.Linear(input_dim, output_dim, bias=isBiased)
            if init == 'uniform':
                self.layer.weight.data.uniform_(0.0, 1.0)
            if init == 'zero':
                self.layer.weight.data.fill_(0)
            self.activation = nn.ReLU()
        else:
            dict = OrderedDict([("input", nn.Linear(input_dim, intern_dim, bias=isBiased)),
                                ("activation", nn.ReLU())])
            for i in range(depth):
                dict.update({str(i): nn.Linear(intern_dim, intern_dim, bias=isBiased),
                             "activation_" + str(i): nn.ReLU()})
            dict.update({"output": nn.Linear(intern_dim, output_dim, bias=isBiased)})
            super().__init__(dict)

        self.reset_init_weights_biases()  # so that we do not use a default initialization

    def reset_init_weights_biases(self, norm = None):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                if norm == None:
                    stdv = 1. / math.sqrt(layer.weight.size(1))
                else :
                    stdv = norm
                
                layer.weight.data.uniform_(-stdv, stdv)
                if layer.bias is not None:
                    layer.bias.data.uniform_(-stdv, stdv)

### GD Optimizer
class GD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001):
        super(GD, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data
                    p.data -= group['lr'] * grad

### Return Weights of a model as Tensor
def get_param(model, d, device=torch.device("cpu")):
    w = torch.eye(d, device=device)
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = w@torch.transpose(layer.weight,0,1)
    return w.squeeze_()

### Trainning function
def train(model, input_data, output_data, untilConv = -1, lossFct = 'MSE', optimizer = 'SGD', lr=0.001, epochs = 20, batch_size=None, return_vals = 'error', return_ws = False, init_norm = None, save = True, debug = False, savename='model.pt'):
    '''
    return_vals: 'error', 'margin' or None/False
    '''
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
    elif optimizer == 'GD':
        optimizer = GD(model.parameters(), lr=lr)
    
    if lossFct == 'MSE' : lossFct = nn.MSELoss()
    
    post_loss = 0
    n,d = input_data.shape

    if init_norm is not None:
        model.reset_init_weights_biases(init_norm)
    
    if return_vals:
        vals = np.zeros(epochs)
    
    if return_ws:
        ws = np.zeros((epochs,d))
        
    if batch_size is not None:
        n_batches = n//batch_size

    for i in range(epochs):
        rand_idx = torch.randperm(n) # permutation of data samples
        if batch_size is not None:
            loss = 0
            for t in range(n_batches):
                idx = rand_idx[t*batch_size:(t+1)*batch_size]
                y_pred = model(input_data[idx,:]).squeeze_()
                loss += lossFct(y_pred, output_data[idx])
        else:
            y_pred = model(input_data[rand_idx,:]).squeeze_()
            loss = lossFct(y_pred, output_data[rand_idx])
        
        if return_ws or return_vals=='margin':
            w = get_param(model,d).detach()
            ws[i,:] = w.cpu()

        if return_vals == 'error':
            vals[i] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        
        if abs(post_loss - loss.item()) <=untilConv:
            print("Convergence")
            break
        post_loss = loss.item()
        
        optimizer.step()

        if debug:
            if (i+1)%(epochs/debug) == 0:
                print(f"Epoch: {i+1}   Loss: {loss.item():.3e}")

    if save:
        torch.save(model.state_dict(), DIRPATH+savename)
    
    if return_vals and not return_ws:
        return vals
    elif return_ws and not return_vals:
        return ws
    elif return_ws and return_vals:
        return vals, ws

def cross_validation(n, k=10, homogeneous=True, sizes=None):
    '''
    Computes masks for training and testing datasets

    Arguments
    ---
    n: int
        number of datapoints
    k: int
        number of groups for cross-validation
    homogeneous: bool
        True to have homogeneous groups (in size)
        False to generate uneven groups (see)
    sizes: (k,) array like of ints
        size of each train group, if homogeneous is True

    Returns 
    ---
    train_masks: list
        list of k index masks for training groups
    test_masks: list
        list of k index masks for testing groups
    '''
    if homogeneous:
        train_masks = []
        test_masks = np.array_split(np.arange(n),k)
        for i in range(k):
            train = list(set(range(n))-set(test_masks[i])) # complement of test
            train_masks.append(train)
    else:
        assert k == len(sizes), f'sizes must be of length k={k}'
        train_masks = []
        test_masks = []
        for i in range(k):
            train_masks.append(np.random.choice(n,size=sizes[i]))
            test_masks.append(list(set(range(n))-set(train_masks[i]))) 
        
    return train_masks, test_masks

### Generate n_vector of dim_p with multivariate normal distribution
def generate_data(p = 200, n = 6000, sigma2 = 1, which_w=1, which_h=1):

    H = np.diag(np.float_power(np.arange(1,p+1), -which_h))
    data = np.random.multivariate_normal(
        np.zeros(p),
        H,
        size=n) # shape (n,p)

    w_true =  np.float_power(np.arange(1,p+1), -which_w)

    observations = [np.random.normal(
        np.dot(w_true, x),
        sigma2)
        for x in data]
    
    observations = np.array(observations)

    return data, observations
