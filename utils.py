import numpy as np
import numbers
from collections import OrderedDict
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from scipy.stats import special_ortho_group
import csv

DIRPATH = 'models/'

### Define objective function
def objective(A,b,x):
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
            inv = np.linalg.inv(A.T@A+lambda_*np.eye(d)) # test for ridge classic
            res = inv@(A.T)@b
        else: # overparametrized
            inv = np.linalg.inv(A@(A.T)+lambda_*np.eye(n))
            res = A.T@inv@b
    else: # multi-dim
        raise NotImplementedError('Penalization per dimension not yet implemented')
    return res
        
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
    
def train_v2(model, input_data, output_data, untilConv = -1, lossFct = 'MSE', optimizer = 'SGD', lr=0.001, epochs = 20, batch_size=None, return_vals = 'error', return_ws = False, init_norm = None, save = True, debug = False, savename='model.pt'):
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
        
        y_pred = model(input_data[rand_idx[i],:]).squeeze_()
        loss = lossFct(y_pred, output_data[rand_idx[i]])
        
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
        False to generate uneven groups (see sizes)
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
        assert max(sizes) < n, f'a train group cannot be bigger than the number of points: k={max(sizes)}>n={n}'
        train_masks = []
        test_masks = []
        for i in range(k):
            train_masks.append(np.random.choice(n,size=sizes[i]))
            test_masks.append(list(set(range(n))-set(train_masks[i]))) 
        
    return train_masks, test_masks

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

def load_data_CSV(file_name = 'data/data.csv', n = 6000, normalize=True):

    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data_array = np.array(data, dtype=float)
    col = data_array.shape[1]
    
    if col <= 1: 
        print("ERROR NO OBSERVATION")
    if n is None:
        n = data_array.shape[0] -1
    elif n >= data_array.shape[0]: 
        print(f'Maximal number of data reached: loading {len(data_array):d} rows instead of n={n:d}')
        n = data_array.shape[0] -1
    
    if normalize:
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0)
        data_array = (data_array-means)/stds

    return data_array[:n,:col-1], data_array[:n, -1], means, stds

def os_command(file,
               ridge_bool=False,
               sgd_bool=False,
               w=0,
               h=1,
               d=50,
               N_ridge=1500,
               N_sgd=500,
               depth=-1,
               intern_dim=10,
               k=None,
               homogeneous=None,
               n_params=None,
               CV_ridge=None,
               CV_sgd=None):
    command = 'python '
    command += file + ' '
    if ridge_bool:
        command += '--Ridge '
    else:
        command += '--no-Ridge '
    if sgd_bool:
        command += '--SGD '
    else:
        command += '--no-SGD '
    command += (f'-w {w:d} ' +
                f'-H {h:d} ' +
                f'-d {d:d} ' +
                f'--N_ridge {N_ridge:d} ' +
                f'--N_SGD {N_sgd:d} ' +
                f'--depth {depth:d} ' +
                f'--intern_dim {intern_dim:d}'
                )
    if k is not None:
        command += f' -k {k:d}'
    if homogeneous is not None:
        if homogeneous:
            command += ' --homogeneous'
        else:
            command += ' --no-homogeneous'
    if n_params is not None:
        command += f' --n_params {n_params:d}'
    if CV_ridge is not None:
        if CV_ridge:
            command += ' --CV_ridge'
        else:
            commmand += ' --no-CV_ridge'
    if CV_sgd is not None:
        if CV_sgd:
            command += ' --CV_sgd'
        else:
            command += ' --no-CV_sgd'
    return command

def suffix_filename(ridge_bool=False,
                    sgd_bool=False,
                    w=0,
                    h=1,
                    d=50,
                    depth=-1,
                    intern_dim=10):
    assert ridge_bool or sgd_bool, 'One of sgd_bool or ridge_bool must be True'
    assert not(ridge_bool and sgd_bool), 'Both ridge_bool and sgd_bool cannot be True'

    if ridge_bool:
        suffix = f'_H{h}_w{w}_d{d}'
    else:
        if depth == -1: # single layer
            suffix = f'_H{h}_w{w}_d{d}'
        else:
            suffix = f'_H{h}_w{w}_d{d}_depth{depth}_indim{intern_dim}'

    return suffix

### Generate n_vector of dim_p with multivariate normal distribution, rotation, and random w_true
def generate_data_V2(p = 200, n = 6000, sigma2 = 1, which_w=1, which_h=1, w_random = 1, rotation = 1):

    R = special_ortho_group.rvs(p)
    H = np.diag(np.float_power(np.arange(1,p+1), -which_h))
    
    if rotation == 1:
        H = np.dot(R,H)
    
    data = np.random.multivariate_normal(
        np.zeros(p),
        H,
        size=n) # shape (n,p)
        
    if w_random != 1:
        w_true =  np.float_power(np.arange(1,p+1), -which_w)
    else:
        i = p**(-which_w)
        w_true = np.full((p),i)
        
    observations = [np.random.normal(
        np.dot(w_true, x),
        sigma2)
        for x in data]
    
    observations = np.array(observations)

    return data, observations
