import numpy as np
from tqdm import tqdm
import torch
import warnings
import argparse

from utils import *

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Parameters
# code parameters
d = 200//4
sigma2 = 1
nb_avg = 20

N_max_ridge = 6000//4 # maximal nb of datapoints
N_max_sgd = 2000//4
n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16) # nb of datapoints for evaluations
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

n_fine_tune_params = 10 # nb of hyperparameters tested

lambda_ = 1e-5*np.ones(len(n_ridge))
lambdas_ = np.logspace(-5,1,n_fine_tune_params, base=10.0)

intern_dim = 10
depth = -1 # Single Layer
optimizer = 'SGD'
learning_rate = 0.001*np.ones(len(n_sgd))
learning_rates = np.logspace(-6,-2,n_fine_tune_params)

which_h = 1 # 1 or 2 -> i**(-...)
which_w = 10 # 0, 1 or 10 -> i**(-...)

GENERATE_RIDGE = True # generate ridge weights
GENERATE_SGD = False # generate SGD weights
USE_SAVED_PARAMS = True # use the params saved
SAME_LR = True # all learning rates are the same in learning_rate, for faster computations

# saving paths
SAVE_DIR_SGD = 'data/SGD/'
SAVE_DIR_RIDGE = 'data/Ridge/'
filename_iterates = f'iterates_H{which_h}_w{which_w}_d{d}.npy'
SAVE_RIDGE_ITERATE = SAVE_DIR_RIDGE + filename_iterates
SAVE_SGD_ITERATE = SAVE_DIR_SGD + filename_iterates
SAVE_RIDGE_LAMBDA = SAVE_DIR_RIDGE + f'lambda_H{which_h}_w{which_w}_d{d}.npy'
SAVE_SGD_GAMMA = SAVE_DIR_SGD + f'gamma_H{which_h}_w{which_w}_d{d}.npy'

### Argument parser
parser = argparse.ArgumentParser(prog='Data generation for implicit regularization of SGD study')
parser.add_argument('--SGD', action=argparse.BooleanOptionalAction, default=GENERATE_SGD,
                    help='generate SGD data')
parser.add_argument('--Ridge', action=argparse.BooleanOptionalAction, default=GENERATE_RIDGE,
                    help='generate Ridge data')
parser.add_argument('-h', default=which_h, choices=[1,2], help='matrix H1 or H2 to use')
parser.add_argument('-w', default=which_w, choices=[0,1,10], help='true vector w0, w1 or w10')
parser.add_argument('-d', default=d, type=int, help='dimension of the data')
parser.add_argument('--N_ridge', default=N_max_ridge, type=int, help='Max number of data for ridge')
parser.add_argument('--N_SGD', default=N_max_sgd, type=int, help='Max number of data for SGD')
parser.add_argument('--depth', default=depth, type=int, help='depth of MLP (i.e nb of hidden layers), -1 for single layer')
parser.add_argument('--intern_dim', default=intern_dim, type=int, help='intern dimension of hidden layers')

args = parser.parse()

GENERATE_RIDGE = args.Ridge
GENERATE_SGD = args.SGD
which_h = args.h
which_w = args.w
d = args.d
N_max_ridge = args.N_ridge
N_max_sgd = args.N_SGD
depth = args.depth
intern_dim = args.intern_dim

### Begin experiment
# Initialization
w_ridge = np.zeros((nb_avg, len(n_ridge), d))
w_sgd = np.zeros((nb_avg, len(n_sgd), d))

if USE_SAVED_PARAMS: # load best coeffs
    if GENERATE_SGD:
        try:
            load = np.load(SAVE_SGD_GAMMA)
            if np.ndim(load) == 0:
                learning_rate = load*np.ones(len(n_sgd))
            elif np.ndim(load) == 1:
                learning_rate = load
            else:
                learning_rate = load[0]
                if (n_sgd != load[1]).any():
                    warnings.warn('Learning rates fine-tuned for different values of n_sgd', UserWarning)
        except FileNotFoundError:
            print(f'No learning rates found - using default lr={learning_rate[0]}')
    if GENERATE_RIDGE:
        try:
            load = np.load(SAVE_RIDGE_LAMBDA)
            if np.ndim(load) == 0:
                lambda_ = load*np.ones(len(n_ridge))
            elif np.ndim(load) == 1:
                lambda_ = load
            else:
                lambda_ = load[0]
                if (n_ridge != load[1]).any():
                    warnings.warn('Lambda values fine-tuned for different values of n_ridge', UserWarning)
        except FileNotFoundError:
            print(f'No lambdas found - using default lambda={lambda_[0]}')

# Averaging results
for i in tqdm(range(nb_avg)):
    ### Data generation
    w_true = np.float_power(np.arange(1,d+1), -which_w) # true parameter
    H = np.diag(np.float_power(np.arange(1,d+1), -which_h))
    data = np.random.multivariate_normal(
        np.zeros(d),
        H,
        size=N_max_ridge) # shape (N_max_ridge,d)

    observations = [np.random.normal(
        np.dot(w_true, x),
        np.sqrt(sigma2))
        for x in data]
    observations = np.array(observations) # shape (n,)

    ### Solving the problem
    # Ridge
    if GENERATE_RIDGE:
        for j,n in enumerate(n_ridge): # generate ridge solution for each n
            w_ridge[i,j,:] = ridge(data[:n,:], observations[:n], lambda_=lambda_[j])

    # SGD
    if GENERATE_SGD:
        input_Tensor = torch.from_numpy(data).to(device, dtype=torch.float32)
        output_Tensor = torch.from_numpy(observations).to(device, dtype=torch.float32)

        model = MultiLayerPerceptron(input_dim=d,
                                     intern_dim=intern_dim,
                                     output_dim=1,
                                     depth=depth,
                                     init='zero',
                                     isBiased = False,
                                    ).to(device)
        if SAME_LR: # train once, and organize weights after
            ws = train(model,
                      input_Tensor,
                      output_Tensor,
                      lossFct = nn.MSELoss(),
                      optimizer=optimizer,
                      epochs=N_max_ridge,
                      batch_size=None,
                      return_vals=False,
                      return_ws=True,
                      init_norm = None,
                      lr = learning_rate[0])
            for j,n in enumerate(n_sgd): # average the appropriate iterates
                w_sgd[i,j,:] = np.mean(ws[n//2:n,:], axis=0)
        else: # train for each n
            for j,n in enumerate(n_sgd):
                ws = train(model,
                      input_Tensor,
                      output_Tensor,
                      lossFct = nn.MSELoss(),
                      optimizer=optimizer,
                      epochs=n,
                      batch_size=None,
                      return_vals=False,
                      return_ws=True,
                      init_norm = None,
                      lr = learning_rate[j])
                w_sgd[i,j,:] = np.mean(ws[n//2:,:], axis=0)

# Save results
if GENERATE_RIDGE:
    np.save(SAVE_RIDGE_ITERATE, w_ridge)
if GENERATE_SGD:
    np.save(SAVE_SGD_ITERATE, w_sgd)