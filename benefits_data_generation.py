import numpy as np
from tqdm import tqdm
import torch
import warnings
import argparse

from utils import *

np.random.seed(9)
torch.manual_seed(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Parameters
# code parameters
d = 50
sigma2 = 1
nb_avg = 20

N_max_ridge = 1500 # maximal nb of datapoints
N_max_sgd = 500
n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16) # nb of datapoints for evaluations
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

n_fine_tune_params = 10 # nb of hyperparameters tested

lambda_ = 1e-5*np.ones(len(n_ridge)) # default lambda for tests
learning_rate = 0.001*np.ones(len(n_sgd)) # default learning rates for tests

intern_dim = 10
depth = -1 # Single Layer
optimizer = 'SGD'

which_h = 1 # 1 or 2 -> i**(-...)
which_w = 10 # 0, 1 or 10 -> i**(-...)

GENERATE_RIDGE = True # generate ridge weights
GENERATE_SGD = True # generate SGD weights
USE_SAVED_PARAMS = True # use the params saved
SAME_LR = True # all learning rates are the same in learning_rate, for faster computations

if __name__=='__main__':
    ### Argument parser
    parser = argparse.ArgumentParser(prog='Data generation for implicit regularization of SGD study')
    parser.add_argument('--SGD', action=argparse.BooleanOptionalAction, default=GENERATE_SGD,
                        help='generate SGD data')
    parser.add_argument('--Ridge', action=argparse.BooleanOptionalAction, default=GENERATE_RIDGE,
                        help='generate Ridge data')
    parser.add_argument('-H', default=which_h, choices=[1,2], type=int, help='matrix H1 or H2 to use')
    parser.add_argument('-w', default=which_w, choices=[0,1,10], type=int, help='true vector w0, w1 or w10')
    parser.add_argument('-d', default=d, type=int, help='dimension of the data')
    parser.add_argument('--N_ridge', default=N_max_ridge, type=int, help='Max number of data for ridge')
    parser.add_argument('--N_SGD', default=N_max_sgd, type=int, help='Max number of data for SGD')
    parser.add_argument('--depth', default=depth, type=int, help='depth of MLP (i.e nb of hidden layers), -1 for single layer')
    parser.add_argument('--intern_dim', default=intern_dim, type=int, help='intern dimension of hidden layers')

    args = parser.parse_args()

    GENERATE_RIDGE = args.Ridge
    GENERATE_SGD = args.SGD
    which_h = int(args.H)
    which_w = int(args.w)
    d = args.d
    N_max_ridge = args.N_ridge
    N_max_sgd = args.N_SGD
    depth = args.depth
    intern_dim = args.intern_dim

    # saving paths
    suffix_ridge = suffix_filename(ridge_bool=True, w=which_w, h=which_h, d=d)
    suffix_sgd = suffix_filename(sgd_bool=True, w=which_w, h=which_h, d=d, depth=depth, intern_dim=intern_dim)
    SAVE_DIR_SGD = 'data/SGD/'
    SAVE_DIR_RIDGE = 'data/Ridge/'
    SAVE_RIDGE_ITERATE = SAVE_DIR_RIDGE + 'iterates'+suffix_ridge+'.npy'
    SAVE_SGD_ITERATE = SAVE_DIR_SGD + 'iterates'+suffix_sgd+'.npy'
    SAVE_RIDGE_LAMBDA = SAVE_DIR_RIDGE + 'lambda'+suffix_ridge+'.npy'
    SAVE_SGD_GAMMA = SAVE_DIR_SGD + 'gamma'+suffix_sgd+'.npy'

    ### Begin experiment
    # Initialization
    w_ridge = np.zeros((nb_avg, len(n_ridge), d))
    w_sgd = np.zeros((nb_avg, len(n_sgd), d))

    # load best coeffs
    if USE_SAVED_PARAMS: 
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
        ### Data generation: data (N_max_ridge,d) ; observations (N_max_ridge,)
        data, observations = generate_data(p=d, n=N_max_ridge, sigma2=sigma2, which_w=which_w, which_h=which_h)

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
                    ws = train_v2(model,
                          input_Tensor[:n],
                          output_Tensor[:n],
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