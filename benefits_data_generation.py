import numpy as np
from tqdm import tqdm
import torch
import warnings

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
FINE_TUNE = False # DEPRECIATED: Fine tune on 1st iteration
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

### Begin experiment
# Initialization
w_ridge = np.zeros((nb_avg, len(n_ridge), d))
w_sgd = np.zeros((nb_avg, len(n_sgd), d))

if not FINE_TUNE and USE_SAVED_PARAMS: # load best coeffs
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
        if FINE_TUNE and i==0:# fine tune only on first iteration
            print('\nFine-tuning Ridge...')
            for j in tqdm(range(len(n_ridge))):
                n = n_ridge[j]
                best_obj, best_w, best_idx = np.inf, None, 0
                for k in range(len(lambdas_)):
                    w = ridge(data[:n,:], observations[:n], lambda_=lambdas_[k])
                    obj = objective(data[:n,:], observations[:n], w)
                    if obj < best_obj:
                        best_obj, best_w, best_idx = obj, w, k
                w_ridge[i,j,:] = best_w
                lambda_[j] = lambdas_[best_idx]
            np.save(SAVE_RIDGE_LAMBDA, np.vstack((lambda_, n_ridge)))
            print('\nDone')
        else:
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
        if FINE_TUNE and i==0:
            print('\nFine tuning SGD...')
            for j in tqdm(range(len(n_sgd))):
                n = n_sgd[j]
                best_obj, best_w, best_idx = np.inf, None, 0
                for k, lr in enumerate(learning_rates):
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
                      lr = lr)
                    w = np.mean(ws[n//2:,:], axis=0)
                    obj = objective(data[:n,:], observations[:n], w)
                    if obj < best_obj:
                        best_obj, best_w, best_idx = obj, w, k
                w_sgd[i,j,:] = best_w
                learning_rate[j] = learning_rates[best_idx]
            np.save(SAVE_SGD_GAMMA, np.vstack((learning_rate, n_sgd)))
            print('\nDone')
        else:
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