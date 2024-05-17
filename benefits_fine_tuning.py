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
nb_avg = 10

N_max_ridge = 6000//4
N_max_sgd = 2000//4
train_test_split = 0.8 # 80% to train, 20% to evaluate
n_train_ridge = int(train_test_split*N_max_ridge)
n_train_sgd = int(train_test_split*N_max_sgd)

n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16)
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

n_fine_tune_params = 10 # nb of hyperparameters tested

lambdas_ = np.logspace(-5,-1,n_fine_tune_params, base=10.0) # range of parameters
learning_rates = np.logspace(-6,-2,n_fine_tune_params)

intern_dim = 10
depth = -1 # Single Layer
optimizer = 'SGD'
learning_rates = np.logspace(-6,-2,n_fine_tune_params)

which_h = 1 # 1 or 2 -> i**(-...)
which_w = 0 # 0, 1 or 10 -> i**(-...)

FINE_TUNE_RIDGE = False
FINE_TUNE_SGD = True

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
if FINE_TUNE_RIDGE:
    objectives_ridge = np.zeros(len(lambdas_))
if FINE_TUNE_SGD:
    objectives_sgd = np.zeros(len(lambdas_))

w_true = np.float_power(np.arange(1,d+1), -which_w) # true parameter
H = np.diag(np.float_power(np.arange(1,d+1), -which_h))

# for each hyperparameter, average its performances
for j in tqdm(range(n_fine_tune_params)):
    # Averaging results
    for i in range(nb_avg):
        ### Data generation
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
        if FINE_TUNE_RIDGE:
            # train on first part of data
            w = ridge(data[:n_train_ridge,:], observations[:n_train_ridge], lambda_=lambdas_[j])
            # evaluate on what remains
            objectives_ridge[j] += objective(data[n_train_ridge:,:], observations[n_train_ridge:], w)
        if FINE_TUNE_SGD:
            model = MultiLayerPerceptron(input_dim=d,
                                    intern_dim=intern_dim,
                                    output_dim=1,
                                    depth=depth,
                                    init='zero',
                                    isBiased = False,
                                ).to(device)
            
            input_Tensor = torch.from_numpy(data[:n_train_sgd,:]).to(device, dtype=torch.float32)
            output_Tensor = torch.from_numpy(observations[:n_train_sgd]).to(device, dtype=torch.float32)

            ws = train(model,
                      input_Tensor,
                      output_Tensor,
                      lossFct = nn.MSELoss(),
                      optimizer=optimizer,
                      epochs=n_train_sgd,
                      batch_size=None,
                      return_vals=False,
                      return_ws=True,
                      init_norm = None,
                      lr = learning_rates[j])
            w = np.mean(ws[n_train_sgd//2:,:], axis=0)
            objectives_sgd[j] += objective(data[n_train_sgd:,:], observations[n_train_sgd:], w)

# save best parameters
if FINE_TUNE_RIDGE:
    idx_best = np.argmin(objectives_ridge)
    print(f'Best lambda_: {lambdas_[idx_best]}')
    print(f'Mean objective: {objectives_ridge[idx_best]/len(lambdas_)}')
    np.save(SAVE_RIDGE_LAMBDA, np.vstack((lambdas_[idx_best]*np.ones(len(n_ridge)), n_ridge)))
if FINE_TUNE_SGD:
    idx_best = np.argmin(objectives_sgd)
    print(f'Best gamma: {learning_rates[idx_best]}')
    print(f'Mean objective: {objectives_sgd[idx_best]/len(learning_rates)}')
    np.save(SAVE_SGD_GAMMA, np.vstack((learning_rates[idx_best]*np.ones(len(n_sgd)), n_sgd)))



