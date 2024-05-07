import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from utils import *

np.random.seed(42)
torch.manual_seed(42)

### Parameters
# code parameters
d = 200
sigma2 = 1
nb_avg = 20

N_max_ridge = 6000
N_max_sgd = 2000
n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16)

lambda_ = 1e-2

intern_dim = 10
depth = -1 # Single Layer
optimizer = 'SGD'
learning_rate = 0.001

which_h = 1 # 1 or 2 -> i**(-...)
which_w = 0 # 0, 1 or 10 -> i**(-...)

GENERATE_RIDGE = True
GENERATE_SGD = True

# saving paths
SAVE_DIR_SGD = 'data/SGD/'
SAVE_DIR_RIDGE = 'data/Ridge/'
SAVE_RIDGE_ITERATE = SAVE_DIR_RIDGE + f'iterates_ridge_H{which_h}_w{which_w}.npy'
SAVE_SGD_ITERATE = SAVE_DIR_SGD + f'iterates_sgd_H{which_h}_w{which_w}.npy'
SAVE_OBSERVATIONS = 'observations.npy'
SAVE_DATA = 'data.npy'


### Begin experiment
# Initialization
w_ridge = np.zeros((nb_avg, len(n_ridge), d))
w_sgd = np.zeros((nb_avg, N_max_sgd, d))

# Averaging results
for i in tqdm(range(nb_avg)):
    ### Data generation
    w_true = 1/(np.arange(1,d+1)**which_w) # true parameter
    H = np.diag(1/(np.arange(1,d+1)**which_h))
    data = np.random.multivariate_normal(
        np.zeros(d),
        H,
        size=N_max_sgd) # shape (n,d)    

    observations = [np.random.normal(
        np.dot(w_true, x),
        sigma2)
        for x in data]
    observations = np.array(observations) # shape (n,)

    ### Solving the problem
    # Ridge
    if GENERATE_RIDGE:
        for j,n in enumerate(n_ridge):
            w_ridge[i,j,:] = ridge(data[:n,:], observations[:n], lambda_=1e-2) # CHANGE THIS TO FINE-TUNE LAMBDA

    # SGD
    if GENERATE_SGD:
        input_Tensor = torch.from_numpy(data).to(torch.float32)
        output_Tensor = torch.from_numpy(observations).to(torch.float32)

        model = MultiLayerPerceptron(input_dim=d,
                                     intern_dim=intern_dim,
                                     output_dim=1,
                                     depth=depth,
                                     init='zero',
                                     isBiased = False,
                                    )
    
        w_sgd[i,:,:] = train(model,
              input_Tensor,
              output_Tensor,
              lossFct = nn.MSELoss(),
              optimizer=optimizer,
              epochs=N_max_sgd,
              batch_size=None,
              return_vals=False,
              return_ws=True,
              init_norm = None,
              lr = learning_rate) # TO CHANGE: FINE-TUNE LEARNING RATE

# Save results
if GENERATE_RIDGE:
    np.save(SAVE_RIDGE_ITERATE, w_ridge)
    np.save(SAVE_DIR_RIDGE+SAVE_OBSERVATIONS, observations)
    np.save(SAVE_DATA, data)
if GENERATE_SGD:
    np.save(SAVE_SGD_ITERATE, w_sgd)
    np.save(SAVE_DIR_RIDGE+SAVE_OBSERVATIONS, observations)
    np.save(SAVE_DATA, data)