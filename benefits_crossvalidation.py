import numpy as np
from tqdm import tqdm
import torch
import warnings
import argparse

from utils import *

# Fix the random seed
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Parameters
d = 50
sigma2 = 1

CROSS_VAL_K = 10

N_max_ridge = int(6000/(1+1/CROSS_VAL_K))
N_max_sgd = int(2000*(1+1/CROSS_VAL_K))

n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16)
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

n_fine_tune_params = 10 # nb of hyperparameters tested

lambdas_ = np.logspace(-5,-1,n_fine_tune_params, base=10.0) # range of parameters
learning_rates = np.logspace(-6,-2,n_fine_tune_params)

intern_dim = 10
depth = -1 # Single Layer
optimizer = 'SGD'
learning_rates = np.logspace(-6,-2,n_fine_tune_params)

which_h = 2 # 1 or 2 -> i**(-...)
which_w = 10 # 0, 1 or 10 -> i**(-...)

FINE_TUNE_RIDGE = True
FINE_TUNE_SGD = False


if __name__=='__main__':
    ### Argument parser
    parser = argparse.ArgumentParser(prog='Data generation for implicit regularization of SGD study')
    parser.add_argument('--SGD', action=argparse.BooleanOptionalAction, default=FINE_TUNE_SGD,
                        help='Fine tune SGD')
    parser.add_argument('--Ridge', action=argparse.BooleanOptionalAction, default=FINE_TUNE_RIDGE,
                        help='Fine tune Ridge')
    parser.add_argument('-H', default=which_h, choices=[1,2], type=int, help='matrix H1 or H2 to use')
    parser.add_argument('-w', default=which_w, choices=[0,1,10], type=int, help='true vector w0, w1 or w10')
    parser.add_argument('-d', default=d, type=int, help='dimension of the data')
    parser.add_argument('--N_ridge', default=N_max_ridge, type=int, help='Max number of data for ridge')
    parser.add_argument('--N_SGD', default=N_max_sgd, type=int, help='Max number of data for SGD')
    parser.add_argument('-k', default=CROSS_VAL_K, type=int, help='k for k fold cross-validation')
    parser.add_argument('--depth', default=depth, type=int, help='depth of MLP (i.e nb of hidden layers), -1 for single layer')
    parser.add_argument('--intern_dim', default=intern_dim, type=int, help='intern dimension of hidden layers')

    args = parser.parse_args()

    FINE_TUNE_RIDGE = args.Ridge
    FINE_TUNE_SGD = args.SGD
    which_h = args.H
    which_w = args.w
    d = args.d
    N_max_ridge = args.N_ridge
    N_max_sgd = args.N_SGD
    depth = args.depth
    intern_dim = args.intern_dim
    CROSS_VAL_K = args.k

    # saving paths
    suffix_ridge = suffix_filename(ridge_bool=True, w=which_w, h=which_h, d=d)
    suffix_sgd = suffix_filename(sgd_bool=True, w=which_w, h=which_h, d=d, depth=depth, intern_dim=intern_dim)
    SAVE_DIR_SGD = 'data/SGD/'
    SAVE_DIR_RIDGE = 'data/Ridge/'
    SAVE_RIDGE_LAMBDA = SAVE_DIR_RIDGE + 'lambda'+suffix_ridge+'.npy'
    SAVE_SGD_GAMMA = SAVE_DIR_SGD + 'gamma'+suffix_sgd+'.npy'

    ### Begin experiment
    # Initialization
    if FINE_TUNE_RIDGE:
        objectives_ridge = np.zeros(len(lambdas_))
    if FINE_TUNE_SGD:
        objectives_sgd = np.zeros(len(lambdas_))

    ### Data generation: data (N_max_ridge,d) ; observations (N_max_ridge,)
    data, observations = generate_data(p=d, n=N_max_ridge, sigma2=sigma2, which_w=which_w, which_h=which_h)

    train_masks_ridge, test_masks_ridge = cross_validation(N_max_ridge,
                                                           k=CROSS_VAL_K, 
                                                           homogeneous=False, 
                                                           sizes=np.floor(np.linspace(d,N_max_ridge,CROSS_VAL_K)).astype(dtype=np.uint16))
    train_masks_sgd, test_masks_sgd = cross_validation(N_max_sgd,k=CROSS_VAL_K)

    # for each hyperparameter, average its performances
    for j in tqdm(range(n_fine_tune_params)):

        # Cross validation
        for i in range(CROSS_VAL_K):
            ### Solving the problem
            if FINE_TUNE_RIDGE:
                train_mask = train_masks_ridge[i]
                test_mask = test_masks_ridge[i]
                # train on first part of data
                w = ridge(data[train_mask], observations[train_mask], lambda_=lambdas_[j])
                # evaluate on what remains
                objectives_ridge[j] += objective(data[test_mask], observations[test_mask], w)

            if FINE_TUNE_SGD:
                train_mask = train_masks_sgd[i]
                test_mask = test_masks_sgd[i]
                model = MultiLayerPerceptron(input_dim=d,
                                        intern_dim=intern_dim,
                                        output_dim=1,
                                        depth=depth,
                                        init='zero',
                                        isBiased = False,
                                    ).to(device)

                input_Tensor = torch.from_numpy(data[train_mask]).to(device, dtype=torch.float32)
                output_Tensor = torch.from_numpy(observations[train_mask]).to(device, dtype=torch.float32)

                ws = train(model,
                          input_Tensor,
                          output_Tensor,
                          lossFct = nn.MSELoss(),
                          optimizer=optimizer,
                          epochs=len(train_mask),
                          batch_size=None,
                          return_vals=False,
                          return_ws=True,
                          init_norm = None,
                          lr = learning_rates[j])
                w = np.mean(ws[len(train_mask)//2:,:], axis=0) # tail averaging
                objectives_sgd[j] += objective(data[test_mask], observations[test_mask], w)

    # save best parameters
    if FINE_TUNE_RIDGE:
        idx_best = np.argmin(objectives_ridge)
        print(f'Best lambda_: {lambdas_[idx_best]}')
        print(f'Mean objective: {objectives_ridge[idx_best]/len(lambdas_)}')
        np.save(SAVE_RIDGE_LAMBDA, lambdas_[idx_best])
    if FINE_TUNE_SGD:
        idx_best = np.argmin(objectives_sgd)
        print(f'Best gamma: {learning_rates[idx_best]}')
        print(f'Mean objective: {objectives_sgd[idx_best]/len(learning_rates)}')
        np.save(SAVE_SGD_GAMMA, learning_rates[idx_best])