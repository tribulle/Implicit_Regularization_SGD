import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from utils import *

### Parameters
COMPUTE_DATA_PLOT = True

d = 100
sigma2 = 1
nb_avg = 20

N_samples = 5000

N_max_ridge = 6000
N_max_sgd = 1000
n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16)
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

all_which_h = [1] # 1 or 2 -> i**(-...)
all_which_w = [0,1,10] # 0, 1 or 10 -> i**(-...)

# saving paths
SAVE_DIR_SGD = 'data/SGD/'
SAVE_DIR_RIDGE = 'data/Ridge/'
SAVE_DATA_PLOT = 'data/data_plot.npy'
SAVE_DIR_FIG = 'figures/'

# Plots
W_LABELS = [r'$w^*[i]=1$', r'$w^*[i]=i^{-1}$', r'$w^*[i]=i^{-10}$']
H_LABELS = [r'$\lambda_i=i^{-1}$', r'$\lambda_i=i^{-2}$']

if COMPUTE_DATA_PLOT:
    ### Study
    y_plot = np.zeros((len(all_which_h), len(all_which_w), len(n_sgd))) # (h,w, n_sgd)
    # y_plot[h,w, n] contains the value n_ridge s.t loss(ridge(n_ridge)) ~= loss(sgd(n_sgd[n]))
    for i,which_h in enumerate(all_which_h):
        for j,which_w in enumerate(all_which_w):
            # Get weights
            filename = f'iterates_H{which_h}_w{which_w}.npy'
            SAVE_RIDGE_ITERATE = SAVE_DIR_RIDGE + filename
            SAVE_SGD_ITERATE = SAVE_DIR_SGD + filename

            ### Load weights
            w_ridge = np.load(SAVE_RIDGE_ITERATE) # (nb_avg, len(n_ridge), d)
            w_sgd = np.load(SAVE_SGD_ITERATE) # (nb_avg, len(n_sgd), d)

            ### Generate new data (from same distribution)
            w_true = np.float_power(np.arange(1,d+1), -which_w) # true parameter
            H = np.diag(np.float_power(np.arange(1,d+1), -which_h))
            data = np.random.multivariate_normal(
                np.zeros(d),
                H,
                size=N_samples) # shape (n,d)    
            observations = [np.random.normal(
                np.dot(w_true, x),
                sigma2)
                for x in data]
            observations = np.array(observations) # shape (n,)

            ### Compute variables of interest
            ridge_errors = np.zeros((nb_avg, len(n_ridge)))
            for k1 in range(nb_avg):
                for k2,n in enumerate(n_ridge):
                    ridge_errors[k1,k2] = objective(data, observations, w_ridge[k1,k2,:])
            ridge_risk = np.mean(ridge_errors, axis=0)

            sgd_errors = np.zeros((nb_avg, len(n_sgd)))
            for k1 in range(nb_avg):
                for k2,n in enumerate(n_sgd):
                    sgd_errors[k1,k2] = objective(data, observations, w_sgd[k1,k2,:])
            sgd_risk = np.mean(sgd_errors, axis=0)

            # for each n_sgd, search minimal n_ridge with same risk
            for k,n in enumerate(n_sgd): 
                valid = n_ridge[np.where(ridge_risk<sgd_risk[k])] # ridge better than sgd
                if len(valid) != 0:
                    y_plot[i,j,k] = valid[0] # smaller n_ridge better than sgd
                else:
                    y_plot[i,j,k] = None #n_ridge[-1] # default: all ridge worse than sgd

    np.save(SAVE_DATA_PLOT, y_plot)
    print('Computation done')
else:
    y_plot = np.load(SAVE_DATA_PLOT)
    print('Data loaded')

for i, which_h in enumerate(all_which_h):
    for j,which_w in enumerate(all_which_w):
        plt.plot(n_sgd, y_plot[i,j,:], label=W_LABELS[j])
    plt.grid(color='black', which="both", linestyle='--', linewidth=0.2)
    plt.legend()
    plt.xlabel(r'$N_{SGD}$')
    plt.ylabel(r'$N_{Ridge}$')
    plt.title('SGD vs Ridge ; H:'+H_LABELS[i])
    plt.savefig(SAVE_DIR_FIG+f'benefits_H{which_h}')
    plt.show()
    