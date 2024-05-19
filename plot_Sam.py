import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from utils import *

np.random.seed(9)
torch.manual_seed(9)

### Parameters
COMPUTE_DATA_PLOT = True

d = 200//4
sigma2 = 1
nb_avg = 20

N_samples = 10000

N_max_ridge = 6000//4
N_max_sgd = 2000//4
n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16)
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

all_which_h = [1,2] # 1 or 2 -> i**(-...)
all_which_w = [0,1,10] # 0, 1 or 10 -> i**(-...)

intern_dims = [10, 30, 50, 80, 100]
depth = 1

OUTLIER_DETECTION = True
threshold_obj = sigma2*4

# saving paths
SAVE_DIR_SGD = 'data/SGD/'
SAVE_DIR_RIDGE = 'data/Ridge/'
SAVE_DATA_PLOT_N = f'data/data_plot_n_d{d}_depth{depth}_indimVARIATION.npy'
SAVE_DATA_PLOT_SGD = f'data/data_plot_sgd_d{d}_depth{depth}_indimVARIATION.npy'
SAVE_DATA_PLOT_RIDGE = f'data/data_plot_ridge_d{d}_depth{depth}_indimVARIATION.npy'
SAVE_DIR_FIG = 'figures/'

# Plots
W_LABELS = [r'$w^*[i]=1$', r'$w^*[i]=i^{-1}$', r'$w^*[i]=i^{-10}$']
H_LABELS = [r'$\lambda_i=i^{-1}$', r'$\lambda_i=i^{-2}$']
COLORS = ['tab:blue', 'tab:orange', 'tab:green']

if COMPUTE_DATA_PLOT:
    ### Study
    y_plot = np.zeros((len(intern_dims), len(all_which_h), len(all_which_w), len(n_sgd))) # (h,w, n_sgd)
    sgd_risks = np.zeros((len(intern_dims), len(all_which_h), len(all_which_w), len(n_sgd)))
    ridge_risks = np.zeros((len(all_which_h), len(all_which_w), len(n_ridge)))
    # y_plot[h,w, n] contains the value n_ridge s.t loss(ridge(n_ridge)) ~= loss(sgd(n_sgd[n]))
    for i,which_h in enumerate(all_which_h):
        for j,which_w in enumerate(all_which_w):
            ### RIDGE PART
            suffix_ridge = suffix_filename(ridge_bool=True, w=which_w, h=which_h, d=d)
            SAVE_RIDGE_ITERATE = SAVE_DIR_RIDGE + 'iterates'+suffix_ridge+'.npy'
            w_ridge = np.load(SAVE_RIDGE_ITERATE) # (nb_avg, len(n_ridge), d)
            ### Generate new data (from same distribution)
            w_true = np.float_power(np.arange(1,d+1), -which_w) # true parameter
            H = np.diag(np.float_power(np.arange(1,d+1), -which_h))
            data = np.random.multivariate_normal(
                np.zeros(d),
                H,
                size=N_samples) # shape (n,d)    
            observations = [np.random.normal(
                np.dot(w_true, x),
                np.sqrt(sigma2))
                for x in data]
            observations = np.array(observations) # shape (n,)

            ### Compute variables of interest
            ridge_errors = np.zeros((nb_avg, len(n_ridge)))
            for k1 in range(nb_avg):
                for k2,n in enumerate(n_ridge):
                    ridge_errors[k1,k2] = objective(data, observations, w_ridge[k1,k2,:])

            # Outlier detection
            if OUTLIER_DETECTION:
                n_out_ridge = (ridge_errors>threshold_obj).sum()
                ridge_risks[i, j,:] = np.mean(ridge_errors, axis=0, where=ridge_errors<threshold_obj)
                print(f'Ridge: {n_out_ridge} outliers for H{which_h}_w{which_w}')
            ### SGD PART
            for k, intern_dim in enumerate(intern_dims):
                suffix_sgd = suffix_filename(sgd_bool=True, w=which_w, h=which_h, d=d, depth=depth, intern_dim=intern_dim)
                SAVE_SGD_ITERATE = SAVE_DIR_SGD + 'iterates'+suffix_sgd+'.npy'
                w_sgd = np.load(SAVE_SGD_ITERATE) # (nb_avg, len(n_sgd), d)            
                sgd_errors = np.zeros((nb_avg, len(n_sgd)))
                for k1 in range(nb_avg):
                    for k2,n in enumerate(n_sgd):
                        sgd_errors[k1,k2] = objective(data, observations, w_sgd[k1,k2,:])
            
                # Outlier detection
                if OUTLIER_DETECTION:
                    n_out_sgd = (sgd_errors > threshold_obj).sum()
                    sgd_risks[k, i, j,:] = np.mean(sgd_errors, axis=0, where=sgd_errors<threshold_obj)

                    print(f'SGD: {n_out_sgd} outliers for H{which_h}_w{which_w} indim{intern_dim} ')
                else:
                    sgd_risks[k,i, j,:] = np.mean(sgd_errors, axis=0)
                    ridge_risks[i, j,:] = np.mean(ridge_errors, axis=0)

                # for each n_sgd, search minimal n_ridge with same risk
                for k1,n in enumerate(n_sgd):
                    if np.isnan(sgd_risks[k, i, j, k1]):
                        valid = [n_ridge[0]]
                    else:
                        valid = n_ridge[np.where(ridge_risks[i, j,:]<sgd_risks[k, i, j,k1])] # ridge better than sgd with n_sgd[k] samples
                    if len(valid) != 0:
                        y_plot[k, i, j,k1] = valid[0] # smallest n_ridge better than sgd
                    else:
                        y_plot[k, i, j,k1] = n_ridge[-1] #n_ridge[-1] # default: all ridge worse than sgd

    np.save(SAVE_DATA_PLOT_N, y_plot)
    np.save(SAVE_DATA_PLOT_SGD, sgd_risks)
    np.save(SAVE_DATA_PLOT_RIDGE, ridge_risks)
    print('Computation done')
else:
    y_plot = np.load(SAVE_DATA_PLOT_N)
    sgd_risks = np.load(SAVE_DATA_PLOT_SGD)
    ridge_risks = np.load(SAVE_DATA_PLOT_RIDGE)
    print('Data loaded')


for i, which_h in enumerate(all_which_h):
    fig,axs = plt.subplots(1,len(all_which_w), figsize=(16,8))
    for j,which_w in enumerate(all_which_w):
        for k, intern_dim in enumerate(intern_dims):
            axs[j].plot(n_sgd, y_plot[k,i,j,:], label=f'intern_dim: {intern_dim}')
        axs[j].grid(color='black', which="both", linestyle='--', linewidth=0.2)
        axs[j].legend()
        axs[j].set_xlabel(r'$N_{SGD}$')
        axs[j].set_ylabel(r'$N_{Ridge}$')
        axs[j].set_title(W_LABELS[j])

    plt.suptitle('SGD vs Ridge - effect of intern dimension; H:'+H_LABELS[i])
    plt.savefig(SAVE_DIR_FIG+f'benefits_H{which_h}_d{d}_depth{depth}_indimVARIATION')
    plt.show()
    