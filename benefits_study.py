import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from utils import *

np.random.seed(5)
torch.manual_seed(5)

### Parameters
COMPUTE_DATA_PLOT = True
PLOT_ONLY_N = False # plot only the n VS n plot, not the loss

d = 50
sigma2 = 1
nb_avg = 20

N_samples = 10000 # nb of points for evaluation of the models

N_max_ridge = 1500
N_max_sgd = 500
n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16)
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

all_which_h = [1,2] # 1 or 2 -> i**(-...)
all_which_w = [0,1,10] # 0, 1 or 10 -> i**(-...)

intern_dim = 10
depth = -1

OUTLIER_DETECTION = True
threshold_obj = sigma2*4

# saving paths
SAVE_DIR_SGD = 'data/SGD/'
SAVE_DIR_RIDGE = 'data/Ridge/'
SAVE_DATA_PLOT_N = f'data/data_plot_n_d{d}_depth{depth}_indim{intern_dim}.npy'
SAVE_DATA_PLOT_SGD = f'data/data_plot_sgd_d{d}_depth{depth}_indim{intern_dim}.npy'
SAVE_DATA_PLOT_RIDGE = f'data/data_plot_ridge_d{d}_depth{depth}_indim{intern_dim}.npy'
SAVE_DIR_FIG = 'figures/'

# Plots
W_LABELS = [r'$w^*[i]=1$', r'$w^*[i]=i^{-1}$', r'$w^*[i]=i^{-10}$']
H_LABELS = [r'$\lambda_i=i^{-1}$', r'$\lambda_i=i^{-2}$']
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
additional_title = ''
if depth >= 0:
    additional_title += f'  (depth={depth} indim={intern_dim})'

if COMPUTE_DATA_PLOT:
    ### Study
    # Initialisation
    y_plot = np.zeros((len(all_which_h), len(all_which_w), len(n_sgd))) # (h,w, n_sgd)
    sgd_risks = np.zeros((len(all_which_h), len(all_which_w), len(n_sgd)))
    ridge_risks = np.zeros((len(all_which_h), len(all_which_w), len(n_ridge)))
    # y_plot[h,w, n] contains the value n_ridge s.t loss(ridge(n_ridge)) ~= loss(sgd(n_sgd[n]))

    for i,which_h in enumerate(all_which_h):
        for j,which_w in enumerate(all_which_w):
            suffix_ridge = suffix_filename(ridge_bool=True, w=which_w, h=which_h, d=d)
            suffix_sgd = suffix_filename(sgd_bool=True, w=which_w, h=which_h, d=d, depth=depth, intern_dim=intern_dim)
            # Get weights
            SAVE_RIDGE_ITERATE = SAVE_DIR_RIDGE + 'iterates'+suffix_ridge+'.npy'
            SAVE_SGD_ITERATE = SAVE_DIR_SGD + 'iterates'+suffix_sgd+'.npy'

            ### Load weights
            w_ridge = np.load(SAVE_RIDGE_ITERATE) # (nb_avg, len(n_ridge), d)
            w_sgd = np.load(SAVE_SGD_ITERATE) # (nb_avg, len(n_sgd), d)

            ### Generate new data (from same distribution)
            data, observations = generate_data(p=d, n=N_samples, sigma2=sigma2, which_w=which_w, which_h=which_h)

            ### Compute variables of interest
            ridge_errors = np.zeros((nb_avg, len(n_ridge)))
            sgd_errors = np.zeros((nb_avg, len(n_sgd)))
            for k1 in range(nb_avg):
                for k2,n in enumerate(n_ridge):
                    ridge_errors[k1,k2] = objective(data, observations, w_ridge[k1,k2,:])
                for k2,n in enumerate(n_sgd):
                    sgd_errors[k1,k2] = objective(data, observations, w_sgd[k1,k2,:])
            
            # Outlier detection
            if OUTLIER_DETECTION:
                n_out_ridge = (ridge_errors>threshold_obj).sum()
                n_out_sgd = (sgd_errors > threshold_obj).sum()

                sgd_risks[i, j,:] = np.mean(sgd_errors, axis=0, where=sgd_errors<threshold_obj)
                ridge_risks[i, j,:] = np.mean(ridge_errors, axis=0, where=ridge_errors<threshold_obj)
                
                print(f'{n_out_ridge+n_out_sgd} outliers for H{which_h}_w{which_w} ' +
                      f'(Ridge: {n_out_ridge}, SGD: {n_out_sgd})'
                      )
            else:
                sgd_risks[i, j,:] = np.mean(sgd_errors, axis=0)
                ridge_risks[i, j,:] = np.mean(ridge_errors, axis=0)

            #sgd_risks[sgd_risks > threshold_obj] = None

            # for each n_sgd, search minimal n_ridge with same risk
            for k,n in enumerate(n_sgd):
                if np.isnan(sgd_risks[i, j,k]):
                    valid = [n_ridge[0]]
                else:
                    valid = n_ridge[np.where(ridge_risks[i, j,:]<sgd_risks[i, j,k])] # ridge better than sgd with n_sgd[k] samples
                if len(valid) != 0:
                    y_plot[i,j,k] = valid[0] # smallest n_ridge better than sgd
                else:
                    y_plot[i,j,k] = n_ridge[-1] #n_ridge[-1] # default: all ridge worse than sgd

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
    if PLOT_ONLY_N:
        for j,which_w in enumerate(all_which_w):
            plt.plot(n_sgd, y_plot[i,j,:], color=COLORS[j], label=W_LABELS[j])
        plt.grid(color='black', which="both", linestyle='--', linewidth=0.2)
        plt.legend()
        plt.xlabel(r'$N_{SGD}$')
        plt.ylabel(r'$N_{Ridge}$')
        plt.ylim(0,N_max_ridge)
        plt.title('SGD vs Ridge ; H:'+H_LABELS[i]+additional_title)
        if depth == -1:
            plt.savefig(SAVE_DIR_FIG+f'benefits_partial_H{which_h}_d{d}')
        else:
            plt.savefig(SAVE_DIR_FIG+f'benefits_partial_H{which_h}_d{d}_depth{depth}_indim{intern_dim}')
        plt.show()
    else:
        fig,axs = plt.subplots(1,2, figsize=(16,8))
        for j,which_w in enumerate(all_which_w):
            axs[0].plot(n_sgd, y_plot[i,j,:], color=COLORS[j], label=W_LABELS[j])
            axs[1].plot(n_sgd, sgd_risks[i, j,:], color=COLORS[j], label='SGD - '+W_LABELS[j])
            axs[1].plot(n_ridge, ridge_risks[i, j,:], linestyle='--', color=COLORS[j], label='Ridge - '+W_LABELS[j])
        axs[0].grid(color='black', which="both", linestyle='--', linewidth=0.2)
        axs[0].legend()
        axs[0].set_xlabel(r'$N_{SGD}$')
        axs[0].set_ylabel(r'$N_{Ridge}$')
        axs[0].set_ylim(0,N_max_ridge)

        axs[1].grid(color='black', which="both", linestyle='--', linewidth=0.2)
        axs[1].legend()
        axs[1].set_yscale('log')
        axs[1].set_xlabel('N')
        axs[1].set_ylabel('Population Risk')

        plt.suptitle('SGD vs Ridge ; H:'+H_LABELS[i]+additional_title)
        if depth == -1:
            plt.savefig(SAVE_DIR_FIG+f'benefits_H{which_h}_d{d}')
        else:
            plt.savefig(SAVE_DIR_FIG+f'benefits_H{which_h}_d{d}_depth{depth}_indim{intern_dim}')
        plt.show()
    