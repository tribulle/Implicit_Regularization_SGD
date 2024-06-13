import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from utils import *

np.random.seed(5)
torch.manual_seed(5)

### Parameters
COMPUTE_DATA_PLOT = True

DATA_FOLDER = 'data/'
DATA_FILENAME = 'data'
EXT='.csv'
TRAIN_TEST_SPLIT = 0.8

which_h = 1

data, observations, means, stds = load_data_CSV(file_name=DATA_FOLDER+DATA_FILENAME+EXT,
                                                n=None, # None to load all the dataset
                                                normalize=True,
                                                which_h=which_h)
data = data[int(TRAIN_TEST_SPLIT*len(data)):,:]
observations = observations[int(TRAIN_TEST_SPLIT*len(observations)):]

d = data.shape[1]
sigma2 = 1
nb_avg = 20

N_max_ridge = 1500
N_max_sgd = 500
n_ridge = np.floor(np.linspace(d,N_max_ridge,100)).astype(dtype=np.uint16)
n_sgd = np.floor(np.linspace(d,N_max_sgd,20)).astype(dtype=np.uint16)

OUTLIER_DETECTION = True
threshold_obj = sigma2*4

# saving paths
SAVE_DIR_SGD = 'data/SGD/'
SAVE_DIR_RIDGE = 'data/Ridge/'
SAVE_DATA_PLOT_N = f'data/data_plot_n_{DATA_FILENAME}.npy'
SAVE_DATA_PLOT_SGD = f'data/data_plot_sgd_{DATA_FILENAME}.npy'
SAVE_DATA_PLOT_RIDGE = f'data/data_plot_ridge_{DATA_FILENAME}.npy'
SAVE_DIR_FIG = 'figures/'

if COMPUTE_DATA_PLOT:
    ### Study
    y_plot = np.zeros(len(n_sgd))

    # Get weights
    SAVE_RIDGE_ITERATE = SAVE_DIR_RIDGE + f'iterates_{DATA_FILENAME}.npy'
    SAVE_SGD_ITERATE = SAVE_DIR_SGD + f'iterates_{DATA_FILENAME}.npy'
    ### Load weights
    w_ridge = np.load(SAVE_RIDGE_ITERATE) # (nb_avg, len(n_ridge), d)
    w_sgd = np.load(SAVE_SGD_ITERATE) # (nb_avg, len(n_sgd), d)

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
        sgd_risks = np.mean(sgd_errors, axis=0, where=sgd_errors<threshold_obj)
        ridge_risks = np.mean(ridge_errors, axis=0, where=ridge_errors<threshold_obj)
        
        print(f'{n_out_ridge+n_out_sgd} outliers' +
              f'(Ridge: {n_out_ridge}, SGD: {n_out_sgd})'
              )
    else:
        sgd_risks = np.mean(sgd_errors, axis=0)
        ridge_risks = np.mean(ridge_errors, axis=0)

    # for each n_sgd, search minimal n_ridge with same risk
    for k,n in enumerate(n_sgd):
        if np.isnan(sgd_risks[k]):
            valid = [n_ridge[0]]
        else:
            valid = n_ridge[np.where(ridge_risks<sgd_risks[k])] # ridge better than sgd with n_sgd[k] samples
        if len(valid) != 0:
            y_plot[k] = valid[0] # smallest n_ridge better than sgd
        else:
            y_plot[k] = n_ridge[-1] #n_ridge[-1] # default: all ridge worse than sgd

    np.save(SAVE_DATA_PLOT_N, y_plot)
    np.save(SAVE_DATA_PLOT_SGD, sgd_risks)
    np.save(SAVE_DATA_PLOT_RIDGE, ridge_risks)
    print('Computation done')
else:
    y_plot = np.load(SAVE_DATA_PLOT_N)
    sgd_risks = np.load(SAVE_DATA_PLOT_SGD)
    ridge_risks = np.load(SAVE_DATA_PLOT_RIDGE)
    print('Data loaded')


fig,axs = plt.subplots(1,2, figsize=(16,8))
axs[0].plot(n_sgd, y_plot)
axs[1].plot(n_sgd, sgd_risks, label='SGD')
axs[1].plot(n_ridge, ridge_risks, linestyle='--', label='Ridge')
axs[0].grid(color='black', which="both", linestyle='--', linewidth=0.2)
axs[0].set_xlabel(r'$N_{SGD}$')
axs[0].set_ylabel(r'$N_{Ridge}$')
axs[0].set_ylim(0,N_max_ridge)
axs[1].grid(color='black', which="both", linestyle='--', linewidth=0.2)
axs[1].legend()
axs[1].set_yscale('log')
axs[1].set_xlabel('N')
axs[1].set_ylabel('Population Risk')
plt.suptitle(f'SGD vs Ridge on "{DATA_FILENAME}" dataset')
plt.savefig(SAVE_DIR_FIG+f'benefits_{DATA_FILENAME}')
plt.show()
