import numpy as np
import argparse
import os

from utils import os_command


d = 50
sigma2 = 1
nb_avg = 10

N_max_ridge = 6000
N_max_sgd = 2000

intern_dim = 10
depth = -1

which_h = 2 # 1 or 2 -> i**(-...)
which_w = 0 # 0, 1 or 10 -> i**(-...)

CROSS_VAL_K = 10
HOMOGENEOUS = True

GENERATE_RIDGE = True # generate ridge weights
GENERATE_SGD = True # generate SGD weights

FINE_TUNE_RIDGE = True
FINE_TUNE_SGD = True
FINE_TUNE_PER_N = False
n_fine_tune_params = 10

if __name__=='__main__':
    # example of command to execute the desired files (generate data for ridge/sgd, fine tune for ridge/sgd on all w and all h)
    for which_h in [2]:
        for which_w in [0,1,10]:
            #for intern_dim in [10,30,50,80,100]:
            if FINE_TUNE_PER_N:
                file = 'benefits_crossvalidation_per_n.py'
            else:
                file = 'benefits_crossvalidation.py'
            command_fine_tune = os_command(file, 
                                           ridge_bool=FINE_TUNE_RIDGE, 
                                           sgd_bool=FINE_TUNE_SGD, 
                                           w=which_w, 
                                           h=which_h, 
                                           d=d, 
                                           N_ridge=N_max_ridge, 
                                           N_sgd=N_max_sgd,
                                           depth=depth,
                                           intern_dim=intern_dim,
                                           k=CROSS_VAL_K,
                                           homogeneous=HOMOGENEOUS,
                                           n_params=n_fine_tune_params)
            os.system(command_fine_tune)
            file = 'benefits_data_generation.py'
            command_data = os_command(file, 
                                      ridge_bool=GENERATE_RIDGE, 
                                      sgd_bool=GENERATE_SGD, 
                                      w=which_w, 
                                      h=which_h, 
                                      d=d, 
                                      N_ridge=N_max_ridge, 
                                      N_sgd=N_max_sgd,
                                      depth=depth,
                                      intern_dim=intern_dim)
            os.system(command_data)