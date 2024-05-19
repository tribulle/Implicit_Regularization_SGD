import numpy as np
import argparse
import os

from utils import os_command


d = 200//4
sigma2 = 1
nb_avg = 10

N_max_ridge = 6000//4
N_max_sgd = 2000//4
train_test_split = 0.8 # 80% to train, 20% to evaluate
n_train_ridge = int(train_test_split*N_max_ridge)
n_train_sgd = int(train_test_split*N_max_sgd)

intern_dim = 10
depth = 1

which_h = 2 # 1 or 2 -> i**(-...)
which_w = 0 # 0, 1 or 10 -> i**(-...)

CROSS_VAL_K = 10

GENERATE_RIDGE = True # generate ridge weights
GENERATE_SGD = True # generate SGD weights

FINE_TUNE_RIDGE = True
FINE_TUNE_SGD = True

if __name__=='__main__':
    # example of command to execute the desired files (generate data for ridge/sgd, fine tune for ridge/sgd on all w)
    for which_w in [0,1,10]:
        for intern_dim in [10,30,50,80,100]:
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
                                           intern_dim=intern_dim)

            os.system(command_fine_tune)

            file = 'benefits_data_generation.py'
            command_data = os_command(file, 
                                      ridge_bool=FINE_TUNE_RIDGE, 
                                      sgd_bool=FINE_TUNE_SGD, 
                                      w=which_w, 
                                      h=which_h, 
                                      d=d, 
                                      N_ridge=N_max_ridge, 
                                      N_sgd=N_max_sgd,
                                      depth=depth,
                                      intern_dim=intern_dim)

            os.system(command_data)