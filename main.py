import numpy as np
import argparse
import os


d = 200//4
sigma2 = 1
nb_avg = 10

N_max_ridge = 6000//4
N_max_sgd = 2000//4
train_test_split = 0.8 # 80% to train, 20% to evaluate
n_train_ridge = int(train_test_split*N_max_ridge)
n_train_sgd = int(train_test_split*N_max_sgd)

intern_dim = 10
depth = -1 # Single Layer

which_h = 1 # 1 or 2 -> i**(-...)
which_w = 0 # 0, 1 or 10 -> i**(-...)

GENERATE_RIDGE = True # generate ridge weights
GENERATE_SGD = False # generate SGD weights

FINE_TUNE_RIDGE = False
FINE_TUNE_SGD = False

if __name__=='__main__':
    # example of command
    for which_w in [0,1,10]:
        command_fine_tune = 'python benefits_fine_tuning.py '
        if FINE_TUNE_RIDGE:
            command_fine_tune += '--Ridge '
        else:
            command_fine_tune += '--no-Ridge '
        if FINE_TUNE_SGD:
            command_fine_tune += '--SGD '
        else:
            command_fine_tune += '--no-SGD '
        command_fine_tune += (f'-w {which_w:d} ' +
                              f'-H {which_h:d} ' +
                              f'-d {d:d} ' +
                              f'--N_ridge {N_max_ridge:d} ' +
                              f'--N_SGD {N_max_sgd:d} ' +
                              f'--depth {depth:d} ' +
                              f'--intern_dim {intern_dim:d}'
                              )

        os.system(command_fine_tune)

        command_data = 'python benefits_data_generation.py '
        # example of command
        if GENERATE_RIDGE:
            command_data += '--Ridge '
        else:
            command_data += '--no-Ridge '
        if GENERATE_SGD:
            command_data += '--SGD '
        else:
            command_data += '--no-SGD '
        command_data += (f'-w {which_w:d} ' +
                              f'-H {which_h:d} ' +
                              f'-d {d:d} ' +
                              f'--N_ridge {N_max_ridge:d} ' +
                              f'--N_SGD {N_max_sgd:d} ' +
                              f'--depth {depth:d} ' +
                              f'--intern_dim {intern_dim:d}'
                              )

        os.system(command_data)