import numpy as np
import argparse
import os

from utils import os_command

########################### RUNNING GUIDELINES ##################################
# Running this file as is will produce the reproduction of the article's result #
# Uncomment the lines in the for loop as indicated to reproduce our results:    #
#     - for the effect of depth                                                 #
#     - for the effect of internal dimension                                    #
#################################################################################

d = 50
sigma2 = 1

N_max_ridge = 1500
N_max_sgd = 500

intern_dim = 10
depth = -1

which_h = 2 # 1 or 2 -> i**(-...)
which_w = 0 # 0, 1 or 10 -> i**(-...)

CROSS_VAL_K = 10
HOMOGENEOUS = False

GENERATE_RIDGE = True # generate ridge weights
GENERATE_SGD = True # generate SGD weights

FINE_TUNE_RIDGE = True # fine-tune ridge regularization
FINE_TUNE_SGD = True # fine-tune SGD learning rate
FINE_TUNE_PER_N = False
ridge_crossval = True
sgd_crossval = True
n_fine_tune_params = 10

depth_study = False
indim_study = False

if __name__=='__main__':
    for which_h in [1,2]:
        for which_w in [0,1,10]:
                
            ### HERE ### uncomment 3 lines to generate the data/plots for internal dimension study
            #depth = 1
            #depth_study=True
            #for intern_dim in [5,20,50,70]: 


            ### HERE ### uncomment 3 lines to generate the data/plots for depth dimension study
            #intern_dim = 10
            #indim_study=True
            #for depth in [1,2,3,4,5]:
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
                                               n_params=n_fine_tune_params,
                                               CV_sgd=sgd_crossval,
                                               CV_ridge=ridge_crossval)
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
    if indim_study:
        os.system('python benefits_study_indimVAR.py')
    elif depth_study:
        os.system('python benefits_study_depthVAR.py')
    else:
        os.system('python benefits_study.py')
            