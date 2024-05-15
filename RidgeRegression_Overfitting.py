import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import os
os.system("utils.py")


np.random.seed(42)
LambdaArray = np.logspace(-6,1,200)
NDataArray = np.array([10,100,1000,10000,100000])

p_Array = [10]
n_Array = [100]
sigma2_Array = [2]
nb_avg_test = 20
nb_avg_train = 20

def plot_RidgeRegression(modelErrorArray,X,modelName,p,n,sigma2):
    fig1,ax1 = plt.subplots(1,1)
    for l in range(modelErrorArray.shape[0]):
        ax1.plot(X, modelErrorArray[l], label = "N = "+str(modelName[l]) , marker='*')
    ax1.set_title(f'Impact_N_data_RidgeReg_p_{p}_n_{n}_sigma_{sigma2}_avgTest_{nb_avg_test}_avgTrain_{nb_avg_train}')
    ax1.set_xlabel(r'$\lambda$')
    ax1.set_ylabel(r'$R^\lambda$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.grid(color='black', which="both", linestyle='-', linewidth=0.2)
    plt.legend()
    plt.savefig(f'figures/RidgeRegression/p_{p}_n_{n}_sigma_{sigma2}_avgTest_{nb_avg_test}_avgTrain_{nb_avg_train}')
    
def train_RidgeRegression(LambdaArray):
    
    data_ , obs_ = Generate_data(p = p, n = n, sigma2 = sigma2)
    
    Ridge_W = np.zeros((LambdaArray.shape[0],p))
    for i in range(LambdaArray.shape[0]):
        Ridge_W[i,:] = ridge(data_, obs_, LambdaArray[i])
        
    return Ridge_W

def tests_RidgeRegression(nb_avg, Ridge_W, NDataArray):
    
    Population_Risk_Avg = np.zeros((NDataArray.shape[0],Ridge_W.shape[0]))
    for i in range(NDataArray.shape[0]):
        
        population_risk = np.zeros((nb_avg,Ridge_W.shape[0]))
        for k in range(nb_avg):
            
            data_ , obs_ = Generate_data(p = p, n = NDataArray[i], sigma2 = sigma2)
            for c in range(Ridge_W.shape[0]):
                
                population_risk[k,c] = objective(data_, obs_, Ridge_W[c])
                
        Population_Risk_Avg[i,:] = population_risk.mean(axis=0)
    
    return Population_Risk_Avg

def test_RidgeRegression(p,n,sigma2):
    Population_Risk = np.zeros((nb_avg_train, NDataArray.shape[0],LambdaArray.shape[0]))
    for m in tqdm(range(nb_avg_train)):
        Ridge_W = train_RidgeRegression(LambdaArray)
        Population_Risk[m,:,:] = tests_RidgeRegression(nb_avg_test, Ridge_W, NDataArray)
    Population_Risk_Avg_ =Population_Risk.mean(axis=0)
    plot_RidgeRegression(Population_Risk_Avg_, LambdaArray, NDataArray,p,n,sigma2)
    
####Launch program

for p in p_Array:
    for n in n_Array:
        for sigma2 in sigma2_Array:
            test_RidgeRegression(p,n,sigma2)
    