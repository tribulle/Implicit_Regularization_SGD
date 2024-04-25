import torch
import matplotlib.pyplot as plt
import numpy as np

from utils_Sam import *

### Global variables
np.random.seed(42)

n,d = 100,20
intern = 5
lambda_ = 1.

snr = 10 # signal to noise ratio

### Define the problem
input = np.random.rand(n,d)
noise = np.random.rand(n,d)
output = np.random.rand(n)*snr
output_noise = output + np.random.rand(n)

### Ridge solution
sol_ridge = ridge(input,output,lambda_)
obj = objective(input,output,sol_ridge)

sol_ridge_noise = ridge(input,output_noise,lambda_)
obj_noise = objective(input,output_noise,sol_ridge_noise)

### Define MLP
MLP = Multi_Layer_Perceptron(input_dim=d,
                             intern_dim=intern,
                             output_dim=n,
                             depth = 1)

input = (torch.from_numpy(input).to(torch.float32))
output = (torch.from_numpy(output).to(torch.float32))
output_noise = (torch.from_numpy(output_noise).to(torch.float32))

#train(MLP, input, output, init_norm = None, epochs = 40000, debug = 1)
#train(MLP, input, output, init_norm = 0, epochs = 40000, debug = 1)
#train(MLP, input, output, init_norm = 1, epochs = 40000, debug = 1)
train(MLP, input, output, init_norm = 2, epochs = 1000, debug = 1, savename='no_noise_Sam.pt')
sol_MLP = torch.eye(d)
for layer in MLP.children():
    sol_MLP = layer.weight@sol_MLP

compare(input, output, sol_ridge, sol_MLP)
#train(MLP, input, output, init_norm = 10, epochs = 40000, debug = 1)

train(MLP, input, output_noise, init_norm = 2, epochs = 1000, debug = 1, savename='noise_Sam.pt')
sol_MLP_noise = torch.eye(d)
for layer in MLP.children():
    sol_MLP_noise = layer.weight@sol_MLP

compare(input, output_noise, sol_ridge_noise, sol_MLP_noise)