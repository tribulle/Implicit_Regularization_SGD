import torch
import numpy as np
import matplotlib.pyplot as plt

from utils_Sam import *

np.random.seed(42)

### Parameters
p = 100
n = 100
sigma2 = 2

lambda_ = 1e-3

epochs = 500
intern_dim=10
optimizer='ASGD'

### Data generation
data = np.random.multivariate_normal(
    np.zeros(p),
    np.ones((p,p)),
    size=n) # shape (n,p)

w_true = np.ones(p)*1/np.sqrt(p)

observations = [np.random.normal(
    np.dot(w_true, x),
    sigma2)
    for x in data]
observations = np.array(observations) # shape (n,)

### Solving the problem
w_ridge = ridge(data, observations, lambda_)

input_Tensor = torch.from_numpy(data).to(torch.float32)
output_Tensor = torch.from_numpy(observations).to(torch.float32)

MLP = Multi_Layer_Perceptron(input_dim=p,
                             intern_dim=intern_dim,
                             output_dim=1,
                             depth=0)

errors_MLP = train(MLP,
      input_Tensor,
      output_Tensor,
      lossFct = nn.MSELoss(),
      optimizer=optimizer,
      epochs=epochs,
      return_vals=True)

fig1,ax1 = plt.subplots(1,1)
ax1.plot(range(epochs), errors_MLP, marker='*')
ax1.set_xlabel(r'Iteration $t$')
ax1.set_ylabel(r'$R(\theta_t) - R^*$')
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.savefig(f'figures/excess_risk_n{n}_t{epochs}_{optimizer}')