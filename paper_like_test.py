import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

np.random.seed(42)

### Parameters
p = 100
n = 100
sigma2 = 2

lambda_ = 1e-3
t_max_ridge = 5

epochs = 7000
intern_dim=10
optimizer='GD'
learning_rate = 0.005

nb_avg = 1

model_name = 'SLN' #'MLP' or 'SLN'

### Repeating experiment
errors = np.zeros((nb_avg, epochs))
error_ridge = np.zeros((nb_avg, epochs))
for i in tqdm(range(nb_avg)):
    ### Data generation
    data = np.random.multivariate_normal(
        np.zeros(p),
        np.ones((p,p)),
        size=n) # shape (n,p)
    
    test_data = np.random.multivariate_normal(
        np.zeros(p),
        np.ones((p,p)),
        size=n)

    w_true = np.ones(p)*1/np.sqrt(p)

    observations = [np.random.normal(
        np.dot(w_true, x),
        sigma2)
        for x in data]
    observations = np.array(observations) # shape (n,)

    test_observations = [np.random.normal(
        np.dot(w_true, x),
        sigma2)
        for x in test_data]
    test_observations = np.array(test_observations)

    ### Solving the problem
    w_ridge = ridge_path(data, observations, nu_regression, np.linspace(1e-1,t_max_ridge, epochs)) # (epochs, p)
    for j in range(epochs):
        error_ridge[i,j] = objective(test_data,test_observations,w_ridge[j,:])

    input_Tensor = torch.from_numpy(data).to(torch.float32)
    output_Tensor = torch.from_numpy(observations).to(torch.float32)

    if model_name == 'MLP':
        model = MultiLayerPerceptron(input_dim=p,
                                     intern_dim=intern_dim,
                                     output_dim=1,
                                     depth=0,
                                     isBiased = False,
                                    )
    elif model_name == 'SLN':
        model = SingleLayerNet(input_size=p,
                             output_size=1,
                             init='zero')

    w_gd = train(model,
          input_Tensor,
          output_Tensor,
          lossFct = nn.MSELoss(),
          optimizer=optimizer,
          epochs=epochs,
          batch_size=None,
          return_ws=True,
          return_vals=False,
          init_norm = None,
          lr = learning_rate)
    
    errors[i,:] = [objective(test_data,test_observations,w_gd[j,:]) for j in range(epochs)]

mean_error = np.mean(errors, axis=0)
mean_error_ridge = np.mean(error_ridge, axis=0)

fig1,ax1 = plt.subplots(1,1)
ax1.plot(range(epochs), mean_error-mean_error_ridge, marker='*', markersize=5)
ax1.set_title(f'Average risk over {nb_avg} experiments')
ax1.set_xlabel(r'Iteration $t$')
ax1.set_ylabel(r'$R(\theta_t) - R^*$')
ax1.set_xscale('log')
ax1.set_yscale('linear')
plt.xlim(left=1)
plt.ylim(bottom=0, top = 1+mean_error.min())
plt.grid(color='black', which="both", linestyle='-', linewidth=0.2)
plt.savefig(f'figures/excess_risk_{model_name}_n{n}_t{epochs}_{optimizer}_test')