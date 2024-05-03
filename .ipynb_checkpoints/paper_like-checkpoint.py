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

epochs = 10000
intern_dim=10
optimizer='GD'

nb_avg = 1

model_name = 'SLN' #'MLP' or 'SLN'

### Repeating experiment
errors = np.zeros((nb_avg, epochs))
for i in tqdm(range(nb_avg)):
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
    #w_ridge = ridge(data, observations, lambda_)

    input_Tensor = torch.from_numpy(data).to(torch.float32)
    output_Tensor = torch.from_numpy(observations).to(torch.float32)

    if model_name == 'MLP':
        model = MultiLayerPerceptron(input_dim=p,
                                     intern_dim=intern_dim,
                                     output_dim=1,
                                     depth=0,
                                     isBiased = False)
    elif model_name == 'SLN':
        model = SingleLayerNet(input_size=p,
                             output_size=1)

    errors[i,:] = train(model,
          input_Tensor,
          output_Tensor,
          lossFct = nn.MSELoss(),
          optimizer=optimizer,
          epochs=epochs,
          batch_size=None,
          return_vals=True,
          init_norm = None,
          lr = 0.005)

mean_error = np.mean(errors, axis=0)

fig1,ax1 = plt.subplots(1,1)
ax1.plot(range(epochs), mean_error, marker='*')
ax1.set_title(f'Average risk over {nb_avg} experiments')
ax1.set_xlabel(r'Iteration $t$')
ax1.set_ylabel(r'$R(\theta_t) - R^*$')
ax1.set_xscale('log')
ax1.set_yscale('linear')
plt.xlim(left=1)
plt.ylim(bottom=0, top = 1+mean_error.min())
plt.grid(color='black', which="both", linestyle='-', linewidth=0.2)
plt.savefig(f'figures/excess_risk_{model_name}_n{n}_t{epochs}_{optimizer}')